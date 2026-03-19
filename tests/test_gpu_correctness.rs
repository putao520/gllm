//! GPU vs CPU 推理一致性测试
//!
//! 验证 CUDA 后端与 CPU 后端在相同模型上产生数值一致的输出。
//!
//! 运行方式:
//!   cargo test --test test_gpu_correctness --features cuda -- --ignored --test-threads=1
//!
//! 所有测试默认 ignore，需要 CUDA 环境才能运行。

use std::sync::Arc;

use gllm::backend::{BackendContext, BackendType, DetectedBackend};
use gllm::compat::{CpuBackend, CudaBackend};
use gllm::loader::{Loader, LoaderConfig};
use gllm::manifest::{ModelKind, ModelManifest, EMPTY_FILE_MAP};
use gllm::model_config::ModelConfig;
use gllm::backend::fallback::{FallbackEmbedder, FallbackGenerator, FallbackReranker};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "embedding dimension mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Build a BackendContext forced to CPU for the given model path and kind.
fn build_cpu_context(model_path: &str, kind: ModelKind) -> BackendContext {
    let config = LoaderConfig::from_env();
    let mut loader = Loader::from_source_with_config(model_path.to_string(), config)
        .expect("loader init");
    loader = loader.load().expect("loader load");

    let dummy = ModelManifest {
        model_id: std::borrow::Cow::Owned(model_path.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch: gllm::ModelArchitecture::Llama4,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: std::collections::HashMap::new(),
    };
    let derived = ModelConfig::from_loader(&dummy, &mut loader).expect("model config");
    let arch = loader.detect_architecture();
    let moe_config = derived.build_moe_config(arch);

    let manifest = Arc::new(ModelManifest {
        model_id: std::borrow::Cow::Owned(model_path.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config,
        tensor_map: std::collections::HashMap::new(),
    });

    let config_path = loader.config_path().map(|p| p.to_path_buf());
    let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
    let weight_paths = loader.weight_paths().to_vec();

    BackendContext::new(
        model_path.to_string(),
        manifest,
        DetectedBackend::Cpu(Box::new(CpuBackend::<f32>::new())),
        weight_paths,
        config_path,
        tokenizer_path,
    )
    .expect("cpu context")
}

/// Build a BackendContext forced to CUDA device 0 for the given model path and kind.
fn build_cuda_context(model_path: &str, kind: ModelKind) -> BackendContext {
    let config = LoaderConfig::from_env();
    let mut loader = Loader::from_source_with_config(model_path.to_string(), config)
        .expect("loader init");
    loader = loader.load().expect("loader load");

    let dummy = ModelManifest {
        model_id: std::borrow::Cow::Owned(model_path.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch: gllm::ModelArchitecture::Llama4,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: std::collections::HashMap::new(),
    };
    let derived = ModelConfig::from_loader(&dummy, &mut loader).expect("model config");
    let arch = loader.detect_architecture();
    let moe_config = derived.build_moe_config(arch);

    let manifest = Arc::new(ModelManifest {
        model_id: std::borrow::Cow::Owned(model_path.to_string()),
        file_map: EMPTY_FILE_MAP,
        arch,
        kind,
        rope_base_override: None,
        max_context_override: None,
        moe_config,
        tensor_map: std::collections::HashMap::new(),
    });

    let config_path = loader.config_path().map(|p| p.to_path_buf());
    let tokenizer_path = loader.tokenizer_path().map(|p| p.to_path_buf());
    let weight_paths = loader.weight_paths().to_vec();

    let cuda_backend = CudaBackend::<f32>::new(0).expect("CUDA device 0 not available");

    BackendContext::new(
        model_path.to_string(),
        manifest,
        DetectedBackend::Cuda(Box::new(cuda_backend)),
        weight_paths,
        config_path,
        tokenizer_path,
    )
    .expect("cuda context")
}

// ---------------------------------------------------------------------------
// TEST-GPU-001: Embedding CPU vs CUDA 一致性
// ---------------------------------------------------------------------------

/// TEST-GPU-001: Embedding CPU vs CUDA cosine similarity > 0.99
///
/// 模型: test_models/safetensors (MiniLM-L6)
/// 验证: 相同输入在 CPU 和 GPU 上产生的 embedding 向量余弦相似度 > 0.99
#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "Requires CUDA backend")]
#[ignore]
fn test_gpu_001_embedding_cpu_vs_cuda() {
    const MODEL: &str = "test_models/safetensors";
    let inputs = vec![
        "hello world".to_string(),
        "rust programming".to_string(),
    ];

    // CPU inference
    let cpu_ctx = build_cpu_context(MODEL, ModelKind::Embedding);
    let cpu_embeddings: Vec<Vec<f32>> = {
        let mut embedder = FallbackEmbedder::new(&cpu_ctx);
        embedder
            .embed_batch(&inputs)
            .expect("cpu embed_batch")
            .value
    };

    // GPU inference
    let gpu_ctx = build_cuda_context(MODEL, ModelKind::Embedding);
    let gpu_embeddings: Vec<Vec<f32>> = {
        let mut embedder = FallbackEmbedder::new(&gpu_ctx);
        embedder
            .embed_batch(&inputs)
            .expect("gpu embed_batch")
            .value
    };

    assert_eq!(cpu_embeddings.len(), gpu_embeddings.len());

    for (i, (cpu_emb, gpu_emb)) in cpu_embeddings.iter().zip(gpu_embeddings.iter()).enumerate() {
        let sim = cosine_similarity(cpu_emb, gpu_emb);
        assert!(
            sim > 0.99,
            "input[{i}]: CPU vs GPU cosine similarity {sim:.4} < 0.99"
        );
    }
}

// ---------------------------------------------------------------------------
// TEST-GPU-002: Reranker CPU vs CUDA 一致性
// ---------------------------------------------------------------------------

/// TEST-GPU-002: Reranker CPU vs CUDA score difference < 0.01
///
/// 模型: test_models/reranker/safetensors
/// 验证: 相同 query/doc 对在 CPU 和 GPU 上的 rerank 分数差值 < 0.01
#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "Requires CUDA backend")]
#[ignore]
fn test_gpu_002_reranker_cpu_vs_cuda() {
    const MODEL: &str = "test_models/reranker/safetensors";
    const QUERY: &str = "efficient storage";
    let docs = vec![
        "Columnar databases compress well.".to_string(),
        "Rust has zero-cost abstractions.".to_string(),
    ];

    // CPU inference
    let cpu_ctx = build_cpu_context(MODEL, ModelKind::Reranker);
    let cpu_scores: Vec<f32> = {
        let mut reranker = FallbackReranker::new(&cpu_ctx);
        reranker
            .rerank_batch(QUERY, &docs)
            .expect("cpu rerank_batch")
            .value
    };

    // GPU inference
    let gpu_ctx = build_cuda_context(MODEL, ModelKind::Reranker);
    let gpu_scores: Vec<f32> = {
        let mut reranker = FallbackReranker::new(&gpu_ctx);
        reranker
            .rerank_batch(QUERY, &docs)
            .expect("gpu rerank_batch")
            .value
    };

    assert_eq!(cpu_scores.len(), gpu_scores.len());

    for (i, (cpu_score, gpu_score)) in cpu_scores.iter().zip(gpu_scores.iter()).enumerate() {
        let diff = (cpu_score - gpu_score).abs();
        assert!(
            diff < 0.01,
            "doc[{i}]: CPU score {cpu_score:.4} vs GPU score {gpu_score:.4}, diff {diff:.4} >= 0.01"
        );
    }
}

// ---------------------------------------------------------------------------
// TEST-GPU-003: Generator CPU vs CUDA 一致性
// ---------------------------------------------------------------------------

/// TEST-GPU-003: Generator CPU vs CUDA — 前 5 个 token 完全相同
///
/// 模型: test_models/smollm2-135m/safetensors
/// 验证: temperature=0.0 下 CPU 和 GPU 生成的文本完全一致
#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "Requires CUDA backend")]
#[ignore]
fn test_gpu_003_generator_cpu_vs_cuda() {
    const MODEL: &str = "test_models/smollm2-135m/safetensors";
    const PROMPT: &str = "The capital of France is";
    const MAX_TOKENS: usize = 5;

    // CPU inference
    let cpu_ctx = build_cpu_context(MODEL, ModelKind::Chat);
    let cpu_text: String = {
        let mut generator = FallbackGenerator::new(&cpu_ctx);
        generator
            .generate(PROMPT, MAX_TOKENS, 0.0, 0, 1.0)
            .expect("cpu generate")
            .value
    };

    // GPU inference
    let gpu_ctx = build_cuda_context(MODEL, ModelKind::Chat);
    let gpu_text: String = {
        let mut generator = FallbackGenerator::new(&gpu_ctx);
        generator
            .generate(PROMPT, MAX_TOKENS, 0.0, 0, 1.0)
            .expect("gpu generate")
            .value
    };

    assert_eq!(
        cpu_text, gpu_text,
        "CPU output {cpu_text:?} != GPU output {gpu_text:?}"
    );
}

// ---------------------------------------------------------------------------
// TEST-GPU-004: CUDA 设备检测
// ---------------------------------------------------------------------------

/// TEST-GPU-004: CUDA 设备检测
///
/// 验证 detect_backend() 在有 CUDA 时返回 CUDA backend，且 GPU 内存 > 0
#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "Requires CUDA backend")]
#[ignore]
fn test_gpu_004_cuda_device_detection() {
    let backend = gllm::backend::detect_backend().expect("detect_backend failed");
    assert_eq!(
        backend.backend_type(),
        BackendType::Cuda,
        "Expected CUDA backend, got {:?}",
        backend.backend_type()
    );

    // Verify GPU memory > 0 via CudaBackend device info
    let cuda = CudaBackend::<f32>::new(0).expect("CUDA device 0 not available");
    assert!(
        cuda.device_info().total_memory > 0,
        "GPU total_memory should be > 0, got {}",
        cuda.device_info().total_memory
    );
}
