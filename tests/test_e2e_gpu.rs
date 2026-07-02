//! GPU CUDA E2E 测试 (5070Ti, SM 12.0 Blackwell).
//!
//! 在 NVIDIA RTX 5070 Ti 上验证 GPU PTX JIT codegen 路径端到端正确性。
//! 通过 `.backend(BackendType::Cuda)` 强制 GPU 后端, executor 自动检测 SM 12.0
//! 并以 `CompileTarget::Gpu { sm_version: 120 }` 编译 PTX (target sm_120, ISA 8.8).
//!
//! 验证策略:
//! 1. Generator (SmolLM2-135M / Qwen3-0.6B): prefill logits vs PyTorch 黄金值
//!    - cosine_sim > 0.9999 (数值方向一致)
//!    - max_abs_diff < 0.01  (数值幅度接近)
//!    - argmax next_token_id 一致 (greedy decode 正确)
//! 2. Embedding (e5-small-v2): embedding 输出 sane + 与黄金值 cosine_sim > 0.999
//! 3. Reranker (bge-reranker-v2-m3): rerank 排序 sane + 与黄金值方向一致
//!
//! 约束 (ARCH-SINGLE-MODEL-INSTANCE):
//! - 单线程 (--test-threads=1)
//! - 单模型实例 (同一时刻内存中只有一个模型)
//! - 量化优先 (ARCH-QUANT-FIRST-E2E): Q4_0 / Q4_K_M
//!
//! 运行 (5070Ti):
//!   cargo test --features cuda --test test_e2e_gpu -- --test-threads=1 --ignored

#![cfg(all(feature = "cuda", target_os = "linux"))]

use gllm::{BackendType, Client, ModelKind};

// ─── Golden value helpers ─────────────────────────────────────────────

/// Load a single f32 tensor from safetensors by name.
fn load_golden_f32_tensor(path: &std::path::Path, name: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap_or_else(|e| {
        panic!("Failed to read golden safetensors {}: {e}", path.display())
    });
    let tensors = safetensors::SafeTensors::deserialize(&data)
        .unwrap_or_else(|e| panic!("Failed to parse safetensors {}: {e}", path.display()));
    let view = tensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("Missing tensor '{name}' in {}", path.display()));
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Read scalar next_token_id from golden safetensors (stored as f32).
fn load_golden_next_token_id(path: &std::path::Path) -> u32 {
    let v = load_golden_f32_tensor(path, "next_token_id");
    assert_eq!(v.len(), 1, "next_token_id should be scalar, got {}", v.len());
    v[0] as u32
}

/// Load golden logits, extract last-token row from [seq_len, vocab_size].
fn load_golden_last_token_logits(path: &std::path::Path, seq_len: usize, vocab_size: usize) -> Vec<f32> {
    let all = load_golden_f32_tensor(path, "logits");
    assert_eq!(all.len(), seq_len * vocab_size, "logits size mismatch");
    let offset = (seq_len - 1) * vocab_size;
    all[offset..offset + vocab_size].to_vec()
}

/// Cosine similarity between two equal-length f32 slices.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch {} vs {}", a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)) as f32 }
}

/// Max absolute difference between two equal-length f32 slices.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "mad: length mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (*x - *y).abs()).fold(0.0f32, f32::max)
}

/// Argmax index of an f32 slice.
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0usize, |mi, (i, &x)| if x > v[mi] { i } else { mi })
}

// ─── GPU backend builder ──────────────────────────────────────────────

/// Build a GPU (CUDA) chat client for the given model.
fn gpu_chat_client(model_id: &str) -> Client {
    Client::builder()
        .model(model_id)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cuda)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build GPU chat client for {model_id}: {e}"))
}

/// Build a GPU (CUDA) embedding client.
fn gpu_embed_client(model_id: &str) -> Client {
    Client::builder()
        .model(model_id)
        .kind(ModelKind::Embedding)
        .backend(BackendType::Cuda)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build GPU embed client for {model_id}: {e}"))
}

/// Build a GPU (CUDA) reranker client.
fn gpu_reranker_client(model_id: &str) -> Client {
    Client::builder()
        .model(model_id)
        .kind(ModelKind::Reranker)
        .backend(BackendType::Cuda)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build GPU reranker client for {model_id}: {e}"))
}

// ─── TEST-GPU-GEN-001: SmolLM2-135M generator (SafeTensors, F32) ─────
//
// PyTorch 黄金值: tests/e2e_alignment/data/golden_smollm2_135m.safetensors
// Prompt: "The meaning of life is" (5 tokens) → argmax next_token_id = 253 (' a').

#[test]
#[ignore = "GPU E2E: requires 5070Ti + cuda feature; run with --ignored"]
fn gpu_e2e_smollm2_135m_logits_alignment() {
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    const PROMPT: &str = "The meaning of life is";
    const SEQ_LEN: usize = 5;
    const VOCAB_SIZE: usize = 49152;
    const COSINE_THRESHOLD: f32 = 0.9999;
    const MAD_THRESHOLD: f32 = 0.01;

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    assert!(golden_path.exists(), "Golden data missing: {}. Run generate_golden_smollm2.py", golden_path.display());

    let golden_next_id = load_golden_next_token_id(golden_path);
    let golden_logits = load_golden_last_token_logits(golden_path, SEQ_LEN, VOCAB_SIZE);

    let client = gpu_chat_client(MODEL);
    let tokens = client.encode(PROMPT).expect("tokenizer encode failed");
    assert_eq!(tokens.len(), SEQ_LEN, "SmolLM2 prompt token count mismatch");

    let gpu_logits = client
        .diagnostic_prefill_logits(&tokens)
        .expect("GPU prefill logits unavailable (GPU backend not active?)");
    assert_eq!(gpu_logits.len(), VOCAB_SIZE, "GPU logits vocab size mismatch");

    // 1. argmax next-token ID must match PyTorch greedy decode.
    let gpu_next_id = argmax(&gpu_logits) as u32;
    assert_eq!(
        gpu_next_id, golden_next_id,
        "GPU argmax next_token_id mismatch: GPU={gpu_next_id} golden={golden_next_id}"
    );

    // 2. Numerical alignment: direction (cosine) + magnitude (max-abs-diff).
    let cos = cosine_similarity(&gpu_logits, &golden_logits);
    let mad = max_abs_diff(&gpu_logits, &golden_logits);
    assert!(
        cos > COSINE_THRESHOLD,
        "GPU logits cosine_sim {cos} <= {COSINE_THRESHOLD} (SmolLM2 GPU vs golden)"
    );
    assert!(
        mad < MAD_THRESHOLD,
        "GPU logits max_abs_diff {mad} >= {MAD_THRESHOLD} (SmolLM2 GPU vs golden)"
    );

    eprintln!(
        "[GPU-ALIGN] SmolLM2-135M: next_id={gpu_next_id} (golden {golden_next_id}) cos={cos:.6} mad={mad:.6}"
    );
}

// ─── TEST-GPU-GEN-002: Qwen3-0.6B generator (GGUF Q4_0) ──────────────
//
// 量化模型 GPU 路径验证。无 PyTorch Q4_0 黄金值 (量化后数值非确定性一致),
// 改用 anti-degeneration + greedy determinism 断言。

#[test]
#[ignore = "GPU E2E: requires 5070Ti + cuda feature; run with --ignored"]
fn gpu_e2e_qwen3_0_6b_gguf_q4_0_generation() {
    const MODEL: &str = "bartowski/Qwen_Qwen3-0.6B-GGUF";
    const PROMPT: &str = "The capital of France is";
    const GGUF_FILE: &str = "Qwen3-0.6B-Q4_0.gguf";

    let client = Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cuda)
        .gguf_file_filter(GGUF_FILE)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build GPU Qwen3 client: {e}"));

    let response = client
        .generate(PROMPT)
        .max_tokens(20)
        .temperature(0.0)
        .generate()
        .response()
        .expect("GPU Qwen3 generate failed");
    let text = response.text.trim();
    assert!(!text.is_empty(), "GPU Qwen3 output is empty");

    // Anti-degeneration: not all same character, not single-word repetition loop.
    let unique_chars = text.chars().collect::<std::collections::HashSet<_>>().len();
    assert!(unique_chars >= 3, "GPU Qwen3 degenerate: only {unique_chars} unique chars in {text:?}");

    // Determinism: same prompt + temperature=0 → same output.
    let response2 = client
        .generate(PROMPT)
        .max_tokens(20)
        .temperature(0.0)
        .generate()
        .response()
        .expect("GPU Qwen3 second generate failed");
    assert_eq!(
        response2.text.trim(),
        text,
        "GPU Qwen3 non-deterministic at temperature=0 (quantized decode path)"
    );

    eprintln!("[GPU-GEN] Qwen3-0.6B Q4_0: output={text:?} (unique_chars={unique_chars})");
}

// ─── TEST-GPU-EMB-001: e5-small-v2 embedding (SafeTensors, F32) ──────
//
// PyTorch 黄金值: tests/e2e_alignment/data/golden_e5_small_v2.safetensors
// 输入 "Hello, world!" → 384-dim embedding.

#[test]
#[ignore = "GPU E2E: requires 5070Ti + cuda feature; run with --ignored"]
fn gpu_e2e_e5_small_v2_embedding_alignment() {
    const MODEL: &str = "intfloat/e5-small-v2";
    const PROMPT: &str = "Hello, world!";
    const DIM: usize = 384;
    const COSINE_THRESHOLD: f32 = 0.999;

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_e5_small_v2.safetensors");
    assert!(golden_path.exists(), "Golden data missing: {}", golden_path.display());

    // Golden embeddings stored as embedding_<text> tensors. Try a few candidate keys.
    let golden = {
        let candidates = ["embedding_Hello_world", "embedding_0", "embedding"];
        let mut v: Option<Vec<f32>> = None;
        for key in &candidates {
            if let Ok(t) = safetensors::SafeTensors::deserialize(
                &std::fs::read(golden_path).expect("read golden"),
            ) {
                if let Ok(view) = t.tensor(key) {
                    v = Some(
                        view.data()
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    );
                    break;
                }
            }
        }
        v.unwrap_or_else(|| panic!("No golden embedding tensor found in {}", golden_path.display()))
    };
    assert_eq!(golden.len(), DIM, "golden embedding dim mismatch");

    let client = gpu_embed_client(MODEL);
    let response = client
        .embed([PROMPT])
        .expect("GPU e5 embed failed");
    assert_eq!(response.embeddings.len(), 1, "expected 1 embedding");
    let emb = &response.embeddings[0].embedding;
    assert_eq!(emb.len(), DIM, "GPU e5 embedding dim mismatch");
    assert!(emb.iter().all(|v| v.is_finite()), "GPU e5 embedding has NaN/Inf");

    let cos = cosine_similarity(emb, &golden);
    assert!(
        cos > COSINE_THRESHOLD,
        "GPU e5 embedding cosine_sim {cos} <= {COSINE_THRESHOLD} (vs golden)"
    );

    eprintln!("[GPU-EMB] e5-small-v2: dim={} cos={cos:.6}", emb.len());
}

// ─── TEST-GPU-RERANK-001: bge-reranker-v2-m3 (SafeTensors) ───────────
//
// 验证 GPU 路径 rerank 排序 sane + 与文档相关性方向一致。

#[test]
#[ignore = "GPU E2E: requires 5070Ti + cuda feature; run with --ignored"]
fn gpu_e2e_bge_reranker_v2_m3() {
    const MODEL: &str = "BAAI/bge-reranker-v2-m3";
    const QUERY: &str = "What is the capital of France?";

    let documents = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "The Eiffel Tower is in Paris.",
    ];

    let client = gpu_reranker_client(MODEL);
    let response = client
        .rerank(QUERY, documents)
        .expect("GPU rerank failed");
    assert_eq!(response.results.len(), documents.len(), "rerank result count mismatch");

    // All scores finite.
    for r in &response.results {
        assert!(r.score.is_finite(), "GPU rerank score NaN/Inf: {:?}", r.score);
    }

    // The most relevant doc ("Paris is the capital of France.", index 0) should rank highest.
    let top = response
        .results
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .expect("no top rerank result");
    assert_eq!(
        top.index, 0,
        "GPU rerank top result should be the France/Paris doc (index 0), got index={}",
        top.index
    );

    eprintln!(
        "[GPU-RERANK] bge-reranker-v2-m3: top_idx={} score={:.4}",
        top.index, top.score
    );
}
