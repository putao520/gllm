//! CPU E2E baseline 测试 (ARCH-UNIFIED-EXEC 块1).
//!
//! 不带 `cuda` feature 编译时, `CudaBackend::new` 返 None (stub),
//! `detect_backend` 强制 `CpuBackend`. 此测试在该配置下跑出真实 CPU argmax,
//! 建立 gllm CPU 路径 vs PyTorch 黄金值的 baseline 断言。
//!
//! 架构师 (sessionId 5d98f4f4) 判决: 全 tests/ 此前无 CPU 测试断言
//! gllm 输出==253, "CPU pass" 是误读 (argmax=253 是 golden-vs-golden)。
//! 块1 建 CPU baseline: 真断言 gllm CPU argmax == golden 253。
//!
//! 运行 (不带 cuda feature):
//!   cargo +nightly test --test test_e2e_cpu -- --nocapture

#![cfg(target_os = "linux")]

use gllm::{BackendType, Client, ModelKind};

use std::io::Write as _;

// ─── Golden value helpers (mirrored from test_e2e_gpu.rs) ────────────

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

// ─── TEST-CPU-GEN-001: SmolLM2-135M generator (SafeTensors, F32) ────
//
// PyTorch 黄金值: tests/e2e_alignment/data/golden_smollm2_135m.safetensors
// Prompt: "The meaning of life is" (5 tokens) → argmax next_token_id = 253 (' a').

#[test]
fn cpu_e2e_smollm2_135m_logits_alignment() {
    eprintln!("[CPU-BASELINE] cpu_e2e_smollm2 entered, stderr flushed");
    std::io::stderr().flush().ok();
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
    eprintln!("[CPU-BASELINE] golden_next_id={golden_next_id}");

    // 强制 CPU 后端. 不带 cuda feature 时, CudaBackend::new 返 None,
    // detect_backend 必然走 CpuBackend (即便 .backend() setter 是死的也无所谓).
    let client = Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cpu)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build CPU chat client for {MODEL}: {e}"));

    let tokens = client.encode(PROMPT).expect("tokenizer encode failed");
    assert_eq!(tokens.len(), SEQ_LEN, "SmolLM2 prompt token count mismatch");
    eprintln!("[CPU-BASELINE] tokens={:?}", tokens);

    let cpu_logits = client
        .diagnostic_prefill_logits(&tokens)
        .expect("CPU prefill logits unavailable (CPU backend not active?)");
    assert_eq!(cpu_logits.len(), VOCAB_SIZE, "CPU logits vocab size mismatch");

    // DIAG: logits 实际值画像
    {
        let sum: f64 = cpu_logits.iter().map(|x| *x as f64).sum();
        let max_v = cpu_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = cpu_logits.iter().cloned().fold(f32::INFINITY, f32::min);
        let nan_cnt = cpu_logits.iter().filter(|x| x.is_nan()).count();
        let zero_cnt = cpu_logits.iter().filter(|x| **x == 0.0).count();
        let inf_cnt = cpu_logits.iter().filter(|x| x.is_infinite()).count();
        eprintln!(
            "[DIAG-LOGITS] len={} sum={sum:.4} min={min_v:.4} max={max_v:.4} nan={nan_cnt} zero={zero_cnt} inf={inf_cnt}",
            cpu_logits.len()
        );
        eprintln!("[DIAG-LOGITS] first5={:?}", &cpu_logits[..5.min(cpu_logits.len())]);
        let mut idx_val: Vec<(usize, f32)> = cpu_logits.iter().cloned().enumerate().collect();
        idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[DIAG-LOGITS] top5={:?}",
            &idx_val[..5.min(idx_val.len())]
        );
        let mut g_idx_val: Vec<(usize, f32)> = golden_logits.iter().cloned().enumerate().collect();
        g_idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!(
            "[DIAG-GOLDEN] top5={:?}",
            &g_idx_val[..5.min(g_idx_val.len())]
        );
        std::io::stderr().flush().ok();
    }

    // 1. argmax next-token ID must match PyTorch greedy decode.
    let cpu_next_id = argmax(&cpu_logits) as u32;
    eprintln!(
        "[CPU-BASELINE] cpu_next_id={cpu_next_id} golden_next_id={golden_next_id}"
    );

    // 2. Numerical alignment: direction (cosine) + magnitude (max-abs-diff).
    let cos = cosine_similarity(&cpu_logits, &golden_logits);
    let mad = max_abs_diff(&cpu_logits, &golden_logits);
    eprintln!(
        "[CPU-BASELINE] cosine={cos:.6} (threshold > {COSINE_THRESHOLD}) mad={mad:.6} (threshold < {MAD_THRESHOLD})"
    );
    std::io::stderr().flush().ok();

    assert_eq!(
        cpu_next_id, golden_next_id,
        "CPU argmax next_token_id mismatch: CPU={cpu_next_id} golden={golden_next_id}"
    );
    assert!(
        cos > COSINE_THRESHOLD,
        "CPU logits cosine_sim {cos} <= {COSINE_THRESHOLD} (SmolLM2 CPU vs golden)"
    );
    assert!(
        mad < MAD_THRESHOLD,
        "CPU logits max_abs_diff {mad} >= {MAD_THRESHOLD} (SmolLM2 CPU vs golden)"
    );

    eprintln!(
        "[CPU-ALIGN] SmolLM2-135M: next_id={cpu_next_id} (golden {golden_next_id}) cos={cos:.6} mad={mad:.6}"
    );
}
