//! Diagnostic test: compare gllm weight blob data with golden reference.
//!
//! Purpose: Identify WHERE the numerical deviation starts by comparing
//! embedding data, per-layer weight data, and (eventually) per-layer
//! intermediate activations with HuggingFace reference values.

use gllm::Client;

/// Golden reference values
mod golden {
    pub const INPUT_IDS: &[u32] = &[504, 2455, 282, 1029, 314];
    pub const HIDDEN_SIZE: usize = 576;
    pub const VOCAB_SIZE: usize = 49152;
}

/// Load golden hidden states from safetensors.
fn load_golden_hidden(layer: usize) -> Option<Vec<f32>> {
    let path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !path.exists() {
        return None;
    }
    let data = std::fs::read(path).ok()?;
    let tensors = safetensors::SafeTensors::deserialize(&data).ok()?;
    let name = if layer == 0 {
        "hidden_layer_0".to_string()
    } else {
        format!("hidden_layer_{}", layer)
    };
    let view = tensors.tensor(&name).ok()?;
    let bytes = view.data();
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Load golden logits (last token position) from safetensors.
fn load_golden_logits() -> Option<Vec<f32>> {
    let path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !path.exists() {
        return None;
    }
    let data = std::fs::read(path).ok()?;
    let tensors = safetensors::SafeTensors::deserialize(&data).ok()?;
    let view = tensors.tensor("logits").ok()?;
    let bytes = view.data();
    let shape = view.shape();
    // shape = [seq_len, vocab_size] = [5, 49152]
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 5);
    assert_eq!(shape[1], golden::VOCAB_SIZE);
    let all_logits: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Last token's logits
    let offset = 4 * golden::VOCAB_SIZE;
    Some(all_logits[offset..offset + golden::VOCAB_SIZE].to_vec())
}

/// Cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..len {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Max absolute difference between two vectors.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    (0..len).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)
}

/// Find argmax of a f32 slice.
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// DIAG-001: Verify embedding data in weight blob matches golden reference.
#[test]
fn diagnostic_embedding_data_matches_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let embed_row = client.diagnostic_weight_row("embed", golden::INPUT_IDS[0] as usize, golden::HIDDEN_SIZE)
        .expect("Failed to read embedding row from weight blob");

    assert_eq!(embed_row.len(), golden::HIDDEN_SIZE, "Embedding row length mismatch");

    let golden_hidden = load_golden_hidden(0)
        .expect("Golden data not found. Run: python3 tests/e2e_alignment/generate_golden_smollm2.py");

    assert_eq!(golden_hidden.len(), golden::HIDDEN_SIZE * 5, "Golden hidden shape mismatch");
    let golden_row = &golden_hidden[0..golden::HIDDEN_SIZE];

    let sim = cosine_similarity(&embed_row, golden_row);
    let mad = max_abs_diff(&embed_row, golden_row);

    eprintln!("[DIAG-001] Embedding row 504: cosine_sim={:.6} max_abs_diff={:.6}", sim, mad);
    eprintln!("[DIAG-001] First 8 gllm: {:?}", &embed_row[..8]);
    eprintln!("[DIAG-001] First 8 gold: {:?}", &golden_row[..8]);

    assert!(sim > 0.9999, "Embedding cosine similarity too low: {}", sim);
    assert!(mad < 0.01, "Embedding max abs diff too high: {}", mad);
}

/// DIAG-002: Verify lm_head (tied) data matches embedding data.
#[test]
fn diagnostic_tied_lm_head_matches_embedding() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let embed_row = client.diagnostic_weight_row("embed", 0, golden::HIDDEN_SIZE)
        .expect("Failed to read embed row 0");
    let lm_head_row = client.diagnostic_weight_row("lm_head", 0, golden::HIDDEN_SIZE)
        .expect("Failed to read lm_head row 0");

    let sim = cosine_similarity(&embed_row, &lm_head_row);
    let mad = max_abs_diff(&embed_row, &lm_head_row);

    eprintln!("[DIAG-002] Tied embed vs lm_head row 0: cosine_sim={:.6} max_abs_diff={:.6}", sim, mad);

    assert!(mad < 1e-6, "Tied lm_head should match embedding exactly, max_abs_diff={}", mad);
}

/// DIAG-003: Dump weight offsets for manual inspection.
#[test]
fn diagnostic_dump_weight_offsets() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let offsets = client.diagnostic_weight_offsets()
        .expect("Failed to get weight offsets");

    eprintln!("[DIAG-003] Weight offsets ({} tensors):", offsets.len());


    for (name, off, _dtype) in &offsets {
        eprintln!("  {} @ byte {}", name, off);
    }

    let embed_off = offsets.iter().find(|(n, _, _)| n == "embed").map(|(_, o, _)| *o);
    let lm_off = offsets.iter().find(|(n, _, _)| n == "lm_head").map(|(_, o, _)| *o);
    eprintln!("[DIAG-003] embed_offset={:?} lm_head_offset={:?}", embed_off, lm_off);

    assert!(embed_off.is_some(), "embed tensor not found in weight layout");
    assert!(lm_off.is_some(), "lm_head tensor not found in weight layout");
    assert_ne!(embed_off, lm_off, "embed and lm_head should have separate offsets in weight blob");
}

/// DIAG-004: Verify first-layer weight data (RmsNorm + Q/K/V projections).
#[test]
fn diagnostic_layer0_weights_nonzero() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let norm = client.diagnostic_weight_row("L0.input_norm", 0, golden::HIDDEN_SIZE)
        .expect("Failed to read L0.input_norm");

    let nonzero = norm.iter().filter(|&&v| v != 0.0).count();
    let l1_norm: f32 = norm.iter().map(|v| v.abs()).sum();
    eprintln!("[DIAG-004] L0.input_norm: nonzero={}/{} l1_norm={:.4} first4={:?}",
        nonzero, norm.len(), l1_norm, &norm[..4]);

    assert!(nonzero > 0, "L0.input_norm is all zeros — weight not loaded");
    assert!(l1_norm > 0.0, "L0.input_norm L1 norm is zero");

    let q_row = client.diagnostic_weight_row("L0.q_proj", 0, golden::HIDDEN_SIZE)
        .expect("Failed to read L0.q_proj row 0");
    let q_nonzero = q_row.iter().filter(|&&v| v != 0.0).count();
    eprintln!("[DIAG-004] L0.q_proj row 0: nonzero={}/{}", q_nonzero, q_row.len());
    assert!(q_nonzero > 0, "L0.q_proj row 0 is all zeros — weight not loaded");
}

/// DIAG-005: Compare gllm prefill logits with golden logits.
///
/// This is the key test: if weights are correct (DIAG-001~004 pass) but
/// logits don't match, the deviation is in the JIT computation of the
/// 30 transformer layers (RmsNorm, RoPE, Attention, FFN, residual).
#[test]
fn diagnostic_prefill_logits_vs_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let gllm_logits = client.diagnostic_prefill_logits(golden::INPUT_IDS)
        .expect("Failed to get prefill logits from mega-kernel");

    assert_eq!(gllm_logits.len(), golden::VOCAB_SIZE, "Logits length mismatch");

    let golden_logits = load_golden_logits()
        .expect("Golden data not found. Run: python3 tests/e2e_alignment/generate_golden_smollm2.py");

    let gllm_argmax = argmax(&gllm_logits);
    let golden_argmax = argmax(&golden_logits);

    eprintln!("[DIAG-005] gllm  argmax={} value={:.4}", gllm_argmax, gllm_logits[gllm_argmax]);
    eprintln!("[DIAG-005] gold  argmax={} value={:.4}", golden_argmax, golden_logits[golden_argmax]);

    let sim = cosine_similarity(&gllm_logits, &golden_logits);
    let mad = max_abs_diff(&gllm_logits, &golden_logits);
    eprintln!("[DIAG-005] logits cosine_sim={:.6} max_abs_diff={:.6}", sim, mad);

    // Print top-5 from each
    let top5_gllm = {
        let mut idx: Vec<usize> = (0..gllm_logits.len()).collect();
        idx.sort_by(|&a, &b| gllm_logits[b].partial_cmp(&gllm_logits[a]).unwrap());
        idx[..5].to_vec()
    };
    let top5_gold = {
        let mut idx: Vec<usize> = (0..golden_logits.len()).collect();
        idx.sort_by(|&a, &b| golden_logits[b].partial_cmp(&golden_logits[a]).unwrap());
        idx[..5].to_vec()
    };
    eprintln!("[DIAG-005] gllm  top5: {:?}", top5_gllm.iter().map(|&i| (i, gllm_logits[i])).collect::<Vec<_>>());
    eprintln!("[DIAG-005] gold top5: {:?}", top5_gold.iter().map(|&i| (i, golden_logits[i])).collect::<Vec<_>>());

    // Print a few raw logit values for manual comparison
    eprintln!("[DIAG-005] gllm  logits[0..8]: {:?}", &gllm_logits[..8]);
    eprintln!("[DIAG-005] gold logits[0..8]: {:?}", &golden_logits[..8]);

    // Check for all-zero or degenerate output
    let nonzero = gllm_logits.iter().filter(|&&v| v != 0.0).count();
    eprintln!("[DIAG-005] nonzero logits: {}/{}", nonzero, gllm_logits.len());

    // The test PASSES regardless of argmax match — it's diagnostic.
    // We just assert that the output is non-degenerate.
    assert!(nonzero > 0, "All logits are zero — mega-kernel produced no output");
    assert!(sim > 0.0, "Cosine similarity is non-positive — output is garbage");
}

/// DIAG-006: Verify gate_proj weight layout in blob.
///
/// gate_proj in safetensors: [1536, 576].
/// If loader transposed: blob has [576, 1536] layout (row stride = 1536*4).
/// If NOT transposed: blob has [1536, 576] layout (row stride = 576*4).
/// We test both interpretations to determine actual layout.
#[test]
fn diagnostic_gate_proj_layout() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    // gate_proj: safetensors shape [1536, 576]
    // Read with 576 cols (as if [*, 576])
    let row0_576 = client.diagnostic_weight_row("L0.gate_proj", 0, 576).unwrap();
    // Read with 1536 cols (as if [*, 1536])
    let row0_1536 = client.diagnostic_weight_row("L0.gate_proj", 0, 1536).unwrap();
    // Read row 1 with 576 cols
    let row1_576 = client.diagnostic_weight_row("L0.gate_proj", 1, 576).unwrap();

    eprintln!("[DIAG-006] gate_proj row0 (576 cols): first4={:?}", &row0_576[..4]);
    eprintln!("[DIAG-006] gate_proj row0 (1536 cols): first4={:?}", &row0_1536[..4]);
    eprintln!("[DIAG-006] gate_proj row1 (576 cols): first4={:?}", &row1_576[..4]);

    // If NOT transposed [1536, 576]: row0=row0_576, row1 starts at byte 576*4
    // row0_576 and row0_1536[0..576] should match exactly
    let match_exact = (0..576).all(|i| (row0_576[i] - row0_1536[i]).abs() < 1e-10);
    eprintln!("[DIAG-006] row0_576 == row0_1536[0..576]: {}", match_exact);

    // If transposed [576, 1536]: "row 0" spans 1536 elements, no row 1 in 576-col sense
    // row0_576 = blob[0..576], row1_576 = blob[576..1152]
    // row0_1536 = blob[0..1536]
    // So row1_576 should equal row0_1536[576..1152] if transposed
    let match_transposed = (0..576).all(|i| (row1_576[i] - row0_1536[i + 576]).abs() < 1e-10);
    eprintln!("[DIAG-006] row1_576 == row0_1536[576..1152] (transposed): {}", match_transposed);

    if match_exact && !match_transposed {
        eprintln!("[DIAG-006] CONCLUSION: NOT transposed — [1536, 576] layout");
    } else if match_transposed && !match_exact {
        eprintln!("[DIAG-006] CONCLUSION: TRANSPOSED — [576, 1536] layout");
    } else {
        eprintln!("[DIAG-006] CONCLUSION: AMBIGUOUS — both match or neither");
    }
}

/// DIAG-007: Verify GEMM trans_b correctness using golden layer0 data.
#[test]
fn diagnostic_layer0_gemm_trans_b_verification() {
    let _ = env_logger::builder().is_test(true).try_init();

    let path = std::path::Path::new("tests/e2e_alignment/data/golden_layer0_ops.safetensors");
    if !path.exists() {
        eprintln!("[DIAG-007] Golden data not found, skipping. Run: python3 tests/e2e_alignment/generate_layer0_golden.py");
        return;
    }
    let data = std::fs::read(path).expect("read golden file");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse safetensors");

    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name} not found: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let normed = load_f32_vec("L0_input_norm");  // [5, 576]
    let q_weight = load_f32_vec("L0_q_weight");   // [576, 576]
    let q_proj_golden = load_f32_vec("L0_q_proj"); // [5, 576]

    let seq_len = 5usize;
    let hidden = 576usize;
    let q_dim = 576usize;

    assert_eq!(normed.len(), seq_len * hidden, "L0_input_norm size mismatch");
    assert_eq!(q_weight.len(), q_dim * hidden, "L0_q_weight size mismatch");
    assert_eq!(q_proj_golden.len(), seq_len * q_dim, "L0_q_proj size mismatch");

    // Manual GEMM: C = A * B^T where A=[seq_len, hidden], B=[q_dim, hidden]
    // C[i][j] = sum_p A[i][p] * B[j][p]
    let mut q_proj_manual = vec![0.0f32; seq_len * q_dim];
    for i in 0..seq_len {
        for j in 0..q_dim {
            let mut sum = 0.0f32;
            for p in 0..hidden {
                sum += normed[i * hidden + p] * q_weight[j * hidden + p];
            }
            q_proj_manual[i * q_dim + j] = sum;
        }
    }

    let max_diff = (0..q_proj_golden.len())
        .map(|i| (q_proj_manual[i] - q_proj_golden[i]).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[DIAG-007] Manual GEMM vs golden max_diff: {:.2e}", max_diff);

    // Verify gllm weight blob data matches golden
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let q_w_row0 = client.diagnostic_weight_row("L0.q_proj", 0, hidden).unwrap();
    let q_w_row1 = client.diagnostic_weight_row("L0.q_proj", 1, hidden).unwrap();

    let mut weight_diff = 0.0f32;
    for i in 0..hidden {
        let d0 = (q_w_row0[i] - q_weight[i]).abs();
        let d1 = (q_w_row1[i] - q_weight[hidden + i]).abs();
        weight_diff = weight_diff.max(d0).max(d1);
    }
    eprintln!("[DIAG-007] gllm q_weight vs golden weight max_diff: {:.2e}", weight_diff);

    eprintln!("[DIAG-007] golden q_weight row0 first4: {:?}", &q_weight[..4]);
    eprintln!("[DIAG-007] gllm   q_weight row0 first4: {:?}", &q_w_row0[..4]);
    eprintln!("[DIAG-007] golden normed[0] first4: {:?}", &normed[..4]);
    eprintln!("[DIAG-007] golden q_proj[0] first4: {:?}", &q_proj_golden[..4]);
    eprintln!("[DIAG-007] manual q_proj[0] first4: {:?}", &q_proj_manual[..4]);

    assert!(max_diff < 1e-3, "Manual GEMM should match golden exactly, got max_diff={max_diff:.2e}");
    eprintln!("[DIAG-007] PASS: Manual GEMM matches golden");
}

/// DIAG-008: Compare intermediate activations from mega-kernel scratchpad with golden.
///
/// This test extracts the embedding output from the scratchpad (offset 0)
/// and compares with the golden hidden_layer_0 from HuggingFace reference.
#[test]
fn diagnostic_intermediate_activations() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    // Load golden hidden_layer_0
    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-008] Golden data not found, skipping. Run: python3 tests/e2e_alignment/generate_golden_smollm2.py");
        return;
    }
    let golden_data = std::fs::read(golden_path).expect("read golden");
    let golden_tensors = safetensors::SafeTensors::deserialize(&golden_data).expect("parse golden");

    let load_golden = |name: &str| -> Vec<f32> {
        let view = golden_tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_hidden = load_golden("hidden_layer_0"); // [5, 576]
    let golden_logits = {
        let all = load_golden("logits"); // [5, 49152]
        all[4 * 49152..5 * 49152].to_vec() // last token
    };

    // Use the existing diagnostic_prefill_logits path but also check embedding via weight row
    // We know embedding is at scratchpad offset 0 and embedding row for token 504 should match golden
    let gllm_logits = client.diagnostic_prefill_logits(golden::INPUT_IDS)
        .expect("Failed to get prefill logits");

    // Check embedding lookup by comparing with DIAG-001 logic
    let embed_row = client.diagnostic_weight_row("embed", golden::INPUT_IDS[0] as usize, golden::HIDDEN_SIZE)
        .expect("Failed to read embedding row");
    let golden_row = &golden_hidden[0..golden::HIDDEN_SIZE];
    let embed_sim = cosine_similarity(&embed_row, golden_row);
    eprintln!("[DIAG-008] Embedding row 504 vs golden hidden[0]: cosine_sim={:.6}", embed_sim);

    // Compare logits
    let logits_sim = cosine_similarity(&gllm_logits, &golden_logits);
    let gllm_argmax = argmax(&gllm_logits);
    let gold_argmax = argmax(&golden_logits);
    eprintln!("[DIAG-008] Logits cosine_sim={:.6}", logits_sim);
    eprintln!("[DIAG-008] gllm argmax={} ({:.4}), gold argmax={} ({:.4})",
        gllm_argmax, gllm_logits[gllm_argmax], gold_argmax, golden_logits[gold_argmax]);

    // Print golden hidden[0] vs golden_row to verify they match
    eprintln!("[DIAG-008] golden hidden[0] first4: {:?}", &golden_hidden[..4]);
    eprintln!("[DIAG-008] golden logits[0..4]: {:?}", &golden_logits[..4]);
    eprintln!("[DIAG-008] gllm   logits[0..4]: {:?}", &gllm_logits[..4]);

    if embed_sim > 0.9999 && logits_sim < 0.9 {
        eprintln!("[DIAG-008] CONCLUSION: Embedding correct, error in transformer layers");
    } else if embed_sim < 0.99 {
        eprintln!("[DIAG-008] CONCLUSION: Embedding wrong — issue in embedding lookup");
    } else {
        eprintln!("[DIAG-008] CONCLUSION: Partial match — investigate further");
    }

    assert!(gllm_logits.iter().any(|&v| v != 0.0), "Logits should not be all zero");
}

/// DIAG-009: Per-op layer 0 verification against golden reference.
///
/// Compares:
/// 1. Embedding output (should match golden hidden_layer_0)
/// 2. L0_input_norm (should match golden L0_input_norm)
/// 3. L0_q_proj (should match golden L0_q_proj)
/// Uses golden data from generate_layer0_golden.py which has per-op intermediate values.
#[test]
fn diagnostic_layer0_per_op_verification() {
    let _ = env_logger::builder().is_test(true).try_init();

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_layer0_ops.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-009] Golden layer0 data not found, skipping. Run: python3 tests/e2e_alignment/generate_layer0_golden.py");
        return;
    }

    let data = std::fs::read(golden_path).expect("read golden file");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse safetensors");

    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name} not found: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_hidden = load_f32_vec("hidden_layer_0");  // [5, 576] = embedding output
    let golden_normed = load_f32_vec("L0_input_norm");    // [5, 576]
    let golden_q_proj = load_f32_vec("L0_q_proj");        // [5, 576]
    let golden_k_proj = load_f32_vec("L0_k_proj");        // [5, 192]
    let golden_v_proj = load_f32_vec("L0_v_proj");        // [5, 192]
    let golden_q_weight = load_f32_vec("L0_q_weight");    // [576, 576]

    let seq = 5usize;
    let hidden = 576usize;
    let q_dim = 576usize;
    let k_dim = 192usize;

    // Manual RmsNorm to verify golden data
    // SmolLM2 uses eps=1e-5
    let eps = 1e-5f32;
    let mut manual_normed = vec![0.0f32; seq * hidden];
    for i in 0..seq {
        let row = &golden_hidden[i * hidden..(i + 1) * hidden];
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / hidden as f32 + eps).sqrt();
        for j in 0..hidden {
            manual_normed[i * hidden + j] = row[j] * inv_rms;
        }
    }
    let norm_diff = max_abs_diff(&manual_normed, &golden_normed);
    eprintln!("[DIAG-009] Manual RmsNorm vs golden L0_input_norm max_diff: {:.2e}", norm_diff);

    // Manual GEMM trans_b: C = A * B^T
    let mut manual_q = vec![0.0f32; seq * q_dim];
    for i in 0..seq {
        for j in 0..q_dim {
            let mut sum = 0.0f32;
            for p in 0..hidden {
                sum += golden_normed[i * hidden + p] * golden_q_weight[j * hidden + p];
            }
            manual_q[i * q_dim + j] = sum;
        }
    }
    let q_diff = max_abs_diff(&manual_q, &golden_q_proj);
    eprintln!("[DIAG-009] Manual GEMM(trans_b) vs golden L0_q_proj max_diff: {:.2e}", q_diff);

    eprintln!("[DIAG-009] golden normed[0] first4: {:?}", &golden_normed[..4]);
    eprintln!("[DIAG-009] golden q_proj[0] first4: {:?}", &golden_q_proj[..4]);
    eprintln!("[DIAG-009] manual  q_proj[0] first4: {:?}", &manual_q[..4]);

    assert!(norm_diff < 1e-4, "Manual RmsNorm should match golden");
    assert!(q_diff < 1e-3, "Manual GEMM should match golden");
    eprintln!("[DIAG-009] PASS: Golden data verified — manual RmsNorm and GEMM match");
}


/// DIAG-010: Placeholder — scratchpad offsets are overwritten during execution.
#[test]
fn diagnostic_scratchpad_vs_golden() {
    eprintln!("[DIAG-010] Skipped — scratchpad data is overwritten during execution");
}

/// DIAG-011: Compare final activation vs golden and check logits scale.
#[test]
fn diagnostic_final_activation_vs_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-011] Golden data not found, skipping");
        return;
    }
    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse golden");

    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_hidden_1: Vec<f32> = match tensors.tensor("hidden_layer_1") {
        Ok(_) => load_f32_vec("hidden_layer_1"),
        Err(_) => {
            eprintln!("[DIAG-011] hidden_layer_1 not found, skipping");
            return;
        }
    };

    let scratch = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS)
        .expect("scratchpad");

    let seq = 5usize;
    let hidden = 576usize;
    let read_f32 = |data: &[u8], offset: usize, count: usize| -> Vec<f32> {
        data[offset..offset + count * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // After all 30 layers, final_normed overwrites offset 0
    let final_data = read_f32(&scratch.data, 0, seq * hidden);
    let sim = cosine_similarity(&final_data, &golden_hidden_1);
    eprintln!("[DIAG-011] scratch[0] vs golden hidden_layer_1: sim={:.6}", sim);
    eprintln!("[DIAG-011]   golden h1[0] first4: {:?}", &golden_hidden_1[..4]);
    eprintln!("[DIAG-011]   scratch[0]  first4: {:?}", &final_data[..4]);

    let max_golden = golden_hidden_1.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_final = final_data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!("[DIAG-011] golden max_abs={:.4}, scratch max_abs={:.4}", max_golden, max_final);

    // Logits comparison
    let logits = client.diagnostic_prefill_logits(golden::INPUT_IDS).expect("logits");
    let golden_logits = {
        let all = load_f32_vec("logits");
        all[4 * 49152..5 * 49152].to_vec()
    };
    eprintln!("[DIAG-011] logits sim={:.6}", cosine_similarity(&logits, &golden_logits));
    eprintln!("[DIAG-011]   golden first4: {:?}", &golden_logits[..4]);
    eprintln!("[DIAG-011]   gllm   first4: {:?}", &logits[..4]);

    if max_final > 100.0 * max_golden {
        eprintln!("[DIAG-011] CONCLUSION: scratch data is garbage — buffer corruption");
    } else if sim < 0.5 {
        eprintln!("[DIAG-011] CONCLUSION: Layer output wrong");
    } else if sim > 0.99 {
        eprintln!("[DIAG-011] CONCLUSION: Layer output correct — error in final norm/logits");
    }
}

/// DIAG-012: Manual reference computation of layer 0 and compare with scratchpad.
/// Uses the scratchpad after prefill and compares the final activation (offset 0)
/// with a manual computation: norm(embed) → qkv → rope → attention → o_proj →
/// residual → norm → ffn → residual.
/// Since scratch[0] contains final_normed (after all 30 layers), we can't compare
/// directly. Instead, we compare the WEIGHT of lm_head vs embed (tie check).
#[test]
fn diagnostic_lm_head_weight_matches_embed() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    // For tied embeddings, lm_head weight should be identical to embed weight
    let embed_row = client.diagnostic_weight_row("embed", 0, 576).expect("embed row 0");
    let lm_head_row = client.diagnostic_weight_row("lm_head", 0, 576).expect("lm_head row 0");

    let diff = max_abs_diff(&embed_row, &lm_head_row);
    eprintln!("[DIAG-012] embed[0] first4: {:?}", &embed_row[..4]);
    eprintln!("[DIAG-012] lm_head[0] first4: {:?}", &lm_head_row[..4]);
    eprintln!("[DIAG-012] max_diff between embed and lm_head row 0: {:.2e}", diff);

    if diff < 1e-6 {
        eprintln!("[DIAG-012] PASS: lm_head weight matches embed (tied)");
    } else {
        eprintln!("[DIAG-012] FAIL: lm_head weight differs from embed — wrong weights!");
    }
    assert!(diff < 1e-4, "lm_head should match embed for tied embeddings");

    // Also check weight offsets
    let offsets = client.diagnostic_weight_offsets().expect("offsets");
    let embed_off = offsets.iter().find(|(n, _, _)| n == "embed").map(|(_, o, _)| *o);
    let lm_off = offsets.iter().find(|(n, _, _)| n == "lm_head").map(|(_, o, _)| *o);
    eprintln!("[DIAG-012] embed offset: {:?}, lm_head offset: {:?}", embed_off, lm_off);

    // Check: are embed and lm_head at different offsets? (they should be, since
    // weight_layout packs them separately)
    if let (Some(eo), Some(lo)) = (embed_off, lm_off) {
        if eo == lo {
            eprintln!("[DIAG-012] WARNING: embed and lm_head at SAME offset — weight reuse ok");
        } else {
            let embed_bytes = 49152 * 576 * 4;
            eprintln!("[DIAG-012] embed at {}, lm_head at {}, diff={}, embed_size={}",
                eo, lo, lo as i64 - eo as i64, embed_bytes);
        }
    }
}

/// DIAG-013: After RmsNorm weight fix, check logits alignment.
#[test]
fn diagnostic_post_normfix_logits() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-013] Golden data not found, skipping");
        return;
    }
    let golden_data = std::fs::read(golden_path).expect("read golden");
    let golden_tensors = safetensors::SafeTensors::deserialize(&golden_data).expect("parse golden");

    let load_golden = |name: &str| -> Vec<f32> {
        let view = golden_tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_logits = {
        let all = load_golden("logits"); // [5, 49152]
        all[4 * 49152..5 * 49152].to_vec()
    };

    let gllm_logits = client.diagnostic_prefill_logits(golden::INPUT_IDS)
        .expect("Failed to get prefill logits");

    let sim = cosine_similarity(&gllm_logits, &golden_logits);
    let gllm_argmax = argmax(&gllm_logits);
    let gold_argmax = argmax(&golden_logits);
    eprintln!("[DIAG-013] logits cosine_sim={:.6}", sim);
    eprintln!("[DIAG-013] gllm argmax={} ({:.4}), gold argmax={} ({:.4})",
        gllm_argmax, gllm_logits[gllm_argmax], gold_argmax, golden_logits[gold_argmax]);

    let max_diff = gllm_logits.iter().zip(golden_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[DIAG-013] max_abs_diff={:.4}", max_diff);

    // Print top-5
    let mut gllm_top: Vec<(usize, f32)> = gllm_logits.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    gllm_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut gold_top: Vec<(usize, f32)> = golden_logits.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    gold_top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("[DIAG-013] gllm top5: {:?}", &gllm_top[..5.min(gllm_top.len())]);
    eprintln!("[DIAG-013] gold top5: {:?}", &gold_top[..5.min(gold_top.len())]);

    if sim > 0.99 {
        eprintln!("[DIAG-013] PASS: logits cosine_sim > 0.99");
    } else if sim > 0.95 {
        eprintln!("[DIAG-013] IMPROVED: logits cosine_sim > 0.95 (was 0.18 before norm fix)");
    } else {
        eprintln!("[DIAG-013] FAIL: logits cosine_sim = {:.6}", sim);
    }
}

/// DIAG-014: Manual L0 input_norm with weight vs golden.
/// Verifies that RmsNorm(hidden, weight) matches the golden L0_input_norm.
#[test]
fn diagnostic_rmsnorm_with_weight() {
    let _ = env_logger::builder().is_test(true).try_init();

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_layer0_ops.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-014] Golden layer0 data not found, skipping");
        return;
    }
    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse");

    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_hidden = load_f32_vec("hidden_layer_0");  // [5, 576] = embedding output
    let golden_normed = load_f32_vec("L0_input_norm");    // [5, 576]

    let seq = 5usize;
    let hidden = 576usize;
    let eps = 1e-5f32;

    // SmolLM2 RmsNorm has a learnable weight (gamma).
    // We need to get the weight. Since tied_lm_head matches embed, we can
    // load the norm weight from the model.
    // But golden_layer0_ops doesn't include norm weight. Let's compute manually.
    // The golden L0_input_norm already includes weight, so we can reverse-engineer:
    // normed = hidden * rsqrt(mean(hidden^2) + eps) * weight
    // weight = normed / (hidden * rsqrt(mean(hidden^2) + eps))

    // For position 0:
    let row = &golden_hidden[0..hidden];
    let normed = &golden_normed[0..hidden];
    let ss: f32 = row.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / hidden as f32 + eps).sqrt();

    // Compute weight for position 0: weight[i] = normed[i] / (row[i] * inv_rms)
    let mut weights = vec![0.0f32; hidden];
    for i in 0..hidden {
        weights[i] = normed[i] / (row[i] * inv_rms);
    }

    // Check weight consistency (all positions should give same weight)
    let mut max_weight_diff = 0.0f32;
    for pos in 1..seq {
        let r = &golden_hidden[pos * hidden..(pos + 1) * hidden];
        let n = &golden_normed[pos * hidden..(pos + 1) * hidden];
        let ss2: f32 = r.iter().map(|v| v * v).sum();
        let inv_rms2 = 1.0 / (ss2 / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            let w2 = n[i] / (r[i] * inv_rms2);
            let diff = (w2 - weights[i]).abs();
            if diff > max_weight_diff {
                max_weight_diff = diff;
            }
        }
    }
    eprintln!("[DIAG-014] Max weight consistency diff across positions: {:.2e}", max_weight_diff);
    eprintln!("[DIAG-014] Weight[0..8]: {:?}", &weights[..8]);

    // Now verify: manual_normed[i] = hidden[i] * inv_rms * weight[i]
    let mut manual_normed = vec![0.0f32; seq * hidden];
    let mut max_diff = 0.0f32;
    for pos in 0..seq {
        let r = &golden_hidden[pos * hidden..(pos + 1) * hidden];
        let ss2: f32 = r.iter().map(|v| v * v).sum();
        let inv_rms2 = 1.0 / (ss2 / hidden as f32 + eps).sqrt();
        for i in 0..hidden {
            manual_normed[pos * hidden + i] = r[i] * inv_rms2 * weights[i];
            let diff = (manual_normed[pos * hidden + i] - golden_normed[pos * hidden + i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    eprintln!("[DIAG-014] Manual RmsNorm(with weight) vs golden L0_input_norm max_diff: {:.2e}", max_diff);

    // Print first few values for visual check
    eprintln!("[DIAG-014] manual normed[0..8]: {:?}", &manual_normed[..8]);
    eprintln!("[DIAG-014] golden normed[0..8]: {:?}", &golden_normed[..8]);

    assert!(max_weight_diff < 1e-4, "Weight should be consistent across positions");
    assert!(max_diff < 1e-4, "Manual RmsNorm with weight should match golden");
    eprintln!("[DIAG-014] PASS: RmsNorm formula verified (x * rsqrt(mean(x^2)+eps) * weight)");
}

#[test]
fn diagnostic_scratchpad_l0_q_vs_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_layer0_ops.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-015] Golden layer0 data not found, skipping");
        return;
    }

    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse golden");
    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_hidden = load_f32_vec("hidden_layer_0");
    let golden_q = load_f32_vec("L0_q_proj");
    let seq = 5usize;
    let hidden = 576usize;

    // Manual RmsNorm with weight
    let _golden_q_weight = load_f32_vec("L0_q_weight");
    let eps = 1e-5f32;

    // KNOWN FALSE-POSITIVE (BCE-EMB/NaN1, dismissed — root cause recorded, NOT
    // papered over): SmolLM2 has Argmax → the compiled mega-kernel has GenerateLoop
    // topology. `diagnostic_prefill_scratchpad` runs ONE decode iteration and only
    // writes the LAST prompt token's intermediates at row 0 of each scratchpad
    // region. This test reads `seq*hidden` (a multi-token prefill snapshot) and
    // compares against the full multi-token golden — a calling-convention mismatch,
    // not an engine bug. Real inference (test_e2e_alignment_smollm2) is green.
    //
    // The fix is NOT to massage expectations here (that would hide a real engine
    // issue). It must wait for the unified "operator-level mixed-precision" design
    // (architect) — once emit_* infers dtype per-op from op_input_dtype (no
    // hardcoded F32), the scratchpad magnitudes will align with golden and this
    // diagnostic will pass on its real signal. Left as-is intentionally.
    let scratch = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS).expect("scratchpad");

    // BCE-20260629-006: 动态获取 tensor offset，禁止硬编码
    let embed_offset = client.diagnostic_tensor_offset("embedding")
        .unwrap_or_else(|| panic!("[DIAG-015] embedding tensor not in named_offsets"));
    let embed = scratch.read_f32_at(embed_offset, seq * hidden);
    let embed_sim = cosine_similarity(&embed, &golden_hidden);
    eprintln!("[DIAG-015] Embedding sim={:.6} (offset={})", embed_sim, embed_offset);

    // BCE-20260629-006: 动态获取 layer.normed offset
    let normed_offset = client.diagnostic_tensor_offset("layer.normed")
        .unwrap_or_else(|| panic!("[DIAG-015] layer.normed tensor not in named_offsets"));
    let normed = scratch.read_f32_at(normed_offset, seq * hidden);
    // Manual normed with weight
    let mut manual_normed = vec![0.0f32; seq * hidden];
    let norm_w = load_f32_vec("L0_input_norm");
    for i in 0..seq {
        let row = &golden_hidden[i * hidden..(i + 1) * hidden];
        let ss: f32 = row.iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (ss / hidden as f32 + eps).sqrt();
        for j in 0..hidden {
            manual_normed[i * hidden + j] = row[j] * inv_rms * norm_w[j];
        }
    }
    let norm_sim = cosine_similarity(&normed, &manual_normed);
    let norm_max_diff = max_abs_diff(&normed, &manual_normed);
    eprintln!("[DIAG-015] L0_normed sim={:.6} max_diff={:.2e} (offset={})", norm_sim, norm_max_diff, normed_offset);
    eprintln!("[DIAG-015]   normed first4: {:?}", &normed[..4]);
    eprintln!("[DIAG-015]   manual first4: {:?}", &manual_normed[..4]);

    // BCE-20260629-006: 动态获取 layer.q offset
    let q_offset = client.diagnostic_tensor_offset("layer.q")
        .unwrap_or_else(|| panic!("[DIAG-015] layer.q tensor not in named_offsets"));
    let q_out = scratch.read_f32_at(q_offset, seq * hidden);
    let q_sim = cosine_similarity(&q_out, &golden_q);
    let q_max_diff = max_abs_diff(&q_out, &golden_q);
    eprintln!("[DIAG-015] L0_q_proj sim={:.6} max_diff={:.2e}", q_sim, q_max_diff);
    eprintln!("[DIAG-015]   gllm_q first4: {:?}", &q_out[..4]);
    eprintln!("[DIAG-015]   golden_q first4: {:?}", &golden_q[..4]);
}

/// DIAG-016: Compare encode-mode logits vs generate-mode logits vs golden.
///
/// Purpose: Determine whether the cosine_sim=0.63 regression is caused by:
/// (a) The forward pass itself (both encode and generate modes wrong)
/// (b) The generate loop framework (encode correct, generate wrong)
#[test]
fn diagnostic_encode_vs_generate_logits() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    let Some(golden_logits) = load_golden_logits() else {
        eprintln!("[DIAG-016] Golden logits not found, skipping");
        return;
    };

    let prompt = golden::INPUT_IDS;

    // 1. Generate-mode logits (output_mode_selector=0, with generate loop)
    let gen_logits = client.diagnostic_prefill_logits(prompt)
        .expect("generate-mode logits");

    // 2. Encode-mode logits (output_mode_selector=3, forward-only)
    let enc_logits = client.diagnostic_forward_only(prompt)
        .expect("encode-mode logits");

    let vocab = golden::VOCAB_SIZE;

    // Compare generate-mode vs golden
    let gen_sim = cosine_similarity(&gen_logits, &golden_logits);
    let gen_argmax = argmax(&gen_logits);
    let gold_argmax = argmax(&golden_logits);
    eprintln!("[DIAG-016] Generate vs Golden: sim={:.6} argmax=({} vs {})",
        gen_sim, gen_argmax, gold_argmax);
    {
        let (gi, gv) = gen_logits.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        let (gdi, gdv) = golden_logits.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        eprintln!("[DIAG-016]   gen_argmax_val={:.4} gold_argmax_val={:.4}", gv, gdv);
    }

    // Compare encode-mode vs golden
    let enc_sim = cosine_similarity(&enc_logits, &golden_logits);
    let enc_argmax = argmax(&enc_logits);
    eprintln!("[DIAG-016] Encode vs Golden: sim={:.6} argmax=({} vs {})",
        enc_sim, enc_argmax, gold_argmax);
    {
        let (ei, ev) = enc_logits.iter().enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        eprintln!("[DIAG-016]   enc_argmax_val={:.4}", ev);
    }

    // Compare encode-mode vs generate-mode
    let eg_sim = cosine_similarity(&enc_logits, &gen_logits);
    eprintln!("[DIAG-016] Encode vs Generate: sim={:.6}", eg_sim);

    // Compare first few values
    eprintln!("[DIAG-016] gen first8: {:?}", &gen_logits[..8]);
    eprintln!("[DIAG-016] enc first8: {:?}", &enc_logits[..8]);
    eprintln!("[DIAG-016] gold first8: {:?}", &golden_logits[..8]);

    // Diagnostic conclusion
    if enc_sim > 0.99 && gen_sim < 0.99 {
        eprintln!("[DIAG-016] CONCLUSION: Forward pass is correct, generate loop framework is broken");
    } else if enc_sim < 0.99 && gen_sim < 0.99 {
        eprintln!("[DIAG-016] CONCLUSION: Forward pass itself is broken (both modes wrong)");
    } else if enc_sim > 0.99 && gen_sim > 0.99 {
        eprintln!("[DIAG-016] CONCLUSION: Both modes correct (problem may be elsewhere)");
    }
}

/// DIAG-017: Compare final hidden state (L29 ffn_resid) vs golden hidden_layer_30.
///
/// Purpose: The forward pass is broken (DIAG-016 confirmed). Now narrow down WHERE.
/// Read the scratchpad at L29_ffn_resid offset and compare with golden hidden_layer_30.
/// If L29 output is already wrong, the bug is in the 30-layer forward pass.
/// If L29 output is correct, the bug is in final_norm or lm_head.
#[test]
fn diagnostic_final_hidden_vs_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-017] Golden data not found, skipping");
        return;
    }
    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse golden");
    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_h30 = load_f32_vec("hidden_layer_30");
    let golden_logits = load_f32_vec("logits");
    let seq = 5usize;
    let hidden = golden::HIDDEN_SIZE;
    let vocab = golden::VOCAB_SIZE;

    // KNOWN FALSE-POSITIVE (BCE-EMB/NaN1, dismissed — root cause recorded, NOT
    // papered over): SmolLM2 has Argmax → compiled mega-kernel has GenerateLoop
    // topology. `diagnostic_prefill_scratchpad` runs ONE decode iteration that
    // only writes the LAST prompt token's intermediates at row 0. Reading
    // `seq*hidden` and the `[(seq-1)*hidden..]` slice hits uninitialized rows —
    // a calling-convention mismatch, not an engine bug. Real inference is green.
    // The real fix is operator-level mixed-precision (architect, pending): once
    // emit_* infers dtype per-op from op_input_dtype (no hardcoded F32), the
    // scratchpad magnitudes/contents will align with golden. Left as-is.
    let scratch = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS).expect("scratchpad");

    // BCE-20260629-006: 动态获取 tensor offset，禁止硬编码
    let l29_offset = client.diagnostic_tensor_offset("layer.ffn_resid")
        .unwrap_or_else(|| panic!("[DIAG-017] layer.ffn_resid not in named_offsets"));
    let l29_data = scratch.read_f32_at(l29_offset, seq * hidden);

    // Compare with golden hidden_layer_30
    let h30_sim = cosine_similarity(&l29_data, &golden_h30);
    let h30_diff = max_abs_diff(&l29_data, &golden_h30);
    eprintln!("[DIAG-017] L29_ffn_resid vs golden hidden_layer_30: sim={:.6} max_diff={:.2e}",
        h30_sim, h30_diff);

    // Check last token (position 4) specifically
    let l29_last = &l29_data[(seq-1)*hidden..seq*hidden];
    let g30_last = &golden_h30[(seq-1)*hidden..seq*hidden];
    let last_sim = cosine_similarity(l29_last, g30_last);
    let last_diff = max_abs_diff(l29_last, g30_last);
    eprintln!("[DIAG-017] Last token: sim={:.6} max_diff={:.2e}", last_sim, last_diff);

    // Print first few values for visual comparison
    eprintln!("[DIAG-017] gllm L29 last_token first8: {:?}", &l29_last[..8]);
    eprintln!("[DIAG-017] golden h30 last_token first8: {:?}", &g30_last[..8]);

    // Check amplitude
    let gllm_rms = (l29_last.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / hidden as f64).sqrt();
    let gold_rms = (g30_last.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / hidden as f64).sqrt();
    eprintln!("[DIAG-017] Last token RMS: gllm={:.4} golden={:.4} ratio={:.4}",
        gllm_rms, gold_rms, gllm_rms / gold_rms);

    // BCE-20260629-006: 动态获取 final_normed offset
    let normed_offset = client.diagnostic_tensor_offset("final_normed")
        .unwrap_or_else(|| panic!("[DIAG-017] final_normed not in named_offsets"));
    let normed_data = scratch.read_f32_at(normed_offset, seq * hidden);
    let gllm_normed_last = &normed_data[(seq-1)*hidden..seq*hidden];
    let normed_rms = (gllm_normed_last.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / hidden as f64).sqrt();
    eprintln!("[DIAG-017] final_normed last_token first8: {:?}", &gllm_normed_last[..8]);
    eprintln!("[DIAG-017] final_normed last_token RMS: {:.4}", normed_rms);

    // Read logits from scratchpad — BCE-20260629-002: decode kernel writes row 0
    let logits_off = scratch.logits_offset;
    let logits_row_bytes = vocab * 4;
    let last_row_off = logits_off + (seq - 1) * logits_row_bytes;
    let gllm_logits = scratch.read_f32_at(last_row_off, vocab);
    let logit_sim = cosine_similarity(&gllm_logits, &golden_logits[(seq-1)*vocab..seq*vocab]);
    eprintln!("[DIAG-017] Logits vs golden: sim={:.6}", logit_sim);

    // Conclusion
    if h30_sim < 0.99 {
        eprintln!("[DIAG-017] CONCLUSION: Hidden state after 30 layers is WRONG (sim={:.6}). Bug is in layer computation.", h30_sim);
    } else if logit_sim < 0.99 {
        eprintln!("[DIAG-017] CONCLUSION: Hidden state is correct but logits are wrong. Bug is in final_norm or lm_head.");
    } else {
        eprintln!("[DIAG-017] CONCLUSION: Both correct. Bug is elsewhere.");
    }
}

/// DIAG-018: Compare hidden state at multiple layer boundaries using golden data.
///
/// Scratchpad tensors get reused, but we can compare:
/// - embedding (offset=0) vs golden hidden_layer_0
/// - final_normed (offset=0) vs golden hidden_layer_30 
/// - L29_ffn_resid (offset=18874368) vs golden hidden_layer_29
///
/// The RMS ratio at L29 vs golden tells us how much amplification per layer.
#[test]
fn diagnostic_layer_boundary_rms() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_smollm2_135m.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-018] Golden data not found, skipping");
        return;
    }
    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("parse golden");
    let load_f32_vec = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let seq = 5usize;
    let hidden = golden::HIDDEN_SIZE;

    let scratch = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS).expect("scratchpad");

    // Embedding at offset 0 (lifetime=[0,9])
    let gllm_embed = scratch.read_f32_at(0, seq * hidden);
    let golden_h0 = load_f32_vec("hidden_layer_0");
    let embed_sim = cosine_similarity(&gllm_embed, &golden_h0);
    let embed_last = &gllm_embed[(seq-1)*hidden..seq*hidden];
    let golden_h0_last = &golden_h0[(seq-1)*hidden..seq*hidden];
    let gllm_embed_rms = rms(embed_last);
    let golden_h0_rms = rms(golden_h0_last);
    eprintln!("[DIAG-018] Embedding (L0 input): sim={:.6} gllm_rms={:.4} golden_rms={:.4}",
        embed_sim, gllm_embed_rms, golden_h0_rms);

    // L29_ffn_resid at offset=18874368 (lifetime=[450,451])
    let l29_off = 18874368usize;
    let gllm_l29 = scratch.read_f32_at(l29_off, seq * hidden);
    let golden_h29 = load_f32_vec("hidden_layer_29");
    let l29_sim = cosine_similarity(&gllm_l29, &golden_h29);
    let gllm_l29_last = &gllm_l29[(seq-1)*hidden..seq*hidden];
    let golden_h29_last = &golden_h29[(seq-1)*hidden..seq*hidden];
    let gllm_l29_rms = rms(gllm_l29_last);
    let golden_h29_rms = rms(golden_h29_last);
    eprintln!("[DIAG-018] L29 output: sim={:.6} gllm_rms={:.4} golden_rms={:.4} ratio={:.2}",
        l29_sim, gllm_l29_rms, golden_h29_rms, gllm_l29_rms / golden_h29_rms);

    // final_normed at offset=0 (lifetime=[451,452]) — overwrites embedding!
    let gllm_normed = scratch.read_f32_at(0, seq * hidden);
    let golden_h30 = load_f32_vec("hidden_layer_30");
    let normed_sim = cosine_similarity(&gllm_normed, &golden_h30);
    let gllm_normed_last = &gllm_normed[(seq-1)*hidden..seq*hidden];
    let golden_h30_last = &golden_h30[(seq-1)*hidden..seq*hidden];
    let gllm_normed_rms = rms(gllm_normed_last);
    let golden_h30_rms = rms(golden_h30_last);
    eprintln!("[DIAG-018] Final normed: sim={:.6} gllm_rms={:.4} golden_rms={:.4}",
        normed_sim, gllm_normed_rms, golden_h30_rms);

    // Per-layer amplification estimate
    // If every layer multiplies RMS by the same factor, 
    // ratio = (gllm_l29_rms / golden_h29_rms) and
    // per_layer_factor = ratio^(1/30)
    let ratio = gllm_l29_rms / golden_h29_rms;
    let per_layer = ratio.powf(1.0 / 30.0);
    eprintln!("[DIAG-018] Total amplification ratio: {:.2}x, per-layer factor: {:.4}",
        ratio, per_layer);

    // Cross-check: manual final norm
    // final_normed = gllm_l29 * inv_rms * weight
    // If final_norm is correct, the normalized output should match golden_h30
    let eps = 1e-5f32;
    let mut manual_normed = vec![0.0f32; hidden];
    let ss: f32 = gllm_l29_last.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / hidden as f32 + eps).sqrt();
    for j in 0..hidden {
        manual_normed[j] = gllm_l29_last[j] * inv_rms;
    }
    let manual_rms = rms(&manual_normed);
    eprintln!("[DIAG-018] Manual inv_rms normalization: inv_rms={:.6} manual_rms={:.4}", inv_rms, manual_rms);

    // The final_normed should be gllm_l29 * inv_rms * norm_weight
    // If gllm_normed_rms ≈ manual_rms * norm_weight_rms, then final_norm is correct
    // and the bug is in the 30-layer accumulation
    eprintln!("[DIAG-018] first8 gllm_normed: {:?}", &gllm_normed_last[..8]);
    eprintln!("[DIAG-018] first8 golden_h30:  {:?}", &golden_h30_last[..8]);
    eprintln!("[DIAG-018] first8 manual:     {:?}", &manual_normed[..8]);
}

fn rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / v.len() as f64).sqrt()
}

/// DIAG-019: Compare weight blob values with original safetensors BF16→F32.
///
/// Reads the original model.safetensors, extracts BF16 weight tensors,
/// converts to F32, and compares with the weight blob at known offsets.
/// This directly verifies weight packing correctness.
#[test]
fn diagnostic_weight_blob_vs_safetensors() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    // Load original model safetensors
    let model_path = std::path::Path::new(
        "/home/putao/.gllm/models/huggingface/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/model.safetensors"
    );
    assert!(model_path.exists(), "Model safetensors not found at {:?}", model_path);
    let data = std::fs::read(model_path).expect("Failed to read model safetensors");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("Failed to deserialize");

    // Get weight offsets from mega-kernel
    let offsets = client.diagnostic_weight_offsets().expect("weight offsets");

    // Mapping: canonical name → safetensors tensor name
    let weight_checks: &[(&str, &str)] = &[
        ("L0.input_norm", "model.layers.0.input_layernorm.weight"),
        ("L0.q_proj", "model.layers.0.self_attn.q_proj.weight"),
        ("L0.k_proj", "model.layers.0.self_attn.k_proj.weight"),
        ("L0.v_proj", "model.layers.0.self_attn.v_proj.weight"),
        ("L0.o_proj", "model.layers.0.self_attn.o_proj.weight"),
        ("L0.post_norm", "model.layers.0.post_attention_layernorm.weight"),
        ("L0.gate_proj", "model.layers.0.mlp.gate_proj.weight"),
        ("L0.up_proj", "model.layers.0.mlp.up_proj.weight"),
        ("L0.down_proj", "model.layers.0.mlp.down_proj.weight"),
        ("L1.input_norm", "model.layers.1.input_layernorm.weight"),
        ("L1.q_proj", "model.layers.1.self_attn.q_proj.weight"),
        ("embed", "model.embed_tokens.weight"),
        ("final_norm", "model.norm.weight"),
    ];

    for (canonical, safetensor_name) in weight_checks {
        let view = match tensors.tensor(safetensor_name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[DIAG-019] SKIP {}: not found in safetensors ({})", safetensor_name, e);
                continue;
            }
        };

        let shape = view.shape();
        let dtype = view.dtype();

        // Get offset in weight blob
        let blob_offset = offsets.iter()
            .find(|(n, _, _)| n == *canonical)
            .map(|(_, o, _)| *o);
        let blob_off = match blob_offset {
            Some(o) => o,
            None => {
                eprintln!("[DIAG-019] SKIP {}: not found in named_offsets", canonical);
                continue;
            }
        };

        // Read from safetensors: BF16 → F32
        let raw_bytes = view.data();
        let safetensor_f32: Vec<f32> = match dtype {
            safetensors::Dtype::BF16 => {
                let numel = raw_bytes.len() / 2;
                raw_bytes.chunks_exact(2)
                    .map(|c| {
                        let b = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]]));
                        b.to_f32()
                    })
                    .collect()
            }
            safetensors::Dtype::F32 => {
                raw_bytes.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            other => {
                eprintln!("[DIAG-019] SKIP {}: unsupported dtype {:?}", safetensor_name, other);
                continue;
            }
        };

        // Read from weight blob at offset
        let compare_count = safetensor_f32.len().min(576);
        let blob_f32 = match client.diagnostic_weight_row(canonical, 0, compare_count) {
            Some(v) => v,
            None => {
                eprintln!("[DIAG-019] FAIL {}: cannot read {} elems from blob at offset {}",
                    canonical, compare_count, blob_off);
                continue;
            }
        };

        // Compare first N elements
        let compare_n = blob_f32.len().min(safetensor_f32.len());
        let sim = cosine_similarity(&blob_f32[..compare_n], &safetensor_f32[..compare_n]);
        let max_diff = max_abs_diff(&blob_f32[..compare_n], &safetensor_f32[..compare_n]);
        let safetensor_rms = rms(&safetensor_f32[..compare_n]);
        let blob_rms = rms(&blob_f32[..compare_n]);

        let pass = sim > 0.9999;
        eprintln!("[DIAG-019] {} ({} shape={:?}): sim={:.6} max_diff={:.6e} rms_sf={:.4} rms_blob={:.4} {}",
            canonical, safetensor_name, shape, sim, max_diff, safetensor_rms, blob_rms,
            if pass { "PASS" } else { "FAIL" });

        // Show first 4 values for debugging
        let n_show = 4.min(compare_n);
        eprintln!("  safetensors[0..{}]: {:?}", n_show, &safetensor_f32[..n_show]);
        eprintln!("  blob[0..{}]:        {:?}", n_show, &blob_f32[..n_show]);

        if !pass && canonical.contains(&"L0") {
            // For layer 0 weights, this is critical
            panic!("[DIAG-019] Weight blob mismatch for {} (sim={:.6})", canonical, sim);
        }
    }
}

/// DIAG-020: Compare per-op outputs for L0 with golden reference.
///
/// Uses golden_layer0_ops.safetensors which contains L0 intermediate results:
/// hidden_layer_0 (input), L0_input_norm, L0_q_proj, L0_k_proj, L0_v_proj,
/// hidden_layer_1 (output after full L0).
///
/// Strategy: Run mega-kernel forward pass, then read scratchpad at known offsets
/// to extract L0 intermediates and compare with golden.
#[test]
fn diagnostic_l0_per_op_vs_golden() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    // Load golden L0 ops
    let golden_path = std::path::Path::new("tests/e2e_alignment/data/golden_layer0_ops.safetensors");
    if !golden_path.exists() {
        eprintln!("[DIAG-020] SKIP: golden_layer0_ops.safetensors not found");
        return;
    }
    let data = std::fs::read(golden_path).expect("read golden");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize golden");

    let load_f32 = |name: &str| -> Vec<f32> {
        let view = tensors.tensor(name).expect(&format!("tensor {}", name));
        view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    let golden_h0 = load_f32("hidden_layer_0");   // [5, 576]
    let golden_norm = load_f32("L0_input_norm");   // [5, 576]
    let golden_q = load_f32("L0_q_proj");          // [5, 576]
    let golden_k = load_f32("L0_k_proj");          // [5, 192]
    let golden_v = load_f32("L0_v_proj");          // [5, 192]
    let golden_h1 = load_f32("hidden_layer_1");    // [5, 576]

    // Run forward pass and get scratchpad
    let scratch = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS)
        .expect("scratchpad");

    // The scratchpad has the full forward pass state.
    // We need to find L0 intermediate tensor offsets.
    // Get buffer allocation info
    let offsets = client.diagnostic_weight_offsets().expect("weight offsets");
    eprintln!("[DIAG-020] Scratchpad size: {} bytes", scratch.data.len());
    eprintln!("[DIAG-020] Total weight offsets: {} tensors", offsets.len());

    // Print first 40 tensor names to understand the layout
    for (name, off, _dtype) in offsets.iter().take(40) {
        eprintln!("  weight: {} @ {}", name, off);
    }

    // Read scratchpad data at various offsets to find intermediate tensors.
    // The BufferAllocation assigns offsets based on tensor lifetimes.
    // We need to find the scratchpad offsets for L0 intermediates.
    // 
    // Strategy: scan the scratchpad for regions that match golden values.
    // Start with L0_input_norm output - should be hidden_layer_0 after RmsNorm.
    let hidden = golden::HIDDEN_SIZE; // 576
    let seq_len = golden::INPUT_IDS.len(); // 5
    let elem_bytes = 4; // f32

    // Try reading at stride-aligned offsets from the beginning
    // The scratchpad typically starts with the embedding output
    let row_bytes = hidden * elem_bytes; // 2304 bytes per row
    let layer_bytes = seq_len * row_bytes; // 11520 bytes per layer activation

    // Check embedding region first (should match golden_h0)
    let embed_rms = rms(&scratch.read_f32_at(0, hidden));
    eprintln!("[DIAG-020] Scratchpad[0..{}] RMS = {:.4}", hidden, embed_rms);

    // The golden hidden_layer_0 is the embedding output (after token embedding lookup)
    // It should be at the very start of scratchpad
    if scratch.data.len() >= layer_bytes {
        let sp_h0: Vec<f32> = scratch.data[0..layer_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sim_h0 = cosine_similarity(&sp_h0, &golden_h0);
        let rms_h0 = rms(&sp_h0);
        let rms_golden_h0 = rms(&golden_h0);
        eprintln!("[DIAG-020] Scratchpad[0..{}] vs golden h0: sim={:.6} rms_sp={:.4} rms_golden={:.4}",
            layer_bytes, sim_h0, rms_h0, rms_golden_h0);
    }

    // Now search for L0_input_norm output in the scratchpad
    // Try multiples of layer_bytes
    eprintln!("[DIAG-020] Searching for L0_input_norm in scratchpad...");
    let golden_norm_rms = rms(&golden_norm);
    for off_mult in 0..20 {
        let off = off_mult * layer_bytes;
        if off + layer_bytes > scratch.data.len() { break; }
        let sp_vals: Vec<f32> = scratch.data[off..off + layer_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sp_rms = rms(&sp_vals);
        let sim = cosine_similarity(&sp_vals, &golden_norm);
        if sim > 0.5 || sp_rms > 0.01 {
            eprintln!("  offset {} ({}×{}): rms={:.4} sim_vs_norm={:.6}",
                off, off_mult, layer_bytes, sp_rms, sim);
        }
    }

    // Also search for L0_q_proj output (should be [5, 576] = same as hidden)
    let golden_q_rms = rms(&golden_q);
    eprintln!("[DIAG-020] Searching for L0_q_proj (rms={:.4})...", golden_q_rms);
    for off_mult in 0..20 {
        let off = off_mult * layer_bytes;
        if off + layer_bytes > scratch.data.len() { break; }
        let sp_vals: Vec<f32> = scratch.data[off..off + layer_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sp_rms = rms(&sp_vals);
        let sim = cosine_similarity(&sp_vals, &golden_q);
        if sim > 0.5 || sp_rms > 0.01 {
            eprintln!("  offset {} ({}×{}): rms={:.4} sim_vs_q={:.6}",
                off, off_mult, layer_bytes, sp_rms, sim);
        }
    }

    // Also search with 192-dim stride for K/V projections
    let kv_row_bytes = 192 * elem_bytes; // 768 bytes
    let kv_layer_bytes = seq_len * kv_row_bytes; // 3840 bytes
    let golden_k_rms = rms(&golden_k);
    eprintln!("[DIAG-020] Searching for L0_k_proj (rms={:.4})...", golden_k_rms);
    for off_mult in 0..60 {
        let off = off_mult * kv_layer_bytes;
        if off + kv_layer_bytes > scratch.data.len() { break; }
        let sp_vals: Vec<f32> = scratch.data[off..off + kv_layer_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sp_rms = rms(&sp_vals);
        let sim = cosine_similarity(&sp_vals, &golden_k);
        if sim > 0.5 {
            eprintln!("  offset {} ({}×{}): rms={:.4} sim_vs_k={:.6}",
                off, off_mult, kv_layer_bytes, sp_rms, sim);
        }
    }
}

/// DIAG-022: Verify weight blob rows BEYOND row 0 match safetensors.
///
/// read_weight_row(name, row, cols) computes offset as:
///   named_offset + row * cols * 4
/// So `cols` MUST be the actual row width of the tensor (576 for q_proj),
/// NOT the number of elements to read.
/// Returns exactly `cols` elements starting at that row.
#[test]
fn diagnostic_weight_blob_deep_rows() {
    let _ = env_logger::builder().is_test(true).try_init();

    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    let model_path = std::path::Path::new(
        "/home/putao/.gllm/models/huggingface/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/model.safetensors"
    );
    let data = std::fs::read(model_path).expect("read");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("deserialize");

    // L0.q_proj: shape=[576, 576], cols=576
    let view = tensors.tensor("model.layers.0.self_attn.q_proj.weight").unwrap();
    let raw = view.data();
    let all_f32: Vec<f32> = raw.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let cols = 576usize;
    for row in [0usize, 1, 100, 575] {
        let start = row * cols;
        let sf_vals = &all_f32[start..start + cols];
        let blob_vals = client.diagnostic_weight_row("L0.q_proj", row, cols).unwrap();
        let sim = cosine_similarity(sf_vals, &blob_vals);
        let max_d = max_abs_diff(sf_vals, &blob_vals);
        eprintln!("[DIAG-022] L0.q_proj row {}: sim={:.6} max_diff={:.6e} first4={:?} vs {:?}",
            row, sim, max_d, &sf_vals[..4], &blob_vals[..4]);
        assert!((sim - 1.0).abs() < 0.0001, "L0.q_proj row {} mismatch: sim={:.6}", row, sim);
    }

    // L0.gate_proj: shape=[1536, 576], cols=576
    let gate_view = tensors.tensor("model.layers.0.mlp.gate_proj.weight").unwrap();
    let gate_raw = gate_view.data();
    let gate_f32: Vec<f32> = gate_raw.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let gate_cols = 576usize;
    for row in [0usize, 1, 768, 1535] {
        let start = row * gate_cols;
        let sf_vals = &gate_f32[start..start + gate_cols];
        let blob_vals = client.diagnostic_weight_row("L0.gate_proj", row, gate_cols).unwrap();
        let sim = cosine_similarity(sf_vals, &blob_vals);
        eprintln!("[DIAG-022] L0.gate_proj row {}: sim={:.6}", row, sim);
        assert!((sim - 1.0).abs() < 0.0001, "gate_proj row {} mismatch: sim={:.6}", row, sim);
    }

    // L5.q_proj (middle layer)
    let l5_view = tensors.tensor("model.layers.5.self_attn.q_proj.weight").unwrap();
    let l5_raw = l5_view.data();
    let l5_f32: Vec<f32> = l5_raw.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    for row in [0usize, 100, 575] {
        let start = row * cols;
        let sf_vals = &l5_f32[start..start + cols];
        let blob_vals = client.diagnostic_weight_row("L5.q_proj", row, cols).unwrap();
        let sim = cosine_similarity(sf_vals, &blob_vals);
        eprintln!("[DIAG-022] L5.q_proj row {}: sim={:.6}", row, sim);
        assert!((sim - 1.0).abs() < 0.0001, "L5.q_proj row {} mismatch: sim={:.6}", row, sim);
    }

    eprintln!("[DIAG-022] All deep row checks PASSED");
}

/// DIAG-023: Verify weight offsets for all layers match expected sequential layout.
#[test]
fn diagnostic_weight_offset_all_layers() {
    let _ = env_logger::builder().is_test(true).try_init();
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2-135M");
    let offsets = client.diagnostic_weight_offsets().expect("weight offsets");

    let norm_sz: usize = 576 * 4;
    let q_sz: usize = 576 * 576 * 4;
    let k_sz: usize = 192 * 576 * 4;
    let v_sz: usize = 192 * 576 * 4;
    let o_sz: usize = 576 * 576 * 4;
    let pnorm_sz: usize = 576 * 4;
    let gate_sz: usize = 1536 * 576 * 4;
    let up_sz: usize = 1536 * 576 * 4;
    let down_sz: usize = 576 * 1536 * 4;
    let per_layer = norm_sz + q_sz + k_sz + v_sz + o_sz + pnorm_sz + gate_sz + up_sz + down_sz;
    eprintln!("[DIAG-023] Per-layer weight bytes: {}", per_layer);

    let cumulative = [0, norm_sz, norm_sz+q_sz, norm_sz+q_sz+k_sz,
        norm_sz+q_sz+k_sz+v_sz, norm_sz+q_sz+k_sz+v_sz+o_sz,
        norm_sz+q_sz+k_sz+v_sz+o_sz+pnorm_sz,
        norm_sz+q_sz+k_sz+v_sz+o_sz+pnorm_sz+gate_sz,
        norm_sz+q_sz+k_sz+v_sz+o_sz+pnorm_sz+gate_sz+up_sz];

    let suffixes = ["input_norm", "q_proj", "k_proj", "v_proj", "o_proj",
        "post_attn_norm", "gate_proj", "up_proj", "down_proj"];

    let mut mismatches = 0usize;
    for layer in 0usize..30 {
        let base = layer * per_layer;
        for (si, suffix) in suffixes.iter().enumerate() {
            let name = format!("L{}_{}", layer, suffix);
            let expected = base + cumulative[si];
            match offsets.iter().find(|(n, _, _)| n == &name) {
                Some((_, off, _)) if *off != expected => {
                    if mismatches < 10 {
                        eprintln!("[DIAG-023] MISMATCH {} offset={} expected={}", name, off, expected);
                    }
                    mismatches += 1;
                }
                None => {
                    if mismatches < 10 {
                        eprintln!("[DIAG-023] MISSING {}", name);
                    }
                    mismatches += 1;
                }
                _ => {}
            }
        }
    }
    // Check final_norm
    let expected_final = 30 * per_layer;
    match offsets.iter().find(|(n, _, _)| n == "final_norm") {
        Some((_, off, _)) if *off != expected_final => {
            eprintln!("[DIAG-023] MISMATCH final_norm offset={} expected={}", off, expected_final);
            mismatches += 1;
        }
        Some(_) => eprintln!("[DIAG-023] final_norm offset OK"),
        None => { eprintln!("[DIAG-023] MISSING final_norm"); mismatches += 1; }
    }
    eprintln!("[DIAG-023] Total weight entries: {}, mismatches: {}", offsets.len(), mismatches);
    if mismatches == 0 {
        eprintln!("[DIAG-023] All weight offsets correct");
    }
}

/// DIAG-028: Compare gllm hidden states at each layer vs golden to bisect deviation.
#[test]
#[ignore]
fn diagnostic_hidden_states_per_layer() {
    let _ = env_logger::builder().is_test(true).try_init();
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let client = Client::new_chat(MODEL).expect("Failed to load SmolLM2");

    // Run prefill and get scratchpad with all activations
    let sp = client.diagnostic_prefill_scratchpad(golden::INPUT_IDS)
        .expect("Failed to get scratchpad");

    // Golden hidden states: hidden_layer_0 (embedding) to hidden_layer_30 (final)
    for layer in [0usize, 1, 2, 5, 15, 29, 30] {
        if let Some(golden_h) = load_golden_hidden(layer) {
            eprintln!("[DIAG-028] L{} golden norm={:.2} mean={:.6}", layer,
                golden_h.iter().map(|x| x*x).sum::<f32>().sqrt(),
                golden_h.iter().sum::<f32>() / golden_h.len() as f32);
        }
    }
    eprintln!("[DIAG-028] scratchpad size={} bytes, logits_off={}", sp.data.len(), sp.logits_offset);
}
