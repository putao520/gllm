//! KV cache layout diagnostic: Q5_K_M vs Q6_K
//!
//! Verifies the "KV cache dtype dual-layer trap" (kv-cache-dtype-dual-layer.md) hypothesis:
//! - Buffer layer: compute_dtype → elem_bytes → kv_row_stride → kv_cache_bytes
//! - JIT layer: K tensor TensorMeta.dtype → elem_bytes → kv_row_stride → MemCopy bytes
//!
//! If these two layers disagree (buffer F32 vs JIT BF16 or vice versa), MemCopy writes
//! more bytes than the buffer row can hold → cross-layer/cross-slot corruption → N≥2 crash.
//!
//! Q5_K_M symptom: N=1 cos 0.9998 perfect, N=2 cos 0.78 crash, N=4 all zeros.
//! Q6_K: N=2 cos 0.9999 perfect (control).
//!
//! This test dumps the actual runtime values and compares Q5_K_M vs Q6_K.
#![cfg(test)]
use gllm::{Client, ModelKind};

/// Qwen3-0.6B geometry (from config.json, verified in domain-knowledge)
const QWEN3_06B_NUM_LAYERS: usize = 28;
const QWEN3_06B_NUM_KV_HEADS: usize = 4;
const QWEN3_06B_HEAD_DIM: usize = 128;
// max_position_embeddings from config.json
const QWEN3_06B_MAX_SEQ: usize = 40960;

fn dump_kv_cache_layout(label: &str, client: &Client, tokens: &[u32]) {
    let bar = "=".repeat(60);
    eprintln!("\n{}", bar);
    eprintln!("=== KV CACHE LAYOUT: {} ===", label);
    eprintln!("{}", bar);

    // 1. Get compute_dtype from diagnostic scratchpad (buffer layer source)
    let sp = match client.diagnostic_prefill_scratchpad(tokens) {
        Some(sp) => sp,
        None => {
            eprintln!("[ERROR] diagnostic_prefill_scratchpad returned None");
            return;
        }
    };

    let compute_dtype = sp.compute_dtype;
    let compute_elem_bytes = compute_dtype.size_bytes();
    eprintln!("\n[Buffer Layer - from DiagnosticScratchpad.compute_dtype]");
    eprintln!("  compute_dtype = {:?}", compute_dtype);
    eprintln!("  compute_dtype.size_bytes() = {}", compute_elem_bytes);

    // 2. KV cache geometry (from Qwen3-0.6B config.json)
    let num_kv_heads = QWEN3_06B_NUM_KV_HEADS;
    let head_dim = QWEN3_06B_HEAD_DIM;
    let num_layers = QWEN3_06B_NUM_LAYERS;
    let max_seq = QWEN3_06B_MAX_SEQ;
    eprintln!("\n[KV Cache Geometry (Qwen3-0.6B config.json)]");
    eprintln!("  num_kv_heads = {}", num_kv_heads);
    eprintln!("  head_dim = {}", head_dim);
    eprintln!("  num_layers = {}", num_layers);
    eprintln!("  max_seq = {}", max_seq);

    // 3. Buffer layer KV cache stride (abi_types.inc.rs)
    // kv_row_stride = num_kv_heads * head_dim * elem_bytes (compute_dtype)
    let buf_kv_row_stride = num_kv_heads * head_dim * compute_elem_bytes;
    // kv_layer_stride = 2 * max_seq * kv_row_stride
    let buf_kv_layer_stride = 2 * max_seq * buf_kv_row_stride;
    // kv_cache_bytes = num_layers * kv_layer_stride
    let buf_kv_cache_bytes = num_layers * buf_kv_layer_stride;
    eprintln!("\n[Buffer Layer KV Cache Stride (abi_types.inc.rs)]");
    eprintln!("  kv_row_stride = {} * {} * {} = {} bytes/row",
        num_kv_heads, head_dim, compute_elem_bytes, buf_kv_row_stride);
    eprintln!("  kv_layer_stride = 2 * {} * {} = {} bytes/layer",
        max_seq, buf_kv_row_stride, buf_kv_layer_stride);
    eprintln!("  kv_cache_bytes = {} * {} = {} bytes (total buffer)",
        num_layers, buf_kv_layer_stride, buf_kv_cache_bytes);

    // 4. JIT layer KV cache stride (lower_op.inc.rs:1507-1549)
    // dtype = graph.tensor(k_tid).dtype.to_quant_precision()
    // k_tid = op.inputs[1] (K output tensor)
    // K output tensor dtype = act_dt = DType::F32 (build_graph.inc.rs:599, hardcoded)
    let jit_k_dtype = gllm_kernels::types::DType::F32; // act_dt, hardcoded in build_graph
    let jit_elem_bytes = jit_k_dtype.size_bytes();
    let jit_kv_row_stride = num_kv_heads * head_dim * jit_elem_bytes;
    let jit_kv_layer_stride = 2 * max_seq * jit_kv_row_stride;
    let jit_kv_cache_bytes = num_layers * jit_kv_layer_stride;
    eprintln!("\n[JIT Layer KV Cache Stride (lower_op.inc.rs:1507-1549)]");
    eprintln!("  K tensor dtype (act_dt, build_graph.inc.rs:94/599) = {:?}", jit_k_dtype);
    eprintln!("  jit_elem_bytes = {}", jit_elem_bytes);
    eprintln!("  kv_row_stride = {} * {} * {} = {} bytes/row",
        num_kv_heads, head_dim, jit_elem_bytes, jit_kv_row_stride);
    eprintln!("  kv_layer_stride = 2 * {} * {} = {} bytes/layer",
        max_seq, jit_kv_row_stride, jit_kv_layer_stride);
    eprintln!("  kv_cache_bytes = {} * {} = {} bytes (JIT expects)",
        num_layers, jit_kv_layer_stride, jit_kv_cache_bytes);

    // 5. CRITICAL: Check for dual-layer mismatch
    eprintln!("\n[CRITICAL: Dual-Layer Consistency Check]");
    let row_stride_match = buf_kv_row_stride == jit_kv_row_stride;
    let layer_stride_match = buf_kv_layer_stride == jit_kv_layer_stride;
    let total_match = buf_kv_cache_bytes == jit_kv_cache_bytes;
    eprintln!("  Buffer kv_row_stride = {} vs JIT kv_row_stride = {} → {}",
        buf_kv_row_stride, jit_kv_row_stride,
        if row_stride_match { "MATCH ✓" } else { "MISMATCH ✗ ← OVERFLOW/CORRUPTION" });
    eprintln!("  Buffer kv_layer_stride = {} vs JIT kv_layer_stride = {} → {}",
        buf_kv_layer_stride, jit_kv_layer_stride,
        if layer_stride_match { "MATCH ✓" } else { "MISMATCH ✗ ← CROSS-LAYER CORRUPTION" });
    eprintln!("  Buffer kv_cache_bytes = {} vs JIT kv_cache_bytes = {} → {}",
        buf_kv_cache_bytes, jit_kv_cache_bytes,
        if total_match { "MATCH ✓" } else { "MISMATCH ✗ ← BUFFER TOO SMALL" });

    if row_stride_match && layer_stride_match && total_match {
        eprintln!("\n  >>> KV cache layout SYMMETRIC and CORRECT for {} <<<", label);
    } else {
        eprintln!("\n  >>> KV cache layout MISMATCH for {} — DUAL-LAYER TRAP ACTIVE <<<", label);
    }

    // 6. Weight dtype inspection (to understand storage_dtype derivation)
    eprintln!("\n[Weight Dtype Inspection (weight_dtypes map)]");
    if let Some(offsets) = client.diagnostic_weight_offsets() {
        let mut dtype_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (_, _, dt) in &offsets {
            *dtype_counts.entry(format!("{:?}", dt)).or_insert(0) += 1;
        }
        eprintln!("  Total named tensors: {}", offsets.len());
        eprintln!("  Dtype distribution:");
        for (dt, count) in dtype_counts.iter() {
            eprintln!("    {}: {} tensors", dt, count);
        }
        // Check specific K/V proj tensors
        for name in &["L0.q_proj", "L0.k_proj", "L0.v_proj", "L0.o_proj",
                       "L0.down_proj", "L0.gate_proj", "L0.up_proj"] {
            if let Some((_, off, dt)) = offsets.iter().find(|(n, _, _)| n == name) {
                eprintln!("  {} → offset={}, dtype={:?}", name, off, dt);
            }
        }
    } else {
        eprintln!("  [WARN] diagnostic_weight_offsets returned None");
    }

    // 7. Named intermediate tensors (K/V output in scratchpad)
    eprintln!("\n[Named Intermediate Tensors (JIT named_offsets)]");
    if let Some(offsets) = client.diagnostic_weight_offsets() {
        // named_offsets from scratchpad includes intermediate tensors too
        let kv_tensors: Vec<_> = offsets.iter()
            .filter(|(n, _, _)| n.contains("layer.k") || n.contains("layer.v") || n.contains("layer.q"))
            .collect();
        if kv_tensors.is_empty() {
            eprintln!("  [NOTE] No layer.k/v/q intermediate tensors in named_offsets");
            eprintln!("  (K/V outputs are in scratchpad via buffer_alloc, may not be in weight_offsets)");
        } else {
            for (n, off, dt) in kv_tensors {
                eprintln!("  {} → offset={}, dtype={:?}", n, off, dt);
            }
        }
    }
}

#[test]
#[ignore]
fn diag_kv_cache_layout_q5km_vs_q6k() {
    let hb = "#".repeat(60);
    eprintln!("\n{}", hb);
    eprintln!("# KV CACHE LAYOUT DIAGNOSTIC: Q5_K_M vs Q6_K");
    eprintln!("# Hypothesis: KV cache dtype dual-layer trap (kv-cache-dtype-dual-layer.md)");
    eprintln!("# If Q5_K_M has buffer/JIT stride mismatch that Q6_K doesn't => root cause found");
    eprintln!("{}", hb);

    let prompt = "The capital of France is";

    // ── Q5_K_M ──
    let bar60 = "=".repeat(60);
    eprintln!("\n\n{}", bar60);
    eprintln!(">>> Building Q5_K_M client...");
    let q5_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m")
        .build()
        .expect("Q5_K_M client");
    let q5_tokens = q5_client.encode(prompt).expect("Q5_K_M encode");
    eprintln!("Q5_K_M prompt tokens: {:?}", q5_tokens);
    dump_kv_cache_layout("Q5_K_M", &q5_client, &q5_tokens);
    drop(q5_client);

    // ── Q6_K ──
    eprintln!("\n\n{}", bar60);
    eprintln!(">>> Building Q6_K client...");
    let q6_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q6_k")
        .build()
        .expect("Q6_K client");
    let q6_tokens = q6_client.encode(prompt).expect("Q6_K encode");
    eprintln!("Q6_K prompt tokens: {:?}", q6_tokens);
    dump_kv_cache_layout("Q6_K", &q6_client, &q6_tokens);
    drop(q6_client);

    eprintln!("\n\n{}", bar60);
    eprintln!("=== DIAGNOSTIC COMPLETE ===");
    eprintln!("Compare the Buffer vs JIT stride values above for both models.");
    eprintln!("If both show MATCH => KV cache direction EXCLUDED.");
    eprintln!("If Q5_K_M shows MISMATCH => root cause found.");
    eprintln!("{}", bar60);
}
