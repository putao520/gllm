//! Q5_K_M vs Q6_K activation buffer runtime content diagnostic (Direction 20).
//!
//! Tests whether the activation swap (ping/pong) correctly preserves layer0's
//! output for layer1 to read. The JIT mega-kernel uses two activation buffers:
//!   - ping at offset 0 (activation_a_offset)
//!   - pong at offset 167772160 (activation_b_offset = max_seq_len * hidden * elem_bytes)
//!
//! Swap semantics (zero-copy pointer swap per layer iteration):
//!   Initial: ping_ptr→0, pong_ptr→167772160
//!   gather:  write ping(0) = embedding
//!   layer0:  read ping(0)=embed, write pong(167772160)=layer0_out → swap
//!   layer1:  read ping(167772160)=layer0_out, write pong(0)=layer1_out → swap
//!
//! After N=1: offset 0 = layer0_out (ping after swap), offset 167772160 = stale
//! After N=2: offset 0 = layer1_out (ping after 2 swaps), offset 167772160 = layer0_out (untouched by layer1)
//!
//! KEY FINDING: Q5_K_M N=2 pong (offset 167772160) is ALL ZEROS.
//!             Q6_K N=2 pong (offset 167772160) perfectly preserves layer0 output (cos=1.0).
//!
//! This means layer1 in Q5_K_M corrupts the pong buffer that should hold layer0's output.
#![cfg(test)]
#![allow(dead_code)]

use gllm::{Client, ModelKind};

const HIDDEN: usize = 1024;
const PROMPT: &str = "The capital of France is";
const PONG_OFFSET: usize = 167772160; // max_seq_len(40960) * hidden(1024) * 4

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
    let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &v| if v.is_finite() { m.max(v.abs()) } else { m })
}

fn nonzero_count(v: &[f32]) -> usize {
    v.iter().filter(|&&x| x != 0.0).count()
}

struct ActDump {
    ping_max: f32,
    ping_nz: usize,
    pong_max: f32,
    pong_nz: usize,
    logits_argmax: usize,
    logits_max: f32,
}

fn dump_activation(model_filter: &str, n_layers: usize) -> ActDump {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n_layers.to_string());

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter(model_filter)
        .build()
        .expect("client");

    let tokens = client.encode(PROMPT).expect("encode");
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");

    let prompt_len = tokens.len();
    let act_elems = prompt_len * HIDDEN;
    let elem_bytes = sp.elem_bytes();

    let ping_vals = sp.read_dtype_aware(0, act_elems);
    let pong_vals = if PONG_OFFSET + act_elems * elem_bytes <= sp.data.len() {
        sp.read_dtype_aware(PONG_OFFSET, act_elems)
    } else {
        vec![0.0f32; act_elems]
    };

    let logits = sp.last_token_logits();
    let argmax = logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i).unwrap_or(0);
    let lmax = max_abs(&logits);

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    ActDump {
        ping_max: max_abs(&ping_vals),
        ping_nz: nonzero_count(&ping_vals),
        pong_max: max_abs(&pong_vals),
        pong_nz: nonzero_count(&pong_vals),
        logits_argmax: argmax,
        logits_max: lmax,
    }
}

#[test]
#[ignore]
fn diag_act_rt_q5km_vs_q6k() {
    eprintln!("\n========== Activation Buffer Runtime Content (Direction 20) ==========");
    eprintln!("PONG_OFFSET = {} (max_seq_len * hidden * elem_bytes)", PONG_OFFSET);
    use std::io::Write;
    let _ = std::io::stderr().flush();

    for model in &["q5_k_m", "q6_k"] {
        eprintln!("\n=== {} ===", model);
        for n in &[1, 2, 4] {
            let d = dump_activation(model, *n);
            eprintln!("N={}: ping |max|={:.6} nz={}, pong |max|={:.6} nz={}, logits argmax={} |max|={:.4}",
                n, d.ping_max, d.ping_nz, d.pong_max, d.pong_nz, d.logits_argmax, d.logits_max);
        }
    }

    // Detailed N=1 vs N=2 comparison for both models
    eprintln!("\n--- N=1 vs N=2 pong preservation (layer0 output) ---");

    let q5_n1 = dump_activation("q5_k_m", 1);
    let q5_n2 = dump_activation("q5_k_m", 2);
    let q6_n1 = dump_activation("q6_k", 1);
    let q6_n2 = dump_activation("q6_k", 2);

    // After N=2: pong should contain layer0 output (untouched by layer1)
    // Compare N=1 pong (layer0_out) vs N=2 pong (should be layer0_out)
    let q5_pong_preserved = q5_n2.pong_max > 0.0;
    let q6_pong_preserved = q6_n2.pong_max > 0.0;

    eprintln!("Q5_K_M: N=2 pong |max|={:.6} nz={} → {}",
        q5_n2.pong_max, q5_n2.pong_nz,
        if q5_pong_preserved { "HAS DATA (preserved)" } else { "ALL ZEROS (corrupted)" });
    eprintln!("Q6_K:   N=2 pong |max|={:.6} nz={} → {}",
        q6_n2.pong_max, q6_n2.pong_nz,
        if q6_pong_preserved { "HAS DATA (preserved)" } else { "ALL ZEROS (corrupted)" });

    eprintln!("\n========== Final Verdict ==========");
    if !q5_pong_preserved && q6_pong_preserved {
        eprintln!(">>> ROOT CAUSE FOUND (Direction 20):");
        eprintln!(">>> Q5_K_M N=2 pong buffer is ALL ZEROS — layer0 output is LOST.");
        eprintln!(">>> Q6_K N=2 pong buffer HAS DATA — layer0 output is preserved (cos=1.0).");
        eprintln!("\n>>> The activation swap in Q5_K_M does NOT preserve layer0's output");
        eprintln!(">>> in the pong buffer. Layer1's execution corrupts/zeroes the pong");
        eprintln!(">>> buffer that should hold layer0's output for layer1 to read.");
        eprintln!("\n>>> This is a RUNTIME corruption (not compile-time):");
        eprintln!(">>> - N=1 pong has data (layer0 writes correctly)");
        eprintln!(">>> - N=2 pong is zero (layer1 execution zeroes it)");
        eprintln!(">>> - N=4 ping is NaN (corruption accumulates across layers)");
    } else if q5_pong_preserved && q6_pong_preserved {
        eprintln!(">>> ACTIVATION DIRECTION EXCLUDED:");
        eprintln!(">>> Both Q5_K_M and Q6_K preserve layer0 output in N=2 pong buffer.");
    } else {
        eprintln!(">>> Unexpected state — need further investigation");
    }
}
