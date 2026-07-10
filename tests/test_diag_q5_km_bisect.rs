//! Q5_K_M 诊断: 1层截断 argmax vs BF16 (BCE-20260710-Q5_K-HIGHBITS).
//! 方法: GLLM_TRUNCATE_LAYERS=1, Q5_K_M vs BF16 1层 logits argmax 应一致 (Paris).
#![cfg(test)]
#![allow(dead_code)]

use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_q5_km_1layer_argmax() {
    eprintln!("\n=== Q5_K_M 截断 argmax vs BF16 (BCE-20260710-Q5_K-HIGHBITS) ===");
    use std::io::Write; let _ = std::io::stderr().flush();

    let prompt = "The capital of France is";
    // 允许 env 覆盖截断层数 (默认 1, 可测 2/4/8/14 定位多层层中哪层开始错)
    let n_layers = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_else(|_| "1".into());
    std::env::set_var("GLLM_TRUNCATE_LAYERS", &n_layers);
    eprintln!("[DIAG] GLLM_TRUNCATE_LAYERS={}", n_layers);

    // Run A: Q5_K_M 1层
    let q5_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m")
        .build()
        .expect("Q5_K_M client");
    let q5_tokens = q5_client.encode(prompt).expect("encode");
    let q5_sp = q5_client.diagnostic_prefill_scratchpad(&q5_tokens).expect("q5 scratchpad");
    let q5_vocab = q5_sp.vocab_size;
    let q5_logits = q5_sp.read_dtype_aware(q5_sp.logits_offset, q5_vocab);
    let q5_argmax = q5_logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i,_)| i).unwrap_or(0);
    let q5_max = q5_logits.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[Q5_K_M 1层] argmax={} (BF16 应 Paris), logits|max|={:.4}, vocab={}", q5_argmax, q5_max, q5_vocab);
    drop(q5_client);

    // Run B: BF16 1层
    let bf_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("bf16")
        .build()
        .expect("BF16 client");
    let bf_tokens = bf_client.encode(prompt).expect("encode");
    let bf_sp = bf_client.diagnostic_prefill_scratchpad(&bf_tokens).expect("bf16 scratchpad");
    let bf_vocab = bf_sp.vocab_size;
    let bf_logits = bf_sp.read_dtype_aware(bf_sp.logits_offset, bf_vocab);
    let bf_argmax = bf_logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i,_)| i).unwrap_or(0);
    let bf_max = bf_logits.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[BF16 1层] argmax={} (Paris), logits|max|={:.4}, vocab={}", bf_argmax, bf_max, bf_vocab);

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    // 完整 logits 对照 (architect: argmax 巧合不足以证明, 需 cosine + topk + max_abs_diff)
    let dot: f64 = q5_logits.iter().zip(bf_logits.iter())
        .map(|(a,b)| (*a as f64) * (*b as f64)).sum();
    let na: f64 = q5_logits.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = bf_logits.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
    let max_abs_diff: f64 = q5_logits.iter().zip(bf_logits.iter())
        .map(|(a,b)| (*a as f64 - *b as f64).abs()).fold(0.0f64, f64::max);
    // top-10 overlap
    let mut q5_idx: Vec<usize> = (0..q5_vocab).collect();
    q5_idx.sort_by(|&i,&j| bf_logits[j].partial_cmp(&bf_logits[i]).unwrap_or(std::cmp::Ordering::Equal));
    let q5_top10: std::collections::HashSet<usize> = q5_idx.iter().take(10).copied().collect();
    let mut bf_idx: Vec<usize> = (0..bf_vocab).collect();
    bf_idx.sort_by(|&i,&j| bf_logits[j].partial_cmp(&bf_logits[i]).unwrap_or(std::cmp::Ordering::Equal));
    let bf_top10: std::collections::HashSet<usize> = bf_idx.iter().take(10).copied().collect();
    let topk_overlap = q5_top10.intersection(&bf_top10).count();
    eprintln!("[DIAG-FULL] cosine(Q5_K,BF16)={:.4} max_abs_diff={:.4} top10_overlap={}/10",
        cosine, max_abs_diff, topk_overlap);

    eprintln!("[DIAG] Q5_K_M argmax={} vs BF16 argmax={} — {}",
        q5_argmax, bf_argmax,
        if q5_argmax == bf_argmax { "MATCH ✓ (Q5_K decode 正确)" } else { "MISMATCH ✗ (Q5_K decode 仍有 bug)" });
}
