//! Q6_K 模型 E2E — 决定性二分: Q6KDecodeStep native call 在 layer-loop 多次执行是否有副作用.
//! Q6_K 模型全层 Q6_K (layer-loop 内全用 Q6KDecodeStep). 若 E2E 正常出 Paris → Q6KDecodeStep reentrancy OK;
//! 若崩 → Q6KDecodeStep native call lowering 在 layer-loop 有副作用 (Q4_0 模型 lm_head 单次不触发).
#![cfg(test)]
#![allow(dead_code)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_q6k_e2e_argmax() {
    eprintln!("\n=== Q6_K 模型 E2E (Q6KDecodeStep layer-loop reentrancy 决定性测试) ===");
    use std::io::Write; let _ = std::io::stderr().flush();
    let prompt = "The capital of France is";

    // Q6_K 1层 vs BF16 1层
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let q6 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q6_k").build().expect("Q6_K client");
    let qt = q6.encode(prompt).expect("encode");
    let qsp = q6.diagnostic_prefill_scratchpad(&qt).expect("sp");
    let qv = qsp.vocab_size;
    let qlg = qsp.read_dtype_aware(qsp.logits_offset, qv);
    let qam = qlg.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
    let qmx = qlg.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[Q6_K N=1] argmax={} |max|={:.4}", qam, qmx);
    drop(q6);

    let bf = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("bf16").build().expect("BF16");
    let bt = bf.encode(prompt).expect("e");
    let bsp = bf.diagnostic_prefill_scratchpad(&bt).expect("sp");
    let bv = bsp.vocab_size;
    let blg = bsp.read_dtype_aware(bsp.logits_offset, bv);
    let bam = blg.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
    eprintln!("[BF16 N=1] argmax={}", bam);
    drop(bf);
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    // cosine
    let dot: f64 = qlg.iter().zip(blg.iter()).map(|(a,b)| (*a as f64)*(*b as f64)).sum();
    let na: f64 = qlg.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = blg.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if na>0.0 && nb>0.0 { dot/(na*nb) } else { 0.0 };
    eprintln!("[Q6_K N=1 vs BF16] cosine={:.6}", cos);

    // 关键: Q6_K 模型全层 Q6_K, layer-loop 内全用 Q6KDecodeStep. N=1 只 1次 (layer0).
    // 若 N=1 cos 高, 需测 N=2 确认 reentrancy.
    if cos > 0.99 {
        eprintln!("[Q6_K N=1] ✓ cos 高 — 测 N=2 reentrancy...");
        std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
        let q6b = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q6_k").build().expect("Q6_K2");
        let qt2 = q6b.encode(prompt).expect("e");
        let qsp2 = q6b.diagnostic_prefill_scratchpad(&qt2).expect("sp");
        let qlg2 = qsp2.read_dtype_aware(qsp2.logits_offset, qsp2.vocab_size);
        let qam2 = qlg2.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
        let qmx2 = qlg2.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
        std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
        let bf2 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("bf16").build().expect("BF2");
        let bt2 = bf2.encode(prompt).expect("e");
        let bsp2 = bf2.diagnostic_prefill_scratchpad(&bt2).expect("sp");
        let blg2 = bsp2.read_dtype_aware(bsp2.logits_offset, bsp2.vocab_size);
        let bam2 = blg2.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
        let dot2: f64 = qlg2.iter().zip(blg2.iter()).map(|(a,b)| (*a as f64)*(*b as f64)).sum();
        let na2: f64 = qlg2.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let nb2: f64 = blg2.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let cos2 = if na2>0.0 && nb2>0.0 { dot2/(na2*nb2) } else { 0.0 };
        eprintln!("[Q6_K N=2] argmax={} |max|={:.4} vs BF16 argmax={} cosine={:.6} {}",
            qam2, qmx2, bam2, cos2, if cos2 > 0.99 { "✓ reentrancy OK" } else { "✗ reentrancy 崩 (Q6KDecodeStep native call 有副作用!)" });
        std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    } else {
        eprintln!("[Q6_K N=1] ✗ cos 低 — Q6KDecodeStep 即便单次也错 (但 Q4_0 lm_head Q6_K 单次对, 矛盾)");
    }
}
