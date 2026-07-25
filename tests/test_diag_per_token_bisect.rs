#![cfg(test)]
//! 方向65 方法B: 逐 token logits 二分（可靠，不依赖 capture 机制）。
//! 思路: prompt 逐位增长, 每次只看"最后一位"的 logits vs BF16。
//!   - pos=1: prompt=[tok0] → logits(tok0 的下一个预测) vs BF16
//!   - pos=2: prompt=[tok0,tok1] → logits(tok1 的预测) vs BF16
//!   - ...
//! 首个 cos 下降的 pos = 发散起始 token。
//! F6 锚定: QuantGemm 在 seq=1 与 seq=5 同码(Gemv), 若 pos=1 就发散 → gen-loop 第 0 迭代结构差异;
//! 若 pos≥2 才发散 → KV/RoPE 累积 (token0 不发散, 验证 F6)。
//! 优化: 每个模型只 build 一次 Client (复用 JIT 编译), 多次调 diagnostic_prefill_logits。
use gllm::{Client, ModelKind};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
    let na: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

fn maxabs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
}

fn build_client(filter: &str) -> Client {
    Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter(filter)
        .build()
        .unwrap_or_else(|e| panic!("build {}: {:?}", filter, e))
}

#[test]
#[ignore]
fn diag_per_token_bisect() {
    let prompt = "The capital of France is";
    let n = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_else(|_| "3".into());
    // 必须在 build Client 前设, 因 graph 构建时读 GLLM_TRUNCATE_LAYERS。
    std::env::set_var("GLLM_TRUNCATE_LAYERS", &n);

    // 用 Q5_K_M 编码取 token 序列 (BF16 同 tokenizer, 序列一致)
    let q = build_client("q5_k_m");
    let tokens = q.encode(prompt).expect("encode");
    eprintln!("=== Q5_K_M vs BF16 per-token logits bisect (N={}, prompt={:?}) ===", n, prompt);
    eprintln!("tokens (len={}): {:?}", tokens.len(), tokens);

    // 每个模型 build 一次, 复用 JIT 编译结果
    let q5 = build_client("q5_k_m");
    let bf = build_client("bf16");

    // 逐 token 增长, 每次只看最后一位 logits
    for len in 1..=tokens.len() {
        let sub = &tokens[..len];
        let q5v = q5.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("q5 logits len={}", len));
        let bfv = bf.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("bf logits len={}", len));
        let cos = cosine(&q5v, &bfv);
        let mq = maxabs(&q5v);
        let mb = maxabs(&bfv);
        let flag = if cos > 0.99 { "✓" } else if cos > 0.9 { "⚠️" } else { "✗" };
        eprintln!("  pos[{}] (len={}): cos={:.6} |q5|={:.2} |bf|={:.2} {}", len, len, cos, mq, mb, flag);
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}

#[test]
#[ignore]
fn diag_cross_prompt_pos2() {
    // P8 验证: 不同 prompt 同 seq=5, 看 pos=2 是否都降。
    // 若都降 → 系统性 (P2 KV 地址); 若有的不降 → token 依赖 (P8 softmax 放大)。
    let n = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_else(|_| "3".into());
    std::env::set_var("GLLM_TRUNCATE_LAYERS", &n);
    let prompts = [
        "The capital of France is",
        "Hello world how are you",
        "Once upon a time there",
        "The quick brown fox jumps",
        "I think that the best way",
    ];
    let q5 = build_client("q5_k_m");
    let bf = build_client("bf16");
    eprintln!("=== cross-prompt pos=2 (Q5_K_M vs BF16, N={}) ===", n);
    for p in &prompts {
        let toks = q5.encode(p).expect("encode");
        if toks.len() < 2 {
            eprintln!("  prompt {:?}: only {} tokens, skip", p, toks.len());
            continue;
        }
        let mut row = String::new();
        for len in 1..=toks.len().min(5) {
            let sub = &toks[..len];
            let q5v = q5.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("q5 len={}", len));
            let bfv = bf.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("bf len={}", len));
            let cos = cosine(&q5v, &bfv);
            row.push_str(&format!(" p{}={:.4}", len, cos));
        }
        eprintln!("  {:?} ({} toks):{}", p, toks.len(), row);
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}

#[test]
#[ignore]
fn diag_quant_compare() {
    // P8 验证: Q4_K_M(4-bit) vs Q5_K_M(5-bit) vs Q6_K(6-bit) vs Q8_0(8-bit) 同 prompt 发散程度。
    // 若低精度发散更严重 → 精度边界 (P8 确诊); 若各精度差不多 → 非纯精度。
    let n = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_else(|_| "3".into());
    std::env::set_var("GLLM_TRUNCATE_LAYERS", &n);
    let bf = build_client("bf16");
    let prompt = "The capital of France is";
    let toks = bf.encode(prompt).expect("encode");
    eprintln!("=== quant precision compare (N={}, prompt={:?}) ===", n, prompt);
    for q in ["q4_k_m", "q5_k_m", "q6_k", "q8_0"] {
        let c = build_client(q);
        let mut row = String::new();
        for len in 1..=toks.len() {
            let sub = &toks[..len];
            let qv = c.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("{} len={}", q, len));
            let bv = bf.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("bf len={}", len));
            let cos = cosine(&qv, &bv);
            row.push_str(&format!(" p{}={:.4}", len, cos));
        }
        eprintln!("  {:8}:{}", q, row);
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
