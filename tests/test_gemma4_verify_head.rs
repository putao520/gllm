#![cfg(test)]
//! Gemma4 full_head_dim 修复验证：对比 gllm argmax vs llama.cpp golden (9081=" catal")
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn gemma4_verify_full_head_dim_fix() {
    // 用 unsloth/gemma-4-E2B-it-GGUF Q8_0 (HF cache 结构已建)
    let client = Client::builder()
        .model("unsloth/gemma-4-E2B-it-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("Q8_0")
        .build()
        .unwrap_or_else(|e| panic!("load gemma4: {:?}", e));

    let prompt = "The capital of France is";
    let toks = client.encode(prompt).expect("encode");
    eprintln!("gllm tokens (len={}): {:?}", toks.len(), toks);

    // golden tokens: 2 818 5279 529 7001 563 (BOS The capital of France is)
    eprintln!("golden tokens: [2, 818, 5279, 529, 7001, 563]");

    let logits = client.diagnostic_prefill_logits(&toks).expect("prefill");
    eprintln!("gllm logits vocab={}", logits.len());

    // argmax
    let mut top: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("gllm top5: {:?}", &top[..5]);
    eprintln!("gllm argmax={} logit={:.4}", top[0].0, top[0].1);

    // golden argmax=9081=" catal" (llama.cpp dump)
    let golden_argmax = 9081usize;
    eprintln!("golden argmax={} (catal)", golden_argmax);
    eprintln!("golden logit[pos5]={:.4}", 19.44f32);

    // 之前乱码 "eisenijima" argmax 是其他值
    if top[0].0 == golden_argmax {
        eprintln!("✅ MATCH! full_head_dim fix 解决乱码");
    } else {
        eprintln!("❌ MISMATCH: gllm argmax={} vs golden={} (乱码可能仍存在)", top[0].0, golden_argmax);
    }
    eprintln!("GEMMA4_VERIFY_DONE argmax={} golden={}", top[0].0, golden_argmax);
}
