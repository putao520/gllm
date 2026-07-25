#![cfg(test)]
//! 对照 gllm vs llama.cpp 逐 token 完整 logits。
//! 对 prompt 每个前缀长度 len，跑 gllm diagnostic_prefill_logits 取最后 token 完整 logits 向量，
//! 写到文件，再和 llama.cpp dump-logits 的输出对照。
//! 语义: gllm pos[len] = 喂 prompt[..len] 后最后 token 的 logits (预测 token[len])。
//!       llama pos[len-1] = 喂 token[0..len] (含BOS) 后 token[len-1] 的 logits (预测 token[len])。
//! 因 llama 加 BOS, gllm 不加 BOS, 位置需对齐 (gllm pos[len] ≈ llama pos[len-1] 当 gllm 第一个 token == llama token[1])。
use gllm::{Client, ModelKind};
use std::io::Write;

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
fn diag_dump_gllm_logits() {
    let prompt = "The capital of France is";
    // N 可通过 GLLM_TRUNCATE_LAYERS 控制（不设=全 28 层）
    let n_layers = std::env::var("GLLM_TRUNCATE_LAYERS").unwrap_or_default();
    if !n_layers.is_empty() { std::env::set_var("GLLM_TRUNCATE_LAYERS", &n_layers); }
    let q = build_client("q5_k_m");
    let toks = q.encode(prompt).expect("encode");
    eprintln!("gllm tokens (len={}): {:?}", toks.len(), toks);

    // dump Q5_K_M and BF16 for each prefix length (可通过 GLLM_DUMP_FILTER 只跑一个)
    let filters: &[&str] = match std::env::var("GLLM_DUMP_FILTER").as_deref() {
        Ok("q5") => &["q5_k_m"],
        Ok("bf16") => &["bf16"],
        _ => &["q5_k_m", "bf16"],
    };
    for filter in filters {
        let c = build_client(filter);
        let out_path = format!("/tmp/gllm_{}.bin", filter.replace("_",""));
        let mut f = std::fs::File::create(&out_path).expect("create");
        let n_pos = toks.len() as u32;
        // gllm vocab from first run
        let first = c.diagnostic_prefill_logits(&toks[..1]).expect("p1");
        let vocab = first.len() as u32;
        eprintln!("{}: n_pos={} vocab={}", filter, n_pos, vocab);
        f.write_all(&n_pos.to_le_bytes()).unwrap();
        f.write_all(&vocab.to_le_bytes()).unwrap();
        for len in 1..=toks.len() {
            let sub = &toks[..len];
            let lg = c.diagnostic_prefill_logits(sub).unwrap_or_else(|| panic!("logits len={}", len));
            let bytes = unsafe { std::slice::from_raw_parts(lg.as_ptr() as *const u8, lg.len() * 4) };
            f.write_all(bytes).unwrap();
        }
        eprintln!("wrote {} ({} pos x {} vocab)", out_path, n_pos, vocab);
    }
    if !n_layers.is_empty() { std::env::remove_var("GLLM_TRUNCATE_LAYERS"); }
}
