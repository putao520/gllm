#![cfg(test)]
//! 逐 N 对照: gllm BF16 在不同截断层数 N 下的最后 token logits vs llama.cpp BF16。
//! 定位 gllm 哪层开始与 llama.cpp 分叉。
//! 输出 /tmp/gllm_bf16_N{N}.bin，与 /tmp/llama_bf16.bin 对照。
use gllm::{Client, ModelKind};
use std::io::Write;

fn build_client(n: usize) -> Client {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("bf16")
        .build()
        .unwrap_or_else(|e| panic!("build N={}: {:?}", n, e))
}

#[test]
#[ignore]
fn diag_dump_gllm_bf16_by_n() {
    let prompt = "The capital of France is";
    // 每个截断 N 单独 build client (graph 重建)
    for &n in &[1usize, 2, 3, 4, 7, 14, 28] {
        let c = build_client(n);
        let toks = c.encode(prompt).expect("encode");
        // 只取最后 token (pos = full prompt) 的 logits
        let lg = c.diagnostic_prefill_logits(&toks).unwrap_or_else(|| panic!("logits N={}", n));
        let out = format!("/tmp/gllm_bf16_N{}.bin", n);
        let mut f = std::fs::File::create(&out).unwrap();
        let np = 1u32; let v = lg.len() as u32;
        f.write_all(&np.to_le_bytes()).unwrap();
        f.write_all(&v.to_le_bytes()).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(lg.as_ptr() as *const u8, lg.len()*4) };
        f.write_all(bytes).unwrap();
        drop(c);
        std::env::remove_var("GLLM_TRUNCATE_LAYERS");
        eprintln!("N={}: wrote {} ({} logits), tokens={:?}", n, out, lg.len(), toks);
    }
}
