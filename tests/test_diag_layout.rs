#![cfg(test)]
use gllm::{Client, ModelKind};
fn dump_layout(n: usize, f: &mut std::fs::File) {
    use std::io::Write;
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let _ = writeln!(f, "=== N={} scratchpad len={} ===", n, sp.data.len());
    // dump any named tensor with offset near 167772160 (pong region) or overlap
    for (name, off, dt) in &sp.named_offsets {
        if *off >= 167000000 && *off < 340000000 {
            let _ = writeln!(f, "  near-pong: {:16} off={} dt={:?}", name, off, dt);
        }
    }
}
#[test]
#[ignore]
fn diag_layout() {
    let mut f = std::fs::File::create("/tmp/layout.txt").expect("f");
    dump_layout(1, &mut f);
    dump_layout(2, &mut f);
}
