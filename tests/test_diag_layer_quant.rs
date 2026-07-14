//! 第一性原理: 查 Q5_K_M 各层 v_proj 的实际量化类型
//! Q5_K_M 的 _M 策略可能让某些层 v_proj 用不同量化
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_layer_quant() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/layer_quant.txt").expect("f");
    // 找所有 L*.v_proj 的 named_offsets
    for (name, off, dtype) in &sp.named_offsets {
        if name.contains("v_proj") || name.contains("v_proj") {
            let _ = writeln!(f, "{} off={} {:?}", name, off, dtype);
        }
    }
    let _ = writeln!(f, "\n=== 所有 L*. 权重 (看是否有 L3. 等) ===");
    for (name, off, dtype) in &sp.named_offsets {
        if name.starts_with("L") && name.contains(".") {
            let _ = writeln!(f, "{} off={} {:?}", name, off, dtype);
        }
    }
    let _ = f.flush();
    eprintln!("done /tmp/layer_quant.txt");
}
