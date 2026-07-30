//! 第一性原理: 开启 diagnostic-layer-capture, dump 每层输出, 定位腐败起始层
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_capture(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let stride = c.diagnostic_layer_capture_stride();
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    let _ = writeln!(f, "=== N={} capture_stride={} data.len={} ===", n, stride, sp.data.len());
    if stride == 0 {
        let _ = writeln!(f, "  capture feature 未开启, 跳过");
        let _ = f.flush();
        return;
    }
    // capture_base: 从 named_offsets 找 capture base? 或者从 abi
    // capture buffer 通常在 scratchpad 某个固定位置
    // 试探: 读 capture_base 附近
    // 找 "capture" 相关 named offset
    for (name, off, dtype) in &sp.named_offsets {
        if name.contains("capture") || name.contains("Capture") {
            let _ = writeln!(f, "  capture named: {} off={} {:?}", name, off, dtype);
        }
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_capture_n3_n4_n5() {
    let _ = std::fs::write("/tmp/capture_n.txt", "");
    dump_capture(3, "/tmp/capture_n.txt");
    dump_capture(4, "/tmp/capture_n.txt");
    dump_capture(5, "/tmp/capture_n.txt");
}
