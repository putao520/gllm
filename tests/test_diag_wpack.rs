//! 验证 N=4 时 layer3 v_proj 权重是否在 blob 内 + 字节正确
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn verify_layer3_vproj_in_blob() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    // 触发编译但不跑推理, 只取 weight_blob
    // 通过 diagnostic 拿 named_offsets + data (data 含 weight_blob? 不, data 是 scratchpad)
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/wpack_check.txt").expect("f");
    // 找 layer.v / layer3.v 的 offset
    let v_off = sp.named_offsets.iter().find(|(n,_,_)| n == "layer.v").map(|(_,o,_)| *o).unwrap_or(0);
    let v3_off = sp.named_offsets.iter().find(|(n,_,_)| n == "layer3.v").map(|(_,o,_)| *o).unwrap_or(0);
    let _ = writeln!(f, "layer.v off={} (layer0)", v_off);
    let _ = writeln!(f, "layer3.v off={} ", v3_off);
    // dump 前 16 字节看是否是合法 Q6K block 头
    for (name, off) in [("layer.v", v_off), ("layer3.v", v3_off)] {
        if off > 0 {
            // 读取前 16 字节 (raw, 不 decode)
            let end = (off + 16).min(sp.data.len());
            if off < sp.data.len() {
                let bytes = &sp.data[off..end];
                let _ = writeln!(f, "  {} first16 bytes: {:02x?}", name, bytes);
            }
        }
    }
    let _ = f.flush();
    eprintln!("done /tmp/wpack_check.txt");
}
