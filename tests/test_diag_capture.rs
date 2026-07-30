//! 用 diagnostic-layer-capture 逐层 dump 真实 activation 输出.
//! capture 在 ActivationSwap 之前拷贝, 保留每层真实输出.
//! layer0 capture = layer0_out (非零, N=1 已证 1.6617)
//! layer1 capture = layer1_out
//! 对比 Q5_K_M vs Q6_K: 若 Q5_K_M layer1 capture 全零/NaN → layer1 计算崩
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_capture(label: &str, filter: &str, n: usize) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(filter).build().expect("client");
    let t = c.encode(" ").expect("encode");  // 单 token: 纯 prefill 无 decode
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true)
        .open("/tmp/q5km_capture.txt").expect("f");

    // layer_capture offset + N * stride
    let cap_off = sp.named_offsets.iter().find(|(n,_,_)| n == "layer_capture")
        .map(|(_,o,_)| *o).unwrap_or(0);
    let _ = writeln!(f, "[{} N={}] layer_capture offset={} named_offsets.len={}",
        label, n, cap_off, sp.named_offsets.len());

    // 读每层 capture (hidden=1024 elem, F32)
    let hidden = 1024;
    for layer in 0..n {
        let off = cap_off + layer * hidden * 4;
        let vals = sp.read_dtype_aware(off, 4);
        let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
        let _ = writeln!(f, "  layer{} capture off={} first4={:?} |max|={:.4}",
            layer, off, &vals, max);
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_capture_per_layer() {
    let _ = std::fs::write("/tmp/q5km_capture.txt", "");
    eprintln!("=== diagnostic-layer-capture N=4 纯prefill 逐层 (找NaN起点) ===");
    dump_capture("Q5_K_M", "q5_k_m", 4);
}
