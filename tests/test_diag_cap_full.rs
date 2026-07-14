//! 第一性原理: dump capture 区域全部非零值, 看是否 copy 写入
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_cap_full() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let stride = c.diagnostic_layer_capture_stride();
    let cap_off = sp.named_offsets.iter().find(|(nm,_,_)| nm == "layer_capture").map(|(_,o,_)| *o).unwrap_or(0);
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/cap_full.txt").expect("f");
    let _ = writeln!(f, "cap_off={} stride={} data.len={}", cap_off, stride, sp.data.len());
    // 扫描每层 capture 区域 (stride bytes = 167772160, 但 hidden=1024 elem=4096 bytes 有意义)
    // 扫前 8KB (2048 f32) 找非零
    for layer in 0..4 {
        let off = cap_off + layer * stride;
        let scan_elem = 2048; // 扫前 2048 f32
        let mut nonzero = 0;
        let mut first_nonzero: Option<(usize, f32)> = None;
        for i in 0..scan_elem {
            let o = off + i*4;
            if o + 4 <= sp.data.len() {
                let bits = u32::from_le_bytes([sp.data[o],sp.data[o+1],sp.data[o+2],sp.data[o+3]]);
                if bits != 0 {
                    nonzero += 1;
                    if first_nonzero.is_none() {
                        let v = f32::from_bits(bits);
                        first_nonzero = Some((i, v));
                    }
                }
            }
        }
        let _ = writeln!(f, "layer{} off={}: nonzero={}/{} first_nonzero={:?}", layer, off, nonzero, scan_elem, first_nonzero);
    }
    let _ = f.flush();
}
