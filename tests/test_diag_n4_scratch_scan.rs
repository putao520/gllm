//! 第一性原理: dump N=4 scratchpad 所有非零/NaN 区域, 定位 NaN 精确 offset
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_n4_scratch_scan() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "3");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");  // 单 token 纯 prefill
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/n3_scratch_scan.txt").expect("f");
    let data = &sp.data;
    let _ = writeln!(f, "scratchpad len={}", data.len());
    // 扫描所有 f32, 找 NaN 和大值区域
    let step = 1024 * 4; // 每 4KB 扫一个 f32
    let mut nan_regions: Vec<(usize, usize)> = Vec::new(); // (offset_start, offset_end)
    let mut cur_nan_start: Option<usize> = None;
    let mut region_idx = 0;
    for off in (0..data.len().saturating_sub(4)).step_by(step) {
        let bits = u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]]);
        let val = f32::from_bits(bits);
        let is_nan = val.is_nan();
        let is_nonzero = bits != 0;
        if is_nan && cur_nan_start.is_none() {
            cur_nan_start = Some(off);
            let _ = writeln!(f, "[NaN region {} start] offset={}", region_idx, off);
        } else if !is_nan && cur_nan_start.is_some() {
            let _ = writeln!(f, "[NaN region {} end] offset={} (size={})", region_idx, off, off - cur_nan_start.unwrap());
            nan_regions.push((cur_nan_start.unwrap(), off));
            cur_nan_start = None;
            region_idx += 1;
        }
        if is_nonzero && !is_nan && off < 167772160 * 2 + 4096 {
            // 只记录关键区域 (ping/pong buffer 附近)
            if off < 4096 || (off > 167772160 - 4096 && off < 167772160 + 4096) {
                let _ = writeln!(f, "  off={} val={:.4} (nonzero)", off, val);
            }
        }
    }
    if let Some(s) = cur_nan_start {
        let _ = writeln!(f, "[NaN region {} end] offset=EOF", region_idx);
        nan_regions.push((s, data.len()));
    }
    let _ = writeln!(f, "\n=== NaN regions: {} ===", nan_regions.len());
    for (s, e) in &nan_regions {
        let _ = writeln!(f, "  NaN: offset {}..{} (size {})", s, e, e - s);
    }
    let _ = f.flush();
    eprintln!("scan done, see /tmp/n4_scratch_scan.txt");
}
