//! 第一性原理: 查 N=4 NaN 区域的 offset 对应哪个 named tensor
//! 定位 NaN 的具体来源 (q_proj/k_proj/v_proj/o_proj/gate/up/down)
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_n4_nan_tensor_names() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "3");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/n3_nan_names.txt").expect("f");
    // NaN 区域 offset (方向44 扫描结果)
    let nan_offsets = [167772160usize, 503316480, 671088640, 838860800, 1342177280, 1845493760, 2369781760];
    let _ = writeln!(f, "=== N=4 NaN 区域对应的 named tensor ===");
    for &nan_off in &nan_offsets {
        // 找哪个 named tensor 的 offset 包含 nan_off
        let mut found = false;
        for (name, off, dtype) in &sp.named_offsets {
            // tensor 大小未知, 假设 offset 最接近的
            if *off <= nan_off && *off + 4 * 1024 * 1024 > nan_off {
                // 读该 tensor 的前 4 个 f32 看是否 NaN
                let vals = sp.read_dtype_aware(*off, 4);
                let is_nan = vals.iter().any(|v| v.is_nan());
                let _ = writeln!(f, "  NaN off={} → tensor '{}' off={} dtype={:?} first4={:?} is_nan={}",
                    nan_off, name, off, dtype, &vals, is_nan);
                found = true;
            }
        }
        if !found {
            let _ = writeln!(f, "  NaN off={} → 无 named tensor 匹配", nan_off);
        }
    }
    // 也 dump 所有 named tensor 的前 4 值 + 是否 NaN
    let _ = writeln!(f, "\n=== 所有 named tensor (前4值 + NaN检查) ===");
    for (name, off, dtype) in &sp.named_offsets {
        let vals = sp.read_dtype_aware(*off, 4);
        let is_nan = vals.iter().any(|v| v.is_nan());
        let max = vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if is_nan || max > 1e-6 {
            let _ = writeln!(f, "  {:30} off={:>12} {:?} first4={:?} NaN={}", name, off, dtype, &vals, is_nan);
        }
    }
    let _ = f.flush();
    eprintln!("done, see /tmp/n4_nan_names.txt");
}
