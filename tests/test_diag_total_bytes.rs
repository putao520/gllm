//! 第一性原理: dump weight_layout.total_bytes + 各关键 offset
//! 验证 layer3 v_proj weight 是否在 blob 范围内
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_total_bytes() {
    for n in [3, 4, 5] {
        std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
        let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
            .gguf_file_filter("q5_k_m").build().expect("client");
        let t = c.encode(" ").expect("encode");
        let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
        std::env::remove_var("GLLM_TRUNCATE_LAYERS");
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/total_bytes.txt").expect("f");
        let _ = writeln!(f, "=== N={} data.len(scratch)={} ===", n, sp.data.len());
        // 找最大 offset + 估算
        let mut max_off = 0;
        let mut max_name = String::new();
        for (name, off, _) in &sp.named_offsets {
            if *off > max_off { max_off = *off; max_name = name.clone(); }
        }
        let _ = writeln!(f, "  max named offset: {} off={}", max_name, max_off);
        // L0.* 区间
        let l0_first = sp.named_offsets.iter().find(|(n,_,_)| n == "L0.input_norm").map(|(_,o,_)| *o);
        let l0_down = sp.named_offsets.iter().find(|(n,_,_)| n == "L0.down_proj").map(|(_,o,_)| *o);
        if let (Some(f_), Some(d)) = (l0_first, l0_down) {
            let _ = writeln!(f, "  L0.input_norm={} L0.down_proj={} stride=11379712", f_, d);
            let layer3_v = 236760064 + 3*11379712;
            let _ = writeln!(f, "  layer3 v_proj weight off = 236760064 + 3*11379712 = {}", layer3_v);
        }
        let _ = f.flush();
    }
}
