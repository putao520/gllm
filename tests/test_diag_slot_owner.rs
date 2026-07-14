//! 第一性原理: dump 所有 named_offsets, 看每个 slot 被哪些 tensor 共享
//! 特别关注 v slot (1174405120) 是否被其他 op 覆盖
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_slot_owners() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/slot_owners.txt").expect("f");
    let _ = writeln!(f, "=== 所有 named_offsets (按 offset 分组) ===");
    let mut entries: Vec<(String, usize, gllm_kernels::types::DType)> = sp.named_offsets.clone();
    entries.sort_by_key(|e| e.1);
    let mut cur_off = 0usize;
    for (name, off, dtype) in &entries {
        if *off != cur_off {
            let _ = writeln!(f, "--- offset {} ---", off);
            cur_off = *off;
        }
        let _ = writeln!(f, "  {:30} off={} {:?}", name, off, dtype);
    }
    // 特别检查 v slot
    let v_off = 1174405120usize;
    let _ = writeln!(f, "\n=== v slot off={} 的所有共享者 ===", v_off);
    for (name, off, dtype) in &entries {
        if *off == v_off {
            let _ = writeln!(f, "  {} off={} {:?}", name, off, dtype);
        }
    }
    // k slot
    let k_off = 1006632960usize;
    let _ = writeln!(f, "\n=== k slot off={} 的所有共享者 ===", k_off);
    for (name, off, dtype) in &entries {
        if *off == k_off {
            let _ = writeln!(f, "  {} off={} {:?}", name, off, dtype);
        }
    }
    let _ = f.flush();
    eprintln!("done /tmp/slot_owners.txt");
}
