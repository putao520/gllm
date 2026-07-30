#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_wblob_l34() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "6");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let blob_pre = c.diagnostic_weight_blob_bytes().expect("blob_pre");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    let blob = c.diagnostic_weight_blob_bytes().expect("blob");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/wblob_l34.txt").expect("f");
    let lbb = 234593280usize; // layer_blob_base
    // offset_table from VmProgram dump: [0, 11370496, 22740992, 34111488, 44924928, 55738368, ...]
    // BUT after BUG-A fix, layer_bytes changed. Need fresh offset_table.
    // Read from named_offsets: L0.input_norm scratchpad off gives layer_blob_base.
    // layer{i} input_norm weight = blob[lbb + offset_table[i]]
    // We need offset_table. Let me read it from the mixed_quant config via a debug env.
    // For now, dump L3/L4 input_norm at estimated offsets using layer_bytes=10824192 for Q5K
    // offset_table[3] = layer_bytes[0]+[1]+[2] (all Q6K) = 3*11379712 = 34139136
    // offset_table[4] = offset_table[3] + layer_bytes[3](Q5K=10824192) = 44963328
    let ot3 = 34139136usize;
    let ot4 = 44961792usize; // from fresh offset_table dump
    // blob BEFORE prefill
    {
        let off = lbb + ot4;
        if off+8 <= blob_pre.len() {
            let _ = writeln!(f, "PRE-PREFILL layer4 input_norm off={} bytes={:02x?}", off, &blob_pre[off..off+8]);
        }
    }
    for (li, ot) in [(3usize, ot3), (4usize, ot4)] {
        let off = lbb + ot;
        if off + 8 <= blob.len() {
            let b: Vec<f32> = (0..2).map(|i| f32::from_le_bytes([
                blob[off+i*4],blob[off+i*4+1],blob[off+i*4+2],blob[off+i*4+3]])).collect();
            let _ = writeln!(f, "layer{} input_norm off={} bytes={:02x?} f32={:?}", li, off, &blob[off..off+8], b);
        }
    }
}
