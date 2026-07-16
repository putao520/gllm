#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_wblob_l1norm() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let _sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    let blob = c.diagnostic_weight_blob_bytes().expect("blob");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/wblob_l1norm.txt").expect("f");
    let _ = writeln!(f, "blob len={}", blob.len());
    // layer_blob_base = 234593280 (embed region)
    // offset_table: [0, 11370496, ...]
    // input_norm intra-layer offset: attn_norm is first, = 0
    // layer0 input_norm: blob[234593280 .. +32]
    // layer1 input_norm: blob[234593280+11370496 .. +32] = blob[245963776..]
    let lbb = 234593280usize;
    let off_l0 = lbb;
    let off_l1 = lbb + 11370496;
    let _ = writeln!(f, "layer0 input_norm off={} first32={:02x?}", off_l0, &blob[off_l0..off_l0+32]);
    let _ = writeln!(f, "layer1 input_norm off={} first32={:02x?}", off_l1, &blob[off_l1..off_l1+32]);
    // decode as f32 (input_norm is F32? or quant?) - check first 4 f32
    let f32_l0: Vec<f32> = (0..4).map(|i| {
        let b=[blob[off_l0+i*4],blob[off_l0+i*4+1],blob[off_l0+i*4+2],blob[off_l0+i*4+3]];
        f32::from_le_bytes(b)
    }).collect();
    let f32_l1: Vec<f32> = (0..4).map(|i| {
        let b=[blob[off_l1+i*4],blob[off_l1+i*4+1],blob[off_l1+i*4+2],blob[off_l1+i*4+3]];
        f32::from_le_bytes(b)
    }).collect();
    let _ = writeln!(f, "layer0 input_norm as f32: {:?}", f32_l0);
    let _ = writeln!(f, "layer1 input_norm as f32: {:?}", f32_l1);
}
