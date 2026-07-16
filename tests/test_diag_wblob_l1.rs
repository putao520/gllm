#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn diag_wblob_l1() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    let blob = c.diagnostic_weight_blob_bytes().expect("blob");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/wblob_l1.txt").expect("f");
    let _ = writeln!(f, "blob len={}", blob.len());
    // layer_blob_base_offset + offset_table[1] = layer1 weight base
    // Need layer_blob_base_offset. From earlier: global_weight_bytes / layer_blob_base
    // Check L0.input_norm named offset (scratchpad) vs weight blob
    // layer0 input_norm at weight blob offset = layer_blob_base + 0 + attn_norm_offset
    // layer1 input_norm at weight blob offset = layer_blob_base + offset_table[1] + attn_norm_offset
    // Dump first 32 bytes at layer0 and layer1 input_norm weight regions
    // layer_blob_base: from build_graph, = global_weight_bytes (embed etc)
    // Let me find it from named_offsets (L0.input_norm is in scratchpad, not blob)
    // Actually weight blob offsets: embed at 0, then layer blob region
    // Let me just dump bytes at offset_table[0] and offset_table[1] regions (relative to layer_blob_base)
    // We need layer_blob_base. Check blob for known layer0 v_proj bytes (from wblob_verify: layer0 v_proj off=236760064)
    // layer0 v_proj off=236760064 in scratchpad. In blob: layer_blob_base + v_rel_off
    // This is getting complex. Let me dump the first 32 bytes of blob at layer_blob_base + 0 vs + 11370496
    // layer_blob_base = ? Let me search named_offsets for L0.input_norm
    for (name, off, dt) in &sp.named_offsets {
        if name == "L0.input_norm" || name == "L0.q_proj" {
            let _ = writeln!(f, "{}: scratchpad off={} dt={:?}", name, off, dt);
        }
    }
}
