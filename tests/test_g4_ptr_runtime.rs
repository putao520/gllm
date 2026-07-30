//! 验证 embed_ptr 运行时指向的 blob 字节 == GGUF token_embd row
#![cfg(test)]
use gllm::Client;
#[test]
#[ignore]
fn g4_ptr_runtime_check() {
    let c = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q8_0.gguf").expect("load");
    let t = c.encode("The").expect("encode");
    eprintln!("tokens: {:?}", t);
    let tid = t[0] as usize;
    eprintln!("token_id={}", tid);
    if let Some(blob) = c.diagnostic_weight_blob_bytes() {
        let row_off = tid * 864;
        eprintln!("blob_len={} row_off={}", blob.len(), row_off);
        if row_off + 144 <= blob.len() {
            let row = &blob[row_off..row_off + 144];
            let d = half::f16::from_le_bytes([row[0], row[1]]).to_f32();
            let dmin = half::f16::from_le_bytes([row[2], row[3]]).to_f32();
            eprintln!("blob row{} d={:.6} dmin={:.6} scales[0..8]={:02x?}", tid, d, dmin, &row[4..12]);
            eprintln!("blob qs[0..16]={:02x?}", &row[16..32]);
        } else {
            eprintln!("row_off {} out of blob_len {}", row_off, blob.len());
        }
    } else {
        eprintln!("NO blob API");
    }
    eprintln!("DONE");
}
