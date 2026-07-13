//! 验证 blob token_embd block0 d 值 vs GGUF token_embd block0 vs 运行时 #0
#![cfg(test)]
use gllm::{Client, ModelKind};
use gllm::loader::gguf::GgufReader;
use half::f16;

#[test]
#[ignore]
fn diag_token_embd_blob_vs_gguf() {
    let path = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
    let r = GgufReader::open(path).expect("open");
    let embd = r.tensors().iter().find(|t| t.name.as_ref() == "token_embd.weight").expect("find token_embd");
    let file_bytes = std::fs::read(path).expect("read");
    let gguf_embd_block0 = &file_bytes[embd.offset..embd.offset + 16];
    let gd = f16::from_le_bytes([gguf_embd_block0[0], gguf_embd_block0[1]]).to_f32();
    eprintln!("[GGUF] token_embd block0 d={:.6} dmin={:.6}", gd,
        f16::from_le_bytes([gguf_embd_block0[2], gguf_embd_block0[3]]).to_f32());

    std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let _ = c.encode("test").expect("encode");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    let blob = c.diagnostic_weight_blob_bytes().expect("blob");
    let woffs = c.diagnostic_weight_offsets().expect("woffs");
    eprintln!("[BLOB] weight_blob len={}", blob.len());
    // 找 embed (可能叫 embed/embd/token_embd)
    for (n, o, d) in &woffs {
        if n.contains("embd") || n.contains("embed") || n == "token_embd" {
            let b = &blob[*o..*o+16];
            let bd = f16::from_le_bytes([b[0], b[1]]).to_f32();
            eprintln!("[BLOB] {} off={} dtype={:?} block0 d={:.6}", n, o, d, bd);
            let m = b.iter().zip(gguf_embd_block0.iter()).filter(|(a,b)| a != b).count();
            eprintln!("  vs GGUF token_embd: {} 处不匹配 (0=pack 正确)", m);
        }
    }
    eprintln!("\n运行时 #0 d=0.000063, GGUF token_embd block0 d={:.6}", gd);
    eprintln!("若 blob token_embd d != GGUF d → pack 改了 embed (第17方向漏检)");
    eprintln!("若 blob token_embd d == GGUF d != 运行时 d → 运行时读错地址");
}
