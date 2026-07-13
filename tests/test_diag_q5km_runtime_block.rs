//! Q5_K_M 运行时 block_base 验证 — 用 diagnostic_prefill_scratchpad 触发真实推理.
//! 验证 JIT 运行时读的 block 地址是否指向 blob 的 L0/L1.q_proj 位置.
#![cfg(test)]
use gllm::{Client, ModelKind};
use gllm::loader::gguf::GgufReader;

#[test]
#[ignore]
fn diag_q5km_runtime_block_addr() {
    eprintln!("\n=== Q5_K_M 运行时 block_base (diagnostic_prefill_scratchpad) ===");
    use std::io::Write; let _ = std::io::stderr().flush();

    // 先在 GGUF 搜索 head16=[1b,04,18,13,...] 的张量 (运行时 #0 的签名)
    let path = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
    let r = GgufReader::open(path).expect("open");
    let file_bytes = std::fs::read(path).expect("read");
    let sig = [0x1b, 0x04, 0x18, 0x13];
    eprintln!("[GGUF 搜索 head16=[1b,04,18,13] 的张量 (运行时 #0 签名):");
    for t in r.tensors().iter() {
        if t.dtype == gllm::loader::gguf::GgmlDType::Q5_K && t.size >= 16 {
            let tb = &file_bytes[t.offset..t.offset + 4];
            if tb == sig {
                eprintln!("  ✓ MATCH: {} offset={} block0 head16={:02x?}",
                    t.name, t.offset, &file_bytes[t.offset..t.offset+16]);
            }
        }
    }

    let prompt = "The capital of France is";
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
    let q5 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("Q5_K_M");
    // dump blob 基地址 + woffs (运行时 block 地址 - blob 基 = blob offset)
    let blob = q5.diagnostic_weight_blob_bytes().expect("blob");
    let blob_base = blob.as_ptr() as usize;
    eprintln!("[BLOB-BASE] blob ptr = {:#x}, len={}", blob_base, blob.len());
    let woffs = q5.diagnostic_weight_offsets().expect("woffs");
    for (n, o, _) in &woffs {
        if n.contains("embd") || n.contains("embed") || n == "L0.q_proj" || n == "L0.o_proj" {
            eprintln!("[WOFF] {} blob_off={} abs_addr={:#x}", n, o, blob_base + o);
        }
    }
    drop(blob);
    let qt = q5.encode(prompt).expect("encode");
    // ★diagnostic_prefill_scratchpad 真实执行 mega-kernel
    let qsp = q5.diagnostic_prefill_scratchpad(&qt).expect("prefill");
    let qv = qsp.vocab_size;
    let qlg = qsp.read_dtype_aware(qsp.logits_offset, qv);
    let qam = qlg.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0);
    let qmx = qlg.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[Q5_K_M N=2] argmax={} |max|={:.4}", qam, qmx);
    drop(q5);
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
