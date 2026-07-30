//! 验证 Q5_K_M 的 KV cache dtype 双地层一致性 (SmolLM2 前车之鉴)
//! JIT 层 ctx.dtype=graph_dtype() vs buffer 层 compute_dtype
//! 若不一致 → stride 不一致 → 越界踩踏 (SmolLM2 的 768 vs 384 陷阱)
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn diag_kv_dtype_dual_layer() {
    eprintln!("\n=== Q5_K_M vs Q6_K KV cache dtype 双地层 ===");

    for (label, filter) in [("Q5_K_M", "q5_k_m"), ("Q6_K", "q6_k")] {
        std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
        let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
            .gguf_file_filter(filter).build().expect("client");
        let t = c.encode("The capital of France is").expect("encode");
        let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
        std::env::remove_var("GLLM_TRUNCATE_LAYERS");
        // compute_dtype 是 buffer 层的 (scratchpad 分配依据)
        eprintln!("[{}] compute_dtype(buffer层)={:?} | logits_off={} vocab={} data.len={}",
            label, sp.compute_dtype, sp.logits_offset, sp.vocab_size, sp.data.len());
        // 关键: data.len 包含 KV cache 吗? SmolLM2 陷阱是 KV cache buffer 按 BF16(384) 但 MemCopy 写 768 越界
        // 如果 compute_dtype=F32 → buffer 按 F32 分配 → stride 一致 → 无越界
        // 如果 compute_dtype=BF16 → buffer 按 BF16(384) 但 JIT MemCopy 按 F32(768) → 越界!
        drop(c);
    }
    eprintln!("\nSmolLM2 前车之鉴: JIT ctx.dtype=F32(768 stride) vs buffer compute_dtype=BF16(384) → 越界踩踏");
    eprintln!("若 Q5_K_M compute_dtype != Q6_K compute_dtype → 嫌疑命中");
}
