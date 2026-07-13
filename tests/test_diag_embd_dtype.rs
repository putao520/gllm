//! 查 Q5_K_M vs Q6_K 的所有非量化 tensor dtype (token_embd / output_norm / norm / output)
//! 关键: 若 Q5_K_M token_embd=BF16 而 Q6_K=F32 → compute_dtype 不同 → KV cache stride 陷阱
#![cfg(test)]
use gllm::loader::gguf::{GgufReader, GgmlDType};

const Q5: &str = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
const Q6: &str = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q6_K.gguf";

fn dump_float_tensors(label: &str, path: &str) {
    let r = GgufReader::open(path).expect("open");
    let mut f = std::fs::OpenOptions::new().create(true).append(true)
        .open("/tmp/q5km_embd_dtype.txt").expect("f");
    use std::io::Write;
    let _ = writeln!(f, "\n[{}] 非量化(float) tensor:", label);
    let mut f32c = 0; let mut bf16c = 0; let mut f16c = 0;
    for t in r.tensors().iter() {
        match t.dtype {
            GgmlDType::F32 => { f32c += 1; if t.name.contains("embd") || t.name.contains("norm") || t.name.contains("output") || !t.name.contains("blk") {
                let _ = writeln!(f, "  F32: {}", t.name); } }
            GgmlDType::BF16 => { bf16c += 1; let _ = writeln!(f, "  BF16: {}", t.name); }
            GgmlDType::F16 => { f16c += 1; let _ = writeln!(f, "  F16: {}", t.name); }
            _ => {}
        }
    }
    let derive = if bf16c >= f32c && bf16c >= f16c && bf16c > 0 { "BF16" }
        else if f16c >= f32c && f16c >= bf16c && f16c > 0 { "F16" }
        else { "F32" };
    let _ = writeln!(f, "[{}] F32 count={}, BF16 count={}, F16 count={} → derive_dtype={}", label, f32c, bf16c, f16c, derive);
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_embd_dtype_compare() {
    let _ = std::fs::write("/tmp/q5km_embd_dtype.txt", "");
    dump_float_tensors("Q5_K_M", Q5);
    dump_float_tensors("Q6_K", Q6);
    eprintln!("\nSmolLM2 陷阱: compute_dtype=BF16 → KV cache buffer 按 384 分配, JIT MemCopy 按 F32(768) 写 → 越界");
    eprintln!("若 Q5_K_M derive_dtype=BF16 而 Q6_K=F32 → 嫌疑命中 (Q5_K_M 踩 SmolLM2 同陷阱)");
}
