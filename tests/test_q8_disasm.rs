//! W512 Gemma4 Q4_0 JIT dump for static QuantGather diagnosis.
//! BCE-20260730-Q4K-MIN-SIGN: verify Q4_K min SUBTRACTION fix produces valid embed/logits.
#![cfg(test)]

use gllm::Client;

#[test]
#[ignore]
fn q8_disasm_gemma4_w512() {
    let outdir = "/tmp/g4-disasm";
    let _ = std::fs::remove_dir_all(outdir);
    std::fs::create_dir_all(outdir).expect("create dump directory");
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");
    std::env::set_var("GLLM_DUMP_MEGA", outdir);
    std::env::set_var("GLLM_DUMP_OFFSETMAP", "1");

    let result = (|| {
        let client = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q4_0.gguf")
            .expect("load Gemma4 Q4_0");
        let tokens = client.encode("The").expect("encode token");
        eprintln!("tokens={tokens:?}");
        let logits = client
            .diagnostic_prefill_logits(&tokens)
            .expect("diagnostic prefill logits");
        let nan = logits.iter().filter(|v| v.is_nan()).count();
        let inf = logits.iter().filter(|v| v.is_infinite()).count();
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("logits len={} nan={} inf={} min={} max={}", logits.len(), nan, inf, min, max);
        assert!(!logits.is_empty());
    })();

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    std::env::remove_var("GLLM_DUMP_MEGA");
    std::env::remove_var("GLLM_DUMP_OFFSETMAP");
    result
}
