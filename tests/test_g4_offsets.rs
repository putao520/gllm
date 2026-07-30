#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_list_offsets() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let client = Client::builder().model("unsloth/gemma-4-E2B-it-GGUF").kind(ModelKind::Chat).build().unwrap();
    let toks = client.encode("The").unwrap();
    if let Some(sp) = client.diagnostic_prefill_scratchpad(&toks) {
        eprintln!("logits_offset={} total_len={}", sp.logits_offset, sp.data.len());
        for (name, off, dt) in &sp.named_offsets {
            eprintln!("  {} off={} dt={:?}", name, off, dt);
        }
    }
    eprintln!("DONE");
}
