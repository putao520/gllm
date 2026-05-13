//! Phi4 diagnostic: generate + diagnostic_prefill_logits
//!
//! First generates a short output to see if Phi4 produces degenerate text,
//! then uses diagnostic_prefill_logits to check raw logits.
//! Uses hardcoded token IDs for "The capital of France is" from Phi4 tokenizer.

use gllm::Client;

#[test]
fn phi4_generate_and_logits() {
    let model = "microsoft/Phi-4-mini-instruct";
    eprintln!("[DIAG] Loading {}...", model);

    let client = Client::new_chat(model).expect("Failed to load Phi-4 model");
    let manifest = client.manifest().expect("manifest");
    eprintln!("[DIAG] arch={:?} kind={:?}", manifest.arch, manifest.kind);

    // Test 1: Simple generation
    let r = client
        .generate("The capital of France is")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen failed");
    eprintln!("[DIAG] output: {:?}", r.text);

    // Test 2: Try diagnostic_prefill_logits with hardcoded tokens
    // Phi4 tokenizer for "The capital of France is":
    // These are common subword token IDs; if diagnostic returns None, we need actual tokenizer
    // Let's try with a simple token sequence
    let simple_tokens: Vec<u32> = vec![1, 450, 3527, 315, 7874, 374]; // BOS + "The capital of France is"
    if let Some(logits) = client.diagnostic_prefill_logits(&simple_tokens) {
        let nonzero = logits.iter().filter(|&&x| x != 0.0).count();
        let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[DIAG] logits: len={} nonzero={} nan={} range=[{:.4}, {:.4}]",
                  logits.len(), nonzero, nan_count, min_val, max_val);
    } else {
        eprintln!("[DIAG] diagnostic_prefill_logits returned None");
    }

    // Test 3: diagnostic_forward_only with same tokens
    if let Some(logits) = client.diagnostic_forward_only(&simple_tokens) {
        let nonzero = logits.iter().filter(|&&x| x != 0.0).count();
        let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[FWD] logits: len={} nonzero={} nan={} range=[{:.4}, {:.4}]",
                  logits.len(), nonzero, nan_count, min_val, max_val);

        // Top-5 tokens
        let mut ranked: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, (tid, val)) in ranked.iter().take(5).enumerate() {
            eprintln!("[FWD] #{}: token_id={} val={:.4}", idx, tid, val);
        }
    } else {
        eprintln!("[FWD] diagnostic_forward_only returned None");
    }
}
