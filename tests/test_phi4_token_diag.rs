//! Phi4 diagnostic: check logits with correct token IDs

use gllm::Client;

#[test]
fn phi4_token_and_logits_diag() {
    let model = "microsoft/Phi-4-mini-instruct";
    eprintln!("[TDIAG] Loading {}...", model);

    let client = Client::new_chat(model).expect("Failed to load Phi-4 model");
    let manifest = client.manifest().expect("manifest");
    eprintln!("[TDIAG] arch={:?} kind={:?}", manifest.arch, manifest.kind);

    // Generate to see output
    let r = client
        .generate("The capital of France is")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen failed");
    eprintln!("[TDIAG] generate output: {:?}", r.text);

    // Correct tokens from tokenizer: "The capital of France is"
    let tokens: Vec<u32> = vec![976, 9029, 328, 10128, 382];
    eprintln!("[TDIAG] prompt tokens: {:?}", tokens);

    if let Some(logits) = client.diagnostic_forward_only(&tokens) {
        let nonzero = logits.iter().filter(|&&x| x != 0.0).count();
        let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[TDIAG] logits: len={} nonzero={} nan={} range=[{:.4}, {:.4}]",
                  logits.len(), nonzero, nan_count, min_val, max_val);

        // Top-10 tokens
        let mut ranked: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, (tid, val)) in ranked.iter().take(10).enumerate() {
            eprintln!("[TDIAG] #{}: token_id={} val={:.4}", idx, tid, val);
        }

        // Check "Paris" candidates (correct token IDs)
        let paris_tokens = [72782, 12650]; // "Paris", " Paris"
        for &pid in &paris_tokens {
            if (pid as usize) < logits.len() {
                eprintln!("[TDIAG] token {} logit={:.4}", pid, logits[pid as usize]);
            }
        }
    } else {
        eprintln!("[TDIAG] diagnostic_forward_only returned None");
    }
}
