//! Phi4 vs SmolLM2 diagnostic: compare forward-only logits
//!
//! Tests if Phi4's forward pass produces reasonable logits compared to SmolLM2.
use gllm::Client;

fn get_diag_logits(model: &str, prompt: &str) -> (String, Vec<f32>, usize, usize) {
    let client = Client::new_chat(model).expect("load");
    let manifest = client.manifest().expect("manifest");
    let arch = manifest.arch.clone();

    // Generate to encode prompt + get tokens
    let r = client.generate(prompt).max_tokens(1).temperature(0.0)
        .generate().response().expect("gen");

    // Get forward-only logits using diagnostic API
    // First, encode the prompt via the backend
    let state = client.state();
    let tokens = state.backend().executor().encode_prompt(prompt).expect("encode");
    eprintln!("[DIAG] {} tokens={:?}", model, tokens);

    let logits = state.backend().executor().diagnostic_forward_only(&tokens)
        .unwrap_or_else(|| panic!("{}: diagnostic_forward_only not available", model));

    (arch, logits, tokens.len(), client.manifest().unwrap().vocab_size)
}

#[test]
fn compare_phi4_vs_smollm_logits() {
    let prompt = "The capital of France is";

    // SmolLM2 (known good)
    let (smol_arch, smol_logits, smol_tok_len, smol_vocab) = get_diag_logits(
        "HuggingFaceTB/SmolLM2-135M-Instruct", prompt
    );
    eprintln!("[SMOL] arch={:?} logits={} tokens={} vocab={}", smol_arch, smol_logits.len(), smol_tok_len, smol_vocab);

    let smol_nonzero = smol_logits.iter().filter(|&&x| x != 0.0).count();
    let smol_nan = smol_logits.iter().filter(|&&x| x.is_nan()).count();
    let smol_max = smol_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let smol_min = smol_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("[SMOL] nonzero={} nan={} range=[{:.4}, {:.4}]", smol_nonzero, smol_nan, smol_min, smol_max);

    // Find top token
    let mut smol_idx: Vec<(usize, f32)> = smol_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    smol_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("[SMOL] top_token_id={} top_val={:.4}", smol_idx[0].0, smol_idx[0].1);

    // Phi4-mini (broken)
    let (phi_arch, phi_logits, phi_tok_len, phi_vocab) = get_diag_logits(
        "microsoft/Phi-4-mini-instruct", prompt
    );
    eprintln!("[PHI] arch={:?} logits={} tokens={} vocab={}", phi_arch, phi_logits.len(), phi_tok_len, phi_vocab);

    let phi_nonzero = phi_logits.iter().filter(|&&x| x != 0.0).count();
    let phi_nan = phi_logits.iter().filter(|&&x| x.is_nan()).count();
    let phi_max = phi_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let phi_min = phi_logits.iter().cloned().fold(f32::INFINITY, f32::min);
    eprintln!("[PHI] nonzero={} nan={} range=[{:.4}, {:.4}]", phi_nonzero, phi_nan, phi_min, phi_max);

    let mut phi_idx: Vec<(usize, f32)> = phi_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    phi_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("[PHI] top_token_id={} top_val={:.4}", phi_idx[0].0, phi_idx[0].1);

    // Both should produce reasonable (non-zero, non-NaN) logits
    assert!(phi_nonzero > 0, "Phi4 logits should not be all zeros");
    assert_eq!(phi_nan, 0, "Phi4 logits should not contain NaN");
}
