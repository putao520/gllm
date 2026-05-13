//! Phi4 diagnostic: test with different temperatures and token counts to identify root cause

use gllm::Client;

#[test]
fn phi4_diag_token_analysis() {
    let model = "microsoft/Phi-4-mini-instruct";
    eprintln!("[DIAG] Loading {}...", model);

    let client = Client::new_chat(model).expect("Failed to load Phi-4 model");
    let manifest = client.manifest().expect("manifest");
    eprintln!("[DIAG] arch={:?} kind={:?}", manifest.arch, manifest.kind);

    // Test 1: temperature=0, max_tokens=1 (just the first token)
    let r1 = client
        .generate("The capital of France is")
        .max_tokens(1)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen1 failed");
    eprintln!("[DIAG] T=0, max_tokens=1: {:?}", r1.text);

    // Test 2: temperature=0, max_tokens=3
    let r2 = client
        .generate("The capital of France is")
        .max_tokens(3)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen2 failed");
    eprintln!("[DIAG] T=0, max_tokens=3: {:?}", r2.text);

    // Test 3: temperature=1.0, max_tokens=10 (see if high temp breaks repetition)
    let r3 = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(1.0)
        .generate()
        .response()
        .expect("gen3 failed");
    eprintln!("[DIAG] T=1, max_tokens=10: {:?}", r3.text);

    // Test 4: different prompt entirely
    let r4 = client
        .generate("2 + 2 =")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen4 failed");
    eprintln!("[DIAG] T=0, '2 + 2 =': {:?}", r4.text);
}
