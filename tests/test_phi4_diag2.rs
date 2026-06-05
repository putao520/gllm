//! Minimal Phi4 diagnostic: check rope_cache parameters
use gllm::Client;

#[test]
fn phi4_rope_diag() {
    let model = "microsoft/Phi-4-mini-instruct";
    let client = Client::new_chat(model).expect("load");

    // Generate with env var GLLM_DEBUG_RESOURCE=1 to get plan-lower output
    let r = client
        .generate("The capital of France is")
        .max_tokens(3)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen");
    eprintln!("[DIAG2] output: {:?}", r.text);
}
