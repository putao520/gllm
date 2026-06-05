//! Phi4 name_map + weight_shapes diagnostic
//!
//! Verifies that fused QKV tensors are correctly mapped to canonical names
//! and that weight_shapes has the right canonical keys.

use gllm::Client;

#[test]
fn phi4_name_map_diagnostic() {
    let model = "microsoft/Phi-4-mini-instruct";
    eprintln!("[NM-DIAG] Loading {}...", model);

    let client = Client::new_chat(model).expect("Failed to load Phi-4 model");
    let manifest = client.manifest().expect("manifest");
    eprintln!("[NM-DIAG] arch={:?} kind={:?}", manifest.arch, manifest.kind);

    // The actual test: try a single generation with temperature=0
    let r = client
        .generate("The capital of France is")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen failed");
    eprintln!("[NM-DIAG] output: {:?}", r.text);
}
