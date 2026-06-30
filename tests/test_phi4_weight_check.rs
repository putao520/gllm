//! Phi4 diagnostic: check weight data and intermediate values

use gllm::Client;

#[test]
fn phi4_weight_and_hidden_check() {
    let model = "microsoft/Phi-4-mini-instruct";
    eprintln!("[WCHK] Loading {}...", model);

    let client = Client::new_chat(model).expect("Failed to load Phi-4 model");

    // Check embedding weights (first 10 values)
    if let Some(row) = client.diagnostic_weight_row("embed", 0, 10) {
        eprintln!("[WCHK] embed row 0 (first 10): {:?}", &row[..10.min(row.len())]);
        let nonzero = row.iter().filter(|&&x| x != 0.0).count();
        eprintln!("[WCHK] embed row 0 nonzero={}/{}", nonzero, row.len());
    } else {
        eprintln!("[WCHK] embed weight row NOT AVAILABLE");
    }

    // Check qkv_proj weights for layer 0
    if let Some(row) = client.diagnostic_weight_row("L0.qkv_proj", 0, 10) {
        eprintln!("[WCHK] L0.qkv_proj row 0 (first 10): {:?}", &row[..10.min(row.len())]);
        let nonzero = row.iter().filter(|&&x| x != 0.0).count();
        eprintln!("[WCHK] L0.qkv_proj row 0 nonzero={}/{}", nonzero, row.len());
    } else {
        eprintln!("[WCHK] L0.qkv_proj weight row NOT AVAILABLE");
    }

    // Check o_proj weights for layer 0
    if let Some(row) = client.diagnostic_weight_row("L0.o_proj", 0, 10) {
        eprintln!("[WCHK] L0.o_proj row 0 (first 10): {:?}", &row[..10.min(row.len())]);
    } else {
        eprintln!("[WCHK] L0.o_proj weight row NOT AVAILABLE");
    }

    // Check all weight offsets
    if let Some(offsets) = client.diagnostic_weight_offsets() {
        eprintln!("[WCHK] Total weight tensors: {}", offsets.len());
        for (name, offset, _dtype) in offsets.iter().filter(|(n, _, _)| n.contains("L0.") || n == "embed" || n == "lm_head") {
            eprintln!("[WCHK]   {} @ offset {} dtype {:?}", name, offset, _dtype);
        }
    }

    // Check generate output
    let r = client
        .generate("The capital of France is")
        .max_tokens(3)
        .temperature(0.0)
        .generate()
        .response()
        .expect("gen failed");
    eprintln!("[WCHK] generate output: {:?}", r.text);
}
