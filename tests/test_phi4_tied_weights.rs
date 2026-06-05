//! Quick test: verify lm_head == embed for tied weights
use gllm::Client;

#[test]
fn phi4_tied_weight_check() {
    let model = "microsoft/Phi-4-mini-instruct";
    let client = Client::new_chat(model).expect("load");
    
    let embed_row = client.diagnostic_weight_row("embed", 0, 20).expect("embed row");
    let lm_head_row = client.diagnostic_weight_row("lm_head", 0, 20).expect("lm_head row");
    
    eprintln!("[TIED] embed[0..10]: {:?}", &embed_row[..10]);
    eprintln!("[TIED] lm_head[0..10]: {:?}", &lm_head_row[..10]);
    
    let match_count = embed_row.iter().zip(lm_head_row.iter()).filter(|(a, b)| a == b).count();
    eprintln!("[TIED] match {}/{} values", match_count, embed_row.len().min(lm_head_row.len()));
    
    // Check a middle row too
    let embed_mid = client.diagnostic_weight_row("embed", 100000, 5).expect("embed mid");
    let lm_head_mid = client.diagnostic_weight_row("lm_head", 100000, 5).expect("lm_head mid");
    eprintln!("[TIED] embed[100000..]: {:?}", &embed_mid[..5]);
    eprintln!("[TIED] lm_head[100000..]: {:?}", &lm_head_mid[..5]);
    let mid_match = embed_mid.iter().zip(lm_head_mid.iter()).filter(|(a, b)| a == b).count();
    eprintln!("[TIED] mid match {}/{}", mid_match, embed_mid.len().min(lm_head_mid.len()));
    
    // Also check total weight blob size
    if let Some(offsets) = client.diagnostic_weight_offsets() {
        let max_offset = offsets.iter().map(|(_, off)| *off).max().unwrap_or(0);
        eprintln!("[TIED] Max offset: {} ({:.2} GB)", max_offset, max_offset as f64 / 1e9);
        eprintln!("[TIED] Total entries: {}", offsets.len());
    }
    
    assert_eq!(match_count, embed_row.len().min(lm_head_row.len()), "embed and lm_head first row should match");
}
