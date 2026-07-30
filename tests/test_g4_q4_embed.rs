#![cfg(test)]
use gllm::{Client, ModelKind};
#[test]
#[ignore]
fn g4_q4_embed_only() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let client = Client::builder()
        .model("unsloth/gemma-4-E2B-it-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("Q4_0")
        .build()
        .unwrap_or_else(|e| panic!("load: {:?}", e));
    let toks = client.encode("The").unwrap();
    eprintln!("tokens={:?}", toks);
    if let Some(sp) = client.diagnostic_prefill_scratchpad(&toks) {
        // embed output
        if let Some((_, off, _)) = sp.named_offsets.iter().find(|(n,_,_)| n == "embed") {
            let v = sp.read_dtype_aware(*off, 16);
            eprintln!("embed first16={:?}", v);
            let bad = v.iter().filter(|x| x.is_infinite() || x.is_nan()).count();
            eprintln!("embed bad_count={}", bad);
        }
        // input_norm output
        if let Some((_, off, _)) = sp.named_offsets.iter().find(|(n,_,_)| n == "L0.input_norm") {
            let v = sp.read_dtype_aware(*off, 16);
            eprintln!("input_norm first16={:?}", v);
        }
    }
    eprintln!("DONE");
}
