use gllm::Client;
#[test]
#[ignore]
fn g4_embed_val() {
    let client = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q4_0.gguf").expect("load");
    let tokens = client.encode("The capital of France is").expect("encode");
    eprintln!("tokens={:?}", tokens);
    if let Some(sp) = client.diagnostic_prefill_scratchpad(&tokens) {
        eprintln!("scratchpad bytes={}", sp.total_bytes());
        // embed output is at scratch offset 0 (first QuantGather writes to scratch+0)
        let vals = sp.read_f32_at(0, 16);
        eprintln!("embed[0..16] = {:?}", vals);
        let vals2 = sp.read_f32_at(0, 1536);
        let nan = vals2.iter().filter(|v| v.is_nan()).count();
        let inf = vals2.iter().filter(|v| v.is_infinite()).count();
        let mx = vals2.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mn = vals2.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "embed[0..1536] nan={} inf={} min={:.4} max={:.4}",
            nan, inf, mn, mx
        );
    } else {
        eprintln!("no scratchpad");
    }
}
