use gllm::Client;
#[test]
#[ignore]
fn g4_logits_full() {
    let client = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q4_0.gguf").expect("load");
    let tokens = client.encode("The capital of France is").expect("encode");
    eprintln!("tokens={tokens:?}");
    let logits = client.diagnostic_prefill_logits(&tokens).expect("logits");
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let inf = logits.iter().filter(|v| v.is_infinite()).count();
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (idx,val) = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,v)|(i,*v)).unwrap();
    eprintln!("logits len={} nan={} inf={} min={:.4} max={:.4} argmax={} val={:.4}", logits.len(), nan, inf, min, max, idx, val);
}
