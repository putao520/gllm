//! Minimal bench: load + 1-token generate timing via public API.
use std::time::Instant;

fn main() {
    eprintln!("[BENCH-P1] load...");
    let t0 = Instant::now();
    let client = gllm::Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")
        .expect("load");
    eprintln!("[BENCH-P1] load: {}ms", t0.elapsed().as_millis());

    // 1-token generate to measure prefill + 1 decode
    eprintln!("[BENCH-P2] generate 1 token...");
    let t1 = Instant::now();
    let resp = client.generate("Hi").max_tokens(1).temperature(0.0).generate();
    let ms1 = t1.elapsed().as_millis();
    let text1 = resp.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH-P2] 1-token: {}ms text='{}'", ms1, text1);

    // 1-token generate again to measure pure decode (KV cached)
    eprintln!("[BENCH-P3] generate 1 token (cached)...");
    let t2 = Instant::now();
    let resp2 = client.generate("Hello").max_tokens(1).temperature(0.0).generate();
    let ms2 = t2.elapsed().as_millis();
    let text2 = resp2.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH-P3] 1-token cached: {}ms text='{}'", ms2, text2);

    // 5-token generate
    eprintln!("[BENCH-P4] generate 5 tokens...");
    let t3 = Instant::now();
    let resp3 = client.generate("What is 2+2?").max_tokens(5).temperature(0.0).generate();
    let ms3 = t3.elapsed().as_millis();
    let text3 = resp3.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH-P4] 5-token: {}ms text='{}' ({}ms/token)", ms3, text3, ms3 / 5);
}
