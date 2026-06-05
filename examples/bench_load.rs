//! Benchmark: Client::new_chat + generate latency breakdown + second run.
use std::time::Instant;

fn main() {
    // Phase 1: Load
    eprintln!("[BENCH] start new_chat...");
    let t0 = Instant::now();
    let client = gllm::Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")
        .expect("load failed");
    let load_ms = t0.elapsed().as_millis();
    eprintln!("[BENCH] new_chat done: {load_ms}ms");

    // Phase 2: First generate
    eprintln!("[BENCH] start generate #1 (20 tokens)...");
    let t1 = Instant::now();
    let resp1 = client
        .generate("What is 1+1?")
        .max_tokens(20)
        .temperature(0.7)
        .generate();
    let gen1_ms = t1.elapsed().as_millis();
    let text1 = resp1.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH] generate #1 done: {gen1_ms}ms, text='{}'", text1.chars().take(80).collect::<String>());

    // Phase 3: Second generate (should reuse compiled executor)
    eprintln!("[BENCH] start generate #2 (20 tokens)...");
    let t2 = Instant::now();
    let resp2 = client
        .generate("What is 2+2?")
        .max_tokens(20)
        .temperature(0.7)
        .generate();
    let gen2_ms = t2.elapsed().as_millis();
    let text2 = resp2.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH] generate #2 done: {gen2_ms}ms, text='{}'", text2.chars().take(80).collect::<String>());

    // Phase 4: Short prompt generate
    eprintln!("[BENCH] start generate #3 (5 tokens)...");
    let t3 = Instant::now();
    let resp3 = client
        .generate("Hi")
        .max_tokens(5)
        .temperature(0.0)
        .generate();
    let gen3_ms = t3.elapsed().as_millis();
    let text3 = resp3.response().map(|r| r.text.clone()).unwrap_or_default();
    eprintln!("[BENCH] generate #3 done: {gen3_ms}ms, text='{}'", text3.chars().take(80).collect::<String>());

    eprintln!("[BENCH] SUMMARY: load={load_ms}ms gen1={gen1_ms}ms gen2={gen2_ms}ms gen3={gen3_ms}ms");
    eprintln!("[BENCH] per-token: gen1={:.0}ms gen2={:.0}ms gen3={:.0}ms",
        gen1_ms as f64 / 20.0, gen2_ms as f64 / 20.0, gen3_ms as f64 / 5.0);
}
