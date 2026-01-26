use gllm::Client;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Investigating Performance for smollm2-135m-instruct ===");

    let model_id = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let prompt = "Explain quantum entanglement in simple terms.";

    // 1. CPU Benchmark
    println!("\n[1/2] Testing CPU Backend...");
    std::env::set_var("GLLM_FORCE_CPU", "1");
    let client = Client::new(model_id)?;
    
    let start = Instant::now();
    let response = client.generate(prompt).run()?;
    let duration = start.elapsed();
    let tokens = response.tokens.len();
    let speed = tokens as f64 / duration.as_secs_f64();
    
    println!("  CPU Tokens: {}, Time: {:.2}s, Speed: {:.2} t/s", tokens, duration.as_secs_f64(), speed);

    // 2. GPU Benchmark
    println!("\n[2/2] Testing GPU Backend...");
    std::env::remove_var("GLLM_FORCE_CPU");
    // Ensure WGPU backend is selected if available
    std::env::set_var("GLLM_BACKEND", "wgpu"); 
    
    let client = Client::new(model_id)?;
    let start = Instant::now();
    let response = client.generate(prompt).run()?;
    let duration = start.elapsed();
    let tokens = response.tokens.len();
    let speed = tokens as f64 / duration.as_secs_f64();
    
    println!("  GPU Tokens: {}, Time: {:.2}s, Speed: {:.2} t/s", tokens, duration.as_secs_f64(), speed);

    Ok(())
}
