use gllm::{Client, ClientConfig, Device, Result};
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Setup
    let model = env::var("GLLM_MODEL").unwrap_or("qwen3-next-0.6b".to_string());
    println!("====================================================");
    println!("       GLLM GENERATOR PERFORMANCE BENCHMARK         ");
    println!("====================================================");
    println!("Model:      {}", model);
    
    if let Ok(val) = env::var("GLLM_FORCE_CPU") {
        if val == "1" || val.to_lowercase() == "true" {
            println!("WARNING: GLLM_FORCE_CPU is set. 'Best Backend' test will likely run on CPU.");
        }
    }

    let prompt = "Explain the theory of relativity to a 5 year old in simple terms.";
    let max_new_tokens = 100;
    
    println!("Prompt:     \"{}\"", prompt);
    println!("Gen Len:    {} tokens", max_new_tokens);
    println!("----------------------------------------------------");

    // 2. CPU Benchmark
    println!("\n[1/2] Running CPU Benchmark...");
    let mut cpu_config = ClientConfig::default();
    cpu_config.device = Device::Cpu;
    
    let client_cpu = Client::with_config(&model, cpu_config).await?;
    
    let start_gen = Instant::now();
    let resp_cpu = client_cpu.generate(prompt)
        .max_new_tokens(max_new_tokens)
        .generate().await?;
    let gen_time_cpu = start_gen.elapsed();
    
    let tokens_generated = resp_cpu.tokens.len().saturating_sub(client_cpu.tokenizer.encode_unpadded(prompt, 2048).len());
    let tps_cpu = tokens_generated as f64 / gen_time_cpu.as_secs_f64();
    
    println!("  Time:       {:.2?}", gen_time_cpu);
    println!("  Tokens:     {}", tokens_generated);
    println!("  Speed:      {:.2} tokens/s", tps_cpu);
    println!("  Output:     {}...", resp_cpu.text.chars().take(40).collect::<String>());

    // 3. Best Backend Benchmark
    println!("\n[2/2] Running Best Backend (Auto)...");
    let mut gpu_config = ClientConfig::default();
    gpu_config.device = Device::Auto;

    let client_gpu = Client::with_config(&model, gpu_config).await?;

    let start_gen = Instant::now();
    let resp_gpu = client_gpu.generate(prompt)
        .max_new_tokens(max_new_tokens)
        .generate().await?;
    let gen_time_gpu = start_gen.elapsed();
    
    let tokens_generated_gpu = resp_gpu.tokens.len().saturating_sub(client_gpu.tokenizer.encode_unpadded(prompt, 2048).len());
    let tps_gpu = tokens_generated_gpu as f64 / gen_time_gpu.as_secs_f64();
    
    println!("  Time:       {:.2?}", gen_time_gpu);
    println!("  Tokens:     {}", tokens_generated_gpu);
    println!("  Speed:      {:.2} tokens/s", tps_gpu);
    println!("  Output:     {}...", resp_gpu.text.chars().take(40).collect::<String>());

    // 4. Comparison
    println!("\n----------------------------------------------------");
    println!("COMPARISON RESULTS");
    println!("----------------------------------------------------");
    let speedup = tps_gpu / tps_cpu;
    println!("CPU Speed:   {:.2} t/s", tps_cpu);
    println!("Best Speed:  {:.2} t/s", tps_gpu);
    println!("Speedup:     {:.2}x", speedup);

    Ok(())
}
