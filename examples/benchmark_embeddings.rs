use gllm::{Client, ClientConfig, Device, Result};
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Setup
    // Use a default small model for benchmarking unless overridden
    let model = env::var("GLLM_MODEL").unwrap_or("qwen3-embedding-0.6b".to_string());
    println!("====================================================");
    println!("       GLLM EMBEDDINGS PERFORMANCE BENCHMARK        ");
    println!("====================================================");
    println!("Model:      {}", model);
    println!("OS:         {}", env::consts::OS);
    
    // Warn if GLLM_FORCE_CPU is set
    if let Ok(val) = env::var("GLLM_FORCE_CPU") {
        if val == "1" || val.to_lowercase() == "true" {
            println!("WARNING: GLLM_FORCE_CPU is set. 'Best Backend' test will likely run on CPU.");
        }
    }

    // Prepare inputs: A mix of short and long sentences
    let base_inputs = vec![
        "The quick brown fox jumps over the lazy dog.",
        "GLLM is a high-performance, local inference library for Large Language Models built in Rust.",
        "Deep learning models require significant computational resources for training and inference.",
        "Benchmarking is crucial for optimizing software performance.",
        "A vector embedding is a numerical representation of text semantics.",
        "Parallel computing leverages multiple processing elements to solve problems faster.",
        "Tokio is an asynchronous runtime for the Rust programming language.",
        "Cross-encoder rerankers provide higher accuracy than bi-encoders but are slower.",
    ];
    
    // Scale up to a reasonable batch size (e.g., 128)
    let batch_size = 128;
    let inputs: Vec<String> = base_inputs.iter().cycle().take(batch_size).map(|&s| s.to_string()).collect();
    println!("Batch Size: {}", inputs.len());
    println!("----------------------------------------------------");

    // 2. CPU Benchmark
    println!("\n[1/2] Running CPU Benchmark...");
    let mut cpu_config = ClientConfig::default();
    cpu_config.device = Device::Cpu;
    
    let start_load = Instant::now();
    let client_cpu = Client::with_config(&model, cpu_config).await?;
    let load_time_cpu = start_load.elapsed();
    println!("  Loaded in: {:.2?}", load_time_cpu);

    let start_infer = Instant::now();
    let resp_cpu = client_cpu.embeddings(&inputs).generate().await?;
    let infer_time_cpu = start_infer.elapsed();
    println!("  Inference: {:.2?}", infer_time_cpu);
    let tps_cpu = inputs.len() as f64 / infer_time_cpu.as_secs_f64();
    println!("  Throughput: {:.2} items/s", tps_cpu);

    // Verify output briefly
    if let Some(emb) = resp_cpu.embeddings.first() {
        println!("  Output Dim: {}", emb.embedding.len());
    }

    // 3. Best Backend Benchmark (GPU/Metal/AVX etc.)
    println!("\n[2/2] Running Best Backend (Auto)...");
    let mut gpu_config = ClientConfig::default();
    gpu_config.device = Device::Auto;

    let start_load = Instant::now();
    let client_gpu = Client::with_config(&model, gpu_config).await?;
    let load_time_gpu = start_load.elapsed();
    println!("  Loaded in: {:.2?}", load_time_gpu);

    let start_infer = Instant::now();
    let _ = client_gpu.embeddings(&inputs).generate().await?;
    let infer_time_gpu = start_infer.elapsed();
    println!("  Inference: {:.2?}", infer_time_gpu);
    let tps_gpu = inputs.len() as f64 / infer_time_gpu.as_secs_f64();
    println!("  Throughput: {:.2} items/s", tps_gpu);

    // 4. Comparison
    println!("\n----------------------------------------------------");
    println!("COMPARISON RESULTS");
    println!("----------------------------------------------------");
    let speedup = infer_time_cpu.as_secs_f64() / infer_time_gpu.as_secs_f64();
    println!("CPU Time:    {:.4} s", infer_time_cpu.as_secs_f64());
    println!("Best Time:   {:.4} s", infer_time_gpu.as_secs_f64());
    println!("Speedup:     {:.2}x", speedup);
    
    if speedup > 0.95 && speedup < 1.05 {
        println!("(Note: Performance is similar. GPU acceleration might not be active or available.)");
    } else if speedup > 1.0 {
        println!("Result: Best Backend is {:.2}x FASTER than CPU", speedup);
    } else {
        println!("Result: CPU is {:.2}x FASTER (Unexpected for large models)", 1.0/speedup);
    }

    Ok(())
}
