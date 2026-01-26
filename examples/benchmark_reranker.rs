use gllm::{Client, ClientConfig, Device, Result};
use std::env;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Setup
    let model = env::var("GLLM_MODEL").unwrap_or("qwen3-reranker-0.6b".to_string());
    println!("====================================================");
    println!("       GLLM RERANKER PERFORMANCE BENCHMARK          ");
    println!("====================================================");
    println!("Model:      {}", model);
    
    if let Ok(val) = env::var("GLLM_FORCE_CPU") {
        if val == "1" || val.to_lowercase() == "true" {
            println!("WARNING: GLLM_FORCE_CPU is set. 'Best Backend' test will likely run on CPU.");
        }
    }

    let query = "What are the benefits of using Rust for machine learning infrastructure?";
    let base_docs = vec![
        "Rust offers memory safety without garbage collection, making it ideal for high-performance applications.",
        "Python is the most popular language for data science due to its rich ecosystem of libraries.",
        "Machine learning models are often deployed using containerization technologies like Docker.",
        "GLLM leverages Rust's ownership model to ensure thread safety during parallel inference.",
        "Hardware acceleration via CUDA or Metal significantly speeds up matrix operations.",
    ];
    
    // Create a larger batch of documents
    let batch_size = 100;
    let docs: Vec<String> = base_docs.iter().cycle().take(batch_size).map(|&s| s.to_string()).collect();
    println!("Query:      \"{}\"", query);
    println!("Docs Batch: {}", docs.len());
    println!("----------------------------------------------------");

    // 2. CPU Benchmark
    println!("\n[1/2] Running CPU Benchmark...");
    let mut cpu_config = ClientConfig::default();
    cpu_config.device = Device::Cpu;
    
    let client_cpu = Client::with_config(&model, cpu_config).await?;
    
    let start_infer = Instant::now();
    let resp_cpu = client_cpu.rerank(query, &docs).generate().await?;
    let infer_time_cpu = start_infer.elapsed();
    println!("  Inference: {:.2?}", infer_time_cpu);
    let tps_cpu = docs.len() as f64 / infer_time_cpu.as_secs_f64();
    println!("  Throughput: {:.2} pairs/s", tps_cpu);
    
    if let Some(res) = resp_cpu.results.first() {
        println!("  Top Score: {:.4} (Index {})", res.score, res.index);
    }

    // 3. Best Backend Benchmark
    println!("\n[2/2] Running Best Backend (Auto)...");
    let mut gpu_config = ClientConfig::default();
    gpu_config.device = Device::Auto;

    let client_gpu = Client::with_config(&model, gpu_config).await?;

    let start_infer = Instant::now();
    let _ = client_gpu.rerank(query, &docs).generate().await?;
    let infer_time_gpu = start_infer.elapsed();
    println!("  Inference: {:.2?}", infer_time_gpu);
    let tps_gpu = docs.len() as f64 / infer_time_gpu.as_secs_f64();
    println!("  Throughput: {:.2} pairs/s", tps_gpu);

    // 4. Comparison
    println!("\n----------------------------------------------------");
    println!("COMPARISON RESULTS");
    println!("----------------------------------------------------");
    let speedup = infer_time_cpu.as_secs_f64() / infer_time_gpu.as_secs_f64();
    println!("CPU Time:    {:.4} s", infer_time_cpu.as_secs_f64());
    println!("Best Time:   {:.4} s", infer_time_gpu.as_secs_f64());
    println!("Speedup:     {:.2}x", speedup);

    Ok(())
}
