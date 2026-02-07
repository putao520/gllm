//! gllm model downloader CLI
//!
//! Usage: cargo run --bin download -- <repo_id>
//! Example: cargo run --bin download -- Qwen/Qwen3-7B

use std::path::PathBuf;

use gllm::loader::{hf_hub::HfHubClient, ParallelLoader};
use gllm::EMPTY_FILE_MAP;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <repo_id>", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} Qwen/Qwen3-7B", args[0]);
        eprintln!("  {} microsoft/Phi-4-mini-instruct", args[0]);
        eprintln!("  {} Qwen/Qwen3-Embedding", args[0]);
        eprintln!("  {} BAAI/bge-m4", args[0]);
        eprintln!("  {} Qwen/Qwen3-Reranker", args[0]);
        eprintln!();
        eprintln!("Test Matrix Models:");
        eprintln!("  Generator: Qwen/Qwen3-7B, meta-llama/Llama-4-8b, microsoft/Phi-4-mini-instruct, Qwen/Qwen3-A22B");
        eprintln!("  Embedding: Qwen/Qwen3-Embedding, BAAI/bge-m4");
        eprintln!("  Reranker: Qwen/Qwen3-Reranker, BAAI/bge-reranker-v3");
        std::process::exit(1);
    }

    let repo_id = &args[1];

    // 使用统一的缓存目录: ~/.gllm/models/
    let cache_dir = PathBuf::from(std::env::var("HOME")?).join(".gllm/models");
    std::fs::create_dir_all(&cache_dir)?;

    println!("📦 下载模型: {}", repo_id);
    println!("📁 缓存目录: {}", cache_dir.display());

    let client = HfHubClient::new(cache_dir)?;
    let files = client.download_model_files(repo_id, EMPTY_FILE_MAP, ParallelLoader::new(true))?;

    println!("✅ 下载完成:");
    println!("  权重文件: {} 个", files.weights.len());
    for w in &files.weights {
        println!("    - {}", w.display());
    }
    println!("  辅助文件: {} 个", files.aux_files.len());
    for f in &files.aux_files {
        println!("    - {}", f.display());
    }

    Ok(())
}
