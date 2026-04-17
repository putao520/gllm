//! 跑 BGE-reranker-v2-m3 的单次 rerank_pair 并 dump 所有 node output。
//!
//! 配合 PyTorch 参考 dump 做逐层数值对比。
//!
//! 用法:
//!     GLLM_DUMP_LAYERS=/tmp/gllm_dump cargo run --example rerank_dump -- \
//!         "What is the capital of France?" "Paris is the capital of France."

use gllm::{Client, ModelKind};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("用法: rerank_dump <query> <doc>");
        std::process::exit(1);
    }
    let query = &args[1];
    let doc = &args[2];

    let client = Client::new("BAAI/bge-reranker-v2-m3", ModelKind::Reranker)
        .expect("load model");
    let response = client.rerank(query, vec![doc.clone()])
        .expect("rerank");
    let score = response.results[0].score;
    println!("JIT score = {score}");
}
