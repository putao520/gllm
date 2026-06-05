//! 跑 SmolLM2 的一次 prefill generate 并 dump 所有 node output。
//! 配合 PyTorch 参考 dump 做逐层数值对比 (tools/pytorch_ref_smollm2.py)。
//!
//! 用法: GLLM_DUMP_LAYERS=/tmp/gllm_dump cargo run --example gen_dump -- "The capital of France is"

use gllm::Client;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: gen_dump <prompt>");
        std::process::exit(1);
    }
    let prompt = &args[1];

    let model_id = std::env::var("GLLM_GEN_MODEL")
        .unwrap_or_else(|_| "HuggingFaceTB/SmolLM2-135M-Instruct".to_string());
    let client = Client::new_chat(&model_id).expect("load model");

    let resp = client.generate(prompt)
        .max_tokens(1)
        .temperature(0.0)
        .generate()
        .response()
        .expect("generation failed");

    println!("Generated: {:?}", resp.text);
}
