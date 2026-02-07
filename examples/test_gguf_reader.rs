use gllm::loader::gguf::GgufReader;

fn main() {
    let path = "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf";
    let reader = GgufReader::open(path).expect("open reader");

    println!("Version: {}", reader.version());
    println!("Tensor count: {}", reader.tensor_count());
    println!("KV count: {}", reader.kv_count());

    println!("\n=== KV Metadata ===");
    for (i, (key, value)) in reader.metadata().iter().enumerate() {
        println!("{}: key='{}', value={:?}", i, key, value);
        if i >= 20 {
            break;
        }
    }
}
