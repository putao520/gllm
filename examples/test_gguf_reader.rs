use gllm::loader::gguf::GgufReader;
use std::env;

fn main() {
    let path = env::args().nth(1).expect("provide GGUF path");
    let reader = GgufReader::open(&path).expect("open reader");

    println!("Version: {}", reader.version());
    println!("Tensor count: {}", reader.tensor_count());
    println!("KV count: {}", reader.kv_count());

    println!("\n=== KV Metadata ===");
    for (i, (key, value)) in reader.metadata().iter().enumerate() {
        println!("{}: key='{}', value={:?}", i, key, value);
        if i >= 45 {
            break;
        }
    }
}
