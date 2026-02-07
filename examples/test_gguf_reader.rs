use gguf::GgufReader;

fn main() {
    let path = "/home/putao/.gllm/models/Mungert--SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-bf16.gguf";

    let file = std::fs::File::open(path).expect("open file");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };

    let reader = GgufReader::new(&mmap).expect("new reader");

    println!("Version: {}", reader.version());
    println!("Tensor count: {}", reader.tensor_count());
    println!("KV count: {}", reader.metadata_count());

    println!("\n=== KV Metadata ===");
    for (i, (key, value)) in reader.metadata().enumerate() {
        println!("{}: key='{}', value={:?}", i, key, value);
        if i >= 20 {
            break;
        }
    }
}
