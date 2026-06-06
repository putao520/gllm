use gllm::GgufReader;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_gemma4_meta <file.gguf>");
    let reader = GgufReader::open(&path).expect("failed to open GGUF");

    let arch = reader.architecture_name().unwrap_or("unknown");
    println!("architecture: {}", arch);
    println!("block_count: {:?}", reader.block_count());
    println!("embedding_length: {:?}", reader.embedding_length());
    println!("head_count: {:?}", reader.head_count());
    println!("head_count_kv: {:?}", reader.head_count_kv());
    println!("context_length: {:?}", reader.context_length());
    println!("rope_freq_base: {:?}", reader.rope_freq_base());
    println!("rope_dimension_count: {:?}", reader.rope_dimension_count());

    let keys = [
        format!("{}.attention.sliding_window", arch),
        format!("{}.attention.num_kv_shared_layers", arch),
        format!("{}.rope.global.freq_base", arch),
        format!("{}.rope.partial_ratio", arch),
        format!("{}.embedding.per_layer_input", arch),
        format!("{}.final_logit_softcapping", arch),
        format!("{}.attention.pattern", arch),
        format!("{}.feed_forward_length", arch),
        format!("{}.attention.layer_norm_rms_epsilon", arch),
        format!("{}.attention.key_length", arch),
        format!("{}.attention.value_length", arch),
    ];

    println!("\n--- Gemma 4 specific ---");
    for key in &keys {
        if let Some(v) = reader.get_metadata_str(key) {
            println!("{} (str): {}", key, v);
        }
        if let Some(v) = reader.get_metadata_u64(key) {
            println!("{} (u64): {}", key, v);
        }
        if let Some(v) = reader.get_metadata_f32(key) {
            println!("{} (f32): {}", key, v);
        }
    }

    // Print tensor names
    println!("\n--- Tensor names (first 30) ---");
    for (i, t) in reader.tensors().iter().take(30).enumerate() {
        println!("{}: {} dtype={:?} shape={:?}", i, t.name, t.dtype, t.shape);
    }
}
