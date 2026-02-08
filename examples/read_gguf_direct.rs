// Direct GGUF reader using gguf crate
use gllm::loader::gguf::GgufReader;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Find the GGUF file
    let gguf_path = Path::new("/home/putao/.gllm/models/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf");

    println!("Reading: {:?}", gguf_path);

    let gguf = GgufReader::open(gguf_path)?;

    // Print architecture
    if let Ok(arch) = gguf.architecture() {
        println!("Architecture: {:?}", arch);
    }

    // Print key metadata
    println!("\n=== Key Metadata ===");
    if let Some(emb_len) = gguf.embedding_length() {
        println!("embedding_length (hidden_size): {:?}", emb_len);
    }
    if let Some(head_count) = gguf.head_count() {
        println!("head_count (num_attention_heads): {:?}", head_count);
    }
    if let Some(head_count_kv) = gguf.head_count_kv() {
        println!("head_count_kv (num_kv_heads): {:?}", head_count_kv);
    }
    if let Some(block_count) = gguf.block_count() {
        println!("block_count (num_layers): {:?}", block_count);
    }

    // Calculate derived values
    let hidden_size = gguf.embedding_length().unwrap_or(0);
    let num_heads = gguf.head_count().unwrap_or(0);
    let rope_dim = gguf.rope_dimension_count();

    println!("\n=== Derived Values ===");
    println!("rope_dimension_count: {:?}", rope_dim);
    if rope_dim.is_some() {
        let head_dim = rope_dim.unwrap();
        println!("  head_dim from rope_dimension_count: {}", head_dim);
        println!("  calculated hidden_size: {} * {} = {}", num_heads, head_dim, num_heads * head_dim);
    } else {
        let head_dim = if num_heads > 0 { hidden_size / num_heads } else { 0 };
        println!("  head_dim derived (hidden/num_heads): {} / {} = {}", hidden_size, num_heads, head_dim);
        println!("  calculated hidden_size: {} * {} = {}", num_heads, head_dim, num_heads * head_dim);
    }

    // Print tensor info for first layer
    println!("\n=== Layer 0 MLP tensors ===");
    for tensor in gguf.tensors().iter() {
        let name = &tensor.name;
        if name.contains("blk.0") && (name.contains("ffn") || name.contains("mlp")) {
            println!("{}: shape={:?}", name, tensor.shape);
        }
    }

    println!("\n=== Layer 0 Attention tensors ===");
    for tensor in gguf.tensors().iter() {
        let name = &tensor.name;
        if name.contains("blk.0") && name.contains("attn") {
            println!("{}: shape={:?}", name, tensor.shape);
        }
    }

    // Print embedding tensor
    println!("\n=== Embedding tensors ===");
    for tensor in gguf.tensors().iter() {
        let name = &tensor.name;
        if name.contains("token_embd") {
            println!("{}: shape={:?}", name, tensor.shape);
        }
    }

    Ok(())
}
