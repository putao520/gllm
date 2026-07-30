use gllm::loader::gguf::GgufReader;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let gguf = GgufReader::open(&args[1])?;
    for t in gguf.tensors() {
        let n = t.name.as_ref();
        if n.contains("norm") || n.contains("embd") || n.contains("embed") || n.contains("q_proj") || n.contains("k_proj") || n.contains("v_proj") || n.contains("o_proj") || n.contains("output") || n == "token_embd.weight" {
            println!("{}: dtype={:?} shape={:?}", n, t.dtype, t.shape);
        }
    }
    Ok(())
}
