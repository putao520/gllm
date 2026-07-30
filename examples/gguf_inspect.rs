use gllm::loader::gguf::{GgufReader, GgufValue};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let gguf = GgufReader::open(&args[1])?;
    eprintln!("arch: {:?}", gguf.architecture()?);
    let meta = gguf.metadata();
    eprintln!("kv count: {}", meta.len());
    for (k, v) in meta.iter() {
        let kl = k.to_lowercase();
        if kl.contains("scale") || kl.contains("embed") || kl.contains("norm")
            || kl.contains("rope") || kl.contains("head") || kl.contains("length")
            || kl.contains("sliding") || kl.contains("block") || kl.contains("count")
            || kl.contains("arch") || kl.contains("value") || kl.contains("shared") {
            let vs = match v {
                GgufValue::Float32(x) => format!("f32 {x:.6}"),
                GgufValue::Uint64(x) => format!("u64 {x}"),
                GgufValue::Int64(x) => format!("i64 {x}"),
                GgufValue::Bool(x) => format!("bool {x}"),
                GgufValue::String(s) => format!("str {s}"),
                GgufValue::Array(a) => {
                    let n = a.items.len();
                    let head: Vec<String> = a.items.iter().take(10).map(|x| match x {
                        GgufValue::Bool(b) => b.to_string(),
                        GgufValue::Uint64(u) => u.to_string(),
                        _ => "?".into(),
                    }).collect();
                    format!("arr[{n}] [{}]", head.join(","))
                }
                _ => format!("other"),
            };
            eprintln!("  {k} = {vs}");
        }
    }
    // Dump tensor types for key tensors
    eprintln!("=== tensor types ===");
    for t in gguf.tensors() {
        if t.name.starts_with("blk.0.") || t.name.contains("token_embd") || t.name.contains("output") {
            eprintln!("  {} dtype={:?} shape={:?}", t.name, t.dtype, t.shape);
        }
    }
    Ok(())
}
