#![cfg(test)]
use gllm::loader::gguf::GgufReader;
use std::collections::BTreeMap;
#[test]
#[ignore]
fn inspect_q6k_tensor_types() {
    for fname in &["Qwen_Qwen3-0.6B-Q6_K.gguf", "Qwen_Qwen3-0.6B-Q6_K_L.gguf"] {
        let path = format!("/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/{}", fname);
        if !std::path::Path::new(&path).exists() { eprintln!("{}: 不存在", fname); continue; }
        let r = GgufReader::open(&path).expect("open");
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut samples: BTreeMap<String, String> = BTreeMap::new();
        for t in r.tensors() {
            let dn = format!("{:?}", t.dtype);
            *counts.entry(dn.clone()).or_insert(0) += 1;
            samples.entry(dn.clone()).or_insert_with(|| t.name.to_string());
        }
        eprintln!("{}: {:?}", fname, counts);
        eprintln!("  samples: {:?}", samples);
    }
}
