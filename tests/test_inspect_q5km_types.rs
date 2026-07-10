//! Inspect Q5_K_M tensor types via GgufReader.
#![cfg(test)]
use std::collections::BTreeMap;
use gllm::loader::gguf::GgufReader;

#[test]
#[ignore]
fn inspect_q5km_tensor_types() {
    let path = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
    let r = GgufReader::open(path).expect("open");
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut samples: BTreeMap<String, String> = BTreeMap::new();
    for t in r.tensors() {
        let dn = format!("{:?}", t.dtype);
        *counts.entry(dn.clone()).or_insert(0) += 1;
        samples.entry(dn.clone()).or_insert_with(|| t.name.as_ref().to_string());
    }
    eprintln!("Q5_K_M tensor type counts: {:?}", counts);
    eprintln!("samples: {:?}", samples);
}

#[test]
#[ignore]
fn inspect_q5km_q6k_tensors() {
    let path = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
    let r = GgufReader::open(path).expect("open");
    let q6k: Vec<String> = r.tensors().iter()
        .filter(|t| format!("{:?}", t.dtype) == "Q6_K")
        .map(|t| t.name.as_ref().to_string()).collect();
    eprintln!("Q6K tensors ({}): first 10 = {:?}", q6k.len(), &q6k[..q6k.len().min(10)]);
    let q5k: Vec<String> = r.tensors().iter()
        .filter(|t| format!("{:?}", t.dtype) == "Q5_K")
        .map(|t| t.name.as_ref().to_string()).collect();
    eprintln!("Q5K tensors ({}): first 10 = {:?}", q5k.len(), &q5k[..q5k.len().min(10)]);
}
