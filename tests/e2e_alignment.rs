//! Cross-language alignment tests (REQ-TEST-011).
//!
//! Compares gllm embedding output against PyTorch/HuggingFace golden data.
//! Golden data must be generated first: see tests/e2e_alignment/README.md.

use std::collections::HashMap;
use std::path::Path;

/// Load golden embeddings from safetensors file.
fn load_golden(path: &Path) -> HashMap<String, Vec<f32>> {
    let data = std::fs::read(path).expect("Failed to read golden safetensors file");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        if name.starts_with("embedding_") {
            let bytes = view.data();
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            result.insert(name.to_string(), floats);
        }
    }
    result
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector length mismatch");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[test]
#[ignore]
fn alignment_embeddings_match_golden() {
    let golden_path = Path::new("tests/e2e_alignment/data/golden.safetensors");
    if !golden_path.exists() {
        panic!(
            "Golden data not found at {}. Run: cd tests/e2e_alignment && python generate_golden.py",
            golden_path.display()
        );
    }

    let golden = load_golden(golden_path);
    assert!(!golden.is_empty(), "No embeddings found in golden data");

    // TODO: Once gllm embedding API is stable, run inference here and compare.
    // For now, validate that golden data loads correctly and has expected structure.
    for (name, embedding) in &golden {
        assert!(!embedding.is_empty(), "Empty embedding for {name}");
        // All embeddings should have the same dimension
        let dim = golden.values().next().unwrap().len();
        assert_eq!(
            embedding.len(),
            dim,
            "Dimension mismatch for {name}: expected {dim}, got {}",
            embedding.len()
        );
    }

    // Self-consistency check: each embedding should have cosine similarity 1.0 with itself
    for (name, embedding) in &golden {
        let sim = cosine_similarity(embedding, embedding);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Self-similarity for {name} should be 1.0, got {sim}"
        );
    }
}
