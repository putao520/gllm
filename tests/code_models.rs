use gllm::{EmbedderHandle, GraphCodeInput};

#[test]
fn test_codebert_base_loading_and_embedding() {
    // This test ensures codebert-base (the general fallback) works
    let handle = EmbedderHandle::new("codebert-base").expect("Failed to init codebert-base");
    let input = "fn main() { println!(\"Hello\"); }";
    let embeddings = handle.embed(input).expect("Failed to embed");
    
    assert_eq!(embeddings.len(), 768, "CodeBERT should have 768 dimensions");
    // Simple check that vector is not all zeros
    assert!(embeddings.iter().any(|&x| x != 0.0));
}

#[test]
fn test_graphcodebert_base_loading() {
    // Test the specific code structure model
    let handle = EmbedderHandle::new("graphcodebert-base").expect("Failed to init graphcodebert-base");
    let input = "def calculate_sum(a, b): return a + b";
    let embeddings = handle.embed(input).expect("Failed to embed");
    
    assert_eq!(embeddings.len(), 768, "GraphCodeBERT should have 768 dimensions");
}

#[test]
fn test_unixcoder_base_loading() {
    // Test the documentation model
    let handle = EmbedderHandle::new("unixcoder-base").expect("Failed to init unixcoder-base");
    let input = "This function calculates the sum of two numbers.";
    let embeddings = handle.embed(input).expect("Failed to embed");
    
    assert_eq!(embeddings.len(), 768, "UniXcoder should have 768 dimensions");
}

#[test]
fn test_concurrent_model_loading() {
    // Simulate the scenario of loading multiple models simultaneously to check resource contention
    // and thread safety of the backend logic.
    
    let t1 = std::thread::spawn(|| {
        let handle = EmbedderHandle::new("codebert-base").expect("Thread 1: codebert init failed");
        let _ = handle.embed("test 1").expect("Thread 1: embed failed");
        true
    });

    let t2 = std::thread::spawn(|| {
        let handle = EmbedderHandle::new("graphcodebert-base").expect("Thread 2: graphcodebert init failed");
        let _ = handle.embed("test 2").expect("Thread 2: embed failed");
        true
    });

    let t3 = std::thread::spawn(|| {
        let handle = EmbedderHandle::new("unixcoder-base").expect("Thread 3: unixcoder init failed");
        let _ = handle.embed("test 3").expect("Thread 3: embed failed");
        true
    });

    assert!(t1.join().expect("Thread 1 panicked"));
    assert!(t2.join().expect("Thread 2 panicked"));
    assert!(t3.join().expect("Thread 3 panicked"));
}

#[test]
fn test_graph_input_fallback() {
    // Verify that EmbedGraphBatch falls back to text embedding correctly (as per current implementation)
    let handle = EmbedderHandle::new("graphcodebert-base").expect("Init failed");
    
    let input = GraphCodeInput {
        code: "fn test() {}".to_string(),
        position_ids: None,
        dfg_mask: None,
    };
    
    let batch = vec![input];
    let embeddings = handle.embed_graph_batch(batch).expect("Graph batch embed failed");
    
    assert_eq!(embeddings.len(), 1);
    assert_eq!(embeddings[0].len(), 768);
}
