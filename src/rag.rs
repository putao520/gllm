//! §16 P1: Late-Fusion RAG
//!
//! Fuses retrieval-augmented information at residual connection points.

/// Late-Fusion RAG configuration
#[derive(Debug, Clone, PartialEq)]
pub struct LateFusionRag {
    pub retrieval_db: Vec<Vec<f32>>,
    pub fusion_layer: usize,
    pub top_k: usize,
    pub fusion_weight: f32,
}

impl LateFusionRag {
    pub fn new(fusion_layer: usize) -> Self {
        Self {
            retrieval_db: Vec::new(),
            fusion_layer,
            top_k: 3,
            fusion_weight: 0.1,
        }
    }

    pub fn retrieve(&self, query: &[f32]) -> Vec<&[f32]> {
        let mut scores: Vec<_> = self
            .retrieval_db
            .iter()
            .map(|doc| (doc.as_slice(), cosine_similarity(query, doc)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter().take(self.top_k).map(|(doc, _)| doc).collect()
    }

    pub fn fuse_at_residual(&self, hidden_state: &mut [f32], layer: usize) {
        if layer != self.fusion_layer || self.retrieval_db.is_empty() {
            return;
        }

        let retrieved = self.retrieve(hidden_state);
        for doc in retrieved {
            for i in 0..hidden_state.len().min(doc.len()) {
                hidden_state[i] += doc[i] * self.fusion_weight;
            }
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let dot: f32 = a[..len].iter().zip(&b[..len]).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_late_fusion_rag() {
        let mut rag = LateFusionRag::new(2);
        rag.retrieval_db = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.1;

        let mut hidden_state = vec![0.5, 0.5, 0.0];
        rag.fuse_at_residual(&mut hidden_state, 2);

        assert!((hidden_state[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rag_retrieve_top_k() {
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
        ];
        rag.top_k = 2;
        let results = rag.retrieve(&[1.0, 0.0]);
        assert_eq!(results.len(), 2);
        // First result should be [1,0] (exact match, highest similarity)
        assert!((results[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rag_fuse_wrong_layer_noop() {
        let mut rag = LateFusionRag::new(5);
        rag.retrieval_db = vec![vec![1.0, 0.0]];
        let mut state = vec![0.5, 0.5];
        rag.fuse_at_residual(&mut state, 3);
        assert_eq!(state, vec![0.5, 0.5]); // unchanged
    }

    #[test]
    fn test_rag_fuse_empty_db_noop() {
        let rag = LateFusionRag::new(1);
        let mut state = vec![1.0, 2.0];
        rag.fuse_at_residual(&mut state, 1);
        assert_eq!(state, vec![1.0, 2.0]);
    }

    #[test]
    fn test_rag_new_defaults() {
        let rag = LateFusionRag::new(3);
        assert_eq!(rag.fusion_layer, 3);
        assert_eq!(rag.top_k, 3);
        assert!((rag.fusion_weight - 0.1).abs() < 1e-6);
        assert!(rag.retrieval_db.is_empty());
    }

    // --- Struct construction, field access, PartialEq ---

    #[test]
    fn test_struct_equality() {
        let a = LateFusionRag::new(2);
        let b = LateFusionRag::new(2);
        assert_eq!(a, b);
    }

    #[test]
    fn test_struct_inequality_different_fusion_layer() {
        let a = LateFusionRag::new(1);
        let b = LateFusionRag::new(2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_struct_clone() {
        let original = LateFusionRag::new(4);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_struct_debug_format() {
        let rag = LateFusionRag::new(1);
        let debug_str = format!("{:?}", rag);
        assert!(debug_str.contains("LateFusionRag"));
        assert!(debug_str.contains("fusion_layer: 1"));
        assert!(debug_str.contains("top_k: 3"));
        assert!(debug_str.contains("fusion_weight: 0.1"));
    }

    #[test]
    fn test_new_with_zero_fusion_layer() {
        let rag = LateFusionRag::new(0);
        assert_eq!(rag.fusion_layer, 0);
    }

    #[test]
    fn test_new_with_max_fusion_layer() {
        let rag = LateFusionRag::new(usize::MAX);
        assert_eq!(rag.fusion_layer, usize::MAX);
    }

    // --- cosine_similarity edge cases ---

    #[test]
    fn test_cosine_similarity_single_element() {
        let a = vec![3.0];
        let b = vec![4.0];
        // Both aligned in same direction: cos = (3*4)/(|3|*|4|) = 12/12 = 1.0
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite_direction() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_both_zero_vectors() {
        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // --- retrieve edge cases ---

    #[test]
    fn test_retrieve_empty_db() {
        let rag = LateFusionRag::new(1);
        let results = rag.retrieve(&[1.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_empty_query() {
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 2.0]];
        let results = rag.retrieve(&[]);
        // query has length 0, min len is 0, dot=0, norms both 0 => similarity 0.0
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_retrieve_top_k_exceeds_db_size() {
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        rag.top_k = 10;
        let results = rag.retrieve(&[1.0, 0.0]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_retrieve_top_k_zero() {
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 0.0]];
        rag.top_k = 0;
        let results = rag.retrieve(&[1.0, 0.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_single_doc_db() {
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 1.0]];
        rag.top_k = 1;
        let results = rag.retrieve(&[1.0, 1.0]);
        assert_eq!(results.len(), 1);
        assert!((results[0][0] - 1.0).abs() < 1e-5);
    }

    // --- fuse_at_residual edge cases ---

    #[test]
    fn test_fuse_doc_shorter_than_hidden_state() {
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0]];
        rag.top_k = 1;
        rag.fusion_weight = 1.0;

        let mut state = vec![0.0, 0.0, 0.0];
        rag.fuse_at_residual(&mut state, 0);

        // Only state[0] gets fused: 0.0 + 1.0 * 1.0 = 1.0
        assert!((state[0] - 1.0).abs() < 1e-5);
        assert!((state[1]).abs() < 1e-5);
        assert!((state[2]).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_hidden_state_shorter_than_doc() {
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 2.0, 3.0, 4.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![0.0, 0.0];
        rag.fuse_at_residual(&mut state, 0);

        // Only 2 elements fused: 0.0 + 1.0*0.5 = 0.5, 0.0 + 2.0*0.5 = 1.0
        assert!((state[0] - 0.5).abs() < 1e-5);
        assert!((state[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_empty_hidden_state() {
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 2.0]];
        let mut state: Vec<f32> = vec![];
        rag.fuse_at_residual(&mut state, 0);
        assert!(state.is_empty());
    }

    #[test]
    fn test_fuse_zero_weight_no_change() {
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![10.0, 20.0]];
        rag.fusion_weight = 0.0;

        let mut state = vec![1.0, 2.0];
        rag.fuse_at_residual(&mut state, 0);
        assert!((state[0] - 1.0).abs() < 1e-5);
        assert!((state[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_multiple_docs_accumulate() {
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        rag.top_k = 2;
        rag.fusion_weight = 1.0;

        let mut state = vec![0.0, 0.0];
        // query = state, retrieves both docs with equal similarity
        rag.fuse_at_residual(&mut state, 0);

        // Both docs fused: state[0] += 1.0, state[1] += 1.0
        // Exact values depend on sort stability for equal scores,
        // but final state should have both components nonzero
        assert!(state[0] > 0.0);
        assert!(state[1] > 0.0);
    }

    // --- 15 additional tests ---

    #[test]
    fn test_cosine_similarity_negative_values() {
        // Arrange: vectors with mixed positive and negative components
        let a = vec![-1.0, 2.0, -3.0];
        let b = vec![1.0, -2.0, 3.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: a and b point in exactly opposite directions → similarity = -1.0
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_large_dimension() {
        // Arrange: two large orthogonal vectors (dimension 1000)
        let mut a = vec![0.0f32; 1000];
        let mut b = vec![0.0f32; 1000];
        a[0] = 1.0;
        b[1] = 1.0;

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: orthogonal → similarity = 0.0
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_symmetric() {
        // Arrange
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Act
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ba = cosine_similarity(&b, &a);

        // Assert: cosine similarity is symmetric
        assert!((sim_ab - sim_ba).abs() < 1e-7);
    }

    #[test]
    fn test_cosine_similarity_45_degree_angle() {
        // Arrange: vectors at 45 degrees → cos(45°) = sqrt(2)/2 ≈ 0.7071
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let expected = std::f32::consts::SQRT_2 / 2.0;

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert
        assert!((sim - expected).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_duplicate_documents() {
        // Arrange: database with identical documents
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        rag.top_k = 3;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: all three duplicates returned, each with similarity 1.0
        assert_eq!(results.len(), 3);
        for doc in &results {
            assert!((doc[0] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_retrieve_ranking_order() {
        // Arrange: four documents with strictly decreasing similarity to [1,0]
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![0.1, 0.9],  // low similarity to [1,0]
            vec![1.0, 0.0],  // exact match → highest
            vec![0.5, 0.5],  // medium similarity
            vec![0.9, 0.1],  // high similarity
        ];
        rag.top_k = 4;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: results ordered by descending similarity
        assert_eq!(results.len(), 4);
        assert!((results[0][0] - 1.0).abs() < 1e-5);  // [1,0] first
        assert!((results[1][0] - 0.9).abs() < 1e-5);  // [0.9,0.1] second
        assert!((results[2][0] - 0.5).abs() < 1e-5);  // [0.5,0.5] third
        assert!((results[3][0] - 0.1).abs() < 1e-5);  // [0.1,0.9] last
    }

    #[test]
    fn test_fuse_negative_weight_subtracts() {
        // Arrange: negative fusion weight causes subtraction
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 1.0]];
        rag.top_k = 1;
        rag.fusion_weight = -0.5;

        let mut state = vec![2.0, 3.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: 2.0 + 1.0 * (-0.5) = 1.5, 3.0 + 1.0 * (-0.5) = 2.5
        assert!((state[0] - 1.5).abs() < 1e-5);
        assert!((state[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_large_weight_amplifies() {
        // Arrange: large fusion weight significantly amplifies document contribution
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 0.0]];
        rag.top_k = 1;
        rag.fusion_weight = 10.0;

        let mut state = vec![0.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 1);

        // Assert: 0.0 + 1.0 * 10.0 = 10.0
        assert!((state[0] - 10.0).abs() < 1e-5);
        assert!((state[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_only_triggers_at_correct_layer() {
        // Arrange: call fuse_at_residual at multiple layers, only one should trigger
        let mut rag = LateFusionRag::new(2);
        rag.retrieval_db = vec![vec![1.0, 1.0]];
        rag.top_k = 1;
        rag.fusion_weight = 1.0;

        let mut state = vec![0.0, 0.0];

        // Act: call at layers 0, 1, 2, 3 — only layer 2 should modify state
        rag.fuse_at_residual(&mut state, 0);
        rag.fuse_at_residual(&mut state, 1);
        rag.fuse_at_residual(&mut state, 2);
        rag.fuse_at_residual(&mut state, 3);

        // Assert: only one fuse applied (layer 2)
        assert!((state[0] - 1.0).abs() < 1e-5);
        assert!((state[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_struct_inequality_different_top_k() {
        // Arrange
        let mut a = LateFusionRag::new(1);
        let mut b = LateFusionRag::new(1);
        a.top_k = 5;
        b.top_k = 10;

        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn test_struct_inequality_different_fusion_weight() {
        // Arrange
        let mut a = LateFusionRag::new(1);
        let mut b = LateFusionRag::new(1);
        a.fusion_weight = 0.1;
        b.fusion_weight = 0.5;

        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn test_struct_inequality_different_retrieval_db() {
        // Arrange
        let mut a = LateFusionRag::new(1);
        let mut b = LateFusionRag::new(1);
        a.retrieval_db = vec![vec![1.0, 2.0]];
        b.retrieval_db = vec![vec![3.0, 4.0]];

        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn test_retrieve_doc_dimension_differs_from_query() {
        // Arrange: query is 2D, doc is 4D — cosine_similarity uses min length (2)
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0, 99.0, 99.0],  // first 2 dims match query perfectly
            vec![0.0, 1.0, 50.0, 50.0],   // orthogonal in first 2 dims
        ];
        rag.top_k = 2;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: first doc ranked higher (sim=1.0 in first 2 dims)
        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 1.0).abs() < 1e-5);
        assert!((results[0][1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_negative_values_in_state_and_doc() {
        // Arrange: both hidden state and documents contain negative values
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![-2.0, -3.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![-1.0, -1.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: -1.0 + (-2.0)*0.5 = -2.0, -1.0 + (-3.0)*0.5 = -2.5
        assert!((state[0] - (-2.0)).abs() < 1e-5);
        assert!((state[1] - (-2.5)).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_all_docs_identical_returns_all() {
        // Arrange: all documents are the same, top_k covers all
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ];
        rag.top_k = 2;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: both documents returned (equal similarity, query not aligned)
        assert_eq!(results.len(), 2);
        for doc in &results {
            assert!((doc[0] - 0.5).abs() < 1e-5);
            assert!((doc[1] - 0.5).abs() < 1e-5);
        }
    }

    // --- 13 additional tests ---

    #[test]
    fn test_modify_retrieval_db_after_construction() {
        // Arrange: construct with empty db, then add documents
        let mut rag = LateFusionRag::new(0);
        assert!(rag.retrieval_db.is_empty());

        // Act: modify retrieval_db post-construction
        rag.retrieval_db = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        rag.top_k = 1;

        let mut state = vec![1.0, 0.0];
        rag.fuse_at_residual(&mut state, 0);

        // Assert: retrieval db was used — state modified by top-1 doc [1,0]
        // state[0] += 1.0 * 0.1 = 1.1, state[1] unchanged (doc[1]=0)
        assert!((state[0] - 1.1).abs() < 1e-5);
        assert!((state[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_partialeq_identical_retrieval_db_content() {
        // Arrange: two instances with identical retrieval_db content
        let mut a = LateFusionRag::new(1);
        let mut b = LateFusionRag::new(1);
        a.retrieval_db = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        b.retrieval_db = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        a.top_k = 2;
        b.top_k = 2;
        a.fusion_weight = 0.5;
        b.fusion_weight = 0.5;

        // Act & Assert: identical content yields equality
        assert_eq!(a, b);
    }

    #[test]
    fn test_cosine_similarity_parallel_vectors_exact_one() {
        // Arrange: two vectors scaled by a factor — perfectly parallel
        let a = vec![2.0, 4.0, 6.0];
        let b = vec![1.0, 2.0, 3.0]; // b = a / 2

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: parallel vectors produce exactly 1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors_exact_zero() {
        // Arrange: standard basis vectors in 3D — orthogonal by construction
        let a = vec![0.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: orthogonal vectors produce exactly 0.0
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fuse_single_doc_exact_arithmetic() {
        // Arrange: single doc with known values, verify each element precisely
        let mut rag = LateFusionRag::new(3);
        rag.retrieval_db = vec![vec![2.0, 4.0, 6.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.25;

        let mut state = vec![1.0, 1.0, 1.0];

        // Act
        rag.fuse_at_residual(&mut state, 3);

        // Assert: state[i] += doc[i] * weight
        // 1.0 + 2.0*0.25 = 1.5, 1.0 + 4.0*0.25 = 2.0, 1.0 + 6.0*0.25 = 2.5
        assert!((state[0] - 1.5).abs() < 1e-6);
        assert!((state[1] - 2.0).abs() < 1e-6);
        assert!((state[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_retrieve_preserves_order_for_same_scores() {
        // Arrange: three documents with identical similarity to query [1,0]
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0],  // doc 0: sim=1.0
            vec![0.6, 0.8],  // doc 1: sim=0.6/(1.0*1.0)=0.6
            vec![0.6, 0.8],  // doc 2: sim=0.6 (same as doc 1)
        ];
        rag.top_k = 3;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: all three returned; doc 0 is first (highest sim)
        assert_eq!(results.len(), 3);
        assert!((results[0][0] - 1.0).abs() < 1e-5);
        // The two equal-score docs are both present in positions 1 and 2
        assert!((results[1][0] - 0.6).abs() < 1e-5);
        assert!((results[2][0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_very_large_top_k_returns_all_docs() {
        // Arrange: small db with top_k set to usize::MAX
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        rag.top_k = usize::MAX;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: all documents returned despite top_k >> db size
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_cosine_similarity_very_small_positive_values() {
        // Arrange: vectors with very small but positive magnitudes
        let a = vec![1e-20, 2e-20];
        let b = vec![1e-20, 2e-20];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: parallel vectors, similarity should be 1.0 regardless of scale
        assert!((sim - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fuse_multiple_calls_accumulate_across_invocations() {
        // Arrange: call fuse_at_residual twice at the same layer
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 0.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![0.0, 0.0];

        // Act: first fuse — state[0] = 0.0 + 1.0*0.5 = 0.5
        rag.fuse_at_residual(&mut state, 0);
        // Second fuse — state[0] = 0.5 + 1.0*0.5 = 1.0 (query now [0.5,0.0])
        rag.fuse_at_residual(&mut state, 0);

        // Assert: two fuse operations accumulated
        assert!((state[0] - 1.0).abs() < 1e-5);
        assert!((state[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_single_element_vectors() {
        // Arrange: database of 1D vectors
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![3.0], vec![-3.0], vec![1.0]];
        rag.top_k = 2;

        // Act: query [1.0], top-2 by similarity
        let results = rag.retrieve(&[1.0]);

        // Assert: [3.0] sim=1.0 and [1.0] sim=1.0 are top, [-3.0] sim=-1.0 excluded
        assert_eq!(results.len(), 2);
        // Both positive docs appear; negative doc does not
        for doc in &results {
            assert!(doc[0] > 0.0);
        }
    }

    #[test]
    fn test_cosine_similarity_opposite_returns_negative_one() {
        // Arrange: vectors in exactly opposite directions
        let a = vec![3.0, 4.0];
        let b = vec![-3.0, -4.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: opposite direction → similarity = -1.0
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_clone_produces_independent_copy() {
        // Arrange
        let mut original = LateFusionRag::new(2);
        original.retrieval_db = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        original.top_k = 5;
        original.fusion_weight = 0.25;

        // Act
        let mut clone = original.clone();
        clone.retrieval_db[0][0] = 99.0;
        clone.fusion_weight = 0.9;

        // Assert: modifying clone does not affect original
        assert!((original.retrieval_db[0][0] - 1.0).abs() < 1e-5);
        assert!((original.fusion_weight - 0.25).abs() < 1e-5);
        assert!((clone.retrieval_db[0][0] - 99.0).abs() < 1e-5);
        assert!((clone.fusion_weight - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_debug_format_includes_all_field_names() {
        // Arrange
        let mut rag = LateFusionRag::new(7);
        rag.retrieval_db = vec![vec![1.0]];
        rag.top_k = 4;
        rag.fusion_weight = 0.3;

        // Act
        let debug_str = format!("{:?}", rag);

        // Assert: Debug output contains all four field names
        assert!(debug_str.contains("retrieval_db"));
        assert!(debug_str.contains("fusion_layer: 7"));
        assert!(debug_str.contains("top_k: 4"));
        assert!(debug_str.contains("fusion_weight: 0.3"));
    }

    // --- Wave 13 additional tests ---

    #[test]
    fn test_cosine_similarity_nan_input_returns_nan() {
        // Arrange: one vector contains NaN
        let a = vec![f32::NAN, 1.0];
        let b = vec![1.0, 0.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: NaN propagates through dot product → result is NaN
        assert!(sim.is_nan());
    }

    #[test]
    fn test_cosine_similarity_inf_input_returns_nan_or_inf() {
        // Arrange: one vector contains +Inf
        let a = vec![f32::INFINITY, 0.0];
        let b = vec![1.0, 0.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: inf*1.0 = inf, sqrt(inf) = inf, inf/inf = NaN
        assert!(sim.is_nan());
    }

    #[test]
    fn test_cosine_similarity_both_empty_slices_returns_zero() {
        // Arrange: both slices have length 0
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: min_len=0, dot=0, norms=0, norm check returns 0.0
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mixed_nan_and_finite() {
        // Arrange: NaN only in second element, first elements are normal
        let a = vec![1.0, f32::NAN];
        let b = vec![1.0, 1.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: dot = 1.0*1.0 + NaN*1.0 = NaN → result NaN
        assert!(sim.is_nan());
    }

    #[test]
    fn test_cosine_similarity_negative_inf_vector() {
        // Arrange: vector with -Inf component
        let a = vec![f32::NEG_INFINITY, 0.0];
        let b = vec![1.0, 0.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: -inf * 1.0 = -inf, norm_a = inf, -inf/inf = NaN
        assert!(sim.is_nan());
    }

    #[test]
    fn test_retrieve_all_zero_docs_returns_all_with_zero_similarity() {
        // Arrange: all documents are zero vectors — similarity to any query is 0.0
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        rag.top_k = 3;

        // Act
        let results = rag.retrieve(&[1.0, 2.0]);

        // Assert: all documents returned (equal similarity 0.0)
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_fuse_with_top_k_one_fuses_only_best_doc() {
        // Arrange: three docs with different similarity to query [1,0]
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![
            vec![10.0, 0.0],  // highest similarity to [1,0]
            vec![0.0, 10.0],  // orthogonal
            vec![5.0, 5.0],   // medium similarity
        ];
        rag.top_k = 1;
        rag.fusion_weight = 1.0;

        let mut state = vec![1.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: only the best doc [10,0] fused
        // state[0] = 1.0 + 10.0*1.0 = 11.0, state[1] = 0.0 + 0.0*1.0 = 0.0
        assert!((state[0] - 11.0).abs() < 1e-5);
        assert!((state[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_with_negative_similarity_docs_still_fused() {
        // Arrange: doc has negative similarity to query, but is still retrieved
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![
            vec![-1.0, 0.0],  // sim to [1,0] = -1.0
        ];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![1.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: state[0] = 1.0 + (-1.0)*0.5 = 0.5
        assert!((state[0] - 0.5).abs() < 1e-5);
        assert!((state[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_identical_scores_top_k_two_fuses_both() {
        // Arrange: two docs with identical similarity to query [1,0]
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![
            vec![0.6, 0.8],   // sim to [1,0] = 0.6
            vec![0.6, 0.8],   // identical, same sim = 0.6
        ];
        rag.top_k = 2;
        rag.fusion_weight = 1.0;

        let mut state = vec![1.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: both docs fused
        // state[0] = 1.0 + 0.6*1.0 + 0.6*1.0 = 2.2
        // state[1] = 0.0 + 0.8*1.0 + 0.8*1.0 = 1.6
        assert!((state[0] - 2.2).abs() < 1e-4);
        assert!((state[1] - 1.6).abs() < 1e-4);
    }

    #[test]
    fn test_struct_equality_empty_vs_nonempty_db() {
        // Arrange: same params but one has documents
        let mut a = LateFusionRag::new(1);
        let b = LateFusionRag::new(1);
        a.retrieval_db = vec![vec![1.0]];

        // Act & Assert: different retrieval_db → not equal
        assert_ne!(a, b);
    }

    #[test]
    fn test_retrieve_with_single_zero_doc() {
        // Arrange: single zero vector in db
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![0.0, 0.0, 0.0]];
        rag.top_k = 1;

        // Act
        let results = rag.retrieve(&[1.0, 2.0, 3.0]);

        // Assert: zero doc returned with similarity 0.0
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_fuse_at_layer_zero() {
        // Arrange: fusion_layer = 0, trigger at layer 0
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![2.0, 3.0]];
        rag.top_k = 1;
        rag.fusion_weight = 1.0;

        let mut state = vec![0.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: layer 0 is a valid fusion target
        assert!((state[0] - 2.0).abs() < 1e-5);
        assert!((state[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_doc_with_very_large_values_produces_inf() {
        // Arrange: document values large enough that weight * doc = Inf
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![f32::MAX]];
        rag.top_k = 1;
        rag.fusion_weight = 2.0;

        let mut state = vec![0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: 0.0 + MAX * 2.0 overflows to +Inf
        assert!(state[0].is_infinite());
        assert!(state[0].is_sign_positive());
    }

    // --- Wave 14 additional tests (13 tests) ---

    #[test]
    fn test_cosine_similarity_very_small_non_subnormal_values() {
        // Arrange: very small but non-subnormal f32 values (squared still non-zero)
        let a = vec![1e-18, 2e-18];
        let b = vec![1e-18, 2e-18];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: parallel vectors → similarity should be 1.0
        assert!((sim - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_similarity_one_element_zero() {
        // Arrange: one element is zero, the other is non-zero in the same slot
        let a = vec![0.0, 5.0];
        let b = vec![0.0, 3.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: only second element contributes, parallel → 1.0
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_high_dimensional_near_orthogonal() {
        // Arrange: 100-dimensional vectors that share only one basis component
        let mut a = vec![0.0f32; 100];
        let mut b = vec![0.0f32; 100];
        a[0] = 1.0;
        b[99] = 1.0;

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: fully orthogonal → exactly 0.0
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_retrieve_negative_sim_docs_ranked_last() {
        // Arrange: mix of positive, zero, and negative similarity documents
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![-1.0, 0.0],  // sim to [1,0] = -1.0
            vec![1.0, 0.0],   // sim to [1,0] =  1.0
            vec![0.0, 1.0],   // sim to [1,0] =  0.0
        ];
        rag.top_k = 3;

        // Act
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: positive sim first, zero sim second, negative sim last
        assert_eq!(results.len(), 3);
        assert!((results[0][0] - 1.0).abs() < 1e-5);
        assert!((results[1][1] - 1.0).abs() < 1e-5);
        assert!((results[2][0] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_fuse_very_small_weight_negligible_change() {
        // Arrange: extremely small fusion weight
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1e6, 1e6]];
        rag.top_k = 1;
        rag.fusion_weight = 1e-30;

        let original = vec![5.0, 5.0];
        let mut state = original.clone();

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: change is negligible — state barely changed from original
        assert!((state[0] - 5.0).abs() < 1e-3);
        assert!((state[1] - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_retrieve_returns_references_not_copies() {
        // Arrange: verify that retrieve returns references into the original db
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        rag.top_k = 2;

        // Act
        let results = rag.retrieve(&[1.0, 2.0]);

        // Assert: returned slices point to original db data — verify by pointer equality
        assert_eq!(results.len(), 2);
        // The first result [1,2] must be a slice of retrieval_db[0]
        assert!(results.iter().any(|r| r.as_ptr() == rag.retrieval_db[0].as_ptr()));
    }

    #[test]
    fn test_fuse_nan_in_hidden_state_propagates() {
        // Arrange: hidden state contains NaN before fusion
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![1.0, 2.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![f32::NAN, 1.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: NaN + anything = NaN in first slot; second slot gets 1.0 + 2.0*0.5 = 2.0
        assert!(state[0].is_nan());
        assert!((state[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_consistent_order_across_calls() {
        // Arrange: fixed db and query
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.5, 0.5],
        ];
        rag.top_k = 3;

        // Act: retrieve twice
        let first = rag.retrieve(&[1.0, 0.0]);
        let second = rag.retrieve(&[1.0, 0.0]);

        // Assert: both calls return the same order
        assert_eq!(first.len(), second.len());
        for i in 0..first.len() {
            assert_eq!(first[i], second[i]);
        }
    }

    #[test]
    fn test_fuse_at_layer_zero_does_not_fire_for_layer_one() {
        // Arrange: fusion_layer = 0, call with layer = 1
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![10.0, 10.0]];
        rag.top_k = 1;
        rag.fusion_weight = 1.0;

        let mut state = vec![1.0, 1.0];

        // Act: wrong layer
        rag.fuse_at_residual(&mut state, 1);

        // Assert: state unchanged
        assert!((state[0] - 1.0).abs() < 1e-5);
        assert!((state[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_retrieve_with_docs_of_decreasing_dimension() {
        // Arrange: documents have different dimensions, all longer than query
        let mut rag = LateFusionRag::new(1);
        rag.retrieval_db = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0],  // 5D
            vec![1.0, 0.0, 0.0],              // 3D
            vec![0.0, 1.0],                    // 2D
        ];
        rag.top_k = 3;

        // Act: 2D query [1,0]
        let results = rag.retrieve(&[1.0, 0.0]);

        // Assert: all returned, sorted by similarity in first 2 dims
        // doc0 sim=1.0 (first 2 dims [1,0]), doc1 sim=1.0, doc2 sim=0.0
        assert_eq!(results.len(), 3);
        // Both docs with sim=1.0 appear before the orthogonal one
        assert!((results[2][0] - 0.0).abs() < 1e-5);
        assert!((results[2][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_struct_clone_with_large_db() {
        // Arrange: RAG with a non-trivial database
        let mut original = LateFusionRag::new(5);
        original.retrieval_db = (0..50).map(|i| vec![i as f32, (i + 1) as f32]).collect();
        original.top_k = 10;
        original.fusion_weight = 0.3;

        // Act
        let cloned = original.clone();

        // Assert: deep copy — same content, independent memory
        assert_eq!(cloned, original);
        assert_eq!(cloned.retrieval_db.len(), 50);
        // Verify independence: original pointer != cloned pointer
        assert_ne!(original.retrieval_db.as_ptr(), cloned.retrieval_db.as_ptr());
    }

    #[test]
    fn test_cosine_similarity_with_f32_min_positive() {
        // Arrange: f32::MIN_POSITIVE squared underflows to 0.0,
        // so cosine_similarity returns 0.0 (zero-norm guard) rather than 1.0.
        // This test documents that behavior.
        let min_pos = f32::MIN_POSITIVE;
        let a = vec![min_pos, 0.0];
        let b = vec![min_pos, 0.0];

        // Act
        let sim = cosine_similarity(&a, &b);

        // Assert: norm_a = sqrt(MIN_POSITIVE^2) = 0.0 due to underflow → returns 0.0
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_fuse_with_db_containing_nan_doc() {
        // Arrange: one document contains NaN
        let mut rag = LateFusionRag::new(0);
        rag.retrieval_db = vec![vec![f32::NAN, 1.0]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut state = vec![0.0, 0.0];

        // Act
        rag.fuse_at_residual(&mut state, 0);

        // Assert: NaN doc fused → state[0] becomes NaN, state[1] = 0.0 + 1.0*0.5 = 0.5
        assert!(state[0].is_nan());
        assert!((state[1] - 0.5).abs() < 1e-5);
    }
}
