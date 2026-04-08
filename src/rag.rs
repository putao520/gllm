//! §16 P1: Late-Fusion RAG
//!
//! Fuses retrieval-augmented information at residual connection points.

/// Late-Fusion RAG configuration
#[derive(Debug, Clone)]
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
}
