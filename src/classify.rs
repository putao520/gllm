//! Classify API — sync-first design (per SPEC 04-API-DESIGN §3.6).
//!
//! Supports sequence classification models (both encoder and decoder based).
//! Encoder-based: BERT/XLM-R + classifier head (e.g. BAAI/bge-reranker, sentiment models).
//! Decoder-based: LLM + score head (e.g. Qwen3ForSequenceClassification).

use crate::client::{Client, GllmError};

/// Builder for text classification.
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_classifier("model-id")?;
/// let result = client.classify(["This movie is great!", "Terrible experience."])?;
/// for r in &result.predictions {
///     println!("{}: label={} score={:.4}", r.index, r.label_id, r.score);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ClassifyBuilder<'a> {
    client: &'a Client,
    texts: Vec<String>,
}

impl<'a> ClassifyBuilder<'a> {
    pub(crate) fn new(client: &'a Client, texts: Vec<String>) -> Self {
        Self { client, texts }
    }

    /// Execute the classification (sync).
    pub fn generate(self) -> Result<ClassifyResponse, GllmError> {
        self.client.execute_classify(self.texts)
    }
}

/// Response from text classification.
#[derive(Debug, Clone)]
pub struct ClassifyResponse {
    /// Classification predictions, one per input text.
    pub predictions: Vec<ClassificationResult>,
}

/// A single classification result.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Index of the input text in the original batch.
    pub index: usize,
    /// Predicted label ID (argmax of logits).
    pub label_id: usize,
    /// Score for the predicted label (softmax probability).
    pub score: f32,
    /// Full logits vector for all labels (raw, pre-softmax).
    pub logits: Vec<f32>,
}
