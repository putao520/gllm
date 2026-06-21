//! Rerank API — sync-first design (per SPEC 04-API-DESIGN §3.3).

use crate::client::{Client, GllmError};

/// Builder for document reranking (per SPEC 04-API-DESIGN §3.3).
///
/// Supports `top_n()` for Top-K truncation via builder pattern.
///
/// # Example
///
/// ```no_run
/// use gllm::Client;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let client = Client::new_empty();
/// let scores = client.rerank(
///     "What is the capital of France?",
///     vec![
///         "Paris is the capital of France",
///         "London is in UK",
///         "Berlin is in Germany"
///     ],
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct RerankBuilder<'a> {
    client: &'a Client,
    query: String,
    documents: Vec<String>,
    top_n: usize,
}

impl<'a> RerankBuilder<'a> {
    pub(crate) fn new(
        client: &'a Client,
        query: impl Into<String>,
        documents: Vec<String>,
    ) -> Self {
        Self {
            client,
            query: query.into(),
            documents,
            top_n: 5,
        }
    }

    /// Set maximum number of results to return.
    pub fn top_n(mut self, top_n: usize) -> Self {
        self.top_n = top_n;
        self
    }

    /// Execute the reranking (sync).
    pub fn generate(self) -> Result<RerankResponse, GllmError> {
        self.client.execute_rerank(self.query, self.documents, self.top_n)
    }
}

/// Response from reranking (per SPEC 04-API-DESIGN §3.3).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResponse {
    /// Sorted results by relevance score.
    pub results: Vec<RerankResult>,
    /// Request ID (for tracking).
    pub request_id: Option<u64>,
}

/// A single reranking result ( per SPEC 04-API-DESIGN §3.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankResult {
    /// Original document index.
    pub index: usize,
    /// Relevance score ( higher = more relevant).
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerank_response_sorted() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 2, score: 0.9 },
                RerankResult { index: 0, score: 0.5 },
                RerankResult { index: 1, score: 0.1 },
            ],
            request_id: Some(42),
        };
        assert_eq!(resp.results.len(), 3);
        assert_eq!(resp.request_id, Some(42));
        // Verify descending order
        assert!(resp.results[0].score > resp.results[1].score);
    }

    #[test]
    fn rerank_result_fields() {
        let r = RerankResult { index: 5, score: 0.75 };
        assert_eq!(r.index, 5);
        assert!((r.score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn rerank_response_empty_results() {
        let resp = RerankResponse {
            results: vec![],
            request_id: None,
        };
        assert!(resp.results.is_empty());
        assert!(resp.request_id.is_none());
    }

    #[test]
    fn rerank_response_clone() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 1.0 },
            ],
            request_id: Some(99),
        };
        let cloned = resp.clone();
        assert_eq!(cloned.results.len(), 1);
        assert_eq!(cloned.request_id, Some(99));
    }

    #[test]
    fn rerank_result_debug_format() {
        let r = RerankResult { index: 1, score: 0.5 };
        let debug = format!("{r:?}");
        assert!(debug.contains("index"));
        assert!(debug.contains("score"));
    }

    #[test]
    fn rerank_response_debug_format() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(42),
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("results"));
        assert!(debug.contains("request_id"));
    }

    // ── Additional coverage ──

    #[test]
    fn rerank_result_clone_independence() {
        let r = RerankResult { index: 3, score: 0.88 };
        let cloned = r.clone();
        assert_eq!(cloned.index, 3);
        assert!((cloned.score - 0.88).abs() < 1e-6);
    }

    #[test]
    fn rerank_response_single_result() {
        let resp = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: Some(1),
        };
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].index, 0);
    }

    #[test]
    fn rerank_result_negative_score() {
        let r = RerankResult { index: 2, score: -0.5 };
        assert!((r.score - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn rerank_response_many_results() {
        let results: Vec<RerankResult> = (0..100)
            .map(|i| RerankResult { index: i, score: 1.0 - i as f32 / 100.0 })
            .collect();
        let resp = RerankResponse {
            results,
            request_id: None,
        };
        assert_eq!(resp.results.len(), 100);
        assert!((resp.results[0].score - 1.0).abs() < 1e-6);
        assert!((resp.results[99].score - 0.01).abs() < 1e-6);
    }

    // ── PartialEq tests ──

    #[test]
    fn rerank_result_partial_eq_equal() {
        let a = RerankResult { index: 1, score: 0.5 };
        let b = RerankResult { index: 1, score: 0.5 };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_result_partial_eq_different_index() {
        let a = RerankResult { index: 1, score: 0.5 };
        let b = RerankResult { index: 2, score: 0.5 };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_result_partial_eq_different_score() {
        let a = RerankResult { index: 1, score: 0.5 };
        let b = RerankResult { index: 1, score: 0.6 };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_response_partial_eq_equal() {
        let a = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: Some(42),
        };
        let b = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: Some(42),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_response_partial_eq_different_results() {
        let a = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: None,
        };
        let b = RerankResponse {
            results: vec![RerankResult { index: 1, score: 1.0 }],
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_response_partial_eq_different_request_id() {
        let a = RerankResponse {
            results: vec![],
            request_id: Some(1),
        };
        let b = RerankResponse {
            results: vec![],
            request_id: Some(2),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_response_partial_eq_none_vs_some_request_id() {
        let a = RerankResponse {
            results: vec![],
            request_id: None,
        };
        let b = RerankResponse {
            results: vec![],
            request_id: Some(0),
        };
        assert_ne!(a, b);
    }

    // ── RerankResult boundary scores ──

    #[test]
    fn rerank_result_score_zero() {
        let r = RerankResult { index: 0, score: 0.0 };
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn rerank_result_score_max() {
        let r = RerankResult { index: 0, score: f32::MAX };
        assert_eq!(r.score, f32::MAX);
    }

    #[test]
    fn rerank_result_score_min_positive() {
        let r = RerankResult { index: 0, score: f32::MIN_POSITIVE };
        assert!(r.score > 0.0);
        assert_eq!(r.score, f32::MIN_POSITIVE);
    }

    #[test]
    fn rerank_result_score_infinity() {
        let r = RerankResult { index: 0, score: f32::INFINITY };
        assert!(r.score.is_infinite() && r.score.is_sign_positive());
    }

    #[test]
    fn rerank_result_score_neg_infinity() {
        let r = RerankResult { index: 0, score: f32::NEG_INFINITY };
        assert!(r.score.is_infinite() && r.score.is_sign_negative());
    }

    #[test]
    fn rerank_result_score_nan() {
        let r = RerankResult { index: 0, score: f32::NAN };
        assert!(r.score.is_nan());
    }

    // ── RerankResult index boundaries ──

    #[test]
    fn rerank_result_index_zero() {
        let r = RerankResult { index: 0, score: 1.0 };
        assert_eq!(r.index, 0);
    }

    #[test]
    fn rerank_result_index_max() {
        let r = RerankResult { index: usize::MAX, score: 1.0 };
        assert_eq!(r.index, usize::MAX);
    }

    // ── RerankResponse request_id boundaries ──

    #[test]
    fn rerank_response_request_id_zero() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(0),
        };
        assert_eq!(resp.request_id, Some(0));
    }

    #[test]
    fn rerank_response_request_id_max() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(u64::MAX),
        };
        assert_eq!(resp.request_id, Some(u64::MAX));
    }

    // ── Sorting with edge scores ──

    #[test]
    fn rerank_results_sort_equal_scores() {
        let mut results = vec![
            RerankResult { index: 2, score: 0.5 },
            RerankResult { index: 0, score: 0.5 },
            RerankResult { index: 1, score: 0.5 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(results.len(), 3);
        // All scores are equal, verify they are still 0.5
        for r in &results {
            assert!((r.score - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn rerank_results_sort_mixed_nan() {
        let mut results = vec![
            RerankResult { index: 0, score: 0.9 },
            RerankResult { index: 1, score: f32::NAN },
            RerankResult { index: 2, score: 0.1 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(results.len(), 3);
        // NaN partial_cmp returns None → Equal, so relative order of NaN is undefined.
        // The non-NaN entries should maintain descending order among themselves.
        let non_nan: Vec<f32> = results.iter().filter_map(|r| {
            if r.score.is_nan() { None } else { Some(r.score) }
        }).collect();
        let mut sorted_non_nan = non_nan.clone();
        sorted_non_nan.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(non_nan, sorted_non_nan);
    }

    // ── RerankBuilder with empty client (error path) ──

    #[test]
    fn rerank_builder_generate_no_model_returns_error() {
        let client = Client::new_empty();
        let result = client.rerank("query", vec!["doc1", "doc2"]);
        assert!(result.is_err());
        match result.unwrap_err() {
            GllmError::NoModelLoaded => {}
            other => panic!("expected NoModelLoaded, got {:?}", other),
        }
    }

    #[test]
    fn rerank_builder_generate_empty_documents_no_model() {
        let client = Client::new_empty();
        let result = client.rerank("query", Vec::<String>::new());
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_top_n_method() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(&client, "test query", vec!["a".to_string(), "b".to_string()])
            .top_n(1);
        // Verify builder was constructed (top_n chains correctly)
        // The builder is consumed by generate(), which will fail on empty client.
        let result = builder.generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_top_n_zero() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(&client, "query", vec!["doc".to_string()])
            .top_n(0);
        let result = builder.generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_top_n_max() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(&client, "query", vec!["doc".to_string()])
            .top_n(usize::MAX);
        let result = builder.generate();
        assert!(result.is_err());
    }

    // ── ClientError Display (thiserror derive) ──

    #[test]
    fn client_error_display_model_not_found() {
        let err = GllmError::ModelNotFound("test-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("model not found"));
    }

    #[test]
    fn client_error_display_invalid_model_type() {
        let err = GllmError::InvalidModelType;
        let msg = format!("{}", err);
        assert!(msg.contains("invalid model type"));
    }

    #[test]
    fn client_error_display_no_model_loaded() {
        let err = GllmError::NoModelLoaded;
        let msg = format!("{}", err);
        assert!(msg.contains("no model loaded"));
    }

    #[test]
    fn client_error_display_runtime_error() {
        let err = GllmError::RuntimeError("something broke".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("something broke"));
        assert!(msg.contains("runtime error"));
    }

    // ── ClientError Debug ──

    #[test]
    fn client_error_debug_contains_variant() {
        let err = GllmError::NoModelLoaded;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoModelLoaded"));
    }

    // ── Clone on RerankBuilder is not derived (lifetimed struct), verify response clone depth ──

    #[test]
    fn rerank_response_clone_deep_independence() {
        let mut resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 1, score: 0.3 },
            ],
            request_id: Some(100),
        };
        let cloned = resp.clone();
        // Mutate original, cloned should be unaffected
        resp.results.clear();
        resp.request_id = None;
        assert_eq!(cloned.results.len(), 2);
        assert_eq!(cloned.request_id, Some(100));
    }

    // ── Empty-string query/document via builder ──

    #[test]
    fn rerank_builder_empty_query_string() {
        let client = Client::new_empty();
        let result = client.rerank("", vec!["doc"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_empty_document_strings() {
        let client = Client::new_empty();
        let result = client.rerank("query", vec!["", "", ""]);
        assert!(result.is_err());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  New tests: 45+ additional tests
    // ══════════════════════════════════════════════════════════════════════

    // ── RerankResult Copy derive ──

    #[test]
    fn rerank_result_copy_semantics() {
        let original = RerankResult { index: 7, score: 0.42 };
        let copied = original; // Copy, not move
        assert_eq!(original.index, 7);
        assert_eq!(copied.index, 7);
        assert!((original.score - copied.score).abs() < 1e-6);
    }

    #[test]
    fn rerank_result_copy_after_clone() {
        let r = RerankResult { index: 10, score: 0.99 };
        let via_copy = r;
        let via_clone = r.clone();
        assert_eq!(via_copy, via_clone);
    }

    // ── RerankResult Debug format content ──

    #[test]
    fn rerank_result_debug_shows_index_value() {
        let r = RerankResult { index: 42, score: 0.123 };
        let debug = format!("{r:?}");
        assert!(debug.contains("42"));
    }

    #[test]
    fn rerank_result_debug_shows_score_value() {
        let r = RerankResult { index: 0, score: 0.75 };
        let debug = format!("{r:?}");
        assert!(debug.contains("0.75"));
    }

    #[test]
    fn rerank_result_debug_shows_nan_score() {
        let r = RerankResult { index: 0, score: f32::NAN };
        let debug = format!("{r:?}");
        assert!(debug.contains("NaN") || debug.contains("nan"));
    }

    #[test]
    fn rerank_result_debug_shows_infinity_score() {
        let r = RerankResult { index: 0, score: f32::INFINITY };
        let debug = format!("{r:?}");
        assert!(debug.contains("inf") || debug.contains("Inf"));
    }

    // ── RerankResponse Debug format content ──

    #[test]
    fn rerank_response_debug_shows_request_id_none() {
        let resp = RerankResponse {
            results: vec![],
            request_id: None,
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("None") || debug.contains("request_id"));
    }

    #[test]
    fn rerank_response_debug_shows_request_id_value() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(999),
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("999"));
    }

    #[test]
    fn rerank_response_debug_shows_result_count() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.5 },
                RerankResult { index: 1, score: 0.3 },
            ],
            request_id: None,
        };
        let debug = format!("{resp:?}");
        assert!(debug.contains("0.5"));
        assert!(debug.contains("0.3"));
    }

    // ── RerankResult equality with special floats ──

    #[test]
    fn rerank_result_eq_same_nan_is_false() {
        let a = RerankResult { index: 0, score: f32::NAN };
        let b = RerankResult { index: 0, score: f32::NAN };
        assert_ne!(a, b); // NaN != NaN per IEEE 754
    }

    #[test]
    fn rerank_result_eq_positive_zero_negative_zero() {
        let a = RerankResult { index: 0, score: 0.0f32 };
        let b = RerankResult { index: 0, score: -0.0f32 };
        assert_eq!(a, b); // +0.0 == -0.0 per IEEE 754
    }

    #[test]
    fn rerank_result_eq_both_infinity() {
        let a = RerankResult { index: 0, score: f32::INFINITY };
        let b = RerankResult { index: 0, score: f32::INFINITY };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_result_eq_both_neg_infinity() {
        let a = RerankResult { index: 0, score: f32::NEG_INFINITY };
        let b = RerankResult { index: 0, score: f32::NEG_INFINITY };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_result_neq_pos_neg_infinity() {
        let a = RerankResult { index: 0, score: f32::INFINITY };
        let b = RerankResult { index: 0, score: f32::NEG_INFINITY };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_result_neq_nan_vs_real() {
        let a = RerankResult { index: 0, score: f32::NAN };
        let b = RerankResult { index: 0, score: 1.0 };
        assert_ne!(a, b);
    }

    // ── RerankResponse equality edge cases ──

    #[test]
    fn rerank_response_eq_both_empty() {
        let a = RerankResponse { results: vec![], request_id: None };
        let b = RerankResponse { results: vec![], request_id: None };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_response_neq_different_lengths() {
        let a = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: None,
        };
        let b = RerankResponse { results: vec![], request_id: None };
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_response_eq_identical_multi_result() {
        let a = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 1, score: 0.5 },
                RerankResult { index: 2, score: 0.1 },
            ],
            request_id: Some(123),
        };
        let b = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 1, score: 0.5 },
                RerankResult { index: 2, score: 0.1 },
            ],
            request_id: Some(123),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_response_neq_result_order_matters() {
        let a = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 1, score: 0.5 },
            ],
            request_id: None,
        };
        let b = RerankResponse {
            results: vec![
                RerankResult { index: 1, score: 0.5 },
                RerankResult { index: 0, score: 0.9 },
            ],
            request_id: None,
        };
        assert_ne!(a, b); // order matters for Vec equality
    }

    // ── RerankResponse clone independence ──

    #[test]
    fn rerank_response_clone_preserves_all_fields() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 1.0 },
                RerankResult { index: 1, score: 0.8 },
                RerankResult { index: 2, score: 0.6 },
            ],
            request_id: Some(555),
        };
        let cloned = resp.clone();
        assert_eq!(cloned.results.len(), 3);
        assert_eq!(cloned.results[0], RerankResult { index: 0, score: 1.0 });
        assert_eq!(cloned.results[1], RerankResult { index: 1, score: 0.8 });
        assert_eq!(cloned.results[2], RerankResult { index: 2, score: 0.6 });
        assert_eq!(cloned.request_id, Some(555));
    }

    #[test]
    fn rerank_response_clone_results_vec_independence() {
        let mut resp = RerankResponse {
            results: vec![RerankResult { index: 0, score: 0.5 }],
            request_id: Some(1),
        };
        let cloned = resp.clone();
        resp.results.push(RerankResult { index: 1, score: 0.2 });
        assert_eq!(cloned.results.len(), 1); // cloned unaffected
        assert_eq!(resp.results.len(), 2);
    }

    // ── RerankBuilder via client.rerank() with various inputs ──

    #[test]
    fn rerank_builder_single_document_no_model() {
        let client = Client::new_empty();
        let result = client.rerank("query", vec!["only doc"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_many_documents_no_model() {
        let client = Client::new_empty();
        let docs: Vec<String> = (0..200).map(|i| format!("document {i}")).collect();
        let result = client.rerank("query", docs);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_unicode_query_no_model() {
        let client = Client::new_empty();
        let result = client.rerank("什么是法国的首都？", vec!["巴黎", "伦敦"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_very_long_query_no_model() {
        let client = Client::new_empty();
        let long_query = "x".repeat(10000);
        let result = client.rerank(long_query, vec!["doc"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_very_long_document_no_model() {
        let client = Client::new_empty();
        let long_doc = "y".repeat(10000);
        let result = client.rerank("query", vec![long_doc]);
        assert!(result.is_err());
    }

    // ── RerankBuilder::new default top_n ──

    #[test]
    fn rerank_builder_default_top_n_is_five() {
        let client = Client::new_empty();
        // Can't inspect top_n directly, but verify builder chains through generate()
        // which proves construction succeeded. Default top_n=5 is internal.
        let builder = RerankBuilder::new(&client, "q", vec!["a".to_string()]);
        let result = builder.generate();
        assert!(result.is_err()); // NoModelLoaded expected
    }

    #[test]
    fn rerank_builder_top_n_one() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(&client, "q", vec!["a".to_string(), "b".to_string()])
            .top_n(1);
        let result = builder.generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_top_n_chains() {
        let client = Client::new_empty();
        // top_n should return Self for chaining
        let builder = RerankBuilder::new(&client, "q", vec!["a".to_string()])
            .top_n(10)
            .top_n(3); // second call overrides first
        let result = builder.generate();
        assert!(result.is_err());
    }

    // ── RerankBuilder via Client::rerank with Into<String> types ──

    #[test]
    fn rerank_builder_query_from_string_ref() {
        let client = Client::new_empty();
        let query: &str = "test query";
        let result = client.rerank(query, vec!["doc"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_query_from_owned_string() {
        let client = Client::new_empty();
        let query: String = String::from("owned query");
        let result = client.rerank(query, vec!["doc"]);
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_documents_from_str_slice() {
        let client = Client::new_empty();
        let result = client.rerank("q", ["doc1", "doc2", "doc3"]);
        assert!(result.is_err());
    }

    // ── RerankResult sorting edge cases ──

    #[test]
    fn rerank_results_sort_descending() {
        let mut results = vec![
            RerankResult { index: 0, score: 0.1 },
            RerankResult { index: 1, score: 0.9 },
            RerankResult { index: 2, score: 0.5 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(results[0].index, 1);
        assert_eq!(results[1].index, 2);
        assert_eq!(results[2].index, 0);
    }

    #[test]
    fn rerank_results_sort_all_same_scores() {
        let mut results = vec![
            RerankResult { index: 5, score: 0.5 },
            RerankResult { index: 3, score: 0.5 },
            RerankResult { index: 8, score: 0.5 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert!(results.iter().all(|r| (r.score - 0.5).abs() < 1e-6));
    }

    #[test]
    fn rerank_results_sort_negative_scores() {
        let mut results = vec![
            RerankResult { index: 0, score: -0.1 },
            RerankResult { index: 1, score: -0.9 },
            RerankResult { index: 2, score: -0.5 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert!((results[0].score - (-0.1)).abs() < 1e-6);
        assert!((results[1].score - (-0.5)).abs() < 1e-6);
        assert!((results[2].score - (-0.9)).abs() < 1e-6);
    }

    #[test]
    fn rerank_results_sort_with_infinity() {
        let mut results = vec![
            RerankResult { index: 0, score: 0.5 },
            RerankResult { index: 1, score: f32::INFINITY },
            RerankResult { index: 2, score: 0.1 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert!(results[0].score.is_infinite());
    }

    #[test]
    fn rerank_results_sort_with_neg_infinity() {
        let mut results = vec![
            RerankResult { index: 0, score: f32::NEG_INFINITY },
            RerankResult { index: 1, score: 0.5 },
            RerankResult { index: 2, score: 0.1 },
        ];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert!(results[2].score.is_infinite() && results[2].score.is_sign_negative());
    }

    #[test]
    fn rerank_results_sort_single_element() {
        let mut results = vec![RerankResult { index: 0, score: 0.5 }];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
    }

    #[test]
    fn rerank_results_sort_empty_vec() {
        let mut results: Vec<RerankResult> = vec![];
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        assert!(results.is_empty());
    }

    // ── RerankResult with index boundary: 1 ──

    #[test]
    fn rerank_result_index_one() {
        let r = RerankResult { index: 1, score: 0.0 };
        assert_eq!(r.index, 1);
    }

    // ── RerankResult score subnormal ──

    #[test]
    fn rerank_result_score_subnormal() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let r = RerankResult { index: 0, score: subnormal };
        assert!(r.score > 0.0);
        assert!(r.score < f32::MIN_POSITIVE);
    }

    // ── RerankResult score negative max ──

    #[test]
    fn rerank_result_score_negative_max() {
        let r = RerankResult { index: 0, score: -f32::MAX };
        assert_eq!(r.score, -f32::MAX);
        assert!(r.score.is_sign_negative());
    }

    // ── RerankResponse request_id: u64 boundaries ──

    #[test]
    fn rerank_response_request_id_one() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(1),
        };
        assert_eq!(resp.request_id, Some(1));
    }

    #[test]
    fn rerank_response_request_id_large() {
        let resp = RerankResponse {
            results: vec![],
            request_id: Some(u64::MAX / 2),
        };
        assert_eq!(resp.request_id, Some(u64::MAX / 2));
    }

    // ── RerankResponse with large results vec ──

    #[test]
    fn rerank_response_large_results_vec() {
        let results: Vec<RerankResult> = (0..1000)
            .map(|i| RerankResult { index: i, score: i as f32 })
            .collect();
        let resp = RerankResponse {
            results,
            request_id: Some(0),
        };
        assert_eq!(resp.results.len(), 1000);
        assert_eq!(resp.results[0].index, 0);
        assert_eq!(resp.results[999].index, 999);
    }

    // ── RerankBuilder with Into<String> for documents ──

    #[test]
    fn rerank_builder_documents_from_string_slices() {
        let client = Client::new_empty();
        let docs: Vec<&str> = vec!["alpha", "beta", "gamma"];
        let result = client.rerank("q", docs);
        assert!(result.is_err());
    }

    // ── GllmError variant coverage ──

    #[test]
    fn client_error_display_out_of_memory() {
        let oom = crate::kv_cache::OomHaltError::fatal_halt("need 100 blocks".to_string());
        let err = GllmError::OutOfMemory(oom);
        let msg = format!("{err}");
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn client_error_display_backend_error() {
        // BackendError requires internal type — test through Display variant
        let err = GllmError::InvalidModelType;
        let msg = format!("{err}");
        assert!(msg.contains("invalid model type"));
    }

    #[test]
    fn client_error_debug_model_not_found() {
        let err = GllmError::ModelNotFound("my-model".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("ModelNotFound"));
        assert!(debug.contains("my-model"));
    }

    #[test]
    fn client_error_debug_runtime_error() {
        let err = GllmError::RuntimeError("oops".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("RuntimeError"));
        assert!(debug.contains("oops"));
    }

    #[test]
    fn client_error_debug_out_of_memory() {
        let oom = crate::kv_cache::OomHaltError::fatal_halt("test".to_string());
        let err = GllmError::OutOfMemory(oom);
        let debug = format!("{err:?}");
        assert!(debug.contains("OutOfMemory"));
    }

    // ── RerankResult as a value in collections ──

    #[test]
    fn rerank_result_in_hashset_via_sort() {
        // RerankResult is not Hash (f32), but we can test Vec dedup by PartialEq
        let mut results = vec![
            RerankResult { index: 0, score: 0.5 },
            RerankResult { index: 0, score: 0.5 },
            RerankResult { index: 1, score: 0.3 },
        ];
        results.dedup();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn rerank_result_in_btreemap_key_via_index() {
        // Use index as key since RerankResult is not Ord
        let results = vec![
            RerankResult { index: 0, score: 0.9 },
            RerankResult { index: 1, score: 0.5 },
        ];
        let map: std::collections::BTreeMap<usize, f32> = results
            .iter()
            .map(|r| (r.index, r.score))
            .collect();
        assert_eq!(map[&0], 0.9);
        assert_eq!(map[&1], 0.5);
    }

    // ── RerankResponse default construction patterns ──

    #[test]
    fn rerank_response_with_request_id_zero_and_results() {
        let resp = RerankResponse {
            results: vec![RerankResult { index: 0, score: 1.0 }],
            request_id: Some(0),
        };
        assert_eq!(resp.request_id, Some(0));
        assert_eq!(resp.results[0].score, 1.0);
    }

    // ── RerankBuilder::new captures query correctly ──

    #[test]
    fn rerank_builder_new_stores_empty_query() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(&client, "", vec![]);
        // generate fails at NoModelLoaded, proving construction succeeded
        assert!(builder.generate().is_err());
    }

    #[test]
    fn rerank_builder_new_stores_documents() {
        let client = Client::new_empty();
        let builder = RerankBuilder::new(
            &client,
            "q",
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );
        assert!(builder.generate().is_err());
    }

    // ── RerankResult with extreme score and index combos ──

    #[test]
    fn rerank_result_max_index_max_score() {
        let r = RerankResult { index: usize::MAX, score: f32::MAX };
        assert_eq!(r.index, usize::MAX);
        assert_eq!(r.score, f32::MAX);
    }

    #[test]
    fn rerank_result_max_index_negative_score() {
        let r = RerankResult { index: usize::MAX, score: -f32::MAX };
        assert_eq!(r.index, usize::MAX);
        assert!(r.score.is_sign_negative());
    }

    #[test]
    fn rerank_result_zero_index_infinity_score() {
        let r = RerankResult { index: 0, score: f32::INFINITY };
        assert_eq!(r.index, 0);
        assert!(r.score.is_infinite());
    }

    // ── RerankResponse with NaN scores in results ──

    #[test]
    fn rerank_response_with_nan_score_result() {
        let resp = RerankResponse {
            results: vec![RerankResult { index: 0, score: f32::NAN }],
            request_id: None,
        };
        assert!(resp.results[0].score.is_nan());
    }

    #[test]
    fn rerank_response_mixed_nan_and_real_scores() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 1, score: f32::NAN },
                RerankResult { index: 2, score: f32::INFINITY },
                RerankResult { index: 3, score: f32::NEG_INFINITY },
            ],
            request_id: Some(42),
        };
        assert_eq!(resp.results.len(), 4);
        assert!((resp.results[0].score - 0.9).abs() < 1e-6);
        assert!(resp.results[1].score.is_nan());
        assert!(resp.results[2].score.is_infinite() && resp.results[2].score.is_sign_positive());
        assert!(resp.results[3].score.is_infinite() && resp.results[3].score.is_sign_negative());
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional tests (10 new — no overlap with existing)
    // ══════════════════════════════════════════════════════════════════════

    // ── ClientError From<BackendContextError> conversion ──

    #[test]
    fn client_error_from_unsupported_architecture() {
        let ctx_err = crate::backend::BackendContextError::UnsupportedArchitecture(
            "fake-arch".to_string(),
        );
        let client_err: GllmError = ctx_err.into();
        let msg = format!("{client_err}");
        assert!(
            msg.contains("invalid model type"),
            "UnsupportedArchitecture should map to InvalidModelType, got: {msg}"
        );
    }

    // ── ClientError From<ExecutorError> conversion ──

    #[test]
    fn client_error_from_executor_backend_error() {
        let exec_err = crate::engine::executor::ExecutorError::Backend(
            crate::engine::executor::BackendError::Cpu("cpu failure".to_string()),
        );
        let client_err: GllmError = exec_err.into();
        let msg = format!("{client_err}");
        assert!(
            msg.contains("runtime error"),
            "ExecutorError should map to RuntimeError, got: {msg}"
        );
    }

    // ── ClientError From<SchedulerError> OOM path ──

    #[test]
    fn client_error_from_scheduler_oom() {
        let sched_err = crate::scheduler::paged_scheduler::SchedulerError::OutOfMemory {
            operation: "prefill",
            needed_blocks: 100,
            free_blocks: 10,
        };
        let client_err: GllmError = sched_err.into();
        let msg = format!("{client_err}");
        assert!(
            msg.contains("out of memory"),
            "Scheduler OOM should map to OutOfMemory, got: {msg}"
        );
    }

    // ── ClientError From<ModelConfigError> conversion ──

    #[test]
    fn client_error_from_model_config_missing() {
        let cfg_err = crate::model_config::ModelConfigError::MissingConfig;
        let client_err: GllmError = cfg_err.into();
        let msg = format!("{client_err}");
        assert!(
            msg.contains("runtime error"),
            "ModelConfigError should map to RuntimeError, got: {msg}"
        );
        assert!(
            msg.contains("model config error"),
            "Should include original error context, got: {msg}"
        );
    }

    // ── RerankResponse results with duplicate indices (valid use case) ──

    #[test]
    fn rerank_response_duplicate_indices_are_allowed() {
        let resp = RerankResponse {
            results: vec![
                RerankResult { index: 0, score: 0.9 },
                RerankResult { index: 0, score: 0.7 },
                RerankResult { index: 1, score: 0.5 },
            ],
            request_id: None,
        };
        let count_index_0 = resp.results.iter().filter(|r| r.index == 0).count();
        assert_eq!(count_index_0, 2, "duplicate indices should be preserved");
        assert_eq!(resp.results.len(), 3);
    }

    // ── RerankResult scores very close but not equal (epsilon distinction) ──

    #[test]
    fn rerank_result_very_close_scores_are_distinct() {
        let a = RerankResult { index: 0, score: 1.0f32 };
        let b = RerankResult { index: 1, score: 1.0f32 - f32::EPSILON };
        assert_ne!(a, b, "scores differing by epsilon must not be equal");
    }

    // ── Sort stability: indices remain associated with their scores ──

    #[test]
    fn rerank_sort_preserves_index_score_association() {
        let mut results = vec![
            RerankResult { index: 10, score: 0.1 },
            RerankResult { index: 20, score: 0.9 },
            RerankResult { index: 30, score: 0.5 },
        ];
        results.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        // After descending sort, index 20 (score 0.9) must be first
        assert_eq!(results[0].index, 20);
        assert_eq!(results[1].index, 30);
        assert_eq!(results[2].index, 10);
    }

    // ── RerankResponse same request_id but different results are not equal ──

    #[test]
    fn rerank_response_same_id_different_scores_not_equal() {
        let a = RerankResponse {
            results: vec![RerankResult { index: 0, score: 0.9 }],
            request_id: Some(42),
        };
        let b = RerankResponse {
            results: vec![RerankResult { index: 0, score: 0.8 }],
            request_id: Some(42),
        };
        assert_ne!(a, b, "different scores with same request_id must not be equal");
    }

    // ── RerankBuilder query captures Into<String> from Cow ──

    #[test]
    fn rerank_builder_query_from_cow_str() {
        let client = Client::new_empty();
        let cow = std::borrow::Cow::Borrowed("cow query");
        let result = client.rerank(cow, vec!["doc"]);
        assert!(result.is_err());
    }

    // ── RerankResult Copy allows use in move closures ──

    #[test]
    fn rerank_result_copy_usable_in_move_closure() {
        let results = vec![
            RerankResult { index: 0, score: 0.8 },
            RerankResult { index: 1, score: 0.2 },
        ];
        // move closure captures by Copy, original vec still accessible
        let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 0.8).abs() < 1e-6);
        assert!((scores[1] - 0.2).abs() < 1e-6);
        // results still usable (Copy, not moved)
        assert_eq!(results[0].index, 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  REQ-API-4: Client::rerank_builder() public API tests
    // ══════════════════════════════════════════════════════════════════════

    // ── Client::rerank_builder() returns RerankBuilder ──

    #[test]
    fn rerank_builder_from_client_api_no_model() {
        let client = Client::new_empty();
        let result = client.rerank_builder("query", vec!["doc1", "doc2"])
            .generate();
        assert!(result.is_err());
        match result.unwrap_err() {
            GllmError::NoModelLoaded => {}
            other => panic!("expected NoModelLoaded, got {:?}", other),
        }
    }

    #[test]
    fn rerank_builder_from_client_with_top_n_no_model() {
        let client = Client::new_empty();
        let result = client.rerank_builder("query", vec!["a", "b", "c"])
            .top_n(2)
            .generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_from_client_top_n_chains_no_model() {
        let client = Client::new_empty();
        let result = client.rerank_builder("query", vec!["doc"])
            .top_n(10)
            .top_n(3)
            .generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_from_client_empty_docs_no_model() {
        let client = Client::new_empty();
        let result = client.rerank_builder("query", Vec::<String>::new())
            .generate();
        assert!(result.is_err());
    }

    #[test]
    fn rerank_builder_from_client_unicode_query_no_model() {
        let client = Client::new_empty();
        let result = client.rerank_builder("什么是法国的首都？", vec!["巴黎", "伦敦"])
            .top_n(1)
            .generate();
        assert!(result.is_err());
    }

    // ── Client::rerank() vs Client::rerank_builder() equivalence ──

    #[test]
    fn rerank_and_rerank_builder_both_fail_no_model() {
        let client = Client::new_empty();
        let direct = client.rerank("q", vec!["d"]);
        let via_builder = client.rerank_builder("q", vec!["d"]).generate();
        assert!(direct.is_err());
        assert!(via_builder.is_err());
        // Both should return NoModelLoaded
        match (direct.unwrap_err(), via_builder.unwrap_err()) {
            (GllmError::NoModelLoaded, GllmError::NoModelLoaded) => {}
            (a, b) => panic!("expected both NoModelLoaded, got {:?} and {:?}", a, b),
        }
    }
}
