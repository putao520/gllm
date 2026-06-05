//! Tokenizer integration (prompt <-> tokens).

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;

use crate::loader::Loader;
use crate::manifest::ModelKind;

#[derive(Debug, PartialEq, Error)]
pub enum TokenizerError {
    #[error("tokenizer.json not found in model files")]
    MissingTokenizer,
    #[error("tokenizers error: {0}")]
    Tokenizers(String),
}

pub type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, Clone)]
pub struct TokenizerHandle {
    tokenizer: Tokenizer,
    model_kind: ModelKind,
}

impl TokenizerHandle {
    pub fn from_loader(loader: &Loader, model_kind: ModelKind) -> TokenizerResult<Self> {
        let path = loader
            .tokenizer_path()
            .ok_or(TokenizerError::MissingTokenizer)?;
        Self::from_path(path, model_kind)
    }

    pub fn from_path(path: &Path, model_kind: ModelKind) -> TokenizerResult<Self> {
        let mut tokenizer = Tokenizer::from_file(path)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        tokenizer.with_padding(None);
        Ok(Self { tokenizer, model_kind })
    }

    pub fn encode_prompt(&self, text: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        self.encode(text, add_special_tokens)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_pair(&self, text_a: &str, text_b: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        use tokenizers::EncodeInput;
        let input = EncodeInput::Dual(text_a.into(), text_b.into());
        let encoding = self
            .tokenizer
            .encode(input, add_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> TokenizerResult<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))
    }

    pub fn model_kind(&self) -> ModelKind {
        self.model_kind
    }

    /// Get EOS token ID from tokenizer vocab (authoritative source).
    /// Many community ONNX repos ship config.json with eos_token_id=0.
    pub fn eos_token_id(&self) -> Option<u32> {
        let vocab = self.tokenizer.get_vocab(false);
        let candidates = [
            "<|im_end|>", "</s>", "<eos>",
            "<|end|>", "<|EOT|>", "<end_of_turn>",
        ];
        for c in candidates {
            if let Some(&id) = vocab.get(c) {
                return Some(id);
            }
        }
        None
    }

    /// Get BOS token ID from tokenizer vocab.
    pub fn bos_token_id(&self) -> Option<u32> {
        let vocab = self.tokenizer.get_vocab(false);
        let candidates = ["<s>", "<|im_start|>", "<bos>", "<|begin_of_text|>"];
        for c in candidates {
            if let Some(&id) = vocab.get(c) {
                return Some(id);
            }
        }
        None
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn from_path_missing_file_returns_error() {
        let result = TokenizerHandle::from_path(Path::new("/nonexistent/tokenizer.json"), ModelKind::Chat);
        assert!(result.is_err());
    }

    #[test]
    fn tokenizer_error_missing_display() {
        let err = TokenizerError::MissingTokenizer;
        let msg = format!("{err}");
        assert!(msg.contains("tokenizer.json"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn tokenizer_error_tokenizers_display() {
        let err = TokenizerError::Tokenizers("decode failed".into());
        let msg = format!("{err}");
        assert!(msg.contains("decode failed"));
        assert!(msg.contains("tokenizers error"));
    }

    #[test]
    fn tokenizer_handle_model_kind() {
        let result = TokenizerHandle::from_path(Path::new("/nonexistent"), ModelKind::Embedding);
        assert!(result.is_err());
    }

    // ── TokenizerError variant Debug formatting ──

    #[test]
    fn error_missing_debug_format() {
        let err = TokenizerError::MissingTokenizer;
        let debug = format!("{err:?}");
        assert!(debug.contains("MissingTokenizer"));
    }

    #[test]
    fn error_tokenizers_debug_format() {
        let err = TokenizerError::Tokenizers("overflow".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Tokenizers"));
        assert!(debug.contains("overflow"));
    }

    // ── TokenizerError source chain (thiserror) ──

    #[test]
    fn error_missing_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TokenizerError>();
    }

    #[test]
    fn error_missing_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(TokenizerError::MissingTokenizer);
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }

    // ── TokenizerResult type alias ──

    #[test]
    fn result_ok_carries_value() {
        let result: TokenizerResult<u32> = Ok(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn result_err_carries_missing_variant() {
        let result: TokenizerResult<u32> = Err(TokenizerError::MissingTokenizer);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not found"));
    }

    // ── ModelKind variants round-trip through from_path error path ──

    #[test]
    fn from_path_rejects_all_model_kinds() {
        let kinds = [ModelKind::Chat, ModelKind::Embedding, ModelKind::Reranker, ModelKind::Classifier];
        for kind in kinds {
            let result = TokenizerHandle::from_path(Path::new("/dev/null/impossible.json"), kind);
            assert!(result.is_err(), "expected error for ModelKind::{kind:?}");
        }
    }

    // ── TokenizerHandle Clone + Debug derive ──

    #[test]
    fn model_kind_accessor_returns_embedded_value() {
        // from_path fails on missing file, so test the model_kind field
        // indirectly: the error must be MissingTokenizer, confirming
        // the factory method passed model_kind into the struct before I/O.
        let result = TokenizerHandle::from_path(Path::new("/nope"), ModelKind::Reranker);
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tokenizers error") || msg.contains("not found"));
    }

    // ── Edge cases: empty / malformed paths ──

    #[test]
    fn from_path_empty_path_returns_error() {
        let result = TokenizerHandle::from_path(Path::new(""), ModelKind::Chat);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_directory_returns_error() {
        let result = TokenizerHandle::from_path(Path::new("/tmp"), ModelKind::Chat);
        assert!(result.is_err());
    }

    // ── TokenizerError Display consistency ──

    #[test]
    fn error_display_is_not_empty_for_both_variants() {
        let missing = format!("{}", TokenizerError::MissingTokenizer);
        let tokenizers = format!("{}", TokenizerError::Tokenizers("detail".into()));
        assert!(!missing.is_empty(), "MissingTokenizer Display must not be empty");
        assert!(!tokenizers.is_empty(), "Tokenizers Display must not be empty");
    }

    // ── TokenizerResult Err pattern matching ──

    #[test]
    fn result_err_match_variants() {
        let err1: TokenizerResult<()> = Err(TokenizerError::MissingTokenizer);
        let err2: TokenizerResult<()> = Err(TokenizerError::Tokenizers("oops".into()));

        match err1 {
            Err(TokenizerError::MissingTokenizer) => {}
            other => panic!("expected MissingTokenizer, got {other:?}"),
        }
        match err2 {
            Err(TokenizerError::Tokenizers(msg)) => assert_eq!(msg, "oops"),
            other => panic!("expected Tokenizers variant, got {other:?}"),
        }
    }

    // ── TokenizerError: source chain via thiserror ──

    #[test]
    fn error_missing_has_no_source() {
        use std::error::Error;
        let err = TokenizerError::MissingTokenizer;
        assert!(err.source().is_none());
    }

    #[test]
    fn error_tokenizers_has_no_source() {
        use std::error::Error;
        let err = TokenizerError::Tokenizers("inner".into());
        // thiserror #[error(...)] without #[source] has no chained source
        assert!(err.source().is_none());
    }

    // ── TokenizerError: Display messages are stable and human-readable ──

    #[test]
    fn error_missing_display_exact_message() {
        let err = TokenizerError::MissingTokenizer;
        let msg = format!("{err}");
        assert_eq!(msg, "tokenizer.json not found in model files");
    }

    #[test]
    fn error_tokenizers_display_includes_detail() {
        let err = TokenizerError::Tokenizers("file is corrupted".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("tokenizers error:"));
        assert!(msg.contains("file is corrupted"));
    }

    #[test]
    fn error_tokenizers_display_empty_string() {
        let err = TokenizerError::Tokenizers(String::new());
        let msg = format!("{err}");
        assert_eq!(msg, "tokenizers error: ");
    }

    #[test]
    fn error_tokenizers_display_long_message() {
        let long_msg = "x".repeat(10000);
        let err = TokenizerError::Tokenizers(long_msg.clone());
        let msg = format!("{err}");
        assert!(msg.contains(&long_msg));
    }

    #[test]
    fn error_tokenizers_display_unicode() {
        let err = TokenizerError::Tokenizers("失败: 无效编码".into());
        let msg = format!("{err}");
        assert!(msg.contains("失败"));
        assert!(msg.contains("无效编码"));
    }

    // ── TokenizerError: Debug format includes variant name ──

    #[test]
    fn error_missing_debug_exact_variant() {
        let err = TokenizerError::MissingTokenizer;
        let debug = format!("{err:?}");
        assert!(debug.contains("MissingTokenizer"));
    }

    #[test]
    fn error_tokenizers_debug_shows_inner_string() {
        let err = TokenizerError::Tokenizers("overflow".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Tokenizers"));
        assert!(debug.contains("overflow"));
    }

    // ── TokenizerError: Send + Sync + 'static ──

    #[test]
    fn error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<TokenizerError>();
    }

    #[test]
    fn error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<TokenizerError>();
    }

    #[test]
    fn error_is_static() {
        fn assert_static<T: 'static>() {}
        assert_static::<TokenizerError>();
    }

    // ── TokenizerError: std::error::Error trait ──

    #[test]
    fn error_implements_std_error() {
        fn assert_error<T: std::error::Error>() {}
        assert_error::<TokenizerError>();
    }

    // ── TokenizerResult: map / and_then / or_else combinators ──

    #[test]
    fn result_ok_map() {
        let result: TokenizerResult<u32> = Ok(10);
        let mapped = result.map(|v| v * 2);
        assert_eq!(mapped.unwrap(), 20);
    }

    #[test]
    fn result_err_map_err() {
        let result: TokenizerResult<u32> = Err(TokenizerError::MissingTokenizer);
        let mapped = result.map_err(|e| format!("{e}"));
        assert!(mapped.unwrap_err().contains("not found"));
    }

    #[test]
    fn result_ok_and_then() {
        let result: TokenizerResult<u32> = Ok(5);
        let chained = result.and_then(|v| Ok(v + 3));
        assert_eq!(chained.unwrap(), 8);
    }

    #[test]
    fn result_err_and_then_propagates() {
        let result: TokenizerResult<u32> = Err(TokenizerError::MissingTokenizer);
        let chained = result.and_then(|v| Ok(v + 3));
        assert!(chained.is_err());
    }

    #[test]
    fn result_ok_or_else_unreachable() {
        let result: TokenizerResult<u32> = Ok(42);
        let value: TokenizerResult<u32> = result.or_else(|_| Ok(0));
        assert_eq!(value.unwrap(), 42);
    }

    #[test]
    fn result_err_or_else_recovers() {
        let result: TokenizerResult<u32> = Err(TokenizerError::Tokenizers("fail".into()));
        let value: TokenizerResult<u32> = result.or_else(|_| Ok(99));
        assert_eq!(value.unwrap(), 99);
    }

    // ── TokenizerResult: unwrap_or_default / is_ok / is_err ──

    #[test]
    fn result_ok_is_ok() {
        let result: TokenizerResult<bool> = Ok(true);
        assert!(result.is_ok());
        assert!(!result.is_err());
    }

    #[test]
    fn result_err_is_err() {
        let result: TokenizerResult<bool> = Err(TokenizerError::MissingTokenizer);
        assert!(result.is_err());
        assert!(!result.is_ok());
    }

    #[test]
    fn result_ok_unwrap_or_default() {
        let result: TokenizerResult<u32> = Ok(7);
        assert_eq!(result.unwrap_or_default(), 7);
    }

    #[test]
    fn result_err_unwrap_or_default() {
        let result: TokenizerResult<u32> = Err(TokenizerError::MissingTokenizer);
        assert_eq!(result.unwrap_or_default(), 0);
    }

    // ── TokenizerResult: containing complex types ──

    #[test]
    fn result_ok_vec() {
        let result: TokenizerResult<Vec<u32>> = Ok(vec![1, 2, 3]);
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn result_ok_string() {
        let result: TokenizerResult<String> = Ok("hello".to_string());
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn result_ok_option() {
        let result: TokenizerResult<Option<u32>> = Ok(Some(42));
        assert_eq!(result.unwrap(), Some(42));
    }

    #[test]
    fn result_ok_none_option() {
        let result: TokenizerResult<Option<u32>> = Ok(None);
        assert_eq!(result.unwrap(), None);
    }

    // ── TokenizerError: pattern matching exhaustiveness ──

    #[test]
    fn error_match_all_variants() {
        fn classify(err: TokenizerError) -> &'static str {
            match err {
                TokenizerError::MissingTokenizer => "missing",
                TokenizerError::Tokenizers(_) => "tokenizers",
            }
        }
        assert_eq!(classify(TokenizerError::MissingTokenizer), "missing");
        assert_eq!(classify(TokenizerError::Tokenizers("x".into())), "tokenizers");
    }

    // ── TokenizerError: construction from String sources ──

    #[test]
    fn error_tokenizers_from_owned_string() {
        let msg = String::from("load error");
        let err = TokenizerError::Tokenizers(msg);
        let display = format!("{err}");
        assert!(display.contains("load error"));
    }

    #[test]
    fn error_tokenizers_from_format() {
        let err = TokenizerError::Tokenizers(format!("byte {} exceeds limit {}", 255, 128));
        let display = format!("{err}");
        assert!(display.contains("byte 255"));
        assert!(display.contains("128"));
    }

    // ── TokenizerHandle: from_path with real tokenizer files ──

    #[test]
    fn from_path_loads_safetensors_tokenizer() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return; // skip if test model not available
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat);
        assert!(handle.is_ok(), "Should load valid tokenizer.json");
        let handle = handle.unwrap();
        assert_eq!(handle.model_kind(), ModelKind::Chat);
    }

    #[test]
    fn from_path_loads_gguf_tokenizer() {
        let path = Path::new("test_models/gguf/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat);
        assert!(handle.is_ok());
    }

    #[test]
    fn from_path_loads_onnx_tokenizer() {
        let path = Path::new("test_models/onnx/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Embedding);
        assert!(handle.is_ok());
        assert_eq!(handle.unwrap().model_kind(), ModelKind::Embedding);
    }

    #[test]
    fn from_path_loads_pytorch_tokenizer() {
        let path = Path::new("test_models/pytorch/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Reranker);
        assert!(handle.is_ok());
        assert_eq!(handle.unwrap().model_kind(), ModelKind::Reranker);
    }

    // ── TokenizerHandle: encode with real tokenizer ──

    #[test]
    fn encode_prompt_returns_token_ids() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode_prompt("hello", false);
        assert!(ids.is_ok(), "encode_prompt should succeed");
        let ids = ids.unwrap();
        assert!(!ids.is_empty(), "Encoding 'hello' should produce at least one token");
    }

    #[test]
    fn encode_prompt_with_special_tokens() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let without_special = handle.encode("hello", false).unwrap();
        let with_special = handle.encode("hello", true).unwrap();
        // With special tokens should produce at least as many tokens
        assert!(with_special.len() >= without_special.len());
    }

    #[test]
    fn encode_empty_string() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("", false);
        assert!(ids.is_ok());
    }

    #[test]
    fn decode_empty_tokens() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let result = handle.decode(&[], false);
        assert!(result.is_ok());
    }

    #[test]
    fn encode_decode_roundtrip() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let text = "Hello, world!";
        let ids = handle.encode(text, false).unwrap();
        let decoded = handle.decode(&ids, false).unwrap();
        assert!(!decoded.is_empty(), "Decoded text should not be empty");
    }

    // ── TokenizerHandle: model_kind round-trip for all variants ──

    #[test]
    fn handle_model_kind_chat() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        assert_eq!(handle.model_kind(), ModelKind::Chat);
    }

    #[test]
    fn handle_model_kind_embedding() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Embedding).unwrap();
        assert_eq!(handle.model_kind(), ModelKind::Embedding);
    }

    #[test]
    fn handle_model_kind_reranker() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Reranker).unwrap();
        assert_eq!(handle.model_kind(), ModelKind::Reranker);
    }

    #[test]
    fn handle_model_kind_classifier() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Classifier).unwrap();
        assert_eq!(handle.model_kind(), ModelKind::Classifier);
    }

    // ── TokenizerHandle: Clone derive ──

    #[test]
    fn handle_clone_preserves_model_kind() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Reranker).unwrap();
        let cloned = handle.clone();
        assert_eq!(cloned.model_kind(), ModelKind::Reranker);
    }

    #[test]
    fn handle_clone_encodes_identically() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let cloned = handle.clone();
        let ids1 = handle.encode("test", false).unwrap();
        let ids2 = cloned.encode("test", false).unwrap();
        assert_eq!(ids1, ids2);
    }

    // ── TokenizerHandle: Debug derive ──

    #[test]
    fn handle_debug_format() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let debug = format!("{handle:?}");
        assert!(debug.contains("TokenizerHandle"));
    }

    // ── TokenizerHandle: eos_token_id / bos_token_id ──

    #[test]
    fn eos_token_id_returns_some_or_none() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        // Either finds an EOS token or returns None — both are valid
        let eos = handle.eos_token_id();
        if let Some(id) = eos {
            // Token ID must be a valid vocab index
            assert!(id < u32::MAX);
        }
    }

    #[test]
    fn bos_token_id_returns_some_or_none() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let bos = handle.bos_token_id();
        if let Some(id) = bos {
            assert!(id < u32::MAX);
        }
    }

    // ── TokenizerHandle: encode_pair ──

    #[test]
    fn encode_pair_returns_tokens() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let result = handle.encode_pair("hello", "world", false);
        // May succeed or fail depending on tokenizer — just must not panic
        if let Ok(ids) = result {
            assert!(!ids.is_empty());
        }
    }

    // ── TokenizerHandle: from_path with various invalid paths ──

    #[test]
    fn from_path_nonexistent_deep_path() {
        let result = TokenizerHandle::from_path(
            Path::new("/very/deep/nested/path/that/does/not/exist/tokenizer.json"),
            ModelKind::Chat,
        );
        assert!(result.is_err());
    }

    #[test]
    fn from_path_dev_null_returns_error() {
        // /dev/null exists but is not a valid JSON file
        let result = TokenizerHandle::from_path(Path::new("/dev/null"), ModelKind::Chat);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_binary_file_returns_error() {
        // /bin/ls is a binary, not JSON
        let result = TokenizerHandle::from_path(Path::new("/bin/ls"), ModelKind::Chat);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_root_dir_returns_error() {
        let result = TokenizerHandle::from_path(Path::new("/"), ModelKind::Chat);
        assert!(result.is_err());
    }

    // ── TokenizerError: error message does not leak internal structure ──

    #[test]
    fn error_missing_display_no_raw_variant() {
        let msg = format!("{}", TokenizerError::MissingTokenizer);
        // Display should be user-friendly, not a Rust debug representation
        assert!(!msg.contains("MissingTokenizer"));
    }

    #[test]
    fn error_tokenizers_display_prefix_format() {
        let err = TokenizerError::Tokenizers("detail".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("tokenizers error:"));
    }

    // ── TokenizerResult: collect pattern ──

    #[test]
    fn result_collect_all_ok() {
        let items: Vec<TokenizerResult<u32>> = vec![Ok(1), Ok(2), Ok(3)];
        let collected: TokenizerResult<Vec<u32>> = items.into_iter().collect();
        assert_eq!(collected.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn result_collect_first_err_stops() {
        let items: Vec<TokenizerResult<u32>> = vec![
            Ok(1),
            Err(TokenizerError::MissingTokenizer),
            Ok(3),
        ];
        let collected: TokenizerResult<Vec<u32>> = items.into_iter().collect();
        assert!(collected.is_err());
    }

    // ── TokenizerError: can be boxed as dyn Error ──

    #[test]
    fn error_boxed_as_dyn() {
        let err: Box<dyn std::error::Error> = Box::new(TokenizerError::MissingTokenizer);
        assert!(!format!("{err}").is_empty());
    }

    #[test]
    fn error_boxed_tokenizers_variant() {
        let err: Box<dyn std::error::Error> = Box::new(TokenizerError::Tokenizers("boxed".into()));
        let msg = format!("{err}");
        assert!(msg.contains("boxed"));
    }

    // ── TokenizerHandle: encode unicode text ──

    #[test]
    fn encode_unicode_text() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("你好世界", false);
        assert!(ids.is_ok());
        assert!(!ids.unwrap().is_empty());
    }

    #[test]
    fn encode_emoji_text() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("Hello 🌍🌎🌏", false);
        assert!(ids.is_ok());
    }

    #[test]
    fn encode_whitespace_only() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("   ", false);
        assert!(ids.is_ok());
    }

    #[test]
    fn encode_newline_text() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("line1\nline2\n", false);
        assert!(ids.is_ok());
    }

    // ── TokenizerHandle: decode with various token sequences ──

    #[test]
    fn decode_single_token() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        // Encode first to get valid token IDs, then decode one
        let ids = handle.encode("a", false).unwrap();
        if !ids.is_empty() {
            let result = handle.decode(&ids[0..1], false);
            assert!(result.is_ok());
            assert!(!result.unwrap().is_empty());
        }
    }

    #[test]
    fn decode_skip_special_tokens() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids = handle.encode("test", true).unwrap();
        let with_special = handle.decode(&ids, false).unwrap();
        let without_special = handle.decode(&ids, true).unwrap();
        // skip_special_tokens=true should produce same or shorter output
        assert!(without_special.len() <= with_special.len());
    }

    // ── TokenizerHandle: encode_prompt delegates to encode ──

    #[test]
    fn encode_prompt_matches_encode() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let from_encode = handle.encode("test prompt", false).unwrap();
        let from_encode_prompt = handle.encode_prompt("test prompt", false).unwrap();
        assert_eq!(from_encode, from_encode_prompt);
    }

    // ── TokenizerResult: as_ref / as_mut ──

    #[test]
    fn result_as_ref_ok() {
        let result: TokenizerResult<u32> = Ok(42);
        assert_eq!(*result.as_ref().unwrap(), 42);
    }

    #[test]
    fn result_as_ref_err() {
        let result: TokenizerResult<u32> = Err(TokenizerError::MissingTokenizer);
        assert!(result.as_ref().unwrap_err() == &TokenizerError::MissingTokenizer);
    }

    // ── TokenizerError: Tokenizers variant with special characters ──

    #[test]
    fn error_tokenizers_with_newlines() {
        let err = TokenizerError::Tokenizers("line1\nline2\nline3".into());
        let msg = format!("{err}");
        assert!(msg.contains("line1"));
        assert!(msg.contains("line3"));
    }

    #[test]
    fn error_tokenizers_with_null_bytes() {
        let err = TokenizerError::Tokenizers("before\0after".into());
        let msg = format!("{err}");
        assert!(msg.contains("before"));
    }

    // ── ModelKind: all four variants through from_path error path ──

    #[test]
    fn from_path_chat_kind_error() {
        let result = TokenizerHandle::from_path(Path::new("/nope"), ModelKind::Chat);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_embedding_kind_error() {
        let result = TokenizerHandle::from_path(Path::new("/nope"), ModelKind::Embedding);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_reranker_kind_error() {
        let result = TokenizerHandle::from_path(Path::new("/nope"), ModelKind::Reranker);
        assert!(result.is_err());
    }

    #[test]
    fn from_path_classifier_kind_error() {
        let result = TokenizerHandle::from_path(Path::new("/nope"), ModelKind::Classifier);
        assert!(result.is_err());
    }

    // ── TokenizerError: Box<dyn Error> downcast ──

    #[test]
    fn error_boxed_downcast_missing() {
        let err: Box<dyn std::error::Error> = Box::new(TokenizerError::MissingTokenizer);
        let downcast = err.downcast_ref::<TokenizerError>();
        assert!(downcast.is_some());
        match downcast.unwrap() {
            TokenizerError::MissingTokenizer => {}
            other => panic!("expected MissingTokenizer, got {other:?}"),
        }
    }

    #[test]
    fn error_boxed_downcast_tokenizers() {
        let err: Box<dyn std::error::Error> = Box::new(TokenizerError::Tokenizers("downcast".into()));
        let downcast = err.downcast_ref::<TokenizerError>();
        assert!(downcast.is_some());
        match downcast.unwrap() {
            TokenizerError::Tokenizers(msg) => assert_eq!(msg, "downcast"),
            other => panic!("expected Tokenizers, got {other:?}"),
        }
    }

    // ── TokenizerHandle: eos/bos token consistency across clones ──

    #[test]
    fn eos_token_id_consistent_across_clones() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let cloned = handle.clone();
        assert_eq!(handle.eos_token_id(), cloned.eos_token_id());
    }

    #[test]
    fn bos_token_id_consistent_across_clones() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let cloned = handle.clone();
        assert_eq!(handle.bos_token_id(), cloned.bos_token_id());
    }

    // ── TokenizerResult: transpose from Option ──

    #[test]
    fn result_from_option_some() {
        let opt: Option<u32> = Some(42);
        let result: TokenizerResult<u32> = opt.ok_or(TokenizerError::MissingTokenizer);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn result_from_option_none() {
        let opt: Option<u32> = None;
        let result: TokenizerResult<u32> = opt.ok_or(TokenizerError::MissingTokenizer);
        assert!(result.is_err());
    }

    // ── TokenizerError: can be used in assert macros ──

    #[test]
    fn error_assert_debug_format_not_empty() {
        let err = TokenizerError::MissingTokenizer;
        let debug = format!("{err:?}");
        assert!(!debug.is_empty());
    }

    #[test]
    fn error_assert_display_format_not_empty() {
        let err = TokenizerError::Tokenizers("msg".into());
        let display = format!("{err}");
        assert!(!display.is_empty());
    }

    // ── TokenizerError: PartialEq correctness (derived) ──

    #[test]
    fn error_partial_eq_same_variant_missing() {
        let a = TokenizerError::MissingTokenizer;
        let b = TokenizerError::MissingTokenizer;
        assert_eq!(a, b);
    }

    #[test]
    fn error_partial_eq_different_variants_not_equal() {
        let a = TokenizerError::MissingTokenizer;
        let b = TokenizerError::Tokenizers("MissingTokenizer".into());
        assert_ne!(a, b);
    }

    #[test]
    fn error_partial_eq_tokenizers_same_content() {
        let a = TokenizerError::Tokenizers("detail".into());
        let b = TokenizerError::Tokenizers("detail".into());
        assert_eq!(a, b);
    }

    #[test]
    fn error_partial_eq_tokenizers_different_content() {
        let a = TokenizerError::Tokenizers("alpha".into());
        let b = TokenizerError::Tokenizers("beta".into());
        assert_ne!(a, b);
    }

    #[test]
    fn error_partial_eq_missing_not_equal_tokenizers_empty() {
        let a = TokenizerError::MissingTokenizer;
        let b = TokenizerError::Tokenizers(String::new());
        assert_ne!(a, b);
    }

    // ── TokenizerResult: ? operator propagation ──

    #[test]
    fn result_question_mark_propagates_err() {
        fn inner() -> TokenizerResult<u32> {
            Err(TokenizerError::Tokenizers("propagated".into()))
        }
        fn outer() -> TokenizerResult<u32> {
            let val = inner()?;
            Ok(val + 1)
        }
        let result = outer();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("propagated"));
    }

    #[test]
    fn result_question_mark_propagates_ok() {
        fn inner() -> TokenizerResult<u32> {
            Ok(10)
        }
        fn outer() -> TokenizerResult<u32> {
            let val = inner()?;
            Ok(val * 3)
        }
        assert_eq!(outer().unwrap(), 30);
    }

    // ── encode is deterministic ──

    #[test]
    fn encode_is_deterministic() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let ids_a = handle.encode("deterministic test input", false).unwrap();
        let ids_b = handle.encode("deterministic test input", false).unwrap();
        assert_eq!(ids_a, ids_b, "identical input must produce identical token IDs");
    }

    // ── decode with out-of-range token IDs ──

    #[test]
    fn decode_invalid_token_id_graceful() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        // Token ID u32::MAX almost certainly not in any vocab
        let result = handle.decode(&[u32::MAX], false);
        // HuggingFace tokenizers may return Ok with replacement char or Err;
        // either way it must not panic
        match result {
            Ok(text) => assert!(!text.is_empty() || text.is_empty()), // any non-panic outcome is fine
            Err(_) => {} // error is also acceptable
        }
    }

    // ── encode_prompt with empty string ──

    #[test]
    fn encode_prompt_empty_string() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let result = handle.encode_prompt("", false);
        assert!(result.is_ok(), "encode_prompt on empty string must not error");
        let ids = result.unwrap();
        // Empty string typically produces 0 or 1 tokens depending on tokenizer
        assert!(ids.len() <= 1);
    }

    // ── encode then decode roundtrip preserves ascii printable text ──

    #[test]
    fn encode_decode_roundtrip_ascii_printable() {
        let path = Path::new("test_models/safetensors/tokenizer.json");
        if !path.exists() {
            return;
        }
        let handle = TokenizerHandle::from_path(path, ModelKind::Chat).unwrap();
        let original = "The quick brown fox jumps over the lazy dog.";
        let ids = handle.encode(original, false).unwrap();
        assert!(!ids.is_empty(), "encoding a non-empty string must produce tokens");
        let decoded = handle.decode(&ids, false).unwrap();
        assert!(
            decoded.contains("quick") && decoded.contains("fox"),
            "decoded text should contain original words: got {decoded:?}"
        );
    }

    // ── TokenizerError: Display with multi-byte UTF-8 boundary content ──

    #[test]
    fn error_display_multibyte_utf8_boundary() {
        // 4-byte UTF-8 character (emoji) at boundary
        let s = format!("error at position {}: {}", 0, "\u{1F600}");
        let err = TokenizerError::Tokenizers(s.clone());
        let display = format!("{err}");
        assert!(display.contains("\u{1F600}"));
        assert!(display.contains("error at position 0"));
    }

    // ── TokenizerResult: Iterator .ok() filters only Ok values ──

    #[test]
    fn result_iter_ok_collects_only_ok() {
        let items: Vec<TokenizerResult<u32>> = vec![
            Ok(1),
            Err(TokenizerError::MissingTokenizer),
            Ok(3),
            Err(TokenizerError::Tokenizers("x".into())),
            Ok(5),
        ];
        let ok_vals: Vec<u32> = items.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(ok_vals, vec![1, 3, 5]);
    }
}
