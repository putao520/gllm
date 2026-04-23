//! Tokenizer integration (prompt <-> tokens).

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;

use crate::loader::Loader;
use crate::manifest::ModelKind;

#[derive(Debug, Error)]
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
        // Disable padding: tokenizer.json may ship with padding enabled (e.g.
        // sentence-transformers models pad to a fixed length).  We run
        // single-sequence inference, so padding is unnecessary and harmful —
        // it inflates seq_len and corrupts mean-pooling / attention.
        tokenizer.with_padding(None);
        Ok(Self { tokenizer, model_kind })
    }

    /// Encode prompt, applying chat template for Chat models.
    pub fn encode_prompt(&self, text: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        let prompt = if self.model_kind == ModelKind::Chat {
            self.apply_chatml_template(text)
        } else {
            text.to_string()
        };
        self.encode(&prompt, add_special_tokens)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode a pair of texts (e.g., query + document for cross-encoder reranking).
    /// Produces `[CLS] text_a [SEP] text_b [SEP]` with proper segment handling.
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

    /// Apply ChatML template for instruct models.
    /// Format: `<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
    fn apply_chatml_template(&self, prompt: &str) -> String {
        format!(
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        )
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
}
