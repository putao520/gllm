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
}
