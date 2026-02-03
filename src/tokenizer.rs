//! Tokenizer integration (prompt <-> tokens).

use std::path::Path;

use thiserror::Error;
use tokenizers::Tokenizer;

use crate::loader::Loader;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("tokenizer.json not found in model files")]
    MissingTokenizer,
    #[error("tokenizers error: {0}")]
    Tokenizers(String),
}

pub type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug)]
pub struct TokenizerHandle {
    tokenizer: Tokenizer,
}

impl TokenizerHandle {
    pub fn from_loader(loader: &Loader) -> TokenizerResult<Self> {
        let path = loader
            .tokenizer_path()
            .ok_or(TokenizerError::MissingTokenizer)?;
        Self::from_path(path)
    }

    pub fn from_path(path: &Path) -> TokenizerResult<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TokenizerResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> TokenizerResult<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|err| TokenizerError::Tokenizers(format!("{err}")))
    }
}
