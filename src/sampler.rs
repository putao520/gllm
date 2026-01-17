//! Sampling operations using gllm-kernels.

// Re-export SamplingConfig from gllm-kernels for convenience
pub use gllm_kernels::SamplingConfig;
use gllm_kernels::sample_tokens;

/// Sample the next token from logits using the given sampling configuration.
///
/// * `logits` - Logits slice of shape [batch, vocab_size]
pub fn sample_next_token(
    logits: &[f32],
    batch_size: usize,
    vocab_size: usize,
    config: &SamplingConfig,
) -> Vec<u32> {
    sample_tokens(logits, batch_size, vocab_size, config)
}

/// Sample the next token from 3D logits [batch, seq_len, vocab_size].
pub fn sample_next_token_3d(
    logits: &[f32],
    batch: usize,
    seq_len: usize,
    vocab_size: usize,
    config: &SamplingConfig,
) -> Vec<u32> {
    let start = (seq_len - 1) * vocab_size;
    let mut last_logits = Vec::with_capacity(batch * vocab_size);
    for b in 0..batch {
        let row_start = b * seq_len * vocab_size + start;
        last_logits.extend_from_slice(&logits[row_start..row_start + vocab_size]);
    }
    sample_next_token(&last_logits, batch, vocab_size, config)
}

/// Greedy decode: select the token with highest probability.
pub fn greedy_decode(logits: &[f32], batch_size: usize, vocab_size: usize) -> Vec<u32> {
    sample_next_token(logits, batch_size, vocab_size, &SamplingConfig::greedy())
}
