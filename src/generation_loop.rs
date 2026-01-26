use crate::generation::{FinishReason, GenerationConfig, GenerationOptions, GenerationOutput};
use crate::kv_cache::{KVCache, KvCompressionStrategy};
use crate::prompt_cache::{PromptCache, PromptCacheSnapshot};
use crate::scratch_buffer::{ScratchBuffer, ScratchConfig};
use crate::engine::TokenizerAdapter;
use crate::types::{Error, Result};
use gllm_kernels::{sample_tokens, SamplingConfig};
use std::sync::Mutex;

pub(crate) struct ForwardOutput {
    pub logits: Vec<f32>,
    pub last_hidden: Vec<f32>,
}

pub(crate) trait GenerationOps {
    fn forward_with_hidden(
        &self,
        input_ids: &[u32],
        cache: &mut KVCache,
        scratch: Option<&mut ScratchBuffer>,
    ) -> Result<ForwardOutput>;
    fn logits_from_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>>;
    fn vocab_size(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
    fn max_position_embeddings(&self) -> usize;
    fn scratch_config(&self) -> ScratchConfig;
}

pub(crate) fn generate_with_ops<M: GenerationOps>(
    model: &M,
    prompt_ids: Vec<i64>,
    config: &GenerationConfig,
    tokenizer: &TokenizerAdapter,
    options: &GenerationOptions,
    prompt_cache: Option<&Mutex<PromptCache>>,
) -> Result<GenerationOutput> {
    if prompt_ids.is_empty() {
        return Err(Error::InvalidConfig(
            "Prompt is required for generation".into(),
        ));
    }

    let mut prompt_tokens = Vec::with_capacity(prompt_ids.len());
    for token in prompt_ids {
        if token < 0 {
            return Err(Error::InvalidConfig(
                "Prompt tokens must be non-negative".into(),
            ));
        }
        let token_u32 = u32::try_from(token).map_err(|_| {
            Error::InvalidConfig("Prompt token exceeds supported range".into())
        })?;
        prompt_tokens.push(token_u32);
    }

    let prompt_len = prompt_tokens.len();
    let max_positions = model.max_position_embeddings();
    if max_positions > 0 && prompt_len > max_positions {
        return Err(Error::InvalidConfig(format!(
            "Prompt length {} exceeds max position {}",
            prompt_len, max_positions
        )));
    }

    if config.max_new_tokens == 0 {
        let tokens: Vec<i64> = prompt_tokens.iter().map(|t| *t as i64).collect();
        let text = tokenizer.decode(&tokens);
        return Ok(GenerationOutput {
            text,
            tokens,
            finish_reason: FinishReason::MaxTokens,
        });
    }

    let total_len = prompt_len + config.max_new_tokens;
    if max_positions > 0 && total_len > max_positions {
        return Err(Error::InvalidConfig(format!(
            "Requested length {} exceeds max position {}",
            total_len, max_positions
        )));
    }

    let kv_compression = options
        .kv_compression
        .unwrap_or(KvCompressionStrategy::None);
    let cache_capacity = if max_positions > 0 {
        total_len.min(max_positions)
    } else {
        total_len
    };
    let mut cache = match kv_compression {
        KvCompressionStrategy::None => KVCache::new(
            model.num_layers(),
            model.num_kv_heads(),
            model.head_dim(),
            cache_capacity,
        ),
        _ => KVCache::new_with_compression(
            model.num_layers(),
            model.num_kv_heads(),
            model.head_dim(),
            cache_capacity,
            kv_compression,
        )?,
    };

    let mut scratch = if options.use_scratch_buffer {
        Some(ScratchBuffer::from_config(
            model.scratch_config(),
            1,
            prompt_len.max(1),
        ))
    } else {
        None
    };

    let mut output_tokens: Vec<i64> = prompt_tokens.iter().map(|t| *t as i64).collect();
    if let Some(last) = output_tokens.last() {
        if config.stop_tokens.contains(last) {
            let text = tokenizer.decode(&output_tokens);
            return Ok(GenerationOutput {
                text,
                tokens: output_tokens,
                finish_reason: FinishReason::StopToken,
            });
        }
    }

    let mut prompt_cache_snapshot = None;
    let mut use_prompt_cache = false;
    if let (Some(entries), Some(prompt_cache)) =
        (options.prompt_cache_entries, prompt_cache)
    {
        let mut guard = prompt_cache.lock().map_err(|_| {
            Error::InternalError("Prompt cache lock poisoned".into())
        })?;
        guard.set_max_entries(entries);
        if entries > 0 {
            use_prompt_cache = true;
            prompt_cache_snapshot = guard.lookup(&prompt_tokens, kv_compression);
        }
    }

    let mut logits = if let Some(snapshot) = prompt_cache_snapshot.take() {
        apply_prompt_cache(&mut cache, &snapshot)?;
        model.logits_from_hidden(&snapshot.last_hidden)?
    } else {
        let ForwardOutput { logits, last_hidden } = model.forward_with_hidden(
            &prompt_tokens,
            &mut cache,
            scratch.as_mut(),
        )?;
        if use_prompt_cache {
            if let Some(prompt_cache) = prompt_cache {
                let snapshot = snapshot_from_cache(
                    &cache,
                    model.num_layers(),
                    &prompt_tokens,
                    last_hidden,
                    kv_compression,
                )?;
                let mut guard = prompt_cache.lock().map_err(|_| {
                    Error::InternalError("Prompt cache lock poisoned".into())
                })?;
                guard.insert(
                    snapshot.tokens,
                    snapshot.k_cache,
                    snapshot.v_cache,
                    snapshot.cached_len,
                    snapshot.total_len,
                    snapshot.last_hidden,
                    snapshot.compression,
                );
            }
        }
        logits
    };

    let mut finish_reason = FinishReason::MaxTokens;
    for _ in 0..config.max_new_tokens {
        let next_token = sample_from_logits(&logits, model.vocab_size(), config)?;
        output_tokens.push(next_token as i64);
        if config.stop_tokens.contains(&(next_token as i64)) {
            finish_reason = FinishReason::StopToken;
            break;
        }
        let forward = model.forward_with_hidden(
            &[next_token],
            &mut cache,
            scratch.as_mut(),
        )?;
        logits = forward.logits;
    }

    let text = tokenizer.decode(&output_tokens);
    Ok(GenerationOutput {
        text,
        tokens: output_tokens,
        finish_reason,
    })
}

fn sample_from_logits(
    logits: &[f32],
    vocab_size: usize,
    config: &GenerationConfig,
) -> Result<u32> {
    if logits.len() != vocab_size {
        return Err(Error::InferenceError(
            "Logits length mismatch during sampling".into(),
        ));
    }
    let sampling_config = SamplingConfig {
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        seed: None,
    };
    let sampled = sample_tokens(logits, 1, vocab_size, &sampling_config);
    sampled
        .into_iter()
        .next()
        .ok_or_else(|| Error::InferenceError("Sampling produced empty output".into()))
}

fn apply_prompt_cache(cache: &mut KVCache, snapshot: &PromptCacheSnapshot) -> Result<()> {
    cache.load_from_snapshot(snapshot.total_len, &snapshot.k_cache, &snapshot.v_cache)
}

fn snapshot_from_cache(
    cache: &KVCache,
    num_layers: usize,
    prompt_tokens: &[u32],
    last_hidden: Vec<f32>,
    compression: KvCompressionStrategy,
) -> Result<PromptCacheSnapshot> {
    let mut k_cache = Vec::with_capacity(num_layers);
    let mut v_cache = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        k_cache.push(cache.layer_k(layer)?.to_vec());
        v_cache.push(cache.layer_v(layer)?.to_vec());
    }
    Ok(PromptCacheSnapshot {
        tokens: prompt_tokens.to_vec(),
        k_cache,
        v_cache,
        cached_len: cache.cached_len(),
        total_len: prompt_tokens.len(),
        last_hidden,
        compression,
    })
}
