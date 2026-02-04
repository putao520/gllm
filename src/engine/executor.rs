//! Executor skeleton.

use gllm_kernels::backend_trait::{
    AttentionTopology, Backend, BackendError, KvCacheHandle, LogitsHandle,
};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::{
    GeneratorForwardConfig, KvCacheConfig, PageId, PositionEncoding, SamplingConfig,
};
use thiserror::Error;

use crate::adapter::{AdapterError, AdapterWeights, Message, ModelAdapter};
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use crate::loader::Loader;
use crate::manifest::ModelManifest;
use std::sync::Arc;
use crate::model_config::{ModelConfig, ModelConfigError};
use crate::tokenizer::{TokenizerError, TokenizerHandle};

use super::scheduler::{RequestId, RequestKind, ScheduledBatch, Scheduler};
use super::vllm2024::Scheduler2024Config;

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Config(#[from] ModelConfigError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    KvCache(#[from] KvCacheError),
    #[error("empty prompt tokens")]
    EmptyPrompt,
    #[error("backend returned empty sample")]
    EmptySample,
}

pub type ExecutorResult<T> = std::result::Result<T, ExecutorError>;

pub struct Executor<B: Backend + 'static> {
    backend: B,
    scheduler: Scheduler,
    manifest: Arc<ModelManifest>,
    adapter: &'static dyn ModelAdapter<B>,
    weights: AdapterWeights<B>,
    model_config: ModelConfig,
    forward_config: GeneratorForwardConfig,
    kv_cache_config: KvCacheConfig,
    tokenizer: TokenizerHandle,
    kv_cache: Option<KvCacheDoubleBuffer>,
    kv_cache_slot: KvCacheSlot,
}

impl<B: Backend + 'static> Executor<B> {
    pub fn from_loader(
        backend: B,
        manifest: Arc<ModelManifest>,
        adapter: &'static dyn ModelAdapter<B>,
        loader: &mut Loader,
    ) -> ExecutorResult<Self> {
        let model_config = ModelConfig::from_loader(manifest.as_ref(), loader)?;
        let forward_config = GeneratorForwardConfig {
            num_layers: model_config.num_hidden_layers,
            num_heads: model_config.num_attention_heads,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            max_seq_len: model_config.max_position_embeddings,
            vocab_size: model_config.vocab_size,
            rope_theta: model_config.rope_theta,
            rope_scale: 1.0,
            rope_interleaved: false,
            rope_precompute: true,
            position_encoding: PositionEncoding::Rope,
        };
        let mut scheduler = Scheduler::new();
        scheduler.enable_vllm_2024(Scheduler2024Config {
            enable_2024_optimizations: true,
            ..Scheduler2024Config::default()
        });
        let page_size = scheduler.config().page_size;

        // CPU backend 只支持 f32 dtype for KV cache
        // 如果模型配置是 f16/bf16，需要强制转换为 f32
        let cpu_dtype_size = if std::any::TypeId::of::<B>() == std::any::TypeId::of::<CpuBackend>()
        {
            Some(4) // f32
        } else {
            None
        };

        let kv_cache_config = KvCacheConfig {
            num_layers: model_config.num_hidden_layers,
            num_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            max_seq_len: model_config.max_position_embeddings,
            dtype_size: cpu_dtype_size.unwrap_or(model_config.dtype_size),
            page_size,
            swap_config: None, // 暂不启用 swap
        };
        let tokenizer = TokenizerHandle::from_loader(loader)?;
        let weights = adapter.load_weights(loader, &backend)?;
        Ok(Self {
            backend,
            scheduler,
            manifest,
            adapter,
            weights,
            model_config,
            forward_config,
            kv_cache_config,
            tokenizer,
            kv_cache: None,
            kv_cache_slot: KvCacheSlot::Front,
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn weights(&self) -> &AdapterWeights<B> {
        &self.weights
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn apply_chat_template(&self, messages: &[Message]) -> String {
        self.adapter.apply_chat_template(messages)
    }

    pub fn allocate_kv_cache(&mut self, config: &KvCacheConfig) -> ExecutorResult<KvCacheHandle> {
        let front = self.backend.alloc_kv_cache(config)?;
        let back = self.backend.alloc_kv_cache(config)?;
        let front = KvCacheState::new(front, *config);
        let back = KvCacheState::new(back, *config);
        self.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
        self.kv_cache_slot = KvCacheSlot::Front;
        Ok(self
            .kv_cache
            .as_ref()
            .expect("kv cache set")
            .front()
            .handle())
    }

    pub fn kv_cache(&self) -> Option<KvCacheHandle> {
        self.kv_cache
            .as_ref()
            .map(|cache| cache.slot(self.kv_cache_slot).handle())
    }

    pub fn enqueue(&mut self, kind: RequestKind, prompt: impl Into<String>) -> RequestId {
        self.scheduler.enqueue(kind, prompt)
    }

    pub fn enqueue_with_tokens(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        tokens: usize,
    ) -> RequestId {
        self.scheduler.enqueue_with_tokens(kind, prompt, tokens)
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        self.scheduler.next_batch()
    }

    pub fn encode_prompt(&self, prompt: &str) -> ExecutorResult<Vec<u32>> {
        let add_special_tokens = self.adapter.add_special_tokens();
        Ok(self.tokenizer.encode(prompt, add_special_tokens)?)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> ExecutorResult<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
    }

    pub fn forward_step(&mut self, tokens: &[u32]) -> ExecutorResult<LogitsHandle> {
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let cache_handle = {
            let slot = self.kv_cache_slot;
            let kv_cache = self.ensure_kv_cache()?;
            let active = kv_cache.slot(slot);
            if tokens.len() > active.remaining() {
                return Err(KvCacheError::Exhausted {
                    requested: tokens.len(),
                    available: active.remaining(),
                }
                .into());
            }
            active.handle()
        };

        let mut cache_handle = cache_handle;
        let logits = self.backend.generator_forward_gpu_pure(
            tokens,
            &AttentionTopology::linear(),
            &self.weights,
            &mut cache_handle,
            &self.forward_config,
        )?;

        // Advance the cache after forward completes
        let slot = self.kv_cache_slot;
        let kv_cache = self
            .kv_cache
            .as_mut()
            .expect("kv cache should be allocated");
        let active = kv_cache.slot_mut(slot);
        active.advance(tokens.len())?;
        Ok(logits)
    }

    pub fn sample_from_logits(
        &self,
        logits: &LogitsHandle,
        temperature: f32,
    ) -> ExecutorResult<u32> {
        let sampling = SamplingConfig {
            temperature,
            ..SamplingConfig::default()
        };
        let tokens = self.backend.sample_from_tensor(
            logits,
            &AttentionTopology::linear(),
            self.model_config.vocab_size,
            &sampling,
        )?;
        tokens.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        let input_tokens = self.encode_prompt(prompt)?;
        if input_tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }

        // LMCache lookup (ARCH-SCHED-LMCACHE). We simulate skipping prefill tokens by
        // advancing the KV cache usage without GPU recompute to stay zero-copy/AOT.
        let model_id = self.manifest.model_id.to_string();
        let lmcache_hit = self.scheduler.lmcache_lookup(&model_id, prompt);

        let slot = self.kv_cache_slot;
        let kv_cache = self.ensure_kv_cache()?;
        kv_cache.slot_mut(slot).reset();

        // LMCache fast path: reuse existing KV/logits handles to skip GPU forward.
        let mut logits = if let Some(hit) = lmcache_hit.clone() {
            if let Some(kv_handle) = hit.kv_handle {
                // Rebind current slot to cached handle and logical cursor.
                let state = KvCacheState::new(kv_handle, self.kv_cache_config);
                self.kv_cache
                    .as_mut()
                    .expect("kv cache should be allocated")
                    .overwrite_slot(slot, state);
                self.kv_cache
                    .as_mut()
                    .expect("kv cache should be allocated")
                    .slot_mut(slot)
                    .set_used(hit.prefix_tokens)
                    .map_err(ExecutorError::from)?;
            }
            if let Some(logits_handle) = hit.logits_handle {
                // Only sampling for the first token; skip forward compute.
                logits_handle
            } else {
                // No cached logits; fall back to computing the final token of the prefix.
                let mut tokens_slice: &[u32] =
                    &input_tokens[hit.prefix_tokens.min(input_tokens.len())..];
                if tokens_slice.is_empty() {
                    let last_idx = input_tokens.len().saturating_sub(1);
                    tokens_slice = &input_tokens[last_idx..];
                }
                self.forward_step(tokens_slice)?
            }
        } else {
            // Cold path: regular forward over the whole prompt.
            self.forward_step(&input_tokens)?
        };

        let mut generated = Vec::with_capacity(max_tokens);
        let eos_token = self.model_config.eos_token_id;
        for _ in 0..max_tokens {
            let next = self.sample_from_logits(&logits, temperature)?;
            if eos_token.is_some_and(|id| id == next) {
                break;
            }
            generated.push(next);
            logits = self.forward_step(&[next])?;
        }

        let text = self.decode_tokens(&generated)?;
        self.kv_cache_slot = self.kv_cache_slot.flip();

        // Write back prefix cache; store the active KV handle/logits for reuse.
        let kv_handle = self
            .kv_cache
            .as_ref()
            .map(|c| c.slot(self.kv_cache_slot.flip()).handle());
        self.scheduler
            .lmcache_put(&model_id, prompt, kv_handle, Some(logits));
        Ok(text)
    }

    pub fn embed(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let embedding = self.backend.embedding_forward_gpu_pure(
            &tokens,
            &AttentionTopology::linear(),
            &self.weights,
            &self.forward_config,
        )?;
        Ok(embedding)
    }

    pub fn rerank(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let scores = self.backend.rerank_forward_gpu_pure(
            &tokens,
            &AttentionTopology::linear(),
            &self.weights,
            &self.forward_config,
        )?;
        Ok(scores)
    }

    fn ensure_kv_cache(&mut self) -> ExecutorResult<&mut KvCacheDoubleBuffer> {
        let needs_alloc = self.kv_cache.as_ref().map_or(true, |existing| {
            existing.front().config() != self.kv_cache_config
                || existing.back().config() != self.kv_cache_config
        });
        if needs_alloc {
            let front = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let back = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let front = KvCacheState::new(front, self.kv_cache_config);
            let back = KvCacheState::new(back, self.kv_cache_config);
            self.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
            self.kv_cache_slot = KvCacheSlot::Front;
        }
        Ok(self.kv_cache.as_mut().expect("kv cache just allocated"))
    }

    fn active_kv_handle(&mut self) -> ExecutorResult<KvCacheHandle> {
        let slot = self.kv_cache_slot;
        let cache = self.ensure_kv_cache()?;
        Ok(cache.slot(slot).handle())
    }

    /// 异步友好的 swap-out；目前内部仍为同步调用，提供 async 接口方便集成。
    pub async fn swap_out_pages_async(&mut self, page_indices: &[usize]) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_out_pages(&mut handle, page_indices)?;
        Ok(())
    }

    /// 异步友好的 swap-in，完成后通知调度器进入 Warm 保护。
    pub async fn swap_in_pages_async(
        &mut self,
        request_id: RequestId,
        page_indices: &[PageId],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_in_pages(&mut handle, page_indices)?;
        self.scheduler.on_swap_in(request_id, page_indices);
        Ok(())
    }

    /// 从 backend 同步页面状态到调度器（集成 get_page_states）。
    pub fn refresh_page_states(&mut self) -> ExecutorResult<()> {
        if self.kv_cache.is_some() {
            let handle = self.active_kv_handle()?;
            let states = self.backend.get_page_states(&handle)?;
            self.scheduler.sync_page_states(&states);
        }
        Ok(())
    }
}
