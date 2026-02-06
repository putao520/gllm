//! Executor skeleton.

use gllm_kernels::backend_trait::{
    AttentionTopology, Backend, BackendError, KvCacheHandle, LogitsHandle,
};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::{
    GeneratorForwardConfig, KvCacheConfig, PageId, PositionEncoding, RequestId, SamplingConfig,
    StorageKey, SwapConfig,
};
use thiserror::Error;

use crate::adapter::{AdapterError, AdapterWeights, Message, ModelAdapter};
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use crate::loader::Loader;
use crate::manifest::ModelManifest;
use crate::model_config::{ModelConfig, ModelConfigError};
use crate::tokenizer::{TokenizerError, TokenizerHandle};
use std::sync::Arc;

use crate::scheduler::batcher::{BatchResult, ContinuousBatcher};
use crate::scheduler::hgal::HGALConfig;
use crate::scheduler::types::RequestKind;
use crate::scheduler::vllm2024::Scheduler2024Config;
use crate::scheduler::{PagedScheduler, ScheduledBatch, Sequence};
use std::collections::HashMap;

#[derive(Debug)]
struct RequestData {
    prompt_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    sampling_config: SamplingConfig,
    is_prefill: bool,
    // kv_cache: KvCacheHandle, // Moved to Scheduler/BlockTable management
    max_new_tokens: usize,
    finished: bool,
}

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
    #[error("scheduler error: {0}")]
    Scheduler(String),
    #[error("empty prompt tokens")]
    EmptyPrompt,
    #[error("backend returned empty sample")]
    EmptySample,
}

pub type ExecutorResult<T> = std::result::Result<T, ExecutorError>;

pub struct Executor<B: Backend + 'static> {
    backend: B,
    scheduler: PagedScheduler,
    batcher: ContinuousBatcher,
    requests: HashMap<RequestId, RequestData>,
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
        loader.set_manifest_if_missing(manifest.as_ref());
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

        // TODO: Get total_blocks from backend or config.
        // For now, assume 1GB KV cache with block_size 16 and head_dim 128, float32.
        // 1 block = 16 * num_heads * head_dim * dtype_size
        // This is backend dependent. We use a safe default of 10240 blocks.
        let total_blocks = 10240;
        let block_size = 16;
        let hgal_config = HGALConfig::default();

        let mut scheduler = PagedScheduler::new(total_blocks, block_size, hgal_config);
        scheduler.enable_vllm_2024(Scheduler2024Config {
            enable_2024_optimizations: true,
            ..Scheduler2024Config::default()
        });
        let page_size = scheduler.page_size();

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
            swap_config: Some(SwapConfig::default()),
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
            batcher: ContinuousBatcher::new(),
            requests: HashMap::new(),
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

    pub fn enqueue(&mut self, _kind: RequestKind, prompt: impl Into<String>) -> RequestId {
        let id = self.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str).unwrap_or_default();
        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config: SamplingConfig::default(),
            is_prefill: true,
            max_new_tokens: 128, // Default
            finished: false,
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        id
    }

    pub fn enqueue_with_config(
        &mut self,
        _kind: RequestKind,
        prompt: impl Into<String>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
    ) -> ExecutorResult<RequestId> {
        let id = self.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str)?;
        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config,
            is_prefill: true,
            max_new_tokens,
            finished: false,
        };

        self.requests.insert(id, request_data);
        self.batcher.enqueue(sequence);
        Ok(id)
    }

    // Deprecated/Modified: enqueue_with_tokens is used by tests, so we adapt it.
    pub fn enqueue_with_tokens(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        tokens: usize,
    ) -> RequestId {
        // Adapt for tests that manually specify token count but don't care about actual encoding
        let _ = self.enqueue_with_config(
            kind,
            prompt,
            tokens, // Interpret tokens as max_new_tokens for tests
            SamplingConfig::default(),
        );
        self.requests.len() as RequestId
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        if !self.batcher.has_pending_work() {
            return None;
        }
        Some(self.batcher.build_batch(&mut self.scheduler))
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

    /// Main Engine Step: Continuous Batching
    ///
    /// 1. Schedule next batch (mix of prefill and decode)
    /// 2. Construct batched inputs (strict causal ordering)
    /// 3. Run backend forward
    /// 4. Sample and update request states
    pub fn step(&mut self) -> ExecutorResult<()> {
        self.check_memory_pressure()?;

        // 1. Schedule
        let batch = match self.next_batch() {
            Some(b) => b,
            None => return Ok(()),
        };

        if batch.requests.is_empty() {
            return Ok(());
        }

        let mut batch_results = Vec::with_capacity(batch.requests.len());

        // 2. Process Batch (Serial Execution for Correctness / Accuracy First)
        // We deliberately process requests serially to avoid "ragged batch" non-determinism.
        // This aligns with our "Accuracy First" philosophy (2026 Standard).
        for req_id in batch.requests {
            self.process_pending_swap_in(req_id)?;

            // Extraction Scope
            let (tokens, sampling_config, max_new_tokens) = {
                let req = self.requests.get(&req_id).expect("request exists");
                if req.finished {
                    continue;
                }
                let tokens = if req.is_prefill {
                    req.prompt_tokens.clone()
                } else {
                    req.output_tokens
                        .last()
                        .map(|t| vec![*t])
                        .unwrap_or_default()
                };
                (tokens, req.sampling_config.clone(), req.max_new_tokens)
            };

            if tokens.is_empty() {
                if let Some(req) = self.requests.get_mut(&req_id) {
                    req.finished = true;
                }
                batch_results.push(BatchResult::fail(req_id));
                continue;
            }

            // Execution Scope (mut borrow of self)
            let logits = self.forward_step(&tokens)?;
            let next_token = self.sample_from_logits(&logits, sampling_config.temperature)?;

            // Update Scope (mut borrow of requests)
            let eos_token = self.model_config.eos_token_id;
            let mut request_finished = false;

            {
                let req = self.requests.get_mut(&req_id).expect("request exists");
                req.output_tokens.push(next_token);
                req.is_prefill = false;

                if eos_token.is_some_and(|id| id == next_token)
                    || req.output_tokens.len() >= max_new_tokens
                {
                    req.finished = true;
                    request_finished = true;
                }
            }

            if request_finished {
                batch_results.push(BatchResult::complete(req_id, Some(next_token)));
            } else {
                batch_results.push(BatchResult::continue_with_token(req_id, next_token));
            }
        }
        self.batcher
            .update_batch(&mut self.scheduler, batch_results.as_slice());
        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        let sampling_config = SamplingConfig {
            temperature,
            ..SamplingConfig::default()
        };

        // 1. Enqueue (Non-blocking)
        let req_id =
            self.enqueue_with_config(RequestKind::Chat, prompt, max_tokens, sampling_config)?;

        // 2. Drive the Engine (Blocking for this request)
        // In a real server, this loop would be global.
        // Here we run the global loop until OUR request is done.
        loop {
            self.step()?;

            if let Some(req) = self.requests.get(&req_id) {
                if req.finished {
                    break;
                }
            } else {
                // Request removed (e.g. error or finished and cleaned up?)
                // If finish_request removes it from batcher, it might still be in self.requests?
                // We typically keep data in `requests` until retrieved.
                break;
            }
        }

        // 3. Retrieve Result
        let req = self.requests.get(&req_id).expect("request data missing");
        let text = self.decode_tokens(&req.output_tokens)?;

        // Cleanup?
        // self.requests.remove(&req_id);

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
    pub async fn swap_out_pages_async(
        &mut self,
        page_mappings: &[(PageId, StorageKey)],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_out_pages(&mut handle, page_mappings)?;
        Ok(())
    }

    /// 异步友好的 swap-in，完成后通知调度器进入 Warm 保护。
    pub async fn swap_in_pages_async(
        &mut self,
        request_id: RequestId,
        page_mappings: &[(PageId, StorageKey)],
    ) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_in_pages(&mut handle, page_mappings)?;
        let page_indices: Vec<PageId> = page_mappings
            .iter()
            .map(|(physical_id, _)| *physical_id)
            .collect();
        self.scheduler.on_swap_in(request_id, &page_indices);
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

    fn check_memory_pressure(&mut self) -> ExecutorResult<()> {
        let Some(swap_cfg) = self.kv_cache_config.swap_config else {
            return Ok(());
        };
        if !swap_cfg.enable_swap || self.kv_cache.is_none() {
            return Ok(());
        }

        let threshold = swap_cfg.swap_threshold.max(0.0).min(1.0);
        let needed_blocks = swap_cfg.lru_granularity.max(1);
        let mut pressure = self.backend.get_memory_pressure()?;
        if pressure <= threshold {
            return Ok(());
        }

        let mut kv_handle = self.active_kv_handle()?;
        while pressure > threshold {
            let victims = self.scheduler.select_victims(needed_blocks);
            if victims.is_empty() {
                break;
            }

            let mut victim_ids = Vec::with_capacity(victims.len());
            let mut swap_out_mappings = Vec::new();
            for (request_id, pages) in victims {
                victim_ids.push(request_id);
                for (logical_idx, physical_id) in pages.into_iter().enumerate() {
                    let storage_key = PagedScheduler::storage_key(request_id, logical_idx)
                        .map_err(ExecutorError::Scheduler)?;
                    swap_out_mappings.push((physical_id, storage_key));
                }
            }

            if swap_out_mappings.is_empty() {
                break;
            }

            self.backend
                .swap_out_pages(&mut kv_handle, &swap_out_mappings)?;
            self.scheduler
                .free_victims(&victim_ids)
                .map_err(ExecutorError::Scheduler)?;
            pressure = self.backend.get_memory_pressure()?;
        }

        Ok(())
    }

    fn process_pending_swap_in(&mut self, request_id: RequestId) -> ExecutorResult<()> {
        let Some(mappings) = self.scheduler.take_pending_swap_in(request_id) else {
            return Ok(());
        };
        if mappings.is_empty() {
            return Ok(());
        }

        let mut kv_handle = self.active_kv_handle()?;
        self.backend.swap_in_pages(&mut kv_handle, &mappings)?;
        let page_indices: Vec<PageId> = mappings
            .iter()
            .map(|(physical_id, _)| *physical_id)
            .collect();
        self.scheduler.on_swap_in(request_id, &page_indices);
        Ok(())
    }
}
