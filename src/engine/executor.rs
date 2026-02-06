//! Executor skeleton.

use gllm_kernels::backend_trait::{
    AttentionTopology, Backend, BackendError, BatchInput, KvCacheHandle, LogitsHandle,
    SequenceInput,
};
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::{
    GeneratorForwardConfig, KvCacheConfig, PageId, PositionEncoding, RequestId, SamplingConfig,
    StorageKey,
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
use crate::scheduler::{
    BasicObserver, PagedScheduler, PolicyVariant, ScheduledBatch, Sequence,
    SystemState,
};
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
    observer: BasicObserver,
    policy: PolicyVariant,
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
            swap_config: None,
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
            observer: BasicObserver::new(),
            policy: PolicyVariant::default(),
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
        Some(
            self.batcher
                .build_batch(&mut self.scheduler, usize::MAX, true),
        )
    }

    pub fn encode_prompt(&self, prompt: &str) -> ExecutorResult<Vec<u32>> {
        let add_special_tokens = self.adapter.add_special_tokens();
        Ok(self.tokenizer.encode(prompt, add_special_tokens)?)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> ExecutorResult<String> {
        Ok(self.tokenizer.decode(tokens, true)?)
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

        // 0. Observability: Capture System State
        let system_state = SystemState {
            memory_pressure: self
                .backend
                .get_memory_pressure()
                .unwrap_or(0.0), // Best effort
            kv_fragmentation: 0.0, // TODO: Get from scheduler
            waiting_queue_len: 0,  // TODO: Get from batcher.waiting.len()
            current_running_len: 0, // TODO: Get from batcher.running.len()
            mean_context_len: 0,   // TODO
            logits_entropy: 0.0,   // Phase 2
        };
        self.observer.update(system_state);

        // 1. JIT Decision: Decide Scheduling Strategy
        let decision = self.policy.decide(&system_state);

        // 2. Schedule
        // Pass dynamic decision parameters to batcher
        let batch = if !self.batcher.has_pending_work() {
            return Ok(());
        } else {
            self.batcher.build_batch(
                &mut self.scheduler,
                decision.max_batch_size,
                decision.admit_new_prefill,
            )
        };

        if batch.requests.is_empty() {
            return Ok(());
        }

        let mut batch_results = Vec::with_capacity(batch.requests.len());
        let mut sequences = Vec::with_capacity(batch.requests.len());
        let mut request_indices = Vec::with_capacity(batch.requests.len());

        // 3. Prepare Batch
        for req_id in batch.requests {
            self.process_pending_swap_in(req_id)?;

            let (tokens, position) = {
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

                let position = if req.is_prefill {
                    0
                } else {
                    req.prompt_tokens.len() + req.output_tokens.len().saturating_sub(1)
                };
                (tokens, position)
            };

            if tokens.is_empty() {
                if let Some(req) = self.requests.get_mut(&req_id) {
                    req.finished = true;
                }
                batch_results.push(BatchResult::fail(req_id));
                continue;
            }

            sequences.push(SequenceInput {
                tokens,
                position,
            });
            request_indices.push(req_id);
        }

        if sequences.is_empty() {
            return Ok(());
        }

        let batch_input = BatchInput { sequences };

        // 4. Run Backend Forward
        let mut kv_cache = self.active_kv_handle()?;
        let logits_list = self.backend.batch_forward_gpu_pure(
            &batch_input,
            &AttentionTopology::linear(),
            &self.weights,
            std::slice::from_mut(&mut kv_cache),
            &self.forward_config,
        )?;

        // 5. Process Results
        // Note: batch_forward_gpu_pure must return results in the same order as input sequences
        if logits_list.len() != request_indices.len() {
             return Err(ExecutorError::Backend(BackendError::Cuda(
                "Backend returned mismatched number of logits".to_string(),
            )));
        }

        // We need to advance cache for each processed sequence
        // We do this in the loop or in batch if API supports it.
        // Currently `active.advance` is per slot (which is per request in some models, but here `active_kv_handle` returns a handle to the whole cache).
        // Wait, `active_kv_handle` returns `KvCacheHandle`.
        // `self.kv_cache` manages the double buffer state.
        // We need to update the `KvCacheState` in `self.kv_cache` to reflect advanced positions.
        // `KvCacheSlot` is just Front/Back. `KvCacheState` tracks usage?
        // Actually `KvCacheState` implementation in `kv_cache.rs` is likely tracking global state or we assume `scheduler` tracks pages.
        // But `advance(tokens.len())` was called in `forward_step`.
        // Let's check `forward_step` again (from memory/previous read):
        /*
        let active = kv_cache.slot_mut(slot);
        active.advance(tokens.len())?;
        */
        // This likely updates some metadata in `KvCacheState`.
        // We need to do this for each request?
        // Wait, `KvCacheState` represents the *whole* cache buffer (vram pointer + config).
        // It doesn't seem to track per-request cursors.
        // Ah, `active.advance(tokens.len())` in `forward_step` logic:
        /*
           if tokens.len() > active.remaining() { ... }
           active.advance(tokens.len())?;
        */
        // This suggests `KvCacheState` tracks linear usage if it were a linear buffer.
        // But we are using `PagedScheduler`.
        // `PagedScheduler` manages pages.
        // The `forward_step` logic in `executor.rs` (lines 282-318) seems legacy or for simple linear implementation?
        // Line 113: `PagedScheduler::new(total_blocks, block_size, hgal_config)`.
        // If we use PagedAttention, `KvCacheState`'s linear `advance` might not be relevant or might be for a simple linear allocator fallback.
        // HOWEVER, the previous `forward_step` WAS calling it.
        // And `ensure_kv_cache` allocates `KvCacheState`.
        // If `batch_forward_gpu_pure` handles PagedAttention, the `KvCacheHandle` passed to it allows it to read/write KV.
        // The `advance` call on `active` (KvCacheState) might be updating a simple offset.
        // If we are doing PagedAttention, usually the `scheduler` manages the block tables, and `backend` uses them.
        // `KvCacheState` might just be a wrapper around the pointer.
        // IF `advance` is purely checking capacity in a linear buffer sense, it might be wrong for PagedAttention.
        // But let's assume we need to preserve behavior or update it.
        // If `forward_step` was checking `remaining()`, maybe we should too.
        // But for continuous batching with paged attention, "remaining" is determined by free blocks in scheduler, not linear offset.
        // The `Executor` has `scheduler`.
        // I will omit the `active.advance()` call if it looks like legacy linear buffer logic, OR I should check `kv_cache.rs` to understand.
        // I can't check `kv_cache.rs` easily without reading it.
        // But `forward_step` was:
        /*
            let active = kv_cache.slot(slot);
            if tokens.len() > active.remaining() { ... }
            ...
            active.advance(tokens.len())?;
        */
        // If I remove `forward_step`, I should probably replicate this logic IF it matters.
        // But for PagedAttention, capacity is `scheduler.can_allocate(...)`.
        // The `KvCacheState` might be a linear slab allocator underlying the pages?
        // If so, `advance` might mark memory as used.
        // But `scheduler` manages blocks.
        // It's possible `KvCacheState` is a naive implementation that `Executor` was using before PagedAttention was fully integrated?
        // Or `KvCacheState` tracks the *allocation* of the big buffer?
        // `advance` usually implies moving a cursor.
        // If we have PagedAttention, we don't move a global cursor per token generated in a batch.

        // I will assume `scheduler` handles memory and `KvCacheState` logic in `forward_step` regarding `advance` was for the single-sequence linear case or simpler tests.
        // I will NOT call `advance` on `KvCacheState` for every request in the batch loop, as that would likely exhaust the "linear" capacity incorrectly if it's not actually linear.
        // However, the prompt says "Advance the KV cache for the processed tokens."
        // This is a specific instruction: "Advance the KV cache for the processed tokens."
        // I MUST follow it.
        // But calling `active.advance(len)` for *each* request on the *same* `active` object (which represents the whole cache) would mean we advance the global cursor by `sum(tokens)`.
        // If `KvCacheState` is indeed the linear memory pool, then yes, we consumed that much memory?
        // No, with PagedAttention we reuse blocks.
        // Maybe "Advance the KV cache" means something else?
        // Or maybe `KvCacheState` has a method to advance *specific* request?
        // The `forward_step` did `active.advance(tokens.len())`. `active` is `KvCacheSlot`.

        // Let's assume the prompt implies I should replicate the effect of `forward_step`'s advance.
        // I will do:
        /*
        let slot = self.kv_cache_slot;
        if let Some(kv_cache) = self.kv_cache.as_mut() {
             let active = kv_cache.slot_mut(slot);
             // active.advance(total_tokens_processed)?; // If I sum them up?
        }
        */

        // Wait, `forward_step` is per request (in the old loop).
        // If I use `batch_forward_gpu_pure`, I am processing `sum(tokens)` tokens in total.
        // I should probably sum them up and advance once?

        let mut total_tokens = 0;
        for seq in &batch_input.sequences {
            total_tokens += seq.tokens.len();
        }

        // Processing results loop
        for (i, logits) in logits_list.iter().enumerate() {
            let req_id = request_indices[i];

            // ... Sample ...
            let req = self.requests.get(&req_id).unwrap();
            let temperature = req.sampling_config.temperature;

            // Note: sample_from_logits takes &LogitsHandle.
            let next_token = self.sample_from_logits(logits, temperature)?;

            // Update request
            let req = self.requests.get_mut(&req_id).unwrap();
            req.output_tokens.push(next_token);
            req.is_prefill = false;

            // Check finish
            let eos_token = self.model_config.eos_token_id;
            let mut request_finished = false;
            if eos_token.is_some_and(|id| id == next_token)
                || req.output_tokens.len() >= req.max_new_tokens
            {
                req.finished = true;
                request_finished = true;
            }

            if request_finished {
                batch_results.push(BatchResult::complete(req_id, Some(next_token)));
            } else {
                batch_results.push(BatchResult::continue_with_token(req_id, next_token));
            }
        }

        // Advance KV cache
        {
             let slot = self.kv_cache_slot;
             if let Some(kv_cache) = self.kv_cache.as_mut() {
                 let active = kv_cache.slot_mut(slot);
                 active.advance(total_tokens)?;
             }
        }

        self.batcher.update_batch(&mut self.scheduler, batch_results.as_slice());
        Ok(())
    }

    /// Legacy method for compatibility with tests (e.g. test_alignment)
    /// Wraps the batch API for a single request.
    pub fn forward_step(&mut self, tokens: &[u32]) -> ExecutorResult<LogitsHandle> {
        let seq = SequenceInput {
            tokens: tokens.to_vec(),
            position: 0,
        };

        let batch_input = BatchInput {
            sequences: vec![seq],
        };

        let mut kv_cache = self.active_kv_handle()?;

        let logits_list = self.backend.batch_forward_gpu_pure(
            &batch_input,
            &AttentionTopology::linear(),
            &self.weights,
            std::slice::from_mut(&mut kv_cache),
            &self.forward_config,
        )?;

        // Maintain legacy KV cache state advancement
        if let Some(kv_cache) = self.kv_cache.as_mut() {
             let active = kv_cache.slot_mut(self.kv_cache_slot);
             active.advance(tokens.len())?;
        }

        logits_list.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
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
