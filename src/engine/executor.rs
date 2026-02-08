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

use crate::adapter::{AdapterError, AdapterWeights, ModelAdapter};
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use crate::loader::onnx::FusedKernel;
use crate::loader::{Loader, LoaderError, WeightFormat};
use crate::manifest::{ModelKind, ModelManifest};
use crate::model_config::{ModelConfig, ModelConfigError};
use crate::tokenizer::{TokenizerError, TokenizerHandle};
use std::sync::Arc;

use crate::scheduler::batcher::{BatchAction, BatchResult, ContinuousBatcher};
use crate::scheduler::hgal::HGALConfig;
use crate::scheduler::types::RequestKind;
use crate::scheduler::vllm2024::Scheduler2024Config;
use crate::scheduler::{
    BasicObserver, GlobalMemoryManager, MemoryManagerError, PagedScheduler, PolicyVariant,
    ScheduledBatch, Sequence, SystemState, Tier, VirtualPageId,
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

#[derive(Debug, Clone, Default)]
struct OnnxFusedKernelStats {
    flash_attention: usize,
    swiglu: usize,
    rope: usize,
    fused_qkv_rope: usize,
    atomic: usize,
}

impl OnnxFusedKernelStats {
    fn fused_total(&self) -> usize {
        self.flash_attention + self.swiglu + self.rope + self.fused_qkv_rope
    }

    fn from_fused_graph(fused_graph: &crate::loader::onnx::FusedGraph) -> Self {
        let mut stats = Self::default();
        for op in &fused_graph.ops {
            match &op.kind {
                FusedKernel::FlashAttention(_) => stats.flash_attention += 1,
                FusedKernel::SwiGlu(_) => stats.swiglu += 1,
                FusedKernel::Rope(_) => stats.rope += 1,
                FusedKernel::FusedQkvRope(_) => stats.fused_qkv_rope += 1,
                FusedKernel::Atomic(_) => stats.atomic += 1,
            }
        }
        stats
    }
}

#[derive(Debug, Clone)]
struct OnnxGeneratorPlan {
    fused_kernels: OnnxFusedKernelStats,
    graph_outputs: Vec<String>,
    kv_outputs: Vec<String>,
    execution_order: Vec<OnnxKernelExecutionOp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnnxKernelExecutionOp {
    FlashAttention,
    SwiGlu,
    Rope,
    FusedQkvRope,
    Atomic,
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
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    KvCache(#[from] KvCacheError),
    #[error(transparent)]
    MemoryManager(#[from] MemoryManagerError),
    #[error("scheduler error: {0}")]
    Scheduler(String),
    #[error("empty prompt tokens")]
    EmptyPrompt,
    #[error("backend returned empty sample")]
    EmptySample,
    #[error("request not found: {request_id}")]
    RequestNotFound { request_id: RequestId },
    #[error("onnx generator plan error: {0}")]
    OnnxPlan(String),
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
    memory_manager: GlobalMemoryManager,
    onnx_generator_plan: Option<OnnxGeneratorPlan>,
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
        loader.set_tie_word_embeddings_hint(model_config.tie_word_embeddings);
        if manifest.kind == ModelKind::Chat && model_config.use_cache == Some(false) {
            return Err(ExecutorError::Config(ModelConfigError::InvalidConfig(
                "config.use_cache=false is not supported for generator models in current gllm executor"
                    .to_string(),
            )));
        }
        let position_encoding = match manifest.kind {
            // Encoder-style embedding/reranker models (e.g. XLM-R/BERT) usually do not expose RoPE.
            // When rope_theta is absent/invalid, skip positional rotation instead of forcing RoPE.
            ModelKind::Embedding | ModelKind::Reranker if model_config.rope_theta <= 0.0 => {
                PositionEncoding::None
            }
            _ => PositionEncoding::Rope,
        };
        let forward_config = GeneratorForwardConfig {
            num_layers: model_config.num_hidden_layers,
            num_heads: model_config.num_attention_heads,
            num_kv_heads: model_config.num_key_value_heads,
            head_dim: model_config.head_dim,
            max_seq_len: model_config.max_position_embeddings,
            vocab_size: model_config.vocab_size,
            rope_theta: model_config.rope_theta,
            rope_scale: model_config.rope_scale,
            rope_interleaved: model_config.rope_interleaved,
            rope_precompute: true,
            position_encoding,
        };

        // DEBUG: 打印配置信息
        eprintln!("=== GeneratorForwardConfig ===");
        eprintln!("num_layers: {}", forward_config.num_layers);
        eprintln!("num_heads: {}", forward_config.num_heads);
        eprintln!("num_kv_heads: {}", forward_config.num_kv_heads);
        eprintln!("head_dim: {}", forward_config.head_dim);
        eprintln!("hidden_size (calc): {} * {} = {}", forward_config.num_heads, forward_config.head_dim, forward_config.num_heads * forward_config.head_dim);
        eprintln!("max_seq_len: {}", forward_config.max_seq_len);
        eprintln!("vocab_size: {}", forward_config.vocab_size);

        let block_size = model_config.kv_cache_block_size;
        let hgal_config = HGALConfig::default();
        let total_blocks = model_config.max_position_embeddings.div_ceil(block_size);

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
        let onnx_generator_plan = Self::build_onnx_generator_plan(manifest.as_ref(), loader)?;
        let tokenizer = TokenizerHandle::from_loader(loader)?;
        let weights = adapter.load_weights(loader, &backend)?;
        let l1_capacity = total_blocks;
        let l2_capacity = total_blocks.saturating_mul(10);
        let l3_capacity = total_blocks.saturating_mul(100);
        let memory_manager =
            GlobalMemoryManager::new_with_capacities(l1_capacity, l2_capacity, l3_capacity);
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
            memory_manager,
            onnx_generator_plan,
            batcher: ContinuousBatcher::new(),
            observer: BasicObserver::new(),
            policy: PolicyVariant::default(),
            requests: HashMap::new(),
        })
    }

    fn build_onnx_generator_plan(
        manifest: &ModelManifest,
        loader: &mut Loader,
    ) -> ExecutorResult<Option<OnnxGeneratorPlan>> {
        if manifest.kind != ModelKind::Chat || loader.weight_format() != WeightFormat::Onnx {
            return Ok(None);
        }

        let onnx = loader.onnx()?;
        let fused_graph = onnx.fused_graph();
        let fused_kernels = OnnxFusedKernelStats::from_fused_graph(fused_graph);
        let execution_order = fused_graph
            .ops
            .iter()
            .map(|op| match &op.kind {
                FusedKernel::FlashAttention(_) => OnnxKernelExecutionOp::FlashAttention,
                FusedKernel::SwiGlu(_) => OnnxKernelExecutionOp::SwiGlu,
                FusedKernel::Rope(_) => OnnxKernelExecutionOp::Rope,
                FusedKernel::FusedQkvRope(_) => OnnxKernelExecutionOp::FusedQkvRope,
                FusedKernel::Atomic(_) => OnnxKernelExecutionOp::Atomic,
            })
            .collect::<Vec<_>>();
        if execution_order.is_empty() {
            return Err(ExecutorError::OnnxPlan(
                "fused matcher produced an empty ONNX execution plan".to_string(),
            ));
        }

        let graph_outputs = onnx
            .graph()
            .outputs
            .iter()
            .map(|value| value.name.clone())
            .collect::<Vec<_>>();
        let kv_outputs = extract_onnx_kv_outputs(&graph_outputs);
        if kv_outputs.is_empty() {
            return Err(ExecutorError::OnnxPlan(
                "ONNX graph does not expose identifiable KV cache outputs".to_string(),
            ));
        }

        Ok(Some(OnnxGeneratorPlan {
            fused_kernels,
            graph_outputs,
            kv_outputs,
            execution_order,
        }))
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

    pub fn allocate_kv_cache(&mut self, config: &KvCacheConfig) -> ExecutorResult<KvCacheHandle> {
        let front = self.backend.alloc_kv_cache(config)?;
        let back = self.backend.alloc_kv_cache(config)?;
        let front = KvCacheState::new(front, config.clone());
        let back = KvCacheState::new(back, config.clone());
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
        sampling: &SamplingConfig,
    ) -> ExecutorResult<u32> {
        let tokens = self.backend.sample_from_tensor(
            logits,
            &AttentionTopology::linear(),
            self.model_config.vocab_size,
            sampling,
        )?;
        tokens.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    fn run_batch_forward(&mut self, batch_input: &BatchInput) -> ExecutorResult<Vec<LogitsHandle>> {
        if let Some(plan) = self.onnx_generator_plan.as_ref() {
            if plan.execution_order.is_empty() {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan is empty".to_string(),
                ));
            }
            let total_kernels = plan.fused_kernels.fused_total() + plan.fused_kernels.atomic;
            if total_kernels != plan.execution_order.len() {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan kernel accounting mismatch".to_string(),
                ));
            }
            if plan.kv_outputs.is_empty() {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan has no KV cache outputs".to_string(),
                ));
            }
            let has_logits_output = plan
                .graph_outputs
                .iter()
                .any(|name| is_onnx_logits_output(name))
                || plan
                    .graph_outputs
                    .len()
                    .saturating_sub(plan.kv_outputs.len())
                    == 1;
            if !has_logits_output {
                return Err(ExecutorError::OnnxPlan(
                    "ONNX execution plan has no logits output".to_string(),
                ));
            }
        }

        let kv_handle = self.active_kv_handle()?;
        let mut kv_caches = vec![kv_handle; batch_input.sequences.len()];
        Ok(self.backend.batch_forward_gpu_pure(
            batch_input,
            &AttentionTopology::linear(),
            &self.weights,
            &mut kv_caches,
            &self.forward_config,
        )?)
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
            memory_pressure: self.backend.get_memory_pressure().unwrap_or(0.0), // Best effort
            kv_fragmentation: self.scheduler.kv_fragmentation_ratio(),
            waiting_queue_len: self.batcher.waiting_len(),
            current_running_len: self.batcher.running_len(),
            mean_context_len: self.batcher.mean_context_len(),
            logits_entropy: 0.0, // Phase 2
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
            self.ensure_pages_resident(req_id)?;

            let (tokens, position) = {
                let Some(req) = self.requests.get(&req_id) else {
                    // Request was removed, skip it
                    continue;
                };
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

            sequences.push(SequenceInput { tokens, position });
            request_indices.push(req_id);
        }

        if sequences.is_empty() {
            return Ok(());
        }

        let batch_input = BatchInput { sequences };

        // 4. Run Backend Forward
        let logits_list = self.run_batch_forward(&batch_input)?;

        // 5. Process Results
        // Note: batch_forward_gpu_pure must return results in the same order as input sequences
        if logits_list.len() != request_indices.len() {
            return Err(ExecutorError::Backend(BackendError::Cuda(
                "Backend returned mismatched number of logits".to_string(),
            )));
        }

        let mut total_tokens = 0;
        for seq in &batch_input.sequences {
            total_tokens += seq.tokens.len();
        }

        // Processing results loop
        for (i, logits) in logits_list.iter().enumerate() {
            let req_id = request_indices[i];
            let sampling_config = self
                .requests
                .get(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?
                .sampling_config;
            let next_token = self.sample_from_logits(logits, &sampling_config)?;

            // Update request
            let req = self
                .requests
                .get_mut(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
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

        for request_id in batch_results
            .iter()
            .filter(|result| matches!(result.action, BatchAction::Complete | BatchAction::Fail))
            .map(|result| result.request_id)
        {
            self.release_request_pages(request_id);
        }

        self.batcher
            .update_batch(&mut self.scheduler, batch_results.as_slice());
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

        logits_list
            .into_iter()
            .next()
            .ok_or(ExecutorError::EmptySample)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        self.generate_with_sampling(prompt, max_tokens, temperature, 0, 1.0)
    }

    pub fn generate_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let sampling_config = SamplingConfig {
            temperature,
            top_k,
            top_p,
        };

        let req_id =
            self.enqueue_with_config(RequestKind::Chat, prompt, max_tokens, sampling_config)?;

        loop {
            self.step()?;

            if let Some(req) = self.requests.get(&req_id) {
                if req.finished {
                    break;
                }
            } else {
                break;
            }
        }

        let req = self
            .requests
            .get(&req_id)
            .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
        let text = self.decode_tokens(&req.output_tokens)?;
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

    pub fn is_finished(&self, request_id: RequestId) -> bool {
        self.requests
            .get(&request_id)
            .map(|r| r.finished)
            .unwrap_or(false) // If removed, considered finished? Or unknown.
    }

    pub fn get_output(&self, request_id: RequestId) -> ExecutorResult<String> {
        let req = self
            .requests
            .get(&request_id)
            .ok_or(ExecutorError::Scheduler("Request not found".into()))?;
        self.decode_tokens(&req.output_tokens)
    }

    fn ensure_kv_cache(&mut self) -> ExecutorResult<&mut KvCacheDoubleBuffer> {
        let needs_alloc = self.kv_cache.as_ref().is_none_or(|existing| {
            existing.front().config() != self.kv_cache_config
                || existing.back().config() != self.kv_cache_config
        });
        if needs_alloc {
            let front = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let back = self.backend.alloc_kv_cache(&self.kv_cache_config)?;
            let front = KvCacheState::new(front, self.kv_cache_config.clone());
            let back = KvCacheState::new(back, self.kv_cache_config.clone());
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

    fn storage_key_to_page_id(storage_key: StorageKey) -> ExecutorResult<PageId> {
        usize::try_from(storage_key).map_err(|_| {
            ExecutorError::Scheduler("storage key does not fit into page id".to_string())
        })
    }

    fn ensure_l1_page_tracked(&mut self, physical_id: PageId) -> ExecutorResult<()> {
        match self.memory_manager.track_page(Tier::L1, physical_id) {
            Ok(()) => Ok(()),
            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 }) => {
                self.reclaim_memory(1)?;
                self.memory_manager.track_page(Tier::L1, physical_id)?;
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    fn release_request_pages(&mut self, request_id: RequestId) {
        for (logical_idx, page_id) in self.scheduler.request_pages(request_id) {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            if let Some(location) = self.memory_manager.unmap_virtual_page(virtual_id) {
                let _ = self
                    .memory_manager
                    .untrack_page(location.tier, location.physical_id);
            } else {
                let _ = self.memory_manager.untrack_page(Tier::L1, page_id);
            }
        }
    }

    fn reclaim_memory(&mut self, required_pages: usize) -> ExecutorResult<()> {
        if required_pages == 0 {
            return Ok(());
        }

        let mut kv_handle = self.active_kv_handle()?;
        loop {
            let usage = self.memory_manager.tier_usage(Tier::L1);
            let free_pages = usage.capacity.saturating_sub(usage.used);
            if free_pages >= required_pages {
                return Ok(());
            }

            let need = required_pages.saturating_sub(free_pages).max(1);
            let victims = self.scheduler.select_victims(need);
            if victims.is_empty() {
                return Ok(());
            }

            let mut victim_ids = Vec::with_capacity(victims.len());
            let mut swap_out_mappings = Vec::new();
            let mut planned_remaps = Vec::new();

            for (request_id, pages) in &victims {
                if pages.is_empty() {
                    continue;
                }
                victim_ids.push(*request_id);
                for (logical_idx, &l1_page_id) in pages.iter().enumerate() {
                    let storage_key = PagedScheduler::storage_key(*request_id, logical_idx)
                        .map_err(|err| ExecutorError::Scheduler(err.to_string()))?;
                    let target_page = Self::storage_key_to_page_id(storage_key)?;
                    let (target_tier, target_page) =
                        match self.memory_manager.track_page(Tier::L2, target_page) {
                            Ok(()) => (Tier::L2, target_page),
                            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L2 }) => {
                                self.memory_manager.track_page(Tier::L3, target_page)?;
                                (Tier::L3, target_page)
                            }
                            Err(err) => return Err(err.into()),
                        };
                    let virtual_id = VirtualPageId::new(*request_id, logical_idx);
                    swap_out_mappings.push((l1_page_id, storage_key));
                    planned_remaps.push((virtual_id, l1_page_id, target_tier, target_page));
                }
            }

            if swap_out_mappings.is_empty() {
                return Ok(());
            }

            self.backend
                .swap_out_pages(&mut kv_handle, &swap_out_mappings)?;

            for (virtual_id, l1_page_id, target_tier, target_page) in planned_remaps {
                if let Some(old_location) = self.memory_manager.unmap_virtual_page(virtual_id) {
                    let _ = self
                        .memory_manager
                        .untrack_page(old_location.tier, old_location.physical_id);
                } else {
                    let _ = self.memory_manager.untrack_page(Tier::L1, l1_page_id);
                }
                self.memory_manager
                    .bind_virtual_page(virtual_id, target_tier, target_page)?;
            }

            for (request_id, pages) in &victims {
                self.scheduler.on_page_evicted(*request_id, pages);
            }
            self.scheduler
                .free_victims(&victim_ids)
                .map_err(|err| ExecutorError::Scheduler(err.to_string()))?;
        }
    }

    fn ensure_pages_resident(&mut self, request_id: RequestId) -> ExecutorResult<()> {
        if let Some(mappings) = self.scheduler.take_pending_swap_in(request_id) {
            if !mappings.is_empty() {
                let mut kv_handle = self.active_kv_handle()?;
                self.backend.swap_in_pages(&mut kv_handle, &mappings)?;
                let page_indices: Vec<PageId> = mappings
                    .iter()
                    .map(|(physical_id, _)| *physical_id)
                    .collect();

                for (logical_idx, (physical_id, storage_key)) in mappings.into_iter().enumerate() {
                    let virtual_id = VirtualPageId::new(request_id, logical_idx);
                    self.ensure_l1_page_tracked(physical_id)?;

                    let old_location = self.memory_manager.resolve(virtual_id).ok();
                    if old_location.is_some() {
                        self.memory_manager.remap_virtual_page(
                            virtual_id,
                            Tier::L1,
                            physical_id,
                        )?;
                    } else {
                        self.memory_manager
                            .bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    }

                    if let Some((tier, page)) = old_location {
                        if tier != Tier::L1 {
                            let _ = self.memory_manager.untrack_page(tier, page);
                        }
                    } else {
                        let offload_page = Self::storage_key_to_page_id(storage_key)?;
                        let _ = self.memory_manager.untrack_page(Tier::L2, offload_page);
                        let _ = self.memory_manager.untrack_page(Tier::L3, offload_page);
                    }
                }

                self.scheduler.on_swap_in(request_id, &page_indices);
            }
        }

        let request_pages = self.scheduler.request_pages(request_id);
        if request_pages.is_empty() {
            return Ok(());
        }

        let mut swap_in_mappings = Vec::new();
        let mut swapped_pages = Vec::new();
        let mut remap_plan = Vec::new();

        for (logical_idx, physical_id) in request_pages {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            match self.memory_manager.resolve(virtual_id) {
                Ok((Tier::L1, mapped)) if mapped == physical_id => {}
                Ok((Tier::L1, mapped)) => {
                    self.ensure_l1_page_tracked(physical_id)?;
                    self.memory_manager
                        .remap_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    let _ = self.memory_manager.untrack_page(Tier::L1, mapped);
                }
                Ok((tier @ Tier::L2, offload_id)) | Ok((tier @ Tier::L3, offload_id)) => {
                    self.reclaim_memory(1)?;
                    self.ensure_l1_page_tracked(physical_id)?;
                    let storage_key: StorageKey = offload_id as StorageKey;
                    swap_in_mappings.push((physical_id, storage_key));
                    swapped_pages.push(physical_id);
                    remap_plan.push((virtual_id, tier, offload_id, physical_id));
                }
                Err(MemoryManagerError::UnknownVirtualPage { .. }) => {
                    self.ensure_l1_page_tracked(physical_id)?;
                    self.memory_manager
                        .bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
                }
                Err(err) => return Err(err.into()),
            }
        }

        if !swap_in_mappings.is_empty() {
            let mut kv_handle = self.active_kv_handle()?;
            self.backend
                .swap_in_pages(&mut kv_handle, &swap_in_mappings)?;

            for (virtual_id, old_tier, old_physical_id, new_physical_id) in remap_plan {
                self.memory_manager
                    .remap_virtual_page(virtual_id, Tier::L1, new_physical_id)?;
                let _ = self.memory_manager.untrack_page(old_tier, old_physical_id);
            }
            self.scheduler.on_swap_in(request_id, &swapped_pages);
        }

        Ok(())
    }

    fn check_memory_pressure(&mut self) -> ExecutorResult<()> {
        let Some(ref swap_cfg) = self.kv_cache_config.swap_config else {
            return Ok(());
        };
        if !swap_cfg.enable_swap || self.kv_cache.is_none() {
            return Ok(());
        }

        let threshold = swap_cfg.swap_threshold.clamp(0.0, 1.0);
        let needed_blocks = swap_cfg.lru_granularity.max(1);
        let mut pressure = self.backend.get_memory_pressure()?;
        if pressure <= threshold {
            return Ok(());
        }

        while pressure > threshold {
            self.reclaim_memory(needed_blocks)?;
            let next_pressure = self.backend.get_memory_pressure()?;
            if next_pressure >= pressure {
                break;
            }
            pressure = next_pressure;
        }

        Ok(())
    }
}

fn extract_onnx_kv_outputs(graph_outputs: &[String]) -> Vec<String> {
    graph_outputs
        .iter()
        .filter(|name| is_onnx_kv_output(name))
        .cloned()
        .collect()
}

fn is_onnx_kv_output(name: &str) -> bool {
    let normalized = name
        .trim()
        .to_ascii_lowercase()
        .replace(['/', '.', '-', ' '], "_");
    let has_cache_hint = normalized.contains("past")
        || normalized.contains("present")
        || normalized.contains("cache")
        || normalized.contains("key_values")
        || normalized.contains("kv");
    let has_kv_axis = normalized.contains("key") || normalized.contains("value");
    has_cache_hint && has_kv_axis
}

fn is_onnx_logits_output(name: &str) -> bool {
    let normalized = name
        .trim()
        .to_ascii_lowercase()
        .replace(['/', '.', '-', ' '], "_");
    normalized.contains("logits")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_kv_outputs_from_onnx_names() {
        let outputs = vec![
            "logits".to_string(),
            "present.0.key".to_string(),
            "present.0.value".to_string(),
            "present.1.key".to_string(),
            "present.1.value".to_string(),
        ];
        let kv_outputs = extract_onnx_kv_outputs(&outputs);
        assert_eq!(kv_outputs.len(), 4);
        assert!(kv_outputs.iter().all(|name| name.contains("present")));
    }

    #[test]
    fn detects_logits_output_name() {
        assert!(is_onnx_logits_output("logits"));
        assert!(is_onnx_logits_output("model/logits_output"));
        assert!(!is_onnx_logits_output("present.0.key"));
    }
}
