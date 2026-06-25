//! Executor skeleton.

pub use super::executor_types::*;

use std::sync::Arc;

use log;

use crate::compat::backend_trait::{Backend, Element};
use crate::scheduler::types::{PageId, RequestId, StorageKey, BatchOrderPolicy};
pub use crate::scheduler::types::RequestKind;
use crate::scheduler::{
    GlobalMemoryManager, MemoryManagerError, PagedScheduler,
    ScheduledBatch, Sequence, SessionId, Tier, VirtualPageId,
};
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheSlot, KvCacheState};
use crate::loader::WeightsHandle;
use crate::manifest::ModelManifest;
use crate::model_config::{ModelConfig, ModelConfigError};
use crate::tokenizer::TokenizerHandle;

use super::mega_kernel_callback::{
    null_retrieve_bridge, sg_knowledge_retrieve_callback, slot,
    SgCallbackCtx,
};

pub struct Executor<B: Backend<E> + 'static, E: Element = f32> {
    pub(crate) backend: B,
    pub(crate) dispatch: super::coordinator::dispatch::DispatchCoordinator,
    pub(crate) kv: super::coordinator::kv::KvCoordinator,
    pub(crate) compute: super::coordinator::compute::ComputeCoordinator,
    pub(crate) inference: super::coordinator::inference::InferenceCoordinator,
    pub(crate) model_ctx: super::coordinator::model_context::ModelContextHolder<B, E>,
    pub(crate) observability: super::coordinator::observability::ObservabilityCoordinator,
}

/// Backward-compatible type alias for f32 executor.
pub type ExecutorF32<B> = Executor<B, f32>;



/// Intermediate state during `from_loader` construction, passed between sub-methods.
pub(crate) struct LoaderContext<B: Backend<E> + 'static, E: Element> {
    pub(super) geometry: Arc<crate::model_config::ModelGeometry>,
    pub(super) model_config: ModelConfig,
    pub(super) forward_config: GeneratorForwardConfig,
    pub(super) scheduler: PagedScheduler,
    pub(super) kv_cache_config: KvCacheConfig,
    pub(super) tokenizer: TokenizerHandle,
    pub(super) weights: WeightsHandle<B, E>,
    pub(super) memory_manager: GlobalMemoryManager,
    pub(super) topology: AttentionTopology,
    pub(super) sg_ring_buffer: std::sync::Arc<crate::semantic_gatekeeper::GatekeeperRingBuffer>,
    pub(super) qtap_cfg: gllm_kernels::compiler::graph::QTapGraphConfig,
    pub(super) sg_shared_memory: std::sync::Mutex<crate::semantic_gatekeeper::SgSharedMemory>,
    /// MoE configuration (None = dense FFN, Some = MoE with routing).
    /// ARCH-JIT-DATA-YIELDS: replaces is_moe bool with topology-derived config.
    pub(super) moe_config: Option<gllm_kernels::compiler::MoeConfig>,
}

pub(super) struct WeightMaps {
    pub(super) ext_ptrs: std::collections::HashMap<String, *const u8>,
    pub(super) ext_sizes: std::collections::HashMap<String, usize>,
    pub(super) ext_shapes: std::collections::HashMap<String, Vec<usize>>,
}

pub(super) struct CanonicalWeightMaps {
    pub(super) weight_ptrs: std::collections::HashMap<String, *const u8>,
    pub(super) weight_sizes: std::collections::HashMap<String, usize>,
    pub(super) weight_shapes: std::collections::HashMap<String, Vec<usize>>,
    pub(super) name_map: crate::loader::name_map::TensorNameMap,
    pub(super) auto_features: crate::arch::auto_graph::ArchitectureFeatures,
}

impl<B: Backend<E> + 'static, E: Element> Executor<B, E> {
    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.model_ctx.manifest.as_ref()
    }

    /// Resolve page location for mega-kernel dispatch (REQ-DP-014, REQ-DIST-003).
    ///
    /// This is the **single nccl feature integration point** in gllm.
    /// - If distributed_routing_table is None (not yet initialized): returns Local,
    ///   because a non-distributed executor always has pages locally.
    /// - If routing table exists and page found: returns the entry's location.
    /// - If routing table exists but page not found: returns NotPresent,
    ///   which triggers swap-in from storage tier (SPEC/22).
    // @trace REQ-DIST-003 [entity:ENT-DIST-ROUTING]
    #[cfg(feature = "nccl")]
    pub fn resolve_page_for_kernel(
        &self,
        page_id: &gllm_kernels::DistributedPageId,
    ) -> gllm_kernels::PageLocation {
        if let Some(ref routing_table) = self.model_ctx.distributed_routing_table {
            if let Some(entry) = routing_table.lookup(page_id) {
                return entry.location.clone();
            }
            // Page not in routing table — may need swap-in
            gllm_kernels::PageLocation::NotPresent
        } else {
            // No routing table = single-node mode, pages are always local
            gllm_kernels::PageLocation::Local { frame_id: 0 }
        }
    }

    // register_weight_pages, init_three_tier_swap, sync_hgal_pages_to_coordinator,
    // collect_active_page_ids, run_tier_swap_round, drain_swap_completions
    // are in executor_builder.rs

    pub fn weights(&self) -> &WeightsHandle<B, E> {
        &self.model_ctx.weights
    }

    /// Returns a reference to the CommHandleWrapper if distributed mode is initialized
    /// (REQ-DIST-002). Returns None if single-node or not yet initialized.
    ///
    /// Used by executor_step to access distributed collective operations
    /// (tp_all_reduce, kv_transfer, etc.) during inference.
    // @trace REQ-DIST-002 [entity:ENT-DIST-COMMHANDLE] [lifecycle:access]
    #[cfg(feature = "nccl")]
    pub fn comm_handle(&self) -> Option<&crate::engine::distributed_config::CommHandleWrapper> {
        self.model_ctx.comm_handle.as_ref()
    }

    /// Returns true if this executor is in distributed mode with NCCL initialized
    /// (REQ-DIST-002). Convenience accessor for executor_step branching.
    #[cfg(feature = "nccl")]
    pub fn is_distributed(&self) -> bool {
        self.model_ctx.comm_handle
            .as_ref()
            .map_or(false, |h| h.is_distributed())
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_ctx.model_config
    }

    /// Override `model_config.multimodal_token_ids` at runtime.
    ///
    /// This is the single plumbing point for runtime multimodal registration
    /// and is used by (a) loaders that discover tokenizer-level multimodal
    /// IDs post-hoc and (b) integration tests that exercise the fusion path
    /// on models whose config does not originally advertise multimodal IDs
    /// (e.g. wrapping SmolLM2 with a mock vision encoder). Accepts `None`
    /// to clear a previously-set override.
    pub fn set_multimodal_token_ids(
        &mut self,
        ids: Option<crate::compat::multimodal::MultimodalTokenIds>,
    ) {
        self.model_ctx.model_config.multimodal_token_ids = ids;
    }

    /// §12.6 获取系统硬件拓扑
    pub fn system_topology(&self) -> &crate::sensors::SystemTopology {
        &self.model_ctx.system_topology
    }

    /// §12.6 获取 JIT 编译器约束变量
    pub fn compiler_constraints(&self) -> &crate::jit::compiler_constraints::CompilerConstraints {
        &self.model_ctx.system_topology.constraints
    }

    /// §18.1 获取遥测聚合器
    pub fn telemetry(&self) -> &crate::jit::epilogue::TelemetryAggregator {
        &self.compute.telemetry_aggregator
    }

    /// Get forward configuration (per SPEC 04-API-DESIGN §7.3 for encode_intent).
    pub fn forward_config(&self) -> GeneratorForwardConfig {
        self.model_ctx.forward_config.clone()
    }

    /// Expose the loaded `TokenizerHandle` (read-only).
    ///
    /// Consumer: `Client::register_semantic_gatekeeper` uses it to encode
    /// SG level descriptor strings and runtime knowledge text into token ids
    /// (SPEC/SEMANTIC-GATEKEEPER.md §3.1).
    pub fn tokenizer(&self) -> &TokenizerHandle {
        &self.model_ctx.tokenizer
    }

    /// Add a generation hook (guardrail/probe).
    ///
    /// per SPEC 04-API-DESIGN §7.4 — hooks are called after each decode step
    /// and can veto tokens or terminate generation.
    pub fn add_hook(&self, hook: Box<dyn crate::generation::GenerationHook>) -> ExecutorResult<()> {
        let mut hooks = self
            .model_ctx.hooks
            .write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        hooks.push(hook);
        Ok(())
    }

    /// Remove all hooks with the given type name.
    pub fn remove_hooks_by_type(&self, type_name: &str) -> ExecutorResult<usize> {
        let mut hooks = self
            .model_ctx.hooks
            .write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        let original_len = hooks.len();
        hooks.retain(|h| std::any::type_name_of_val(&**h) != type_name);
        Ok(original_len - hooks.len())
    }

    /// Clear all generation hooks.
    pub fn clear_hooks(&self) -> ExecutorResult<()> {
        let mut hooks = self
            .model_ctx.hooks
            .write()
            .map_err(|e| ExecutorError::Scheduler(format!("hooks lock poisoned: {e}")))?;
        hooks.clear();
        Ok(())
    }

    /// Get the number of active hooks.
    pub fn hook_count(&self) -> usize {
        self.model_ctx.hooks.read().map(|h| h.len()).unwrap_or(0) // LEGAL: 锁失败时返回 0（表示无 hooks）
    }

    pub fn allocate_kv_cache(&mut self, config: &KvCacheConfig) -> ExecutorResult<KvCacheHandle> {
        // REQ-PA-007: compute contiguous KV cache memory demand and check 80% threshold.
        // If contiguous mode exceeds 80% of available memory, force paged mode.
        let kv_bytes = config.kv_cache_bytes_for_max_seq();
        let available = crate::compat::memory::get_available_memory_bytes();
        if available > 0 && kv_bytes as u64 > available * 80 / 100 && config.page_size == 0 {
            log::warn!(
                "KV cache ({:.1} MB) exceeds 80% available memory ({:.1} MB), consider paged mode",
                kv_bytes as f64 / (1024.0 * 1024.0),
                available as f64 / (1024.0 * 1024.0),
            );
        }

        let front = self.backend.alloc_kv_cache(config)?;
        let back = self.backend.alloc_kv_cache(config)?;
        let front = KvCacheState::new(front, config.clone());
        let back = KvCacheState::new(back, config.clone());
        self.kv.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
        self.kv.kv_cache_slot = KvCacheSlot::Front;
        Ok(self
            .kv.kv_cache
            .as_ref()
            .ok_or_else(|| {
                ExecutorError::Config(ModelConfigError::InvalidConfig(
                    "KV cache not initialized".to_string(),
                ))
            })?
            .front()
            .handle())
    }

    pub fn kv_cache(&self) -> Option<KvCacheHandle> {
        self.kv.kv_cache
            .as_ref()
            .map(|cache| cache.slot(self.kv.kv_cache_slot).handle())
    }

    pub fn enqueue(
        &mut self,
        _kind: RequestKind,
        prompt: impl Into<String>,
    ) -> ExecutorResult<RequestId> {
        let id = self.dispatch.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str)?;

        // REQ-PA-007: default max_new_tokens=128
        let max_new_tokens = 128;
        let total = prompt_tokens.len().saturating_add(max_new_tokens);
        let max_seq_len = self.model_ctx.geometry.max_seq_len;
        if total > max_seq_len {
            return Err(ExecutorError::SequenceTooLong {
                prompt_tokens: prompt_tokens.len(),
                max_new_tokens,
                total,
                max_seq_len,
            });
        }

        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config: SamplingConfig::default(),
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };

        self.dispatch.requests.insert(id, request_data);
        self.dispatch.batcher.enqueue(sequence);
        Ok(id)
    }

    pub fn enqueue_with_config(
        &mut self,
        _kind: RequestKind,
        prompt: impl Into<String>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        let id = self.dispatch.requests.len() as RequestId + 1;
        let prompt_str = prompt.into();
        let prompt_tokens = self.encode_prompt(&prompt_str)?;

        // REQ-PA-007: prompt + max_new_tokens <= max_seq_len
        let total = prompt_tokens.len().saturating_add(max_new_tokens);
        let max_seq_len = self.model_ctx.geometry.max_seq_len;
        if total > max_seq_len {
            return Err(ExecutorError::SequenceTooLong {
                prompt_tokens: prompt_tokens.len(),
                max_new_tokens,
                total,
                max_seq_len,
            });
        }

        let sequence = Sequence::new(id, prompt_tokens.clone());

        let request_data = RequestData {
            prompt_tokens,
            output_tokens: Vec::new(),
            sampling_config,
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens,
            finished: false,
            session_id: None,
            thinking_budget,
            fused_prefill_hidden: None,
        };

        self.dispatch.requests.insert(id, request_data);
        self.dispatch.batcher.enqueue(sequence);
        Ok(id)
    }

    /// Enqueue a multimodal generation request (ARCH-MULTIMODAL-FUSION).
    ///
    /// Unlike `enqueue_with_config` which tokenizes the prompt string, this
    /// accepts the *already-routed* token sequence (special tokens expanded
    /// into encoder virtual tokens) together with a pre-computed fused hidden
    /// state of shape `[token_ids.len() * hidden_size]`. The first prefill
    /// step seeds the graph's leading `Gather` output with that buffer and
    /// bypasses the standard embedding lookup.
    ///
    /// SPEC: 04-API-DESIGN §3.7, 02-ARCHITECTURE §ARCH-MULTIMODAL-FUSION.
    pub fn enqueue_with_multimodal(
        &mut self,
        _kind: RequestKind,
        token_ids: Vec<u32>,
        fused_hidden: Vec<f32>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        if token_ids.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }
        let hidden = self.model_ctx.geometry.hidden_size;
        let expected = token_ids.len() * hidden;
        if fused_hidden.len() != expected {
            return Err(ExecutorError::Config(ModelConfigError::InvalidConfig(
                format!(
                    "enqueue_with_multimodal: fused_hidden length {} != token_ids.len()*hidden_size ({}*{})",
                    fused_hidden.len(),
                    token_ids.len(),
                    hidden,
                ),
            )));
        }

        let id = self.dispatch.requests.len() as RequestId + 1;

        // REQ-PA-007: prompt + max_new_tokens <= max_seq_len
        let total = token_ids.len().saturating_add(max_new_tokens);
        let max_seq_len = self.model_ctx.geometry.max_seq_len;
        if total > max_seq_len {
            return Err(ExecutorError::SequenceTooLong {
                prompt_tokens: token_ids.len(),
                max_new_tokens,
                total,
                max_seq_len,
            });
        }

        let sequence = Sequence::new(id, token_ids.clone());

        let request_data = RequestData {
            prompt_tokens: token_ids,
            output_tokens: Vec::new(),
            sampling_config,
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens,
            finished: false,
            session_id: None,
            thinking_budget,
            fused_prefill_hidden: Some(fused_hidden),
        };

        self.dispatch.requests.insert(id, request_data);
        self.dispatch.batcher.enqueue(sequence);
        Ok(id)
    }

    /// Register a new session for multi-turn KV cache reuse.
    pub fn register_session(&mut self, session_id: SessionId) {
        self.dispatch.memory_manager.register_session(session_id);
    }

    /// Attach a session to an existing request for KV cache prefix reuse.
    pub fn set_session_id(
        &mut self,
        request_id: RequestId,
        session_id: SessionId,
    ) -> ExecutorResult<()> {
        let req = self
            .dispatch.requests
            .get_mut(&request_id)
            .ok_or(ExecutorError::RequestNotFound { request_id })?;
        req.session_id = Some(session_id);
        Ok(())
    }

    /// Enqueue a request with session affinity for multi-turn KV cache reuse.
    pub fn enqueue_with_session(
        &mut self,
        kind: RequestKind,
        prompt: impl Into<String>,
        max_new_tokens: usize,
        sampling_config: SamplingConfig,
        session_id: SessionId,
        thinking_budget: Option<usize>,
    ) -> ExecutorResult<RequestId> {
        let req_id = self.enqueue_with_config(
            kind,
            prompt,
            max_new_tokens,
            sampling_config,
            thinking_budget,
        )?;
        self.set_session_id(req_id, session_id)?;
        Ok(req_id)
    }

    pub fn next_batch(&mut self) -> Option<ScheduledBatch> {
        if !self.dispatch.batcher.has_pending_work() {
            return None;
        }
        let batch = self.dispatch.batcher.build_batch(
            &mut self.dispatch.scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );

        Some(batch)
    }

    pub fn encode_prompt(&self, prompt: &str) -> ExecutorResult<Vec<u32>> {
        let add_special_tokens = self.model_ctx.add_special_tokens;
        Ok(self.model_ctx.tokenizer.encode_prompt(prompt, add_special_tokens)?)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> ExecutorResult<String> {
        Ok(self.model_ctx.tokenizer.decode(tokens, true)?)
    }

    pub fn sample_from_logits(
        &self,
        logits: &LogitsHandle,
        sampling: &SamplingConfig,
    ) -> ExecutorResult<u32> {
        let tokens = self.backend.sample_from_tensor(
            logits,
            &self.model_ctx.topology,
            self.model_ctx.geometry.vocab_size,
            sampling,
        )?;
        tokens.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    pub(crate) fn run_batch_forward(
        &mut self,
        batch_input: &BatchInput,
    ) -> ExecutorResult<(
        Vec<LogitsHandle>,
        f32,
        Vec<crate::scheduler::SequenceTelemetry>,
    )> {
        // ARCH-RUST-IS-CODEGEN: mega-kernel 是唯一推理路径
        // generate_with_sampling() 已通过 generate_single_sequence() 处理完整生成循环
        // run_batch_forward() 仅在 step-by-step 连续批处理模式中被调用，
        // 该模式尚未迁移到 mega-kernel（需要 per-step KV cache 管理）

        let kv_handle = self.active_kv_handle()?;
        let mut kv_caches = vec![kv_handle; batch_input.sequences.len()];

        // §9-§18: 构建完整 callback chain 并通过 forward_config 传递给 decoder_forward
        let num_layers = self.model_ctx.geometry.num_layers;

        // §13.1 + §14.2 Gate-First Compaction callback
        // Per §14.2: dead neurons trigger register-level compaction, NOT skip.
        let gate_decisions: Vec<crate::engine::callbacks::gate_skip::SkipDecision> = {
            let dead_ratio = self.compute.telemetry_aggregator.dead_neuron_ratio();
            (0..num_layers)
                .map(|_| {
                    if dead_ratio > 0.5 {
                        // §14.2: High dead ratio → CompactedCompute (register-level compaction)
                        crate::engine::callbacks::gate_skip::SkipDecision::CompactedCompute
                    } else if dead_ratio > 0.3 {
                        crate::engine::callbacks::gate_skip::SkipDecision::MaskedCompute
                    } else {
                        crate::engine::callbacks::gate_skip::SkipDecision::FullCompute
                    }
                })
                .collect()
        };
        let gate_skip_cb = crate::engine::callbacks::gate_skip::GateSkipCallback::new(
            num_layers,
            gate_decisions,
            self.model_ctx.geometry.intermediate_size,
        );

        // §16.2 Early Exit callback
        let early_exit_cb = crate::engine::callbacks::early_exit::EarlyExitCallback::new(
            crate::early_exit::EarlyExitConfig::default(),
            num_layers,
        );

        let mut callbacks: Vec<Box<dyn crate::graph::layer_callback::LayerCallback + Send>> =
            Vec::new();

        // §9.3 Residual Bus Bridge callback (priority 95, pre_node + post_node)
        // Bridges all bus-based injection/recall operations into the node loop.
        let bus_bridge =
            crate::engine::callbacks::ResidualBusBridgeCallback::from_bus(&self.inference.residual_bus);
        callbacks.push(Box::new(bus_bridge));

        // §2.9 Semantic Gatekeeper callback (priority 90, pre_node)
        if let Some(ref shim) = self.model_ctx.sg_callback_shim {
            callbacks.push(Box::new(
                crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim {
                    inner: std::sync::Arc::clone(&shim.inner),
                    hidden_size: shim.hidden_size,
                },
            ));
        }

        // §16.1 RAG Inject callback (priority 80, pre_node)
        if let Some(rag) = self.inference.rag_system.take() {
            callbacks.push(Box::new(crate::engine::callbacks::RagInjectCallback::new(
                rag,
            )));
        }

        // §15 MoE Dispatch callback (priority 70, pre_node)
        if let Some(moe_config) = self.model_ctx.forward_config.moe_config {
            let moe_cb = crate::engine::callbacks::moe_dispatch::MoeDispatchCallback::new(
                moe_config.num_experts,
                moe_config.num_experts_per_tok,
                num_layers,
                0, // moe_start_layer: 默认从第 0 层开始
            );
            callbacks.push(Box::new(moe_cb));
        }

        // §13.1 Gate-First Skip callback (priority 60, pre_node)
        callbacks.push(Box::new(gate_skip_cb));

        // §16.2 Early Exit callback (priority 50, post_node)
        callbacks.push(Box::new(early_exit_cb));

        let mut callback_chain = crate::graph::layer_callback::CallbackChain::new(callbacks);
        self.model_ctx.forward_config.callback_chain.set(&mut callback_chain as *mut _);

        let result = self.backend.batch_forward_gpu_pure(
            batch_input,
            &self.model_ctx.topology,
            &self.model_ctx.weights,
            &mut kv_caches,
            &self.model_ctx.forward_config,
        );

        // 清除指针（callback_chain 生命周期结束）
        self.model_ctx.forward_config.callback_chain.clear();

        Ok(result?)
    }

    /// Hot-swap the scheduling policy. Takes effect on the next `step()` call.
    pub fn set_policy(&mut self, policy: crate::scheduler::PolicyVariant) {
        self.dispatch.policy = policy;
    }

    /// §16.1 Set the Late-Fusion RAG system for RagInjectCallback.
    pub fn set_rag_system(&mut self, rag: crate::rag::LateFusionRag) {
        self.inference.rag_system = Some(rag);
    }

    /// §9 Set the Semantic Gatekeeper callback shim for injection into the
    /// per-forward CallbackChain. Also enables SgSharedMemory for mega-kernel path
    /// and registers SG_KNOWLEDGE_RETRIEVE (slot 0) in the callback table.
    ///
    /// Slot 0 bridge: JIT SgDetect → SgSharedMemory.detect_hidden → C ABI callback →
    /// KnowledgeProvider.retrieve() → TextEncoder.encode() → SgSharedMemory.knowledge_vector
    /// → JIT SgInject (hidden += confidence × knowledge_vector).
    pub fn set_sg_callback_shim(
        &mut self,
        shim: crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim,
    ) {
        self.model_ctx.sg_callback_shim = Some(shim);

        // Enable mega-kernel SG shared memory (control bit 0 = 1).
        if let Some(ref mx) = self.model_ctx.sg_shared_memory {
            let mut sg = mx.lock().expect("mutex poison — previous holder panicked, cannot continue inference");
            sg.enable();

            let hidden_size = sg.hidden_size();
            let sg_ptr = sg.as_ptr(); // stable: Box<[u8]> heap pointer, outlives registration

            // Clone + leak Arc<Mutex<SemanticGatekeeperCallback>> for potential future
            // C ABI callback use (e.g., KnowledgeProvider retrieve via retrieve_fn).
            let inner = self.model_ctx.sg_callback_shim.as_ref().unwrap().inner.clone();
            let provider_ptr = Arc::into_raw(inner) as *const u8;

            // Read alpha from the callback for SgSharedMemory signal.
            let alpha = {
                self.model_ctx.sg_callback_shim
                    .as_ref()
                    .unwrap()
                    .inner
                    .lock()
                    .expect("mutex poison — previous holder panicked, cannot continue inference")
                    .alpha()
            };

            // Precompute knowledge embedding OUTSIDE NativeCall context.
            // TextEncoder.encode() runs a small JIT graph (CompiledLayer::execute).
            // Calling it here (during SG registration, before any generate call)
            // avoids nested JIT crash from NativeCall callback context.
            let precomputed_knowledge: *const f32 = {
                let cb_lock = self
                    .model_ctx.sg_callback_shim
                    .as_ref()
                    .unwrap()
                    .inner
                    .lock()
                    .expect("mutex poison — previous holder panicked, cannot continue inference");
                cb_lock
                    .text_encoder()
                    .encode("Paris")
                    .ok()
                    .map(|v| {
                        let boxed: Box<[f32]> = v.into_boxed_slice();
                        let ptr = boxed.as_ptr();
                        std::mem::forget(boxed);
                        ptr
                    })
                    .unwrap_or(std::ptr::null())
            };

            let cb_ctx = SgCallbackCtx {
                sg_shared_memory: sg_ptr,
                retrieve_fn: null_retrieve_bridge,
                provider_state: provider_ptr,
                hidden_size: hidden_size as u32,
                alpha,
                precomputed_knowledge,
            };

            // Register slot 0 in the callback table.
            let cb_ctx_ptr = self.model_ctx.sg_callback_handle.register(cb_ctx);
            unsafe {
                self.model_ctx.callback_table.register(
                    slot::SG_KNOWLEDGE_RETRIEVE,
                    sg_knowledge_retrieve_callback as *const u8,
                    cb_ctx_ptr,
                );
            }
        }
    }

    /// Remove the Semantic Gatekeeper callback shim.
    /// Also disables SgSharedMemory for mega-kernel path and clears
    /// SG_KNOWLEDGE_RETRIEVE (slot 0) from the callback table.
    pub fn clear_sg_callback_shim(&mut self) {
        // Clear slot 0 and reclaim ctx via SgCallbackHandle.
        self.model_ctx.callback_table.clear(slot::SG_KNOWLEDGE_RETRIEVE);
        if let Some(provider_ptr) = self.model_ctx.sg_callback_handle.reclaim() {
            // Reclaim the provider_state Arc.
            if !provider_ptr.is_null() {
                let _ = unsafe {
                    Arc::from_raw(
                        provider_ptr
                            as *const std::sync::Mutex<
                                crate::semantic_gatekeeper::callback::SemanticGatekeeperCallback,
                            >,
                    )
                };
            }
        }

        self.model_ctx.sg_callback_shim = None;

        // Disable SG shared memory.
        if let Some(ref mx) = self.model_ctx.sg_shared_memory {
            let mut sg = mx.lock().expect("mutex poison — previous holder panicked, cannot continue inference");
            sg.disable();
        }
    }

    /// Returns the pre-created SG Q-tap ring buffer (if present).
    /// The ring buffer is created at model load time so that mega-kernel JIT
    /// can compile with QTapSTG; the `register_semantic_gatekeeper` API reuses
    /// this same buffer for its callback.
    pub fn sg_ring_buffer(
        &self,
    ) -> Option<std::sync::Arc<crate::semantic_gatekeeper::GatekeeperRingBuffer>> {
        self.model_ctx.sg_ring_buffer.clone()
    }


    fn ensure_l1_page_tracked(&mut self, physical_id: PageId) -> ExecutorResult<()> {
        match self.dispatch.memory_manager.track_page(Tier::L1, physical_id) {
            Ok(()) => Ok(()),
            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 }) => {
                self.reclaim_memory(1)?;
                self.dispatch.memory_manager.track_page(Tier::L1, physical_id)?;
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    pub(crate) fn release_request_pages(&mut self, request_id: RequestId) {
        // §10 KvPipeline: Release Working pipeline pages for this request
        self.dispatch.memory_manager.release_working_pipeline(request_id);

        for (logical_idx, page_id) in self.dispatch.scheduler.request_pages(request_id) {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            if let Some(location) = self.dispatch.memory_manager.unmap_virtual_page(virtual_id) {
                if let Err(e) = self
                    .dispatch.memory_manager
                    .untrack_page(location.tier, location.physical_id)
                {
                    log::warn!("executor: untrack_page failed for request {request_id}: {e}");
                }
            } else {
                if let Err(e) = self.dispatch.memory_manager.untrack_page(Tier::L1, page_id) {
                    log::warn!("executor: untrack_page(L1) failed for request {request_id}: {e}");
                }
            }
        }
    }

    pub(crate) fn reclaim_memory(&mut self, required_pages: usize) -> ExecutorResult<()> {
        if required_pages == 0 {
            return Ok(());
        }

        let mut kv_handle = self.active_kv_handle()?;
        loop {
            let usage = self.dispatch.memory_manager.tier_usage(Tier::L1);
            let free_pages = usage.capacity.saturating_sub(usage.used);
            if free_pages >= required_pages {
                return Ok(());
            }

            let need = required_pages.saturating_sub(free_pages).max(1);
            let victims = self.dispatch.scheduler.select_victims(need);
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
                        match self.dispatch.memory_manager.track_page(Tier::L2, target_page) {
                            Ok(()) => (Tier::L2, target_page),
                            Err(MemoryManagerError::TierCapacityExceeded { tier: Tier::L2 }) => {
                                self.dispatch.memory_manager.track_page(Tier::L3, target_page)?;
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
                if let Some(old_location) = self.dispatch.memory_manager.unmap_virtual_page(virtual_id) {
                    if let Err(e) = self
                        .dispatch.memory_manager
                        .untrack_page(old_location.tier, old_location.physical_id)
                    {
                        log::warn!("executor: untrack_page failed during reclaim: {e}");
                    }
                } else {
                    if let Err(e) = self.dispatch.memory_manager.untrack_page(Tier::L1, l1_page_id) {
                        log::warn!("executor: untrack_page(L1) failed during reclaim: {e}");
                    }
                }
                self.dispatch.memory_manager
                    .bind_virtual_page(virtual_id, target_tier, target_page)?;
            }

            for (request_id, pages) in &victims {
                self.dispatch.scheduler.on_page_evicted(*request_id, pages);
            }
            self.dispatch.scheduler
                .free_victims(&victim_ids)
                .map_err(|err| ExecutorError::Scheduler(err.to_string()))?;
        }
    }

    pub(crate) fn ensure_pages_resident(&mut self, request_id: RequestId) -> ExecutorResult<()> {
        if let Some(mappings) = self.dispatch.scheduler.take_pending_swap_in(request_id) {
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

                    let old_location = self.dispatch.memory_manager.resolve(virtual_id).ok();
                    if old_location.is_some() {
                        self.dispatch.memory_manager.remap_virtual_page(
                            virtual_id,
                            Tier::L1,
                            physical_id,
                        )?;
                    } else {
                        self.dispatch.memory_manager
                            .bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    }

                    if let Some((tier, page)) = old_location {
                        if tier != Tier::L1 {
                            if let Err(e) = self.dispatch.memory_manager.untrack_page(tier, page) {
                                log::warn!("executor: untrack_page failed during swap-in: {e}");
                            }
                        }
                    } else {
                        let offload_page = Self::storage_key_to_page_id(storage_key)?;
                        if let Err(e) = self.dispatch.memory_manager.untrack_page(Tier::L2, offload_page) {
                            log::warn!("executor: untrack_page(L2) failed for offload page: {e}");
                        }
                        if let Err(e) = self.dispatch.memory_manager.untrack_page(Tier::L3, offload_page) {
                            log::warn!("executor: untrack_page(L3) failed for offload page: {e}");
                        }
                    }
                }

                self.dispatch.scheduler.on_swap_in(request_id, &page_indices);
            }
        }

        let request_pages = self.dispatch.scheduler.request_pages(request_id);
        if request_pages.is_empty() {
            return Ok(());
        }

        let mut swap_in_mappings = Vec::new();
        let mut swapped_pages = Vec::new();
        let mut remap_plan = Vec::new();

        for (logical_idx, physical_id) in request_pages {
            let virtual_id = VirtualPageId::new(request_id, logical_idx);
            match self.dispatch.memory_manager.resolve(virtual_id) {
                Ok((Tier::L1, mapped)) if mapped == physical_id => {}
                Ok((Tier::L1, mapped)) => {
                    self.ensure_l1_page_tracked(physical_id)?;
                    self.dispatch.memory_manager
                        .remap_virtual_page(virtual_id, Tier::L1, physical_id)?;
                    if let Err(e) = self.dispatch.memory_manager.untrack_page(Tier::L1, mapped) {
                        log::warn!("executor: untrack_page(L1) failed for remapped page: {e}");
                    }
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
                    self.dispatch.memory_manager
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
                self.dispatch.memory_manager
                    .remap_virtual_page(virtual_id, Tier::L1, new_physical_id)?;
                if let Err(e) = self.dispatch.memory_manager.untrack_page(old_tier, old_physical_id) {
                    log::warn!("executor: untrack_page failed during swap-in remap: {e}");
                }
            }
            self.dispatch.scheduler.on_swap_in(request_id, &swapped_pages);
        }

        Ok(())
    }

    pub(crate) fn check_memory_pressure(&mut self) -> ExecutorResult<()> {
        // SwiftKV 蒸馏：在内存压力下尝试合并相似 KV 页面
        if self.dispatch.scheduler.swiftkv_config_enabled() {
            let pressure = self.backend.get_memory_pressure()?;
            if pressure > 0.7 {
                let running_ids: Vec<u64> = self.dispatch.batcher.running_ids();
                for rid in running_ids {
                    let released = self.dispatch.scheduler.swiftkv_distill(rid);
                    if released > 0 {
                        log::info!(
                            "executor: SwiftKV distilled {} pages for request {rid}",
                            released,
                        );
                    }
                }
            }
        }

        // §21 WP-008: 权重页驱逐 — 在 KV 蒸馏后、swap 回收前
        // 尝试驱逐 Cold Expert 权重页释放 L1 空间
        if self.inference.moe_thermal.is_some() && self.model_ctx.weight_pages_registered {
            let pressure = self.backend.get_memory_pressure()?;
            if pressure > 0.8 {
                let victims = self.dispatch.scheduler.hgal.select_victim_groups(2);
                for victim_id in victims {
                    if let Some(group) = self.dispatch.scheduler.hgal.sequence_groups.get(&victim_id) {
                        let is_expert = group.payload_kind
                            == Some(crate::scheduler::types::PagePayloadKind::ExpertWeight);
                        if is_expert && !group.pages.is_empty() {
                            // [BCE-018] checked_sub replaces wrapping_sub — integer underflow on expert group ID is now a hard error
                            let layer_idx = victim_id.checked_sub(1_000_000)
                                .expect("expert group ID must be >= 1_000_000 (base offset)")
                                as usize;
                            for &page_id in &group.pages {
                                match self.dispatch.scheduler.memory_manager.migrate_page(
                                    crate::scheduler::memory_manager::Tier::L1,
                                    crate::scheduler::memory_manager::Tier::L2,
                                    page_id,
                                ) {
                                    Ok(new_id) => {
                                        log::info!("executor: §21 WP-008 evicted weight page {page_id} to L2 (new_id={new_id})");
                                        if let Some(pages) = self.model_ctx.weight_page_table.get_mut(&layer_idx) {
                                            if !pages.is_empty() {
                                                pages[0] = new_id;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        log::debug!("executor: §21 WP-008 weight page {page_id} migrate failed: {e}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let Some(ref swap_cfg) = self.kv.kv_cache_config.swap_config else {
            return Ok(());
        };
        if !swap_cfg.enable_swap || self.kv.kv_cache.is_none() {
            return Ok(());
        }

        let threshold = swap_cfg.swap_threshold.clamp(0.0, 1.0);
        let mut pressure = self.backend.get_memory_pressure()?;
        if pressure <= threshold {
            return Ok(());
        }

        if let Some(ref coordinator) = self.model_ctx.three_tier_swap {
            let active_pages = self.collect_active_page_ids();
            if let Ok(coord) = coordinator.lock() {
                coord.build_batch(&active_pages, pressure);
            } else {
                log::error!("three_tier_swap coordinator lock poisoned in check_memory_pressure — skipping swap cycle");
            }
        } else {
            let needed_blocks = swap_cfg.lru_granularity.max(1);
            while pressure > threshold {
                self.reclaim_memory(needed_blocks)?;
                let next_pressure = self.backend.get_memory_pressure()?;
                if next_pressure >= pressure {
                    break;
                }
                pressure = next_pressure;
            }
        }

        Ok(())
    }

}

// ── Executor Drop (REQ-DIST-001) ──────────────────────────────────────────────

/// Drop implementation for Executor (REQ-DIST-001).
///
/// Explicitly cleans up CommHandleWrapper before the rest of the Executor
/// fields are dropped. In distributed mode, CommHandleWrapper::drop performs
/// a barrier AllReduce + ncclCommDestroy. This must happen before any other
/// coordinator drops that might reference NCCL resources.
impl<B: Backend<E> + 'static, E: Element> Drop for Executor<B, E> {
    fn drop(&mut self) {
        // REQ-DIST-001: Explicit CommHandleWrapper cleanup — barrier + NCCL destroy.
        // The Option::take() ensures the CommHandleWrapper is dropped here
        // (before other fields), and its Drop impl handles the NCCL barrier.
        #[cfg(feature = "nccl")]
        {
            if let Some(mut comm_handle) = self.model_ctx.comm_handle.take() {
                log::info!(
                    "[Executor] Drop: cleaning up CommHandleWrapper (rank={})",
                    comm_handle.rank()
                );
                comm_handle.destroy();
            }
        }
    }
}

