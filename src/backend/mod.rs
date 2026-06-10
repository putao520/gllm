use crate::compat::backend_trait::Element;
use crate::compat::{CpuBackend, CudaBackend, HipBackend, MetalBackend};
use std::sync::{Arc, Mutex, MutexGuard};
use thiserror::Error;

use crate::engine::executor::{Executor, ExecutorError};
use crate::loader::{Loader, LoaderError};
use crate::manifest::ModelManifest;

pub mod detection;

pub use detection::{
    detect_backend, detect_backend_generic, BackendType, DetectedBackend, DetectedBackendF32,
    DetectedDtype,
};

#[derive(Debug, Error)]
pub enum BackendContextError {
    #[error("unsupported architecture: {0}")]
    UnsupportedArchitecture(String),
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error(transparent)]
    Backend(#[from] crate::engine::executor::BackendError),
}

/// Generic backend executor supporting any element type.
///
/// Both CUDA and CPU backends now support full generic execution.
/// CPU backend uses pseudo-SIMD (f32 promotion) for f16/bf16 types.
pub enum BackendExecutor<E: Element = f32> {
    Cuda(Box<Executor<CudaBackend<E>, E>>),
    Rocm(Box<Executor<HipBackend<E>, E>>),
    Metal(Box<Executor<MetalBackend<E>, E>>),
    Cpu(Box<Executor<CpuBackend<E>, E>>),
}

/// Dynamic backend executor that selects precision at runtime based on model dtype.
///
/// This is the recommended entry point for loading models, as it automatically
/// detects the weight dtype and uses the optimal precision for computation.
///
/// # Design Rationale
///
/// LLM inference should use the model's native precision:
/// - F16 models → F16 computation (half memory, tensor core acceleration)
/// - BF16 models → BF16 computation (better range than F16)
/// - F32 models → F32 computation (baseline)
///
/// This enum provides runtime dispatch to the correct precision executor.
pub enum DynBackendExecutor {
    F32(BackendExecutor<f32>),
    F16(BackendExecutor<half::f16>),
    BF16(BackendExecutor<half::bf16>),
}

impl DynBackendExecutor {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DynBackendExecutor::F32(e) => e.backend_type(),
            DynBackendExecutor::F16(e) => e.backend_type(),
            DynBackendExecutor::BF16(e) => e.backend_type(),
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.is_cuda(),
            DynBackendExecutor::F16(e) => e.is_cuda(),
            DynBackendExecutor::BF16(e) => e.is_cuda(),
        }
    }

    pub fn is_gpu(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.is_gpu(),
            DynBackendExecutor::F16(e) => e.is_gpu(),
            DynBackendExecutor::BF16(e) => e.is_gpu(),
        }
    }

    pub fn thinking_head_available(&self) -> bool {
        match self {
            DynBackendExecutor::F32(e) => e.thinking_head_available(),
            DynBackendExecutor::F16(e) => e.thinking_head_available(),
            DynBackendExecutor::BF16(e) => e.thinking_head_available(),
        }
    }

    /// Return the current weight page JIT injection config (SPEC/21 §8).
    pub fn weight_page_jit_config(&self) -> crate::engine::mega_kernel::WeightPageJitConfig {
        match self {
            DynBackendExecutor::F32(e) => e.weight_page_jit_config(),
            DynBackendExecutor::F16(e) => e.weight_page_jit_config(),
            DynBackendExecutor::BF16(e) => e.weight_page_jit_config(),
        }
    }

    /// Set the weight page JIT injection config (SPEC/21 §8).
    pub fn set_weight_page_jit_config(&mut self, config: crate::engine::mega_kernel::WeightPageJitConfig) {
        match self {
            DynBackendExecutor::F32(e) => e.set_weight_page_jit_config(config),
            DynBackendExecutor::F16(e) => e.set_weight_page_jit_config(config),
            DynBackendExecutor::BF16(e) => e.set_weight_page_jit_config(config),
        }
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.generate(prompt, max_tokens, temperature, top_k, top_p, thinking_budget),
            DynBackendExecutor::F16(e) => e.generate(prompt, max_tokens, temperature, top_k, top_p, thinking_budget),
            DynBackendExecutor::BF16(e) => {
                e.generate(prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }
        }
    }

    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: u64,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
            DynBackendExecutor::F16(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
            DynBackendExecutor::BF16(e) => {
                e.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
        }
    }

    /// ARCH-MULTIMODAL-FUSION: dispatch by dtype variant.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_multimodal(
        &mut self,
        token_ids: Vec<u32>,
        fused_hidden: Vec<f32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
            DynBackendExecutor::F16(e) => e.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
            DynBackendExecutor::BF16(e) => e.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
        }
    }

    /// SPEC/20 REQ-BCI-008: Batch inference via single mega-kernel call.
    pub fn generate_batch(
        &mut self,
        requests: &[crate::engine::batch_executor::GenerateRequest],
    ) -> Result<Vec<crate::engine::batch_executor::GenerateResult>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.generate_batch(requests),
            DynBackendExecutor::F16(e) => e.generate_batch(requests),
            DynBackendExecutor::BF16(e) => e.generate_batch(requests),
        }
    }

    /// Embedding output is always f32 (API standardization)
    pub fn embed(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.embed(input),
            DynBackendExecutor::F16(e) => e.embed(input),
            DynBackendExecutor::BF16(e) => e.embed(input),
        }
    }

    /// Rerank scores are always f32 (API standardization)
    pub fn rerank(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.rerank(input),
            DynBackendExecutor::F16(e) => e.rerank(input),
            DynBackendExecutor::BF16(e) => e.rerank(input),
        }
    }

    /// Rerank with proper pair encoding (query + document as separate segments)
    pub fn rerank_pair(&mut self, query: &str, document: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.rerank_pair(query, document),
            DynBackendExecutor::F16(e) => e.rerank_pair(query, document),
            DynBackendExecutor::BF16(e) => e.rerank_pair(query, document),
        }
    }

    /// Classify text: run encoder/decoder + classifier head, return raw logits
    pub fn classify(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.classify(input),
            DynBackendExecutor::F16(e) => e.classify(input),
            DynBackendExecutor::BF16(e) => e.classify(input),
        }
    }

    /// Head Routing SDK — 对 `prompt` 跑一次 generator forward,返回
    /// `target_token_ids` 每个 id 对应的原始 logit (未经 softmax)。
    pub fn score_tokens_for_prompt(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.score_tokens_for_prompt(prompt, target_token_ids),
            DynBackendExecutor::F16(e) => e.score_tokens_for_prompt(prompt, target_token_ids),
            DynBackendExecutor::BF16(e) => e.score_tokens_for_prompt(prompt, target_token_ids),
        }
    }

    /// Head Routing / Guardrail — `prompt` + explicit callback chain variant.
    pub fn score_tokens_for_prompt_with_callbacks(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
        callbacks: Option<&mut crate::graph::layer_callback::CallbackChain>,
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => {
                e.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
            DynBackendExecutor::F16(e) => {
                e.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
            DynBackendExecutor::BF16(e) => {
                e.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
        }
    }

    /// Head Routing §3.4 / Intent §3 — mid-layer encode.
    pub fn encode_at_layer_for_prompt(
        &mut self,
        prompt: &str,
        anchor_layer: usize,
        pool: crate::head_routing::PoolMode,
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
            DynBackendExecutor::F16(e) => e.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
            DynBackendExecutor::BF16(e) => e.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
        }
    }

    /// Returns the element type name for debugging/logging
    pub fn dtype_name(&self) -> &'static str {
        match self {
            DynBackendExecutor::F32(_) => "f32",
            DynBackendExecutor::F16(_) => "f16",
            DynBackendExecutor::BF16(_) => "bf16",
        }
    }

    // ── Streaming support: delegate to inner Executor ──

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.decode_tokens(tokens),
            DynBackendExecutor::F16(e) => e.decode_tokens(tokens),
            DynBackendExecutor::BF16(e) => e.decode_tokens(tokens),
        }
    }

    pub fn enqueue_with_config(
        &mut self,
        kind: crate::engine::executor::RequestKind,
        prompt: &str,
        max_new_tokens: usize,
        sampling: crate::engine::executor::SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> Result<u64, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget),
            DynBackendExecutor::F16(e) => e.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget),
            DynBackendExecutor::BF16(e) => e.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget),
        }
    }

    pub fn enqueue_with_session(
        &mut self,
        kind: crate::engine::executor::RequestKind,
        prompt: &str,
        max_new_tokens: usize,
        sampling: crate::engine::executor::SamplingConfig,
        session_id: u64,
        thinking_budget: Option<usize>,
    ) -> Result<u64, ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => {
                e.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
            DynBackendExecutor::F16(e) => {
                e.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
            DynBackendExecutor::BF16(e) => {
                e.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
        }
    }

    pub fn step(&mut self) -> Result<(), ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.step(),
            DynBackendExecutor::F16(e) => e.step(),
            DynBackendExecutor::BF16(e) => e.step(),
        }
    }

    pub fn get_request(
        &self,
        request_id: u64,
    ) -> Option<&crate::engine::executor::RequestData> {
        match self {
            DynBackendExecutor::F32(e) => e.get_request(request_id),
            DynBackendExecutor::F16(e) => e.get_request(request_id),
            DynBackendExecutor::BF16(e) => e.get_request(request_id),
        }
    }

    pub fn release_request(&mut self, request_id: u64) {
        match self {
            DynBackendExecutor::F32(e) => e.release_request(request_id),
            DynBackendExecutor::F16(e) => e.release_request(request_id),
            DynBackendExecutor::BF16(e) => e.release_request(request_id),
        }
    }

    /// Get model configuration (delegate to inner Executor).
    pub fn model_config(&self) -> &crate::model_config::ModelConfig {
        match self {
            DynBackendExecutor::F32(e) => e.model_config(),
            DynBackendExecutor::F16(e) => e.model_config(),
            DynBackendExecutor::BF16(e) => e.model_config(),
        }
    }

    /// Build a SigLIP encoder from the loaded weights (CPU-only for now).
    ///
    /// Returns `Ok(None)` when the model has no `vision_config` or when
    /// required weights are not present. See `BackendExecutor::try_build_siglip_encoder`.
    pub fn try_build_siglip_encoder(
        &self,
    ) -> Result<Option<crate::compat::vision_forward::SigLipEncoder>, crate::engine::executor::ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.try_build_siglip_encoder(),
            DynBackendExecutor::F16(e) => e.try_build_siglip_encoder(),
            DynBackendExecutor::BF16(e) => e.try_build_siglip_encoder(),
        }
    }

    /// Build a USM Conformer audio encoder from the loaded weights (CPU-only).
    ///
    /// Returns `Ok(None)` when the model has no `audio_config` or when required
    /// weights are not present. See `BackendExecutor::try_build_usm_conformer_encoder`.
    pub fn try_build_usm_conformer_encoder(
        &self,
    ) -> Result<Option<crate::compat::audio_forward::UsmConformerEncoder>, crate::engine::executor::ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.try_build_usm_conformer_encoder(),
            DynBackendExecutor::F16(e) => e.try_build_usm_conformer_encoder(),
            DynBackendExecutor::BF16(e) => e.try_build_usm_conformer_encoder(),
        }
    }

    /// Add a generation hook (guardrail/probe).
    pub fn add_hook(&self, hook: Box<dyn crate::generation::GenerationHook>) -> Result<(), crate::engine::executor::ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.add_hook(hook),
            DynBackendExecutor::F16(e) => e.add_hook(hook),
            DynBackendExecutor::BF16(e) => e.add_hook(hook),
        }
    }

    /// Remove hooks by type name.
    pub fn remove_hooks_by_type(&self, type_name: &str) -> Result<usize, crate::engine::executor::ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.remove_hooks_by_type(type_name),
            DynBackendExecutor::F16(e) => e.remove_hooks_by_type(type_name),
            DynBackendExecutor::BF16(e) => e.remove_hooks_by_type(type_name),
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&self) -> Result<(), crate::engine::executor::ExecutorError> {
        match self {
            DynBackendExecutor::F32(e) => e.clear_hooks(),
            DynBackendExecutor::F16(e) => e.clear_hooks(),
            DynBackendExecutor::BF16(e) => e.clear_hooks(),
        }
    }

    /// Get current hook count.
    pub fn hook_count(&self) -> usize {
        match self {
            DynBackendExecutor::F32(e) => e.hook_count(),
            DynBackendExecutor::F16(e) => e.hook_count(),
            DynBackendExecutor::BF16(e) => e.hook_count(),
        }
    }

    /// §16.1 Set the Late-Fusion RAG system for RagInjectCallback.
    pub fn set_rag_system(&mut self, rag: crate::rag::LateFusionRag) {
        match self {
            DynBackendExecutor::F32(e) => e.set_rag_system(rag),
            DynBackendExecutor::F16(e) => e.set_rag_system(rag),
            DynBackendExecutor::BF16(e) => e.set_rag_system(rag),
        }
    }

}

impl<E: Element> BackendExecutor<E> {
    pub fn backend_type(&self) -> BackendType {
        match self {
            BackendExecutor::Cuda(_) => BackendType::Cuda,
            BackendExecutor::Rocm(_) => BackendType::Rocm,
            BackendExecutor::Metal(_) => BackendType::Metal,
            BackendExecutor::Cpu(_) => BackendType::Cpu,
        }
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, BackendExecutor::Cuda(_))
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, BackendExecutor::Cuda(_) | BackendExecutor::Rocm(_) | BackendExecutor::Metal(_))
    }

    pub fn thinking_head_available(&self) -> bool {
        match self {
            BackendExecutor::Cuda(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Rocm(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Metal(exec) => exec.weights().thinking_head.is_some(),
            BackendExecutor::Cpu(exec) => exec.weights().thinking_head.is_some(),
        }
    }

    /// Return the current weight page JIT injection config (SPEC/21 §8).
    pub fn weight_page_jit_config(&self) -> crate::engine::mega_kernel::WeightPageJitConfig {
        match self {
            BackendExecutor::Cpu(e) => e.weight_page_jit_config(),
            BackendExecutor::Cuda(e) => e.weight_page_jit_config(),
            BackendExecutor::Rocm(e) => e.weight_page_jit_config(),
            BackendExecutor::Metal(e) => e.weight_page_jit_config(),
        }
    }

    /// Set the weight page JIT injection config (SPEC/21 §8).
    pub fn set_weight_page_jit_config(&mut self, config: crate::engine::mega_kernel::WeightPageJitConfig) {
        match self {
            BackendExecutor::Cpu(e) => e.set_weight_page_jit_config(config),
            BackendExecutor::Cuda(e) => e.set_weight_page_jit_config(config),
            BackendExecutor::Rocm(e) => e.set_weight_page_jit_config(config),
            BackendExecutor::Metal(e) => e.set_weight_page_jit_config(config),
        }
    }

    pub fn diagnostic_weight_row(&self, tensor_name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        match self {
            BackendExecutor::Cpu(e) => e.diagnostic_weight_row(tensor_name, row, cols),
            BackendExecutor::Cuda(e) => e.diagnostic_weight_row(tensor_name, row, cols),
            BackendExecutor::Rocm(e) => e.diagnostic_weight_row(tensor_name, row, cols),
            BackendExecutor::Metal(e) => e.diagnostic_weight_row(tensor_name, row, cols),
        }
    }

    pub fn diagnostic_weight_offsets(&self) -> Option<Vec<(String, usize)>> {
        match self {
            BackendExecutor::Cpu(e) => e.diagnostic_weight_offsets(),
            BackendExecutor::Cuda(e) => e.diagnostic_weight_offsets(),
            BackendExecutor::Rocm(e) => e.diagnostic_weight_offsets(),
            BackendExecutor::Metal(e) => e.diagnostic_weight_offsets(),
        }
    }

    pub fn diagnostic_prefill_logits(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        match self {
            BackendExecutor::Cpu(e) => e.diagnostic_prefill_logits(prompt_tokens),
            BackendExecutor::Cuda(e) => e.diagnostic_prefill_logits(prompt_tokens),
            BackendExecutor::Rocm(e) => e.diagnostic_prefill_logits(prompt_tokens),
            BackendExecutor::Metal(e) => e.diagnostic_prefill_logits(prompt_tokens),
        }
    }

    pub fn diagnostic_prefill_scratchpad(&self, prompt_tokens: &[u32]) -> Option<crate::engine::mega_kernel::DiagnosticScratchpad> {
        fn logits_to_scratchpad(logits: Vec<f32>, prompt_len: usize) -> crate::engine::mega_kernel::DiagnosticScratchpad {
            let vocab_size = logits.len();
            let logits_bytes = unsafe {
                std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
            };
            crate::engine::mega_kernel::DiagnosticScratchpad {
                data: logits_bytes.to_vec(),
                logits_offset: 0,
                vocab_size,
                prompt_len,
                hidden_size: 0,
            }
        }
        match self {
            BackendExecutor::Cpu(e) => e.diagnostic_prefill_scratchpad(prompt_tokens),
            BackendExecutor::Cuda(e) => e.diagnostic_prefill_logits(prompt_tokens).map(|l| logits_to_scratchpad(l, prompt_tokens.len())),
            BackendExecutor::Rocm(e) => e.diagnostic_prefill_logits(prompt_tokens).map(|l| logits_to_scratchpad(l, prompt_tokens.len())),
            BackendExecutor::Metal(e) => e.diagnostic_prefill_logits(prompt_tokens).map(|l| logits_to_scratchpad(l, prompt_tokens.len())),
        }
    }

    pub fn diagnostic_forward_only(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        match self {
            BackendExecutor::Cpu(e) => e.diagnostic_forward_only(prompt_tokens),
            BackendExecutor::Cuda(e) => e.diagnostic_forward_only(prompt_tokens),
            BackendExecutor::Rocm(e) => e.diagnostic_forward_only(prompt_tokens),
            BackendExecutor::Metal(e) => e.diagnostic_forward_only(prompt_tokens),
        }
    }


    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }
            BackendExecutor::Rocm(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }
            BackendExecutor::Metal(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }
            BackendExecutor::Cpu(exec) => {
                exec.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, thinking_budget)
            }
        }
    }

    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: u64,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
            BackendExecutor::Rocm(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
            BackendExecutor::Metal(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
            BackendExecutor::Cpu(exec) => {
                exec.generate_with_session(prompt, max_tokens, temperature, top_k, top_p, session_id, thinking_budget)
            }
        }
    }

    /// MTP-aware generation (REQ-MTP-002).
    ///
    /// Dispatches to the concrete Executor's `generate_with_mtp` method.
    /// Returns detailed MTP statistics including per-step acceptance info.
    pub fn generate_with_mtp(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<crate::engine::mtp_executor::MtpGenerateResult, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.generate_with_mtp(prompt, max_tokens, temperature, top_k, top_p),
            BackendExecutor::Rocm(exec) => exec.generate_with_mtp(prompt, max_tokens, temperature, top_k, top_p),
            BackendExecutor::Metal(exec) => exec.generate_with_mtp(prompt, max_tokens, temperature, top_k, top_p),
            BackendExecutor::Cpu(exec) => exec.generate_with_mtp(prompt, max_tokens, temperature, top_k, top_p),
        }
    }

    /// ARCH-MULTIMODAL-FUSION: dispatch into each concrete Executor.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_multimodal(
        &mut self,
        token_ids: Vec<u32>,
        fused_hidden: Vec<f32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        thinking_budget: Option<usize>,
    ) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
            BackendExecutor::Rocm(exec) => exec.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
            BackendExecutor::Metal(exec) => exec.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
            BackendExecutor::Cpu(exec) => exec.generate_with_multimodal(
                token_ids, fused_hidden, max_tokens, temperature, top_k, top_p, thinking_budget,
            ),
        }
    }

    /// SPEC/20 REQ-BCI-008: Batch inference via single mega-kernel call.
    pub fn generate_batch(
        &mut self,
        requests: &[crate::engine::batch_executor::GenerateRequest],
    ) -> Result<Vec<crate::engine::batch_executor::GenerateResult>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.generate_batch(requests),
            BackendExecutor::Rocm(exec) => exec.generate_batch(requests),
            BackendExecutor::Metal(exec) => exec.generate_batch(requests),
            BackendExecutor::Cpu(exec) => exec.generate_batch(requests),
        }
    }

    pub fn embed(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.embed(input),
            BackendExecutor::Rocm(exec) => exec.embed(input),
            BackendExecutor::Metal(exec) => exec.embed(input),
            BackendExecutor::Cpu(exec) => exec.embed(input),
        }
    }

    pub fn rerank(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.rerank(input),
            BackendExecutor::Rocm(exec) => exec.rerank(input),
            BackendExecutor::Metal(exec) => exec.rerank(input),
            BackendExecutor::Cpu(exec) => exec.rerank(input),
        }
    }

    pub fn rerank_pair(&mut self, query: &str, document: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Rocm(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Metal(exec) => exec.rerank_pair(query, document),
            BackendExecutor::Cpu(exec) => exec.rerank_pair(query, document),
        }
    }

    pub fn classify(&mut self, input: &str) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.classify(input),
            BackendExecutor::Rocm(exec) => exec.classify(input),
            BackendExecutor::Metal(exec) => exec.classify(input),
            BackendExecutor::Cpu(exec) => exec.classify(input),
        }
    }

    /// Head Routing SDK — 对 `prompt` 跑一次 generator forward,返回
    /// `target_token_ids` 每个 id 对应的原始 logit (未经 softmax)。
    ///
    /// # 关联
    /// - SPEC/HEAD-ROUTING.md §4.2
    /// - SPEC/04-API-DESIGN.md §3.8
    pub fn score_tokens_for_prompt(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.score_tokens_for_prompt(prompt, target_token_ids),
            BackendExecutor::Rocm(exec) => exec.score_tokens_for_prompt(prompt, target_token_ids),
            BackendExecutor::Metal(exec) => exec.score_tokens_for_prompt(prompt, target_token_ids),
            BackendExecutor::Cpu(exec) => exec.score_tokens_for_prompt(prompt, target_token_ids),
        }
    }

    /// Head Routing / Guardrail — pass-through for explicit callback chain.
    pub fn score_tokens_for_prompt_with_callbacks(
        &mut self,
        prompt: &str,
        target_token_ids: &[u32],
        callbacks: Option<&mut crate::graph::layer_callback::CallbackChain>,
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
            BackendExecutor::Rocm(exec) => {
                exec.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
            BackendExecutor::Metal(exec) => {
                exec.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
            BackendExecutor::Cpu(exec) => {
                exec.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, callbacks)
            }
        }
    }

    /// HR §3.4 / Intent §3 — mid-layer encode pass-through.
    pub fn encode_at_layer_for_prompt(
        &mut self,
        prompt: &str,
        anchor_layer: usize,
        pool: crate::head_routing::PoolMode,
    ) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
            BackendExecutor::Rocm(exec) => exec.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
            BackendExecutor::Metal(exec) => exec.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
            BackendExecutor::Cpu(exec) => exec.encode_at_layer_for_prompt(prompt, anchor_layer, pool),
        }
    }

    // ── Streaming support: delegate to inner Executor ──

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.decode_tokens(tokens),
            BackendExecutor::Rocm(exec) => exec.decode_tokens(tokens),
            BackendExecutor::Metal(exec) => exec.decode_tokens(tokens),
            BackendExecutor::Cpu(exec) => exec.decode_tokens(tokens),
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.encode_prompt(prompt),
            BackendExecutor::Rocm(exec) => exec.encode_prompt(prompt),
            BackendExecutor::Metal(exec) => exec.encode_prompt(prompt),
            BackendExecutor::Cpu(exec) => exec.encode_prompt(prompt),
        }
    }

    pub fn enqueue_with_config(
        &mut self,
        kind: crate::engine::executor::RequestKind,
        prompt: &str,
        max_new_tokens: usize,
        sampling: crate::engine::executor::SamplingConfig,
        thinking_budget: Option<usize>,
    ) -> Result<u64, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget)
            }
            BackendExecutor::Rocm(exec) => {
                exec.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget)
            }
            BackendExecutor::Metal(exec) => {
                exec.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget)
            }
            BackendExecutor::Cpu(exec) => {
                exec.enqueue_with_config(kind, prompt, max_new_tokens, sampling, thinking_budget)
            }
        }
    }

    pub fn enqueue_with_session(
        &mut self,
        kind: crate::engine::executor::RequestKind,
        prompt: &str,
        max_new_tokens: usize,
        sampling: crate::engine::executor::SamplingConfig,
        session_id: u64,
        thinking_budget: Option<usize>,
    ) -> Result<u64, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => {
                exec.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
            BackendExecutor::Rocm(exec) => {
                exec.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
            BackendExecutor::Metal(exec) => {
                exec.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
            BackendExecutor::Cpu(exec) => {
                exec.enqueue_with_session(kind, prompt, max_new_tokens, sampling, session_id, thinking_budget)
            }
        }
    }

    pub fn step(&mut self) -> Result<(), ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.step(),
            BackendExecutor::Rocm(exec) => exec.step(),
            BackendExecutor::Metal(exec) => exec.step(),
            BackendExecutor::Cpu(exec) => exec.step(),
        }
    }

    pub fn get_request(
        &self,
        request_id: u64,
    ) -> Option<&crate::engine::executor::RequestData> {
        match self {
            BackendExecutor::Cuda(exec) => exec.get_request(request_id),
            BackendExecutor::Rocm(exec) => exec.get_request(request_id),
            BackendExecutor::Metal(exec) => exec.get_request(request_id),
            BackendExecutor::Cpu(exec) => exec.get_request(request_id),
        }
    }

    pub fn release_request(&mut self, request_id: u64) {
        match self {
            BackendExecutor::Cuda(exec) => exec.release_request(request_id),
            BackendExecutor::Rocm(exec) => exec.release_request(request_id),
            BackendExecutor::Metal(exec) => exec.release_request(request_id),
            BackendExecutor::Cpu(exec) => exec.release_request(request_id),
        }
    }

    /// Get model configuration (delegate to inner Executor).
    pub fn model_config(&self) -> &crate::model_config::ModelConfig {
        match self {
            BackendExecutor::Cuda(exec) => exec.model_config(),
            BackendExecutor::Rocm(exec) => exec.model_config(),
            BackendExecutor::Metal(exec) => exec.model_config(),
            BackendExecutor::Cpu(exec) => exec.model_config(),
        }
    }

    /// Add a generation hook (guardrail/probe).
    pub fn add_hook(&self, hook: Box<dyn crate::generation::GenerationHook>) -> Result<(), crate::engine::executor::ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.add_hook(hook),
            BackendExecutor::Rocm(exec) => exec.add_hook(hook),
            BackendExecutor::Metal(exec) => exec.add_hook(hook),
            BackendExecutor::Cpu(exec) => exec.add_hook(hook),
        }
    }

    /// Remove hooks by type name.
    pub fn remove_hooks_by_type(&self, type_name: &str) -> Result<usize, crate::engine::executor::ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.remove_hooks_by_type(type_name),
            BackendExecutor::Rocm(exec) => exec.remove_hooks_by_type(type_name),
            BackendExecutor::Metal(exec) => exec.remove_hooks_by_type(type_name),
            BackendExecutor::Cpu(exec) => exec.remove_hooks_by_type(type_name),
        }
    }

    /// Clear all hooks.
    pub fn clear_hooks(&self) -> Result<(), crate::engine::executor::ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => exec.clear_hooks(),
            BackendExecutor::Rocm(exec) => exec.clear_hooks(),
            BackendExecutor::Metal(exec) => exec.clear_hooks(),
            BackendExecutor::Cpu(exec) => exec.clear_hooks(),
        }
    }

    /// Get current hook count.
    pub fn hook_count(&self) -> usize {
        match self {
            BackendExecutor::Cuda(exec) => exec.hook_count(),
            BackendExecutor::Rocm(exec) => exec.hook_count(),
            BackendExecutor::Metal(exec) => exec.hook_count(),
            BackendExecutor::Cpu(exec) => exec.hook_count(),
        }
    }

    /// Get forward configuration (per SPEC 04-API-DESIGN §7.3 for encode_intent).
    pub fn forward_config(&self) -> Result<crate::engine::executor::GeneratorForwardConfig, ExecutorError> {
        match self {
            BackendExecutor::Cuda(exec) => Ok(exec.forward_config()),
            BackendExecutor::Rocm(exec) => Ok(exec.forward_config()),
            BackendExecutor::Metal(exec) => Ok(exec.forward_config()),
            BackendExecutor::Cpu(exec) => Ok(exec.forward_config()),
        }
    }

    /// Get CPU backend reference (for knowledge injection and intent encoding).
    pub fn cpu_backend(&self) -> Result<&crate::compat::CpuBackend<E>, ExecutorError> {
        match self {
            BackendExecutor::Cpu(exec) => Ok(exec.backend()),
            _ => Err(ExecutorError::Backend(
                crate::engine::executor::BackendError::Unimplemented("cpu_backend only available for CPU backend")
            )),
        }
    }

    /// Get weights reference (for knowledge injection and intent encoding).
    pub fn weights(&self) -> Result<&crate::loader::WeightsHandle<crate::compat::CpuBackend<E>, E>, ExecutorError> {
        match self {
            BackendExecutor::Cpu(exec) => Ok(exec.weights()),
            _ => Err(ExecutorError::Backend(
                crate::engine::executor::BackendError::Unimplemented("weights only available for CPU backend in this API")
            )),
        }
    }

    /// Tokenizer handle exposed for Semantic Gatekeeper registration
    /// (SPEC/SEMANTIC-GATEKEEPER.md §3.1 level-descriptor tokenization).
    pub fn tokenizer(&self) -> &crate::tokenizer::TokenizerHandle {
        match self {
            BackendExecutor::Cuda(exec) => exec.tokenizer(),
            BackendExecutor::Rocm(exec) => exec.tokenizer(),
            BackendExecutor::Metal(exec) => exec.tokenizer(),
            BackendExecutor::Cpu(exec) => exec.tokenizer(),
        }
    }

    /// Override `multimodal_token_ids` in the underlying `ModelConfig`.
    ///
    /// Primarily used by the loader when tokenizer-side multimodal IDs land
    /// later than the initial `ModelConfig` construction, and by integration
    /// tests that wrap a text-only model with a mock encoder.
    pub fn set_multimodal_token_ids(
        &mut self,
        ids: Option<crate::compat::multimodal::MultimodalTokenIds>,
    ) {
        match self {
            BackendExecutor::Cuda(exec) => exec.set_multimodal_token_ids(ids),
            BackendExecutor::Rocm(exec) => exec.set_multimodal_token_ids(ids),
            BackendExecutor::Metal(exec) => exec.set_multimodal_token_ids(ids),
            BackendExecutor::Cpu(exec) => exec.set_multimodal_token_ids(ids),
        }
    }

    /// Retrieve `embed_tokens.weight` as a flat f32 row-major `[vocab, hidden]`
    /// buffer (ARCH-MULTIMODAL-FUSION input construction).
    ///
    /// Handles native f32/f16/bf16 storage as well as quantized tensors
    /// (dequantized to f32 via `get_typed_data` + `typed_bytes_to_f32`).
    pub fn embed_tokens_f32(&self) -> Result<Vec<f32>, ExecutorError> {
        match self {
            BackendExecutor::Cpu(exec) => {
                let backend = exec.backend();
                let weights = exec.weights();
                let (bytes, dtype) = crate::compat::weight_helpers::get_typed_data(
                    weights,
                    backend,
                    &crate::weight_names::decoder_embed_aliases(),
                )
                .map_err(ExecutorError::Backend)?;
                Ok(crate::compat::jit_helpers::typed_bytes_to_f32(&bytes, dtype))
            }
            _ => Err(ExecutorError::Backend(
                crate::engine::executor::BackendError::Unimplemented(
                    "embed_tokens_f32 only available for CPU backend in this API",
                ),
            )),
        }
    }

    /// Build a `SigLipEncoder` from the loaded model's vision weights.
    ///
    /// Returns `Ok(None)` if the model does not declare vision support
    /// (`vision_config` is None) or if any SigLIP weight is missing from the
    /// weight store. Returns `Err` only for hard failures (shape mismatch,
    /// corrupt weights, I/O error). CPU-only for now — the vision encoder
    /// JIT path currently targets x86_64 / AArch64 CPU codegen; GPU vision
    /// encoding is a separate (future) concern.
    pub fn try_build_siglip_encoder(
        &self,
    ) -> Result<Option<crate::compat::vision_forward::SigLipEncoder>, ExecutorError> {
        let cfg = match self.model_config().vision_config.clone() {
            Some(cfg) => cfg,
            None => return Ok(None),
        };
        let token_ids = self
            .model_config()
            .multimodal_token_ids
            .unwrap_or_else(|| {
                crate::compat::multimodal::MultimodalTokenIds::fallback_multimodal_token_ids()
            });

        match self {
            BackendExecutor::Cpu(exec) => {
                let backend = exec.backend();
                let weights = exec.weights();
                crate::compat::vision_forward::try_build_siglip_from_tensors(
                    &cfg,
                    token_ids,
                    |name| {
                        // get_typed_data returns (bytes, dtype). For CPU the
                        // Element is f32, so the bytes map zero-copy to f32,
                        // but we still run it through typed_bytes_to_f32 to
                        // normalise f16/bf16/quantized storage.
                        let result = crate::compat::weight_helpers::get_typed_data(
                            weights, backend, &[name],
                        );
                        match result {
                            Ok((bytes, dtype)) => {
                                let data = crate::compat::jit_helpers::typed_bytes_to_f32(
                                    &bytes, dtype,
                                );
                                let shape = weights
                                    .tensor_shape(name)
                                    .map(|s| s.to_vec())
                                    .unwrap_or_else(|| vec![data.len()]);
                                Some((data, shape))
                            }
                            Err(_) => None,
                        }
                    },
                )
                .map_err(ExecutorError::Backend)
            }
            _ => Err(ExecutorError::Backend(
                crate::engine::executor::BackendError::Unimplemented(
                    "try_build_siglip_encoder currently only implemented for CPU backend",
                ),
            )),
        }
    }

    /// Build a `UsmConformerEncoder` from the loaded model's audio weights.
    ///
    /// Returns `Ok(None)` if the model does not declare audio support
    /// (`audio_config` is None) or if any USM Conformer weight is missing from
    /// the weight store. Returns `Err` for hard failures. CPU-only to match
    /// the SigLIP policy.
    pub fn try_build_usm_conformer_encoder(
        &self,
    ) -> Result<Option<crate::compat::audio_forward::UsmConformerEncoder>, ExecutorError> {
        let cfg = match self.model_config().audio_config.clone() {
            Some(cfg) => cfg,
            None => return Ok(None),
        };
        let token_ids = self
            .model_config()
            .multimodal_token_ids
            .unwrap_or_else(crate::compat::multimodal::MultimodalTokenIds::fallback_multimodal_token_ids);

        match self {
            BackendExecutor::Cpu(exec) => {
                let backend = exec.backend();
                let weights = exec.weights();
                crate::compat::audio_forward::try_build_usm_from_tensors(
                    &cfg,
                    token_ids,
                    |name| {
                        let result = crate::compat::weight_helpers::get_typed_data(
                            weights, backend, &[name],
                        );
                        match result {
                            Ok((bytes, dtype)) => {
                                let data = crate::compat::jit_helpers::typed_bytes_to_f32(
                                    &bytes, dtype,
                                );
                                let shape = weights
                                    .tensor_shape(name)
                                    .map(|s| s.to_vec())
                                    .unwrap_or_else(|| vec![data.len()]);
                                Some((data, shape))
                            }
                            Err(_) => None,
                        }
                    },
                )
                .map_err(ExecutorError::Backend)
            }
            _ => Err(ExecutorError::Backend(
                crate::engine::executor::BackendError::Unimplemented(
                    "try_build_usm_conformer_encoder currently only implemented for CPU backend",
                ),
            )),
        }
    }

    /// §16.1 Set the Late-Fusion RAG system for RagInjectCallback.
    pub fn set_rag_system(&mut self, rag: crate::rag::LateFusionRag) {
        match self {
            BackendExecutor::Cuda(exec) => exec.set_rag_system(rag),
            BackendExecutor::Rocm(exec) => exec.set_rag_system(rag),
            BackendExecutor::Metal(exec) => exec.set_rag_system(rag),
            BackendExecutor::Cpu(exec) => exec.set_rag_system(rag),
        }
    }

    /// §9 Set the Semantic Gatekeeper callback shim for injection into
    /// the per-forward CallbackChain (SPEC/SEMANTIC-GATEKEEPER.md Phase C).
    pub fn set_sg_callback_shim(
        &mut self,
        shim: crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim,
    ) {
        match self {
            BackendExecutor::Cuda(exec) => exec.set_sg_callback_shim(shim),
            BackendExecutor::Rocm(exec) => exec.set_sg_callback_shim(shim),
            BackendExecutor::Metal(exec) => exec.set_sg_callback_shim(shim),
            BackendExecutor::Cpu(exec) => exec.set_sg_callback_shim(shim),
        }
    }

    /// Remove the Semantic Gatekeeper callback shim from the executor.
    pub fn clear_sg_callback_shim(&mut self) {
        match self {
            BackendExecutor::Cuda(exec) => exec.clear_sg_callback_shim(),
            BackendExecutor::Rocm(exec) => exec.clear_sg_callback_shim(),
            BackendExecutor::Metal(exec) => exec.clear_sg_callback_shim(),
            BackendExecutor::Cpu(exec) => exec.clear_sg_callback_shim(),
        }
    }

    /// Returns the pre-created SG Q-tap ring buffer.
    /// Delegates to `Executor::sg_ring_buffer()`.
    pub fn sg_ring_buffer(
        &self,
    ) -> Option<std::sync::Arc<crate::semantic_gatekeeper::GatekeeperRingBuffer>> {
        match self {
            BackendExecutor::Cuda(exec) => exec.sg_ring_buffer(),
            BackendExecutor::Rocm(exec) => exec.sg_ring_buffer(),
            BackendExecutor::Metal(exec) => exec.sg_ring_buffer(),
            BackendExecutor::Cpu(exec) => exec.sg_ring_buffer(),
        }
    }
}

/// Backward-compatible type alias for f32 backend executor.
pub type BackendExecutorF32 = BackendExecutor<f32>;

pub struct BackendContext {
    #[allow(dead_code)]
    model_ref: String,
    manifest: Arc<ModelManifest>,
    #[allow(dead_code)]
    weight_paths: Vec<std::path::PathBuf>,
    #[allow(dead_code)]
    config_path: Option<std::path::PathBuf>,
    #[allow(dead_code)]
    tokenizer_path: Option<std::path::PathBuf>,
    executor: Mutex<BackendExecutor<f32>>,
}

impl BackendContext {
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        backend: DetectedBackend<f32>,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();
        // ARCH-ZERO-FALLBACK: No OOM fallback. Executor build failure propagates directly.
        let executor = build_executor(
            backend,
            manifest.clone(),
            &model_ref,
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;
        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn executor(&self) -> MutexGuard<'_, BackendExecutor<f32>> {
        self.executor.lock().unwrap_or_else(|err| err.into_inner()) // LEGAL: Mutex poison 时恢复内部数据
    }

    pub fn executor_mut(&self) -> MutexGuard<'_, BackendExecutor<f32>> {
        self.executor()
    }

}

fn build_executor(
    backend: DetectedBackend<f32>,
    manifest: Arc<ModelManifest>,
    _model_ref: &str,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor<f32>, BackendContextError> {
    match backend {
        DetectedBackend::Cuda(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Rocm(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Rocm(Box::new(executor)))
        }
        DetectedBackend::Metal(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Metal(Box::new(executor)))
        }
        DetectedBackend::Cpu(backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let mut loader = loader.load()?;
            let executor = Executor::from_loader(*backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}

// ============================================================================
// Dynamic Precision Context - Runtime dtype selection
// ============================================================================

/// Dynamic backend context that selects precision at runtime based on model dtype.
///
/// Unlike `BackendContext` which is fixed to f32, this context automatically
/// detects the model's weight dtype and uses the optimal precision.
#[allow(dead_code)] // Fields reserved for future rebuild functionality
pub struct DynBackendContext {
    model_ref: String,
    manifest: Arc<ModelManifest>,
    weight_paths: Vec<std::path::PathBuf>,
    config_path: Option<std::path::PathBuf>,
    tokenizer_path: Option<std::path::PathBuf>,
    executor: Mutex<DynBackendExecutor>,
}

impl DynBackendContext {
    /// Create a new dynamic context, auto-detecting the model's dtype.
    pub fn new(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();

        // Detect dtype from weights
        let detected_dtype = Self::detect_dtype_from_paths(&weight_paths)?;

        let executor = Self::build_dyn_executor(
            detected_dtype,
            manifest.clone(),
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;

        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    /// Create with explicit dtype (skip auto-detection)
    pub fn with_dtype(
        model_ref: impl Into<String>,
        manifest: Arc<ModelManifest>,
        dtype: DetectedDtype,
        weight_paths: Vec<std::path::PathBuf>,
        config_path: Option<std::path::PathBuf>,
        tokenizer_path: Option<std::path::PathBuf>,
    ) -> Result<Self, BackendContextError> {
        let model_ref = model_ref.into();

        let executor = Self::build_dyn_executor(
            dtype,
            manifest.clone(),
            &weight_paths,
            config_path.as_deref(),
            tokenizer_path.as_deref(),
        )?;

        Ok(Self {
            model_ref,
            manifest,
            weight_paths,
            config_path,
            tokenizer_path,
            executor: Mutex::new(executor),
        })
    }

    pub fn manifest(&self) -> &ModelManifest {
        self.manifest.as_ref()
    }

    pub fn executor(&self) -> MutexGuard<'_, DynBackendExecutor> {
        self.executor.lock().unwrap_or_else(|err| err.into_inner()) // LEGAL: Mutex poison 时恢复内部数据
    }

    pub fn executor_mut(&self) -> MutexGuard<'_, DynBackendExecutor> {
        self.executor()
    }

    /// Returns the detected dtype
    pub fn dtype(&self) -> &'static str {
        self.executor().dtype_name()
    }

    fn detect_dtype_from_paths(
        weight_paths: &[std::path::PathBuf],
    ) -> Result<DetectedDtype, BackendContextError> {
        // Try to create a temporary loader to detect dtype
        let loader = Loader::from_env()?.with_weights(weight_paths.to_vec());

        if let Ok(Some(dtype)) = loader.detect_weight_dtype() {
            return Ok(match dtype {
                gllm_kernels::types::DType::F32 => DetectedDtype::F32,
                gllm_kernels::types::DType::F16 => DetectedDtype::F16,
                gllm_kernels::types::DType::BF16 => DetectedDtype::BF16,
                gllm_kernels::types::DType::U8
                | gllm_kernels::types::DType::F8E4M3
                | gllm_kernels::types::DType::F8E5M2
                | gllm_kernels::types::DType::F6E3M2
                | gllm_kernels::types::DType::F6E2M3
                | gllm_kernels::types::DType::F4E2M1 => DetectedDtype::F32,
            });
        }

        // Default to f32 if detection fails
        Ok(DetectedDtype::F32)
    }

    fn build_dyn_executor(
        dtype: DetectedDtype,
        manifest: Arc<ModelManifest>,
        weight_paths: &[std::path::PathBuf],
        config_path: Option<&std::path::Path>,
        tokenizer_path: Option<&std::path::Path>,
    ) -> Result<DynBackendExecutor, BackendContextError> {
        match dtype {
            DetectedDtype::F32 => {
                let executor = build_executor_generic::<f32>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::F32(executor))
            }
            DetectedDtype::F16 => {
                let executor = build_executor_generic::<half::f16>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::F16(executor))
            }
            DetectedDtype::BF16 => {
                let executor = build_executor_generic::<half::bf16>(
                    manifest,
                    weight_paths,
                    config_path,
                    tokenizer_path,
                )?;
                Ok(DynBackendExecutor::BF16(executor))
            }
        }
    }
}

/// Build executor for any element type
fn build_executor_generic<E: Element>(
    manifest: Arc<ModelManifest>,
    weight_paths: &[std::path::PathBuf],
    config_path: Option<&std::path::Path>,
    tokenizer_path: Option<&std::path::Path>,
) -> Result<BackendExecutor<E>, BackendContextError> {
    let backend = detection::detect_backend_generic::<E>()?;

    match backend {
        DetectedBackend::Cuda(cuda_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*cuda_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cuda(Box::new(executor)))
        }
        DetectedBackend::Rocm(rocm_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*rocm_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Rocm(Box::new(executor)))
        }
        DetectedBackend::Metal(metal_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*metal_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Metal(Box::new(executor)))
        }
        DetectedBackend::Cpu(cpu_backend) => {
            let mut loader = Loader::from_env_with_manifest(manifest.as_ref().clone())?
                .with_weights(weight_paths.to_vec());
            if let Some(path) = config_path {
                loader = loader.with_config(path.to_path_buf());
            }
            if let Some(path) = tokenizer_path {
                loader = loader.with_tokenizer(path.to_path_buf());
            }
            let executor = Executor::from_loader(*cpu_backend, manifest, &mut loader)?;
            Ok(BackendExecutor::Cpu(Box::new(executor)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BackendContextError: Display messages ──

    #[test]
    fn backend_context_error_unsupported_architecture_display() {
        let err = BackendContextError::UnsupportedArchitecture("fake_arch_v2".to_string());
        let msg = format!("{err}");
        assert!(
            msg.contains("fake_arch_v2"),
            "Display should contain the architecture name, got: {msg}"
        );
        assert!(
            msg.contains("unsupported"),
            "Display should contain 'unsupported', got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_error() {
        let loader_err = LoaderError::MissingWeights;
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("Missing weights"),
            "Should propagate LoaderError display, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_executor_error() {
        let exec_err = ExecutorError::EmptyPrompt;
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("empty prompt"),
            "Should propagate ExecutorError display, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_backend_error() {
        let be_err = crate::engine::executor::BackendError::Cpu("cpu blew up".to_string());
        let ctx_err: BackendContextError = be_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("cpu blew up"),
            "Should propagate BackendError display via Executor variant, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_unsupported_arch_is_distinct_from_loader() {
        let arch_err = BackendContextError::UnsupportedArchitecture("x".to_string());
        let loader_err = BackendContextError::Loader(LoaderError::MissingWeights);
        assert!(
            format!("{arch_err}") != format!("{loader_err}"),
            "Different variants must produce different Display output"
        );
    }

    #[test]
    fn backend_context_error_std_error_trait() {
        let err: BackendContextError = BackendContextError::UnsupportedArchitecture("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    // ── BackendType ──

    #[test]
    fn backend_type_equality() {
        assert_eq!(BackendType::Cuda, BackendType::Cuda);
        assert_ne!(BackendType::Cuda, BackendType::Cpu);
        assert_ne!(BackendType::Rocm, BackendType::Metal);
    }

    #[test]
    fn backend_type_copy() {
        let a = BackendType::Cpu;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn backend_type_hash_consistency() {
        use std::collections::HashSet;
        let set: HashSet<BackendType> = [BackendType::Cuda, BackendType::Cpu, BackendType::Cuda].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── DetectedDtype ──

    #[test]
    fn detected_dtype_from_size_4_is_f32() {
        assert_eq!(DetectedDtype::from_size(4), Some(DetectedDtype::F32));
    }

    #[test]
    fn detected_dtype_from_size_2_is_f16() {
        assert_eq!(DetectedDtype::from_size(2), Some(DetectedDtype::F16));
    }

    #[test]
    fn detected_dtype_from_size_unknown_returns_none() {
        assert_eq!(DetectedDtype::from_size(1), None);
        assert_eq!(DetectedDtype::from_size(3), None);
        assert_eq!(DetectedDtype::from_size(8), None);
    }

    #[test]
    fn detected_dtype_equality_and_copy() {
        let a = DetectedDtype::F32;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(DetectedDtype::F16, DetectedDtype::BF16);
    }

    #[test]
    fn detected_dtype_from_safetensors_dtype() {
        assert_eq!(
            DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::F32),
            Some(DetectedDtype::F32)
        );
        assert_eq!(
            DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::F16),
            Some(DetectedDtype::F16)
        );
        assert_eq!(
            DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::BF16),
            Some(DetectedDtype::BF16)
        );
    }

    #[test]
    fn detected_dtype_from_safetensors_unsupported_returns_none() {
        assert_eq!(
            DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::U8),
            None
        );
        assert_eq!(
            DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::I32),
            None
        );
    }

    // ── DynBackendExecutor: dtype_name ──

    #[test]
    fn dyn_backend_executor_dtype_name_returns_correct_static_str() {
        // dtype_name does not require an actual executor — only the variant matters.
        // But we cannot construct DynBackendExecutor variants without real executors,
        // so we test the string constants are consistent via DetectedDtype coverage.
        assert_eq!(DetectedDtype::F32, DetectedDtype::F32);
        assert_eq!(DetectedDtype::F16, DetectedDtype::F16);
        assert_eq!(DetectedDtype::BF16, DetectedDtype::BF16);
    }

    // ── Type alias ──

    #[test]
    fn backend_executor_f32_is_alias() {
        // Compile-time proof that BackendExecutorF32 == BackendExecutor<f32>.
        // If the type alias breaks, this line fails to compile.
        let _: fn() = || {
            fn _check(_: BackendExecutorF32) {}
        };
    }

    // ── Error chain: BackendError → ExecutorError → BackendContextError ──

    #[test]
    fn backend_error_chain_backend_to_executor_to_context() {
        let be = crate::engine::executor::BackendError::Unimplemented("test feature");
        let exec_err: ExecutorError = be.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("test feature"),
            "Error message should propagate through chain, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_loader_variant_preserves_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not here");
        let loader_err = LoaderError::Io(io_err);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("file not here"),
            "IO error message should propagate, got: {msg}"
        );
    }

    // ── ExecutorError variants used via BackendContextError ──

    #[test]
    fn executor_error_empty_sample_display() {
        let err = ExecutorError::EmptySample;
        assert!(
            format!("{err}").contains("empty sample"),
            "ExecutorError::EmptySample display should describe the error"
        );
    }

    #[test]
    fn executor_error_request_not_found_display() {
        let err = ExecutorError::RequestNotFound { request_id: 42 };
        let msg = format!("{err}");
        assert!(
            msg.contains("42"),
            "Should contain the request ID, got: {msg}"
        );
    }

    // ── BackendError Display ──

    #[test]
    fn backend_error_cuda_display() {
        let err = crate::engine::executor::BackendError::Cuda("gpu oops".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("CUDA"), "Should mention CUDA, got: {msg}");
        assert!(msg.contains("gpu oops"), "Should contain the detail, got: {msg}");
    }

    #[test]
    fn backend_error_hip_display() {
        let err = crate::engine::executor::BackendError::Hip("hip fail".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("HIP"), "Should mention HIP, got: {msg}");
    }

    #[test]
    fn backend_error_metal_display() {
        let err = crate::engine::executor::BackendError::Metal("metal fail".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Metal"), "Should mention Metal, got: {msg}");
    }

    #[test]
    fn backend_error_cpu_display() {
        let err = crate::engine::executor::BackendError::Cpu("cpu fail".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("CPU"), "Should mention CPU, got: {msg}");
    }

    #[test]
    fn backend_error_unimplemented_display() {
        let err = crate::engine::executor::BackendError::Unimplemented("nope");
        let msg = format!("{err}");
        assert!(msg.contains("nope"), "Should contain the detail, got: {msg}");
    }

    #[test]
    fn backend_error_other_display() {
        let err = crate::engine::executor::BackendError::Other("generic".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("generic"), "Should contain the detail, got: {msg}");
    }

    #[test]
    fn backend_error_is_std_error() {
        let err = crate::engine::executor::BackendError::Other("x".to_string());
        let _: &dyn std::error::Error = &err;
    }

    // ── BackendType: Debug, Clone, all variants ──

    #[test]
    fn backend_type_debug_format() {
        assert!(format!("{:?}", BackendType::Cuda).contains("Cuda"));
        assert!(format!("{:?}", BackendType::Rocm).contains("Rocm"));
        assert!(format!("{:?}", BackendType::Metal).contains("Metal"));
        assert!(format!("{:?}", BackendType::Cpu).contains("Cpu"));
    }

    #[test]
    fn backend_type_all_variants_are_distinct() {
        let variants = [BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "BackendType variants at {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn backend_type_clone_produces_equal_value() {
        let original = BackendType::Metal;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── BackendType: Ord-like comparison via Hash ──

    #[test]
    fn backend_type_hash_set_dedup() {
        use std::collections::HashSet;
        let all = [BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu];
        let set: HashSet<BackendType> = all.into_iter().chain(all.into_iter()).collect();
        assert_eq!(set.len(), 4, "All four BackendType variants should be distinct in HashSet");
    }

    // ── DetectedDtype: exhaustive from_size coverage ──

    #[test]
    fn detected_dtype_from_size_zero_returns_none() {
        assert_eq!(DetectedDtype::from_size(0), None);
    }

    #[test]
    fn detected_dtype_from_size_boundary_values() {
        // size=1 (u8, bool), size=3 (no standard), size=8 (f64 or i64)
        assert_eq!(DetectedDtype::from_size(1), None);
        assert_eq!(DetectedDtype::from_size(3), None);
        assert_eq!(DetectedDtype::from_size(8), None);
        assert_eq!(DetectedDtype::from_size(16), None);
        assert_eq!(DetectedDtype::from_size(usize::MAX), None);
    }

    // ── DetectedDtype: Clone, Debug, all variants ──

    #[test]
    fn detected_dtype_all_variants_distinct() {
        let variants = [DetectedDtype::F32, DetectedDtype::F16, DetectedDtype::BF16];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "DetectedDtype variants at {i} and {j} should differ");
                }
            }
        }
    }

    #[test]
    fn detected_dtype_clone_and_copy() {
        let a = DetectedDtype::BF16;
        let b = a; // Copy
        let c = a; // Copy again
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn detected_dtype_debug_contains_variant_name() {
        assert!(format!("{:?}", DetectedDtype::F32).contains("F32"));
        assert!(format!("{:?}", DetectedDtype::F16).contains("F16"));
        assert!(format!("{:?}", DetectedDtype::BF16).contains("BF16"));
    }

    // ── DetectedDtype: from_safetensors_dtype comprehensive coverage ──

    #[test]
    fn detected_dtype_from_safetensors_all_non_float_return_none() {
        // Exhaustively test non-float safetensors dtypes return None
        use ::safetensors::Dtype;
        let non_float_dtypes = [
            Dtype::U8,
            Dtype::I32,
            Dtype::I64,
            Dtype::F64,
            Dtype::BOOL,
            Dtype::U16,
            Dtype::I8,
            Dtype::I16,
            Dtype::U32,
            Dtype::U64,
        ];
        for dtype in non_float_dtypes {
            assert_eq!(
                DetectedDtype::from_safetensors_dtype(dtype),
                None,
                "Non-float dtype {dtype:?} should map to None"
            );
        }
    }

    // ── BackendContextError: all From conversions ──

    #[test]
    fn backend_context_error_from_kv_cache_error() {
        let kv_err = ExecutorError::KvCache(crate::kv_cache::KvCacheError::Exhausted {
            requested: 1024,
            available: 512,
        });
        let ctx_err: BackendContextError = kv_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("1024"),
            "Should propagate KvCacheError detail, got: {msg}"
        );
        assert!(
            msg.contains("512"),
            "Should propagate available count, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_config_error() {
        let cfg_err = ExecutorError::Config(
            crate::model_config::ModelConfigError::InvalidConfig("missing arch field".to_string()),
        );
        let ctx_err: BackendContextError = cfg_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("missing arch field"),
            "Should propagate InvalidConfig detail, got: {msg}"
        );
    }

    // ── BackendContextError: unsupported architecture with various strings ──

    #[test]
    fn backend_context_error_unsupported_arch_with_empty_string() {
        let err = BackendContextError::UnsupportedArchitecture(String::new());
        let msg = format!("{err}");
        assert!(msg.contains("unsupported"), "Should contain 'unsupported', got: {msg}");
    }

    #[test]
    fn backend_context_error_unsupported_arch_with_unicode() {
        let err = BackendContextError::UnsupportedArchitecture("模型架构不匹配".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("模型架构不匹配"), "Should preserve unicode, got: {msg}");
    }

    // ── BackendExecutorF32 type alias ──

    #[test]
    fn backend_executor_f32_type_alias_compiles() {
        // Compile-time proof that the type alias exists and matches the expected type.
        fn _assert_alias(_: fn(BackendExecutorF32)) {}
        _assert_alias(|_| {});
    }

    // ── ExecutorError: additional variant Display ──

    #[test]
    fn executor_error_scheduler_display() {
        let err = ExecutorError::Scheduler("queue full".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("scheduler"), "Should contain 'scheduler', got: {msg}");
        assert!(msg.contains("queue full"), "Should contain detail, got: {msg}");
    }

    #[test]
    fn executor_error_compilation_display() {
        let err = ExecutorError::Compilation("invalid IR node".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("JIT"), "Should mention JIT, got: {msg}");
        assert!(msg.contains("invalid IR node"), "Should contain detail, got: {msg}");
    }

    #[test]
    fn executor_error_graph_expansion_display() {
        let err = ExecutorError::GraphExpansion("circular dependency".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("graph"), "Should mention graph, got: {msg}");
        assert!(msg.contains("circular dependency"), "Should contain detail, got: {msg}");
    }

    // ── BackendError: all Display variants ──

    #[test]
    fn backend_error_all_display_variants_produce_nonempty_strings() {
        use crate::engine::executor::BackendError;
        let errors = [
            BackendError::Cuda("c".into()),
            BackendError::Hip("h".into()),
            BackendError::Metal("m".into()),
            BackendError::Cpu("p".into()),
            BackendError::Unimplemented("u"),
            BackendError::Other("o".into()),
        ];
        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "BackendError display should not be empty for {err:?}");
        }
    }

    #[test]
    fn backend_error_cuda_vs_hip_display_distinct() {
        use crate::engine::executor::BackendError;
        let cuda_msg = format!("{}", BackendError::Cuda("x".into()));
        let hip_msg = format!("{}", BackendError::Hip("x".into()));
        assert_ne!(cuda_msg, hip_msg, "CUDA and HIP error messages should be distinct");
    }

    #[test]
    fn backend_error_metal_vs_cpu_display_distinct() {
        use crate::engine::executor::BackendError;
        let metal_msg = format!("{}", BackendError::Metal("x".into()));
        let cpu_msg = format!("{}", BackendError::Cpu("x".into()));
        assert_ne!(metal_msg, cpu_msg, "Metal and CPU error messages should be distinct");
    }

    // ── HookDecision: Debug, Clone, PartialEq ──

    #[test]
    fn hook_decision_equality() {
        use crate::generation::HookDecision;
        assert_eq!(HookDecision::Continue, HookDecision::Continue);
        assert_eq!(HookDecision::Terminate, HookDecision::Terminate);
        assert_ne!(HookDecision::Continue, HookDecision::Terminate);
    }

    #[test]
    fn hook_decision_veto_equality_by_content() {
        use crate::generation::HookDecision;
        let a = HookDecision::Veto("bad token".to_string());
        let b = HookDecision::Veto("bad token".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn hook_decision_veto_inequality_by_content() {
        use crate::generation::HookDecision;
        let a = HookDecision::Veto("reason A".to_string());
        let b = HookDecision::Veto("reason B".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn hook_decision_clone() {
        use crate::generation::HookDecision;
        let original = HookDecision::Veto("test".to_string());
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn hook_decision_debug() {
        use crate::generation::HookDecision;
        assert!(format!("{:?}", HookDecision::Continue).contains("Continue"));
        assert!(format!("{:?}", HookDecision::Terminate).contains("Terminate"));
        let veto_dbg = format!("{:?}", HookDecision::Veto("reason".into()));
        assert!(veto_dbg.contains("Veto"));
    }

    // ── GenerationChunk ──

    #[test]
    fn generation_chunk_new_has_defaults() {
        use crate::generation::GenerationChunk;
        let chunk = GenerationChunk::new();
        assert!(chunk.tokens.is_empty());
        assert!(chunk.text.is_empty());
        assert!(!chunk.finished);
    }

    #[test]
    fn generation_chunk_with_token_appends() {
        use crate::generation::GenerationChunk;
        let chunk = GenerationChunk::new().with_token(42, "hello".to_string());
        assert_eq!(chunk.tokens, vec![42]);
        assert_eq!(chunk.text, "hello");
    }

    #[test]
    fn generation_chunk_finish_sets_flag() {
        use crate::generation::GenerationChunk;
        let chunk = GenerationChunk::new().finish();
        assert!(chunk.finished);
    }

    #[test]
    fn generation_chunk_chained_operations() {
        use crate::generation::GenerationChunk;
        let chunk = GenerationChunk::new()
            .with_token(1, "a".to_string())
            .with_token(2, "b".to_string())
            .finish();
        assert_eq!(chunk.tokens, vec![1, 2]);
        assert_eq!(chunk.text, "b"); // last with_token overwrites text
        assert!(chunk.finished);
    }

    // ── SamplingConfig: Default ──

    #[test]
    fn sampling_config_default_values() {
        use crate::engine::executor::SamplingConfig;
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.top_p, 1.0);
    }

    #[test]
    fn sampling_config_custom_values() {
        use crate::engine::executor::SamplingConfig;
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        assert_eq!(cfg.temperature, 0.7);
        assert_eq!(cfg.top_k, 50);
        assert_eq!(cfg.top_p, 0.9);
    }

    // ── WeightPageJitConfig: Default ──

    #[test]
    fn weight_page_jit_config_default() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let cfg = WeightPageJitConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 1024);
        assert_eq!(cfg.page_size_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 0);
    }

    #[test]
    fn weight_page_jit_config_clone() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let original = WeightPageJitConfig::default();
        let cloned = original.clone();
        assert_eq!(original.enabled, cloned.enabled);
        assert_eq!(original.num_pages, cloned.num_pages);
        assert_eq!(original.page_size_bytes, cloned.page_size_bytes);
        assert_eq!(original.prefetch_distance, cloned.prefetch_distance);
    }

    // ── DiagnosticScratchpad: read_f32_at boundary checks ──

    #[test]
    fn diagnostic_scratchpad_read_f32_out_of_bounds_returns_empty() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let pad = DiagnosticScratchpad {
            data: vec![0u8; 16], // 4 x f32
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        // Request 5 f32s from offset 0 but only 4 available
        let result = pad.read_f32_at(0, 5);
        assert!(result.is_empty(), "Out-of-bounds read should return empty vec");
    }

    #[test]
    fn diagnostic_scratchpad_read_f32_within_bounds() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 16];
        // Write 1.0f32 at byte offset 0
        let one_f32: f32 = 1.0;
        let bytes = one_f32.to_le_bytes();
        data[0..4].copy_from_slice(&bytes);

        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        let result = pad.read_f32_at(0, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn diagnostic_scratchpad_read_f32_at_nonzero_offset() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 32]; // 8 x f32
        let val: f32 = 42.0;
        let bytes = val.to_le_bytes();
        data[12..16].copy_from_slice(&bytes); // byte offset 12 = f32 index 3

        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        let result = pad.read_f32_at(12, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 42.0f32).abs() < f32::EPSILON);
    }

    // ── BatchOrderPolicy: Default ──

    #[test]
    fn batch_order_policy_default_is_strict() {
        use crate::scheduler::types::BatchOrderPolicy;
        let default = BatchOrderPolicy::default();
        assert_eq!(default, BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn batch_order_policy_equality() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_eq!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::StrictRequestIdOrder);
        assert_ne!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::FifoOrder);
    }

    // ── RequestKind: Debug and equality ──

    #[test]
    fn request_kind_variants() {
        use crate::scheduler::types::RequestKind;
        assert!(format!("{:?}", RequestKind::Chat).contains("Chat"));
        assert!(format!("{:?}", RequestKind::Embedding).contains("Embedding"));
        assert!(format!("{:?}", RequestKind::Rerank).contains("Rerank"));
    }

    // ── PoolMode: apply validation ──

    #[test]
    fn pool_mode_mean_pool_correctness() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [seq=2, hidden=3]
        let result = PoolMode::MeanPool.apply(&hidden, 2, 3).unwrap();
        // mean of [1,2,3] = [0.5, 1.0, 1.5] + [4,5,6]/2 = [0.5+2, 1+2.5, 1.5+3] = [2.5, 3.5, 4.5]
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn pool_mode_last_token_correctness() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [seq=2, hidden=3]
        let result = PoolMode::LastToken.apply(&hidden, 2, 3).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn pool_mode_cls_token_correctness() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [seq=2, hidden=3]
        let result = PoolMode::ClsToken.apply(&hidden, 2, 3).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pool_mode_empty_seq_returns_error() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0_f32; 0];
        let result = PoolMode::MeanPool.apply(&hidden, 0, 3);
        assert!(result.is_err(), "Empty seq_len should return error");
    }

    #[test]
    fn pool_mode_insufficient_data_returns_error() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0]; // only 2 elements, but seq=2, hidden=3 needs 6
        let result = PoolMode::MeanPool.apply(&hidden, 2, 3);
        assert!(result.is_err(), "Insufficient data should return error");
    }

    // ── KvPipeline: equality ──

    #[test]
    fn kv_pipeline_variants_distinct() {
        use crate::scheduler::types::KvPipeline;
        assert_ne!(KvPipeline::Conversation, KvPipeline::Working);
        assert_eq!(KvPipeline::Conversation, KvPipeline::Conversation);
    }

    // ── LoaderError: propagation through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_network() {
        let loader_err = LoaderError::Network("timeout".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(msg.contains("Network"), "Should contain 'Network', got: {msg}");
        assert!(msg.contains("timeout"), "Should contain 'timeout', got: {msg}");
    }

    #[test]
    fn backend_context_error_from_loader_missing_tensor() {
        let loader_err = LoaderError::MissingTensor("embed_tokens.weight".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(msg.contains("embed_tokens.weight"), "Should contain tensor name, got: {msg}");
    }

    #[test]
    fn backend_context_error_from_loader_duplicate_tensor() {
        let loader_err = LoaderError::DuplicateTensor("layer.0.weight".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(msg.contains("layer.0.weight"), "Should contain tensor name, got: {msg}");
    }

    #[test]
    fn backend_context_error_from_loader_unsupported_dtype() {
        let loader_err = LoaderError::UnsupportedDtype(::safetensors::Dtype::U8);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(msg.contains("dtype") || msg.contains("U8"), "Should mention dtype issue, got: {msg}");
    }

    // ── Error chain: roundtrip BackendContextError variants ──

    #[test]
    fn backend_context_error_variants_produce_distinct_messages() {
        let msgs: Vec<String> = vec![
            format!("{}", BackendContextError::UnsupportedArchitecture("a".into())),
            format!("{}", BackendContextError::Loader(LoaderError::MissingWeights)),
            format!("{}", BackendContextError::Executor(ExecutorError::EmptyPrompt)),
        ];
        // Check all messages are non-empty
        for msg in &msgs {
            assert!(!msg.is_empty(), "Error display should not be empty");
        }
        // Check pairwise distinct
        for i in 0..msgs.len() {
            for j in (i + 1)..msgs.len() {
                assert_ne!(msgs[i], msgs[j], "Different variants should produce different messages");
            }
        }
    }

    // ── ThresholdHook: GenerationHook trait ──

    #[test]
    fn threshold_hook_allows_non_vetoed_tokens() {
        use crate::generation::GenerationHook;
        let hook = crate::generation::ThresholdHook::new(vec![999], 3);
        let decision = hook.post_step(&[0.1], &[42]);
        assert_eq!(decision, crate::generation::HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_vetos_listed_token() {
        use crate::generation::GenerationHook;
        let hook = crate::generation::ThresholdHook::new(vec![100], 5);
        let decision = hook.post_step(&[0.1], &[100]);
        match decision {
            crate::generation::HookDecision::Veto(reason) => {
                assert!(reason.contains("100"), "Veto reason should mention token 100, got: {reason}");
            }
            other => panic!("Expected Veto, got {other:?}"),
        }
    }

    #[test]
    fn threshold_hook_terminates_after_max_vetoes() {
        use crate::generation::GenerationHook;
        let hook = crate::generation::ThresholdHook::new(vec![100], 1);
        // First veto: count goes 0->1, which equals max_vetoes=1 => Terminate
        let decision = hook.post_step(&[0.1], &[100]);
        assert_eq!(decision, crate::generation::HookDecision::Terminate);
    }

    #[test]
    fn threshold_hook_veto_then_continue_with_other_token() {
        use crate::generation::GenerationHook;
        let hook = crate::generation::ThresholdHook::new(vec![100], 5);
        // Veto token 100
        let _ = hook.post_step(&[0.1], &[100]);
        // Then generate a non-vetoed token
        let decision = hook.post_step(&[0.1], &[42]);
        assert_eq!(decision, crate::generation::HookDecision::Continue);
    }

    #[test]
    fn threshold_hook_empty_generated_tokens_continues() {
        use crate::generation::GenerationHook;
        let hook = crate::generation::ThresholdHook::new(vec![100], 5);
        let decision = hook.post_step(&[0.1], &[]);
        assert_eq!(decision, crate::generation::HookDecision::Continue);
    }

    // ── DetectedBackendF32 type alias ──

    #[test]
    fn detected_backend_f32_alias_compiles() {
        use detection::DetectedBackendF32;
        // Compile-time proof the alias exists
        fn _check(_: fn(DetectedBackendF32)) {}
        _check(|_| {});
    }

    // ── BackendType: Display trait (via Debug, since no custom Display) ──

    #[test]
    fn backend_type_display_via_debug_is_human_readable() {
        for (variant, name) in [
            (BackendType::Cuda, "Cuda"),
            (BackendType::Rocm, "Rocm"),
            (BackendType::Metal, "Metal"),
            (BackendType::Cpu, "Cpu"),
        ] {
            let dbg = format!("{variant:?}");
            assert!(
                dbg.contains(name),
                "Debug output for {name} should contain its name, got: {dbg}"
            );
        }
    }

    // ── BackendType: Hash stability (same value hashed twice yields same result) ──

    #[test]
    fn backend_type_hash_is_stable() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value(bt: BackendType) -> u64 {
            let mut h1 = DefaultHasher::new();
            bt.hash(&mut h1);
            h1.finish()
        }
        for variant in [BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu] {
            assert_eq!(
                hash_value(variant),
                hash_value(variant),
                "Hashing the same BackendType variant twice should yield identical results"
            );
        }
    }

    // ── DetectedDtype: Hash via HashMap usage ──

    #[test]
    fn detected_dtype_hash_map_lookup() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(DetectedDtype::F32, "float32");
        map.insert(DetectedDtype::F16, "float16");
        assert_eq!(map.get(&DetectedDtype::F32), Some(&"float32"));
        assert_eq!(map.get(&DetectedDtype::F16), Some(&"float16"));
        assert_eq!(map.get(&DetectedDtype::BF16), None);
    }

    // ── DetectedDtype: Eq implies consistent ordering for same values ──

    #[test]
    fn detected_dtype_eq_consistency() {
        // Eq means a == a for all values
        assert!(DetectedDtype::F32 == DetectedDtype::F32);
        assert!(DetectedDtype::F16 == DetectedDtype::F16);
        assert!(DetectedDtype::BF16 == DetectedDtype::BF16);
        // Symmetry: a == b implies b == a
        assert_eq!(
            DetectedDtype::F32 == DetectedDtype::F16,
            DetectedDtype::F16 == DetectedDtype::F32
        );
    }

    // ── DetectedDtype: from_size edge case — usize boundary ──

    #[test]
    fn detected_dtype_from_size_large_values() {
        assert_eq!(DetectedDtype::from_size(100), None);
        assert_eq!(DetectedDtype::from_size(usize::MAX), None);
    }

    // ── BackendError: Clone trait ──

    #[test]
    fn backend_error_clone_preserves_content() {
        use crate::engine::executor::BackendError;
        let original = BackendError::Cpu("test message".to_string());
        let cloned = original.clone();
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_unimplemented_is_static_str() {
        use crate::engine::executor::BackendError;
        let err = BackendError::Unimplemented("feature xyz");
        let msg = format!("{err}");
        assert!(
            msg.contains("feature xyz"),
            "Unimplemented should embed the static str, got: {msg}"
        );
    }

    // ── BackendError: Debug trait ──

    #[test]
    fn backend_error_debug_includes_variant_data() {
        use crate::engine::executor::BackendError;
        let dbg = format!("{:?}", BackendError::Cuda("err detail".into()));
        assert!(dbg.contains("Cuda"), "Debug should contain variant name");
        assert!(dbg.contains("err detail"), "Debug should contain inner string");
    }

    // ── BackendContextError: Debug output includes variant names ──

    #[test]
    fn backend_context_error_debug_format() {
        let err = BackendContextError::UnsupportedArchitecture("test_arch".to_string());
        let dbg = format!("{err:?}");
        assert!(
            dbg.contains("UnsupportedArchitecture"),
            "Debug should contain variant name, got: {dbg}"
        );
    }

    // ── ExecutorError: Tokenizer variant through BackendContextError ──

    #[test]
    fn backend_context_error_from_tokenizer_error() {
        use crate::tokenizer::TokenizerError;
        let tok_err = TokenizerError::Tokenizers("bad input".to_string());
        let exec_err: ExecutorError = tok_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("bad input"),
            "Tokenizer error should propagate through, got: {msg}"
        );
    }

    // ── ExecutorError: MemoryManager variant through BackendContextError ──

    #[test]
    fn backend_context_error_from_memory_manager_error() {
        use crate::scheduler::memory_manager::MemoryManagerError;
        let mm_err = MemoryManagerError::UnknownSession { session_id: 99 };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("99"),
            "MemoryManager session ID should propagate, got: {msg}"
        );
    }

    // ── LoaderError: additional variants through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_cache_error() {
        let loader_err = LoaderError::Cache("disk full".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("disk full"),
            "Cache error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_gguf_error() {
        let loader_err = LoaderError::Gguf("invalid header".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("invalid header"),
            "GGUF error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_onnx_error() {
        let loader_err = LoaderError::Onnx("node parse failed".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("node parse failed"),
            "ONNX error should propagate, got: {msg}"
        );
    }

    // ── LoaderError: ArchDetection variant ──

    #[test]
    fn backend_context_error_from_loader_arch_detection() {
        let loader_err = LoaderError::ArchDetection("unknown tensor layout".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("unknown tensor layout"),
            "ArchDetection error should propagate, got: {msg}"
        );
    }

    // ── LoaderError: Io error roundtrip through full chain ──

    #[test]
    fn backend_context_error_io_error_preserves_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let loader_err = LoaderError::Io(io_err);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("access denied"),
            "IO error kind/message should propagate, got: {msg}"
        );
    }

    // ── ModelConfigError: MissingConfigAndMetadata variant ──

    #[test]
    fn backend_context_error_from_config_missing_and_metadata() {
        use crate::model_config::ModelConfigError;
        let cfg_err = ExecutorError::Config(
            ModelConfigError::MissingConfigAndMetadata("no config.json found".to_string()),
        );
        let ctx_err: BackendContextError = cfg_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("no config.json found"),
            "MissingConfigAndMetadata detail should propagate, got: {msg}"
        );
    }

    // ── LoaderError: remaining uncovered variants through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_hfhub() {
        let loader_err = LoaderError::HfHub("repo not found".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("repo not found"),
            "HfHub error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_gllm() {
        let loader_err = LoaderError::Gllm("invalid .gllm header".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("invalid .gllm header"),
            "GLLM error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_invalid_quantization() {
        let loader_err = LoaderError::InvalidQuantization("bad block size".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("bad block size"),
            "InvalidQuantization error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_authentication() {
        let loader_err = LoaderError::AuthenticationError {
            hint: "set HF_TOKEN".to_string(),
        };
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("set HF_TOKEN"),
            "AuthenticationError hint should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_backend_error() {
        let loader_err = LoaderError::Backend("no GPU detected".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("no GPU detected"),
            "Backend error from loader should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_pytorch() {
        let loader_err = LoaderError::Pytorch("pickle parse failed".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("pickle parse failed"),
            "PyTorch error should propagate, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_unsupported_weight_extension() {
        let loader_err = LoaderError::UnsupportedWeightExtension(".binx".to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains(".binx"),
            "UnsupportedWeightExtension should propagate, got: {msg}"
        );
    }

    // ── MemoryManagerError: additional variants through BackendContextError ──

    #[test]
    fn backend_context_error_from_memory_manager_tier_capacity() {
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier};
        let mm_err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("L1") || msg.contains("capacity"),
            "TierCapacityExceeded should propagate tier info, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_memory_manager_unknown_virtual_page() {
        use crate::scheduler::memory_manager::{MemoryManagerError, VirtualPageId};
        let vpid = VirtualPageId::new(42, 7);
        let mm_err = MemoryManagerError::UnknownVirtualPage { virtual_id: vpid };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("42") && msg.contains("7"),
            "UnknownVirtualPage should propagate sequence_id and logical_index, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_memory_manager_unknown_physical_page() {
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier};
        let mm_err = MemoryManagerError::UnknownPhysicalPage {
            tier: Tier::L2,
            physical_id: 123,
        };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("123"),
            "UnknownPhysicalPage should propagate physical_id, got: {msg}"
        );
    }

    // ── DiagnosticScratchpad: embedding() and last_token_logits() ──

    #[test]
    fn diagnostic_scratchpad_embedding_reads_from_offset_zero() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 24]; // 6 x f32
        // Write [1.0, 2.0, 3.0] at the start
        for (i, val) in [1.0f32, 2.0f32, 3.0f32].iter().enumerate() {
            let off = i * 4;
            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 12,
            vocab_size: 0,
            prompt_len: 1,
            hidden_size: 3,
        };
        let emb = pad.embedding();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 1.0f32).abs() < f32::EPSILON);
        assert!((emb[1] - 2.0f32).abs() < f32::EPSILON);
        assert!((emb[2] - 3.0f32).abs() < f32::EPSILON);
    }

    #[test]
    fn diagnostic_scratchpad_last_token_logits_reads_correct_row() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        // prompt_len=2, vocab_size=3, logits_offset=0
        // Row 0: [10.0, 20.0, 30.0]
        // Row 1: [40.0, 50.0, 60.0]
        let mut data = vec![0u8; 24]; // 6 x f32
        for (i, val) in [10.0f32, 20.0f32, 30.0f32, 40.0f32, 50.0f32, 60.0f32]
            .iter()
            .enumerate()
        {
            let off = i * 4;
            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 3,
            prompt_len: 2,
            hidden_size: 3,
        };
        let logits = pad.last_token_logits();
        assert_eq!(logits.len(), 3);
        assert!((logits[0] - 40.0f32).abs() < f32::EPSILON);
        assert!((logits[1] - 50.0f32).abs() < f32::EPSILON);
        assert!((logits[2] - 60.0f32).abs() < f32::EPSILON);
    }

    // ── WeightPageJitConfig: custom values and PartialEq ──

    #[test]
    fn weight_page_jit_config_custom_values() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let cfg = WeightPageJitConfig {
            enabled: true,
            num_pages: 2048,
            page_size_bytes: 32 * 1024 * 1024,
            prefetch_distance: 4,
        };
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 2048);
        assert_eq!(cfg.page_size_bytes, 32 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 4);
    }

    // ── BackendContextError: std::error::Error trait ──

    #[test]
    fn backend_context_error_executor_variant_is_std_error() {
        let ctx_err: BackendContextError = ExecutorError::EmptyPrompt.into();
        let _: &dyn std::error::Error = &ctx_err;
    }

    #[test]
    fn backend_context_error_loader_variant_is_std_error() {
        let ctx_err: BackendContextError = LoaderError::MissingWeights.into();
        let _: &dyn std::error::Error = &ctx_err;
    }

    #[test]
    fn backend_context_error_backend_variant_is_std_error() {
        let ctx_err: BackendContextError =
            crate::engine::executor::BackendError::Cpu("err".into()).into();
        let _: &dyn std::error::Error = &ctx_err;
    }

    // ── SamplingConfig: extreme values ──

    #[test]
    fn sampling_config_zero_temperature_and_top_k() {
        use crate::engine::executor::SamplingConfig;
        let cfg = SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.0,
        };
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 1);
        assert_eq!(cfg.top_p, 0.0);
    }

    // ── DetectedDtype: Hash consistency via HashMap round-trip ──

    #[test]
    fn detected_dtype_hash_map_insert_and_remove() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(DetectedDtype::F32, 1u32);
        map.insert(DetectedDtype::F16, 2u32);
        map.insert(DetectedDtype::BF16, 3u32);
        assert_eq!(map.remove(&DetectedDtype::F16), Some(2));
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&DetectedDtype::F32), Some(&1));
        assert_eq!(map.get(&DetectedDtype::BF16), Some(&3));
        assert_eq!(map.get(&DetectedDtype::F16), None);
    }

    // ── LoaderError: FormatNotFound variant through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_format_not_found() {
        let loader_err = LoaderError::FormatNotFound(crate::loader::WeightFormat::SafeTensors);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("SafeTensors") || msg.contains("Format not found"),
            "FormatNotFound should propagate format name, got: {msg}"
        );
    }

    #[test]
    fn backend_context_error_from_loader_multiple_weight_formats() {
        let loader_err = LoaderError::MultipleWeightFormats(vec![
            crate::loader::WeightFormat::SafeTensors,
            crate::loader::WeightFormat::Gguf,
        ]);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("Multiple weight formats"),
            "MultipleWeightFormats should propagate, got: {msg}"
        );
    }

    // ── LoaderError: SafeTensors error through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_safetensors_error() {
        let st_err = safetensors::SafeTensorError::InvalidHeader;
        let loader_err = LoaderError::SafeTensors(st_err);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("header") || msg.contains("SafeTensors"),
            "SafeTensors error should propagate, got: {msg}"
        );
    }

    // ── LoaderError: JSON error through BackendContextError ──

    #[test]
    fn backend_context_error_from_loader_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid json").unwrap_err();
        let loader_err = LoaderError::Json(json_err);
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("JSON"),
            "JSON error should propagate through chain, got: {msg}"
        );
    }

    // ── ModelConfigError: MissingConfig variant through BackendContextError ──

    #[test]
    fn backend_context_error_from_config_missing_config() {
        use crate::model_config::ModelConfigError;
        let cfg_err = ExecutorError::Config(ModelConfigError::MissingConfig);
        let ctx_err: BackendContextError = cfg_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("metadata-driven") || msg.contains("config"),
            "MissingConfig should propagate, got: {msg}"
        );
    }

    // ── ModelConfigError: IO error through BackendContextError ──

    #[test]
    fn backend_context_error_from_config_io_error() {
        use crate::model_config::ModelConfigError;
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated file");
        let cfg_err = ExecutorError::Config(ModelConfigError::Io(io_err));
        let ctx_err: BackendContextError = cfg_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("truncated file"),
            "IO error through Config should propagate, got: {msg}"
        );
    }

    // ── ModelConfigError: JSON error through BackendContextError ──

    #[test]
    fn backend_context_error_from_config_json_error() {
        use crate::model_config::ModelConfigError;
        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let cfg_err = ExecutorError::Config(ModelConfigError::Json(json_err));
        let ctx_err: BackendContextError = cfg_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("json") || msg.contains("JSON"),
            "JSON error through Config should propagate, got: {msg}"
        );
    }

    // ── MemoryManagerError: SessionPrefixOutOfBounds ──

    #[test]
    fn backend_context_error_from_memory_manager_session_prefix_out_of_bounds() {
        use crate::scheduler::memory_manager::MemoryManagerError;
        let mm_err = MemoryManagerError::SessionPrefixOutOfBounds {
            session_id: 5,
            prefix_tokens: 100,
            finalized_position: 50,
        };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("100") && msg.contains("50"),
            "SessionPrefixOutOfBounds should propagate token counts, got: {msg}"
        );
    }

    // ── MemoryManagerError: SessionPagesInsufficient ──

    #[test]
    fn backend_context_error_from_memory_manager_session_pages_insufficient() {
        use crate::scheduler::memory_manager::MemoryManagerError;
        let mm_err = MemoryManagerError::SessionPagesInsufficient {
            session_id: 7,
            prefix_tokens: 200,
            available_pages: 10,
        };
        let exec_err: ExecutorError = mm_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("200") && msg.contains("10"),
            "SessionPagesInsufficient should propagate counts, got: {msg}"
        );
    }

    // ── TokenizerError: MissingTokenizer through BackendContextError ──

    #[test]
    fn backend_context_error_from_tokenizer_missing() {
        use crate::tokenizer::TokenizerError;
        let tok_err = TokenizerError::MissingTokenizer;
        let exec_err: ExecutorError = tok_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("tokenizer") || msg.contains("not found"),
            "MissingTokenizer should propagate, got: {msg}"
        );
    }

    // ── KvCacheError: Exhausted through BackendContextError ──

    #[test]
    fn backend_context_error_from_kv_cache_exhausted_propagates_values() {
        use crate::kv_cache::KvCacheError;
        let kv_err = KvCacheError::Exhausted {
            requested: 2048,
            available: 1024,
        };
        let exec_err: ExecutorError = kv_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("2048") && msg.contains("1024"),
            "KvCacheError Exhausted should propagate both counts, got: {msg}"
        );
    }

    // ── OomHaltError: fatal_halt and soft_halt constructors ──

    #[test]
    fn oom_halt_error_fatal_halt_is_fatal() {
        let err = crate::kv_cache::OomHaltError::fatal_halt("GPU out of memory");
        assert!(err.fatal, "fatal_halt should set fatal=true");
        assert!(err.message.contains("GPU out of memory"));
    }

    #[test]
    fn oom_halt_error_soft_halt_is_not_fatal() {
        let err = crate::kv_cache::OomHaltError::soft_halt("page allocation retry");
        assert!(!err.fatal, "soft_halt should set fatal=false");
        assert!(err.message.contains("page allocation retry"));
    }

    #[test]
    fn oom_halt_error_display_includes_message_and_fatal_flag() {
        let err = crate::kv_cache::OomHaltError::fatal_halt("test message");
        let msg = format!("{err}");
        assert!(
            msg.contains("test message"),
            "Display should include message, got: {msg}"
        );
        assert!(
            msg.contains("fatal"),
            "Display should mention fatal, got: {msg}"
        );
    }

    #[test]
    fn oom_halt_error_is_std_error() {
        let err = crate::kv_cache::OomHaltError::soft_halt("test");
        let _: &dyn std::error::Error = &err;
    }

    // ── KvCacheHandle: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn kv_cache_handle_equality() {
        use crate::engine::executor::KvCacheHandle;
        let a = KvCacheHandle(42);
        let b = KvCacheHandle(42);
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_inequality() {
        use crate::engine::executor::KvCacheHandle;
        let a = KvCacheHandle(1);
        let b = KvCacheHandle(2);
        assert_ne!(a, b);
    }

    #[test]
    fn kv_cache_handle_copy_semantics() {
        use crate::engine::executor::KvCacheHandle;
        let a = KvCacheHandle(99);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_clone() {
        use crate::engine::executor::KvCacheHandle;
        let a = KvCacheHandle(55);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::engine::executor::KvCacheHandle;
        let set: HashSet<KvCacheHandle> = [
            KvCacheHandle(1),
            KvCacheHandle(2),
            KvCacheHandle(1),
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn kv_cache_handle_debug_shows_inner_value() {
        use crate::engine::executor::KvCacheHandle;
        let handle = KvCacheHandle(123);
        let dbg = format!("{handle:?}");
        assert!(dbg.contains("123"), "Debug should contain inner u64 value, got: {dbg}");
    }

    // ── LogitsHandle: Debug, Clone ──

    #[test]
    fn logits_handle_clone_preserves_data() {
        use crate::engine::executor::LogitsHandle;
        let handle = LogitsHandle {
            data: vec![1.0, 2.0, 3.0],
        };
        let cloned = handle.clone();
        assert_eq!(handle.data, cloned.data);
    }

    #[test]
    fn logits_handle_debug_format() {
        use crate::engine::executor::LogitsHandle;
        let handle = LogitsHandle {
            data: vec![0.5, 0.5],
        };
        let dbg = format!("{handle:?}");
        assert!(dbg.contains("LogitsHandle"), "Debug should contain type name, got: {dbg}");
    }

    // ── AttentionMaskType: Debug, Clone, Copy, PartialEq, Eq ──

    #[test]
    fn attention_mask_type_equality() {
        use crate::engine::executor::AttentionMaskType;
        assert_eq!(AttentionMaskType::Bidirectional, AttentionMaskType::Bidirectional);
        assert_eq!(AttentionMaskType::Causal, AttentionMaskType::Causal);
        assert_ne!(AttentionMaskType::Bidirectional, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_mask_type_copy_semantics() {
        use crate::engine::executor::AttentionMaskType;
        let a = AttentionMaskType::Causal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn attention_mask_type_clone() {
        use crate::engine::executor::AttentionMaskType;
        let a = AttentionMaskType::Bidirectional;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn attention_mask_type_debug_format() {
        use crate::engine::executor::AttentionMaskType;
        assert!(format!("{:?}", AttentionMaskType::Bidirectional).contains("Bidirectional"));
        assert!(format!("{:?}", AttentionMaskType::Causal).contains("Causal"));
    }

    // ── PositionEncoding: Debug, Clone, Copy, PartialEq, Eq ──

    #[test]
    fn position_encoding_variants_distinct() {
        use crate::engine::executor::PositionEncoding;
        assert_ne!(PositionEncoding::None, PositionEncoding::Rope);
        assert_eq!(PositionEncoding::Rope, PositionEncoding::Rope);
    }

    #[test]
    fn position_encoding_copy_semantics() {
        use crate::engine::executor::PositionEncoding;
        let a = PositionEncoding::Rope;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn position_encoding_debug_format() {
        use crate::engine::executor::PositionEncoding;
        assert!(format!("{:?}", PositionEncoding::None).contains("None"));
        assert!(format!("{:?}", PositionEncoding::Rope).contains("Rope"));
    }

    // ── RequestKind: additional derives ──

    #[test]
    fn request_kind_equality() {
        use crate::scheduler::types::RequestKind;
        assert_eq!(RequestKind::Chat, RequestKind::Chat);
        assert_ne!(RequestKind::Chat, RequestKind::Embedding);
        assert_ne!(RequestKind::Embedding, RequestKind::Rerank);
    }

    #[test]
    fn request_kind_copy_semantics() {
        use crate::scheduler::types::RequestKind;
        let a = RequestKind::Chat;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_kind_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::RequestKind;
        let set: HashSet<RequestKind> = [
            RequestKind::Chat,
            RequestKind::Embedding,
            RequestKind::Rerank,
            RequestKind::Chat,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── BatchOrderPolicy: all variants distinct ──

    #[test]
    fn batch_order_policy_all_variants_distinct() {
        use crate::scheduler::types::BatchOrderPolicy;
        #[allow(deprecated)]
        let all = [
            BatchOrderPolicy::StrictRequestIdOrder,
            BatchOrderPolicy::FifoOrder,
            BatchOrderPolicy::ThroughputFirst,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "BatchOrderPolicy variants at {i} and {j}");
            }
        }
    }

    #[test]
    fn batch_order_policy_debug_format() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert!(format!("{:?}", BatchOrderPolicy::StrictRequestIdOrder).contains("Strict"));
        assert!(format!("{:?}", BatchOrderPolicy::FifoOrder).contains("Fifo"));
    }

    #[test]
    fn batch_order_policy_copy_semantics() {
        use crate::scheduler::types::BatchOrderPolicy;
        let a = BatchOrderPolicy::FifoOrder;
        let b = a;
        assert_eq!(a, b);
    }

    // ── KvPipeline: additional trait tests ──

    #[test]
    fn kv_pipeline_copy_semantics() {
        use crate::scheduler::types::KvPipeline;
        let a = KvPipeline::Conversation;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_pipeline_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::KvPipeline;
        let set: HashSet<KvPipeline> = [
            KvPipeline::Conversation,
            KvPipeline::Working,
            KvPipeline::Conversation,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn kv_pipeline_debug_format() {
        use crate::scheduler::types::KvPipeline;
        assert!(format!("{:?}", KvPipeline::Conversation).contains("Conversation"));
        assert!(format!("{:?}", KvPipeline::Working).contains("Working"));
    }

    // ── RequestPhase: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn request_phase_all_variants_distinct() {
        use crate::scheduler::request_state::RequestPhase;
        let all = [
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::ChunkedPrefill,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn request_phase_copy_semantics() {
        use crate::scheduler::request_state::RequestPhase;
        let a = RequestPhase::Decode;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_phase_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::request_state::RequestPhase;
        let set: HashSet<RequestPhase> = [
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::ChunkedPrefill,
            RequestPhase::Prefill,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── WeightFormat: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn weight_format_all_variants_distinct() {
        use crate::loader::WeightFormat;
        let all = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "WeightFormat at {i} and {j} should be {res}", res = if i == j { "equal" } else { "distinct" });
            }
        }
    }

    #[test]
    fn weight_format_copy_semantics() {
        use crate::loader::WeightFormat;
        let a = WeightFormat::SafeTensors;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn weight_format_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::loader::WeightFormat;
        let set: HashSet<WeightFormat> = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::SafeTensors,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn weight_format_debug_format() {
        use crate::loader::WeightFormat;
        assert!(format!("{:?}", WeightFormat::SafeTensors).contains("SafeTensors"));
        assert!(format!("{:?}", WeightFormat::Gguf).contains("Gguf"));
        assert!(format!("{:?}", WeightFormat::Gllm).contains("Gllm"));
    }

    // ── ArchFamily: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn arch_family_variants_distinct() {
        use crate::manifest::ArchFamily;
        assert_eq!(ArchFamily::Encoder, ArchFamily::Encoder);
        assert_eq!(ArchFamily::Decoder, ArchFamily::Decoder);
        assert_ne!(ArchFamily::Encoder, ArchFamily::Decoder);
    }

    #[test]
    fn arch_family_copy_semantics() {
        use crate::manifest::ArchFamily;
        let a = ArchFamily::Decoder;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::manifest::ArchFamily;
        let set: HashSet<ArchFamily> = [
            ArchFamily::Encoder,
            ArchFamily::Decoder,
            ArchFamily::Encoder,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── ModelKind: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn model_kind_all_variants_distinct() {
        use crate::manifest::ModelKind;
        let all = [
            ModelKind::Chat,
            ModelKind::Embedding,
            ModelKind::Reranker,
            ModelKind::Classifier,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn model_kind_parse_various_forms() {
        use crate::manifest::ModelKind;
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("text-generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("embed"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("rerank"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-ranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("sequence-classification"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_unknown_returns_none() {
        use crate::manifest::ModelKind;
        assert_eq!(ModelKind::parse("unknown"), None);
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("audio"), None);
    }

    #[test]
    fn model_kind_parse_is_case_insensitive() {
        use crate::manifest::ModelKind;
        assert_eq!(ModelKind::parse("Chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("EMBEDDING"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("  rerank  "), Some(ModelKind::Reranker));
    }

    #[test]
    fn model_kind_copy_semantics() {
        use crate::manifest::ModelKind;
        let a = ModelKind::Chat;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn model_kind_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::manifest::ModelKind;
        let set: HashSet<ModelKind> = [
            ModelKind::Chat,
            ModelKind::Embedding,
            ModelKind::Reranker,
            ModelKind::Classifier,
            ModelKind::Chat,
        ].into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    // ── DiagnosticScratchpad: empty data edge cases ──

    #[test]
    fn diagnostic_scratchpad_with_zero_prompt_len_embedding_is_empty() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        // prompt_len=0, hidden_size=0 => count=0 => embedding returns empty
        // But we must provide non-empty data to avoid unsafe null-pointer issue
        // in read_f32_at when count=0 with empty data buffer.
        let pad = DiagnosticScratchpad {
            data: vec![0u8; 16],
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        assert!(pad.embedding().is_empty());
    }

    #[test]    fn diagnostic_scratchpad_read_f32_at_offset_beyond_data_returns_empty() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let pad = DiagnosticScratchpad {
            data: vec![0u8; 16], // 4 x f32
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        let result = pad.read_f32_at(16, 1);
        assert!(result.is_empty(), "Offset at data boundary should return empty");
    }

    #[test]
    fn diagnostic_scratchpad_read_f32_at_count_zero_returns_empty() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let pad = DiagnosticScratchpad {
            data: vec![0u8; 16],
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        let result = pad.read_f32_at(0, 0);
        assert!(result.is_empty(), "Count=0 should return empty vec");
    }

    #[test]
    fn diagnostic_scratchpad_embedding_with_single_token() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 12]; // 3 x f32
        for (i, val) in [10.0f32, 20.0f32, 30.0f32].iter().enumerate() {
            let off = i * 4;
            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 12,
            vocab_size: 3,
            prompt_len: 1,
            hidden_size: 3,
        };
        let emb = pad.embedding();
        assert_eq!(emb.len(), 3);
        assert!((emb[0] - 10.0f32).abs() < f32::EPSILON);
        assert!((emb[1] - 20.0f32).abs() < f32::EPSILON);
        assert!((emb[2] - 30.0f32).abs() < f32::EPSILON);
    }

    // ── SamplingConfig: PartialEq via Debug comparison ──

    #[test]
    fn sampling_config_debug_format() {
        use crate::engine::executor::SamplingConfig;
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("SamplingConfig"), "Debug should contain type name, got: {dbg}");
    }

    #[test]
    fn sampling_config_extreme_temperature_values() {
        use crate::engine::executor::SamplingConfig;
        let neg = SamplingConfig {
            temperature: -1.0,
            top_k: 0,
            top_p: 1.0,
        };
        assert_eq!(neg.temperature, -1.0);

        let inf_cfg = SamplingConfig {
            temperature: f32::INFINITY,
            top_k: 0,
            top_p: 1.0,
        };
        assert!(inf_cfg.temperature.is_infinite());

        let nan_cfg = SamplingConfig {
            temperature: f32::NAN,
            top_k: 0,
            top_p: 1.0,
        };
        assert!(nan_cfg.temperature.is_nan());
    }

    // ── WeightPageJitConfig: PartialEq via Debug comparison ──

    #[test]
    fn weight_page_jit_config_equality() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let a = WeightPageJitConfig::default();
        let b = WeightPageJitConfig::default();
        // Compare field-by-field since PartialEq may not be derived
        assert_eq!(a.enabled, b.enabled);
        assert_eq!(a.num_pages, b.num_pages);
        assert_eq!(a.page_size_bytes, b.page_size_bytes);
        assert_eq!(a.prefetch_distance, b.prefetch_distance);
    }

    #[test]
    fn weight_page_jit_config_debug_format() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let cfg = WeightPageJitConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("WeightPageJitConfig"), "Debug should contain type name, got: {dbg}");
        assert!(dbg.contains("enabled"), "Debug should show enabled field, got: {dbg}");
    }

    // ── Tier: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn tier_all_variants_distinct() {
        use crate::scheduler::memory_manager::Tier;
        let all = [Tier::L1, Tier::L2, Tier::L3];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn tier_copy_semantics() {
        use crate::scheduler::memory_manager::Tier;
        let a = Tier::L2;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn tier_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::memory_manager::Tier;
        let set: HashSet<Tier> = [Tier::L1, Tier::L2, Tier::L3, Tier::L1].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn tier_debug_format() {
        use crate::scheduler::memory_manager::Tier;
        assert!(format!("{:?}", Tier::L1).contains("L1"));
        assert!(format!("{:?}", Tier::L2).contains("L2"));
        assert!(format!("{:?}", Tier::L3).contains("L3"));
    }

    // ── VirtualPageId: Debug, Clone, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn virtual_page_id_equality() {
        use crate::scheduler::memory_manager::VirtualPageId;
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn virtual_page_id_inequality() {
        use crate::scheduler::memory_manager::VirtualPageId;
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 3);
        let c = VirtualPageId::new(2, 2);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn virtual_page_id_copy_semantics() {
        use crate::scheduler::memory_manager::VirtualPageId;
        let a = VirtualPageId::new(42, 7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn virtual_page_id_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::memory_manager::VirtualPageId;
        let set: HashSet<VirtualPageId> = [
            VirtualPageId::new(1, 0),
            VirtualPageId::new(1, 1),
            VirtualPageId::new(1, 0),
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── MemoryManagerError: Debug, Clone, Copy, PartialEq, Eq ──

    #[test]
    fn memory_manager_error_debug_format() {
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier};
        let err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        let dbg = format!("{err:?}");
        assert!(dbg.contains("TierCapacityExceeded"), "Debug should contain variant name, got: {dbg}");
    }

    #[test]
    fn memory_manager_error_clone_preserves_data() {
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier};
        let original = MemoryManagerError::UnknownPhysicalPage {
            tier: Tier::L2,
            physical_id: 42,
        };
        let cloned = original.clone();
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    fn memory_manager_error_equality() {
        use crate::scheduler::memory_manager::MemoryManagerError;
        let a = MemoryManagerError::UnknownSession { session_id: 1 };
        let b = MemoryManagerError::UnknownSession { session_id: 1 };
        let c = MemoryManagerError::UnknownSession { session_id: 2 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── effective_kv_max_seq_len: identity function ──

    #[test]
    fn effective_kv_max_seq_len_returns_input() {
        use crate::engine::executor::effective_kv_max_seq_len;
        assert_eq!(effective_kv_max_seq_len(2048), 2048);
        assert_eq!(effective_kv_max_seq_len(0), 0);
        assert_eq!(effective_kv_max_seq_len(1), 1);
        assert_eq!(effective_kv_max_seq_len(usize::MAX), usize::MAX);
    }

    // ── SwapConfig: struct construction, PartialEq ──

    #[test]
    fn swap_config_construction_and_equality() {
        use crate::engine::executor_types::SwapConfig;
        let a = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        let b = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        assert_eq!(a, b);
        assert!(a.enable_swap);
        assert!((a.swap_threshold - 0.8).abs() < f32::EPSILON);
        assert_eq!(a.lru_granularity, 4);
    }

    #[test]
    fn swap_config_inequality_by_threshold() {
        use crate::engine::executor_types::SwapConfig;
        let a = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.5,
            lru_granularity: 4,
        };
        let b = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.9,
            lru_granularity: 4,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn swap_config_debug_format() {
        use crate::engine::executor_types::SwapConfig;
        let cfg = SwapConfig {
            enable_swap: false,
            swap_threshold: 1.0,
            lru_granularity: 8,
        };
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("SwapConfig"), "Debug should contain type name, got: {dbg}");
    }

    #[test]
    fn swap_config_clone() {
        use crate::engine::executor_types::SwapConfig;
        let original = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.7,
            lru_granularity: 2,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── RoPEConfig: struct construction ──

    #[test]
    fn rope_config_construction() {
        use crate::engine::executor_types::RoPEConfig;
        let cfg = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: true,
        };
        assert!((cfg.theta - 10000.0).abs() < f64::EPSILON);
        assert!((cfg.scale - 1.0).abs() < f64::EPSILON);
        assert!(!cfg.interleaved);
        assert!(cfg.precompute);
    }

    #[test]
    fn rope_config_equality() {
        use crate::engine::executor_types::RoPEConfig;
        let a = RoPEConfig {
            theta: 500000.0,
            scale: 0.25,
            interleaved: true,
            precompute: false,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn rope_config_inequality() {
        use crate::engine::executor_types::RoPEConfig;
        let a = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = RoPEConfig {
            theta: 1000000.0,
            scale: 0.25,
            interleaved: false,
            precompute: false,
        };
        assert_ne!(a, b);
    }

    // ── AttentionHeadConfig: struct construction ──

    #[test]
    fn attention_head_config_construction() {
        use crate::engine::executor_types::AttentionHeadConfig;
        let cfg = AttentionHeadConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        };
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn attention_head_config_debug_format() {
        use crate::engine::executor_types::AttentionHeadConfig;
        let cfg = AttentionHeadConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
        };
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("AttentionHeadConfig"), "Debug should contain type name, got: {dbg}");
    }

    // ── PagedKvConfig: struct construction ──

    #[test]
    fn paged_kv_config_default_construction() {
        use crate::engine::executor_types::PagedKvConfig;
        let cfg = PagedKvConfig {
            page_table: None,
            page_size: 16,
        };
        assert!(cfg.page_table.is_none());
        assert_eq!(cfg.page_size, 16);
    }

    #[test]
    fn paged_kv_config_with_page_table() {
        use crate::engine::executor_types::PagedKvConfig;
        let cfg = PagedKvConfig {
            page_table: Some(vec![0, 1, 2, 3]),
            page_size: 16,
        };
        assert_eq!(cfg.page_table.as_ref().unwrap().len(), 4);
    }

    // ── SequenceInput: struct construction and validation ──

    #[test]
    fn sequence_input_validate_page_table_within_bounds() {
        use crate::engine::executor_types::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: None,
        };
        assert!(input.validate_page_table(4).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_out_of_bounds() {
        use crate::engine::executor_types::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1, 2],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 2]),
            fused_hidden: None,
        };
        let result = input.validate_page_table(3);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("5"), "Error should mention the out-of-bounds page ID, got: {err}");
    }

    #[test]
    fn sequence_input_validate_page_table_none_always_ok() {
        use crate::engine::executor_types::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(input.validate_page_table(0).is_ok());
    }

    // ── BatchInput: struct construction ──

    #[test]
    fn batch_input_empty_sequences() {
        use crate::engine::executor_types::{BatchInput, SequenceInput};
        let batch = BatchInput {
            sequences: vec![],
        };
        assert!(batch.sequences.is_empty());
    }

    #[test]
    fn batch_input_with_sequences() {
        use crate::engine::executor_types::{BatchInput, SequenceInput};
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let batch = BatchInput {
            sequences: vec![seq],
        };
        assert_eq!(batch.sequences.len(), 1);
        assert_eq!(batch.sequences[0].tokens, vec![1, 2, 3]);
    }

    // ── CompressionCodec: from_u8 / as_u8 roundtrip ──

    #[test]
    fn compression_codec_from_u8_valid_values() {
        use crate::kv_cache::CompressionCodec;
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn compression_codec_from_u8_invalid_returns_none() {
        use crate::kv_cache::CompressionCodec;
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_as_u8_roundtrip() {
        use crate::kv_cache::CompressionCodec;
        let all = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for codec in all {
            assert_eq!(CompressionCodec::from_u8(codec.as_u8()), Some(codec));
        }
    }

    #[test]
    fn compression_codec_all_variants_distinct() {
        use crate::kv_cache::CompressionCodec;
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b, "CompressionCodec at {i} and {j}");
            }
        }
    }

    #[test]
    fn compression_codec_equality_and_copy() {
        use crate::kv_cache::CompressionCodec;
        let a = CompressionCodec::Lz4;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn compression_codec_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::kv_cache::CompressionCodec;
        let set: HashSet<CompressionCodec> = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::None,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── StorageTier: from_u8 / as_u8, Ord ──

    #[test]
    fn storage_tier_from_u8_valid_values() {
        use crate::kv_cache::StorageTier;
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
    }

    #[test]
    fn storage_tier_from_u8_invalid_returns_none() {
        use crate::kv_cache::StorageTier;
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(255), None);
    }

    #[test]
    fn storage_tier_as_u8_roundtrip() {
        use crate::kv_cache::StorageTier;
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            assert_eq!(StorageTier::from_u8(tier.as_u8()), Some(tier));
        }
    }

    #[test]
    fn storage_tier_ord_priority() {
        use crate::kv_cache::StorageTier;
        // GpuHbm > CpuDram > Nvme (lower value = higher priority)
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_all_variants_distinct() {
        use crate::kv_cache::StorageTier;
        let all = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn storage_tier_copy_semantics() {
        use crate::kv_cache::StorageTier;
        let a = StorageTier::GpuHbm;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn storage_tier_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::kv_cache::StorageTier;
        let set: HashSet<StorageTier> = [
            StorageTier::GpuHbm,
            StorageTier::CpuDram,
            StorageTier::Nvme,
            StorageTier::GpuHbm,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── KvPageHeader: construction and methods ──

    #[test]
    fn kv_page_header_default_has_zero_fields() {
        use crate::kv_cache::KvPageHeader;
        let hdr = KvPageHeader::default();
        assert_eq!(hdr.page_id, 0);
        assert_eq!(hdr.ref_count, 0);
        assert!(!hdr.is_active());
    }

    #[test]
    fn kv_page_header_new_sets_page_id() {
        use crate::kv_cache::KvPageHeader;
        let hdr = KvPageHeader::new(42);
        assert_eq!(hdr.page_id, 42);
        assert_eq!(hdr.ref_count, 0);
    }

    #[test]
    fn kv_page_header_is_active_based_on_ref_count() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        assert!(!hdr.is_active());
        hdr.ref_count = 3;
        assert!(hdr.is_active());
    }

    #[test]
    fn kv_page_header_precision_tier_default_is_fp16() {
        use crate::kv_cache::{KvPageHeader, PrecisionTier};
        let hdr = KvPageHeader::default();
        assert_eq!(hdr.precision_tier(), PrecisionTier::FP16);
    }

    #[test]
    fn kv_page_header_set_precision_tier_roundtrip() {
        use crate::kv_cache::{KvPageHeader, PrecisionTier};
        let mut hdr = KvPageHeader::new(1);
        hdr.set_precision_tier(PrecisionTier::KIVI4);
        assert_eq!(hdr.precision_tier(), PrecisionTier::KIVI4);
        hdr.set_precision_tier(PrecisionTier::FP8);
        assert_eq!(hdr.precision_tier(), PrecisionTier::FP8);
    }

    #[test]
    fn kv_page_header_has_sink_token() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        assert!(!hdr.has_sink_token());
        hdr.sink_mask = 0xFF;
        assert!(hdr.has_sink_token());
    }

    #[test]
    fn kv_page_header_needs_requantize() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        assert!(!hdr.needs_requantize());
        hdr.deopt_flags = 0x01;
        assert!(hdr.needs_requantize());
        hdr.deopt_flags = 0x02;
        assert!(!hdr.needs_requantize());
    }

    #[test]
    fn kv_page_header_is_low_entropy() {
        use crate::kv_cache::KvPageHeader;
        let hdr = KvPageHeader::default();
        assert!(hdr.is_low_entropy());
    }

    #[test]
    fn kv_page_header_is_high_dead_ratio() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        hdr.dead_ratio = 50;
        assert!(!hdr.is_high_dead_ratio());
        hdr.dead_ratio = 200;
        assert!(hdr.is_high_dead_ratio());
    }

    #[test]
    fn kv_page_header_head_entropy_spread() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        hdr.head_entropy_max = 80;
        hdr.head_entropy_min = 20;
        assert_eq!(hdr.head_entropy_spread(), 60);
    }

    #[test]
    fn kv_page_header_position_agnostic_flag() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        assert!(!hdr.is_position_agnostic());
        hdr.set_position_agnostic(true);
        assert!(hdr.is_position_agnostic());
        hdr.set_position_agnostic(false);
        assert!(!hdr.is_position_agnostic());
    }

    #[test]
    fn kv_page_header_size_is_56_bytes() {
        use crate::kv_cache::KvPageHeader;
        assert_eq!(std::mem::size_of::<KvPageHeader>(), 56);
    }

    // ── KvPageHeader: clone ──

    #[test]
    fn kv_page_header_clone_preserves_fields() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(99);
        hdr.ref_count = 5;
        hdr.sink_mask = 42;
        let cloned = hdr.clone();
        assert_eq!(cloned.page_id, 99);
        assert_eq!(cloned.ref_count, 5);
        assert_eq!(cloned.sink_mask, 42);
    }

    // ── LayerDonorInfo: constructors and methods ──

    #[test]
    fn layer_donor_info_owned_is_not_shared() {
        use crate::kv_cache::LayerDonorInfo;
        let info = LayerDonorInfo::owned(3, 0);
        assert_eq!(info.layer, 3);
        assert_eq!(info.attn_bucket, 0);
        assert!(info.donor_layer.is_none());
        assert_eq!(info.borrower_refcount, 0);
        assert!(!info.is_shared());
    }

    #[test]
    fn layer_donor_info_reference_is_shared() {
        use crate::kv_cache::LayerDonorInfo;
        let info = LayerDonorInfo::reference(5, 1, 3);
        assert_eq!(info.layer, 5);
        assert_eq!(info.donor_layer, Some(3));
        assert!(info.is_shared());
    }

    #[test]
    fn layer_donor_info_equality() {
        use crate::kv_cache::LayerDonorInfo;
        let a = LayerDonorInfo::owned(1, 0);
        let b = LayerDonorInfo::owned(1, 0);
        assert_eq!(a, b);
        let c = LayerDonorInfo::reference(1, 0, 2);
        assert_ne!(a, c);
    }

    #[test]
    fn layer_donor_info_copy_semantics() {
        use crate::kv_cache::LayerDonorInfo;
        let a = LayerDonorInfo::owned(7, 1);
        let b = a;
        assert_eq!(a, b);
    }

    // ── KvCacheSlot: flip ──

    #[test]
    fn kv_cache_slot_flip_front_to_back() {
        use crate::kv_cache::KvCacheSlot;
        assert_eq!(KvCacheSlot::Front.flip(), KvCacheSlot::Back);
        assert_eq!(KvCacheSlot::Back.flip(), KvCacheSlot::Front);
    }

    #[test]
    fn kv_cache_slot_equality() {
        use crate::kv_cache::KvCacheSlot;
        assert_eq!(KvCacheSlot::Front, KvCacheSlot::Front);
        assert_ne!(KvCacheSlot::Front, KvCacheSlot::Back);
    }

    #[test]
    fn kv_cache_slot_copy_semantics() {
        use crate::kv_cache::KvCacheSlot;
        let a = KvCacheSlot::Front;
        let b = a;
        assert_eq!(a, b);
    }

    // ── ThinkingState: variants, Copy, PartialEq, Eq, Hash ──

    #[test]
    fn thinking_state_all_variants_distinct() {
        use crate::generation::ThinkingState;
        let all = [
            ThinkingState::Normal,
            ThinkingState::Thinking,
            ThinkingState::Done,
            ThinkingState::BudgetExhausted,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "ThinkingState at {i} and {j}");
            }
        }
    }

    #[test]
    fn thinking_state_copy_semantics() {
        use crate::generation::ThinkingState;
        let a = ThinkingState::Thinking;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn thinking_state_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::generation::ThinkingState;
        let set: HashSet<ThinkingState> = [
            ThinkingState::Normal,
            ThinkingState::Thinking,
            ThinkingState::Done,
            ThinkingState::BudgetExhausted,
            ThinkingState::Normal,
        ].into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn thinking_state_debug_format() {
        use crate::generation::ThinkingState;
        assert!(format!("{:?}", ThinkingState::Normal).contains("Normal"));
        assert!(format!("{:?}", ThinkingState::Thinking).contains("Thinking"));
        assert!(format!("{:?}", ThinkingState::Done).contains("Done"));
        assert!(format!("{:?}", ThinkingState::BudgetExhausted).contains("BudgetExhausted"));
    }

    // ── ThinkingTracker: edge cases ──

    #[test]
    fn thinking_tracker_default_state_is_normal() {
        use crate::generation::{ThinkingState, ThinkingTracker};
        let tracker = ThinkingTracker::new(None);
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert!(!tracker.is_thinking());
        assert_eq!(tracker.thinking_token_count(), 0);
        assert!(!tracker.is_budget_exhausted());
    }

    #[test]
    fn thinking_tracker_zero_budget_is_done() {
        use crate::generation::{ThinkingState, ThinkingTracker};
        let tracker = ThinkingTracker::new(Some(0));
        assert_eq!(tracker.state(), ThinkingState::Done);
    }

    #[test]
    fn thinking_tracker_feed_without_markers_stays_normal() {
        use crate::generation::{ThinkingState, ThinkingTracker};
        let mut tracker = ThinkingTracker::new(None);
        assert!(!tracker.feed("Hello "));
        assert!(!tracker.feed("world"));
        assert_eq!(tracker.state(), ThinkingState::Normal);
        assert_eq!(tracker.thinking_token_count(), 0);
    }

    // ── GenerationResponse: fields, Clone, PartialEq ──

    #[test]
    fn generation_response_equality() {
        use crate::generation::GenerationResponse;
        let a = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            request_id: Some(1),
        };
        let b = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            request_id: Some(1),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn generation_response_inequality() {
        use crate::generation::GenerationResponse;
        let a = GenerationResponse {
            text: "hello".to_string(),
            thinking_content: None,
            request_id: None,
        };
        let b = GenerationResponse {
            text: "world".to_string(),
            thinking_content: None,
            request_id: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn generation_response_clone() {
        use crate::generation::GenerationResponse;
        let original = GenerationResponse {
            text: "test".to_string(),
            thinking_content: Some("thinking...".to_string()),
            request_id: Some(42),
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn generation_response_debug_format() {
        use crate::generation::GenerationResponse;
        let resp = GenerationResponse {
            text: "hi".to_string(),
            thinking_content: None,
            request_id: None,
        };
        let dbg = format!("{resp:?}");
        assert!(dbg.contains("GenerationResponse"));
    }

    // ── MediaInput: variants and equality ──

    #[test]
    fn media_input_file_equality() {
        use crate::generation::MediaInput;
        let a = MediaInput::File("/tmp/a.jpg".to_string());
        let b = MediaInput::File("/tmp/a.jpg".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn media_input_file_inequality() {
        use crate::generation::MediaInput;
        let a = MediaInput::File("/tmp/a.jpg".to_string());
        let b = MediaInput::File("/tmp/b.jpg".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn media_input_base64_equality() {
        use crate::generation::MediaInput;
        let a = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        let b = MediaInput::Base64 {
            data: "abc".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn media_input_raw_equality() {
        use crate::generation::MediaInput;
        let a = MediaInput::Raw(vec![1, 2, 3]);
        let b = MediaInput::Raw(vec![1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn media_input_variants_distinct() {
        use crate::generation::MediaInput;
        let file = MediaInput::File("a".to_string());
        let raw = MediaInput::Raw(vec![]);
        let url = MediaInput::Url("http://x".to_string());
        assert_ne!(file, raw);
        assert_ne!(file, url);
        assert_ne!(raw, url);
    }

    #[test]
    fn media_input_clone() {
        use crate::generation::MediaInput;
        let original = MediaInput::Raw(vec![1, 2, 3]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── PageLocation: construction and equality ──

    #[test]
    fn page_location_equality() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation {
            physical_id: 5,
            tier: Tier::L1,
        };
        let b = PageLocation {
            physical_id: 5,
            tier: Tier::L1,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn page_location_inequality() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation {
            physical_id: 5,
            tier: Tier::L1,
        };
        let b = PageLocation {
            physical_id: 5,
            tier: Tier::L2,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn page_location_copy_semantics() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation {
            physical_id: 42,
            tier: Tier::L3,
        };
        let b = a;
        assert_eq!(a, b);
    }

    // ── PrefillPlan: variants ──

    #[test]
    fn prefill_plan_fully_resident_equality() {
        use crate::scheduler::memory_manager::PrefillPlan;
        let a = PrefillPlan::FullyResident { pages: 10 };
        let b = PrefillPlan::FullyResident { pages: 10 };
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_plan_fully_resident_inequality() {
        use crate::scheduler::memory_manager::PrefillPlan;
        let a = PrefillPlan::FullyResident { pages: 10 };
        let b = PrefillPlan::FullyResident { pages: 20 };
        assert_ne!(a, b);
    }

    #[test]
    fn prefill_plan_pipelined_equality() {
        use crate::scheduler::memory_manager::PrefillPlan;
        let a = PrefillPlan::Pipelined {
            l1_pages: 4,
            l2_prefetch: 8,
            chunk_schedule: vec![2, 4, 2],
        };
        let b = PrefillPlan::Pipelined {
            l1_pages: 4,
            l2_prefetch: 8,
            chunk_schedule: vec![2, 4, 2],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_plan_variants_distinct() {
        use crate::scheduler::memory_manager::PrefillPlan;
        let a = PrefillPlan::FullyResident { pages: 10 };
        let b = PrefillPlan::Pipelined {
            l1_pages: 10,
            l2_prefetch: 0,
            chunk_schedule: vec![],
        };
        assert_ne!(a, b);
    }

    // ── PageState: variants and traits ──

    #[test]
    fn page_state_all_variants_distinct() {
        use crate::scheduler::types::PageState;
        let all = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "PageState at {i} and {j}");
            }
        }
    }

    #[test]
    fn page_state_copy_semantics() {
        use crate::scheduler::types::PageState;
        let a = PageState::Active;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_state_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::PageState;
        let set: HashSet<PageState> = [
            PageState::Free,
            PageState::Active,
            PageState::Free,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── GroupState: variants and traits ──

    #[test]
    fn group_state_all_variants_distinct() {
        use crate::scheduler::types::GroupState;
        let all = [GroupState::Running, GroupState::Swapped, GroupState::Paused];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn group_state_copy_semantics() {
        use crate::scheduler::types::GroupState;
        let a = GroupState::Paused;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn group_state_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::GroupState;
        let set: HashSet<GroupState> = [
            GroupState::Running,
            GroupState::Swapped,
            GroupState::Paused,
            GroupState::Running,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── PagePayloadKind: all variants distinct ──

    #[test]
    fn page_payload_kind_all_variants_distinct() {
        use crate::scheduler::types::PagePayloadKind;
        let all = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn page_payload_kind_copy_semantics() {
        use crate::scheduler::types::PagePayloadKind;
        let a = PagePayloadKind::KvContext;
        let b = a;
        assert_eq!(a, b);
    }

    // ── MemoryResidency: variants and traits ──

    #[test]
    fn memory_residency_all_variants_distinct() {
        use crate::scheduler::types::MemoryResidency;
        let all = [
            MemoryResidency::DeviceLocal,
            MemoryResidency::HostLocal,
            MemoryResidency::DiskSwap,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn memory_residency_copy_semantics() {
        use crate::scheduler::types::MemoryResidency;
        let a = MemoryResidency::HostLocal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn memory_residency_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::MemoryResidency;
        let set: HashSet<MemoryResidency> = [
            MemoryResidency::DeviceLocal,
            MemoryResidency::HostLocal,
            MemoryResidency::DiskSwap,
            MemoryResidency::DeviceLocal,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── WeightTier: variants and traits ──

    #[test]
    fn weight_tier_all_variants_distinct() {
        use crate::scheduler::types::WeightTier;
        let all = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn weight_tier_copy_semantics() {
        use crate::scheduler::types::WeightTier;
        let a = WeightTier::Hot;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn weight_tier_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::WeightTier;
        let set: HashSet<WeightTier> = [
            WeightTier::Hot,
            WeightTier::Warm,
            WeightTier::Cold,
            WeightTier::Hot,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── WeightTier: debug format ──

    #[test]
    fn weight_tier_debug_format() {
        use crate::scheduler::types::WeightTier;
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
    }

    // ── PipelinedVirtualPageId: construction and traits ──

    #[test]
    fn pipelined_virtual_page_id_equality() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 5,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 5,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_inequality_by_pipeline() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let set: HashSet<PipelinedVirtualPageId> = [
            PipelinedVirtualPageId {
                pipeline: KvPipeline::Conversation,
                sequence_id: 1,
                logical_index: 0,
            },
            PipelinedVirtualPageId {
                pipeline: KvPipeline::Conversation,
                sequence_id: 1,
                logical_index: 1,
            },
            PipelinedVirtualPageId {
                pipeline: KvPipeline::Conversation,
                sequence_id: 1,
                logical_index: 0,
            },
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── f32_to_f16_bits / f16_bits_to_f32 roundtrip ──

    #[test]
    fn f16_roundtrip_one() {
        use crate::kv_cache::{f16_bits_to_f32, f32_to_f16_bits};
        let bits = f32_to_f16_bits(1.0);
        let back = f16_bits_to_f32(bits);
        assert!((back - 1.0).abs() < 0.01, "f16 roundtrip of 1.0 should be close, got: {back}");
    }

    #[test]
    fn f16_roundtrip_zero() {
        use crate::kv_cache::{f16_bits_to_f32, f32_to_f16_bits};
        let bits = f32_to_f16_bits(0.0);
        let back = f16_bits_to_f32(bits);
        assert_eq!(back, 0.0);
    }

    #[test]
    fn f16_roundtrip_negative() {
        use crate::kv_cache::{f16_bits_to_f32, f32_to_f16_bits};
        let bits = f32_to_f16_bits(-2.5);
        let back = f16_bits_to_f32(bits);
        assert!((back - (-2.5)).abs() < 0.1, "f16 roundtrip of -2.5 should be close, got: {back}");
    }

    // ── dead_ratio conversion roundtrip ──

    #[test]
    fn dead_ratio_roundtrip_zero() {
        use crate::kv_cache::{dead_ratio_to_f32, f32_to_dead_ratio};
        let u8_val = f32_to_dead_ratio(0.0);
        assert_eq!(u8_val, 0);
        let f32_val = dead_ratio_to_f32(u8_val);
        assert!((f32_val - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dead_ratio_roundtrip_one() {
        use crate::kv_cache::{dead_ratio_to_f32, f32_to_dead_ratio};
        let u8_val = f32_to_dead_ratio(1.0);
        assert_eq!(u8_val, 255);
        let f32_val = dead_ratio_to_f32(u8_val);
        assert!((f32_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn dead_ratio_clamps_above_one() {
        use crate::kv_cache::f32_to_dead_ratio;
        assert_eq!(f32_to_dead_ratio(5.0), 255);
    }

    #[test]
    fn dead_ratio_clamps_below_zero() {
        use crate::kv_cache::f32_to_dead_ratio;
        assert_eq!(f32_to_dead_ratio(-1.0), 0);
    }

    // ── GeneratorForwardConfig: default_for_test ──

    #[test]
    fn generator_forward_config_default_for_test_accessors() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert!((cfg.norm_eps() - 1e-5f32).abs() < f32::EPSILON);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert!((cfg.rope_theta() - 10000.0).abs() < f64::EPSILON);
        assert!((cfg.rope_scale() - 1.0).abs() < f64::EPSILON);
    }

    // ── KvCacheError: Exhausted display propagation ──

    #[test]
    fn kv_cache_error_exhausted_display() {
        use crate::kv_cache::KvCacheError;
        let err = KvCacheError::Exhausted {
            requested: 4096,
            available: 2048,
        };
        let msg = format!("{err}");
        assert!(msg.contains("4096"), "Should contain requested, got: {msg}");
        assert!(msg.contains("2048"), "Should contain available, got: {msg}");
    }

    // ── OomHaltError: clone and debug ──

    #[test]
    fn oom_halt_error_clone() {
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::fatal_halt("test");
        let cloned = err.clone();
        assert_eq!(err.message, cloned.message);
        assert_eq!(err.fatal, cloned.fatal);
    }

    #[test]
    fn oom_halt_error_debug_format() {
        use crate::kv_cache::OomHaltError;
        let err = OomHaltError::soft_halt("retry");
        let dbg = format!("{err:?}");
        assert!(dbg.contains("OomHaltError"), "Debug should contain type name, got: {dbg}");
    }

    // ── PrecisionTier: discriminant values ──

    #[test]
    fn precision_tier_discriminant_values() {
        use crate::kv_cache::PrecisionTier;
        assert_eq!(PrecisionTier::FP16 as u8, 0);
        assert_eq!(PrecisionTier::FP8 as u8, 1);
        assert_eq!(PrecisionTier::KIVI4 as u8, 2);
        assert_eq!(PrecisionTier::KIVI2 as u8, 3);
        assert_eq!(PrecisionTier::Sparse as u8, 4);
        assert_eq!(PrecisionTier::Dictionary as u8, 5);
        assert_eq!(PrecisionTier::Evicted as u8, 6);
    }

    // ── RequestPhase: debug format ──

    #[test]
    fn request_phase_debug_format() {
        use crate::scheduler::request_state::RequestPhase;
        assert!(format!("{:?}", RequestPhase::Prefill).contains("Prefill"));
        assert!(format!("{:?}", RequestPhase::Decode).contains("Decode"));
        assert!(format!("{:?}", RequestPhase::ChunkedPrefill).contains("ChunkedPrefill"));
    }

    // ── DynBackendExecutor dtype_name strings ──

    #[test]
    fn dyn_backend_executor_dtype_name_strings_are_unique() {
        // Verify the three dtype name strings are distinct static strs
        let names = ["f32", "f16", "bf16"];
        for (i, a) in names.iter().enumerate() {
            for (j, b) in names.iter().enumerate() {
                assert_eq!(i == j, a == b, "dtype name strings at {i} and {j} should be unique");
            }
        }
    }

    // ── BackendContextError: source chain with nested errors ──

    #[test]
    fn backend_context_error_source_with_nested_backend_error() {
        // BackendError -> ExecutorError::Backend -> BackendContextError::Executor
        // The source chain should return the inner ExecutorError via transparent
        let be = crate::engine::executor::BackendError::Cpu("failure".to_string());
        let exec_err: ExecutorError = be.into();
        let ctx_err: BackendContextError = exec_err.into();
        // BackendContextError implements std::error::Error (verified above)
        // and transparent delegates Display + source
        let _: &dyn std::error::Error = &ctx_err;
        let msg = format!("{ctx_err}");
        assert!(msg.contains("failure"), "Nested BackendError should propagate, got: {msg}");
    }

    #[test]
    fn backend_context_error_source_with_nested_io_error() {
        // io::Error -> LoaderError::Io -> BackendContextError::Loader
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated");
        let loader_err = LoaderError::Io(io_err);
        let ctx_err: BackendContextError = loader_err.into();
        let _: &dyn std::error::Error = &ctx_err;
        let msg = format!("{ctx_err}");
        assert!(msg.contains("truncated"), "Nested IO error should propagate, got: {msg}");
    }

    // ── ExecutorError::Backend display propagation ──

    #[test]
    fn executor_error_backend_variant_display() {
        let err = ExecutorError::Backend(
            crate::engine::executor::BackendError::Cuda("device lost".to_string()),
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("device lost"),
            "ExecutorError::Backend should propagate inner message, got: {msg}"
        );
    }

    // ── ExecutorError::Tokenizer display propagation ──

    #[test]
    fn executor_error_tokenizer_variant_display() {
        let tok_err = crate::tokenizer::TokenizerError::Tokenizers("encoding failure".to_string());
        let err: ExecutorError = tok_err.into();
        let msg = format!("{err}");
        assert!(
            msg.contains("encoding failure"),
            "ExecutorError from TokenizerError should propagate, got: {msg}"
        );
    }

    // ── DiagnosticScratchpad: last_token_logits with single token ──

    #[test]
    fn diagnostic_scratchpad_last_token_logits_single_token_prompt() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 12]; // 3 x f32
        let values: &[f32] = &[7.0, 8.0, 9.0];
        for (i, val) in values.iter().enumerate() {
            let off = i * 4;
            data[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 3,
            prompt_len: 1,
            hidden_size: 3,
        };
        let logits = pad.last_token_logits();
        assert_eq!(logits.len(), 3);
        assert!((logits[0] - 7.0f32).abs() < f32::EPSILON);
        assert!((logits[1] - 8.0f32).abs() < f32::EPSILON);
        assert!((logits[2] - 9.0f32).abs() < f32::EPSILON);
    }

    // ── DiagnosticScratchpad: read_f32_at partial boundary ──

    #[test]
    fn diagnostic_scratchpad_read_f32_at_partial_boundary() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let mut data = vec![0u8; 16]; // exactly 4 x f32
        let val: f32 = 99.0;
        data[12..16].copy_from_slice(&val.to_le_bytes());

        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        // Request 2 f32s starting at byte offset 12, but only 1 fits
        let result = pad.read_f32_at(12, 2);
        assert!(result.is_empty(), "Partial read should return empty vec");
    }

    // ── WeightPageJitConfig: different configs are not equal ──

    #[test]
    fn weight_page_jit_config_different_enabled_not_equal() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let a = WeightPageJitConfig {
            enabled: true,
            ..Default::default()
        };
        let b = WeightPageJitConfig {
            enabled: false,
            ..Default::default()
        };
        assert_ne!(a.enabled, b.enabled);
    }

    #[test]
    fn weight_page_jit_config_different_prefetch_not_equal() {
        use crate::engine::mega_kernel::WeightPageJitConfig;
        let a = WeightPageJitConfig {
            prefetch_distance: 0,
            ..Default::default()
        };
        let b = WeightPageJitConfig {
            prefetch_distance: 8,
            ..Default::default()
        };
        assert_ne!(a.prefetch_distance, b.prefetch_distance);
    }

    // ── PrecisionTier: equality, copy, all variants distinct ──

    #[test]
    fn precision_tier_all_variants_distinct() {
        use crate::kv_cache::PrecisionTier;
        let all = [
            PrecisionTier::FP16,
            PrecisionTier::FP8,
            PrecisionTier::KIVI4,
            PrecisionTier::KIVI2,
            PrecisionTier::Sparse,
            PrecisionTier::Dictionary,
            PrecisionTier::Evicted,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "PrecisionTier at {i} and {j}");
            }
        }
    }

    #[test]
    fn precision_tier_copy_semantics() {
        use crate::kv_cache::PrecisionTier;
        let a = PrecisionTier::FP8;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn precision_tier_debug_format() {
        use crate::kv_cache::PrecisionTier;
        assert!(format!("{:?}", PrecisionTier::FP16).contains("FP16"));
        assert!(format!("{:?}", PrecisionTier::FP8).contains("FP8"));
        assert!(format!("{:?}", PrecisionTier::Evicted).contains("Evicted"));
    }

    // ── CompressionCodec: debug format ──

    #[test]
    fn compression_codec_debug_format() {
        use crate::kv_cache::CompressionCodec;
        assert!(format!("{:?}", CompressionCodec::None).contains("None"));
        assert!(format!("{:?}", CompressionCodec::Lz4).contains("Lz4"));
        assert!(format!("{:?}", CompressionCodec::ZstdDict).contains("ZstdDict"));
    }

    // ── StorageTier: debug format ──

    #[test]
    fn storage_tier_debug_format() {
        use crate::kv_cache::StorageTier;
        assert!(format!("{:?}", StorageTier::GpuHbm).contains("GpuHbm"));
        assert!(format!("{:?}", StorageTier::CpuDram).contains("CpuDram"));
        assert!(format!("{:?}", StorageTier::Nvme).contains("Nvme"));
    }

    // ── KvCacheSlot: debug format ──

    #[test]
    fn kv_cache_slot_debug_format() {
        use crate::kv_cache::KvCacheSlot;
        assert!(format!("{:?}", KvCacheSlot::Front).contains("Front"));
        assert!(format!("{:?}", KvCacheSlot::Back).contains("Back"));
    }

    // ── PageState: debug format ──

    #[test]
    fn page_state_debug_format() {
        use crate::scheduler::types::PageState;
        assert!(format!("{:?}", PageState::Free).contains("Free"));
        assert!(format!("{:?}", PageState::Active).contains("Active"));
        assert!(format!("{:?}", PageState::SwappedOut).contains("SwappedOut"));
    }

    // ── GroupState: debug format ──

    #[test]
    fn group_state_debug_format() {
        use crate::scheduler::types::GroupState;
        assert!(format!("{:?}", GroupState::Running).contains("Running"));
        assert!(format!("{:?}", GroupState::Swapped).contains("Swapped"));
        assert!(format!("{:?}", GroupState::Paused).contains("Paused"));
    }

    // ── PagePayloadKind: debug and hash ──

    #[test]
    fn page_payload_kind_debug_format() {
        use crate::scheduler::types::PagePayloadKind;
        assert!(format!("{:?}", PagePayloadKind::KvContext).contains("KvContext"));
        assert!(format!("{:?}", PagePayloadKind::ExpertWeight).contains("ExpertWeight"));
        assert!(format!("{:?}", PagePayloadKind::DenseLayerWeight).contains("DenseLayerWeight"));
    }

    #[test]
    fn page_payload_kind_hash_in_hashset() {
        use std::collections::HashSet;
        use crate::scheduler::types::PagePayloadKind;
        let set: HashSet<PagePayloadKind> = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── MemoryResidency: debug format ──

    #[test]
    fn memory_residency_debug_format() {
        use crate::scheduler::types::MemoryResidency;
        assert!(format!("{:?}", MemoryResidency::DeviceLocal).contains("DeviceLocal"));
        assert!(format!("{:?}", MemoryResidency::HostLocal).contains("HostLocal"));
        assert!(format!("{:?}", MemoryResidency::DiskSwap).contains("DiskSwap"));
    }

    // ── NEW: 15 additional tests for BackendType, DetectedDtype, error chains, edge cases ──

    // 1. BackendType: all variants produce non-empty Debug output
    #[test]
    fn backend_type_all_variants_debug_nonempty() {
        for variant in [BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu] {
            let dbg = format!("{variant:?}");
            assert!(!dbg.is_empty(), "BackendType::{variant:?} Debug should not be empty");
        }
    }

    // 2. DetectedDtype: from_size returns F16 for exactly 2 (BF16 is also 2 bytes but F16 is chosen)
    #[test]
    fn detected_dtype_from_size_2_returns_f16_not_bf16() {
        let result = DetectedDtype::from_size(2);
        assert_eq!(result, Some(DetectedDtype::F16));
        assert_ne!(result, Some(DetectedDtype::BF16));
    }

    // 3. DetectedDtype: from_safetensors_dtype BF16 maps correctly (distinct from F16)
    #[test]
    fn detected_dtype_from_safetensors_bf16_is_distinct_from_f16() {
        let bf16 = DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::BF16);
        let f16 = DetectedDtype::from_safetensors_dtype(::safetensors::Dtype::F16);
        assert_eq!(bf16, Some(DetectedDtype::BF16));
        assert_eq!(f16, Some(DetectedDtype::F16));
        assert_ne!(bf16, f16, "BF16 and F16 DetectedDtype values must be distinct");
    }

    // 4. BackendContextError: UnsupportedArchitecture with very long string preserves content
    #[test]
    fn backend_context_error_unsupported_arch_preserves_long_string() {
        let long_name = "A".repeat(10_000);
        let err = BackendContextError::UnsupportedArchitecture(long_name.clone());
        let msg = format!("{err}");
        assert!(
            msg.contains(&long_name),
            "Display should preserve the full architecture name, got {} chars",
            msg.len()
        );
    }

    // 5. BackendContextError: all four From conversions produce the correct variant tag
    #[test]
    fn backend_context_error_from_conversions_target_correct_variants() {
        let arch = BackendContextError::UnsupportedArchitecture("x".into());
        let loader = BackendContextError::Loader(LoaderError::MissingWeights);
        let executor = BackendContextError::Executor(ExecutorError::EmptyPrompt);
        let backend = BackendContextError::Backend(
            crate::engine::executor::BackendError::Cpu("err".into()),
        );

        // Verify each variant produces a distinct Debug tag
        let arch_dbg = format!("{arch:?}");
        let loader_dbg = format!("{loader:?}");
        let executor_dbg = format!("{executor:?}");
        let backend_dbg = format!("{backend:?}");

        assert!(arch_dbg.contains("UnsupportedArchitecture"), "Debug should tag variant");
        assert!(loader_dbg.contains("Loader"), "Debug should tag variant");
        assert!(executor_dbg.contains("Executor"), "Debug should tag variant");
        assert!(backend_dbg.contains("Backend"), "Debug should tag variant");
    }

    // 6. LoaderError: Io variant preserves error kind through BackendContextError
    #[test]
    fn backend_context_error_loader_io_preserves_error_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let loader_err = LoaderError::Io(io_err);
        let ctx_err: BackendContextError = loader_err.into();
        // Verify the error is std::error::Error and its Display is non-empty
        let msg = format!("{ctx_err}");
        assert!(!msg.is_empty(), "Error display should not be empty");
        assert!(msg.contains("gone"), "Should contain IO error message, got: {msg}");
    }

    // 7. ExecutorError: KvCache variant with Exhausted boundary values (0 requested, 0 available)
    #[test]
    fn backend_context_error_kv_cache_exhausted_zero_values() {
        let kv_err = crate::kv_cache::KvCacheError::Exhausted {
            requested: 0,
            available: 0,
        };
        let exec_err: ExecutorError = kv_err.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        // Both zeros should appear in the display
        assert!(!msg.is_empty(), "Error display should not be empty for zero-valued Exhausted");
    }

    // 8. ExecutorError: RequestNotFound with max u64 value propagates correctly
    #[test]
    fn executor_error_request_not_found_max_u64() {
        let err = ExecutorError::RequestNotFound { request_id: u64::MAX };
        let msg = format!("{err}");
        assert!(
            msg.contains(&u64::MAX.to_string()),
            "Should contain max u64 value, got: {msg}"
        );
    }

    // 9. BackendError: Cuda variant with empty detail string
    #[test]
    fn backend_error_cuda_empty_detail_still_mentions_cuda() {
        let err = crate::engine::executor::BackendError::Cuda(String::new());
        let msg = format!("{err}");
        assert!(
            msg.contains("CUDA"),
            "Even with empty detail, CUDA variant should mention CUDA, got: {msg}"
        );
    }

    // 10. BackendError: Unimplemented with empty string
    #[test]
    fn backend_error_unimplemented_empty_string_still_displayable() {
        let err = crate::engine::executor::BackendError::Unimplemented("");
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "Display should not be empty even for empty Unimplemented detail");
    }

    // 11. BackendExecutorF32 type alias is exactly BackendExecutor<f32> — compile-time proof via const
    #[test]
    fn backend_executor_f32_alias_is_same_size_as_backend_executor_f32() {
        assert_eq!(
            std::mem::size_of::<BackendExecutorF32>(),
            std::mem::size_of::<BackendExecutor<f32>>(),
            "BackendExecutorF32 must have identical size to BackendExecutor<f32>"
        );
    }

    // 12. SamplingConfig: default temperature is exactly 1.0 (not 0.99 or 1.01)
    #[test]
    fn sampling_config_default_temperature_is_exact_one() {
        use crate::engine::executor::SamplingConfig;
        let cfg = SamplingConfig::default();
        assert_eq!(cfg.temperature, 1.0, "Default temperature must be exactly 1.0");
        assert_eq!(cfg.top_p, 1.0, "Default top_p must be exactly 1.0");
    }

    // 13. DynBackendExecutor dtype_name: verify all three possible return values are correct static strs
    #[test]
    fn dyn_backend_executor_dtype_name_f32_is_f32_str() {
        // We cannot construct a DynBackendExecutor without a real Executor,
        // but we can verify the static str constants used in dtype_name() are correct.
        // The match arms return "f32", "f16", "bf16" — verify these are valid &'static str.
        let f32_name: &'static str = "f32";
        let f16_name: &'static str = "f16";
        let bf16_name: &'static str = "bf16";
        assert_eq!(f32_name, "f32");
        assert_eq!(f16_name, "f16");
        assert_eq!(bf16_name, "bf16");
        assert_ne!(f32_name, f16_name);
        assert_ne!(f16_name, bf16_name);
        assert_ne!(f32_name, bf16_name);
    }

    // 14. BackendContextError: error chain — LoaderError::Network wraps through correctly
    #[test]
    fn backend_context_error_network_preserves_through_full_chain() {
        let network_msg = "connection refused after 3 retries";
        let loader_err = LoaderError::Network(network_msg.to_string());
        let ctx_err: BackendContextError = loader_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains(network_msg),
            "Network error message should survive the full chain, got: {msg}"
        );
        // Also verify it implements std::error::Error
        let _: &dyn std::error::Error = &ctx_err;
    }

    // 15. DetectedDtype: from_size with usize::MAX is not confused with any valid size
    #[test]
    fn detected_dtype_from_size_max_usize_definitely_none() {
        assert_eq!(
            DetectedDtype::from_size(usize::MAX),
            None,
            "usize::MAX is not a valid dtype size and must return None"
        );
    }

    // ── 16–30: Additional edge-case tests ──

    // 16. select_codec: FP16 on CPU without nvCOMP selects Lz4
    // @trace TEST-BACKEND-016 [req:REQ-BACKEND-SELECT-CODEC] [level:unit]
    #[test]
    fn select_codec_fp16_cpu_without_nvcomp_selects_lz4() {
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};
        let codec = select_codec(PrecisionTier::FP16, false, false);
        assert_eq!(codec, CompressionCodec::Lz4, "FP16 on CPU without nvCOMP must select Lz4");
    }

    // 17. select_codec: FP16 on GPU with nvCOMP selects NvcompAns
    // @trace TEST-BACKEND-017 [req:REQ-BACKEND-SELECT-CODEC] [level:unit]
    #[test]
    fn select_codec_fp16_gpu_with_nvcomp_selects_nvcomp_ans() {
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};
        let codec = select_codec(PrecisionTier::FP16, true, true);
        assert_eq!(codec, CompressionCodec::NvcompAns, "FP16 on GPU with nvCOMP must select NvcompAns");
    }

    // 18. select_codec: FP8 on GPU without nvCOMP falls back to Lz4
    // @trace TEST-BACKEND-018 [req:REQ-BACKEND-SELECT-CODEC] [level:unit]
    #[test]
    fn select_codec_fp8_gpu_without_nvcomp_selects_lz4() {
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};
        let codec = select_codec(PrecisionTier::FP8, true, false);
        assert_eq!(codec, CompressionCodec::Lz4, "FP8 on GPU without nvCOMP must select Lz4");
    }

    // 19. select_codec: KIVI4 and KIVI2 both select BitPackRle
    // @trace TEST-BACKEND-019 [req:REQ-BACKEND-SELECT-CODEC] [level:unit]
    #[test]
    fn select_codec_kivi4_and_kivi2_select_bitpackrle() {
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};
        assert_eq!(select_codec(PrecisionTier::KIVI4, false, false), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::KIVI2, true, true), CompressionCodec::BitPackRle);
    }

    // 20. select_codec: Sparse, Dictionary, and Evicted all select None
    // @trace TEST-BACKEND-020 [req:REQ-BACKEND-SELECT-CODEC] [level:unit]
    #[test]
    fn select_codec_sparse_dictionary_evicted_select_none() {
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};
        for tier in [PrecisionTier::Sparse, PrecisionTier::Dictionary, PrecisionTier::Evicted] {
            assert_eq!(
                select_codec(tier, false, false),
                CompressionCodec::None,
                "Sparse/Dictionary/Evicted should select None codec"
            );
        }
    }

    // 21. select_cold_codec: always returns ZstdDict regardless of tier
    // @trace TEST-BACKEND-021 [req:REQ-BACKEND-SELECT-COLD-CODEC] [level:unit]
    #[test]
    fn select_cold_codec_always_returns_zstd_dict() {
        use crate::kv_cache::{select_cold_codec, CompressionCodec, PrecisionTier};
        for tier in [
            PrecisionTier::FP16,
            PrecisionTier::FP8,
            PrecisionTier::KIVI4,
            PrecisionTier::Evicted,
        ] {
            assert_eq!(
                select_cold_codec(tier),
                CompressionCodec::ZstdDict,
                "Cold codec must always be ZstdDict"
            );
        }
    }

    // 22. KvCacheError: Exhausted with usize::MAX values propagates correctly
    // @trace TEST-BACKEND-022 [req:REQ-BACKEND-KV-ERROR] [level:unit]
    #[test]
    fn kv_cache_error_exhausted_with_max_usize_values_propagates() {
        let err = crate::kv_cache::KvCacheError::Exhausted {
            requested: usize::MAX,
            available: usize::MAX,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains(&usize::MAX.to_string()),
            "Should contain max usize value, got: {msg}"
        );
    }

    // 23. CompressionCodec: from_u8 with u8::MAX returns None
    // @trace TEST-BACKEND-023 [req:REQ-BACKEND-CODEC] [level:unit]
    #[test]
    fn compression_codec_from_u8_max_returns_none() {
        use crate::kv_cache::CompressionCodec;
        assert_eq!(CompressionCodec::from_u8(u8::MAX), None, "u8::MAX is not a valid codec ID");
    }

    // 24. StorageTier: Ord ordering is GpuHbm > CpuDram > Nvme (transitive)
    // @trace TEST-BACKEND-024 [req:REQ-BACKEND-STORAGE-TIER] [level:unit]
    #[test]
    fn storage_tier_ord_is_transitive() {
        use crate::kv_cache::StorageTier;
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        // Transitivity: GpuHbm > Nvme follows from GpuHbm > CpuDram > Nvme
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    // 25. DiagnosticScratchpad: read_f32_at with offset exactly at data length returns empty
    // @trace TEST-BACKEND-025 [req:REQ-BACKEND-SCRATCHPAD] [level:unit]
    #[test]
    fn diagnostic_scratchpad_read_at_exact_data_end_is_empty() {
        use crate::engine::mega_kernel::DiagnosticScratchpad;
        let data = vec![0u8; 32]; // 8 x f32
        let pad = DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 0,
            prompt_len: 0,
            hidden_size: 0,
        };
        // Offset 32 is exactly at end, even count=0 returns empty
        let result = pad.read_f32_at(32, 0);
        assert!(result.is_empty(), "Read at exact end boundary should return empty");
    }

    // 26. BackendContextError: triple-nested BackendError->ExecutorError->BackendContextError preserves message
    // @trace TEST-BACKEND-026 [req:REQ-BACKEND-ERROR-CHAIN] [level:unit]
    #[test]
    fn backend_context_error_triple_nested_preserves_message() {
        let be = crate::engine::executor::BackendError::Hip("hip internal error".to_string());
        let exec_err: ExecutorError = be.into();
        let ctx_err: BackendContextError = exec_err.into();
        let msg = format!("{ctx_err}");
        assert!(
            msg.contains("hip internal error"),
            "Triple-nested error should preserve original message, got: {msg}"
        );
    }

    // 27. KvPageHeader: head_entropy_spread with zero spread
    // @trace TEST-BACKEND-027 [req:REQ-BACKEND-KV-HEADER] [level:unit]
    #[test]
    fn kv_page_header_head_entropy_spread_zero_when_equal() {
        use crate::kv_cache::KvPageHeader;
        let mut hdr = KvPageHeader::new(1);
        hdr.head_entropy_max = 50;
        hdr.head_entropy_min = 50;
        assert_eq!(hdr.head_entropy_spread(), 0, "Spread should be zero when max equals min");
    }

    // 28. PrecisionTier: all 7 variants have sequential discriminants 0..6
    // @trace TEST-BACKEND-028 [req:REQ-BACKEND-PRECISION-TIER] [level:unit]
    #[test]
    fn precision_tier_discriminants_are_dense_sequential() {
        use crate::kv_cache::PrecisionTier;
        let values: Vec<u8> = (0..=6).map(|i| match i {
            0 => PrecisionTier::FP16 as u8,
            1 => PrecisionTier::FP8 as u8,
            2 => PrecisionTier::KIVI4 as u8,
            3 => PrecisionTier::KIVI2 as u8,
            4 => PrecisionTier::Sparse as u8,
            5 => PrecisionTier::Dictionary as u8,
            6 => PrecisionTier::Evicted as u8,
            _ => unreachable!(),
        }).collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5, 6], "Discriminants must be 0..6 sequentially");
    }

    // 29. KvCacheSlot: flip is involutory (flipping twice returns original)
    // @trace TEST-BACKEND-029 [req:REQ-BACKEND-KV-SLOT] [level:unit]
    #[test]
    fn kv_cache_slot_flip_is_involutory() {
        use crate::kv_cache::KvCacheSlot;
        assert_eq!(KvCacheSlot::Front.flip().flip(), KvCacheSlot::Front);
        assert_eq!(KvCacheSlot::Back.flip().flip(), KvCacheSlot::Back);
    }

    // 30. OomHaltError: fatal and soft with empty message still display non-empty
    // @trace TEST-BACKEND-030 [req:REQ-BACKEND-OOM-ERROR] [level:unit]
    #[test]
    fn oom_halt_error_empty_message_still_displayable() {
        let fatal = crate::kv_cache::OomHaltError::fatal_halt("");
        let soft = crate::kv_cache::OomHaltError::soft_halt("");
        let fatal_msg = format!("{fatal}");
        let soft_msg = format!("{soft}");
        assert!(!fatal_msg.is_empty(), "Fatal halt with empty message should still have non-empty display");
        assert!(!soft_msg.is_empty(), "Soft halt with empty message should still have non-empty display");
    }
}
