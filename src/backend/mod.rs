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
                crate::compat::multimodal::MultimodalTokenIds::gemma4_defaults()
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
            .unwrap_or_else(|| crate::compat::multimodal::MultimodalTokenIds::gemma4_defaults());

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
