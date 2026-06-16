//! Audio Encoder — USM-style Conformer (Gemma 4 多模态)
//!
//! Pipeline:
//! 1. **Mel spectrogram preprocessing** — raw PCM waveform → log-mel frames
//!    `[num_frames, num_mels]` (non-JIT: pure Rust short-time FFT + mel filter;
//!    this is fixed signal-processing math, not model compute).
//! 2. **Conformer encoder** — JIT-compiled `CompilerGraph` per Conformer block
//!    (half-step FFN → MHA → ConvModule(+DepthwiseConv1D) → half-step FFN).
//!    Each of the `num_layers` blocks is compiled at load time and executed
//!    sequentially with layer-specific weights.
//!
//! **零妥协铁律**:
//! - Mel preprocessing 是输入信号转换 (音频 → 特征),不走 JIT。
//! - 所有 Conformer block 计算 (LayerNorm / GEMM / SiLU / DepthwiseConv1D /
//!   MHA / Residual) **必须** 经过 `InferenceCompiler::compile_mega_kernel_from_graph`。
//! - 绝无 `scalar_*` fallback, 绝无 `emit_nop`,绝无硬编码 workaround。
//!
//! 权重来源由 `AudioTensorLookup` 注入 (仿 `VisionTensorLookup`)。
//! Caller (典型为 Gemma 4 加载路径) 负责把权重张量按名字暴露出来。
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 4 个片段):
//! - `audio_fragments/config_mel.inc.rs`   — AudioConfig + AudioTensorLookup + Mel spectrogram
//! - `audio_fragments/jit_encode.inc.rs`   — JIT Conformer block + 权重打包 + audio_encode
//! - `audio_fragments/conformer.inc.rs`    — UsmConformerEncoder + try_build + InMemoryAudioWeights
//! - `audio_fragments/tests.inc.rs`        — 测试模块

use std::collections::HashMap;
use std::f32::consts::PI;

use gllm_kernels::compiler::{
    CompilerGraph, InferenceCompiler, OpKind, SymDim,
};
use gllm_kernels::compiler::mega_kernel_abi::{CompileConfig, CompileTarget};
use gllm_kernels::types::DType;

use crate::compat::multimodal::{MediaKind, MultimodalEncoded, MultimodalEncoder, EncoderMedia};
use crate::engine::executor::BackendError;

include!("audio_fragments/config_mel.inc.rs");
include!("audio_fragments/jit_encode.inc.rs");
include!("audio_fragments/conformer.inc.rs");

#[cfg(test)]
include!("audio_fragments/tests.inc.rs");
