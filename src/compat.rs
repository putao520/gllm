//! Compatibility shim for gllm_kernels phantom types.
//!
//! This module provides:
//! - `Element` re-export from gllm-kernels
//! - `Backend<E>` trait + `TensorLookup` trait
//! - `CpuBackend<E>` / `CudaBackend<E>` stub implementations
//! - Backward-compatible re-export modules (`kernel_types`, `backend_trait`, `cpu_backend`)

// Re-export the *real* Element trait from gllm-kernels.
pub use gllm_kernels::traits::Element;

// Backward-compatible re-export modules
pub mod kernel_types {
    pub use crate::engine::executor::{
        GeneratorForwardConfig, KvCacheConfig, PositionEncoding, SamplingConfig, SwapConfig,
    };
    pub use crate::scheduler::types::{PageId, PageState, PhysicalId, RequestId, StorageKey};
}

pub mod backend_trait {
    pub use super::Element;
    pub use crate::engine::executor::{
        AttentionTopology, BackendError, BatchInput, GeneratorForwardConfig, KvCacheConfig,
        KvCacheHandle, LogitsHandle, SamplingConfig,
    };
    pub use crate::scheduler::types::{PageId, PageState, StorageKey};
    pub use gllm_kernels::quant::QuantType;

    /// Abstract compute backend.
    pub trait Backend<E: Element>: Send + Sync + 'static + std::fmt::Debug {
        type Tensor: std::fmt::Debug + Clone + Send + Sync + 'static;

        fn alloc_kv_cache(&self, config: &KvCacheConfig) -> Result<KvCacheHandle, BackendError>;

        fn batch_forward_gpu_pure(
            &self,
            input: &BatchInput,
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            kv_caches: &mut [KvCacheHandle],
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<LogitsHandle>, BackendError>
        where
            Self: Sized;

        fn sample_from_tensor(
            &self,
            logits: &LogitsHandle,
            topology: &AttentionTopology,
            vocab_size: usize,
            sampling: &SamplingConfig,
        ) -> Result<Vec<u32>, BackendError>;

        fn embedding_forward_gpu_pure(
            &self,
            tokens: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        fn rerank_forward_gpu_pure(
            &self,
            tokens: &[u32],
            topology: &AttentionTopology,
            weights: &dyn TensorLookup<E, Self>,
            config: &GeneratorForwardConfig,
        ) -> Result<Vec<f32>, BackendError>
        where
            Self: Sized;

        fn get_memory_pressure(&self) -> Result<f32, BackendError>;

        fn swap_out_pages(
            &self,
            handle: &mut KvCacheHandle,
            mappings: &[(PageId, StorageKey)],
        ) -> Result<(), BackendError>;

        fn swap_in_pages(
            &self,
            handle: &mut KvCacheHandle,
            mappings: &[(PageId, StorageKey)],
        ) -> Result<(), BackendError>;

        fn get_page_states(
            &self,
            handle: &KvCacheHandle,
        ) -> Result<Vec<(PageId, PageState)>, BackendError>;

        fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BackendError>;

        /// Quantized matrix multiplication dispatching to K-Quant/Classic/IQ kernels.
        fn quantized_matmul(
            &self,
            _weight_blocks: &[u8],
            _input: &[E],
            _output: &mut [E],
            _quant_type: QuantType,
            _m: usize,
            _n: usize,
            _k: usize,
        ) -> Result<(), BackendError> {
            Err(BackendError::Unimplemented("quantized_matmul"))
        }

        /// Dequantize raw block bytes to f32 output.
        fn dequantize(
            &self,
            _block_data: &[u8],
            _output: &mut [f32],
            _quant_type: QuantType,
        ) -> Result<(), BackendError> {
            Err(BackendError::Unimplemented("dequantize"))
        }
    }

    /// Trait for looking up named tensors in a weight store.
    pub trait TensorLookup<E: Element, B: Backend<E> + ?Sized> {
        fn get_tensor(&self, name: &str) -> Option<&B::Tensor>;
        fn tensor_shape(&self, name: &str) -> Option<&[usize]>;

        /// Returns a quantized tensor by name (if stored in the quantized map).
        fn get_quantized(&self, name: &str) -> Option<&crate::loader::QuantizedTensor> {
            let _ = name;
            None
        }
    }
}

pub mod cpu_kernels {
    use super::Element;

    /// Marker trait for floating-point Element types.
    pub trait Float: Element {}

    impl Float for f32 {}
    impl Float for half::f16 {}
    impl Float for half::bf16 {}
}

pub mod cpu_backend {
    pub use super::CpuBackend;
}

// Root-level re-exports
pub use backend_trait::{Backend, BackendError};
pub use crate::loader::adapter::{DType, PackedBits};
pub use crate::loader::QuantizedTensor;

// ---------------------------------------------------------------------------
// CpuBackend<E> — stub CPU backend
// ---------------------------------------------------------------------------

use crate::engine::executor::{
    AttentionTopology, BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheConfig,
    KvCacheHandle, LogitsHandle, SamplingConfig,
};
use crate::scheduler::types::{PageId, PageState, StorageKey};

/// Stub CPU backend that satisfies the phantom `Backend<E>` trait.
#[derive(Debug, Clone)]
pub struct CpuBackend<E: Element = f32> {
    _marker: std::marker::PhantomData<E>,
}

impl<E: Element> CpuBackend<E> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Element> Default for CpuBackend<E> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BERT encoder forward pass (used by embedding_forward_gpu_pure)
// ---------------------------------------------------------------------------

/// Reinterpret a slice of Element as &[f32]. Only valid when E is f32.
fn as_f32_slice<E: Element>(data: &[E]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) }
}

/// Try to get tensor data as an owned Vec<f32>. Tries each name in order.
/// Handles both native (f32) tensors and quantized (GGUF) tensors.
fn get_f32_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    names: &[&str],
) -> Result<Vec<f32>, BE> {
    for name in names {
        if let Some(t) = weights.get_tensor(name) {
            let data = as_f32_slice(t.as_slice()).to_vec();
            return Ok(data);
        }
        if let Some(qt) = weights.get_quantized(name) {
            let n: usize = qt.shape.iter().product();
            let mut out = vec![0.0f32; n];
            let blk_bytes = qt.quant_type.block_bytes();
            let blk_elems = qt.quant_type.block_size();

            // GGUF pads each row (innermost dim) to the block boundary.
            // When ne0 % block_size != 0, we must dequantize row-by-row,
            // skipping the padding blocks at the end of each row.
            // GGUF shape is [ne0, ne1, ...] where ne0 is the innermost dim (first element).
            let ne0 = if qt.shape.is_empty() { n } else { qt.shape[0] };
            let blocks_per_row = (ne0 + blk_elems - 1) / blk_elems;
            let row_data_bytes = blocks_per_row * blk_bytes;
            let needs_row_dequant = ne0 % blk_elems != 0 && qt.shape.len() >= 2;

            if needs_row_dequant {
                let n_rows: usize = qt.shape[1..].iter().product();
                let mut out_off = 0;
                let mut data_off = 0;
                // Temp buffer for one full row (including padding elements)
                let row_elems_padded = blocks_per_row * blk_elems;
                let mut row_buf = vec![0.0f32; row_elems_padded];
                for _row in 0..n_rows {
                    let row_data = &qt.data[data_off..data_off + row_data_bytes];
                    backend.dequantize(row_data, &mut row_buf, qt.quant_type)?;
                    if _row == 0 {
                        let nan_count = row_buf.iter().filter(|v| v.is_nan()).count();
                        let inf_count = row_buf.iter().filter(|v| v.is_infinite()).count();
                        eprintln!("[DEBUG get_f32_data] tensor={} row=0 ne0={} padded={} nan={} inf={} first8={:?} raw_bytes_first16={:?}",
                            name, ne0, row_elems_padded, nan_count, inf_count,
                            &row_buf[..8.min(row_elems_padded)],
                            &row_data[..16.min(row_data.len())]);
                    }
                    // Copy only the valid (non-padding) elements
                    out[out_off..out_off + ne0].copy_from_slice(&row_buf[..ne0]);
                    out_off += ne0;
                    data_off += row_data_bytes;
                }
            } else {
                eprintln!("[DEBUG get_f32_data] tensor={} direct dequant: n={} data_len={} blk_bytes={} blk_elems={} quant_type={:?}",
                    name, n, qt.data.len(), blk_bytes, blk_elems, qt.quant_type);
                backend.dequantize(&qt.data, &mut out, qt.quant_type)?;
                let nan_count = out.iter().filter(|v| v.is_nan()).count();
                if nan_count > 0 {
                    eprintln!("[DEBUG get_f32_data] tensor={} AFTER dequant: nan={} first8={:?}", name, nan_count, &out[..8.min(out.len())]);
                    // Find first NaN position
                    if let Some(pos) = out.iter().position(|v| v.is_nan()) {
                        let block_idx = pos / blk_elems;
                        let in_block = pos % blk_elems;
                        eprintln!("[DEBUG get_f32_data] first NaN at pos={} block_idx={} in_block={}", pos, block_idx, in_block);
                        // Dump the raw block bytes
                        let blk_start = block_idx * blk_bytes;
                        let blk_end = (blk_start + blk_bytes).min(qt.data.len());
                        eprintln!("[DEBUG get_f32_data] block bytes[..16]: {:?}", &qt.data[blk_start..blk_start+16.min(blk_end-blk_start)]);
                    }
                }
            }

            return Ok(out);
        }
    }
    Err(BE::Other(format!("tensor not found: {:?}", names)))
}

/// Try to get bias data as Vec<f32>. Returns zeros if not found.
fn get_bias_data<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    names: &[&str],
    size: usize,
) -> Vec<f32> {
    for name in names {
        if let Some(t) = weights.get_tensor(name) {
            return as_f32_slice(t.as_slice()).to_vec();
        }
    }
    vec![0.0; size]
}

/// Detect whether linear-layer weights need transposing before GEMM.
///
/// Returns `true` when the dequantized flat array stores weights in
/// `[out_dim, in_dim]` row-major order (SafeTensors / PyTorch convention AND
/// GGUF convention — GGUF shape `[ne0=in, ne1=out]` is ne1 rows of ne0 cols,
/// i.e. `[out, in]` in memory).
///
/// Returns `false` only for genuine ONNX layout where the flat array is
/// already `[in_dim, out_dim]`.
fn needs_weight_transpose<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
) -> bool {
    // GGUF models use blk.* names — they always need transpose because
    // GGUF shape [ne0, ne1] stores ne1 rows of ne0 elements, giving
    // [out_dim, in_dim] in memory (same as SafeTensors).
    const GGUF_PROBE: &str = "blk.0.ffn_up.weight";
    if weights.tensor_shape(GGUF_PROBE).is_some() {
        return true;
    }

    // Probe a non-square SafeTensors / ONNX weight.
    const PROBE_NAMES: &[&str] = &[
        "encoder.layer.0.intermediate.dense.weight",
        "bert.encoder.layer.0.intermediate.dense.weight",
        "model.encoder.layer.0.intermediate.dense.weight",
    ];
    for name in PROBE_NAMES {
        if let Some(shape) = weights.tensor_shape(name) {
            if shape.len() == 2 && shape[0] != shape[1] {
                // SafeTensors: [out_dim(1536), in_dim(384)] → first > second → true
                // ONNX:        [in_dim(384), out_dim(1536)] → first < second → false
                return shape[0] > shape[1];
            }
        }
    }
    // Fallback: name-existence heuristic (original behaviour).
    // SafeTensors models use "embeddings.word_embeddings.weight" style names.
    const NAMES: &[&str] = &[
        "embeddings.word_embeddings.weight",
        "bert.embeddings.word_embeddings.weight",
        "model.embeddings.word_embeddings.weight",
    ];
    NAMES.iter().any(|n| weights.get_tensor(n).is_some() || weights.get_quantized(n).is_some())
}

/// Transpose a row-major matrix [rows, cols] → [cols, rows].
fn transpose_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// JIT compilation for BERT encoder layers
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single BERT encoder layer (post-norm).
///
/// JIT compiler will auto-fuse:
/// - GemmBias + Gelu → EpilogueInjection
/// - Q/K/V GemmBias sharing input → ComputeRoot (shared pack_a)
#[cfg(target_arch = "x86_64")]
fn build_bert_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;

    // ── Graph inputs ──
    let input = g.add_tensor("input", vec![s, h], dt);

    // Attention weights + biases
    let w_q = g.add_tensor("w_q", vec![h, h], dt);
    let b_q = g.add_tensor("b_q", vec![h], dt);
    let w_k = g.add_tensor("w_k", vec![h, h], dt);
    let b_k = g.add_tensor("b_k", vec![h], dt);
    let w_v = g.add_tensor("w_v", vec![h, h], dt);
    let b_v = g.add_tensor("b_v", vec![h], dt);
    let w_o = g.add_tensor("w_o", vec![h, h], dt);
    let b_o = g.add_tensor("b_o", vec![h], dt);

    // LayerNorm 1 weights
    let ln1_w = g.add_tensor("ln1_w", vec![h], dt);
    let ln1_b = g.add_tensor("ln1_b", vec![h], dt);

    // FFN weights + biases
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let b_up = g.add_tensor("b_up", vec![inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);
    let b_down = g.add_tensor("b_down", vec![h], dt);

    // LayerNorm 2 weights
    let ln2_w = g.add_tensor("ln2_w", vec![h], dt);
    let ln2_b = g.add_tensor("ln2_b", vec![h], dt);

    g.inputs = vec![
        input, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        ln1_w, ln1_b, w_up, b_up, w_down, b_down, ln2_w, ln2_b,
    ];

    // ── Self-Attention ──

    // Q = input * W_q + b_q  [s, h] × [h, h] → [s, h]
    let q_out = g.add_tensor("q", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_q, b_q],
        vec![q_out],
        "gemm_q",
    );

    // K = input * W_k + b_k
    let k_out = g.add_tensor("k", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_k, b_k],
        vec![k_out],
        "gemm_k",
    );

    // V = input * W_v + b_v
    let v_out = g.add_tensor("v", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![input, w_v, b_v],
        vec![v_out],
        "gemm_v",
    );

    // Multi-head attention: Q[s,h], K[s,h], V[s,h] → attn_out[s,h]
    let attn_out = g.add_tensor("attn_out", vec![s, h], dt);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, head_dim },
        vec![q_out, k_out, v_out],
        vec![attn_out],
        "mha",
    );

    // Output projection + bias
    let o_out = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: h },
        vec![attn_out, w_o, b_o],
        vec![o_out],
        "gemm_o",
    );

    // Residual₁: input + o_out
    let resid1 = g.add_tensor("residual1", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_out],
        vec![resid1],
        "residual_1",
    );

    // Post-attention LayerNorm
    let normed1 = g.add_tensor("normed1", vec![s, h], dt);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![resid1, ln1_w, ln1_b],
        vec![normed1],
        "layer_norm_1",
    );

    // ── FFN ──

    // Up projection + GELU: GemmBias + Gelu → EpilogueInjection candidate
    let up_out = g.add_tensor("ffn_up", vec![s, inter], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: inter, k: h },
        vec![normed1, w_up, b_up],
        vec![up_out],
        "gemm_ffn_up",
    );

    let act_out = g.add_tensor("ffn_act", vec![s, inter], dt);
    g.add_op(
        OpKind::Gelu,
        vec![up_out],
        vec![act_out],
        "gelu",
    );

    // Down projection
    let down_out = g.add_tensor("ffn_down", vec![s, h], dt);
    g.add_op(
        OpKind::GemmBias { m: s, n: h, k: inter },
        vec![act_out, w_down, b_down],
        vec![down_out],
        "gemm_ffn_down",
    );

    // Residual₂: normed1 + down_out
    let resid2 = g.add_tensor("residual2", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![normed1, down_out],
        vec![resid2],
        "residual_2",
    );

    // Post-FFN LayerNorm
    let output = g.add_tensor("output", vec![s, h], dt);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![resid2, ln2_w, ln2_b],
        vec![output],
        "layer_norm_2",
    );

    g.outputs = vec![output];
    g
}

/// Execute a JIT-compiled BERT encoder layer.
///
/// Packs all weight tensors into a contiguous buffer matching the graph's
/// input tensor order, then calls the compiled layer function.
#[cfg(target_arch = "x86_64")]
fn execute_jit_bert_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    q_w: &[f32], q_b: &[f32],
    k_w: &[f32], k_b: &[f32],
    v_w: &[f32], v_b: &[f32],
    out_w: &[f32], out_b: &[f32],
    ln1_w: &[f32], ln1_b: &[f32],
    ffn_up_w: &[f32], ffn_up_b: &[f32],
    ffn_down_w: &[f32], ffn_down_b: &[f32],
    ln2_w: &[f32], ln2_b: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    // Pack weights contiguously in graph input order:
    // [w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, ln1_w, ln1_b,
    //  w_up, b_up, w_down, b_down, ln2_w, ln2_b]
    let weight_slices: &[&[f32]] = &[
        q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
        ln1_w, ln1_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b,
        ln2_w, ln2_b,
    ];
    let total_weight_bytes: usize = weight_slices.iter().map(|s| s.len() * 4).sum();
    let mut weights_buf = vec![0u8; total_weight_bytes];
    let mut offset = 0;
    for slice in weight_slices.iter() {
        let bytes = slice.len() * 4;
        weights_buf[offset..offset + bytes].copy_from_slice(
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, bytes) }
        );
        offset += bytes;
    }

    let scratchpad_bytes = compiled.scratchpad_bytes;
    let mut scratchpad = vec![0u8; scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr() as *const u8,
            std::ptr::null_mut(), // no KV cache for BERT
            std::ptr::null(),     // no positions
            std::ptr::null(),     // no seq_lens
            1,                    // batch_size = 1
            seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

/// Build a CompilerGraph for mean pooling: average seq_len rows into one.
fn build_mean_pool_graph(
    seq_len: usize,
    hidden: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let output = g.add_tensor("output", vec![hidden], dt);

    g.add_op(
        OpKind::MeanPool { seq_len, hidden },
        vec![input],
        vec![output],
        "mean_pool",
    );

    g.inputs = vec![input];
    g.outputs = vec![output];
    g
}

/// Full BERT encoder forward pass.
fn bert_encoder_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::Kernels;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let kern = gllm_kernels::cpu_kernels::CpuKernels::<f32>::new();
    let transpose_weights = needs_weight_transpose(weights);

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;

    // Build BERT layer graph and attempt JIT compilation.
    #[cfg(target_arch = "x86_64")]
    let jit_layer: Option<gllm_kernels::compiler::CompiledLayer> = {
        let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        match compiler.compile_graph(&graph) {
            Ok(compiled) => {
                Some(compiled)
            }
            Err(e) => {
                return Err(BE::Other(format!("BERT JIT compilation failed: {e}")));
            }
        }
    };

    // Step (a): Token embedding lookup
    // All formats store embeddings as [vocab, hidden] in row-major after dequant.
    // (GGUF shape [ne0=hidden, ne1=vocab] means vocab rows of hidden elements.)
    let word_emb = get_f32_data(
        weights, backend,
        &["embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings.weight", "model.embeddings.word_embeddings.weight", "token_embd.weight"],
    )?;
    {
        let nan_count = word_emb.iter().filter(|v| v.is_nan()).count();
        eprintln!("[DEBUG] word_emb len={} nan={} first8={:?}", word_emb.len(), nan_count, &word_emb[..8.min(word_emb.len())]);
    }
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    {
        let vocab = word_emb.len() / hidden;
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for word_embeddings (vocab {})", tok, vocab
                )));
            }
            hidden_state[s * hidden..(s + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
        }
    }
    {
        let nan_count = hidden_state.iter().filter(|v| v.is_nan()).count();
        eprintln!("[DEBUG] after word_emb lookup: hidden nan={} first8={:?}", nan_count, &hidden_state[..8.min(hidden_state.len())]);
    }

    // Step (b): Add position embeddings (positions 0..seq_len)
    // All formats: [max_pos, hidden] row-major after dequant.
    let pos_emb = get_f32_data(
        weights, backend,
        &["embeddings.position_embeddings.weight", "bert.embeddings.position_embeddings.weight", "model.embeddings.position_embeddings.weight", "position_embd.weight"],
    )?;
    {
        let max_pos = pos_emb.len() / hidden;
        for s in 0..seq_len {
            if s >= max_pos {
                return Err(BE::Other(format!(
                    "position {} out of range for position_embeddings (max {})", s, max_pos
                )));
            }
            let row = &mut hidden_state[s * hidden..(s + 1) * hidden];
            for i in 0..hidden {
                row[i] += pos_emb[s * hidden + i];
            }
        }
    }

    // Step (c): Add token_type embeddings (all type 0)
    let tt_emb = get_f32_data(
        weights, backend,
        &["embeddings.token_type_embeddings.weight", "bert.embeddings.token_type_embeddings.weight", "model.embeddings.token_type_embeddings.weight", "token_types.weight"],
    )?;
    // All formats: [num_types, hidden] row-major. Type 0 is the first row.
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            let row = &mut hidden_state[s * hidden..(s + 1) * hidden];
            for i in 0..hidden {
                row[i] += tt_emb[i]; // first row = type 0
            }
        }
    }
    {
        let nan_count = hidden_state.iter().filter(|v| v.is_nan()).count();
        eprintln!("[DEBUG] after tt_emb: hidden nan={} first8={:?}", nan_count, &hidden_state[..8.min(hidden_state.len())]);
    }

    // Step (d): Embedding LayerNorm
    // Step (d): Embedding LayerNorm
    let emb_ln_w = get_f32_data(
        weights, backend,
        &["embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.weight", "model.embeddings.LayerNorm.weight", "token_embd_norm.weight"],
    )?;
    let emb_ln_b = get_bias_data(
        weights,
        &["embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.bias", "model.embeddings.LayerNorm.bias", "token_embd_norm.bias"],
        hidden,
    );
    {
        let mut normed = vec![0.0f32; hidden];
        for s in 0..seq_len {
            let row = &hidden_state[s * hidden..(s + 1) * hidden];
            kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
            hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
        }
    }
    {
        let nan_count = hidden_state.iter().filter(|v| v.is_nan()).count();
        eprintln!("[DEBUG] after emb_ln: hidden nan={} first8={:?}", nan_count, &hidden_state[..8.min(hidden_state.len())]);
    }

    // Step (e): Encoder layers
    #[allow(unused_mut, unused_variables)]
    let mut buf_out = vec![0.0f32; seq_len * hidden];
    #[allow(unused_mut, unused_variables)]
    let mut buf_inter = vec![0.0f32; seq_len * inter];
    #[allow(unused_mut, unused_variables)]
    let mut normed = vec![0.0f32; hidden];

    for layer in 0..num_layers {
        let st_prefix = format!("encoder.layer.{layer}");
        let gg_prefix = format!("blk.{layer}");
        let onnx_bert_prefix = format!("bert.encoder.layer.{layer}");
        let onnx_model_prefix = format!("model.encoder.layer.{layer}");

        // ── JIT fast path ──
        #[cfg(target_arch = "x86_64")]
        if let Some(ref compiled) = jit_layer {
            let q_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.attention.self.query.weight"),
                &format!("{onnx_bert_prefix}.attention.self.query.weight"),
                &format!("{onnx_model_prefix}.attention.self.query.weight"),
                &format!("{gg_prefix}.attn_q.weight"),
            ])?;
            let q_b = get_bias_data(weights, &[
                &format!("{st_prefix}.attention.self.query.bias"),
                &format!("{onnx_bert_prefix}.attention.self.query.bias"),
                &format!("{onnx_model_prefix}.attention.self.query.bias"),
                &format!("{gg_prefix}.attn_q.bias"),
            ], hidden);
            let k_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.attention.self.key.weight"),
                &format!("{onnx_bert_prefix}.attention.self.key.weight"),
                &format!("{onnx_model_prefix}.attention.self.key.weight"),
                &format!("{gg_prefix}.attn_k.weight"),
            ])?;
            let k_b = get_bias_data(weights, &[
                &format!("{st_prefix}.attention.self.key.bias"),
                &format!("{onnx_bert_prefix}.attention.self.key.bias"),
                &format!("{onnx_model_prefix}.attention.self.key.bias"),
                &format!("{gg_prefix}.attn_k.bias"),
            ], hidden);
            let v_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.attention.self.value.weight"),
                &format!("{onnx_bert_prefix}.attention.self.value.weight"),
                &format!("{onnx_model_prefix}.attention.self.value.weight"),
                &format!("{gg_prefix}.attn_v.weight"),
            ])?;
            let v_b = get_bias_data(weights, &[
                &format!("{st_prefix}.attention.self.value.bias"),
                &format!("{onnx_bert_prefix}.attention.self.value.bias"),
                &format!("{onnx_model_prefix}.attention.self.value.bias"),
                &format!("{gg_prefix}.attn_v.bias"),
            ], hidden);
            let out_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.attention.output.dense.weight"),
                &format!("{onnx_bert_prefix}.attention.output.dense.weight"),
                &format!("{onnx_model_prefix}.attention.output.dense.weight"),
                &format!("{gg_prefix}.attn_output.weight"),
            ])?;
            let out_b = get_bias_data(weights, &[
                &format!("{st_prefix}.attention.output.dense.bias"),
                &format!("{onnx_bert_prefix}.attention.output.dense.bias"),
                &format!("{onnx_model_prefix}.attention.output.dense.bias"),
                &format!("{gg_prefix}.attn_output.bias"),
            ], hidden);
            let ln1_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.attention.output.LayerNorm.weight"),
                &format!("{onnx_bert_prefix}.attention.output.LayerNorm.weight"),
                &format!("{onnx_model_prefix}.attention.output.LayerNorm.weight"),
                &format!("{gg_prefix}.attn_output_norm.weight"),
            ])?;
            let ln1_b = get_bias_data(weights, &[
                &format!("{st_prefix}.attention.output.LayerNorm.bias"),
                &format!("{onnx_bert_prefix}.attention.output.LayerNorm.bias"),
                &format!("{onnx_model_prefix}.attention.output.LayerNorm.bias"),
                &format!("{gg_prefix}.attn_output_norm.bias"),
            ], hidden);
            let ffn_up_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.intermediate.dense.weight"),
                &format!("{onnx_bert_prefix}.intermediate.dense.weight"),
                &format!("{onnx_model_prefix}.intermediate.dense.weight"),
                &format!("{gg_prefix}.ffn_up.weight"),
            ])?;
            let ffn_up_b = get_bias_data(weights, &[
                &format!("{st_prefix}.intermediate.dense.bias"),
                &format!("{onnx_bert_prefix}.intermediate.dense.bias"),
                &format!("{onnx_model_prefix}.intermediate.dense.bias"),
                &format!("{gg_prefix}.ffn_up.bias"),
            ], inter);
            let ffn_down_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.output.dense.weight"),
                &format!("{onnx_bert_prefix}.output.dense.weight"),
                &format!("{onnx_model_prefix}.output.dense.weight"),
                &format!("{gg_prefix}.ffn_down.weight"),
            ])?;
            let ffn_down_b = get_bias_data(weights, &[
                &format!("{st_prefix}.output.dense.bias"),
                &format!("{onnx_bert_prefix}.output.dense.bias"),
                &format!("{onnx_model_prefix}.output.dense.bias"),
                &format!("{gg_prefix}.ffn_down.bias"),
            ], hidden);
            let ln2_w = get_f32_data(weights, backend, &[
                &format!("{st_prefix}.output.LayerNorm.weight"),
                &format!("{onnx_bert_prefix}.output.LayerNorm.weight"),
                &format!("{onnx_model_prefix}.output.LayerNorm.weight"),
                &format!("{gg_prefix}.layer_output_norm.weight"),
            ])?;
            let ln2_b = get_bias_data(weights, &[
                &format!("{st_prefix}.output.LayerNorm.bias"),
                &format!("{onnx_bert_prefix}.output.LayerNorm.bias"),
                &format!("{onnx_model_prefix}.output.LayerNorm.bias"),
                &format!("{gg_prefix}.layer_output_norm.bias"),
            ], hidden);


            // SafeTensors stores [out, in]; our GEMM expects [in, out] (B in C=A@B)
            let (q_w, k_w, v_w, out_w, ffn_up_w, ffn_down_w) = if transpose_weights {
                (
                    transpose_f32(&q_w, hidden, hidden),
                    transpose_f32(&k_w, hidden, hidden),
                    transpose_f32(&v_w, hidden, hidden),
                    transpose_f32(&out_w, hidden, hidden),
                    transpose_f32(&ffn_up_w, inter, hidden),
                    transpose_f32(&ffn_down_w, hidden, inter),
                )
            } else {
                (q_w, k_w, v_w, out_w, ffn_up_w, ffn_down_w)
            };

            if layer == 0 {
                eprintln!("[DEBUG] transpose_weights={} q_w len={} nan={} first8={:?}",
                    transpose_weights, q_w.len(),
                    q_w.iter().filter(|v| v.is_nan()).count(),
                    &q_w[..8.min(q_w.len())]);
                eprintln!("[DEBUG] q_b len={} nan={} first8={:?}",
                    q_b.len(),
                    q_b.iter().filter(|v| v.is_nan()).count(),
                    &q_b[..8.min(q_b.len())]);
                eprintln!("[DEBUG] ffn_up_w len={} (expect {}x{}={}) nan={} first8={:?}",
                    ffn_up_w.len(), hidden, inter, hidden*inter,
                    ffn_up_w.iter().filter(|v| v.is_nan()).count(),
                    &ffn_up_w[..8.min(ffn_up_w.len())]);
                eprintln!("[DEBUG] ffn_down_w len={} (expect {}x{}={}) nan={} first8={:?}",
                    ffn_down_w.len(), inter, hidden, inter*hidden,
                    ffn_down_w.iter().filter(|v| v.is_nan()).count(),
                    &ffn_down_w[..8.min(ffn_down_w.len())]);
            }

            let mut layer_out = vec![0.0f32; seq_len * hidden];
            execute_jit_bert_layer(
                compiled,
                &hidden_state,
                &q_w, &q_b, &k_w, &k_b, &v_w, &v_b,
                &out_w, &out_b, &ln1_w, &ln1_b,
                &ffn_up_w, &ffn_up_b, &ffn_down_w, &ffn_down_b,
                &ln2_w, &ln2_b,
                seq_len,
                &mut layer_out,
            );

            {
                let nan_count = layer_out.iter().filter(|v| v.is_nan()).count();
                let inf_count = layer_out.iter().filter(|v| v.is_infinite()).count();
                eprintln!("[DEBUG] layer {} JIT output: nan={} inf={} first8={:?}", layer, nan_count, inf_count, &layer_out[..8.min(layer_out.len())]);
            }

            hidden_state.copy_from_slice(&layer_out);
            continue;
        }

        // No fallback — JIT is mandatory
        return Err(BE::Other("BERT encoder requires JIT compilation".to_string()));
    }
    // Step (f): Mean pooling over all tokens (JIT-compiled SIMD, no fallback)
    let mut pooled = vec![0.0f32; hidden];
    {
        let pool_graph = build_mean_pool_graph(seq_len, hidden);
        let mut pool_compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = pool_compiler.compile_graph(&pool_graph)
            .map_err(|e| BE::Other(format!("MeanPool JIT compilation failed: {e}")))?;
        let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
        unsafe {
            compiled.execute(
                hidden_state.as_ptr() as *const u8,
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                seq_len,
                pooled.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
            );
        }
    }

    {
        let nan_count = pooled.iter().filter(|v| v.is_nan()).count();
        let inf_count = pooled.iter().filter(|v| v.is_infinite()).count();
        eprintln!("[DEBUG] final pooled: nan={} inf={} first8={:?}", nan_count, inf_count, &pooled[..8.min(pooled.len())]);
    }

    Ok(pooled)
}

impl<E: Element> backend_trait::Backend<E> for CpuBackend<E> {
    type Tensor = Vec<E>;

    fn alloc_kv_cache(&self, _config: &KvCacheConfig) -> Result<KvCacheHandle, BE> {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(KvCacheHandle(id))
    }

    fn batch_forward_gpu_pure(
        &self,
        input: &BatchInput,
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        _kv_caches: &mut [KvCacheHandle],
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<LogitsHandle>, BE> {
        Ok(input
            .sequences
            .iter()
            .map(|_| LogitsHandle {
                data: vec![0.0; config.vocab_size],
            })
            .collect())
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsHandle,
        _topology: &AttentionTopology,
        _vocab_size: usize,
        _sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        let token = logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        Ok(vec![token])
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        bert_encoder_forward(self, tokens, weights, config)
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        // Reranker uses the same BERT encoder as embedding, then applies classifier head
        bert_encoder_forward(self, tokens, weights, config)
    }

    fn get_memory_pressure(&self) -> Result<f32, BE> {
        Ok(0.0)
    }

    fn swap_out_pages(
        &self,
        _handle: &mut KvCacheHandle,
        _mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        Ok(())
    }

    fn swap_in_pages(
        &self,
        _handle: &mut KvCacheHandle,
        _mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        Ok(())
    }

    fn get_page_states(
        &self,
        _handle: &KvCacheHandle,
    ) -> Result<Vec<(PageId, PageState)>, BE> {
        Ok(Vec::new())
    }

    fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BE> {
        Ok(data.to_vec())
    }

    fn quantized_matmul(
        &self,
        weight_blocks: &[u8],
        input: &[E],
        output: &mut [E],
        quant_type: backend_trait::QuantType,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BE> {
        use gllm_kernels::quant::QuantType::*;
        use gllm_kernels::Kernels;
        let kern = gllm_kernels::cpu_kernels::CpuKernels::<E>::new();
        match quant_type {
            // K-Quant family
            Q2K | Q3K | Q4K | Q5K | Q6K | Q8K => {
                kern.kquant_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // Classic GGML family
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => {
                kern.classic_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // IQ family
            IQ1S | IQ1M | IQ2XXS | IQ2XS | IQ2S | IQ3XXS | IQ3S | IQ4NL | IQ4XS => {
                kern.iq_matmul(weight_blocks, input, output, quant_type, m, n, k);
            }
            // External formats not supported through this unified path
            AWQ4 | GPTQ4 | Squeeze => {
                return Err(BE::Unimplemented("quantized_matmul: external formats (AWQ4/GPTQ4/Squeeze) require dedicated API"));
            }
        }
        Ok(())
    }

    fn dequantize(
        &self,
        block_data: &[u8],
        output: &mut [f32],
        quant_type: backend_trait::QuantType,
    ) -> Result<(), BE> {
        use gllm_kernels::quant::QuantType::*;
        use gllm_kernels::Kernels;
        let kern = gllm_kernels::cpu_kernels::CpuKernels::<E>::new();

        // Each dequant_* kernel decodes a single block. We must loop over
        // all blocks in the tensor, advancing the byte and output pointers.
        let blk_elems = quant_type.block_size();
        let blk_bytes = quant_type.block_bytes();

        macro_rules! decode_all_blocks {
            ($method:ident) => {
                for (blk_in, blk_out) in block_data
                    .chunks_exact(blk_bytes)
                    .zip(output.chunks_exact_mut(blk_elems))
                {
                    kern.$method(blk_in, blk_out);
                }
            };
        }

        match quant_type {
            // K-Quant family
            Q2K => decode_all_blocks!(dequant_q2_k),
            Q3K => decode_all_blocks!(dequant_q3_k),
            Q4K => decode_all_blocks!(dequant_q4_k),
            Q5K => decode_all_blocks!(dequant_q5_k),
            Q6K => decode_all_blocks!(dequant_q6_k),
            Q8K => decode_all_blocks!(dequant_q8_k),
            // Classic GGML family
            Q4_0 => decode_all_blocks!(dequant_q4_0),
            Q4_1 => decode_all_blocks!(dequant_q4_1),
            Q5_0 => decode_all_blocks!(dequant_q5_0),
            Q5_1 => decode_all_blocks!(dequant_q5_1),
            Q8_0 => decode_all_blocks!(dequant_q8_0),
            Q8_1 => decode_all_blocks!(dequant_q8_1),
            // IQ family
            IQ1S => decode_all_blocks!(dequant_iq1_s),
            IQ1M => decode_all_blocks!(dequant_iq1_m),
            IQ2XXS => decode_all_blocks!(dequant_iq2_xxs),
            IQ2XS => decode_all_blocks!(dequant_iq2_xs),
            IQ2S => decode_all_blocks!(dequant_iq2_s),
            IQ3XXS => decode_all_blocks!(dequant_iq3_xxs),
            IQ3S => decode_all_blocks!(dequant_iq3_s),
            IQ4NL => decode_all_blocks!(dequant_iq4_nl),
            IQ4XS => decode_all_blocks!(dequant_iq4_xs),
            // External formats not supported through this unified path
            AWQ4 | GPTQ4 | Squeeze => {
                return Err(BE::Unimplemented("dequantize: external formats (AWQ4/GPTQ4/Squeeze) require dedicated API"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CudaBackend<E> — stub CUDA backend (always fails to init)
// ---------------------------------------------------------------------------

/// Stub CUDA backend. `new()` always returns `None` since we have no real CUDA.
#[derive(Debug, Clone)]
pub struct CudaBackend<E: Element = f32> {
    _marker: std::marker::PhantomData<E>,
}

impl<E: Element> CudaBackend<E> {
    /// Attempt to create a CUDA backend on the given device ordinal.
    /// Always returns `None` in this stub.
    pub fn new(_device: usize) -> Option<Self> {
        None
    }
}

impl<E: Element> backend_trait::Backend<E> for CudaBackend<E> {
    type Tensor = Vec<E>;

    fn alloc_kv_cache(&self, _config: &KvCacheConfig) -> Result<KvCacheHandle, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn batch_forward_gpu_pure(
        &self,
        _input: &BatchInput,
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        _kv_caches: &mut [KvCacheHandle],
        _config: &GeneratorForwardConfig,
    ) -> Result<Vec<LogitsHandle>, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn sample_from_tensor(
        &self,
        _logits: &LogitsHandle,
        _topology: &AttentionTopology,
        _vocab_size: usize,
        _sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn embedding_forward_gpu_pure(
        &self,
        _tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        _config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn rerank_forward_gpu_pure(
        &self,
        _tokens: &[u32],
        _topology: &AttentionTopology,
        _weights: &dyn backend_trait::TensorLookup<E, Self>,
        _config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn get_memory_pressure(&self) -> Result<f32, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn swap_out_pages(
        &self,
        _handle: &mut KvCacheHandle,
        _mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn swap_in_pages(
        &self,
        _handle: &mut KvCacheHandle,
        _mappings: &[(PageId, StorageKey)],
    ) -> Result<(), BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn get_page_states(
        &self,
        _handle: &KvCacheHandle,
    ) -> Result<Vec<(PageId, PageState)>, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }

    fn upload_weights(&self, _data: &[E]) -> Result<Self::Tensor, BE> {
        Err(BE::Unimplemented("cuda stub"))
    }
}
