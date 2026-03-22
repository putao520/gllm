//! Generic GPU helper functions that eliminate CUDA/HIP/Metal triplication.
//!
//! All functions are parameterized over `B: Backend<E>` so a single implementation
//! serves all three GPU backends. The logic is byte-for-byte identical to the
//! original per-backend versions — only the type parameter differs.

use super::backend_trait::{self, Backend};

use super::Element;
use crate::engine::executor::BackendError as BE;
use gllm_kernels::types::DType;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::KvCacheConfig;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::KvCacheHandle;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::scheduler::types::{PageId, PageState, StorageKey};

// ---------------------------------------------------------------------------
// GpuBackendOps — trait abstracting device/error/store differences
// ---------------------------------------------------------------------------

/// Trait that abstracts the 3 backend-specific differences:
/// 1. Device access (for alloc/htod/dtoh)
/// 2. Error variant construction (BE::Cuda/Hip/Metal)
/// 3. Swap store and KV metadata store access
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) trait GpuBackendOps {
    /// Raw device→host copy: src_ptr is a device pointer, dst is host slice.
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE>;
    /// Raw host→device copy: src is host slice, dst_ptr is a device pointer.
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE>;
    /// Allocate device memory, return device pointer.
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE>;
    fn gpu_error(&self, msg: String) -> BE;
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore;
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore;
}

#[cfg(feature = "cuda")]
impl GpuBackendOps for super::cuda_backend::CudaBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        let res = unsafe {
            (self.device.driver().cuMemcpyDtoH_v2)(
                dst.as_mut_ptr() as *mut _,
                src_ptr,
                dst.len(),
            )
        };
        if res != 0 { Err(BE::Cuda(format!("cuMemcpyDtoH failed: {res}"))) } else { Ok(()) }
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        let res = unsafe {
            (self.device.driver().cuMemcpyHtoD_v2)(
                dst_ptr,
                src.as_ptr() as *const _,
                src.len(),
            )
        };
        if res != 0 { Err(BE::Cuda(format!("cuMemcpyHtoD failed: {res}"))) } else { Ok(()) }
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::GpuDevice;
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device.alloc(bytes).map_err(|e| BE::Cuda(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Cuda(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

#[cfg(feature = "hip")]
impl GpuBackendOps for super::hip_backend::HipBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        self.device.dtoh_raw(src_ptr, dst)
            .map_err(|e| BE::Hip(format!("dtoh_raw failed: {e}")))
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        self.device.htod_raw(src, dst_ptr)
            .map_err(|e| BE::Hip(format!("htod_raw failed: {e}")))
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device.alloc(bytes).map_err(|e| BE::Hip(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Hip(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuBackendOps for super::metal_backend::MetalBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        self.device().dtoh_raw(src_ptr, dst)
            .map_err(|e| BE::Metal(format!("dtoh_raw failed: {e}")))
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        self.device().htod_raw(src, dst_ptr)
            .map_err(|e| BE::Metal(format!("htod_raw failed: {e}")))
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device().alloc(bytes).map_err(|e| BE::Metal(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Metal(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

/// Get tensor data as f32, trying quantized dequant first, then f32 fallback.
///
/// Replaces `get_f32_data_cuda` / `get_f32_data_hip` / `get_f32_data_metal`.
pub(super) fn get_f32_data_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    _backend: &B,
    aliases: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    // Try quantized path first
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut out = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes)
                .zip(out.chunks_exact_mut(blk_elems))
            {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4NL => kern.dequant_iq4_nl(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4XS => kern.dequant_iq4_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1S => kern.dequant_iq1_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1M => kern.dequant_iq1_m(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XXS => kern.dequant_iq2_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XS => kern.dequant_iq2_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2S => kern.dequant_iq2_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3XXS => kern.dequant_iq3_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3S => kern.dequant_iq3_s(blk_in, blk_out),
                    _ => {
                        return Err(BE::Other(format!(
                            "Unsupported quantization type {:?} for weight {:?}",
                            qt.quant_type, name.as_ref()
                        )));
                    }
                }
            }
            return Ok(out);
        }
    }

    // Fall back to f32 tensor
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = tensor.as_ref();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const f32,
                    std::mem::size_of_val(e_slice) / 4,
                )
            };
            return Ok(slice.to_vec());
        }
    }

    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Get tensor data as raw bytes in the target dtype.
///
/// Quantized weights: dequant to f32 then convert to target dtype.
/// Non-quantized weights: reinterpret raw bytes (assumes storage matches dtype_size).
/// ARCH-DTYPE-ADAPTIVE: returns bytes in model's native dtype, not always f32.
pub(super) fn get_typed_data_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    _backend: &B,
    aliases: &[impl AsRef<str>],
    target_dtype: DType,
) -> Result<Vec<u8>, BE> {
    // Try quantized path first — dequant to f32, then convert to target dtype
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut f32_buf = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes)
                .zip(f32_buf.chunks_exact_mut(blk_elems))
            {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4NL => kern.dequant_iq4_nl(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ4XS => kern.dequant_iq4_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1S => kern.dequant_iq1_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ1M => kern.dequant_iq1_m(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XXS => kern.dequant_iq2_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2XS => kern.dequant_iq2_xs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ2S => kern.dequant_iq2_s(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3XXS => kern.dequant_iq3_xxs(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::IQ3S => kern.dequant_iq3_s(blk_in, blk_out),
                    _ => {
                        return Err(BE::Other(format!(
                            "Unsupported quantization type {:?} for weight {:?}",
                            qt.quant_type, name.as_ref()
                        )));
                    }
                }
            }
            return Ok(super::jit_helpers::pack_weights_typed(&[&f32_buf], target_dtype));
        }
    }

    // Non-quantized: return raw bytes in storage dtype
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = tensor.as_ref();
            let raw_bytes = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const u8,
                    std::mem::size_of_val(e_slice),
                )
            };
            let storage_elem_bytes = std::mem::size_of_val(e_slice) / e_slice.len().max(1);
            if storage_elem_bytes == target_dtype.size_bytes() {
                return Ok(raw_bytes.to_vec());
            }
            // Storage dtype differs from target — convert via f32
            let f32_slice = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const f32,
                    std::mem::size_of_val(e_slice) / 4,
                )
            };
            return Ok(super::jit_helpers::pack_weights_typed(&[f32_slice], target_dtype));
        }
    }

    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Transpose raw bytes in-place for a given dtype.
///
/// Operates on raw byte buffer where each element is `elem_bytes` wide.
fn transpose_typed_bytes(data: &[u8], rows: usize, cols: usize, elem_bytes: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols * elem_bytes);
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = (r * cols + c) * elem_bytes;
            let dst = (c * rows + r) * elem_bytes;
            out[dst..dst + elem_bytes].copy_from_slice(&data[src..src + elem_bytes]);
        }
    }
    out
}

/// Load decoder layer weights as raw bytes in the target dtype.
///
/// ARCH-DTYPE-ADAPTIVE: weights are loaded in model's native dtype, not always f32.
pub(super) fn load_decoder_layer_weights_gpu_typed<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    layer: usize,
    hidden: usize,
    q_dim: usize,
    kv_hidden: usize,
    inter: usize,
    dtype: DType,
) -> Result<std::collections::HashMap<String, Vec<u8>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_typed_data_gpu(weights, backend, $aliases, dtype)?);
        };
    }

    load!("attn_norm_w", &crate::weight_names::layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")));
    load!("w_q", &crate::weight_names::layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")));
    load!("w_k", &crate::weight_names::layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")));
    load!("w_v", &crate::weight_names::layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")));
    load!("w_o", &crate::weight_names::layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")));
    load!("ffn_norm_w", &crate::weight_names::layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")));
    load!("w_gate", &crate::weight_names::layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")));
    load!("w_up", &crate::weight_names::layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")));
    load!("w_down", &crate::weight_names::layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")));

    if needs_weight_transpose_gpu(weights) {
        let eb = dtype.size_bytes();
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_typed_bytes(data, rows, cols, eb);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", q_dim, hidden);
        t("w_k", kv_hidden, hidden);
        t("w_v", kv_hidden, hidden);
        t("w_o", hidden, q_dim);
        t("w_gate", inter, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Load BERT encoder layer weights as raw bytes in the target dtype.
///
/// ARCH-DTYPE-ADAPTIVE: weights are loaded in model's native dtype, not always f32.
pub(super) fn load_bert_layer_weights_gpu_typed<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
    dtype: DType,
) -> Result<std::collections::HashMap<String, Vec<u8>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_typed_data_gpu(weights, backend, $aliases, dtype)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            // Bias: get as f32 then convert to target dtype
            let f32_data = get_bias_data_gpu(weights, $aliases, $size);
            m.insert($graph_name.to_string(), super::jit_helpers::pack_weights_typed(&[&f32_data], dtype));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    if transpose {
        let eb = dtype.size_bytes();
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_typed_bytes(data, rows, cols, eb);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Get bias data (zeros if not found).
///
/// Replaces `get_bias_data_cuda` / `get_bias_data_hip` / `get_bias_data_metal`.
pub(super) fn get_bias_data_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    aliases: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let e_slice: &[E] = tensor.as_ref();
            let slice = unsafe {
                std::slice::from_raw_parts(
                    e_slice.as_ptr() as *const f32,
                    std::mem::size_of_val(e_slice) / 4,
                )
            };
            return slice.to_vec();
        }
    }
    vec![0.0f32; size]
}

/// Check if weights need transposition (SafeTensors stores [out, in]).
///
/// Replaces `needs_weight_transpose_cuda` / `_hip` / `_metal`.
pub(super) fn needs_weight_transpose_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
) -> bool {
    crate::weight_names::has_any_embedding_weight(|n| weights.get_tensor(n).is_some())
}

/// CPU-side token embedding lookup for GPU backends.
///
/// Replaces `embed_tokens_cpu` / `embed_tokens_cpu_hip` / `embed_tokens_cpu_metal`.
pub(super) fn embed_tokens_gpu<E: Element, B: Backend<E>>(
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    config: &crate::engine::executor::GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    let hidden = config.hidden_size;
    let _eps = config.norm_eps;

    // Word embeddings
    let word_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")),
    )?;

    let seq_len = tokens.len();
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
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

    // Position embeddings
    let pos_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")),
    )?;
    let max_pos = pos_emb.len() / hidden;
    for s in 0..seq_len {
        if s >= max_pos {
            return Err(BE::Other(format!(
                "position {} out of range for position_embeddings (max {})", s, max_pos
            )));
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let added = super::jit_helpers::jit_add(&hidden_state, &pos_emb[..seq_len * hidden],
            super::jit_helpers::computation_dtype_from_config(config),
        )
            .map_err(|e| BE::Other(format!("pos embed add JIT failed: {e}")))?;
        hidden_state.copy_from_slice(&added);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { return Err(BE::Other("pos embed add JIT requires x86_64 or aarch64".to_string())); }

    // Token type embeddings (type 0)
    let tt_emb = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")),
    )?;
    if tt_emb.len() >= hidden {
        // Broadcast tt_emb[hidden] across all seq positions → [seq_len, hidden]
        let tt_broadcast: Vec<f32> = tt_emb[..hidden].iter().cloned().cycle().take(seq_len * hidden).collect();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let added = super::jit_helpers::jit_add(&hidden_state, &tt_broadcast,
                super::jit_helpers::computation_dtype_from_config(config),
            )
                .map_err(|e| BE::Other(format!("token type embed add JIT failed: {e}")))?;
            hidden_state.copy_from_slice(&added);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return Err(BE::Other("token type embed add JIT requires x86_64 or aarch64".to_string())); }
    }

    // Embedding LayerNorm
    let emb_ln_w = get_f32_data_gpu(
        weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")),
    )?;
    let emb_ln_b = get_bias_data_gpu(
        weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")),
        hidden,
    );
    {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            hidden_state = super::jit_helpers::jit_layer_norm(&hidden_state, &emb_ln_w, &emb_ln_b, seq_len, hidden,
                super::jit_helpers::computation_dtype_from_config(config),
            )
                .map_err(|e| BE::Other(format!("embedding LayerNorm JIT failed: {e}")))?;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        compile_error!("embedding LayerNorm requires JIT support (x86_64 or aarch64)");
    }

    Ok(hidden_state)
}

/// Load decoder layer weights into a HashMap for JIT execution.
///
/// Replaces `load_decoder_layer_weights_cuda` / `_hip` / `_metal`.
pub(super) fn load_decoder_layer_weights_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    layer: usize,
    hidden: usize,
    q_dim: usize,
    kv_hidden: usize,
    inter: usize,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_gpu(weights, backend, $aliases)?);
        };
    }

    load!("attn_norm_w", &crate::weight_names::layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")));
    load!("w_q", &crate::weight_names::layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")));
    load!("w_k", &crate::weight_names::layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")));
    load!("w_v", &crate::weight_names::layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")));
    load!("w_o", &crate::weight_names::layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")));
    load!("ffn_norm_w", &crate::weight_names::layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")));
    load!("w_gate", &crate::weight_names::layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")));
    load!("w_up", &crate::weight_names::layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")));
    load!("w_down", &crate::weight_names::layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")));

    if needs_weight_transpose_gpu(weights) {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = crate::compat::weight_helpers::transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", q_dim, hidden);
        t("w_k", kv_hidden, hidden);
        t("w_v", kv_hidden, hidden);
        t("w_o", hidden, q_dim);
        t("w_gate", inter, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Load BERT encoder layer weights into a HashMap for JIT execution.
///
/// Replaces `load_bert_layer_weights_cuda` / `_hip` / `_metal`.
pub(super) fn load_bert_layer_weights_gpu<E: Element, B: Backend<E>>(
    weights: &dyn backend_trait::TensorLookup<E, B>,
    backend: &B,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_gpu(weights, backend, $aliases)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            m.insert($graph_name.to_string(), get_bias_data_gpu(weights, $aliases, $size));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    if transpose {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = crate::compat::weight_helpers::transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

// ---------------------------------------------------------------------------
// Generic GPU ops using GpuBackendOps trait
// ---------------------------------------------------------------------------

/// GPU alloc_kv_cache: allocate GPU buffer for KV cache and register metadata.
///
/// Replaces `cuda_alloc_kv_cache` / `hip_alloc_kv_cache` / `metal_alloc_kv_cache`.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_alloc_kv_cache(
    backend: &dyn GpuBackendOps,
    config: &KvCacheConfig,
) -> Result<KvCacheHandle, BE> {
    use gllm_kernels::gpu::GpuBuffer;

    let total_bytes = config.num_layers * 2
        * config.num_heads * config.max_seq_len * config.head_dim * config.dtype_size;

    let ptr = backend.raw_alloc(total_bytes)
        .map_err(|e| backend.gpu_error(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    let meta = super::gpu_compile::GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// GPU sample_from_tensor: download logits to CPU and sample.
///
/// Replaces `cuda_sample_from_tensor` / `hip_sample_from_tensor` / `metal_sample_from_tensor`.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_sample_from_tensor(
    logits: &crate::engine::executor::LogitsHandle,
    _topology: &crate::engine::executor::AttentionTopology,
    vocab_size: usize,
    sampling: &crate::engine::executor::SamplingConfig,
) -> Result<Vec<u32>, BE> {
    Ok(super::gpu_compile::sample_logits_cpu(&logits.data, vocab_size, sampling))
}

/// GPU swap_out_pages: download page data from GPU to host swap store.
///
/// Replaces `cuda_swap_out_pages` / `hip_swap_out_pages` / `metal_swap_out_pages`.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_swap_out_pages(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let total_page_bytes = meta.num_layers * meta.num_kv_heads * page_slice_bytes * 2;

    let mut swap_store = backend.swap_store().lock()
        .map_err(|e| backend.gpu_error(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(backend.gpu_error(format!(
                "swap_out: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut host_buf = vec![0u8; total_page_bytes];
        let mut dst_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let src_ptr = half_base + src_offset;

                    backend.raw_dtoh(src_ptr, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes])
                        .map_err(|e| backend.gpu_error(format!("swap_out dtoh failed: {e}")))?;

                    dst_offset += page_slice_bytes;
                }
            }
        }

        swap_store.insert(storage_key, host_buf);
    }

    Ok(())
}

/// GPU swap_in_pages: upload page data from host swap store to GPU.
///
/// Replaces `cuda_swap_in_pages` / `hip_swap_in_pages` / `metal_swap_in_pages`.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_swap_in_pages(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;

    let mut swap_store = backend.swap_store().lock()
        .map_err(|e| backend.gpu_error(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let host_buf = swap_store.remove(&storage_key)
            .ok_or_else(|| backend.gpu_error(format!("swap_in: storage key {} not found in swap store", storage_key)))?;

        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(backend.gpu_error(format!(
                "swap_in: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut src_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let dst_ptr = half_base + dst_offset_gpu;

                    backend.raw_htod(&host_buf[src_offset..src_offset + actual_slice_bytes], dst_ptr)
                        .map_err(|e| backend.gpu_error(format!("swap_in htod failed: {e}")))?;

                    src_offset += page_slice_bytes;
                }
            }
        }
    }

    Ok(())
}

/// GPU get_page_states: return page states based on metadata.
///
/// Replaces `cuda_get_page_states` / `hip_get_page_states` / `metal_get_page_states`.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_get_page_states(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
) -> Result<Vec<(PageId, PageState)>, BE> {
    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let active = meta.active_pages();
    let total = meta.total_pages();
    let mut states = Vec::with_capacity(total);
    for page_id in 0..total {
        let state = if page_id < active { PageState::Active } else { PageState::Free };
        states.push((page_id, state));
    }
    Ok(states)
}
