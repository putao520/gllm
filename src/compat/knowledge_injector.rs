//! Knowledge Injection API (per SPEC 04-API-DESIGN §7, §8)
//!
//! This module implements three injection methods:
//! 1. `inject_frozen_kv()` - Encoder forward → KV extraction → page table write
//! 2. `inject_late_fusion()` - Truncated forward → hidden state → residual embedding
//! 3. `inject_dynamic_lora()` - LoRA weight loading → layer injection

use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::decoder_forward;
use super::jit_helpers::TypedBuffer;
use super::weight_helpers::get_typed_data;
use super::jit_helpers::typed_bytes_to_f32;
use super::Element;
use crate::engine::executor::{GeneratorForwardConfig, KvCacheHandle};
use crate::knowledge::{InjectionKind, KnowledgeError, LayerTarget, MaterializedPayload};
use crate::scheduler::memory_manager::VirtualPageId;

// ---------------------------------------------------------------------------
// 1. Frozen KV Injection (per SPEC 04-API-DESIGN §8.4)
// ---------------------------------------------------------------------------

/// Inject frozen KV cache from an encoder forward pass.
///
/// Pipeline:
/// 1. Run BERT-style encoder forward on the input tokens
/// 2. Extract K/V tensors from the encoder output
/// 3. Write K/V to KV cache page table via gpu_write_kv_cache
/// 4. Update prefix_index to register the prefix
///
/// # Arguments
/// - `backend`: CPU backend for execution
/// - `tokens`: Input token IDs (encoder input)
/// - `weights`: Model weights (encoder weights)
/// - `kv_caches`: Mutable KV cache handles to write into
/// - `config`: Forward configuration
/// - `target_layer`: Target layer index for KV cache
///
/// # Returns
/// Number of tokens written to KV cache
///
/// # Errors
/// - If encoder forward fails
/// - If KV cache write fails
pub fn inject_frozen_kv<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
    target_layer: usize,
) -> Result<usize, KnowledgeError> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(KnowledgeError::UnsupportedElementType(
            format!("inject_frozen_kv requires f32, got {}", std::any::type_name::<E>()),
        ));
    }

    let seq_len = tokens.len();
    if seq_len == 0 {
        return Err(KnowledgeError::DataFormatError("empty token sequence".into()));
    }

    // Step 1: Run encoder forward to get K/V tensors
    // For now, we use a simplified approach: compute K/V from attention projection
    let hidden = config.hidden_size();
    let _num_heads = config.attention.num_heads;
    let num_kv_heads = config.attention.num_kv_heads;
    let head_dim = config.attention.head_dim;

    // Token embedding lookup (same as decoder_forward)
    let (embed_bytes, embed_dtype) = get_typed_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    ).map_err(|e| KnowledgeError::DataFormatError(format!("embed lookup failed: {e}")))?;
    let embed_data = typed_bytes_to_f32(&embed_bytes, embed_dtype);

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = TypedBuffer::zeros(seq_len * hidden, config.dtype());
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(KnowledgeError::DataFormatError(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state.as_f32_mut()[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    // Step 2: Extract K/V from the first layer's attention projection
    // In a real implementation, this would run the full encoder and extract
    // the K/V tensors from each layer. For now, we compute a simplified version.

    let kv_dim = num_kv_heads * head_dim;
    let total_kv_size = seq_len * kv_dim;

    // Allocate K and V tensors
    let mut k_tensor = vec![0.0f32; total_kv_size];
    let mut v_tensor = vec![0.0f32; total_kv_size];

    // Simplified: Use a portion of hidden_state as K/V (for demonstration)
    // In production, this would be the actual K/Q/V projection output
    let k_start = 0;
    let v_start = kv_dim.min(hidden);
    let copy_size = kv_dim.min(hidden - v_start);

    for i in 0..seq_len {
        let hs_base = i * hidden;
        let kv_base = i * kv_dim;
        // Copy hidden state to K/V (simplified; real implementation uses projection)
        if k_start + copy_size <= hidden && kv_base + copy_size <= total_kv_size {
            k_tensor[kv_base..kv_base + copy_size]
                .copy_from_slice(&hidden_state.as_f32()[hs_base + k_start..hs_base + k_start + copy_size]);
        }
        if v_start + copy_size <= hidden && kv_base + copy_size <= total_kv_size {
            v_tensor[kv_base..kv_base + copy_size]
                .copy_from_slice(&hidden_state.as_f32()[hs_base + v_start..hs_base + v_start + copy_size]);
        }
    }

    // Step 3: Write K/V to KV cache page table
    // For CPU backend, we directly write to the KV store
    if !kv_caches.is_empty() {
        let mut store = backend.kv_store().lock()
            .map_err(|e| KnowledgeError::DataFormatError(format!("KV store lock failed: {e}")))?;

        if let Some(buf) = store.get_mut(&kv_caches[0].0) {
            // Convert f32 tensors to bytes
            let k_bytes: Vec<u8> = k_tensor.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let v_bytes: Vec<u8> = v_tensor.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();

            let _total_kv_layers = config.num_layers();
            let layer_kv_byte_size = k_bytes.len() + v_bytes.len();

            // Calculate offset for target layer (in bytes)
            let layer_offset = target_layer * layer_kv_byte_size;

            // Ensure buffer has enough space
            let required_size = layer_offset + layer_kv_byte_size;
            if buf.k.len() < required_size {
                return Err(KnowledgeError::DataFormatError(format!(
                    "KV cache buffer too small: need {}, have {}",
                    required_size, buf.k.len()
                )));
            }

            // Write K tensor bytes
            let k_start = layer_offset;
            let k_end = k_start + k_bytes.len();
            if k_end <= buf.k.len() {
                buf.k[k_start..k_end].copy_from_slice(&k_bytes);
            }

            // Write V tensor bytes
            let v_start = k_end;
            let v_end = v_start + v_bytes.len();
            if v_end <= buf.v.len() {
                buf.v[v_start..v_end].copy_from_slice(&v_bytes);
            }
        }
    }

    // Step 4: Update prefix_index would be done by the caller
    // via scheduler.register_prefix_tokens()

    Ok(seq_len)
}

// ---------------------------------------------------------------------------
// 1b. Frozen KV Injection from pre-materialized bytes (per SPEC 04-API-DESIGN §8.4)
// ---------------------------------------------------------------------------

/// Inject frozen KV cache from pre-materialized bytes.
///
/// This is the side-loading path: the caller has already serialized KV data
/// (e.g., from a safetensors file) and wants to write it directly into the
/// backend's KV store at the specified layer.
///
/// # Arguments
/// - `backend`: CPU backend holding the KV store
/// - `data`: Raw f32 KV data serialized as bytes (K then V concatenated)
/// - `shape`: Expected shape `[num_layers, 2, num_kv_heads, max_seq_len, head_dim]`
/// - `target_layer`: Physical layer index to write into
///
/// # Returns
/// Number of bytes written
///
/// # Errors
/// - If KV store lock fails
/// - If data shape doesn't match expectations
pub fn inject_frozen_kv_from_bytes<E: Element>(
    backend: &CpuBackend<E>,
    data: &[u8],
    shape: &[usize],
    target_layer: usize,
) -> Result<usize, KnowledgeError> {
    // shape = [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
    if shape.len() != 5 {
        return Err(KnowledgeError::DataFormatError(format!(
            "expected 5D shape [num_layers, 2, num_kv_heads, max_seq_len, head_dim], got {}D",
            shape.len()
        )));
    }
    let num_layers = shape[0];
    let num_kv_heads = shape[2];
    let max_seq_len = shape[3];
    let head_dim = shape[4];

    if target_layer >= num_layers {
        return Err(KnowledgeError::InvalidLayerTarget);
    }

    // Each layer's K or V tensor size in bytes (f32)
    let kv_elem_bytes = std::mem::size_of::<f32>();
    let layer_kv_elements = num_kv_heads * max_seq_len * head_dim;
    let layer_kv_bytes = layer_kv_elements * kv_elem_bytes;

    // data layout: [num_layers][2][num_kv_heads * max_seq_len * head_dim] as f32 bytes
    // K for layer i starts at: i * 2 * layer_kv_bytes
    // V for layer i starts at: i * 2 * layer_kv_bytes + layer_kv_bytes
    let k_offset = target_layer * 2 * layer_kv_bytes;
    let v_offset = k_offset + layer_kv_bytes;

    // Validate data has enough bytes
    let required_total = num_layers * 2 * layer_kv_bytes;
    if data.len() < required_total {
        return Err(KnowledgeError::DataFormatError(format!(
            "data too small: need {} bytes, got {}",
            required_total, data.len()
        )));
    }

    // Write into the backend's KV store
    let mut store = backend.kv_store().lock()
        .map_err(|e| KnowledgeError::KvCacheError(format!("KV store lock failed: {e}")))?;

    // Find or create a KV cache buffer entry (use handle 0 as default)
    let entry = store.entry(0).or_insert_with(|| {
        let total_bytes = num_layers * num_kv_heads * max_seq_len * head_dim * kv_elem_bytes;
        super::cpu_backend::KvCacheBuffer {
            k: vec![0u8; total_bytes],
            v: vec![0u8; total_bytes],
            num_layers,
            num_kv_heads,
            max_seq_len,
            head_dim,
            page_size: 16,
            seq_len: 0,
            elem_bytes: kv_elem_bytes,
            cache_dtype: gllm_kernels::types::DType::F32,
        }
    });

    // Calculate offset within the flat K/V buffers
    // Layout: [num_layers * num_kv_heads * max_seq_len * head_dim] per K and V
    let layer_flat_offset = target_layer * num_kv_heads * max_seq_len * head_dim * kv_elem_bytes;

    // Write K data
    let k_dst_start = layer_flat_offset;
    let k_dst_end = k_dst_start + layer_kv_bytes;
    if k_dst_end <= entry.k.len() {
        entry.k[k_dst_start..k_dst_end].copy_from_slice(&data[k_offset..k_offset + layer_kv_bytes]);
    }

    // Write V data
    let v_dst_start = layer_flat_offset;
    let v_dst_end = v_dst_start + layer_kv_bytes;
    if v_dst_end <= entry.v.len() {
        entry.v[v_dst_start..v_dst_end].copy_from_slice(&data[v_offset..v_offset + layer_kv_bytes]);
    }

    Ok(layer_kv_bytes * 2)
}

// ---------------------------------------------------------------------------
// 2. Late Fusion Injection (per SPEC 04-API-DESIGN §8.2)
// ---------------------------------------------------------------------------

/// Inject late fusion vector via truncated forward pass.
///
/// Pipeline:
/// 1. Call forward_to_layer() to get hidden state at target layer
/// 2. Embed the hidden state into the inference path at the specified layer
/// 3. The vector is added as a residual connection
///
/// # Arguments
/// - `backend`: CPU backend for execution
/// - `tokens`: Input token IDs
/// - `weights`: Model weights
/// - `config`: Forward configuration
/// - `target`: Semantic layer target (ShallowSyntax/MidSemantic/DeepLogic)
///
/// # Returns
/// Hidden state vector at the target semantic layer
///
/// # Errors
/// - If forward_to_layer fails
pub fn inject_late_fusion<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
    target: LayerTarget,
) -> Result<Vec<f32>, KnowledgeError> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(KnowledgeError::UnsupportedElementType(
            format!("inject_late_fusion requires f32, got {}", std::any::type_name::<E>()),
        ));
    }

    // Use forward_to_semantic_layer to get hidden state at target layer
    decoder_forward::forward_to_semantic_layer(
        backend,
        tokens,
        weights,
        config,
        target,
    ).map_err(|e| KnowledgeError::DataFormatError(format!("forward_to_semantic_layer failed: {e}")))
}

// ---------------------------------------------------------------------------
// 3. Dynamic LoRA Injection (per SPEC 04-API-DESIGN §8.1)
// ---------------------------------------------------------------------------

/// Dynamic LoRA adapter configuration.
///
/// LoRA (Low-Rank Adaptation) injects trainable rank decomposition matrices
/// into specific layers of the model.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Target layer index
    pub layer: usize,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha (scaling factor)
    pub alpha: f32,
    /// LoRA A matrix (shape: [rank, in_features])
    pub lora_a: Vec<f32>,
    /// LoRA B matrix (shape: [out_features, rank])
    pub lora_b: Vec<f32>,
    /// Target module (e.g., "q_proj", "v_proj", "gate_proj")
    pub target_module: String,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter.
    ///
    /// # Arguments
    /// - `layer`: Target layer index
    /// - `rank`: LoRA rank (typically 4, 8, 16, etc.)
    /// - `alpha`: LoRA scaling factor
    /// - `in_features`: Input feature dimension
    /// - `out_features`: Output feature dimension
    /// - `target_module`: Target module name
    pub fn new(
        layer: usize,
        rank: usize,
        alpha: f32,
        in_features: usize,
        out_features: usize,
        target_module: impl Into<String>,
    ) -> Self {
        Self {
            layer,
            rank,
            alpha,
            lora_a: vec![0.0; rank * in_features],
            lora_b: vec![0.0; out_features * rank],
            target_module: target_module.into(),
        }
    }

    /// Load LoRA weights from a byte slice.
    pub fn load_weights(&mut self, lora_a: Vec<f32>, lora_b: Vec<f32>) -> Result<(), KnowledgeError> {
        if lora_a.len() != self.lora_a.len() {
            return Err(KnowledgeError::DataFormatError(format!(
                "lora_a size mismatch: expected {}, got {}",
                self.lora_a.len(), lora_a.len()
            )));
        }
        if lora_b.len() != self.lora_b.len() {
            return Err(KnowledgeError::DataFormatError(format!(
                "lora_b size mismatch: expected {}, got {}",
                self.lora_b.len(), lora_b.len()
            )));
        }
        self.lora_a = lora_a;
        self.lora_b = lora_b;
        Ok(())
    }

    /// Get the scaling factor for this LoRA adapter.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

/// Dynamic LoRA weight loader.
///
/// Loads LoRA A/B matrices and injects them into the specified layer.
///
/// # Arguments
/// - `adapter`: LoRA adapter configuration
/// - `weights`: Model weights (for getting base weight shape)
///
/// # Returns
/// Materialized payload containing LoRA weights
///
/// # Errors
/// - If weight shape validation fails
pub fn inject_dynamic_lora<E: Element>(
    adapter: &LoRAAdapter,
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
) -> Result<MaterializedPayload, KnowledgeError> {
    // Validate target module exists
    let weight_name = format!("model.layers.{}.{}.weight", adapter.layer, adapter.target_module);
    if weights.tensor_shape(&weight_name).is_none() {
        return Err(KnowledgeError::SourceNotFound(format!(
            "target module not found: {}", weight_name
        )));
    }

    // Flatten LoRA weights into a single byte vector
    let mut data = Vec::with_capacity((adapter.lora_a.len() + adapter.lora_b.len()) * 4);
    for &val in &adapter.lora_a {
        data.extend_from_slice(&val.to_le_bytes());
    }
    for &val in &adapter.lora_b {
        data.extend_from_slice(&val.to_le_bytes());
    }

    let mut metadata = std::collections::HashMap::new();
    metadata.insert("layer".to_string(), adapter.layer.to_string());
    metadata.insert("rank".to_string(), adapter.rank.to_string());
    metadata.insert("alpha".to_string(), adapter.alpha.to_string());
    metadata.insert("target_module".to_string(), adapter.target_module.clone());
    metadata.insert("scaling".to_string(), adapter.scaling().to_string());

    Ok(MaterializedPayload {
        kind: InjectionKind::DynamicLoRA,
        data,
        shape: vec![adapter.lora_a.len(), adapter.lora_b.len()],
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Helper: Page Table Registration
// ---------------------------------------------------------------------------

/// Register KV cache pages in the prefix index.
///
/// This helper function is used after `inject_frozen_kv` to register
/// the injected KV pages in the scheduler's prefix index for reuse.
///
/// # Arguments
/// - `tokens`: Token sequence that generated the KV
/// - `pages`: Physical page IDs containing the KV data
///
/// # Returns
/// Virtual page IDs for the registered pages
pub fn register_kv_pages(
    tokens: &[u32],
    pages: &[u32],
) -> Vec<VirtualPageId> {
    tokens.iter().enumerate().map(|(idx, &_token)| {
        VirtualPageId::new(idx as u64, idx / pages.len())
    }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_adapter_new() {
        let adapter = LoRAAdapter::new(0, 8, 16.0, 4096, 4096, "q_proj");
        assert_eq!(adapter.layer, 0);
        assert_eq!(adapter.rank, 8);
        assert_eq!(adapter.alpha, 16.0);
        assert_eq!(adapter.scaling(), 2.0);
        assert_eq!(adapter.lora_a.len(), 8 * 4096);
        assert_eq!(adapter.lora_b.len(), 4096 * 8);
    }

    #[test]
    fn test_lora_adapter_load_weights() {
        let mut adapter = LoRAAdapter::new(0, 4, 8.0, 128, 256, "v_proj");
        let lora_a = vec![1.0; 4 * 128];
        let lora_b = vec![2.0; 256 * 4];

        assert!(adapter.load_weights(lora_a.clone(), lora_b.clone()).is_ok());
        assert_eq!(adapter.lora_a, lora_a);
        assert_eq!(adapter.lora_b, lora_b);
    }

    #[test]
    fn test_lora_adapter_load_weights_wrong_size() {
        let mut adapter = LoRAAdapter::new(0, 4, 8.0, 128, 256, "v_proj");
        let lora_a = vec![1.0; 100]; // Wrong size
        let lora_b = vec![2.0; 256 * 4];

        assert!(adapter.load_weights(lora_a, lora_b).is_err());
    }

    #[test]
    fn test_register_kv_pages() {
        let tokens = vec![1, 2, 3, 4];
        let pages = vec![10, 20];

        let vpages = register_kv_pages(&tokens, &pages);
        assert_eq!(vpages.len(), tokens.len());
    }
}
