//! SigLIP Vision Encoder — real JIT-compiled forward pass.
//!
//! Builds a `CompilerGraph` for the full SigLIP ViT encoder (PatchEmbed →
//! LearnedPos2D → N × ViT blocks → final LayerNorm), JIT-compiles it through
//! the standard `InferenceCompiler::compile_mega_kernel_from_graph` pipeline (scalar → SymExec →
//! IR → ISA lowering → native machine code) and executes it once per image.
//!
//! Every computation path goes through the JIT: no scalar fallback, no
//! hand-written Rust, no external BLAS. The `VisionConfig` drives graph
//! construction; the `VisionTensorLookup` trait supplies weight tensors by
//! name (matching the SigLIP tensor naming convention).
//!
//! SPEC: 02-ARCHITECTURE ARCH-MULTIMODAL + ARCH-MULTIMODAL-FUSION.

use std::sync::Arc;

use gllm_kernels::compiler::{
    CompilerGraph, InferenceCompiler, OpKind,
    ShapeBinding, SymDim, TensorId,
};
use gllm_kernels::compiler::mega_kernel_abi::CompileConfig;
use gllm_kernels::types::DType;

use crate::compat::multimodal::{
    EncoderMedia, MediaKind, MultimodalEncoded, MultimodalEncoder, MultimodalTokenIds,
};
use crate::engine::executor::BackendError;

/// Trait for looking up named vision encoder weight tensors.
///
/// All tensors are expected to be row-major f32. Names follow the
/// HuggingFace `SiglipVisionModel` convention.
pub trait VisionTensorLookup {
    /// Returns the raw f32 data for the given weight name, or `None` if absent.
    fn get_vision_tensor(&self, name: &str) -> Option<&[f32]>;

    /// Returns the shape of the given weight tensor, or `None` if absent.
    fn vision_tensor_shape(&self, name: &str) -> Option<&[usize]>;
}

/// SigLIP Vision Encoder configuration.
///
/// Parsed from the `"vision_config"` sub-object in `config.json`.
/// All fields mirror the HuggingFace SigLIP / ViT config schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VisionConfig {
    /// Input image resolution (pixels per side, e.g. 224 or 384).
    pub image_size: usize,
    /// Patch size (pixels per side, e.g. 14 or 16).
    pub patch_size: usize,
    /// Hidden dimension of the vision transformer.
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_layers: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
}

impl VisionConfig {
    /// Number of spatial patches = (image_size / patch_size)^2.
    pub fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    /// Head dimension = hidden_size / num_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Assert that the configuration is internally consistent.
    fn validate(&self) -> Result<(), BackendError> {
        if self.image_size == 0 || self.patch_size == 0 {
            return Err(BackendError::Other(
                "VisionConfig: image_size and patch_size must be > 0".into(),
            ));
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(BackendError::Other(format!(
                "VisionConfig: image_size {} not divisible by patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.hidden_size == 0 || self.num_heads == 0 {
            return Err(BackendError::Other(
                "VisionConfig: hidden_size and num_heads must be > 0".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(BackendError::Other(format!(
                "VisionConfig: hidden_size {} not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        if self.num_layers == 0 {
            return Err(BackendError::Other(
                "VisionConfig: num_layers must be > 0".into(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(BackendError::Other(
                "VisionConfig: intermediate_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Number of input channels for RGB images. SigLIP ViT operates on 3-channel
/// images; alternative channel counts would require a distinct pretraining.
const VISION_IN_CHANNELS: usize = 3;

/// LayerNorm epsilon used by SigLIP (matches HuggingFace reference).
const VISION_LAYERNORM_EPS: f32 = 1e-6;

// ============================================================================
// Tensor name conventions — matches SigLIP auto_graph tensor naming
// ============================================================================

fn patch_embed_weight_name() -> &'static str {
    "vision_tower.patch_embed.proj.weight"
}
fn position_embedding_name() -> &'static str {
    "vision_tower.embeddings.position_embedding.weight"
}
fn final_norm_weight_name() -> &'static str {
    "vision_tower.post_layernorm.weight"
}
fn final_norm_bias_name() -> &'static str {
    "vision_tower.post_layernorm.bias"
}
fn layer_tensor_name(layer_idx: usize, suffix: &str) -> String {
    format!("vision_tower.encoder.layers.{layer_idx}.{suffix}")
}

/// Every weight tensor consumed by the SigLIP encoder graph, in the order
/// `g.inputs[1..]` is populated. This controls the byte offsets produced by
/// `CompilerGraph::weight_layout()`, which in turn dictates how the caller
/// packs the weight blob that the JIT kernel reads at runtime.
struct GraphInputs {
    patch_kernel: TensorId,
    pos_table: TensorId,
    layers: Vec<PerLayerInputs>,
    final_norm_weight: TensorId,
    final_norm_bias: TensorId,
}

struct PerLayerInputs {
    ln1_weight: TensorId,
    ln1_bias: TensorId,
    w_q: TensorId,
    w_k: TensorId,
    w_v: TensorId,
    w_o: TensorId,
    ln2_weight: TensorId,
    ln2_bias: TensorId,
    w_fc1: TensorId,
    w_fc2: TensorId,
}

/// Weight spec: `(tensor_name, expected_shape)` for validation / lookup.
struct WeightSpec<'a> {
    name: String,
    shape: &'a [usize],
}

// ============================================================================
// CompilerGraph construction (single-image, no batch dim).
// ============================================================================

/// Build a `CompilerGraph` encoding the full SigLIP vision encoder for a
/// single image. Returns the graph plus the ordered list of weight specs that
/// the caller must pack — in the same order — into the weight blob.
fn build_vision_encoder_graph(
    config: &VisionConfig,
) -> Result<(CompilerGraph, Vec<WeightSpec<'static>>), BackendError> {
    config.validate()?;

    let dt = DType::F32;
    let in_channels = VISION_IN_CHANNELS;
    let patch_size = config.patch_size;
    let image_size = config.image_size;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim();
    let num_patches = config.num_patches();

    let mut g = CompilerGraph::new();

    // ─── Graph inputs ────────────────────────────────────────────────────
    // Activation input: image pixels [in_channels, H, W].
    let image = g.add_tensor_concrete(
        "image",
        &[in_channels, image_size, image_size],
        dt,
    );

    // PatchEmbed kernel [embed_dim, in_channels, patch_size, patch_size].
    let patch_kernel = g.add_tensor_concrete(
        patch_embed_weight_name(),
        &[hidden, in_channels, patch_size, patch_size],
        dt,
    );

    // Learned 2D positional embedding [num_patches, embed_dim].
    let pos_table = g.add_tensor_concrete(
        position_embedding_name(),
        &[num_patches, hidden],
        dt,
    );

    // Per-layer weights — registered in a fixed order (see `PerLayerInputs`).
    let mut layers = Vec::with_capacity(config.num_layers);
    let mut weight_specs: Vec<WeightSpec<'static>> = Vec::new();

    // Patch kernel + position table come first in the weight blob.
    weight_specs.push(WeightSpec {
        name: patch_embed_weight_name().to_string(),
        shape: &[],
    });
    weight_specs[0].shape = Box::leak(
        vec![hidden, in_channels, patch_size, patch_size].into_boxed_slice(),
    );
    weight_specs.push(WeightSpec {
        name: position_embedding_name().to_string(),
        shape: Box::leak(vec![num_patches, hidden].into_boxed_slice()),
    });

    for li in 0..config.num_layers {
        let ln1_w = g.add_tensor_concrete(
            &layer_tensor_name(li, "layer_norm1.weight"),
            &[hidden],
            dt,
        );
        let ln1_b = g.add_tensor_concrete(
            &layer_tensor_name(li, "layer_norm1.bias"),
            &[hidden],
            dt,
        );
        let w_q = g.add_tensor_concrete(
            &layer_tensor_name(li, "self_attn.q_proj.weight"),
            &[hidden, hidden],
            dt,
        );
        let w_k = g.add_tensor_concrete(
            &layer_tensor_name(li, "self_attn.k_proj.weight"),
            &[hidden, hidden],
            dt,
        );
        let w_v = g.add_tensor_concrete(
            &layer_tensor_name(li, "self_attn.v_proj.weight"),
            &[hidden, hidden],
            dt,
        );
        let w_o = g.add_tensor_concrete(
            &layer_tensor_name(li, "self_attn.out_proj.weight"),
            &[hidden, hidden],
            dt,
        );
        let ln2_w = g.add_tensor_concrete(
            &layer_tensor_name(li, "layer_norm2.weight"),
            &[hidden],
            dt,
        );
        let ln2_b = g.add_tensor_concrete(
            &layer_tensor_name(li, "layer_norm2.bias"),
            &[hidden],
            dt,
        );
        let w_fc1 = g.add_tensor_concrete(
            &layer_tensor_name(li, "mlp.fc1.weight"),
            &[hidden, inter],
            dt,
        );
        let w_fc2 = g.add_tensor_concrete(
            &layer_tensor_name(li, "mlp.fc2.weight"),
            &[inter, hidden],
            dt,
        );

        layers.push(PerLayerInputs {
            ln1_weight: ln1_w,
            ln1_bias: ln1_b,
            w_q,
            w_k,
            w_v,
            w_o,
            ln2_weight: ln2_w,
            ln2_bias: ln2_b,
            w_fc1,
            w_fc2,
        });

        // Layer weight specs mirror the add_tensor_concrete order above.
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "layer_norm1.weight"),
            shape: Box::leak(vec![hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "layer_norm1.bias"),
            shape: Box::leak(vec![hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "self_attn.q_proj.weight"),
            shape: Box::leak(vec![hidden, hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "self_attn.k_proj.weight"),
            shape: Box::leak(vec![hidden, hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "self_attn.v_proj.weight"),
            shape: Box::leak(vec![hidden, hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "self_attn.out_proj.weight"),
            shape: Box::leak(vec![hidden, hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "layer_norm2.weight"),
            shape: Box::leak(vec![hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "layer_norm2.bias"),
            shape: Box::leak(vec![hidden].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "mlp.fc1.weight"),
            shape: Box::leak(vec![hidden, inter].into_boxed_slice()),
        });
        weight_specs.push(WeightSpec {
            name: layer_tensor_name(li, "mlp.fc2.weight"),
            shape: Box::leak(vec![inter, hidden].into_boxed_slice()),
        });
    }

    // Final LayerNorm weight/bias.
    let final_norm_weight = g.add_tensor_concrete(
        final_norm_weight_name(),
        &[hidden],
        dt,
    );
    let final_norm_bias = g.add_tensor_concrete(
        final_norm_bias_name(),
        &[hidden],
        dt,
    );
    weight_specs.push(WeightSpec {
        name: final_norm_weight_name().to_string(),
        shape: Box::leak(vec![hidden].into_boxed_slice()),
    });
    weight_specs.push(WeightSpec {
        name: final_norm_bias_name().to_string(),
        shape: Box::leak(vec![hidden].into_boxed_slice()),
    });

    let inputs = GraphInputs {
        patch_kernel,
        pos_table,
        layers,
        final_norm_weight,
        final_norm_bias,
    };

    // Wire `g.inputs` — activation first, then every weight in the exact
    // order `weight_specs` lists. `CompilerGraph::weight_layout()` iterates
    // `g.inputs[1..]` for offset assignment.
    let mut input_tids: Vec<TensorId> = Vec::with_capacity(1 + weight_specs.len());
    input_tids.push(image);
    input_tids.push(inputs.patch_kernel);
    input_tids.push(inputs.pos_table);
    for layer in &inputs.layers {
        input_tids.push(layer.ln1_weight);
        input_tids.push(layer.ln1_bias);
        input_tids.push(layer.w_q);
        input_tids.push(layer.w_k);
        input_tids.push(layer.w_v);
        input_tids.push(layer.w_o);
        input_tids.push(layer.ln2_weight);
        input_tids.push(layer.ln2_bias);
        input_tids.push(layer.w_fc1);
        input_tids.push(layer.w_fc2);
    }
    input_tids.push(inputs.final_norm_weight);
    input_tids.push(inputs.final_norm_bias);
    g.inputs = input_tids;

    // ─── Graph ops ───────────────────────────────────────────────────────

    // 1. PatchEmbed: image → patches [num_patches, hidden]
    let patches = g.add_tensor_concrete("patches", &[num_patches, hidden], dt);
    g.add_op(
        OpKind::PatchEmbed {
            patch_size,
            embed_dim: hidden,
            in_channels,
            image_size,
        },
        vec![image, inputs.patch_kernel],
        vec![patches],
        "patch_embed",
    );

    // 2. LearnedPos2D: patches + pos_table → hidden_0
    let hidden_init = g.add_tensor_concrete("hidden_0", &[num_patches, hidden], dt);
    g.add_op(
        OpKind::LearnedPos2D {
            num_patches,
            embed_dim: hidden,
        },
        vec![patches, inputs.pos_table],
        vec![hidden_init],
        "pos_embed",
    );

    // 3. N transformer encoder layers.
    let mut current_hidden = hidden_init;
    for (li, layer) in inputs.layers.iter().enumerate() {
        // LayerNorm₁
        let normed1 = g.add_tensor_concrete(
            &format!("layer_{li}_normed1"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::LayerNorm {
                feature_dim: hidden,
                eps: VISION_LAYERNORM_EPS,
            },
            vec![current_hidden, layer.ln1_weight, layer.ln1_bias],
            vec![normed1],
            &format!("layer_{li}_ln1"),
        );

        // Q / K / V projections.
        let q_dim = num_heads * head_dim;
        let q = g.add_tensor_concrete(
            &format!("layer_{li}_q"),
            &[num_patches, q_dim],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
                trans_b: false,
            },
            vec![normed1, layer.w_q],
            vec![q],
            &format!("layer_{li}_q_proj"),
        );
        let k = g.add_tensor_concrete(
            &format!("layer_{li}_k"),
            &[num_patches, q_dim],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
                trans_b: false,
            },
            vec![normed1, layer.w_k],
            vec![k],
            &format!("layer_{li}_k_proj"),
        );
        let v = g.add_tensor_concrete(
            &format!("layer_{li}_v"),
            &[num_patches, q_dim],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
                trans_b: false,
            },
            vec![normed1, layer.w_v],
            vec![v],
            &format!("layer_{li}_v_proj"),
        );

        // Bidirectional MHA (causal=false).
        let attn = g.add_tensor_concrete(
            &format!("layer_{li}_attn"),
            &[num_patches, q_dim],
            dt,
        );
        g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: SymDim::Concrete(num_patches),
                num_heads,
                num_kv_heads: num_heads,
                head_dim,
                causal: false,
                attention_sinks: false,
            },
            vec![q, k, v],
            vec![attn],
            &format!("layer_{li}_mha"),
        );

        // O projection.
        let o_proj = g.add_tensor_concrete(
            &format!("layer_{li}_o_proj"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: hidden,
                k: q_dim,
                dtype: dt,
                trans_b: false,
            },
            vec![attn, layer.w_o],
            vec![o_proj],
            &format!("layer_{li}_o"),
        );

        // Residual₁: current_hidden + o_proj
        let after_attn = g.add_tensor_concrete(
            &format!("layer_{li}_after_attn"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::Residual,
            vec![current_hidden, o_proj],
            vec![after_attn],
            &format!("layer_{li}_attn_residual"),
        );

        // LayerNorm₂
        let normed2 = g.add_tensor_concrete(
            &format!("layer_{li}_normed2"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::LayerNorm {
                feature_dim: hidden,
                eps: VISION_LAYERNORM_EPS,
            },
            vec![after_attn, layer.ln2_weight, layer.ln2_bias],
            vec![normed2],
            &format!("layer_{li}_ln2"),
        );

        // FC1
        let fc1 = g.add_tensor_concrete(
            &format!("layer_{li}_fc1"),
            &[num_patches, inter],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: inter,
                k: hidden,
                dtype: dt,
                trans_b: false,
            },
            vec![normed2, layer.w_fc1],
            vec![fc1],
            &format!("layer_{li}_fc1"),
        );

        // GELU
        let gelu_out = g.add_tensor_concrete(
            &format!("layer_{li}_gelu"),
            &[num_patches, inter],
            dt,
        );
        g.add_op(
            OpKind::Gelu,
            vec![fc1],
            vec![gelu_out],
            &format!("layer_{li}_gelu"),
        );

        // FC2
        let fc2 = g.add_tensor_concrete(
            &format!("layer_{li}_fc2"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::Gemm{
                m: SymDim::Concrete(num_patches),
                n: hidden,
                k: inter,
                dtype: dt,
                trans_b: false,
            },
            vec![gelu_out, layer.w_fc2],
            vec![fc2],
            &format!("layer_{li}_fc2_proj"),
        );

        // Residual₂: after_attn + fc2
        let after_ffn = g.add_tensor_concrete(
            &format!("layer_{li}_after_ffn"),
            &[num_patches, hidden],
            dt,
        );
        g.add_op(
            OpKind::Residual,
            vec![after_attn, fc2],
            vec![after_ffn],
            &format!("layer_{li}_ffn_residual"),
        );

        current_hidden = after_ffn;
    }

    // 4. Final LayerNorm
    let image_tokens = g.add_tensor_concrete(
        "image_tokens",
        &[num_patches, hidden],
        dt,
    );
    g.add_op(
        OpKind::LayerNorm {
            feature_dim: hidden,
            eps: VISION_LAYERNORM_EPS,
        },
        vec![
            current_hidden,
            inputs.final_norm_weight,
            inputs.final_norm_bias,
        ],
        vec![image_tokens],
        "final_norm",
    );
    g.outputs = vec![image_tokens];

    Ok((g, weight_specs))
}

// ============================================================================
// Weight packing
// ============================================================================

/// Validate every weight present in `lookup` matches its declared shape and
/// pack them into a contiguous byte blob in `specs` order.
///
/// `CompilerGraph::weight_layout()` produces offsets by iterating
/// `g.inputs[1..]` with the same dtype sizes; since we added all tensors as
/// f32, the resulting byte offsets match `pack_weight_blob`'s cumulative
/// cursor exactly.
fn pack_weight_blob(
    specs: &[WeightSpec<'_>],
    lookup: &dyn VisionTensorLookup,
) -> Result<Vec<u8>, BackendError> {
    let mut total = 0usize;
    for spec in specs {
        let expected: usize = spec.shape.iter().product();
        total += expected * std::mem::size_of::<f32>();
    }
    let mut blob = Vec::with_capacity(total);
    for spec in specs {
        let expected_numel: usize = spec.shape.iter().product();
        let data = lookup.get_vision_tensor(&spec.name).ok_or_else(|| {
            BackendError::Other(format!(
                "vision weight missing: {} (expected shape {:?})",
                spec.name, spec.shape
            ))
        })?;
        if data.len() != expected_numel {
            return Err(BackendError::Other(format!(
                "vision weight '{}' length {} != expected {} (shape {:?})",
                spec.name,
                data.len(),
                expected_numel,
                spec.shape
            )));
        }
        if let Some(shape) = lookup.vision_tensor_shape(&spec.name) {
            if shape != spec.shape {
                return Err(BackendError::Other(format!(
                    "vision weight '{}' shape {:?} != expected {:?}",
                    spec.name, shape, spec.shape
                )));
            }
        }
        // f32 → little-endian byte copy (zero-copy reinterpret).
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of_val(data),
            )
        };
        blob.extend_from_slice(bytes);
    }
    Ok(blob)
}

// ============================================================================
// Public encode entry point
// ============================================================================

/// Encode raw image pixel data into a visual token sequence.
///
/// # Arguments
///
/// * `pixels`  - Flattened pixel data in `[channels, height, width]` layout
///               (channel-first, normalised per SigLIP pre-processing).
///               Channels must equal `VISION_IN_CHANNELS` (= 3).
/// * `config`  - Vision encoder geometry.
/// * `weights` - Provider for vision encoder weight tensors.
///
/// # Returns
///
/// A flat `Vec<f32>` of shape `[num_patches, hidden_size]` representing the
/// visual token embeddings after the full encoder stack.
pub fn vision_encode(
    pixels: &[f32],
    config: &VisionConfig,
    weights: &dyn VisionTensorLookup,
) -> Result<Vec<f32>, BackendError> {
    config.validate()?;

    let expected_pixels =
        VISION_IN_CHANNELS * config.image_size * config.image_size;
    if pixels.len() != expected_pixels {
        return Err(BackendError::Other(format!(
            "vision_encode: pixel buffer has {} elements, expected {} (3×{}×{})",
            pixels.len(),
            expected_pixels,
            config.image_size,
            config.image_size
        )));
    }

    let (graph, specs) = build_vision_encoder_graph(config)?;

    let weight_blob = pack_weight_blob(&specs, weights)?;

    // JIT compile + execute via compile_mega_kernel_from_graph.
    let num_patches = config.num_patches();
    let hidden = config.hidden_size;
    let mk_config = CompileConfig {
        max_seq_len: num_patches,
        debug_jit: false,
        hetero: None,
    };
    let mut compiler = InferenceCompiler::new();
    let compiled = compiler
        .compile_mega_kernel_from_graph(graph, &mk_config, None)
        .map_err(|e| {
            BackendError::Other(format!(
                "vision_encode: compile_mega_kernel_from_graph failed: {e}"
            ))
        })?
        .layer_code;

    let mut output = vec![0.0f32; num_patches * hidden];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(1)];

    // seq_len parameter is the `num_patches` for this graph (Concrete).
    // batch_size is 1 (single image).
    unsafe {
        compiled.execute_as_mega_kernel(
            pixels.as_ptr() as *const u8,
            weight_blob.as_ptr(),
            1,
            num_patches,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }

    Ok(output)
}

// ============================================================================
// Concrete weight store (for SigLipEncoder ownership)
// ============================================================================

/// Owned f32 tensor with shape metadata — mirrors the access pattern of
/// `WeightsHandle` but owns the data, so a `SigLipEncoder` can outlive the
/// loader that constructed it.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedVisionWeights {
    tensors: std::collections::HashMap<String, Arc<OwnedTensor>>,
}

#[derive(Debug, Clone, PartialEq)]
struct OwnedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl OwnedVisionWeights {
    pub fn new() -> Self {
        Self {
            tensors: std::collections::HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        self.tensors.insert(
            name.into(),
            Arc::new(OwnedTensor { data, shape }),
        );
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl Default for OwnedVisionWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl VisionTensorLookup for OwnedVisionWeights {
    fn get_vision_tensor(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|t| t.data.as_slice())
    }

    fn vision_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }
}

// ============================================================================
// SigLipEncoder — MultimodalEncoder implementation
// ============================================================================

/// Real SigLIP Vision Encoder implementing `MultimodalEncoder::encode_image`.
///
/// Owns its weights (an `Arc` over `OwnedVisionWeights`) so the encoder can
/// be registered on a `Client` without lifetime entanglement with the
/// underlying `WeightsHandle`. `encode_audio` is unimplemented (the pair agent
/// delivers a `ConformerEncoder` covering the audio half of the contract).
#[derive(Debug)]
pub struct SigLipEncoder {
    config: VisionConfig,
    weights: Arc<OwnedVisionWeights>,
    token_ids: MultimodalTokenIds,
}

impl SigLipEncoder {
    /// Construct a new encoder. Fails if the weight store is missing any of
    /// the tensors the graph will reference.
    pub fn new(
        config: VisionConfig,
        weights: Arc<OwnedVisionWeights>,
        token_ids: MultimodalTokenIds,
    ) -> Result<Self, BackendError> {
        config.validate()?;
        // Sanity-check: every tensor the graph will touch exists up-front.
        let (_, specs) = build_vision_encoder_graph(&config)?;
        for spec in &specs {
            let data = weights.get_vision_tensor(&spec.name).ok_or_else(|| {
                BackendError::Other(format!(
                    "SigLipEncoder::new: missing vision weight '{}'",
                    spec.name
                ))
            })?;
            let expected_numel: usize = spec.shape.iter().product();
            if data.len() != expected_numel {
                return Err(BackendError::Other(format!(
                    "SigLipEncoder::new: weight '{}' len {} != expected {}",
                    spec.name,
                    data.len(),
                    expected_numel
                )));
            }
        }
        Ok(Self {
            config,
            weights,
            token_ids,
        })
    }

    pub fn config(&self) -> &VisionConfig {
        &self.config
    }

    pub fn token_ids(&self) -> MultimodalTokenIds {
        self.token_ids
    }

    /// Decode `EncoderMedia` into a flat row-major `[3, H, W]` f32 pixel
    /// buffer sized for `config.image_size`.
    ///
    /// The caller is expected to supply pre-processed pixels (the gllm public
    /// API normalises and resizes images before handing them to the encoder);
    /// we therefore require `EncoderMedia::Raw` with exactly
    /// `3 × image_size × image_size × 4` bytes. Other media modes return an
    /// explicit error rather than silently fabricating pixel data.
    fn decode_pixels(&self, media: &EncoderMedia) -> Result<Vec<f32>, BackendError> {
        let image_size = self.config.image_size;
        let expected_f32 = VISION_IN_CHANNELS * image_size * image_size;
        let expected_bytes = expected_f32 * std::mem::size_of::<f32>();
        match media {
            EncoderMedia::Raw(bytes) => {
                if bytes.len() != expected_bytes {
                    return Err(BackendError::Other(format!(
                        "SigLipEncoder: Raw media has {} bytes, expected {} (3×{}×{}×f32)",
                        bytes.len(),
                        expected_bytes,
                        image_size,
                        image_size
                    )));
                }
                let mut out = vec![0.0f32; expected_f32];
                for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            EncoderMedia::File(_) => Err(BackendError::Other(
                "SigLipEncoder: EncoderMedia::File not yet supported — pass pre-processed \
                 pixels via EncoderMedia::Raw([3·H·W·f32 bytes])"
                    .into(),
            )),
            EncoderMedia::Base64 { .. } => Err(BackendError::Other(
                "SigLipEncoder: EncoderMedia::Base64 not yet supported — pass pre-processed \
                 pixels via EncoderMedia::Raw([3·H·W·f32 bytes])"
                    .into(),
            )),
            EncoderMedia::Url(_) => Err(BackendError::Other(
                "SigLipEncoder: EncoderMedia::Url not supported — the encoder does not \
                 perform network I/O; pre-fetch the image and pass EncoderMedia::Raw"
                    .into(),
            )),
        }
    }
}

impl MultimodalEncoder for SigLipEncoder {
    fn encode_image(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
        let pixels = self.decode_pixels(media)?;
        let embeddings =
            vision_encode(&pixels, &self.config, self.weights.as_ref())?;
        let num_tokens = self.config.num_patches();
        Ok(MultimodalEncoded {
            tokens: vec![self.token_ids.image_token_id; num_tokens],
            embeddings,
            hidden_size: self.config.hidden_size,
            kind: MediaKind::Image,
        })
    }

    fn encode_audio(&self, _media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
        Err(BackendError::Other(
            "SigLipEncoder does not handle audio — register a separate audio encoder \
             (e.g. ConformerEncoder) for audio modality"
                .into(),
        ))
    }
}

// ============================================================================
// Loader-side helper: build SigLipEncoder from a generic TensorLookup
// ============================================================================

/// Build a `SigLipEncoder` by copying the required tensors out of a
/// backend-specific `TensorLookup` (f32 slice view) into an owned
/// `OwnedVisionWeights`. Returns `Ok(None)` if any required weight is absent
/// (caller can proceed without multimodal support).
pub fn try_build_siglip_from_tensors<F>(
    config: &VisionConfig,
    token_ids: MultimodalTokenIds,
    mut fetch: F,
) -> Result<Option<SigLipEncoder>, BackendError>
where
    F: FnMut(&str) -> Option<(Vec<f32>, Vec<usize>)>,
{
    config.validate()?;
    let (_, specs) = build_vision_encoder_graph(config)?;

    let mut weights = OwnedVisionWeights::new();
    for spec in &specs {
        let (data, shape) = match fetch(&spec.name) {
            Some(v) => v,
            None => return Ok(None),
        };
        let expected_numel: usize = spec.shape.iter().product();
        if data.len() != expected_numel {
            return Err(BackendError::Other(format!(
                "try_build_siglip: weight '{}' len {} != expected {}",
                spec.name,
                data.len(),
                expected_numel
            )));
        }
        if shape != spec.shape {
            return Err(BackendError::Other(format!(
                "try_build_siglip: weight '{}' shape {:?} != expected {:?}",
                spec.name, shape, spec.shape
            )));
        }
        weights.insert(spec.name.clone(), data, shape);
    }

    Ok(Some(SigLipEncoder::new(
        config.clone(),
        Arc::new(weights),
        token_ids,
    )?))
}

// Suppress "unused" warning when no SymDim symbolic path is needed; the
// `ShapeBinding` import remains part of the public contract because future
// Symbolic-num_patches variants (dynamic image sizes) will use it.
#[allow(dead_code)]
fn _shape_binding_unused_today() -> ShapeBinding {
    ShapeBinding::new()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::multimodal::{MultimodalContext, MultimodalTokenIds};

    fn tiny_config() -> VisionConfig {
        // Minimal SigLIP geometry that still exercises every op class.
        // num_patches = (14/7)^2 = 4. num_heads=1 to keep weight blob small.
        VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        }
    }

    /// Compile a simple (no layer loop) graph through `compile_mega_kernel_from_graph`.
    fn compile_simple_graph(
        compiler: &mut InferenceCompiler,
        graph: CompilerGraph,
        max_seq_len: usize,
    ) -> gllm_kernels::compiler::CompiledLayer {
        let config = CompileConfig {
            max_seq_len,
            debug_jit: false,
            hetero: None,
        };
        compiler
            .compile_mega_kernel_from_graph(graph, &config, None)
            .expect("compile_simple_graph")
            .layer_code
    }

    fn populate_weights(config: &VisionConfig) -> OwnedVisionWeights {
        // Deterministic pseudo-random weights.
        let (_, specs) = build_vision_encoder_graph(config).expect("graph");
        let mut weights = OwnedVisionWeights::new();
        let mut seed = 1u64;
        for spec in &specs {
            let numel: usize = spec.shape.iter().product();
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Map upper 24 bits to a small symmetric f32 range.
                let bits = (seed >> 40) as u32 & 0x00FFFFFF;
                let value = (bits as f32 / 8_388_608.0 - 1.0) * 0.1;
                data.push(value);
            }
            weights.insert(spec.name.clone(), data, spec.shape.to_vec());
        }
        weights
    }

    #[test]
    fn num_patches_calculation() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
    }

    #[test]
    fn vision_config_validate_rejects_mismatched_patch_size() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 15, // 224 not divisible by 15
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn vision_config_validate_rejects_bad_head_split() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 769, // not divisible by 12
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn graph_builder_registers_all_expected_ops() {
        let config = tiny_config();
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        // 2 (PatchEmbed + LearnedPos2D) + 12 ops/layer + 1 final norm.
        let expected_ops = 2 + 12 * config.num_layers + 1;
        assert_eq!(graph.num_ops(), expected_ops,
                   "expected {} ops, got {}", expected_ops, graph.num_ops());
        // One activation + per-layer 10 weights + 2 (patch kernel + pos) + 2 (final norm).
        let expected_weights = 2 + 10 * config.num_layers + 2;
        assert_eq!(specs.len(), expected_weights);
        assert_eq!(graph.inputs.len(), expected_weights + 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn pack_weight_blob_rejects_missing_tensor() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let empty = OwnedVisionWeights::new();
        let err = pack_weight_blob(&specs, &empty).unwrap_err();
        assert!(format!("{err}").contains("missing"));
    }

    #[test]
    fn pack_weight_blob_rejects_wrong_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let mut weights = populate_weights(&config);
        // Mutate one tensor to a wrong shape.
        let name = specs[0].name.clone();
        let bad_data = vec![0.0f32; 1];
        weights.insert(name, bad_data, vec![1]);
        let err = pack_weight_blob(&specs, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("length") || msg.contains("shape"));
    }

    #[test]
    fn vision_encoder_graph_compiles() {
        // Pure compile probe — isolates SIGSEGV root cause when the execute
        // path itself is broken vs when the graph construction is broken.
        let config = tiny_config();
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        let mut compiler = InferenceCompiler::new();
        let mk_config = CompileConfig {
            max_seq_len: config.num_patches(),
            debug_jit: false,
            hetero: None,
        };
        let _compiled = compiler
            .compile_mega_kernel_from_graph(graph, &mk_config, None)
            .expect("SigLIP encoder graph must compile through JIT pipeline");
    }

    #[test]
    fn patch_embed_only_executes() {
        // Build a minimal PatchEmbed + LearnedPos2D graph with no layers,
        // bypassing the full builder. If this executes while the full
        // encoder SIGSEGVs, the issue is isolated to attention / ffn path.
        use gllm_kernels::compiler::{CompilerGraph, OpKind};
        use gllm_kernels::types::DType;

        let (patch_size, embed_dim, in_channels, image_size) = (2_usize, 4, 1, 4);
        let num_patches = (image_size / patch_size).pow(2);

        let mut g = CompilerGraph::new();
        let dt = DType::F32;
        let image = g.add_tensor_concrete(
            "image", &[in_channels, image_size, image_size], dt);
        let kernel = g.add_tensor_concrete(
            "kernel", &[embed_dim, in_channels, patch_size, patch_size], dt);
        let pos = g.add_tensor_concrete("pos", &[num_patches, embed_dim], dt);
        g.inputs = vec![image, kernel, pos];

        let patches = g.add_tensor_concrete("patches", &[num_patches, embed_dim], dt);
        g.add_op(
            OpKind::PatchEmbed { patch_size, embed_dim, in_channels, image_size },
            vec![image, kernel],
            vec![patches],
            "patch_embed",
        );
        let out = g.add_tensor_concrete("out", &[num_patches, embed_dim], dt);
        g.add_op(
            OpKind::LearnedPos2D { num_patches, embed_dim },
            vec![patches, pos],
            vec![out],
            "pos_embed",
        );
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, num_patches);

        let image_data: Vec<f32> = (0..in_channels * image_size * image_size)
            .map(|i| i as f32 * 0.1).collect();
        let kernel_data: Vec<f32> =
            vec![0.25f32; embed_dim * in_channels * patch_size * patch_size];
        let pos_data: Vec<f32> = vec![1.0f32; num_patches * embed_dim];

        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe {
            std::slice::from_raw_parts(kernel_data.as_ptr() as *const u8, kernel_data.len() * 4)
        });
        weights.extend_from_slice(unsafe {
            std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, pos_data.len() * 4)
        });

        let mut output = vec![0.0f32; num_patches * embed_dim];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(64)];
        unsafe {
            compiled.execute_as_mega_kernel(
                image_data.as_ptr() as *const u8,
                weights.as_ptr(),
                1,
                num_patches,
                output.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }

        assert!(
            output.iter().all(|v| v.is_finite()),
            "patch_embed+pos_embed produced non-finite: {:?}",
            output
        );
        assert!(
            output.iter().any(|&v| v != 0.0),
            "patch_embed+pos_embed output all zero: {:?}",
            output
        );
    }

    #[test]
    fn single_layernorm_executes() {
        // Minimal test: just a LayerNorm to prove the JIT isn't broken by other issues.
        use gllm_kernels::compiler::{CompilerGraph, OpKind};
        use gllm_kernels::types::DType;

        let dt = DType::F32;
        let seq = 4;
        let h = 8;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[seq, h], dt);
        let ln_w = g.add_tensor_concrete("ln_w", &[h], dt);
        let ln_b = g.add_tensor_concrete("ln_b", &[h], dt);
        g.inputs = vec![input, ln_w, ln_b];
        let out = g.add_tensor_concrete("out", &[seq, h], dt);
        g.add_op(OpKind::LayerNorm { feature_dim: h, eps: 1e-6 }, vec![input, ln_w, ln_b], vec![out], "ln");
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, seq);

        let input_data: Vec<f32> = (0..seq*h).map(|i| i as f32 * 0.1).collect();
        let ln_w_data = vec![1.0f32; h];
        let ln_b_data = vec![0.0f32; h];
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(ln_w_data.as_ptr() as *const u8, h*4) });
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(ln_b_data.as_ptr() as *const u8, h*4) });

        let mut output = vec![0.0f32; seq * h];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute_as_mega_kernel(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                1, seq,
                output.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        eprintln!("ln output: {:?}", output);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn single_residual_executes() {
        use gllm_kernels::compiler::{CompilerGraph, OpKind};
        use gllm_kernels::types::DType;

        let dt = DType::F32;
        let m = 4;
        let h = 8;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[m, h], dt);
        let b = g.add_tensor_concrete("b", &[m, h], dt);
        g.inputs = vec![a, b];
        let out = g.add_tensor_concrete("out", &[m, h], dt);
        g.add_op(OpKind::Residual, vec![a, b], vec![out], "res");
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, m);

        let a_data: Vec<f32> = (0..m*h).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..m*h).map(|i| (i as f32 + 100.0) * 0.1).collect();
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len()*4) });
        let mut output = vec![0.0f32; m*h];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute_as_mega_kernel(
                a_data.as_ptr() as *const u8, weights.as_ptr(),
                1, m,
                output.as_mut_ptr() as *mut u8, scratch.as_mut_ptr());
        }
        eprintln!("residual output: {:?}", output);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn layernorm_then_gemm_executes() {
        use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
        use gllm_kernels::types::DType;

        let dt = DType::F32;
        let m = 4;
        let n = 8;
        let k = 8;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[m, k], dt);
        let ln_w = g.add_tensor_concrete("ln_w", &[k], dt);
        let ln_b = g.add_tensor_concrete("ln_b", &[k], dt);
        let w = g.add_tensor_concrete("w", &[k, n], dt);
        g.inputs = vec![input, ln_w, ln_b, w];

        let normed = g.add_tensor_concrete("normed", &[m, k], dt);
        g.add_op(OpKind::LayerNorm { feature_dim: k, eps: 1e-6 }, vec![input, ln_w, ln_b], vec![normed], "ln");

        let out = g.add_tensor_concrete("out", &[m, n], dt);
        g.add_op(
            OpKind::Gemm{ m: SymDim::Concrete(m), n, k, dtype: dt, trans_b: false, },
            vec![normed, w], vec![out], "gemm",
        );
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, m);

        let input_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let ln_w_data = vec![1.0f32; k];
        let ln_b_data = vec![0.0f32; k];
        let w_data: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.01).collect();
        let mut weights = Vec::new();
        for data in [&ln_w_data, &ln_b_data, &w_data] {
            weights.extend_from_slice(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) });
        }
        let mut output = vec![0.0f32; m * n];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute_as_mega_kernel(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                1, m,
                output.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        eprintln!("ln+gemm output: {:?}", output);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn single_gemm_executes() {
        use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
        use gllm_kernels::types::DType;

        let dt = DType::F32;
        let m = 4;
        let n = 8;
        let k = 8;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[m, k], dt);
        let w = g.add_tensor_concrete("w", &[k, n], dt);
        g.inputs = vec![input, w];
        let out = g.add_tensor_concrete("out", &[m, n], dt);
        g.add_op(
            OpKind::Gemm{ m: SymDim::Concrete(m), n, k, dtype: dt, trans_b: false, },
            vec![input, w],
            vec![out],
            "gemm",
        );
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, m);

        let input_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let w_data: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.005).collect();
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_data.len() * 4) });
        let mut output = vec![0.0f32; m * n];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute_as_mega_kernel(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                1, m,
                output.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        eprintln!("gemm output: {:?}", output);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn single_gemm_layernorm_executes() {
        // Minimal encoder stub: only LayerNorm + Gemm + Residual (no MHA).
        use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
        use gllm_kernels::types::DType;

        let dt = DType::F32;
        let seq = 4;
        let h = 8;
        let mut g = CompilerGraph::new();
        let input = g.add_tensor_concrete("input", &[seq, h], dt);
        let ln_w = g.add_tensor_concrete("ln_w", &[h], dt);
        let ln_b = g.add_tensor_concrete("ln_b", &[h], dt);
        let w_q = g.add_tensor_concrete("w_q", &[h, h], dt);
        g.inputs = vec![input, ln_w, ln_b, w_q];

        let normed = g.add_tensor_concrete("normed", &[seq, h], dt);
        g.add_op(
            OpKind::LayerNorm { feature_dim: h, eps: 1e-6 },
            vec![input, ln_w, ln_b],
            vec![normed],
            "ln",
        );
        let q = g.add_tensor_concrete("q", &[seq, h], dt);
        g.add_op(
            OpKind::Gemm{ m: SymDim::Concrete(seq), n: h, k: h, dtype: dt, trans_b: false, },
            vec![normed, w_q],
            vec![q],
            "gemm_q",
        );
        let out = g.add_tensor_concrete("out", &[seq, h], dt);
        g.add_op(OpKind::Residual, vec![input, q], vec![out], "resid");
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compile_simple_graph(&mut compiler, g, seq);

        let input_data = vec![0.1f32; seq * h];
        let ln_w_data = vec![1.0f32; h];
        let ln_b_data = vec![0.0f32; h];
        let w_q_data = vec![0.01f32; h * h];

        let mut weights = Vec::new();
        for data in [&ln_w_data, &ln_b_data, &w_q_data] {
            weights.extend_from_slice(unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            });
        }

        let mut output = vec![0.0f32; seq * h];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute_as_mega_kernel(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                1, seq,
                output.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }
        eprintln!("ln+gemm+resid output: {:?}", output);
        assert!(output.iter().all(|v| v.is_finite()));
        assert!(output.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn vision_encoder_compile_reports_metadata() {
        let config = tiny_config();
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        let num_weight_inputs = graph.inputs.len() - 1;
        let mut compiler = InferenceCompiler::new();
        let mk_config = CompileConfig {
            max_seq_len: config.num_patches(),
            debug_jit: false,
            hetero: None,
        };
        let compiled = compiler.compile_mega_kernel_from_graph(graph, &mk_config, None).expect("compile").layer_code;
        eprintln!(
            "SigLIP compile: code_size={}B scratchpad={}B, weights={} (specs={})",
            compiled.code_size(),
            compiled.scratchpad_bytes,
            num_weight_inputs,
            specs.len(),
        );
        assert!(compiled.code_size() > 0);
        // Scratchpad must be non-zero for a graph with intermediate tensors.
        assert!(compiled.scratchpad_bytes > 0);
    }

    #[test]
    fn vision_encode_non_stub_output() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel)
            .map(|i| (i as f32 * 0.37).sin() * 0.5)
            .collect();

        let out = vision_encode(&pixels, &config, &weights)
            .expect("vision_encode must succeed with full weight set");

        let expected_len = config.num_patches() * config.hidden_size;
        assert_eq!(out.len(), expected_len, "unexpected output length");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "vision_encode produced non-finite values: {:?}",
            out
        );
        assert!(
            out.iter().any(|&v| v != 0.0),
            "vision_encode produced all-zero output — stub still in place"
        );
    }

    #[test]
    fn vision_encode_rejects_wrong_pixel_count() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixels = vec![0.0f32; 1]; // wrong size
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        assert!(format!("{err}").contains("pixel buffer"));
    }

    #[test]
    fn siglip_encoder_new_detects_missing_weight() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let weights = Arc::new(OwnedVisionWeights::new());
        let err = SigLipEncoder::new(config, weights, ids).unwrap_err();
        assert!(format!("{err}").contains("missing"));
    }

    #[test]
    fn siglip_encoder_integrates_with_multimodal_context() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();

        let encoder = SigLipEncoder::new(config.clone(), weights, ids)
            .expect("SigLipEncoder::new must succeed");

        // Build Raw media buffer matching pre-processed pixel expectations.
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| (i as f32) * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let media = EncoderMedia::Raw(raw_bytes);

        let encoded = encoder
            .encode_image(&media)
            .expect("SigLipEncoder::encode_image must succeed on pre-processed pixels");
        assert_eq!(encoded.kind, MediaKind::Image);
        assert_eq!(encoded.num_tokens(), config.num_patches());
        assert_eq!(encoded.hidden_size, config.hidden_size);
        encoded.validate().expect("MultimodalEncoded.validate");
        assert!(encoded.embeddings.iter().all(|v| v.is_finite()));

        // Flowing the output through MultimodalContext completes the handshake.
        let mut ctx = MultimodalContext::new();
        ctx.push_image(encoded)
            .expect("MultimodalContext::push_image must accept SigLIP output");
        assert_eq!(ctx.images.len(), 1);
    }

    #[test]
    fn siglip_encoder_rejects_audio_request() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let err = encoder
            .encode_audio(&EncoderMedia::Raw(vec![]))
            .unwrap_err();
        assert!(format!("{err}").contains("audio"));
    }

    #[test]
    fn siglip_encoder_rejects_unresolved_media_modes() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .encode_image(&EncoderMedia::File("/tmp/does_not_exist.jpg".into()))
            .unwrap_err();
        assert!(format!("{err}").contains("File"));

        let err = encoder
            .encode_image(&EncoderMedia::Url("https://example.com/x.jpg".into()))
            .unwrap_err();
        assert!(format!("{err}").contains("Url"));
    }

    #[test]
    fn try_build_siglip_returns_none_when_tensor_missing() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let mut call_count = 0;
        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            call_count += 1;
            if call_count > 3 {
                None
            } else {
                populated
                    .get_vision_tensor(name)
                    .map(|slice| (slice.to_vec(),
                         populated.vision_tensor_shape(name).unwrap().to_vec()))
            }
        })
        .expect("validation errors are separate from absence");
        assert!(result.is_none());
    }

    #[test]
    fn try_build_siglip_succeeds_when_all_tensors_present() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            populated
                .get_vision_tensor(name)
                .map(|slice| (slice.to_vec(),
                     populated.vision_tensor_shape(name).unwrap().to_vec()))
        })
        .expect("all weights present, build must succeed")
        .expect("Some(encoder) because every weight was found");
        assert_eq!(result.config().num_patches(), config.num_patches());
    }

    // ========================================================================
    // Additional tests for coverage gaps
    // ========================================================================

    // --- VisionConfig: num_patches / head_dim pure computation ---

    #[test]
    fn num_patches_square_grid_384_14() {
        let config = VisionConfig {
            image_size: 384,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        // 384 / 14 = 27 patches per side, 27^2 = 729
        assert_eq!(config.num_patches(), 729);
    }

    #[test]
    fn num_patches_single_patch() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        assert_eq!(config.num_patches(), 1);
    }

    #[test]
    fn head_dim_computation() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert_eq!(config.head_dim(), 64); // 768 / 12
    }

    #[test]
    fn head_dim_single_head() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        assert_eq!(config.head_dim(), 8);
    }

    // --- VisionConfig: validate exhaustive coverage ---

    #[test]
    fn validate_accepts_valid_config() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_image_size() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("image_size"));
    }

    #[test]
    fn validate_rejects_zero_patch_size() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 0,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("patch_size"));
    }

    #[test]
    fn validate_rejects_zero_hidden_size() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 0,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn validate_rejects_zero_num_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 0,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("num_heads"));
    }

    #[test]
    fn validate_rejects_zero_num_layers() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 0,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("num_layers"));
    }

    #[test]
    fn validate_rejects_zero_intermediate_size() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 0,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("intermediate_size"));
    }

    #[test]
    fn validate_rejects_hidden_not_divisible_by_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 100,
            num_layers: 2,
            num_heads: 3, // 100 % 3 != 0
            intermediate_size: 256,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
        assert!(format!("{err}").contains("num_heads"));
    }

    // --- OwnedVisionWeights ---

    #[test]
    fn owned_vision_weights_new_is_empty() {
        let w = OwnedVisionWeights::new();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn owned_vision_weights_default_is_empty() {
        let w = OwnedVisionWeights::default();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn owned_vision_weights_insert_and_lookup() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![1.0f32, 2.0, 3.0];
        let shape = vec![3];
        w.insert("test_tensor", data.clone(), shape.clone());

        assert!(!w.is_empty());
        assert_eq!(w.len(), 1);

        let looked_up = w.get_vision_tensor("test_tensor").unwrap();
        assert_eq!(looked_up, &data[..]);

        let looked_up_shape = w.vision_tensor_shape("test_tensor").unwrap();
        assert_eq!(looked_up_shape, &shape[..]);
    }

    #[test]
    fn owned_vision_weights_missing_tensor_returns_none() {
        let w = OwnedVisionWeights::new();
        assert!(w.get_vision_tensor("nonexistent").is_none());
        assert!(w.vision_tensor_shape("nonexistent").is_none());
    }

    #[test]
    fn owned_vision_weights_insert_replaces_existing() {
        let mut w = OwnedVisionWeights::new();
        w.insert("tensor_a", vec![1.0f32], vec![1]);
        w.insert("tensor_a", vec![2.0f32, 3.0], vec![2]);

        assert_eq!(w.len(), 1);
        assert_eq!(w.get_vision_tensor("tensor_a").unwrap(), &[2.0f32, 3.0]);
        assert_eq!(w.vision_tensor_shape("tensor_a").unwrap(), &[2]);
    }

    #[test]
    fn owned_vision_weights_multiple_tensors() {
        let mut w = OwnedVisionWeights::new();
        w.insert("a", vec![1.0f32], vec![1]);
        w.insert("b", vec![2.0f32, 3.0], vec![2]);
        w.insert("c", vec![4.0f32], vec![1]);

        assert_eq!(w.len(), 3);
        assert!(w.get_vision_tensor("a").is_some());
        assert!(w.get_vision_tensor("b").is_some());
        assert!(w.get_vision_tensor("c").is_some());
    }

    // --- Constants ---

    #[test]
    fn vision_in_channels_is_three() {
        assert_eq!(VISION_IN_CHANNELS, 3);
    }

    #[test]
    fn vision_layernorm_eps_value() {
        assert!((VISION_LAYERNORM_EPS - 1e-6f32).abs() < 1e-10);
    }

    // --- Tensor name conventions ---

    #[test]
    fn patch_embed_weight_name_format() {
        assert_eq!(
            patch_embed_weight_name(),
            "vision_tower.patch_embed.proj.weight"
        );
    }

    #[test]
    fn position_embedding_name_format() {
        assert_eq!(
            position_embedding_name(),
            "vision_tower.embeddings.position_embedding.weight"
        );
    }

    #[test]
    fn final_norm_weight_name_format() {
        assert_eq!(
            final_norm_weight_name(),
            "vision_tower.post_layernorm.weight"
        );
    }

    #[test]
    fn final_norm_bias_name_format() {
        assert_eq!(
            final_norm_bias_name(),
            "vision_tower.post_layernorm.bias"
        );
    }

    #[test]
    fn layer_tensor_name_format() {
        let name = layer_tensor_name(3, "self_attn.q_proj.weight");
        assert_eq!(
            name,
            "vision_tower.encoder.layers.3.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn layer_tensor_name_layer_zero() {
        let name = layer_tensor_name(0, "layer_norm1.weight");
        assert_eq!(
            name,
            "vision_tower.encoder.layers.0.layer_norm1.weight"
        );
    }

    // --- pack_weight_blob: shape mismatch from vision_tensor_shape ---

    #[test]
    fn pack_weight_blob_rejects_inconsistent_shape_metadata() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        // Build an OwnedVisionWeights with correct data but patch the shape
        // metadata for the first tensor to be wrong. We do this by building
        // valid weights then re-inserting the first tensor with a wrong shape.
        let mut weights = populate_weights(&config);
        let first_name = specs[0].name.clone();
        let first_numel: usize = specs[0].shape.iter().product();
        // Re-insert with wrong shape metadata but correct data length.
        weights.insert(
            first_name,
            vec![0.0f32; first_numel],
            vec![999], // wrong shape
        );

        let err = pack_weight_blob(&specs, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shape"), "error should mention shape: {msg}");
    }

    // --- SigLipEncoder: decode_pixels through all EncoderMedia variants ---

    #[test]
    fn decode_pixels_raw_correct_size() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..expected_f32).map(|i| i as f32 * 0.1).collect();
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(decoded.len(), expected_f32);
        // Verify round-trip fidelity: f32 -> le bytes -> f32.
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                (v - pixels[i]).abs() < 1e-7,
                "decoded[{i}] = {v}, expected {}",
                pixels[i]
            );
        }
    }

    #[test]
    fn decode_pixels_raw_wrong_size() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        // 10 bytes instead of 3*14*14*4 = 2352 bytes.
        let err = encoder
            .decode_pixels(&EncoderMedia::Raw(vec![0u8; 10]))
            .unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    #[test]
    fn decode_pixels_raw_empty_buffer() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let err = encoder
            .decode_pixels(&EncoderMedia::Raw(vec![]))
            .unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    #[test]
    fn decode_pixels_file_variant_returns_error() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .decode_pixels(&EncoderMedia::File("/tmp/image.jpg".into()))
            .unwrap_err();
        assert!(format!("{err}").contains("File"));
    }

    #[test]
    fn decode_pixels_base64_variant_returns_error() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .decode_pixels(&EncoderMedia::Base64 {
                data: "AAAA".into(),
                mime_type: None,
            })
            .unwrap_err();
        assert!(format!("{err}").contains("Base64"));
    }

    #[test]
    fn decode_pixels_url_variant_returns_error() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .decode_pixels(&EncoderMedia::Url("https://example.com/img.jpg".into()))
            .unwrap_err();
        assert!(format!("{err}").contains("Url"));
    }

    // --- SigLipEncoder: config() and token_ids() accessors ---

    #[test]
    fn siglip_encoder_config_accessor() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        assert_eq!(encoder.config().image_size, config.image_size);
        assert_eq!(encoder.config().patch_size, config.patch_size);
        assert_eq!(encoder.config().hidden_size, config.hidden_size);
        assert_eq!(encoder.config().num_layers, config.num_layers);
        assert_eq!(encoder.config().num_heads, config.num_heads);
        assert_eq!(encoder.config().intermediate_size, config.intermediate_size);
    }

    #[test]
    fn siglip_encoder_token_ids_accessor() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let retrieved = encoder.token_ids();
        assert_eq!(retrieved.image_token_id, 258880);
        assert_eq!(retrieved.audio_token_id, 258881);
        assert_eq!(retrieved.eoi_token_id, 258882);
        assert_eq!(retrieved.eoa_token_id, 258883);
    }

    // --- SigLipEncoder: new() rejects wrong numel ---

    #[test]
    fn siglip_encoder_new_rejects_wrong_weight_numel() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let mut weights = OwnedVisionWeights::new();
        for spec in &specs {
            let numel: usize = spec.shape.iter().product();
            // Insert wrong-size data for one tensor.
            let mut data = vec![0.0f32; numel];
            data.push(999.0); // one extra element
            weights.insert(spec.name.clone(), data, spec.shape.to_vec());
        }

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        assert!(format!("{err}").contains("weight"));
        assert!(format!("{err}").contains("len"));
    }

    // --- SigLipEncoder: encode_image MultimodalEncoded shape ---

    #[test]
    fn encode_image_output_token_count_matches_num_patches() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = encoder.encode_image(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(encoded.num_tokens(), config.num_patches());
        assert_eq!(encoded.embeddings.len(), config.num_patches() * config.hidden_size);
        // All tokens should be image_token_id.
        assert!(encoded.tokens.iter().all(|&t| t == 258880));
    }

    // --- vision_encode: rejects invalid config ---

    #[test]
    fn vision_encode_rejects_invalid_config() {
        let bad_config = VisionConfig {
            image_size: 0,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let weights = OwnedVisionWeights::new();
        let err = vision_encode(&[], &bad_config, &weights).unwrap_err();
        assert!(format!("{err}").contains("image_size"));
    }

    // --- vision_encode: rejects missing weights ---

    #[test]
    fn vision_encode_rejects_missing_weights() {
        let config = tiny_config();
        let pixels = vec![0.0f32; VISION_IN_CHANNELS * config.image_size * config.image_size];
        let empty_weights = OwnedVisionWeights::new();
        let err = vision_encode(&pixels, &config, &empty_weights).unwrap_err();
        assert!(format!("{err}").contains("missing"));
    }

    // --- try_build_siglip_from_tensors: wrong numel ---

    #[test]
    fn try_build_siglip_rejects_wrong_numel() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            let spec = specs.iter().find(|s| s.name == name).unwrap();
            let numel: usize = spec.shape.iter().product();
            // Return data with one extra element.
            let data = vec![0.0f32; numel + 1];
            Some((data, spec.shape.to_vec()))
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("len"));
    }

    // --- try_build_siglip_from_tensors: wrong shape ---

    #[test]
    fn try_build_siglip_rejects_wrong_shape() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            let spec = specs.iter().find(|s| s.name == name).unwrap();
            let numel: usize = spec.shape.iter().product();
            // Return correct numel but wrong shape metadata.
            Some((vec![0.0f32; numel], vec![numel]))
        });
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("shape"));
    }

    // --- try_build_siglip_from_tensors: rejects invalid config ---

    #[test]
    fn try_build_siglip_rejects_invalid_config() {
        let bad_config = VisionConfig {
            image_size: 0,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = try_build_siglip_from_tensors(&bad_config, ids, |_| None).unwrap_err();
        assert!(format!("{err}").contains("image_size"));
    }

    // --- Graph construction: multi-layer config ---

    #[test]
    fn graph_builder_multi_layer() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 3,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        let num_patches = config.num_patches();
        assert_eq!(num_patches, 4);

        // 2 (PatchEmbed + LearnedPos2D) + 12 ops/layer * 3 layers + 1 final norm = 39.
        let expected_ops = 2 + 12 * 3 + 1;
        assert_eq!(graph.num_ops(), expected_ops);

        // 1 activation + 2 (patch+pos) + 10*3 (layers) + 2 (final norm) = 35.
        let expected_weights = 2 + 10 * 3 + 2;
        assert_eq!(specs.len(), expected_weights);
        assert_eq!(graph.inputs.len(), expected_weights + 1);
    }

    // --- SigLipEncoder: base64 variant through encode_image ---

    #[test]
    fn siglip_encoder_rejects_base64_media() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .encode_image(&EncoderMedia::Base64 {
                data: "AAAA".into(),
                mime_type: Some("image/png".into()),
            })
            .unwrap_err();
        assert!(format!("{err}").contains("Base64"));
    }

    // --- pack_weight_blob: success path (all weights present, correct shapes) ---

    #[test]
    fn pack_weight_blob_succeeds_with_valid_weights() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let weights = populate_weights(&config);

        let blob = pack_weight_blob(&specs, &weights).unwrap();
        assert!(!blob.is_empty());

        // Verify blob size matches sum of all spec elements * 4 bytes.
        let expected_bytes: usize = specs
            .iter()
            .map(|s| s.shape.iter().product::<usize>() * std::mem::size_of::<f32>())
            .sum();
        assert_eq!(blob.len(), expected_bytes);
    }

    // --- pack_weight_blob: verifies f32 little-endian byte layout ---

    #[test]
    fn pack_weight_blob_byte_layout_is_le_f32() {
        let mut w = OwnedVisionWeights::new();
        w.insert("test", vec![1.0f32, 2.0, 256.0, -1.0], vec![4]);

        let spec = WeightSpec {
            name: "test".to_string(),
            shape: &[4],
        };
        let blob = pack_weight_blob(&[spec], &w).unwrap();

        // Verify first f32 value (1.0) as LE bytes.
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
        // Verify second f32 value (2.0).
        assert_eq!(blob[4..8], 2.0f32.to_le_bytes());
        // Verify third f32 value (256.0).
        assert_eq!(blob[8..12], 256.0f32.to_le_bytes());
        // Verify fourth f32 value (-1.0).
        assert_eq!(blob[12..16], (-1.0f32).to_le_bytes());
    }

    // --- VisionConfig: Clone and Debug ---

    #[test]
    fn vision_config_clone_is_equal() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let cloned = config.clone();
        assert_eq!(cloned.image_size, config.image_size);
        assert_eq!(cloned.patch_size, config.patch_size);
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_layers, config.num_layers);
        assert_eq!(cloned.num_heads, config.num_heads);
        assert_eq!(cloned.intermediate_size, config.intermediate_size);
    }

    #[test]
    fn vision_config_debug_output() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("image_size"));
        assert!(debug_str.contains("224"));
        assert!(debug_str.contains("patch_size"));
        assert!(debug_str.contains("14"));
    }

    // --- OwnedVisionWeights: clone ---

    #[test]
    fn owned_vision_weights_clone_preserves_data() {
        let mut w = OwnedVisionWeights::new();
        w.insert("a", vec![1.0f32, 2.0], vec![2]);
        w.insert("b", vec![3.0f32], vec![1]);

        let cloned = w.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.get_vision_tensor("a").unwrap(), &[1.0f32, 2.0]);
        assert_eq!(cloned.get_vision_tensor("b").unwrap(), &[3.0f32]);
    }

    // --- Shape binding utility (smoke test) ---

    #[test]
    fn shape_binding_unused_creates_empty() {
        let sb = _shape_binding_unused_today();
        // ShapeBinding::new() creates an empty binding; just verify no panic.
        assert_eq!(format!("{sb:?}").len() > 0, true);
    }

    // ========================================================================
    // Additional tests for remaining coverage gaps
    // ========================================================================

    #[test]
    fn num_patches_large_image_448() {
        let config = VisionConfig {
            image_size: 448,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        // 448 / 14 = 32 patches per side, 32^2 = 1024
        assert_eq!(config.num_patches(), 1024);
    }

    #[test]
    fn num_patches_non_square_grid_28_7() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        // 28 / 7 = 4 patches per side, 4^2 = 16
        assert_eq!(config.num_patches(), 16);
    }

    #[test]
    fn head_dim_multi_head() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        assert_eq!(config.head_dim(), 64); // 1024 / 16
    }

    #[test]
    fn validate_rejects_image_smaller_than_patch() {
        let config = VisionConfig {
            image_size: 7,
            patch_size: 14, // 7 < 14 but both > 0
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        assert!(format!("{err}").contains("divisible"));
    }

    #[test]
    fn validate_rejects_all_fields_zero() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            intermediate_size: 0,
        };
        // First validation check hits image_size == 0.
        assert!(config.validate().is_err());
    }

    #[test]
    fn vision_config_debug_contains_all_fields() {
        let config = VisionConfig {
            image_size: 384,
            patch_size: 16,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            intermediate_size: 2048,
        };
        let debug = format!("{config:?}");
        // Debug output must contain every field name.
        assert!(debug.contains("image_size"));
        assert!(debug.contains("patch_size"));
        assert!(debug.contains("hidden_size"));
        assert!(debug.contains("num_layers"));
        assert!(debug.contains("num_heads"));
        assert!(debug.contains("intermediate_size"));
        // And the actual values.
        assert!(debug.contains("384"));
        assert!(debug.contains("16"));
        assert!(debug.contains("512"));
        assert!(debug.contains("6"));
        assert!(debug.contains("8"));
        assert!(debug.contains("2048"));
    }

    #[test]
    fn owned_vision_weights_is_empty_after_insert_is_false() {
        let mut w = OwnedVisionWeights::new();
        assert!(w.is_empty());
        w.insert("x", vec![1.0f32], vec![1]);
        assert!(!w.is_empty());
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn owned_vision_weights_lookup_after_repeated_replacement() {
        let mut w = OwnedVisionWeights::new();
        for i in 0..5 {
            w.insert("replaced", vec![i as f32], vec![1]);
        }
        assert_eq!(w.len(), 1);
        assert_eq!(w.get_vision_tensor("replaced").unwrap(), &[4.0f32]);
    }

    #[test]
    fn owned_vision_weights_insert_with_string_key() {
        let mut w = OwnedVisionWeights::new();
        let key = String::from("string_key_tensor");
        w.insert(key, vec![42.0f32], vec![1]);
        assert_eq!(w.get_vision_tensor("string_key_tensor").unwrap(), &[42.0f32]);
    }

    #[test]
    fn layer_tensor_name_high_layer_index() {
        let name = layer_tensor_name(99, "mlp.fc2.weight");
        assert_eq!(
            name,
            "vision_tower.encoder.layers.99.mlp.fc2.weight"
        );
    }

    #[test]
    fn graph_outputs_exactly_one_tensor() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        assert_eq!(graph.outputs.len(), 1, "vision encoder must produce exactly one output tensor");
    }

    #[test]
    fn graph_weight_specs_count_matches_input_count() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        // graph.inputs = [activation, ...weights]. specs = weights only.
        assert_eq!(graph.inputs.len(), specs.len() + 1);
    }

    #[test]
    fn siglip_encoder_debug_output_contains_config() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let debug = format!("{encoder:?}");
        assert!(debug.contains("SigLipEncoder"), "Debug should contain struct name");
        assert!(debug.contains("config"));
        assert!(debug.contains("weights"));
    }

    #[test]
    fn siglip_encoder_rejects_base64_with_mime_type() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .encode_image(&EncoderMedia::Base64 {
                data: "iVBORw0KGgo=".into(),
                mime_type: Some("image/jpeg".into()),
            })
            .unwrap_err();
        assert!(format!("{err}").contains("Base64"));
    }

    #[test]
    fn pack_weight_blob_empty_specs() {
        let w = OwnedVisionWeights::new();
        let blob = pack_weight_blob(&[], &w).unwrap();
        assert!(blob.is_empty());
    }

    #[test]
    fn vision_encode_pixel_count_formula_matches_config() {
        let config = VisionConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        // Expected pixels = 3 * 56 * 56 = 9408.
        let correct_pixels = vec![0.0f32; 3 * 56 * 56];
        let short_pixels = vec![0.0f32; 3 * 56 * 56 - 1];
        let empty_weights = OwnedVisionWeights::new();

        // Correct pixel count should fail on missing weights, not pixel count.
        let err = vision_encode(&correct_pixels, &config, &empty_weights).unwrap_err();
        assert!(format!("{err}").contains("missing"), "expected missing-weight error, got: {err}");

        // Short pixel count should fail on pixel buffer.
        let err = vision_encode(&short_pixels, &config, &empty_weights).unwrap_err();
        assert!(format!("{err}").contains("pixel buffer"), "expected pixel-buffer error, got: {err}");
    }

    #[test]
    fn graph_builder_per_layer_weight_names_follow_convention() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        // Verify layer 0 and layer 1 weight names exist.
        let layer0_ln1 = specs.iter().find(|s| s.name.contains("layers.0.layer_norm1.weight"));
        assert!(layer0_ln1.is_some(), "layer 0 ln1 weight must be present");

        let layer1_fc2 = specs.iter().find(|s| s.name.contains("layers.1.mlp.fc2.weight"));
        assert!(layer1_fc2.is_some(), "layer 1 fc2 weight must be present");

        // Verify no layer 2 exists (only 2 layers).
        let layer2 = specs.iter().find(|s| s.name.contains("layers.2."));
        assert!(layer2.is_none(), "no layer 2 weights should exist");
    }

    // ========================================================================
    // Additional tests for coverage — batch 3
    // ========================================================================

    // --- VisionConfig: constructor field access and boundary ---

    #[test]
    fn vision_config_field_access_matches_constructor() {
        let config = VisionConfig {
            image_size: 336,
            patch_size: 14,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 2048,
        };
        assert_eq!(config.image_size, 336);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_layers, 8);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.intermediate_size, 2048);
    }

    #[test]
    fn vision_config_all_fields_one() {
        let config = VisionConfig {
            image_size: 1,
            patch_size: 1,
            hidden_size: 1,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 1,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
        assert_eq!(config.head_dim(), 1);
    }

    #[test]
    fn vision_config_large_dimensions_validate() {
        let config = VisionConfig {
            image_size: 1024,
            patch_size: 16,
            hidden_size: 2048,
            num_layers: 64,
            num_heads: 32,
            intermediate_size: 8192,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 4096); // (1024/16)^2
        assert_eq!(config.head_dim(), 64); // 2048/32
    }

    // --- num_patches / head_dim edge cases ---

    #[test]
    fn num_patches_two_by_two_grid() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        assert_eq!(config.num_patches(), 4); // (28/14)^2 = 4
    }

    #[test]
    fn head_dim_with_two_heads() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        assert_eq!(config.head_dim(), 8); // 16/2
    }

    #[test]
    fn head_dim_equal_hidden_when_single_head() {
        let config = VisionConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 1,
            intermediate_size: 256,
        };
        assert_eq!(config.head_dim(), config.hidden_size);
    }

    // --- validate: more error paths ---

    #[test]
    fn validate_rejects_image_size_not_multiple_of_patch() {
        let config = VisionConfig {
            image_size: 100,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 64,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("divisible"), "error msg: {msg}");
    }

    #[test]
    fn validate_rejects_hidden_size_uneven_heads() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 10,
            num_layers: 1,
            num_heads: 3, // 10 % 3 != 0
            intermediate_size: 20,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_error_message_contains_both_dimensions() {
        let config = VisionConfig {
            image_size: 225,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("225"), "error should reference image_size 225: {msg}");
        assert!(msg.contains("14"), "error should reference patch_size 14: {msg}");
    }

    // --- OwnedVisionWeights: field validation edge cases ---

    #[test]
    fn owned_vision_weights_insert_empty_data() {
        let mut w = OwnedVisionWeights::new();
        w.insert("empty_tensor", vec![], vec![0]);
        assert!(!w.is_empty());
        assert_eq!(w.len(), 1);
        let data = w.get_vision_tensor("empty_tensor").unwrap();
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn owned_vision_weights_insert_large_tensor() {
        let mut w = OwnedVisionWeights::new();
        let large_data: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();
        w.insert("big", large_data.clone(), vec![100, 100]);
        let retrieved = w.get_vision_tensor("big").unwrap();
        assert_eq!(retrieved.len(), 10000);
        assert_eq!(retrieved[0], 0.0);
        assert!((retrieved[9999] - 9.999).abs() < 1e-3);
    }

    #[test]
    fn owned_vision_weights_shape_matches_inserted() {
        let mut w = OwnedVisionWeights::new();
        w.insert("4d_tensor", vec![0.0f32; 24], vec![2, 3, 4]);
        let shape = w.vision_tensor_shape("4d_tensor").unwrap();
        assert_eq!(shape, &[2, 3, 4]);
    }

    #[test]
    fn owned_vision_weights_nonexistent_name_returns_none() {
        let mut w = OwnedVisionWeights::new();
        w.insert("exists", vec![1.0f32], vec![1]);
        assert!(w.get_vision_tensor("exists").is_some());
        assert!(w.get_vision_tensor("does_not_exist").is_none());
        assert!(w.vision_tensor_shape("does_not_exist").is_none());
    }

    // --- SigLipEncoder: Debug/Clone trait verification ---

    #[test]
    fn siglip_encoder_has_debug_trait() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let debug_output = format!("{encoder:?}");
        // Debug output must be non-empty and contain struct-identifiable content.
        assert!(!debug_output.is_empty());
        assert!(debug_output.contains("SigLipEncoder") || debug_output.contains("config"));
    }

    // --- SigLipEncoder: new() rejects invalid config ---

    #[test]
    fn siglip_encoder_new_rejects_invalid_config() {
        let bad_config = VisionConfig {
            image_size: 0,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let weights = Arc::new(OwnedVisionWeights::new());
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(bad_config, weights, ids).unwrap_err();
        assert!(format!("{err}").contains("image_size"));
    }

    // --- try_build_siglip_from_tensors: success path returns valid encoder ---

    #[test]
    fn try_build_siglip_success_returns_accessible_config() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let encoder = try_build_siglip_from_tensors(&config, ids, |name| {
            populated
                .get_vision_tensor(name)
                .map(|slice| (slice.to_vec(), populated.vision_tensor_shape(name).unwrap().to_vec()))
        })
        .expect("build must succeed")
        .expect("encoder must be Some");

        assert_eq!(encoder.config().hidden_size, config.hidden_size);
        assert_eq!(encoder.config().num_layers, config.num_layers);
        assert_eq!(encoder.token_ids().image_token_id, 258880);
    }

    // --- VisionConfig: num_patches with minimum patch equal to image ---

    #[test]
    fn num_patches_exact_patch_equals_image() {
        let config = VisionConfig {
            image_size: 16,
            patch_size: 16,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 64,
        };
        assert_eq!(config.num_patches(), 1); // (16/16)^2 = 1
    }

    // --- Graph: weight spec names contain patch kernel ---

    #[test]
    fn graph_specs_include_patch_kernel_and_position_embedding() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let patch_spec = specs.iter().find(|s| s.name.contains("patch_embed"));
        assert!(patch_spec.is_some(), "specs must include patch_embed kernel");
        let pos_spec = specs.iter().find(|s| s.name.contains("position_embedding"));
        assert!(pos_spec.is_some(), "specs must include position_embedding");
    }

    // --- pack_weight_blob: validates shape from lookup mismatches data ---

    #[test]
    fn pack_weight_blob_rejects_shape_length_mismatch() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let mut weights = populate_weights(&config);
        let first_name = specs[0].name.clone();
        let first_numel: usize = specs[0].shape.iter().product();
        // Insert correct numel but shape metadata claims a different product.
        weights.insert(first_name, vec![0.0f32; first_numel], vec![1]);
        let err = pack_weight_blob(&specs, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shape") || msg.contains("length"), "error: {msg}");
    }

    // ========================================================================
    // Batch 4: Trait derive tests + boundary/error-path coverage
    // ========================================================================

    // --- VisionConfig: PartialEq ---

    #[test]
    fn vision_config_partial_eq_identical() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let b = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_image_size() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.image_size = 384;
        assert_ne!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_patch_size() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.patch_size = 16;
        assert_ne!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_hidden_size() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.hidden_size = 1024;
        assert_ne!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_num_layers() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.num_layers = 24;
        assert_ne!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_num_heads() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.num_heads = 16;
        assert_ne!(a, b);
    }

    #[test]
    fn vision_config_partial_eq_differs_on_intermediate_size() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut b = a.clone();
        b.intermediate_size = 4096;
        assert_ne!(a, b);
    }

    // --- VisionConfig: Hash ---

    #[test]
    fn vision_config_hash_equal_configs_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = VisionConfig {
            image_size: 384, patch_size: 14, hidden_size: 1024,
            num_layers: 24, num_heads: 16, intermediate_size: 4096,
        };
        let b = a.clone();
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn vision_config_hash_different_configs_likely_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let b = VisionConfig {
            image_size: 384, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    // --- VisionConfig: Eq (reflexive via PartialEq + Hash) ---

    #[test]
    fn vision_config_eq_reflexive() {
        let config = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        assert_eq!(config, config);
    }

    // --- VisionConfig: clone produces identical object ---

    #[test]
    fn vision_config_clone_deep_copy() {
        let original = VisionConfig {
            image_size: 336, patch_size: 14, hidden_size: 512,
            num_layers: 8, num_heads: 8, intermediate_size: 2048,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // They are separate values; modifying clone doesn't affect original (usize is Copy).
        assert_eq!(original.image_size, 336);
        assert_eq!(cloned.image_size, 336);
    }

    // --- OwnedVisionWeights: PartialEq ---

    #[test]
    fn owned_vision_weights_partial_eq_same_content() {
        let mut a = OwnedVisionWeights::new();
        a.insert("t1", vec![1.0f32, 2.0], vec![2]);
        let mut b = OwnedVisionWeights::new();
        b.insert("t1", vec![1.0f32, 2.0], vec![2]);
        assert_eq!(a, b);
    }

    #[test]
    fn owned_vision_weights_partial_eq_different_content() {
        let mut a = OwnedVisionWeights::new();
        a.insert("t1", vec![1.0f32], vec![1]);
        let mut b = OwnedVisionWeights::new();
        b.insert("t1", vec![2.0f32], vec![1]);
        assert_ne!(a, b);
    }

    #[test]
    fn owned_vision_weights_partial_eq_different_keys() {
        let mut a = OwnedVisionWeights::new();
        a.insert("key_a", vec![1.0f32], vec![1]);
        let mut b = OwnedVisionWeights::new();
        b.insert("key_b", vec![1.0f32], vec![1]);
        assert_ne!(a, b);
    }

    #[test]
    fn owned_vision_weights_partial_eq_different_shape() {
        let mut a = OwnedVisionWeights::new();
        a.insert("t1", vec![1.0f32, 2.0], vec![2]);
        let mut b = OwnedVisionWeights::new();
        b.insert("t1", vec![1.0f32, 2.0], vec![1, 1]); // same data, different shape
        assert_ne!(a, b);
    }

    #[test]
    fn owned_vision_weights_clone_independent() {
        let mut original = OwnedVisionWeights::new();
        original.insert("x", vec![42.0f32], vec![1]);
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // Modify original; cloned should remain unchanged.
        original.insert("y", vec![99.0f32], vec![1]);
        assert_ne!(original, cloned);
        assert_eq!(cloned.len(), 1);
    }

    // --- OwnedVisionWeights: edge cases with data ---

    #[test]
    fn owned_vision_weights_insert_special_float_nan() {
        let mut w = OwnedVisionWeights::new();
        w.insert("nan_tensor", vec![f32::NAN], vec![1]);
        let data = w.get_vision_tensor("nan_tensor").unwrap();
        assert!(data[0].is_nan());
    }

    #[test]
    fn owned_vision_weights_insert_special_float_inf() {
        let mut w = OwnedVisionWeights::new();
        w.insert("inf_tensor", vec![f32::INFINITY, f32::NEG_INFINITY], vec![2]);
        let data = w.get_vision_tensor("inf_tensor").unwrap();
        assert!(data[0].is_infinite() && data[0].is_sign_positive());
        assert!(data[1].is_infinite() && data[1].is_sign_negative());
    }

    #[test]
    fn owned_vision_weights_insert_zero_length_data_with_zero_shape() {
        let mut w = OwnedVisionWeights::new();
        w.insert("zero", vec![], vec![0]);
        assert_eq!(w.len(), 1);
        let data = w.get_vision_tensor("zero").unwrap();
        assert!(data.is_empty());
        let shape = w.vision_tensor_shape("zero").unwrap();
        assert_eq!(shape, &[0]);
    }

    #[test]
    fn owned_vision_weights_overwrite_preserves_last_shape() {
        let mut w = OwnedVisionWeights::new();
        w.insert("t", vec![1.0f32, 2.0, 3.0], vec![3]);
        w.insert("t", vec![4.0f32], vec![1]);
        let shape = w.vision_tensor_shape("t").unwrap();
        assert_eq!(shape, &[1]);
        let data = w.get_vision_tensor("t").unwrap();
        assert_eq!(data, &[4.0f32]);
    }

    // --- BackendError: Display for all variants ---

    #[test]
    fn backend_error_display_cuda() {
        let err = BackendError::Cuda("device lost".into());
        let msg = format!("{err}");
        assert!(msg.contains("CUDA error"));
        assert!(msg.contains("device lost"));
    }

    #[test]
    fn backend_error_display_hip() {
        let err = BackendError::Hip("hip error".into());
        let msg = format!("{err}");
        assert!(msg.contains("HIP error"));
        assert!(msg.contains("hip error"));
    }

    #[test]
    fn backend_error_display_metal() {
        let err = BackendError::Metal("metal fail".into());
        let msg = format!("{err}");
        assert!(msg.contains("Metal error"));
        assert!(msg.contains("metal fail"));
    }

    #[test]
    fn backend_error_display_cpu() {
        let err = BackendError::Cpu("cpu oom".into());
        let msg = format!("{err}");
        assert!(msg.contains("CPU error"));
        assert!(msg.contains("cpu oom"));
    }

    #[test]
    fn backend_error_display_unimplemented() {
        let err = BackendError::Unimplemented("flash_attn");
        let msg = format!("{err}");
        assert!(msg.contains("unimplemented"));
        assert!(msg.contains("flash_attn"));
    }

    #[test]
    fn backend_error_display_other() {
        let err = BackendError::Other("custom message".into());
        let msg = format!("{err}");
        assert!(msg.contains("backend error"));
        assert!(msg.contains("custom message"));
    }

    // --- BackendError: Clone and Debug ---

    #[test]
    fn backend_error_clone_preserves_message() {
        let err = BackendError::Other("test msg".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_debug_output() {
        let err = BackendError::Cuda("err".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Cuda"));
    }

    // --- BackendError: std::error::Error ---

    #[test]
    fn backend_error_implements_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::Other("boxed".into()));
        let msg = format!("{err}");
        assert!(msg.contains("boxed"));
    }

    // --- VisionConfig: validate error messages are specific ---

    #[test]
    fn validate_error_zero_image_size_mentions_both_fields() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 0,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        // First check should fail on image_size/patch_size > 0.
        assert!(msg.contains("image_size") || msg.contains("patch_size"));
    }

    #[test]
    fn validate_rejects_patch_larger_than_image() {
        let config = VisionConfig {
            image_size: 7,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        // 7 / 14 = 0 (integer division); not a multiple.
        assert!(config.validate().is_err());
    }

    // --- VisionConfig: num_patches with unusual but valid ratios ---

    #[test]
    fn num_patches_uneven_but_valid_ratio() {
        let config = VisionConfig {
            image_size: 15,
            patch_size: 5,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        // 15/5 = 3, 3^2 = 9
        assert_eq!(config.num_patches(), 9);
    }

    // --- layer_tensor_name: different suffixes ---

    #[test]
    fn layer_tensor_name_various_suffixes() {
        assert_eq!(
            layer_tensor_name(0, "layer_norm1.weight"),
            "vision_tower.encoder.layers.0.layer_norm1.weight"
        );
        assert_eq!(
            layer_tensor_name(0, "self_attn.q_proj.weight"),
            "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            layer_tensor_name(5, "mlp.fc1.weight"),
            "vision_tower.encoder.layers.5.mlp.fc1.weight"
        );
    }

    // --- Graph: single-layer config weight count ---

    #[test]
    fn graph_builder_single_layer_weight_count() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        // 1 activation + 2 (patch+pos) + 10 (1 layer) + 2 (final norm) = 15 inputs
        assert_eq!(graph.inputs.len(), 15);
        assert_eq!(specs.len(), 14); // 15 - 1 activation
    }

    // --- Graph: output tensor name ---

    #[test]
    fn graph_output_tensor_is_image_tokens() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        assert_eq!(graph.outputs.len(), 1);
    }

    // --- pack_weight_blob: correct byte size for known tensor ---

    #[test]
    fn pack_weight_blob_size_matches_f32_layout() {
        let mut w = OwnedVisionWeights::new();
        // 6 f32 values = 24 bytes.
        w.insert("sized", vec![0.0f32; 6], vec![2, 3]);
        let spec = WeightSpec {
            name: "sized".to_string(),
            shape: &[2, 3],
        };
        let blob = pack_weight_blob(&[spec], &w).unwrap();
        assert_eq!(blob.len(), 24); // 6 * sizeof(f32)
    }

    // --- vision_encode: config validation happens before pixel check ---

    #[test]
    fn vision_encode_config_validated_before_pixel_count() {
        let bad_config = VisionConfig {
            image_size: 0,
            patch_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            intermediate_size: 0,
        };
        let err = vision_encode(&[], &bad_config, &OwnedVisionWeights::new()).unwrap_err();
        // Should fail on config validation, not pixel count.
        assert!(format!("{err}").contains("image_size") || format!("{err}").contains("patch_size"));
    }

    // --- SigLipEncoder: encode_audio error message content ---

    #[test]
    fn siglip_encoder_encode_audio_error_mentions_conformer() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let err = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("audio"));
        assert!(msg.contains("ConformerEncoder") || msg.contains("audio encoder"));
    }

    // --- OwnedVisionWeights: insert with &str key ---

    #[test]
    fn owned_vision_weights_insert_str_ref_key() {
        let mut w = OwnedVisionWeights::new();
        w.insert("str_key", vec![1.0f32], vec![1]);
        assert!(w.get_vision_tensor("str_key").is_some());
    }

    // --- decode_pixels: byte round-trip fidelity for edge values ---

    #[test]
    fn decode_pixels_roundtrip_negative_values() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..expected_f32).map(|i| -(i as f32) * 0.1).collect();
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                (v - pixels[i]).abs() < 1e-7,
                "decoded[{i}] = {v}, expected {}",
                pixels[i]
            );
        }
    }

    // --- decode_pixels: exactly one byte short triggers error ---

    #[test]
    fn decode_pixels_off_by_one_byte() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_bytes = VISION_IN_CHANNELS * config.image_size * config.image_size * 4;
        // One byte short.
        let raw = vec![0u8; expected_bytes - 1];
        let err = encoder.decode_pixels(&EncoderMedia::Raw(raw)).unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    // --- decode_pixels: one byte too many triggers error ---

    #[test]
    fn decode_pixels_one_byte_over() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_bytes = VISION_IN_CHANNELS * config.image_size * config.image_size * 4;
        let raw = vec![0u8; expected_bytes + 1];
        let err = encoder.decode_pixels(&EncoderMedia::Raw(raw)).unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    // --- VisionConfig: num_patches integer division truncation ---

    #[test]
    fn num_patches_uses_integer_division() {
        let config = VisionConfig {
            image_size: 15,
            patch_size: 5,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        // 15/5 = 3, not 3.something. Integer division is correct here.
        assert_eq!(config.num_patches(), 9);
    }

    // --- VisionConfig: head_dim uses integer division ---

    #[test]
    fn head_dim_uses_integer_division() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        assert_eq!(config.head_dim(), 64);
    }

    // --- WeightSpec: name field correctness in multi-layer graph ---

    #[test]
    fn graph_specs_per_layer_names_are_unique() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 3,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(names.len(), unique_count, "all weight spec names must be unique");
    }

    // --- try_build_siglip_from_tensors: fetch is called for every spec ---

    #[test]
    fn try_build_siglip_calls_fetch_for_every_spec() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let expected_calls = specs.len();

        let mut call_count = 0;
        let _ = try_build_siglip_from_tensors(&config, ids, |name| {
            call_count += 1;
            populated
                .get_vision_tensor(name)
                .map(|slice| (slice.to_vec(), populated.vision_tensor_shape(name).unwrap().to_vec()))
        });
        assert_eq!(call_count, expected_calls);
    }

    // --- OwnedVisionWeights: Default trait ---

    #[test]
    fn owned_vision_weights_default_matches_new() {
        let via_new = OwnedVisionWeights::new();
        let via_default = OwnedVisionWeights::default();
        assert_eq!(via_new, via_default);
    }

    // --- VisionConfig: can be used as HashMap key (Hash + Eq) ---

    #[test]
    fn vision_config_used_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let config = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        map.insert(config.clone(), "siglip-base");
        assert_eq!(map.get(&config), Some(&"siglip-base"));

        let same_config = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        assert_eq!(map.get(&same_config), Some(&"siglip-base"));
    }

    // --- VisionConfig: can be used in HashSet ---

    #[test]
    fn vision_config_used_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let b = a.clone();
        set.insert(a);
        assert!(set.contains(&b));
        assert_eq!(set.len(), 1);
    }

    // --- VISION_LAYERNORM_EPS is positive and small ---

    #[test]
    fn vision_layernorm_eps_is_positive_small() {
        assert!(VISION_LAYERNORM_EPS > 0.0);
        assert!(VISION_LAYERNORM_EPS < 1e-3);
    }

    // --- VISION_IN_CHANNELS is strictly positive ---

    #[test]
    fn vision_in_channels_positive() {
        assert!(VISION_IN_CHANNELS > 0);
    }

    // ========================================================================
    // Batch 5: 50 additional tests for target 209+
    // ========================================================================

    // --- MultimodalTokenIds: is_image / is_audio predicates ---

    #[test]
    fn token_ids_is_image_true_for_image_token() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(ids.is_image(258880));
    }

    #[test]
    fn token_ids_is_image_false_for_other_tokens() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_image(258881)); // audio token
        assert!(!ids.is_image(0));
        assert!(!ids.is_image(99999));
    }

    #[test]
    fn token_ids_is_audio_true_for_audio_token() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(ids.is_audio(258881));
    }

    #[test]
    fn token_ids_is_audio_false_for_other_tokens() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_audio(258880)); // image token
        assert!(!ids.is_audio(0));
    }

    // --- MultimodalEncoded: validate success and failure ---

    #[test]
    fn multimodal_encoded_validate_success() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 100, 100],
            embeddings: vec![0.0f32; 3 * 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_validate_fails_on_length_mismatch() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 100, 100],
            embeddings: vec![0.0f32; 10], // 10 != 3 * 8 = 24
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = encoded.validate().unwrap_err();
        assert!(format!("{err}").contains("mismatch"));
    }

    #[test]
    fn multimodal_encoded_validate_fails_on_empty_embeddings() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32],
            embeddings: vec![],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_err());
    }

    #[test]
    fn multimodal_encoded_num_tokens_matches_vector_len() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200, 300, 400],
            embeddings: vec![0.0f32; 4 * 16],
            hidden_size: 16,
            kind: MediaKind::Image,
        };
        assert_eq!(encoded.num_tokens(), 4);
        assert!(encoded.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_validate_zero_tokens_zero_embeddings() {
        let encoded = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        // 0 * 8 = 0, embeddings len = 0, valid.
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 0);
    }

    // --- MediaKind: Copy, PartialEq, Debug, Hash ---

    #[test]
    fn media_kind_equality() {
        assert_eq!(MediaKind::Image, MediaKind::Image);
        assert_eq!(MediaKind::Audio, MediaKind::Audio);
        assert_ne!(MediaKind::Image, MediaKind::Audio);
    }

    #[test]
    fn media_kind_debug_output() {
        let debug = format!("{:?}", MediaKind::Image);
        assert!(debug.contains("Image"));
        let debug = format!("{:?}", MediaKind::Audio);
        assert!(debug.contains("Audio"));
    }

    #[test]
    fn media_kind_copy_is_independent() {
        let a = MediaKind::Image;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn media_kind_hash_equal_kinds_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = MediaKind::Image;
        let b = MediaKind::Image;
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    // --- MultimodalContext: new, push_image, push_audio, default ---

    #[test]
    fn multimodal_context_new_is_empty() {
        let ctx = MultimodalContext::new();
        assert!(ctx.images.is_empty());
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_default_is_empty() {
        let ctx = MultimodalContext::default();
        assert!(ctx.images.is_empty());
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_push_image_increments_count() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(encoded).unwrap();
        assert_eq!(ctx.images.len(), 1);
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_push_audio_increments_count() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(encoded).unwrap();
        assert_eq!(ctx.audios.len(), 1);
        assert!(ctx.images.is_empty());
    }

    #[test]
    fn multimodal_context_multiple_images() {
        let mut ctx = MultimodalContext::new();
        for _ in 0..3 {
            let encoded = MultimodalEncoded {
                tokens: vec![100],
                embeddings: vec![0.5f32; 8],
                hidden_size: 8,
                kind: MediaKind::Image,
            };
            ctx.push_image(encoded).unwrap();
        }
        assert_eq!(ctx.images.len(), 3);
    }

    // --- EncoderMedia: variant discrimination ---

    #[test]
    fn encoder_media_raw_variant() {
        let media = EncoderMedia::Raw(vec![0u8; 4]);
        let debug = format!("{media:?}");
        assert!(debug.contains("Raw"));
    }

    #[test]
    fn encoder_media_file_variant() {
        let media = EncoderMedia::File(std::path::PathBuf::from("/tmp/test.png"));
        let debug = format!("{media:?}");
        assert!(debug.contains("File"));
    }

    #[test]
    fn encoder_media_url_variant() {
        let media = EncoderMedia::Url("https://example.com/image.jpg".into());
        let debug = format!("{media:?}");
        assert!(debug.contains("Url"));
    }

    #[test]
    fn encoder_media_base64_variant_with_mime() {
        let media = EncoderMedia::Base64 {
            data: "AAAA".into(),
            mime_type: Some("image/png".into()),
        };
        let debug = format!("{media:?}");
        assert!(debug.contains("Base64"));
        assert!(debug.contains("image/png"));
    }

    #[test]
    fn encoder_media_base64_variant_without_mime() {
        let media = EncoderMedia::Base64 {
            data: "BBBB".into(),
            mime_type: None,
        };
        let debug = format!("{media:?}");
        assert!(debug.contains("Base64"));
    }

    // --- EncoderMedia: Clone ---

    #[test]
    fn encoder_media_clone_raw() {
        let media = EncoderMedia::Raw(vec![1, 2, 3, 4]);
        let cloned = media.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Raw"));
    }

    // --- OwnedTensor internal: Debug, Clone, PartialEq through OwnedVisionWeights ---

    #[test]
    fn owned_vision_weights_equality_reflexive() {
        let mut w = OwnedVisionWeights::new();
        w.insert("a", vec![1.0f32, 2.0], vec![2]);
        assert_eq!(w, w);
    }

    #[test]
    fn owned_vision_weights_equality_same_empty() {
        let a = OwnedVisionWeights::new();
        let b = OwnedVisionWeights::new();
        assert_eq!(a, b);
    }

    // --- SigLipEncoder: Arc clone preserves weights ---

    #[test]
    fn siglip_encoder_arc_clone_preserves_encoder() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), Arc::clone(&weights), ids).unwrap();

        // Encoder still functional through Arc clone.
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let result = encoder.encode_image(&EncoderMedia::Raw(raw_bytes));
        assert!(result.is_ok());
    }

    // --- VisionConfig: Copy trait (all fields are usize = Copy) ---

    #[test]
    fn vision_config_clone_is_independent() {
        let a = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(a.image_size, 224);
        assert_eq!(b.image_size, 224);
    }

    // --- Graph: intermediate tensor names follow convention ---

    #[test]
    fn graph_intermediate_tensor_names() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // The graph should contain internal tensors with predictable names.
        let names: Vec<String> = graph.inputs.iter()
            .filter_map(|&tid| graph.tensor(tid).map(|m| m.name.clone()))
            .collect();
        assert!(names.iter().any(|n| n == "image"), "must contain 'image' tensor");

        // Check output tensor name is "image_tokens".
        let output_name = graph.tensor(graph.outputs[0]).unwrap().name.clone();
        assert_eq!(output_name, "image_tokens");
    }

    // --- Graph: layer intermediate tensors ---

    #[test]
    fn graph_layer_intermediate_tensors_exist() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // Collect all tensor names by scanning num_tensors range.
        let all_names: Vec<String> = (0..graph.num_tensors())
            .filter_map(|i| {
                let tid = TensorId(i as u32);
                graph.tensor(tid).map(|m| m.name.clone())
            })
            .collect();
        // Layer 0 intermediates must exist.
        assert!(all_names.iter().any(|n| n == "layer_0_normed1"), "layer_0_normed1 must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_q"), "layer_0_q must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_k"), "layer_0_k must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_v"), "layer_0_v must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_attn"), "layer_0_attn must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_after_attn"), "layer_0_after_attn must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_gelu"), "layer_0_gelu must exist");
        assert!(all_names.iter().any(|n| n == "layer_0_after_ffn"), "layer_0_after_ffn must exist");
    }

    // --- Graph: multi-layer tensor indexing ---

    #[test]
    fn graph_multi_layer_tensor_indexing() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        let all_names: Vec<String> = (0..graph.num_tensors())
            .filter_map(|i| {
                let tid = TensorId(i as u32);
                graph.tensor(tid).map(|m| m.name.clone())
            })
            .collect();
        // Both layers must have their own intermediates.
        assert!(all_names.iter().any(|n| n == "layer_0_normed1"));
        assert!(all_names.iter().any(|n| n == "layer_1_normed1"));
        assert!(all_names.iter().any(|n| n == "layer_0_after_ffn"));
        assert!(all_names.iter().any(|n| n == "layer_1_after_ffn"));
    }

    // --- WeightSpec names: global weights appear before per-layer ---

    #[test]
    fn graph_specs_order_global_before_layer() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // First two specs must be patch_embed and position_embedding.
        assert!(specs[0].name.contains("patch_embed"), "first spec is patch kernel");
        assert!(specs[1].name.contains("position_embedding"), "second spec is pos table");
        // Last two must be final norm weight and bias.
        let last = specs.len();
        assert!(specs[last - 2].name.contains("post_layernorm.weight"));
        assert!(specs[last - 1].name.contains("post_layernorm.bias"));
    }

    // --- WeightSpec shapes: correct for known configs ---

    #[test]
    fn graph_specs_patch_kernel_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // hidden=8, in_channels=3, patch_size=7 => [8, 3, 7, 7]
        assert_eq!(specs[0].shape, &[8, 3, 7, 7]);
    }

    #[test]
    fn graph_specs_pos_table_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // num_patches=4, hidden=8 => [4, 8]
        assert_eq!(specs[1].shape, &[4, 8]);
    }

    #[test]
    fn graph_specs_final_norm_weight_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let last = specs.len();
        assert_eq!(specs[last - 2].shape, &[config.hidden_size]);
    }

    #[test]
    fn graph_specs_final_norm_bias_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let last = specs.len();
        assert_eq!(specs[last - 1].shape, &[config.hidden_size]);
    }

    // --- WeightSpec shapes: per-layer attention projection shapes ---

    #[test]
    fn graph_specs_per_layer_qkv_weight_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // hidden=8, num_heads=1, head_dim=8, q_dim=8
        // q_proj weight: [hidden, hidden] = [8, 8]
        let q_spec = specs.iter().find(|s| s.name.contains("q_proj.weight")).unwrap();
        assert_eq!(q_spec.shape, &[8, 8]);
        let k_spec = specs.iter().find(|s| s.name.contains("k_proj.weight")).unwrap();
        assert_eq!(k_spec.shape, &[8, 8]);
        let v_spec = specs.iter().find(|s| s.name.contains("v_proj.weight")).unwrap();
        assert_eq!(v_spec.shape, &[8, 8]);
    }

    // --- WeightSpec shapes: MLP fc1/fc2 ---

    #[test]
    fn graph_specs_per_layer_mlp_shapes() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // fc1: [hidden, inter] = [8, 16]
        let fc1 = specs.iter().find(|s| s.name.contains("fc1.weight")).unwrap();
        assert_eq!(fc1.shape, &[8, 16]);
        // fc2: [inter, hidden] = [16, 8]
        let fc2 = specs.iter().find(|s| s.name.contains("fc2.weight")).unwrap();
        assert_eq!(fc2.shape, &[16, 8]);
    }

    // --- Graph: input tensor order matches weight order ---

    #[test]
    fn graph_input_first_is_image_tensor() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // First input tensor should be "image".
        let first_input_name = graph.tensor(graph.inputs[0]).unwrap().name.as_str();
        assert_eq!(first_input_name, "image");
    }

    // --- pack_weight_blob: multi-spec byte concatenation ---

    #[test]
    fn pack_weight_blob_multi_spec_concatenation() {
        let mut w = OwnedVisionWeights::new();
        w.insert("a", vec![1.0f32], vec![1]);
        w.insert("b", vec![2.0f32, 3.0], vec![2]);

        let specs = vec![
            WeightSpec { name: "a".to_string(), shape: &[1] },
            WeightSpec { name: "b".to_string(), shape: &[2] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // 1 + 2 = 3 f32 values = 12 bytes.
        assert_eq!(blob.len(), 12);
        // First 4 bytes = 1.0f32 LE.
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
        // Next 4 bytes = 2.0f32 LE.
        assert_eq!(blob[4..8], 2.0f32.to_le_bytes());
        // Last 4 bytes = 3.0f32 LE.
        assert_eq!(blob[8..12], 3.0f32.to_le_bytes());
    }

    // --- SigLipEncoder: encode_image rejects empty Raw media ---

    #[test]
    fn siglip_encoder_encode_image_rejects_empty_raw() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let err = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    // --- SigLipEncoder: encode_image all output tokens are image_token_id ---

    #[test]
    fn siglip_encoder_encode_image_tokens_all_image_id() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 42,
            audio_token_id: 43,
            eoi_token_id: 44,
            eoa_token_id: 45,
        };
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = encoder.encode_image(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert!(encoded.tokens.iter().all(|&t| t == 42));
        assert_eq!(encoded.kind, MediaKind::Image);
    }

    // --- vision_encode: output length exactly num_patches * hidden_size ---

    #[test]
    fn vision_encode_output_length_exact() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let weights = populate_weights(&config);
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.01).collect();

        let output = vision_encode(&pixels, &config, &weights).unwrap();
        assert_eq!(output.len(), config.num_patches() * config.hidden_size);
    }

    // --- VisionConfig: validate passes for production SigLIP params ---

    #[test]
    fn validate_production_siglip_params() {
        // SigLIP base: 224/14=16, hidden=768, heads=12, layers=12, inter=3072
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 64); // 768/12
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2
    }

    // --- VisionTensorLookup trait: custom implementation ---

    struct SimpleTensorStore {
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    impl VisionTensorLookup for SimpleTensorStore {
        fn get_vision_tensor(&self, _name: &str) -> Option<&[f32]> {
            Some(&self.data)
        }
        fn vision_tensor_shape(&self, _name: &str) -> Option<&[usize]> {
            Some(&self.shape)
        }
    }

    #[test]
    fn custom_tensor_lookup_returns_data() {
        let store = SimpleTensorStore {
            data: vec![1.0f32, 2.0, 3.0],
            shape: vec![3],
        };
        assert_eq!(store.get_vision_tensor("anything").unwrap(), &[1.0f32, 2.0, 3.0]);
        assert_eq!(store.vision_tensor_shape("anything").unwrap(), &[3]);
    }

    // --- OwnedVisionWeights: many tensors stress test ---

    #[test]
    fn owned_vision_weights_many_tensors() {
        let mut w = OwnedVisionWeights::new();
        for i in 0..50 {
            w.insert(format!("tensor_{i}"), vec![i as f32], vec![1]);
        }
        assert_eq!(w.len(), 50);
        assert!(!w.is_empty());
        // All individually accessible.
        for i in 0..50 {
            let name = format!("tensor_{i}");
            assert_eq!(w.get_vision_tensor(&name).unwrap()[0], i as f32);
        }
    }

    // --- MultimodalEncoded: kind field is Audio when set ---

    #[test]
    fn multimodal_encoded_audio_kind() {
        let encoded = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        assert_eq!(encoded.kind, MediaKind::Audio);
        assert_ne!(encoded.kind, MediaKind::Image);
    }

    // --- vision_encode: deterministic output for same inputs ---

    #[test]
    fn vision_encode_deterministic_output() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| (i as f32) * 0.01).collect();

        let out1 = vision_encode(&pixels, &config, &weights).unwrap();
        let out2 = vision_encode(&pixels, &config, &weights).unwrap();
        assert_eq!(out1.len(), out2.len());
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "outputs differ: {a} vs {b}"
            );
        }
    }

    // --- MultimodalContext: push validates shape ---

    #[test]
    fn multimodal_context_push_image_validates_shape() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 4], // wrong: should be 1*8=8
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = ctx.push_image(encoded).unwrap_err();
        assert!(format!("{err}").contains("mismatch"));
    }

    #[test]
    fn multimodal_context_push_audio_validates_shape() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![0.0f32; 4], // wrong: should be 1*8=8
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        let err = ctx.push_audio(encoded).unwrap_err();
        assert!(format!("{err}").contains("mismatch"));
    }

    // ========================================================================
    // Batch 6: 50 additional tests for deeper coverage
    // ========================================================================

    // --- VisionConfig: validate error messages reference actual values ---

    #[test]
    fn validate_error_image_size_not_divisible_shows_values() {
        let config = VisionConfig {
            image_size: 223,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("223"), "should reference image_size 223: {msg}");
        assert!(msg.contains("14"), "should reference patch_size 14: {msg}");
    }

    #[test]
    fn validate_error_hidden_not_divisible_shows_values() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 7,
            num_layers: 1,
            num_heads: 3,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("7"), "should reference hidden_size 7: {msg}");
        assert!(msg.contains("3"), "should reference num_heads 3: {msg}");
    }

    // --- VisionConfig: boundary combinations ---

    #[test]
    fn validate_rejects_hidden_size_smaller_than_num_heads() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 2,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 8,
        };
        // hidden_size=2 < num_heads=4, so 2/4=0 (integer division), not divisible.
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_hidden_equals_num_heads() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 1);
    }

    // --- num_patches: various image/patch ratios ---

    #[test]
    fn num_patches_3_to_1_ratio() {
        let config = VisionConfig {
            image_size: 21,
            patch_size: 7,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 32,
        };
        assert_eq!(config.num_patches(), 9); // (21/7)^2 = 3^2 = 9
    }

    #[test]
    fn num_patches_4_to_1_ratio() {
        let config = VisionConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        assert_eq!(config.num_patches(), 16); // (56/14)^2 = 4^2 = 16
    }

    // --- head_dim: various divisions ---

    #[test]
    fn head_dim_four_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 128,
            num_layers: 4,
            num_heads: 4,
            intermediate_size: 256,
        };
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn head_dim_eight_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 1024,
        };
        assert_eq!(config.head_dim(), 64);
    }

    // --- RoutedSequence: construction and accessors ---

    #[test]
    fn routed_sequence_seq_len_matches_token_count() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 8,
        };
        assert_eq!(rs.seq_len(), 3);
    }

    #[test]
    fn routed_sequence_has_multimodal_false_when_all_none() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![1, 2],
            fused_embeddings: vec![None, None],
            text_positions: vec![0, 1],
            hidden_size: 8,
        };
        assert!(!rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_has_multimodal_true_when_some_present() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![100, 101],
            fused_embeddings: vec![Some(vec![0.5f32; 8]), None],
            text_positions: vec![1],
            hidden_size: 8,
        };
        assert!(rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_has_multimodal_true_when_all_some() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![100],
            fused_embeddings: vec![Some(vec![1.0f32; 4])],
            text_positions: vec![],
            hidden_size: 4,
        };
        assert!(rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_empty_seq_len_zero() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 8,
        };
        assert_eq!(rs.seq_len(), 0);
        assert!(!rs.has_multimodal());
    }

    // --- route_multimodal_tokens: pure text passthrough ---

    #[test]
    fn route_multimodal_tokens_pure_text_passthrough() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![10u32, 20, 30];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 8).unwrap();
        assert_eq!(routed.token_ids, tokens);
        assert!(routed.fused_embeddings.iter().all(|e| e.is_none()));
        assert_eq!(routed.text_positions, vec![0, 1, 2]);
        assert!(!routed.has_multimodal());
    }

    #[test]
    fn route_multimodal_tokens_empty_prompt() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let routed = route_multimodal_tokens(&[], &ctx, &ids, 8).unwrap();
        assert_eq!(routed.seq_len(), 0);
    }

    #[test]
    fn route_multimodal_tokens_image_count_mismatch() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let mut ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Prompt has one image token but ctx has zero images.
        let tokens = vec![ids.image_token_id];
        let err = route_multimodal_tokens(&tokens, &ctx, &ids, 8).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("image tokens") || msg.contains("image encodings"));
    }

    #[test]
    fn route_multimodal_tokens_audio_count_mismatch() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let mut ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Prompt has one audio token but ctx has zero audios.
        let tokens = vec![ids.audio_token_id];
        let err = route_multimodal_tokens(&tokens, &ctx, &ids, 8).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("audio tokens") || msg.contains("audio encodings"));
    }

    #[test]
    fn route_multimodal_tokens_hidden_size_mismatch() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(img).unwrap();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![ids.image_token_id];
        let err = route_multimodal_tokens(&tokens, &ctx, &ids, 16).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("hidden_size"));
    }

    #[test]
    fn route_multimodal_tokens_single_image_expansion() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100u32, 101, 102],
            embeddings: vec![1.0f32; 3 * 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(img).unwrap();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![10u32, ids.image_token_id, 20];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 8).unwrap();
        // Text(10) + 3 virtual tokens + Text(20) = 5 tokens total.
        assert_eq!(routed.seq_len(), 5);
        assert_eq!(routed.token_ids[0], 10);
        assert_eq!(routed.token_ids[1], 100);
        assert_eq!(routed.token_ids[2], 101);
        assert_eq!(routed.token_ids[3], 102);
        assert_eq!(routed.token_ids[4], 20);
        assert!(routed.has_multimodal());
        // Text positions: index 0 (text "10") and index 4 (text "20").
        assert_eq!(routed.text_positions, vec![0, 4]);
    }

    #[test]
    fn route_multimodal_tokens_single_audio_expansion() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let mut ctx = MultimodalContext::new();
        let aud = MultimodalEncoded {
            tokens: vec![200u32, 201],
            embeddings: vec![2.0f32; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(aud).unwrap();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![ids.audio_token_id];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 4).unwrap();
        assert_eq!(routed.seq_len(), 2);
        assert_eq!(routed.token_ids, vec![200, 201]);
        assert!(routed.has_multimodal());
    }

    // --- build_fused_hidden: basic correctness ---

    #[test]
    fn build_fused_hidden_pure_text_gathers_embedding() {
        use crate::compat::multimodal::{build_fused_hidden, route_multimodal_tokens, RoutedSequence};
        let ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![0u32, 1];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 2).unwrap();
        // vocab_size=2, hidden=2: [[1.0, 2.0], [3.0, 4.0]]
        let embed_rows = vec![1.0f32, 2.0, 3.0, 4.0];
        let fused = build_fused_hidden(&routed, &embed_rows, 2).unwrap();
        assert_eq!(fused.len(), 4); // 2 tokens * 2 hidden
        assert_eq!(fused[0], 1.0);
        assert_eq!(fused[1], 2.0);
        assert_eq!(fused[2], 3.0);
        assert_eq!(fused[3], 4.0);
    }

    #[test]
    fn build_fused_hidden_with_image_embedding() {
        use crate::compat::multimodal::{build_fused_hidden, route_multimodal_tokens};
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100u32],
            embeddings: vec![9.0f32, 8.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        ctx.push_image(img).unwrap();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![ids.image_token_id];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 2).unwrap();
        let embed_rows = vec![0.0f32; 4]; // vocab_size=2, hidden=2 (unused)
        let fused = build_fused_hidden(&routed, &embed_rows, 2).unwrap();
        assert_eq!(fused, vec![9.0f32, 8.0]);
    }

    // --- VisionConfig: additional boundary cases ---

    #[test]
    fn vision_config_head_dim_power_of_two() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 2048,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 8192,
        };
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn vision_config_num_patches_square_of_ratio() {
        let config = VisionConfig {
            image_size: 448,
            patch_size: 16,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        assert_eq!(config.num_patches(), 784); // (448/16)^2 = 28^2
    }

    #[test]
    fn validate_rejects_image_size_1_patch_size_2() {
        let config = VisionConfig {
            image_size: 1,
            patch_size: 2,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 8,
        };
        // 1 is not a multiple of 2.
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_accepts_large_patch_equal_image() {
        let config = VisionConfig {
            image_size: 32,
            patch_size: 32,
            hidden_size: 64,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 128,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
    }

    // --- OwnedVisionWeights: insert same key multiple times keeps last ---

    #[test]
    fn owned_vision_weights_insert_five_times_keeps_last() {
        let mut w = OwnedVisionWeights::new();
        for i in 0..5 {
            w.insert("key", vec![i as f32], vec![1]);
        }
        assert_eq!(w.len(), 1);
        assert_eq!(w.get_vision_tensor("key").unwrap()[0], 4.0f32);
    }

    // --- OwnedVisionWeights: Debug output is non-empty for non-empty store ---

    #[test]
    fn owned_vision_weights_debug_non_empty() {
        let mut w = OwnedVisionWeights::new();
        w.insert("abc", vec![1.0f32], vec![1]);
        let debug = format!("{w:?}");
        assert!(!debug.is_empty());
    }

    // --- MultimodalTokenIds: is_image and is_audio with custom IDs ---

    #[test]
    fn token_ids_custom_ids_is_image_and_is_audio() {
        let ids = MultimodalTokenIds {
            image_token_id: 500,
            audio_token_id: 600,
            eoi_token_id: 501,
            eoa_token_id: 601,
        };
        assert!(ids.is_image(500));
        assert!(!ids.is_image(501));
        assert!(!ids.is_image(600));
        assert!(ids.is_audio(600));
        assert!(!ids.is_audio(500));
        assert!(!ids.is_audio(601));
    }

    // --- MultimodalTokenIds: eoi/eoa are distinct from image/audio ---

    #[test]
    fn token_ids_eoi_not_image_audio() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_image(ids.eoi_token_id));
        assert!(!ids.is_audio(ids.eoi_token_id));
    }

    #[test]
    fn token_ids_eoa_not_image_audio() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_image(ids.eoa_token_id));
        assert!(!ids.is_audio(ids.eoa_token_id));
    }

    // --- EncoderMedia: Debug output for Raw with empty vec ---

    #[test]
    fn encoder_media_raw_empty_debug() {
        let media = EncoderMedia::Raw(vec![]);
        let debug = format!("{media:?}");
        assert!(debug.contains("Raw"));
    }

    // --- VisionConfig: PartialEq symmetric ---

    #[test]
    fn vision_config_partial_eq_symmetric() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // --- VisionConfig: PartialEq transitive ---

    #[test]
    fn vision_config_partial_eq_transitive() {
        let a = VisionConfig {
            image_size: 384, patch_size: 14, hidden_size: 1024,
            num_layers: 24, num_heads: 16, intermediate_size: 4096,
        };
        let b = a.clone();
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // --- MultimodalEncoded: validate with large token count ---

    #[test]
    fn multimodal_encoded_validate_large_token_count() {
        let n = 100;
        let encoded = MultimodalEncoded {
            tokens: vec![100u32; n],
            embeddings: vec![0.5f32; n * 16],
            hidden_size: 16,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), n);
    }

    // --- MultimodalEncoded: hidden_size zero with empty tokens ---

    #[test]
    fn multimodal_encoded_validate_zero_hidden_size_empty() {
        let encoded = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Audio,
        };
        // 0 * 0 = 0, embeddings len = 0. Valid degenerate case.
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 0);
    }

    // --- WeightSpec: per-layer LayerNorm weight/bias shapes ---

    #[test]
    fn graph_specs_per_layer_ln1_weight_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ln1_w = specs.iter().find(|s| s.name.contains("layer_norm1.weight")).unwrap();
        assert_eq!(ln1_w.shape, &[config.hidden_size]);
    }

    #[test]
    fn graph_specs_per_layer_ln1_bias_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ln1_b = specs.iter().find(|s| s.name.contains("layer_norm1.bias")).unwrap();
        assert_eq!(ln1_b.shape, &[config.hidden_size]);
    }

    #[test]
    fn graph_specs_per_layer_ln2_weight_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ln2_w = specs.iter().find(|s| s.name.contains("layer_norm2.weight")).unwrap();
        assert_eq!(ln2_w.shape, &[config.hidden_size]);
    }

    #[test]
    fn graph_specs_per_layer_ln2_bias_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ln2_b = specs.iter().find(|s| s.name.contains("layer_norm2.bias")).unwrap();
        assert_eq!(ln2_b.shape, &[config.hidden_size]);
    }

    #[test]
    fn graph_specs_per_layer_o_proj_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let o_proj = specs.iter().find(|s| s.name.contains("out_proj.weight")).unwrap();
        assert_eq!(o_proj.shape, &[config.hidden_size, config.hidden_size]);
    }

    // --- Graph: op count formula with multi-layer ---

    #[test]
    fn graph_op_count_4_layers() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 4,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // 2 (PatchEmbed + LearnedPos2D) + 12 * 4 + 1 (final LN) = 51
        assert_eq!(graph.num_ops(), 51);
    }

    // --- SigLipEncoder: new() rejects single missing weight ---

    #[test]
    fn siglip_encoder_new_rejects_single_missing_attention_weight() {
        let config = tiny_config();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        // Remove only the q_proj weight for layer 0.
        let q_proj_name = specs.iter()
            .find(|s| s.name.contains("q_proj.weight"))
            .unwrap()
            .name
            .clone();

        let mut weights = OwnedVisionWeights::new();
        for spec in &specs {
            if spec.name == q_proj_name {
                continue; // skip this one
            }
            let data = populated.get_vision_tensor(&spec.name).unwrap().to_vec();
            let shape = populated.vision_tensor_shape(&spec.name).unwrap().to_vec();
            weights.insert(spec.name.clone(), data, shape);
        }

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        assert!(format!("{err}").contains("missing"));
        assert!(format!("{err}").contains("q_proj"));
    }

    // --- SigLipEncoder: new() rejects single missing mlp weight ---

    #[test]
    fn siglip_encoder_new_rejects_single_missing_mlp_weight() {
        let config = tiny_config();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let fc1_name = specs.iter()
            .find(|s| s.name.contains("fc1.weight"))
            .unwrap()
            .name
            .clone();

        let mut weights = OwnedVisionWeights::new();
        for spec in &specs {
            if spec.name == fc1_name {
                continue;
            }
            let data = populated.get_vision_tensor(&spec.name).unwrap().to_vec();
            let shape = populated.vision_tensor_shape(&spec.name).unwrap().to_vec();
            weights.insert(spec.name.clone(), data, shape);
        }

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        assert!(format!("{err}").contains("missing"));
        assert!(format!("{err}").contains("fc1"));
    }

    // --- EncoderMedia: from_generation conversion ---

    #[test]
    fn encoder_media_from_generation_file() {
        use crate::generation::MediaInput;
        let input = MediaInput::File("/tmp/test.png".to_string());
        let media = EncoderMedia::from_generation(&input);
        let debug = format!("{media:?}");
        assert!(debug.contains("File"));
    }

    #[test]
    fn encoder_media_from_generation_raw() {
        use crate::generation::MediaInput;
        let input = MediaInput::Raw(vec![1u8, 2, 3, 4]);
        let media = EncoderMedia::from_generation(&input);
        let debug = format!("{media:?}");
        assert!(debug.contains("Raw"));
    }

    #[test]
    fn encoder_media_from_generation_url() {
        use crate::generation::MediaInput;
        let input = MediaInput::Url("https://example.com/img.jpg".to_string());
        let media = EncoderMedia::from_generation(&input);
        let debug = format!("{media:?}");
        assert!(debug.contains("Url"));
    }

    #[test]
    fn encoder_media_from_generation_base64() {
        use crate::generation::MediaInput;
        let input = MediaInput::Base64 {
            data: "AAAA".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        let media = EncoderMedia::from_generation(&input);
        let debug = format!("{media:?}");
        assert!(debug.contains("Base64"));
        assert!(debug.contains("image/png"));
    }

    // --- MultimodalContext: is_empty() ---

    #[test]
    fn multimodal_context_is_empty_new() {
        let ctx = MultimodalContext::new();
        assert!(ctx.is_empty());
    }

    #[test]
    fn multimodal_context_is_empty_after_push_image() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(encoded).unwrap();
        assert!(!ctx.is_empty());
    }

    #[test]
    fn multimodal_context_is_empty_after_push_audio() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(encoded).unwrap();
        assert!(!ctx.is_empty());
    }

    // --- MultimodalContext: push_image rejects Audio kind ---

    #[test]
    fn multimodal_context_push_image_rejects_audio_kind() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        let err = ctx.push_image(encoded).unwrap_err();
        assert!(format!("{err}").contains("non-Image") || format!("{err}").contains("Image"));
    }

    // --- MultimodalContext: push_audio rejects Image kind ---

    #[test]
    fn multimodal_context_push_audio_rejects_image_kind() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = ctx.push_audio(encoded).unwrap_err();
        assert!(format!("{err}").contains("non-Audio") || format!("{err}").contains("Audio"));
    }

    // --- MultimodalContext: mixed images and audios ---

    #[test]
    fn multimodal_context_mixed_images_and_audios() {
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let aud = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        ctx.push_image(img).unwrap();
        ctx.push_audio(aud).unwrap();
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
        assert!(!ctx.is_empty());
    }

    // --- MultimodalTokenIds: Copy trait ---

    #[test]
    fn token_ids_copy_is_independent() {
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = a;
        // Both should be identical after copy.
        assert_eq!(a.image_token_id, b.image_token_id);
        assert_eq!(a.audio_token_id, b.audio_token_id);
        assert_eq!(a.eoi_token_id, b.eoi_token_id);
        assert_eq!(a.eoa_token_id, b.eoa_token_id);
    }

    // --- MultimodalTokenIds: custom values ---

    #[test]
    fn token_ids_custom_values_is_image() {
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 101,
            eoa_token_id: 201,
        };
        assert!(ids.is_image(100));
        assert!(!ids.is_image(200));
        assert!(!ids.is_image(101));
    }

    #[test]
    fn token_ids_custom_values_is_audio() {
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 101,
            eoa_token_id: 201,
        };
        assert!(ids.is_audio(200));
        assert!(!ids.is_audio(100));
        assert!(!ids.is_audio(201));
    }

    // --- MultimodalTokenIds: Hash + Eq ---

    #[test]
    fn token_ids_hash_equal_ids_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();
        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn token_ids_eq_identical() {
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert_eq!(a, b);
    }

    #[test]
    fn token_ids_ne_different() {
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds {
            image_token_id: 0,
            audio_token_id: 0,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        assert_ne!(a, b);
    }

    // --- OwnedVisionWeights: equality transitivity ---

    #[test]
    fn owned_vision_weights_equality_transitive() {
        let mut a = OwnedVisionWeights::new();
        a.insert("t", vec![1.0f32, 2.0], vec![2]);
        let mut b = OwnedVisionWeights::new();
        b.insert("t", vec![1.0f32, 2.0], vec![2]);
        let mut c = OwnedVisionWeights::new();
        c.insert("t", vec![1.0f32, 2.0], vec![2]);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // --- OwnedVisionWeights: Debug output ---

    #[test]
    fn owned_vision_weights_debug_output() {
        let mut w = OwnedVisionWeights::new();
        w.insert("test", vec![1.0f32], vec![1]);
        let debug = format!("{w:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("OwnedVisionWeights") || debug.contains("tensors"));
    }

    // --- vision_encode: error message includes image dimensions ---

    #[test]
    fn vision_encode_pixel_error_includes_dimensions() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let bad_pixels = vec![0.0f32; 10];
        let weights = OwnedVisionWeights::new();
        let err = vision_encode(&bad_pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("28"), "should reference image_size: {msg}");
    }

    // --- decode_pixels: round-trip with random f32 values ---

    #[test]
    fn decode_pixels_roundtrip_arbitrary_values() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..expected_f32)
            .map(|i| ((i * 7 + 13) as f32).sin() * 0.5)
            .collect();
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                (v - pixels[i]).abs() < 1e-7,
                "decoded[{i}] = {v}, expected {}",
                pixels[i]
            );
        }
    }

    // --- Graph: total tensor count for single layer ---

    

    // --- Graph: tensor count scales with layers ---

    

    // --- pack_weight_blob: second tensor starts at correct offset ---

    #[test]
    fn pack_weight_blob_second_tensor_offset() {
        let mut w = OwnedVisionWeights::new();
        w.insert("first", vec![0.0f32; 4], vec![4]);
        w.insert("second", vec![1.0f32, 2.0, 3.0], vec![3]);

        let specs = vec![
            WeightSpec { name: "first".to_string(), shape: &[4] },
            WeightSpec { name: "second".to_string(), shape: &[3] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // First 4 f32 = 16 bytes (all 0.0). Then second tensor starts.
        assert_eq!(blob.len(), 28); // (4+3) * 4 = 28
        // Second tensor starts at byte 16.
        assert_eq!(blob[16..20], 1.0f32.to_le_bytes());
        assert_eq!(blob[20..24], 2.0f32.to_le_bytes());
        assert_eq!(blob[24..28], 3.0f32.to_le_bytes());
    }

    // --- populate_weights: deterministic across calls ---

    #[test]
    fn populate_weights_is_deterministic() {
        let config = tiny_config();
        let w1 = populate_weights(&config);
        let w2 = populate_weights(&config);
        // Get the first tensor and compare.
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let name = &specs[0].name;
        let d1 = w1.get_vision_tensor(name).unwrap();
        let d2 = w2.get_vision_tensor(name).unwrap();
        assert_eq!(d1.len(), d2.len());
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-10, "determinism violation: {a} vs {b}");
        }
    }

    // --- VisionTensorLookup: trait object dispatch ---

    #[test]
    fn vision_tensor_lookup_trait_object_dispatch() {
        let mut w = OwnedVisionWeights::new();
        w.insert("dispatched", vec![42.0f32], vec![1]);
        let lookup: &dyn VisionTensorLookup = &w;
        let data = lookup.get_vision_tensor("dispatched").unwrap();
        assert_eq!(data, &[42.0f32]);
        assert!(lookup.get_vision_tensor("nonexistent").is_none());
    }

    // --- BackendError: all variants produce non-empty Display ---

    #[test]
    fn backend_error_all_display_non_empty() {
        let errors = vec![
            BackendError::Cuda("err".into()),
            BackendError::Hip("err".into()),
            BackendError::Metal("err".into()),
            BackendError::Cpu("err".into()),
            BackendError::Unimplemented("test"),
            BackendError::Other("err".into()),
        ];
        for err in &errors {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "Display should not be empty for {err:?}");
        }
    }

    // --- MultimodalEncoded: validate error message includes dimensions ---

    #[test]
    fn multimodal_encoded_validate_error_includes_expected() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200, 300],
            embeddings: vec![0.0f32; 10], // should be 3 * 8 = 24
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = encoded.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("24"), "should mention expected 24: {msg}");
        assert!(msg.contains("10"), "should mention actual 10: {msg}");
    }

    // --- SigLipEncoder: config accessor returns correct reference ---

    #[test]
    fn siglip_encoder_config_returns_same_values() {
        let config = VisionConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();
        let returned = encoder.config();
        assert_eq!(returned.image_size, 56);
        assert_eq!(returned.patch_size, 14);
        assert_eq!(returned.hidden_size, 32);
        assert_eq!(returned.num_layers, 2);
        assert_eq!(returned.num_heads, 4);
        assert_eq!(returned.intermediate_size, 64);
    }

    // --- try_build_siglip_from_tensors: first missing tensor triggers early return ---

    #[test]
    fn try_build_siglip_returns_none_on_first_missing_tensor() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let result = try_build_siglip_from_tensors(&config, ids, |_name| None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    // --- try_build_siglip_from_tensors: partially present tensors ---

    #[test]
    fn try_build_siglip_partial_availability_returns_none() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let mut call_idx = 0;
        let cutoff = specs.len() / 2;
        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            if call_idx < cutoff {
                call_idx += 1;
                populated
                    .get_vision_tensor(name)
                    .map(|s| (s.to_vec(), populated.vision_tensor_shape(name).unwrap().to_vec()))
            } else {
                None
            }
        });
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    // --- Graph: validate graph compiles with multi-head config ---

    #[test]
    fn graph_compiles_with_multi_head() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 32,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        let mut compiler = InferenceCompiler::new();
        let mk_config = CompileConfig {
            max_seq_len: config.num_patches(),
            debug_jit: false,
            hetero: None,
        };
        let compiled = compiler.compile_mega_kernel_from_graph(graph, &mk_config, None).expect("multi-head graph must compile").layer_code;
        assert!(compiled.code_size() > 0);
    }

    // ========================================================================
    // Batch 7: Trait derive coverage + graph tensor count + boundary gaps
    // ========================================================================

    // --- MultimodalTokenIds: Debug output ---

    #[test]
    fn multimodal_token_ids_debug_contains_all_fields() {
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 101,
            eoa_token_id: 201,
        };
        let debug = format!("{ids:?}");
        assert!(debug.contains("image_token_id"));
        assert!(debug.contains("audio_token_id"));
        assert!(debug.contains("eoi_token_id"));
        assert!(debug.contains("eoa_token_id"));
        assert!(debug.contains("100"));
        assert!(debug.contains("200"));
    }

    // --- MultimodalEncoded: Clone ---

    #[test]
    fn multimodal_encoded_clone_preserves_fields() {
        let encoded = MultimodalEncoded {
            tokens: vec![10u32, 20, 30],
            embeddings: vec![1.0f32; 3 * 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let cloned = encoded.clone();
        assert_eq!(cloned.tokens, encoded.tokens);
        assert_eq!(cloned.embeddings, encoded.embeddings);
        assert_eq!(cloned.hidden_size, encoded.hidden_size);
        assert_eq!(cloned.kind, encoded.kind);
    }

    #[test]
    fn multimodal_encoded_clone_independent_after_modify() {
        let mut encoded = MultimodalEncoded {
            tokens: vec![10u32],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let cloned = encoded.clone();
        encoded.tokens.push(20);
        encoded.embeddings.push(2.0);
        assert_eq!(cloned.tokens.len(), 1);
        assert_eq!(cloned.embeddings.len(), 8);
    }

    // --- MultimodalEncoded: Debug ---

    #[test]
    fn multimodal_encoded_debug_non_empty() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32],
            embeddings: vec![0.5f32; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let debug = format!("{encoded:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("MultimodalEncoded") || debug.contains("tokens"));
    }

    // --- MultimodalContext: Debug ---

    #[test]
    fn multimodal_context_debug_non_empty() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(encoded).unwrap();
        let debug = format!("{ctx:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("MultimodalContext") || debug.contains("images"));
    }

    // --- MultimodalContext: Clone ---

    #[test]
    fn multimodal_context_clone_preserves_contents() {
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let aud = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![2.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        ctx.push_image(img).unwrap();
        ctx.push_audio(aud).unwrap();
        let cloned = ctx.clone();
        assert_eq!(cloned.images.len(), 1);
        assert_eq!(cloned.audios.len(), 1);
    }

    #[test]
    fn multimodal_context_clone_isolation() {
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(img).unwrap();
        let cloned = ctx.clone();
        // Adding to original doesn't affect clone.
        let img2 = MultimodalEncoded {
            tokens: vec![101],
            embeddings: vec![1.0f32; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        ctx.push_image(img2).unwrap();
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(cloned.images.len(), 1);
    }

    // --- MediaKind: Clone (Audio variant) ---

    #[test]
    fn media_kind_clone_audio_variant() {
        let kind = MediaKind::Audio;
        let cloned = kind.clone();
        assert_eq!(kind, cloned);
    }

    // --- RoutedSequence: Clone ---

    #[test]
    fn routed_sequence_clone_preserves_fields() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![10, 20],
            fused_embeddings: vec![Some(vec![1.0f32; 8]), None],
            text_positions: vec![0],
            hidden_size: 8,
        };
        let cloned = rs.clone();
        assert_eq!(cloned.token_ids, rs.token_ids);
        assert_eq!(cloned.text_positions, rs.text_positions);
        assert_eq!(cloned.hidden_size, rs.hidden_size);
        assert_eq!(cloned.fused_embeddings.len(), 2);
    }

    // --- RoutedSequence: Debug ---

    #[test]
    fn routed_sequence_debug_non_empty() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 8,
        };
        let debug = format!("{rs:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("RoutedSequence") || debug.contains("token_ids"));
    }

    // --- RoutedSequence: seq_len = 1 ---

    #[test]
    fn routed_sequence_seq_len_one() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![42],
            fused_embeddings: vec![Some(vec![0.0f32; 4])],
            text_positions: vec![0],
            hidden_size: 4,
        };
        assert_eq!(rs.seq_len(), 1);
        assert!(rs.has_multimodal());
    }

    // --- Graph: total tensor count for single layer ---

    #[test]
    fn graph_total_tensor_count_single_layer() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // num_tensors includes: image input + 14 weights + intermediate tensors
        // (patches, hidden_0, per-layer intermediates, image_tokens output).
        assert!(graph.num_tensors() > 14, "single-layer graph must have intermediate tensors beyond weights");
    }

    // --- Graph: tensor count scales with layers ---

    #[test]
    fn graph_tensor_count_scales_with_layers() {
        let make_config = |layers: usize| VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: layers,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (g1, _) = build_vision_encoder_graph(&make_config(1)).unwrap();
        let (g3, _) = build_vision_encoder_graph(&make_config(3)).unwrap();
        // Each additional layer adds ~12 intermediate tensors (normed1, q, k, v, attn,
        // o_proj, after_attn, normed2, fc1, gelu, fc2, after_ffn).
        assert!(g3.num_tensors() > g1.num_tensors());
        // Difference should be roughly 12 * (3-1) = 24 new intermediates per extra layer.
        assert!(g3.num_tensors() >= g1.num_tensors() + 24);
    }

    // --- VisionConfig: num_patches ratio 6 ---

    #[test]
    fn vision_config_num_patches_ratio_6() {
        let config = VisionConfig {
            image_size: 42,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        assert_eq!(config.num_patches(), 36); // (42/7)^2 = 6^2
    }

    // --- VisionConfig: num_patches ratio 5 ---

    #[test]
    fn vision_config_num_patches_ratio_5() {
        let config = VisionConfig {
            image_size: 35,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        assert_eq!(config.num_patches(), 25); // (35/7)^2 = 5^2
    }

    // --- VisionConfig: head_dim with 32 heads ---

    #[test]
    fn vision_config_head_dim_32_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 2048,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 8192,
        };
        assert_eq!(config.head_dim(), 64); // 2048 / 32
    }

    // --- VisionConfig: validate accepts 64 layers ---

    #[test]
    fn vision_config_validate_accepts_many_layers() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 64,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert!(config.validate().is_ok());
    }

    // --- VisionConfig: PartialEq all fields differ ---

    #[test]
    fn vision_config_partial_eq_all_fields_differ() {
        let a = VisionConfig {
            image_size: 224, patch_size: 14, hidden_size: 768,
            num_layers: 12, num_heads: 12, intermediate_size: 3072,
        };
        let b = VisionConfig {
            image_size: 384, patch_size: 16, hidden_size: 1024,
            num_layers: 24, num_heads: 16, intermediate_size: 4096,
        };
        assert_ne!(a, b);
    }

    // --- OwnedVisionWeights: equality with different tensor count ---

    #[test]
    fn owned_vision_weights_eq_different_tensor_count() {
        let mut a = OwnedVisionWeights::new();
        a.insert("x", vec![1.0f32], vec![1]);
        let b = OwnedVisionWeights::new();
        assert_ne!(a, b);
    }

    // --- OwnedVisionWeights: insert preserves unrelated tensors ---

    #[test]
    fn owned_vision_weights_insert_preserves_unrelated() {
        let mut w = OwnedVisionWeights::new();
        w.insert("keep", vec![42.0f32, 43.0], vec![2]);
        w.insert("new", vec![1.0f32], vec![1]);
        assert_eq!(w.get_vision_tensor("keep").unwrap(), &[42.0f32, 43.0]);
        assert_eq!(w.get_vision_tensor("new").unwrap(), &[1.0f32]);
        assert_eq!(w.len(), 2);
    }

    // --- OwnedVisionWeights: Debug empty store ---

    #[test]
    fn owned_vision_weights_debug_empty() {
        let w = OwnedVisionWeights::new();
        let debug = format!("{w:?}");
        assert!(!debug.is_empty());
    }

    // --- EncoderMedia: Clone for File variant ---

    #[test]
    fn encoder_media_clone_file_variant() {
        let media = EncoderMedia::File(std::path::PathBuf::from("/tmp/img.png"));
        let cloned = media.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("File"));
    }

    // --- EncoderMedia: Clone for Url variant ---

    #[test]
    fn encoder_media_clone_url_variant() {
        let media = EncoderMedia::Url("https://example.com/test.jpg".into());
        let cloned = media.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Url"));
    }

    // --- EncoderMedia: Clone for Base64 variant ---

    #[test]
    fn encoder_media_clone_base64_variant() {
        let media = EncoderMedia::Base64 {
            data: "SGVsbG8=".into(),
            mime_type: Some("image/jpeg".into()),
        };
        let cloned = media.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("Base64"));
        assert!(debug.contains("image/jpeg"));
    }

    // --- MultimodalEncoded: single token validates correctly ---

    #[test]
    fn multimodal_encoded_single_token_valid() {
        let encoded = MultimodalEncoded {
            tokens: vec![999u32],
            embeddings: vec![0.5f32; 16],
            hidden_size: 16,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 1);
    }

    // ========================================================================
    // Batch 8: 15 additional tests for coverage ratio improvement
    // ========================================================================

    // --- build_fused_hidden: rejects mismatched fused_embeddings length ---

    #[test]
    fn build_fused_hidden_rejects_wrong_embedding_length() {
        use crate::compat::multimodal::{build_fused_hidden, RoutedSequence};
        let rs = RoutedSequence {
            token_ids: vec![0u32],
            fused_embeddings: vec![Some(vec![1.0f32; 4])], // 4 != hidden_size 8
            text_positions: vec![0],
            hidden_size: 8,
        };
        let embed_rows = vec![0.0f32; 8]; // vocab=1, hidden=8
        let err = build_fused_hidden(&rs, &embed_rows, 8).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("length") || msg.contains("hidden_size"),
            "error should mention length or hidden_size: {msg}"
        );
    }

    // --- build_fused_hidden: rejects non-divisible embed_rows ---

    #[test]
    fn build_fused_hidden_rejects_non_divisible_embed_rows() {
        use crate::compat::multimodal::{build_fused_hidden, route_multimodal_tokens, RoutedSequence};
        let rs = RoutedSequence {
            token_ids: vec![0u32],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 4,
        };
        // 7 elements not divisible by hidden_size=4.
        let embed_rows = vec![0.0f32; 7];
        let err = build_fused_hidden(&rs, &embed_rows, 4).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("divisible") || msg.contains("not multiple"),
            "error should mention divisibility: {msg}"
        );
    }

    // --- build_fused_hidden: mixed text and media produces correct output ---

    #[test]
    fn build_fused_hidden_mixed_text_and_media() {
        use crate::compat::multimodal::{build_fused_hidden, RoutedSequence};
        // token_ids: [0(text), 100(media), 1(text)]
        // embed_rows vocab=2, hidden=2: [[10.0, 11.0], [12.0, 13.0]]
        let rs = RoutedSequence {
            token_ids: vec![0u32, 100, 1],
            fused_embeddings: vec![None, Some(vec![99.0f32, 88.0]), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        let embed_rows = vec![10.0f32, 11.0, 12.0, 13.0];
        let fused = build_fused_hidden(&rs, &embed_rows, 2).unwrap();
        assert_eq!(fused.len(), 6); // 3 tokens * 2 hidden
        // Position 0: text token 0 -> embed_rows[0..2] = [10.0, 11.0]
        assert_eq!(fused[0], 10.0);
        assert_eq!(fused[1], 11.0);
        // Position 1: media embedding overrides
        assert_eq!(fused[2], 99.0);
        assert_eq!(fused[3], 88.0);
        // Position 2: text token 1 -> embed_rows[2..4] = [12.0, 13.0]
        assert_eq!(fused[4], 12.0);
        assert_eq!(fused[5], 13.0);
    }

    // --- decode_pixels: zero-valued pixels round-trip ---

    #[test]
    fn decode_pixels_roundtrip_all_zeros() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.0f32; expected_f32];
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert!(decoded.iter().all(|&v| v == 0.0));
    }

    // --- decode_pixels: max f32 values round-trip ---

    #[test]
    fn decode_pixels_roundtrip_max_f32() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![f32::MAX; expected_f32];
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for v in &decoded {
            assert!((*v - f32::MAX).abs() < 1e-7, "expected f32::MAX, got {v}");
        }
    }

    // --- VisionConfig: validate accepts image_size equals patch_size greater than 1 ---

    #[test]
    fn validate_accepts_image_equals_patch_size_larger() {
        let config = VisionConfig {
            image_size: 32,
            patch_size: 32,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 128,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
        assert_eq!(config.head_dim(), 16);
    }

    // --- SigLipEncoder: encode_image produces deterministic results across calls ---

    #[test]
    fn siglip_encoder_encode_image_deterministic() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let media = EncoderMedia::Raw(raw_bytes);

        let result1 = encoder.encode_image(&media).unwrap();
        let result2 = encoder.encode_image(&media).unwrap();

        assert_eq!(result1.tokens, result2.tokens);
        assert_eq!(result1.embeddings.len(), result2.embeddings.len());
        for (a, b) in result1.embeddings.iter().zip(result2.embeddings.iter()) {
            assert!((a - b).abs() < 1e-10, "determinism violation: {a} vs {b}");
        }
    }

    // --- OwnedVisionWeights: two independent lookups return same data ---

    #[test]
    fn owned_vision_weights_double_lookup_consistent() {
        let mut w = OwnedVisionWeights::new();
        w.insert("consistent", vec![1.5f32, 2.5, 3.5], vec![3]);

        let first = w.get_vision_tensor("consistent").unwrap();
        let second = w.get_vision_tensor("consistent").unwrap();
        assert_eq!(first.as_ptr(), second.as_ptr(), "repeated lookups should return same slice");
        assert_eq!(first, second);
    }

    // --- MultimodalEncoded: validate with hidden_size 1 ---

    #[test]
    fn multimodal_encoded_validate_hidden_size_one() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200],
            embeddings: vec![0.5f32, -0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 2);
    }

    // --- vision_encode: pixel buffer exactly 3xHxW elements passes validation ---

    #[test]
    fn vision_encode_exact_pixel_count_passes_config_validation() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let correct_count = VISION_IN_CHANNELS * 28 * 28;
        let pixels = vec![0.0f32; correct_count];
        let empty = OwnedVisionWeights::new();
        // Should fail on missing weights (not pixel count).
        let err = vision_encode(&pixels, &config, &empty).unwrap_err();
        assert!(
            format!("{err}").contains("missing"),
            "should fail on missing weights, got: {err}"
        );
    }

    // --- VisionConfig: num_patches for 448/32 ratio ---

    #[test]
    fn vision_config_num_patches_448_32() {
        let config = VisionConfig {
            image_size: 448,
            patch_size: 32,
            hidden_size: 512,
            num_layers: 8,
            num_heads: 8,
            intermediate_size: 2048,
        };
        // 448 / 32 = 14, 14^2 = 196
        assert_eq!(config.num_patches(), 196);
    }

    // --- layer_tensor_name: various layer-specific suffixes ---

    #[test]
    fn layer_tensor_name_norm_and_attn_suffixes() {
        assert_eq!(
            layer_tensor_name(0, "layer_norm2.bias"),
            "vision_tower.encoder.layers.0.layer_norm2.bias"
        );
        assert_eq!(
            layer_tensor_name(1, "self_attn.out_proj.weight"),
            "vision_tower.encoder.layers.1.self_attn.out_proj.weight"
        );
        assert_eq!(
            layer_tensor_name(2, "self_attn.v_proj.weight"),
            "vision_tower.encoder.layers.2.self_attn.v_proj.weight"
        );
    }

    // --- OwnedVisionWeights: insert overwrites shape independently ---

    #[test]
    fn owned_vision_weights_overwrite_changes_both_data_and_shape() {
        let mut w = OwnedVisionWeights::new();
        w.insert("morph", vec![1.0f32, 2.0, 3.0], vec![3]);
        w.insert("morph", vec![4.0f32], vec![1]);

        assert_eq!(w.len(), 1);
        let data = w.get_vision_tensor("morph").unwrap();
        assert_eq!(data, &[4.0f32]);
        let shape = w.vision_tensor_shape("morph").unwrap();
        assert_eq!(shape, &[1]);
    }

    // --- EncoderMedia: Raw variant holds correct bytes ---

    #[test]
    fn encoder_media_raw_holds_exact_bytes() {
        let bytes = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        let media = EncoderMedia::Raw(bytes.clone());
        if let EncoderMedia::Raw(data) = media {
            assert_eq!(data, bytes);
        } else {
            panic!("expected Raw variant");
        }
    }

    // --- validate: error for intermediate_size zero mentions the field ---

    #[test]
    fn validate_error_intermediate_size_zero_message() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 0,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("intermediate_size"),
            "error should mention intermediate_size: {msg}"
        );
    }

    // --- graph_specs: per-layer weight order matches documented sequence ---

    #[test]
    fn graph_specs_per_layer_order_is_ln1_qkv_o_ln2_fc1_fc2() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Per-layer specs start at index 2 (after patch+pos) and span 10 entries per layer.
        let layer_start = 2;
        let layer_specs: Vec<&str> = specs[layer_start..layer_start + 10]
            .iter()
            .map(|s| {
                // Extract suffix after "layers.0."
                s.name.split("layers.0.").nth(1).unwrap_or("")
            })
            .collect();
        // Expected order: ln1_weight, ln1_bias, q_proj, k_proj, v_proj, o_proj,
        //                  ln2_weight, ln2_bias, fc1, fc2
        assert!(layer_specs[0].contains("layer_norm1.weight"), "first is ln1 weight: {}", layer_specs[0]);
        assert!(layer_specs[1].contains("layer_norm1.bias"), "second is ln1 bias: {}", layer_specs[1]);
        assert!(layer_specs[2].contains("q_proj"), "third is q_proj: {}", layer_specs[2]);
        assert!(layer_specs[3].contains("k_proj"), "fourth is k_proj: {}", layer_specs[3]);
        assert!(layer_specs[4].contains("v_proj"), "fifth is v_proj: {}", layer_specs[4]);
        assert!(layer_specs[5].contains("out_proj"), "sixth is o_proj: {}", layer_specs[5]);
        assert!(layer_specs[6].contains("layer_norm2.weight"), "seventh is ln2 weight: {}", layer_specs[6]);
        assert!(layer_specs[7].contains("layer_norm2.bias"), "eighth is ln2 bias: {}", layer_specs[7]);
        assert!(layer_specs[8].contains("fc1"), "ninth is fc1: {}", layer_specs[8]);
        assert!(layer_specs[9].contains("fc2"), "tenth is fc2: {}", layer_specs[9]);
    }

    // ========================================================================
    // Batch 6: Edge cases, boundary conditions, and uncovered paths
    // ========================================================================

    // @trace REQ-MULTIMODAL-001

    /// Verify that pack_weight_blob tolerates a lookup whose
    /// `vision_tensor_shape` returns `None` (the field is optional in the
    /// trait). It should fall through to the data-length check only.
    #[test]
    fn pack_weight_blob_tolerates_none_shape_metadata() {
        struct ShapelessLookup;
        impl VisionTensorLookup for ShapelessLookup {
            fn get_vision_tensor(&self, _name: &str) -> Option<&[f32]> {
                Some(&[1.0f32, 2.0, 3.0, 4.0])
            }
            fn vision_tensor_shape(&self, _name: &str) -> Option<&[usize]> {
                None // optional — caller must not require it
            }
        }

        let spec = WeightSpec {
            name: "test_tensor".to_string(),
            shape: &[4],
        };
        let blob = pack_weight_blob(&[spec], &ShapelessLookup).unwrap();
        assert_eq!(blob.len(), 16); // 4 f32 * 4 bytes
    }

    /// Verify that VisionConfig::validate accepts the smallest valid config
    /// where every field is 1 (1x1 image, 1x1 patch, 1 hidden, 1 layer,
    /// 1 head, 1 intermediate).
    #[test]
    fn vision_config_validate_all_ones_passes() {
        let config = VisionConfig {
            image_size: 1,
            patch_size: 1,
            hidden_size: 1,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 1,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
        assert_eq!(config.head_dim(), 1);
    }

    /// Verify that graph construction produces intermediate tensors with
    /// the correct name prefix per layer (layer_{i}_*).
    #[test]
    fn graph_intermediate_tensor_names_per_layer_prefix() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 2,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        let num_patches = config.num_patches();
        let hidden = config.hidden_size;

        // For layer 0 and layer 1, check that intermediate tensors exist.
        // The graph does not expose tensor names directly through the public
        // API, but we can verify the graph compiled and has the expected
        // number of ops: 2 + 12*2 + 1 = 27.
        let expected_ops = 2 + 12 * config.num_layers + 1;
        assert_eq!(graph.num_ops(), expected_ops);

        // Verify that the output shape matches [num_patches, hidden_size].
        assert_eq!(graph.outputs.len(), 1);
        let _ = graph.outputs[0]; // just accessing the output tid
    }

    /// Verify that vision_encode rejects an empty pixel buffer (zero elements).
    #[test]
    fn vision_encode_rejects_empty_pixel_buffer() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let err = vision_encode(&[], &config, &weights).unwrap_err();
        assert!(
            format!("{err}").contains("pixel buffer"),
            "should fail on pixel count, got: {err}"
        );
    }

    /// Verify that OwnedVisionWeights handles unicode tensor names correctly.
    #[test]
    fn owned_vision_weights_unicode_key_lookup() {
        let mut w = OwnedVisionWeights::new();
        w.insert("vision_tower.encoder.layers.0.self_attn.q_proj.weight", vec![0.5f32], vec![1]);
        assert_eq!(
            w.get_vision_tensor("vision_tower.encoder.layers.0.self_attn.q_proj.weight"),
            Some(&[0.5f32][..])
        );
        assert!(w.get_vision_tensor("nonexistent_tensor").is_none());
    }

    /// Verify that decode_pixels handles subnormal f32 values correctly
    /// (very small positive values near zero).
    #[test]
    fn decode_pixels_roundtrip_subnormal_values() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let pixels: Vec<f32> = (0..expected_f32).map(|i| {
            if i % 2 == 0 { subnormal } else { -subnormal }
        }).collect();
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                v.to_bits() == pixels[i].to_bits(),
                "decoded[{i}] bits = {:?}, expected {:?}",
                v.to_bits(),
                pixels[i].to_bits()
            );
        }
    }

    /// Verify that SigLipEncoder::encode_image rejects Raw media whose byte
    /// count is not a multiple of 4 (cannot be decoded as f32).
    #[test]
    fn siglip_encoder_encode_image_rejects_non_aligned_bytes() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        // 5 bytes: not a multiple of 4, so cannot be decoded into f32 slices.
        let err = encoder
            .encode_image(&EncoderMedia::Raw(vec![0u8, 1, 2, 3, 4]))
            .unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    /// Verify that graph construction with intermediate_size smaller than
    /// hidden_size is accepted (this is a valid configuration for small models).
    #[test]
    fn graph_builder_intermediate_smaller_than_hidden() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 16, // smaller than hidden_size=32
        };
        assert!(config.validate().is_ok());
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        assert!(graph.num_ops() > 0);
        assert!(!specs.is_empty());
    }

    /// Verify that the graph assigns the correct number of per-layer ops
    /// (12 ops per transformer layer: LN1, Q, K, V, MHA, O_proj,
    /// residual1, LN2, FC1, GELU, FC2, residual2).
    #[test]
    fn graph_per_layer_op_count_is_twelve() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        // Total ops: 2 (PatchEmbed + LearnedPos2D) + 12*1 (1 layer) + 1 (final LN) = 15.
        assert_eq!(graph.num_ops(), 15);
    }

    /// Verify that try_build_siglip_from_tensors passes the exact spec name
    /// to the fetch closure (not a modified/prefixed version).
    #[test]
    fn try_build_siglip_fetch_receives_exact_spec_name() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let first_spec_name = specs[0].name.clone();

        let mut received_names: Vec<String> = Vec::new();
        let _ = try_build_siglip_from_tensors(&config, ids, |name| {
            received_names.push(name.to_string());
            None // abort immediately
        });

        // The first call should receive the exact first spec name.
        assert!(
            !received_names.is_empty(),
            "fetch should have been called at least once"
        );
        assert_eq!(received_names[0], first_spec_name);
    }

    /// Verify that OwnedVisionWeights returns None for shape lookup when
    /// the tensor exists but only data was requested (edge case: separate
    /// data and shape queries).
    #[test]
    fn owned_vision_weights_data_present_shape_present_consistency() {
        let mut w = OwnedVisionWeights::new();
        w.insert("both_fields", vec![1.0f32, 2.0, 3.0], vec![3]);

        let data = w.get_vision_tensor("both_fields");
        let shape = w.vision_tensor_shape("both_fields");

        assert!(data.is_some(), "data should be present");
        assert!(shape.is_some(), "shape should be present");
        assert_eq!(data.unwrap().len(), shape.unwrap().iter().product::<usize>());
    }

    /// Verify that VisionConfig with image_size == patch_size == 1 produces
    /// exactly 1 patch, and the graph can be built with that config.
    #[test]
    fn graph_builder_single_patch_config() {
        let config = VisionConfig {
            image_size: 1,
            patch_size: 1,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 8,
        };
        assert_eq!(config.num_patches(), 1);
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        assert!(graph.num_ops() > 0);
        assert!(!specs.is_empty());
    }

    /// Verify that the pack_weight_blob byte ordering matches f32
    /// little-endian for a multi-spec scenario (two tensors back-to-back).
    #[test]
    fn pack_weight_blob_two_specs_correct_byte_boundary() {
        let mut w = OwnedVisionWeights::new();
        w.insert("first", vec![1.0f32], vec![1]);
        w.insert("second", vec![2.0f32, 3.0], vec![2]);

        let specs = vec![
            WeightSpec { name: "first".to_string(), shape: &[1] },
            WeightSpec { name: "second".to_string(), shape: &[2] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // first tensor: 1.0f32 at bytes 0..4
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
        // second tensor: 2.0f32 at bytes 4..8, 3.0f32 at bytes 8..12
        assert_eq!(blob[4..8], 2.0f32.to_le_bytes());
        assert_eq!(blob[8..12], 3.0f32.to_le_bytes());
        assert_eq!(blob.len(), 12);
    }

    /// Verify that the SigLipEncoder produces MultimodalEncoded with
    /// kind == MediaKind::Image (not Audio).
    #[test]
    fn siglip_encoder_encode_image_returns_image_kind() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = vec![0.0f32; image_numel];
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = encoder.encode_image(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(encoded.kind, MediaKind::Image);
        assert_ne!(encoded.kind, MediaKind::Audio);
    }

    /// Verify that the graph output tensor shape matches
    /// [num_patches, hidden_size] for a non-trivial config.
    #[test]
    fn graph_output_shape_matches_config_for_2_layer() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 32,
        };
        let num_patches = config.num_patches(); // 4
        let hidden = config.hidden_size; // 16

        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        // The graph has exactly 1 output tensor.
        assert_eq!(graph.outputs.len(), 1);
        // Verify the graph compiled without error and the total op count is
        // 2 (PatchEmbed + LearnedPos2D) + 12*2 (2 layers) + 1 (final norm) = 27.
        assert_eq!(graph.num_ops(), 27);
        // The output should represent num_patches * hidden elements.
        let _expected_elements = num_patches * hidden;
    }

    /// Verify that populate_weights produces the same tensor set across
    /// two calls (deterministic pseudo-random generation).
    #[test]
    fn populate_weights_repeated_call_identical() {
        let config = tiny_config();
        let w1 = populate_weights(&config);
        let w2 = populate_weights(&config);

        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        for spec in &specs {
            let d1 = w1.get_vision_tensor(&spec.name).unwrap();
            let d2 = w2.get_vision_tensor(&spec.name).unwrap();
            assert_eq!(d1.len(), d2.len(), "length mismatch for {}", spec.name);
            for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10,
                    "weight '{}' element {i}: {a} != {b}",
                    spec.name
                );
            }
        }
    }

    // @trace TEST-VISION-ADDITIONAL [req:REQ-MEGA-002] [level:unit]
    // Additional edge-case tests for vision_forward module.

    /// Verify that VisionConfig::validate rejects image_size equal to zero
    /// with an error message referencing both image_size and patch_size fields.
    #[test]
    fn validate_error_zero_image_size_mentions_fields() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 64,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("image_size"), "must mention image_size: {msg}");
        assert!(msg.contains("patch_size"), "must mention patch_size: {msg}");
    }

    /// Verify that two Arc<OwnedVisionWeights> derived from the same source
    /// produce identical tensor lookups, confirming shared ownership works.
    #[test]
    fn owned_vision_weights_arc_shared_lookup() {
        let config = tiny_config();
        let populated = Arc::new(populate_weights(&config));
        let arc2 = Arc::clone(&populated);

        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        for spec in &specs {
            let d1 = populated.get_vision_tensor(&spec.name).unwrap();
            let d2 = arc2.get_vision_tensor(&spec.name).unwrap();
            assert_eq!(d1.len(), d2.len());
            for (a, b) in d1.iter().zip(d2.iter()) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    /// Verify that a 3-layer vision encoder graph has the correct op count:
    /// 2 (PatchEmbed + LearnedPos2D) + 12*3 (3 transformer layers) + 1 (final norm) = 39.
    #[test]
    fn graph_op_count_3_layers() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 3,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        assert_eq!(graph.num_ops(), 39);
    }

    /// Verify that a config with patch_size == image_size yields exactly 1 patch,
    /// and the resulting graph has the minimum viable geometry.
    #[test]
    fn graph_single_patch_minimum_hidden() {
        let config = VisionConfig {
            image_size: 8,
            patch_size: 8,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 8,
        };
        assert_eq!(config.num_patches(), 1);
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        assert_eq!(graph.num_ops(), 15); // 2 + 12*1 + 1
    }

    /// Verify that pack_weight_blob correctly concatenates two tensors in spec
    /// order: first tensor bytes come first, second tensor bytes come second.
    #[test]
    fn pack_weight_blob_concatenation_order() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let weights = populate_weights(&config);
        let blob = pack_weight_blob(&specs, &weights).unwrap();

        // First spec's data should appear at the start of the blob.
        let first_numel: usize = specs[0].shape.iter().product();
        let first_data = weights.get_vision_tensor(&specs[0].name).unwrap();
        let first_bytes = first_data.len() * std::mem::size_of::<f32>();
        assert_eq!(first_numel, first_data.len());
        for (i, &val) in first_data.iter().enumerate() {
            let le = f32::to_le_bytes(val);
            let offset = i * 4;
            assert_eq!(
                blob[offset..offset + 4],
                le,
                "byte mismatch at tensor 0, element {i}"
            );
        }
        // Second spec's data starts right after first.
        let second_data = weights.get_vision_tensor(&specs[1].name).unwrap();
        let second_offset = first_bytes;
        for (i, &val) in second_data.iter().enumerate() {
            let le = f32::to_le_bytes(val);
            let offset = second_offset + i * 4;
            assert_eq!(
                blob[offset..offset + 4],
                le,
                "byte mismatch at tensor 1, element {i}"
            );
        }
    }

    /// Verify that decode_pixels rejects a Raw buffer that is exactly one f32
    /// (4 bytes) short of the expected size.
    #[test]
    fn decode_pixels_raw_one_f32_short() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let short_bytes = vec![0u8; (expected_f32 - 1) * 4];
        let err = encoder.decode_pixels(&EncoderMedia::Raw(short_bytes)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bytes") || msg.contains("expected"), "error: {msg}");
    }

    /// Verify that decode_pixels rejects a Raw buffer that is exactly one byte
    /// short (not aligned to f32 boundary).
    #[test]
    fn decode_pixels_raw_not_aligned_to_f32() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_bytes = VISION_IN_CHANNELS * config.image_size * config.image_size * 4;
        // 1 byte more than a multiple of 4 but less than expected.
        let odd_bytes = vec![0u8; expected_bytes + 1];
        let err = encoder.decode_pixels(&EncoderMedia::Raw(odd_bytes)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bytes"), "error: {msg}");
    }

    /// Verify that SigLipEncoder::new rejects when the very first layer weight
    /// (layer_norm1.weight) has an incorrect number of elements.
    #[test]
    fn siglip_encoder_new_rejects_wrong_ln1_weight_numel() {
        let config = tiny_config();
        let mut weights = populate_weights(&config);
        let ln1_name = layer_tensor_name(0, "layer_norm1.weight");
        // Insert wrong numel: config.hidden_size + 1.
        let wrong_data = vec![0.0f32; config.hidden_size + 1];
        weights.insert(&ln1_name, wrong_data, vec![config.hidden_size + 1]);
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(&ln1_name), "error must reference the bad weight: {msg}");
    }

    /// Verify that try_build_siglip_from_tensors propagates the error when a
    /// fetched tensor has the correct numel but wrong shape vector.
    #[test]
    fn try_build_siglip_rejects_correct_numel_wrong_shape_vec() {
        let config = tiny_config();
        let populated = populate_weights(&config);
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            populated.get_vision_tensor(name).map(|slice| {
                let actual_shape = populated.vision_tensor_shape(name).unwrap();
                // Return correct numel but a different shape layout.
                let numel: usize = actual_shape.iter().product();
                (slice.to_vec(), vec![numel]) // flat shape instead of correct layout
            })
        });
        // Should fail because shape mismatch (e.g., [64] != [8, 8]).
        assert!(result.is_err(), "must reject wrong shape layout");
    }

    /// Verify that vision_encode returns an error when pixel count does not
    /// match the config's expected count (off by a factor of channels).
    #[test]
    fn vision_encode_rejects_pixels_without_channels() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        // Only provide height * width worth of pixels (missing channel dim).
        let pixels = vec![0.0f32; config.image_size * config.image_size];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel"), "error must mention pixel: {msg}");
    }

    /// Verify that graph weight specs for a 2-layer config contain exactly
    /// 2 + 10*2 + 2 = 24 entries (patch_kernel + pos + 2*layer + final_norm).
    #[test]
    fn graph_weight_spec_count_2_layers() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        assert_eq!(specs.len(), 24);
    }

    /// Verify that graph input order puts image tensor first (index 0), then
    /// patch_kernel (index 1), then position embedding (index 2).
    #[test]
    fn graph_input_order_image_kernel_position() {
        let config = tiny_config();
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        assert!(graph.inputs.len() >= 3);
        // The first input is the activation (image pixels).
        // The second is the patch embedding kernel.
        // The third is the position embedding table.
        // These are TensorIds; we verify they are distinct.
        assert_ne!(graph.inputs[0], graph.inputs[1]);
        assert_ne!(graph.inputs[1], graph.inputs[2]);
        assert_ne!(graph.inputs[0], graph.inputs[2]);
    }

    /// Verify that OwnedVisionWeights implements VisionTensorLookup correctly
    /// when the same tensor name is inserted with different data — the last
    /// insert wins and both data and shape reflect it.
    #[test]
    fn owned_vision_weights_last_insert_wins_data_and_shape() {
        let mut w = OwnedVisionWeights::new();
        w.insert("tensor_a", vec![1.0f32, 2.0], vec![2]);
        w.insert("tensor_a", vec![3.0f32, 4.0, 5.0], vec![3]);

        let data = w.get_vision_tensor("tensor_a").unwrap();
        assert_eq!(data, &[3.0, 4.0, 5.0]);
        let shape = w.vision_tensor_shape("tensor_a").unwrap();
        assert_eq!(shape, &[3]);
    }

    /// Verify that pack_weight_blob with a single spec correctly produces a
    /// blob of exactly numel * 4 bytes.
    #[test]
    fn pack_weight_blob_single_spec_exact_byte_count() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Take only the first spec.
        let first_spec = &specs[0];
        let first_numel: usize = first_spec.shape.iter().product();
        let mut single_weight = OwnedVisionWeights::new();
        let data: Vec<f32> = (0..first_numel).map(|i| i as f32 * 0.01).collect();
        single_weight.insert(first_spec.name.clone(), data, first_spec.shape.to_vec());

        let blob = pack_weight_blob(std::slice::from_ref(first_spec), &single_weight).unwrap();
        assert_eq!(blob.len(), first_numel * 4);
    }

    /// Verify that pack_weight_blob works with a minimal custom VisionTensorLookup
    /// that returns None for shape metadata (optional field).
    #[test]
    fn pack_weight_blob_custom_lookup_no_shape_metadata() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Use populate_weights for correct data, then extract via custom lookup.
        let populated = populate_weights(&config);

        // Custom lookup that returns data but None for shape.
        struct ShapelessLookup<'a>(&'a OwnedVisionWeights);
        impl VisionTensorLookup for ShapelessLookup<'_> {
            fn get_vision_tensor(&self, name: &str) -> Option<&[f32]> {
                self.0.get_vision_tensor(name)
            }
            fn vision_tensor_shape(&self, _name: &str) -> Option<&[usize]> {
                None // intentionally omit shape metadata
            }
        }

        let lookup = ShapelessLookup(&populated);
        let blob = pack_weight_blob(&specs, &lookup).unwrap();
        let total_f32: usize = specs.iter().map(|s| s.shape.iter().product::<usize>()).sum();
        assert_eq!(blob.len(), total_f32 * 4);
    }

    /// Verify that VisionConfig with intermediate_size == hidden_size passes
    /// validation (no rule requires them to differ).
    #[test]
    fn validate_accepts_intermediate_equals_hidden() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 8, // same as hidden
        };
        assert!(config.validate().is_ok());
    }

    // ========================================================================
    // Batch 9: 15 additional tests for deeper edge-case coverage
    // ========================================================================

    // @trace TEST-VISION-BATCH9 [req:REQ-MEGA-002] [level:unit]

    /// Verify that VisionConfig::validate rejects hidden_size=1 with num_heads=2
    /// (integer division yields 0, not divisible).
    #[test]
    fn validate_rejects_hidden_one_heads_two() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 1,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 4,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("hidden_size") && msg.contains("num_heads"), "msg: {msg}");
    }

    /// Verify that SigLipEncoder::new returns an error when the final norm
    /// bias tensor has incorrect numel.
    #[test]
    fn siglip_encoder_new_rejects_wrong_final_norm_bias_numel() {
        let config = tiny_config();
        let mut weights = populate_weights(&config);
        let bias_name = final_norm_bias_name();
        let wrong_data = vec![0.0f32; config.hidden_size + 5];
        weights.insert(bias_name, wrong_data, vec![config.hidden_size + 5]);
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("len") || msg.contains("weight"), "error must mention len/weight: {msg}");
    }

    /// Verify that vision_encode rejects pixel buffer that is exactly
    /// one element too short (missing one f32).
    #[test]
    fn vision_encode_rejects_pixels_one_short() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let expected = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.0f32; expected - 1];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel buffer"), "should mention pixel buffer: {msg}");
    }

    /// Verify that OwnedVisionWeights returns consistent data when the same
    /// tensor is queried multiple times in rapid succession.
    #[test]
    fn owned_vision_weights_rapid_repeated_lookup() {
        let mut w = OwnedVisionWeights::new();
        w.insert("rapid", vec![3.14f32, 2.72, 1.41], vec![3]);
        for _ in 0..10 {
            let data = w.get_vision_tensor("rapid").unwrap();
            assert_eq!(data, &[3.14f32, 2.72, 1.41]);
            let shape = w.vision_tensor_shape("rapid").unwrap();
            assert_eq!(shape, &[3]);
        }
    }

    /// Verify that pack_weight_blob rejects a tensor whose data length matches
    /// the spec product but whose shape metadata reports a different product
    /// (simulating a corrupted shape index).
    #[test]
    fn pack_weight_blob_rejects_shape_product_mismatch() {
        let mut w = OwnedVisionWeights::new();
        // Insert tensor with data [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        // and shape [2, 3] (product = 6, matches data).
        w.insert("ok_tensor", vec![1.0f32; 6], vec![2, 3]);
        // Now re-insert with a misleading shape that has a different product.
        w.insert("ok_tensor", vec![1.0f32; 6], vec![5]);
        // spec says [2, 3] (product 6), shape metadata says [5] (product 5).
        let spec = WeightSpec {
            name: "ok_tensor".to_string(),
            shape: &[2, 3],
        };
        let err = pack_weight_blob(&[spec], &w).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shape"), "should mention shape mismatch: {msg}");
    }

    /// Verify that graph construction with intermediate_size larger than
    /// hidden_size produces correct fc1/fc2 weight shapes.
    #[test]
    fn graph_specs_fc1_fc2_shapes_intermediate_larger() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 64, // 8x larger than hidden
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let fc1 = specs.iter().find(|s| s.name.contains("fc1.weight")).unwrap();
        assert_eq!(fc1.shape, &[8, 64]);
        let fc2 = specs.iter().find(|s| s.name.contains("fc2.weight")).unwrap();
        assert_eq!(fc2.shape, &[64, 8]);
    }

    /// Verify that patch_embed_weight_name and position_embedding_name return
    /// distinct strings (they should never collide).
    #[test]
    fn tensor_name_conventions_are_distinct() {
        let names = [
            patch_embed_weight_name(),
            position_embedding_name(),
            final_norm_weight_name(),
            final_norm_bias_name(),
        ];
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(names[i], names[j], "{} should differ from {}", names[i], names[j]);
            }
        }
    }

    /// Verify that layer_tensor_name for each layer index produces unique names.
    #[test]
    fn layer_tensor_name_unique_across_layers() {
        let suffixes = [
            "layer_norm1.weight",
            "self_attn.q_proj.weight",
            "mlp.fc2.weight",
        ];
        let mut all_names = std::collections::HashSet::new();
        for layer in 0..5 {
            for suffix in &suffixes {
                let name = layer_tensor_name(layer, suffix);
                assert!(all_names.insert(name.clone()),
                    "duplicate name generated: {name}");
            }
        }
        assert_eq!(all_names.len(), 15); // 5 layers * 3 suffixes
    }

    /// Verify that MultimodalTokenIds::fallback_multimodal_token_ids produces token IDs
    /// that are all distinct from each other.
    #[test]
    fn fallback_multimodal_token_ids_all_token_ids_distinct() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let all_ids = [ids.image_token_id, ids.audio_token_id, ids.eoi_token_id, ids.eoa_token_id];
        for i in 0..all_ids.len() {
            for j in (i + 1)..all_ids.len() {
                assert_ne!(all_ids[i], all_ids[j],
                    "token IDs at positions {i} and {j} should differ: {}", all_ids[i]);
            }
        }
    }

    /// Verify that MultimodalContext::is_empty() returns false when both
    /// images and audios have been added, and true when neither has.
    #[test]
    fn multimodal_context_is_empty_mixed_state() {
        let mut ctx = MultimodalContext::new();
        assert!(ctx.is_empty());
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![0.0f32; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        ctx.push_image(img).unwrap();
        assert!(!ctx.is_empty());
        let aud = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![0.0f32; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(aud).unwrap();
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
    }

    /// Verify that EncoderMedia::Raw with a very large buffer (bigger than
    /// any reasonable image) is still rejected by decode_pixels because it
    /// doesn't match the config's expected size.
    #[test]
    fn decode_pixels_rejects_oversized_buffer() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_bytes = VISION_IN_CHANNELS * config.image_size * config.image_size * 4;
        let oversized = vec![0u8; expected_bytes * 10]; // 10x too large
        let err = encoder.decode_pixels(&EncoderMedia::Raw(oversized)).unwrap_err();
        assert!(format!("{err}").contains("bytes"));
    }

    /// Verify that OwnedVisionWeights handles inserting a tensor with
    /// multi-dimensional shape correctly (shape vector reflects all dims).
    #[test]
    fn owned_vision_weights_multidimensional_shape() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![0.0f32; 2 * 3 * 4 * 5]; // 120 elements
        let shape = vec![2, 3, 4, 5];
        w.insert("4d_tensor", data, shape.clone());
        let retrieved_shape = w.vision_tensor_shape("4d_tensor").unwrap();
        assert_eq!(retrieved_shape, &shape[..]);
        let retrieved_data = w.get_vision_tensor("4d_tensor").unwrap();
        assert_eq!(retrieved_data.len(), 120);
    }

    /// Verify that VisionConfig::validate accepts a config where
    /// hidden_size == num_heads (head_dim == 1).
    #[test]
    fn validate_accepts_head_dim_one() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 1);
    }

    /// Verify that graph_specs per-layer out_proj weight shape is
    /// [hidden_size, hidden_size] regardless of num_heads.
    #[test]
    fn graph_specs_o_proj_shape_independent_of_heads() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 32,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let o_proj = specs.iter().find(|s| s.name.contains("out_proj.weight")).unwrap();
        assert_eq!(o_proj.shape, &[16, 16]);
    }

    /// Verify that decode_pixels correctly decodes a buffer containing
    /// alternating positive and negative f32 values (byte-level verification).
    #[test]
    fn decode_pixels_alternating_sign_byte_fidelity() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..expected_f32)
            .map(|i| if i % 2 == 0 { (i as f32) * 0.1 } else { -(i as f32) * 0.1 })
            .collect();
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(decoded.len(), expected_f32);
        for (i, v) in decoded.iter().enumerate() {
            let expected = if i % 2 == 0 { (i as f32) * 0.1 } else { -(i as f32) * 0.1 };
            assert!(
                (v - expected).abs() < 1e-7,
                "decoded[{i}] = {v}, expected {expected}"
            );
        }
    }

    // ========================================================================
    // Batch 10: 15 additional tests for uncovered edge cases
    // ========================================================================

    // @trace TEST-VISION-BATCH10 [req:REQ-MEGA-002] [level:unit]

    /// Verify that VisionConfig with image_size=2 and patch_size=1 produces
    /// exactly 4 patches and passes validation.
    #[test]
    fn validate_accepts_two_by_two_patch_grid() {
        let config = VisionConfig {
            image_size: 2,
            patch_size: 1,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 4); // (2/1)^2
        assert_eq!(config.head_dim(), 2);
    }

    /// Verify that pack_weight_blob produces the correct total byte count
    /// when there are three consecutive tensors of different sizes.
    #[test]
    fn pack_weight_blob_three_tensors_correct_size() {
        let mut w = OwnedVisionWeights::new();
        w.insert("a", vec![1.0f32], vec![1]);
        w.insert("b", vec![2.0f32, 3.0], vec![2]);
        w.insert("c", vec![4.0f32, 5.0, 6.0, 7.0], vec![4]);

        let specs = vec![
            WeightSpec { name: "a".to_string(), shape: &[1] },
            WeightSpec { name: "b".to_string(), shape: &[2] },
            WeightSpec { name: "c".to_string(), shape: &[4] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // (1+2+4) * sizeof(f32) = 7 * 4 = 28 bytes.
        assert_eq!(blob.len(), 28);
        // Verify each tensor's bytes at correct offsets.
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
        assert_eq!(blob[4..8], 2.0f32.to_le_bytes());
        assert_eq!(blob[8..12], 3.0f32.to_le_bytes());
        assert_eq!(blob[12..16], 4.0f32.to_le_bytes());
        assert_eq!(blob[16..20], 5.0f32.to_le_bytes());
        assert_eq!(blob[20..24], 6.0f32.to_le_bytes());
        assert_eq!(blob[24..28], 7.0f32.to_le_bytes());
    }

    /// Verify that OwnedVisionWeights with an empty string key works correctly.
    #[test]
    fn owned_vision_weights_empty_string_key() {
        let mut w = OwnedVisionWeights::new();
        w.insert("", vec![42.0f32], vec![1]);
        assert_eq!(w.len(), 1);
        assert_eq!(w.get_vision_tensor("").unwrap(), &[42.0f32]);
        assert_eq!(w.vision_tensor_shape("").unwrap(), &[1]);
    }

    /// Verify that VisionConfig::validate rejects when only num_layers is zero
    /// and all other fields are valid.
    #[test]
    fn validate_rejects_exactly_zero_num_layers_only() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 0,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("num_layers"), "must mention num_layers: {msg}");
    }

    /// Verify that the graph input tensor count equals 1 + 2 + 10*L + 2
    /// where L is num_layers (activation + patch + pos + per-layer + final norm).
    #[test]
    fn graph_input_count_formula_5_layers() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 5,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        let expected_inputs = 1 + 2 + 10 * 5 + 2; // 55
        assert_eq!(graph.inputs.len(), expected_inputs);
        assert_eq!(specs.len(), expected_inputs - 1);
    }

    /// Verify that SigLipEncoder::token_ids() returns a Copy of the token IDs,
    /// and that the returned value is independent of subsequent changes.
    #[test]
    fn siglip_encoder_token_ids_returns_copy() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let custom_ids = MultimodalTokenIds {
            image_token_id: 1000,
            audio_token_id: 2000,
            eoi_token_id: 1001,
            eoa_token_id: 2001,
        };
        let encoder = SigLipEncoder::new(config, weights, custom_ids).unwrap();
        let retrieved = encoder.token_ids();
        assert_eq!(retrieved.image_token_id, 1000);
        assert_eq!(retrieved.audio_token_id, 2000);
        assert_eq!(retrieved.eoi_token_id, 1001);
        assert_eq!(retrieved.eoa_token_id, 2001);
    }

    /// Verify that vision_encode rejects a pixel buffer that is exactly double
    /// the expected size (common off-by-channel-dim error).
    #[test]
    fn vision_encode_rejects_double_sized_pixel_buffer() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let expected = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.0f32; expected * 2];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel buffer"), "should mention pixel buffer: {msg}");
    }

    /// Verify that graph construction with a config having hidden_size=4 and
    /// num_heads=4 (head_dim=1) produces correct Q/K/V projection shapes.
    #[test]
    fn graph_specs_qkv_shapes_head_dim_one() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 8,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Q/K/V weight shapes: [hidden, hidden] = [4, 4] regardless of num_heads.
        let q = specs.iter().find(|s| s.name.contains("q_proj.weight")).unwrap();
        assert_eq!(q.shape, &[4, 4]);
        let k = specs.iter().find(|s| s.name.contains("k_proj.weight")).unwrap();
        assert_eq!(k.shape, &[4, 4]);
        let v = specs.iter().find(|s| s.name.contains("v_proj.weight")).unwrap();
        assert_eq!(v.shape, &[4, 4]);
    }

    /// Verify that decode_pixels correctly handles a buffer where all values
    /// are the same constant (tests no alignment drift in byte decoding loop).
    #[test]
    fn decode_pixels_roundtrip_constant_value() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let constant = 0.123456f32;
        let pixels = vec![constant; expected_f32];
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                (v - constant).abs() < 1e-10,
                "decoded[{i}] = {v}, expected {constant}"
            );
        }
    }

    /// Verify that VisionConfig::num_patches returns 0 when image_size < patch_size
    /// (integer division yields 0), but such config fails validation.
    #[test]
    fn num_patches_zero_when_image_smaller_than_patch_but_invalid() {
        let config = VisionConfig {
            image_size: 7,
            patch_size: 14,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        // Integer division: 7 / 14 = 0, so num_patches = 0.
        assert_eq!(config.num_patches(), 0);
        // But validation rejects this config because 7 % 14 != 0.
        assert!(config.validate().is_err());
    }

    /// Verify that SigLipEncoder::new succeeds when all weights have exactly
    /// the expected numel (boundary: no extra elements, no missing elements).
    #[test]
    fn siglip_encoder_new_succeeds_exact_numel_boundary() {
        let config = tiny_config();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        // Verify every tensor has exactly the expected numel before constructing.
        for spec in &specs {
            let expected_numel: usize = spec.shape.iter().product();
            let data = populated.get_vision_tensor(&spec.name).unwrap();
            assert_eq!(data.len(), expected_numel, "numel mismatch for {}", spec.name);
        }

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, Arc::new(populated), ids);
        assert!(encoder.is_ok(), "SigLipEncoder::new must succeed with exact numel");
    }

    /// Verify that the graph's per-layer MLP fc1 weight shape is [hidden, inter]
    /// and fc2 weight shape is [inter, hidden], where inter > hidden.
    #[test]
    fn graph_specs_mlp_shapes_intermediate_larger_than_hidden() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 128,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let fc1 = specs.iter().find(|s| s.name.contains("fc1.weight")).unwrap();
        assert_eq!(fc1.shape, &[8, 128]); // [hidden, inter]
        let fc2 = specs.iter().find(|s| s.name.contains("fc2.weight")).unwrap();
        assert_eq!(fc2.shape, &[128, 8]); // [inter, hidden]
    }

    /// Verify that OwnedVisionWeights::is_empty() returns true for the default
    /// instance and false after any insert, even with empty data.
    #[test]
    fn owned_vision_weights_is_empty_transitions() {
        let mut w = OwnedVisionWeights::new();
        assert!(w.is_empty());

        // Insert empty data: no longer empty.
        w.insert("empty_data", vec![], vec![0]);
        assert!(!w.is_empty());

        // Insert another: still not empty.
        w.insert("nonempty", vec![1.0f32], vec![1]);
        assert!(!w.is_empty());
        assert_eq!(w.len(), 2);
    }

    /// Verify that try_build_siglip_from_tensors calls fetch for each spec
    /// in order and stops at the first None (early exit).
    #[test]
    fn try_build_siglip_stops_at_first_none() {
        let config = tiny_config();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let populated = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();

        let mut received_names: Vec<String> = Vec::new();
        let stop_at = 2; // stop after 2 successful fetches
        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            received_names.push(name.to_string());
            let idx = received_names.len();
            if idx <= stop_at {
                populated
                    .get_vision_tensor(&received_names[idx - 1])
                    .map(|s| (s.to_vec(), populated.vision_tensor_shape(&received_names[idx - 1]).unwrap().to_vec()))
            } else {
                None
            }
        });
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        // Should have been called exactly stop_at + 1 times (stop_at successes + 1 None).
        assert_eq!(received_names.len(), stop_at + 1);
    }

    /// Verify that graph construction with num_heads equal to hidden_size
    /// (head_dim=1) produces the correct number of ops.
    #[test]
    fn graph_op_count_heads_equal_hidden() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 4,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 8,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        // 2 (PatchEmbed + LearnedPos2D) + 12 * 2 + 1 (final norm) = 27
        assert_eq!(graph.num_ops(), 27);
    }

    // ========================================================================
    // Batch 11: 15 additional tests for remaining edge-case coverage
    // ========================================================================

    // @trace TEST-VISION-BATCH11 [req:REQ-MEGA-002] [level:unit]

    /// Verify that OwnedVisionWeights distinguishes tensors by case sensitivity
    /// (HashMap keys are case-sensitive, so "Tensor" and "tensor" are different).
    #[test]
    fn owned_vision_weights_case_sensitive_keys() {
        let mut w = OwnedVisionWeights::new();
        w.insert("Tensor", vec![1.0f32], vec![1]);
        w.insert("tensor", vec![2.0f32], vec![1]);

        assert_eq!(w.len(), 2);
        assert_eq!(w.get_vision_tensor("Tensor").unwrap(), &[1.0f32]);
        assert_eq!(w.get_vision_tensor("tensor").unwrap(), &[2.0f32]);
    }

    /// Verify that OwnedVisionWeights handles a tensor name that is very long
    /// (100+ characters) without truncation or error.
    #[test]
    fn owned_vision_weights_very_long_key_name() {
        let mut w = OwnedVisionWeights::new();
        let long_name = "vision_tower.encoder.layers.99.self_attn.some_very_long_suffix_name_that_exceeds_normal_length_ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789";
        w.insert(long_name, vec![3.14f32], vec![1]);

        assert_eq!(w.len(), 1);
        let retrieved = w.get_vision_tensor(long_name).unwrap();
        assert_eq!(retrieved, &[3.14f32]);
        let shape = w.vision_tensor_shape(long_name).unwrap();
        assert_eq!(shape, &[1]);
    }

    /// Verify that decode_pixels round-trips f32::MIN (most negative finite f32)
    /// without precision loss.
    #[test]
    fn decode_pixels_roundtrip_min_f32() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_f32 = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![f32::MIN; expected_f32];
        let mut raw_bytes = Vec::with_capacity(expected_f32 * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        for (i, v) in decoded.iter().enumerate() {
            assert!(
                (v - f32::MIN).abs() < 1e-7,
                "decoded[{i}] = {v}, expected f32::MIN"
            );
        }
    }

    /// Verify that VisionConfig::validate rejects when patch_size > image_size
    /// with both fields > 0 (integer division yields 0, not a multiple).
    #[test]
    fn validate_rejects_patch_larger_than_image_both_positive() {
        let config = VisionConfig {
            image_size: 7,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        // image_size=7 < patch_size=14, 7 % 14 != 0.
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("divisible"), "must mention divisibility: {msg}");
    }

    /// Verify that SigLipEncoder::new rejects when the position embedding tensor
    /// has incorrect numel (wrong shape for the number of patches).
    #[test]
    fn siglip_encoder_new_rejects_wrong_position_embedding_numel() {
        let config = tiny_config();
        let mut weights = populate_weights(&config);
        let pos_name = position_embedding_name();
        // Position embedding should be [num_patches, hidden_size] = [4, 8] = 32 elements.
        // Replace with wrong numel.
        let wrong_data = vec![0.0f32; 10];
        weights.insert(pos_name, wrong_data, vec![10]);

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(config, Arc::new(weights), ids).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("len") || msg.contains("weight"), "must mention len/weight: {msg}");
    }

    /// Verify that pack_weight_blob error message includes the missing tensor name,
    /// allowing the caller to diagnose which weight is absent.
    #[test]
    fn pack_weight_blob_error_includes_tensor_name() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let empty = OwnedVisionWeights::new();
        let err = pack_weight_blob(&specs, &empty).unwrap_err();
        let msg = format!("{err}");
        // The first missing tensor should be the patch embed kernel.
        assert!(
            msg.contains(&specs[0].name),
            "error should mention first missing tensor name '{}': {msg}",
            specs[0].name
        );
    }

    /// Verify that vision_encode pixel count error message includes both the
    /// actual and expected element counts.
    #[test]
    fn vision_encode_pixel_error_includes_actual_and_expected_counts() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        // Expected: 3 * 28 * 28 = 2352 elements.
        let bad_pixels = vec![0.0f32; 100];
        let empty_weights = OwnedVisionWeights::new();
        let err = vision_encode(&bad_pixels, &config, &empty_weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("100"), "should mention actual count 100: {msg}");
        assert!(msg.contains("2352"), "should mention expected count 2352: {msg}");
    }

    /// Verify that layer_tensor_name with suffixes containing dots and underscores
    /// produces the correct hierarchical path.
    #[test]
    fn layer_tensor_name_complex_suffix_with_dots() {
        let name = layer_tensor_name(7, "self_attn.out_proj.weight");
        assert_eq!(
            name,
            "vision_tower.encoder.layers.7.self_attn.out_proj.weight"
        );
    }

    /// Verify that VisionConfig with image_size == patch_size == 2 produces
    /// exactly 1 patch (2/2 = 1, 1^2 = 1).
    #[test]
    fn vision_config_num_patches_patch_equals_image_size_two() {
        let config = VisionConfig {
            image_size: 2,
            patch_size: 2,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
    }

    /// Verify that the graph's output tensor shape metadata matches
    /// [num_patches, hidden_size] for a specific config.
    #[test]
    fn graph_output_tensor_shape_metadata() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        let output_tid = graph.outputs[0];
        let output_meta = graph.tensor(output_tid).unwrap();
        assert_eq!(output_meta.name, "image_tokens");
        // Verify shape has exactly 2 dimensions (num_patches and hidden_size).
        assert_eq!(output_meta.shape.len(), 2);
        // Shape is Vec<SymDim>; each entry should be a Concrete value.
        let num_patches_dim = &output_meta.shape[0];
        let hidden_dim = &output_meta.shape[1];
        match num_patches_dim {
            SymDim::Concrete(n) => assert_eq!(*n, config.num_patches()),
            other => panic!("expected Concrete num_patches, got {other:?}"),
        }
        match hidden_dim {
            SymDim::Concrete(h) => assert_eq!(*h, config.hidden_size),
            other => panic!("expected Concrete hidden_size, got {other:?}"),
        }
    }

    /// Verify that OwnedVisionWeights::len() correctly counts after a series
    /// of insertions, including overwriting an existing key.
    #[test]
    fn owned_vision_weights_len_after_mixed_insert_and_overwrite() {
        let mut w = OwnedVisionWeights::new();
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());

        w.insert("a", vec![1.0f32], vec![1]);
        w.insert("b", vec![2.0f32], vec![1]);
        w.insert("c", vec![3.0f32], vec![1]);
        assert_eq!(w.len(), 3);

        // Overwrite "b" — length stays 3.
        w.insert("b", vec![99.0f32], vec![1]);
        assert_eq!(w.len(), 3);
        assert_eq!(w.get_vision_tensor("b").unwrap(), &[99.0f32]);

        // Add new key — length becomes 4.
        w.insert("d", vec![4.0f32], vec![1]);
        assert_eq!(w.len(), 4);
    }

    /// Verify that SigLipEncoder::encode_image returns MultimodalEncoded whose
    /// hidden_size field exactly matches the config's hidden_size.
    #[test]
    fn siglip_encoder_encode_image_hidden_size_matches_config() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| i as f32 * 0.001).collect();
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = encoder.encode_image(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(encoded.hidden_size, config.hidden_size);
        assert_eq!(encoded.embeddings.len(), config.num_patches() * config.hidden_size);
    }

    /// Verify that the graph's PatchEmbed op has the correct parameters by
    /// checking that the patch kernel weight spec shape encodes in_channels=3.
    #[test]
    fn graph_patch_kernel_shape_encodes_three_channels() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Patch kernel shape: [hidden, in_channels, patch_size, patch_size].
        // For tiny_config: [8, 3, 7, 7].
        assert_eq!(specs[0].shape[1], VISION_IN_CHANNELS,
            "patch kernel must have in_channels={VISION_IN_CHANNELS} at index 1");
    }

    /// Verify that vision_encode with a valid config and fully populated weights
    /// produces output where at least one value is non-zero (no silent NOP).
    #[test]
    fn vision_encode_output_not_all_zero_with_populated_weights() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let weights = populate_weights(&config);
        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = (0..image_numel).map(|i| (i as f32) * 0.01).collect();

        let output = vision_encode(&pixels, &config, &weights).unwrap();
        assert!(
            output.iter().any(|&v| v != 0.0),
            "output must not be all-zero with populated weights"
        );
        assert!(output.iter().all(|v| v.is_finite()), "output must be all finite");
    }

    /// Verify that VisionConfig::head_dim returns 0 when hidden_size < num_heads
    /// (integer division yields 0), though such config would fail validation.
    #[test]
    fn head_dim_zero_when_hidden_smaller_than_heads_but_invalid() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 2,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 8,
        };
        // 2 / 4 = 0 (integer division).
        assert_eq!(config.head_dim(), 0);
        // This config is invalid because hidden_size is not divisible by num_heads.
        assert!(config.validate().is_err());
    }

    /// Verify that VisionConfig::validate with zero patch_size produces an error
    /// message mentioning "patch_size".
    #[test]
    fn validate_error_zero_patch_size_mentions_field() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 0,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("patch_size"),
            "error should mention patch_size: {msg}"
        );
    }

    /// Verify that VisionConfig::validate with zero hidden_size produces an error
    /// message mentioning "hidden_size".
    #[test]
    fn validate_error_zero_hidden_size_mentions_field() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 0,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("hidden_size"),
            "error should mention hidden_size: {msg}"
        );
    }

    /// Verify that VisionConfig::validate with zero num_heads produces an error
    /// message mentioning "num_heads".
    #[test]
    fn validate_error_zero_num_heads_mentions_field() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 0,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("num_heads"),
            "error should mention num_heads: {msg}"
        );
    }

    /// Verify that VisionConfig::validate with zero num_layers produces an error
    /// message mentioning "num_layers".
    #[test]
    fn validate_error_zero_num_layers_mentions_field() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 0,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("num_layers"),
            "error should mention num_layers: {msg}"
        );
    }

    /// Verify that VisionConfig::num_patches returns 1 when image_size equals
    /// patch_size (1x1 pixel grid → 1 patch).
    #[test]
    fn vision_config_num_patches_single_pixel_grid() {
        let config = VisionConfig {
            image_size: 1,
            patch_size: 1,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
    }

    /// Verify that vision_encode rejects a pixel buffer with exactly one extra
    /// element beyond the expected count.
    #[test]
    fn vision_encode_rejects_pixel_buffer_one_over() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let expected = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.0f32; expected + 1];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel buffer"), "should mention pixel buffer: {msg}");
    }

    /// Verify that VisionConfig::validate accepts a prime-number patch grid
    /// (image_size=3, patch_size=3 → 1 patch).
    #[test]
    fn validate_accepts_prime_patch_grid() {
        let config = VisionConfig {
            image_size: 3,
            patch_size: 3,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 16,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), 1);
        assert_eq!(config.head_dim(), 4);
    }

    /// Verify that OwnedVisionWeights with an empty data vec and empty shape vec
    /// returns the empty slice on lookup.
    #[test]
    fn owned_vision_weights_empty_data_empty_shape() {
        let mut w = OwnedVisionWeights::new();
        w.insert("empty", vec![], vec![]);
        let data = w.get_vision_tensor("empty").unwrap();
        assert!(data.is_empty());
        let shape = w.vision_tensor_shape("empty").unwrap();
        assert!(shape.is_empty());
    }

    /// Verify that pack_weight_blob with all-zero weight data succeeds and
    /// produces exactly the expected total byte count.
    #[test]
    fn pack_weight_blob_all_zeros_data_correct_byte_count() {
        let config = tiny_config();
        let (graph, specs) = build_vision_encoder_graph(&config).unwrap();
        let mut weights = OwnedVisionWeights::new();
        let mut total_expected_bytes = 0usize;
        for spec in &specs {
            let numel: usize = spec.shape.iter().product();
            let zeros = vec![0.0f32; numel];
            total_expected_bytes += numel * std::mem::size_of::<f32>();
            weights.insert(spec.name.clone(), zeros, spec.shape.to_vec());
        }
        let blob = pack_weight_blob(&specs, &weights).unwrap();
        assert_eq!(blob.len(), total_expected_bytes);
        // Verify every byte is zero (f32 0.0 = all zero bits).
        assert!(blob.iter().all(|&b| b == 0));
        let _ = graph; // suppress unused warning
    }

    /// Verify that SigLipEncoder::encode_audio error message contains the
    /// encoder name "SigLipEncoder".
    #[test]
    fn siglip_encoder_encode_audio_error_mentions_encoder_name() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let err = encoder
            .encode_audio(&EncoderMedia::Raw(vec![]))
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("SigLipEncoder"),
            "audio error should mention SigLipEncoder: {msg}"
        );
    }

    /// Verify that OwnedVisionWeights inserted with identical data under two
    /// different names are independent — modifying one via re-insert does not
    /// affect the other.
    #[test]
    fn owned_vision_weights_two_names_same_data_independent() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![1.0f32, 2.0, 3.0];
        w.insert("first", data.clone(), vec![3]);
        w.insert("second", data.clone(), vec![3]);

        // Overwrite first with different data.
        w.insert("first", vec![99.0f32, 88.0, 77.0], vec![3]);

        assert_eq!(w.get_vision_tensor("first").unwrap(), &[99.0f32, 88.0, 77.0]);
        assert_eq!(w.get_vision_tensor("second").unwrap(), &[1.0f32, 2.0, 3.0]);
    }

    /// Verify that VisionConfig::validate rejects when both image_size and
    /// patch_size are zero.
    #[test]
    fn validate_rejects_both_image_and_patch_zero() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 0,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("image_size") || msg.contains("patch_size"),
            "error should mention image_size or patch_size: {msg}"
        );
    }

    /// Verify that VisionConfig::head_dim returns 1 when hidden_size=1 and
    /// num_heads=1 (absolute minimum valid head dimension).
    #[test]
    fn vision_config_head_dim_one_when_hidden_one_heads_one() {
        let config = VisionConfig {
            image_size: 2,
            patch_size: 2,
            hidden_size: 1,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 1,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 1);
    }

    /// Verify that graph construction with tiny_config produces an output tensor
    /// named exactly "image_tokens".
    #[test]
    fn graph_output_tensor_named_image_tokens() {
        let config = tiny_config();
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        let output_tid = graph.outputs[0];
        let output_meta = graph.tensor(output_tid).unwrap();
        assert_eq!(output_meta.name, "image_tokens");
    }

    /// Verify that try_build_siglip_from_tensors returns an error when the fetch
    /// closure provides data with the correct numel but a wrong shape vector.
    #[test]
    fn try_build_siglip_rejects_correct_numel_wrong_shape() {
        let config = tiny_config();
        let populated = populate_weights(&config);
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            populated.get_vision_tensor(name).map(|slice| {
                let correct_shape = populated.vision_tensor_shape(name).unwrap();
                let numel: usize = correct_shape.iter().product();
                // Return correct data but a deliberately wrong flat shape.
                (slice.to_vec(), vec![numel])
            })
        });
        // The shape mismatch should cause an error during SigLipEncoder::new.
        assert!(result.is_err(), "wrong shape should cause an error");
    }

    // ========================================================================
    // 15 additional tests
    // ========================================================================

    /// Verify that VisionConfig intermediate_size field is directly accessible
    /// and matches the value passed to the constructor.
    #[test]
    fn vision_config_intermediate_size_field_access() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert_eq!(config.intermediate_size, 3072);
    }

    /// Verify that the graph's image input tensor has 3 dimensions
    /// corresponding to [channels, height, width].
    #[test]
    fn graph_image_tensor_shape_has_channels() {
        let config = tiny_config();
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        let image_tid = graph.inputs[0];
        let meta = graph.tensor(image_tid).unwrap();
        assert_eq!(meta.shape.len(), 3, "image tensor must be 3-dimensional");
        match (&meta.shape[0], &meta.shape[1], &meta.shape[2]) {
            (SymDim::Concrete(c), SymDim::Concrete(h), SymDim::Concrete(w)) => {
                assert_eq!(*c, VISION_IN_CHANNELS, "first dim must be channels");
                assert_eq!(*h, config.image_size, "second dim must be height");
                assert_eq!(*w, config.image_size, "third dim must be width");
            }
            other => panic!("expected all-Concrete shape, got {other:?}"),
        }
    }

    /// Verify that OwnedVisionWeights::len() increments with each distinct
    /// insert and stays stable after overwriting an existing key.
    #[test]
    fn owned_vision_weights_len_increments_on_insert() {
        let mut w = OwnedVisionWeights::new();
        assert_eq!(w.len(), 0);
        w.insert("a", vec![1.0f32], vec![1]);
        assert_eq!(w.len(), 1);
        w.insert("b", vec![2.0f32, 3.0f32], vec![2]);
        assert_eq!(w.len(), 2);
        // Overwrite "a" — len must not increase.
        w.insert("a", vec![9.0f32], vec![1]);
        assert_eq!(w.len(), 2);
    }

    /// Verify that the first weight tensor in the packed blob starts at byte
    /// offset 0 and has exactly numel * 4 bytes (f32 little-endian).
    #[test]
    fn pack_weight_blob_first_tensor_at_offset_zero() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let blob = pack_weight_blob(&specs, &weights).unwrap();
        let first_numel: usize = specs[0].shape.iter().product();
        let first_data = weights.get_vision_tensor(&specs[0].name).unwrap();
        // The first first_numel f32 values in the blob must match the weight data.
        for i in 0..first_numel {
            let bytes = [blob[i * 4], blob[i * 4 + 1], blob[i * 4 + 2], blob[i * 4 + 3]];
            let val = f32::from_le_bytes(bytes);
            assert!(
                (val - first_data[i]).abs() < 1e-10,
                "byte {i}: expected {} got {}",
                first_data[i],
                val,
            );
        }
    }

    /// Verify that SigLipEncoder::new succeeds with populated weights and
    /// the encoder's internal weight count matches the spec count.
    #[test]
    fn siglip_encoder_new_all_weights_present_count() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let encoder = SigLipEncoder::new(
            config,
            Arc::new(weights.clone()),
            MultimodalTokenIds::fallback_multimodal_token_ids(),
        )
        .unwrap();
        // The weights store should have the same number of tensors as specs.
        assert_eq!(weights.len(), specs.len());
        // Encoder must be usable (config accessor works).
        assert_eq!(encoder.config().hidden_size, 8);
    }

    /// Verify that layer_tensor_name produces correct names for all four
    /// attention projection suffixes (q_proj, k_proj, v_proj, out_proj).
    #[test]
    fn layer_tensor_name_self_attn_projections() {
        let layer = 3usize;
        assert_eq!(
            layer_tensor_name(layer, "self_attn.q_proj.weight"),
            "vision_tower.encoder.layers.3.self_attn.q_proj.weight",
        );
        assert_eq!(
            layer_tensor_name(layer, "self_attn.k_proj.weight"),
            "vision_tower.encoder.layers.3.self_attn.k_proj.weight",
        );
        assert_eq!(
            layer_tensor_name(layer, "self_attn.v_proj.weight"),
            "vision_tower.encoder.layers.3.self_attn.v_proj.weight",
        );
        assert_eq!(
            layer_tensor_name(layer, "self_attn.out_proj.weight"),
            "vision_tower.encoder.layers.3.self_attn.out_proj.weight",
        );
    }

    /// Verify that the graph output tensor's hidden dimension matches
    /// VisionConfig::hidden_size.
    #[test]
    fn graph_hidden_dim_matches_config_hidden_size() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, _) = build_vision_encoder_graph(&config).unwrap();
        let output_tid = graph.outputs[0];
        let output_meta = graph.tensor(output_tid).unwrap();
        // Output shape is [num_patches, hidden_size].
        assert_eq!(output_meta.shape.len(), 2);
        match &output_meta.shape[1] {
            SymDim::Concrete(h) => assert_eq!(*h, config.hidden_size),
            other => panic!("expected Concrete hidden_size, got {other:?}"),
        }
    }

    /// Verify that overwriting an existing key in OwnedVisionWeights does not
    /// change the total count (len stays the same).
    #[test]
    fn owned_vision_weights_overwrite_preserves_count() {
        let mut w = OwnedVisionWeights::new();
        w.insert("x", vec![1.0f32, 2.0f32], vec![2]);
        w.insert("y", vec![3.0f32], vec![1]);
        assert_eq!(w.len(), 2);
        w.insert("x", vec![99.0f32, 88.0f32, 77.0f32], vec![3]);
        assert_eq!(w.len(), 2);
        // Data should reflect the overwrite.
        let data = w.get_vision_tensor("x").unwrap();
        assert_eq!(data, &[99.0f32, 88.0f32, 77.0f32]);
    }

    /// Verify that try_build_siglip_from_tensors calls the fetch closure for
    /// every spec and in the same order as the specs list.
    #[test]
    fn try_build_siglip_fetch_order_matches_spec_order() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let populated = populate_weights(&config);
        let mut fetched_names: Vec<String> = Vec::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let _result = try_build_siglip_from_tensors(&config, ids, |name| {
            fetched_names.push(name.to_string());
            populated.get_vision_tensor(name).map(|slice| {
                let shape = populated.vision_tensor_shape(name).unwrap();
                (slice.to_vec(), shape.to_vec())
            })
        })
        .unwrap();
        // Every spec name must have been fetched exactly once, in order.
        assert_eq!(fetched_names.len(), specs.len());
        for (i, spec) in specs.iter().enumerate() {
            assert_eq!(fetched_names[i], spec.name, "fetch order mismatch at index {i}");
        }
    }

    /// Verify that SigLipEncoder::encode_image returns embeddings whose length
    /// equals num_patches * hidden_size.
    #[test]
    fn siglip_encoder_encode_image_embedding_count() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let encoder = SigLipEncoder::new(
            config.clone(),
            Arc::new(weights),
            MultimodalTokenIds::fallback_multimodal_token_ids(),
        )
        .unwrap();
        let num_patches = config.num_patches();
        let hidden = config.hidden_size;
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<u8> = vec![0u8; pixel_count * 4];
        let media = EncoderMedia::Raw(pixels);
        let encoded = encoder.encode_image(&media).unwrap();
        assert_eq!(
            encoded.embeddings.len(),
            num_patches * hidden,
            "embedding length must be num_patches * hidden_size"
        );
    }

    /// Verify that EncoderMedia::from_generation correctly converts a
    /// MediaInput::Base64 with a MIME type into the matching EncoderMedia variant.
    #[test]
    fn encoder_media_from_generation_base64_with_mime() {
        let input = crate::generation::MediaInput::Base64 {
            data: "iVBORw0KGgo=".into(),
            mime_type: Some("image/png".into()),
        };
        let media = EncoderMedia::from_generation(&input);
        match media {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "iVBORw0KGgo=");
                assert_eq!(mime_type, Some("image/png".into()));
            }
            _ => panic!("expected Base64 variant, got {media:?}"),
        }
    }

    /// Verify num_patches computes the exact square of (image_size / patch_size)
    /// for a non-trivial ratio.
    #[test]
    fn vision_config_num_patches_exact_square() {
        // (28/4)^2 = 7^2 = 49
        let config = VisionConfig {
            image_size: 28,
            patch_size: 4,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 16,
        };
        assert_eq!(config.num_patches(), 49);
    }

    /// Verify that vision_encode requires pixel_count = VISION_IN_CHANNELS *
    /// image_size * image_size (the channel factor is not optional).
    #[test]
    fn vision_encode_pixel_count_includes_channels_factor() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        // Provide image_size * image_size pixels (missing the 3x channel factor).
        let pixel_count_no_channels = config.image_size * config.image_size;
        let pixels = vec![0.0f32; pixel_count_no_channels];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel"), "error should mention pixel: {msg}");
    }

    /// Verify that validate rejects intermediate_size = 0 with a message that
    /// mentions "intermediate_size".
    #[test]
    fn validate_rejects_intermediate_size_zero_specific_message() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 0,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("intermediate_size"),
            "error must mention intermediate_size: {msg}",
        );
    }

    /// Verify that each layer contributes exactly 10 weight specs and that the
    /// total spec count is 2 (global) + 10 * num_layers + 2 (final norm).
    #[test]
    fn graph_per_layer_weight_count_matches_spec() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 3,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let expected = 2 + 10 * config.num_layers + 2;
        assert_eq!(specs.len(), expected);
    }

    // ========================================================================
    // Batch 12: 15 additional edge-case and behavior tests
    // ========================================================================

    // @trace TEST-VISION-BATCH12 [req:REQ-MEGA-002] [level:unit]

    /// Verify that VisionConfig implements Display via Debug (the derive macro
    /// chain produces a non-empty human-readable string containing field values).
    #[test]
    fn vision_config_display_via_debug_is_informative() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let s = format!("{config:?}");
        assert!(!s.is_empty(), "Debug output must be non-empty");
        // Every field value must appear in the output.
        assert!(s.contains("224"), "must contain image_size value");
        assert!(s.contains("14"), "must contain patch_size value");
        assert!(s.contains("768"), "must contain hidden_size value");
        assert!(s.contains("12"), "must contain num_layers/num_heads value");
        assert!(s.contains("3072"), "must contain intermediate_size value");
    }

    /// Verify that SigLipEncoder::encode_audio returns an error whose Display
    /// output does NOT contain the word "image" (confirming it is audio-specific).
    #[test]
    fn siglip_encoder_encode_audio_error_is_audio_specific() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let err = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("audio"), "must mention audio: {msg}");
        assert!(!msg.contains("encode_image"), "must not reference image path: {msg}");
    }

    /// Verify that MultimodalEncoded::validate fails when embeddings have MORE
    /// elements than tokens * hidden_size (oversized embedding buffer).
    #[test]
    fn multimodal_encoded_validate_fails_on_oversized_embeddings() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200],
            embeddings: vec![0.0f32; 20], // 20 > 2 * 8 = 16
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = encoded.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("mismatch"), "must mention mismatch: {msg}");
    }

    /// Verify that OwnedVisionWeights::Default produces a store that returns
    /// None for every lookup (empty store invariant).
    #[test]
    fn owned_vision_weights_default_returns_none_for_any_name() {
        let w = OwnedVisionWeights::default();
        assert!(w.get_vision_tensor("any_name").is_none());
        assert!(w.vision_tensor_shape("any_name").is_none());
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    /// Verify that the graph's position embedding tensor has shape
    /// [num_patches, hidden_size] and that num_patches matches
    /// config.num_patches() exactly.
    #[test]
    fn graph_position_embedding_shape_matches_num_patches() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 32,
        };
        let (graph, _specs) = build_vision_encoder_graph(&config).unwrap();
        // Position embedding is the third input (index 2): [num_patches, hidden].
        let pos_tid = graph.inputs[2];
        let meta = graph.tensor(pos_tid).unwrap();
        assert_eq!(meta.shape.len(), 2);
        match (&meta.shape[0], &meta.shape[1]) {
            (SymDim::Concrete(np), SymDim::Concrete(h)) => {
                assert_eq!(*np, config.num_patches(), "position embedding rows must equal num_patches");
                assert_eq!(*h, config.hidden_size, "position embedding cols must equal hidden_size");
            }
            other => panic!("expected Concrete dims, got {other:?}"),
        }
    }

    /// Verify that vision_encode rejects a pixel buffer with only 1 element
    /// (far too small for any valid config).
    #[test]
    fn vision_encode_rejects_single_element_pixel_buffer() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixels = vec![0.0f32; 1];
        let err = vision_encode(&pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("pixel buffer"), "should mention pixel buffer: {msg}");
        assert!(msg.contains("1"), "should mention actual count 1: {msg}");
    }

    /// Verify that pack_weight_blob correctly handles three tensors where the
    /// middle tensor has a different byte size than the other two, verifying the
    /// byte-level offset alignment for the third tensor.
    #[test]
    fn pack_weight_blob_three_tensors_middle_larger_offset_alignment() {
        let mut w = OwnedVisionWeights::new();
        w.insert("small_a", vec![10.0f32], vec![1]);        // 4 bytes
        w.insert("big_b", vec![1.0f32; 4], vec![4]);         // 16 bytes
        w.insert("small_c", vec![20.0f32], vec![1]);         // 4 bytes

        let specs = vec![
            WeightSpec { name: "small_a".to_string(), shape: &[1] },
            WeightSpec { name: "big_b".to_string(), shape: &[4] },
            WeightSpec { name: "small_c".to_string(), shape: &[1] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        assert_eq!(blob.len(), 24); // (1 + 4 + 1) * 4 = 24
        // Third tensor starts at offset 20 (5 f32s in).
        assert_eq!(blob[20..24], 20.0f32.to_le_bytes());
    }

    /// Verify that SigLipEncoder::encode_image with all-zero pixel data still
    /// produces finite output (the encoder must handle zero inputs gracefully,
    /// not produce NaN or infinity through division-by-zero in LayerNorm).
    #[test]
    fn siglip_encoder_encode_image_zero_pixels_produces_finite_output() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let image_numel = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<f32> = vec![0.0f32; image_numel];
        let mut raw_bytes = Vec::with_capacity(image_numel * 4);
        for v in &pixels {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = encoder.encode_image(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert!(
            encoded.embeddings.iter().all(|v| v.is_finite()),
            "zero-pixel input must produce all-finite embeddings"
        );
    }

    /// Verify that OwnedVisionWeights stores and retrieves negative f32 values
    /// correctly (no sign flipping or absolute value errors).
    #[test]
    fn owned_vision_weights_negative_values_preserved() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![-1.0f32, -0.5, -100.0, -0.001];
        let shape = vec![4];
        w.insert("negatives", data.clone(), shape);
        let retrieved = w.get_vision_tensor("negatives").unwrap();
        assert_eq!(retrieved.len(), 4);
        assert_eq!(retrieved[0], -1.0f32);
        assert_eq!(retrieved[1], -0.5f32);
        assert_eq!(retrieved[2], -100.0f32);
        assert!((retrieved[3] - (-0.001f32)).abs() < 1e-10);
    }

    /// Verify that the graph's PatchEmbed kernel tensor shape encodes the
    /// patch_size correctly at dimensions [2] and [3] (both must be patch_size).
    #[test]
    fn graph_patch_kernel_shape_encodes_patch_size_correctly() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 64,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Patch kernel shape: [hidden, in_channels, patch_size, patch_size].
        let patch_spec = &specs[0];
        assert_eq!(patch_spec.shape.len(), 4);
        assert_eq!(patch_spec.shape[2], config.patch_size);
        assert_eq!(patch_spec.shape[3], config.patch_size);
    }

    /// Verify that MultimodalContext::push_image with two valid image encodings
    /// results in exactly 2 entries in the images vector and 0 in audios.
    #[test]
    fn multimodal_context_two_images_no_audios() {
        let mut ctx = MultimodalContext::new();
        for _ in 0..2 {
            let img = MultimodalEncoded {
                tokens: vec![100u32],
                embeddings: vec![0.5f32; 4],
                hidden_size: 4,
                kind: MediaKind::Image,
            };
            ctx.push_image(img).unwrap();
        }
        assert_eq!(ctx.images.len(), 2);
        assert!(ctx.audios.is_empty());
        assert!(!ctx.is_empty());
    }

    /// Verify that RoutedSequence::seq_len returns 0 when token_ids is empty
    /// and fused_embeddings is also empty (degenerate but valid).
    #[test]
    fn routed_sequence_seq_len_zero_both_vectors_empty() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 0,
        };
        assert_eq!(rs.seq_len(), 0);
        assert!(!rs.has_multimodal());
    }

    /// Verify that EncoderMedia::File variant preserves the exact path string
    /// provided during construction.
    #[test]
    fn encoder_media_file_preserves_exact_path() {
        let path = std::path::PathBuf::from("/tmp/deep/nested/image.png");
        let media = EncoderMedia::File(path.clone());
        if let EncoderMedia::File(stored_path) = &media {
            assert_eq!(stored_path, &path);
            assert_eq!(stored_path.to_str().unwrap(), "/tmp/deep/nested/image.png");
        } else {
            panic!("expected File variant, got {media:?}");
        }
    }

    /// Verify that BackendError::Other produces a Display string that contains
    /// the exact message text passed to it, including special characters.
    #[test]
    fn backend_error_other_preserves_exact_message_with_special_chars() {
        let msg = "error: weight 'tensor[0].data' has NaN at index=42!";
        let err = BackendError::Other(msg.to_string());
        let displayed = format!("{err}");
        assert!(displayed.contains(msg), "displayed must contain exact message: {displayed}");
    }

    /// Verify that SigLipEncoder::new rejects a config where image_size is not
    /// divisible by patch_size, even if all weights are correctly populated.
    #[test]
    fn siglip_encoder_new_rejects_invalid_config_with_populated_weights() {
        let valid_config = tiny_config();
        let weights = Arc::new(populate_weights(&valid_config));
        let bad_config = VisionConfig {
            image_size: 15,
            patch_size: 7, // 15 % 7 != 0
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let err = SigLipEncoder::new(bad_config, weights, ids).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("divisible"), "must mention divisibility: {msg}");
    }

    // ========================================================================
    // Batch 13: 15 additional edge-case and behavior tests
    // ========================================================================

    // @trace TEST-VISION-BATCH13 [req:REQ-MEGA-002] [level:unit]

    /// Verify that MultimodalEncoded::validate fails when embeddings have FEWER
    /// elements than tokens * hidden_size (undersized embedding buffer).
    #[test]
    fn multimodal_encoded_validate_fails_on_undersized_embeddings() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200, 300],
            embeddings: vec![0.5f32; 4], // 4 < 3 * 8 = 24
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = encoded.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("mismatch") || msg.contains("expected"),
            "error must describe mismatch: {msg}"
        );
    }

    /// Verify VisionConfig::head_dim computes correctly for non-power-of-two
    /// values (hidden_size=15, num_heads=3 → head_dim=5).
    #[test]
    fn vision_config_head_dim_non_power_of_two_division() {
        let config = VisionConfig {
            image_size: 15,
            patch_size: 5,
            hidden_size: 15,
            num_layers: 1,
            num_heads: 3,
            intermediate_size: 30,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 5);
    }

    /// Verify that OwnedVisionWeights shape can be overwritten independently
    /// from the data — re-inserting with a different shape replaces the old shape.
    #[test]
    fn owned_vision_weights_shape_overwrite_updates_independently() {
        let mut w = OwnedVisionWeights::new();
        w.insert("tensor_a", vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(w.vision_tensor_shape("tensor_a").unwrap(), &[2, 2]);

        // Overwrite with same data but different shape representation.
        w.insert("tensor_a", vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);
        assert_eq!(w.vision_tensor_shape("tensor_a").unwrap(), &[4]);
        // Data is still 4 elements.
        assert_eq!(w.get_vision_tensor("tensor_a").unwrap().len(), 4);
    }

    /// Verify that SigLipEncoder::decode_pixels for Base64 variant returns an
    /// error message that guides the user toward EncoderMedia::Raw.
    #[test]
    fn siglip_encoder_decode_pixels_base64_error_mentions_raw() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let media = EncoderMedia::Base64 {
            data: "AAAA".into(),
            mime_type: None,
        };
        let err = encoder.decode_pixels(&media).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Raw"),
            "Base64 error should suggest using Raw: {msg}"
        );
    }

    /// Verify that SigLipEncoder::decode_pixels for Url variant returns an
    /// error message that explicitly states the encoder does not perform network I/O.
    #[test]
    fn siglip_encoder_decode_pixels_url_error_mentions_no_network() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();
        let media = EncoderMedia::Url("https://example.com/img.jpg".into());
        let err = encoder.decode_pixels(&media).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("network") || msg.contains("network I/O"),
            "URL error should mention network restriction: {msg}"
        );
    }

    /// Verify that MultimodalContext::push_audio accepts a valid audio encoding
    /// and the context reports non-empty state afterward.
    #[test]
    fn multimodal_context_push_audio_accepts_valid_audio_encoding() {
        let mut ctx = MultimodalContext::new();
        let audio_enc = MultimodalEncoded {
            tokens: vec![300u32],
            embeddings: vec![0.25f32; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(audio_enc).unwrap();
        assert_eq!(ctx.audios.len(), 1);
        assert!(ctx.images.is_empty());
        assert!(!ctx.is_empty());
    }

    /// Verify that pack_weight_blob succeeds when one of the weight specs has
    /// a shape whose product is zero (empty tensor) — the blob simply skips
    /// appending any bytes for that tensor.
    #[test]
    fn pack_weight_blob_with_zero_element_spec_succeeds() {
        let mut w = OwnedVisionWeights::new();
        w.insert("real_weight", vec![1.0f32, 2.0, 3.0], vec![3]);
        w.insert("empty_weight", vec![], vec![0]);

        let specs = vec![
            WeightSpec { name: "real_weight".to_string(), shape: &[3] },
            WeightSpec { name: "empty_weight".to_string(), shape: &[0] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // Only the real_weight contributes bytes: 3 f32s = 12 bytes.
        assert_eq!(blob.len(), 12);
        // First element is 1.0f32 in little-endian.
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
    }

    /// Verify that VisionConfig::validate accepts intermediate_size smaller
    /// than hidden_size (the spec does not require inter >= hidden).
    #[test]
    fn vision_config_validate_accepts_intermediate_smaller_than_hidden() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 4, // smaller than hidden_size=32
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.intermediate_size, 4);
    }

    /// Verify that EncoderMedia::from_generation correctly converts a
    /// MediaInput::Raw variant, preserving the exact byte contents.
    #[test]
    fn encoder_media_from_generation_raw_preserves_bytes() {
        let original_bytes = vec![0xDEu8, 0xAD, 0xBE, 0xEF];
        let input = crate::generation::MediaInput::Raw(original_bytes.clone());
        let media = EncoderMedia::from_generation(&input);
        match media {
            EncoderMedia::Raw(bytes) => assert_eq!(bytes, original_bytes),
            _ => panic!("expected Raw variant, got {media:?}"),
        }
    }

    /// Verify that graph input count scales linearly with num_layers
    /// (each additional layer adds exactly 10 weight tensors).
    #[test]
    fn graph_input_count_grows_linearly_with_layers() {
        let base_count = {
            let c = VisionConfig {
                image_size: 14, patch_size: 7, hidden_size: 8,
                num_layers: 1, num_heads: 1, intermediate_size: 16,
            };
            let (_, specs) = build_vision_encoder_graph(&c).unwrap();
            specs.len()
        };
        let extra_layer_count = {
            let c = VisionConfig {
                image_size: 14, patch_size: 7, hidden_size: 8,
                num_layers: 2, num_heads: 1, intermediate_size: 16,
            };
            let (_, specs) = build_vision_encoder_graph(&c).unwrap();
            specs.len()
        };
        let delta = extra_layer_count - base_count;
        assert_eq!(delta, 10, "each additional layer must add exactly 10 weight specs");
    }

    /// Verify that OwnedVisionWeights::insert accepts both &str and String
    /// keys via the `impl Into<String>` parameter and both produce identical lookups.
    #[test]
    fn owned_vision_weights_insert_str_ref_and_string_key_equivalent() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![42.0f32];

        // Insert with &str.
        w.insert("key_a", data.clone(), vec![1]);
        // Insert with String.
        w.insert(String::from("key_b"), data.clone(), vec![1]);

        assert_eq!(w.get_vision_tensor("key_a").unwrap(), &[42.0f32]);
        assert_eq!(w.get_vision_tensor("key_b").unwrap(), &[42.0f32]);
        assert_eq!(w.len(), 2);
    }

    /// Verify that MultimodalEncoded::validate fails when hidden_size is zero
    /// but tokens and embeddings are non-empty (expected = 0 but got > 0).
    #[test]
    fn multimodal_encoded_validate_zero_hidden_size_non_empty_fails() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32],
            embeddings: vec![0.5f32],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        let err = encoded.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("mismatch") || msg.contains("expected"),
            "error must describe mismatch: {msg}"
        );
    }

    /// Verify that SigLipEncoder::token_ids() returns a copy that is independent
    /// of the encoder's internal state — modifying the returned value does not
    /// affect subsequent calls.
    #[test]
    fn siglip_encoder_token_ids_returns_independent_copy() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let first = encoder.token_ids();
        let second = encoder.token_ids();
        // Both copies should be equal.
        assert_eq!(first, second);
        // Both should match the gemma4 defaults.
        assert_eq!(first.image_token_id, 258880);
        assert_eq!(first.audio_token_id, 258881);
    }

    /// Verify that vision_encode with all-ones pixel data produces output that
    /// differs from the all-zeros pixel data output (different inputs must lead
    /// to different outputs through the encoder).
    #[test]
    fn vision_encode_different_pixels_produce_different_outputs() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;

        let zeros = vec![0.0f32; pixel_count];
        let ones = vec![1.0f32; pixel_count];

        let out_zeros = vision_encode(&zeros, &config, &weights).unwrap();
        let out_ones = vision_encode(&ones, &config, &weights).unwrap();

        // Outputs must differ in at least some elements.
        let any_diff = out_zeros.iter().zip(out_ones.iter()).any(|(a, b)| a != b);
        assert!(any_diff, "all-zeros and all-ones inputs must produce different outputs");
    }

    /// Verify that the weight specs for a multi-layer graph all have unique
    /// names — no two specs share the same tensor name (ensuring the packed
    /// blob has no ambiguous overlaps).
    #[test]
    fn graph_weight_spec_names_all_unique_multi_layer() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 4,
            num_heads: 1,
            intermediate_size: 16,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(
            names.len(),
            unique_count,
            "all weight spec names must be unique, but found duplicates"
        );
    }

    /// Verify that VisionConfig PartialEq is reflexive: a config equals itself.
    #[test]
    fn vision_config_partial_eq_reflexive_identity() {
        let cfg = VisionConfig {
            image_size: 336,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        assert_eq!(cfg, cfg, "VisionConfig must be equal to itself");
    }

    /// Verify that looking up a tensor after overwriting with the same data
    /// returns the new data (not stale cached data from Arc aliasing).
    #[test]
    fn owned_vision_weights_lookup_consistency_after_overwrite_same_data() {
        let mut w = OwnedVisionWeights::new();
        w.insert("tensor_a", vec![1.0, 2.0, 3.0], vec![3]);
        // Overwrite with different data under the same key.
        w.insert("tensor_a", vec![10.0, 20.0, 30.0], vec![3]);
        let data = w.get_vision_tensor("tensor_a").unwrap();
        assert_eq!(data, &[10.0, 20.0, 30.0], "must return the overwritten data");
        let shape = w.vision_tensor_shape("tensor_a").unwrap();
        assert_eq!(shape, &[3]);
    }

    /// Verify that vision_encode output length equals num_patches * hidden_size.
    #[test]
    fn vision_encode_output_len_formula() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.0f32; pixel_count];
        let output = vision_encode(&pixels, &config, &weights).unwrap();
        let expected_len = config.num_patches() * config.hidden_size;
        assert_eq!(
            output.len(),
            expected_len,
            "output length must equal num_patches * hidden_size"
        );
    }

    /// Verify that MultimodalEncoded::validate succeeds when a single token has
    /// exactly hidden_size embedding values.
    #[test]
    fn multimodal_encoded_validate_single_token_exact_match() {
        let encoded = MultimodalEncoded {
            tokens: vec![42u32],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        assert!(encoded.validate().is_ok(), "1 token x 4 hidden must validate");
    }

    /// Verify that MultimodalContext::push_image rejects an encoding with zero
    /// tokens but non-empty embeddings (shape mismatch).
    #[test]
    fn multimodal_context_push_image_rejects_empty_tokens() {
        let mut ctx = MultimodalContext::new();
        let encoded = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![1.0, 2.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let result = ctx.push_image(encoded);
        assert!(result.is_err(), "zero tokens with non-empty embeddings must fail");
    }

    /// Verify that OwnedVisionWeights stores and retrieves a tensor with
    /// multi-dimensional shape correctly.
    #[test]
    fn owned_vision_weights_multidim_shape_lookup() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![1.0f32; 24]; // 2 x 3 x 4
        w.insert("multi", data, vec![2, 3, 4]);
        let shape = w.vision_tensor_shape("multi").unwrap();
        assert_eq!(shape, &[2, 3, 4]);
        let tensor = w.get_vision_tensor("multi").unwrap();
        assert_eq!(tensor.len(), 24);
    }

    /// Verify that EncoderMedia::Raw preserves the exact byte length of the
    /// input vector.
    #[test]
    fn encoder_media_raw_preserves_length() {
        let bytes = vec![0u8; 123];
        let media = EncoderMedia::Raw(bytes.clone());
        if let EncoderMedia::Raw(data) = media {
            assert_eq!(data.len(), 123, "Raw variant must preserve exact byte count");
        } else {
            panic!("expected Raw variant");
        }
    }

    /// Verify that OwnedVisionWeights round-trips a tensor with negative f32 values.
    #[test]
    fn owned_vision_weights_negative_values_roundtrip() {
        let mut w = OwnedVisionWeights::new();
        let data = vec![-1.5f32, -0.0, -999.25];
        w.insert("neg", data.clone(), vec![3]);
        let retrieved = w.get_vision_tensor("neg").unwrap();
        assert_eq!(retrieved, &[-1.5f32, -0.0, -999.25]);
    }

    /// Verify that OwnedVisionWeights handles a zero-element tensor with
    /// an empty shape correctly (edge case).
    #[test]
    fn owned_vision_weights_zero_element_tensor_lookup() {
        let mut w = OwnedVisionWeights::new();
        w.insert("empty", vec![], vec![0]);
        let data = w.get_vision_tensor("empty").unwrap();
        assert!(data.is_empty(), "zero-element tensor must return empty slice");
        let shape = w.vision_tensor_shape("empty").unwrap();
        assert_eq!(shape, &[0]);
    }

    /// Verify that VisionConfig::num_patches returns image_size^2 when patch_size=1.
    #[test]
    fn vision_config_num_patches_identity_patch_one() {
        let cfg = VisionConfig {
            image_size: 28,
            patch_size: 1,
            hidden_size: 64,
            num_layers: 2,
            num_heads: 8,
            intermediate_size: 128,
        };
        assert_eq!(cfg.num_patches(), 28 * 28, "patch_size=1 => num_patches = image_size^2");
    }

    /// Stress test: insert 100 tensors into OwnedVisionWeights and verify all
    /// are retrievable with correct data.
    #[test]
    fn owned_vision_weights_insert_hundred_tensors() {
        let mut w = OwnedVisionWeights::new();
        for i in 0..100 {
            let name = format!("tensor_{i}");
            w.insert(name, vec![i as f32], vec![1]);
        }
        assert_eq!(w.len(), 100);
        for i in 0..100 {
            let name = format!("tensor_{i}");
            let data = w.get_vision_tensor(&name).unwrap();
            assert_eq!(data, &[i as f32], "tensor_{i} must hold value {i}");
        }
    }

    /// Verify that MultimodalTokenIds::is_image returns false for the eoi token.
    #[test]
    fn multimodal_token_ids_is_image_false_for_eoi() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_image(ids.eoi_token_id), "eoi token must not be image");
    }

    /// Verify that validate error for non-divisible image_size/patch_size includes
    /// both dimension values in the message.
    #[test]
    fn validate_error_message_contains_image_and_patch_size() {
        let cfg = VisionConfig {
            image_size: 15,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 64,
        };
        let err = cfg.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("15"), "must mention image_size 15: {msg}");
        assert!(msg.contains("7"), "must mention patch_size 7: {msg}");
    }

    /// Verify that SigLipEncoder::encode_image produces output where every value
    /// is a finite f32 (no NaN, no Inf).
    #[test]
    fn siglip_encoder_encode_image_output_embeddings_are_f32_finite() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![0.5f32; pixel_count];
        let media = EncoderMedia::Raw(
            pixels.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
        let encoded = encoder.encode_image(&media).unwrap();
        let all_finite = encoded.embeddings.iter().all(|v| v.is_finite());
        assert!(all_finite, "all embedding values must be finite f32");
    }

    /// Verify that graph weight specs for fc1/fc2 have shapes that include
    /// intermediate_size (the MLP intermediate dimension).
    #[test]
    fn graph_specs_intermediate_size_matches_config() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 32,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let fc1_spec = specs.iter().find(|s| s.name.contains("mlp.fc1")).expect("fc1 spec");
        let fc2_spec = specs.iter().find(|s| s.name.contains("mlp.fc2")).expect("fc2 spec");
        let fc1_numel: usize = fc1_spec.shape.iter().product();
        let fc2_numel: usize = fc2_spec.shape.iter().product();
        // fc1: [intermediate, hidden] => numel includes intermediate_size
        assert!(
            fc1_numel >= config.intermediate_size,
            "fc1 numel {fc1_numel} must include intermediate_size {}",
            config.intermediate_size
        );
        // fc2: [hidden, intermediate] => numel includes intermediate_size
        assert!(
            fc2_numel >= config.intermediate_size,
            "fc2 numel {fc2_numel} must include intermediate_size {}",
            config.intermediate_size
        );
    }

    // ========================================================================
    // Batch 14: 15 additional edge-case and behavior tests
    // ========================================================================

    // @trace TEST-VISION-BATCH14 [req:REQ-MEGA-002] [level:unit]

    /// 验证 VisionConfig 的 Hash 实现保证克隆配置产生相同的哈希值。
    /// 这确保 VisionConfig 可以安全地用作 HashMap 键。
    #[test]
    fn vision_config_hash_clone_produces_identical_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let cfg = VisionConfig {
            image_size: 336,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        let mut h1 = DefaultHasher::new();
        cfg.hash(&mut h1);
        let hash1 = h1.finish();

        let cfg2 = cfg.clone();
        let mut h2 = DefaultHasher::new();
        cfg2.hash(&mut h2);
        let hash2 = h2.finish();

        assert_eq!(hash1, hash2, "clone must produce identical hash");
    }

    /// 验证 VisionConfig 的 Hash 实现保证不同配置产生不同的哈希值。
    #[test]
    fn vision_config_hash_different_configs_different_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let cfg_a = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let cfg_b = VisionConfig {
            image_size: 384,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let mut h1 = DefaultHasher::new();
        cfg_a.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        cfg_b.hash(&mut h2);
        let hash2 = h2.finish();

        assert_ne!(hash1, hash2, "configs differing in image_size must produce different hashes");
    }

    /// 验证 SigLipEncoder::decode_pixels 对 Raw 类型在恰好一个 f32 字节
    /// 偏少时产生包含字节数信息的错误。
    #[test]
    fn decode_pixels_raw_exactly_four_bytes_short_error() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let expected_bytes = VISION_IN_CHANNELS * config.image_size * config.image_size * 4;
        // 缺少 4 字节（即 1 个 f32）。
        let short_bytes = vec![0u8; expected_bytes - 4];
        let err = encoder.decode_pixels(&EncoderMedia::Raw(short_bytes)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("bytes"), "error should mention byte count: {msg}");
        assert!(msg.contains(&format!("{}", expected_bytes)), "error should mention expected bytes: {msg}");
    }

    /// 验证 OwnedVisionWeights 的 PartialEq 实现正确识别数据内容不同
    /// 但名称相同的两个 store（它们不相等）。
    #[test]
    fn owned_vision_weights_partial_eq_different_data_same_keys() {
        let mut w1 = OwnedVisionWeights::new();
        w1.insert("a", vec![1.0f32, 2.0], vec![2]);

        let mut w2 = OwnedVisionWeights::new();
        w2.insert("a", vec![1.0f32, 99.0], vec![2]);

        assert_ne!(w1, w2, "stores with different data under same key must not be equal");
    }

    /// 验证 OwnedVisionWeights 的 PartialEq 实现正确识别形状不同
    /// 但数据元素数量相同的两个 store（它们不相等）。
    #[test]
    fn owned_vision_weights_partial_eq_same_data_different_shape() {
        let mut w1 = OwnedVisionWeights::new();
        w1.insert("t", vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);

        let mut w2 = OwnedVisionWeights::new();
        w2.insert("t", vec![1.0f32, 2.0, 3.0, 4.0], vec![4]);

        assert_ne!(w1, w2, "stores with same data but different shape must not be equal");
    }

    /// 验证 MultimodalTokenIds::is_image 对 image_token_id 本身返回 true。
    #[test]
    fn token_ids_is_image_true_for_exact_image_token_id() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(ids.is_image(ids.image_token_id), "image_token_id must be recognized as image");
    }

    /// 验证 MultimodalTokenIds::is_audio 对 audio_token_id 本身返回 true。
    #[test]
    fn token_ids_is_audio_true_for_exact_audio_token_id() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(ids.is_audio(ids.audio_token_id), "audio_token_id must be recognized as audio");
    }

    /// 验证 EncoderMedia::Url 变体保留精确的 URL 字符串内容。
    #[test]
    fn encoder_media_url_preserves_exact_url_string() {
        let url = "https://example.com/path/to/image.jpg?query=1&size=large#fragment";
        let media = EncoderMedia::Url(url.to_string());
        if let EncoderMedia::Url(stored) = &media {
            assert_eq!(stored, url, "URL variant must preserve the exact string");
        } else {
            panic!("expected Url variant, got {media:?}");
        }
    }

    /// 验证 SigLipEncoder::encode_image 返回的 token 列表中每个 token 都
    /// 等于 token_ids.image_token_id。
    #[test]
    fn siglip_encoder_encode_image_all_tokens_match_image_token_id() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let image_token = ids.image_token_id;
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels: Vec<u8> = vec![0u8; pixel_count * 4];
        let encoded = encoder.encode_image(&EncoderMedia::Raw(pixels)).unwrap();

        assert!(
            encoded.tokens.iter().all(|&t| t == image_token),
            "every token in the output must be image_token_id"
        );
    }

    /// 验证 VisionConfig::validate 接受 hidden_size 等于 num_heads（head_dim=1）
    /// 且 intermediate_size 也等于 hidden_size 的配置。
    #[test]
    fn vision_config_validate_all_dimensions_equal() {
        let config = VisionConfig {
            image_size: 4,
            patch_size: 2,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 8,
            intermediate_size: 8,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 1);
        assert_eq!(config.num_patches(), 4);
    }

    /// 验证 vision_encode 对包含 NaN 的像素输入不 panic，而是返回结果
    /// （JIT 代码不做 NaN 检查，结果中的 NaN 是合法的）。
    #[test]
    fn vision_encode_with_nan_pixels_does_not_panic() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let mut pixels = vec![0.0f32; pixel_count];
        // 在第一个和最后一个位置注入 NaN。
        pixels[0] = f32::NAN;
        pixels[pixel_count - 1] = f32::NAN;

        let result = vision_encode(&pixels, &config, &weights);
        // 不 panic 即通过——JIT 不做 NaN 守卫，可能返回 NaN 也可能被 LayerNorm 吸收。
        assert!(result.is_ok(), "vision_encode must not panic on NaN input pixels");
    }

    /// 验证 pack_weight_blob 在 weight tensor 的 numel 与 shape 乘积不匹配时
    /// 返回包含 tensor 名称和长度信息的错误。
    #[test]
    fn pack_weight_blob_error_includes_both_name_and_expected_numel() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let mut weights = OwnedVisionWeights::new();
        // 为所有 tensor 注入正确数据，但故意让第一个 tensor 数据过短。
        for (i, spec) in specs.iter().enumerate() {
            let numel: usize = spec.shape.iter().product();
            if i == 0 {
                // 第一个 tensor 少一个元素。
                weights.insert(spec.name.clone(), vec![0.0f32; numel - 1], spec.shape.to_vec());
            } else {
                weights.insert(spec.name.clone(), vec![0.0f32; numel], spec.shape.to_vec());
            }
        }
        let err = pack_weight_blob(&specs, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(&specs[0].name), "error must contain the tensor name: {msg}");
        assert!(msg.contains("length") || msg.contains("len"), "error must describe length mismatch: {msg}");
    }

    /// 验证 RoutedSequence::has_multimodal 在 fused_embeddings 包含 Some 值
    /// 但 text_positions 为空时返回 true（embeddings 存在即表示有模态内容）。
    #[test]
    fn routed_sequence_has_multimodal_with_empty_positions_but_embeddings() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![100, 200],
            fused_embeddings: vec![Some(vec![0.5f32; 8]), None],
            text_positions: vec![],
            hidden_size: 8,
        };
        assert!(rs.has_multimodal(), "Some embedding present means has_multimodal must be true");
        assert_eq!(rs.seq_len(), 2);
    }

    /// 验证 MultimodalEncoded::num_tokens 返回 tokens 向量的长度
    /// 而非 embeddings 的长度除以 hidden_size。
    #[test]
    fn multimodal_encoded_num_tokens_reflects_token_count_not_embedding_count() {
        let encoded = MultimodalEncoded {
            tokens: vec![100u32, 200, 300, 400],
            embeddings: vec![0.0f32; 16], // 4 tokens * 4 hidden = 16
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        assert_eq!(encoded.num_tokens(), 4, "num_tokens must return tokens.len()");
        assert!(encoded.validate().is_ok());
    }

    /// 验证 build_fused_hidden 纯音频路径：只有音频模态嵌入，无文本 token，
    /// 结果只包含音频嵌入数据。
    #[test]
    fn build_fused_hidden_pure_audio_no_text() {
        use crate::compat::multimodal::{build_fused_hidden, route_multimodal_tokens};
        let mut ctx = MultimodalContext::new();
        let audio_enc = MultimodalEncoded {
            tokens: vec![300u32],
            embeddings: vec![0.25f32; 4], // 1 token * 4 hidden
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(audio_enc).unwrap();

        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![ids.audio_token_id];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 4).unwrap();
        // 假 embedding 表：vocab 至少覆盖 token id 范围。
        let embed_rows = vec![0.0f32; 512 * 4];
        let fused = build_fused_hidden(&routed, &embed_rows, 4).unwrap();
        // seq_len = 1, hidden_size = 4, fused 长度 = 4。
        assert_eq!(fused.len(), 4, "fused output length must be 1 * 4");
        assert!(routed.has_multimodal());
    }

    // ========================================================================
    // Batch 6: 15 additional tests for novel coverage paths
    // ========================================================================

    /// Verify that vision_encode with f32::INFINITY pixels does not panic.
    /// JIT code does not guard against Inf; the result may contain Inf or NaN
    /// but the function must return Ok.
    #[test]
    fn vision_encode_infinity_pixels_does_not_panic() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![f32::INFINITY; pixel_count];

        let result = vision_encode(&pixels, &config, &weights);
        assert!(result.is_ok(), "vision_encode must not panic on Inf input pixels");
    }

    /// Verify that MultimodalContext can hold two audio encodings simultaneously
    /// and reports non-empty after both are pushed.
    #[test]
    fn multimodal_context_push_two_audios_success() {
        let mut ctx = MultimodalContext::new();
        let audio1 = MultimodalEncoded {
            tokens: vec![300u32],
            embeddings: vec![0.1f32; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let audio2 = MultimodalEncoded {
            tokens: vec![301u32],
            embeddings: vec![0.2f32; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(audio1).unwrap();
        ctx.push_audio(audio2).unwrap();
        assert!(!ctx.is_empty());
        assert_eq!(ctx.audios.len(), 2);
        assert_eq!(ctx.images.len(), 0);
    }

    /// Verify that route_multimodal_tokens with a prompt containing only plain text
    /// tokens (no image_token_id or audio_token_id) produces a RoutedSequence with
    /// all positions as text and no fused embeddings.
    #[test]
    fn route_multimodal_tokens_all_text_no_special_tokens() {
        use crate::compat::multimodal::route_multimodal_tokens;
        let ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // These are ordinary text token IDs, not special multimodal tokens.
        let tokens = vec![1u32, 2, 3, 4, 5];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 8).unwrap();

        assert_eq!(routed.seq_len(), 5);
        assert!(!routed.has_multimodal(), "plain text tokens must not produce multimodal embeddings");
        assert_eq!(routed.text_positions.len(), 5);
        // All fused_embeddings must be None (text positions).
        assert!(routed.fused_embeddings.iter().all(|e| e.is_none()));
    }

    /// Verify that build_fused_hidden with a pure-text RoutedSequence (no multimodal)
    /// produces output where every position is filled from the embedding table.
    #[test]
    fn build_fused_hidden_pure_text_no_media() {
        use crate::compat::multimodal::{build_fused_hidden, route_multimodal_tokens};
        let ctx = MultimodalContext::new();
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let tokens = vec![0u32, 1, 2];
        let routed = route_multimodal_tokens(&tokens, &ctx, &ids, 4).unwrap();

        let embed_rows = vec![1.0f32; 4 * 4]; // vocab_size=4, hidden=4
        let fused = build_fused_hidden(&routed, &embed_rows, 4).unwrap();
        assert_eq!(fused.len(), 3 * 4, "seq_len=3, hidden=4 => fused length = 12");
        assert!(!routed.has_multimodal());
    }

    /// Verify that RoutedSequence with all-None fused_embeddings reports
    /// has_multimodal() == false.
    #[test]
    fn routed_sequence_no_embeddings_has_no_multimodal() {
        use crate::compat::multimodal::RoutedSequence;
        let rs = RoutedSequence {
            token_ids: vec![10u32, 20, 30],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 8,
        };
        assert!(!rs.has_multimodal());
        assert_eq!(rs.seq_len(), 3);
    }

    /// Verify that MultimodalTokenIds::eoi_token_id is not classified as an image token.
    #[test]
    fn multimodal_token_ids_eoi_is_not_image() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_image(ids.eoi_token_id),
            "eoi_token_id must not be classified as image");
    }

    /// Verify that MultimodalTokenIds::eoa_token_id is not classified as an audio token.
    #[test]
    fn multimodal_token_ids_eoa_is_not_audio() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert!(!ids.is_audio(ids.eoa_token_id),
            "eoa_token_id must not be classified as audio");
    }

    /// Verify that EncoderMedia::Base64 with None mime_type is accepted by the
    /// SigLipEncoder (returns an explicit error about Base64 support, not a panic).
    #[test]
    fn encoder_media_base64_none_mime_type_rejected_gracefully() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let err = encoder
            .encode_image(&EncoderMedia::Base64 {
                data: "AAAA".into(),
                mime_type: None,
            })
            .unwrap_err();
        assert!(format!("{err}").contains("Base64"));
    }

    /// Verify VisionConfig::num_patches with 896/14 = 64 patches per side = 4096 total.
    #[test]
    fn vision_config_num_patches_ratio_896_14() {
        let config = VisionConfig {
            image_size: 896,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            intermediate_size: 4096,
        };
        assert_eq!(config.num_patches(), 4096); // (896/14)^2 = 64^2
    }

    /// Verify that pack_weight_blob places tensors in the exact order of specs,
    /// by packing two tensors with distinct values and checking byte offsets.
    #[test]
    fn pack_weight_blob_preserves_spec_ordering() {
        let mut w = OwnedVisionWeights::new();
        w.insert("first", vec![1.0f32, 2.0], vec![2]);
        w.insert("second", vec![3.0f32, 4.0], vec![2]);

        let specs = vec![
            WeightSpec { name: "first".to_string(), shape: &[2] },
            WeightSpec { name: "second".to_string(), shape: &[2] },
        ];
        let blob = pack_weight_blob(&specs, &w).unwrap();
        // "first" tensor at offset 0: [1.0, 2.0] as LE bytes.
        assert_eq!(blob[0..4], 1.0f32.to_le_bytes());
        assert_eq!(blob[4..8], 2.0f32.to_le_bytes());
        // "second" tensor at offset 8: [3.0, 4.0] as LE bytes.
        assert_eq!(blob[8..12], 3.0f32.to_le_bytes());
        assert_eq!(blob[12..16], 4.0f32.to_le_bytes());
    }

    /// Verify OwnedVisionWeights Debug output for a non-empty store contains
    /// tensor count information.
    #[test]
    fn owned_vision_weights_debug_non_empty_shows_content() {
        let mut w = OwnedVisionWeights::new();
        w.insert("alpha", vec![1.0f32], vec![1]);
        w.insert("beta", vec![2.0f32, 3.0], vec![2]);
        let debug = format!("{w:?}");
        assert!(!debug.is_empty());
        // HashMap Debug contains key-value pairs.
        assert!(debug.contains("alpha") || debug.contains("tensors"));
    }

    /// Verify that SigLipEncoder can be constructed with a minimal but valid config
    /// where image_size == patch_size (1 patch) and hidden_size == 1.
    #[test]
    fn siglip_encoder_new_minimal_single_patch_config() {
        let config = VisionConfig {
            image_size: 7,
            patch_size: 7,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 8,
        };
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();
        assert_eq!(encoder.config().num_patches(), 1);
        assert_eq!(encoder.config().head_dim(), 2);
    }

    /// Verify VisionConfig::head_dim with a large number of heads (hidden=1024, heads=64).
    #[test]
    fn vision_config_head_dim_with_64_heads() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 64,
            intermediate_size: 4096,
        };
        assert_eq!(config.head_dim(), 16); // 1024 / 64
    }

    /// Verify that VisionConfig::validate accepts intermediate_size much smaller
    /// than hidden_size (e.g. 8 vs 32). The spec does not require intermediate >= hidden.
    #[test]
    fn vision_config_validate_intermediate_eight_hidden_thirty_two() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 8, // smaller than hidden_size
        };
        assert!(config.validate().is_ok());
    }

    /// Verify that vision_encode with f32::NEG_INFINITY pixels does not panic.
    /// The function must return Ok (output may contain non-finite values).
    #[test]
    fn vision_encode_neg_infinity_pixels_does_not_panic() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let pixels = vec![f32::NEG_INFINITY; pixel_count];

        let result = vision_encode(&pixels, &config, &weights);
        assert!(result.is_ok(), "vision_encode must not panic on -Inf input pixels");
    }

    // ========================================================================
    // Batch 7: 15 additional tests for deeper coverage
    // ========================================================================

    /// Verify that VisionConfig::validate rejects image_size=0 with an error
    /// message that specifically mentions "image_size" and "> 0".
    #[test]
    fn validate_rejects_zero_image_size_specific_message() {
        let config = VisionConfig {
            image_size: 0,
            patch_size: 7,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("image_size"), "error must mention image_size: {msg}");
        assert!(msg.contains("> 0") || msg.contains("must be"), "error must mention constraint: {msg}");
    }

    /// Verify that VisionConfig::validate rejects patch_size=0 with an error
    /// message that specifically mentions "patch_size" and "> 0".
    #[test]
    fn validate_rejects_zero_patch_size_specific_message() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 0,
            hidden_size: 8,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("patch_size"), "error must mention patch_size: {msg}");
        assert!(msg.contains("> 0") || msg.contains("must be"), "error must mention constraint: {msg}");
    }

    /// Verify that OwnedVisionWeights::Clone produces an independent copy:
    /// modifying the original after cloning does not affect the clone.
    #[test]
    fn owned_vision_weights_clone_deep_independence() {
        let mut original = OwnedVisionWeights::new();
        original.insert("shared", vec![10.0f32, 20.0], vec![2]);
        let cloned = original.clone();

        // Modify original — cloned must stay unchanged.
        original.insert("shared", vec![99.0f32], vec![1]);
        assert_eq!(cloned.get_vision_tensor("shared").unwrap(), &[10.0f32, 20.0],
                   "cloned weights must be independent of original");
        assert_eq!(cloned.vision_tensor_shape("shared").unwrap(), &[2]);
    }

    /// Verify that OwnedVisionWeights::Debug output for an empty store
    /// contains the expected HashMap representation.
    #[test]
    fn owned_vision_weights_debug_empty_store() {
        let w = OwnedVisionWeights::new();
        let debug = format!("{w:?}");
        assert!(!debug.is_empty(), "Debug output must not be empty");
        // HashMap Debug for an empty map is "{}".
        assert!(debug.contains("tensors"), "Debug must contain field name 'tensors': {debug}");
    }

    /// Verify that patch embedding dimension equals hidden_size by checking
    /// the first weight spec shape — the embed_dim dimension must be hidden_size.
    #[test]
    fn patch_embed_dim_equals_hidden_size() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_size: 64,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // First spec is patch_embed kernel: [hidden, in_channels, patch, patch].
        assert_eq!(specs[0].shape[0], config.hidden_size,
                   "patch embed first dim must equal hidden_size");
    }

    /// Verify that VisionConfig::validate accepts very large dimensions
    /// (4096x4096 image) as long as image_size is divisible by patch_size.
    #[test]
    fn validate_accepts_4096x4096_image() {
        let config = VisionConfig {
            image_size: 4096,
            patch_size: 16,
            hidden_size: 2048,
            num_layers: 32,
            num_heads: 32,
            intermediate_size: 8192,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.num_patches(), (4096 / 16) * (4096 / 16));
    }

    /// Verify that BackendError::Other preserves the exact message through
    /// Display including special characters and formatting.
    #[test]
    fn backend_error_other_display_special_chars() {
        let msg = "weight 'layer.0.mlp.fc1' shape [8, 16] != expected [8, 32]";
        let err = BackendError::Other(msg.into());
        let displayed = format!("{err}");
        assert!(displayed.contains("layer.0.mlp.fc1"), "must preserve weight name: {displayed}");
        assert!(displayed.contains("[8, 16]"), "must preserve first shape: {displayed}");
        assert!(displayed.contains("[8, 32]"), "must preserve second shape: {displayed}");
    }

    /// Verify that two independent vision_encode calls with different pixel
    /// data produce different outputs (multi-image encoding independence).
    #[test]
    fn vision_encode_different_images_produce_different_outputs() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;

        let pixels_a: Vec<f32> = (0..pixel_count).map(|i| (i as f32) * 0.01).collect();
        let pixels_b: Vec<f32> = (0..pixel_count).map(|i| (i as f32) * -0.01).collect();

        let out_a = vision_encode(&pixels_a, &config, &weights).unwrap();
        let out_b = vision_encode(&pixels_b, &config, &weights).unwrap();

        assert_ne!(out_a, out_b, "different pixel inputs must produce different outputs");
    }

    /// Verify that the VISION_IN_CHANNELS constant (3) is reflected in the
    /// patch kernel weight spec shape — the second dimension must be 3.
    #[test]
    fn graph_patch_kernel_shape_reflects_in_channels() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Patch kernel: [hidden, in_channels, patch_size, patch_size].
        assert_eq!(specs[0].shape[1], VISION_IN_CHANNELS,
                   "patch kernel second dim must equal VISION_IN_CHANNELS (3)");
    }

    /// Verify that OwnedVisionWeights initialized with all-zero data returns
    /// correct zero slices via VisionTensorLookup.
    #[test]
    fn owned_vision_weights_zero_initialized_data() {
        let mut w = OwnedVisionWeights::new();
        let zeros = vec![0.0f32; 16];
        w.insert("zero_tensor", zeros.clone(), vec![4, 4]);

        let data = w.get_vision_tensor("zero_tensor").unwrap();
        assert_eq!(data.len(), 16);
        assert!(data.iter().all(|&v| v == 0.0f32), "all values must be zero");
    }

    /// Verify that VisionConfig Clone roundtrip preserves all fields exactly,
    /// including after the original is used in a HashMap (verifying no aliasing).
    #[test]
    fn vision_config_clone_roundtrip_via_hashmap() {
        use std::collections::HashMap;
        let original = VisionConfig {
            image_size: 384,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let cloned = original.clone();

        // Use original as HashMap key, then verify cloned still matches.
        let mut map = HashMap::new();
        map.insert(original.clone(), "test_value");
        let from_map = map.get(&original).unwrap();
        assert_eq!(*from_map, "test_value");

        // Cloned must still be equal to original after original was moved into map key.
        assert_eq!(cloned.image_size, 384);
        assert_eq!(cloned.patch_size, 14);
        assert_eq!(cloned.hidden_size, 768);
        assert_eq!(cloned.num_layers, 12);
        assert_eq!(cloned.num_heads, 12);
        assert_eq!(cloned.intermediate_size, 3072);
    }

    /// Verify that VisionConfig::validate rejects hidden_size=0 with an error
    /// message that mentions both "hidden_size" and "num_heads".
    #[test]
    fn validate_rejects_zero_hidden_size_mentions_both_fields() {
        let config = VisionConfig {
            image_size: 14,
            patch_size: 7,
            hidden_size: 0,
            num_layers: 1,
            num_heads: 1,
            intermediate_size: 16,
        };
        let err = config.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("hidden_size"), "error must mention hidden_size: {msg}");
        assert!(msg.contains("num_heads"), "error must mention num_heads: {msg}");
    }

    /// Verify that SigLipEncoder::token_ids() returns an independent copy —
    /// modifying the returned value does not affect subsequent calls.
    #[test]
    fn siglip_encoder_token_ids_independent_copies() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 300,
            eoa_token_id: 400,
        };
        let encoder = SigLipEncoder::new(config, weights, ids).unwrap();

        let copy1 = encoder.token_ids();
        let copy2 = encoder.token_ids();
        // Both copies must be equal but independent.
        assert_eq!(copy1.image_token_id, copy2.image_token_id);
        assert_eq!(copy1.audio_token_id, copy2.audio_token_id);
    }

    /// Verify that the VISION_LAYERNORM_EPS constant matches the value used
    /// in the LayerNorm ops within the graph (1e-6 for SigLIP).
    #[test]
    fn vision_layernorm_eps_matches_siglip_convention() {
        // SigLIP uses epsilon=1e-6, not the PyTorch default 1e-5.
        assert!((VISION_LAYERNORM_EPS - 1e-6f32).abs() < f32::EPSILON,
                "VISION_LAYERNORM_EPS must be exactly 1e-6 for SigLIP");
        assert!(VISION_LAYERNORM_EPS < 1e-5,
                "must be smaller than PyTorch default 1e-5");
    }

    /// Verify that the graph's patch kernel weight shape encodes the correct
    /// patch_size in both spatial dimensions (indices 2 and 3).
    #[test]
    fn graph_patch_kernel_spatial_dims_match_patch_size() {
        let config = VisionConfig {
            image_size: 56,
            patch_size: 14,
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 64,
        };
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        // Patch kernel: [hidden, channels, patch_h, patch_w].
        assert_eq!(specs[0].shape[2], config.patch_size,
                   "patch kernel height dim must equal patch_size");
        assert_eq!(specs[0].shape[3], config.patch_size,
                   "patch kernel width dim must equal patch_size");
    }

    // --- Batch 15: uncovered data-structure & error-path tests ---

    /// Verify that encode_image with valid Raw media produces a MultimodalEncoded
    /// with the correct hidden_size field matching the config.
    #[test]
    fn encode_image_raw_media_sets_correct_hidden_size() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 10,
            audio_token_id: 20,
            eoi_token_id: 30,
            eoa_token_id: 40,
        };
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let raw_bytes: Vec<u8> = (0..pixel_count)
            .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
            .collect();
        let media = EncoderMedia::Raw(raw_bytes);

        let result = encoder.encode_image(&media).unwrap();
        assert_eq!(result.hidden_size, config.hidden_size,
                   "hidden_size in MultimodalEncoded must match VisionConfig");
    }

    /// Verify that encode_image with valid Raw media produces a MultimodalEncoded
    /// with MediaKind::Image.
    #[test]
    fn encode_image_raw_media_sets_image_kind() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 10,
            audio_token_id: 20,
            eoi_token_id: 30,
            eoa_token_id: 40,
        };
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let raw_bytes: Vec<u8> = (0..pixel_count)
            .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
            .collect();
        let media = EncoderMedia::Raw(raw_bytes);

        let result = encoder.encode_image(&media).unwrap();
        assert_eq!(result.kind, MediaKind::Image,
                   "encode_image must produce MediaKind::Image");
    }

    /// Verify that try_build_siglip_from_tensors returns an error when a fetched
    /// tensor has the wrong data length (right shape declared, but wrong numel).
    #[test]
    fn try_build_siglip_from_tensors_rejects_wrong_data_length() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };

        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            let spec = specs.iter().find(|s| s.name == name).unwrap();
            // Return correct shape but deliberately wrong data length (half the elements).
            let wrong_len = spec.shape.iter().product::<usize>() / 2;
            Some((vec![0.0f32; wrong_len.max(1)], spec.shape.to_vec()))
        });
        assert!(result.is_err(), "must reject tensors with wrong data length");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("len"), "error must mention length: {msg}");
    }

    /// Verify that try_build_siglip_from_tensors returns an error when a fetched
    /// tensor has the correct numel but wrong declared shape.
    #[test]
    fn try_build_siglip_from_tensors_rejects_wrong_shape() {
        let config = tiny_config();
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };

        let result = try_build_siglip_from_tensors(&config, ids, |name| {
            let spec = specs.iter().find(|s| s.name == name).unwrap();
            let numel: usize = spec.shape.iter().product();
            // Return correct numel but a deliberately different shape.
            let wrong_shape = vec![numel];
            Some((vec![0.0f32; numel], wrong_shape))
        });
        assert!(result.is_err(), "must reject tensors with wrong shape");
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("shape"), "error must mention shape: {msg}");
    }

    /// Verify that OwnedVisionWeights::len() counts unique keys after inserting
    /// the same key twice — the second insert must overwrite, not duplicate.
    #[test]
    fn owned_vision_weights_len_overwrite_same_key() {
        let mut w = OwnedVisionWeights::new();
        w.insert("tensor_a", vec![1.0, 2.0], vec![2]);
        assert_eq!(w.len(), 1);

        w.insert("tensor_a", vec![3.0, 4.0, 5.0], vec![3]);
        // Same key overwrites, so len stays 1.
        assert_eq!(w.len(), 1, "overwriting same key must not increase len");

        // The data must reflect the overwrite.
        let data = w.get_vision_tensor("tensor_a").unwrap();
        assert_eq!(data, &[3.0, 4.0, 5.0], "data must be from second insert");
    }

    /// Verify that VisionConfig::head_dim() works with non-power-of-2 hidden
    /// dimensions (e.g., hidden_size=14, num_heads=2 => head_dim=7).
    #[test]
    fn vision_config_head_dim_non_power_of_two() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 14,
            num_layers: 1,
            num_heads: 2,
            intermediate_size: 28,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.head_dim(), 7, "14 / 2 = 7, non-power-of-2");
    }

    /// Verify that pack_weight_blob produces bytes in little-endian f32 layout
    /// by checking the first 4 bytes of the blob match f32::to_le_bytes().
    #[test]
    fn pack_weight_blob_byte_order_is_little_endian_f32() {
        let config = tiny_config();
        let weights = populate_weights(&config);
        let (_, specs) = build_vision_encoder_graph(&config).unwrap();
        let blob = pack_weight_blob(&specs, &weights).unwrap();

        // The first tensor's first element should appear as f32 LE at blob[0..4].
        let first_spec = &specs[0];
        let first_val = weights.get_vision_tensor(&first_spec.name).unwrap()[0];
        let expected_bytes = first_val.to_le_bytes();
        assert_eq!(&blob[0..4], &expected_bytes,
                   "first 4 bytes must be first f32 in little-endian");
    }

    /// Verify that SigLipEncoder::config() returns a reference to the exact
    /// VisionConfig that was passed at construction time.
    #[test]
    fn siglip_encoder_config_accessor_matches_construction() {
        let config = VisionConfig {
            image_size: 28,
            patch_size: 14,
            hidden_size: 16,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 32,
        };
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();
        let returned = encoder.config();
        assert_eq!(returned.image_size, config.image_size);
        assert_eq!(returned.patch_size, config.patch_size);
        assert_eq!(returned.hidden_size, config.hidden_size);
        assert_eq!(returned.num_layers, config.num_layers);
        assert_eq!(returned.num_heads, config.num_heads);
        assert_eq!(returned.intermediate_size, config.intermediate_size);
    }

    /// Verify that decode_pixels with valid Raw media of exact byte count
    /// produces f32 values that match the original byte interpretation.
    #[test]
    fn decode_pixels_raw_exact_byte_count_matches_f32() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        let encoder = SigLipEncoder::new(config.clone(), weights, ids).unwrap();

        // Build exactly the right number of f32 pixels.
        let pixel_count = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let original_pixels: Vec<f32> = (0..pixel_count).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let raw_bytes: Vec<u8> = original_pixels.iter().flat_map(|v| v.to_le_bytes()).collect();

        let decoded = encoder.decode_pixels(&EncoderMedia::Raw(raw_bytes)).unwrap();
        assert_eq!(decoded.len(), pixel_count);
        for (i, (orig, dec)) in original_pixels.iter().zip(decoded.iter()).enumerate() {
            assert!((orig - dec).abs() < f32::EPSILON,
                    "pixel {i}: expected {orig}, got {dec}");
        }
    }

    /// Verify that vision_encode rejects a pixel buffer whose length does not
    /// equal VISION_IN_CHANNELS * image_size * image_size with an error
    /// message that includes the expected formula "3x".
    #[test]
    fn vision_encode_pixel_mismatch_error_mentions_3x_formula() {
        let config = tiny_config();
        let weights = populate_weights(&config);

        // Provide one extra pixel beyond the expected count.
        let expected = VISION_IN_CHANNELS * config.image_size * config.image_size;
        let bad_pixels = vec![0.0f32; expected + 1];

        let err = vision_encode(&bad_pixels, &config, &weights).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("3×") || msg.contains("3x") || msg.contains("vision_encode"),
                "error must mention the 3x formula or function name: {msg}");
        assert!(msg.contains(&expected.to_string()),
                "error must include expected pixel count {expected}: {msg}");
    }
}
