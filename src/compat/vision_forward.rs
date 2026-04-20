//! SigLIP Vision Encoder — real JIT-compiled forward pass.
//!
//! Builds a `CompilerGraph` for the full SigLIP ViT encoder (PatchEmbed →
//! LearnedPos2D → N × ViT blocks → final LayerNorm), JIT-compiles it through
//! the standard `InferenceCompiler::compile_graph` pipeline (scalar → SymExec →
//! IR → ISA lowering → native machine code) and executes it once per image.
//!
//! Every computation path goes through the JIT: no scalar fallback, no
//! hand-written Rust, no external BLAS. The `VisionConfig` drives graph
//! construction; the `VisionTensorLookup` trait supplies weight tensors by
//! name (matching the `src/arch/templates/siglip.yaml` naming convention).
//!
//! SPEC: 02-ARCHITECTURE ARCH-MULTIMODAL + ARCH-MULTIMODAL-FUSION.

use std::sync::Arc;

use gllm_kernels::compiler::{
    CompilerGraph, InferenceCompiler, OpKind, ShapeBinding, SymDim, TensorId,
};
use gllm_kernels::types::DType;

use crate::compat::multimodal::{
    EncoderMedia, MediaKind, MultimodalEncoded, MultimodalEncoder, MultimodalTokenIds,
};
use crate::engine::executor::BackendError;

/// Trait for looking up named vision encoder weight tensors.
///
/// All tensors are expected to be row-major f32. Names follow the
/// HuggingFace `SiglipVisionModel` convention (see
/// `src/arch/templates/siglip.yaml`).
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
#[derive(Debug, Clone)]
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
        if self.image_size % self.patch_size != 0 {
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
        if self.hidden_size % self.num_heads != 0 {
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
// Tensor name conventions — matches src/arch/templates/siglip.yaml
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: q_dim,
                k: hidden,
                dtype: dt,
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: hidden,
                k: q_dim,
                dtype: dt,
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: inter,
                k: hidden,
                dtype: dt,
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
            OpKind::Gemm {
                m: SymDim::Concrete(num_patches),
                n: hidden,
                k: inter,
                dtype: dt,
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

    // JIT compile + execute. InferenceCompiler::compile_graph returns a
    // `CompiledLayer` whose `execute` ABI matches `CompiledLayerFn` (see
    // gllm-kernels/src/compiler/executable.rs).
    let mut compiler = InferenceCompiler::new();
    let compiled = compiler.compile_graph(&graph).map_err(|e| {
        BackendError::Other(format!(
            "vision_encode: InferenceCompiler::compile_graph failed: {e}"
        ))
    })?;

    let num_patches = config.num_patches();
    let hidden = config.hidden_size;
    let mut output = vec![0.0f32; num_patches * hidden];
    let mut scratchpad = vec![0u8; compiled.scratchpad_bytes.max(1)];

    // seq_len parameter is the `num_patches` for this graph (Concrete).
    // batch_size is 1 (single image).
    unsafe {
        compiled.execute(
            pixels.as_ptr() as *const u8,
            weight_blob.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
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
#[derive(Debug, Clone)]
pub struct OwnedVisionWeights {
    tensors: std::collections::HashMap<String, Arc<OwnedTensor>>,
}

#[derive(Debug)]
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
        let _compiled = compiler
            .compile_graph(&graph)
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
        let compiled = compiler.compile_graph(&g).expect("compile");

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
            compiled.execute(
                image_data.as_ptr() as *const u8,
                weights.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
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
        g.add_op(OpKind::LayerNorm { eps: 1e-6 }, vec![input, ln_w, ln_b], vec![out], "ln");
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&g).expect("compile");

        let input_data: Vec<f32> = (0..seq*h).map(|i| i as f32 * 0.1).collect();
        let ln_w_data = vec![1.0f32; h];
        let ln_b_data = vec![0.0f32; h];
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(ln_w_data.as_ptr() as *const u8, h*4) });
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(ln_b_data.as_ptr() as *const u8, h*4) });

        let mut output = vec![0.0f32; seq * h];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
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
        let compiled = compiler.compile_graph(&g).expect("compile");

        let a_data: Vec<f32> = (0..m*h).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..m*h).map(|i| (i as f32 + 100.0) * 0.1).collect();
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len()*4) });
        let mut output = vec![0.0f32; m*h];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute(
                a_data.as_ptr() as *const u8, weights.as_ptr(), std::ptr::null_mut(),
                std::ptr::null(), std::ptr::null(), 1, m,
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
        g.add_op(OpKind::LayerNorm { eps: 1e-6 }, vec![input, ln_w, ln_b], vec![normed], "ln");

        let out = g.add_tensor_concrete("out", &[m, n], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: dt },
            vec![normed, w], vec![out], "gemm",
        );
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&g).expect("compile");

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
            compiled.execute(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
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
            OpKind::Gemm { m: SymDim::Concrete(m), n, k, dtype: dt },
            vec![input, w],
            vec![out],
            "gemm",
        );
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&g).expect("compile");

        let input_data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.01).collect();
        let w_data: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.005).collect();
        let mut weights = Vec::new();
        weights.extend_from_slice(unsafe { std::slice::from_raw_parts(w_data.as_ptr() as *const u8, w_data.len() * 4) });
        let mut output = vec![0.0f32; m * n];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes.max(1)];
        unsafe {
            compiled.execute(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
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
            OpKind::LayerNorm { eps: 1e-6 },
            vec![input, ln_w, ln_b],
            vec![normed],
            "ln",
        );
        let q = g.add_tensor_concrete("q", &[seq, h], dt);
        g.add_op(
            OpKind::Gemm { m: SymDim::Concrete(seq), n: h, k: h, dtype: dt },
            vec![normed, w_q],
            vec![q],
            "gemm_q",
        );
        let out = g.add_tensor_concrete("out", &[seq, h], dt);
        g.add_op(OpKind::Residual, vec![input, q], vec![out], "resid");
        g.outputs = vec![out];

        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&g).expect("compile");

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
            compiled.execute(
                input_data.as_ptr() as *const u8,
                weights.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
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
        let mut compiler = InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).expect("compile");
        eprintln!(
            "SigLIP compile: code_size={}B scratchpad={}B, weights={} (specs={})",
            compiled.code_size(),
            compiled.scratchpad_bytes,
            graph.inputs.len() - 1,
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
            .map(|i| ((i as f32 * 0.37).sin() * 0.5))
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
        let ids = MultimodalTokenIds::gemma4_defaults();
        let weights = Arc::new(OwnedVisionWeights::new());
        let err = SigLipEncoder::new(config, weights, ids).unwrap_err();
        assert!(format!("{err}").contains("missing"));
    }

    #[test]
    fn siglip_encoder_integrates_with_multimodal_context() {
        let config = tiny_config();
        let weights = Arc::new(populate_weights(&config));
        let ids = MultimodalTokenIds::gemma4_defaults();

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
        let ids = MultimodalTokenIds::gemma4_defaults();
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
        let ids = MultimodalTokenIds::gemma4_defaults();
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
        let ids = MultimodalTokenIds::gemma4_defaults();
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
        let ids = MultimodalTokenIds::gemma4_defaults();
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
}
