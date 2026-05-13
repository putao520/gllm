//! Tensor-name-driven graph builder.
//!
//! Automatically derives a complete CompilerGraph from tensor names + shapes,
//! eliminating the need for YAML architecture templates.
//!
//! Architecture: `tensor names → role index → ArchitectureFeatures → CompilerGraph`

use std::collections::HashMap;

use gllm_kernels::compiler::graph::{CompilerGraph, OpKind, SymDim};
use gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig;
use gllm_kernels::types::DType;

use crate::manifest::TensorRole;
use super::resolve::ResolvedConfig;

// ---------------------------------------------------------------------------
// Architecture Features — derived from role index
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Family {
    Decoder,
    Encoder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FfnType {
    SwiGLU,
    GeGLU,
    Standard,
    MoE,
}

/// Architecture features derived purely from tensor name presence/absence.
///
/// Every field is determined by exact role-index lookups — no heuristics.
#[derive(Debug)]
pub struct ArchitectureFeatures {
    pub family: Family,
    pub num_layers: usize,

    // Attention
    pub has_rope: bool,
    pub has_head_rms_norm: bool,
    pub has_attention_bias: bool,
    pub attention_sinks: bool,
    pub has_qk_norm: bool,
    pub has_value_norm: bool,
    pub has_embedding_scale: bool,

    // Norm
    pub norm_type: NormType,

    // FFN
    pub ffn_type: FfnType,

    // MoE
    pub is_moe: bool,
    pub has_shared_experts: bool,
    pub num_experts: usize,
    pub moe_top_k: usize,

    // Vision/Audio
    pub is_vision: bool,
    pub is_audio: bool,

    // Special
    pub has_classifier: bool,
    pub tie_lm_head: bool,
}

/// Analyze architecture features from a tensor role index.
///
/// `role_index`: maps `(TensorRole, layer_idx)` → tensor name
/// `weight_shapes`: maps tensor name → shape
/// `arch_name`: optional canonical architecture name (e.g., "gemma4", "qwen3") — used for
///              features not derivable from tensor names alone.
pub fn analyze_architecture(
    role_index: &HashMap<(TensorRole, Option<usize>), String>,
    weight_shapes: &HashMap<String, Vec<usize>>,
    arch_name: Option<&str>,
) -> ArchitectureFeatures {
    // ── Family ──
    let has_output_head = role_index.contains_key(&(TensorRole::OutputHead, None));
    let has_classifier = role_index.contains_key(&(TensorRole::ClassifierDense, None))
        || role_index.contains_key(&(TensorRole::ClassifierOutProj, None));
    let has_final_norm = role_index.contains_key(&(TensorRole::FinalNorm, None));

    // Decoder: has OutputHead or FinalNorm
    // Encoder: has Classifier or no OutputHead
    let family = if has_output_head || has_final_norm {
        Family::Decoder
    } else {
        Family::Encoder
    };

    // ── Num layers ──
    let num_layers = role_index.keys()
        .filter_map(|(_, layer_idx)| *layer_idx)
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0);

    // ── Attention features ──
    let has_head_rms_norm = role_index.contains_key(&(TensorRole::AttentionQNorm, Some(0)));
    let has_attention_bias = weight_shapes.keys().any(|n| {
        let lower = n.to_ascii_lowercase();
        lower.ends_with(".bias") && (lower.contains("q_proj") || lower.contains("k_proj") || lower.contains("v_proj"))
    });
    let attention_sinks = role_index.contains_key(&(TensorRole::AttentionSinks, Some(0)));

    // Gemma-4 QkNorm: Q/K normalization WITHOUT learnable weight (pure L2 + sqrt(d) scale).
    // Distinct from Qwen3 HeadRmsNorm which HAS a weight tensor (AttentionQNorm/KNorm roles).
    // Detection: arch is "gemma4" AND no q_norm/k_norm weight tensors present.
    let is_gemma4 = arch_name == Some("gemma4");
    let has_qk_norm = is_gemma4 && !has_head_rms_norm;

    // Gemma-4 ValueNorm: V normalization WITHOUT learnable weight (pure RMS, no gamma).
    // Detection: arch is "gemma4".
    let has_value_norm = is_gemma4;

    // Embedding scale: Gemma-4 multiplies embeddings by sqrt(hidden_size).
    // Detection: arch is "gemma4" (config.json `embedding_scale_factor`).
    let has_embedding_scale = is_gemma4;

    // ── RoPE: decoders have RoPE by default unless encoder ──
    let has_rope = family == Family::Decoder;

    // ── Norm type ──
    // Check if layer 0 InputNorm has a corresponding bias tensor
    let norm_type = if let Some(norm_name) = role_index.get(&(TensorRole::InputNorm, Some(0))) {
        let bias_name = norm_name.replace(".weight", ".bias");
        if weight_shapes.contains_key(&bias_name) {
            NormType::LayerNorm
        } else {
            NormType::RmsNorm
        }
    } else {
        // Default for decoders: RmsNorm; for encoders (BERT): LayerNorm
        if family == Family::Encoder {
            NormType::LayerNorm
        } else {
            NormType::RmsNorm
        }
    };

    // ── FFN type ──
    let has_gate = role_index.contains_key(&(TensorRole::FfnGate, Some(0)));
    let has_up = role_index.contains_key(&(TensorRole::FfnUp, Some(0)));
    let has_down = role_index.contains_key(&(TensorRole::FfnDown, Some(0)));
    let is_moe = role_index.contains_key(&(TensorRole::MoEGate, Some(0)));

    let ffn_type = if is_moe {
        FfnType::MoE
    } else if has_gate && has_up && has_down {
        // Both gate+up+down → SwiGLU or GeGLU (determined by activation config)
        FfnType::SwiGLU
    } else if has_gate && has_down {
        // gate_up_proj fused (Phi4 style) + down
        FfnType::SwiGLU
    } else {
        FfnType::Standard
    };

    // ── MoE ──
    let has_shared_experts = role_index.keys().any(|(role, _)| *role == TensorRole::MoESharedExpert);

    // num_experts from MoEGate router weight shape [hidden, num_experts]
    let num_experts = role_index.get(&(TensorRole::MoEGate, Some(0)))
        .and_then(|name| weight_shapes.get(name))
        .map(|shape| if shape.len() >= 2 { shape[1] } else { shape[0] })
        .unwrap_or(0);

    // moe_top_k: not derivable from tensor shapes alone.
    // Default to 2 (Mixtral/DeepSeek common); executor overrides from config.
    let moe_top_k = if num_experts > 0 { 2 } else { 0 };

    // ── Vision/Audio ──
    let is_vision = role_index.contains_key(&(TensorRole::PatchEmbed, None));
    let is_audio = role_index.keys().any(|(role, _)| *role == TensorRole::DepthwiseConv);

    // ── Tie lm_head ──
    let tie_lm_head = if has_output_head {
        if let (Some(embed_name), Some(lm_name)) = (
            role_index.get(&(TensorRole::Embedding, None)),
            role_index.get(&(TensorRole::OutputHead, None)),
        ) {
            embed_name == lm_name
        } else {
            false
        }
    } else {
        false
    };

    ArchitectureFeatures {
        family,
        num_layers,
        has_rope,
        has_head_rms_norm,
        has_attention_bias,
        attention_sinks,
        has_qk_norm,
        has_value_norm,
        has_embedding_scale,
        norm_type,
        ffn_type,
        is_moe,
        has_shared_experts,
        num_experts,
        moe_top_k,
        is_vision,
        is_audio,
        has_classifier,
        tie_lm_head,
    }
}

// ---------------------------------------------------------------------------
// AutoGraphBuilder — CompilerGraph from features + weight_shapes
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum GraphBuildError {
    MissingTensor(String),
    InvalidDimension(String),
    UnsupportedArchitecture(String),
}

impl std::fmt::Display for GraphBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor(s) => write!(f, "missing tensor: {s}"),
            Self::InvalidDimension(s) => write!(f, "invalid dimension: {s}"),
            Self::UnsupportedArchitecture(s) => write!(f, "unsupported architecture: {s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical name helpers
// ---------------------------------------------------------------------------

/// Canonical weight name for a per-layer tensor.
#[inline]
fn cn_layer(layer: usize, suffix: &str) -> String {
    format!("L{}.{suffix}", layer)
}

/// Canonical bias name for a per-layer tensor.
#[inline]
fn cn_layer_bias(layer: usize, suffix: &str) -> String {
    format!("L{}.{suffix}.bias", layer)
}

/// Canonical name for an MoE expert weight.
#[inline]
fn cn_expert(layer: usize, expert: usize, proj: &str) -> String {
    format!("L{}.expert.{}.{}", layer, expert, proj)
}

/// Canonical name for a shared expert weight.
#[inline]
fn cn_shared(layer: usize, proj: &str) -> String {
    format!("L{}.shared_expert.{}", layer, proj)
}

/// Helper: get shape from canonical-keyed weight_shapes, or return error.
fn get_shape(
    weight_shapes: &HashMap<String, Vec<usize>>,
    canonical: &str,
) -> Result<Vec<usize>, GraphBuildError> {
    weight_shapes.get(canonical)
        .cloned()
        .ok_or_else(|| GraphBuildError::MissingTensor(canonical.to_string()))
}

/// Build a CompilerGraph from architecture features + weight shapes.
///
/// All tensor names are canonical (e.g., `embed`, `L0.q_proj`).
/// `weight_shapes` must be keyed by canonical names (executor converts).
pub fn build_compiler_graph(
    features: &ArchitectureFeatures,
    config: &ResolvedConfig,
    weight_shapes: &HashMap<String, Vec<usize>>,
    weight_dtypes: &HashMap<String, DType>,
    business_config: &MegaKernelBusinessConfig,
    max_seq_len: usize,
) -> Result<CompilerGraph, GraphBuildError> {
    let mut g = CompilerGraph::new();

    let s = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(max_seq_len),
    };
    // Weight packing widens BF16→F32 at pack time (see mega_kernel::pack_weights_from_graph).
    // GEMM always operates on F32 packed data regardless of source dtype.
    let _ = weight_dtypes;
    let dt = DType::F32;
    let tdt = |_: &str| -> DType { dt };

    let is_encoder = features.family == Family::Encoder;
    let eps = 1e-5f32;

    // ── Derive dimensions from weight shapes (canonical names) ──
    let embed_shape = get_shape(weight_shapes, "embed")?;
    let vocab_size = embed_shape[0];
    let hidden = embed_shape[1];

    let head_dim = config.head_dim;
    let (q_dim, k_dim) = if let Some(qkv_shape) = weight_shapes.get(&cn_layer(0, "qkv_proj")) {
        // Fused QKV: derive from model geometry, not fused_n/3 (wrong for GQA)
        let qd = config.num_attention_heads * head_dim;
        let kd = config.num_key_value_heads * head_dim;
        let expected_fused = qd + 2 * kd;
        if qkv_shape[0] != expected_fused {
            return Err(GraphBuildError::InvalidDimension(
                format!("fused QKV output dim {} != expected {} (q={}+k={}+v={})",
                    qkv_shape[0], expected_fused, qd, kd, kd)
            ));
        }
        (qd, kd)
    } else {
        let q_shape = weight_shapes.get(&cn_layer(0, "q_proj"))
            .ok_or_else(|| GraphBuildError::MissingTensor(format!("{} or {}", cn_layer(0, "q_proj"), cn_layer(0, "qkv_proj"))))?;
        let k_shape = weight_shapes.get(&cn_layer(0, "k_proj"))
            .ok_or_else(|| GraphBuildError::MissingTensor(format!("{} or {}", cn_layer(0, "k_proj"), cn_layer(0, "qkv_proj"))))?;
        (q_shape[0], k_shape[0])
    };

    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;

    let intermediate_size = {
        let from_gate = weight_shapes.get(&cn_layer(0, "gate_proj")).map(|s| s[0]);
        let from_up = weight_shapes.get(&cn_layer(0, "up_proj")).map(|s| s[0]);
        let from_config = config.intermediate_size.unwrap_or(4 * hidden);
        // Fused gate_up_proj: gate_proj dim = 2 * actual intermediate_size
        match (from_gate, from_up) {
            (Some(g), None) if g == 2 * from_config => from_config,
            (Some(g), Some(_)) | (Some(g), None) => g,
            (None, Some(u)) => u,
            (None, None) => from_config,
        }
    };

    let use_rms = features.norm_type == NormType::RmsNorm;

    let mut tensor_map: HashMap<String, gllm_kernels::compiler::graph::TensorId> = HashMap::new();

    // ── SharedKvRef: per-layer K/V tensor tracking ──
    // Maps layer_idx -> (k_for_attn_tensor_id, v_out_tensor_id) so that
    // KV-sharing consumer layers can reference their donor's K/V tensors.
    let mut layer_kv_tensors: HashMap<usize, (gllm_kernels::compiler::graph::TensorId, gllm_kernels::compiler::graph::TensorId)> = HashMap::new();

    // ── Embedding: token_ids → Gather → hidden_0 (shared by decoder & encoder) ──
    let token_ids = g.add_tensor("token_ids", vec![s.clone()], dt);
    let embed_w = g.add_tensor_concrete("embed", &[vocab_size, hidden], tdt("embed"));
    let embedding = g.add_tensor("embedding", vec![s.clone(), SymDim::Concrete(hidden)], dt);
    g.add_op(
        OpKind::Gather {
            table_rows: vocab_size, embed_dim: hidden,
            index_dim: s.clone(), indices_kind: Default::default(),
        },
        vec![token_ids, embed_w],
        vec![embedding],
        "embed_gather",
    );
    tensor_map.insert("hidden_0".to_string(), embedding);

    // ── Layer loop ──
    for i in 0..features.num_layers {
        let hidden_tid = *tensor_map.get("hidden_0")
            .ok_or_else(|| GraphBuildError::InvalidDimension("no hidden_0".into()))?;

        // ── Input norm ──
        let in_norm_cn = cn_layer(i, "input_norm");
        let norm_w_tid = g.add_tensor_concrete(&in_norm_cn, &[hidden], tdt(&in_norm_cn));
        let normed = g.add_tensor(&format!("L{}_normed", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
        if use_rms {
            g.add_op(OpKind::RmsNorm { eps }, vec![hidden_tid, norm_w_tid], vec![normed], &format!("L{}_input_norm", i));
        } else {
            let bias_cn = cn_layer_bias(i, "input_norm");
            let bias_tid = g.add_tensor_concrete(&bias_cn, &[hidden], tdt(&bias_cn));
            g.add_op(OpKind::LayerNorm { eps }, vec![hidden_tid, norm_w_tid, bias_tid], vec![normed], &format!("L{}_input_norm", i));
        }

        // ── SharedKvRef: check if this layer shares KV with a donor ──
        let is_shared_kv = config.is_kv_shared_layer(i);

        // ── Q projection (always computed, even for KV-shared layers) ──
        // Check for fused QKV (e.g., Phi4-mini qkv_proj)
        let fused_qkv_shape = weight_shapes.get(&cn_layer(i, "qkv_proj")).cloned();
        let (q_out, fused_qkv_out, q_n) = if let Some(ref fqkv_shape) = fused_qkv_shape {
            // Fused QKV: single GEMM → ColumnSlice into Q, K, V
            let fused_n = fqkv_shape[0];
            let fused_k = fqkv_shape[1];
            let fqkv_cn = cn_layer(i, "qkv_proj");
            let fqkv_w = g.add_tensor_concrete(&fqkv_cn, &[fused_n, fused_k], tdt(&fqkv_cn));
            let fused_out = g.add_tensor(&format!("L{}_qkv", i), vec![s.clone(), SymDim::Concrete(fused_n)], dt);
            g.add_op(OpKind::Gemm { m: s.clone(), n: fused_n, k: fused_k, dtype: dt, trans_b: true },
                vec![normed, fqkv_w], vec![fused_out], &format!("L{}_qkv_proj", i));
            let q_slice = g.add_tensor(&format!("L{}_q", i), vec![s.clone(), SymDim::Concrete(q_dim)], dt);
            g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: 0, slice_dim: q_dim },
                vec![fused_out], vec![q_slice], &format!("L{}_q_slice", i));
            (q_slice, Some((fused_out, fused_n)), q_dim)
        } else {
            let q_s = weight_shapes.get(&cn_layer(i, "q_proj"));
            let q_n = q_s.map(|s| s[0]).unwrap_or(q_dim);
            let q_k = q_s.map(|s| s[1]).unwrap_or(hidden);
            let q_cn = cn_layer(i, "q_proj");
            let q_w = g.add_tensor_concrete(&q_cn, &[q_n, q_k], tdt(&q_cn));
            let q_out = g.add_tensor(&format!("L{}_q", i), vec![s.clone(), SymDim::Concrete(q_n)], dt);
            g.add_op(OpKind::Gemm { m: s.clone(), n: q_n, k: q_k, dtype: dt, trans_b: true },
                vec![normed, q_w], vec![q_out], &format!("L{}_q_proj", i));
            (q_out, None, q_n)
        };

        // ── K and V projections ──
        // For KV-sharing consumer layers: skip K/V GEMMs and reference the
        // donor layer's K/V tensors (post-norm, post-RoPE) instead.
        // For fused QKV layers: ColumnSlice from the fused output.
        let (k_for_attn_final, v_out) = if let Some((ref fused_out_tid, fused_n)) = fused_qkv_out {
            // Fused QKV: ColumnSlice K and V from the fused output
            let k_slice = g.add_tensor(&format!("L{}_k", i), vec![s.clone(), SymDim::Concrete(k_dim)], dt);
            g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: q_dim, slice_dim: k_dim },
                vec![*fused_out_tid], vec![k_slice], &format!("L{}_k_slice", i));
            let v_slice = g.add_tensor(&format!("L{}_v", i), vec![s.clone(), SymDim::Concrete(k_dim)], dt);
            g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: q_dim + k_dim, slice_dim: k_dim },
                vec![*fused_out_tid], vec![v_slice], &format!("L{}_v_slice", i));
            (k_slice, v_slice)
        } else if is_shared_kv {
            // Resolve donor layer index and reference its K/V tensors.
            // The donor is strictly before the shared window, so its K/V
            // tensor IDs have already been stored in layer_kv_tensors.
            let donor_idx = config.donor_layer(i)
                .map_err(|e| GraphBuildError::InvalidDimension(
                    format!("layer {i} SharedKvRef donor resolution: {e}")
                ))?
                .ok_or_else(|| GraphBuildError::InvalidDimension(
                    format!("layer {i} is KV-shared but donor_layer returned None")
                ))?;
            *layer_kv_tensors.get(&donor_idx)
                .ok_or_else(|| GraphBuildError::InvalidDimension(
                    format!("layer {i} SharedKvRef: donor layer {donor_idx} K/V tensors not found")
                ))?
        } else {
            // Normal layer: add K and V projection GEMMs.
            let k_s = weight_shapes.get(&cn_layer(i, "k_proj"));
            let k_n = k_s.map(|s| s[0]).unwrap_or(k_dim);
            let k_k = k_s.map(|s| s[1]).unwrap_or(hidden);

            let v_s = weight_shapes.get(&cn_layer(i, "v_proj"));
            let v_n = v_s.map(|s| s[0]).unwrap_or(k_dim);
            let v_k = v_s.map(|s| s[1]).unwrap_or(hidden);

            let k_cn = cn_layer(i, "k_proj");
            let k_w = g.add_tensor_concrete(&k_cn, &[k_n, k_k], tdt(&k_cn));
            let k_out = g.add_tensor(&format!("L{}_k", i), vec![s.clone(), SymDim::Concrete(k_n)], dt);
            g.add_op(OpKind::Gemm { m: s.clone(), n: k_n, k: k_k, dtype: dt, trans_b: true },
                vec![normed, k_w], vec![k_out], &format!("L{}_k_proj", i));

            let v_cn = cn_layer(i, "v_proj");
            let v_w = g.add_tensor_concrete(&v_cn, &[v_n, v_k], tdt(&v_cn));
            let v_out = g.add_tensor(&format!("L{}_v", i), vec![s.clone(), SymDim::Concrete(v_n)], dt);
            g.add_op(OpKind::Gemm { m: s.clone(), n: v_n, k: v_k, dtype: dt, trans_b: true },
                vec![normed, v_w], vec![v_out], &format!("L{}_v_proj", i));

            (k_out, v_out)
        };

        // ── HeadRmsNorm (optional, Qwen3-style with learnable weights) ──
        // Q always gets normalized (computed locally).
        // K: for KV-shared layers, donor already normalized, so skip.
        let mut q_for_attn = q_out;
        let mut k_for_attn = k_for_attn_final;
        if features.has_head_rms_norm {
            let head_rms_eps = 1e-6f32;
            // Q normalization: always applied
            let q_norm_cn = cn_layer(i, "q_norm");
            let q_norm_w = g.add_tensor_concrete(&q_norm_cn, &[head_dim], tdt(&q_norm_cn));
            let q_normed = g.add_tensor(&format!("L{}_q_normed", i), vec![s.clone(), SymDim::Concrete(q_n)], dt);
            g.add_op(OpKind::HeadRmsNorm { head_dim, eps: head_rms_eps },
                vec![q_out, q_norm_w], vec![q_normed], &format!("L{}_q_norm", i));
            q_for_attn = q_normed;

            // K normalization: only for non-shared layers
            if !is_shared_kv {
                let k_n_for_norm = weight_shapes.get(&cn_layer(i, "k_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let k_norm_cn = cn_layer(i, "k_norm");
                let k_norm_w = g.add_tensor_concrete(&k_norm_cn, &[head_dim], tdt(&k_norm_cn));
                let k_normed = g.add_tensor(&format!("L{}_k_normed", i), vec![s.clone(), SymDim::Concrete(k_n_for_norm)], dt);
                g.add_op(OpKind::HeadRmsNorm { head_dim, eps: head_rms_eps },
                    vec![k_for_attn, k_norm_w], vec![k_normed], &format!("L{}_k_norm", i));
                k_for_attn = k_normed;
            }
        }

        // ── RoPE (optional) ──
        // Q always gets RoPE (computed locally).
        // K: for KV-shared layers, donor already applied RoPE, so skip.
        if features.has_rope {
            let theta = config.rope_theta;
            let rope_q = g.add_tensor(&format!("L{}_q_rope", i), vec![s.clone(), SymDim::Concrete(q_n)], dt);
            g.add_op(
                OpKind::RoPE { num_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: None },
                vec![q_for_attn], vec![rope_q], &format!("L{}_rope_q", i));
            q_for_attn = rope_q;

            // K RoPE: only for non-shared layers
            if !is_shared_kv {
                let k_n_for_rope = weight_shapes.get(&cn_layer(i, "k_proj"))
                    .map(|s| s[0]).unwrap_or(k_dim);
                let rope_k = g.add_tensor(&format!("L{}_k_rope", i), vec![s.clone(), SymDim::Concrete(k_n_for_rope)], dt);
                g.add_op(
                    OpKind::RoPE { num_heads: num_kv_heads, head_dim, theta, partial: config.rope_partial_ratio, rope_scaling: None },
                    vec![k_for_attn], vec![rope_k], &format!("L{}_rope_k", i));
                k_for_attn = rope_k;
            }
        }

        // ── Store K/V tensors for SharedKvRef donor lookup ──
        // Only non-shared layers can be donors; their K/V tensor IDs are
        // the final values after optional normalization/RoPE.
        if !is_shared_kv {
            layer_kv_tensors.insert(i, (k_for_attn, v_out));
        }

        // ── Attention ──
        let causal = !is_encoder;
        let attn_out = g.add_tensor(&format!("L{}_attn", i), vec![s.clone(), SymDim::Concrete(q_n)], dt);
        g.add_op(
            OpKind::MultiHeadAttention {
                seq_len: s.clone(), num_heads, num_kv_heads, head_dim, causal,
                attention_sinks: features.attention_sinks,
            },
            vec![q_for_attn, k_for_attn, v_out],
            vec![attn_out],
            &format!("L{}_mha", i),
        );

        // ── O projection ──
        let o_cn = cn_layer(i, "o_proj");
        let (o_n, o_k_dim) = weight_shapes.get(&o_cn)
            .map(|s| (s[0], s[1]))
            .unwrap_or((hidden, hidden));

        let o_w = g.add_tensor_concrete(&o_cn, &[o_n, o_k_dim], tdt(&o_cn));
        let o_out = g.add_tensor(&format!("L{}_o", i), vec![s.clone(), SymDim::Concrete(o_n)], dt);
        g.add_op(OpKind::Gemm { m: s.clone(), n: o_n, k: o_k_dim, dtype: dt, trans_b: true },
            vec![attn_out, o_w], vec![o_out], &format!("L{}_o_proj", i));

        // ── Residual ──
        let resid = g.add_tensor(&format!("L{}_attn_resid", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
        g.add_op(OpKind::Add, vec![hidden_tid, o_out], vec![resid], &format!("L{}_attn_resid", i));

        // ── Post-attention norm ──
        let post_cn = cn_layer(i, "post_attn_norm");
        let post_norm_w = g.add_tensor_concrete(&post_cn, &[hidden], tdt(&post_cn));
        let post_normed = g.add_tensor(&format!("L{}_post_normed", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
        if use_rms {
            g.add_op(OpKind::RmsNorm { eps }, vec![resid, post_norm_w], vec![post_normed], &format!("L{}_post_norm", i));
        } else {
            let bias_cn = cn_layer_bias(i, "post_attn_norm");
            let bias_tid = g.add_tensor_concrete(&bias_cn, &[hidden], tdt(&bias_cn));
            g.add_op(OpKind::LayerNorm { eps }, vec![resid, post_norm_w, bias_tid], vec![post_normed], &format!("L{}_post_norm", i));
        }

        // ── FFN ──
        match &features.ffn_type {
            FfnType::SwiGLU | FfnType::GeGLU => {
                let gate_cn = cn_layer(i, "gate_proj");
                let up_cn = cn_layer(i, "up_proj");
                let has_fused_gate_up = weight_shapes.get(&up_cn).is_none()
                    && weight_shapes.get(&gate_cn).map(|s| s[0]) == Some(2 * intermediate_size);

                let (gate_out, up_out) = if has_fused_gate_up {
                    // Fused gate_up_proj: single GEMM → ColumnSlice into gate and up
                    let fused_n = 2 * intermediate_size;
                    let gate_w = g.add_tensor_concrete(&gate_cn, &[fused_n, hidden], tdt(&gate_cn));
                    let fused_out = g.add_tensor(&format!("L{}_gate_up", i), vec![s.clone(), SymDim::Concrete(fused_n)], dt);
                    g.add_op(OpKind::Gemm { m: s.clone(), n: fused_n, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, gate_w], vec![fused_out], &format!("L{}_gate_up_proj", i));

                    let gate_slice = g.add_tensor(&format!("L{}_gate", i), vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                    g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: 0, slice_dim: intermediate_size },
                        vec![fused_out], vec![gate_slice], &format!("L{}_gate_slice", i));

                    let up_slice = g.add_tensor(&format!("L{}_up", i), vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                    g.add_op(OpKind::ColumnSlice { seq_len: s.clone(), input_inner: fused_n, start: intermediate_size, slice_dim: intermediate_size },
                        vec![fused_out], vec![up_slice], &format!("L{}_up_slice", i));
                    (gate_slice, up_slice)
                } else {
                    // Separate gate_proj + up_proj
                    let gate_n = weight_shapes.get(&gate_cn).map(|s| s[0]).unwrap_or(intermediate_size);
                    let gate_w = g.add_tensor_concrete(&gate_cn, &[gate_n, hidden], tdt(&gate_cn));
                    let gate_o = g.add_tensor(&format!("L{}_gate", i), vec![s.clone(), SymDim::Concrete(gate_n)], dt);
                    g.add_op(OpKind::Gemm { m: s.clone(), n: gate_n, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, gate_w], vec![gate_o], &format!("L{}_gate_proj", i));

                    let up_n = weight_shapes.get(&up_cn).map(|s| s[0]).unwrap_or(intermediate_size);
                    let up_w = g.add_tensor_concrete(&up_cn, &[up_n, hidden], tdt(&up_cn));
                    let up_o = g.add_tensor(&format!("L{}_up", i), vec![s.clone(), SymDim::Concrete(up_n)], dt);
                    g.add_op(OpKind::Gemm { m: s.clone(), n: up_n, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, up_w], vec![up_o], &format!("L{}_up_proj", i));
                    (gate_o, up_o)
                };

                let swiglu_out = g.add_tensor(&format!("L{}_swiglu", i), vec![s.clone(), SymDim::Concrete(intermediate_size)], dt);
                g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![swiglu_out], &format!("L{}_swiglu", i));

                let down_cn = cn_layer(i, "down_proj");
                let down_k = weight_shapes.get(&down_cn).map(|s| s[1]).unwrap_or(intermediate_size);

                let down_w = g.add_tensor_concrete(&down_cn, &[hidden, down_k], tdt(&down_cn));
                let down_out = g.add_tensor(&format!("L{}_down", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Gemm { m: s.clone(), n: hidden, k: down_k, dtype: dt, trans_b: true },
                    vec![swiglu_out, down_w], vec![down_out], &format!("L{}_down_proj", i));

                let ffn_resid = g.add_tensor(&format!("L{}_ffn_resid", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![resid, down_out], vec![ffn_resid], &format!("L{}_ffn_resid", i));
                tensor_map.insert("hidden_0".to_string(), ffn_resid);
            }
            FfnType::Standard => {
                let up_cn = cn_layer(i, "up_proj");
                let up_n = weight_shapes.get(&up_cn).map(|s| s[0]).unwrap_or(intermediate_size);

                let up_w = g.add_tensor_concrete(&up_cn, &[up_n, hidden], tdt(&up_cn));
                let up_out = g.add_tensor(&format!("L{}_up", i), vec![s.clone(), SymDim::Concrete(up_n)], dt);
                g.add_op(OpKind::Gemm { m: s.clone(), n: up_n, k: hidden, dtype: dt, trans_b: true },
                    vec![post_normed, up_w], vec![up_out], &format!("L{}_up_proj", i));

                let act_out = g.add_tensor(&format!("L{}_act", i), vec![s.clone(), SymDim::Concrete(up_n)], dt);
                g.add_op(OpKind::Gelu, vec![up_out], vec![act_out], &format!("L{}_gelu", i));

                let down_cn = cn_layer(i, "down_proj");
                let down_w = g.add_tensor_concrete(&down_cn, &[hidden, up_n], tdt(&down_cn));
                let down_out = g.add_tensor(&format!("L{}_down", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Gemm { m: s.clone(), n: hidden, k: up_n, dtype: dt, trans_b: true },
                    vec![act_out, down_w], vec![down_out], &format!("L{}_down_proj", i));

                let ffn_resid = g.add_tensor(&format!("L{}_ffn_resid", i), vec![s.clone(), SymDim::Concrete(hidden)], dt);
                g.add_op(OpKind::Add, vec![resid, down_out], vec![ffn_resid], &format!("L{}_ffn_resid", i));
                tensor_map.insert("hidden_0".to_string(), ffn_resid);
            }
            FfnType::MoE => {
                let num_experts = features.num_experts;
                let top_k = features.moe_top_k;

                if num_experts == 0 {
                    return Err(GraphBuildError::MissingTensor(
                        format!("layer {} MoE router weight", i),
                    ));
                }

                // Router weight dimensions
                let router_cn = cn_layer(i, "moe_gate");
                let (router_n, router_k) = weight_shapes.get(&router_cn)
                    .map(|s| (s[0], s[1]))
                    .unwrap_or((hidden, num_experts));

                // Expert intermediate size from canonical name
                let inter = weight_shapes.get(&cn_expert(i, 0, "gate_proj"))
                    .map(|s| s[0])
                    .unwrap_or(intermediate_size);

                // Router
                let w_router = g.add_tensor_concrete(&router_cn, &[router_n, router_k], tdt(&router_cn));
                let gate_probs = g.add_tensor(
                    &format!("L{}_gate_probs", i),
                    vec![s.clone(), SymDim::Concrete(num_experts)],
                    dt,
                );
                g.add_op(
                    OpKind::MoEGate {
                        seq_len: max_seq_len,
                        num_experts, hidden, top_k,
                    },
                    vec![post_normed, w_router],
                    vec![gate_probs],
                    &format!("L{}_moe_gate", i),
                );

                // Top-K selection
                let topk_idx = g.add_tensor(
                    &format!("L{}_topk_idx", i),
                    vec![s.clone(), SymDim::Concrete(top_k)],
                    DType::F32,
                );
                let topk_w = g.add_tensor(
                    &format!("L{}_topk_w", i),
                    vec![s.clone(), SymDim::Concrete(top_k)],
                    DType::F32,
                );
                g.add_op(
                    OpKind::TopK {
                        seq_len: max_seq_len,
                        num_experts, top_k,
                    },
                    vec![gate_probs],
                    vec![topk_idx, topk_w],
                    &format!("L{}_topk", i),
                );

                // Expert accumulator (zeroed buffer, bound at runtime)
                let mut current_acc = g.add_tensor(
                    &format!("L{}_expert_acc", i),
                    vec![s.clone(), SymDim::Concrete(hidden)],
                    dt,
                );

                // Per-expert FFN: gate → mask → up → swiglu → down → conditional_add
                for e in 0..num_experts {
                    let gate_exp_cn = cn_expert(i, e, "gate_proj");
                    let w_gate_e = g.add_tensor_concrete(&gate_exp_cn, &[hidden, inter], tdt(&gate_exp_cn));
                    let gate_out = g.add_tensor(
                        &format!("L{}_exp{}_gate", i, e),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    g.add_op(
                        OpKind::Gemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, w_gate_e],
                        vec![gate_out],
                        &format!("L{}_exp{}_gate_gemm", i, e),
                    );

                    let mask_out = g.add_tensor(
                        &format!("L{}_exp{}_mask", i, e),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    g.add_op(
                        OpKind::GateMask { hidden: inter },
                        vec![gate_out],
                        vec![mask_out],
                        &format!("L{}_exp{}_gate_mask", i, e),
                    );

                    let up_exp_cn = cn_expert(i, e, "up_proj");
                    let w_up_e = g.add_tensor_concrete(&up_exp_cn, &[hidden, inter], tdt(&up_exp_cn));
                    let up_out = g.add_tensor(
                        &format!("L{}_exp{}_up", i, e),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    g.add_op(
                        OpKind::MaskedGemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, w_up_e, mask_out],
                        vec![up_out],
                        &format!("L{}_exp{}_up_gemm", i, e),
                    );

                    let swiglu_out = g.add_tensor(
                        &format!("L{}_exp{}_swiglu", i, e),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    g.add_op(
                        OpKind::SwiGlu,
                        vec![gate_out, up_out],
                        vec![swiglu_out],
                        &format!("L{}_exp{}_swiglu", i, e),
                    );

                    let down_exp_cn = cn_expert(i, e, "down_proj");
                    let w_down_e = g.add_tensor_concrete(&down_exp_cn, &[inter, hidden], tdt(&down_exp_cn));
                    let down_out = g.add_tensor(
                        &format!("L{}_exp{}_down", i, e),
                        vec![s.clone(), SymDim::Concrete(hidden)],
                        dt,
                    );
                    g.add_op(
                        OpKind::Gemm { m: s.clone(), n: hidden, k: inter, dtype: dt, trans_b: true },
                        vec![swiglu_out, w_down_e],
                        vec![down_out],
                        &format!("L{}_exp{}_down_gemm", i, e),
                    );

                    let next_acc = g.add_tensor(
                        &format!("L{}_exp{}_acc", i, e),
                        vec![s.clone(), SymDim::Concrete(hidden)],
                        dt,
                    );
                    g.add_op(
                        OpKind::MoEConditionalAdd {
                            seq_len: s.clone(),
                            hidden,
                            num_experts,
                            expert_idx: e,
                        },
                        vec![current_acc, down_out, gate_probs],
                        vec![next_acc],
                        &format!("L{}_exp{}_cond_add", i, e),
                    );
                    current_acc = next_acc;
                }

                // Shared experts (optional)
                if features.has_shared_experts {
                    let se_gate_cn = cn_shared(i, "gate_proj");
                    let se_gate_w = g.add_tensor_concrete(&se_gate_cn, &[hidden, inter], tdt(&se_gate_cn));
                    let se_gate = g.add_tensor(
                        &format!("L{}_shared_gate", i),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    let se_up_cn = cn_shared(i, "up_proj");
                    let se_up_w = g.add_tensor_concrete(&se_up_cn, &[hidden, inter], tdt(&se_up_cn));
                    let se_up = g.add_tensor(
                        &format!("L{}_shared_up", i),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    let se_swiglu = g.add_tensor(
                        &format!("L{}_shared_swiglu", i),
                        vec![s.clone(), SymDim::Concrete(inter)],
                        dt,
                    );
                    let se_down_cn = cn_shared(i, "down_proj");
                    let se_down_w = g.add_tensor_concrete(&se_down_cn, &[inter, hidden], tdt(&se_down_cn));
                    let se_down = g.add_tensor(
                        &format!("L{}_shared_down", i),
                        vec![s.clone(), SymDim::Concrete(hidden)],
                        dt,
                    );
                    let se_out = g.add_tensor(
                        &format!("L{}_shared_out", i),
                        vec![s.clone(), SymDim::Concrete(hidden)],
                        dt,
                    );

                    g.add_op(OpKind::Gemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, se_gate_w], vec![se_gate], &format!("L{}_shared_gate_gemm", i));
                    g.add_op(OpKind::Gemm { m: s.clone(), n: inter, k: hidden, dtype: dt, trans_b: true },
                        vec![post_normed, se_up_w], vec![se_up], &format!("L{}_shared_up_gemm", i));
                    g.add_op(OpKind::SwiGlu, vec![se_gate, se_up], vec![se_swiglu], &format!("L{}_shared_swiglu", i));
                    g.add_op(OpKind::Gemm { m: s.clone(), n: hidden, k: inter, dtype: dt, trans_b: true },
                        vec![se_swiglu, se_down_w], vec![se_down], &format!("L{}_shared_down_gemm", i));
                    g.add_op(OpKind::Add, vec![current_acc, se_down], vec![se_out], &format!("L{}_shared_add", i));
                    current_acc = se_out;
                }

                // FFN residual
                let ffn_resid = g.add_tensor(
                    &format!("L{}_ffn_resid", i),
                    vec![s.clone(), SymDim::Concrete(hidden)],
                    dt,
                );
                g.add_op(
                    OpKind::Add,
                    vec![resid, current_acc],
                    vec![ffn_resid],
                    &format!("L{}_moe_resid", i),
                );
                tensor_map.insert("hidden_0".to_string(), ffn_resid);
            }
        }
    }

    let final_hidden = tensor_map.get("hidden_0")
        .copied()
        .ok_or_else(|| GraphBuildError::InvalidDimension("no hidden_0 after layer loop".into()))?;

    // ── Encoder post-layer: MeanPool → Classifier head ──
    if is_encoder {
        // MeanPool: average over seq dimension → [hidden]
        let pooled = g.add_tensor("pooled", vec![SymDim::Concrete(hidden)], dt);
        g.add_op(OpKind::MeanPool { seq_len: 0, hidden, cls_mode: false }, vec![final_hidden], vec![pooled], "meanpool");

        // Classifier head (if present): Dense → tanh → OutProj → output
        if features.has_classifier {
            // Detect classifier dense weight by canonical name patterns
            let cls_dense_cn = weight_shapes.keys()
                .find(|k| k.contains("classifier") && k.contains("dense") && !k.contains("bias"))
                .cloned().unwrap_or("classifier.dense.weight".to_string());
            let cls_dense_shape = weight_shapes.get(&cls_dense_cn);
            let (cls_n, cls_k) = match cls_dense_shape {
                Some(s) if s.len() >= 2 => (s[0], s[1]),
                Some(s) if s.len() == 1 => (s[0], hidden),
                _ => (hidden, hidden),
            };
            let cls_dense_w = g.add_tensor_concrete(&cls_dense_cn, &[cls_n, cls_k], tdt(&cls_dense_cn));
            let cls_dense_out = g.add_tensor("cls_dense_out", vec![SymDim::Concrete(cls_n)], dt);
            g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: cls_n, k: cls_k, dtype: dt, trans_b: true },
                vec![pooled, cls_dense_w], vec![cls_dense_out], "cls_dense");

            // tanh activation
            let cls_act = g.add_tensor("cls_act", vec![SymDim::Concrete(cls_n)], dt);
            g.add_op(OpKind::Tanh, vec![cls_dense_out], vec![cls_act], "cls_tanh");

            // OutProj: [cls_n] → [num_labels] (typically 1 for rerankers)
            let cls_out_cn = weight_shapes.keys()
                .find(|k| (k.contains("classifier") || k.contains("score")) && !k.contains("dense") && !k.contains("bias"))
                .cloned().unwrap_or("classifier.out_proj.weight".to_string());
            let cls_out_shape = weight_shapes.get(&cls_out_cn);
            let (num_labels, cls_out_k) = match cls_out_shape {
                Some(s) if s.len() >= 2 => (s[0], s[1]),
                Some(s) if s.len() == 1 => (s[0], cls_n),
                _ => (1, cls_n),
            };
            let cls_out_w = g.add_tensor_concrete(&cls_out_cn, &[num_labels, cls_out_k], tdt(&cls_out_cn));
            let cls_result = g.add_tensor("cls_result", vec![SymDim::Concrete(num_labels)], dt);
            g.add_op(OpKind::Gemm { m: SymDim::Concrete(1), n: num_labels, k: cls_out_k, dtype: dt, trans_b: true },
                vec![cls_act, cls_out_w], vec![cls_result], "cls_out_proj");

            // Check for classifier bias — use GemmBias if present
            let cls_out_bias_cn = cls_out_cn.clone().replace(".weight", ".bias");
            let has_bias = weight_shapes.contains_key(&cls_out_bias_cn);
            if has_bias {
                // Rebuild with GemmBias: undo the Gemm, redo as GemmBias
                let bias_tid = g.add_tensor_concrete(&cls_out_bias_cn, &[num_labels], tdt(&cls_out_bias_cn));
                // Overwrite the last Gemm with GemmBias (reuse same output tensor)
                let biased = g.add_tensor("cls_result_biased", vec![SymDim::Concrete(num_labels)], dt);
                g.add_op(OpKind::GemmBias { m: SymDim::Concrete(1), n: num_labels, k: cls_out_k, dtype: dt, trans_b: true },
                    vec![cls_act, cls_out_w, bias_tid], vec![biased], "cls_out_proj_biased");
                g.outputs = vec![biased];
            } else {
                g.outputs = vec![cls_result];
            }
        } else {
            // No classifier: output the pooled hidden state
            g.outputs = vec![pooled];
        }
    }

    // ── Decoder post-layer ──
    if !is_encoder {
        let final_norm_w = g.add_tensor_concrete("final_norm", &[hidden], tdt("final_norm"));
        let final_normed = g.add_tensor("final_normed", vec![s.clone(), SymDim::Concrete(hidden)], dt);
        if use_rms {
            g.add_op(OpKind::RmsNorm { eps }, vec![final_hidden, final_norm_w], vec![final_normed], "final_norm");
        } else {
            let bias_tid = g.add_tensor_concrete("final_norm.bias", &[hidden], tdt("final_norm.bias"));
            g.add_op(OpKind::LayerNorm { eps }, vec![final_hidden, final_norm_w, bias_tid], vec![final_normed], "final_norm");
        }

        use gllm_kernels::compiler::mega_kernel_abi::OutputMode;
        let is_embed_or_rerank = business_config.output_modes.iter().any(|m| matches!(m, OutputMode::EncodeToLayer { .. }));

        if is_embed_or_rerank {
            // Decoder used as embedding/reranker: MeanPool → output hidden state
            let pooled = g.add_tensor("pooled", vec![SymDim::Concrete(hidden)], dt);
            g.add_op(OpKind::MeanPool { seq_len: 0, hidden, cls_mode: false }, vec![final_normed], vec![pooled], "meanpool");
            g.outputs = vec![pooled];
        } else {
            // Generator: lm_head → Argmax → generate loop
            let lm_head_w = g.add_tensor_concrete("lm_head", &[vocab_size, hidden], tdt("lm_head"));
            let logits = g.add_tensor("logits", vec![s.clone(), SymDim::Concrete(vocab_size)], dt);
            g.add_op(
                OpKind::Gemm { m: s.clone(), n: vocab_size, k: hidden, dtype: dt, trans_b: true },
                vec![final_normed, lm_head_w],
                vec![logits],
                "lm_head",
            );

            for mode in &business_config.output_modes {
                match mode {
                    OutputMode::Generate { .. } => {
                        let token_id = g.add_tensor("token_id", vec![SymDim::Concrete(1)], dt);
                        g.add_op(OpKind::Argmax { vocab_size }, vec![logits], vec![token_id], "argmax");
                        g.add_op(OpKind::StoreToken, vec![token_id], vec![], "store_token");
                        g.add_op(OpKind::CheckStopCondition, vec![token_id], vec![], "check_stop");
                    }
                    _ => {}
                }
            }
        }
    }

    // Set graph inputs: all tensors without a producer (external inputs).
    // First input = activation (first non-weight tensor), rest = weights.
    // weight_layout() uses inputs[1..] for offset calculation.
    {
        let mut external: Vec<_> = g.tensors.iter()
            .filter(|t| t.producer.is_none())
            .collect();
        // Sort: activation-like tensors first (hidden/embedding), then weights
        external.sort_by_key(|t| {
            let name = t.name.to_ascii_lowercase();
            // Activation tensors go first (token_ids, input, hidden_0, embedding)
            if name.contains("token") || name == "input" || name == "hidden_0" {
                0u8
            } else {
                1u8
            }
        });
        g.inputs = external.iter().map(|t| t.id).collect();
    }

    g.max_seq_len = max_seq_len;
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::compiler::mega_kernel_abi::MegaKernelBusinessConfig;

    fn make_role_index(entries: Vec<(TensorRole, Option<usize>, &str)>) -> HashMap<(TensorRole, Option<usize>), String> {
        entries.into_iter().map(|(r, l, n)| ((r, l), n.to_string())).collect()
    }

    fn make_weight_shapes(entries: Vec<(&str, Vec<usize>)>) -> HashMap<String, Vec<usize>> {
        entries.into_iter().map(|(n, s)| (n.to_string(), s)).collect()
    }

    fn make_config(num_layers: usize, hidden: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize) -> ResolvedConfig {
        ResolvedConfig {
            num_hidden_layers: num_layers,
            hidden_size: hidden,
            num_attention_heads: num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
            intermediate_size: Some(hidden * 4),
            vocab_size: 100,
            rope_theta: 10000.0,
            dtype: "f32".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn auto_decoder_with_swiglu() {
        let config = make_config(2, 64, 4, 2, 16);

        let mut ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (TensorRole::InputNorm, Some(0), "model.layers.0.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(0), "model.layers.0.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(0), "model.layers.0.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(0), "model.layers.0.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(0), "model.layers.0.self_attn.o_proj.weight"),
            (TensorRole::PostAttnNorm, Some(0), "model.layers.0.post_attention_layernorm.weight"),
            (TensorRole::FfnGate, Some(0), "model.layers.0.mlp.gate_proj.weight"),
            (TensorRole::FfnUp, Some(0), "model.layers.0.mlp.up_proj.weight"),
            (TensorRole::FfnDown, Some(0), "model.layers.0.mlp.down_proj.weight"),
            // Layer 1
            (TensorRole::InputNorm, Some(1), "model.layers.1.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(1), "model.layers.1.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(1), "model.layers.1.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(1), "model.layers.1.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(1), "model.layers.1.self_attn.o_proj.weight"),
            (TensorRole::PostAttnNorm, Some(1), "model.layers.1.post_attention_layernorm.weight"),
            (TensorRole::FfnGate, Some(1), "model.layers.1.mlp.gate_proj.weight"),
            (TensorRole::FfnUp, Some(1), "model.layers.1.mlp.up_proj.weight"),
            (TensorRole::FfnDown, Some(1), "model.layers.1.mlp.down_proj.weight"),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None);
        assert_eq!(features.family, Family::Decoder);
        assert_eq!(features.num_layers, 2);
        assert!(features.has_rope);
        assert!(!features.has_head_rms_norm);
        assert_eq!(features.ffn_type, FfnType::SwiGLU);

        // Canonical-keyed weight_shapes (as executor would provide)
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("L1.input_norm", vec![64]),
            ("L1.q_proj", vec![64, 64]),
            ("L1.k_proj", vec![32, 64]),
            ("L1.v_proj", vec![32, 64]),
            ("L1.o_proj", vec![64, 64]),
            ("L1.post_attn_norm", vec![64]),
            ("L1.gate_proj", vec![256, 64]),
            ("L1.up_proj", vec![256, 64]),
            ("L1.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(&features, &config, &ws, &std::collections::HashMap::new(), &MegaKernelBusinessConfig::default(), 2048)
            .expect("graph build should succeed");

        // 2 layers × (input_norm + q + k + v + rope_q + rope_k + mha + o + resid + post_norm + gate + up + swiglu + down + ffn_resid)
        // = 2 × 15 = 30
        // + embed_gather + final_norm + lm_head = 3
        // + argmax + store_token + check_stop = 3
        assert_eq!(graph.ops.len(), 36, "expected 36 ops, got {}: {:?}",
            graph.ops.len(), graph.ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>());

        // Verify canonical tensor names are used
        let tensor_names: Vec<&str> = graph.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(tensor_names.iter().any(|n| *n == "embed"),
            "embedding tensor should use canonical name 'embed', got: {:?}", tensor_names);
        assert!(tensor_names.iter().any(|n| *n == "L0.input_norm"),
            "input_norm tensor should use canonical name 'L0.input_norm', got: {:?}", tensor_names);
        assert!(tensor_names.iter().any(|n| *n == "L0.q_proj"),
            "q_proj tensor should use canonical name 'L0.q_proj', got: {:?}", tensor_names);
        assert!(tensor_names.iter().any(|n| *n == "L0.gate_proj"),
            "ffn gate tensor should use canonical name 'L0.gate_proj', got: {:?}", tensor_names);
        assert!(tensor_names.iter().any(|n| *n == "final_norm"),
            "final_norm tensor should use canonical name 'final_norm', got: {:?}", tensor_names);
        assert!(tensor_names.iter().any(|n| *n == "lm_head"),
            "lm_head tensor should use canonical name 'lm_head', got: {:?}", tensor_names);

        // Verify MHA dims
        let mha = graph.ops.iter().find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. })).unwrap();
        if let OpKind::MultiHeadAttention { num_heads, num_kv_heads, head_dim, .. } = mha.kind {
            assert_eq!(num_heads, 4);
            assert_eq!(num_kv_heads, 2);
            assert_eq!(head_dim, 16);
        }
    }

    #[test]
    fn auto_encoder_with_layer_norm() {
        let config = make_config(2, 32, 2, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "roberta.embeddings.word_embeddings.weight"),
            // Layer 0
            (TensorRole::InputNorm, Some(0), "roberta.encoder.layer.0.attention.output.LayerNorm.weight"),
            (TensorRole::AttentionQuery, Some(0), "roberta.encoder.layer.0.attention.self.query.weight"),
            (TensorRole::AttentionKey, Some(0), "roberta.encoder.layer.0.attention.self.key.weight"),
            (TensorRole::AttentionValue, Some(0), "roberta.encoder.layer.0.attention.self.value.weight"),
            (TensorRole::AttentionOutput, Some(0), "roberta.encoder.layer.0.attention.output.dense.weight"),
            (TensorRole::PostAttnNorm, Some(0), "roberta.encoder.layer.0.output.LayerNorm.weight"),
            (TensorRole::FfnUp, Some(0), "roberta.encoder.layer.0.intermediate.dense.weight"),
            (TensorRole::FfnDown, Some(0), "roberta.encoder.layer.0.output.dense.weight"),
            // Layer 1
            (TensorRole::InputNorm, Some(1), "roberta.encoder.layer.1.attention.output.LayerNorm.weight"),
            (TensorRole::AttentionQuery, Some(1), "roberta.encoder.layer.1.attention.self.query.weight"),
            (TensorRole::AttentionKey, Some(1), "roberta.encoder.layer.1.attention.self.key.weight"),
            (TensorRole::AttentionValue, Some(1), "roberta.encoder.layer.1.attention.self.value.weight"),
            (TensorRole::AttentionOutput, Some(1), "roberta.encoder.layer.1.attention.output.dense.weight"),
            (TensorRole::PostAttnNorm, Some(1), "roberta.encoder.layer.1.output.LayerNorm.weight"),
            (TensorRole::FfnUp, Some(1), "roberta.encoder.layer.1.intermediate.dense.weight"),
            (TensorRole::FfnDown, Some(1), "roberta.encoder.layer.1.output.dense.weight"),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("roberta.embeddings.word_embeddings.weight", vec![50, 32]),
            ("roberta.encoder.layer.0.attention.self.query.weight", vec![32, 32]),
            ("roberta.encoder.layer.0.attention.self.key.weight", vec![32, 32]),
            ("roberta.encoder.layer.0.attention.self.value.weight", vec![32, 32]),
            ("roberta.encoder.layer.0.intermediate.dense.weight", vec![64, 32]),
            // BERT LayerNorm has bias — presence triggers LayerNorm detection
            ("roberta.encoder.layer.0.attention.output.LayerNorm.bias", vec![32]),
            ("roberta.encoder.layer.0.output.LayerNorm.bias", vec![32]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None);
        assert_eq!(features.family, Family::Encoder);
        assert_eq!(features.num_layers, 2);
        assert!(!features.has_rope);
        assert_eq!(features.norm_type, NormType::LayerNorm); // Bias detected → LayerNorm
        assert_eq!(features.ffn_type, FfnType::Standard);

        // Canonical-keyed weight_shapes
        let ws = make_weight_shapes(vec![
            ("embed", vec![50, 32]),
            ("L0.input_norm", vec![32]),
            ("L0.input_norm.bias", vec![32]),
            ("L0.q_proj", vec![32, 32]),
            ("L0.k_proj", vec![32, 32]),
            ("L0.v_proj", vec![32, 32]),
            ("L0.o_proj", vec![32, 32]),
            ("L0.post_attn_norm", vec![32]),
            ("L0.post_attn_norm.bias", vec![32]),
            ("L0.up_proj", vec![64, 32]),
            ("L0.down_proj", vec![32, 64]),
            ("L1.input_norm", vec![32]),
            ("L1.input_norm.bias", vec![32]),
            ("L1.q_proj", vec![32, 32]),
            ("L1.k_proj", vec![32, 32]),
            ("L1.v_proj", vec![32, 32]),
            ("L1.o_proj", vec![32, 32]),
            ("L1.post_attn_norm", vec![32]),
            ("L1.post_attn_norm.bias", vec![32]),
            ("L1.up_proj", vec![64, 32]),
            ("L1.down_proj", vec![32, 64]),
        ]);

        let graph = build_compiler_graph(&features, &config, &ws, &std::collections::HashMap::new(), &MegaKernelBusinessConfig::default(), 2048)
            .expect("graph build should succeed");

        // embed_gather(1) + 2 layers × (input_norm + q + k + v + mha + o + resid + post_norm + up + gelu + down + ffn_resid) + meanpool(1)
        // = 1 + 2 × 12 + 1 = 26
        assert_eq!(graph.ops.len(), 26, "encoder should have 26 ops (embed_gather + 2 layers × 12 + meanpool), got {}", graph.ops.len());

        // Verify LayerNorm
        let ln_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::LayerNorm { .. }))
            .count();
        assert_eq!(ln_count, 4, "2 layers × 2 norms = 4 LayerNorm ops");

        // Verify Gelu
        assert!(graph.ops.iter().any(|op| matches!(op.kind, OpKind::Gelu)));
    }

    #[test]
    fn auto_qwen3_with_head_rms_norm() {
        let config = make_config(2, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (TensorRole::InputNorm, Some(0), "model.layers.0.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(0), "model.layers.0.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(0), "model.layers.0.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(0), "model.layers.0.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(0), "model.layers.0.self_attn.o_proj.weight"),
            (TensorRole::AttentionQNorm, Some(0), "model.layers.0.self_attn.q_norm.weight"),
            (TensorRole::AttentionKNorm, Some(0), "model.layers.0.self_attn.k_norm.weight"),
            (TensorRole::PostAttnNorm, Some(0), "model.layers.0.post_attention_layernorm.weight"),
            (TensorRole::FfnGate, Some(0), "model.layers.0.mlp.gate_proj.weight"),
            (TensorRole::FfnUp, Some(0), "model.layers.0.mlp.up_proj.weight"),
            (TensorRole::FfnDown, Some(0), "model.layers.0.mlp.down_proj.weight"),
            // Layer 1
            (TensorRole::InputNorm, Some(1), "model.layers.1.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(1), "model.layers.1.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(1), "model.layers.1.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(1), "model.layers.1.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(1), "model.layers.1.self_attn.o_proj.weight"),
            (TensorRole::AttentionQNorm, Some(1), "model.layers.1.self_attn.q_norm.weight"),
            (TensorRole::AttentionKNorm, Some(1), "model.layers.1.self_attn.k_norm.weight"),
            (TensorRole::PostAttnNorm, Some(1), "model.layers.1.post_attention_layernorm.weight"),
            (TensorRole::FfnGate, Some(1), "model.layers.1.mlp.gate_proj.weight"),
            (TensorRole::FfnUp, Some(1), "model.layers.1.mlp.up_proj.weight"),
            (TensorRole::FfnDown, Some(1), "model.layers.1.mlp.down_proj.weight"),
        ]);

        let ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        let features = analyze_architecture(&ri, &ws_ext, None);
        assert!(features.has_head_rms_norm, "Qwen3 should have head_rms_norm");

        // Canonical-keyed weight_shapes
        let ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.q_norm", vec![16]),
            ("L0.k_norm", vec![16]),
            ("L0.gate_proj", vec![256, 64]),
            ("L0.up_proj", vec![256, 64]),
            ("L0.down_proj", vec![64, 256]),
            ("L1.q_proj", vec![64, 64]),
            ("L1.k_proj", vec![32, 64]),
            ("L1.v_proj", vec![32, 64]),
            ("L1.o_proj", vec![64, 64]),
            ("L1.q_norm", vec![16]),
            ("L1.k_norm", vec![16]),
            ("L1.gate_proj", vec![256, 64]),
            ("L1.up_proj", vec![256, 64]),
            ("L1.down_proj", vec![64, 256]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);

        let graph = build_compiler_graph(&features, &config, &ws, &std::collections::HashMap::new(), &MegaKernelBusinessConfig::default(), 2048)
            .expect("graph build should succeed");

        // 2 layers × (input_norm + q + k + v + q_norm + k_norm + rope_q + rope_k + mha + o + resid + post_norm
        //             + gate + up + swiglu + down + ffn_resid) = 2 × 17
        // + embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        let expected = 2 * 17 + 6;
        assert_eq!(graph.ops.len(), expected, "expected {} ops, got {}: {:?}",
            expected, graph.ops.len(), graph.ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>());

        // Verify HeadRmsNorm ops
        let hrn_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::HeadRmsNorm { .. }))
            .count();
        assert_eq!(hrn_count, 4, "2 layers × 2 (q+k) = 4 HeadRmsNorm ops");
    }

    #[test]
    fn auto_moe_decoder() {
        let config = make_config(1, 64, 4, 2, 16);
        let num_experts = 4;
        let top_k = 2;
        let inter = 128;

        let mut ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            // Layer 0
            (TensorRole::InputNorm, Some(0), "model.layers.0.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(0), "model.layers.0.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(0), "model.layers.0.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(0), "model.layers.0.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(0), "model.layers.0.self_attn.o_proj.weight"),
            (TensorRole::PostAttnNorm, Some(0), "model.layers.0.post_attention_layernorm.weight"),
            (TensorRole::MoEGate, Some(0), "model.layers.0.mlp.gate.weight"),
        ]);

        let mut ws_ext = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate.weight", vec![64, num_experts]),
        ]);
        ws_ext.insert("model.layers.0.mlp.experts.0.gate_proj.weight".to_string(), vec![inter, 64]);

        let features = analyze_architecture(&ri, &ws_ext, None);
        assert_eq!(features.family, Family::Decoder);
        assert_eq!(features.num_layers, 1);
        assert!(features.is_moe);
        assert_eq!(features.ffn_type, FfnType::MoE);
        assert_eq!(features.num_experts, num_experts);
        assert_eq!(features.moe_top_k, top_k);

        // Canonical-keyed weight_shapes
        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("L0.q_proj", vec![64, 64]),
            ("L0.k_proj", vec![32, 64]),
            ("L0.v_proj", vec![32, 64]),
            ("L0.o_proj", vec![64, 64]),
            ("L0.moe_gate", vec![64, num_experts]),
        ]);
        // Expert weights
        for e in 0..num_experts {
            ws.insert(cn_expert(0, e, "gate_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "up_proj"), vec![inter, 64]);
            ws.insert(cn_expert(0, e, "down_proj"), vec![64, inter]);
        }
        ws.insert("final_norm".to_string(), vec![64]);
        ws.insert("lm_head".to_string(), vec![100, 64]);

        let graph = build_compiler_graph(&features, &config, &ws, &std::collections::HashMap::new(), &MegaKernelBusinessConfig::default(), 2048)
            .expect("MoE graph build should succeed");

        // Per-layer ops:
        //   input_norm + q + k + v + rope_q + rope_k + mha + o + resid + post_norm = 10
        //   MoE: moe_gate + topk = 2
        //   Per expert (4): gate_gemm + gate_mask + up_gemm + swiglu + down_gemm + cond_add = 6×4 = 24
        //   moe_resid = 1
        //   Total per-layer = 10 + 2 + 24 + 1 = 37
        // Global: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        let expected = 37 + 6;
        assert_eq!(graph.ops.len(), expected, "expected {} ops, got {}: {:?}",
            expected, graph.ops.len(),
            graph.ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>());

        // Verify MoE-specific ops
        let moe_gate_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::MoEGate { .. }))
            .count();
        assert_eq!(moe_gate_count, 1, "should have 1 MoEGate op");

        let topk_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::TopK { .. }))
            .count();
        assert_eq!(topk_count, 1, "should have 1 TopK op");

        let cond_add_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::MoEConditionalAdd { .. }))
            .count();
        assert_eq!(cond_add_count, num_experts, "should have {} MoEConditionalAdd ops", num_experts);

        let swiglu_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::SwiGlu))
            .count();
        assert_eq!(swiglu_count, num_experts, "should have {} SwiGlu ops (one per expert)", num_experts);
    }

    #[test]
    fn auto_gemma4_qknorm_value_norm_embedding_scale() {
        let config = make_config(1, 64, 4, 2, 16);

        let ri = make_role_index(vec![
            (TensorRole::Embedding, None, "model.embed_tokens.weight"),
            (TensorRole::OutputHead, None, "lm_head.weight"),
            (TensorRole::FinalNorm, None, "model.norm.weight"),
            (TensorRole::InputNorm, Some(0), "model.layers.0.input_layernorm.weight"),
            (TensorRole::AttentionQuery, Some(0), "model.layers.0.self_attn.q_proj.weight"),
            (TensorRole::AttentionKey, Some(0), "model.layers.0.self_attn.k_proj.weight"),
            (TensorRole::AttentionValue, Some(0), "model.layers.0.self_attn.v_proj.weight"),
            (TensorRole::AttentionOutput, Some(0), "model.layers.0.self_attn.o_proj.weight"),
            (TensorRole::PostAttnNorm, Some(0), "model.layers.0.post_attention_layernorm.weight"),
            (TensorRole::FfnGate, Some(0), "model.layers.0.mlp.gate_proj.weight"),
            (TensorRole::FfnUp, Some(0), "model.layers.0.mlp.up_proj.weight"),
            (TensorRole::FfnDown, Some(0), "model.layers.0.mlp.down_proj.weight"),
        ]);

        let ws = make_weight_shapes(vec![
            ("model.embed_tokens.weight", vec![100, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![256, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 256]),
        ]);

        // Without arch name → no Gemma-4 features
        let features_no_arch = analyze_architecture(&ri, &ws, None);
        assert!(!features_no_arch.has_qk_norm, "no arch → no qk_norm");
        assert!(!features_no_arch.has_value_norm, "no arch → no value_norm");
        assert!(!features_no_arch.has_embedding_scale, "no arch → no embedding_scale");

        // With "gemma4" arch → Gemma-4 features enabled
        let features = analyze_architecture(&ri, &ws, Some("gemma4"));
        assert!(features.has_qk_norm, "gemma4 should have qk_norm");
        assert!(features.has_value_norm, "gemma4 should have value_norm");
        assert!(features.has_embedding_scale, "gemma4 should have embedding_scale");
        assert!(!features.has_head_rms_norm, "gemma4 should NOT have head_rms_norm (no weight tensors)");

        // Qwen3 with q_norm weights → HeadRmsNorm, not QkNorm
        let mut ri_qwen3 = ri.clone();
        ri_qwen3.insert((TensorRole::AttentionQNorm, Some(0)), "model.layers.0.self_attn.q_norm.weight".to_string());
        ri_qwen3.insert((TensorRole::AttentionKNorm, Some(0)), "model.layers.0.self_attn.k_norm.weight".to_string());
        let features_qwen3 = analyze_architecture(&ri_qwen3, &ws, Some("qwen3"));
        assert!(features_qwen3.has_head_rms_norm, "qwen3 should have head_rms_norm");
        assert!(!features_qwen3.has_qk_norm, "qwen3 should NOT have qk_norm");
        assert!(!features_qwen3.has_value_norm, "qwen3 should NOT have value_norm");
    }

    /// T43: SharedKvRef graph layer — consumer layers skip K/V GEMMs and
    /// reference donor layer's K/V tensors.
    #[test]
    fn auto_shared_kv_ref_skips_k_v_projections() {
        // 4 layers, last 2 share KV with donors.
        // attention_pattern: [0, 1, 0, 1] — sliding/global alternating.
        // Layer 2 (bucket 0) → donor = layer 0 (bucket 0)
        // Layer 3 (bucket 1) → donor = layer 1 (bucket 1)
        let num_layers = 4;
        let num_shared = 2;
        let mut config = make_config(num_layers, 64, 4, 2, 16);
        config.num_kv_shared_layers = num_shared;
        config.attention_pattern = vec![0, 1, 0, 1];

        let mut ws = make_weight_shapes(vec![
            ("embed", vec![100, 64]),
            ("final_norm", vec![64]),
            ("lm_head", vec![100, 64]),
        ]);
        for i in 0..num_layers {
            ws.insert(cn_layer(i, "input_norm"), vec![64]);
            ws.insert(cn_layer(i, "q_proj"), vec![64, 64]);
            ws.insert(cn_layer(i, "o_proj"), vec![64, 64]);
            ws.insert(cn_layer(i, "post_attn_norm"), vec![64]);
            ws.insert(cn_layer(i, "gate_proj"), vec![256, 64]);
            ws.insert(cn_layer(i, "up_proj"), vec![256, 64]);
            ws.insert(cn_layer(i, "down_proj"), vec![64, 256]);
            // K/V weights only for non-shared layers (donors)
            if i < num_layers - num_shared {
                ws.insert(cn_layer(i, "k_proj"), vec![32, 64]);
                ws.insert(cn_layer(i, "v_proj"), vec![32, 64]);
            }
        }

        let features = ArchitectureFeatures {
            family: Family::Decoder,
            num_layers,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_embedding_scale: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_vision: false,
            is_audio: false,
            has_classifier: false,
            tie_lm_head: false,
        };

        let graph = build_compiler_graph(
            &features, &config, &ws,
            &std::collections::HashMap::new(),
            &MegaKernelBusinessConfig::default(),
            2048,
        ).expect("graph build should succeed");

        // Verify op labels — donor layers (0, 1) have k_proj/v_proj, consumer
        // layers (2, 3) do not.
        let op_labels: Vec<&str> = graph.ops.iter().map(|o| o.label.as_str()).collect();

        // Donor layers should have k_proj and v_proj
        for i in 0..2 {
            let k_label = format!("L{}_k_proj", i);
            let v_label = format!("L{}_v_proj", i);
            assert!(op_labels.iter().any(|l| *l == k_label),
                "donor layer {i} should have k_proj");
            assert!(op_labels.iter().any(|l| *l == v_label),
                "donor layer {i} should have v_proj");
        }

        // Consumer layers should NOT have k_proj or v_proj
        for i in 2..4 {
            let k_label = format!("L{}_k_proj", i);
            let v_label = format!("L{}_v_proj", i);
            assert!(!op_labels.iter().any(|l| *l == k_label),
                "consumer layer {i} should NOT have k_proj");
            assert!(!op_labels.iter().any(|l| *l == v_label),
                "consumer layer {i} should NOT have v_proj");
        }

        // All layers should have q_proj (always computed)
        for i in 0..4 {
            let q_label = format!("L{}_q_proj", i);
            assert!(op_labels.iter().any(|l| *l == q_label),
                "layer {i} should have q_proj");
        }

        // All layers should have MHA (attention still runs with donor K/V)
        for i in 0..4 {
            let mha_label = format!("L{}_mha", i);
            assert!(op_labels.iter().any(|l| *l == mha_label),
                "layer {i} should have MHA");
        }

        // Donor layers have rope_k, consumer layers do not
        for i in 0..2 {
            let rope_k_label = format!("L{}_rope_k", i);
            assert!(op_labels.iter().any(|l| *l == rope_k_label),
                "donor layer {i} should have rope_k");
        }
        for i in 2..4 {
            let rope_k_label = format!("L{}_rope_k", i);
            assert!(!op_labels.iter().any(|l| *l == rope_k_label),
                "consumer layer {i} should NOT have rope_k");
        }

        // Count ops:
        // Donor layers (0,1): input_norm + q + k + v + rope_q + rope_k + mha + o + resid + post_norm
        //   + gate + up + swiglu + down + ffn_resid = 15 each
        // Consumer layers (2,3): input_norm + q + rope_q + mha + o + resid + post_norm
        //   + gate + up + swiglu + down + ffn_resid = 12 each (no k, v, rope_k)
        // Global: embed_gather + final_norm + lm_head + argmax + store_token + check_stop = 6
        let expected = 2 * 15 + 2 * 12 + 6;
        assert_eq!(graph.ops.len(), expected,
            "expected {} ops, got {}: {:?}",
            expected, graph.ops.len(),
            graph.ops.iter().map(|o| o.label.clone()).collect::<Vec<_>>());
    }

}
