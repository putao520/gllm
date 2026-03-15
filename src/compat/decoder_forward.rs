
use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::jit_helpers::{
    build_decoder_layer_graph, build_final_norm_graph, build_kv_projection_graph,
    build_lm_head_graph, execute_jit_decoder_layer, execute_jit_final_norm, execute_jit_lm_head,
    execute_kv_projection, update_kv_cache, write_kv_to_cache,
};
use super::scalar_ops::{
    cached_gqa_attention, scalar_gemm, scalar_moe_ffn, scalar_rms_norm, scalar_rope, swiglu_ffn,
};
use super::types::{AttentionGeometry, KvCacheSlice, LayerDims, SeqContext};
use super::weight_helpers::{
    get_f32_data, get_weight_data, needs_weight_transpose, quantized_linear, transpose_f32,
    try_get_f32_data, weight_data_to_f32, WeightData,
};
use super::Element;
use crate::engine::executor::{
    BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheHandle, LogitsHandle,
};

// ---------------------------------------------------------------------------
// Scalar MoE-aware prefill layer
// ---------------------------------------------------------------------------

/// Scalar MoE-aware decoder layer (full prefill, no KV cache).
///
/// Same structure as the JIT decoder layer but replaces the dense SwiGLU FFN
/// with MoE routing (gate -> top-k -> expert dispatch -> combine).
fn scalar_moe_prefill_layer(
    hidden_state: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    o_w: &[f32],
    rn1_w: &[f32],
    rn2_w: &[f32],
    router_w: &[f32],
    expert_weights: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    shared_expert: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
    eps: f32,
    rope_theta: f64,
    output: &mut [f32],
) {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads;

    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    let mut q_proj = vec![0.0f32; seq_len * q_dim];
    scalar_gemm(&normed, q_w, &mut q_proj, seq_len, q_dim, hidden);
    let mut k_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, k_w, &mut k_proj, seq_len, kv_dim, hidden);
    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, v_w, &mut v_proj, seq_len, kv_dim, hidden);

    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);
    scalar_rope(&mut k_proj, positions, head_dim, rope_theta);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];
    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        for s in 0..seq_len {
            let q_off = s * q_dim + h * head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for (t, score) in scores.iter_mut().enumerate().take(s + 1) {
                let k_off = t * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_proj[q_off + d] * k_proj[k_off + d]; }
                *score = dot * scale;
            }
            let max_s = scores[..=s].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(s + 1) { *score = (*score - max_s).exp(); sum += *score; }
            if sum > 0.0 { for score in scores.iter_mut().take(s + 1) { *score /= sum; } }
            let o_off = s * q_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..=s { val += scores[t] * v_proj[t * kv_dim + kv_h * head_dim + d]; }
                attn_out[o_off + d] = val;
            }
        }
    }

    let mut o_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&attn_out, o_w, &mut o_out, seq_len, hidden, q_dim);
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden { resid1[i] = hidden_state[i] + o_out[i]; }

    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);

    let moe_out = scalar_moe_ffn(
        &normed2, router_w, expert_weights, shared_expert,
        seq_len, hidden, inter, num_experts, top_k,
    );

    for i in 0..seq_len * hidden { output[i] = resid1[i] + moe_out[i]; }
}

// ---------------------------------------------------------------------------
// Scalar incremental decode layer (uses cached K/V, O(n) per step)
// ---------------------------------------------------------------------------

/// Execute a single decoder layer using cached K/V for attention.
///
/// For incremental decode (position > 0), this avoids recomputing all K/V
/// from scratch. Instead, it:
/// 1. Computes Q for the new token(s) only
/// 2. Computes new K/V and appends to cache (done by update_kv_cache)
/// 3. Runs attention using full cached K/V sequence
/// 4. Runs FFN (SwiGLU) on the attention output
///
/// This is O(total_seq * head_dim) per step instead of O(total_seq^2).
#[allow(dead_code)] // Superseded by quantized_incremental_decode_layer; kept as reference
fn scalar_incremental_decode_layer(
    hidden_state: &[f32],
    q_w: &[f32],
    o_w: &[f32],
    rn1_w: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    rn2_w: &[f32],
    positions: &[u32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    layer: usize,
    total_seq: usize,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
    max_seq_len: usize,
    output: &mut [f32],
) {
    let geom = AttentionGeometry {
        num_heads, num_kv_heads, head_dim,
        q_dim: num_heads * head_dim,
        kv_dim: num_kv_heads * head_dim,
        heads_per_group: num_heads / num_kv_heads,
    };
    let dims = LayerDims { hidden, inter, eps, rope_theta };
    let seq = SeqContext { positions, seq_len, total_seq };
    let kv = KvCacheSlice { k: kv_cache_k, v: kv_cache_v, layer, max_seq_len };

    // RMSNorm + Q projection + RoPE
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);
    let mut q_proj = vec![0.0f32; seq_len * geom.q_dim];
    scalar_gemm(&normed, q_w, &mut q_proj, seq_len, geom.q_dim, hidden);
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Cached GQA attention (replaces ~55 lines of duplicated attention code)
    let attn_out = cached_gqa_attention(&q_proj, &kv, &seq, &geom);

    // O projection + residual
    let mut o_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&attn_out, o_w, &mut o_out, seq_len, hidden, geom.q_dim);
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden { resid1[i] = hidden_state[i] + o_out[i]; }

    // Pre-FFN RMSNorm + SwiGLU FFN (replaces ~20 lines of duplicated FFN code)
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);
    let down_out = swiglu_ffn(&normed2, gate_w, up_w, down_w, &dims, seq_len);

    for i in 0..seq_len * hidden { output[i] = resid1[i] + down_out[i]; }
}

/// Execute a single decoder layer using cached K/V, with quantized matmul acceleration.
///
/// Same logic as `scalar_incremental_decode_layer` but dispatches GEMM operations
/// through `quantized_linear` which uses `quantized_matmul` for quantized weights,
/// avoiding the expensive dequantize + transpose + scalar_gemm path.
fn quantized_incremental_decode_layer<E: Element>(
    backend: &CpuBackend<E>,
    hidden_state: &[f32],
    q_w: &WeightData,
    o_w: &WeightData,
    rn1_w: &[f32],
    gate_w: &WeightData,
    up_w: &WeightData,
    down_w: &WeightData,
    rn2_w: &[f32],
    positions: &[u32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    layer: usize,
    total_seq: usize,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
    max_seq_len: usize,
    transpose_weights: bool,
    output: &mut [f32],
) -> Result<(), BE> {
    let geom = AttentionGeometry {
        num_heads, num_kv_heads, head_dim,
        q_dim: num_heads * head_dim,
        kv_dim: num_kv_heads * head_dim,
        heads_per_group: num_heads / num_kv_heads,
    };
    let seq = SeqContext { positions, seq_len, total_seq };
    let kv = KvCacheSlice { k: kv_cache_k, v: kv_cache_v, layer, max_seq_len };

    // RMSNorm + Q projection (quantized) + RoPE
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);
    let mut q_proj = vec![0.0f32; seq_len * geom.q_dim];
    quantized_linear(backend, &normed, q_w, &mut q_proj, seq_len, geom.q_dim, hidden, transpose_weights)?;
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Cached GQA attention
    let attn_out = cached_gqa_attention(&q_proj, &kv, &seq, &geom);

    // O projection (quantized) + residual
    let mut o_out = vec![0.0f32; seq_len * hidden];
    quantized_linear(backend, &attn_out, o_w, &mut o_out, seq_len, hidden, geom.q_dim, transpose_weights)?;
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden { resid1[i] = hidden_state[i] + o_out[i]; }

    // Pre-FFN RMSNorm + SwiGLU FFN (quantized)
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);
    let mut gate_out = vec![0.0f32; seq_len * inter];
    quantized_linear(backend, &normed2, gate_w, &mut gate_out, seq_len, inter, hidden, transpose_weights)?;
    let mut up_out = vec![0.0f32; seq_len * inter];
    quantized_linear(backend, &normed2, up_w, &mut up_out, seq_len, inter, hidden, transpose_weights)?;
    let mut swiglu = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        swiglu[i] = silu_g * up_out[i];
    }
    let mut down_out = vec![0.0f32; seq_len * hidden];
    quantized_linear(backend, &swiglu, down_w, &mut down_out, seq_len, hidden, inter, transpose_weights)?;

    for i in 0..seq_len * hidden { output[i] = resid1[i] + down_out[i]; }
    Ok(())
}

/// Execute a single MoE decoder layer using cached K/V for attention.
///
/// Same as `scalar_incremental_decode_layer` but replaces the SwiGLU FFN
/// with MoE routing: gate → top-k selection → expert FFN → weighted combine.
fn scalar_incremental_moe_decode_layer(
    hidden_state: &[f32],
    q_w: &[f32],
    o_w: &[f32],
    rn1_w: &[f32],
    rn2_w: &[f32],
    router_w: &[f32],
    expert_weights: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    shared_expert: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    positions: &[u32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    layer: usize,
    total_seq: usize,
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
    max_seq_len: usize,
    num_experts: usize,
    top_k: usize,
    output: &mut [f32],
) {
    let geom = AttentionGeometry {
        num_heads, num_kv_heads, head_dim,
        q_dim: num_heads * head_dim,
        kv_dim: num_kv_heads * head_dim,
        heads_per_group: num_heads / num_kv_heads,
    };
    let seq = SeqContext { positions, seq_len, total_seq };
    let kv = KvCacheSlice { k: kv_cache_k, v: kv_cache_v, layer, max_seq_len };

    // RMSNorm + Q projection + RoPE
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);
    let mut q_proj = vec![0.0f32; seq_len * geom.q_dim];
    scalar_gemm(&normed, q_w, &mut q_proj, seq_len, geom.q_dim, hidden);
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Cached GQA attention
    let attn_out = cached_gqa_attention(&q_proj, &kv, &seq, &geom);

    // O projection + residual
    let mut o_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&attn_out, o_w, &mut o_out, seq_len, hidden, geom.q_dim);
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden { resid1[i] = hidden_state[i] + o_out[i]; }

    // MoE FFN
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);
    let moe_out = scalar_moe_ffn(
        &normed2, router_w, expert_weights, shared_expert,
        seq_len, hidden, inter, num_experts, top_k,
    );

    for i in 0..seq_len * hidden { output[i] = resid1[i] + moe_out[i]; }
}

// ---------------------------------------------------------------------------
// MoE weight loading helpers
// ---------------------------------------------------------------------------

/// Load all routed expert weights for a given layer.
/// Returns a Vec of (gate_proj, up_proj, down_proj) tuples, one per expert.
/// If the router gate weight is not found, returns None (dense layer).
fn load_moe_weights<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    backend: &CpuBackend<E>,
    layer: usize,
    num_experts: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<
    Option<(
        Vec<f32>,                              // router gate weight
        Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,   // expert (gate, up, down)
        Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, // shared expert (gate, up, down)
    )>,
    BE,
> {
    // Try to load router gate weight; if absent, this is a dense layer
    let router_w = match try_get_f32_data(
        weights, backend,
        &crate::weight_names::moe_gate_aliases(layer),
    ) {
        Some(w) => w,
        None => return Ok(None),
    };

    let router_w = if transpose {
        transpose_f32(&router_w, num_experts, hidden)
    } else {
        router_w
    };

    // Load each routed expert's FFN weights
    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gw = get_f32_data(weights, backend,
            &crate::weight_names::moe_expert_aliases(layer, e, "gate_proj.weight"))?;
        let uw = get_f32_data(weights, backend,
            &crate::weight_names::moe_expert_aliases(layer, e, "up_proj.weight"))?;
        let dw = get_f32_data(weights, backend,
            &crate::weight_names::moe_expert_aliases(layer, e, "down_proj.weight"))?;

        let (gw, uw, dw) = if transpose {
            (
                transpose_f32(&gw, inter, hidden),
                transpose_f32(&uw, inter, hidden),
                transpose_f32(&dw, hidden, inter),
            )
        } else {
            (gw, uw, dw)
        };
        experts.push((gw, uw, dw));
    }

    // Try to load shared expert weights (DeepSeek-style); optional
    let shared = {
        let sg = try_get_f32_data(weights, backend,
            &crate::weight_names::moe_shared_expert_aliases(layer, "gate_proj.weight"));
        let su = try_get_f32_data(weights, backend,
            &crate::weight_names::moe_shared_expert_aliases(layer, "up_proj.weight"));
        let sd = try_get_f32_data(weights, backend,
            &crate::weight_names::moe_shared_expert_aliases(layer, "down_proj.weight"));
        match (sg, su, sd) {
            (Some(g), Some(u), Some(d)) => {
                let (g, u, d) = if transpose {
                    (
                        transpose_f32(&g, inter, hidden),
                        transpose_f32(&u, inter, hidden),
                        transpose_f32(&d, hidden, inter),
                    )
                } else {
                    (g, u, d)
                };
                Some((g, u, d))
            }
            _ => None,
        }
    };

    Ok(Some((router_w, experts, shared)))
}

// ---------------------------------------------------------------------------
// Full decoder forward pass
// ---------------------------------------------------------------------------

/// Full decoder forward pass for a single sequence.
///
/// Pipeline:
/// 1. Token embedding lookup
/// 2. For each layer: JIT-compiled decoder layer + KV cache update
/// 3. Final RMSNorm + lm_head projection → logits
///
/// Returns logits for the last token position only (for generation).
pub(crate) fn decoder_forward<E: Element>(
    backend: &CpuBackend<E>,
    input: &BatchInput,
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
) -> Result<Vec<LogitsHandle>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder forward only supports f32 element type".into()));
    }

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;
    let rope_theta = config.rope_theta;
    let transpose_weights = needs_weight_transpose(weights);

    // MoE configuration
    let moe_cfg = config.moe_config.as_ref();
    let moe_num_experts = moe_cfg.map(|c| c.num_experts).unwrap_or(0);
    let moe_top_k = moe_cfg.map(|c| c.num_experts_per_tok).unwrap_or(0);

    let mut results = Vec::with_capacity(input.sequences.len());

    for (seq_idx, seq) in input.sequences.iter().enumerate() {
        let tokens = &seq.tokens;
        let position = seq.position;
        let seq_len = tokens.len();

        if seq_len == 0 {
            return Err(BE::Other("empty sequence in decoder forward".into()));
        }

        // (a) Token embedding lookup
        let embed_data = get_f32_data(
            weights, backend,
            &crate::weight_names::decoder_embed_aliases(),
        )?;

        let embed_vocab = embed_data.len() / hidden;
        let mut hidden_state = vec![0.0f32; seq_len * hidden];
        for (s, &tok) in tokens.iter().enumerate() {
            let v = tok as usize;
            if v >= embed_vocab {
                return Err(BE::Other(format!(
                    "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
                )));
            }
            hidden_state[s * hidden..(s + 1) * hidden]
                .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
        }

        // (b) Build position array
        let positions: Vec<u32> = (0..seq_len).map(|i| (position + i) as u32).collect();

        // (c) Determine if this is an incremental decode step (position > 0 with KV cache)
        let has_kv_cache = seq_idx < kv_caches.len();
        let cached_seq_len = if has_kv_cache {
            let store = backend.kv_store().lock().map_err(|e| {
                BE::Cpu(format!("KV store lock poisoned: {e}"))
            })?;
            store.get(&kv_caches[seq_idx].0).map(|b| b.seq_len).unwrap_or(0)
        } else {
            0
        };
        let is_incremental = has_kv_cache && cached_seq_len > 0 && position > 0;

        let kv_dim = num_kv_heads * head_dim;

        if is_incremental {
            // ── Incremental decode path: use cached K/V, O(n) per step ──
            log::debug!(
                "decoder_forward: incremental decode, position={position}, seq_len={seq_len}, cached_seq={cached_seq_len}"
            );

            let max_seq_len = {
                let store = backend.kv_store().lock().map_err(|e| {
                    BE::Cpu(format!("KV store lock poisoned: {e}"))
                })?;
                store.get(&kv_caches[seq_idx].0).map(|b| b.max_seq_len).unwrap_or(0)
            };

            // Compile KV projection graph for incremental decode (seq_len tokens)
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let kv_proj_decode: gllm_kernels::compiler::CompiledLayer = {
                let kv_graph = build_kv_projection_graph(
                    seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                );
                let mut kv_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                kv_compiler.compile_graph(&kv_graph).map_err(|e| {
                    BE::Other(format!("KV projection (decode) JIT compilation failed: {e}"))
                })?
            };

            for layer in 0..num_layers {
                // Load attention + norm weights
                let q_w = get_weight_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")))?;
                let k_w = get_weight_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")))?;
                let v_w = get_weight_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")))?;
                let o_w = get_weight_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")))?;
                let rn1_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")))?;
                let rn2_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")))?;

                // K/V weights as f32 for KV projection
                let k_w_f32 = weight_data_to_f32(&k_w, backend, transpose_weights, kv_dim, hidden)?;
                let v_w_f32 = weight_data_to_f32(&v_w, backend, transpose_weights, kv_dim, hidden)?;

                // Check if this layer uses MoE
                let moe_weights = if moe_num_experts > 0 {
                    load_moe_weights(
                        weights, backend, layer,
                        moe_num_experts, hidden, inter, transpose_weights,
                    )?
                } else {
                    None
                };

                if moe_weights.is_some() {
                    // MoE layers: use scalar update_kv_cache (no JIT optimization)
                    update_kv_cache(
                        backend, kv_caches[seq_idx],
                        layer, &hidden_state, &k_w_f32, &v_w_f32,
                        &rn1_w, &positions,
                        seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                    )?;
                } else {
                    // Dense layers: JIT KV projection
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let (k_rope, v_proj) = execute_kv_projection(
                            &kv_proj_decode,
                            &hidden_state, &rn1_w, &k_w_f32, &v_w_f32,
                            &positions, seq_len, hidden, num_kv_heads, head_dim, eps,
                        );
                        write_kv_to_cache(
                            backend, kv_caches[seq_idx],
                            layer, &k_rope, &v_proj,
                            seq_len, num_kv_heads, head_dim,
                        )?;
                    }
                }

                // Read cached K/V for attention
                let total_seq = cached_seq_len + seq_len;
                let (kv_cache_k, kv_cache_v) = {
                    let store = backend.kv_store().lock().map_err(|e| {
                        BE::Cpu(format!("KV store lock poisoned: {e}"))
                    })?;
                    let buffer = store.get(&kv_caches[seq_idx].0).ok_or_else(|| {
                        BE::Cpu(format!("KV cache handle {} not found", kv_caches[seq_idx].0))
                    })?;
                    (buffer.k.clone(), buffer.v.clone())
                };

                let mut layer_out = vec![0.0f32; seq_len * hidden];

                if let Some((router_w, expert_weights, shared_expert)) = moe_weights {
                    // MoE incremental: dequantize attention weights to f32
                    let q_w_f32 = weight_data_to_f32(
                        &q_w, backend, transpose_weights, num_heads * head_dim, hidden)?;
                    let o_w_f32 = weight_data_to_f32(
                        &o_w, backend, transpose_weights, hidden, num_heads * head_dim)?;

                    scalar_incremental_moe_decode_layer(
                        &hidden_state,
                        &q_w_f32, &o_w_f32, &rn1_w, &rn2_w,
                        &router_w, &expert_weights,
                        shared_expert.as_ref(),
                        &positions,
                        &kv_cache_k, &kv_cache_v,
                        layer, total_seq, seq_len,
                        hidden, num_heads, num_kv_heads, head_dim, inter,
                        eps, rope_theta, max_seq_len,
                        moe_num_experts, moe_top_k,
                        &mut layer_out,
                    );
                } else {
                    // Dense layer: load standard FFN weights
                    let gate_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")))?;
                    let up_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")))?;
                    let down_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")))?;

                    quantized_incremental_decode_layer(
                        backend,
                        &hidden_state,
                        &q_w, &o_w, &rn1_w,
                        &gate_w, &up_w, &down_w, &rn2_w,
                        &positions,
                        &kv_cache_k, &kv_cache_v,
                        layer, total_seq, seq_len,
                        hidden, num_heads, num_kv_heads, head_dim, inter,
                        eps, rope_theta, max_seq_len,
                        transpose_weights,
                        &mut layer_out,
                    )?;
                }

                hidden_state.copy_from_slice(&layer_out);
            }
        } else if moe_num_experts > 0 {
            // ── MoE Prefill path: scalar execution with expert routing ──

            for layer in 0..num_layers {
                let q_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")))?;
                let k_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")))?;
                let v_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")))?;
                let o_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")))?;
                let rn1_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")))?;
                let rn2_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")))?;

                let (q_w, k_w, v_w, o_w) = if transpose_weights {
                    (
                        transpose_f32(&q_w, num_heads * head_dim, hidden),
                        transpose_f32(&k_w, kv_dim, hidden),
                        transpose_f32(&v_w, kv_dim, hidden),
                        transpose_f32(&o_w, hidden, num_heads * head_dim),
                    )
                } else {
                    (q_w, k_w, v_w, o_w)
                };

                // Load MoE router weight
                let router_w = get_f32_data(weights, backend,
                    &crate::weight_names::moe_gate_aliases(layer))?;
                let router_w = if transpose_weights {
                    transpose_f32(&router_w, moe_num_experts, hidden)
                } else {
                    router_w
                };

                // Load per-expert weights
                let mut expert_weights = Vec::with_capacity(moe_num_experts);
                for e in 0..moe_num_experts {
                    let ew_gate = get_f32_data(weights, backend,
                        &crate::weight_names::moe_expert_aliases(layer, e, "gate_proj.weight"))?;
                    let ew_up = get_f32_data(weights, backend,
                        &crate::weight_names::moe_expert_aliases(layer, e, "up_proj.weight"))?;
                    let ew_down = get_f32_data(weights, backend,
                        &crate::weight_names::moe_expert_aliases(layer, e, "down_proj.weight"))?;
                    let (ew_gate, ew_up, ew_down) = if transpose_weights {
                        (
                            transpose_f32(&ew_gate, inter, hidden),
                            transpose_f32(&ew_up, inter, hidden),
                            transpose_f32(&ew_down, hidden, inter),
                        )
                    } else {
                        (ew_gate, ew_up, ew_down)
                    };
                    expert_weights.push((ew_gate, ew_up, ew_down));
                }

                // Load shared expert weights (optional, e.g. DeepSeek)
                let shared_expert = {
                    let sg = get_f32_data(weights, backend,
                        &crate::weight_names::moe_shared_expert_aliases(layer, "gate_proj.weight"));
                    let su = get_f32_data(weights, backend,
                        &crate::weight_names::moe_shared_expert_aliases(layer, "up_proj.weight"));
                    let sd = get_f32_data(weights, backend,
                        &crate::weight_names::moe_shared_expert_aliases(layer, "down_proj.weight"));
                    match (sg, su, sd) {
                        (Ok(sg), Ok(su), Ok(sd)) if !sg.is_empty() && !su.is_empty() && !sd.is_empty() => {
                            let (sg, su, sd) = if transpose_weights {
                                (
                                    transpose_f32(&sg, inter, hidden),
                                    transpose_f32(&su, inter, hidden),
                                    transpose_f32(&sd, hidden, inter),
                                )
                            } else {
                                (sg, su, sd)
                            };
                            Some((sg, su, sd))
                        }
                        _ => None,
                    }
                };

                let mut layer_out = vec![0.0f32; seq_len * hidden];
                scalar_moe_prefill_layer(
                    &hidden_state,
                    &q_w, &k_w, &v_w, &o_w,
                    &rn1_w, &rn2_w,
                    &router_w, &expert_weights,
                    shared_expert.as_ref(),
                    &positions,
                    seq_len, hidden, num_heads, num_kv_heads, head_dim,
                    inter, moe_num_experts, moe_top_k, eps, rope_theta,
                    &mut layer_out,
                );

                // Update KV cache for this layer
                if has_kv_cache {
                    update_kv_cache(
                        backend, kv_caches[seq_idx],
                        layer, &hidden_state, &k_w, &v_w,
                        &rn1_w, &positions,
                        seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                    )?;
                }

                hidden_state.copy_from_slice(&layer_out);
            }
        } else {
            // ── Dense Prefill path: JIT-compiled full sequence ──

            // (c) JIT compile decoder layer graph (once, reused across layers)
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let jit_layer: gllm_kernels::compiler::CompiledLayer = {
                let graph = build_decoder_layer_graph(
                    seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta,
                );
                let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                compiler.compile_graph(&graph).map_err(|e| {
                    BE::Other(format!("Decoder layer JIT compilation failed: {e}"))
                })?
            };

            // Compile KV projection graph for prefill (reused across layers)
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let kv_proj_compiled: Option<gllm_kernels::compiler::CompiledLayer> = if has_kv_cache {
                let kv_graph = build_kv_projection_graph(
                    seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                );
                let mut kv_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                Some(kv_compiler.compile_graph(&kv_graph).map_err(|e| {
                    BE::Other(format!("KV projection JIT compilation failed: {e}"))
                })?)
            } else {
                None
            };

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            return Err(BE::Other("Decoder forward requires JIT compilation (x86_64 or aarch64)".into()));

            // (d) Run through decoder layers
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            for layer in 0..num_layers {
                let q_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")))?;
                let k_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")))?;
                let v_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")))?;
                let o_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")))?;
                let rn1_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")))?;
                let gate_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")))?;
                let up_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")))?;
                let down_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")))?;
                let rn2_w = get_f32_data(weights, backend,
                    &crate::weight_names::decoder_layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")))?;

                let (q_w, k_w, v_w, o_w, gate_w, up_w, down_w) = if transpose_weights {
                    (
                        transpose_f32(&q_w, num_heads * head_dim, hidden),
                        transpose_f32(&k_w, kv_dim, hidden),
                        transpose_f32(&v_w, kv_dim, hidden),
                        transpose_f32(&o_w, hidden, num_heads * head_dim),
                        transpose_f32(&gate_w, inter, hidden),
                        transpose_f32(&up_w, inter, hidden),
                        transpose_f32(&down_w, hidden, inter),
                    )
                } else {
                    (q_w, k_w, v_w, o_w, gate_w, up_w, down_w)
                };

                // KV projection: JIT K (RmsNorm→Gemm→RoPE) + scalar V, then write to cache
                if let Some(ref kv_compiled) = kv_proj_compiled {
                    let (k_rope, v_proj) = execute_kv_projection(
                        kv_compiled,
                        &hidden_state, &rn1_w, &k_w, &v_w,
                        &positions, seq_len, hidden, num_kv_heads, head_dim, eps,
                    );
                    write_kv_to_cache(
                        backend, kv_caches[seq_idx],
                        layer, &k_rope, &v_proj,
                        seq_len, num_kv_heads, head_dim,
                    )?;
                }

                // Execute JIT-compiled layer
                let mut layer_out = vec![0.0f32; seq_len * hidden];
                execute_jit_decoder_layer(
                    &jit_layer,
                    &hidden_state,
                    &q_w, &k_w, &v_w, &o_w, &rn1_w,
                    &gate_w, &up_w, &down_w, &rn2_w,
                    &positions,
                    seq_len,
                    &mut layer_out,
                );

                hidden_state.copy_from_slice(&layer_out);
            }
        }

        // (e) Final RMSNorm + lm_head
        let final_norm_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_final_norm_aliases())?;

        let lm_head_w = get_f32_data(weights, backend,
            &crate::weight_names::lm_head_aliases())?;

        // If lm_head weight not found, try tied embeddings (embed_tokens.weight)
        let lm_head_w = if lm_head_w.is_empty() {
            get_f32_data(weights, backend, &crate::weight_names::decoder_embed_aliases())?
        } else {
            lm_head_w
        };

        let lm_head_w = if transpose_weights {
            transpose_f32(&lm_head_w, vocab_size, hidden)
        } else {
            lm_head_w
        };

        // JIT compile and execute lm_head
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let logits = {
            let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size, eps);
            let mut lm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
            let compiled_lm = lm_compiler.compile_graph(&lm_graph).map_err(|e| {
                BE::Other(format!("lm_head JIT compilation failed: {e}"))
            })?;

            let mut all_logits = vec![0.0f32; seq_len * vocab_size];
            execute_jit_lm_head(
                &compiled_lm,
                &hidden_state,
                &final_norm_w,
                &lm_head_w,
                seq_len,
                &mut all_logits,
            );

            // Return only the last token's logits (for generation)
            let last_start = (seq_len - 1) * vocab_size;
            all_logits[last_start..last_start + vocab_size].to_vec()
        };

        results.push(LogitsHandle { data: logits });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Decoder-based Embedding Forward (for Qwen3-Embedding, etc.)
// ---------------------------------------------------------------------------

/// Decoder-based embedding forward pass (for models like Qwen3-Embedding that
/// use decoder architecture with RoPE instead of BERT-style absolute position embeddings).
///
/// Flow:
/// 1. Token embedding lookup (embed_tokens.weight)
/// 2. Per-layer JIT decoder execution (no KV cache)
/// 3. Final RMSNorm
/// 4. Mean pooling → output vector
pub(crate) fn decoder_embedding_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder embedding forward only supports f32".into()));
    }

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let rope_theta = config.rope_theta;
    let transpose_weights = needs_weight_transpose(weights);
    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder embedding".into()));
    }

    // (a) Token embedding lookup
    let embed_data = get_f32_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    // (b) Positions: 0..seq_len (single-pass, no incremental decoding)
    let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

    // (c) JIT compile decoder layer graph
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let jit_layer: gllm_kernels::compiler::CompiledLayer = {
        let graph = build_decoder_layer_graph(
            seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta,
        );
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| {
            BE::Other(format!("Decoder embedding layer JIT compilation failed: {e}"))
        })?
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(BE::Other("Decoder embedding forward requires JIT compilation (x86_64 or aarch64)".into()));

    let kv_dim = num_kv_heads * head_dim;

    // (d) Run through all decoder layers (no KV cache for embedding)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    for layer in 0..num_layers {
        let q_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")))?;
        let k_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")))?;
        let v_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")))?;
        let o_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")))?;
        let rn1_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")))?;
        let gate_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")))?;
        let up_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")))?;
        let down_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")))?;
        let rn2_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")))?;

        let q_dim = num_heads * head_dim;
        let (q_w, k_w, v_w, o_w, gate_w, up_w, down_w) = if transpose_weights {
            (
                transpose_f32(&q_w, q_dim, hidden),
                transpose_f32(&k_w, kv_dim, hidden),
                transpose_f32(&v_w, kv_dim, hidden),
                transpose_f32(&o_w, hidden, q_dim),
                transpose_f32(&gate_w, inter, hidden),
                transpose_f32(&up_w, inter, hidden),
                transpose_f32(&down_w, hidden, inter),
            )
        } else {
            (q_w, k_w, v_w, o_w, gate_w, up_w, down_w)
        };

        let mut layer_out = vec![0.0f32; seq_len * hidden];
        execute_jit_decoder_layer(
            &jit_layer,
            &hidden_state,
            &q_w, &k_w, &v_w, &o_w, &rn1_w,
            &gate_w, &up_w, &down_w, &rn2_w,
            &positions,
            seq_len,
            &mut layer_out,
        );

        hidden_state.copy_from_slice(&layer_out);
    }

    // (e) Final RMSNorm
    let final_norm_w = get_f32_data(weights, backend,
        &crate::weight_names::decoder_final_norm_aliases())?;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let normed = {
        let norm_graph = build_final_norm_graph(seq_len, hidden, eps);
        let mut norm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled_norm = norm_compiler.compile_graph(&norm_graph).map_err(|e| {
            BE::Other(format!("Final norm JIT compilation failed: {e}"))
        })?;

        let mut normed_out = vec![0.0f32; seq_len * hidden];
        execute_jit_final_norm(
            &compiled_norm,
            &hidden_state,
            &final_norm_w,
            seq_len,
            &mut normed_out,
        );
        normed_out
    };

    // (f) Mean pooling: average across all token positions
    let mut pooled = vec![0.0f32; hidden];
    for s in 0..seq_len {
        for (d, p) in pooled.iter_mut().enumerate() {
            *p += normed[s * hidden + d];
        }
    }
    let scale = 1.0 / seq_len as f32;
    for p in pooled.iter_mut() {
        *p *= scale;
    }

    // (g) L2 normalize (standard for embedding models)
    let l2_norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if l2_norm > 1e-12 {
        for p in pooled.iter_mut() {
            *p /= l2_norm;
        }
    }

    Ok(pooled)
}

/// Decoder-based reranker forward pass (for models like Qwen3-Reranker that
/// use decoder architecture with a score/classifier head).
///
/// Flow:
/// 1. Token embedding lookup (embed_tokens.weight)
/// 2. Per-layer JIT decoder execution (no KV cache)
/// 3. Final RMSNorm
/// 4. Last token hidden state → score head → sigmoid → relevance score
pub(crate) fn decoder_rerank_forward<E: Element>(
    backend: &CpuBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CpuBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder rerank forward only supports f32".into()));
    }

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let rope_theta = config.rope_theta;
    let transpose_weights = needs_weight_transpose(weights);
    let seq_len = tokens.len();

    if seq_len == 0 {
        return Err(BE::Other("empty token sequence for decoder rerank".into()));
    }

    // (a) Token embedding lookup
    let embed_data = get_f32_data(
        weights, backend,
        &crate::weight_names::decoder_embed_aliases(),
    )?;

    let embed_vocab = embed_data.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= embed_vocab {
            return Err(BE::Other(format!(
                "token id {} out of range for embed_tokens (vocab {})", tok, embed_vocab
            )));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&embed_data[v * hidden..(v + 1) * hidden]);
    }

    // (b) Positions: 0..seq_len
    let positions: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

    // (c) JIT compile decoder layer graph
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let jit_layer: gllm_kernels::compiler::CompiledLayer = {
        let graph = build_decoder_layer_graph(
            seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta,
        );
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        compiler.compile_graph(&graph).map_err(|e| {
            BE::Other(format!("Decoder rerank layer JIT compilation failed: {e}"))
        })?
    };

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    return Err(BE::Other("Decoder rerank forward requires JIT compilation (x86_64 or aarch64)".into()));

    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_heads * head_dim;

    // (d) Run through all decoder layers
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    for layer in 0..num_layers {
        let q_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")))?;
        let k_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")))?;
        let v_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")))?;
        let o_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")))?;
        let rn1_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")))?;
        let gate_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")))?;
        let up_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")))?;
        let down_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")))?;
        let rn2_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")))?;

        let (q_w, k_w, v_w, o_w, gate_w, up_w, down_w) = if transpose_weights {
            (
                transpose_f32(&q_w, q_dim, hidden),
                transpose_f32(&k_w, kv_dim, hidden),
                transpose_f32(&v_w, kv_dim, hidden),
                transpose_f32(&o_w, hidden, q_dim),
                transpose_f32(&gate_w, inter, hidden),
                transpose_f32(&up_w, inter, hidden),
                transpose_f32(&down_w, hidden, inter),
            )
        } else {
            (q_w, k_w, v_w, o_w, gate_w, up_w, down_w)
        };

        let mut layer_out = vec![0.0f32; seq_len * hidden];
        execute_jit_decoder_layer(
            &jit_layer,
            &hidden_state,
            &q_w, &k_w, &v_w, &o_w, &rn1_w,
            &gate_w, &up_w, &down_w, &rn2_w,
            &positions,
            seq_len,
            &mut layer_out,
        );

        hidden_state.copy_from_slice(&layer_out);
    }

    // (e) Final RMSNorm
    let final_norm_w = get_f32_data(weights, backend,
        &crate::weight_names::decoder_final_norm_aliases())?;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let normed = {
        let norm_graph = build_final_norm_graph(seq_len, hidden, eps);
        let mut norm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled_norm = norm_compiler.compile_graph(&norm_graph).map_err(|e| {
            BE::Other(format!("Final norm JIT compilation failed: {e}"))
        })?;

        let mut normed_out = vec![0.0f32; seq_len * hidden];
        execute_jit_final_norm(
            &compiled_norm,
            &hidden_state,
            &final_norm_w,
            seq_len,
            &mut normed_out,
        );
        normed_out
    };

    // (f) Extract last token's hidden state
    let last_hidden = &normed[(seq_len - 1) * hidden..seq_len * hidden];

    // (g) Try score head first, then fall back to lm_head + yes/no logits
    //
    // Strategy:
    // 1. Try explicit score aliases (score.weight, classifier.weight, cls.weight)
    // 2. If not found, try `output.weight` with size disambiguation:
    //    - len <= hidden * 16 → classification head (e.g. [1, hidden] for num_labels=1)
    //    - len > hidden * 16 → lm_head (e.g. [vocab_size, hidden]), use generative path
    // 3. If no output.weight either → use tied embeddings (generative path)
    let score_aliases = crate::weight_names::decoder_score_aliases();
    let score_w_result = get_f32_data(weights, backend, &score_aliases);

    // Try to find score weight, with output.weight fallback + size check
    let score_w_opt: Option<Vec<f32>> = if let Ok(sw) = score_w_result {
        Some(sw)
    } else {
        // Try output.weight with size-based disambiguation
        let output_aliases = vec!["output.weight".to_string()];
        if let Ok(ow) = get_f32_data(weights, backend, &output_aliases) {
            if ow.len() <= hidden * 16 && ow.len() % hidden == 0 {
                // Small enough to be a classification head (num_labels <= 16)
                Some(ow)
            } else {
                // Too large — this is lm_head (vocab_size × hidden), use generative path
                None
            }
        } else {
            None
        }
    };

    if let Some(score_w) = score_w_opt {
        // Classification head path: last_hidden × score_weight → score
        let num_labels = score_w.len() / hidden;
        if num_labels == 0 || score_w.len() % hidden != 0 {
            return Err(BE::Other(format!(
                "score.weight has {} elements, not divisible by hidden_size {}",
                score_w.len(), hidden
            )));
        }

        let mut scores = vec![0.0f32; num_labels];
        for (label, score) in scores.iter_mut().enumerate() {
            let row_start = label * hidden;
            let mut dot = 0.0f32;
            for d in 0..hidden {
                dot += last_hidden[d] * score_w[row_start + d];
            }
            *score = 1.0 / (1.0 + (-dot).exp());
        }
        Ok(scores)
    } else {
        // Generative reranker path: use tied embeddings (lm_head) to get logits,
        // then compute score from "yes"/"no" token probabilities.
        let embed_data = get_f32_data(
            weights, backend,
            &crate::weight_names::decoder_embed_aliases(),
        )?;

        let vocab_size = config.vocab_size;
        let yes_id = config.rerank_yes_token_id.unwrap_or(9454) as usize;
        let no_id = config.rerank_no_token_id.unwrap_or(2753) as usize;

        // embed_data is [vocab_size, hidden] in row-major (each row = one token embedding)
        // lm_head logit for token t = dot(last_hidden, embed_data[t])
        let logit_fn = |token_id: usize| -> f32 {
            if token_id >= vocab_size { return 0.0; }
            let row_start = token_id * hidden;
            let mut dot = 0.0f32;
            for d in 0..hidden {
                dot += last_hidden[d] * embed_data[row_start + d];
            }
            dot
        };

        let logit_yes = logit_fn(yes_id);
        let logit_no = logit_fn(no_id);

        // Score = sigmoid(logit_yes - logit_no)
        let score = 1.0 / (1.0 + (-(logit_yes - logit_no)).exp());
        Ok(vec![score])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::scalar_ops::{scalar_moe_gate, scalar_top_k_experts, scalar_expert_ffn};

    #[test]
    fn moe_gate_softmax_sums_to_one() {
        let hidden_size = 4;
        let num_experts = 3;
        let seq_len = 2;
        let hidden = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4];
        let gate_w = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.5, 0.5, 0.0,
        ];
        let probs = scalar_moe_gate(&hidden, &gate_w, seq_len, num_experts, hidden_size);
        assert_eq!(probs.len(), seq_len * num_experts);
        for s in 0..seq_len {
            let row = &probs[s * num_experts..(s + 1) * num_experts];
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {s} sum = {sum}");
            for &p in row {
                assert!(p >= 0.0, "negative probability: {p}");
            }
        }
    }

    #[test]
    fn top_k_selects_correct_experts() {
        let num_experts = 4;
        let top_k = 2;
        let seq_len = 1;
        let probs = vec![0.3, 0.1, 0.5, 0.1];
        let selections = scalar_top_k_experts(&probs, num_experts, top_k, seq_len);
        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].len(), 2);
        assert_eq!(selections[0][0].0, 2);
        assert_eq!(selections[0][1].0, 0);
        let sum: f32 = selections[0].iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "renormalized sum = {sum}");
    }

    #[test]
    fn top_k_renormalizes_weights() {
        let num_experts = 3;
        let top_k = 2;
        let seq_len = 1;
        let probs = vec![0.6, 0.3, 0.1];
        let selections = scalar_top_k_experts(&probs, num_experts, top_k, seq_len);
        let (idx0, w0) = selections[0][0];
        let (idx1, w1) = selections[0][1];
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert!((w0 - 0.6667).abs() < 0.01, "w0 = {w0}");
        assert!((w1 - 0.3333).abs() < 0.01, "w1 = {w1}");
    }

    #[test]
    fn expert_ffn_produces_correct_shape() {
        let hidden = 4;
        let inter = 6;
        let seq_len = 2;
        let input = vec![0.1f32; seq_len * hidden];
        let gate_w = vec![0.01f32; hidden * inter];
        let up_w = vec![0.01f32; hidden * inter];
        let down_w = vec![0.01f32; inter * hidden];
        let out = scalar_expert_ffn(&input, &gate_w, &up_w, &down_w, seq_len, hidden, inter);
        assert_eq!(out.len(), seq_len * hidden);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn moe_ffn_weighted_combine() {
        let hidden = 4;
        let inter = 6;
        let seq_len = 1;
        let num_experts = 2;
        let top_k = 1;
        let input = vec![1.0f32; hidden];
        let router_w = vec![10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e0_gate = vec![0.1f32; hidden * inter];
        let e0_up = vec![0.1f32; hidden * inter];
        let e0_down = vec![0.1f32; inter * hidden];
        let e1_gate = vec![0.2f32; hidden * inter];
        let e1_up = vec![0.2f32; hidden * inter];
        let e1_down = vec![0.2f32; inter * hidden];
        let experts = vec![
            (e0_gate, e0_up, e0_down),
            (e1_gate, e1_up, e1_down),
        ];
        let out = scalar_moe_ffn(
            &input, &router_w, &experts, None,
            seq_len, hidden, inter, num_experts, top_k,
        );
        assert_eq!(out.len(), hidden);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
