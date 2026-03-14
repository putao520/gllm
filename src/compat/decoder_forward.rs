use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::weight_helpers::{
    get_f32_data, get_weight_data, needs_weight_transpose, quantized_linear,
    transpose_f32, try_get_f32_data, weight_data_to_f32, WeightData,
};
use super::Element;
use crate::engine::executor::{
    BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheHandle, LogitsHandle,
};

// ---------------------------------------------------------------------------
// Scalar math helpers for KV cache computation (outside JIT)
// ---------------------------------------------------------------------------

/// Scalar RMSNorm: out[i] = (x[i] / rms) * w[i]
/// where rms = sqrt(mean(x^2) + eps)
fn scalar_rms_norm(x: &[f32], w: &[f32], out: &mut [f32], hidden: usize, eps: f32) {
    let n = x.len() / hidden;
    for row in 0..n {
        let start = row * hidden;
        let end = start + hidden;
        let slice = &x[start..end];
        let ss: f32 = slice.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let rms = (ss + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for d in 0..hidden {
            out[start + d] = slice[d] * inv_rms * w[d];
        }
    }
}

/// Scalar GEMM: C = A * B, A is [m, k], B is [k, n], C is [m, n] (row-major)
pub(super) fn scalar_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Scalar RoPE: apply rotary position embedding to x[seq_len, dim]
/// dim = num_heads * head_dim, positions[seq_len]
pub(super) fn scalar_rope(x: &mut [f32], positions: &[u32], head_dim: usize, theta: f64) {
    let seq_len = positions.len();
    let dim = x.len() / seq_len;
    let n_heads = dim / head_dim;
    let half = head_dim / 2;
    for s in 0..seq_len {
        let pos = positions[s] as f64;
        for h in 0..n_heads {
            let base = s * dim + h * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos * freq;
                let cos_val = angle.cos() as f32;
                let sin_val = angle.sin() as f32;
                let x0 = x[base + i];
                let x1 = x[base + i + half];
                x[base + i] = x0 * cos_val - x1 * sin_val;
                x[base + i + half] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar MoE (Mixture of Experts) helpers
// ---------------------------------------------------------------------------

/// Scalar MoE gate: hidden [seq_len, hidden_size] × gate_w [hidden_size, num_experts] → logits [seq_len, num_experts]
/// Then softmax per row.
fn scalar_moe_gate(
    hidden: &[f32],
    gate_w: &[f32],
    seq_len: usize,
    num_experts: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; seq_len * num_experts];
    scalar_gemm(hidden, gate_w, &mut logits, seq_len, num_experts, hidden_size);
    // Softmax per row
    for s in 0..seq_len {
        let row = &mut logits[s * num_experts..(s + 1) * num_experts];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
        for x in row.iter_mut() {
            *x = (*x - max).exp() / exp_sum;
        }
    }
    logits
}

/// Scalar top-k expert selection with renormalization.
/// Returns (expert_index, renormalized_weight) pairs for each token position.
fn scalar_top_k_experts(
    gate_probs: &[f32],
    num_experts: usize,
    top_k: usize,
    seq_len: usize,
) -> Vec<Vec<(usize, f32)>> {
    let mut result = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let row = &gate_probs[s * num_experts..(s + 1) * num_experts];
        let mut indexed: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(top_k);
        // Renormalize weights
        let sum: f32 = indexed.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            for (_, w) in indexed.iter_mut() {
                *w /= sum;
            }
        }
        result.push(indexed);
    }
    result
}

/// Scalar SwiGLU FFN for a single expert.
/// input: [seq_len, hidden_size]
/// gate_w: [hidden_size, inter], up_w: [hidden_size, inter], down_w: [inter, hidden_size]
/// output: [seq_len, hidden_size]
fn scalar_expert_ffn(
    input: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    seq_len: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let mut gate_out = vec![0.0f32; seq_len * inter];
    scalar_gemm(input, gate_w, &mut gate_out, seq_len, inter, hidden);

    let mut up_out = vec![0.0f32; seq_len * inter];
    scalar_gemm(input, up_w, &mut up_out, seq_len, inter, hidden);

    // SwiGLU: silu(gate) * up
    let mut swiglu = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        swiglu[i] = silu_g * up_out[i];
    }

    let mut down_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&swiglu, down_w, &mut down_out, seq_len, hidden, inter);
    down_out
}

/// Scalar MoE forward: gate routing + expert FFN + weighted combine.
///
/// For each token position:
///   1. Compute gate logits → softmax probabilities
///   2. Select top_k experts
///   3. Run each selected expert's SwiGLU FFN
///   4. Weighted sum of expert outputs (by renormalized gate probabilities)
///
/// If shared experts exist, their output is added to the routed output.
fn scalar_moe_ffn(
    input: &[f32],
    router_w: &[f32],
    expert_weights: &[(Vec<f32>, Vec<f32>, Vec<f32>)], // [(gate, up, down) per expert]
    shared_expert: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    seq_len: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
) -> Vec<f32> {
    // Step 1: Gate routing
    let gate_probs = scalar_moe_gate(input, router_w, seq_len, num_experts, hidden);

    // Step 2: Top-K selection
    let selections = scalar_top_k_experts(&gate_probs, num_experts, top_k, seq_len);

    // Step 3+4: Per-token expert execution and weighted combine
    let mut output = vec![0.0f32; seq_len * hidden];

    for s in 0..seq_len {
        let token_input = &input[s * hidden..(s + 1) * hidden];
        for &(expert_idx, weight) in &selections[s] {
            if expert_idx >= expert_weights.len() {
                continue;
            }
            let (ref gw, ref uw, ref dw) = expert_weights[expert_idx];
            let expert_out = scalar_expert_ffn(token_input, gw, uw, dw, 1, hidden, inter);
            for d in 0..hidden {
                output[s * hidden + d] += weight * expert_out[d];
            }
        }
    }

    // Step 5: Add shared expert output if present
    if let Some((ref sg, ref su, ref sd)) = shared_expert {
        let shared_out = scalar_expert_ffn(input, sg, su, sd, seq_len, hidden, inter);
        for i in 0..seq_len * hidden {
            output[i] += shared_out[i];
        }
    }

    output
}

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
            for t in 0..=s {
                let k_off = t * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim { dot += q_proj[q_off + d] * k_proj[k_off + d]; }
                scores[t] = dot * scale;
            }
            let max_s = scores[..=s].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..=s { scores[t] = (scores[t] - max_s).exp(); sum += scores[t]; }
            if sum > 0.0 { for t in 0..=s { scores[t] /= sum; } }
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
    let q_dim = num_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads;

    // Step 1: Pre-attention RMSNorm
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    // Step 2: Q projection for new tokens only
    let mut q_proj = vec![0.0f32; seq_len * q_dim];
    scalar_gemm(&normed, q_w, &mut q_proj, seq_len, q_dim, hidden);

    // Step 3: Apply RoPE to Q
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Step 4: Attention using cached K/V
    // K/V cache layout: [num_layers][num_kv_heads][max_seq_len][head_dim]
    // We read total_seq tokens from the cache for this layer.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let cache_base = (layer * num_kv_heads + kv_h) * max_seq_len * head_dim;

        for s in 0..seq_len {
            // Q vector for this head and position
            let q_offset = s * q_dim + h * head_dim;

            // Compute attention scores: dot(Q, K[t]) for t in 0..total_seq
            let mut scores = vec![0.0f32; total_seq];
            for t in 0..total_seq {
                let k_offset = cache_base + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_proj[q_offset + d] * kv_cache_k[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            // Causal mask: only attend to positions <= current position
            let cur_pos = positions[s] as usize;
            for t in 0..total_seq {
                if t > cur_pos {
                    scores[t] = f32::NEG_INFINITY;
                }
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..total_seq {
                scores[t] = (scores[t] - max_score).exp();
                sum += scores[t];
            }
            if sum > 0.0 {
                for t in 0..total_seq {
                    scores[t] /= sum;
                }
            }

            // Weighted sum of V
            let out_offset = s * q_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..total_seq {
                    let v_offset = cache_base + t * head_dim;
                    val += scores[t] * kv_cache_v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }
    }

    // Step 5: Output projection
    let mut o_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&attn_out, o_w, &mut o_out, seq_len, hidden, q_dim);

    // Step 6: Residual connection 1
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        resid1[i] = hidden_state[i] + o_out[i];
    }

    // Step 7: Pre-FFN RMSNorm
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);

    // Step 8: SwiGLU FFN
    let mut gate_out = vec![0.0f32; seq_len * inter];
    scalar_gemm(&normed2, gate_w, &mut gate_out, seq_len, inter, hidden);

    let mut up_out = vec![0.0f32; seq_len * inter];
    scalar_gemm(&normed2, up_w, &mut up_out, seq_len, inter, hidden);

    // SwiGLU: silu(gate) * up
    let mut swiglu = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        swiglu[i] = silu_g * up_out[i];
    }

    let mut down_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&swiglu, down_w, &mut down_out, seq_len, hidden, inter);

    // Step 9: Residual connection 2
    for i in 0..seq_len * hidden {
        output[i] = resid1[i] + down_out[i];
    }
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
    let q_dim = num_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads;

    // Step 1: Pre-attention RMSNorm
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    // Step 2: Q projection (quantized acceleration)
    let mut q_proj = vec![0.0f32; seq_len * q_dim];
    quantized_linear(
        backend, &normed, q_w, &mut q_proj,
        seq_len, q_dim, hidden, transpose_weights,
    )?;

    // Step 3: Apply RoPE to Q
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Step 4: Attention using cached K/V
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let cache_base = (layer * num_kv_heads + kv_h) * max_seq_len * head_dim;

        for s in 0..seq_len {
            let q_offset = s * q_dim + h * head_dim;
            let mut scores = vec![0.0f32; total_seq];
            for t in 0..total_seq {
                let k_offset = cache_base + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_proj[q_offset + d] * kv_cache_k[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            let cur_pos = positions[s] as usize;
            for t in 0..total_seq {
                if t > cur_pos {
                    scores[t] = f32::NEG_INFINITY;
                }
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..total_seq {
                scores[t] = (scores[t] - max_score).exp();
                sum += scores[t];
            }
            if sum > 0.0 {
                for t in 0..total_seq {
                    scores[t] /= sum;
                }
            }

            let out_offset = s * q_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..total_seq {
                    let v_offset = cache_base + t * head_dim;
                    val += scores[t] * kv_cache_v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }
    }

    // Step 5: Output projection (quantized acceleration)
    let mut o_out = vec![0.0f32; seq_len * hidden];
    quantized_linear(
        backend, &attn_out, o_w, &mut o_out,
        seq_len, hidden, q_dim, transpose_weights,
    )?;

    // Step 6: Residual connection 1
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        resid1[i] = hidden_state[i] + o_out[i];
    }

    // Step 7: Pre-FFN RMSNorm
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);

    // Step 8: SwiGLU FFN (quantized acceleration)
    let mut gate_out = vec![0.0f32; seq_len * inter];
    quantized_linear(
        backend, &normed2, gate_w, &mut gate_out,
        seq_len, inter, hidden, transpose_weights,
    )?;

    let mut up_out = vec![0.0f32; seq_len * inter];
    quantized_linear(
        backend, &normed2, up_w, &mut up_out,
        seq_len, inter, hidden, transpose_weights,
    )?;

    let mut swiglu = vec![0.0f32; seq_len * inter];
    for i in 0..seq_len * inter {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        swiglu[i] = silu_g * up_out[i];
    }

    let mut down_out = vec![0.0f32; seq_len * hidden];
    quantized_linear(
        backend, &swiglu, down_w, &mut down_out,
        seq_len, hidden, inter, transpose_weights,
    )?;

    // Step 9: Residual connection 2
    for i in 0..seq_len * hidden {
        output[i] = resid1[i] + down_out[i];
    }
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
    let q_dim = num_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads;

    // Step 1: Pre-attention RMSNorm
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    // Step 2: Q projection
    let mut q_proj = vec![0.0f32; seq_len * q_dim];
    scalar_gemm(&normed, q_w, &mut q_proj, seq_len, q_dim, hidden);

    // Step 3: RoPE on Q
    scalar_rope(&mut q_proj, positions, head_dim, rope_theta);

    // Step 4: Attention using cached K/V
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let cache_base = (layer * num_kv_heads + kv_h) * max_seq_len * head_dim;

        for s in 0..seq_len {
            let q_offset = s * q_dim + h * head_dim;
            let mut scores = vec![0.0f32; total_seq];
            for t in 0..total_seq {
                let k_offset = cache_base + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_proj[q_offset + d] * kv_cache_k[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            let cur_pos = positions[s] as usize;
            for t in 0..total_seq {
                if t > cur_pos {
                    scores[t] = f32::NEG_INFINITY;
                }
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..total_seq {
                scores[t] = (scores[t] - max_score).exp();
                sum += scores[t];
            }
            if sum > 0.0 {
                for t in 0..total_seq {
                    scores[t] /= sum;
                }
            }

            let out_offset = s * q_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..total_seq {
                    let v_offset = cache_base + t * head_dim;
                    val += scores[t] * kv_cache_v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }
    }

    // Step 5: Output projection
    let mut o_out = vec![0.0f32; seq_len * hidden];
    scalar_gemm(&attn_out, o_w, &mut o_out, seq_len, hidden, q_dim);

    // Step 6: Residual connection 1
    let mut resid1 = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len * hidden {
        resid1[i] = hidden_state[i] + o_out[i];
    }

    // Step 7: Pre-FFN RMSNorm
    let mut normed2 = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(&resid1, rn2_w, &mut normed2, hidden, eps);

    // Step 8: MoE FFN (replaces standard SwiGLU)
    let moe_out = scalar_moe_ffn(
        &normed2, router_w, expert_weights, shared_expert,
        seq_len, hidden, inter, num_experts, top_k,
    );

    // Step 9: Residual connection 2
    for i in 0..seq_len * hidden {
        output[i] = resid1[i] + moe_out[i];
    }
}

// ---------------------------------------------------------------------------
// JIT compilation for decoder (LLaMA/Qwen-style) layers
// ---------------------------------------------------------------------------

/// Build a CompilerGraph for a single decoder layer (pre-norm, RMSNorm + SwiGLU).
///
/// Graph structure (per-layer):
///   RMSNorm → Q/K/V projection → RoPE → MultiHeadAttention → O projection
///   → Residual → RMSNorm → SwiGLU FFN → Residual
///
/// KV cache is handled outside the JIT graph (pre/post copy).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_decoder_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // ── Graph inputs ──
    let input = g.add_tensor("input", vec![s, h], dt);

    // Attention weights (no bias for LLaMA-style)
    // Q maps hidden → q_dim (q_dim may differ from hidden for Qwen3 etc.)
    let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_dim], dt);
    let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);

    // RMSNorm 1 (pre-attention)
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);

    // FFN weights (SwiGLU: gate, up, down)
    let w_gate = g.add_tensor("w_gate", vec![h, inter], dt);
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);

    // RMSNorm 2 (pre-FFN)
    let rn2_w = g.add_tensor("rn2_w", vec![h], dt);

    g.inputs = vec![
        input, w_q, w_k, w_v, w_o, rn1_w,
        w_gate, w_up, w_down, rn2_w,
    ];

    // ── Pre-attention RMSNorm ──
    let normed1 = g.add_tensor("normed1", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, rn1_w],
        vec![normed1],
        "rms_norm_1",
    );

    // ── Q/K/V Projections ──
    // Q = normed1 * W_q  [s, h] × [h, q_dim] → [s, q_dim]
    let q_out = g.add_tensor("q", vec![s, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: q_dim, k: h },
        vec![normed1, w_q],
        vec![q_out],
        "gemm_q",
    );

    // K = normed1 * W_k  [s, h] × [h, kv_dim] → [s, kv_dim]
    let k_out = g.add_tensor("k", vec![s, kv_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_dim, k: h },
        vec![normed1, w_k],
        vec![k_out],
        "gemm_k",
    );

    // V = normed1 * W_v  [s, h] × [h, kv_dim] → [s, kv_dim]
    let v_out = g.add_tensor("v", vec![s, kv_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_dim, k: h },
        vec![normed1, w_v],
        vec![v_out],
        "gemm_v",
    );

    // ── RoPE on Q and K ──
    let q_rope = g.add_tensor("q_rope", vec![s, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![q_out],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor("k_rope", vec![s, kv_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![k_out],
        vec![k_rope],
        "rope_k",
    );

    // ── Multi-Head Attention ──
    // For GQA: Q has num_heads, K/V have num_kv_heads.
    // The MHA op handles head reshaping internally.
    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention { seq_len: s, num_heads, head_dim },
        vec![q_rope, k_rope, v_out],
        vec![attn_out],
        "mha",
    );

    // ── Output projection ──
    // O = attn_out * W_o  [s, q_dim] × [q_dim, h] → [s, h]
    let o_out = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim },
        vec![attn_out, w_o],
        vec![o_out],
        "gemm_o",
    );

    // ── Residual connection 1 ──
    let resid1 = g.add_tensor("residual1", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_out],
        vec![resid1],
        "residual_1",
    );

    // ── Pre-FFN RMSNorm ──
    let normed2 = g.add_tensor("normed2", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![resid1, rn2_w],
        vec![normed2],
        "rms_norm_2",
    );

    // ── SwiGLU FFN ──
    // gate = normed2 * W_gate  [s, h] × [h, inter] → [s, inter]
    let gate_out = g.add_tensor("ffn_gate", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h },
        vec![normed2, w_gate],
        vec![gate_out],
        "gemm_gate",
    );

    // up = normed2 * W_up  [s, h] × [h, inter] → [s, inter]
    let up_out = g.add_tensor("ffn_up", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h },
        vec![normed2, w_up],
        vec![up_out],
        "gemm_up",
    );

    // SwiGLU: silu(gate) * up → [s, inter]
    let swiglu_out = g.add_tensor("ffn_swiglu", vec![s, inter], dt);
    g.add_op(
        OpKind::SwiGlu,
        vec![gate_out, up_out],
        vec![swiglu_out],
        "swiglu",
    );

    // down = swiglu_out * W_down  [s, inter] × [inter, h] → [s, h]
    let down_out = g.add_tensor("ffn_down", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: inter },
        vec![swiglu_out, w_down],
        vec![down_out],
        "gemm_down",
    );

    // ── Residual connection 2 ──
    let output = g.add_tensor("output", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![resid1, down_out],
        vec![output],
        "residual_2",
    );

    g.outputs = vec![output];
    g
}

/// Build a CompilerGraph for KV projection only: RmsNorm → K Gemm → K RoPE.
///
/// This graph computes the K projection with RoPE applied, which is the most
/// compute-intensive part of KV cache update. V projection (RmsNorm → V Gemm)
/// is computed separately via scalar GEMM since there is no Concat OpKind.
///
/// Inputs: [input, rn1_w, w_k]
/// Output: k_rope [seq_len, kv_dim]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn build_kv_projection_graph(
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let kv_dim = num_kv_heads * head_dim;

    // Graph inputs
    let input = g.add_tensor("input", vec![s, h], dt);
    let rn1_w = g.add_tensor("rn1_w", vec![h], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_dim], dt);

    g.inputs = vec![input, rn1_w, w_k];

    // RmsNorm
    let normed = g.add_tensor("normed", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, rn1_w],
        vec![normed],
        "rms_norm_kv",
    );

    // K = normed * W_k  [s, h] × [h, kv_dim] → [s, kv_dim]
    let k_out = g.add_tensor("k", vec![s, kv_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_dim, k: h },
        vec![normed, w_k],
        vec![k_out],
        "gemm_k",
    );

    // RoPE on K
    let k_rope = g.add_tensor("k_rope", vec![s, kv_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![k_out],
        vec![k_rope],
        "rope_k",
    );

    g.outputs = vec![k_rope];
    g
}

/// Execute a JIT-compiled KV projection graph.
///
/// Computes K projection with RoPE via JIT, and V projection via scalar GEMM.
/// The RmsNorm is computed once inside the JIT graph (for K) and reused for V
/// via scalar computation.
///
/// Returns (k_rope, v_proj) both as [seq_len, kv_dim].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn execute_kv_projection(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    rn1_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let kv_dim = num_kv_heads * head_dim;

    // Execute JIT graph for K projection (RmsNorm → Gemm → RoPE)
    let weight_slices: &[&[f32]] = &[rn1_w, k_w];
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

    let mut k_rope = vec![0.0f32; seq_len * kv_dim];
    let scratchpad_bytes = compiled.scratchpad_bytes;
    let mut scratchpad = vec![0u8; scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr() as *const u8,
            std::ptr::null_mut(),
            positions.as_ptr(),
            std::ptr::null(),
            1,
            seq_len,
            k_rope.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }

    // Compute V projection via scalar: RmsNorm → V Gemm (no RoPE on V)
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, v_w, &mut v_proj, seq_len, kv_dim, hidden);

    (k_rope, v_proj)
}

/// Write pre-computed K/V data into the KV cache buffer.
///
/// Pure write operation — no computation. Takes k_data (post-RoPE) and v_data,
/// copies them into the correct positions in the KvCacheBuffer.
///
/// KvCacheBuffer layout: [num_layers][num_kv_heads][max_seq_len][head_dim]
fn write_kv_to_cache<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    k_data: &[f32],
    v_data: &[f32],
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), BE> {
    let kv_dim = num_kv_heads * head_dim;
    let mut store = backend.kv_store().lock().map_err(|e| {
        BE::Cpu(format!("KV store lock poisoned: {e}"))
    })?;

    let buffer = store.get_mut(&handle.0).ok_or_else(|| {
        BE::Cpu(format!("KV cache handle {} not found", handle.0))
    })?;

    let write_start = if layer == 0 { buffer.seq_len } else { buffer.seq_len.saturating_sub(seq_len) };
    let max_seq = buffer.max_seq_len;

    if write_start + seq_len > max_seq {
        return Err(BE::Cpu(format!(
            "KV cache overflow: write_start={write_start} + seq_len={seq_len} > max_seq_len={max_seq}"
        )));
    }

    for h in 0..num_kv_heads {
        let layer_head_base = (layer * num_kv_heads + h) * max_seq * head_dim;
        for s in 0..seq_len {
            let cache_offset = layer_head_base + (write_start + s) * head_dim;
            let proj_offset = s * kv_dim + h * head_dim;
            buffer.k[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&k_data[proj_offset..proj_offset + head_dim]);
            buffer.v[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&v_data[proj_offset..proj_offset + head_dim]);
        }
    }

    if layer == 0 {
        buffer.seq_len = (buffer.seq_len + seq_len).min(max_seq);
    }

    log::debug!(
        "write_kv_to_cache: layer={layer}, wrote {seq_len} tokens at pos {write_start}, total_seq={}",
        if layer == 0 { write_start + seq_len } else { buffer.seq_len }
    );

    Ok(())
}

/// Execute a JIT-compiled decoder layer.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_decoder_layer(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    o_w: &[f32],
    rn1_w: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    rn2_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    output: &mut [f32],
) {
    // Pack weights in graph input order:
    // [w_q, w_k, w_v, w_o, rn1_w, w_gate, w_up, w_down, rn2_w]
    let weight_slices: &[&[f32]] = &[
        q_w, k_w, v_w, o_w, rn1_w,
        gate_w, up_w, down_w, rn2_w,
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
            std::ptr::null_mut(), // KV cache managed externally
            positions.as_ptr(),
            std::ptr::null(),     // no seq_lens array needed for single-batch
            1,                    // batch_size = 1
            seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

/// Build a CompilerGraph for the final RMSNorm + lm_head projection.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_lm_head_graph(
    seq_len: usize,
    hidden: usize,
    vocab_size: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let norm_w = g.add_tensor("norm_w", vec![hidden], dt);
    let lm_w = g.add_tensor("lm_w", vec![hidden, vocab_size], dt);

    g.inputs = vec![input, norm_w, lm_w];

    // Final RMSNorm
    let normed = g.add_tensor("normed", vec![seq_len, hidden], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, norm_w],
        vec![normed],
        "final_rms_norm",
    );

    // lm_head: [seq_len, hidden] × [hidden, vocab_size] → [seq_len, vocab_size]
    let logits = g.add_tensor("logits", vec![seq_len, vocab_size], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden },
        vec![normed, lm_w],
        vec![logits],
        "lm_head",
    );

    g.outputs = vec![logits];
    g
}

/// Execute the JIT-compiled lm_head (final norm + projection).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_lm_head(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    norm_w: &[f32],
    lm_w: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    let weight_slices: &[&[f32]] = &[norm_w, lm_w];
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
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
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

/// Build a CompilerGraph for final RMSNorm only (no lm_head projection).
///
/// Used for decoder-based embedding models: after running all decoder layers,
/// apply RMSNorm to get the hidden state, then mean pool externally.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn build_final_norm_graph(
    seq_len: usize,
    hidden: usize,
    eps: f32,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let norm_w = g.add_tensor("norm_w", vec![hidden], dt);
    g.inputs = vec![input, norm_w];

    let normed = g.add_tensor("normed", vec![seq_len, hidden], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, norm_w],
        vec![normed],
        "final_rms_norm",
    );

    g.outputs = vec![normed];
    g
}

/// Execute JIT-compiled final norm (RMSNorm only, no lm_head).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn execute_jit_final_norm(
    compiled: &gllm_kernels::compiler::CompiledLayer,
    hidden_state: &[f32],
    norm_w: &[f32],
    seq_len: usize,
    output: &mut [f32],
) {
    let weight_bytes = norm_w.len() * 4;
    let mut weights_buf = vec![0u8; weight_bytes];
    weights_buf.copy_from_slice(
        unsafe { std::slice::from_raw_parts(norm_w.as_ptr() as *const u8, weight_bytes) }
    );

    let scratchpad_bytes = compiled.scratchpad_bytes;
    let mut scratchpad = vec![0u8; scratchpad_bytes];

    unsafe {
        compiled.execute(
            hidden_state.as_ptr() as *const u8,
            weights_buf.as_ptr() as *const u8,
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            seq_len,
            output.as_mut_ptr() as *mut u8,
            scratchpad.as_mut_ptr(),
        );
    }
}

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
        for d in 0..hidden {
            pooled[d] += normed[s * hidden + d];
        }
    }
    let scale = 1.0 / seq_len as f32;
    for d in 0..hidden {
        pooled[d] *= scale;
    }

    // (g) L2 normalize (standard for embedding models)
    let l2_norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if l2_norm > 1e-12 {
        for d in 0..hidden {
            pooled[d] /= l2_norm;
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
        for label in 0..num_labels {
            let row_start = label * hidden;
            let mut dot = 0.0f32;
            for d in 0..hidden {
                dot += last_hidden[d] * score_w[row_start + d];
            }
            scores[label] = 1.0 / (1.0 + (-dot).exp());
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

/// Compute K/V post-RoPE for new tokens and write them into the KV cache buffer.
///
/// REQ-KV-005: KV Cache Incremental Persistence.
///
/// This function computes K and V projections (RMSNorm → GEMM → RoPE for K)
/// using scalar Rust, then writes the results into the KvCacheBuffer at the
/// correct position offset. This avoids recomputing all K/V from scratch on
/// each decode step.
///
/// KvCacheBuffer layout: [num_layers][num_kv_heads][max_seq_len][head_dim]
fn update_kv_cache<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    hidden_state: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    rn1_w: &[f32],
    positions: &[u32],
    seq_len: usize,
    hidden: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
) -> Result<(), BE> {
    let kv_dim = num_kv_heads * head_dim;

    // Step 1: RMSNorm on hidden_state (same norm as pre-attention)
    let mut normed = vec![0.0f32; seq_len * hidden];
    scalar_rms_norm(hidden_state, rn1_w, &mut normed, hidden, eps);

    // Step 2: K = normed * W_k  [seq_len, hidden] × [hidden, kv_dim] → [seq_len, kv_dim]
    let mut k_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, k_w, &mut k_proj, seq_len, kv_dim, hidden);

    // Step 3: V = normed * W_v  [seq_len, hidden] × [hidden, kv_dim] → [seq_len, kv_dim]
    let mut v_proj = vec![0.0f32; seq_len * kv_dim];
    scalar_gemm(&normed, v_w, &mut v_proj, seq_len, kv_dim, hidden);

    // Step 4: Apply RoPE to K (V does not get RoPE)
    scalar_rope(&mut k_proj, positions, head_dim, rope_theta);

    // Step 5: Write K/V into cache buffer
    let mut store = backend.kv_store().lock().map_err(|e| {
        BE::Cpu(format!("KV store lock poisoned: {e}"))
    })?;

    let buffer = store.get_mut(&handle.0).ok_or_else(|| {
        BE::Cpu(format!("KV cache handle {} not found", handle.0))
    })?;

    // Determine write position: current seq_len in buffer is the start offset
    // (only read from layer 0 to avoid double-counting)
    let write_start = if layer == 0 { buffer.seq_len } else { buffer.seq_len.saturating_sub(seq_len) };
    let max_seq = buffer.max_seq_len;

    if write_start + seq_len > max_seq {
        return Err(BE::Cpu(format!(
            "KV cache overflow: write_start={write_start} + seq_len={seq_len} > max_seq_len={max_seq}"
        )));
    }

    // Buffer layout: [num_layers][num_kv_heads][max_seq_len][head_dim]
    // For a given (layer, head, pos), the flat index is:
    //   (layer * num_kv_heads + head) * max_seq_len * head_dim + pos * head_dim
    for h in 0..num_kv_heads {
        let layer_head_base = (layer * num_kv_heads + h) * max_seq * head_dim;
        for s in 0..seq_len {
            let cache_offset = layer_head_base + (write_start + s) * head_dim;
            let proj_offset = s * kv_dim + h * head_dim;
            buffer.k[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&k_proj[proj_offset..proj_offset + head_dim]);
            buffer.v[cache_offset..cache_offset + head_dim]
                .copy_from_slice(&v_proj[proj_offset..proj_offset + head_dim]);
        }
    }

    // Update seq_len counter (only on layer 0 to avoid double-counting)
    if layer == 0 {
        buffer.seq_len = (buffer.seq_len + seq_len).min(max_seq);
    }

    log::debug!(
        "update_kv_cache: layer={layer}, wrote {seq_len} tokens at pos {write_start}, total_seq={}",
        if layer == 0 { write_start + seq_len } else { buffer.seq_len }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
