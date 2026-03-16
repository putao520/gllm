//! Scalar math helpers extracted from decoder_forward.rs.
//!
//! Pure-Rust reference implementations used by CPU forward passes.
//! These are the ground-truth algorithms; JIT codegen must match them numerically.

use super::types::{AttentionGeometry, KvCacheSlice, LayerDims, SeqContext};

/// Scalar RMSNorm: out[i] = (x[i] / rms) * w[i]
/// where rms = sqrt(mean(x^2) + eps)
#[allow(dead_code)]
pub(crate) fn scalar_rms_norm(x: &[f32], w: &[f32], out: &mut [f32], hidden: usize, eps: f32) {
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
pub(crate) fn scalar_gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
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
#[allow(dead_code)]
pub(crate) fn scalar_rope(x: &mut [f32], positions: &[u32], head_dim: usize, theta: f64) {
    let seq_len = positions.len();
    let dim = x.len() / seq_len;
    let n_heads = dim / head_dim;
    let half = head_dim / 2;
    for (s, &pos_val) in positions.iter().enumerate() {
        let pos = pos_val as f64;
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
// MoE (Mixture of Experts) helpers
// ---------------------------------------------------------------------------

/// Scalar MoE gate: hidden × gate_w → softmax logits.
pub(crate) fn scalar_moe_gate(
    hidden: &[f32],
    gate_w: &[f32],
    seq_len: usize,
    num_experts: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; seq_len * num_experts];
    scalar_gemm(hidden, gate_w, &mut logits, seq_len, num_experts, hidden_size);
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
pub(crate) fn scalar_top_k_experts(
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
        let sum: f32 = indexed.iter().map(|(_, w)| w).sum();
        if sum > 0.0 {
            for (_, w) in &mut indexed {
                *w /= sum;
            }
        }
        result.push(indexed);
    }
    result
}

/// Scalar SwiGLU FFN for a single expert.
pub(crate) fn scalar_expert_ffn(
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
#[allow(dead_code)]
pub(crate) fn scalar_moe_ffn(
    input: &[f32],
    router_w: &[f32],
    expert_weights: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    shared_expert: Option<&(Vec<f32>, Vec<f32>, Vec<f32>)>,
    seq_len: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    top_k: usize,
) -> Vec<f32> {
    let gate_probs = scalar_moe_gate(input, router_w, seq_len, num_experts, hidden);
    let selections = scalar_top_k_experts(&gate_probs, num_experts, top_k, seq_len);

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

    if let Some((ref sg, ref su, ref sd)) = shared_expert {
        let shared_out = scalar_expert_ffn(input, sg, su, sd, seq_len, hidden, inter);
        for i in 0..seq_len * hidden {
            output[i] += shared_out[i];
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Extracted composite operations (eliminate 3× attention duplication)
// ---------------------------------------------------------------------------

/// Sparsity threshold: softmax weights below this are considered "sparse".
#[allow(dead_code)]
const ATTN_SPARSITY_THRESHOLD: f32 = 0.01;

/// Cached GQA attention: compute attention scores using cached K/V.
///
/// Returns `(attn_output, sparsity)` where sparsity is the fraction of
/// softmax weights below `ATTN_SPARSITY_THRESHOLD` across all heads and positions.
#[allow(dead_code)]
pub(crate) fn cached_gqa_attention(
    q: &[f32],
    kv: &KvCacheSlice,
    seq: &SeqContext,
    geom: &AttentionGeometry,
) -> (Vec<f32>, f32) {
    let scale = 1.0 / (geom.head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq.seq_len * geom.q_dim];
    let mut sparse_count: u64 = 0;
    let mut total_count: u64 = 0;

    for h in 0..geom.num_heads {
        let kv_h = h / geom.heads_per_group;
        let cache_base = (kv.layer * geom.num_kv_heads + kv_h) * kv.max_seq_len * geom.head_dim;

        for s in 0..seq.seq_len {
            let q_offset = s * geom.q_dim + h * geom.head_dim;
            let mut scores = vec![0.0f32; seq.total_seq];

            for (t, score) in scores.iter_mut().enumerate().take(seq.total_seq) {
                let k_offset = cache_base + t * geom.head_dim;
                let mut dot = 0.0f32;
                for d in 0..geom.head_dim {
                    dot += q[q_offset + d] * kv.k[k_offset + d];
                }
                *score = dot * scale;
            }

            // Causal mask
            let cur_pos = seq.positions[s] as usize;
            for (t, score) in scores.iter_mut().enumerate().take(seq.total_seq) {
                if t > cur_pos {
                    *score = f32::NEG_INFINITY;
                }
            }

            // Softmax
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            if sum > 0.0 {
                for score in &mut scores {
                    *score /= sum;
                }
            }

            // Sparsity: count weights below threshold (only valid positions)
            let valid_len = (cur_pos + 1).min(seq.total_seq);
            for &w in scores.iter().take(valid_len) {
                if w < ATTN_SPARSITY_THRESHOLD {
                    sparse_count += 1;
                }
                total_count += 1;
            }

            // Weighted sum of V
            let out_offset = s * geom.q_dim + h * geom.head_dim;
            for d in 0..geom.head_dim {
                let mut val = 0.0f32;
                for (t, &score) in scores.iter().enumerate().take(seq.total_seq) {
                    let v_offset = cache_base + t * geom.head_dim;
                    val += score * kv.v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }
    }

    let sparsity = if total_count > 0 {
        sparse_count as f32 / total_count as f32
    } else {
        0.0
    };
    (attn_out, sparsity)
}

/// SwiGLU FFN: RMSNorm → gate/up projections → SiLU → down projection.
///
/// Replaces the duplicated FFN blocks in incremental decode layers.
#[allow(dead_code)]
pub(crate) fn swiglu_ffn(
    normed: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    dims: &LayerDims,
    seq_len: usize,
) -> Vec<f32> {
    let mut gate_out = vec![0.0f32; seq_len * dims.inter];
    scalar_gemm(normed, gate_w, &mut gate_out, seq_len, dims.inter, dims.hidden);

    let mut up_out = vec![0.0f32; seq_len * dims.inter];
    scalar_gemm(normed, up_w, &mut up_out, seq_len, dims.inter, dims.hidden);

    let mut swiglu = vec![0.0f32; seq_len * dims.inter];
    for i in 0..seq_len * dims.inter {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        swiglu[i] = silu_g * up_out[i];
    }

    let mut down_out = vec![0.0f32; seq_len * dims.hidden];
    scalar_gemm(&swiglu, down_w, &mut down_out, seq_len, dims.hidden, dims.inter);
    down_out
}

/// Full-sequence GQA attention for prefill (no KV cache).
///
/// Returns `(attn_output, sparsity)` where sparsity is the fraction of
/// softmax weights below `ATTN_SPARSITY_THRESHOLD` across all heads and positions.
#[allow(clippy::needless_range_loop)]
#[allow(dead_code)]
pub(crate) fn prefill_gqa_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    geom: &AttentionGeometry,
) -> (Vec<f32>, f32) {
    let scale = 1.0 / (geom.head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * geom.q_dim];
    let mut sparse_count: u64 = 0;
    let mut total_count: u64 = 0;

    for h in 0..geom.num_heads {
        let kv_h = h / geom.heads_per_group;
        for s in 0..seq_len {
            let q_off = s * geom.q_dim + h * geom.head_dim;
            let mut scores = vec![f32::NEG_INFINITY; seq_len];
            for t in 0..=s {
                let k_off = t * geom.kv_dim + kv_h * geom.head_dim;
                let mut dot = 0.0f32;
                for d in 0..geom.head_dim {
                    dot += q[q_off + d] * k[k_off + d];
                }
                scores[t] = dot * scale;
            }
            let max_s = scores[..=s].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(s + 1) {
                *score = (*score - max_s).exp();
                sum += *score;
            }
            if sum > 0.0 {
                for score in scores.iter_mut().take(s + 1) {
                    *score /= sum;
                }
            }

            // Sparsity: count weights below threshold
            for &w in scores.iter().take(s + 1) {
                if w < ATTN_SPARSITY_THRESHOLD {
                    sparse_count += 1;
                }
                total_count += 1;
            }

            let o_off = s * geom.q_dim + h * geom.head_dim;
            for d in 0..geom.head_dim {
                let mut val = 0.0f32;
                for t in 0..=s {
                    val += scores[t] * v[t * geom.kv_dim + kv_h * geom.head_dim + d];
                }
                attn_out[o_off + d] = val;
            }
        }
    }

    let sparsity = if total_count > 0 {
        sparse_count as f32 / total_count as f32
    } else {
        0.0
    };
    (attn_out, sparsity)
}
