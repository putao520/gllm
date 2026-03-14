use super::backend_trait;
use super::cpu_backend::CpuBackend;
use super::weight_helpers::{get_f32_data, needs_weight_transpose, transpose_f32};
use super::Element;
use crate::engine::executor::{
    BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheHandle, LogitsHandle,
};

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

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return Err(BE::Other("Decoder forward requires JIT compilation (x86_64 or aarch64)".into()));

        let kv_dim = num_kv_heads * head_dim;

        // (d) Run through decoder layers
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        for layer in 0..num_layers {
            // Load weights for this layer
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

            // Transpose weights if needed (SafeTensors/GGUF store [out, in])
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

            // Update KV cache for this layer
            if seq_idx < kv_caches.len() {
                update_kv_cache(
                    backend, kv_caches[seq_idx],
                    layer, &hidden_state, &q_w, &k_w, &v_w,
                    &rn1_w, &positions,
                    seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                )?;
            }

            hidden_state.copy_from_slice(&layer_out);
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

/// Update the KV cache metadata after a decoder layer execution.
///
/// REQ-KV-005: KV cache incremental persistence is NOT yet implemented.
///
/// The current gllm-kernels MHA op (`OpKind::MultiHeadAttention`) takes Q, K, V
/// as graph inputs and produces only `attn_out` as output. K/V values after RoPE
/// are computed and consumed entirely within the JIT scratchpad -- they are never
/// exported to an external buffer. The `kv_cache` pointer in `CompiledLayerFn` is
/// accepted by the ABI but unused by the MHA codegen (both x86_64 and aarch64).
///
/// Consequence: every decode step recomputes all K/V from scratch (full
/// recomputation). This is correct but O(n^2) in total compute across n steps.
///
/// To fix this, gllm-kernels must be extended so the MHA op either:
///   (a) accepts external K/V buffer pointers and appends new K/V into them, or
///   (b) exports post-RoPE K/V as additional graph outputs.
/// Until then, only `seq_len` bookkeeping is performed here.
fn update_kv_cache<E: Element>(
    backend: &CpuBackend<E>,
    handle: KvCacheHandle,
    layer: usize,
    _hidden_state: &[f32],
    _q_w: &[f32],
    _k_w: &[f32],
    _v_w: &[f32],
    _rn1_w: &[f32],
    _positions: &[u32],
    seq_len: usize,
    _hidden: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    _eps: f32,
    _rope_theta: f64,
) -> Result<(), BE> {
    // REQ-KV-005: KV Cache Incremental Persistence
    //
    // CURRENT LIMITATION: The JIT-compiled MHA op computes K/V internally
    // during attention but does not export them to an external buffer.
    // This means every decode step recomputes all K/V from scratch.
    //
    // To implement true incremental KV caching, the MHA op in gllm-kernels
    // needs to support external K/V buffer output pointers. This requires:
    // 1. gllm-kernels MHA op to accept kv_cache_ptr as additional output
    // 2. build_decoder_layer_graph() to wire the buffer into the graph
    // 3. This function to manage buffer lifecycle and seq_len tracking
    //
    // Until then, seq_len tracking is maintained for correctness of
    // position encoding and attention mask computation.
    log::debug!("update_kv_cache: layer={layer}, seq_len={seq_len} (metadata-only, full K/V recomputation per step)");

    let mut store = backend.kv_store().lock().map_err(|e| {
        BE::Cpu(format!("KV store lock poisoned: {e}"))
    })?;

    if let Some(buffer) = store.get_mut(&handle.0) {
        if layer == 0 {
            buffer.seq_len = buffer.seq_len.saturating_add(seq_len);
            if buffer.seq_len > buffer.max_seq_len {
                buffer.seq_len = buffer.max_seq_len;
            }
        }
    }

    Ok(())
}
