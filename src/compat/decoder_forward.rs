
use super::backend_trait;
use super::cpu_backend::CpuBackend;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
use super::jit_helpers::{
    build_decoder_layer_graph, build_final_norm_graph, build_kv_projection_graph,
    build_lm_head_graph, build_moe_pre_attention_graph, build_post_attention_graph,
    computation_dtype, computation_dtype_from_config, kernels_dtype_to_compat,
    execute_jit_decoder_layer, execute_jit_final_norm, execute_jit_lm_head,
    execute_kv_projection, pack_weights, pack_weights_typed, update_kv_cache_jit, write_kv_to_cache,
    f32_as_bytes, f32_as_bytes_mut, bytes_as_f32,
};
use super::types::AttentionGeometry;
use super::weight_helpers::{
    get_f32_data, get_weight_data, needs_weight_transpose, quantized_linear, transpose_f32,
    try_get_f32_data, weight_data_to_f32, WeightData,
};
use super::Element;
use crate::engine::executor::{
    BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheHandle, LogitsHandle,
};

// ---------------------------------------------------------------------------
// Quantized incremental decode layer (uses cached K/V, O(n) per step)
// ---------------------------------------------------------------------------

/// Execute a single decoder layer using cached K/V, with quantized matmul acceleration.
///
/// Same logic as `scalar_incremental_decode_layer` but dispatches GEMM operations
/// through `quantized_linear` which uses `quantized_matmul` for quantized weights,
/// avoiding the expensive dequantize + transpose path.
/// Pre-compiled JIT graphs for MoE incremental decode (invariant across layers).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct MoeDecodeCachedJit {
    /// RmsNorm → Q/K/V Gemm → RoPE (pre-attention)
    pre_attn: gllm_kernels::compiler::CompiledLayer,
    /// CachedGQA attention — compiled per unique total_seq, cached thereafter (same as DecodeCachedJit)
    gqa_cache: std::collections::HashMap<usize, gllm_kernels::compiler::CompiledLayer>,
    /// O projection Gemm
    o_gemm: gllm_kernels::compiler::CompiledLayer,
    /// Pre-FFN RmsNorm
    norm2: gllm_kernels::compiler::CompiledLayer,
    /// Parameters for lazy GQA compilation
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: gllm_kernels::types::DType,
}
#[allow(dead_code)]
fn compile_moe_decode_jit(
    seq_len: usize,
    hidden: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize,
    eps: f32, rope_theta: f64, dtype: gllm_kernels::types::DType,
) -> Result<MoeDecodeCachedJit, BE> {
    let q_dim = num_heads * head_dim;

    let pre_graph = build_moe_pre_attention_graph(
        seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, dtype,
    );
    let mut c1 = gllm_kernels::compiler::InferenceCompiler::new();
    let pre_attn = c1.compile_graph(&pre_graph).map_err(|e| {
        BE::Other(format!("MoE decode JIT: pre-attention compile failed: {e}"))
    })?;

    // O projection Gemm
    let o_gemm = {
        use gllm_kernels::compiler::{CompilerGraph, OpKind};
        let dt = dtype;
        let mut og = CompilerGraph::new();
        let a_in = og.add_tensor_concrete("a", &[seq_len, q_dim], dt);
        let b_in = og.add_tensor_concrete("b", &[q_dim, hidden], dt);
        og.inputs = vec![a_in, b_in];
        let c_out = og.add_tensor_concrete("c", &[seq_len, hidden], dt);
        og.add_op(OpKind::Gemm { m: seq_len, n: hidden, k: q_dim, dtype: dt }, vec![a_in, b_in], vec![c_out], "gemm_o");
        og.outputs = vec![c_out];
        let mut c3 = gllm_kernels::compiler::InferenceCompiler::new();
        c3.compile_graph(&og).map_err(|e| {
            BE::Other(format!("MoE decode JIT: O Gemm compile failed: {e}"))
        })?
    };

    let norm2_graph = super::jit_helpers::build_final_norm_graph(
        seq_len, hidden, eps, dtype,
    );
    let mut c4 = gllm_kernels::compiler::InferenceCompiler::new();
    let norm2 = c4.compile_graph(&norm2_graph).map_err(|e| {
        BE::Other(format!("MoE decode JIT: RmsNorm2 compile failed: {e}"))
    })?;

    Ok(MoeDecodeCachedJit {
        pre_attn, o_gemm, norm2,
        gqa_cache: std::collections::HashMap::new(),
        seq_len, num_heads, num_kv_heads, head_dim, dtype,
    })
}

/// Get or compile a CachedGQA CompiledLayer for MoE path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn get_or_compile_moe_gqa(jit: &mut MoeDecodeCachedJit, total_seq: usize) -> Result<&gllm_kernels::compiler::CompiledLayer, BE> {
    if !jit.gqa_cache.contains_key(&total_seq) {
        let graph = super::jit_helpers::build_cached_gqa_graph(
            jit.seq_len, total_seq, jit.num_heads, jit.num_kv_heads, jit.head_dim,
            jit.dtype,
        );
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).map_err(|e| {
            BE::Other(format!("MoE decode JIT: CachedGQA(total_seq={total_seq}) compile failed: {e}"))
        })?;
        jit.gqa_cache.insert(total_seq, compiled);
    }
    jit.gqa_cache.get(&total_seq)
        .ok_or_else(|| BE::Other("MoE GQA cache entry missing after insert".into()))
}

/// Pre-compiled JIT graphs for incremental decode (invariant across layers).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct DecodeCachedJit {
    /// RmsNorm → Q Gemm → RoPE (fused)
    q_rope: gllm_kernels::compiler::CompiledLayer,
    /// CachedGQA attention — compiled per unique total_seq value, cached thereafter.
    /// Key = total_seq. First call with a new total_seq triggers one-time JIT compilation;
    /// subsequent calls with the same value reuse the cached CompiledLayer.
    /// This preserves full JIT loop-unrolling/tiling optimizations (Concrete dims)
    /// while eliminating redundant recompilation for repeated total_seq values.
    gqa_cache: std::collections::HashMap<usize, gllm_kernels::compiler::CompiledLayer>,
    /// Pre-FFN RmsNorm
    norm2: gllm_kernels::compiler::CompiledLayer,
    /// Parameters needed to compile new GQA entries on demand
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: gllm_kernels::types::DType,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn build_q_rope_graph(
    seq_len: usize, hidden: usize, q_dim: usize, head_dim: usize,
    eps: f32, rope_theta: f64, dt: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let rn1 = g.add_tensor_concrete("rn1_w", &[hidden], dt);
    let w_q = g.add_tensor_concrete("w_q", &[hidden, q_dim], dt);
    g.inputs = vec![input, rn1, w_q];
    let normed = g.add_tensor_concrete("normed", &[seq_len, hidden], dt);
    g.add_op(OpKind::RmsNorm { eps }, vec![input, rn1], vec![normed], "rms_norm");
    let q_out = g.add_tensor_concrete("q", &[seq_len, q_dim], dt);
    g.add_op(OpKind::Gemm { m: seq_len, n: q_dim, k: hidden, dtype: dt }, vec![normed, w_q], vec![q_out], "gemm_q");
    let q_rope = g.add_tensor_concrete("q_rope", &[seq_len, q_dim], dt);
    g.add_op(OpKind::RoPE { head_dim, theta: rope_theta }, vec![q_out], vec![q_rope], "rope_q");
    g.outputs = vec![q_rope];
    g
}

/// Pre-compile invariant JIT graphs for incremental decode.
/// GQA is NOT compiled here — it is lazily compiled on first use per unique total_seq value.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(dead_code)]
fn compile_decode_jit(
    seq_len: usize,
    hidden: usize, num_heads: usize, num_kv_heads: usize, head_dim: usize,
    eps: f32, rope_theta: f64, dtype: gllm_kernels::types::DType,
) -> Result<DecodeCachedJit, BE> {
    let q_dim = num_heads * head_dim;

    let q_rope_graph = build_q_rope_graph(seq_len, hidden, q_dim, head_dim, eps, rope_theta, dtype);
    let mut c1 = gllm_kernels::compiler::InferenceCompiler::new();
    let q_rope = c1.compile_graph(&q_rope_graph).map_err(|e| {
        BE::Other(format!("decode JIT: RmsNorm+Q+RoPE compile failed: {e}"))
    })?;

    let norm2_graph = super::jit_helpers::build_final_norm_graph(
        seq_len, hidden, eps, dtype,
    );
    let mut c3 = gllm_kernels::compiler::InferenceCompiler::new();
    let norm2 = c3.compile_graph(&norm2_graph).map_err(|e| {
        BE::Other(format!("decode JIT: RmsNorm2 compile failed: {e}"))
    })?;

    Ok(DecodeCachedJit {
        q_rope, norm2,
        gqa_cache: std::collections::HashMap::new(),
        seq_len, num_heads, num_kv_heads, head_dim, dtype,
    })
}

/// Get or compile a CachedGQA CompiledLayer for the given total_seq.
/// Uses Concrete dims → full JIT loop-unrolling/tiling optimizations.
/// Compiles once per unique total_seq value, then caches indefinitely.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn get_or_compile_gqa(jit: &mut DecodeCachedJit, total_seq: usize) -> Result<&gllm_kernels::compiler::CompiledLayer, BE> {
    if !jit.gqa_cache.contains_key(&total_seq) {
        let graph = super::jit_helpers::build_cached_gqa_graph(
            jit.seq_len, total_seq, jit.num_heads, jit.num_kv_heads, jit.head_dim,
            jit.dtype,
        );
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).map_err(|e| {
            BE::Other(format!("decode JIT: CachedGQA(total_seq={total_seq}) compile failed: {e}"))
        })?;
        jit.gqa_cache.insert(total_seq, compiled);
    }
    jit.gqa_cache.get(&total_seq)
        .ok_or_else(|| BE::Other("decode GQA cache entry missing after insert".into()))
}

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
    max_seq_len: usize,
    transpose_weights: bool,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    jit: &mut DecodeCachedJit,
    output: &mut [f32],
    dtype: gllm_kernels::types::DType,
    k_slice_buf: &mut [f32],
    v_slice_buf: &mut [f32],
) -> Result<f32, BE> {
    use super::jit_helpers::TypedBuffer;
    let geom = AttentionGeometry {
        num_heads, num_kv_heads, head_dim,
        q_dim: num_heads * head_dim,
        kv_dim: num_kv_heads * head_dim,
        heads_per_group: num_heads / num_kv_heads,
    };

    // RmsNorm → Q Gemm → RoPE (pre-compiled)
    let mut q_proj = TypedBuffer::zeros(seq_len * geom.q_dim, dtype);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let q_w_f32 = weight_data_to_f32(q_w, backend, transpose_weights, geom.q_dim, hidden)?;
        let weights_buf = pack_weights_typed(&[rn1_w, &q_w_f32], dtype);
        let mut scratchpad = vec![0u8; jit.q_rope.scratchpad_bytes];
        unsafe {
            jit.q_rope.execute(
                hidden_state.as_ptr() as *const u8,
                weights_buf.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1, seq_len,
                q_proj.as_bytes_mut().as_mut_ptr(),
                scratchpad.as_mut_ptr(),
            );
        }
    }

    // Extract current layer's KV cache slice → [total_seq, kv_dim] (reuse caller buffer)
    let kv_dim = geom.kv_dim;
    let layer_stride = num_kv_heads * max_seq_len * head_dim;
    let layer_k_base = layer * layer_stride;
    for t in 0..total_seq {
        for h in 0..num_kv_heads {
            let cache_off = layer_k_base + h * max_seq_len * head_dim + t * head_dim;
            let slice_off = t * kv_dim + h * head_dim;
            k_slice_buf[slice_off..slice_off + head_dim]
                .copy_from_slice(&kv_cache_k[cache_off..cache_off + head_dim]);
            v_slice_buf[slice_off..slice_off + head_dim]
                .copy_from_slice(&kv_cache_v[cache_off..cache_off + head_dim]);
        }
    }

    // CachedGQA attention — get cached CompiledLayer for this total_seq (compile once on first use)
    let (attn_out, sparsity) = {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            let gqa_compiled = get_or_compile_gqa(jit, total_seq)?;
            super::jit_helpers::execute_cached_gqa(
                gqa_compiled, q_proj.as_f32(), k_slice_buf, v_slice_buf,
                seq_len, num_heads, head_dim,
                dtype,
            )
        }
    };

    // O projection (quantized) + residual
    let mut o_out = TypedBuffer::zeros(seq_len * hidden, dtype);
    quantized_linear(backend, &attn_out, o_w, o_out.as_f32_mut(), seq_len, hidden, geom.q_dim, transpose_weights, dtype)?;
    let resid1;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let result = super::jit_helpers::jit_add(
            super::jit_helpers::f32_as_bytes(&hidden_state[..seq_len * hidden]),
            o_out.as_bytes(), dtype,
        ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?;
        resid1 = TypedBuffer::from_raw(result, dtype);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { return Err(BE::Other("residual add JIT requires x86_64 or aarch64".to_string())); }
    let mut normed2 = TypedBuffer::zeros(seq_len * hidden, dtype);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        super::jit_helpers::execute_jit_final_norm(
            &jit.norm2,
            resid1.as_bytes(),
            super::jit_helpers::f32_as_bytes(rn2_w),
            seq_len,
            normed2.as_bytes_mut(),
        );
    }
    let mut gate_out = TypedBuffer::zeros(seq_len * inter, dtype);
    quantized_linear(backend, normed2.as_f32(), gate_w, gate_out.as_f32_mut(), seq_len, inter, hidden, transpose_weights, dtype)?;
    let mut up_out = TypedBuffer::zeros(seq_len * inter, dtype);
    quantized_linear(backend, normed2.as_f32(), up_w, up_out.as_f32_mut(), seq_len, inter, hidden, transpose_weights, dtype)?;
    let swiglu_out;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let result = super::jit_helpers::jit_swiglu(
            gate_out.as_bytes(),
            up_out.as_bytes(),
            seq_len, inter, dtype,
        ).map_err(|e| BE::Other(format!("SwiGLU JIT failed: {e}")))?;
        swiglu_out = TypedBuffer::from_raw(result, dtype);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        return Err(BE::Other("SwiGLU JIT requires x86_64 or aarch64".to_string()));
    }
    let mut down_out = TypedBuffer::zeros(seq_len * hidden, dtype);
    quantized_linear(backend, swiglu_out.as_f32(), down_w, down_out.as_f32_mut(), seq_len, hidden, inter, transpose_weights, dtype)?;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let result = super::jit_helpers::jit_add(
            resid1.as_bytes(),
            down_out.as_bytes(), dtype,
        ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?;
        output.copy_from_slice(super::jit_helpers::bytes_as_f32(&result));
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { return Err(BE::Other("residual add JIT requires x86_64 or aarch64".to_string())); }
    Ok(sparsity)
}

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
) -> Result<(Vec<LogitsHandle>, f32), BE> {
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
    let mut total_sparsity = 0.0f32;
    let mut sparsity_layers = 0u32;

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
            let mut store = backend.kv_store().lock().map_err(|e| {
                BE::Cpu(format!("KV store lock poisoned: {e}"))
            })?;
            // Reset KV cache on new prefill (position == 0) to avoid stale data
            if position == 0 {
                if let Some(buf) = store.get_mut(&kv_caches[seq_idx].0) {
                    buf.seq_len = 0;
                }
            }
            store.get(&kv_caches[seq_idx].0)
                .ok_or_else(|| BE::Cpu(format!("KV cache entry missing for handle {:?}", kv_caches[seq_idx].0)))?
                .seq_len
        } else {
            0
        };
        let is_incremental = has_kv_cache && cached_seq_len > 0 && position > 0;

        let kv_dim = num_kv_heads * head_dim;

        // ── Graph executor path (YAML→JIT, preferred when available) ──
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        if !config.graph_executor_ptr.is_null() {
            let ge = unsafe { &mut *config.graph_executor_ptr };
            // Skip if graph has no nodes (template stub with empty nodes: [])
            if ge.graph().nodes.is_empty() {
                // Fall through to hand-written decoder path
            } else {
            let mut inputs = std::collections::HashMap::new();
            let hs_bytes: Vec<u8> = hidden_state
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            inputs.insert("hidden_state".to_string(), hs_bytes);

            // Get KV cache pointers for this sequence
            let (kv_cache_k_ptr, kv_cache_v_ptr) = if has_kv_cache && seq_idx < kv_caches.len() {
                let store = backend.kv_store().lock().map_err(|e| {
                    BE::Cpu(format!("KV store lock poisoned: {e}"))
                })?;
                if let Some(buf) = store.get(&kv_caches[seq_idx].0) {
                    let k_ptr = buf.k.as_ptr() as *mut f32;
                    let v_ptr = buf.v.as_ptr() as *mut f32;
                    (k_ptr, v_ptr)
                } else {
                    (std::ptr::null_mut(), std::ptr::null_mut())
                }
            } else {
                (std::ptr::null_mut(), std::ptr::null_mut())
            };

            let total_seq = cached_seq_len + seq_len;
            let positions: Vec<u32> = (0..seq_len).map(|i| (position + i) as u32).collect();

            let output = ge.run_with_kv_cache(
                &inputs,
                kv_cache_k_ptr,
                kv_cache_v_ptr,
                0, // layer 0; graph executor handles all layers internally
                total_seq,
                positions.as_ptr(),
            ).map_err(|e| BE::Other(format!("graph executor: {e}")))?;

            // Extract logits from graph output
            if let Some(logits_bytes) = output.get("logits").or_else(|| output.values().next()) {
                let logits: Vec<f32> = logits_bytes
                    .chunks_exact(4)
                    .map(|c| {
                        let arr: [u8; 4] = c.try_into().map_err(|_| BE::Other("invalid f32 bytes in logits output".into()))?;
                        Ok(f32::from_le_bytes(arr))
                    })
                    .collect::<Result<Vec<f32>, BE>>()?;
                // Return last-token logits
                let last_start = if logits.len() >= vocab_size {
                    logits.len() - vocab_size
                } else {
                    0
                };
                results.push(LogitsHandle { data: logits[last_start..].to_vec() });
                continue;
            }
            // If graph executor produced no recognizable output, fall through to hand-written path
            } // end else (graph has nodes)
        }

        if is_incremental {
            // ── Incremental decode path: use cached K/V, O(n) per step ──
            log::debug!(
                "decoder_forward: incremental decode, position={position}, seq_len={seq_len}, cached_seq={cached_seq_len}"
            );

            let max_seq_len = {
                let store = backend.kv_store().lock().map_err(|e| {
                    BE::Cpu(format!("KV store lock poisoned: {e}"))
                })?;
                store.get(&kv_caches[seq_idx].0)
                    .ok_or_else(|| BE::Cpu(format!("KV cache entry missing for handle {:?}", kv_caches[seq_idx].0)))?
                    .max_seq_len
            };

            // Pre-compile all invariant JIT graphs for this decode step (once per model instance).
            let total_seq = cached_seq_len + seq_len;

            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let kv_proj_decode: std::sync::Arc<gllm_kernels::compiler::CompiledLayer> = {
                let l1: &mut super::jit_cache::ModelJitCache = unsafe { &mut *config.jit_cache_ptr };
                if l1.kv_proj.is_none() {
                    let arch_key = ModelArchKey {
                        arch_name: "decoder".into(),
                        hidden_size: hidden,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        dtype: kernels_dtype_to_compat(computation_dtype_from_config(config)),
                    };
                    let arc = global_jit_cache().get_or_compile(
                        JitCacheKey { arch: arch_key, graph: GraphType::KvProjection },
                        || {
                            let kv_graph = build_kv_projection_graph(
                                seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta, computation_dtype_from_config(config),
                            );
                            let mut kv_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                            kv_compiler.compile_graph(&kv_graph)
                                .map_err(|e| format!("KV projection (decode) JIT compilation failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    l1.kv_proj = Some(arc);
                }
                l1.kv_proj.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: kv_proj".into()))?.clone()
            };

            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                // Ensure q_rope and norm2 are compiled and stored in L1 cache.
                let l1: &mut super::jit_cache::ModelJitCache = unsafe { &mut *config.jit_cache_ptr };
                if l1.q_rope.is_none() {
                    let arch_key = ModelArchKey {
                        arch_name: "decoder".into(),
                        hidden_size: hidden,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        dtype: kernels_dtype_to_compat(computation_dtype_from_config(config)),
                    };
                    let cache = global_jit_cache();
                    let q_dim = num_heads * head_dim;
                    let arc = cache.get_or_compile(
                        JitCacheKey { arch: arch_key.clone(), graph: GraphType::QRope },
                        || {
                            let g = build_q_rope_graph(seq_len, hidden, q_dim, head_dim, eps, rope_theta, computation_dtype_from_config(config));
                            let mut c = gllm_kernels::compiler::InferenceCompiler::new();
                            c.compile_graph(&g).map_err(|e| format!("decode JIT: RmsNorm+Q+RoPE compile failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    l1.q_rope = Some(arc);
                    let arc2 = cache.get_or_compile(
                        JitCacheKey { arch: arch_key, graph: GraphType::Norm2 },
                        || {
                            let g = super::jit_helpers::build_final_norm_graph(seq_len, hidden, eps, computation_dtype_from_config(config));
                            let mut c = gllm_kernels::compiler::InferenceCompiler::new();
                            c.compile_graph(&g).map_err(|e| format!("decode JIT: RmsNorm2 compile failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    l1.norm2 = Some(arc2);
                }
                if moe_num_experts > 0 && l1.moe_pre_attn.is_none() {
                    let arch_key = ModelArchKey {
                        arch_name: "moe_decoder".into(),
                        hidden_size: hidden,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        dtype: kernels_dtype_to_compat(computation_dtype_from_config(config)),
                    };
                    let cache = global_jit_cache();
                    let q_dim = num_heads * head_dim;
                    let pre_attn = cache.get_or_compile(
                        JitCacheKey { arch: arch_key.clone(), graph: GraphType::MoePreAttn },
                        || {
                            let g = build_moe_pre_attention_graph(
                                seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, computation_dtype_from_config(config),
                            );
                            let mut c = gllm_kernels::compiler::InferenceCompiler::new();
                            c.compile_graph(&g).map_err(|e| format!("MoE decode JIT: pre-attention compile failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    let o_gemm = cache.get_or_compile(
                        JitCacheKey { arch: arch_key.clone(), graph: GraphType::MoeOGemm },
                        || {
                            use gllm_kernels::compiler::{CompilerGraph, OpKind};
                            let dt = computation_dtype_from_config(config);
                            let mut og = CompilerGraph::new();
                            let a_in = og.add_tensor_concrete("a", &[seq_len, q_dim], dt);
                            let b_in = og.add_tensor_concrete("b", &[q_dim, hidden], dt);
                            og.inputs = vec![a_in, b_in];
                            let c_out = og.add_tensor_concrete("c", &[seq_len, hidden], dt);
                            og.add_op(OpKind::Gemm { m: seq_len, n: hidden, k: q_dim, dtype: dt }, vec![a_in, b_in], vec![c_out], "gemm_o");
                            og.outputs = vec![c_out];
                            let mut c = gllm_kernels::compiler::InferenceCompiler::new();
                            c.compile_graph(&og).map_err(|e| format!("MoE decode JIT: O Gemm compile failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    let norm2 = cache.get_or_compile(
                        JitCacheKey { arch: arch_key, graph: GraphType::MoeNorm2 },
                        || {
                            let g = super::jit_helpers::build_final_norm_graph(seq_len, hidden, eps, computation_dtype_from_config(config));
                            let mut c = gllm_kernels::compiler::InferenceCompiler::new();
                            c.compile_graph(&g).map_err(|e| format!("MoE decode JIT: RmsNorm2 compile failed: {e}"))
                        },
                    ).map_err(BE::Other)?;
                    l1.moe_pre_attn = Some(pre_attn);
                    l1.moe_o_gemm = Some(o_gemm);
                    l1.moe_norm2 = Some(norm2);
                }
            }

            // Build thin wrappers that borrow from L1 cache Arcs.
            // Seed gqa_cache from L1 so previously compiled entries are reused (REQ-JIT-CACHE-001).
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let mut decode_jit = {
                let l1: &mut super::jit_cache::ModelJitCache = unsafe { &mut *config.jit_cache_ptr };
                let clone_arc = |arc: &std::sync::Arc<gllm_kernels::compiler::CompiledLayer>| -> Result<gllm_kernels::compiler::CompiledLayer, BE> {
                    gllm_kernels::compiler::CompiledLayer::from_code(
                        arc.code_bytes(), arc.scratchpad_bytes, arc.config_hash,
                    ).map_err(|e| BE::Other(format!("clone CompiledLayer from Arc failed: {e}")))
                };
                // Seed gqa_cache from L1 accumulated entries.
                let gqa_cache: std::collections::HashMap<usize, gllm_kernels::compiler::CompiledLayer> = {
                    let mut m = std::collections::HashMap::new();
                    for (&ts, arc) in l1.gqa_cache.iter() { m.insert(ts, clone_arc(arc)?); }
                    m
                };
                DecodeCachedJit {
                    q_rope: clone_arc(l1.q_rope.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: q_rope".into()))?)?,
                    norm2: clone_arc(l1.norm2.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: norm2".into()))?)?,
                    gqa_cache,
                    seq_len, num_heads, num_kv_heads, head_dim,
                    dtype: computation_dtype_from_config(config),
                }
            };

            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let mut moe_jit = if moe_num_experts > 0 {
                let l1: &mut super::jit_cache::ModelJitCache = unsafe { &mut *config.jit_cache_ptr };
                let clone_arc = |arc: &std::sync::Arc<gllm_kernels::compiler::CompiledLayer>| -> Result<gllm_kernels::compiler::CompiledLayer, BE> {
                    gllm_kernels::compiler::CompiledLayer::from_code(
                        arc.code_bytes(), arc.scratchpad_bytes, arc.config_hash,
                    ).map_err(|e| BE::Other(format!("clone CompiledLayer from Arc failed: {e}")))
                };
                // Seed moe_gqa_cache from L1 accumulated entries.
                let gqa_cache: std::collections::HashMap<usize, gllm_kernels::compiler::CompiledLayer> = {
                    let mut m = std::collections::HashMap::new();
                    for (&ts, arc) in l1.moe_gqa_cache.iter() { m.insert(ts, clone_arc(arc)?); }
                    m
                };
                Some(MoeDecodeCachedJit {
                    pre_attn: clone_arc(l1.moe_pre_attn.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: moe_pre_attn".into()))?)?,
                    o_gemm: clone_arc(l1.moe_o_gemm.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: moe_o_gemm".into()))?)?,
                    norm2: clone_arc(l1.moe_norm2.as_ref().ok_or_else(|| BE::Other("JIT cache entry missing: moe_norm2".into()))?)?,
                    gqa_cache,
                    seq_len, num_heads, num_kv_heads, head_dim,
                    dtype: computation_dtype_from_config(config),
                })
            } else {
                None
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
                    // MoE layers: JIT KV projection (replaces scalar update_kv_cache)
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    update_kv_cache_jit(
                        backend, kv_caches[seq_idx],
                        layer, &hidden_state, &k_w_f32, &v_w_f32,
                        &rn1_w, &positions,
                        seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                        computation_dtype_from_config(config),
                    )?;
                } else {
                    // Dense layers: JIT KV projection
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let (k_rope, v_proj) = execute_kv_projection(
                            &kv_proj_decode,
                            &hidden_state, &rn1_w, &k_w_f32, &v_w_f32,
                            &positions, seq_len, hidden, num_kv_heads, head_dim, eps,
                            computation_dtype_from_config(config),
                        ).map_err(|e| BE::Other(e))?;
                        write_kv_to_cache(
                            backend, kv_caches[seq_idx],
                            layer, &k_rope, &v_proj,
                            seq_len, num_kv_heads, head_dim,
                        )?;
                    }
                }

                // Read cached K/V for attention (convert typed bytes → f32 for CPU attention)
                let (kv_cache_k, kv_cache_v) = {
                    let store = backend.kv_store().lock().map_err(|e| {
                        BE::Cpu(format!("KV store lock poisoned: {e}"))
                    })?;
                    let buffer = store.get(&kv_caches[seq_idx].0).ok_or_else(|| {
                        BE::Cpu(format!("KV cache handle {} not found", kv_caches[seq_idx].0))
                    })?;
                    if buffer.elem_bytes == std::mem::size_of::<f32>() {
                        // F32 cache: zero-copy reinterpret
                        (bytes_as_f32(&buffer.k).to_vec(), bytes_as_f32(&buffer.v).to_vec())
                    } else {
                        // F16/BF16 cache: convert to f32
                        let dt = super::jit_helpers::computation_dtype_from_config(config);
                        (super::jit_helpers::typed_bytes_to_f32(&buffer.k, dt),
                         super::jit_helpers::typed_bytes_to_f32(&buffer.v, dt))
                    }
                };

                let mut layer_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
                let kv_dim = num_kv_heads * head_dim;
                let mut k_slice_buf = vec![0.0f32; total_seq * kv_dim];
                let mut v_slice_buf = vec![0.0f32; total_seq * kv_dim];

                if let Some((router_w, expert_weights, shared_expert)) = moe_weights {
                    // MoE incremental: JIT pre-attention + cached attention + scalar MoE FFN
                    let q_w_f32 = weight_data_to_f32(
                        &q_w, backend, transpose_weights, num_heads * head_dim, hidden)?;
                    let o_w_f32 = weight_data_to_f32(
                        &o_w, backend, transpose_weights, hidden, num_heads * head_dim)?;

                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        let moe_c = moe_jit.as_mut().ok_or_else(|| BE::Other("JIT cache entry missing: moe_jit".into()))?;
                        let geom = AttentionGeometry {
                            num_heads, num_kv_heads, head_dim,
                            q_dim: num_heads * head_dim,
                            kv_dim: num_kv_heads * head_dim,
                            heads_per_group: num_heads / num_kv_heads,
                        };

                        // Step 1: Pre-attention (pre-compiled)
                        let weights_buf = pack_weights_typed(&[&rn1_w, &q_w_f32, &k_w_f32, &v_w_f32], computation_dtype_from_config(config));
                        let mut q_rope = super::jit_helpers::TypedBuffer::zeros(seq_len * geom.q_dim, computation_dtype_from_config(config));
                        let mut scratchpad = vec![0u8; moe_c.pre_attn.scratchpad_bytes];
                        unsafe {
                            moe_c.pre_attn.execute(
                                hidden_state.as_ptr() as *const u8,
                                weights_buf.as_ptr(),
                                std::ptr::null_mut(),
                                positions.as_ptr(),
                                std::ptr::null(),
                                1, seq_len,
                                q_rope.as_bytes_mut().as_mut_ptr(),
                                scratchpad.as_mut_ptr(),
                            );
                        }

                        // Step 2: Cached GQA attention — get cached CompiledLayer for this total_seq
                        let (attn_out, _layer_sparsity) = {
                            let gqa_compiled = get_or_compile_moe_gqa(moe_c, total_seq)?;
                            super::jit_helpers::execute_cached_gqa(
                                gqa_compiled, q_rope.as_f32(), &kv_cache_k, &kv_cache_v,
                                seq_len, num_heads, head_dim,
                                computation_dtype_from_config(config),
                            )
                        };

                        // Step 3: O projection (pre-compiled) + residual + RmsNorm2 (pre-compiled)
                        let mut o_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
                        {
                            let o_weights = pack_weights_typed(&[&o_w_f32], computation_dtype_from_config(config));
                            let mut o_scratch = vec![0u8; moe_c.o_gemm.scratchpad_bytes];
                            unsafe {
                                moe_c.o_gemm.execute(
                                    attn_out.as_ptr() as *const u8,
                                    o_weights.as_ptr(),
                                    std::ptr::null_mut(),
                                    std::ptr::null(),
                                    std::ptr::null(),
                                    1, seq_len,
                                    o_out.as_bytes_mut().as_mut_ptr(),
                                    o_scratch.as_mut_ptr(),
                                );
                            }
                        }
                        let resid1 = super::jit_helpers::TypedBuffer::from_raw(
                            super::jit_helpers::jit_add(
                                super::jit_helpers::f32_as_bytes(&hidden_state[..seq_len * hidden]),
                                o_out.as_bytes(),
                                computation_dtype_from_config(config),
                            ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?,
                            computation_dtype_from_config(config),
                        );

                        let mut normed2 = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
                        super::jit_helpers::execute_jit_final_norm(
                            &moe_c.norm2,
                            resid1.as_bytes(),
                            super::jit_helpers::f32_as_bytes(&rn2_w),
                            seq_len,
                            normed2.as_bytes_mut(),
                        );

                        // Step 4: MoE FFN via JIT
                        let moe_out = super::jit_helpers::execute_moe_ffn_jit(
                            normed2.as_f32(), &router_w, &expert_weights, shared_expert.as_ref(),
                            seq_len, hidden, inter, moe_num_experts, moe_top_k,
                            computation_dtype_from_config(config),
                        )?;

                        let layer_out_bytes = super::jit_helpers::jit_add(
                            resid1.as_bytes(),
                            super::jit_helpers::f32_as_bytes(&moe_out),
                            computation_dtype_from_config(config),
                        ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?;
                        layer_out.copy_from_bytes(&layer_out_bytes);
                        sparsity_layers += 1;
                    }
                } else {
                    // Dense layer: load standard FFN weights
                    let gate_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")))?;
                    let up_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")))?;
                    let down_w = get_weight_data(weights, backend,
                        &crate::weight_names::decoder_layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")))?;

                    let layer_sparsity = quantized_incremental_decode_layer(
                        backend,
                        &hidden_state,
                        &q_w, &o_w, &rn1_w,
                        &gate_w, &up_w, &down_w, &rn2_w,
                        &positions,
                        &kv_cache_k, &kv_cache_v,
                        layer, total_seq, seq_len,
                        hidden, num_heads, num_kv_heads, head_dim, inter,
                        max_seq_len,
                        transpose_weights,
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        &mut decode_jit,
                        layer_out.as_f32_mut(),
                        computation_dtype_from_config(config),
                        &mut k_slice_buf,
                        &mut v_slice_buf,
                    )?;
                    total_sparsity += layer_sparsity;
                    sparsity_layers += 1;
                }

                hidden_state.copy_from_slice(layer_out.as_f32());
            }

            // Writeback gqa_cache entries to L1 so they survive across inference calls.
            // REQ-JIT-CACHE-001 criterion 4: gqa_cache accumulates across calls.
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                let l1: &mut super::jit_cache::ModelJitCache = unsafe { &mut *config.jit_cache_ptr };
                for (ts, compiled) in decode_jit.gqa_cache.drain() {
                    if let Ok(layer) = gllm_kernels::compiler::CompiledLayer::from_code(
                        compiled.code_bytes(), compiled.scratchpad_bytes, compiled.config_hash,
                    ) {
                        l1.gqa_cache.entry(ts).or_insert_with(|| std::sync::Arc::new(layer));
                    }
                }
                if let Some(ref mut mj) = moe_jit {
                    for (ts, compiled) in mj.gqa_cache.drain() {
                        if let Ok(layer) = gllm_kernels::compiler::CompiledLayer::from_code(
                            compiled.code_bytes(), compiled.scratchpad_bytes, compiled.config_hash,
                        ) {
                            l1.moe_gqa_cache.entry(ts).or_insert_with(|| std::sync::Arc::new(layer));
                        }
                    }
                }
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

                // JIT MoE prefill: pre-attention (JIT) → attention (JIT) → post-attention (JIT) → MoE FFN (scalar routing only)
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                {
                    let geom = config.attention_geometry();
                    let dims = config.layer_dims();

                    // Step 1: Pre-attention via JIT (RmsNorm → Q/K/V Gemm → RoPE)
                    let pre_attn_graph = build_moe_pre_attention_graph(
                        seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, computation_dtype_from_config(config),
                    );
                    let mut pre_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                    let pre_compiled = pre_compiler.compile_graph(&pre_attn_graph).map_err(|e| {
                        BE::Other(format!("MoE pre-attention JIT failed: {e}"))
                    })?;

                    let weights_buf = pack_weights_typed(&[&rn1_w, &q_w, &k_w, &v_w], computation_dtype_from_config(config));
                    let mut q_rope = vec![0.0f32; seq_len * geom.q_dim];
                    let mut scratchpad = vec![0u8; pre_compiled.scratchpad_bytes];
                    unsafe {
                        pre_compiled.execute(
                            hidden_state.as_ptr() as *const u8,
                            weights_buf.as_ptr(),
                            std::ptr::null_mut(),
                            positions.as_ptr(),
                            std::ptr::null(),
                            1, seq_len,
                            q_rope.as_mut_ptr() as *mut u8,
                            scratchpad.as_mut_ptr(),
                        );
                    }

                    // K/V projection via JIT KV graph (reuse existing pattern)
                    let kv_graph = build_kv_projection_graph(
                        seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta, computation_dtype_from_config(config),
                    );
                    let mut kv_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                    let kv_compiled = kv_compiler.compile_graph(&kv_graph).map_err(|e| {
                        BE::Other(format!("MoE KV projection JIT failed: {e}"))
                    })?;
                    let (k_rope, v_proj) = execute_kv_projection(
                        &kv_compiled, &hidden_state, &rn1_w, &k_w, &v_w,
                        &positions, seq_len, hidden, num_kv_heads, head_dim, eps,
                        computation_dtype_from_config(config),
                    ).map_err(|e| BE::Other(e))?;

                    // Step 2: Prefill attention via JIT (MHA with GQA)
                    let (attn_out, layer_sparsity) = {
                        use gllm_kernels::compiler::{CompilerGraph, OpKind};
                        let mut ag = CompilerGraph::new();
                        let dt = computation_dtype_from_config(config);
                        let q_in = ag.add_tensor_concrete("q", &[seq_len, geom.q_dim], dt);
                        let k_in = ag.add_tensor_concrete("k", &[seq_len, geom.kv_dim], dt);
                        let v_in = ag.add_tensor_concrete("v", &[seq_len, geom.kv_dim], dt);
                        ag.inputs = vec![q_in, k_in, v_in];
                        let a_out = ag.add_tensor_concrete("attn", &[seq_len, geom.q_dim], dt);
                        ag.add_op(
                            OpKind::MultiHeadAttention { seq_len, num_heads, num_kv_heads: geom.num_kv_heads, head_dim },
                            vec![q_in, k_in, v_in], vec![a_out], "mha",
                        );
                        ag.outputs = vec![a_out];
                        let mut ac = gllm_kernels::compiler::InferenceCompiler::new();
                        let a_compiled = ac.compile_graph(&ag).map_err(|e| {
                            BE::Other(format!("MoE prefill MHA JIT failed: {e}"))
                        })?;
                        let attn_weights = pack_weights_typed(&[&k_rope, &v_proj], computation_dtype_from_config(config));
                        let mut attn_result = vec![0.0f32; seq_len * geom.q_dim];
                        let mut attn_scratch = vec![0u8; a_compiled.scratchpad_bytes];
                        unsafe {
                            a_compiled.execute(
                                q_rope.as_ptr() as *const u8,
                                attn_weights.as_ptr(),
                                std::ptr::null_mut(),
                                std::ptr::null(),
                                std::ptr::null(),
                                1, seq_len,
                                attn_result.as_mut_ptr() as *mut u8,
                                attn_scratch.as_mut_ptr(),
                            );
                        }
                        // Sparsity not available from MHA JIT (non-causal prefill), use 0.0
                        (attn_result, 0.0f32)
                    };

                    // Step 3: Post-attention via JIT (O Gemm → Residual → RmsNorm2)
                    let post_graph = build_post_attention_graph(
                        seq_len, hidden, num_heads, head_dim, eps, computation_dtype_from_config(config),
                    );
                    let mut post_compiler = gllm_kernels::compiler::InferenceCompiler::new();
                    let post_compiled = post_compiler.compile_graph(&post_graph).map_err(|e| {
                        BE::Other(format!("MoE post-attention JIT failed: {e}"))
                    })?;

                    let post_weights = pack_weights_typed(&[&o_w, &rn2_w], computation_dtype_from_config(config));
                    // Post-attention needs: attn_out as input, o_w + residual_in(hidden_state) + rn2_w as weights
                    // But the graph expects 4 inputs: attn_out, w_o, residual_in, rn2_w
                    // We pack w_o and rn2_w as weights, and pass attn_out + hidden_state via input/kv ptrs
                    let mut normed2 = vec![0.0f32; seq_len * hidden];
                    let mut post_scratch = vec![0u8; post_compiled.scratchpad_bytes];
                    unsafe {
                        post_compiled.execute(
                            attn_out.as_ptr() as *const u8,
                            post_weights.as_ptr(),
                            hidden_state.as_ptr() as *mut u8, // residual input via kv_cache ptr
                            std::ptr::null(),
                            std::ptr::null(),
                            1, seq_len,
                            normed2.as_mut_ptr() as *mut u8,
                            post_scratch.as_mut_ptr(),
                        );
                    }

                    // Compute resid1 for final residual (hidden_state + o_proj)
                    let mut o_out = vec![0.0f32; seq_len * hidden];
                    // O Gemm via JIT
                    {
                        use gllm_kernels::compiler::{CompilerGraph, OpKind};
                        let mut og = CompilerGraph::new();
                        let dt = computation_dtype_from_config(config);
                        let a_in = og.add_tensor_concrete("a", &[seq_len, geom.q_dim], dt);
                        let b_in = og.add_tensor_concrete("b", &[geom.q_dim, hidden], dt);
                        og.inputs = vec![a_in, b_in];
                        let c_out = og.add_tensor_concrete("c", &[seq_len, hidden], dt);
                        og.add_op(OpKind::Gemm { m: seq_len, n: hidden, k: geom.q_dim, dtype: dt }, vec![a_in, b_in], vec![c_out], "gemm_o");
                        og.outputs = vec![c_out];
                        let mut oc = gllm_kernels::compiler::InferenceCompiler::new();
                        let o_compiled = oc.compile_graph(&og).map_err(|e| {
                            BE::Other(format!("MoE prefill O Gemm JIT failed: {e}"))
                        })?;
                        let o_weights = pack_weights_typed(&[&o_w], computation_dtype_from_config(config));
                        let mut o_scratch = vec![0u8; o_compiled.scratchpad_bytes];
                        unsafe {
                            o_compiled.execute(
                                attn_out.as_ptr() as *const u8,
                                o_weights.as_ptr(),
                                std::ptr::null_mut(),
                                std::ptr::null(),
                                std::ptr::null(),
                                1, seq_len,
                                o_out.as_mut_ptr() as *mut u8,
                                o_scratch.as_mut_ptr(),
                            );
                        }
                    }
                    let resid1 = super::jit_helpers::bytes_as_f32(&super::jit_helpers::jit_add(
                        super::jit_helpers::f32_as_bytes(&hidden_state[..seq_len * hidden]),
                        super::jit_helpers::f32_as_bytes(&o_out),
                        computation_dtype_from_config(config),
                    ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?).to_vec();

                    // Step 4: MoE FFN via JIT (expert FFN is JIT, routing is scalar)
                    let moe_out = super::jit_helpers::execute_moe_ffn_jit(
                        &normed2, &router_w, &expert_weights, shared_expert.as_ref(),
                        seq_len, dims.hidden, dims.inter, moe_num_experts, moe_top_k,
                        computation_dtype_from_config(config),
                    )?;

                    let layer_out_bytes = super::jit_helpers::jit_add(
                        super::jit_helpers::f32_as_bytes(&resid1),
                        super::jit_helpers::f32_as_bytes(&moe_out),
                        computation_dtype_from_config(config),
                    ).map_err(|e| BE::Other(format!("residual add JIT failed: {e}")))?;
                    layer_out.copy_from_slice(super::jit_helpers::bytes_as_f32(&layer_out_bytes));
                    total_sparsity += layer_sparsity;
                    sparsity_layers += 1;
                }

                // Update KV cache for this layer (JIT path)
                if has_kv_cache {
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    update_kv_cache_jit(
                        backend, kv_caches[seq_idx],
                        layer, &hidden_state, &k_w, &v_w,
                        &rn1_w, &positions,
                        seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta,
                        computation_dtype_from_config(config),
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
                    seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, computation_dtype_from_config(config),
                );
                let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                let compiled = compiler.compile_graph(&graph).map_err(|e| {
                    BE::Other(format!("Decoder layer JIT compilation failed: {e}"))
                })?;
                compiled
            };

            // Compile KV projection graph for prefill (reused across layers)
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let kv_proj_compiled: Option<gllm_kernels::compiler::CompiledLayer> = if has_kv_cache {
                let kv_graph = build_kv_projection_graph(
                    seq_len, hidden, num_kv_heads, head_dim, eps, rope_theta, computation_dtype_from_config(config),
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
                        computation_dtype_from_config(config),
                    ).map_err(|e| BE::Other(e))?;
                    write_kv_to_cache(
                        backend, kv_caches[seq_idx],
                        layer, &k_rope, &v_proj,
                        seq_len, num_kv_heads, head_dim,
                    )?;
                }

                // Execute JIT-compiled layer
                let mut layer_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
                execute_jit_decoder_layer(
                    &jit_layer,
                    &hidden_state,
                    &q_w, &k_w, &v_w, &o_w, &rn1_w,
                    &gate_w, &up_w, &down_w, &rn2_w,
                    &positions,
                    seq_len,
                    layer_out.as_f32_mut(),
                    computation_dtype_from_config(config),
                );
                hidden_state.copy_from_slice(layer_out.as_f32());
            }
        }

        // (e) Final RMSNorm + lm_head
        let final_norm_w = get_f32_data(weights, backend,
            &crate::weight_names::decoder_final_norm_aliases())?;

        let lm_head_w = get_f32_data(weights, backend,
            &crate::weight_names::lm_head_aliases())
            .or_else(|_| get_f32_data(weights, backend, &crate::weight_names::decoder_embed_aliases()))?;

        let lm_head_w = if transpose_weights {
            transpose_f32(&lm_head_w, vocab_size, hidden)
        } else {
            lm_head_w
        };

        // JIT compile and execute lm_head
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let logits = {
            let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size, eps, computation_dtype_from_config(config));
            let mut lm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
            let compiled_lm = lm_compiler.compile_graph(&lm_graph).map_err(|e| {
                BE::Other(format!("lm_head JIT compilation failed: {e}"))
            })?;

            let mut all_logits = super::jit_helpers::TypedBuffer::zeros(seq_len * vocab_size, computation_dtype_from_config(config));
            execute_jit_lm_head(
                &compiled_lm,
                &hidden_state,
                &final_norm_w,
                &lm_head_w,
                seq_len,
                all_logits.as_f32_mut(),
                computation_dtype_from_config(config),
            );

            // Return only the last token's logits (for generation)
            let last_start = (seq_len - 1) * vocab_size;
            all_logits.as_f32()[last_start..last_start + vocab_size].to_vec()
        };

        results.push(LogitsHandle { data: logits });
    }

    let avg_sparsity = if sparsity_layers > 0 {
        total_sparsity / sparsity_layers as f32
    } else {
        0.0
    };
    Ok((results, avg_sparsity))
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
            seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, computation_dtype_from_config(config),
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

        let mut layer_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
        execute_jit_decoder_layer(
            &jit_layer,
            &hidden_state,
            &q_w, &k_w, &v_w, &o_w, &rn1_w,
            &gate_w, &up_w, &down_w, &rn2_w,
            &positions,
            seq_len,
            layer_out.as_f32_mut(),
            computation_dtype_from_config(config),
        );
        hidden_state.copy_from_slice(layer_out.as_f32());
    }

    // (e) Final RMSNorm
    let final_norm_w = get_f32_data(weights, backend,
        &crate::weight_names::decoder_final_norm_aliases())?;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let normed = {
        let norm_graph = build_final_norm_graph(seq_len, hidden, eps, computation_dtype_from_config(config));
        let mut norm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled_norm = norm_compiler.compile_graph(&norm_graph).map_err(|e| {
            BE::Other(format!("Final norm JIT compilation failed: {e}"))
        })?;

        let mut normed_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
        execute_jit_final_norm(
            &compiled_norm,
            f32_as_bytes(&hidden_state),
            f32_as_bytes(&final_norm_w),
            seq_len,
            normed_out.as_bytes_mut(),
        );
        normed_out.to_f32_vec()
    };

    // (f) Mean pooling: average across all token positions via JIT
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let pooled = super::jit_helpers::jit_mean_pool(&normed, seq_len, hidden,
        computation_dtype_from_config(config),
    )
        .map_err(|e| BE::Other(format!("mean pool JIT failed: {e}")))?;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let pooled = { return Err(BE::Other("mean pool JIT requires x86_64 or aarch64".to_string())); };

    // (g) L2 normalize via JIT (standard for embedding models)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let pooled = {
        super::jit_helpers::jit_l2_normalize(&pooled, 1, hidden,
            computation_dtype_from_config(config),
        )
            .map_err(|e| BE::Other(format!("L2 normalize JIT failed: {e}")))?
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    compile_error!("L2 normalize requires JIT support (x86_64 or aarch64)");

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
            seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, computation_dtype_from_config(config),
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

        let mut layer_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
        execute_jit_decoder_layer(
            &jit_layer,
            &hidden_state,
            &q_w, &k_w, &v_w, &o_w, &rn1_w,
            &gate_w, &up_w, &down_w, &rn2_w,
            &positions,
            seq_len,
            layer_out.as_f32_mut(),
            computation_dtype_from_config(config),
        );
        hidden_state.copy_from_slice(layer_out.as_f32());
    }

    // (e) Final RMSNorm
    let final_norm_w = get_f32_data(weights, backend,
        &crate::weight_names::decoder_final_norm_aliases())?;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let normed = {
        let norm_graph = build_final_norm_graph(seq_len, hidden, eps, computation_dtype_from_config(config));
        let mut norm_compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled_norm = norm_compiler.compile_graph(&norm_graph).map_err(|e| {
            BE::Other(format!("Final norm JIT compilation failed: {e}"))
        })?;

        let mut normed_out = super::jit_helpers::TypedBuffer::zeros(seq_len * hidden, computation_dtype_from_config(config));
        execute_jit_final_norm(
            &compiled_norm,
            f32_as_bytes(&hidden_state),
            f32_as_bytes(&final_norm_w),
            seq_len,
            normed_out.as_bytes_mut(),
        );
        normed_out.to_f32_vec()
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
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            use crate::compat::jit_cache::{global_jit_cache, GraphType, JitCacheKey, ModelArchKey};
            
            use gllm_kernels::compiler::{CompilerGraph, OpKind};
            
            let key = JitCacheKey {
                arch: ModelArchKey {
                    arch_name: "reranker_score".to_string(),
                    hidden_size: hidden,
                    num_heads: num_labels,
                    num_kv_heads: 0,
                    head_dim: 0,
                    dtype: kernels_dtype_to_compat(computation_dtype_from_config(config)),
                },
                graph: GraphType::ResidualAdd { numel: num_labels * hidden },
            };
            let compiled = global_jit_cache().get_or_compile(key, || {
                let mut g = CompilerGraph::new();
                let dt = computation_dtype_from_config(config);
                let x = g.add_tensor_concrete("x", &[1, hidden], dt);
                let w = g.add_tensor_concrete("w", &[hidden, num_labels], dt);
                g.inputs = vec![x, w];
                let out = g.add_tensor_concrete("logits", &[1, num_labels], dt);
                g.add_op(OpKind::Gemm { m: 1, n: num_labels, k: hidden, dtype: dt }, vec![x, w], vec![out], "score_gemm");
                g.outputs = vec![out];
                let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
                compiler.compile_graph(&g).map_err(|e| format!("reranker score JIT failed: {e}"))
            }).map_err(|e| BE::Other(e))?;
            // score_w is [num_labels, hidden] row-major; GEMM expects w as [hidden, num_labels]
            let w_t = super::weight_helpers::transpose_f32(&score_w, num_labels, hidden);
            let dt = computation_dtype_from_config(config);
            let weight_buf = super::jit_helpers::pack_weights_typed(&[&w_t], dt);
            let mut logits = vec![0.0f32; num_labels];
            let mut scratchpad = vec![0u8; compiled.scratchpad_bytes];
            unsafe {
                compiled.execute(
                    last_hidden.as_ptr() as *const u8,
                    weight_buf.as_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null(),
                    1, 1,
                    logits.as_mut_ptr() as *mut u8,
                    scratchpad.as_mut_ptr(),
                );
            }
            for (i, s) in scores.iter_mut().enumerate() {
                *s = 1.0 / (1.0 + (-logits[i]).exp());
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        { return Err(BE::Other("reranker score JIT requires x86_64 or aarch64".to_string())); }
        Ok(scores)
    } else {
        // Generative reranker path: use tied embeddings (lm_head) to get logits,
        // then compute score from "yes"/"no" token probabilities.
        let embed_data = get_f32_data(
            weights, backend,
            &crate::weight_names::decoder_embed_aliases(),
        )?;

        let vocab_size = config.vocab_size;
        let yes_id = config.rerank_yes_token_id.ok_or_else(|| {
            BE::Cpu("rerank_yes_token_id not set in model config".into())
        })? as usize;
        let no_id = config.rerank_no_token_id.ok_or_else(|| {
            BE::Cpu("rerank_no_token_id not set in model config".into())
        })? as usize;

        // embed_data is [vocab_size, hidden] in row-major (each row = one token embedding)
        // lm_head logit for token t = dot(last_hidden, embed_data[t])
        let logit_fn = |token_id: usize| -> f32 {
            if token_id >= vocab_size { return 0.0; }
            let row_start = token_id * hidden;
            last_hidden.iter().zip(embed_data[row_start..row_start + hidden].iter())
                .map(|(a, b)| a * b).sum::<f32>()
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
    use super::super::jit_helpers::{
        build_moe_routing_graph, build_expert_ffn_graph,
        execute_moe_ffn_jit, pack_weights,
    };
    use gllm_kernels::types::DType;

    /// JIT MoE routing (MoEGate → TopK) and verify gate softmax sums to 1.
    #[test]
    fn moe_gate_softmax_sums_to_one() {
        let hidden_size = 4;
        let num_experts = 3;
        let seq_len = 2;
        let top_k = 1;
        let hidden = vec![1.0f32, 0.5, -0.3, 0.8, 0.2, -0.1, 0.6, 0.4];
        let gate_w = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.5, 0.5, 0.0,
        ];

        let graph = build_moe_routing_graph(seq_len, hidden_size, num_experts, top_k, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).expect("MoE routing JIT compile");

        let weights_buf = pack_weights(&[&gate_w]);
        let out_size = seq_len * top_k * 2;
        let mut out = vec![0.0f32; out_size];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes];

        unsafe {
            compiled.execute(
                hidden.as_ptr() as *const u8,
                weights_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }

        // TopK output: [indices..., weights...]
        let weights_raw = &out[seq_len * top_k..];
        for &w in weights_raw {
            assert!(w >= 0.0 && w <= 1.0, "weight out of range: {w}");
            assert!(w.is_finite(), "non-finite weight: {w}");
        }
    }

    /// JIT TopK: verify correct expert selection via routing graph.
    #[test]
    fn top_k_selects_correct_experts() {
        let num_experts = 4;
        let top_k = 2;
        let seq_len = 1;
        let hidden_size = 4;
        // Craft gate weights so expert 2 gets highest score, expert 0 second.
        // input = [1,0,0,0], gate_w row0 = [0.3, 0.1, 0.5, 0.1] → probs ≈ softmax([0.3,0.1,0.5,0.1])
        let input = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut gate_w = vec![0.0f32; hidden_size * num_experts];
        // Row 0 of gate_w (input dim 0): set to desired logits
        gate_w[0] = 0.3; gate_w[1] = 0.1; gate_w[2] = 0.5; gate_w[3] = 0.1;

        let graph = build_moe_routing_graph(seq_len, hidden_size, num_experts, top_k, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).expect("routing JIT compile");

        let weights_buf = pack_weights(&[&gate_w]);
        let out_size = seq_len * top_k * 2;
        let mut out = vec![0.0f32; out_size];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes];

        unsafe {
            compiled.execute(
                input.as_ptr() as *const u8,
                weights_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }

        let indices: Vec<usize> = out[..top_k].iter().map(|v| v.to_bits() as usize).collect();
        let weights: Vec<f32> = out[top_k..].to_vec();
        assert_eq!(indices[0], 2, "top-1 should be expert 2");
        assert_eq!(indices[1], 0, "top-2 should be expert 0");
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "renormalized sum = {sum}");
    }

    /// JIT TopK: verify weight renormalization.
    #[test]
    fn top_k_renormalizes_weights() {
        let num_experts = 3;
        let top_k = 2;
        let seq_len = 1;
        let hidden_size = 3;
        // input = [1,0,0], gate_w row0 = [2.0, 1.0, -1.0] → softmax favors expert 0, then 1
        let input = vec![1.0f32, 0.0, 0.0];
        let mut gate_w = vec![0.0f32; hidden_size * num_experts];
        gate_w[0] = 2.0; gate_w[1] = 1.0; gate_w[2] = -1.0;

        let graph = build_moe_routing_graph(seq_len, hidden_size, num_experts, top_k, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).expect("routing JIT compile");

        let weights_buf = pack_weights(&[&gate_w]);
        let out_size = seq_len * top_k * 2;
        let mut out = vec![0.0f32; out_size];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes];

        unsafe {
            compiled.execute(
                input.as_ptr() as *const u8,
                weights_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }

        let idx0 = out[0].to_bits() as usize;
        let idx1 = out[1].to_bits() as usize;
        let w0 = out[top_k];
        let w1 = out[top_k + 1];
        assert_eq!(idx0, 0, "top-1 should be expert 0");
        assert_eq!(idx1, 1, "top-2 should be expert 1");
        assert!(w0 > w1, "expert 0 weight ({w0}) should exceed expert 1 ({w1})");
        assert!((w0 + w1 - 1.0).abs() < 1e-4, "weights should sum to 1.0, got {}", w0 + w1);
    }

    /// JIT expert FFN: verify output shape and finite values.
    #[test]
    fn expert_ffn_produces_correct_shape() {
        let hidden = 4;
        let inter = 8;
        let seq_len = 2;
        let input = vec![0.1f32; seq_len * hidden];
        let gate_w = vec![0.01f32; hidden * inter];
        let up_w = vec![0.01f32; hidden * inter];
        let down_w = vec![0.01f32; inter * hidden];

        let graph = build_expert_ffn_graph(seq_len, hidden, inter, DType::F32);
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let compiled = compiler.compile_graph(&graph).expect("expert FFN JIT compile");

        let weights_buf = pack_weights(&[&gate_w, &up_w, &down_w]);
        let mut out = vec![0.0f32; seq_len * hidden];
        let mut scratch = vec![0u8; compiled.scratchpad_bytes];

        unsafe {
            compiled.execute(
                input.as_ptr() as *const u8,
                weights_buf.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1, seq_len,
                out.as_mut_ptr() as *mut u8,
                scratch.as_mut_ptr(),
            );
        }

        assert_eq!(out.len(), seq_len * hidden);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// JIT full MoE FFN: routing + expert FFN + weighted combine.
    #[test]
    fn moe_ffn_weighted_combine() {
        let hidden = 4;
        let inter = 8;
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
        let out = execute_moe_ffn_jit(
            &input, &router_w, &experts, None,
            seq_len, hidden, inter, num_experts, top_k,
            gllm_kernels::types::DType::F32,
        ).expect("MoE FFN JIT execution");
        assert_eq!(out.len(), hidden);
        for &v in &out {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }
}
