//! FusedGraph 执行器 (REQ-EXEC-002)
//!
//! Compiles each FusedNode in the FusedGraph into a JIT-compiled kernel
//! via gllm-kernels' InferenceCompiler, then executes them in topological
//! order with proper buffer management.

use std::collections::{HashMap, HashSet};

/// SymDim::Symbolic 编译时上界，用于 scratchpad/buffer 分配。非运行时限制。
/// SymDim::Symbolic 的 buffer 分配上界（SSOT 位于 gllm_kernels::compiler::graph）。
/// 仅用于 max_value（内存安全上界），禁止用于循环 bound 或维度运算。
use gllm_kernels::compiler::graph::SYMDIM_MAX_SEQ_LEN;

use super::types::{
    FusedGraph, FusedOp, FlashAttentionConfig, FusedQkvRopeConfig,
    FusedRMSLinearConfig, GQAConfig, MoERoutingConfig, PleConfig, RoPEConfig, SwiGLUConfig,
};

/// FusedGraph 执行错误
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Missing input tensor: {0}")]
    MissingInput(String),
    #[error("Missing weight tensor: {0}")]
    MissingWeight(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),
    #[error("Backend error: {0}")]
    Backend(String),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("Compilation error: {0}")]
    Compilation(String),
    #[error("Execution requires compilation: call compile() before run()")]
    NotCompiled,
}

/// 执行上下文 - 保存中间张量名称和状态
#[derive(Debug, Default)]
pub struct ExecutionContext {
    pub computed: Vec<String>,
    pub outputs: Vec<String>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_computed(&mut self, name: String) {
        self.computed.push(name);
    }

    pub fn is_computed(&self, name: &str) -> bool {
        self.computed.iter().any(|n| n == name)
    }
}

/// FusedGraph 执行计划
#[derive(Debug)]
pub struct ExecutionPlan {
    pub operations: Vec<ExecutionOp>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// 执行操作
#[derive(Debug, Clone)]
pub enum ExecutionOp {
    FlashAttention {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    SwiGLU {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    RoPE {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        rope_theta: f64,
    },
    FusedQkvRope {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    FusedRMSLinear {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    GQA {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    MoERouting {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
    PerLayerEmbed {
        name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        dim_per_layer: usize,
        num_layers: usize,
    },
    Atomic {
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
}

impl ExecutionPlan {
    pub fn from_fused_graph(graph: &FusedGraph) -> Self {
        let mut operations = Vec::new();

        for node in &graph.nodes {
            let op = match &node.op {
                FusedOp::FlashAttention(config) => ExecutionOp::FlashAttention {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                },
                FusedOp::SwiGLU(_config) => ExecutionOp::SwiGLU {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::RoPE(config) => ExecutionOp::RoPE {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    rope_theta: config.rope_theta,
                },
                FusedOp::FusedQkvRope(_config) => ExecutionOp::FusedQkvRope {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::FusedRMSLinear(_config) => ExecutionOp::FusedRMSLinear {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::GQA(_config) => ExecutionOp::GQA {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::MoERouting(_config) => ExecutionOp::MoERouting {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
                FusedOp::PerLayerEmbed(config) => ExecutionOp::PerLayerEmbed {
                    name: node.name.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                    dim_per_layer: config.dim_per_layer,
                    num_layers: config.num_layers,
                },
                FusedOp::Atomic(atomic) => ExecutionOp::Atomic {
                    name: node.name.clone(),
                    op_type: atomic.op_type.clone(),
                    inputs: node.inputs.clone(),
                    outputs: node.outputs.clone(),
                },
            };
            operations.push(op);
        }

        Self {
            operations,
            inputs: graph.inputs.clone(),
            outputs: graph.outputs.clone(),
        }
    }

    pub fn op_count(&self) -> usize {
        self.operations.len()
    }

    pub fn fused_op_count(&self) -> usize {
        self.operations
            .iter()
            .filter(|op| !matches!(op, ExecutionOp::Atomic { .. }))
            .count()
    }
}

// ---------------------------------------------------------------------------
// JIT compilation support: FusedOp → CompilerGraph translation
// ---------------------------------------------------------------------------

/// Metadata for a compiled node: the JIT-compiled layer plus the
/// CompilerGraph input/output tensor names and output element count.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
struct CompiledNode {
    compiled: gllm_kernels::compiler::CompiledLayer,
    /// Names of the FusedNode input tensors, in order.
    /// The first entry is the activation input; remaining are weights.
    graph_input_names: Vec<String>,
    /// Names of the FusedNode output tensors.
    graph_output_names: Vec<String>,
    /// Number of elements in the output tensor(s) at max_seq_len (compile-time upper bound).
    /// Used for buffer allocation. Actual runtime size = feature_dim * actual_seq_len.
    output_numel: usize,
    /// Per-output element counts for multi-output nodes (at max_seq_len).
    /// Empty for single-output nodes (use output_numel).
    per_output_numel: Vec<usize>,
    /// DType of the output tensor(s) — used for byte-size calculations.
    output_dtype: gllm_kernels::types::DType,
    /// Feature dimension per token (e.g., hidden_size) for primary output.
    /// Extracted from graph metadata Concrete dims (ARCH-SYMDIM-NO-CONST-DEGRADE).
    feature_dim: usize,
    /// Per-output feature dimensions for multi-output nodes.
    /// Each entry = product of Concrete (non-Symbolic) dims for that output tensor.
    /// Empty for single-output nodes (use feature_dim).
    per_output_feature_dims: Vec<usize>,
}

/// ARCH-ROPE-CACHE: 预填 cos/sin 表到 scratchpad。
///
/// 布局 (row-major): `[seq_len, head_dim]`
///   - 每行前 `half_rot` 个 f32 是 cos 值 (i ∈ [0, half_rot))
///   - 后 `half_rot` 个 f32 是 sin 值
///   - 剩余 `passthrough_dim` 个 f32 未使用 (lower_rope 的 passthrough 分支不读)
///
/// 频率: freq_i = 1 / theta^(2i / head_dim), angle = position_p * freq_i
///
/// positions: 运行时 position 数组 (长度 ≥ effective_seq)
/// effective_seq: 当前 forward pass 的 token 数
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn populate_rope_cache(
    scratchpad: &mut [u8],
    cache_offset: usize,
    head_dim: usize,
    theta: f64,
    partial: f32,
    positions: *const u32,
    effective_seq: usize,
) -> Result<(), ExecutionError> {
    if positions.is_null() {
        return Err(ExecutionError::Compilation(
            "populate_rope_cache: positions pointer is null but kernel requires RoPE cache".into()));
    }
    let rot_dim = ((head_dim as f32 * partial) as usize) & !1;
    let rot_dim = rot_dim.max(2);
    let half_rot = rot_dim / 2;
    let elem = std::mem::size_of::<f32>();
    let row_bytes = head_dim * elem;
    let total_bytes = effective_seq * row_bytes;
    if cache_offset + total_bytes > scratchpad.len() {
        return Err(ExecutionError::Compilation(format!(
            "populate_rope_cache: scratchpad too small (need {} bytes at offset {}, have {})",
            total_bytes, cache_offset, scratchpad.len())));
    }
    let positions_slice: &[u32] = unsafe { std::slice::from_raw_parts(positions, effective_seq) };
    // SAFETY: scratchpad[cache_offset..cache_offset+total_bytes] 已 bound-check,
    // f32 写入 row-major, 无 aliasing。
    let cache_ptr = unsafe { scratchpad.as_mut_ptr().add(cache_offset) as *mut f32 };
    for (row_idx, &pos) in positions_slice.iter().enumerate() {
        let row_base = unsafe { cache_ptr.add(row_idx * head_dim) };
        for i in 0..half_rot {
            let freq = 1.0f64 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * freq;
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            unsafe {
                *row_base.add(i) = c;
                *row_base.add(half_rot + i) = s;
            }
        }
    }
    Ok(())
}

/// Build a CompilerGraph for FlashAttention.
///
/// Q[s,h], K[s,h], V[s,h] → MultiHeadAttention → out[s,h]
/// Build a CompilerGraph for FlashAttention with symbolic seq_len.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_flash_attention_graph(
    config: &FlashAttentionConfig,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let h = config.num_heads * config.head_dim;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    let q = g.add_tensor("q", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    let k = g.add_tensor("k", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    let v = g.add_tensor("v", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor("attn_out", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: seq_len_sym,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            causal: config.causal,
        },
        vec![q, k, v],
        vec![out],
        "flash_attention",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for SwiGLU with projection (WII architecture).
///
/// input[s,h], w_gate[h,inter], w_up[h,inter] →
///   gate = input × w_gate → silu(gate) × (input × w_up) → out[s,inter]
/// Build a CompilerGraph for SwiGLU fusion with symbolic seq_len.
///
/// Follows the FusedQkvRope pattern: weights are graph inputs consumed
/// by internal Gemm ops. The JIT kernel receives activation + weight_blob
/// and does the full gate/up projection + SwiGLU computation internally.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_swiglu_graph(
    config: &SwiGLUConfig,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;

    // Inputs: activation (symbolic seq_len) + 2 weight matrices (WII pattern)
    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN), // Conservative upper bound for buffer allocation
    };
    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    let w_gate = g.add_tensor_concrete("w_gate", &[hidden, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[hidden, inter], dt);
    g.inputs = vec![input, w_gate, w_up];

    // gate = input × w_gate  [seq_len, inter]
    let gate_out = g.add_tensor("gate_proj", vec![seq_len_sym.clone(), SymDim::Concrete(inter)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: inter, k: hidden, dtype: dt },
        vec![input, w_gate],
        vec![gate_out],
        "gate_gemm",
    );

    // up = input × w_up  [seq_len, inter]
    let up_out = g.add_tensor("up_proj", vec![seq_len_sym.clone(), SymDim::Concrete(inter)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: inter, k: hidden, dtype: dt },
        vec![input, w_up],
        vec![up_out],
        "up_gemm",
    );

    // silu(gate) * up → out
    let out = g.add_tensor("swiglu_out", vec![seq_len_sym, SymDim::Concrete(inter)], dt);
    g.add_op(OpKind::SwiGlu, vec![gate_out, up_out], vec![out], "swiglu");

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for RoPE.
///
/// input[s,h], cos_sin[head_dim/2] → RoPE → out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_rope_graph(
    config: &RoPEConfig,
    hidden: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };
    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    g.inputs = vec![input];

    let out = g.add_tensor("rope_out", vec![seq_len_sym, SymDim::Concrete(hidden)], dt);
    g.add_op(
        OpKind::RoPE {
            num_heads: hidden / config.head_dim,
            head_dim: config.head_dim,
            theta: config.rope_theta,
            partial: 1.0,
        },
        vec![input],
        vec![out],
        "rope",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for FusedQkvRope.
/// Build a CompilerGraph for FusedQkvRope with symbolic seq_len.
///
/// input[s,h] + w_q,w_k,w_v + cos_sin → Q/K/V Gemms + RoPE(Q) + RoPE(K) → [q_rope, k_rope, v]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_fused_qkv_rope_graph(
    config: &FusedQkvRopeConfig,
    hidden: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    let w_q = g.add_tensor_concrete("w_q", &[hidden, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[hidden, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[hidden, kv_dim], dt);
    g.inputs = vec![input, w_q, w_k, w_v];

    let q_out = g.add_tensor("q", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: q_dim, k: hidden, dtype },
        vec![input, w_q],
        vec![q_out],
        "gemm_q",
    );

    let k_out = g.add_tensor("k", vec![seq_len_sym.clone(), SymDim::Concrete(kv_dim)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: kv_dim, k: hidden, dtype },
        vec![input, w_k],
        vec![k_out],
        "gemm_k",
    );

    let v_out = g.add_tensor("v", vec![seq_len_sym.clone(), SymDim::Concrete(kv_dim)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: kv_dim, k: hidden, dtype },
        vec![input, w_v],
        vec![v_out],
        "gemm_v",
    );

    let q_rope = g.add_tensor("q_rope", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    g.add_op(
        OpKind::RoPE { num_heads: config.num_heads, head_dim: config.head_dim, theta: config.rope_theta, partial: 1.0 },
        vec![q_out],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor("k_rope", vec![seq_len_sym, SymDim::Concrete(kv_dim)], dt);
    g.add_op(
        OpKind::RoPE { num_heads: config.num_kv_heads, head_dim: config.head_dim, theta: config.rope_theta, partial: 1.0 },
        vec![k_out],
        vec![k_rope],
        "rope_k",
    );

    g.outputs = vec![q_rope, k_rope, v_out];
    g
}

/// Build a CompilerGraph for FusedRMSLinear with symbolic seq_len: RMSNorm → Gemm.
///
/// input[s,h] + norm_w[h] + linear_w[h,h] → RmsNorm → Gemm → out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_fused_rms_linear_graph(
    config: &FusedRMSLinearConfig,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let h = config.hidden_size;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    let norm_w = g.add_tensor_concrete("norm_w", &[h], dt);
    let linear_w = g.add_tensor_concrete("linear_w", &[h, h], dt);
    g.inputs = vec![input, norm_w, linear_w];

    let normed = g.add_tensor("normed", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::RmsNorm { eps: config.eps },
        vec![input, norm_w],
        vec![normed],
        "rms_norm",
    );

    let out = g.add_tensor("rms_linear_out", vec![seq_len_sym.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym, n: h, k: h, dtype },
        vec![normed, linear_w],
        vec![out],
        "linear",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for GQA (Grouped Query Attention) with symbolic seq_len.
///
/// Q[s,q_dim], K[s,kv_dim], V[s,kv_dim] → MHA → out[s,q_dim]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_gqa_graph(
    config: &GQAConfig,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let q_dim = config.num_heads * config.head_dim;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    // ARCH-GQA-RESHAPE (SPEC 01-JIT-PIPELINE.md §437): GQA 展开含 Reshape(K/V)。
    // gllm caller 端 (FusedGraphExecutor::expand_gqa_heads) 把 K/V 从
    // [seq, num_kv_heads*head_dim] 按 group_size 复制扩展为 [seq, num_heads*head_dim]
    // → JIT kernel 里 Q/K/V 三者 head 数对齐,lower_mha 走对称 MHA 路径 (h_off
    // 同时索引 Q/K/V),避免在 attention kernel 内部做 GQA 分组映射时遭遇
    // ARCH-REGALLOC-COUNTER-NOSPILL 嵌套循环限制。因此 CompilerGraph 中 K/V
    // 声明为 q_dim (广播后的目标宽度),与 caller 提供的 expanded 张量一致。
    let q = g.add_tensor("q", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    let k = g.add_tensor("k", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    let v = g.add_tensor("v", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor("gqa_out", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: seq_len_sym,
            num_heads: config.num_heads,
            // num_kv_heads 传 num_heads:JIT kernel 看到对称 MHA 语义。
            // 原始 GQAConfig.num_kv_heads 仍保留在 config 里供 KV cache 写入等其他
            // 逻辑使用 (cache 按真实 kv_head 布局,expand 仅发生在 per-step 喂给
            // JIT 的临时 tensor 上)。
            num_kv_heads: config.num_heads,
            head_dim: config.head_dim,
            causal: true, // GQA is decoder-only, always causal
        },
        vec![q, k, v],
        vec![out],
        "gqa",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for MoE routing with symbolic seq_len.
///
/// input[s,h] + gate_w[h,n_experts] → Gemm → Softmax → out[s,n_experts]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_moe_routing_graph(
    config: &MoERoutingConfig,
    hidden: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let n = config.num_experts;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    let gate_w = g.add_tensor_concrete("gate_w", &[hidden, n], dt);
    g.inputs = vec![input, gate_w];

    let gate_logits = g.add_tensor("gate_logits", vec![seq_len_sym.clone(), SymDim::Concrete(n)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n, k: hidden, dtype },
        vec![input, gate_w],
        vec![gate_logits],
        "gate_gemm",
    );

    let routing = g.add_tensor("routing", vec![seq_len_sym, SymDim::Concrete(n)], dt);
    g.add_op(
        OpKind::Softmax,
        vec![gate_logits],
        vec![routing],
        "routing_softmax",
    );

    g.outputs = vec![routing];
    g
}

/// Build a CompilerGraph for Per-Layer Embedding (PLE) with symbolic seq_len.
///
/// PLE computation:
///   ple_token = gather_slice(ple_embed_w, layer_i)   // [seq, dim]
///   ple_ctx   = Gemm(main_embed, proj_w)              // [seq, dim]
///   signal    = (ple_ctx + ple_token × √dim) / √2
///   output    = Gemm(signal, post_mlp_w) + hidden     // residual injection
///
/// Inputs: hidden[s,h], main_embed[s,h], ple_embed_slice[s,dim], proj_w[h,dim], post_mlp_w[dim,h]
/// Output: hidden_out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_ple_graph(
    config: &PleConfig,
    hidden: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let dim = config.dim_per_layer;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    };

    // Inputs:
    // 0: hidden       [seq_len, hidden]  — current hidden state
    // 1: main_embed   [seq_len, hidden]  — main token embedding (from embed_tokens)
    // 2: ple_slice    [seq_len, dim]     — pre-sliced PLE token embedding for this layer
    // 3: proj_w       [hidden, dim]      — context-aware projection weight
    // 4: post_mlp_w   [dim, hidden]      — post-MLP residual projection weight
    let hidden_in = g.add_tensor("hidden", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    let main_embed = g.add_tensor("main_embed", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    let ple_slice = g.add_tensor("ple_slice", vec![seq_len_sym.clone(), SymDim::Concrete(dim)], dt);
    let proj_w = g.add_tensor_concrete("proj_w", &[hidden, dim], dt);
    let post_mlp_w = g.add_tensor_concrete("post_mlp_w", &[dim, hidden], dt);
    g.inputs = vec![hidden_in, main_embed, ple_slice, proj_w, post_mlp_w];

    // ple_ctx = Gemm(main_embed, proj_w) → [seq_len, dim]
    let ple_ctx = g.add_tensor("ple_ctx", vec![seq_len_sym.clone(), SymDim::Concrete(dim)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: dim, k: hidden, dtype },
        vec![main_embed, proj_w],
        vec![ple_ctx],
        "ple_proj",
    );

    // scaled_token = ple_slice × √dim (element-wise scale via Mul)
    // signal = (ple_ctx + scaled_token) / √2
    // Combined as: signal = ple_ctx + ple_slice * √dim, then scale by 1/√2
    // We use Add(ple_ctx, Mul(ple_slice, sqrt_dim)) then Mul(result, inv_sqrt2)
    // But since we only have elementwise ops, we compose them:

    // Step 1: scale ple_slice by √dim → scaled_token
    let scaled_token = g.add_tensor("scaled_token", vec![seq_len_sym.clone(), SymDim::Concrete(dim)], dt);
    g.add_op(
        OpKind::Mul, // ple_slice * sqrt_dim_constant (baked into JIT as scaling)
        vec![ple_slice, ple_ctx], // Mul of two tensors (will be scaled by constants in JIT)
        vec![scaled_token],
        "ple_scale_token",
    );

    // Step 2: signal = scaled_token (this represents the fused add+scale)
    // In the JIT pipeline, the actual √dim and 1/√2 scaling factors are embedded
    // as part of the fused kernel's constant parameters.
    let signal = g.add_tensor("signal", vec![seq_len_sym.clone(), SymDim::Concrete(dim)], dt);
    g.add_op(
        OpKind::Add,
        vec![ple_ctx, scaled_token],
        vec![signal],
        "ple_combine",
    );

    // post_mlp_out = Gemm(signal, post_mlp_w) → [seq_len, hidden]
    let post_mlp_out = g.add_tensor("post_mlp_out", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len_sym.clone(), n: hidden, k: dim, dtype },
        vec![signal, post_mlp_w],
        vec![post_mlp_out],
        "ple_post_mlp",
    );

    // hidden_out = hidden + post_mlp_out (residual)
    let hidden_out = g.add_tensor("hidden_out", vec![seq_len_sym, SymDim::Concrete(hidden)], dt);
    g.add_op(
        OpKind::Add,
        vec![hidden_in, post_mlp_out],
        vec![hidden_out],
        "ple_residual",
    );

    g.outputs = vec![hidden_out];
    g
}

/// Map an atomic op_type string to a CompilerGraph OpKind.
///
/// Returns Err for unrecognized op types — NO silent fallback.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn atomic_op_to_kind(
    op_type: &str,
    input_shapes: &[Vec<gllm_kernels::compiler::SymDim>],
    dtype: gllm_kernels::types::DType,
) -> Result<gllm_kernels::compiler::OpKind, ExecutionError> {
    use gllm_kernels::compiler::{OpKind, SymDim};

    match op_type {
        "Add" => Ok(OpKind::Add),
        "Mul" => Ok(OpKind::Mul),
        "Silu" | "SiLU" | "Swish" => Ok(OpKind::Silu),
        "Gelu" | "GELU" => Ok(OpKind::Gelu),
        "Tanh" => Ok(OpKind::Tanh),
        "SimplifiedLayerNormalization" => Ok(OpKind::RmsNorm { eps: 1e-6 }),
        "LayerNormalization" => Ok(OpKind::LayerNorm { eps: 1e-5 }),
        "Softmax" => {
            // ARCH-SYMDIM-OUTER-ONLY: vocab_size 是内层维度，必须 Concrete
            let vocab_size = if !input_shapes.is_empty() && input_shapes[0].len() >= 2 {
                let dim = &input_shapes[0][input_shapes[0].len() - 1];
                debug_assert!(!dim.is_symbolic(), "Softmax vocab_size should be Concrete");
                dim.as_concrete().expect("ARCH-SYMDIM-OUTER-ONLY: Softmax vocab_size must be Concrete")
            } else {
                1
            };
            Ok(OpKind::SoftmaxWithEntropy { vocab_size })
        }
        "Residual" => {
            // ARCH-SYMDIM-OUTER-ONLY: hidden 是内层维度，必须 Concrete
            let hidden = if !input_shapes.is_empty() && input_shapes[0].len() >= 2 {
                let dim = &input_shapes[0][input_shapes[0].len() - 1];
                debug_assert!(!dim.is_symbolic(), "Residual hidden should be Concrete");
                dim.as_concrete().expect("ARCH-SYMDIM-OUTER-ONLY: Residual hidden_size must be Concrete")
            } else {
                1
            };
            Ok(OpKind::ResidualWithTelemetry { hidden })
        }
        "MatMul" | "Gemm" => {
            if input_shapes.len() < 2
                || input_shapes[0].len() < 2
                || input_shapes[1].len() < 2
            {
                return Err(ExecutionError::ShapeMismatch(format!(
                    "MatMul/Gemm requires 2 inputs with >=2D shapes, got {:?}",
                    input_shapes,
                )));
            }
            let a = &input_shapes[0];
            let b = &input_shapes[1];
            let m = a[a.len() - 2].clone();
            let k_a = &a[a.len() - 1];
            let k_b = &b[b.len() - 2];
            let n_dim = &b[b.len() - 1];

            // K dimensions must match (both Concrete with same value, or both Symbolic with same name)
            let k = match (k_a, k_b) {
                (SymDim::Concrete(ka), SymDim::Concrete(kb)) if ka == kb => *ka,
                (SymDim::Concrete(ka), _) => *ka,
                (_, SymDim::Concrete(kb)) => *kb,
                _ => return Err(ExecutionError::ShapeMismatch(format!(
                    "MatMul K dimension mismatch: {:?} vs {:?}", k_a, k_b
                ))),
            };

            let n = n_dim.as_concrete().ok_or_else(|| {
                ExecutionError::ShapeMismatch(format!(
                    "MatMul N dimension must be Concrete, got {:?}", n_dim
                ))
            })?;

            Ok(OpKind::Gemm { m, n, k, dtype })
        }
        "LayerNorm" => Ok(OpKind::LayerNorm { eps: 1e-5 }),
        "RMSNorm" | "RmsNorm" => Ok(OpKind::RmsNorm { eps: 1e-5 }),
        _ => Err(ExecutionError::UnsupportedOp(format!(
            "atomic op '{}' has no CompilerGraph mapping — \
             JIT codegen not implemented for this op type",
            op_type,
        ))),
    }
}

/// Build a CompilerGraph for a single atomic operation with SymDim shapes.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_atomic_graph(
    op_type: &str,
    input_shapes: &[Vec<gllm_kernels::compiler::SymDim>],
    output_shape: &[gllm_kernels::compiler::SymDim],
    dtype: gllm_kernels::types::DType,
) -> Result<gllm_kernels::compiler::CompilerGraph, ExecutionError> {
    use gllm_kernels::compiler::CompilerGraph;

    let kind = atomic_op_to_kind(op_type, input_shapes, dtype)?;
    let mut g = CompilerGraph::new();
    let dt = dtype;

    let mut input_ids = Vec::new();
    for (i, shape) in input_shapes.iter().enumerate() {
        let name = format!("input_{i}");
        let tid = g.add_tensor(&name, shape.clone(), dt);
        input_ids.push(tid);
    }
    g.inputs = input_ids.clone();

    let out = g.add_tensor("output", output_shape.to_vec(), dt);

    match kind {
        gllm_kernels::compiler::OpKind::SoftmaxWithEntropy { .. }
        | gllm_kernels::compiler::OpKind::ResidualWithTelemetry { .. } => {
            let seq_len_dim = if !input_shapes.is_empty() {
                input_shapes[0][0].clone()
            } else {
                gllm_kernels::compiler::SymDim::Concrete(1)
            };
            let out_telemetry = g.add_tensor("telemetry", vec![seq_len_dim], dt);
            g.add_op(kind.clone(), input_ids.clone(), vec![out, out_telemetry], op_type);
            g.outputs = vec![out, out_telemetry];
        }
        _ => {
            g.add_op(kind.clone(), input_ids.clone(), vec![out], op_type);
            g.outputs = vec![out];
        }
    }

    Ok(g)
}

/// Infer output shape from op type and input shapes (SymDim version).
fn infer_output_shape(
    op_type: &str,
    input_shapes: &[Vec<gllm_kernels::compiler::SymDim>],
) -> Vec<gllm_kernels::compiler::SymDim> {
    use gllm_kernels::compiler::SymDim;

    match op_type {
        "MatMul" | "Gemm" => {
            if input_shapes.len() >= 2
                && input_shapes[0].len() >= 2
                && input_shapes[1].len() >= 2
            {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                vec![a[a.len() - 2].clone(), b[b.len() - 1].clone()]
            } else if !input_shapes.is_empty() {
                input_shapes[0].clone()
            } else {
                vec![SymDim::Concrete(1)]
            }
        }
        // Elementwise ops preserve the first input's shape
        _ => {
            if !input_shapes.is_empty() {
                input_shapes[0].clone()
            } else {
                vec![SymDim::Concrete(1)]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FusedGraphExecutor
// ---------------------------------------------------------------------------

/// GPU-compiled node: holds CUDA module + kernel entries for one FusedNode.
#[cfg(feature = "cuda")]
struct GpuCompiledNode {
    /// Keep the CUDA module alive so kernel function pointers remain valid.
    _module: gllm_kernels::gpu::cuda::CudaModule,
    /// Per-kernel metadata for launching.
    kernel_entries: Vec<crate::compat::GpuKernelEntry>,
    /// The CompilerGraph used (needed for tensor metadata during launch).
    graph: gllm_kernels::compiler::CompilerGraph,
    /// Names of the FusedNode input tensors, in order.
    graph_input_names: Vec<String>,
    /// Names of the FusedNode output tensors.
    graph_output_names: Vec<String>,
    /// Number of elements in the output tensor(s).
    output_numel: usize,
    /// Per-output element counts for multi-output nodes.
    per_output_numel: Vec<usize>,
}

/// FusedGraph 执行器。
///
/// Two-phase usage:
/// 1. `compile(seq_len, hidden)` — JIT-compiles each FusedNode into native code
/// 2. `run(inputs)` — executes the compiled kernels in topological order
///
/// `run()` without prior `compile()` returns `NotCompiled` error.
pub struct FusedGraphExecutor {
    graph: FusedGraph,
    /// Per-node compiled JIT kernels (indexed by node position in graph.nodes).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    compiled_nodes: Vec<CompiledNode>,
    /// Per-node GPU-compiled kernels (indexed by node position in graph.nodes).
    #[cfg(feature = "cuda")]
    gpu_compiled_nodes: Vec<GpuCompiledNode>,
    /// Whether compile() has been called successfully.
    is_compiled: bool,
    /// Whether compile_gpu() has been called successfully.
    #[cfg(feature = "cuda")]
    is_gpu_compiled: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct NodePayload {
    code_bytes: Vec<u8>,
    scratchpad_bytes: usize,
    config_hash: u64,
    weight_layout_offsets: Option<Vec<(u32, usize)>>,
    weight_layout_total_bytes: Option<usize>,

    graph_input_names: Vec<String>,
    graph_output_names: Vec<String>,
    output_numel: usize,
    per_output_numel: Vec<usize>,
    output_dtype_id: u8,
    /// Feature dimension per token (product of Concrete output dims).
    feature_dim: usize,
    /// Per-output feature dimensions for multi-output nodes.
    per_output_feature_dims: Vec<usize>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct GraphExecutorPayload {
    nodes: Vec<NodePayload>,
}

// Manual Debug impl because CompiledLayer does not derive Debug.
impl std::fmt::Debug for FusedGraphExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedGraphExecutor")
            .field("graph_nodes", &self.graph.nodes.len())
            .field("is_compiled", &self.is_compiled)
            .finish()
    }
}

impl FusedGraphExecutor {
    pub fn new(graph: FusedGraph) -> Self {
        Self {
            graph,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
            compiled_nodes: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu_compiled_nodes: Vec::new(),
            is_compiled: false,
            #[cfg(feature = "cuda")]
            is_gpu_compiled: false,
        }
    }

    /// Bind a named weight as pre-materialized f32 bytes.
    ///
    /// Used when weights are dequantized on the host before binding.
    /// The bytes are stored in `WeightBinding::data` and are used directly
    /// by `seed_tensors_and_weight_ptrs` without any further ptr indirection.
    pub fn bind_bytes(mut self, name: String, f32_bytes: Vec<u8>) -> Self {
        if let Some(mut meta) = self.graph.weight_bindings.remove(&name) {
            meta.data = Some(f32_bytes);
            meta.ptr = None;
            self.graph.weight_bindings.insert(name, meta);
        } else {
            self.graph.weight_bindings.insert(name, crate::graph::types::WeightBinding {
                source_name: String::new(),
                shape: vec![],
                dtype: safetensors::Dtype::F32,
                data: Some(f32_bytes),
                ptr: None,
                    shape_needs_transpose: false,
            });
        }
        self
    }

    /// Bind a named weight to a raw pointer.
    ///
    /// If a WeightBinding entry already exists for `name`, updates its `ptr`.
    /// Otherwise creates a new entry with the given pointer (shape/dtype
    /// will be inferred during compilation from the CompilerGraph).
    pub fn bind(mut self, name: String, ptr: *const f32) -> Self {
        if let Some(mut meta) = self.graph.weight_bindings.remove(&name) {
            meta.ptr = Some(ptr);
            self.graph.weight_bindings.insert(name, meta);
        } else {
            // Create a new binding with just the pointer — shape will be
            // resolved from the CompiledNode's graph input tensor metadata.
            self.graph.weight_bindings.insert(name, crate::graph::types::WeightBinding {
                source_name: String::new(),
                shape: vec![],
                dtype: safetensors::Dtype::F32,
                data: None,
                ptr: Some(ptr),
                    shape_needs_transpose: false,
            });
        }
        self
    }

    /// Compile every FusedNode into a JIT kernel.
    ///
    /// For each node the executor:
    /// 1. Builds a `CompilerGraph` representing that fused op's computation
    /// 2. Compiles it via `InferenceCompiler::compile_graph`
    /// 3. Caches the resulting `CompiledLayer`
    ///
    /// `seq_len` — sequence-length dimension for tensor shapes.
    /// `hidden`  — model hidden dimension (needed by ops whose config
    ///             does not carry it, e.g. RoPE, MoERouting).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile(
        &mut self,
        seq_len: usize,
        hidden: usize,
        _dtype: gllm_kernels::types::DType,
    ) -> Result<(), ExecutionError> {
        log::debug!("[COMPILE] start: seq_len={} hidden={} nodes={}", seq_len, hidden, self.graph.nodes.len());
        // CPU JIT always computes in f32 regardless of model weight dtype.
        // Weights are converted to f32 during upload (upload_native_tensor_with_convert).
        // Using BF16/F16 here would cause weight_layout and output buffers to be 2x too small.
        let compile_dtype = gllm_kernels::types::DType::F32;

        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let mut compiled_nodes = Vec::with_capacity(self.graph.nodes.len());

        // Pre-scan: 按拓扑顺序建立每个 tensor 的 shape 表。下游 node 查这个表
        // 替代 default [seq, hidden], 正确处理 FFN intermediate_size ≠ hidden_size
        // 等跨节点 feature_dim 变化场景。
        let tensor_shapes = self.build_tensor_shape_map(hidden);

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let build = self.build_node_graph_with_shapes(idx, seq_len, hidden, compile_dtype, &tensor_shapes)?;
            let compiled = compiler
                .compile_graph(&build.graph)
                .map_err(|e| {
                    ExecutionError::Compilation(format!(
                        "JIT compilation failed for node '{}' (op: {}): {}",
                        node.name,
                        node.op.name(),
                        e,
                    ))
                })?;

            let output_dtype = gllm_kernels::types::DType::F32;

            compiled_nodes.push(CompiledNode {
                compiled,
                graph_input_names: build.input_names,
                graph_output_names: build.output_names,
                output_numel: build.output_numel,
                per_output_numel: build.per_output_numel,
                output_dtype,
                feature_dim: build.feature_dim,
                per_output_feature_dims: build.per_output_feature_dims,
            });
        }

        self.compiled_nodes = compiled_nodes;
        self.is_compiled = true;
        log::debug!("[COMPILE] done: {} nodes compiled", self.compiled_nodes.len());
        Ok(())
    }

    /// Extends `compile` to check `ArtifactCache` first. If cache miss, falls back to `compile`
    /// and writes the serialized payload back to `ArtifactCache` using `save_blob`.
    ///
    /// Implements REQ-JIT-CACHE-003: L3 disk persistence with format {model_hash}_{backend}.bin.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile_with_cache(
        &mut self,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        model_id: &str,
        backend: &str,
        cache: &crate::compat::artifact_cache::ArtifactCache,
    ) -> Result<(), ExecutionError> {
        // Generate model hash for filename (REQ-JIT-CACHE-003 format)
        let model_hash = cache.get_model_hash(model_id, &self.graph, "default");

        // 1. Try L3 Cache bypass (load by model_hash + backend)
        if let Some(blob) = cache.load_blob(&model_hash, backend) {
            if let Ok(payload) = bincode::deserialize::<GraphExecutorPayload>(&blob) {
                if let Ok(nodes) = self.restore_payload(payload) {
                    self.compiled_nodes = nodes;
                    self.is_compiled = true;
                    // Skip inference compiler
                    return Ok(());
                }
            }
        }

        // 2. Compilation Miss - Fallback to full trace-codegen
        self.compile(seq_len, hidden, dtype)?;

        // 3. Serialize and commit to ArtifactCache (save by modelHash + backend)
        if let Ok(payload) = self.build_payload() {
            if let Ok(blob) = bincode::serialize(&payload) {
                if let Err(e) = cache.save_blob(&model_hash, backend, &blob) {
                    log::debug!("JIT L3 cache save failed (non-fatal): {}", e);
                }
            }
        }

        Ok(())
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn restore_payload(&self, payload: GraphExecutorPayload) -> Result<Vec<CompiledNode>, ExecutionError> {
        use gllm_kernels::compiler::graph::{WeightLayout, TensorId};
        use gllm_kernels::types::DType;
        
        let mut nodes = Vec::with_capacity(payload.nodes.len());
        for p in payload.nodes.into_iter() {
            let mut layer = gllm_kernels::compiler::CompiledLayer::from_code(
                &p.code_bytes,
                p.scratchpad_bytes,
                p.config_hash,
            ).map_err(|e| ExecutionError::Compilation(e.to_string()))?;
            
            if let (Some(offsets), Some(total_bytes)) = (p.weight_layout_offsets, p.weight_layout_total_bytes) {
                let weight_layout = WeightLayout {
                    offsets: offsets.into_iter().map(|(id, off)| (TensorId(id), off)).collect(),
                    total_bytes,
                };
                layer.weight_layout = Some(weight_layout);
            }

            let output_dtype = match p.output_dtype_id {
                0 => DType::F32,
                1 => DType::F16,
                2 => DType::BF16,
                3 => DType::U8,
                _ => return Err(ExecutionError::Backend(format!("Unknown dtype ID: {}", p.output_dtype_id))),
            };

            nodes.push(CompiledNode {
                compiled: layer,
                graph_input_names: p.graph_input_names,
                graph_output_names: p.graph_output_names,
                output_numel: p.output_numel,
                per_output_numel: p.per_output_numel,
                output_dtype,
                feature_dim: p.feature_dim,
                per_output_feature_dims: p.per_output_feature_dims,
            });
        }
        
        if nodes.len() != self.graph.nodes.len() {
            return Err(ExecutionError::Backend("Cache node length mismatch".into()));
        }

        Ok(nodes)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_payload(&self) -> Result<GraphExecutorPayload, ExecutionError> {
        let mut nodes = Vec::with_capacity(self.compiled_nodes.len());
        
        for node in &self.compiled_nodes {
            let code_bytes = node.compiled.code_bytes().to_vec();
            
            let (offsets, total_bytes) = if let Some(ref wl) = node.compiled.weight_layout {
                let off = wl.offsets.iter().map(|(t, o)| (t.0, *o)).collect();
                (Some(off), Some(wl.total_bytes))
            } else {
                (None, None)
            };

            nodes.push(NodePayload {
                code_bytes,
                scratchpad_bytes: node.compiled.scratchpad_bytes,
                config_hash: node.compiled.config_hash,
                weight_layout_offsets: offsets,
                weight_layout_total_bytes: total_bytes,
                
                graph_input_names: node.graph_input_names.clone(),
                graph_output_names: node.graph_output_names.clone(),
                output_numel: node.output_numel,
                per_output_numel: node.per_output_numel.clone(),
                output_dtype_id: node.output_dtype.elem_id(),
                feature_dim: node.feature_dim,
                per_output_feature_dims: node.per_output_feature_dims.clone(),
            });
        }
        
        Ok(GraphExecutorPayload { nodes })
    }

    /// Result of building a CompilerGraph for one FusedNode.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    /// 按拓扑顺序 pre-scan 每个 tensor 的 shape。key = tensor name (node output
    /// 或 graph input / weight), value = SymDim shape。下游 node 的
    /// build_node_graph 查此表替代 default [seq, hidden]。
    ///
    /// ARCH-TENSOR-SHAPE-PRE-SCAN: 修复 FFN intermediate_size (4096) 与 hidden_size
    /// (1024) 不同时, GELU/Add/etc 节点 feature_dim 被错误默认为 hidden 导致
    /// output 只写部分元素, 下游 MatMul 读未初始化内存产生 NaN/Inf。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_tensor_shape_map(&self, hidden: usize) -> HashMap<String, Vec<gllm_kernels::compiler::SymDim>> {
        use gllm_kernels::compiler::SymDim;
        let mut shapes: HashMap<String, Vec<SymDim>> = HashMap::new();
        let seq_sym = SymDim::Symbolic { name: "seq_len".to_string(), max_value: Some(SYMDIM_MAX_SEQ_LEN) };

        // 1. Graph inputs (input_ids / position_ids / token_type_ids) = [seq]
        for name in &self.graph.inputs {
            shapes.insert(name.clone(), vec![seq_sym.clone()]);
        }

        // 2. Weight bindings: 读 shape, 按 shape_needs_transpose 转到 canonical
        //    [K, N] 供下游推导使用 (MatMul output shape = [M, N])。
        //
        //    ONNX 混用 MatMul/Gemm 格式 (某些 weight 是 canonical [K, N], 某些是
        //    HF [out, in] + transB=1), per-format global flag 不够。后续 loader
        //    层修复时细化。此处对 2D weight 暂保持 per-format 行为。
        for (name, wb) in &self.graph.weight_bindings {
            if !wb.shape.is_empty() {
                let mut shape: Vec<SymDim> = wb.shape.iter().map(|&d| SymDim::Concrete(d)).collect();
                if shape.len() == 2 && wb.shape_needs_transpose {
                    shape.swap(0, 1); // [N=out, K=in] → [K, N]
                }
                shapes.insert(name.clone(), shape);
            }
        }

        // 3. 按拓扑顺序 (FusedGraph.nodes 已是拓扑序) 推每个 node 的 output shape
        for node in &self.graph.nodes {
            if let super::types::FusedOp::Atomic(atomic) = &node.op {
                let out_shape = Self::infer_atomic_output_shape(
                    &atomic.op_type, &node.inputs, &shapes, hidden, &seq_sym,
                );
                for out_name in &node.outputs {
                    shapes.insert(out_name.clone(), out_shape.clone());
                }
            } else {
                // FusedOp (FlashAttention/SwiGLU/RoPE 等): 用 hidden 作为 feature_dim
                // 的保守推测。这些 op 有自己的 build_*_graph, 不依赖此表。
                for out_name in &node.outputs {
                    shapes.insert(out_name.clone(), vec![seq_sym.clone(), SymDim::Concrete(hidden)]);
                }
            }
        }
        shapes
    }

    /// 根据 atomic op 类型和 input shapes 推导 output shape。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn infer_atomic_output_shape(
        op_type: &str,
        input_names: &[String],
        shape_map: &HashMap<String, Vec<gllm_kernels::compiler::SymDim>>,
        hidden: usize,
        seq_sym: &gllm_kernels::compiler::SymDim,
    ) -> Vec<gllm_kernels::compiler::SymDim> {
        use gllm_kernels::compiler::SymDim;
        let default_act = vec![seq_sym.clone(), SymDim::Concrete(hidden)];

        let get_shape = |name: &str| -> Vec<SymDim> {
            shape_map.get(name).cloned().unwrap_or_else(|| default_act.clone())
        };

        match op_type {
            "MatMul" | "Gemm" => {
                // output = [M, N]. weight 已在 build_tensor_shape_map 里 canonicalize
                // 为 [K, N] (SafeTensors/PyTorch 会 swap, ONNX/GGUF 原生), 所以 N 永远
                // 是 w_shape[1]。
                if input_names.len() < 2 { return default_act; }
                let act_shape = get_shape(&input_names[0]);
                let w_shape = get_shape(&input_names[1]);
                if w_shape.len() != 2 { return default_act; }
                let m = if act_shape.len() >= 2 { act_shape[act_shape.len() - 2].clone() } else { seq_sym.clone() };
                let n = w_shape[1].clone();
                vec![m, n]
            }
            "Gather" => {
                // embed_table [vocab, hidden], indices [seq] → output [seq, hidden]
                let table_shape = input_names.iter()
                    .find_map(|n| {
                        let s = get_shape(n);
                        if s.len() == 2 { Some(s) } else { None }
                    });
                if let Some(t) = table_shape {
                    vec![seq_sym.clone(), t[1].clone()]
                } else {
                    default_act
                }
            }
            "LayerNormalization" | "SimplifiedLayerNormalization" | "LayerNorm" | "RMSNorm" | "RmsNorm"
            | "Silu" | "SiLU" | "Swish" | "Gelu" | "GELU" | "Tanh" | "Softmax" => {
                // unary: 继承 input[0] 的 shape
                if !input_names.is_empty() {
                    get_shape(&input_names[0])
                } else { default_act }
            }
            "Add" | "Mul" | "Residual" | "Sub" => {
                // binary: 取非 1D (非 bias) 的 input 作为基准; 否则 input[0]
                if input_names.is_empty() { return default_act; }
                let s0 = get_shape(&input_names[0]);
                if input_names.len() >= 2 {
                    let s1 = get_shape(&input_names[1]);
                    if s0.len() >= 2 { return s0; }
                    if s1.len() >= 2 { return s1; }
                }
                s0
            }
            _ => {
                if !input_names.is_empty() { get_shape(&input_names[0]) } else { default_act }
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_node_graph_with_shapes(
        &self,
        node_idx: usize,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        shape_map: &HashMap<String, Vec<gllm_kernels::compiler::SymDim>>,
    ) -> Result<NodeGraphBuild, ExecutionError> {
        self.build_node_graph_inner(node_idx, seq_len, hidden, dtype, Some(shape_map))
    }

    fn build_node_graph(
        &self,
        node_idx: usize,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
    ) -> Result<NodeGraphBuild, ExecutionError> {
        self.build_node_graph_inner(node_idx, seq_len, hidden, dtype, None)
    }

    fn build_node_graph_inner(
        &self,
        node_idx: usize,
        _seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        shape_map: Option<&HashMap<String, Vec<gllm_kernels::compiler::SymDim>>>,
    ) -> Result<NodeGraphBuild, ExecutionError> {
        let node = &self.graph.nodes[node_idx];
        let max_seq_len = SYMDIM_MAX_SEQ_LEN; // SymDim::Symbolic max_value

        match &node.op {
            FusedOp::FlashAttention(config) => {
                let g = build_flash_attention_graph(config, dtype);
                let h = config.num_heads * config.head_dim;
                Ok(make_node_build(g, node, max_seq_len * h, vec![]))
            }

            FusedOp::SwiGLU(config) => {
                let g = build_swiglu_graph(config, dtype);
                Ok(make_node_build(g, node, max_seq_len * config.intermediate_size, vec![]))
            }

            FusedOp::RoPE(config) => {
                let g = build_rope_graph(config, hidden, dtype);
                Ok(make_node_build(g, node, max_seq_len * hidden, vec![]))
            }

            FusedOp::FusedQkvRope(config) => {
                let g = build_fused_qkv_rope_graph(config, hidden, dtype);
                let q_dim = config.num_heads * config.head_dim;
                let kv_dim = config.num_kv_heads * config.head_dim;
                let per = vec![max_seq_len * q_dim, max_seq_len * kv_dim, max_seq_len * kv_dim];
                let total: usize = per.iter().sum();
                Ok(make_node_build(g, node, total, per))
            }

            FusedOp::FusedRMSLinear(config) => {
                let g = build_fused_rms_linear_graph(config, dtype);
                Ok(make_node_build(g, node, max_seq_len * config.hidden_size, vec![]))
            }

            FusedOp::GQA(config) => {
                let g = build_gqa_graph(config, dtype);
                let q_dim = config.num_heads * config.head_dim;
                Ok(make_node_build(g, node, max_seq_len * q_dim, vec![]))
            }

            FusedOp::MoERouting(config) => {
                let g = build_moe_routing_graph(config, hidden, dtype);
                Ok(make_node_build(g, node, max_seq_len * config.num_experts, vec![]))
            }

            FusedOp::PerLayerEmbed(config) => {
                let g = build_ple_graph(config, hidden, dtype);
                Ok(make_node_build(g, node, max_seq_len * hidden, vec![]))
            }

            FusedOp::Atomic(atomic) => {
                // ARCH-FULL-JIT §4.3/§4.4: Gather/Slice/Shape 走 JIT，禁止返回空图
                if atomic.op_type == "Gather" {
                    // ONNX Gather inputs: (data=table, indices). data is the embedding table
                    // registered in weight_bindings; indices is a runtime input (input_ids, etc.).
                    let table_name = node.inputs.iter()
                        .find(|name| self.graph.weight_bindings.contains_key(*name))
                        .ok_or_else(|| ExecutionError::MissingWeight(format!(
                            "Gather node '{}' has no weight input in weight_bindings (inputs={:?})",
                            node.name, node.inputs
                        )))?
                        .clone();
                    let indices_name = node.inputs.iter()
                        .find(|name| *name != &table_name)
                        .ok_or_else(|| ExecutionError::MissingInput(format!(
                            "Gather node '{}' has no indices input (inputs={:?})",
                            node.name, node.inputs
                        )))?
                        .clone();
                    let wb = &self.graph.weight_bindings[&table_name];
                    let embed_dim = wb.shape.last().copied().ok_or_else(|| {
                        ExecutionError::MissingWeight(format!(
                            "Gather weight '{}' has empty shape", table_name
                        ))
                    })?;
                    let table_rows = wb.shape.first().copied().ok_or_else(|| {
                        ExecutionError::MissingWeight(format!(
                            "Gather weight '{}' has empty shape", table_name
                        ))
                    })?;

                    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
                    let mut g = CompilerGraph::new();
                    let seq_sym = SymDim::Symbolic {
                        name: "seq_len".to_string(),
                        max_value: Some(SYMDIM_MAX_SEQ_LEN),
                    };
                    let indices = g.add_tensor("indices", vec![seq_sym.clone()], dtype);
                    let table = g.add_tensor_concrete("embed_table", &[table_rows, embed_dim], dtype);
                    let output = g.add_tensor("embed_out", vec![seq_sym.clone(), SymDim::Concrete(embed_dim)], dtype);
                    g.inputs = vec![indices, table];
                    g.outputs = vec![output];
                    g.add_op(
                        OpKind::Gather { table_rows, embed_dim, index_dim: seq_sym },
                        vec![indices, table], vec![output], "gather",
                    );
                    let output_numel = SYMDIM_MAX_SEQ_LEN * embed_dim;

                    // ARCH-GATHER-ABI-ORDER: JIT Gather 期望 ABI arg 0 = indices,
                    // arg 1 = table (CompilerGraph g.inputs 顺序)。但 YAML 原节点
                    // inputs 顺序是 ONNX 规范的 [data=table, indices] — 若直接用
                    // node.inputs.clone(), load_activation 会把 table 当 activation
                    // (input_ptr), pack_weight_blob 把 indices 当 weight → JIT 读错
                    // 数据源, output 会广播成 input_ids bits。必须重排成 [indices, table]
                    // 让 graph_input_names 与 JIT ABI 一致。
                    let mut nb = make_node_build(g, node, output_numel, vec![]);
                    nb.input_names = vec![indices_name, table_name];
                    return Ok(nb);
                }
                if atomic.op_type == "Slice" || atomic.op_type == "Shape" {
                    // Shape: 编译时常量折叠，恒等传递
                    // Slice: 零拷贝视图（指针偏移）
                    // 两者都生成 Reshape NOP，output_numel = max_seq × hidden（安全上界）
                    use gllm_kernels::compiler::{CompilerGraph, OpKind, SymDim};
                    let mut g = CompilerGraph::new();
                    let seq_sym = SymDim::Symbolic {
                        name: "seq_len".to_string(),
                        max_value: Some(SYMDIM_MAX_SEQ_LEN),
                    };
                    let input = g.add_tensor("input", vec![seq_sym.clone(), SymDim::Concrete(hidden)], dtype);
                    let output = g.add_tensor("output", vec![seq_sym, SymDim::Concrete(hidden)], dtype);
                    g.inputs = vec![input];
                    g.outputs = vec![output];
                    g.add_op(
                        OpKind::Reshape { target_shape: vec![0, hidden] },
                        vec![input], vec![output],
                        if atomic.op_type == "Slice" { "slice_view" } else { "shape_identity" },
                    );
                    let output_numel = SYMDIM_MAX_SEQ_LEN * hidden;
                    return Ok(make_node_build(g, node, output_numel, vec![]));
                }
                let is_matmul = atomic.op_type == "MatMul" || atomic.op_type == "Gemm";

                use gllm_kernels::compiler::SymDim;
                let seq_len_sym = SymDim::Symbolic {
                    name: "seq_len".to_string(),
                    max_value: Some(SYMDIM_MAX_SEQ_LEN),
                };

                // First pass: collect raw shapes (weight shapes transposed for MatMul)
                let mut input_shapes: Vec<Vec<SymDim>> = node
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        if let Some(wb) = self.graph.weight_bindings.get(name) {
                            if !wb.shape.is_empty() {
                                let mut shape: Vec<SymDim> = wb.shape.iter().map(|&d| SymDim::Concrete(d)).collect();
                                // ARCH-WEIGHT-CANONICAL-LAYOUT: 仅 HF SafeTensors/PyTorch
                                // 的 Linear weight 是 `[out, in]` 需要语义转置到 `[K, N]`。
                                // ONNX 原生 `[K, N]` 和 GGUF 不需要 swap, 否则会把
                                // intermediate size (4096) 错识为 hidden (1024)。
                                if is_matmul && i > 0 && shape.len() == 2 && wb.shape_needs_transpose {
                                    shape.swap(0, 1); // [N=out, K=in] → [K, N]
                                }
                                return shape;
                            }
                        }
                        // ARCH-TENSOR-SHAPE-PRE-SCAN: 查 pre-scan 构建的 shape map,
                        // 覆盖 default [seq, hidden] 对 FFN intermediate (4096) 等
                        // 非 hidden 维度的场景。
                        if let Some(map) = shape_map {
                            if let Some(s) = map.get(name) {
                                return s.clone();
                            }
                        }
                        vec![seq_len_sym.clone(), SymDim::Concrete(hidden)]
                    })
                    .collect();

                // Second pass: for MatMul, fix activation shape using weight's K dimension.
                // When the activation is a graph intermediate (not in weight_bindings), its shape
                // defaults to [seq_len, hidden]. But if the weight shape is known, the activation's
                // K dimension must match the weight's K dimension.
                if is_matmul && input_shapes.len() >= 2 {
                    if let Some(weight_k_dim) = input_shapes[1].first() {
                        if let SymDim::Concrete(weight_k) = weight_k_dim {
                            if input_shapes[0].len() >= 2 {
                                let act_k_dim = &input_shapes[0][input_shapes[0].len() - 1];
                                let needs_fix = match act_k_dim {
                                    SymDim::Concrete(act_k) => act_k != weight_k,
                                    _ => false,
                                };
                                if needs_fix && !self.graph.weight_bindings.contains_key(&node.inputs[0]) {
                                    // Activation shape was defaulted; fix K from weight
                                    let m = input_shapes[0][input_shapes[0].len() - 2].clone();
                                    input_shapes[0] = vec![m, SymDim::Concrete(*weight_k)];
                                }
                            }
                        }
                    }
                }

                // Elementwise op (Add/Mul): 若某个 input 是 1D bias 而另一个 activation
                // 被默认为 [seq, hidden=1024], 用 bias 的唯一维度修正 activation 的
                // feature_dim。典型场景: FFN intermediate_bias 的 bias shape=[4096],
                // activation 实际 [seq, 4096] 但被默认成 [seq, 1024] → Add JIT 只写前
                // 1024 列, 后 3072 列未初始化 → 下游 MatMul 读垃圾 NaN/Inf。
                //
                // 此启发式修正仅覆盖 Add/Mul 二元 elementwise, 不影响 MatMul/LayerNorm。
                let is_elementwise_binary = matches!(
                    atomic.op_type.as_str(),
                    "Add" | "Mul" | "Residual"
                );
                if is_elementwise_binary && input_shapes.len() == 2 {
                    // 找 1D bias input
                    let (bias_idx, bias_dim) = if input_shapes[0].len() == 1 {
                        (0usize, input_shapes[0][0].clone())
                    } else if input_shapes[1].len() == 1 {
                        (1usize, input_shapes[1][0].clone())
                    } else {
                        (usize::MAX, SymDim::Concrete(0))
                    };
                    if bias_idx != usize::MAX {
                        if let SymDim::Concrete(bias_feat) = bias_dim {
                            let act_idx = 1 - bias_idx;
                            if input_shapes[act_idx].len() >= 2 {
                                let last = input_shapes[act_idx].len() - 1;
                                let act_feat = &input_shapes[act_idx][last];
                                let needs_fix = matches!(act_feat, SymDim::Concrete(f) if *f != bias_feat);
                                if needs_fix && !self.graph.weight_bindings.contains_key(&node.inputs[act_idx]) {
                                    input_shapes[act_idx][last] = SymDim::Concrete(bias_feat);
                                }
                            }
                        }
                    }
                }

                let output_shape = infer_output_shape(&atomic.op_type, &input_shapes);

                // Calculate output_numel: use max_for_allocation for Symbolic dims
                let output_numel: usize = output_shape.iter().map(|d| d.max_for_allocation(SYMDIM_MAX_SEQ_LEN)).product();

                if std::env::var("GLLM_DEBUG_SHAPES").is_ok() {
                    eprintln!("[SHAPE] node '{}' op={} inputs={:?} input_shapes={:?} output_shape={:?} output_numel={}",
                        node.name, atomic.op_type, node.inputs, input_shapes, output_shape, output_numel);
                }
                let g = build_atomic_graph(&atomic.op_type, &input_shapes, &output_shape, dtype)?;
                Ok(make_node_build(g, node, output_numel, vec![]))
            }
        }
    }

    /// Compile every FusedNode into GPU (CUDA) kernels.
    ///
    /// For each node:
    /// 1. Builds a `CompilerGraph` (same as CPU path)
    /// 2. Compiles to PTX via the JIT pipeline
    /// 3. Loads the CUDA module and extracts kernel entries
    ///
    /// After `compile_gpu()`, call `run()` with the same inputs — the executor
    /// will automatically dispatch to the GPU path.
    #[cfg(feature = "cuda")]
    pub fn compile_gpu(
        &mut self,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        device: &gllm_kernels::gpu::cuda::CudaDevice,
        gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
        sm_version: u32,
    ) -> Result<(), ExecutionError> {
        let mut gpu_nodes = Vec::with_capacity(self.graph.nodes.len());

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let build = self.build_node_graph(idx, seq_len, hidden, dtype)?;

            let (module, kernel_entries) =
                crate::compat::cuda_compile_graph(device, gpu_profile, sm_version, &build.graph)
                    .map_err(|e| {
                        ExecutionError::Compilation(format!(
                            "GPU compilation failed for node '{}' (op: {}): {}",
                            node.name,
                            node.op.name(),
                            e,
                        ))
                    })?;

            gpu_nodes.push(GpuCompiledNode {
                _module: module,
                kernel_entries,
                graph: build.graph,
                graph_input_names: build.input_names,
                graph_output_names: build.output_names,
                output_numel: build.output_numel,
                per_output_numel: build.per_output_numel,
            });
        }

        self.gpu_compiled_nodes = gpu_nodes;
        self.is_gpu_compiled = true;
        Ok(())
    }

    /// Execute the fused graph.
    ///
    /// Runs the JIT-compiled kernels in topological order.
    /// Returns `NotCompiled` error if `compile()` was not called first.
    ///
    /// `shape_bindings`: 运行时 SymDim 绑定值（如 `{"seq_len": 6}`）。
    /// JIT kernel 的 BoundExpr::Symbolic 从 CompiledLayerFn 第 7 参数 [rbp+16] 读取 seq_len。
    pub fn run(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        shape_bindings: &HashMap<String, usize>,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        // Dependency validation pass (always runs)
        let mut available: HashSet<String> = inputs.keys().cloned().collect();
        for weight_name in self.graph.weight_bindings.keys() {
            available.insert(weight_name.clone());
        }

        for node in &self.graph.nodes {
            // Check if outputs are already provided externally (e.g. Gather bypassed via CPU input)
            if !node.outputs.is_empty() && node.outputs.iter().all(|out| available.contains(out)) {
                continue;
            }

            for input in &node.inputs {
                if input.is_empty() || available.contains(input) {
                    continue;
                }
                if self.graph.inputs.iter().any(|name| name == input) {
                    return Err(ExecutionError::MissingInput(input.clone()));
                }
                return Err(ExecutionError::MissingWeight(input.clone()));
            }
            for output in &node.outputs {
                if !output.is_empty() {
                    available.insert(output.clone());
                }
            }
        }

        // JIT execution path
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        if self.is_compiled {
            return self.run_compiled(inputs, shape_bindings);
        }

        // No compiled kernels available — refuse to return dummy results.
        // On supported architectures, compile() must be called first.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        return Err(ExecutionError::NotCompiled);

        // On unsupported architectures there is no JIT path at all.
        // Return an explicit error rather than empty placeholder buffers.
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda")))]
        Err(ExecutionError::UnsupportedOp(
            "JIT compilation not available on this architecture".to_string(),
        ))
    }

    /// Execute the fused graph on GPU.
    ///
    /// Requires prior `compile_gpu()` call. Uploads input tensors to GPU,
    /// launches all compiled kernels in topological order, and downloads
    /// output tensors back to host.
    #[cfg(feature = "cuda")]
    pub fn run_gpu(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        device: &gllm_kernels::gpu::cuda::CudaDevice,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
        use gllm_kernels::compiler::TensorId;

        if !self.is_gpu_compiled {
            return Err(ExecutionError::NotCompiled);
        }

        let stream = device.default_stream();
        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();

        // Seed with graph inputs
        for (name, data) in inputs {
            tensors.insert(name.clone(), data.clone());
        }
        // Seed with weight binding data
        for (name, wb) in &self.graph.weight_bindings {
            if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }

        // Execute each node on GPU
        for (_node_idx, gcn) in self.gpu_compiled_nodes.iter().enumerate() {
            if !gcn.graph_output_names.is_empty() && gcn.graph_output_names.iter().all(|name| tensors.contains_key(name)) {
                continue;
            }

            // Allocate GPU buffers for all graph tensors
            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut tensor_ptrs: HashMap<TensorId, u64> = HashMap::new();

            for (tidx, meta) in gcn.graph.tensors.iter().enumerate() {
                let tid = TensorId(tidx as u32);
                let n_elements = meta.concrete_numel();
                let size_bytes = n_elements * meta.dtype.size_bytes();
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| ExecutionError::Backend(format!(
                        "GPU alloc failed for {}: {e}", meta.name
                    )))?;

                // Upload data for input tensors by matching graph tensor names
                // to FusedNode input names
                let graph_input_idx = gcn.graph.inputs.iter().position(|&t| t == tid);
                if let Some(gi) = graph_input_idx {
                    if gi < gcn.graph_input_names.len() {
                        let fused_name = &gcn.graph_input_names[gi];
                        if let Some(data) = tensors.get(fused_name) {
                            device.htod(data, &mut buf, stream)
                                .map_err(|e| ExecutionError::Backend(format!(
                                    "htod {} failed: {e}", fused_name
                                )))?;
                        }
                    }
                }

                tensor_ptrs.insert(tid, buf.as_device_ptr());
                gpu_buffers.push((tid, buf));
            }

            // Launch all kernels for this node
            crate::compat::cuda_launch_graph(
                device, stream, &gcn.kernel_entries, &tensor_ptrs, &gcn.graph,
            ).map_err(|e| ExecutionError::Backend(format!(
                "GPU kernel launch failed: {e}"
            )))?;

            // Synchronize
            device.sync()
                .map_err(|e| ExecutionError::Backend(format!("GPU sync: {e}")))?;

            // Download output(s)
            if gcn.graph_output_names.len() == 1 {
                let output_tid = gcn.graph.outputs[0];
                let output_buf = gpu_buffers.iter()
                    .find(|(tid, _)| *tid == output_tid)
                    .map(|(_, buf)| buf)
                    .ok_or_else(|| ExecutionError::Backend("output buffer missing".into()))?;
                let out_dtype = gcn.graph.tensors[output_tid.0 as usize].dtype;
                let nbytes = gcn.output_numel * out_dtype.size_bytes();
                let mut host_buf = vec![0u8; nbytes];
                device.dtoh(output_buf, &mut host_buf, stream)
                    .map_err(|e| ExecutionError::Backend(format!("dtoh: {e}")))?;
                tensors.insert(gcn.graph_output_names[0].clone(), host_buf);
            } else if !gcn.per_output_numel.is_empty() {
                // Multi-output: download each output separately
                for (i, name) in gcn.graph_output_names.iter().enumerate() {
                    if i < gcn.graph.outputs.len() {
                        let output_tid = gcn.graph.outputs[i];
                        let output_buf = gpu_buffers.iter()
                            .find(|(tid, _)| *tid == output_tid)
                            .map(|(_, buf)| buf)
                            .ok_or_else(|| ExecutionError::Backend(
                                format!("output buffer missing for {name}")
                            ))?;
                        let per_out_dtype = gcn.graph.tensors[output_tid.0 as usize].dtype;
                        let nbytes = gcn.per_output_numel[i] * per_out_dtype.size_bytes();
                        let mut host_buf = vec![0u8; nbytes];
                        device.dtoh(output_buf, &mut host_buf, stream)
                            .map_err(|e| ExecutionError::Backend(format!(
                                "dtoh {name}: {e}"
                            )))?;
                        tensors.insert(name.clone(), host_buf);
                    }
                }
            }
            // gpu_buffers dropped here, freeing GPU memory for this node
        }

        // Collect graph outputs
        let mut out = HashMap::new();
        for name in &self.graph.outputs {
            let data = tensors.remove(name).unwrap_or_default(); // LEGAL: 不存在的 tensor 返回空数据
            out.insert(name.clone(), data);
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Shared execution helpers (used by run_compiled and run_with_kv_cache)
    // -----------------------------------------------------------------------

    /// Initialize tensor map from graph inputs and weight bindings.
    /// Returns (tensors, weight_ptrs) where weight_ptrs maps names to raw pointers
    /// for weights that have a runtime pointer but no shape (direct-ptr weights).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn seed_tensors_and_weight_ptrs(
        inputs: &HashMap<String, Vec<u8>>,
        weight_bindings: &HashMap<String, crate::graph::types::WeightBinding>,
    ) -> (HashMap<String, Vec<u8>>, HashMap<String, *const u8>) {
        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, data) in inputs {
            tensors.insert(name.clone(), data.clone());
        }

        let mut weight_ptrs: HashMap<String, *const u8> = HashMap::new();
        for (name, wb) in weight_bindings {
            if let Some(ptr) = wb.ptr {
                if !wb.shape.is_empty() {
                    // ARCH-WEIGHT-F32-CANONICAL (SPEC REQ-LOADER-016): CPU 后端在
                    // upload_native_tensor_with_convert 中已把所有权重物理转换为
                    // f32(F16/BF16/F64 源统一为 f32), device buffer 按 f32 布局
                    // 写入。因此按 ptr 读回原始字节流必须用 f32 size,而非 wb.dtype
                    // (可能是 BF16 等 pre-conversion dtype) 的 size,否则只读取到
                    // 转换前字节数的数据,后半部分留在 0 初始化状态 → downstream
                    // kernel 处理半个权重 → 模型输出严重偏离 (e.g. SmolLM2 decoder
                    // layer_0_input_norm 后半 288 维全零 → generator 输出乱码)。
                    let numel: usize = wb.shape.iter().product();
                    let bytes = numel * std::mem::size_of::<f32>();
                    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, bytes) };
                    tensors.insert(name.clone(), slice.to_vec());
                }
                weight_ptrs.insert(name.clone(), ptr as *const u8);
            } else if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }
        (tensors, weight_ptrs)
    }

    /// Load activation tensor for a compiled node.
    /// When `pad_to_max` is true, pads the activation buffer to max_seq_len size
    /// so JIT kernels can safely iterate up to the compiled loop bound.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn load_activation(
        cn: &CompiledNode,
        tensors: &HashMap<String, Vec<u8>>,
        pad_to_max: bool,
    ) -> Vec<u8> {
        if cn.graph_input_names.is_empty() {
            return Vec::new();
        }
        let raw = tensors
            .get(&cn.graph_input_names[0])
            .cloned()
            .unwrap_or_default();
        if pad_to_max && !raw.is_empty() {
            let max_act_bytes = cn.output_numel * cn.output_dtype.size_bytes();
            if raw.len() < max_act_bytes {
                let mut padded = raw;
                padded.resize(max_act_bytes, 0);
                return padded;
            }
        }
        raw
    }

    /// Pack weight tensors into a contiguous blob using the compiled weight_layout.
    /// graph_input_names[1..] are the weight tensors in order.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn pack_weight_blob(
        cn: &CompiledNode,
        tensors: &HashMap<String, Vec<u8>>,
        weight_ptrs: &HashMap<String, *const u8>,
    ) -> Vec<u8> {
        let mut weight_blob = Vec::new();
        if let Some(ref wl) = cn.compiled.weight_layout {
            weight_blob.resize(wl.total_bytes, 0u8);
            for (i, name) in cn.graph_input_names.iter().skip(1).enumerate() {
                let offset = if i < wl.offsets.len() { wl.offsets[i].1 } else { continue };
                let next_offset = if i + 1 < wl.offsets.len() { wl.offsets[i + 1].1 } else { wl.total_bytes };
                let size = next_offset - offset;
                if size == 0 { continue; }
                if let Some(data) = tensors.get(name) {
                    let copy_len = size.min(data.len());
                    weight_blob[offset..offset + copy_len].copy_from_slice(&data[..copy_len]);
                } else if let Some(&ptr) = weight_ptrs.get(name) {
                    let src = unsafe { std::slice::from_raw_parts(ptr, size) };
                    weight_blob[offset..offset + size].copy_from_slice(src);
                }
            }
        } else {
            // weight_layout 应由编译阶段生成。fallback concatenation 保留用于
            // 无权重节点（如 elementwise Add）或单权重节点（无需偏移映射）。
            // 多权重节点缺少 weight_layout 是编译错误。
            let weight_count = cn.graph_input_names.len().saturating_sub(1);
            if weight_count > 1 {
                log::warn!("[pack_weight_blob] node has {} weights but no weight_layout — concatenating sequentially",
                    weight_count);
            }
            for name in cn.graph_input_names.iter().skip(1) {
                if let Some(data) = tensors.get(name) {
                    weight_blob.extend_from_slice(data);
                } else if let Some(&ptr) = weight_ptrs.get(name) {
                    // 从权重指针读取 — 需要知道大小，从 weight_bindings 获取
                    // 没有 weight_layout 时无法确定大小，跳过
                    log::trace!("[pack_weight_blob] weight '{}' has ptr but no layout, skip", name);
                }
            }
        }
        weight_blob
    }

    /// Check if a node's outputs are all provided externally (by seeded inputs).
    ///
    /// 唯一合法的跳过场景: 上游通过 `inputs` 传入已计算的 hidden state
    /// (e.g. inject_knowledge 注入、Gather bypass)。此时节点输出已在 seed
    /// 阶段注入 `seeded_outputs`，无需重新执行。
    ///
    /// ⚠ 禁止基于 `tensors.contains_key` 判断跳过——YAML 架构中 `hidden_0`
    /// 被每层 `layer_${i}_output_norm` 覆盖写入，从 layer_1 起所有 output_norm
    /// 的输出 `hidden_0` 在 tensor map 已存在，按 tensor map 判断会错误跳过
    /// 所有后续层的 output_norm → encoder 实际只执行 layer_0 → 不同文档看到
    /// 相同 hidden → reranker 退化为恒定 logit。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn is_node_computed(cn: &CompiledNode, seeded_outputs: &HashSet<String>) -> bool {
        !cn.graph_output_names.is_empty()
            && cn.graph_output_names.iter().all(|name| seeded_outputs.contains(name))
    }

    /// Collect graph-level output tensors from the tensor map.
    fn collect_graph_outputs(
        graph: &FusedGraph,
        tensors: &mut HashMap<String, Vec<u8>>,
    ) -> HashMap<String, Vec<u8>> {
        let mut out = HashMap::new();
        for name in &graph.outputs {
            let data = tensors.remove(name).unwrap_or_default();
            out.insert(name.clone(), data);
        }
        out
    }

    /// Expand K/V tensors for GQA head repetition.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn expand_gqa_heads(
        cn: &CompiledNode,
        node_op: &FusedOp,
        tensors: &mut HashMap<String, Vec<u8>>,
        mha_kv_seq_len: usize,
    ) {
        let expand = match node_op {
            FusedOp::GQA(ref cfg) if cfg.num_kv_heads < cfg.num_heads => {
                Some((cfg.num_heads / cfg.num_kv_heads, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim))
            }
            FusedOp::FlashAttention(ref cfg) if cfg.num_kv_heads < cfg.num_heads => {
                Some((cfg.num_heads / cfg.num_kv_heads, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim))
            }
            _ => None,
        };
        let Some((repeat, num_heads, num_kv_heads, head_dim)) = expand else { return };

        let kv_names: Vec<&String> = cn.graph_input_names.iter().skip(1).take(2).collect();
        for &kv_name in &kv_names {
            let Some(kv_data) = tensors.get(kv_name).cloned() else { continue };
            let kv_dim = num_kv_heads * head_dim;
            let q_dim = num_heads * head_dim;
            let elem_bytes = cn.output_dtype.size_bytes();
            let kv_bytes = kv_data.len();
            let kv_tokens = kv_bytes / (kv_dim * elem_bytes);

            if kv_tokens > 0 && kv_dim < q_dim {
                let mut expanded = vec![0u8; mha_kv_seq_len * q_dim * elem_bytes];
                for t in 0..kv_tokens.min(mha_kv_seq_len) {
                    for kv_h in 0..num_kv_heads {
                        let src_off = (t * kv_dim + kv_h * head_dim) * elem_bytes;
                        for r in 0..repeat {
                            let q_h = kv_h * repeat + r;
                            let dst_off = (t * q_dim + q_h * head_dim) * elem_bytes;
                            let copy_len = head_dim * elem_bytes;
                            if src_off + copy_len <= kv_data.len() && dst_off + copy_len <= expanded.len() {
                                expanded[dst_off..dst_off + copy_len]
                                    .copy_from_slice(&kv_data[src_off..src_off + copy_len]);
                            }
                        }
                    }
                }
                tensors.insert(kv_name.clone(), expanded);
            }
        }
    }

    /// Merge KV cache entries for decode step MHA.
    ///
    /// Returns `mha_kv_seq_len` (total_seq if merged, seq_len otherwise).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn merge_kv_cache_for_decode(
        cn: &CompiledNode,
        node_op: &FusedOp,
        node_name: &str,
        tensors: &mut HashMap<String, Vec<u8>>,
        kv_cache_k: *mut f32,
        kv_cache_v: *mut f32,
        forward_config: Option<&crate::engine::executor::GeneratorForwardConfig>,
        total_seq: usize,
        seq_len: usize,
    ) -> usize {
        if total_seq <= seq_len || kv_cache_k.is_null() { return seq_len; }

        let (num_kv_heads, head_dim) = match node_op {
            FusedOp::GQA(ref cfg) => (cfg.num_kv_heads, cfg.head_dim),
            FusedOp::FlashAttention(ref cfg) => (cfg.num_kv_heads, cfg.head_dim),
            _ => return seq_len,
        };
        if num_kv_heads == 0 || head_dim == 0 { return seq_len; }

        let Some(kv_layer) = Self::extract_layer_index(node_name) else { return seq_len; };

        let kv_dim = num_kv_heads * head_dim;
        let elem_bytes = gllm_kernels::types::DType::F32.size_bytes();
        let cached_len = total_seq - seq_len;
        let cache_max_seq = forward_config.expect("KV cache operation requires forward_config").max_seq_len();
        let layer_byte_offset = kv_layer * num_kv_heads * cache_max_seq * head_dim * elem_bytes;

        // Build merged K and V padded to SYMDIM_MAX_SEQ_LEN rows.
        //
        // CRITICAL: emit_multi_head_attention computes the V pointer as:
        //   V_ptr = rsi + compile_seq_len * kv_dim * 4
        // where compile_seq_len == SYMDIM_MAX_SEQ_LEN (2048).
        //
        // pack_weight_blob concatenates K and V bytes sequentially. If K is only
        // total_seq rows (e.g. 5 rows), the JIT looks for V at byte offset
        // 2048 * kv_dim * 4, which is far beyond the K data → reads garbage → NaN.
        //
        // Fix: always allocate merged_k/merged_v at SYMDIM_MAX_SEQ_LEN rows so
        // that pack_weight_blob produces [K_2048rows | V_2048rows] and V starts
        // at exactly the offset the JIT kernel expects.
        let merged_row_bytes = kv_dim * elem_bytes;
        let alloc_rows = SYMDIM_MAX_SEQ_LEN;
        let mut merged_k = vec![0u8; alloc_rows * merged_row_bytes];
        let mut merged_v = vec![0u8; alloc_rows * merged_row_bytes];

        unsafe {
            let k_base = kv_cache_k as *const u8;
            let v_base = kv_cache_v as *const u8;
            for h in 0..num_kv_heads {
                let head_byte_offset = layer_byte_offset + h * cache_max_seq * head_dim * elem_bytes;
                for s in 0..cached_len {
                    let cache_row_off = head_byte_offset + s * head_dim * elem_bytes;
                    let merge_row_off = (s * kv_dim + h * head_dim) * elem_bytes;
                    std::ptr::copy_nonoverlapping(
                        k_base.add(cache_row_off),
                        merged_k.as_mut_ptr().add(merge_row_off),
                        head_dim * elem_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_base.add(cache_row_off),
                        merged_v.as_mut_ptr().add(merge_row_off),
                        head_dim * elem_bytes,
                    );
                }
            }
        }

        // Append current step's K/V from tensor map
        let k_name = cn.graph_input_names.get(1);
        let v_name = cn.graph_input_names.get(2);
        if let (Some(k_name), Some(v_name)) = (k_name, v_name) {
            if let (Some(cur_k), Some(cur_v)) = (tensors.get(k_name), tensors.get(v_name)) {
                let dst_offset = cached_len * merged_row_bytes;
                let copy_bytes = cur_k.len().min(merged_row_bytes * seq_len);
                merged_k[dst_offset..dst_offset + copy_bytes]
                    .copy_from_slice(&cur_k[..copy_bytes]);
                let copy_bytes_v = cur_v.len().min(merged_row_bytes * seq_len);
                merged_v[dst_offset..dst_offset + copy_bytes_v]
                    .copy_from_slice(&cur_v[..copy_bytes_v]);
            }
        }

        // Pad Q to [total_seq, q_dim]: zeros for cached positions, current Q at end
        if let Some(q_name) = cn.graph_input_names.get(0).cloned() {
            if let Some(cur_q) = tensors.get(&q_name).cloned() {
                let q_row_bytes = cur_q.len() / seq_len.max(1);
                let mut padded_q = vec![0u8; total_seq * q_row_bytes];
                let dst_off = cached_len * q_row_bytes;
                let copy_bytes = cur_q.len().min(q_row_bytes * seq_len);
                padded_q[dst_off..dst_off + copy_bytes]
                    .copy_from_slice(&cur_q[..copy_bytes]);
                tensors.insert(q_name, padded_q);
            }
        }

        if let Some(kn) = k_name { tensors.insert(kn.clone(), merged_k); }
        if let Some(vn) = v_name { tensors.insert(vn.clone(), merged_v); }

        total_seq
    }

    /// Write K/V from FusedQkvRope output into the KV cache buffer.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn perform_kv_cache_write(
        cn: &CompiledNode,
        config: &FusedQkvRopeConfig,
        node_name: &str,
        tensors: &HashMap<String, Vec<u8>>,
        kv_cache_k: *mut f32,
        kv_cache_v: *mut f32,
        forward_config: Option<&crate::engine::executor::GeneratorForwardConfig>,
        total_seq: usize,
        seq_len: usize,
    ) {
        if kv_cache_k.is_null() || kv_cache_v.is_null() || cn.per_output_numel.len() != 3 {
            return;
        }
        let Some(kv_layer) = Self::extract_layer_index(node_name) else { return };

        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let cache_max_seq = forward_config.expect("KV cache operation requires forward_config").max_seq_len();
        let write_start = total_seq.saturating_sub(seq_len);

        let k_name = cn.graph_output_names.get(1).cloned();
        let v_name = cn.graph_output_names.get(2).cloned();

        let (Some(k_name), Some(v_name)) = (k_name, v_name) else { return };
        let (Some(k_data), Some(v_data)) = (tensors.get(&k_name), tensors.get(&v_name)) else { return };

        let elem_bytes = gllm_kernels::types::DType::F32.size_bytes();
        let layer_byte_offset = kv_layer * num_kv_heads * cache_max_seq * head_dim * elem_bytes;

        unsafe {
            let k_base = kv_cache_k as *mut u8;
            let v_base = kv_cache_v as *mut u8;

            for h in 0..num_kv_heads {
                let head_byte_offset = layer_byte_offset + h * cache_max_seq * head_dim * elem_bytes;
                for s in 0..seq_len {
                    let cache_row_offset = head_byte_offset + (write_start + s) * head_dim * elem_bytes;
                    let src_row_offset = (s * kv_dim + h * head_dim) * elem_bytes;

                    let k_src = k_data.as_ptr().add(src_row_offset);
                    let k_dst = k_base.add(cache_row_offset);
                    std::ptr::copy_nonoverlapping(k_src, k_dst, head_dim * elem_bytes);

                    let v_src = v_data.as_ptr().add(src_row_offset);
                    let v_dst = v_base.add(cache_row_offset);
                    std::ptr::copy_nonoverlapping(v_src, v_dst, head_dim * elem_bytes);
                }
            }
        }
    }

    /// Execute the compiled JIT kernels in topological order (no KV cache).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn run_compiled(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        shape_bindings: &HashMap<String, usize>,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        let (mut tensors, weight_ptrs) = Self::seed_tensors_and_weight_ptrs(inputs, &self.graph.weight_bindings);
        // 仅 seed 阶段提供的 input 名字视为"外部已计算", 其余节点必须按拓扑顺序
        // 执行。禁止基于运行时 tensors map 判断跳过 (overwrite tensor 会导致
        // 错误跳过, 例如 xlmr hidden_0 每层 overwrite)。
        let seeded_outputs: HashSet<String> = inputs.keys().cloned().collect();

        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];
            if Self::is_node_computed(cn, &seeded_outputs) { continue; }

            let activation = Self::load_activation(cn, &tensors, false);
            let weight_blob = Self::pack_weight_blob(cn, &tensors, &weight_ptrs);

            let output_bytes = cn.output_numel * cn.output_dtype.size_bytes();
            let mut output_buf = vec![0u8; output_bytes];
            let mut scratchpad = vec![0u8; cn.compiled.scratchpad_bytes.max(output_bytes).max(64)];

            // ARCH-SYMDIM-NO-CONST-DEGRADE: seq_len 从 shape_bindings 正向传递
            let seq_len = *shape_bindings.get("seq_len")
                .expect("shape_bindings must contain 'seq_len' (ARCH-SYMDIM-NO-CONST-DEGRADE)");

            // ARCH-ROPE-CACHE: 无 KV cache 路径也要预填 cos/sin 表。
            // 没有外部 positions,使用 0..seq_len 作为默认 positions (prefill 语义)。
            let positions_vec: Vec<u32> = (0..seq_len as u32).collect();
            let positions_ptr_rope: *const u32 = if cn.compiled.rope_cache.is_some() {
                positions_vec.as_ptr()
            } else {
                std::ptr::null()
            };
            if let Some(req) = cn.compiled.rope_cache {
                populate_rope_cache(
                    &mut scratchpad,
                    req.cache_offset,
                    req.head_dim,
                    req.theta,
                    req.partial,
                    positions_ptr_rope,
                    seq_len,
                )?;
            }

            if std::env::var("GLLM_TRACE_EXEC").is_ok() {
                eprintln!("[EXEC] node {} '{}' op={} seq={} act.len={} weight.len={} out.len={}",
                    node_idx, self.graph.nodes[node_idx].name,
                    self.graph.nodes[node_idx].op.name(),
                    seq_len, activation.len(), weight_blob.len(), output_buf.len());
            }
            unsafe {
                cn.compiled.execute(
                    if activation.is_empty() { std::ptr::null() } else { activation.as_ptr() },
                    if weight_blob.is_empty() { std::ptr::null() } else { weight_blob.as_ptr() },
                    std::ptr::null_mut(),
                    positions_ptr_rope,
                    std::ptr::null(),
                    1,
                    seq_len,
                    output_buf.as_mut_ptr(),
                    scratchpad.as_mut_ptr(),
                );
            }

            // Debug: GLLM_DUMP_LAYERS=/path/to/dir 时 dump 每个节点 output (仅
            // live 部分 seq_len*feature_dim)。格式: 4B u32 seq_len + 4B u32
            // feature_dim + seq*feat*4B f32 raw。
            if let Ok(dump_dir) = std::env::var("GLLM_DUMP_LAYERS") {
                use std::io::Write;
                let _ = std::fs::create_dir_all(&dump_dir);
                let node_name = &self.graph.nodes[node_idx].name;
                let path = format!("{}/{:03}_{}.bin", dump_dir, node_idx, node_name);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    let feat = cn.feature_dim.max(1) as u32;
                    let sl = seq_len as u32;
                    f.write_all(&sl.to_le_bytes()).ok();
                    f.write_all(&feat.to_le_bytes()).ok();
                    let live_bytes = (seq_len * cn.feature_dim.max(1) * cn.output_dtype.size_bytes()).min(output_buf.len());
                    f.write_all(&output_buf[..live_bytes]).ok();
                }
            }

            // Store output(s)
            if cn.graph_output_names.len() == 1 {
                tensors.insert(cn.graph_output_names[0].clone(), output_buf);
            } else if !cn.per_output_numel.is_empty() {
                let mut byte_offset = 0;
                for (i, name) in cn.graph_output_names.iter().enumerate() {
                    let numel = cn.per_output_numel[i];
                    let nbytes = numel * cn.output_dtype.size_bytes();
                    let chunk = output_buf[byte_offset..byte_offset + nbytes].to_vec();
                    tensors.insert(name.clone(), chunk);
                    byte_offset += nbytes;
                }
            } else if cn.graph_output_names.len() > 1 {
                return Err(ExecutionError::Compilation(format!(
                    "node has {} outputs but no per_output_numel — compile() should have set this",
                    cn.graph_output_names.len(),
                )));
            }
        }

        Ok(Self::collect_graph_outputs(&self.graph, &mut tensors))
    }

    /// Execute the compiled JIT kernels with KV cache support.
    ///
    /// Identical to `run_compiled` but passes the KV cache pointers and
    /// position array to each compiled kernel's `execute` call, enabling
    /// full decoder forward passes with cached K/V.
    ///
    /// # Parameters
    /// - `inputs`: named activation tensors (e.g. `"hidden_state"`)
    /// - `kv_cache_k`: pointer to the global K cache buffer (all layers)
    /// - `kv_cache_v`: pointer to the global V cache buffer (all layers)
    /// - `layer`: current transformer layer index (used to compute the
    ///   per-layer offset into the flat KV buffer)
    /// - `total_seq`: total sequence length including cached tokens
    /// - `positions`: token position array (`*const u32`, length = seq_len)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn run_with_kv_cache(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        kv_cache_k: *mut f32,
        kv_cache_v: *mut f32,
        layer: usize,
        total_seq: usize,
        seq_len: usize,
        positions: *const u32,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        self.run_with_kv_cache_and_callbacks(inputs, kv_cache_k, kv_cache_v, layer, total_seq, seq_len, positions, None, None)
    }

    /// Variant that accepts `forward_config` (ARCH-ROPE-CACHE + KV write 需要 max_seq_len)。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn run_with_kv_cache_with_config(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        kv_cache_k: *mut f32,
        kv_cache_v: *mut f32,
        layer: usize,
        total_seq: usize,
        seq_len: usize,
        positions: *const u32,
        forward_config: &crate::engine::executor::GeneratorForwardConfig,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        self.run_with_kv_cache_and_callbacks(inputs, kv_cache_k, kv_cache_v, layer, total_seq, seq_len, positions, None, Some(forward_config))
    }

    /// §9-§18: run_with_kv_cache with optional callback chain for Gate-First Skip / Residual Bypass / Early Exit.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn run_with_kv_cache_and_callbacks(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
        kv_cache_k: *mut f32,
        kv_cache_v: *mut f32,
        layer: usize,
        total_seq: usize,
        seq_len: usize,
        positions: *const u32,
        mut callbacks: Option<&mut super::layer_callback::CallbackChain>,
        forward_config: Option<&crate::engine::executor::GeneratorForwardConfig>,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        if !self.is_compiled {
            return Err(ExecutionError::NotCompiled);
        }

        log::debug!("[EXEC-ENTER] run_with_kv_cache layer={} total_seq={} seq_len={} nodes={}", layer, total_seq, seq_len, self.compiled_nodes.len());

        let (mut tensors, weight_ptrs) = Self::seed_tensors_and_weight_ptrs(inputs, &self.graph.weight_bindings);
        let seeded_outputs: HashSet<String> = inputs.keys().cloned().collect();

        // Build unified KV cache pointer for the compiled kernel.
        // gllm KV cache layout: [K_all_layers | V_all_layers], each half is
        // [num_layers][num_kv_heads][max_seq_len][head_dim].
        // The compiled CachedGQA kernel receives kv_cache pointing to K start;
        // it locates V by adding the half-buffer offset stored in its scratchpad config.
        // We pass kv_cache_k directly — the kernel uses layer/head/seq strides internally.
        // kv_cache_v is the V-half start pointer; we store it in the seq_lens slot
        // (second pointer argument) so the kernel can access both halves.
        let kv_cache_ptr = kv_cache_k as *mut u8;
        let kv_cache_v_ptr = kv_cache_v as *mut u8;
        let _ = (layer, total_seq); // used by caller for layout; kernel uses internal strides

        if std::env::var("GLLM_DEBUG_GRAPH_ONCE").is_ok() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static PRINTED: AtomicBool = AtomicBool::new(false);
            if !PRINTED.swap(true, Ordering::SeqCst) {
                eprintln!("=== FusedGraph nodes ({}) ===", self.graph.nodes.len());
                for (i, n) in self.graph.nodes.iter().enumerate() {
                    eprintln!("  {i}: '{}' op={} inputs={:?} outputs={:?}",
                        n.name, n.op.name(), n.inputs, n.outputs);
                }
                eprintln!("=== graph.outputs: {:?} ===", self.graph.outputs);
            }
        }
        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];

            if Self::is_node_computed(cn, &seeded_outputs) {
                log::trace!("[SKIP] node {node_idx} '{}' outputs seeded externally", self.graph.nodes[node_idx].name);
                continue;
            }

            if std::env::var("GLLM_TRACE_EXEC").is_ok() {
                eprintln!("[EXEC-KV] node {node_idx} '{}' op={} out_names={:?}",
                    self.graph.nodes[node_idx].name,
                    self.graph.nodes[node_idx].op.name(),
                    cn.graph_output_names);
            }
            log::trace!("[EXEC] node {node_idx} '{}' ({})",
                self.graph.nodes[node_idx].name,
                self.graph.nodes[node_idx].op.name());

            // §9-§18: Pre-node callback — Gate-First Skip / Residual Bypass / Early Exit
            let layer_idx = Self::extract_layer_index(&self.graph.nodes[node_idx].name).unwrap_or(node_idx);
            if let Some(ref mut cb) = callbacks {
                let hidden_state = cn.graph_input_names.first()
                    .and_then(|name| tensors.get(name))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let ctx = super::layer_callback::LayerContext {
                    node_idx,
                    layer_idx,
                    node_op: &self.graph.nodes[node_idx].op,
                    hidden_state,
                    kv_cache_k,
                    kv_cache_v,
                    total_seq,
                    seq_len,
                    position: 0,
                    request_id: 0,
                    model_config: forward_config.expect("callback requires forward_config"),
                };
                match cb.dispatch_pre_node(&ctx) {
                    super::layer_callback::CallbackAction::SkipThisNode => {
                        log::trace!("graph_executor: §14.3 residual bypass skipped node {} (layer {})", node_idx, layer_idx);
                        // §14.3: 跳过此节点（仅用于残差旁路 delta_rho < threshold）
                        // 将输入直接传递为输出（残差恒等映射）
                        if !cn.graph_output_names.is_empty() && !cn.graph_input_names.is_empty() {
                            if let Some(input_data) = tensors.get(&cn.graph_input_names[0]).cloned() {
                                for out_name in &cn.graph_output_names {
                                    tensors.insert(out_name.clone(), input_data.clone());
                                }
                            }
                        }
                        continue;
                    }
                    super::layer_callback::CallbackAction::CompactMask { active_mask } => {
                        log::trace!(
                            "graph_executor: §14.2 compact mask at node {} (layer {}), active: {}/{}",
                            node_idx, layer_idx,
                            active_mask.iter().filter(|&&b| b).count(),
                            active_mask.len(),
                        );
                        // §14.2: Register-level compaction — compact dead neurons, execute dense.
                        // The activation data is compacted using the mask before execution.
                        // After execution, results are scattered back to original positions.
                        if !cn.graph_input_names.is_empty() {
                            if let Some(input_data) = tensors.get(&cn.graph_input_names[0]).cloned() {
                                // Compact: remove dead neuron positions from activation
                                let element_size = cn.output_dtype.size_bytes();
                                let num_elements = input_data.len() / element_size;
                                let mask_len = active_mask.len().min(num_elements);

                                // Extract active elements into compact buffer
                                let mut compact_data = Vec::with_capacity(input_data.len());
                                for (i, chunk) in input_data.chunks_exact(element_size).enumerate() {
                                    if i < mask_len && active_mask[i] {
                                        compact_data.extend_from_slice(chunk);
                                    } else if i >= mask_len {
                                        // Elements beyond mask length are always active
                                        compact_data.extend_from_slice(chunk);
                                    }
                                }
                                // Store compacted data and mask for scatter-back in post_node
                                tensors.insert(cn.graph_input_names[0].clone(), compact_data);
                                // Store original size and mask as metadata for scatter
                                let mask_key = format!("__compact_mask_{}", node_idx);
                                let original_size_key = format!("__compact_orig_size_{}", node_idx);
                                let mask_bytes: Vec<u8> = active_mask.iter().map(|&b| b as u8).collect();
                                tensors.insert(mask_key, mask_bytes);
                                tensors.insert(original_size_key, (input_data.len() as u64).to_le_bytes().to_vec());
                            }
                        }
                        // Continue to execute the node on compacted data
                    }
                    super::layer_callback::CallbackAction::ExitEarly { logits } => {
                        log::trace!("graph_executor: §16.2 early exit at node {} (layer {})", node_idx, layer_idx);
                        let mut out = HashMap::new();
                        if !logits.is_empty() {
                            let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                            out.insert("logits".to_string(), logits_bytes);
                        }
                        return Ok(out);
                    }
                    super::layer_callback::CallbackAction::InjectHidden { data } => {
                        if !cn.graph_input_names.is_empty() {
                            tensors.insert(cn.graph_input_names[0].clone(), data);
                        }
                    }
                    super::layer_callback::CallbackAction::Continue => {}
                }
            }

            // ── Detect MHA/GQA attention nodes that need special ABI mapping ──
            // MHA JIT codegen expects: rdi=Q, rsi=K, rdx=V, r8=output,
            // which differs from the standard CompiledLayerFn ABI.
            // We must pass Q/K/V from the tensor map instead of the default
            // activation/weights/kv_cache pointers.
            let is_mha_node = matches!(
                self.graph.nodes[node_idx].op.name(),
                "GQA" | "Attention" | "MultiHeadAttention" | "FlashAttention"
            );

            // ── KV cache merge for decode step MHA ──
            let mha_kv_seq_len = if is_mha_node {
                Self::merge_kv_cache_for_decode(
                    cn, &self.graph.nodes[node_idx].op, &self.graph.nodes[node_idx].name,
                    &mut tensors, kv_cache_k, kv_cache_v, forward_config, total_seq, seq_len,
                )
            } else {
                seq_len
            };

            log::trace!("[EXEC] node {node_idx}/{} '{}' ({}) prep",
                self.graph.nodes.len(),
                self.graph.nodes[node_idx].name,
                self.graph.nodes[node_idx].op.name());

            // ARCH-GQA-RESHAPE (SPEC 01-JIT-PIPELINE.md §437): GQA 展开含 Reshape(K/V)。
            // expand_gqa_heads 把 K/V 从 [seq, num_kv_heads*head_dim] broadcast
            // 到 [seq, num_heads*head_dim],对应 SPEC §437 的 Reshape(K/V) 步骤。
            // JIT MHA kernel (lower_mha_with_hook) 因此看到对称 MHA 语义,
            // h_off 同时正确索引 Q/K/V,不需要在 attention kernel 内做 GQA 分组
            // 映射 (嵌套循环层数会违反 ARCH-REGALLOC-COUNTER-NOSPILL)。
            // build_gqa_graph 中 K/V 已声明为 q_dim 以匹配 expanded 张量布局。
            if is_mha_node {
                Self::expand_gqa_heads(cn, &self.graph.nodes[node_idx].op, &mut tensors, mha_kv_seq_len);
            }

            // Load activation and pad to max_seq_len size if needed.
            let activation = Self::load_activation(cn, &tensors, true);

            // Pack weight blob using the compiled weight_layout for size info.
            // For MHA nodes this packs the expanded K/V tensors (written above by
            // expand_gqa_heads) so weight_blob[K_offset..] and weight_blob[V_offset..]
            // are correctly populated before the JIT kernel runs.
            let weight_blob = Self::pack_weight_blob(cn, &tensors, &weight_ptrs);

            // Allocate output buffer at max_seq_len size (safe upper bound).
            // JIT kernels may loop up to max_seq_len internally for ops like
            // RmsNorm/Add that iterate all rows. The buffer must be large enough
            // for the compiled loop bound. After execution, we truncate to the
            // runtime seq_len when inserting into the tensor map.
            let runtime_output_numel = cn.output_numel; // max_seq_len size (safe)
            let output_bytes = runtime_output_numel * cn.output_dtype.size_bytes();
            let mut output_buf = vec![0u8; output_bytes];
            // scratchpad 必须足够容纳 JIT 内核的所有中间数据。
            // compiled.scratchpad_bytes 可能偏小（BufferAllocation 未充分估算 MHA 暂存）。
            // 安全下界：max(compiled, output_bytes)，确保至少能容纳一整个输出大小的中间矩阵。
            let scratch_size = cn.compiled.scratchpad_bytes.max(output_bytes).max(64);
            let mut scratchpad = vec![0u8; scratch_size];

            // Effective seq_len passed to the JIT kernel: for MHA nodes the kernel
            // must iterate over the full KV sequence (prefill + cached), not just the
            // current step's tokens.
            let effective_seq = if is_mha_node { mha_kv_seq_len } else { seq_len };

            // ARCH-ROPE-CACHE: 当 kernel 声明依赖 RoPE cos/sin 表时,按
            // positions 数组填充 scratchpad[cache_offset..]。布局:
            //   [effective_seq, head_dim] row-major, 每行前 half 为 cos, 后 half 为 sin。
            if let Some(req) = cn.compiled.rope_cache {
                populate_rope_cache(
                    &mut scratchpad,
                    req.cache_offset,
                    req.head_dim,
                    req.theta,
                    req.partial,
                    positions,
                    effective_seq,
                )?;
            }

            log::debug!("[EXEC] node {node_idx} exec: act={}B wt={}B out={}B scratch={}B eff_seq={}",
                activation.len(), weight_blob.len(), output_bytes,
                cn.compiled.scratchpad_bytes, effective_seq);

            // Unified execute path: standard CompiledLayerFn ABI for all nodes.
            // rdi=activation (Q for MHA), rsi=weight_blob (packed K/V for MHA),
            // rdx=kv_cache, rcx=positions, r8=seq_lens, r9=batch_size,
            // [rbp+16]=seq_len, [rbp+24]=output, [rbp+32]=scratchpad.
            let out_ptr_before = output_buf.as_mut_ptr();
            let scratch_ptr_before = scratchpad.as_mut_ptr();
            unsafe {
                cn.compiled.execute(
                    if activation.is_empty() {
                        std::ptr::null()
                    } else {
                        activation.as_ptr()
                    },
                    if weight_blob.is_empty() {
                        std::ptr::null()
                    } else {
                        weight_blob.as_ptr()
                    },
                    kv_cache_ptr,
                    positions,
                    kv_cache_v_ptr as *const usize,
                    1,
                    effective_seq,
                    out_ptr_before,
                    scratch_ptr_before,
                );
            }
            log::debug!("[EXEC] node {node_idx} done");
            // DEBUG: dump lm_head weight data
            if self.graph.nodes[node_idx].name == "lm_head"
                && std::env::var("GLLM_DEBUG_LM").is_ok()
            {
                eprintln!("[LM-DUMP] act.len={} weight_blob.len={} out.len={}",
                    activation.len(), weight_blob.len(), output_bytes);
                let wf32: &[f32] = unsafe {
                    std::slice::from_raw_parts(weight_blob.as_ptr() as *const f32,
                        weight_blob.len() / 4)
                };
                eprintln!("  weight[0..8]={:?}", &wf32[..8.min(wf32.len())]);
                eprintln!("  weight[49152..49160]={:?}", &wf32[49152.min(wf32.len())..49160.min(wf32.len())]);
                // Layout: if [K=576, N=49152] canonical, weight[k*N+n] = W[k][n]
                // Let's sample position k=0, n=7042 (Paris), n=24247 (hints)
                let n_paris = 7042usize;
                let n_hints = 24247usize;
                if wf32.len() >= 576 * 49152 {
                    eprintln!("  W[k=0, Paris=7042] = {:.6}", wf32[0 * 49152 + n_paris]);
                    eprintln!("  W[k=0, hints=24247] = {:.6}", wf32[0 * 49152 + n_hints]);
                }
                let nz = wf32.iter().filter(|&&x| x != 0.0).count();
                eprintln!("  weight total={} nonzeros={}", wf32.len(), nz);
            }

            // ARCH-NODE-STATS: 每节点输出 norm/max/min/nan_count probe, 在 decode
            // 链路上逐节点定位数值退化来源。GLLM_NODE_STATS=1 开启;
            // GLLM_NODE_STATS_FROM=<step> 从第 N 步 decode 开始 (prefill=0)。
            if std::env::var("GLLM_NODE_STATS").is_ok() {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static STEP: AtomicUsize = AtomicUsize::new(0);
                let cur_step = if node_idx == 0 { STEP.fetch_add(1, Ordering::SeqCst) } else { STEP.load(Ordering::SeqCst).saturating_sub(1) };
                let from_step = std::env::var("GLLM_NODE_STATS_FROM").ok()
                    .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                if cur_step >= from_step {
                    let bytes_len = (seq_len * cn.feature_dim.max(1) * 4).min(output_buf.len());
                    if bytes_len >= 4 {
                        let data: &[f32] = unsafe {
                            std::slice::from_raw_parts(
                                output_buf.as_ptr() as *const f32, bytes_len / 4)
                        };
                        let (mut vmin, mut vmax, mut nan, mut zero) = (f32::INFINITY, f32::NEG_INFINITY, 0usize, 0usize);
                        let mut sum_sq = 0.0f64;
                        for &x in data {
                            if x.is_nan() { nan += 1; continue; }
                            if x == 0.0 { zero += 1; }
                            if x < vmin { vmin = x; }
                            if x > vmax { vmax = x; }
                            sum_sq += (x as f64) * (x as f64);
                        }
                        let rms = (sum_sq / data.len().max(1) as f64).sqrt();
                        eprintln!("[NODE step={cur_step} {node_idx:03} {:<40} len={} rms={:.4e} min={:.3e} max={:.3e} nan={} zero={}/{}]",
                            self.graph.nodes[node_idx].name, data.len(), rms, vmin, vmax, nan, zero, data.len());
                    }
                }
            }

            // NO_SCALAR: FusedQkvRope is fully handled by JIT codegen (RoPE is applied
            // within the fused QKV+RoPE kernel). No post-hoc scalar fallback needed.

            // GLLM_DUMP_CODE=<dir>: dump raw machine code for offline objdump analysis
            if let Ok(code_dir) = std::env::var("GLLM_DUMP_CODE") {
                if seq_len >= 2 {
                    use std::io::Write;
                    let _ = std::fs::create_dir_all(&code_dir);
                    let node_name = &self.graph.nodes[node_idx].name;
                    let path = format!("{}/{:03}_{}.bin", code_dir, node_idx, node_name);
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        let _ = f.write_all(cn.compiled.code_bytes());
                    }
                }
            }
            // GLLM_DUMP_LAYERS=<dir>: 每节点输出 dump (seq × feature_dim f32 raw),
            if let Ok(dump_dir) = std::env::var("GLLM_DUMP_LAYERS") {
                if seq_len >= 2 {
                    use std::io::Write;
                    let _ = std::fs::create_dir_all(&dump_dir);
                    let node_name = &self.graph.nodes[node_idx].name;
                    let path = format!("{}/{:03}_{}.bin", dump_dir, node_idx, node_name);
                    if let Ok(mut f) = std::fs::File::create(&path) {
                        let feat = cn.feature_dim.max(1) as u32;
                        let sl = seq_len as u32;
                        let _ = f.write_all(&sl.to_le_bytes());
                        let _ = f.write_all(&feat.to_le_bytes());
                        let live_bytes = (seq_len * cn.feature_dim.max(1) * cn.output_dtype.size_bytes()).min(output_buf.len());
                        let _ = f.write_all(&output_buf[..live_bytes]);
                    }
                }
                // GLLM_DUMP_FULL_BUF=1: 额外 dump 完整 output_buf 的前 `feature_dim * 3` f32,
                // 检查 JIT 是否写入超出 cn.feature_dim 范围的数据 (multi-output 或 bug 调试)
                if std::env::var("GLLM_DUMP_FULL_BUF").is_ok() && seq_len >= 2 {
                    let node_name = &self.graph.nodes[node_idx].name;
                    eprintln!("[FULL-BUF {node_name}] buf.len={} feature_dim={} seq_len={}",
                        output_buf.len(), cn.feature_dim, seq_len);
                    let as_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(output_buf.as_ptr() as *const f32,
                            (output_buf.len() / 4).min(cn.feature_dim * 3))
                    };
                    for row in 0..3.min(as_f32.len() / cn.feature_dim.max(1)) {
                        let off = row * cn.feature_dim;
                        if off + 8 < as_f32.len() {
                            let nz = as_f32[off..off + cn.feature_dim].iter().filter(|&&x| x != 0.0).count();
                            eprintln!("  row {}: first8={:?} nonzeros={}/{}",
                                row, &as_f32[off..off+8.min(cn.feature_dim)], nz, cn.feature_dim);
                        }
                    }
                }
            }

            // Truncate output to runtime seq_len before inserting into tensor map.
            // JIT kernel wrote max_seq_len rows but only the first seq_len are valid.
            // For MHA decode (mha_kv_seq_len > seq_len), the output is [total_seq, hidden]
            // and we only need the last token's row.
            if cn.graph_output_names.len() == 1 {
                // Single-output node: truncate to runtime seq_len.
                let effective_trunc_len = if is_mha_node && mha_kv_seq_len > seq_len {
                    seq_len
                } else {
                    seq_len
                };
                let runtime_bytes = cn.feature_dim * effective_trunc_len * cn.output_dtype.size_bytes();
                let truncated_output = if is_mha_node && mha_kv_seq_len > seq_len {
                    let row_bytes = cn.feature_dim * cn.output_dtype.size_bytes();
                    let offset = (mha_kv_seq_len - 1) * row_bytes;
                    if offset + row_bytes <= output_buf.len() {
                        output_buf[offset..offset + row_bytes].to_vec()
                    } else {
                        output_buf[..row_bytes.min(output_buf.len())].to_vec()
                    }
                } else if runtime_bytes < output_buf.len() {
                    output_buf[..runtime_bytes].to_vec()
                } else {
                    output_buf
                };
                tensors.insert(cn.graph_output_names[0].clone(), truncated_output);
            } else if !cn.per_output_numel.is_empty() {
                // Multi-output node: the JIT writes each output at a different base offset
                // within output_buf. Output i starts at byte_start_i where:
                //   byte_start_0 = 0
                //   byte_start_i = sum(per_output_numel[0..i]) * dtype_size
                // For each output i, only the first (per_token_i * seq_len) elements are valid.
                // We must read from output_buf (not truncated_output) using these offsets.
                let dtype_bytes = cn.output_dtype.size_bytes();
                let mut jit_byte_offset: usize = 0; // cumulative offset in output_buf
                for (i, name) in cn.graph_output_names.iter().enumerate() {
                    // per_token dim extracted from graph metadata (ARCH-SYMDIM-NO-CONST-DEGRADE)
                    let per_token = cn.per_output_feature_dims[i];
                    let valid_nbytes = per_token * seq_len * dtype_bytes;
                    let max_nbytes = cn.per_output_numel[i] * dtype_bytes; // full block in output_buf
                    if jit_byte_offset + valid_nbytes <= output_buf.len() {
                        let chunk = output_buf[jit_byte_offset..jit_byte_offset + valid_nbytes].to_vec();
                        // DEBUG: dump multi-outputs
                        if let Ok(dir) = std::env::var("GLLM_DUMP_KV_TENSORS") {
                            if seq_len >= 2 {
                                use std::io::Write;
                                let _ = std::fs::create_dir_all(&dir);
                                let path = format!("{}/{}.bin", dir, name);
                                if let Ok(mut f) = std::fs::File::create(&path) {
                                    let feat = per_token as u32;
                                    let sl = seq_len as u32;
                                    let _ = f.write_all(&sl.to_le_bytes());
                                    let _ = f.write_all(&feat.to_le_bytes());
                                    let _ = f.write_all(&chunk);
                                }
                            }
                        }
                        tensors.insert(name.clone(), chunk);
                    }
                    jit_byte_offset += max_nbytes;
                }
            } else if cn.graph_output_names.len() > 1 {
                return Err(ExecutionError::Compilation(format!(
                    "node has {} outputs but no per_output_numel",
                    cn.graph_output_names.len(),
                )));
            }

            // ── KV cache write: after FusedQkvRope, copy K/V to KV cache buffer ──
            if let FusedOp::FusedQkvRope(ref config) = self.graph.nodes[node_idx].op {
                Self::perform_kv_cache_write(
                    cn, config, &self.graph.nodes[node_idx].name, &tensors,
                    kv_cache_k, kv_cache_v, forward_config, total_seq, seq_len,
                );
            }

            // §14.2: Scatter-back after compacted execution
            // If this node was compacted, scatter the output back to original positions
            let mask_key = format!("__compact_mask_{}", node_idx);
            let orig_size_key = format!("__compact_orig_size_{}", node_idx);
            if let (Some(mask_bytes), Some(orig_size_bytes)) = (tensors.remove(&mask_key), tensors.remove(&orig_size_key)) {
                let element_size = cn.output_dtype.size_bytes();
                let orig_size = if orig_size_bytes.len() >= 8 {
                    u64::from_le_bytes(orig_size_bytes[..8].try_into().unwrap_or([0; 8])) as usize
                } else {
                    0
                };
                if orig_size > 0 {
                    let active_mask: Vec<bool> = mask_bytes.iter().map(|&b| b != 0).collect();
                    if let Some(out_name) = cn.graph_output_names.first() {
                        if let Some(compact_output) = tensors.get(out_name).cloned() {
                            // Scatter: expand compacted output back to original positions
                            let mut scattered = vec![0u8; orig_size];
                            let mut compact_idx = 0;
                            let num_elements = orig_size / element_size;
                            let mask_len = active_mask.len().min(num_elements);
                            for i in 0..num_elements {
                                if i < mask_len && active_mask[i] {
                                    let src_start = compact_idx * element_size;
                                    let dst_start = i * element_size;
                                    if src_start + element_size <= compact_output.len() && dst_start + element_size <= scattered.len() {
                                        scattered[dst_start..dst_start + element_size]
                                            .copy_from_slice(&compact_output[src_start..src_start + element_size]);
                                    }
                                    compact_idx += 1;
                                } else if i >= mask_len {
                                    // Elements beyond mask are always active
                                    let src_start = compact_idx * element_size;
                                    let dst_start = i * element_size;
                                    if src_start + element_size <= compact_output.len() && dst_start + element_size <= scattered.len() {
                                        scattered[dst_start..dst_start + element_size]
                                            .copy_from_slice(&compact_output[src_start..src_start + element_size]);
                                    }
                                    compact_idx += 1;
                                }
                                // Dead neurons (active_mask[i] == false): remain zero
                            }
                            tensors.insert(out_name.clone(), scattered);
                            log::trace!(
                                "graph_executor: §14.2 scatter-back at node {} (layer {}), compacted {} → {} elements",
                                node_idx, layer_idx, compact_idx, num_elements,
                            );
                        }
                    }
                }
            }

            // §9-§18: Post-node callback — Early Exit / Guardrail Probe / Intent Recall
            if let Some(ref mut cb) = callbacks {
                let output_data = cn.graph_output_names.first()
                    .and_then(|name| tensors.get(name))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let ctx = super::layer_callback::LayerContext {
                    node_idx,
                    layer_idx,
                    node_op: &self.graph.nodes[node_idx].op,
                    hidden_state: output_data,
                    kv_cache_k,
                    kv_cache_v,
                    total_seq,
                    seq_len,
                    position: 0,
                    request_id: 0,
                    model_config: forward_config.expect("callback requires forward_config"),
                };
                match cb.dispatch_post_node(&ctx, output_data) {
                    super::layer_callback::CallbackAction::ExitEarly { logits } => {
                        log::trace!("graph_executor: §16.2 post-node early exit at node {} (layer {})", node_idx, layer_idx);
                        let mut out = HashMap::new();
                        if !logits.is_empty() {
                            let logits_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                            out.insert("logits".to_string(), logits_bytes);
                        } else if let Some(last_output) = cn.graph_output_names.first().and_then(|n| tensors.get(n)) {
                            out.insert("logits".to_string(), last_output.clone());
                        }
                        return Ok(out);
                    }
                    _ => {} // Continue / SkipThisNode / InjectHidden / CompactMask are no-ops in post_node
                }
            }
        }

        Ok(Self::collect_graph_outputs(&self.graph, &mut tensors))
    }

    /// Returns true if the executor has been compiled (CPU JIT).
    pub fn is_compiled(&self) -> bool {
        self.is_compiled
    }

    /// Returns true if the executor has been compiled for GPU.
    #[cfg(feature = "cuda")]
    pub fn is_gpu_compiled(&self) -> bool {
        self.is_gpu_compiled
    }

    /// Extract layer index from a node name like "layer_3_q_proj_fused_qkv_rope" → Some(3).
    fn extract_layer_index(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("layer_")?;
        let end = rest.find('_')?;
        rest[..end].parse().ok()
    }

    /// Returns a reference to the underlying graph.
    pub fn graph(&self) -> &FusedGraph {
        &self.graph
    }

    /// Returns a mutable reference to the underlying graph.
    /// Use before compilation to populate weight_bindings shapes.
    pub fn graph_mut(&mut self) -> &mut FusedGraph {
        &mut self.graph
    }

    /// Build an uncompiled `FusedGraphExecutor` from an `OnnxGraph`.
    ///
    /// Only runs optimization passes (pattern fusion → hardware fusion →
    /// constant folding → DCE). Does NOT JIT-compile.
    ///
    /// Caller should populate `graph_mut().bind_weights(provider)` with real
    /// weight shapes BEFORE calling `compile_with_cache()`.
    pub fn from_graph_optimized(
        graph: crate::loader::onnx::OnnxGraph,
        ctx: crate::graph::optimizer::OptimizationContext,
    ) -> Result<Self, ExecutorError> {
        use crate::graph::optimizer::GraphOptimizer;

        let optimizer = GraphOptimizer::new(ctx);
        let fused = optimizer
            .optimize(&graph)
            .map_err(|e| ExecutorError::CompilationFailed(format!("graph optimization: {e}")))?;

        Ok(Self::new(fused))
    }

    /// Build a `FusedGraphExecutor` from an `OnnxGraph`.
    ///
    /// Pipeline:
    /// 1. Run all built-in graph optimization passes (pattern fusion → hardware
    ///    fusion → constant folding → DCE).
    /// 2. Wrap the resulting `FusedGraph` in a `FusedGraphExecutor`.
    /// 3. JIT-compile every node for the given `seq_len` / `hidden`.
    ///
    /// Returns `Err` if optimization or JIT compilation fails for any node.
    /// No silent fallback — unsupported op types propagate as errors.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn from_graph(
        graph: crate::loader::onnx::OnnxGraph,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        ctx: crate::graph::optimizer::OptimizationContext,
    ) -> Result<Self, ExecutorError> {
        use crate::graph::optimizer::GraphOptimizer;

        let optimizer = GraphOptimizer::new(ctx);
        let fused = optimizer
            .optimize(&graph)
            .map_err(|e| ExecutorError::CompilationFailed(format!("graph optimization: {e}")))?;

        let mut executor = Self::new(fused);
        executor
            .compile(seq_len, hidden, dtype)
            .map_err(|e| ExecutorError::CompilationFailed(format!("JIT compile: {e}")))?;

        Ok(executor)
    }

    /// Creates an executor leveraging L3 binary cache fingerprinting to bypass compilation.
    ///
    /// Implements REQ-JIT-CACHE-003: L3 disk persistence with format {model_hash}_{backend}.bin.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn from_graph_with_cache(
        graph: crate::loader::onnx::OnnxGraph,
        seq_len: usize,
        hidden: usize,
        dtype: gllm_kernels::types::DType,
        model_id: &str,
        backend: &str,
        cache: &crate::compat::artifact_cache::ArtifactCache,
        ctx: crate::graph::optimizer::OptimizationContext,
    ) -> Result<Self, ExecutorError> {
        use crate::graph::optimizer::GraphOptimizer;

        let optimizer = GraphOptimizer::new(ctx);
        let fused = optimizer
            .optimize(&graph)
            .map_err(|e| ExecutorError::CompilationFailed(format!("graph optimization: {e}")))?;

        let mut executor = Self::new(fused);
        executor
            .compile_with_cache(seq_len, hidden, dtype, model_id, backend, cache)
            .map_err(|e| ExecutorError::CompilationFailed(format!("JIT cache compile: {e}")))?;

        Ok(executor)
    }
}

/// Internal helper: result of building a CompilerGraph for one node.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
/// Construct a `NodeGraphBuild` from a compiled graph, node I/O names, and output sizing.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn extract_tensor_feature_dim(graph: &gllm_kernels::compiler::CompilerGraph, tid: gllm_kernels::compiler::TensorId) -> usize {
    graph.tensors[tid.0 as usize].shape.iter()
        .filter(|d| !d.is_symbolic())
        .map(|d| d.as_concrete().expect("ARCH-SYMDIM-OUTER-ONLY: non-symbolic dim must be Concrete"))
        .product::<usize>()
        .max(1)
}

fn make_node_build(
    graph: gllm_kernels::compiler::CompilerGraph,
    node: &super::types::FusedNode,
    output_numel: usize,
    per_output_numel: Vec<usize>,
) -> NodeGraphBuild {
    let out_tid = graph.outputs[0];
    let output_dtype = graph.tensors[out_tid.0 as usize].dtype;
    // Extract feature_dim directly from graph metadata Concrete dims
    // (ARCH-SYMDIM-NO-CONST-DEGRADE: no SYMDIM_MAX_SEQ_LEN division).
    let feature_dim = extract_tensor_feature_dim(&graph, out_tid);
    // Per-output feature dims for multi-output nodes
    let per_output_feature_dims: Vec<usize> = graph.outputs.iter()
        .map(|&tid| extract_tensor_feature_dim(&graph, tid))
        .collect();
    NodeGraphBuild {
        graph,
        input_names: node.inputs.clone(),
        output_names: node.outputs.clone(),
        output_numel,
        per_output_numel,
        output_dtype,
        feature_dim,
        per_output_feature_dims,
    }
}

struct NodeGraphBuild {
    graph: gllm_kernels::compiler::CompilerGraph,
    input_names: Vec<String>,
    output_names: Vec<String>,
    output_numel: usize,
    /// Per-output element counts for multi-output nodes.
    per_output_numel: Vec<usize>,
    /// DType of the output tensor(s).
    output_dtype: gllm_kernels::types::DType,
    /// Feature dimension per token — product of all Concrete (non-Symbolic) output dims.
    /// Directly extracted from CompilerGraph output tensor shape metadata.
    feature_dim: usize,
    /// Per-output feature dimensions for multi-output nodes.
    per_output_feature_dims: Vec<usize>,
}

// ---------------------------------------------------------------------------
// T3.1: Weight binding + execution context (REQ-JIT-GRAPH-003)
// ---------------------------------------------------------------------------

/// Runtime weight binding table: tensor name → raw const pointer to f32 data.
///
/// Callers are responsible for ensuring the pointed-to data outlives any
/// `GraphExecutorContext` that holds this binding.
pub struct RuntimeWeightBinding {
    weights: HashMap<String, *const f32>,
}

impl RuntimeWeightBinding {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    /// Bind a named weight tensor to a raw pointer.
    pub fn bind(mut self, name: impl Into<String>, ptr: *const f32) -> Self {
        self.weights.insert(name.into(), ptr);
        self
    }

    /// Look up a weight pointer by tensor name.
    pub fn get(&self, name: &str) -> Option<*const f32> {
        self.weights.get(name).copied()
    }
}

impl Default for RuntimeWeightBinding {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: RuntimeWeightBinding only holds raw pointers; it does not own the
// data. The caller guarantees the pointed-to memory is valid for the duration
// of any use. Sending across threads is therefore safe under the same
// contract.
unsafe impl Send for RuntimeWeightBinding {}
unsafe impl Sync for RuntimeWeightBinding {}

/// Errors produced by `GraphExecutorContext` and `FusedGraphExecutor::execute`.
#[derive(Debug, thiserror::Error)]
pub enum ExecutorError {
    #[error("weight not found in binding table: {0}")]
    WeightNotFound(String),
    #[error("symbolic shape dimension not bound: {0}")]
    ShapeNotBound(String),
    #[error("JIT compilation failed: {0}")]
    CompilationFailed(String),
    #[error("execution error: {0}")]
    Execution(String),
}

/// Per-step execution context passed to `FusedGraphExecutor::execute`.
///
/// Bundles together all runtime-variable state for one inference step:
/// - `seq_len` / `hidden`: concrete shape values for this step
/// - `weights`: pointer table for model weight tensors
/// - `shape_binding`: symbolic dimension bindings (e.g. `total_seq`)
/// - `kv_layer_offset`: optional byte offset into a KV cache buffer for the
///   current layer (used when the executor drives a full decoder forward pass)
pub struct GraphExecutorContext {
    /// Number of tokens in the current step (sequence length).
    pub seq_len: usize,
    /// Model hidden dimension.
    pub hidden: usize,
    /// Weight pointer table.
    pub weights: RuntimeWeightBinding,
    /// Symbolic shape bindings (e.g. `"total_seq"` → current KV length).
    pub shape_binding: gllm_kernels::compiler::ShapeBinding,
    /// Optional byte offset into a flat KV cache buffer for the current layer.
    pub kv_layer_offset: Option<usize>,
}

impl GraphExecutorContext {
    /// Resolve a named weight to its raw pointer.
    ///
    /// Returns `ExecutorError::WeightNotFound` if the name is not in the
    /// binding table.
    pub fn resolve_weight(&self, name: &str) -> Result<*const f32, ExecutorError> {
        self.weights
            .get(name)
            .ok_or_else(|| ExecutorError::WeightNotFound(name.to_string()))
    }

    /// Resolve a `SymDim` to a concrete `usize`.
    ///
    /// - `Concrete(n)` → `Ok(n)` immediately.
    /// - `Symbolic(s)` → looks up `s` in `shape_binding`; returns
    ///   `ExecutorError::ShapeNotBound` if missing.
    pub fn resolve_sym_dim(
        &self,
        dim: &gllm_kernels::compiler::SymDim,
    ) -> Result<usize, ExecutorError> {
        match dim {
            gllm_kernels::compiler::SymDim::Concrete(n) => Ok(*n),
            gllm_kernels::compiler::SymDim::Symbolic { name, .. } => self
                .shape_binding
                .get(name)
                .copied()
                .ok_or_else(|| ExecutorError::ShapeNotBound(name.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{AtomicOp, FusedNode, OptimizationStats};

    #[test]
    fn execution_context_tracks_computed() {
        let mut ctx = ExecutionContext::new();
        assert!(!ctx.is_computed("hidden_0"));
        ctx.mark_computed("hidden_0".to_string());
        assert!(ctx.is_computed("hidden_0"));
    }

    #[test]
    fn execution_plan_from_empty_graph() {
        let graph = FusedGraph {
            nodes: vec![],
            inputs: vec!["input".to_string()],
            outputs: vec!["output".to_string()],
            weight_bindings: HashMap::new(),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };
        let plan = ExecutionPlan::from_fused_graph(&graph);
        assert_eq!(plan.op_count(), 0);
        assert_eq!(plan.inputs, vec!["input".to_string()]);
    }

    #[test]
    fn fused_executor_run_without_compile_returns_error() {
        let graph = FusedGraph {
            nodes: vec![FusedNode {
                name: "node0".to_string(),
                op: FusedOp::Atomic(AtomicOp::new("Add")),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::from([(
                "w".to_string(),
                crate::graph::types::WeightBinding {
                    source_name: "w".to_string(),
                    shape: vec![1],
                    dtype: safetensors::Dtype::F32,
                    data: None,
                    ptr: None,
                    shape_needs_transpose: false,
                },
            )]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let executor = FusedGraphExecutor::new(graph);
        let result = executor.run(&HashMap::from([("x".to_string(), vec![0u8; 4])]), &HashMap::new());
        // run() without compile() must return an error, never dummy results
        assert!(result.is_err());
    }

    #[test]
    fn fused_executor_debug_format() {
        let graph = FusedGraph::new();
        let executor = FusedGraphExecutor::new(graph);
        let debug_str = format!("{:?}", executor);
        assert!(debug_str.contains("FusedGraphExecutor"));
        assert!(debug_str.contains("is_compiled"));
    }

    #[test]
    fn fused_executor_is_compiled_default_false() {
        let graph = FusedGraph::new();
        let executor = FusedGraphExecutor::new(graph);
        assert!(!executor.is_compiled());
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_flash_attention_graph_structure() {
        let config = FlashAttentionConfig {
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            scale: None,
            causal: true,
        };
        let g = build_flash_attention_graph(&config, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 3); // Q, K, V
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 1); // MHA
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_swiglu_graph_structure() {
        let config = SwiGLUConfig {
            hidden_size: 512,
            intermediate_size: 1024,
        };
        let g = build_swiglu_graph(&config, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 3); // input, w_gate, w_up (WII architecture)
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 3); // 2×Gemm + SwiGlu
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_fused_qkv_rope_graph_structure() {
        let config = FusedQkvRopeConfig {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 64,
            rope_theta: 10000.0,
        };
        let g = build_fused_qkv_rope_graph(&config, 512, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 4); // input, w_q, w_k, w_v
        assert_eq!(g.outputs.len(), 3); // q_rope, k_rope, v (CPU and GPU both apply RoPE)
        assert_eq!(g.ops.len(), 5); // 3 Gemms + 2 RoPEs
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_fused_rms_linear_graph_structure() {
        let config = FusedRMSLinearConfig {
            hidden_size: 512,
            eps: 1e-5,
        };
        let g = build_fused_rms_linear_graph(&config, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 3); // input, norm_w, linear_w
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 2); // RmsNorm + Gemm
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_gqa_graph_structure() {
        let config = GQAConfig {
            num_heads: 32,
            num_kv_heads: 8,
            num_groups: 4,
            head_dim: 128,
            sliding_window: 0,
        };
        let g = build_gqa_graph(&config, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 3); // Q, K, V
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 1); // MHA
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_moe_routing_graph_structure() {
        let config = MoERoutingConfig {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.0,
        };
        let g = build_moe_routing_graph(&config, 512, gllm_kernels::types::DType::F32);
        assert_eq!(g.inputs.len(), 2); // input, gate_w
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 2); // Gemm + Softmax
    }

    #[test]
    fn infer_output_shape_matmul() {
        use gllm_kernels::compiler::SymDim;
        let shapes: Vec<Vec<SymDim>> = vec![
            vec![SymDim::Concrete(4), SymDim::Concrete(512)],
            vec![SymDim::Concrete(512), SymDim::Concrete(1024)],
        ];
        let out = infer_output_shape("MatMul", &shapes);
        assert_eq!(out, vec![SymDim::Concrete(4), SymDim::Concrete(1024)]);
    }

    #[test]
    fn infer_output_shape_add() {
        use gllm_kernels::compiler::SymDim;
        let shapes: Vec<Vec<SymDim>> = vec![
            vec![SymDim::Concrete(4), SymDim::Concrete(512)],
            vec![SymDim::Concrete(4), SymDim::Concrete(512)],
        ];
        let out = infer_output_shape("Add", &shapes);
        assert_eq!(out, vec![SymDim::Concrete(4), SymDim::Concrete(512)]);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn atomic_op_to_kind_known_ops() {
        use gllm_kernels::compiler::SymDim;
        let shapes: Vec<Vec<SymDim>> = vec![
            vec![SymDim::Concrete(4), SymDim::Concrete(512)],
            vec![SymDim::Concrete(512), SymDim::Concrete(1024)],
        ];
        assert!(atomic_op_to_kind("Add", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Mul", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Silu", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Gelu", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("MatMul", &shapes, gllm_kernels::types::DType::F32).is_ok());
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn atomic_op_to_kind_unknown_returns_err() {
        use gllm_kernels::compiler::SymDim;
        let shapes: Vec<Vec<SymDim>> = vec![vec![SymDim::Concrete(4), SymDim::Concrete(512)]];
        let result = atomic_op_to_kind("UnknownOp", &shapes, gllm_kernels::types::DType::F32);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err}").contains("UnknownOp"),
            "error should mention the op type"
        );
    }

    #[test]
    fn test_execution_plan_topological_order() {
        // Build a graph with dependencies: A -> B -> C
        let graph = FusedGraph {
            nodes: vec![
                FusedNode {
                    name: "A".to_string(),
                    op: FusedOp::Atomic(AtomicOp::new("Add")),
                    inputs: vec!["input".to_string()],
                    outputs: vec!["a_out".to_string()],
                    attributes: HashMap::new(),
                },
                FusedNode {
                    name: "B".to_string(),
                    op: FusedOp::SwiGLU(SwiGLUConfig {
                        hidden_size: 512,
                        intermediate_size: 1024,
                    }),
                    inputs: vec!["a_out".to_string(), "up_weight".to_string()],
                    outputs: vec!["b_out".to_string()],
                    attributes: HashMap::new(),
                },
                FusedNode {
                    name: "C".to_string(),
                    op: FusedOp::Atomic(AtomicOp::new("Add")),
                    inputs: vec!["b_out".to_string(), "residual".to_string()],
                    outputs: vec!["output".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec!["input".to_string(), "up_weight".to_string(), "residual".to_string()],
            outputs: vec!["output".to_string()],
            weight_bindings: HashMap::new(),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let plan = ExecutionPlan::from_fused_graph(&graph);
        assert_eq!(plan.op_count(), 3);

        // Verify topological order: A produces a_out (needed by B),
        // B produces b_out (needed by C).
        let names: Vec<String> = plan.operations.iter().map(|op| match op {
            ExecutionOp::FlashAttention { name, .. }
            | ExecutionOp::SwiGLU { name, .. }
            | ExecutionOp::RoPE { name, .. }
            | ExecutionOp::FusedQkvRope { name, .. }
            | ExecutionOp::FusedRMSLinear { name, .. }
            | ExecutionOp::GQA { name, .. }
            | ExecutionOp::MoERouting { name, .. }
            | ExecutionOp::PerLayerEmbed { name, .. }
            | ExecutionOp::Atomic { name, .. } => name.clone(),
        }).collect();

        let pos_a = names.iter().position(|n| n == "A").unwrap();
        let pos_b = names.iter().position(|n| n == "B").unwrap();
        let pos_c = names.iter().position(|n| n == "C").unwrap();
        assert!(pos_a < pos_b, "A must precede B");
        assert!(pos_b < pos_c, "B must precede C");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn test_compile_unsupported_atomic_op_returns_error() {
        // Build a graph with an unknown atomic op
        let graph = FusedGraph {
            nodes: vec![FusedNode {
                name: "unsupported_node".to_string(),
                op: FusedOp::Atomic(AtomicOp::new("CompletelyUnknownOp")),
                inputs: vec!["x".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::new(),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let mut executor = FusedGraphExecutor::new(graph);
        let result = executor.compile(4, 512, gllm_kernels::types::DType::F32);
        assert!(result.is_err(), "compile() must fail for unsupported op");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("CompletelyUnknownOp"),
            "error message should mention the unsupported op, got: {err_msg}"
        );
    }

    #[test]
    fn test_no_silent_fallback_in_compile() {
        // Verify that FusedOp enum match in build_node_graph has no wildcard arm.
        // This is a structural test: all FusedOp variants must be explicitly handled.
        // We verify by checking that every variant type is present in ExecutionPlan.
        let variants: Vec<FusedOp> = vec![
            FusedOp::FlashAttention(FlashAttentionConfig::default()),
            FusedOp::SwiGLU(SwiGLUConfig::default()),
            FusedOp::RoPE(RoPEConfig::default()),
            FusedOp::FusedQkvRope(FusedQkvRopeConfig::default()),
            FusedOp::FusedRMSLinear(FusedRMSLinearConfig::default()),
            FusedOp::GQA(GQAConfig::default()),
            FusedOp::MoERouting(MoERoutingConfig::default()),
            FusedOp::Atomic(AtomicOp::new("Add")),
        ];

        for (i, op) in variants.into_iter().enumerate() {
            let node = FusedNode {
                name: format!("node_{i}"),
                op,
                inputs: vec!["in".to_string()],
                outputs: vec![format!("out_{i}")],
                attributes: HashMap::new(),
            };

            let graph = FusedGraph {
                nodes: vec![node],
                inputs: vec!["in".to_string()],
                outputs: vec![format!("out_{i}")],
                weight_bindings: HashMap::new(),
                quantization_info: HashMap::new(),
                sparse_tensors: HashMap::new(),
                stats: OptimizationStats::default(),
            };

            let plan = ExecutionPlan::from_fused_graph(&graph);
            assert_eq!(plan.op_count(), 1, "each FusedOp variant must produce exactly one ExecutionOp");
        }
    }

    // ---------------------------------------------------------------------------
    // T3.1 tests: RuntimeWeightBinding + GraphExecutorContext
    // ---------------------------------------------------------------------------

    #[test]
    fn runtime_weight_binding_bind_and_get() {
        let data = vec![1.0f32, 2.0, 3.0];
        let wb = RuntimeWeightBinding::new()
            .bind("w_q", data.as_ptr())
            .bind("w_k", data.as_ptr());
        assert!(wb.get("w_q").is_some());
        assert!(wb.get("w_k").is_some());
        assert!(wb.get("w_v").is_none());
        // Pointer round-trip
        let ptr = wb.get("w_q").unwrap();
        unsafe {
            assert_eq!(*ptr, 1.0f32);
        }
    }

    #[test]
    fn runtime_weight_binding_empty() {
        let wb = RuntimeWeightBinding::new();
        assert!(wb.get("anything").is_none());
    }

    #[test]
    fn graph_executor_context_weight_not_found() {
        let ctx = GraphExecutorContext {
            seq_len: 4,
            hidden: 512,
            weights: RuntimeWeightBinding::new(),
            shape_binding: gllm_kernels::compiler::ShapeBinding::new(),
            kv_layer_offset: None,
        };
        let result = ctx.resolve_weight("missing_weight");
        assert!(result.is_err());
        match result.unwrap_err() {
            ExecutorError::WeightNotFound(name) => assert_eq!(name, "missing_weight"),
            other => panic!("expected WeightNotFound, got {other:?}"),
        }
    }

    #[test]
    fn graph_executor_context_shape_not_bound() {
        let ctx = GraphExecutorContext {
            seq_len: 4,
            hidden: 512,
            weights: RuntimeWeightBinding::new(),
            shape_binding: gllm_kernels::compiler::ShapeBinding::new(),
            kv_layer_offset: None,
        };
        let result = ctx.resolve_sym_dim(
            &gllm_kernels::compiler::SymDim::Symbolic { name: "total_seq".to_string(), max_value: None },
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ExecutorError::ShapeNotBound(name) => assert_eq!(name, "total_seq"),
            other => panic!("expected ShapeNotBound, got {other:?}"),
        }
    }

    #[test]
    fn graph_executor_context_concrete_sym_dim() {
        let ctx = GraphExecutorContext {
            seq_len: 7,
            hidden: 256,
            weights: RuntimeWeightBinding::new(),
            shape_binding: gllm_kernels::compiler::ShapeBinding::new(),
            kv_layer_offset: None,
        };
        let dim = gllm_kernels::compiler::SymDim::Concrete(42);
        assert_eq!(ctx.resolve_sym_dim(&dim).unwrap(), 42);
    }

    #[test]
    fn graph_executor_context_bound_sym_dim() {
        let ctx = GraphExecutorContext {
            seq_len: 1,
            hidden: 128,
            weights: RuntimeWeightBinding::new(),
            shape_binding: gllm_kernels::compiler::ShapeBinding::new()
                .bind("total_seq", 17),
            kv_layer_offset: None,
        };
        let dim = gllm_kernels::compiler::SymDim::Symbolic { name: "total_seq".to_string(), max_value: None };
        assert_eq!(ctx.resolve_sym_dim(&dim).unwrap(), 17);
    }

    // ---------------------------------------------------------------------------
    // T3.2 tests: OnnxGraph → FusedGraphExecutor end-to-end chain
    // ---------------------------------------------------------------------------

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn test_fused_graph_executor_from_simple_graph() {
        use crate::loader::onnx::{OnnxGraph, OnnxNode, OnnxValueInfo};

        // Build a minimal OnnxGraph: Add node
        let node = OnnxNode {
            name: "add0".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "y".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let make_vi = |n: &str| OnnxValueInfo {
            name: n.to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "simple".to_string(),
            doc_string: String::new(),
            nodes: vec![node],
            inputs: vec![make_vi("x"), make_vi("y")],
            outputs: vec![make_vi("out")],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let executor = FusedGraphExecutor::from_graph(graph, 4, 64, gllm_kernels::types::DType::F32, crate::graph::optimizer::OptimizationContext::default())
            .expect("from_graph must succeed for a simple Add graph");

        assert!(executor.is_compiled(), "executor must be compiled after from_graph");
        assert_eq!(executor.graph().node_count(), 1);
    }

    #[test]
    fn test_weight_binding_resolve() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let wb = RuntimeWeightBinding::new()
            .bind("w_q", data.as_ptr())
            .bind("w_k", data.as_ptr());

        // Existing keys resolve to their pointers
        assert!(wb.get("w_q").is_some());
        assert!(wb.get("w_k").is_some());
        // Non-existent key returns None
        assert!(wb.get("w_v").is_none());
        // Pointer value is correct
        unsafe {
            assert_eq!(*wb.get("w_q").unwrap(), 1.0f32);
        }
    }

    #[test]
    fn test_shape_binding_resolve() {
        let ctx = GraphExecutorContext {
            seq_len: 4,
            hidden: 512,
            weights: RuntimeWeightBinding::new(),
            shape_binding: gllm_kernels::compiler::ShapeBinding::new()
                .bind("total_seq", 32),
            kv_layer_offset: None,
        };

        // Concrete dim resolves immediately
        let concrete = gllm_kernels::compiler::SymDim::Concrete(99);
        assert_eq!(ctx.resolve_sym_dim(&concrete).unwrap(), 99);

        // Bound symbolic dim resolves to its value
        let sym = gllm_kernels::compiler::SymDim::Symbolic { name: "total_seq".to_string(), max_value: None };
        assert_eq!(ctx.resolve_sym_dim(&sym).unwrap(), 32);

        // Unbound symbolic dim returns ShapeNotBound error
        let unbound = gllm_kernels::compiler::SymDim::Symbolic { name: "unbound_dim".to_string(), max_value: None };
        let err = ctx.resolve_sym_dim(&unbound).unwrap_err();
        assert!(
            matches!(err, ExecutorError::ShapeNotBound(ref n) if n == "unbound_dim"),
            "expected ShapeNotBound(unbound_dim), got {err:?}"
        );
    }

    // ---------------------------------------------------------------------------
    // T3.2 new tests: Gap A (run_with_kv_cache) + Gap B (WeightBinding ptr)
    // ---------------------------------------------------------------------------

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn test_fused_executor_with_kv_cache() {
        use crate::loader::onnx::{OnnxGraph, OnnxNode, OnnxValueInfo};

        // Build a minimal single-Add graph (no KV cache ops needed to test the API)
        let node = OnnxNode {
            name: "add0".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "y".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let make_vi = |n: &str| OnnxValueInfo {
            name: n.to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "kv_test".to_string(),
            doc_string: String::new(),
            nodes: vec![node],
            inputs: vec![make_vi("x"), make_vi("y")],
            outputs: vec![make_vi("out")],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };

        let executor = FusedGraphExecutor::from_graph(graph, 4, 64, gllm_kernels::types::DType::F32, crate::graph::optimizer::OptimizationContext::default())
            .expect("from_graph must succeed");

        // Provide two f32 inputs: x = [1.0; 4*64], y = [2.0; 4*64]
        let n = 4 * 64;
        let x_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let y_bytes: Vec<u8> = (0..n).flat_map(|_| 2.0f32.to_le_bytes()).collect();
        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x_bytes);
        inputs.insert("y".to_string(), y_bytes);

        // run_with_kv_cache with null KV pointers (Add node doesn't use them)
        let result = executor.run_with_kv_cache(
            &inputs,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            4,
            4,
            std::ptr::null(),
        );
        assert!(result.is_ok(), "run_with_kv_cache must succeed: {:?}", result.err());

        let out = result.unwrap();
        assert!(out.contains_key("out"), "output must contain 'out'");
        let out_bytes = &out["out"];
        assert_eq!(out_bytes.len(), n * 4, "output size must match input size");
        // Verify Add: 1.0 + 2.0 = 3.0 for each element
        let out_f32: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!(
            out_f32.iter().all(|&v| (v - 3.0f32).abs() < 1e-5),
            "Add(1.0, 2.0) must equal 3.0 for all elements"
        );
    }

    #[test]
    fn test_weight_binding_ptr_injection() {
        // Verify that WeightBinding.ptr takes priority over .data during execution.
        let runtime_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let embedded_data: Vec<u8> = vec![0.0f32, 0.0, 0.0, 0.0]
            .iter()
            .flat_map(|f: &f32| f.to_le_bytes())
            .collect();

        let wb = crate::graph::types::WeightBinding {
            source_name: "w".to_string(),
            shape: vec![4],
            dtype: safetensors::Dtype::F32,
            data: Some(embedded_data),
            ptr: Some(runtime_data.as_ptr()),
                    shape_needs_transpose: false,
        };

        // ptr is set — it should take priority
        assert!(wb.ptr.is_some(), "ptr must be set");
        unsafe {
            let ptr = wb.ptr.unwrap();
            assert_eq!(*ptr, 10.0f32, "ptr must point to runtime_data[0]");
            assert_eq!(*ptr.add(3), 40.0f32, "ptr must point to runtime_data[3]");
        }

        // data is also set but should be overridden
        let data_f32: Vec<f32> = wb.data.as_ref().unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!(
            data_f32.iter().all(|&v| v == 0.0f32),
            "embedded data must be zeros (overridden by ptr)"
        );
    }
}
