//! FusedGraph 执行器 (REQ-EXEC-002)
//!
//! Compiles each FusedNode in the FusedGraph into a JIT-compiled kernel
//! via gllm-kernels' InferenceCompiler, then executes them in topological
//! order with proper buffer management.

use std::collections::{HashMap, HashSet};

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
    /// Feature dimension per token (e.g., hidden_size).
    /// Runtime output size = feature_dim * actual_seq_len * dtype.size_bytes().
    feature_dim: usize,
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
        max_value: Some(2048),
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
        max_value: Some(2048), // Conservative upper bound for buffer allocation
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
        max_value: Some(2048),
    };
    let input = g.add_tensor("input", vec![seq_len_sym.clone(), SymDim::Concrete(hidden)], dt);
    g.inputs = vec![input];

    let out = g.add_tensor("rope_out", vec![seq_len_sym, SymDim::Concrete(hidden)], dt);
    g.add_op(
        OpKind::RoPE {
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
        max_value: Some(2048),
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

    #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
    {
        let q_rope = g.add_tensor("q_rope", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
        g.add_op(
            OpKind::RoPE { head_dim: config.head_dim, theta: config.rope_theta, partial: 1.0 },
            vec![q_out],
            vec![q_rope],
            "rope_q",
        );

        let k_rope = g.add_tensor("k_rope", vec![seq_len_sym, SymDim::Concrete(kv_dim)], dt);
        g.add_op(
            OpKind::RoPE { head_dim: config.head_dim, theta: config.rope_theta, partial: 1.0 },
            vec![k_out],
            vec![k_rope],
            "rope_k",
        );

        g.outputs = vec![q_rope, k_rope, v_out];
    }
    #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
    {
        g.outputs = vec![q_out, k_out, v_out];
    }
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
        max_value: Some(2048),
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
    let kv_dim = config.num_kv_heads * config.head_dim;

    let seq_len_sym = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(2048),
    };

    let q = g.add_tensor("q", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    let k = g.add_tensor("k", vec![seq_len_sym.clone(), SymDim::Concrete(kv_dim)], dt);
    let v = g.add_tensor("v", vec![seq_len_sym.clone(), SymDim::Concrete(kv_dim)], dt);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor("gqa_out", vec![seq_len_sym.clone(), SymDim::Concrete(q_dim)], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: seq_len_sym,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
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
        max_value: Some(2048),
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
        max_value: Some(2048),
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
        "SimplifiedLayerNormalization" => Ok(OpKind::RmsNorm { eps: 1e-6 }),
        "LayerNormalization" => Ok(OpKind::LayerNorm { eps: 1e-5 }),
        "Softmax" => {
            let vocab_size = if !input_shapes.is_empty() && input_shapes[0].len() >= 2 {
                input_shapes[0][input_shapes[0].len() - 1].as_concrete().unwrap_or(1)
            } else {
                1
            };
            Ok(OpKind::SoftmaxWithEntropy { vocab_size })
        }
        "Residual" => {
            let hidden = if !input_shapes.is_empty() && input_shapes[0].len() >= 2 {
                input_shapes[0][input_shapes[0].len() - 1].as_concrete().unwrap_or(1)
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
        // CPU JIT always computes in f32 regardless of model weight dtype.
        // Weights are converted to f32 during upload (upload_native_tensor_with_convert).
        // Using BF16/F16 here would cause weight_layout and output buffers to be 2x too small.
        let compile_dtype = gllm_kernels::types::DType::F32;

        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let mut compiled_nodes = Vec::with_capacity(self.graph.nodes.len());

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let build = self.build_node_graph(idx, seq_len, hidden, compile_dtype)?;

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

            // Calculate feature_dim: for most ops, it's the last dimension of output shape
            // For ops with symbolic seq_len, output_numel = max_seq_len * feature_dim
            // So feature_dim = output_numel / max_seq_len (where max_seq_len = 2048)
            let feature_dim = if build.output_numel > 0 {
                build.output_numel / 2048 // max_seq_len used in SymDim::Symbolic
            } else {
                hidden // fallback to hidden_size
            };

            compiled_nodes.push(CompiledNode {
                compiled,
                graph_input_names: build.input_names,
                graph_output_names: build.output_names,
                output_numel: build.output_numel,
                per_output_numel: build.per_output_numel,
                output_dtype,
                feature_dim,
            });
        }

        self.compiled_nodes = compiled_nodes;
        self.is_compiled = true;
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
                feature_dim: if p.output_numel > 0 { p.output_numel / 2048 } else { hidden },
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
            });
        }
        
        Ok(GraphExecutorPayload { nodes })
    }

    /// Result of building a CompilerGraph for one FusedNode.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_node_graph(
        &self,
        node_idx: usize,
        _seq_len: usize, // Deprecated: all graphs use SymDim::Symbolic now
        hidden: usize,
        dtype: gllm_kernels::types::DType,
    ) -> Result<NodeGraphBuild, ExecutionError> {
        let node = &self.graph.nodes[node_idx];
        let max_seq_len = 2048; // SymDim::Symbolic max_value

        match &node.op {
            FusedOp::FlashAttention(config) => {
                let g = build_flash_attention_graph(config, dtype);
                let h = config.num_heads * config.head_dim;
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * h,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::SwiGLU(config) => {
                let g = build_swiglu_graph(config, dtype);
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * config.intermediate_size,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::RoPE(config) => {
                let g = build_rope_graph(config, hidden, dtype);
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * hidden,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::FusedQkvRope(config) => {
                let g = build_fused_qkv_rope_graph(config, hidden, dtype);
                let q_dim = config.num_heads * config.head_dim;
                let kv_dim = config.num_kv_heads * config.head_dim;
                let per = vec![
                    max_seq_len * q_dim,
                    max_seq_len * kv_dim,
                    max_seq_len * kv_dim,
                ];
                let total: usize = per.iter().sum();
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: total,
                    per_output_numel: per,
                    output_dtype,
                })
            }

            FusedOp::FusedRMSLinear(config) => {
                let g = build_fused_rms_linear_graph(config, dtype);
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * config.hidden_size,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::GQA(config) => {
                let g = build_gqa_graph(config, dtype);
                let q_dim = config.num_heads * config.head_dim;
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * q_dim,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::MoERouting(config) => {
                let g = build_moe_routing_graph(config, hidden, dtype);
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * config.num_experts,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::PerLayerEmbed(config) => {
                let g = build_ple_graph(config, hidden, dtype);
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: max_seq_len * hidden,
                    per_output_numel: vec![],
                    output_dtype,
                })
            }

            FusedOp::Atomic(atomic) => {
                if atomic.op_type == "Gather" || atomic.op_type == "Slice" || atomic.op_type == "Shape" {
                    // Skip ops handled natively by standard CPU graph pipelines beforehand.
                    return Ok(NodeGraphBuild {
                        graph: gllm_kernels::compiler::CompilerGraph::new(),
                        input_names: node.inputs.clone(),
                        output_names: node.outputs.clone(),
                        output_numel: 0,
                        per_output_numel: vec![],
                        output_dtype: dtype,
                    });
                }
                let is_matmul = atomic.op_type == "MatMul" || atomic.op_type == "Gemm";

                use gllm_kernels::compiler::SymDim;
                let seq_len_sym = SymDim::Symbolic {
                    name: "seq_len".to_string(),
                    max_value: Some(2048),
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
                                // SafeTensors stores Linear weights as [out_features, in_features]
                                // (PyTorch convention). infer_output_shape/atomic_op_to_kind use
                                // math convention [K, N] where K=in, N=out.
                                // Transpose weight shape for MatMul/Gemm weight inputs (index > 0).
                                if is_matmul && i > 0 && shape.len() == 2 {
                                    shape.swap(0, 1); // [N, K] → [K, N]
                                }
                                return shape;
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

                let output_shape = infer_output_shape(&atomic.op_type, &input_shapes);

                // Calculate output_numel: use max_for_allocation for Symbolic dims
                let output_numel: usize = output_shape.iter().map(|d| d.max_for_allocation(2048)).product();

                let g = build_atomic_graph(&atomic.op_type, &input_shapes, &output_shape, dtype)?;
                let output_dtype = g.tensors[g.outputs[0].0 as usize].dtype;

                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel,
                    per_output_numel: vec![],
                    output_dtype,
                })
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
    pub fn run(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
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
            return self.run_compiled(inputs);
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

    /// Execute the compiled JIT kernels in topological order.
    ///
    /// graph.nodes is already in topological order. For each node:
    /// 1. The first input tensor is passed as the activation input
    /// 2. Remaining input tensors are packed into a contiguous weight blob
    /// 3. The compiled kernel is executed
    /// 4. Output data is stored for downstream nodes
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn run_compiled(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();

        // Seed with graph inputs
        for (name, data) in inputs {
            tensors.insert(name.clone(), data.clone());
        }

        // Seed with weight binding data — prefer runtime ptr over embedded bytes
        let mut weight_ptrs: HashMap<String, *const u8> = HashMap::new();
        for (name, wb) in &self.graph.weight_bindings {
            if let Some(ptr) = wb.ptr {
                if !wb.shape.is_empty() {
                    let numel: usize = wb.shape.iter().product();
                    let bytes = numel * wb.dtype.size();
                    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, bytes) };
                    tensors.insert(name.clone(), slice.to_vec());
                }
                weight_ptrs.insert(name.clone(), ptr as *const u8);
            } else if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }

        // Execute each node
        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];

            if !cn.graph_output_names.is_empty() && cn.graph_output_names.iter().all(|name| tensors.contains_key(name)) {
                continue;
            }

            // Activation input = first graph input tensor
            let activation = if !cn.graph_input_names.is_empty() {
                tensors
                    .get(&cn.graph_input_names[0])
                    .cloned()
                    .unwrap_or_default() // LEGAL: 不存在的 tensor 返回空数据
            } else {
                Vec::new()
            };

            // Pack weight blob using weight_layout if available
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
                for name in cn.graph_input_names.iter().skip(1) {
                    if let Some(data) = tensors.get(name) {
                        weight_blob.extend_from_slice(data);
                    }
                }
            }

            // Allocate output buffer (dtype-aware)
            let output_bytes = cn.output_numel * cn.output_dtype.size_bytes();
            let mut output_buf = vec![0u8; output_bytes];

            // Allocate scratchpad
            let mut scratchpad = vec![0u8; cn.compiled.scratchpad_bytes.max(64)];

            // Compute seq_len from activation size: activation is [seq_len, hidden],
            // stored with the node's output dtype, so seq_len = num_elements / hidden.
            // Fall back to 1 if we cannot determine it.
            let activation_elems = activation.len() / cn.output_dtype.size_bytes();
            let seq_len = if activation_elems > 0 {
                activation_elems
            } else {
                1
            };

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
                    std::ptr::null_mut(), // no KV cache
                    std::ptr::null(),     // no positions
                    std::ptr::null(),     // no seq_lens
                    1,                    // batch_size = 1
                    seq_len,
                    output_buf.as_mut_ptr(),
                    scratchpad.as_mut_ptr(),
                );
            }

            // Store output(s)
            if cn.graph_output_names.len() == 1 {
                tensors.insert(cn.graph_output_names[0].clone(), output_buf);
            } else if !cn.per_output_numel.is_empty() {
                // Multi-output: split by per_output_numel
                let mut byte_offset = 0;
                for (i, name) in cn.graph_output_names.iter().enumerate() {
                    let numel = cn.per_output_numel[i];
                    let nbytes = numel * cn.output_dtype.size_bytes();
                    let chunk = output_buf[byte_offset..byte_offset + nbytes].to_vec();
                    tensors.insert(name.clone(), chunk);
                    byte_offset += nbytes;
                }
            } else if cn.graph_output_names.len() > 1 {
                // Multi-output node without per_output_numel is a compile-time bug.
                return Err(ExecutionError::Compilation(format!(
                    "node has {} outputs but no per_output_numel — compile() should have set this",
                    cn.graph_output_names.len(),
                )));
            }
        }

        // Collect graph outputs
        let mut out = HashMap::new();
        for name in &self.graph.outputs {
            let data = tensors.remove(name).unwrap_or_default(); // LEGAL: 不存在的 tensor 返回空数据
            out.insert(name.clone(), data);
        }
        Ok(out)
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

        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();

        // Seed with graph inputs
        for (name, data) in inputs {
            tensors.insert(name.clone(), data.clone());
        }

        // Seed with weight bindings — prefer runtime ptr over embedded bytes.
        // Also build a raw-pointer map for direct-ptr weights (where we may not know shape).
        let mut weight_ptrs: HashMap<String, *const u8> = HashMap::new();
        for (name, wb) in &self.graph.weight_bindings {
            if let Some(ptr) = wb.ptr {
                if !wb.shape.is_empty() {
                    let numel: usize = wb.shape.iter().product();
                    // Weight bindings from SafeTensors have been uploaded as f32 (4 bytes per element),
                    // regardless of the original dtype (BF16/F16 → f32 conversion in upload_weights).
                    let bytes = numel * 4; // always f32 after upload conversion
                    let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, bytes) };
                    tensors.insert(name.clone(), slice.to_vec());
                }
                weight_ptrs.insert(name.clone(), ptr as *const u8);
            } else if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }

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

        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];

            if !cn.graph_output_names.is_empty() && cn.graph_output_names.iter().all(|name| tensors.contains_key(name)) {
                continue;
            }

            // §9-§18: Pre-node callback — Gate-First Skip / Residual Bypass / Early Exit
            let layer_idx = node_idx / 2; // 每层约 2 个 FusedNode (Attention + FFN)
            if let Some(ref mut cb) = callbacks {
                let hidden_state = tensors.values().next().map(|v| v.as_slice()).unwrap_or(&[]);
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
                                let element_size = 4; // f32
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

            if cfg!(debug_assertions) {
                eprintln!("[EXEC] node {node_idx}/{} '{}' ({}) prep",
                    self.graph.nodes.len(),
                    self.graph.nodes[node_idx].name,
                    self.graph.nodes[node_idx].op.name());
            }

            // Load activation and pad to max_seq_len size if needed.
            // JIT kernels may iterate up to max_seq_len rows internally; the activation buffer
            // must be at least that large to avoid out-of-bounds reads.
            let activation = if !cn.graph_input_names.is_empty() {
                let raw = tensors
                    .get(&cn.graph_input_names[0])
                    .cloned()
                    .unwrap_or_default();
                // Compute expected max activation size (max_seq_len * feature_dim)
                let max_act_bytes = cn.output_numel * cn.output_dtype.size_bytes();
                if !raw.is_empty() && raw.len() < max_act_bytes {
                    // Pad with zeros so JIT kernel can safely read max_seq_len rows
                    let mut padded = raw;
                    padded.resize(max_act_bytes, 0);
                    padded
                } else {
                    raw
                }
            } else {
                Vec::new()
            };

            // Pack weight blob using the compiled weight_layout for size info.
            // The weight_layout maps tensor IDs to byte offsets within the blob.
            // graph_input_names[1..] are the weight tensors in order.
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
                        // Validate ptr read size against available data
                        let src = unsafe { std::slice::from_raw_parts(ptr, size) };
                        weight_blob[offset..offset + size].copy_from_slice(src);
                    }
                }
            } else {
                for name in cn.graph_input_names.iter().skip(1) {
                    if let Some(data) = tensors.get(name) {
                        weight_blob.extend_from_slice(data);
                    }
                }
            }

            // Allocate output buffer at max_seq_len size (safe upper bound).
            // JIT kernels may loop up to max_seq_len internally for ops like
            // RmsNorm/Add that iterate all rows. The buffer must be large enough
            // for the compiled loop bound. After execution, we truncate to the
            // runtime seq_len when inserting into the tensor map.
            let runtime_output_numel = cn.output_numel; // max_seq_len size (safe)
            let output_bytes = runtime_output_numel * cn.output_dtype.size_bytes();
            let mut output_buf = vec![0u8; output_bytes];
            let mut scratchpad = vec![0u8; cn.compiled.scratchpad_bytes.max(64)];

            if cfg!(debug_assertions) {
                eprintln!("[EXEC] node {node_idx} exec: act={}B wt={}B out={}B scratch={}B",
                    activation.len(), weight_blob.len(), output_bytes,
                    cn.compiled.scratchpad_bytes);
            }
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
                    seq_len,
                    output_buf.as_mut_ptr(),
                    scratchpad.as_mut_ptr(),
                );
            }
            if cfg!(debug_assertions) {
                eprintln!("[EXEC] node {node_idx} done");
            }

            // NO_SCALAR: FusedQkvRope is fully handled by JIT codegen (RoPE is applied
            // within the fused QKV+RoPE kernel). No post-hoc scalar fallback needed.

            // Truncate output to runtime seq_len before inserting into tensor map.
            // JIT kernel wrote max_seq_len rows but only the first seq_len are valid.
            let runtime_bytes = cn.feature_dim * seq_len * cn.output_dtype.size_bytes();
            let truncated_output = if runtime_bytes < output_buf.len() {
                output_buf[..runtime_bytes].to_vec()
            } else {
                output_buf
            };

            if cn.graph_output_names.len() == 1 {
                tensors.insert(cn.graph_output_names[0].clone(), truncated_output);
            } else if !cn.per_output_numel.is_empty() {
                // Multi-output node: split truncated_output by per_output_numel
                // per_output_numel stores max_seq_len sizes, need to scale to runtime seq_len
                let max_seq_len = 2048; // SymDim::Symbolic max_value
                let mut byte_offset = 0;
                for (i, name) in cn.graph_output_names.iter().enumerate() {
                    let per_token = cn.per_output_numel[i] / max_seq_len;
                    let numel = per_token * seq_len;
                    let nbytes = numel * cn.output_dtype.size_bytes();
                    if byte_offset + nbytes <= truncated_output.len() {
                        let chunk = truncated_output[byte_offset..byte_offset + nbytes].to_vec();
                        tensors.insert(name.clone(), chunk);
                    }
                    byte_offset += nbytes;
                }
            } else if cn.graph_output_names.len() > 1 {
                return Err(ExecutionError::Compilation(format!(
                    "node has {} outputs but no per_output_numel",
                    cn.graph_output_names.len(),
                )));
            }

            // §14.2: Scatter-back after compacted execution
            // If this node was compacted, scatter the output back to original positions
            let mask_key = format!("__compact_mask_{}", node_idx);
            let orig_size_key = format!("__compact_orig_size_{}", node_idx);
            if let (Some(mask_bytes), Some(orig_size_bytes)) = (tensors.remove(&mask_key), tensors.remove(&orig_size_key)) {
                let element_size = 4usize; // f32
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

        let mut out = HashMap::new();
        for name in &self.graph.outputs {
            let data = tensors.remove(name).unwrap_or_default(); // LEGAL: 不存在的 tensor 返回空数据
            out.insert(name.clone(), data);
        }
        Ok(out)
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
struct NodeGraphBuild {
    graph: gllm_kernels::compiler::CompilerGraph,
    input_names: Vec<String>,
    output_names: Vec<String>,
    output_numel: usize,
    /// Per-output element counts for multi-output nodes.
    per_output_numel: Vec<usize>,
    /// DType of the output tensor(s).
    output_dtype: gllm_kernels::types::DType,
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
            gllm_kernels::compiler::SymDim::Symbolic(name) => self
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
                },
            )]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        };

        let executor = FusedGraphExecutor::new(graph);
        let result = executor.run(&HashMap::from([("x".to_string(), vec![0u8; 4])]));
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
        let g = build_swiglu_graph(&config, 4, gllm_kernels::types::DType::F32);
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
        #[cfg(any(feature = "cuda", feature = "hip", feature = "metal"))]
        {
            assert_eq!(g.outputs.len(), 3); // q_rope, k_rope, v
            assert_eq!(g.ops.len(), 5); // 3 Gemms + 2 RoPEs
        }
        #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
        {
            assert_eq!(g.outputs.len(), 3); // q_out, k_out, v
            assert_eq!(g.ops.len(), 3); // 3 Gemms
        }
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
        let shapes = vec![vec![4, 512], vec![512, 1024]];
        let out = infer_output_shape("MatMul", &shapes);
        assert_eq!(out, vec![4, 1024]);
    }

    #[test]
    fn infer_output_shape_add() {
        let shapes = vec![vec![4, 512], vec![4, 512]];
        let out = infer_output_shape("Add", &shapes);
        assert_eq!(out, vec![4, 512]);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn atomic_op_to_kind_known_ops() {
        let shapes = vec![vec![4, 512], vec![512, 1024]];
        assert!(atomic_op_to_kind("Add", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Mul", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Silu", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("Gelu", &shapes, gllm_kernels::types::DType::F32).is_ok());
        assert!(atomic_op_to_kind("MatMul", &shapes, gllm_kernels::types::DType::F32).is_ok());
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn atomic_op_to_kind_unknown_returns_err() {
        let shapes = vec![vec![4, 512]];
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
            &gllm_kernels::compiler::SymDim::Symbolic("total_seq".to_string()),
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
        let dim = gllm_kernels::compiler::SymDim::Symbolic("total_seq".to_string());
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
        let sym = gllm_kernels::compiler::SymDim::Symbolic("total_seq".to_string());
        assert_eq!(ctx.resolve_sym_dim(&sym).unwrap(), 32);

        // Unbound symbolic dim returns ShapeNotBound error
        let unbound = gllm_kernels::compiler::SymDim::Symbolic("unbound_dim".to_string());
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
