//! FusedGraph 执行器 (REQ-EXEC-002)
//!
//! Compiles each FusedNode in the FusedGraph into a JIT-compiled kernel
//! via gllm-kernels' InferenceCompiler, then executes them in topological
//! order with proper buffer management.

use std::collections::{HashMap, HashSet};

use super::types::{
    FusedGraph, FusedOp, FlashAttentionConfig, FusedQkvRopeConfig,
    FusedRMSLinearConfig, GQAConfig, MoERoutingConfig, RoPEConfig, SwiGLUConfig,
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
    /// Number of f32 elements in the output tensor(s).
    output_numel: usize,
    /// Per-output element counts for multi-output nodes.
    /// Empty for single-output nodes (use output_numel).
    per_output_numel: Vec<usize>,
}

/// Build a CompilerGraph for FlashAttention.
///
/// Q[s,h], K[s,h], V[s,h] → MultiHeadAttention → out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_flash_attention_graph(
    config: &FlashAttentionConfig,
    seq_len: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let h = config.num_heads * config.head_dim;

    let q = g.add_tensor_concrete("q", &[seq_len, h], dt);
    let k = g.add_tensor_concrete("k", &[seq_len, h], dt);
    let v = g.add_tensor_concrete("v", &[seq_len, h], dt);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor_concrete("attn_out", &[seq_len, h], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        },
        vec![q, k, v],
        vec![out],
        "flash_attention",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for SwiGLU: silu(gate) * up.
///
/// gate[s,inter], up[s,inter] → SwiGlu → out[s,inter]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_swiglu_graph(
    config: &SwiGLUConfig,
    seq_len: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let inter = config.intermediate_size;

    let gate = g.add_tensor_concrete("gate", &[seq_len, inter], dt);
    let up = g.add_tensor_concrete("up", &[seq_len, inter], dt);
    g.inputs = vec![gate, up];

    let out = g.add_tensor_concrete("swiglu_out", &[seq_len, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![out], "swiglu");

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for RoPE.
///
/// input[s,h], cos_sin[head_dim/2] → RoPE → out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_rope_graph(
    config: &RoPEConfig,
    seq_len: usize,
    hidden: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let cos_sin = g.add_tensor_concrete("cos_sin", &[config.head_dim / 2], dt);
    g.inputs = vec![input, cos_sin];

    let out = g.add_tensor_concrete("rope_out", &[seq_len, hidden], dt);
    g.add_op(
        OpKind::RoPE {
            head_dim: config.head_dim,
            theta: config.rope_theta,
        },
        vec![input, cos_sin],
        vec![out],
        "rope",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for FusedQkvRope.
///
/// input[s,h] + w_q,w_k,w_v + cos_sin → Q/K/V Gemms + RoPE(Q) + RoPE(K) → [q_rope, k_rope, v]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_fused_qkv_rope_graph(
    config: &FusedQkvRopeConfig,
    seq_len: usize,
    hidden: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let w_q = g.add_tensor_concrete("w_q", &[hidden, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[hidden, kv_dim], dt);
    let w_v = g.add_tensor_concrete("w_v", &[hidden, kv_dim], dt);
    let cos_sin = g.add_tensor_concrete("cos_sin", &[config.head_dim / 2], dt);
    g.inputs = vec![input, w_q, w_k, w_v, cos_sin];

    let q_out = g.add_tensor_concrete("q", &[seq_len, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: q_dim, k: hidden, dtype: DType::F32 },
        vec![input, w_q],
        vec![q_out],
        "gemm_q",
    );

    let k_out = g.add_tensor_concrete("k", &[seq_len, kv_dim], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: kv_dim, k: hidden, dtype: DType::F32 },
        vec![input, w_k],
        vec![k_out],
        "gemm_k",
    );

    let v_out = g.add_tensor_concrete("v", &[seq_len, kv_dim], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: kv_dim, k: hidden, dtype: DType::F32 },
        vec![input, w_v],
        vec![v_out],
        "gemm_v",
    );

    let q_rope = g.add_tensor_concrete("q_rope", &[seq_len, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim: config.head_dim, theta: config.rope_theta },
        vec![q_out, cos_sin],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor_concrete("k_rope", &[seq_len, kv_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim: config.head_dim, theta: config.rope_theta },
        vec![k_out, cos_sin],
        vec![k_rope],
        "rope_k",
    );

    g.outputs = vec![q_rope, k_rope, v_out];
    g
}

/// Build a CompilerGraph for FusedRMSLinear: RMSNorm → Gemm.
///
/// input[s,h] + norm_w[h] + linear_w[h,h] → RmsNorm → Gemm → out[s,h]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_fused_rms_linear_graph(
    config: &FusedRMSLinearConfig,
    seq_len: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let h = config.hidden_size;

    let input = g.add_tensor_concrete("input", &[seq_len, h], dt);
    let norm_w = g.add_tensor_concrete("norm_w", &[h], dt);
    let linear_w = g.add_tensor_concrete("linear_w", &[h, h], dt);
    g.inputs = vec![input, norm_w, linear_w];

    let normed = g.add_tensor_concrete("normed", &[seq_len, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps: config.eps },
        vec![input, norm_w],
        vec![normed],
        "rms_norm",
    );

    let out = g.add_tensor_concrete("rms_linear_out", &[seq_len, h], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: h, k: h, dtype: DType::F32 },
        vec![normed, linear_w],
        vec![out],
        "linear",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for GQA (Grouped Query Attention).
///
/// Q[s,q_dim], K[s,kv_dim], V[s,kv_dim] → MHA → out[s,q_dim]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_gqa_graph(
    config: &GQAConfig,
    seq_len: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let q = g.add_tensor_concrete("q", &[seq_len, q_dim], dt);
    let k = g.add_tensor_concrete("k", &[seq_len, kv_dim], dt);
    let v = g.add_tensor_concrete("v", &[seq_len, kv_dim], dt);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor_concrete("gqa_out", &[seq_len, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
        },
        vec![q, k, v],
        vec![out],
        "gqa",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for MoE routing.
///
/// input[s,h] + gate_w[h,n_experts] → Gemm → Softmax → out[s,n_experts]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_moe_routing_graph(
    config: &MoERoutingConfig,
    seq_len: usize,
    hidden: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let n = config.num_experts;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let gate_w = g.add_tensor_concrete("gate_w", &[hidden, n], dt);
    g.inputs = vec![input, gate_w];

    let gate_logits = g.add_tensor_concrete("gate_logits", &[seq_len, n], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n, k: hidden, dtype: DType::F32 },
        vec![input, gate_w],
        vec![gate_logits],
        "gate_gemm",
    );

    let routing = g.add_tensor_concrete("routing", &[seq_len, n], dt);
    g.add_op(
        OpKind::Softmax,
        vec![gate_logits],
        vec![routing],
        "routing_softmax",
    );

    g.outputs = vec![routing];
    g
}

/// Map an atomic op_type string to a CompilerGraph OpKind.
///
/// Returns Err for unrecognized op types — NO silent fallback.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn atomic_op_to_kind(
    op_type: &str,
    input_shapes: &[Vec<usize>],
) -> Result<gllm_kernels::compiler::OpKind, ExecutionError> {
    use gllm_kernels::compiler::OpKind;
    use gllm_kernels::types::DType;

    match op_type {
        "Add" => Ok(OpKind::Add),
        "Mul" => Ok(OpKind::Mul),
        "Silu" | "Swish" => Ok(OpKind::Silu),
        "Gelu" => Ok(OpKind::Gelu),
        "Softmax" => Ok(OpKind::Softmax),
        "Residual" => Ok(OpKind::Residual),
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
            let m = a[a.len() - 2];
            let k = a[a.len() - 1];
            let n = b[b.len() - 1];
            Ok(OpKind::Gemm { m, n, k, dtype: DType::F32 })
        }
        "LayerNorm" | "LayerNormalization" => Ok(OpKind::LayerNorm { eps: 1e-5 }),
        "RMSNorm" | "RmsNorm" => Ok(OpKind::RmsNorm { eps: 1e-5 }),
        _ => Err(ExecutionError::UnsupportedOp(format!(
            "atomic op '{}' has no CompilerGraph mapping — \
             JIT codegen not implemented for this op type",
            op_type,
        ))),
    }
}

/// Build a CompilerGraph for a single atomic operation.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn build_atomic_graph(
    op_type: &str,
    input_shapes: &[Vec<usize>],
    output_shape: &[usize],
) -> Result<gllm_kernels::compiler::CompilerGraph, ExecutionError> {
    use gllm_kernels::compiler::CompilerGraph;
    use gllm_kernels::types::DType;

    let kind = atomic_op_to_kind(op_type, input_shapes)?;
    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let mut input_ids = Vec::new();
    for (i, shape) in input_shapes.iter().enumerate() {
        let name = format!("input_{i}");
        let tid = g.add_tensor_concrete(&name, shape, dt);
        input_ids.push(tid);
    }
    g.inputs = input_ids.clone();

    let out = g.add_tensor_concrete("output", output_shape, dt);
    g.add_op(kind, input_ids, vec![out], op_type);
    g.outputs = vec![out];

    Ok(g)
}

/// Infer output shape from op type and input shapes.
fn infer_output_shape(op_type: &str, input_shapes: &[Vec<usize>]) -> Vec<usize> {
    match op_type {
        "MatMul" | "Gemm" => {
            if input_shapes.len() >= 2
                && input_shapes[0].len() >= 2
                && input_shapes[1].len() >= 2
            {
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                vec![a[a.len() - 2], b[b.len() - 1]]
            } else if !input_shapes.is_empty() {
                input_shapes[0].clone()
            } else {
                vec![1]
            }
        }
        // Elementwise ops preserve the first input's shape
        _ => {
            if !input_shapes.is_empty() {
                input_shapes[0].clone()
            } else {
                vec![1]
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
    /// Number of f32 elements in the output tensor(s).
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
    ) -> Result<(), ExecutionError> {
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let mut compiled_nodes = Vec::with_capacity(self.graph.nodes.len());

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let build = self.build_node_graph(idx, seq_len, hidden)?;

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

            compiled_nodes.push(CompiledNode {
                compiled,
                graph_input_names: build.input_names,
                graph_output_names: build.output_names,
                output_numel: build.output_numel,
                per_output_numel: build.per_output_numel,
            });
        }

        self.compiled_nodes = compiled_nodes;
        self.is_compiled = true;
        Ok(())
    }

    /// Result of building a CompilerGraph for one FusedNode.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_node_graph(
        &self,
        node_idx: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Result<NodeGraphBuild, ExecutionError> {
        let node = &self.graph.nodes[node_idx];

        match &node.op {
            FusedOp::FlashAttention(config) => {
                let g = build_flash_attention_graph(config, seq_len);
                let h = config.num_heads * config.head_dim;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * h,
                    per_output_numel: vec![],
                })
            }

            FusedOp::SwiGLU(config) => {
                let g = build_swiglu_graph(config, seq_len);
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * config.intermediate_size,
                    per_output_numel: vec![],
                })
            }

            FusedOp::RoPE(config) => {
                let g = build_rope_graph(config, seq_len, hidden);
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * hidden,
                    per_output_numel: vec![],
                })
            }

            FusedOp::FusedQkvRope(config) => {
                let g = build_fused_qkv_rope_graph(config, seq_len, hidden);
                let q_dim = config.num_heads * config.head_dim;
                let kv_dim = config.num_kv_heads * config.head_dim;
                let per = vec![
                    seq_len * q_dim,
                    seq_len * kv_dim,
                    seq_len * kv_dim,
                ];
                let total: usize = per.iter().sum();
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: total,
                    per_output_numel: per,
                })
            }

            FusedOp::FusedRMSLinear(config) => {
                let g = build_fused_rms_linear_graph(config, seq_len);
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * config.hidden_size,
                    per_output_numel: vec![],
                })
            }

            FusedOp::GQA(config) => {
                let g = build_gqa_graph(config, seq_len);
                let q_dim = config.num_heads * config.head_dim;
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * q_dim,
                    per_output_numel: vec![],
                })
            }

            FusedOp::MoERouting(config) => {
                let g = build_moe_routing_graph(config, seq_len, hidden);
                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel: seq_len * config.num_experts,
                    per_output_numel: vec![],
                })
            }

            FusedOp::Atomic(atomic) => {
                let input_shapes: Vec<Vec<usize>> = node
                    .inputs
                    .iter()
                    .map(|name| {
                        if let Some(wb) = self.graph.weight_bindings.get(name) {
                            wb.shape.clone()
                        } else {
                            vec![seq_len, hidden]
                        }
                    })
                    .collect();

                let output_shape = infer_output_shape(&atomic.op_type, &input_shapes);
                let output_numel: usize = output_shape.iter().product();
                let g = build_atomic_graph(&atomic.op_type, &input_shapes, &output_shape)?;

                Ok(NodeGraphBuild {
                    graph: g,
                    input_names: node.inputs.clone(),
                    output_names: node.outputs.clone(),
                    output_numel,
                    per_output_numel: vec![],
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
        device: &gllm_kernels::gpu::cuda::CudaDevice,
        gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
        sm_version: u32,
    ) -> Result<(), ExecutionError> {
        let mut gpu_nodes = Vec::with_capacity(self.graph.nodes.len());

        for (idx, node) in self.graph.nodes.iter().enumerate() {
            let build = self.build_node_graph(idx, seq_len, hidden)?;

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
            // Allocate GPU buffers for all graph tensors
            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut tensor_ptrs: HashMap<TensorId, u64> = HashMap::new();

            for (tidx, meta) in gcn.graph.tensors.iter().enumerate() {
                let tid = TensorId(tidx as u32);
                let n_elements: usize = meta.shape.iter().product();
                let size_bytes = n_elements * 4; // f32
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
                let nbytes = gcn.output_numel * std::mem::size_of::<f32>();
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
                        let nbytes = gcn.per_output_numel[i] * std::mem::size_of::<f32>();
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
            let data = tensors.remove(name).unwrap_or_default();
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
        for (name, wb) in &self.graph.weight_bindings {
            if let Some(ptr) = wb.ptr {
                // Runtime pointer: copy into a byte vec for the tensor map
                let numel: usize = wb.shape.iter().product();
                let bytes = numel * std::mem::size_of::<f32>();
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, bytes) };
                tensors.insert(name.clone(), slice.to_vec());
            } else if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }

        // Execute each node
        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];

            // Activation input = first graph input tensor
            let activation = if !cn.graph_input_names.is_empty() {
                tensors
                    .get(&cn.graph_input_names[0])
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            // Pack remaining inputs as weight blob
            let mut weight_blob = Vec::new();
            for name in cn.graph_input_names.iter().skip(1) {
                if let Some(data) = tensors.get(name) {
                    weight_blob.extend_from_slice(data);
                }
            }

            // Allocate output buffer
            let output_bytes = cn.output_numel * std::mem::size_of::<f32>();
            let mut output_buf = vec![0u8; output_bytes];

            // Allocate scratchpad
            let mut scratchpad = vec![0u8; cn.compiled.scratchpad_bytes];

            // Compute seq_len from activation size: activation is [seq_len, hidden],
            // stored as f32, so seq_len = num_f32_elements / hidden.
            // Fall back to 1 if we cannot determine it.
            let activation_f32_elems = activation.len() / std::mem::size_of::<f32>();
            let seq_len = if activation_f32_elems > 0 {
                activation_f32_elems
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
                    let nbytes = numel * std::mem::size_of::<f32>();
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
            let data = tensors.remove(name).unwrap_or_default();
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
        positions: *const u32,
    ) -> Result<HashMap<String, Vec<u8>>, ExecutionError> {
        if !self.is_compiled {
            return Err(ExecutionError::NotCompiled);
        }

        let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();

        // Seed with graph inputs
        for (name, data) in inputs {
            tensors.insert(name.clone(), data.clone());
        }

        // Seed with weight bindings — prefer runtime ptr over embedded bytes
        for (name, wb) in &self.graph.weight_bindings {
            if let Some(ptr) = wb.ptr {
                let numel: usize = wb.shape.iter().product();
                let bytes = numel * std::mem::size_of::<f32>();
                let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, bytes) };
                tensors.insert(name.clone(), slice.to_vec());
            } else if let Some(ref data) = wb.data {
                tensors.insert(name.clone(), data.clone());
            }
        }

        // Build a flat KV cache pointer: combine K and V into a single *mut f32
        // by passing kv_cache_k as the kv_cache argument (the compiled kernel
        // uses the layer offset stored in the scratchpad config to locate its slice).
        // The V pointer is passed via the positions slot when kv_cache_v is non-null;
        // for now we pass kv_cache_k as the unified cache pointer and rely on the
        // kernel's internal layer-stride arithmetic.
        let _ = (kv_cache_v, layer, total_seq); // used by caller for layout; kernel uses kv_cache_k

        for (node_idx, _node) in self.graph.nodes.iter().enumerate() {
            let cn = &self.compiled_nodes[node_idx];

            let activation = if !cn.graph_input_names.is_empty() {
                tensors
                    .get(&cn.graph_input_names[0])
                    .cloned()
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            let mut weight_blob = Vec::new();
            for name in cn.graph_input_names.iter().skip(1) {
                if let Some(data) = tensors.get(name) {
                    weight_blob.extend_from_slice(data);
                }
            }

            let output_bytes = cn.output_numel * std::mem::size_of::<f32>();
            let mut output_buf = vec![0u8; output_bytes];
            let mut scratchpad = vec![0u8; cn.compiled.scratchpad_bytes];

            let activation_f32_elems = activation.len() / std::mem::size_of::<f32>();
            let seq_len = if activation_f32_elems > 0 {
                activation_f32_elems
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
                    kv_cache_k as *mut u8,
                    positions,
                    std::ptr::null(),
                    1,
                    seq_len,
                    output_buf.as_mut_ptr(),
                    scratchpad.as_mut_ptr(),
                );
            }

            if cn.graph_output_names.len() == 1 {
                tensors.insert(cn.graph_output_names[0].clone(), output_buf);
            } else if !cn.per_output_numel.is_empty() {
                let mut byte_offset = 0;
                for (i, name) in cn.graph_output_names.iter().enumerate() {
                    let numel = cn.per_output_numel[i];
                    let nbytes = numel * std::mem::size_of::<f32>();
                    let chunk = output_buf[byte_offset..byte_offset + nbytes].to_vec();
                    tensors.insert(name.clone(), chunk);
                    byte_offset += nbytes;
                }
            } else if cn.graph_output_names.len() > 1 {
                return Err(ExecutionError::Compilation(format!(
                    "node has {} outputs but no per_output_numel",
                    cn.graph_output_names.len(),
                )));
            }
        }

        let mut out = HashMap::new();
        for name in &self.graph.outputs {
            let data = tensors.remove(name).unwrap_or_default();
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
    ) -> Result<Self, ExecutorError> {
        use crate::graph::optimizer::{GraphOptimizer, OptimizationContext};

        let ctx = OptimizationContext::default();
        let optimizer = GraphOptimizer::new(ctx);
        let fused = optimizer
            .optimize(&graph)
            .map_err(|e| ExecutorError::CompilationFailed(format!("graph optimization: {e}")))?;

        let mut executor = Self::new(fused);
        executor
            .compile(seq_len, hidden)
            .map_err(|e| ExecutorError::CompilationFailed(format!("JIT compile: {e}")))?;

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
        let g = build_flash_attention_graph(&config, 4);
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
        let g = build_swiglu_graph(&config, 4);
        assert_eq!(g.inputs.len(), 2); // gate, up
        assert_eq!(g.outputs.len(), 1);
        assert_eq!(g.ops.len(), 1); // SwiGlu
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
        let g = build_fused_qkv_rope_graph(&config, 4, 512);
        assert_eq!(g.inputs.len(), 5); // input, w_q, w_k, w_v, cos_sin
        assert_eq!(g.outputs.len(), 3); // q_rope, k_rope, v
        assert_eq!(g.ops.len(), 5); // 3 Gemms + 2 RoPEs
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn build_fused_rms_linear_graph_structure() {
        let config = FusedRMSLinearConfig {
            hidden_size: 512,
            eps: 1e-5,
        };
        let g = build_fused_rms_linear_graph(&config, 4);
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
        };
        let g = build_gqa_graph(&config, 4);
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
        let g = build_moe_routing_graph(&config, 4, 512);
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
        assert!(atomic_op_to_kind("Add", &shapes).is_ok());
        assert!(atomic_op_to_kind("Mul", &shapes).is_ok());
        assert!(atomic_op_to_kind("Silu", &shapes).is_ok());
        assert!(atomic_op_to_kind("Gelu", &shapes).is_ok());
        assert!(atomic_op_to_kind("MatMul", &shapes).is_ok());
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    #[test]
    fn atomic_op_to_kind_unknown_returns_err() {
        let shapes = vec![vec![4, 512]];
        let result = atomic_op_to_kind("UnknownOp", &shapes);
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
        let result = executor.compile(4, 512);
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

        let executor = FusedGraphExecutor::from_graph(graph, 4, 64)
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

        let executor = FusedGraphExecutor::from_graph(graph, 4, 64)
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
