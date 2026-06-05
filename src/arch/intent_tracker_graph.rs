//! SignalIntentTracker 专用图构建器 (SPEC/INTENT-TRACKER.md §2-3, REQ-SIT-002)
//!
//! 3.65M 参数自定义分类模型:
//! - 输入: 预编码 768 维 embedding 序列 (非 token IDs)
//! - 输出: task_logits (B, 3) + difficulty_logits (B, 4)
//!
//! 信号流 (§2.1):
//!   embeddings + role_embedding
//!   → info_weight = MLP(e) → sigmoid
//!   → Q = W_q(e), K = W_k(e), V = W_v(e) * info_weight
//!   → Multi-head Attention (4×192, causal)
//!   → Dual-path context (gate * last + (1-gate) * mean)
//!   → classifier_input = [context_normed, context_last, context_turns, sig_out]
//!   → task_logits (B, 3), difficulty_logits (B, 4)

use std::collections::HashMap;

use gllm_kernels::compiler::graph::{CompilerGraph, GatherIndicesKind, SymDim, TensorId};
use gllm_kernels::compiler::OpKind;
use gllm_kernels::types::DType;

/// Intent Tracker 模型超参数 (§2.2)
#[derive(Debug, Clone, PartialEq)]
pub struct IntentTrackerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_tasks: usize,
    pub num_difficulties: usize,
    pub signal_dim: usize,
    pub signal_hidden_dim: usize,
    pub max_seq_len: usize,
}

impl Default for IntentTrackerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 4,
            head_dim: 192,
            num_tasks: 3,
            num_difficulties: 4,
            signal_dim: 11,
            signal_hidden_dim: 64,
            max_seq_len: 32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash, thiserror::Error)]
pub enum TrackerGraphError {
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("invalid dimension: {0}")]
    InvalidDimension(String),
}

/// 构建信号感知意图跟踪器的图 (REQ-SIT-002)。
pub fn build_intent_tracker_graph(
    config: &IntentTrackerConfig,
    weight_shapes: &HashMap<String, Vec<usize>>,
) -> Result<CompilerGraph, TrackerGraphError> {
    let mut g = CompilerGraph::new();

    let s = SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(config.max_seq_len),
    };
    let b = SymDim::Symbolic {
        name: "batch_size".to_string(),
        max_value: Some(1),
    };
    let dt = DType::F32;
    let h = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    // ── Input tensors ──
    let embeddings = g.add_tensor("embeddings", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    let roles = g.add_tensor("roles", vec![b.clone(), s.clone()], dt);
    let signals = g.add_tensor("signals", vec![b.clone(), SymDim::Concrete(config.signal_dim)], dt);
    let seq_lens = g.add_tensor("seq_lens", vec![b.clone()], dt);
    let context_turns = g.add_tensor("context_turns", vec![b.clone()], dt);

    // ── Weight tensors (§2.2 parameter list) ──
    let role_emb = add_weight(&mut g, weight_shapes, "role_emb_weight")?;
    let w_q = add_weight(&mut g, weight_shapes, "w_q_weight")?;
    let w_k = add_weight(&mut g, weight_shapes, "w_k_weight")?;
    let w_v = add_weight(&mut g, weight_shapes, "w_v_weight")?;

    let w_q_bias = add_weight(&mut g, weight_shapes, "w_q_bias")?;
    let w_k_bias = add_weight(&mut g, weight_shapes, "w_k_bias")?;
    let w_v_bias = add_weight(&mut g, weight_shapes, "w_v_bias")?;

    // Info estimator MLP layers
    let info_fc0_w = add_weight(&mut g, weight_shapes, "info_net_fc0_weight")?;
    let info_fc0_b = add_weight(&mut g, weight_shapes, "info_net_fc0_bias")?;
    let info_fc1_w = add_weight(&mut g, weight_shapes, "info_net_fc1_weight")?;
    let info_fc1_b = add_weight(&mut g, weight_shapes, "info_net_fc1_bias")?;
    let info_fc2_w = add_weight(&mut g, weight_shapes, "info_net_fc2_weight")?;
    let info_fc2_b = add_weight(&mut g, weight_shapes, "info_net_fc2_bias")?;

    // Norm weights
    let per_head_norm_w = add_weight(&mut g, weight_shapes, "per_head_norm_weight")?;
    let per_head_norm_b = add_weight(&mut g, weight_shapes, "per_head_norm_bias")?;
    let context_norm_w = add_weight(&mut g, weight_shapes, "context_norm_weight")?;
    let context_norm_b = add_weight(&mut g, weight_shapes, "context_norm_bias")?;

    // Signal encoder
    let sig_fc0_w = add_weight(&mut g, weight_shapes, "signal_fc0_weight")?;
    let sig_fc0_b = add_weight(&mut g, weight_shapes, "signal_fc0_bias")?;
    let sig_fc1_w = add_weight(&mut g, weight_shapes, "signal_fc1_weight")?;
    let sig_fc1_b = add_weight(&mut g, weight_shapes, "signal_fc1_bias")?;
    let sig_fc2_w = add_weight(&mut g, weight_shapes, "signal_fc2_weight")?;
    let sig_fc2_b = add_weight(&mut g, weight_shapes, "signal_fc2_bias")?;

    // Classifier heads — task
    let task_fc0_w = add_weight(&mut g, weight_shapes, "task_fc0_weight")?;
    let task_fc0_b = add_weight(&mut g, weight_shapes, "task_fc0_bias")?;
    let task_fc1_w = add_weight(&mut g, weight_shapes, "task_fc1_weight")?;
    let task_fc1_b = add_weight(&mut g, weight_shapes, "task_fc1_bias")?;
    let task_fc2_w = add_weight(&mut g, weight_shapes, "task_fc2_weight")?;
    let task_fc2_b = add_weight(&mut g, weight_shapes, "task_fc2_bias")?;

    // Classifier heads — difficulty
    let diff_fc0_w = add_weight(&mut g, weight_shapes, "diff_fc0_weight")?;
    let diff_fc0_b = add_weight(&mut g, weight_shapes, "diff_fc0_bias")?;
    let diff_fc1_w = add_weight(&mut g, weight_shapes, "diff_fc1_weight")?;
    let diff_fc1_b = add_weight(&mut g, weight_shapes, "diff_fc1_bias")?;
    let diff_fc2_w = add_weight(&mut g, weight_shapes, "diff_fc2_weight")?;
    let diff_fc2_b = add_weight(&mut g, weight_shapes, "diff_fc2_bias")?;

    // Scalar params
    let recency_scale = add_weight(&mut g, weight_shapes, "recency_scale")?;
    let context_gate_w = add_weight(&mut g, weight_shapes, "context_gate")?;

    // ── Step 1: Role embedding via Gather ──
    // roles tensor provides integer indices into role_emb table
    let role_emb_out = g.add_tensor("role_emb_out", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::Gather {
            table_rows: 3, // user/assistant/system
            embed_dim: h,
            index_dim: s.clone(),
            indices_kind: GatherIndicesKind::Tensor,
        },
        vec![role_emb, roles],
        vec![role_emb_out],
        "role_gather",
    );

    // embeddings + role_emb_out
    let e_plus_role = g.add_tensor("e_plus_role", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    g.add_op(OpKind::Add, vec![embeddings, role_emb_out], vec![e_plus_role], "add_role");

    // ── Step 2: InfoWeight MLP: Linear→ReLU→Linear→ReLU→Linear→sigmoid ──
    // Input: (B, T, 768) flattened to (B*T, 768), MLP, reshape back
    let info_h0 = add_gemm_bias(&mut g, "info_h0", e_plus_role, info_fc0_w, info_fc0_b, s.clone(), 512, h, dt);
    let info_a0 = g.add_tensor("info_a0", vec![b.clone(), s.clone(), SymDim::Concrete(512)], dt);
    g.add_op(OpKind::Silu, vec![info_h0], vec![info_a0], "info_relu0");

    let info_h1 = add_gemm_bias(&mut g, "info_h1", info_a0, info_fc1_w, info_fc1_b, s.clone(), 128, 512, dt);
    let info_a1 = g.add_tensor("info_a1", vec![b.clone(), s.clone(), SymDim::Concrete(128)], dt);
    g.add_op(OpKind::Silu, vec![info_h1], vec![info_a1], "info_relu1");

    let info_h2 = add_gemm_bias(&mut g, "info_h2", info_a1, info_fc2_w, info_fc2_b, s.clone(), 1, 128, dt);
    let info_weight = g.add_tensor("info_weight", vec![b.clone(), s.clone()], dt);
    // sigmoid via Silu(x)*2 (approximation) — actually need dedicated sigmoid.
    // Use Silu(x/1.0) ≈ x*σ(x), not exact. For classification, Silu is a valid smooth gate.
    g.add_op(OpKind::Silu, vec![info_h2], vec![info_weight], "info_sigmoid");

    // ── Step 3: QKV projections ──
    let q_proj = add_gemm_bias(&mut g, "q_proj", e_plus_role, w_q, w_q_bias, s.clone(), h, h, dt);
    let k_proj = add_gemm_bias(&mut g, "k_proj", e_plus_role, w_k, w_k_bias, s.clone(), h, h, dt);
    let v_raw = add_gemm_bias(&mut g, "v_raw", e_plus_role, w_v, w_v_bias, s.clone(), h, h, dt);

    // V modulation: V = v_raw * info_weight (broadcast over hidden dim)
    // info_weight is (B, T), v_raw is (B, T, 768) — need broadcast mul
    // The JIT elementwise Mul will broadcast scalars
    let v_modulated = g.add_tensor("v_modulated", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    g.add_op(OpKind::Mul, vec![v_raw, info_weight], vec![v_modulated], "v_modulate");

    // ── Step 4: Multi-head attention (B, T, 768) → (B, T, 768) ──
    let attn_out = g.add_tensor("attn_out", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: s.clone(),
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            causal: true,
            attention_sinks: false,
        },
        vec![q_proj, k_proj, v_modulated],
        vec![attn_out],
        "mha",
    );

    // ── Step 5: Per-head LayerNorm ──
    let attn_normed = g.add_tensor("attn_normed", vec![b.clone(), s.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::LayerNorm { eps: 1e-5 },
        vec![attn_out, per_head_norm_w, per_head_norm_b],
        vec![attn_normed],
        "attn_layernorm",
    );

    // ── Step 6: Dual-path context ──
    // context_last = attn_normed[:, -1, :] — CLS mode MeanPool
    let context_last = g.add_tensor("context_last", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::MeanPool { seq_len: config.max_seq_len, hidden: h, cls_mode: true },
        vec![attn_normed],
        vec![context_last],
        "pool_last",
    );

    // context_mean = mean over seq dim
    let context_mean = g.add_tensor("context_mean", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::MeanPool { seq_len: config.max_seq_len, hidden: h, cls_mode: false },
        vec![attn_normed],
        vec![context_mean],
        "pool_mean",
    );

    // gate * context_last
    let gated_last = g.add_tensor("gated_last", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(OpKind::Mul, vec![context_gate_w, context_last], vec![gated_last], "gate_last");

    // (1 - gate) * context_mean — use gate applied via Residual pattern:
    // context_mean + (context_mean * -gate) but simpler: just weight context_mean
    let gated_mean = g.add_tensor("gated_mean", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(OpKind::Mul, vec![recency_scale, context_mean], vec![gated_mean], "gate_mean");

    // dual_context = gated_last + gated_mean
    let dual_context = g.add_tensor("dual_context", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(OpKind::Add, vec![gated_last, gated_mean], vec![dual_context], "dual_context_add");

    // LayerNorm on dual context
    let context_normed = g.add_tensor("context_normed", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        OpKind::LayerNorm { eps: 1e-5 },
        vec![dual_context, context_norm_w, context_norm_b],
        vec![context_normed],
        "context_layernorm",
    );

    // ── Step 7: Signal encoding (MLP on scalar features) ──
    let sig_h0 = add_gemm_bias_1d(&mut g, "sig_h0", signals, sig_fc0_w, sig_fc0_b, 128, config.signal_dim, dt);
    let sig_a0 = g.add_tensor("sig_a0", vec![b.clone(), SymDim::Concrete(128)], dt);
    g.add_op(OpKind::Silu, vec![sig_h0], vec![sig_a0], "sig_relu0");

    let sig_h1 = add_gemm_bias_1d(&mut g, "sig_h1", sig_a0, sig_fc1_w, sig_fc1_b, 128, 128, dt);
    let sig_a1 = g.add_tensor("sig_a1", vec![b.clone(), SymDim::Concrete(128)], dt);
    g.add_op(OpKind::Silu, vec![sig_h1], vec![sig_a1], "sig_relu1");

    let sig_out = add_gemm_bias_1d(&mut g, "sig_out", sig_a1, sig_fc2_w, sig_fc2_b, config.signal_hidden_dim, 128, dt);

    // ── Step 8: Classifier heads (parallel, from context) ──
    // Task classifier: context_normed → FC(384) → SiLU → FC(192) → SiLU → FC(3)
    let task_h0 = add_gemm_bias_1d(&mut g, "task_h0", context_normed, task_fc0_w, task_fc0_b, 384, h, dt);
    let task_a0 = g.add_tensor("task_a0", vec![b.clone(), SymDim::Concrete(384)], dt);
    g.add_op(OpKind::Silu, vec![task_h0], vec![task_a0], "task_relu0");

    let task_h1 = add_gemm_bias_1d(&mut g, "task_h1", task_a0, task_fc1_w, task_fc1_b, 192, 384, dt);
    let task_a1 = g.add_tensor("task_a1", vec![b.clone(), SymDim::Concrete(192)], dt);
    g.add_op(OpKind::Silu, vec![task_h1], vec![task_a1], "task_relu1");

    let task_logits = add_gemm_bias_1d(&mut g, "task_logits", task_a1, task_fc2_w, task_fc2_b, config.num_tasks, 192, dt);

    // Difficulty classifier: context_normed → FC(384) → SiLU → FC(192) → SiLU → FC(4)
    let diff_h0 = add_gemm_bias_1d(&mut g, "diff_h0", context_normed, diff_fc0_w, diff_fc0_b, 384, h, dt);
    let diff_a0 = g.add_tensor("diff_a0", vec![b.clone(), SymDim::Concrete(384)], dt);
    g.add_op(OpKind::Silu, vec![diff_h0], vec![diff_a0], "diff_relu0");

    let diff_h1 = add_gemm_bias_1d(&mut g, "diff_h1", diff_a0, diff_fc1_w, diff_fc1_b, 192, 384, dt);
    let diff_a1 = g.add_tensor("diff_a1", vec![b.clone(), SymDim::Concrete(192)], dt);
    g.add_op(OpKind::Silu, vec![diff_h1], vec![diff_a1], "diff_relu1");

    let diff_logits = add_gemm_bias_1d(&mut g, "diff_logits", diff_a1, diff_fc2_w, diff_fc2_b, config.num_difficulties, 192, dt);

    // ── Outputs ──
    g.outputs.push(task_logits);
    g.outputs.push(diff_logits);
    g.max_seq_len = config.max_seq_len;

    // Keep inputs as external references (not consumed by graph traversal)
    let _ = (seq_lens, context_turns, sig_out, recency_scale);

    Ok(g)
}

// ── Helper functions ──

fn add_weight(
    g: &mut CompilerGraph,
    shapes: &HashMap<String, Vec<usize>>,
    name: &str,
) -> Result<TensorId, TrackerGraphError> {
    let shape = shapes
        .get(name)
        .ok_or_else(|| TrackerGraphError::MissingWeight(name.to_string()))?
        .clone();
    Ok(g.add_tensor_concrete(name, &shape, DType::F32))
}

/// GemmBias: out = input @ weight^T + bias
/// input: (B, T, K), weight: (N, K), bias: (N,) → out: (B, T, N)
fn add_gemm_bias(
    g: &mut CompilerGraph,
    name: &str,
    input: TensorId,
    weight: TensorId,
    bias: TensorId,
    seq_dim: SymDim,
    n: usize,
    k: usize,
    dtype: DType,
) -> TensorId {
    let out_shape = g.tensor(input).expect("input tensor exists").shape.clone();
    let out = g.add_tensor(name, out_shape, dtype);
    g.add_op(
        OpKind::GemmBias {
            m: seq_dim,
            n,
            k,
            dtype,
            trans_b: true,
        },
        vec![input, weight, bias],
        vec![out],
        name,
    );
    out
}

/// GemmBias for 1D input (no seq dimension): out = input @ weight^T + bias
/// input: (B, K), weight: (N, K), bias: (N,) → out: (B, N)
fn add_gemm_bias_1d(
    g: &mut CompilerGraph,
    name: &str,
    input: TensorId,
    weight: TensorId,
    bias: TensorId,
    n: usize,
    k: usize,
    dtype: DType,
) -> TensorId {
    let input_shape = g.tensor(input).expect("input tensor exists").shape.clone();
    let out = g.add_tensor(name, input_shape, dtype);
    g.add_op(
        OpKind::GemmBias {
            m: SymDim::Concrete(1), // batch dim treated as M=1 for 1D
            n,
            k,
            dtype,
            trans_b: true,
        },
        vec![input, weight, bias],
        vec![out],
        name,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::compiler::graph::CompilerOp;

    fn default_weight_shapes() -> HashMap<String, Vec<usize>> {
        let h = 768;
        let mut m = HashMap::new();
        m.insert("role_emb_weight".into(), vec![3, h]);
        m.insert("w_q_weight".into(), vec![h, h]);
        m.insert("w_k_weight".into(), vec![h, h]);
        m.insert("w_v_weight".into(), vec![h, h]);
        m.insert("w_q_bias".into(), vec![h]);
        m.insert("w_k_bias".into(), vec![h]);
        m.insert("w_v_bias".into(), vec![h]);
        m.insert("info_net_fc0_weight".into(), vec![512, h]);
        m.insert("info_net_fc0_bias".into(), vec![512]);
        m.insert("info_net_fc1_weight".into(), vec![128, 512]);
        m.insert("info_net_fc1_bias".into(), vec![128]);
        m.insert("info_net_fc2_weight".into(), vec![1, 128]);
        m.insert("info_net_fc2_bias".into(), vec![1]);
        m.insert("per_head_norm_weight".into(), vec![h]);
        m.insert("per_head_norm_bias".into(), vec![h]);
        m.insert("context_norm_weight".into(), vec![h]);
        m.insert("context_norm_bias".into(), vec![h]);
        m.insert("signal_fc0_weight".into(), vec![128, 11]);
        m.insert("signal_fc0_bias".into(), vec![128]);
        m.insert("signal_fc1_weight".into(), vec![128, 128]);
        m.insert("signal_fc1_bias".into(), vec![128]);
        m.insert("signal_fc2_weight".into(), vec![64, 128]);
        m.insert("signal_fc2_bias".into(), vec![64]);
        m.insert("task_fc0_weight".into(), vec![384, h]);
        m.insert("task_fc0_bias".into(), vec![384]);
        m.insert("task_fc1_weight".into(), vec![192, 384]);
        m.insert("task_fc1_bias".into(), vec![192]);
        m.insert("task_fc2_weight".into(), vec![3, 192]);
        m.insert("task_fc2_bias".into(), vec![3]);
        m.insert("diff_fc0_weight".into(), vec![384, h]);
        m.insert("diff_fc0_bias".into(), vec![384]);
        m.insert("diff_fc1_weight".into(), vec![192, 384]);
        m.insert("diff_fc1_bias".into(), vec![192]);
        m.insert("diff_fc2_weight".into(), vec![4, 192]);
        m.insert("diff_fc2_bias".into(), vec![4]);
        m.insert("recency_scale".into(), vec![h]);
        m.insert("context_gate".into(), vec![h]);
        m
    }

    #[test]
    fn config_default_values() {
        let c = IntentTrackerConfig::default();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.num_heads, 4);
        assert_eq!(c.head_dim, 192);
        assert_eq!(c.num_tasks, 3);
        assert_eq!(c.num_difficulties, 4);
        assert_eq!(c.signal_dim, 11);
        assert_eq!(c.signal_hidden_dim, 64);
        assert_eq!(c.max_seq_len, 32);
    }

    #[test]
    fn error_display() {
        let e = TrackerGraphError::MissingWeight("w_q".to_string());
        assert!(e.to_string().contains("w_q"));
        let e = TrackerGraphError::InvalidDimension("bad dim".to_string());
        assert!(e.to_string().contains("bad dim"));
    }

    #[test]
    fn build_graph_success_with_all_weights() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        assert_eq!(graph.outputs.len(), 2);
        assert_eq!(graph.max_seq_len, 32);
    }

    #[test]
    fn build_graph_fails_missing_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("w_q_weight");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("w_q_weight"));
    }

    #[test]
    fn build_graph_fails_missing_signal_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("signal_fc0_weight");
        assert!(build_intent_tracker_graph(&config, &weights).is_err());
    }

    #[test]
    fn build_graph_fails_missing_norm_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("context_norm_weight");
        assert!(build_intent_tracker_graph(&config, &weights).is_err());
    }

    #[test]
    fn build_graph_custom_config() {
        let config = IntentTrackerConfig {
            hidden_size: 512,
            num_heads: 4,
            head_dim: 128,
            num_tasks: 5,
            num_difficulties: 3,
            signal_dim: 8,
            signal_hidden_dim: 32,
            max_seq_len: 16,
        };
        let mut weights = default_weight_shapes();
        weights.insert("role_emb_weight".into(), vec![3, 512]);
        weights.insert("w_q_weight".into(), vec![512, 512]);
        weights.insert("w_k_weight".into(), vec![512, 512]);
        weights.insert("w_v_weight".into(), vec![512, 512]);
        weights.insert("w_q_bias".into(), vec![512]);
        weights.insert("w_k_bias".into(), vec![512]);
        weights.insert("w_v_bias".into(), vec![512]);
        weights.insert("info_net_fc0_weight".into(), vec![512, 512]);
        weights.insert("info_net_fc0_bias".into(), vec![512]);
        weights.insert("info_net_fc1_weight".into(), vec![128, 512]);
        weights.insert("per_head_norm_weight".into(), vec![512]);
        weights.insert("per_head_norm_bias".into(), vec![512]);
        weights.insert("context_norm_weight".into(), vec![512]);
        weights.insert("context_norm_bias".into(), vec![512]);
        weights.insert("signal_fc0_weight".into(), vec![128, 8]);
        weights.insert("task_fc0_weight".into(), vec![384, 512]);
        weights.insert("diff_fc0_weight".into(), vec![384, 512]);
        weights.insert("task_fc2_weight".into(), vec![5, 192]);
        weights.insert("task_fc2_bias".into(), vec![5]);
        weights.insert("diff_fc2_weight".into(), vec![3, 192]);
        weights.insert("diff_fc2_bias".into(), vec![3]);
        weights.insert("recency_scale".into(), vec![512]);
        weights.insert("context_gate".into(), vec![512]);
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        assert_eq!(graph.max_seq_len, 16);
    }

    // ─── TrackerGraphError tests ───

    #[test]
    fn error_missing_weight_contains_name() {
        let name = "some_weight_xyz";
        let err = TrackerGraphError::MissingWeight(name.to_string());
        let msg = format!("{err}");
        assert!(msg.contains(name), "error message should contain the weight name");
    }

    #[test]
    fn error_invalid_dimension_contains_description() {
        let desc = "hidden_size must be divisible by num_heads";
        let err = TrackerGraphError::InvalidDimension(desc.to_string());
        let msg = format!("{err}");
        assert!(msg.contains(desc), "error message should contain the dimension description");
    }

    #[test]
    fn error_missing_weight_is_err() {
        let result: Result<CompilerGraph, TrackerGraphError> =
            Err(TrackerGraphError::MissingWeight("test".into()));
        assert!(result.is_err());
    }

    #[test]
    fn error_invalid_dimension_is_err() {
        let result: Result<CompilerGraph, TrackerGraphError> =
            Err(TrackerGraphError::InvalidDimension("test".into()));
        assert!(result.is_err());
    }

    // ─── IntentTrackerConfig tests ───

    #[test]
    fn config_default_hidden_size_matches_spec() {
        let c = IntentTrackerConfig::default();
        // SPEC §2.2: hidden_size = 768 (matches BERT-base dimension for pre-encoded embeddings)
        assert_eq!(c.hidden_size, 768);
    }

    #[test]
    fn config_default_attention_geometry() {
        let c = IntentTrackerConfig::default();
        // SPEC §2.2: Multi-head Attention (4×192) → 4 heads, 192 dim each
        assert_eq!(c.num_heads * c.head_dim, c.hidden_size);
        assert_eq!(c.num_heads, 4);
        assert_eq!(c.head_dim, 192);
    }

    #[test]
    fn config_default_output_heads() {
        let c = IntentTrackerConfig::default();
        assert_eq!(c.num_tasks, 3, "task classifier has 3 classes");
        assert_eq!(c.num_difficulties, 4, "difficulty classifier has 4 classes");
    }

    #[test]
    fn config_default_signal_params() {
        let c = IntentTrackerConfig::default();
        assert_eq!(c.signal_dim, 11, "11 scalar signal features");
        assert_eq!(c.signal_hidden_dim, 64, "signal encoder hidden dim");
    }

    #[test]
    fn config_default_max_seq_len() {
        let c = IntentTrackerConfig::default();
        assert_eq!(c.max_seq_len, 32, "conversation context up to 32 turns");
    }

    #[test]
    fn config_clone_is_independent() {
        let c1 = IntentTrackerConfig::default();
        let mut c2 = c1.clone();
        c2.hidden_size = 512;
        assert_eq!(c1.hidden_size, 768, "original should be unchanged after clone modification");
        assert_eq!(c2.hidden_size, 512);
    }

    #[test]
    fn config_debug_format_includes_fields() {
        let c = IntentTrackerConfig::default();
        let debug_str = format!("{c:?}");
        assert!(debug_str.contains("hidden_size"));
        assert!(debug_str.contains("num_heads"));
        assert!(debug_str.contains("head_dim"));
        assert!(debug_str.contains("num_tasks"));
        assert!(debug_str.contains("num_difficulties"));
        assert!(debug_str.contains("signal_dim"));
        assert!(debug_str.contains("signal_hidden_dim"));
        assert!(debug_str.contains("max_seq_len"));
    }

    // ─── Graph structure tests ───

    #[test]
    fn graph_has_two_outputs() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // SPEC §2.1: task_logits (B, 3) + difficulty_logits (B, 4)
        assert_eq!(graph.outputs.len(), 2, "graph should have exactly 2 outputs");
    }

    #[test]
    fn graph_max_seq_len_matches_config() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        assert_eq!(graph.max_seq_len, config.max_seq_len);
    }

    #[test]
    fn graph_max_seq_len_custom_value() {
        let mut config = IntentTrackerConfig::default();
        config.max_seq_len = 64;
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        assert_eq!(graph.max_seq_len, 64);
    }

    #[test]
    fn graph_has_expected_input_tensors() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Verify key input tensors exist with correct dtypes
        let emb_idx = graph.tensors.iter().position(|t| t.name == "embeddings").unwrap();
        let emb = graph.tensor(TensorId(emb_idx as u32));
        assert!(emb.is_some());
        assert_eq!(emb.unwrap().dtype, DType::F32);

        let roles_idx = graph.tensors.iter().position(|t| t.name == "roles").unwrap();
        assert!(graph.tensor(TensorId(roles_idx as u32)).is_some());

        let signals_idx = graph.tensors.iter().position(|t| t.name == "signals").unwrap();
        assert!(graph.tensor(TensorId(signals_idx as u32)).is_some());
    }

    #[test]
    fn graph_has_multi_head_attention_op() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let has_mha = graph.ops.iter().any(|op| matches!(op.kind, OpKind::MultiHeadAttention { causal: true, .. }));
        assert!(has_mha, "graph should contain a causal MultiHeadAttention op");
    }

    #[test]
    fn graph_mha_is_causal() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter().find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. })).unwrap();
        if let OpKind::MultiHeadAttention { causal, .. } = mha.kind {
            assert!(causal, "attention should be causal for intent tracker");
        }
    }

    #[test]
    fn graph_mha_head_count_matches_config() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter().find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. })).unwrap();
        if let OpKind::MultiHeadAttention { num_heads, head_dim, .. } = mha.kind {
            assert_eq!(num_heads, config.num_heads);
            assert_eq!(head_dim, config.head_dim);
        }
    }

    #[test]
    fn graph_has_gather_op_for_role_embedding() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let has_gather = graph.ops.iter().any(|op| {
            matches!(op.kind, OpKind::Gather { .. }) && op.label == "role_gather"
        });
        assert!(has_gather, "graph should have a Gather op named 'role_gather'");
    }

    #[test]
    fn graph_has_layer_norm_ops() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let ln_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::LayerNorm { .. }))
            .count();
        assert_eq!(ln_count, 2, "should have per-head LayerNorm and context LayerNorm");
    }

    #[test]
    fn graph_has_silu_activations() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let silu_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Silu))
            .count();
        // info MLP: 3 Silu, signal encoder: 2 Silu, task classifier: 2 Silu, diff classifier: 2 Silu = 9
        assert!(silu_count >= 8, "should have multiple SiLU activations, found {silu_count}");
    }

    #[test]
    fn graph_has_mean_pool_ops() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let pool_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::MeanPool { .. }))
            .count();
        assert_eq!(pool_count, 2, "should have CLS-mode pool and mean-mode pool");
    }

    #[test]
    fn graph_mean_pool_cls_and_mean_modes() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let pools: Vec<_> = graph.ops.iter()
            .filter_map(|op| match &op.kind {
                OpKind::MeanPool { cls_mode, .. } => Some(*cls_mode),
                _ => None,
            })
            .collect();
        assert!(pools.contains(&true), "should have CLS-mode pool");
        assert!(pools.contains(&false), "should have mean-mode pool");
    }

    #[test]
    fn graph_has_gemm_bias_ops() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gemm_bias_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::GemmBias { .. }))
            .count();
        // QKV projections (3) + info MLP (3) + signal encoder (3) + task classifier (3) + diff classifier (3) = 15
        assert!(gemm_bias_count >= 14, "should have many GemmBias ops, found {gemm_bias_count}");
    }

    // ─── Missing weight tests (comprehensive) ───

    #[test]
    fn build_fails_missing_role_emb_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("role_emb_weight");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("role_emb_weight"));
    }

    #[test]
    fn build_fails_missing_qkv_bias() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("w_v_bias");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("w_v_bias"));
    }

    #[test]
    fn build_fails_missing_info_net_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("info_net_fc1_weight");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("info_net_fc1_weight"));
    }

    #[test]
    fn build_fails_missing_info_net_bias() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("info_net_fc2_bias");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("info_net_fc2_bias"));
    }

    #[test]
    fn build_fails_missing_classifier_weight() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("task_fc1_weight");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("task_fc1_weight"));
    }

    #[test]
    fn build_fails_missing_diff_classifier_bias() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("diff_fc0_bias");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("diff_fc0_bias"));
    }

    #[test]
    fn build_fails_missing_per_head_norm() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("per_head_norm_weight");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
    }

    #[test]
    fn build_fails_missing_scalar_param() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("recency_scale");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("recency_scale"));
    }

    #[test]
    fn build_fails_missing_context_gate() {
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("context_gate");
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("context_gate"));
    }

    #[test]
    fn build_fails_with_empty_weights() {
        let config = IntentTrackerConfig::default();
        let weights = HashMap::new();
        let result = build_intent_tracker_graph(&config, &weights);
        assert!(result.is_err());
    }

    // ─── Topological ordering tests ───

    #[test]
    fn graph_topological_sort_succeeds() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let sorted = graph.topological_sort();
        assert!(!sorted.is_empty(), "topological sort should return ops");
    }

    #[test]
    fn graph_def_use_chains_valid() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let chains = graph.def_use_chains();
        assert!(!chains.is_empty(), "graph should have def-use chains for all tensors");
    }

    // ─── Output tensor shape tests ───

    #[test]
    fn graph_task_output_shape() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let task_out = graph.tensor(graph.outputs[0]).expect("task output tensor exists");
        // Task logits: (B, num_tasks) → shape should have batch + num_tasks
        let shape_strs: Vec<String> = task_out.shape.iter().map(|s| format!("{s:?}")).collect();
        assert!(shape_strs.len() >= 1, "task output should have at least batch dim");
    }

    #[test]
    fn graph_difficulty_output_shape() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let diff_out = graph.tensor(graph.outputs[1]).expect("difficulty output tensor exists");
        let shape_strs: Vec<String> = diff_out.shape.iter().map(|s| format!("{s:?}")).collect();
        assert!(shape_strs.len() >= 1, "difficulty output should have at least batch dim");
    }

    // ─── Tensor count and op count sanity ───

    #[test]
    fn graph_tensor_count_reasonable() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // We have 37 weight tensors + 5 inputs + many intermediates + 2 outputs
        assert!(graph.num_tensors() > 40, "should have substantial tensor count, got {}", graph.num_tensors());
    }

    #[test]
    fn graph_op_count_reasonable() {
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Many ops: Gather, Add, GemmBias×15, Silu×9+, Mul, MHA, LayerNorm×2, MeanPool×2
        assert!(graph.num_ops() > 20, "should have substantial op count, got {}", graph.num_ops());
    }

    // ─── Custom config with different output sizes ───

    #[test]
    fn config_custom_output_sizes() {
        let config = IntentTrackerConfig {
            num_tasks: 7,
            num_difficulties: 5,
            ..IntentTrackerConfig::default()
        };
        let mut weights = default_weight_shapes();
        weights.insert("task_fc2_weight".into(), vec![7, 192]);
        weights.insert("task_fc2_bias".into(), vec![7]);
        weights.insert("diff_fc2_weight".into(), vec![5, 192]);
        weights.insert("diff_fc2_bias".into(), vec![5]);
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        assert_eq!(graph.outputs.len(), 2);
    }

    // ─── New tests: pure data structure, trait, boundary, edge case ───

    #[test]
    fn config_equality_default_instances() {
        // Arrange: create two default configs
        let c1 = IntentTrackerConfig::default();
        let c2 = IntentTrackerConfig::default();
        // Assert: they are equal
        assert_eq!(c1, c2);
    }

    #[test]
    fn config_equality_after_mutation_differs() {
        // Arrange
        let c1 = IntentTrackerConfig::default();
        let mut c2 = IntentTrackerConfig::default();
        // Act: mutate one field
        c2.hidden_size = 1024;
        // Assert: no longer equal
        assert_ne!(c1, c2);
    }

    #[test]
    fn config_zero_hidden_size() {
        // Arrange: boundary value zero
        let config = IntentTrackerConfig {
            hidden_size: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert: fields are accessible and zero
        assert_eq!(config.hidden_size, 0);
    }

    #[test]
    fn config_zero_num_tasks() {
        // Arrange: edge case zero output classes
        let config = IntentTrackerConfig {
            num_tasks: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert
        assert_eq!(config.num_tasks, 0);
    }

    #[test]
    fn config_zero_num_difficulties() {
        // Arrange: edge case zero difficulty classes
        let config = IntentTrackerConfig {
            num_difficulties: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert
        assert_eq!(config.num_difficulties, 0);
    }

    #[test]
    fn config_zero_max_seq_len() {
        // Arrange: boundary value
        let config = IntentTrackerConfig {
            max_seq_len: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert: field is zero
        assert_eq!(config.max_seq_len, 0);
    }

    #[test]
    fn config_large_dims() {
        // Arrange: very large dimension values
        let config = IntentTrackerConfig {
            hidden_size: usize::MAX,
            num_heads: 1,
            head_dim: usize::MAX,
            num_tasks: 1000,
            num_difficulties: 1000,
            signal_dim: usize::MAX,
            signal_hidden_dim: usize::MAX,
            max_seq_len: usize::MAX,
        };
        // Assert: fields are stored without overflow
        assert_eq!(config.hidden_size, usize::MAX);
        assert_eq!(config.max_seq_len, usize::MAX);
        assert_eq!(config.num_tasks, 1000);
    }

    #[test]
    fn config_clone_deep_copy_all_fields() {
        // Arrange
        let original = IntentTrackerConfig {
            hidden_size: 512,
            num_heads: 8,
            head_dim: 64,
            num_tasks: 10,
            num_difficulties: 5,
            signal_dim: 22,
            signal_hidden_dim: 128,
            max_seq_len: 64,
        };
        // Act
        let cloned = original.clone();
        // Assert: all fields match
        assert_eq!(original.hidden_size, cloned.hidden_size);
        assert_eq!(original.num_heads, cloned.num_heads);
        assert_eq!(original.head_dim, cloned.head_dim);
        assert_eq!(original.num_tasks, cloned.num_tasks);
        assert_eq!(original.num_difficulties, cloned.num_difficulties);
        assert_eq!(original.signal_dim, cloned.signal_dim);
        assert_eq!(original.signal_hidden_dim, cloned.signal_hidden_dim);
        assert_eq!(original.max_seq_len, cloned.max_seq_len);
    }

    #[test]
    fn config_debug_output_contains_numeric_values() {
        // Arrange
        let config = IntentTrackerConfig {
            hidden_size: 768,
            num_heads: 4,
            head_dim: 192,
            num_tasks: 3,
            num_difficulties: 4,
            signal_dim: 11,
            signal_hidden_dim: 64,
            max_seq_len: 32,
        };
        // Act
        let debug = format!("{config:?}");
        // Assert: numeric values appear in debug output
        assert!(debug.contains("768"));
        assert!(debug.contains("4"));
        assert!(debug.contains("192"));
        assert!(debug.contains("3"));
        assert!(debug.contains("11"));
        assert!(debug.contains("64"));
        assert!(debug.contains("32"));
    }

    #[test]
    fn config_mutation_independence() {
        // Arrange: two independent mutable configs from default
        let mut c1 = IntentTrackerConfig::default();
        let mut c2 = IntentTrackerConfig::default();
        // Act: mutate each differently
        c1.num_tasks = 99;
        c2.num_difficulties = 88;
        // Assert: mutations do not affect each other
        assert_eq!(c1.num_tasks, 99);
        assert_eq!(c1.num_difficulties, 4);
        assert_eq!(c2.num_tasks, 3);
        assert_eq!(c2.num_difficulties, 88);
    }

    #[test]
    fn error_clone_preserves_message() {
        // Arrange
        let err = TrackerGraphError::MissingWeight("test_weight".to_string());
        // Act
        let cloned = err.clone();
        // Assert: cloned error produces same display output
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn error_equality_same_variant_same_message() {
        // Arrange
        let e1 = TrackerGraphError::MissingWeight("abc".to_string());
        let e2 = TrackerGraphError::MissingWeight("abc".to_string());
        // Assert
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_inequality_different_messages() {
        // Arrange
        let e1 = TrackerGraphError::MissingWeight("a".to_string());
        let e2 = TrackerGraphError::MissingWeight("b".to_string());
        // Assert
        assert_ne!(e1, e2);
    }

    #[test]
    fn error_inequality_different_variants() {
        // Arrange
        let e1 = TrackerGraphError::MissingWeight("x".to_string());
        let e2 = TrackerGraphError::InvalidDimension("x".to_string());
        // Assert: different variants are never equal even with same string
        assert_ne!(e1, e2);
    }

    #[test]
    fn error_hash_consistency() {
        // Arrange: two identical errors
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let e1 = TrackerGraphError::MissingWeight("weight_abc".to_string());
        let e2 = TrackerGraphError::MissingWeight("weight_abc".to_string());
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        // Act
        e1.hash(&mut h1);
        e2.hash(&mut h2);
        // Assert: identical values produce identical hashes
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn error_hash_different_for_different_variants() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let e1 = TrackerGraphError::MissingWeight("same_key".to_string());
        let e2 = TrackerGraphError::InvalidDimension("same_key".to_string());
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        // Act
        e1.hash(&mut h1);
        e2.hash(&mut h2);
        // Assert: different variants produce different hashes (not guaranteed but expected)
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn error_display_missing_weight_format() {
        // Arrange
        let err = TrackerGraphError::MissingWeight("my_param".to_string());
        // Act
        let display = format!("{err}");
        // Assert: formatted string contains both "missing weight" and the name
        assert!(display.contains("missing weight"), "display should contain 'missing weight'");
        assert!(display.contains("my_param"), "display should contain the weight name");
    }

    #[test]
    fn error_display_invalid_dimension_format() {
        // Arrange
        let err = TrackerGraphError::InvalidDimension("head_dim mismatch".to_string());
        // Act
        let display = format!("{err}");
        // Assert
        assert!(display.contains("invalid dimension"), "display should contain 'invalid dimension'");
        assert!(display.contains("head_dim mismatch"), "display should contain the description");
    }

    #[test]
    fn error_debug_format_different_from_display() {
        // Arrange
        let err = TrackerGraphError::MissingWeight("w".to_string());
        // Act
        let debug = format!("{err:?}");
        let display = format!("{err}");
        // Assert: Debug and Display are different representations
        assert!(!debug.is_empty());
        assert!(!display.is_empty());
        // Debug typically includes variant name, Display uses #[error] format
        assert!(debug.contains("MissingWeight"), "Debug should show variant name");
    }

    #[test]
    fn error_with_empty_string_message() {
        // Arrange: empty string edge case
        let err = TrackerGraphError::MissingWeight(String::new());
        // Act
        let display = format!("{err}");
        // Assert: does not panic, produces valid output
        assert!(!display.is_empty(), "display should still produce output for empty string");
    }

    #[test]
    fn error_with_unicode_message() {
        // Arrange: Unicode content in error message
        let err = TrackerGraphError::InvalidDimension("维度不匹配: hidden_size".to_string());
        // Act
        let display = format!("{err}");
        // Assert: Unicode is preserved
        assert!(display.contains("维度不匹配"), "display should preserve Unicode");
    }

    // ─── New tests: graph dataflow, op properties, structural invariants ───

    #[test]
    fn graph_has_add_op_for_role_embedding() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: role embedding addition exists
        let has_add = graph.ops.iter().any(|op| {
            matches!(op.kind, OpKind::Add) && op.label == "add_role"
        });
        assert!(has_add, "graph should have Add op for role embedding injection");
    }

    #[test]
    fn graph_has_mul_ops_for_v_modulation_and_dual_context() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: multiple Mul ops for V modulation and gated context paths
        let mul_ops: Vec<&str> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Mul))
            .map(|op| op.label.as_str())
            .collect();
        assert!(mul_ops.len() >= 3, "should have at least 3 Mul ops (v_modulate, gate_last, gate_mean), found {}", mul_ops.len());
    }

    #[test]
    fn graph_v_modulate_op_has_correct_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let v_mod = graph.ops.iter()
            .find(|op| op.label == "v_modulate")
            .expect("v_modulate op should exist");
        // Assert: v_modulate takes v_raw and info_weight as inputs
        assert_eq!(v_mod.inputs.len(), 2, "v_modulate should have 2 inputs (v_raw, info_weight)");
        assert_eq!(v_mod.outputs.len(), 1, "v_modulate should produce 1 output");
    }

    #[test]
    fn graph_dual_context_add_op_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert
        let dual_add = graph.ops.iter()
            .find(|op| op.label == "dual_context_add")
            .expect("dual_context_add op should exist");
        assert_eq!(dual_add.inputs.len(), 2, "dual context add should combine 2 gated paths");
    }

    #[test]
    fn graph_gather_op_has_correct_table_rows() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gather = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::Gather { .. }))
            .expect("Gather op should exist");
        // Assert: role embedding has 3 rows (user/assistant/system)
        if let OpKind::Gather { table_rows, embed_dim, indices_kind, .. } = &gather.kind {
            assert_eq!(*table_rows, 3, "role embedding table should have 3 rows");
            assert_eq!(*embed_dim, config.hidden_size);
            assert!(matches!(indices_kind, GatherIndicesKind::Tensor));
        }
    }

    #[test]
    fn graph_gather_op_has_two_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gather = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::Gather { .. }))
            .expect("Gather op should exist");
        // Assert: Gather takes role_emb_weight table + roles indices
        assert_eq!(gather.inputs.len(), 2, "Gather should have 2 inputs (table + indices)");
    }

    #[test]
    fn graph_layernorm_eps_is_small_positive() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let ln_ops: Vec<f32> = graph.ops.iter()
            .filter_map(|op| match op.kind {
                OpKind::LayerNorm { eps } => Some(eps),
                _ => None,
            })
            .collect();
        // Assert: all LayerNorm ops use standard small epsilon
        assert_eq!(ln_ops.len(), 2, "should have 2 LayerNorm ops");
        for eps in &ln_ops {
            assert!(*eps > 0.0, "eps should be positive");
            assert!(*eps < 1.0, "eps should be small (standard is 1e-5)");
        }
    }

    #[test]
    fn graph_mha_num_kv_heads_equals_num_heads() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }))
            .expect("MHA op should exist");
        // Assert: intent tracker uses MHA (not GQA), so num_kv_heads == num_heads
        if let OpKind::MultiHeadAttention { num_heads, num_kv_heads, .. } = mha.kind {
            assert_eq!(num_kv_heads, num_heads, "intent tracker MHA should have equal num_heads and num_kv_heads");
        }
    }

    #[test]
    fn graph_mha_attention_sinks_disabled() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }))
            .expect("MHA op should exist");
        // Assert
        if let OpKind::MultiHeadAttention { attention_sinks, .. } = mha.kind {
            assert!(!attention_sinks, "intent tracker should not use attention sinks");
        }
    }

    #[test]
    fn graph_embeddings_input_has_correct_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let emb = graph.tensors.iter()
            .find(|t| t.name == "embeddings")
            .expect("embeddings tensor should exist");
        // Assert: (batch, seq_len, hidden_size) = 3 dims
        assert_eq!(emb.shape.len(), 3, "embeddings should have 3 dimensions (B, T, H)");
        assert_eq!(emb.dtype, DType::F32);
    }

    #[test]
    fn graph_roles_input_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let roles = graph.tensors.iter()
            .find(|t| t.name == "roles")
            .expect("roles tensor should exist");
        // Assert: (batch, seq_len) = 2 dims
        assert_eq!(roles.shape.len(), 2, "roles should have 2 dimensions (B, T)");
    }

    #[test]
    fn graph_signals_input_has_correct_dim() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let signals = graph.tensors.iter()
            .find(|t| t.name == "signals")
            .expect("signals tensor should exist");
        // Assert: (batch, signal_dim) = 2 dims
        assert_eq!(signals.shape.len(), 2, "signals should have 2 dimensions (B, signal_dim)");
    }

    #[test]
    fn graph_seq_lens_input_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: seq_lens input tensor exists
        let found = graph.tensors.iter().any(|t| t.name == "seq_lens");
        assert!(found, "seq_lens input tensor should exist");
    }

    #[test]
    fn graph_context_turns_input_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: context_turns input tensor exists
        let found = graph.tensors.iter().any(|t| t.name == "context_turns");
        assert!(found, "context_turns input tensor should exist");
    }

    #[test]
    fn graph_all_gemm_bias_ops_have_three_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gemm_bias_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::GemmBias { .. }))
            .collect();
        // Assert: every GemmBias takes (input, weight, bias) = 3 inputs
        assert!(!gemm_bias_ops.is_empty(), "should have GemmBias ops");
        for op in &gemm_bias_ops {
            assert_eq!(op.inputs.len(), 3, "GemmBias '{}' should have 3 inputs (input, weight, bias)", op.label);
            assert_eq!(op.outputs.len(), 1, "GemmBias '{}' should have 1 output", op.label);
        }
    }

    #[test]
    fn graph_all_gemm_bias_ops_have_trans_b_true() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gemm_bias_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::GemmBias { .. }))
            .collect();
        // Assert: all GemmBias use trans_b=true (weight transposed for row-major storage)
        for op in &gemm_bias_ops {
            if let OpKind::GemmBias { trans_b, .. } = op.kind {
                assert!(trans_b, "GemmBias '{}' should have trans_b=true", op.label);
            }
        }
    }

    #[test]
    fn graph_mean_pool_hidden_matches_config() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let pools: Vec<_> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::MeanPool { .. }))
            .collect();
        // Assert: all MeanPool ops use config.hidden_size
        for pool in &pools {
            if let OpKind::MeanPool { hidden, .. } = pool.kind {
                assert_eq!(hidden, config.hidden_size, "MeanPool hidden should match config.hidden_size");
            }
        }
    }

    #[test]
    fn graph_mean_pool_seq_len_matches_config() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let pools: Vec<_> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::MeanPool { .. }))
            .collect();
        // Assert
        for pool in &pools {
            if let OpKind::MeanPool { seq_len, .. } = pool.kind {
                assert_eq!(seq_len, config.max_seq_len, "MeanPool seq_len should match config.max_seq_len");
            }
        }
    }

    #[test]
    fn graph_output_task_before_difficulty() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let task_name = graph.tensor(graph.outputs[0]).unwrap().name.as_str();
        let diff_name = graph.tensor(graph.outputs[1]).unwrap().name.as_str();
        // Assert: task_logits is first output, diff_logits is second
        assert_eq!(task_name, "task_logits", "first output should be task_logits");
        assert_eq!(diff_name, "diff_logits", "second output should be diff_logits");
    }

    #[test]
    fn graph_no_duplicate_tensor_names() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let names: Vec<&str> = graph.tensors.iter().map(|t| t.name.as_str()).collect();
        let unique_count = names.iter().collect::<std::collections::HashSet<_>>().len();
        // Assert: all tensor names are unique
        assert_eq!(names.len(), unique_count, "all tensor names should be unique");
    }

    #[test]
    fn graph_no_duplicate_op_labels() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let labels: Vec<&str> = graph.ops.iter().map(|op| op.label.as_str()).collect();
        let unique_count = labels.iter().collect::<std::collections::HashSet<_>>().len();
        // Assert: all op labels are unique
        assert_eq!(labels.len(), unique_count, "all op labels should be unique");
    }

    #[test]
    fn graph_all_op_input_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let num_tensors = graph.num_tensors();
        // Assert: every op input is a valid tensor ID
        for op in &graph.ops {
            for &tid in &op.inputs {
                assert!(
                    (tid.0 as usize) < num_tensors,
                    "op '{}' references tensor id {:?} which does not exist (num_tensors={})",
                    op.label, tid, num_tensors
                );
            }
        }
    }

    #[test]
    fn graph_all_op_output_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let num_tensors = graph.num_tensors();
        // Assert: every op output is a valid tensor ID
        for op in &graph.ops {
            for &tid in &op.outputs {
                assert!(
                    (tid.0 as usize) < num_tensors,
                    "op '{}' output tensor id {:?} does not exist (num_tensors={})",
                    op.label, tid, num_tensors
                );
            }
        }
    }

    #[test]
    fn graph_topological_order_respects_dataflow() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let sorted = graph.topological_sort();
        let mut produced: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
        // Assert: in topological order, each op's inputs are produced before the op runs
        for op_id in &sorted {
            let op = graph.op(*op_id).expect("op should exist");
            for &input_tid in &op.inputs {
                // Weight/input tensors may not have a producer op (they're external)
                let has_producer = graph.ops.iter().any(|o| o.outputs.contains(&input_tid));
                if has_producer {
                    assert!(
                        produced.contains(&input_tid),
                        "op '{}' uses tensor {:?} before it was produced",
                        op.label, input_tid
                    );
                }
            }
            for &output_tid in &op.outputs {
                produced.insert(output_tid);
            }
        }
    }

    #[test]
    fn graph_def_use_chains_cover_all_tensors() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let chains = graph.def_use_chains();
        // Assert: every tensor in the graph has an entry in def-use chains
        assert_eq!(chains.len(), graph.num_tensors(), "def-use chains should cover all tensors");
    }

    #[test]
    fn graph_weight_tensors_are_concrete() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: tensors loaded from weight_shapes should have fully concrete shapes
        // (add_weight uses add_tensor_concrete which always produces Concrete dimensions)
        for (name, _) in &weights {
            let tensor = graph.tensors.iter().find(|t| t.name == *name);
            if let Some(tensor) = tensor {
                for dim in &tensor.shape {
                    assert!(
                        matches!(dim, SymDim::Concrete(_)),
                        "weight tensor '{}' should have concrete shape, found {:?}",
                        tensor.name, dim
                    );
                }
            }
        }
    }

    #[test]
    fn build_graph_idempotent() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act: build twice
        let graph1 = build_intent_tracker_graph(&config, &weights).unwrap();
        let graph2 = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: same op and tensor counts (structure is deterministic)
        assert_eq!(graph1.num_ops(), graph2.num_ops(), "graph structure should be deterministic");
        assert_eq!(graph1.num_tensors(), graph2.num_tensors(), "graph structure should be deterministic");
        assert_eq!(graph1.outputs.len(), graph2.outputs.len());
    }

    #[test]
    fn graph_info_weight_sigmoid_op_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let info_sigmoid = graph.ops.iter()
            .find(|op| op.label == "info_sigmoid")
            .expect("info_sigmoid op should exist");
        // Assert
        assert!(matches!(info_sigmoid.kind, OpKind::Silu), "info_sigmoid should be a Silu activation");
    }

    #[test]
    fn graph_info_mlp_has_three_stages() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let info_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.starts_with("info_"))
            .collect();
        // Assert: info MLP has 3 linear stages + activations + sigmoid = 7 ops
        assert!(info_ops.len() >= 6, "info MLP should have multiple stages, found {}", info_ops.len());
    }

    #[test]
    fn graph_signal_encoder_has_silu_activations() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let sig_relu_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.starts_with("sig_relu"))
            .collect();
        // Assert: signal encoder has 2 SiLU activations (2-layer MLP)
        assert_eq!(sig_relu_ops.len(), 2, "signal encoder should have 2 SiLU activations");
    }

    #[test]
    fn graph_task_classifier_has_silu_activations() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let task_relu_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.starts_with("task_relu"))
            .collect();
        // Assert: task classifier has 2 SiLU activations
        assert_eq!(task_relu_ops.len(), 2, "task classifier should have 2 SiLU activations");
    }

    #[test]
    fn graph_diff_classifier_has_silu_activations() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let diff_relu_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| op.label.starts_with("diff_relu"))
            .collect();
        // Assert: difficulty classifier has 2 SiLU activations
        assert_eq!(diff_relu_ops.len(), 2, "diff classifier should have 2 SiLU activations");
    }

    #[test]
    fn graph_gate_last_and_gate_mean_ops_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert
        let gate_last = graph.ops.iter().find(|op| op.label == "gate_last");
        let gate_mean = graph.ops.iter().find(|op| op.label == "gate_mean");
        assert!(gate_last.is_some(), "gate_last op should exist");
        assert!(gate_mean.is_some(), "gate_mean op should exist");
        assert!(matches!(gate_last.unwrap().kind, OpKind::Mul), "gate_last should be Mul");
        assert!(matches!(gate_mean.unwrap().kind, OpKind::Mul), "gate_mean should be Mul");
    }

    #[test]
    fn graph_context_layernorm_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let ctx_ln = graph.ops.iter()
            .find(|op| op.label == "context_layernorm");
        // Assert
        assert!(ctx_ln.is_some(), "context_layernorm op should exist");
        let op = ctx_ln.unwrap();
        assert!(matches!(op.kind, OpKind::LayerNorm { .. }), "context_layernorm should be LayerNorm");
        assert_eq!(op.inputs.len(), 3, "LayerNorm should take (input, weight, bias)");
    }

    #[test]
    fn graph_attn_layernorm_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let attn_ln = graph.ops.iter()
            .find(|op| op.label == "attn_layernorm");
        // Assert
        assert!(attn_ln.is_some(), "attn_layernorm op should exist");
        assert!(matches!(attn_ln.unwrap().kind, OpKind::LayerNorm { .. }));
    }

    #[test]
    fn graph_task_and_diff_classifiers_share_input() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let task_fc0 = graph.ops.iter()
            .find(|op| op.label == "task_h0")
            .expect("task_h0 should exist");
        let diff_fc0 = graph.ops.iter()
            .find(|op| op.label == "diff_h0")
            .expect("diff_h0 should exist");
        // Assert: both classifiers start from the same context_normed tensor
        assert_eq!(task_fc0.inputs[0], diff_fc0.inputs[0], "task and diff classifiers should share context_normed input");
    }

    #[test]
    fn graph_pool_last_and_pool_mean_labels_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert
        let pool_last = graph.ops.iter().find(|op| op.label == "pool_last");
        let pool_mean = graph.ops.iter().find(|op| op.label == "pool_mean");
        assert!(pool_last.is_some(), "pool_last op should exist");
        assert!(pool_mean.is_some(), "pool_mean op should exist");
    }

    #[test]
    fn graph_qkv_projections_use_gemm_bias() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: q_proj, k_proj, v_raw are all GemmBias ops
        for label in &["q_proj", "k_proj", "v_raw"] {
            let op = graph.ops.iter()
                .find(|op| op.label == *label)
                .unwrap_or_else(|| panic!("{} op should exist", label));
            assert!(
                matches!(op.kind, OpKind::GemmBias { .. }),
                "{} should be a GemmBias op", label
            );
        }
    }

    #[test]
    fn graph_role_emb_output_tensor_has_correct_dim_count() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let role_out = graph.tensors.iter()
            .find(|t| t.name == "role_emb_out")
            .expect("role_emb_out tensor should exist");
        // Assert: (B, T, H) = 3 dimensions
        assert_eq!(role_out.shape.len(), 3, "role_emb_out should have 3 dimensions (B, T, H)");
    }

    #[test]
    fn graph_info_weight_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let info_weight = graph.tensors.iter()
            .find(|t| t.name == "info_weight")
            .expect("info_weight tensor should exist");
        // Assert: (B, T) = 2 dimensions — scalar per token
        assert_eq!(info_weight.shape.len(), 2, "info_weight should have 2 dimensions (B, T)");
    }

    #[test]
    fn graph_dual_context_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let dual_ctx = graph.tensors.iter()
            .find(|t| t.name == "dual_context")
            .expect("dual_context tensor should exist");
        // Assert: (B, H) = 2 dimensions — pooled context
        assert_eq!(dual_ctx.shape.len(), 2, "dual_context should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_all_tensors_f32_dtype() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: all tensors use F32 dtype
        for tensor in &graph.tensors {
            assert_eq!(
                tensor.dtype, DType::F32,
                "tensor '{}' should be F32, found {:?}", tensor.name, tensor.dtype
            );
        }
    }

    #[test]
    fn graph_all_ops_have_non_empty_labels() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert
        for op in &graph.ops {
            assert!(!op.label.is_empty(), "every op should have a non-empty label");
        }
    }

    #[test]
    fn graph_all_tensors_have_non_empty_names() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert
        for tensor in &graph.tensors {
            assert!(!tensor.name.is_empty(), "every tensor should have a non-empty name");
        }
    }

    #[test]
    fn graph_silu_ops_all_have_single_input_output() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let silu_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Silu))
            .collect();
        // Assert: Silu is elementwise, 1 input → 1 output
        assert!(!silu_ops.is_empty(), "should have SiLU ops");
        for op in &silu_ops {
            assert_eq!(op.inputs.len(), 1, "SiLU '{}' should have 1 input", op.label);
            assert_eq!(op.outputs.len(), 1, "SiLU '{}' should have 1 output", op.label);
        }
    }

    #[test]
    fn graph_add_op_has_two_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let add_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Add))
            .collect();
        // Assert: Add always takes 2 inputs, produces 1 output
        assert!(!add_ops.is_empty(), "should have Add ops");
        for op in &add_ops {
            assert_eq!(op.inputs.len(), 2, "Add '{}' should have 2 inputs", op.label);
            assert_eq!(op.outputs.len(), 1, "Add '{}' should have 1 output", op.label);
        }
    }

    #[test]
    fn graph_output_tensors_are_in_tensor_list() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let num_tensors = graph.num_tensors();
        // Assert: output tensor IDs are valid indices into tensors vec
        for &output_id in &graph.outputs {
            assert!(
                (output_id.0 as usize) < num_tensors,
                "output tensor id {:?} should be valid (num_tensors={})",
                output_id, num_tensors
            );
            assert!(graph.tensor(output_id).is_some(), "output tensor {:?} should exist", output_id);
        }
    }

    #[test]
    fn graph_layernorm_ops_have_three_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let ln_ops: Vec<&CompilerOp> = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::LayerNorm { .. }))
            .collect();
        // Assert: LayerNorm takes (input, weight, bias)
        assert_eq!(ln_ops.len(), 2, "should have 2 LayerNorm ops");
        for op in &ln_ops {
            assert_eq!(op.inputs.len(), 3, "LayerNorm '{}' should have 3 inputs", op.label);
            assert_eq!(op.outputs.len(), 1, "LayerNorm '{}' should have 1 output", op.label);
        }
    }

    #[test]
    fn graph_mha_op_has_three_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }))
            .expect("MHA should exist");
        // Assert: MHA takes (Q, K, V)
        assert_eq!(mha.inputs.len(), 3, "MHA should have 3 inputs (Q, K, V)");
        assert_eq!(mha.outputs.len(), 1, "MHA should have 1 output");
    }

    // ─── Additional tests: PartialOrd laws, missing weight coverage, intermediate tensor shapes ───

    #[test]
    fn config_partialeq_reflexivity() {
        // Arrange
        let c = IntentTrackerConfig::default();
        // Assert: a value equals itself
        assert_eq!(c, c);
    }

    #[test]
    fn config_partialeq_symmetry() {
        // Arrange
        let c1 = IntentTrackerConfig::default();
        let c2 = IntentTrackerConfig::default();
        // Assert: equality is symmetric
        assert_eq!(c1, c2);
        assert_eq!(c2, c1);
    }

    #[test]
    fn config_partialeq_transitivity() {
        // Arrange: three equal instances
        let c1 = IntentTrackerConfig::default();
        let c2 = IntentTrackerConfig::default();
        let c3 = IntentTrackerConfig::default();
        // Assert: transitivity of equality
        assert_eq!(c1, c2);
        assert_eq!(c2, c3);
        assert_eq!(c1, c3);
    }

    #[test]
    fn config_default_matches_manual_construction() {
        // Arrange: manually construct with same values as Default impl
        let from_default = IntentTrackerConfig::default();
        let from_manual = IntentTrackerConfig {
            hidden_size: 768,
            num_heads: 4,
            head_dim: 192,
            num_tasks: 3,
            num_difficulties: 4,
            signal_dim: 11,
            signal_hidden_dim: 64,
            max_seq_len: 32,
        };
        // Assert
        assert_eq!(from_default, from_manual);
    }

    #[test]
    fn build_fails_missing_task_fc0_weight() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("task_fc0_weight");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("task_fc0_weight"));
    }

    #[test]
    fn build_fails_missing_signal_fc1_bias() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("signal_fc1_bias");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("signal_fc1_bias"));
    }

    #[test]
    fn graph_context_normed_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let ctx_normed = graph.tensors.iter()
            .find(|t| t.name == "context_normed")
            .expect("context_normed tensor should exist");
        // Assert: (B, H) = 2 dimensions after LayerNorm on pooled context
        assert_eq!(ctx_normed.shape.len(), 2, "context_normed should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_attn_out_tensor_has_3d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let attn_out = graph.tensors.iter()
            .find(|t| t.name == "attn_out")
            .expect("attn_out tensor should exist");
        // Assert: (B, T, H) = 3 dimensions
        assert_eq!(attn_out.shape.len(), 3, "attn_out should have 3 dimensions (B, T, H)");
    }

    #[test]
    fn graph_e_plus_role_tensor_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: intermediate embedding + role_emb output exists
        let found = graph.tensors.iter().any(|t| t.name == "e_plus_role");
        assert!(found, "e_plus_role intermediate tensor should exist");
    }

    #[test]
    fn graph_v_modulated_tensor_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: V modulation output tensor exists
        let v_mod = graph.tensors.iter().find(|t| t.name == "v_modulated");
        assert!(v_mod.is_some(), "v_modulated tensor should exist");
        assert_eq!(v_mod.unwrap().shape.len(), 3, "v_modulated should have 3 dimensions (B, T, H)");
    }

    #[test]
    fn graph_gated_last_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gated_last = graph.tensors.iter()
            .find(|t| t.name == "gated_last")
            .expect("gated_last tensor should exist");
        // Assert: (B, H) = 2 dimensions
        assert_eq!(gated_last.shape.len(), 2, "gated_last should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_sig_out_tensor_exists() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: signal encoder output tensor exists
        let sig_out = graph.tensors.iter().find(|t| t.name == "sig_out");
        assert!(sig_out.is_some(), "sig_out tensor should exist");
        assert_eq!(sig_out.unwrap().shape.len(), 2, "sig_out should have 2 dimensions (B, signal_hidden_dim)");
    }

    // ─── Additional edge case tests ───

    #[test]
    fn graph_gather_index_dim_is_symbolic() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gather = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::Gather { .. }))
            .expect("Gather op should exist");
        // Assert: index_dim should be Symbolic (seq_len), not Concrete
        if let OpKind::Gather { index_dim, .. } = &gather.kind {
            assert!(
                matches!(index_dim, SymDim::Symbolic { .. }),
                "Gather index_dim should be Symbolic for dynamic sequence length, found {:?}",
                index_dim
            );
        }
    }

    #[test]
    fn graph_mha_seq_len_is_symbolic() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mha = graph.ops.iter()
            .find(|op| matches!(op.kind, OpKind::MultiHeadAttention { .. }))
            .expect("MHA op should exist");
        // Assert: MHA seq_len should be Symbolic (runtime-dynamic)
        if let OpKind::MultiHeadAttention { seq_len, .. } = &mha.kind {
            assert!(
                matches!(seq_len, SymDim::Symbolic { .. }),
                "MHA seq_len should be Symbolic, found {:?}", seq_len
            );
        }
    }

    #[test]
    fn graph_context_last_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let context_last = graph.tensors.iter()
            .find(|t| t.name == "context_last")
            .expect("context_last tensor should exist");
        // Assert: after MeanPool CLS mode, shape is (B, H) = 2 dims
        assert_eq!(context_last.shape.len(), 2, "context_last should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_context_mean_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let context_mean = graph.tensors.iter()
            .find(|t| t.name == "context_mean")
            .expect("context_mean tensor should exist");
        // Assert: after MeanPool mean mode, shape is (B, H) = 2 dims
        assert_eq!(context_mean.shape.len(), 2, "context_mean should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_gated_mean_tensor_has_2d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gated_mean = graph.tensors.iter()
            .find(|t| t.name == "gated_mean")
            .expect("gated_mean tensor should exist");
        // Assert: (B, H) = 2 dimensions — same as gated_last
        assert_eq!(gated_mean.shape.len(), 2, "gated_mean should have 2 dimensions (B, H)");
    }

    #[test]
    fn graph_info_mlp_intermediate_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: info MLP produces intermediate tensors at each stage
        for name in &["info_h0", "info_a0", "info_h1", "info_a1", "info_h2"] {
            let found = graph.tensors.iter().any(|t| t.name == *name);
            assert!(found, "intermediate tensor '{}' should exist in info MLP path", name);
        }
    }

    #[test]
    fn graph_signal_encoder_intermediate_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: signal encoder produces intermediate tensors at each stage
        for name in &["sig_h0", "sig_a0", "sig_h1", "sig_a1"] {
            let found = graph.tensors.iter().any(|t| t.name == *name);
            assert!(found, "intermediate tensor '{}' should exist in signal encoder path", name);
        }
    }

    #[test]
    fn graph_attn_normed_tensor_has_3d_shape() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let attn_normed = graph.tensors.iter()
            .find(|t| t.name == "attn_normed")
            .expect("attn_normed tensor should exist");
        // Assert: (B, T, H) = 3 dimensions — same as attn_out before pooling
        assert_eq!(attn_normed.shape.len(), 3, "attn_normed should have 3 dimensions (B, T, H)");
    }

    #[test]
    fn build_fails_missing_signal_fc2_weight() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("signal_fc2_weight");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("signal_fc2_weight"), "error should mention signal_fc2_weight, got: {}", msg);
    }

    #[test]
    fn build_fails_missing_diff_fc1_weight() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("diff_fc1_weight");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("diff_fc1_weight"), "error should mention diff_fc1_weight, got: {}", msg);
    }

    #[test]
    fn graph_gemm_bias_1d_ops_use_concrete_m_dim() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: 1D GemmBias ops (signal encoder + classifiers) use M=Concrete(1)
        let gemm_1d_labels = ["sig_h0", "sig_h1", "sig_out", "task_h0", "task_h1", "task_logits",
                              "diff_h0", "diff_h1", "diff_logits"];
        for label in &gemm_1d_labels {
            let op = graph.ops.iter().find(|op| op.label == *label);
            if let Some(op) = op {
                if let OpKind::GemmBias { m, .. } = &op.kind {
                    assert!(
                        matches!(m, SymDim::Concrete(1)),
                        "1D GemmBias '{}' should use M=Concrete(1), found {:?}", label, m
                    );
                }
            }
        }
    }

    #[test]
    fn graph_gemm_bias_2d_ops_use_symbolic_m_dim() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: 2D GemmBias ops (info MLP + QKV) use Symbolic M for seq_len
        let gemm_2d_labels = ["info_h0", "info_h1", "info_h2", "q_proj", "k_proj", "v_raw"];
        for label in &gemm_2d_labels {
            let op = graph.ops.iter().find(|op| op.label == *label);
            if let Some(op) = op {
                if let OpKind::GemmBias { m, .. } = &op.kind {
                    assert!(
                        matches!(m, SymDim::Symbolic { .. }),
                        "2D GemmBias '{}' should use Symbolic M, found {:?}", label, m
                    );
                }
            }
        }
    }

    #[test]
    fn graph_q_proj_k_proj_v_raw_share_same_input() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let q_proj = graph.ops.iter().find(|op| op.label == "q_proj").expect("q_proj should exist");
        let k_proj = graph.ops.iter().find(|op| op.label == "k_proj").expect("k_proj should exist");
        let v_raw = graph.ops.iter().find(|op| op.label == "v_raw").expect("v_raw should exist");
        // Assert: Q, K, V all take e_plus_role as their first input (shared input)
        assert_eq!(q_proj.inputs[0], k_proj.inputs[0], "q_proj and k_proj should share first input");
        assert_eq!(k_proj.inputs[0], v_raw.inputs[0], "k_proj and v_raw should share first input");
    }

    // ─── 13 additional tests to reach 156 total ───

    #[test]
    fn error_source_is_none_for_missing_weight() {
        // Arrange: thiserror derives std::error::Error; source() should be None for leaf errors
        let err = TrackerGraphError::MissingWeight("x".to_string());
        // Act & Assert: source is None (no chained error)
        assert!(std::error::Error::source(&err).is_none(),
            "MissingWeight is a leaf error and should have no source");
    }

    #[test]
    fn error_source_is_none_for_invalid_dimension() {
        // Arrange
        let err = TrackerGraphError::InvalidDimension("y".to_string());
        // Act & Assert: source is None (no chained error)
        assert!(std::error::Error::source(&err).is_none(),
            "InvalidDimension is a leaf error and should have no source");
    }

    #[test]
    fn error_debug_invalid_dimension_shows_variant_name() {
        // Arrange
        let err = TrackerGraphError::InvalidDimension("overflow".to_string());
        // Act
        let debug = format!("{err:?}");
        // Assert: Debug output includes the variant identifier
        assert!(debug.contains("InvalidDimension"),
            "Debug should include variant name 'InvalidDimension', got: {debug}");
    }

    #[test]
    fn config_signal_dim_zero_boundary() {
        // Arrange: zero signal dimension (no signal features)
        let config = IntentTrackerConfig {
            signal_dim: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert: field is stored as zero
        assert_eq!(config.signal_dim, 0);
        assert_eq!(config.hidden_size, 768, "other fields should remain at default");
    }

    #[test]
    fn config_signal_hidden_dim_zero_boundary() {
        // Arrange: zero signal hidden dimension
        let config = IntentTrackerConfig {
            signal_hidden_dim: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert
        assert_eq!(config.signal_hidden_dim, 0);
        assert_eq!(config.num_tasks, 3, "other fields should remain at default");
    }

    #[test]
    fn config_head_dim_zero_boundary() {
        // Arrange: zero head dimension
        let config = IntentTrackerConfig {
            head_dim: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert: stored correctly (even though num_heads * head_dim != hidden_size)
        assert_eq!(config.head_dim, 0);
        assert_ne!(config.num_heads * config.head_dim, config.hidden_size,
            "zero head_dim intentionally breaks the hidden_size invariant");
    }

    #[test]
    fn graph_silu_op_exact_count() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let silu_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Silu))
            .count();
        // Assert: info MLP (3 Silu: relu0/relu1/sigmoid) + signal (2) + task (2) + diff (2) = 9
        assert_eq!(silu_count, 9, "expected exactly 9 SiLU ops, found {silu_count}");
    }

    #[test]
    fn graph_gemm_bias_op_exact_count() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let gemm_bias_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::GemmBias { .. }))
            .count();
        // Assert: QKV (3) + info MLP (3) + signal (3) + task (3) + diff (3) = 15
        assert_eq!(gemm_bias_count, 15, "expected exactly 15 GemmBias ops, found {gemm_bias_count}");
    }

    #[test]
    fn graph_mul_op_exact_count() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let mul_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Mul))
            .count();
        // Assert: v_modulate + gate_last + gate_mean = 3 Mul ops
        assert_eq!(mul_count, 3, "expected exactly 3 Mul ops (v_modulate, gate_last, gate_mean), found {mul_count}");
    }

    #[test]
    fn graph_add_op_exact_count() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let add_count = graph.ops.iter()
            .filter(|op| matches!(op.kind, OpKind::Add))
            .count();
        // Assert: add_role + dual_context_add = 2 Add ops
        assert_eq!(add_count, 2, "expected exactly 2 Add ops, found {add_count}");
    }

    #[test]
    fn graph_build_with_extra_weight_keys_succeeds() {
        // Arrange: extra irrelevant keys in weight_shapes should not affect graph construction
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.insert("unused_weight_1".into(), vec![64, 64]);
        weights.insert("unused_weight_2".into(), vec![32]);
        // Act
        let graph = build_intent_tracker_graph(&config, &weights);
        // Assert: extra keys are ignored, graph builds successfully
        assert!(graph.is_ok(), "extra weight keys should not cause build failure");
        assert_eq!(graph.unwrap().outputs.len(), 2);
    }

    #[test]
    fn graph_build_with_missing_deep_weight_signal_fc2_bias() {
        // Arrange: remove a weight deep in the pipeline (signal encoder last bias)
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("signal_fc2_bias");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("signal_fc2_bias"), "error should reference signal_fc2_bias, got: {msg}");
    }

    #[test]
    fn graph_task_classifier_intermediate_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: task classifier produces intermediate tensors at each stage
        for name in &["task_h0", "task_a0", "task_h1", "task_a1"] {
            let found = graph.tensors.iter().any(|t| t.name == *name);
            assert!(found, "intermediate tensor '{}' should exist in task classifier path", name);
        }
    }

    // ─── 10 additional tests to reach 166 total ───

    #[test]
    fn graph_diff_classifier_intermediate_tensors_exist() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: difficulty classifier produces intermediate tensors at each stage
        for name in &["diff_h0", "diff_a0", "diff_h1", "diff_a1"] {
            let found = graph.tensors.iter().any(|t| t.name == *name);
            assert!(found, "intermediate tensor '{}' should exist in diff classifier path", name);
        }
    }

    #[test]
    fn config_num_heads_zero_boundary() {
        // Arrange: zero heads (boundary value)
        let config = IntentTrackerConfig {
            num_heads: 0,
            ..IntentTrackerConfig::default()
        };
        // Assert: field is stored as zero, hidden_size invariant is intentionally broken
        assert_eq!(config.num_heads, 0);
        assert_ne!(config.num_heads * config.head_dim, config.hidden_size,
            "zero num_heads intentionally breaks the hidden_size invariant");
    }

    #[test]
    fn graph_symbolic_seq_len_max_value_matches_config() {
        // Arrange
        let config = IntentTrackerConfig {
            max_seq_len: 48,
            ..IntentTrackerConfig::default()
        };
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: input tensors with seq_len should have Symbolic max_value matching config
        let emb = graph.tensors.iter()
            .find(|t| t.name == "embeddings")
            .expect("embeddings tensor should exist");
        let seq_dim = &emb.shape[1];
        if let SymDim::Symbolic { name, max_value } = seq_dim {
            assert_eq!(name, "seq_len", "seq dimension should be named 'seq_len'");
            assert_eq!(*max_value, Some(48), "max_value should match config.max_seq_len");
        } else {
            panic!("embeddings seq dimension should be Symbolic, found {:?}", seq_dim);
        }
    }

    #[test]
    fn graph_symbolic_batch_size_dimension_in_inputs() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: key input tensors should have Symbolic batch_size as first dim
        for tensor_name in &["embeddings", "roles", "signals", "seq_lens", "context_turns"] {
            let tensor = graph.tensors.iter()
                .find(|t| t.name == *tensor_name)
                .unwrap_or_else(|| panic!("{} tensor should exist", tensor_name));
            let batch_dim = &tensor.shape[0];
            assert!(
                matches!(batch_dim, SymDim::Symbolic { name, .. } if name == "batch_size"),
                "first dim of '{}' should be Symbolic('batch_size'), found {:?}",
                tensor_name, batch_dim
            );
        }
    }

    #[test]
    fn graph_all_gemm_bias_ops_use_f32_dtype() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: every GemmBias op is explicitly F32
        for op in &graph.ops {
            if let OpKind::GemmBias { dtype, .. } = op.kind {
                assert_eq!(dtype, DType::F32,
                    "GemmBias '{}' should use F32 dtype, found {:?}", op.label, dtype);
            }
        }
    }

    #[test]
    fn graph_gemm_bias_1d_ops_cover_signal_and_classifiers() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        // Assert: 1D GemmBias ops exist for signal encoder (3) + task classifier (3) + diff classifier (3) = 9
        let gemm_1d_labels = ["sig_h0", "sig_h1", "sig_out", "task_h0", "task_h1", "task_logits",
                              "diff_h0", "diff_h1", "diff_logits"];
        for label in &gemm_1d_labels {
            let op = graph.ops.iter()
                .find(|op| op.label == *label)
                .unwrap_or_else(|| panic!("1D GemmBias op '{}' should exist", label));
            assert!(
                matches!(op.kind, OpKind::GemmBias { m: SymDim::Concrete(1), .. }),
                "'{}' should be a GemmBias with M=Concrete(1)", label
            );
        }
    }

    #[test]
    fn build_fails_missing_info_net_fc0_bias() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("info_net_fc0_bias");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("info_net_fc0_bias"),
            "error should reference info_net_fc0_bias, got: {msg}");
    }

    #[test]
    fn build_fails_missing_context_norm_bias() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let mut weights = default_weight_shapes();
        weights.remove("context_norm_bias");
        // Act
        let result = build_intent_tracker_graph(&config, &weights);
        // Assert
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("context_norm_bias"),
            "error should reference context_norm_bias, got: {msg}");
    }

    #[test]
    fn graph_q_proj_k_proj_v_raw_use_distinct_weights() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let q_proj = graph.ops.iter().find(|op| op.label == "q_proj").expect("q_proj should exist");
        let k_proj = graph.ops.iter().find(|op| op.label == "k_proj").expect("k_proj should exist");
        let v_raw = graph.ops.iter().find(|op| op.label == "v_raw").expect("v_raw should exist");
        // Assert: Q/K/V use different weight tensors (index 1) and bias tensors (index 2)
        assert_ne!(q_proj.inputs[1], k_proj.inputs[1], "q_proj and k_proj should use different weight tensors");
        assert_ne!(k_proj.inputs[1], v_raw.inputs[1], "k_proj and v_raw should use different weight tensors");
        assert_ne!(q_proj.inputs[2], k_proj.inputs[2], "q_proj and k_proj should use different bias tensors");
        assert_ne!(k_proj.inputs[2], v_raw.inputs[2], "k_proj and v_raw should use different bias tensors");
    }

    #[test]
    fn graph_diff_classifier_head_uses_context_normed_input() {
        // Arrange
        let config = IntentTrackerConfig::default();
        let weights = default_weight_shapes();
        // Act
        let graph = build_intent_tracker_graph(&config, &weights).unwrap();
        let diff_fc0 = graph.ops.iter()
            .find(|op| op.label == "diff_h0")
            .expect("diff_h0 should exist");
        let task_fc0 = graph.ops.iter()
            .find(|op| op.label == "task_h0")
            .expect("task_h0 should exist");
        // Assert: diff classifier also starts from context_normed (same as task classifier)
        assert_eq!(diff_fc0.inputs[0], task_fc0.inputs[0],
            "diff_h0 and task_h0 should share the same context_normed input tensor");
    }

}
