//! c1 v2 DialogueGate Intent Tracker 专用图构建器
//!
//! 7-intent 对话状态追踪模型 (v2_granite_best.pt, 47 keys)。
//! 架构 (c1/src/intent/model/dialogue_gate.py + scorer.py):
//! - encoder: granite-embedding-311m (ModernBERT, 冻结, 外部预编码 → e_t(768))
//! - tracker: 3层 stacked DialogueGateCell (跨turn状态递推)
//! - heads:   IntentDifficultyHead (label_queries + global MLP + CORN difficulty)
//!
//! DialogueGateCell 递推 (DA-Mamba §3.2-3.3):
//!   x_int = W_int_x(x); h_int = W_int_h(h); interact = x_int * h_int
//!   consistency = σ(interact)
//!   α_prior = σ(bias_α); α_dynamic = σ(W_α(x) + consistency ⊙ U_α(h))
//!   α = α_prior ⊙ α_dynamic
//!   gate = σ(W_i(x) + U_i(h) + interact)
//!   candidate = tanh(W_c(x) + U_c(h))
//!   h' = α * h + gate * candidate
//!
//! 本图是**单步状态递推**: 输入 (e_t, h_prev[3]) → 输出 (h_next[3], intent_logits[7], diff_logits[3])
//! 调用者管理跨turn的 h_prev/h_next (stateful runtime API)。

use std::collections::HashMap;

use gllm_kernels::compiler::graph::{CompilerGraph, GemmSpec, NormSpec, SymDim, TensorId};
use gllm_kernels::compiler::Op;
use gllm_kernels::types::DType;

/// c1 v2 模型超参数 (config["model"])
#[derive(Debug, Clone, PartialEq)]
pub struct C1V2Config {
    pub hidden_dim: usize,
    pub num_cell_layers: usize,
    pub num_intents: usize,
    pub num_difficulty: usize, // CORN: K-1 (4 类 → 3 输出)
    pub intent_hidden_dim: usize,
    pub difficulty_hidden_dim: usize,
}

impl Default for C1V2Config {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_cell_layers: 3,
            num_intents: 7,
            num_difficulty: 3,
            intent_hidden_dim: 384,
            difficulty_hidden_dim: 384,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Hash, thiserror::Error)]
pub enum C1V2GraphError {
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("invalid dimension for {name}: expected {expected:?}, got {actual:?}")]
    InvalidDimension { name: String, expected: Vec<usize>, actual: Vec<usize> },
}

/// 构建 c1 v2 DialogueGate 单步递推图 (REQ-C1-002)。
///
/// 输入 tensors:
///   - `turn_embed`: (B, H) 当前 turn 的 encoder 输出 (768)
///   - `h_prev_{i}`: (B, H) 第 i 层 cell 的前一步状态 (i=0..num_cell_layers-1)
///
/// 输出 tensors:
///   - `intent_logits`: (B, num_intents)
///   - `diff_logits`:   (B, num_difficulty)
///   - `h_next_{i}`:    (B, H) 第 i 层 cell 的新状态
///
/// 权重命名映射 (c1 .pt tensor name → 本图规范名, 详见 C1V2Tracker::graph_weight_shapes):
///   tracker.stack.cells.{i}.W_int_x.weight → cell_{i}_w_int_x
///   tracker.stack.cells.{i}.bias_α         → cell_{i}_bias_alpha
///   heads.label_queries  → heads_label_queries
///   heads.global_w2.weight → heads_global_w2
pub fn build_c1_v2_graph(
    config: &C1V2Config,
    weight_shapes: &HashMap<String, Vec<usize>>,
) -> Result<CompilerGraph, C1V2GraphError> {
    let mut g = CompilerGraph::new();

    let b = SymDim::Symbolic { name: "batch_size".to_string(), max_value: Some(1) };
    let h = config.hidden_dim;
    let dt = DType::F32;
    let h_sym = SymDim::Concrete(h);

    // ── Input tensors ──
    let turn_embed = g.add_tensor("turn_embed", vec![b.clone(), h_sym.clone()], dt);
    let mut h_prev: Vec<TensorId> = Vec::with_capacity(config.num_cell_layers);
    for i in 0..config.num_cell_layers {
        let tid = g.add_tensor(format!("h_prev_{i}").as_str(), vec![b.clone(), h_sym.clone()], dt);
        h_prev.push(tid);
    }

    // ── 3-layer stacked DialogueGateCell ──
    let mut h_next: Vec<TensorId> = Vec::with_capacity(config.num_cell_layers);
    let mut cur_input = turn_embed;

    for i in 0..config.num_cell_layers {
        let w_int_x = add_weight(&mut g, weight_shapes, &format!("cell_{i}_w_int_x"))?;
        let w_int_h = add_weight(&mut g, weight_shapes, &format!("cell_{i}_w_int_h"))?;
        let w_alpha = add_weight(&mut g, weight_shapes, &format!("cell_{i}_w_alpha"))?;
        let u_alpha = add_weight(&mut g, weight_shapes, &format!("cell_{i}_u_alpha"))?;
        let w_i = add_weight(&mut g, weight_shapes, &format!("cell_{i}_w_i"))?;
        let u_i = add_weight(&mut g, weight_shapes, &format!("cell_{i}_u_i"))?;
        let w_c = add_weight(&mut g, weight_shapes, &format!("cell_{i}_w_c"))?;
        let u_c = add_weight(&mut g, weight_shapes, &format!("cell_{i}_u_c"))?;
        let bias_alpha = add_weight(&mut g, weight_shapes, &format!("cell_{i}_bias_alpha"))?;
        let prev = h_prev[i];

        // x_int = W_int_x(x); h_int = W_int_h(h_prev); interact = x_int * h_int
        let x_int = add_gemm_no_bias(&mut g, format!("cell_{i}_x_int").as_str(), cur_input, w_int_x, b.clone(), h, h, dt);
        let h_int = add_gemm_no_bias(&mut g, format!("cell_{i}_h_int").as_str(), prev, w_int_h, b.clone(), h, h, dt);
        let interact = g.add_tensor(format!("cell_{i}_interact").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Mul, vec![x_int, h_int], vec![interact], format!("cell_{i}_interact_mul").as_str());

        // consistency = σ(interact); α_prior = σ(bias_α)
        let consistency = g.add_tensor(format!("cell_{i}_consistency").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Sigmoid, vec![interact], vec![consistency], format!("cell_{i}_consistency_sig").as_str());
        let alpha_prior = g.add_tensor(format!("cell_{i}_alpha_prior").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Sigmoid, vec![bias_alpha], vec![alpha_prior], format!("cell_{i}_alpha_prior_sig").as_str());

        // α_dynamic = σ(W_α(x) + consistency ⊙ U_α(h))
        let w_alpha_x = add_gemm_no_bias(&mut g, format!("cell_{i}_w_alpha_x").as_str(), cur_input, w_alpha, b.clone(), h, h, dt);
        let u_alpha_h = add_gemm_no_bias(&mut g, format!("cell_{i}_u_alpha_h").as_str(), prev, u_alpha, b.clone(), h, h, dt);
        let cons_u_alpha = g.add_tensor(format!("cell_{i}_cons_u_alpha").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Mul, vec![consistency, u_alpha_h], vec![cons_u_alpha], format!("cell_{i}_cons_u_alpha_mul").as_str());
        let alpha_dyn_logit = g.add_tensor(format!("cell_{i}_alpha_dyn_logit").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Add, vec![w_alpha_x, cons_u_alpha], vec![alpha_dyn_logit], format!("cell_{i}_alpha_dyn_add").as_str());
        let alpha_dyn = g.add_tensor(format!("cell_{i}_alpha_dyn").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Sigmoid, vec![alpha_dyn_logit], vec![alpha_dyn], format!("cell_{i}_alpha_dyn_sig").as_str());

        // α = α_prior ⊙ α_dynamic
        let alpha = g.add_tensor(format!("cell_{i}_alpha").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Mul, vec![alpha_prior, alpha_dyn], vec![alpha], format!("cell_{i}_alpha_mul").as_str());

        // gate = σ(W_i(x) + U_i(h) + interact)
        let w_i_x = add_gemm_no_bias(&mut g, format!("cell_{i}_w_i_x").as_str(), cur_input, w_i, b.clone(), h, h, dt);
        let u_i_h = add_gemm_no_bias(&mut g, format!("cell_{i}_u_i_h").as_str(), prev, u_i, b.clone(), h, h, dt);
        let gate_sum1 = g.add_tensor(format!("cell_{i}_gate_sum1").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Add, vec![w_i_x, u_i_h], vec![gate_sum1], format!("cell_{i}_gate_add1").as_str());
        let gate_logit = g.add_tensor(format!("cell_{i}_gate_logit").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Add, vec![gate_sum1, interact], vec![gate_logit], format!("cell_{i}_gate_add2").as_str());
        let gate = g.add_tensor(format!("cell_{i}_gate").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Sigmoid, vec![gate_logit], vec![gate], format!("cell_{i}_gate_sig").as_str());

        // candidate = tanh(W_c(x) + U_c(h))
        let w_c_x = add_gemm_no_bias(&mut g, format!("cell_{i}_w_c_x").as_str(), cur_input, w_c, b.clone(), h, h, dt);
        let u_c_h = add_gemm_no_bias(&mut g, format!("cell_{i}_u_c_h").as_str(), prev, u_c, b.clone(), h, h, dt);
        let cand_logit = g.add_tensor(format!("cell_{i}_cand_logit").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Add, vec![w_c_x, u_c_h], vec![cand_logit], format!("cell_{i}_cand_add").as_str());
        let candidate = g.add_tensor(format!("cell_{i}_candidate").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Tanh, vec![cand_logit], vec![candidate], format!("cell_{i}_cand_tanh").as_str());

        // h' = α * h + gate * candidate
        let alpha_h = g.add_tensor(format!("cell_{i}_alpha_h").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Mul, vec![alpha, prev], vec![alpha_h], format!("cell_{i}_alpha_h_mul").as_str());
        let gate_cand = g.add_tensor(format!("cell_{i}_gate_cand").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Mul, vec![gate, candidate], vec![gate_cand], format!("cell_{i}_gate_cand_mul").as_str());
        let h_out = g.add_tensor(format!("h_next_{i}").as_str(), vec![b.clone(), h_sym.clone()], dt);
        g.add_op(Op::Add, vec![alpha_h, gate_cand], vec![h_out], format!("cell_{i}_h_out_add").as_str());

        h_next.push(h_out);
        cur_input = h_out;
    }

    // ── IntentDifficultyHead (state readout from final cell layer) ──
    let state = cur_input;
    let label_queries = add_weight(&mut g, weight_shapes, "heads_label_queries")?;
    let input_norm = add_weight(&mut g, weight_shapes, "heads_input_norm")?;
    let context_proj = add_weight(&mut g, weight_shapes, "heads_context_proj")?;
    let label_proj = add_weight(&mut g, weight_shapes, "heads_label_proj")?;
    let global_w1 = add_weight(&mut g, weight_shapes, "heads_global_w1")?;
    let global_w2 = add_weight(&mut g, weight_shapes, "heads_global_w2")?;
    let diff_w2 = add_weight(&mut g, weight_shapes, "heads_diff_w2")?;

    // normed = RMSNorm(state)
    let normed = g.add_tensor("heads_normed", vec![b.clone(), h_sym.clone()], dt);
    g.add_op(
        Op::RmsNorm(NormSpec { feature_dim: h, eps: 1e-6, dtype: dt, has_weight: true }),
        vec![state, input_norm],
        vec![normed],
        "heads_rmsnorm",
    );

    // context = context_proj(normed)
    let context = add_gemm_no_bias(&mut g, "heads_context", normed, context_proj, b.clone(), h, h, dt);

    // queries = label_proj(label_queries)  (L,H)@(H,H)^T → (L,H)
    let queries = g.add_tensor("heads_queries", vec![SymDim::Concrete(config.num_intents), h_sym.clone()], dt);
    g.add_op(
        Op::Gemm(GemmSpec { m: SymDim::Concrete(config.num_intents), n: h, k: h, dtype: dt, trans_b: true, has_bias: false }),
        vec![label_queries, label_proj],
        vec![queries],
        "heads_label_proj",
    );

    // label_scores = (context @ queries^T) * (1/sqrt(H))
    let scale = 1.0_f32 / (h as f32).sqrt();
    let label_scores_raw = g.add_tensor("heads_label_scores_raw", vec![b.clone(), SymDim::Concrete(config.num_intents)], dt);
    g.add_op(
        Op::Gemm(GemmSpec { m: b.clone(), n: config.num_intents, k: h, dtype: dt, trans_b: true, has_bias: false }),
        vec![context, queries],
        vec![label_scores_raw],
        "heads_label_scores",
    );
    let label_scores = g.add_tensor("heads_label_scores", vec![b.clone(), SymDim::Concrete(config.num_intents)], dt);
    g.add_op(Op::ScaleConst { value: scale }, vec![label_scores_raw], vec![label_scores], "heads_label_scale");

    // global_scores = global_w2(silu(global_w1(normed)))
    let global_h = add_gemm_no_bias(&mut g, "heads_global_h", normed, global_w1, b.clone(), config.intent_hidden_dim, h, dt);
    let global_a = g.add_tensor("heads_global_a", vec![b.clone(), SymDim::Concrete(config.intent_hidden_dim)], dt);
    g.add_op(Op::Silu, vec![global_h], vec![global_a], "heads_global_silu");
    let global_scores = add_gemm_no_bias(&mut g, "heads_global_scores", global_a, global_w2, b.clone(), config.num_intents, config.intent_hidden_dim, dt);

    // intent_logits = label_scores + global_scores
    let intent_logits = g.add_tensor("intent_logits", vec![b.clone(), SymDim::Concrete(config.num_intents)], dt);
    g.add_op(Op::Add, vec![label_scores, global_scores], vec![intent_logits], "heads_intent_add");

    // ── Difficulty head (CORN) — graph uses state-only path (runtime does real concat) ──
    // Per scorer.py: diff_input = cat([state, intent_summary], -1), RMSNorm, diff_w1, silu, diff_w2.
    // Graph IR has no Op::Concat (3-task intent_tracker_graph likewise omits its concat);
    // the runtime forward (C1V2Tracker::heads_forward) performs the real concat in Rust.
    // The *verification graph* feeds state into a (B,H)-shaped RMSNorm/diff_w1 path.
    let diff_norm_graph = g.add_tensor("heads_diff_norm_graph", vec![SymDim::Concrete(h)], dt);
    let diff_normed = g.add_tensor("heads_diff_normed", vec![b.clone(), SymDim::Concrete(h)], dt);
    g.add_op(
        Op::RmsNorm(NormSpec { feature_dim: h, eps: 1e-6, dtype: dt, has_weight: true }),
        vec![state, diff_norm_graph],
        vec![diff_normed],
        "heads_diff_rmsnorm",
    );
    let diff_w1_graph = g.add_tensor("heads_diff_w1_graph", vec![SymDim::Concrete(config.difficulty_hidden_dim), h_sym.clone()], dt);
    let diff_h = add_gemm_no_bias(&mut g, "heads_diff_h", diff_normed, diff_w1_graph, b.clone(), config.difficulty_hidden_dim, h, dt);
    let diff_a = g.add_tensor("heads_diff_a", vec![b.clone(), SymDim::Concrete(config.difficulty_hidden_dim)], dt);
    g.add_op(Op::Silu, vec![diff_h], vec![diff_a], "heads_diff_silu");
    let diff_logits = add_gemm_no_bias(&mut g, "diff_logits", diff_a, diff_w2, b.clone(), config.num_difficulty, config.difficulty_hidden_dim, dt);

    // ── Outputs ──
    g.outputs.push(intent_logits);
    g.outputs.push(diff_logits);
    for tid in &h_next {
        g.outputs.push(*tid);
    }
    g.max_seq_len = 1;

    Ok(g)
}

// ── Helpers ──

fn add_weight(
    g: &mut CompilerGraph,
    shapes: &HashMap<String, Vec<usize>>,
    name: &str,
) -> Result<TensorId, C1V2GraphError> {
    let shape = shapes
        .get(name)
        .ok_or_else(|| C1V2GraphError::MissingWeight(name.to_string()))?
        .clone();
    Ok(g.add_tensor_concrete(name, &shape, DType::F32))
}

/// Gemm without bias: out = input @ weight^T. input (B, K), weight (N, K) → (B, N).
fn add_gemm_no_bias(
    g: &mut CompilerGraph,
    name: &str,
    input: TensorId,
    weight: TensorId,
    b: SymDim,
    n: usize,
    k: usize,
    dtype: DType,
) -> TensorId {
    let out = g.add_tensor(name, vec![b.clone(), SymDim::Concrete(n)], dtype);
    g.add_op(
        Op::Gemm(GemmSpec { m: b, n, k, dtype, trans_b: true, has_bias: false }),
        vec![input, weight],
        vec![out],
        name,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_weight_shapes() -> HashMap<String, Vec<usize>> {
        let mut m = HashMap::new();
        let h = 768;
        for i in 0..3 {
            for (suf, shp) in [
                ("w_int_x", vec![h, h]), ("w_int_h", vec![h, h]),
                ("w_alpha", vec![h, h]), ("u_alpha", vec![h, h]),
                ("w_i", vec![h, h]), ("u_i", vec![h, h]),
                ("w_c", vec![h, h]), ("u_c", vec![h, h]),
                ("bias_alpha", vec![h]),
            ] {
                m.insert(format!("cell_{i}_{suf}"), shp);
            }
        }
        for (suf, shp) in [
            ("heads_label_queries", vec![7, h]),
            ("heads_input_norm", vec![h]),
            ("heads_context_proj", vec![h, h]),
            ("heads_label_proj", vec![h, h]),
            ("heads_global_w1", vec![384, h]),
            ("heads_global_w2", vec![7, 384]),
            ("heads_diff_w2", vec![3, 384]),
        ] {
            m.insert(suf.to_string(), shp);
        }
        m
    }

    #[test]
    fn build_graph_success_with_all_weights() {
        let cfg = C1V2Config::default();
        let shapes = default_weight_shapes();
        let g = build_c1_v2_graph(&cfg, &shapes).expect("graph build ok");
        assert_eq!(g.outputs.len(), 5); // 2 logits + 3 state outputs
    }

    #[test]
    fn build_graph_fails_missing_weight() {
        let mut shapes = default_weight_shapes();
        shapes.remove("cell_0_w_int_x");
        let err = build_c1_v2_graph(&C1V2Config::default(), &shapes).unwrap_err();
        assert!(matches!(err, C1V2GraphError::MissingWeight(_)));
    }

    #[test]
    fn config_default_values() {
        let c = C1V2Config::default();
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.num_cell_layers, 3);
        assert_eq!(c.num_intents, 7);
        assert_eq!(c.num_difficulty, 3);
    }
}
