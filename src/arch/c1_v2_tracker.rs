//! c1 v2 DialogueGate Intent Tracker — stateful runtime (REQ-C1-001~006)
//!
//! Loads `v2_granite_best.pt` (PyTorch checkpoint, nested under `model` key)
//! and performs single-turn state-recurrent inference:
//!
//!   step(turn_embed[768], h_prev[3×768]) → (h_next[3×768], intent_logits[7], diff_logits[3])
//!
//! Numerics mirror `c1/src/intent/model/dialogue_gate.py` (DialogueGateCell,
//! 3-layer stack) and `scorer.py` (IntentDifficultyHead with label-query
//! attention + CORN ordinal difficulty). The caller manages cross-turn state
//! (`h_prev`/`h_next`); this tracker is stateless across calls.
//!
//! The encoder (granite-embedding-311m ModernBERT) is frozen and run
//! separately by the caller — this tracker consumes pre-encoded 768-dim
//! turn embeddings, matching the c1 `TrainModel` (encoder-free) design.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};

use crate::arch::c1_v2_graph::{build_c1_v2_graph, C1V2Config, C1V2GraphError};
use crate::loader::pytorch::{PytorchLoader, PytorchLoaderConfig};
use crate::loader::TensorProvider;

/// Intent label order (v2_granite 7-intent model, from c1 config).
pub const INTENT_LABELS: [&str; 7] = [
    "code_generation",
    "bug_fixing",
    "concept_explanation",
    "refactoring",
    "testing",
    "documentation",
    "deployment",
];

/// Errors raised by the c1 v2 tracker.
#[derive(Debug, Clone, thiserror::Error)]
pub enum C1V2TrackerError {
    #[error("load failed: {0}")]
    LoadFailed(String),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("invalid dimension for {name}: expected {expected:?}, got {actual:?}")]
    InvalidDimension { name: String, expected: Vec<usize>, actual: Vec<usize> },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    #[error("graph error: {0}")]
    GraphError(String),
}

impl From<C1V2GraphError> for C1V2TrackerError {
    fn from(e: C1V2GraphError) -> Self {
        Self::GraphError(e.to_string())
    }
}

/// Single-turn state-recurrent intent classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct C1V2StepResult {
    /// Updated dialogue state, one (hidden_dim,) vector per cell layer.
    pub h_next: Vec<Vec<f32>>,
    /// Multi-label intent logits (7).
    pub intent_logits: Vec<f32>,
    /// CORN ordinal difficulty logits (3).
    pub diff_logits: Vec<f32>,
}

/// c1 v2 DialogueGate intent tracker (stateful across calls via h_prev).
#[derive(Debug, Clone)]
pub struct C1V2Tracker {
    config: C1V2Config,
    /// Weight name → flat f32 data (row-major, torch Linear weight (out,in)).
    weights: HashMap<String, Vec<f32>>,
    /// Weight name → shape.
    weight_shapes: HashMap<String, Vec<usize>>,
}

impl C1V2Tracker {
    /// Load a c1 v2 checkpoint from a `.pt` file (PyTorch pickle, nested
    /// under the `model` key). Weights convert to f32 on load
    /// (ARCH-JIT-DATA-YIELDS: dtype inferred from .pt; F32/BF16/F16 supported).
    ///
    /// # Errors
    /// - `LoadFailed` if the file cannot be read/parsed
    /// - `MissingWeight` if a required weight is absent
    /// - `InvalidDimension` if a weight shape mismatches the config
    pub fn from_pt<P: AsRef<Path>>(path: P) -> Result<Self, C1V2TrackerError> {
        Self::from_pt_with_config(path, C1V2Config::default())
    }

    /// Load with an explicit config (for non-default hidden dims).
    pub fn from_pt_with_config<P: AsRef<Path>>(
        path: P,
        config: C1V2Config,
    ) -> Result<Self, C1V2TrackerError> {
        let loader = PytorchLoader::from_files_with_config(
            &[path.as_ref().to_path_buf()],
            PytorchLoaderConfig {
                state_dict_key: Some("model".to_string()),
                int4_name_hints: vec![],
            },
        )
        .map_err(|e| C1V2TrackerError::LoadFailed(format!("pytorch load: {e}")))?;

        let mut weights = HashMap::new();
        let mut weight_shapes = HashMap::new();

        for meta in loader.iter_tensors() {
            let raw = loader
                .load_tensor_data(&meta.name)
                .map_err(|e| C1V2TrackerError::LoadFailed(format!("load {}: {e}", meta.name)))?;
            let numel: usize = meta.shape.iter().product();
            let float_data: Vec<f32> = match meta.dtype {
                safetensors::Dtype::F32 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => raw
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => raw
                    .chunks_exact(2)
                    .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                other => {
                    return Err(C1V2TrackerError::LoadFailed(format!(
                        "tensor {}: unsupported dtype {other:?}, expected F32/F16/BF16",
                        meta.name
                    )));
                }
            };
            if float_data.len() != numel {
                return Err(C1V2TrackerError::LoadFailed(format!(
                    "tensor {}: parsed {} elements, expected {numel}",
                    meta.name, float_data.len()
                )));
            }
            weight_shapes.insert(meta.name.clone(), meta.shape.clone());
            weights.insert(meta.name, float_data);
        }

        let tracker = Self { config, weights, weight_shapes };
        tracker.validate_weights()?;
        Ok(tracker)
    }

    /// Build from explicit weights (for testing without a .pt file).
    pub fn from_weights(
        config: C1V2Config,
        weights: HashMap<String, Vec<f32>>,
        weight_shapes: HashMap<String, Vec<usize>>,
    ) -> Result<Self, C1V2TrackerError> {
        let tracker = Self { config, weights, weight_shapes };
        tracker.validate_weights()?;
        Ok(tracker)
    }

    /// Returns the model config.
    pub fn config(&self) -> &C1V2Config {
        &self.config
    }

    /// Map canonical .pt tensor names → flat graph-builder names for the
    /// structural verification graph (cell_{i}_*, heads_*).
    fn graph_weight_shapes(&self) -> HashMap<String, Vec<usize>> {
        let mut out = HashMap::new();
        for i in 0..self.config.num_cell_layers {
            for (pt_suf, flat_suf) in [
                ("W_int_x.weight", "w_int_x"),
                ("W_int_h.weight", "w_int_h"),
                ("W_α.weight", "w_alpha"),
                ("U_α.weight", "u_alpha"),
                ("W_i.weight", "w_i"),
                ("U_i.weight", "u_i"),
                ("W_c.weight", "w_c"),
                ("U_c.weight", "u_c"),
                ("bias_α", "bias_alpha"),
            ] {
                let pt_name = format!("tracker.stack.cells.{i}.{pt_suf}");
                let flat_name = format!("cell_{i}_{flat_suf}");
                if let Some(shape) = self.weight_shapes.get(&pt_name) {
                    out.insert(flat_name, shape.clone());
                }
            }
        }
        for (pt_name, flat_name) in [
            ("heads.label_queries", "heads_label_queries"),
            ("heads.input_norm.weight", "heads_input_norm"),
            ("heads.context_proj.weight", "heads_context_proj"),
            ("heads.label_proj.weight", "heads_label_proj"),
            ("heads.global_w1.weight", "heads_global_w1"),
            ("heads.global_w2.weight", "heads_global_w2"),
            ("heads.diff_w2.weight", "heads_diff_w2"),
        ] {
            if let Some(shape) = self.weight_shapes.get(pt_name) {
                out.insert(flat_name.to_string(), shape.clone());
            }
        }
        out
    }

    /// Initial dialogue state: zeros (num_cell_layers × hidden_dim).
    pub fn initial_state(&self) -> Vec<Vec<f32>> {
        vec![vec![0.0; self.config.hidden_dim]; self.config.num_cell_layers]
    }

    /// Single-turn state-recurrent step (REQ-C1-003).
    ///
    /// `turn_embed` is the (hidden_dim,) encoder output for the current turn.
    /// `h_prev` is one (hidden_dim,) vector per cell layer (caller-managed).
    /// Returns the updated state + intent/difficulty readouts.
    ///
    /// # Errors
    /// - `InvalidInput` if dimensions mismatch
    /// - `InferenceFailed` if a weight lookup fails
    pub fn step(
        &self,
        turn_embed: &[f32],
        h_prev: &[Vec<f32>],
    ) -> Result<C1V2StepResult, C1V2TrackerError> {
        let h = self.config.hidden_dim;
        if turn_embed.len() != h {
            return Err(C1V2TrackerError::InvalidInput(format!(
                "turn_embed len {} != hidden_dim {h}",
                turn_embed.len()
            )));
        }
        if h_prev.len() != self.config.num_cell_layers {
            return Err(C1V2TrackerError::InvalidInput(format!(
                "h_prev len {} != num_cell_layers {}",
                h_prev.len(),
                self.config.num_cell_layers
            )));
        }
        for (i, s) in h_prev.iter().enumerate() {
            if s.len() != h {
                return Err(C1V2TrackerError::InvalidInput(format!(
                    "h_prev[{i}] len {} != hidden_dim {h}",
                    s.len()
                )));
            }
        }

        // Build the structural verification graph (mirrors the 3-task
        // IntentTracker pattern: graph = structure, Rust forward = numerics).
        let graph_shapes = self.graph_weight_shapes();
        let _graph = build_c1_v2_graph(&self.config, &graph_shapes)?;

        // ── 3-layer stacked DialogueGateCell recursion ──
        let mut cur_input = turn_embed.to_vec();
        let mut h_next: Vec<Vec<f32>> = Vec::with_capacity(self.config.num_cell_layers);
        for i in 0..self.config.num_cell_layers {
            let prev = &h_prev[i];
            let h_out = self.dialogue_gate_step(i, &cur_input, prev)?;
            h_next.push(h_out.clone());
            cur_input = h_out;
        }

        // ── IntentDifficultyHead (state readout from final cell layer) ──
        let state = &cur_input;
        let (intent_logits, diff_logits) = self.heads_forward(state)?;

        Ok(C1V2StepResult {
            h_next,
            intent_logits,
            diff_logits,
        })
    }

    /// One DialogueGateCell step for layer `i`.
    fn dialogue_gate_step(
        &self,
        i: usize,
        x: &[f32],
        h_prev: &[f32],
    ) -> Result<Vec<f32>, C1V2TrackerError> {
        let d = self.config.hidden_dim;

        let w_int_x = self.get_weight(&format!("tracker.stack.cells.{i}.W_int_x.weight"))?;
        let w_int_h = self.get_weight(&format!("tracker.stack.cells.{i}.W_int_h.weight"))?;
        let x_int = matvec_no_bias(x, w_int_x, d, d)?;
        let h_int = matvec_no_bias(h_prev, w_int_h, d, d)?;

        let interact = hadamard(&x_int, &h_int);
        let consistency: Vec<f32> = interact.iter().map(|v| sigmoid(*v)).collect();

        let bias_alpha = self.get_weight(&format!("tracker.stack.cells.{i}.bias_α"))?;
        let alpha_prior: Vec<f32> = bias_alpha.iter().map(|v| sigmoid(*v)).collect();

        let w_alpha = self.get_weight(&format!("tracker.stack.cells.{i}.W_α.weight"))?;
        let u_alpha = self.get_weight(&format!("tracker.stack.cells.{i}.U_α.weight"))?;
        let w_alpha_x = matvec_no_bias(x, w_alpha, d, d)?;
        let u_alpha_h = matvec_no_bias(h_prev, u_alpha, d, d)?;
        let cons_u_alpha = hadamard(&consistency, &u_alpha_h);
        let alpha_dyn_logit = add_vec(&w_alpha_x, &cons_u_alpha);
        let alpha_dyn: Vec<f32> = alpha_dyn_logit.iter().map(|v| sigmoid(*v)).collect();

        let alpha = hadamard(&alpha_prior, &alpha_dyn);

        let w_i = self.get_weight(&format!("tracker.stack.cells.{i}.W_i.weight"))?;
        let u_i = self.get_weight(&format!("tracker.stack.cells.{i}.U_i.weight"))?;
        let w_i_x = matvec_no_bias(x, w_i, d, d)?;
        let u_i_h = matvec_no_bias(h_prev, u_i, d, d)?;
        let gate_logit = add_vec(&add_vec(&w_i_x, &u_i_h), &interact);
        let gate: Vec<f32> = gate_logit.iter().map(|v| sigmoid(*v)).collect();

        let w_c = self.get_weight(&format!("tracker.stack.cells.{i}.W_c.weight"))?;
        let u_c = self.get_weight(&format!("tracker.stack.cells.{i}.U_c.weight"))?;
        let w_c_x = matvec_no_bias(x, w_c, d, d)?;
        let u_c_h = matvec_no_bias(h_prev, u_c, d, d)?;
        let cand_logit = add_vec(&w_c_x, &u_c_h);
        let candidate: Vec<f32> = cand_logit.iter().map(|v| v.tanh()).collect();

        let alpha_h = hadamard(&alpha, h_prev);
        let gate_cand = hadamard(&gate, &candidate);
        Ok(add_vec(&alpha_h, &gate_cand))
    }

    /// IntentDifficultyHead forward (label-query + global + CORN difficulty).
    fn heads_forward(&self, state: &[f32]) -> Result<(Vec<f32>, Vec<f32>), C1V2TrackerError> {
        let d = self.config.hidden_dim;
        let num_intents = self.config.num_intents;

        let input_norm = self.get_weight("heads.input_norm.weight")?;
        let normed = rmsnorm(state, input_norm, d, 1e-6);

        let context_proj = self.get_weight("heads.context_proj.weight")?;
        let context = matvec_no_bias(&normed, context_proj, d, d)?;

        let label_queries = self.get_weight("heads.label_queries")?;
        let label_proj = self.get_weight("heads.label_proj.weight")?;
        let mut queries = vec![0.0f32; num_intents * d];
        for l in 0..num_intents {
            let lq_row = &label_queries[l * d..(l + 1) * d];
            let q = matvec_no_bias(lq_row, label_proj, d, d)?;
            queries[l * d..(l + 1) * d].copy_from_slice(&q);
        }

        let scale = 1.0_f32 / (d as f32).sqrt();
        let mut label_scores = vec![0.0f32; num_intents];
        for l in 0..num_intents {
            let mut s = 0.0f32;
            for k in 0..d {
                s += context[k] * queries[l * d + k];
            }
            label_scores[l] = s * scale;
        }

        let global_w1 = self.get_weight("heads.global_w1.weight")?;
        let global_w2 = self.get_weight("heads.global_w2.weight")?;
        let global_h = matvec_no_bias(&normed, global_w1, self.config.intent_hidden_dim, d)?;
        let global_a: Vec<f32> = global_h.iter().map(|v| silu(*v)).collect();
        let global_scores = matvec_no_bias(&global_a, global_w2, num_intents, self.config.intent_hidden_dim)?;

        let intent_logits = add_vec(&label_scores, &global_scores);

        // ── CORN difficulty head (real concat, per scorer.py) ──
        let intent_sig: Vec<f32> = intent_logits.iter().map(|v| sigmoid(*v)).collect();
        let mut intent_summary = vec![0.0f32; d];
        for l in 0..num_intents {
            let s = intent_sig[l];
            for k in 0..d {
                intent_summary[k] += s * label_queries[l * d + k];
            }
        }

        let mut diff_input = Vec::with_capacity(2 * d);
        diff_input.extend_from_slice(state);
        diff_input.extend_from_slice(&intent_summary);

        let diff_norm = self.get_weight("heads.diff_norm.weight")?;
        let diff_normed = rmsnorm(&diff_input, diff_norm, 2 * d, 1e-6);

        let diff_w1 = self.get_weight("heads.diff_w1.weight")?;
        let diff_w2 = self.get_weight("heads.diff_w2.weight")?;
        let diff_h = matvec_no_bias(&diff_normed, diff_w1, self.config.difficulty_hidden_dim, 2 * d)?;
        let diff_a: Vec<f32> = diff_h.iter().map(|v| silu(*v)).collect();
        let diff_logits = matvec_no_bias(&diff_a, diff_w2, self.config.num_difficulty, self.config.difficulty_hidden_dim)?;

        Ok((intent_logits, diff_logits))
    }

    fn get_weight(&self, name: &str) -> Result<&[f32], C1V2TrackerError> {
        self.weights
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| C1V2TrackerError::MissingWeight(name.to_string()))
    }

    fn validate_weights(&self) -> Result<(), C1V2TrackerError> {
        let d = self.config.hidden_dim;
        for i in 0..self.config.num_cell_layers {
            for (suf, expected) in [
                ("W_int_x.weight", vec![d, d]),
                ("W_int_h.weight", vec![d, d]),
                ("W_α.weight", vec![d, d]),
                ("U_α.weight", vec![d, d]),
                ("W_i.weight", vec![d, d]),
                ("U_i.weight", vec![d, d]),
                ("W_c.weight", vec![d, d]),
                ("U_c.weight", vec![d, d]),
                ("bias_α", vec![d]),
            ] {
                let name = format!("tracker.stack.cells.{i}.{suf}");
                self.check_shape(&name, &expected)?;
            }
        }
        let ih = self.config.intent_hidden_dim;
        let dh = self.config.difficulty_hidden_dim;
        let ni = self.config.num_intents;
        let nd = self.config.num_difficulty;
        for (name, expected) in [
            ("heads.label_queries", vec![ni, d]),
            ("heads.input_norm.weight", vec![d]),
            ("heads.context_proj.weight", vec![d, d]),
            ("heads.label_proj.weight", vec![d, d]),
            ("heads.global_w1.weight", vec![ih, d]),
            ("heads.global_w2.weight", vec![ni, ih]),
            ("heads.diff_norm.weight", vec![2 * d]),
            ("heads.diff_w1.weight", vec![dh, 2 * d]),
            ("heads.diff_w2.weight", vec![nd, dh]),
        ] {
            self.check_shape(name, &expected)?;
        }
        Ok(())
    }

    fn check_shape(&self, name: &str, expected: &[usize]) -> Result<(), C1V2TrackerError> {
        let actual = self
            .weight_shapes
            .get(name)
            .ok_or_else(|| C1V2TrackerError::MissingWeight(name.to_string()))?;
        if actual != expected {
            return Err(C1V2TrackerError::InvalidDimension {
                name: name.to_string(),
                expected: expected.to_vec(),
                actual: actual.clone(),
            });
        }
        Ok(())
    }
}

// ── Numerics helpers (Rust forward, mirroring dialogue_gate.py + scorer.py) ──

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// out = x @ W^T  where W is (out_dim, in_dim) row-major (torch Linear layout).
fn matvec_no_bias(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Result<Vec<f32>, C1V2TrackerError> {
    if x.len() != in_dim {
        return Err(C1V2TrackerError::InferenceFailed(format!(
            "matvec: x len {} != in_dim {in_dim}",
            x.len()
        )));
    }
    if w.len() != out_dim * in_dim {
        return Err(C1V2TrackerError::InferenceFailed(format!(
            "matvec: w len {} != out_dim*in_dim {}",
            w.len(),
            out_dim * in_dim
        )));
    }
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut s = 0.0f32;
        for k in 0..in_dim {
            s += x[k] * row[k];
        }
        out[o] = s;
    }
    Ok(out)
}

fn hadamard(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
fn rmsnorm(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut sum_sq = 0.0f32;
    for v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v / rms * w)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_tracker() -> C1V2Tracker {
        let config = C1V2Config::default();
        let d = config.hidden_dim;
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        let mut add = |name: &str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001) - 0.5).collect();
            weights.insert(name.to_string(), data);
            shapes.insert(name.to_string(), shape.to_vec());
        };
        for i in 0..3 {
            for suf in ["W_int_x.weight", "W_int_h.weight", "W_α.weight", "U_α.weight",
                "W_i.weight", "U_i.weight", "W_c.weight", "U_c.weight"] {
                add(&format!("tracker.stack.cells.{i}.{suf}"), &[d, d]);
            }
            add(&format!("tracker.stack.cells.{i}.bias_α"), &[d]);
        }
        add("heads.label_queries", &[7, d]);
        add("heads.input_norm.weight", &[d]);
        add("heads.context_proj.weight", &[d, d]);
        add("heads.label_proj.weight", &[d, d]);
        add("heads.global_w1.weight", &[384, d]);
        add("heads.global_w2.weight", &[7, 384]);
        add("heads.diff_norm.weight", &[2 * d]);
        add("heads.diff_w1.weight", &[384, 2 * d]);
        add("heads.diff_w2.weight", &[3, 384]);
        C1V2Tracker::from_weights(config, weights, shapes).expect("valid weights")
    }

    #[test]
    fn initial_state_shape() {
        let t = dummy_tracker();
        let s = t.initial_state();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].len(), 768);
        assert!(s[0].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn step_produces_correct_output_dims() {
        let t = dummy_tracker();
        let embed = vec![0.1; 768];
        let h_prev = t.initial_state();
        let r = t.step(&embed, &h_prev).expect("step ok");
        assert_eq!(r.intent_logits.len(), 7);
        assert_eq!(r.diff_logits.len(), 3);
        assert_eq!(r.h_next.len(), 3);
        assert_eq!(r.h_next[0].len(), 768);
        assert!(r.h_next[0].iter().any(|v| *v != 0.0));
    }

    #[test]
    fn step_rejects_bad_input_dims() {
        let t = dummy_tracker();
        let embed = vec![0.1; 100];
        let h_prev = t.initial_state();
        let err = t.step(&embed, &h_prev).unwrap_err();
        assert!(matches!(err, C1V2TrackerError::InvalidInput(_)));
    }

    #[test]
    fn state_propagates_across_turns() {
        let t = dummy_tracker();
        let e1 = vec![0.2; 768];
        let e2 = vec![0.5; 768];
        let h0 = t.initial_state();
        let r1 = t.step(&e1, &h0).expect("turn1");
        let r2 = t.step(&e2, &r1.h_next).expect("turn2");
        assert_ne!(r1.intent_logits, r2.intent_logits);
    }

    #[test]
    fn from_weights_rejects_missing_weight() {
        let mut t = dummy_tracker();
        t.weights.remove("heads.diff_w2.weight");
        t.weight_shapes.remove("heads.diff_w2.weight");
        assert!(t.validate_weights().is_err());
    }
}
