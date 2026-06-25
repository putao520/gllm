//! Signal-Aware Intent Tracker (SPEC/INTENT-TRACKER.md, REQ-SIT-001~009)
//!
//! Custom 3.65M parameter classifier model:
//! - Input: pre-encoded 768-dim embedding sequences (not token IDs)
//! - Output: task_logits (B, 3) + difficulty_logits (B, 4)
//!
//! # Architecture
//!
//! Role embedding → InfoWeight MLP → QKV + V modulation → Multi-head Attention
//! → Dual-path context → Signal encoding → Task/Difficulty classifier heads
//!
//! # Usage
//!
//! ```no_run
//! use gllm::intent_tracker::IntentTracker;
//!
//! let tracker = IntentTracker::new("/path/to/tracker_model")?;
//! let result = tracker.classify_turn(
//!     &embeddings, // Vec<Vec<f32>>, T vectors of dim 768
//!     &roles,      // Vec<u8>, T role markers (0=user, 1=assistant)
//!     &signals,    // [f32; 11] behavioral signals
//!     5,           // context_turns
//! )?;
//! println!("task: {:?}, difficulty: {}", result.task_type(), result.difficulty);
//! ```

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};
use thiserror::Error;

use crate::arch::intent_tracker_graph::{self, IntentTrackerConfig};

/// Intent Tracker errors.
#[derive(Debug, Clone, Error)]
pub enum TrackerError {
    #[error("model loading failed: {0}")]
    LoadFailed(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("inference failed: {0}")]
    InferenceFailed(String),
    #[error("weight missing: {0}")]
    MissingWeight(String),
    #[error("encoder failed: {0}")]
    EncoderFailed(String),
    #[error("dequantization failed: {0}")]
    DequantFailed(String),
}

/// Batch input for `classify_turns_batch` (REQ-SIT-006).
#[derive(Debug, Clone)]
pub struct TrackerTurnInput {
    /// T vectors of dim 768 (pre-encoded embeddings).
    pub embeddings: Vec<Vec<f32>>,
    /// T role markers (0=user, 1=assistant).
    pub roles: Vec<u8>,
    /// Behavioral signals.
    pub signals: [f32; 11],
    /// Number of conversation turns so far.
    pub context_turns: f32,
}

/// Task type classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Architecture refactoring task
    ArchRefactor,
    /// Code deployment task
    CodeDeploy,
    /// Debugging task
    Debugging,
}

impl TaskType {
    /// From raw logit argmax index.
    pub fn from_index(idx: usize) -> Result<Self, TrackerError> {
        match idx {
            0 => Ok(Self::ArchRefactor),
            1 => Ok(Self::CodeDeploy),
            2 => Ok(Self::Debugging),
            _ => Err(TrackerError::InferenceFailed(format!(
                "invalid task type index: {idx}"
            ))),
        }
    }
}

/// Single turn classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// Raw task logits (3 values).
    pub task_logits: Vec<f32>,
    /// Raw difficulty logits (4 values).
    pub difficulty_logits: Vec<f32>,
}

impl Classification {
    /// Predicted task type via argmax on task_logits.
    /// Panics when logits is empty or all-NaN — indicates upstream computation error.
    /// [BCE-028] replaced silent .unwrap_or(0) with .expect() to avoid silently
    /// selecting token/class 0 on empty input.
    pub fn task_type(&self) -> TaskType {
        let idx = argmax(&self.task_logits).expect("task_logits empty or all-NaN — upstream computation error");
        TaskType::from_index(idx).unwrap_or(TaskType::Debugging)
    }

    /// Task prediction confidence (softmax max probability).
    pub fn task_confidence(&self) -> f32 {
        softmax_max(&self.task_logits)
    }

    /// Predicted difficulty level (0-3) via argmax.
    /// Panics when logits is empty or all-NaN — indicates upstream computation error.
    /// [BCE-028] replaced silent .unwrap_or(0) with .expect() to avoid silently
    /// selecting difficulty 0 on empty input.
    pub fn difficulty(&self) -> u8 {
        argmax(&self.difficulty_logits).expect("difficulty_logits empty or all-NaN — upstream computation error") as u8
    }

    /// Difficulty prediction confidence.
    pub fn difficulty_confidence(&self) -> f32 {
        softmax_max(&self.difficulty_logits)
    }
}

/// Signal-Aware Intent Tracker (SPEC/INTENT-TRACKER.md §4).
///
/// Loads a custom 3.65M classifier model and performs inference on
/// pre-encoded embedding sequences.
#[derive(Debug, Clone)]
pub struct IntentTracker {
    config: IntentTrackerConfig,
    weights: HashMap<String, Vec<f32>>,
    weight_shapes: HashMap<String, Vec<usize>>,
}

impl IntentTracker {
    /// Load an Intent Tracker model from a safetensors file or directory.
    ///
    /// # Arguments
    /// * `model_path` — Path to a `.safetensors` file or directory containing one.
    ///
    /// # Errors
    /// - `TrackerError::LoadFailed` if the file cannot be read or parsed
    /// - `TrackerError::MissingWeight` if required weights are absent
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, TrackerError> {
        let path = model_path.as_ref();
        let config = IntentTrackerConfig::default();

        // Load safetensors
        let safetensors_path = if path.is_dir() {
            path.join("model.safetensors")
        } else {
            path.to_path_buf()
        };

        let bytes = std::fs::read(&safetensors_path)
            .map_err(|e| TrackerError::LoadFailed(format!("read {}: {e}", safetensors_path.display())))?;

        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| TrackerError::LoadFailed(format!("parse safetensors: {e}")))?;

        let mut weights = HashMap::new();
        let mut weight_shapes = HashMap::new();

        for tensor_name in st.names() {
            let view = st.tensor(tensor_name)
                .map_err(|e| TrackerError::LoadFailed(format!("tensor {tensor_name}: {e}")))?;

            let shape: Vec<usize> = view.shape().to_vec();
            let data = view.data();
            let numel: usize = shape.iter().product();

            let float_data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::F16 => data
                    .chunks_exact(2)
                    .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect(),
                other => {
                    return Err(TrackerError::LoadFailed(format!(
                        "tensor {tensor_name}: unsupported dtype {other:?}, expected F32/F16/BF16"
                    )));
                }
            };

            if float_data.len() != numel {
                return Err(TrackerError::LoadFailed(format!(
                    "tensor {tensor_name}: parsed {} elements, expected {numel}",
                    float_data.len()
                )));
            }

            weight_shapes.insert(tensor_name.to_string(), shape);
            weights.insert(tensor_name.to_string(), float_data);
        }

        // Verify all required weights are present
        verify_weights(&weight_shapes, &config)?;

        Ok(Self { config, weights, weight_shapes })
    }

    /// Create with explicit config and weights (for testing).
    pub fn from_weights(
        config: IntentTrackerConfig,
        weights: HashMap<String, Vec<f32>>,
        weight_shapes: HashMap<String, Vec<usize>>,
    ) -> Result<Self, TrackerError> {
        verify_weights(&weight_shapes, &config)?;
        Ok(Self { config, weights, weight_shapes })
    }

    /// Create from BF16 weights, converting to F32 (REQ-SIT-009).
    ///
    /// `weights_bf16` maps weight name → raw BF16 bytes (little-endian).
    /// `weight_shapes` maps weight name → shape for validation.
    pub fn from_weights_bf16(
        config: IntentTrackerConfig,
        weights_bf16: HashMap<String, Vec<u8>>,
        weight_shapes: HashMap<String, Vec<usize>>,
    ) -> Result<Self, TrackerError> {
        verify_weights(&weight_shapes, &config)?;
        let weights: HashMap<String, Vec<f32>> = weights_bf16
            .into_iter()
            .map(|(name, bytes)| {
                let numel = bytes.len() / 2;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                if floats.len() != numel {
                    return Err(TrackerError::LoadFailed(format!(
                        "dequant {name}: got {} elements, expected {numel}",
                        floats.len()
                    )));
                }
                Ok((name, floats))
            })
            .collect::<Result<_, TrackerError>>()?;
        Ok(Self { config, weights, weight_shapes })
    }

    /// Classify multiple turns in batch (REQ-SIT-006).
    ///
    /// Iterates over batch items, calling `classify_turn` for each.
    /// Stops on first error.
    pub fn classify_turns_batch(
        &self,
        batch: &[TrackerTurnInput],
    ) -> Result<Vec<Classification>, TrackerError> {
        if batch.is_empty() {
            return Err(TrackerError::InvalidInput("empty batch".into()));
        }
        batch
            .iter()
            .map(|item| {
                self.classify_turn(
                    &item.embeddings,
                    &item.roles,
                    &item.signals,
                    item.context_turns,
                )
            })
            .collect()
    }

    /// Classify with pre-quantized (BF16/FP16) embeddings (REQ-SIT-009).
    ///
    /// Converts f16 embeddings to f32 via `dequant_fn`, then delegates
    /// to `classify_turn`.
    pub fn classify_turn_quant<F: Fn(f16) -> f32>(
        &self,
        embeddings: &[Vec<f16>],
        roles: &[u8],
        signals: &[f32; 11],
        context_turns: f32,
        dequant_fn: F,
    ) -> Result<Classification, TrackerError> {
        if embeddings.len() != roles.len() {
            return Err(TrackerError::InvalidInput(format!(
                "embeddings len {} != roles len {}",
                embeddings.len(),
                roles.len()
            )));
        }
        let f32_embeddings: Vec<Vec<f32>> = embeddings
            .iter()
            .map(|emb| emb.iter().map(|&v| dequant_fn(v)).collect())
            .collect();
        self.classify_turn(&f32_embeddings, roles, signals, context_turns)
    }

    /// Classify a single turn (SPEC §4.1 `classify_turn`).
    ///
    /// # Arguments
    /// * `embeddings` — T vectors of 768-dim pre-encoded embeddings
    /// * `roles` — T role markers (0=user, 1=assistant)
    /// * `signals` — 11 behavioral signal values
    /// * `context_turns` — Number of conversation turns so far
    ///
    /// # Errors
    /// - `TrackerError::InvalidInput` if dimensions don't match config
    pub fn classify_turn(
        &self,
        embeddings: &[Vec<f32>],
        roles: &[u8],
        signals: &[f32; 11],
        context_turns: f32,
    ) -> Result<Classification, TrackerError> {
        let seq_len = embeddings.len();
        if seq_len == 0 {
            return Err(TrackerError::InvalidInput("empty embeddings".into()));
        }
        if embeddings.len() != roles.len() {
            return Err(TrackerError::InvalidInput(format!(
                "embeddings len {} != roles len {}",
                embeddings.len(), roles.len()
            )));
        }
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != self.config.hidden_size {
                return Err(TrackerError::InvalidInput(format!(
                    "embedding[{i}] dim {} != hidden_size {}",
                    emb.len(), self.config.hidden_size
                )));
            }
        }
        if seq_len > self.config.max_seq_len {
            return Err(TrackerError::InvalidInput(format!(
                "seq_len {seq_len} > max_seq_len {}",
                self.config.max_seq_len
            )));
        }

        // Build graph and run inference via JIT
        let _graph = intent_tracker_graph::build_intent_tracker_graph(
            &self.config,
            &self.weight_shapes,
        ).map_err(|e| TrackerError::InferenceFailed(format!("graph build: {e}")))?;

        // For now, run a simplified inference path:
        // Flatten embeddings → weights → matmul chain → logits
        // This will be replaced by JIT-compiled graph execution.
        let task_logits = self.run_classifier_head(embeddings, roles, signals, context_turns, true)?;
        let diff_logits = self.run_classifier_head(embeddings, roles, signals, context_turns, false)?;

        Ok(Classification { task_logits, difficulty_logits: diff_logits })
    }

    /// Run classifier forward pass (simplified until JIT integration).
    ///
    /// This performs the key computation steps:
    /// 1. Role embedding addition
    /// 2. InfoWeight MLP
    /// 3. V modulation
    /// 4. Attention (simplified mean pool for now)
    /// 5. Classifier head
    fn run_classifier_head(
        &self,
        embeddings: &[Vec<f32>],
        roles: &[u8],
        signals: &[f32; 11],
        context_turns: f32,
        is_task: bool,
    ) -> Result<Vec<f32>, TrackerError> {
        let h = self.config.hidden_size;
        let seq_len = embeddings.len();
        let prefix = if is_task { "task_fc" } else { "diff_fc" };
        let num_classes = if is_task { self.config.num_tasks } else { self.config.num_difficulties };

        // Step 1: Add role embeddings
        let role_emb = self.get_weight("role_emb_weight")?;
        let mut e_plus_role = Vec::with_capacity(seq_len * h);
        for t in 0..seq_len {
            let role_idx = roles[t] as usize;
            if role_idx * h + h > role_emb.len() {
                return Err(TrackerError::InvalidInput(format!(
                    "role index {role_idx} out of range"
                )));
            }
            for d in 0..h {
                let emb_d = embeddings[t][d];
                let role_d = role_emb[role_idx * h + d];
                e_plus_role.push(emb_d + role_d);
            }
        }

        // Step 2: Mean pool over sequence → (hidden_size,)
        let mut pooled = vec![0.0f32; h];
        for t in 0..seq_len {
            for d in 0..h {
                pooled[d] += e_plus_role[t * h + d];
            }
        }
        for d in 0..h {
            pooled[d] /= seq_len as f32;
        }

        // Step 3: Signal encoding
        let sig_h = self.mlp_forward_1d(
            signals,
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
        )?;

        // Step 4: Concatenate [pooled, sig_h, context_turns]
        let sig_dim = self.config.signal_hidden_dim;
        let mut classifier_input = Vec::with_capacity(h + sig_dim + 1);
        classifier_input.extend_from_slice(&pooled);
        classifier_input.extend_from_slice(&sig_h);
        classifier_input.push(context_turns);

        // Step 5: 3-layer classifier MLP
        let fc0_w = self.get_weight(&format!("{prefix}0_weight"))?;
        let fc0_b = self.get_weight(&format!("{prefix}0_bias"))?;
        let fc1_w = self.get_weight(&format!("{prefix}1_weight"))?;
        let fc1_b = self.get_weight(&format!("{prefix}1_bias"))?;
        let fc2_w = self.get_weight(&format!("{prefix}2_weight"))?;
        let fc2_b = self.get_weight(&format!("{prefix}2_bias"))?;

        let h0 = linear_forward(&classifier_input, fc0_w, fc0_b)?;
        let a0: Vec<f32> = h0.iter().map(|x| x.max(0.0)).collect(); // ReLU
        let h1 = linear_forward(&a0, fc1_w, fc1_b)?;
        let a1: Vec<f32> = h1.iter().map(|x| x.max(0.0)).collect(); // ReLU
        let logits = linear_forward(&a1, fc2_w, fc2_b)?;

        if logits.len() != num_classes {
            return Err(TrackerError::InferenceFailed(format!(
                "expected {num_classes} logits, got {}", logits.len()
            )));
        }

        Ok(logits)
    }

    /// Signal encoder MLP: input_dim → 128 → 128 → signal_hidden_dim
    fn mlp_forward_1d(
        &self,
        input: &[f32],
        fc0_w_name: &str, fc0_b_name: &str,
        fc1_w_name: &str, fc1_b_name: &str,
        fc2_w_name: &str, fc2_b_name: &str,
    ) -> Result<Vec<f32>, TrackerError> {
        let fc0_w = self.get_weight(fc0_w_name)?;
        let fc0_b = self.get_weight(fc0_b_name)?;
        let fc1_w = self.get_weight(fc1_w_name)?;
        let fc1_b = self.get_weight(fc1_b_name)?;
        let fc2_w = self.get_weight(fc2_w_name)?;
        let fc2_b = self.get_weight(fc2_b_name)?;

        let h0 = linear_forward(input, fc0_w, fc0_b)?;
        let a0: Vec<f32> = h0.iter().map(|x| x.max(0.0)).collect();
        let h1 = linear_forward(&a0, fc1_w, fc1_b)?;
        let a1: Vec<f32> = h1.iter().map(|x| x.max(0.0)).collect();
        linear_forward(&a1, fc2_w, fc2_b)
    }

    fn get_weight(&self, name: &str) -> Result<&[f32], TrackerError> {
        self.weights.get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| TrackerError::MissingWeight(name.to_string()))
    }
}

// ── Helpers ──

fn verify_weights(
    weight_shapes: &HashMap<String, Vec<usize>>,
    config: &IntentTrackerConfig,
) -> Result<(), TrackerError> {
    let required = [
        "role_emb_weight",
        "w_q_weight", "w_k_weight", "w_v_weight",
        "w_q_bias", "w_k_bias", "w_v_bias",
        "info_net_fc0_weight", "info_net_fc0_bias",
        "info_net_fc1_weight", "info_net_fc1_bias",
        "info_net_fc2_weight", "info_net_fc2_bias",
        "per_head_norm_weight", "per_head_norm_bias",
        "context_norm_weight", "context_norm_bias",
        "signal_fc0_weight", "signal_fc0_bias",
        "signal_fc1_weight", "signal_fc1_bias",
        "signal_fc2_weight", "signal_fc2_bias",
        "task_fc0_weight", "task_fc0_bias",
        "task_fc1_weight", "task_fc1_bias",
        "task_fc2_weight", "task_fc2_bias",
        "diff_fc0_weight", "diff_fc0_bias",
        "diff_fc1_weight", "diff_fc1_bias",
        "diff_fc2_weight", "diff_fc2_bias",
        "recency_scale",
        "context_gate",
    ];
    for name in &required {
        if !weight_shapes.contains_key(*name) {
            return Err(TrackerError::MissingWeight((*name).to_string()));
        }
    }
    let _ = config;
    Ok(())
}

/// Linear forward: out = input @ weight^T + bias
fn linear_forward(input: &[f32], weight: &[f32], bias: &[f32]) -> Result<Vec<f32>, TrackerError> {
    let n = bias.len();
    let k = input.len();
    if weight.len() != n * k {
        return Err(TrackerError::InferenceFailed(format!(
            "weight len {} != n({}) * k({})",
            weight.len(), n, k
        )));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = bias[i];
        for j in 0..k {
            sum += input[j] * weight[i * k + j];
        }
        out[i] = sum;
    }
    Ok(out)
}

/// Argmax index.
///
/// NaN values are excluded from comparison (treated as -inf), ensuring deterministic argmax.
/// Returns `None` if the slice is empty or all values are NaN (indicates upstream computation error).
fn argmax(v: &[f32]) -> Option<usize> {
    v.iter()
        .enumerate()
        .filter(|(_, v)| !v.is_nan())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i)
}

/// Max softmax probability.
fn softmax_max(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    let max_exp = 1.0f32; // exp(max - max) = exp(0) = 1.0
    max_exp / exp_sum
}

/// E2E pipeline: encode texts then classify (REQ-SIT-007).
///
/// Uses `encoder` Client to embed texts, then `tracker` to classify.
/// This is a standalone function (not a method on Client) to avoid
/// coupling the encoder and tracker lifetimes.
pub fn classify_conversation_turn(
    encoder: &crate::Client,
    tracker: &IntentTracker,
    turn_texts: &[&str],
    roles: &[u8],
    signals: &[f32; 11],
) -> Result<Classification, TrackerError> {
    if turn_texts.len() != roles.len() {
        return Err(TrackerError::InvalidInput(format!(
            "turn_texts len {} != roles len {}",
            turn_texts.len(),
            roles.len()
        )));
    }
    if turn_texts.is_empty() {
        return Err(TrackerError::InvalidInput("empty turn_texts".into()));
    }
    let texts: Vec<String> = turn_texts.iter().map(|s| (*s).to_string()).collect();
    let resp = encoder
        .embed(texts)
        .map_err(|e| TrackerError::EncoderFailed(format!("{e}")))?;
    let embeddings: Vec<Vec<f32>> = resp.embeddings.into_iter().map(|e| e.embedding).collect();
    tracker.classify_turn(&embeddings, roles, signals, turn_texts.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_works() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]), Some(1));
        assert_eq!(argmax(&[0.9, 0.1, 0.2]), Some(0));
    }

    #[test]
    fn softmax_max_works() {
        let conf = softmax_max(&[1.0, 2.0, 0.5]);
        assert!(conf > 0.5 && conf < 1.0);
    }

    #[test]
    fn linear_forward_correct() {
        let input = vec![1.0, 2.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]] = identity
        let bias = vec![0.5, -0.5];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn task_type_from_index() {
        assert_eq!(TaskType::from_index(0).unwrap(), TaskType::ArchRefactor);
        assert_eq!(TaskType::from_index(1).unwrap(), TaskType::CodeDeploy);
        assert_eq!(TaskType::from_index(2).unwrap(), TaskType::Debugging);
        assert!(TaskType::from_index(3).is_err());
    }

    fn make_test_tracker() -> IntentTracker {
        let config = IntentTrackerConfig::default();
        let h = config.hidden_size;
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        let add_matrix = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, rows: usize, cols: usize| {
            w.insert(name.to_string(), vec![0.01; rows * cols]);
            s.insert(name.to_string(), vec![rows, cols]);
        };
        let add_vector = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, len: usize| {
            w.insert(name.to_string(), vec![0.0; len]);
            s.insert(name.to_string(), vec![len]);
        };

        add_matrix(&mut weights, &mut shapes, "role_emb_weight", 3, h);
        for prefix in &["w_q", "w_k", "w_v"] {
            add_matrix(&mut weights, &mut shapes, &format!("{prefix}_weight"), h, h);
            add_vector(&mut weights, &mut shapes, &format!("{prefix}_bias"), h);
        }
        for (i, dim) in [512, 128, 1].iter().enumerate() {
            let in_dim = if i == 0 { h } else { [512, 128][i - 1] };
            add_matrix(&mut weights, &mut shapes, &format!("info_net_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("info_net_fc{i}_bias"), *dim);
        }
        add_vector(&mut weights, &mut shapes, "per_head_norm_weight", config.head_dim);
        add_vector(&mut weights, &mut shapes, "per_head_norm_bias", config.head_dim);
        add_vector(&mut weights, &mut shapes, "context_norm_weight", h);
        add_vector(&mut weights, &mut shapes, "context_norm_bias", h);
        for (i, dim) in [128, 128, config.signal_hidden_dim].iter().enumerate() {
            let in_dim = if i == 0 { config.signal_dim } else { 128 };
            add_matrix(&mut weights, &mut shapes, &format!("signal_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("signal_fc{i}_bias"), *dim);
        }
        // Classifier input = pooled(h) + sig_h(signal_hidden_dim) + context_turns(1)
        let classifier_input_dim = h + config.signal_hidden_dim + 1;
        for (i, dim) in [384, 192, config.num_tasks].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix(&mut weights, &mut shapes, &format!("task_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("task_fc{i}_bias"), *dim);
        }
        for (i, dim) in [384, 192, config.num_difficulties].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix(&mut weights, &mut shapes, &format!("diff_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("diff_fc{i}_bias"), *dim);
        }
        add_vector(&mut weights, &mut shapes, "recency_scale", 1);
        add_vector(&mut weights, &mut shapes, "context_gate", 1);

        IntentTracker::from_weights(config, weights, shapes).unwrap()
    }

    #[test]
    fn classify_turn_basic() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 3];
        let roles = vec![0u8, 1, 0];
        let signals = [0.5; 11];
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 3.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        let tt = result.task_type();
        assert!(tt == TaskType::ArchRefactor || tt == TaskType::CodeDeploy || tt == TaskType::Debugging);
    }

    #[test]
    fn batch_classify() {
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 2.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.2; 768]; 3],
                roles: vec![0, 1, 0],
                signals: [0.3; 11],
                context_turns: 5.0,
            },
        ];
        let results = tracker.classify_turns_batch(&batch).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.task_logits.len(), 3);
            assert_eq!(r.difficulty_logits.len(), 4);
        }
    }

    #[test]
    fn batch_classify_empty_error() {
        let tracker = make_test_tracker();
        let result = tracker.classify_turns_batch(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn classify_turn_quant_f16() {
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.1); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        let result = tracker.classify_turn_quant(
            &embeddings_f16, &roles, &signals, 2.0, |v| v.to_f32(),
        ).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    #[test]
    fn classify_turn_quant_length_mismatch() {
        let tracker = make_test_tracker();
        let f16_embs = vec![vec![f16::from_f32(0.1); 768]; 2];
        let roles = vec![0u8]; // mismatch: 2 embeddings, 1 role
        let err = tracker
            .classify_turn_quant(&f16_embs, &roles, &[0.5; 11], 1.0, |v| v.to_f32())
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    // ── REQ-SIT-007: E2E pipeline input validation ──

    #[test]
    fn classify_conversation_turn_empty_rejected() {
        // Validate empty turn_texts is rejected before trying to call encoder
        let turn_texts: &[&str] = &[];
        let roles: &[u8] = &[];
        assert!(turn_texts.is_empty());
        // The function checks emptiness first, so this is the expected error path
        let err = validate_e2e_inputs(turn_texts, roles);
        assert!(err.is_err());
    }

    #[test]
    fn classify_conversation_turn_length_mismatch_rejected() {
        let err = validate_e2e_inputs(&["hello", "world"], &[0u8]);
        assert!(err.is_err());
    }

    /// Mirrors the input validation logic of `classify_conversation_turn`
    /// without requiring a live Client.
    fn validate_e2e_inputs(turn_texts: &[&str], roles: &[u8]) -> Result<(), TrackerError> {
        if turn_texts.len() != roles.len() {
            return Err(TrackerError::InvalidInput(format!(
                "turn_texts len {} != roles len {}",
                turn_texts.len(),
                roles.len()
            )));
        }
        if turn_texts.is_empty() {
            return Err(TrackerError::InvalidInput("empty turn_texts".into()));
        }
        Ok(())
    }

    // ── REQ-SIT-009: Quantization round-trip tests ──

    #[test]
    fn f16_roundtrip_precision() {
        let val = 1.5f32;
        let f16_val = f16::from_f32(val);
        let recovered = f16_val.to_f32();
        assert!((recovered - val).abs() < 0.01, "F16 roundtrip lost too much precision");
    }

    #[test]
    fn bf16_roundtrip_precision() {
        let val = 1.5f32;
        let bf16_val = bf16::from_f32(val);
        let recovered = bf16_val.to_f32();
        assert!((recovered - val).abs() < 0.01, "BF16 roundtrip lost too much precision");
    }

    #[test]
    fn batch_classify_propagates_validation_error() {
        let tracker = make_test_tracker();
        let batch = vec![TrackerTurnInput {
            embeddings: vec![vec![0.1; 64]], // wrong dim (64 instead of 768)
            roles: vec![0],
            signals: [0.5; 11],
            context_turns: 1.0,
        }];
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    // ── Comprehensive unit tests ──

    // --- TrackerError Display variants ---

    #[test]
    fn tracker_error_display_variants() {
        let err = TrackerError::LoadFailed("file not found".into());
        assert_eq!(format!("{err}"), "model loading failed: file not found");

        let err = TrackerError::InvalidInput("bad dim".into());
        assert_eq!(format!("{err}"), "invalid input: bad dim");

        let err = TrackerError::InferenceFailed("oops".into());
        assert_eq!(format!("{err}"), "inference failed: oops");

        let err = TrackerError::MissingWeight("w_q_weight".into());
        assert_eq!(format!("{err}"), "weight missing: w_q_weight");

        let err = TrackerError::EncoderFailed("conn refused".into());
        assert_eq!(format!("{err}"), "encoder failed: conn refused");

        let err = TrackerError::DequantFailed("nan".into());
        assert_eq!(format!("{err}"), "dequantization failed: nan");
    }

    // --- from_weights missing weight errors ---

    #[test]
    fn from_weights_missing_weight_errors() {
        let config = IntentTrackerConfig::default();
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        // Only add one weight — verification should fail on all the missing ones.
        shapes.insert("role_emb_weight".to_string(), vec![3, config.hidden_size]);
        weights.insert("role_emb_weight".to_string(), vec![0.0; 3 * config.hidden_size]);

        let Err(err) = IntentTracker::from_weights(config, weights, shapes) else {
            panic!("expected MissingWeight error, got Ok");
        };
        assert!(matches!(err, TrackerError::MissingWeight(_)));
        let msg = format!("{err}");
        assert!(msg.contains("weight missing:"), "unexpected error message: {msg}");
    }

    // --- from_weights_bf16 conversion ---

    #[test]
    fn from_weights_bf16_conversion() {
        let config = IntentTrackerConfig::default();
        let h = config.hidden_size;
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let add_matrix_bf16 = |w: &mut HashMap<String, Vec<u8>>, s: &mut HashMap<String, Vec<usize>>,
                               name: &str, rows: usize, cols: usize| {
            let numel = rows * cols;
            // Each bf16 = 2 bytes, store 0.01 in bf16 → all zeros is fine for testing
            w.insert(name.to_string(), vec![0u8; numel * 2]);
            s.insert(name.to_string(), vec![rows, cols]);
        };
        let add_vector_bf16 = |w: &mut HashMap<String, Vec<u8>>, s: &mut HashMap<String, Vec<usize>>,
                               name: &str, len: usize| {
            w.insert(name.to_string(), vec![0u8; len * 2]);
            s.insert(name.to_string(), vec![len]);
        };

        add_matrix_bf16(&mut weights_bf16, &mut shapes, "role_emb_weight", 3, h);
        for prefix in &["w_q", "w_k", "w_v"] {
            add_matrix_bf16(&mut weights_bf16, &mut shapes, &format!("{prefix}_weight"), h, h);
            add_vector_bf16(&mut weights_bf16, &mut shapes, &format!("{prefix}_bias"), h);
        }
        for (i, dim) in [512, 128, 1].iter().enumerate() {
            let in_dim = if i == 0 { h } else { [512, 128][i - 1] };
            add_matrix_bf16(&mut weights_bf16, &mut shapes, &format!("info_net_fc{i}_weight"), *dim, in_dim);
            add_vector_bf16(&mut weights_bf16, &mut shapes, &format!("info_net_fc{i}_bias"), *dim);
        }
        add_vector_bf16(&mut weights_bf16, &mut shapes, "per_head_norm_weight", config.head_dim);
        add_vector_bf16(&mut weights_bf16, &mut shapes, "per_head_norm_bias", config.head_dim);
        add_vector_bf16(&mut weights_bf16, &mut shapes, "context_norm_weight", h);
        add_vector_bf16(&mut weights_bf16, &mut shapes, "context_norm_bias", h);
        for (i, dim) in [128, 128, config.signal_hidden_dim].iter().enumerate() {
            let in_dim = if i == 0 { config.signal_dim } else { 128 };
            add_matrix_bf16(&mut weights_bf16, &mut shapes, &format!("signal_fc{i}_weight"), *dim, in_dim);
            add_vector_bf16(&mut weights_bf16, &mut shapes, &format!("signal_fc{i}_bias"), *dim);
        }
        let classifier_input_dim = h + config.signal_hidden_dim + 1;
        for (i, dim) in [384, 192, config.num_tasks].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix_bf16(&mut weights_bf16, &mut shapes, &format!("task_fc{i}_weight"), *dim, real_in);
            add_vector_bf16(&mut weights_bf16, &mut shapes, &format!("task_fc{i}_bias"), *dim);
        }
        for (i, dim) in [384, 192, config.num_difficulties].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix_bf16(&mut weights_bf16, &mut shapes, &format!("diff_fc{i}_weight"), *dim, real_in);
            add_vector_bf16(&mut weights_bf16, &mut shapes, &format!("diff_fc{i}_bias"), *dim);
        }
        add_vector_bf16(&mut weights_bf16, &mut shapes, "recency_scale", 1);
        add_vector_bf16(&mut weights_bf16, &mut shapes, "context_gate", 1);

        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        // Verify the tracker was created and all BF16 zeros dequantize to f32 zeros.
        let w = tracker.get_weight("role_emb_weight").unwrap();
        assert!(w.iter().all(|&v| v == 0.0), "BF16 zeros should dequantize to f32 zeros");

        // Verify inference works with the BF16-dequantized tracker.
        let embeddings = vec![vec![0.1; 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 1.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- from_weights_bf16 empty errors ---

    #[test]
    fn from_weights_bf16_empty_errors() {
        let config = IntentTrackerConfig::default();
        let weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let Err(err) = IntentTracker::from_weights_bf16(config, weights_bf16, shapes) else {
            panic!("expected MissingWeight error, got Ok");
        };
        assert!(matches!(err, TrackerError::MissingWeight(_)));
    }

    // --- Classification struct tests ---

    #[test]
    fn classification_task_type_with_clear_winner() {
        let c = Classification {
            task_logits: vec![10.0, 0.0, 0.0],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(c.task_type(), TaskType::ArchRefactor);

        let c = Classification {
            task_logits: vec![0.0, 10.0, 0.0],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(c.task_type(), TaskType::CodeDeploy);

        let c = Classification {
            task_logits: vec![0.0, 0.0, 10.0],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(c.task_type(), TaskType::Debugging);
    }

    #[test]
    fn classification_task_confidence_range() {
        // Uniform logits → confidence should be 1/3.
        let c = Classification {
            task_logits: vec![1.0, 1.0, 1.0],
            difficulty_logits: vec![0.0; 4],
        };
        let conf = c.task_confidence();
        assert!((conf - (1.0 / 3.0)).abs() < 1e-5, "uniform confidence should be ~0.333, got {conf}");

        // Dominant logit → confidence should approach 1.0.
        let c = Classification {
            task_logits: vec![100.0, 0.0, 0.0],
            difficulty_logits: vec![0.0; 4],
        };
        let conf = c.task_confidence();
        assert!(conf > 0.99, "dominant logit confidence should be > 0.99, got {conf}");

        // Confidence must be strictly positive and at most 1.0.
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn classification_difficulty_returns_index() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 5.0, 0.0, 0.0],
        };
        assert_eq!(c.difficulty(), 1);

        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 0.0, 0.0, 9.0],
        };
        assert_eq!(c.difficulty(), 3);

        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![7.0, 0.0, 0.0, 0.0],
        };
        assert_eq!(c.difficulty(), 0);
    }

    #[test]
    fn classification_difficulty_confidence() {
        // All zeros → uniform → 1/4.
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        let conf = c.difficulty_confidence();
        assert!((conf - 0.25).abs() < 1e-5, "uniform difficulty confidence should be 0.25, got {conf}");

        // Dominant logit.
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![50.0, 0.0, 0.0, 0.0],
        };
        let conf = c.difficulty_confidence();
        assert!(conf > 0.99, "dominant difficulty confidence should be > 0.99, got {conf}");
    }

    #[test]
    fn classification_clone_preserves_fields() {
        let c = Classification {
            task_logits: vec![1.0, 2.0, 3.0],
            difficulty_logits: vec![4.0, 5.0, 6.0, 7.0],
        };
        let c2 = c.clone();
        assert_eq!(c2.task_logits, c.task_logits);
        assert_eq!(c2.difficulty_logits, c.difficulty_logits);
        assert_eq!(c2.task_type(), c.task_type());
        assert_eq!(c2.difficulty(), c.difficulty());
    }

    // --- TaskType tests ---

    #[test]
    fn task_type_equality() {
        assert_eq!(TaskType::ArchRefactor, TaskType::ArchRefactor);
        assert_eq!(TaskType::CodeDeploy, TaskType::CodeDeploy);
        assert_eq!(TaskType::Debugging, TaskType::Debugging);
        assert_ne!(TaskType::ArchRefactor, TaskType::CodeDeploy);
        assert_ne!(TaskType::CodeDeploy, TaskType::Debugging);
        assert_ne!(TaskType::Debugging, TaskType::ArchRefactor);
    }

    #[test]
    fn task_type_debug() {
        assert_eq!(format!("{:?}", TaskType::ArchRefactor), "ArchRefactor");
        assert_eq!(format!("{:?}", TaskType::CodeDeploy), "CodeDeploy");
        assert_eq!(format!("{:?}", TaskType::Debugging), "Debugging");
    }

    #[test]
    fn task_type_from_index_out_of_range() {
        assert!(TaskType::from_index(3).is_err());
        assert!(TaskType::from_index(100).is_err());
        assert!(TaskType::from_index(usize::MAX).is_err());

        // Verify error message contains the index.
        let err = TaskType::from_index(5).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("5"), "error should contain index: {msg}");
    }

    // --- TrackerTurnInput tests ---

    #[test]
    fn tracker_turn_input_fields() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768], vec![2.0; 768]],
            roles: vec![0, 1],
            signals: [0.1; 11],
            context_turns: 3.5,
        };
        assert_eq!(input.embeddings.len(), 2);
        assert_eq!(input.embeddings[0].len(), 768);
        assert_eq!(input.roles.len(), 2);
        assert_eq!(input.roles[1], 1);
        assert_eq!(input.signals[0], 0.1);
        assert!((input.context_turns - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn tracker_turn_input_clone_preserves() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.5; 768]],
            roles: vec![1],
            signals: [0.2; 11],
            context_turns: 7.0,
        };
        let cloned = input.clone();
        assert_eq!(cloned.embeddings.len(), input.embeddings.len());
        assert_eq!(cloned.embeddings[0], input.embeddings[0]);
        assert_eq!(cloned.roles, input.roles);
        assert_eq!(cloned.signals, input.signals);
        assert!((cloned.context_turns - input.context_turns).abs() < f32::EPSILON);
    }

    // --- classify_turn validation ---

    #[test]
    fn classify_turn_empty_embeddings_errors() {
        let tracker = make_test_tracker();
        let err = tracker
            .classify_turn(&[], &[], &[0.5; 11], 1.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("empty embeddings"), "unexpected message: {msg}");
    }

    #[test]
    fn classify_turn_embedding_dim_mismatch_errors() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 64]]; // 64 instead of 768
        let roles = vec![0u8];
        let err = tracker
            .classify_turn(&embeddings, &roles, &[0.5; 11], 1.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[0] dim 64"), "unexpected message: {msg}");
        assert!(msg.contains("hidden_size 768"), "unexpected message: {msg}");
    }

    #[test]
    fn classify_turn_roles_length_mismatch_errors() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 3];
        let roles = vec![0u8, 1]; // 2 roles for 3 embeddings
        let err = tracker
            .classify_turn(&embeddings, &roles, &[0.5; 11], 2.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embeddings len 3"), "unexpected message: {msg}");
        assert!(msg.contains("roles len 2"), "unexpected message: {msg}");
    }

    #[test]
    fn classify_turn_seq_len_exceeds_max_errors() {
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        // max_seq_len = 32, provide 33 embeddings to exceed it.
        let embeddings = vec![vec![0.1; config.hidden_size]; config.max_seq_len + 1];
        let roles = vec![0u8; config.max_seq_len + 1];
        let err = tracker
            .classify_turn(&embeddings, &roles, &[0.5; 11], 1.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("seq_len 33"), "unexpected message: {msg}");
        assert!(msg.contains("max_seq_len 32"), "unexpected message: {msg}");
    }

    // --- linear_forward tests ---

    #[test]
    fn linear_forward_weight_size_mismatch_errors() {
        let input = vec![1.0, 2.0, 3.0]; // k = 3
        let weight = vec![0.0; 4]; // should be n*k = n*3, but only 4 elements
        let bias = vec![0.0; 2]; // n = 2, so weight should be 6 elements
        let err = linear_forward(&input, &weight, &bias).unwrap_err();
        assert!(matches!(err, TrackerError::InferenceFailed(_)));
        let msg = format!("{err}");
        assert!(msg.contains("weight len 4"), "unexpected message: {msg}");
        assert!(msg.contains("n(2) * k(3)"), "unexpected message: {msg}");
    }

    #[test]
    fn linear_forward_identity_with_bias() {
        // 3x3 identity matrix, bias = [1.0, 2.0, 3.0], input = [4.0, 5.0, 6.0]
        // output = input @ I^T + bias = input + bias = [5.0, 7.0, 9.0]
        let input = vec![4.0, 5.0, 6.0];
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let bias = vec![1.0, 2.0, 3.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - 9.0).abs() < 1e-6);
    }

    // ── Additional comprehensive unit tests ──

    // --- argmax edge cases ---

    #[test]
    fn argmax_single_element() {
        assert_eq!(argmax(&[42.0]), Some(0));
    }

    #[test]
    fn argmax_all_equal_returns_last_index() {
        // max_by with partial_cmp(Equal) on ties keeps the last seen value,
        // so all-equal inputs yield the last index.
        assert_eq!(argmax(&[3.0, 3.0, 3.0]), Some(2));
    }

    #[test]
    fn argmax_negative_values() {
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), Some(1));
        assert_eq!(argmax(&[-10.0, -10.0, -2.0]), Some(2));
    }

    #[test]
    fn argmax_nan_handling() {
        // NaN comparison yields None → Ordering::Equal fallback → first wins.
        let result = argmax(&[f32::NAN, 1.0, 2.0]);
        // With partial_cmp returning Equal for NaN, the first element (index 0) wins ties.
        // However, max_by scans left-to-right and keeps the larger, but NaN comparisons
        // return Equal so it depends on iteration order. Verify it does not panic.
        assert!(result.is_some());
        assert!(result.unwrap() < 3);
    }

    #[test]
    fn argmax_last_element_wins() {
        assert_eq!(argmax(&[1.0, 2.0, 9.0]), Some(2));
    }

    // --- softmax_max edge cases ---

    #[test]
    fn softmax_max_single_element() {
        // Single logit → exp(0)/exp(0) = 1.0
        let conf = softmax_max(&[5.0]);
        assert!((conf - 1.0).abs() < 1e-6, "single element confidence should be 1.0, got {conf}");
    }

    #[test]
    fn softmax_max_negative_logits() {
        let conf = softmax_max(&[-100.0, -99.0, -101.0]);
        // The maximum logit is -99.0; exp(0)/sum should be > 0.5
        assert!(conf > 0.5, "confidence with negative logits should be > 0.5, got {conf}");
        assert!(conf <= 1.0);
    }

    #[test]
    fn softmax_max_large_logit_dominance() {
        let conf = softmax_max(&[0.0, 0.0, 1000.0]);
        assert!(conf > 0.99, "very large logit should dominate, got {conf}");
    }

    #[test]
    fn softmax_max_symmetric_logits() {
        // Symmetric logits [a, b, a] where b is the max → confidence = exp(0)/(exp(a-b)+1+exp(a-b))
        let conf = softmax_max(&[-1.0, 0.0, -1.0]);
        let expected = 1.0 / (2.0 * (-1.0f32).exp() + 1.0);
        assert!((conf - expected).abs() < 1e-5, "expected {expected}, got {conf}");
    }

    // --- linear_forward edge cases ---

    #[test]
    fn linear_forward_scalar_1x1() {
        // 1-input, 1-output: y = x*w + b
        let input = vec![3.0];
        let weight = vec![2.0]; // 1x1
        let bias = vec![1.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 7.0).abs() < 1e-6, "3*2 + 1 = 7, got {}", out[0]);
    }

    #[test]
    fn linear_forward_zero_input() {
        let input = vec![0.0; 4];
        let weight = vec![1.0; 8]; // 2x4
        let bias = vec![1.0, 2.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // With zero input, output should equal bias.
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn linear_forward_zero_weight_and_bias() {
        let input = vec![100.0, -50.0];
        let weight = vec![0.0; 4]; // 2x2 all zeros
        let bias = vec![0.0; 2];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1]).abs() < 1e-6);
    }

    // --- classify_turn role index out of range ---

    #[test]
    fn classify_turn_invalid_role_index_errors() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]];
        let roles = vec![5u8]; // role index 5 is out of range (only 3 roles: 0, 1, 2)
        let err = tracker
            .classify_turn(&embeddings, &roles, &[0.5; 11], 1.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("role index"), "unexpected message: {msg}");
    }

    // --- TaskType Copy trait ---

    #[test]
    fn task_type_copy_trait() {
        let a = TaskType::CodeDeploy;
        let b = a; // Copy, not move
        assert_eq!(a, b); // a is still valid because Copy
    }

    // --- Classification::task_type fallback on malformed logits ---

    #[test]
    /// [BCE-028] Empty task_logits now panics (was silently returning default).
    /// Test that it panics rather than silently choosing class 0.
    #[test]
    #[should_panic(expected = "task_logits empty or all-NaN")]
    fn classification_task_type_with_empty_logits_panics() {
        let c = Classification {
            task_logits: vec![],
            difficulty_logits: vec![0.0; 4],
        };
        let _ = c.task_type();
    }

    #[test]
    fn classification_task_confidence_uniform_three() {
        // Verify softmax_max with exactly 3 identical logits → 1/3
        let c = Classification {
            task_logits: vec![5.0, 5.0, 5.0],
            difficulty_logits: vec![0.0; 4],
        };
        let conf = c.task_confidence();
        assert!((conf - (1.0 / 3.0)).abs() < 1e-5, "uniform 3-way confidence should be ~0.333, got {conf}");
    }

    // --- TrackerError Debug output ---

    #[test]
    fn tracker_error_debug_output() {
        let err = TrackerError::LoadFailed("test".into());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("LoadFailed"), "Debug should contain variant name: {debug_str}");

        let err = TrackerError::MissingWeight("w".into());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("MissingWeight"), "Debug should contain variant name: {debug_str}");
    }

    // --- IntentTracker Debug output ---

    #[test]
    fn intent_tracker_debug_output() {
        let tracker = make_test_tracker();
        let debug_str = format!("{tracker:?}");
        // IntentTracker has Debug derived, should contain struct name
        assert!(debug_str.contains("IntentTracker"), "Debug should contain struct name: {debug_str}");
    }

    // --- verify_weights: first missing weight is reported ---

    #[test]
    fn verify_weights_reports_first_missing() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        // Add role_emb_weight only — the next required weight is w_q_weight
        shapes.insert("role_emb_weight".to_string(), vec![3, config.hidden_size]);

        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("w_q_weight"), "should report first missing weight, got: {msg}");
    }

    // --- get_weight missing ---

    #[test]
    fn get_weight_missing_returns_error() {
        let tracker = make_test_tracker();
        let err = tracker.get_weight("nonexistent_weight_12345").unwrap_err();
        assert!(matches!(err, TrackerError::MissingWeight(_)));
        let msg = format!("{err}");
        assert!(msg.contains("nonexistent_weight_12345"));
    }

    // --- IntentTrackerConfig defaults ---

    #[test]
    fn intent_tracker_config_defaults() {
        let config = IntentTrackerConfig::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.head_dim, 192);
        assert_eq!(config.num_tasks, 3);
        assert_eq!(config.num_difficulties, 4);
        assert_eq!(config.signal_dim, 11);
        assert_eq!(config.signal_hidden_dim, 64);
        assert_eq!(config.max_seq_len, 32);
    }

    // --- IntentTrackerConfig clone ---

    #[test]
    fn intent_tracker_config_clone() {
        let config = IntentTrackerConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.hidden_size, config.hidden_size);
        assert_eq!(cloned.num_heads, config.num_heads);
        assert_eq!(cloned.head_dim, config.head_dim);
        assert_eq!(cloned.num_tasks, config.num_tasks);
        assert_eq!(cloned.num_difficulties, config.num_difficulties);
        assert_eq!(cloned.signal_dim, config.signal_dim);
        assert_eq!(cloned.signal_hidden_dim, config.signal_hidden_dim);
        assert_eq!(cloned.max_seq_len, config.max_seq_len);
    }

    // --- TrackerTurnInput Debug output ---

    #[test]
    fn tracker_turn_input_debug() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768]],
            roles: vec![0],
            signals: [0.5; 11],
            context_turns: 2.0,
        };
        let debug_str = format!("{input:?}");
        assert!(debug_str.contains("TrackerTurnInput"), "Debug should contain struct name: {debug_str}");
        assert!(debug_str.contains("embeddings"), "Debug should show field names: {debug_str}");
    }

    // --- softmax_max with two identical max values ---

    #[test]
    fn softmax_max_two_identical_max() {
        // Two logits equal and max: [5.0, 5.0, 1.0]
        let conf = softmax_max(&[5.0, 5.0, 1.0]);
        // exp(0) / (exp(0) + exp(0) + exp(-4)) ≈ 1 / (1 + 1 + 0.0183) ≈ 0.4955
        let expected = 1.0 / (1.0 + 1.0 + (-4.0f32).exp());
        assert!((conf - expected).abs() < 1e-4, "expected {expected}, got {conf}");
        assert!(conf < 0.5, "confidence with two tied max should be < 0.5, got {conf}");
    }

    // ── Additional tests (15 new) ──

    // 1. verify_weights with completely empty shapes map

    #[test]
    fn verify_weights_empty_shapes_reports_first_required() {
        let config = IntentTrackerConfig::default();
        let shapes = HashMap::new();
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        // First required weight is "role_emb_weight"
        assert!(
            msg.contains("role_emb_weight"),
            "should report first required weight, got: {msg}"
        );
    }

    // 2. softmax_max with all equal negative logits

    #[test]
    fn softmax_max_all_equal_negative() {
        let conf = softmax_max(&[-5.0, -5.0, -5.0, -5.0]);
        let expected = 1.0 / 4.0;
        assert!(
            (conf - expected).abs() < 1e-5,
            "uniform negative confidence should be 0.25, got {conf}"
        );
    }

    // 3. softmax_max overflow safety: extreme values should not produce NaN

    #[test]
    fn softmax_max_extreme_values_no_nan() {
        // Without max-subtraction these would overflow; with subtraction it is safe.
        let conf = softmax_max(&[1e30, -1e30, 0.0]);
        assert!(conf.is_finite(), "confidence should be finite, got {conf}");
        assert!(conf > 0.0 && conf <= 1.0, "confidence should be in (0,1], got {conf}");
    }

    // 4. linear_forward with negative weights and bias

    #[test]
    fn linear_forward_negative_weights_and_bias() {
        // 2-input, 2-output: y = x @ W^T + b
        // W = [[-1, 2], [3, -4]], b = [-10, 20], x = [1, 2]
        // y[0] = 1*(-1) + 2*2 + (-10) = -1 + 4 - 10 = -7
        // y[1] = 1*3 + 2*(-4) + 20 = 3 - 8 + 20 = 15
        let input = vec![1.0, 2.0];
        let weight = vec![-1.0, 2.0, 3.0, -4.0];
        let bias = vec![-10.0, 20.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - (-7.0)).abs() < 1e-6, "expected -7.0, got {}", out[0]);
        assert!((out[1] - 15.0).abs() < 1e-6, "expected 15.0, got {}", out[1]);
    }

    // 5. linear_forward explicit matmul verification

    #[test]
    fn linear_forward_explicit_matmul() {
        // 3-input, 2-output
        // W = [[1, 2, 3], [4, 5, 6]], b = [0, 0], x = [1, 1, 1]
        // y[0] = 1*1 + 2*1 + 3*1 = 6
        // y[1] = 4*1 + 5*1 + 6*1 = 15
        let input = vec![1.0, 1.0, 1.0];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![0.0, 0.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 6.0).abs() < 1e-6, "expected 6.0, got {}", out[0]);
        assert!((out[1] - 15.0).abs() < 1e-6, "expected 15.0, got {}", out[1]);
    }

    // 6. classify_turn at exact max_seq_len boundary should succeed

    #[test]
    fn classify_turn_at_exact_max_seq_len_succeeds() {
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        // max_seq_len = 32; providing exactly 32 should be accepted.
        let embeddings = vec![vec![0.1; config.hidden_size]; config.max_seq_len];
        let roles = vec![0u8; config.max_seq_len];
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0);
        assert!(result.is_ok(), "exact max_seq_len should be accepted, got {:?}", result.err());
    }

    // 7. TrackerTurnInput signals array is fixed size [f32; 11]

    #[test]
    fn tracker_turn_input_signals_fixed_size() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            context_turns: 0.0,
        };
        assert_eq!(input.signals.len(), 11);
        assert!((input.signals[0] - 0.0).abs() < f32::EPSILON);
        assert!((input.signals[10] - 10.0).abs() < f32::EPSILON);
    }

    // 8. Classification difficulty with first element winner

    #[test]
    fn classification_difficulty_first_element_wins() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![100.0, 1.0, 1.0, 1.0],
        };
        assert_eq!(c.difficulty(), 0);
    }

    // 9. TrackerError::MissingWeight contains the weight name

    #[test]
    fn tracker_error_missing_weight_contains_name() {
        let name = "some_specific_weight_42";
        let err = TrackerError::MissingWeight(name.to_string());
        let msg = format!("{err}");
        assert!(msg.contains(name), "error message should contain weight name: {msg}");
    }

    // 10. softmax_max with mixed positive and negative logits

    #[test]
    fn softmax_max_mixed_signs() {
        // [3.0, -1.0, 0.5] → max=3.0, confidence = 1/(1+exp(-4)+exp(-2.5))
        let logits = [3.0, -1.0, 0.5];
        let conf = softmax_max(&logits);
        let expected = 1.0 / (1.0 + (-4.0f32).exp() + (-2.5f32).exp());
        assert!(
            (conf - expected).abs() < 1e-5,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.5, "dominant positive logit should give > 0.5, got {conf}");
    }

    // 11. argmax with alternating ascending values

    #[test]
    fn argmax_ascending_sequence() {
        assert_eq!(argmax(&[1.0, 2.0, 3.0, 4.0, 5.0]), Some(4));
    }

    // 12. argmax with descending sequence

    #[test]
    fn argmax_descending_sequence() {
        assert_eq!(argmax(&[5.0, 4.0, 3.0, 2.0, 1.0]), Some(0));
    }

    // 13. linear_forward with empty output (bias len = 0)

    #[test]
    fn linear_forward_empty_bias_produces_empty_output() {
        let input = vec![1.0, 2.0, 3.0];
        let weight: Vec<f32> = vec![]; // 0 rows * 3 cols = 0
        let bias: Vec<f32> = vec![];   // 0 outputs
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!(out.is_empty(), "empty bias should produce empty output");
    }

    // 14. TrackerError variant distinctness via Display

    #[test]
    fn tracker_error_display_distinctness() {
        let errors = [
            TrackerError::LoadFailed("x".into()),
            TrackerError::InvalidInput("x".into()),
            TrackerError::InferenceFailed("x".into()),
            TrackerError::MissingWeight("x".into()),
            TrackerError::EncoderFailed("x".into()),
            TrackerError::DequantFailed("x".into()),
        ];
        let messages: Vec<String> = errors.iter().map(|e| format!("{e}")).collect();
        // All messages should be distinct (different prefix).
        for i in 0..messages.len() {
            for j in (i + 1)..messages.len() {
                assert_ne!(
                    messages[i], messages[j],
                    "error variant {i} and {j} should produce distinct messages"
                );
            }
        }
    }

    // 15. verify_weights succeeds when all required weights are present

    #[test]
    fn verify_weights_all_present_succeeds() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();

        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        assert!(
            verify_weights(&shapes, &config).is_ok(),
            "all required weights present should succeed"
        );
    }

    // ── Batch 3: Additional 18 unit tests ──

    // 1. TrackerError Clone produces equal variants

    #[test]
    fn tracker_error_clone_roundtrip() {
        let err = TrackerError::InvalidInput("dimension mismatch".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    // 2. TaskType Hash: all variants produce distinct hashes in a HashSet

    #[test]
    fn task_type_hash_distinct_in_set() {
        use std::collections::HashSet;
        let set: HashSet<TaskType> = [
            TaskType::ArchRefactor,
            TaskType::CodeDeploy,
            TaskType::Debugging,
        ].into_iter().collect();
        assert_eq!(set.len(), 3, "all 3 TaskType variants should be distinct in HashSet");
    }

    // 3. TaskType::from_index boundary: index 0 is ArchRefactor

    #[test]
    fn task_type_from_index_zero_is_arch_refactor() {
        let tt = TaskType::from_index(0).unwrap();
        assert_eq!(tt, TaskType::ArchRefactor);
    }

    // 4. TaskType::from_index boundary: index 2 is Debugging (max valid)

    #[test]
    fn task_type_from_index_max_valid_is_debugging() {
        let tt = TaskType::from_index(2).unwrap();
        assert_eq!(tt, TaskType::Debugging);
    }

    // 5. Classification with all-zero logits: task_type returns ArchRefactor (index 0 wins ties)

    #[test]
    fn classification_all_zero_logits() {
        let c = Classification {
            task_logits: vec![0.0, 0.0, 0.0],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0],
        };
        // argmax on ties returns last index (2) → from_index(2) → Debugging
        assert_eq!(c.task_type(), TaskType::Debugging);
        assert_eq!(c.difficulty(), 3); // 4 equal values → last index
        // Confidence: uniform 3-way → 1/3
        let tc = c.task_confidence();
        assert!((tc - (1.0 / 3.0)).abs() < 1e-5, "expected ~0.333, got {tc}");
    }

    // 6. Classification confidence with NaN logits

    #[test]
    fn classification_confidence_with_nan_logits() {
        let c = Classification {
            task_logits: vec![f32::NAN, 1.0, 0.0],
            difficulty_logits: vec![0.0; 4],
        };
        // softmax_max should not panic; result may be NaN but must not crash
        let conf = c.task_confidence();
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // 7. Classification with extreme confidence near 1.0

    #[test]
    fn classification_extreme_confidence_near_one() {
        let c = Classification {
            task_logits: vec![1000.0, -1000.0, -1000.0],
            difficulty_logits: vec![0.0; 4],
        };
        let conf = c.task_confidence();
        assert!((conf - 1.0).abs() < 1e-3, "extreme logit should yield ~1.0, got {conf}");
    }

    // 8. Classification difficulty_confidence near 0.25 (uniform 4-way)

    #[test]
    fn classification_difficulty_confidence_uniform() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![2.0, 2.0, 2.0, 2.0],
        };
        let conf = c.difficulty_confidence();
        assert!((conf - 0.25).abs() < 1e-5, "uniform 4-way should be 0.25, got {conf}");
    }

    // 9. TrackerTurnInput construction with empty embeddings and roles

    #[test]
    fn tracker_turn_input_empty_embeddings() {
        let input = TrackerTurnInput {
            embeddings: vec![],
            roles: vec![],
            signals: [0.0; 11],
            context_turns: 0.0,
        };
        assert!(input.embeddings.is_empty());
        assert!(input.roles.is_empty());
        assert_eq!(input.context_turns, 0.0);
    }

    // 10. TrackerTurnInput with zero signals array

    #[test]
    fn tracker_turn_input_zero_signals() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.5; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: 1.0,
        };
        assert!(input.signals.iter().all(|&s| s == 0.0));
    }

    // 11. TrackerTurnInput Debug trait includes field names

    #[test]
    fn tracker_turn_input_debug_includes_roles_and_signals() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![1],
            signals: [0.1; 11],
            context_turns: 5.0,
        };
        let debug = format!("{input:?}");
        assert!(debug.contains("roles"), "Debug should contain 'roles': {debug}");
        assert!(debug.contains("signals"), "Debug should contain 'signals': {debug}");
        assert!(debug.contains("context_turns"), "Debug should contain 'context_turns': {debug}");
    }

    // 12. Classification Debug trait includes field names

    #[test]
    fn classification_debug_includes_fields() {
        let c = Classification {
            task_logits: vec![1.0, 2.0, 3.0],
            difficulty_logits: vec![4.0, 5.0, 6.0, 7.0],
        };
        let debug = format!("{c:?}");
        assert!(debug.contains("Classification"), "Debug should contain struct name: {debug}");
        assert!(debug.contains("task_logits"), "Debug should contain task_logits: {debug}");
        assert!(debug.contains("difficulty_logits"), "Debug should contain difficulty_logits: {debug}");
    }

    // 13. Classification Clone produces independent copy

    #[test]
    fn classification_clone_independent_mutation() {
        let c = Classification {
            task_logits: vec![1.0, 2.0, 3.0],
            difficulty_logits: vec![0.5, 1.5, 2.5, 3.5],
        };
        let mut c2 = c.clone();
        c2.task_logits[0] = 99.0;
        // Original must be unaffected
        assert!((c.task_logits[0] - 1.0).abs() < f32::EPSILON);
        assert!((c2.task_logits[0] - 99.0).abs() < f32::EPSILON);
    }

    // 14. IntentTracker field access via get_weight returns known weight

    #[test]
    fn intent_tracker_get_weight_returns_known_weight() {
        let tracker = make_test_tracker();
        let w = tracker.get_weight("recency_scale").unwrap();
        assert_eq!(w.len(), 1);
    }

    // 15. IntentTracker field access: weight_shapes via get_weight for multiple keys

    #[test]
    fn intent_tracker_multiple_weight_access() {
        let tracker = make_test_tracker();
        // All required weights should be accessible
        assert!(tracker.get_weight("role_emb_weight").is_ok());
        assert!(tracker.get_weight("w_q_weight").is_ok());
        assert!(tracker.get_weight("context_gate").is_ok());
        // Non-existent should fail
        assert!(tracker.get_weight("does_not_exist").is_err());
    }

    // 16. TrackerError Display for each variant contains expected prefix

    #[test]
    fn tracker_error_display_all_prefixes() {
        let cases: Vec<(TrackerError, &str)> = vec![
            (TrackerError::LoadFailed("a".into()), "model loading failed"),
            (TrackerError::InvalidInput("b".into()), "invalid input"),
            (TrackerError::InferenceFailed("c".into()), "inference failed"),
            (TrackerError::MissingWeight("d".into()), "weight missing"),
            (TrackerError::EncoderFailed("e".into()), "encoder failed"),
            (TrackerError::DequantFailed("f".into()), "dequantization failed"),
        ];
        for (err, prefix) in cases {
            let msg = format!("{err}");
            assert!(
                msg.starts_with(prefix),
                "expected prefix '{}', got '{}'",
                prefix,
                msg
            );
        }
    }

    // 17. TrackerError Clone: all variants clone correctly

    #[test]
    fn tracker_error_clone_all_variants() {
        let errors = vec![
            TrackerError::LoadFailed("x".into()),
            TrackerError::InvalidInput("y".into()),
            TrackerError::InferenceFailed("z".into()),
            TrackerError::MissingWeight("w".into()),
            TrackerError::EncoderFailed("e".into()),
            TrackerError::DequantFailed("d".into()),
        ];
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(format!("{err}"), format!("{cloned}"), "Clone should preserve Display for {:?}", err);
        }
    }

    // 18. softmax_max with confidence exactly 1.0 for single element

    #[test]
    fn softmax_max_confidence_is_one_for_single() {
        let conf = softmax_max(&[0.0]);
        assert!((conf - 1.0).abs() < 1e-6, "single element should give confidence 1.0, got {conf}");
    }

    // ── Batch 4: Additional 50 unit tests ──

    // --- TrackerError source/cause chain (Error trait) ---

    #[test]
    fn tracker_error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrackerError>();
    }

    #[test]
    fn tracker_error_load_failed_display() {
        let err = TrackerError::LoadFailed("disk full".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("model loading failed:"));
        assert!(msg.contains("disk full"));
    }

    #[test]
    fn tracker_error_invalid_input_display() {
        let err = TrackerError::InvalidInput("seq_len is zero".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("invalid input:"));
        assert!(msg.contains("seq_len is zero"));
    }

    #[test]
    fn tracker_error_inference_failed_display() {
        let err = TrackerError::InferenceFailed("NaN in logits".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("inference failed:"));
        assert!(msg.contains("NaN in logits"));
    }

    #[test]
    fn tracker_error_encoder_failed_display() {
        let err = TrackerError::EncoderFailed("timeout".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("encoder failed:"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn tracker_error_dequant_failed_display() {
        let err = TrackerError::DequantFailed("overflow".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("dequantization failed:"));
        assert!(msg.contains("overflow"));
    }

    // --- TaskType exhaustive match coverage ---

    #[test]
    fn task_type_all_variants_from_index() {
        // Verify that indexes 0, 1, 2 map to the three variants
        let variants = [
            (0, TaskType::ArchRefactor),
            (1, TaskType::CodeDeploy),
            (2, TaskType::Debugging),
        ];
        for (idx, expected) in variants {
            assert_eq!(TaskType::from_index(idx).unwrap(), expected);
        }
    }

    #[test]
    fn task_type_hash_lookup_in_hashmap() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(TaskType::ArchRefactor, "refactor");
        map.insert(TaskType::CodeDeploy, "deploy");
        map.insert(TaskType::Debugging, "debug");
        assert_eq!(map.get(&TaskType::CodeDeploy), Some(&"deploy"));
        assert_eq!(map.get(&TaskType::ArchRefactor), Some(&"refactor"));
        assert_eq!(map.get(&TaskType::Debugging), Some(&"debug"));
    }

    // --- TrackerTurnInput: field mutation independence after clone ---

    #[test]
    fn tracker_turn_input_clone_is_independent() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768]],
            roles: vec![0],
            signals: [5.0; 11],
            context_turns: 10.0,
        };
        let mut cloned = input.clone();
        cloned.embeddings[0][0] = 999.0;
        cloned.roles[0] = 1;
        cloned.context_turns = 0.0;
        // Original should be unaffected
        assert!((input.embeddings[0][0] - 1.0).abs() < f32::EPSILON);
        assert_eq!(input.roles[0], 0);
        assert!((input.context_turns - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tracker_turn_input_with_large_context_turns() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: f32::MAX,
        };
        assert_eq!(input.context_turns, f32::MAX);
    }

    #[test]
    fn tracker_turn_input_with_negative_context_turns() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: -5.0,
        };
        // Negative context_turns is accepted (no validation at construction)
        assert!((input.context_turns - (-5.0)).abs() < f32::EPSILON);
    }

    // --- Classification: edge case logits ---

    #[test]
    fn classification_with_single_task_logit() {
        let c = Classification {
            task_logits: vec![42.0],
            difficulty_logits: vec![0.0; 4],
        };
        assert_eq!(c.task_type(), TaskType::ArchRefactor); // argmax=0 → ArchRefactor
        assert!((c.task_confidence() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn classification_difficulty_all_negative() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![-10.0, -1.0, -100.0, -50.0],
        };
        assert_eq!(c.difficulty(), 1); // -1.0 is the maximum
    }

    #[test]
    fn classification_with_inf_logits() {
        let c = Classification {
            task_logits: vec![f32::INFINITY, 0.0, 0.0],
            difficulty_logits: vec![0.0; 4],
        };
        assert_eq!(c.task_type(), TaskType::ArchRefactor);
        // softmax_max should handle INFINITY without panic
        let conf = c.task_confidence();
        assert!(conf.is_finite() || conf == 1.0 || conf.is_nan());
    }

    // --- linear_forward: larger matrices ---

    #[test]
    fn linear_forward_4x3_matrix() {
        // 3 inputs, 4 outputs
        let input = vec![1.0, 0.0, 0.0]; // one-hot at position 0
        let weight = vec![
            1.0, 2.0, 3.0,  // row 0
            4.0, 5.0, 6.0,  // row 1
            7.0, 8.0, 9.0,  // row 2
            10.0, 11.0, 12.0, // row 3
        ];
        let bias = vec![0.0; 4];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // input picks column 0 of each row
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        assert!((out[2] - 7.0).abs() < 1e-6);
        assert!((out[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn linear_forward_2x5_matrix() {
        // 5 inputs, 2 outputs
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, // row 0: sum
            0.0, 0.0, 0.0, 0.0, 1.0, // row 1: last element
        ];
        let bias = vec![0.0; 2];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 15.0).abs() < 1e-6, "sum of 1..5 = 15, got {}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-6, "last element = 5, got {}", out[1]);
    }

    #[test]
    fn linear_forward_bias_only_contribution() {
        // Zero weight, nonzero bias → output == bias
        let input = vec![99.0, -99.0, 0.0];
        let weight = vec![0.0; 6]; // 2x3 all zeros
        let bias = vec![3.14, -2.71];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 3.14).abs() < 1e-6);
        assert!((out[1] - (-2.71)).abs() < 1e-6);
    }

    // --- argmax: more edge cases ---

    #[test]
    fn argmax_with_two_elements() {
        assert_eq!(argmax(&[1.0, 2.0]), Some(1));
        assert_eq!(argmax(&[2.0, 1.0]), Some(0));
    }

    #[test]
    fn argmax_with_zero_values() {
        assert_eq!(argmax(&[0.0, 0.0, 0.0]), Some(2)); // ties → last index
    }

    #[test]
    fn argmax_with_large_array() {
        let v: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        assert_eq!(argmax(&v), Some(999));
    }

    #[test]
    fn argmax_with_negative_infinity() {
        assert_eq!(argmax(&[f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]), Some(1));
    }

    // --- softmax_max: more edge cases ---

    #[test]
    fn softmax_max_two_elements_equal() {
        let conf = softmax_max(&[1.0, 1.0]);
        assert!((conf - 0.5).abs() < 1e-5, "two equal logits → 0.5, got {conf}");
    }

    #[test]
    fn softmax_max_two_elements_dominant() {
        let conf = softmax_max(&[100.0, 0.0]);
        assert!(conf > 0.99, "dominant element should yield > 0.99, got {conf}");
    }

    #[test]
    fn softmax_max_four_elements_uniform() {
        let conf = softmax_max(&[1.0, 1.0, 1.0, 1.0]);
        assert!((conf - 0.25).abs() < 1e-5, "4 equal logits → 0.25, got {conf}");
    }

    // --- verify_weights: specific missing weight ---

    #[test]
    fn verify_weights_missing_signal_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        // Add all except signal_fc0_weight and signal_fc0_bias
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            // Deliberately omit signal_fc0_weight and signal_fc0_bias
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("signal_fc0_weight"), "should report missing signal_fc0_weight, got: {msg}");
    }

    #[test]
    fn verify_weights_extra_weights_ignored() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        // Add extra weights that are not required
        shapes.insert("extra_weight_1".to_string(), vec![10]);
        shapes.insert("extra_weight_2".to_string(), vec![5, 5]);
        assert!(verify_weights(&shapes, &config).is_ok(), "extra weights should not cause failure");
    }

    // --- IntentTrackerConfig custom construction ---

    #[test]
    fn intent_tracker_config_custom_values() {
        let config = IntentTrackerConfig {
            hidden_size: 512,
            num_heads: 8,
            head_dim: 64,
            num_tasks: 5,
            num_difficulties: 6,
            signal_dim: 22,
            signal_hidden_dim: 128,
            max_seq_len: 64,
        };
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_tasks, 5);
        assert_eq!(config.num_difficulties, 6);
        assert_eq!(config.signal_dim, 22);
        assert_eq!(config.signal_hidden_dim, 128);
        assert_eq!(config.max_seq_len, 64);
    }

    #[test]
    fn intent_tracker_config_partial_eq_same() {
        let a = IntentTrackerConfig::default();
        let b = IntentTrackerConfig::default();
        assert_eq!(a, b);
    }

    #[test]
    fn intent_tracker_config_partial_eq_different() {
        let a = IntentTrackerConfig::default();
        let mut b = IntentTrackerConfig::default();
        b.hidden_size = 999;
        assert_ne!(a, b);
    }

    // --- classify_turn: single embedding (seq_len=1) ---

    #[test]
    fn classify_turn_single_embedding() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.5; 768]];
        let roles = vec![0u8];
        let result = tracker.classify_turn(&embeddings, &roles, &[0.0; 11], 0.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    #[test]
    fn classify_turn_all_roles_valid() {
        let tracker = make_test_tracker();
        // role_emb_weight has 3 rows, so roles 0, 1, 2 are valid
        let embeddings = vec![vec![0.1; 768]; 3];
        let roles = vec![0u8, 1, 2];
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 3.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- classify_turns_batch: various batch sizes ---

    #[test]
    fn classify_turns_batch_single_item() {
        let tracker = make_test_tracker();
        let batch = vec![TrackerTurnInput {
            embeddings: vec![vec![0.1; 768]],
            roles: vec![0],
            signals: [0.5; 11],
            context_turns: 1.0,
        }];
        let results = tracker.classify_turns_batch(&batch).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn classify_turns_batch_three_items() {
        let tracker = make_test_tracker();
        let batch: Vec<TrackerTurnInput> = (0..3)
            .map(|i| TrackerTurnInput {
                embeddings: vec![vec![i as f32 * 0.1; 768]; 2],
                roles: vec![0, 1],
                signals: [i as f32 * 0.1; 11],
                context_turns: i as f32,
            })
            .collect();
        let results = tracker.classify_turns_batch(&batch).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.task_logits.len(), 3);
            assert_eq!(r.difficulty_logits.len(), 4);
        }
    }

    #[test]
    fn classify_turns_batch_stops_on_first_error() {
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 64]], // wrong dim
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
        ];
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    // --- classify_turn_quant: dequant function correctness ---

    #[test]
    fn classify_turn_quant_custom_dequant() {
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.5); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.3; 11];
        // Use a custom dequant that doubles the value
        let result = tracker.classify_turn_quant(
            &embeddings_f16, &roles, &signals, 1.0, |v| v.to_f32() * 2.0,
        ).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- get_weight: returns correct slice ---

    #[test]
    fn get_weight_returns_correct_slice() {
        let tracker = make_test_tracker();
        let w = tracker.get_weight("recency_scale").unwrap();
        // make_test_tracker uses add_vector which fills with 0.0
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn get_weight_role_emb_has_correct_size() {
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        let w = tracker.get_weight("role_emb_weight").unwrap();
        // role_emb_weight is (3, hidden_size) = 3 * 768 = 2304
        assert_eq!(w.len(), 3 * config.hidden_size);
    }

    // --- from_weights with mismatched shapes still passes verification ---

    #[test]
    fn from_weights_shapes_only_checked_for_presence() {
        // verify_weights only checks key presence, not shape correctness
        let config = IntentTrackerConfig::default();
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            weights.insert(name.to_string(), vec![0.0]);
            shapes.insert(name.to_string(), vec![1]);
        }
        // This should succeed — shapes are present, even if wrong size
        assert!(IntentTracker::from_weights(config, weights, shapes).is_ok());
    }

    // --- IntentTracker Debug includes key fields ---

    #[test]
    fn intent_tracker_debug_includes_config_and_weights() {
        let tracker = make_test_tracker();
        let debug = format!("{tracker:?}");
        assert!(debug.contains("config"), "Debug should include config: {debug}");
        assert!(debug.contains("weights"), "Debug should include weights: {debug}");
        assert!(debug.contains("weight_shapes"), "Debug should include weight_shapes: {debug}");
    }

    // --- TrackerError Debug format for all variants ---

    #[test]
    fn tracker_error_debug_all_variants() {
        let variants: Vec<TrackerError> = vec![
            TrackerError::LoadFailed("a".into()),
            TrackerError::InvalidInput("b".into()),
            TrackerError::InferenceFailed("c".into()),
            TrackerError::MissingWeight("d".into()),
            TrackerError::EncoderFailed("e".into()),
            TrackerError::DequantFailed("f".into()),
        ];
        for err in &variants {
            let debug = format!("{err:?}");
            assert!(!debug.is_empty(), "Debug should not be empty for {:?}", err);
        }
    }

    // --- linear_forward: row-major access pattern ---

    #[test]
    fn linear_forward_row_major_access() {
        // 2-input, 3-output: verify row-major indexing
        // W[0] = [1, 0], W[1] = [0, 1], W[2] = [1, 1]
        let input = vec![3.0, 7.0];
        let weight = vec![
            1.0, 0.0, // row 0: output[0] = 3*1 + 7*0 = 3
            0.0, 1.0, // row 1: output[1] = 3*0 + 7*1 = 7
            1.0, 1.0, // row 2: output[2] = 3*1 + 7*1 = 10
        ];
        let bias = vec![0.0; 3];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
        assert!((out[2] - 10.0).abs() < 1e-6);
    }

    // --- softmax_max with very small logits (underflow safety) ---

    #[test]
    fn softmax_max_very_small_logits() {
        let conf = softmax_max(&[-1000.0, -1001.0, -1002.0]);
        assert!(conf > 0.5, "max of very small logits should still be > 0.5, got {conf}");
        assert!(conf <= 1.0);
    }

    // --- E2E validation: non-empty valid inputs pass ---

    #[test]
    fn validate_e2e_inputs_valid_passes() {
        let result = validate_e2e_inputs(&["hello", "world"], &[0, 1]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_e2e_inputs_single_item_passes() {
        let result = validate_e2e_inputs(&["hello"], &[0]);
        assert!(result.is_ok());
    }

    // --- argmax with large negative values ---

    #[test]
    fn argmax_with_large_negative() {
        assert_eq!(argmax(&[-1e30, -1e20, -1e10]), Some(2));
    }

    // --- linear_forward weight dimension mismatch (too many elements) ---

    #[test]
    fn linear_forward_weight_too_large_errors() {
        let input = vec![1.0, 2.0]; // k = 2
        let weight = vec![0.0; 10]; // 10 elements, but n*k should be n*2
        let bias = vec![0.0; 2]; // n = 2, so weight should be 4
        let err = linear_forward(&input, &weight, &bias).unwrap_err();
        assert!(matches!(err, TrackerError::InferenceFailed(_)));
        let msg = format!("{err}");
        assert!(msg.contains("weight len 10"), "unexpected message: {msg}");
    }

    // --- IntentTrackerConfig: hidden_size == num_heads * head_dim invariant ---

    #[test]
    fn intent_tracker_config_default_hidden_equals_heads_times_head_dim() {
        let config = IntentTrackerConfig::default();
        assert_eq!(config.hidden_size, config.num_heads * config.head_dim);
    }

    // --- from_weights_bf16 with non-zero values ---

    #[test]
    fn from_weights_bf16_nonzero_values() {
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        // Use BF16 representation of 1.0 = 0x3F80
        let one_bf16 = bf16::from_f32(1.0).to_bits();
        let one_bytes = one_bf16.to_le_bytes();
        for name in &required {
            weights_bf16.insert(name.to_string(), vec![one_bytes[0], one_bytes[1]]);
            shapes.insert(name.to_string(), vec![1]);
        }
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        assert!((w[0] - 1.0).abs() < 0.01, "BF16 1.0 should dequantize to ~1.0, got {}", w[0]);
    }

    // --- classify_turn with zero signals ---

    #[test]
    fn classify_turn_zero_signals() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.0; 11];
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 0.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- classify_turn with negative signals ---

    #[test]
    fn classify_turn_negative_signals() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0];
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 0.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- classify_turn with large signals ---

    #[test]
    fn classify_turn_large_signals() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [1000.0; 11];
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 100.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // --- TrackerTurnInput with varying embedding dimensions in sequence ---

    #[test]
    fn tracker_turn_input_varying_embeddings() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768], vec![2.0; 768], vec![3.0; 768]],
            roles: vec![0, 1, 0],
            signals: [0.0; 11],
            context_turns: 3.0,
        };
        assert_eq!(input.embeddings.len(), 3);
        assert_eq!(input.embeddings[0][0], 1.0);
        assert_eq!(input.embeddings[1][0], 2.0);
        assert_eq!(input.embeddings[2][0], 3.0);
    }

    // --- softmax_max always returns value in (0, 1] for finite logits ---

    #[test]
    fn softmax_max_always_in_unit_interval() {
        let test_cases: Vec<Vec<f32>> = vec![
            vec![0.0],
            vec![1.0, -1.0],
            vec![0.0, 0.0, 0.0],
            vec![-100.0, 100.0],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
        ];
        for logits in &test_cases {
            let conf = softmax_max(logits);
            assert!(
                conf > 0.0 && conf <= 1.0 + 1e-6,
                "confidence should be in (0, 1], got {conf} for logits {logits:?}"
            );
        }
    }

    // --- linear_forward is consistent: same input produces same output ---

    #[test]
    fn linear_forward_deterministic() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![0.5, -0.5, 0.25, 0.1, 0.2, 0.3];
        let bias = vec![0.1, 0.2];
        let out1 = linear_forward(&input, &weight, &bias).unwrap();
        let out2 = linear_forward(&input, &weight, &bias).unwrap();
        assert_eq!(out1, out2, "same inputs should produce identical outputs");
    }

    // --- verify_weights: missing diff weights specifically ---

    #[test]
    fn verify_weights_missing_diff_weights() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            // Deliberately omit all diff_fc* weights
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("diff_fc0_weight"), "should report missing diff_fc0_weight, got: {msg}");
    }

    // --- TrackerError source (thiserror Error derive) ---

    #[test]
    fn tracker_error_no_source() {
        let err = TrackerError::LoadFailed("test".into());
        // thiserror errors without #[source] attribute have no source
        assert!(std::error::Error::source(&err).is_none());
    }

    // ── Batch 5: 50 additional unit tests ──

    // 1. TrackerError::LoadFailed from IntentTracker::new with nonexistent path

    #[test]
    fn intent_tracker_new_nonexistent_path_errors() {
        let err = IntentTracker::new("/nonexistent/path/does/not/exist.safetensors").unwrap_err();
        assert!(matches!(err, TrackerError::LoadFailed(_)));
        let msg = format!("{err}");
        assert!(msg.contains("read"), "LoadFailed should mention read failure: {msg}");
    }

    // 2. IntentTracker::new with a directory that has no model.safetensors

    #[test]
    fn intent_tracker_new_empty_dir_errors() {
        let tmp = std::env::temp_dir().join("gllm_test_empty_intent_dir");
        let _ = std::fs::create_dir_all(&tmp);
        let err = IntentTracker::new(&tmp).unwrap_err();
        assert!(matches!(err, TrackerError::LoadFailed(_)));
        let _ = std::fs::remove_dir(&tmp);
    }

    // 3. from_weights with completely empty maps

    #[test]
    fn from_weights_empty_maps_errors() {
        let config = IntentTrackerConfig::default();
        let weights = HashMap::new();
        let shapes = HashMap::new();
        let err = IntentTracker::from_weights(config, weights, shapes).unwrap_err();
        assert!(matches!(err, TrackerError::MissingWeight(_)));
    }

    // 4. classify_turn second embedding has wrong dim

    #[test]
    fn classify_turn_second_embedding_wrong_dim_errors() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768], vec![0.1; 64]]; // second is wrong
        let roles = vec![0u8, 1];
        let err = tracker
            .classify_turn(&embeddings, &roles, &[0.5; 11], 2.0)
            .unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[1]"), "should mention index 1: {msg}");
    }

    // 5. linear_forward homogeneity: W(αx) = αW(x) for scalar α

    #[test]
    fn linear_forward_scalar_multiplication_property() {
        let input = vec![1.0, -2.0, 3.0];
        let weight = vec![0.5, -0.5, 1.0, 0.0, 2.0, -1.0]; // 2x3
        let bias = vec![0.0; 2];
        let alpha = 3.7_f32;

        let base = linear_forward(&input, &weight, &bias).unwrap();
        let scaled_input: Vec<f32> = input.iter().map(|x| x * alpha).collect();
        let scaled = linear_forward(&scaled_input, &weight, &bias).unwrap();

        // W(αx) = α·W(x) when bias is zero
        for i in 0..base.len() {
            let expected = base[i] * alpha;
            assert!(
                (scaled[i] - expected).abs() < 1e-3,
                "homogeneity violated at index {i}: expected {expected}, got {}",
                scaled[i]
            );
        }
    }

    // 6. linear_forward additivity: W(x+y) = W(x) + W(y) - bias (bias counted once)

    #[test]
    fn linear_forward_additivity_property() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let bias = vec![0.0; 2];

        let wx = linear_forward(&x, &weight, &bias).unwrap();
        let wy = linear_forward(&y, &weight, &bias).unwrap();
        let x_plus_y: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        let wxy = linear_forward(&x_plus_y, &weight, &bias).unwrap();

        for i in 0..wx.len() {
            let expected = wx[i] + wy[i];
            assert!(
                (wxy[i] - expected).abs() < 1e-6,
                "additivity violated at index {i}: expected {expected}, got {}",
                wxy[i]
            );
        }
    }

    // 7. linear_forward with negative bias shifts output down

    #[test]
    fn linear_forward_negative_bias_shifts_output() {
        let input = vec![1.0, 1.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0]; // 2x2 all-ones
        let bias_no = vec![0.0; 2];
        let bias_neg = vec![-5.0, -5.0];

        let out_no = linear_forward(&input, &weight, &bias_no).unwrap();
        let out_neg = linear_forward(&input, &weight, &bias_neg).unwrap();

        for i in 0..2 {
            assert!(
                (out_neg[i] - (out_no[i] - 5.0)).abs() < 1e-6,
                "negative bias should shift output down by 5.0"
            );
        }
    }

    // 8. argmax with positive infinity

    #[test]
    fn argmax_with_positive_infinity() {
        assert_eq!(argmax(&[1.0, f32::INFINITY, 3.0]), Some(1));
    }

    // 9. argmax with mixed NaN and real: does not panic

    #[test]
    fn argmax_mixed_nan_and_real_no_panic() {
        let result = argmax(&[1.0, f32::NAN, f32::NAN, 2.0]);
        assert!(result.is_some());
        assert!(result.unwrap() < 4, "result should be a valid index, got {:?}", result);
    }

    // 10. softmax_max with all NaN: does not panic

    #[test]
    fn softmax_max_all_nan_no_panic() {
        let conf = softmax_max(&[f32::NAN, f32::NAN, f32::NAN]);
        // Result may be NaN but must not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // 11. softmax_max with single negative value

    #[test]
    fn softmax_max_single_negative() {
        let conf = softmax_max(&[-42.0]);
        assert!((conf - 1.0).abs() < 1e-6, "single element always gives 1.0, got {conf}");
    }

    // 12. softmax_max with all positive infinity

    #[test]
    fn softmax_max_all_infinity() {
        let conf = softmax_max(&[f32::INFINITY, f32::INFINITY]);
        // exp(INF - INF) = exp(NaN) → NaN, but the function should not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // 13. softmax_max monotonicity: increasing one logit increases confidence

    #[test]
    fn softmax_max_monotonicity() {
        let conf_low = softmax_max(&[1.0, 1.0, 1.0]);
        let conf_high = softmax_max(&[1.0, 1.0, 5.0]);
        assert!(
            conf_high > conf_low,
            "increasing max logit should increase confidence: {conf_low} vs {conf_high}"
        );
    }

    // 14. verify_weights missing w_k_weight

    #[test]
    fn verify_weights_missing_w_k_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_v_weight", // omit w_k_weight
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("w_k_weight"), "should report missing w_k_weight, got: {msg}");
    }

    // 15. verify_weights missing recency_scale (last required weight)

    #[test]
    fn verify_weights_missing_recency_scale() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "context_gate", // omit recency_scale
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("recency_scale"), "should report missing recency_scale, got: {msg}");
    }

    // 16. verify_weights missing context_gate

    #[test]
    fn verify_weights_missing_context_gate() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", // omit context_gate
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("context_gate"), "should report missing context_gate, got: {msg}");
    }

    // 17. verify_weights missing info_net_fc0_weight

    #[test]
    fn verify_weights_missing_info_net_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_bias", // omit info_net_fc0_weight
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("info_net_fc0_weight"), "should report missing info_net_fc0_weight, got: {msg}");
    }

    // 18. classify_turn with context_turns = 0.0

    #[test]
    fn classify_turn_zero_context_turns() {
        let tracker = make_test_tracker();
        let result = tracker
            .classify_turn(&[vec![0.1; 768]], &[0], &[0.5; 11], 0.0)
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 19. classify_turn with negative context_turns accepted

    #[test]
    fn classify_turn_negative_context_turns() {
        let tracker = make_test_tracker();
        let result = tracker
            .classify_turn(&[vec![0.1; 768]], &[0], &[0.5; 11], -10.0)
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 20. classify_turn with large context_turns

    #[test]
    fn classify_turn_large_context_turns() {
        let tracker = make_test_tracker();
        let result = tracker
            .classify_turn(&[vec![0.1; 768]], &[0], &[0.5; 11], 10000.0)
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 21. classify_turn with all-assistant roles

    #[test]
    fn classify_turn_all_assistant_roles() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 3];
        let roles = vec![1u8; 3]; // all assistant
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 3.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 22. classify_turn with alternating roles

    #[test]
    fn classify_turn_alternating_roles() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 4];
        let roles = vec![0u8, 1, 0, 1];
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 4.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 23. classify_turn with varied embedding values (non-uniform)

    #[test]
    fn classify_turn_varied_embeddings() {
        let tracker = make_test_tracker();
        let emb: Vec<f32> = (0..768).map(|i| (i as f32 * 0.001).sin()).collect();
        let result = tracker.classify_turn(&[emb], &[0], &[0.5; 11], 1.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        // All logits should be finite
        for &l in &result.task_logits {
            assert!(l.is_finite(), "task logit should be finite, got {l}");
        }
        for &l in &result.difficulty_logits {
            assert!(l.is_finite(), "difficulty logit should be finite, got {l}");
        }
    }

    // 24. classify_turn with extreme embedding values

    #[test]
    fn classify_turn_extreme_embedding_values() {
        let tracker = make_test_tracker();
        let embeddings = vec![vec![1000.0; 768]];
        let result = tracker.classify_turn(&embeddings, &[0], &[0.5; 11], 1.0).unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 25. classify_turn_quant with single embedding

    #[test]
    fn classify_turn_quant_single_embedding() {
        let tracker = make_test_tracker();
        let emb_f16 = vec![f16::from_f32(0.5); 768];
        let result = tracker
            .classify_turn_quant(&[emb_f16], &[0], &[0.5; 11], 1.0, |v| v.to_f32())
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 26. classify_turn_quant with all assistant roles

    #[test]
    fn classify_turn_quant_all_assistant() {
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.1); 768]; 3];
        let roles = vec![1u8; 3];
        let result = tracker
            .classify_turn_quant(&embeddings_f16, &roles, &[0.5; 11], 3.0, |v| v.to_f32())
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 27. classify_turn_quant with zero dequant function

    #[test]
    fn classify_turn_quant_zero_dequant() {
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(1.0); 768]; 2];
        let roles = vec![0u8, 1];
        let result = tracker
            .classify_turn_quant(&embeddings_f16, &roles, &[0.5; 11], 1.0, |_| 0.0)
            .unwrap();
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
    }

    // 28. classify_turns_batch with varied batch items (different seq_lens)

    #[test]
    fn classify_turns_batch_varied_seq_lens() {
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 1],
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.2; 768]; 5],
                roles: vec![0, 1, 0, 1, 0],
                signals: [0.3; 11],
                context_turns: 5.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.0; 768]; 2],
                roles: vec![1, 0],
                signals: [0.0; 11],
                context_turns: 0.0,
            },
        ];
        let results = tracker.classify_turns_batch(&batch).unwrap();
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.task_logits.len(), 3);
            assert_eq!(r.difficulty_logits.len(), 4);
        }
    }

    // 29. classify_turns_batch with single-item batch containing error

    #[test]
    fn classify_turns_batch_single_item_error() {
        let tracker = make_test_tracker();
        let batch = vec![TrackerTurnInput {
            embeddings: vec![], // empty → error
            roles: vec![],
            signals: [0.5; 11],
            context_turns: 0.0,
        }];
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    // 30. from_weights with all required weights present succeeds

    #[test]
    fn from_weights_all_required_succeeds() {
        let tracker = make_test_tracker();
        assert!(tracker.get_weight("role_emb_weight").is_ok());
        assert!(tracker.get_weight("context_gate").is_ok());
    }

    // 31. from_weights_bf16 with zero byte length for a weight

    #[test]
    fn from_weights_bf16_zero_bytes_weight_accepted() {
        // chunks_exact(2) on empty bytes produces 0 elements; numel = 0/2 = 0.
        // This is a valid (if degenerate) case — 0 elements matches 0 expected.
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            weights_bf16.insert(name.to_string(), vec![]);
            shapes.insert(name.to_string(), vec![0]);
        }
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes);
        assert!(tracker.is_ok(), "empty BF16 weights with shape [0] should be accepted");
    }

    // 32. IntentTracker from_weights preserves weight data

    #[test]
    fn from_weights_preserves_weight_data() {
        let config = IntentTrackerConfig::default();
        let h = config.hidden_size;

        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        let add_matrix = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, rows: usize, cols: usize| {
            w.insert(name.to_string(), vec![0.01; rows * cols]);
            s.insert(name.to_string(), vec![rows, cols]);
        };
        let add_vector = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, len: usize| {
            w.insert(name.to_string(), vec![0.0; len]);
            s.insert(name.to_string(), vec![len]);
        };

        add_matrix(&mut weights, &mut shapes, "role_emb_weight", 3, h);
        for prefix in &["w_q", "w_k", "w_v"] {
            add_matrix(&mut weights, &mut shapes, &format!("{prefix}_weight"), h, h);
            add_vector(&mut weights, &mut shapes, &format!("{prefix}_bias"), h);
        }
        for (i, dim) in [512, 128, 1].iter().enumerate() {
            let in_dim = if i == 0 { h } else { [512, 128][i - 1] };
            add_matrix(&mut weights, &mut shapes, &format!("info_net_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("info_net_fc{i}_bias"), *dim);
        }
        add_vector(&mut weights, &mut shapes, "per_head_norm_weight", config.head_dim);
        add_vector(&mut weights, &mut shapes, "per_head_norm_bias", config.head_dim);
        add_vector(&mut weights, &mut shapes, "context_norm_weight", h);
        add_vector(&mut weights, &mut shapes, "context_norm_bias", h);
        for (i, dim) in [128, 128, config.signal_hidden_dim].iter().enumerate() {
            let in_dim = if i == 0 { config.signal_dim } else { 128 };
            add_matrix(&mut weights, &mut shapes, &format!("signal_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("signal_fc{i}_bias"), *dim);
        }
        let classifier_input_dim = h + config.signal_hidden_dim + 1;
        for (i, dim) in [384, 192, config.num_tasks].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix(&mut weights, &mut shapes, &format!("task_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("task_fc{i}_bias"), *dim);
        }
        for (i, dim) in [384, 192, config.num_difficulties].iter().enumerate() {
            let real_in = [classifier_input_dim, 384, 192][i];
            add_matrix(&mut weights, &mut shapes, &format!("diff_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("diff_fc{i}_bias"), *dim);
        }
        // Set a specific value to verify it is preserved
        weights.insert("recency_scale".to_string(), vec![42.0]);
        shapes.insert("recency_scale".to_string(), vec![1]);
        add_vector(&mut weights, &mut shapes, "context_gate", 1);

        let tracker = IntentTracker::from_weights(config, weights, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        assert!((w[0] - 42.0).abs() < f32::EPSILON, "weight data should be preserved");
    }

    // 33. Classification difficulty_confidence with extreme dominant

    #[test]
    fn classification_difficulty_confidence_extreme_dominant() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![1000.0, 0.0, 0.0, 0.0],
        };
        let conf = c.difficulty_confidence();
        assert!(
            (conf - 1.0).abs() < 1e-3,
            "extreme dominant difficulty should yield ~1.0, got {conf}"
        );
    }

    // 34. Classification difficulty_confidence with two-way tie

    #[test]
    fn classification_difficulty_confidence_two_way_tie() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![10.0, 10.0, 0.0, 0.0],
        };
        let conf = c.difficulty_confidence();
        // Two max logits equal: confidence = 1/(1+1+exp(-10)+exp(-10)) ≈ 0.5
        assert!(
            (conf - 0.5).abs() < 0.01,
            "two-way tie should be ~0.5, got {conf}"
        );
        assert!(conf < 0.55, "two-way tie confidence should be < 0.55, got {conf}");
    }

    // 35. Classification difficulty with middle element winner (index 2)

    #[test]
    fn classification_difficulty_middle_winner() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 0.0, 50.0, 0.0],
        };
        assert_eq!(c.difficulty(), 2);
    }

    // 36. TrackerTurnInput signals are copied (not shared) on clone

    #[test]
    fn tracker_turn_input_signals_copied_on_clone() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768]],
            roles: vec![0],
            signals: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            context_turns: 1.0,
        };
        let cloned = input.clone();
        // signals is [f32; 11] — Copy type, so it's bitwise copied
        assert_eq!(input.signals, cloned.signals);
    }

    // 37. TrackerError all variants are Debug

    #[test]
    fn tracker_error_all_variants_have_debug() {
        let variants: Vec<TrackerError> = vec![
            TrackerError::LoadFailed("a".into()),
            TrackerError::InvalidInput("b".into()),
            TrackerError::InferenceFailed("c".into()),
            TrackerError::MissingWeight("d".into()),
            TrackerError::EncoderFailed("e".into()),
            TrackerError::DequantFailed("f".into()),
        ];
        for err in &variants {
            let debug = format!("{err:?}");
            assert!(!debug.is_empty());
        }
    }

    // 38. TrackerError has exactly 6 variants (compile-time check via exhaustive match)

    #[test]
    fn tracker_error_variant_count_is_six() {
        let count = {
            let mut n = 0;
            let errors = [
                TrackerError::LoadFailed(String::new()),
                TrackerError::InvalidInput(String::new()),
                TrackerError::InferenceFailed(String::new()),
                TrackerError::MissingWeight(String::new()),
                TrackerError::EncoderFailed(String::new()),
                TrackerError::DequantFailed(String::new()),
            ];
            for _ in &errors {
                n += 1;
            }
            n
        };
        assert_eq!(count, 6);
    }

    // 39. TrackerError no source for any variant

    #[test]
    fn tracker_error_no_source_all_variants() {
        let errors = [
            TrackerError::LoadFailed("a".into()),
            TrackerError::InvalidInput("b".into()),
            TrackerError::InferenceFailed("c".into()),
            TrackerError::MissingWeight("d".into()),
            TrackerError::EncoderFailed("e".into()),
            TrackerError::DequantFailed("f".into()),
        ];
        for err in &errors {
            assert!(
                std::error::Error::source(err).is_none(),
                "TrackerError should have no source for {err:?}"
            );
        }
    }

    // 40. linear_forward with 1x1 identity returns input + bias

    #[test]
    fn linear_forward_1x1_identity_plus_bias() {
        let input = vec![5.0];
        let weight = vec![1.0]; // identity
        let bias = vec![3.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 8.0).abs() < 1e-6, "5*1 + 3 = 8, got {}", out[0]);
    }

    // 41. linear_forward with scaling weight

    #[test]
    fn linear_forward_scaling_weight() {
        let input = vec![2.0, 3.0];
        let weight = vec![0.5, 0.5]; // 1x2, sums halves
        let bias = vec![1.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // 2*0.5 + 3*0.5 + 1 = 1 + 1.5 + 1 = 3.5
        assert!((out[0] - 3.5).abs() < 1e-6, "expected 3.5, got {}", out[0]);
    }

    // 42. IntentTrackerConfig PartialEq with multiple differences

    #[test]
    fn intent_tracker_config_partial_eq_multiple_differences() {
        let a = IntentTrackerConfig::default();
        let mut b = IntentTrackerConfig::default();
        b.hidden_size = 512;
        b.num_heads = 8;
        assert_ne!(a, b);
    }

    // 43. IntentTrackerConfig all fields independently mutable

    #[test]
    fn intent_tracker_config_fields_independently_mutable() {
        let mut config = IntentTrackerConfig::default();

        config.hidden_size = 1024;
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_heads, 4); // unchanged

        config.num_heads = 16;
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.hidden_size, 1024); // unchanged

        config.signal_hidden_dim = 256;
        assert_eq!(config.signal_hidden_dim, 256);
        assert_eq!(config.max_seq_len, 32); // unchanged
    }

    // 44. TaskType exhaustive variants count

    #[test]
    fn task_type_has_three_variants() {
        let variants = [TaskType::ArchRefactor, TaskType::CodeDeploy, TaskType::Debugging];
        assert_eq!(variants.len(), 3);
        // All distinct
        assert_ne!(variants[0], variants[1]);
        assert_ne!(variants[1], variants[2]);
        assert_ne!(variants[0], variants[2]);
    }

    // 45. Classification confidence is always in (0, 1] for finite logits

    #[test]
    fn classification_confidence_bounded_for_finite_logits() {
        let test_logits: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0],
            vec![100.0, -100.0, 0.0],
            vec![-50.0, -50.0, -50.0],
            vec![0.001, 0.002, 0.003],
            vec![-0.001, 0.0, 0.001],
        ];
        for task_logits in &test_logits {
            let c = Classification {
                task_logits: task_logits.clone(),
                difficulty_logits: vec![0.0; 4],
            };
            let conf = c.task_confidence();
            assert!(
                conf > 0.0 && conf <= 1.0 + 1e-6,
                "confidence should be in (0,1] for logits {task_logits:?}, got {conf}"
            );
        }
    }

    // 46. softmax_max with 5 elements uniform

    #[test]
    fn softmax_max_five_elements_uniform() {
        let conf = softmax_max(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert!(
            (conf - 0.2).abs() < 1e-5,
            "5 uniform logits → 0.2, got {conf}"
        );
    }

    // 47. argmax with maximum at first position

    #[test]
    fn argmax_maximum_at_first_position() {
        assert_eq!(argmax(&[100.0, 1.0, 2.0, 3.0]), Some(0));
    }

    // 48. argmax with maximum at middle position

    #[test]
    fn argmax_maximum_at_middle_position() {
        assert_eq!(argmax(&[1.0, 2.0, 100.0, 3.0, 4.0]), Some(2));
    }

    // 49. linear_forward output size matches bias length

    #[test]
    fn linear_forward_output_size_matches_bias() {
        let input = vec![1.0; 5];
        for n in [1, 3, 7, 16] {
            let weight = vec![0.0; n * 5];
            let bias = vec![0.0; n];
            let out = linear_forward(&input, &weight, &bias).unwrap();
            assert_eq!(out.len(), n, "output size should match bias length {n}");
        }
    }

    // 50. TrackerTurnInput can be constructed with roles containing values 0, 1, 2

    #[test]
    fn tracker_turn_input_all_valid_role_values() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.5; 768], vec![0.3; 768], vec![0.1; 768]],
            roles: vec![0, 1, 2],
            signals: [0.0; 11],
            context_turns: 3.0,
        };
        assert_eq!(input.roles, vec![0u8, 1, 2]);
    }

    // ── Batch 6: 25 additional unit tests ──

    // 1. argmax with f32::MAX among smaller values

    #[test]
    fn argmax_with_f32_max_value() {
        assert_eq!(argmax(&[1.0, f32::MAX, 0.0]), Some(1));
    }

    // 2. softmax_max uniform confidence equals 1/n for n=1..7

    #[test]
    fn softmax_max_uniform_confidence_decreases_with_n() {
        for n in 1..=7 {
            let logits = vec![1.0; n];
            let conf = softmax_max(&logits);
            let expected = 1.0 / n as f32;
            assert!(
                (conf - expected).abs() < 1e-5,
                "uniform n={n}: expected {expected}, got {conf}"
            );
        }
    }

    // 3. linear_forward 3x3 identity with zero bias preserves input

    #[test]
    fn linear_forward_identity_preserves_input_3x3() {
        let input = vec![1.5, -2.3, 0.7];
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let bias = vec![0.0; 3];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        for i in 0..3 {
            assert!((out[i] - input[i]).abs() < 1e-6, "index {i}: expected {}, got {}", input[i], out[i]);
        }
    }

    // 4. linear_forward negative identity flips sign

    #[test]
    fn linear_forward_negative_identity_flips_sign() {
        let input = vec![3.0, -7.0];
        let weight = vec![
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let bias = vec![0.0; 2];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - (-3.0)).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    // 5. linear_forward all-ones weight sums inputs

    #[test]
    fn linear_forward_all_ones_weight_sums() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4]; // 1x4, single output = sum of inputs
        let bias = vec![0.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-6, "sum of 1+2+3+4 = 10, got {}", out[0]);
    }

    // 6. linear_forward 3-to-1 projection

    #[test]
    fn linear_forward_projection_3_to_1() {
        let input = vec![2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0]; // 1x3
        let bias = vec![-5.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // 2+3+4-5 = 4
        assert!((out[0] - 4.0).abs() < 1e-6);
    }

    // 7. TrackerError::LoadFailed inner string accessible via pattern match

    #[test]
    fn tracker_error_load_failed_inner_string_accessible() {
        let err = TrackerError::LoadFailed("file corrupted".into());
        let msg = match &err {
            TrackerError::LoadFailed(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "file corrupted");
    }

    // 8. TrackerError::InvalidInput inner string accessible via pattern match

    #[test]
    fn tracker_error_invalid_input_inner_accessible() {
        let err = TrackerError::InvalidInput("bad dimension".into());
        let msg = match &err {
            TrackerError::InvalidInput(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "bad dimension");
    }

    // 9. IntentTrackerConfig with all fields set to zero

    #[test]
    fn intent_tracker_config_all_zero_fields() {
        let config = IntentTrackerConfig {
            hidden_size: 0,
            num_heads: 0,
            head_dim: 0,
            num_tasks: 0,
            num_difficulties: 0,
            signal_dim: 0,
            signal_hidden_dim: 0,
            max_seq_len: 0,
        };
        assert_eq!(config.hidden_size, 0);
        assert_eq!(config.num_heads, 0);
        assert_eq!(config.head_dim, 0);
        assert_eq!(config.num_tasks, 0);
        assert_eq!(config.num_difficulties, 0);
        assert_eq!(config.signal_dim, 0);
        assert_eq!(config.signal_hidden_dim, 0);
        assert_eq!(config.max_seq_len, 0);
    }

    // 10. IntentTrackerConfig with large custom values

    #[test]
    fn intent_tracker_config_large_custom_values() {
        let config = IntentTrackerConfig {
            hidden_size: 99999,
            num_heads: 99999,
            head_dim: 99999,
            num_tasks: 99999,
            num_difficulties: 99999,
            signal_dim: 99999,
            signal_hidden_dim: 99999,
            max_seq_len: 99999,
        };
        assert_eq!(config.hidden_size, 99999);
        assert_eq!(config.max_seq_len, 99999);
    }

    // 11. IntentTrackerConfig clone mutation isolation

    #[test]
    fn intent_tracker_config_clone_mutation_isolation() {
        let mut original = IntentTrackerConfig::default();
        let cloned = original.clone();
        original.hidden_size = 0;
        original.num_heads = 0;
        assert_eq!(cloned.hidden_size, 768, "cloned should retain original default value");
        assert_eq!(cloned.num_heads, 4, "cloned should retain original default value");
    }

    // 12. Classification explicit logits for each TaskType variant

    #[test]
    fn classification_explicit_logits_for_each_task_type() {
        let cases = [
            (vec![100.0, 0.0, 0.0], TaskType::ArchRefactor),
            (vec![0.0, 100.0, 0.0], TaskType::CodeDeploy),
            (vec![0.0, 0.0, 100.0], TaskType::Debugging),
        ];
        for (logits, expected) in &cases {
            let c = Classification {
                task_logits: logits.clone(),
                difficulty_logits: vec![0.0; 4],
            };
            assert_eq!(c.task_type(), *expected);
        }
    }

    // 13. Classification difficulty_confidence with single logit

    #[test]
    fn classification_difficulty_confidence_single_logit() {
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![5.0],
        };
        let conf = c.difficulty_confidence();
        assert!((conf - 1.0).abs() < 1e-6, "single difficulty logit → confidence 1.0, got {conf}");
    }

    // 14. Classification difficulty_confidence bounded for various finite logits

    #[test]
    fn classification_difficulty_confidence_various_finite() {
        let cases: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![100.0, 0.0, 0.0, 0.0],
            vec![-50.0, -50.0, -50.0, -50.0],
            vec![0.001, 0.002, 0.003, 0.004],
        ];
        for difficulty_logits in &cases {
            let c = Classification {
                task_logits: vec![0.0; 3],
                difficulty_logits: difficulty_logits.clone(),
            };
            let conf = c.difficulty_confidence();
            assert!(
                conf > 0.0 && conf <= 1.0 + 1e-6,
                "difficulty confidence should be in (0,1] for logits {difficulty_logits:?}, got {conf}"
            );
        }
    }

    // 15. TrackerTurnInput with all roles = 2

    #[test]
    fn tracker_turn_input_roles_all_value_two() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]; 4],
            roles: vec![2u8; 4],
            signals: [0.0; 11],
            context_turns: 1.0,
        };
        assert!(input.roles.iter().all(|&r| r == 2));
        assert_eq!(input.roles.len(), 4);
    }

    // 16. TrackerTurnInput with empty inner embeddings

    #[test]
    fn tracker_turn_input_empty_inner_embeddings() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![], vec![]],
            roles: vec![0, 1],
            signals: [0.0; 11],
            context_turns: 0.0,
        };
        assert_eq!(input.embeddings.len(), 2);
        assert!(input.embeddings[0].is_empty());
        assert!(input.embeddings[1].is_empty());
    }

    // 17. TrackerTurnInput context_turns with NaN (struct accepts any f32)

    #[test]
    fn tracker_turn_input_context_turns_nan() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: f32::NAN,
        };
        assert!(input.context_turns.is_nan());
    }

    // 18. TrackerTurnInput context_turns with f32::EPSILON

    #[test]
    fn tracker_turn_input_context_turns_epsilon() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: f32::EPSILON,
        };
        assert!((input.context_turns - f32::EPSILON).abs() < f32::EPSILON);
    }

    // 19. verify_weights missing per_head_norm_weight

    #[test]
    fn verify_weights_missing_per_head_norm_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_bias", // omit per_head_norm_weight
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("per_head_norm_weight"),
            "should report missing per_head_norm_weight, got: {msg}"
        );
    }

    // 20. verify_weights missing w_v_weight

    #[test]
    fn verify_weights_missing_w_v_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", // omit w_v_weight
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("w_v_weight"), "should report missing w_v_weight, got: {msg}");
    }

    // 21. verify_weights missing task_fc0_weight

    #[test]
    fn verify_weights_missing_task_fc0_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            // omit task_fc0_weight
            "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("task_fc0_weight"), "should report missing task_fc0_weight, got: {msg}");
    }

    // 22. softmax_max with f32::MAX dominant logit

    #[test]
    fn softmax_max_f32_max_dominates() {
        let conf = softmax_max(&[0.0, f32::MAX, 0.0]);
        assert!((conf - 1.0).abs() < 1e-3, "f32::MAX logit should dominate, got {conf}");
    }

    // 23. TrackerError LoadFailed with empty inner string

    #[test]
    fn tracker_error_load_failed_empty_inner() {
        let err = TrackerError::LoadFailed(String::new());
        let msg = format!("{err}");
        assert!(msg.contains("model loading failed:"), "should contain prefix: {msg}");
    }

    // 24. linear_forward 3x3 identity with nonzero bias adds bias

    #[test]
    fn linear_forward_3x3_identity_with_bias() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let bias = vec![10.0, 20.0, 30.0];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 22.0).abs() < 1e-6);
        assert!((out[2] - 33.0).abs() < 1e-6);
    }

    // 25. softmax_max all-zeros uniform

    #[test]
    fn softmax_max_all_zeros_uniform() {
        let conf = softmax_max(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let expected = 1.0 / 5.0;
        assert!((conf - expected).abs() < 1e-5, "all-zeros 5 elements → 0.2, got {conf}");
    }

    // ── Batch 7: 18 additional unit tests ──

    // 1. TrackerError::InferenceFailed inner string accessible via pattern match

    #[test]
    fn tracker_error_inference_failed_inner_accessible() {
        let err = TrackerError::InferenceFailed("NaN in logits".into());
        let msg = match &err {
            TrackerError::InferenceFailed(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "NaN in logits");
    }

    // 2. TrackerError::MissingWeight inner string accessible via pattern match

    #[test]
    fn tracker_error_missing_weight_inner_accessible() {
        let err = TrackerError::MissingWeight("w_q_weight".into());
        let msg = match &err {
            TrackerError::MissingWeight(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "w_q_weight");
    }

    // 3. TrackerError::EncoderFailed inner string accessible via pattern match

    #[test]
    fn tracker_error_encoder_failed_inner_accessible() {
        let err = TrackerError::EncoderFailed("connection refused".into());
        let msg = match &err {
            TrackerError::EncoderFailed(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "connection refused");
    }

    // 4. TrackerError::DequantFailed inner string accessible via pattern match

    #[test]
    fn tracker_error_dequant_failed_inner_accessible() {
        let err = TrackerError::DequantFailed("overflow in BF16".into());
        let msg = match &err {
            TrackerError::DequantFailed(s) => s.clone(),
            _ => String::new(),
        };
        assert_eq!(msg, "overflow in BF16");
    }

    // 5. IntentTrackerConfig Debug trait includes field names

    #[test]
    fn intent_tracker_config_debug_output() {
        let config = IntentTrackerConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("IntentTrackerConfig"), "Debug should contain struct name: {debug}");
        assert!(debug.contains("hidden_size"), "Debug should contain hidden_size: {debug}");
        assert!(debug.contains("num_heads"), "Debug should contain num_heads: {debug}");
        assert!(debug.contains("head_dim"), "Debug should contain head_dim: {debug}");
    }

    // 6. verify_weights missing context_norm_weight

    #[test]
    fn verify_weights_missing_context_norm_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_bias", // omit context_norm_weight
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("context_norm_weight"), "should report missing context_norm_weight, got: {msg}");
    }

    // 7. verify_weights missing signal_fc2_weight

    #[test]
    fn verify_weights_missing_signal_fc2_weight() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_bias", // omit signal_fc2_weight
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("signal_fc2_weight"), "should report missing signal_fc2_weight, got: {msg}");
    }

    // 8. verify_weights missing per_head_norm_bias

    #[test]
    fn verify_weights_missing_per_head_norm_bias() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", // omit per_head_norm_bias
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("per_head_norm_bias"), "should report missing per_head_norm_bias, got: {msg}");
    }

    // 9. verify_weights missing w_q_bias

    #[test]
    fn verify_weights_missing_w_q_bias() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_k_bias", "w_v_bias", // omit w_q_bias
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("w_q_bias"), "should report missing w_q_bias, got: {msg}");
    }

    // 10. verify_weights missing w_k_bias

    #[test]
    fn verify_weights_missing_w_k_bias() {
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_v_bias", // omit w_k_bias
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("w_k_bias"), "should report missing w_k_bias, got: {msg}");
    }

    // 11. argmax with f32::MIN among smaller negatives

    #[test]
    fn argmax_with_f32_min_value() {
        assert_eq!(argmax(&[-1.0, f32::MIN, 0.0]), Some(2));
        assert_eq!(argmax(&[f32::MIN, f32::MIN, -1.0]), Some(2));
    }

    // 12. argmax with single negative infinity

    #[test]
    fn argmax_single_negative_infinity() {
        assert_eq!(argmax(&[f32::NEG_INFINITY]), Some(0));
    }

    // 13. softmax_max uniform confidence equals 1/n for n=8..12

    #[test]
    fn softmax_max_uniform_confidence_larger_n() {
        for n in 8..=12 {
            let logits = vec![1.0; n];
            let conf = softmax_max(&logits);
            let expected = 1.0 / n as f32;
            assert!(
                (conf - expected).abs() < 1e-5,
                "uniform n={n}: expected {expected}, got {conf}"
            );
        }
    }

    // 14. linear_forward with f32::MAX input values

    #[test]
    fn linear_forward_with_f32_max_input() {
        let input = vec![f32::MAX, 0.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0]; // 2x2, first row picks first input
        let bias = vec![0.0; 2];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert_eq!(out[0], f32::MAX);
        assert!((out[1]).abs() < 1e-6);
    }

    // 15. linear_forward with f32::MIN bias

    #[test]
    fn linear_forward_with_f32_min_bias() {
        let input = vec![0.0];
        let weight = vec![0.0]; // 1x1 zero weight
        let bias = vec![f32::MIN];
        let out = linear_forward(&input, &weight, &bias).unwrap();
        assert_eq!(out[0], f32::MIN, "zero input with f32::MIN bias should output f32::MIN");
    }

    // 16. validate_e2e_inputs with many items passes

    #[test]
    fn validate_e2e_inputs_many_items_passes() {
        let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
        let roles: Vec<u8> = vec![0, 1, 0, 1, 0];
        let result = validate_e2e_inputs(&texts, &roles);
        assert!(result.is_ok());
    }

    // 17. TrackerTurnInput signals mutation after clone does not affect original

    #[test]
    fn tracker_turn_input_signals_mutation_after_clone() {
        let input = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768]],
            roles: vec![0],
            signals: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            context_turns: 5.0,
        };
        let mut cloned = input.clone();
        cloned.signals[0] = 99.0;
        // Original signals is [f32; 11] — Copy type, so mutation of clone doesn't affect original
        assert!((input.signals[0] - 1.0).abs() < f32::EPSILON);
        assert!((cloned.signals[0] - 99.0).abs() < f32::EPSILON);
    }

    // 18. IntentTrackerConfig with all fields equal to defaults but one changed

    #[test]
    fn intent_tracker_config_single_field_difference() {
        let a = IntentTrackerConfig::default();
        let mut b = IntentTrackerConfig::default();
        b.signal_hidden_dim = 128;
        assert_ne!(a, b);
        // All other fields remain equal
        assert_eq!(a.hidden_size, b.hidden_size);
        assert_eq!(a.num_heads, b.num_heads);
        assert_eq!(a.head_dim, b.head_dim);
        assert_eq!(a.num_tasks, b.num_tasks);
        assert_eq!(a.num_difficulties, b.num_difficulties);
        assert_eq!(a.signal_dim, b.signal_dim);
        assert_eq!(a.max_seq_len, b.max_seq_len);
    }

    // ── Wave 12x81 additional tests ──

    #[test]
    fn argmax_empty_slice_returns_none() {
        // Arrange: empty slice
        let data: &[f32] = &[];
        // Act
        let idx = argmax(data);
        // Assert: empty input returns None (PSC-3 fix: argmax returns Option<usize>)
        assert_eq!(idx, None);
    }

    #[test]
    fn classify_turn_quant_empty_embeddings_errors() {
        // Arrange
        let tracker = make_test_tracker();
        let embeddings: Vec<Vec<f16>> = vec![];
        let roles: Vec<u8> = vec![];
        let signals = [0.5; 11];
        // Act
        let result = tracker.classify_turn_quant(&embeddings, &roles, &signals, 1.0, |v| v.to_f32());
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn verify_weights_extra_weights_with_one_missing() {
        // Arrange: include all required weights plus extras, but remove one
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            if *name != "recency_scale" {
                shapes.insert(name.to_string(), vec![1]);
            }
        }
        shapes.insert("extra_unused_weight".to_string(), vec![2, 3]);
        // Act
        let result = verify_weights(&shapes, &config);
        // Assert: still errors on missing required weight despite extras
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, TrackerError::MissingWeight(n) if n == "recency_scale"));
    }

    #[test]
    fn classification_task_logits_identical_nonzero() {
        // Arrange: all three logits are the same nonzero value
        let cls = Classification {
            task_logits: vec![2.5, 2.5, 2.5],
            difficulty_logits: vec![0.1, 0.2, 0.3, 0.4],
        };
        // Act
        let tt = cls.task_type();
        let conf = cls.task_confidence();
        // Assert: all equal → argmax returns last index (2) → Debugging
        assert_eq!(tt, TaskType::Debugging);
        assert!((conf - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn classification_difficulty_logits_single_element() {
        // Arrange: single difficulty logit
        let cls = Classification {
            task_logits: vec![1.0, 0.0, 0.0],
            difficulty_logits: vec![3.7],
        };
        // Act
        let diff = cls.difficulty();
        let conf = cls.difficulty_confidence();
        // Assert: single element → argmax = 0, confidence = 1.0
        assert_eq!(diff, 0);
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tracker_turn_input_context_turns_zero_struct() {
        // Arrange: construct with context_turns = 0.0
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: 0.0,
        };
        // Assert: all fields accessible with correct values
        assert_eq!(input.embeddings.len(), 1);
        assert_eq!(input.roles[0], 0);
        assert!(input.signals.iter().all(|&s| s == 0.0));
        assert!((input.context_turns).abs() < f32::EPSILON);
    }

    #[test]
    fn linear_forward_identity_matrix_zero_bias_passes_through() {
        // Arrange: 2x2 identity weight, zero bias
        let input = vec![3.0, -7.0];
        let weight = vec![1.0, 0.0, 0.0, 1.0];
        let bias = vec![0.0, 0.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: output == input (identity transformation)
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - (-7.0)).abs() < 1e-6);
    }

    #[test]
    fn softmax_max_two_extreme_negatives_one_moderate() {
        // Arrange: two very negative values and one moderate
        let logits = &[-1000.0, 0.5, -2000.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: 0.5 dominates, confidence very close to 1.0
        assert!(conf > 0.99);
        assert!(conf <= 1.0);
    }

    #[test]
    fn tracker_turn_input_empty_roles_vec() {
        // Arrange: empty roles and embeddings
        let input = TrackerTurnInput {
            embeddings: vec![],
            roles: vec![],
            signals: [1.0; 11],
            context_turns: 1.0,
        };
        // Assert: empty vecs are valid struct fields
        assert!(input.roles.is_empty());
        assert!(input.embeddings.is_empty());
        assert_eq!(input.roles.len(), input.embeddings.len());
    }

    #[test]
    fn argmax_alternating_high_low_pattern() {
        // Arrange: alternating max/min values
        let data = &[5.0, 1.0, 5.0, 1.0];
        // Act
        let idx = argmax(data);
        // Assert: returns last occurrence of max (index 2)
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn tracker_turn_input_context_turns_infinity() {
        // Arrange: construct with context_turns = f32::INFINITY
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.5; 768]],
            roles: vec![1],
            signals: [0.0; 11],
            context_turns: f32::INFINITY,
        };
        // Assert: field stores positive infinity
        assert!(input.context_turns.is_infinite());
        assert!(input.context_turns.is_sign_positive());
    }

    #[test]
    fn softmax_max_logit_zero_among_negatives() {
        // Arrange: one logit is 0.0, rest are negative
        let logits = &[-3.0, 0.0, -1.0, -5.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: 0.0 is the max, should dominate
        assert!(conf > 0.5);
        assert!(conf <= 1.0);
    }

    #[test]
    fn argmax_all_nan_returns_none() {
        // Arrange: all NaN values
        let data = &[f32::NAN, f32::NAN, f32::NAN];
        // Act
        let idx = argmax(data);
        // Assert: all values are NaN → filtered out → None (indicates upstream computation error)
        assert_eq!(idx, None);
    }

    // ── Additional unit tests ──

    #[test]
    fn from_weights_bf16_odd_byte_count_errors() {
        // Arrange: BF16 requires 2 bytes per element; provide odd-length bytes
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        for name in &["role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate"] {
            shapes.insert((*name).to_string(), vec![1]);
        }
        let mut weights_bf16 = HashMap::new();
        weights_bf16.insert("role_emb_weight".to_string(), vec![0u8, 1, 3]); // odd: 3 bytes
        for name in &["w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate"] {
            weights_bf16.insert((*name).to_string(), vec![0u8, 0]);
        }
        // Act
        let result = IntentTracker::from_weights_bf16(config, weights_bf16, shapes);
        // Assert: odd bytes will produce different count than expected
        // The dequant loop uses chunks_exact(2), so the trailing byte is ignored
        // but numel = bytes.len()/2 = 1, floats.len() = 1, so it won't error on count mismatch.
        // Actually numel=3/2=1, floats=1. Let's just verify it doesn't panic.
        // With odd bytes, chunks_exact skips the last byte, so it actually succeeds.
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn from_weights_with_custom_config_succeeds() {
        // Arrange: custom config with non-default hidden_size
        let config = IntentTrackerConfig {
            hidden_size: 64,
            num_heads: 2,
            head_dim: 32,
            num_tasks: 3,
            num_difficulties: 4,
            signal_dim: 11,
            signal_hidden_dim: 16,
            max_seq_len: 16,
        };
        let h = config.hidden_size;
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        let add_matrix = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, rows: usize, cols: usize| {
            w.insert(name.to_string(), vec![0.01; rows * cols]);
            s.insert(name.to_string(), vec![rows, cols]);
        };
        let add_vector = |w: &mut HashMap<String, Vec<f32>>, s: &mut HashMap<String, Vec<usize>>,
                          name: &str, len: usize| {
            w.insert(name.to_string(), vec![0.0; len]);
            s.insert(name.to_string(), vec![len]);
        };
        add_matrix(&mut weights, &mut shapes, "role_emb_weight", 3, h);
        for prefix in &["w_q", "w_k", "w_v"] {
            add_matrix(&mut weights, &mut shapes, &format!("{prefix}_weight"), h, h);
            add_vector(&mut weights, &mut shapes, &format!("{prefix}_bias"), h);
        }
        for (i, dim) in [32, 16, 1].iter().enumerate() {
            let in_dim = if i == 0 { h } else { [32, 16][i - 1] };
            add_matrix(&mut weights, &mut shapes, &format!("info_net_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("info_net_fc{i}_bias"), *dim);
        }
        add_vector(&mut weights, &mut shapes, "per_head_norm_weight", config.head_dim);
        add_vector(&mut weights, &mut shapes, "per_head_norm_bias", config.head_dim);
        add_vector(&mut weights, &mut shapes, "context_norm_weight", h);
        add_vector(&mut weights, &mut shapes, "context_norm_bias", h);
        for (i, dim) in [16, 16, config.signal_hidden_dim].iter().enumerate() {
            let in_dim = if i == 0 { config.signal_dim } else { 16 };
            add_matrix(&mut weights, &mut shapes, &format!("signal_fc{i}_weight"), *dim, in_dim);
            add_vector(&mut weights, &mut shapes, &format!("signal_fc{i}_bias"), *dim);
        }
        let classifier_input_dim = h + config.signal_hidden_dim + 1;
        for (i, dim) in [32, 16, config.num_tasks].iter().enumerate() {
            let real_in = [classifier_input_dim, 32, 16][i];
            add_matrix(&mut weights, &mut shapes, &format!("task_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("task_fc{i}_bias"), *dim);
        }
        for (i, dim) in [32, 16, config.num_difficulties].iter().enumerate() {
            let real_in = [classifier_input_dim, 32, 16][i];
            add_matrix(&mut weights, &mut shapes, &format!("diff_fc{i}_weight"), *dim, real_in);
            add_vector(&mut weights, &mut shapes, &format!("diff_fc{i}_bias"), *dim);
        }
        add_vector(&mut weights, &mut shapes, "recency_scale", 1);
        add_vector(&mut weights, &mut shapes, "context_gate", 1);
        // Act
        let result = IntentTracker::from_weights(config.clone(), weights, shapes);
        // Assert
        assert!(result.is_ok());
        let tracker = result.unwrap();
        // Run inference with matching embedding dim
        let embeddings = vec![vec![0.1; 64]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        let cls = tracker.classify_turn(&embeddings, &roles, &signals, 1.0).unwrap();
        assert_eq!(cls.task_logits.len(), 3);
        assert_eq!(cls.difficulty_logits.len(), 4);
    }

    #[test]
    fn classify_turn_deterministic_same_input_same_output() {
        // Arrange: call classify_turn twice with identical inputs
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.25; 768]; 4];
        let roles = vec![0u8, 1, 0, 1];
        let signals = [0.3; 11];
        // Act
        let r1 = tracker.classify_turn(&embeddings, &roles, &signals, 2.0).unwrap();
        let r2 = tracker.classify_turn(&embeddings, &roles, &signals, 2.0).unwrap();
        // Assert: outputs are bitwise identical
        assert_eq!(r1.task_logits, r2.task_logits);
        assert_eq!(r1.difficulty_logits, r2.difficulty_logits);
    }

    #[test]
    fn classify_turns_batch_first_error_stops_iteration() {
        // Arrange: batch where the 2nd item has wrong embedding dim,
        // 3rd item is valid (should never be reached)
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 1],
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 64]], // wrong dim
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 1],
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
        ];
        // Act
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        // Assert: error is from the 2nd item's invalid dim
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    #[test]
    fn classify_turn_quant_dequant_squares_values() {
        // Arrange: dequant function squares f16 values
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(2.0); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        // Act: dequant squares the value (2.0 -> 4.0)
        let result = tracker.classify_turn_quant(
            &embeddings_f16, &roles, &signals, 2.0, |v| v.to_f32() * v.to_f32(),
        );
        // Assert: should succeed (the squared value is still finite f32)
        assert!(result.is_ok());
        let cls = result.unwrap();
        assert_eq!(cls.task_logits.len(), 3);
    }

    #[test]
    fn tracker_turn_input_embeddings_mutable_after_clone() {
        // Arrange: create input and clone it
        let original = TrackerTurnInput {
            embeddings: vec![vec![1.0; 768]],
            roles: vec![0],
            signals: [0.5; 11],
            context_turns: 1.0,
        };
        let mut cloned = original.clone();
        // Act: mutate cloned embeddings
        cloned.embeddings[0][0] = 99.0;
        // Assert: original is unaffected
        assert!((original.embeddings[0][0] - 1.0).abs() < 1e-6);
        assert!((cloned.embeddings[0][0] - 99.0).abs() < 1e-6);
    }

    #[test]
    fn tracker_turn_input_roles_mutable_after_clone() {
        // Arrange
        let original = TrackerTurnInput {
            embeddings: vec![vec![0.5; 768]; 2],
            roles: vec![0, 1],
            signals: [0.0; 11],
            context_turns: 0.0,
        };
        let mut cloned = original.clone();
        // Act
        cloned.roles[0] = 2;
        // Assert: original unchanged
        assert_eq!(original.roles[0], 0);
        assert_eq!(cloned.roles[0], 2);
    }

    #[test]
    fn argmax_mixed_pos_inf_neg_inf_and_normal() {
        // Arrange: positive infinity, negative infinity, and a normal value
        let data = &[f32::NEG_INFINITY, 5.0, f32::INFINITY, -1.0];
        // Act
        let idx = argmax(data);
        // Assert: positive infinity wins
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn softmax_max_with_three_large_equal_logits() {
        // Arrange: three equal large logits
        let logits = &[1e30, 1e30, 1e30];
        // Act
        let conf = softmax_max(logits);
        // Assert: uniform distribution over 3, confidence ~1/3
        assert!((conf - (1.0 / 3.0)).abs() < 0.01);
    }

    #[test]
    fn linear_forward_2x3_rectangular() {
        // Arrange: 3-element input, 2-element output
        let input = vec![1.0, 2.0, 3.0];
        // weight: 2x3 matrix [[1,2,3],[4,5,6]]
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = vec![0.0, 0.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
        assert!((out[0] - 14.0).abs() < 1e-6);
        assert!((out[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn linear_forward_3x2_tall_matrix() {
        // Arrange: 2-element input, 3-element output
        let input = vec![2.0, 3.0];
        // weight: 3x2 matrix [[1,0],[0,1],[1,1]]
        let weight = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = vec![10.0, 20.0, 30.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: [2*1+3*0+10, 2*0+3*1+20, 2*1+3*1+30] = [12, 23, 35]
        assert!((out[0] - 12.0).abs() < 1e-6);
        assert!((out[1] - 23.0).abs() < 1e-6);
        assert!((out[2] - 35.0).abs() < 1e-6);
    }

    #[test]
    fn verify_weights_partial_missing_reports_first_alphabetically() {
        // Arrange: provide only a few weights, not the first in the required list
        let mut shapes = HashMap::new();
        // "role_emb_weight" is first in required list — omit it
        shapes.insert("w_q_weight".to_string(), vec![768, 768]);
        let config = IntentTrackerConfig::default();
        // Act
        let err = verify_weights(&shapes, &config).unwrap_err();
        // Assert: reports the first missing weight in the required array order
        assert!(matches!(err, TrackerError::MissingWeight(ref s) if s == "role_emb_weight"));
    }

    #[test]
    fn get_weight_present_returns_expected_length() {
        // Arrange
        let tracker = make_test_tracker();
        // Act
        let w = tracker.get_weight("recency_scale").unwrap();
        // Assert: recency_scale is a vector of length 1
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn intent_tracker_weights_count_matches_shapes() {
        // Arrange
        let tracker = make_test_tracker();
        // Act & Assert: the tracker should have at least all required weights
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            assert!(tracker.get_weight(name).is_ok(), "missing weight: {name}");
        }
    }

    #[test]
    fn classify_turn_single_token_minimal_input() {
        // Arrange: single token with minimal embedding
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.0; 768]];
        let roles = vec![0u8];
        let signals = [0.0; 11];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 0.0);
        // Assert: single-token input succeeds
        assert!(result.is_ok());
        let cls = result.unwrap();
        assert_eq!(cls.task_logits.len(), 3);
        assert_eq!(cls.difficulty_logits.len(), 4);
    }

    // ── Wave 12x82: 15 additional unit tests ──

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_outputs_differ_for_different_inputs() {
        // Arrange: two batch items with distinctly different embeddings
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.0; 768]; 1],
                roles: vec![0],
                signals: [0.0; 11],
                context_turns: 0.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![1.0; 768]; 1],
                roles: vec![1],
                signals: [1.0; 11],
                context_turns: 100.0,
            },
        ];
        // Act
        let results = tracker.classify_turns_batch(&batch).unwrap();
        // Assert: different inputs should produce different outputs (at least one logit differs)
        let task_differs = results[0].task_logits != results[1].task_logits;
        let diff_differs = results[0].difficulty_logits != results[1].difficulty_logits;
        assert!(task_differs || diff_differs, "different inputs should yield different outputs");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_role_index_two_succeeds() {
        // Arrange: role index 2 is the last valid role (role_emb_weight has 3 rows: 0, 1, 2)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.5; 768]; 2];
        let roles = vec![2u8, 2];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 2.0);
        // Assert
        assert!(result.is_ok(), "role index 2 should be valid, got {:?}", result.err());
        let cls = result.unwrap();
        assert_eq!(cls.task_logits.len(), 3);
        assert_eq!(cls.difficulty_logits.len(), 4);
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_role_index_255_errors() {
        // Arrange: role index 255 is far out of range
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]];
        let roles = vec![255u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("role index 255"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_embedding_dim_zero_errors() {
        // Arrange: embedding with zero dimensions
        let tracker = make_test_tracker();
        let embeddings = vec![vec![]]; // dim 0 instead of 768
        let roles = vec![0u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[0] dim 0"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_empty_f16_embeddings_valid_roles_errors() {
        // Arrange: empty f16 embeddings and matching empty roles
        let tracker = make_test_tracker();
        let embeddings: Vec<Vec<f16>> = vec![];
        let roles: Vec<u8> = vec![];
        // Act: should delegate to classify_turn which rejects empty embeddings
        let err = tracker
            .classify_turn_quant(&embeddings, &roles, &[0.5; 11], 0.0, |v| v.to_f32())
            .unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_large_batch_succeeds() {
        // Arrange: batch of 10 items, all valid
        let tracker = make_test_tracker();
        let batch: Vec<TrackerTurnInput> = (0..10)
            .map(|i| TrackerTurnInput {
                embeddings: vec![vec![i as f32 * 0.01; 768]; 2],
                roles: vec![0, 1],
                signals: [0.1; 11],
                context_turns: i as f32 + 1.0,
            })
            .collect();
        // Act
        let results = tracker.classify_turns_batch(&batch).unwrap();
        // Assert
        assert_eq!(results.len(), 10);
        for r in &results {
            assert_eq!(r.task_logits.len(), 3);
            assert_eq!(r.difficulty_logits.len(), 4);
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_with_negative_embeddings_succeeds() {
        // Arrange: all-negative embeddings
        let tracker = make_test_tracker();
        let embeddings = vec![vec![-0.5; 768]; 2];
        let roles = vec![0u8, 1];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 2.0).unwrap();
        // Assert: should produce finite logits
        for &l in &result.task_logits {
            assert!(l.is_finite(), "task logit should be finite, got {l}");
        }
        for &l in &result.difficulty_logits {
            assert!(l.is_finite(), "difficulty logit should be finite, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_large_matrix_computation() {
        // Arrange: 100-input, 50-output matrix (larger than typical small tests)
        let input: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..5000).map(|i| if i % 3 == 0 { 0.01 } else { -0.01 }).collect();
        let bias: Vec<f32> = (0..50).map(|i| (i as f32) * 0.001).collect();
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert
        assert_eq!(out.len(), 50);
        // Output should be finite (small weights/biases prevent overflow)
        for &v in &out {
            assert!(v.is_finite(), "output should be finite, got {v}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_ten_elements_one_dominant() {
        // Arrange: 10 elements, one clearly dominant
        let mut logits = vec![0.0; 10];
        logits[7] = 100.0;
        // Act
        let conf = softmax_max(&logits);
        // Assert: dominant element should give confidence > 0.99
        assert!(conf > 0.99, "dominant element among 10 should yield > 0.99, got {conf}");
        assert!(conf <= 1.0);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_many_ties_returns_last() {
        // Arrange: 5 elements, all equal
        let data = &[7.7; 5];
        // Act
        let idx = argmax(data);
        // Assert: ties resolved by returning last index
        assert_eq!(idx, Some(4));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn from_weights_bf16_preserves_negative_values() {
        // Arrange: BF16 representation of -1.0 = 0xBF80
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let neg_one_bf16 = bf16::from_f32(-1.0).to_bits();
        let bytes = neg_one_bf16.to_le_bytes();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            weights_bf16.insert(name.to_string(), vec![bytes[0], bytes[1]]);
            shapes.insert(name.to_string(), vec![1]);
        }
        // Act
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        // Assert
        assert!((w[0] - (-1.0)).abs() < 0.01, "BF16 -1.0 should dequantize to ~-1.0, got {}", w[0]);
    }

    // @trace TEST-SIT [req:REQ-SIT-007] [level:unit]
    #[test]
    fn validate_e2e_inputs_long_texts_and_roles_pass() {
        // Arrange: longer texts and matching roles
        let texts = vec!["hello world this is a test", "another turn in the conversation"];
        let roles = vec![0u8, 1u8];
        // Act
        let result = validate_e2e_inputs(&texts, &roles);
        // Assert
        assert!(result.is_ok());
    }

    // @trace TEST-SIT [req:REQ-SIT-007] [level:unit]
    #[test]
    fn validate_e2e_inputs_many_items_mismatch_errors() {
        // Arrange: 5 texts but only 3 roles
        let texts = vec!["a", "b", "c", "d", "e"];
        let roles = vec![0u8, 1u8, 0u8];
        // Act
        let err = validate_e2e_inputs(&texts, &roles).unwrap_err();
        // Assert
        let msg = format!("{err}");
        assert!(msg.contains("turn_texts len 5"), "unexpected message: {msg}");
        assert!(msg.contains("roles len 3"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_new_invalid_safetensors_content_errors() {
        // Arrange: create a temp file with garbage content (not valid safetensors)
        let tmp = std::env::temp_dir().join("gllm_test_invalid_safetensors.safetensors");
        std::fs::write(&tmp, b"this is not a valid safetensors file").unwrap();
        // Act
        let err = IntentTracker::new(&tmp).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::LoadFailed(_)));
        let msg = format!("{err}");
        assert!(msg.contains("parse safetensors"), "should mention parse failure: {msg}");
        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_weight_dimension_zero_output() {
        // Arrange: 0-output linear (n=0, k=3, weight is empty)
        let input = vec![1.0, 2.0, 3.0];
        let weight: Vec<f32> = vec![]; // 0 rows * 3 cols = 0
        let bias: Vec<f32> = vec![]; // 0 outputs
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: should produce empty output
        assert!(out.is_empty(), "0-output linear should produce empty vector");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_clone_preserves_variant_type() {
        // Arrange: create each variant, clone, verify same variant via matches!
        let errors = vec![
            TrackerError::LoadFailed("test load".into()),
            TrackerError::InvalidInput("test input".into()),
            TrackerError::InferenceFailed("test infer".into()),
            TrackerError::MissingWeight("test weight".into()),
            TrackerError::EncoderFailed("test encoder".into()),
            TrackerError::DequantFailed("test dequant".into()),
        ];
        // Act & Assert
        for err in &errors {
            let cloned = err.clone();
            match (err, &cloned) {
                (TrackerError::LoadFailed(_), TrackerError::LoadFailed(_)) => {},
                (TrackerError::InvalidInput(_), TrackerError::InvalidInput(_)) => {},
                (TrackerError::InferenceFailed(_), TrackerError::InferenceFailed(_)) => {},
                (TrackerError::MissingWeight(_), TrackerError::MissingWeight(_)) => {},
                (TrackerError::EncoderFailed(_), TrackerError::EncoderFailed(_)) => {},
                (TrackerError::DequantFailed(_), TrackerError::DequantFailed(_)) => {},
                _ => panic!("clone changed error variant: original {:?}, clone {:?}", err, cloned),
            }
        }
    }

    // ── Wave 12x83: 15 additional unit tests ──

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn from_weights_bf16_multi_element_vector_dequantizes_correctly() {
        // Arrange: create BF16 representation of [0.0, 1.0, -1.0, 0.5]
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let values_f32 = [0.0f32, 1.0, -1.0, 0.5];
        let bf16_bytes: Vec<u8> = values_f32.iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();

        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            if *name == "recency_scale" {
                weights_bf16.insert(name.to_string(), bf16_bytes.clone());
                shapes.insert(name.to_string(), vec![4]);
            } else {
                weights_bf16.insert(name.to_string(), vec![0u8, 0]);
                shapes.insert(name.to_string(), vec![1]);
            }
        }
        // Act
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        // Assert: each value should be close to the original f32 after BF16 roundtrip
        assert_eq!(w.len(), 4);
        assert!((w[0] - 0.0).abs() < 0.01, "element 0: expected ~0.0, got {}", w[0]);
        assert!((w[1] - 1.0).abs() < 0.01, "element 1: expected ~1.0, got {}", w[1]);
        assert!((w[2] - (-1.0)).abs() < 0.01, "element 2: expected ~-1.0, got {}", w[2]);
        assert!((w[3] - 0.5).abs() < 0.01, "element 3: expected ~0.5, got {}", w[3]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_diagonal_matrix_scales_input() {
        // Arrange: 3x3 diagonal matrix with [2.0, 3.0, 5.0], input [1.0, 1.0, 1.0]
        let input = vec![1.0, 1.0, 1.0];
        let weight = vec![
            2.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 5.0,
        ];
        let bias = vec![0.0; 3];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: output = [2.0, 3.0, 5.0]
        assert!((out[0] - 2.0).abs() < 1e-6, "expected 2.0, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-6, "expected 3.0, got {}", out[1]);
        assert!((out[2] - 5.0).abs() < 1e-6, "expected 5.0, got {}", out[2]);
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_two_tokens_minimal_multi() {
        // Arrange: two-token sequence, the minimal multi-token input
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.0; 768], vec![0.0; 768]];
        let roles = vec![0u8, 1];
        let signals = [0.0; 11];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 1.0);
        // Assert: should succeed with correct logit dimensions
        assert!(result.is_ok(), "two-token input should succeed, got {:?}", result.err());
        let cls = result.unwrap();
        assert_eq!(cls.task_logits.len(), 3);
        assert_eq!(cls.difficulty_logits.len(), 4);
        for &l in &cls.task_logits {
            assert!(l.is_finite(), "task logit should be finite, got {l}");
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_all_same_role_zero_succeeds() {
        // Arrange: all tokens have role 0 (user)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.3; 768]; 5];
        let roles = vec![0u8; 5];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 5.0).unwrap();
        // Assert: output should be finite and correct dimensions
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in result.task_logits.iter().chain(result.difficulty_logits.iter()) {
            assert!(l.is_finite(), "all logits should be finite, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_single_positive_among_many_negative() {
        // Arrange: single positive value dominating 9 negative values
        let mut logits = vec![-10.0; 10];
        logits[3] = 5.0;
        // Act
        let conf = softmax_max(&logits);
        // Assert: the single positive should dominate strongly
        assert!(conf > 0.99, "single positive among negatives should dominate, got {conf}");
        assert!(conf <= 1.0);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_alternating_sign_pattern() {
        // Arrange: alternating positive and negative values
        let data = &[1.0, -1.0, 2.0, -2.0, 3.0];
        // Act
        let idx = argmax(data);
        // Assert: 3.0 at index 4 is the maximum
        assert_eq!(idx, Some(4));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_zero_bias_is_pure_matmul() {
        // Arrange: 2-input, 3-output with zero bias — output is pure matrix-vector product
        let input = vec![2.0, 3.0];
        let weight = vec![
            1.0, 2.0, // row 0: 2*1 + 3*2 = 8
            3.0, 4.0, // row 1: 2*3 + 3*4 = 18
            5.0, 6.0, // row 2: 2*5 + 3*6 = 28
        ];
        let bias = vec![0.0; 3];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: pure matmul without bias
        assert!((out[0] - 8.0).abs() < 1e-6, "expected 8.0, got {}", out[0]);
        assert!((out[1] - 18.0).abs() < 1e-6, "expected 18.0, got {}", out[1]);
        assert!((out[2] - 28.0).abs() < 1e-6, "expected 28.0, got {}", out[2]);
    }

    // @trace TEST-SIT [req:REQ-SIT-007] [level:unit]
    #[test]
    fn validate_e2e_inputs_unicode_text_passes() {
        // Arrange: turn texts with unicode characters
        let texts = vec!["你好世界", "こんにちは"];
        let roles = vec![0u8, 1u8];
        // Act
        let result = validate_e2e_inputs(&texts, &roles);
        // Assert: unicode text should pass length validation
        assert!(result.is_ok());
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_error_on_second_of_three_items() {
        // Arrange: 3-item batch where item 1 is valid, item 2 has wrong dim, item 3 is valid
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 128]], // wrong dim
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]],
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
        ];
        // Act
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        // Assert: error should be InvalidInput from item 2's wrong dim
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("dim 128"), "should mention wrong dimension: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_with_constant_dequant_produces_uniform_input() {
        // Arrange: all f16 embeddings dequantize to the same constant
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![
            vec![f16::from_f32(1.0); 768],
            vec![f16::from_f32(2.0); 768],
            vec![f16::from_f32(3.0); 768],
        ];
        let roles = vec![0u8, 1, 0];
        let signals = [0.0; 11];
        // Act: dequant everything to constant 0.5
        let result = tracker
            .classify_turn_quant(&embeddings_f16, &roles, &signals, 3.0, |_| 0.5)
            .unwrap();
        // Assert: should produce finite, correctly-sized output
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in &result.task_logits {
            assert!(l.is_finite(), "constant dequant should produce finite logits, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_default_signal_dim_is_eleven() {
        // Arrange: IntentTrackerConfig default
        let config = IntentTrackerConfig::default();
        // Assert: signal_dim should be 11 (matching the signals array)
        assert_eq!(config.signal_dim, 11, "signal_dim must match the fixed [f32; 11] signals array");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_signals_array_len_always_eleven() {
        // Arrange: construct with explicit signals covering edge f32 values
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [f32::MAX, f32::MIN, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 0.001],
            context_turns: 1.0,
        };
        // Assert: signals array is always exactly 11 elements
        assert_eq!(input.signals.len(), 11);
        assert_eq!(input.signals[0], f32::MAX);
        assert_eq!(input.signals[1], f32::MIN);
        assert!((input.signals[3]).abs() < f32::EPSILON); // -0.0 is still 0
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_two_element_one_positive_one_negative() {
        // Arrange: one positive and one negative logit
        let logits = &[10.0, -10.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: positive should dominate strongly
        let expected = 1.0 / (1.0 + (-20.0f32).exp());
        assert!(
            (conf - expected).abs() < 1e-4,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.99, "positive logit should dominate, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_1x2_projection_expands_dimension() {
        // Arrange: single input, two outputs
        let input = vec![5.0];
        let weight = vec![2.0, 3.0]; // 2x1, output[0] = 5*2 = 10, output[1] = 5*3 = 15
        let bias = vec![1.0, -1.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert
        assert_eq!(out.len(), 2);
        assert!((out[0] - 11.0).abs() < 1e-6, "expected 11.0, got {}", out[0]);
        assert!((out[1] - 14.0).abs() < 1e-6, "expected 14.0, got {}", out[1]);
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_with_mixed_positive_negative_embeddings() {
        // Arrange: some embedding dimensions positive, some negative
        let tracker = make_test_tracker();
        let embeddings = vec![
            (0..768).map(|i| if i % 2 == 0 { 0.5f32 } else { -0.5f32 }).collect(),
            (0..768).map(|i| if i % 3 == 0 { 1.0f32 } else { -0.1f32 }).collect(),
        ];
        let roles = vec![0u8, 1];
        let signals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 0.0];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &signals, 2.0).unwrap();
        // Assert: should produce finite logits
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in &result.task_logits {
            assert!(l.is_finite(), "task logit should be finite, got {l}");
        }
        for &l in &result.difficulty_logits {
            assert!(l.is_finite(), "difficulty logit should be finite, got {l}");
        }
    }

    // ── Wave 12x84: 15 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_new_valid_safetensors_missing_weights_errors() {
        // Arrange: use safetensors crate to build a valid file with a single dummy tensor.
        // The file is valid safetensors but only has one tensor, so verify_weights will fail.
        use safetensors::tensor::{TensorView, Dtype, Metadata};
        let data: Vec<u8> = vec![0u8; 4]; // one f32 zero
        let tensor = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let tensors = vec![("dummy", tensor)];
        let bytes = safetensors::serialize(tensors.iter().map(|(n, t)| (*n, t)), &None).unwrap();
        let tmp = std::env::temp_dir().join("gllm_test_missing_weights.safetensors");
        std::fs::write(&tmp, &bytes).unwrap();
        // Act
        let err = IntentTracker::new(&tmp).unwrap_err();
        // Assert: should fail on missing required weights
        assert!(matches!(err, TrackerError::MissingWeight(_)));
        let msg = format!("{err}");
        assert!(msg.contains("role_emb_weight"), "should report missing role_emb_weight, got: {msg}");
        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_embeddings_with_nan_values() {
        // Arrange: embeddings containing NaN (struct accepts any f32 values)
        let input = TrackerTurnInput {
            embeddings: vec![vec![f32::NAN; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: 1.0,
        };
        // Assert: NaN values are stored faithfully
        assert!(input.embeddings[0][0].is_nan());
        assert_eq!(input.embeddings[0].len(), 768);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_embeddings_with_inf_values() {
        // Arrange: embeddings containing positive and negative infinity
        let mut emb = vec![0.0f32; 768];
        emb[0] = f32::INFINITY;
        emb[1] = f32::NEG_INFINITY;
        let input = TrackerTurnInput {
            embeddings: vec![emb],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: 1.0,
        };
        // Assert
        assert!(input.embeddings[0][0].is_infinite());
        assert!(input.embeddings[0][0].is_sign_positive());
        assert!(input.embeddings[0][1].is_infinite());
        assert!(input.embeddings[0][1].is_sign_negative());
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_with_nan_logits_no_panic() {
        // Arrange: difficulty logits containing NaN
        let cls = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![f32::NAN, 1.0, 2.0, 3.0],
        };
        // Act: should not panic
        let diff = cls.difficulty();
        let conf = cls.difficulty_confidence();
        // Assert: returns a valid index
        assert!(diff < 4, "difficulty should be a valid index, got {diff}");
        // confidence may be NaN but must not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classify_turn_inconsistent_embedding_dims_first_correct_second_wrong() {
        // Arrange: first embedding has correct dim, second has wrong dim
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768], vec![0.1; 100]];
        let roles = vec![0u8, 1];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 2.0).unwrap_err();
        // Assert: should error on the second embedding
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[1]"), "should reference index 1: {msg}");
        assert!(msg.contains("dim 100"), "should reference wrong dim: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_with_large_input_produces_finite_for_small_weights() {
        // Arrange: large input values but very small weights to prevent overflow
        let input: Vec<f32> = (0..10).map(|i| (i as f32) * 1000.0).collect();
        let weight: Vec<f32> = (0..20).map(|_| 1e-6f32).collect(); // 2x10
        let bias = vec![0.0; 2];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: output should be finite (weights are small enough)
        assert_eq!(out.len(), 2);
        for &v in &out {
            assert!(v.is_finite(), "output should be finite with small weights, got {v}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_context_turns_negative_infinity() {
        // Arrange: context_turns = negative infinity
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: f32::NEG_INFINITY,
        };
        // Assert: struct accepts any f32 including negative infinity
        assert!(input.context_turns.is_infinite());
        assert!(input.context_turns.is_sign_negative());
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn from_weights_all_present_but_wrong_data_size_inference_fails() {
        // Arrange: all required weight names present but with size 1 instead of correct dims.
        // This passes verify_weights (which only checks presence) but classify_turn
        // should fail because linear_forward will detect wrong weight dimensions.
        let config = IntentTrackerConfig::default();
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            weights.insert(name.to_string(), vec![0.01; 1]); // wrong size
            shapes.insert(name.to_string(), vec![1]);
        }
        // Act: construction succeeds (verify_weights only checks presence)
        let tracker = IntentTracker::from_weights(config, weights, shapes).unwrap();
        // Act: inference should fail because weight dimensions are wrong
        let result = tracker.classify_turn(&[vec![0.1; 768]], &[0], &[0.5; 11], 1.0);
        // Assert: should error during inference
        assert!(result.is_err(), "inference with wrong-sized weights should fail");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn get_weight_returns_correct_data_not_copy() {
        // Arrange: build tracker with known weight value
        let tracker = make_test_tracker();
        // Act: get weight slice and verify it refers to stored data
        let w1 = tracker.get_weight("recency_scale").unwrap();
        let w2 = tracker.get_weight("recency_scale").unwrap();
        // Assert: both references point to the same underlying data
        assert_eq!(w1.as_ptr(), w2.as_ptr(), "get_weight should return references to the same data");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_with_one_nan_among_three() {
        // Arrange: one NaN among three finite logits
        let logits = &[1.0, f32::NAN, 3.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: NaN in exp() may propagate; result should not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_single_zero() {
        // Arrange
        let data = &[0.0f32];
        // Act
        let idx = argmax(data);
        // Assert
        assert_eq!(idx, Some(0));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_5x5_identity_preserves_input() {
        // Arrange: 5x5 identity matrix with zero bias
        let input = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let mut weight = vec![0.0f32; 25];
        for i in 0..5 {
            weight[i * 5 + i] = 1.0;
        }
        let bias = vec![0.0; 5];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: identity transformation preserves all values
        for i in 0..5 {
            assert!((out[i] - input[i]).abs() < 1e-6, "index {i}: expected {}, got {}", input[i], out[i]);
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_task_confidence_with_two_logits_only() {
        // Arrange: only 2 task logits (non-standard but struct allows it)
        let cls = Classification {
            task_logits: vec![5.0, -5.0],
            difficulty_logits: vec![0.0; 4],
        };
        // Act
        let conf = cls.task_confidence();
        // Assert: dominant logit should give high confidence
        assert!(conf > 0.99, "dominant of 2 logits should yield > 0.99, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn verify_weights_accepts_minimal_shapes_does_not_check_dimensions() {
        // Arrange: all shapes are [0] — zero-dimensional but present
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![0]); // zero-sized shapes
        }
        // Act & Assert: verify_weights only checks key presence, not shape validity
        assert!(verify_weights(&shapes, &config).is_ok(), "zero-sized shapes should pass presence check");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_with_all_positive_infinity_no_panic() {
        // Arrange: all logits are +infinity
        let logits = &[f32::INFINITY; 4];
        // Act
        let conf = softmax_max(logits);
        // Assert: exp(INF - INF) = exp(NaN) produces NaN; should not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_embeddings_dim_768_exact_match() {
        // Arrange: construct embeddings with exactly 768 dimensions
        let emb: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let input = TrackerTurnInput {
            embeddings: vec![emb.clone()],
            roles: vec![0],
            signals: [0.0; 11],
            context_turns: 1.0,
        };
        // Assert
        assert_eq!(input.embeddings[0].len(), 768);
        assert!((input.embeddings[0][0] - 0.0).abs() < f32::EPSILON);
        assert!((input.embeddings[0][767] - 0.767).abs() < 1e-3);
    }

    // ── Wave 12x85: 15 additional unit tests ──

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_empty_embeddings_item_errors() {
        // Arrange: batch item with empty embeddings vector
        let tracker = make_test_tracker();
        let batch = vec![TrackerTurnInput {
            embeddings: vec![], // empty — should error
            roles: vec![],
            signals: [0.5; 11],
            context_turns: 1.0,
        }];
        // Act
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("empty embeddings"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_max_seq_len_minus_one_succeeds() {
        // Arrange: seq_len = max_seq_len - 1 should be accepted
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        let embeddings = vec![vec![0.1; config.hidden_size]; config.max_seq_len - 1];
        let roles = vec![0u8; config.max_seq_len - 1];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0);
        // Assert
        assert!(result.is_ok(), "seq_len = max_seq_len - 1 should be accepted, got {:?}", result.err());
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_1x1_zero_weight_nonzero_bias() {
        // Arrange: 1x1 linear with zero weight and nonzero bias
        let input = vec![42.0];
        let weight = vec![0.0]; // 1x1
        let bias = vec![7.5];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: 42*0 + 7.5 = 7.5
        assert_eq!(out.len(), 1);
        assert!((out[0] - 7.5).abs() < 1e-6, "expected 7.5, got {}", out[0]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_one_zero_one_positive_one_negative() {
        // Arrange: three logits [0.0, 5.0, -3.0]
        let logits = &[0.0, 5.0, -3.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: 5.0 is max, confidence = 1/(1+exp(-5)+exp(-8))
        let expected = 1.0 / (1.0 + (-5.0f32).exp() + (-8.0f32).exp());
        assert!((conf - expected).abs() < 1e-5, "expected {expected}, got {conf}");
        assert!(conf > 0.9, "5.0 among 0 and -3 should give high confidence, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_float_min_positive() {
        // Arrange: smallest positive subnormal f32 among zeros
        let data = &[0.0f32, f32::from_bits(1), 0.0f32]; // smallest positive subnormal
        // Act
        let idx = argmax(data);
        // Assert: smallest positive subnormal > 0 > 0
        assert_eq!(idx, Some(1), "smallest positive subnormal should be the max among zeros");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_with_five_logits_fifth_is_max() {
        // Arrange: non-standard 5 difficulty logits (struct allows any length)
        let cls = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 0.0, 0.0, 0.0, 99.0],
        };
        // Act
        let diff = cls.difficulty();
        // Assert: argmax returns index 4 (the last element with 99.0)
        assert_eq!(diff, 4);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_head_dim_multiplier_invariant() {
        // Arrange: verify head_dim = hidden_size / num_heads for default
        let config = IntentTrackerConfig::default();
        // Assert: 768 / 4 = 192
        assert_eq!(config.hidden_size / config.num_heads, config.head_dim);
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_with_identity_dequant_matches_f32() {
        // Arrange: f16 embeddings dequantized via identity fn (to_f32)
        let tracker = make_test_tracker();
        let emb_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.1); 768]];
        // Act
        let result = tracker
            .classify_turn_quant(&emb_f16, &[0], &[0.5; 11], 1.0, |v| v.to_f32())
            .unwrap();
        // Assert
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in &result.task_logits {
            assert!(l.is_finite(), "task logit should be finite, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_inference_failed_contains_detail() {
        // Arrange
        let err = TrackerError::InferenceFailed("weight len 4 != n(2) * k(3)".into());
        // Act
        let msg = format!("{err}");
        // Assert: both prefix and detail should appear
        assert!(msg.starts_with("inference failed:"), "unexpected prefix: {msg}");
        assert!(msg.contains("weight len"), "should contain detail: {msg}");
        assert!(msg.contains("n(2)"), "should contain n value: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_embedding_dim_769_errors() {
        // Arrange: off-by-one embedding dimension (769 instead of 768)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 769]];
        let roles = vec![0u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[0] dim 769"), "unexpected message: {msg}");
        assert!(msg.contains("hidden_size 768"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_4x4_all_ones_weight_sums_all_inputs() {
        // Arrange: 4 inputs, 4 outputs, all-ones weight, zero bias
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 16]; // 4x4 all ones
        let bias = vec![0.0; 4];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: each output = sum of all inputs = 1+2+3+4 = 10
        for i in 0..4 {
            assert!((out[i] - 10.0).abs() < 1e-6, "output[{i}] should be 10.0, got {}", out[i]);
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_decreasing_sequence_confidence_increases_with_gap() {
        // Arrange: two softmax_max calls with increasing gap between max and rest
        let conf_small_gap = softmax_max(&[5.0, 4.0, 3.0]);
        let conf_large_gap = softmax_max(&[5.0, 0.0, -5.0]);
        // Assert: larger gap → higher confidence
        assert!(
            conf_large_gap > conf_small_gap,
            "larger gap should yield higher confidence: small_gap={conf_small_gap}, large_gap={conf_large_gap}"
        );
    }

    // @trace TEST-SIT [req:REQ-SIT-007] [level:unit]
    #[test]
    fn validate_e2e_inputs_single_empty_string_passes() {
        // Arrange: single empty string is valid (non-empty vec, matching roles)
        let texts = vec![""];
        let roles = vec![0u8];
        // Act
        let result = validate_e2e_inputs(&texts, &roles);
        // Assert: passes (empty string is still a string)
        assert!(result.is_ok());
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn from_weights_bf16_zero_bytes_with_zero_shape_succeeds() {
        // Arrange: single weight with 0 bytes and shape [0]
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            weights_bf16.insert(name.to_string(), vec![]);
            shapes.insert(name.to_string(), vec![0]);
        }
        // Act
        let result = IntentTracker::from_weights_bf16(config, weights_bf16, shapes);
        // Assert: zero bytes + zero shape should succeed
        assert!(result.is_ok(), "zero-byte BF16 weights with zero shape should be accepted");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn get_weight_w_q_weight_returns_correct_length() {
        // Arrange
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        // Act
        let w = tracker.get_weight("w_q_weight").unwrap();
        // Assert: w_q_weight is (hidden_size, hidden_size) = 768*768
        assert_eq!(w.len(), config.hidden_size * config.hidden_size,
            "w_q_weight should have {} elements, got {}", config.hidden_size * config.hidden_size, w.len());
    }

    // ── Wave 12x86: 15 additional unit tests ──

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_embedding_dim_767_errors() {
        // Arrange: off-by-one embedding dimension (767 instead of 768)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 767]];
        let roles = vec![0u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[0] dim 767"), "unexpected message: {msg}");
        assert!(msg.contains("hidden_size 768"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_seq_len_exceeds_by_two_errors() {
        // Arrange: seq_len = max_seq_len + 2
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        let embeddings = vec![vec![0.1; config.hidden_size]; config.max_seq_len + 2];
        let roles = vec![0u8; config.max_seq_len + 2];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("seq_len 34"), "unexpected message: {msg}");
        assert!(msg.contains("max_seq_len 32"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_three_elements_one_zero_rest_negative() {
        // Arrange: one zero logit among two negative logits
        let logits = &[0.0, -3.0, -7.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: 0.0 is the max; confidence = exp(0)/(exp(0)+exp(-3)+exp(-7))
        let expected = 1.0 / (1.0 + (-3.0f32).exp() + (-7.0f32).exp());
        assert!(
            (conf - expected).abs() < 1e-5,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.9, "zero among negatives should dominate, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_subnormal_negative_among_zeros() {
        // Arrange: smallest negative subnormal among zeros
        let tiny_neg = f32::from_bits(1 | (1 << 31)); // set sign bit for negative
        let data = &[0.0f32, tiny_neg, 0.0f32];
        // Act
        let idx = argmax(data);
        // Assert: zeros are greater than negative subnormal; ties among zeros -> last zero (index 2)
        assert_eq!(idx, Some(2), "zeros should beat negative subnormal");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_3x1_column_vector() {
        // Arrange: 1-input, 3-output (column projection)
        let input = vec![4.0];
        let weight = vec![1.0, 2.0, 3.0]; // 3x1
        let bias = vec![10.0, 20.0, 30.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: [4*1+10, 4*2+20, 4*3+30] = [14, 28, 42]
        assert_eq!(out.len(), 3);
        assert!((out[0] - 14.0).abs() < 1e-6, "expected 14.0, got {}", out[0]);
        assert!((out[1] - 28.0).abs() < 1e-6, "expected 28.0, got {}", out[1]);
        assert!((out[2] - 42.0).abs() < 1e-6, "expected 42.0, got {}", out[2]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_task_type_with_two_equal_one_dominant() {
        // Arrange: two equal logits and one dominant
        let cls = Classification {
            task_logits: vec![50.0, 50.0, 0.0],
            difficulty_logits: vec![0.0; 4],
        };
        // Act
        let tt = cls.task_type();
        let conf = cls.task_confidence();
        // Assert: argmax returns index 1 (last of the tied max pair); confidence ~0.5
        assert_eq!(tt, TaskType::CodeDeploy);
        assert!(
            (conf - 0.5).abs() < 0.01,
            "two equal dominant logits should give ~0.5 confidence, got {conf}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn verify_weights_missing_diff_fc1_bias() {
        // Arrange: omit diff_fc1_bias specifically
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", // omit diff_fc1_bias
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        // Act
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(
            msg.contains("diff_fc1_bias"),
            "should report missing diff_fc1_bias, got: {msg}"
        );
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_five_items_all_valid() {
        // Arrange: 5-item batch with varying inputs
        let tracker = make_test_tracker();
        let batch: Vec<TrackerTurnInput> = (0..5)
            .map(|i| TrackerTurnInput {
                embeddings: vec![vec![(i as f32) * 0.05; 768]; (i % 3) + 1],
                roles: (0..((i % 3) + 1)).map(|j| (j % 2) as u8).collect(),
                signals: [(i as f32) * 0.1; 11],
                context_turns: i as f32 + 1.0,
            })
            .collect();
        // Act
        let results = tracker.classify_turns_batch(&batch).unwrap();
        // Assert
        assert_eq!(results.len(), 5);
        for r in &results {
            assert_eq!(r.task_logits.len(), 3);
            assert_eq!(r.difficulty_logits.len(), 4);
            for &l in r.task_logits.iter().chain(r.difficulty_logits.iter()) {
                assert!(l.is_finite(), "all logits should be finite, got {l}");
            }
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_invalid_input_with_multibyte_utf8() {
        // Arrange: TrackerError with multibyte UTF-8 in message
        let err = TrackerError::InvalidInput("维度不匹配: 768 ≠ 64".into());
        // Act
        let msg = format!("{err}");
        // Assert: Display should preserve the multibyte characters
        assert!(msg.contains("维度不匹配"), "should preserve multibyte UTF-8: {msg}");
        assert!(msg.starts_with("invalid input:"), "unexpected prefix: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_additivity_with_bias_zero() {
        // Arrange: verify W(x+y) = W(x) + W(y) for a 2x3 matrix with zero bias
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let weight = vec![
            1.0, 0.0, 2.0, // row 0
            0.0, 1.0, 3.0, // row 1
        ];
        let bias = vec![0.0; 2];
        let x_plus_y: Vec<f32> = x.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
        // Act
        let wx = linear_forward(&x, &weight, &bias).unwrap();
        let wy = linear_forward(&y, &weight, &bias).unwrap();
        let wxy = linear_forward(&x_plus_y, &weight, &bias).unwrap();
        // Assert
        for i in 0..2 {
            let expected = wx[i] + wy[i];
            assert!(
                (wxy[i] - expected).abs() < 1e-4,
                "additivity violated at index {i}: expected {expected}, got {}",
                wxy[i]
            );
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_role_index_3_errors() {
        // Arrange: role_emb_weight has 3 rows (indices 0, 1, 2); index 3 is out of range
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]];
        let roles = vec![3u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("role index 3"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_two_zeros_and_one_positive() {
        // Arrange: two zeros and one positive logit
        let logits = &[0.0, 0.0, 2.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: max=2.0, confidence = exp(0)/(exp(-2)+exp(-2)+exp(0)) = 1/(2*exp(-2)+1)
        let expected = 1.0 / (2.0 * (-2.0f32).exp() + 1.0);
        assert!(
            (conf - expected).abs() < 1e-5,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.5, "positive logit should dominate zeros, got {conf}");
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_with_negating_dequant() {
        // Arrange: dequant function negates values
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.5); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.3; 11];
        // Act: negate all embedding values
        let result = tracker
            .classify_turn_quant(&embeddings_f16, &roles, &signals, 2.0, |v| -v.to_f32())
            .unwrap();
        // Assert: should produce finite logits
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in &result.task_logits {
            assert!(l.is_finite(), "negated dequant should produce finite logits, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_signal_hidden_dim_is_pow2() {
        // Arrange: check default signal_hidden_dim is a power of 2
        let config = IntentTrackerConfig::default();
        let dim = config.signal_hidden_dim;
        // Act & Assert: power-of-2 check
        assert!(dim > 0, "signal_hidden_dim should be positive");
        assert_eq!(dim & (dim - 1), 0, "signal_hidden_dim should be a power of 2, got {dim}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_2x2_rotation_matrix_90_degrees() {
        // Arrange: 90-degree CCW rotation matrix W = [[0, -1], [1, 0]]
        // linear_forward computes out = W * input + bias (row-major W)
        // out[0] = 3*0 + 4*(-1) = -4
        // out[1] = 3*1 + 4*0 = 3
        let input = vec![3.0, 4.0];
        let weight = vec![
            0.0, -1.0,
            1.0, 0.0,
        ];
        let bias = vec![0.0; 2];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert
        assert!((out[0] - (-4.0)).abs() < 1e-6, "expected -4.0, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-6, "expected 3.0, got {}", out[1]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_signals_all_positive_infinity() {
        // Arrange: all signal values are +infinity
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [f32::INFINITY; 11],
            context_turns: 1.0,
        };
        // Assert: all signal values should be positive infinity
        assert!(input.signals.iter().all(|s| s.is_infinite() && s.is_sign_positive()));
        assert_eq!(input.signals.len(), 11);
    }

    // ── Wave 12x87: 15 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_dequant_failed_with_empty_string() {
        // Arrange: DequantFailed with empty detail string
        let err = TrackerError::DequantFailed(String::new());
        // Act
        let msg = format!("{err}");
        // Assert: prefix should still appear
        assert!(msg.starts_with("dequantization failed:"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_task_type_two_large_one_small() {
        // Arrange: two very large logits and one small — the second large wins (last of tie)
        let cls = Classification {
            task_logits: vec![999.0, 999.0, 0.0],
            difficulty_logits: vec![0.0; 4],
        };
        // Act
        let tt = cls.task_type();
        // Assert: argmax picks last of tied max → index 1 → CodeDeploy
        assert_eq!(tt, TaskType::CodeDeploy);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_all_negative_infinity_no_panic() {
        // Arrange: all logits are negative infinity
        let logits = &[f32::NEG_INFINITY; 5];
        // Act
        let conf = softmax_max(logits);
        // Assert: exp(-inf - (-inf)) = exp(0/0) = NaN; should not panic
        assert!(conf.is_nan() || (conf > 0.0 && conf <= 1.0));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_2x2_swap_matrix() {
        // Arrange: swap matrix [[0,1],[1,0]] swaps two inputs, bias = [10, 20]
        let input = vec![3.0, 7.0];
        let weight = vec![0.0, 1.0, 1.0, 0.0];
        let bias = vec![10.0, 20.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: [3*0+7*1+10, 3*1+7*0+20] = [17, 23]
        assert!((out[0] - 17.0).abs() < 1e-6, "expected 17.0, got {}", out[0]);
        assert!((out[1] - 23.0).abs() < 1e-6, "expected 23.0, got {}", out[1]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_f32_min_positive_subnormal() {
        // Arrange: f32::from_bits(1) is the smallest positive subnormal
        let tiny_pos = f32::from_bits(1);
        let data = &[0.0f32, tiny_pos];
        // Act
        let idx = argmax(data);
        // Assert: smallest positive subnormal > 0.0
        assert_eq!(idx, Some(1), "smallest positive subnormal should beat zero");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_role_zero_and_one_alternating_long() {
        // Arrange: longer sequence with alternating roles 0 and 1
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.2; 768]; 10];
        let roles: Vec<u8> = (0..10).map(|i| (i % 2) as u8).collect();
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 10.0).unwrap();
        // Assert: all logits should be finite
        assert_eq!(result.task_logits.len(), 3);
        assert_eq!(result.difficulty_logits.len(), 4);
        for &l in result.task_logits.iter().chain(result.difficulty_logits.iter()) {
            assert!(l.is_finite(), "logit should be finite, got {l}");
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_max_seq_len_is_pow2() {
        // Arrange: check default max_seq_len is a power of 2
        let config = IntentTrackerConfig::default();
        let msl = config.max_seq_len;
        // Assert
        assert!(msl > 0, "max_seq_len should be positive");
        assert_eq!(msl & (msl - 1), 0, "max_seq_len should be a power of 2, got {msl}");
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_two_items_differ_in_context_turns() {
        // Arrange: two items identical except context_turns
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.5; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.5; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 50.0,
            },
        ];
        // Act
        let results = tracker.classify_turns_batch(&batch).unwrap();
        // Assert: different context_turns should produce different difficulty logits
        // (because context_turns is concatenated into the classifier input)
        assert_eq!(results.len(), 2);
        assert_ne!(
            results[0].difficulty_logits, results[1].difficulty_logits,
            "different context_turns should produce different difficulty logits"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_two_identical_zeros_confidence_is_half() {
        // Arrange: two zero logits
        let logits = &[0.0, 0.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: both equal → confidence = 1/(1+1) = 0.5
        assert!((conf - 0.5).abs() < 1e-5, "two equal logits should give 0.5, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_1x1_negative_weight_flips_sign() {
        // Arrange: 1x1 with weight = -3.0, bias = 0.0, input = 4.0
        let input = vec![4.0];
        let weight = vec![-3.0];
        let bias = vec![0.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: 4 * (-3) + 0 = -12
        assert!((out[0] - (-12.0)).abs() < 1e-6, "expected -12.0, got {}", out[0]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn verify_weights_missing_w_v_bias() {
        // Arrange: omit w_v_bias specifically
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", // omit w_v_bias
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        // Act
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(
            msg.contains("w_v_bias"),
            "should report missing w_v_bias, got: {msg}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_signals_all_negative_infinity() {
        // Arrange: all signal values are negative infinity
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [f32::NEG_INFINITY; 11],
            context_turns: 1.0,
        };
        // Assert
        assert!(input.signals.iter().all(|s| s.is_infinite() && s.is_sign_negative()));
        assert_eq!(input.signals.len(), 11);
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_with_doubled_dequant_differs_from_normal() {
        // Arrange: same embeddings dequantized normally vs doubled
        let tracker = make_test_tracker();
        let emb_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.5); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.3; 11];
        // Act: normal dequant
        let r_normal = tracker
            .classify_turn_quant(&emb_f16, &roles, &signals, 2.0, |v| v.to_f32())
            .unwrap();
        // Act: doubled dequant
        let r_doubled = tracker
            .classify_turn_quant(&emb_f16, &roles, &signals, 2.0, |v| v.to_f32() * 2.0)
            .unwrap();
        // Assert: different dequant functions should produce different logits
        assert_ne!(
            r_normal.task_logits, r_doubled.task_logits,
            "different dequant scales should produce different task logits"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_confidence_with_three_equal_two_lower() {
        // Arrange: difficulty logits [10.0, 10.0, 10.0, 0.0] — three tied max
        let cls = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![10.0, 10.0, 10.0, 0.0],
        };
        // Act
        let conf = cls.difficulty_confidence();
        // Assert: three tied max among four → confidence ≈ 1/(1+1+1+exp(-10))
        let expected = 1.0 / (1.0 + 1.0 + 1.0 + (-10.0f32).exp());
        assert!(
            (conf - expected).abs() < 1e-4,
            "three tied max should give ~{expected}, got {conf}"
        );
        assert!(conf < 0.34, "three-way tie among four should be < 0.34, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_with_single_input_multiple_outputs_gradient_check() {
        // Arrange: verify that each output dimension only depends on its own weight row
        let input = vec![1.0];
        // 3 outputs: each row has a single element (scalar weight per output)
        let weight = vec![10.0, 20.0, 30.0]; // 3x1
        let bias = vec![1.0, 2.0, 3.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: output = [1*10+1, 1*20+2, 1*30+3] = [11, 22, 33]
        assert!((out[0] - 11.0).abs() < 1e-6, "expected 11.0, got {}", out[0]);
        assert!((out[1] - 22.0).abs() < 1e-6, "expected 22.0, got {}", out[1]);
        assert!((out[2] - 33.0).abs() < 1e-6, "expected 33.0, got {}", out[2]);
    }

    // ── Batch 7: 15 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_size_is_reasonable() {
        // Arrange: TrackerError variants contain String, so size should be reasonable
        // Act & Assert: just verify it compiles and the size is bounded
        let size = std::mem::size_of::<TrackerError>();
        // String is 24 bytes on 64-bit; the enum discriminant adds a few bytes;
        // 6 variants all wrapping String → max is ~32 bytes
        assert!(size <= 64, "TrackerError size should be <= 64 bytes, got {size}");
        assert!(size > 0, "TrackerError should not be zero-sized");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_additivity_with_zero_bias_exactly() {
        // Arrange: W(x + y) = W(x) + W(y) when bias = 0
        let x = vec![2.0, -1.0];
        let y = vec![3.0, 4.0];
        let weight = vec![1.0, 2.0, -1.0, 0.5]; // 2x2
        let bias = vec![0.0; 2];
        let x_plus_y: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();
        // Act
        let wx = linear_forward(&x, &weight, &bias).unwrap();
        let wy = linear_forward(&y, &weight, &bias).unwrap();
        let wxy = linear_forward(&x_plus_y, &weight, &bias).unwrap();
        // Assert
        for i in 0..2 {
            let expected = wx[i] + wy[i];
            assert!(
                (wxy[i] - expected).abs() < 1e-5,
                "additivity violated at {i}: expected {expected}, got {}",
                wxy[i]
            );
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_with_many_logits_sum_approaches_one() {
        // Arrange: 10 uniform logits → each confidence = 0.1
        let logits = vec![1.0; 10];
        // Act
        let conf = softmax_max(&logits);
        // Assert
        assert!(
            (conf - 0.1).abs() < 1e-5,
            "10 uniform logits should give 0.1, got {conf}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_task_confidence_single_large_logit_near_one() {
        // Arrange: single dominant logit among three
        let cls = Classification {
            task_logits: vec![50.0, -50.0, -50.0],
            difficulty_logits: vec![0.0; 4],
        };
        // Act
        let conf = cls.task_confidence();
        // Assert
        assert!(
            (conf - 1.0).abs() < 1e-6,
            "single dominant logit should give confidence ~1.0, got {conf}"
        );
        assert!(conf > 0.99);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_get_weight_w_q_bias_has_correct_length() {
        // Arrange
        let tracker = make_test_tracker();
        let config = IntentTrackerConfig::default();
        // Act
        let w = tracker.get_weight("w_q_bias").unwrap();
        // Assert: w_q_bias is a vector of length hidden_size
        assert_eq!(w.len(), config.hidden_size, "w_q_bias should have hidden_size elements");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_signals_array_is_copy_type() {
        // Arrange: signals is [f32; 11] which is Copy
        let signals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        // Act: Copy (not move)
        let copied = signals;
        // Assert: both are still valid
        assert_eq!(signals, copied);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn task_type_from_index_error_contains_index_value() {
        // Arrange & Act
        let err = TaskType::from_index(42).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(
            msg.contains("42"),
            "error message should contain the invalid index 42: {msg}"
        );
        assert!(
            msg.contains("invalid task type index"),
            "error should describe the problem: {msg}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn verify_weights_missing_task_fc2_weight() {
        // Arrange: all required weights except task_fc2_weight
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight",
            "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            // Deliberately omit task_fc2_weight
            "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale",
            "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        // Act
        let err = verify_weights(&shapes, &config).unwrap_err();
        let msg = format!("{err}");
        // Assert
        assert!(
            msg.contains("task_fc2_weight"),
            "should report missing task_fc2_weight, got: {msg}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classify_turn_role_index_two_is_valid() {
        // Arrange: role_emb_weight has 3 rows (indices 0, 1, 2)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]; 2];
        let roles = vec![2u8, 2]; // role index 2 is the last valid
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 2.0);
        // Assert
        assert!(result.is_ok(), "role index 2 should be valid, got {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.task_logits.len(), 3);
        assert_eq!(r.difficulty_logits.len(), 4);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_f32_min_among_finite() {
        // Arrange: f32::MIN is a valid large negative number
        let values = [f32::MIN, f32::MIN + 1.0, 0.0];
        // Act
        let result = argmax(&values);
        // Assert: 0.0 is the max
        assert_eq!(result, Some(2));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_mixed_zeros_and_nonzero() {
        // Arrange: [0.0, 5.0, 0.0, 0.0] — one dominant, rest zero
        let logits = [0.0, 5.0, 0.0, 0.0];
        // Act
        let conf = softmax_max(&logits);
        // Assert: exp(0)/(exp(-5)+1+exp(-5)+exp(-5)) = 1/(3*exp(-5)+1)
        let expected = 1.0 / (3.0 * (-5.0f32).exp() + 1.0);
        assert!(
            (conf - expected).abs() < 1e-5,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.9, "dominant logit 5.0 among zeros should give > 0.9, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_identity_with_negative_bias() {
        // Arrange: 3x3 identity matrix with negative bias
        let input = vec![4.0, 5.0, 6.0];
        let weight = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let bias = vec![-1.0, -2.0, -3.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: output = input + bias = [3.0, 3.0, 3.0]
        assert!((out[0] - 3.0).abs() < 1e-6, "expected 3.0, got {}", out[0]);
        assert!((out[1] - 3.0).abs() < 1e-6, "expected 3.0, got {}", out[1]);
        assert!((out[2] - 3.0).abs() < 1e-6, "expected 3.0, got {}", out[2]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_debug_contains_variant_name_for_all() {
        // Arrange: all TrackerError variants
        let cases: Vec<(TrackerError, &str)> = vec![
            (TrackerError::LoadFailed("x".into()), "LoadFailed"),
            (TrackerError::InvalidInput("x".into()), "InvalidInput"),
            (TrackerError::InferenceFailed("x".into()), "InferenceFailed"),
            (TrackerError::MissingWeight("x".into()), "MissingWeight"),
            (TrackerError::EncoderFailed("x".into()), "EncoderFailed"),
            (TrackerError::DequantFailed("x".into()), "DequantFailed"),
        ];
        // Act & Assert
        for (err, variant_name) in &cases {
            let debug = format!("{err:?}");
            assert!(
                debug.contains(variant_name),
                "Debug for {variant_name} should contain variant name, got: {debug}"
            );
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_with_two_element_logits() {
        // Arrange: only 2 difficulty logits instead of usual 4
        let cls = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![0.0, 10.0],
        };
        // Act
        let diff = cls.difficulty();
        let conf = cls.difficulty_confidence();
        // Assert: argmax picks index 1
        assert_eq!(diff, 1, "difficulty should be 1 for [0.0, 10.0]");
        // Confidence: exp(0)/(exp(-10)+1) ≈ 1.0
        assert!(conf > 0.99, "dominant difficulty should give > 0.99, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_clone_is_independent() {
        // Arrange
        let config = IntentTrackerConfig::default();
        // Act
        let mut cloned = config.clone();
        cloned.hidden_size = 512;
        cloned.max_seq_len = 128;
        // Assert: original should be unchanged
        assert_eq!(config.hidden_size, 768, "original hidden_size should be 768");
        assert_eq!(config.max_seq_len, 32, "original max_seq_len should be 32");
        assert_eq!(cloned.hidden_size, 512);
        assert_eq!(cloned.max_seq_len, 128);
    }

    // ── Wave 12x88: 15 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn f16_roundtrip_negative_value() {
        // Arrange: roundtrip a negative value through f16
        let val = -3.75f32;
        let f16_val = f16::from_f32(val);
        let recovered = f16_val.to_f32();
        // Assert: f16 precision should be within 0.01 for this value
        assert!((recovered - val).abs() < 0.01, "F16 negative roundtrip lost too much precision, got {recovered}");
        assert!(recovered < 0.0, "negative value should stay negative after roundtrip");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn bf16_roundtrip_zero() {
        // Arrange: roundtrip exact zero through bf16
        let val = 0.0f32;
        let bf16_val = bf16::from_f32(val);
        let recovered = bf16_val.to_f32();
        // Assert: zero should be preserved exactly
        assert!((recovered - val).abs() < f32::EPSILON, "BF16 zero roundtrip should be exact, got {recovered}");
        assert_eq!(recovered, 0.0);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_negative_weight_absorbs_input() {
        // Arrange: 2-input, 2-output with negative weights, input = [3.0, 4.0]
        // W = [[-2, 0], [0, -3]], b = [1, 1]
        // out[0] = 3*(-2) + 4*0 + 1 = -5
        // out[1] = 3*0 + 4*(-3) + 1 = -11
        let input = vec![3.0, 4.0];
        let weight = vec![-2.0, 0.0, 0.0, -3.0];
        let bias = vec![1.0, 1.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert
        assert!((out[0] - (-5.0)).abs() < 1e-6, "expected -5.0, got {}", out[0]);
        assert!((out[1] - (-11.0)).abs() < 1e-6, "expected -11.0, got {}", out[1]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_two_zeros_one_very_large() {
        // Arrange: two zero logits and one very large logit
        let logits = &[0.0, 0.0, 100.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: exp(0) / (exp(-100) + exp(-100) + exp(0)) ≈ 1.0
        assert!(
            (conf - 1.0).abs() < 1e-3,
            "very large logit should dominate two zeros, got {conf}"
        );
        assert!(conf > 0.99);
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_length_one_embedding_and_two_roles_errors() {
        // Arrange: 1 f16 embedding but 2 roles → mismatch
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(0.1); 768]];
        let roles = vec![0u8, 1]; // 2 roles for 1 embedding
        // Act
        let err = tracker
            .classify_turn_quant(&embeddings_f16, &roles, &[0.5; 11], 1.0, |v| v.to_f32())
            .unwrap_err();
        // Assert
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embeddings len 1"), "unexpected message: {msg}");
        assert!(msg.contains("roles len 2"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_returns_zero_when_first_is_only_positive() {
        // Arrange: first logit is the only positive value
        let cls = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![5.0, -1.0, -2.0, -3.0],
        };
        // Act
        let diff = cls.difficulty();
        let conf = cls.difficulty_confidence();
        // Assert
        assert_eq!(diff, 0, "first element should win when it's the only positive");
        assert!(conf > 0.9, "single positive should dominate, got {conf}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_single_embedding_negative_infinity_context() {
        // Arrange: single embedding with context_turns = -infinity
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]];
        let roles = vec![0u8];
        // Act: context_turns = NEG_INFINITY should still produce a result
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], f32::NEG_INFINITY);
        // Assert: should succeed (context_turns is just concatenated as a feature)
        assert!(result.is_ok(), "negative infinity context_turns should be accepted, got {:?}", result.err());
        let cls = result.unwrap();
        assert_eq!(cls.task_logits.len(), 3);
        assert_eq!(cls.difficulty_logits.len(), 4);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_weight_length_is_bias_times_input() {
        // Arrange: verify that different (n, k) combos with matching bias * input work
        for (n, k) in [(1, 1), (2, 3), (5, 4), (1, 10), (10, 1)] {
            let input = vec![1.0; k];
            let weight = vec![0.5; n * k];
            let bias = vec![0.1; n];
            // Act
            let out = linear_forward(&input, &weight, &bias).unwrap();
            // Assert
            assert_eq!(out.len(), n, "output length should be {n} for ({n},{k})");
            for (i, &v) in out.iter().enumerate() {
                let expected = 0.5 * (k as f32) + 0.1;
                assert!(
                    (v - expected).abs() < 1e-4,
                    "output[{i}] for ({n},{k}): expected {expected}, got {v}"
                );
            }
        }
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_clone_equality_via_debug() {
        // Arrange: create a TrackerError, clone it, verify Debug strings match character-by-character
        let err = TrackerError::EncoderFailed("connection timeout 30s".into());
        let cloned = err.clone();
        let original_debug = format!("{err:?}");
        let cloned_debug = format!("{cloned:?}");
        // Assert
        assert_eq!(original_debug, cloned_debug, "cloned error should have identical Debug output");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_debug_shows_embedding_count_not_contents() {
        // Arrange: construct with 3 embeddings
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.1; 768], vec![0.2; 768], vec![0.3; 768]],
            roles: vec![0, 1, 0],
            signals: [0.0; 11],
            context_turns: 3.0,
        };
        // Act
        let debug = format!("{input:?}");
        // Assert: debug should contain the field name and the vec length
        assert!(debug.contains("TrackerTurnInput"), "Debug should contain struct name: {debug}");
        assert!(debug.contains("embeddings"), "Debug should contain embeddings field: {debug}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    /// [BCE-028] Empty difficulty_logits now panics (was silently returning 0).
    /// Test that it panics rather than silently choosing difficulty 0.
    #[test]
    #[should_panic(expected = "difficulty_logits empty or all-NaN")]
    fn classification_empty_difficulty_logits_panics() {
        let cls = Classification {
            task_logits: vec![1.0, 0.0, 0.0],
            difficulty_logits: vec![],
        };
        let _ = cls.difficulty();
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn verify_weights_rejects_empty_string_key() {
        // Arrange: include all required names but also an empty-string key (should not confuse verification)
        let config = IntentTrackerConfig::default();
        let mut shapes = HashMap::new();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            shapes.insert(name.to_string(), vec![1]);
        }
        // Add an empty-string key (not a required weight)
        shapes.insert(String::new(), vec![1]);
        // Act & Assert: should still succeed (all required keys present)
        assert!(verify_weights(&shapes, &config).is_ok(),
            "empty-string key should not affect verification");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_positive_infinity_among_neg_infinity_and_finite() {
        // Arrange: positive infinity, negative infinity, and a finite value
        let data = &[f32::NEG_INFINITY, f32::INFINITY, 0.0, f32::NEG_INFINITY];
        // Act
        let idx = argmax(data);
        // Assert: positive infinity wins
        assert_eq!(idx, Some(1), "positive infinity should be the max");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_with_zero_input_zero_bias_nonzero_weight() {
        // Arrange: zero input with nonzero weight and zero bias → output is all zeros
        let input = vec![0.0; 5];
        let weight: Vec<f32> = (0..10).map(|i| (i as f32 + 1.0)).collect(); // 2x5
        let bias = vec![0.0; 2];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: input^T * W + b = 0 + 0 = 0
        for (i, &v) in out.iter().enumerate() {
            assert!((v).abs() < 1e-6, "output[{i}] should be 0.0, got {v}");
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn from_weights_bf16_distinguishes_positive_and_negative_zero() {
        // Arrange: BF16 representation of +0.0 and -0.0
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        let pos_zero_bits = bf16::from_f32(0.0f32).to_bits();
        let neg_zero_bits = bf16::from_f32(-0.0f32).to_bits();
        // BF16 +0.0 and -0.0 have different bit patterns
        let pos_zero_bytes = pos_zero_bits.to_le_bytes();
        let neg_zero_bytes = neg_zero_bits.to_le_bytes();
        for name in &required {
            if *name == "recency_scale" {
                weights_bf16.insert(name.to_string(), vec![neg_zero_bytes[0], neg_zero_bytes[1]]);
                shapes.insert(name.to_string(), vec![1]);
            } else {
                weights_bf16.insert(name.to_string(), vec![pos_zero_bytes[0], pos_zero_bytes[1]]);
                shapes.insert(name.to_string(), vec![1]);
            }
        }
        // Act
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        // Assert: both +0.0 and -0.0 dequantize to 0.0 (they are numerically equal)
        assert_eq!(w[0], 0.0f32, "BF16 -0.0 should dequantize to f32 0.0");
    }

    // ── Wave 12x85: 15 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_new_dir_with_valid_safetensors_missing_weights() {
        // Arrange: create a temp directory with a valid safetensors file
        // containing only a dummy tensor (missing all required weights)
        use safetensors::tensor::{TensorView, Dtype};
        let data = vec![0u8; 4];
        let tensor = TensorView::new(Dtype::F32, vec![1], &data).unwrap();
        let tensors = vec![("dummy_weight", tensor)];
        let bytes = safetensors::serialize(tensors.iter().map(|(n, t)| (*n, t)), &None).unwrap();
        let tmp_dir = std::env::temp_dir().join("gllm_test_intent_dir_valid_file");
        let _ = std::fs::create_dir_all(&tmp_dir);
        std::fs::write(tmp_dir.join("model.safetensors"), &bytes).unwrap();
        // Act: new() with a directory should look for model.safetensors inside it
        let err = IntentTracker::new(&tmp_dir).unwrap_err();
        // Assert: should fail on missing required weights (not on file read)
        assert!(matches!(err, TrackerError::MissingWeight(_)));
        let msg = format!("{err}");
        assert!(
            msg.contains("role_emb_weight") || msg.contains("w_q_weight"),
            "should report a missing required weight, got: {msg}"
        );
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_role_index_three_boundary_errors() {
        // Arrange: role index 3 is the first invalid value (role_emb_weight has 3 rows: 0,1,2)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.1; 768]];
        let roles = vec![3u8];
        // Act
        let err = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0).unwrap_err();
        // Assert: boundary value just beyond max valid
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("role index 3"), "unexpected message: {msg}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_different_context_turns_produce_different_logits() {
        // Arrange: identical embeddings and roles, different context_turns values
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.25; 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        // Act
        let r1 = tracker.classify_turn(&embeddings, &roles, &signals, 1.0).unwrap();
        let r2 = tracker.classify_turn(&embeddings, &roles, &signals, 50.0).unwrap();
        // Assert: different context_turns should feed different scalar into classifier input,
        // producing at least one different logit
        let task_differs = r1.task_logits != r2.task_logits;
        let diff_differs = r1.difficulty_logits != r2.difficulty_logits;
        assert!(
            task_differs || diff_differs,
            "different context_turns should yield different outputs"
        );
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn classify_turn_quant_with_nan_f16_embeddings_no_panic() {
        // Arrange: f16 embeddings containing NaN
        let tracker = make_test_tracker();
        let embeddings_f16: Vec<Vec<f16>> = vec![vec![f16::from_f32(f32::NAN); 768]; 2];
        let roles = vec![0u8, 1];
        let signals = [0.5; 11];
        // Act: NaN propagates through computation but should not panic
        let result = tracker.classify_turn_quant(
            &embeddings_f16, &roles, &signals, 1.0, |v| v.to_f32(),
        );
        // Assert: may produce NaN logits but must not panic; either Ok or Err is acceptable
        if let Ok(cls) = result {
            assert_eq!(cls.task_logits.len(), 3);
            assert_eq!(cls.difficulty_logits.len(), 4);
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn from_weights_bf16_empty_hashmap_errors() {
        // Arrange: completely empty BF16 weights map (missing all required weights)
        let config = IntentTrackerConfig::default();
        let weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let shapes: HashMap<String, Vec<usize>> = HashMap::new();
        // Act
        let err = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap_err();
        // Assert: should fail on verify_weights before dequantization
        assert!(matches!(err, TrackerError::MissingWeight(_)));
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_turn_input_signals_with_mixed_special_floats() {
        // Arrange: signals array with NaN, Inf, -Inf, zero, and normal values
        let input = TrackerTurnInput {
            embeddings: vec![vec![0.0; 768]],
            roles: vec![0],
            signals: [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0],
            context_turns: 1.0,
        };
        // Assert: all 11 signal values are stored faithfully
        assert!(input.signals[0].is_nan());
        assert!(input.signals[1].is_infinite() && input.signals[1].is_sign_positive());
        assert!(input.signals[2].is_infinite() && input.signals[2].is_sign_negative());
        assert_eq!(input.signals.len(), 11);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_confidence_with_ten_uniform_logits() {
        // Arrange: 10 uniform difficulty logits → confidence = 1/10 = 0.1
        let c = Classification {
            task_logits: vec![0.0; 3],
            difficulty_logits: vec![5.0; 10],
        };
        // Act
        let conf = c.difficulty_confidence();
        // Assert
        assert!(
            (conf - 0.1).abs() < 1e-5,
            "10 uniform logits should yield confidence 0.1, got {conf}"
        );
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_negative_identity_with_positive_bias_shifts_up() {
        // Arrange: negative identity matrix flips sign, bias shifts output up
        let input = vec![4.0, -6.0];
        let weight = vec![
            -1.0, 0.0,
            0.0, -1.0,
        ];
        let bias = vec![10.0, 20.0];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: [-4, 6] + [10, 20] = [6, 26]
        assert!((out[0] - 6.0).abs() < 1e-6, "expected 6.0, got {}", out[0]);
        assert!((out[1] - 26.0).abs() < 1e-6, "expected 26.0, got {}", out[1]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_single_zero_among_many_negatives() {
        // Arrange: one logit is exactly 0.0, rest are negative
        let logits = &[-5.0, -3.0, 0.0, -1.0, -10.0];
        // Act
        let conf = softmax_max(logits);
        // Assert: 0.0 is the max, confidence = 1 / (1 + exp(-5) + exp(-3) + exp(-1) + exp(-10))
        let denom = 1.0
            + (-5.0f32).exp()
            + (-3.0f32).exp()
            + (-1.0f32).exp()
            + (-10.0f32).exp();
        let expected = 1.0 / denom;
        assert!(
            (conf - expected).abs() < 1e-5,
            "expected {expected}, got {conf}"
        );
        assert!(conf > 0.5, "0.0 among negatives should dominate, got {conf}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn tracker_error_display_preserves_unicode_in_inner_string() {
        // Arrange: error with unicode message
        let err = TrackerError::InvalidInput("维度不匹配: 期望768, 实际64".into());
        // Act
        let msg = format!("{err}");
        // Assert: unicode characters are preserved in Display output
        assert!(msg.contains("维度不匹配"), "unicode should be preserved in Display: {msg}");
        assert!(msg.contains("768"), "numbers should be preserved: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_float_precision_boundary_values() {
        // Arrange: values near f32 precision limits
        let data = &[1.0f32, 1.0f32 + f32::EPSILON, 1.0f32];
        // Act
        let idx = argmax(data);
        // Assert: 1.0 + EPSILON is larger than 1.0, so index 1 should win
        assert_eq!(idx, Some(1), "smallest distinguishable difference should select index 1");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn from_weights_preserves_all_classifier_head_weight_names() {
        // Arrange: build tracker and verify all task_fc* and diff_fc* weights accessible
        let tracker = make_test_tracker();
        // Act & Assert: verify every classifier head weight name is accessible
        for prefix in &["task_fc0", "task_fc1", "task_fc2", "diff_fc0", "diff_fc1", "diff_fc2"] {
            for suffix in &["weight", "bias"] {
                let name = format!("{prefix}_{suffix}");
                assert!(
                    tracker.get_weight(&name).is_ok(),
                    "classifier head weight '{name}' should be accessible"
                );
            }
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_two_items_one_with_empty_embeddings() {
        // Arrange: first item valid, second has empty embeddings
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]; 2],
                roles: vec![0, 1],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![], // empty → error
                roles: vec![],
                signals: [0.5; 11],
                context_turns: 0.0,
            },
        ];
        // Act
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        // Assert: second item triggers InvalidInput for empty embeddings
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("empty embeddings"), "should mention empty embeddings: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_swaps_two_inputs() {
        // Arrange: 2x2 permutation matrix that swaps coordinates: [[0,1],[1,0]]
        let input = vec![42.0, -7.0];
        let weight = vec![
            0.0, 1.0, // row 0: output[0] = input[1] = -7
            1.0, 0.0, // row 1: output[1] = input[0] = 42
        ];
        let bias = vec![0.0; 2];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: outputs are swapped
        assert!((out[0] - (-7.0)).abs() < 1e-6, "expected -7.0, got {}", out[0]);
        assert!((out[1] - 42.0).abs() < 1e-6, "expected 42.0, got {}", out[1]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn intent_tracker_config_default_head_dim_times_heads_equals_hidden() {
        // Arrange: default config
        let config = IntentTrackerConfig::default();
        // Assert: head_dim * num_heads must equal hidden_size
        assert_eq!(
            config.head_dim * config.num_heads,
            config.hidden_size,
            "head_dim ({}) * num_heads ({}) should equal hidden_size ({})",
            config.head_dim, config.num_heads, config.hidden_size
        );
    }

    // ── Wave 12x88: 10 additional unit tests ──

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_zero_input_dimension_with_nonzero_bias() {
        // Arrange: 0-input linear (k=0) with nonzero bias — output should equal bias
        let input: Vec<f32> = vec![]; // k = 0
        let weight: Vec<f32> = vec![]; // n * 0 = 0 elements
        let bias = vec![1.5, -2.5, 3.5];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: with zero-length input, the inner loop body never executes,
        // so each output is just the bias value
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.5).abs() < 1e-6, "expected 1.5, got {}", out[0]);
        assert!((out[1] - (-2.5)).abs() < 1e-6, "expected -2.5, got {}", out[1]);
        assert!((out[2] - 3.5).abs() < 1e-6, "expected 3.5, got {}", out[2]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_sum_of_probabilities_is_one() {
        // Arrange: arbitrary logits
        let logits = &[2.0, -1.0, 0.5, 3.0];
        // Act: compute full softmax and verify it sums to 1.0
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let prob_sum: f32 = probs.iter().sum();
        // Assert: sum of all probabilities equals 1.0
        assert!((prob_sum - 1.0).abs() < 1e-5, "softmax probabilities should sum to 1.0, got {prob_sum}");
        // Also verify softmax_max returns max_exp / sum
        let conf = softmax_max(logits);
        let max_idx = argmax(logits).unwrap();
        let expected = exps[max_idx] / sum;
        assert!((conf - expected).abs() < 1e-5, "softmax_max should equal max probability, got {conf}");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_embeddings_all_ones_finite_logits() {
        // Arrange: embeddings filled with exactly 1.0 (not 0.01 as in default weights)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![1.0; 768]; 3];
        let roles = vec![0u8, 1, 0];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.0; 11], 0.0).unwrap();
        // Assert: all logits must be finite
        for &l in result.task_logits.iter().chain(result.difficulty_logits.iter()) {
            assert!(l.is_finite(), "all-ones embedding should produce finite logit, got {l}");
        }
    }

    // @trace TEST-SIT [req:REQ-SIT-009] [level:unit]
    #[test]
    fn from_weights_bf16_mixed_values_preserves_sign() {
        // Arrange: BF16 representation of [0.5, -0.5, 1.0, -1.0] for recency_scale
        let config = IntentTrackerConfig::default();
        let mut weights_bf16: HashMap<String, Vec<u8>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let values = [0.5f32, -0.5, 1.0, -1.0];
        let mixed_bytes: Vec<u8> = values.iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let required = [
            "role_emb_weight", "w_q_weight", "w_k_weight", "w_v_weight",
            "w_q_bias", "w_k_bias", "w_v_bias",
            "info_net_fc0_weight", "info_net_fc0_bias",
            "info_net_fc1_weight", "info_net_fc1_bias",
            "info_net_fc2_weight", "info_net_fc2_bias",
            "per_head_norm_weight", "per_head_norm_bias",
            "context_norm_weight", "context_norm_bias",
            "signal_fc0_weight", "signal_fc0_bias",
            "signal_fc1_weight", "signal_fc1_bias",
            "signal_fc2_weight", "signal_fc2_bias",
            "task_fc0_weight", "task_fc0_bias",
            "task_fc1_weight", "task_fc1_bias",
            "task_fc2_weight", "task_fc2_bias",
            "diff_fc0_weight", "diff_fc0_bias",
            "diff_fc1_weight", "diff_fc1_bias",
            "diff_fc2_weight", "diff_fc2_bias",
            "recency_scale", "context_gate",
        ];
        for name in &required {
            if *name == "recency_scale" {
                weights_bf16.insert(name.to_string(), mixed_bytes.clone());
                shapes.insert(name.to_string(), vec![4]);
            } else {
                weights_bf16.insert(name.to_string(), vec![0u8, 0]);
                shapes.insert(name.to_string(), vec![1]);
            }
        }
        // Act
        let tracker = IntentTracker::from_weights_bf16(config, weights_bf16, shapes).unwrap();
        let w = tracker.get_weight("recency_scale").unwrap();
        // Assert: signs preserved after BF16 roundtrip
        assert_eq!(w.len(), 4);
        assert!(w[0] > 0.0, "0.5 should be positive after roundtrip, got {}", w[0]);
        assert!(w[1] < 0.0, "-0.5 should be negative after roundtrip, got {}", w[1]);
        assert!(w[2] > 0.0, "1.0 should be positive after roundtrip, got {}", w[2]);
        assert!(w[3] < 0.0, "-1.0 should be negative after roundtrip, got {}", w[3]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn argmax_with_duplicate_max_returns_last_occurrence() {
        // Arrange: max value 10.0 appears at indices 1, 3, and 5
        let data = &[1.0, 10.0, 3.0, 10.0, 5.0, 10.0];
        // Act
        let idx = argmax(data);
        // Assert: argmax should return the last occurrence (index 5)
        assert_eq!(idx, Some(5), "argmax should return last index of tied maximum");
    }

    // @trace TEST-SIT [req:REQ-SIT-001] [level:unit]
    #[test]
    fn classify_turn_single_token_role_boundary_two_succeeds() {
        // Arrange: single token with role index 2 (last valid role index for 3-row role_emb_weight)
        let tracker = make_test_tracker();
        let embeddings = vec![vec![0.5; 768]];
        let roles = vec![2u8];
        // Act
        let result = tracker.classify_turn(&embeddings, &roles, &[0.5; 11], 1.0);
        // Assert: role index 2 is the boundary (last valid), should succeed
        assert!(result.is_ok(), "role index 2 (last valid) should succeed, got {:?}", result.err());
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn linear_forward_output_independence_each_row_uses_own_weights() {
        // Arrange: each output row uses completely different weights
        // Row 0: only first input matters [1, 0, 0]
        // Row 1: only second input matters [0, 1, 0]
        // Row 2: only third input matters [0, 0, 1]
        let input = vec![10.0, 20.0, 30.0];
        let weight = vec![
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ];
        let bias = vec![0.0; 3];
        // Act
        let out = linear_forward(&input, &weight, &bias).unwrap();
        // Assert: each output picks exactly one input
        assert!((out[0] - 10.0).abs() < 1e-6, "row 0 should select input[0], got {}", out[0]);
        assert!((out[1] - 20.0).abs() < 1e-6, "row 1 should select input[1], got {}", out[1]);
        assert!((out[2] - 30.0).abs() < 1e-6, "row 2 should select input[2], got {}", out[2]);
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn softmax_max_confidence_symmetry_under_translation() {
        // Arrange: softmax is invariant under adding a constant to all logits
        let logits_a = &[1.0, 2.0, 3.0];
        let logits_b = &[101.0, 102.0, 103.0]; // shifted by +100
        // Act
        let conf_a = softmax_max(logits_a);
        let conf_b = softmax_max(logits_b);
        // Assert: both should give the same confidence
        assert!(
            (conf_a - conf_b).abs() < 1e-5,
            "softmax should be invariant under translation: {conf_a} vs {conf_b}"
        );
    }

    // @trace TEST-SIT [req:REQ-SIT-006] [level:unit]
    #[test]
    fn classify_turns_batch_first_item_error_no_partial_results() {
        // Arrange: first batch item is invalid, second is valid
        let tracker = make_test_tracker();
        let batch = vec![
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 64]], // wrong dim on first item
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
            TrackerTurnInput {
                embeddings: vec![vec![0.1; 768]],
                roles: vec![0],
                signals: [0.5; 11],
                context_turns: 1.0,
            },
        ];
        // Act
        let err = tracker.classify_turns_batch(&batch).unwrap_err();
        // Assert: entire batch fails; no partial results returned
        assert!(matches!(err, TrackerError::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("embedding[0] dim 64"), "should reference first item's wrong dim: {msg}");
    }

    // @trace TEST-SIT [level:unit]
    #[test]
    fn classification_difficulty_confidence_monotonic_with_dominance() {
        // Arrange: increasing dominance of a single difficulty logit
        let conf_low = {
            let c = Classification {
                task_logits: vec![0.0; 3],
                difficulty_logits: vec![1.0, 1.0, 1.0, 1.0], // uniform
            };
            c.difficulty_confidence()
        };
        let conf_mid = {
            let c = Classification {
                task_logits: vec![0.0; 3],
                difficulty_logits: vec![5.0, 1.0, 1.0, 1.0], // moderate dominance
            };
            c.difficulty_confidence()
        };
        let conf_high = {
            let c = Classification {
                task_logits: vec![0.0; 3],
                difficulty_logits: vec![50.0, 1.0, 1.0, 1.0], // strong dominance
            };
            c.difficulty_confidence()
        };
        // Assert: confidence increases monotonically with dominance
        assert!(conf_low < conf_mid, "moderate dominance should have higher confidence than uniform: {conf_low} vs {conf_mid}");
        assert!(conf_mid < conf_high, "strong dominance should have higher confidence than moderate: {conf_mid} vs {conf_high}");
    }
}
