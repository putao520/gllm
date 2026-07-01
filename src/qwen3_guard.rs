//! Qwen3Guard-Stream — per-token streaming moderation classifier (B2, 方向B)
//!
//! Qwen3Guard-Stream-0.6B = Qwen3-0.6B backbone (already supported by gllm's
//! qwen3 arch) + a per-token classification head emitting 4 logit groups:
//!
//!   response path:  risk_level (3) + category (8)
//!   query path:     query_risk_level (3) + query_category (9)
//!
//! Head topology (from modeling_qwen3_guard.py + config.json):
//!   risk_level_category_pre:      Linear(1024 → 512, no bias) → RMSNorm(512)
//!     ├─ risk_level_head: Linear(512 → 3)   → risk_level_logits
//!     └─ category_head:   Linear(512 → 8)   → category_logits
//!   query_risk_level_category_pre: Linear(1024 → 512, no bias) → RMSNorm(512)
//!     ├─ query_risk_level_head: Linear(512 → 3) → query_risk_level_logits
//!     └─ query_category_head:   Linear(512 → 9) → query_category_logits
//!
//! The head runs per-token on the backbone's last hidden state. For streaming,
//! the caller feeds one token's hidden state at a time (the backbone's KV cache
//! handles incremental decode); this module is stateless across tokens.

use std::collections::HashMap;
use std::path::Path;

use half::{bf16, f16};

/// Qwen3Guard head config (subset of config.json relevant to the guard head).
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3GuardConfig {
    pub hidden_size: usize,
    pub guard_inner_size: usize,
    pub num_risk_level: usize,
    pub num_category: usize,
    pub num_query_risk_level: usize,
    pub num_query_category: usize,
    pub rms_norm_eps: f32,
}

impl Default for Qwen3GuardConfig {
    fn default() -> Self {
        // Qwen3Guard-Stream-0.6B canonical values.
        Self {
            hidden_size: 1024,
            guard_inner_size: 512,
            num_risk_level: 3,
            num_category: 8,
            num_query_risk_level: 3,
            num_query_category: 9,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Errors raised by the Qwen3Guard head.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Qwen3GuardError {
    #[error("load failed: {0}")]
    LoadFailed(String),
    #[error("missing weight: {0}")]
    MissingWeight(String),
    #[error("invalid dimension for {name}: expected {expected:?}, got {actual:?}")]
    InvalidDimension { name: String, expected: Vec<usize>, actual: Vec<usize> },
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Per-token moderation result (4 logit groups).
///
/// `risk_level` / `category` classify the response being generated;
/// `query_risk_level` / `query_category` classify the user's query.
/// Each Vec has one entry per class; apply softmax/argmax downstream.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardModerationResult {
    pub risk_level_logits: Vec<f32>,
    pub category_logits: Vec<f32>,
    pub query_risk_level_logits: Vec<f32>,
    pub query_category_logits: Vec<f32>,
}

impl GuardModerationResult {
    /// Number of total logit outputs (3+8+3+9 = 23 for the 0.6B model).
    pub fn total_logits(&self) -> usize {
        self.risk_level_logits.len()
            + self.category_logits.len()
            + self.query_risk_level_logits.len()
            + self.query_category_logits.len()
    }
}

/// Qwen3Guard per-token classification head (stateless across tokens).
///
/// Loads the 8 guard-head weights from a safetensors file (the backbone Qwen3
/// weights are loaded separately by the standard qwen3 loader). The head runs
/// on a single token's hidden state (hidden_size,) at a time.
#[derive(Debug, Clone)]
pub struct Qwen3GuardHead {
    config: Qwen3GuardConfig,
    // Response path
    risk_pre: Vec<f32>,            // (guard_inner, hidden)
    risk_norm: Vec<f32>,           // (guard_inner,)
    risk_head: Vec<f32>,           // (num_risk_level, guard_inner)
    category_head: Vec<f32>,       // (num_category, guard_inner)
    // Query path
    query_pre: Vec<f32>,           // (guard_inner, hidden)
    query_norm: Vec<f32>,          // (guard_inner,)
    query_risk_head: Vec<f32>,     // (num_query_risk_level, guard_inner)
    query_category_head: Vec<f32>, // (num_query_category, guard_inner)
}

impl Qwen3GuardHead {
    /// Load the guard head from a safetensors file containing the 8 head
    /// tensors (BF16/F16/F32 supported; ARCH-JIT-DATA-YIELDS: dtype from file).
    ///
    /// The file may be the full Qwen3Guard model.safetensors (backbone tensors
    /// are ignored) or a head-only extract.
    pub fn from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, Qwen3GuardError> {
        Self::from_safetensors_with_config(path, Qwen3GuardConfig::default())
    }

    /// Load with an explicit config.
    pub fn from_safetensors_with_config<P: AsRef<Path>>(
        path: P,
        config: Qwen3GuardConfig,
    ) -> Result<Self, Qwen3GuardError> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| Qwen3GuardError::LoadFailed(format!("read: {e}")))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| Qwen3GuardError::LoadFailed(format!("deserialize: {e}")))?;

        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| Qwen3GuardError::LoadFailed(format!("tensor {name}: {e}")))?;
            let shape: Vec<usize> = view.shape().to_vec();
            let numel: usize = shape.iter().product();
            let data = view.data();
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
                    return Err(Qwen3GuardError::LoadFailed(format!(
                        "tensor {name}: unsupported dtype {other:?}, expected F32/F16/BF16"
                    )));
                }
            };
            if float_data.len() != numel {
                return Err(Qwen3GuardError::LoadFailed(format!(
                    "tensor {name}: parsed {} elements, expected {numel}",
                    float_data.len()
                )));
            }
            weights.insert(name.to_string(), float_data);
            shapes.insert(name.to_string(), shape);
        }

        let h = config.hidden_size;
        let g = config.guard_inner_size;
        let nr = config.num_risk_level;
        let nc = config.num_category;
        let nqr = config.num_query_risk_level;
        let nqc = config.num_query_category;
        let take = |name: &str, expected: &[usize], src: &mut HashMap<String, Vec<f32>>, shapes: &HashMap<String, Vec<usize>>| -> Result<Vec<f32>, Qwen3GuardError> {
            let shape = shapes
                .get(name)
                .ok_or_else(|| Qwen3GuardError::MissingWeight(name.to_string()))?;
            if shape != expected {
                return Err(Qwen3GuardError::InvalidDimension {
                    name: name.to_string(),
                    expected: expected.to_vec(),
                    actual: shape.clone(),
                });
            }
            Ok(src.remove(name).unwrap())
        };

        let risk_pre = take("risk_level_category_pre.weight", &[g, h], &mut weights, &shapes)?;
        let risk_norm = take("risk_level_category_layernorm.weight", &[g], &mut weights, &shapes)?;
        let risk_head = take("risk_level_head.weight", &[nr, g], &mut weights, &shapes)?;
        let category_head = take("category_head.weight", &[nc, g], &mut weights, &shapes)?;
        let query_pre = take("query_risk_level_category_pre.weight", &[g, h], &mut weights, &shapes)?;
        let query_norm = take("query_risk_level_category_layernorm.weight", &[g], &mut weights, &shapes)?;
        let query_risk_head = take("query_risk_level_head.weight", &[nqr, g], &mut weights, &shapes)?;
        let query_category_head = take("query_category_head.weight", &[nqc, g], &mut weights, &shapes)?;

        Ok(Self {
            config,
            risk_pre,
            risk_norm,
            risk_head,
            category_head,
            query_pre,
            query_norm,
            query_risk_head,
            query_category_head,
        })
    }

    /// Build from explicit weights (for testing).
    pub fn from_weights(
        config: Qwen3GuardConfig,
        risk_pre: Vec<f32>,
        risk_norm: Vec<f32>,
        risk_head: Vec<f32>,
        category_head: Vec<f32>,
        query_pre: Vec<f32>,
        query_norm: Vec<f32>,
        query_risk_head: Vec<f32>,
        query_category_head: Vec<f32>,
    ) -> Result<Self, Qwen3GuardError> {
        let h = config.hidden_size;
        let g = config.guard_inner_size;
        if risk_pre.len() != g * h { return Err(Qwen3GuardError::InvalidInput(format!("risk_pre len {} != {}", risk_pre.len(), g*h))); }
        if risk_norm.len() != g { return Err(Qwen3GuardError::InvalidInput(format!("risk_norm len {} != {g}", risk_norm.len()))); }
        if risk_head.len() != config.num_risk_level * g { return Err(Qwen3GuardError::InvalidInput("risk_head".into())); }
        if category_head.len() != config.num_category * g { return Err(Qwen3GuardError::InvalidInput("category_head".into())); }
        if query_pre.len() != g * h { return Err(Qwen3GuardError::InvalidInput("query_pre".into())); }
        if query_norm.len() != g { return Err(Qwen3GuardError::InvalidInput("query_norm".into())); }
        if query_risk_head.len() != config.num_query_risk_level * g { return Err(Qwen3GuardError::InvalidInput("query_risk_head".into())); }
        if query_category_head.len() != config.num_query_category * g { return Err(Qwen3GuardError::InvalidInput("query_category_head".into())); }
        Ok(Self { config, risk_pre, risk_norm, risk_head, category_head, query_pre, query_norm, query_risk_head, query_category_head })
    }

    /// Returns the head config.
    pub fn config(&self) -> &Qwen3GuardConfig {
        &self.config
    }

    /// Classify one token's hidden state (hidden_size,) → 4 logit groups.
    ///
    /// This is the per-token streaming primitive: the caller drives the Qwen3
    /// backbone's incremental decode (KV cache) and feeds each new token's
    /// last-layer hidden state here.
    pub fn moderate_token(&self, hidden: &[f32]) -> Result<GuardModerationResult, Qwen3GuardError> {
        if hidden.len() != self.config.hidden_size {
            return Err(Qwen3GuardError::InvalidInput(format!(
                "hidden len {} != hidden_size {}",
                hidden.len(),
                self.config.hidden_size
            )));
        }
        let g = self.config.guard_inner_size;

        // Response path: pre (1024→512) → RMSNorm → {risk_head, category_head}
        let risk_x = matvec_no_bias(hidden, &self.risk_pre, g, self.config.hidden_size);
        let risk_x = rmsnorm(&risk_x, &self.risk_norm, g, self.config.rms_norm_eps);
        let risk_level_logits = matvec_no_bias(&risk_x, &self.risk_head, self.config.num_risk_level, g);
        let category_logits = matvec_no_bias(&risk_x, &self.category_head, self.config.num_category, g);

        // Query path: pre (1024→512) → RMSNorm → {query_risk_head, query_category_head}
        let query_x = matvec_no_bias(hidden, &self.query_pre, g, self.config.hidden_size);
        let query_x = rmsnorm(&query_x, &self.query_norm, g, self.config.rms_norm_eps);
        let query_risk_level_logits = matvec_no_bias(&query_x, &self.query_risk_head, self.config.num_query_risk_level, g);
        let query_category_logits = matvec_no_bias(&query_x, &self.query_category_head, self.config.num_query_category, g);

        Ok(GuardModerationResult {
            risk_level_logits,
            category_logits,
            query_risk_level_logits,
            query_category_logits,
        })
    }

    /// Classify a sequence of token hidden states (T × hidden_size) → T results.
    ///
    /// Convenience for non-streaming batch moderation.
    pub fn moderate_sequence(&self, hidden_seq: &[f32]) -> Result<Vec<GuardModerationResult>, Qwen3GuardError> {
        let h = self.config.hidden_size;
        if hidden_seq.len() % h != 0 {
            return Err(Qwen3GuardError::InvalidInput(format!(
                "hidden_seq len {} not a multiple of hidden_size {h}",
                hidden_seq.len()
            )));
        }
        let n_tokens = hidden_seq.len() / h;
        (0..n_tokens)
            .map(|t| self.moderate_token(&hidden_seq[t * h..(t + 1) * h]))
            .collect()
    }
}

// ── Numerics helpers ──

/// out = x @ W^T, W is (out_dim, in_dim) row-major (torch Linear, no bias).
fn matvec_no_bias(x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut s = 0.0f32;
        for k in 0..in_dim {
            s += x[k] * row[k];
        }
        out[o] = s;
    }
    out
}

/// RMSNorm: x / sqrt(mean(x^2) + eps) * weight (matches Qwen3RMSNorm).
fn rmsnorm(x: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut sum_sq = 0.0f32;
    for v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(v, w)| v / rms * w).collect()
}

/// Response risk level label map (config.json `response_risk_level_map`).
pub const RESPONSE_RISK_LEVELS: [&str; 3] = ["Safe", "Low Risk", "High Risk"];

/// Response category label map (config.json `response_category_map`).
pub const RESPONSE_CATEGORIES: [&str; 8] = [
    "No Violation",
    "Privacy",
    "Underage Pornography",
    "Violence",
    "Sexual Violence",
    "Self-Harm",
    "Criminal Activity",
    "Insults",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_head() -> Qwen3GuardHead {
        let config = Qwen3GuardConfig::default();
        let h = config.hidden_size;
        let g = config.guard_inner_size;
        let mk = |n: usize| (0..n).map(|i| ((i as f32) * 0.001) - 0.5).collect::<Vec<f32>>();
        Qwen3GuardHead::from_weights(
            config.clone(),
            mk(g * h), mk(g), mk(config.num_risk_level * g), mk(config.num_category * g),
            mk(g * h), mk(g), mk(config.num_query_risk_level * g), mk(config.num_query_category * g),
        )
        .expect("valid head weights")
    }

    #[test]
    fn moderate_token_output_dims() {
        let head = dummy_head();
        let hidden = vec![0.1; 1024];
        let r = head.moderate_token(&hidden).expect("moderate ok");
        assert_eq!(r.risk_level_logits.len(), 3);
        assert_eq!(r.category_logits.len(), 8);
        assert_eq!(r.query_risk_level_logits.len(), 3);
        assert_eq!(r.query_category_logits.len(), 9);
        assert_eq!(r.total_logits(), 23);
        for v in r.risk_level_logits.iter().chain(&r.category_logits)
            .chain(&r.query_risk_level_logits).chain(&r.query_category_logits) {
            assert!(v.is_finite(), "non-finite logit");
        }
    }

    #[test]
    fn moderate_token_rejects_bad_dim() {
        let head = dummy_head();
        let hidden = vec![0.1; 100];
        assert!(head.moderate_token(&hidden).is_err());
    }

    #[test]
    fn moderate_sequence_multi_token() {
        let head = dummy_head();
        let seq: Vec<f32> = (0..(3 * 1024)).map(|i| ((i as f32) % 10.0) * 0.01).collect();
        let results = head.moderate_sequence(&seq).expect("seq ok");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].total_logits(), 23);
        assert_ne!(results[0].risk_level_logits, results[1].risk_level_logits);
    }

    #[test]
    fn from_weights_rejects_bad_dim() {
        let config = Qwen3GuardConfig::default();
        let bad = vec![0.0; 10];
        let r = Qwen3GuardHead::from_weights(
            config, bad.clone(), bad.clone(), bad.clone(), bad.clone(),
            bad.clone(), bad.clone(), bad.clone(), bad,
        );
        assert!(r.is_err());
    }

    #[test]
    fn config_default_matches_06b() {
        let c = Qwen3GuardConfig::default();
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.guard_inner_size, 512);
        assert_eq!(c.num_risk_level, 3);
        assert_eq!(c.num_category, 8);
        assert_eq!(c.num_query_risk_level, 3);
        assert_eq!(c.num_query_category, 9);
    }
}
