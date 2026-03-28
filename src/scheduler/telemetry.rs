use serde::{Deserialize, Serialize};

/// 5-Dimensional Telemetry data extracted natively from the GPU JIT Kernels (Tier III Features).
/// Transferred without Host/Device synchronization via pinned/mapped zero-copy memory.
#[derive(Debug, Clone, Default, Copy, Serialize, Deserialize, PartialEq)]
#[repr(C)]
pub struct SequenceTelemetry {
    /// L2 Delta from Attention patterns (triggers speculative admission)
    pub l2_delta: f32,
    /// Whether an outlier was detected in RmsNorm
    pub has_outlier: bool,
    /// Sparsity of SwiGLU dead rows
    pub dead_density: f32,
    /// Entropy of the probability distribution across attention heads
    pub per_head_entropy: f32,
    /// Residual ratio (RMS(x - prev) / RMS(prev)) used for Early-Exit
    pub transform_ratio: f32,
    /// Final token probability entropy from LM_Head Softmax
    pub output_entropy: f32,
}

impl SequenceTelemetry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Tier V.3 Profile-Guided Re-Fusion
/// Tracks the 5-dimensional telemetry across decoding steps.
/// If `transform_ratio < 0.05` for 95% of the last 100 steps on a given layer,
/// it triggers a RecompileHint.
#[derive(Debug, Clone)]
pub struct ProfileAccumulator {
    history: std::collections::HashMap<usize, std::collections::VecDeque<f32>>,
    required_stable_steps: usize,
    stable_threshold: f32,
    history_capacity: usize,
}

impl Default for ProfileAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileAccumulator {
    pub fn new() -> Self {
        Self {
            history: std::collections::HashMap::new(),
            required_stable_steps: 95,
            stable_threshold: 0.05,
            history_capacity: 100,
        }
    }

    /// Add a new telemetry sample for a given layer.
    /// Returns `true` if the layer has hit the stability threshold and should trigger Re-Fusion.
    pub fn record_and_check(&mut self, layer: usize, transform_ratio: f32) -> bool {
        let q = self.history.entry(layer).or_insert_with(|| std::collections::VecDeque::with_capacity(self.history_capacity));
        if q.len() >= self.history_capacity {
            q.pop_front();
        }
        q.push_back(transform_ratio);
        
        if q.len() < self.history_capacity {
            return false;
        }

        let stable_count = q.iter().filter(|&&r| r < self.stable_threshold).count();
        if stable_count >= self.required_stable_steps {
            // Clear history after triggering so we don't spam recompiles
            q.clear();
            true
        } else {
            false
        }
    }
}
