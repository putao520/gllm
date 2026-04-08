//! TurboQuant 2.0 Runtime Integration Layer (SPEC §11)
//!
//! Coordinates the existing TurboQuant components (FWHT, KV quantization, RaBitQ correction,
//! DualTrack pool) into the unified runtime that the executor integrates.
//!
//! ## Architecture
//! ```text
//! Executor → TurboQuantRuntime → {
//!     config: KvQuantConfig,
//!     fwht: FWHT insertion at 3 boundary points
//!     kv_quant: K/V asymmetric quantization
//!     rabitq: RaBitQ unbiased correction
//!     dual_track: DualTrackMemoryPool (optional)
//! }
//! ```

use super::dual_track::{DualTrackMemoryPool, TrackConfig};
use super::quant::{
    apply_rabitq_correction, dequantize_k_per_channel, dequantize_v_per_token,
    quantize_k_per_channel_with_config,    quantize_v_per_token_with_config, KvQuantConfig, QuantMode, QuantResult, RabitqCorrection,
};

// ── TurboQuant Runtime Configuration ──

/// TurboQuant 2.0 runtime configuration (SPEC §11)
///
/// Aggregates all TurboQuant settings into a single configuration struct.
/// Passed to the executor at initialization time.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// KV quantization bit width (3 or 4)
    pub bits: u8,
    /// Attention Sink protection: first N tokens preserved in FP16
    pub sink_count: usize,
    /// Enable FWHT rotation at 3 nonlinear boundaries (§11.1)
    pub fwht_enabled: bool,
    /// Quantization mode: Deterministic or RaBitQ (§11.3)
    pub mode: QuantMode,
    /// Enable DualTrack memory pool (§11.5)
    pub dual_track_enabled: bool,
}

impl Default for TurboQuantConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            sink_count: 4,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
            dual_track_enabled: false,
        }
    }
}

impl TurboQuantConfig {
    /// Convert to legacy KvQuantConfig
    pub fn to_kv_quant_config(&self) -> KvQuantConfig {
        KvQuantConfig {
            bits: self.bits,
            sink_count: self.sink_count,
            fwht_enabled: self.fwht_enabled,
            mode: self.mode,
        }
    }

    /// Check if TurboQuant is enabled (any feature active)
    pub fn is_enabled(&self) -> bool {
        self.fwht_enabled
            || self.mode != QuantMode::Deterministic
            || self.dual_track_enabled
    }
}

// ── FWHT Insertion Points (SPEC §11.1) ──

/// FWHT insertion point identifier.
///
/// The 3 nonlinear boundaries where FWHT must be inserted:
/// 1. After Attention Epilogue (Softmax(QK^T)V output)
/// 2. After FFN Epilogue (SwiGLU(Gate)*Up output)
/// 3. Before KV Cache write (RoPE(K))
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FwhtInsertionPoint {
    /// After Softmax(QK^T)V — Attention Epilogue tail
    AttentionEpilogue,
    /// After SwiGLU(Gate)*Up — FFN Epilogue tail
    FfnEpilogue,
    /// Before KV Cache write — RoPE(K)
    KvWrite,
}

/// Apply FWHT at a given insertion point.
///
/// # Arguments
/// * `data` - Hidden state slice (must be power-of-2 length for full FWHT)
/// * `point` - Which insertion point (for logging/debugging)
///
/// # Panics
/// Panics if data.len() is not a power of 2 (debug assertion)
pub fn apply_fwht(data: &mut [f32], _point: FwhtInsertionPoint) {
    crate::kv_cache::quant::fwht_inplace(data);
}

// ── TurboQuant Runtime ──

/// TurboQuant 2.0 runtime (SPEC §11)
///
/// Manages runtime state for TurboQuant operations during inference.
/// One instance per executor, Created at model load time.
pub struct TurboQuantRuntime {
    /// Configuration
    config: TurboQuantConfig,
    /// Per-layer RaBitQ correction factors for K cache.
    /// Map: layer_index → RabitqCorrection
    k_corrections: std::collections::HashMap<usize, RabitqCorrection>,
    /// Per-layer scale buffers from RMSNorm epilogue.
    /// Map: layer_index → per-channel scales
    k_scales: std::collections::HashMap<usize, Vec<f32>>,
    /// DualTrack memory pool (optional, §11.5)
    dual_track: Option<DualTrackMemoryPool>,
}

impl TurboQuantRuntime {
    /// Create a new TurboQuant runtime with the given configuration.
    pub fn new(config: TurboQuantConfig) -> Self {
        let dual_track = if config.dual_track_enabled {
            let track_config = TrackConfig {
                quant_bits: config.bits,
                ..TrackConfig::default()
            };
            Some(DualTrackMemoryPool::new(
                track_config.main_capacity,
                track_config.xnor_capacity_bits,
                track_config.block_size,
                track_config.quant_bits,
            ).unwrap_or_else(|e| {
                log::warn!("DualTrack pool creation failed: {}, using defaults", e);
                DualTrackMemoryPool::new(1024, 1024, 64, config.bits)
                    .expect("DualTrack fallback allocation failed")
            }))
        } else {
            None
        };
        Self {
            config,
            k_corrections: std::collections::HashMap::new(),
            k_scales: std::collections::HashMap::new(),
            dual_track,
        }
    }

    /// Create a disabled TurboQuant runtime (no quantization).
    pub fn disabled() -> Self {
        Self {
            config: TurboQuantConfig::default(),
            k_corrections: std::collections::HashMap::new(),
            k_scales: std::collections::HashMap::new(),
            dual_track: None,
        }
    }

    /// Whether TurboQuant is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.is_enabled()
    }

    /// Get the quantization bit width.
    pub fn bits(&self) -> u8 {
        self.config.bits
    }

    /// Get the Attention Sink count.
    pub fn sink_count(&self) -> usize {
        self.config.sink_count
    }

    /// Whether FWHT rotation is enabled.
    pub fn fwht_enabled(&self) -> bool {
        self.config.fwht_enabled
    }

    /// Store per-channel scales from RMSNorm epilogue (§11.2).
    ///
    /// These scales are used for K cache per-channel quantization.
    /// Piggybacked from RMSNorm's output: scale = rms * sqrt(d).
    pub fn store_k_scales(&mut self, layer: usize, scales: Vec<f32>) {
        self.k_scales.insert(layer, scales);
    }

    /// Get per-channel scales for a specific layer.
    pub fn get_k_scales(&self, layer: usize) -> Option<&[f32]> {
        self.k_scales.get(&layer).map(|s| s.as_slice())
    }

    /// Quantize K cache with per-channel granularity (§11.2).
    ///
    /// # Arguments
    /// * `k` - Key tensor [seq_len, kv_dim] in row-major
    /// * `layer` - Layer index (for scale lookup)
    /// * `seq_len` - Sequence length
    /// * `kv_dim` - KV dimension (num_kv_heads * head_dim)
    ///
    /// # Returns
    /// Quantized K bytes
    pub fn quantize_k(
        &self,
        k: &[f32],
        layer: usize,
        seq_len: usize,
        kv_dim: usize,
    ) -> QuantResult<Vec<u8>> {
        let scales = self.k_scales.get(&layer).cloned().unwrap_or_else(|| {
            // Fallback: compute scales from K data itself
            let mut scales = vec![0.0f32; kv_dim];
            for t in 0..seq_len {
                for d in 0..kv_dim {
                    scales[d] = scales[d].max(k[t * kv_dim + d].abs());
                }
            }
            scales
        });

        let kv_config = self.config.to_kv_quant_config();
        let mut rng = rand::thread_rng();
        quantize_k_per_channel_with_config(k, &scales, self.config.bits, seq_len, kv_dim, &kv_config, &mut rng)
    }

    /// Quantize V cache with per-token granularity (§11.2).
    ///
    /// # Arguments
    /// * `v` - Value tensor [seq_len, kv_dim] in row-major
    /// * `seq_len` - Sequence length
    /// * `kv_dim` - KV dimension
    ///
    /// # Returns
    /// (Quantized V bytes, per-token scales)
    pub fn quantize_v(
        &self,
        v: &[f32],
        seq_len: usize,
        kv_dim: usize,
    ) -> QuantResult<(Vec<u8>, Vec<f32>)> {
        let kv_config = self.config.to_kv_quant_config();
        let mut rng = rand::thread_rng();
        quantize_v_per_token_with_config(v, self.config.bits, seq_len, kv_dim, &kv_config, &mut rng)
    }

    /// Dequantize K cache (§11.2).
    pub fn dequantize_k(
        &self,
        quantized: &[u8],
        scales: &[f32],
        seq_len: usize,
        kv_dim: usize,
    ) -> QuantResult<Vec<f32>> {
        dequantize_k_per_channel(quantized, scales, self.config.bits, seq_len, kv_dim)
    }

    /// Dequantize V cache (§11.2).
    pub fn dequantize_v(
        &self,
        quantized: &[u8],
        scales: &[f32],
        seq_len: usize,
        kv_dim: usize,
    ) -> QuantResult<Vec<f32>> {
        dequantize_v_per_token(quantized, scales, self.config.bits, seq_len, kv_dim)
    }

    /// Apply RaBitQ correction to attention scores (§11.3).
    ///
    /// # Arguments
    /// * `scores` - Raw QK^T attention scores (modified in-place)
    /// * `q_norm` - Query vector norm (‖q‖)
    /// * `layer` - Layer index (for correction lookup)
    pub fn correct_attention_scores(
        &self,
        scores: &mut [f32],
        q_norm: f32,
        layer: usize,
    ) {
        if let Some(correction) = self.k_corrections.get(&layer) {
            apply_rabitq_correction(scores, correction, q_norm);
        }
    }

    /// Store RaBitQ correction for a layer.
    pub fn store_correction(&mut self, layer: usize, correction: RabitqCorrection) {
        self.k_corrections.insert(layer, correction);
    }

    /// Get the DualTrack memory pool (§11.5).
    pub fn dual_track(&self) -> Option<&DualTrackMemoryPool> {
        self.dual_track.as_ref()
    }

    /// Get mutable reference to DualTrack pool.
    pub fn dual_track_mut(&mut self) -> Option<&mut DualTrackMemoryPool> {
        self.dual_track.as_mut()
    }

    /// Check if a token should be preserved in FP16 (Attention Sink, §11.2).
    pub fn should_preserve_fp16(&self, token_idx: usize, is_sink: bool) -> bool {
        super::quant::should_preserve_fp16(token_idx, self.config.sink_count, is_sink)
    }

    /// Reset per-layer state for a new sequence.
    pub fn reset(&mut self) {
        self.k_corrections.clear();
        self.k_scales.clear();
    }
}

// ── TurboQuant Integration Hooks for Forward Pass ──

/// Result of a TurboQuant-aware forward pass step.
#[derive(Debug, Clone)]
pub struct TurboQuantLayerOutput {
    /// Hidden state after layer computation
    pub hidden_state: Vec<f32>,
    /// Whether FWHT was applied at this layer
    pub fwht_applied: bool,
    /// Per-channel scales computed from RMSNorm (for K quantization)
    pub rms_scales: Option<Vec<f32>>,
}

impl TurboQuantLayerOutput {
    /// Create output without FWHT.
    pub fn plain(hidden: Vec<f32>) -> Self {
        Self {
            hidden_state: hidden,
            fwht_applied: false,
            rms_scales: None,
        }
    }

    /// Create output with FWHT applied and scales captured.
    pub fn with_fwht_and_scales(hidden: Vec<f32>, scales: Vec<f32>) -> Self {
        Self {
            hidden_state: hidden,
            fwht_applied: true,
            rms_scales: Some(scales),
        }
    }
}
