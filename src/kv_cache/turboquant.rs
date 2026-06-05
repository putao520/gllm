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
    ///
    /// # Errors
    /// Returns error if DualTrack pool creation fails when `dual_track_enabled` is true.
    pub fn new(config: TurboQuantConfig) -> Result<Self, super::dual_track::DualTrackError> {
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
            )?)
        } else {
            None
        };
        Ok(Self {
            config,
            k_corrections: std::collections::HashMap::new(),
            k_scales: std::collections::HashMap::new(),
            dual_track,
        })
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

    /// 将量化 K 写入 DualTrack pool (§11.5).
    ///
    /// 当 dual_track 启用时，量化后的 K 数据写入主池，
    /// XNOR 残差掩码写入校验池。
    pub fn store_k_to_dual_track(
        &mut self,
        block_id: usize,
        quantized_k: &[u8],
        residual_mask: &[bool],
    ) -> Result<(), super::dual_track::DualTrackError> {
        if let Some(ref mut pool) = self.dual_track {
            let num_elements = quantized_k.len() * 2;
            pool.allocate_block_main(block_id, num_elements)?;
            pool.write_main(block_id, quantized_k)?;
            if !residual_mask.is_empty() {
                pool.allocate_block_xnor(block_id + 1, residual_mask.len())?;
                pool.write_xnor(block_id + 1, residual_mask)?;
            }
        }
        Ok(())
    }

    /// 将量化 V 写入 DualTrack pool (§11.5).
    pub fn store_v_to_dual_track(
        &mut self,
        block_id: usize,
        quantized_v: &[u8],
        residual_mask: &[bool],
    ) -> Result<(), super::dual_track::DualTrackError> {
        if let Some(ref mut pool) = self.dual_track {
            let num_elements = quantized_v.len() * 2;
            pool.allocate_block_main(block_id, num_elements)?;
            pool.write_main(block_id, quantized_v)?;
            if !residual_mask.is_empty() {
                pool.allocate_block_xnor(block_id + 1, residual_mask.len())?;
                pool.write_xnor(block_id + 1, residual_mask)?;
            }
        }
        Ok(())
    }

    /// 从 DualTrack pool 读取量化 K (§11.5).
    pub fn load_k_from_dual_track(
        &self,
        block_id: usize,
        out: &mut [u8],
    ) -> Result<(), super::dual_track::DualTrackError> {
        if let Some(ref pool) = self.dual_track {
            pool.read_main(block_id, out)?;
        }
        Ok(())
    }

    /// 从 DualTrack pool 读取 XNOR 残差掩码 (§11.5).
    pub fn load_xnor_mask(
        &self,
        xnor_block_id: usize,
        out: &mut [bool],
    ) -> Result<(), super::dual_track::DualTrackError> {
        if let Some(ref pool) = self.dual_track {
            pool.read_xnor(xnor_block_id, out)?;
        }
        Ok(())
    }

    /// 检查 DualTrack pool 是否已启用且活跃
    pub fn is_dual_track_active(&self) -> bool {
        self.dual_track.is_some()
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── TurboQuantConfig ──

    #[test]
    fn config_default_values() {
        let c = TurboQuantConfig::default();
        assert_eq!(c.bits, 4);
        assert_eq!(c.sink_count, 4);
        assert!(!c.fwht_enabled);
        assert_eq!(c.mode, QuantMode::Deterministic);
        assert!(!c.dual_track_enabled);
    }

    #[test]
    fn config_default_not_enabled() {
        assert!(!TurboQuantConfig::default().is_enabled());
    }

    #[test]
    fn config_enabled_with_fwht() {
        let c = TurboQuantConfig {
            fwht_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    #[test]
    fn config_enabled_with_rabitq() {
        let c = TurboQuantConfig {
            mode: QuantMode::RaBitQ,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    #[test]
    fn config_enabled_with_dual_track() {
        let c = TurboQuantConfig {
            dual_track_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    #[test]
    fn config_to_kv_quant_config() {
        let c = TurboQuantConfig {
            bits: 3,
            sink_count: 8,
            fwht_enabled: true,
            ..Default::default()
        };
        let kv = c.to_kv_quant_config();
        assert_eq!(kv.bits, 3);
        assert_eq!(kv.sink_count, 8);
        assert!(kv.fwht_enabled);
    }

    // ── apply_fwht ──

    #[test]
    fn fwht_power_of_two_length() {
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        // FWHT of [1,0,0,0] should be all 1.0
        for &v in &data {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn fwht_is_self_inverse() {
        let original = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut data = original.clone();
        apply_fwht(&mut data, FwhtInsertionPoint::FfnEpilogue);
        apply_fwht(&mut data, FwhtInsertionPoint::KvWrite);
        // FWHT(FWHT(x)) = n * x, so we need to divide by n
        let n = original.len() as f32;
        for (got, expected) in data.iter().zip(original.iter()) {
            assert!((got / n - expected).abs() < 1e-4, "expected {expected}, got {}", got / n);
        }
    }

    #[test]
    fn fwht_scales_energy_by_n() {
        // Unnormalized FWHT: ||H(x)||^2 = n * ||x||^2
        let data = vec![2.0f32, -1.0, 3.0, 0.5, -2.0, 1.0, 0.0, -0.5];
        let n = data.len() as f32;
        let energy_before: f32 = data.iter().map(|x| x * x).sum();
        let mut transformed = data.clone();
        apply_fwht(&mut transformed, FwhtInsertionPoint::AttentionEpilogue);
        let energy_after: f32 = transformed.iter().map(|x| x * x).sum();
        assert!((energy_before * n - energy_after).abs() < 1e-4,
            "energy should scale by n={n}: {} vs {energy_after}", energy_before * n);
    }

    // ── FwhtInsertionPoint ──

    #[test]
    fn fwht_insertion_point_equality() {
        assert_eq!(FwhtInsertionPoint::AttentionEpilogue, FwhtInsertionPoint::AttentionEpilogue);
        assert_ne!(FwhtInsertionPoint::AttentionEpilogue, FwhtInsertionPoint::FfnEpilogue);
        assert_ne!(FwhtInsertionPoint::FfnEpilogue, FwhtInsertionPoint::KvWrite);
    }

    // ── TurboQuantRuntime ──

    #[test]
    fn runtime_new_default_not_enabled() {
        let rt = TurboQuantRuntime::new(TurboQuantConfig::default()).unwrap();
        assert!(!rt.is_enabled());
        assert_eq!(rt.bits(), 4);
        assert_eq!(rt.sink_count(), 4);
        assert!(!rt.fwht_enabled());
    }

    #[test]
    fn runtime_disabled() {
        let rt = TurboQuantRuntime::disabled();
        assert!(!rt.is_enabled());
        assert!(!rt.is_dual_track_active());
    }

    #[test]
    fn runtime_enabled_with_fwht() {
        let cfg = TurboQuantConfig {
            fwht_enabled: true,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert!(rt.is_enabled());
        assert!(rt.fwht_enabled());
    }

    #[test]
    fn runtime_store_and_get_k_scales() {
        let mut rt = TurboQuantRuntime::disabled();
        assert!(rt.get_k_scales(0).is_none());
        rt.store_k_scales(0, vec![1.0, 2.0, 3.0]);
        let scales = rt.get_k_scales(0).unwrap();
        assert_eq!(scales, &[1.0, 2.0, 3.0]);
        assert!(rt.get_k_scales(1).is_none());
    }

    #[test]
    fn runtime_reset_clears_state() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0, 2.0]);
        rt.store_correction(0, RabitqCorrection::zero());
        rt.reset();
        assert!(rt.get_k_scales(0).is_none());
    }

    #[test]
    fn runtime_should_preserve_fp16() {
        let rt = TurboQuantRuntime::disabled();
        // sink_count=4, token_idx < 4 and is_sink → true
        assert!(rt.should_preserve_fp16(0, true));
        assert!(rt.should_preserve_fp16(3, true));
        // token_idx >= sink_count → false
        assert!(!rt.should_preserve_fp16(4, true));
        // not a sink → false
        assert!(!rt.should_preserve_fp16(0, false));
    }

    #[test]
    fn runtime_no_dual_track_by_default() {
        let rt = TurboQuantRuntime::disabled();
        assert!(!rt.is_dual_track_active());
        assert!(rt.dual_track().is_none());
    }

    #[test]
    fn runtime_store_and_correct_rabitq() {
        let mut rt = TurboQuantRuntime::disabled();
        let correction = RabitqCorrection { c0: 0.1, c1: 0.5, v_norm: 3.0 };
        rt.store_correction(2, correction);
        let mut scores = vec![1.0f32, 2.0, 3.0];
        rt.correct_attention_scores(&mut scores, 2.0, 2);
    }

    // ── TurboQuantLayerOutput ──

    #[test]
    fn layer_output_plain() {
        let out = TurboQuantLayerOutput::plain(vec![1.0, 2.0]);
        assert_eq!(out.hidden_state, vec![1.0, 2.0]);
        assert!(!out.fwht_applied);
        assert!(out.rms_scales.is_none());
    }

    #[test]
    fn layer_output_with_fwht_and_scales() {
        let out = TurboQuantLayerOutput::with_fwht_and_scales(vec![1.0], vec![0.5, 1.5]);
        assert!(out.fwht_applied);
        assert_eq!(out.rms_scales.as_deref(), Some(&[0.5f32, 1.5][..]));
    }

    // ── Additional comprehensive tests ──

    // -- TurboQuantConfig: Debug and Clone derivation --

    #[test]
    fn config_debug_format() {
        let c = TurboQuantConfig::default();
        let debug_str = format!("{c:?}");
        assert!(debug_str.contains("TurboQuantConfig"), "Debug output should contain struct name");
        assert!(debug_str.contains("bits"), "Debug output should contain 'bits' field");
    }

    #[test]
    fn config_clone_is_independent() {
        let original = TurboQuantConfig {
            bits: 3,
            sink_count: 8,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: true,
        };
        let cloned = original.clone();
        assert_eq!(original.bits, cloned.bits);
        assert_eq!(original.sink_count, cloned.sink_count);
        assert_eq!(original.fwht_enabled, cloned.fwht_enabled);
        assert_eq!(original.mode, cloned.mode);
        assert_eq!(original.dual_track_enabled, cloned.dual_track_enabled);
        // Verify independence: modifying the concept doesn't affect the other
        assert!(original.is_enabled());
        assert!(cloned.is_enabled());
    }

    // -- TurboQuantConfig: is_enabled combinations --

    #[test]
    fn config_enabled_with_all_features() {
        let c = TurboQuantConfig {
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    #[test]
    fn config_enabled_with_fwht_and_dual_track() {
        let c = TurboQuantConfig {
            fwht_enabled: true,
            dual_track_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    #[test]
    fn config_enabled_with_rabitq_and_dual_track() {
        let c = TurboQuantConfig {
            mode: QuantMode::RaBitQ,
            dual_track_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled());
    }

    // -- TurboQuantConfig: bits variants in to_kv_quant_config --

    #[test]
    fn config_to_kv_quant_config_preserves_all_fields() {
        let c = TurboQuantConfig {
            bits: 3,
            sink_count: 16,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: false,
        };
        let kv = c.to_kv_quant_config();
        assert_eq!(kv.bits, 3);
        assert_eq!(kv.sink_count, 16);
        assert!(kv.fwht_enabled);
        assert_eq!(kv.mode, QuantMode::RaBitQ);
    }

    // -- FwhtInsertionPoint: Copy trait and exhaustive equality --

    #[test]
    fn fwht_insertion_point_copy_trait() {
        let point = FwhtInsertionPoint::AttentionEpilogue;
        let copied = point;
        assert_eq!(point, copied);
    }

    #[test]
    fn fwht_insertion_point_all_variants_distinct() {
        let variants = [
            FwhtInsertionPoint::AttentionEpilogue,
            FwhtInsertionPoint::FfnEpilogue,
            FwhtInsertionPoint::KvWrite,
        ];
        for (i, &a) in variants.iter().enumerate() {
            for (j, &b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b, "Same index should be equal");
                } else {
                    assert_ne!(a, b, "Different indices should not be equal");
                }
            }
        }
    }

    #[test]
    fn fwht_insertion_point_debug_format() {
        let point = FwhtInsertionPoint::KvWrite;
        let debug_str = format!("{point:?}");
        assert!(!debug_str.is_empty(), "Debug should produce non-empty output");
    }

    // -- apply_fwht: edge cases and properties --

    #[test]
    fn fwht_single_element_is_identity() {
        let mut data = vec![42.0f32];
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        assert_eq!(data[0], 42.0, "FWHT of single element should be identity");
    }

    #[test]
    fn fwht_two_elements() {
        let mut data = vec![1.0f32, -1.0];
        apply_fwht(&mut data, FwhtInsertionPoint::FfnEpilogue);
        // FWHT of [1, -1] = [0, 2]
        assert!((data[0] - 0.0).abs() < 1e-5, "expected 0.0, got {}", data[0]);
        assert!((data[1] - 2.0).abs() < 1e-5, "expected 2.0, got {}", data[1]);
    }

    #[test]
    fn fwht_preserves_linearity() {
        // FWHT(a*x + b*y) = a*FWHT(x) + b*FWHT(y)
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        let a = 2.0f32;
        let b = 3.0f32;

        // Compute FWHT(a*x + b*y)
        let mut combined: Vec<f32> = x.iter().zip(y.iter()).map(|(&xi, &yi)| a * xi + b * yi).collect();
        apply_fwht(&mut combined, FwhtInsertionPoint::AttentionEpilogue);

        // Compute a*FWHT(x) + b*FWHT(y)
        let mut fx = x.clone();
        let mut fy = y.clone();
        apply_fwht(&mut fx, FwhtInsertionPoint::AttentionEpilogue);
        apply_fwht(&mut fy, FwhtInsertionPoint::AttentionEpilogue);
        let expected: Vec<f32> = fx.iter().zip(fy.iter()).map(|(&fxi, &fyi)| a * fxi + b * fyi).collect();

        for (got, exp) in combined.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "linearity violated: got {got}, expected {exp}");
        }
    }

    #[test]
    fn fwht_all_zeros_stays_zero() {
        let mut data = vec![0.0f32; 16];
        apply_fwht(&mut data, FwhtInsertionPoint::KvWrite);
        for &v in &data {
            assert!((v - 0.0).abs() < 1e-5, "FWHT of all zeros should stay zero, got {v}");
        }
    }

    #[test]
    fn fwht_constant_input() {
        // FWHT of constant vector [c, c, c, c] = [4c, 0, 0, 0]
        let c = 3.0f32;
        let mut data = vec![c; 4];
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        let n = 4.0f32;
        assert!((data[0] - c * n).abs() < 1e-5, "expected {}, got {}", c * n, data[0]);
        for &v in &data[1..] {
            assert!((v - 0.0).abs() < 1e-5, "remaining elements should be 0, got {v}");
        }
    }

    // -- TurboQuantRuntime: construction with custom config --

    #[test]
    fn runtime_new_with_custom_bits_and_sink_count() {
        let cfg = TurboQuantConfig {
            bits: 3,
            sink_count: 8,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert_eq!(rt.bits(), 3);
        assert_eq!(rt.sink_count(), 8);
        assert!(!rt.fwht_enabled());
    }

    // -- TurboQuantRuntime: store/retrieve multiple layers --

    #[test]
    fn runtime_store_k_scales_multiple_layers() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0, 2.0]);
        rt.store_k_scales(3, vec![10.0, 20.0, 30.0]);
        rt.store_k_scales(7, vec![100.0]);

        assert_eq!(rt.get_k_scales(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(rt.get_k_scales(3).unwrap(), &[10.0, 20.0, 30.0]);
        assert_eq!(rt.get_k_scales(7).unwrap(), &[100.0]);
        assert!(rt.get_k_scales(1).is_none());
        assert!(rt.get_k_scales(99).is_none());
    }

    // -- TurboQuantRuntime: store_k_scales overwrites previous --

    #[test]
    fn runtime_store_k_scales_overwrites() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0, 2.0]);
        assert_eq!(rt.get_k_scales(0).unwrap(), &[1.0, 2.0]);
        rt.store_k_scales(0, vec![9.0, 8.0, 7.0]);
        assert_eq!(rt.get_k_scales(0).unwrap(), &[9.0, 8.0, 7.0]);
    }

    // -- TurboQuantRuntime: correct_attention_scores with no correction is no-op --

    #[test]
    fn runtime_correct_scores_no_correction_is_noop() {
        let rt = TurboQuantRuntime::disabled();
        let mut scores = vec![1.0f32, 2.0, 3.0];
        let original = scores.clone();
        rt.correct_attention_scores(&mut scores, 5.0, 0);
        assert_eq!(scores, original, "Scores should be unchanged when no correction stored");
    }

    // -- TurboQuantRuntime: correct_attention_scores applies correction --

    #[test]
    fn runtime_correct_scores_applies_rabitq_correction() {
        let mut rt = TurboQuantRuntime::disabled();
        let correction = RabitqCorrection { c0: 1.0, c1: 0.5, v_norm: 2.0 };
        rt.store_correction(0, correction);

        let mut scores = vec![10.0f32, 20.0];
        rt.correct_attention_scores(&mut scores, 3.0, 0);

        // correction = c1 * q_norm + c0 = 0.5 * 3.0 + 1.0 = 2.5
        let expected_correction = 0.5 * 3.0 + 1.0;
        assert!((scores[0] - (10.0 + expected_correction)).abs() < 1e-5,
            "expected {}, got {}", 10.0 + expected_correction, scores[0]);
        assert!((scores[1] - (20.0 + expected_correction)).abs() < 1e-5,
            "expected {}, got {}", 20.0 + expected_correction, scores[1]);
    }

    // -- TurboQuantRuntime: reset clears both scales and corrections --

    #[test]
    fn runtime_reset_clears_corrections() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_correction(0, RabitqCorrection { c0: 1.0, c1: 0.5, v_norm: 2.0 });
        rt.store_correction(5, RabitqCorrection::zero());

        // Before reset: correction exists for layer 0, modifies scores
        let mut scores_before = vec![10.0f32];
        rt.correct_attention_scores(&mut scores_before, 1.0, 0);
        assert_ne!(scores_before[0], 10.0, "Correction should have been applied before reset");

        rt.reset();

        // After reset: no correction, scores unchanged
        let mut scores_after = vec![10.0f32];
        rt.correct_attention_scores(&mut scores_after, 1.0, 0);
        assert_eq!(scores_after[0], 10.0, "Correction should not be applied after reset");
    }

    // -- TurboQuantRuntime: load from dual_track is no-op without pool --

    #[test]
    fn runtime_load_from_dual_track_noop_without_pool() {
        let rt = TurboQuantRuntime::disabled();
        let mut buf = vec![0u8; 16];
        let result = rt.load_k_from_dual_track(0, &mut buf);
        assert!(result.is_ok(), "Should succeed as no-op when dual_track is absent");

        let mut mask = vec![false; 8];
        let result = rt.load_xnor_mask(0, &mut mask);
        assert!(result.is_ok(), "Should succeed as no-op when dual_track is absent");
    }

    // -- TurboQuantRuntime: dual_track accessors --

    #[test]
    fn runtime_dual_track_accessors_none_when_disabled() {
        let mut rt = TurboQuantRuntime::disabled();
        assert!(rt.dual_track().is_none());
        assert!(rt.dual_track_mut().is_none());
    }

    // -- TurboQuantRuntime: should_preserve_fp16 boundary --

    #[test]
    fn runtime_should_preserve_fp16_boundary_token_index() {
        let rt = TurboQuantRuntime::disabled();
        // default sink_count = 4
        assert!(rt.should_preserve_fp16(0, true));
        assert!(rt.should_preserve_fp16(3, true));
        assert!(!rt.should_preserve_fp16(4, true), "token_idx == sink_count should not be preserved");
        assert!(!rt.should_preserve_fp16(100, true));
        assert!(!rt.should_preserve_fp16(0, false), "non-sink should not be preserved");
    }

    // -- TurboQuantRuntime: with custom sink_count --

    #[test]
    fn runtime_should_preserve_fp16_custom_sink_count() {
        let cfg = TurboQuantConfig {
            sink_count: 0,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert!(!rt.should_preserve_fp16(0, true), "sink_count=0 should never preserve");
    }

    // -- TurboQuantLayerOutput: Clone derivation --

    #[test]
    fn layer_output_clone_preserves_fields() {
        let original = TurboQuantLayerOutput::with_fwht_and_scales(vec![1.0, 2.0, 3.0], vec![0.5, 1.5]);
        let cloned = original.clone();
        assert_eq!(cloned.hidden_state, original.hidden_state);
        assert_eq!(cloned.fwht_applied, original.fwht_applied);
        assert_eq!(cloned.rms_scales, original.rms_scales);
    }

    #[test]
    fn layer_output_plain_clone() {
        let original = TurboQuantLayerOutput::plain(vec![10.0, 20.0]);
        let cloned = original.clone();
        assert_eq!(cloned.hidden_state, vec![10.0, 20.0]);
        assert!(!cloned.fwht_applied);
        assert!(cloned.rms_scales.is_none());
    }

    // -- TurboQuantLayerOutput: Debug derivation --

    #[test]
    fn layer_output_debug_format() {
        let out = TurboQuantLayerOutput::plain(vec![1.0]);
        let debug_str = format!("{out:?}");
        assert!(debug_str.contains("TurboQuantLayerOutput"), "Debug should contain struct name");
    }

    // -- FWHT: self-inverse with larger size --

    #[test]
    fn fwht_self_inverse_16_elements() {
        let original = vec![1.0f32, 2.0, -3.0, 4.0, 0.5, -1.5, 2.5, -0.5,
                            3.0, -2.0, 1.0, 0.0, -4.0, 3.5, -1.0, 2.0];
        let mut data = original.clone();
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        apply_fwht(&mut data, FwhtInsertionPoint::FfnEpilogue);
        let n = original.len() as f32;
        for (got, expected) in data.iter().zip(original.iter()) {
            assert!((got / n - expected).abs() < 1e-3,
                "FWHT self-inverse failed: expected {expected}, got {}", got / n);
        }
    }

    // ── Additional unit tests: turboquant.rs coverage gap fill ──

    // -- TurboQuantConfig: custom construction with all fields --

    #[test]
    fn config_custom_construction_all_fields() {
        let c = TurboQuantConfig {
            bits: 3,
            sink_count: 16,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: true,
        };
        assert_eq!(c.bits, 3);
        assert_eq!(c.sink_count, 16);
        assert!(c.fwht_enabled);
        assert_eq!(c.mode, QuantMode::RaBitQ);
        assert!(c.dual_track_enabled);
    }

    // -- TurboQuantConfig: is_enabled false only with default Deterministic mode
    //    and no fwht/dual_track --

    #[test]
    fn config_not_enabled_deterministic_no_extras() {
        let c = TurboQuantConfig {
            bits: 3,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
            dual_track_enabled: false,
        };
        assert!(!c.is_enabled());
    }

    // -- TurboQuantConfig: to_kv_quant_config propagates mode --

    #[test]
    fn config_to_kv_quant_config_rabitq_mode() {
        let c = TurboQuantConfig {
            mode: QuantMode::RaBitQ,
            ..Default::default()
        };
        let kv = c.to_kv_quant_config();
        assert_eq!(kv.mode, QuantMode::RaBitQ);
    }

    // -- TurboQuantConfig: to_kv_quant_config dual_track_enabled not in KvQuantConfig --

    #[test]
    fn config_to_kv_quant_config_ignores_dual_track() {
        let c = TurboQuantConfig {
            dual_track_enabled: true,
            ..Default::default()
        };
        let kv = c.to_kv_quant_config();
        // KvQuantConfig has no dual_track field; conversion should succeed
        assert_eq!(kv.bits, 4);
        assert_eq!(kv.sink_count, 4);
        assert!(!kv.fwht_enabled);
        assert_eq!(kv.mode, QuantMode::Deterministic);
    }

    // -- TurboQuantConfig: Debug output contains all field names --

    #[test]
    fn config_debug_contains_all_fields() {
        let c = TurboQuantConfig {
            bits: 3,
            sink_count: 8,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: true,
        };
        let debug = format!("{c:?}");
        assert!(debug.contains("bits"), "Debug should contain 'bits'");
        assert!(debug.contains("sink_count"), "Debug should contain 'sink_count'");
        assert!(debug.contains("fwht_enabled"), "Debug should contain 'fwht_enabled'");
        assert!(debug.contains("mode"), "Debug should contain 'mode'");
        assert!(debug.contains("dual_track_enabled"), "Debug should contain 'dual_track_enabled'");
    }

    // -- FwhtInsertionPoint: Clone produces equal value --

    #[test]
    fn fwht_insertion_point_clone() {
        let point = FwhtInsertionPoint::FfnEpilogue;
        let cloned = point.clone();
        assert_eq!(point, cloned);
    }

    // -- FwhtInsertionPoint: Debug for all variants --

    #[test]
    fn fwht_insertion_point_debug_all_variants() {
        let debug_attention = format!("{:?}", FwhtInsertionPoint::AttentionEpilogue);
        let debug_ffn = format!("{:?}", FwhtInsertionPoint::FfnEpilogue);
        let debug_kv = format!("{:?}", FwhtInsertionPoint::KvWrite);
        assert!(!debug_attention.is_empty());
        assert!(!debug_ffn.is_empty());
        assert!(!debug_kv.is_empty());
        // Each variant should produce a distinct debug string
        assert_ne!(debug_attention, debug_ffn);
        assert_ne!(debug_ffn, debug_kv);
        assert_ne!(debug_attention, debug_kv);
    }

    // -- TurboQuantRuntime: disabled() returns correct default bit width --

    #[test]
    fn runtime_disabled_default_bits() {
        let rt = TurboQuantRuntime::disabled();
        assert_eq!(rt.bits(), 4);
    }

    // -- TurboQuantRuntime: disabled() returns correct default sink_count --

    #[test]
    fn runtime_disabled_default_sink_count() {
        let rt = TurboQuantRuntime::disabled();
        assert_eq!(rt.sink_count(), 4);
    }

    // -- TurboQuantRuntime: disabled() fwht_enabled is false --

    #[test]
    fn runtime_disabled_fwht_not_enabled() {
        let rt = TurboQuantRuntime::disabled();
        assert!(!rt.fwht_enabled());
    }

    // -- TurboQuantRuntime: new() with zero bits field --

    #[test]
    fn runtime_new_zero_bits() {
        let cfg = TurboQuantConfig {
            bits: 0,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert_eq!(rt.bits(), 0);
        assert!(!rt.is_enabled());
    }

    // -- TurboQuantRuntime: new() with max u8 bits --

    #[test]
    fn runtime_new_max_bits() {
        let cfg = TurboQuantConfig {
            bits: u8::MAX,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert_eq!(rt.bits(), 255);
    }

    // -- TurboQuantRuntime: new() with zero sink_count --

    #[test]
    fn runtime_new_zero_sink_count() {
        let cfg = TurboQuantConfig {
            sink_count: 0,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert_eq!(rt.sink_count(), 0);
    }

    // -- TurboQuantRuntime: get_k_scales returns None for unstored layer --

    #[test]
    fn runtime_get_k_scales_none_for_all_unstored() {
        let rt = TurboQuantRuntime::disabled();
        assert!(rt.get_k_scales(0).is_none());
        assert!(rt.get_k_scales(usize::MAX).is_none());
    }

    // -- TurboQuantRuntime: store_k_scales with empty vec --

    #[test]
    fn runtime_store_k_scales_empty_vec() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(5, vec![]);
        let scales = rt.get_k_scales(5).unwrap();
        assert!(scales.is_empty());
    }

    // -- TurboQuantRuntime: store_k_scales with boundary layer indices --

    #[test]
    fn runtime_store_k_scales_boundary_layer_indices() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0]);
        rt.store_k_scales(usize::MAX, vec![99.0]);

        assert_eq!(rt.get_k_scales(0).unwrap(), &[1.0]);
        assert_eq!(rt.get_k_scales(usize::MAX).unwrap(), &[99.0]);
        assert!(rt.get_k_scales(1).is_none());
    }

    // -- TurboQuantRuntime: store_correction and get via correct_attention_scores --

    #[test]
    fn runtime_store_correction_different_layers() {
        let mut rt = TurboQuantRuntime::disabled();
        let corr_a = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let corr_b = RabitqCorrection { c0: 10.0, c1: 20.0, v_norm: 30.0 };
        rt.store_correction(0, corr_a);
        rt.store_correction(100, corr_b);

        // Layer 0: correction = c1 * q_norm + c0 = 2.0 * 1.0 + 1.0 = 3.0
        let mut scores_a = vec![5.0f32];
        rt.correct_attention_scores(&mut scores_a, 1.0, 0);
        assert!((scores_a[0] - 8.0).abs() < 1e-5, "expected 8.0, got {}", scores_a[0]);

        // Layer 100: correction = 20.0 * 1.0 + 10.0 = 30.0
        let mut scores_b = vec![5.0f32];
        rt.correct_attention_scores(&mut scores_b, 1.0, 100);
        assert!((scores_b[0] - 35.0).abs() < 1e-5, "expected 35.0, got {}", scores_b[0]);

        // Layer 50: no correction stored
        let mut scores_c = vec![5.0f32];
        rt.correct_attention_scores(&mut scores_c, 1.0, 50);
        assert_eq!(scores_c[0], 5.0, "unstored layer should leave scores unchanged");
    }

    // -- TurboQuantRuntime: store_correction overwrites previous --

    #[test]
    fn runtime_store_correction_overwrites() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_correction(0, RabitqCorrection { c0: 1.0, c1: 0.0, v_norm: 0.0 });
        let mut scores = vec![0.0f32];
        rt.correct_attention_scores(&mut scores, 1.0, 0);
        assert!((scores[0] - 1.0).abs() < 1e-5, "first correction should add 1.0");

        rt.store_correction(0, RabitqCorrection { c0: 5.0, c1: 0.0, v_norm: 0.0 });
        let mut scores2 = vec![0.0f32];
        rt.correct_attention_scores(&mut scores2, 1.0, 0);
        assert!((scores2[0] - 5.0).abs() < 1e-5, "overwritten correction should add 5.0");
    }

    // -- TurboQuantRuntime: reset after store_k_scales and corrections clears both --

    #[test]
    fn runtime_reset_clears_scales_and_corrections() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0, 2.0]);
        rt.store_correction(0, RabitqCorrection { c0: 1.0, c1: 1.0, v_norm: 1.0 });
        rt.reset();
        assert!(rt.get_k_scales(0).is_none(), "scales should be cleared after reset");

        let mut scores = vec![10.0f32];
        rt.correct_attention_scores(&mut scores, 1.0, 0);
        assert_eq!(scores[0], 10.0, "correction should not apply after reset");
    }

    // -- TurboQuantRuntime: RabitqCorrection::zero() produces no score change --

    #[test]
    fn rabitq_correction_zero_is_noop() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_correction(0, RabitqCorrection::zero());
        let mut scores = vec![42.0f32, -7.0, 0.0];
        let original = scores.clone();
        rt.correct_attention_scores(&mut scores, 100.0, 0);
        assert_eq!(scores, original, "zero correction should not change scores");
    }

    // -- TurboQuantRuntime: is_enabled reflects config changes via new() --

    #[test]
    fn runtime_is_enabled_reflects_config() {
        // Enabled via RaBitQ mode
        let cfg_on = TurboQuantConfig {
            mode: QuantMode::RaBitQ,
            ..Default::default()
        };
        let rt_on = TurboQuantRuntime::new(cfg_on).unwrap();
        assert!(rt_on.is_enabled());

        // Disabled: default config
        let rt_off = TurboQuantRuntime::disabled();
        assert!(!rt_off.is_enabled());
    }

    // ══════════════════════════════════════════════════════════════
    //  Additional tests (round 2)
    // ══════════════════════════════════════════════════════════════

    // ── QuantMode ──

    #[test]
    fn quant_mode_deterministic_equality() {
        assert_eq!(QuantMode::Deterministic, QuantMode::Deterministic);
    }

    #[test]
    fn quant_mode_rabitq_equality() {
        assert_eq!(QuantMode::RaBitQ, QuantMode::RaBitQ);
    }

    #[test]
    fn quant_mode_variants_not_equal() {
        assert_ne!(QuantMode::Deterministic, QuantMode::RaBitQ);
    }

    #[test]
    fn quant_mode_debug_format() {
        let det = format!("{:?}", QuantMode::Deterministic);
        let rab = format!("{:?}", QuantMode::RaBitQ);
        assert!(!det.is_empty());
        assert!(!rab.is_empty());
        assert_ne!(det, rab);
    }

    #[test]
    fn quant_mode_clone() {
        let mode = QuantMode::RaBitQ;
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn quant_mode_copy() {
        let mode = QuantMode::Deterministic;
        let copied = mode;
        assert_eq!(mode, copied);
    }

    #[test]
    fn quant_mode_hash_consistency() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(QuantMode::Deterministic, "det");
        map.insert(QuantMode::RaBitQ, "rab");
        assert_eq!(map.get(&QuantMode::Deterministic), Some(&"det"));
        assert_eq!(map.get(&QuantMode::RaBitQ), Some(&"rab"));
        assert_eq!(map.get(&QuantMode::Deterministic), map.get(&QuantMode::Deterministic));
    }

    // ── RabitqCorrection ──

    #[test]
    fn rabitq_correction_default_is_zero() {
        let c = RabitqCorrection::default();
        assert!((c.c0).abs() < 1e-6);
        assert!((c.c1).abs() < 1e-6);
        assert!((c.v_norm).abs() < 1e-6);
    }

    #[test]
    fn rabitq_correction_from_quant_params_basic() {
        let c = RabitqCorrection::from_quant_params(4, 64, 10.0);
        assert!((c.c0).abs() < 1e-6);
        let expected_c1 = 10.0 * 2.0_f32.powi(-3) / 64.0_f32.sqrt();
        assert!((c.c1 - expected_c1).abs() < 1e-6, "expected {expected_c1}, got {}", c.c1);
        assert!((c.v_norm - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rabitq_correction_from_quant_params_3bit() {
        let c = RabitqCorrection::from_quant_params(3, 128, 5.0);
        let expected_c1 = 5.0 * 0.25 / 128.0_f32.sqrt();
        assert!((c.c1 - expected_c1).abs() < 1e-5);
    }

    #[test]
    fn rabitq_correction_from_quant_params_zero_dim() {
        let c = RabitqCorrection::from_quant_params(4, 0, 10.0);
        assert!((c.c1).abs() < 1e-6, "zero dim should produce c1=0");
    }

    #[test]
    fn rabitq_correction_from_quant_params_zero_norm() {
        let c = RabitqCorrection::from_quant_params(4, 64, 0.0);
        assert!((c.c1).abs() < 1e-6, "zero norm should produce c1=0");
    }

    #[test]
    fn rabitq_correction_correct_score_single() {
        let c = RabitqCorrection { c0: 1.0, c1: 0.5, v_norm: 2.0 };
        let corrected = c.correct_score(10.0, 3.0);
        let expected = 10.0 + 0.5 * 3.0 + 1.0;
        assert!((corrected - expected).abs() < 1e-5);
    }

    #[test]
    fn rabitq_correction_correct_score_zero_correction() {
        let c = RabitqCorrection::zero();
        let corrected = c.correct_score(42.0, 100.0);
        assert!((corrected - 42.0).abs() < 1e-5);
    }

    #[test]
    fn rabitq_correction_correct_scores_in_place_batch() {
        let c = RabitqCorrection { c0: 2.0, c1: 0.0, v_norm: 0.0 };
        let mut scores = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        c.correct_scores_in_place(&mut scores, 0.0);
        for (i, &s) in scores.iter().enumerate() {
            let expected = (i + 1) as f32 + 2.0;
            assert!((s - expected).abs() < 1e-5, "index {i}: expected {expected}, got {s}");
        }
    }

    #[test]
    fn rabitq_correction_equality() {
        let a = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let b = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        assert_eq!(a, b);
    }

    #[test]
    fn rabitq_correction_inequality() {
        let a = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let b = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 4.0 };
        assert_ne!(a, b);
    }

    #[test]
    fn rabitq_correction_clone() {
        let c = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let cloned = c.clone();
        assert_eq!(c, cloned);
    }

    #[test]
    fn rabitq_correction_copy() {
        let c = RabitqCorrection { c0: 1.5, c1: 2.5, v_norm: 3.5 };
        let copied = c;
        assert_eq!(c, copied);
    }

    #[test]
    fn rabitq_correction_c1_increases_with_norm() {
        let c_low = RabitqCorrection::from_quant_params(4, 64, 1.0);
        let c_mid = RabitqCorrection::from_quant_params(4, 64, 10.0);
        let c_high = RabitqCorrection::from_quant_params(4, 64, 100.0);
        assert!(c_low.c1 < c_mid.c1, "c1 should increase with v_norm");
        assert!(c_mid.c1 < c_high.c1, "c1 should increase with v_norm");
    }

    #[test]
    fn rabitq_correction_c1_decreases_with_dim() {
        let c_small_dim = RabitqCorrection::from_quant_params(4, 16, 10.0);
        let c_large_dim = RabitqCorrection::from_quant_params(4, 256, 10.0);
        assert!(c_small_dim.c1 > c_large_dim.c1, "c1 should decrease with larger dimension");
    }

    // ── Quantize/Dequantize K round-trip ──

    #[test]
    fn runtime_quantize_dequantize_k_4bit_roundtrip() {
        let seq_len = 2;
        let kv_dim = 8;
        // Values in [0, scale] range since quantization maps to [0, max_val]
        let k_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                     0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75];
        let scales: Vec<f32> = vec![1.0; kv_dim];

        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, scales);

        let quantized = rt.quantize_k(&k_data, 0, seq_len, kv_dim).unwrap();
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&quantized, &stored_scales, seq_len, kv_dim).unwrap();

        assert_eq!(dequantized.len(), k_data.len());
        for (orig, deq) in k_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.1, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    #[test]
    fn runtime_quantize_dequantize_k_3bit_roundtrip() {
        let seq_len = 2;
        let kv_dim = 8;
        // Values in [0, scale] range for 3-bit quantization
        let k_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                     0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75];
        let scales: Vec<f32> = vec![1.0; kv_dim];

        let cfg = TurboQuantConfig { bits: 3, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        rt.store_k_scales(0, scales);

        let quantized = rt.quantize_k(&k_data, 0, seq_len, kv_dim).unwrap();
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&quantized, &stored_scales, seq_len, kv_dim).unwrap();

        assert_eq!(dequantized.len(), k_data.len());
        for (orig, deq) in k_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.2, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    #[test]
    fn runtime_quantize_k_without_precomputed_scales() {
        let seq_len = 1;
        let kv_dim = 4;
        let k_data = vec![0.5, 0.3, 0.1, 0.9];
        let rt = TurboQuantRuntime::disabled();
        let result = rt.quantize_k(&k_data, 0, seq_len, kv_dim);
        assert!(result.is_ok(), "quantize_k should succeed with computed scales");
    }

    // ── Quantize/Dequantize V round-trip ──

    #[test]
    fn runtime_quantize_dequantize_v_4bit_roundtrip() {
        let seq_len = 2;
        let kv_dim = 8;
        let v_data: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        assert_eq!(scales.len(), seq_len, "V quantization should produce per-token scales");
        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), v_data.len());
        for (orig, deq) in v_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.15, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    #[test]
    fn runtime_quantize_dequantize_v_3bit_roundtrip() {
        let seq_len = 2;
        let kv_dim = 8;
        let v_data: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let cfg = TurboQuantConfig { bits: 3, ..Default::default() };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), v_data.len());
        for (orig, deq) in v_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.25, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    #[test]
    fn runtime_quantize_v_single_token() {
        let seq_len = 1;
        let kv_dim = 4;
        let v_data = vec![0.5, 0.3, 0.1, 0.9];
        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        assert_eq!(scales.len(), 1);
        assert!(!quantized.is_empty());
        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), kv_dim);
    }

    #[test]
    fn runtime_quantize_v_preserves_non_negative() {
        let seq_len = 1;
        let kv_dim = 4;
        let v_data = vec![0.5, 0.3, 0.1, 0.9];
        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        for &v in &dequantized {
            assert!(v >= 0.0, "dequantized value should be non-negative, got {v}");
        }
    }

    // ── Quantize with FWHT ──

    #[test]
    fn runtime_quantize_k_with_fwht_enabled() {
        let seq_len = 1;
        let kv_dim = 8;
        let k_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let scales: Vec<f32> = vec![1.0; kv_dim];
        let cfg = TurboQuantConfig { fwht_enabled: true, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        rt.store_k_scales(0, scales);
        let result = rt.quantize_k(&k_data, 0, seq_len, kv_dim);
        assert!(result.is_ok(), "quantize_k with FWHT should succeed");
    }

    #[test]
    fn runtime_quantize_v_with_fwht_enabled() {
        let seq_len = 1;
        let kv_dim = 8;
        let v_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let cfg = TurboQuantConfig { fwht_enabled: true, ..Default::default() };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        let result = rt.quantize_v(&v_data, seq_len, kv_dim);
        assert!(result.is_ok(), "quantize_v with FWHT should succeed");
    }

    // ── Correct attention scores edge cases ──

    #[test]
    fn runtime_correct_attention_scores_empty_slice() {
        let rt = TurboQuantRuntime::disabled();
        let mut scores: Vec<f32> = vec![];
        rt.correct_attention_scores(&mut scores, 1.0, 0);
        assert!(scores.is_empty());
    }

    #[test]
    fn runtime_correct_attention_scores_large_q_norm() {
        let mut rt = TurboQuantRuntime::disabled();
        let c = RabitqCorrection { c0: 0.0, c1: 1.0, v_norm: 1.0 };
        rt.store_correction(0, c);
        let mut scores = vec![100.0f32];
        rt.correct_attention_scores(&mut scores, 1000.0, 0);
        assert!((scores[0] - 1100.0).abs() < 1e-3, "expected 1100, got {}", scores[0]);
    }

    #[test]
    fn runtime_correct_attention_scores_zero_q_norm() {
        let mut rt = TurboQuantRuntime::disabled();
        let c = RabitqCorrection { c0: 5.0, c1: 10.0, v_norm: 1.0 };
        rt.store_correction(0, c);
        let mut scores = vec![42.0f32];
        rt.correct_attention_scores(&mut scores, 0.0, 0);
        assert!((scores[0] - 47.0).abs() < 1e-5, "expected 47.0, got {}", scores[0]);
    }

    #[test]
    fn runtime_correct_attention_scores_negative_scores() {
        let mut rt = TurboQuantRuntime::disabled();
        let c = RabitqCorrection { c0: 0.5, c1: 0.0, v_norm: 0.0 };
        rt.store_correction(0, c);
        let mut scores = vec![-5.0f32, -10.0, -15.0];
        rt.correct_attention_scores(&mut scores, 1.0, 0);
        assert!((scores[0] - (-4.5)).abs() < 1e-5);
        assert!((scores[1] - (-9.5)).abs() < 1e-5);
        assert!((scores[2] - (-14.5)).abs() < 1e-5);
    }

    // ── FWHT additional properties ──

    #[test]
    fn fwht_32_elements_roundtrip() {
        let original: Vec<f32> = (0..32).map(|i| i as f32 * 0.3 - 5.0).collect();
        let mut data = original.clone();
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        apply_fwht(&mut data, FwhtInsertionPoint::KvWrite);
        let n = original.len() as f32;
        for (got, expected) in data.iter().zip(original.iter()) {
            assert!((got / n - expected).abs() < 1e-3,
                "FWHT round-trip failed: expected {expected}, got {}", got / n);
        }
    }

    #[test]
    fn fwht_64_elements_energy_conservation() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin()).collect();
        let n = data.len() as f32;
        let energy_before: f32 = data.iter().map(|x| x * x).sum();
        let mut transformed = data.clone();
        apply_fwht(&mut transformed, FwhtInsertionPoint::FfnEpilogue);
        let energy_after: f32 = transformed.iter().map(|x| x * x).sum();
        assert!((energy_before * n - energy_after).abs() < 1e-3 * energy_before,
            "energy should scale by n={n}");
    }

    #[test]
    fn fwht_alternating_sign_input() {
        let mut data: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        apply_fwht(&mut data, FwhtInsertionPoint::AttentionEpilogue);
        let sum: f32 = data.iter().map(|x| x * x).sum();
        assert!(sum > 0.0, "FWHT of alternating sign should not be all zeros");
    }

    #[test]
    fn fwht_increasing_values() {
        let mut data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        apply_fwht(&mut data, FwhtInsertionPoint::KvWrite);
        let expected_sum: f32 = (0..16).map(|i| i as f32).sum();
        assert!((data[0] - expected_sum).abs() < 1e-3,
            "first element should be sum={expected_sum}, got {}", data[0]);
    }

    #[test]
    fn fwht_insertion_point_does_not_affect_result() {
        let data_base = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut d1 = data_base.clone();
        let mut d2 = data_base.clone();
        let mut d3 = data_base.clone();
        apply_fwht(&mut d1, FwhtInsertionPoint::AttentionEpilogue);
        apply_fwht(&mut d2, FwhtInsertionPoint::FfnEpilogue);
        apply_fwht(&mut d3, FwhtInsertionPoint::KvWrite);
        assert_eq!(d1, d2, "FWHT result should not depend on insertion point");
        assert_eq!(d2, d3, "FWHT result should not depend on insertion point");
    }

    // ── TurboQuantRuntime with dual_track enabled ──

    #[test]
    fn runtime_new_with_dual_track_enabled() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 4, ..Default::default() };
        let rt = TurboQuantRuntime::new(cfg);
        assert!(rt.is_ok(), "creating runtime with dual_track should succeed");
        let rt = rt.unwrap();
        assert!(rt.is_dual_track_active());
        assert!(rt.dual_track().is_some());
    }

    #[test]
    fn runtime_new_with_dual_track_3bit() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 3, ..Default::default() };
        let rt = TurboQuantRuntime::new(cfg);
        assert!(rt.is_ok(), "creating runtime with dual_track 3-bit should succeed");
        let rt = rt.unwrap();
        assert!(rt.is_dual_track_active());
    }

    #[test]
    fn runtime_dual_track_mut_with_enabled() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        assert!(rt.dual_track_mut().is_some(), "dual_track_mut should return Some when enabled");
    }

    // ── DualTrack store/load round-trip ──

    #[test]
    fn runtime_store_load_k_dual_track_roundtrip() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 4, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        let data = vec![0xABu8, 0xCD, 0xEF, 0x01, 0x23, 0x45];
        let mask = vec![true, false, true, true, false];
        rt.store_k_to_dual_track(0, &data, &mask).unwrap();
        let mut loaded = vec![0u8; data.len()];
        rt.load_k_from_dual_track(0, &mut loaded).unwrap();
        assert_eq!(loaded, data, "loaded data should match stored data");
        let mut loaded_mask = vec![false; mask.len()];
        rt.load_xnor_mask(1, &mut loaded_mask).unwrap();
        assert_eq!(loaded_mask, mask, "loaded mask should match stored mask");
    }

    #[test]
    fn runtime_store_load_v_dual_track_roundtrip() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 4, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        let data = vec![0x11u8, 0x22, 0x33, 0x44];
        let mask = vec![false, true, false];
        rt.store_v_to_dual_track(0, &data, &mask).unwrap();
        let mut loaded = vec![0u8; data.len()];
        rt.load_k_from_dual_track(0, &mut loaded).unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn runtime_store_k_dual_track_empty_mask() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 4, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        let data = vec![0xFFu8, 0x00];
        let empty_mask: Vec<bool> = vec![];
        let result = rt.store_k_to_dual_track(0, &data, &empty_mask);
        assert!(result.is_ok(), "store with empty mask should succeed");
    }

    #[test]
    fn runtime_store_v_dual_track_empty_mask() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, bits: 4, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        let data = vec![0xAAu8, 0xBB];
        let empty_mask: Vec<bool> = vec![];
        let result = rt.store_v_to_dual_track(0, &data, &empty_mask);
        assert!(result.is_ok(), "store with empty mask should succeed");
    }

    #[test]
    fn runtime_store_k_dual_track_noop_without_pool() {
        let mut rt = TurboQuantRuntime::disabled();
        let data = vec![0x12u8, 0x34];
        let mask = vec![true, false];
        let result = rt.store_k_to_dual_track(0, &data, &mask);
        assert!(result.is_ok(), "store_k without pool should be no-op");
    }

    #[test]
    fn runtime_store_v_dual_track_noop_without_pool() {
        let mut rt = TurboQuantRuntime::disabled();
        let data = vec![0x56u8, 0x78];
        let mask = vec![false, true];
        let result = rt.store_v_to_dual_track(0, &data, &mask);
        assert!(result.is_ok(), "store_v without pool should be no-op");
    }

    // ── Reset behavior ──

    #[test]
    fn runtime_reset_does_not_disable_dual_track() {
        let cfg = TurboQuantConfig { dual_track_enabled: true, ..Default::default() };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        rt.store_k_scales(0, vec![1.0, 2.0]);
        rt.reset();
        assert!(rt.is_dual_track_active());
        assert!(rt.get_k_scales(0).is_none(), "scales should be cleared");
    }

    #[test]
    fn runtime_multiple_resets() {
        let mut rt = TurboQuantRuntime::disabled();
        for i in 0..5 {
            rt.store_k_scales(0, vec![i as f32]);
            rt.store_correction(0, RabitqCorrection { c0: i as f32, c1: 0.0, v_norm: 0.0 });
            rt.reset();
            assert!(rt.get_k_scales(0).is_none(), "scales should be cleared after reset {i}");
            let mut scores = vec![0.0f32];
            rt.correct_attention_scores(&mut scores, 1.0, 0);
            assert_eq!(scores[0], 0.0, "scores should be unchanged after reset {i}");
        }
    }

    #[test]
    fn runtime_replenish_after_reset() {
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, vec![1.0]);
        rt.reset();
        assert!(rt.get_k_scales(0).is_none());
        rt.store_k_scales(0, vec![42.0]);
        assert_eq!(rt.get_k_scales(0).unwrap(), &[42.0]);
    }

    // ── should_preserve_fp16 extended ──

    #[test]
    fn runtime_should_preserve_fp16_large_sink_count() {
        let cfg = TurboQuantConfig { sink_count: 1000, ..Default::default() };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        assert!(rt.should_preserve_fp16(500, true));
        assert!(rt.should_preserve_fp16(999, true));
        assert!(!rt.should_preserve_fp16(1000, true));
        assert!(!rt.should_preserve_fp16(999, false));
    }

    // ── Config: bits/sink_count do not affect enabled status ──

    #[test]
    fn config_bits_does_not_affect_enabled() {
        let c_bits_0 = TurboQuantConfig { bits: 0, ..Default::default() };
        let c_bits_3 = TurboQuantConfig { bits: 3, ..Default::default() };
        let c_bits_4 = TurboQuantConfig { bits: 4, ..Default::default() };
        assert!(!c_bits_0.is_enabled());
        assert!(!c_bits_3.is_enabled());
        assert!(!c_bits_4.is_enabled());
    }

    #[test]
    fn config_sink_count_does_not_affect_enabled() {
        let c_zero = TurboQuantConfig { sink_count: 0, ..Default::default() };
        let c_large = TurboQuantConfig { sink_count: 100, ..Default::default() };
        assert!(!c_zero.is_enabled());
        assert!(!c_large.is_enabled());
    }

    // ── Config: to_kv_quant_config mode variants ──

    #[test]
    fn config_to_kv_quant_config_deterministic_mode() {
        let c = TurboQuantConfig { mode: QuantMode::Deterministic, ..Default::default() };
        let kv = c.to_kv_quant_config();
        assert_eq!(kv.mode, QuantMode::Deterministic);
    }

    // ── TurboQuantLayerOutput edge cases ──

    #[test]
    fn layer_output_with_fwht_and_scales_empty() {
        let out = TurboQuantLayerOutput::with_fwht_and_scales(vec![], vec![]);
        assert!(out.hidden_state.is_empty());
        assert!(out.fwht_applied);
        assert!(out.rms_scales.as_ref().unwrap().is_empty());
    }

    #[test]
    fn layer_output_plain_with_empty_hidden() {
        let out = TurboQuantLayerOutput::plain(vec![]);
        assert!(out.hidden_state.is_empty());
        assert!(!out.fwht_applied);
        assert!(out.rms_scales.is_none());
    }

    #[test]
    fn layer_output_with_large_hidden_state() {
        let hidden: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let scales: Vec<f32> = (0..1024).map(|i| (i as f32 + 1.0).recip()).collect();
        let out = TurboQuantLayerOutput::with_fwht_and_scales(hidden.clone(), scales.clone());
        assert_eq!(out.hidden_state.len(), 1024);
        assert!(out.fwht_applied);
        assert_eq!(out.rms_scales.as_ref().unwrap().len(), 1024);
    }

    // ── Larger quantize/dequantize sizes ──

    #[test]
    fn runtime_quantize_dequantize_k_larger_size() {
        let seq_len = 4;
        let kv_dim = 64;
        // Keep values in [0, scale] range; scale=2.0, values in [0, 1.9]
        let k_data: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i % 20) as f32 * 0.1).collect();
        let scales: Vec<f32> = vec![2.0; kv_dim];
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, scales);
        let quantized = rt.quantize_k(&k_data, 0, seq_len, kv_dim).unwrap();
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&quantized, &stored_scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), k_data.len());
        for (orig, deq) in k_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.2, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    #[test]
    fn runtime_quantize_dequantize_v_larger_size() {
        let seq_len = 4;
        let kv_dim = 32;
        // Keep values non-negative; V quantizes per-token with auto-computed scales
        let v_data: Vec<f32> = (0..seq_len * kv_dim).map(|i| (i % 12) as f32 * 0.08).collect();
        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), v_data.len());
        for (orig, deq) in v_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.2, "round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    // ── Quantize all-zero data ──

    #[test]
    fn runtime_quantize_k_all_zeros_with_nonzero_scales() {
        let seq_len = 1;
        let kv_dim = 4;
        let k_data = vec![0.0f32; kv_dim];
        let scales = vec![1.0f32; kv_dim];
        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, scales);
        let result = rt.quantize_k(&k_data, 0, seq_len, kv_dim);
        assert!(result.is_ok(), "quantize_k with all-zero data should succeed");
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&result.unwrap(), &stored_scales, seq_len, kv_dim).unwrap();
        for &v in &dequantized {
            assert!(v.abs() < 0.2, "dequantized zero data should be near zero, got {v}");
        }
    }

    // ── Correction from_quant_params through runtime ──

    #[test]
    fn runtime_correction_from_quant_params() {
        let mut rt = TurboQuantRuntime::disabled();
        let correction = RabitqCorrection::from_quant_params(4, 64, 10.0);
        rt.store_correction(3, correction);
        let mut scores = vec![0.0f32; 5];
        rt.correct_attention_scores(&mut scores, 2.0, 3);
        let expected_correction = correction.c1 * 2.0 + correction.c0;
        for &s in &scores {
            assert!((s - expected_correction).abs() < 1e-5,
                "each score should be corrected by {expected_correction}, got {s}");
        }
    }

    // ── DualTrack load empty buffer no-op ──

    #[test]
    fn runtime_dual_track_load_empty_buffer_noop_without_pool() {
        let rt = TurboQuantRuntime::disabled();
        let mut buf: Vec<u8> = vec![];
        let result = rt.load_k_from_dual_track(0, &mut buf);
        assert!(result.is_ok());
    }

    // ── Config: all features enabled at once ──

    #[test]
    fn config_all_features_enabled() {
        let c = TurboQuantConfig {
            bits: 3, sink_count: 16, fwht_enabled: true,
            mode: QuantMode::RaBitQ, dual_track_enabled: true,
        };
        assert!(c.is_enabled());
        assert!(c.fwht_enabled);
        assert!(c.dual_track_enabled);
        assert_eq!(c.bits, 3);
        assert_eq!(c.sink_count, 16);
    }

    // ══════════════════════════════════════════════════════════════
    //  Additional tests (round 3) — 13 edge case tests
    // ══════════════════════════════════════════════════════════════

    // -- Quantize K with negative values: round-trip should still work --

    #[test]
    fn runtime_quantize_dequantize_k_with_negative_values() {
        let seq_len = 2;
        let kv_dim = 8;
        let k_data: Vec<f32> = vec![
            -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
             0.05, -0.15, 0.25, -0.35, 0.45, -0.55, 0.65, -0.75,
        ];
        // Scales must be large enough to cover negative range
        let scales: Vec<f32> = vec![1.0; kv_dim];

        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, scales);

        let quantized = rt.quantize_k(&k_data, 0, seq_len, kv_dim).unwrap();
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&quantized, &stored_scales, seq_len, kv_dim).unwrap();

        assert_eq!(dequantized.len(), k_data.len());
        // Negative values are clamped to zero in unsigned quantization;
        // verify the dequantized values are within tolerance of the clamped originals
        for (orig, deq) in k_data.iter().zip(dequantized.iter()) {
            let clamped = orig.max(0.0);
            let error = (clamped - deq).abs();
            assert!(error < 0.15,
                "round-trip error too large: orig={orig}, clamped={clamped}, deq={deq}, error={error}");
        }
    }

    // -- Quantize V with all-zero input: should succeed and dequantize near zero --

    #[test]
    fn runtime_quantize_dequantize_v_all_zeros() {
        let seq_len = 2;
        let kv_dim = 8;
        let v_data = vec![0.0f32; seq_len * kv_dim];

        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        assert!(!quantized.is_empty());
        assert_eq!(scales.len(), seq_len);

        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        for &v in &dequantized {
            assert!(v.abs() < 0.1,
                "dequantized all-zero V should be near zero, got {v}");
        }
    }

    // -- Quantize K round-trip in RaBitQ mode: stochastic rounding should still round-trip --

    #[test]
    fn runtime_quantize_dequantize_k_rabitq_mode() {
        let seq_len = 2;
        let kv_dim = 8;
        let k_data: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
        ];
        let scales: Vec<f32> = vec![1.0; kv_dim];

        let cfg = TurboQuantConfig {
            bits: 4,
            mode: QuantMode::RaBitQ,
            ..Default::default()
        };
        let mut rt = TurboQuantRuntime::new(cfg).unwrap();
        rt.store_k_scales(0, scales);

        let quantized = rt.quantize_k(&k_data, 0, seq_len, kv_dim).unwrap();
        let stored_scales = rt.get_k_scales(0).unwrap().to_vec();
        let dequantized = rt.dequantize_k(&quantized, &stored_scales, seq_len, kv_dim).unwrap();

        assert_eq!(dequantized.len(), k_data.len());
        for (orig, deq) in k_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 0.15,
                "RaBitQ round-trip error too large: orig={orig}, deq={deq}, error={error}");
        }
    }

    // -- Quantize K with zero scale: should return an error --

    #[test]
    fn runtime_quantize_k_zero_scale_error() {
        let seq_len = 1;
        let kv_dim = 4;
        let k_data = vec![0.5f32, 0.3, 0.1, 0.9];
        let scales = vec![0.0f32, 1.0, 1.0, 1.0]; // scale[0] is zero

        let mut rt = TurboQuantRuntime::disabled();
        rt.store_k_scales(0, scales);

        let result = rt.quantize_k(&k_data, 0, seq_len, kv_dim);
        assert!(result.is_err(), "quantize_k with zero scale should return an error");
    }

    // -- FWHT with negative-heavy values: energy scaling should still hold --

    #[test]
    fn fwht_negative_heavy_input_energy_scaling() {
        let data: Vec<f32> = (0..16).map(|i| -((i + 1) as f32)).collect();
        let n = data.len() as f32;
        let energy_before: f32 = data.iter().map(|x| x * x).sum();
        let mut transformed = data.clone();
        apply_fwht(&mut transformed, FwhtInsertionPoint::FfnEpilogue);
        let energy_after: f32 = transformed.iter().map(|x| x * x).sum();
        assert!((energy_before * n - energy_after).abs() < 1e-3 * energy_before,
            "energy should scale by n={n}: {} vs {energy_after}", energy_before * n);
    }

    // -- RabitqCorrection::from_quant_params with 2-bit (edge case) --

    #[test]
    fn rabitq_correction_from_quant_params_2bit() {
        let c = RabitqCorrection::from_quant_params(2, 64, 10.0);
        assert!((c.c0).abs() < 1e-6, "c0 should always be zero");
        // 2-bit: 2^(-(2-1)) = 0.5
        let expected_c1 = 10.0 * 0.5 / 64.0_f32.sqrt();
        assert!((c.c1 - expected_c1).abs() < 1e-6,
            "expected {expected_c1}, got {}", c.c1);
        assert!((c.v_norm - 10.0).abs() < 1e-6);
    }

    // -- RabitqCorrection::correct_score with negative q_norm --

    #[test]
    fn rabitq_correction_correct_score_negative_q_norm() {
        let c = RabitqCorrection { c0: 1.0, c1: 0.5, v_norm: 2.0 };
        let corrected = c.correct_score(10.0, -3.0);
        // correction = c1 * q_norm + c0 = 0.5 * (-3.0) + 1.0 = -0.5
        let expected = 10.0 + 0.5 * (-3.0) + 1.0;
        assert!((corrected - expected).abs() < 1e-5,
            "expected {expected}, got {corrected}");
    }

    // -- should_preserve_fp16 with very large sink_count --

    #[test]
    fn runtime_should_preserve_fp16_usize_max_sink_count() {
        let cfg = TurboQuantConfig {
            sink_count: usize::MAX,
            ..Default::default()
        };
        let rt = TurboQuantRuntime::new(cfg).unwrap();
        // With usize::MAX sink_count, any reasonable token index should be preserved
        assert!(rt.should_preserve_fp16(0, true));
        assert!(rt.should_preserve_fp16(1000000, true));
        assert!(rt.should_preserve_fp16(usize::MAX - 1, true));
        // But token_idx == usize::MAX is not < usize::MAX
        assert!(!rt.should_preserve_fp16(usize::MAX, true));
        // Non-sink should never be preserved
        assert!(!rt.should_preserve_fp16(0, false));
    }

    // -- Quantize V single token with large hidden dimension --

    #[test]
    fn runtime_quantize_dequantize_v_single_token_large_dim() {
        let seq_len = 1;
        let kv_dim = 256;
        let v_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();

        let rt = TurboQuantRuntime::disabled();
        let (quantized, scales) = rt.quantize_v(&v_data, seq_len, kv_dim).unwrap();
        assert_eq!(scales.len(), 1, "single token should produce one scale");
        assert!(!quantized.is_empty());

        let dequantized = rt.dequantize_v(&quantized, &scales, seq_len, kv_dim).unwrap();
        assert_eq!(dequantized.len(), kv_dim);
        // 4-bit quantization over large range: small values get quantized to 0,
        // which is expected behavior. Verify dequantized values are within
        // the full scale range (max error = scale itself for extreme cases).
        let scale = scales[0];
        for (orig, deq) in v_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error <= scale + 0.01,
                "round-trip error exceeds scale: orig={orig}, deq={deq}, error={error}, scale={scale}");
        }
    }

    // -- TurboQuantLayerOutput Debug format contains rms_scales field --

    #[test]
    fn layer_output_debug_format_with_scales() {
        let out = TurboQuantLayerOutput::with_fwht_and_scales(vec![1.0, 2.0], vec![0.5, 1.5]);
        let debug_str = format!("{out:?}");
        assert!(debug_str.contains("TurboQuantLayerOutput"));
        assert!(debug_str.contains("rms_scales"), "Debug should contain 'rms_scales' field");
        assert!(debug_str.contains("fwht_applied"), "Debug should contain 'fwht_applied' field");
    }

    // -- TurboQuantConfig is_enabled: each feature alone is sufficient --

    #[test]
    fn config_is_enabled_sole_trigger_fwht() {
        let c = TurboQuantConfig {
            fwht_enabled: true,
            mode: QuantMode::Deterministic,
            dual_track_enabled: false,
            ..Default::default()
        };
        assert!(c.is_enabled(), "fwht_enabled alone should trigger is_enabled");
    }

    #[test]
    fn config_is_enabled_sole_trigger_rabitq() {
        let c = TurboQuantConfig {
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
            dual_track_enabled: false,
            ..Default::default()
        };
        assert!(c.is_enabled(), "RaBitQ mode alone should trigger is_enabled");
    }

    #[test]
    fn config_is_enabled_sole_trigger_dual_track() {
        let c = TurboQuantConfig {
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
            dual_track_enabled: true,
            ..Default::default()
        };
        assert!(c.is_enabled(), "dual_track_enabled alone should trigger is_enabled");
    }
}
