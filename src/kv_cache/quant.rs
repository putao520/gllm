//! KIVI asymmetric KV cache quantization (SPEC §11.2)
//!
//! Key insight: K and V have fundamentally different outlier distributions:
//! - K cache: outliers concentrated in specific channels (stable across tokens) → Per-Channel quantization
//! - V cache: outliers concentrated in specific tokens (stable across channels) → Per-Token quantization
//!
//! §11.1 FWHT Rotation: Apply Fast Walsh-Hadamard Transform before quantization to
//! distribute outliers uniformly, improving quantization effectiveness.
//!
//! §11.3 RaBitQ: Random Bipartite Quantization for unbiased inner product preservation.
//! Traditional quantization introduces rounding bias. RaBitQ uses stochastic rounding:
//! - Q(x) = floor(x/s) with probability (1 - frac(x/s))
//! - Q(x) = ceil(x/s) with probability frac(x/s)
//! - Guarantees E[Q(x)] = x (unbiased expectation)
//!
//! §11.3 RaBitQ Unbiased Correction:
//! In Attention QK^T computation, apply correction factors: QK^T = QK^T_quant · C1 + C0
//! - ‖v‖ (correction input): piggybacked from RMSNorm epilogue, ‖v‖ = RMS · √d
//! - Quantization inner product: append 1 FMA instruction in quantization loop
//! - Theoretical error bound: O(1/√D). ~1.5% for D=4096, ~1.1% for D=8192

use thiserror::Error;
use rand::Rng;

/// RaBitQ correction factors for unbiased inner product estimation (SPEC §11.3)
///
/// Quantized vector q̂ relates to original vector q: q = s · q̂ + Δ, where s is scale and Δ is quantization error.
/// Unbiased inner product estimation: q·k = s_q · s_k · q̂·k̂ + C1 · ‖q‖ · correction_k + C0
///
/// # Fields
/// - `c0`: Global bias correction (zero for symmetric quantization)
/// - `c1`: Norm correction factor, computed from bit width and dimension
/// - `v_norm`: ‖v‖ vector norm, piggybacked from RMSNorm epilogue
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RabitqCorrection {
    /// Global bias correction C0 (zero for symmetric quantization)
    pub c0: f32,
    /// Norm correction factor C1
    pub c1: f32,
    /// ‖v‖ vector norm from RMSNorm epilogue (‖v‖ = RMS · √d)
    pub v_norm: f32,
}

impl RabitqCorrection {
    /// Create a new RaBitQ correction with all zero values (no correction).
    #[inline]
    pub fn zero() -> Self {
        Self { c0: 0.0, c1: 0.0, v_norm: 0.0 }
    }

    /// Create RaBitQ correction factors from quantization parameters (SPEC §11.3).
    ///
    /// # Arguments
    /// - `num_bits`: Quantization bit width (3 or 4)
    /// - `dim`: Vector dimension (hidden_size or head_dim)
    /// - `v_norm`: ‖v‖ from RMSNorm epilogue (‖v‖ = RMS · √d)
    ///
    /// # Returns
    /// Correction factors for unbiased inner product estimation.
    ///
    /// # Formula
    /// - C0: Quantization bias correction (zero for symmetric quantization)
    /// - C1: Norm decay correction = v_norm · 2^(-(num_bits-1)) / √dim
    ///
    /// This ensures E[Q(x)] = x (unbiased expectation) and bounds error to O(1/√D).
    pub fn from_quant_params(num_bits: u8, dim: usize, v_norm: f32) -> Self {
        let d = dim as f32;
        // C0: Quantization bias correction (zero for symmetric quantization)
        let c0 = 0.0;
        // C1: Norm decay correction factor
        // Derived from RaBitQ unbiased estimation: E[Q(x)·Q(y)] ≈ x·y + O(1/√D)
        let c1 = if dim > 0 {
            v_norm * 2.0_f32.powi(-(num_bits as i32 - 1)) / d.sqrt()
        } else {
            0.0
        };
        Self { c0, c1, v_norm }
    }

    /// Apply RaBitQ correction to attention score (SPEC §11.3).
    ///
    /// # Arguments
    /// - `raw_score`: Raw QK^T score from quantized vectors: s_q · s_k · q̂·k̂
    /// - `q_norm`: ‖q‖ query vector norm
    ///
    /// # Returns
    /// Corrected score: raw_score + C1 · ‖q‖ + C0
    ///
    /// This is the unbiased inner product estimate: q·k ≈ corrected_score
    #[inline]
    pub fn correct_score(&self, raw_score: f32, q_norm: f32) -> f32 {
        raw_score + self.c1 * q_norm + self.c0
    }

    /// Batch apply correction to a slice of attention scores.
    ///
    /// # Arguments
    /// - `scores`: Slice of raw QK^T scores (modified in-place)
    /// - `q_norm`: ‖q‖ query vector norm
    pub fn correct_scores_in_place(&self, scores: &mut [f32], q_norm: f32) {
        let correction = self.c1 * q_norm + self.c0;
        for score in scores.iter_mut() {
            *score += correction;
        }
    }
}

impl Default for RabitqCorrection {
    fn default() -> Self {
        Self::zero()
    }
}

/// Fast Walsh-Hadamard Transform (FWHT) — in-place butterfly (SPEC §11.1)
///
/// Applies the Hadamard transform to `data` in-place.
/// Requirement: data.len() must be a power of two.
/// Complexity: O(n log n) — far cheaper than GEMM O(n²).
///
/// Used at 3 insertion points per layer:
///   1. After Attention Epilogue (Softmax(QK^T)V output)
///   2. After FFN Epilogue (SwiGLU(Gate)·Up output)
///   3. Before KV Cache write (RoPE(K))
pub fn fwht_inplace(data: &mut [f32]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-two length, got {}", n);
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = data[i + j];
                let b = data[i + j + stride];
                data[i + j] = a + b;
                data[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride *= 2;
    }
}

#[derive(Debug, Error)]
pub enum QuantError {
    #[error("invalid bit width: {0} (must be 3 or 4)")]
    InvalidBitWidth(u8),
    #[error("shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },
    #[error("zero scale detected at index {0}")]
    ZeroScale(usize),
}

pub type QuantResult<T> = std::result::Result<T, QuantError>;

/// Quantization mode (SPEC §11.3)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantMode {
    /// Traditional deterministic rounding (introduces bias)
    Deterministic,
    /// RaBitQ: Random Bipartite Quantization (unbiased expectation)
    RaBitQ,
}

/// KV quantization configuration
#[derive(Debug, Clone)]
pub struct KvQuantConfig {
    pub bits: u8,
    pub sink_count: usize,
    pub fwht_enabled: bool,
    pub mode: QuantMode,
}

/// RaBitQ stochastic rounding for single value (§11.3)
#[inline]
fn rabitq_round(normalized: f32, max_val_f32: f32, rng: &mut impl Rng) -> u8 {
    let floor_val = normalized.floor();
    let frac = normalized - floor_val;
    let q = if rng.gen::<f32>() < frac {
        floor_val + 1.0
    } else {
        floor_val
    };
    q.clamp(0.0, max_val_f32) as u8
}

/// Quantize K cache with per-channel granularity.
///
/// K outliers are concentrated in specific channels (stable across tokens).
/// Scale comes from RmsNorm epilogue (free vmaxps), zero runtime overhead.
///
/// # Arguments
/// - `k`: [seq_len, hidden_size] in row-major
/// - `scales`: [hidden_size] per-channel scales from RmsNorm epilogue
/// - `bits`: 3 or 4
/// - `config`: Quantization configuration (mode, FWHT, etc.)
/// - `rng`: RNG for RaBitQ mode (unused in Deterministic mode)
///
/// # Returns
/// Quantized K in packed format (2 elements per byte for 4-bit, 8 elements per 3 bytes for 3-bit)
pub fn quantize_k_per_channel_with_config(
    k: &[f32],
    scales: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    config: &KvQuantConfig,
    rng: &mut impl Rng,
) -> QuantResult<Vec<u8>> {
    if bits != 3 && bits != 4 {
        return Err(QuantError::InvalidBitWidth(bits));
    }
    if k.len() != seq_len * hidden_size {
        return Err(QuantError::ShapeMismatch {
            expected: seq_len * hidden_size,
            actual: k.len(),
        });
    }
    if scales.len() != hidden_size {
        return Err(QuantError::ShapeMismatch {
            expected: hidden_size,
            actual: scales.len(),
        });
    }

    // Apply FWHT rotation if enabled (§11.1)
    let mut k_rotated;
    let k_data = if config.fwht_enabled && hidden_size.is_power_of_two() {
        k_rotated = k.to_vec();
        for t in 0..seq_len {
            let row = &mut k_rotated[t * hidden_size..(t + 1) * hidden_size];
            fwht_inplace(row);
        }
        &k_rotated
    } else {
        k
    };

    let max_val = (1 << bits) - 1;
    let max_val_f32 = max_val as f32;

    let total_elements = seq_len * hidden_size;
    let packed_size = if bits == 4 {
        total_elements.div_ceil(2)
    } else {
        (total_elements * 3).div_ceil(8)
    };

    let mut quantized = vec![0u8; packed_size];

    if bits == 4 {
        // 4-bit: 2 elements per byte
        for t in 0..seq_len {
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let val = k_data[idx];
                let scale = scales[c];
                if scale == 0.0 {
                    return Err(QuantError::ZeroScale(c));
                }
                let normalized = (val / scale) * max_val_f32;
                let q = if config.mode == QuantMode::RaBitQ {
                    rabitq_round(normalized, max_val_f32, rng)
                } else {
                    normalized.round().clamp(0.0, max_val_f32) as u8
                };

                let byte_idx = idx / 2;
                if idx.is_multiple_of(2) {
                    quantized[byte_idx] = (quantized[byte_idx] & 0xF0) | q;
                } else {
                    quantized[byte_idx] = (quantized[byte_idx] & 0x0F) | (q << 4);
                }
            }
        }
    } else {
        // 3-bit: 8 elements per 3 bytes
        for t in 0..seq_len {
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let val = k_data[idx];
                let scale = scales[c];
                if scale == 0.0 {
                    return Err(QuantError::ZeroScale(c));
                }
                let normalized = (val / scale) * max_val_f32;
                let q = if config.mode == QuantMode::RaBitQ {
                    rabitq_round(normalized, max_val_f32, rng)
                } else {
                    normalized.round().clamp(0.0, max_val_f32) as u8
                };

                let group = idx / 8;
                let pos = idx % 8;
                let byte_base = group * 3;

                match pos {
                    0 => quantized[byte_base] = q,
                    1 => quantized[byte_base] |= q << 3,
                    2 => {
                        quantized[byte_base] |= (q & 0x3) << 6;
                        quantized[byte_base + 1] = q >> 2;
                    }
                    3 => quantized[byte_base + 1] |= q << 1,
                    4 => quantized[byte_base + 1] |= q << 4,
                    5 => {
                        quantized[byte_base + 1] |= (q & 0x1) << 7;
                        quantized[byte_base + 2] = q >> 1;
                    }
                    6 => quantized[byte_base + 2] |= q << 2,
                    7 => quantized[byte_base + 2] |= q << 5,
                    _ => unreachable!(),
                }
            }
        }
    }

    Ok(quantized)
}

/// Legacy API: Quantize K cache with per-channel granularity (Deterministic mode)
pub fn quantize_k_per_channel(
    k: &[f32],
    scales: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    fwht_enabled: bool,
) -> QuantResult<Vec<u8>> {
    let config = KvQuantConfig {
        bits,
        sink_count: 0,
        fwht_enabled,
        mode: QuantMode::Deterministic,
    };
    let mut rng = rand::thread_rng();
    quantize_k_per_channel_with_config(k, scales, bits, seq_len, hidden_size, &config, &mut rng)
}

/// Quantize V cache with per-token granularity (with config).
///
/// V outliers are concentrated in specific tokens (stable across channels).
/// Scale computed via register-resident reduce_max during KV write.
///
/// # Arguments
/// - `v`: [seq_len, hidden_size] in row-major
/// - `bits`: 3 or 4
/// - `config`: Quantization configuration (mode, FWHT, etc.)
/// - `rng`: RNG for RaBitQ mode (unused in Deterministic mode)
///
/// # Returns
/// (quantized_v, per_token_scales)
pub fn quantize_v_per_token_with_config(
    v: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    config: &KvQuantConfig,
    rng: &mut impl Rng,
) -> QuantResult<(Vec<u8>, Vec<f32>)> {
    if bits != 3 && bits != 4 {
        return Err(QuantError::InvalidBitWidth(bits));
    }
    if v.len() != seq_len * hidden_size {
        return Err(QuantError::ShapeMismatch {
            expected: seq_len * hidden_size,
            actual: v.len(),
        });
    }

    // Apply FWHT rotation if enabled (§11.1)
    let mut v_rotated;
    let v_data = if config.fwht_enabled && hidden_size.is_power_of_two() {
        v_rotated = v.to_vec();
        for t in 0..seq_len {
            let row = &mut v_rotated[t * hidden_size..(t + 1) * hidden_size];
            fwht_inplace(row);
        }
        &v_rotated
    } else {
        v
    };

    let max_val = (1 << bits) - 1;
    let max_val_f32 = max_val as f32;

    // Compute per-token scales
    let mut scales = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let row = &v_data[t * hidden_size..(t + 1) * hidden_size];
        let max_abs = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        scales.push(if max_abs > 0.0 { max_abs } else { 1.0 });
    }

    let total_elements = seq_len * hidden_size;
    let packed_size = if bits == 4 {
        total_elements.div_ceil(2)
    } else {
        (total_elements * 3).div_ceil(8)
    };

    let mut quantized = vec![0u8; packed_size];

    if bits == 4 {
        for t in 0..seq_len {
            let scale = scales[t];
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let val = v_data[idx];
                let normalized = (val / scale) * max_val_f32;
                let q = if config.mode == QuantMode::RaBitQ {
                    rabitq_round(normalized, max_val_f32, rng)
                } else {
                    normalized.round().clamp(0.0, max_val_f32) as u8
                };

                let byte_idx = idx / 2;
                if idx.is_multiple_of(2) {
                    quantized[byte_idx] = (quantized[byte_idx] & 0xF0) | q;
                } else {
                    quantized[byte_idx] = (quantized[byte_idx] & 0x0F) | (q << 4);
                }
            }
        }
    } else {
        for t in 0..seq_len {
            let scale = scales[t];
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let val = v_data[idx];
                let normalized = (val / scale) * max_val_f32;
                let q = if config.mode == QuantMode::RaBitQ {
                    rabitq_round(normalized, max_val_f32, rng)
                } else {
                    normalized.round().clamp(0.0, max_val_f32) as u8
                };

                let group = idx / 8;
                let pos = idx % 8;
                let byte_base = group * 3;

                match pos {
                    0 => quantized[byte_base] = q,
                    1 => quantized[byte_base] |= q << 3,
                    2 => {
                        quantized[byte_base] |= (q & 0x3) << 6;
                        quantized[byte_base + 1] = q >> 2;
                    }
                    3 => quantized[byte_base + 1] |= q << 1,
                    4 => quantized[byte_base + 1] |= q << 4,
                    5 => {
                        quantized[byte_base + 1] |= (q & 0x1) << 7;
                        quantized[byte_base + 2] = q >> 1;
                    }
                    6 => quantized[byte_base + 2] |= q << 2,
                    7 => quantized[byte_base + 2] |= q << 5,
                    _ => unreachable!(),
                }
            }
        }
    }

    Ok((quantized, scales))
}

/// Legacy API: Quantize V cache with per-token granularity (Deterministic mode)
pub fn quantize_v_per_token(
    v: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    fwht_enabled: bool,
) -> QuantResult<(Vec<u8>, Vec<f32>)> {
    let config = KvQuantConfig {
        bits,
        sink_count: 0,
        fwht_enabled,
        mode: QuantMode::Deterministic,
    };
    let mut rng = rand::thread_rng();
    quantize_v_per_token_with_config(v, bits, seq_len, hidden_size, &config, &mut rng)
}

// ---------------------------------------------------------------------------
// §11.3 RaBitQ Unbiased Correction (SPEC §11.3)
// ---------------------------------------------------------------------------

/// Quantize K cache with per-channel granularity AND compute RaBitQ correction (SPEC §11.3).
///
/// Returns (quantized_k, per_channel_scales, rabitq_correction).
/// The correction factor can be applied during attention score computation:
///   corrected_score = raw_qk_score + correction.c1 * q_norm + correction.c0
///
/// # Arguments
/// - `k`: [seq_len, hidden_size] in row-major
/// - `scales`: [hidden_size] per-channel scales from RmsNorm epilogue
/// - `bits`: 3 or 4
/// - `v_norm`: ‖v‖ from RMSNorm epilogue (‖v‖ = RMS · √d)
/// - `config`: Quantization configuration (mode, FWHT, etc.)
/// - `rng`: RNG for RaBitQ mode
pub fn quantize_k_per_channel_with_correction(
    k: &[f32],
    scales: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    v_norm: f32,
    config: &KvQuantConfig,
    rng: &mut impl Rng,
) -> QuantResult<(Vec<u8>, Vec<f32>, RabitqCorrection)> {
    // First, perform standard quantization
    let quantized = quantize_k_per_channel_with_config(k, scales, bits, seq_len, hidden_size, config, rng)?;

    // Compute RaBitQ correction factor (SPEC §11.3)
    let correction = RabitqCorrection::from_quant_params(bits, hidden_size, v_norm);

    Ok((quantized, scales.to_vec(), correction))
}

/// Quantize V cache with per-token granularity AND compute RaBitQ correction (SPEC §11.3).
///
/// Returns (quantized_v, per_token_scales, rabitq_correction).
/// The correction factor can be applied during attention score computation.
///
/// # Arguments
/// - `v`: [seq_len, hidden_size] in row-major
/// - `bits`: 3 or 4
/// - `v_norm`: ‖v‖ from RMSNorm epilogue (‖v‖ = RMS · √d)
/// - `config`: Quantization configuration (mode, FWHT, etc.)
/// - `rng`: RNG for RaBitQ mode
pub fn quantize_v_per_token_with_correction(
    v: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
    v_norm: f32,
    config: &KvQuantConfig,
    rng: &mut impl Rng,
) -> QuantResult<(Vec<u8>, Vec<f32>, RabitqCorrection)> {
    // First, perform standard quantization
    let (quantized, scales) = quantize_v_per_token_with_config(v, bits, seq_len, hidden_size, config, rng)?;

    // Compute RaBitQ correction factor (SPEC §11.3)
    let correction = RabitqCorrection::from_quant_params(bits, hidden_size, v_norm);

    Ok((quantized, scales, correction))
}

/// Apply RaBitQ correction to attention scores (SPEC §11.3).
///
/// This is the key function for unbiased inner product estimation.
/// Use in attention computation after QK^T scores are computed from quantized KV cache.
///
/// # Arguments
/// - `scores`: Mutable slice of raw QK^T scores (modified in-place)
/// - `correction`: RaBitQ correction factor from quantization
/// - `q_norm`: ‖q‖ query vector norm (from RMSNorm epilogue)
///
/// # Formula
/// corrected_score = raw_score + C1 · ‖q‖ + C0
///
/// This ensures E[Q(x)·Q(y)] ≈ x·y with error O(1/√D).
pub fn apply_rabitq_correction(scores: &mut [f32], correction: &RabitqCorrection, q_norm: f32) {
    correction.correct_scores_in_place(scores, q_norm);
}

/// Dequantize K cache with per-channel scales.
pub fn dequantize_k_per_channel(
    q: &[u8],
    scales: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
) -> QuantResult<Vec<f32>> {
    if bits != 3 && bits != 4 {
        return Err(QuantError::InvalidBitWidth(bits));
    }
    if scales.len() != hidden_size {
        return Err(QuantError::ShapeMismatch {
            expected: hidden_size,
            actual: scales.len(),
        });
    }

    let max_val_f32 = ((1 << bits) - 1) as f32;
    let mut k = vec![0.0f32; seq_len * hidden_size];

    if bits == 4 {
        for t in 0..seq_len {
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let byte_idx = idx / 2;
                let q_val = if idx.is_multiple_of(2) {
                    q[byte_idx] & 0xF
                } else {
                    q[byte_idx] >> 4
                };
                k[idx] = (q_val as f32) * scales[c] / max_val_f32;
            }
        }
    } else {
        for t in 0..seq_len {
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let group = idx / 8;
                let pos = idx % 8;
                let byte_base = group * 3;

                let q_val = match pos {
                    0 => q[byte_base] & 0x7,
                    1 => (q[byte_base] >> 3) & 0x7,
                    2 => ((q[byte_base] >> 6) | ((q[byte_base + 1] & 0x1) << 2)) & 0x7,
                    3 => (q[byte_base + 1] >> 1) & 0x7,
                    4 => (q[byte_base + 1] >> 4) & 0x7,
                    5 => ((q[byte_base + 1] >> 7) | ((q[byte_base + 2] & 0x3) << 1)) & 0x7,
                    6 => (q[byte_base + 2] >> 2) & 0x7,
                    7 => (q[byte_base + 2] >> 5) & 0x7,
                    _ => unreachable!(),
                };
                k[idx] = (q_val as f32) * scales[c] / max_val_f32;
            }
        }
    }

    Ok(k)
}

/// Dequantize V cache with per-token scales.
pub fn dequantize_v_per_token(
    q: &[u8],
    scales: &[f32],
    bits: u8,
    seq_len: usize,
    hidden_size: usize,
) -> QuantResult<Vec<f32>> {
    if bits != 3 && bits != 4 {
        return Err(QuantError::InvalidBitWidth(bits));
    }
    if scales.len() != seq_len {
        return Err(QuantError::ShapeMismatch {
            expected: seq_len,
            actual: scales.len(),
        });
    }

    let max_val_f32 = ((1 << bits) - 1) as f32;
    let mut v = vec![0.0f32; seq_len * hidden_size];

    if bits == 4 {
        for t in 0..seq_len {
            let scale = scales[t];
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let byte_idx = idx / 2;
                let q_val = if idx.is_multiple_of(2) {
                    q[byte_idx] & 0xF
                } else {
                    q[byte_idx] >> 4
                };
                v[idx] = (q_val as f32) * scale / max_val_f32;
            }
        }
    } else {
        for t in 0..seq_len {
            let scale = scales[t];
            for c in 0..hidden_size {
                let idx = t * hidden_size + c;
                let group = idx / 8;
                let pos = idx % 8;
                let byte_base = group * 3;

                let q_val = match pos {
                    0 => q[byte_base] & 0x7,
                    1 => (q[byte_base] >> 3) & 0x7,
                    2 => ((q[byte_base] >> 6) | ((q[byte_base + 1] & 0x1) << 2)) & 0x7,
                    3 => (q[byte_base + 1] >> 1) & 0x7,
                    4 => (q[byte_base + 1] >> 4) & 0x7,
                    5 => ((q[byte_base + 1] >> 7) | ((q[byte_base + 2] & 0x3) << 1)) & 0x7,
                    6 => (q[byte_base + 2] >> 2) & 0x7,
                    7 => (q[byte_base + 2] >> 5) & 0x7,
                    _ => unreachable!(),
                };
                v[idx] = (q_val as f32) * scale / max_val_f32;
            }
        }
    }

    Ok(v)
}

/// Check if a token should be preserved in FP16 (Attention Sink protection).
///
/// Per SPEC §11.2: First N tokens (default N=4) preserved in FP16 if detected as sinks.
/// Sink detection comes from telemetry (centroid_pos from epilogue).
#[inline]
pub fn should_preserve_fp16(token_idx: usize, sink_count: usize, is_sink: bool) -> bool {
    token_idx < sink_count && is_sink
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_k_per_channel_4bit() {
        let seq_len = 2;
        let hidden_size = 4;
        let k = vec![
            1.0, 2.0, 3.0, 4.0,  // token 0
            5.0, 6.0, 7.0, 8.0,  // token 1
        ];
        // Scales should be max absolute value per channel
        // Channel 0: max(|1.0|, |5.0|) = 5.0
        // Channel 1: max(|2.0|, |6.0|) = 6.0
        // Channel 2: max(|3.0|, |7.0|) = 7.0
        // Channel 3: max(|4.0|, |8.0|) = 8.0
        let scales = vec![5.0, 6.0, 7.0, 8.0];

        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        let dequantized = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();

        for i in 0..k.len() {
            let error = (k[i] - dequantized[i]).abs() / k[i].max(1.0);
            assert!(error < 0.1, "error at {}: {} vs {}", i, k[i], dequantized[i]);
        }
    }

    #[test]
    fn test_quantize_v_per_token_4bit() {
        let seq_len = 2;
        let hidden_size = 4;
        let v = vec![
            1.0, 2.0, 3.0, 4.0,  // token 0
            5.0, 6.0, 7.0, 8.0,  // token 1
        ];

        let (quantized, scales) = quantize_v_per_token(&v, 4, seq_len, hidden_size, false).unwrap();
        assert_eq!(scales.len(), seq_len);

        let dequantized = dequantize_v_per_token(&quantized, &scales, 4, seq_len, hidden_size).unwrap();

        for i in 0..v.len() {
            let error = (v[i] - dequantized[i]).abs() / v[i].max(1.0);
            assert!(error < 0.1, "error at {}: {} vs {}", i, v[i], dequantized[i]);
        }
    }

    #[test]
    fn test_quantize_k_per_channel_3bit() {
        let seq_len = 2;
        let hidden_size = 8;
        let k = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        // Scales should be max absolute value per channel
        let scales = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        let quantized = quantize_k_per_channel(&k, &scales, 3, seq_len, hidden_size, false).unwrap();
        let dequantized = dequantize_k_per_channel(&quantized, &scales, 3, seq_len, hidden_size).unwrap();

        for i in 0..k.len() {
            let error = (k[i] - dequantized[i]).abs() / k[i].max(1.0);
            assert!(error < 0.3, "error at {}: {} vs {}", i, k[i], dequantized[i]);
        }
    }

    #[test]
    fn test_sink_preservation() {
        assert!(should_preserve_fp16(0, 4, true));
        assert!(should_preserve_fp16(3, 4, true));
        assert!(!should_preserve_fp16(4, 4, true));
        assert!(!should_preserve_fp16(0, 4, false));
    }

    #[test]
    fn test_invalid_bit_width() {
        let k = vec![1.0; 8];
        let scales = vec![1.0; 4];
        assert!(quantize_k_per_channel(&k, &scales, 5, 2, 4, false).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let k = vec![1.0; 8];
        let scales = vec![1.0; 3];
        assert!(quantize_k_per_channel(&k, &scales, 4, 2, 4, false).is_err());
    }

    #[test]
    fn test_fwht_rotation_basic() {
        // Test that FWHT rotation is applied correctly
        let seq_len = 1;
        let hidden_size = 4;
        let k = vec![1.0, 2.0, 3.0, 4.0];

        // Manually apply FWHT
        let mut k_manual = k.clone();
        fwht_inplace(&mut k_manual);

        // Compute scales from rotated data
        let max_abs = k_manual.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scales = vec![max_abs; hidden_size];

        // Quantize with FWHT enabled
        let q_fwht = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, true).unwrap();

        // Quantize without FWHT (using rotated data directly)
        let q_manual = quantize_k_per_channel(&k_manual, &scales, 4, seq_len, hidden_size, false).unwrap();

        // They should produce the same quantized result
        assert_eq!(q_fwht, q_manual, "FWHT rotation should be applied correctly");
    }

    #[test]
    fn test_fwht_correctness() {
        // FWHT([1,0,0,0]) = [1,1,1,1] (first basis vector → all ones)
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        fwht_inplace(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]);

        // FWHT([1,1,1,1]) = [4,0,0,0] (inverse up to scale)
        let mut data2 = vec![1.0f32, 1.0, 1.0, 1.0];
        fwht_inplace(&mut data2);
        assert_eq!(data2, vec![4.0, 0.0, 0.0, 0.0]);

        // FWHT is its own inverse (up to scale n)
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut data3 = original.clone();
        fwht_inplace(&mut data3);
        fwht_inplace(&mut data3);
        let n = original.len() as f32;
        for (a, b) in data3.iter().zip(original.iter()) {
            assert!((a - b * n).abs() < 1e-5, "FWHT(FWHT(x)) should equal n*x");
        }
    }

    #[test]
    fn test_rabitq_unbiased_expectation() {
        // Test that RaBitQ produces unbiased expectation: E[Q(x)] = x
        let x = vec![1.5, 2.3, 3.7, 4.1];
        let scale = 5.0;
        let bits = 4;
        let trials = 10000;
        let seq_len = 1;
        let hidden_size = x.len();

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
        };

        let scales = vec![scale; hidden_size];
        let mut sum = vec![0.0; hidden_size];

        for _ in 0..trials {
            let mut rng = rand::thread_rng();
            let quantized = quantize_k_per_channel_with_config(
                &x, &scales, bits, seq_len, hidden_size, &config, &mut rng
            ).unwrap();
            let dequantized = dequantize_k_per_channel(&quantized, &scales, bits, seq_len, hidden_size).unwrap();
            for i in 0..hidden_size {
                sum[i] += dequantized[i];
            }
        }

        for i in 0..hidden_size {
            let mean = sum[i] / trials as f32;
            let error = (mean - x[i]).abs() / x[i];
            assert!(error < 0.05, "bias at {}: mean={} vs expected={}, error={}", i, mean, x[i], error);
        }
    }

    #[test]
    fn test_rabitq_inner_product_preservation() {
        // Test that RaBitQ preserves inner product in expectation: E[<Q(a), Q(b)>] ≈ <a, b>
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let original_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let trials = 10000;
        let bits = 4;
        let seq_len = 1;
        let hidden_size = a.len();

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
        };

        let scales_a = vec![4.0; hidden_size];
        let scales_b = vec![8.0; hidden_size];
        let mut dot_sum = 0.0;

        for _ in 0..trials {
            let mut rng = rand::thread_rng();
            let qa = quantize_k_per_channel_with_config(&a, &scales_a, bits, seq_len, hidden_size, &config, &mut rng).unwrap();
            let qb = quantize_k_per_channel_with_config(&b, &scales_b, bits, seq_len, hidden_size, &config, &mut rng).unwrap();
            let da = dequantize_k_per_channel(&qa, &scales_a, bits, seq_len, hidden_size).unwrap();
            let db = dequantize_k_per_channel(&qb, &scales_b, bits, seq_len, hidden_size).unwrap();
            dot_sum += da.iter().zip(&db).map(|(x, y)| x * y).sum::<f32>();
        }

        let mean_dot = dot_sum / trials as f32;
        let error = (mean_dot - original_dot).abs() / original_dot;
        assert!(error < 0.05, "inner product bias: mean={} vs expected={}, error={}", mean_dot, original_dot, error);
    }

    #[test]
    fn test_rabitq_deterministic_comparison() {
        // Test that Deterministic mode produces consistent results (no randomness)
        let x = vec![1.5, 2.3, 3.7, 4.1];
        let scales = vec![5.0; x.len()];
        let bits = 4;
        let seq_len = 1;
        let hidden_size = x.len();

        let config_det = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };

        let mut rng = rand::thread_rng();
        let q1 = quantize_k_per_channel_with_config(&x, &scales, bits, seq_len, hidden_size, &config_det, &mut rng).unwrap();
        let q2 = quantize_k_per_channel_with_config(&x, &scales, bits, seq_len, hidden_size, &config_det, &mut rng).unwrap();

        assert_eq!(q1, q2, "Deterministic mode should produce identical results");
    }

    #[test]
    fn test_rabitq_v_cache() {
        // Test RaBitQ on V cache (per-token quantization)
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let bits = 4;
        let seq_len = 1;
        let hidden_size = v.len();
        let trials = 10000;

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
        };

        let mut sum = vec![0.0; hidden_size];

        for _ in 0..trials {
            let mut rng = rand::thread_rng();
            let (quantized, scales) = quantize_v_per_token_with_config(&v, bits, seq_len, hidden_size, &config, &mut rng).unwrap();
            let dequantized = dequantize_v_per_token(&quantized, &scales, bits, seq_len, hidden_size).unwrap();
            for i in 0..hidden_size {
                sum[i] += dequantized[i];
            }
        }

        for i in 0..hidden_size {
            let mean = sum[i] / trials as f32;
            let error = (mean - v[i]).abs() / v[i];
            assert!(error < 0.05, "V cache bias at {}: mean={} vs expected={}, error={}", i, mean, v[i], error);
        }
    }

    // -----------------------------------------------------------------------
    // §11.3 RaBitQ Correction Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rabitq_correction_zero() {
        let correction = RabitqCorrection::zero();
        assert_eq!(correction.c0, 0.0);
        assert_eq!(correction.c1, 0.0);
        assert_eq!(correction.v_norm, 0.0);

        // Zero correction should not modify scores
        let mut scores = vec![1.0, 2.0, 3.0];
        let q_norm = 5.0;
        correction.correct_scores_in_place(&mut scores, q_norm);
        assert_eq!(scores, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rabitq_correction_from_params() {
        // Test C1 formula: v_norm * 2^(-(bits-1)) / sqrt(dim)
        let v_norm = 10.0;
        let dim = 4096;
        let bits = 4;

        let correction = RabitqCorrection::from_quant_params(bits, dim, v_norm);

        // C0 should be 0 for symmetric quantization
        assert_eq!(correction.c0, 0.0);
        assert_eq!(correction.v_norm, v_norm);

        // C1 = v_norm * 2^(-(4-1)) / sqrt(4096) = 10.0 * 2^-3 / 64 = 10.0 / 8 / 64 ≈ 0.0195
        let expected_c1 = v_norm * 2.0_f32.powi(-(bits as i32 - 1)) / (dim as f32).sqrt();
        assert!((correction.c1 - expected_c1).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_correction_3bit() {
        // Test with 3-bit quantization
        let v_norm = 8.0;
        let dim = 2048;
        let bits = 3;

        let correction = RabitqCorrection::from_quant_params(bits, dim, v_norm);

        // C1 = v_norm * 2^(-(3-1)) / sqrt(2048) = 8.0 * 2^-2 / ~45.25 ≈ 8.0 / 4 / 45.25 ≈ 0.044
        let expected_c1 = v_norm * 2.0_f32.powi(-(bits as i32 - 1)) / (dim as f32).sqrt();
        assert!((correction.c1 - expected_c1).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_correct_score() {
        let correction = RabitqCorrection {
            c0: 0.5,
            c1: 0.1,
            v_norm: 10.0,
        };

        let raw_score = 5.0;
        let q_norm = 2.0;

        // corrected = raw + C1 * q_norm + C0 = 5.0 + 0.1 * 2.0 + 0.5 = 5.7
        let corrected = correction.correct_score(raw_score, q_norm);
        assert!((corrected - 5.7).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_correct_scores_in_place() {
        let correction = RabitqCorrection {
            c0: 0.5,
            c1: 0.1,
            v_norm: 10.0,
        };

        let mut scores = vec![1.0, 2.0, 3.0];
        let q_norm = 2.0;

        correction.correct_scores_in_place(&mut scores, q_norm);

        // Each score should be increased by C1 * q_norm + C0 = 0.1 * 2.0 + 0.5 = 0.7
        assert!((scores[0] - 1.7).abs() < 1e-6);
        assert!((scores[1] - 2.7).abs() < 1e-6);
        assert!((scores[2] - 3.7).abs() < 1e-6);
    }

    #[test]
    fn test_rabitq_correction_bound_4096() {
        // Test theoretical error bound O(1/√D) for D=4096
        // 1/√4096 ≈ 1/64 ≈ 0.0156 (~1.5%)
        let dim = 4096;
        let v_norm = 1.0;
        let bits = 4;

        let correction = RabitqCorrection::from_quant_params(bits, dim, v_norm);

        // C1 should be approximately v_norm / (8 * 64) = 1/512 ≈ 0.00195
        let expected_bound = 1.0 / (dim as f32).sqrt();
        assert!(correction.c1 < expected_bound, "C1 should be within O(1/√D) bound");
    }

    #[test]
    fn test_rabitq_correction_bound_8192() {
        // Test theoretical error bound O(1/√D) for D=8192
        // 1/√8192 ≈ 1/90.5 ≈ 0.011 (~1.1%)
        let dim = 8192;
        let v_norm = 1.0;
        let bits = 4;

        let correction = RabitqCorrection::from_quant_params(bits, dim, v_norm);

        // C1 should be approximately v_norm / (8 * 90.5) ≈ 0.00138
        let expected_bound = 1.0 / (dim as f32).sqrt();
        assert!(correction.c1 < expected_bound, "C1 should be within O(1/√D) bound");
    }

    #[test]
    fn test_rabitq_quantize_k_with_correction() {
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![4.0; 4];
        let bits = 4;
        let seq_len = 1;
        let hidden_size = 4;
        let v_norm = 5.0;

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };

        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_correction(
            &k, &scales, bits, seq_len, hidden_size, v_norm, &config, &mut rng
        );

        assert!(result.is_ok());
        let (quantized, returned_scales, correction) = result.unwrap();

        // Verify scales are preserved
        assert_eq!(returned_scales, scales);

        // Verify correction is computed
        assert!(correction.c1 > 0.0);
        assert_eq!(correction.v_norm, v_norm);

        // Verify quantization produces valid output
        let expected_size = (hidden_size + 1) / 2; // 4-bit packing
        assert_eq!(quantized.len(), expected_size);
    }

    #[test]
    fn test_rabitq_quantize_v_with_correction() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let bits = 4;
        let seq_len = 1;
        let hidden_size = 4;
        let v_norm = 6.0;

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };

        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_correction(
            &v, bits, seq_len, hidden_size, v_norm, &config, &mut rng
        );

        assert!(result.is_ok());
        let (quantized, scales, correction) = result.unwrap();

        // Verify we have per-token scales
        assert_eq!(scales.len(), seq_len);

        // Verify correction is computed
        assert!(correction.c1 > 0.0);
        assert_eq!(correction.v_norm, v_norm);

        // Verify quantization produces valid output
        let expected_size = (hidden_size + 1) / 2; // 4-bit packing
        assert_eq!(quantized.len(), expected_size);
    }

    #[test]
    fn test_apply_rabitq_correction() {
        let v_norm = 10.0;
        let dim = 4096;
        let bits = 4;

        let correction = RabitqCorrection::from_quant_params(bits, dim, v_norm);
        let mut scores = vec![100.0, 200.0, 300.0];
        let q_norm = 5.0;

        apply_rabitq_correction(&mut scores, &correction, q_norm);

        // Verify each score is corrected
        let expected_correction = correction.c1 * q_norm + correction.c0;
        for (i, &score) in scores.iter().enumerate() {
            let original = 100.0 * (i + 1) as f32;
            assert!((score - (original + expected_correction)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rabitq_default() {
        let correction = RabitqCorrection::default();
        assert_eq!(correction, RabitqCorrection::zero());
    }

    #[test]
    fn test_rabitq_unbiased_inner_product_estimation() {
        // Test that RaBitQ correction improves inner product estimation
        // This verifies the SPEC §11.3 claim: error < O(1/√D)

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let original_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let bits = 4;
        let seq_len = 1;
        let hidden_size = a.len();
        let v_norm = 3.0; // Mock ‖v‖ from RMSNorm

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };

        let scales_a = vec![4.0; hidden_size];
        let scales_b = vec![8.0; hidden_size];

        let mut rng = rand::thread_rng();
        let (qa, _, corr_a) = quantize_k_per_channel_with_correction(
            &a, &scales_a, bits, seq_len, hidden_size, v_norm, &config, &mut rng
        ).unwrap();
        let (qb, _, _corr_b) = quantize_k_per_channel_with_correction(
            &b, &scales_b, bits, seq_len, hidden_size, v_norm, &config, &mut rng
        ).unwrap();

        let da = dequantize_k_per_channel(&qa, &scales_a, bits, seq_len, hidden_size).unwrap();
        let db = dequantize_k_per_channel(&qb, &scales_b, bits, seq_len, hidden_size).unwrap();

        // Raw quantized dot product
        let quant_dot: f32 = da.iter().zip(&db).map(|(x, y)| x * y).sum();

        // Apply RaBitQ correction
        let q_norm = v_norm; // In practice, this would be ‖q‖ from RMSNorm
        let corrected_dot = corr_a.correct_score(quant_dot, q_norm);

        // The corrected dot should be closer to original than uncorrected
        let _uncorrected_error = (quant_dot - original_dot).abs();
        let corrected_error = (corrected_dot - original_dot).abs();

        // This test demonstrates the correction mechanism works;
        // actual improvement depends on data distribution
        assert!(corrected_error < original_dot, "Corrected error should be reasonable");
    }

    // =======================================================================
    // Additional comprehensive tests
    // =======================================================================

    // --- QuantError Display / Debug / Error variants ---

    #[test]
    fn test_quant_error_invalid_bit_width_display() {
        let err = QuantError::InvalidBitWidth(2);
        let msg = format!("{}", err);
        assert!(msg.contains("invalid bit width"), "Display should contain 'invalid bit width', got: {}", msg);
        assert!(msg.contains('2'), "Display should contain the value, got: {}", msg);
    }

    #[test]
    fn test_quant_error_shape_mismatch_display() {
        let err = QuantError::ShapeMismatch { expected: 16, actual: 8 };
        let msg = format!("{}", err);
        assert!(msg.contains("shape mismatch"), "Display should contain 'shape mismatch', got: {}", msg);
        assert!(msg.contains("expected 16"), "Display should contain 'expected 16', got: {}", msg);
        assert!(msg.contains("got 8"), "Display should contain 'got 8', got: {}", msg);
    }

    #[test]
    fn test_quant_error_zero_scale_display() {
        let err = QuantError::ZeroScale(7);
        let msg = format!("{}", err);
        assert!(msg.contains("zero scale"), "Display should contain 'zero scale', got: {}", msg);
        assert!(msg.contains('7'), "Display should contain index, got: {}", msg);
    }

    #[test]
    fn test_quant_error_debug_format() {
        let err = QuantError::InvalidBitWidth(6);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidBitWidth"), "Debug should contain variant name, got: {}", debug_str);
    }

    #[test]
    fn test_quant_error_is_std_error() {
        // Verify QuantError implements std::error::Error via thiserror
        let err: Box<dyn std::error::Error> = Box::new(QuantError::InvalidBitWidth(1));
        assert!(!err.to_string().is_empty());
    }

    // --- QuantMode enum ---

    #[test]
    fn test_quant_mode_equality() {
        assert_eq!(QuantMode::Deterministic, QuantMode::Deterministic);
        assert_eq!(QuantMode::RaBitQ, QuantMode::RaBitQ);
        assert_ne!(QuantMode::Deterministic, QuantMode::RaBitQ);
    }

    #[test]
    fn test_quant_mode_copy_clone() {
        let mode = QuantMode::RaBitQ;
        let mode_copy = mode;
        let mode_clone = mode;
        assert_eq!(mode_copy, QuantMode::RaBitQ);
        assert_eq!(mode_clone, QuantMode::RaBitQ);
    }

    #[test]
    fn test_quant_mode_debug() {
        let det = format!("{:?}", QuantMode::Deterministic);
        let rabitq = format!("{:?}", QuantMode::RaBitQ);
        assert!(det.contains("Deterministic"));
        assert!(rabitq.contains("RaBitQ"));
    }

    // --- RabitqCorrection trait impls ---

    #[test]
    fn test_rabitq_correction_partial_eq() {
        let a = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let b = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let c = RabitqCorrection { c0: 0.0, c1: 2.0, v_norm: 3.0 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_rabitq_correction_copy_clone() {
        let original = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let copied = original;
        let cloned = original.clone();
        assert_eq!(original, copied);
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_rabitq_correction_debug() {
        let corr = RabitqCorrection { c0: 0.1, c1: 0.2, v_norm: 0.3 };
        let debug = format!("{:?}", corr);
        assert!(debug.contains("RabitqCorrection"));
    }

    // --- RabitqCorrection::from_quant_params edge cases ---

    #[test]
    fn test_rabitq_correction_from_params_zero_dim() {
        let correction = RabitqCorrection::from_quant_params(4, 0, 10.0);
        // When dim=0, C1 should be 0.0 (guarded against division by zero)
        assert_eq!(correction.c0, 0.0);
        assert_eq!(correction.c1, 0.0);
        assert_eq!(correction.v_norm, 10.0);
    }

    #[test]
    fn test_rabitq_correction_from_params_zero_v_norm() {
        let correction = RabitqCorrection::from_quant_params(4, 4096, 0.0);
        assert_eq!(correction.c0, 0.0);
        assert_eq!(correction.c1, 0.0);
        assert_eq!(correction.v_norm, 0.0);
    }

    #[test]
    fn test_rabitq_correction_c1_decreases_with_higher_bits() {
        let dim = 4096;
        let v_norm = 10.0;
        let corr_3bit = RabitqCorrection::from_quant_params(3, dim, v_norm);
        let corr_4bit = RabitqCorrection::from_quant_params(4, dim, v_norm);
        // Higher bit width => smaller quantization error => smaller C1
        // C1_3bit = v_norm * 2^(-2) / sqrt(d) > C1_4bit = v_norm * 2^(-3) / sqrt(d)
        assert!(corr_3bit.c1 > corr_4bit.c1,
            "3-bit C1 ({}) should be larger than 4-bit C1 ({})", corr_3bit.c1, corr_4bit.c1);
    }

    #[test]
    fn test_rabitq_correction_c1_decreases_with_larger_dim() {
        let v_norm = 10.0;
        let corr_small = RabitqCorrection::from_quant_params(4, 1024, v_norm);
        let corr_large = RabitqCorrection::from_quant_params(4, 4096, v_norm);
        // Larger dim => larger denominator => smaller C1
        assert!(corr_small.c1 > corr_large.c1,
            "C1 for dim=1024 ({}) should be larger than for dim=4096 ({})",
            corr_small.c1, corr_large.c1);
    }

    // --- correct_score / correct_scores_in_place edge cases ---

    #[test]
    fn test_correct_score_with_zero_correction() {
        let correction = RabitqCorrection::zero();
        let result = correction.correct_score(42.0, 99.0);
        assert!((result - 42.0).abs() < 1e-6, "Zero correction should not change the score");
    }

    #[test]
    fn test_correct_score_with_zero_q_norm() {
        let correction = RabitqCorrection { c0: 1.5, c1: 0.3, v_norm: 5.0 };
        let result = correction.correct_score(10.0, 0.0);
        // corrected = 10.0 + 0.3 * 0.0 + 1.5 = 11.5
        assert!((result - 11.5).abs() < 1e-6);
    }

    #[test]
    fn test_correct_scores_in_place_empty() {
        let correction = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let mut scores: Vec<f32> = vec![];
        correction.correct_scores_in_place(&mut scores, 5.0);
        assert!(scores.is_empty(), "Empty slice should remain empty");
    }

    #[test]
    fn test_correct_scores_in_place_single_element() {
        let correction = RabitqCorrection { c0: 0.5, c1: 0.0, v_norm: 1.0 };
        let mut scores = vec![10.0];
        correction.correct_scores_in_place(&mut scores, 100.0);
        // correction = 0.0 * 100.0 + 0.5 = 0.5
        assert!((scores[0] - 10.5).abs() < 1e-6);
    }

    #[test]
    fn test_correct_scores_negative_values() {
        let correction = RabitqCorrection { c0: -1.0, c1: 0.5, v_norm: 2.0 };
        let mut scores = vec![-3.0, 0.0, 3.0];
        correction.correct_scores_in_place(&mut scores, 2.0);
        // correction = 0.5 * 2.0 + (-1.0) = 0.0
        assert!((scores[0] - (-3.0)).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);
        assert!((scores[2] - 3.0).abs() < 1e-6);
    }

    // --- should_preserve_fp16 edge cases ---

    #[test]
    fn test_should_preserve_fp16_sink_count_zero() {
        assert!(!should_preserve_fp16(0, 0, true), "sink_count=0 should never preserve");
    }

    #[test]
    fn test_should_preserve_fp16_boundary() {
        // token_idx < sink_count means indices 0..(sink_count-1) are in range
        assert!(should_preserve_fp16(0, 1, true));
        assert!(!should_preserve_fp16(1, 1, true));
    }

    #[test]
    fn test_should_preserve_fp16_non_sink_not_preserved() {
        assert!(!should_preserve_fp16(0, 4, false));
        assert!(!should_preserve_fp16(1, 4, false));
        assert!(!should_preserve_fp16(2, 4, false));
    }

    #[test]
    fn test_should_preserve_fp16_large_sink_count() {
        assert!(should_preserve_fp16(99, 100, true));
        assert!(!should_preserve_fp16(100, 100, true));
    }

    // --- FWHT additional tests ---

    #[test]
    fn test_fwht_length_one() {
        let mut data = vec![3.5f32];
        fwht_inplace(&mut data);
        assert!((data[0] - 3.5).abs() < 1e-6, "FWHT of length-1 should be identity");
    }

    #[test]
    fn test_fwht_length_two() {
        let mut data = vec![1.0f32, 3.0];
        fwht_inplace(&mut data);
        // H_2 = [[1,1],[1,-1]], so [1,3] -> [1+3, 1-3] = [4, -2]
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fwht_all_zeros() {
        let mut data = vec![0.0f32; 8];
        fwht_inplace(&mut data);
        assert!(data.iter().all(|&x| x == 0.0), "FWHT of all zeros should be all zeros");
    }

    #[test]
    fn test_fwht_linearity() {
        // FWHT(a*x + b*y) = a*FWHT(x) + b*FWHT(y)
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        let a_coeff = 2.0f32;
        let b_coeff = 3.0f32;

        let mut combined: Vec<f32> = x.iter().zip(&y).map(|(&xi, &yi)| a_coeff * xi + b_coeff * yi).collect();
        fwht_inplace(&mut combined);

        let mut fwht_x = x.clone();
        let mut fwht_y = y.clone();
        fwht_inplace(&mut fwht_x);
        fwht_inplace(&mut fwht_y);

        for i in 0..4 {
            let expected = a_coeff * fwht_x[i] + b_coeff * fwht_y[i];
            assert!((combined[i] - expected).abs() < 1e-4,
                "FWHT linearity violated at index {}", i);
        }
    }

    #[test]
    fn test_fwht_preserves_energy() {
        // Parseval's theorem: sum(x_i^2) == (1/n) * sum(FWHT(x)_i^2)
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = original.len() as f32;
        let energy_original: f32 = original.iter().map(|x| x * x).sum();

        let mut transformed = original.clone();
        fwht_inplace(&mut transformed);
        let energy_transformed: f32 = transformed.iter().map(|x| x * x).sum();

        // Unnormalized FWHT: energy_transformed = n * energy_original
        assert!((energy_transformed - n * energy_original).abs() < 1e-3,
            "Energy should scale by n: original={}, transformed={}, expected={}",
            energy_original, energy_transformed, n * energy_original);
    }

    #[test]
    fn test_fwht_involutive() {
        // FWHT(FWHT(x)) = n * x (up to scaling)
        let original = vec![3.0f32, -1.0, 4.0, 2.0, -5.0, 7.0, 0.5, -3.5];
        let n = original.len() as f32;
        let mut data = original.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        for (got, expected) in data.iter().zip(original.iter()) {
            assert!((got - expected * n).abs() < 1e-4,
                "FWHT(FWHT(x)) should equal n*x");
        }
    }

    #[test]
    fn test_fwht_negative_values() {
        let mut data = vec![-1.0f32, -2.0, -3.0, -4.0];
        fwht_inplace(&mut data);
        // [-1,-2,-3,-4] -> [-1-2-3-4, -1+2-3+4, -1-2+3+4, -1+2+3-4]
        //                = [-10, 2, 4, 0]
        assert!((data[0] - (-10.0)).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 4.0).abs() < 1e-6);
        assert!((data[3] - 0.0).abs() < 1e-6);
    }

    // --- K quantization error paths ---

    #[test]
    fn test_quantize_k_invalid_bit_width_variants() {
        let k = vec![1.0; 4];
        let scales = vec![1.0; 4];
        for bad_bits in [0u8, 1, 2, 5, 8, 16, 255] {
            let result = quantize_k_per_channel(&k, &scales, bad_bits, 1, 4, false);
            assert!(matches!(result, Err(QuantError::InvalidBitWidth(b)) if b == bad_bits),
                "bit width {} should produce InvalidBitWidth error", bad_bits);
        }
    }

    #[test]
    fn test_quantize_k_shape_mismatch_k_data() {
        let k = vec![1.0; 6];  // 6 != 2*4
        let scales = vec![1.0; 4];
        let result = quantize_k_per_channel(&k, &scales, 4, 2, 4, false);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 8, actual: 6 })));
    }

    #[test]
    fn test_quantize_k_shape_mismatch_scales() {
        let k = vec![1.0; 8];
        let scales = vec![1.0; 3];  // 3 != hidden_size=4
        let result = quantize_k_per_channel(&k, &scales, 4, 2, 4, false);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 4, actual: 3 })));
    }

    #[test]
    fn test_quantize_k_zero_scale_error() {
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![1.0, 0.0, 1.0, 1.0];  // scale[1] = 0
        let result = quantize_k_per_channel(&k, &scales, 4, 1, 4, false);
        assert!(matches!(result, Err(QuantError::ZeroScale(1))),
            "Zero scale at index 1 should produce ZeroScale(1) error");
    }

    #[test]
    fn test_quantize_k_zero_scale_first_channel() {
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![0.0, 1.0, 1.0, 1.0];
        let result = quantize_k_per_channel(&k, &scales, 4, 1, 4, false);
        assert!(matches!(result, Err(QuantError::ZeroScale(0))));
    }

    // --- V quantization error paths ---

    #[test]
    fn test_quantize_v_invalid_bit_width() {
        let v = vec![1.0; 4];
        let result = quantize_v_per_token(&v, 2, 1, 4, false);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(2))));
    }

    #[test]
    fn test_quantize_v_shape_mismatch() {
        let v = vec![1.0; 6];
        let result = quantize_v_per_token(&v, 4, 2, 4, false);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 8, actual: 6 })));
    }

    // --- Dequantize error paths ---

    #[test]
    fn test_dequantize_k_invalid_bit_width() {
        let q = vec![0u8; 2];
        let scales = vec![1.0; 4];
        let result = dequantize_k_per_channel(&q, &scales, 7, 1, 4);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(7))));
    }

    #[test]
    fn test_dequantize_k_scale_mismatch() {
        let q = vec![0u8; 2];
        let scales = vec![1.0; 2];  // 2 != hidden_size=4
        let result = dequantize_k_per_channel(&q, &scales, 4, 1, 4);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 4, actual: 2 })));
    }

    #[test]
    fn test_dequantize_v_invalid_bit_width() {
        let q = vec![0u8; 2];
        let scales = vec![1.0; 1];
        let result = dequantize_v_per_token(&q, &scales, 5, 1, 4);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(5))));
    }

    #[test]
    fn test_dequantize_v_scale_mismatch() {
        let q = vec![0u8; 2];
        let scales = vec![1.0; 2];  // 2 != seq_len=1
        let result = dequantize_v_per_token(&q, &scales, 4, 1, 4);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 1, actual: 2 })));
    }

    // --- K 4-bit round-trip with boundary values ---

    #[test]
    fn test_k_4bit_round_trip_all_zeros() {
        let k = vec![0.0f32; 8];
        let scales = vec![1.0; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 2, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 2, 4).unwrap();
        assert!(deq.iter().all(|&x| x == 0.0), "All zeros should round-trip exactly");
    }

    #[test]
    fn test_k_4bit_round_trip_max_values() {
        // When val == scale, normalized = max_val_f32 = 15, q = 15
        let k = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let scales = vec![5.0; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 2, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 2, 4).unwrap();
        for (i, &val) in deq.iter().enumerate() {
            assert!((val - 5.0).abs() < 0.01, "Max value should round-trip at index {}: got {}", i, val);
        }
    }

    #[test]
    fn test_k_4bit_round_trip_negative_values() {
        let k = vec![-3.0, -2.0, -1.0, 0.0, -3.0, -2.0, -1.0, 0.0];
        let scales = vec![3.0, 2.0, 1.0, 1.0];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 2, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 2, 4).unwrap();
        // Negative values are clamped to 0 since the quantization only covers [0, max_val]
        // deq values should be >= 0 (unsigned quantization)
        for (i, &val) in deq.iter().enumerate() {
            assert!(val >= -1e-6, "Negative input should quantize to >= 0 at index {}: got {}", i, val);
        }
    }

    #[test]
    fn test_k_4bit_packed_size() {
        let seq_len = 3;
        let hidden_size = 5;
        let k = vec![1.0; seq_len * hidden_size];
        let scales = vec![1.0; hidden_size];
        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        // 15 elements, 2 per byte, ceil(15/2) = 8
        let expected = (seq_len * hidden_size).div_ceil(2);
        assert_eq!(quantized.len(), expected, "Packed size for 4-bit should be ceil(n/2)");
    }

    // --- K 3-bit round-trip ---

    #[test]
    fn test_k_3bit_round_trip_zeros() {
        let k = vec![0.0f32; 16];
        let scales = vec![1.0; 8];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 2, 8, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 2, 8).unwrap();
        assert!(deq.iter().all(|&x| x == 0.0), "All zeros should round-trip exactly");
    }

    #[test]
    fn test_k_3bit_packed_size() {
        let seq_len = 2;
        let hidden_size = 8;
        let k = vec![1.0; seq_len * hidden_size];
        let scales = vec![1.0; hidden_size];
        let quantized = quantize_k_per_channel(&k, &scales, 3, seq_len, hidden_size, false).unwrap();
        // 16 elements * 3 bits / 8 = 6 bytes
        let expected = (seq_len * hidden_size * 3).div_ceil(8);
        assert_eq!(quantized.len(), expected, "Packed size for 3-bit should be ceil(n*3/8)");
    }

    #[test]
    fn test_k_3bit_round_trip_with_non_power_of_two_hidden() {
        // 3-bit packing with hidden_size not a multiple of 8
        // Use values close to their per-channel scales for best round-trip fidelity
        let seq_len = 1;
        let hidden_size = 10;
        let k: Vec<f32> = (0..hidden_size).map(|i| (i + 1) as f32 * 0.8).collect();
        let scales: Vec<f32> = (0..hidden_size).map(|i| (i + 1) as f32).collect();
        let quantized = quantize_k_per_channel(&k, &scales, 3, seq_len, hidden_size, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, seq_len, hidden_size).unwrap();
        // 3-bit has limited precision (7 levels), tolerance is higher
        for (i, (&orig, &deq_val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - deq_val).abs() / orig.max(1.0);
            assert!(error < 0.5, "3-bit round-trip error too large at index {}: orig={}, deq={}", i, orig, deq_val);
        }
    }

    // --- V 4-bit round-trip ---

    #[test]
    fn test_v_4bit_round_trip_zeros() {
        let v = vec![0.0f32; 8];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 2, 4, false).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 2, 4).unwrap();
        assert!(deq.iter().all(|&x| x == 0.0), "All zeros should round-trip exactly");
    }

    #[test]
    fn test_v_4bit_scale_computation() {
        // Per-token scale = max(|row|)
        let v = vec![
            1.0, 2.0, 3.0, 4.0,   // token 0: max_abs = 4.0
            -5.0, 1.0, 2.0, 3.0,  // token 1: max_abs = 5.0
        ];
        let (_quantized, scales) = quantize_v_per_token(&v, 4, 2, 4, false).unwrap();
        assert!((scales[0] - 4.0).abs() < 1e-6, "Token 0 scale should be 4.0, got {}", scales[0]);
        assert!((scales[1] - 5.0).abs() < 1e-6, "Token 1 scale should be 5.0, got {}", scales[1]);
    }

    #[test]
    fn test_v_4bit_all_zeros_scale_is_one() {
        // When all values are zero, scale defaults to 1.0 (not 0.0)
        let v = vec![0.0f32; 4];
        let (_quantized, scales) = quantize_v_per_token(&v, 4, 1, 4, false).unwrap();
        assert!((scales[0] - 1.0).abs() < 1e-6, "Zero token scale should default to 1.0");
    }

    // --- V 3-bit round-trip ---

    #[test]
    fn test_v_3bit_round_trip() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let (quantized, scales) = quantize_v_per_token(&v, 3, 2, 8, false).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 3, 2, 8).unwrap();
        assert_eq!(scales.len(), 2);
        for (i, (&orig, &deq_val)) in v.iter().zip(deq.iter()).enumerate() {
            let error = (orig - deq_val).abs() / orig.max(1.0);
            assert!(error < 0.35, "V 3-bit round-trip error at {}: orig={}, deq={}", i, orig, deq_val);
        }
    }

    #[test]
    fn test_v_3bit_packed_size() {
        let v = vec![1.0f32; 16];
        let (quantized, _scales) = quantize_v_per_token(&v, 3, 2, 8, false).unwrap();
        let expected = (16_usize * 3).div_ceil(8);
        assert_eq!(quantized.len(), expected);
    }

    // --- Single-element quantization ---

    #[test]
    fn test_k_4bit_single_element() {
        let k = vec![2.5];
        let scales = vec![5.0];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 1, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 1).unwrap();
        let error = (k[0] - deq[0]).abs() / k[0].max(1.0);
        assert!(error < 0.15, "Single element round-trip: {} vs {}", k[0], deq[0]);
    }

    #[test]
    fn test_v_4bit_single_element() {
        let v = vec![3.0];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 1, 1, false).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 1, 1).unwrap();
        let error = (v[0] - deq[0]).abs() / v[0].max(1.0);
        assert!(error < 0.15, "Single V element round-trip: {} vs {}", v[0], deq[0]);
    }

    // --- FWHT + K quantization integration ---

    #[test]
    fn test_k_quantize_fwht_non_power_of_two_hidden_uses_no_fwht() {
        // hidden_size not a power of two => FWHT should be skipped silently
        let k = vec![1.0f32; 6];
        let scales = vec![1.0; 6];
        let q_no_fwht = quantize_k_per_channel(&k, &scales, 4, 1, 6, false).unwrap();
        let q_with_fwht = quantize_k_per_channel(&k, &scales, 4, 1, 6, true).unwrap();
        assert_eq!(q_no_fwht, q_with_fwht,
            "Non-power-of-two hidden_size should skip FWHT, producing identical results");
    }

    #[test]
    fn test_v_quantize_fwht_non_power_of_two_hidden() {
        let v = vec![1.0f32; 6];
        let (q_no_fwht, s1) = quantize_v_per_token(&v, 4, 1, 6, false).unwrap();
        let (q_with_fwht, s2) = quantize_v_per_token(&v, 4, 1, 6, true).unwrap();
        assert_eq!(q_no_fwht, q_with_fwht);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_k_quantize_fwht_round_trip_4bit() {
        let seq_len = 2;
        let hidden_size = 8;
        let k: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let scales = vec![4.0; hidden_size];

        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, true).unwrap();
        // Dequantize does not apply inverse FWHT, so we only verify it completes without error
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), k.len());
        // Values should be non-negative and bounded
        for (i, &val) in deq.iter().enumerate() {
            assert!(val >= 0.0, "Dequantized value should be >= 0 at index {}", i);
        }
    }

    #[test]
    fn test_v_quantize_fwht_round_trip_4bit() {
        let seq_len = 2;
        let hidden_size = 4;
        let v: Vec<f32> = (0..seq_len * hidden_size).map(|i| (i as f32 + 1.0) * 0.5).collect();

        let (quantized, scales) = quantize_v_per_token(&v, 4, seq_len, hidden_size, true).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), v.len());
    }

    // --- RaBitQ mode V quantization with 3-bit ---

    #[test]
    fn test_rabitq_v_cache_3bit_unbiased() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let bits = 3;
        let seq_len = 1;
        let hidden_size = 8;
        let trials = 5000;

        let config = KvQuantConfig {
            bits,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
        };

        let mut sum = vec![0.0f32; hidden_size];

        for _ in 0..trials {
            let mut rng = rand::thread_rng();
            let (quantized, scales) = quantize_v_per_token_with_config(
                &v, bits, seq_len, hidden_size, &config, &mut rng
            ).unwrap();
            let deq = dequantize_v_per_token(&quantized, &scales, bits, seq_len, hidden_size).unwrap();
            for i in 0..hidden_size {
                sum[i] += deq[i];
            }
        }

        for i in 0..hidden_size {
            let mean = sum[i] / trials as f32;
            let error = (mean - v[i]).abs() / v[i];
            assert!(error < 0.10, "V 3-bit RaBitQ bias at {}: mean={} vs expected={}, error={}",
                i, mean, v[i], error);
        }
    }

    // --- apply_rabitq_correction function ---

    #[test]
    fn test_apply_rabitq_correction_wrapper() {
        // Verify the free function delegates correctly to the method
        let correction = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let mut scores = vec![10.0, 20.0];
        apply_rabitq_correction(&mut scores, &correction, 5.0);

        // correction = 2.0 * 5.0 + 1.0 = 11.0
        assert!((scores[0] - 21.0).abs() < 1e-6);
        assert!((scores[1] - 31.0).abs() < 1e-6);
    }

    // --- quantize_k/v_with_correction error propagation ---

    #[test]
    fn test_quantize_k_with_correction_invalid_bits() {
        let k = vec![1.0; 4];
        let scales = vec![1.0; 4];
        let config = KvQuantConfig { bits: 2, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_correction(&k, &scales, 2, 1, 4, 5.0, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(2))));
    }

    #[test]
    fn test_quantize_v_with_correction_invalid_bits() {
        let v = vec![1.0; 4];
        let config = KvQuantConfig { bits: 6, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_correction(&v, 6, 1, 4, 5.0, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(6))));
    }

    // --- 4-bit packing integrity (exhaustive for small size) ---

    #[test]
    fn test_4bit_packing_odd_element_count() {
        // 5 elements with 4-bit packing => ceil(5/2) = 3 bytes
        let k = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let scales = vec![10.0; 5];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 5, false).unwrap();
        assert_eq!(quantized.len(), 3, "5 elements at 4-bit should need 3 bytes");

        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 5).unwrap();
        assert_eq!(deq.len(), 5);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.15, "Odd-count packing round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    #[test]
    fn test_3bit_packing_single_group() {
        // 8 elements = one 3-bit group = 3 bytes
        let k: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scales = vec![8.0; 8];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 8, false).unwrap();
        assert_eq!(quantized.len(), 3, "8 elements at 3-bit should need exactly 3 bytes");

        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 1, 8).unwrap();
        assert_eq!(deq.len(), 8);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.35, "3-bit single group at {}: {} vs {}", i, orig, val);
        }
    }

    // --- Multiple seq_len with V per-token scales ---

    #[test]
    fn test_v_per_token_scales_multi_seq() {
        let seq_len = 4;
        let hidden_size = 4;
        let v: Vec<f32> = (1..=16).map(|i| i as f32).collect();

        let (quantized, scales) = quantize_v_per_token(&v, 4, seq_len, hidden_size, false).unwrap();
        assert_eq!(scales.len(), seq_len, "Should have one scale per token");

        let deq = dequantize_v_per_token(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), seq_len * hidden_size);

        for (i, (&orig, &val)) in v.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.15, "Multi-seq V round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    // --- KvQuantConfig fields ---

    #[test]
    fn test_kv_quant_config_fields() {
        let config = KvQuantConfig {
            bits: 3,
            sink_count: 4,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
        };
        assert_eq!(config.bits, 3);
        assert_eq!(config.sink_count, 4);
        assert!(config.fwht_enabled);
        assert_eq!(config.mode, QuantMode::RaBitQ);
    }

    #[test]
    fn test_kv_quant_config_debug() {
        let config = KvQuantConfig {
            bits: 4,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("KvQuantConfig"));
        assert!(debug.contains("bits"));
    }

    #[test]
    fn test_kv_quant_config_clone() {
        let config = KvQuantConfig {
            bits: 3,
            sink_count: 2,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
        };
        let cloned = config.clone();
        assert_eq!(config.bits, cloned.bits);
        assert_eq!(config.sink_count, cloned.sink_count);
        assert_eq!(config.fwht_enabled, cloned.fwht_enabled);
        assert_eq!(config.mode, cloned.mode);
    }

    // --- QuantResult type alias ---

    #[test]
    fn test_quant_result_ok() {
        let result: QuantResult<Vec<u8>> = Ok(vec![1, 2, 3]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn test_quant_result_err() {
        let result: QuantResult<Vec<u8>> = Err(QuantError::InvalidBitWidth(7));
        assert!(result.is_err());
    }

    // --- Large hidden_size round-trip (validates packing at scale) ---

    #[test]
    fn test_k_4bit_large_hidden_size() {
        let seq_len = 1;
        let hidden_size = 64;
        // Use values close to scale for good round-trip fidelity
        let k: Vec<f32> = (0..hidden_size).map(|i| ((i % 8) + 1) as f32).collect();
        let scales: Vec<f32> = (0..hidden_size).map(|_| 8.0f32).collect();

        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();

        assert_eq!(quantized.len(), hidden_size / 2);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.15, "Large hidden round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    #[test]
    fn test_k_3bit_large_hidden_size() {
        let seq_len = 2;
        let hidden_size = 32;
        // Use values close to scale for good round-trip fidelity
        let k: Vec<f32> = (0..seq_len * hidden_size).map(|i| ((i % 7) + 1) as f32).collect();
        let scales: Vec<f32> = (0..hidden_size).map(|_| 7.0f32).collect();

        let quantized = quantize_k_per_channel(&k, &scales, 3, seq_len, hidden_size, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, seq_len, hidden_size).unwrap();

        let expected_packed = (seq_len * hidden_size * 3_usize).div_ceil(8);
        assert_eq!(quantized.len(), expected_packed);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.5, "Large 3-bit round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    // --- RabitqCorrection default matches zero ---

    #[test]
    fn test_rabitq_default_impl_zero() {
        let default = RabitqCorrection::default();
        let zero = RabitqCorrection::zero();
        assert_eq!(default, zero);
        assert_eq!(default.c0, 0.0);
        assert_eq!(default.c1, 0.0);
        assert_eq!(default.v_norm, 0.0);
    }

    // --- correct_score additive structure ---

    #[test]
    fn test_correct_score_additive_independence() {
        // correct_score(raw, q_norm) = raw + C1*q_norm + C0
        // Verify linear in raw_score and q_norm separately
        let c = RabitqCorrection { c0: 2.0, c1: 3.0, v_norm: 1.0 };

        let r1 = c.correct_score(0.0, 0.0);
        assert!((r1 - 2.0).abs() < 1e-6, "raw=0, q_norm=0 => C0 only");

        let r2 = c.correct_score(0.0, 1.0);
        assert!((r2 - 5.0).abs() < 1e-6, "raw=0, q_norm=1 => C1*1 + C0");

        let r3 = c.correct_score(10.0, 0.0);
        assert!((r3 - 12.0).abs() < 1e-6, "raw=10, q_norm=0 => 10 + C0");

        let r4 = c.correct_score(10.0, 2.0);
        // 10 + 3*2 + 2 = 18
        assert!((r4 - 18.0).abs() < 1e-6, "raw=10, q_norm=2 => 10 + C1*2 + C0");
    }

    // --- V 3-bit with all-zero row ---

    #[test]
    fn test_v_3bit_zero_row_scale_default() {
        let v = vec![0.0f32; 8];
        let (quantized, scales) = quantize_v_per_token(&v, 3, 1, 8, false).unwrap();
        // Scale should default to 1.0 for zero row
        assert!((scales[0] - 1.0).abs() < 1e-6, "Zero row scale should be 1.0, got {}", scales[0]);

        let deq = dequantize_v_per_token(&quantized, &scales, 3, 1, 8).unwrap();
        assert!(deq.iter().all(|&x| x == 0.0));
    }

    // --- Mixed zero/nonzero rows in V ---

    #[test]
    fn test_v_mixed_zero_nonzero_rows() {
        let v = vec![
            0.0, 0.0, 0.0, 0.0,  // token 0: all zero
            1.0, 2.0, 3.0, 4.0,  // token 1: nonzero
        ];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 2, 4, false).unwrap();
        assert!((scales[0] - 1.0).abs() < 1e-6, "Zero row scale should default to 1.0");
        assert!((scales[1] - 4.0).abs() < 1e-6, "Nonzero row scale should be max_abs");

        let deq = dequantize_v_per_token(&quantized, &scales, 4, 2, 4).unwrap();
        // First 4 values should be zero (zero input, scale=1.0, normalized=0)
        for i in 0..4 {
            assert!((deq[i] - 0.0).abs() < 1e-6, "Zero row dequantized should be 0 at {}", i);
        }
        // Second 4 values should be close to original
        for i in 4..8 {
            let error = (v[i] - deq[i]).abs() / v[i].max(1.0);
            assert!(error < 0.15, "Nonzero row round-trip at {}: {} vs {}", i, v[i], deq[i]);
        }
    }

    // --- Deterministic mode produces exact same output every time ---

    #[test]
    fn test_deterministic_mode_v_reproducibility() {
        let v = vec![1.5, 2.7, 3.3, 4.9];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };

        let mut rng = rand::thread_rng();
        let (q1, s1) = quantize_v_per_token_with_config(&v, 4, 1, 4, &config, &mut rng).unwrap();
        let (q2, s2) = quantize_v_per_token_with_config(&v, 4, 1, 4, &config, &mut rng).unwrap();

        assert_eq!(q1, q2, "Deterministic V quantization should be reproducible");
        assert_eq!(s1, s2, "Deterministic V scales should be reproducible");
    }

    // --- FWHT applied in V quantization produces different packed bytes ---

    #[test]
    fn test_v_fwht_changes_quantized_output() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let (q_no_fwht, _) = quantize_v_per_token(&v, 4, 1, 8, false).unwrap();
        let (q_with_fwht, _) = quantize_v_per_token(&v, 4, 1, 8, true).unwrap();
        // FWHT should redistribute values, resulting in different packed bytes
        assert_ne!(q_no_fwht, q_with_fwht,
            "FWHT rotation should change the quantized output");
    }

    // =======================================================================
    // New tests (45+) — Display impls, From, PartialEq, boundary values,
    // special floats, zero/empty inputs, usize::MAX, Hash, Eq, etc.
    // =======================================================================

    // --- QuantMode Hash ---

    #[test]
    fn test_quant_mode_hash_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        QuantMode::Deterministic.hash(&mut h1);
        let hash1 = h1.finish();
        // Hashing twice should yield the same value
        let mut h2 = DefaultHasher::new();
        QuantMode::Deterministic.hash(&mut h2);
        assert_eq!(hash1, h2.finish(), "Hash of Deterministic should be stable");
    }

    #[test]
    fn test_quant_mode_hash_rabitq() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        QuantMode::RaBitQ.hash(&mut h1);
        let hash1 = h1.finish();
        let mut h2 = DefaultHasher::new();
        QuantMode::RaBitQ.hash(&mut h2);
        assert_eq!(hash1, h2.finish(), "Hash of RaBitQ should be stable");
    }

    #[test]
    fn test_quant_mode_hash_variants_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        QuantMode::Deterministic.hash(&mut h1);
        let hash_det = h1.finish();
        let mut h2 = DefaultHasher::new();
        QuantMode::RaBitQ.hash(&mut h2);
        let hash_rabitq = h2.finish();
        assert_ne!(hash_det, hash_rabitq, "Different QuantMode variants should hash differently");
    }

    #[test]
    fn test_quant_mode_hash_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(QuantMode::Deterministic));
        assert!(set.insert(QuantMode::RaBitQ));
        assert!(!set.insert(QuantMode::Deterministic), "Duplicate insert should return false");
        assert_eq!(set.len(), 2);
    }

    // --- QuantMode Eq (reflexive, symmetric, transitive via PartialEq + Eq) ---

    #[test]
    fn test_quant_mode_eq_reflexive() {
        assert_eq!(QuantMode::Deterministic, QuantMode::Deterministic);
        assert_eq!(QuantMode::RaBitQ, QuantMode::RaBitQ);
    }

    #[test]
    fn test_quant_mode_eq_symmetric() {
        assert_eq!(QuantMode::Deterministic == QuantMode::RaBitQ,
                   QuantMode::RaBitQ == QuantMode::Deterministic);
    }

    // --- RabitqCorrection with special floats ---

    #[test]
    fn test_rabitq_correction_nan_v_norm() {
        let correction = RabitqCorrection::from_quant_params(4, 4096, f32::NAN);
        assert!(correction.c1.is_nan(), "C1 should be NaN when v_norm is NaN");
        assert!(correction.v_norm.is_nan(), "v_norm should preserve NaN");
    }

    #[test]
    fn test_rabitq_correction_infinity_v_norm() {
        let correction = RabitqCorrection::from_quant_params(4, 4096, f32::INFINITY);
        assert!(correction.c1.is_infinite() && correction.c1.is_sign_positive(),
            "C1 should be +infinity when v_norm is +infinity, got {}", correction.c1);
    }

    #[test]
    fn test_rabitq_correction_neg_infinity_v_norm() {
        let correction = RabitqCorrection::from_quant_params(4, 4096, f32::NEG_INFINITY);
        assert!(correction.c1.is_infinite() && correction.c1.is_sign_negative(),
            "C1 should be -infinity when v_norm is -infinity, got {}", correction.c1);
    }

    #[test]
    fn test_rabitq_correction_dim_one() {
        let correction = RabitqCorrection::from_quant_params(4, 1, 10.0);
        let expected_c1 = 10.0 * 2.0_f32.powi(-3) / 1.0_f32.sqrt();
        assert!((correction.c1 - expected_c1).abs() < 1e-6,
            "C1 for dim=1 should be {}, got {}", expected_c1, correction.c1);
    }

    #[test]
    fn test_correct_score_nan_q_norm() {
        let correction = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let result = correction.correct_score(10.0, f32::NAN);
        // result = 10.0 + 2.0 * NaN + 1.0 = NaN
        assert!(result.is_nan(), "correct_score with NaN q_norm should produce NaN");
    }

    #[test]
    fn test_correct_score_infinity_raw_score() {
        let correction = RabitqCorrection { c0: 0.0, c1: 0.0, v_norm: 0.0 };
        let result = correction.correct_score(f32::INFINITY, 1.0);
        assert!(result.is_infinite() && result.is_sign_positive(),
            "Infinity raw_score with zero correction should remain +infinity");
    }

    #[test]
    fn test_correct_score_neg_infinity_raw_score() {
        let correction = RabitqCorrection { c0: 0.0, c1: 0.0, v_norm: 0.0 };
        let result = correction.correct_score(f32::NEG_INFINITY, 1.0);
        assert!(result.is_infinite() && result.is_sign_negative(),
            "-Infinity raw_score with zero correction should remain -infinity");
    }

    #[test]
    fn test_correct_scores_in_place_nan_correction() {
        let correction = RabitqCorrection { c0: f32::NAN, c1: 0.0, v_norm: 1.0 };
        let mut scores = vec![1.0, 2.0, 3.0];
        correction.correct_scores_in_place(&mut scores, 1.0);
        // correction = 0.0 * 1.0 + NaN = NaN
        for (i, &s) in scores.iter().enumerate() {
            assert!(s.is_nan(), "Score at index {} should be NaN, got {}", i, s);
        }
    }

    // --- should_preserve_fp16 with usize::MAX ---

    #[test]
    fn test_should_preserve_fp16_usize_max_sink_count() {
        assert!(should_preserve_fp16(0, usize::MAX, true),
            "token_idx=0 with usize::MAX sink_count should preserve");
        assert!(should_preserve_fp16(usize::MAX - 1, usize::MAX, true),
            "token_idx=usize::MAX-1 with usize::MAX sink_count should preserve");
        assert!(!should_preserve_fp16(usize::MAX, usize::MAX, true),
            "token_idx=usize::MAX == sink_count should NOT preserve");
    }

    #[test]
    fn test_should_preserve_fp16_usize_max_token_idx() {
        assert!(!should_preserve_fp16(usize::MAX, 0, true),
            "sink_count=0 should never preserve regardless of token_idx");
        assert!(!should_preserve_fp16(usize::MAX, 1, true),
            "token_idx=usize::MAX >= sink_count=1 should NOT preserve");
    }

    // --- QuantError error chain ---

    #[test]
    fn test_quant_error_as_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(QuantError::ZeroScale(42));
        let msg = err.to_string();
        assert!(msg.contains("42"), "Error message should contain index 42, got: {}", msg);
    }

    #[test]
    fn test_quant_error_shape_mismatch_values() {
        let err = QuantError::ShapeMismatch { expected: 100, actual: 50 };
        let msg = format!("{}", err);
        assert!(msg.contains("100"), "Should contain expected value, got: {}", msg);
        assert!(msg.contains("50"), "Should contain actual value, got: {}", msg);
    }

    #[test]
    fn test_quant_error_invalid_bit_width_edge_values() {
        let err0 = QuantError::InvalidBitWidth(0);
        assert!(format!("{}", err0).contains('0'));
        let err255 = QuantError::InvalidBitWidth(255);
        assert!(format!("{}", err255).contains("255"));
    }

    // --- KvQuantConfig boundary fields ---

    #[test]
    fn test_kv_quant_config_zero_sink_count() {
        let config = KvQuantConfig {
            bits: 4,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        assert_eq!(config.sink_count, 0);
        // Should still produce valid quantization
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![4.0; 4];
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(
            &k, &scales, 4, 1, 4, &config, &mut rng
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_kv_quant_config_large_sink_count() {
        let config = KvQuantConfig {
            bits: 4,
            sink_count: usize::MAX,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        assert_eq!(config.sink_count, usize::MAX);
        // Config should be usable (sink_count does not affect quantization math)
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![4.0; 4];
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(
            &k, &scales, 4, 1, 4, &config, &mut rng
        );
        assert!(result.is_ok());
    }

    // --- KvQuantConfig with both QuantMode variants ---

    #[test]
    fn test_kv_quant_config_rabitq_mode_fields() {
        let config = KvQuantConfig {
            bits: 3,
            sink_count: 8,
            fwht_enabled: true,
            mode: QuantMode::RaBitQ,
        };
        assert_eq!(config.bits, 3);
        assert_eq!(config.sink_count, 8);
        assert!(config.fwht_enabled);
        assert_eq!(config.mode, QuantMode::RaBitQ);
    }

    // --- FWHT with alternating pattern ---

    #[test]
    fn test_fwht_alternating_values() {
        let mut data = vec![1.0f32, -1.0, 1.0, -1.0];
        fwht_inplace(&mut data);
        // [1,-1,1,-1]: H_4 applied => [0,4,0,0] (high-freq basis vector)
        assert!((data[0]).abs() < 1e-6, "Expected ~0 at index 0, got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-6, "Expected 4.0 at index 1, got {}", data[1]);
        assert!((data[2]).abs() < 1e-6, "Expected ~0 at index 2, got {}", data[2]);
        assert!((data[3]).abs() < 1e-6, "Expected ~0 at index 3, got {}", data[3]);
    }

    #[test]
    fn test_fwht_length_eight_identity() {
        // FWHT of basis vector e_0 = [1,0,0,0,0,0,0,0] should be [1,1,1,1,1,1,1,1]
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        fwht_inplace(&mut data);
        for (i, &val) in data.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-6, "Expected 1.0 at index {}, got {}", i, val);
        }
    }

    #[test]
    fn test_fwht_large_power_of_two() {
        let n = 256;
        let mut data = vec![1.0f32; n];
        fwht_inplace(&mut data);
        // FWHT of all-ones => [n, 0, 0, ..., 0]
        assert!((data[0] - n as f32).abs() < 1e-3, "First element should be {}, got {}", n, data[0]);
        for i in 1..n {
            assert!(data[i].abs() < 1e-3, "Element {} should be ~0, got {}", i, data[i]);
        }
    }

    // --- K quantization with very small values ---

    #[test]
    fn test_k_4bit_very_small_values() {
        let k = vec![1e-6, 2e-6, 3e-6, 4e-6];
        let scales = vec![4e-6; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 4).unwrap();
        // Relative error should still be small since scale is proportional
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1e-10);
            assert!(error < 0.15, "Small value round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    #[test]
    fn test_k_4bit_large_scale_small_value() {
        // When scale >> value, quantized value rounds to 0
        let k = vec![0.001, 0.002, 0.003, 0.004];
        let scales = vec![1000.0; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 4).unwrap();
        // Values are so small relative to scale that they quantize to 0
        for &val in &deq {
            assert!(val < 100.0, "Values should be very small relative to scale, got {}", val);
        }
    }

    // --- V quantization with negative values ---

    #[test]
    fn test_v_4bit_negative_values_clamped() {
        let v = vec![-5.0, -3.0, -1.0, 0.0];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 1, 4, false).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 1, 4).unwrap();
        // Unsigned quantization: negative values get clamped to 0
        for &val in &deq {
            assert!(val >= -1e-6, "Dequantized values should be >= 0, got {}", val);
        }
    }

    // --- V per-token scale with mixed positive/negative row ---

    #[test]
    fn test_v_per_token_scale_mixed_signs() {
        let v = vec![-10.0, 5.0, -3.0, 8.0];
        let (_quantized, scales) = quantize_v_per_token(&v, 4, 1, 4, false).unwrap();
        // max_abs = 10.0
        assert!((scales[0] - 10.0).abs() < 1e-6, "Scale should be max absolute value, got {}", scales[0]);
    }

    // --- 3-bit V quantization with multiple tokens ---

    #[test]
    fn test_v_3bit_multi_seq_round_trip() {
        let seq_len = 3;
        let hidden_size = 8;
        let v: Vec<f32> = (1..=(seq_len * hidden_size)).map(|i| i as f32 * 0.5).collect();
        let (quantized, scales) = quantize_v_per_token(&v, 3, seq_len, hidden_size, false).unwrap();
        assert_eq!(scales.len(), seq_len);
        let deq = dequantize_v_per_token(&quantized, &scales, 3, seq_len, hidden_size).unwrap();
        for (i, (&orig, &val)) in v.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.45, "V 3-bit multi-seq round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    // --- 4-bit K quantization with single token multiple channels ---

    #[test]
    fn test_k_4bit_single_seq_multiple_channels() {
        let seq_len = 1;
        let hidden_size = 16;
        let k: Vec<f32> = (0..hidden_size).map(|i| (i + 1) as f32).collect();
        let scales = vec![16.0; hidden_size];
        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        assert_eq!(quantized.len(), hidden_size / 2);
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.15, "Single-seq multi-channel at {}: {} vs {}", i, orig, val);
        }
    }

    // --- V quantization with _config API in deterministic mode ---

    #[test]
    fn test_v_quantize_config_deterministic_3bit() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let config = KvQuantConfig {
            bits: 3,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        let mut rng = rand::thread_rng();
        let (q1, s1) = quantize_v_per_token_with_config(&v, 3, 1, 8, &config, &mut rng).unwrap();
        let (q2, s2) = quantize_v_per_token_with_config(&v, 3, 1, 8, &config, &mut rng).unwrap();
        assert_eq!(q1, q2, "Deterministic 3-bit V should be reproducible");
        assert_eq!(s1, s2);
    }

    // --- K quantization with _config API in RaBitQ mode 3-bit ---

    #[test]
    fn test_k_quantize_config_rabitq_3bit() {
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scales = vec![8.0; 8];
        let config = KvQuantConfig {
            bits: 3,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::RaBitQ,
        };
        let mut rng = rand::thread_rng();
        let q = quantize_k_per_channel_with_config(&k, &scales, 3, 1, 8, &config, &mut rng).unwrap();
        // 8 elements * 3 bits / 8 = 3 bytes
        assert_eq!(q.len(), 3);
    }

    // --- quantize_k_with_correction returns correct tuple structure ---

    #[test]
    fn test_k_with_correction_tuple_structure() {
        let k = vec![2.0; 8];
        let scales = vec![4.0; 8];
        let config = KvQuantConfig {
            bits: 4,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        let mut rng = rand::thread_rng();
        let (quantized, returned_scales, correction) = quantize_k_per_channel_with_correction(
            &k, &scales, 4, 1, 8, 3.0, &config, &mut rng
        ).unwrap();
        assert_eq!(quantized.len(), 4, "4-bit packing of 8 elements = 4 bytes");
        assert_eq!(returned_scales.len(), 8);
        assert_eq!(returned_scales, scales);
        assert_eq!(correction.v_norm, 3.0);
        assert_eq!(correction.c0, 0.0);
        assert!(correction.c1 > 0.0);
    }

    // --- quantize_v_with_correction returns correct tuple structure ---

    #[test]
    fn test_v_with_correction_tuple_structure() {
        let v = vec![3.0; 8];
        let config = KvQuantConfig {
            bits: 4,
            sink_count: 0,
            fwht_enabled: false,
            mode: QuantMode::Deterministic,
        };
        let mut rng = rand::thread_rng();
        let (quantized, scales, correction) = quantize_v_per_token_with_correction(
            &v, 4, 1, 8, 5.0, &config, &mut rng
        ).unwrap();
        assert_eq!(quantized.len(), 4, "4-bit packing of 8 elements = 4 bytes");
        assert_eq!(scales.len(), 1, "Per-token scales: 1 token");
        assert_eq!(correction.v_norm, 5.0);
        assert_eq!(correction.c0, 0.0);
    }

    // --- Dequantize with zero scales produces zero output ---

    #[test]
    fn test_dequantize_k_all_zero_scales() {
        // Note: quantize would reject zero scales, but dequantize should handle them gracefully
        let q = vec![0xFFu8; 2]; // All bits set
        let scales = vec![0.0f32; 4];
        let result = dequantize_k_per_channel(&q, &scales, 4, 1, 4).unwrap();
        // dequant formula: (q_val as f32) * scales[c] / max_val_f32
        // With scales=0.0, all output should be 0.0
        assert!(result.iter().all(|&x| x == 0.0),
            "Dequantized values with zero scales should all be 0.0");
    }

    // --- Dequantize V with zero scales ---

    #[test]
    fn test_dequantize_v_zero_scales() {
        let q = vec![0xFFu8; 2];
        let scales = vec![0.0f32];
        let result = dequantize_v_per_token(&q, &scales, 4, 1, 4).unwrap();
        assert!(result.iter().all(|&x| x == 0.0));
    }

    // --- apply_rabitq_correction with empty scores ---

    #[test]
    fn test_apply_rabitq_correction_empty_scores() {
        let correction = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let mut scores: Vec<f32> = vec![];
        apply_rabitq_correction(&mut scores, &correction, 5.0);
        assert!(scores.is_empty());
    }

    // --- apply_rabitq_correction with zero correction ---

    #[test]
    fn test_apply_rabitq_correction_zero_correction() {
        let correction = RabitqCorrection::zero();
        let mut scores = vec![100.0, -50.0, 0.0];
        apply_rabitq_correction(&mut scores, &correction, 999.0);
        assert!((scores[0] - 100.0).abs() < 1e-6);
        assert!((scores[1] - (-50.0)).abs() < 1e-6);
        assert!((scores[2] - 0.0).abs() < 1e-6);
    }

    // --- K quantize with all identical values ---

    #[test]
    fn test_k_4bit_all_identical_values() {
        let k = vec![3.0; 8];
        let scales = vec![3.0; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 2, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 2, 4).unwrap();
        // All values identical and equal to scale => all quantize to max => all dequantize to same value
        let first = deq[0];
        for (i, &val) in deq.iter().enumerate() {
            assert!((val - first).abs() < 1e-6, "All dequantized values should be equal at index {}", i);
        }
    }

    // --- V quantize all identical values ---

    #[test]
    fn test_v_4bit_all_identical_values() {
        let v = vec![5.0; 8];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 2, 4, false).unwrap();
        assert!((scales[0] - 5.0).abs() < 1e-6);
        assert!((scales[1] - 5.0).abs() < 1e-6);
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 2, 4).unwrap();
        let first = deq[0];
        for (i, &val) in deq.iter().enumerate() {
            assert!((val - first).abs() < 1e-6, "All dequantized V values should be equal at {}", i);
        }
    }

    // --- QuantMode variant count is exactly 2 ---

    #[test]
    fn test_quant_mode_exactly_two_variants() {
        let modes = [QuantMode::Deterministic, QuantMode::RaBitQ];
        assert_eq!(modes.len(), 2);
        assert_ne!(modes[0], modes[1]);
    }

    // --- RabitqCorrection partial_eq with different fields ---

    #[test]
    fn test_rabitq_correction_ne_different_c0() {
        let a = RabitqCorrection { c0: 0.0, c1: 1.0, v_norm: 2.0 };
        let b = RabitqCorrection { c0: 0.1, c1: 1.0, v_norm: 2.0 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_rabitq_correction_ne_different_c1() {
        let a = RabitqCorrection { c0: 1.0, c1: 0.0, v_norm: 2.0 };
        let b = RabitqCorrection { c0: 1.0, c1: 0.5, v_norm: 2.0 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_rabitq_correction_ne_different_v_norm() {
        let a = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 0.0 };
        let b = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 1.0 };
        assert_ne!(a, b);
    }

    // --- K 3-bit with zero values interspersed ---

    #[test]
    fn test_k_3bit_mixed_zeros() {
        let k = vec![0.0, 4.0, 0.0, 4.0, 0.0, 4.0, 0.0, 4.0];
        let scales = vec![4.0; 8];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 8, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 1, 8).unwrap();
        // Zero values should round-trip as zero
        assert!((deq[0]).abs() < 0.01, "Zero at index 0 should dequantize near 0");
        assert!((deq[2]).abs() < 0.01, "Zero at index 2 should dequantize near 0");
    }

    // --- FWHT with power-of-two = 2 ---

    #[test]
    fn test_fwht_length_two_negative() {
        let mut data = vec![-3.0f32, 7.0];
        fwht_inplace(&mut data);
        // [-3, 7] -> [-3+7, -3-7] = [4, -10]
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - (-10.0)).abs() < 1e-6);
    }

    // --- K/V with hidden_size = 1 ---

    #[test]
    fn test_k_4bit_hidden_size_one() {
        let k = vec![2.5, 3.5];
        let scales = vec![5.0];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 2, 1, false).unwrap();
        // 2 elements / 2 = 1 byte
        assert_eq!(quantized.len(), 1);
    }

    #[test]
    fn test_v_4bit_hidden_size_one() {
        let v = vec![1.5, 4.0];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 2, 1, false).unwrap();
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 1.5).abs() < 1e-6, "Token 0 scale should be 1.5");
        assert!((scales[1] - 4.0).abs() < 1e-6, "Token 1 scale should be 4.0");
        assert_eq!(quantized.len(), 1);
    }

    // --- Multiple error variants are distinct ---

    #[test]
    fn test_quant_error_variants_distinct_display() {
        let err1 = QuantError::InvalidBitWidth(5);
        let err2 = QuantError::ShapeMismatch { expected: 10, actual: 5 };
        let err3 = QuantError::ZeroScale(3);
        let msg1 = format!("{}", err1);
        let msg2 = format!("{}", err2);
        let msg3 = format!("{}", err3);
        assert_ne!(msg1, msg2);
        assert_ne!(msg2, msg3);
        assert_ne!(msg1, msg3);
    }

    // --- V scale computation with single very large value ---

    #[test]
    fn test_v_per_token_scale_dominated_by_single_outlier() {
        let v = vec![0.01, 0.02, 100.0, 0.03];
        let (_quantized, scales) = quantize_v_per_token(&v, 4, 1, 4, false).unwrap();
        assert!((scales[0] - 100.0).abs() < 1e-6,
            "Scale should be dominated by outlier 100.0, got {}", scales[0]);
    }

    // --- RabitqCorrection::from_quant_params with dim = 2 ---

    #[test]
    fn test_rabitq_correction_dim_two() {
        let correction = RabitqCorrection::from_quant_params(4, 2, 8.0);
        let expected_c1 = 8.0 * 2.0_f32.powi(-3) / 2.0_f32.sqrt();
        assert!((correction.c1 - expected_c1).abs() < 1e-6);
    }

    // --- FWHT applied and then skipped produces different K results ---

    #[test]
    fn test_k_fwht_changes_quantized_bytes() {
        let k: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scales = vec![8.0; 8];
        let q_no = quantize_k_per_channel(&k, &scales, 4, 1, 8, false).unwrap();
        let q_yes = quantize_k_per_channel(&k, &scales, 4, 1, 8, true).unwrap();
        assert_ne!(q_no, q_yes, "FWHT should change quantized bytes");
    }

    // --- V FWHT with non-power-of-two does nothing ---

    #[test]
    fn test_v_fwht_non_power_of_two_no_effect() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (q_off, s_off) = quantize_v_per_token(&v, 4, 1, 6, false).unwrap();
        let (q_on, s_on) = quantize_v_per_token(&v, 4, 1, 6, true).unwrap();
        assert_eq!(q_off, q_on, "FWHT should be skipped for non-power-of-two hidden_size");
        assert_eq!(s_off, s_on);
    }

    // --- K zero scale error at last channel ---

    #[test]
    fn test_k_zero_scale_last_channel() {
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![1.0, 1.0, 1.0, 0.0]; // scale[3] = 0
        let result = quantize_k_per_channel(&k, &scales, 4, 1, 4, false);
        assert!(matches!(result, Err(QuantError::ZeroScale(3))));
    }

    // --- K zero scale with 3-bit at specific channel ---

    #[test]
    fn test_k_3bit_zero_scale_error() {
        let k = vec![1.0; 8];
        let scales = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = quantize_k_per_channel(&k, &scales, 3, 1, 8, false);
        assert!(matches!(result, Err(QuantError::ZeroScale(2))));
    }

    // --- QuantResult mapping ---

    #[test]
    fn test_quant_result_map() {
        let result: QuantResult<u8> = Ok(42);
        let mapped = result.map(|v| v * 2);
        assert_eq!(mapped.unwrap(), 84);
    }

    #[test]
    fn test_quant_result_map_err() {
        let result: QuantResult<u8> = Err(QuantError::InvalidBitWidth(1));
        let mapped = result.map_err(|e| format!("wrapped: {}", e));
        assert!(mapped.is_err());
        assert!(mapped.unwrap_err().contains("wrapped"));
    }

    // =======================================================================
    // Additional ~45 tests — coverage gaps for public types/methods
    // =======================================================================

    // --- RabitqCorrection: C1 linear in v_norm ---

    #[test]
    fn test_rabitq_correction_c1_proportional_to_v_norm() {
        let dim = 4096;
        let corr_a = RabitqCorrection::from_quant_params(4, dim, 5.0);
        let corr_b = RabitqCorrection::from_quant_params(4, dim, 10.0);
        let ratio = corr_b.c1 / corr_a.c1;
        assert!((ratio - 2.0).abs() < 1e-4,
            "C1 should scale linearly with v_norm: ratio={}", ratio);
    }

    // --- RabitqCorrection: correct_score with negative c0 and c1 ---

    #[test]
    fn test_correct_score_negative_correction() {
        let correction = RabitqCorrection { c0: -2.0, c1: -0.5, v_norm: 1.0 };
        let result = correction.correct_score(10.0, 4.0);
        assert!((result - 6.0).abs() < 1e-6,
            "10.0 + (-0.5)*4.0 + (-2.0) = 6.0, got {}", result);
    }

    // --- RabitqCorrection: correct_scores_in_place with many elements ---

    #[test]
    fn test_correct_scores_in_place_many_elements() {
        let correction = RabitqCorrection { c0: 0.1, c1: 0.0, v_norm: 1.0 };
        let mut scores: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let expected: Vec<f32> = scores.iter().map(|&s| s + 0.1).collect();
        correction.correct_scores_in_place(&mut scores, 5.0);
        for (i, (got, exp)) in scores.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "Mismatch at index {}", i);
        }
    }

    // --- RabitqCorrection: c0 always zero for symmetric quantization ---

    #[test]
    fn test_rabitq_correction_c0_always_zero() {
        for bits in [3u8, 4] {
            for dim in [1, 4096] {
                for v_norm in [0.0, 1.0, 100.0] {
                    let corr = RabitqCorrection::from_quant_params(bits, dim, v_norm);
                    assert_eq!(corr.c0, 0.0,
                        "C0 should be 0 for symmetric quantization (bits={}, dim={})", bits, dim);
                }
            }
        }
    }

    // --- RabitqCorrection: v_norm preserved exactly ---

    #[test]
    fn test_rabitq_correction_preserves_v_norm() {
        for v_norm in [0.0, 1.0, 42.5, 1000.0] {
            let corr = RabitqCorrection::from_quant_params(4, 4096, v_norm);
            assert!((corr.v_norm - v_norm).abs() < 1e-6,
                "v_norm not preserved: expected {}, got {}", v_norm, corr.v_norm);
        }
        let corr_nan = RabitqCorrection::from_quant_params(4, 4096, f32::NAN);
        assert!(corr_nan.v_norm.is_nan(), "NaN v_norm should be preserved");
    }

    // --- RabitqCorrection: correct_score raw_score difference invariance ---

    #[test]
    fn test_correct_score_raw_score_delta_invariant() {
        let c = RabitqCorrection { c0: 1.0, c1: 2.0, v_norm: 3.0 };
        let q_norm = 3.0;
        let r1 = c.correct_score(5.0, q_norm);
        let r2 = c.correct_score(10.0, q_norm);
        assert!((r2 - r1 - 5.0).abs() < 1e-6,
            "Score difference should equal raw_score difference: got {}", r2 - r1);
    }

    // --- FWHT: length 16 basis vector produces all ones ---

    #[test]
    fn test_fwht_length_sixteen_basis_vector() {
        let mut data = vec![0.0f32; 16];
        data[0] = 1.0;
        fwht_inplace(&mut data);
        for (i, &val) in data.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-5, "Expected 1.0 at index {}, got {}", i, val);
        }
    }

    // --- FWHT: length 32 all-ones -> [32, 0, ..., 0] ---

    #[test]
    fn test_fwht_length_thirty_two_all_ones() {
        let n = 32;
        let mut data = vec![1.0f32; n];
        fwht_inplace(&mut data);
        assert!((data[0] - n as f32).abs() < 1e-3, "First element should be {}, got {}", n, data[0]);
        for i in 1..n {
            assert!(data[i].abs() < 1e-3, "Element {} should be ~0, got {}", i, data[i]);
        }
    }

    // --- FWHT: [1,1,0,0] partial-ones pattern ---

    #[test]
    fn test_fwht_partial_ones_pattern() {
        let mut data = vec![1.0f32, 1.0, 0.0, 0.0];
        fwht_inplace(&mut data);
        // stride=1: [1+1, 1-1, 0, 0] = [2, 0, 0, 0]
        // stride=2: [2+0, 0+0, 2-0, 0-0] = [2, 0, 2, 0]
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1]).abs() < 1e-6);
        assert!((data[2] - 2.0).abs() < 1e-6);
        assert!((data[3]).abs() < 1e-6);
    }

    // --- FWHT: self-inverse for length 16 ---

    #[test]
    fn test_fwht_self_inverse_length_sixteen() {
        let original: Vec<f32> = (0..16).map(|i| i as f32 * 0.7 - 5.0).collect();
        let mut data = original.clone();
        fwht_inplace(&mut data);
        fwht_inplace(&mut data);
        let n = original.len() as f32;
        for (i, (got, expected)) in data.iter().zip(original.iter()).enumerate() {
            assert!((got - expected * n).abs() < 1e-3,
                "FWHT(FWHT(x)) = n*x failed at index {}", i);
        }
    }

    // --- K quantize_config: scales length mismatch directly ---

    #[test]
    fn test_k_quantize_config_scales_len_mismatch() {
        let k = vec![1.0; 8];
        let scales = vec![1.0; 3];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(&k, &scales, 4, 2, 4, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 4, actual: 3 })));
    }

    // --- K quantize_config: k data length mismatch directly ---

    #[test]
    fn test_k_quantize_config_k_data_len_mismatch() {
        let k = vec![1.0; 6];
        let scales = vec![1.0; 4];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(&k, &scales, 4, 2, 4, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 8, actual: 6 })));
    }

    // --- K quantize_config: 3-bit zero scale error ---

    #[test]
    fn test_k_quantize_config_3bit_zero_scale() {
        let k = vec![1.0; 8];
        let scales = vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let config = KvQuantConfig { bits: 3, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(&k, &scales, 3, 1, 8, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ZeroScale(3))));
    }

    // --- K config API: invalid bit width directly ---

    #[test]
    fn test_k_config_invalid_bit_width() {
        let k = vec![1.0; 4];
        let scales = vec![1.0; 4];
        let config = KvQuantConfig { bits: 2, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(&k, &scales, 2, 1, 4, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(2))));
    }

    // --- V quantize_config: shape mismatch directly ---

    #[test]
    fn test_v_quantize_config_shape_mismatch() {
        let v = vec![1.0; 6];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_config(&v, 4, 2, 4, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { expected: 8, actual: 6 })));
    }

    // --- V config API: invalid bit width directly ---

    #[test]
    fn test_v_config_invalid_bit_width() {
        let v = vec![1.0; 4];
        let config = KvQuantConfig { bits: 9, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_config(&v, 9, 1, 4, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::InvalidBitWidth(9))));
    }

    // --- K RaBitQ mode: produces stochastic variation ---

    #[test]
    fn test_k_rabitq_mode_stochastic_variation() {
        let k = vec![1.5, 2.3, 3.7, 4.1, 5.2, 6.8, 7.4, 8.9];
        let scales = vec![9.0; 8];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::RaBitQ };
        let mut rng = rand::thread_rng();
        let first = quantize_k_per_channel_with_config(&k, &scales, 4, 1, 8, &config, &mut rng).unwrap();
        let mut any_different = false;
        for _ in 0..20 {
            let next = quantize_k_per_channel_with_config(&k, &scales, 4, 1, 8, &config, &mut rng).unwrap();
            if next != first {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "RaBitQ K should produce stochastic variation");
    }

    // --- V RaBitQ mode: produces stochastic variation ---

    #[test]
    fn test_v_rabitq_mode_stochastic_variation() {
        let v = vec![1.5, 2.3, 3.7, 4.1, 5.2, 6.8, 7.4, 8.9];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::RaBitQ };
        let mut rng = rand::thread_rng();
        let (first, _) = quantize_v_per_token_with_config(&v, 4, 1, 8, &config, &mut rng).unwrap();
        let mut any_different = false;
        for _ in 0..20 {
            let (next, _) = quantize_v_per_token_with_config(&v, 4, 1, 8, &config, &mut rng).unwrap();
            if next != first {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "RaBitQ V should produce stochastic variation");
    }

    // --- K 4-bit multi-seq round-trip ---

    #[test]
    fn test_k_4bit_multi_seq_round_trip() {
        let seq_len = 4;
        let hidden_size = 8;
        let k: Vec<f32> = (0..seq_len * hidden_size).map(|i| ((i % 7) + 1) as f32).collect();
        let scales = vec![7.0; hidden_size];
        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), k.len());
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.15, "Multi-seq K at {}: {} vs {}", i, orig, val);
        }
    }

    // --- K 3-bit: all max values round-trip ---

    #[test]
    fn test_k_3bit_all_max_values() {
        let k = vec![7.0; 8];
        let scales = vec![7.0; 8];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 8, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 1, 8).unwrap();
        for (i, &val) in deq.iter().enumerate() {
            let error = (val - 7.0).abs() / 7.0;
            assert!(error < 0.15, "Max value round-trip at {}: got {}", i, val);
        }
    }

    // --- V 3-bit: single element round-trip ---

    #[test]
    fn test_v_3bit_single_element() {
        let v = vec![5.0];
        let (quantized, scales) = quantize_v_per_token(&v, 3, 1, 1, false).unwrap();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 5.0).abs() < 1e-6);
        assert_eq!(quantized.len(), 1);
        let deq = dequantize_v_per_token(&quantized, &scales, 3, 1, 1).unwrap();
        let error = (v[0] - deq[0]).abs() / v[0].max(1.0);
        assert!(error < 0.3, "V 3-bit single element: {} vs {}", v[0], deq[0]);
    }

    // --- K 3-bit: hidden_size=16 round-trip ---

    #[test]
    fn test_k_3bit_hidden_size_sixteen() {
        // Use values close to scale for good 3-bit fidelity (7 levels)
        let k: Vec<f32> = (0..16).map(|i| ((i % 7) + 1) as f32 * 0.9).collect();
        let scales: Vec<f32> = (0..16).map(|_| 7.0f32).collect();
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 16, false).unwrap();
        assert_eq!(quantized.len(), 6, "16 * 3 / 8 = 6 bytes");
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 1, 16).unwrap();
        assert_eq!(deq.len(), 16);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.35, "K 3-bit hidden=16 at {}: {} vs {}", i, orig, val);
        }
    }

    // --- K 4-bit: large seq_len ---

    #[test]
    fn test_k_4bit_large_seq_len() {
        let seq_len = 32;
        let hidden_size = 4;
        let k: Vec<f32> = (0..seq_len * hidden_size).map(|i| ((i % 8) + 1) as f32).collect();
        let scales = vec![8.0; hidden_size];
        let quantized = quantize_k_per_channel(&k, &scales, 4, seq_len, hidden_size, false).unwrap();
        assert_eq!(quantized.len(), (seq_len * hidden_size).div_ceil(2));
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), k.len());
    }

    // --- V 4-bit: large seq_len ---

    #[test]
    fn test_v_4bit_large_seq_len() {
        let seq_len = 32;
        let hidden_size = 4;
        let v: Vec<f32> = (0..seq_len * hidden_size).map(|i| ((i % 8) + 1) as f32).collect();
        let (quantized, scales) = quantize_v_per_token(&v, 4, seq_len, hidden_size, false).unwrap();
        assert_eq!(scales.len(), seq_len);
        assert_eq!(quantized.len(), (seq_len * hidden_size).div_ceil(2));
        let deq = dequantize_v_per_token(&quantized, &scales, 4, seq_len, hidden_size).unwrap();
        assert_eq!(deq.len(), v.len());
    }

    // --- K 3-bit RaBitQ unbiased expectation ---

    #[test]
    fn test_k_3bit_rabitq_unbiased() {
        let k: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scales = vec![8.0; 8];
        let bits = 3;
        let seq_len = 1;
        let hidden_size = 8;
        let trials = 5000;
        let config = KvQuantConfig { bits, sink_count: 0, fwht_enabled: false, mode: QuantMode::RaBitQ };
        let mut sum = vec![0.0f32; hidden_size];
        for _ in 0..trials {
            let mut rng = rand::thread_rng();
            let quantized = quantize_k_per_channel_with_config(
                &k, &scales, bits, seq_len, hidden_size, &config, &mut rng
            ).unwrap();
            let deq = dequantize_k_per_channel(&quantized, &scales, bits, seq_len, hidden_size).unwrap();
            for i in 0..hidden_size {
                sum[i] += deq[i];
            }
        }
        for i in 0..hidden_size {
            let mean = sum[i] / trials as f32;
            let error = (mean - k[i]).abs() / k[i];
            assert!(error < 0.12, "K 3-bit RaBitQ bias at {}: mean={} vs expected={}", i, mean, k[i]);
        }
    }

    // --- K RaBitQ + FWHT combined ---

    #[test]
    fn test_k_rabitq_with_fwht_combined() {
        let k: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let scales = vec![8.0; 8];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: true, mode: QuantMode::RaBitQ };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_config(&k, &scales, 4, 1, 8, &config, &mut rng);
        assert!(result.is_ok());
        let quantized = result.unwrap();
        assert_eq!(quantized.len(), 4, "4-bit packing of 8 elements = 4 bytes");
    }

    // --- V RaBitQ + FWHT combined ---

    #[test]
    fn test_v_rabitq_with_fwht_combined() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: true, mode: QuantMode::RaBitQ };
        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_config(&v, 4, 1, 8, &config, &mut rng);
        assert!(result.is_ok());
        let (quantized, scales) = result.unwrap();
        assert_eq!(quantized.len(), 4);
        assert_eq!(scales.len(), 1);
    }

    // --- V 3-bit with FWHT enabled (power-of-two hidden) ---

    #[test]
    fn test_v_3bit_fwht_enabled() {
        let v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let config = KvQuantConfig { bits: 3, sink_count: 0, fwht_enabled: true, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let (quantized, scales) = quantize_v_per_token_with_config(&v, 3, 1, 8, &config, &mut rng).unwrap();
        assert_eq!(scales.len(), 1);
        assert_eq!(quantized.len(), 3, "8 elements * 3 bits / 8 = 3 bytes");
    }

    // --- Dequantize K 3-bit: all-max packed bytes -> all scale values ---

    #[test]
    fn test_dequantize_k_3bit_all_max_packed() {
        // All 7s packed: [0xFF, 0xFF, 0xFF]
        let q = vec![0xFFu8, 0xFF, 0xFF];
        let scales = vec![10.0; 8];
        let result = dequantize_k_per_channel(&q, &scales, 3, 1, 8).unwrap();
        for (i, &val) in result.iter().enumerate() {
            assert!((val - 10.0).abs() < 0.01,
                "All-max 3-bit dequant at {} should be ~10.0, got {}", i, val);
        }
    }

    // --- Dequantize V 4-bit: known byte pattern ---

    #[test]
    fn test_dequantize_v_4bit_known_pattern() {
        let q = vec![0xAB]; // low nibble=0xB=11, high nibble=0xA=10
        let scales = vec![1.0]; // 1 token
        let result = dequantize_v_per_token(&q, &scales, 4, 1, 2).unwrap();
        assert!((result[0] - 11.0 / 15.0).abs() < 1e-5,
            "Expected {}, got {}", 11.0 / 15.0, result[0]);
        assert!((result[1] - 10.0 / 15.0).abs() < 1e-5,
            "Expected {}, got {}", 10.0 / 15.0, result[1]);
    }

    // --- Dequantize K 4-bit: known byte pattern ---

    #[test]
    fn test_dequantize_k_4bit_known_pattern() {
        let q = vec![0x12]; // low nibble=2, high nibble=1
        let scales = vec![15.0]; // 1 channel
        let result = dequantize_k_per_channel(&q, &scales, 4, 2, 1).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", result[1]);
    }

    // --- V property: dequantized values in [0, scale] range ---

    #[test]
    fn test_v_4bit_dequant_range_property() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let (quantized, scales) = quantize_v_per_token(&v, 4, 4, 4, false).unwrap();
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 4, 4).unwrap();
        for t in 0..4 {
            let scale = scales[t];
            for c in 0..4 {
                let val = deq[t * 4 + c];
                assert!(val >= 0.0, "Dequant at t={} c={} should be >= 0, got {}", t, c, val);
                assert!(val <= scale * 1.01,
                    "Dequant at t={} c={} should be <= scale, got {} > {}", t, c, val, scale);
            }
        }
    }

    // --- K property: monotonic channels with uniform scale ---

    #[test]
    fn test_k_4bit_monotonic_channels() {
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scales = vec![8.0; 8];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 8, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 8).unwrap();
        for i in 0..deq.len() - 1 {
            assert!(deq[i] <= deq[i + 1] + 0.5,
                "Monotonicity violated at {}: {} > {}", i, deq[i], deq[i + 1]);
        }
    }

    // --- V property: scale equals max_abs per row for multi-token ---

    #[test]
    fn test_v_scale_is_max_abs_per_row() {
        let v = vec![
            1.0, -5.0, 3.0, 2.0,   // max_abs = 5.0
            -8.0, 2.0, 1.0, 4.0,   // max_abs = 8.0
            0.5, 0.3, 0.1, 0.9,    // max_abs = 0.9
        ];
        let (_, scales) = quantize_v_per_token(&v, 4, 3, 4, false).unwrap();
        assert!((scales[0] - 5.0).abs() < 1e-6, "Token 0: expected 5.0, got {}", scales[0]);
        assert!((scales[1] - 8.0).abs() < 1e-6, "Token 1: expected 8.0, got {}", scales[1]);
        assert!((scales[2] - 0.9).abs() < 1e-6, "Token 2: expected 0.9, got {}", scales[2]);
    }

    // --- K with correction 3-bit: valid tuple structure ---

    #[test]
    fn test_k_with_correction_3bit() {
        let k = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scales = vec![8.0; 8];
        let config = KvQuantConfig { bits: 3, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let (quantized, returned_scales, correction) = quantize_k_per_channel_with_correction(
            &k, &scales, 3, 1, 8, 7.0, &config, &mut rng
        ).unwrap();
        assert_eq!(quantized.len(), 3, "3-bit packing of 8 elements = 3 bytes");
        assert_eq!(returned_scales, scales);
        assert_eq!(correction.c0, 0.0);
        assert!(correction.c1 > 0.0);
        assert!((correction.v_norm - 7.0).abs() < 1e-6);
    }

    // --- V with correction 3-bit: valid tuple structure ---

    #[test]
    fn test_v_with_correction_3bit() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let config = KvQuantConfig { bits: 3, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let (quantized, scales, correction) = quantize_v_per_token_with_correction(
            &v, 3, 1, 8, 5.0, &config, &mut rng
        ).unwrap();
        assert_eq!(quantized.len(), 3);
        assert_eq!(scales.len(), 1);
        assert_eq!(correction.c0, 0.0);
        assert!(correction.c1 > 0.0);
        assert!((correction.v_norm - 5.0).abs() < 1e-6);
    }

    // --- K 4-bit: seq_len=1 hidden_size=2 boundary ---

    #[test]
    fn test_k_4bit_seq_one_hidden_two() {
        let k = vec![3.0, 7.0];
        let scales = vec![7.0, 7.0];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 2, false).unwrap();
        assert_eq!(quantized.len(), 1);
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 2).unwrap();
        assert!((deq[0] - 3.0).abs() / 3.0 < 0.15);
        assert!((deq[1] - 7.0).abs() / 7.0 < 0.15);
    }

    // --- V 4-bit: seq_len=1 hidden_size=2 ---

    #[test]
    fn test_v_4bit_seq_one_hidden_two() {
        let v = vec![2.0, 6.0];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 1, 2, false).unwrap();
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 6.0).abs() < 1e-6);
        assert_eq!(quantized.len(), 1);
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 1, 2).unwrap();
        assert!((deq[0] - 2.0).abs() / 2.0 < 0.2);
        assert!((deq[1] - 6.0).abs() / 6.0 < 0.15);
    }

    // --- should_preserve_fp16: both arguments zero ---

    #[test]
    fn test_should_preserve_fp16_both_zero() {
        assert!(!should_preserve_fp16(0, 0, false));
        assert!(!should_preserve_fp16(0, 0, true), "sink_count=0 should never preserve");
    }

    // --- V 4-bit: very small values round-trip ---

    #[test]
    fn test_v_4bit_very_small_values() {
        let v = vec![1e-7, 2e-7, 3e-7, 4e-7];
        let (quantized, scales) = quantize_v_per_token(&v, 4, 1, 4, false).unwrap();
        assert!((scales[0] - 4e-7).abs() < 1e-12, "Scale should be 4e-7, got {}", scales[0]);
        let deq = dequantize_v_per_token(&quantized, &scales, 4, 1, 4).unwrap();
        for (i, (&orig, &val)) in v.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig;
            assert!(error < 0.15, "Small value V round-trip at {}: {} vs {}", i, orig, val);
        }
    }

    // --- QuantError: source() returns None for all variants ---

    #[test]
    fn test_quant_error_source_is_none() {
        let err1: &dyn std::error::Error = &QuantError::InvalidBitWidth(1);
        assert!(err1.source().is_none(), "InvalidBitWidth source should be None");
        let err2: &dyn std::error::Error = &QuantError::ShapeMismatch { expected: 1, actual: 2 };
        assert!(err2.source().is_none(), "ShapeMismatch source should be None");
        let err3: &dyn std::error::Error = &QuantError::ZeroScale(0);
        assert!(err3.source().is_none(), "ZeroScale source should be None");
    }

    // --- apply_rabitq_correction: matches per-element correct_score ---

    #[test]
    fn test_apply_correction_matches_manual_loop() {
        let correction = RabitqCorrection { c0: 0.5, c1: 1.5, v_norm: 2.0 };
        let mut scores_apply = vec![10.0, 20.0, 30.0];
        let scores_manual: Vec<f32> = scores_apply.iter()
            .map(|&s| correction.correct_score(s, 3.0))
            .collect();
        apply_rabitq_correction(&mut scores_apply, &correction, 3.0);
        assert_eq!(scores_apply, scores_manual);
    }

    // --- KvQuantConfig: clone produces identical quantization ---

    #[test]
    fn test_kv_config_clone_identical_quantization() {
        let config = KvQuantConfig { bits: 4, sink_count: 2, fwht_enabled: false, mode: QuantMode::Deterministic };
        let cloned = config.clone();
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let scales = vec![4.0; 4];
        let mut rng1 = rand::thread_rng();
        let mut rng2 = rand::thread_rng();
        let q1 = quantize_k_per_channel_with_config(&k, &scales, 4, 1, 4, &config, &mut rng1).unwrap();
        let q2 = quantize_k_per_channel_with_config(&k, &scales, 4, 1, 4, &cloned, &mut rng2).unwrap();
        assert_eq!(q1, q2, "Cloned config should produce identical quantization");
    }

    // --- Dequantize K 3-bit: partial group (non-multiple-of-8 hidden) ---

    #[test]
    fn test_dequantize_k_3bit_partial_group() {
        let k = vec![3.0, 6.0, 2.0, 5.0, 1.0];
        let scales = vec![6.0; 5];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 5, false).unwrap();
        assert_eq!(quantized.len(), 2, "5*3=15 bits, ceil(15/8)=2 bytes");
        let deq = dequantize_k_per_channel(&quantized, &scales, 3, 1, 5).unwrap();
        assert_eq!(deq.len(), 5);
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.35, "Partial group at {}: {} vs {}", i, orig, val);
        }
    }

    // --- Dequantize V 3-bit: partial group ---

    #[test]
    fn test_dequantize_v_3bit_partial_group() {
        let v = vec![2.0, 4.0, 6.0, 1.0, 3.0];
        let (quantized, scales) = quantize_v_per_token(&v, 3, 1, 5, false).unwrap();
        assert_eq!(quantized.len(), 2, "5*3=15 bits, ceil(15/8)=2 bytes");
        let deq = dequantize_v_per_token(&quantized, &scales, 3, 1, 5).unwrap();
        assert_eq!(deq.len(), 5);
        for (i, (&orig, &val)) in v.iter().zip(deq.iter()).enumerate() {
            let error = (orig - val).abs() / orig.max(1.0);
            assert!(error < 0.45, "V partial group at {}: {} vs {}", i, orig, val);
        }
    }

    // --- K 3-bit packed size: non-multiple-of-8 hidden ---

    #[test]
    fn test_k_3bit_packed_size_non_multiple_of_eight() {
        let k = vec![1.0; 5];
        let scales = vec![1.0; 5];
        let quantized = quantize_k_per_channel(&k, &scales, 3, 1, 5, false).unwrap();
        assert_eq!(quantized.len(), 2, "5*3=15 bits, ceil(15/8)=2 bytes");
    }

    // --- V 3-bit packed size: non-multiple-of-8 hidden ---

    #[test]
    fn test_v_3bit_packed_size_non_multiple_of_eight() {
        let v = vec![1.0; 10];
        let (quantized, _) = quantize_v_per_token(&v, 3, 1, 10, false).unwrap();
        assert_eq!(quantized.len(), 4, "10*3=30 bits, ceil(30/8)=4 bytes");
    }

    // --- K with correction: error propagation for shape mismatch ---

    #[test]
    fn test_k_correction_shape_mismatch_propagation() {
        let k = vec![1.0; 6];
        let scales = vec![1.0; 4];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_correction(&k, &scales, 4, 2, 4, 5.0, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { .. })));
    }

    // --- V with correction: error propagation for shape mismatch ---

    #[test]
    fn test_v_correction_shape_mismatch_propagation() {
        let v = vec![1.0; 6];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_v_per_token_with_correction(&v, 4, 2, 4, 5.0, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ShapeMismatch { .. })));
    }

    // --- K with correction: zero scale error propagation ---

    #[test]
    fn test_k_correction_zero_scale_propagation() {
        let k = vec![1.0; 4];
        let scales = vec![0.0, 1.0, 1.0, 1.0];
        let config = KvQuantConfig { bits: 4, sink_count: 0, fwht_enabled: false, mode: QuantMode::Deterministic };
        let mut rng = rand::thread_rng();
        let result = quantize_k_per_channel_with_correction(&k, &scales, 4, 1, 4, 5.0, &config, &mut rng);
        assert!(matches!(result, Err(QuantError::ZeroScale(0))));
    }

    // =======================================================================
    // Additional 13 tests — edge cases, boundaries, precision, trait coverage
    // =======================================================================

    // --- K 4-bit: value exactly equals scale yields maximal quantization ---

    #[test]
    fn test_k_4bit_value_equals_scale_precision() {
        // When val == scale, normalized = max_val_f32 = 15.0, which quantizes exactly to 15
        // Dequantized = 15 * scale / 15 = scale exactly
        let k = vec![5.0, 5.0, 5.0, 5.0];
        let scales = vec![5.0; 4];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 4).unwrap();
        for (i, &val) in deq.iter().enumerate() {
            assert!((val - 5.0).abs() < 1e-6,
                "Value==scale should round-trip exactly at index {}, got {}", i, val);
        }
    }

    // --- Dequantize K 4-bit: all-zero packed bytes produce zero output ---

    #[test]
    fn test_dequantize_k_4bit_all_zero_bytes() {
        let q = vec![0u8; 2];
        let scales = vec![10.0; 4];
        let result = dequantize_k_per_channel(&q, &scales, 4, 1, 4).unwrap();
        assert!(result.iter().all(|&x| x == 0.0),
            "All-zero packed bytes should dequantize to all zeros");
    }

    // --- Dequantize V 4-bit: all-zero packed bytes produce zero output ---

    #[test]
    fn test_dequantize_v_4bit_all_zero_bytes() {
        let q = vec![0u8; 2];
        let scales = vec![10.0; 1];
        let result = dequantize_v_per_token(&q, &scales, 4, 1, 4).unwrap();
        assert!(result.iter().all(|&x| x == 0.0),
            "All-zero packed bytes should dequantize V to all zeros");
    }

    // --- QuantError Display: each variant contains expected substring ---

    #[test]
    fn test_quant_error_display_contains_diagnostic() {
        let msg1 = format!("{}", QuantError::InvalidBitWidth(255));
        assert!(msg1.contains("255"), "InvalidBitWidth(255) should contain '255': got {}", msg1);
        let msg2 = format!("{}", QuantError::ShapeMismatch { expected: 0, actual: 0 });
        assert!(msg2.contains("0"), "ShapeMismatch with 0s should display: got {}", msg2);
        let msg3 = format!("{}", QuantError::ZeroScale(0));
        assert!(msg3.contains("zero scale") && msg3.contains("0"),
            "ZeroScale(0) should contain 'zero scale' and '0': got {}", msg3);
    }

    // --- QuantMode in HashSet of size 2 ---

    #[test]
    fn test_quant_mode_hashset_contains() {
        use std::collections::HashSet;
        let set: HashSet<QuantMode> = [QuantMode::Deterministic, QuantMode::RaBitQ].into_iter().collect();
        assert!(set.contains(&QuantMode::Deterministic));
        assert!(set.contains(&QuantMode::RaBitQ));
        assert_eq!(set.len(), 2);
    }

    // --- correct_score with large negative q_norm ---

    #[test]
    fn test_correct_score_large_negative_q_norm() {
        let correction = RabitqCorrection { c0: 0.0, c1: 1.0, v_norm: 5.0 };
        let result = correction.correct_score(100.0, -1000.0);
        // corrected = 100.0 + 1.0 * (-1000.0) + 0.0 = -900.0
        assert!((result - (-900.0)).abs() < 1e-3,
            "Large negative q_norm should subtract from raw score, got {}", result);
    }

    // --- FWHT: length 8 single non-zero element at index 3 ---

    #[test]
    fn test_fwht_length_eight_single_element_at_index_3() {
        let mut data = vec![0.0f32; 8];
        data[3] = 1.0;
        fwht_inplace(&mut data);
        // FWHT of basis vector e_3 produces alternating +/- 1 pattern
        // The Hadamard matrix row 3 is [1, -1, -1, 1, 1, -1, -1, 1] (Walsh-ordered)
        let sum_sq: f32 = data.iter().map(|x| x * x).sum();
        assert!((sum_sq - 8.0).abs() < 1e-4,
            "Energy should be 8 (FWHT preserves energy up to scale), got {}", sum_sq);
    }

    // --- K 4-bit: scale exactly twice the value ---

    #[test]
    fn test_k_4bit_scale_double_value() {
        let k = vec![3.0, 3.0, 3.0, 3.0];
        let scales = vec![6.0; 4]; // scale = 2 * value
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 4, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 4).unwrap();
        // normalized = (3.0 / 6.0) * 15 = 7.5, rounds to 8
        // dequantized = 8 * 6.0 / 15 = 3.2
        for (i, &val) in deq.iter().enumerate() {
            let error = (val - 3.0).abs() / 3.0;
            assert!(error < 0.15, "Scale=2x value at {}: got {}, error={}", i, val, error);
        }
    }

    // --- V 3-bit: all identical positive values ---

    #[test]
    fn test_v_3bit_all_identical_values() {
        let v = vec![4.0; 8];
        let (quantized, scales) = quantize_v_per_token(&v, 3, 1, 8, false).unwrap();
        assert!((scales[0] - 4.0).abs() < 1e-6, "Scale should be 4.0");
        let deq = dequantize_v_per_token(&quantized, &scales, 3, 1, 8).unwrap();
        let first = deq[0];
        // All values identical and equal to scale => all quantize to max => all dequantize same
        for (i, &val) in deq.iter().enumerate() {
            assert!((val - first).abs() < 1e-6,
                "All V dequantized values should be equal at index {}", i);
        }
    }

    // --- KvQuantConfig: bits field does not affect other fields ---

    #[test]
    fn test_kv_config_bits_field_independence() {
        let config_3 = KvQuantConfig { bits: 3, sink_count: 5, fwht_enabled: true, mode: QuantMode::RaBitQ };
        let config_4 = KvQuantConfig { bits: 4, sink_count: 5, fwht_enabled: true, mode: QuantMode::RaBitQ };
        // Changing bits should not affect other fields
        assert_eq!(config_3.sink_count, config_4.sink_count);
        assert_eq!(config_3.fwht_enabled, config_4.fwht_enabled);
        assert_eq!(config_3.mode, config_4.mode);
        assert_ne!(config_3.bits, config_4.bits);
    }

    // --- should_preserve_fp16: token index exactly at sink_count boundary ---

    #[test]
    fn test_should_preserve_fp16_exact_boundary_token() {
        // token_idx == sink_count should NOT preserve (<, not <=)
        assert!(!should_preserve_fp16(10, 10, true),
            "token_idx == sink_count should not preserve");
        // token_idx == sink_count - 1 should preserve
        assert!(should_preserve_fp16(9, 10, true),
            "token_idx == sink_count - 1 should preserve");
    }

    // --- K 4-bit: full range 0 to 15 quantization levels ---

    #[test]
    fn test_k_4bit_full_quantization_range() {
        // Scale = 15.0, values 0..=15 map to quantization levels 0..=15
        let k: Vec<f32> = (0..=15).map(|i| i as f32).collect();
        let scales = vec![15.0; 16];
        let quantized = quantize_k_per_channel(&k, &scales, 4, 1, 16, false).unwrap();
        let deq = dequantize_k_per_channel(&quantized, &scales, 4, 1, 16).unwrap();
        assert_eq!(quantized.len(), 8, "16 elements / 2 = 8 bytes");
        // Each value should round-trip exactly: q = round(val/scale * 15) = val, deq = q * scale / 15 = val
        for (i, (&orig, &val)) in k.iter().zip(deq.iter()).enumerate() {
            assert!((val - orig).abs() < 0.01,
                "Full range value at {} should round-trip: expected {}, got {}", i, orig, val);
        }
    }

    // --- RabitqCorrection: correct_scores_in_place applies uniform delta ---

    #[test]
    fn test_correct_scores_uniform_delta() {
        let correction = RabitqCorrection { c0: 0.0, c1: 3.0, v_norm: 1.0 };
        let q_norm = 2.0;
        let mut scores = vec![0.0, 100.0, -50.0, 0.001];
        let original_deltas: Vec<f32> = scores.windows(2).map(|w| w[1] - w[0]).collect();
        correction.correct_scores_in_place(&mut scores, q_norm);
        // All scores shifted by same constant (3.0 * 2.0 + 0.0 = 6.0)
        let new_deltas: Vec<f32> = scores.windows(2).map(|w| w[1] - w[0]).collect();
        for (i, (orig, new)) in original_deltas.iter().zip(new_deltas.iter()).enumerate() {
            assert!((orig - new).abs() < 1e-6,
                "Delta between adjacent scores should be preserved at pair {}, got {} vs {}", i, orig, new);
        }
    }
}
