// KIVI Key/Val mixed-precision strategy (per SPEC 19-KV-CACHE-OPTIMIZATION.md §3)
//
// KIVI asymmetric quantization exploits the fundamentally different outlier
// distributions of key and value caches:
//
// - **K cache**: outliers concentrated in specific *channels* (stable across tokens)
//   → Per-Channel quantization with higher precision (FP16/FP8)
// - **V cache**: outliers concentrated in specific *tokens* (stable across channels)
//   → Per-Token quantization with lower precision (KIVI4/KIVI2)
//
// ## Write Path
// 1. K data → per-channel scale computation → FP16/FP8 write (keep high precision)
// 2. V data → per-token scale computation → INT4/INT2 write (aggressive compression)
//
// ## Read Path
// 1. K data → direct FP16/FP8 read (no dequant needed)
// 2. V data → INT4/INT2 dequant + per-token scale → FP32 compute precision
//
// ## Integration
// KiviStrategy is consulted by the KV cache write path to determine the
// quantization precision for each page. The per-page PrecisionTier in
// KvPageHeader reflects the *page-level* tier, while KiviStrategy provides
// the *key vs value* asymmetry within a page.


// ============================================================================
// KiviStrategy — asymmetric K/V quantization strategy
// ============================================================================

/// KIVI mixed-precision strategy (SPEC §3)
///
/// Key cache retains high precision (FP16/FP8) with per-channel quantization.
/// Value cache uses aggressive low-bit quantization (KIVI4/KIVI2) with
/// per-token quantization.
///
/// This asymmetry is motivated by the observation that K outliers are
/// channel-stable (protect entire channels) while V outliers are token-stable
/// (protect individual tokens but compress everything else).
#[derive(Debug, Clone)]
pub struct KiviStrategy {
    /// Key cache precision tier (default: FP16)
    pub key_precision: PrecisionTier,
    /// Value cache precision tier (default: KIVI4)
    pub val_precision: PrecisionTier,
    /// Number of attention sink tokens to protect at full precision
    pub sink_count: usize,
    /// Whether KIVI asymmetric quantization is active
    pub enabled: bool,
    /// Per-channel K scales (num_kv_heads * head_dim) for dequantization
    k_channel_scales: Vec<f32>,
    /// Per-token V scales buffer for dequantization
    v_token_scales: Vec<f32>,
}

impl Default for KiviStrategy {
    fn default() -> Self {
        Self {
            key_precision: PrecisionTier::FP16,
            val_precision: PrecisionTier::KIVI4,
            sink_count: 4,
            enabled: true,
            k_channel_scales: Vec::new(),
            v_token_scales: Vec::new(),
        }
    }
}

impl KiviStrategy {
    /// Create a new KIVI strategy with default asymmetric settings.
    ///
    /// Key: FP16 (high precision), Value: KIVI4 (4-bit INT).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled KIVI strategy (both K and V at FP16).
    pub fn disabled() -> Self {
        Self {
            key_precision: PrecisionTier::FP16,
            val_precision: PrecisionTier::FP16,
            sink_count: 0,
            enabled: false,
            k_channel_scales: Vec::new(),
            v_token_scales: Vec::new(),
        }
    }

    /// Configure key precision tier.
    pub fn with_key_precision(mut self, tier: PrecisionTier) -> Self {
        self.key_precision = tier;
        self
    }

    /// Configure value precision tier.
    pub fn with_val_precision(mut self, tier: PrecisionTier) -> Self {
        self.val_precision = tier;
        self
    }

    /// Set sink token count for attention sink protection.
    pub fn with_sink_count(mut self, count: usize) -> Self {
        self.sink_count = count;
        self
    }

    // ── K cache: Per-Channel quantization ──

    /// Compute per-channel scales for K cache.
    ///
    /// K cache shape: [num_tokens, num_kv_heads, head_dim]
    /// Scales are computed per (head, channel_dim) pair, collapsing the token dimension.
    ///
    /// # Arguments
    /// * `k_data` - K cache data as f32 slice, shape [num_tokens * num_kv_heads * head_dim]
    /// * `num_tokens` - number of tokens in this batch
    /// * `num_kv_heads` - number of KV heads
    /// * `head_dim` - dimension per head
    ///
    /// # Returns
    /// Per-channel scales, shape [num_kv_heads * head_dim]
    pub fn compute_k_channel_scales(
        &mut self,
        k_data: &[f32],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> &[f32] {
        let num_channels = num_kv_heads * head_dim;

        if num_tokens == 0 || num_channels == 0 {
            self.k_channel_scales.clear();
            return &self.k_channel_scales;
        }

        self.k_channel_scales.resize(num_channels, 0.0f32);

        // Per-channel: for each (head, channel), find max abs across all tokens
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let chan_idx = h * head_dim + d;
                let mut max_abs = 0.0f32;
                for t in 0..num_tokens {
                    let idx = t * num_kv_heads * head_dim + chan_idx;
                    let abs_val = k_data[idx].abs();
                    if abs_val > max_abs {
                        max_abs = abs_val;
                    }
                }
                // Scale for FP16: use max_abs directly (FP16 range is large enough)
                // For FP8: clamp to FP8 max representable
                let scale = match self.key_precision {
                    PrecisionTier::FP16 => {
                        // FP16: scale = 1.0 (no quantization, just range tracking)
                        if max_abs > 0.0 { max_abs } else { 1.0 }
                    }
                    PrecisionTier::FP8 => {
                        // Symmetric int8: scale = max_abs, maps [-max_abs, max_abs] → [-127, 127]
                        if max_abs > 0.0 { max_abs } else { 1.0 }
                    }
                    _ => {
                        // For other tiers: use max_abs as scale reference
                        if max_abs > 0.0 { max_abs } else { 1.0 }
                    }
                };
                self.k_channel_scales[chan_idx] = scale;
            }
        }

        &self.k_channel_scales
    }

    /// Quantize K cache with per-channel scaling (write path).
    ///
    /// For FP16 precision tier, values are stored as-is (f32 → f16 truncation
    /// handled by caller). For FP8, per-channel scaling is applied.
    ///
    /// # Returns
    /// Packed bytes + metadata for the quantized K cache.
    pub fn quantize_k(
        &self,
        k_data: &[f32],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> KiviQuantResult {
        let num_channels = num_kv_heads * head_dim;
        let total_elements = num_tokens * num_channels;

        match self.key_precision {
            PrecisionTier::FP16 => {
                // FP16: pack f32 → f16 bits (2 bytes per element)
                let mut packed = Vec::with_capacity(total_elements * 2);
                for &val in k_data.iter().take(total_elements) {
                    let bits = super::f32_to_f16_bits(val);
                    packed.extend_from_slice(&bits.to_le_bytes());
                }
                KiviQuantResult {
                    data: packed,
                    scales: self.k_channel_scales.clone(),
                    bytes_per_element: 2,
                    precision_tier: PrecisionTier::FP16,
                }
            }
            PrecisionTier::FP8 => {
                // FP8: pack f32 → 8-bit symmetric int8 (per-channel scale, 1 byte per element)
                let mut packed = Vec::with_capacity(total_elements);
                let scales = &self.k_channel_scales;
                for t in 0..num_tokens {
                    for c in 0..num_channels {
                        let val = k_data[t * num_channels + c];
                        let scale = if scales.len() > c { scales[c] } else { 1.0 };
                        let scaled = if scale.abs() > 1e-8 {
                            (val / scale * 127.0).round().clamp(-127.0, 127.0)
                        } else {
                            0.0
                        };
                        let q = scaled as i8 as u8;
                        packed.push(q);
                    }
                }
                KiviQuantResult {
                    data: packed,
                    scales: self.k_channel_scales.clone(),
                    bytes_per_element: 1,
                    precision_tier: PrecisionTier::FP8,
                }
            }
            _ => {
                // Fallback for unsupported K precision: treat as FP16
                let mut packed = Vec::with_capacity(total_elements * 2);
                for &val in k_data.iter().take(total_elements) {
                    let bits = super::f32_to_f16_bits(val);
                    packed.extend_from_slice(&bits.to_le_bytes());
                }
                KiviQuantResult {
                    data: packed,
                    scales: Vec::new(),
                    bytes_per_element: 2,
                    precision_tier: PrecisionTier::FP16,
                }
            }
        }
    }

    /// Dequantize K cache back to f32 (read path).
    ///
    /// Reverses the per-channel quantization applied in `quantize_k`.
    pub fn dequantize_k(
        &self,
        packed: &[u8],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        precision_tier: PrecisionTier,
    ) -> Vec<f32> {
        let num_channels = num_kv_heads * head_dim;
        let total_elements = num_tokens * num_channels;
        let mut out = vec![0.0f32; total_elements];

        match precision_tier {
            PrecisionTier::FP16 => {
                // FP16: unpack f16 bits → f32 (2 bytes per element)
                for i in 0..total_elements {
                    let byte_offset = i * 2;
                    if byte_offset + 2 <= packed.len() {
                        let lo = packed[byte_offset] as u16;
                        let hi = packed[byte_offset + 1] as u16;
                        let bits = lo | (hi << 8);
                        out[i] = super::f16_bits_to_f32(bits);
                    }
                }
            }
            PrecisionTier::FP8 => {
                // FP8: unpack symmetric int8 → f32 using per-channel scales
                for t in 0..num_tokens {
                    for c in 0..num_channels {
                        let idx = t * num_channels + c;
                        if idx < packed.len() {
                            let q = packed[idx] as i8;
                            let scale = if c < self.k_channel_scales.len() {
                                self.k_channel_scales[c]
                            } else {
                                1.0
                            };
                            out[idx] = q as f32 / 127.0 * scale;
                        }
                    }
                }
            }
            _ => {
                // Unsupported: return zeros
                out.fill(0.0);
            }
        }

        out
    }

    // ── V cache: Per-Token quantization ──

    /// Compute per-token scales for V cache.
    ///
    /// V cache shape: [num_tokens, num_kv_heads, head_dim]
    /// Scales are computed per token, collapsing the head+channel dimensions.
    ///
    /// # Arguments
    /// * `v_data` - V cache data as f32 slice, shape [num_tokens * num_kv_heads * head_dim]
    /// * `num_tokens` - number of tokens in this batch
    /// * `num_kv_heads` - number of KV heads
    /// * `head_dim` - dimension per head
    ///
    /// # Returns
    /// Per-token scales, shape [num_tokens]
    pub fn compute_v_token_scales(
        &mut self,
        v_data: &[f32],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> &[f32] {
        if num_tokens == 0 {
            self.v_token_scales.clear();
            return &self.v_token_scales;
        }

        self.v_token_scales.resize(num_tokens, 0.0f32);

        let stride = num_kv_heads * head_dim;

        // Per-token: for each token, find max abs across all (head, channel)
        for t in 0..num_tokens {
            let start = t * stride;
            let end = (start + stride).min(v_data.len());
            let mut max_abs = 0.0f32;
            for idx in start..end {
                let abs_val = v_data[idx].abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }
            self.v_token_scales[t] = if max_abs > 0.0 { max_abs } else { 1.0 };
        }

        &self.v_token_scales
    }

    /// Quantize V cache with per-token scaling (write path).
    ///
    /// Aggressively compresses V cache using low-bit quantization
    /// (KIVI4=4-bit or KIVI2=2-bit).
    ///
    /// # Returns
    /// Packed bytes + per-token scales for the quantized V cache.
    pub fn quantize_v(
        &self,
        v_data: &[f32],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> KiviQuantResult {
        let _ = num_kv_heads;
        let _ = head_dim;
        let stride = num_kv_heads * head_dim;
        let scales = &self.v_token_scales;

        match self.val_precision {
            PrecisionTier::KIVI4 => {
                // 4-bit: 2 values per byte, symmetric signed [-7, 7]
                let elements_per_byte = 2;
                let total_bytes = (num_tokens * stride).div_ceil(elements_per_byte);
                let mut packed = vec![0u8; total_bytes];

                for t in 0..num_tokens {
                    let token_scale = if t < scales.len() && scales[t].abs() > 1e-8 {
                        scales[t]
                    } else {
                        1.0
                    };
                    let start = t * stride;
                    let end = (start + stride).min(v_data.len());
                    let token_len = end - start;
                    for i in 0..token_len {
                        let val = v_data[start + i];
                        // Symmetric 4-bit: map [-scale, scale] → [-7, 7]
                        let normalized = (val / token_scale).clamp(-1.0, 1.0);
                        let q = (normalized * 7.0).round() as i8; // -7..7
                        let q_u4 = (q + 7) as u8; // 0..14, but actually 0..14 range → pack into nibble
                        let byte_idx = (start + i) / 2;
                        let nibble_idx = (start + i) % 2;
                        if byte_idx < packed.len() {
                            if nibble_idx == 0 {
                                packed[byte_idx] = (packed[byte_idx] & 0xF0) | (q_u4 & 0x0F);
                            } else {
                                packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((q_u4 & 0x0F) << 4);
                            }
                        }
                    }
                }

                KiviQuantResult {
                    data: packed,
                    scales: scales.to_vec(),
                    bytes_per_element: 0, // 0.5 bytes per element (4-bit)
                    precision_tier: PrecisionTier::KIVI4,
                }
            }
            PrecisionTier::KIVI2 => {
                // 2-bit: 4 values per byte, symmetric signed [-1, 1]
                let elements_per_byte = 4;
                let total_bytes = (num_tokens * stride).div_ceil(elements_per_byte);
                let mut packed = vec![0u8; total_bytes];

                for t in 0..num_tokens {
                    let token_scale = if t < scales.len() && scales[t].abs() > 1e-8 {
                        scales[t]
                    } else {
                        1.0
                    };
                    let start = t * stride;
                    let end = (start + stride).min(v_data.len());
                    let token_len = end - start;
                    for i in 0..token_len {
                        let val = v_data[start + i];
                        let normalized = (val / token_scale).clamp(-1.0, 1.0);
                        let q = (normalized).round() as i8; // -1, 0, 1
                        let q_u2 = (q + 1) as u8; // 0..2, map to 2-bit
                        let global_idx = start + i;
                        let byte_idx = global_idx / 4;
                        let shift = (global_idx % 4) * 2;
                        if byte_idx < packed.len() {
                            packed[byte_idx] |= (q_u2 & 0x03) << shift;
                        }
                    }
                }

                KiviQuantResult {
                    data: packed,
                    scales: scales.to_vec(),
                    bytes_per_element: 0, // 0.25 bytes per element (2-bit)
                    precision_tier: PrecisionTier::KIVI2,
                }
            }
            PrecisionTier::FP16 => {
                // No quantization: store as FP16
                let total_bytes = num_tokens * stride * 2;
                let mut packed = vec![0u8; total_bytes];
                for (i, &val) in v_data.iter().take(num_tokens * stride).enumerate() {
                    let bits = super::f32_to_f16_bits(val);
                    let byte_offset = i * 2;
                    if byte_offset + 2 <= total_bytes {
                        packed[byte_offset] = bits as u8;
                        packed[byte_offset + 1] = (bits >> 8) as u8;
                    }
                }
                KiviQuantResult {
                    data: packed,
                    scales: Vec::new(),
                    bytes_per_element: 2,
                    precision_tier: PrecisionTier::FP16,
                }
            }
            _ => {
                // Fallback for unsupported V tiers: KIVI4
                let total_bytes = (num_tokens * stride).div_ceil(2);
                let packed = vec![0u8; total_bytes];
                KiviQuantResult {
                    data: packed,
                    scales: scales.to_vec(),
                    bytes_per_element: 0,
                    precision_tier: self.val_precision,
                }
            }
        }
    }

    /// Dequantize V cache back to f32 (read path).
    ///
    /// Reverses the per-token quantization applied in `quantize_v`.
    /// Converts low-precision INT4/INT2 values back to f32 compute precision.
    pub fn dequantize_v(
        &self,
        packed: &[u8],
        num_tokens: usize,
        num_kv_heads: usize,
        head_dim: usize,
        precision_tier: PrecisionTier,
    ) -> Vec<f32> {
        let stride = num_kv_heads * head_dim;
        let total_elements = num_tokens * stride;
        let mut out = vec![0.0f32; total_elements];

        match precision_tier {
            PrecisionTier::KIVI4 => {
                // 4-bit: unpack nibbles, dequantize with per-token scales
                for t in 0..num_tokens {
                    let token_scale = if t < self.v_token_scales.len()
                        && self.v_token_scales[t].abs() > 1e-8
                    {
                        self.v_token_scales[t]
                    } else {
                        1.0
                    };
                    let start = t * stride;
                    let end = (start + stride).min(total_elements);
                    for i in start..end {
                        let byte_idx = i / 2;
                        let nibble_idx = i % 2;
                        if byte_idx < packed.len() {
                            let nibble = if nibble_idx == 0 {
                                packed[byte_idx] & 0x0F
                            } else {
                                (packed[byte_idx] >> 4) & 0x0F
                            };
                            // Reverse: q_u4 in 0..14 → q in -7..7 → f32
                            let q = nibble as i8 - 7i8;
                            let normalized = q as f32 / 7.0;
                            out[i] = normalized * token_scale;
                        }
                    }
                }
            }
            PrecisionTier::KIVI2 => {
                // 2-bit: unpack 2-bit values, dequantize with per-token scales
                for t in 0..num_tokens {
                    let token_scale = if t < self.v_token_scales.len()
                        && self.v_token_scales[t].abs() > 1e-8
                    {
                        self.v_token_scales[t]
                    } else {
                        1.0
                    };
                    let start = t * stride;
                    let end = (start + stride).min(total_elements);
                    for i in start..end {
                        let byte_idx = i / 4;
                        let shift = (i % 4) * 2;
                        if byte_idx < packed.len() {
                            let bits = (packed[byte_idx] >> shift) & 0x03;
                            let q = bits as i8 - 1i8; // 0..2 → -1..1
                            let normalized = q as f32; // -1.0, 0.0, 1.0
                            out[i] = normalized * token_scale;
                        }
                    }
                }
            }
            PrecisionTier::FP16 => {
                // FP16: unpack f16 → f32
                for i in 0..total_elements {
                    let byte_offset = i * 2;
                    if byte_offset + 2 <= packed.len() {
                        let lo = packed[byte_offset] as u16;
                        let hi = packed[byte_offset + 1] as u16;
                        let bits = lo | (hi << 8);
                        out[i] = super::f16_bits_to_f32(bits);
                    }
                }
            }
            _ => {
                // Unsupported: return zeros
                out.fill(0.0);
            }
        }

        out
    }

    // ── Page-level integration ──

    /// Apply KIVI strategy to a page header.
    ///
    /// Sets the precision tier and V scale factor in the header
    /// to reflect the asymmetric K/V quantization.
    ///
    /// The page-level `precision_tier` tracks the *value* cache's precision
    /// (since K is always kept at higher precision). The `v_scale_factor`
    /// stores a quantized representation of the per-token scale range.
    pub fn apply_to_header(
        &self,
        header: &mut KvPageHeader,
        is_sink_page: bool,
    ) {
        if !self.enabled {
            header.set_precision_tier(PrecisionTier::FP16);
            header.v_scale_factor = 0;
            return;
        }

        if is_sink_page || header.has_sink_token() {
            // Sink pages: both K and V at FP16
            header.set_precision_tier(PrecisionTier::FP16);
            header.v_scale_factor = 0;
        } else {
            // Normal page: K at key_precision, V at val_precision
            // Page precision_tier represents V precision
            header.set_precision_tier(self.val_precision);

            // Encode max per-token scale into v_scale_factor (u8)
            let max_scale = self.v_token_scales.iter().cloned().fold(0.0f32, f32::max);
            header.v_scale_factor = encode_scale_to_u8(max_scale);
        }
    }

    /// Check if a token at the given index should be preserved at FP16
    /// (attention sink protection).
    #[inline]
    pub fn should_preserve_fp16(&self, token_idx: usize, is_sink: bool) -> bool {
        if !self.enabled {
            return true;
        }
        token_idx < self.sink_count && is_sink
    }

    /// Get the per-channel K scales for external use.
    pub fn k_scales(&self) -> &[f32] {
        &self.k_channel_scales
    }

    /// Get the per-token V scales for external use.
    pub fn v_scales(&self) -> &[f32] {
        &self.v_token_scales
    }

    /// Estimated compression ratio for V cache vs FP16 baseline.
    ///
    /// FP16 baseline: 2 bytes per element.
    /// KIVI4: 0.5 bytes per element → 4x compression.
    /// KIVI2: 0.25 bytes per element → 8x compression.
    pub fn v_compression_ratio(&self) -> f32 {
        match self.val_precision {
            PrecisionTier::KIVI2 => 8.0,
            PrecisionTier::KIVI4 => 4.0,
            PrecisionTier::FP8 => 2.0,
            _ => 1.0,
        }
    }

    /// Reset internal scale buffers for a new sequence.
    pub fn reset(&mut self) {
        self.k_channel_scales.clear();
        self.v_token_scales.clear();
    }
}

// ============================================================================
// KiviQuantResult — output of KIVI quantization
// ============================================================================

/// Result of quantizing K or V cache with KIVI strategy.
#[derive(Debug, Clone)]
pub struct KiviQuantResult {
    /// Packed quantized bytes
    pub data: Vec<u8>,
    /// Per-channel (K) or per-token (V) scales
    pub scales: Vec<f32>,
    /// Bytes per element (0 for sub-byte: KIVI4=0.5, KIVI2=0.25)
    pub bytes_per_element: usize,
    /// Precision tier used for this quantization
    pub precision_tier: PrecisionTier,
}

// ============================================================================
// MustafarStrategy — MUSTAFAR token retention (SPEC 19 §5)
// ============================================================================

/// MUSTAFAR token retention strategy (SPEC 19-KV-CACHE-OPTIMIZATION §5)
///
/// Identifies "must-have" tokens — tokens that are critical for generation
/// quality (first entity occurrences, special markers, high-attention tokens)
/// and ensures they are retained at higher precision during KV cache
/// compression and eviction.
///
/// ## Detection Heuristics
///
/// MUSTAFAR tokens are identified using three telemetry signals from the
/// Epilogue (zero-cost "free lunch"):
///
/// 1. **Attention concentration** — tokens with high softmax_max_avg and low
///    entropy_avg receive high concentration scores (sink-like behavior).
/// 2. **Head entropy spread** — large differences between head_entropy_max
///    and head_entropy_min indicate specialized attention patterns; such tokens
///    carry rich semantic information (entity first-mention, syntax markers).
/// 3. **Cross-layer delta** — low delta_rho_avg suggests stable representation
///    across layers; these tokens are likely structural rather than transient.
///
/// ## Retention Policy
///
/// Tokens classified as MUSTAFAR receive:
/// - **Precision protection**: floor at FP8 (never drop to KIVI2)
/// - **Eviction protection**: eviction priority set to maximum (retained longest)
/// - **Channel bitmap**: sparse heads marked in `channel_bitmap_lo` to guide
///   selective precision allocation (active heads stay FP16, inactive drop)
///
/// ## Integration with KiviStrategy
///
/// `MustafarStrategy` extends `KiviStrategy::should_preserve_fp16()`:
/// even beyond the attention sink window, MUSTAFAR tokens are preserved
/// at higher precision than normal tokens. The two strategies compose:
///
/// ```text
/// preserve = is_sink_token(index) || is_mustafar_token(index)
/// ```
///
/// ## Eviction Priority
///
/// Eviction priority is computed as a u32 score where **higher = retain longer**:
/// - MUSTAFAR tokens: priority = u32::MAX (never evict)
/// - Sink tokens: priority = 0xFF00_0000
/// - Normal tokens: priority = importance_score << 24 | (255 - tier_age) << 16
/// - Low-entropy tokens: priority = 0 (evict first)
#[derive(Debug, Clone)]
pub struct MustafarStrategy {
    /// Entropy spread threshold for MUSTAFAR detection (0-255).
    /// Tokens with `head_entropy_spread() > threshold` are MUSTAFAR candidates.
    pub entropy_spread_threshold: u8,

    /// Minimum importance score (0-255) to qualify as MUSTAFAR.
    /// Even with high entropy spread, tokens below this threshold are excluded.
    pub importance_threshold: u8,

    /// Whether MUSTAFAR detection is active.
    pub enabled: bool,

    /// Maximum number of MUSTAFAR tokens retained per sequence.
    /// Beyond this limit, lowest-importance MUSTAFAR tokens are demoted.
    pub max_mustafar_tokens: usize,

    /// Per-token importance scores (index = token position within page).
    importance_scores: Vec<u8>,

    /// Per-token MUSTAFAR flags (true = must retain at high precision).
    mustafar_flags: Vec<bool>,

    /// Per-token eviction priority (higher = retain longer, 0 = evict first).
    eviction_priority: Vec<u32>,

    /// Current token count for scoring.
    token_count: usize,
}

impl Default for MustafarStrategy {
    fn default() -> Self {
        Self {
            entropy_spread_threshold: 80,
            importance_threshold: 120,
            enabled: true,
            max_mustafar_tokens: 16,
            importance_scores: Vec::new(),
            mustafar_flags: Vec::new(),
            eviction_priority: Vec::new(),
            token_count: 0,
        }
    }
}

impl MustafarStrategy {
    /// Create a new MUSTAFAR strategy with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled MUSTAFAR strategy (no special retention).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Configure entropy spread threshold.
    pub fn with_entropy_threshold(mut self, threshold: u8) -> Self {
        self.entropy_spread_threshold = threshold;
        self
    }

    /// Configure importance threshold.
    pub fn with_importance_threshold(mut self, threshold: u8) -> Self {
        self.importance_threshold = threshold;
        self
    }

    /// Configure max MUSTAFAR tokens.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_mustafar_tokens = max;
        self
    }

    // ── Token Importance Scoring ──

    /// Compute importance score for a single token from KvPageHeader telemetry.
    ///
    /// Combines three Epilogue signals into a 0-255 score:
    ///
    /// - **attention_concentration** (0.0–1.0): `1 - entropy_avg / max_entropy`
    ///   High concentration = token captures focused attention.
    /// - **sink_bonus** (0 or 50): if softmax_max_avg exceeds SINK_THRESHOLD,
    ///   indicates token-level attention peak.
    /// - **head_spread_factor** (0.0–1.0): `head_entropy_spread / 255`
    ///   Large spread = diverse head usage = semantically rich token.
    /// - **stability** (0.0–1.0): `1 - delta_rho_avg`
    ///   Low delta = stable cross-layer representation = structural token.
    ///
    /// Weighted sum: `concentration*100 + sink_bonus + spread*80 + stability*25`
    #[inline]
    pub fn score_token_importance(&self, header: &KvPageHeader) -> u8 {
        if !self.enabled {
            return 0;
        }

        let entropy_avg = super::f16_bits_to_f32(header.entropy_avg);
        let softmax_max_avg = super::f16_bits_to_f32(header.softmax_max_avg);
        let delta_rho_avg = super::f16_bits_to_f32(header.delta_rho_avg);

        // Attention concentration: low entropy → high concentration
        let max_entropy = 6.93_f32;
        let concentration = 1.0 - (entropy_avg / max_entropy).min(1.0);

        // Sink bonus: high softmax peak → token is an attention sink
        let sink_bonus = if softmax_max_avg > 0.8 { 50.0 } else { 0.0 };

        // Head spread: large max-min gap → diverse head usage
        let head_spread = header.head_entropy_spread() as f32;
        let spread_factor = (head_spread / 255.0).min(1.0);

        // Stability: low delta_rho → stable cross-layer representation
        let stability = 1.0 - delta_rho_avg.min(1.0);

        // Dead ratio penalty: high dead_ratio → less important
        let dead_penalty = (header.dead_ratio as f32 / 255.0) * 30.0;

        let raw = concentration * 100.0 + sink_bonus + spread_factor * 80.0 + stability * 25.0 - dead_penalty;

        (raw.clamp(0.0, 255.0)) as u8
    }

    /// Score a batch of tokens and update internal tracking.
    ///
    /// Processes each header's telemetry, computes importance scores,
    /// and determines MUSTAFAR classification for the batch.
    ///
    /// # Arguments
    /// * `headers` - slice of KvPageHeader to score
    pub fn score_batch(&mut self, headers: &[KvPageHeader]) {
        let n = headers.len();
        self.importance_scores.resize(n, 0u8);
        self.mustafar_flags.resize(n, false);
        self.eviction_priority.resize(n, 0u32);
        self.token_count = n;

        if n == 0 || !self.enabled {
            return;
        }

        // Phase 1: compute raw importance scores
        for (i, header) in headers.iter().enumerate() {
            self.importance_scores[i] = self.score_token_importance(header);
        }

        // Phase 2: classify MUSTAFAR tokens
        // Criteria: importance >= threshold AND entropy_spread >= threshold
        let mut mustafar_candidates: Vec<(usize, u8)> = (0..n)
            .filter_map(|i| {
                let header = &headers[i];
                let importance = self.importance_scores[i];
                let spread = header.head_entropy_spread();
                if importance >= self.importance_threshold && spread >= self.entropy_spread_threshold {
                    Some((i, importance))
                } else {
                    None
                }
            })
            .collect();

        // Sort by importance descending, take top max_mustafar_tokens
        mustafar_candidates.sort_by(|a, b| b.1.cmp(&a.1));
        let keep_count = mustafar_candidates.len().min(self.max_mustafar_tokens);

        for &(idx, _) in &mustafar_candidates[..keep_count] {
            self.mustafar_flags[idx] = true;
        }

        // Phase 3: compute eviction priority
        for i in 0..n {
            self.eviction_priority[i] = self.compute_eviction_priority(i, &headers[i]);
        }
    }

    /// Check if a token at the given index is classified as MUSTAFAR.
    #[inline]
    pub fn is_mustafar(&self, idx: usize) -> bool {
        if !self.enabled {
            return false;
        }
        idx < self.mustafar_flags.len() && self.mustafar_flags[idx]
    }

    /// Get the importance score for a token.
    #[inline]
    pub fn importance(&self, idx: usize) -> u8 {
        match self.importance_scores.get(idx) {
            Some(&s) => s,
            None => 0,
        }
    }

    /// Get the eviction priority for a token (higher = retain longer).
    #[inline]
    pub fn eviction_priority_for(&self, idx: usize) -> u32 {
        match self.eviction_priority.get(idx) {
            Some(&p) => p,
            None => 0,
        }
    }

    /// Check if the token at `idx` should be retained at high precision.
    ///
    /// Returns true for MUSTAFAR tokens. Composes with
    /// `KiviStrategy::should_preserve_fp16()` for sink token coverage.
    #[inline]
    pub fn should_retain(&self, idx: usize) -> bool {
        self.is_mustafar(idx)
    }

    // ── Eviction Priority ──

    /// Compute eviction priority for a single token.
    ///
    /// Priority formula (higher = retain longer):
    ///
    /// | Token Class      | Priority Range      |
    /// |-----------------|---------------------|
    /// | MUSTAFAR         | 0xFFFF_0000 – MAX  |
    /// | Sink             | 0xFF00_0000 – FFFF |
    /// | Normal (high imp)| 0x8000_0000 – FEFF |
    /// | Normal (low imp) | 0x0000_0000 – 7FFF |
    /// | Dead/low entropy | 0x0000_0000         |
    fn compute_eviction_priority(&self, idx: usize, header: &KvPageHeader) -> u32 {
        if self.is_mustafar(idx) {
            // MUSTAFAR: absolute top priority, never evict
            let importance = self.importance_scores[idx] as u32;
            return 0xFFFF_0000u32 | importance << 8 | importance;
        }

        let importance = self.importance_scores[idx] as u32;

        // Sink tokens: high priority
        let softmax_max = super::f16_bits_to_f32(header.softmax_max_avg);
        let is_sink = softmax_max > 0.8;

        if is_sink {
            return 0xFF00_0000u32 | (importance << 16) | (importance);
        }

        // Dead tokens: lowest priority
        if header.dead_ratio > 200 || header.is_low_entropy() {
            return 0;
        }

        // Normal tokens: importance-weighted, aged by tier_age
        let age_factor = (header.tier_age as u32).min(255);
        let age_penalty = age_factor << 8;
        (importance << 24).saturating_sub(age_penalty)
    }

    // ── Channel Bitmap ──

    /// Compute sparse channel bitmap for a page header.
    ///
    /// Based on `head_entropy_spread()`: when head-to-head entropy variation
    /// is large, some heads contribute little and can be marked as inactive
    /// in `channel_bitmap_lo`.
    ///
    /// # Heuristic
    ///
    /// Since KvPageHeader stores only `head_entropy_max` and `head_entropy_min`
    /// (not per-head values), we use a distribution heuristic:
    ///
    /// - Map `num_kv_heads` heads uniformly across the [min, max] range
    /// - Heads falling below the midpoint are marked inactive (bit = 0)
    /// - Heads at or above midpoint are active (bit = 1)
    ///
    /// # Returns
    /// u32 bitmap where bit `h` = 1 means head `h` channels are active.
    pub fn compute_channel_bitmap(&self, header: &KvPageHeader, num_kv_heads: usize) -> u32 {
        if !self.enabled || num_kv_heads == 0 {
            return 0xFFFF_FFFF;
        }

        let spread = header.head_entropy_spread();
        if spread < self.entropy_spread_threshold {
            // Low spread: all heads are similarly active, no sparsity benefit
            return 0xFFFF_FFFF;
        }

        let min_val = header.head_entropy_min as f32;
        let max_val = header.head_entropy_max as f32;
        if max_val <= min_val {
            return 0xFFFF_FFFF;
        }

        let range = max_val - min_val;
        let midpoint = min_val + range * 0.5;
        let head_count = num_kv_heads.min(32);

        let mut bitmap: u32 = 0;
        for h in 0..head_count {
            // Map head index to entropy value: linear interpolation
            let t = if head_count > 1 {
                h as f32 / (head_count as f32 - 1.0)
            } else {
                0.5
            };
            let entropy_est = min_val + t * range;
            if entropy_est >= midpoint {
                bitmap |= 1u32 << h;
            }
        }

        // Ensure at least one head is active
        if bitmap == 0 {
            bitmap = 0x0000_0001;
        }

        bitmap
    }

    /// Apply MUSTAFAR decisions to a page header.
    ///
    /// Updates `importance_score`, `channel_bitmap_lo`, and `deopt_flags`
    /// based on MUSTAFAR classification.
    ///
    /// # Arguments
    /// * `header` - page header to update
    /// * `token_idx` - token index within the batch (used to look up scores)
    /// * `num_kv_heads` - number of KV heads for channel bitmap
    pub fn apply_to_header(
        &self,
        header: &mut KvPageHeader,
        token_idx: usize,
        num_kv_heads: usize,
    ) {
        if !self.enabled {
            return;
        }

        // Write importance score
        header.importance_score = self.importance(token_idx);

        // Compute and write channel bitmap
        let bitmap = self.compute_channel_bitmap(header, num_kv_heads);
        header.channel_bitmap_lo = bitmap;

        // Mark MUSTAFAR tokens for requantize protection
        if self.is_mustafar(token_idx) {
            // Set deopt bit 1 to signal that this token's precision should not
            // be downgraded below FP8
            header.deopt_flags |= 0x02;
        }
    }

    /// Determine the recommended precision tier for a token, considering
    /// MUSTAFAR retention policy.
    ///
    /// MUSTAFAR tokens floor at FP8 (never drop to KIVI2).
    /// Sink tokens retain FP16.
    /// Normal tokens use the caller's default tier.
    ///
    /// # Returns
    /// `None` if no special tier is required (caller should use default),
    /// `Some(tier)` if MUSTAFAR policy mandates a specific floor.
    pub fn precision_floor(&self, token_idx: usize, header: &KvPageHeader) -> Option<PrecisionTier> {
        if !self.enabled {
            return None;
        }

        let softmax_max = super::f16_bits_to_f32(header.softmax_max_avg);

        if softmax_max > 0.8 {
            // Sink token: FP16
            return Some(PrecisionTier::FP16);
        }

        if self.is_mustafar(token_idx) {
            // MUSTAFAR token: at least FP8
            return Some(PrecisionTier::FP8);
        }

        None
    }

    /// Get the number of MUSTAFAR tokens in the current batch.
    pub fn mustafar_count(&self) -> usize {
        self.mustafar_flags.iter().filter(|&&f| f).count()
    }

    /// Get all importance scores for the current batch.
    pub fn importance_scores(&self) -> &[u8] {
        &self.importance_scores
    }

    /// Get all MUSTAFAR flags for the current batch.
    pub fn mustafar_flags(&self) -> &[bool] {
        &self.mustafar_flags
    }

    /// Get all eviction priorities for the current batch.
    pub fn eviction_priorities(&self) -> &[u32] {
        &self.eviction_priority
    }

    /// Reset internal state for a new sequence.
    pub fn reset(&mut self) {
        self.importance_scores.clear();
        self.mustafar_flags.clear();
        self.eviction_priority.clear();
        self.token_count = 0;
    }
}

// ============================================================================
