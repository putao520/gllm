// KvTunerStrategy — dynamic precision adjustment (SPEC 19 §4)
// ============================================================================

/// KVTuner dynamic precision adjustment strategy (SPEC 19 §4, SPEC §3.2)
///
/// Monitors per-page importance scores (pre-computed by Epilogue telemetry
/// and stored in `KvPageHeader.importance_score`) and dynamically adjusts
/// per-layer K/V precision configuration per SPEC §3.2 tier decision rules.
///
/// ## Tier Decision (importance_score based)
///
/// | Score Range     | Target Tier | Meaning                          |
/// |-----------------|-------------|----------------------------------|
/// | >200 or sink    | FP16        | High importance / sink → full    |
/// | 150-200         | FP8         | Medium-high importance → 8-bit   |
/// | 80-150          | KIVI4       | Normal → 4-bit (default)         |
/// | 40-80           | KIVI2       | Low importance → 2-bit           |
/// | 15-40           | Sparse      | Very low → sparse                |
/// | ≤15             | Evicted     | Extremely low → eviction         |
///
/// ## Layer-Aware Modulation (SPEC §3.2)
///
/// | Layer Depth     | V Tier Floor | Rationale                        |
/// |-----------------|-------------|----------------------------------|
/// | [0..L/3)        | FP8         | Understanding layers are sensitive|
/// | [L/3..2L/3)     | KIVI4       | Middle layers, moderate tolerance |
/// | [2L/3..L]       | None        | Generation layers are insensitive |
///
/// ## Telemetry
///
/// Each precision adjustment generates a `KvTunerEvent` recorded in an
/// internal ring buffer for observability. Events track the adjustment
/// direction, triggering importance score, and reason.
#[derive(Debug, Clone)]
pub struct KvTunerStrategy {
    /// Base KIVI strategy being tuned by entropy observations
    pub kivi: KiviStrategy,
    /// Smoothed attention entropy via EMA (normalized [0, 1])
    smoothed_entropy: f32,
    /// EMA decay factor (0.0–1.0], higher = more responsive
    ema_alpha: f32,
    /// Monotonic counter of precision adjustments performed
    adjustment_count: u64,
    /// Sequence length threshold for long-sequence downgrade trigger
    long_seq_threshold: usize,
    /// Precision adjustment event ring buffer
    events: Vec<KvTunerEvent>,
    /// Maximum events retained (ring buffer cap)
    max_events: usize,
    /// Whether dynamic tuning is active
    pub enabled: bool,
    /// Minimum precision tier floor (never tune below this)
    pub min_tier: PrecisionTier,
    /// Maximum precision tier ceiling (never tune above this)
    pub max_tier: PrecisionTier,
    /// Most recent adjustment reason
    last_reason: KvTunerReason,
}

/// Precision adjustment event for KVTuner telemetry (SPEC 19 §4)
#[derive(Debug, Clone)]
pub struct KvTunerEvent {
    /// Monotonic sequence number of this adjustment
    pub seq: u64,
    /// Previous key precision tier
    pub from_k_tier: PrecisionTier,
    /// New key precision tier after adjustment
    pub to_k_tier: PrecisionTier,
    /// Previous value precision tier
    pub from_v_tier: PrecisionTier,
    /// New value precision tier after adjustment
    pub to_v_tier: PrecisionTier,
    /// Importance score that triggered this adjustment (normalized [0, 1])
    pub entropy: f32,
    /// Reason for the adjustment
    pub reason: KvTunerReason,
}

/// Reason for a KVTuner precision adjustment (SPEC 19 §4)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTunerReason {
    /// Low importance score (≤40): attention spread uniform, safe to downgrade
    HighEntropyDowngrade,
    /// High importance score (>150): concentrated attention, upgrade precision
    LowEntropyUpgrade,
    /// Long sequence (>threshold): redundant late tokens, downgrade one tier
    LongSeqDowngrade,
    /// Sink token detected: lock both K/V to FP16 unconditionally
    SinkProtection,
    /// Constrained by tier floor or ceiling limits
    TierFloorConstraint,
    /// Initial calibration (before first entropy observation)
    InitialCalibration,
    /// No change needed (entropy in stable mid-range)
    Stable,
}

/// Numeric rank for tier comparison (higher = more precise).
const fn tuner_tier_rank(tier: PrecisionTier) -> u8 {
    match tier {
        PrecisionTier::Evicted => 0,
        PrecisionTier::Dictionary => 1,
        PrecisionTier::Sparse => 2,
        PrecisionTier::KIVI2 => 3,
        PrecisionTier::KIVI4 => 4,
        PrecisionTier::FP8 => 5,
        PrecisionTier::FP16 => 6,
    }
}

/// Clamp a tier between floor and ceiling (inclusive).
fn clamp_tier(tier: PrecisionTier, floor: PrecisionTier, ceiling: PrecisionTier) -> PrecisionTier {
    let r = tuner_tier_rank(tier);
    let r_floor = tuner_tier_rank(floor);
    let r_ceil = tuner_tier_rank(ceiling);
    if r < r_floor {
        floor
    } else if r > r_ceil {
        ceiling
    } else {
        tier
    }
}

impl Default for KvTunerStrategy {
    fn default() -> Self {
        Self {
            kivi: KiviStrategy::default(),
            smoothed_entropy: 0.5,
            ema_alpha: 0.1,
            adjustment_count: 0,
            long_seq_threshold: 4096,
            events: Vec::with_capacity(256),
            max_events: 256,
            enabled: true,
            min_tier: PrecisionTier::KIVI2,
            max_tier: PrecisionTier::FP16,
            last_reason: KvTunerReason::InitialCalibration,
        }
    }
}

impl KvTunerStrategy {
    /// Create a new KVTuner with default KIVI strategy and tuning parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a KVTuner wrapping an existing KiviStrategy.
    pub fn with_kivi(kivi: KiviStrategy) -> Self {
        Self {
            kivi,
            ..Default::default()
        }
    }

    /// Create a disabled tuner (always returns current KIVI tiers unchanged).
    pub fn disabled() -> Self {
        Self {
            kivi: KiviStrategy::disabled(),
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the EMA decay factor (α). Clamped to [0.01, 1.0].
    /// Higher α makes the tuner more responsive to recent entropy changes.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha.clamp(0.01, 1.0);
        self
    }

    /// Set the long sequence threshold for triggering downgrades.
    pub fn with_long_seq_threshold(mut self, threshold: usize) -> Self {
        self.long_seq_threshold = threshold;
        self
    }

    /// Set the minimum and maximum tier bounds.
    pub fn with_tier_bounds(mut self, min: PrecisionTier, max: PrecisionTier) -> Self {
        self.min_tier = min;
        self.max_tier = max;
        self
    }

    /// Feed a raw entropy observation (normalized [0, 1]) into the EMA.
    ///
    /// Returns the updated smoothed entropy value.
    pub fn observe_entropy(&mut self, raw_entropy: f32) -> f32 {
        let clamped = raw_entropy.clamp(0.0, 1.0);
        if self.adjustment_count == 0 && self.smoothed_entropy == 0.5 {
            // First observation: initialize directly
            self.smoothed_entropy = clamped;
        } else {
            // EMA: s_new = α × raw + (1 − α) × s_old
            self.smoothed_entropy =
                self.ema_alpha * clamped + (1.0 - self.ema_alpha) * self.smoothed_entropy;
        }
        self.smoothed_entropy
    }

    /// Observe entropy from a KvPageHeader's Epilogue telemetry.
    ///
    /// Decodes the `entropy_avg` field (f16 bits → f32) and normalizes
    /// to [0, 1] using the theoretical max entropy ln(head_dim).
    pub fn observe_from_header(
        &mut self,
        header: &KvPageHeader,
        head_dim: usize,
    ) -> f32 {
        let raw = super::f16_bits_to_f32(header.entropy_avg);
        // Normalize: maximum entropy for attention over head_dim logits is ln(head_dim)
        let max_entropy = (head_dim as f32).ln();
        let normalized = if max_entropy > 0.0 {
            (raw / max_entropy).min(1.0)
        } else {
            0.5
        };
        self.observe_entropy(normalized)
    }

    /// Get the current smoothed entropy (normalized [0, 1]).
    #[inline]
    pub fn entropy(&self) -> f32 {
        self.smoothed_entropy
    }

    /// Get the number of precision adjustments performed.
    #[inline]
    pub fn adjustment_count(&self) -> u64 {
        self.adjustment_count
    }

    /// Get the most recent adjustment reason.
    #[inline]
    pub fn last_reason(&self) -> KvTunerReason {
        self.last_reason
    }

    /// Get a reference to the adjustment event log.
    #[inline]
    pub fn events(&self) -> &[KvTunerEvent] {
        &self.events
    }

    /// Drain and return all recorded events (for telemetry export).
    pub fn drain_events(&mut self) -> Vec<KvTunerEvent> {
        std::mem::take(&mut self.events)
    }

    // ── Core Adjustment Logic ──

    /// Adjust KIVI precision tiers based on importance score (SPEC §3.2).
    ///
    /// # Arguments
    /// * `importance_score` — Epilogue-computed importance score [0, 255]
    /// * `is_sink` — Whether the current page contains sink tokens
    /// * `layer_depth_ratio` — Layer position [0.0=first, 1.0=last]
    /// * `seq_len` — Current sequence length (tokens processed)
    ///
    /// # Returns
    /// `(recommended_k_tier, recommended_v_tier)` — the adjusted precision
    /// tiers. If tuning made no change, returns current tiers unchanged.
    ///
    /// # Side effects
    /// Updates the internal `kivi.key_precision` / `kivi.val_precision` and
    /// records a `KvTunerEvent` when tiers change.
    pub fn adjust_precision(
        &mut self,
        importance_score: u8,
        is_sink: bool,
        layer_depth_ratio: f32,
        _seq_len: usize,
    ) -> (PrecisionTier, PrecisionTier) {
        if !self.enabled {
            return (self.kivi.key_precision, self.kivi.val_precision);
        }

        let current_k = self.kivi.key_precision;
        let current_v = self.kivi.val_precision;

        // SPEC §3.2: importance-score-based tier decision
        let (target_k, target_v, reason) = if is_sink {
            // SPEC §3.2: Sink tokens must remain FP16 unconditionally
            (PrecisionTier::FP16, PrecisionTier::FP16, KvTunerReason::SinkProtection)
        } else if importance_score > 200 {
            // High importance / sink → full precision
            (PrecisionTier::FP16, PrecisionTier::FP16, KvTunerReason::LowEntropyUpgrade)
        } else if importance_score > 150 {
            // Medium-high importance → 8-bit
            (PrecisionTier::FP16, PrecisionTier::FP8, KvTunerReason::LowEntropyUpgrade)
        } else if importance_score > 80 {
            // Normal → 4-bit (default tier)
            (PrecisionTier::FP16, PrecisionTier::KIVI4, KvTunerReason::Stable)
        } else if importance_score > 40 {
            // Low importance → 2-bit aggressive compression
            (PrecisionTier::FP16, PrecisionTier::KIVI2, KvTunerReason::HighEntropyDowngrade)
        } else if importance_score > 15 {
            // Very low → sparse representation
            (PrecisionTier::FP8, PrecisionTier::Sparse, KvTunerReason::HighEntropyDowngrade)
        } else {
            // Extremely low → eviction candidate
            (PrecisionTier::FP8, PrecisionTier::Evicted, KvTunerReason::HighEntropyDowngrade)
        };

        // SPEC §3.2 Layer-aware modulation: apply depth-based floors
        let (layer_k, layer_v) = if layer_depth_ratio < 1.0 / 3.0 {
            // Shallow layers [0..L/3): tier cannot be below FP8 (understanding layers)
            let k_floor = clamp_tier(target_k, PrecisionTier::FP8, self.max_tier);
            let v_floor = clamp_tier(target_v, PrecisionTier::FP8, self.max_tier);
            (k_floor, v_floor)
        } else if layer_depth_ratio < 2.0 / 3.0 {
            // Middle layers [L/3..2L/3): tier cannot be below KIVI4
            let k_floor = clamp_tier(target_k, self.min_tier, self.max_tier);
            let v_floor = clamp_tier(target_v, PrecisionTier::KIVI4, self.max_tier);
            (k_floor, v_floor)
        } else {
            // Deep layers [2L/3..L]: no floor (generation layers are insensitive)
            let k_floor = clamp_tier(target_k, self.min_tier, self.max_tier);
            let v_floor = clamp_tier(target_v, self.min_tier, self.max_tier);
            (k_floor, v_floor)
        };

        let actual_reason = if layer_k != target_k || layer_v != target_v {
            KvTunerReason::TierFloorConstraint
        } else {
            reason
        };

        // Record event if tiers changed
        if layer_k != current_k || layer_v != current_v {
            self.adjustment_count = self.adjustment_count.wrapping_add(1);
            self.record_event(current_k, layer_k, current_v, layer_v, importance_score as f32 / 255.0, actual_reason);
            self.kivi.key_precision = layer_k;
            self.kivi.val_precision = layer_v;
        }

        self.last_reason = actual_reason;
        (layer_k, layer_v)
    }

    /// Apply KVTuner adjustment and also update the KvPageHeader's
    /// precision tier field (writes `v` tier to header per KIVI convention).
    ///
    /// Reads `importance_score` from the header (pre-computed by Epilogue
    /// or MUSTAFAR scoring) to drive the SPEC §3.2 tier decision.
    ///
    /// # Returns
    /// `(k_tier, v_tier)` — the newly assigned tiers, and whether the page
    /// tier was modified (header updated).
    pub fn adjust_and_apply(
        &mut self,
        header: &mut KvPageHeader,
        seq_len: usize,
        is_sink: bool,
        layer_depth_ratio: f32,
    ) -> (PrecisionTier, PrecisionTier, bool) {
        let importance_score = header.importance_score;
        let (k_tier, v_tier) = self.adjust_precision(importance_score, is_sink, layer_depth_ratio, seq_len);

        let old_page_tier = header.precision_tier();
        // Page header tracks V precision (per KIVI convention)
        let page_changed = old_page_tier != v_tier;
        if page_changed {
            header.set_precision_tier(v_tier);
            header.deopt_flags |= 0x01; // mark for requantize
        }

        (k_tier, v_tier, page_changed)
    }

    /// One-shot version: read importance_score from header, then adjust.
    ///
    /// Convenience wrapper combining importance_score read + adjust_precision.
    /// Reads the pre-computed `importance_score` from the header (written by
    /// Epilogue or MUSTAFAR scoring) rather than computing raw entropy.
    pub fn observe_and_adjust(
        &mut self,
        header: &KvPageHeader,
        _head_dim: usize,
        seq_len: usize,
        is_sink: bool,
        layer_depth_ratio: f32,
    ) -> (PrecisionTier, PrecisionTier) {
        let importance_score = header.importance_score;
        self.adjust_precision(importance_score, is_sink, layer_depth_ratio, seq_len)
    }

    // ── Event recording ──

    fn record_event(
        &mut self,
        from_k: PrecisionTier,
        to_k: PrecisionTier,
        from_v: PrecisionTier,
        to_v: PrecisionTier,
        entropy: f32,
        reason: KvTunerReason,
    ) {
        let event = KvTunerEvent {
            seq: self.adjustment_count,
            from_k_tier: from_k,
            to_k_tier: to_k,
            from_v_tier: from_v,
            to_v_tier: to_v,
            entropy,
            reason,
        };
        // Ring buffer: evict oldest if at capacity
        if self.events.len() >= self.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    // ── Bulk operations ──

    /// Adjust precision for a batch of page headers using each header's
    /// pre-computed importance_score. Each page's tier is determined by its
    /// own importance score, sink status, and layer depth.
    ///
    /// Returns the count of pages whose tier was modified.
    pub fn adjust_batch(
        &mut self,
        headers: &mut [KvPageHeader],
        seq_len: usize,
        layer_depth_ratio: f32,
    ) -> usize {
        let mut changed = 0usize;
        for header in headers.iter_mut() {
            if !header.is_active() {
                continue;
            }
            let is_sink = header.has_sink_token();
            let (_k, _v, page_changed) =
                self.adjust_and_apply(header, seq_len, is_sink, layer_depth_ratio);
            if page_changed {
                changed += 1;
            }
        }
        changed
    }

    // ── Reset / reinit ──

    /// Reset the tuner state: clear entropy smoothing, event log,
    /// and adjustment counter. KIVI strategy is reset as well.
    pub fn reset(&mut self) {
        self.kivi.reset();
        self.smoothed_entropy = 0.5;
        self.adjustment_count = 0;
        self.events.clear();
        self.last_reason = KvTunerReason::InitialCalibration;
    }

    /// Reset only the entropy smoothing state (keep KIVI and event history).
    pub fn reset_entropy(&mut self) {
        self.smoothed_entropy = 0.5;
    }

    /// Estimated current compression ratio for V cache vs FP16 baseline
    /// based on the dynamically adjusted val_precision tier.
    #[inline]
    pub fn current_v_compression_ratio(&self) -> f32 {
        self.kivi.v_compression_ratio()
    }
}

/// Type alias for LSP validation — `KVTuner` resolves to `KvTunerStrategy`.
pub type KVTuner = KvTunerStrategy;

/// Downgrade a precision tier by one step (toward lower precision).
/// Evicted is the floor — cannot downgrade further.
fn downgrade_one_tier(tier: PrecisionTier) -> PrecisionTier {
    match tier {
        PrecisionTier::FP16 => PrecisionTier::FP8,
        PrecisionTier::FP8 => PrecisionTier::KIVI4,
        PrecisionTier::KIVI4 => PrecisionTier::KIVI2,
        PrecisionTier::KIVI2 => PrecisionTier::Sparse,
        PrecisionTier::Sparse => PrecisionTier::Evicted,
        _ => tier, // Dictionary, Evicted: no further downgrade
    }
}

/// Upgrade a precision tier by one step (toward higher precision).
/// FP16 is the ceiling — cannot upgrade further.
#[allow(dead_code)]
fn upgrade_one_tier(tier: PrecisionTier) -> PrecisionTier {
    match tier {
        PrecisionTier::Evicted => PrecisionTier::Sparse,
        PrecisionTier::Sparse => PrecisionTier::KIVI2,
        PrecisionTier::KIVI2 => PrecisionTier::KIVI4,
        PrecisionTier::KIVI4 => PrecisionTier::FP8,
        PrecisionTier::FP8 => PrecisionTier::FP16,
        _ => tier, // FP16, Dictionary: no further upgrade
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Encode a f32 scale value into u8 for KvPageHeader.v_scale_factor.
///
/// Uses a log2 encoding to cover a wide dynamic range:
/// - 0: scale = 0 (no quantization / FP16)
/// - 1..254: scale = 2^((v - 128) / 16), range ~2^-8 to 2^7.9
/// - 255: scale > 2^7.9 (clamped)
fn encode_scale_to_u8(scale: f32) -> u8 {
    if scale <= 0.0 {
        return 0;
    }
    // log2 encoding: v = (log2(scale) * 10 + 128).clamp(1, 254)
    // Range: 2^(-12.7) ≈ 0.000118 to 2^(12.6) ≈ 6198
    let log2 = scale.log2();
    let encoded = (log2 * 10.0 + 128.0).round() as i32;
    encoded.clamp(1, 254) as u8
}

/// Decode a u8 scale factor back to f32.
#[allow(dead_code)]
fn decode_scale_from_u8(encoded: u8) -> f32 {
    if encoded == 0 {
        return 0.0;
    }
    let log2 = (encoded as f32 - 128.0) / 10.0;
    2.0f32.powf(log2)
}

// ============================================================================
// ChunkKvStrategy — block-level KV cache organization (SPEC 19 §6)
// ============================================================================

/// Chunk residency state for migration tracking.
///
/// Tracks whether a chunk is in GPU VRAM (Resident), has been swapped
/// to CPU RAM (Evicted), or is an unoccupied slot (Empty).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkResidency {
    /// Chunk is in GPU VRAM, ready for access
    Resident,
    /// Chunk has been evicted to CPU RAM; must be restored before access
    Evicted,
    /// Chunk slot is unoccupied
    Empty,
}

/// Per-chunk metadata for block-level KV cache management.
///
/// Each chunk tracks its token range, residency state, compression tier,
/// and access timestamps for LRU eviction decisions.
#[derive(Debug, Clone)]
pub struct ChunkInfo {
    /// Monotonic chunk index (0 = oldest / sink chunk)
    pub chunk_id: usize,
    /// Starting token index within the sequence (inclusive)
    pub token_start: usize,
    /// Number of tokens in this chunk
    pub token_count: usize,
    /// Current residency state
    pub residency: ChunkResidency,
    /// Approximate byte size (K + V combined)
    pub size_bytes: usize,
    /// Compression tier applied to this chunk
    pub compression_tier: PrecisionTier,
    /// Generation counter — incremented on each write to this chunk
    pub generation: u64,
    /// Last access timestamp (for LRU eviction ordering)
    pub last_access_ts: u64,
}

/// Chunk-level compression strategy for tier assignment.
///
/// Controls how compression tiers are distributed across chunks:
/// - **Uniform**: all chunks share the same tier (from KIVI strategy).
/// - **Tiered**: older chunks receive progressively lower precision.
/// - **Adaptive**: sink chunks preserved at FP16, recent chunks at current
///   tier, intermediate chunks degraded by age ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkCompressStrategy {
    /// All chunks use the same tier (set by KIVI strategy)
    Uniform,
    /// Older chunks use progressively lower precision
    Tiered,
    /// Sink chunks at FP16, recent at current tier, others degraded by age
    Adaptive,
}

/// ChunkKV block-level KV cache strategy (SPEC 19-KV-CACHE-OPTIMIZATION §6).
///
/// Organizes KV cache tokens into fixed-size blocks ("chunks") for block-level
/// compression, eviction, and migration. Operating on chunks rather than
/// individual tokens reduces fine-grained management overhead and enables
/// efficient I/O batching for swap-in/swap-out.
///
/// ## Chunk Layout
///
/// Tokens are partitioned into chunks of `chunk_size` tokens. The final chunk
/// may be partial (fewer than `chunk_size` tokens). Each chunk carries its own
/// metadata — token range, residency, compression tier, and LRU timestamp.
///
/// ## Block-Level Compression
///
/// Instead of compressing individual token pages, ChunkKV compresses entire
/// chunks at once. Within a chunk, scale factors can be shared across tokens,
/// dense packing improves memory locality, and batch dequantization reduces
/// per-token dispatch overhead.
///
/// ## Migration / Swap
///
/// When GPU memory is constrained, chunks are evicted to CPU RAM (host) and
/// restored on demand. The migration interface selects eviction candidates
/// via LRU ordering (oldest access first), always preserving the sink chunk
/// (chunk 0) as the last to evict.
///
/// ## Integration with KiviStrategy
///
/// `ChunkKvStrategy` determines *which* compression tier applies per chunk;
/// `KiviStrategy` performs the actual per-element quantization within each
/// chunk. The two compose: chunk → tier, tier → quantize/dequantize.
#[derive(Debug, Clone)]
pub struct ChunkKvStrategy {
    /// Number of tokens per chunk (default: 64)
    pub chunk_size: usize,
    /// Maximum chunks allowed resident in GPU memory (0 = unlimited)
    pub max_resident_chunks: usize,
    /// Whether chunk-based management is active
    pub enabled: bool,
    /// Compression strategy for tier assignment across chunks
    pub compress_strategy: ChunkCompressStrategy,
    /// Per-chunk metadata (index = chunk_id)
    chunks: Vec<ChunkInfo>,
    /// Total tokens tracked across all chunks
    total_tokens: usize,
    /// Monotonic access timestamp counter (incremented on touch)
    clock: u64,
    /// Cached chunk layout: (token_start, token_count) per chunk
    chunk_layout: Vec<(usize, usize)>,
}

impl Default for ChunkKvStrategy {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            max_resident_chunks: 0,
            enabled: true,
            compress_strategy: ChunkCompressStrategy::Adaptive,
            chunks: Vec::new(),
            total_tokens: 0,
            clock: 0,
            chunk_layout: Vec::new(),
        }
    }
}

impl ChunkKvStrategy {
    /// Create a new ChunkKV strategy with default parameters.
    ///
    /// Default: 64 tokens per chunk, unlimited resident chunks,
    /// Adaptive compression strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled ChunkKV strategy (all tokens treated as one chunk,
    /// no block-level management).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the chunk size in tokens.
    ///
    /// # Panics
    /// If `size` is 0 (chunk size must be at least 1).
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        assert!(size > 0, "chunk_size must be >= 1");
        self.chunk_size = size;
        self
    }

    /// Set the maximum number of resident chunks (0 = unlimited).
    pub fn with_max_resident(mut self, max: usize) -> Self {
        self.max_resident_chunks = max;
        self
    }

    /// Set the chunk-level compression strategy.
    pub fn with_compress_strategy(mut self, strategy: ChunkCompressStrategy) -> Self {
        self.compress_strategy = strategy;
        self
    }

    // ── Chunk Layout Computation ──

    /// Compute chunk layout for a sequence of `total_tokens`.
    ///
    /// Partitions tokens into `ceil(total_tokens / chunk_size)` chunks,
    /// each of `chunk_size` tokens except possibly the last. Returns a
    /// cached reference to the layout; subsequent calls with the same
    /// `total_tokens` return the cached result.
    ///
    /// # Returns
    /// Slice of `(token_start, token_count)` pairs, one per chunk.
    pub fn compute_chunk_layout(&mut self, total_tokens: usize) -> &[(usize, usize)] {
        self.total_tokens = total_tokens;
        self.chunk_layout.clear();

        if total_tokens == 0 || self.chunk_size == 0 {
            return &self.chunk_layout;
        }

        let num_chunks = total_tokens.div_ceil(self.chunk_size);
        for i in 0..num_chunks {
            let start = i * self.chunk_size;
            let count = if start + self.chunk_size <= total_tokens {
                self.chunk_size
            } else {
                total_tokens - start
            };
            self.chunk_layout.push((start, count));
        }

        &self.chunk_layout
    }

    /// Return the chunk index for a token position.
    ///
    /// Uses integer division: `token_idx / chunk_size`.
    #[inline]
    pub fn chunk_for_token(&self, token_idx: usize) -> usize {
        if self.chunk_size == 0 {
            return 0;
        }
        token_idx / self.chunk_size
    }

    /// Get the half-open token range `[start, end)` for a chunk.
    ///
    /// Returns `None` if `chunk_idx` is out of bounds.
    pub fn chunk_range(&self, chunk_idx: usize) -> Option<(usize, usize)> {
        self.chunk_layout
            .get(chunk_idx)
            .map(|&(s, c)| (s, s + c))
    }

    /// Number of chunks in the current layout.
    #[inline]
    pub fn num_chunks(&self) -> usize {
        self.chunk_layout.len()
    }

    /// Total tokens in the current layout.
    #[inline]
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    // ── Chunk Metadata Management ──

    /// Initialize per-chunk metadata from the current layout.
    ///
    /// All chunks start as `Resident` with `FP16` compression tier and
    /// zero size/generation/timestamp. Call after `compute_chunk_layout`.
    pub fn init_chunks(&mut self) -> &[ChunkInfo] {
        self.chunks.clear();
        for (i, &(start, count)) in self.chunk_layout.iter().enumerate() {
            self.chunks.push(ChunkInfo {
                chunk_id: i,
                token_start: start,
                token_count: count,
                residency: ChunkResidency::Resident,
                size_bytes: 0,
                compression_tier: PrecisionTier::FP16,
                generation: 0,
                last_access_ts: 0,
            });
        }
        &self.chunks
    }

    /// Record an access to a chunk, updating its LRU timestamp.
    ///
    /// The clock is a wrapping counter; ordering is valid as long as
    /// the total number of touches within a sequence stays below `u64::MAX`.
    pub fn touch_chunk(&mut self, chunk_idx: usize) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(chunk) = self.chunks.get_mut(chunk_idx) {
            chunk.last_access_ts = self.clock;
        }
    }

    /// Immutable reference to chunk metadata.
    #[inline]
    pub fn chunk_info(&self, chunk_idx: usize) -> Option<&ChunkInfo> {
        self.chunks.get(chunk_idx)
    }

    /// Mutable reference to chunk metadata.
    #[inline]
    pub fn chunk_info_mut(&mut self, chunk_idx: usize) -> Option<&mut ChunkInfo> {
        self.chunks.get_mut(chunk_idx)
    }

    /// All chunk metadata.
    #[inline]
    pub fn chunks(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    /// Update size_bytes for a chunk (e.g. after compression).
    pub fn set_chunk_size_bytes(&mut self, chunk_idx: usize, bytes: usize) {
        if let Some(chunk) = self.chunks.get_mut(chunk_idx) {
            chunk.size_bytes = bytes;
        }
    }

    /// Increment the generation counter for a chunk (after a write).
    pub fn bump_chunk_generation(&mut self, chunk_idx: usize) {
        if let Some(chunk) = self.chunks.get_mut(chunk_idx) {
            chunk.generation = chunk.generation.wrapping_add(1);
        }
    }

    // ── Block-Level Compression Tier Assignment ──

    /// Determine the compression tier for a chunk.
    ///
    /// The tier depends on `compress_strategy`:
    ///
    /// | Strategy   | Chunk 0 (sink) | Last chunk   | Middle chunks          |
    /// |------------|---------------|--------------|------------------------|
    /// | Uniform    | base_v_tier   | base_v_tier  | base_v_tier            |
    /// | Tiered     | FP16          | base_v_tier  | 1–2 steps downgraded   |
    /// | Adaptive   | FP16          | base_v_tier  | age-ratio degraded     |
    ///
    /// `base_v_tier` is the value-cache tier from the KIVI strategy.
    pub fn chunk_compression_tier(
        &self,
        chunk_idx: usize,
        base_v_tier: PrecisionTier,
    ) -> PrecisionTier {
        if !self.enabled {
            return PrecisionTier::FP16;
        }

        let num = self.chunk_layout.len();
        if num == 0 {
            return PrecisionTier::FP16;
        }

        match self.compress_strategy {
            ChunkCompressStrategy::Uniform => base_v_tier,

            ChunkCompressStrategy::Tiered => {
                if chunk_idx >= num {
                    return base_v_tier;
                }
                if chunk_idx == 0 {
                    // Sink chunk: always FP16
                    PrecisionTier::FP16
                } else if chunk_idx == num.saturating_sub(1) {
                    // Newest chunk: use the base tier unchanged
                    base_v_tier
                } else if chunk_idx >= num.saturating_sub(2) {
                    // Recent chunk: one step downgrade
                    downgrade_one_tier(base_v_tier)
                } else {
                    // Old chunk: two steps downgrade
                    downgrade_one_tier(downgrade_one_tier(base_v_tier))
                }
            }

            ChunkCompressStrategy::Adaptive => {
                if chunk_idx >= num {
                    return base_v_tier;
                }
                if chunk_idx == 0 {
                    // Sink chunk preserved at full precision
                    PrecisionTier::FP16
                } else if chunk_idx == num.saturating_sub(1) {
                    // Current (most recent) chunk
                    base_v_tier
                } else {
                    // Intermediate chunk: degrade by age ratio
                    let age_ratio = (num - 1 - chunk_idx) as f32 / num as f32;
                    if age_ratio < 0.3 {
                        base_v_tier
                    } else if age_ratio < 0.6 {
                        downgrade_one_tier(base_v_tier)
                    } else {
                        downgrade_one_tier(downgrade_one_tier(base_v_tier))
                    }
                }
            }
        }
    }

    /// Estimated byte size of a full chunk given model dimensions.
    ///
    /// Computed as: `chunk_size × num_kv_heads × head_dim × bytes_per_element × 2`
    /// (×2 accounts for K + V caches).
    pub fn chunk_size_bytes(
        &self,
        num_kv_heads: usize,
        head_dim: usize,
        bytes_per_element: usize,
    ) -> usize {
        self.chunk_size * num_kv_heads * head_dim * bytes_per_element * 2
    }

    /// Total bytes across all resident chunks.
    pub fn total_resident_bytes(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.residency == ChunkResidency::Resident)
            .map(|c| c.size_bytes)
            .sum()
    }

    // ── Migration / Eviction / Swap ──

    /// Select chunks for eviction using LRU ordering.
    ///
    /// Chunk 0 (sink) is always sorted last (never evicted if avoidable).
    /// Other chunks are ordered by `last_access_ts` ascending (oldest first).
    ///
    /// # Returns
    /// Chunk indices sorted by eviction preference (evict first = earliest element).
    pub fn select_eviction_candidates(&self, num_to_evict: usize) -> Vec<usize> {
        let mut candidates: Vec<(usize, &ChunkInfo)> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.residency == ChunkResidency::Resident)
            .collect();

        // Sink chunk (idx 0) goes last; otherwise sort by last_access_ts ascending
        candidates.sort_by(|(ai, a), (bi, b)| {
            if *ai == 0 {
                std::cmp::Ordering::Greater
            } else if *bi == 0 {
                std::cmp::Ordering::Less
            } else {
                a.last_access_ts.cmp(&b.last_access_ts)
            }
        });

        candidates
            .into_iter()
            .take(num_to_evict)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Mark a chunk as evicted (swapped to CPU RAM).
    ///
    /// Returns `true` if the chunk was resident and is now evicted,
    /// `false` if it was already evicted or does not exist.
    pub fn evict_chunk(&mut self, chunk_idx: usize) -> bool {
        if let Some(chunk) = self.chunks.get_mut(chunk_idx) {
            if chunk.residency == ChunkResidency::Resident {
                chunk.residency = ChunkResidency::Evicted;
                return true;
            }
        }
        false
    }

    /// Mark a chunk as resident (restored from CPU RAM).
    ///
    /// Updates the access timestamp on restore. Returns `true` if the
    /// chunk was evicted and is now resident.
    pub fn restore_chunk(&mut self, chunk_idx: usize) -> bool {
        if let Some(chunk) = self.chunks.get_mut(chunk_idx) {
            if chunk.residency == ChunkResidency::Evicted {
                chunk.residency = ChunkResidency::Resident;
                self.clock = self.clock.wrapping_add(1);
                chunk.last_access_ts = self.clock;
                return true;
            }
        }
        false
    }

    /// Check whether a chunk needs to be restored before access.
    #[inline]
    pub fn needs_restore(&self, chunk_idx: usize) -> bool {
        match self.chunks.get(chunk_idx) {
            Some(c) => c.residency == ChunkResidency::Evicted,
            None => false,
        }
    }

    /// Count of resident (GPU-resident) chunks.
    #[inline]
    pub fn resident_count(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.residency == ChunkResidency::Resident)
            .count()
    }

    /// Count of evicted (CPU-resident) chunks.
    #[inline]
    pub fn evicted_count(&self) -> usize {
        self.chunks
            .iter()
            .filter(|c| c.residency == ChunkResidency::Evicted)
            .count()
    }

    /// Generate a migration plan: which chunks to evict and which to restore.
    ///
    /// Eviction candidates are selected when `resident_count` exceeds
    /// `max_resident_chunks` (if non-zero). All evicted chunks are listed
    /// for restore; the caller decides which to actually restore based on
    /// access needs.
    pub fn migration_plan(&self) -> ChunkMigrationPlan {
        let evict = if self.max_resident_chunks > 0
            && self.resident_count() > self.max_resident_chunks
        {
            self.select_eviction_candidates(
                self.resident_count() - self.max_resident_chunks,
            )
        } else {
            Vec::new()
        };

        let restore: Vec<usize> = self
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.residency == ChunkResidency::Evicted)
            .map(|(i, _)| i)
            .collect();

        ChunkMigrationPlan { evict, restore }
    }

    // ── Reset ──

    /// Reset all chunk state for a new sequence.
    ///
    /// Clears metadata, layout, and clock. Call at the start of each
    /// new sequence to reinitialize chunk tracking.
    pub fn reset(&mut self) {
        self.chunks.clear();
        self.chunk_layout.clear();
        self.total_tokens = 0;
        self.clock = 0;
    }
}

