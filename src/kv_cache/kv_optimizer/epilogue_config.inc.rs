// EpilogueSparse — Epilogue-driven dynamic sparsity (SPEC 19 §9)
// ============================================================================

/// Epilogue-driven dynamic sparsity action for a single KV page.
///
/// Classifies each page based on Mega-Kernel Epilogue telemetry in
/// [`KvPageHeader`], determining how aggressively the page can be
/// sparsified via channel-bitmap masking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpilogueSparseAction {
    /// Page is a strong sparsity candidate — most channels can be zeroed.
    /// Maps to dense channel_bitmap_lo mask (few bits set).
    Aggressive,
    /// Page is a moderate sparsity candidate — ~50% channels zeroed.
    Moderate,
    /// Page must be preserved at full density — all channels active.
    Preserve,
}

impl EpilogueSparseAction {
    /// Human-readable label for logging / decision records.
    pub fn label(self) -> &'static str {
        match self {
            EpilogueSparseAction::Aggressive => "aggressive",
            EpilogueSparseAction::Moderate => "moderate",
            EpilogueSparseAction::Preserve => "preserve",
        }
    }
}

/// Accumulated statistics for epilogue-driven sparsity decisions.
#[derive(Debug, Clone, Default)]
pub struct EpilogueSparseStats {
    /// Pages classified as [`EpilogueSparseAction::Aggressive`].
    pub aggressive_count: usize,
    /// Pages classified as [`EpilogueSparseAction::Moderate`].
    pub moderate_count: usize,
    /// Pages classified as [`EpilogueSparseAction::Preserve`].
    pub preserve_count: usize,
    /// Total pages analyzed across all batches.
    pub total_analyzed: usize,
    /// Approximate bytes saved through sparsification (channels zeroed ×
    /// bytes-per-channel × pages).
    pub estimated_bytes_saved: usize,
}

/// Epilogue-driven dynamic sparse analyzer (SPEC 19 §9).
///
/// Reads the epilogue telemetry fields written by the Mega-Kernel Epilogue
/// stage into each [`KvPageHeader`] and determines which KV positions can
/// be dynamically sparsified. The sparsity decision is then consumed by
/// the [`VariantMatrix`] dispatch to select the appropriate sparse-matmul
/// code path.
///
/// ## Epilogue Telemetry Analysis
///
/// | Telemetry Field     | Sparsity Signal                                    |
/// |---------------------|---------------------------------------------------|
/// | `entropy_avg`       | Low entropy → concentrated attention → sparsifiable |
/// | `centroid_pos`      | Extreme centroid → structural pattern → sparsifiable |
/// | `softmax_max_avg`   | High peak → few tokens dominate → rest sparsifiable |
/// | `delta_rho_avg`     | Low delta → stable representation → safe to sparse |
/// | `dead_ratio`        | High dead → channels already inactive → aggressive |
/// | `head_entropy_spread` | Low spread → uniform heads → channel-level sparse |
///
/// ## Integration with VariantMatrix
///
/// After the epilogue writes telemetry and before the next forward pass,
/// `analyze_batch` is called. The resulting decisions feed into the
/// `channel_bitmap_lo` mask on each header, which the VariantMatrix
/// MUSTAFAR path uses to drive sparse matmul.
#[derive(Debug, Clone)]
pub struct EpilogueSparse {
    /// Entropy lower threshold (f16 bits). Pages with entropy below this
    /// are eligible for aggressive sparsity.
    pub entropy_threshold_lo: u16,
    /// Entropy upper threshold (f16 bits). Pages with entropy above this
    /// are preserved at full density.
    pub entropy_threshold_hi: u16,
    /// Minimum softmax peak for sparsity eligibility (f16 bits).
    /// Pages with lower peaks have diffuse attention → preserve.
    pub softmax_peak_threshold: u16,
    /// Maximum delta_rho for sparsity eligibility (f16 bits).
    /// Pages with higher delta are unstable → preserve.
    pub delta_rho_threshold: u16,
    /// Dead ratio threshold [0, 255]. Pages above this receive aggressive
    /// sparsity regardless of other signals.
    pub dead_ratio_threshold: u8,
    /// Maximum head entropy spread for sparsity. Uniform heads (low spread)
    /// are better sparsity candidates than diverse heads.
    pub head_spread_threshold: u8,
    /// Whether epilogue-driven dynamic sparsity is active.
    pub enabled: bool,
    /// Per-page decisions from the most recent `analyze_batch` call.
    decisions: Vec<EpilogueSparseAction>,
    /// Running statistics across all analysis batches.
    pub stats: EpilogueSparseStats,
    /// Bytes per channel element (default: 2 for FP16).
    bytes_per_channel: usize,
    /// Channels per page (default: num_kv_heads × head_dim, set externally).
    channels_per_page: usize,
}

impl Default for EpilogueSparse {
    fn default() -> Self {
        Self {
            entropy_threshold_lo: super::f32_to_f16_bits(2.0),
            entropy_threshold_hi: super::f32_to_f16_bits(4.5),
            softmax_peak_threshold: super::f32_to_f16_bits(0.5),
            delta_rho_threshold: super::f32_to_f16_bits(0.3),
            dead_ratio_threshold: 180,
            head_spread_threshold: 30,
            enabled: true,
            decisions: Vec::new(),
            stats: EpilogueSparseStats::default(),
            bytes_per_channel: 2,
            channels_per_page: 0,
        }
    }
}

impl EpilogueSparse {
    /// Create a new epilogue sparse analyzer with default thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a disabled analyzer (always returns Preserve).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the number of KV channels per page for byte-saving estimates.
    pub fn with_channels_per_page(mut self, n: usize) -> Self {
        self.channels_per_page = n;
        self
    }

    /// Set the bytes-per-channel for byte-saving estimates.
    pub fn with_bytes_per_channel(mut self, n: usize) -> Self {
        self.bytes_per_channel = n;
        self
    }

    // ── Core Analysis ──

    /// Analyze a single page header and return the recommended sparsity action.
    ///
    /// Decodes the f16 telemetry fields, computes a composite sparsity score,
    /// and thresholds it into [`EpilogueSparseAction`].
    ///
    /// When `enabled` is false, always returns [`EpilogueSparseAction::Preserve`].
    pub fn analyze_page(&self, header: &KvPageHeader) -> EpilogueSparseAction {
        if !self.enabled || !header.is_active() {
            return EpilogueSparseAction::Preserve;
        }

        let score = self.epilogue_sparse_score(header);
        self.score_to_action(score)
    }

    /// Analyze a batch of page headers, storing decisions internally.
    ///
    /// This is the primary integration point: after the Mega-Kernel Epilogue
    /// writes telemetry into each header, call this method to classify every
    /// page for downstream sparsity treatment.
    ///
    /// # Returns
    /// Slice of per-page decisions (same length as `headers`).
    pub fn analyze_batch(&mut self, headers: &[KvPageHeader]) -> &[EpilogueSparseAction] {
        self.decisions.clear();
        self.decisions.reserve(headers.len());

        let mut agg = 0usize;
        let mut mod_count = 0usize;
        let mut pres = 0usize;

        for header in headers {
            let action = self.analyze_page(header);
            match action {
                EpilogueSparseAction::Aggressive => agg += 1,
                EpilogueSparseAction::Moderate => mod_count += 1,
                EpilogueSparseAction::Preserve => pres += 1,
            }
            self.decisions.push(action);
        }

        self.stats.total_analyzed += headers.len();
        self.stats.aggressive_count += agg;
        self.stats.moderate_count += mod_count;
        self.stats.preserve_count += pres;
        self.stats.estimated_bytes_saved += self.estimate_bytes_saved(&self.decisions);

        &self.decisions
    }

    /// Retrieve the decisions from the most recent [`analyze_batch`] call.
    pub fn decisions(&self) -> &[EpilogueSparseAction] {
        &self.decisions
    }

    /// Retrieve the decision for a specific page index (post-batch).
    pub fn decision_for(&self, idx: usize) -> EpilogueSparseAction {
        match self.decisions.get(idx) {
            Some(&d) => d,
            None => EpilogueSparseAction::Preserve,
        }
    }

    // ── Header Mutation ──

    /// Apply an epilogue sparse decision to a page header by writing the
    /// channel bitmap mask.
    ///
    /// For [`EpilogueSparseAction::Aggressive`], only the most important
    /// channels are kept (mask with sparse pattern). For Moderate, ~50%
    /// channels remain active. For Preserve, the full bitmap is set.
    ///
    /// The resulting `channel_bitmap_lo` is read by the MUSTAFAR sparse
    /// matmul path in the VariantMatrix dispatch.
    pub fn apply_to_header(&self, header: &mut KvPageHeader, action: EpilogueSparseAction) {
        match action {
            EpilogueSparseAction::Aggressive => {
                // Keep only ~25% of channels: every 4th bit set
                header.channel_bitmap_lo = 0x1111_1111u32;
            }
            EpilogueSparseAction::Moderate => {
                // Keep ~50% of channels: alternating bits
                header.channel_bitmap_lo = 0x5555_5555u32;
            }
            EpilogueSparseAction::Preserve => {
                // All channels active: full mask
                header.channel_bitmap_lo = 0xFFFF_FFFFu32;
            }
        }
    }

    /// Apply epilogue sparse decisions for an entire batch.
    ///
    /// Requires that [`analyze_batch`] was called first (or decisions are
    /// set externally). Headers and decisions are paired by index.
    pub fn apply_batch(&self, headers: &mut [KvPageHeader]) {
        for (i, header) in headers.iter_mut().enumerate() {
            let action = self.decision_for(i);
            self.apply_to_header(header, action);
        }
    }

    // ── Score Computation ──

    /// Compute the raw epilogue sparsity score [0, 255] for a single header.
    ///
    /// Higher score → stronger sparsity candidate.
    ///
    /// ## Formula
    ///
    /// ```text
    /// score = entropy_score + peak_score + delta_score + dead_score + spread_score
    /// ```
    ///
    /// Each component contributes 0–51, clamped to [0, 255].
    #[inline]
    pub fn epilogue_sparse_score(&self, header: &KvPageHeader) -> u8 {
        let entropy_f = super::f16_bits_to_f32(header.entropy_avg);
        let _centroid_f = super::f16_bits_to_f32(header.centroid_pos);
        let softmax_f = super::f16_bits_to_f32(header.softmax_max_avg);
        let delta_f = super::f16_bits_to_f32(header.delta_rho_avg);

        let entropy_lo_f = super::f16_bits_to_f32(self.entropy_threshold_lo);
        let entropy_hi_f = super::f16_bits_to_f32(self.entropy_threshold_hi);
        let peak_f = super::f16_bits_to_f32(self.softmax_peak_threshold);
        let delta_f_thresh = super::f16_bits_to_f32(self.delta_rho_threshold);

        // 1. Entropy score: low entropy → high sparsity potential (0–51)
        let entropy_score = if entropy_f <= entropy_lo_f {
            51u32
        } else if entropy_f >= entropy_hi_f {
            0u32
        } else {
            let range = entropy_hi_f - entropy_lo_f;
            if range > 0.0 {
                (51.0 * (1.0 - (entropy_f - entropy_lo_f) / range)) as u32
            } else {
                0u32
            }
        };

        // 2. Softmax peak score: high peak → concentrated → sparsifiable (0–51)
        let peak_score = if softmax_f >= peak_f {
            let excess = (softmax_f - peak_f).min(0.5);
            (51.0 * (excess / 0.5)) as u32
        } else {
            0u32
        };

        // 3. Delta rho score: low delta → stable → safe to sparse (0–51)
        let delta_score = if delta_f <= delta_f_thresh {
            51u32
        } else {
            let excess = (delta_f - delta_f_thresh).min(0.7);
            (51.0 * (1.0 - excess / 0.7)) as u32
        };

        // 4. Dead ratio score: high dead → aggressive (0–51)
        let dead_score = if header.dead_ratio >= self.dead_ratio_threshold {
            51u32
        } else {
            (51u32 * header.dead_ratio as u32 / self.dead_ratio_threshold as u32).min(51)
        };

        // 5. Head spread score: low spread → uniform heads → channel-sparsifiable (0–51)
        let spread = header.head_entropy_spread();
        let spread_score = if spread <= self.head_spread_threshold {
            51u32
        } else {
            let clamped = spread;
            if clamped > self.head_spread_threshold {
                let excess = clamped - self.head_spread_threshold;
                (51.0 * (1.0 - excess as f32 / (255 - self.head_spread_threshold) as f32)) as u32
            } else {
                51u32
            }
        };

        let total = entropy_score + peak_score + delta_score + dead_score + spread_score;
        total.min(255) as u8
    }

    /// Convert a raw sparsity score to an action.
    #[inline]
    fn score_to_action(&self, score: u8) -> EpilogueSparseAction {
        if score >= 180 {
            EpilogueSparseAction::Aggressive
        } else if score >= 80 {
            EpilogueSparseAction::Moderate
        } else {
            EpilogueSparseAction::Preserve
        }
    }

    // ── Statistics ──

    /// Estimate bytes saved from a set of decisions.
    fn estimate_bytes_saved(&self, decisions: &[EpilogueSparseAction]) -> usize {
        if self.channels_per_page == 0 {
            return 0;
        }
        let per_channel = self.bytes_per_channel;
        let total_channels = self.channels_per_page;

        decisions
            .iter()
            .map(|d| match d {
                EpilogueSparseAction::Aggressive => {
                    // ~25% channels kept → 75% saved
                    (total_channels * 3 / 4) * per_channel
                }
                EpilogueSparseAction::Moderate => {
                    // ~50% channels kept → 50% saved
                    (total_channels / 2) * per_channel
                }
                EpilogueSparseAction::Preserve => 0,
            })
            .sum()
    }

    /// Reset accumulated statistics.
    pub fn reset_stats(&mut self) {
        self.stats = EpilogueSparseStats::default();
    }

    /// Reset decisions buffer (does not clear stats).
    pub fn reset_decisions(&mut self) {
        self.decisions.clear();
    }
}

// ============================================================================
// EpilogueSparse + VariantMatrix integration helpers
// ============================================================================

/// Integrate epilogue sparsity analysis into a header batch, producing a
/// variant bitmask suitable for VariantMatrix dispatch.
///
/// This is the key linkage between epilogue-driven sparsity (SPEC 19 §9)
/// and the compile-time variant dispatch table (KVO8). After epilogue
/// telemetry is written, the sparsity analyzer classifies each page and
/// writes the `channel_bitmap_lo` mask. The returned bitmask encodes
/// which strategies should be active for the next forward pass, taking
/// the epilogue sparsity decisions into account.
///
/// # Returns
/// A 4-bit variant mask (`variant_bits::*`) that can be passed directly
/// to [`VariantMatrix::dispatch_batch`].
pub fn epilogue_dynamic_sparse(
    sparse: &mut EpilogueSparse,
    headers: &mut [KvPageHeader],
    base_variant_bits: u8,
) -> u8 {
    if !sparse.enabled || headers.is_empty() {
        return base_variant_bits;
    }

    sparse.analyze_batch(headers);
    sparse.apply_batch(headers);

    // If any page received aggressive or moderate sparsity, ensure the
    // MUSTAFAR bit is set (it drives the sparse matmul code path).
    let has_sparse_action = sparse.decisions().iter().any(|d| {
        matches!(d, EpilogueSparseAction::Aggressive | EpilogueSparseAction::Moderate)
    });

    if has_sparse_action {
        base_variant_bits | variant_bits::MUSTAFAR
    } else {
        base_variant_bits
    }
}

// ============================================================================
// KvOptimizationConfig — unified KV optimization configuration (SPEC 19 §10)
// ============================================================================

/// Unified KV cache optimization configuration (SPEC 19 §10).
///
/// Aggregates all strategy-specific configurations — KIVI, KVTuner, MUSTAFAR,
/// ChunkKV, EpilogueSparse, CrossDecision, and VariantMatrix — into a single
/// initialization struct consumed by [`KvOptimization`] at startup.
///
/// ## Strategy Coverage
///
/// | Strategy        | Config Fields                   | SPEC § |
/// |-----------------|---------------------------------|--------|
/// | KIVI            | `kivi_*`                        | §3     |
/// | KVTuner         | `kv_tuner_*`                    | §4     |
/// | MUSTAFAR        | `mustafar_*`                    | §5     |
/// | ChunkKV         | `chunk_kv_*`                    | §6     |
/// | EpilogueSparse  | `epilogue_sparse_*`             | §9     |
/// | CrossDecision   | `cross_decision_*`              | §7     |
///
/// ## Usage
///
/// ```text
/// let config = KvOptimizationConfig::default()
///     .with_kivi_enabled(true)
///     .with_mustafar_enabled(true)
///     .with_epilogue_sparse_enabled(true);
/// let opt = KvOptimization::from_config(config, num_layers, hardware);
/// ```
#[derive(Debug, Clone)]
pub struct KvOptimizationConfig {
    /// Globally enable/disable all KV optimization strategies.
    /// When `false`, all individual strategy enablement flags are ignored
    /// and the system operates at FP16 baseline.
    pub enabled: bool,

    // ── KIVI (SPEC 19 §3) ──
    /// Whether asymmetric K/V quantization is active.
    pub kivi_enabled: bool,
    /// Key cache precision tier.
    pub kivi_key_precision: PrecisionTier,
    /// Value cache precision tier.
    pub kivi_val_precision: PrecisionTier,
    /// Number of attention sink tokens protected at full precision.
    pub kivi_sink_count: usize,

    // ── KVTuner (SPEC 19 §4) ──
    /// Whether dynamic precision tuning is active.
    pub kv_tuner_enabled: bool,
    /// EMA smoothing factor for entropy tracking.
    pub kv_tuner_alpha: f32,
    /// Sequence length threshold above which long-sequence mode activates.
    pub kv_tuner_long_seq_threshold: usize,
    /// Minimum tier the tuner may select.
    pub kv_tuner_min_tier: PrecisionTier,
    /// Maximum tier the tuner may select.
    pub kv_tuner_max_tier: PrecisionTier,

    // ── MUSTAFAR (SPEC 19 §5) ──
    /// Whether MUSTAFAR token retention is active.
    pub mustafar_enabled: bool,
    /// Entropy spread threshold for MUSTAFAR detection.
    pub mustafar_entropy_threshold: u8,
    /// Minimum importance score to qualify as MUSTAFAR.
    pub mustafar_importance_threshold: u8,
    /// Maximum MUSTAFAR tokens retained per sequence.
    pub mustafar_max_tokens: usize,

    // ── ChunkKV (SPEC 19 §6) ──
    /// Whether ChunkKV block-level management is active.
    pub chunk_kv_enabled: bool,
    /// Chunk size in tokens.
    pub chunk_kv_chunk_size: usize,
    /// Maximum chunks resident in GPU HBM simultaneously.
    pub chunk_kv_max_resident_chunks: usize,

    // ── EpilogueSparse (SPEC 19 §9) ──
    /// Whether epilogue-driven dynamic sparsity is active.
    pub epilogue_sparse_enabled: bool,
    /// Entropy lower threshold (f16 bits). Below → aggressive sparsity.
    pub epilogue_sparse_entropy_lo: u16,
    /// Entropy upper threshold (f16 bits). Above → preserve full density.
    pub epilogue_sparse_entropy_hi: u16,
    /// Minimum softmax peak for sparsity eligibility (f16 bits).
    pub epilogue_sparse_softmax_peak: u16,
    /// Maximum delta_rho for sparsity eligibility (f16 bits).
    pub epilogue_sparse_delta_rho: u16,
    /// Dead ratio threshold [0,255] for aggressive sparsity override.
    pub epilogue_sparse_dead_ratio: u8,
    /// Maximum head entropy spread for sparsity eligibility.
    pub epilogue_sparse_head_spread: u8,

    // ── CrossDecision (SPEC 19 §7) ──
    /// Maximum decisions retained in the ring buffer.
    pub cross_decision_max_decisions: usize,
    /// Initial active variant bitmask.
    /// Bits: [3]=ChunkKV, [2]=MUSTAFAR, [1]=KVTuner, [0]=KIVI.
    pub cross_decision_initial_variant: u8,
}

impl Default for KvOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,

            // KIVI defaults
            kivi_enabled: true,
            kivi_key_precision: PrecisionTier::FP16,
            kivi_val_precision: PrecisionTier::KIVI4,
            kivi_sink_count: 4,

            // KVTuner defaults
            kv_tuner_enabled: true,
            kv_tuner_alpha: 0.85,
            kv_tuner_long_seq_threshold: 4096,
            kv_tuner_min_tier: PrecisionTier::KIVI2,
            kv_tuner_max_tier: PrecisionTier::FP16,

            // MUSTAFAR defaults
            mustafar_enabled: true,
            mustafar_entropy_threshold: 100,
            mustafar_importance_threshold: 128,
            mustafar_max_tokens: 64,

            // ChunkKV defaults
            chunk_kv_enabled: true,
            chunk_kv_chunk_size: 64,
            chunk_kv_max_resident_chunks: 128,

            // EpilogueSparse defaults
            epilogue_sparse_enabled: true,
            epilogue_sparse_entropy_lo: super::f32_to_f16_bits(2.0),
            epilogue_sparse_entropy_hi: super::f32_to_f16_bits(4.5),
            epilogue_sparse_softmax_peak: super::f32_to_f16_bits(0.5),
            epilogue_sparse_delta_rho: super::f32_to_f16_bits(0.3),
            epilogue_sparse_dead_ratio: 180,
            epilogue_sparse_head_spread: 30,

            // CrossDecision defaults
            cross_decision_max_decisions: 64,
            cross_decision_initial_variant: variant_bits::ALL,
        }
    }
}

impl KvOptimizationConfig {
    /// Create a fully-disabled configuration (all strategies off, baseline FP16).
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            kivi_enabled: false,
            kv_tuner_enabled: false,
            mustafar_enabled: false,
            chunk_kv_enabled: false,
            epilogue_sparse_enabled: false,
            ..Default::default()
        }
    }

    /// Create a minimal configuration suitable for small-GPU / embedded scenarios.
    /// Enables only KIVI baseline (asymmetric K/V) for memory savings with
    /// minimal compute overhead.
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            kivi_enabled: true,
            kv_tuner_enabled: false,
            mustafar_enabled: false,
            chunk_kv_enabled: false,
            epilogue_sparse_enabled: false,
            kivi_sink_count: 1,
            ..Default::default()
        }
    }

    /// Create a high-performance configuration with all strategies active.
    pub fn full_stack() -> Self {
        Self::default()
    }

    // ── Builder Methods ──

    /// Set global enable/disable.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Enable or disable KIVI.
    pub fn with_kivi_enabled(mut self, enabled: bool) -> Self {
        self.kivi_enabled = enabled;
        self
    }

    /// Set KIVI key precision.
    pub fn with_kivi_key_precision(mut self, tier: PrecisionTier) -> Self {
        self.kivi_key_precision = tier;
        self
    }

    /// Set KIVI value precision.
    pub fn with_kivi_val_precision(mut self, tier: PrecisionTier) -> Self {
        self.kivi_val_precision = tier;
        self
    }

    /// Set KIVI sink count.
    pub fn with_kivi_sink_count(mut self, count: usize) -> Self {
        self.kivi_sink_count = count;
        self
    }

    /// Enable or disable KVTuner.
    pub fn with_kv_tuner_enabled(mut self, enabled: bool) -> Self {
        self.kv_tuner_enabled = enabled;
        self
    }

    /// Set KVTuner EMA alpha.
    pub fn with_kv_tuner_alpha(mut self, alpha: f32) -> Self {
        self.kv_tuner_alpha = alpha;
        self
    }

    /// Enable or disable MUSTAFAR.
    pub fn with_mustafar_enabled(mut self, enabled: bool) -> Self {
        self.mustafar_enabled = enabled;
        self
    }

    /// Set MUSTAFAR max tokens.
    pub fn with_mustafar_max_tokens(mut self, max: usize) -> Self {
        self.mustafar_max_tokens = max;
        self
    }

    /// Enable or disable ChunkKV.
    pub fn with_chunk_kv_enabled(mut self, enabled: bool) -> Self {
        self.chunk_kv_enabled = enabled;
        self
    }

    /// Set ChunkKV chunk size.
    pub fn with_chunk_kv_chunk_size(mut self, size: usize) -> Self {
        self.chunk_kv_chunk_size = size;
        self
    }

    /// Enable or disable EpilogueSparse.
    pub fn with_epilogue_sparse_enabled(mut self, enabled: bool) -> Self {
        self.epilogue_sparse_enabled = enabled;
        self
    }

    /// Set initial cross-decision variant from a bitmask.
    pub fn with_initial_variant(mut self, bits: u8) -> Self {
        self.cross_decision_initial_variant = bits & variant_bits::ALL;
        self
    }

    // ── Validation ──

    /// Validate configuration consistency.
    ///
    /// Returns `Ok(())` if the configuration is internally consistent,
    /// or `Err(message)` if conflicts are detected.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.kv_tuner_enabled && !self.kivi_enabled {
            return Err("KVTuner requires KIVI to be enabled (KVTuner wraps KIVI)");
        }
        if tuner_tier_rank(self.kv_tuner_min_tier)
            > tuner_tier_rank(self.kv_tuner_max_tier)
        {
            return Err("kv_tuner_min_tier must be <= kv_tuner_max_tier");
        }
        if self.kv_tuner_alpha < 0.0 || self.kv_tuner_alpha > 1.0 {
            return Err("kv_tuner_alpha must be in [0.0, 1.0]");
        }
        if self.chunk_kv_chunk_size == 0 {
            return Err("chunk_kv_chunk_size must be > 0");
        }
        Ok(())
    }
}

// ============================================================================
// KvOptimization — runtime KV optimization manager (SPEC 19 §10)
// ============================================================================

/// Status snapshot returned by [`KvOptimization::status`].
///
/// Provides a point-in-time view of all optimization strategy states
/// for telemetry / observability integration.
#[derive(Debug, Clone)]
pub struct KvOptimizationStatus {
    /// Whether optimization is globally enabled.
    pub enabled: bool,
    /// Active KIVI key precision.
    pub kivi_key_precision: PrecisionTier,
    /// Active KIVI value precision.
    pub kivi_val_precision: PrecisionTier,
    /// Whether KIVI is active.
    pub kivi_active: bool,
    /// Whether KVTuner is active.
    pub kv_tuner_active: bool,
    /// KVTuner smoothed entropy.
    pub kv_tuner_smoothed_entropy: f32,
    /// KVTuner last decision reason.
    pub kv_tuner_last_reason: KvTunerReason,
    /// Whether MUSTAFAR is active.
    pub mustafar_active: bool,
    /// Number of MUSTAFAR-classified tokens in current batch.
    pub mustafar_token_count: usize,
    /// Whether ChunkKV is active.
    pub chunk_kv_active: bool,
    /// Number of resident chunks.
    pub chunk_kv_resident_chunks: usize,
    /// Whether EpilogueSparse is active.
    pub epilogue_sparse_active: bool,
    /// EpilogueSparse aggressive page count (cumulative).
    pub epilogue_sparse_aggressive_count: usize,
    /// EpilogueSparse moderate page count (cumulative).
    pub epilogue_sparse_moderate_count: usize,
    /// EpilogueSparse preserve page count (cumulative).
    pub epilogue_sparse_preserve_count: usize,
    /// EpilogueSparse estimated bytes saved.
    pub epilogue_sparse_bytes_saved: usize,
    /// Active cross-decision variant.
    pub active_variant: DecisionVariant,
    /// Active variant bitmask.
    pub active_variant_bits: u8,
    /// Number of recorded cross-decisions.
    pub decision_count: usize,
}

/// Runtime KV cache optimization manager (SPEC 19 §10).
///
/// Owns all strategy instances and provides unified initialization,
/// status query, and telemetry integration. Created from a
/// [`KvOptimizationConfig`] at model-load time.
///
/// ## Lifecycle
///
/// ```text
/// KvOptimizationConfig::default()
///   → KvOptimization::from_config(config, num_layers, hardware)
///   → optimizer.kivi, optimizer.mustafar, ...
///   → optimizer.status() for observability
/// ```
///
/// ## Integration Points
///
/// | Component          | Method                          | When                        |
/// |--------------------|---------------------------------|-----------------------------|
/// | Epilogue telemetry | `epilogue_sparse.analyze_batch` | After Mega-Kernel Epilogue  |
/// | Variant dispatch   | `variant_matrix.dispatch_batch` | Before attention forward    |
/// | Precision decision  | `cross_decision.compose_for_page` | Page write path           |
/// | Observability      | `status()`                      | Any time (cheap snapshot)   |
#[derive(Debug, Clone)]
pub struct KvOptimization {
    /// Configuration that produced this instance.
    pub config: KvOptimizationConfig,
    /// KIVI asymmetric quantization strategy.
    pub kivi: KiviStrategy,
    /// Dynamic precision tuner.
    pub kv_tuner: KvTunerStrategy,
    /// MUSTAFAR token retention classifier.
    pub mustafar: MustafarStrategy,
    /// Chunk-level KV cache manager.
    pub chunk_kv: ChunkKvStrategy,
    /// Epilogue-driven dynamic sparse analyzer.
    pub epilogue_sparse: EpilogueSparse,
    /// Cross-dimensional decision matrix.
    pub cross_decision: CrossDecisionMatrix,
    /// Compile-time variant dispatch table.
    pub variant_matrix: VariantMatrix,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Active variant bitmask for dispatch.
    active_variant_bits: u8,
}

impl Default for KvOptimization {
    fn default() -> Self {
        Self::from_config(KvOptimizationConfig::default(), 32, HardwareProfile::default())
    }
}

impl KvOptimization {
    /// Initialize all optimization strategies from a unified configuration.
    ///
    /// # Arguments
    /// * `config` — strategy enablement and parameter configuration.
    /// * `num_layers` — total transformer layers.
    /// * `hardware` — hardware capability profile for cross-decision.
    ///
    /// # Panics
    /// Panics if `config.validate()` fails — callers should validate before
    /// passing the config to this constructor.
    pub fn from_config(
        config: KvOptimizationConfig,
        num_layers: usize,
        hardware: HardwareProfile,
    ) -> Self {
        // Validate before proceeding
        if let Err(msg) = config.validate() {
            panic!("KvOptimizationConfig validation failed: {}", msg);
        }

        let global_enabled = config.enabled;

        // ── KIVI ──
        let kivi = if global_enabled && config.kivi_enabled {
            KiviStrategy::new()
                .with_key_precision(config.kivi_key_precision)
                .with_val_precision(config.kivi_val_precision)
                .with_sink_count(config.kivi_sink_count)
        } else {
            KiviStrategy::disabled()
        };

        // ── KVTuner ──
        let kv_tuner = if global_enabled && config.kv_tuner_enabled {
            KvTunerStrategy::with_kivi(kivi.clone())
                .with_alpha(config.kv_tuner_alpha)
                .with_long_seq_threshold(config.kv_tuner_long_seq_threshold)
                .with_tier_bounds(config.kv_tuner_min_tier, config.kv_tuner_max_tier)
        } else {
            KvTunerStrategy::disabled()
        };

        // ── MUSTAFAR ──
        let mustafar = if global_enabled && config.mustafar_enabled {
            MustafarStrategy::new()
                .with_entropy_threshold(config.mustafar_entropy_threshold)
                .with_importance_threshold(config.mustafar_importance_threshold)
                .with_max_tokens(config.mustafar_max_tokens)
        } else {
            MustafarStrategy::disabled()
        };

        // ── ChunkKV ──
        let chunk_kv = if global_enabled && config.chunk_kv_enabled {
            ChunkKvStrategy::new()
                .with_chunk_size(config.chunk_kv_chunk_size)
                .with_max_resident(config.chunk_kv_max_resident_chunks)
        } else {
            ChunkKvStrategy::disabled()
                .with_chunk_size(config.chunk_kv_chunk_size)
        };

        // ── EpilogueSparse ──
        let epilogue_sparse = if global_enabled && config.epilogue_sparse_enabled {
            let mut es = EpilogueSparse::new();
            es.entropy_threshold_lo = config.epilogue_sparse_entropy_lo;
            es.entropy_threshold_hi = config.epilogue_sparse_entropy_hi;
            es.softmax_peak_threshold = config.epilogue_sparse_softmax_peak;
            es.delta_rho_threshold = config.epilogue_sparse_delta_rho;
            es.dead_ratio_threshold = config.epilogue_sparse_dead_ratio;
            es.head_spread_threshold = config.epilogue_sparse_head_spread;
            es.enabled = true;
            es
        } else {
            EpilogueSparse::disabled()
        };

        // ── CrossDecision ──
        let initial_variant = DecisionVariant::from_bits(config.cross_decision_initial_variant);
        let cross_decision = CrossDecisionMatrix::with_hardware(hardware)
            .with_kivi(kivi.clone())
            .with_kv_tuner(kv_tuner.clone())
            .with_mustafar(mustafar.clone())
            .with_chunk_kv(chunk_kv.clone())
            .with_variant(initial_variant);

        // ── VariantMatrix ──
        let variant_matrix = VariantMatrix::new();

        let active_bits = initial_variant.bits();

        Self {
            config,
            kivi,
            kv_tuner,
            mustafar,
            chunk_kv,
            epilogue_sparse,
            cross_decision,
            variant_matrix,
            num_layers,
            active_variant_bits: active_bits,
        }
    }

    /// Create a disabled optimization instance (all strategies off).
    pub fn disabled(num_layers: usize, hardware: HardwareProfile) -> Self {
        Self::from_config(KvOptimizationConfig::disabled(), num_layers, hardware)
    }

    // ── Status Query ──

    /// Return a point-in-time snapshot of all optimization strategy states.
    ///
    /// This is the primary observability hook: call it any time to get a
    /// complete picture of which strategies are active and their current
    /// telemetry values.
    pub fn status(&self) -> KvOptimizationStatus {
        KvOptimizationStatus {
            enabled: self.config.enabled,
            kivi_key_precision: self.kivi.key_precision,
            kivi_val_precision: self.kivi.val_precision,
            kivi_active: self.kivi.enabled,
            kv_tuner_active: self.kv_tuner.enabled,
            kv_tuner_smoothed_entropy: self.kv_tuner.entropy(),
            kv_tuner_last_reason: self.kv_tuner.last_reason(),
            mustafar_active: self.mustafar.enabled,
            mustafar_token_count: self.mustafar.mustafar_count(),
            chunk_kv_active: self.chunk_kv.enabled,
            chunk_kv_resident_chunks: self.chunk_kv.num_chunks(),
            epilogue_sparse_active: self.epilogue_sparse.enabled,
            epilogue_sparse_aggressive_count: self.epilogue_sparse.stats.aggressive_count,
            epilogue_sparse_moderate_count: self.epilogue_sparse.stats.moderate_count,
            epilogue_sparse_preserve_count: self.epilogue_sparse.stats.preserve_count,
            epilogue_sparse_bytes_saved: self.epilogue_sparse.stats.estimated_bytes_saved,
            active_variant: self.cross_decision.active_variant,
            active_variant_bits: self.active_variant_bits,
            decision_count: self.cross_decision.decision_count(),
        }
    }

    /// Return the currently active variant bitmask for VariantMatrix dispatch.
    #[inline]
    pub fn active_variant_bits(&self) -> u8 {
        self.active_variant_bits
    }

    /// Update the active variant bitmask (e.g., after epilogue sparse analysis).
    ///
    /// This ensures the cross-decision matrix and variant dispatch stay
    /// synchronized.
    pub fn set_active_variant_bits(&mut self, bits: u8) {
        self.active_variant_bits = bits & variant_bits::ALL;
        self.cross_decision.active_variant = DecisionVariant::from_bits(self.active_variant_bits);
    }

    // ── Telemetry Integration ──

    /// Run epilogue-driven dynamic sparsity analysis on a batch of page headers.
    ///
    /// This is the primary integration point between the Mega-Kernel Epilogue
    /// telemetry and the VariantMatrix dispatch. After the epilogue writes
    /// telemetry fields into each header, call this method to:
    ///
    /// 1. Analyze epilogue telemetry for each page.
    /// 2. Write `channel_bitmap_lo` masks based on sparsity decisions.
    /// 3. Update the active variant bits to include MUSTAFAR if sparse
    ///    actions are present.
    ///
    /// # Returns
    /// Updated variant bitmask suitable for [`VariantMatrix::dispatch_batch`].
    pub fn run_epilogue_sparse(&mut self, headers: &mut [KvPageHeader]) -> u8 {
        let new_bits = epilogue_dynamic_sparse(
            &mut self.epilogue_sparse,
            headers,
            self.active_variant_bits,
        );
        self.set_active_variant_bits(new_bits);
        new_bits
    }

    /// Re-evaluate the cross-decision variant based on current hardware profile
    /// and return the updated variant.
    pub fn reevaluate(&mut self) -> (DecisionVariant, &'static str) {
        let (variant, reason) = self.cross_decision.evaluate();
        self.active_variant_bits = variant.bits();
        (variant, reason)
    }

    /// Re-evaluate with sequence state (entropy + length).
    pub fn reevaluate_with_state(
        &mut self,
        seq_len: usize,
        entropy: f32,
    ) -> (DecisionVariant, &'static str) {
        let (variant, reason) = self.cross_decision.evaluate_with_state(seq_len, entropy);
        self.active_variant_bits = variant.bits();
        (variant, reason)
    }

    // ── Batch/Page Composition Delegates ──

    /// Compose all active strategies for a batch of page headers.
    ///
    /// Delegates to [`VariantMatrix::dispatch_batch`] using the current
    /// active variant bitmask.
    #[inline]
    pub fn compose_batch(
        &mut self,
        headers: &mut [KvPageHeader],
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> usize {
        self.variant_matrix.dispatch_batch(
            self.active_variant_bits,
            &self.kivi,
            &mut self.kv_tuner,
            &mut self.mustafar,
            &mut self.chunk_kv,
            headers,
            seq_len,
            layer_depth_ratio,
            num_kv_heads,
        )
    }

    /// Compose all active strategies for a single page header.
    #[inline]
    pub fn compose_page(
        &mut self,
        header: &mut KvPageHeader,
        token_idx: usize,
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> (PrecisionTier, PrecisionTier) {
        self.variant_matrix.dispatch_page(
            self.active_variant_bits,
            &self.kivi,
            &mut self.kv_tuner,
            &mut self.mustafar,
            &mut self.chunk_kv,
            header,
            token_idx,
            seq_len,
            layer_depth_ratio,
            num_kv_heads,
        )
    }

    /// Get the recommended precision tier for a single token.
    #[inline]
    pub fn recommend_tier(
        &self,
        token_idx: usize,
        header: &KvPageHeader,
        seq_len: usize,
    ) -> PrecisionTier {
        self.variant_matrix.dispatch_recommend(
            self.active_variant_bits,
            &self.kivi,
            &self.kv_tuner,
            &self.mustafar,
            &self.chunk_kv,
            token_idx,
            header,
            seq_len,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

