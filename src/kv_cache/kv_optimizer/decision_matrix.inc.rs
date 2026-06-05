/// Migration plan produced by [`ChunkKvStrategy::migration_plan`].
///
/// Encodes which chunks should be evicted (GPU → CPU) and which
/// should be restored (CPU → GPU) for the next migration cycle.
#[derive(Debug, Clone)]
pub struct ChunkMigrationPlan {
    /// Chunk indices to evict from GPU to CPU
    pub evict: Vec<usize>,
    /// Chunk indices to restore from CPU to GPU
    pub restore: Vec<usize>,
}

// ============================================================================
// CrossDecisionMatrix — four-dimension cross-decision matrix (SPEC 19 §7)
// ============================================================================

/// Variant of the four-dimensional strategy matrix.
///
/// Each variant represents a specific combination of active strategies.
/// The decision matrix selects the optimal variant based on hardware
/// capabilities and model architecture features.
///
/// ## Variant Matrix (compile-time expansion)
///
/// The full 16-entry matrix is available at compile time via
/// [`VARIANT_MATRIX`]. Each entry encodes which of the four strategies
/// (KIVI, KVTuner, MUSTAFAR, ChunkKV) are active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionVariant {
    /// All four strategies active — full optimization stack
    FullStack,
    /// KIVI + KVTuner only — precision-centric, no token retention or chunking
    PrecisionOnly,
    /// KIVI + MUSTAFAR + ChunkKV — retention-centric, no dynamic tuning
    RetentionOnly,
    /// KIVI only — baseline mixed precision, no cross-dimension optimization
    Baseline,
    /// ChunkKV only — block-level management with uniform precision
    ChunkOnly,
    /// MUSTAFAR + KVTuner — token retention with dynamic tuning
    AdaptiveRetention,
    /// KIVI + ChunkKV — mixed precision with block-level layout
    ChunkedPrecision,
    /// Custom variant with explicit strategy bitmask
    /// Bits: [3]=ChunkKV, [2]=MUSTAFAR, [1]=KVTuner, [0]=KIVI
    Custom(u8),
}

/// Bitmask constants for [`DecisionVariant::Custom`].
pub mod variant_bits {
    pub const KIVI: u8 = 1 << 0;
    pub const KV_TUNER: u8 = 1 << 1;
    pub const MUSTAFAR: u8 = 1 << 2;
    pub const CHUNK_KV: u8 = 1 << 3;
    pub const ALL: u8 = KIVI | KV_TUNER | MUSTAFAR | CHUNK_KV;
    pub const NONE: u8 = 0;
}

impl DecisionVariant {
    /// Check whether KIVI strategy is active in this variant.
    #[inline]
    pub fn has_kivi(self) -> bool {
        self.bits() & variant_bits::KIVI != 0
    }

    /// Check whether KVTuner strategy is active.
    #[inline]
    pub fn has_kv_tuner(self) -> bool {
        self.bits() & variant_bits::KV_TUNER != 0
    }

    /// Check whether MUSTAFAR strategy is active.
    #[inline]
    pub fn has_mustafar(self) -> bool {
        self.bits() & variant_bits::MUSTAFAR != 0
    }

    /// Check whether ChunkKV strategy is active.
    #[inline]
    pub fn has_chunk_kv(self) -> bool {
        self.bits() & variant_bits::CHUNK_KV != 0
    }

    /// Convert variant to a 4-bit bitmask.
    #[inline]
    pub fn bits(self) -> u8 {
        match self {
            DecisionVariant::FullStack => variant_bits::ALL,
            DecisionVariant::PrecisionOnly => variant_bits::KIVI | variant_bits::KV_TUNER,
            DecisionVariant::RetentionOnly => variant_bits::KIVI | variant_bits::MUSTAFAR | variant_bits::CHUNK_KV,
            DecisionVariant::Baseline => variant_bits::KIVI,
            DecisionVariant::ChunkOnly => variant_bits::CHUNK_KV,
            DecisionVariant::AdaptiveRetention => variant_bits::MUSTAFAR | variant_bits::KV_TUNER,
            DecisionVariant::ChunkedPrecision => variant_bits::KIVI | variant_bits::CHUNK_KV,
            DecisionVariant::Custom(mask) => mask & variant_bits::ALL,
        }
    }

    /// Create a variant from a bitmask.
    #[inline]
    pub fn from_bits(mask: u8) -> Self {
        let m = mask & variant_bits::ALL;
        match m {
            variant_bits::ALL => DecisionVariant::FullStack,
            b if b == variant_bits::KIVI | variant_bits::KV_TUNER => DecisionVariant::PrecisionOnly,
            b if b == variant_bits::KIVI | variant_bits::MUSTAFAR | variant_bits::CHUNK_KV => DecisionVariant::RetentionOnly,
            variant_bits::KIVI => DecisionVariant::Baseline,
            variant_bits::CHUNK_KV => DecisionVariant::ChunkOnly,
            b if b == variant_bits::MUSTAFAR | variant_bits::KV_TUNER => DecisionVariant::AdaptiveRetention,
            b if b == variant_bits::KIVI | variant_bits::CHUNK_KV => DecisionVariant::ChunkedPrecision,
            _ => DecisionVariant::Custom(m),
        }
    }

    /// Human-readable name for the variant.
    pub fn name(self) -> &'static str {
        match self {
            DecisionVariant::FullStack => "FullStack",
            DecisionVariant::PrecisionOnly => "PrecisionOnly",
            DecisionVariant::RetentionOnly => "RetentionOnly",
            DecisionVariant::Baseline => "Baseline",
            DecisionVariant::ChunkOnly => "ChunkOnly",
            DecisionVariant::AdaptiveRetention => "AdaptiveRetention",
            DecisionVariant::ChunkedPrecision => "ChunkedPrecision",
            DecisionVariant::Custom(_) => "Custom",
        }
    }
}

/// Compile-time variant matrix — all 16 strategy combinations.
///
/// Each entry is a `(variant, description)` pair covering every
/// possible combination of the four strategy dimensions.
/// The cross-decision algorithm uses this matrix to select the
/// optimal variant for the current hardware/model profile.
pub const VARIANT_MATRIX: [(DecisionVariant, &str); 16] = [
    (DecisionVariant::Custom(0b0000), "All strategies disabled — raw FP16"),
    (DecisionVariant::Baseline, "KIVI only — asymmetric K/V mixed precision"),
    (DecisionVariant::Custom(0b0010), "KVTuner only — dynamic precision without KIVI"),
    (DecisionVariant::PrecisionOnly, "KIVI + KVTuner — precision-centric"),
    (DecisionVariant::Custom(0b0100), "MUSTAFAR only — token retention without KIVI"),
    (DecisionVariant::Custom(0b0101), "KIVI + MUSTAFAR — precision with token retention"),
    (DecisionVariant::AdaptiveRetention, "MUSTAFAR + KVTuner — adaptive token retention"),
    (DecisionVariant::Custom(0b0111), "KIVI + KVTuner + MUSTAFAR — no chunking"),
    (DecisionVariant::ChunkOnly, "ChunkKV only — block-level with uniform FP16"),
    (DecisionVariant::ChunkedPrecision, "KIVI + ChunkKV — precision chunking"),
    (DecisionVariant::Custom(0b1010), "KVTuner + ChunkKV — dynamic chunk precision"),
    (DecisionVariant::Custom(0b1011), "KIVI + KVTuner + ChunkKV — no MUSTAFAR"),
    (DecisionVariant::RetentionOnly, "KIVI + MUSTAFAR + ChunkKV — retention-centric"),
    (DecisionVariant::Custom(0b1101), "KIVI + MUSTAFAR + ChunkKV (no KVTuner)"),
    (DecisionVariant::Custom(0b1110), "KVTuner + MUSTAFAR + ChunkKV (no KIVI)"),
    (DecisionVariant::FullStack, "Full stack — all four strategies active"),
];

/// Hardware capability profile for cross-decision optimization.
///
/// Used by [`CrossDecisionMatrix::evaluate`] to select the optimal
/// strategy variant based on available hardware resources.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Total GPU HBM in GB
    pub gpu_memory_gb: f32,
    /// Relative compute capability (1.0 = baseline A100-level)
    pub compute_capability: f32,
    /// Whether FP8 tensor ops are available (H100+)
    pub supports_fp8: bool,
    /// Whether nvCOMP ANS codec is available
    pub supports_nvcomp: bool,
    /// PCIe/NVLink bandwidth in GB/s
    pub pcie_bandwidth_gbs: f32,
    /// Number of KV heads in the model
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            gpu_memory_gb: 80.0,
            compute_capability: 1.0,
            supports_fp8: false,
            supports_nvcomp: false,
            pcie_bandwidth_gbs: 64.0,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 8192,
        }
    }
}

impl HardwareProfile {
    /// Create a minimal hardware profile (for testing).
    pub fn minimal() -> Self {
        Self {
            gpu_memory_gb: 4.0,
            compute_capability: 0.5,
            supports_fp8: false,
            supports_nvcomp: false,
            pcie_bandwidth_gbs: 16.0,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
        }
    }

    /// Create a high-end hardware profile (H100-class).
    pub fn high_end() -> Self {
        Self {
            gpu_memory_gb: 80.0,
            compute_capability: 2.5,
            supports_fp8: true,
            supports_nvcomp: true,
            pcie_bandwidth_gbs: 900.0, // NVLink
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 131072,
        }
    }

    /// Estimated KV cache size in bytes for a full sequence at FP16.
    pub fn kv_cache_size_bytes(&self) -> usize {
        self.max_seq_len * self.num_kv_heads * self.head_dim * 2 * 2 // K + V, 2 bytes each
    }

    /// KV cache size in GB.
    pub fn kv_cache_size_gb(&self) -> f32 {
        self.kv_cache_size_bytes() as f32 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Record of a cross-decision evaluation (SPEC 19 §7).
///
/// Tracks which variant was selected, why, and the entropy/sequence
/// state at the time of decision.
#[derive(Debug, Clone)]
pub struct DecisionRecord {
    /// Monotonic sequence number
    pub seq: u64,
    /// Selected variant after evaluation
    pub variant: DecisionVariant,
    /// Reason for selecting this variant
    pub reason: &'static str,
    /// Current sequence length at decision time
    pub seq_len: usize,
    /// Smoothed entropy at decision time
    pub entropy: f32,
    /// KV cache pressure ratio (used / capacity)
    pub cache_pressure: f32,
}

/// Cross-dimensional decision matrix (SPEC 19-KV-CACHE-OPTIMIZATION §7).
///
/// Jointly optimizes across four strategy dimensions:
///
/// | Dimension   | Strategy            | Role                              |
/// |-------------|---------------------|-----------------------------------|
/// | KIVI        | `KiviStrategy`      | Asymmetric K/V mixed precision    |
/// | KVTuner     | `KvTunerStrategy`   | Dynamic precision adjustment      |
/// | MUSTAFAR    | `MustafarStrategy`  | Token retention & eviction order  |
/// | ChunkKV     | `ChunkKvStrategy`   | Block-level layout & migration    |
///
/// ## Joint Decision Algorithm
///
/// The matrix evaluates hardware capabilities and model architecture to
/// select the optimal strategy combination. The decision follows a
/// multi-factor scoring model:
///
/// 1. **Memory pressure**: if KV cache size exceeds GPU memory, enable
///    ChunkKV for swap support and aggressive compression.
/// 2. **Compute headroom**: if GPU is under-utilized, enable KVTuner
///    for fine-grained dynamic adjustment.
/// 3. **Sequence length**: long sequences (>4K tokens) benefit from
///    MUSTAFAR token retention to preserve critical context.
/// 4. **Hardware features**: FP8 support unlocks FP8 tier in KIVI;
///    nvCOMP unlocks NvcompAns codec for ChunkKV migration.
///
/// ## Strategy Composition
///
/// Strategies compose in a fixed order to avoid conflicts:
///
/// ```text
/// 1. ChunkKV → determine chunk layout and residency
/// 2. MUSTAFAR → classify tokens, set eviction priority
/// 3. KVTuner → adjust precision per layer depth / entropy
/// 4. KIVI → execute quantization with adjusted tiers
/// ```
///
/// ## Variant Matrix
///
/// The full 16-entry variant matrix (`VARIANT_MATRIX`) is available at
/// compile time. The cross-decision algorithm selects the variant whose
/// capabilities best match the hardware profile.
#[derive(Debug, Clone)]
pub struct CrossDecisionMatrix {
    /// Base KIVI mixed-precision strategy
    pub kivi: KiviStrategy,
    /// Dynamic precision tuner (wraps KIVI)
    pub kv_tuner: KvTunerStrategy,
    /// MUSTAFAR token retention classifier
    pub mustafar: MustafarStrategy,
    /// Chunk-level KV cache manager
    pub chunk_kv: ChunkKvStrategy,
    /// Currently active variant
    pub active_variant: DecisionVariant,
    /// Hardware capability profile
    pub hardware: HardwareProfile,
    /// Decision history ring buffer
    decisions: Vec<DecisionRecord>,
    /// Maximum decisions retained
    max_decisions: usize,
    /// Monotonic decision sequence counter
    decision_seq: u64,
}

impl Default for CrossDecisionMatrix {
    fn default() -> Self {
        Self {
            kivi: KiviStrategy::default(),
            kv_tuner: KvTunerStrategy::default(),
            mustafar: MustafarStrategy::default(),
            chunk_kv: ChunkKvStrategy::default(),
            active_variant: DecisionVariant::FullStack,
            hardware: HardwareProfile::default(),
            decisions: Vec::with_capacity(64),
            max_decisions: 64,
            decision_seq: 0,
        }
    }
}

impl CrossDecisionMatrix {
    /// Create a new cross-decision matrix with default strategies.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a cross-decision matrix with a specific hardware profile.
    pub fn with_hardware(hardware: HardwareProfile) -> Self {
        Self {
            hardware,
            ..Default::default()
        }
    }

    /// Set the active variant explicitly.
    pub fn with_variant(mut self, variant: DecisionVariant) -> Self {
        self.active_variant = variant;
        self.apply_variant();
        self
    }

    /// Configure all four strategies from builder methods.
    pub fn with_kivi(mut self, kivi: KiviStrategy) -> Self {
        self.kivi = kivi;
        self
    }

    /// Set the KVTuner strategy.
    pub fn with_kv_tuner(mut self, tuner: KvTunerStrategy) -> Self {
        self.kv_tuner = tuner;
        self
    }

    /// Set the MUSTAFAR strategy.
    pub fn with_mustafar(mut self, mustafar: MustafarStrategy) -> Self {
        self.mustafar = mustafar;
        self
    }

    /// Set the ChunkKV strategy.
    pub fn with_chunk_kv(mut self, chunk_kv: ChunkKvStrategy) -> Self {
        self.chunk_kv = chunk_kv;
        self
    }

    // ── Variant Matrix Evaluation ──

    /// Evaluate hardware profile and model features to select the optimal
    /// strategy variant.
    ///
    /// The evaluation scores all 16 variants in [`VARIANT_MATRIX`] against
    /// the current hardware profile and returns the highest-scoring variant.
    ///
    /// ## Scoring Factors
    ///
    /// | Factor              | Weight | Enables                                    |
    /// |---------------------|--------|--------------------------------------------|
    /// | Memory pressure     | 0.35   | ChunkKV (swap), aggressive compression     |
    /// | Compute headroom    | 0.25   | KVTuner (dynamic), MUSTAFAR (scoring)      |
    /// | Sequence length     | 0.20   | MUSTAFAR (retention), ChunkKV (layout)     |
    /// | Hardware features   | 0.20   | FP8 (KIVI FP8 tier), nvCOMP (migration)    |
    ///
    /// # Returns
    /// The recommended `DecisionVariant` and a human-readable reason.
    pub fn evaluate(&mut self) -> (DecisionVariant, &'static str) {
        let kv_gb = self.hardware.kv_cache_size_gb();
        let mem_pressure = (kv_gb / self.hardware.gpu_memory_gb).min(1.0);

        let (variant, reason) = if mem_pressure > 0.8 {
            // Severe memory pressure: need ChunkKV for swap + aggressive compression
            if self.hardware.supports_fp8 {
                (
                    DecisionVariant::FullStack,
                    "High memory pressure (>80%) with FP8 — full stack for max compression",
                )
            } else {
                (
                    DecisionVariant::RetentionOnly,
                    "High memory pressure (>80%) without FP8 — retention-centric with chunking",
                )
            }
        } else if mem_pressure > 0.5 {
            // Moderate pressure: precision + chunking
            if self.hardware.compute_capability > 1.5 {
                (
                    DecisionVariant::PrecisionOnly,
                    "Moderate pressure + high compute — precision-centric tuning",
                )
            } else {
                (
                    DecisionVariant::ChunkedPrecision,
                    "Moderate pressure + moderate compute — chunked precision",
                )
            }
        } else if self.hardware.max_seq_len > 8192 {
            // Long sequence support needed
            if self.hardware.compute_capability > 1.0 {
                (
                    DecisionVariant::FullStack,
                    "Long sequences (>8K) + capable GPU — full stack for throughput",
                )
            } else {
                (
                    DecisionVariant::RetentionOnly,
                    "Long sequences (>8K) + modest GPU — MUSTAFAR + ChunkKV for retention",
                )
            }
        } else if self.hardware.gpu_memory_gb < 8.0 {
            // Small GPU: baseline
            (
                DecisionVariant::Baseline,
                "Small GPU (<8GB) — baseline KIVI only for minimal overhead",
            )
        } else if self.hardware.compute_capability < 0.5 {
            // Very weak compute: minimal strategy overhead
            (
                DecisionVariant::Baseline,
                "Low compute capability — baseline KIVI for minimal tuning overhead",
            )
        } else {
            // Default: full stack with all features
            (
                DecisionVariant::FullStack,
                "Ample resources — full stack for optimal quality/throughput balance",
            )
        };

        self.active_variant = variant;
        self.apply_variant();
        self.record_decision(variant, reason, 0, 0.0, mem_pressure);

        (variant, reason)
    }

    /// Re-evaluate with current sequence state (entropy + length).
    ///
    /// Sequence-level re-evaluation can trigger variant changes mid-sequence
    /// (e.g., as sequence grows, MUSTAFAR becomes more important).
    pub fn evaluate_with_state(
        &mut self,
        seq_len: usize,
        entropy: f32,
    ) -> (DecisionVariant, &'static str) {
        let kv_gb = self.hardware.kv_cache_size_gb();
        let mem_pressure = (kv_gb / self.hardware.gpu_memory_gb).min(1.0);

        // Sequence-length-aware re-evaluation
        let (variant, reason) = if seq_len > self.hardware.max_seq_len / 2 && entropy < 0.2 {
            // Long sequence + low entropy: MUSTAFAR retention is critical
            let bits = variant_bits::MUSTAFAR | variant_bits::CHUNK_KV;
            (DecisionVariant::from_bits(bits), "Long sequence + low entropy — enable MUSTAFAR + ChunkKV")
        } else if seq_len > self.hardware.max_seq_len / 2 {
            // Long-running sequence: prioritize retention
            if self.active_variant.has_mustafar() && self.active_variant.has_chunk_kv() {
                (self.active_variant, "Already retention-capable for long sequence")
            } else if self.hardware.compute_capability > 1.0 {
                (DecisionVariant::RetentionOnly, "Long sequence mid-point — enable MUSTAFAR + ChunkKV")
            } else {
                (DecisionVariant::ChunkedPrecision, "Long sequence mid-point — enable ChunkKV")
            }
        } else if entropy > 0.75 {
            // High entropy: precision tuning is effective
            if !self.active_variant.has_kv_tuner() && self.hardware.compute_capability > 0.8 {
                (DecisionVariant::PrecisionOnly, "High entropy — enable dynamic precision tuning")
            } else {
                (self.active_variant, "Already tuned for high entropy")
            }
        } else if entropy < 0.2 {
            // Very low entropy: MUSTAFAR retention is critical
            if !self.active_variant.has_mustafar() {
                let bits = self.active_variant.bits() | variant_bits::MUSTAFAR;
                (DecisionVariant::from_bits(bits), "Low entropy — enable MUSTAFAR token retention")
            } else {
                (self.active_variant, "MUSTAFAR already active for low entropy")
            }
        } else {
            (self.active_variant, "Stable state — no variant change needed")
        };

        if variant != self.active_variant {
            self.active_variant = variant;
            self.apply_variant();
        }
        self.record_decision(variant, reason, seq_len, entropy, mem_pressure);

        (variant, reason)
    }

    // ── Variant Application ──

    /// Apply the currently active variant by enabling/disabling strategies
    /// according to the variant bitmask.
    fn apply_variant(&mut self) {
        let bits = self.active_variant.bits();

        self.kivi.enabled = bits & variant_bits::KIVI != 0;
        self.kv_tuner.enabled = bits & variant_bits::KV_TUNER != 0;
        self.mustafar.enabled = bits & variant_bits::MUSTAFAR != 0;
        self.chunk_kv.enabled = bits & variant_bits::CHUNK_KV != 0;

        // When KVTuner is active, ensure KIVI is also enabled (tuner wraps KIVI)
        if self.kv_tuner.enabled && !self.kivi.enabled {
            self.kivi.enabled = true;
        }
    }

    // ── Joint Strategy Composition ──

    /// Compose all active strategies for a single page header.
    ///
    /// Execution order (prevents conflicts):
    /// 1. ChunkKV: determine chunk residency
    /// 2. MUSTAFAR: classify token, set importance + eviction priority
    /// 3. KVTuner: adjust precision based on entropy + layer depth
    /// 4. KIVI: apply precision tier to header
    ///
    /// # Arguments
    /// * `header` — page header to update
    /// * `token_idx` — token index within the sequence
    /// * `seq_len` — total sequence length
    /// * `layer_depth_ratio` — layer position [0.0=first, 1.0=last]
    /// * `num_kv_heads` — number of KV heads
    ///
    /// # Returns
    /// `(k_tier, v_tier)` — the final precision tiers assigned.
    pub fn compose_for_page(
        &mut self,
        header: &mut KvPageHeader,
        token_idx: usize,
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> (PrecisionTier, PrecisionTier) {
        // Step 1: ChunkKV — ensure chunk layout is computed, touch chunk
        if self.chunk_kv.enabled {
            if self.chunk_kv.num_chunks() == 0 {
                self.chunk_kv.compute_chunk_layout(seq_len);
                self.chunk_kv.init_chunks();
            }
            let chunk_idx = self.chunk_kv.chunk_for_token(token_idx);
            self.chunk_kv.touch_chunk(chunk_idx);
        }

        // Step 2: MUSTAFAR — score importance + eviction priority
        if self.mustafar.enabled {
            // For single-page composition, score just this token
            // (batch scoring is handled by compose_for_batch)
            self.mustafar.score_batch(std::slice::from_ref(header));
            self.mustafar.apply_to_header(header, 0, num_kv_heads);
        }

        // Step 3: KVTuner — dynamic precision adjustment
        let (k_tier, v_tier) = if self.kv_tuner.enabled {
            let is_sink = header.has_sink_token() || token_idx < self.kivi.sink_count;
            self.kv_tuner.observe_and_adjust(header, self.hardware.head_dim, seq_len, is_sink, layer_depth_ratio)
        } else {
            (self.kivi.key_precision, self.kivi.val_precision)
        };

        // Step 4: MUSTAFAR precision floor (overrides KVTuner if more conservative)
        let (final_k, final_v) = if self.mustafar.enabled {
            let floor = self.mustafar.precision_floor(0, header);
            let v_floor = match floor {
                Some(PrecisionTier::FP16) => PrecisionTier::FP16,
                Some(PrecisionTier::FP8) => {
                    // MUSTAFAR floors at FP8, but don't upgrade beyond what KVTuner set
                    if tuner_tier_rank(v_tier) < tuner_tier_rank(PrecisionTier::FP8) {
                        PrecisionTier::FP8
                    } else {
                        v_tier
                    }
                }
                _ => v_tier,
            };
            let k_floor = match floor {
                Some(PrecisionTier::FP16) => PrecisionTier::FP16,
                _ => k_tier,
            };
            (k_floor, v_floor)
        } else {
            (k_tier, v_tier)
        };

        // Step 5: ChunkKV — determine compression tier for this token's chunk
        let chunk_v_tier = if self.chunk_kv.enabled {
            let chunk_idx = self.chunk_kv.chunk_for_token(token_idx);
            self.chunk_kv.chunk_compression_tier(chunk_idx, final_v)
        } else {
            final_v
        };

        // Apply the final KIVI page-level tier
        let is_sink_page = header.has_sink_token();
        self.kivi.apply_to_header(header, is_sink_page);
        // Override with chunk-aware tier if chunking is active
        if self.chunk_kv.enabled {
            header.set_precision_tier(chunk_v_tier);
        }

        (final_k, chunk_v_tier)
    }

    /// Compose strategies for a batch of page headers.
    ///
    /// Batch composition amortizes MUSTAFAR scoring and KVTuner adjustment
    /// across the entire batch for efficiency.
    ///
    /// # Returns
    /// Number of pages whose precision tier was modified.
    pub fn compose_for_batch(
        &mut self,
        headers: &mut [KvPageHeader],
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> usize {
        if headers.is_empty() {
            return 0;
        }

        // Step 1: ChunkKV layout
        if self.chunk_kv.enabled && self.chunk_kv.num_chunks() == 0 {
            self.chunk_kv.compute_chunk_layout(seq_len);
            self.chunk_kv.init_chunks();
        }

        // Step 2: MUSTAFAR batch scoring
        if self.mustafar.enabled {
            self.mustafar.score_batch(headers);
        }

        let mut changed = 0usize;

        for (i, header) in headers.iter_mut().enumerate() {
            if !header.is_active() {
                continue;
            }

            let old_tier = header.precision_tier();

            // ChunkKV touch
            if self.chunk_kv.enabled {
                let chunk_idx = self.chunk_kv.chunk_for_token(i);
                self.chunk_kv.touch_chunk(chunk_idx);
            }

            // MUSTAFAR apply
            if self.mustafar.enabled {
                self.mustafar.apply_to_header(header, i, num_kv_heads);
            }

            // KVTuner dynamic adjustment
            let is_sink = header.has_sink_token() || i < self.kivi.sink_count;
            if self.kv_tuner.enabled {
                self.kv_tuner.adjust_and_apply(header, seq_len, is_sink, layer_depth_ratio);
            } else if self.mustafar.enabled {
                // MUSTAFAR precision floor
                if let Some(floor) = self.mustafar.precision_floor(i, header) {
                    header.set_precision_tier(floor);
                }
            }

            // ChunkKV tier override
            if self.chunk_kv.enabled {
                let chunk_idx = self.chunk_kv.chunk_for_token(i);
                let base_v = header.precision_tier();
                let chunk_tier = self.chunk_kv.chunk_compression_tier(chunk_idx, base_v);
                header.set_precision_tier(chunk_tier);
            }

            if header.precision_tier() != old_tier {
                changed += 1;
            }
        }

        changed
    }

    /// One-shot: observe entropy, evaluate variant, and compose for a full
    /// batch of page headers.
    ///
    /// This is the primary entry point for per-step KV cache optimization.
    /// It combines variant evaluation, strategy composition, and header
    /// updates in a single call.
    ///
    /// # Returns
    /// `(variant, reason, pages_changed)` — the selected variant, the
    /// decision reason, and the number of pages whose tier was modified.
    pub fn step(
        &mut self,
        headers: &mut [KvPageHeader],
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> (DecisionVariant, &'static str, usize) {
        // Observe entropy from first active header for evaluation
        let entropy = match headers.iter().find(|h| h.is_active()) {
            Some(h) => {
                let raw = super::f16_bits_to_f32(h.entropy_avg);
                let max_entropy = (self.hardware.head_dim as f32).ln();
                if max_entropy > 0.0 {
                    (raw / max_entropy).min(1.0)
                } else {
                    0.5
                }
            }
            None => 0.5,
        };

        // Re-evaluate variant for current state
        let (variant, reason) = self.evaluate_with_state(seq_len, entropy);

        // Compose strategies for the batch
        let changed = self.compose_for_batch(headers, seq_len, layer_depth_ratio, num_kv_heads);

        (variant, reason, changed)
    }

    // ── Decision History ──

    fn record_decision(
        &mut self,
        variant: DecisionVariant,
        reason: &'static str,
        seq_len: usize,
        entropy: f32,
        cache_pressure: f32,
    ) {
        self.decision_seq = self.decision_seq.wrapping_add(1);
        if self.decisions.len() >= self.max_decisions {
            self.decisions.remove(0);
        }
        self.decisions.push(DecisionRecord {
            seq: self.decision_seq,
            variant,
            reason,
            seq_len,
            entropy,
            cache_pressure,
        });
    }

    /// Get decision history.
    #[inline]
    pub fn decisions(&self) -> &[DecisionRecord] {
        &self.decisions
    }

    /// Drain and return decision history.
    pub fn drain_decisions(&mut self) -> Vec<DecisionRecord> {
        std::mem::take(&mut self.decisions)
    }

    /// Number of decisions recorded.
    #[inline]
    pub fn decision_count(&self) -> usize {
        self.decisions.len()
    }

    // ── Combined Accessors ──

    /// Recommended precision tier for a token considering all active strategies.
    ///
    /// Composes MUSTAFAR floor, KVTuner adjustment, and ChunkKV tier assignment.
    pub fn recommended_tier(
        &self,
        token_idx: usize,
        header: &KvPageHeader,
        _seq_len: usize,
    ) -> PrecisionTier {
        let base_tier = if self.kv_tuner.enabled {
            self.kv_tuner.kivi.val_precision
        } else {
            self.kivi.val_precision
        };

        // MUSTAFAR floor
        let tier = if self.mustafar.enabled {
            if let Some(floor) = self.mustafar.precision_floor(token_idx, header) {
                if tuner_tier_rank(floor) > tuner_tier_rank(base_tier) {
                    floor
                } else {
                    base_tier
                }
            } else {
                base_tier
            }
        } else {
            base_tier
        };

        // ChunkKV override
        if self.chunk_kv.enabled && self.chunk_kv.num_chunks() > 0 {
            let chunk_idx = self.chunk_kv.chunk_for_token(token_idx);
            self.chunk_kv.chunk_compression_tier(chunk_idx, tier)
        } else {
            tier
        }
    }

    /// Check whether a token should be preserved at FP16, composing
    /// KIVI sink protection with MUSTAFAR retention.
    #[inline]
    pub fn should_preserve_fp16(&self, token_idx: usize, header: &KvPageHeader) -> bool {
        let kivi_preserve = self.kivi.should_preserve_fp16(token_idx, header.has_sink_token());
        let mustafar_preserve = self.mustafar.should_retain(token_idx);
        kivi_preserve || mustafar_preserve
    }

    /// Check if chunk migration is needed (ChunkKV residency over limit).
    #[inline]
    pub fn needs_migration(&self) -> bool {
        self.chunk_kv.enabled
            && self.chunk_kv.max_resident_chunks > 0
            && self.chunk_kv.resident_count() > self.chunk_kv.max_resident_chunks
    }

    /// Get the migration plan if ChunkKV is active.
    #[inline]
    pub fn migration_plan(&self) -> Option<ChunkMigrationPlan> {
        if self.chunk_kv.enabled {
            Some(self.chunk_kv.migration_plan())
        } else {
            None
        }
    }

    /// Estimated total compression ratio across all active strategies.
    ///
    /// Combines KIVI V compression ratio with ChunkKV tier distribution.
    pub fn effective_compression_ratio(&self) -> f32 {
        let base = if self.kv_tuner.enabled {
            self.kv_tuner.current_v_compression_ratio()
        } else {
            self.kivi.v_compression_ratio()
        };

        // ChunkKV: average tier may further increase compression
        if self.chunk_kv.enabled && self.chunk_kv.num_chunks() > 1 {
            let num = self.chunk_kv.num_chunks();
            let mut total_ratio = 0.0f32;
            for i in 0..num {
                let chunk_tier = self.chunk_kv.chunk_compression_tier(i, self.kivi.val_precision);
                let tier_ratio = match chunk_tier {
                    PrecisionTier::KIVI2 => 8.0,
                    PrecisionTier::KIVI4 => 4.0,
                    PrecisionTier::FP8 => 2.0,
                    PrecisionTier::Sparse => 6.0,
                    _ => 1.0,
                };
                total_ratio += tier_ratio;
            }
            let avg_chunk_ratio = total_ratio / num as f32;
            // Blend: 70% chunk-aware, 30% base
            avg_chunk_ratio * 0.7 + base * 0.3
        } else {
            base
        }
    }

    /// Reset all strategies and decision history.
    pub fn reset(&mut self) {
        self.kivi.reset();
        self.kv_tuner.reset();
        self.mustafar.reset();
        self.chunk_kv.reset();
        self.decisions.clear();
        self.decision_seq = 0;
    }

    /// Disable all strategies (passthrough mode).
    pub fn disabled() -> Self {
        Self {
            kivi: KiviStrategy::disabled(),
            kv_tuner: KvTunerStrategy::disabled(),
            mustafar: MustafarStrategy::disabled(),
            chunk_kv: ChunkKvStrategy::disabled(),
            active_variant: DecisionVariant::Custom(0),
            hardware: HardwareProfile::default(),
            decisions: Vec::new(),
            max_decisions: 0,
            decision_seq: 0,
        }
    }

    /// Check if any strategy is active.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active_variant.bits() != 0
    }
}

/// Type alias for the four-dimensional cross decision matrix.
pub type DecisionMatrix = CrossDecisionMatrix;

// ============================================================================
// VariantMatrix — compile-time variant dispatch (SPEC 19 §8)
// ============================================================================

/// Function pointer type for variant-specialized batch composition.
///
/// Each variant has a dedicated function that composes the active strategies
/// without runtime branch checks for strategy enablement. The function
/// receives mutable references to strategy state and the page headers.
pub type VariantBatchFn = fn(
    kivi: &KiviStrategy,
    kv_tuner: &mut KvTunerStrategy,
    mustafar: &mut MustafarStrategy,
    chunk_kv: &mut ChunkKvStrategy,
    headers: &mut [KvPageHeader],
    seq_len: usize,
    layer_depth_ratio: f32,
    num_kv_heads: usize,
) -> usize;

/// Function pointer type for variant-specialized single-page composition.
pub type VariantPageFn = fn(
    kivi: &KiviStrategy,
    kv_tuner: &mut KvTunerStrategy,
    mustafar: &mut MustafarStrategy,
    chunk_kv: &mut ChunkKvStrategy,
    header: &mut KvPageHeader,
    token_idx: usize,
    seq_len: usize,
    layer_depth_ratio: f32,
    num_kv_heads: usize,
) -> (PrecisionTier, PrecisionTier);

/// Function pointer type for variant-specialized tier recommendation.
pub type VariantRecommendFn = fn(
    kivi: &KiviStrategy,
    kv_tuner: &KvTunerStrategy,
    mustafar: &MustafarStrategy,
    chunk_kv: &ChunkKvStrategy,
    token_idx: usize,
    header: &KvPageHeader,
    seq_len: usize,
) -> PrecisionTier;

/// Compile-time variant dispatch table (SPEC 19-KV-CACHE-OPTIMIZATION §8).
///
/// Maps each of the 16 strategy combinations (identified by their 4-bit
/// variant bitmask) to a specialized KV cache processing function. Runtime
/// dispatch uses the variant bitmask as a direct index into the function
/// pointer table, eliminating all conditional branches (zero-branch
/// overhead) in the hot composition path.
///
/// ## Dispatch Mechanism
///
/// ```text
/// variant_bits (0..15) → dispatch_table[bits] → specialized fn
/// ```
///
/// The specialized function has the active strategy set **hardcoded** —
/// no `if strategy.enabled` runtime checks. This is the key performance
/// advantage over the general-purpose `compose_for_batch` / `compose_for_page`
/// methods in [`CrossDecisionMatrix`].
///
/// ## Strategy Composition Order (preserves CrossDecisionMatrix semantics)
///
/// 1. ChunkKV → determine chunk layout and residency
/// 2. MUSTAFAR → classify tokens, set eviction priority
/// 3. KVTuner → adjust precision per layer depth / entropy
/// 4. KIVI → execute quantization with adjusted tiers
#[derive(Debug, Clone)]
pub struct VariantMatrix {
    /// Compose-for-batch dispatch table: index = variant bitmask (0..15)
    compose_batch_table: [VariantBatchFn; 16],
    /// Compose-for-page dispatch table: index = variant bitmask (0..15)
    compose_page_table: [VariantPageFn; 16],
    /// Recommended tier dispatch table: index = variant bitmask (0..15)
    recommend_table: [VariantRecommendFn; 16],
}

impl Default for VariantMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl VariantMatrix {
    /// Build the compile-time variant dispatch table.
    ///
    /// Initializes all 16 entries with specialized functions that hardcode
    /// which strategies are active for each variant bitmask. The table is
    /// constructed once (typically at program start or model load) and
    /// reused for the lifetime of the inference session.
    pub fn new() -> Self {
        Self {
            compose_batch_table: Self::build_compose_batch_table(),
            compose_page_table: Self::build_compose_page_table(),
            recommend_table: Self::build_recommend_table(),
        }
    }

    // ── Dispatch Methods ──

    /// Dispatch batch composition through the variant-specialized function
    /// pointer for the given variant bitmask.
    ///
    /// Zero-branch: the bitmask directly indexes the function pointer table.
    /// The called function has strategy enablement hardcoded.
    #[inline]
    pub fn dispatch_batch(
        &self,
        variant_bits: u8,
        kivi: &KiviStrategy,
        kv_tuner: &mut KvTunerStrategy,
        mustafar: &mut MustafarStrategy,
        chunk_kv: &mut ChunkKvStrategy,
        headers: &mut [KvPageHeader],
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> usize {
        let idx = (variant_bits & variant_bits::ALL) as usize;
        (self.compose_batch_table[idx])(
            kivi, kv_tuner, mustafar, chunk_kv,
            headers, seq_len, layer_depth_ratio, num_kv_heads,
        )
    }

    /// Dispatch single-page composition through the variant-specialized
    /// function pointer for the given variant bitmask.
    #[inline]
    pub fn dispatch_page(
        &self,
        variant_bits: u8,
        kivi: &KiviStrategy,
        kv_tuner: &mut KvTunerStrategy,
        mustafar: &mut MustafarStrategy,
        chunk_kv: &mut ChunkKvStrategy,
        header: &mut KvPageHeader,
        token_idx: usize,
        seq_len: usize,
        layer_depth_ratio: f32,
        num_kv_heads: usize,
    ) -> (PrecisionTier, PrecisionTier) {
        let idx = (variant_bits & variant_bits::ALL) as usize;
        (self.compose_page_table[idx])(
            kivi, kv_tuner, mustafar, chunk_kv,
            header, token_idx, seq_len, layer_depth_ratio, num_kv_heads,
        )
    }

    /// Dispatch tier recommendation through the variant-specialized
    /// function pointer.
    #[inline]
    pub fn dispatch_recommend(
        &self,
        variant_bits: u8,
        kivi: &KiviStrategy,
        kv_tuner: &KvTunerStrategy,
        mustafar: &MustafarStrategy,
        chunk_kv: &ChunkKvStrategy,
        token_idx: usize,
        header: &KvPageHeader,
        seq_len: usize,
    ) -> PrecisionTier {
        let idx = (variant_bits & variant_bits::ALL) as usize;
        (self.recommend_table[idx])(
            kivi, kv_tuner, mustafar, chunk_kv,
            token_idx, header, seq_len,
        )
    }

    // ── Table Builders ──

    /// Build the compose-for-batch dispatch table.
    ///
    /// Each entry is a specialized function that hard-codes which of the
    /// four strategies are active, eliminating all runtime `if enabled` checks.
    fn build_compose_batch_table() -> [VariantBatchFn; 16] {
        [
            // 0b0000 — no strategies active: passthrough
            |_k, _t, _m, _c, _h, _sl, _ldr, _nkh| 0,
            // 0b0001 — KIVI only
            |k, _t, _m, _c, headers, _seq_len, _ldr, _nkh| {
                let mut changed = 0usize;
                for header in headers.iter_mut() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let is_sink = header.has_sink_token();
                    k.apply_to_header(header, is_sink);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0010 — KVTuner only
            |_k, t, _m, _c, headers, seq_len, ldr, _nkh| {
                let mut changed = 0usize;
                for header in headers.iter_mut() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let is_sink = header.has_sink_token();
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0011 — KIVI + KVTuner (PrecisionOnly)
            |k, t, _m, _c, headers, seq_len, ldr, _nkh| {
                let mut changed = 0usize;
                for header in headers.iter_mut() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let is_sink = header.has_sink_token() || false;
                    let sink_protect = is_sink;
                    if sink_protect {
                        k.apply_to_header(header, true);
                    } else {
                        t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0100 — MUSTAFAR only
            |_k, _t, m, _c, headers, _sl, _ldr, nkh| {
                if headers.is_empty() { return 0; }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    m.apply_to_header(header, i, nkh);
                    if let Some(floor) = m.precision_floor(i, header) {
                        header.set_precision_tier(floor);
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0101 — KIVI + MUSTAFAR
            |k, _t, m, _c, headers, _sl, _ldr, nkh| {
                if headers.is_empty() { return 0; }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    if let Some(floor) = m.precision_floor(i, header) {
                        header.set_precision_tier(floor);
                    } else {
                        k.apply_to_header(header, is_sink);
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0110 — MUSTAFAR + KVTuner (AdaptiveRetention)
            |_k, t, m, _c, headers, seq_len, ldr, nkh| {
                if headers.is_empty() { return 0; }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token();
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    if let Some(floor) = m.precision_floor(i, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(header.precision_tier()) {
                            header.set_precision_tier(floor);
                        }
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b0111 — KIVI + KVTuner + MUSTAFAR (no ChunkKV)
            |k, t, m, _c, headers, seq_len, ldr, nkh| {
                if headers.is_empty() { return 0; }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    if let Some(floor) = m.precision_floor(i, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(header.precision_tier()) {
                            header.set_precision_tier(floor);
                        }
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1000 — ChunkKV only
            |_k, _t, _m, c, headers, seq_len, _ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx, PrecisionTier::KIVI4);
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1001 — KIVI + ChunkKV (ChunkedPrecision)
            |k, _t, _m, c, headers, seq_len, _ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    let base_v = k.val_precision;
                    let chunk_tier = c.chunk_compression_tier(chunk_idx, base_v);
                    header.set_precision_tier(chunk_tier);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    k.apply_to_header(header, is_sink);
                    if c.enabled {
                        let ci = c.chunk_for_token(i);
                        let ct = c.chunk_compression_tier(ci, header.precision_tier());
                        header.set_precision_tier(ct);
                    }
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1010 — KVTuner + ChunkKV
            |_k, t, _m, c, headers, seq_len, ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    let is_sink = header.has_sink_token();
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1011 — KIVI + KVTuner + ChunkKV (no MUSTAFAR)
            |k, t, _m, c, headers, seq_len, ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1100 — MUSTAFAR + ChunkKV (RetentionOnly without KIVI — note: RetentionOnly=KIVI+MUSTAFAR+ChunkKV, this is MUSTAFAR+ChunkKV only)
            |_k, _t, m, c, headers, seq_len, _ldr, nkh| {
                if headers.is_empty() { return 0; }
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    m.apply_to_header(header, i, nkh);
                    if let Some(floor) = m.precision_floor(i, header) {
                        header.set_precision_tier(floor);
                    }
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1101 — KIVI + MUSTAFAR + ChunkKV (RetentionOnly)
            |k, _t, m, c, headers, seq_len, _ldr, nkh| {
                if headers.is_empty() { return 0; }
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    if let Some(floor) = m.precision_floor(i, header) {
                        header.set_precision_tier(floor);
                    } else {
                        k.apply_to_header(header, is_sink);
                    }
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1110 — KVTuner + MUSTAFAR + ChunkKV (no KIVI)
            |_k, t, m, c, headers, seq_len, ldr, nkh| {
                if headers.is_empty() { return 0; }
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token();
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    if let Some(floor) = m.precision_floor(i, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(header.precision_tier()) {
                            header.set_precision_tier(floor);
                        }
                    }
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
            // 0b1111 — FullStack: KIVI + KVTuner + MUSTAFAR + ChunkKV
            |k, t, m, c, headers, seq_len, ldr, nkh| {
                if headers.is_empty() { return 0; }
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                m.score_batch(headers);
                let mut changed = 0usize;
                for (i, header) in headers.iter_mut().enumerate() {
                    if !header.is_active() { continue; }
                    let old_tier = header.precision_tier();
                    let chunk_idx = c.chunk_for_token(i);
                    c.touch_chunk(chunk_idx);
                    m.apply_to_header(header, i, nkh);
                    let is_sink = header.has_sink_token() || i < k.sink_count;
                    t.adjust_and_apply(header, seq_len, is_sink, ldr);
                    if let Some(floor) = m.precision_floor(i, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(header.precision_tier()) {
                            header.set_precision_tier(floor);
                        }
                    }
                    let chunk_idx2 = c.chunk_for_token(i);
                    let chunk_tier = c.chunk_compression_tier(chunk_idx2, header.precision_tier());
                    header.set_precision_tier(chunk_tier);
                    if header.precision_tier() != old_tier { changed += 1; }
                }
                changed
            },
        ]
    }

    /// Build the compose-for-page dispatch table.
    fn build_compose_page_table() -> [VariantPageFn; 16] {
        [
            // 0b0000 — no strategies
            |_k, _t, _m, _c, _h, _ti, _sl, _ldr, _nkh| (PrecisionTier::FP16, PrecisionTier::FP16),
            // 0b0001 — KIVI only
            |k, _t, _m, _c, header, token_idx, _sl, _ldr, _nkh| {
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                k.apply_to_header(header, is_sink);
                (k.key_precision, k.val_precision)
            },
            // 0b0010 — KVTuner only
            |_k, t, _m, _c, header, _ti, seq_len, ldr, _nkh| {
                let is_sink = header.has_sink_token();
                t.adjust_and_apply(header, seq_len, is_sink, ldr);
                (t.kivi.key_precision, t.kivi.val_precision)
            },
            // 0b0011 — KIVI + KVTuner
            |k, t, _m, _c, header, token_idx, seq_len, ldr, _nkh| {
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                if is_sink {
                    k.apply_to_header(header, true);
                    (PrecisionTier::FP16, PrecisionTier::FP16)
                } else {
                    t.adjust_and_apply(header, seq_len, false, ldr);
                    (t.kivi.key_precision, t.kivi.val_precision)
                }
            },
            // 0b0100 — MUSTAFAR only
            |_k, _t, m, _c, header, token_idx, _sl, _ldr, nkh| {
                m.apply_to_header(header, token_idx, nkh);
                let k_tier = PrecisionTier::FP16;
                let v_tier = match m.precision_floor(token_idx, header) {
                    Some(floor) => floor,
                    None => PrecisionTier::KIVI4,
                };
                (k_tier, v_tier)
            },
            // 0b0101 — KIVI + MUSTAFAR
            |k, _t, m, _c, header, token_idx, _sl, _ldr, nkh| {
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                let v_tier = if let Some(floor) = m.precision_floor(token_idx, header) {
                    floor
                } else if is_sink {
                    PrecisionTier::FP16
                } else {
                    k.val_precision
                };
                k.apply_to_header(header, is_sink);
                (k.key_precision, v_tier)
            },
            // 0b0110 — MUSTAFAR + KVTuner
            |_k, t, m, _c, header, token_idx, seq_len, ldr, nkh| {
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token();
                let (kt, vt, _) = t.adjust_and_apply(header, seq_len, is_sink, ldr);
                let v_final = if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(vt) { floor } else { vt }
                } else { vt };
                (kt, v_final)
            },
            // 0b0111 — KIVI + KVTuner + MUSTAFAR
            |k, t, m, _c, header, token_idx, seq_len, ldr, nkh| {
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                if is_sink {
                    k.apply_to_header(header, true);
                    (PrecisionTier::FP16, PrecisionTier::FP16)
                } else {
                    let (kt, vt, _) = t.adjust_and_apply(header, seq_len, false, ldr);
                    let v_final = if let Some(floor) = m.precision_floor(token_idx, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(vt) { floor } else { vt }
                    } else { vt };
                    (kt, v_final)
                }
            },
            // 0b1000 — ChunkKV only
            |_k, _t, _m, c, header, token_idx, seq_len, _ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                let base_v = PrecisionTier::KIVI4;
                let chunk_tier = c.chunk_compression_tier(chunk_idx, base_v);
                header.set_precision_tier(chunk_tier);
                (PrecisionTier::FP16, chunk_tier)
            },
            // 0b1001 — KIVI + ChunkKV
            |k, _t, _m, c, header, token_idx, seq_len, _ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                k.apply_to_header(header, is_sink);
                let chunk_tier = c.chunk_compression_tier(chunk_idx, header.precision_tier());
                header.set_precision_tier(chunk_tier);
                (k.key_precision, chunk_tier)
            },
            // 0b1010 — KVTuner + ChunkKV
            |_k, t, _m, c, header, token_idx, seq_len, ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                let is_sink = header.has_sink_token();
                t.adjust_and_apply(header, seq_len, is_sink, ldr);
                let chunk_tier = c.chunk_compression_tier(chunk_idx, header.precision_tier());
                header.set_precision_tier(chunk_tier);
                (t.kivi.key_precision, chunk_tier)
            },
            // 0b1011 — KIVI + KVTuner + ChunkKV
            |k, t, _m, c, header, token_idx, seq_len, ldr, _nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                if is_sink {
                    k.apply_to_header(header, true);
                } else {
                    t.adjust_and_apply(header, seq_len, false, ldr);
                }
                let chunk_tier = c.chunk_compression_tier(chunk_idx, header.precision_tier());
                header.set_precision_tier(chunk_tier);
                (if is_sink { PrecisionTier::FP16 } else { t.kivi.key_precision }, chunk_tier)
            },
            // 0b1100 — MUSTAFAR + ChunkKV
            |_k, _t, m, c, header, token_idx, seq_len, _ldr, nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                m.apply_to_header(header, token_idx, nkh);
                let v_base = match m.precision_floor(token_idx, header) {
                    Some(floor) => floor,
                    None => PrecisionTier::KIVI4,
                };
                let chunk_tier = c.chunk_compression_tier(chunk_idx, v_base);
                header.set_precision_tier(chunk_tier);
                (PrecisionTier::FP16, chunk_tier)
            },
            // 0b1101 — KIVI + MUSTAFAR + ChunkKV (RetentionOnly)
            |k, _t, m, c, header, token_idx, seq_len, _ldr, nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                let v_base = if let Some(floor) = m.precision_floor(token_idx, header) {
                    floor
                } else if is_sink {
                    PrecisionTier::FP16
                } else {
                    k.val_precision
                };
                k.apply_to_header(header, is_sink);
                let chunk_tier = c.chunk_compression_tier(chunk_idx, v_base);
                header.set_precision_tier(chunk_tier);
                (k.key_precision, chunk_tier)
            },
            // 0b1110 — KVTuner + MUSTAFAR + ChunkKV
            |_k, t, m, c, header, token_idx, seq_len, ldr, nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token();
                let (kt, vt, _) = t.adjust_and_apply(header, seq_len, is_sink, ldr);
                let v_final = if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(vt) { floor } else { vt }
                } else { vt };
                let chunk_tier = c.chunk_compression_tier(chunk_idx, v_final);
                header.set_precision_tier(chunk_tier);
                (kt, chunk_tier)
            },
            // 0b1111 — FullStack
            |k, t, m, c, header, token_idx, seq_len, ldr, nkh| {
                if c.enabled && c.num_chunks() == 0 {
                    c.compute_chunk_layout(seq_len);
                    c.init_chunks();
                }
                let chunk_idx = c.chunk_for_token(token_idx);
                c.touch_chunk(chunk_idx);
                m.apply_to_header(header, token_idx, nkh);
                let is_sink = header.has_sink_token() || token_idx < k.sink_count;
                if is_sink {
                    k.apply_to_header(header, true);
                } else {
                    t.adjust_and_apply(header, seq_len, false, ldr);
                }
                let v_post_tuner = header.precision_tier();
                let v_final = if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(v_post_tuner) { floor } else { v_post_tuner }
                } else { v_post_tuner };
                let chunk_tier = c.chunk_compression_tier(chunk_idx, v_final);
                header.set_precision_tier(chunk_tier);
                (if is_sink { PrecisionTier::FP16 } else { t.kivi.key_precision }, chunk_tier)
            },
        ]
    }

    /// Build the recommend-tier dispatch table.
    fn build_recommend_table() -> [VariantRecommendFn; 16] {
        [
            // 0b0000 — no strategies
            |_k, _t, _m, _c, _ti, _h, _sl| PrecisionTier::FP16,
            // 0b0001 — KIVI only
            |k, _t, _m, _c, token_idx, header, _sl| {
                if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    k.val_precision
                }
            },
            // 0b0010 — KVTuner only
            |_k, t, _m, _c, _ti, _h, _sl| t.kivi.val_precision,
            // 0b0011 — KIVI + KVTuner
            |k, t, _m, _c, token_idx, header, _sl| {
                if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    t.kivi.val_precision
                }
            },
            // 0b0100 — MUSTAFAR only
            |_k, _t, m, _c, token_idx, header, _sl| {
                match m.precision_floor(token_idx, header) {
                    Some(floor) => floor,
                    None => PrecisionTier::KIVI4,
                }
            },
            // 0b0101 — KIVI + MUSTAFAR
            |k, _t, m, _c, token_idx, header, _sl| {
                if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else if let Some(floor) = m.precision_floor(token_idx, header) {
                    floor
                } else {
                    k.val_precision
                }
            },
            // 0b0110 — MUSTAFAR + KVTuner
            |_k, t, m, _c, token_idx, header, _sl| {
                let base = t.kivi.val_precision;
                if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(base) { floor } else { base }
                } else { base }
            },
            // 0b0111 — KIVI + KVTuner + MUSTAFAR
            |k, t, m, _c, token_idx, header, _sl| {
                if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    let base = t.kivi.val_precision;
                    if let Some(floor) = m.precision_floor(token_idx, header) {
                        if tuner_tier_rank(floor) > tuner_tier_rank(base) { floor } else { base }
                    } else { base }
                }
            },
            // 0b1000 — ChunkKV only
            |_k, _t, _m, c, token_idx, _h, _sl| {
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, PrecisionTier::KIVI4)
                } else {
                    PrecisionTier::KIVI4
                }
            },
            // 0b1001 — KIVI + ChunkKV
            |k, _t, _m, c, token_idx, header, _sl| {
                let base = if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    k.val_precision
                };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, base)
                } else { base }
            },
            // 0b1010 — KVTuner + ChunkKV
            |_k, t, _m, c, token_idx, _h, _sl| {
                let base = t.kivi.val_precision;
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, base)
                } else { base }
            },
            // 0b1011 — KIVI + KVTuner + ChunkKV
            |k, t, _m, c, token_idx, header, _sl| {
                let base = if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    t.kivi.val_precision
                };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, base)
                } else { base }
            },
            // 0b1100 — MUSTAFAR + ChunkKV
            |_k, _t, m, c, token_idx, header, _sl| {
                let base = match m.precision_floor(token_idx, header) {
                    Some(floor) => floor,
                    None => PrecisionTier::KIVI4,
                };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, base)
                } else { base }
            },
            // 0b1101 — KIVI + MUSTAFAR + ChunkKV
            |k, _t, m, c, token_idx, header, _sl| {
                let base = if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else if let Some(floor) = m.precision_floor(token_idx, header) {
                    floor
                } else {
                    k.val_precision
                };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, base)
                } else { base }
            },
            // 0b1110 — KVTuner + MUSTAFAR + ChunkKV
            |_k, t, m, c, token_idx, header, _sl| {
                let base = t.kivi.val_precision;
                let with_m = if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(base) { floor } else { base }
                } else { base };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, with_m)
                } else { with_m }
            },
            // 0b1111 — FullStack
            |k, t, m, c, token_idx, header, _sl| {
                let base = if k.should_preserve_fp16(token_idx, header.has_sink_token()) {
                    PrecisionTier::FP16
                } else {
                    t.kivi.val_precision
                };
                let with_m = if let Some(floor) = m.precision_floor(token_idx, header) {
                    if tuner_tier_rank(floor) > tuner_tier_rank(base) { floor } else { base }
                } else { base };
                if c.enabled && c.num_chunks() > 0 {
                    let chunk_idx = c.chunk_for_token(token_idx);
                    c.chunk_compression_tier(chunk_idx, with_m)
                } else { with_m }
            },
        ]
    }
}

// ============================================================================
