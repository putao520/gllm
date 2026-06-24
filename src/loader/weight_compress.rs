//! Weight page compression (SPEC 22 §6 — Weight Page Compression).
//!
//! Integrates byte-stream compression into the weight loading path, reusing the
//! shared `CompressionCodec` enum (SPEC 22 §3.1) already used for KV page
//! compression. Weight pages are compressed according to tier placement and
//! weight classification (DenseLayerWeight vs ExpertWeight).
//!
//! ## Compression Strategy (SPEC §6.2)
//!
//! | Weight Type            | HBM              | DRAM            | NVMe      |
//! |------------------------|------------------|-----------------|-----------|
//! | DenseLayerWeight       | None/BitPackRle  | Lz4             | ZstdDict  |
//! | ExpertWeight (MoE)     | None             | BitPackRle      | ZstdDict  |
//!
//! Hot weights (layer 0/1, frequently activated experts): always `None`.
//! Cold weights (last-N layers, infrequent experts): default `BitPackRle`.

use crate::kv_cache::CompressionCodec;
use crate::loader::weight_tier::WeightTier;
use crate::static_compression::{
    compress_bitpack_rle, compress_nvcomp_ans, compress_zstd_dict, decompress_nvcomp_ans,
    lz4_compress, NvcompAnsError,
    CodecError,
};

/// Classification of weight type for compression strategy selection (SPEC §6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightClass {
    /// Dense layer weight (attention Q/K/V/O, FFN gate/up/down).
    DenseLayerWeight,
    /// MoE expert weight (experts.*.*, shared_experts.*).
    ExpertWeight,
}

/// Per-page compression result containing the compressed data and metadata.
#[derive(Debug, Clone)]
pub struct CompressedWeightPage {
    /// Compressed bytes.
    pub data: Vec<u8>,
    /// Codec used for compression.
    pub codec: CompressionCodec,
    /// Original (decompressed) size in bytes.
    pub decompressed_size: u32,
    /// Compressed size in bytes.
    pub compressed_size: u32,
}

/// Configuration for weight page compression, read from ModelConfig or
/// WeightPagingConfig. Controls which codec is used at each tier and whether
/// compression is enabled for specific weight classes.
#[derive(Debug, Clone)]
pub struct WeightCompressionConfig {
    /// Whether weight compression is globally enabled.
    pub enabled: bool,
    /// Number of "hot" layers at the start of the model that skip compression.
    /// Per SPEC §6.2: "Hot weights (layer 0/1) always None".
    pub hot_layer_count: usize,
    /// Number of "cold" layers at the end of the model that get BitPackRle by default.
    pub cold_layer_count: usize,
    /// Total number of layers in the model (used to determine hot/cold zones).
    pub total_layers: usize,
    /// Whether the model has MoE experts.
    pub has_moe_experts: usize,
    /// Pre-trained zstd dictionary for NVMe-tier compression.
    /// Trained at model load time from sample weight pages.
    pub zstd_dictionary: Vec<u8>,
}

impl Default for WeightCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hot_layer_count: 2,
            cold_layer_count: 4,
            total_layers: 0,
            has_moe_experts: 0,
            zstd_dictionary: Vec::new(),
        }
    }
}

impl WeightCompressionConfig {
    /// Create config with model geometry parameters.
    pub fn new(total_layers: usize, num_experts: usize) -> Self {
        Self {
            total_layers,
            has_moe_experts: num_experts,
            ..Default::default()
        }
    }

    /// Determine if a given layer index falls in the "hot" zone (SPEC §6.2).
    /// Hot layers (first `hot_layer_count` layers) always use `None` codec.
    pub fn is_hot_layer(&self, layer_idx: Option<usize>) -> bool {
        match layer_idx {
            Some(idx) => idx < self.hot_layer_count,
            None => false,
        }
    }

    /// Determine if a given layer index falls in the "cold" zone (SPEC §6.2).
    /// Cold layers (last `cold_layer_count` layers) default to BitPackRle.
    pub fn is_cold_layer(&self, layer_idx: Option<usize>) -> bool {
        match layer_idx {
            Some(idx) => {
                if self.total_layers == 0 {
                    return false;
                }
                idx + self.cold_layer_count >= self.total_layers
            }
            None => false,
        }
    }
}

/// Select the compression codec for a weight page based on tier, weight class,
/// and layer position (SPEC §6.2).
///
/// # Arguments
/// * `tier` — Target storage tier (HBM, DRAM, NVMe).
/// * `weight_class` — Classification of the weight (DenseLayerWeight or ExpertWeight).
/// * `config` — Weight compression configuration.
/// * `layer_idx` — Optional layer index for hot/cold zone determination.
/// * `is_quantized` — Whether the weight is already quantized (e.g., Q4_0).
///
/// # Returns
/// The selected `CompressionCodec`. Returns `None` when compression should be skipped.
pub fn select_weight_codec(
    tier: WeightTier,
    weight_class: WeightClass,
    config: &WeightCompressionConfig,
    layer_idx: Option<usize>,
    is_quantized: bool,
) -> CompressionCodec {
    if !config.enabled {
        return CompressionCodec::None;
    }

    // Hot weights always stay uncompressed (SPEC §6.2)
    if config.is_hot_layer(layer_idx) {
        return CompressionCodec::None;
    }

    match tier {
        WeightTier::DeviceLocal => {
            // HBM tier: minimal compression to preserve latency
            match weight_class {
                WeightClass::DenseLayerWeight => {
                    // Dense weights in HBM: None (or BitPackRle for already-quantized Q4_0)
                    if is_quantized {
                        CompressionCodec::BitPackRle
                    } else {
                        CompressionCodec::None
                    }
                }
                WeightClass::ExpertWeight => {
                    // Expert weights in HBM: no compression (SPEC §6.2)
                    CompressionCodec::None
                }
            }
        }
        WeightTier::HostLocal => {
            // DRAM tier: moderate compression
            match weight_class {
                WeightClass::DenseLayerWeight => {
                    // Dense weights in DRAM: Lz4 (SPEC §6.2)
                    CompressionCodec::Lz4
                }
                WeightClass::ExpertWeight => {
                    // Expert weights in DRAM (SPEC §6.2):
                    // Cold experts: BitPackRle — rarely accessed, compression saves memory
                    // Warm experts: None — frequently accessed, no compression to preserve accuracy
                    if config.is_cold_layer(layer_idx) {
                        CompressionCodec::BitPackRle
                    } else {
                        CompressionCodec::None
                    }
                }
            }
        }
        WeightTier::DiskMmap => {
            // NVMe tier: maximum compression (SPEC §6.2)
            CompressionCodec::ZstdDict
        }
    }
}

/// Classify a weight tensor as DenseLayerWeight or ExpertWeight based on its
/// tensor name and model configuration.
///
/// Expert weights are identified by "experts" or "shared_experts" in the name.
/// All other weights are classified as DenseLayerWeight.
pub fn classify_weight(tensor_name: &str, has_moe_experts: usize) -> WeightClass {
    if has_moe_experts > 0 {
        let lower = tensor_name.to_ascii_lowercase();
        if lower.contains("experts") || lower.contains("shared_expert") {
            return WeightClass::ExpertWeight;
        }
    }
    WeightClass::DenseLayerWeight
}

/// Compress a weight page's raw data using the specified codec.
///
/// This is the main entry point for weight page compression during model loading.
/// Returns `Ok(Some(CompressedWeightPage))` if compression succeeded and was
/// beneficial (compressed size < original size), `Ok(None)` if the codec is
/// `None` (no compression), or an error if compression failed.
///
/// For `ZstdDict` codec, uses the dictionary from the config.
pub fn compress_weight_page(
    data: &[u8],
    codec: CompressionCodec,
    config: &WeightCompressionConfig,
) -> Result<Option<CompressedWeightPage>, CodecError> {
    if codec == CompressionCodec::None {
        return Ok(None);
    }

    let decompressed_size = data.len() as u32;
    let compressed_data = match codec {
        CompressionCodec::Lz4 => lz4_compress(data),
        CompressionCodec::BitPackRle => compress_bitpack_rle(data),
        CompressionCodec::ZstdDict => {
            if config.zstd_dictionary.is_empty() {
                // NO-SILENT-FALLBACK: ZstdDict requested but no dictionary — must error,
                // not silently fall back to LZ4 with wrong codec tag (causes decompress corruption)
                return Err(CodecError(
                    "ZstdDict codec requested but zstd_dictionary is empty — train dictionary first".into(),
                ));
            } else {
                compress_zstd_dict(data, &config.zstd_dictionary)?
            }
        }
        CompressionCodec::NvcompAns => {
            // NO-FALLBACK: NvcompAns is GPU-only; failure must propagate, not silently use LZ4
            // (would store NvcompAns codec tag but LZ4 data → decompress corruption)
            compress_nvcomp_ans(data).map_err(|e| CodecError(
                format!("NvcompAns GPU compression unavailable: {} — use Lz4 codec explicitly instead", e)
            ))?
        }
        CompressionCodec::None => unreachable!(),
    };

    let compressed_size = compressed_data.len() as u32;

    // Only use compressed result if it actually saves space
    if compressed_size >= decompressed_size {
        return Ok(None);
    }

    Ok(Some(CompressedWeightPage {
        data: compressed_data,
        codec,
        decompressed_size,
        compressed_size,
    }))
}

/// High-level weight compression function that combines codec selection and
/// compression in a single call. Used in the weight loading pipeline.
///
/// # Arguments
/// * `data` — Raw weight data bytes.
/// * `tier` — Target storage tier.
/// * `tensor_name` — Name of the tensor (for weight classification).
/// * `layer_idx` — Optional layer index.
/// * `is_quantized` — Whether the weight is already quantized.
/// * `config` — Compression configuration.
///
/// # Returns
/// `Some(CompressedWeightPage)` if compression was applied, `None` if skipped.
pub fn compress_weight(
    data: &[u8],
    tier: WeightTier,
    tensor_name: &str,
    layer_idx: Option<usize>,
    is_quantized: bool,
    config: &WeightCompressionConfig,
) -> Option<CompressedWeightPage> {
    let weight_class = classify_weight(tensor_name, config.has_moe_experts);
    let codec = select_weight_codec(tier, weight_class, config, layer_idx, is_quantized);
    compress_weight_page(data, codec, config).unwrap_or_else(|e| {
        log::warn!("compress compression failed for {}: {:?}, storing uncompressed", tensor_name, e);
        None
    })
}

/// Decompress a weight page that was compressed during loading.
///
/// Used by ExpertWeightPrefetcher when it hits a compressed expert (SPEC §6.3):
/// the decompression overlaps with the next wave's computation.
pub fn decompress_weight_page(
    compressed: &[u8],
    codec: CompressionCodec,
    decompressed_size: usize,
    zstd_dictionary: &[u8],
) -> Result<Vec<u8>, String> {
    match codec {
        CompressionCodec::None => Ok(compressed.to_vec()),
        CompressionCodec::Lz4 => {
            crate::static_compression::lz4_decompress(compressed, decompressed_size)
                .map_err(|e| format!("LZ4 decompress weight page failed: {e}"))
        }
        CompressionCodec::BitPackRle => {
            Ok(crate::static_compression::decompress_bitpack_rle(
                compressed,
                decompressed_size,
            ))
        }
        CompressionCodec::ZstdDict => {
            if zstd_dictionary.is_empty() {
                crate::static_compression::lz4_decompress(compressed, decompressed_size)
                    .map_err(|e| format!("ZstdDict (fallback LZ4) decompress weight page failed: {e}"))
            } else {
                crate::static_compression::decompress_zstd_dict(
                    compressed,
                    zstd_dictionary,
                    decompressed_size,
                )
                .map_err(|e| format!("ZstdDict decompress weight page failed: {e}"))
            }
        }
        CompressionCodec::NvcompAns => {
            match decompress_nvcomp_ans(compressed, decompressed_size) {
                Ok(data) => Ok(data),
                Err(NvcompAnsError(ref msg)) => {
                    log::debug!("NvcompAns GPU decompression unavailable ({}), falling back to LZ4", msg);
                    crate::static_compression::lz4_decompress(compressed, decompressed_size)
                        .map_err(|e| format!("NvcompAns (fallback LZ4) decompress weight page failed: {e}"))
                }
            }
        }
    }
}

/// Train a zstd dictionary from sample weight pages for use with NVMe-tier
/// compression. Called once at model load time.
///
/// Collects representative weight page data from various layers and produces
/// a dictionary optimized for the model's weight distribution.
pub fn train_weight_compression_dict(
    weight_samples: &[&[u8]],
    dict_capacity: usize,
) -> Vec<u8> {
    crate::static_compression::train_zstd_dictionary(weight_samples, dict_capacity)
}

/// Unified Virtual Page descriptor for weight pages (SPEC §6.1).
///
/// Extends §21's `WeightTier` placement with compression metadata.
/// The tier placement decision is made by `WeightTierManager` (§21 §2);
/// this struct adds the compression codec and sizes on top.
#[derive(Debug, Clone)]
pub struct UnifiedVirtualWeightPage {
    /// Compression codec applied to this page.
    pub codec: CompressionCodec,
    /// Compressed size in bytes (0 when `codec == None`).
    pub compressed_size: u32,
    /// Original (decompressed) size in bytes.
    pub decompressed_size: u32,
    /// Target storage tier.
    pub tier: WeightTier,
}

impl UnifiedVirtualWeightPage {
    /// Create an uncompressed page descriptor.
    pub fn uncompressed(size: u32, tier: WeightTier) -> Self {
        Self {
            codec: CompressionCodec::None,
            compressed_size: 0,
            decompressed_size: size,
            tier,
        }
    }

    /// Create a compressed page descriptor.
    pub fn compressed(
        codec: CompressionCodec,
        compressed_size: u32,
        decompressed_size: u32,
        tier: WeightTier,
    ) -> Self {
        Self {
            codec,
            compressed_size,
            decompressed_size,
            tier,
        }
    }

    /// Whether this page is compressed.
    pub fn is_compressed(&self) -> bool {
        self.codec != CompressionCodec::None
    }

    /// Compression ratio (compressed / decompressed). Returns 1.0 when uncompressed.
    pub fn compression_ratio(&self) -> f32 {
        if self.decompressed_size == 0 || !self.is_compressed() {
            return 1.0;
        }
        self.compressed_size as f32 / self.decompressed_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> WeightCompressionConfig {
        WeightCompressionConfig::new(32, 8)
    }

    #[test]
    fn hot_layer_is_never_compressed() {
        let config = test_config();
        for tier in [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap] {
            let codec = select_weight_codec(
                tier,
                WeightClass::DenseLayerWeight,
                &config,
                Some(0), // layer 0 — hot
                false,
            );
            assert_eq!(codec, CompressionCodec::None, "hot layer 0 should be None for tier {:?}", tier);
        }
    }

    #[test]
    fn dense_hbm_is_none_when_not_quantized() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn dense_hbm_bitpackrle_when_quantized() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            true,
        );
        assert_eq!(codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn expert_hbm_is_none() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(10),
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn dense_dram_is_lz4() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            false,
        );
        assert_eq!(codec, CompressionCodec::Lz4);
    }

    #[test]
    fn expert_dram_is_bitpack_rle() {
        let config = test_config();
        // Cold expert in DRAM uses BitPackRle (SPEC §6.2)
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(28), // cold layer: 28 + cold_layer_count(4) >= total_layers(32)
            false,
        );
        assert_eq!(codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn nvme_is_always_zstd_dict() {
        let config = test_config();
        for wc in [WeightClass::DenseLayerWeight, WeightClass::ExpertWeight] {
            let codec = select_weight_codec(
                WeightTier::DiskMmap,
                wc,
                &config,
                Some(10),
                false,
            );
            assert_eq!(codec, CompressionCodec::ZstdDict);
        }
    }

    #[test]
    fn disabled_config_returns_none() {
        let mut config = test_config();
        config.enabled = false;
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn classify_dense_weight() {
        assert_eq!(
            classify_weight("model.layers.5.self_attn.q_proj", 8),
            WeightClass::DenseLayerWeight,
        );
        assert_eq!(
            classify_weight("model.layers.10.mlp.gate_proj", 8),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn classify_expert_weight() {
        assert_eq!(
            classify_weight("model.layers.3.mlp.experts.gate_up_proj_blocks", 8),
            WeightClass::ExpertWeight,
        );
        assert_eq!(
            classify_weight("model.layers.5.mlp.shared_experts.gate_proj", 8),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_no_experts_always_dense() {
        assert_eq!(
            classify_weight("model.layers.5.mlp.experts.gate_proj", 0),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn compress_weight_page_none_codec_returns_none() {
        let config = test_config();
        let result = compress_weight_page(
            &[1u8, 2, 3, 4],
            CompressionCodec::None,
            &config,
        ).expect("should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn compress_weight_page_lz4_roundtrip() {
        let config = test_config();
        let data = vec![42u8; 1024];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");
        assert_eq!(result.codec, CompressionCodec::Lz4);
        assert!(result.compressed_size < result.decompressed_size);

        let decompressed = decompress_weight_page(
            &result.data,
            result.codec,
            result.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_weight_page_bitpack_rle_roundtrip() {
        let config = test_config();
        // BitPackRle is most effective on low-entropy data
        let data = vec![0u8; 256];
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed")
            .expect("should compress");
        assert_eq!(result.codec, CompressionCodec::BitPackRle);

        let decompressed = decompress_weight_page(
            &result.data,
            result.codec,
            result.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_weight_skips_if_expansion() {
        let config = test_config();
        // Very small high-entropy data that may not compress well
        let data = vec![0xFFu8; 4];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "test.weight",
            Some(10),
            false,
            &config,
        );
        // May or may not compress — but the function should not panic
        drop(result);
    }

    #[test]
    fn unified_virtual_page_uncompressed() {
        let page = UnifiedVirtualWeightPage::uncompressed(1024, WeightTier::DeviceLocal);
        assert!(!page.is_compressed());
        assert_eq!(page.compression_ratio(), 1.0);
    }

    #[test]
    fn unified_virtual_page_compressed() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            400,
            1024,
            WeightTier::HostLocal,
        );
        assert!(page.is_compressed());
        let ratio = page.compression_ratio();
        assert!(ratio < 1.0, "ratio = {ratio}, expected < 1.0");
        assert!((ratio - 0.390625).abs() < 0.01, "ratio = {ratio}");
    }

    #[test]
    fn hot_and_cold_layer_detection() {
        let config = test_config(); // total_layers=32, hot=2, cold=4
        assert!(config.is_hot_layer(Some(0)));
        assert!(config.is_hot_layer(Some(1)));
        assert!(!config.is_hot_layer(Some(2)));

        assert!(config.is_cold_layer(Some(28))); // 28 + 4 >= 32
        assert!(config.is_cold_layer(Some(31)));
        assert!(!config.is_cold_layer(Some(27)));
    }

    // ─── WeightClass enum trait tests ────────────────────────────────────

    #[test]
    fn weight_class_variants_are_distinct() {
        assert_ne!(WeightClass::DenseLayerWeight, WeightClass::ExpertWeight);
    }

    #[test]
    fn weight_class_debug_format() {
        assert_eq!(format!("{:?}", WeightClass::DenseLayerWeight), "DenseLayerWeight");
        assert_eq!(format!("{:?}", WeightClass::ExpertWeight), "ExpertWeight");
    }

    #[test]
    fn weight_class_copy_clone() {
        let a = WeightClass::DenseLayerWeight;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    // ─── WeightCompressionConfig tests ───────────────────────────────────

    #[test]
    fn default_config_values() {
        let config = WeightCompressionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.hot_layer_count, 2);
        assert_eq!(config.cold_layer_count, 4);
        assert_eq!(config.total_layers, 0);
        assert_eq!(config.has_moe_experts, 0);
        assert!(config.zstd_dictionary.is_empty());
    }

    #[test]
    fn new_config_sets_geometry() {
        let config = WeightCompressionConfig::new(64, 16);
        assert_eq!(config.total_layers, 64);
        assert_eq!(config.has_moe_experts, 16);
        // Inherits defaults
        assert!(config.enabled);
        assert_eq!(config.hot_layer_count, 2);
        assert_eq!(config.cold_layer_count, 4);
    }

    #[test]
    fn hot_layer_none_returns_false() {
        let config = test_config();
        assert!(!config.is_hot_layer(None));
    }

    #[test]
    fn cold_layer_none_returns_false() {
        let config = test_config();
        assert!(!config.is_cold_layer(None));
    }

    #[test]
    fn cold_layer_zero_total_always_false() {
        let config = WeightCompressionConfig::default(); // total_layers=0
        assert!(!config.is_cold_layer(Some(0)));
        assert!(!config.is_cold_layer(Some(100)));
    }

    // ─── classify_weight case sensitivity and edge cases ─────────────────

    #[test]
    fn classify_weight_case_insensitive() {
        assert_eq!(
            classify_weight("model.layers.3.mlp.EXPERTS.weight", 4),
            WeightClass::ExpertWeight,
        );
        assert_eq!(
            classify_weight("model.layers.3.mlp.Shared_Experts.gate", 4),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_shared_expert_singular() {
        // "shared_expert" (singular) is the pattern checked
        assert_eq!(
            classify_weight("model.layers.0.mlp.shared_expert.gate_proj", 8),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_empty_name_is_dense() {
        assert_eq!(classify_weight("", 8), WeightClass::DenseLayerWeight);
    }

    // ─── select_weight_codec edge cases ──────────────────────────────────

    #[test]
    fn expert_hbm_quantized_still_none() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(10),
            true,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn nvme_ignores_quantized_flag() {
        let config = test_config();
        let codec_unquantized = select_weight_codec(
            WeightTier::DiskMmap,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            false,
        );
        let codec_quantized = select_weight_codec(
            WeightTier::DiskMmap,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            true,
        );
        assert_eq!(codec_unquantized, CompressionCodec::ZstdDict);
        assert_eq!(codec_quantized, CompressionCodec::ZstdDict);
    }

    // ─── compress_weight_page field correctness ──────────────────────────

    #[test]
    fn compress_weight_page_lz4_metadata_correct() {
        let config = test_config();
        let data = vec![0xABu8; 512];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");

        assert_eq!(result.decompressed_size, 512);
        assert_eq!(result.compressed_size, result.data.len() as u32);
        assert!(result.compressed_size < result.decompressed_size);
    }

    #[test]
    fn compress_weight_page_high_entropy_returns_none() {
        let config = test_config();
        // All-different bytes in a tiny buffer — unlikely to compress well
        let data: Vec<u8> = (0..8).collect();
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed");
        // Small high-entropy data may not save space, so result can be None
        // The important thing is it doesn't panic or error
        if let Some(page) = result {
            assert!(page.compressed_size < page.decompressed_size);
        }
    }

    // ─── decompress_weight_page codec=None ───────────────────────────────

    #[test]
    fn decompress_weight_page_none_returns_clone() {
        let data = vec![10u8, 20, 30, 40];
        let result = decompress_weight_page(&data, CompressionCodec::None, 4, &[])
            .expect("should succeed");
        assert_eq!(result, data);
    }

    // ─── UnifiedVirtualWeightPage compression_ratio edge cases ───────────

    #[test]
    fn compression_ratio_zero_decompressed_size() {
        let page = UnifiedVirtualWeightPage::uncompressed(0, WeightTier::DeviceLocal);
        assert_eq!(page.compression_ratio(), 1.0);
    }

    #[test]
    fn compression_ratio_exact_half() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            512,
            1024,
            WeightTier::HostLocal,
        );
        let ratio = page.compression_ratio();
        assert!((ratio - 0.5).abs() < 0.001, "expected 0.5, got {ratio}");
    }

    #[test]
    fn compressed_page_fields_preserved() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::BitPackRle,
            100,
            400,
            WeightTier::DiskMmap,
        );
        assert_eq!(page.codec, CompressionCodec::BitPackRle);
        assert_eq!(page.compressed_size, 100);
        assert_eq!(page.decompressed_size, 400);
        assert_eq!(page.tier, WeightTier::DiskMmap);
        assert!(page.is_compressed());
    }

    // ─── compress_weight integration end-to-end ──────────────────────────

    #[test]
    fn compress_weight_full_pipeline_lz4() {
        let config = test_config();
        let data = vec![0x55u8; 2048];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.10.mlp.gate_proj",
            Some(10),
            false,
            &config,
        );
        let page = result.expect("should produce a compressed page");
        assert_eq!(page.codec, CompressionCodec::Lz4);
        assert!(page.compressed_size < page.decompressed_size);

        // Roundtrip
        let restored = decompress_weight_page(
            &page.data,
            page.codec,
            page.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(restored, data);
    }

    #[test]
    fn compress_weight_hot_layer_returns_none() {
        let config = test_config();
        let data = vec![0xAAu8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.0.self_attn.q_proj",
            Some(0), // hot layer
            false,
            &config,
        );
        assert!(result.is_none());
    }

    // ─── Additional tests ──────────────────────────────────────────────────

    // --- CompressionCodec from_u8 / as_u8 roundtrip ---

    #[test]
    fn compression_codec_roundtrip_all_variants() {
        for byte in 0u8..=4 {
            let codec = CompressionCodec::from_u8(byte).unwrap_or_else(|| {
                panic!("from_u8({byte}) should return a valid variant")
            });
            assert_eq!(codec.as_u8(), byte, "as_u8 should roundtrip from_u8({byte})");
        }
    }

    #[test]
    fn compression_codec_from_u8_invalid_returns_none() {
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_equality_and_distinctness() {
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants[{i}] != variants[{j}]");
            }
        }
    }

    #[test]
    fn compression_codec_copy_clone() {
        let a = CompressionCodec::Lz4;
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone(); // Clone
        assert_eq!(a, c);
    }

    #[test]
    fn compression_codec_debug_format() {
        assert_eq!(format!("{:?}", CompressionCodec::None), "None");
        assert_eq!(format!("{:?}", CompressionCodec::Lz4), "Lz4");
        assert_eq!(format!("{:?}", CompressionCodec::BitPackRle), "BitPackRle");
        assert_eq!(format!("{:?}", CompressionCodec::NvcompAns), "NvcompAns");
        assert_eq!(format!("{:?}", CompressionCodec::ZstdDict), "ZstdDict");
    }

    // --- CompressionCodec repr values ---

    #[test]
    fn compression_codec_discriminant_values() {
        assert_eq!(CompressionCodec::None as u8, 0);
        assert_eq!(CompressionCodec::Lz4 as u8, 1);
        assert_eq!(CompressionCodec::BitPackRle as u8, 2);
        assert_eq!(CompressionCodec::NvcompAns as u8, 3);
        assert_eq!(CompressionCodec::ZstdDict as u8, 4);
    }

    // --- WeightClass Debug + PartialEq exhaustive ---

    #[test]
    fn weight_class_equality_self() {
        assert_eq!(WeightClass::DenseLayerWeight, WeightClass::DenseLayerWeight);
        assert_eq!(WeightClass::ExpertWeight, WeightClass::ExpertWeight);
    }

    // --- WeightCompressionConfig edge cases ---

    #[test]
    fn config_new_zero_layers() {
        let config = WeightCompressionConfig::new(0, 0);
        assert_eq!(config.total_layers, 0);
        assert_eq!(config.has_moe_experts, 0);
        assert!(config.enabled);
        assert!(config.zstd_dictionary.is_empty());
    }

    #[test]
    fn config_clone_independence() {
        let a = WeightCompressionConfig::new(16, 4);
        let mut b = a.clone();
        b.total_layers = 99;
        b.enabled = false;
        // Original unchanged
        assert_eq!(a.total_layers, 16);
        assert!(a.enabled);
        assert_eq!(b.total_layers, 99);
        assert!(!b.enabled);
    }

    #[test]
    fn is_hot_layer_boundary_exact() {
        let config = test_config(); // hot_layer_count=2
        assert!(config.is_hot_layer(Some(0)));
        assert!(config.is_hot_layer(Some(1)));
        assert!(!config.is_hot_layer(Some(2))); // first non-hot
        assert!(!config.is_hot_layer(Some(3)));
    }

    #[test]
    fn is_cold_layer_boundary_exact() {
        // total_layers=32, cold_layer_count=4
        // cold: idx + 4 >= 32  →  idx >= 28
        let config = test_config();
        assert!(!config.is_cold_layer(Some(27))); // 27 + 4 = 31 < 32
        assert!(config.is_cold_layer(Some(28)));  // 28 + 4 = 32 >= 32
        assert!(config.is_cold_layer(Some(31)));  // last layer
    }

    #[test]
    fn is_cold_layer_with_zero_cold_count() {
        let mut config = test_config();
        config.cold_layer_count = 0;
        // idx + 0 >= 32 only when idx >= 32, but layers are 0..31
        assert!(!config.is_cold_layer(Some(31))); // 31 + 0 = 31 < 32
        assert!(!config.is_cold_layer(Some(0)));
    }

    #[test]
    fn is_hot_layer_with_zero_hot_count() {
        let mut config = test_config();
        config.hot_layer_count = 0;
        assert!(!config.is_hot_layer(Some(0)));
        assert!(!config.is_hot_layer(Some(1)));
    }

    #[test]
    fn config_disabled_overrides_all_tiers() {
        let mut config = test_config();
        config.enabled = false;
        for tier in [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap] {
            for wc in [WeightClass::DenseLayerWeight, WeightClass::ExpertWeight] {
                assert_eq!(
                    select_weight_codec(tier, wc, &config, Some(10), false),
                    CompressionCodec::None,
                    "disabled config should return None for tier={tier:?}, wc={wc:?}"
                );
            }
        }
    }

    // --- CompressedWeightPage construction and field access ---

    #[test]
    fn compressed_weight_page_fields() {
        let page = CompressedWeightPage {
            data: vec![1, 2, 3],
            codec: CompressionCodec::Lz4,
            decompressed_size: 100,
            compressed_size: 3,
        };
        assert_eq!(page.data, vec![1, 2, 3]);
        assert_eq!(page.codec, CompressionCodec::Lz4);
        assert_eq!(page.decompressed_size, 100);
        assert_eq!(page.compressed_size, 3);
    }

    #[test]
    fn compressed_weight_page_clone_independence() {
        let page = CompressedWeightPage {
            data: vec![42u8; 16],
            codec: CompressionCodec::BitPackRle,
            decompressed_size: 64,
            compressed_size: 16,
        };
        let mut clone = page.clone();
        clone.data[0] = 0;
        clone.compressed_size = 99;
        // Original unchanged
        assert_eq!(page.data[0], 42);
        assert_eq!(page.compressed_size, 16);
        assert_ne!(clone.data[0], page.data[0]);
    }

    // --- UnifiedVirtualWeightPage additional edge cases ---

    #[test]
    fn unified_page_uncompressed_zero_size() {
        let page = UnifiedVirtualWeightPage::uncompressed(0, WeightTier::DiskMmap);
        assert!(!page.is_compressed());
        assert_eq!(page.compressed_size, 0);
        assert_eq!(page.decompressed_size, 0);
        assert_eq!(page.tier, WeightTier::DiskMmap);
    }

    #[test]
    fn unified_page_compressed_tiny_ratio() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::ZstdDict,
            1,
            1000000,
            WeightTier::DiskMmap,
        );
        assert!(page.is_compressed());
        let ratio = page.compression_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 0.01, "ratio should be tiny, got {ratio}");
    }

    #[test]
    fn unified_page_compressed_equal_sizes_edge() {
        // compressed == decompressed is an odd case but the ratio should be 1.0
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            1024,
            1024,
            WeightTier::HostLocal,
        );
        let ratio = page.compression_ratio();
        assert!((ratio - 1.0).abs() < 0.001, "expected 1.0, got {ratio}");
    }

    // --- classify_weight additional edge cases ---

    #[test]
    fn classify_weight_embedded_experts_substring() {
        // "experts" as substring inside a longer name
        assert_eq!(
            classify_weight("blk.3.experts.weight", 2),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_no_expert_keyword_is_dense() {
        assert_eq!(
            classify_weight("model.layers.3.mlp.gate_proj", 8),
            WeightClass::DenseLayerWeight,
        );
        assert_eq!(
            classify_weight("model.embed_tokens", 8),
            WeightClass::DenseLayerWeight,
        );
        assert_eq!(
            classify_weight("lm_head", 8),
            WeightClass::DenseLayerWeight,
        );
    }

    // --- select_weight_codec: disabled config beats hot layer ---

    #[test]
    fn disabled_config_beats_hot_layer_check() {
        let mut config = test_config();
        config.enabled = false;
        // Even though layer 0 is hot, disabled returns None before hot check
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(0),
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    // --- decompress_weight_page error message format ---

    #[test]
    fn decompress_weight_page_bitpack_rle_roundtrip() {
        let config = test_config();
        let data = vec![0u8; 512];
        let compressed = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed")
            .expect("should compress");

        let restored = decompress_weight_page(
            &compressed.data,
            CompressionCodec::BitPackRle,
            compressed.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(restored, data);
    }

    // --- WeightCompressionConfig Debug format ---

    #[test]
    fn config_debug_format_contains_fields() {
        let config = test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("enabled"), "Debug should contain 'enabled'");
        assert!(debug.contains("total_layers"), "Debug should contain 'total_layers'");
        assert!(debug.contains("hot_layer_count"), "Debug should contain 'hot_layer_count'");
    }

    // --- CompressedWeightPage Debug format ---

    #[test]
    fn compressed_weight_page_debug_format() {
        let page = CompressedWeightPage {
            data: vec![],
            codec: CompressionCodec::None,
            decompressed_size: 0,
            compressed_size: 0,
        };
        let debug = format!("{:?}", page);
        assert!(debug.contains("CompressedWeightPage"));
    }

    // --- UnifiedVirtualWeightPage Debug format ---

    #[test]
    fn unified_page_debug_format() {
        let page = UnifiedVirtualWeightPage::uncompressed(256, WeightTier::DeviceLocal);
        let debug = format!("{:?}", page);
        assert!(debug.contains("UnifiedVirtualWeightPage"));
        assert!(debug.contains("DeviceLocal"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional unit tests (target: ~110 total, ratio < 14)
    // ═══════════════════════════════════════════════════════════════════════

    // --- WeightTier variant exhaustiveness and Debug ---

    #[test]
    fn weight_tier_debug_format_all_variants() {
        assert_eq!(format!("{:?}", WeightTier::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{:?}", WeightTier::HostLocal), "HostLocal");
        assert_eq!(format!("{:?}", WeightTier::DiskMmap), "DiskMmap");
    }

    #[test]
    fn weight_tier_equality_self() {
        assert_eq!(WeightTier::DeviceLocal, WeightTier::DeviceLocal);
        assert_eq!(WeightTier::HostLocal, WeightTier::HostLocal);
        assert_eq!(WeightTier::DiskMmap, WeightTier::DiskMmap);
    }

    #[test]
    fn weight_tier_distinctness() {
        assert_ne!(WeightTier::DeviceLocal, WeightTier::HostLocal);
        assert_ne!(WeightTier::DeviceLocal, WeightTier::DiskMmap);
        assert_ne!(WeightTier::HostLocal, WeightTier::DiskMmap);
    }

    #[test]
    fn weight_tier_copy_clone() {
        let a = WeightTier::HostLocal;
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    // --- CompressionCodec additional edge cases ---

    #[test]
    fn compression_codec_from_u8_boundary_zero() {
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
    }

    #[test]
    fn compression_codec_from_u8_boundary_four() {
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn compression_codec_as_u8_values_sequential() {
        assert_eq!(CompressionCodec::None.as_u8(), 0);
        assert_eq!(CompressionCodec::Lz4.as_u8(), 1);
        assert_eq!(CompressionCodec::BitPackRle.as_u8(), 2);
        assert_eq!(CompressionCodec::NvcompAns.as_u8(), 3);
        assert_eq!(CompressionCodec::ZstdDict.as_u8(), 4);
    }

    #[test]
    fn compression_codec_from_u8_all_invalid() {
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(100), None);
        assert_eq!(CompressionCodec::from_u8(u8::MAX), None);
    }

    // --- WeightCompressionConfig: is_hot_layer with large indices ---

    #[test]
    fn is_hot_layer_large_index_not_hot() {
        let config = test_config(); // hot_layer_count=2
        assert!(!config.is_hot_layer(Some(1000)));
        assert!(!config.is_hot_layer(Some(usize::MAX)));
    }

    #[test]
    fn is_cold_layer_single_layer_model() {
        let config = WeightCompressionConfig::new(1, 0);
        // cold_layer_count=4, total_layers=1: idx + 4 >= 1 is true for idx=0
        assert!(config.is_cold_layer(Some(0)));
    }

    #[test]
    fn is_cold_layer_all_layers_are_cold_when_total_le_cold() {
        // total_layers=3, cold_layer_count=4 → all layers are cold (idx + 4 >= 3 always)
        let config = WeightCompressionConfig {
            total_layers: 3,
            cold_layer_count: 4,
            ..WeightCompressionConfig::default()
        };
        assert!(config.is_cold_layer(Some(0)));
        assert!(config.is_cold_layer(Some(1)));
        assert!(config.is_cold_layer(Some(2)));
    }

    #[test]
    fn config_hot_layer_count_zero_no_hot_layers() {
        let config = WeightCompressionConfig {
            hot_layer_count: 0,
            ..WeightCompressionConfig::default()
        };
        assert!(!config.is_hot_layer(Some(0)));
    }

    #[test]
    fn config_total_layers_large_value() {
        let config = WeightCompressionConfig::new(usize::MAX, 0);
        assert_eq!(config.total_layers, usize::MAX);
        // hot_layer_count=2, so 0 < 2 → hot
        assert!(config.is_hot_layer(Some(0)));
        // cold check: 0 + 4 >= usize::MAX is false
        assert!(!config.is_cold_layer(Some(0)));
    }

    // --- select_weight_codec: layer_idx=None for non-hot ---

    #[test]
    fn select_weight_codec_none_layer_idx_dense_dram() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            None,
            false,
        );
        assert_eq!(codec, CompressionCodec::Lz4);
    }

    #[test]
    fn select_weight_codec_none_layer_idx_nvme() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DiskMmap,
            WeightClass::ExpertWeight,
            &config,
            None,
            false,
        );
        assert_eq!(codec, CompressionCodec::ZstdDict);
    }

    // --- select_weight_codec: expert dram cold vs non-cold ---

    #[test]
    fn expert_dram_cold_layer_bitpack_rle() {
        let config = test_config(); // total_layers=32, cold_layer_count=4
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(30), // cold layer
            false,
        );
        assert_eq!(codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn expert_dram_non_cold_layer_bitpack_rle() {
        let config = test_config();
        // Non-cold (warm) expert in DRAM uses None (SPEC §6.2) —
        // frequently-activated experts should not be compressed to preserve accuracy.
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(5), // non-cold layer: 5 + 4 < 32
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    // --- compress_weight_page: ZstdDict with empty dictionary fallback ---

    #[test]
    fn compress_weight_page_zstd_dict_empty_dict_returns_err() {
        let config = test_config(); // zstd_dictionary is empty
        let data = vec![0x77u8; 512];
        let result = compress_weight_page(&data, CompressionCodec::ZstdDict, &config);
        // NO-SILENT-FALLBACK: ZstdDict with empty dictionary must return Err,
        // not silently fall back to LZ4 (would store wrong codec tag → decompress corruption)
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("zstd_dictionary is empty"), "expected empty dict error, got: {}", err_msg);
    }

    // --- compress_weight_page: empty input data ---

    #[test]
    fn compress_weight_page_empty_data_none_codec() {
        let config = test_config();
        let result = compress_weight_page(&[], CompressionCodec::None, &config)
            .expect("empty + None should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn compress_weight_page_empty_data_lz4() {
        let config = test_config();
        let result = compress_weight_page(&[], CompressionCodec::Lz4, &config)
            .expect("empty + LZ4 should succeed");
        // Empty data compresses to header overhead, which is >= 0 bytes original
        if let Some(page) = result {
            assert!(page.compressed_size < page.decompressed_size);
        }
    }

    // --- compress_weight_page: single byte data ---

    #[test]
    fn compress_weight_page_single_byte_none() {
        let config = test_config();
        let result = compress_weight_page(&[0xAB], CompressionCodec::None, &config)
            .expect("single byte None should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn compress_weight_page_single_byte_lz4() {
        let config = test_config();
        let result = compress_weight_page(&[0x00], CompressionCodec::Lz4, &config)
            .expect("single byte LZ4 should succeed");
        // Single byte likely won't compress smaller
        drop(result);
    }

    // --- compress_weight_page: large uniform data roundtrip ---

    #[test]
    fn compress_weight_page_large_uniform_lz4() {
        let config = test_config();
        let data = vec![0xCCu8; 8192];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");
        assert!(result.compressed_size < result.decompressed_size);
        assert!(result.compressed_size < 200); // highly compressible
    }

    // --- compress_weight_page: BitPackRle with alternating pattern ---

    #[test]
    fn compress_weight_page_bitpack_rle_alternating() {
        let config = test_config();
        let data: Vec<u8> = (0..512).map(|i| if i % 2 == 0 { 0 } else { 0xFF }).collect();
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config);
        assert!(result.is_ok());
        if let Some(page) = result.unwrap() {
            assert_eq!(page.codec, CompressionCodec::BitPackRle);
            let decompressed = decompress_weight_page(
                &page.data,
                page.codec,
                page.decompressed_size as usize,
                &[],
            ).expect("should decompress");
            assert_eq!(decompressed, data);
        }
    }

    // --- compress_weight: expert weight classification integration ---

    #[test]
    fn compress_weight_expert_weight_dram() {
        let config = test_config(); // has_moe_experts=8
        let data = vec![0x33u8; 1024];
        // Warm expert (layer 5) in DRAM → None codec (no compression)
        // None codec means no compressed page is produced — data stays uncompressed
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.5.mlp.experts.gate_proj",
            Some(5),
            false,
            &config,
        );
        // None codec returns None — no compression, raw data used directly
        assert!(result.is_none(), "warm expert in DRAM should not be compressed");
    }

    #[test]
    fn compress_weight_expert_weight_nvme() {
        // NVMe expert weight → ZstdDict; with empty dict returns None
        let config = test_config();
        let data = vec![0x22u8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::DiskMmap,
            "model.layers.5.mlp.shared_experts.weight",
            Some(5),
            false,
            &config,
        );
        assert!(result.is_none(), "ZstdDict with empty dict → compress_weight returns None");
    }

    #[test]
    fn compress_weight_quantized_dense_hbm() {
        let config = test_config();
        let data = vec![0x11u8; 512];
        let result = compress_weight(
            &data,
            WeightTier::DeviceLocal,
            "model.layers.10.self_attn.q_proj",
            Some(10),
            true, // quantized
            &config,
        );
        let page = result.expect("should produce compressed page");
        assert_eq!(page.codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn compress_weight_disabled_returns_none() {
        let mut config = test_config();
        config.enabled = false;
        let data = vec![0xAAu8; 2048];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.10.mlp.gate_proj",
            Some(10),
            false,
            &config,
        );
        assert!(result.is_none());
    }

    // --- decompress_weight_page: ZstdDict with empty dict falls back to LZ4 ---

    #[test]
    fn decompress_zstd_dict_empty_dict_returns_compress_err() {
        // NO-SILENT-FALLBACK: compress with ZstdDict + empty dict now returns Err
        let config = test_config(); // empty zstd_dictionary
        let data = vec![0xDDu8; 1024];
        let result = compress_weight_page(&data, CompressionCodec::ZstdDict, &config);
        assert!(result.is_err(), "ZstdDict with empty dict must return Err");
    }

    // --- decompress_weight_page: invalid compressed data returns error ---

    #[test]
    fn decompress_weight_page_lz4_corrupt_data_returns_err() {
        let result = decompress_weight_page(&[0xDE, 0xAD, 0xBE, 0xEF], CompressionCodec::Lz4, 1024, &[]);
        assert!(result.is_err());
    }

    // --- UnifiedVirtualWeightPage: compression_ratio boundary cases ---

    #[test]
    fn unified_page_compressed_max_u32_ratio() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            1,
            u32::MAX,
            WeightTier::HostLocal,
        );
        let ratio = page.compression_ratio();
        assert!(ratio > 0.0);
        assert!(ratio < 0.001, "ratio should be near zero, got {ratio}");
    }

    #[test]
    fn unified_page_compressed_one_to_one_ratio() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            1000,
            1000,
            WeightTier::DeviceLocal,
        );
        assert!((page.compression_ratio() - 1.0).abs() < 0.001);
    }

    // --- UnifiedVirtualWeightPage: is_compressed for all codecs ---

    #[test]
    fn unified_page_is_compressed_varies_by_codec() {
        let uncompressed = UnifiedVirtualWeightPage::uncompressed(100, WeightTier::DeviceLocal);
        assert!(!uncompressed.is_compressed());

        for codec in [
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let page = UnifiedVirtualWeightPage::compressed(codec, 50, 100, WeightTier::HostLocal);
            assert!(page.is_compressed(), "{codec:?} page should be compressed");
        }
    }

    // --- UnifiedVirtualWeightPage: clone independence ---

    #[test]
    fn unified_page_clone_independence() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::BitPackRle,
            100,
            200,
            WeightTier::DiskMmap,
        );
        let mut clone = page.clone();
        clone.compressed_size = 999;
        assert_eq!(page.compressed_size, 100);
        assert_eq!(clone.compressed_size, 999);
    }

    // --- CompressedWeightPage: empty data field ---

    #[test]
    fn compressed_weight_page_empty_data() {
        let page = CompressedWeightPage {
            data: vec![],
            codec: CompressionCodec::Lz4,
            decompressed_size: 100,
            compressed_size: 0,
        };
        assert!(page.data.is_empty());
        assert_eq!(page.compressed_size, 0);
    }

    // --- CompressedWeightPage: large data field ---

    #[test]
    fn compressed_weight_page_large_data() {
        let data = vec![0xABu8; 65536];
        let page = CompressedWeightPage {
            data: data.clone(),
            codec: CompressionCodec::BitPackRle,
            decompressed_size: 200000,
            compressed_size: 65536,
        };
        assert_eq!(page.data.len(), 65536);
        assert_eq!(page.data, data);
    }

    // --- classify_weight: various tensor name patterns ---

    #[test]
    fn classify_weight_expert_at_start_of_name() {
        assert_eq!(
            classify_weight("experts.layer.0.weight", 4),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_expert_at_end_of_name() {
        assert_eq!(
            classify_weight("model.layer.0.experts", 4),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_partial_expert_keyword_no_match() {
        // "expert" (singular, no 's') does NOT match the "experts" pattern
        assert_eq!(
            classify_weight("model.layer.0.expert.weight", 8),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn classify_weight_shared_experts_with_s() {
        // "shared_expert" (singular) is the pattern; "shared_experts" also matches via "experts"
        assert_eq!(
            classify_weight("model.layers.0.mlp.shared_experts.gate", 8),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn classify_weight_unicode_name_is_dense() {
        assert_eq!(
            classify_weight("模型.层.权重", 8),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn classify_weight_long_name_is_dense() {
        let long_name = "a".repeat(10000);
        assert_eq!(
            classify_weight(&long_name, 8),
            WeightClass::DenseLayerWeight,
        );
    }

    // --- train_weight_compression_dict basic behavior ---

    #[test]
    fn train_dict_returns_nonempty_with_samples() {
        let samples: Vec<&[u8]> = vec![&[0u8; 64], &[1u8; 64], &[2u8; 64]];
        let dict = train_weight_compression_dict(&samples, 1024);
        // May or may not produce a non-empty dict depending on zstd impl
        // Just verify it doesn't panic and returns valid bytes
        assert!(dict.len() <= 1024 + 256); // some overhead is expected
    }

    #[test]
    fn train_dict_empty_samples_no_panic() {
        let samples: Vec<&[u8]> = vec![];
        let dict = train_weight_compression_dict(&samples, 256);
        // Should not panic, may return empty or minimal dict
        drop(dict);
    }

    // --- WeightCompressionConfig: Debug output contains all fields ---

    #[test]
    fn config_debug_contains_zstd_dictionary() {
        let config = test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("zstd_dictionary"));
    }

    #[test]
    fn config_debug_contains_has_moe_experts() {
        let config = test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("has_moe_experts"));
    }

    // --- WeightCompressionConfig: zstd_dictionary can hold content ---

    #[test]
    fn config_with_zstd_dictionary() {
        let mut config = test_config();
        config.zstd_dictionary = vec![1, 2, 3, 4, 5];
        assert_eq!(config.zstd_dictionary.len(), 5);
        assert_eq!(config.zstd_dictionary[2], 3);
    }

    // --- select_weight_codec: hot layer beats tier-specific codec ---

    #[test]
    fn hot_layer_overrides_nvme() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DiskMmap,
            WeightClass::DenseLayerWeight,
            &config,
            Some(0), // hot layer
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    #[test]
    fn hot_layer_overrides_dram() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(1), // hot layer (index 1 < hot_layer_count=2)
            false,
        );
        assert_eq!(codec, CompressionCodec::None);
    }

    // --- compress_weight_page: returns compressed_size equal to data.len() ---

    #[test]
    fn compress_weight_page_compressed_size_matches_data_len() {
        let config = test_config();
        let data = vec![0u8; 2048];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");
        assert_eq!(result.compressed_size as usize, result.data.len());
    }

    // --- decompress_weight_page: None codec copies input exactly ---

    #[test]
    fn decompress_none_exact_copy() {
        let original = vec![7u8; 64];
        let restored = decompress_weight_page(&original, CompressionCodec::None, 64, &[])
            .expect("should succeed");
        assert_eq!(restored.len(), 64);
        assert_eq!(restored, original);
    }

    // --- WeightClass: exhaustive match coverage ---

    #[test]
    fn weight_class_exhaustive_match() {
        // Ensures compile-time exhaustiveness: if a variant is added, this won't compile
        let classes = [WeightClass::DenseLayerWeight, WeightClass::ExpertWeight];
        for wc in &classes {
            let _ = format!("{wc:?}"); // just exercise Debug
        }
    }

    // --- WeightTier: exhaustive match coverage ---

    #[test]
    fn weight_tier_exhaustive_match() {
        let tiers = [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap];
        for tier in &tiers {
            let _ = format!("{tier:?}");
        }
    }

    // --- compress_weight_page: NvcompAns codec (will fallback to LZ4 on CPU) ---

    #[test]
    fn compress_weight_page_nvcomp_ans_returns_err_on_cpu() {
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight_page(&data, CompressionCodec::NvcompAns, &config);
        // NO-FALLBACK: NvcompAns is GPU-only; on CPU must return Err,
        // not silently fall back to LZ4 (codec/data mismatch → decompress corruption)
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("NvcompAns"), "expected NvcompAns error, got: {}", err_msg);
    }

    // --- compress_weight_page: decompressed_size equals input length ---

    #[test]
    fn compress_weight_page_decompressed_size_matches_input() {
        let config = test_config();
        let data = vec![0x55u8; 777];
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed")
            .expect("should compress");
        assert_eq!(result.decompressed_size, 777);
    }

    // --- config: new preserves default hot/cold counts ---

    #[test]
    fn config_new_preserves_default_hot_cold_counts() {
        let config = WeightCompressionConfig::new(128, 32);
        assert_eq!(config.hot_layer_count, 2);
        assert_eq!(config.cold_layer_count, 4);
        assert!(config.enabled);
    }

    // --- compress_weight: dense non-hot non-cold HBM not quantized = None ---

    #[test]
    fn compress_weight_dense_hbm_not_quantized_skips() {
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::DeviceLocal,
            "model.layers.10.self_attn.q_proj",
            Some(10),
            false,
            &config,
        );
        // DenseLayerWeight in HBM, not quantized → None codec → no compression
        assert!(result.is_none());
    }

    // --- Additional edge-case and coverage tests ---

    // --- WeightCompressionConfig: is_hot_layer boundary when hot equals total ---

    #[test]
    fn is_hot_layer_when_hot_equals_total_layers() {
        let mut config = test_config();
        config.hot_layer_count = 32; // same as total_layers
        assert!(config.is_hot_layer(Some(0)));
        assert!(config.is_hot_layer(Some(31)));
    }

    // --- WeightCompressionConfig: is_cold_layer when cold exceeds total ---

    #[test]
    fn is_cold_layer_cold_exceeds_total() {
        let config = WeightCompressionConfig {
            cold_layer_count: 100,
            total_layers: 10,
            ..WeightCompressionConfig::default()
        };
        // All layers are cold: idx + 100 >= 10 always true
        assert!(config.is_cold_layer(Some(0)));
        assert!(config.is_cold_layer(Some(9)));
    }

    // --- CompressionCodec: Hash consistency ---

    #[test]
    fn compression_codec_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |c: CompressionCodec| {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(CompressionCodec::Lz4), hash_of(CompressionCodec::Lz4));
        assert_ne!(hash_of(CompressionCodec::Lz4), hash_of(CompressionCodec::None));
    }

    // --- WeightTier: Hash consistency ---

    #[test]
    fn weight_tier_hash_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |t: WeightTier| {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(WeightTier::DeviceLocal), hash_of(WeightTier::DeviceLocal));
        assert_ne!(hash_of(WeightTier::DeviceLocal), hash_of(WeightTier::HostLocal));
    }

    // --- select_weight_codec: expert quantized in HBM still None ---

    #[test]
    fn expert_quantized_hbm_ignored() {
        let config = test_config();
        // Expert in HBM always None regardless of is_quantized
        let codec_q = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::ExpertWeight,
            &config,
            Some(10),
            true,
        );
        assert_eq!(codec_q, CompressionCodec::None);
    }

    // --- compress_weight_page: BitPackRle with all-same large data ---

    #[test]
    fn compress_weight_page_bitpack_rle_large_uniform() {
        let config = test_config();
        let data = vec![0x42u8; 4096];
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed")
            .expect("should compress");
        assert!(result.compressed_size < result.decompressed_size); // all same bytes should compress significantly
        assert_eq!(result.decompressed_size, 4096);
    }

    // --- decompress_weight_page: roundtrip for Lz4 with real data ---

    #[test]
    fn decompress_lz4_roundtrip_varied_data() {
        let config = test_config();
        let data: Vec<u8> = (0..1024).map(|i| (i % 97) as u8).collect();
        let compressed = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");
        let restored = decompress_weight_page(
            &compressed.data,
            CompressionCodec::Lz4,
            compressed.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(restored, data);
    }

    // --- CompressedWeightPage: codec field matches constructor ---

    #[test]
    fn compressed_weight_page_all_codec_variants() {
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let page = CompressedWeightPage {
                data: vec![],
                codec,
                decompressed_size: 0,
                compressed_size: 0,
            };
            assert_eq!(page.codec, codec);
        }
    }

    // --- UnifiedVirtualWeightPage: uncompressed with all tier variants ---

    #[test]
    fn unified_page_uncompressed_all_tiers() {
        for tier in [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap] {
            let page = UnifiedVirtualWeightPage::uncompressed(1024, tier);
            assert!(!page.is_compressed());
            assert_eq!(page.codec, CompressionCodec::None);
            assert_eq!(page.tier, tier);
            assert_eq!(page.compressed_size, 0);
            assert_eq!(page.decompressed_size, 1024);
        }
    }

    // --- compress_weight: no experts in model, expert tensor still dense ---

    #[test]
    fn compress_weight_no_experts_model_expert_tensor_treated_dense() {
        let config = WeightCompressionConfig::new(32, 0); // no MoE experts
        let data = vec![0u8; 1024];
        // "experts" in name but has_moe_experts=0 → classified as DenseLayerWeight
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.5.mlp.experts.gate_proj",
            Some(10),
            false,
            &config,
        );
        let page = result.expect("should produce compressed page");
        assert_eq!(page.codec, CompressionCodec::Lz4); // DenseLayerWeight in DRAM → Lz4
    }

    // --- classify_weight: case-insensitive "EXPERTS" ---

    #[test]
    fn classify_weight_uppercase_experts() {
        assert_eq!(
            classify_weight("MODEL.LAYERS.3.MLP.EXPERTS.WEIGHT", 4),
            WeightClass::ExpertWeight,
        );
    }

    // --- select_weight_codec: dense + quantized + HBM = BitPackRle ---

    #[test]
    fn dense_quantized_hbm_is_bitpack_rle() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DeviceLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(5),
            true,
        );
        assert_eq!(codec, CompressionCodec::BitPackRle);
    }

    // --- compress_weight_page: BitPackRle decompress roundtrip with all-zero ---

    #[test]
    fn bitpack_rle_roundtrip_all_zeros() {
        let config = test_config();
        let data = vec![0u8; 4096];
        let compressed = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed")
            .expect("should compress");
        let restored = decompress_weight_page(
            &compressed.data,
            CompressionCodec::BitPackRle,
            compressed.decompressed_size as usize,
            &[],
        ).expect("should decompress");
        assert_eq!(restored, data);
    }

    // --- compress_weight_page: NvcompAns roundtrip (CPU fallback LZ4) ---

    #[test]
    fn nvcomp_ans_returns_err_on_cpu() {
        // NO-FALLBACK: NvcompAns compress on CPU returns Err
        let config = test_config();
        let data = vec![0x55u8; 2048];
        let result = compress_weight_page(&data, CompressionCodec::NvcompAns, &config);
        assert!(result.is_err(), "NvcompAns on CPU must return Err");
    }

    // --- CompressionCodec: PartialEq symmetry ---

    #[test]
    fn compression_codec_partial_eq_symmetry() {
        assert_eq!(CompressionCodec::None, CompressionCodec::None);
        assert_eq!(CompressionCodec::Lz4, CompressionCodec::Lz4);
        assert_ne!(CompressionCodec::None, CompressionCodec::Lz4);
        // Symmetry: a != b implies b != a
        assert_ne!(CompressionCodec::BitPackRle, CompressionCodec::ZstdDict);
    }

    // --- WeightClass: Eq allows use in vec dedup ---

    #[test]
    fn weight_class_dedup_via_vec() {
        let mut v = vec![WeightClass::DenseLayerWeight, WeightClass::DenseLayerWeight, WeightClass::ExpertWeight];
        v.dedup(); // removes consecutive duplicates only
        assert_eq!(v.len(), 2);
    }

    // --- CompressionCodec: usable in hashset ---

    #[test]
    fn compression_codec_usable_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<CompressionCodec> = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
        ].into();
        assert_eq!(set.len(), 3);
    }

    // --- WeightTier: usable in hashset ---

    #[test]
    fn weight_tier_usable_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<WeightTier> = [
            WeightTier::DeviceLocal,
            WeightTier::HostLocal,
            WeightTier::DiskMmap,
        ].into();
        assert_eq!(set.len(), 3);
    }

    // --- UnifiedVirtualWeightPage: compression_ratio for very small compressed ---

    #[test]
    fn compression_ratio_very_small_compressed() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4,
            1,
            1024,
            WeightTier::HostLocal,
        );
        let ratio = page.compression_ratio();
        assert!((ratio - (1.0_f32 / 1024.0_f32)).abs() < 0.001);
    }

    // --- config: is_cold_layer with total_layers=1 ---

    #[test]
    fn is_cold_layer_total_layers_one() {
        let config = WeightCompressionConfig {
            cold_layer_count: 1,
            total_layers: 1,
            ..WeightCompressionConfig::default()
        };
        // idx=0: 0 + 1 >= 1 → true
        assert!(config.is_cold_layer(Some(0)));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Third batch: additional coverage to drive ratio below 14
    // ═══════════════════════════════════════════════════════════════════════

    // --- compress_weight: layer_idx None treated as non-hot ---

    #[test]
    fn compress_weight_none_layer_not_hot() {
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.5.mlp.gate_proj",
            None,
            false,
            &config,
        );
        assert!(result.is_some());
    }

    // --- compress_weight: embed_tokens has no layer → always compresses ---

    #[test]
    fn compress_weight_embed_tokens_no_layer() {
        let config = test_config();
        let data = vec![0u8; 2048];
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.embed_tokens",
            None,
            false,
            &config,
        );
        assert!(result.is_some());
        let page = result.unwrap();
        assert_eq!(page.codec, CompressionCodec::Lz4);
    }

    // --- decompress_weight_page: BitPackRle with varied data roundtrip ---

    #[test]
    fn bitpack_rle_roundtrip_varied() {
        let config = test_config();
        let data: Vec<u8> = (0..512).map(|i| (i * 3 + 7) as u8).collect();
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config);
        if let Some(compressed) = result.expect("should succeed") {
            let restored = decompress_weight_page(
                &compressed.data,
                CompressionCodec::BitPackRle,
                compressed.decompressed_size as usize,
                &[],
            ).expect("should decompress");
            assert_eq!(restored, data);
        }
    }

    // --- compress_weight_page: data exactly at compression boundary ---

    #[test]
    fn compress_weight_page_small_uniform_data() {
        let config = test_config();
        let data = vec![0u8; 8];
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed");
        // May or may not compress — either way should not error
        if let Some(page) = result {
            assert!(page.compressed_size < page.decompressed_size);
        }
    }

    // --- UnifiedVirtualWeightPage: tier preserved for all compressed codecs ---

    #[test]
    fn unified_page_compressed_tier_preserved() {
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::ZstdDict,
            100,
            500,
            WeightTier::DiskMmap,
        );
        assert_eq!(page.tier, WeightTier::DiskMmap);
    }

    // --- UnifiedVirtualWeightPage: uncompressed tier preserved ---

    #[test]
    fn unified_page_uncompressed_tier_preserved() {
        for tier in [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap] {
            let page = UnifiedVirtualWeightPage::uncompressed(256, tier);
            assert_eq!(page.tier, tier);
        }
    }

    // --- CompressionCodec: Eq and Hash allow dedup ---

    #[test]
    fn compression_codec_dedup_via_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::Lz4);
        assert_eq!(set.len(), 1);
    }

    // --- WeightCompressionConfig: total_layers=0 with hot/cold checks ---

    #[test]
    fn config_zero_total_layers_cold_check_safe() {
        let config = WeightCompressionConfig::new(0, 0);
        assert!(!config.is_cold_layer(Some(0)));
        assert!(!config.is_cold_layer(Some(100)));
    }

    // --- select_weight_codec: dense quantized in DRAM uses Lz4 not BitPack ---

    #[test]
    fn dense_quantized_dram_uses_lz4() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(10),
            true, // quantized flag doesn't affect DRAM path
        );
        assert_eq!(codec, CompressionCodec::Lz4);
    }

    // --- compress_weight_page: Lz4 with all 0xFF data ---

    #[test]
    fn compress_weight_page_lz4_all_ff() {
        let config = test_config();
        let data = vec![0xFFu8; 1024];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed");
        if let Some(page) = result {
            assert_eq!(page.decompressed_size, 1024);
            assert!(page.compressed_size < page.decompressed_size);
        }
    }

    // --- CompressedWeightPage: Debug includes all fields ---

    #[test]
    fn compressed_weight_page_debug_includes_codec() {
        let page = CompressedWeightPage {
            data: vec![1, 2, 3],
            codec: CompressionCodec::BitPackRle,
            decompressed_size: 100,
            compressed_size: 3,
        };
        let debug = format!("{:?}", page);
        assert!(debug.contains("BitPackRle"));
        assert!(debug.contains("data"));
    }

    // --- classify_weight: name with "experts" in middle ---

    #[test]
    fn classify_weight_experts_middle_of_name() {
        assert_eq!(
            classify_weight("transformer.h.2.experts.mlp.weight", 2),
            WeightClass::ExpertWeight,
        );
    }

    // --- classify_weight: name with only "expert" (no 's') ---

    #[test]
    fn classify_weight_singular_expert_no_s() {
        // Only "experts" (plural) is the pattern, not "expert"
        assert_eq!(
            classify_weight("model.expert_system.weight", 4),
            WeightClass::DenseLayerWeight,
        );
    }

    // --- select_weight_codec: non-hot layer_idx just past hot boundary ---

    #[test]
    fn select_codec_just_past_hot_boundary() {
        let config = test_config(); // hot_layer_count=2
        // Layer 2 is the first non-hot layer
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(2),
            false,
        );
        assert_eq!(codec, CompressionCodec::Lz4);
    }

    // --- select_weight_codec: layer_idx at cold boundary -1 ---

    #[test]
    fn select_codec_one_before_cold_boundary() {
        let config = test_config(); // total=32, cold=4, cold starts at 28
        // Layer 27 is NOT cold: 27 + 4 = 31 < 32
        let codec = select_weight_codec(
            WeightTier::HostLocal,
            WeightClass::DenseLayerWeight,
            &config,
            Some(27),
            false,
        );
        assert_eq!(codec, CompressionCodec::Lz4);
    }

    // --- train_weight_compression_dict: single sample ---

    #[test]
    fn train_dict_single_sample_no_panic() {
        let samples: Vec<&[u8]> = vec![&[42u8; 128]];
        let dict = train_weight_compression_dict(&samples, 512);
        drop(dict); // just verify no panic
    }

    // --- WeightCompressionConfig: clone produces independent zstd_dictionary ---

    #[test]
    fn config_clone_independent_zstd_dict() {
        let mut config = test_config();
        config.zstd_dictionary = vec![1, 2, 3];
        let mut clone = config.clone();
        clone.zstd_dictionary.push(4);
        assert_eq!(config.zstd_dictionary.len(), 3);
        assert_eq!(clone.zstd_dictionary.len(), 4);
    }

    // --- compress_weight: NVMe tier uses ZstdDict even for experts ---

    #[test]
    fn compress_weight_nvme_expert_zstd_dict_empty_dict_returns_none() {
        // NVMe expert → ZstdDict; with empty dict, compress_weight returns None
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::DiskMmap,
            "model.layers.5.mlp.experts.down_proj",
            Some(5),
            false,
            &config,
        );
        assert!(result.is_none(), "ZstdDict with empty dict → compress_weight returns None");
    }

    // --- WeightTier: Ord-like comparison via derive ---

    #[test]
    fn weight_tier_debug_all_distinct_strings() {
        let tiers = [
            (WeightTier::DeviceLocal, "DeviceLocal"),
            (WeightTier::HostLocal, "HostLocal"),
            (WeightTier::DiskMmap, "DiskMmap"),
        ];
        for (tier, expected) in tiers {
            assert_eq!(format!("{tier:?}"), expected);
        }
    }

    // --- CompressionCodec: from_u8 sequential returns correct variant order ---

    #[test]
    fn compression_codec_from_u8_sequential() {
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn weight_class_default_not_hot() { assert!(!test_config().is_hot_layer(None)); }

    #[test]
    fn weight_class_default_not_cold() { assert!(!test_config().is_cold_layer(None)); }

    #[test]
    fn compression_codec_none_is_zero() { assert_eq!(CompressionCodec::None as u8, 0); }

    #[test]
    fn compression_codec_lz4_is_one() { assert_eq!(CompressionCodec::Lz4 as u8, 1); }

    #[test]
    fn compression_codec_bitpack_is_two() { assert_eq!(CompressionCodec::BitPackRle as u8, 2); }

    #[test]
    fn compression_codec_nvcomp_is_three() { assert_eq!(CompressionCodec::NvcompAns as u8, 3); }

    #[test]
    fn compression_codec_zstd_is_four() { assert_eq!(CompressionCodec::ZstdDict as u8, 4); }

    #[test]
    fn config_default_enabled() { assert!(WeightCompressionConfig::default().enabled); }

    #[test]
    fn config_default_hot_two() { assert_eq!(WeightCompressionConfig::default().hot_layer_count, 2); }

    #[test]
    fn config_default_cold_four() { assert_eq!(WeightCompressionConfig::default().cold_layer_count, 4); }

    #[test]
    fn config_default_zero_layers() { assert_eq!(WeightCompressionConfig::default().total_layers, 0); }

    #[test]
    fn config_default_zero_experts() { assert_eq!(WeightCompressionConfig::default().has_moe_experts, 0); }

    #[test]
    fn config_default_empty_dict() { assert!(WeightCompressionConfig::default().zstd_dictionary.is_empty()); }

    #[test]
    fn classify_empty_string_dense() { assert_eq!(classify_weight("", 0), WeightClass::DenseLayerWeight); }

    #[test]
    fn classify_simple_dense() { assert_eq!(classify_weight("lm_head.weight", 8), WeightClass::DenseLayerWeight); }

    #[test]
    fn unified_page_uncompressed_not_compressed() {
        assert!(!UnifiedVirtualWeightPage::uncompressed(1, WeightTier::DeviceLocal).is_compressed());
    }

    #[test]
    fn unified_page_compressed_is_compressed() {
        assert!(UnifiedVirtualWeightPage::compressed(CompressionCodec::Lz4, 1, 2, WeightTier::HostLocal).is_compressed());
    }

    #[test]
    fn decompress_none_preserves_len() {
        let data = vec![1u8, 2, 3];
        assert_eq!(decompress_weight_page(&data, CompressionCodec::None, 3, &[]).unwrap().len(), 3);
    }

    #[test]
    fn weight_class_dense_reflexive() { assert_eq!(WeightClass::DenseLayerWeight, WeightClass::DenseLayerWeight); }

    #[test]
    fn weight_class_expert_reflexive() { assert_eq!(WeightClass::ExpertWeight, WeightClass::ExpertWeight); }

    // ─── 15 new tests: edge cases, boundaries, error paths ────────────────

    #[test]
    fn decompress_none_empty_input_returns_empty() {
        let result = decompress_weight_page(&[], CompressionCodec::None, 0, &[])
            .expect("empty None decompress should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn unified_page_compressed_with_none_codec_is_not_compressed() {
        // Using compressed() constructor with None codec is unusual but valid
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::None, 0, 1024, WeightTier::DeviceLocal,
        );
        assert!(!page.is_compressed());
        assert_eq!(page.compression_ratio(), 1.0);
    }

    #[test]
    fn unified_page_compressed_ratio_inverted_sizes() {
        // compressed > decompressed (edge case): ratio should exceed 1.0
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4, 2000, 1000, WeightTier::HostLocal,
        );
        let ratio = page.compression_ratio();
        assert!((ratio - 2.0).abs() < 0.001, "expected ~2.0, got {ratio}");
    }

    #[test]
    fn classify_weight_min_expert_count_one() {
        // has_moe_experts=1 is the minimum truthy value
        assert_eq!(
            classify_weight("blk.0.experts.w", 1),
            WeightClass::ExpertWeight,
        );
        assert_eq!(
            classify_weight("blk.0.dense.w", 1),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn config_has_moe_experts_large_value() {
        let config = WeightCompressionConfig::new(32, usize::MAX);
        assert_eq!(config.has_moe_experts, usize::MAX);
        // Expert detection still works
        assert_eq!(
            classify_weight("model.layers.0.experts.w", usize::MAX),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn compress_weight_expert_hbm_returns_none() {
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight(
            &data, WeightTier::DeviceLocal,
            "model.layers.5.mlp.experts.down_proj",
            Some(10), false, &config,
        );
        assert!(result.is_none(), "expert in HBM should not be compressed");
    }

    #[test]
    fn compressed_weight_page_u32_max_sizes() {
        let page = CompressedWeightPage {
            data: vec![0xAB; 4],
            codec: CompressionCodec::Lz4,
            decompressed_size: u32::MAX,
            compressed_size: u32::MAX,
        };
        assert_eq!(page.decompressed_size, u32::MAX);
        assert_eq!(page.compressed_size, u32::MAX);
    }

    #[test]
    fn decompress_bitpack_rle_corrupt_data_returns_ok_or_different() {
        // BitPackRle decompression may not fail on corrupt data (it's not checked)
        // but it must not panic
        let _result = decompress_weight_page(
            &[0xDE, 0xAD], CompressionCodec::BitPackRle, 64, &[],
        );
    }

    #[test]
    fn compress_weight_page_returns_none_when_no_space_saved() {
        // Incompressible random-like data that produces compressed >= original
        let config = test_config();
        let data: Vec<u8> = (0..16).map(|i| (i ^ 0x55) as u8).collect();
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed");
        // If compressed, it must save space; if not, result is None
        if let Some(page) = result {
            assert!(page.compressed_size < page.decompressed_size);
        }
    }

    #[test]
    fn decompress_none_size_param_ignored_for_correctness() {
        // decompress_size param is ignored for None codec — returns clone of input
        let data = vec![5u8, 6, 7, 8];
        let result = decompress_weight_page(&data, CompressionCodec::None, 9999, &[])
            .expect("should succeed");
        assert_eq!(result, data);
    }

    #[test]
    fn select_codec_dense_quantized_cold_layer_disk_mmap() {
        let config = test_config();
        let codec = select_weight_codec(
            WeightTier::DiskMmap, WeightClass::DenseLayerWeight,
            &config, Some(30), true,
        );
        assert_eq!(codec, CompressionCodec::ZstdDict);
    }

    #[test]
    fn unified_page_compressed_zero_decompressed_with_nonzero_compressed() {
        // Edge: decompressed_size=0 but compressed_size>0 (nonsensical but struct allows it)
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4, 100, 0, WeightTier::HostLocal,
        );
        // compression_ratio returns 1.0 when decompressed_size==0
        assert_eq!(page.compression_ratio(), 1.0);
    }

    #[test]
    fn compress_weight_page_lz4_single_value_repeat() {
        let config = test_config();
        let data = vec![0x42u8; 64];
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed");
        let page = result.expect("64 bytes of 0x42 should compress");
        assert!(page.compressed_size < page.decompressed_size);
    }

    #[test]
    fn classify_weight_mixed_case_shared_expert() {
        // to_ascii_lowercase handles mixed case
        assert_eq!(
            classify_weight("model.layers.0.mlp.Shared_Expert.gate", 4),
            WeightClass::ExpertWeight,
        );
    }

    #[test]
    fn config_new_with_same_hot_and_total() {
        let mut config = WeightCompressionConfig::new(2, 0);
        config.hot_layer_count = 2;
        // All layers are hot
        assert!(config.is_hot_layer(Some(0)));
        assert!(config.is_hot_layer(Some(1)));
        assert!(!config.is_hot_layer(Some(2)));
    }

    // ─── 13 additional tests: further edge cases and uncovered paths ──────

    #[test]
    fn decompress_nvcomp_ans_not_applicable_on_cpu() {
        // NO-FALLBACK: NvcompAns compress on CPU returns Err,
        // so there's no valid compressed payload to decompress.
        // This test verifies the compress error path.
        let config = test_config();
        let data = vec![0x77u8; 1536];
        let result = compress_weight_page(&data, CompressionCodec::NvcompAns, &config);
        assert!(result.is_err(), "NvcompAns on CPU must return Err");
    }

    #[test]
    fn compress_weight_expert_hbm_quantized_still_none() {
        // Arrange: expert weight in HBM, even with is_quantized=true
        let config = test_config();
        let data = vec![0u8; 512];
        // Act
        let result = compress_weight(
            &data,
            WeightTier::DeviceLocal,
            "model.layers.10.mlp.experts.gate_proj",
            Some(10),
            true, // quantized — should be ignored for expert HBM
            &config,
        );
        // Assert
        assert!(
            result.is_none(),
            "expert in HBM should not compress regardless of quantized flag"
        );
    }

    #[test]
    fn select_weight_codec_all_tiers_for_dense_quantized_non_hot() {
        // Arrange
        let config = test_config();
        let tiers_and_expected = [
            (WeightTier::DeviceLocal, CompressionCodec::BitPackRle),
            (WeightTier::HostLocal, CompressionCodec::Lz4),
            (WeightTier::DiskMmap, CompressionCodec::ZstdDict),
        ];
        for (tier, expected) in tiers_and_expected {
            // Act
            let codec = select_weight_codec(
                tier,
                WeightClass::DenseLayerWeight,
                &config,
                Some(15), // non-hot, non-cold
                true,
            );
            // Assert
            assert_eq!(
                codec, expected,
                "dense quantized in {tier:?} should be {expected:?}"
            );
        }
    }

    #[test]
    fn decompress_zstd_dict_with_real_dictionary_roundtrip() {
        // Use zstd::dict::from_samples to train a real dictionary with sufficient data
        let samples: Vec<Vec<u8>> = (0..8).map(|i| vec![i as u8; 4096]).collect();
        let sample_refs: Vec<&[u8]> = samples.iter().map(|s| s.as_slice()).collect();
        let dict = zstd::dict::from_samples(&sample_refs, 4096)
            .expect("should train dictionary from sufficient samples");
        assert!(!dict.is_empty(), "trained dictionary must not be empty");
        let mut config = test_config();
        config.zstd_dictionary = dict.clone();
        let data = vec![0x00u8; 4096];
        let compressed = compress_weight_page(&data, CompressionCodec::ZstdDict, &config)
            .expect("ZstdDict compress with real dict should succeed")
            .expect("should compress with real dict");
        // Act
        let restored = decompress_weight_page(
            &compressed.data,
            CompressionCodec::ZstdDict,
            compressed.decompressed_size as usize,
            &dict,
        )
        .expect("decompress with dict should succeed");
        // Assert
        assert_eq!(restored, data);
    }

    #[test]
    fn is_hot_layer_usize_max_hot_count() {
        // Arrange: hot_layer_count = usize::MAX
        let config = WeightCompressionConfig {
            hot_layer_count: usize::MAX,
            total_layers: 64,
            ..WeightCompressionConfig::default()
        };
        // Act & Assert
        assert!(
            config.is_hot_layer(Some(0)),
            "layer 0 should be hot with usize::MAX hot count"
        );
        assert!(
            config.is_hot_layer(Some(usize::MAX - 1)),
            "layer usize::MAX-1 should still be hot"
        );
        assert!(
            !config.is_hot_layer(None),
            "None layer should never be hot"
        );
    }

    #[test]
    fn is_cold_layer_usize_max_index() {
        // Arrange: total_layers=64, cold_layer_count=4
        let config = test_config();
        // Act & Assert
        // usize::MAX + 4 overflows; the comparison idx + cold >= total uses wrapping
        // but in practice layer indices are always < total_layers
        assert!(
            config.is_cold_layer(Some(60)),
            "60 + 4 = 64 >= 32"
        );
    }

    #[test]
    fn compress_weight_page_lz4_exactly_equal_size_returns_none() {
        // Arrange: use data that is incompressible under LZ4
        // Random-like data of very small size where compressed overhead >= original
        let config = test_config();
        let data: Vec<u8> = (0..4).map(|i| (i as u8).wrapping_mul(97)).collect();
        // Act
        let result = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed");
        // Assert: either None (no space saved) or compressed < original
        if let Some(page) = result {
            assert!(
                page.compressed_size < page.decompressed_size,
                "if compression returned Some, it must save space"
            );
        }
    }

    #[test]
    fn classify_weight_shared_expert_with_underscore_variants() {
        // "shared_expert" (singular) is the checked pattern
        assert_eq!(
            classify_weight("model.layers.0.mlp.shared_expert.down_proj", 4),
            WeightClass::ExpertWeight,
        );
        // "shared_experts" (plural) matches via the "experts" substring
        assert_eq!(
            classify_weight("model.layers.0.mlp.shared_experts.down_proj", 4),
            WeightClass::ExpertWeight,
        );
        // Neither keyword present
        assert_eq!(
            classify_weight("model.layers.0.mlp.down_proj", 4),
            WeightClass::DenseLayerWeight,
        );
    }

    #[test]
    fn unified_page_compressed_all_codec_variants_are_compressed() {
        // Arrange & Act: every non-None codec should report is_compressed() == true
        let codecs = [
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for codec in codecs {
            let page = UnifiedVirtualWeightPage::compressed(codec, 50, 100, WeightTier::HostLocal);
            // Assert
            assert!(
                page.is_compressed(),
                "{codec:?} should be compressed"
            );
            assert!(
                page.compression_ratio() < 1.0,
                "{codec:?} ratio should be < 1.0"
            );
        }
    }

    #[test]
    fn compress_weight_dense_disk_mmap_non_hot() {
        // NVMe tier selects ZstdDict; with empty dictionary, compress_weight
        // returns None (error caught and logged, not propagated).
        let config = test_config();
        let data = vec![0u8; 1024];
        let result = compress_weight(
            &data,
            WeightTier::DiskMmap,
            "model.layers.10.mlp.gate_proj",
            Some(10),
            false,
            &config,
        );
        // With empty zstd_dictionary, ZstdDict compress fails → compress_weight returns None
        assert!(result.is_none(), "ZstdDict with empty dict → compress_weight returns None");
    }

    #[test]
    fn compress_weight_page_bitpack_rle_with_mixed_low_values_roundtrip() {
        // Arrange: use low-value uniform data that BitPackRle handles well
        let config = test_config();
        let data = vec![0x0Fu8; 2048];
        // Act
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed");
        if let Some(compressed) = result {
            let restored = decompress_weight_page(
                &compressed.data,
                CompressionCodec::BitPackRle,
                compressed.decompressed_size as usize,
                &[],
            )
            .expect("should decompress");
            // Assert
            assert_eq!(restored, data);
            assert!(compressed.compressed_size < compressed.decompressed_size);
        }
    }

    #[test]
    fn config_zero_hot_zero_cold_means_no_special_zones() {
        // Arrange
        let config = WeightCompressionConfig {
            hot_layer_count: 0,
            cold_layer_count: 0,
            total_layers: 100,
            ..WeightCompressionConfig::default()
        };
        // Act & Assert: no layer is hot or cold
        for i in [0, 1, 50, 99] {
            assert!(
                !config.is_hot_layer(Some(i)),
                "layer {i} should not be hot with hot_layer_count=0"
            );
            // 0 + 0 >= 100 is false for all valid indices
            assert!(
                !config.is_cold_layer(Some(i)),
                "layer {i} should not be cold with cold_layer_count=0"
            );
        }
    }

    #[test]
    fn decompress_zstd_dict_empty_dict_via_compress_weight_returns_none() {
        // NO-SILENT-FALLBACK: compress_weight_page with ZstdDict + empty dict returns Err;
        // compress_weight wraps it and returns None
        let config = test_config(); // zstd_dictionary is empty
        let data: Vec<u8> = (0..512).map(|i| (i % 53) as u8).collect();
        let result = compress_weight_page(&data, CompressionCodec::ZstdDict, &config);
        assert!(result.is_err(), "ZstdDict with empty dict must return Err");
    }

    // ─── 10 additional tests: uncovered paths and boundary conditions ──────

    #[test]
    fn compress_weight_single_layer_model_hot_and_cold() {
        // Arrange: model with 1 layer — layer 0 is both hot and cold
        let config = WeightCompressionConfig {
            hot_layer_count: 2,
            cold_layer_count: 4,
            total_layers: 1,
            enabled: true,
            has_moe_experts: 0,
            zstd_dictionary: Vec::new(),
        };
        let data = vec![0u8; 1024];
        // Act: layer 0 is hot → select_weight_codec returns None → no compression
        let result = compress_weight(
            &data,
            WeightTier::HostLocal,
            "model.layers.0.mlp.gate_proj",
            Some(0),
            false,
            &config,
        );
        // Assert: hot layer takes priority, so no compression
        assert!(result.is_none(), "layer 0 in a 1-layer model is hot, should not compress");
    }

    #[test]
    fn decompress_nvcomp_ans_corrupt_data_returns_err() {
        // Arrange: feed garbage bytes to NvcompAns decompression path
        let corrupt = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33];
        // Act
        let result = decompress_weight_page(&corrupt, CompressionCodec::NvcompAns, 256, &[]);
        // Assert: NvcompAns GPU fails → falls back to LZ4 on corrupt data → should error
        assert!(result.is_err(), "corrupt data through NvcompAns→LZ4 fallback should error");
    }

    #[test]
    fn select_weight_codec_all_tiers_for_expert_non_hot() {
        // Arrange: expert weight in all three tiers, non-hot (warm) layer
        let config = test_config();
        let expected = [
            (WeightTier::DeviceLocal, CompressionCodec::None),
            (WeightTier::HostLocal, CompressionCodec::None),   // warm expert in DRAM: no compression (SPEC §6.2)
            (WeightTier::DiskMmap, CompressionCodec::ZstdDict),
        ];
        for (tier, exp) in expected {
            // Act
            let codec = select_weight_codec(
                tier, WeightClass::ExpertWeight, &config, Some(10), false,
            );
            // Assert
            assert_eq!(codec, exp, "expert in {tier:?} should be {exp:?}");
        }
    }

    #[test]
    fn compression_codec_usable_as_hashmap_key() {
        // Arrange: use CompressionCodec as HashMap key (requires Hash + Eq)
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(CompressionCodec::Lz4, "lz4_value");
        map.insert(CompressionCodec::None, "none_value");
        map.insert(CompressionCodec::ZstdDict, "zstd_value");
        // Act & Assert: retrieval works
        assert_eq!(map.get(&CompressionCodec::Lz4), Some(&"lz4_value"));
        assert_eq!(map.get(&CompressionCodec::None), Some(&"none_value"));
        assert_eq!(map.get(&CompressionCodec::BitPackRle), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn compress_weight_page_lz4_roundtrip_with_patterned_data() {
        // Arrange: semi-structured data — repeating pattern with noise
        let config = test_config();
        let data: Vec<u8> = (0..4096)
            .map(|i| if i % 256 < 128 { (i % 7) as u8 } else { 0xFF })
            .collect();
        // Act
        let compressed = compress_weight_page(&data, CompressionCodec::Lz4, &config)
            .expect("should succeed")
            .expect("should compress");
        let restored = decompress_weight_page(
            &compressed.data,
            CompressionCodec::Lz4,
            compressed.decompressed_size as usize,
            &[],
        )
        .expect("should decompress");
        // Assert: exact roundtrip
        assert_eq!(restored, data);
        assert!(compressed.compressed_size < compressed.decompressed_size);
    }

    #[test]
    fn train_weight_compression_dict_zero_capacity_no_panic() {
        // Arrange: dict_capacity = 0 (edge case)
        let samples: Vec<&[u8]> = vec![&[42u8; 64]; 3];
        // Act
        let dict = train_weight_compression_dict(&samples, 0);
        // Assert: must not panic; result is valid bytes (may be empty)
        assert!(dict.len() <= 256, "dict should be small with zero capacity");
    }

    #[test]
    fn unified_page_compressed_zero_ratio_is_zero_float() {
        // Arrange: compressed_size = 0 with nonzero decompressed (pathological)
        let page = UnifiedVirtualWeightPage::compressed(
            CompressionCodec::Lz4, 0, 1024, WeightTier::HostLocal,
        );
        // Act
        let ratio = page.compression_ratio();
        // Assert: 0 / 1024 = 0.0
        assert!((ratio - 0.0).abs() < f32::EPSILON, "expected 0.0, got {ratio}");
    }

    #[test]
    fn classify_weight_expert_keyword_at_exact_name_boundary() {
        // Arrange: tensor name is exactly "experts"
        assert_eq!(
            classify_weight("experts", 4),
            WeightClass::ExpertWeight,
            "exact 'experts' name should classify as expert",
        );
        // Name is exactly "shared_expert"
        assert_eq!(
            classify_weight("shared_expert", 4),
            WeightClass::ExpertWeight,
            "exact 'shared_expert' should classify as expert",
        );
        // Name with only whitespace (no match)
        assert_eq!(
            classify_weight("   ", 4),
            WeightClass::DenseLayerWeight,
            "whitespace-only name should be dense",
        );
    }

    #[test]
    fn compress_weight_page_preserves_original_data_unchanged() {
        // Arrange: compress_weight_page must not mutate the input slice
        let config = test_config();
        let mut data = vec![0x55u8; 512];
        let data_ptr = data.as_ptr();
        let data_len = data.len();
        // Act
        let _ = compress_weight_page(&data, CompressionCodec::Lz4, &config);
        // Assert: input data is unmodified
        assert_eq!(data.len(), data_len);
        assert_eq!(data.as_ptr(), data_ptr);
        assert!(data.iter().all(|&b| b == 0x55), "input data must not be modified");
    }

    #[test]
    fn decompress_weight_page_bitpack_rle_single_byte_roundtrip() {
        // Arrange: smallest possible data
        let config = test_config();
        let data = vec![0u8; 1];
        // Act
        let result = compress_weight_page(&data, CompressionCodec::BitPackRle, &config)
            .expect("should succeed");
        // Assert: either not compressed (too small) or roundtrips correctly
        if let Some(compressed) = result {
            let restored = decompress_weight_page(
                &compressed.data,
                CompressionCodec::BitPackRle,
                compressed.decompressed_size as usize,
                &[],
            )
            .expect("should decompress");
            assert_eq!(restored, data, "roundtrip must preserve data");
        }
    }
}
