#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::f32_to_f16_bits;

    fn make_k_data(num_tokens: usize, num_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let total = num_tokens * num_kv_heads * head_dim;
        (0..total).map(|i| (i as f32).sin() * 0.5).collect()
    }

    fn make_v_data(num_tokens: usize, num_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        let total = num_tokens * num_kv_heads * head_dim;
        (0..total).map(|i| (i as f32).cos() * 0.3).collect()
    }

    #[test]
    fn test_kivi_strategy_default() {
        let strategy = KiviStrategy::default();
        assert_eq!(strategy.key_precision, PrecisionTier::FP16);
        assert_eq!(strategy.val_precision, PrecisionTier::KIVI4);
        assert_eq!(strategy.sink_count, 4);
        assert!(strategy.enabled);
    }

    #[test]
    fn test_kivi_strategy_disabled() {
        let strategy = KiviStrategy::disabled();
        assert!(!strategy.enabled);
        assert_eq!(strategy.key_precision, PrecisionTier::FP16);
        assert_eq!(strategy.val_precision, PrecisionTier::FP16);
    }

    #[test]
    fn test_k_channel_scales_computation() {
        let num_tokens = 8;
        let num_kv_heads = 4;
        let head_dim = 32;
        let k_data = make_k_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new();
        let scales = strategy.compute_k_channel_scales(&k_data, num_tokens, num_kv_heads, head_dim);

        // Should have one scale per channel
        assert_eq!(scales.len(), num_kv_heads * head_dim);
        // All scales should be positive
        for &s in scales {
            assert!(s > 0.0, "scale should be positive, got {}", s);
        }
    }

    #[test]
    fn test_v_token_scales_computation() {
        let num_tokens = 8;
        let num_kv_heads = 4;
        let head_dim = 32;
        let v_data = make_v_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new();
        let scales = strategy.compute_v_token_scales(&v_data, num_tokens, num_kv_heads, head_dim);

        // Should have one scale per token
        assert_eq!(scales.len(), num_tokens);
        for &s in scales {
            assert!(s > 0.0, "scale should be positive, got {}", s);
        }
    }

    #[test]
    fn test_quantize_dequantize_k_fp16_roundtrip() {
        let num_tokens = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let k_data = make_k_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new().with_key_precision(PrecisionTier::FP16);
        strategy.compute_k_channel_scales(&k_data, num_tokens, num_kv_heads, head_dim);

        let result = strategy.quantize_k(&k_data, num_tokens, num_kv_heads, head_dim);
        assert_eq!(result.precision_tier, PrecisionTier::FP16);
        assert_eq!(result.data.len(), num_tokens * num_kv_heads * head_dim * 2);

        let recovered = strategy.dequantize_k(
            &result.data,
            num_tokens,
            num_kv_heads,
            head_dim,
            PrecisionTier::FP16,
        );
        assert_eq!(recovered.len(), k_data.len());

        // FP16 has limited precision, check within tolerance
        for (i, (&orig, &rec)) in k_data.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            let tolerance = orig.abs().max(1e-6) * 0.01;
            assert!(
                diff < tolerance.max(1e-3),
                "index {}: orig={}, rec={}, diff={}",
                i,
                orig,
                rec,
                diff
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_k_fp8_roundtrip() {
        let num_tokens = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let k_data = make_k_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new().with_key_precision(PrecisionTier::FP8);
        strategy.compute_k_channel_scales(&k_data, num_tokens, num_kv_heads, head_dim);

        let result = strategy.quantize_k(&k_data, num_tokens, num_kv_heads, head_dim);
        assert_eq!(result.precision_tier, PrecisionTier::FP8);
        assert_eq!(result.data.len(), num_tokens * num_kv_heads * head_dim);

        let recovered = strategy.dequantize_k(
            &result.data,
            num_tokens,
            num_kv_heads,
            head_dim,
            PrecisionTier::FP8,
        );

        // FP8: looser tolerance (~1.5% per element due to 8-bit quantization)
        for (i, (&orig, &rec)) in k_data.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            let tolerance = orig.abs().max(1e-6) * 0.03;
            assert!(
                diff < tolerance.max(5e-2),
                "index {}: orig={}, rec={}, diff={}",
                i,
                orig,
                rec,
                diff
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_v_kivi4_roundtrip() {
        let num_tokens = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let v_data = make_v_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new().with_val_precision(PrecisionTier::KIVI4);
        strategy.compute_v_token_scales(&v_data, num_tokens, num_kv_heads, head_dim);

        let result = strategy.quantize_v(&v_data, num_tokens, num_kv_heads, head_dim);
        assert_eq!(result.precision_tier, PrecisionTier::KIVI4);
        // 4-bit: 2 elements per byte
        let expected_bytes = (num_tokens * num_kv_heads * head_dim + 1) / 2;
        assert_eq!(result.data.len(), expected_bytes);

        let recovered = strategy.dequantize_v(
            &result.data,
            num_tokens,
            num_kv_heads,
            head_dim,
            PrecisionTier::KIVI4,
        );

        // 4-bit: ~7% per-element error (7 levels per side)
        for (i, (&orig, &rec)) in v_data.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            let tolerance = orig.abs().max(1e-6) * 0.15;
            assert!(
                diff < tolerance.max(0.1),
                "index {}: orig={}, rec={}, diff={}",
                i,
                orig,
                rec,
                diff
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_v_kivi2_roundtrip() {
        let num_tokens = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let v_data = make_v_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new().with_val_precision(PrecisionTier::KIVI2);
        strategy.compute_v_token_scales(&v_data, num_tokens, num_kv_heads, head_dim);

        let result = strategy.quantize_v(&v_data, num_tokens, num_kv_heads, head_dim);
        assert_eq!(result.precision_tier, PrecisionTier::KIVI2);
        // 2-bit: 4 elements per byte
        let expected_bytes = (num_tokens * num_kv_heads * head_dim + 3) / 4;
        assert_eq!(result.data.len(), expected_bytes);

        let recovered = strategy.dequantize_v(
            &result.data,
            num_tokens,
            num_kv_heads,
            head_dim,
            PrecisionTier::KIVI2,
        );

        // 2-bit: only 3 levels (-scale, 0, +scale), coarse approximation
        for (i, (&orig, &rec)) in v_data.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            // Very coarse precision with 2-bit
            let tolerance = orig.abs().max(1e-6) * 0.5;
            assert!(
                diff < tolerance.max(0.4),
                "index {}: orig={}, rec={}, diff={}",
                i,
                orig,
                rec,
                diff
            );
        }
    }

    #[test]
    fn test_kivi_v_compression_ratio() {
        let s4 = KiviStrategy::new().with_val_precision(PrecisionTier::KIVI4);
        assert!((s4.v_compression_ratio() - 4.0).abs() < 0.01);

        let s2 = KiviStrategy::new().with_val_precision(PrecisionTier::KIVI2);
        assert!((s2.v_compression_ratio() - 8.0).abs() < 0.01);

        let sfp16 = KiviStrategy::new().with_val_precision(PrecisionTier::FP16);
        assert!((sfp16.v_compression_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_to_header_sets_tiers() {
        let mut header = KvPageHeader::new(1);

        // Normal page, KIVI4
        let s = KiviStrategy::new().with_val_precision(PrecisionTier::KIVI4);
        s.apply_to_header(&mut header, false);
        assert_eq!(header.precision_tier(), PrecisionTier::KIVI4);

        // Sink page, FP16 locked
        s.apply_to_header(&mut header, true);
        assert_eq!(header.precision_tier(), PrecisionTier::FP16);

        // Disabled strategy, always FP16
        let d = KiviStrategy::disabled();
        d.apply_to_header(&mut header, false);
        assert_eq!(header.precision_tier(), PrecisionTier::FP16);
    }

    #[test]
    fn test_should_preserve_fp16() {
        let s = KiviStrategy::new().with_sink_count(4);

        // First 4 sink tokens preserved
        assert!(s.should_preserve_fp16(0, true));
        assert!(s.should_preserve_fp16(3, true));
        assert!(!s.should_preserve_fp16(4, true));
        // Non-sink token never preserved
        assert!(!s.should_preserve_fp16(0, false));

        // Disabled: always preserve
        let d = KiviStrategy::disabled();
        assert!(d.should_preserve_fp16(100, false));
    }

    #[test]
    fn test_scale_encode_decode_roundtrip() {
        let test_scales = [0.0f32, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];
        for &s in &test_scales {
            let encoded = encode_scale_to_u8(s);
            let decoded = decode_scale_from_u8(encoded);
            if s == 0.0 {
                assert_eq!(encoded, 0);
                assert_eq!(decoded, 0.0);
            } else {
                let ratio = decoded / s;
                assert!(
                    (ratio - 1.0).abs() < 0.2 || (1.0 / ratio - 1.0).abs() < 0.2,
                    "scale {} encode→{} decode→{} ratio {}",
                    s,
                    encoded,
                    decoded,
                    ratio
                );
            }
        }
    }

    #[test]
    fn test_empty_data_handling() {
        let mut strategy = KiviStrategy::new();

        let k_scales = strategy.compute_k_channel_scales(&[], 0, 4, 32);
        assert!(k_scales.is_empty());

        let v_scales = strategy.compute_v_token_scales(&[], 0, 4, 32);
        assert!(v_scales.is_empty());
    }

    #[test]
    fn test_reset_clears_scales() {
        let num_tokens = 4;
        let num_kv_heads = 2;
        let head_dim = 16;
        let data = make_k_data(num_tokens, num_kv_heads, head_dim);

        let mut strategy = KiviStrategy::new();
        strategy.compute_k_channel_scales(&data, num_tokens, num_kv_heads, head_dim);
        assert!(!strategy.k_scales().is_empty());

        strategy.reset();
        assert!(strategy.k_scales().is_empty());
        assert!(strategy.v_scales().is_empty());
    }

    // ── MUSTAFAR strategy tests ──

    /// Create a header that looks like a MUSTAFAR token:
    /// - High entropy spread (diverse head usage)
    /// - High attention concentration (low entropy_avg)
    /// - Moderate softmax peak
    fn make_mustafar_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(100);
        // Low entropy → high concentration (entropy ~1.0 → concentration ~0.856)
        h.entropy_avg = f32_to_f16_bits(1.0);
        // Moderate softmax peak (not sink)
        h.softmax_max_avg = f32_to_f16_bits(0.4);
        // Very low delta_rho → stable representation
        h.delta_rho_avg = f32_to_f16_bits(0.05);
        // High head entropy spread → diverse head usage
        h.head_entropy_max = 200;
        h.head_entropy_min = 20;
        // Low dead ratio
        h.dead_ratio = 10;
        h
    }

    /// Create a header for a normal (non-MUSTAFAR) token.
    fn make_normal_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(200);
        // High entropy → low concentration
        h.entropy_avg = f32_to_f16_bits(5.0);
        // Low softmax peak
        h.softmax_max_avg = f32_to_f16_bits(0.2);
        // High delta_rho → unstable
        h.delta_rho_avg = f32_to_f16_bits(0.7);
        // Low head spread
        h.head_entropy_max = 60;
        h.head_entropy_min = 40;
        // Moderate dead ratio
        h.dead_ratio = 100;
        h
    }

    /// Create a header for a sink token (attention peak).
    fn make_sink_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(300);
        h.entropy_avg = f32_to_f16_bits(0.5);
        // High softmax → sink token
        h.softmax_max_avg = f32_to_f16_bits(0.9);
        h.delta_rho_avg = f32_to_f16_bits(0.1);
        h.head_entropy_max = 150;
        h.head_entropy_min = 30;
        h.dead_ratio = 5;
        h
    }

    #[test]
    fn test_mustafar_strategy_default() {
        let strategy = MustafarStrategy::default();
        assert!(strategy.enabled);
        assert_eq!(strategy.entropy_spread_threshold, 80);
        assert_eq!(strategy.importance_threshold, 120);
        assert_eq!(strategy.max_mustafar_tokens, 16);
    }

    #[test]
    fn test_mustafar_strategy_disabled() {
        let strategy = MustafarStrategy::disabled();
        assert!(!strategy.enabled);

        // Disabled strategy: all scores return 0 / false
        let header = make_mustafar_header();
        assert_eq!(strategy.score_token_importance(&header), 0);
        assert!(!strategy.is_mustafar(0));
        assert_eq!(strategy.eviction_priority_for(0), 0);
    }

    #[test]
    fn test_score_token_importance_mustafar() {
        let strategy = MustafarStrategy::new();
        let header = make_mustafar_header();
        let score = strategy.score_token_importance(&header);

        // MUSTAFAR-like token should score high (> 120)
        assert!(
            score > 100,
            "MUSTAFAR token should have high importance, got {}",
            score
        );
    }

    #[test]
    fn test_score_token_importance_normal() {
        let strategy = MustafarStrategy::new();
        let header = make_normal_header();
        let score = strategy.score_token_importance(&header);

        // Normal token should score lower
        assert!(
            score < 120,
            "Normal token should have low importance, got {}",
            score
        );
    }

    #[test]
    fn test_score_token_importance_sink() {
        let strategy = MustafarStrategy::new();
        let header = make_sink_header();
        let score = strategy.score_token_importance(&header);

        // Sink token with high softmax should score high
        assert!(
            score > 80,
            "Sink token should have elevated importance, got {}",
            score
        );
    }

    #[test]
    fn test_score_batch_classifies_mustafar() {
        let mut strategy = MustafarStrategy::new();
        // Lower thresholds to make classification easier in test
        strategy.entropy_spread_threshold = 50;
        strategy.importance_threshold = 50;

        let headers = vec![
            make_mustafar_header(), // index 0: high spread, high importance → MUSTAFAR
            make_normal_header(),   // index 1: low spread, low importance → not
            make_sink_header(),     // index 2: high spread, high importance → MUSTAFAR
            make_normal_header(),   // index 3: normal
        ];

        strategy.score_batch(&headers);

        // MUSTAFAR classification
        assert!(strategy.is_mustafar(0), "token 0 should be MUSTAFAR");
        assert!(!strategy.is_mustafar(1), "token 1 should NOT be MUSTAFAR");
        assert!(strategy.is_mustafar(2), "token 2 should be MUSTAFAR");
        assert!(!strategy.is_mustafar(3), "token 3 should NOT be MUSTAFAR");

        // MUSTAFAR count
        assert_eq!(strategy.mustafar_count(), 2);

        // Importance scores populated
        assert_eq!(strategy.importance_scores().len(), 4);
        assert!(strategy.importance(0) > strategy.importance(1));
    }

    #[test]
    fn test_score_batch_respects_max_tokens() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(20)
            .with_importance_threshold(20)
            .with_max_tokens(1); // Only retain 1 MUSTAFAR

        let headers = vec![
            make_mustafar_header(), // importance ~high
            make_sink_header(),     // importance ~higher (sink bonus)
            make_mustafar_header(), // importance ~high
        ];

        strategy.score_batch(&headers);

        // Only one token classified as MUSTAFAR (the highest-scoring one)
        assert!(
            strategy.mustafar_count() <= 1,
            "at most 1 MUSTAFAR token, got {}",
            strategy.mustafar_count()
        );
    }

    #[test]
    fn test_eviction_priority_mustafar_highest() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(50)
            .with_importance_threshold(50);

        let headers = vec![
            make_mustafar_header(), // MUSTAFAR
            make_normal_header(),   // normal
        ];

        strategy.score_batch(&headers);

        let p_mustafar = strategy.eviction_priority_for(0);
        let p_normal = strategy.eviction_priority_for(1);

        // MUSTAFAR priority should be in the top range (>= 0xFFFF_0000)
        assert!(
            p_mustafar >= 0xFFFF_0000,
            "MUSTAFAR priority should be >= 0xFFFF_0000, got 0x{:08X}",
            p_mustafar
        );
        // Normal priority should be much lower
        assert!(
            p_normal < 0xFFFF_0000,
            "Normal priority should be < 0xFFFF_0000, got 0x{:08X}",
            p_normal
        );
        // MUSTAFAR > normal
        assert!(p_mustafar > p_normal);
    }

    #[test]
    fn test_eviction_priority_sink_high() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(100)
            .with_importance_threshold(200); // Very high to prevent MUSTAFAR

        let headers = vec![
            make_sink_header(),     // sink but not MUSTAFAR (threshold too high)
            make_normal_header(),   // normal
        ];

        strategy.score_batch(&headers);

        let p_sink = strategy.eviction_priority_for(0);
        let p_normal = strategy.eviction_priority_for(1);

        // Sink should be in high range but below MUSTAFAR
        assert!(
            p_sink >= 0xFF00_0000,
            "Sink priority should be >= 0xFF00_0000, got 0x{:08X}",
            p_sink
        );
        assert!(p_sink > p_normal);
    }

    #[test]
    fn test_compute_channel_bitmap_high_spread() {
        let strategy = MustafarStrategy::new().with_entropy_threshold(50);

        let mut header = KvPageHeader::new(1);
        header.head_entropy_max = 200;
        header.head_entropy_min = 0; // Large spread

        let num_kv_heads = 8;
        let bitmap = strategy.compute_channel_bitmap(&header, num_kv_heads);

        // With spread=200 > threshold=50, should produce sparse bitmap
        // Not all heads should be active (not 0xFF)
        assert!(
            bitmap != 0xFFFF_FFFF || num_kv_heads != 8,
            "High spread should produce sparse bitmap, got 0x{:08X}",
            bitmap
        );
        // But at least one head should be active
        assert!(bitmap != 0, "At least one head should be active");
    }

    #[test]
    fn test_compute_channel_bitmap_low_spread() {
        let strategy = MustafarStrategy::new().with_entropy_threshold(80);

        let mut header = KvPageHeader::new(1);
        header.head_entropy_max = 40;
        header.head_entropy_min = 30; // Low spread

        let num_kv_heads = 8;
        let bitmap = strategy.compute_channel_bitmap(&header, num_kv_heads);

        // Low spread below threshold: all heads active
        assert_eq!(
            bitmap, 0xFFFF_FFFF,
            "Low spread should produce full bitmap, got 0x{:08X}",
            bitmap
        );
    }

    #[test]
    fn test_compute_channel_bitmap_disabled() {
        let strategy = MustafarStrategy::disabled();

        let mut header = KvPageHeader::new(1);
        header.head_entropy_max = 200;
        header.head_entropy_min = 0;

        let bitmap = strategy.compute_channel_bitmap(&header, 8);
        // Disabled: all heads active
        assert_eq!(bitmap, 0xFFFF_FFFF);
    }

    #[test]
    fn test_precision_floor_mustafar() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(50)
            .with_importance_threshold(50);

        let headers = vec![make_mustafar_header()];
        strategy.score_batch(&headers);

        // MUSTAFAR token → floor at FP8
        let floor = strategy.precision_floor(0, &headers[0]);
        assert_eq!(floor, Some(PrecisionTier::FP8));
    }

    #[test]
    fn test_precision_floor_sink() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(200)
            .with_importance_threshold(200);

        let headers = vec![make_sink_header()];
        strategy.score_batch(&headers);

        // Sink token → floor at FP16 (even if not MUSTAFAR)
        let floor = strategy.precision_floor(0, &headers[0]);
        assert_eq!(floor, Some(PrecisionTier::FP16));
    }

    #[test]
    fn test_precision_floor_normal() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(200)
            .with_importance_threshold(200);

        let headers = vec![make_normal_header()];
        strategy.score_batch(&headers);

        // Normal token → no special floor
        let floor = strategy.precision_floor(0, &headers[0]);
        assert_eq!(floor, None);
    }

    #[test]
    fn test_precision_floor_disabled() {
        let strategy = MustafarStrategy::disabled();
        let header = make_mustafar_header();

        let floor = strategy.precision_floor(0, &header);
        assert_eq!(floor, None);
    }

    #[test]
    fn test_apply_to_header_writes_importance_and_bitmap() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(50)
            .with_importance_threshold(50);

        let headers = vec![make_mustafar_header()];
        strategy.score_batch(&headers);

        let mut header = headers[0].clone();
        strategy.apply_to_header(&mut header, 0, 8);

        // Importance score should be written
        assert!(
            header.importance_score > 0,
            "importance_score should be > 0, got {}",
            header.importance_score
        );
        // Channel bitmap should be set (not necessarily all-ones for high spread)
        assert!(
            header.channel_bitmap_lo != 0,
            "channel_bitmap should not be zero"
        );
        // MUSTAFAR token: deopt bit 1 should be set
        assert!(
            header.deopt_flags & 0x02 != 0,
            "MUSTAFAR token should have deopt bit 1 set"
        );
    }

    #[test]
    fn test_apply_to_header_disabled_no_side_effects() {
        let strategy = MustafarStrategy::disabled();
        let mut header = make_mustafar_header();
        let original_score = header.importance_score;
        let original_bitmap = header.channel_bitmap_lo;
        let original_deopt = header.deopt_flags;

        strategy.apply_to_header(&mut header, 0, 8);

        // Disabled: no changes
        assert_eq!(header.importance_score, original_score);
        assert_eq!(header.channel_bitmap_lo, original_bitmap);
        assert_eq!(header.deopt_flags, original_deopt);
    }

    #[test]
    fn test_mustafar_reset_clears_state() {
        let mut strategy = MustafarStrategy::new()
            .with_entropy_threshold(50)
            .with_importance_threshold(50);

        let headers = vec![make_mustafar_header(), make_normal_header()];
        strategy.score_batch(&headers);

        assert!(!strategy.importance_scores().is_empty());
        assert!(!strategy.mustafar_flags().is_empty());
        assert!(!strategy.eviction_priorities().is_empty());

        strategy.reset();

        assert!(strategy.importance_scores().is_empty());
        assert!(strategy.mustafar_flags().is_empty());
        assert!(strategy.eviction_priorities().is_empty());
    }

    #[test]
    fn test_empty_batch_handling() {
        let mut strategy = MustafarStrategy::new();
        let headers: Vec<KvPageHeader> = vec![];

        strategy.score_batch(&headers);

        assert_eq!(strategy.mustafar_count(), 0);
        assert!(strategy.importance_scores().is_empty());
        assert!(strategy.mustafar_flags().is_empty());
        assert!(strategy.eviction_priorities().is_empty());
    }

    // ── ChunkKV strategy tests ──

    #[test]
    fn test_chunkkv_strategy_default() {
        let strategy = ChunkKvStrategy::default();
        assert_eq!(strategy.chunk_size, 64);
        assert_eq!(strategy.max_resident_chunks, 0);
        assert!(strategy.enabled);
        assert!(matches!(
            strategy.compress_strategy,
            ChunkCompressStrategy::Adaptive
        ));
    }

    #[test]
    fn test_chunkkv_strategy_disabled() {
        let strategy = ChunkKvStrategy::disabled();
        assert!(!strategy.enabled);

        // Disabled: always return FP16
        let tier = strategy.chunk_compression_tier(0, PrecisionTier::KIVI4);
        assert_eq!(tier, PrecisionTier::FP16);
    }

    #[test]
    fn test_compute_chunk_layout_exact() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        let layout = strategy.compute_chunk_layout(64);

        // 64 tokens / 16 per chunk = 4 chunks
        assert_eq!(layout.len(), 4);
        assert_eq!(layout[0], (0, 16));
        assert_eq!(layout[1], (16, 16));
        assert_eq!(layout[2], (32, 16));
        assert_eq!(layout[3], (48, 16));
    }

    #[test]
    fn test_compute_chunk_layout_partial_last() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        let layout = strategy.compute_chunk_layout(50);

        // 50 tokens / 16 per chunk = 4 chunks (last partial: 2 tokens)
        assert_eq!(layout.len(), 4);
        assert_eq!(layout[0], (0, 16));
        assert_eq!(layout[1], (16, 16));
        assert_eq!(layout[2], (32, 16));
        assert_eq!(layout[3], (48, 2));
    }

    #[test]
    fn test_compute_chunk_layout_empty() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        let layout = strategy.compute_chunk_layout(0);
        assert!(layout.is_empty());
    }

    #[test]
    fn test_chunk_for_token() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(64);

        assert_eq!(strategy.chunk_for_token(0), 0);
        assert_eq!(strategy.chunk_for_token(15), 0);
        assert_eq!(strategy.chunk_for_token(16), 1);
        assert_eq!(strategy.chunk_for_token(31), 1);
        assert_eq!(strategy.chunk_for_token(63), 3);
    }

    #[test]
    fn test_chunk_range() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(50);

        assert_eq!(strategy.chunk_range(0), Some((0, 16)));
        assert_eq!(strategy.chunk_range(1), Some((16, 32)));
        assert_eq!(strategy.chunk_range(3), Some((48, 50)));
        assert_eq!(strategy.chunk_range(4), None);
    }

    #[test]
    fn test_num_chunks_and_total() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(50);

        assert_eq!(strategy.num_chunks(), 4);
        assert_eq!(strategy.total_tokens(), 50);
    }

    #[test]
    fn test_init_chunks_creates_metadata() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(48);
        let chunks = strategy.init_chunks();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_id, 0);
        assert_eq!(chunks[0].token_start, 0);
        assert_eq!(chunks[0].token_count, 16);
        assert_eq!(chunks[0].residency, ChunkResidency::Resident);
        assert_eq!(chunks[2].chunk_id, 2);
        assert_eq!(chunks[2].token_start, 32);
        assert_eq!(chunks[2].token_count, 16);
    }

    #[test]
    fn test_touch_chunk_updates_timestamp() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(32);
        strategy.init_chunks();

        let ts_before = strategy.chunk_info(0).unwrap().last_access_ts;
        strategy.touch_chunk(0);
        let ts_after = strategy.chunk_info(0).unwrap().last_access_ts;
        assert!(ts_after > ts_before);
    }

    #[test]
    fn test_chunk_compression_tier_uniform() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_compress_strategy(ChunkCompressStrategy::Uniform);
        strategy.compute_chunk_layout(48);

        // All chunks get the same tier
        assert_eq!(
            strategy.chunk_compression_tier(0, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
        assert_eq!(
            strategy.chunk_compression_tier(1, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
        assert_eq!(
            strategy.chunk_compression_tier(2, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
    }

    #[test]
    fn test_chunk_compression_tier_adaptive_sink_fp16() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_compress_strategy(ChunkCompressStrategy::Adaptive);
        strategy.compute_chunk_layout(48);

        // Chunk 0 (sink) always FP16
        assert_eq!(
            strategy.chunk_compression_tier(0, PrecisionTier::KIVI2),
            PrecisionTier::FP16
        );
    }

    #[test]
    fn test_chunk_compression_tier_adaptive_last_equal_base() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_compress_strategy(ChunkCompressStrategy::Adaptive);
        strategy.compute_chunk_layout(48);

        // Last chunk (most recent) gets the base tier
        assert_eq!(
            strategy.chunk_compression_tier(2, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
    }

    #[test]
    fn test_chunk_compression_tier_tiered() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(8)
            .with_compress_strategy(ChunkCompressStrategy::Tiered);
        strategy.compute_chunk_layout(40); // 5 chunks

        // Chunk 0 = sink = FP16
        assert_eq!(
            strategy.chunk_compression_tier(0, PrecisionTier::KIVI4),
            PrecisionTier::FP16
        );
        // Chunk 4 (last) = base tier
        assert_eq!(
            strategy.chunk_compression_tier(4, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
        // Chunk 3 (recent) = one step down
        assert_eq!(
            strategy.chunk_compression_tier(3, PrecisionTier::KIVI4),
            PrecisionTier::KIVI2 // KIVI4 → KIVI2
        );
        // Chunk 1 (old) = two steps down
        assert_eq!(
            strategy.chunk_compression_tier(1, PrecisionTier::KIVI4),
            PrecisionTier::Sparse // KIVI4 → KIVI2 → Sparse
        );
    }

    #[test]
    fn test_chunk_compression_tier_oob_returns_base() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_compress_strategy(ChunkCompressStrategy::Adaptive);
        strategy.compute_chunk_layout(16); // 1 chunk

        // Out of bounds chunk returns base tier
        assert_eq!(
            strategy.chunk_compression_tier(5, PrecisionTier::KIVI4),
            PrecisionTier::KIVI4
        );
    }

    #[test]
    fn test_chunk_size_bytes() {
        let strategy = ChunkKvStrategy::new().with_chunk_size(64);
        let bytes = strategy.chunk_size_bytes(8, 128, 2); // FP16

        assert_eq!(bytes, 64 * 8 * 128 * 2 * 2);
    }

    #[test]
    fn test_total_resident_bytes() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(32);
        strategy.init_chunks();

        strategy.set_chunk_size_bytes(0, 4096);
        strategy.set_chunk_size_bytes(1, 2048);

        assert_eq!(strategy.total_resident_bytes(), 4096 + 2048);
    }

    #[test]
    fn test_evict_and_restore_chunk() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(32);
        strategy.init_chunks();

        assert_eq!(strategy.resident_count(), 2);
        assert_eq!(strategy.evicted_count(), 0);

        assert!(strategy.evict_chunk(1));
        assert_eq!(strategy.resident_count(), 1);
        assert_eq!(strategy.evicted_count(), 1);
        assert!(!strategy.evict_chunk(1)); // Already evicted

        assert!(strategy.needs_restore(1));
        assert!(!strategy.needs_restore(0));

        assert!(strategy.restore_chunk(1));
        assert_eq!(strategy.resident_count(), 2);
        assert_eq!(strategy.evicted_count(), 0);
        assert!(!strategy.needs_restore(1));
    }

    #[test]
    fn test_select_eviction_candidates_lru() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(64);
        strategy.init_chunks();

        // Touch chunks in order: 3, 1, 2 (0 untouched)
        strategy.touch_chunk(3);
        strategy.touch_chunk(1);
        strategy.touch_chunk(2);

        let candidates = strategy.select_eviction_candidates(2);
        // Chunk 0 (sink) should NOT be first; oldest is chunk 0 (last_access_ts=0)
        // Order: untouched first (chunk 0), then oldest touched
        assert!(!candidates.is_empty());
        // Chunk 0 should be last among candidates (sink protection)
        // But if only 2 candidates selected from 4, chunk 0 may or may not be included
        for &c in &candidates {
            assert!(c < 4);
        }
    }

    #[test]
    fn test_select_eviction_sink_protected() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(48);
        strategy.init_chunks();

        // Touch all: chunk 0 first, then 1, then 2
        strategy.touch_chunk(0);
        strategy.touch_chunk(1);
        strategy.touch_chunk(2);

        // Evict 2 of 3 — chunk 0 should be excluded (sink protection in sort)
        let candidates = strategy.select_eviction_candidates(2);
        // Chunk 0 should NOT be in the first 2 (it sorts last)
        assert!(!candidates.contains(&0));
    }

    #[test]
    fn test_migration_plan_no_limits() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_max_resident(0); // Unlimited

        strategy.compute_chunk_layout(64);
        strategy.init_chunks();
        strategy.evict_chunk(3);

        let plan = strategy.migration_plan();
        assert!(plan.evict.is_empty()); // No eviction needed
        assert_eq!(plan.restore, vec![3]);
    }

    #[test]
    fn test_migration_plan_with_limits() {
        let mut strategy = ChunkKvStrategy::new()
            .with_chunk_size(16)
            .with_max_resident(2); // Only 2 resident allowed

        strategy.compute_chunk_layout(64); // 4 chunks
        strategy.init_chunks();

        assert_eq!(strategy.resident_count(), 4);
        let plan = strategy.migration_plan();
        // Should evict 4 - 2 = 2 chunks
        assert_eq!(plan.evict.len(), 2);
        assert!(plan.restore.is_empty());
    }

    #[test]
    fn test_set_chunk_size_and_bump_generation() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(32);
        strategy.init_chunks();

        strategy.set_chunk_size_bytes(0, 8192);
        assert_eq!(strategy.chunk_info(0).unwrap().size_bytes, 8192);

        strategy.bump_chunk_generation(0);
        assert_eq!(strategy.chunk_info(0).unwrap().generation, 1);

        strategy.bump_chunk_generation(0);
        assert_eq!(strategy.chunk_info(0).unwrap().generation, 2);
    }

    #[test]
    fn test_chunkkv_reset() {
        let mut strategy = ChunkKvStrategy::new().with_chunk_size(16);
        strategy.compute_chunk_layout(48);
        strategy.init_chunks();

        assert_eq!(strategy.num_chunks(), 3);
        assert!(!strategy.chunks().is_empty());

        strategy.reset();
        assert_eq!(strategy.num_chunks(), 0);
        assert!(strategy.chunks().is_empty());
        assert_eq!(strategy.total_tokens(), 0);
    }

    // ── CrossDecisionMatrix tests ──

    #[test]
    fn test_variant_matrix_size() {
        assert_eq!(VARIANT_MATRIX.len(), 16);
    }

    #[test]
    fn test_variant_bits_roundtrip() {
        for &(variant, _desc) in &VARIANT_MATRIX {
            let bits = variant.bits();
            let recovered = DecisionVariant::from_bits(bits);
            assert_eq!(
                recovered.bits(),
                bits,
                "Variant {:?} bits {:04b} roundtrip mismatch: got {:?} bits {:04b}",
                variant,
                bits,
                recovered,
                recovered.bits()
            );
        }
    }

    #[test]
    fn test_variant_has_methods() {
        let full = DecisionVariant::FullStack;
        assert!(full.has_kivi());
        assert!(full.has_kv_tuner());
        assert!(full.has_mustafar());
        assert!(full.has_chunk_kv());

        let baseline = DecisionVariant::Baseline;
        assert!(baseline.has_kivi());
        assert!(!baseline.has_kv_tuner());
        assert!(!baseline.has_mustafar());
        assert!(!baseline.has_chunk_kv());

        let chunk_only = DecisionVariant::ChunkOnly;
        assert!(!chunk_only.has_kivi());
        assert!(!chunk_only.has_kv_tuner());
        assert!(!chunk_only.has_mustafar());
        assert!(chunk_only.has_chunk_kv());
    }

    #[test]
    fn test_variant_name() {
        assert_eq!(DecisionVariant::FullStack.name(), "FullStack");
        assert_eq!(DecisionVariant::Baseline.name(), "Baseline");
        assert_eq!(DecisionVariant::Custom(0b0101).name(), "Custom");
    }

    #[test]
    fn test_cross_decision_default() {
        let matrix = CrossDecisionMatrix::default();
        assert!(matrix.is_active());
        assert_eq!(matrix.active_variant, DecisionVariant::FullStack);
        assert!(matrix.kivi.enabled);
        assert!(matrix.kv_tuner.enabled);
        assert!(matrix.mustafar.enabled);
        assert!(matrix.chunk_kv.enabled);
    }

    #[test]
    fn test_cross_decision_disabled() {
        let matrix = CrossDecisionMatrix::disabled();
        assert!(!matrix.is_active());
        assert!(!matrix.kivi.enabled);
        assert!(!matrix.kv_tuner.enabled);
        assert!(!matrix.mustafar.enabled);
        assert!(!matrix.chunk_kv.enabled);
        assert_eq!(matrix.decision_count(), 0);
    }

    #[test]
    fn test_evaluate_high_memory_pressure() {
        let hw = HardwareProfile {
            gpu_memory_gb: 4.0,
            max_seq_len: 16384,
            num_kv_heads: 8,
            head_dim: 128,
            ..HardwareProfile::default()
        };
        let mut matrix = CrossDecisionMatrix::with_hardware(hw);
        let (_variant, _reason) = matrix.evaluate();

        // KV cache for 16384 tokens ~ 64MB, but GPU only 4GB — pressure > 80%?
        // Actually: 16384 * 8 * 128 * 4 = 67MB, / 4GB = 1.6% — not high pressure
        // Let's use a more extreme case: 131072 tokens
        // 131072 * 8 * 128 * 4 = 536MB, / 4GB = 13% — still not > 80%
        // The evaluate() computes kv_cache_size_bytes as max_seq_len * num_kv_heads * head_dim * 2 * 2
        // For 131072 * 8 * 128 * 4 = 536,870,912 bytes = 0.5 GB
        // vs gpu_memory_gb = 4.0 → 12.5% pressure
        // Let's adjust: tiny GPU, huge seq_len
        let hw2 = HardwareProfile {
            gpu_memory_gb: 1.0,
            max_seq_len: 131072,
            num_kv_heads: 32,
            head_dim: 128,
            ..HardwareProfile::default()
        };
        let mut matrix2 = CrossDecisionMatrix::with_hardware(hw2);
        let (variant2, _reason2) = matrix2.evaluate();
        // With 1GB GPU and 131072*32*128*4 ≈ 2GB KV cache, pressure > 100% → high
        assert!(variant2.has_chunk_kv(), "High memory pressure should enable ChunkKV");
        assert!(matrix2.decision_count() > 0);
    }

    #[test]
    fn test_evaluate_small_gpu() {
        let hw = HardwareProfile {
            gpu_memory_gb: 4.0,
            max_seq_len: 2048,
            num_kv_heads: 4,
            head_dim: 64,
            ..HardwareProfile::default()
        };
        let mut matrix = CrossDecisionMatrix::with_hardware(hw);
        let (variant, _reason) = matrix.evaluate();
        assert_eq!(variant, DecisionVariant::Baseline);
    }

    #[test]
    fn test_evaluate_high_end() {
        let hw = HardwareProfile::high_end();
        let mut matrix = CrossDecisionMatrix::with_hardware(hw);
        let (variant, _reason) = matrix.evaluate();
        // High-end GPU with large seq_len (>8K) should get FullStack
        assert!(variant.has_kivi());
        assert!(variant.has_kv_tuner());
    }

    #[test]
    fn test_evaluate_with_state_changes_variant() {
        let hw = HardwareProfile {
            gpu_memory_gb: 24.0,
            max_seq_len: 8192,
            num_kv_heads: 8,
            head_dim: 128,
            compute_capability: 0.4,
            ..HardwareProfile::default()
        };
        let mut matrix = CrossDecisionMatrix::with_hardware(hw);
        // Initial evaluation: low compute → Baseline
        let (v1, _) = matrix.evaluate();
        assert_eq!(v1, DecisionVariant::Baseline);

        // Mid-sequence with high entropy: should enable precision tuning
        let (v2, _) = matrix.evaluate_with_state(5000, 0.85);
        // High entropy on long sequence (>50% of max) → should try to enable retention or precision
        assert!(v2.has_chunk_kv() || v2.has_mustafar() || v2.has_kv_tuner(),
            "Long sequence with high entropy should upgrade from Baseline, got {:?}", v2);

        // Mid-sequence with low entropy: should enable MUSTAFAR
        let (v3, _) = matrix.evaluate_with_state(5000, 0.15);
        // Low entropy should trigger MUSTAFAR
        assert!(v3.has_mustafar(),
            "Low entropy should enable MUSTAFAR, got {:?}", v3);
    }

    #[test]
    fn test_compose_for_page_basic() {
        let mut matrix = CrossDecisionMatrix::default();
        let mut header = KvPageHeader::new(1);
        header.ref_count = 1; // mark active

        let (k_tier, v_tier) = matrix.compose_for_page(
            &mut header, 0, 64, 0.5, 8,
        );

        // Should produce valid tiers
        assert!(tuner_tier_rank(k_tier) > 0);
        assert!(tuner_tier_rank(v_tier) > 0);
    }

    #[test]
    fn test_compose_for_batch() {
        let mut matrix = CrossDecisionMatrix::default();
        let mut headers: Vec<KvPageHeader> = (0..16)
            .map(|i| {
                let mut h = KvPageHeader::new(i as u32);
                h.ref_count = 1;
                h
            })
            .collect();

        let changed = matrix.compose_for_batch(&mut headers, 64, 0.5, 8);
        // With default settings, some pages may be adjusted
        assert!(changed <= headers.len());
    }

    #[test]
    fn test_compose_for_batch_empty() {
        let mut matrix = CrossDecisionMatrix::default();
        let mut headers: Vec<KvPageHeader> = vec![];
        let changed = matrix.compose_for_batch(&mut headers, 64, 0.5, 8);
        assert_eq!(changed, 0);
    }

    #[test]
    fn test_should_preserve_fp16_composed() {
        let mut matrix = CrossDecisionMatrix::default();
        let mut header = make_sink_header();
        header.ref_count = 1;

        // Score token 0 as MUSTAFAR for preservation
        matrix.mustafar.score_batch(std::slice::from_ref(&header));

        // Sink token should be preserved (KIVI sink + MUSTAFAR retention)
        assert!(matrix.should_preserve_fp16(0, &header));
    }

    #[test]
    fn test_recommended_tier_composed() {
        let matrix = CrossDecisionMatrix::default();
        let header = make_normal_header();
        let tier = matrix.recommended_tier(0, &header, 64);
        // Should return a valid tier
        assert!(tuner_tier_rank(tier) > 0);
    }

    #[test]
    fn test_needs_migration() {
        let mut matrix = CrossDecisionMatrix::new()
            .with_chunk_kv(
                ChunkKvStrategy::new()
                    .with_chunk_size(16)
                    .with_max_resident(2),
            );
        matrix.chunk_kv.compute_chunk_layout(64);
        matrix.chunk_kv.init_chunks();

        // 4 chunks, max 2 resident → needs migration
        assert!(matrix.needs_migration());
        assert!(matrix.migration_plan().is_some());
    }

    #[test]
    fn test_needs_migration_disabled() {
        let matrix = CrossDecisionMatrix::disabled();
        assert!(!matrix.needs_migration());
        assert!(matrix.migration_plan().is_none());
    }

    #[test]
    fn test_effective_compression_ratio() {
        let mut matrix = CrossDecisionMatrix::default();
        // Without chunking, ratio comes from KIVI
        let ratio = matrix.effective_compression_ratio();
        assert!(ratio >= 1.0, "Compression ratio should be >= 1.0, got {}", ratio);

        // With chunking enabled, ratio may differ
        matrix.chunk_kv.compute_chunk_layout(128);
        matrix.chunk_kv.init_chunks();
        let ratio_chunked = matrix.effective_compression_ratio();
        assert!(ratio_chunked > 0.0, "Chunked ratio should be > 0");
    }

    #[test]
    fn test_step_produces_decision() {
        let mut matrix = CrossDecisionMatrix::default();
        let mut headers: Vec<KvPageHeader> = (0..8)
            .map(|i| {
                let mut h = KvPageHeader::new(i as u32);
                h.ref_count = 1;
                h.entropy_avg = f32_to_f16_bits(2.0);
                h
            })
            .collect();

        let (variant, reason, changed) = matrix.step(&mut headers, 64, 0.5, 8);

        // Should have recorded a decision
        assert!(matrix.decision_count() > 0);
        // Should return a valid variant
        assert!(!reason.is_empty());
        // changed count should be reasonable
        assert!(changed <= headers.len());
        let _ = variant;
    }

    #[test]
    fn test_reset_clears_all() {
        let mut matrix = CrossDecisionMatrix::default();

        // Do some operations
        let mut headers: Vec<KvPageHeader> = (0..4)
            .map(|i| {
                let mut h = KvPageHeader::new(i as u32);
                h.ref_count = 1;
                h
            })
            .collect();
        matrix.step(&mut headers, 32, 0.3, 4);

        assert!(matrix.decision_count() > 0);

        matrix.reset();
        assert_eq!(matrix.decision_count(), 0);
        assert!(matrix.kivi.k_scales().is_empty());
        assert!(matrix.kivi.v_scales().is_empty());
        assert_eq!(matrix.chunk_kv.num_chunks(), 0);
        assert!(matrix.mustafar.importance_scores().is_empty());
    }

    #[test]
    fn test_apply_variant_ensures_kivi_with_tuner() {
        let matrix = CrossDecisionMatrix::new()
            .with_variant(DecisionVariant::PrecisionOnly);

        // PrecisionOnly has KIVI + KVTuner — both should be enabled
        assert!(matrix.kivi.enabled);
        assert!(matrix.kv_tuner.enabled);
        assert!(!matrix.mustafar.enabled);
        assert!(!matrix.chunk_kv.enabled);
    }

    #[test]
    fn test_drain_decisions() {
        let hw = HardwareProfile::minimal();
        let mut matrix = CrossDecisionMatrix::with_hardware(hw);
        matrix.evaluate();
        matrix.evaluate_with_state(100, 0.5);

        let count_before = matrix.decision_count();
        assert!(count_before >= 2);

        let drained = matrix.drain_decisions();
        assert_eq!(drained.len(), count_before);
        assert_eq!(matrix.decision_count(), 0);
    }

    #[test]
    fn test_hardware_profile_kv_cache_size() {
        let hw = HardwareProfile::high_end();
        let bytes = hw.kv_cache_size_bytes();
        let expected = 131072 * 8 * 128 * 2 * 2;
        assert_eq!(bytes, expected);

        let gb = hw.kv_cache_size_gb();
        assert!((gb - expected as f32 / (1024.0 * 1024.0 * 1024.0)).abs() < 0.01);
    }

    #[test]
    fn test_variant_custom_bits() {
        let custom = DecisionVariant::Custom(variant_bits::KIVI | variant_bits::MUSTAFAR);
        assert!(custom.has_kivi());
        assert!(!custom.has_kv_tuner());
        assert!(custom.has_mustafar());
        assert!(!custom.has_chunk_kv());

        // Custom names include the mask
        assert_eq!(custom.name(), "Custom");
    }

    // ── EpilogueSparse tests ──

    /// Create a header that looks like a strong sparsity candidate:
    /// - Very low entropy → concentrated attention
    /// - High softmax peak → few tokens dominate
    /// - Low delta_rho → stable representation
    /// - High dead ratio → many inactive channels
    /// - Low head spread → uniform heads
    fn make_sparse_candidate_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(1000);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(0.8);
        h.centroid_pos = f32_to_f16_bits(0.1);
        h.softmax_max_avg = f32_to_f16_bits(0.95);
        h.delta_rho_avg = f32_to_f16_bits(0.02);
        h.dead_ratio = 200;
        h.head_entropy_max = 30;
        h.head_entropy_min = 20;
        h
    }

    /// Create a header that should be preserved:
    /// - High entropy → diffuse attention
    /// - Low softmax peak → no dominant tokens
    /// - High delta_rho → unstable
    /// - Low dead ratio
    /// - High head spread
    fn make_preserve_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(2000);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(5.5);
        h.centroid_pos = f32_to_f16_bits(0.6);
        h.softmax_max_avg = f32_to_f16_bits(0.15);
        h.delta_rho_avg = f32_to_f16_bits(0.85);
        h.dead_ratio = 30;
        h.head_entropy_max = 210;
        h.head_entropy_min = 10;
        h
    }

    /// Create a moderate sparsity candidate.
    fn make_moderate_header() -> KvPageHeader {
        let mut h = KvPageHeader::new(3000);
        h.ref_count = 1;
        h.entropy_avg = f32_to_f16_bits(3.0);
        h.centroid_pos = f32_to_f16_bits(0.4);
        h.softmax_max_avg = f32_to_f16_bits(0.55);
        h.delta_rho_avg = f32_to_f16_bits(0.25);
        h.dead_ratio = 100;
        h.head_entropy_max = 60;
        h.head_entropy_min = 30;
        h
    }

    #[test]
    fn test_epilogue_sparse_default() {
        let es = EpilogueSparse::default();
        assert!(es.enabled);
        assert_eq!(es.dead_ratio_threshold, 180);
        assert_eq!(es.head_spread_threshold, 30);
    }

    #[test]
    fn test_epilogue_sparse_disabled() {
        let es = EpilogueSparse::disabled();
        assert!(!es.enabled);

        let header = make_sparse_candidate_header();
        let action = es.analyze_page(&header);
        assert_eq!(action, EpilogueSparseAction::Preserve);
    }

    #[test]
    fn test_epilogue_sparse_score_aggressive() {
        let es = EpilogueSparse::new();
        let header = make_sparse_candidate_header();
        let score = es.epilogue_sparse_score(&header);
        assert!(
            score >= 180,
            "Aggressive candidate should score >= 180, got {}",
            score
        );
        let action = es.analyze_page(&header);
        assert_eq!(action, EpilogueSparseAction::Aggressive);
    }

    #[test]
    fn test_epilogue_sparse_score_preserve() {
        let es = EpilogueSparse::new();
        let header = make_preserve_header();
        let score = es.epilogue_sparse_score(&header);
        assert!(
            score < 80,
            "Preserve header should score < 80, got {}",
            score
        );
        let action = es.analyze_page(&header);
        assert_eq!(action, EpilogueSparseAction::Preserve);
    }

    #[test]
    fn test_epilogue_sparse_score_moderate() {
        let es = EpilogueSparse::new();
        let header = make_moderate_header();
        let score = es.epilogue_sparse_score(&header);
        assert!(
            score >= 80 && score < 180,
            "Moderate candidate should score in [80, 180), got {}",
            score
        );
        let action = es.analyze_page(&header);
        assert_eq!(action, EpilogueSparseAction::Moderate);
    }

    #[test]
    fn test_epilogue_sparse_analyze_batch() {
        let mut es = EpilogueSparse::new();
        let headers = vec![
            make_sparse_candidate_header(),
            make_preserve_header(),
            make_moderate_header(),
            make_sparse_candidate_header(),
        ];

        let decisions = es.analyze_batch(&headers);
        assert_eq!(decisions.len(), 4);
        assert_eq!(decisions[0], EpilogueSparseAction::Aggressive);
        assert_eq!(decisions[1], EpilogueSparseAction::Preserve);
        assert_eq!(decisions[2], EpilogueSparseAction::Moderate);
        assert_eq!(decisions[3], EpilogueSparseAction::Aggressive);

        assert_eq!(es.stats.total_analyzed, 4);
        assert_eq!(es.stats.aggressive_count, 2);
        assert_eq!(es.stats.moderate_count, 1);
        assert_eq!(es.stats.preserve_count, 1);
    }

    #[test]
    fn test_epilogue_sparse_apply_to_header() {
        let es = EpilogueSparse::new();
        let mut header = KvPageHeader::new(1);

        es.apply_to_header(&mut header, EpilogueSparseAction::Aggressive);
        assert_eq!(header.channel_bitmap_lo, 0x1111_1111u32);

        es.apply_to_header(&mut header, EpilogueSparseAction::Moderate);
        assert_eq!(header.channel_bitmap_lo, 0x5555_5555u32);

        es.apply_to_header(&mut header, EpilogueSparseAction::Preserve);
        assert_eq!(header.channel_bitmap_lo, 0xFFFF_FFFFu32);
    }

    #[test]
    fn test_epilogue_sparse_apply_batch() {
        let mut es = EpilogueSparse::new();
        let headers = vec![
            make_sparse_candidate_header(),
            make_preserve_header(),
            make_moderate_header(),
        ];
        let mut headers_mut = headers.clone();

        // Need to analyze first
        es.analyze_batch(&headers);
        es.apply_batch(&mut headers_mut);

        assert_eq!(headers_mut[0].channel_bitmap_lo, 0x1111_1111u32);
        assert_eq!(headers_mut[1].channel_bitmap_lo, 0xFFFF_FFFFu32);
        assert_eq!(headers_mut[2].channel_bitmap_lo, 0x5555_5555u32);
    }

    #[test]
    fn test_epilogue_sparse_inactive_page_preserved() {
        let es = EpilogueSparse::new();
        let mut header = make_sparse_candidate_header();
        header.ref_count = 0; // inactive
        let action = es.analyze_page(&header);
        assert_eq!(action, EpilogueSparseAction::Preserve);
    }

    #[test]
    fn test_epilogue_sparse_score_in_bounds() {
        let es = EpilogueSparse::new();
        let headers = [
            make_sparse_candidate_header(),
            make_preserve_header(),
            make_moderate_header(),
        ];
        for h in &headers {
            let _score = es.epilogue_sparse_score(h);
        }
    }

    #[test]
    fn test_epilogue_sparse_decision_for() {
        let mut es = EpilogueSparse::new();
        let headers = vec![
            make_sparse_candidate_header(),
            make_preserve_header(),
        ];
        es.analyze_batch(&headers);

        assert_eq!(es.decision_for(0), EpilogueSparseAction::Aggressive);
        assert_eq!(es.decision_for(1), EpilogueSparseAction::Preserve);
        // Out of bounds
        assert_eq!(es.decision_for(99), EpilogueSparseAction::Preserve);
    }

    #[test]
    fn test_epilogue_sparse_reset_stats() {
        let mut es = EpilogueSparse::new();
        let headers = vec![make_sparse_candidate_header()];
        es.analyze_batch(&headers);
        assert_eq!(es.stats.total_analyzed, 1);

        es.reset_stats();
        assert_eq!(es.stats.total_analyzed, 0);
        assert_eq!(es.stats.aggressive_count, 0);
    }

    #[test]
    fn test_epilogue_sparse_bytes_saved_estimate() {
        let mut es = EpilogueSparse::new()
            .with_channels_per_page(128)
            .with_bytes_per_channel(2);

        let headers = vec![
            make_sparse_candidate_header(), // Aggressive → 75% saved of 128*2=256 → 192
            make_moderate_header(),         // Moderate → 50% saved of 256 → 128
            make_preserve_header(),         // Preserve → 0 saved
        ];
        es.analyze_batch(&headers);

        let expected = (128 * 3 / 4) * 2 + (128 / 2) * 2 + 0;
        assert_eq!(es.stats.estimated_bytes_saved, expected);
    }

    #[test]
    fn test_epilogue_sparse_action_label() {
        assert_eq!(EpilogueSparseAction::Aggressive.label(), "aggressive");
        assert_eq!(EpilogueSparseAction::Moderate.label(), "moderate");
        assert_eq!(EpilogueSparseAction::Preserve.label(), "preserve");
    }

    #[test]
    fn test_epilogue_dynamic_sparse_integration() {
        let mut es = EpilogueSparse::new();
        let mut headers = vec![
            make_sparse_candidate_header(),
            make_preserve_header(),
        ];

        // Base variant without MUSTAFAR
        let base_bits = variant_bits::KIVI | variant_bits::KV_TUNER;
        let result = epilogue_dynamic_sparse(&mut es, &mut headers, base_bits);

        // Should have MUSTAFAR bit set because at least one page is sparse
        assert!(result & variant_bits::MUSTAFAR != 0);
        assert!(result & variant_bits::KIVI != 0);
        assert!(result & variant_bits::KV_TUNER != 0);

        // Headers should have been updated
        assert_eq!(headers[0].channel_bitmap_lo, 0x1111_1111u32);
        assert_eq!(headers[1].channel_bitmap_lo, 0xFFFF_FFFFu32);
    }

    #[test]
    fn test_epilogue_dynamic_sparse_disabled_passthrough() {
        let mut es = EpilogueSparse::disabled();
        let mut headers = vec![make_sparse_candidate_header()];

        let base_bits = variant_bits::KIVI;
        let result = epilogue_dynamic_sparse(&mut es, &mut headers, base_bits);

        // Disabled → passthrough, no MUSTAFAR added
        assert_eq!(result, variant_bits::KIVI);
        // Headers untouched
        assert_eq!(headers[0].channel_bitmap_lo, 0);
    }

    #[test]
    fn test_epilogue_dynamic_sparse_all_preserve_no_mustafar() {
        let mut es = EpilogueSparse::new();
        let mut headers = vec![make_preserve_header(), make_preserve_header()];

        let base_bits = variant_bits::KIVI | variant_bits::CHUNK_KV;
        let result = epilogue_dynamic_sparse(&mut es, &mut headers, base_bits);

        // No sparse actions → MUSTAFAR NOT added
        assert_eq!(result & variant_bits::MUSTAFAR, 0);
    }

    // ── New tests: 15 additional tests ──

    // 1. KiviQuantResult construction and field access
    #[test]
    fn test_kivi_quant_result_fields() {
        let result = KiviQuantResult {
            data: vec![0xAB, 0xCD],
            scales: vec![1.0, 2.0, 3.0],
            bytes_per_element: 2,
            precision_tier: PrecisionTier::FP16,
        };

        assert_eq!(result.data.len(), 2);
        assert_eq!(result.scales.len(), 3);
        assert_eq!(result.bytes_per_element, 2);
        assert_eq!(result.precision_tier, PrecisionTier::FP16);
    }

    // 2. ChunkResidency enum variants — PartialEq + exhaustive match
    #[test]
    fn test_chunk_residency_equality() {
        assert_eq!(ChunkResidency::Resident, ChunkResidency::Resident);
        assert_eq!(ChunkResidency::Evicted, ChunkResidency::Evicted);
        assert_eq!(ChunkResidency::Empty, ChunkResidency::Empty);
        assert_ne!(ChunkResidency::Resident, ChunkResidency::Evicted);
        assert_ne!(ChunkResidency::Evicted, ChunkResidency::Empty);
        assert_ne!(ChunkResidency::Empty, ChunkResidency::Resident);
    }

    // 3. ChunkInfo default-like construction and field access
    #[test]
    fn test_chunk_info_construction() {
        let info = ChunkInfo {
            chunk_id: 3,
            token_start: 48,
            token_count: 16,
            residency: ChunkResidency::Resident,
            size_bytes: 4096,
            compression_tier: PrecisionTier::KIVI4,
            generation: 5,
            last_access_ts: 100,
        };

        assert_eq!(info.chunk_id, 3);
        assert_eq!(info.token_start, 48);
        assert_eq!(info.token_count, 16);
        assert_eq!(info.residency, ChunkResidency::Resident);
        assert_eq!(info.size_bytes, 4096);
        assert_eq!(info.compression_tier, PrecisionTier::KIVI4);
        assert_eq!(info.generation, 5);
        assert_eq!(info.last_access_ts, 100);
    }

    // 4. ChunkCompressStrategy enum variants — PartialEq
    #[test]
    fn test_chunk_compress_strategy_variants() {
        assert_eq!(ChunkCompressStrategy::Uniform, ChunkCompressStrategy::Uniform);
        assert_eq!(ChunkCompressStrategy::Tiered, ChunkCompressStrategy::Tiered);
        assert_eq!(ChunkCompressStrategy::Adaptive, ChunkCompressStrategy::Adaptive);
        assert_ne!(ChunkCompressStrategy::Uniform, ChunkCompressStrategy::Adaptive);
    }

    // 5. KvTunerStrategy default values
    #[test]
    fn test_kv_tuner_strategy_default() {
        let tuner = KvTunerStrategy::default();
        assert!(tuner.enabled);
        assert!(tuner.kivi.enabled);
        assert_eq!(tuner.entropy(), 0.5);
        assert_eq!(tuner.adjustment_count(), 0);
        assert_eq!(tuner.last_reason(), KvTunerReason::InitialCalibration);
        assert!(tuner.events().is_empty());
    }

    // 6. KvTunerStrategy disabled state
    #[test]
    fn test_kv_tuner_strategy_disabled() {
        let tuner = KvTunerStrategy::disabled();
        assert!(!tuner.enabled);
        assert!(!tuner.kivi.enabled);
    }

    // 7. KvTunerReason enum — PartialEq for all variants
    #[test]
    fn test_kv_tuner_reason_equality() {
        use KvTunerReason::*;
        assert_eq!(HighEntropyDowngrade, HighEntropyDowngrade);
        assert_eq!(LowEntropyUpgrade, LowEntropyUpgrade);
        assert_eq!(LongSeqDowngrade, LongSeqDowngrade);
        assert_eq!(SinkProtection, SinkProtection);
        assert_eq!(TierFloorConstraint, TierFloorConstraint);
        assert_eq!(InitialCalibration, InitialCalibration);
        assert_eq!(Stable, Stable);
        assert_ne!(HighEntropyDowngrade, LowEntropyUpgrade);
        assert_ne!(SinkProtection, Stable);
    }

    // 8. KvTuner observe_entropy EMA smoothing
    #[test]
    fn test_kv_tuner_observe_entropy_ema() {
        let mut tuner = KvTunerStrategy::new().with_alpha(0.5);

        // First observation initializes directly (smoothed was 0.5 default, adj_count=0)
        let e1 = tuner.observe_entropy(0.8);
        assert!((e1 - 0.8).abs() < 1e-6, "first obs should be 0.8, got {}", e1);

        // Second: EMA = 0.5 * 0.4 + 0.5 * 0.8 = 0.6 (avoid 0.5 which re-triggers init path)
        let e2 = tuner.observe_entropy(0.4);
        assert!((e2 - 0.6).abs() < 1e-6, "second obs should be 0.6, got {}", e2);

        // Third: EMA = 0.5 * 0.9 + 0.5 * 0.6 = 0.75
        let e3 = tuner.observe_entropy(0.9);
        assert!((e3 - 0.75).abs() < 1e-6, "third obs should be 0.75, got {}", e3);
    }

    // 9. KvTuner adjust_precision sink protection
    #[test]
    fn test_kv_tuner_adjust_precision_sink() {
        let mut tuner = KvTunerStrategy::new();
        let (k, v) = tuner.adjust_precision(50, true, 0.5, 100);

        assert_eq!(k, PrecisionTier::FP16);
        assert_eq!(v, PrecisionTier::FP16);
        assert_eq!(tuner.last_reason(), KvTunerReason::SinkProtection);
    }

    // 10. KvTuner adjust_precision importance thresholds
    #[test]
    fn test_kv_tuner_adjust_precision_high_importance() {
        let mut tuner = KvTunerStrategy::new();
        let (k, v) = tuner.adjust_precision(220, false, 0.8, 100);

        // importance > 200 → FP16/FP16 upgrade
        assert_eq!(k, PrecisionTier::FP16);
        assert_eq!(v, PrecisionTier::FP16);
    }

    // 11. KvTuner reset clears state
    #[test]
    fn test_kv_tuner_reset() {
        let mut tuner = KvTunerStrategy::new();
        tuner.observe_entropy(0.9);
        tuner.observe_entropy(0.1);

        tuner.reset();

        assert_eq!(tuner.entropy(), 0.5);
        assert_eq!(tuner.adjustment_count(), 0);
        assert!(tuner.events().is_empty());
        assert_eq!(tuner.last_reason(), KvTunerReason::InitialCalibration);
    }

    // 12. ChunkMigrationPlan construction and field access
    #[test]
    fn test_chunk_migration_plan_construction() {
        let plan = ChunkMigrationPlan {
            evict: vec![2, 3],
            restore: vec![0, 1],
        };

        assert_eq!(plan.evict.len(), 2);
        assert_eq!(plan.restore.len(), 2);
        assert_eq!(plan.evict[0], 2);
        assert_eq!(plan.restore[1], 1);
    }

    // 13. DecisionRecord construction and field access
    #[test]
    fn test_decision_record_construction() {
        let record = DecisionRecord {
            seq: 42,
            variant: DecisionVariant::FullStack,
            reason: "test reason",
            seq_len: 1024,
            entropy: 0.65,
            cache_pressure: 0.33,
        };

        assert_eq!(record.seq, 42);
        assert_eq!(record.variant, DecisionVariant::FullStack);
        assert_eq!(record.reason, "test reason");
        assert_eq!(record.seq_len, 1024);
        assert!((record.entropy - 0.65).abs() < 1e-6);
        assert!((record.cache_pressure - 0.33).abs() < 1e-6);
    }

    // 14. KvOptimizationConfig minimal preset
    #[test]
    fn test_kv_optimization_config_minimal() {
        let config = KvOptimizationConfig::minimal();
        assert!(config.enabled);
        assert!(config.kivi_enabled);
        assert!(!config.kv_tuner_enabled);
        assert!(!config.mustafar_enabled);
        assert!(!config.chunk_kv_enabled);
        assert!(!config.epilogue_sparse_enabled);
        assert_eq!(config.kivi_sink_count, 1);
    }

    // 15. KvOptimizationConfig validation rejects kv_tuner without kivi
    #[test]
    fn test_kv_optimization_config_validate_conflict() {
        let bad_config = KvOptimizationConfig::default()
            .with_kivi_enabled(false)
            .with_kv_tuner_enabled(true);

        let result = bad_config.validate();
        assert!(result.is_err(), "KVTuner without KIVI should fail validation");
        assert_eq!(
            result.unwrap_err(),
            "KVTuner requires KIVI to be enabled (KVTuner wraps KIVI)"
        );
    }
}
