#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use ::safetensors::tensor::Dtype;

    /// Naive reference transpose: dst[c * rows + r] = src[r * cols + c].
    fn naive_transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        out
    }

    #[test]
    fn cache_blocked_transpose_matches_naive_small_sizes() {
        // Exercise exact-tile, tile+tail, non-multiple sizes, 1×N, N×1.
        let cases: &[(usize, usize)] = &[
            (1, 1), (1, 7), (7, 1), (8, 8), (63, 65), (64, 64), (65, 63),
            (128, 65), (65, 128), (100, 200), (200, 100), (129, 129),
        ];
        for &(rows, cols) in cases {
            let n = rows * cols;
            let src: Vec<f32> = (0..n as u32).map(|x| x as f32 * 0.5 - 7.25).collect();
            let mut blocked = vec![0.0f32; n];
            cache_blocked_transpose_f32(&src, &mut blocked, rows, cols);
            let reference = naive_transpose(&src, rows, cols);
            assert_eq!(
                blocked, reference,
                "cache_blocked_transpose diverged at rows={}, cols={}",
                rows, cols
            );
        }
    }

    #[test]
    fn cache_blocked_transpose_matches_naive_weight_size() {
        // Realistic Linear weight: [1536, 12288] — exactly the case naive
        // transpose degrades on (6144-byte write stride vs 64-byte L1 line).
        let rows = 1536;
        let cols = 12288;
        let n = rows * cols;
        // Build a deterministic pattern; f32 bit-exactness required.
        let src: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761);
                f32::from_bits(x)
            })
            // Filter out NaN so equality works; map bit pattern to a finite value.
            .map(|f| if f.is_finite() { f } else { 0.0 })
            .collect();
        let mut blocked = vec![0.0f32; n];
        cache_blocked_transpose_f32(&src, &mut blocked, rows, cols);

        // Spot-check a scattered set of coordinates rather than allocate a second
        // full buffer (the naive path is painfully slow on this shape).
        let sample_points: [(usize, usize); 32] = [
            (0, 0), (1535, 12287), (0, 12287), (1535, 0),
            (1, 1), (2, 3), (5, 7), (11, 13), (17, 19),
            (23, 29), (31, 37), (41, 43), (47, 53), (59, 61),
            (67, 71), (73, 79), (83, 89), (97, 101), (103, 107),
            (109, 113), (127, 131), (137, 139), (149, 151), (157, 163),
            (167, 173), (179, 181), (191, 193), (197, 199), (211, 223),
            (1000, 7000), (1234, 5678), (999, 9999),
        ];
        for (r, c) in sample_points {
            assert_eq!(
                blocked[c * rows + r], src[r * cols + c],
                "mismatch at (r={}, c={})", r, c
            );
        }
    }

    #[test]
    fn parallel_bf16_to_f32_matches_serial() {
        // 100_000 BF16 elements covering finite positive + negative + zero.
        let n = 100_000usize;
        let bf16s: Vec<half::bf16> = (0..n)
            .map(|i| {
                let x = (i as i32) - (n as i32 / 2);
                half::bf16::from_f32(x as f32 * 0.0078125)
            })
            .collect();
        // Flatten to bytes
        let mut bytes = Vec::with_capacity(n * 2);
        for v in &bf16s {
            let raw: u16 = v.to_bits();
            bytes.extend_from_slice(&raw.to_le_bytes());
        }

        let parallel_out = parallel_half_to_f32::<half::bf16>(&bytes).expect("parallel bf16 conversion failed");
        // Reference: single-threaded conversion.
        let serial_out: Vec<f32> = bf16s.iter().map(|v| v.to_f32()).collect();

        assert_eq!(parallel_out.len(), serial_out.len());
        for (i, (p, s)) in parallel_out.iter().zip(serial_out.iter()).enumerate() {
            assert_eq!(
                p.to_bits(), s.to_bits(),
                "bit-pattern mismatch at index {}: parallel={:?}, serial={:?}",
                i, p, s
            );
        }
    }

    #[test]
    fn parallel_f16_to_f32_matches_serial() {
        // F16 path: 50_000 values, including subnormals near zero.
        let n = 50_000usize;
        let f16s: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32(((i as f32) - (n as f32) / 2.0) * 1.0e-3))
            .collect();
        let mut bytes = Vec::with_capacity(n * 2);
        for v in &f16s {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }

        let parallel_out = parallel_half_to_f32::<half::f16>(&bytes).expect("parallel f16 conversion failed");
        let serial_out: Vec<f32> = f16s.iter().map(|v| v.to_f32()).collect();
        for (i, (p, s)) in parallel_out.iter().zip(serial_out.iter()).enumerate() {
            assert_eq!(
                p.to_bits(), s.to_bits(),
                "bit-pattern mismatch at index {}: parallel={:?}, serial={:?}",
                i, p, s
            );
        }
    }

    #[test]
    fn parallel_f32_passthrough_is_exact() {
        let n = 12345usize;
        let src: Vec<f32> = (0..n).map(|i| (i as f32) * 1.5 - 3.0).collect();
        let mut bytes = Vec::with_capacity(n * 4);
        for v in &src {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = parallel_bytes_to_f32_lossless(&bytes).expect("parallel f32 passthrough failed");
        assert_eq!(out, src);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_24_prune_applied_on_load() {
        // 构造一个待测试的 gate_proj (能触发真 2:4 降维条件的 TensorName)
        let mut meta = TensorMeta {
            name: "layers.0.mlp.gate_proj".to_string(),
            shape: vec![8, 16], // 8 rows, 16 cols
            dtype: Dtype::F32,
        };
        // 8 * 16 = 128 elements, ones to represent some non-zero data
        // For actual zeroing to be visible or not matter, we just care that shape halves
        let mut data = vec![1.0f32; 128];
        
        let sp_meta_opt = compress_24_sparsity_heuristic(&mut meta, &mut data);
        
        // 1. 产生 sp_meta 位掩码
        assert!(sp_meta_opt.is_some(), "Hardware 2:4 sp_meta must be returned for gate_proj");
        
        // 2. 原本的结构体 metadata 的列 (col) 应当被严格砍半
        assert_eq!(meta.shape, vec![8, 8], "Column dimension of shape metadata must be exactly halved");
        
        // 3. F32 buffer 数据长度必须被严格砍半 (128 -> 64)，实现真显存下降
        assert_eq!(data.len(), 64, "Flat byte data length must be halved into a physically dense structure");
    }

    /// Verify AWQ4 repacking produces correct 72-byte block format.
    #[test]
    fn test_awq4_repack_block_layout() {
        let n = 2; // 2 output rows
        let k = 128; // 1 group of 128 elements
        let qw_cols = k / 8; // 16 int32s per row

        // Build simple qweight: each row has sequential nibbles 0..15 repeated
        let mut qweight = vec![0u32; n * qw_cols];
        for row in 0..n {
            for pack in 0..qw_cols {
                let base = ((row * 16 + pack) % 16) as u32;
                let word = base | (base << 4) | (base << 8) | (base << 12)
                    | (base << 16) | (base << 20) | (base << 24) | (base << 28);
                qweight[row * qw_cols + pack] = word;
            }
        }
        let qweight_bytes: Vec<u8> = qweight.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Scales: 1 f16 per (row, group) = 2 total
        let scales: Vec<half::f16> = vec![half::f16::from_f32(2.0), half::f16::from_f32(3.0)];
        let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Zeros: 1 group × ceil(2/8) = 1 int32. Row 0 nibble = 0x5, row 1 nibble = 0x3
        let zeros_u32: Vec<u32> = vec![0x35]; // row0 nibble=5, row1 nibble=3
        let zeros_bytes: Vec<u8> = zeros_u32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let packed = repack_awq_gptq_blocks(
            &qweight_bytes, &scales_bytes, &zeros_bytes,
            None, gllm_kernels::quant::QuantType::AWQ4, n, k,
        );

        // Should be 2 blocks × 72 bytes = 144 bytes
        assert_eq!(packed.len(), 144);

        // Block 0 (row 0): scale=2.0, zero=5+1=6.0
        let scale0 = half::f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        assert!((scale0 - 2.0).abs() < 0.01);
        let zero0 = half::f16::from_le_bytes([packed[4], packed[5]]).to_f32();
        assert!((zero0 - 6.0).abs() < 0.01);

        // Block 1 (row 1): scale=3.0, zero=3+1=4.0
        let scale1 = half::f16::from_le_bytes([packed[72], packed[73]]).to_f32();
        assert!((scale1 - 3.0).abs() < 0.01);
        let zero1 = half::f16::from_le_bytes([packed[76], packed[77]]).to_f32();
        assert!((zero1 - 4.0).abs() < 0.01);

        // Qweight data at offset 8 should match original
        for i in 0..64 {
            assert_eq!(packed[8 + i], qweight_bytes[i], "qweight byte mismatch at {}", i);
        }
    }

    /// REQ-QCG-010/012 验收:repack 后的 byte 量与 SPEC §2.2 AWQ4 (72 bytes/group) 一致,
    /// 总 block 数 = N × (K/group_size)。process_single_tensor 推导出的 (n, k) 在
    /// element-level shape [N, K] 下与 repack 内部 group 数学完全等价。
    #[test]
    fn test_awq4_repack_total_size_matches_spec() {
        use gllm_kernels::quant::QuantType;

        // SPEC §2.2: AWQ4 group_size=128, block_bytes=72 per group
        let n = 4usize;   // 4 output features
        let k = 256usize; // 2 groups of 128
        let qw_cols = k / 8; // 32 u32 per row (packed nibbles)

        // dummy qweight: zeroed for size check
        let qweight_bytes = vec![0u8; n * qw_cols * 4];
        let scales_bytes = vec![0u8; n * (k / 128) * 2]; // f16 per (row, group)
        let qz_packed_cols = (n + 7) / 8; // 1
        let zeros_bytes = vec![0u8; (k / 128) * qz_packed_cols * 4];

        let packed = repack_awq_gptq_blocks(
            &qweight_bytes, &scales_bytes, &zeros_bytes,
            None, QuantType::AWQ4, n, k,
        );

        // Total = N rows × (K/group_size) groups × 72 bytes
        let expected = n * (k / 128) * 72;
        assert_eq!(packed.len(), expected,
            "repack_awq_gptq_blocks total bytes mismatch: expected {} = {}×{}×72, got {}",
            expected, n, k / 128, packed.len());

        // SPEC §2.2 invariant: each block is exactly 72 bytes.
        // block_size=128 elements per block ⇒ K/128 blocks per row.
        assert_eq!(QuantType::AWQ4.block_bytes(), 72);
        assert_eq!(QuantType::AWQ4.block_size(), 128);
        assert_eq!(QuantType::GPTQ4.block_bytes(), 72);
        assert_eq!(QuantType::GPTQ4.block_size(), 128);
    }

    /// REQ-QCG-010 验收:repack 后 block 内的 dequant 数学等价于 HF 原始
    /// (qw - zp) × scale 公式。这是 JIT QuantGemm 寄存器内反量化的字节级准确性前提。
    #[test]
    fn test_awq4_repack_dequant_math_equivalent() {
        use gllm_kernels::quant::QuantType;

        let n = 2usize;
        let k = 128usize;
        let qw_cols = k / 8;

        let mut qw = vec![0u32; n * qw_cols];
        for col in 0..qw_cols {
            qw[col] = 0xFEDC_BA98;
            qw[qw_cols + col] = 0x7654_3210;
        }
        let qweight_bytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();

        let scales_f16 = [half::f16::from_f32(2.0), half::f16::from_f32(0.5)];
        let scales_bytes: Vec<u8> = scales_f16.iter().flat_map(|v| v.to_le_bytes()).collect();

        let zeros_u32 = [0x73u32];
        let zeros_bytes: Vec<u8> = zeros_u32.iter().flat_map(|v| v.to_le_bytes()).collect();

        let packed = repack_awq_gptq_blocks(
            &qweight_bytes, &scales_bytes, &zeros_bytes,
            None, QuantType::AWQ4, n, k,
        );

        // Block 0: row 0
        let scale0 = half::f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        let zero0 = half::f16::from_le_bytes([packed[4], packed[5]]).to_f32();
        assert!((scale0 - 2.0).abs() < 0.01, "row 0 scale = {}", scale0);
        assert!((zero0 - 4.0).abs() < 0.01, "row 0 zero (3+1) = {}", zero0);

        let qw0_first = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);
        assert_eq!(qw0_first, 0xFEDC_BA98, "row 0 qweight u32[0]");

        let nibble_0 = (qw0_first & 0xF) as f32;
        let dequant_0 = scale0 * (nibble_0 - zero0);
        assert!((dequant_0 - 8.0).abs() < 0.01,
            "dequant math: 2.0 * (8 - 4) = 8.0, got {}", dequant_0);

        // Block 1: row 1
        let scale1 = half::f16::from_le_bytes([packed[72], packed[73]]).to_f32();
        let zero1 = half::f16::from_le_bytes([packed[76], packed[77]]).to_f32();
        assert!((scale1 - 0.5).abs() < 0.01);
        assert!((zero1 - 8.0).abs() < 0.01);

        let qw1_first = u32::from_le_bytes([packed[80], packed[81], packed[82], packed[83]]);
        assert_eq!(qw1_first, 0x7654_3210);
        let nibble_r1 = (qw1_first & 0xF) as f32;
        let dequant_r1 = scale1 * (nibble_r1 - zero1);
        assert!((dequant_r1 - (-4.0)).abs() < 0.01,
            "row 1 dequant: 0.5 * (0 - 8) = -4.0, got {}", dequant_r1);
    }

    /// End-to-end AWQ4 pipeline test: synthetic HF data → repack → QuantizedTensor metadata.
    ///
    /// Validates the full path from HuggingFace-style separate tensors to the
    /// QuantizedTensor that auto_graph would consume for QuantGemm dispatch.
    #[test]
    fn test_awq4_e2e_pipeline() {
        use gllm_kernels::quant::QuantType;

        let n = 8usize;   // 8 output features
        let k = 256usize; // 2 groups of 128
        let qw_cols = k / 8; // 32 u32 per row

        // Synthetic qweight: deterministic pattern for reproducibility
        let mut qw = vec![0u32; n * qw_cols];
        for row in 0..n {
            for col in 0..qw_cols {
                let nibble = ((row + col) % 16) as u32;
                qw[row * qw_cols + col] = nibble | (nibble << 4) | (nibble << 8)
                    | (nibble << 12) | (nibble << 16) | (nibble << 20)
                    | (nibble << 24) | (nibble << 28);
            }
        }
        let qweight_bytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Scales: f16 per (row, group)
        let num_groups = k / AWQ_GROUP_SIZE;
        let mut scales = Vec::with_capacity(n * num_groups);
        for row in 0..n {
            for g in 0..num_groups {
                let val = 1.0 + (row as f32 * 0.1) + (g as f32 * 0.5);
                scales.push(half::f16::from_f32(val));
            }
        }
        let scales_bytes: Vec<u8> = scales.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Zeros: packed INT4 per (group, rows_packed)
        let rows_packed = (n + 7) / 8; // 1
        let mut zeros_u32 = vec![0u32; num_groups * rows_packed];
        for g in 0..num_groups {
            let mut packed = 0u32;
            for row in 0..n {
                let zp = ((row + g + 1) % 16) as u32;
                packed |= zp << ((row % 8) * 4);
            }
            zeros_u32[g * rows_packed] = packed;
        }
        let zeros_bytes: Vec<u8> = zeros_u32.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Repack to block format
        let packed = repack_awq_gptq_blocks(
            &qweight_bytes, &scales_bytes, &zeros_bytes,
            None, QuantType::AWQ4, n, k,
        );

        // Build QuantizedTensor
        let qt = QuantizedTensor {
            data: packed.clone(),
            quant_type: QuantType::AWQ4,
            shape: vec![n, k],
            ggml_dtype: crate::loader::gguf::GgmlDType::AWQ4,
        };

        // Validate metadata
        assert_eq!(qt.quant_type, QuantType::AWQ4);
        assert_eq!(qt.shape, vec![n, k]);
        assert_eq!(qt.quant_type.block_size(), 128);
        assert_eq!(qt.quant_type.block_bytes(), 72);

        // Total blocks = n * num_groups = 8 * 2 = 16
        let total_blocks = n * num_groups;
        assert_eq!(packed.len(), total_blocks * 72);

        // Spot-check block 0: scale should be scales[0] = 1.0
        let scale0 = half::f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        assert!((scale0 - 1.0).abs() < 0.01, "block 0 scale = {scale0}");

        // Zero-point: first row's zero = (0 + 0 + 1) % 16 = 1, +1 offset = 2.0
        let zero0 = half::f16::from_le_bytes([packed[4], packed[5]]).to_f32();
        assert!((zero0 - 2.0).abs() < 0.01, "block 0 zero = {zero0}");

        // Spot-check block 1 (row 0, group 1): scale = 1.0 + 0 + 0.5 = 1.5
        let scale1 = half::f16::from_le_bytes([packed[72], packed[73]]).to_f32();
        assert!((scale1 - 1.5).abs() < 0.01, "block 1 scale = {scale1}");

        // Spot-check last block (row 7, group 1)
        let last_offset = (total_blocks - 1) * 72;
        let scale_last = half::f16::from_le_bytes([
            packed[last_offset], packed[last_offset + 1],
        ]).to_f32();
        // scales for row 7, group 1: 1.0 + 7*0.1 + 1*0.5 = 2.2
        assert!((scale_last - 2.2).abs() < 0.01, "last block scale = {scale_last}");
    }

    /// End-to-end GPTQ4 pipeline test: synthetic HF data → repack → QuantizedTensor.
    /// Validates GPTQ's INT4 packed zero-point +1 offset compensation in repack.
    #[test]
    fn test_gptq4_e2e_pipeline() {
        use gllm_kernels::quant::QuantType;

        let n = 4usize;
        let k = 128usize; // 1 group
        let qw_cols = k / 8;

        let qweight_bytes = vec![0xABu8; n * qw_cols * 4];
        let scales_f16: Vec<half::f16> = (0..n).map(|i| half::f16::from_f32(1.0 + i as f32)).collect();
        let scales_bytes: Vec<u8> = scales_f16.iter().flat_map(|v| v.to_le_bytes()).collect();

        // GPTQ zeros: each u32 packs 8 zero-point nibbles. row i's zp = i+2.
        let mut z = 0u32;
        for i in 0..n.min(8) {
            z |= ((i + 2) as u32) << (i * 4);
        }
        let zeros_bytes: Vec<u8> = z.to_le_bytes().to_vec();

        // g_idx: sequential 0..K (all in group 0)
        let g_idx: Vec<i32> = (0..k as i32).collect();

        let packed = repack_awq_gptq_blocks(
            &qweight_bytes, &scales_bytes, &zeros_bytes,
            Some(&g_idx), QuantType::GPTQ4, n, k,
        );

        let qt = QuantizedTensor {
            data: packed.clone(),
            quant_type: QuantType::GPTQ4,
            shape: vec![n, k],
            ggml_dtype: crate::loader::gguf::GgmlDType::GPTQ4,
        };

        assert_eq!(qt.quant_type, QuantType::GPTQ4);
        assert_eq!(qt.shape, vec![n, k]);
        assert_eq!(qt.quant_type.block_bytes(), 72);

        let total_blocks = n * (k / 128);
        assert_eq!(packed.len(), total_blocks * 72);

        // Block 0: scale = scales_f16[0] = 1.0
        let scale0 = half::f16::from_le_bytes([packed[0], packed[1]]).to_f32();
        assert!((scale0 - 1.0).abs() < 0.01);

        // Block 0: zero = (2 + 1) = 3.0 (GPTQ INT4 packed +1 offset)
        let zero0 = half::f16::from_le_bytes([packed[4], packed[5]]).to_f32();
        assert!((zero0 - 3.0).abs() < 0.01, "GPTQ zero-point +1 offset: expected 3.0, got {zero0}");
    }

    /// Validates that QuantFormatDescriptor for AWQ4/GPTQ4 matches the repack
    /// block layout (72 bytes, block_size=128, PackedInt4 data kind).
    #[test]
    fn test_awq_gptq_quant_format_descriptor_matches_repack() {
        use gllm_kernels::quant::QuantType;
        use gllm_kernels::quant_format::QuantFormatRegistry;

        let reg = QuantFormatRegistry::new();

        // AWQ4 descriptor consistency
        let awq_desc = reg.get(&QuantType::AWQ4).expect("AWQ4 registered");
        assert_eq!(awq_desc.block_bytes, 72);
        assert_eq!(awq_desc.block_size, 128);
        assert_eq!(awq_desc.bits_per_element, 4);
        assert!(matches!(awq_desc.data_kind, gllm_kernels::quant_format::QuantDataKind::PackedInt4));

        // GPTQ4 descriptor consistency
        let gptq_desc = reg.get(&QuantType::GPTQ4).expect("GPTQ4 registered");
        assert_eq!(gptq_desc.block_bytes, 72);
        assert_eq!(gptq_desc.block_size, 128);
        assert_eq!(gptq_desc.bits_per_element, 4);
        assert!(matches!(gptq_desc.data_kind, gllm_kernels::quant_format::QuantDataKind::PackedInt4));
    }

    // ── MTP tensor role matching tests ──

    #[test]
    fn mtp_projection_deepseek_v3_global() {
        // DeepSeek V3: model.mtp_head.{k}.weight
        let (role, layer) = match_tensor_role("model.mtp_head.0.weight").unwrap();
        assert_eq!(role, TensorRole::MtpProjection);
        assert_eq!(layer, None);

        let (role2, layer2) = match_tensor_role("model.mtp_head.1.weight").unwrap();
        assert_eq!(role2, TensorRole::MtpProjection);
        assert_eq!(layer2, None);
    }

    #[test]
    fn mtp_projection_qwen3_global() {
        // Qwen3: model.mtp.{k}.weight
        let (role, layer) = match_tensor_role("model.mtp.0.weight").unwrap();
        assert_eq!(role, TensorRole::MtpProjection);
        assert_eq!(layer, None);
    }

    #[test]
    fn mtp_projection_per_layer() {
        // Per-layer variant: model.layers.{N}.mtp_proj.{k}.weight
        let (role, layer) = match_tensor_role("model.layers.5.mtp_proj.0.weight").unwrap();
        assert_eq!(role, TensorRole::MtpProjection);
        assert_eq!(layer, Some(5));
    }

    #[test]
    fn mtp_projection_rejects_unrelated() {
        // "output.weight" should still map to OutputHead, not MTP
        let (role, _) = match_tensor_role("model.output.weight").unwrap();
        assert_eq!(role, TensorRole::OutputHead);

        // Names without mtp patterns should return None for MTP
        assert!(match_tensor_role("model.lm_head.weight").is_some());
        let (lm_role, _) = match_tensor_role("model.lm_head.weight").unwrap();
        assert_eq!(lm_role, TensorRole::OutputHead);
    }

    // ── match_tensor_role: LLaMA-style names ──

    #[test]
    fn match_llama_embedding() {
        let (role, layer) = match_tensor_role("model.embed_tokens.weight").unwrap();
        assert_eq!(role, TensorRole::Embedding);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_llama_lm_head() {
        let (role, layer) = match_tensor_role("lm_head.weight").unwrap();
        assert_eq!(role, TensorRole::OutputHead);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_llama_final_norm() {
        let (role, layer) = match_tensor_role("model.norm.weight").unwrap();
        assert_eq!(role, TensorRole::FinalNorm);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_llama_layer_q_proj() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionQuery);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_llama_layer_k_proj() {
        let (role, layer) = match_tensor_role("model.layers.5.self_attn.k_proj.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionKey);
        assert_eq!(layer, Some(5));
    }

    #[test]
    fn match_llama_ffn_gate() {
        let (role, layer) = match_tensor_role("model.layers.3.mlp.gate_proj.weight").unwrap();
        assert_eq!(role, TensorRole::FfnGate);
        assert_eq!(layer, Some(3));
    }

    #[test]
    fn match_llama_ffn_up() {
        let (role, layer) = match_tensor_role("model.layers.1.mlp.up_proj.weight").unwrap();
        assert_eq!(role, TensorRole::FfnUp);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_llama_ffn_down() {
        let (role, layer) = match_tensor_role("model.layers.2.mlp.down_proj.weight").unwrap();
        assert_eq!(role, TensorRole::FfnDown);
        assert_eq!(layer, Some(2));
    }

    // ── match_tensor_role: BERT-style names ──

    #[test]
    fn match_bert_query() {
        let (role, layer) = match_tensor_role("roberta.encoder.layer.0.attention.self.query.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionQuery);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_bert_key() {
        let (role, layer) = match_tensor_role("roberta.encoder.layer.1.attention.self.key.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionKey);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_bert_ffn_up() {
        let (role, layer) = match_tensor_role("roberta.encoder.layer.0.intermediate.dense.weight").unwrap();
        assert_eq!(role, TensorRole::FfnUp);
        assert_eq!(layer, Some(0));
    }

    // ── match_tensor_role: GGUF-style names ──

    #[test]
    fn match_gguf_token_embd() {
        let (role, layer) = match_tensor_role("token_embd.weight").unwrap();
        assert_eq!(role, TensorRole::Embedding);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_gguf_output() {
        let (role, layer) = match_tensor_role("output.weight").unwrap();
        assert_eq!(role, TensorRole::OutputHead);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_gguf_blk_wq() {
        let (role, layer) = match_tensor_role("blk.0.attn_q.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionQuery);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_gguf_blk_ffn_gate() {
        let (role, layer) = match_tensor_role("blk.2.ffn_gate.weight").unwrap();
        assert_eq!(role, TensorRole::FfnGate);
        assert_eq!(layer, Some(2));
    }

    #[test]
    fn match_gguf_blk_norm() {
        let (role, layer) = match_tensor_role("blk.0.attn_norm.weight").unwrap();
        assert_eq!(role, TensorRole::InputNorm);
        assert_eq!(layer, Some(0));
    }

    // ── match_tensor_role: MLA (DeepSeek) ──

    #[test]
    fn match_mla_kv_compress() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.kv_b_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaKvCompress);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_mla_q_compress() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.q_a_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaQCompress);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_mla_rope_key() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.k_pe_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaRopeKey);
        assert_eq!(layer, Some(0));
    }

    // ── match_tensor_role: edge cases ──

    #[test]
    fn match_bias_tensor_returns_none() {
        assert!(match_tensor_role("model.layers.0.self_attn.q_proj.bias").is_none());
        assert!(match_tensor_role("model.layers.0.self_attn.q_proj_weight_bias").is_none());
    }

    #[test]
    fn match_unknown_tensor_returns_none() {
        assert!(match_tensor_role("some_random_tensor").is_none());
        assert!(match_tensor_role("model.foo.bar.weight").is_none());
    }

    #[test]
    fn match_case_insensitive() {
        // match_tensor_role lowercases internally
        let (role, _) = match_tensor_role("MODEL.EMBED_TOKENS.WEIGHT").unwrap();
        assert_eq!(role, TensorRole::Embedding);
        let (role2, _) = match_tensor_role("LM_HEAD.WEIGHT").unwrap();
        assert_eq!(role2, TensorRole::OutputHead);
    }

    #[test]
    fn match_moe_gate() {
        let (role, layer) = match_tensor_role("model.layers.0.mlp.gate.weight").unwrap();
        assert_eq!(role, TensorRole::MoEGate);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_moe_router() {
        let (role, layer) = match_tensor_role("model.layers.1.mlp.router.weight").unwrap();
        assert_eq!(role, TensorRole::MoEGate);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_position_embedding() {
        let (role, layer) = match_tensor_role("roberta.embeddings.position_embedding.weight").unwrap();
        assert_eq!(role, TensorRole::PositionEmbedding);
        assert_eq!(layer, None);
    }

    #[test]
    fn match_classifier_dense() {
        let (role, layer) = match_tensor_role("classifier.dense.weight").unwrap();
        assert_eq!(role, TensorRole::ClassifierDense);
        assert_eq!(layer, None);
    }

    // ── fallback_source ──

    #[test]
    fn fallback_source_hf_to_modelscope() {
        assert_eq!(fallback_source(ModelSource::HuggingFace), ModelSource::ModelScope);
    }

    #[test]
    fn fallback_source_modelscope_to_hf() {
        assert_eq!(fallback_source(ModelSource::ModelScope), ModelSource::HuggingFace);
    }

    // ── is_recoverable_error ──

    #[test]
    fn recoverable_network_error() {
        let err = LoaderError::Network("timeout".into());
        assert!(is_recoverable_error(&err));
    }

    #[test]
    fn recoverable_io_error() {
        let err = LoaderError::Io(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout"));
        assert!(is_recoverable_error(&err));
    }

    #[test]
    fn non_recoverable_missing_weights() {
        let err = LoaderError::MissingWeights;
        assert!(!is_recoverable_error(&err));
    }

    #[test]
    fn non_recoverable_duplicate_tensor() {
        let err = LoaderError::DuplicateTensor("x".into());
        assert!(!is_recoverable_error(&err));
    }

    // ── ChecksumPolicy / ModelSource ──

    #[test]
    fn checksum_policy_default_is_ignore() {
        assert_eq!(ChecksumPolicy::default(), ChecksumPolicy::Ignore);
    }

    #[test]
    fn model_source_equality() {
        assert_eq!(ModelSource::HuggingFace, ModelSource::HuggingFace);
        assert_ne!(ModelSource::HuggingFace, ModelSource::ModelScope);
    }

    #[test]
    fn loader_config_default_source() {
        let config = LoaderConfig::default();
        assert_eq!(config.source, ModelSource::HuggingFace);
        assert!(config.enable_fallback);
        assert_eq!(config.checksum_policy, ChecksumPolicy::Ignore);
        assert!(config.gguf_file_filter.is_none());
    }

    // ── CacheLayout ──

    #[test]
    fn cache_layout_dirs() {
        let layout = CacheLayout::new(PathBuf::from("/tmp/test_cache")).unwrap();
        assert_eq!(layout.hf_cache_dir(), PathBuf::from("/tmp/test_cache/huggingface"));
        assert_eq!(layout.modelscope_cache_dir(), PathBuf::from("/tmp/test_cache/modelscope"));
    }

    #[test]
    fn cache_layout_ensure_creates_dir() {
        let dir = std::env::temp_dir().join("gllm_test_cache_layout");
        let layout = CacheLayout::new(dir.clone()).unwrap();
        layout.ensure().unwrap();
        assert!(dir.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── WeightFormat ──

    #[test]
    fn weight_format_variants() {
        assert_eq!(WeightFormat::SafeTensors, WeightFormat::SafeTensors);
        assert_ne!(WeightFormat::Gguf, WeightFormat::Onnx);
    }

    // ── LoaderError ──

    #[test]
    fn loader_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let err: LoaderError = io_err.into();
        assert!(matches!(err, LoaderError::Io(_)));
    }

    #[test]
    fn loader_error_from_json() {
        let json_err = serde_json::from_str::<serde_json::Value>("bad json");
        let err: LoaderError = json_err.unwrap_err().into();
        assert!(matches!(err, LoaderError::Json(_)));
    }

    #[test]
    fn loader_error_display() {
        let err = LoaderError::MissingWeights;
        assert!(err.to_string().contains("Missing weights"));
        let err = LoaderError::Network("timeout".into());
        assert!(err.to_string().contains("timeout"));
        let err = LoaderError::UnsupportedWeightExtension("xyz".into());
        assert!(err.to_string().contains("xyz"));
    }

    // ── QuantizedTensor ──

    #[test]
    fn quantized_tensor_metadata() {
        let qt = QuantizedTensor {
            data: vec![0u8; 72],
            quant_type: gllm_kernels::quant::QuantType::AWQ4,
            shape: vec![1, 128],
            ggml_dtype: crate::loader::gguf::GgmlDType::AWQ4,
        };
        assert_eq!(qt.data.len(), 72);
        assert_eq!(qt.shape, vec![1, 128]);
        assert_eq!(qt.quant_type, gllm_kernels::quant::QuantType::AWQ4);
    }

    #[test]
    fn quantized_tensor_clone() {
        let qt = QuantizedTensor {
            data: vec![1, 2, 3],
            quant_type: gllm_kernels::quant::QuantType::GPTQ4,
            shape: vec![2, 64],
            ggml_dtype: crate::loader::gguf::GgmlDType::GPTQ4,
        };
        let cloned = qt.clone();
        assert_eq!(cloned.data, qt.data);
        assert_eq!(cloned.shape, qt.shape);
    }

    // ── RawFloatTensor ──

    #[test]
    fn raw_float_tensor_metadata() {
        let rft = RawFloatTensor {
            data: vec![0u8; 256],
            dtype: Dtype::BF16,
            shape: vec![64, 32],
        };
        assert_eq!(rft.data.len(), 256);
        assert_eq!(rft.dtype, Dtype::BF16);
        assert_eq!(rft.shape, vec![64, 32]);
    }

    // ── build_tensor_role_index ──

    #[test]
    fn build_role_index_llama_layer() {
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "lm_head.weight",
        ];
        let (role_index, _) = build_tensor_role_index(names.into_iter());
        assert!(role_index.contains_key(&(TensorRole::Embedding, None)));
        assert!(role_index.contains_key(&(TensorRole::AttentionQuery, Some(0))));
        assert!(role_index.contains_key(&(TensorRole::AttentionKey, Some(0))));
        assert!(role_index.contains_key(&(TensorRole::OutputHead, None)));
    }

    #[test]
    fn build_role_index_bias_detection() {
        let names = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
        ];
        let (role_index, bias_index) = build_tensor_role_index(names.into_iter());
        assert!(role_index.contains_key(&(TensorRole::AttentionQuery, Some(0))));
        let weight_name = "model.layers.0.self_attn.q_proj.weight";
        assert!(bias_index.contains_key(weight_name));
        assert_eq!(bias_index.get(weight_name).unwrap(), "model.layers.0.self_attn.q_proj.bias");
    }

    #[test]
    fn build_role_index_empty() {
        let (role_index, bias_index) = build_tensor_role_index(std::iter::empty::<&str>());
        assert!(role_index.is_empty());
        assert!(bias_index.is_empty());
    }

    // ── extract_layer_index ──

    #[test]
    fn extract_layer_valid() {
        assert_eq!(extract_layer_index("model.layers.3.foo"), Some(3));
        assert_eq!(extract_layer_index("blk.7.bar"), Some(7));
        assert_eq!(extract_layer_index("blocks.0.weight"), Some(0));
    }

    #[test]
    fn extract_layer_none() {
        assert_eq!(extract_layer_index("model.embed_tokens.weight"), None);
        assert_eq!(extract_layer_index("lm_head.weight"), None);
    }

    // ── should_skip_tensor ──

    #[test]
    fn skip_vision_tower() {
        assert!(should_skip_tensor("model.vision_tower.patch_embed.weight"));
    }

    #[test]
    fn skip_audio_tower() {
        assert!(should_skip_tensor("model.audio_tower.conv.weight"));
    }

    #[test]
    fn skip_per_layer_embedding() {
        assert!(should_skip_tensor("model.embed_tokens_per_layer.weight"));
    }

    #[test]
    fn no_skip_regular_tensor() {
        assert!(!should_skip_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(!should_skip_tensor("lm_head.weight"));
    }

    // ── tensor_load_priority ──

    #[test]
    fn priority_embedding_highest() {
        let p = tensor_load_priority("model.embed_tokens.weight");
        assert_eq!(p, 1000);
    }

    #[test]
    fn priority_lm_head() {
        let p = tensor_load_priority("lm_head.weight");
        assert_eq!(p, 999);
    }

    #[test]
    fn priority_layer_back_to_front() {
        let p0 = tensor_load_priority("model.layers.0.self_attn.q_proj.weight");
        let p1 = tensor_load_priority("model.layers.1.self_attn.q_proj.weight");
        assert!(p0 > p1, "layer 0 should have higher priority than layer 1");
    }

    #[test]
    fn priority_unknown_defaults() {
        let p = tensor_load_priority("some_unknown_tensor");
        assert_eq!(p, 500);
    }

    // ── TensorMeta ──

    #[test]
    fn tensor_meta_fields() {
        let meta = TensorMeta {
            name: "test.weight".into(),
            shape: vec![128, 64],
            dtype: Dtype::F32,
        };
        assert_eq!(meta.name, "test.weight");
        assert_eq!(meta.shape, vec![128, 64]);
    }

    // ── AWQ constants ──

    #[test]
    fn awq_block_constants() {
        // AWQ_BLOCK_BYTES = 72, AWQ_GROUP_SIZE = 128 (private, verify via repack output)
        let n = 1usize;
        let k = 128usize;
        let qw_bytes = vec![0u8; n * (k / 8) * 4];
        let scales_bytes = vec![0u8; n * 2]; // 1 f16 per row
        let zeros_bytes = vec![0u8; 4]; // 1 u32
        let packed = repack_awq_gptq_blocks(
            &qw_bytes, &scales_bytes, &zeros_bytes,
            None, gllm_kernels::quant::QuantType::AWQ4, n, k,
        );
        assert_eq!(packed.len(), 72, "AWQ block should be 72 bytes");
    }

    // ══════════════════════════════════════════════════════════════════════
    // ~40 new tests: public types, Loader construction, pure functions
    // ══════════════════════════════════════════════════════════════════════

    // ── Loader construction (no files needed) ──

    #[test]
    fn loader_new_default_format_is_safetensors() {
        let loader = Loader::new(ModelManifest::default());
        assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);
    }

    #[test]
    fn loader_new_has_empty_weight_paths() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.weight_paths().is_empty());
    }

    #[test]
    fn loader_new_config_path_is_none() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.config_path().is_none());
    }

    #[test]
    fn loader_new_tokenizer_path_is_none() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.tokenizer_path().is_none());
    }

    #[test]
    fn loader_new_compute_dtype_is_none() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.compute_dtype().is_none());
    }

    #[test]
    fn loader_from_env_succeeds() {
        let loader = Loader::from_env().expect("from_env should succeed with default manifest");
        assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);
    }

    #[test]
    fn loader_from_env_with_manifest_succeeds() {
        let manifest = ModelManifest::default();
        let loader = Loader::from_env_with_manifest(manifest).expect("from_env_with_manifest should succeed");
        assert!(loader.weight_paths().is_empty());
    }

    // ── Loader builder methods (consume self, return new Loader) ──

    #[test]
    fn loader_with_weights_detects_gguf() {
        let loader = Loader::new(ModelManifest::default())
            .with_weights(vec![PathBuf::from("model.gguf")]);
        assert_eq!(loader.weight_format(), WeightFormat::Gguf);
        assert_eq!(loader.weight_paths().len(), 1);
    }

    #[test]
    fn loader_with_weights_detects_onnx() {
        let loader = Loader::new(ModelManifest::default())
            .with_weights(vec![PathBuf::from("model.onnx")]);
        assert_eq!(loader.weight_format(), WeightFormat::Onnx);
    }

    #[test]
    fn loader_with_weights_detects_pytorch_pt() {
        let loader = Loader::new(ModelManifest::default())
            .with_weights(vec![PathBuf::from("pytorch_model.bin")]);
        assert_eq!(loader.weight_format(), WeightFormat::PyTorch);
    }

    #[test]
    fn loader_with_weights_detects_pytorch_bin() {
        let loader = Loader::new(ModelManifest::default())
            .with_weights(vec![PathBuf::from("model.pth")]);
        assert_eq!(loader.weight_format(), WeightFormat::PyTorch);
    }

    #[test]
    fn loader_with_weights_defaults_safetensors_for_unknown() {
        let loader = Loader::new(ModelManifest::default())
            .with_weights(vec![PathBuf::from("model.safetensors")]);
        assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);
    }

    #[test]
    fn loader_with_config_sets_path() {
        let path = PathBuf::from("/tmp/config.json");
        let loader = Loader::new(ModelManifest::default()).with_config(path.clone());
        assert_eq!(loader.config_path(), Some(path.as_path()));
    }

    #[test]
    fn loader_with_tokenizer_sets_path() {
        let path = PathBuf::from("/tmp/tokenizer.json");
        let loader = Loader::new(ModelManifest::default()).with_tokenizer(path.clone());
        assert_eq!(loader.tokenizer_path(), Some(path.as_path()));
    }

    #[test]
    fn loader_with_compute_dtype_overrides() {
        use gllm_kernels::types::DType as KDType;
        let loader = Loader::new(ModelManifest::default())
            .with_compute_dtype(KDType::BF16);
        assert_eq!(loader.compute_dtype(), Some(KDType::BF16));
    }

    // ── Loader internal-accessor error paths (no data loaded) ──

    #[test]
    fn loader_safetensors_loader_err_without_load() {
        let mut loader = Loader::new(ModelManifest::default());
        let err = loader.safetensors_loader().unwrap_err();
        assert!(matches!(err, LoaderError::MissingWeights));
    }

    #[test]
    fn loader_gguf_reader_err_without_load() {
        let mut loader = Loader::new(ModelManifest::default());
        let err = loader.gguf_reader().unwrap_err();
        assert!(matches!(err, LoaderError::MissingWeights));
    }

    #[test]
    fn loader_onnx_loader_err_without_load() {
        let mut loader = Loader::new(ModelManifest::default());
        let err = loader.onnx_loader().unwrap_err();
        assert!(matches!(err, LoaderError::MissingWeights));
    }

    #[test]
    fn loader_gllm_reader_err_without_load() {
        let mut loader = Loader::new(ModelManifest::default());
        let err = loader.gllm_reader().unwrap_err();
        assert!(matches!(err, LoaderError::MissingWeights));
    }

    #[test]
    fn loader_safetensors_ref_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.safetensors_ref().is_none());
    }

    #[test]
    fn loader_gguf_ref_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.gguf_ref().is_none());
    }

    #[test]
    fn loader_onnx_ref_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.onnx_ref().is_none());
    }

    #[test]
    fn loader_gllm_ref_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.gllm_ref().is_none());
    }

    #[test]
    fn loader_safetensors_gllm_config_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.safetensors_gllm_config().unwrap().is_none());
    }

    #[test]
    fn loader_detect_weight_dtype_none_without_load() {
        let loader = Loader::new(ModelManifest::default());
        assert!(loader.detect_weight_dtype().unwrap().is_none());
    }

    // ── Loader detect_architecture fallback to manifest ──

    #[test]
    fn loader_detect_architecture_falls_back_to_manifest() {
        let mut manifest = ModelManifest::default();
        manifest.arch = "qwen3".to_string();
        let loader = Loader::new(manifest);
        // No weights loaded, no config → falls back to manifest.arch
        let arch = loader.detect_architecture();
        assert_eq!(arch, "qwen3");
    }

    #[test]
    fn loader_set_manifest_if_missing_updates() {
        let mut loader = Loader::new(ModelManifest::default());
        let mut new_manifest = ModelManifest::default();
        new_manifest.arch = "mistral3".to_string();
        loader.set_manifest_if_missing(&new_manifest);
        let arch = loader.detect_architecture();
        assert_eq!(arch, "mistral3");
    }

    // ── TensorSlice ──

    #[test]
    fn tensor_slice_new_preserves_fields() {
        let data = &[1u8, 2, 3, 4];
        let slice = TensorSlice::new(Dtype::F32, vec![1], data);
        assert_eq!(slice.dtype, Dtype::F32);
        assert_eq!(slice.shape, vec![1]);
        assert_eq!(slice.data, data);
    }

    #[test]
    fn tensor_slice_empty_data() {
        let data = &[];
        let slice = TensorSlice::new(Dtype::F32, vec![0], data);
        assert!(slice.data.is_empty());
    }

    // ── CompanionConfig ──

    #[test]
    fn companion_config_equality() {
        let a = CompanionConfig {
            scales: "scales".to_string(),
            zeros: Some("zeros".to_string()),
        };
        let b = CompanionConfig {
            scales: "scales".to_string(),
            zeros: Some("zeros".to_string()),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn companion_config_serde_roundtrip() {
        let original = CompanionConfig {
            scales: "model.scales".to_string(),
            zeros: None,
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: CompanionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    // ── QuantizationMetadata ──

    #[test]
    fn quantization_metadata_from_metadata_missing_key_returns_none() {
        let meta: HashMap<String, String> = HashMap::new();
        let result = QuantizationMetadata::from_metadata(&meta).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn quantization_metadata_from_metadata_valid_json() {
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert(
            "gllm.quantization".to_string(),
            r#"{"fc1": {"block_size": 128, "bits": 4}}"#.to_string(),
        );
        let result = QuantizationMetadata::from_metadata(&meta).unwrap().unwrap();
        let entry = result.get("fc1").unwrap();
        assert_eq!(entry.block_size, 128);
        assert_eq!(entry.bits, 4);
        assert!(!entry.desc_act);
        assert!(!entry.is_sym);
        assert!(!entry.signed);
    }

    #[test]
    fn quantization_metadata_serde_roundtrip() {
        let original = QuantizationMetadata {
            block_size: 64,
            bits: 8,
            desc_act: true,
            is_sym: false,
            signed: true,
            companions: Some(CompanionConfig {
                scales: "s.weight".to_string(),
                zeros: None,
            }),
        };
        let json = serde_json::to_string(&original).unwrap();
        let restored: QuantizationMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(original, restored);
    }

    // ── ThinkingHead ──

    #[test]
    fn thinking_head_default_empty_tensors() {
        let th = ThinkingHead::default();
        assert!(th.tensors.is_empty());
    }

    #[test]
    fn thinking_head_clone_preserves() {
        let th = ThinkingHead {
            tensors: vec!["think.weight".to_string()],
        };
        let cloned = th.clone();
        assert_eq!(cloned.tensors, th.tensors);
    }

    // ── ParallelPolicy ──

    #[test]
    fn parallel_policy_default_enabled() {
        let policy = ParallelPolicy::default();
        assert!(policy.enabled);
    }

    #[test]
    fn parallel_policy_clone_preserves_state() {
        let policy = ParallelPolicy { enabled: false };
        let cloned = policy.clone();
        assert_eq!(cloned.enabled, false);
    }

    // ── UploadedTensor ──

    #[test]
    fn uploaded_tensor_fields() {
        let ut = UploadedTensor {
            name: "q_proj".to_string(),
            shape: vec![4096, 4096],
        };
        assert_eq!(ut.name, "q_proj");
        assert_eq!(ut.shape, vec![4096, 4096]);
    }

    // ── is_linear_weight (private, accessible via include!) ──

    #[test]
    fn is_linear_weight_accepts_2d_proj() {
        assert!(is_linear_weight("model.layers.0.self_attn.q_proj.weight", &[4096, 4096]));
    }

    #[test]
    fn is_linear_weight_rejects_embedding() {
        assert!(!is_linear_weight("model.embed_tokens.weight", &[32000, 4096]));
    }

    #[test]
    fn is_linear_weight_rejects_1d_shape() {
        assert!(!is_linear_weight("model.layers.0.self_attn.q_proj.weight", &[4096]));
    }

    #[test]
    fn is_linear_weight_rejects_non_weight_suffix() {
        assert!(!is_linear_weight("model.layers.0.self_attn.q_proj.bias", &[4096, 4096]));
    }

    #[test]
    fn is_linear_weight_rejects_token_embd() {
        assert!(!is_linear_weight("token_embd.weight", &[32000, 4096]));
    }

    // ── parallel_f64_to_f32 (private) ──

    #[test]
    fn parallel_f64_to_f32_matches_scalar() {
        let n = 1000usize;
        let src: Vec<f64> = (0..n).map(|i| (i as f64) * 0.123 - 50.0).collect();
        let mut bytes = Vec::with_capacity(n * 8);
        for v in &src {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let out = parallel_f64_to_f32(&bytes).expect("f64→f32 conversion failed");
        assert_eq!(out.len(), n);
        for (i, (got, expected)) in out.iter().zip(src.iter()).enumerate() {
            assert!(
                (got - (*expected as f32)).abs() < 1e-6,
                "mismatch at index {i}: got {got}, expected {}",
                *expected as f32,
            );
        }
    }

    #[test]
    fn parallel_f64_to_f32_rejects_non_multiple() {
        let bytes = vec![0u8; 7]; // not a multiple of 8
        let result = parallel_f64_to_f32(&bytes);
        assert!(result.is_err());
    }

    // ── cache_blocked_transpose_bytes (private) ──

    #[test]
    fn cache_blocked_transpose_bytes_matches_naive() {
        // BF16-like: 2 bytes per element, small matrix
        let rows = 4usize;
        let cols = 8usize;
        let elem_size = 2usize;
        let total = rows * cols * elem_size;
        let src: Vec<u8> = (0..total as u8).collect();
        let mut dst = vec![0u8; total];
        cache_blocked_transpose_bytes(&src, &mut dst, rows, cols, elem_size);

        // Naive byte-level transpose check: element at (r,c) moves to (c,r)
        for r in 0..rows {
            for c in 0..cols {
                let src_off = (r * cols + c) * elem_size;
                let dst_off = (c * rows + r) * elem_size;
                assert_eq!(
                    &dst[dst_off..dst_off + elem_size],
                    &src[src_off..src_off + elem_size],
                    "byte-transpose mismatch at ({r},{c})"
                );
            }
        }
    }

    // ── Additional match_tensor_role coverage ──

    #[test]
    fn match_gguf_blk_wk() {
        let (role, layer) = match_tensor_role("blk.0.attn_k.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionKey);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_gguf_blk_wv() {
        let (role, layer) = match_tensor_role("blk.3.attn_v.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionValue);
        assert_eq!(layer, Some(3));
    }

    #[test]
    fn match_gguf_blk_wo() {
        let (role, layer) = match_tensor_role("blk.1.attn_output.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionOutput);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_mla_q_expand() {
        let (role, layer) = match_tensor_role("model.layers.2.self_attn.q_b_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaQExpand);
        assert_eq!(layer, Some(2));
    }

    #[test]
    fn match_mla_key_absorb() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.k_b_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaKeyAbsorb);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_mla_value_absorb() {
        let (role, layer) = match_tensor_role("model.layers.4.self_attn.v_b_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MlaValueAbsorb);
        assert_eq!(layer, Some(4));
    }

    #[test]
    fn match_depthwise_conv() {
        let (role, layer) = match_tensor_role("model.layers.1.conv_module.depthwise_conv.weight").unwrap();
        assert_eq!(role, TensorRole::DepthwiseConv);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_gguf_ffn_down() {
        let (role, layer) = match_tensor_role("blk.0.ffn_down.weight").unwrap();
        assert_eq!(role, TensorRole::FfnDown);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_gguf_ffn_up() {
        let (role, layer) = match_tensor_role("blk.0.ffn_up.weight").unwrap();
        assert_eq!(role, TensorRole::FfnUp);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_moe_shared_expert_gate_proj() {
        let (role, layer) = match_tensor_role("model.layers.0.mlp.shared_experts.gate_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MoESharedExpert);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_moe_shared_expert_down_proj() {
        let (role, layer) = match_tensor_role("model.layers.1.mlp.shared_experts.down_proj.weight").unwrap();
        assert_eq!(role, TensorRole::MoESharedExpert);
        assert_eq!(layer, Some(1));
    }

    #[test]
    fn match_attention_sinks() {
        let (role, layer) = match_tensor_role("model.layers.0.self_attn.sinks.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionSinks);
        assert_eq!(layer, Some(0));
    }

    #[test]
    fn match_qk_norm_via_attn_prefix() {
        let (role, layer) = match_tensor_role("model.layers.0.attn_q_norm.weight").unwrap();
        assert_eq!(role, TensorRole::AttentionQNorm);
        assert_eq!(layer, Some(0));

        let (role2, layer2) = match_tensor_role("model.layers.0.attn_k_norm.weight").unwrap();
        assert_eq!(role2, TensorRole::AttentionKNorm);
        assert_eq!(layer2, Some(0));
    }

    // ── Additional tensor_load_priority coverage ──

    #[test]
    fn priority_token_embd_highest() {
        let p = tensor_load_priority("token_embd.weight");
        assert_eq!(p, 1000);
    }

    #[test]
    fn priority_word_embeddings() {
        let p = tensor_load_priority("model.word_embeddings.weight");
        assert_eq!(p, 1000);
    }

    #[test]
    fn priority_model_norm() {
        let p = tensor_load_priority("model.norm.weight");
        assert_eq!(p, 998);
    }

    // ── Additional should_skip_tensor coverage ──

    #[test]
    fn skip_embed_vision() {
        assert!(should_skip_tensor("model.embed_vision.weight"));
    }

    #[test]
    fn skip_embed_audio() {
        assert!(should_skip_tensor("model.embed_audio.weight"));
    }

    #[test]
    fn skip_per_layer_projection() {
        assert!(should_skip_tensor("model.per_layer_projection.weight"));
    }

    #[test]
    fn skip_post_mlp_projection() {
        assert!(should_skip_tensor("model.post_mlp_projection.weight"));
    }

    // ── Additional LoaderError variant coverage ──

    #[test]
    fn loader_error_from_safe_tensors() {
        let st_err = ::safetensors::SafeTensorError::InvalidHeader;
        let err: LoaderError = st_err.into();
        assert!(matches!(err, LoaderError::SafeTensors(_)));
    }

    #[test]
    fn non_recoverable_cache_error() {
        let err = LoaderError::Cache("disk full".into());
        assert!(!is_recoverable_error(&err));
    }

    #[test]
    fn non_recoverable_arch_detection() {
        let err = LoaderError::ArchDetection("unknown arch".into());
        assert!(!is_recoverable_error(&err));
    }

    #[test]
    fn recoverable_hf_hub_error() {
        let err = LoaderError::HfHub("connection refused".into());
        assert!(is_recoverable_error(&err));
    }

    #[test]
    fn loader_error_from_gguf_error() {
        let gguf_err = gguf::GgufError::InvalidMagic(0);
        let err: LoaderError = gguf_err.into();
        assert!(matches!(err, LoaderError::Gguf(_)));
    }

    // ── ChecksumPolicy exhaustive variants ──

    #[test]
    fn checksum_policy_all_variants_distinct() {
        let variants = [ChecksumPolicy::Ignore, ChecksumPolicy::Verify, ChecksumPolicy::Default];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // ── LoaderConfig::from_env ──

    #[test]
    fn loader_config_from_env_preserves_defaults_when_unset() {
        let config = LoaderConfig::from_env();
        assert_eq!(config.source, ModelSource::HuggingFace);
        assert!(config.enable_fallback);
        assert_eq!(config.checksum_policy, ChecksumPolicy::Ignore);
        assert!(config.gguf_file_filter.is_none());
        assert!(config.hf_token_path.is_none());
    }

    // ── Additional extract_layer_index coverage ──

    #[test]
    fn extract_layer_block_prefix() {
        assert_eq!(extract_layer_index("block.3.weight"), Some(3));
    }

    #[test]
    fn extract_layer_encoder_prefix() {
        assert_eq!(extract_layer_index("encoder.layer.5.attention.self.query.weight"), Some(5));
    }

    #[test]
    fn extract_layer_rejects_bare_number() {
        // "3.weight" — no recognized prefix before the number
        assert_eq!(extract_layer_index("3.weight"), None);
    }

    // ── build_tensor_role_index: standalone bias detection ──

    #[test]
    fn build_role_index_standalone_bias() {
        let names = [
            "roberta.encoder.layer.0.attention.self.query.weight",
            "roberta.encoder.layer.0.attention.self.query.bias",
        ];
        let (_, bias_index) = build_tensor_role_index(names.into_iter());
        let weight_name = "roberta.encoder.layer.0.attention.self.query.weight";
        assert!(bias_index.contains_key(weight_name));
        assert_eq!(
            bias_index.get(weight_name).unwrap(),
            "roberta.encoder.layer.0.attention.self.query.bias"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // TP 权重分片 — WeightsHandle::shard_for_tp (REQ-DIST-004, TEST-DIST-004)
    // nccl feature-gated
    // ═══════════════════════════════════════════════════════════════════════

    #[cfg(feature = "nccl")]
    mod shard_for_tp_tests {
        use super::*;
        use crate::compat::backend_trait::{Backend, Element, WeightPlacement};
        use crate::engine::distributed_config::ParallelConfig;
        use crate::loader::{QuantizedTensor, RawFloatTensor, WeightsHandle};

        fn make_config(tp_size: u32, rank: u32) -> ParallelConfig {
            ParallelConfig {
                tp_size,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank,
                world_size: tp_size,
                unique_id: String::new(),
            }
        }

        fn make_raw_float_weights_handle() -> WeightsHandle<crate::compat::cpu_backend::CpuBackend<f32>, f32> {
            // Build a WeightsHandle with BF16 raw_floats simulating a small model:
            // L0.q_proj: [4, 8] BF16 — column parallel
            // L0.o_proj: [8, 4] BF16 — row parallel
            // embed: [4, 8] BF16 — no shard (embedding)
            let mut raw_floats = std::collections::HashMap::new();

            // L0.q_proj: [4, 8] BF16 → 4*8*2 = 64 bytes
            let q_proj_data: Vec<u8> = (0..32)
                .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
                .collect();
            raw_floats.insert(
                "L0.q_proj.weight".to_string(),
                RawFloatTensor {
                    data: q_proj_data,
                    dtype: ::safetensors::Dtype::BF16,
                    shape: vec![4, 8],
                },
            );

            // L0.o_proj: [8, 4] BF16 → 8*4*2 = 64 bytes
            let o_proj_data: Vec<u8> = (0..32)
                .flat_map(|i| half::bf16::from_f32((i + 100) as f32).to_le_bytes())
                .collect();
            raw_floats.insert(
                "L0.o_proj.weight".to_string(),
                RawFloatTensor {
                    data: o_proj_data,
                    dtype: ::safetensors::Dtype::BF16,
                    shape: vec![8, 4],
                },
            );

            // embed: [4, 8] BF16 → no sharding
            let embed_data: Vec<u8> = (0..32)
                .flat_map(|i| half::bf16::from_f32((i + 200) as f32).to_le_bytes())
                .collect();
            raw_floats.insert(
                "model.embed_tokens.weight".to_string(),
                RawFloatTensor {
                    data: embed_data.clone(),
                    dtype: ::safetensors::Dtype::BF16,
                    shape: vec![4, 8],
                },
            );

            let mut shapes = std::collections::HashMap::new();
            shapes.insert("L0.q_proj.weight".to_string(), vec![4, 8]);
            shapes.insert("L0.o_proj.weight".to_string(), vec![8, 4]);
            shapes.insert("model.embed_tokens.weight".to_string(), vec![4, 8]);

            let mut meta = std::collections::HashMap::new();
            for name in ["L0.q_proj.weight", "L0.o_proj.weight", "model.embed_tokens.weight"] {
                meta.insert(
                    name.to_string(),
                    crate::loader::TensorMeta {
                        name: name.to_string(),
                        shape: shapes[name].clone(),
                        dtype: ::safetensors::Dtype::BF16,
                    },
                );
            }

            WeightsHandle::new_with_quantized_and_sparse(
                std::collections::HashMap::new(),
                shapes,
                meta,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            )
        }

        // @trace TEST-DIST-004 [req:REQ-DIST-004] [level:unit]
        #[test]
        fn shard_for_tp_single_node_is_noop() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(1, 0);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());
            // Shape unchanged
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[4, 8][..]));
            assert_eq!(handle.tensor_shape("L0.o_proj.weight"), Some(&[8, 4][..]));
        }

        #[test]
        fn shard_for_tp_tp0_is_noop() {
            let mut handle = make_raw_float_weights_handle();
            let config = ParallelConfig {
                tp_size: 0,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 0,
                unique_id: String::new(),
            };

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());
        }

        #[test]
        fn shard_for_tp_column_parallel_halves_cols() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(2, 0);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // L0.q_proj: ColumnParallel → [4, 8/2] = [4, 4]
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[4, 4][..]));

            // Data size should also be halved: 4*4*2 = 32 bytes
            let q_proj = handle.raw_float_tensor("L0.q_proj.weight").unwrap();
            assert_eq!(q_proj.data.len(), 32);
        }

        #[test]
        fn shard_for_tp_column_parallel_rank1_gets_right_half() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(2, 1);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // L0.q_proj: ColumnParallel rank 1 → [4, 4]
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[4, 4][..]));
        }

        #[test]
        fn shard_for_tp_row_parallel_halves_rows() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(2, 0);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // L0.o_proj: RowParallel → [8/2, 4] = [4, 4]
            assert_eq!(handle.tensor_shape("L0.o_proj.weight"), Some(&[4, 4][..]));

            // Data size should also be halved: 4*4*2 = 32 bytes
            let o_proj = handle.raw_float_tensor("L0.o_proj.weight").unwrap();
            assert_eq!(o_proj.data.len(), 32);
        }

        #[test]
        fn shard_for_tp_embedding_not_sharded() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(2, 0);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // embed_tokens: not sharded → shape unchanged
            assert_eq!(
                handle.tensor_shape("model.embed_tokens.weight"),
                Some(&[4, 8][..])
            );
        }

        #[test]
        fn shard_for_tp_meta_shapes_updated() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(2, 0);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // Verify that both `shapes` HashMap and `meta` HashMap are updated
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[4, 4][..]));
            let meta = handle.meta.get("L0.q_proj.weight").unwrap();
            assert_eq!(meta.shape, vec![4, 4]);
        }

        #[test]
        fn shard_for_tp_not_divisible_returns_err() {
            let mut raw_floats = std::collections::HashMap::new();
            // [3, 5] BF16 — 3 rows, 5 cols → neither 3 nor 5 is divisible by 2
            let data: Vec<u8> = (0..15)
                .flat_map(|i| half::bf16::from_f32(i as f32).to_le_bytes())
                .collect();
            raw_floats.insert(
                "L0.q_proj.weight".to_string(),
                RawFloatTensor {
                    data,
                    dtype: ::safetensors::Dtype::BF16,
                    shape: vec![3, 5],
                },
            );

            let mut shapes = std::collections::HashMap::new();
            shapes.insert("L0.q_proj.weight".to_string(), vec![3, 5]);

            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "L0.q_proj.weight".to_string(),
                crate::loader::TensorMeta {
                    name: "L0.q_proj.weight".to_string(),
                    shape: vec![3, 5],
                    dtype: ::safetensors::Dtype::BF16,
                },
            );

            let mut handle: WeightsHandle<crate::compat::cpu_backend::CpuBackend<f32>, f32> = WeightsHandle::new_with_quantized_and_sparse(
                std::collections::HashMap::new(),
                shapes,
                meta,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            );

            let config = make_config(2, 0);
            let result = handle.shard_for_tp(&config);
            assert!(result.is_err());
        }

        #[test]
        fn shard_for_tp_quantized_weight_needs_shard_returns_err() {
            let mut quantized = std::collections::HashMap::new();
            quantized.insert(
                "L0.q_proj.weight".to_string(),
                QuantizedTensor {
                    data: vec![0u8; 64],
                    quant_type: gllm_kernels::quant::QuantType::Q4_0,
                    shape: vec![4, 8],
                    ggml_dtype: crate::loader::gguf::GgmlDType::Q4_0,
                },
            );

            let mut shapes = std::collections::HashMap::new();
            shapes.insert("L0.q_proj.weight".to_string(), vec![4, 8]);

            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "L0.q_proj.weight".to_string(),
                crate::loader::TensorMeta {
                    name: "L0.q_proj.weight".to_string(),
                    shape: vec![4, 8],
                    dtype: ::safetensors::Dtype::F32,
                },
            );

            let mut handle: WeightsHandle<crate::compat::cpu_backend::CpuBackend<f32>, f32> = WeightsHandle::new_with_quantized_and_sparse(
                std::collections::HashMap::new(),
                shapes,
                meta,
                quantized,
                std::collections::HashMap::new(),
            );

            let config = make_config(2, 0);
            let result = handle.shard_for_tp(&config);
            assert!(result.is_err());
        }

        #[test]
        fn shard_for_tp_f16_column_parallel() {
            // Test F16 path
            let mut raw_floats = std::collections::HashMap::new();
            let data: Vec<u8> = (0..16)
                .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
                .collect();
            raw_floats.insert(
                "L0.q_proj.weight".to_string(),
                RawFloatTensor {
                    data,
                    dtype: ::safetensors::Dtype::F16,
                    shape: vec![2, 8],
                },
            );

            let mut shapes = std::collections::HashMap::new();
            shapes.insert("L0.q_proj.weight".to_string(), vec![2, 8]);

            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "L0.q_proj.weight".to_string(),
                crate::loader::TensorMeta {
                    name: "L0.q_proj.weight".to_string(),
                    shape: vec![2, 8],
                    dtype: ::safetensors::Dtype::F16,
                },
            );

            let mut handle: WeightsHandle<crate::compat::cpu_backend::CpuBackend<f32>, f32> = WeightsHandle::new_with_quantized_and_sparse(
                std::collections::HashMap::new(),
                shapes,
                meta,
                std::collections::HashMap::new(),
                std::collections::HashMap::new(),
            );

            let config = make_config(2, 0);
            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());
            // ColumnParallel: [2, 8] → [2, 4]
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[2, 4][..]));
            let q_proj = handle.raw_float_tensor("L0.q_proj.weight").unwrap();
            assert_eq!(q_proj.dtype, ::safetensors::Dtype::F16);
            assert_eq!(q_proj.data.len(), 2 * 4 * 2); // 2*4 elements * 2 bytes each
        }

        #[test]
        fn shard_for_tp_tp4_rank2_column_parallel() {
            let mut handle = make_raw_float_weights_handle();
            let config = make_config(4, 2);

            let result = handle.shard_for_tp(&config);
            assert!(result.is_ok());

            // L0.q_proj: ColumnParallel → [4, 8/4] = [4, 2]
            assert_eq!(handle.tensor_shape("L0.q_proj.weight"), Some(&[4, 2][..]));

            // L0.o_proj: RowParallel → [8/4, 4] = [2, 4]
            assert_eq!(handle.tensor_shape("L0.o_proj.weight"), Some(&[2, 4][..]));
        }
    }
}
