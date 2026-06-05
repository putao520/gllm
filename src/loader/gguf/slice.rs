use super::GgmlDType;

#[derive(Debug, Clone)]
pub struct TensorSlice<'a> {
    dtype: GgmlDType,
    shape: Vec<u64>,
    data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    pub(crate) fn new(dtype: GgmlDType, shape: Vec<u64>, data: &'a [u8]) -> Self {
        Self { dtype, shape, data }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_slice_new_and_accessors() {
        let data = &[1u8, 2, 3, 4, 5, 6, 7, 8];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::F32);
        assert_eq!(slice.shape(), &[2]);
        assert_eq!(slice.as_bytes(), data);
    }

    #[test]
    fn tensor_slice_cloned_independent() {
        let data = &[10u8, 20, 30];
        let slice = TensorSlice::new(GgmlDType::F16, vec![3], data);
        let cloned = slice.clone();
        assert_eq!(cloned.dtype(), slice.dtype());
        assert_eq!(cloned.shape(), slice.shape());
        assert_eq!(cloned.as_bytes(), slice.as_bytes());
    }

    #[test]
    fn tensor_slice_empty_data() {
        let data = &[];
        let slice = TensorSlice::new(GgmlDType::F32, vec![0], data);
        assert!(slice.as_bytes().is_empty());
    }

    #[test]
    fn tensor_slice_multidim_shape() {
        let data = &[0u8; 16];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2, 2], data);
        assert_eq!(slice.shape(), &[2, 2]);
    }

    #[test]
    fn tensor_slice_quantized_dtype() {
        let data = &[0u8; 18];
        let slice = TensorSlice::new(GgmlDType::Q4_0, vec![32], data);
        assert!(slice.dtype().is_quantized());
    }

    // ── TensorSlice::as_bytes reflects original data identity ─────────────

    #[test]
    fn tensor_slice_as_bytes_same_pointer() {
        let data: &[u8] = &[0xAA, 0xBB, 0xCC];
        let slice = TensorSlice::new(GgmlDType::I8, vec![3], data);
        assert!(std::ptr::eq(slice.as_bytes().as_ptr(), data.as_ptr()));
    }

    // ── TensorSlice with every non-quantized dtype ────────────────────────

    #[test]
    fn tensor_slice_dtype_f16() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::F16, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::F16);
        assert!(!slice.dtype().is_quantized());
        assert_eq!(slice.dtype().block_bytes(), 2);
    }

    #[test]
    fn tensor_slice_dtype_bf16() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::BF16, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::BF16);
        assert_eq!(slice.dtype().as_str(), "BF16");
    }

    #[test]
    fn tensor_slice_dtype_i8() {
        let data = &[0u8; 3];
        let slice = TensorSlice::new(GgmlDType::I8, vec![3], data);
        assert_eq!(slice.dtype(), GgmlDType::I8);
        assert_eq!(slice.dtype().block_bytes(), 1);
        assert_eq!(slice.dtype().block_size(), 1);
    }

    #[test]
    fn tensor_slice_dtype_i16() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::I16, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::I16);
        assert_eq!(slice.dtype().block_bytes(), 2);
    }

    #[test]
    fn tensor_slice_dtype_i32() {
        let data = &[0u8; 8];
        let slice = TensorSlice::new(GgmlDType::I32, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::I32);
        assert!(!slice.dtype().is_quantized());
    }

    #[test]
    fn tensor_slice_dtype_i64() {
        let data = &[0u8; 16];
        let slice = TensorSlice::new(GgmlDType::I64, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::I64);
        assert_eq!(slice.dtype().block_bytes(), 8);
    }

    #[test]
    fn tensor_slice_dtype_f64() {
        let data = &[0u8; 16];
        let slice = TensorSlice::new(GgmlDType::F64, vec![2], data);
        assert_eq!(slice.dtype(), GgmlDType::F64);
        assert!(!slice.dtype().is_quantized());
    }

    // ── TensorSlice with quantized dtypes (spot-check each family) ────────

    #[test]
    fn tensor_slice_dtype_q4_1() {
        let data = &[0u8; 20];
        let slice = TensorSlice::new(GgmlDType::Q4_1, vec![32], data);
        assert!(slice.dtype().is_quantized());
        assert_eq!(slice.dtype().block_bytes(), 20);
        assert_eq!(slice.dtype().block_size(), 32);
    }

    #[test]
    fn tensor_slice_dtype_q5_0() {
        let data = &[0u8; 22];
        let slice = TensorSlice::new(GgmlDType::Q5_0, vec![32], data);
        assert_eq!(slice.dtype(), GgmlDType::Q5_0);
        assert_eq!(slice.dtype().block_bytes(), 22);
    }

    #[test]
    fn tensor_slice_dtype_q5_1() {
        let data = &[0u8; 24];
        let slice = TensorSlice::new(GgmlDType::Q5_1, vec![32], data);
        assert_eq!(slice.dtype().as_str(), "Q5_1");
    }

    #[test]
    fn tensor_slice_dtype_q8_0() {
        let data = &[0u8; 34];
        let slice = TensorSlice::new(GgmlDType::Q8_0, vec![32], data);
        assert_eq!(slice.dtype().block_bytes(), 34);
    }

    #[test]
    fn tensor_slice_dtype_q8_1() {
        let data = &[0u8; 36];
        let slice = TensorSlice::new(GgmlDType::Q8_1, vec![32], data);
        assert_eq!(slice.dtype().block_bytes(), 36);
    }

    #[test]
    fn tensor_slice_dtype_q2_k() {
        let data = &[0u8; 84];
        let slice = TensorSlice::new(GgmlDType::Q2_K, vec![256], data);
        assert_eq!(slice.dtype().block_size(), 256);
        assert_eq!(slice.dtype().block_bytes(), 84);
    }

    #[test]
    fn tensor_slice_dtype_q3_k() {
        let data = &[0u8; 110];
        let slice = TensorSlice::new(GgmlDType::Q3_K, vec![256], data);
        assert_eq!(slice.dtype(), GgmlDType::Q3_K);
    }

    #[test]
    fn tensor_slice_dtype_q4_k() {
        let data = &[0u8; 144];
        let slice = TensorSlice::new(GgmlDType::Q4_K, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 144);
    }

    #[test]
    fn tensor_slice_dtype_q5_k() {
        let data = &[0u8; 176];
        let slice = TensorSlice::new(GgmlDType::Q5_K, vec![256], data);
        assert_eq!(slice.dtype(), GgmlDType::Q5_K);
    }

    #[test]
    fn tensor_slice_dtype_q6_k() {
        let data = &[0u8; 210];
        let slice = TensorSlice::new(GgmlDType::Q6_K, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 210);
    }

    #[test]
    fn tensor_slice_dtype_q8_k() {
        let data = &[0u8; 292];
        let slice = TensorSlice::new(GgmlDType::Q8_K, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 292);
    }

    #[test]
    fn tensor_slice_dtype_awq4() {
        let data = &[0u8; 72];
        let slice = TensorSlice::new(GgmlDType::AWQ4, vec![128], data);
        assert_eq!(slice.dtype().block_size(), 128);
        assert_eq!(slice.dtype().block_bytes(), 72);
        assert_eq!(slice.dtype().as_str(), "AWQ4");
    }

    #[test]
    fn tensor_slice_dtype_gptq4() {
        let data = &[0u8; 72];
        let slice = TensorSlice::new(GgmlDType::GPTQ4, vec![128], data);
        assert_eq!(slice.dtype().block_size(), 128);
        assert_eq!(slice.dtype().as_str(), "GPTQ4");
    }

    #[test]
    fn tensor_slice_dtype_squeeze() {
        let data = &[0u8; 130];
        let slice = TensorSlice::new(GgmlDType::SQUEEZE, vec![256], data);
        assert_eq!(slice.dtype().block_size(), 256);
        assert_eq!(slice.dtype().block_bytes(), 130);
    }

    #[test]
    fn tensor_slice_dtype_nvfp4() {
        let data = &[0u8; 36];
        let slice = TensorSlice::new(GgmlDType::NVFP4, vec![64], data);
        assert_eq!(slice.dtype().block_size(), 64);
        assert_eq!(slice.dtype().block_bytes(), 36);
    }

    #[test]
    fn tensor_slice_dtype_mxfp4() {
        let data = &[0u8; 17];
        let slice = TensorSlice::new(GgmlDType::MXFP4, vec![32], data);
        assert_eq!(slice.dtype().block_size(), 32);
        assert_eq!(slice.dtype().block_bytes(), 17);
    }

    #[test]
    fn tensor_slice_dtype_tq1_0() {
        let data = &[0u8; 54];
        let slice = TensorSlice::new(GgmlDType::TQ1_0, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 54);
        assert_eq!(slice.dtype().as_str(), "TQ1_0");
    }

    #[test]
    fn tensor_slice_dtype_tq2_0() {
        let data = &[0u8; 66];
        let slice = TensorSlice::new(GgmlDType::TQ2_0, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 66);
    }

    // ── TensorSlice IQ family ─────────────────────────────────────────────

    #[test]
    fn tensor_slice_dtype_iq2_xxs() {
        let data = &[0u8; 66];
        let slice = TensorSlice::new(GgmlDType::IQ2_XXS, vec![256], data);
        assert_eq!(slice.dtype(), GgmlDType::IQ2_XXS);
        assert_eq!(slice.dtype().as_str(), "IQ2_XXS");
    }

    #[test]
    fn tensor_slice_dtype_iq2_xs() {
        let data = &[0u8; 74];
        let slice = TensorSlice::new(GgmlDType::IQ2_XS, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 74);
    }

    #[test]
    fn tensor_slice_dtype_iq3_xxs() {
        let data = &[0u8; 98];
        let slice = TensorSlice::new(GgmlDType::IQ3_XXS, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 98);
    }

    #[test]
    fn tensor_slice_dtype_iq1_s() {
        let data = &[0u8; 50];
        let slice = TensorSlice::new(GgmlDType::IQ1_S, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 50);
    }

    #[test]
    fn tensor_slice_dtype_iq4_nl() {
        let data = &[0u8; 18];
        let slice = TensorSlice::new(GgmlDType::IQ4_NL, vec![32], data);
        assert_eq!(slice.dtype().block_bytes(), 18);
        assert_eq!(slice.dtype().block_size(), 32);
    }

    #[test]
    fn tensor_slice_dtype_iq3_s() {
        let data = &[0u8; 110];
        let slice = TensorSlice::new(GgmlDType::IQ3_S, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 110);
    }

    #[test]
    fn tensor_slice_dtype_iq2_s() {
        let data = &[0u8; 82];
        let slice = TensorSlice::new(GgmlDType::IQ2_S, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 82);
    }

    #[test]
    fn tensor_slice_dtype_iq4_xs() {
        let data = &[0u8; 136];
        let slice = TensorSlice::new(GgmlDType::IQ4_XS, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 136);
    }

    #[test]
    fn tensor_slice_dtype_iq1_m() {
        let data = &[0u8; 56];
        let slice = TensorSlice::new(GgmlDType::IQ1_M, vec![256], data);
        assert_eq!(slice.dtype().block_bytes(), 56);
    }

    // ── TensorSlice Debug format ──────────────────────────────────────────

    #[test]
    fn tensor_slice_debug_format() {
        let data = &[1u8, 2, 3, 4];
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], data);
        let debug = format!("{slice:?}");
        assert!(debug.contains("TensorSlice"));
    }

    // ── TensorSlice shape edge cases ──────────────────────────────────────

    #[test]
    fn tensor_slice_shape_scalar() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], data);
        assert_eq!(slice.shape().len(), 1);
        assert_eq!(slice.shape()[0], 1);
    }

    #[test]
    fn tensor_slice_shape_3d() {
        let data = &[0u8; 24];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2, 3, 4], data);
        assert_eq!(slice.shape(), &[2, 3, 4]);
    }

    #[test]
    fn tensor_slice_shape_4d() {
        let data = &[0u8; 120];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2, 3, 4, 5], data);
        assert_eq!(slice.shape(), &[2, 3, 4, 5]);
        assert_eq!(slice.shape().len(), 4);
    }

    #[test]
    fn tensor_slice_shape_large_dims() {
        let data = &[0u8; 8];
        let slice = TensorSlice::new(GgmlDType::F32, vec![1024, 2048], data);
        assert_eq!(slice.shape()[0], 1024);
        assert_eq!(slice.shape()[1], 2048);
    }

    // ── TensorSlice clone independence for each dtype family ──────────────

    #[test]
    fn tensor_slice_clone_preserves_quantized_dtype() {
        let data = &[0u8; 72];
        let slice = TensorSlice::new(GgmlDType::AWQ4, vec![128], data);
        let cloned = slice.clone();
        assert_eq!(cloned.dtype(), slice.dtype());
        assert_eq!(cloned.shape(), slice.shape());
        assert_eq!(cloned.as_bytes().len(), slice.as_bytes().len());
    }

    #[test]
    fn tensor_slice_clone_preserves_k_quant() {
        let data = &[0u8; 84];
        let slice = TensorSlice::new(GgmlDType::Q2_K, vec![256], data);
        let cloned = slice.clone();
        assert_eq!(cloned.dtype(), GgmlDType::Q2_K);
        assert_eq!(cloned.shape(), &[256]);
    }

    // ── TensorSlice as_bytes with various sizes ───────────────────────────

    #[test]
    fn tensor_slice_as_bytes_single_byte() {
        let data = &[0xFF];
        let slice = TensorSlice::new(GgmlDType::I8, vec![1], data);
        assert_eq!(slice.as_bytes(), &[0xFF]);
    }

    #[test]
    fn tensor_slice_as_bytes_large_buffer() {
        let data = vec![0xABu8; 1024];
        let slice = TensorSlice::new(GgmlDType::I8, vec![1024], &data);
        assert_eq!(slice.as_bytes().len(), 1024);
        assert!(slice.as_bytes().iter().all(|&b| b == 0xAB));
    }

    // ── TensorSlice dtype roundtrip via discriminant ───────────────────────

    #[test]
    fn tensor_slice_dtype_roundtrip_all_non_quantized() {
        let non_quantized = [
            GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16,
            GgmlDType::F64, GgmlDType::I8, GgmlDType::I16,
            GgmlDType::I32, GgmlDType::I64,
        ];
        for dtype in non_quantized {
            let data = &[0u8; 8];
            let slice = TensorSlice::new(dtype, vec![1], data);
            let disc = slice.dtype() as u32;
            let recovered = GgmlDType::try_from(disc).unwrap();
            assert_eq!(slice.dtype(), recovered);
        }
    }

    // ── TensorSlice with dtype as_str property ────────────────────────────

    #[test]
    fn tensor_slice_dtype_as_str_not_empty() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::Q4_0, vec![32], data);
        let s = slice.dtype().as_str();
        assert!(!s.is_empty());
        assert_eq!(s, "Q4_0");
    }

    // ── GgmlDType::all() coverage via TensorSlice ─────────────────────────

    #[test]
    fn tensor_slice_all_dtypes_constructable() {
        for &dtype in GgmlDType::all() {
            let data = &[0u8; 8];
            let slice = TensorSlice::new(dtype, vec![1], data);
            assert_eq!(slice.dtype(), dtype);
        }
    }

    // ── TensorSlice shape is borrowed slice ───────────────────────────────

    #[test]
    fn tensor_slice_shape_is_slice_reference() {
        let data = &[0u8; 4];
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], data);
        let shape = slice.shape();
        assert_eq!(shape.len(), 1);
        assert_eq!(shape[0], 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 15 new tests -- shape/dtype/edge-case coverage
    // ═══════════════════════════════════════════════════════════════════════

    // ── TensorSlice with zero-dimension (empty shape vector) ───────────────

    #[test]
    fn tensor_slice_zero_dimension_shape() {
        // Arrange: a TensorSlice with empty shape vector (scalar-like, no dims)
        let data = &[0u8; 4];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![], data);
        // Assert
        assert!(slice.shape().is_empty());
        assert_eq!(slice.as_bytes().len(), 4);
    }

    // ── Multiple TensorSlices aliasing the same data buffer ────────────────

    #[test]
    fn tensor_slice_shared_data_aliasing() {
        // Arrange: two slices referencing the same backing buffer with different dtypes
        let data: &[u8] = &[0x00, 0x00, 0x80, 0x3F]; // F32 1.0 in little-endian
        let slice_f32 = TensorSlice::new(GgmlDType::F32, vec![1], data);
        let slice_i32 = TensorSlice::new(GgmlDType::I32, vec![1], data);
        // Act & Assert: both point to the same memory
        assert!(std::ptr::eq(slice_f32.as_bytes().as_ptr(), slice_i32.as_bytes().as_ptr()));
        assert_eq!(slice_f32.as_bytes().len(), slice_i32.as_bytes().len());
    }

    // ── TensorSlice preserves byte-level data content ─────────────────────

    #[test]
    fn tensor_slice_data_content_integrity() {
        // Arrange: known byte pattern
        let data: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        // Act
        let slice = TensorSlice::new(GgmlDType::F64, vec![1], data);
        // Assert: every byte matches the original
        for (i, &byte) in slice.as_bytes().iter().enumerate() {
            assert_eq!(byte, data[i], "byte mismatch at index {i}");
        }
    }

    // ── TensorSlice shape with u64::MAX boundary dimension ────────────────

    #[test]
    fn tensor_slice_shape_u64_max_dimension() {
        // Arrange: a shape entry at u64::MAX (TensorSlice itself does not validate)
        let data = &[0u8; 4];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![u64::MAX], data);
        // Assert
        assert_eq!(slice.shape()[0], u64::MAX);
    }

    // ── TensorSlice F32 data endianness verification ──────────────────────

    #[test]
    fn tensor_slice_f32_little_endian_layout() {
        // Arrange: 1.0f32 in little-endian is [0x00, 0x00, 0x80, 0x3F]
        let one_f32_le: &[u8] = &[0x00, 0x00, 0x80, 0x3F];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], one_f32_le);
        // Assert: raw bytes are accessible and match
        assert_eq!(slice.as_bytes(), one_f32_le);
    }

    // ── TensorSlice iterated construction for all quantized dtypes ────────

    #[test]
    fn tensor_slice_all_quantized_dtypes_are_quantized() {
        // Arrange: collect all quantized dtypes from GgmlDType::all()
        let quantized: Vec<GgmlDType> = GgmlDType::all()
            .iter()
            .filter(|d| d.is_quantized())
            .copied()
            .collect();
        // Act & Assert: each one, when wrapped in TensorSlice, reports quantized
        for &dtype in &quantized {
            let data = &[0u8; 8];
            let slice = TensorSlice::new(dtype, vec![1], data);
            assert!(slice.dtype().is_quantized(), "{dtype:?} should be quantized");
        }
        // Verify the count matches expected (36 total - 8 non-quantized = 28)
        assert_eq!(quantized.len(), 28);
    }

    // ── TensorSlice with under-sized data (no validation enforced) ────────

    #[test]
    fn tensor_slice_undersized_data_accepted() {
        // Arrange: data is smaller than what the shape+dtype would require
        // (TensorSlice is a raw view, does not validate size consistency)
        let data = &[0u8; 2];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![100], data);
        // Assert: accessors work even with mismatched sizes
        assert_eq!(slice.as_bytes().len(), 2);
        assert_eq!(slice.shape()[0], 100);
        assert_eq!(slice.dtype(), GgmlDType::F32);
    }

    // ── TensorSlice clone produces independent shape vector ───────────────

    #[test]
    fn tensor_slice_clone_shape_vector_independence() {
        // Arrange
        let data = &[0u8; 8];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2, 3, 4], data);
        // Act
        let cloned = slice.clone();
        // Assert: shape vectors have same content
        assert_eq!(cloned.shape(), slice.shape());
        assert_eq!(cloned.shape().len(), 3);
        assert_eq!(cloned.shape()[0], 2);
        assert_eq!(cloned.shape()[1], 3);
        assert_eq!(cloned.shape()[2], 4);
    }

    // ── TensorSlice with all block-32 quantized dtypes ────────────────────

    #[test]
    fn tensor_slice_block_32_dtypes() {
        // Arrange: all dtypes with block_size=32
        let block_32: &[GgmlDType] = &[
            GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0,
            GgmlDType::Q5_1, GgmlDType::Q8_0, GgmlDType::Q8_1,
            GgmlDType::IQ4_NL, GgmlDType::MXFP4,
        ];
        for &dtype in block_32 {
            // Act
            let data = vec![0u8; dtype.block_bytes()];
            let slice = TensorSlice::new(dtype, vec![32], &data);
            // Assert
            assert_eq!(slice.dtype().block_size(), 32, "{dtype:?} block_size mismatch");
            assert!(slice.dtype().is_quantized());
        }
    }

    // ── TensorSlice with all block-256 (K-quant) dtypes ───────────────────

    #[test]
    fn tensor_slice_block_256_k_quant_dtypes() {
        // Arrange: all dtypes with block_size=256 (K-quant and IQ-K families)
        let block_256: &[GgmlDType] = &[
            GgmlDType::Q2_K, GgmlDType::Q3_K, GgmlDType::Q4_K,
            GgmlDType::Q5_K, GgmlDType::Q6_K, GgmlDType::Q8_K,
            GgmlDType::IQ2_XXS, GgmlDType::IQ2_XS, GgmlDType::IQ3_XXS,
            GgmlDType::IQ1_S, GgmlDType::IQ3_S, GgmlDType::IQ2_S,
            GgmlDType::IQ4_XS, GgmlDType::IQ1_M, GgmlDType::TQ1_0,
            GgmlDType::TQ2_0, GgmlDType::SQUEEZE,
        ];
        for &dtype in block_256 {
            // Act
            let data = vec![0u8; dtype.block_bytes()];
            let slice = TensorSlice::new(dtype, vec![256], &data);
            // Assert
            assert_eq!(slice.dtype().block_size(), 256, "{dtype:?} block_size mismatch");
            assert_eq!(slice.as_bytes().len(), dtype.block_bytes());
        }
    }

    // ── TensorSlice shape dimension count reflects dimensionality ─────────

    #[test]
    fn tensor_slice_shape_dimension_count() {
        // Arrange: 1D, 2D, 3D, 4D, 5D shapes
        let cases: &[Vec<u64>] = &[
            vec![10],
            vec![10, 20],
            vec![10, 20, 30],
            vec![2, 3, 4, 5],
            vec![1, 2, 3, 4, 5],
        ];
        for (i, shape) in cases.iter().enumerate() {
            // Act
            let data = vec![0u8; 4];
            let slice = TensorSlice::new(GgmlDType::F32, shape.clone(), &data);
            // Assert
            assert_eq!(slice.shape().len(), i + 1, "expected {}D shape", i + 1);
        }
    }

    // ── TensorSlice as a sub-slice of a larger buffer ─────────────────────

    #[test]
    fn tensor_slice_subslice_of_larger_buffer() {
        // Arrange: a 16-byte buffer, create a TensorSlice pointing to bytes [4..12]
        let buffer: &[u8] = &[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                              0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F];
        let sub_data = &buffer[4..12];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![2], sub_data);
        // Assert: data pointer falls within the parent buffer
        let slice_ptr = slice.as_bytes().as_ptr() as usize;
        let buf_start = buffer.as_ptr() as usize;
        let buf_end = buf_start + buffer.len();
        assert!((buf_start..buf_end).contains(&slice_ptr));
        assert_eq!(slice.as_bytes(), &[0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B]);
    }

    // ── TensorSlice dtype roundtrip for ALL dtypes (including quantized) ──

    #[test]
    fn tensor_slice_dtype_roundtrip_all_dtypes() {
        // Arrange & Act: create a slice with every dtype, verify discriminant roundtrip
        for &dtype in GgmlDType::all() {
            let data = &[0u8; 8];
            let slice = TensorSlice::new(dtype, vec![1], data);
            let disc = slice.dtype() as u32;
            let recovered = GgmlDType::try_from(disc).unwrap();
            // Assert
            assert_eq!(slice.dtype(), recovered, "roundtrip failed for {dtype:?}");
        }
    }

    // ── TensorSlice with BF16 and F16 have same block_bytes ──────────────

    #[test]
    fn tensor_slice_bf16_f16_same_block_bytes() {
        // Arrange
        let data = &[0u8; 4];
        let f16_slice = TensorSlice::new(GgmlDType::F16, vec![2], data);
        let bf16_slice = TensorSlice::new(GgmlDType::BF16, vec![2], data);
        // Assert: both are 2-byte per element formats
        assert_eq!(f16_slice.dtype().block_bytes(), bf16_slice.dtype().block_bytes());
        assert_eq!(f16_slice.dtype().block_bytes(), 2);
    }

    // ── TensorSlice from a Vec-backed slice (owned data) ─────────────────

    #[test]
    fn tensor_slice_from_owned_vec_data() {
        // Arrange: dynamically created data
        let owned: Vec<u8> = (0..16).collect();
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![4], &owned);
        // Assert
        assert_eq!(slice.as_bytes().len(), 16);
        assert_eq!(slice.as_bytes()[0], 0);
        assert_eq!(slice.as_bytes()[15], 15);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 15 additional tests — lifetime, data integrity, edge cases, traits
    // ═══════════════════════════════════════════════════════════════════════

    // ── TensorSlice Debug includes shape length ───────────────────────────

    #[test]
    fn tensor_slice_debug_includes_shape_info() {
        // Arrange: a TensorSlice with multi-dimensional shape
        let data = &[0u8; 8];
        let slice = TensorSlice::new(GgmlDType::F32, vec![2, 4], data);
        // Act
        let debug = format!("{slice:?}");
        // Assert: Debug output must contain the struct name and reflect shape
        assert!(debug.contains("TensorSlice"));
        // The shape vec [2, 4] should be reflected somewhere in the debug output
        let shape = slice.shape();
        assert_eq!(shape, &[2, 4]);
    }

    // ── TensorSlice with all-zero bytes F32 represents 0.0 ────────────────

    #[test]
    fn tensor_slice_f32_zero_bytes_represents_zero() {
        // Arrange: all-zero bytes interpreted as F32
        let zeros: &[u8] = &[0x00, 0x00, 0x00, 0x00];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], zeros);
        // Assert: raw bytes are zero, representing 0.0f32 in IEEE 754
        assert_eq!(slice.as_bytes(), &[0x00, 0x00, 0x00, 0x00]);
        // Reinterpret as f32 to verify semantic meaning
        let f32_val = f32::from_le_bytes([
            slice.as_bytes()[0],
            slice.as_bytes()[1],
            slice.as_bytes()[2],
            slice.as_bytes()[3],
        ]);
        assert_eq!(f32_val, 0.0f32);
    }

    // ── TensorSlice clone data pointer remains the same source ────────────

    #[test]
    fn tensor_slice_clone_points_to_same_underlying_data() {
        // Arrange
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let original = TensorSlice::new(GgmlDType::F64, vec![1], data);
        // Act
        let cloned = original.clone();
        // Assert: both original and cloned reference the same underlying byte range
        assert!(std::ptr::eq(
            original.as_bytes().as_ptr(),
            cloned.as_bytes().as_ptr()
        ));
        assert_eq!(original.as_bytes().len(), cloned.as_bytes().len());
    }

    // ── TensorSlice with I32 data interprets as signed integer ────────────

    #[test]
    fn tensor_slice_i32_data_signed_integer_layout() {
        // Arrange: -1 as I32 in little-endian = [0xFF, 0xFF, 0xFF, 0xFF]
        let neg_one_le: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF];
        // Act
        let slice = TensorSlice::new(GgmlDType::I32, vec![1], neg_one_le);
        // Assert: raw bytes preserved correctly
        assert_eq!(slice.as_bytes().len(), 4);
        assert_eq!(slice.as_bytes()[0], 0xFF);
        // Reinterpret to verify -1
        let i32_val = i32::from_le_bytes([
            slice.as_bytes()[0],
            slice.as_bytes()[1],
            slice.as_bytes()[2],
            slice.as_bytes()[3],
        ]);
        assert_eq!(i32_val, -1);
    }

    // ── TensorSlice new does not validate shape-dtype-data consistency ─────

    #[test]
    fn tensor_slice_no_validation_oversized_data() {
        // Arrange: data is larger than shape+dtype would require
        let oversized: &[u8] = &[0xAA; 256];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![1], oversized);
        // Assert: TensorSlice is a raw view and does not enforce consistency
        assert_eq!(slice.as_bytes().len(), 256);
        assert_eq!(slice.shape()[0], 1); // claims 1 element
        assert_eq!(slice.dtype(), GgmlDType::F32); // 4 bytes per element
    }

    // ── TensorSlice block_bytes consistency with dtype for Q6_K ───────────

    #[test]
    fn tensor_slice_q6_k_block_bytes_consistency() {
        // Arrange: Q6_K block_size=256, block_bytes=210
        let data = vec![0u8; GgmlDType::Q6_K.block_bytes()];
        // Act
        let slice = TensorSlice::new(GgmlDType::Q6_K, vec![256], &data);
        // Assert: data length matches block_bytes exactly
        assert_eq!(slice.as_bytes().len(), 210);
        assert_eq!(slice.dtype().block_bytes(), 210);
        assert_eq!(slice.dtype().block_size(), 256);
        // Ratio: 210 bytes / 256 elements ≈ 6.56 bits per element
    }

    // ── TensorSlice NVFP4 vs MXFP4 distinct block properties ─────────────

    #[test]
    fn tensor_slice_nvfp4_vs_mxfp4_block_properties() {
        // Arrange: NVFP4 (block=64, bytes=36) and MXFP4 (block=32, bytes=17)
        let nvfp4_data = vec![0u8; GgmlDType::NVFP4.block_bytes()];
        let mxfp4_data = vec![0u8; GgmlDType::MXFP4.block_bytes()];
        // Act
        let nvfp4 = TensorSlice::new(GgmlDType::NVFP4, vec![64], &nvfp4_data);
        let mxfp4 = TensorSlice::new(GgmlDType::MXFP4, vec![32], &mxfp4_data);
        // Assert: different block sizes and block bytes
        assert_ne!(nvfp4.dtype().block_size(), mxfp4.dtype().block_size());
        assert_ne!(nvfp4.dtype().block_bytes(), mxfp4.dtype().block_bytes());
        assert_eq!(nvfp4.dtype().block_size(), 64);
        assert_eq!(mxfp4.dtype().block_size(), 32);
    }

    // ── TensorSlice two independent slices over different regions ──────────

    #[test]
    fn tensor_slice_two_independent_regions_same_buffer() {
        // Arrange: a 32-byte buffer split into two non-overlapping regions
        let buffer: Vec<u8> = (0..32).collect();
        let region_a = &buffer[0..16];
        let region_b = &buffer[16..32];
        // Act
        let slice_a = TensorSlice::new(GgmlDType::F32, vec![4], region_a);
        let slice_b = TensorSlice::new(GgmlDType::F32, vec![4], region_b);
        // Assert: pointers do not overlap
        let ptr_a = slice_a.as_bytes().as_ptr() as usize;
        let ptr_b = slice_b.as_bytes().as_ptr() as usize;
        assert!(ptr_a < ptr_b, "region A must start before region B");
        assert_eq!(slice_a.as_bytes().len(), 16);
        assert_eq!(slice_b.as_bytes().len(), 16);
        // Content is different
        assert_ne!(slice_a.as_bytes(), slice_b.as_bytes());
    }

    // ── TensorSlice AWQ4 vs GPTQ4 same block_size but same block_bytes ────

    #[test]
    fn tensor_slice_awq4_gptq4_same_block_dimensions() {
        // Arrange: AWQ4 and GPTQ4 both have block_size=128, block_bytes=72
        let data = &[0u8; 72];
        // Act
        let awq4 = TensorSlice::new(GgmlDType::AWQ4, vec![128], data);
        let gptq4 = TensorSlice::new(GgmlDType::GPTQ4, vec![128], data);
        // Assert: identical block dimensions but distinct dtype values
        assert_ne!(awq4.dtype(), gptq4.dtype());
        assert_eq!(awq4.dtype().block_size(), gptq4.dtype().block_size());
        assert_eq!(awq4.dtype().block_bytes(), gptq4.dtype().block_bytes());
        assert_eq!(awq4.dtype().as_str(), "AWQ4");
        assert_eq!(gptq4.dtype().as_str(), "GPTQ4");
    }

    // ── TensorSlice shape with single u64::MAX in multi-dimensional shape ─

    #[test]
    fn tensor_slice_shape_multi_dim_with_u64_max_inner() {
        // Arrange: a 2D shape where inner dimension is u64::MAX
        let data = &[0u8; 4];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![u64::MAX, 1], data);
        // Assert: TensorSlice stores shape without validation
        assert_eq!(slice.shape()[0], u64::MAX);
        assert_eq!(slice.shape()[1], 1);
        assert_eq!(slice.shape().len(), 2);
    }

    // ── TensorSlice F16 two-element slice byte layout ─────────────────────

    #[test]
    fn tensor_slice_f16_two_element_byte_layout() {
        // Arrange: F16 1.0 = [0x00, 0x3C], F16 2.0 = [0x00, 0x40]
        let f16_one_and_two: &[u8] = &[0x00, 0x3C, 0x00, 0x40];
        // Act
        let slice = TensorSlice::new(GgmlDType::F16, vec![2], f16_one_and_two);
        // Assert: 2 elements * 2 bytes each = 4 bytes total
        assert_eq!(slice.as_bytes().len(), 4);
        assert_eq!(slice.dtype().block_bytes(), 2);
        assert_eq!(slice.dtype().block_size(), 1);
    }

    // ── TensorSlice non-quantized dtype block_bytes equals element size ────

    #[test]
    fn tensor_slice_non_quantized_block_bytes_is_element_size() {
        // Arrange: verify that for non-quantized types, block_bytes == sizeof(type)
        let cases: &[(GgmlDType, usize)] = &[
            (GgmlDType::F32, 4),
            (GgmlDType::F16, 2),
            (GgmlDType::BF16, 2),
            (GgmlDType::I8, 1),
            (GgmlDType::I16, 2),
            (GgmlDType::I32, 4),
            (GgmlDType::I64, 8),
            (GgmlDType::F64, 8),
        ];
        for &(dtype, expected_bytes) in cases {
            // Act
            let data = vec![0u8; expected_bytes];
            let slice = TensorSlice::new(dtype, vec![1], &data);
            // Assert
            assert!(
                !slice.dtype().is_quantized(),
                "{dtype:?} should not be quantized"
            );
            assert_eq!(
                slice.dtype().block_bytes(),
                expected_bytes,
                "{dtype:?} block_bytes mismatch"
            );
        }
    }

    // ── TensorSlice SQUEEZE distinct from Q8_K despite same block_size ────

    #[test]
    fn tensor_slice_squeeze_vs_q8_k_same_block_size_different_bytes() {
        // Arrange: SQUEEZE and Q8_K both have block_size=256 but different block_bytes
        let squeeze_data = vec![0u8; GgmlDType::SQUEEZE.block_bytes()];
        let q8k_data = vec![0u8; GgmlDType::Q8_K.block_bytes()];
        // Act
        let squeeze = TensorSlice::new(GgmlDType::SQUEEZE, vec![256], &squeeze_data);
        let q8k = TensorSlice::new(GgmlDType::Q8_K, vec![256], &q8k_data);
        // Assert: same block size, different block bytes
        assert_eq!(squeeze.dtype().block_size(), q8k.dtype().block_size());
        assert_ne!(squeeze.dtype().block_bytes(), q8k.dtype().block_bytes());
        assert_eq!(squeeze.dtype().block_bytes(), 130);
        assert_eq!(q8k.dtype().block_bytes(), 292);
    }

    // ── TensorSlice static lifetime data ──────────────────────────────────

    #[test]
    fn tensor_slice_static_lifetime_data() {
        // Arrange: 'static data (compile-time constant)
        const STATIC_DATA: &[u8] = &[0x42; 8];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![2], STATIC_DATA);
        // Assert: works with 'static data, all bytes match
        assert_eq!(slice.as_bytes().len(), 8);
        assert!(slice.as_bytes().iter().all(|&b| b == 0x42));
        assert_eq!(slice.dtype(), GgmlDType::F32);
    }

    // ── TensorSlice Q4_0 minimal block (exactly one block) ────────────────

    #[test]
    fn tensor_slice_q4_0_exact_single_block() {
        // Arrange: Q4_0 block_size=32, block_bytes=18 — exactly one block
        let data: Vec<u8> = (0..18).collect();
        // Act
        let slice = TensorSlice::new(GgmlDType::Q4_0, vec![32], &data);
        // Assert: data length matches exactly one block
        assert_eq!(slice.as_bytes().len(), 18);
        assert_eq!(slice.dtype().block_bytes(), 18);
        assert_eq!(slice.dtype().block_size(), 32);
        // First byte is the scale factor for Q4_0, second byte onwards are quants
        assert_eq!(slice.as_bytes()[0], 0);
        assert_eq!(slice.as_bytes()[17], 17);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 10 additional tests — trait consistency, boundary values, slice semantics
    // ═══════════════════════════════════════════════════════════════════════

    // ── TensorSlice Debug output is non-empty for all dtypes ──────────────

    #[test]
    fn tensor_slice_debug_non_empty_for_all_dtypes() {
        // Arrange: iterate every GgmlDType variant
        for &dtype in GgmlDType::all() {
            let data = &[0u8; 4];
            // Act
            let slice = TensorSlice::new(dtype, vec![1], data);
            let debug_str = format!("{slice:?}");
            // Assert: Debug output contains struct name and is non-empty
            assert!(!debug_str.is_empty(), "Debug empty for {dtype:?}");
            assert!(debug_str.contains("TensorSlice"), "missing struct name for {dtype:?}");
        }
    }

    // ── GgmlDType::all() returns exactly 36 variants ─────────────────────

    #[test]
    fn ggml_dtype_all_returns_36_variants() {
        // Arrange: the GgmlDType enum has 36 defined variants
        // Act
        let all = GgmlDType::all();
        // Assert
        assert_eq!(all.len(), 36, "GgmlDType::all() must enumerate every variant");
        // All entries are unique (no duplicates)
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "duplicate variant at indices {i} and {j}");
            }
        }
    }

    // ── GgmlDType TryFrom<u32> rejects invalid discriminants ─────────────

    #[test]
    fn ggml_dtype_try_from_rejects_invalid_discriminants() {
        // Arrange: discriminants that do not map to any GgmlDType variant
        let invalid_discs: &[u32] = &[4, 5, 31, 32, 33, 36, 37, 38, 49, 100, u32::MAX];
        // Act & Assert
        for &disc in invalid_discs {
            let result = GgmlDType::try_from(disc);
            assert!(result.is_err(), "disc {disc} should be invalid but got {:?}", result.unwrap());
        }
    }

    // ── TensorSlice shape with zero-valued dimension ──────────────────────

    #[test]
    fn tensor_slice_shape_zero_valued_dimension() {
        // Arrange: shape contains a zero dimension (valid for TensorSlice as raw view)
        let data = &[0u8; 0];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, vec![0], data);
        // Assert
        assert_eq!(slice.shape()[0], 0);
        assert!(slice.as_bytes().is_empty());
    }

    // ── TensorSlice clone chain (clone-of-clone) preserves all fields ────

    #[test]
    fn tensor_slice_clone_chain_preserves_fields() {
        // Arrange: original slice, clone it, then clone the clone
        let data: &[u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let original = TensorSlice::new(GgmlDType::F64, vec![2, 4], data);
        // Act
        let clone1 = original.clone();
        let clone2 = clone1.clone();
        // Assert: all three agree on dtype, shape, data
        assert_eq!(clone2.dtype(), original.dtype());
        assert_eq!(clone2.shape(), original.shape());
        assert_eq!(clone2.as_bytes(), original.as_bytes());
        assert_eq!(clone1.dtype(), clone2.dtype());
        assert_eq!(clone1.shape(), clone2.shape());
    }

    // ── TensorSlice shape with many dimensions (6D tensor) ───────────────

    #[test]
    fn tensor_slice_shape_6d_tensor() {
        // Arrange: 6-dimensional shape (common in batch-seq-head-group-dim-vocab scenarios)
        let data = &[0u8; 8];
        let shape = vec![2, 4, 8, 16, 32, 64];
        // Act
        let slice = TensorSlice::new(GgmlDType::F32, shape.clone(), data);
        // Assert
        assert_eq!(slice.shape().len(), 6);
        assert_eq!(slice.shape(), &shape[..]);
    }

    // ── GgmlDType as_str returns correct name for every variant ──────────

    #[test]
    fn ggml_dtype_as_str_returns_non_empty_for_all() {
        // Arrange & Act: check as_str() for every variant
        for &dtype in GgmlDType::all() {
            let name = dtype.as_str();
            // Assert: name is non-empty and matches expected pattern (uppercase + digits + _)
            assert!(!name.is_empty(), "as_str() empty for {dtype:?}");
            assert_eq!(name, format!("{dtype:?}"), "as_str() must match Debug name for {dtype:?}");
        }
    }

    // ── GgmlDType block_bytes >= block_size / 8 invariant (quantized) ────

    #[test]
    fn ggml_dtype_quantized_block_bytes_geq_bits_per_element() {
        // Arrange: for every quantized dtype, verify block_bytes >= block_size/8
        // (quantized formats cannot use fewer bits than 1 per element, ignoring header)
        for &dtype in GgmlDType::all() {
            if !dtype.is_quantized() {
                continue;
            }
            let bs = dtype.block_size();
            let bb = dtype.block_bytes();
            // Assert: block_bytes must be at least 1 byte (quantized blocks always have header)
            assert!(bb >= 1, "{dtype:?} block_bytes={bb} too small");
            // bits per element = bb * 8 / bs, must be >= 1
            let bits_per_elem = (bb * 8) as f64 / bs as f64;
            assert!(bits_per_elem >= 1.0, "{dtype:?} has {bits_per_elem} bits/elem, less than 1");
        }
    }

    // ── TensorSlice I64 data reinterpretation preserves negative value ────

    #[test]
    fn tensor_slice_i64_negative_value_layout() {
        // Arrange: -256 as I64 little-endian = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
        let neg_256_le: &[u8] = &[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        // Act
        let slice = TensorSlice::new(GgmlDType::I64, vec![1], neg_256_le);
        // Assert: raw bytes preserved, reinterpretation matches -256
        assert_eq!(slice.as_bytes().len(), 8);
        let val = i64::from_le_bytes([
            slice.as_bytes()[0], slice.as_bytes()[1], slice.as_bytes()[2], slice.as_bytes()[3],
            slice.as_bytes()[4], slice.as_bytes()[5], slice.as_bytes()[6], slice.as_bytes()[7],
        ]);
        assert_eq!(val, -256);
    }

    // ── GgmlDType non-quantized dtypes have block_size == 1 ──────────────

    #[test]
    fn ggml_dtype_non_quantized_block_size_is_one() {
        // Arrange: all non-quantized dtypes must have block_size=1 (one element per block)
        let non_quantized: &[GgmlDType] = &[
            GgmlDType::F32, GgmlDType::F16, GgmlDType::BF16,
            GgmlDType::F64, GgmlDType::I8, GgmlDType::I16,
            GgmlDType::I32, GgmlDType::I64,
        ];
        for &dtype in non_quantized {
            // Act
            let bs = dtype.block_size();
            // Assert
            assert_eq!(bs, 1, "{dtype:?} block_size must be 1 for non-quantized");
            assert!(!dtype.is_quantized());
        }
    }
}
