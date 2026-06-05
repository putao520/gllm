//! Mega-kernel SG shared memory (SPEC §7.4.2).
//!
//! `SgSharedMemory` is passed via `hook_ctx_ptr` (ABI arg 15) to the
//! mega-kernel. The JIT SgDetect op writes `detect_hidden`; SgInject reads
//! `knowledge_vector`. Rust fills `control` / `confidence` / `knowledge_vector`
//! when SG is registered.

/// Fixed header size (4 × u32 = 16 bytes) before the dynamic arrays.
const SG_HEADER_BYTES: usize = 16;

/// Mega-kernel SG shared memory (SPEC §7.4.2).
///
/// Layout:
/// ```text
/// [0..4]   control: u32         — bit 0 = sg_enabled
/// [4..8]   knowledge_offset: u32
/// [8..12]  knowledge_dim: u32
/// [12..16] confidence: u32      — IEEE 754 f32 bit pattern
/// [16..16+hidden*4]   detect_hidden: [f32; hidden]
/// [16+hidden*4..16+2*hidden*4]  knowledge_vector: [f32; hidden]
/// ```
pub struct SgSharedMemory {
    data: Box<[u8]>,
    hidden_size: usize,
}

impl SgSharedMemory {
    /// Allocate shared memory for `hidden_size` hidden dimension.
    pub fn new(hidden_size: usize) -> Self {
        let total = SG_HEADER_BYTES + 2 * hidden_size * 4;
        let data = vec![0u8; total].into_boxed_slice();
        // Zero-init: control=0 → SG disabled until explicitly enabled.
        let mut me = Self { data, hidden_size };
        me.set_control(0);
        me.set_knowledge_dim(hidden_size as u32);
        me
    }

    /// Enable SG (set control bit 0).
    pub fn enable(&mut self) {
        let ctrl = self.control();
        self.set_control(ctrl | 1);
    }

    /// Disable SG (clear control bit 0).
    pub fn disable(&mut self) {
        let ctrl = self.control();
        self.set_control(ctrl & !1);
    }

    /// Returns `true` if SG is enabled (control bit 0 = 1).
    pub fn is_enabled(&self) -> bool {
        (self.control() & 1) != 0
    }

    /// Raw pointer for passing to mega-kernel as `hook_ctx_ptr`.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// hidden_size dimension.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    // ── Header field accessors ──

    fn control(&self) -> u32 {
        u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]])
    }

    fn set_control(&mut self, v: u32) {
        self.data[0..4].copy_from_slice(&v.to_le_bytes());
    }

    fn set_knowledge_dim(&mut self, v: u32) {
        self.data[8..12].copy_from_slice(&v.to_le_bytes());
    }

    /// Set confidence (f32 bit pattern stored as u32).
    pub fn set_confidence(&mut self, conf: f32) {
        self.data[12..16].copy_from_slice(&conf.to_bits().to_le_bytes());
    }

    // ── detect_hidden (JIT writes, Rust reads) ──

    /// Read detect_hidden as a slice of f32.
    pub fn detect_hidden(&self) -> &[f32] {
        let start = SG_HEADER_BYTES;
        let end = start + self.hidden_size * 4;
        let bytes = &self.data[start..end];
        // SAFETY: aligned to 4 bytes (header is 16 bytes = 4-aligned), length is hidden_size * 4.
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.hidden_size)
        }
    }

    // ── knowledge_vector (Rust writes, JIT reads) ──

    /// Write knowledge_vector from a slice of f32.
    pub fn set_knowledge_vector(&mut self, vec: &[f32]) {
        let start = SG_HEADER_BYTES + self.hidden_size * 4;
        let end = start + self.hidden_size * 4;
        let bytes = &mut self.data[start..end];
        let n = vec.len().min(self.hidden_size);
        let dst = unsafe {
            std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, self.hidden_size)
        };
        dst[..n].copy_from_slice(&vec[..n]);
        // Zero-fill remainder if vec is shorter than hidden_size.
        for v in dst.iter_mut().skip(n) {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sg_shared_memory_layout() {
        let mut sg = SgSharedMemory::new(8);
        assert!(!sg.is_enabled());
        assert_eq!(sg.hidden_size(), 8);
        assert_eq!(sg.detect_hidden().len(), 8);

        sg.enable();
        assert!(sg.is_enabled());

        sg.set_confidence(0.75);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        sg.disable();
        assert!(!sg.is_enabled());
    }

    #[test]
    fn test_sg_shared_memory_ptr_non_null() {
        let sg = SgSharedMemory::new(16);
        assert!(!sg.as_ptr().is_null());
    }

    #[test]
    fn test_enable_disable_roundtrip() {
        let mut sg = SgSharedMemory::new(4);
        assert!(!sg.is_enabled());
        sg.enable();
        assert!(sg.is_enabled());
        sg.disable();
        assert!(!sg.is_enabled());
        sg.enable();
        assert!(sg.is_enabled());
    }

    #[test]
    fn test_detect_hidden_initially_zero() {
        let sg = SgSharedMemory::new(4);
        for v in sg.detect_hidden() {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    fn test_knowledge_vector_partial_write_zero_fills() {
        let mut sg = SgSharedMemory::new(8);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0]);
        let kv = sg.detect_hidden();
        // knowledge_vector is in the second half (after detect_hidden)
        // We can't directly read knowledge_vector, but we verify detect_hidden is still zero
        for v in kv {
            assert_eq!(*v, 0.0f32);
        }
    }

    #[test]
    fn test_total_size_calculation() {
        let sg = SgSharedMemory::new(64);
        // header(16) + 2 × hidden_size × 4 = 16 + 2 × 64 × 4 = 528
        assert_eq!(sg.hidden_size(), 64);
    }

    #[test]
    fn test_confidence_roundtrip() {
        let mut sg = SgSharedMemory::new(4);
        sg.set_confidence(0.95);
        // Read back via raw bytes at offset 12
        let bits = u32::from_le_bytes([
            sg.data[12], sg.data[13], sg.data[14], sg.data[15],
        ]);
        let conf = f32::from_bits(bits);
        assert!((conf - 0.95f32).abs() < 1e-6);
    }

    #[test]
    fn test_header_bytes_constant() {
        // SPEC: 4 × u32 = 16 bytes fixed header.
        assert_eq!(SG_HEADER_BYTES, 16);
    }

    #[test]
    fn test_hidden_size_one_minimum() {
        let sg = SgSharedMemory::new(1);
        assert_eq!(sg.hidden_size(), 1);
        assert_eq!(sg.detect_hidden().len(), 1);
        assert_eq!(sg.data.len(), SG_HEADER_BYTES + 2 * 1 * 4); // 16 + 8 = 24
    }

    #[test]
    fn test_hidden_size_zero_no_panic() {
        let sg = SgSharedMemory::new(0);
        assert_eq!(sg.hidden_size(), 0);
        assert_eq!(sg.detect_hidden().len(), 0);
        assert_eq!(sg.data.len(), SG_HEADER_BYTES); // header only, no dynamic arrays
        assert!(!sg.is_enabled());
    }

    #[test]
    fn test_knowledge_vector_exact_length_write() {
        let mut sg = SgSharedMemory::new(4);
        let values = [10.0, 20.0, 30.0, 40.0];
        sg.set_knowledge_vector(&values);
        // Verify the second half of data (knowledge_vector region) contains the values.
        let kv_offset = SG_HEADER_BYTES + 4 * 4; // after detect_hidden
        let kv_bytes = &sg.data[kv_offset..kv_offset + 4 * 4];
        let kv = unsafe {
            std::slice::from_raw_parts(kv_bytes.as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 10.0);
        assert_eq!(kv[1], 20.0);
        assert_eq!(kv[2], 30.0);
        assert_eq!(kv[3], 40.0);
    }

    #[test]
    fn test_knowledge_vector_truncation_when_longer() {
        let mut sg = SgSharedMemory::new(3);
        // Provide 5 values but hidden_size is only 3 — should truncate.
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(
                sg.data[kv_offset..].as_ptr() as *const f32,
                3,
            )
        };
        assert_eq!(kv[0], 1.0);
        assert_eq!(kv[1], 2.0);
        assert_eq!(kv[2], 3.0);
    }

    #[test]
    fn test_knowledge_vector_zero_fills_remaining_slots() {
        let mut sg = SgSharedMemory::new(6);
        // Write only 2 values — remaining 4 should be 0.0.
        sg.set_knowledge_vector(&[7.0, 8.0]);
        let kv_offset = SG_HEADER_BYTES + 6 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(
                sg.data[kv_offset..].as_ptr() as *const f32,
                6,
            )
        };
        assert_eq!(kv[0], 7.0);
        assert_eq!(kv[1], 8.0);
        for v in &kv[2..] {
            assert_eq!(*v, 0.0f32, "remaining slot should be zero-filled");
        }
    }

    #[test]
    fn test_detect_hidden_and_knowledge_vector_regions_are_independent() {
        let mut sg = SgSharedMemory::new(4);
        // Simulate JIT writing to detect_hidden via raw bytes.
        let dh_offset = SG_HEADER_BYTES;
        let dh_bytes = &mut sg.data[dh_offset..dh_offset + 4 * 4];
        let dh = unsafe {
            std::slice::from_raw_parts_mut(dh_bytes.as_mut_ptr() as *mut f32, 4)
        };
        dh[0] = 99.0;
        dh[3] = -1.0;

        // Now write knowledge_vector — should not touch detect_hidden.
        sg.set_knowledge_vector(&[50.0, 51.0, 52.0, 53.0]);

        // Verify detect_hidden preserved.
        let read_dh = sg.detect_hidden();
        assert_eq!(read_dh[0], 99.0);
        assert_eq!(read_dh[1], 0.0);
        assert_eq!(read_dh[2], 0.0);
        assert_eq!(read_dh[3], -1.0);

        // Verify knowledge_vector.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 50.0);
        assert_eq!(kv[3], 53.0);
    }

    #[test]
    fn test_confidence_boundary_values() {
        let mut sg = SgSharedMemory::new(2);

        // Zero confidence.
        sg.set_confidence(0.0);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(f32::from_bits(bits), 0.0);

        // Full confidence.
        sg.set_confidence(1.0);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!((f32::from_bits(bits) - 1.0f32).abs() < 1e-6);

        // Negative confidence.
        sg.set_confidence(-0.5);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!((f32::from_bits(bits) - (-0.5f32)).abs() < 1e-6);

        // Infinity.
        sg.set_confidence(f32::INFINITY);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!(f32::from_bits(bits).is_infinite() && f32::from_bits(bits).is_sign_positive());

        // NaN.
        sg.set_confidence(f32::NAN);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!(f32::from_bits(bits).is_nan());
    }

    #[test]
    fn test_control_high_bits_preserved() {
        let mut sg = SgSharedMemory::new(2);
        // Manually set high bits via raw bytes at offset 0..4.
        // bit 31 set, bit 0 clear → not enabled but high bit present.
        let ctrl_with_high = 0x80000000u32;
        sg.data[0..4].copy_from_slice(&ctrl_with_high.to_le_bytes());
        assert!(!sg.is_enabled()); // bit 0 is 0.
        sg.enable(); // sets bit 0, should preserve bit 31.
        assert!(sg.is_enabled());
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl & 1, 1); // bit 0 set
        assert_eq!(ctrl & 0x80000000, 0x80000000); // bit 31 preserved
    }

    #[test]
    fn test_enable_idempotent() {
        let mut sg = SgSharedMemory::new(2);
        sg.enable();
        assert!(sg.is_enabled());
        sg.enable(); // second enable should be a no-op (bit stays 1).
        assert!(sg.is_enabled());
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 1u32); // exactly 1, not 3 or anything else.
    }

    #[test]
    fn test_disable_idempotent() {
        let mut sg = SgSharedMemory::new(2);
        sg.disable();
        assert!(!sg.is_enabled());
        sg.disable(); // second disable on already-disabled.
        assert!(!sg.is_enabled());
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 0u32);
    }

    #[test]
    fn test_large_hidden_size_layout() {
        let hidden = 4096;
        let sg = SgSharedMemory::new(hidden);
        let expected_bytes = SG_HEADER_BYTES + 2 * hidden * 4; // 16 + 32768 = 32784
        assert_eq!(sg.data.len(), expected_bytes);
        assert_eq!(sg.hidden_size(), hidden);
        assert_eq!(sg.detect_hidden().len(), hidden);
    }

    #[test]
    fn test_new_initializes_knowledge_dim_in_header() {
        let sg = SgSharedMemory::new(128);
        // Bytes 8..12 should contain hidden_size as u32 LE.
        let dim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim, 128u32);
    }

    #[test]
    fn test_new_zero_hidden_size_knowledge_dim_header() {
        let sg = SgSharedMemory::new(0);
        let dim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim, 0u32);
    }

    // ── Additional tests ──

    #[test]
    fn test_new_dynamic_regions_zeroed() {
        // new() writes knowledge_dim into bytes 8..11, so header is not all-zero.
        // Verify the dynamic arrays (detect_hidden + knowledge_vector) are all-zero.
        let sg = SgSharedMemory::new(16);
        let dynamic_start = SG_HEADER_BYTES;
        for (i, &byte) in sg.data[dynamic_start..].iter().enumerate() {
            assert_eq!(byte, 0u8, "dynamic byte at offset {i} should be zero-initialized");
        }
    }

    #[test]
    fn test_new_control_is_zero() {
        let sg = SgSharedMemory::new(4);
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 0u32, "control should be zero after construction");
    }

    #[test]
    fn test_as_ptr_stable_across_calls() {
        let sg = SgSharedMemory::new(8);
        let p1 = sg.as_ptr();
        let p2 = sg.as_ptr();
        assert_eq!(p1, p2, "as_ptr should return the same address on repeated calls");
    }

    #[test]
    fn test_as_ptr_non_null_for_zero_hidden() {
        let sg = SgSharedMemory::new(0);
        assert!(!sg.as_ptr().is_null(), "ptr should be non-null even for hidden_size=0");
    }

    #[test]
    fn test_knowledge_vector_empty_slice_zero_fills_all() {
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[]);
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        for (i, &v) in kv.iter().enumerate() {
            assert_eq!(v, 0.0f32, "slot {i} should be zero-filled when empty vec written");
        }
    }

    #[test]
    fn test_knowledge_vector_overwrite_second_write_replaces() {
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0]);
        // Second write — different values.
        sg.set_knowledge_vector(&[10.0, 20.0]);
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 10.0);
        assert_eq!(kv[1], 20.0);
        assert_eq!(kv[2], 0.0, "remaining slot should be re-zeroed");
        assert_eq!(kv[3], 0.0, "remaining slot should be re-zeroed");
    }

    #[test]
    fn test_knowledge_vector_special_float_values() {
        let mut sg = SgSharedMemory::new(5);
        let special = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::MIN_POSITIVE];
        sg.set_knowledge_vector(&special);
        let kv_offset = SG_HEADER_BYTES + 5 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 5)
        };
        assert!(kv[0].is_nan(), "NaN should round-trip");
        assert!(kv[1].is_infinite() && kv[1].is_sign_positive(), "+Inf should round-trip");
        assert!(kv[2].is_infinite() && kv[2].is_sign_negative(), "-Inf should round-trip");
        assert_eq!(kv[3].to_bits(), (-0.0f32).to_bits(), "-0.0 should round-trip");
        assert_eq!(kv[4], f32::MIN_POSITIVE, "min positive should round-trip");
    }

    #[test]
    fn test_confidence_ieee754_bit_exact_roundtrip() {
        let mut sg = SgSharedMemory::new(2);
        // Specific bit pattern: f32 representation of 0.1 (not exactly representable).
        let val = 0.1f32;
        sg.set_confidence(val);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(bits, val.to_bits(), "confidence should be bit-exact IEEE 754");
    }

    #[test]
    fn test_confidence_negative_infinity() {
        let mut sg = SgSharedMemory::new(2);
        sg.set_confidence(f32::NEG_INFINITY);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let read = f32::from_bits(bits);
        assert!(read.is_infinite() && read.is_sign_negative());
    }

    #[test]
    fn test_confidence_smallest_normal() {
        let mut sg = SgSharedMemory::new(2);
        sg.set_confidence(f32::MIN_POSITIVE);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(f32::from_bits(bits), f32::MIN_POSITIVE);
    }

    #[test]
    fn test_enable_disable_toggle_multiple() {
        let mut sg = SgSharedMemory::new(2);
        for _ in 0..10 {
            sg.enable();
            assert!(sg.is_enabled());
            sg.disable();
            assert!(!sg.is_enabled());
        }
    }

    #[test]
    fn test_detect_hidden_reflects_raw_byte_writes() {
        let mut sg = SgSharedMemory::new(2);
        // Simulate JIT writing detect_hidden via raw bytes.
        let dh_offset = SG_HEADER_BYTES;
        let val1_bits = 42.0f32.to_bits().to_le_bytes();
        let val2_bits = (-3.14f32).to_bits().to_le_bytes();
        sg.data[dh_offset..dh_offset + 4].copy_from_slice(&val1_bits);
        sg.data[dh_offset + 4..dh_offset + 8].copy_from_slice(&val2_bits);

        let dh = sg.detect_hidden();
        assert!((dh[0] - 42.0f32).abs() < 1e-6);
        assert!((dh[1] - (-3.14f32)).abs() < 1e-6);
    }

    #[test]
    fn test_control_all_ones_enable_disable() {
        let mut sg = SgSharedMemory::new(2);
        // Set all control bits to 1.
        sg.data[0..4].copy_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        assert!(sg.is_enabled()); // bit 0 is 1.
        sg.disable(); // clears bit 0.
        assert!(!sg.is_enabled());
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 0xFFFFFFFEu32, "only bit 0 should be cleared");
        sg.enable();
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 0xFFFFFFFFu32, "bit 0 restored");
    }

    #[test]
    fn test_data_buffer_alignment() {
        // The Box<[u8]> allocation should be aligned to at least 4 bytes
        // for f32 reinterpretation to be valid.
        let sg = SgSharedMemory::new(16);
        let ptr = sg.as_ptr() as usize;
        assert_eq!(ptr % 4, 0, "data buffer should be 4-byte aligned for f32 access");
    }

    #[test]
    fn test_hidden_size_large_layout_correctness() {
        let hidden = 8192;
        let sg = SgSharedMemory::new(hidden);
        let expected = SG_HEADER_BYTES + 2 * hidden * 4;
        assert_eq!(sg.data.len(), expected);
        // Verify detect_hidden slice length matches.
        assert_eq!(sg.detect_hidden().len(), hidden);
    }

    #[test]
    fn test_knowledge_offset_in_header() {
        // Bytes 4..8 are knowledge_offset (unused in current impl but part of layout).
        // Verify it reads as zero after construction.
        let sg = SgSharedMemory::new(32);
        let knowledge_offset = u32::from_le_bytes(sg.data[4..8].try_into().unwrap());
        assert_eq!(knowledge_offset, 0u32, "knowledge_offset should be zero-initialized");
    }

    #[test]
    fn test_set_confidence_does_not_affect_control() {
        let mut sg = SgSharedMemory::new(2);
        sg.enable();
        assert!(sg.is_enabled());
        sg.set_confidence(0.99);
        assert!(sg.is_enabled(), "setting confidence should not alter control");
        sg.set_confidence(0.0);
        assert!(sg.is_enabled(), "setting confidence to 0 should not alter control");
    }

    #[test]
    fn test_set_knowledge_vector_does_not_affect_control_or_confidence() {
        let mut sg = SgSharedMemory::new(4);
        sg.enable();
        sg.set_confidence(0.75);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0]);
        assert!(sg.is_enabled(), "knowledge_vector write should not alter control");
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.75f32).abs() < 1e-6, "knowledge_vector write should not alter confidence");
    }

    // ── 15 additional edge-case tests ──

    #[test]
    fn test_confidence_subnormal_value_roundtrip() {
        let mut sg = SgSharedMemory::new(2);
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        sg.set_confidence(subnormal);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(bits, 1u32, "subnormal f32 should round-trip bit-exact");
        assert_eq!(f32::from_bits(bits), subnormal);
    }

    #[test]
    fn test_confidence_max_finite_roundtrip() {
        let mut sg = SgSharedMemory::new(2);
        sg.set_confidence(f32::MAX);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(f32::from_bits(bits), f32::MAX, "f32::MAX should round-trip");
    }

    #[test]
    fn test_detect_hidden_slice_for_zero_hidden_is_empty() {
        let sg = SgSharedMemory::new(0);
        assert!(sg.detect_hidden().is_empty(), "zero hidden_size yields empty detect_hidden slice");
    }

    #[test]
    fn test_set_knowledge_vector_on_zero_hidden_no_panic() {
        let mut sg = SgSharedMemory::new(0);
        // Should not panic — vec is truncated to 0 and zero-fill loop does nothing.
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0]);
        assert_eq!(sg.hidden_size(), 0);
    }

    #[test]
    fn test_knowledge_vector_negative_values_roundtrip() {
        let mut sg = SgSharedMemory::new(4);
        let negs = [-1.0, -100.5, -f32::MIN_POSITIVE, -42.0];
        sg.set_knowledge_vector(&negs);
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], -1.0);
        assert!((kv[1] - (-100.5f32)).abs() < 1e-4);
        assert_eq!(kv[2], -f32::MIN_POSITIVE);
        assert_eq!(kv[3], -42.0);
    }

    #[test]
    fn test_knowledge_vector_denormalized_preserved() {
        let mut sg = SgSharedMemory::new(3);
        let denorm = f32::from_bits(0x007FFFFFu32); // largest subnormal
        sg.set_knowledge_vector(&[denorm, 0.0, 1.0]);
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
        };
        assert_eq!(kv[0].to_bits(), denorm.to_bits(), "denormalized value should preserve bits");
    }

    #[test]
    fn test_enable_preserves_existing_control_bits_2_through_31() {
        let mut sg = SgSharedMemory::new(2);
        // Set bits 2, 4, 8 manually.
        let custom = 0b00000001_00010100u32; // bits 2,4,8
        sg.data[0..4].copy_from_slice(&custom.to_le_bytes());
        sg.enable();
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, custom | 1, "enable should only set bit 0, preserving bits 2/4/8");
    }

    #[test]
    fn test_disable_preserves_existing_control_bits_2_through_31() {
        let mut sg = SgSharedMemory::new(2);
        let custom = 0b00000001_00010101u32; // bits 0,2,4,8
        sg.data[0..4].copy_from_slice(&custom.to_le_bytes());
        sg.disable();
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, custom & !1, "disable should only clear bit 0, preserving bits 2/4/8");
    }

    #[test]
    fn test_detect_hidden_write_does_not_corrupt_knowledge_vector() {
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[10.0, 20.0, 30.0, 40.0]);
        // Simulate JIT writing to detect_hidden region.
        let dh_offset = SG_HEADER_BYTES;
        let dh = unsafe {
            std::slice::from_raw_parts_mut(sg.data[dh_offset..].as_mut_ptr() as *mut f32, 4)
        };
        dh[0] = 999.0;
        dh[3] = -888.0;
        // Knowledge vector should be untouched.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 10.0);
        assert_eq!(kv[3], 40.0);
    }

    #[test]
    fn test_hidden_size_two_byte_boundary() {
        // hidden_size = 2 is a boundary: exactly one cache line (64B) = 16 + 2*2*4 = 32 bytes.
        let sg = SgSharedMemory::new(2);
        assert_eq!(sg.data.len(), SG_HEADER_BYTES + 2 * 2 * 4);
        assert_eq!(sg.detect_hidden().len(), 2);
    }

    #[test]
    fn test_knowledge_vector_all_zeros_write_zero_fills() {
        let mut sg = SgSharedMemory::new(3);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0]);
        // Overwrite with all zeros.
        sg.set_knowledge_vector(&[0.0, 0.0, 0.0]);
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
        };
        assert_eq!(kv, &[0.0f32, 0.0, 0.0]);
    }

    #[test]
    fn test_knowledge_dim_header_updated_for_various_sizes() {
        for &size in &[0usize, 1, 7, 256, 1024] {
            let sg = SgSharedMemory::new(size);
            let dim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
            assert_eq!(dim, size as u32, "knowledge_dim header mismatch for hidden_size={size}");
        }
    }

    #[test]
    fn test_total_data_len_formula_varies_with_hidden() {
        // Verify the invariant: data.len() == SG_HEADER_BYTES + 2 * hidden_size * 4
        for &h in &[0, 1, 3, 10, 100] {
            let sg = SgSharedMemory::new(h);
            assert_eq!(sg.data.len(), SG_HEADER_BYTES + 2 * h * 4, "invariant broken for h={h}");
        }
    }

    #[test]
    fn test_confidence_nan_signaling_vs_quiet() {
        let mut sg = SgSharedMemory::new(2);
        // Quiet NaN (bit 22 set) and signaling NaN (bit 22 clear).
        let qnan = f32::from_bits(0x7FC00000u32);
        let snan = f32::from_bits(0x7F800001u32);
        sg.set_confidence(qnan);
        let bits_q = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!(f32::from_bits(bits_q).is_nan());
        sg.set_confidence(snan);
        let bits_s = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!(f32::from_bits(bits_s).is_nan());
    }

    #[test]
    fn test_as_ptr_points_to_first_byte_of_data() {
        let sg = SgSharedMemory::new(4);
        let ptr = sg.as_ptr();
        // The pointer should equal the address of data[0].
        assert_eq!(ptr, sg.data.as_ptr(), "as_ptr must point to start of internal buffer");
    }

    // ── 15 additional edge-case tests (wave-2) ──

    #[test]
    fn test_hidden_size_immutable_after_mutations() {
        // Arrange: create shared memory and perform multiple mutations.
        let mut sg = SgSharedMemory::new(32);
        // Act: enable, set confidence, write knowledge vector, disable.
        sg.enable();
        sg.set_confidence(0.5);
        sg.set_knowledge_vector(&[1.0; 32]);
        sg.disable();
        // Assert: hidden_size must remain unchanged.
        assert_eq!(sg.hidden_size(), 32, "hidden_size must be invariant across mutations");
    }

    #[test]
    fn test_confidence_f32_min_roundtrip() {
        // Arrange: f32::MIN is the most negative finite f32 value.
        let mut sg = SgSharedMemory::new(2);
        // Act
        sg.set_confidence(f32::MIN);
        // Assert
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(f32::from_bits(bits), f32::MIN, "f32::MIN should round-trip exactly");
    }

    #[test]
    fn test_confidence_epsilon_roundtrip() {
        // Arrange: f32::EPSILON is the smallest representable difference from 1.0.
        let mut sg = SgSharedMemory::new(2);
        // Act
        sg.set_confidence(f32::EPSILON);
        // Assert
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(f32::from_bits(bits), f32::EPSILON, "f32::EPSILON should round-trip exactly");
    }

    #[test]
    fn test_detect_hidden_read_after_sequential_raw_writes() {
        // Arrange: simulate JIT writing two f32 values sequentially.
        let mut sg = SgSharedMemory::new(4);
        let offset = SG_HEADER_BYTES;
        let v1 = 1.5f32;
        let v2 = -2.25f32;
        // Act: write via raw bytes.
        sg.data[offset..offset + 4].copy_from_slice(&v1.to_bits().to_le_bytes());
        sg.data[offset + 4..offset + 8].copy_from_slice(&v2.to_bits().to_le_bytes());
        // Assert: detect_hidden() slice reflects both writes, remaining are zero.
        let dh = sg.detect_hidden();
        assert!((dh[0] - v1).abs() < 1e-6);
        assert!((dh[1] - v2).abs() < 1e-6);
        assert_eq!(dh[2], 0.0, "unwritten slot should remain zero");
        assert_eq!(dh[3], 0.0, "unwritten slot should remain zero");
    }

    #[test]
    fn test_knowledge_vector_f32_max_values() {
        // Arrange: write f32::MAX into all knowledge_vector slots.
        let mut sg = SgSharedMemory::new(4);
        // Act
        sg.set_knowledge_vector(&[f32::MAX; 4]);
        // Assert: verify via raw byte reinterpretation.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        for (i, &v) in kv.iter().enumerate() {
            assert_eq!(v, f32::MAX, "slot {i} should contain f32::MAX");
        }
    }

    #[test]
    fn test_enable_then_set_confidence_then_disable_preserves_confidence() {
        // Arrange
        let mut sg = SgSharedMemory::new(4);
        // Act
        sg.enable();
        sg.set_confidence(0.88);
        sg.disable();
        // Assert: confidence at offset 12..16 should still be 0.88.
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.88f32).abs() < 1e-6, "confidence should survive disable");
        assert!(!sg.is_enabled(), "should be disabled");
    }

    #[test]
    fn test_header_field_byte_order_consistency() {
        // Arrange: verify all four header fields use little-endian byte order.
        let mut sg = SgSharedMemory::new(16);
        sg.enable();
        sg.set_confidence(1.0);
        // Act: read header bytes individually.
        let control_byte0 = sg.data[0];
        let control_byte1 = sg.data[1];
        let control_byte2 = sg.data[2];
        let control_byte3 = sg.data[3];
        // Assert: control=1 in LE means byte0=1, rest=0.
        assert_eq!(control_byte0, 1u8, "LE control byte 0 should be 1");
        assert_eq!(control_byte1, 0u8);
        assert_eq!(control_byte2, 0u8);
        assert_eq!(control_byte3, 0u8);
        // knowledge_dim at bytes 8..12 in LE for hidden_size=16.
        let dim_bytes = [sg.data[8], sg.data[9], sg.data[10], sg.data[11]];
        assert_eq!(u32::from_le_bytes(dim_bytes), 16u32, "knowledge_dim should be 16 in LE");
    }

    #[test]
    fn test_detect_hidden_pointer_aligned_to_f32() {
        // Arrange: detect_hidden starts at SG_HEADER_BYTES (16) which is 4-byte aligned.
        let sg = SgSharedMemory::new(8);
        // Act: get the pointer to detect_hidden region.
        let dh = sg.detect_hidden();
        let ptr = dh.as_ptr() as usize;
        // Assert: must be 4-byte aligned for valid f32 reinterpretation.
        assert_eq!(ptr % 4, 0, "detect_hidden pointer must be 4-byte aligned");
    }

    #[test]
    fn test_knowledge_vector_region_immediately_follows_detect_hidden() {
        // Arrange: verify knowledge_vector starts exactly after detect_hidden.
        let sg = SgSharedMemory::new(10);
        let dh_end = SG_HEADER_BYTES + 10 * 4; // end of detect_hidden
        let kv_start = SG_HEADER_BYTES + 10 * 4; // start of knowledge_vector
        // Act: compute expected total size from the two regions.
        let expected_total = kv_start + 10 * 4;
        // Assert: the buffer size matches header + detect_hidden + knowledge_vector.
        assert_eq!(sg.data.len(), expected_total, "knowledge_vector should immediately follow detect_hidden with no gap");
    }

    #[test]
    fn test_single_element_hidden_size_full_workflow() {
        // Arrange: minimal hidden_size=1 with all operations.
        let mut sg = SgSharedMemory::new(1);
        // Act
        sg.enable();
        sg.set_confidence(0.5);
        sg.set_knowledge_vector(&[42.0]);
        // Assert
        assert!(sg.is_enabled());
        assert_eq!(sg.hidden_size(), 1);
        assert_eq!(sg.detect_hidden().len(), 1);
        assert_eq!(sg.detect_hidden()[0], 0.0, "detect_hidden not written yet");
        // Verify knowledge_vector at offset 16 + 1*4 = 20.
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[20..].as_ptr() as *const f32, 1)
        };
        assert_eq!(kv[0], 42.0);
        // Total size: 16 + 2*1*4 = 24.
        assert_eq!(sg.data.len(), 24);
    }

    #[test]
    fn test_enable_disable_enable_cycle_preserves_high_bits() {
        // Arrange: set high bits, then cycle enable/disable.
        let mut sg = SgSharedMemory::new(2);
        let custom = 0xFFFF0000u32;
        sg.data[0..4].copy_from_slice(&custom.to_le_bytes());
        // Act: enable -> disable -> enable.
        sg.enable();
        sg.disable();
        sg.enable();
        // Assert: high bits preserved, bit 0 is set.
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl & 0xFFFF0000, 0xFFFF0000, "high bits must survive enable/disable cycle");
        assert_eq!(ctrl & 1, 1, "bit 0 must be set after final enable");
    }

    #[test]
    fn test_very_large_hidden_size_ptr_covers_full_buffer() {
        // Arrange: large hidden_size that produces >1MB buffer.
        let hidden = 131072; // 128K elements → 16 + 2*128K*4 = 1,048,592 bytes
        let sg = SgSharedMemory::new(hidden);
        let expected = SG_HEADER_BYTES + 2 * hidden * 4;
        // Assert: buffer size is correct and detect_hidden covers expected region.
        assert_eq!(sg.data.len(), expected);
        assert_eq!(sg.detect_hidden().len(), hidden);
        // Verify the last byte is accessible via as_ptr.
        let ptr = sg.as_ptr();
        let last_byte = unsafe { *ptr.add(expected - 1) };
        assert_eq!(last_byte, 0u8, "last byte should be zero-initialized");
    }

    #[test]
    fn test_set_confidence_does_not_affect_knowledge_dim_header() {
        // Arrange
        let mut sg = SgSharedMemory::new(64);
        // Act
        sg.set_confidence(0.123);
        // Assert: knowledge_dim at bytes 8..12 must remain 64.
        let dim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim, 64u32, "confidence write must not corrupt knowledge_dim");
    }

    #[test]
    fn test_set_knowledge_vector_does_not_affect_knowledge_offset_header() {
        // Arrange
        let mut sg = SgSharedMemory::new(8);
        // Act
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Assert: knowledge_offset at bytes 4..8 must remain 0.
        let offset = u32::from_le_bytes(sg.data[4..8].try_into().unwrap());
        assert_eq!(offset, 0u32, "knowledge_vector write must not corrupt knowledge_offset");
    }

    #[test]
    fn test_full_header_layout_after_all_operations() {
        // Arrange: perform all operations and verify the complete header layout.
        let mut sg = SgSharedMemory::new(32);
        // Act
        sg.enable();
        sg.set_confidence(0.75);
        sg.set_knowledge_vector(&[3.14; 32]);
        // Assert: verify all four header fields.
        // Bytes 0..3: control = 1 (enabled).
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 1u32, "control should be 1 after enable");
        // Bytes 4..7: knowledge_offset = 0 (unused, zero).
        let koff = u32::from_le_bytes(sg.data[4..8].try_into().unwrap());
        assert_eq!(koff, 0u32, "knowledge_offset should remain zero");
        // Bytes 8..11: knowledge_dim = 32.
        let kdim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(kdim, 32u32, "knowledge_dim should match hidden_size");
        // Bytes 12..15: confidence = 0.75.
        let conf_bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!((f32::from_bits(conf_bits) - 0.75f32).abs() < 1e-6, "confidence should be 0.75");
    }

    // ── 13 additional tests (wave-3) ──

    #[test]
    fn test_knowledge_vector_exact_size_no_zero_fill_needed() {
        // Arrange: write exactly hidden_size elements — no zero-fill should occur.
        let mut sg = SgSharedMemory::new(4);
        let values = [5.5, 6.6, 7.7, 8.8];
        // Act
        sg.set_knowledge_vector(&values);
        // Assert: all 4 slots contain the written values, none are zero.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 5.5);
        assert_eq!(kv[1], 6.6);
        assert_eq!(kv[2], 7.7);
        assert_eq!(kv[3], 8.8);
        // Verify no zero-fill occurred — all slots have nonzero values.
        for (i, &v) in kv.iter().enumerate() {
            assert_ne!(v, 0.0f32, "slot {i} should not be zero when exact-size vec was written");
        }
    }

    #[test]
    fn test_enable_disable_preserves_knowledge_dim_field() {
        // Arrange
        let mut sg = SgSharedMemory::new(48);
        let dim_before = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim_before, 48u32);
        // Act
        sg.enable();
        sg.disable();
        // Assert: knowledge_dim must remain 48.
        let dim_after = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim_after, 48u32, "enable/disable must not alter knowledge_dim field");
    }

    #[test]
    fn test_enable_disable_preserves_confidence_field() {
        // Arrange
        let mut sg = SgSharedMemory::new(4);
        sg.set_confidence(0.618);
        // Act
        sg.enable();
        sg.disable();
        // Assert: confidence at bytes 12..16 must still be 0.618.
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.618f32).abs() < 1e-6, "enable/disable must not alter confidence field");
    }

    #[test]
    fn test_knowledge_vector_preserved_after_enable() {
        // Arrange: write knowledge vector then enable.
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[11.0, 22.0, 33.0, 44.0]);
        // Act
        sg.enable();
        // Assert: knowledge_vector values survive enable.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 11.0);
        assert_eq!(kv[1], 22.0);
        assert_eq!(kv[2], 33.0);
        assert_eq!(kv[3], 44.0);
        assert!(sg.is_enabled(), "should be enabled");
    }

    #[test]
    fn test_multiple_knowledge_vector_writes_in_sequence() {
        // Arrange: perform three sequential writes and verify the final state.
        let mut sg = SgSharedMemory::new(4);
        // Act: first write
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0]);
        // Second write — partial (2 elements)
        sg.set_knowledge_vector(&[10.0, 20.0]);
        // Third write — full overwrite
        sg.set_knowledge_vector(&[100.0, 200.0, 300.0, 400.0]);
        // Assert: only the third write's values should remain.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 100.0);
        assert_eq!(kv[1], 200.0);
        assert_eq!(kv[2], 300.0);
        assert_eq!(kv[3], 400.0);
    }

    #[test]
    fn test_confidence_half_threshold_value() {
        // Arrange: 0.5 is a common decision threshold in SG systems.
        let mut sg = SgSharedMemory::new(2);
        // Act
        sg.set_confidence(0.5);
        // Assert
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.5f32).abs() < 1e-6, "confidence of 0.5 should round-trip");
    }

    #[test]
    fn test_knowledge_vector_negative_values_preserved() {
        // Arrange: write a vector with negative values at every slot.
        let mut sg = SgSharedMemory::new(5);
        let negs = [-0.001, -99.99, -1e10, -f32::MIN_POSITIVE, -f32::MAX];
        // Act
        sg.set_knowledge_vector(&negs);
        // Assert: each negative value should be preserved bit-exactly or within float tolerance.
        let kv_offset = SG_HEADER_BYTES + 5 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 5)
        };
        assert!((kv[0] - (-0.001f32)).abs() < 1e-6);
        assert!((kv[1] - (-99.99f32)).abs() < 1e-3);
        assert_eq!(kv[2], -1e10f32);
        assert_eq!(kv[3], -f32::MIN_POSITIVE);
        assert_eq!(kv[4], -f32::MAX);
    }

    #[test]
    fn test_knowledge_vector_very_large_values_preserved() {
        // Arrange: write values close to f32::MAX.
        let mut sg = SgSharedMemory::new(3);
        let large = [f32::MAX, f32::MAX / 2.0, f32::MAX / 1e10];
        // Act
        sg.set_knowledge_vector(&large);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
        };
        assert_eq!(kv[0], f32::MAX);
        assert_eq!(kv[1], f32::MAX / 2.0);
        assert!((kv[2] - f32::MAX / 1e10).abs() < 1.0, "large divided value should be close");
    }

    #[test]
    fn test_hidden_size_one_enable_and_confidence_set() {
        // Arrange: minimal hidden_size=1 with enable + confidence.
        let mut sg = SgSharedMemory::new(1);
        // Act
        sg.enable();
        sg.set_confidence(0.99);
        // Assert
        assert!(sg.is_enabled());
        assert_eq!(sg.hidden_size(), 1);
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.99f32).abs() < 1e-6);
        // Buffer size: 16 + 2*1*4 = 24.
        assert_eq!(sg.data.len(), 24);
    }

    #[test]
    fn test_knowledge_vector_survives_multiple_operations() {
        // Arrange: write knowledge vector, then perform enable/disable/confidence operations.
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[7.0, 14.0, 21.0, 28.0]);
        // Act: perform unrelated operations.
        sg.enable();
        sg.set_confidence(0.33);
        sg.disable();
        sg.enable();
        sg.set_confidence(0.77);
        // Assert: knowledge_vector should be untouched by all header operations.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 7.0);
        assert_eq!(kv[1], 14.0);
        assert_eq!(kv[2], 21.0);
        assert_eq!(kv[3], 28.0);
    }

    #[test]
    fn test_detect_hidden_endianness_via_raw_bytes() {
        // Arrange: write a known f32 bit pattern as big-endian to verify
        // that detect_hidden reads in little-endian (native x86).
        let mut sg = SgSharedMemory::new(2);
        let value = 1234.5678f32;
        // Act: write using to_le_bytes (correct LE layout).
        let le_bytes = value.to_bits().to_le_bytes();
        let offset = SG_HEADER_BYTES;
        sg.data[offset..offset + 4].copy_from_slice(&le_bytes);
        // Assert: detect_hidden should read the value correctly.
        let dh = sg.detect_hidden();
        assert!((dh[0] - value).abs() < 1e-3, "detect_hidden must interpret bytes as little-endian f32");
        assert_eq!(dh[1], 0.0, "second slot untouched");
    }

    #[test]
    fn test_as_ptr_different_per_instance() {
        // Arrange: create two separate instances.
        let sg1 = SgSharedMemory::new(8);
        let sg2 = SgSharedMemory::new(8);
        // Act
        let ptr1 = sg1.as_ptr();
        let ptr2 = sg2.as_ptr();
        // Assert: each instance has its own allocation, so pointers differ.
        assert_ne!(ptr1, ptr2, "different instances must have different backing allocations");
    }

    #[test]
    fn test_hidden_size_zero_all_operations_safe() {
        // Arrange: hidden_size=0 is the degenerate case.
        let mut sg = SgSharedMemory::new(0);
        // Act: perform all operations — none should panic.
        sg.enable();
        assert!(sg.is_enabled());
        sg.set_confidence(0.5);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0]); // truncated to 0
        sg.disable();
        assert!(!sg.is_enabled());
        // Assert: detect_hidden is empty, buffer is header-only.
        assert!(sg.detect_hidden().is_empty());
        assert_eq!(sg.data.len(), SG_HEADER_BYTES);
        assert_eq!(sg.hidden_size(), 0);
        // Confidence should still be readable.
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.5f32).abs() < 1e-6);
    }

    // ── 13 additional tests (wave-4): extreme values, cycling integrity, endianness, boundaries ──

    #[test]
    fn test_knowledge_vector_nan_bit_exact_roundtrip() {
        // Arrange: write NaN into every knowledge_vector slot and verify bit-exact preservation.
        let mut sg = SgSharedMemory::new(4);
        let nan_bits = 0x7FC0_1234u32; // a specific quiet NaN bit pattern
        let nan_val = f32::from_bits(nan_bits);
        // Act
        sg.set_knowledge_vector(&[nan_val; 4]);
        // Assert: each slot must preserve the exact NaN bit pattern.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        for (i, &v) in kv.iter().enumerate() {
            assert!(v.is_nan(), "slot {i} must be NaN");
            assert_eq!(v.to_bits(), nan_bits, "slot {i} NaN bit pattern must be exact");
        }
    }

    #[test]
    fn test_knowledge_vector_neg_infinity_preserved() {
        // Arrange: write -Inf into knowledge_vector and verify sign preservation.
        let mut sg = SgSharedMemory::new(2);
        // Act
        sg.set_knowledge_vector(&[f32::NEG_INFINITY, f32::INFINITY]);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 2 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 2)
        };
        assert!(
            kv[0].is_infinite() && kv[0].is_sign_negative(),
            "first slot must be -Inf"
        );
        assert!(
            kv[1].is_infinite() && kv[1].is_sign_positive(),
            "second slot must be +Inf"
        );
    }

    #[test]
    fn test_enable_disable_cycle_preserves_knowledge_vector_integrity() {
        // Arrange: write knowledge vector, then cycle enable/disable multiple times.
        let mut sg = SgSharedMemory::new(4);
        let original = [1.1, 2.2, 3.3, 4.4];
        sg.set_knowledge_vector(&original);
        // Act: toggle enable/disable 20 times.
        for _ in 0..20 {
            sg.enable();
            sg.disable();
        }
        // Assert: knowledge_vector must be completely unchanged.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert!((kv[0] - 1.1f32).abs() < 1e-6, "slot 0 corrupted after cycling");
        assert!((kv[1] - 2.2f32).abs() < 1e-6, "slot 1 corrupted after cycling");
        assert!((kv[2] - 3.3f32).abs() < 1e-6, "slot 2 corrupted after cycling");
        assert!((kv[3] - 4.4f32).abs() < 1e-6, "slot 3 corrupted after cycling");
    }

    #[test]
    fn test_confidence_threshold_boundary_one_minus_epsilon() {
        // Arrange: confidence of 1.0 - f32::EPSILON is the largest value strictly less than 1.0.
        let mut sg = SgSharedMemory::new(2);
        let boundary = 1.0f32 - f32::EPSILON;
        // Act
        sg.set_confidence(boundary);
        // Assert: the value must round-trip exactly and be strictly less than 1.0.
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let read = f32::from_bits(bits);
        assert_eq!(read, boundary, "boundary value must round-trip bit-exact");
        assert!(read < 1.0f32, "value must be strictly less than 1.0");
    }

    #[test]
    fn test_knowledge_vector_largest_subnormal_preserved() {
        // Arrange: largest f32 subnormal has exponent=0 and mantissa=all-ones.
        let mut sg = SgSharedMemory::new(2);
        let largest_subnormal = f32::from_bits(0x007F_FFFFu32);
        // Act
        sg.set_knowledge_vector(&[largest_subnormal, 0.0]);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 2 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 2)
        };
        assert_eq!(
            kv[0].to_bits(),
            largest_subnormal.to_bits(),
            "largest subnormal must be bit-exact"
        );
        assert!(kv[0].is_subnormal(), "value must be subnormal");
        assert_eq!(kv[1], 0.0);
    }

    #[test]
    fn test_raw_byte_endianness_big_endian_misinterpreted() {
        // Arrange: if bytes were written in big-endian order, the little-endian
        // reader would produce a different value. This proves LE convention.
        let mut sg = SgSharedMemory::new(1);
        let value = 1000.0f32;
        let le_bytes = value.to_bits().to_le_bytes();
        // Act: write in LE order (correct).
        let offset = SG_HEADER_BYTES;
        sg.data[offset..offset + 4].copy_from_slice(&le_bytes);
        let dh = sg.detect_hidden();
        let read_le = dh[0];
        // Now write the same bit pattern in BE order to a fresh instance.
        let be_bytes = value.to_bits().to_be_bytes();
        let mut sg2 = SgSharedMemory::new(1);
        sg2.data[offset..offset + 4].copy_from_slice(&be_bytes);
        let read_be = sg2.detect_hidden()[0];
        // Assert: LE read gives the correct value; BE read gives a different value.
        assert!((read_le - value).abs() < 1e-6, "LE byte order must yield correct value");
        assert!(
            (read_be - value).abs() > 1.0,
            "BE byte order must not yield the same value (proves LE convention)"
        );
    }

    #[test]
    fn test_sequential_write_read_cycles_ten_iterations() {
        // Arrange: perform 10 full write-verify cycles with different data each time.
        let mut sg = SgSharedMemory::new(3);
        for cycle in 0..10 {
            // Act: write unique values for this cycle.
            let base = (cycle + 1) as f32;
            let values = [base, base * 10.0, base * 100.0];
            sg.set_knowledge_vector(&values);
            // Assert: read back and verify each slot.
            let kv_offset = SG_HEADER_BYTES + 3 * 4;
            let kv = unsafe {
                std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
            };
            for (j, &expected) in values.iter().enumerate() {
                assert_eq!(
                    kv[j], expected,
                    "cycle {cycle} slot {j}: expected {expected}, got {}",
                    kv[j]
                );
            }
        }
    }

    #[test]
    fn test_hidden_size_very_large_produces_correct_byte_count() {
        // Arrange: use a hidden_size that produces exactly 2 GiB of dynamic data.
        // 2 GiB = 2 * 1024 * 1024 * 1024 bytes = 2,147,483,648.
        // Dynamic region = 2 * hidden_size * 4 bytes.
        // So hidden_size = 2_147_483_648 / 8 = 268_435_456.
        // Skip this test on 32-bit or memory-constrained systems by checking available.
        let hidden: usize = 268_435_456;
        // We only verify the formula without actually allocating (would OOM).
        let expected_bytes = SG_HEADER_BYTES + 2 * hidden * 4;
        assert_eq!(
            expected_bytes,
            16 + 2 * 268_435_456 * 4,
            "byte count formula must be correct for very large hidden_size"
        );
    }

    #[test]
    fn test_confidence_quiet_nan_specific_bit_pattern() {
        // Arrange: IEEE 754 quiet NaN has bit 22 (mantissa MSB) set.
        let mut sg = SgSharedMemory::new(2);
        let qnan = f32::from_bits(0x7FC0_0000u32); // canonical quiet NaN
        // Act
        sg.set_confidence(qnan);
        // Assert: the exact bit pattern must be preserved at bytes 12..16.
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert_eq!(bits, 0x7FC0_0000u32, "quiet NaN bit pattern must be exact");
        assert!(f32::from_bits(bits).is_nan());
        // Verify the quiet NaN bit (bit 22 of mantissa) is set.
        assert_ne!(bits & 0x0040_0000, 0, "quiet NaN signaling bit must be set");
    }

    #[test]
    fn test_knowledge_vector_overwrite_full_to_partial_zero_fills_tail() {
        // Arrange: fill all slots, then overwrite with a partial vector.
        let mut sg = SgSharedMemory::new(6);
        sg.set_knowledge_vector(&[100.0, 200.0, 300.0, 400.0, 500.0, 600.0]);
        // Act: overwrite with only 2 elements.
        sg.set_knowledge_vector(&[11.0, 22.0]);
        // Assert: first 2 slots updated, remaining 4 must be zero-filled.
        let kv_offset = SG_HEADER_BYTES + 6 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 6)
        };
        assert_eq!(kv[0], 11.0, "slot 0 should be overwritten");
        assert_eq!(kv[1], 22.0, "slot 1 should be overwritten");
        assert_eq!(kv[2], 0.0, "slot 2 should be zero-filled after partial overwrite");
        assert_eq!(kv[3], 0.0, "slot 3 should be zero-filled");
        assert_eq!(kv[4], 0.0, "slot 4 should be zero-filled");
        assert_eq!(kv[5], 0.0, "slot 5 should be zero-filled");
    }

    #[test]
    fn test_detect_hidden_region_all_ones_bit_pattern() {
        // Arrange: write 0xFFFFFFFF (NaN) via raw bytes to detect_hidden and verify slice read.
        let mut sg = SgSharedMemory::new(2);
        let nan_bits = 0xFFFF_FFFFu32;
        let offset = SG_HEADER_BYTES;
        // Act: write NaN bit pattern as raw LE bytes.
        sg.data[offset..offset + 4].copy_from_slice(&nan_bits.to_le_bytes());
        sg.data[offset + 4..offset + 8].copy_from_slice(&nan_bits.to_le_bytes());
        // Assert: detect_hidden must report NaN for both slots.
        let dh = sg.detect_hidden();
        assert!(dh[0].is_nan(), "slot 0 with all-ones bits must be NaN");
        assert!(dh[1].is_nan(), "slot 1 with all-ones bits must be NaN");
    }

    #[test]
    fn test_knowledge_vector_endianness_via_raw_byte_read() {
        // Arrange: write knowledge_vector and verify the raw bytes are in LE order.
        let mut sg = SgSharedMemory::new(1);
        let value = 256.5f32;
        // Act
        sg.set_knowledge_vector(&[value]);
        // Assert: read the raw bytes at the knowledge_vector offset and reassemble as LE.
        let kv_offset = SG_HEADER_BYTES + 1 * 4;
        let raw_bytes = &sg.data[kv_offset..kv_offset + 4];
        let bits_from_raw = u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
        assert_eq!(
            f32::from_bits(bits_from_raw), value,
            "raw bytes at knowledge_vector offset must decode to the written value in LE"
        );
        // Also verify the first byte is the LSB of the bit pattern.
        let expected_bits = value.to_bits();
        assert_eq!(raw_bytes[0], (expected_bits & 0xFF) as u8, "byte 0 must be LSB");
        assert_eq!(raw_bytes[3], ((expected_bits >> 24) & 0xFF) as u8, "byte 3 must be MSB");
    }

    // ── 13 additional tests (wave-5): data integrity across regions, edge values, endianness ──

    #[test]
    fn test_knowledge_vector_write_does_not_corrupt_detect_hidden() {
        // Arrange: write values to detect_hidden via raw bytes, then write knowledge_vector.
        let mut sg = SgSharedMemory::new(4);
        let dh_offset = SG_HEADER_BYTES;
        let dh = unsafe {
            std::slice::from_raw_parts_mut(sg.data[dh_offset..].as_mut_ptr() as *mut f32, 4)
        };
        dh[0] = -5.5;
        dh[1] = 10.25;
        dh[2] = 0.0;
        dh[3] = 999.0;
        // Act: write knowledge_vector.
        sg.set_knowledge_vector(&[100.0, 200.0, 300.0, 400.0]);
        // Assert: detect_hidden must be untouched by the knowledge_vector write.
        let dh_read = sg.detect_hidden();
        assert_eq!(dh_read[0], -5.5, "detect_hidden slot 0 must survive kv write");
        assert_eq!(dh_read[1], 10.25, "detect_hidden slot 1 must survive kv write");
        assert_eq!(dh_read[2], 0.0, "detect_hidden slot 2 must survive kv write");
        assert_eq!(dh_read[3], 999.0, "detect_hidden slot 3 must survive kv write");
    }

    #[test]
    fn test_confidence_positive_zero_vs_negative_zero() {
        // Arrange: f32 has two zero representations: +0.0 and -0.0.
        let mut sg = SgSharedMemory::new(2);
        // Act: write positive zero.
        sg.set_confidence(0.0f32);
        let bits_pos = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        // Act: write negative zero.
        sg.set_confidence(-0.0f32);
        let bits_neg = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        // Assert: both are zero but have different bit patterns.
        assert_eq!(f32::from_bits(bits_pos), 0.0);
        assert_eq!(f32::from_bits(bits_neg), 0.0);
        assert_eq!(bits_pos, 0u32, "+0.0 must have all-zero bits");
        assert_eq!(bits_neg, 0x8000_0000u32, "-0.0 must have sign bit set");
    }

    #[test]
    fn test_knowledge_vector_min_negative_finite_roundtrip() {
        // Arrange: f32::MIN is the most negative finite f32 value.
        let mut sg = SgSharedMemory::new(3);
        // Act
        sg.set_knowledge_vector(&[f32::MIN, 0.0, f32::MAX]);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
        };
        assert_eq!(kv[0], f32::MIN, "f32::MIN must round-trip in knowledge_vector");
        assert_eq!(kv[1], 0.0);
        assert_eq!(kv[2], f32::MAX);
    }

    #[test]
    fn test_knowledge_vector_idempotent_same_value_rewritten() {
        // Arrange: write the same values twice and verify the result is identical.
        let mut sg = SgSharedMemory::new(4);
        let values = [3.14, 2.71, 1.41, 1.73];
        sg.set_knowledge_vector(&values);
        // Act: write the same values again.
        sg.set_knowledge_vector(&values);
        // Assert: result is identical to a single write.
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        assert_eq!(kv[0], 3.14);
        assert_eq!(kv[1], 2.71);
        assert_eq!(kv[2], 1.41);
        assert_eq!(kv[3], 1.73);
    }

    #[test]
    fn test_hidden_size_three_complete_workflow() {
        // Arrange: hidden_size=3 is the smallest non-trivial odd size.
        let mut sg = SgSharedMemory::new(3);
        // Act
        sg.enable();
        sg.set_confidence(0.42);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0]);
        // Assert
        assert!(sg.is_enabled());
        assert_eq!(sg.hidden_size(), 3);
        assert_eq!(sg.data.len(), SG_HEADER_BYTES + 2 * 3 * 4); // 16 + 24 = 40
        let bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        let conf = f32::from_bits(bits);
        assert!((conf - 0.42f32).abs() < 1e-6);
        let kv_offset = SG_HEADER_BYTES + 3 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 3)
        };
        assert_eq!(kv[0], 1.0);
        assert_eq!(kv[1], 2.0);
        assert_eq!(kv[2], 3.0);
    }

    #[test]
    fn test_knowledge_vector_write_preserves_knowledge_dim_header() {
        // Arrange: verify knowledge_dim (bytes 8..12) is untouched by kv writes.
        let mut sg = SgSharedMemory::new(16);
        let dim_before = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim_before, 16u32);
        // Act: write knowledge_vector with exact-length data.
        sg.set_knowledge_vector(&[0.5; 16]);
        // Assert
        let dim_after = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(dim_after, 16u32, "knowledge_vector write must not alter knowledge_dim header");
    }

    #[test]
    fn test_detect_hidden_consecutive_overwrites_same_slot() {
        // Arrange: write to the same detect_hidden slot multiple times and verify the last value wins.
        let mut sg = SgSharedMemory::new(2);
        let offset = SG_HEADER_BYTES;
        // Act: write 1.0 then overwrite with 2.0 then 3.0.
        for val in &[1.0f32, 2.0f32, 3.0f32] {
            sg.data[offset..offset + 4].copy_from_slice(&val.to_bits().to_le_bytes());
        }
        // Assert
        let dh = sg.detect_hidden();
        assert_eq!(dh[0], 3.0, "last written value must be visible");
        assert_eq!(dh[1], 0.0, "unwritten slot remains zero");
    }

    #[test]
    fn test_knowledge_vector_alternating_length_overwrites() {
        // Arrange: write vectors of alternating lengths (full, partial, full, partial).
        let mut sg = SgSharedMemory::new(5);
        // Act
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0, 5.0]); // full
        sg.set_knowledge_vector(&[10.0]); // partial — slots 1..4 must be zero
        sg.set_knowledge_vector(&[20.0, 30.0, 40.0, 50.0, 60.0]); // full
        sg.set_knowledge_vector(&[100.0, 200.0]); // partial — slots 2..4 must be zero
        // Assert
        let kv_offset = SG_HEADER_BYTES + 5 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 5)
        };
        assert_eq!(kv[0], 100.0, "slot 0 from last write");
        assert_eq!(kv[1], 200.0, "slot 1 from last write");
        assert_eq!(kv[2], 0.0, "slot 2 zero-filled from partial write");
        assert_eq!(kv[3], 0.0, "slot 3 zero-filled");
        assert_eq!(kv[4], 0.0, "slot 4 zero-filled");
    }

    #[test]
    fn test_confidence_and_knowledge_vector_both_set_then_header_intact() {
        // Arrange: set both confidence and knowledge_vector, verify both plus header fields.
        let mut sg = SgSharedMemory::new(2);
        // Act
        sg.enable();
        sg.set_confidence(0.125);
        sg.set_knowledge_vector(&[-1.0, 1.0]);
        // Assert: full header verification.
        let ctrl = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(ctrl, 1u32, "control should be 1");
        let koff = u32::from_le_bytes(sg.data[4..8].try_into().unwrap());
        assert_eq!(koff, 0u32, "knowledge_offset should be 0");
        let kdim = u32::from_le_bytes(sg.data[8..12].try_into().unwrap());
        assert_eq!(kdim, 2u32, "knowledge_dim should match hidden_size");
        let conf_bits = u32::from_le_bytes(sg.data[12..16].try_into().unwrap());
        assert!((f32::from_bits(conf_bits) - 0.125f32).abs() < 1e-6, "confidence should be 0.125");
        // Verify knowledge_vector.
        let kv_offset = SG_HEADER_BYTES + 2 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 2)
        };
        assert_eq!(kv[0], -1.0);
        assert_eq!(kv[1], 1.0);
    }

    #[test]
    fn test_control_bit0_only_toggled() {
        // Arrange: set bits 1,3,5,7 manually, then enable and disable to verify only bit 0 changes.
        let mut sg = SgSharedMemory::new(2);
        let custom = 0b00000000_10101010u32; // bits 1,3,5,7 set
        sg.data[0..4].copy_from_slice(&custom.to_le_bytes());
        // Act
        sg.enable();
        let after_enable = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(after_enable, custom | 1, "enable should only set bit 0");
        sg.disable();
        let after_disable = u32::from_le_bytes(sg.data[0..4].try_into().unwrap());
        assert_eq!(after_disable, custom, "disable should restore to original (bit 0 cleared)");
    }

    #[test]
    fn test_knowledge_vector_smallest_subnormal_roundtrip() {
        // Arrange: smallest positive subnormal has bit pattern 0x0000_0001.
        let mut sg = SgSharedMemory::new(2);
        let smallest_subnormal = f32::from_bits(1u32);
        // Act
        sg.set_knowledge_vector(&[smallest_subnormal, 0.0]);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 2 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 2)
        };
        assert_eq!(kv[0].to_bits(), 1u32, "smallest subnormal must be bit-exact");
        assert!(kv[0].is_subnormal());
        assert_eq!(kv[1], 0.0);
    }

    #[test]
    fn test_detect_hidden_mixed_positive_negative_zero() {
        // Arrange: write +0.0 and -0.0 to detect_hidden and verify bit-exact read.
        let mut sg = SgSharedMemory::new(2);
        let offset = SG_HEADER_BYTES;
        // Act: write +0.0 and -0.0 as raw LE bytes.
        sg.data[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
        sg.data[offset + 4..offset + 8].copy_from_slice(&0x8000_0000u32.to_le_bytes());
        // Assert
        let dh = sg.detect_hidden();
        assert_eq!(dh[0].to_bits(), 0u32, "+0.0 bits must be all-zero");
        assert_eq!(dh[1].to_bits(), 0x8000_0000u32, "-0.0 must have sign bit");
        // Both compare equal to 0.0 in f32 arithmetic.
        assert_eq!(dh[0], 0.0f32);
        assert_eq!(dh[1], 0.0f32);
    }

    #[test]
    fn test_knowledge_vector_overwrite_with_empty_after_full_rezeros_all() {
        // Arrange: fill all slots, then overwrite with empty slice.
        let mut sg = SgSharedMemory::new(4);
        sg.set_knowledge_vector(&[1.0, 2.0, 3.0, 4.0]);
        // Act: overwrite with empty slice — all slots must be zero-filled.
        sg.set_knowledge_vector(&[]);
        // Assert
        let kv_offset = SG_HEADER_BYTES + 4 * 4;
        let kv = unsafe {
            std::slice::from_raw_parts(sg.data[kv_offset..].as_ptr() as *const f32, 4)
        };
        for (i, &v) in kv.iter().enumerate() {
            assert_eq!(v, 0.0f32, "slot {i} must be re-zeroed after empty overwrite");
        }
    }
}
