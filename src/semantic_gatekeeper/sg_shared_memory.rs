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
        let mut data = vec![0u8; total].into_boxed_slice();
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
}
