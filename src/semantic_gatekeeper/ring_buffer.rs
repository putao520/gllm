//! Gatekeeper Q-Tap ring buffer (SPEC §4.4).
//!
//! FusedAttentionLayer 的 Q-tap epilogue 将 `q_proj(hidden)[-1]` 写入此
//! 双缓冲区. SemanticGatekeeperCallback 在 JIT SgDetect Op 中读取.
//!
//! 同步协议: JIT 的 STG 指令 `release` 写 + bump step_index;
//! callback `acquire` 读 step_index 验证. Step 不匹配则返回 StaleQTap.

use std::sync::atomic::{AtomicU64, Ordering};

/// Ring buffer 读取错误.
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum QTapReadError {
    #[error("stale Q-tap: buffer step={buf_step} expected={expected_step}")]
    StaleQTap { buf_step: u64, expected_step: u64 },
    #[error("Q-tap buffer not initialized")]
    Uninitialized,
    #[error("Q-tap buffer capacity {capacity} < required {required}")]
    InsufficientCapacity { capacity: usize, required: usize },
}

/// 双缓冲 ring buffer.
///
/// 布局:
/// ```text
/// [header]
///   step_index: AtomicU64      — 最近一次写入完成的 step
///   write_cursor: AtomicU64    — 当前写入的 slot 索引 (双缓冲: 0 或 1)
///   q_dim: u32                 — 每条 Q 向量的维度
/// [data]
///   [u8; 2 × q_dim × dtype_size]  — 双 slot
/// ```
#[allow(dead_code)]
pub struct GatekeeperRingBuffer {
    step_index: AtomicU64,
    write_cursor: AtomicU64,
    q_dim: usize,
    q_element_bytes: usize,
    /// 原始字节缓冲区 (2 × q_dim × q_element_bytes).
    ///
    /// 使用 `Box<[u8]>` 保证地址稳定; JIT 通过 `data.as_ptr() as u64` 获取
    /// 设备可见指针 (CPU 路径直接用虚拟地址;GPU 路径由 executor 负责
    /// 将宿主页面映射为 pinned / device-accessible).
    data: Box<[u8]>,
}

impl GatekeeperRingBuffer {
    /// 构造 buffer. `q_dim` 为 `num_heads × head_dim` (Q 向量维度).
    pub fn new(q_dim: usize, q_element_bytes: usize) -> Self {
        let slot_bytes = q_dim * q_element_bytes;
        let total_bytes = 2 * slot_bytes;
        let data = vec![0u8; total_bytes].into_boxed_slice();
        Self {
            step_index: AtomicU64::new(0),
            write_cursor: AtomicU64::new(0),
            q_dim,
            q_element_bytes,
            data,
        }
    }

    /// 返回 data 区的宿主指针, 供 JIT codegen 作为 `QTapConfig.sink_ptr`.
    pub fn sink_ptr(&self) -> u64 {
        self.data.as_ptr() as u64
    }

    /// 返回 step_index AtomicU64 的宿主指针, 供 JIT codegen 作为
    /// `QTapConfig.step_index_ptr`.
    pub fn step_index_ptr(&self) -> u64 {
        &self.step_index as *const AtomicU64 as u64
    }

    /// Q 向量维度.
    pub fn q_dim(&self) -> usize {
        self.q_dim
    }

    /// 每个 Q 元素字节数 (通常 = `std::mem::size_of::<f32>()` 或 bf16/fp16
    /// 对应 2 字节).
    pub fn element_bytes(&self) -> usize {
        self.q_element_bytes
    }

    /// 每个 slot 的字节数 = `q_dim × element_bytes`.
    pub fn slot_bytes(&self) -> usize {
        self.q_dim * self.q_element_bytes
    }

    /// 读取最新写入的 Q 向量,验证 step 匹配.
    ///
    /// 使用 acquire 序保证观察到 JIT 写入的最新内容.
    pub fn read_latest(&self, expected_step: u64) -> Result<Vec<f32>, QTapReadError> {
        if self.q_dim == 0 {
            return Err(QTapReadError::Uninitialized);
        }
        let buf_step = self.step_index.load(Ordering::Acquire);
        if buf_step != expected_step {
            return Err(QTapReadError::StaleQTap {
                buf_step,
                expected_step,
            });
        }
        let cursor = self.write_cursor.load(Ordering::Acquire) as usize;
        let slot_idx = cursor % 2;
        let slot_bytes = self.slot_bytes();
        let start = slot_idx * slot_bytes;
        let end = start + slot_bytes;
        let raw = &self.data[start..end];
        decode_q_slot(raw, self.q_dim, self.q_element_bytes)
    }

    /// 供测试 / 非 JIT 路径手动写入 Q slot. 正式运行时由 JIT STG 指令直接
    /// 写 data 区,不通过此 API.
    #[cfg(any(test, feature = "sg-debug"))]
    pub fn debug_write(&self, q_bytes: &[u8], step: u64) -> Result<(), QTapReadError> {
        let slot_bytes = self.slot_bytes();
        if q_bytes.len() != slot_bytes {
            return Err(QTapReadError::InsufficientCapacity {
                capacity: slot_bytes,
                required: q_bytes.len(),
            });
        }
        let cursor = self.write_cursor.fetch_add(1, Ordering::Release) as usize;
        let slot_idx = (cursor + 1) % 2;
        let start = slot_idx * slot_bytes;
        let end = start + slot_bytes;
        // SAFETY: `data: Box<[u8]>` 的内部可变性通过这里的独占锁语义保证
        // (debug_write 只在单线程测试里调用).
        unsafe {
            let dst = self.data.as_ptr().add(start) as *mut u8;
            std::ptr::copy_nonoverlapping(q_bytes.as_ptr(), dst, slot_bytes);
        }
        // 修正 cursor 到新 slot
        self.write_cursor.store((cursor + 1) as u64, Ordering::Release);
        self.step_index.store(step, Ordering::Release);
        let _ = (start, end);
        Ok(())
    }
}

/// 将 raw slot 字节解码为 `Vec<f32>`.
///
/// 支持 4 字节 (F32) 和 2 字节 (F16). 其他宽度返回 `InsufficientCapacity`
/// (作为显式错误而非静默降级,符合 NO_SILENT_FALLBACK).
///
/// BCE-20260626-CC-002/004 根治：偏移从 `element_bytes` 派生，
/// 消除硬编码 `i*4`/`i*2`；decode 逻辑复用共享 `decode::decode_slice_to_f32`。
fn decode_q_slot(
    raw: &[u8],
    q_dim: usize,
    element_bytes: usize,
) -> Result<Vec<f32>, QTapReadError> {
    // element_bytes → DType 映射（Q-tap ring buffer 仅承载 F32/F16）。
    let dtype = match element_bytes {
        4 => gllm_kernels::types::DType::F32,
        2 => gllm_kernels::types::DType::F16,
        _ => {
            return Err(QTapReadError::InsufficientCapacity {
                capacity: element_bytes,
                required: 4,
            })
        }
    };
    super::decode::decode_slice_to_f32(raw, q_dim, dtype).map_err(|e| match e {
        super::decode::DecodeError::ByteLengthMismatch { actual, expected } => {
            QTapReadError::InsufficientCapacity {
                capacity: actual,
                required: expected,
            }
        }
        other => QTapReadError::InsufficientCapacity {
            capacity: 0,
            required: other.to_string().len(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ── Construction & accessors ──

    #[test]
    fn new_buffer_accessors() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        assert_eq!(rb.q_dim(), 4);
        assert_eq!(rb.element_bytes(), 4);
        assert_eq!(rb.slot_bytes(), 16);
        assert_ne!(rb.sink_ptr(), 0);
        assert_ne!(rb.step_index_ptr(), 0);
    }

    #[test]
    fn new_buffer_16bit() {
        let rb = GatekeeperRingBuffer::new(8, 2);
        assert_eq!(rb.element_bytes(), 2);
        assert_eq!(rb.slot_bytes(), 16);
    }

    // ── read_latest errors ──

    #[test]
    fn read_uninitialized_when_zero_dim() {
        let rb = GatekeeperRingBuffer::new(0, 4);
        let err = rb.read_latest(0).unwrap_err();
        assert!(matches!(err, QTapReadError::Uninitialized));
    }

    #[test]
    fn read_stale_when_step_mismatch() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        let err = rb.read_latest(1).unwrap_err();
        assert!(matches!(err, QTapReadError::StaleQTap { .. }));
    }

    // ── debug_write + read_latest roundtrip (F32) ──

    #[test]
    fn write_read_f32_roundtrip() {
        let rb = GatekeeperRingBuffer::new(3, 4);
        let data = f32_bytes(&[1.0, 2.0, 3.0]);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(q.len(), 3);
        assert!((q[0] - 1.0).abs() < 1e-6);
        assert!((q[1] - 2.0).abs() < 1e-6);
        assert!((q[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn write_twice_reads_latest() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[3.0, 4.0]), 2).unwrap();
        let q = rb.read_latest(2).unwrap();
        assert!((q[0] - 3.0).abs() < 1e-6);
        assert!((q[1] - 4.0).abs() < 1e-6);
    }

    // ── debug_write size mismatch ──

    #[test]
    fn write_wrong_size_errors() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        let too_short = f32_bytes(&[1.0]); // only 4 bytes, need 8
        let err = rb.debug_write(&too_short, 1).unwrap_err();
        assert!(matches!(err, QTapReadError::InsufficientCapacity { .. }));
    }

    // ── decode_q_slot via 16-bit path ──

    #[test]
    fn decode_f16_slot() {
        let rb = GatekeeperRingBuffer::new(2, 2);
        let f16_1 = half::f16::from_f32(1.5);
        let f16_2 = half::f16::from_f32(-2.0);
        let mut raw = Vec::new();
        raw.extend_from_slice(&f16_1.to_le_bytes());
        raw.extend_from_slice(&f16_2.to_le_bytes());
        rb.debug_write(&raw, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!((q[0] - 1.5).abs() < 0.01);
        assert!((q[1] - (-2.0)).abs() < 0.01);
    }

    // ── QTapReadError display ──

    #[test]
    fn error_display_messages() {
        let e = QTapReadError::StaleQTap { buf_step: 1, expected_step: 2 };
        assert!(e.to_string().contains("stale"));
        let e = QTapReadError::Uninitialized;
        assert!(e.to_string().contains("not initialized"));
        let e = QTapReadError::InsufficientCapacity { capacity: 4, required: 8 };
        assert!(e.to_string().contains("capacity"));
    }

    // ── QTapReadError Debug formatting ──

    #[test]
    fn error_debug_formatting() {
        let e = QTapReadError::StaleQTap { buf_step: 10, expected_step: 20 };
        let debug_str = format!("{:?}", e);
        assert!(debug_str.contains("StaleQTap"));
        assert!(debug_str.contains("10"));
        assert!(debug_str.contains("20"));

        let e = QTapReadError::Uninitialized;
        assert!(format!("{:?}", e).contains("Uninitialized"));

        let e = QTapReadError::InsufficientCapacity { capacity: 16, required: 32 };
        let debug_str = format!("{:?}", e);
        assert!(debug_str.contains("InsufficientCapacity"));
        assert!(debug_str.contains("16"));
        assert!(debug_str.contains("32"));
    }

    // ── QTapReadError Clone ──

    #[test]
    fn error_clone_preserves_data() {
        let original = QTapReadError::StaleQTap { buf_step: 5, expected_step: 7 };
        let cloned = original.clone();
        assert_eq!(original.to_string(), cloned.to_string());
    }

    // ── QTapReadError display contains actual step values ──

    #[test]
    fn stale_error_contains_step_values() {
        let e = QTapReadError::StaleQTap { buf_step: 42, expected_step: 99 };
        let msg = e.to_string();
        assert!(msg.contains("42"), "message should contain buf_step=42");
        assert!(msg.contains("99"), "message should contain expected_step=99");
    }

    #[test]
    fn capacity_error_contains_values() {
        let e = QTapReadError::InsufficientCapacity { capacity: 64, required: 128 };
        let msg = e.to_string();
        assert!(msg.contains("64"), "message should contain capacity=64");
        assert!(msg.contains("128"), "message should contain required=128");
    }

    // ── decode_q_slot: unsupported element_bytes ──

    #[test]
    fn decode_q_slot_unsupported_element_bytes_1() {
        let raw = vec![0u8; 8];
        let result = decode_q_slot(&raw, 2, 1);
        assert!(matches!(
            result,
            Err(QTapReadError::InsufficientCapacity { capacity: 1, required: 4 })
        ));
    }

    #[test]
    fn decode_q_slot_unsupported_element_bytes_3() {
        let raw = vec![0u8; 9];
        let result = decode_q_slot(&raw, 3, 3);
        assert!(matches!(
            result,
            Err(QTapReadError::InsufficientCapacity { capacity: 3, required: 4 })
        ));
    }

    #[test]
    fn decode_q_slot_unsupported_element_bytes_8() {
        let raw = vec![0u8; 16];
        let result = decode_q_slot(&raw, 2, 8);
        assert!(matches!(
            result,
            Err(QTapReadError::InsufficientCapacity { capacity: 8, required: 4 })
        ));
    }

    // ── decode_q_slot: F32 direct unit test ──

    #[test]
    fn decode_q_slot_f32_direct() {
        let vals = &[10.5f32, -3.25, 0.0, 100.0];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = decode_q_slot(&raw, 4, 4).unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0] - 10.5).abs() < 1e-6);
        assert!((result[1] - (-3.25)).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
        assert!((result[3] - 100.0).abs() < 1e-6);
    }

    // ── decode_q_slot: F16 direct unit test ──

    #[test]
    fn decode_q_slot_f16_direct() {
        let f16_vals = [half::f16::from_f32(0.5), half::f16::from_f32(-1.0)];
        let raw: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = decode_q_slot(&raw, 2, 2).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - (-1.0)).abs() < 0.01);
    }

    // ── Pointer stability: sink_ptr and step_index_ptr are distinct ──

    #[test]
    fn pointers_are_distinct() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        let sink = rb.sink_ptr();
        let step = rb.step_index_ptr();
        assert_ne!(sink, 0, "sink_ptr must not be null");
        assert_ne!(step, 0, "step_index_ptr must not be null");
        assert_ne!(
            sink, step,
            "sink_ptr and step_index_ptr must point to different memory"
        );
    }

    // ── Double-buffer slot isolation ──

    #[test]
    fn double_buffer_slot_isolation() {
        // Write first value, then second. Reading second value must not
        // contain remnants of the first.
        let rb = GatekeeperRingBuffer::new(2, 4);
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[5.0, 6.0]), 2).unwrap();
        let q = rb.read_latest(2).unwrap();
        assert!((q[0] - 5.0).abs() < 1e-6, "should read latest slot data");
        assert!((q[1] - 6.0).abs() < 1e-6, "should read latest slot data");
    }

    // ── Triple write verifies cursor wraps correctly ──

    #[test]
    fn triple_write_wrap_around() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[1.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[2.0]), 2).unwrap();
        rb.debug_write(&f32_bytes(&[3.0]), 3).unwrap();
        // After 3 writes the latest step is 3
        let q = rb.read_latest(3).unwrap();
        assert_eq!(q.len(), 1);
        assert!((q[0] - 3.0).abs() < 1e-6);
    }

    // ── Step mismatch after multiple writes ──

    #[test]
    fn stale_step_after_two_writes() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 10).unwrap();
        rb.debug_write(&f32_bytes(&[3.0, 4.0]), 11).unwrap();
        // Requesting step=10 is stale; only step=11 is current
        let err = rb.read_latest(10).unwrap_err();
        assert!(matches!(
            err,
            QTapReadError::StaleQTap {
                buf_step: 11,
                expected_step: 10
            }
        ));
    }

    // ── Large q_dim buffer ──

    #[test]
    fn large_q_dim_construction_and_roundtrip() {
        let dim = 1024;
        let rb = GatekeeperRingBuffer::new(dim, 4);
        assert_eq!(rb.slot_bytes(), dim * 4);
        let vals: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5).collect();
        let data = f32_bytes(&vals);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(q.len(), dim);
        for i in 0..dim {
            assert!(
                (q[i] - i as f32 * 0.5).abs() < 1e-4,
                "mismatch at index {}: got {} expected {}",
                i,
                q[i],
                i as f32 * 0.5
            );
        }
    }

    // ── Zero element_bytes slot_bytes calculation ──

    #[test]
    fn slot_bytes_with_various_configs() {
        // slot_bytes = q_dim * element_bytes
        let rb = GatekeeperRingBuffer::new(128, 4);
        assert_eq!(rb.slot_bytes(), 512);
        let rb = GatekeeperRingBuffer::new(256, 2);
        assert_eq!(rb.slot_bytes(), 512);
        let rb = GatekeeperRingBuffer::new(1, 4);
        assert_eq!(rb.slot_bytes(), 4);
    }

    // ── F16 roundtrip with negative and extreme values ──

    #[test]
    fn f16_roundtrip_with_special_values() {
        let rb = GatekeeperRingBuffer::new(3, 2);
        let f16_vals = [
            half::f16::from_f32(0.0),
            half::f16::from_f32(-0.0),
            half::f16::from_f32(65504.0), // max f16
        ];
        let raw: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        rb.debug_write(&raw, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!((q[0] - 0.0).abs() < 0.01);
        assert!((q[1]).abs() < 0.01, "negative zero should decode to ~0.0");
        assert!((q[2] - 65504.0).abs() < 1.0, "max f16 roundtrip");
    }

    // ── debug_write oversized payload ──

    #[test]
    fn write_oversized_payload_errors() {
        let rb = GatekeeperRingBuffer::new(2, 4); // slot_bytes = 8
        let oversized = f32_bytes(&[1.0, 2.0, 3.0]); // 12 bytes, need 8
        let err = rb.debug_write(&oversized, 1).unwrap_err();
        assert!(matches!(
            err,
            QTapReadError::InsufficientCapacity {
                capacity: 8,
                required: 12
            }
        ));
    }

    // ── read_latest with uninitialized (zero q_dim) different steps ──

    #[test]
    fn read_uninitialized_any_step() {
        let rb = GatekeeperRingBuffer::new(0, 4);
        // Uninitialized check fires before step check, regardless of expected_step
        let err = rb.read_latest(0).unwrap_err();
        assert!(matches!(err, QTapReadError::Uninitialized));
        let err = rb.read_latest(999).unwrap_err();
        assert!(matches!(err, QTapReadError::Uninitialized));
    }

    // ── Sink pointer points to data start (non-zero data) ──

    #[test]
    fn sink_ptr_after_write_still_valid() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        let ptr_before = rb.sink_ptr();
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 1).unwrap();
        let ptr_after = rb.sink_ptr();
        // Pointer must remain stable (Box does not move)
        assert_eq!(ptr_before, ptr_after);
    }

    // ── decode_q_slot: zero-length F32 ──

    #[test]
    fn decode_q_slot_zero_elements_f32() {
        let raw: Vec<u8> = vec![];
        let result = decode_q_slot(&raw, 0, 4).unwrap();
        assert!(result.is_empty());
    }

    // ── decode_q_slot: zero-length F16 ──

    #[test]
    fn decode_q_slot_zero_elements_f16() {
        let raw: Vec<u8> = vec![];
        let result = decode_q_slot(&raw, 0, 2).unwrap();
        assert!(result.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════
    //  New tests
    // ═══════════════════════════════════════════════════════════════

    // ── QTapReadError PartialEq ──

    #[test]
    fn error_partial_eq_stale() {
        let a = QTapReadError::StaleQTap { buf_step: 1, expected_step: 2 };
        let b = QTapReadError::StaleQTap { buf_step: 1, expected_step: 2 };
        assert_eq!(a, b);
        let c = QTapReadError::StaleQTap { buf_step: 1, expected_step: 3 };
        assert_ne!(a, c);
    }

    #[test]
    fn error_partial_eq_uninitialized() {
        assert_eq!(QTapReadError::Uninitialized, QTapReadError::Uninitialized);
        assert_ne!(
            QTapReadError::Uninitialized,
            QTapReadError::StaleQTap { buf_step: 0, expected_step: 0 }
        );
    }

    #[test]
    fn error_partial_eq_capacity() {
        let a = QTapReadError::InsufficientCapacity { capacity: 4, required: 8 };
        let b = QTapReadError::InsufficientCapacity { capacity: 4, required: 8 };
        assert_eq!(a, b);
        let c = QTapReadError::InsufficientCapacity { capacity: 4, required: 16 };
        assert_ne!(a, c);
    }

    // ── Construction with element_bytes=0 ──

    #[test]
    fn new_buffer_zero_element_bytes() {
        let rb = GatekeeperRingBuffer::new(4, 0);
        assert_eq!(rb.q_dim(), 4);
        assert_eq!(rb.element_bytes(), 0);
        assert_eq!(rb.slot_bytes(), 0);
    }

    // ── Construction with q_dim=0 and element_bytes=0 ──

    #[test]
    fn new_buffer_both_zero() {
        let rb = GatekeeperRingBuffer::new(0, 0);
        assert_eq!(rb.q_dim(), 0);
        assert_eq!(rb.element_bytes(), 0);
        assert_eq!(rb.slot_bytes(), 0);
    }

    // ── slot_bytes for q_dim=1, element_bytes=1 ──

    #[test]
    fn slot_bytes_minimal() {
        let rb = GatekeeperRingBuffer::new(1, 1);
        assert_eq!(rb.slot_bytes(), 1);
    }

    // ── debug_write with zero-length payload into zero-slot buffer ──

    #[test]
    fn write_zero_bytes_to_zero_slot() {
        let rb = GatekeeperRingBuffer::new(0, 4);
        let empty: Vec<u8> = vec![];
        let result = rb.debug_write(&empty, 1);
        assert!(result.is_ok(), "zero-length write into zero-slot buffer should succeed");
    }

    // ── debug_write mismatched size on zero-capacity buffer ──

    #[test]
    fn write_nonzero_to_zero_slot_errors() {
        let rb = GatekeeperRingBuffer::new(0, 4);
        let data = f32_bytes(&[1.0]);
        let err = rb.debug_write(&data, 1).unwrap_err();
        assert_eq!(
            err,
            QTapReadError::InsufficientCapacity { capacity: 0, required: 4 }
        );
    }

    // ── read_latest with u64::MAX step ──

    #[test]
    fn read_stale_with_max_step() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        let err = rb.read_latest(u64::MAX).unwrap_err();
        assert_eq!(
            err,
            QTapReadError::StaleQTap { buf_step: 0, expected_step: u64::MAX }
        );
    }

    // ── Multiple consecutive reads of same slot ──

    #[test]
    fn consecutive_reads_same_slot() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        rb.debug_write(&f32_bytes(&[7.0, 8.0]), 5).unwrap();

        let q1 = rb.read_latest(5).unwrap();
        let q2 = rb.read_latest(5).unwrap();
        assert_eq!(q1, q2, "consecutive reads must return identical data");
    }

    // ── read_latest returns StaleQTap after overwrite ──

    #[test]
    fn read_old_step_fails_after_overwrite() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[1.0]), 1).unwrap();
        // step=1 is readable
        assert!(rb.read_latest(1).is_ok());
        rb.debug_write(&f32_bytes(&[2.0]), 2).unwrap();
        // step=1 is now stale
        let err = rb.read_latest(1).unwrap_err();
        assert_eq!(err, QTapReadError::StaleQTap { buf_step: 2, expected_step: 1 });
    }

    // ── decode_q_slot with single F32 element ──

    #[test]
    fn decode_q_slot_single_f32() {
        let raw = (-42.5f32).to_le_bytes().to_vec();
        let result = decode_q_slot(&raw, 1, 4).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-42.5)).abs() < 1e-6);
    }

    // ── decode_q_slot with single F16 element ──

    #[test]
    fn decode_q_slot_single_f16() {
        let raw = half::f16::from_f32(3.75).to_le_bytes().to_vec();
        let result = decode_q_slot(&raw, 1, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.75).abs() < 0.01);
    }

    // ── decode_q_slot unsupported element_bytes=0 ──

    #[test]
    fn decode_q_slot_unsupported_zero_bytes() {
        let raw: Vec<u8> = vec![];
        let result = decode_q_slot(&raw, 0, 0);
        assert_eq!(
            result,
            Err(QTapReadError::InsufficientCapacity { capacity: 0, required: 4 })
        );
    }

    // ── decode_q_slot unsupported element_bytes=5 ──

    #[test]
    fn decode_q_slot_unsupported_element_bytes_5() {
        let raw = vec![0u8; 10];
        let result = decode_q_slot(&raw, 2, 5);
        assert_eq!(
            result,
            Err(QTapReadError::InsufficientCapacity { capacity: 5, required: 4 })
        );
    }

    // ── F16 decode with negative max ──

    #[test]
    fn f16_roundtrip_negative_max() {
        let rb = GatekeeperRingBuffer::new(1, 2);
        let raw = half::f16::from_f32(-65504.0).to_le_bytes().to_vec();
        rb.debug_write(&raw, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!((q[0] - (-65504.0)).abs() < 1.0, "negative f16 max roundtrip");
    }

    // ── F32 decode with special float values (infinity, negative infinity) ──

    #[test]
    fn f32_roundtrip_special_floats() {
        let rb = GatekeeperRingBuffer::new(3, 4);
        let vals = [f32::INFINITY, f32::NEG_INFINITY, 0.0f32];
        let data = f32_bytes(&vals);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!(q[0].is_infinite() && q[0].is_sign_positive());
        assert!(q[1].is_infinite() && q[1].is_sign_negative());
        assert_eq!(q[2], 0.0);
    }

    // ── Four writes exercise both slots twice ──

    #[test]
    fn four_writes_double_wrap() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[1.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[2.0]), 2).unwrap();
        rb.debug_write(&f32_bytes(&[3.0]), 3).unwrap();
        rb.debug_write(&f32_bytes(&[4.0]), 4).unwrap();
        let q = rb.read_latest(4).unwrap();
        assert!((q[0] - 4.0).abs() < 1e-6);
        // Older steps are stale
        assert!(rb.read_latest(3).is_err());
    }

    // ── Error clone for each variant ──

    #[test]
    fn clone_all_error_variants() {
        let e1 = QTapReadError::StaleQTap { buf_step: 1, expected_step: 2 };
        assert_eq!(e1.clone(), e1);

        let e2 = QTapReadError::Uninitialized;
        assert_eq!(e2.clone(), e2);

        let e3 = QTapReadError::InsufficientCapacity { capacity: 4, required: 8 };
        assert_eq!(e3.clone(), e3);
    }

    // ── sink_ptr alignment: data starts at offset 0 of the Box ──

    #[test]
    fn sink_ptr_is_data_start() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        // Write to step 1 and read back; sink_ptr should point to
        // the beginning of the data region which contains both slots.
        assert_ne!(rb.sink_ptr(), 0);
        // Writing should not change the pointer.
        let ptr = rb.sink_ptr();
        rb.debug_write(&f32_bytes(&[1.0, 2.0, 3.0, 4.0]), 1).unwrap();
        assert_eq!(rb.sink_ptr(), ptr);
    }

    // ── Large q_dim with F16 ──

    #[test]
    fn large_q_dim_f16_roundtrip() {
        let dim = 512;
        let rb = GatekeeperRingBuffer::new(dim, 2);
        assert_eq!(rb.slot_bytes(), dim * 2);
        let f16_vals: Vec<u8> = (0..dim)
            .flat_map(|i| half::f16::from_f32(i as f32).to_le_bytes())
            .collect();
        rb.debug_write(&f16_vals, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(q.len(), dim);
        for i in 0..dim {
            assert!(
                (q[i] - i as f32).abs() < 0.5,
                "mismatch at index {}: got {} expected {}",
                i,
                q[i],
                i as f32
            );
        }
    }

    // ── step_index_ptr distinct per buffer instance ──

    #[test]
    fn step_index_ptr_unique_per_instance() {
        let rb1 = GatekeeperRingBuffer::new(2, 4);
        let rb2 = GatekeeperRingBuffer::new(2, 4);
        assert_ne!(rb1.step_index_ptr(), rb2.step_index_ptr());
    }

    // ── sink_ptr distinct per buffer instance ──

    #[test]
    fn sink_ptr_unique_per_instance() {
        let rb1 = GatekeeperRingBuffer::new(2, 4);
        let rb2 = GatekeeperRingBuffer::new(2, 4);
        assert_ne!(rb1.sink_ptr(), rb2.sink_ptr());
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional edge-case tests (wave-2)
    // ═══════════════════════════════════════════════════════════════

    // ── F32 NaN roundtrip ──

    #[test]
    fn f32_roundtrip_nan() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        let data = f32_bytes(&[f32::NAN]);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!(q[0].is_nan(), "NaN must roundtrip as NaN");
    }

    // ── F32 subnormal (smallest positive subnormal) roundtrip ──

    #[test]
    fn f32_roundtrip_subnormal() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let data = f32_bytes(&[subnormal]);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(q[0].to_bits(), subnormal.to_bits());
    }

    // ── F32 largest negative finite value roundtrip ──

    #[test]
    fn f32_roundtrip_min_positive_and_max_negative() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        let vals = [f32::MIN_POSITIVE, f32::MIN]; // smallest normalized positive + most negative
        let data = f32_bytes(&vals);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(q[0].to_bits(), f32::MIN_POSITIVE.to_bits());
        assert_eq!(q[1].to_bits(), f32::MIN.to_bits());
    }

    // ── StaleQTap error carries exact u64 boundary values ──

    #[test]
    fn stale_error_exact_boundary_values() {
        let e = QTapReadError::StaleQTap { buf_step: 0, expected_step: 1 };
        let msg = e.to_string();
        assert!(msg.contains("0") && msg.contains("1"), "must contain both step values");
    }

    // ── debug_write returns Ok with correct step stored ──

    #[test]
    fn write_stores_step_zero() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[99.0]), 0).unwrap();
        let q = rb.read_latest(0).unwrap();
        assert!((q[0] - 99.0).abs() < 1e-6);
    }

    // ── Double buffer: old slot data does not leak after 3 writes ──

    #[test]
    fn old_slot_not_readable_after_three_writes() {
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[10.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[20.0]), 2).unwrap();
        rb.debug_write(&f32_bytes(&[30.0]), 3).unwrap();
        // Only step=3 is valid; step=1 and step=2 are stale
        assert!(rb.read_latest(1).is_err());
        assert!(rb.read_latest(2).is_err());
        let q = rb.read_latest(3).unwrap();
        assert!((q[0] - 30.0).abs() < 1e-6);
    }

    // ── QTapReadError is Send + Sync (compile-time check) ──

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QTapReadError>();
    }

    // ── GatekeeperRingBuffer with q_dim=1 element_bytes=2 (minimal F16) ──

    #[test]
    fn minimal_f16_roundtrip() {
        let rb = GatekeeperRingBuffer::new(1, 2);
        let raw = half::f16::from_f32(-0.5).to_le_bytes().to_vec();
        rb.debug_write(&raw, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert!((q[0] - (-0.5)).abs() < 0.01);
    }

    // ── decode_q_slot with f32 MAX value ──

    #[test]
    fn decode_q_slot_f32_max() {
        let raw = f32::MAX.to_le_bytes().to_vec();
        let result = decode_q_slot(&raw, 1, 4).unwrap();
        assert_eq!(result[0].to_bits(), f32::MAX.to_bits());
    }

    // ── decode_q_slot with f32 epsilon ──

    #[test]
    fn decode_q_slot_f32_epsilon() {
        let raw = f32::EPSILON.to_le_bytes().to_vec();
        let result = decode_q_slot(&raw, 1, 4).unwrap();
        assert_eq!(result[0].to_bits(), f32::EPSILON.to_bits());
    }

    // ── QTapReadError source is None (no chained error) ──

    #[test]
    fn error_source_is_none() {
        use std::error::Error;
        let e = QTapReadError::StaleQTap { buf_step: 1, expected_step: 2 };
        assert!(e.source().is_none());
        let e = QTapReadError::Uninitialized;
        assert!(e.source().is_none());
        let e = QTapReadError::InsufficientCapacity { capacity: 1, required: 2 };
        assert!(e.source().is_none());
    }

    // ── F16 roundtrip with smallest positive subnormal f16 ──

    #[test]
    fn f16_roundtrip_subnormal() {
        let rb = GatekeeperRingBuffer::new(1, 2);
        let sub = half::f16::from_bits(1u16); // smallest positive f16 subnormal
        let raw = sub.to_le_bytes().to_vec();
        rb.debug_write(&raw, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        assert_eq!(half::f16::from_f32(q[0]).to_bits(), sub.to_bits());
    }

    // ── InsufficientCapacity error message contains numeric ordering ──

    #[test]
    fn capacity_error_message_ordering() {
        let e = QTapReadError::InsufficientCapacity { capacity: 10, required: 20 };
        let msg = e.to_string();
        let cap_pos = msg.find("10").unwrap();
        let req_pos = msg.find("20").unwrap();
        assert!(cap_pos < req_pos, "capacity value should appear before required in message");
    }

    // ── Read after construction (step=0, no writes) is stale ──

    #[test]
    fn read_fresh_buffer_step_zero_is_ok() {
        let rb = GatekeeperRingBuffer::new(2, 4);
        // step_index starts at 0, write_cursor starts at 0.
        // read_latest(0) checks step match — step_index==0, so step matches.
        // But data is all zeros, so it reads zero-initialized f32 values.
        let q = rb.read_latest(0).unwrap();
        assert_eq!(q.len(), 2);
        assert_eq!(q[0], 0.0);
        assert_eq!(q[1], 0.0);
    }

    // ── F32 roundtrip with alternating signs ──

    #[test]
    fn f32_roundtrip_alternating_signs() {
        let rb = GatekeeperRingBuffer::new(4, 4);
        let vals = [1.0f32, -1.0, 2.5, -2.5];
        let data = f32_bytes(&vals);
        rb.debug_write(&data, 1).unwrap();
        let q = rb.read_latest(1).unwrap();
        for (i, &expected) in vals.iter().enumerate() {
            assert!((q[i] - expected).abs() < 1e-6, "mismatch at index {}", i);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests (wave-3)
    // ═══════════════════════════════════════════════════════════════

    // ── 1. q_dim accessor returns constructor value ──

    #[test]
    fn q_dim_accessor_matches_constructor() {
        // Arrange
        let expected_dim = 64;

        // Act
        let rb = GatekeeperRingBuffer::new(expected_dim, 4);

        // Assert
        assert_eq!(rb.q_dim(), expected_dim);
    }

    // ── 2. element_bytes accessor returns constructor value ──

    #[test]
    fn element_bytes_accessor_matches_constructor() {
        // Arrange
        let expected_elem_bytes = 4;

        // Act
        let rb = GatekeeperRingBuffer::new(32, expected_elem_bytes);

        // Assert
        assert_eq!(rb.element_bytes(), expected_elem_bytes);
    }

    // ── 3. slot_bytes equals q_dim times element_bytes ──

    #[test]
    fn slot_bytes_equals_dim_times_elem_bytes() {
        // Arrange
        let q_dim = 48;
        let elem_bytes = 4;
        let expected_slot = q_dim * elem_bytes;

        // Act
        let rb = GatekeeperRingBuffer::new(q_dim, elem_bytes);

        // Assert
        assert_eq!(rb.slot_bytes(), expected_slot);
    }

    // ── 4. Two writes then read returns latest step's data ──

    #[test]
    fn two_writes_read_returns_second_step_data() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(3, 4);
        let first = f32_bytes(&[10.0, 20.0, 30.0]);
        let second = f32_bytes(&[40.0, 50.0, 60.0]);

        // Act
        rb.debug_write(&first, 1).unwrap();
        rb.debug_write(&second, 2).unwrap();
        let result = rb.read_latest(2).unwrap();

        // Assert
        assert!((result[0] - 40.0).abs() < 1e-6);
        assert!((result[1] - 50.0).abs() < 1e-6);
        assert!((result[2] - 60.0).abs() < 1e-6);
    }

    // ── 5. Write with exact slot_bytes payload succeeds ──

    #[test]
    fn write_exact_slot_bytes_succeeds() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(4, 4);
        let exact_payload = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(exact_payload.len(), rb.slot_bytes());

        // Act
        let result = rb.debug_write(&exact_payload, 1);

        // Assert
        assert!(result.is_ok());
    }

    // ── 6. Write with wrong byte count fails with InsufficientCapacity ──

    #[test]
    fn write_one_extra_byte_fails_capacity() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(2, 4); // slot_bytes = 8
        let payload = vec![0u8; 9]; // one byte too many

        // Act
        let result = rb.debug_write(&payload, 1);

        // Assert
        let err = result.unwrap_err();
        assert_eq!(
            err,
            QTapReadError::InsufficientCapacity {
                capacity: 8,
                required: 9
            }
        );
    }

    // ── 7. QTapReadError Display: StaleQTap format contains step values ──

    #[test]
    fn stale_qtap_display_contains_both_steps() {
        // Arrange
        let err = QTapReadError::StaleQTap {
            buf_step: 7,
            expected_step: 13,
        };

        // Act
        let msg = err.to_string();

        // Assert
        assert!(msg.contains("7"), "display must contain buf_step");
        assert!(msg.contains("13"), "display must contain expected_step");
        assert!(msg.contains("stale"), "display must contain 'stale'");
    }

    // ── 8. QTapReadError Display: Uninitialized format ──

    #[test]
    fn uninitialized_display_matches_spec() {
        // Arrange
        let err = QTapReadError::Uninitialized;

        // Act
        let msg = err.to_string();

        // Assert
        assert!(
            msg.to_lowercase().contains("not initialized"),
            "display must contain 'not initialized', got: {}",
            msg
        );
    }

    // ── 9. QTapReadError Display: InsufficientCapacity format ──

    #[test]
    fn insufficient_capacity_display_contains_values() {
        // Arrange
        let err = QTapReadError::InsufficientCapacity {
            capacity: 16,
            required: 32,
        };

        // Act
        let msg = err.to_string();

        // Assert
        assert!(msg.contains("16"), "display must contain capacity");
        assert!(msg.contains("32"), "display must contain required");
        assert!(
            msg.contains("capacity"),
            "display must contain 'capacity'"
        );
    }

    // ── 10. QTapReadError Clone preserves all variant fields ──

    #[test]
    fn clone_preserves_insufficient_capacity_fields() {
        // Arrange
        let original = QTapReadError::InsufficientCapacity {
            capacity: 100,
            required: 200,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned, "cloned error must equal original");
        assert_eq!(
            original.to_string(),
            cloned.to_string(),
            "cloned error display must match"
        );
    }

    // ── 11. Buffer with element_bytes=2 (BF16/FP16 scenario) roundtrip ──

    #[test]
    fn element_bytes_two_full_roundtrip() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(4, 2);
        let f16_vals = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(-2.5),
            half::f16::from_f32(0.0),
            half::f16::from_f32(100.0),
        ];
        let raw: Vec<u8> = f16_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        rb.debug_write(&raw, 1).unwrap();
        let result = rb.read_latest(1).unwrap();

        // Assert
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - (-2.5)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
        assert!((result[3] - 100.0).abs() < 0.5);
    }

    // ── 12. Multiple writes cycle through both slots (5 writes) ──

    #[test]
    fn five_writes_cycle_both_slots() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);

        // Act
        for step in 1..=5u64 {
            rb.debug_write(&f32_bytes(&[step as f32]), step).unwrap();
        }
        let latest = rb.read_latest(5).unwrap();

        // Assert
        assert!((latest[0] - 5.0).abs() < 1e-6);
        // All prior steps are stale
        for old_step in 1..=4u64 {
            assert!(
                rb.read_latest(old_step).is_err(),
                "step {} should be stale after step 5",
                old_step
            );
        }
    }

    // ── 13. Read from step 0 after debug_write with step 0 ──

    #[test]
    fn read_step_zero_after_write_step_zero() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(2, 4);
        let data = f32_bytes(&[42.0, -7.5]);

        // Act
        rb.debug_write(&data, 0).unwrap();
        let result = rb.read_latest(0).unwrap();

        // Assert
        assert_eq!(result.len(), 2);
        assert!((result[0] - 42.0).abs() < 1e-6);
        assert!((result[1] - (-7.5)).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests (wave-4) — ~13 edge-case tests
    // ═══════════════════════════════════════════════════════════════

    // ── 1. Interleaved write-read-write-read data integrity ──

    #[test]
    fn interleaved_write_read_preserves_data() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(2, 4);

        // Act & Assert: first write + read
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 1).unwrap();
        let q1 = rb.read_latest(1).unwrap();
        assert!((q1[0] - 1.0).abs() < 1e-6);
        assert!((q1[1] - 2.0).abs() < 1e-6);

        // Act & Assert: second write + read
        rb.debug_write(&f32_bytes(&[3.0, 4.0]), 2).unwrap();
        let q2 = rb.read_latest(2).unwrap();
        assert!((q2[0] - 3.0).abs() < 1e-6);
        assert!((q2[1] - 4.0).abs() < 1e-6);

        // First step is now stale
        assert!(rb.read_latest(1).is_err());
    }

    // ── 2. Slot content isolation after wraparound (write over slot 0) ──

    #[test]
    fn slot_overwrite_isolation_after_wraparound() {
        // Arrange: q_dim=2 forces each slot to hold exactly 8 bytes.
        // Write A to slot 0 (cursor wraps to slot 1), then B to slot 1,
        // then C back to slot 0. Verify C replaced A entirely.
        let rb = GatekeeperRingBuffer::new(2, 4);

        // Act
        rb.debug_write(&f32_bytes(&[100.0, 200.0]), 1).unwrap(); // slot 0
        rb.debug_write(&f32_bytes(&[300.0, 400.0]), 2).unwrap(); // slot 1
        rb.debug_write(&f32_bytes(&[500.0, 600.0]), 3).unwrap(); // slot 0 again

        // Assert: latest is step 3 in slot 0
        let q = rb.read_latest(3).unwrap();
        assert!((q[0] - 500.0).abs() < 1e-6, "slot 0 should contain C, not A");
        assert!((q[1] - 600.0).abs() < 1e-6, "slot 0 should contain C, not A");
    }

    // ── 3. Write with step = u64::MAX then read ──

    #[test]
    fn write_and_read_step_u64_max() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);
        let data = f32_bytes(&[7.0]);

        // Act
        rb.debug_write(&data, u64::MAX).unwrap();
        let q = rb.read_latest(u64::MAX).unwrap();

        // Assert
        assert!((q[0] - 7.0).abs() < 1e-6);

        // step=0 is now stale
        let err = rb.read_latest(0).unwrap_err();
        assert_eq!(
            err,
            QTapReadError::StaleQTap {
                buf_step: u64::MAX,
                expected_step: 0
            }
        );
    }

    // ── 4. QTapReadError variant inequality (cross-variant) ──

    #[test]
    fn error_cross_variant_not_equal() {
        // Arrange
        let stale = QTapReadError::StaleQTap { buf_step: 0, expected_step: 0 };
        let uninit = QTapReadError::Uninitialized;
        let cap = QTapReadError::InsufficientCapacity { capacity: 0, required: 0 };

        // Assert: all three variants are distinct even with zero fields
        assert_ne!(stale, uninit);
        assert_ne!(stale, cap);
        assert_ne!(uninit, cap);
    }

    // ── 5. Rapid 100-write cycling with data integrity on final read ──

    #[test]
    fn rapid_cycling_100_writes_final_data_integrity() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);
        let final_step = 100u64;
        let final_val = 999.0f32;

        // Act: write 99 filler values, then one final value
        for step in 1..100u64 {
            let val = step as f32;
            rb.debug_write(&f32_bytes(&[val]), step).unwrap();
        }
        rb.debug_write(&f32_bytes(&[final_val]), final_step).unwrap();

        // Assert
        let q = rb.read_latest(final_step).unwrap();
        assert!((q[0] - final_val).abs() < 1e-6, "final value must be exact");
        // All prior steps stale
        assert!(rb.read_latest(99).is_err());
        assert!(rb.read_latest(1).is_err());
    }

    // ── 6. F16 oversized payload InsufficientCapacity ──

    #[test]
    fn f16_write_oversized_payload_errors() {
        // Arrange: slot_bytes = 4 (q_dim=2, element_bytes=2)
        let rb = GatekeeperRingBuffer::new(2, 2);
        let oversized: Vec<u8> = vec![0xAA; 6]; // 6 bytes, need 4

        // Act
        let err = rb.debug_write(&oversized, 1).unwrap_err();

        // Assert
        assert_eq!(
            err,
            QTapReadError::InsufficientCapacity {
                capacity: 4,
                required: 6
            }
        );
    }

    // ── 7. F16 undersized payload InsufficientCapacity ──

    #[test]
    fn f16_write_undersized_payload_errors() {
        // Arrange: slot_bytes = 8 (q_dim=4, element_bytes=2)
        let rb = GatekeeperRingBuffer::new(4, 2);
        let undersized: Vec<u8> = vec![0xBB; 4]; // 4 bytes, need 8

        // Act
        let err = rb.debug_write(&undersized, 1).unwrap_err();

        // Assert
        assert_eq!(
            err,
            QTapReadError::InsufficientCapacity {
                capacity: 8,
                required: 4
            }
        );
    }

    // ── 8. Double buffer: read between two writes shows stale for first ──

    #[test]
    fn read_between_two_writes_shows_latest_only() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);

        // Act: first write
        rb.debug_write(&f32_bytes(&[10.0]), 1).unwrap();
        // Read step 1 succeeds
        let q1 = rb.read_latest(1).unwrap();
        assert!((q1[0] - 10.0).abs() < 1e-6);

        // Second write
        rb.debug_write(&f32_bytes(&[20.0]), 2).unwrap();

        // Assert: step 1 is now stale
        assert!(rb.read_latest(1).is_err());
        let q2 = rb.read_latest(2).unwrap();
        assert!((q2[0] - 20.0).abs() < 1e-6);
    }

    // ── 9. Step gap: write step=5, then step=100, verify intermediate steps stale ──

    #[test]
    fn step_gap_intermediate_steps_stale() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);
        rb.debug_write(&f32_bytes(&[1.0]), 5).unwrap();
        rb.debug_write(&f32_bytes(&[2.0]), 100).unwrap();

        // Assert: only step=100 is valid; steps 5-99 are stale
        assert!(rb.read_latest(5).is_err());
        assert!(rb.read_latest(50).is_err());
        assert!(rb.read_latest(99).is_err());
        let q = rb.read_latest(100).unwrap();
        assert!((q[0] - 2.0).abs() < 1e-6);
    }

    // ── 10. Clone error then modify clone does not affect original ──

    #[test]
    fn cloned_error_is_independent() {
        // Arrange
        let original = QTapReadError::StaleQTap { buf_step: 42, expected_step: 99 };

        // Act
        let _cloned = original.clone();
        // (Clone produces a value type; no shared mutation possible.)

        // Assert: original still intact after clone
        assert_eq!(
            original,
            QTapReadError::StaleQTap { buf_step: 42, expected_step: 99 }
        );
    }

    // ── 11. Double buffer slot alternation verified with distinct patterns ──

    #[test]
    fn double_buffer_alternation_verified_per_slot() {
        // Arrange: use 2-element buffer so we can track which slot holds what
        let rb = GatekeeperRingBuffer::new(2, 4);

        // Act: write 4 times, alternating slots
        // Write 1 -> slot 0: [1,2], Write 2 -> slot 1: [3,4]
        // Write 3 -> slot 0: [5,6], Write 4 -> slot 1: [7,8]
        rb.debug_write(&f32_bytes(&[1.0, 2.0]), 1).unwrap();
        rb.debug_write(&f32_bytes(&[3.0, 4.0]), 2).unwrap();
        rb.debug_write(&f32_bytes(&[5.0, 6.0]), 3).unwrap();
        rb.debug_write(&f32_bytes(&[7.0, 8.0]), 4).unwrap();

        // Assert: only step 4 is readable with exact data
        let q = rb.read_latest(4).unwrap();
        assert!((q[0] - 7.0).abs() < 1e-6);
        assert!((q[1] - 8.0).abs() < 1e-6);

        // Steps 1-3 all stale
        for step in 1..=3u64 {
            assert!(rb.read_latest(step).is_err(), "step {} should be stale", step);
        }
    }

    // ── 12. decode_q_slot: F32 with MAX and MIN in same slot ──

    #[test]
    fn decode_q_slot_f32_max_and_min_combined() {
        // Arrange
        let vals = [f32::MAX, f32::MIN];
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let result = decode_q_slot(&raw, 2, 4).unwrap();

        // Assert
        assert_eq!(result[0].to_bits(), f32::MAX.to_bits());
        assert_eq!(result[1].to_bits(), f32::MIN.to_bits());
    }

    // ── 13. Step value u64::MAX - 1 boundary write and read ──

    #[test]
    fn write_read_step_near_u64_max() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(1, 4);
        let step = u64::MAX - 1;
        let data = f32_bytes(&[-13.37]);

        // Act
        rb.debug_write(&data, step).unwrap();
        let q = rb.read_latest(step).unwrap();

        // Assert
        assert!((q[0] - (-13.37)).abs() < 1e-3);
        // u64::MAX is stale
        let err = rb.read_latest(u64::MAX).unwrap_err();
        assert_eq!(
            err,
            QTapReadError::StaleQTap {
                buf_step: u64::MAX - 1,
                expected_step: u64::MAX
            }
        );
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests (wave-5) — 10 new tests
    // ═══════════════════════════════════════════════════════════════

    // ── 14. Write step 0 then step 1, verify step 0 becomes stale ──

    #[test]
    fn test_step_zero_then_step_one_stale_check() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(2, 4);
        rb.debug_write(&f32_bytes(&[11.0, 22.0]), 0).unwrap();
        assert!(rb.read_latest(0).is_ok(), "step 0 must be readable immediately after write");

        // Act: advance to step 1
        rb.debug_write(&f32_bytes(&[33.0, 44.0]), 1).unwrap();

        // Assert: step 0 is now stale
        let err = rb.read_latest(0).unwrap_err();
        assert_eq!(err, QTapReadError::StaleQTap { buf_step: 1, expected_step: 0 });
        let q = rb.read_latest(1).unwrap();
        assert!((q[0] - 33.0).abs() < 1e-6);
        assert!((q[1] - 44.0).abs() < 1e-6);
    }

    // ── 15. F16 decode_q_slot with all-zero bytes yields zero values ──

    #[test]
    fn test_decode_f16_all_zero_bytes_yields_zero() {
        // Arrange: 4 F16 elements = 8 bytes of zeros
        let raw = vec![0u8; 8];

        // Act
        let result = decode_q_slot(&raw, 4, 2).unwrap();

        // Assert
        assert_eq!(result.len(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "zero-byte F16 at index {} should decode to ~0.0, got {}",
                i,
                val
            );
        }
    }

    // ── 16. F32 decode_q_slot with all 0xFF bytes yields negative NaN ──

    #[test]
    fn test_decode_f32_all_ff_bytes_is_nan_or_special() {
        // Arrange: 2 F32 elements of all 0xFF = NaN (one quiet NaN representation)
        let raw = vec![0xFFu8; 8];

        // Act
        let result = decode_q_slot(&raw, 2, 4).unwrap();

        // Assert: all-FF f32 is a NaN
        assert_eq!(result.len(), 2);
        assert!(result[0].is_nan(), "all-FF bytes should decode to NaN");
        assert!(result[1].is_nan(), "all-FF bytes should decode to NaN");
    }

    // ── 17. Buffer total data size equals 2 * slot_bytes ──

    #[test]
    fn test_total_data_region_is_double_slot_bytes() {
        // Arrange: verify internal allocation by observing that debug_write
        // with exact slot_bytes succeeds for both alternating slots
        let q_dim = 16;
        let elem_bytes = 4;
        let rb = GatekeeperRingBuffer::new(q_dim, elem_bytes);
        let slot = rb.slot_bytes();
        assert_eq!(slot, q_dim * elem_bytes);

        // Act: write to slot 0 then slot 1 — both must succeed with slot-sized payloads
        let payload_a = vec![0xAAu8; slot];
        let payload_b = vec![0xBBu8; slot];
        let res_a = rb.debug_write(&payload_a, 1);
        let res_b = rb.debug_write(&payload_b, 2);

        // Assert: both writes succeed, confirming both slots are allocated
        assert!(res_a.is_ok(), "write to first slot must succeed");
        assert!(res_b.is_ok(), "write to second slot must succeed");
    }

    // ── 18. Odd q_dim roundtrip integrity with F32 ──

    #[test]
    fn test_odd_q_dim_f32_roundtrip() {
        // Arrange: odd dimension (7 elements) to verify no alignment assumptions
        let rb = GatekeeperRingBuffer::new(7, 4);
        let vals: Vec<f32> = (0..7).map(|i| (i as f32) * -1.5).collect();
        let data = f32_bytes(&vals);

        // Act
        rb.debug_write(&data, 1).unwrap();
        let result = rb.read_latest(1).unwrap();

        // Assert
        assert_eq!(result.len(), 7);
        for i in 0..7 {
            assert!(
                (result[i] - (i as f32) * -1.5).abs() < 1e-6,
                "mismatch at index {}: got {} expected {}",
                i,
                result[i],
                (i as f32) * -1.5
            );
        }
    }

    // ── 19. Odd q_dim roundtrip integrity with F16 ──

    #[test]
    fn test_odd_q_dim_f16_roundtrip() {
        // Arrange: odd dimension (5 elements) with F16
        let rb = GatekeeperRingBuffer::new(5, 2);
        let f16_vals: Vec<u8> = (0..5)
            .flat_map(|i| half::f16::from_f32(i as f32 * 3.0).to_le_bytes())
            .collect();

        // Act
        rb.debug_write(&f16_vals, 1).unwrap();
        let result = rb.read_latest(1).unwrap();

        // Assert
        assert_eq!(result.len(), 5);
        for i in 0..5 {
            assert!(
                (result[i] - (i as f32) * 3.0).abs() < 0.5,
                "mismatch at index {}: got {} expected {}",
                i,
                result[i],
                (i as f32) * 3.0
            );
        }
    }

    // ── 20. Seven writes fully exercise double-buffer wraparound twice ──

    #[test]
    fn test_seven_writes_double_wraparound_integrity() {
        // Arrange: 7 writes means slot 0 written 4 times, slot 1 written 3 times
        let rb = GatekeeperRingBuffer::new(1, 4);

        // Act
        for step in 1..=7u64 {
            rb.debug_write(&f32_bytes(&[step as f32 * 10.0]), step).unwrap();
        }

        // Assert: only step 7 is readable
        let q = rb.read_latest(7).unwrap();
        assert!((q[0] - 70.0).abs() < 1e-6, "final value should be 70.0");

        // All intermediate steps are stale
        for old in 1u64..=6 {
            assert!(
                rb.read_latest(old).is_err(),
                "step {} must be stale after step 7",
                old
            );
        }
    }

    // ── 21. debug_write returns unit Ok, confirming return type ──

    #[test]
    fn test_debug_write_return_type_is_unit_ok() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(2, 4);
        let data = f32_bytes(&[1.0, 2.0]);

        // Act
        let result: Result<(), QTapReadError> = rb.debug_write(&data, 1);

        // Assert
        assert_eq!(result, Ok(()));
    }

    // ── 22. F16 f16::ZERO roundtrip via decode_q_slot directly ──

    #[test]
    fn test_decode_f16_zero_value_direct() {
        // Arrange
        let raw = half::f16::ZERO.to_le_bytes().to_vec();

        // Act
        let result = decode_q_slot(&raw, 1, 2).unwrap();

        // Assert
        assert_eq!(result.len(), 1);
        assert!(
            result[0].abs() < 1e-6,
            "f16 zero must decode to approximately 0.0, got {}",
            result[0]
        );
    }

    // ── 23. Write-then-read, then re-read after no intervening writes ──

    #[test]
    fn test_repeat_read_without_write_returns_identical() {
        // Arrange
        let rb = GatekeeperRingBuffer::new(3, 4);
        let vals = [1.5f32, -2.5, 3.5];
        rb.debug_write(&f32_bytes(&vals), 10).unwrap();

        // Act: read the same step three times with no writes between
        let r1 = rb.read_latest(10).unwrap();
        let r2 = rb.read_latest(10).unwrap();
        let r3 = rb.read_latest(10).unwrap();

        // Assert: all three reads return identical data
        assert_eq!(r1, r2, "second read must match first");
        assert_eq!(r2, r3, "third read must match second");
        for (i, &expected) in vals.iter().enumerate() {
            assert!((r3[i] - expected).abs() < 1e-6, "mismatch at index {}", i);
        }
    }
}
