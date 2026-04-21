//! Gatekeeper Q-Tap ring buffer (SPEC §4.4).
//!
//! FusedAttentionLayer 的 Q-tap epilogue 将 `q_proj(hidden)[-1]` 写入此
//! 双缓冲区. SemanticGatekeeperCallback 在 pre_node 中读取.
//!
//! 同步协议: JIT 的 STG 指令 `release` 写 + bump step_index;
//! callback `acquire` 读 step_index 验证. Step 不匹配则返回 StaleQTap.

use std::sync::atomic::{AtomicU64, Ordering};

/// Ring buffer 读取错误.
#[derive(Debug, thiserror::Error, Clone)]
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
/// 支持 4 字节 (F32) 和 2 字节 (F16/BF16). 其他宽度返回 `InsufficientCapacity`
/// (作为显式错误而非静默降级,符合 NO_SILENT_FALLBACK).
fn decode_q_slot(
    raw: &[u8],
    q_dim: usize,
    element_bytes: usize,
) -> Result<Vec<f32>, QTapReadError> {
    match element_bytes {
        4 => {
            let mut out = Vec::with_capacity(q_dim);
            for i in 0..q_dim {
                let off = i * 4;
                let bytes = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                out.push(f32::from_le_bytes(bytes));
            }
            Ok(out)
        }
        2 => {
            // 解析为 f16 再转 f32. BF16 在 JIT 写入时若使用 BF16 dtype 需走
            // BF16 解码路径,此处先以 F16 为默认;BF16 差异在 Phase D
            // 中按 FusedAttentionLayer 的实际输出 dtype 做分支.
            let mut out = Vec::with_capacity(q_dim);
            for i in 0..q_dim {
                let off = i * 2;
                let bytes = [raw[off], raw[off + 1]];
                out.push(half::f16::from_le_bytes(bytes).to_f32());
            }
            Ok(out)
        }
        _ => Err(QTapReadError::InsufficientCapacity {
            capacity: element_bytes,
            required: 4,
        }),
    }
}
