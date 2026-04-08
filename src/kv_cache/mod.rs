//! KV cache tracking for executor (per SPEC 03-DATA-STRUCTURE.md, 07-OBSERVABILITY.md §7.1)

pub mod quant;
pub mod dual_track;
pub mod turboquant;

use crate::engine::executor::{KvCacheHandle, KvCacheConfig};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Error)]
pub enum KvCacheError {
    #[error("kv cache exhausted: requested {requested}, available {available}")]
    Exhausted { requested: usize, available: usize },
}

pub type KvCacheResult<T> = std::result::Result<T, KvCacheError>;

// ============================================================================
// OOM Halt Error (per SPEC 07-OBSERVABILITY.md §5.1)
// ============================================================================

/// OOM Halt 错误 (per SPEC 07-OBSERVABILITY.md §5.1, ARCH-ZERO-FALLBACK)
///
/// 当发生硬件内存越界时，系统必须当场截断 (Halt) 并抛出严重错误。
/// 禁止框架层面自行容错 — 这是架构底线。
#[derive(Debug, Clone, Error)]
#[error("OOM Halt: {message} (fatal={fatal})")]
pub struct OomHaltError {
    /// 错误消息
    pub message: String,
    /// 是否为致命错误（true = Hardware OOM, process must halt）
    pub fatal: bool,
}

impl OomHaltError {
    /// 创建致命 OOM 错误（硬件级别，必须终止进程）
    pub fn fatal_halt(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
        }
    }

    /// 创建可恢复 OOM 错误（软件级别，可重试）
    pub fn soft_halt(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
        }
    }
}

// ============================================================================
// KV Page Header (per SPEC 07-OBSERVABILITY.md §7.1)
// ============================================================================

/// KV 物理页头 (per SPEC 07-OBSERVABILITY.md §7.1)
///
/// 总计 40 Bytes，用于 Epilogue 白嫖遥测扩展。
/// 所有字段由 Mega-Kernel Epilogue 阶段直接写入，利用 STG 指令硬贴机制。
///
/// # 内存布局
/// ```text
/// Offset | Field                | Size  | Description
/// -------|---------------------|-------|-----------------------------------
/// 0x00   | page_id             | 4     | 物理页唯一标识
/// 0x04   | ref_count           | 4     | 引用计数（多请求共享）
/// 0x08   | fragmentation_metric| 4     | 页内碎片度 [0.0, 1.0]
/// 0x0C   | logits_entropy      | 4     | 输出熵值（不确定性度量）
/// 0x10   | guard_veto_flag     | 4     | 安全护栏拦截标志 (0=pass, 1=halt)
/// 0x14   | softmax_max         | 4     | Attention Sink 信号
/// 0x18   | softmax_sharpness   | 4     | max/sum 比值（尖锐度）
/// 0x1C   | residual_delta_rho  | 4     | 跨层残差能量差
/// 0x20   | dead_neuron_ratio   | 4     | 死神经元占比 [0.0, 1.0]
/// 0x24   | per_channel_scale   | 4     | per-channel scale (KIVI 路径)
/// -------|---------------------|-------|-----------------------------------
/// Total  |                     | 40    |
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvPageHeader {
    // 基础管理 (8 Bytes)
    /// 物理页唯一标识
    pub page_id: u32,
    /// 引用计数（多请求共享时 > 1）
    pub ref_count: u32,

    // Phase 1 — Epilogue 自动写入 (12 Bytes)
    /// 页内碎片度量 [0.0, 1.0]，越高表示页内有效 token 越少
    pub fragmentation_metric: f32,
    /// 输出分布熵值，用于检测生成质量下降
    pub logits_entropy: f32,
    /// 安全护栏拦截标志（0=pass, 1=halt）
    pub guard_veto_flag: u32,

    // Phase 2 — 全链路白嫖扩展 (20 Bytes)
    /// Attention Sink 信号（最大注意力权重）
    pub softmax_max: f32,
    /// max/sum 比值，衡量注意力分布尖锐度
    pub softmax_sharpness: f32,
    /// 跨层残差能量差（用于检测激活异常）
    pub residual_delta_rho: f32,
    /// 死神经元占比 [0.0, 1.0]
    pub dead_neuron_ratio: f32,
    /// Per-channel scale（仅 KIVI 量化路径写入）
    pub per_channel_scale: f32,
}

// 静态断言：确保结构体大小为 40 字节
const _: [(); 40] = [(); std::mem::size_of::<KvPageHeader>()];

impl Default for KvPageHeader {
    fn default() -> Self {
        Self {
            page_id: 0,
            ref_count: 0,
            fragmentation_metric: 0.0,
            logits_entropy: 0.0,
            guard_veto_flag: 0,
            softmax_max: 0.0,
            softmax_sharpness: 0.0,
            residual_delta_rho: 0.0,
            dead_neuron_ratio: 0.0,
            per_channel_scale: 0.0,
        }
    }
}

impl KvPageHeader {
    /// 创建新的页头
    pub fn new(page_id: u32) -> Self {
        Self {
            page_id,
            ..Default::default()
        }
    }

    /// 检查是否被护栏拦截
    #[inline]
    pub fn is_guard_vetoed(&self) -> bool {
        self.guard_veto_flag != 0
    }

    /// 检查页是否被引用
    #[inline]
    pub fn is_active(&self) -> bool {
        self.ref_count > 0
    }

    /// 检查页是否高度碎片化（> 50%）
    #[inline]
    pub fn is_fragmented(&self) -> bool {
        self.fragmentation_metric > 0.5
    }

    /// 检查输出熵是否异常低（< 0.1 bits，可能退化）
    #[inline]
    pub fn is_low_entropy(&self) -> bool {
        self.logits_entropy < 0.1
    }
}

// ============================================================================
// KV Cache Slot
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheSlot {
    Front,
    Back,
}

impl KvCacheSlot {
    pub fn flip(self) -> Self {
        match self {
            KvCacheSlot::Front => KvCacheSlot::Back,
            KvCacheSlot::Back => KvCacheSlot::Front,
        }
    }
}

// ============================================================================
// KV Cache State
// ============================================================================

#[derive(Debug, Clone)]
pub struct KvCacheState {
    handle: KvCacheHandle,
    config: KvCacheConfig,
    used: usize,
}

impl KvCacheState {
    pub fn new(handle: KvCacheHandle, config: KvCacheConfig) -> Self {
        Self {
            handle,
            config,
            used: 0,
        }
    }

    pub fn handle(&self) -> KvCacheHandle {
        self.handle
    }

    pub fn handle_mut(&mut self) -> &mut KvCacheHandle {
        &mut self.handle
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config.clone()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.config.max_seq_len.saturating_sub(self.used)
    }

    /// For LMCache reuse we sometimes need to restore the consumed length to
    /// a previously snapshotted value without touching the underlying GPU
    /// storage. This keeps zero-copy semantics while making the logical
    /// cursor reusable.
    pub fn set_used(&mut self, used: usize) -> KvCacheResult<()> {
        if used > self.config.max_seq_len {
            return Err(KvCacheError::Exhausted {
                requested: used,
                available: self.config.max_seq_len,
            });
        }
        self.used = used;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn advance(&mut self, tokens: usize) -> KvCacheResult<()> {
        let remaining = self.remaining();
        if tokens > remaining {
            return Err(KvCacheError::Exhausted {
                requested: tokens,
                available: remaining,
            });
        }
        self.used = self.used.saturating_add(tokens);
        Ok(())
    }
}

// ============================================================================
// KV Cache Double Buffer
// ============================================================================

#[derive(Debug, Clone)]
pub struct KvCacheDoubleBuffer {
    front: KvCacheState,
    back: KvCacheState,
}

impl KvCacheDoubleBuffer {
    pub fn new(front: KvCacheState, back: KvCacheState) -> Self {
        Self { front, back }
    }

    pub fn front(&self) -> &KvCacheState {
        &self.front
    }

    pub fn back(&self) -> &KvCacheState {
        &self.back
    }

    pub fn front_mut(&mut self) -> &mut KvCacheState {
        &mut self.front
    }

    pub fn back_mut(&mut self) -> &mut KvCacheState {
        &mut self.back
    }

    pub fn slot(&self, slot: KvCacheSlot) -> &KvCacheState {
        match slot {
            KvCacheSlot::Front => &self.front,
            KvCacheSlot::Back => &self.back,
        }
    }

    pub fn slot_mut(&mut self, slot: KvCacheSlot) -> &mut KvCacheState {
        match slot {
            KvCacheSlot::Front => &mut self.front,
            KvCacheSlot::Back => &mut self.back,
        }
    }

    pub fn overwrite_slot(&mut self, slot: KvCacheSlot, state: KvCacheState) {
        match slot {
            KvCacheSlot::Front => self.front = state,
            KvCacheSlot::Back => self.back = state,
        }
    }

    pub fn reset_all(&mut self) {
        self.front.reset();
        self.back.reset();
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.front, &mut self.back);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::types::DType;

    #[test]
    fn test_kv_page_header_size() {
        assert_eq!(std::mem::size_of::<KvPageHeader>(), 40);
    }

    #[test]
    fn test_kv_page_header_default() {
        let header = KvPageHeader::default();
        assert_eq!(header.page_id, 0);
        assert_eq!(header.ref_count, 0);
        assert_eq!(header.fragmentation_metric, 0.0);
        assert_eq!(header.logits_entropy, 0.0);
        assert_eq!(header.guard_veto_flag, 0);
    }

    #[test]
    fn test_kv_page_header_new() {
        let header = KvPageHeader::new(42);
        assert_eq!(header.page_id, 42);
        assert_eq!(header.ref_count, 0);
        assert!(!header.is_active());
        assert!(!header.is_guard_vetoed());
        assert!(!header.is_fragmented());
        assert!(header.is_low_entropy()); // entropy = 0.0 < 0.1
    }

    #[test]
    fn test_kv_page_header_guard_veto() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.is_guard_vetoed());
        header.guard_veto_flag = 1;
        assert!(header.is_guard_vetoed());
    }

    #[test]
    fn test_kv_page_header_fragmented() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.is_fragmented());
        header.fragmentation_metric = 0.6;
        assert!(header.is_fragmented());
    }

    #[test]
    fn test_kv_page_header_low_entropy() {
        let mut header = KvPageHeader::new(1);
        assert!(header.is_low_entropy());
        header.logits_entropy = 0.5;
        assert!(!header.is_low_entropy());
    }

    #[test]
    fn test_oom_halt_error() {
        let fatal = OomHaltError::fatal_halt("GPU memory exhausted");
        assert!(fatal.fatal);
        assert_eq!(fatal.message, "GPU memory exhausted");
        assert_eq!(fatal.to_string(), "OOM Halt: GPU memory exhausted (fatal=true)");

        let soft = OomHaltError::soft_halt("Temporarily out of memory");
        assert!(!soft.fatal);
        assert_eq!(soft.message, "Temporarily out of memory");
        assert_eq!(soft.to_string(), "OOM Halt: Temporarily out of memory (fatal=false)");
    }

    #[test]
    fn test_kv_cache_slot_flip() {
        assert_eq!(KvCacheSlot::Front.flip(), KvCacheSlot::Back);
        assert_eq!(KvCacheSlot::Back.flip(), KvCacheSlot::Front);
    }

    #[test]
    fn test_kv_cache_state_advance() {
        let handle = KvCacheHandle(1);
        let config = KvCacheConfig {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            max_seq_len: 100,
            kv_dtype: DType::F16,
            page_size: 16,
            swap_config: None,
        };
        let mut state = KvCacheState::new(handle, config);
        assert_eq!(state.used(), 0);
        assert_eq!(state.remaining(), 100);

        state.advance(10).unwrap();
        assert_eq!(state.used(), 10);
        assert_eq!(state.remaining(), 90);

        assert!(state.advance(200).is_err());
    }

    #[test]
    fn test_kv_cache_state_reset() {
        let handle = KvCacheHandle(1);
        let config = KvCacheConfig {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            max_seq_len: 100,
            kv_dtype: DType::F16,
            page_size: 16,
            swap_config: None,
        };
        let mut state = KvCacheState::new(handle, config);
        state.advance(50).unwrap();
        assert_eq!(state.used(), 50);

        state.reset();
        assert_eq!(state.used(), 0);
    }

    #[test]
    fn test_kv_cache_double_buffer_swap() {
        let handle1 = KvCacheHandle(1);
        let handle2 = KvCacheHandle(2);
        let config = KvCacheConfig {
            num_layers: 32,
            num_heads: 32,
            head_dim: 128,
            max_seq_len: 100,
            kv_dtype: DType::F16,
            page_size: 16,
            swap_config: None,
        };
        let front = KvCacheState::new(handle1, config.clone());
        let back = KvCacheState::new(handle2, config);

        let mut buffer = KvCacheDoubleBuffer::new(front, back);
        let front_id = buffer.front().handle();
        let back_id = buffer.back().handle();

        buffer.swap();
        assert_eq!(buffer.front().handle(), back_id);
        assert_eq!(buffer.back().handle(), front_id);
    }
}
