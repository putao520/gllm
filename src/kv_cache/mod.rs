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
// KV Page Header (per SPEC 07-OBSERVABILITY.md §7.1, 19-KV-CACHE-OPTIMIZATION.md §2.1)
// ============================================================================

/// KV 物理页头 (48B, per SPEC 19-KV-CACHE-OPTIMIZATION.md §2.1)
///
/// 四区域布局：基础管理 + Epilogue 遥测 + 量化元数据 + 调度元数据。
/// Epilogue 遥测由 Mega-Kernel Epilogue 阶段直接写入。
/// 量化/调度元数据由 Rust 调度器 (kv_optimizer) 写入。
///
/// # 内存布局 (repr(C), 自然对齐)
/// ```text
/// Offset | Field                | Size  | Region
/// -------|---------------------|-------|-------------------------------
/// 0x00   | page_id             | 4     | 基础管理
/// 0x04   | ref_count           | 4     | 基础管理
/// 0x08   | entropy_avg         | 2     | Epilogue 遥测 (f16 bits as u16)
/// 0x0A   | centroid_pos        | 2     | Epilogue 遥测
/// 0x0C   | softmax_max_avg     | 2     | Epilogue 遥测
/// 0x0E   | delta_rho_avg       | 2     | Epilogue 遥测
/// 0x10   | dead_ratio          | 1     | Epilogue 遥测
/// 0x11   | importance_score    | 1     | Epilogue 遥测 (0-255)
/// 0x12   | head_entropy_max    | 1     | Epilogue 遥测
/// 0x13   | head_entropy_min    | 1     | Epilogue 遥测
/// 0x14   | sink_mask           | 4     | 量化元数据 (sink token 位掩码)
/// 0x18   | channel_bitmap_lo   | 4     | 量化元数据 (MUSTAFAR 稀疏掩码)
/// 0x1C   | k_scale_offset      | 2     | 量化元数据 (per-channel K scale 偏移)
/// 0x1E   | precision_tier      | 1     | 量化元数据 (见 PrecisionTier 枚举)
/// 0x1F   | v_scale_factor      | 1     | 量化元数据 (per-token V scale 指数)
/// 0x20   | layer_mask          | 4     | 调度元数据 (有效层位掩码)
/// 0x24   | tier_age            | 2     | 调度元数据 (精度等级 tick 计数)
/// 0x26   | pipeline_id         | 1     | 调度元数据 (0=Conversation, 1=Working)
/// 0x27   | deopt_flags         | 1     | 调度元数据 (Deopt 标志)
/// 0x28   | _reserved           | 8     | 对齐填充 + 未来扩展
/// -------|---------------------|-------|-------------------------------
/// Total  |                     | 48    |
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvPageHeader {
    // ── 基础管理 (8B) ──
    /// 物理页唯一标识
    pub page_id: u32,
    /// 引用计数（多请求共享时 > 1）
    pub ref_count: u32,

    // ── Epilogue 遥测区域 (16B) — Mega-Kernel Epilogue 自动写入 ──
    /// Softmax Epilogue: 注意力分散度 (f16 bits, 用 f16_to_f32 解码)
    pub entropy_avg: u16,
    /// Softmax Epilogue: 注意力重心位置 (f16 bits)
    pub centroid_pos: u16,
    /// Softmax Epilogue: 注意力峰值 (f16 bits)
    pub softmax_max_avg: u16,
    /// Residual Epilogue: 跨层能量差 (f16 bits)
    pub delta_rho_avg: u16,
    /// FFN Epilogue: 死神经元比例 [0, 255] → 映射到 [0.0, 1.0]
    pub dead_ratio: u8,
    /// 综合重要性评分 (0-255, 由 kv_optimizer 计算)
    pub importance_score: u8,
    /// per-head 最大 entropy (Epilogue 写入)
    pub head_entropy_max: u8,
    /// per-head 最小 entropy (Epilogue 写入)
    pub head_entropy_min: u8,

    // ── 量化元数据区域 (12B) — Rust 调度器写入 ──
    /// page 内 sink token 位掩码 (bit i = 1 → token i 是 sink)
    pub sink_mask: u32,
    /// 通道稀疏掩码低 32 位 (MUSTAFAR, bit = 1 → 通道活跃)
    pub channel_bitmap_lo: u32,
    /// per-channel K scale 在页内偏移 (字节)
    pub k_scale_offset: u16,
    /// 精度等级 (见 PrecisionTier 枚举值)
    pub precision_tier: u8,
    /// per-token V scale 指数
    pub v_scale_factor: u8,

    // ── 调度元数据区域 (8B) — Rust 调度器写入 ──
    /// 有效层位掩码 (跨层共享页时标识哪些层有效)
    pub layer_mask: u32,
    /// 精度等级赋值后的 tick 计数 (用于老化决策)
    pub tier_age: u16,
    /// 所属管线 (0=Conversation, 1=Working)
    pub pipeline_id: u8,
    /// Deopt 标志 (bit 0: 需 requantize, bit 1: 数据不一致)
    pub deopt_flags: u8,

    // ── 对齐填充 + 未来扩展 (8B) ──
    _reserved: [u8; 8],
}

const _: [(); 48] = [(); std::mem::size_of::<KvPageHeader>()];

/// 页级精度等级 (per SPEC 19-KV-CACHE-OPTIMIZATION.md §2.2)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionTier {
    /// 全精度 — sink token / 刚 prefill 的页
    FP16 = 0,
    /// 8-bit 量化 — 高重要性正常 token
    FP8 = 1,
    /// 4-bit KIVI 量化 — 正常 token (默认)
    KIVI4 = 2,
    /// 2-bit 激进量化 — 低重要性 / Working 管线
    KIVI2 = 3,
    /// 非结构化稀疏 (MUSTAFAR) — channel_bitmap 标记零通道
    Sparse = 4,
    /// 字典稀疏编码 (Lexico)
    Dictionary = 5,
    /// 已 eviction — 页数据无效，可被回收
    Evicted = 6,
}

impl Default for KvPageHeader {
    fn default() -> Self {
        Self {
            page_id: 0,
            ref_count: 0,
            entropy_avg: 0,
            centroid_pos: 0,
            softmax_max_avg: 0,
            delta_rho_avg: 0,
            dead_ratio: 0,
            importance_score: 0,
            head_entropy_max: 0,
            head_entropy_min: 0,
            sink_mask: 0,
            channel_bitmap_lo: 0,
            k_scale_offset: 0,
            precision_tier: PrecisionTier::FP16 as u8,
            v_scale_factor: 0,
            layer_mask: 0,
            tier_age: 0,
            pipeline_id: 0,
            deopt_flags: 0,
            _reserved: [0; 8],
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

    /// 检查页是否被引用
    #[inline]
    pub fn is_active(&self) -> bool {
        self.ref_count > 0
    }

    /// 获取精度等级
    #[inline]
    pub fn precision_tier(&self) -> PrecisionTier {
        match self.precision_tier {
            0 => PrecisionTier::FP16,
            1 => PrecisionTier::FP8,
            2 => PrecisionTier::KIVI4,
            3 => PrecisionTier::KIVI2,
            4 => PrecisionTier::Sparse,
            5 => PrecisionTier::Dictionary,
            6 => PrecisionTier::Evicted,
            _ => PrecisionTier::FP16,
        }
    }

    /// 设置精度等级
    #[inline]
    pub fn set_precision_tier(&mut self, tier: PrecisionTier) {
        self.precision_tier = tier as u8;
    }

    /// 检查是否包含 sink token (sink_mask 非零)
    #[inline]
    pub fn has_sink_token(&self) -> bool {
        self.sink_mask != 0
    }

    /// 检查是否需要 requantize (deopt bit 0)
    #[inline]
    pub fn needs_requantize(&self) -> bool {
        self.deopt_flags & 0x01 != 0
    }

    /// 获取 entropy 差值 (head_entropy_max - head_entropy_min)
    /// 差值大说明 head 间注意力模式差异大，是 MUSTAFAR 稀疏化的候选
    #[inline]
    pub fn head_entropy_spread(&self) -> u8 {
        self.head_entropy_max.saturating_sub(self.head_entropy_min)
    }

    /// 检查输出熵是否异常低 (entropy_avg 为 0 或极低值)
    #[inline]
    pub fn is_low_entropy(&self) -> bool {
        self.entropy_avg == 0
    }

    /// 检查死神经元占比是否高 (> 50%, dead_ratio 映射到 [0.0, 1.0])
    #[inline]
    pub fn is_high_dead_ratio(&self) -> bool {
        self.dead_ratio > 127
    }

    /// REQ-KV-OPT-010: 标记为 position-agnostic (CacheSlide)
    /// System prompt 页跳过 RoPE 注入，实现跨请求 KV 复用。
    /// 使用 _reserved[0] bit 0 作为标志。
    #[inline]
    pub fn set_position_agnostic(&mut self, value: bool) {
        if value {
            self._reserved[0] |= 0x01;
        } else {
            self._reserved[0] &= !0x01;
        }
    }

    /// 检查是否为 position-agnostic 页 (CacheSlide)
    #[inline]
    pub fn is_position_agnostic(&self) -> bool {
        self._reserved[0] & 0x01 != 0
    }
}

// ============================================================================
// f16 conversion helpers (IEEE 754 half-precision, no external dependency)
// ============================================================================

/// Convert f32 → f16 bits (IEEE 754, round-to-nearest-even).
/// Used for storing telemetry values in KvPageHeader u16 fields.
#[inline]
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = (bits >> 23) & 0xFF;
    let mant = bits & 0x7FFFFF;

    let (sign_out, exp_out, mant_out) = if exp == 0 {
        (sign, 0u16, 0u16)
    } else if exp == 255 {
        (sign, 0x1F, if mant != 0 { 0x200 } else { 0 })
    } else {
        let new_exp = exp as i32 - 127 + 15;
        if new_exp <= 0 {
            let shift = 1 - new_exp;
            if shift > 24 {
                (sign, 0, 0)
            } else {
                let rounded = (mant | 0x800000) >> shift;
                (sign, 0, (rounded >> 13) as u16)
            }
        } else if new_exp >= 0x1F {
            (sign, 0x1F, 0)
        } else {
            let m = (mant >> 13) as u16;
            (sign, new_exp as u16, m)
        }
    };

    (sign_out as u16) << 15 | (exp_out << 10) | (mant_out & 0x3FF)
}

/// Convert f16 bits → f32 (IEEE 754).
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;

    let f = if exp == 0 {
        if mant == 0 {
            sign as u32
        } else {
            let mut m = mant as u32;
            let mut e: u32 = 0;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3FF;
            let normalized_exp = 127 - 15 - e + 1;
            (sign as u32) << 31 | normalized_exp << 23 | (m << 13)
        }
    } else if exp == 0x1F {
        (sign as u32) << 31 | 0x7F800000 | ((mant as u32) << 13)
    } else {
        let biased = (exp as u32 + 127 - 15) << 23;
        (sign as u32) << 31 | biased | ((mant as u32) << 13)
    };

    f32::from_bits(f)
}

/// Convert f32 to dead_ratio u8 (0.0 → 0, 1.0 → 255).
#[inline]
pub fn f32_to_dead_ratio(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0) as u8
}

/// Convert dead_ratio u8 to f32 (0 → 0.0, 255 → 1.0).
#[inline]
pub fn dead_ratio_to_f32(x: u8) -> f32 {
    x as f32 / 255.0
}

// ============================================================================
// Layer Donor Info (SharedKvRef §P1.1)
// ============================================================================

/// Per-page donor/refcount metadata for **layer-level** KV sharing
/// (Gemma 4 E2B/E4B). Kept outside of `KvPageHeader` so the 48-byte hardware
/// page-header contract stays intact (see REQ-KV-OPT-001).
///
/// - `donor_layer = Some(n)` → this page is a *reference* pointing at the
///   physical storage of layer `n`. Writes are forbidden; reads follow the
///   donor.
/// - `donor_layer = None` → this page owns its physical storage. The
///   `borrower_refcount` tracks how many consumer layers hold a reference
///   against it; the owner's physical block MUST NOT be reclaimed while
///   `borrower_refcount > 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerDonorInfo {
    /// Layer this entry represents.
    pub layer: u16,
    /// Attention-pattern bucket (0 = sliding, 1 = global).
    pub attn_bucket: u8,
    /// When `Some(n)`, this entry references layer `n`'s page; do not write.
    pub donor_layer: Option<u16>,
    /// How many *consumer* layers currently reference this owned page.
    /// Only meaningful when `donor_layer.is_none()`.
    pub borrower_refcount: u32,
}

impl LayerDonorInfo {
    /// Construct an owned (donor-capable) entry with zero borrowers.
    pub fn owned(layer: u16, attn_bucket: u8) -> Self {
        Self {
            layer,
            attn_bucket,
            donor_layer: None,
            borrower_refcount: 0,
        }
    }

    /// Construct a reference entry pointing at `donor_layer`.
    pub fn reference(layer: u16, attn_bucket: u8, donor_layer: u16) -> Self {
        Self {
            layer,
            attn_bucket,
            donor_layer: Some(donor_layer),
            borrower_refcount: 0,
        }
    }

    /// True when this entry references another layer's storage.
    #[inline]
    pub fn is_shared(&self) -> bool {
        self.donor_layer.is_some()
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
        self.config.max_seq_len().saturating_sub(self.used)
    }

    /// For LMCache reuse we sometimes need to restore the consumed length to
    /// a previously snapshotted value without touching the underlying GPU
    /// storage. This keeps zero-copy semantics while making the logical
    /// cursor reusable.
    pub fn set_used(&mut self, used: usize) -> KvCacheResult<()> {
        if used > self.config.max_seq_len() {
            return Err(KvCacheError::Exhausted {
                requested: used,
                available: self.config.max_seq_len(),
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
    use std::sync::Arc;

    /// Create a test KvCacheConfig with geometry for KV cache tests.
    fn test_kv_config(max_seq_len: usize) -> KvCacheConfig {
        let geometry = Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            max_seq_len,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: DType::F16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
        });
        KvCacheConfig {
            geometry,
            kv_dtype: DType::F16,
            page_size: 16,
            swap_config: None,
        }
    }

    #[test]
    fn test_kv_page_header_size() {
        assert_eq!(std::mem::size_of::<KvPageHeader>(), 48);
    }

    #[test]
    fn test_kv_page_header_default() {
        let header = KvPageHeader::default();
        assert_eq!(header.page_id, 0);
        assert_eq!(header.ref_count, 0);
        assert_eq!(header.entropy_avg, 0);
        assert_eq!(header.importance_score, 0);
        assert_eq!(header.precision_tier, PrecisionTier::FP16 as u8);
        assert_eq!(header.sink_mask, 0);
        assert_eq!(header.channel_bitmap_lo, 0);
        assert_eq!(header.pipeline_id, 0);
        assert_eq!(header.deopt_flags, 0);
    }

    #[test]
    fn test_kv_page_header_new() {
        let header = KvPageHeader::new(42);
        assert_eq!(header.page_id, 42);
        assert_eq!(header.ref_count, 0);
        assert!(!header.is_active());
        assert!(header.is_low_entropy());
        assert!(!header.has_sink_token());
        assert!(!header.needs_requantize());
    }

    #[test]
    fn test_kv_page_header_precision_tier() {
        let mut header = KvPageHeader::new(1);
        assert_eq!(header.precision_tier(), PrecisionTier::FP16);
        header.set_precision_tier(PrecisionTier::KIVI4);
        assert_eq!(header.precision_tier(), PrecisionTier::KIVI4);
    }

    #[test]
    fn test_kv_page_header_sink_mask() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.has_sink_token());
        header.sink_mask = 0b101; // token 0 and 2 are sinks
        assert!(header.has_sink_token());
    }

    #[test]
    fn test_kv_page_header_entropy_spread() {
        let mut header = KvPageHeader::new(1);
        assert_eq!(header.head_entropy_spread(), 0);
        header.head_entropy_max = 200;
        header.head_entropy_min = 50;
        assert_eq!(header.head_entropy_spread(), 150);
    }

    #[test]
    fn test_kv_page_header_deopt() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.needs_requantize());
        header.deopt_flags = 0x01;
        assert!(header.needs_requantize());
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
        let config = test_kv_config(100);
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
        let config = test_kv_config(100);
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
        let config1 = test_kv_config(100);
        let config2 = test_kv_config(100);
        let front = KvCacheState::new(handle1, config1);
        let back = KvCacheState::new(handle2, config2);

        let mut buffer = KvCacheDoubleBuffer::new(front, back);
        let front_id = buffer.front().handle();
        let back_id = buffer.back().handle();

        buffer.swap();
        assert_eq!(buffer.front().handle(), back_id);
        assert_eq!(buffer.back().handle(), front_id);
    }

    // ------------------------------------------------------------------
    // LayerDonorInfo (SharedKvRef §P1.1)
    // ------------------------------------------------------------------

    #[test]
    fn layer_donor_info_owned_has_no_donor() {
        let entry = LayerDonorInfo::owned(5, 0);
        assert_eq!(entry.layer, 5);
        assert_eq!(entry.attn_bucket, 0);
        assert!(entry.donor_layer.is_none());
        assert!(!entry.is_shared());
        assert_eq!(entry.borrower_refcount, 0);
    }

    #[test]
    fn layer_donor_info_reference_points_at_donor() {
        let entry = LayerDonorInfo::reference(25, 1, 11);
        assert_eq!(entry.layer, 25);
        assert_eq!(entry.attn_bucket, 1);
        assert_eq!(entry.donor_layer, Some(11));
        assert!(entry.is_shared());
    }
}
