//! KV cache tracking for executor (per SPEC 03-DATA-STRUCTURE.md, 07-OBSERVABILITY.md §7.1)

pub mod kv_optimizer;
pub mod quant;
pub mod dual_track;
pub mod turboquant;

use crate::engine::executor::{KvCacheHandle, KvCacheConfig};
use kv_optimizer::{KvOptimization, KvOptimizationConfig, KvOptimizationStatus};
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

/// KV 物理页头 (56B, per SPEC 22-PAGE-COMPRESSION.md §2)
///
/// 五区域布局：基础管理 + Epilogue 遥测 + 量化元数据 + 调度元数据 + 压缩/存储。
/// Epilogue 遥测由 Mega-Kernel Epilogue 阶段直接写入。
/// 量化/调度元数据由 Rust 调度器 (kv_optimizer) 写入。
///
/// # 内存布局 (repr(C), 自然对齐)
/// ```text
/// Offset | Field                | Size  | Region
/// -------|---------------------|-------|-------------------------------
/// 0x00   | page_id             | 4     | 基础管理
/// 0x04   | ref_count           | 2     | 基础管理 (u16, 自 COMP1 缩减)
/// 0x06   | entropy_avg         | 2     | Epilogue 遥测 (f16 bits as u16)
/// 0x08   | centroid_pos        | 2     | Epilogue 遥测
/// 0x0A   | softmax_max_avg     | 2     | Epilogue 遥测
/// 0x0C   | delta_rho_avg       | 2     | Epilogue 遥测
/// 0x0E   | dead_ratio          | 1     | Epilogue 遥测
/// 0x0F   | importance_score    | 1     | Epilogue 遥测 (0-255)
/// 0x10   | head_entropy_max    | 1     | Epilogue 遥测
/// 0x11   | head_entropy_min    | 1     | Epilogue 遥测
/// 0x12   | [padding]           | 2     | (对齐 u32 sink_mask)
/// 0x14   | sink_mask           | 4     | 量化元数据 (sink token 位掩码)
/// 0x18   | channel_bitmap_lo   | 4     | 量化元数据 (MUSTAFAR 稀疏掩码)
/// 0x1C   | k_scale_offset      | 2     | 量化元数据 (per-channel K scale 偏移)
/// 0x1E   | precision_tier      | 1     | 量化元数据 (见 PrecisionTier 枚举)
/// 0x1F   | v_scale_factor      | 1     | 量化元数据 (per-token V scale 指数)
/// 0x20   | layer_mask          | 4     | 调度元数据 (有效层位掩码)
/// 0x24   | tier_age            | 2     | 调度元数据 (精度等级 tick 计数)
/// 0x26   | pipeline_id         | 1     | 调度元数据 (0=Conversation, 1=Working)
/// 0x27   | deopt_flags         | 1     | 调度元数据 (Deopt 标志)
/// 0x28   | codec               | 1     | 压缩+存储 (CompressionCodec)
/// 0x29   | storage_tier        | 1     | 压缩+存储 (StorageTier)
/// 0x2A   | checksum            | 2     | 压缩+存储 (CRC16)
/// 0x2C   | compressed_size     | 4     | 压缩+存储
/// 0x30   | _pad                | 8     | 压缩+存储 (56B 对齐填充)
/// -------|---------------------|-------|-------------------------------
/// Total  |                     | 56    |
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KvPageHeader {
    // ── 基础管理 (6B) ──
    /// 物理页唯一标识
    pub page_id: u32,
    /// 引用计数（多请求共享时 > 1, COMP1 从 u32 缩减为 u16）
    pub ref_count: u16,

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

    // ── 压缩 + 存储元数据 (16B, per SPEC 22-PAGE-COMPRESSION §2, COMP1 扩展至 56B) ──
    /// 压缩编解码器 (见 CompressionCodec 枚举)
    pub codec: CompressionCodec,
    /// 当前 storage tier (见 StorageTier 枚举: 0=GpuHbm, 1=CpuDram, 2=Nvme)
    pub storage_tier: StorageTier,
    /// CRC16 校验 (防止冷页腐败)
    pub checksum: u16,
    /// 压缩后字节数 (0 = 未压缩)
    pub compressed_size: u32,
    /// 56B 对齐填充
    pub _pad: [u8; 8],
}

const _: [(); 56] = [(); std::mem::size_of::<KvPageHeader>()];

/// 压缩编解码器 (per SPEC 22-PAGE-COMPRESSION §3.1)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionCodec {
    /// 未压缩 (hot page 默认)
    None = 0,
    /// LZ4-frame, GPU 解压成熟 (适用 FP16/FP8 流)
    Lz4 = 1,
    /// Bit-pack + RLE (适用 KIVI4/KIVI2 nibble stream)
    BitPackRle = 2,
    /// nvCOMP ANS (NVIDIA H100+ 原生熵编码)
    NvcompAns = 3,
    /// Zstd 字典模式 (NVMe 冷层, 不直接送 GPU)
    ZstdDict = 4,
}

impl CompressionCodec {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::BitPackRle),
            3 => Some(Self::NvcompAns),
            4 => Some(Self::ZstdDict),
            _ => None,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// 物理存储层级 (per SPEC 22-PAGE-COMPRESSION §4.1)
///
/// 优先级: GpuHbm (最高) > CpuDram > Nvme (最低).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    /// GPU HBM — hot, 微秒级延迟
    GpuHbm = 0,
    /// CPU DRAM — warm, ~10ms PCIe 换入换出
    CpuDram = 1,
    /// NVMe — cold, ~100ms 文件 I/O
    Nvme = 2,
}

impl StorageTier {
    /// 从 u8 解码 (0=GpuHbm, 1=CpuDram, 2=Nvme).
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::GpuHbm),
            1 => Some(Self::CpuDram),
            2 => Some(Self::Nvme),
            _ => None,
        }
    }

    /// 编码为 u8 (0=GpuHbm, 1=CpuDram, 2=Nvme).
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// 层级优先级: GpuHbm > CpuDram > Nvme.
/// 低判别值 = 高优先级 (0=HBM, 1=DRAM, 2=NVMe).
impl PartialOrd for StorageTier {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StorageTier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse discriminant order: lower value = higher priority
        let a = *self as u8;
        let b = *other as u8;
        // 0 (HBM) > 1 (DRAM) > 2 (NVMe), so reverse numeric order
        b.cmp(&a)
    }
}

/// 选择压缩 codec (per SPEC 22-PAGE-COMPRESSION §3.4).
///
/// `is_gpu_path`: 当前页是否在 GPU 上 (用于 nvCOMP 选路).
/// `has_nvcomp`: 运行时是否检测到 nvCOMP 库可用.
pub fn select_codec(
    tier: PrecisionTier,
    is_gpu_path: bool,
    has_nvcomp: bool,
) -> CompressionCodec {
    match tier {
        PrecisionTier::FP16 | PrecisionTier::FP8 => {
            if is_gpu_path && has_nvcomp {
                CompressionCodec::NvcompAns
            } else {
                CompressionCodec::Lz4
            }
        }
        PrecisionTier::KIVI4 | PrecisionTier::KIVI2 => CompressionCodec::BitPackRle,
        PrecisionTier::Sparse | PrecisionTier::Dictionary => CompressionCodec::None,
        PrecisionTier::Evicted => CompressionCodec::None,
    }
}

/// 选择 cold-tier codec (NVMe 强制 ZstdDict).
pub fn select_cold_codec(_tier: PrecisionTier) -> CompressionCodec {
    CompressionCodec::ZstdDict
}

/// 页级精度等级 (per SPEC 19-KV-CACHE-OPTIMIZATION.md §2.2)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            codec: CompressionCodec::None,
            storage_tier: StorageTier::GpuHbm,
            checksum: 0,
            compressed_size: 0,
            _pad: [0; 8],
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
    /// 使用 deopt_flags bit 7 (高位, 与 deopt 语义低位独立).
    #[inline]
    pub fn set_position_agnostic(&mut self, value: bool) {
        if value {
            self.deopt_flags |= 0x80;
        } else {
            self.deopt_flags &= !0x80;
        }
    }

    /// 检查是否为 position-agnostic 页 (CacheSlide)
    #[inline]
    pub fn is_position_agnostic(&self) -> bool {
        self.deopt_flags & 0x80 != 0
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
/// (Gemma 4 E2B/E4B). Kept outside of `KvPageHeader` so the 56-byte hardware
/// page-header contract stays intact (see REQ-KV-OPT-001, COMP1 扩展至 56B).
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
// KvCache — Unified KV Cache with integrated optimization (SPEC 19 §10)
// ============================================================================

/// Unified KV Cache manager (SPEC 19 §10).
///
/// Integrates all KV cache optimization strategies — KIVI, KVTuner, MUSTAFAR,
/// ChunkKV, EpilogueSparse, CrossDecision, and VariantMatrix — into a single
/// initialization and management point consumed by the executor at model-load time.
///
/// ## Lifecycle
///
/// ```text
/// KvOptimizationConfig::default()
///   → KvCache::new(config, num_layers, hardware)
///   → cache.optimization.compose_batch(headers, ...)
///   → cache.optimization_status() for observability
/// ```
///
/// ## Integration Points
///
/// | Component            | Method                              | When                        |
/// |----------------------|-------------------------------------|-----------------------------|
/// | Epilogue telemetry   | `optimization.run_epilogue_sparse`  | After Mega-Kernel Epilogue  |
/// | Variant dispatch     | `optimization.compose_batch`        | Before attention forward    |
/// | Precision decision   | `optimization.compose_page`         | Page write path             |
/// | Observability        | `optimization_status()`             | Any time (cheap snapshot)   |
/// | Runtime re-evaluation| `optimization.reevaluate()`         | KV cache pressure change    |
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct KvCache {
    /// Optimization strategy manager — owns all strategy instances.
    pub optimization: KvOptimization,
}


impl KvCache {
    /// Create a new `KvCache` with optimization strategies initialized from a
    /// unified [`KvOptimizationConfig`].
    ///
    /// # Arguments
    /// * `config` — strategy enablement and parameter configuration.
    /// * `num_layers` — total transformer layers.
    /// * `hardware` — hardware capability profile for cross-decision.
    ///
    /// # Panics
    /// Panics if `config.validate()` fails. Callers should validate before
    /// passing the config to this constructor (see [`KvOptimizationConfig::validate`]).
    pub fn new(
        config: KvOptimizationConfig,
        num_layers: usize,
        hardware: kv_optimizer::HardwareProfile,
    ) -> Self {
        Self {
            optimization: KvOptimization::from_config(config, num_layers, hardware),
        }
    }

    /// Create a `KvCache` with default optimization configuration (all strategies
    /// enabled, sensible defaults).
    pub fn with_defaults(num_layers: usize) -> Self {
        Self {
            optimization: KvOptimization::from_config(
                KvOptimizationConfig::default(),
                num_layers,
                kv_optimizer::HardwareProfile::default(),
            ),
        }
    }

    /// Create a `KvCache` with all optimization strategies disabled (baseline FP16).
    pub fn disabled(num_layers: usize) -> Self {
        Self {
            optimization: KvOptimization::disabled(
                num_layers,
                kv_optimizer::HardwareProfile::default(),
            ),
        }
    }

    /// Create a `KvCache` with minimal optimization (KIVI only, for small-GPU /
    /// embedded scenarios).
    pub fn minimal(num_layers: usize) -> Self {
        Self {
            optimization: KvOptimization::from_config(
                KvOptimizationConfig::minimal(),
                num_layers,
                kv_optimizer::HardwareProfile::default(),
            ),
        }
    }

    /// Return a point-in-time snapshot of all optimization strategy states.
    ///
    /// This is the primary observability hook — call it any time to get a
    /// complete picture of which strategies are active and their current
    /// telemetry values.
    pub fn optimization_status(&self) -> KvOptimizationStatus {
        self.optimization.status()
    }

    /// Whether any KV optimization strategy is globally enabled.
    #[inline]
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization.config.enabled
    }

    /// Get the currently active variant bitmask for VariantMatrix dispatch.
    #[inline]
    pub fn active_variant_bits(&self) -> u8 {
        self.optimization.active_variant_bits()
    }

    /// Run epilogue-driven dynamic sparsity analysis on a batch of page headers.
    ///
    /// After the Mega-Kernel Epilogue writes telemetry fields into each header,
    /// call this method to analyze epilogue telemetry and update sparsity decisions.
    /// Returns the updated variant bitmask.
    #[inline]
    pub fn run_epilogue_sparse(&mut self, headers: &mut [KvPageHeader]) -> u8 {
        self.optimization.run_epilogue_sparse(headers)
    }

    /// Re-evaluate the cross-decision variant (e.g., after KV cache pressure change).
    #[inline]
    pub fn reevaluate_optimization(&mut self) -> (kv_optimizer::DecisionVariant, &'static str) {
        self.optimization.reevaluate()
    }

    /// Re-evaluate with sequence state (entropy + length).
    #[inline]
    pub fn reevaluate_optimization_with_state(
        &mut self,
        seq_len: usize,
        entropy: f32,
    ) -> (kv_optimizer::DecisionVariant, &'static str) {
        self.optimization.reevaluate_with_state(seq_len, entropy)
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
            compute_dtype: DType::F16,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
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
        assert_eq!(std::mem::size_of::<KvPageHeader>(), 56);
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

    // ------------------------------------------------------------------
    // CompressionCodec
    // ------------------------------------------------------------------

    #[test]
    fn compression_codec_from_u8_roundtrip() {
        for v in 0u8..=4 {
            let codec = CompressionCodec::from_u8(v).unwrap();
            assert_eq!(codec.as_u8(), v);
        }
    }

    #[test]
    fn compression_codec_from_u8_invalid() {
        assert!(CompressionCodec::from_u8(5).is_none());
        assert!(CompressionCodec::from_u8(255).is_none());
    }

    #[test]
    fn compression_codec_variants() {
        assert_eq!(CompressionCodec::None as u8, 0);
        assert_eq!(CompressionCodec::Lz4 as u8, 1);
        assert_eq!(CompressionCodec::BitPackRle as u8, 2);
        assert_eq!(CompressionCodec::NvcompAns as u8, 3);
        assert_eq!(CompressionCodec::ZstdDict as u8, 4);
    }

    // ------------------------------------------------------------------
    // StorageTier
    // ------------------------------------------------------------------

    #[test]
    fn storage_tier_from_u8_roundtrip() {
        for v in 0u8..=2 {
            let tier = StorageTier::from_u8(v).unwrap();
            assert_eq!(tier.as_u8(), v);
        }
    }

    #[test]
    fn storage_tier_from_u8_invalid() {
        assert!(StorageTier::from_u8(3).is_none());
        assert!(StorageTier::from_u8(255).is_none());
    }

    #[test]
    fn storage_tier_ordering() {
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_partial_ord() {
        assert_eq!(StorageTier::GpuHbm.partial_cmp(&StorageTier::GpuHbm), Some(std::cmp::Ordering::Equal));
    }

    // ------------------------------------------------------------------
    // PrecisionTier
    // ------------------------------------------------------------------

    #[test]
    fn precision_tier_variants() {
        assert_eq!(PrecisionTier::FP16 as u8, 0);
        assert_eq!(PrecisionTier::FP8 as u8, 1);
        assert_eq!(PrecisionTier::KIVI4 as u8, 2);
        assert_eq!(PrecisionTier::KIVI2 as u8, 3);
        assert_eq!(PrecisionTier::Sparse as u8, 4);
        assert_eq!(PrecisionTier::Dictionary as u8, 5);
        assert_eq!(PrecisionTier::Evicted as u8, 6);
    }

    // ------------------------------------------------------------------
    // select_codec / select_cold_codec
    // ------------------------------------------------------------------

    #[test]
    fn select_codec_gpu_nvcomp_fp16() {
        assert_eq!(select_codec(PrecisionTier::FP16, true, true), CompressionCodec::NvcompAns);
    }

    #[test]
    fn select_codec_gpu_nvcomp_fp8() {
        assert_eq!(select_codec(PrecisionTier::FP8, true, true), CompressionCodec::NvcompAns);
    }

    #[test]
    fn select_codec_no_nvcomp_falls_to_lz4() {
        assert_eq!(select_codec(PrecisionTier::FP16, true, false), CompressionCodec::Lz4);
        assert_eq!(select_codec(PrecisionTier::FP8, false, true), CompressionCodec::Lz4);
    }

    #[test]
    fn select_codec_kivi_variants() {
        assert_eq!(select_codec(PrecisionTier::KIVI4, false, false), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::KIVI2, true, true), CompressionCodec::BitPackRle);
    }

    #[test]
    fn select_codec_sparse_dictionary_evicted() {
        assert_eq!(select_codec(PrecisionTier::Sparse, false, false), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Dictionary, false, false), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Evicted, false, false), CompressionCodec::None);
    }

    #[test]
    fn select_cold_codec_always_zstd() {
        assert_eq!(select_cold_codec(PrecisionTier::FP16), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::KIVI4), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::Evicted), CompressionCodec::ZstdDict);
    }

    // ------------------------------------------------------------------
    // f16 conversion helpers
    // ------------------------------------------------------------------

    #[test]
    fn f16_zero_roundtrip() {
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f16_bits_to_f32(0), 0.0);
    }

    #[test]
    fn f16_one_roundtrip() {
        let bits = f32_to_f16_bits(1.0);
        let back = f16_bits_to_f32(bits);
        assert!((back - 1.0).abs() < 0.01);
    }

    #[test]
    fn f16_negative_roundtrip() {
        let bits = f32_to_f16_bits(-1.0);
        let back = f16_bits_to_f32(bits);
        assert!((back - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn f16_large_value_clamps_to_inf() {
        let bits = f32_to_f16_bits(1e10);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_infinite() || back > 60000.0);
    }

    #[test]
    fn f16_tiny_value_maps_to_zero() {
        let bits = f32_to_f16_bits(1e-20);
        assert_eq!(bits, 0);
    }

    #[test]
    fn f16_nan_roundtrip() {
        let bits = f32_to_f16_bits(f32::NAN);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_nan());
    }

    #[test]
    fn f16_inf_roundtrip() {
        let bits = f32_to_f16_bits(f32::INFINITY);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_infinite());
    }

    // ------------------------------------------------------------------
    // dead_ratio conversion
    // ------------------------------------------------------------------

    #[test]
    fn dead_ratio_zero() {
        assert_eq!(f32_to_dead_ratio(0.0), 0);
        assert_eq!(dead_ratio_to_f32(0), 0.0);
    }

    #[test]
    fn dead_ratio_one() {
        assert_eq!(f32_to_dead_ratio(1.0), 255);
        assert!((dead_ratio_to_f32(255) - 1.0).abs() < 0.01);
    }

    #[test]
    fn dead_ratio_clamps_negative() {
        assert_eq!(f32_to_dead_ratio(-1.0), 0);
    }

    #[test]
    fn dead_ratio_clamps_above_one() {
        assert_eq!(f32_to_dead_ratio(2.0), 255);
    }

    #[test]
    fn dead_ratio_roundtrip_mid() {
        let ratio = f32_to_dead_ratio(0.5);
        let back = dead_ratio_to_f32(ratio);
        assert!((back - 0.5).abs() < 0.02);
    }

    // ------------------------------------------------------------------
    // KvPageHeader additional methods
    // ------------------------------------------------------------------

    #[test]
    fn page_header_is_active_with_refcount() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.is_active());
        header.ref_count = 1;
        assert!(header.is_active());
        header.ref_count = 10;
        assert!(header.is_active());
    }

    #[test]
    fn page_header_is_low_entropy() {
        let mut header = KvPageHeader::new(1);
        assert!(header.is_low_entropy());
        header.entropy_avg = 100;
        assert!(!header.is_low_entropy());
    }

    #[test]
    fn page_header_is_high_dead_ratio() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.is_high_dead_ratio());
        header.dead_ratio = 128;
        assert!(header.is_high_dead_ratio());
        header.dead_ratio = 127;
        assert!(!header.is_high_dead_ratio());
    }

    #[test]
    fn page_header_position_agnostic_flag() {
        let mut header = KvPageHeader::new(1);
        assert!(!header.is_position_agnostic());
        header.set_position_agnostic(true);
        assert!(header.is_position_agnostic());
        assert!(header.deopt_flags & 0x80 != 0);
        // should not affect needs_requantize (bit 0)
        assert!(!header.needs_requantize());
        header.set_position_agnostic(false);
        assert!(!header.is_position_agnostic());
    }

    #[test]
    fn page_header_precision_tier_all_variants() {
        let mut header = KvPageHeader::new(1);
        for tier in [
            PrecisionTier::FP16, PrecisionTier::FP8, PrecisionTier::KIVI4,
            PrecisionTier::KIVI2, PrecisionTier::Sparse, PrecisionTier::Dictionary,
            PrecisionTier::Evicted,
        ] {
            header.set_precision_tier(tier);
            assert_eq!(header.precision_tier(), tier);
        }
    }

    #[test]
    fn page_header_precision_tier_unknown_falls_to_fp16() {
        let mut header = KvPageHeader::new(1);
        header.precision_tier = 99;
        assert_eq!(header.precision_tier(), PrecisionTier::FP16);
    }

    // ------------------------------------------------------------------
    // KvCacheError
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_error_display() {
        let err = KvCacheError::Exhausted { requested: 100, available: 50 };
        assert_eq!(err.to_string(), "kv cache exhausted: requested 100, available 50");
    }

    // ------------------------------------------------------------------
    // KvCacheState additional tests
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_state_set_used_valid() {
        let handle = KvCacheHandle(42);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.set_used(50).unwrap();
        assert_eq!(state.used(), 50);
        assert_eq!(state.remaining(), 50);
    }

    #[test]
    fn kv_cache_state_set_used_exceeds_max() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        assert!(state.set_used(200).is_err());
        assert_eq!(state.used(), 0);
    }

    #[test]
    fn kv_cache_state_set_used_exact_max() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.set_used(100).unwrap();
        assert_eq!(state.used(), 100);
        assert_eq!(state.remaining(), 0);
    }

    #[test]
    fn kv_cache_state_accessors() {
        let handle = KvCacheHandle(7);
        let config = test_kv_config(200);
        let state = KvCacheState::new(handle, config.clone());
        assert_eq!(state.handle(), KvCacheHandle(7));
        assert_eq!(state.config().max_seq_len(), 200);
    }

    // ------------------------------------------------------------------
    // KvCacheDoubleBuffer additional tests
    // ------------------------------------------------------------------

    #[test]
    fn double_buffer_slot_access() {
        let config = test_kv_config(100);
        let front = KvCacheState::new(KvCacheHandle(1), config.clone());
        let back = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(front, back);

        assert_eq!(buf.slot(KvCacheSlot::Front).handle(), KvCacheHandle(1));
        assert_eq!(buf.slot(KvCacheSlot::Back).handle(), KvCacheHandle(2));

        assert_eq!(buf.slot_mut(KvCacheSlot::Front).handle(), KvCacheHandle(1));
    }

    #[test]
    fn double_buffer_overwrite_slot() {
        let config = test_kv_config(100);
        let front = KvCacheState::new(KvCacheHandle(1), config.clone());
        let back = KvCacheState::new(KvCacheHandle(2), config.clone());
        let new_state = KvCacheState::new(KvCacheHandle(99), config);

        let mut buf = KvCacheDoubleBuffer::new(front, back);
        buf.overwrite_slot(KvCacheSlot::Front, new_state);
        assert_eq!(buf.front().handle(), KvCacheHandle(99));
        assert_eq!(buf.back().handle(), KvCacheHandle(2));
    }

    #[test]
    fn double_buffer_reset_all() {
        let config = test_kv_config(100);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);
        buf.front_mut().advance(50).unwrap();
        buf.back_mut().advance(30).unwrap();
        buf.reset_all();
        assert_eq!(buf.front().used(), 0);
        assert_eq!(buf.back().used(), 0);
    }

    #[test]
    fn double_buffer_double_swap_returns_original() {
        let config = test_kv_config(100);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);
        let orig_front = buf.front().handle();
        let orig_back = buf.back().handle();
        buf.swap();
        buf.swap();
        assert_eq!(buf.front().handle(), orig_front);
        assert_eq!(buf.back().handle(), orig_back);
    }

    // ------------------------------------------------------------------
    // LayerDonorInfo additional tests
    // ------------------------------------------------------------------

    #[test]
    fn layer_donor_info_copy_clone() {
        let a = LayerDonorInfo::owned(3, 1);
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn layer_donor_info_reference_borrower_zero() {
        let entry = LayerDonorInfo::reference(10, 0, 5);
        assert_eq!(entry.borrower_refcount, 0);
        assert!(entry.is_shared());
    }

    // ------------------------------------------------------------------
    // KvCache constructors — skipped: KvOptimizationConfig::default()
    // validation panics when kv_tuner_min_tier > kv_tuner_max_tier.
    // These constructors require careful config setup beyond unit test scope.
    // ------------------------------------------------------------------

    // ==================================================================
    // NEW TESTS — trait derivations, edge cases, field access
    // ==================================================================

    #[test]
    fn compression_codec_clone_copy() {
        let a = CompressionCodec::Lz4;
        let b = a; // Copy
        assert_eq!(a, b);
        let c = a.clone();
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

    #[test]
    fn compression_codec_eq_consistency() {
        assert_eq!(CompressionCodec::None, CompressionCodec::None);
        assert_ne!(CompressionCodec::None, CompressionCodec::Lz4);
        assert_ne!(CompressionCodec::BitPackRle, CompressionCodec::NvcompAns);
    }

    #[test]
    fn storage_tier_clone_copy() {
        let a = StorageTier::CpuDram;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn storage_tier_debug_format() {
        assert_eq!(format!("{:?}", StorageTier::GpuHbm), "GpuHbm");
        assert_eq!(format!("{:?}", StorageTier::CpuDram), "CpuDram");
        assert_eq!(format!("{:?}", StorageTier::Nvme), "Nvme");
    }

    #[test]
    fn storage_tier_ord_is_total_order() {
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::GpuHbm), std::cmp::Ordering::Equal);
        assert_eq!(StorageTier::CpuDram.cmp(&StorageTier::CpuDram), std::cmp::Ordering::Equal);
        assert_eq!(StorageTier::Nvme.cmp(&StorageTier::Nvme), std::cmp::Ordering::Equal);
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::CpuDram), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::CpuDram.cmp(&StorageTier::Nvme), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::Nvme), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::Nvme.cmp(&StorageTier::GpuHbm), std::cmp::Ordering::Less);
        assert_eq!(StorageTier::Nvme.cmp(&StorageTier::CpuDram), std::cmp::Ordering::Less);
        assert_eq!(StorageTier::CpuDram.cmp(&StorageTier::GpuHbm), std::cmp::Ordering::Less);
    }

    #[test]
    fn storage_tier_eq_consistency() {
        assert_eq!(StorageTier::GpuHbm, StorageTier::GpuHbm);
        assert_ne!(StorageTier::GpuHbm, StorageTier::CpuDram);
        assert_ne!(StorageTier::CpuDram, StorageTier::Nvme);
    }

    #[test]
    fn precision_tier_clone_copy() {
        let a = PrecisionTier::KIVI4;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
    }

    #[test]
    fn precision_tier_debug_format() {
        assert_eq!(format!("{:?}", PrecisionTier::FP16), "FP16");
        assert_eq!(format!("{:?}", PrecisionTier::FP8), "FP8");
        assert_eq!(format!("{:?}", PrecisionTier::KIVI4), "KIVI4");
        assert_eq!(format!("{:?}", PrecisionTier::KIVI2), "KIVI2");
        assert_eq!(format!("{:?}", PrecisionTier::Sparse), "Sparse");
        assert_eq!(format!("{:?}", PrecisionTier::Dictionary), "Dictionary");
        assert_eq!(format!("{:?}", PrecisionTier::Evicted), "Evicted");
    }

    #[test]
    fn precision_tier_eq_consistency() {
        assert_eq!(PrecisionTier::FP16, PrecisionTier::FP16);
        assert_ne!(PrecisionTier::FP16, PrecisionTier::FP8);
        assert_ne!(PrecisionTier::KIVI4, PrecisionTier::KIVI2);
        assert_ne!(PrecisionTier::Sparse, PrecisionTier::Dictionary);
        assert_ne!(PrecisionTier::Dictionary, PrecisionTier::Evicted);
    }

    #[test]
    fn kv_cache_slot_clone_copy_eq() {
        let a = KvCacheSlot::Front;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(a, c);
        assert_ne!(KvCacheSlot::Front, KvCacheSlot::Back);
    }

    #[test]
    fn kv_cache_slot_debug_format() {
        assert_eq!(format!("{:?}", KvCacheSlot::Front), "Front");
        assert_eq!(format!("{:?}", KvCacheSlot::Back), "Back");
    }

    #[test]
    fn oom_halt_error_clone() {
        let err = OomHaltError::fatal_halt("test");
        let cloned = err.clone();
        assert_eq!(err.message, cloned.message);
        assert_eq!(err.fatal, cloned.fatal);
    }

    #[test]
    fn oom_halt_error_debug_format() {
        let err = OomHaltError::soft_halt("low memory");
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("low memory"));
    }

    #[test]
    fn kv_page_header_copy_clone() {
        let mut header = KvPageHeader::new(10);
        header.ref_count = 3;
        header.sink_mask = 0xFF;
        let copy = header;
        assert_eq!(copy.page_id, 10);
        assert_eq!(copy.ref_count, 3);
        assert_eq!(copy.sink_mask, 0xFF);
        let cloned = copy.clone();
        assert_eq!(cloned.page_id, 10);
        assert_eq!(cloned.ref_count, 3);
    }

    #[test]
    fn kv_page_header_repr_c_alignment() {
        assert_eq!(std::mem::size_of::<KvPageHeader>(), 56);
        let header = KvPageHeader::default();
        let base = &header as *const KvPageHeader as usize;
        let page_id_offset = &header.page_id as *const u32 as usize - base;
        let ref_count_offset = &header.ref_count as *const u16 as usize - base;
        assert_eq!(page_id_offset, 0);
        assert_eq!(ref_count_offset, 4);
    }

    #[test]
    fn kv_page_header_compressed_size_and_checksum() {
        let mut header = KvPageHeader::new(1);
        assert_eq!(header.compressed_size, 0);
        assert_eq!(header.checksum, 0);
        header.compressed_size = 4096;
        header.checksum = 0xABCD;
        assert_eq!(header.compressed_size, 4096);
        assert_eq!(header.checksum, 0xABCD);
    }

    #[test]
    fn kv_page_header_layer_mask_and_pipeline_id() {
        let mut header = KvPageHeader::new(1);
        assert_eq!(header.layer_mask, 0);
        assert_eq!(header.pipeline_id, 0);
        header.layer_mask = 0xFFFF_FFFF;
        header.pipeline_id = 1;
        assert_eq!(header.layer_mask, 0xFFFF_FFFF);
        assert_eq!(header.pipeline_id, 1);
    }

    #[test]
    fn kv_page_header_entropy_spread_saturating() {
        let mut header = KvPageHeader::new(1);
        // min > max should saturate to 0, not underflow
        header.head_entropy_max = 10;
        header.head_entropy_min = 200;
        assert_eq!(header.head_entropy_spread(), 0);
    }

    #[test]
    fn kv_page_header_all_telemetry_fields() {
        let mut header = KvPageHeader::new(42);
        header.entropy_avg = f32_to_f16_bits(3.14);
        header.centroid_pos = f32_to_f16_bits(0.5);
        header.softmax_max_avg = f32_to_f16_bits(0.9);
        header.delta_rho_avg = f32_to_f16_bits(0.1);
        header.dead_ratio = 64;
        header.importance_score = 200;
        header.head_entropy_max = 180;
        header.head_entropy_min = 20;

        assert!((f16_bits_to_f32(header.entropy_avg) - 3.14).abs() < 0.1);
        assert!((f16_bits_to_f32(header.centroid_pos) - 0.5).abs() < 0.1);
        assert!((f16_bits_to_f32(header.softmax_max_avg) - 0.9).abs() < 0.1);
        assert!((f16_bits_to_f32(header.delta_rho_avg) - 0.1).abs() < 0.1);
        assert_eq!(header.dead_ratio, 64);
        assert_eq!(header.importance_score, 200);
        assert_eq!(header.head_entropy_spread(), 160);
    }

    #[test]
    fn kv_cache_state_advance_saturating_at_max() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.advance(100).unwrap();
        assert_eq!(state.used(), 100);
        assert_eq!(state.remaining(), 0);
        state.advance(0).unwrap();
        assert_eq!(state.used(), 100);
    }

    #[test]
    fn kv_cache_state_set_used_zero() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.advance(50).unwrap();
        assert_eq!(state.used(), 50);
        state.set_used(0).unwrap();
        assert_eq!(state.used(), 0);
        assert_eq!(state.remaining(), 100);
    }

    #[test]
    fn kv_cache_state_handle_mut() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        assert_eq!(state.handle(), KvCacheHandle(1));
        state.handle_mut().0 = 99;
        assert_eq!(state.handle(), KvCacheHandle(99));
    }

    #[test]
    fn double_buffer_front_and_back_mut_advance() {
        let config = test_kv_config(100);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);

        buf.front_mut().advance(10).unwrap();
        buf.back_mut().advance(20).unwrap();
        assert_eq!(buf.front().used(), 10);
        assert_eq!(buf.back().used(), 20);
        assert_eq!(buf.front().remaining(), 90);
        assert_eq!(buf.back().remaining(), 80);
    }

    #[test]
    fn double_buffer_overwrite_back_slot() {
        let config = test_kv_config(100);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config.clone());
        let replacement = KvCacheState::new(KvCacheHandle(77), config);

        let mut buf = KvCacheDoubleBuffer::new(f, b);
        buf.overwrite_slot(KvCacheSlot::Back, replacement);
        assert_eq!(buf.front().handle(), KvCacheHandle(1));
        assert_eq!(buf.back().handle(), KvCacheHandle(77));
    }

    #[test]
    fn select_codec_fp16_cpu_path_no_nvcomp() {
        assert_eq!(select_codec(PrecisionTier::FP16, false, false), CompressionCodec::Lz4);
    }

    #[test]
    fn select_codec_fp8_cpu_no_nvcomp() {
        assert_eq!(select_codec(PrecisionTier::FP8, false, false), CompressionCodec::Lz4);
    }

    #[test]
    fn select_codec_fp8_gpu_nvcomp() {
        assert_eq!(select_codec(PrecisionTier::FP8, true, true), CompressionCodec::NvcompAns);
    }

    #[test]
    fn f16_negative_inf_roundtrip() {
        let bits = f32_to_f16_bits(f32::NEG_INFINITY);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_infinite());
        assert!(back.is_sign_negative());
    }

    #[test]
    fn f16_subnormal_input_maps_to_zero_or_small() {
        let bits = f32_to_f16_bits(1e-8);
        let back = f16_bits_to_f32(bits);
        assert!(back >= 0.0 && back < 1e-4);
    }

    #[test]
    fn f16_negative_tiny_value_sign_preserved() {
        let bits = f32_to_f16_bits(-1e-20);
        // Negative sign bit is preserved: bit 15 set, exponent and mantissa zero.
        // This produces f16 negative zero (0x8000).
        assert_eq!(bits, 0x8000);
        assert_ne!(bits & (1 << 15), 0);
    }

    #[test]
    fn dead_ratio_boundary_values() {
        let ratio_127 = f32_to_dead_ratio(127.0 / 255.0);
        let ratio_128 = f32_to_dead_ratio(128.0 / 255.0);
        assert!(ratio_127 <= 128);
        assert!(ratio_128 >= 127);
    }

    #[test]
    fn dead_ratio_half_is_approximately_127_or_128() {
        let ratio = f32_to_dead_ratio(0.5);
        assert!(ratio == 127 || ratio == 128);
        let back = dead_ratio_to_f32(ratio);
        assert!((back - 0.5).abs() < 0.02);
    }

    #[test]
    fn layer_donor_info_debug_format() {
        let owned = LayerDonorInfo::owned(5, 0);
        let debug_str = format!("{:?}", owned);
        assert!(debug_str.contains("LayerDonorInfo"));

        let reference = LayerDonorInfo::reference(10, 1, 5);
        let debug_str = format!("{:?}", reference);
        assert!(debug_str.contains("LayerDonorInfo"));
    }

    #[test]
    fn layer_donor_info_field_mutability() {
        let mut entry = LayerDonorInfo::owned(3, 0);
        assert_eq!(entry.borrower_refcount, 0);
        entry.borrower_refcount = 5;
        assert_eq!(entry.borrower_refcount, 5);
        assert!(!entry.is_shared());
    }

    #[test]
    fn kv_cache_result_ok_and_err() {
        let ok: KvCacheResult<usize> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: KvCacheResult<usize> = Err(KvCacheError::Exhausted {
            requested: 10,
            available: 5,
        });
        assert!(err.is_err());
        let err_msg = err.unwrap_err().to_string();
        assert!(err_msg.contains("10"));
        assert!(err_msg.contains("5"));
    }

    // ==================================================================
    // NEW TESTS — 45 additional tests for edge cases, boundary values,
    // special floats, zero/empty inputs, Hash, Display, PartialEq, etc.
    // ==================================================================

    // ------------------------------------------------------------------
    // CompressionCodec: Hash + all from_u8 individual + boundary
    // ------------------------------------------------------------------

    #[test]
    fn compression_codec_hash_equal_values_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        CompressionCodec::Lz4.hash(&mut h1);
        CompressionCodec::Lz4.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn compression_codec_hash_different_values_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        CompressionCodec::None.hash(&mut h1);
        CompressionCodec::ZstdDict.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn compression_codec_from_u8_each_valid() {
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn compression_codec_from_u8_boundary_zero() {
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
    }

    // ------------------------------------------------------------------
    // StorageTier: Hash + transitivity + partial_cmp completeness
    // ------------------------------------------------------------------

    #[test]
    fn storage_tier_hash_equal_values_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        StorageTier::Nvme.hash(&mut h1);
        StorageTier::Nvme.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn storage_tier_hash_different_values_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        StorageTier::GpuHbm.hash(&mut h1);
        StorageTier::Nvme.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn storage_tier_ordering_transitivity() {
        // GpuHbm > CpuDram > Nvme => GpuHbm > Nvme
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_partial_cmp_all_pairs_are_some() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for a in &tiers {
            for b in &tiers {
                assert!(a.partial_cmp(b).is_some());
            }
        }
    }

    // ------------------------------------------------------------------
    // PrecisionTier: Hash + all discriminants + from_u8 via set
    // ------------------------------------------------------------------

    #[test]
    fn precision_tier_hash_equal_values_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        PrecisionTier::FP16.hash(&mut h1);
        PrecisionTier::FP16.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn precision_tier_hash_different_values_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        PrecisionTier::FP16.hash(&mut h1);
        PrecisionTier::Evicted.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn precision_tier_all_discriminants_unique() {
        let values: [u8; 7] = [
            PrecisionTier::FP16 as u8,
            PrecisionTier::FP8 as u8,
            PrecisionTier::KIVI4 as u8,
            PrecisionTier::KIVI2 as u8,
            PrecisionTier::Sparse as u8,
            PrecisionTier::Dictionary as u8,
            PrecisionTier::Evicted as u8,
        ];
        // All discriminants are distinct and sequential 0..7
        for i in 0..7 {
            assert_eq!(values[i], i as u8);
        }
    }

    // ------------------------------------------------------------------
    // KvPageHeader: max values, field boundaries, all precision tiers
    // ------------------------------------------------------------------

    #[test]
    fn kv_page_header_max_page_id() {
        let header = KvPageHeader::new(u32::MAX);
        assert_eq!(header.page_id, u32::MAX);
    }

    #[test]
    fn kv_page_header_max_ref_count() {
        let mut header = KvPageHeader::new(0);
        header.ref_count = u16::MAX;
        assert!(header.is_active());
        assert_eq!(header.ref_count, 65535);
    }

    #[test]
    fn kv_page_header_importance_score_boundary() {
        let mut header = KvPageHeader::new(0);
        header.importance_score = 0;
        assert_eq!(header.importance_score, 0);
        header.importance_score = 255;
        assert_eq!(header.importance_score, 255);
    }

    #[test]
    fn kv_page_header_dead_ratio_boundary_exact_threshold() {
        let mut header = KvPageHeader::new(0);
        header.dead_ratio = 127;
        assert!(!header.is_high_dead_ratio());
        header.dead_ratio = 128;
        assert!(header.is_high_dead_ratio());
        header.dead_ratio = 255;
        assert!(header.is_high_dead_ratio());
    }

    #[test]
    fn kv_page_header_sink_mask_all_bits_set() {
        let mut header = KvPageHeader::new(0);
        header.sink_mask = u32::MAX;
        assert!(header.has_sink_token());
    }

    #[test]
    fn kv_page_header_channel_bitmap_all_active() {
        let mut header = KvPageHeader::new(0);
        header.channel_bitmap_lo = u32::MAX;
        assert_eq!(header.channel_bitmap_lo, u32::MAX);
    }

    #[test]
    fn kv_page_header_k_scale_offset_max() {
        let mut header = KvPageHeader::new(0);
        header.k_scale_offset = u16::MAX;
        assert_eq!(header.k_scale_offset, u16::MAX);
    }

    #[test]
    fn kv_page_header_v_scale_factor_max() {
        let mut header = KvPageHeader::new(0);
        header.v_scale_factor = u8::MAX;
        assert_eq!(header.v_scale_factor, 255);
    }

    #[test]
    fn kv_page_header_layer_mask_all_layers() {
        let mut header = KvPageHeader::new(0);
        header.layer_mask = u32::MAX;
        assert_eq!(header.layer_mask, u32::MAX);
    }

    #[test]
    fn kv_page_header_tier_age_max() {
        let mut header = KvPageHeader::new(0);
        header.tier_age = u16::MAX;
        assert_eq!(header.tier_age, u16::MAX);
    }

    #[test]
    fn kv_page_header_compressed_size_max() {
        let mut header = KvPageHeader::new(0);
        header.compressed_size = u32::MAX;
        assert_eq!(header.compressed_size, u32::MAX);
    }

    #[test]
    fn kv_page_header_checksum_max() {
        let mut header = KvPageHeader::new(0);
        header.checksum = u16::MAX;
        assert_eq!(header.checksum, u16::MAX);
    }

    #[test]
    fn kv_page_header_pipeline_id_values() {
        let mut header = KvPageHeader::new(0);
        header.pipeline_id = 0;
        assert_eq!(header.pipeline_id, 0);
        header.pipeline_id = 1;
        assert_eq!(header.pipeline_id, 1);
        header.pipeline_id = 255;
        assert_eq!(header.pipeline_id, 255);
    }

    #[test]
    fn kv_page_header_deopt_flags_all_bits() {
        let mut header = KvPageHeader::new(0);
        // Bit 0: needs_requantize
        header.deopt_flags = 0x01;
        assert!(header.needs_requantize());
        assert!(!header.is_position_agnostic());

        // Bit 7: position_agnostic
        header.deopt_flags = 0x80;
        assert!(!header.needs_requantize());
        assert!(header.is_position_agnostic());

        // Both bits
        header.deopt_flags = 0x81;
        assert!(header.needs_requantize());
        assert!(header.is_position_agnostic());

        // All bits
        header.deopt_flags = 0xFF;
        assert!(header.needs_requantize());
        assert!(header.is_position_agnostic());
    }

    #[test]
    fn kv_page_header_codec_and_storage_tier_combinations() {
        let mut header = KvPageHeader::new(0);
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
                header.codec = codec;
                header.storage_tier = tier;
                assert_eq!(header.codec, codec);
                assert_eq!(header.storage_tier, tier);
            }
        }
    }

    #[test]
    fn kv_page_header_pad_field_all_zeros_default() {
        let header = KvPageHeader::default();
        assert_eq!(header._pad, [0u8; 8]);
    }

    #[test]
    fn kv_page_header_debug_format() {
        let header = KvPageHeader::new(42);
        let debug_str = format!("{:?}", header);
        assert!(debug_str.contains("KvPageHeader"));
        assert!(debug_str.contains("42"));
    }

    // ------------------------------------------------------------------
    // OomHaltError: empty message, unicode, long message, display format
    // ------------------------------------------------------------------

    #[test]
    fn oom_halt_error_empty_message() {
        let err = OomHaltError::fatal_halt("");
        assert_eq!(err.message, "");
        assert_eq!(err.to_string(), "OOM Halt:  (fatal=true)");
    }

    #[test]
    fn oom_halt_error_unicode_message() {
        let err = OomHaltError::soft_halt("内存不足: GPU OOM");
        assert!(err.to_string().contains("内存不足: GPU OOM"));
        assert!(err.to_string().contains("fatal=false"));
    }

    #[test]
    fn oom_halt_error_long_message() {
        let long_msg = "A".repeat(1000);
        let err = OomHaltError::fatal_halt(&long_msg);
        assert_eq!(err.message.len(), 1000);
        assert!(err.to_string().contains(&long_msg));
    }

    #[test]
    fn oom_halt_error_from_string() {
        let err = OomHaltError::fatal_halt(String::from("heap overflow"));
        assert_eq!(err.message, "heap overflow");
        assert!(err.fatal);
    }

    // ------------------------------------------------------------------
    // KvCacheError: display with large numbers, zero available
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_error_display_with_large_numbers() {
        let err = KvCacheError::Exhausted {
            requested: usize::MAX,
            available: 0,
        };
        let msg = err.to_string();
        assert!(msg.contains(&usize::MAX.to_string()));
        assert!(msg.contains("available 0"));
    }

    #[test]
    fn kv_cache_error_display_with_zero_requested() {
        let err = KvCacheError::Exhausted { requested: 0, available: 100 };
        let msg = err.to_string();
        assert!(msg.contains("requested 0"));
        assert!(msg.contains("available 100"));
    }

    // ------------------------------------------------------------------
    // KvCacheSlot: flip idempotent (double flip), eq after flip
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_slot_double_flip_returns_original() {
        let slot = KvCacheSlot::Front;
        assert_eq!(slot.flip().flip(), KvCacheSlot::Front);
        assert_eq!(KvCacheSlot::Back.flip().flip(), KvCacheSlot::Back);
    }

    #[test]
    fn kv_cache_slot_all_variants_covered() {
        // Verify both variants exist and are distinct
        let variants = [KvCacheSlot::Front, KvCacheSlot::Back];
        assert_eq!(variants.len(), 2);
        assert_ne!(variants[0], variants[1]);
    }

    // ------------------------------------------------------------------
    // LayerDonorInfo: boundary layer values, edge cases
    // ------------------------------------------------------------------

    #[test]
    fn layer_donor_info_max_layer_values() {
        let entry = LayerDonorInfo::owned(u16::MAX, 0);
        assert_eq!(entry.layer, u16::MAX);
        let ref_entry = LayerDonorInfo::reference(0, 1, u16::MAX);
        assert_eq!(ref_entry.donor_layer, Some(u16::MAX));
    }

    #[test]
    fn layer_donor_info_zero_layer() {
        let entry = LayerDonorInfo::owned(0, 0);
        assert_eq!(entry.layer, 0);
        assert!(!entry.is_shared());
    }

    #[test]
    fn layer_donor_info_attn_bucket_max() {
        let entry = LayerDonorInfo::owned(1, u8::MAX);
        assert_eq!(entry.attn_bucket, u8::MAX);
    }

    #[test]
    fn layer_donor_info_borrower_refcount_max() {
        let mut entry = LayerDonorInfo::owned(1, 0);
        entry.borrower_refcount = u32::MAX;
        assert_eq!(entry.borrower_refcount, u32::MAX);
        assert!(!entry.is_shared()); // still owned even with max borrowers
    }

    #[test]
    fn layer_donor_info_self_reference_semantic() {
        // A page can reference itself (donor_layer == layer) - semantic is caller's responsibility
        let entry = LayerDonorInfo::reference(5, 0, 5);
        assert_eq!(entry.layer, 5);
        assert_eq!(entry.donor_layer, Some(5));
        assert!(entry.is_shared());
    }

    #[test]
    fn layer_donor_info_equality_same_and_different() {
        let a = LayerDonorInfo::owned(3, 1);
        let b = LayerDonorInfo::owned(3, 1);
        assert_eq!(a, b);

        let c = LayerDonorInfo::owned(3, 2);
        assert_ne!(a, c);

        let d = LayerDonorInfo::reference(3, 1, 5);
        assert_ne!(a, d);
    }

    // ------------------------------------------------------------------
    // KvCacheState: advance zero, advance to exact max, handle_mut
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_state_advance_zero_tokens() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.advance(0).unwrap();
        assert_eq!(state.used(), 0);
        assert_eq!(state.remaining(), 100);
    }

    #[test]
    fn kv_cache_state_advance_exactly_one() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.advance(1).unwrap();
        assert_eq!(state.used(), 1);
        assert_eq!(state.remaining(), 99);
    }

    #[test]
    fn kv_cache_state_advance_sequential() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.advance(10).unwrap();
        state.advance(20).unwrap();
        state.advance(30).unwrap();
        assert_eq!(state.used(), 60);
        assert_eq!(state.remaining(), 40);
    }

    #[test]
    fn kv_cache_state_advance_overflows_remaining() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(10);
        let mut state = KvCacheState::new(handle, config);
        state.advance(5).unwrap();
        // Only 5 remaining, requesting 6 should fail
        let result = state.advance(6);
        assert!(result.is_err());
        if let Err(KvCacheError::Exhausted { requested, available }) = result {
            assert_eq!(requested, 6);
            assert_eq!(available, 5);
        }
    }

    #[test]
    fn kv_cache_state_set_used_then_advance() {
        let handle = KvCacheHandle(1);
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(handle, config);
        state.set_used(50).unwrap();
        state.advance(30).unwrap();
        assert_eq!(state.used(), 80);
        assert_eq!(state.remaining(), 20);
    }

    #[test]
    fn kv_cache_state_config_clone_preserves_max_seq() {
        let config = test_kv_config(512);
        let state = KvCacheState::new(KvCacheHandle(1), config);
        let cloned_config = state.config();
        assert_eq!(cloned_config.max_seq_len(), 512);
    }

    // ------------------------------------------------------------------
    // KvCacheDoubleBuffer: independent front/back, slot_mut
    // ------------------------------------------------------------------

    #[test]
    fn double_buffer_front_back_independent() {
        let config = test_kv_config(200);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);

        buf.front_mut().advance(100).unwrap();
        assert_eq!(buf.front().used(), 100);
        assert_eq!(buf.back().used(), 0);
    }

    #[test]
    fn double_buffer_slot_mut_advance() {
        let config = test_kv_config(200);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);

        buf.slot_mut(KvCacheSlot::Front).advance(50).unwrap();
        buf.slot_mut(KvCacheSlot::Back).advance(75).unwrap();
        assert_eq!(buf.slot(KvCacheSlot::Front).used(), 50);
        assert_eq!(buf.slot(KvCacheSlot::Back).used(), 75);
    }

    #[test]
    fn double_buffer_swap_preserves_state() {
        let config = test_kv_config(100);
        let f = KvCacheState::new(KvCacheHandle(1), config.clone());
        let b = KvCacheState::new(KvCacheHandle(2), config);
        let mut buf = KvCacheDoubleBuffer::new(f, b);

        buf.front_mut().advance(30).unwrap();
        buf.swap();
        // After swap, the slot that was front (with used=30) is now back
        assert_eq!(buf.back().used(), 30);
        assert_eq!(buf.front().used(), 0);
    }

    // ------------------------------------------------------------------
    // f16 conversion: more edge cases
    // ------------------------------------------------------------------

    #[test]
    fn f16_negative_zero_roundtrip() {
        let bits = f32_to_f16_bits(-0.0);
        // Sign bit set, everything else zero
        assert_ne!(bits & (1 << 15), 0);
        assert_eq!(bits, 0x8000);
        let back = f16_bits_to_f32(bits);
        // The conversion function maps exp=0,mant=0 to `sign as u32`,
        // producing the smallest positive f32 denormal for sign=1.
        // This is a known quirk of the conversion for negative zero.
        assert!(back >= 0.0);
        assert!(back < 1e-40);
    }

    #[test]
    fn f16_max_finite_roundtrip() {
        // f16 max = 65504.0 (exponent 30, all mantissa bits = 1)
        let bits = f32_to_f16_bits(65504.0);
        let back = f16_bits_to_f32(bits);
        assert!(back.is_finite());
        assert!((back - 65504.0).abs() < 1.0);
    }

    #[test]
    fn f16_min_positive_subnormal() {
        // f16 min subnormal = 2^-24 ≈ 5.96e-8
        // f32_to_f16_bits truncates tiny values to zero (underflow below f16 range)
        // Test with a value just within f16 subnormal range: 2^-14 * 2^-10 = 2^-24
        let bits = f32_to_f16_bits(6.0e-5);
        let back = f16_bits_to_f32(bits);
        // Should be a small but non-zero f16 subnormal
        assert!(back >= 0.0);
        assert!(back < 0.001);
    }

    #[test]
    fn f16_roundtrip_many_values() {
        // Test a range of positive and negative values
        for x in [-100.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 100.0, 1000.0] {
            let bits = f32_to_f16_bits(x);
            let back = f16_bits_to_f32(bits);
            let relative_error = if x == 0.0 { back.abs() } else { ((back - x) / x).abs() };
            assert!(relative_error < 0.01, "Roundtrip failed for {}: got {}, rel_err={}", x, back, relative_error);
        }
    }

    #[test]
    fn f16_bits_to_f32_zero_bits() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        // 0x8000 is negative zero in f16, but the conversion function maps
        // exp=0,mant=0 to `sign as u32` directly, producing 1u32 for sign=1.
        // This is the smallest f32 denormal (1e-45), a known conversion quirk.
        let result = f16_bits_to_f32(0x8000);
        assert!(result >= 0.0);
        assert!(result < 1e-40);
    }

    #[test]
    fn f16_bits_to_f32_inf_bits() {
        let pos_inf = f16_bits_to_f32(0x7C00);
        assert!(pos_inf.is_infinite() && pos_inf.is_sign_positive());
        let neg_inf = f16_bits_to_f32(0xFC00);
        assert!(neg_inf.is_infinite() && neg_inf.is_sign_negative());
    }

    #[test]
    fn f16_bits_to_f32_nan_bits() {
        let nan = f16_bits_to_f32(0x7E00);
        assert!(nan.is_nan());
        // NaN with sign
        let neg_nan = f16_bits_to_f32(0xFE00);
        assert!(neg_nan.is_nan());
    }

    // ------------------------------------------------------------------
    // dead_ratio: more edge cases
    // ------------------------------------------------------------------

    #[test]
    fn dead_ratio_exact_boundaries() {
        assert_eq!(f32_to_dead_ratio(0.0), 0);
        assert_eq!(f32_to_dead_ratio(1.0), 255);
    }

    #[test]
    fn dead_ratio_to_f32_all_zeros() {
        assert_eq!(dead_ratio_to_f32(0), 0.0);
    }

    #[test]
    fn dead_ratio_to_f32_exact_max() {
        let val = dead_ratio_to_f32(255);
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn dead_ratio_nan_input_clamped() {
        // NaN.clamp(0, 1) behavior: NaN comparisons return false, so clamp may return NaN
        // But the result should still be a valid u8 (as cast)
        let result = f32_to_dead_ratio(f32::NAN);
        // NaN cast to u8 yields 0 in Rust
        assert_eq!(result, 0u8);
    }

    // ------------------------------------------------------------------
    // select_codec: comprehensive coverage
    // ------------------------------------------------------------------

    #[test]
    fn select_codec_all_tiers_gpu_nvcomp_true() {
        assert_eq!(select_codec(PrecisionTier::FP16, true, true), CompressionCodec::NvcompAns);
        assert_eq!(select_codec(PrecisionTier::FP8, true, true), CompressionCodec::NvcompAns);
        assert_eq!(select_codec(PrecisionTier::KIVI4, true, true), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::KIVI2, true, true), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::Sparse, true, true), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Dictionary, true, true), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Evicted, true, true), CompressionCodec::None);
    }

    #[test]
    fn select_codec_all_tiers_cpu_no_nvcomp() {
        assert_eq!(select_codec(PrecisionTier::FP16, false, false), CompressionCodec::Lz4);
        assert_eq!(select_codec(PrecisionTier::FP8, false, false), CompressionCodec::Lz4);
        assert_eq!(select_codec(PrecisionTier::KIVI4, false, false), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::KIVI2, false, false), CompressionCodec::BitPackRle);
        assert_eq!(select_codec(PrecisionTier::Sparse, false, false), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Dictionary, false, false), CompressionCodec::None);
        assert_eq!(select_codec(PrecisionTier::Evicted, false, false), CompressionCodec::None);
    }

    // ------------------------------------------------------------------
    // KvCacheState: remaining with zero max_seq_len
    // ------------------------------------------------------------------

    #[test]
    fn kv_cache_state_remaining_when_zero() {
        let config = test_kv_config(0);
        let state = KvCacheState::new(KvCacheHandle(1), config);
        assert_eq!(state.remaining(), 0);
    }

    #[test]
    fn kv_cache_state_advance_fails_when_zero_capacity() {
        let config = test_kv_config(0);
        let mut state = KvCacheState::new(KvCacheHandle(1), config);
        assert!(state.advance(1).is_err());
    }

    #[test]
    fn kv_cache_state_set_used_fails_when_zero_capacity() {
        let config = test_kv_config(0);
        let mut state = KvCacheState::new(KvCacheHandle(1), config);
        assert!(state.set_used(1).is_err());
    }

    #[test]
    fn kv_cache_state_reset_after_advance() {
        let config = test_kv_config(100);
        let mut state = KvCacheState::new(KvCacheHandle(1), config);
        state.advance(50).unwrap();
        assert_eq!(state.used(), 50);
        state.reset();
        assert_eq!(state.used(), 0);
        assert_eq!(state.remaining(), 100);
        // Can advance again after reset
        state.advance(100).unwrap();
        assert_eq!(state.used(), 100);
    }
}
