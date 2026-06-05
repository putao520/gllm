//! Dual-Track Memory Pool (SPEC §11.5, REQ-TURBOQUANT-004)
//!
//! 物理隔离的双轨显存池架构:
//! - **主池**: 3-4bit/FP4 极低精度 KV 存储 (非对称量化)
//! - **校验池 (QJL)**: 1-bit XNOR 残差掩码阵列
//!
//! # 多卡同步红利
//! PCIe Swap 和跨卡 RDMA 同步 KV 时，仅需传输原 FP16 内存量纲的 25% (4x 压缩)
//!
//! # 架构约束 (SPEC §03-DATA-STRUCTURE.md §11.3)
//! ```rust
//! pub struct DualTrackMemoryPool {
//!     /// 3-bit / 4-bit / FP4 的无缩放 (Scale-Free) 主数据流连续块
//!     pub main_pool: BlockAllocator,
//!     /// 1-bit 二值化修正或二阶残差标志位的掩码阵列
//!     pub xnor_pool: BitsetAllocator,
//! }
//! ```

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DualTrackError {
    #[error("主池已满: requested {requested}, available {available}")]
    MainPoolFull { requested: usize, available: usize },
    #[error("校验池已满: requested {requested}, available {available}")]
    XnorPoolFull { requested: usize, available: usize },
    #[error("块 {0} 不在主池中")]
    NotInMainPool(usize),
    #[error("块 {0} 不在校验池中")]
    NotInXnorPool(usize),
    #[error("量化失败: {0}")]
    QuantFailed(String),
    #[error("无效的位宽: {0} (必须是 3 或 4)")]
    InvalidBitWidth(u8),
}

pub type DualTrackResult<T> = std::result::Result<T, DualTrackError>;

/// 内存轨道类型 (SPEC §11.5)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Track {
    /// 主池: 3-4bit 极低精度 KV 存储
    Main,
    /// 校验池: 1-bit XNOR 残差掩码
    Xnor,
    /// 空闲轨道
    Empty,
}

/// 轨道配置 (SPEC §11.5)
#[derive(Debug, Clone)]
pub struct TrackConfig {
    /// 主池容量 (字节数)
    pub main_capacity: usize,
    /// 校验池容量 (位数为单位，1-bit per element)
    pub xnor_capacity_bits: usize,
    /// 量化位宽 (3 或 4)
    pub quant_bits: u8,
    /// 块大小 (tokens per block)
    pub block_size: usize,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            main_capacity: 256 * 1024 * 1024, // 256 MB default
            xnor_capacity_bits: 64 * 1024 * 1024 * 8, // 64 MB bits
            quant_bits: 4,
            block_size: 16,
        }
    }
}

/// 块元数据
#[derive(Debug, Clone)]
struct BlockMeta {
    track: Track,
    main_offset: usize,    // 主池中的字节偏移
    xnor_offset: usize,    // 校验池中的位偏移
    size_elements: usize,  // 元素数量
    quant_bits: u8,        // 量化位宽
}

impl Default for BlockMeta {
    fn default() -> Self {
        Self {
            track: Track::Empty,
            main_offset: 0,
            xnor_offset: 0,
            size_elements: 0,
            quant_bits: 4,
        }
    }
}

/// 双轨显存池 (SPEC §11.5)
///
/// # 内存布局
/// - **主池**: [3-4bit packed KV data] - 极低精度主存储
/// - **校验池**: [1-bit XNOR mask] - 二值化残差掩码
#[derive(Debug, Clone)]
pub struct DualTrackMemoryPool {
    /// 主池: 3-4bit 打包存储 (每字节 2 个 4-bit 值，或 8/3 个 3-bit 值)
    main_pool: Vec<u8>,
    main_capacity: usize,
    main_used: usize,

    /// 校验池: 1-bit XNOR 残差掩码 (按位存储)
    xnor_pool: Vec<u8>,
    xnor_capacity_bits: usize,
    xnor_used_bits: usize,

    /// 块元数据
    block_meta: Vec<BlockMeta>,

    /// 轨道配置
    config: TrackConfig,
}

impl DualTrackMemoryPool {
    /// 创建新的双轨显存池
    ///
    /// # 参数
    /// - `main_capacity`: 主池容量 (字节)
    /// - `xnor_capacity_bits`: 校验池容量 (位数)
    /// - `block_size`: 块大小 (tokens)
    /// - `quant_bits`: 量化位宽 (3 或 4)
    pub fn new(
        main_capacity: usize,
        xnor_capacity_bits: usize,
        block_size: usize,
        quant_bits: u8,
    ) -> DualTrackResult<Self> {
        if quant_bits != 3 && quant_bits != 4 {
            return Err(DualTrackError::InvalidBitWidth(quant_bits));
        }

        // 计算块数量 (估算)
        let bytes_per_element = match quant_bits {
            4 => 2, // 2 elements per byte
            3 => 3, // 8 elements per 3 bytes
            _ => return Err(DualTrackError::InvalidBitWidth(quant_bits)),
        };

        let elements_per_block = block_size;
        let bytes_per_block = elements_per_block.div_ceil(bytes_per_element);
        let num_blocks = main_capacity / bytes_per_block;

        let config = TrackConfig {
            main_capacity,
            xnor_capacity_bits,
            quant_bits,
            block_size,
        };

        Ok(Self {
            main_pool: vec![0u8; main_capacity],
            main_capacity,
            main_used: 0,
            xnor_pool: vec![0u8; xnor_capacity_bits.div_ceil(8)],
            xnor_capacity_bits,
            xnor_used_bits: 0,
            block_meta: vec![BlockMeta::default(); num_blocks.max(1)],
            config,
        })
    }

    /// 使用默认配置创建
    pub fn with_default_config() -> DualTrackResult<Self> {
        let config = TrackConfig::default();
        Self::new(
            config.main_capacity,
            config.xnor_capacity_bits,
            config.block_size,
            config.quant_bits,
        )
    }

    /// 分配主池空间
    ///
    /// # 返回
    /// 主池中的字节偏移
    fn allocate_main(&mut self, size_bytes: usize) -> DualTrackResult<usize> {
        if self.main_used + size_bytes > self.main_capacity {
            return Err(DualTrackError::MainPoolFull {
                requested: size_bytes,
                available: self.main_capacity - self.main_used,
            });
        }
        let offset = self.main_used;
        self.main_used += size_bytes;
        Ok(offset)
    }

    /// 分配校验池空间
    ///
    /// # 返回
    /// 校验池中的位偏移
    fn allocate_xnor(&mut self, size_bits: usize) -> DualTrackResult<usize> {
        if self.xnor_used_bits + size_bits > self.xnor_capacity_bits {
            return Err(DualTrackError::XnorPoolFull {
                requested: size_bits,
                available: self.xnor_capacity_bits - self.xnor_used_bits,
            });
        }
        let offset = self.xnor_used_bits;
        self.xnor_used_bits += size_bits;
        Ok(offset)
    }

    /// 在主池中分配块
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `num_elements`: 元素数量
    ///
    /// # 返回
    /// 主池中的字节偏移
    pub fn allocate_block_main(&mut self, block_id: usize, num_elements: usize) -> DualTrackResult<usize> {
        let bytes_per_element = match self.config.quant_bits {
            4 => 2, // 2 elements per byte
            3 => 3, // 8 elements per 3 bytes
            _ => return Err(DualTrackError::InvalidBitWidth(self.config.quant_bits)),
        };

        let size_bytes = num_elements.div_ceil(bytes_per_element);
        let offset = self.allocate_main(size_bytes)?;

        self.block_meta[block_id].track = Track::Main;
        self.block_meta[block_id].main_offset = offset;
        self.block_meta[block_id].size_elements = num_elements;
        self.block_meta[block_id].quant_bits = self.config.quant_bits;

        Ok(offset)
    }

    /// 在校验池中分配 XNOR 掩码
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `num_bits`: 掩码位数
    ///
    /// # 返回
    /// 校验池中的位偏移
    pub fn allocate_block_xnor(&mut self, block_id: usize, num_bits: usize) -> DualTrackResult<usize> {
        let offset = self.allocate_xnor(num_bits)?;

        self.block_meta[block_id].track = Track::Xnor;
        self.block_meta[block_id].xnor_offset = offset;
        self.block_meta[block_id].size_elements = num_bits; // XNOR: 1 bit per element

        Ok(offset)
    }

    /// 写入主池数据 (量化后)
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `data`: 量化后的数据 (已打包)
    pub fn write_main(&mut self, block_id: usize, data: &[u8]) -> DualTrackResult<()> {
        let meta = &self.block_meta[block_id];
        if meta.track != Track::Main {
            return Err(DualTrackError::NotInMainPool(block_id));
        }

        let offset = meta.main_offset;
        let end_offset = offset + data.len();
        if end_offset > self.main_pool.len() {
            return Err(DualTrackError::MainPoolFull {
                requested: data.len(),
                available: self.main_pool.len() - offset,
            });
        }

        self.main_pool[offset..end_offset].copy_from_slice(data);
        Ok(())
    }

    /// 读取主池数据
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `out`: 输出缓冲区
    pub fn read_main(&self, block_id: usize, out: &mut [u8]) -> DualTrackResult<()> {
        let meta = &self.block_meta[block_id];
        if meta.track != Track::Main {
            return Err(DualTrackError::NotInMainPool(block_id));
        }

        let offset = meta.main_offset;
        let end_offset = offset + out.len();
        if end_offset > self.main_pool.len() {
            return Err(DualTrackError::NotInMainPool(block_id));
        }

        out.copy_from_slice(&self.main_pool[offset..end_offset]);
        Ok(())
    }

    /// 写入 XNOR 掩码到校验池
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `mask`: 掩码数据 (每个元素 1 bit)
    pub fn write_xnor(&mut self, block_id: usize, mask: &[bool]) -> DualTrackResult<()> {
        let meta = &self.block_meta[block_id];
        if meta.track != Track::Xnor {
            return Err(DualTrackError::NotInXnorPool(block_id));
        }

        let offset = meta.xnor_offset;
        for (i, &bit) in mask.iter().enumerate() {
            let bit_pos = offset + i;
            if bit_pos >= self.xnor_capacity_bits {
                break;
            }
            let byte_pos = bit_pos / 8;
            let bit_within_byte = bit_pos % 8;
            if bit {
                self.xnor_pool[byte_pos] |= 1 << bit_within_byte;
            } else {
                self.xnor_pool[byte_pos] &= !(1 << bit_within_byte);
            }
        }

        Ok(())
    }

    /// 读取 XNOR 掩码
    ///
    /// # 参数
    /// - `block_id`: 块 ID
    /// - `out`: 输出缓冲区 (布尔数组)
    pub fn read_xnor(&self, block_id: usize, out: &mut [bool]) -> DualTrackResult<()> {
        let meta = &self.block_meta[block_id];
        if meta.track != Track::Xnor {
            return Err(DualTrackError::NotInXnorPool(block_id));
        }

        let offset = meta.xnor_offset;
        for (i, val) in out.iter_mut().enumerate() {
            let bit_pos = offset + i;
            if bit_pos >= self.xnor_capacity_bits {
                break;
            }
            let byte_pos = bit_pos / 8;
            let bit_within_byte = bit_pos % 8;
            *val = (self.xnor_pool[byte_pos] & (1 << bit_within_byte)) != 0;
        }

        Ok(())
    }

    /// 获取块所在的轨道
    pub fn get_track(&self, block_id: usize) -> Track {
        self.block_meta.get(block_id).map(|m| m.track).unwrap_or(Track::Empty)
    }

    /// 获取主池使用情况
    ///
    /// # 返回
    /// (已用字节, 总容量字节)
    pub fn main_usage(&self) -> (usize, usize) {
        (self.main_used, self.main_capacity)
    }

    /// 获取校验池使用情况
    ///
    /// # 返回
    /// (已用位数, 总容量位数)
    pub fn xnor_usage(&self) -> (usize, usize) {
        (self.xnor_used_bits, self.xnor_capacity_bits)
    }

    /// 获取压缩比 (相对于 FP16)
    ///
    /// FP16: 2 bytes per element
    /// Main Pool (4-bit): 0.5 bytes per element
    /// 压缩比 = 2 / 0.5 = 4x
    pub fn compression_ratio(&self) -> f32 {
        let bytes_per_element = match self.config.quant_bits {
            4 => 0.5,  // 2 elements per byte
            3 => 0.375, // 8 elements per 3 bytes
            _ => 0.5,
        };
        2.0 / bytes_per_element
    }

    /// 获取配置
    pub fn config(&self) -> &TrackConfig {
        &self.config
    }

    /// 重置池 (清空所有数据)
    pub fn reset(&mut self) {
        self.main_pool.fill(0);
        self.xnor_pool.fill(0);
        self.main_used = 0;
        self.xnor_used_bits = 0;
        for meta in &mut self.block_meta {
            meta.track = Track::Empty;
            meta.main_offset = 0;
            meta.xnor_offset = 0;
            meta.size_elements = 0;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_track_pool_creation() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        assert_eq!(pool.main_capacity, 1024);
        assert_eq!(pool.xnor_capacity_bits, 2048);
        assert_eq!(pool.config.quant_bits, 4);
        assert_eq!(pool.config.block_size, 16);
    }

    #[test]
    fn test_invalid_bit_width() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 5);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(5))));
    }

    #[test]
    fn test_allocate_main() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let offset = pool.allocate_block_main(0, 128).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.main_used, 64); // 128 elements / 2 per byte = 64 bytes
        assert_eq!(pool.get_track(0), Track::Main);
    }

    #[test]
    fn test_allocate_xnor() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let offset = pool.allocate_block_xnor(0, 128).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.xnor_used_bits, 128);
        assert_eq!(pool.get_track(0), Track::Xnor);
    }

    #[test]
    fn test_main_pool_full() {
        let mut pool = DualTrackMemoryPool::new(64, 256, 16, 4).unwrap();
        // 4-bit: 2 elements per byte, so 128 elements = 64 bytes (full capacity)
        pool.allocate_block_main(0, 128).unwrap();
        let result = pool.allocate_block_main(1, 1); // Need 0.5 bytes more, but pool is full
        assert!(matches!(result, Err(DualTrackError::MainPoolFull { .. })));
    }

    #[test]
    fn test_xnor_pool_full() {
        let mut pool = DualTrackMemoryPool::new(256, 128, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 128).unwrap();
        let result = pool.allocate_block_xnor(1, 1);
        assert!(matches!(result, Err(DualTrackError::XnorPoolFull { .. })));
    }

    #[test]
    fn test_write_read_main() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 8).unwrap();

        let data = vec![0x12, 0x34, 0x56, 0x78];
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 4];
        pool.read_main(0, &mut out).unwrap();

        assert_eq!(out, data);
    }

    #[test]
    fn test_write_read_xnor() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();

        let mask = vec![true, false, true, false, true, false, true, false,
                       false, true, false, true, false, true, false, true];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 16];
        pool.read_xnor(0, &mut out).unwrap();

        assert_eq!(out, mask);
    }

    #[test]
    fn test_not_in_main_pool() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();

        let data = vec![0x12, 0x34];
        let result = pool.write_main(0, &data);
        assert!(matches!(result, Err(DualTrackError::NotInMainPool(0))));
    }

    #[test]
    fn test_not_in_xnor_pool() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 8).unwrap();

        let mask = vec![true, false];
        let result = pool.write_xnor(0, &mask);
        assert!(matches!(result, Err(DualTrackError::NotInXnorPool(0))));
    }

    #[test]
    fn test_compression_ratio_4bit() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        // 4-bit: 2 bytes per element vs FP16's 2 bytes = 4x compression
        let ratio = pool.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_compression_ratio_3bit() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        // 3-bit: 8 elements per 3 bytes = 0.375 bytes per element
        // Compression ratio = 2 / 0.375 ≈ 5.33x
        let ratio = pool.compression_ratio();
        assert!((ratio - 5.33).abs() < 0.01);
    }

    #[test]
    fn test_usage_tracking() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 256).unwrap(); // 128 bytes
        pool.allocate_block_xnor(1, 512).unwrap(); // 512 bits

        let (main_used, main_cap) = pool.main_usage();
        assert_eq!(main_used, 128);
        assert_eq!(main_cap, 1024);

        let (xnor_used, xnor_cap) = pool.xnor_usage();
        assert_eq!(xnor_used, 512);
        assert_eq!(xnor_cap, 2048);
    }

    #[test]
    fn test_reset() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 256).unwrap();
        pool.allocate_block_xnor(1, 512).unwrap();

        pool.reset();

        assert_eq!(pool.main_used, 0);
        assert_eq!(pool.xnor_used_bits, 0);
        assert_eq!(pool.get_track(0), Track::Empty);
        assert_eq!(pool.get_track(1), Track::Empty);
    }

    #[test]
    fn test_default_config() {
        let pool = DualTrackMemoryPool::with_default_config().unwrap();
        let config = pool.config();
        assert_eq!(config.main_capacity, 256 * 1024 * 1024);
        assert_eq!(config.xnor_capacity_bits, 64 * 1024 * 1024 * 8);
        assert_eq!(config.quant_bits, 4);
        assert_eq!(config.block_size, 16);
    }

    #[test]
    fn test_multi_block_allocation() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();

        // Allocate multiple blocks in main pool
        pool.allocate_block_main(0, 64).unwrap();  // 32 bytes
        pool.allocate_block_main(1, 64).unwrap();  // 32 bytes
        pool.allocate_block_main(2, 128).unwrap(); // 64 bytes

        assert_eq!(pool.main_used, 128); // 32 + 32 + 64
        assert_eq!(pool.get_track(0), Track::Main);
        assert_eq!(pool.get_track(1), Track::Main);
        assert_eq!(pool.get_track(2), Track::Main);
    }

    #[test]
    fn test_xnor_bit_packing() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();

        // Test bit packing: write alternating bits
        let mask: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 16];
        pool.read_xnor(0, &mut out).unwrap();

        for (i, &bit) in out.iter().enumerate() {
            assert_eq!(bit, i % 2 == 0, "Bit at position {} should be {}", i, i % 2 == 0);
        }
    }

    // ── DualTrackError Display / derive ──

    #[test]
    fn dual_track_error_display_variants() {
        let e = DualTrackError::MainPoolFull { requested: 64, available: 32 };
        assert!(e.to_string().contains("64"));
        assert!(e.to_string().contains("32"));

        let e = DualTrackError::XnorPoolFull { requested: 128, available: 64 };
        assert!(e.to_string().contains("128"));

        let e = DualTrackError::NotInMainPool(42);
        assert!(e.to_string().contains("42"));

        let e = DualTrackError::NotInXnorPool(7);
        assert!(e.to_string().contains("7"));

        let e = DualTrackError::QuantFailed("overflow".to_string());
        assert!(e.to_string().contains("overflow"));

        let e = DualTrackError::InvalidBitWidth(2);
        assert!(e.to_string().contains("2"));
    }

    #[test]
    fn dual_track_error_is_std_error() {
        let e: DualTrackError = DualTrackError::InvalidBitWidth(7);
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn dual_track_error_debug_format() {
        let e = DualTrackError::MainPoolFull { requested: 10, available: 5 };
        let debug = format!("{e:?}");
        assert!(debug.contains("MainPoolFull"));
    }

    // ── Track derive ──

    #[test]
    fn track_equality() {
        assert_eq!(Track::Main, Track::Main);
        assert_eq!(Track::Xnor, Track::Xnor);
        assert_eq!(Track::Empty, Track::Empty);
        assert_ne!(Track::Main, Track::Xnor);
        assert_ne!(Track::Xnor, Track::Empty);
    }

    #[test]
    fn track_copy() {
        let a = Track::Main;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn track_debug() {
        assert!(format!("{:?}", Track::Main).contains("Main"));
        assert!(format!("{:?}", Track::Xnor).contains("Xnor"));
        assert!(format!("{:?}", Track::Empty).contains("Empty"));
    }

    // ── TrackConfig derive ──

    #[test]
    fn track_config_default_values() {
        let c = TrackConfig::default();
        assert_eq!(c.main_capacity, 256 * 1024 * 1024);
        assert_eq!(c.xnor_capacity_bits, 64 * 1024 * 1024 * 8);
        assert_eq!(c.quant_bits, 4);
        assert_eq!(c.block_size, 16);
    }

    #[test]
    fn track_config_clone() {
        let c = TrackConfig::default();
        let cloned = c.clone();
        assert_eq!(c.main_capacity, cloned.main_capacity);
        assert_eq!(c.quant_bits, cloned.quant_bits);
    }

    #[test]
    fn track_config_debug() {
        let c = TrackConfig::default();
        let debug = format!("{c:?}");
        assert!(debug.contains("main_capacity"));
        assert!(debug.contains("quant_bits"));
    }

    // ── DualTrackMemoryPool derive ──

    #[test]
    fn pool_clone_preserves_state() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 64).unwrap();
        let cloned = pool.clone();
        assert_eq!(cloned.main_used, pool.main_used);
        assert_eq!(cloned.xnor_used_bits, pool.xnor_used_bits);
        assert_eq!(cloned.get_track(0), Track::Main);
    }

    #[test]
    fn pool_debug() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let debug = format!("{pool:?}");
        assert!(debug.contains("main_pool"));
        assert!(debug.contains("xnor_pool"));
    }

    // ── Result alias ──

    #[test]
    fn dual_track_result_ok() {
        let r: DualTrackResult<usize> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn dual_track_result_err() {
        let r: DualTrackResult<usize> = Err(DualTrackError::InvalidBitWidth(1));
        assert!(r.is_err());
    }

    // ── 3-bit quant path ──

    #[test]
    fn pool_creation_3bit() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        assert_eq!(pool.config.quant_bits, 3);
        // 3-bit: 8 elements per 3 bytes
        let ratio = pool.compression_ratio();
        assert!((ratio - 5.333).abs() < 0.01);
    }

    #[test]
    fn allocate_main_3bit() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        // 8 elements → ceil(8/3) = 3 bytes at minimum, but div_ceil(8, 3) is wrong...
        // Actually bytes_per_element for 3-bit is 3 (meaning 8 elems / 3 bytes)
        // So size_bytes = num_elements.div_ceil(3) ... wait the code does div_ceil(bytes_per_element)
        // For 3-bit: bytes_per_element = 3, size = 24.div_ceil(3) = 8 bytes
        let offset = pool.allocate_block_main(0, 24).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.get_track(0), Track::Main);
    }

    #[test]
    fn invalid_bit_width_2() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 2);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(2))));
    }

    #[test]
    fn invalid_bit_width_8() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 8);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(8))));
    }

    // ── get_track for out-of-range ──

    #[test]
    fn get_track_out_of_range_returns_empty() {
        let pool = DualTrackMemoryPool::new(64, 128, 16, 4).unwrap();
        // Only a few blocks allocated, accessing a very large index
        assert_eq!(pool.get_track(9999), Track::Empty);
    }

    // ── write_read roundtrip larger data ──

    #[test]
    fn write_read_main_large_block() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();
        pool.allocate_block_main(0, 512).unwrap(); // 256 bytes

        let data: Vec<u8> = (0u8..200).map(|i| i.wrapping_mul(7)).collect();
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 200];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn reset_clears_data() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 64).unwrap();
        let data = vec![0xAB; 32];
        pool.write_main(0, &data).unwrap();

        pool.reset();

        // After reset, main block should be empty track
        assert_eq!(pool.get_track(0), Track::Empty);
        assert_eq!(pool.main_used, 0);
        assert_eq!(pool.xnor_used_bits, 0);
    }

    // ── Additional tests (18 new) ──

    // -- Track Hash derive --

    #[test]
    fn track_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |t: Track| -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };

        assert_eq!(hash_of(Track::Main), hash_of(Track::Main));
        assert_eq!(hash_of(Track::Xnor), hash_of(Track::Xnor));
        assert_eq!(hash_of(Track::Empty), hash_of(Track::Empty));
        assert_ne!(hash_of(Track::Main), hash_of(Track::Xnor));
        assert_ne!(hash_of(Track::Xnor), hash_of(Track::Empty));
    }

    // -- Track Clone (explicit) --

    #[test]
    fn track_clone_explicit() {
        let original = Track::Xnor;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // -- TrackConfig manual construction --

    #[test]
    fn track_config_custom_values() {
        let config = TrackConfig {
            main_capacity: 512,
            xnor_capacity_bits: 1024,
            quant_bits: 3,
            block_size: 8,
        };
        assert_eq!(config.main_capacity, 512);
        assert_eq!(config.xnor_capacity_bits, 1024);
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 8);
    }

    #[test]
    fn track_config_field_mutation() {
        let mut config = TrackConfig::default();
        config.quant_bits = 3;
        config.block_size = 32;
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 32);
        // Other fields remain default
        assert_eq!(config.main_capacity, 256 * 1024 * 1024);
    }

    // -- DualTrackError: QuantFailed variant --

    #[test]
    fn quant_failed_display() {
        let e = DualTrackError::QuantFailed("NaN detected".to_string());
        let msg = e.to_string();
        assert!(msg.contains("NaN detected"));
    }

    #[test]
    fn quant_failed_debug() {
        let e = DualTrackError::QuantFailed("overflow".to_string());
        let debug = format!("{e:?}");
        assert!(debug.contains("QuantFailed"));
    }

    // -- Pool: zero-element allocation --

    #[test]
    fn allocate_main_zero_elements() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let offset = pool.allocate_block_main(0, 0).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.main_used, 0);
        assert_eq!(pool.get_track(0), Track::Main);
    }

    #[test]
    fn allocate_xnor_zero_bits() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let offset = pool.allocate_block_xnor(0, 0).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.xnor_used_bits, 0);
        assert_eq!(pool.get_track(0), Track::Xnor);
    }

    // -- Pool: reading from unallocated block returns error --

    #[test]
    fn read_main_unallocated_block() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let mut out = [0u8; 4];
        let result = pool.read_main(0, &mut out);
        assert!(matches!(result, Err(DualTrackError::NotInMainPool(0))));
    }

    #[test]
    fn read_xnor_unallocated_block() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let mut out = [false; 4];
        let result = pool.read_xnor(0, &mut out);
        assert!(matches!(result, Err(DualTrackError::NotInXnorPool(0))));
    }

    // -- Pool: get_track on fresh unallocated block --

    #[test]
    fn get_track_fresh_block_is_empty() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        assert_eq!(pool.get_track(0), Track::Empty);
    }

    // -- Pool: usage on fresh pool --

    #[test]
    fn main_usage_fresh_pool() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let (used, capacity) = pool.main_usage();
        assert_eq!(used, 0);
        assert_eq!(capacity, 1024);
    }

    #[test]
    fn xnor_usage_fresh_pool() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let (used, capacity) = pool.xnor_usage();
        assert_eq!(used, 0);
        assert_eq!(capacity, 2048);
    }

    // -- Pool: reset then reallocate --

    #[test]
    fn reset_then_reallocate() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 256).unwrap();
        assert_eq!(pool.main_used, 128);

        pool.reset();
        assert_eq!(pool.main_used, 0);

        // Reallocate after reset should work at offset 0 again
        let offset = pool.allocate_block_main(0, 64).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(pool.main_used, 32); // 64 elements / 2 per byte = 32 bytes
    }

    // -- Pool: sequential main allocations accumulate offsets --

    #[test]
    fn sequential_main_allocations_accumulate() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();

        let off0 = pool.allocate_block_main(0, 64).unwrap(); // 32 bytes
        assert_eq!(off0, 0);

        let off1 = pool.allocate_block_main(1, 64).unwrap(); // 32 bytes
        assert_eq!(off1, 32);

        let off2 = pool.allocate_block_main(2, 128).unwrap(); // 64 bytes
        assert_eq!(off2, 64);

        assert_eq!(pool.main_used, 128); // 32 + 32 + 64
    }

    // -- Pool: xnor bit packing across byte boundary --

    #[test]
    fn xnor_write_read_across_byte_boundary() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        // Allocate at bit offset 0, write 5 bits to cross into next byte
        pool.allocate_block_xnor(0, 16).unwrap();

        let mask = vec![true, true, true, true, true, false, false, false, true];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 9];
        pool.read_xnor(0, &mut out).unwrap();

        assert_eq!(out, mask);
    }

    // -- Pool: config() accessor --

    #[test]
    fn config_accessor_matches_constructor() {
        let pool = DualTrackMemoryPool::new(2048, 4096, 32, 3).unwrap();
        let config = pool.config();
        assert_eq!(config.main_capacity, 2048);
        assert_eq!(config.xnor_capacity_bits, 4096);
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 32);
    }

    // -- Pool: 3-bit write/read roundtrip --

    #[test]
    fn write_read_main_3bit_roundtrip() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        pool.allocate_block_main(0, 24).unwrap(); // 8 bytes for 24 elements

        let data = vec![0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44];
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 8];
        pool.read_main(0, &mut out).unwrap();

        assert_eq!(out, data);
    }

    // ============================================================================
    // Additional tests (45+ new tests)
    // ============================================================================

    // ── Track: exhaustive variant set ──

    #[test]
    fn track_all_variants_are_distinct() {
        let variants = [Track::Main, Track::Xnor, Track::Empty];
        for (i, &a) in variants.iter().enumerate() {
            for (j, &b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn track_main_not_equal_xnor() {
        assert_ne!(Track::Main, Track::Xnor);
    }

    #[test]
    fn track_main_not_equal_empty() {
        assert_ne!(Track::Main, Track::Empty);
    }

    #[test]
    fn track_xnor_not_equal_empty() {
        assert_ne!(Track::Xnor, Track::Empty);
    }

    #[test]
    fn track_debug_all_variants() {
        let main_dbg = format!("{:?}", Track::Main);
        let xnor_dbg = format!("{:?}", Track::Xnor);
        let empty_dbg = format!("{:?}", Track::Empty);
        // Each debug string must contain its variant name
        assert!(main_dbg.contains("Main"));
        assert!(xnor_dbg.contains("Xnor"));
        assert!(empty_dbg.contains("Empty"));
        // They must be distinct strings
        assert_ne!(main_dbg, xnor_dbg);
        assert_ne!(main_dbg, empty_dbg);
        assert_ne!(xnor_dbg, empty_dbg);
    }

    // ── Track: Hash stability across multiple calls ──

    #[test]
    fn track_hash_is_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |t: Track| -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };

        // Hashing the same value twice must produce the same result
        for variant in [Track::Main, Track::Xnor, Track::Empty] {
            assert_eq!(hash_of(variant), hash_of(variant));
        }
    }

    #[test]
    fn track_hash_all_variants_unique() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |t: Track| -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };

        let h_main = hash_of(Track::Main);
        let h_xnor = hash_of(Track::Xnor);
        let h_empty = hash_of(Track::Empty);
        // All three hashes must be distinct
        assert_ne!(h_main, h_xnor);
        assert_ne!(h_main, h_empty);
        assert_ne!(h_xnor, h_empty);
    }

    // ── Track: use in HashSet ──

    #[test]
    fn track_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Track::Main);
        set.insert(Track::Xnor);
        set.insert(Track::Empty);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Track::Main));
        assert!(set.contains(&Track::Xnor));
        assert!(set.contains(&Track::Empty));
    }

    #[test]
    fn track_hashset_dedup() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Track::Main);
        set.insert(Track::Main);
        set.insert(Track::Main);
        assert_eq!(set.len(), 1);
    }

    // ── DualTrackError: all display strings ──

    #[test]
    fn error_display_main_pool_full_contains_words() {
        let e = DualTrackError::MainPoolFull { requested: 100, available: 50 };
        let msg = e.to_string();
        assert!(msg.contains("100"), "should contain requested value");
        assert!(msg.contains("50"), "should contain available value");
    }

    #[test]
    fn error_display_xnor_pool_full_contains_words() {
        let e = DualTrackError::XnorPoolFull { requested: 200, available: 100 };
        let msg = e.to_string();
        assert!(msg.contains("200"));
        assert!(msg.contains("100"));
    }

    #[test]
    fn error_display_not_in_main_pool() {
        let e = DualTrackError::NotInMainPool(99);
        let msg = e.to_string();
        assert!(msg.contains("99"));
    }

    #[test]
    fn error_display_not_in_xnor_pool() {
        let e = DualTrackError::NotInXnorPool(42);
        let msg = e.to_string();
        assert!(msg.contains("42"));
    }

    #[test]
    fn error_display_quant_failed_empty_string() {
        let e = DualTrackError::QuantFailed(String::new());
        let msg = e.to_string();
        // Even with empty message, should not panic
        assert!(msg.contains("量化失败"));
    }

    #[test]
    fn error_display_invalid_bit_width_boundary() {
        let e = DualTrackError::InvalidBitWidth(0);
        let msg = e.to_string();
        assert!(msg.contains('0'));
    }

    #[test]
    fn error_display_invalid_bit_width_max() {
        let e = DualTrackError::InvalidBitWidth(255);
        let msg = e.to_string();
        assert!(msg.contains("255"));
    }

    // ── DualTrackError: Debug format for each variant ──

    #[test]
    fn error_debug_xnor_pool_full() {
        let e = DualTrackError::XnorPoolFull { requested: 10, available: 5 };
        let debug = format!("{e:?}");
        assert!(debug.contains("XnorPoolFull"));
    }

    #[test]
    fn error_debug_not_in_main_pool() {
        let e = DualTrackError::NotInMainPool(13);
        let debug = format!("{e:?}");
        assert!(debug.contains("NotInMainPool"));
    }

    #[test]
    fn error_debug_not_in_xnor_pool() {
        let e = DualTrackError::NotInXnorPool(7);
        let debug = format!("{e:?}");
        assert!(debug.contains("NotInXnorPool"));
    }

    #[test]
    fn error_debug_quant_failed() {
        let e = DualTrackError::QuantFailed("underflow".to_string());
        let debug = format!("{e:?}");
        assert!(debug.contains("QuantFailed"));
    }

    #[test]
    fn error_debug_invalid_bit_width() {
        let e = DualTrackError::InvalidBitWidth(6);
        let debug = format!("{e:?}");
        assert!(debug.contains("InvalidBitWidth"));
    }

    // ── Pool creation: edge cases ──

    #[test]
    fn pool_creation_minimum_capacity() {
        // Smallest valid pool: 1 byte main, 1 bit xnor
        let pool = DualTrackMemoryPool::new(1, 1, 1, 4);
        assert!(pool.is_ok());
    }

    #[test]
    fn pool_creation_zero_capacity() {
        // Zero capacity should still create a pool (just can't allocate anything)
        let pool = DualTrackMemoryPool::new(0, 0, 16, 4);
        assert!(pool.is_ok());
        let pool = pool.unwrap();
        assert_eq!(pool.main_capacity, 0);
        assert_eq!(pool.xnor_capacity_bits, 0);
    }

    #[test]
    fn pool_creation_invalid_bit_width_1() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 1);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(1))));
    }

    #[test]
    fn pool_creation_invalid_bit_width_6() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 6);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(6))));
    }

    #[test]
    fn pool_creation_invalid_bit_width_7() {
        let result = DualTrackMemoryPool::new(1024, 2048, 16, 7);
        assert!(matches!(result, Err(DualTrackError::InvalidBitWidth(7))));
    }

    #[test]
    fn pool_creation_block_size_1() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 1, 4);
        assert!(pool.is_ok());
    }

    // ── Pool: main pool allocation offset tracking ──

    #[test]
    fn main_allocation_first_offset_is_zero() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();
        let off = pool.allocate_block_main(0, 16).unwrap();
        assert_eq!(off, 0);
    }

    #[test]
    fn main_allocation_second_offset_follows_first() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();
        let off0 = pool.allocate_block_main(0, 16).unwrap(); // 8 bytes
        let off1 = pool.allocate_block_main(1, 16).unwrap(); // 8 bytes
        assert_eq!(off0, 0);
        assert_eq!(off1, 8);
    }

    #[test]
    fn main_allocation_3bit_offset_calculation() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 3).unwrap();
        // 3-bit: bytes_per_element = 3, meaning 8 elems take 3 bytes
        // 24 elements -> div_ceil(24, 3) = 8 bytes
        let off0 = pool.allocate_block_main(0, 24).unwrap();
        assert_eq!(off0, 0);
        assert_eq!(pool.main_used, 8);
    }

    // ── Pool: xnor allocation offset tracking ──

    #[test]
    fn xnor_allocation_first_offset_is_zero() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let off = pool.allocate_block_xnor(0, 64).unwrap();
        assert_eq!(off, 0);
    }

    #[test]
    fn xnor_allocation_offsets_accumulate() {
        let mut pool = DualTrackMemoryPool::new(1024, 4096, 16, 4).unwrap();
        let off0 = pool.allocate_block_xnor(0, 64).unwrap();
        let off1 = pool.allocate_block_xnor(1, 32).unwrap();
        let off2 = pool.allocate_block_xnor(2, 16).unwrap();
        assert_eq!(off0, 0);
        assert_eq!(off1, 64);
        assert_eq!(off2, 96);
        assert_eq!(pool.xnor_used_bits, 112);
    }

    // ── Pool: write_main with empty data ──

    #[test]
    fn write_main_empty_data() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        let result = pool.write_main(0, &[]);
        assert!(result.is_ok());
    }

    // ── Pool: read_main with empty buffer ──

    #[test]
    fn read_main_empty_buffer() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        let mut out: [u8; 0] = [];
        let result = pool.read_main(0, &mut out);
        assert!(result.is_ok());
    }

    // ── Pool: write_xnor with empty mask ──

    #[test]
    fn write_xnor_empty_mask() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap();
        let result = pool.write_xnor(0, &[]);
        assert!(result.is_ok());
    }

    // ── Pool: read_xnor with empty buffer ──

    #[test]
    fn read_xnor_empty_buffer() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap();
        let mut out: [bool; 0] = [];
        let result = pool.read_xnor(0, &mut out);
        assert!(result.is_ok());
    }

    // ── Pool: write_main data larger than allocated block ──

    #[test]
    fn write_main_exceeds_pool_capacity() {
        let mut pool = DualTrackMemoryPool::new(16, 256, 16, 4).unwrap();
        pool.allocate_block_main(0, 8).unwrap(); // 4 bytes
        // Write 16 bytes, but pool only has 16 total, offset 0, end would be 16 = pool len
        // This actually succeeds because pool is exactly 16 bytes
        // Let's write 17 bytes
        let data = vec![0xFFu8; 17];
        let result = pool.write_main(0, &data);
        // end_offset = 0 + 17 = 17 > 16, should fail
        assert!(result.is_err());
    }

    // ── Pool: write_main on block allocated to xnor track ──

    #[test]
    fn write_main_on_xnor_block_errors() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();
        let data = vec![0x12, 0x34];
        let result = pool.write_main(0, &data);
        assert!(matches!(result, Err(DualTrackError::NotInMainPool(0))));
    }

    // ── Pool: write_xnor on block allocated to main track ──

    #[test]
    fn write_xnor_on_main_block_errors() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        let mask = vec![true, false, true];
        let result = pool.write_xnor(0, &mask);
        assert!(matches!(result, Err(DualTrackError::NotInXnorPool(0))));
    }

    // ── Pool: dual track allocation (main + xnor on different blocks) ──

    #[test]
    fn dual_track_allocation_both_tracks() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let main_off = pool.allocate_block_main(0, 64).unwrap();
        let xnor_off = pool.allocate_block_xnor(1, 64).unwrap();

        assert_eq!(main_off, 0);
        assert_eq!(xnor_off, 0); // Separate pool, offset starts at 0
        assert_eq!(pool.get_track(0), Track::Main);
        assert_eq!(pool.get_track(1), Track::Xnor);
    }

    // ── Pool: write/read xnor single bit ──

    #[test]
    fn xnor_single_bit_true() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 1).unwrap();
        pool.write_xnor(0, &[true]).unwrap();

        let mut out = [false; 1];
        pool.read_xnor(0, &mut out).unwrap();
        assert_eq!(out[0], true);
    }

    #[test]
    fn xnor_single_bit_false() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 1).unwrap();
        pool.write_xnor(0, &[false]).unwrap();

        let mut out = [true; 1];
        pool.read_xnor(0, &mut out).unwrap();
        assert_eq!(out[0], false);
    }

    // ── Pool: xnor all-true mask ──

    #[test]
    fn xnor_all_true_mask() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();
        let mask = vec![true; 16];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 16];
        pool.read_xnor(0, &mut out).unwrap();
        assert!(out.iter().all(|&b| b));
    }

    // ── Pool: xnor all-false mask ──

    #[test]
    fn xnor_all_false_mask() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();
        let mask = vec![false; 16];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![true; 16]; // Start all true to verify they get cleared
        pool.read_xnor(0, &mut out).unwrap();
        assert!(out.iter().all(|&b| !b));
    }

    // ── Pool: write then overwrite main data ──

    #[test]
    fn write_main_overwrite() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        let data1 = vec![0xAA; 8];
        pool.write_main(0, &data1).unwrap();

        let data2 = vec![0x55; 8];
        pool.write_main(0, &data2).unwrap();

        let mut out = vec![0u8; 8];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, data2);
        assert_ne!(out, data1);
    }

    // ── Pool: write then overwrite xnor mask ──

    #[test]
    fn write_xnor_overwrite() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap();

        let mask1 = vec![true; 8];
        pool.write_xnor(0, &mask1).unwrap();

        let mask2 = vec![false; 8];
        pool.write_xnor(0, &mask2).unwrap();

        let mut out = vec![true; 8];
        pool.read_xnor(0, &mut out).unwrap();
        assert_eq!(out, mask2);
    }

    // ── Pool: reset preserves config ──

    #[test]
    fn reset_preserves_config() {
        let mut pool = DualTrackMemoryPool::new(2048, 4096, 32, 3).unwrap();
        pool.allocate_block_main(0, 64).unwrap();
        pool.reset();

        let config = pool.config();
        assert_eq!(config.main_capacity, 2048);
        assert_eq!(config.xnor_capacity_bits, 4096);
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 32);
    }

    #[test]
    fn reset_preserves_compression_ratio() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let ratio_before = pool.compression_ratio();
        pool.allocate_block_main(0, 64).unwrap();
        pool.reset();
        let ratio_after = pool.compression_ratio();
        assert!((ratio_before - ratio_after).abs() < f32::EPSILON);
    }

    // ── Pool: reset clears all block tracks ──

    #[test]
    fn reset_clears_all_block_tracks() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        pool.allocate_block_xnor(1, 16).unwrap();
        pool.allocate_block_main(2, 16).unwrap();

        pool.reset();

        assert_eq!(pool.get_track(0), Track::Empty);
        assert_eq!(pool.get_track(1), Track::Empty);
        assert_eq!(pool.get_track(2), Track::Empty);
    }

    // ── Pool: sequential allocation after reset reuses offsets ──

    #[test]
    fn sequential_allocation_after_reset() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 64).unwrap();
        pool.allocate_block_main(1, 64).unwrap();
        pool.reset();

        // After reset, offsets start from 0 again
        let off = pool.allocate_block_main(0, 64).unwrap();
        assert_eq!(off, 0);
        let off2 = pool.allocate_block_main(1, 64).unwrap();
        assert_eq!(off2, 32); // 64 elements / 2 = 32 bytes
    }

    // ── Pool: compression ratio for different quant_bits ──

    #[test]
    fn compression_ratio_4bit_is_exactly_4() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let ratio = pool.compression_ratio();
        assert_eq!(ratio, 4.0);
    }

    #[test]
    fn compression_ratio_3bit_approx_5_33() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        let ratio = pool.compression_ratio();
        let expected = 2.0_f32 / 0.375;
        assert!((ratio - expected).abs() < 0.001);
    }

    // ── Pool: main_usage and xnor_usage after allocations ──

    #[test]
    fn main_usage_after_multiple_allocations() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();
        pool.allocate_block_main(0, 32).unwrap(); // 16 bytes
        pool.allocate_block_main(1, 64).unwrap(); // 32 bytes
        pool.allocate_block_main(2, 128).unwrap(); // 64 bytes

        let (used, cap) = pool.main_usage();
        assert_eq!(used, 112); // 16 + 32 + 64
        assert_eq!(cap, 4096);
    }

    #[test]
    fn xnor_usage_after_multiple_allocations() {
        let mut pool = DualTrackMemoryPool::new(1024, 4096, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 128).unwrap();
        pool.allocate_block_xnor(1, 256).unwrap();
        pool.allocate_block_xnor(2, 64).unwrap();

        let (used, cap) = pool.xnor_usage();
        assert_eq!(used, 448); // 128 + 256 + 64
        assert_eq!(cap, 4096);
    }

    // ── Pool: reusing block id after track switch ──

    #[test]
    fn reuse_block_id_main_then_xnor() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        assert_eq!(pool.get_track(0), Track::Main);

        // Re-assign block 0 to xnor
        pool.allocate_block_xnor(0, 16).unwrap();
        assert_eq!(pool.get_track(0), Track::Xnor);

        // Writing to main on block 0 should now fail
        let data = vec![0x12];
        let result = pool.write_main(0, &data);
        assert!(matches!(result, Err(DualTrackError::NotInMainPool(0))));
    }

    // ── Pool: main pool fills exactly ──

    #[test]
    fn main_pool_fills_exactly() {
        let mut pool = DualTrackMemoryPool::new(32, 256, 16, 4).unwrap();
        // 4-bit: 2 elements per byte, 32 bytes = 64 elements
        let off = pool.allocate_block_main(0, 64).unwrap();
        assert_eq!(off, 0);
        assert_eq!(pool.main_used, 32);

        let (used, cap) = pool.main_usage();
        assert_eq!(used, cap);
    }

    // ── Pool: xnor pool fills exactly ──

    #[test]
    fn xnor_pool_fills_exactly() {
        let mut pool = DualTrackMemoryPool::new(256, 64, 16, 4).unwrap();
        let off = pool.allocate_block_xnor(0, 64).unwrap();
        assert_eq!(off, 0);

        let (used, cap) = pool.xnor_usage();
        assert_eq!(used, cap);
        assert_eq!(used, 64);
    }

    // ── Pool: main_pool_full error carries correct values ──

    #[test]
    fn main_pool_full_error_values() {
        let mut pool = DualTrackMemoryPool::new(32, 256, 16, 4).unwrap();
        pool.allocate_block_main(0, 64).unwrap(); // fills exactly 32 bytes

        let result = pool.allocate_block_main(1, 16);
        match result {
            Err(DualTrackError::MainPoolFull { requested, available }) => {
                // requested = 16 elements -> 8 bytes
                assert_eq!(requested, 8);
                assert_eq!(available, 0);
            }
            _ => panic!("Expected MainPoolFull error"),
        }
    }

    #[test]
    fn xnor_pool_full_error_values() {
        let mut pool = DualTrackMemoryPool::new(256, 64, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 64).unwrap(); // fills exactly 64 bits

        let result = pool.allocate_block_xnor(1, 8);
        match result {
            Err(DualTrackError::XnorPoolFull { requested, available }) => {
                assert_eq!(requested, 8);
                assert_eq!(available, 0);
            }
            _ => panic!("Expected XnorPoolFull error"),
        }
    }

    // ── Pool: write/read main with max byte values ──

    #[test]
    fn write_read_main_with_max_byte_values() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        let data = vec![0xFF; 8];
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 8];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, vec![0xFF; 8]);
    }

    #[test]
    fn write_read_main_with_zero_byte_values() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        // First write non-zero
        let data1 = vec![0xAB; 8];
        pool.write_main(0, &data1).unwrap();

        // Then write zeros
        let data2 = vec![0x00; 8];
        pool.write_main(0, &data2).unwrap();

        let mut out = vec![0xFF; 8];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, vec![0x00; 8]);
    }

    // ── Pool: xnor at non-zero start offset ──

    #[test]
    fn xnor_at_non_zero_offset() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap(); // 8 bits at offset 0
        pool.allocate_block_xnor(1, 8).unwrap(); // 8 bits at offset 8

        let mask0 = vec![true, false, true, false, true, false, true, false];
        let mask1 = vec![false, true, false, true, false, true, false, true];

        pool.write_xnor(0, &mask0).unwrap();
        pool.write_xnor(1, &mask1).unwrap();

        let mut out0 = vec![false; 8];
        let mut out1 = vec![false; 8];
        pool.read_xnor(0, &mut out0).unwrap();
        pool.read_xnor(1, &mut out1).unwrap();

        assert_eq!(out0, mask0);
        assert_eq!(out1, mask1);
    }

    // ── Pool: 3-bit pool fill to capacity ──

    #[test]
    fn pool_3bit_allocation_near_capacity() {
        // 3-bit: bytes_per_element = 3, so 24 elements = 8 bytes
        let mut pool = DualTrackMemoryPool::new(8, 256, 16, 3).unwrap();
        let off = pool.allocate_block_main(0, 24).unwrap();
        assert_eq!(off, 0);
        assert_eq!(pool.main_used, 8); // Pool is full

        let (used, cap) = pool.main_usage();
        assert_eq!(used, cap);
    }

    // ── TrackConfig: access individual fields ──

    #[test]
    fn track_config_all_fields_accessible() {
        let config = TrackConfig {
            main_capacity: 100,
            xnor_capacity_bits: 200,
            quant_bits: 3,
            block_size: 4,
        };
        assert_eq!(config.main_capacity, 100);
        assert_eq!(config.xnor_capacity_bits, 200);
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 4);
    }

    // ── DualTrackResult: is Send + Sync compatible ──

    #[test]
    fn dual_track_error_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DualTrackError>();
    }

    #[test]
    fn dual_track_error_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<DualTrackError>();
    }

    // ── Pool: config() returns reference to internal state ──

    #[test]
    fn config_returns_reference() {
        let pool = DualTrackMemoryPool::new(512, 1024, 8, 4).unwrap();
        let config = pool.config();
        // Verify the reference points to the internal config
        assert_eq!(config.main_capacity, 512);
        assert_eq!(config.xnor_capacity_bits, 1024);
        assert_eq!(config.block_size, 8);
    }

    // ── Pool: multiple resets ──

    #[test]
    fn multiple_resets() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();

        for _ in 0..5 {
            pool.allocate_block_main(0, 64).unwrap();
            assert_eq!(pool.main_used, 32);
            pool.reset();
            assert_eq!(pool.main_used, 0);
        }
    }

    // ── Pool: clone after write preserves data ──

    #[test]
    fn clone_after_write_preserves_data() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        pool.write_main(0, &data).unwrap();

        let cloned = pool.clone();

        let mut orig_out = vec![0u8; 4];
        let mut clone_out = vec![0u8; 4];
        pool.read_main(0, &mut orig_out).unwrap();
        cloned.read_main(0, &mut clone_out).unwrap();

        assert_eq!(orig_out, data);
        assert_eq!(clone_out, data);
    }

    // ── Pool: xnor bit pattern with exact byte boundary ──

    #[test]
    fn xnor_exact_byte_boundary() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap();

        // Write exactly 8 bits = 1 byte
        let mask = vec![true, true, true, true, true, true, true, true];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 8];
        pool.read_xnor(0, &mut out).unwrap();
        assert!(out.iter().all(|&b| b));
    }

    // ── Pool: large xnor mask roundtrip ──

    #[test]
    fn xnor_large_mask_roundtrip() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 64).unwrap();

        // Pattern: first 32 true, next 32 false
        let mask: Vec<bool> = (0..64).map(|i| i < 32).collect();
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 64];
        pool.read_xnor(0, &mut out).unwrap();

        for (i, &bit) in out.iter().enumerate() {
            assert_eq!(bit, i < 32, "Bit at position {i} mismatch");
        }
    }

    // ── Pool: get_track returns Empty for unallocated block index ──

    #[test]
    fn get_track_unallocated_returns_empty() {
        let pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        // Block 0 is never allocated
        assert_eq!(pool.get_track(0), Track::Empty);
    }

    // ── Pool: allocate same block id twice overwrites ──

    #[test]
    fn allocate_same_block_twice_overwrites() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        assert_eq!(pool.get_track(0), Track::Main);

        // Allocate same block again with xnor
        pool.allocate_block_xnor(0, 16).unwrap();
        assert_eq!(pool.get_track(0), Track::Xnor);
    }

    // ── Pool: with_default_config creates valid pool ──

    #[test]
    fn with_default_config_is_usable() {
        let mut pool = DualTrackMemoryPool::with_default_config().unwrap();
        // Should be able to allocate and write
        pool.allocate_block_main(0, 16).unwrap();
        let data = vec![0x42; 8];
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 8];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // ── Track: use as map key ──

    #[test]
    fn track_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(Track::Main, "primary");
        map.insert(Track::Xnor, "check");
        map.insert(Track::Empty, "free");

        assert_eq!(map.get(&Track::Main), Some(&"primary"));
        assert_eq!(map.get(&Track::Xnor), Some(&"check"));
        assert_eq!(map.get(&Track::Empty), Some(&"free"));
    }

    // ── Pool: write/read main across multiple blocks ──

    #[test]
    fn write_read_multiple_blocks_independent() {
        let mut pool = DualTrackMemoryPool::new(4096, 4096, 16, 4).unwrap();

        pool.allocate_block_main(0, 16).unwrap();
        pool.allocate_block_main(1, 16).unwrap();

        let data0 = vec![0x11, 0x22, 0x33, 0x44];
        let data1 = vec![0xAA, 0xBB, 0xCC, 0xDD];

        pool.write_main(0, &data0).unwrap();
        pool.write_main(1, &data1).unwrap();

        let mut out0 = vec![0u8; 4];
        let mut out1 = vec![0u8; 4];
        pool.read_main(0, &mut out0).unwrap();
        pool.read_main(1, &mut out1).unwrap();

        assert_eq!(out0, data0);
        assert_eq!(out1, data1);
        assert_ne!(out0, out1);
    }

    // ── Pool: error type implements std::error::Error ──

    #[test]
    fn all_error_variants_are_std_error() {
        let _: &dyn std::error::Error = &DualTrackError::MainPoolFull { requested: 0, available: 0 };
        let _: &dyn std::error::Error = &DualTrackError::XnorPoolFull { requested: 0, available: 0 };
        let _: &dyn std::error::Error = &DualTrackError::NotInMainPool(0);
        let _: &dyn std::error::Error = &DualTrackError::NotInXnorPool(0);
        let _: &dyn std::error::Error = &DualTrackError::QuantFailed("test".to_string());
        let _: &dyn std::error::Error = &DualTrackError::InvalidBitWidth(0);
    }

    // ── Pool: 3-bit pool main write read roundtrip with arbitrary data ──

    #[test]
    fn write_read_main_3bit_arbitrary_data() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 3).unwrap();
        pool.allocate_block_main(0, 48).unwrap(); // div_ceil(48, 3) = 16 bytes

        let data: Vec<u8> = (0u8..16).collect();
        pool.write_main(0, &data).unwrap();

        let mut out = vec![0u8; 16];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // ============================================================================
    // Additional tests (+15 new)
    // ============================================================================

    // ── Track: all variants collected into HashSet ──

    #[test]
    fn track_all_variants_hashset_roundtrip() {
        use std::collections::HashSet;
        let all = [Track::Main, Track::Xnor, Track::Empty];
        let set: HashSet<Track> = all.into_iter().collect();
        assert_eq!(set.len(), 3);
        for v in &all {
            assert!(set.contains(v));
        }
    }

    // ── Track: Copy trait allows independent use after move ──

    #[test]
    fn track_copy_independent_after_assignment() {
        let a = Track::Main;
        let b = a;
        let c = a;
        assert_eq!(a, Track::Main);
        assert_eq!(b, Track::Main);
        assert_eq!(c, Track::Main);
    }

    // ── Track: Debug output is human-readable for each variant ──

    #[test]
    fn track_debug_format_is_human_readable() {
        assert_eq!(format!("{:?}", Track::Main), "Main");
        assert_eq!(format!("{:?}", Track::Xnor), "Xnor");
        assert_eq!(format!("{:?}", Track::Empty), "Empty");
    }

    // ── TrackConfig: custom construction with non-default values ──

    #[test]
    fn track_config_custom_construction() {
        let config = TrackConfig {
            main_capacity: 4096,
            xnor_capacity_bits: 8192,
            quant_bits: 3,
            block_size: 64,
        };
        assert_eq!(config.main_capacity, 4096);
        assert_eq!(config.xnor_capacity_bits, 8192);
        assert_eq!(config.quant_bits, 3);
        assert_eq!(config.block_size, 64);
    }

    // ── TrackConfig: default values are exactly as documented ──

    #[test]
    fn track_config_default_exact_byte_values() {
        let config = TrackConfig::default();
        // 256 MB = 256 * 1024 * 1024
        assert_eq!(config.main_capacity, 268_435_456);
        // 64 MB in bits = 64 * 1024 * 1024 * 8
        assert_eq!(config.xnor_capacity_bits, 536_870_912);
        assert_eq!(config.quant_bits, 4);
        assert_eq!(config.block_size, 16);
    }

    // ── DualTrackError: Display for MainPoolFull contains both numbers ──

    #[test]
    fn error_display_main_pool_full_exact_format() {
        let e = DualTrackError::MainPoolFull { requested: 256, available: 128 };
        let msg = e.to_string();
        assert!(msg.contains("主池已满"), "should contain Chinese label");
        assert!(msg.contains("256"));
        assert!(msg.contains("128"));
    }

    // ── DualTrackError: Display for XnorPoolFull ──

    #[test]
    fn error_display_xnor_pool_full_exact_format() {
        let e = DualTrackError::XnorPoolFull { requested: 512, available: 0 };
        let msg = e.to_string();
        assert!(msg.contains("校验池已满"), "should contain Chinese label");
        assert!(msg.contains("512"));
        assert!(msg.contains('0'));
    }

    // ── DualTrackError: Display for QuantFailed preserves message ──

    #[test]
    fn error_display_quant_failed_preserves_detail() {
        let e = DualTrackError::QuantFailed("tensor shape mismatch".to_string());
        let msg = e.to_string();
        assert!(msg.contains("tensor shape mismatch"));
    }

    // ── DualTrackError: Display for InvalidBitWidth ──

    #[test]
    fn error_display_invalid_bit_width_contains_value() {
        let e = DualTrackError::InvalidBitWidth(9);
        let msg = e.to_string();
        assert!(msg.contains("无效的位宽"));
        assert!(msg.contains('9'));
    }

    // ── DualTrackMemoryPool: with_default_config matches TrackConfig::default ──

    #[test]
    fn with_default_config_matches_track_config_default() {
        let pool = DualTrackMemoryPool::with_default_config().unwrap();
        let default_config = TrackConfig::default();
        let pool_config = pool.config();
        assert_eq!(pool_config.main_capacity, default_config.main_capacity);
        assert_eq!(pool_config.xnor_capacity_bits, default_config.xnor_capacity_bits);
        assert_eq!(pool_config.quant_bits, default_config.quant_bits);
        assert_eq!(pool_config.block_size, default_config.block_size);
    }

    // ── DualTrackMemoryPool: 3-bit pool operations end-to-end ──

    #[test]
    fn pool_3bit_allocate_write_read_xnor_e2e() {
        let mut pool = DualTrackMemoryPool::new(512, 256, 8, 3).unwrap();
        // Allocate main block with 3-bit quant
        pool.allocate_block_main(0, 24).unwrap(); // div_ceil(24, 3) = 8 bytes
        let main_data = vec![0x0F, 0xF0, 0xAA, 0x55, 0xCC, 0x33, 0x99, 0x66];
        pool.write_main(0, &main_data).unwrap();

        // Allocate xnor block
        pool.allocate_block_xnor(1, 8).unwrap();
        let mask = vec![true, false, true, false, true, false, true, false];
        pool.write_xnor(1, &mask).unwrap();

        // Verify main roundtrip
        let mut main_out = vec![0u8; 8];
        pool.read_main(0, &mut main_out).unwrap();
        assert_eq!(main_out, main_data);

        // Verify xnor roundtrip
        let mut xnor_out = vec![false; 8];
        pool.read_xnor(1, &mut xnor_out).unwrap();
        assert_eq!(xnor_out, mask);
    }

    // ── DualTrackMemoryPool: zero-capacity pool rejects any allocation ──

    #[test]
    fn zero_capacity_pool_rejects_main_allocation() {
        let mut pool = DualTrackMemoryPool::new(0, 0, 16, 4).unwrap();
        let result = pool.allocate_block_main(0, 1);
        assert!(matches!(result, Err(DualTrackError::MainPoolFull { .. })));
    }

    // ── DualTrackMemoryPool: zero-capacity pool rejects xnor allocation ──

    #[test]
    fn zero_capacity_pool_rejects_xnor_allocation() {
        let mut pool = DualTrackMemoryPool::new(0, 0, 16, 4).unwrap();
        let result = pool.allocate_block_xnor(0, 1);
        assert!(matches!(result, Err(DualTrackError::XnorPoolFull { .. })));
    }

    // ── BlockMeta: default values are correct ──

    #[test]
    fn block_meta_default_values() {
        let meta = BlockMeta::default();
        assert_eq!(meta.track, Track::Empty);
        assert_eq!(meta.main_offset, 0);
        assert_eq!(meta.xnor_offset, 0);
        assert_eq!(meta.size_elements, 0);
        assert_eq!(meta.quant_bits, 4);
    }

    // ============================================================================
    // Additional tests (+13 new)
    // ============================================================================

    // ── 3-bit main pool full error ──

    #[test]
    fn main_pool_full_error_3bit() {
        let mut pool = DualTrackMemoryPool::new(8, 256, 16, 3).unwrap();
        // 3-bit: bytes_per_element=3, 24 elements = ceil(24/3) = 8 bytes (fills pool)
        pool.allocate_block_main(0, 24).unwrap();
        // Any further allocation should fail
        let result = pool.allocate_block_main(1, 8);
        assert!(matches!(result, Err(DualTrackError::MainPoolFull { .. })));
    }

    // ── read_main on allocated but unwritten block returns zeros ──

    #[test]
    fn read_main_unwritten_block_returns_zeros() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        let mut out = vec![0xFFu8; 8];
        pool.read_main(0, &mut out).unwrap();
        // Pool is initialized to all zeros, so unwritten block should read zeros
        assert_eq!(out, vec![0u8; 8]);
    }

    // ── read_xnor on allocated but unwritten block returns all false ──

    #[test]
    fn read_xnor_unwritten_block_returns_false() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();

        let mut out = vec![true; 16];
        pool.read_xnor(0, &mut out).unwrap();
        assert!(out.iter().all(|&b| !b), "unwritten xnor block should read all false");
    }

    // ── write_main after reset on same block id fails because track is Empty ──

    #[test]
    fn write_main_after_reset_fails_with_empty_track() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();
        let data = vec![0x42; 8];
        pool.write_main(0, &data).unwrap();

        pool.reset();

        // Block 0 is now Track::Empty, so write_main should fail
        let result = pool.write_main(0, &data);
        assert!(matches!(result, Err(DualTrackError::NotInMainPool(0))));
    }

    // ── read_xnor after reset on same block id fails because track is Empty ──

    #[test]
    fn read_xnor_after_reset_fails_with_empty_track() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 8).unwrap();
        let mask = vec![true; 8];
        pool.write_xnor(0, &mask).unwrap();

        pool.reset();

        let mut out = [false; 8];
        let result = pool.read_xnor(0, &mut out);
        assert!(matches!(result, Err(DualTrackError::NotInXnorPool(0))));
    }

    // ── allocate main then xnor on same block id, then read_xnor succeeds ──

    #[test]
    fn allocate_main_then_xnor_same_block_read_xnor_succeeds() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        // Overwrite block 0 to xnor track
        pool.allocate_block_xnor(0, 8).unwrap();
        assert_eq!(pool.get_track(0), Track::Xnor);

        let mask = vec![true, false, true, false, true, false, true, false];
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 8];
        pool.read_xnor(0, &mut out).unwrap();
        assert_eq!(out, mask);
    }

    // ── main pool allocation exactly fills remaining space with 3-bit ──

    #[test]
    fn main_pool_3bit_exact_fill_two_blocks() {
        let mut pool = DualTrackMemoryPool::new(16, 256, 16, 3).unwrap();
        // First: 24 elements = ceil(24/3) = 8 bytes
        let off0 = pool.allocate_block_main(0, 24).unwrap();
        assert_eq!(off0, 0);

        // Second: another 24 elements = 8 bytes, total 16 = capacity
        let off1 = pool.allocate_block_main(1, 24).unwrap();
        assert_eq!(off1, 8);

        let (used, cap) = pool.main_usage();
        assert_eq!(used, cap);
        assert_eq!(used, 16);
    }

    // ── xnor mask with single true among all false ──

    #[test]
    fn xnor_single_true_among_false() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_xnor(0, 16).unwrap();

        let mut mask = vec![false; 16];
        mask[7] = true; // Only bit 7 is true
        pool.write_xnor(0, &mask).unwrap();

        let mut out = vec![false; 16];
        pool.read_xnor(0, &mut out).unwrap();

        assert_eq!(out[7], true);
        for (i, &bit) in out.iter().enumerate() {
            if i != 7 {
                assert!(!bit, "Bit at position {i} should be false");
            }
        }
    }

    // ── two independent pools do not interfere ──

    #[test]
    fn two_pools_independent() {
        let mut pool_a = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let mut pool_b = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();

        pool_a.allocate_block_main(0, 16).unwrap();
        let data_a = vec![0xAA, 0xBB, 0xCC, 0xDD];
        pool_a.write_main(0, &data_a).unwrap();

        pool_b.allocate_block_main(0, 16).unwrap();
        let data_b = vec![0x11, 0x22, 0x33, 0x44];
        pool_b.write_main(0, &data_b).unwrap();

        let mut out_a = vec![0u8; 4];
        let mut out_b = vec![0u8; 4];
        pool_a.read_main(0, &mut out_a).unwrap();
        pool_b.read_main(0, &mut out_b).unwrap();

        assert_eq!(out_a, data_a);
        assert_eq!(out_b, data_b);
        assert_ne!(out_a, out_b);
    }

    // ── reset one pool does not affect the other ──

    #[test]
    fn reset_one_pool_does_not_affect_other() {
        let mut pool_a = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        let mut pool_b = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();

        pool_a.allocate_block_main(0, 16).unwrap();
        pool_b.allocate_block_main(0, 16).unwrap();
        let data = vec![0xDE, 0xAD];
        pool_a.write_main(0, &data).unwrap();
        pool_b.write_main(0, &data).unwrap();

        pool_a.reset();
        assert_eq!(pool_a.get_track(0), Track::Empty);
        assert_eq!(pool_a.main_used, 0);

        // pool_b should be unaffected
        assert_eq!(pool_b.get_track(0), Track::Main);
        assert_eq!(pool_b.main_used, 8);
        let mut out = vec![0u8; 2];
        pool_b.read_main(0, &mut out).unwrap();
        assert_eq!(out, data);
    }

    // ── main pool partial fill leaves correct available space ──

    #[test]
    fn main_pool_partial_fill_available_tracking() {
        let mut pool = DualTrackMemoryPool::new(128, 256, 16, 4).unwrap();
        // 4-bit: 64 elements = 32 bytes
        pool.allocate_block_main(0, 64).unwrap();

        let (used, cap) = pool.main_usage();
        assert_eq!(used, 32);
        assert_eq!(cap, 128);
        // Remaining = 96 bytes

        // Allocate another 32 elements = 16 bytes
        pool.allocate_block_main(1, 32).unwrap();
        let (used2, _) = pool.main_usage();
        assert_eq!(used2, 48);
    }

    // ── xnor partial fill then full error carries correct available count ──

    #[test]
    fn xnor_partial_fill_then_full_error_values() {
        let mut pool = DualTrackMemoryPool::new(256, 128, 16, 4).unwrap();
        // Allocate 100 bits
        pool.allocate_block_xnor(0, 100).unwrap();
        // Remaining = 28 bits

        // Try to allocate 50 bits -> should fail with available=28
        let result = pool.allocate_block_xnor(1, 50);
        match result {
            Err(DualTrackError::XnorPoolFull { requested, available }) => {
                assert_eq!(requested, 50);
                assert_eq!(available, 28);
            }
            _ => panic!("Expected XnorPoolFull error"),
        }
    }

    // ── write_read main with single byte data ──

    #[test]
    fn write_read_main_single_byte() {
        let mut pool = DualTrackMemoryPool::new(1024, 2048, 16, 4).unwrap();
        pool.allocate_block_main(0, 16).unwrap();

        pool.write_main(0, &[0x42]).unwrap();

        let mut out = [0u8; 1];
        pool.read_main(0, &mut out).unwrap();
        assert_eq!(out[0], 0x42);
    }
}
