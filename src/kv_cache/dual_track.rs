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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        let bytes_per_block = (elements_per_block + bytes_per_element - 1) / bytes_per_element;
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
            xnor_pool: vec![0u8; (xnor_capacity_bits + 7) / 8],
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

        let size_bytes = (num_elements + bytes_per_element - 1) / bytes_per_element;
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
}
