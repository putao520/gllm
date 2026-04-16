//! RaggedCompaction — Compact→Execute→Scatter 三段式融合 (SPEC §9.1)
//!
//! ## 核心职责
//! 在 Mega-Kernel 内处理异构 batch：不同请求可能有不同的 active 状态
//! （如已完成的请求、被 Early-Exit 截断的请求、被 Guardrail Veto 的请求）。
//!
//! ## 三段式流程
//! 1. **Compact**: 将活跃元素挤压到连续内存 (vcompressps / warp prefix sum / SVE predicate)
//! 2. **Execute**: 在紧凑数据上执行 GEMM/Attention（100% SIMD lane 利用率）
//! 3. **Scatter**: 按原始偏移写回结果
//!
//! ## 约束
//! - Compact/Scatter 开销必须 < wasted SIMD lanes 的浪费（阈值: 25%）
//! - 禁止在 Mega-Kernel 内部分配额外内存 — 所有 buffer 预分配
//! - 与 VariableLengthBatch OpKind 协同工作

use std::fmt;

/// SIMD lane 浪费率阈值 — 超过此值触发 compact
///
/// SPEC §17.6.1: "自适应阈值: SIMD lane 浪费 > 25% 时触发 compact, 否则跳过"
pub const COMPACT_THRESHOLD: f32 = 0.25;

/// 硬件平台 — 决定 Compact/Scatter 的具体实现路径
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactPlatform {
    /// x86_64: AVX-512 vcompressps / vpcompressd
    X86Avx512,
    /// x86_64: AVX2 通用寄存器 emulation (无硬件 compress)
    X86Avx2,
    /// AArch64 NEON: 软件 compact
    Aarch64Neon,
    /// AArch64 SVE: predicate register 自动 mask（无需显式 compact）
    Aarch64Sve { vl_bytes: usize },
    /// NVIDIA GPU: Warp Vote + Prefix Sum + __shfl_sync
    GpuCuda { warp_size: u32 },
    /// AMD GPU: 同 CUDA，warp_size 可能不同 (wavefront = 32 or 64)
    GpuHip { wavefront_size: u32 },
    /// Apple Metal: SIMD group vote + prefix sum
    GpuMetal,
}

impl CompactPlatform {
    /// 从设备信息推导 CompactPlatform
    pub fn detect(
        backend: &str,
        has_avx512: bool,
        has_sve: bool,
        sve_vl_bytes: usize,
        warp_size: u32,
    ) -> Self {
        match backend {
            "cuda" => Self::GpuCuda { warp_size },
            "hip" => Self::GpuHip { wavefront_size: warp_size },
            "metal" => Self::GpuMetal,
            _ => {
                // CPU
                if has_sve {
                    Self::Aarch64Sve { vl_bytes: sve_vl_bytes }
                } else if has_avx512 {
                    Self::X86Avx512
                } else {
                    Self::X86Avx2
                }
            }
        }
    }

    /// 返回该平台是否支持硬件级 compact 指令
    pub fn has_hardware_compress(&self) -> bool {
        matches!(
            self,
            Self::X86Avx512 | Self::Aarch64Sve { .. }
        )
    }

    /// 返回 SIMD 宽度（bytes）
    pub fn simd_width_bytes(&self) -> usize {
        match self {
            Self::X86Avx512 => 64,       // 512-bit = 64 bytes
            Self::X86Avx2 => 32,         // 256-bit = 32 bytes
            Self::Aarch64Neon => 16,     // 128-bit = 16 bytes
            Self::Aarch64Sve { vl_bytes } => *vl_bytes,
            Self::GpuCuda { warp_size } => (*warp_size as usize) * std::mem::size_of::<f32>(), // 32 threads × sizeof(f32)
            Self::GpuHip { wavefront_size } => (*wavefront_size as usize) * std::mem::size_of::<f32>(),
            Self::GpuMetal => 128,       // SIMD group = 32 threads × sizeof(f32)
        }
    }
}

/// Per-request 活跃状态 — Compact 阶段的输入
#[derive(Debug, Clone)]
pub struct RequestActiveMask {
    /// 每个请求是否活跃 (true = 活跃, 需要参与计算)
    mask: Vec<bool>,
    /// 活跃请求数量
    active_count: usize,
}

impl RequestActiveMask {
    /// 从活跃 mask 创建
    pub fn new(mask: Vec<bool>) -> Self {
        let active_count = mask.iter().filter(|&&b| b).count();
        Self { mask, active_count }
    }

    /// 全部活跃的 mask
    pub fn all_active(batch_size: usize) -> Self {
        Self {
            mask: vec![true; batch_size],
            active_count: batch_size,
        }
    }

    /// 返回 batch 大小
    pub fn batch_size(&self) -> usize {
        self.mask.len()
    }

    /// 返回活跃请求数量
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// 返回 SIMD lane 浪费率
    pub fn waste_ratio(&self) -> f32 {
        if self.mask.is_empty() {
            return 0.0;
        }
        1.0 - (self.active_count as f32 / self.mask.len() as f32)
    }

    /// 是否应该触发 compact（浪费率 > 阈值）
    pub fn should_compact(&self) -> bool {
        self.waste_ratio() > COMPACT_THRESHOLD
    }

    /// 返回原始 mask 的引用
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

/// Compact 索引映射 — 记录原始位置 ↔ 紧凑位置的关系
///
/// 在 Compact 阶段生成，在 Execute 阶段使用紧凑索引，
/// 在 Scatter 阶段使用反向映射写回。
#[derive(Debug, Clone)]
pub struct CompactIndex {
    /// original_to_compact[i] = 原始位置 i 在紧凑数据中的位置
    /// 如果 i 不活跃，值为 usize::MAX
    original_to_compact: Vec<usize>,

    /// compact_to_original[j] = 紧凑位置 j 对应的原始位置
    compact_to_original: Vec<usize>,

    /// 活跃请求数量 = 紧凑数据的长度
    active_count: usize,
}

impl CompactIndex {
    /// 从活跃 mask 构建 CompactIndex（prefix sum 算法）
    pub fn from_mask(mask: &RequestActiveMask) -> Self {
        let n = mask.batch_size();
        let mut original_to_compact = vec![usize::MAX; n];
        let mut compact_to_original = Vec::with_capacity(mask.active_count());

        let mut compact_pos = 0;
        for (i, &active) in mask.mask().iter().enumerate() {
            if active {
                original_to_compact[i] = compact_pos;
                compact_to_original.push(i);
                compact_pos += 1;
            }
        }

        Self {
            original_to_compact,
            compact_to_original,
            active_count: compact_pos,
        }
    }

    /// 原始位置 → 紧凑位置 (usize::MAX 表示不活跃)
    pub fn to_compact(&self, original_idx: usize) -> usize {
        self.original_to_compact[original_idx]
    }

    /// 紧凑位置 → 原始位置
    pub fn to_original(&self, compact_idx: usize) -> usize {
        self.compact_to_original[compact_idx]
    }

    /// 活跃数量
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// 是否为空（无活跃请求）
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// compact_to_original 切片
    pub fn compact_to_original(&self) -> &[usize] {
        &self.compact_to_original
    }

    /// original_to_compact 切片
    pub fn original_to_compact(&self) -> &[usize] {
        &self.original_to_compact
    }
}

/// 三段式执行计划的决策结果
#[derive(Debug, Clone)]
pub enum CompactDecision {
    /// 跳过 Compact/Scatter — 浪费率低于阈值，直接执行
    SkipDirect {
        /// batch 大小
        batch_size: usize,
    },
    /// 执行 Compact→Execute→Scatter
    Compact {
        /// 原始 batch 大小
        original_batch_size: usize,
        /// 紧凑后的 batch 大小（活跃数量）
        compact_batch_size: usize,
        /// Compact 索引映射
        index: CompactIndex,
        /// 硬件平台
        platform: CompactPlatform,
    },
}

impl CompactDecision {
    /// 根据活跃 mask 和硬件平台决策是否需要 compact
    pub fn decide(mask: RequestActiveMask, platform: CompactPlatform) -> Self {
        // SVE 平台天然支持 predicate mask，不需要显式 compact
        if matches!(platform, CompactPlatform::Aarch64Sve { .. }) {
            return CompactDecision::SkipDirect {
                batch_size: mask.batch_size(),
            };
        }

        // 如果浪费率低于阈值，跳过 compact
        if !mask.should_compact() {
            return CompactDecision::SkipDirect {
                batch_size: mask.batch_size(),
            };
        }

        // 全部活跃，无需 compact
        if mask.active_count() == mask.batch_size() {
            return CompactDecision::SkipDirect {
                batch_size: mask.batch_size(),
            };
        }

        // 无活跃请求 — 不执行任何操作
        if mask.active_count() == 0 {
            return CompactDecision::SkipDirect {
                batch_size: 0,
            };
        }

        let index = CompactIndex::from_mask(&mask);
        let original_batch_size = mask.batch_size();
        let compact_batch_size = mask.active_count();

        CompactDecision::Compact {
            original_batch_size,
            compact_batch_size,
            index,
            platform,
        }
    }

    /// 返回有效 batch 大小（紧凑后或原始）
    pub fn effective_batch_size(&self) -> usize {
        match self {
            CompactDecision::SkipDirect { batch_size } => *batch_size,
            CompactDecision::Compact { compact_batch_size, .. } => *compact_batch_size,
        }
    }

    /// 是否需要 compact
    pub fn needs_compact(&self) -> bool {
        matches!(self, CompactDecision::Compact { .. })
    }
}

/// Compact 阶段的输出 — 紧凑化的连续数据
///
/// 泛型 T 表示数据类型 (f32, u8 等)
#[derive(Debug, Clone)]
pub struct CompactData<T: Clone> {
    /// 紧凑化的数据（仅包含活跃请求）
    pub data: Vec<T>,
    /// 紧凑数据的逻辑形状 [compact_batch, ...]
    pub compact_shape: Vec<usize>,
    /// 索引映射
    pub index: CompactIndex,
}

impl<T: Clone + Copy + Default> CompactData<T> {
    /// 从原始数据执行 Compact 阶段
    ///
    /// `source`: 原始数据, shape = [batch_size, inner_dim...]
    /// `index`: Compact 索引映射
    /// `inner_dims`: 除 batch 维度外的内部维度
    pub fn compact(source: &[T], index: &CompactIndex, inner_dims: &[usize]) -> Self {
        let inner_size: usize = inner_dims.iter().product();
        let compact_batch = index.active_count();
        let mut data = vec![T::default(); compact_batch * inner_size];

        for compact_idx in 0..compact_batch {
            let original_idx = index.to_original(compact_idx);
            let src_offset = original_idx * inner_size;
            let dst_offset = compact_idx * inner_size;
            data[dst_offset..dst_offset + inner_size]
                .copy_from_slice(&source[src_offset..src_offset + inner_size]);
        }

        let mut compact_shape = vec![compact_batch];
        compact_shape.extend_from_slice(inner_dims);

        CompactData {
            data,
            compact_shape,
            index: index.clone(),
        }
    }
}

/// Scatter 阶段 — 将紧凑数据写回原始位置
pub struct ScatterWriter<'a, T: Clone> {
    /// 目标 buffer（原始大小）
    target: &'a mut [T],
    /// 索引映射
    index: &'a CompactIndex,
    /// 内部维度大小
    inner_size: usize,
}

impl<'a, T: Clone + Copy> ScatterWriter<'a, T> {
    /// 创建 ScatterWriter
    pub fn new(
        target: &'a mut [T],
        index: &'a CompactIndex,
        inner_dims: &[usize],
    ) -> Self {
        let inner_size: usize = inner_dims.iter().product();
        Self {
            target,
            index,
            inner_size,
        }
    }

    /// 执行 Scatter — 将紧凑结果写回原始位置
    pub fn scatter(&mut self, compact_data: &[T]) {
        for compact_idx in 0..self.index.active_count() {
            let original_idx = self.index.to_original(compact_idx);
            let src_offset = compact_idx * self.inner_size;
            let dst_offset = original_idx * self.inner_size;
            self.target[dst_offset..dst_offset + self.inner_size]
                .copy_from_slice(&compact_data[src_offset..src_offset + self.inner_size]);
        }
    }
}

/// 完整的 Compact→Execute→Scatter 执行上下文
///
/// 封装三段式流程的管理，与 JIT 图执行器集成
pub struct RaggedCompaction {
    /// 硬件平台
    platform: CompactPlatform,
}

impl RaggedCompaction {
    /// 创建新的 RaggedCompaction 上下文
    pub fn new(platform: CompactPlatform) -> Self {
        Self { platform }
    }

    /// 执行完整的 Compact→Execute→Scatter 流程
    ///
    /// # 类型参数
    /// - `T`: 数据类型
    /// - `F`: Execute 阶段的闭包，接收紧凑数据和形状，返回计算结果
    ///
    /// # 参数
    /// - `input`: 原始输入数据 [batch_size, inner_size]
    /// - `inner_dims`: 除 batch 维度外的内部维度
    /// - `mask`: 请求活跃 mask
    /// - `executor`: Execute 阶段的闭包
    ///
    /// # 返回
    /// - 写回到原始位置的结果数据
    pub fn execute<T, F>(
        &self,
        input: &[T],
        inner_dims: &[usize],
        mask: RequestActiveMask,
        output: &mut [T],
        executor: F,
    ) where
        T: Clone + Copy + Default,
        F: FnOnce(&[T], &[usize]) -> Vec<T>,
    {
        let decision = CompactDecision::decide(mask, self.platform);

        match decision {
            CompactDecision::SkipDirect { batch_size } => {
                if batch_size == 0 {
                    return;
                }
                // 直接执行，无 compact/scatter
                let mut shape = vec![batch_size];
                shape.extend_from_slice(inner_dims);
                let result = executor(input, &shape);
                let _ = &result;
                output.copy_from_slice(&result);
            }
            CompactDecision::Compact {
                original_batch_size: _,
                compact_batch_size,
                index,
                platform: _,
            } => {
                // Phase 1: Compact
                let compact_input = CompactData::compact(input, &index, inner_dims);

                // Phase 2: Execute
                let compact_result = executor(&compact_input.data, &compact_input.compact_shape);

                // Phase 3: Scatter
                let mut writer = ScatterWriter::new(output, &index, inner_dims);
                writer.scatter(&compact_result);

                let _ = compact_batch_size; // suppress unused warning
            }
        }
    }

    /// 返回硬件平台
    pub fn platform(&self) -> &CompactPlatform {
        &self.platform
    }

    /// 决策是否需要 compact（不执行实际计算）
    pub fn should_compact(&self, mask: &RequestActiveMask) -> bool {
        // SVE 天然支持 predicate，不需要 compact
        if matches!(self.platform, CompactPlatform::Aarch64Sve { .. }) {
            return false;
        }
        mask.should_compact()
    }
}

impl fmt::Display for CompactPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86Avx512 => write!(f, "x86_avx512"),
            Self::X86Avx2 => write!(f, "x86_avx2"),
            Self::Aarch64Neon => write!(f, "aarch64_neon"),
            Self::Aarch64Sve { vl_bytes } => write!(f, "aarch64_sve_{}b", vl_bytes),
            Self::GpuCuda { warp_size } => write!(f, "cuda_w{}", warp_size),
            Self::GpuHip { wavefront_size } => write!(f, "hip_wf{}", wavefront_size),
            Self::GpuMetal => write!(f, "metal"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_active_mask_basic() {
        let mask = RequestActiveMask::new(vec![true, false, true, true, false]);
        assert_eq!(mask.batch_size(), 5);
        assert_eq!(mask.active_count(), 3);
        assert!((mask.waste_ratio() - 0.4).abs() < 0.001);
        assert!(mask.should_compact()); // 40% > 25%
    }

    #[test]
    fn test_request_active_mask_all_active() {
        let mask = RequestActiveMask::all_active(8);
        assert_eq!(mask.active_count(), 8);
        assert!((mask.waste_ratio()).abs() < 0.001);
        assert!(!mask.should_compact()); // 0% < 25%
    }

    #[test]
    fn test_request_active_mask_low_waste() {
        // 7/8 active = 12.5% waste, below threshold
        let mask = RequestActiveMask::new(vec![true, true, true, true, true, true, true, false]);
        assert_eq!(mask.active_count(), 7);
        assert!(!mask.should_compact()); // 12.5% < 25%
    }

    #[test]
    fn test_compact_index_basic() {
        let mask = RequestActiveMask::new(vec![true, false, true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        assert_eq!(index.active_count(), 3);
        // original 0 → compact 0
        assert_eq!(index.to_compact(0), 0);
        // original 1 → inactive
        assert_eq!(index.to_compact(1), usize::MAX);
        // original 2 → compact 1
        assert_eq!(index.to_compact(2), 1);
        // original 3 → inactive
        assert_eq!(index.to_compact(3), usize::MAX);
        // original 4 → compact 2
        assert_eq!(index.to_compact(4), 2);

        // Reverse mapping
        assert_eq!(index.to_original(0), 0);
        assert_eq!(index.to_original(1), 2);
        assert_eq!(index.to_original(2), 4);
    }

    #[test]
    fn test_compact_index_all_active() {
        let mask = RequestActiveMask::all_active(4);
        let index = CompactIndex::from_mask(&mask);

        assert_eq!(index.active_count(), 4);
        for i in 0..4 {
            assert_eq!(index.to_compact(i), i);
            assert_eq!(index.to_original(i), i);
        }
    }

    #[test]
    fn test_compact_data_f32() {
        // 4 requests, hidden=2, request 1 inactive
        let source: Vec<f32> = vec![
            1.0, 2.0,  // req 0
            3.0, 4.0,  // req 1 (inactive)
            5.0, 6.0,  // req 2
            7.0, 8.0,  // req 3
        ];
        let mask = RequestActiveMask::new(vec![true, false, true, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[2]);

        assert_eq!(compact.compact_shape, vec![3, 2]);
        // compact: req 0, req 2, req 3
        assert_eq!(compact.data, vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_scatter_write_back() {
        // 4 requests, hidden=2
        let mut output = vec![0.0f32; 8];
        let mask = RequestActiveMask::new(vec![true, false, true, true]);
        let index = CompactIndex::from_mask(&mask);

        // compact result: req 0 → [10.0, 20.0], req 2 → [50.0, 60.0], req 3 → [70.0, 80.0]
        let compact_result = vec![10.0, 20.0, 50.0, 60.0, 70.0, 80.0];

        let mut writer = ScatterWriter::new(&mut output, &index, &[2]);
        writer.scatter(&compact_result);

        // Verify: req 0, req 1 untouched (0.0), req 2, req 3
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 20.0);
        assert_eq!(output[2], 0.0);  // req 1 untouched
        assert_eq!(output[3], 0.0);  // req 1 untouched
        assert_eq!(output[4], 50.0);
        assert_eq!(output[5], 60.0);
        assert_eq!(output[6], 70.0);
        assert_eq!(output[7], 80.0);
    }

    #[test]
    fn test_compact_decision_skip() {
        // 7/8 active = 12.5% waste, below threshold
        let mask = RequestActiveMask::new(vec![true, true, true, true, true, true, true, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx512);

        assert!(!decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 8);
    }

    #[test]
    fn test_compact_decision_trigger() {
        // 3/8 active = 62.5% waste, above threshold
        let mask = RequestActiveMask::new(vec![true, false, false, true, false, false, true, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx512);

        assert!(decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 3);
    }

    #[test]
    fn test_sve_always_skip() {
        // SVE never needs explicit compact — predicate register handles it
        let mask = RequestActiveMask::new(vec![true, false]); // 50% waste
        let decision = CompactDecision::decide(
            mask,
            CompactPlatform::Aarch64Sve { vl_bytes: 32 },
        );

        assert!(!decision.needs_compact());
    }

    #[test]
    fn test_ragged_compaction_full_flow() {
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx512);

        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0,  // req 0
            4.0, 5.0, 6.0,  // req 1 (inactive)
            7.0, 8.0, 9.0,  // req 2
            10.0, 11.0, 12.0, // req 3 (inactive)
            13.0, 14.0, 15.0, // req 4
        ];
        let mask = RequestActiveMask::new(vec![true, false, true, false, true]);
        let mut output = vec![0.0f32; 15];

        rc.execute(&input, &[3], mask, &mut output, |compact_data, shape| {
            // Execute: simple element-wise multiply by 2
            assert_eq!(shape, &[3, 3]);
            compact_data.iter().map(|&x| x * 2.0).collect()
        });

        // Verify scatter: req 0, req 1 untouched, req 2, req 3 untouched, req 4
        assert_eq!(output[0], 2.0);   // req 0: 1*2
        assert_eq!(output[1], 4.0);   // req 0: 2*2
        assert_eq!(output[2], 6.0);   // req 0: 3*2
        assert_eq!(output[3], 0.0);   // req 1: untouched
        assert_eq!(output[6], 14.0);  // req 2: 7*2
        assert_eq!(output[9], 0.0);   // req 3: untouched
        assert_eq!(output[12], 26.0); // req 4: 13*2
    }

    #[test]
    fn test_platform_detect() {
        assert_eq!(
            CompactPlatform::detect("cpu", true, false, 0, 32),
            CompactPlatform::X86Avx512
        );
        assert_eq!(
            CompactPlatform::detect("cpu", false, false, 0, 32),
            CompactPlatform::X86Avx2
        );
        assert_eq!(
            CompactPlatform::detect("cpu", false, true, 32, 32),
            CompactPlatform::Aarch64Sve { vl_bytes: 32 }
        );
        assert_eq!(
            CompactPlatform::detect("cuda", false, false, 0, 32),
            CompactPlatform::GpuCuda { warp_size: 32 }
        );
        assert_eq!(
            CompactPlatform::detect("hip", false, false, 0, 64),
            CompactPlatform::GpuHip { wavefront_size: 64 }
        );
        assert_eq!(
            CompactPlatform::detect("metal", false, false, 0, 32),
            CompactPlatform::GpuMetal
        );
    }

    #[test]
    fn test_empty_mask() {
        let mask = RequestActiveMask::new(vec![false, false, false]);
        assert_eq!(mask.active_count(), 0);
        assert!(mask.waste_ratio() > 0.99);

        let index = CompactIndex::from_mask(&mask);
        assert_eq!(index.active_count(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_single_active() {
        let mask = RequestActiveMask::new(vec![false, false, true, false]);
        let index = CompactIndex::from_mask(&mask);

        assert_eq!(index.active_count(), 1);
        assert_eq!(index.to_compact(2), 0);
        assert_eq!(index.to_original(0), 2);
    }
}
