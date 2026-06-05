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

    // ── CompactPlatform Display trait ──────────────────────────────────

    #[test]
    fn test_platform_display_x86_avx512() {
        assert_eq!(format!("{}", CompactPlatform::X86Avx512), "x86_avx512");
    }

    #[test]
    fn test_platform_display_x86_avx2() {
        assert_eq!(format!("{}", CompactPlatform::X86Avx2), "x86_avx2");
    }

    #[test]
    fn test_platform_display_aarch64_neon() {
        assert_eq!(format!("{}", CompactPlatform::Aarch64Neon), "aarch64_neon");
    }

    #[test]
    fn test_platform_display_aarch64_sve() {
        let platform = CompactPlatform::Aarch64Sve { vl_bytes: 64 };
        assert_eq!(format!("{}", platform), "aarch64_sve_64b");
    }

    #[test]
    fn test_platform_display_cuda() {
        let platform = CompactPlatform::GpuCuda { warp_size: 32 };
        assert_eq!(format!("{}", platform), "cuda_w32");
    }

    #[test]
    fn test_platform_display_hip() {
        let platform = CompactPlatform::GpuHip { wavefront_size: 64 };
        assert_eq!(format!("{}", platform), "hip_wf64");
    }

    #[test]
    fn test_platform_display_metal() {
        assert_eq!(format!("{}", CompactPlatform::GpuMetal), "metal");
    }

    // ── CompactPlatform has_hardware_compress ──────────────────────────

    #[test]
    fn test_has_hardware_compress_avx512() {
        assert!(CompactPlatform::X86Avx512.has_hardware_compress());
    }

    #[test]
    fn test_has_hardware_compress_sve() {
        let sve = CompactPlatform::Aarch64Sve { vl_bytes: 32 };
        assert!(sve.has_hardware_compress());
    }

    #[test]
    fn test_has_hardware_compress_sve_large_vl() {
        let sve = CompactPlatform::Aarch64Sve { vl_bytes: 256 };
        assert!(sve.has_hardware_compress());
    }

    #[test]
    fn test_no_hardware_compress_avx2() {
        assert!(!CompactPlatform::X86Avx2.has_hardware_compress());
    }

    #[test]
    fn test_no_hardware_compress_neon() {
        assert!(!CompactPlatform::Aarch64Neon.has_hardware_compress());
    }

    #[test]
    fn test_no_hardware_compress_cuda() {
        let cuda = CompactPlatform::GpuCuda { warp_size: 32 };
        assert!(!cuda.has_hardware_compress());
    }

    #[test]
    fn test_no_hardware_compress_hip() {
        let hip = CompactPlatform::GpuHip { wavefront_size: 64 };
        assert!(!hip.has_hardware_compress());
    }

    #[test]
    fn test_no_hardware_compress_metal() {
        assert!(!CompactPlatform::GpuMetal.has_hardware_compress());
    }

    // ── CompactPlatform simd_width_bytes ───────────────────────────────

    #[test]
    fn test_simd_width_avx512() {
        assert_eq!(CompactPlatform::X86Avx512.simd_width_bytes(), 64);
    }

    #[test]
    fn test_simd_width_avx2() {
        assert_eq!(CompactPlatform::X86Avx2.simd_width_bytes(), 32);
    }

    #[test]
    fn test_simd_width_neon() {
        assert_eq!(CompactPlatform::Aarch64Neon.simd_width_bytes(), 16);
    }

    #[test]
    fn test_simd_width_sve() {
        let sve = CompactPlatform::Aarch64Sve { vl_bytes: 48 };
        assert_eq!(sve.simd_width_bytes(), 48);
    }

    #[test]
    fn test_simd_width_cuda_warp32() {
        let cuda = CompactPlatform::GpuCuda { warp_size: 32 };
        assert_eq!(cuda.simd_width_bytes(), 128); // 32 * 4 bytes
    }

    #[test]
    fn test_simd_width_hip_wavefront64() {
        let hip = CompactPlatform::GpuHip { wavefront_size: 64 };
        assert_eq!(hip.simd_width_bytes(), 256); // 64 * 4 bytes
    }

    #[test]
    fn test_simd_width_metal() {
        assert_eq!(CompactPlatform::GpuMetal.simd_width_bytes(), 128);
    }

    // ── CompactPlatform PartialEq + Copy + Clone ───────────────────────

    #[test]
    fn test_platform_equality_sve_same_vl() {
        let a = CompactPlatform::Aarch64Sve { vl_bytes: 32 };
        let b = CompactPlatform::Aarch64Sve { vl_bytes: 32 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_platform_inequality_sve_different_vl() {
        let a = CompactPlatform::Aarch64Sve { vl_bytes: 16 };
        let b = CompactPlatform::Aarch64Sve { vl_bytes: 32 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_platform_copy_clone() {
        let original = CompactPlatform::GpuCuda { warp_size: 32 };
        let cloned = original.clone();
        let copied = original; // Copy
        assert_eq!(original, cloned);
        assert_eq!(original, copied);
    }

    // ── CompactPlatform::detect edge cases ─────────────────────────────

    #[test]
    fn test_detect_unknown_backend_with_avx512() {
        // Any unknown backend string falls through to CPU detection
        let platform = CompactPlatform::detect("custom_accelerator", true, false, 0, 32);
        assert_eq!(platform, CompactPlatform::X86Avx512);
    }

    #[test]
    fn test_detect_sve_priority_over_avx512() {
        // SVE takes priority over AVX512 when both are true
        let platform = CompactPlatform::detect("cpu", true, true, 64, 32);
        assert_eq!(platform, CompactPlatform::Aarch64Sve { vl_bytes: 64 });
    }

    #[test]
    fn test_detect_cpu_no_features() {
        let platform = CompactPlatform::detect("cpu", false, false, 0, 32);
        assert_eq!(platform, CompactPlatform::X86Avx2);
    }

    // ── COMPACT_THRESHOLD constant ─────────────────────────────────────

    #[test]
    fn test_compact_threshold_value() {
        assert!((COMPACT_THRESHOLD - 0.25).abs() < f32::EPSILON);
    }

    // ── RequestActiveMask edge cases ───────────────────────────────────

    #[test]
    fn test_mask_empty_vector() {
        let mask = RequestActiveMask::new(vec![]);
        assert_eq!(mask.batch_size(), 0);
        assert_eq!(mask.active_count(), 0);
        assert_eq!(mask.waste_ratio(), 0.0);
        assert!(!mask.should_compact()); // 0% waste, empty edge case
    }

    #[test]
    fn test_mask_batch_size_one_active() {
        let mask = RequestActiveMask::new(vec![true]);
        assert_eq!(mask.batch_size(), 1);
        assert_eq!(mask.active_count(), 1);
        assert!(!mask.should_compact()); // 0% waste
    }

    #[test]
    fn test_mask_batch_size_one_inactive() {
        let mask = RequestActiveMask::new(vec![false]);
        assert_eq!(mask.batch_size(), 1);
        assert_eq!(mask.active_count(), 0);
        assert!((mask.waste_ratio() - 1.0).abs() < 0.001);
        assert!(mask.should_compact()); // 100% > 25%
    }

    #[test]
    fn test_mask_all_inactive() {
        let mask = RequestActiveMask::new(vec![false, false, false, false]);
        assert_eq!(mask.active_count(), 0);
        assert_eq!(mask.batch_size(), 4);
        assert!((mask.waste_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mask_waste_ratio_exact_threshold() {
        // Exactly 75% active = 25% waste = exactly at threshold
        // should_compact uses strict >, so exactly 25% should NOT trigger
        let mask = RequestActiveMask::new(vec![true, true, true, false]);
        assert!((mask.waste_ratio() - 0.25).abs() < 0.001);
        assert!(!mask.should_compact()); // 25% is NOT > 25%
    }

    #[test]
    fn test_mask_waste_ratio_just_above_threshold() {
        // 2/8 = 25% active = 75% waste > 25%
        let mask = RequestActiveMask::new(vec![true, false, false, false, true, false, false, false]);
        assert_eq!(mask.active_count(), 2);
        assert_eq!(mask.batch_size(), 8);
        let waste = mask.waste_ratio();
        assert!(waste > 0.25, "waste {} should be > 0.25", waste);
        assert!(mask.should_compact());
    }

    #[test]
    fn test_mask_accessor_returns_correct_slice() {
        let raw = vec![true, false, true];
        let mask = RequestActiveMask::new(raw.clone());
        assert_eq!(mask.mask(), &raw);
    }

    #[test]
    fn test_mask_all_active_zero_batch() {
        let mask = RequestActiveMask::all_active(0);
        assert_eq!(mask.batch_size(), 0);
        assert_eq!(mask.active_count(), 0);
    }

    // ── CompactIndex edge cases ────────────────────────────────────────

    #[test]
    fn test_index_is_empty_false() {
        let mask = RequestActiveMask::new(vec![true, false]);
        let index = CompactIndex::from_mask(&mask);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_index_is_empty_true_all_inactive() {
        let mask = RequestActiveMask::new(vec![false, false, false]);
        let index = CompactIndex::from_mask(&mask);
        assert!(index.is_empty());
        assert_eq!(index.active_count(), 0);
    }

    #[test]
    fn test_index_slice_accessors() {
        let mask = RequestActiveMask::new(vec![true, false, true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        assert_eq!(index.compact_to_original(), &[0, 2, 4]);
        assert_eq!(index.original_to_compact(), &[0, usize::MAX, 1, usize::MAX, 2]);
    }

    #[test]
    fn test_index_all_inactive_original_to_compact() {
        let mask = RequestActiveMask::new(vec![false, false]);
        let index = CompactIndex::from_mask(&mask);

        assert_eq!(index.original_to_compact(), &[usize::MAX, usize::MAX]);
        assert_eq!(index.compact_to_original(), &[] as &[usize]);
    }

    #[test]
    fn test_index_clone_preserves_mapping() {
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);
        let cloned = index.clone();

        assert_eq!(cloned.active_count(), index.active_count());
        assert_eq!(cloned.to_compact(0), 0);
        assert_eq!(cloned.to_compact(1), usize::MAX);
        assert_eq!(cloned.to_compact(2), 1);
        assert_eq!(cloned.to_original(0), 0);
        assert_eq!(cloned.to_original(1), 2);
    }

    #[test]
    fn test_index_identity_all_active() {
        let mask = RequestActiveMask::all_active(5);
        let index = CompactIndex::from_mask(&mask);

        for i in 0..5 {
            assert_eq!(index.to_compact(i), i);
            assert_eq!(index.to_original(i), i);
        }
    }

    // ── CompactData with multi-dimensional inner dims ──────────────────

    #[test]
    fn test_compact_data_multi_inner_dims() {
        // batch=3, inner=[2, 3] → 6 elements per request
        // request 1 inactive
        let source: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,     // req 0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, // req 1 (inactive)
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,   // req 2
        ];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[2, 3]);

        assert_eq!(compact.compact_shape, vec![2, 2, 3]);
        assert_eq!(compact.data, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
    }

    #[test]
    fn test_compact_data_u8_type() {
        let source: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[2]);

        assert_eq!(compact.compact_shape, vec![2, 2]);
        assert_eq!(compact.data, vec![10, 20, 50, 60]);
    }

    #[test]
    fn test_compact_data_all_active_no_op() {
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mask = RequestActiveMask::all_active(2);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[2]);

        assert_eq!(compact.compact_shape, vec![2, 2]);
        assert_eq!(compact.data, vec![1.0, 2.0, 3.0, 4.0]); // unchanged
    }

    #[test]
    fn test_compact_data_single_inner_dim() {
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = RequestActiveMask::new(vec![false, true, false, true, false]);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[1]);

        assert_eq!(compact.compact_shape, vec![2, 1]);
        assert_eq!(compact.data, vec![2.0, 4.0]);
    }

    #[test]
    fn test_compact_data_empty_inner_dims_product() {
        // inner_dims = [1] → product = 1, batch with single active
        let source: Vec<f32> = vec![42.0, 99.0, 7.0];
        let mask = RequestActiveMask::new(vec![false, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact = CompactData::compact(&source, &index, &[1]);

        assert_eq!(compact.data, vec![7.0]);
    }

    #[test]
    fn test_compact_data_clone() {
        // batch=3, inner=2 → source has 6 elements
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);
        let compact = CompactData::compact(&source, &index, &[2]);

        let cloned = compact.clone();
        assert_eq!(cloned.data, compact.data);
        assert_eq!(cloned.compact_shape, compact.compact_shape);
    }

    // ── ScatterWriter edge cases ───────────────────────────────────────

    #[test]
    fn test_scatter_multi_inner_dims() {
        // batch=3, inner=[2, 2] → 4 elements per request
        // request 0 and 2 active
        let mut output = vec![0u8; 12];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact_result: Vec<u8> = vec![11, 12, 13, 14, 21, 22, 23, 24];

        let mut writer = ScatterWriter::new(&mut output, &index, &[2, 2]);
        writer.scatter(&compact_result);

        assert_eq!(&output[0..4], &[11, 12, 13, 14]); // req 0
        assert_eq!(&output[4..8], &[0, 0, 0, 0]);     // req 1 untouched
        assert_eq!(&output[8..12], &[21, 22, 23, 24]); // req 2
    }

    #[test]
    fn test_scatter_preserves_inactive_slots() {
        let mut output = vec![99.0f32; 6]; // pre-filled with non-zero
        let mask = RequestActiveMask::new(vec![false, true, false]);
        let index = CompactIndex::from_mask(&mask);

        let compact_result = vec![42.0, 43.0]; // only req 1 data

        let mut writer = ScatterWriter::new(&mut output, &index, &[2]);
        writer.scatter(&compact_result);

        assert_eq!(output[0], 99.0); // req 0: pre-fill preserved
        assert_eq!(output[1], 99.0);
        assert_eq!(output[2], 42.0); // req 1: scattered
        assert_eq!(output[3], 43.0);
        assert_eq!(output[4], 99.0); // req 2: pre-fill preserved
        assert_eq!(output[5], 99.0);
    }

    #[test]
    fn test_scatter_single_element_per_request() {
        let mut output = vec![0.0f32; 4];
        let mask = RequestActiveMask::new(vec![true, false, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let compact_result = vec![100.0, 400.0];

        let mut writer = ScatterWriter::new(&mut output, &index, &[1]);
        writer.scatter(&compact_result);

        assert_eq!(output[0], 100.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 400.0);
    }

    // ── CompactDecision edge cases ─────────────────────────────────────

    #[test]
    fn test_decision_all_active_skip() {
        // All active → waste = 0% → skip regardless of platform
        let mask = RequestActiveMask::all_active(8);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx2);

        assert!(!decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 8);
    }

    #[test]
    fn test_decision_zero_active_skip() {
        // No active → nothing to execute, skip with batch_size=0
        let mask = RequestActiveMask::new(vec![false, false, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx512);

        assert!(!decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 0);
    }

    #[test]
    fn test_decision_compact_variants_have_platform() {
        let mask = RequestActiveMask::new(vec![true, false, false, false]); // 75% waste
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx2);

        if let CompactDecision::Compact { platform, .. } = &decision {
            assert_eq!(*platform, CompactPlatform::X86Avx2);
        } else {
            panic!("expected Compact variant");
        }
    }

    #[test]
    fn test_decision_sve_always_skip_even_high_waste() {
        // SVE skips compact even with 99% waste
        let mask = RequestActiveMask::new(vec![true, false, false, false, false, false, false, false, false, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::Aarch64Sve { vl_bytes: 32 });

        assert!(!decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 10);
    }

    #[test]
    fn test_decision_neon_compact_when_waste_high() {
        // NEON platform does not have special treatment, follows waste threshold
        let mask = RequestActiveMask::new(vec![true, false, false, false]); // 75% waste
        let decision = CompactDecision::decide(mask, CompactPlatform::Aarch64Neon);

        assert!(decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 1);
    }

    #[test]
    fn test_decision_metal_compact_when_waste_high() {
        let mask = RequestActiveMask::new(vec![true, false, false, false, false]); // 80% waste
        let decision = CompactDecision::decide(mask, CompactPlatform::GpuMetal);

        assert!(decision.needs_compact());
        assert_eq!(decision.effective_batch_size(), 1);
    }

    #[test]
    fn test_decision_cuda_compact_with_index() {
        let mask = RequestActiveMask::new(vec![false, true, false, true]); // 50% waste
        let decision = CompactDecision::decide(mask, CompactPlatform::GpuCuda { warp_size: 32 });

        assert!(decision.needs_compact());
        if let CompactDecision::Compact { index, original_batch_size, compact_batch_size, .. } = &decision {
            assert_eq!(*original_batch_size, 4);
            assert_eq!(*compact_batch_size, 2);
            assert_eq!(index.to_compact(1), 0);
            assert_eq!(index.to_compact(3), 1);
        }
    }

    #[test]
    fn test_decision_hip_skip_low_waste() {
        // 7/8 active = 12.5% waste < 25%
        let mask = RequestActiveMask::new(vec![true, true, true, true, true, true, true, false]);
        let decision = CompactDecision::decide(mask, CompactPlatform::GpuHip { wavefront_size: 64 });

        assert!(!decision.needs_compact());
    }

    // ── RaggedCompaction accessors and should_compact ──────────────────

    #[test]
    fn test_ragged_platform_accessor() {
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx512);
        assert_eq!(*rc.platform(), CompactPlatform::X86Avx512);
    }

    #[test]
    fn test_ragged_should_compact_sve_false() {
        let rc = RaggedCompaction::new(CompactPlatform::Aarch64Sve { vl_bytes: 32 });
        let mask = RequestActiveMask::new(vec![true, false]); // 50% waste
        assert!(!rc.should_compact(&mask)); // SVE always false
    }

    #[test]
    fn test_ragged_should_compact_high_waste() {
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let mask = RequestActiveMask::new(vec![true, false, false, false]); // 75% waste
        assert!(rc.should_compact(&mask));
    }

    #[test]
    fn test_ragged_should_compact_low_waste() {
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx512);
        let mask = RequestActiveMask::all_active(4); // 0% waste
        assert!(!rc.should_compact(&mask));
    }

    #[test]
    fn test_ragged_execute_skip_direct_path() {
        // All active → skip compact, execute directly
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mask = RequestActiveMask::all_active(2);
        let mut output = vec![0.0f32; 4];

        rc.execute(&input, &[2], mask, &mut output, |data, shape| {
            assert_eq!(shape, &[2, 2]);
            data.iter().map(|&x| x + 10.0).collect()
        });

        assert_eq!(output, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_ragged_execute_zero_active_noop() {
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mask = RequestActiveMask::new(vec![false, false]);
        let mut output = vec![0.0f32; 4];

        rc.execute(&input, &[2], mask, &mut output, |_data, _shape| {
            panic!("executor should not be called with 0 active");
        });

        // output unchanged
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    // ── Full roundtrip: Compact → Execute → Scatter ────────────────────

    #[test]
    fn test_full_roundtrip_identity_executor() {
        // Executor returns input unchanged → scatter should reconstruct original for active slots
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx512);

        let input: Vec<f32> = vec![
            1.0, 2.0,  // req 0 (active)
            3.0, 4.0,  // req 1 (inactive)
            5.0, 6.0,  // req 2 (active)
        ];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let mut output = vec![0.0f32; 6];

        rc.execute(&input, &[2], mask, &mut output, |data, shape| {
            assert_eq!(shape, &[2, 2]);
            data.to_vec() // identity
        });

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 0.0); // untouched
        assert_eq!(output[3], 0.0); // untouched
        assert_eq!(output[4], 5.0);
        assert_eq!(output[5], 6.0);
    }

    #[test]
    fn test_full_roundtrip_transform_executor() {
        let rc = RaggedCompaction::new(CompactPlatform::Aarch64Neon);

        let input: Vec<i32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        // 4 requests × 2 inner, request 1 and 2 inactive
        let mask = RequestActiveMask::new(vec![true, false, false, true]);
        let mut output = vec![0i32; 8];

        rc.execute(&input, &[2], mask, &mut output, |data, shape| {
            assert_eq!(shape, &[2, 2]);
            data.iter().map(|&x| x * -1).collect()
        });

        assert_eq!(output[0], -10);
        assert_eq!(output[1], -20);
        assert_eq!(output[2], 0);  // untouched
        assert_eq!(output[3], 0);  // untouched
        assert_eq!(output[4], 0);  // untouched
        assert_eq!(output[5], 0);  // untouched
        assert_eq!(output[6], -70);
        assert_eq!(output[7], -80);
    }

    #[test]
    fn test_full_roundtrip_large_batch() {
        // 16 requests, 8 active (every other one)
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);

        let input: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let active_mask: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
        let mask = RequestActiveMask::new(active_mask);
        let mut output = vec![0.0f32; 16];

        rc.execute(&input, &[1], mask, &mut output, |data, shape| {
            assert_eq!(shape, &[8, 1]);
            data.iter().map(|&x| x + 100.0).collect()
        });

        for i in 0..16 {
            if i % 2 == 0 {
                assert_eq!(output[i], i as f32 + 100.0, "active slot {}", i);
            } else {
                assert_eq!(output[i], 0.0, "inactive slot {}", i);
            }
        }
    }

    // ── CompactData Debug trait ────────────────────────────────────────

    #[test]
    fn test_compact_data_debug_format() {
        // batch=3, inner=2 → source has 6 elements
        let source: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);
        let compact = CompactData::compact(&source, &index, &[2]);

        let debug_str = format!("{:?}", compact);
        assert!(debug_str.contains("CompactData"));
    }

    // ── CompactIndex Debug trait ───────────────────────────────────────

    #[test]
    fn test_compact_index_debug_format() {
        let mask = RequestActiveMask::new(vec![true, false, true]);
        let index = CompactIndex::from_mask(&mask);

        let debug_str = format!("{:?}", index);
        assert!(debug_str.contains("CompactIndex"));
    }

    // ── CompactPlatform Debug trait ────────────────────────────────────

    #[test]
    fn test_platform_debug_format() {
        let platform = CompactPlatform::GpuCuda { warp_size: 32 };
        let debug_str = format!("{:?}", platform);
        assert!(debug_str.contains("GpuCuda"));
    }

    // ── CompactDecision Debug trait ────────────────────────────────────

    #[test]
    fn test_decision_debug_format() {
        let decision = CompactDecision::SkipDirect { batch_size: 8 };
        let debug_str = format!("{:?}", decision);
        assert!(debug_str.contains("SkipDirect"));
    }

    // ── RequestActiveMask Debug trait ──────────────────────────────────

    #[test]
    fn test_mask_debug_format() {
        let mask = RequestActiveMask::new(vec![true, false]);
        let debug_str = format!("{:?}", mask);
        assert!(debug_str.contains("RequestActiveMask"));
    }

    // ── New tests: boundary, edge cases, and additional coverage ───────

    #[test]
    fn test_display_cuda_warp_size_1() {
        // Arrange: CUDA platform with minimal warp_size = 1
        let platform = CompactPlatform::GpuCuda { warp_size: 1 };
        // Act
        let display = format!("{}", platform);
        // Assert
        assert_eq!(display, "cuda_w1");
    }

    #[test]
    fn test_display_hip_wavefront_size_1() {
        // Arrange: HIP platform with minimal wavefront_size = 1
        let platform = CompactPlatform::GpuHip { wavefront_size: 1 };
        // Act
        let display = format!("{}", platform);
        // Assert
        assert_eq!(display, "hip_wf1");
    }

    #[test]
    fn test_simd_width_cuda_warp_size_1() {
        // Arrange: CUDA with warp_size=1 → 1 * sizeof(f32) = 4 bytes
        let platform = CompactPlatform::GpuCuda { warp_size: 1 };
        // Act
        let width = platform.simd_width_bytes();
        // Assert
        assert_eq!(width, 4);
    }

    #[test]
    fn test_simd_width_hip_wavefront_size_1() {
        // Arrange: HIP with wavefront_size=1 → 1 * sizeof(f32) = 4 bytes
        let platform = CompactPlatform::GpuHip { wavefront_size: 1 };
        // Act
        let width = platform.simd_width_bytes();
        // Assert
        assert_eq!(width, 4);
    }

    #[test]
    fn test_compact_data_i32_type() {
        // Arrange: i32 data with 4 requests, inner_dim=3, request 2 inactive
        let source: Vec<i32> = vec![
            100, 200, 300,   // req 0
            -10, -20, -30,   // req 1 (inactive)
            999, 888, 777,   // req 2
            0, 0, 0,         // req 3
        ];
        let mask = RequestActiveMask::new(vec![true, false, true, true]);
        let index = CompactIndex::from_mask(&mask);
        // Act
        let compact = CompactData::compact(&source, &index, &[3]);
        // Assert
        assert_eq!(compact.compact_shape, vec![3, 3]);
        assert_eq!(compact.data, vec![100, 200, 300, 999, 888, 777, 0, 0, 0]);
    }

    #[test]
    fn test_compact_data_preserves_order() {
        // Arrange: 6 requests, every other active, verify order is preserved
        let source: Vec<f32> = (0..6).map(|i| (i * 10) as f32).collect();
        let mask = RequestActiveMask::new(vec![false, true, false, true, false, true]);
        let index = CompactIndex::from_mask(&mask);
        // Act
        let compact = CompactData::compact(&source, &index, &[1]);
        // Assert: compact order must be [req1, req3, req5] = [10.0, 30.0, 50.0]
        assert_eq!(compact.data, vec![10.0, 30.0, 50.0]);
        assert_eq!(compact.index.compact_to_original()[..], [1, 3, 5]);
    }

    #[test]
    fn test_compact_data_empty_inner_dims_slice() {
        // Arrange: inner_dims = &[] → product = 1 (empty iterator product is 1)
        let source: Vec<f32> = vec![5.0, 6.0, 7.0];
        let mask = RequestActiveMask::new(vec![false, true, false]);
        let index = CompactIndex::from_mask(&mask);
        // Act
        let compact = CompactData::compact(&source, &index, &[]);
        // Assert: empty inner_dims product is 1, compact_shape = [1]
        assert_eq!(compact.compact_shape, vec![1]);
        assert_eq!(compact.data, vec![6.0]);
    }

    #[test]
    fn test_scatter_overwrites_pre_filled_target() {
        // Arrange: target pre-filled with sentinel, scatter should overwrite active slots
        let mut target = vec![255u8; 8];
        let mask = RequestActiveMask::new(vec![true, false, false, true]);
        let index = CompactIndex::from_mask(&mask);
        let compact_result: Vec<u8> = vec![10, 20, 70, 80];
        // Act
        let mut writer = ScatterWriter::new(&mut target, &index, &[2]);
        writer.scatter(&compact_result);
        // Assert: req 0 and req 3 overwritten, req 1 and req 2 still 255
        assert_eq!(target[0], 10);
        assert_eq!(target[1], 20);
        assert_eq!(target[2], 255);
        assert_eq!(target[3], 255);
        assert_eq!(target[4], 255);
        assert_eq!(target[5], 255);
        assert_eq!(target[6], 70);
        assert_eq!(target[7], 80);
    }

    #[test]
    fn test_scatter_all_active_no_gaps() {
        // Arrange: all requests active → scatter writes to every position
        let mut output = vec![0u8; 6];
        let mask = RequestActiveMask::all_active(3);
        let index = CompactIndex::from_mask(&mask);
        let compact_result: Vec<u8> = vec![11, 22, 33, 44, 55, 66];
        // Act
        let mut writer = ScatterWriter::new(&mut output, &index, &[2]);
        writer.scatter(&compact_result);
        // Assert
        assert_eq!(output, vec![11, 22, 33, 44, 55, 66]);
    }

    #[test]
    fn test_index_first_element_inactive_mapping() {
        // Arrange: first element inactive, verify correct compact positions
        let mask = RequestActiveMask::new(vec![false, true, true, true]);
        let index = CompactIndex::from_mask(&mask);
        // Act & Assert
        assert_eq!(index.to_compact(0), usize::MAX); // inactive
        assert_eq!(index.to_compact(1), 0);
        assert_eq!(index.to_compact(2), 1);
        assert_eq!(index.to_compact(3), 2);
        assert_eq!(index.to_original(0), 1);
        assert_eq!(index.to_original(1), 2);
        assert_eq!(index.to_original(2), 3);
        assert_eq!(index.active_count(), 3);
    }

    #[test]
    fn test_index_last_element_only_active() {
        // Arrange: only last element active in a batch of 5
        let mask = RequestActiveMask::new(vec![false, false, false, false, true]);
        let index = CompactIndex::from_mask(&mask);
        // Act & Assert
        assert_eq!(index.active_count(), 1);
        assert_eq!(index.to_compact(4), 0);
        assert_eq!(index.to_original(0), 4);
        for i in 0..4 {
            assert_eq!(index.to_compact(i), usize::MAX);
        }
    }

    #[test]
    fn test_decision_waste_exactly_at_threshold_batch4() {
        // Arrange: 3/4 active = 25% waste, exactly at threshold → should NOT compact
        let mask = RequestActiveMask::new(vec![true, true, true, false]);
        // Act
        let decision = CompactDecision::decide(mask, CompactPlatform::X86Avx2);
        // Assert: strict > means exactly 25% does NOT trigger
        assert!(!decision.needs_compact());
    }

    #[test]
    fn test_mask_large_batch_alternating() {
        // Arrange: large batch with alternating active/inactive
        let raw: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let mask = RequestActiveMask::new(raw);
        // Act & Assert
        assert_eq!(mask.batch_size(), 100);
        assert_eq!(mask.active_count(), 50);
        let waste = mask.waste_ratio();
        assert!((waste - 0.5).abs() < 0.001, "waste should be 0.5, got {}", waste);
        assert!(mask.should_compact()); // 50% > 25%
    }

    #[test]
    fn test_ragged_execute_compact_path_produces_correct_shape() {
        // Arrange: 6 requests, 2 active (indices 0 and 5), inner_dim=3
        let rc = RaggedCompaction::new(CompactPlatform::X86Avx2);
        let input: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let mask = RequestActiveMask::new(vec![true, false, false, false, false, true]);
        let mut output = vec![0.0f32; 18];
        // Act
        rc.execute(&input, &[3], mask, &mut output, |data, shape| {
            // Assert inside executor: shape must be [2, 3] (compact batch = 2)
            assert_eq!(shape, &[2, 3]);
            // data must be req 0 (0,1,2) and req 5 (15,16,17)
            assert_eq!(data.len(), 6);
            assert_eq!(data[0], 0.0);
            assert_eq!(data[1], 1.0);
            assert_eq!(data[2], 2.0);
            assert_eq!(data[3], 15.0);
            assert_eq!(data[4], 16.0);
            assert_eq!(data[5], 17.0);
            data.iter().map(|&x| x * 3.0).collect()
        });
        // Assert: output[0..3] scattered, output[15..18] scattered, rest untouched
        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 3.0);
        assert_eq!(output[2], 6.0);
        assert_eq!(output[3], 0.0);  // req 1 untouched
        assert_eq!(output[15], 45.0);
        assert_eq!(output[16], 48.0);
        assert_eq!(output[17], 51.0);
    }

    #[test]
    fn test_compact_data_with_only_first_active() {
        // Arrange: only the first request is active
        let source: Vec<f32> = vec![42.0, 43.0, 1.0, 2.0, 5.0, 6.0];
        let mask = RequestActiveMask::new(vec![true, false, false]);
        let index = CompactIndex::from_mask(&mask);
        // Act
        let compact = CompactData::compact(&source, &index, &[2]);
        // Assert
        assert_eq!(compact.compact_shape, vec![1, 2]);
        assert_eq!(compact.data, vec![42.0, 43.0]);
    }

    #[test]
    fn test_ragged_should_compact_all_platforms_consistency() {
        // Arrange: mask with 50% waste
        let mask = RequestActiveMask::new(vec![true, false]);
        // Act & Assert: all non-SVE platforms should report should_compact = true
        for platform in [
            CompactPlatform::X86Avx512,
            CompactPlatform::X86Avx2,
            CompactPlatform::Aarch64Neon,
            CompactPlatform::GpuCuda { warp_size: 32 },
            CompactPlatform::GpuHip { wavefront_size: 64 },
            CompactPlatform::GpuMetal,
        ] {
            let rc = RaggedCompaction::new(platform);
            assert!(rc.should_compact(&mask), "platform {:?} should compact with 50% waste", platform);
        }
    }
}
