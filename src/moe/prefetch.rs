//! 专家权重预取 (SPEC §15.2)
//!
//! ## 核心职责
//! 在 Gate 路由计算完成后，立即启动专家权重的异步预取:
//! - 热专家: 权重常驻 GPU L2，零延迟
//! - 温专家: 权重在 CPU RAM，通过 PCIe 预取
//! - 冷专家: 权重经 TurboQuant 4-bit 压缩，通过 RDMA 远程预取
//!
//! ## §15.2 TurboQuant + 预瞄 → Zero-Stall Swapping
//! Gate 层算出路由表的瞬间，立刻启动 cuMemPrefetchAsync 无阻塞预加载。
//! Thread Block 走到该 Expert 汇编入口时，被极度压缩的权重已躺在 GPU L2 Cache 里。

use super::thermal::ExpertHeatLevel;

/// 专家权重存储位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertWeightLocation {
    /// 权重在 GPU L2 Cache 中 (热专家)
    GpuL2,
    /// 权重在 GPU VRAM 中 (温专家)
    GpuVram,
    /// 权重在 CPU RAM 中 (冷专家)
    CpuRam,
    /// 权重在远程节点 (RDMA)
    RemoteNode,
    /// 权重已被封杀 (Deopt 替换)
    Evicted,
}

impl ExpertWeightLocation {
    /// 从热度级别推导默认存储位置
    pub fn from_heat_level(level: ExpertHeatLevel) -> Self {
        match level {
            ExpertHeatLevel::Hot => ExpertWeightLocation::GpuL2,
            ExpertHeatLevel::Warm => ExpertWeightLocation::CpuRam,
            ExpertHeatLevel::Cold => ExpertWeightLocation::CpuRam,
            ExpertHeatLevel::Evicted => ExpertWeightLocation::Evicted,
        }
    }

    /// 预取延迟估算 (μs)
    pub fn estimated_latency_us(&self) -> f32 {
        match self {
            ExpertWeightLocation::GpuL2 => 0.0,
            ExpertWeightLocation::GpuVram => 5.0,
            ExpertWeightLocation::CpuRam => 50.0,
            ExpertWeightLocation::RemoteNode => 200.0,
            ExpertWeightLocation::Evicted => f32::INFINITY,
        }
    }
}

/// 专家权重布局描述
#[derive(Debug, Clone)]
pub struct ExpertWeightLayout {
    /// 专家索引
    pub expert_idx: usize,
    /// 权重字节数 (原始精度)
    pub weight_bytes: usize,
    /// 权重字节数 (TurboQuant 压缩后)
    pub compressed_bytes: usize,
    /// 压缩比
    pub compression_ratio: f32,
    /// 当前存储位置
    pub location: ExpertWeightLocation,
}

/// 单个专家的预取请求
#[derive(Debug, Clone)]
pub struct ExpertPrefetchRequest {
    /// 专家索引
    pub expert_idx: usize,
    /// 权重存储位置
    pub source: ExpertWeightLocation,
    /// 预取目标位置
    pub destination: ExpertWeightLocation,
    /// 预取字节数 (压缩后)
    pub bytes: usize,
    /// 估算预取延迟 (μs)
    pub estimated_latency_us: f32,
    /// 优先级 (0 = 最高)
    pub priority: u32,
}

/// 专家权重预取调度器
///
/// §15.2: 在 Gate 计算完成后，根据路由表和专家热度状态，
/// 生成最优的预取调度。
pub struct ExpertWeightPrefetcher {
    /// 专家数量
    num_experts: usize,
    /// 专家权重布局
    weight_layouts: Vec<ExpertWeightLayout>,
    /// PCIe 带宽 (GB/s)
    pcie_bandwidth_gbs: f32,
    /// RDMA 带宽 (GB/s)
    rdma_bandwidth_gbs: f32,
    /// GPU 单层计算时间 (μs)
    layer_compute_time_us: f32,
}

impl ExpertWeightPrefetcher {
    /// 创建新的专家权重预取调度器
    pub fn new(num_experts: usize, weight_bytes_per_expert: usize) -> Self {
        // 假设 TurboQuant 4-bit 压缩 → 压缩比 ≈ 8x (FP32→4bit)
        let compressed_bytes = weight_bytes_per_expert / 8;
        let compression_ratio = weight_bytes_per_expert as f32 / compressed_bytes as f32;

        let weight_layouts: Vec<ExpertWeightLayout> = (0..num_experts)
            .map(|idx| ExpertWeightLayout {
                expert_idx: idx,
                weight_bytes: weight_bytes_per_expert,
                compressed_bytes,
                compression_ratio,
                location: ExpertWeightLocation::CpuRam, // 默认在 CPU
            })
            .collect();

        Self {
            num_experts,
            weight_layouts,
            pcie_bandwidth_gbs: 32.0,   // PCIe 4.0 x16
            rdma_bandwidth_gbs: 100.0,  // 100 Gbps RDMA
            layer_compute_time_us: 100.0, // ~100μs per layer
        }
    }

    /// 配置带宽参数
    pub fn with_bandwidth(mut self, pcie_gbs: f32, rdma_gbs: f32) -> Self {
        self.pcie_bandwidth_gbs = pcie_gbs;
        self.rdma_bandwidth_gbs = rdma_gbs;
        self
    }

    /// 配置 GPU 层计算时间
    pub fn with_layer_compute_time(mut self, us: f32) -> Self {
        self.layer_compute_time_us = us;
        self
    }

    /// 更新专家权重位置
    pub fn update_location(&mut self, expert_idx: usize, location: ExpertWeightLocation) {
        if let Some(layout) = self.weight_layouts.get_mut(expert_idx) {
            layout.location = location;
        }
    }

    /// 根据路由表和热度状态生成预取调度
    ///
    /// §15.2: Gate 算完路由表后立即调用此方法。
    /// 生成最优预取请求列表，按优先级排序。
    pub fn schedule_prefetch(
        &self,
        routed_experts: &[usize],
        heat_levels: &[ExpertHeatLevel],
    ) -> Vec<ExpertPrefetchRequest> {
        let mut requests = Vec::new();

        for (priority, &expert_idx) in routed_experts.iter().enumerate() {
            if expert_idx >= self.num_experts {
                continue;
            }

            let layout = &self.weight_layouts[expert_idx];
            let heat = heat_levels.get(expert_idx).copied().unwrap_or(ExpertHeatLevel::Warm);

            // 热专家: 权重已在 GPU L2，无需预取
            if heat == ExpertHeatLevel::Hot {
                continue;
            }

            // 封杀专家: 不预取
            if heat == ExpertHeatLevel::Evicted {
                continue;
            }

            let source = layout.location;
            let destination = ExpertWeightLocation::GpuVram;

            // 估算传输时间
            let transfer_time_us = self.estimate_transfer_time(
                layout.compressed_bytes,
                source,
            );

            requests.push(ExpertPrefetchRequest {
                expert_idx,
                source,
                destination,
                bytes: layout.compressed_bytes,
                estimated_latency_us: transfer_time_us,
                priority: priority as u32,
            });
        }

        // 按优先级排序 (低优先级数字 = 高优先级)
        requests.sort_by_key(|r| r.priority);

        requests
    }

    /// 检查预取是否能被计算流水线掩盖
    ///
    /// §15.2: "冷专家卡顿被计算流水线完美掩盖 (Pipelining)"
    /// 如果预取延迟 < 层计算时间，则可以完全掩盖。
    pub fn can_pipeline_hide(
        &self,
        requests: &[ExpertPrefetchRequest],
        layers_until_needed: usize,
    ) -> Vec<(usize, bool)> {
        let available_time_us = layers_until_needed as f32 * self.layer_compute_time_us;

        requests
            .iter()
            .map(|req| {
                (req.expert_idx, req.estimated_latency_us <= available_time_us)
            })
            .collect()
    }

    /// 估算传输时间 (μs)
    fn estimate_transfer_time(&self, bytes: usize, source: ExpertWeightLocation) -> f32 {
        let bytes_gb = bytes as f32 / 1e9;

        match source {
            ExpertWeightLocation::GpuL2 => 0.0,
            ExpertWeightLocation::GpuVram => {
                // GPU 内部 HBM → L2: 几乎瞬时
                bytes_gb / 1000.0 * 1e6 // ~1 TB/s HBM
            }
            ExpertWeightLocation::CpuRam => {
                // PCIe 传输
                bytes_gb / self.pcie_bandwidth_gbs * 1e6
            }
            ExpertWeightLocation::RemoteNode => {
                // RDMA 传输
                bytes_gb / self.rdma_bandwidth_gbs * 1e6
            }
            ExpertWeightLocation::Evicted => f32::INFINITY,
        }
    }

    /// 获取专家权重布局
    pub fn layout(&self, expert_idx: usize) -> Option<&ExpertWeightLayout> {
        self.weight_layouts.get(expert_idx)
    }

    /// 获取所有权重布局
    pub fn layouts(&self) -> &[ExpertWeightLayout] {
        &self.weight_layouts
    }

    /// 计算总 GPU VRAM 占用 (仅热/温专家)
    pub fn total_gpu_vram_bytes(&self) -> usize {
        self.weight_layouts
            .iter()
            .filter(|l| matches!(l.location, ExpertWeightLocation::GpuL2 | ExpertWeightLocation::GpuVram))
            .map(|l| l.compressed_bytes)
            .sum()
    }

    /// 计算预取节省的带宽 (通过 TurboQuant 压缩)
    pub fn bandwidth_savings_ratio(&self) -> f32 {
        let total_original: usize = self.weight_layouts.iter().map(|l| l.weight_bytes).sum();
        let total_compressed: usize = self.weight_layouts.iter().map(|l| l.compressed_bytes).sum();

        if total_original > 0 {
            1.0 - (total_compressed as f32 / total_original as f32)
        } else {
            0.0
        }
    }

    /// 获取专家数量
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_weight_location_latency() {
        assert_eq!(ExpertWeightLocation::GpuL2.estimated_latency_us(), 0.0);
        assert!(ExpertWeightLocation::GpuVram.estimated_latency_us() < ExpertWeightLocation::CpuRam.estimated_latency_us());
        assert!(ExpertWeightLocation::CpuRam.estimated_latency_us() < ExpertWeightLocation::RemoteNode.estimated_latency_us());
        assert!(ExpertWeightLocation::Evicted.estimated_latency_us().is_infinite());
    }

    #[test]
    fn test_expert_weight_prefetcher_creation() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 1024 * 1024); // 8 experts, 1MB each
        assert_eq!(prefetcher.num_experts(), 8);

        // Check compression: 1MB / 8 = 128KB per expert
        let layout = prefetcher.layout(0).unwrap();
        assert_eq!(layout.weight_bytes, 1024 * 1024);
        assert_eq!(layout.compressed_bytes, 1024 * 1024 / 8);
        assert!((layout.compression_ratio - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_schedule_prefetch_hot_experts_skip() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024 * 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2); // Hot
        prefetcher.update_location(1, ExpertWeightLocation::GpuL2); // Hot

        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
        ];

        // Route to experts 0, 1, 2 (hot, hot, warm)
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2], &heat_levels);

        // Only expert 2 should need prefetch
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 2);
        assert_eq!(requests[0].source, ExpertWeightLocation::CpuRam);
    }

    #[test]
    fn test_schedule_prefetch_evicted_experts_skip() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024 * 1024);

        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Evicted, // Evicted
            ExpertHeatLevel::Warm,
        ];

        let requests = prefetcher.schedule_prefetch(&[0, 2, 3], &heat_levels);

        // Expert 2 (evicted) should be skipped, expert 3 should be prefetched
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 3);
    }

    #[test]
    fn test_pipeline_hiding() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024 * 1024)
            .with_layer_compute_time(100.0);

        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
        ];

        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);

        // 10 layers until needed = 1000μs available
        let hiding = prefetcher.can_pipeline_hide(&requests, 10);
        for (expert_idx, can_hide) in &hiding {
            // Small weights should be easily hidden
            assert!(*can_hide || *expert_idx >= 100);
        }
    }

    #[test]
    fn test_bandwidth_savings() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 1024 * 1024);
        let savings = prefetcher.bandwidth_savings_ratio();
        // TurboQuant 4-bit compression: 87.5% savings
        assert!(savings > 0.8 && savings < 0.9);
    }

    #[test]
    fn test_total_gpu_vram() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024 * 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        // Experts 2,3 in CpuRam

        let vram = prefetcher.total_gpu_vram_bytes();
        // 2 experts on GPU, each compressed to 128KB
        assert_eq!(vram, 2 * 1024 * 1024 / 8);
    }
}
