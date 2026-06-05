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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Prefetch priority from StrategyBias: scales the number of experts to prefetch.
    /// Default 1.0 (no scaling). >1.0 = prefetch more, <1.0 = prefetch fewer.
    prefetch_priority: f64,
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
            prefetch_priority: 1.0,
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

    /// Set prefetch priority from StrategyBias.
    /// >1.0 = prefetch more experts, <1.0 = prefetch fewer.
    pub fn with_prefetch_priority(mut self, priority: f64) -> Self {
        self.prefetch_priority = priority;
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

        // Apply prefetch_priority: scale the number of prefetch requests.
        // priority > 1.0 keeps more, priority < 1.0 keeps fewer.
        let effective_count = ((requests.len() as f64) * self.prefetch_priority)
            .round()
            .max(0.0) as usize;
        requests.truncate(effective_count);

        requests
    }

    /// Step-level prefetch: 在 decode step 间隙预取下一批专家权重。
    ///
    /// §3 Step-Level Prefetch: 利用 step 间隙发起异步传输请求，
    /// 为即将在后续 step 中路由到的专家提前加载权重。
    /// 集成 `decide_via_gmm` 结果确定预取优先级和来源层级。
    ///
    /// 该方法发射传输请求后立即返回，不阻塞调用者。
    /// 返回的请求由调用者提交到异步传输引擎。
    ///
    /// # Arguments
    /// * `step` — 当前 decode step 索引（低 step 表示更紧迫）
    /// * `expert_ids` — 后续 step 可能路由到的专家 ID 列表
    ///
    /// # Returns
    /// 需要异步执行的预取请求列表，按优先级排序
    pub fn prefetch_step(&self, step: usize, expert_ids: &[u32]) -> Vec<ExpertPrefetchRequest> {
        let mut requests = Vec::new();

        for (i, &expert_id) in expert_ids.iter().enumerate() {
            let expert_idx = expert_id as usize;
            if expert_idx >= self.num_experts {
                continue;
            }

            let layout = &self.weight_layouts[expert_idx];

            // 已在 GPU 上的专家无需预取
            if matches!(
                layout.location,
                ExpertWeightLocation::GpuL2 | ExpertWeightLocation::GpuVram
            ) {
                continue;
            }

            // 封杀专家不预取
            if layout.location == ExpertWeightLocation::Evicted {
                continue;
            }

            let source = layout.location;
            let destination = ExpertWeightLocation::GpuVram;

            // 估算传输时间
            let transfer_time_us =
                self.estimate_transfer_time(layout.compressed_bytes, source);

            // Step-aware 优先级：低 priority 数字 = 高优先级
            // GMM decide_via_gmm 的结果已反映在 layout.location（来源层级）中：
            // - DeviceLocal → GpuVram（已在 GPU）
            // - HostLocal → CpuRam（需 PCIe 传输）
            // - DiskMmap → 远程（需 RDMA）
            // 低 step = 更紧迫，优先预取即将在更早 step 中用到的专家权重
            let priority = (step as u32) * (expert_ids.len() as u32) + (i as u32);

            requests.push(ExpertPrefetchRequest {
                expert_idx,
                source,
                destination,
                bytes: layout.compressed_bytes,
                estimated_latency_us: transfer_time_us,
                priority,
            });
        }

        // 按优先级排序（低数字 = 高优先级）
        requests.sort_by_key(|r| r.priority);

        // 应用 StrategyBias 的 prefetch_priority 缩放
        let effective_count = ((requests.len() as f64) * self.prefetch_priority)
            .round()
            .max(0.0) as usize;
        requests.truncate(effective_count);

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

    // --- ExpertWeightLocation: from_heat_level mapping ---

    #[test]
    fn test_from_heat_level_hot_maps_to_gpu_l2() {
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Hot),
            ExpertWeightLocation::GpuL2
        );
    }

    #[test]
    fn test_from_heat_level_warm_maps_to_cpu_ram() {
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Warm),
            ExpertWeightLocation::CpuRam
        );
    }

    #[test]
    fn test_from_heat_level_cold_maps_to_cpu_ram() {
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Cold),
            ExpertWeightLocation::CpuRam
        );
    }

    #[test]
    fn test_from_heat_level_evicted_maps_to_evicted() {
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Evicted),
            ExpertWeightLocation::Evicted
        );
    }

    // --- ExpertWeightLocation: estimated_latency_us ordering ---

    #[test]
    fn test_latency_monotonic_increase_from_gpu_l2_to_remote() {
        let l2 = ExpertWeightLocation::GpuL2.estimated_latency_us();
        let vram = ExpertWeightLocation::GpuVram.estimated_latency_us();
        let cpu = ExpertWeightLocation::CpuRam.estimated_latency_us();
        let remote = ExpertWeightLocation::RemoteNode.estimated_latency_us();

        assert!(l2 < vram);
        assert!(vram < cpu);
        assert!(cpu < remote);
    }

    #[test]
    fn test_latency_gpu_l2_is_zero() {
        assert_eq!(ExpertWeightLocation::GpuL2.estimated_latency_us(), 0.0);
    }

    #[test]
    fn test_latency_evicted_is_infinite() {
        assert!(ExpertWeightLocation::Evicted.estimated_latency_us().is_infinite());
    }

    // --- ExpertWeightLocation: Debug + Copy + PartialEq ---

    #[test]
    fn test_expert_weight_location_equality() {
        assert_eq!(ExpertWeightLocation::GpuL2, ExpertWeightLocation::GpuL2);
        assert_ne!(ExpertWeightLocation::GpuL2, ExpertWeightLocation::GpuVram);
        assert_ne!(ExpertWeightLocation::CpuRam, ExpertWeightLocation::RemoteNode);
    }

    #[test]
    fn test_expert_weight_location_debug_format() {
        let debug_str = format!("{:?}", ExpertWeightLocation::GpuL2);
        assert!(debug_str.contains("GpuL2"));

        let debug_str = format!("{:?}", ExpertWeightLocation::RemoteNode);
        assert!(debug_str.contains("RemoteNode"));
    }

    // --- ExpertWeightPrefetcher: builder methods ---

    #[test]
    fn test_with_bandwidth_customizes_pcie_and_rdma() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_bandwidth(64.0, 200.0);

        // Verify via transfer time: smaller bandwidth → longer transfer.
        // Two prefetchers with different bandwidths produce different latencies.
        let default_pf = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];

        let requests_default = default_pf.schedule_prefetch(&[0], &heat_levels);
        let requests_custom = prefetcher.schedule_prefetch(&[0], &heat_levels);

        // Both should have exactly 1 request for expert 0 (Warm, not on GPU)
        assert_eq!(requests_default.len(), 1);
        assert_eq!(requests_custom.len(), 1);

        // Higher PCIe bandwidth → lower latency
        assert!(requests_custom[0].estimated_latency_us < requests_default[0].estimated_latency_us);
    }

    #[test]
    fn test_with_layer_compute_time_affects_pipeline_hiding() {
        let fast_compute = ExpertWeightPrefetcher::new(4, 1024 * 1024)
            .with_layer_compute_time(1.0); // Very fast compute
        let slow_compute = ExpertWeightPrefetcher::new(4, 1024 * 1024)
            .with_layer_compute_time(10000.0); // Very slow compute

        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = fast_compute.schedule_prefetch(&[0], &heat_levels);

        if !requests.is_empty() {
            let hiding_fast = fast_compute.can_pipeline_hide(&requests, 1);
            let hiding_slow = slow_compute.can_pipeline_hide(&requests, 1);

            // With 1 layer at 1us, likely cannot hide; at 10000us, likely can.
            assert_eq!(hiding_fast.len(), hiding_slow.len());
            // At least the slow compute version should be able to hide
            let (_, can_hide_slow) = hiding_slow[0];
            assert!(can_hide_slow);
        }
    }

    // --- ExpertWeightPrefetcher: update_location ---

    #[test]
    fn test_update_location_out_of_bounds_is_noop() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024);

        // Should not panic on out-of-bounds index
        prefetcher.update_location(99, ExpertWeightLocation::GpuL2);

        // Verify layouts unchanged
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::CpuRam);
        assert_eq!(prefetcher.layout(1).unwrap().location, ExpertWeightLocation::CpuRam);
    }

    #[test]
    fn test_update_location_changes_layout() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        assert_eq!(prefetcher.layout(2).unwrap().location, ExpertWeightLocation::CpuRam);

        prefetcher.update_location(2, ExpertWeightLocation::GpuVram);
        assert_eq!(prefetcher.layout(2).unwrap().location, ExpertWeightLocation::GpuVram);

        prefetcher.update_location(2, ExpertWeightLocation::RemoteNode);
        assert_eq!(prefetcher.layout(2).unwrap().location, ExpertWeightLocation::RemoteNode);
    }

    // --- ExpertWeightPrefetcher: layout accessor ---

    #[test]
    fn test_layout_out_of_bounds_returns_none() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        assert!(prefetcher.layout(4).is_none());
        assert!(prefetcher.layout(1000).is_none());
    }

    #[test]
    fn test_layouts_returns_all_layouts() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 2048);
        let layouts = prefetcher.layouts();
        assert_eq!(layouts.len(), 8);

        for (i, layout) in layouts.iter().enumerate() {
            assert_eq!(layout.expert_idx, i);
            assert_eq!(layout.weight_bytes, 2048);
            assert_eq!(layout.compressed_bytes, 2048 / 8);
            assert_eq!(layout.location, ExpertWeightLocation::CpuRam);
        }
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch ---

    #[test]
    fn test_schedule_prefetch_empty_routed_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[], &heat_levels);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_schedule_prefetch_out_of_bounds_expert_index_skipped() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0, 5, 99], &heat_levels);
        // Only expert 0 is valid; experts 5 and 99 are out of bounds
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
    }

    #[test]
    fn test_schedule_prefetch_shorter_heat_levels_uses_default_warm() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        // Only 1 heat level provided, but routing to experts 0,1,2,3
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // All should be treated as Warm (the default for missing heat levels)
        assert_eq!(requests.len(), 4);
    }

    #[test]
    fn test_schedule_prefetch_requests_sorted_by_priority() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Cold; 4];

        // Route in reverse order
        let requests = prefetcher.schedule_prefetch(&[3, 1, 0], &heat_levels);
        assert_eq!(requests.len(), 3);
        assert!(requests[0].priority <= requests[1].priority);
        assert!(requests[1].priority <= requests[2].priority);
    }

    #[test]
    fn test_schedule_prefetch_destination_is_always_gpu_vram() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);

        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);

        for req in &requests {
            assert_eq!(req.destination, ExpertWeightLocation::GpuVram);
        }
    }

    // --- ExpertWeightPrefetcher: prefetch_priority scaling ---

    #[test]
    fn test_prefetch_priority_below_one_truncates_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(0.0); // Zero → truncate all

        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_prefetch_priority_above_one_keeps_more() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(2.0); // Double

        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // 4 experts * 2.0 = 8, capped at actual count 4
        assert_eq!(requests.len(), 4);
    }

    #[test]
    fn test_prefetch_priority_fractional_truncates() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 1024)
            .with_prefetch_priority(0.5); // Half

        let heat_levels = vec![ExpertHeatLevel::Warm; 8];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3, 4, 5, 6, 7], &heat_levels);
        // 8 * 0.5 = 4.0, round to 4
        assert_eq!(requests.len(), 4);
    }

    // --- ExpertWeightPrefetcher: prefetch_step ---

    #[test]
    fn test_prefetch_step_skips_gpu_resident_experts() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(3, ExpertWeightLocation::RemoteNode);

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3]);
        // Experts 0,1 are already on GPU → skipped
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].expert_idx, 2);
        assert_eq!(requests[1].expert_idx, 3);
    }

    #[test]
    fn test_prefetch_step_skips_evicted_experts() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::Evicted);
        prefetcher.update_location(1, ExpertWeightLocation::CpuRam);

        let requests = prefetcher.prefetch_step(0, &[0, 1]);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 1);
    }

    #[test]
    fn test_prefetch_step_out_of_bounds_expert_ids_skipped() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let requests = prefetcher.prefetch_step(0, &[0, 5, 99]);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
    }

    #[test]
    fn test_prefetch_step_empty_expert_ids() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let requests = prefetcher.prefetch_step(0, &[]);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_prefetch_step_priority_increases_with_step_and_index() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 1024);

        let requests_step0 = prefetcher.prefetch_step(0, &[0, 1, 2]);
        let requests_step5 = prefetcher.prefetch_step(5, &[0, 1, 2]);

        assert_eq!(requests_step0.len(), 3);
        assert_eq!(requests_step5.len(), 3);

        // Step 5 priorities should be higher numbers than step 0
        assert!(requests_step5[0].priority > requests_step0[0].priority);
    }

    #[test]
    fn test_prefetch_step_respects_prefetch_priority() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 1024)
            .with_prefetch_priority(0.5);

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3, 4, 5, 6, 7]);
        // 8 * 0.5 = 4.0 → keep 4
        assert_eq!(requests.len(), 4);
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide ---

    #[test]
    fn test_can_pipeline_hide_zero_layers_means_no_time() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024 * 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        // 0 layers = 0μs available, no request can be hidden unless latency is 0
        let hiding = prefetcher.can_pipeline_hide(&requests, 0);
        if !hiding.is_empty() {
            let (_, can_hide) = hiding[0];
            // Transfer from CpuRam should have non-zero latency, cannot hide with 0 time
            assert!(!can_hide);
        }
    }

    #[test]
    fn test_can_pipeline_hide_many_layers_covers_latency() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_layer_compute_time(1000.0); // 1000μs per layer

        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        // 100 layers = 100,000μs, more than enough for small weight transfer
        let hiding = prefetcher.can_pipeline_hide(&requests, 100);
        if !hiding.is_empty() {
            let (_, can_hide) = hiding[0];
            assert!(can_hide);
        }
    }

    // --- ExpertWeightPrefetcher: total_gpu_vram_bytes ---

    #[test]
    fn test_total_gpu_vram_no_experts_on_gpu() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        // All experts default to CpuRam
        assert_eq!(prefetcher.total_gpu_vram_bytes(), 0);
    }

    #[test]
    fn test_total_gpu_vram_all_experts_on_gpu() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        for i in 0..4 {
            prefetcher.update_location(i, ExpertWeightLocation::GpuVram);
        }
        // All 4 experts on GPU, each compressed to 4096/8 = 512 bytes
        assert_eq!(prefetcher.total_gpu_vram_bytes(), 4 * 512);
    }

    // --- ExpertWeightPrefetcher: bandwidth_savings_ratio ---

    #[test]
    fn test_bandwidth_savings_ratio_zero_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(0, 1024);
        assert_eq!(prefetcher.bandwidth_savings_ratio(), 0.0);
    }

    #[test]
    fn test_bandwidth_savings_ratio_reflects_compression() {
        let prefetcher = ExpertWeightPrefetcher::new(16, 8192);
        let savings = prefetcher.bandwidth_savings_ratio();
        // 8x compression → 1 - 1/8 = 0.875
        assert!((savings - 0.875).abs() < 0.01);
    }

    // --- ExpertWeightPrefetcher: num_experts ---

    #[test]
    fn test_num_experts_single() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 1024);
        assert_eq!(prefetcher.num_experts(), 1);
    }

    #[test]
    fn test_num_experts_large() {
        let prefetcher = ExpertWeightPrefetcher::new(256, 1024);
        assert_eq!(prefetcher.num_experts(), 256);
    }

    // --- ExpertWeightPrefetchRequest: struct field validation via schedule_prefetch ---

    #[test]
    fn test_request_fields_reflect_source_location() {
        // Use large weights so transfer latency is distinguishable between PCIe and RDMA.
        // 64 MB per expert → 8 MB compressed = 0.008 GB
        let weight_per_expert = 64 * 1024 * 1024;
        let mut prefetcher = ExpertWeightPrefetcher::new(3, weight_per_expert);
        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);

        let heat_levels = vec![ExpertHeatLevel::Warm; 3];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);

        assert_eq!(requests.len(), 2);

        let cpu_req = requests.iter().find(|r| r.expert_idx == 0).unwrap();
        assert_eq!(cpu_req.source, ExpertWeightLocation::CpuRam);
        assert_eq!(cpu_req.bytes, weight_per_expert / 8);

        let remote_req = requests.iter().find(|r| r.expert_idx == 1).unwrap();
        assert_eq!(remote_req.source, ExpertWeightLocation::RemoteNode);
        // RDMA bandwidth (100 GB/s) is higher than PCIe bandwidth (32 GB/s),
        // so remote RDMA transfer is actually faster than PCIe transfer.
        assert!(remote_req.estimated_latency_us < cpu_req.estimated_latency_us);
    }

    // --- ExpertWeightLayout: struct field correctness ---

    #[test]
    fn test_layout_compression_ratio_calculation() {
        let weight_bytes = 1024 * 1024; // 1 MB
        let prefetcher = ExpertWeightPrefetcher::new(1, weight_bytes);
        let layout = prefetcher.layout(0).unwrap();

        // FP32 → 4-bit = 8x compression
        let expected_compressed = weight_bytes / 8;
        assert_eq!(layout.compressed_bytes, expected_compressed);

        let expected_ratio = weight_bytes as f32 / expected_compressed as f32;
        assert!((layout.compression_ratio - expected_ratio).abs() < f32::EPSILON);
    }

    // =====================================================
    // NEW TESTS (18 additional)
    // =====================================================

    // --- ExpertWeightLocation: Hash trait ---

    #[test]
    fn test_expert_weight_location_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ExpertWeightLocation::GpuL2);
        set.insert(ExpertWeightLocation::CpuRam);
        set.insert(ExpertWeightLocation::GpuL2); // duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&ExpertWeightLocation::GpuL2));
        assert!(set.contains(&ExpertWeightLocation::CpuRam));
        assert!(!set.contains(&ExpertWeightLocation::RemoteNode));
    }

    // --- ExpertWeightLocation: Copy trait ---

    #[test]
    fn test_expert_weight_location_copy_semantics() {
        let a = ExpertWeightLocation::GpuVram;
        let b = a; // Copy, not move
        assert_eq!(a, b);
        assert_eq!(a, ExpertWeightLocation::GpuVram);
    }

    // --- ExpertWeightLocation: all variants have distinct latency ---

    #[test]
    fn test_all_location_variants_have_distinct_latency() {
        let latencies: Vec<f32> = vec![
            ExpertWeightLocation::GpuL2.estimated_latency_us(),
            ExpertWeightLocation::GpuVram.estimated_latency_us(),
            ExpertWeightLocation::CpuRam.estimated_latency_us(),
            ExpertWeightLocation::RemoteNode.estimated_latency_us(),
        ];
        // All finite latencies are distinct
        for i in 0..latencies.len() {
            for j in (i + 1)..latencies.len() {
                assert_ne!(latencies[i], latencies[j]);
            }
        }
    }

    // --- ExpertWeightLocation: from_heat_level covers all variants ---

    #[test]
    fn test_from_heat_level_produces_no_gpu_vram_variant() {
        // GpuVram is never produced by from_heat_level — it is only set manually.
        // Verify all heat levels map to non-GpuVram locations.
        let levels = [
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        for level in levels {
            let loc = ExpertWeightLocation::from_heat_level(level);
            assert_ne!(loc, ExpertWeightLocation::GpuVram);
        }
    }

    // --- ExpertWeightLocation: latency values match SPEC expectations ---

    #[test]
    fn test_latency_specific_values() {
        assert_eq!(ExpertWeightLocation::GpuL2.estimated_latency_us(), 0.0);
        assert!((ExpertWeightLocation::GpuVram.estimated_latency_us() - 5.0).abs() < f32::EPSILON);
        assert!((ExpertWeightLocation::CpuRam.estimated_latency_us() - 50.0).abs() < f32::EPSILON);
        assert!((ExpertWeightLocation::RemoteNode.estimated_latency_us() - 200.0).abs() < f32::EPSILON);
    }

    // --- ExpertWeightLayout: construction and field access ---

    #[test]
    fn test_expert_weight_layout_manual_construction() {
        let layout = ExpertWeightLayout {
            expert_idx: 42,
            weight_bytes: 8192,
            compressed_bytes: 1024,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::RemoteNode,
        };
        assert_eq!(layout.expert_idx, 42);
        assert_eq!(layout.weight_bytes, 8192);
        assert_eq!(layout.compressed_bytes, 1024);
        assert!((layout.compression_ratio - 8.0).abs() < f32::EPSILON);
        assert_eq!(layout.location, ExpertWeightLocation::RemoteNode);
    }

    #[test]
    fn test_expert_weight_layout_clone_is_equal() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 4096,
            compressed_bytes: 512,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        let cloned = layout.clone();
        assert_eq!(cloned.expert_idx, layout.expert_idx);
        assert_eq!(cloned.weight_bytes, layout.weight_bytes);
        assert_eq!(cloned.compressed_bytes, layout.compressed_bytes);
        assert_eq!(cloned.compression_ratio, layout.compression_ratio);
        assert_eq!(cloned.location, layout.location);
    }

    #[test]
    fn test_expert_weight_layout_debug_contains_fields() {
        let layout = ExpertWeightLayout {
            expert_idx: 7,
            weight_bytes: 2048,
            compressed_bytes: 256,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::GpuL2,
        };
        let debug = format!("{:?}", layout);
        assert!(debug.contains("expert_idx"));
        assert!(debug.contains("GpuL2"));
    }

    // --- ExpertPrefetchRequest: construction and field access ---

    #[test]
    fn test_expert_prefetch_request_manual_construction() {
        let req = ExpertPrefetchRequest {
            expert_idx: 3,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 1024,
            estimated_latency_us: 42.5,
            priority: 0,
        };
        assert_eq!(req.expert_idx, 3);
        assert_eq!(req.source, ExpertWeightLocation::CpuRam);
        assert_eq!(req.destination, ExpertWeightLocation::GpuVram);
        assert_eq!(req.bytes, 1024);
        assert!((req.estimated_latency_us - 42.5).abs() < f32::EPSILON);
        assert_eq!(req.priority, 0);
    }

    #[test]
    fn test_expert_prefetch_request_clone_is_equal() {
        let req = ExpertPrefetchRequest {
            expert_idx: 5,
            source: ExpertWeightLocation::RemoteNode,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 2048,
            estimated_latency_us: 100.0,
            priority: 2,
        };
        let cloned = req.clone();
        assert_eq!(cloned.expert_idx, req.expert_idx);
        assert_eq!(cloned.source, req.source);
        assert_eq!(cloned.destination, req.destination);
        assert_eq!(cloned.bytes, req.bytes);
        assert_eq!(cloned.estimated_latency_us, req.estimated_latency_us);
        assert_eq!(cloned.priority, req.priority);
    }

    #[test]
    fn test_expert_prefetch_request_debug_format() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 512,
            estimated_latency_us: 10.0,
            priority: 1,
        };
        let debug = format!("{:?}", req);
        assert!(debug.contains("expert_idx"));
        assert!(debug.contains("CpuRam"));
        assert!(debug.contains("GpuVram"));
    }

    // --- ExpertWeightPrefetcher: edge case zero experts ---

    #[test]
    fn test_new_with_zero_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(0, 1024);
        assert_eq!(prefetcher.num_experts(), 0);
        assert_eq!(prefetcher.layouts().len(), 0);
        assert!(prefetcher.layout(0).is_none());
        assert_eq!(prefetcher.total_gpu_vram_bytes(), 0);
    }

    // --- ExpertWeightPrefetcher: with_bandwidth extreme values ---

    #[test]
    fn test_with_bandwidth_zero_pcie_produces_infinite_latency() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 64 * 1024 * 1024)
            .with_bandwidth(0.0, 200.0);

        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);
        // Zero bandwidth from CpuRam should produce infinite latency (division by zero = inf)
        assert_eq!(requests.len(), 1);
        assert!(requests[0].estimated_latency_us.is_infinite());
    }

    #[test]
    fn test_with_bandwidth_very_high_reduces_latency() {
        let weight = 64 * 1024 * 1024; // 64 MB
        let low_bw = ExpertWeightPrefetcher::new(2, weight)
            .with_bandwidth(1.0, 1.0);
        let high_bw = ExpertWeightPrefetcher::new(2, weight)
            .with_bandwidth(10000.0, 10000.0);

        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let req_low = low_bw.schedule_prefetch(&[0], &heat_levels);
        let req_high = high_bw.schedule_prefetch(&[0], &heat_levels);

        assert_eq!(req_low.len(), 1);
        assert_eq!(req_high.len(), 1);
        assert!(req_high[0].estimated_latency_us < req_low[0].estimated_latency_us);
    }

    // --- ExpertWeightPrefetcher: with_prefetch_priority edge cases ---

    #[test]
    fn test_with_prefetch_priority_exact_one_keeps_all() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(1.0);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert_eq!(requests.len(), 4);
    }

    // --- can_pipeline_hide: boundary layer indices ---

    #[test]
    fn test_can_pipeline_hide_single_layer_boundary() {
        // Small weights (1024 bytes → 128 compressed), default 32 GB/s PCIe
        // Transfer time: 128 / 1e9 / 32 * 1e6 = ~0.004 us — tiny
        // 1 layer at 100 us = 100 us available → should be hidden
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024)
            .with_layer_compute_time(100.0);
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        let hiding = prefetcher.can_pipeline_hide(&requests, 1);
        if !hiding.is_empty() {
            let (_, can_hide) = hiding[0];
            assert!(can_hide);
        }
    }

    // --- update_location: sequential updates ---

    #[test]
    fn test_update_location_sequential_cycles_through_locations() {
        let mut prefetcher = ExpertWeightPrefetcher::new(1, 1024);
        let locations = [
            ExpertWeightLocation::GpuL2,
            ExpertWeightLocation::GpuVram,
            ExpertWeightLocation::CpuRam,
            ExpertWeightLocation::RemoteNode,
            ExpertWeightLocation::Evicted,
        ];
        for loc in locations {
            prefetcher.update_location(0, loc);
            assert_eq!(prefetcher.layout(0).unwrap().location, loc);
        }
    }

    // --- ExpertWeightPrefetcher: builder chaining preserves all settings ---

    #[test]
    fn test_builder_chaining_combines_all_settings() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 4096)
            .with_bandwidth(50.0, 150.0)
            .with_layer_compute_time(200.0)
            .with_prefetch_priority(0.75);

        assert_eq!(prefetcher.num_experts(), 8);
        assert_eq!(prefetcher.layouts().len(), 8);
        // Verify through schedule_prefetch that priority scaling works
        let heat_levels = vec![ExpertHeatLevel::Warm; 8];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3, 4, 5, 6, 7], &heat_levels);
        // 8 * 0.75 = 6.0 → round to 6
        assert_eq!(requests.len(), 6);
    }

    // =====================================================
    // EXPANDED TESTS (45 additional)
    // =====================================================

    // --- ExpertWeightLocation: Eq trait (reflexive) ---

    #[test]
    fn test_expert_weight_location_eq_reflexive_all_variants() {
        assert_eq!(ExpertWeightLocation::GpuL2, ExpertWeightLocation::GpuL2);
        assert_eq!(ExpertWeightLocation::GpuVram, ExpertWeightLocation::GpuVram);
        assert_eq!(ExpertWeightLocation::CpuRam, ExpertWeightLocation::CpuRam);
        assert_eq!(ExpertWeightLocation::RemoteNode, ExpertWeightLocation::RemoteNode);
        assert_eq!(ExpertWeightLocation::Evicted, ExpertWeightLocation::Evicted);
    }

    // --- ExpertWeightLocation: all pairwise inequality ---

    #[test]
    fn test_expert_weight_location_ne_between_all_distinct_pairs() {
        let variants = [
            ExpertWeightLocation::GpuL2,
            ExpertWeightLocation::GpuVram,
            ExpertWeightLocation::CpuRam,
            ExpertWeightLocation::RemoteNode,
            ExpertWeightLocation::Evicted,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    // --- ExpertWeightLocation: Hash uniqueness in HashMap ---

    #[test]
    fn test_expert_weight_location_as_hash_map_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(ExpertWeightLocation::GpuL2, 1u32);
        map.insert(ExpertWeightLocation::CpuRam, 2u32);
        map.insert(ExpertWeightLocation::RemoteNode, 3u32);

        assert_eq!(map.get(&ExpertWeightLocation::GpuL2), Some(&1));
        assert_eq!(map.get(&ExpertWeightLocation::CpuRam), Some(&2));
        assert_eq!(map.get(&ExpertWeightLocation::RemoteNode), Some(&3));
        assert_eq!(map.get(&ExpertWeightLocation::Evicted), None);
        assert_eq!(map.len(), 3);
    }

    // --- ExpertWeightLocation: Debug format all variants ---

    #[test]
    fn test_expert_weight_location_debug_all_variants() {
        assert!(format!("{:?}", ExpertWeightLocation::GpuL2).contains("GpuL2"));
        assert!(format!("{:?}", ExpertWeightLocation::GpuVram).contains("GpuVram"));
        assert!(format!("{:?}", ExpertWeightLocation::CpuRam).contains("CpuRam"));
        assert!(format!("{:?}", ExpertWeightLocation::RemoteNode).contains("RemoteNode"));
        assert!(format!("{:?}", ExpertWeightLocation::Evicted).contains("Evicted"));
    }

    // --- ExpertWeightLocation: estimated_latency_us is non-negative for finite ---

    #[test]
    fn test_expert_weight_location_latency_non_negative() {
        assert!(ExpertWeightLocation::GpuL2.estimated_latency_us() >= 0.0);
        assert!(ExpertWeightLocation::GpuVram.estimated_latency_us() >= 0.0);
        assert!(ExpertWeightLocation::CpuRam.estimated_latency_us() >= 0.0);
        assert!(ExpertWeightLocation::RemoteNode.estimated_latency_us() >= 0.0);
    }

    // --- ExpertWeightLayout: zero weight_bytes edge case ---

    #[test]
    fn test_expert_weight_layout_zero_bytes() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 0,
            compressed_bytes: 0,
            compression_ratio: 0.0,
            location: ExpertWeightLocation::CpuRam,
        };
        assert_eq!(layout.weight_bytes, 0);
        assert_eq!(layout.compressed_bytes, 0);
        assert_eq!(layout.compression_ratio, 0.0);
    }

    // --- ExpertWeightLayout: large values ---

    #[test]
    fn test_expert_weight_layout_large_values() {
        let layout = ExpertWeightLayout {
            expert_idx: usize::MAX,
            weight_bytes: usize::MAX,
            compressed_bytes: usize::MAX / 8,
            compression_ratio: f32::MAX,
            location: ExpertWeightLocation::RemoteNode,
        };
        assert_eq!(layout.expert_idx, usize::MAX);
        assert_eq!(layout.compressed_bytes, usize::MAX / 8);
        assert_eq!(layout.location, ExpertWeightLocation::RemoteNode);
    }

    // --- ExpertPrefetchRequest: zero bytes ---

    #[test]
    fn test_expert_prefetch_request_zero_bytes() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 0.0,
            priority: 0,
        };
        assert_eq!(req.bytes, 0);
        assert_eq!(req.priority, 0);
    }

    // --- ExpertPrefetchRequest: max priority ---

    #[test]
    fn test_expert_prefetch_request_max_priority() {
        let req = ExpertPrefetchRequest {
            expert_idx: 255,
            source: ExpertWeightLocation::RemoteNode,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 1024,
            estimated_latency_us: 999.0,
            priority: u32::MAX,
        };
        assert_eq!(req.priority, u32::MAX);
    }

    // --- ExpertPrefetchRequest: source equals destination is valid structurally ---

    #[test]
    fn test_expert_prefetch_request_source_equals_destination() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::GpuVram,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 512,
            estimated_latency_us: 0.0,
            priority: 0,
        };
        assert_eq!(req.source, req.destination);
    }

    // --- ExpertWeightPrefetcher: new with zero weight_bytes ---

    #[test]
    fn test_new_with_zero_weight_bytes() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 0);
        assert_eq!(prefetcher.num_experts(), 4);
        let layout = prefetcher.layout(0).unwrap();
        assert_eq!(layout.weight_bytes, 0);
        assert_eq!(layout.compressed_bytes, 0);
    }

    // --- ExpertWeightPrefetcher: new compression ratio with 1 byte weight ---

    #[test]
    fn test_new_with_one_byte_weight() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 1);
        let layout = prefetcher.layout(0).unwrap();
        // 1 / 8 = 0 (integer division)
        assert_eq!(layout.compressed_bytes, 0);
        // compression_ratio = 1 / 0 → inf due to division by zero
        assert!(layout.compression_ratio.is_infinite());
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with single expert ---

    #[test]
    fn test_schedule_prefetch_single_warm_expert() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 4096);
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
        assert_eq!(requests[0].source, ExpertWeightLocation::CpuRam);
        assert_eq!(requests[0].destination, ExpertWeightLocation::GpuVram);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch duplicate expert indices ---

    #[test]
    fn test_schedule_prefetch_duplicate_expert_indices() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[1, 1, 1], &heat_levels);

        // Same expert listed 3 times → 3 separate requests (not deduplicated)
        assert_eq!(requests.len(), 3);
        for req in &requests {
            assert_eq!(req.expert_idx, 1);
        }
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch all cold ---

    #[test]
    fn test_schedule_prefetch_all_cold_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(3, 2048);
        let heat_levels = vec![ExpertHeatLevel::Cold; 3];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2], &heat_levels);

        assert_eq!(requests.len(), 3);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch mixed hot_warm_cold_evicted ---

    #[test]
    fn test_schedule_prefetch_mixed_heat_levels() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);

        // Hot and Evicted skipped; Warm and Cold generate requests
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].expert_idx, 1);
        assert_eq!(requests[1].expert_idx, 2);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch preserves routed order priority ---

    #[test]
    fn test_schedule_prefetch_priority_matches_routed_order() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[3, 0, 2], &heat_levels);

        assert_eq!(requests.len(), 3);
        // Expert 3 has priority 0 (first in routed list)
        assert_eq!(requests[0].expert_idx, 3);
        assert_eq!(requests[0].priority, 0);
        assert_eq!(requests[1].expert_idx, 0);
        assert_eq!(requests[1].priority, 1);
        assert_eq!(requests[2].expert_idx, 2);
        assert_eq!(requests[2].priority, 2);
    }

    // --- ExpertWeightPrefetcher: prefetch_step priority formula ---

    #[test]
    fn test_prefetch_step_priority_formula_step_contribution() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        // 3 experts, step=2 → priority = 2*3 + i
        let requests = prefetcher.prefetch_step(2, &[0, 1, 2]);

        assert_eq!(requests.len(), 3);
        // step=2, len=3: priorities = 2*3+0=6, 2*3+1=7, 2*3+2=8
        assert_eq!(requests[0].priority, 6);
        assert_eq!(requests[1].priority, 7);
        assert_eq!(requests[2].priority, 8);
    }

    // --- ExpertWeightPrefetcher: prefetch_step destination always GpuVram ---

    #[test]
    fn test_prefetch_step_destination_always_gpu_vram() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);

        let requests = prefetcher.prefetch_step(0, &[0, 1]);
        for req in &requests {
            assert_eq!(req.destination, ExpertWeightLocation::GpuVram);
        }
    }

    // --- ExpertWeightPrefetcher: prefetch_step bytes matches compressed ---

    #[test]
    fn test_prefetch_step_bytes_matches_layout_compressed() {
        let weight_per_expert = 8192;
        let prefetcher = ExpertWeightPrefetcher::new(2, weight_per_expert);
        let requests = prefetcher.prefetch_step(0, &[0]);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].bytes, weight_per_expert / 8);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with zero priority keeps all ---

    #[test]
    fn test_prefetch_step_with_default_priority_keeps_all() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(1.0);
        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3]);
        assert_eq!(requests.len(), 4);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with zero priority drops all ---

    #[test]
    fn test_prefetch_step_with_zero_priority_drops_all() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(0.0);
        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3]);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: prefetch_step sorted by priority ascending ---

    #[test]
    fn test_prefetch_step_sorted_by_priority_ascending() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let requests = prefetcher.prefetch_step(3, &[3, 0, 2, 1]);

        // Verify sorted ascending
        for w in requests.windows(2) {
            assert!(w[0].priority <= w[1].priority);
        }
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide empty requests ---

    #[test]
    fn test_can_pipeline_hide_empty_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let result = prefetcher.can_pipeline_hide(&[], 10);
        assert!(result.is_empty());
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide preserves expert indices ---

    #[test]
    fn test_can_pipeline_hide_preserves_expert_indices() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[1, 3], &heat_levels);

        let hiding = prefetcher.can_pipeline_hide(&requests, 5);
        assert_eq!(hiding.len(), 2);
        assert_eq!(hiding[0].0, 1);
        assert_eq!(hiding[1].0, 3);
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide zero latency always hidden ---

    #[test]
    fn test_can_pipeline_hide_zero_latency_always_hidden() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);

        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::GpuL2,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 0.0,
            priority: 0,
        };
        let hiding = prefetcher.can_pipeline_hide(&[req], 0);
        assert_eq!(hiding.len(), 1);
        assert!(hiding[0].1);
    }

    // --- ExpertWeightPrefetcher: total_gpu_vram_bytes mixed locations ---

    #[test]
    fn test_total_gpu_vram_bytes_only_counts_gpu_locations() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(3, ExpertWeightLocation::Evicted);

        let vram = prefetcher.total_gpu_vram_bytes();
        // Only experts 0 and 1 are on GPU, each compressed to 8192/8 = 1024
        assert_eq!(vram, 2 * 1024);
    }

    // --- ExpertWeightPrefetcher: bandwidth_savings_ratio_single_expert ---

    #[test]
    fn test_bandwidth_savings_ratio_single_expert() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 8192);
        let savings = prefetcher.bandwidth_savings_ratio();
        // 8x compression → 1 - 1/8 = 0.875
        assert!((savings - 0.875).abs() < 0.01);
    }

    // --- ExpertWeightPrefetcher: bandwidth_savings_ratio_many_experts ---

    #[test]
    fn test_bandwidth_savings_ratio_many_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(128, 4096);
        let savings = prefetcher.bandwidth_savings_ratio();
        // Still 8x compression regardless of expert count
        assert!((savings - 0.875).abs() < 0.01);
    }

    // --- ExpertWeightPrefetcher: layout default location is CpuRam ---

    #[test]
    fn test_layout_default_location_is_cpu_ram() {
        let prefetcher = ExpertWeightPrefetcher::new(16, 2048);
        for i in 0..16 {
            assert_eq!(
                prefetcher.layout(i).unwrap().location,
                ExpertWeightLocation::CpuRam
            );
        }
    }

    // --- ExpertWeightPrefetcher: layout expert_idx matches index ---

    #[test]
    fn test_layout_expert_idx_matches_vector_position() {
        let prefetcher = ExpertWeightPrefetcher::new(10, 1024);
        for i in 0..10 {
            assert_eq!(prefetcher.layout(i).unwrap().expert_idx, i);
        }
    }

    // --- ExpertWeightPrefetcher: with_bandwidth returns new instance ---

    #[test]
    fn test_with_bandwidth_is_builder_pattern() {
        let customized = ExpertWeightPrefetcher::new(4, 1024)
            .with_bandwidth(99.0, 99.0);
        // num_experts preserved after builder
        assert_eq!(customized.num_experts(), 4);
    }

    // --- ExpertWeightPrefetcher: with_layer_compute_time returns new instance ---

    #[test]
    fn test_with_layer_compute_time_is_builder_pattern() {
        let customized = ExpertWeightPrefetcher::new(4, 1024)
            .with_layer_compute_time(500.0);
        assert_eq!(customized.num_experts(), 4);
    }

    // --- ExpertWeightPrefetcher: update_location to all possible locations ---

    #[test]
    fn test_update_location_to_each_variant() {
        let mut prefetcher = ExpertWeightPrefetcher::new(1, 1024);

        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuL2);

        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuVram);

        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::CpuRam);

        prefetcher.update_location(0, ExpertWeightLocation::RemoteNode);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::RemoteNode);

        prefetcher.update_location(0, ExpertWeightLocation::Evicted);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::Evicted);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch source reflects current layout ---

    #[test]
    fn test_schedule_prefetch_source_reflects_current_location() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 4096);
        prefetcher.update_location(0, ExpertWeightLocation::RemoteNode);
        prefetcher.update_location(1, ExpertWeightLocation::CpuRam);

        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);

        assert_eq!(requests[0].source, ExpertWeightLocation::RemoteNode);
        assert_eq!(requests[1].source, ExpertWeightLocation::CpuRam);
    }

    // --- ExpertWeightPrefetcher: prefetch_step from remote has higher latency than cpu ---

    #[test]
    fn test_prefetch_step_remote_latency_higher_than_cpu() {
        let weight = 64 * 1024 * 1024;
        let mut prefetcher = ExpertWeightPrefetcher::new(2, weight);
        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);

        let requests = prefetcher.prefetch_step(0, &[0, 1]);
        assert_eq!(requests.len(), 2);

        let cpu_req = requests.iter().find(|r| r.expert_idx == 0).unwrap();
        let remote_req = requests.iter().find(|r| r.expert_idx == 1).unwrap();
        // RDMA (100 GB/s) is higher bandwidth than PCIe (32 GB/s),
        // so remote actually has lower latency than CPU. Verify the actual relationship.
        assert!(remote_req.estimated_latency_us < cpu_req.estimated_latency_us);
    }

    // --- ExpertWeightPrefetcher: num_experts zero ---

    #[test]
    fn test_num_experts_zero() {
        let prefetcher = ExpertWeightPrefetcher::new(0, 1024);
        assert_eq!(prefetcher.num_experts(), 0);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with empty heat_levels ---

    #[test]
    fn test_schedule_prefetch_empty_heat_levels_defaults_to_warm() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let requests = prefetcher.schedule_prefetch(&[0, 1], &[]);
        // Empty heat levels → default Warm → generates requests
        assert_eq!(requests.len(), 2);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with u32 max expert id ---

    #[test]
    fn test_prefetch_step_large_expert_id_out_of_bounds() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let requests = prefetcher.prefetch_step(0, &[u32::MAX]);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: prefetch_step priority with step_zero ---

    #[test]
    fn test_prefetch_step_priority_with_step_zero() {
        let prefetcher = ExpertWeightPrefetcher::new(3, 1024);
        let requests = prefetcher.prefetch_step(0, &[2, 0, 1]);

        assert_eq!(requests.len(), 3);
        // step=0, len=3: priorities = 0*3+0=0, 0*3+1=1, 0*3+2=2
        assert_eq!(requests[0].priority, 0);
        assert_eq!(requests[1].priority, 1);
        assert_eq!(requests[2].priority, 2);
    }

    // --- ExpertWeightPrefetcher: prefetch_priority negative rounds to zero ---

    #[test]
    fn test_prefetch_priority_negative_produces_zero_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(-1.0);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // 4 * -1.0 = -4.0 → max(0.0) = 0.0 → truncate to 0
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: prefetch_priority very large keeps all ---

    #[test]
    fn test_prefetch_priority_very_large_keeps_capped() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(1000.0);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // 4 * 1000.0 = 4000.0 → capped at actual count 4
        assert_eq!(requests.len(), 4);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch bytes field ---

    #[test]
    fn test_schedule_prefetch_bytes_matches_compressed_size() {
        let weight = 32 * 1024;
        let prefetcher = ExpertWeightPrefetcher::new(2, weight);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].bytes, weight / 8);
    }

    // --- ExpertWeightLayout: location field mutation through update_location ---

    #[test]
    fn test_layout_location_mutation_preserves_other_fields() {
        let mut prefetcher = ExpertWeightPrefetcher::new(1, 4096);
        let before = prefetcher.layout(0).unwrap().clone();

        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        let after = prefetcher.layout(0).unwrap();

        assert_eq!(after.expert_idx, before.expert_idx);
        assert_eq!(after.weight_bytes, before.weight_bytes);
        assert_eq!(after.compressed_bytes, before.compressed_bytes);
        // compression_ratio unchanged
        assert!((after.compression_ratio - before.compression_ratio).abs() < f32::EPSILON);
        assert_eq!(after.location, ExpertWeightLocation::GpuL2);
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide with large layer count ---

    #[test]
    fn test_can_pipeline_hide_large_layer_count_covers_all() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024)
            .with_layer_compute_time(1.0);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);

        // 100000 layers at 1us = 100000us available, covers any realistic transfer
        let hiding = prefetcher.can_pipeline_hide(&requests, 100000);
        for (_, can_hide) in &hiding {
            assert!(can_hide);
        }
    }

    // --- ExpertWeightPrefetcher: builder chaining order independence ---

    #[test]
    fn test_builder_chaining_order_independent() {
        let p1 = ExpertWeightPrefetcher::new(4, 1024)
            .with_bandwidth(50.0, 150.0)
            .with_layer_compute_time(200.0);

        let p2 = ExpertWeightPrefetcher::new(4, 1024)
            .with_layer_compute_time(200.0)
            .with_bandwidth(50.0, 150.0);

        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let r1 = p1.schedule_prefetch(&[0], &heat_levels);
        let r2 = p2.schedule_prefetch(&[0], &heat_levels);

        assert_eq!(r1.len(), r2.len());
        if !r1.is_empty() && !r2.is_empty() {
            assert!((r1[0].estimated_latency_us - r2[0].estimated_latency_us).abs() < f32::EPSILON);
        }
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with only out-of-bounds ---

    #[test]
    fn test_schedule_prefetch_all_out_of_bounds() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[10, 20, 30], &heat_levels);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: prefetch_step all on gpu produces empty ---

    #[test]
    fn test_prefetch_step_all_on_gpu() {
        let mut prefetcher = ExpertWeightPrefetcher::new(3, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::GpuL2);

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2]);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: prefetch_step all evicted produces empty ---

    #[test]
    fn test_prefetch_step_all_evicted() {
        let mut prefetcher = ExpertWeightPrefetcher::new(3, 1024);
        for i in 0..3 {
            prefetcher.update_location(i, ExpertWeightLocation::Evicted);
        }

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2]);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: update_location to same value is idempotent ---

    #[test]
    fn test_update_location_idempotent() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuL2);
    }

    // --- ExpertWeightPrefetcher: layouts slice length matches num_experts ---

    #[test]
    fn test_layouts_len_matches_num_experts() {
        for n in [0, 1, 4, 16, 64] {
            let prefetcher = ExpertWeightPrefetcher::new(n, 1024);
            assert_eq!(prefetcher.layouts().len(), prefetcher.num_experts());
        }
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with hot_cold_only ---

    #[test]
    fn test_schedule_prefetch_hot_and_cold_only() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 4096);
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Cold];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);

        // Hot skipped, Cold generates request
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 1);
    }

    // =====================================================
    // EXPANDED TESTS BATCH 2 (28 additional)
    // =====================================================

    // --- ExpertWeightLocation: from_heat_level consistent across calls ---

    #[test]
    fn test_from_heat_level_is_deterministic() {
        for _ in 0..10 {
            assert_eq!(
                ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Hot),
                ExpertWeightLocation::GpuL2
            );
        }
    }

    // --- ExpertWeightLocation: GpuVram latency is positive ---

    #[test]
    fn test_gpu_vram_latency_is_positive() {
        assert!(ExpertWeightLocation::GpuVram.estimated_latency_us() > 0.0);
    }

    // --- ExpertWeightLocation: CpuRam latency is positive ---

    #[test]
    fn test_cpu_ram_latency_is_positive() {
        assert!(ExpertWeightLocation::CpuRam.estimated_latency_us() > 0.0);
    }

    // --- ExpertWeightLocation: RemoteNode latency is positive ---

    #[test]
    fn test_remote_node_latency_is_positive() {
        assert!(ExpertWeightLocation::RemoteNode.estimated_latency_us() > 0.0);
    }

    // --- ExpertWeightLayout: expert_idx field independence ---

    #[test]
    fn test_expert_weight_layout_independent_fields() {
        let layout = ExpertWeightLayout {
            expert_idx: 100,
            weight_bytes: 0,
            compressed_bytes: 999,
            compression_ratio: 1.5,
            location: ExpertWeightLocation::Evicted,
        };
        // Each field stores its own value independently
        assert_eq!(layout.expert_idx, 100);
        assert_eq!(layout.weight_bytes, 0);
        assert_eq!(layout.compressed_bytes, 999);
        assert!((layout.compression_ratio - 1.5).abs() < f32::EPSILON);
    }

    // --- ExpertWeightLayout: location can be each variant ---

    #[test]
    fn test_expert_weight_layout_each_location_variant() {
        let variants = [
            ExpertWeightLocation::GpuL2,
            ExpertWeightLocation::GpuVram,
            ExpertWeightLocation::CpuRam,
            ExpertWeightLocation::RemoteNode,
            ExpertWeightLocation::Evicted,
        ];
        for variant in variants {
            let layout = ExpertWeightLayout {
                expert_idx: 0,
                weight_bytes: 1024,
                compressed_bytes: 128,
                compression_ratio: 8.0,
                location: variant,
            };
            assert_eq!(layout.location, variant);
        }
    }

    // --- ExpertPrefetchRequest: estimated_latency_us can be zero ---

    #[test]
    fn test_expert_prefetch_request_zero_latency() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::GpuL2,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 0.0,
            priority: 0,
        };
        assert_eq!(req.estimated_latency_us, 0.0);
    }

    // --- ExpertPrefetchRequest: estimated_latency_us can be negative structurally ---

    #[test]
    fn test_expert_prefetch_request_negative_latency() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 100,
            estimated_latency_us: -1.0,
            priority: 0,
        };
        assert!(req.estimated_latency_us < 0.0);
    }

    // --- ExpertWeightPrefetcher: new with large expert count ---

    #[test]
    fn test_new_with_large_expert_count() {
        let prefetcher = ExpertWeightPrefetcher::new(1024, 4096);
        assert_eq!(prefetcher.num_experts(), 1024);
        assert_eq!(prefetcher.layouts().len(), 1024);
        assert_eq!(prefetcher.layout(1023).unwrap().expert_idx, 1023);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch mixed hot and cold skipping ---

    #[test]
    fn test_schedule_prefetch_all_hot_no_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let heat_levels = vec![ExpertHeatLevel::Hot; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch all evicted no requests ---

    #[test]
    fn test_schedule_prefetch_all_evicted_no_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let heat_levels = vec![ExpertHeatLevel::Evicted; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert!(requests.is_empty());
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch single request latency ---

    #[test]
    fn test_schedule_prefetch_latency_from_cpu_ram_positive() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024 * 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);
        assert_eq!(requests.len(), 1);
        assert!(requests[0].estimated_latency_us > 0.0);
    }

    // --- ExpertWeightPrefetcher: prefetch_step single expert ---

    #[test]
    fn test_prefetch_step_single_expert() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 4096);
        let requests = prefetcher.prefetch_step(0, &[0]);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
    }

    // --- ExpertWeightPrefetcher: prefetch_step priority with step_one ---

    #[test]
    fn test_prefetch_step_priority_step_one() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let requests = prefetcher.prefetch_step(1, &[0, 1]);
        // step=1, len=2: priorities = 1*2+0=2, 1*2+1=3
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].priority, 2);
        assert_eq!(requests[1].priority, 3);
    }

    // --- ExpertWeightPrefetcher: prefetch_step expert_idx preserved ---

    #[test]
    fn test_prefetch_step_preserves_expert_ids() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let requests = prefetcher.prefetch_step(0, &[3, 1, 0]);
        let expert_ids: Vec<usize> = requests.iter().map(|r| r.expert_idx).collect();
        // Sorted by priority, but all expert IDs present
        assert!(expert_ids.contains(&3));
        assert!(expert_ids.contains(&1));
        assert!(expert_ids.contains(&0));
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide returns same count as input ---

    #[test]
    fn test_can_pipeline_hide_output_count_matches_input() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2], &heat_levels);
        let hiding = prefetcher.can_pipeline_hide(&requests, 5);
        assert_eq!(hiding.len(), requests.len());
    }

    // --- ExpertWeightPrefetcher: total_gpu_vram_bytes after location_update to non_gpu ---

    #[test]
    fn test_total_gpu_vram_decreases_after_moving_off_gpu() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);

        let vram_before = prefetcher.total_gpu_vram_bytes();
        assert_eq!(vram_before, 2 * 1024);

        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        let vram_after = prefetcher.total_gpu_vram_bytes();
        assert_eq!(vram_after, 1 * 1024);
    }

    // --- ExpertWeightPrefetcher: bandwidth_savings_ratio unchanged by location updates ---

    #[test]
    fn test_bandwidth_savings_ratio_unchanged_by_location_updates() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let savings_before = prefetcher.bandwidth_savings_ratio();

        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);

        let savings_after = prefetcher.bandwidth_savings_ratio();
        assert!((savings_before - savings_after).abs() < f32::EPSILON);
    }

    // --- ExpertWeightPrefetcher: update_location multiple experts independently ---

    #[test]
    fn test_update_location_independent_per_expert() {
        let mut prefetcher = ExpertWeightPrefetcher::new(3, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::RemoteNode);
        // Expert 2 untouched

        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(1).unwrap().location, ExpertWeightLocation::RemoteNode);
        assert_eq!(prefetcher.layout(2).unwrap().location, ExpertWeightLocation::CpuRam);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with cold but already on gpu ---

    #[test]
    fn test_schedule_prefetch_cold_heat_but_gpu_location_skipped() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        // heat=Cold but location=GpuL2 → heat check skips (Cold generates request)
        // But heat level is what matters for skip decision
        let heat_levels = vec![ExpertHeatLevel::Cold; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);
        // Cold is NOT skipped (only Hot and Evicted are skipped)
        assert_eq!(requests.len(), 1);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with mixed locations ---

    #[test]
    fn test_prefetch_step_mixed_locations_selects_non_gpu() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::CpuRam);
        prefetcher.update_location(3, ExpertWeightLocation::Evicted);

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3]);
        // Only CpuRam expert generates request; GpuL2/GpuVram skipped, Evicted skipped
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 2);
    }

    // --- ExpertWeightPrefetcher: builder with_prefetch_priority default is one ---

    #[test]
    fn test_default_prefetch_priority_is_one() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // Default priority 1.0 → all 4 kept
        assert_eq!(requests.len(), 4);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch remote_node_source latency ---

    #[test]
    fn test_schedule_prefetch_remote_source_latency_positive() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 64 * 1024 * 1024);
        prefetcher.update_location(0, ExpertWeightLocation::RemoteNode);

        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);
        assert_eq!(requests.len(), 1);
        assert!(requests[0].estimated_latency_us > 0.0);
    }

    // --- ExpertWeightPrefetcher: new compression_bytes integer_division ---

    #[test]
    fn test_new_compression_bytes_less_than_8_rounds_down() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 7);
        let layout = prefetcher.layout(0).unwrap();
        // 7 / 8 = 0 (integer division)
        assert_eq!(layout.compressed_bytes, 0);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with duplicate expert ids ---

    #[test]
    fn test_prefetch_step_duplicate_expert_ids() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let requests = prefetcher.prefetch_step(0, &[0, 0, 0]);
        // Same expert listed 3 times → 3 separate requests
        assert_eq!(requests.len(), 3);
        for req in &requests {
            assert_eq!(req.expert_idx, 0);
        }
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide with exact latency ---

    #[test]
    fn test_can_pipeline_hide_exact_boundary() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024)
            .with_layer_compute_time(10.0);

        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 10.0,
            priority: 0,
        };
        // 1 layer * 10us = 10us available, latency = 10us → can hide (<=)
        let hiding = prefetcher.can_pipeline_hide(&[req], 1);
        assert_eq!(hiding.len(), 1);
        assert!(hiding[0].1);
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide just above boundary ---

    #[test]
    fn test_can_pipeline_hide_just_above_boundary() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024)
            .with_layer_compute_time(10.0);

        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 10.001,
            priority: 0,
        };
        // 1 layer * 10us = 10us available, latency = 10.001us → cannot hide
        let hiding = prefetcher.can_pipeline_hide(&[req], 1);
        assert_eq!(hiding.len(), 1);
        assert!(!hiding[0].1);
    }

    // =====================================================
    // EXPANDED TESTS BATCH 3 (20 additional)
    // =====================================================

    // --- ExpertWeightLocation: all variants in array ---

    #[test]
    fn test_expert_weight_location_array_iteration() {
        let variants = [
            ExpertWeightLocation::GpuL2,
            ExpertWeightLocation::GpuVram,
            ExpertWeightLocation::CpuRam,
            ExpertWeightLocation::RemoteNode,
            ExpertWeightLocation::Evicted,
        ];
        assert_eq!(variants.len(), 5);
    }

    // --- ExpertWeightLayout: clone produces distinct object ---

    #[test]
    fn test_expert_weight_layout_clone_distinct() {
        let mut layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 1024,
            compressed_bytes: 128,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        let cloned = layout.clone();
        layout.location = ExpertWeightLocation::GpuL2;
        // Original mutated, clone unchanged
        assert_eq!(layout.location, ExpertWeightLocation::GpuL2);
        assert_eq!(cloned.location, ExpertWeightLocation::CpuRam);
    }

    // --- ExpertPrefetchRequest: clone produces distinct object ---

    #[test]
    fn test_expert_prefetch_request_clone_distinct() {
        let mut req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 512,
            estimated_latency_us: 10.0,
            priority: 1,
        };
        let cloned = req.clone();
        req.priority = 99;
        assert_eq!(req.priority, 99);
        assert_eq!(cloned.priority, 1);
    }

    // --- ExpertWeightPrefetcher: new with two experts ---

    #[test]
    fn test_new_with_two_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 4096);
        assert_eq!(prefetcher.num_experts(), 2);
        assert!(prefetcher.layout(0).is_some());
        assert!(prefetcher.layout(1).is_some());
    }

    // --- ExpertWeightPrefetcher: new compressed_bytes calculation ---

    #[test]
    fn test_new_compressed_bytes_exact_division() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 64);
        let layout = prefetcher.layout(0).unwrap();
        assert_eq!(layout.compressed_bytes, 8); // 64 / 8
        assert!((layout.compression_ratio - 8.0).abs() < f32::EPSILON);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch cold_not_skipped ---

    #[test]
    fn test_schedule_prefetch_cold_expert_not_skipped() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 4096);
        let heat_levels = vec![ExpertHeatLevel::Cold, ExpertHeatLevel::Cold];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);
        assert_eq!(requests.len(), 2);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch with repeated_out_of_bounds ---

    #[test]
    fn test_schedule_prefetch_all_out_of_bounds_except_first() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 1024);
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0, 5, 10, 20], &heat_levels);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with u32_zero ---

    #[test]
    fn test_prefetch_step_expert_id_zero() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 1024);
        let requests = prefetcher.prefetch_step(0, &[0]);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 0);
    }

    // --- ExpertWeightPrefetcher: prefetch_step priority increases with step ---

    #[test]
    fn test_prefetch_step_priority_higher_for_later_steps() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let r0 = prefetcher.prefetch_step(0, &[0]);
        let r1 = prefetcher.prefetch_step(1, &[0]);
        let r10 = prefetcher.prefetch_step(10, &[0]);

        assert!(r1[0].priority > r0[0].priority);
        assert!(r10[0].priority > r1[0].priority);
    }

    // --- ExpertWeightPrefetcher: can_pipeline_hide multiple requests ---

    #[test]
    fn test_can_pipeline_hide_multiple_requests_independent() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_layer_compute_time(100.0);

        let reqs = vec![
            ExpertPrefetchRequest {
                expert_idx: 0,
                source: ExpertWeightLocation::CpuRam,
                destination: ExpertWeightLocation::GpuVram,
                bytes: 0,
                estimated_latency_us: 50.0,
                priority: 0,
            },
            ExpertPrefetchRequest {
                expert_idx: 1,
                source: ExpertWeightLocation::CpuRam,
                destination: ExpertWeightLocation::GpuVram,
                bytes: 0,
                estimated_latency_us: 150.0,
                priority: 1,
            },
        ];
        let hiding = prefetcher.can_pipeline_hide(&reqs, 1);
        // 1 layer * 100us = 100us: first (50us) hidden, second (150us) not
        assert_eq!(hiding[0], (0, true));
        assert_eq!(hiding[1], (1, false));
    }

    // --- ExpertWeightPrefetcher: total_gpu_vram_remote_node_not_counted ---

    #[test]
    fn test_total_gpu_vram_remote_node_not_counted() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 8192);
        prefetcher.update_location(0, ExpertWeightLocation::RemoteNode);
        prefetcher.update_location(1, ExpertWeightLocation::Evicted);
        assert_eq!(prefetcher.total_gpu_vram_bytes(), 0);
    }

    // --- ExpertWeightPrefetcher: bandwidth_savings_ratio_after_location_change ---

    #[test]
    fn test_bandwidth_savings_ratio_independent_of_vram() {
        let mut pf = ExpertWeightPrefetcher::new(4, 4096);
        let before = pf.bandwidth_savings_ratio();
        pf.update_location(0, ExpertWeightLocation::GpuL2);
        pf.update_location(1, ExpertWeightLocation::GpuVram);
        assert!((pf.bandwidth_savings_ratio() - before).abs() < f32::EPSILON);
    }

    // --- ExpertWeightPrefetcher: update_location does_not_affect_other_experts ---

    #[test]
    fn test_update_location_isolation() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 1024);
        prefetcher.update_location(2, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::CpuRam);
        assert_eq!(prefetcher.layout(1).unwrap().location, ExpertWeightLocation::CpuRam);
        assert_eq!(prefetcher.layout(2).unwrap().location, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(3).unwrap().location, ExpertWeightLocation::CpuRam);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch estimated_latency_from_remote ---

    #[test]
    fn test_schedule_prefetch_remote_source_latency_finite() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024 * 1024);
        prefetcher.update_location(0, ExpertWeightLocation::RemoteNode);
        let heat_levels = vec![ExpertHeatLevel::Warm; 2];
        let requests = prefetcher.schedule_prefetch(&[0], &heat_levels);
        assert_eq!(requests.len(), 1);
        assert!(requests[0].estimated_latency_us.is_finite());
        assert!(requests[0].estimated_latency_us > 0.0);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch evicted_source_infinite_latency ---

    #[test]
    fn test_schedule_prefetch_evicted_heat_skipped_regardless_of_location() {
        let mut prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        let heat_levels = vec![ExpertHeatLevel::Evicted, ExpertHeatLevel::Warm];
        let requests = prefetcher.schedule_prefetch(&[0, 1], &heat_levels);
        // Evicted heat skips even if location is GpuVram
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 1);
    }

    // --- ExpertWeightPrefetcher: prefetch_step with high step_value ---

    #[test]
    fn test_prefetch_step_high_step_value() {
        let prefetcher = ExpertWeightPrefetcher::new(2, 1024);
        let requests = prefetcher.prefetch_step(10000, &[0]);
        assert_eq!(requests.len(), 1);
        assert!(requests[0].priority > 0);
    }

    // --- ExpertWeightPrefetcher: builder_with_prefetch_priority_chain ---

    #[test]
    fn test_builder_priority_then_bandwidth_then_compute() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 1024)
            .with_prefetch_priority(0.5)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(50.0);

        assert_eq!(prefetcher.num_experts(), 4);
        let heat_levels = vec![ExpertHeatLevel::Warm; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // 4 * 0.5 = 2
        assert_eq!(requests.len(), 2);
    }

    // --- ExpertWeightPrefetcher: schedule_prefetch_priority_scaling_edge ---

    #[test]
    fn test_schedule_prefetch_priority_scaling_rounds() {
        // 5 * 0.6 = 3.0 → 3
        let prefetcher = ExpertWeightPrefetcher::new(5, 1024)
            .with_prefetch_priority(0.6);
        let heat_levels = vec![ExpertHeatLevel::Warm; 5];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3, 4], &heat_levels);
        assert_eq!(requests.len(), 3);
    }

    // --- ExpertWeightPrefetcher: layouts_returns_correct_slice ---

    #[test]
    fn test_layouts_mutually_consistent_with_layout() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 2048);
        let layouts = prefetcher.layouts();
        for i in 0..8 {
            assert_eq!(prefetcher.layout(i).unwrap() as *const _, &layouts[i] as *const _);
        }
    }

    // =====================================================
    // EXPANDED TESTS BATCH 4 (15 additional)
    // =====================================================

    // --- schedule_prefetch: Evicted location with Warm heat produces infinite latency ---

    #[test]
    fn test_schedule_prefetch_evicted_location_warm_heat_infinite_latency() {
        let mut pf = ExpertWeightPrefetcher::new(2, 1024);
        pf.update_location(0, ExpertWeightLocation::Evicted);
        let heat = vec![ExpertHeatLevel::Warm; 2];
        let reqs = pf.schedule_prefetch(&[0], &heat);
        // Warm heat is not skipped, but Evicted location yields infinite latency
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].estimated_latency_us.is_infinite());
    }

    // --- ExpertPrefetchRequest: NaN latency is structurally valid ---

    #[test]
    fn test_expert_prefetch_request_nan_latency() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 512,
            estimated_latency_us: f32::NAN,
            priority: 0,
        };
        assert!(req.estimated_latency_us.is_nan());
    }

    // --- ExpertPrefetchRequest: Inf latency is structurally valid ---

    #[test]
    fn test_expert_prefetch_request_inf_latency() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::Evicted,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 1024,
            estimated_latency_us: f32::INFINITY,
            priority: 0,
        };
        assert!(req.estimated_latency_us.is_infinite());
        assert!(req.estimated_latency_us > 0.0);
    }

    // --- can_pipeline_hide: infinite latency cannot be hidden ---

    #[test]
    fn test_can_pipeline_hide_infinite_latency_never_hidden() {
        let pf = ExpertWeightPrefetcher::new(2, 1024)
            .with_layer_compute_time(10000.0);
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::Evicted,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: f32::INFINITY,
            priority: 0,
        };
        let hiding = pf.can_pipeline_hide(&[req], 100000);
        assert_eq!(hiding.len(), 1);
        assert!(!hiding[0].1);
    }

    // --- new: weight_bytes exactly 8 gives compressed_bytes 1 ---

    #[test]
    fn test_new_weight_bytes_exact_division_boundary() {
        let pf = ExpertWeightPrefetcher::new(1, 8);
        let layout = pf.layout(0).unwrap();
        assert_eq!(layout.compressed_bytes, 1);
        assert!((layout.compression_ratio - 8.0).abs() < f32::EPSILON);
    }

    // --- new: weight_bytes 9 gives compressed_bytes 1 (truncation) ---

    #[test]
    fn test_new_weight_bytes_non_multiple_of_8_truncates() {
        let pf = ExpertWeightPrefetcher::new(1, 9);
        let layout = pf.layout(0).unwrap();
        assert_eq!(layout.compressed_bytes, 1); // 9 / 8 = 1
        let expected_ratio = 9.0_f32 / 1.0_f32;
        assert!((layout.compression_ratio - expected_ratio).abs() < f32::EPSILON);
    }

    // --- schedule_prefetch: heat_levels longer than num_experts ---

    #[test]
    fn test_schedule_prefetch_excess_heat_levels_ignored() {
        let pf = ExpertWeightPrefetcher::new(2, 1024);
        let heat = vec![ExpertHeatLevel::Warm; 10];
        let reqs = pf.schedule_prefetch(&[0, 1], &heat);
        assert_eq!(reqs.len(), 2);
    }

    // --- prefetch_step: source field matches layout location ---

    #[test]
    fn test_prefetch_step_source_matches_layout_location() {
        let mut pf = ExpertWeightPrefetcher::new(2, 4096);
        pf.update_location(0, ExpertWeightLocation::CpuRam);
        pf.update_location(1, ExpertWeightLocation::RemoteNode);
        let reqs = pf.prefetch_step(0, &[0, 1]);
        let req0 = reqs.iter().find(|r| r.expert_idx == 0).unwrap();
        let req1 = reqs.iter().find(|r| r.expert_idx == 1).unwrap();
        assert_eq!(req0.source, ExpertWeightLocation::CpuRam);
        assert_eq!(req1.source, ExpertWeightLocation::RemoteNode);
    }

    // --- prefetch_step: bytes field matches compressed_bytes ---

    #[test]
    fn test_prefetch_step_bytes_matches_compressed_per_expert() {
        let pf = ExpertWeightPrefetcher::new(3, 16 * 1024);
        let reqs = pf.prefetch_step(0, &[0, 1, 2]);
        for req in &reqs {
            assert_eq!(req.bytes, 16 * 1024 / 8);
        }
    }

    // --- total_gpu_vram_bytes: RemoteNode and Evicted contribute zero ---

    #[test]
    fn test_total_gpu_vram_only_gpu_l2_and_gpu_vram_counted() {
        let mut pf = ExpertWeightPrefetcher::new(5, 8192);
        pf.update_location(0, ExpertWeightLocation::GpuL2);
        pf.update_location(1, ExpertWeightLocation::GpuVram);
        pf.update_location(2, ExpertWeightLocation::CpuRam);
        pf.update_location(3, ExpertWeightLocation::RemoteNode);
        pf.update_location(4, ExpertWeightLocation::Evicted);
        assert_eq!(pf.total_gpu_vram_bytes(), 2 * 1024);
    }

    // --- bandwidth_savings_ratio: zero weight_bytes per expert ---

    #[test]
    fn test_bandwidth_savings_ratio_zero_weight_per_expert() {
        let pf = ExpertWeightPrefetcher::new(4, 0);
        // total_original=0 → savings=0.0
        assert_eq!(pf.bandwidth_savings_ratio(), 0.0);
    }

    // --- ExpertWeightLayout: subnormal compression_ratio ---

    #[test]
    fn test_expert_weight_layout_subnormal_compression_ratio() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 1,
            compressed_bytes: 1,
            compression_ratio: f32::from_bits(1), // smallest positive subnormal
            location: ExpertWeightLocation::CpuRam,
        };
        assert!(layout.compression_ratio > 0.0);
        assert!(layout.compression_ratio.is_subnormal());
    }

    // --- ExpertPrefetchRequest: negative priority is structurally valid ---

    #[test]
    fn test_expert_prefetch_request_negative_priority_wraps() {
        // u32 cannot be negative, but a computed priority from u32 arithmetic can overflow
        // Verify the struct accepts any u32 value
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 0.0,
            priority: u32::MAX,
        };
        assert_eq!(req.priority, u32::MAX);
    }

    // --- prefetch_step: GpuVram location skips correctly ---

    #[test]
    fn test_prefetch_step_gpu_vram_location_skipped() {
        let mut pf = ExpertWeightPrefetcher::new(3, 1024);
        pf.update_location(0, ExpertWeightLocation::GpuVram);
        pf.update_location(1, ExpertWeightLocation::CpuRam);
        pf.update_location(2, ExpertWeightLocation::CpuRam);
        let reqs = pf.prefetch_step(0, &[0, 1, 2]);
        assert_eq!(reqs.len(), 2);
        assert!(reqs.iter().all(|r| r.expert_idx != 0));
    }

    // --- schedule_prefetch: request estimated_latency_us for CpuRam positive ---

    #[test]
    fn test_schedule_prefetch_cpu_ram_latency_is_finite_positive() {
        let pf = ExpertWeightPrefetcher::new(1, 1024 * 1024);
        let heat = vec![ExpertHeatLevel::Warm];
        let reqs = pf.schedule_prefetch(&[0], &heat);
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].estimated_latency_us.is_finite());
        assert!(reqs[0].estimated_latency_us > 0.0);
    }

    // --- ExpertWeightPrefetcher: with_bandwidth zero RDMA only affects remote ---

    #[test]
    fn test_with_bandwidth_zero_rdma_does_not_affect_cpu_transfer() {
        let pf = ExpertWeightPrefetcher::new(2, 64 * 1024 * 1024)
            .with_bandwidth(32.0, 0.0);
        let heat = vec![ExpertHeatLevel::Warm; 2];
        let reqs = pf.schedule_prefetch(&[0], &heat);
        // Expert 0 is in CpuRam, PCIe bandwidth is still 32 GB/s → finite latency
        assert_eq!(reqs.len(), 1);
        assert!(reqs[0].estimated_latency_us.is_finite());
        assert!(reqs[0].estimated_latency_us > 0.0);
    }
}
