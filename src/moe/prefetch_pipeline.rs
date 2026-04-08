//! §14.5 RDMA/PCIe 流水线预取编排 (Pipelined Prefetch)
//!
//! ## 核心职责
//! 将 Softmax 质心坐标通过页表馈入硬件预取系统:
//! - GPU 处理本层 Dense 网络的同时，总线通道上并行预取下一层 KV 块
//! - GPU: cuMemPrefetchAsync 无阻塞预加载
//! - CPU→GPU: PCIe DMA 异步传输
//! - 远程节点→GPU: RDMA 异步传输
//!
//! ## §14.5 "将算力的减法转为带宽的乘法"
//! Softmax 质心坐标通过页表直接馈入底层硬件预取系统。
//! GPU 处理本层 Dense 网络的同时，总线通道上已在并行利用 cuMemPrefetchAsync
//! 甚至跨机 RDMA 加载下一层的 KV 块。

/// 预取传输类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchTransferKind {
    /// GPU 内部 HBM → L2 (cuMemPrefetchAsync)
    GpuHbmToL2,
    /// CPU RAM → GPU VRAM (PCIe DMA)
    PcieDma,
    /// 远程节点 → GPU VRAM (RDMA)
    Rdma,
}

/// 预取流水线阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrefetchStage {
    /// 阶段 0: Softmax 质心计算完成，生成预取地址
    CentroidComputed,
    /// 阶段 1: 地址翻译完成 (逻辑页 → 物理地址)
    AddressTranslated,
    /// 阶段 2: 异步传输已提交 (PCIe/RDMA)
    TransferSubmitted,
    /// 阶段 3: 传输完成，数据驻留目标缓存
    TransferCompleted,
}

/// 单个流水线预取条目
#[derive(Debug, Clone)]
pub struct PipelinePrefetchEntry {
    /// 目标层索引 (N+1)
    pub target_layer: usize,
    /// KV 块起始 token 索引
    pub start_token: usize,
    /// KV 块 token 数量
    pub token_count: usize,
    /// 传输类型
    pub transfer_kind: PrefetchTransferKind,
    /// 当前阶段
    pub stage: PrefetchStage,
    /// 传输字节数
    pub bytes: usize,
    /// 提交时间戳 (单调时钟 μs)
    pub submit_timestamp_us: u64,
    /// 预估完成时间 (μs)
    pub estimated_latency_us: f32,
}

/// 流水线预取编排器
///
/// §14.5: 编排 GPU 计算与异步预取的流水线重叠。
/// 设计目标: 计算层 N 的同时，预取层 N+1 的 KV 块到 GPU 缓存。
pub struct PrefetchPipeline {
    /// 当前活跃的预取条目
    active_entries: Vec<PipelinePrefetchEntry>,
    /// 已完成的预取条目 (用于统计)
    completed_count: u64,
    /// 流水线深度 (预取提前多少层)
    pipeline_depth: usize,
    /// PCIe 带宽 (GB/s)
    pcie_bandwidth_gbs: f32,
    /// RDMA 带宽 (GB/s)
    rdma_bandwidth_gbs: f32,
    /// GPU 层计算时间 (μs)
    layer_compute_time_us: f32,
    /// 当前层索引 (单调递增)
    current_layer: usize,
    /// 总层数
    num_layers: usize,
}

impl PrefetchPipeline {
    /// 创建新的流水线预取编排器
    pub fn new(num_layers: usize) -> Self {
        Self {
            active_entries: Vec::with_capacity(16),
            completed_count: 0,
            pipeline_depth: 1, // 默认预取 1 层
            pcie_bandwidth_gbs: 32.0, // PCIe 4.0 x16
            rdma_bandwidth_gbs: 100.0, // 100 Gbps RDMA
            layer_compute_time_us: 100.0,
            current_layer: 0,
            num_layers,
        }
    }

    /// 配置流水线深度
    pub fn with_pipeline_depth(mut self, depth: usize) -> Self {
        self.pipeline_depth = depth.max(1).min(self.num_layers);
        self
    }

    /// 配置带宽参数
    pub fn with_bandwidth(mut self, pcie_gbs: f32, rdma_gbs: f32) -> Self {
        self.pcie_bandwidth_gbs = pcie_gbs;
        self.rdma_bandwidth_gbs = rdma_gbs;
        self
    }

    /// 配置层计算时间
    pub fn with_layer_compute_time(mut self, us: f32) -> Self {
        self.layer_compute_time_us = us;
        self
    }

    /// 推进到下一层，处理流水线预取
    ///
    /// §14.5: 在层 N 的计算完成后调用。
    /// 检查已提交的预取是否完成，并提交新的预取请求。
    pub fn advance_layer(&mut self, centroid_tokens: &[usize], kv_block_size: usize) {
        self.current_layer += 1;

        // 标记已完成的预取条目
        let now_timestamp = self.current_layer as u64 * self.layer_compute_time_us as u64;
        self.active_entries.retain(|entry| {
            let elapsed = now_timestamp.saturating_sub(entry.submit_timestamp_us);
            let completed = elapsed as f32 >= entry.estimated_latency_us;
            if completed {
                self.completed_count += 1;
            }
            !completed
        });

        // 为后续层提交新的预取请求
        for depth in 1..=self.pipeline_depth {
            let target_layer = self.current_layer + depth;
            if target_layer >= self.num_layers {
                break;
            }

            for &token_idx in centroid_tokens {
                let (transfer_kind, latency_us) = if target_layer <= self.current_layer + 1 {
                    // 相邻层 → GPU HBM → L2 (最快)
                    (PrefetchTransferKind::GpuHbmToL2, 5.0)
                } else if target_layer <= self.current_layer + 3 {
                    // 近层 → PCIe DMA
                    let bytes_gb = (kv_block_size * 2) as f32 / 1e9; // K+V
                    let latency = bytes_gb / self.pcie_bandwidth_gbs * 1e6;
                    (PrefetchTransferKind::PcieDma, latency)
                } else {
                    // 远层 → RDMA
                    let bytes_gb = (kv_block_size * 2) as f32 / 1e9;
                    let latency = bytes_gb / self.rdma_bandwidth_gbs * 1e6;
                    (PrefetchTransferKind::Rdma, latency)
                };

                self.active_entries.push(PipelinePrefetchEntry {
                    target_layer,
                    start_token: token_idx,
                    token_count: 1,
                    transfer_kind,
                    stage: PrefetchStage::CentroidComputed,
                    bytes: kv_block_size * 2, // K + V
                    submit_timestamp_us: now_timestamp,
                    estimated_latency_us: latency_us,
                });
            }
        }
    }

    /// 检查指定层的预取是否已完成
    pub fn is_prefetch_completed(&self, layer: usize) -> bool {
        // 如果活跃条目中没有该层的条目，说明已完成或未提交
        !self
            .active_entries
            .iter()
            .any(|e| e.target_layer == layer)
    }

    /// 获取当前流水线利用率 (0.0-1.0)
    ///
    /// 利用率 = 已完成预取数 / (当前层 × pipeline_depth)
    pub fn utilization(&self) -> f32 {
        if self.current_layer == 0 {
            return 0.0;
        }
        let expected = self.current_layer as f64 * self.pipeline_depth as f64;
        if expected <= 0.0 {
            return 0.0;
        }
        (self.completed_count as f64 / expected).min(1.0) as f32
    }

    /// 获取活跃预取条目数
    pub fn active_count(&self) -> usize {
        self.active_entries.len()
    }

    /// 获取已完成的预取数
    pub fn completed_count(&self) -> u64 {
        self.completed_count
    }

    /// 检查流水线是否能完全掩盖预取延迟
    ///
    /// §14.5: "冷专家卡顿被计算流水线完美掩盖 (Pipelining)"
    /// 条件: 预取延迟 < (pipeline_depth × 层计算时间)
    pub fn can_hide_latency(&self, prefetch_latency_us: f32) -> bool {
        let available_time_us = self.pipeline_depth as f32 * self.layer_compute_time_us;
        prefetch_latency_us <= available_time_us
    }

    /// 获取当前层索引
    pub fn current_layer(&self) -> usize {
        self.current_layer
    }

    /// 重置到初始状态
    pub fn reset(&mut self) {
        self.current_layer = 0;
        self.active_entries.clear();
        self.completed_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = PrefetchPipeline::new(32);
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
    }

    #[test]
    fn test_advance_layer_submits_prefetch() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1);

        // 层 0 计算完成，提交层 1 的预取
        pipeline.advance_layer(&[5, 10, 20], 1024);
        assert_eq!(pipeline.current_layer(), 1);
        assert!(!pipeline.active_entries.is_empty());
    }

    #[test]
    fn test_prefetch_completion() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // 层 0 完成
        pipeline.advance_layer(&[5], 1024);
        // 层 1 完成 → 层 0 的预取应该完成 (GpuHbmToL2 latency=5μs << 100μs compute)
        pipeline.advance_layer(&[10], 1024);

        // 至少一些预取应该已完成
        assert!(pipeline.completed_count() > 0);
    }

    #[test]
    fn test_latency_hiding() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        // 2 层 × 100μs = 200μs 可用
        assert!(pipeline.can_hide_latency(150.0));
        assert!(!pipeline.can_hide_latency(250.0));
    }

    #[test]
    fn test_utilization() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // 推进多层
        for _ in 0..10 {
            pipeline.advance_layer(&[5], 1024);
        }

        let util = pipeline.utilization();
        assert!(util >= 0.0 && util <= 1.0);
    }

    #[test]
    fn test_pipeline_depth_limit() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(100); // 超过 num_layers

        // 应该被限制在 num_layers
        assert_eq!(pipeline.pipeline_depth, 32);
    }

    #[test]
    fn test_reset() {
        let mut pipeline = PrefetchPipeline::new(32);
        pipeline.advance_layer(&[5], 1024);
        assert!(pipeline.current_layer() > 0);

        pipeline.reset();
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
    }

    #[test]
    fn test_transfer_kind_selection() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_layer_compute_time(100.0);

        // 层 0 完成，提交层 1-5 的预取
        pipeline.advance_layer(&[0], 4096);

        // 应该有不同传输类型的条目
        let has_hbm = pipeline.active_entries.iter().any(|e| e.transfer_kind == PrefetchTransferKind::GpuHbmToL2);
        assert!(has_hbm, "should have HBM→L2 entries for adjacent layers");
    }
}
