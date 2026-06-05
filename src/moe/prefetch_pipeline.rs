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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefetchTransferKind {
    /// GPU 内部 HBM → L2 (cuMemPrefetchAsync)
    GpuHbmToL2,
    /// CPU RAM → GPU VRAM (PCIe DMA)
    PcieDma,
    /// 远程节点 → GPU VRAM (RDMA)
    Rdma,
}

/// 预取流水线阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

    // ── Enum derive traits ──

    #[test]
    fn prefetch_transfer_kind_equality() {
        assert_eq!(PrefetchTransferKind::GpuHbmToL2, PrefetchTransferKind::GpuHbmToL2);
        assert_ne!(PrefetchTransferKind::GpuHbmToL2, PrefetchTransferKind::PcieDma);
        assert_ne!(PrefetchTransferKind::PcieDma, PrefetchTransferKind::Rdma);
    }

    #[test]
    fn prefetch_transfer_kind_copy() {
        let a = PrefetchTransferKind::Rdma;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn prefetch_transfer_kind_debug() {
        let debug = format!("{:?}", PrefetchTransferKind::PcieDma);
        assert!(debug.contains("PcieDma"));
    }

    #[test]
    fn prefetch_stage_ordering() {
        assert!(PrefetchStage::CentroidComputed < PrefetchStage::AddressTranslated);
        assert!(PrefetchStage::AddressTranslated < PrefetchStage::TransferSubmitted);
        assert!(PrefetchStage::TransferSubmitted < PrefetchStage::TransferCompleted);
    }

    #[test]
    fn prefetch_stage_equality() {
        assert_eq!(PrefetchStage::CentroidComputed, PrefetchStage::CentroidComputed);
        assert_ne!(PrefetchStage::CentroidComputed, PrefetchStage::TransferCompleted);
    }

    #[test]
    fn prefetch_stage_debug() {
        let debug = format!("{:?}", PrefetchStage::TransferSubmitted);
        assert!(debug.contains("TransferSubmitted"));
    }

    // ── PipelinePrefetchEntry ──

    #[test]
    fn entry_fields() {
        let entry = PipelinePrefetchEntry {
            target_layer: 5,
            start_token: 10,
            token_count: 4,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferSubmitted,
            bytes: 8192,
            submit_timestamp_us: 1000,
            estimated_latency_us: 50.0,
        };
        assert_eq!(entry.target_layer, 5);
        assert_eq!(entry.start_token, 10);
        assert_eq!(entry.token_count, 4);
        assert_eq!(entry.bytes, 8192);
        assert!((entry.estimated_latency_us - 50.0).abs() < 1e-6);
    }

    #[test]
    fn entry_debug() {
        let entry = PipelinePrefetchEntry {
            target_layer: 1,
            start_token: 0,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::GpuHbmToL2,
            stage: PrefetchStage::CentroidComputed,
            bytes: 2048,
            submit_timestamp_us: 0,
            estimated_latency_us: 5.0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("target_layer"));
        assert!(debug.contains("transfer_kind"));
    }

    #[test]
    fn entry_clone() {
        let entry = PipelinePrefetchEntry {
            target_layer: 3,
            start_token: 7,
            token_count: 2,
            transfer_kind: PrefetchTransferKind::Rdma,
            stage: PrefetchStage::AddressTranslated,
            bytes: 4096,
            submit_timestamp_us: 500,
            estimated_latency_us: 200.0,
        };
        let cloned = entry.clone();
        assert_eq!(entry.target_layer, cloned.target_layer);
        assert_eq!(entry.bytes, cloned.bytes);
    }

    // ── PrefetchPipeline builder ──

    #[test]
    fn with_pipeline_depth_min_one() {
        let pipeline = PrefetchPipeline::new(32).with_pipeline_depth(0);
        assert_eq!(pipeline.pipeline_depth, 1);
    }

    #[test]
    fn with_bandwidth_custom() {
        let pipeline = PrefetchPipeline::new(32).with_bandwidth(64.0, 200.0);
        assert!((pipeline.pcie_bandwidth_gbs - 64.0).abs() < 1e-6);
        assert!((pipeline.rdma_bandwidth_gbs - 200.0).abs() < 1e-6);
    }

    #[test]
    fn with_layer_compute_time_custom() {
        let pipeline = PrefetchPipeline::new(32).with_layer_compute_time(200.0);
        assert!((pipeline.layer_compute_time_us - 200.0).abs() < 1e-6);
    }

    // ── Advance layer behavior ──

    #[test]
    fn advance_multiple_centroids_submits_multiple_entries() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[1, 2, 3], 1024);
        // 3 centroids × 1 depth = 3 entries
        assert_eq!(pipeline.active_count(), 3);
    }

    #[test]
    fn advance_does_not_exceed_num_layers() {
        let mut pipeline = PrefetchPipeline::new(4).with_pipeline_depth(5);
        pipeline.advance_layer(&[0], 1024);
        // current_layer=1, target layers 2,3,4,5,6 → only 2,3 (4 layers total)
        let max_layer = pipeline.active_entries.iter().map(|e| e.target_layer).max().unwrap_or(0);
        assert!(max_layer < 4, "target layer {max_layer} exceeds num_layers");
    }

    #[test]
    fn is_prefetch_completed_for_unsubmitted_layer() {
        let pipeline = PrefetchPipeline::new(32);
        // No entries submitted → layer 0 is "completed" (no active entries for it)
        assert!(pipeline.is_prefetch_completed(0));
    }

    #[test]
    fn utilization_zero_before_advance() {
        let pipeline = PrefetchPipeline::new(32);
        assert_eq!(pipeline.utilization(), 0.0);
    }

    #[test]
    fn can_hide_latency_exact_boundary() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);
        // 2 × 100 = 200μs available; latency=200.0 → exactly equal → true (<=)
        assert!(pipeline.can_hide_latency(200.0));
    }

    // ── Reset preserves config ──

    #[test]
    fn reset_preserves_num_layers() {
        let mut pipeline = PrefetchPipeline::new(64);
        pipeline.advance_layer(&[0], 1024);
        pipeline.reset();
        assert_eq!(pipeline.num_layers, 64);
    }

    // ── PCIe and RDMA transfer types ──

    #[test]
    fn pcie_entries_for_near_layers() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_layer_compute_time(100.0);
        pipeline.advance_layer(&[0], 4096);
        let pcie = pipeline.active_entries.iter()
            .any(|e| e.transfer_kind == PrefetchTransferKind::PcieDma);
        assert!(pcie, "depth=5 should generate PCIe entries for layers > current+1");
    }

    #[test]
    fn rdma_entries_for_far_layers() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_layer_compute_time(100.0);
        pipeline.advance_layer(&[0], 4096);
        let rdma = pipeline.active_entries.iter()
            .any(|e| e.transfer_kind == PrefetchTransferKind::Rdma);
        assert!(rdma, "depth=10 should generate RDMA entries for distant layers");
    }

    #[test]
    fn entry_bytes_is_kv_doubled() {
        let block_size = 2048;
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], block_size);
        let entry = &pipeline.active_entries[0];
        assert_eq!(entry.bytes, block_size * 2, "bytes should be kv_block_size * 2 (K+V)");
    }

    // ── PrefetchTransferKind additional trait tests ──

    #[test]
    fn prefetch_transfer_kind_clone() {
        let a = PrefetchTransferKind::GpuHbmToL2;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn prefetch_transfer_kind_all_variants_distinct() {
        let variants = [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
        ];
        for (i, vi) in variants.iter().enumerate() {
            for (j, vj) in variants.iter().enumerate() {
                assert_eq!(i == j, vi == vj);
            }
        }
    }

    #[test]
    fn prefetch_transfer_kind_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        PrefetchTransferKind::PcieDma.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        PrefetchTransferKind::PcieDma.hash(&mut h2);
        let hash2 = h2.finish();

        assert_eq!(hash1, hash2, "equal values must produce equal hashes");
    }

    #[test]
    fn prefetch_transfer_kind_hash_in_set() {
        use std::collections::HashSet;
        let set: HashSet<PrefetchTransferKind> = [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
            PrefetchTransferKind::PcieDma, // duplicate
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn prefetch_transfer_kind_debug_all_variants() {
        assert!(format!("{:?}", PrefetchTransferKind::GpuHbmToL2).contains("GpuHbmToL2"));
        assert!(format!("{:?}", PrefetchTransferKind::PcieDma).contains("PcieDma"));
        assert!(format!("{:?}", PrefetchTransferKind::Rdma).contains("Rdma"));
    }

    // ── PrefetchStage additional trait tests ──

    #[test]
    fn prefetch_stage_copy() {
        let a = PrefetchStage::CentroidComputed;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn prefetch_stage_clone() {
        let a = PrefetchStage::TransferCompleted;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn prefetch_stage_ord_total_order() {
        let stages = [
            PrefetchStage::CentroidComputed,
            PrefetchStage::AddressTranslated,
            PrefetchStage::TransferSubmitted,
            PrefetchStage::TransferCompleted,
        ];
        for i in 0..stages.len() {
            for j in 0..stages.len() {
                assert_eq!(i < j, stages[i] < stages[j]);
            }
        }
    }

    #[test]
    fn prefetch_stage_debug_all_variants() {
        assert!(format!("{:?}", PrefetchStage::CentroidComputed).contains("CentroidComputed"));
        assert!(format!("{:?}", PrefetchStage::AddressTranslated).contains("AddressTranslated"));
        assert!(format!("{:?}", PrefetchStage::TransferSubmitted).contains("TransferSubmitted"));
        assert!(format!("{:?}", PrefetchStage::TransferCompleted).contains("TransferCompleted"));
    }

    #[test]
    fn prefetch_stage_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        PrefetchStage::TransferSubmitted.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        PrefetchStage::TransferSubmitted.hash(&mut h2);
        let hash2 = h2.finish();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn prefetch_stage_hash_in_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PrefetchStage::CentroidComputed, "stage0");
        map.insert(PrefetchStage::TransferCompleted, "stage3");
        assert_eq!(map.get(&PrefetchStage::CentroidComputed), Some(&"stage0"));
        assert_eq!(map.get(&PrefetchStage::TransferCompleted), Some(&"stage3"));
        assert_eq!(map.get(&PrefetchStage::AddressTranslated), None);
    }

    // ── Boundary: num_layers = 0 ──

    #[test]
    fn new_with_zero_layers() {
        let pipeline = PrefetchPipeline::new(0);
        assert_eq!(pipeline.num_layers, 0);
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
    }

    #[test]
    fn advance_with_zero_layers_no_entries() {
        let mut pipeline = PrefetchPipeline::new(0).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], 1024);
        // current_layer=1, target_layer=2 >= num_layers=0, no entries submitted
        assert_eq!(pipeline.active_count(), 0);
    }

    // ── Boundary: num_layers = 1 ──

    #[test]
    fn advance_with_single_layer_no_prefetch() {
        let mut pipeline = PrefetchPipeline::new(1).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], 1024);
        // current_layer=1, target_layer=2 >= num_layers=1
        assert_eq!(pipeline.active_count(), 0);
    }

    // ── Boundary: empty centroids ──

    #[test]
    fn advance_with_empty_centroids() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[], 1024);
        assert_eq!(pipeline.active_count(), 0);
        assert_eq!(pipeline.current_layer(), 1);
    }

    // ── Boundary: zero block size ──

    #[test]
    fn advance_with_zero_block_size() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], 0);
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(pipeline.active_entries[0].bytes, 0);
    }

    // ── Pipeline depth clamping ──

    #[test]
    fn pipeline_depth_clamps_to_num_layers() {
        let pipeline = PrefetchPipeline::new(4).with_pipeline_depth(10);
        assert_eq!(pipeline.pipeline_depth, 4);
    }

    #[test]
    fn pipeline_depth_minimum_is_one() {
        let pipeline = PrefetchPipeline::new(32).with_pipeline_depth(0);
        assert_eq!(pipeline.pipeline_depth, 1);
    }

    // ── advance_layer: last layer ──

    #[test]
    fn advance_to_last_layer_no_further_prefetch() {
        let mut pipeline = PrefetchPipeline::new(4).with_pipeline_depth(1);
        // advance from layer 0 to layer 1: target_layer=2 < 4 → 1 entry
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1);
        // advance from layer 1 to layer 2: target_layer=3 < 4 → 1 new entry
        pipeline.advance_layer(&[0], 1024);
        // advance from layer 2 to layer 3: target_layer=4 >= 4 → no new entry
        pipeline.advance_layer(&[0], 1024);
        let has_layer_4 = pipeline
            .active_entries
            .iter()
            .any(|e| e.target_layer >= 4);
        assert!(!has_layer_4, "no entries for layers >= num_layers");
    }

    // ── advance_layer: multiple rounds with completion ──

    #[test]
    fn advance_multiple_rounds_completes_old_entries() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        // GpuHbmToL2 latency = 5μs, so after 1 round (100μs) entries complete
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(pipeline.completed_count(), 0);

        pipeline.advance_layer(&[0], 1024);
        // First entry (latency 5μs) should have completed after 200μs total
        assert!(pipeline.completed_count() >= 1);
    }

    // ── is_prefetch_completed: active layer returns false ──

    #[test]
    fn is_prefetch_completed_returns_false_for_active_layer() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0); // very long compute so entries don't complete
        pipeline.advance_layer(&[0], 1024);
        // target_layer=2 has an active entry (current_layer=1, target=1+1=2)
        assert!(!pipeline.is_prefetch_completed(2));
    }

    #[test]
    fn is_prefetch_completed_returns_true_for_unrelated_layer() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], 1024);
        // layer 100 has no active entries
        assert!(pipeline.is_prefetch_completed(100));
    }

    // ── can_hide_latency ──

    #[test]
    fn can_hide_latency_zero() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        assert!(pipeline.can_hide_latency(0.0));
    }

    #[test]
    fn can_hide_latency_negative() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        // negative latency is nonsensical but should return true (<= 100)
        assert!(pipeline.can_hide_latency(-10.0));
    }

    // ── utilization with no completions ──

    #[test]
    fn utilization_after_first_advance_no_completions() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0); // very long compute → nothing completes
        pipeline.advance_layer(&[0], 1024);
        // completed_count=0, expected=1*1=1
        let util = pipeline.utilization();
        assert!((util - 0.0).abs() < 1e-6);
    }

    // ── reset after multiple advances ──

    #[test]
    fn reset_after_multiple_advances() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }
        assert!(pipeline.current_layer() > 0);
        assert!(pipeline.active_count() > 0 || pipeline.completed_count() > 0);

        pipeline.reset();
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
        assert_eq!(pipeline.completed_count(), 0);
    }

    // ── builder chain preserves all settings ──

    #[test]
    fn builder_chain_all_options() {
        let pipeline = PrefetchPipeline::new(48)
            .with_pipeline_depth(3)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(250.0);
        assert_eq!(pipeline.pipeline_depth, 3);
        assert!((pipeline.pcie_bandwidth_gbs - 64.0).abs() < 1e-6);
        assert!((pipeline.rdma_bandwidth_gbs - 200.0).abs() < 1e-6);
        assert!((pipeline.layer_compute_time_us - 250.0).abs() < 1e-6);
        assert_eq!(pipeline.num_layers, 48);
    }

    // ── transfer kind latency estimation ──

    #[test]
    fn pcie_latency_scales_with_block_size() {
        let mut small = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);
        small.advance_layer(&[0], 1024);

        let mut large = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);
        large.advance_layer(&[0], 8192);

        let small_pcie = small
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .map(|e| e.estimated_latency_us);
        let large_pcie = large
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .map(|e| e.estimated_latency_us);

        // Both should exist and larger block size should have higher latency
        assert!(small_pcie.is_some());
        assert!(large_pcie.is_some());
        assert!(
            large_pcie.unwrap() > small_pcie.unwrap(),
            "larger blocks should have higher PCIe latency"
        );
    }

    // ── entry stage initial value ──

    #[test]
    fn entry_initial_stage_is_centroid_computed() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&[0], 1024);
        let entry = &pipeline.active_entries[0];
        assert_eq!(entry.stage, PrefetchStage::CentroidComputed);
    }

    // ── entry submit timestamp ──

    #[test]
    fn entry_timestamp_matches_layer_compute_time() {
        let compute_us = 200.0f32;
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(compute_us);
        pipeline.advance_layer(&[0], 1024);
        // current_layer=1, timestamp = 1 * 200 = 200μs
        let entry = &pipeline.active_entries[0];
        assert_eq!(entry.submit_timestamp_us, 200);
    }

    // ── completed_count accumulation ──

    #[test]
    fn completed_count_accumulates_across_advances() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        // Advance 5 rounds; each GpuHbmToL2 entry (5μs latency) completes in 1 round
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }
        assert_eq!(
            pipeline.completed_count(),
            4, // rounds 2-5 each complete the previous entry
        );
    }

    // ── advance with multiple centroids and depth > 1 ──

    #[test]
    fn advance_multi_centroid_multi_depth() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(3);
        pipeline.advance_layer(&[0, 1], 1024);
        // 2 centroids × 3 depth = 6 entries
        assert_eq!(pipeline.active_count(), 6);
    }

    // ── utilization capped at 1.0 ──

    #[test]
    fn utilization_capped_at_one() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        // Advance many rounds — all entries complete, utilization should not exceed 1.0
        for _ in 0..100 {
            pipeline.advance_layer(&[0], 1024);
        }
        assert!(pipeline.utilization() <= 1.0);
    }

    // ── PrefetchStage Copy ensures independent values ──

    #[test]
    fn prefetch_stage_copy_independence() {
        let a = PrefetchStage::CentroidComputed;
        let b = a;
        // Verify the copy is independent — original concept proven by Copy semantics
        assert_eq!(b, PrefetchStage::CentroidComputed);
    }

    // ── PrefetchTransferKind Copy ensures independent values ──

    #[test]
    fn prefetch_transfer_kind_copy_independence() {
        let a = PrefetchTransferKind::GpuHbmToL2;
        let b = a;
        assert_eq!(b, PrefetchTransferKind::GpuHbmToL2);
    }

    // ── New edge-case tests ──

    #[test]
    fn entry_clone_preserves_float_fields() {
        let entry = PipelinePrefetchEntry {
            target_layer: 7,
            start_token: 3,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::Rdma,
            stage: PrefetchStage::AddressTranslated,
            bytes: 4096,
            submit_timestamp_us: 999,
            estimated_latency_us: 1.5,
        };
        let cloned = entry.clone();
        assert!((cloned.estimated_latency_us - 1.5).abs() < 1e-6);
        assert_eq!(cloned.submit_timestamp_us, 999);
        assert_eq!(cloned.transfer_kind, entry.transfer_kind);
        assert_eq!(cloned.stage, entry.stage);
    }

    #[test]
    fn entry_token_count_always_one() {
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(3);
        pipeline.advance_layer(&[10, 20, 30], 1024);
        for entry in &pipeline.active_entries {
            assert_eq!(entry.token_count, 1, "token_count must always be 1 per entry");
        }
    }

    #[test]
    fn prefetch_stage_ord_reverse_ordering() {
        assert!(PrefetchStage::TransferCompleted > PrefetchStage::TransferSubmitted);
        assert!(PrefetchStage::TransferSubmitted > PrefetchStage::AddressTranslated);
        assert!(PrefetchStage::AddressTranslated > PrefetchStage::CentroidComputed);
    }

    #[test]
    fn can_hide_latency_with_infinity() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);
        // available = 200μs, f32::INFINITY > 200 → false
        assert!(!pipeline.can_hide_latency(f32::INFINITY));
    }

    #[test]
    fn can_hide_latency_with_nan() {
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        // NaN <= anything is false, so can_hide_latency returns false
        assert!(!pipeline.can_hide_latency(f32::NAN));
    }

    #[test]
    fn entry_start_token_matches_centroid_input() {
        let centroids = [5, 15, 25];
        let mut pipeline = PrefetchPipeline::new(32).with_pipeline_depth(1);
        pipeline.advance_layer(&centroids, 1024);
        let tokens: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.start_token)
            .collect();
        assert_eq!(tokens, centroids);
    }

    #[test]
    fn transfer_kind_hash_distinct_for_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let variants = [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
        ];
        let hashes: Vec<u64> = variants
            .iter()
            .map(|v| {
                let mut h = DefaultHasher::new();
                v.hash(&mut h);
                h.finish()
            })
            .collect();
        // Hashes should be distinct (probabilistically near-certain for 3 values)
        assert_eq!(hashes.len(), 3, "all variants should hash");
        // No duplicate hashes
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "variants {i} and {j} should have distinct hashes");
            }
        }
    }

    #[test]
    fn prefetch_stage_hash_distinct_for_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let stages = [
            PrefetchStage::CentroidComputed,
            PrefetchStage::AddressTranslated,
            PrefetchStage::TransferSubmitted,
            PrefetchStage::TransferCompleted,
        ];
        let hashes: Vec<u64> = stages
            .iter()
            .map(|s| {
                let mut h = DefaultHasher::new();
                s.hash(&mut h);
                h.finish()
            })
            .collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "stages {i} and {j} should have distinct hashes");
            }
        }
    }

    #[test]
    fn reset_preserves_pipeline_depth_and_bandwidth() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(50.0);
        pipeline.advance_layer(&[0], 1024);
        pipeline.reset();
        assert_eq!(pipeline.pipeline_depth, 3);
        assert!((pipeline.pcie_bandwidth_gbs - 64.0).abs() < 1e-6);
        assert!((pipeline.rdma_bandwidth_gbs - 200.0).abs() < 1e-6);
        assert!((pipeline.layer_compute_time_us - 50.0).abs() < 1e-6);
    }

    #[test]
    fn advance_with_depth_two_tracks_both_target_layers() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1000000.0); // prevent completion
        pipeline.advance_layer(&[7], 1024);
        // current_layer=1, target layers: 2 and 3
        let targets: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        assert!(targets.contains(&2), "should have entry for layer 2");
        assert!(targets.contains(&3), "should have entry for layer 3");
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn advance_two_rounds_then_check_active_layers() {
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1.0); // tiny compute so entries persist (latencies >> 1μs)
        pipeline.advance_layer(&[0], 1024); // layer 0→1, targets 2,3
        pipeline.advance_layer(&[0], 1024); // layer 1→2, targets 3,4
        // With compute=1μs: timestamp advances 1μs per round, all entries have latency >= 5μs
        // so entries from round 1 (timestamp=1, elapsed=1) still active
        let active_layers: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        assert!(active_layers.contains(&2), "layer 2 entry from round 1 should still be active");
        assert!(active_layers.contains(&4), "layer 4 entry from round 2 should be active");
    }

    #[test]
    fn rdma_latency_scales_with_block_size() {
        let mut small = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);
        small.advance_layer(&[0], 1024);

        let mut large = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);
        large.advance_layer(&[0], 8192);

        let small_rdma = small
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .map(|e| e.estimated_latency_us);
        let large_rdma = large
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .map(|e| e.estimated_latency_us);
        assert!(small_rdma.is_some(), "should have RDMA entries");
        assert!(large_rdma.is_some(), "should have RDMA entries");
        assert!(
            large_rdma.unwrap() > small_rdma.unwrap(),
            "larger blocks should have higher RDMA latency"
        );
    }

    #[test]
    fn entry_debug_contains_all_field_names() {
        let entry = PipelinePrefetchEntry {
            target_layer: 2,
            start_token: 5,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferSubmitted,
            bytes: 2048,
            submit_timestamp_us: 300,
            estimated_latency_us: 10.0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("start_token"), "Debug output missing start_token");
        assert!(debug.contains("token_count"), "Debug output missing token_count");
        assert!(debug.contains("bytes"), "Debug output missing bytes");
        assert!(debug.contains("submit_timestamp_us"), "Debug output missing submit_timestamp_us");
        assert!(debug.contains("estimated_latency_us"), "Debug output missing estimated_latency_us");
    }

    #[test]
    fn pipeline_default_bandwidth_values() {
        let pipeline = PrefetchPipeline::new(32);
        assert!((pipeline.pcie_bandwidth_gbs - 32.0).abs() < 1e-6, "default PCIe should be 32 GB/s");
        assert!((pipeline.rdma_bandwidth_gbs - 100.0).abs() < 1e-6, "default RDMA should be 100 GB/s");
        assert!((pipeline.layer_compute_time_us - 100.0).abs() < 1e-6, "default compute should be 100 μs");
    }

    // ── 15 new tests ──

    #[test]
    fn advance_multiple_times_same_centroid_token() {
        // Arrange: pipeline with depth=1, repeatedly advance with the same centroid
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance 3 times using centroid token 42 each time
        for _ in 0..3 {
            pipeline.advance_layer(&[42], 1024);
        }

        // Assert: current_layer should be 3, and the active entry for the latest
        // round should target layer 4 (current_layer=3 + depth=1 = 4)
        assert_eq!(pipeline.current_layer(), 3);
        let has_layer_4 = pipeline
            .active_entries
            .iter()
            .any(|e| e.target_layer == 4);
        assert!(has_layer_4, "latest advance should target layer 4");
    }

    #[test]
    fn pcie_bandwidth_affects_latency_estimate() {
        // Arrange: two pipelines with different PCIe bandwidths
        let mut slow_pcie = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_bandwidth(16.0, 100.0) // 16 GB/s PCIe
            .with_layer_compute_time(100.0);
        let mut fast_pcie = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_bandwidth(64.0, 100.0) // 64 GB/s PCIe (4x faster)
            .with_layer_compute_time(100.0);

        // Act: advance both with the same block size
        slow_pcie.advance_layer(&[0], 4096);
        fast_pcie.advance_layer(&[0], 4096);

        // Assert: slower PCIe should produce higher latency for PCIe entries
        let slow_lat = slow_pcie
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .map(|e| e.estimated_latency_us);
        let fast_lat = fast_pcie
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .map(|e| e.estimated_latency_us);
        assert!(slow_lat.is_some(), "should have PCIe entry in slow pipeline");
        assert!(fast_lat.is_some(), "should have PCIe entry in fast pipeline");
        assert!(
            slow_lat.unwrap() > fast_lat.unwrap(),
            "slower PCIe bandwidth must produce higher latency (got {} vs {})",
            slow_lat.unwrap(),
            fast_lat.unwrap()
        );
    }

    #[test]
    fn rdma_bandwidth_affects_latency_estimate() {
        // Arrange: two pipelines with different RDMA bandwidths
        let mut slow_rdma = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_bandwidth(32.0, 50.0) // 50 GB/s RDMA
            .with_layer_compute_time(100.0);
        let mut fast_rdma = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_bandwidth(32.0, 200.0) // 200 GB/s RDMA (4x faster)
            .with_layer_compute_time(100.0);

        // Act
        slow_rdma.advance_layer(&[0], 4096);
        fast_rdma.advance_layer(&[0], 4096);

        // Assert: slower RDMA should produce higher latency for RDMA entries
        let slow_lat = slow_rdma
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .map(|e| e.estimated_latency_us);
        let fast_lat = fast_rdma
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .map(|e| e.estimated_latency_us);
        assert!(slow_lat.is_some(), "should have RDMA entry in slow pipeline");
        assert!(fast_lat.is_some(), "should have RDMA entry in fast pipeline");
        assert!(
            slow_lat.unwrap() > fast_lat.unwrap(),
            "slower RDMA bandwidth must produce higher latency (got {} vs {})",
            slow_lat.unwrap(),
            fast_lat.unwrap()
        );
    }

    #[test]
    fn advance_with_large_centroid_indices() {
        // Arrange: use very large token indices
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1);

        // Act: advance with large centroid token indices
        pipeline.advance_layer(&[1_000_000, 999_999_999], 1024);

        // Assert: entries should preserve the exact large token indices
        assert_eq!(pipeline.active_count(), 2);
        let tokens: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.start_token)
            .collect();
        assert!(tokens.contains(&1_000_000), "large centroid 1000000 should be preserved");
        assert!(tokens.contains(&999_999_999), "large centroid 999999999 should be preserved");
    }

    #[test]
    fn is_prefetch_completed_after_entry_removed_by_advance() {
        // Arrange: pipeline with fast compute so entries complete quickly
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance twice; first entry's latency (5μs) << compute (100μs)
        pipeline.advance_layer(&[0], 1024); // current_layer=1, target=2
        assert!(!pipeline.is_prefetch_completed(2), "layer 2 has active entry");

        pipeline.advance_layer(&[0], 1024); // current_layer=2, first entry completes
        let completed = pipeline.is_prefetch_completed(2);

        // Assert: layer 2 should now be "completed" (no active entries for it)
        assert!(
            completed,
            "layer 2 should be completed after its entry expired"
        );
    }

    #[test]
    fn entry_transfer_kind_hbm_for_adjacent_only() {
        // Arrange: pipeline with depth=1 so only current+1 is targeted
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act
        pipeline.advance_layer(&[0], 1024);

        // Assert: all entries with depth=1 target current_layer+1, which is adjacent → GpuHbmToL2
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(
            pipeline.active_entries[0].transfer_kind,
            PrefetchTransferKind::GpuHbmToL2,
            "depth=1 should only produce HBM→L2 transfers"
        );
    }

    #[test]
    fn advance_with_two_layers_only_depth_one() {
        // Arrange: exactly 2 layers, depth 1
        let mut pipeline = PrefetchPipeline::new(2)
            .with_pipeline_depth(1);

        // Act: advance from layer 0 to layer 1; target_layer = 2 >= num_layers=2
        pipeline.advance_layer(&[0], 1024);

        // Assert: no entries because target_layer (2) >= num_layers (2)
        assert_eq!(pipeline.active_count(), 0, "no prefetch targets remain in 2-layer model after first advance");
        assert_eq!(pipeline.current_layer(), 1);
    }

    #[test]
    fn reset_then_readvance_works_correctly() {
        // Arrange: advance a pipeline, reset, then advance again
        let mut pipeline = PrefetchPipeline::new(16)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        pipeline.advance_layer(&[0], 1024);
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.current_layer(), 2);

        // Act: reset and advance again
        pipeline.reset();
        assert_eq!(pipeline.current_layer(), 0);

        pipeline.advance_layer(&[5], 2048);

        // Assert: state is clean and new advance works
        assert_eq!(pipeline.current_layer(), 1);
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(pipeline.active_entries[0].start_token, 5);
        assert_eq!(pipeline.active_entries[0].bytes, 2048 * 2);
    }

    #[test]
    fn utilization_after_many_completions() {
        // Arrange: pipeline with fast compute (GpuHbmToL2 latency = 5μs << 100μs compute)
        let mut pipeline = PrefetchPipeline::new(100)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance 50 times; each prior entry completes (5μs << 100μs)
        for _ in 0..50 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: utilization should be close to 1.0 (49 completed / (50 * 1) = 0.98)
        let util = pipeline.utilization();
        assert!(
            util >= 0.9,
            "utilization should be high after many completions, got {util}"
        );
        assert!(util <= 1.0, "utilization should not exceed 1.0");
    }

    #[test]
    fn can_hide_latency_with_large_depth() {
        // Arrange: depth=10, compute=100μs → 1000μs available
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_layer_compute_time(100.0);

        // Act & Assert: large depth should hide larger latencies
        assert!(
            pipeline.can_hide_latency(900.0),
            "900μs should be hidden by 10×100μs pipeline"
        );
        assert!(
            !pipeline.can_hide_latency(1001.0),
            "1001μs should NOT be hidden by 10×100μs pipeline"
        );
    }

    #[test]
    fn advance_with_depth_exceeding_remaining_layers() {
        // Arrange: 10 layers, advance to layer 8, depth=5 (exceeds remaining)
        let mut pipeline = PrefetchPipeline::new(10)
            .with_pipeline_depth(5)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Advance to current_layer=8
        for _ in 0..8 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Act: advance to layer 9; remaining layers = only layer 9
        pipeline.advance_layer(&[0], 1024);

        // Assert: only entries for layer 9 should exist (not 10, 11, 12, 13)
        let max_target = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .max()
            .unwrap_or(0);
        assert!(
            max_target < 10,
            "no target should exceed num_layers=10, got max={max_target}"
        );
    }

    #[test]
    fn multiple_centroids_same_layer_produce_distinct_entries() {
        // Arrange: pipeline with depth=1 and 4 distinct centroids
        let centroids = [0, 100, 200, 300];
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&centroids, 1024);

        // Assert: 4 entries for the same target layer, each with a different start_token
        assert_eq!(pipeline.active_count(), 4);
        let target_layer = pipeline.active_entries[0].target_layer;
        for entry in &pipeline.active_entries {
            assert_eq!(
                entry.target_layer, target_layer,
                "all entries should target the same layer"
            );
        }
        let tokens: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.start_token)
            .collect();
        assert_eq!(tokens.len(), 4, "should have 4 distinct entries");
        assert!(tokens.contains(&0));
        assert!(tokens.contains(&100));
        assert!(tokens.contains(&200));
        assert!(tokens.contains(&300));
    }

    #[test]
    fn completed_count_zero_initially() {
        // Arrange & Act: fresh pipeline
        let pipeline = PrefetchPipeline::new(32);

        // Assert: no completions yet
        assert_eq!(
            pipeline.completed_count(), 0,
            "completed_count must start at 0"
        );
    }

    #[test]
    fn entry_bytes_doubled_for_various_block_sizes() {
        // Arrange: test several block sizes to verify bytes = 2 * block_size invariant
        let block_sizes = [512, 1024, 2048, 4096, 8192];

        for &block_size in &block_sizes {
            let mut pipeline = PrefetchPipeline::new(32)
                .with_pipeline_depth(1);
            pipeline.advance_layer(&[0], block_size);

            // Act & Assert: each entry's bytes should be exactly 2× block_size
            let entry = &pipeline.active_entries[0];
            assert_eq!(
                entry.bytes,
                block_size * 2,
                "bytes should be 2*{block_size} (K+V), got {}",
                entry.bytes
            );
        }
    }

    #[test]
    fn advance_timestamp_increases_monotonically() {
        // Arrange: pipeline with compute=250μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(250.0);

        // Act: advance 4 times
        let mut timestamps: Vec<u64> = Vec::new();
        for _ in 0..4 {
            pipeline.advance_layer(&[0], 1024);
            if let Some(entry) = pipeline.active_entries.last() {
                timestamps.push(entry.submit_timestamp_us);
            }
        }

        // Assert: timestamps must be strictly increasing
        for i in 1..timestamps.len() {
            assert!(
                timestamps[i] > timestamps[i - 1],
                "timestamp at round {i} ({}) must be > round {} ({})",
                timestamps[i],
                i - 1,
                timestamps[i - 1]
            );
        }
    }

    // ── 15 additional tests for uncovered edge cases ──

    #[test]
    fn transfer_kind_boundary_adjacent_is_hbm() {
        // Arrange: depth=1 targets only current_layer+1 → GpuHbmToL2
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act
        pipeline.advance_layer(&[0], 1024);

        // Assert: the single entry must be GpuHbmToL2 (target_layer = current_layer+1 = 2)
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(
            pipeline.active_entries[0].transfer_kind,
            PrefetchTransferKind::GpuHbmToL2,
            "current_layer+1 must use GpuHbmToL2"
        );
    }

    #[test]
    fn transfer_kind_boundary_near_layers_are_pcie() {
        // Arrange: depth=4 so we get layers current+2 and current+3 → PcieDma
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(4)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Act
        pipeline.advance_layer(&[0], 1024); // current_layer=1, targets: 2,3,4,5

        // Assert: layers 3,4 should be PcieDma (current+2, current+3)
        let pcie_layers: Vec<usize> = pipeline
            .active_entries
            .iter()
            .filter(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .map(|e| e.target_layer)
            .collect();
        assert!(
            pcie_layers.contains(&3),
            "current_layer+2 should be PcieDma, got layers: {pcie_layers:?}"
        );
        assert!(
            pcie_layers.contains(&4),
            "current_layer+3 should be PcieDma, got layers: {pcie_layers:?}"
        );
    }

    #[test]
    fn transfer_kind_boundary_far_layers_are_rdma() {
        // Arrange: depth=6 so we get layers current+4 and beyond → Rdma
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(6)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Act
        pipeline.advance_layer(&[0], 1024); // current_layer=1, targets: 2,3,4,5,6,7

        // Assert: layers 5,6,7 should be Rdma (current+4, current+5, current+6)
        let rdma_layers: Vec<usize> = pipeline
            .active_entries
            .iter()
            .filter(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .map(|e| e.target_layer)
            .collect();
        assert!(
            rdma_layers.contains(&5),
            "current_layer+4 should be Rdma, got layers: {rdma_layers:?}"
        );
        assert!(
            rdma_layers.contains(&6),
            "current_layer+5 should be Rdma, got layers: {rdma_layers:?}"
        );
        assert!(
            rdma_layers.contains(&7),
            "current_layer+6 should be Rdma, got layers: {rdma_layers:?}"
        );
    }

    #[test]
    fn hbm_latency_is_constant_five_microseconds() {
        // Arrange: pipelines with different bandwidths and block sizes
        let configs = [
            (32.0, 100.0, 1024),
            (64.0, 200.0, 8192),
            (16.0, 50.0, 4096),
        ];

        for (pcie, rdma, block) in configs {
            let mut pipeline = PrefetchPipeline::new(32)
                .with_pipeline_depth(1)
                .with_bandwidth(pcie, rdma)
                .with_layer_compute_time(100.0);

            // Act
            pipeline.advance_layer(&[0], block);

            // Assert: GpuHbmToL2 latency is always 5.0μs regardless of bandwidth/block
            let entry = &pipeline.active_entries[0];
            assert_eq!(
                entry.transfer_kind,
                PrefetchTransferKind::GpuHbmToL2,
                "depth=1 should produce HBM entry"
            );
            assert!(
                (entry.estimated_latency_us - 5.0).abs() < 1e-6,
                "HBM latency must be 5.0μs for pcie={pcie}, rdma={rdma}, block={block}, got {}",
                entry.estimated_latency_us
            );
        }
    }

    #[test]
    fn active_count_decreases_as_entries_complete() {
        // Arrange: pipeline where GpuHbmToL2 entries complete each round (5μs << 100μs)
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance once → 1 active entry
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1, "after first advance: 1 entry");

        // Advance again → old entry completes, new one submitted → 1 active
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1, "after second advance: old completed, new submitted");

        // Assert: 1 completion recorded
        assert_eq!(pipeline.completed_count(), 1, "one entry should have completed");
    }

    #[test]
    fn multiple_rounds_depth_two_completion_tracking() {
        // Arrange: depth=2, compute=100μs, HBM latency=5μs, PCIe latency depends on block
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        // Act: advance 5 rounds
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: entries from round 1 (timestamp=100) have had 400μs to complete
        // HBM (5μs) and PCIe entries should all be done by now
        assert!(
            pipeline.completed_count() >= 5,
            "should have at least 5 completions after 5 rounds with depth=2, got {}",
            pipeline.completed_count()
        );
    }

    #[test]
    fn utilization_with_depth_two() {
        // Arrange: pipeline with depth=2, compute=100μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        // Act: advance 10 times (all entries complete due to 5μs HBM latency << 100μs)
        for _ in 0..10 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: utilization = completed / (current_layer * pipeline_depth)
        // Expected ≈ 18 completed / (10 * 2) = 0.9
        let util = pipeline.utilization();
        assert!(
            util >= 0.0 && util <= 1.0,
            "utilization must be in [0,1], got {util}"
        );
    }

    #[test]
    fn completed_count_resets_to_zero() {
        // Arrange: build up some completions
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }
        let before = pipeline.completed_count();
        assert!(before > 0, "should have completions before reset");

        // Act
        pipeline.reset();

        // Assert
        assert_eq!(
            pipeline.completed_count(), 0,
            "completed_count must be 0 after reset"
        );
    }

    #[test]
    fn two_identical_pipelines_produce_same_entries() {
        // Arrange: two pipelines with identical config
        let config = || {
            PrefetchPipeline::new(32)
                .with_pipeline_depth(3)
                .with_bandwidth(48.0, 150.0)
                .with_layer_compute_time(100.0)
        };
        let mut p1 = config();
        let mut p2 = config();

        // Act: advance both with the same centroids
        let centroids = [0, 50, 100];
        p1.advance_layer(&centroids, 2048);
        p2.advance_layer(&centroids, 2048);

        // Assert: identical entries
        assert_eq!(p1.active_count(), p2.active_count());
        for (e1, e2) in p1.active_entries.iter().zip(p2.active_entries.iter()) {
            assert_eq!(e1.target_layer, e2.target_layer);
            assert_eq!(e1.start_token, e2.start_token);
            assert_eq!(e1.transfer_kind, e2.transfer_kind);
            assert!((e1.estimated_latency_us - e2.estimated_latency_us).abs() < 1e-6);
            assert_eq!(e1.bytes, e2.bytes);
        }
    }

    #[test]
    fn reset_then_rerun_produces_fresh_pattern() {
        // Arrange: run pipeline to layer 5, reset, rerun to layer 3
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }
        pipeline.reset();

        // Act: re-advance 3 times with different centroid
        for _ in 0..3 {
            pipeline.advance_layer(&[99], 2048);
        }

        // Assert: behaves like a fresh pipeline at layer 3
        assert_eq!(pipeline.current_layer(), 3);
        assert_eq!(pipeline.completed_count(), 2); // rounds 2 and 3 complete round 1 and 2 entries
        let entry = &pipeline.active_entries[0];
        assert_eq!(entry.start_token, 99, "centroid should be 99 after reset");
        assert_eq!(entry.bytes, 2048 * 2, "bytes should reflect new block_size");
    }

    #[test]
    fn prefetch_stage_partial_ord_transitivity() {
        // Arrange: three stages in order
        let a = PrefetchStage::CentroidComputed;
        let b = PrefetchStage::TransferSubmitted;
        let c = PrefetchStage::TransferCompleted;

        // Assert: transitivity: a < b && b < c → a < c
        assert!(a < b, "CentroidComputed < TransferSubmitted");
        assert!(b < c, "TransferSubmitted < TransferCompleted");
        assert!(a < c, "transitivity: CentroidComputed < TransferCompleted");
    }

    #[test]
    fn is_prefetch_completed_for_high_layer_after_many_advances() {
        // Arrange: pipeline with depth=1, advance far
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance to layer 10
        for _ in 0..10 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: layer 100 was never targeted and has no active entries
        assert!(
            pipeline.is_prefetch_completed(100),
            "layer 100 was never targeted, should be considered completed"
        );
        // Layer 10 itself may or may not have an active entry depending on timing
        // but layer 11 is the target of the latest advance and should be active
        assert!(
            !pipeline.is_prefetch_completed(11),
            "layer 11 is the latest prefetch target and should still be active"
        );
    }

    #[test]
    fn pipeline_with_very_large_num_layers() {
        // Arrange: extremely large model
        let mut pipeline = PrefetchPipeline::new(1000)
            .with_pipeline_depth(10)
            .with_layer_compute_time(100.0);

        // Act: advance from layer 0 to layer 5
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: should have up to 10 entries per centroid per round (minus completed)
        assert!(
            pipeline.active_count() > 0,
            "should have active entries with 1000-layer model"
        );
        let max_target = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .max()
            .unwrap_or(0);
        // After 5 advances, current_layer=5, max target = 5+10 = 15
        assert!(
            max_target <= 15,
            "max target should not exceed current_layer + depth = 15, got {max_target}"
        );
    }

    #[test]
    fn advance_retain_preserves_uncompleted_entry_integrity() {
        // Arrange: pipeline where entries persist across advances.
        // Use depth=2 with large compute time. After round 1: target layers 2 (HBM, 5μs) and 3 (PCIe).
        // After round 2: PCIe entry from round 1 may still be active (latency > compute time difference).
        // Key insight: the pipeline only REMOVES completed entries via retain() — it never mutates
        // fields of surviving entries. We verify surviving entries keep their original field values.
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_bandwidth(0.001, 0.001) // extremely slow bandwidth → huge latency → entries persist
            .with_layer_compute_time(1.0); // tiny compute → entries take many rounds to complete

        // Act: first advance
        pipeline.advance_layer(&[42], 2048);
        assert_eq!(pipeline.active_count(), 3, "depth=3 should produce 3 entries");

        // Clone all entries from round 1 for later comparison
        let round1_entries: Vec<PipelinePrefetchEntry> = pipeline.active_entries.clone();

        // Second advance — entries from round 1 should survive (huge latency >> tiny compute)
        pipeline.advance_layer(&[99], 1024);

        // Assert: entries from round 1 should still exist with unchanged fields
        for r1 in &round1_entries {
            let survivor = pipeline
                .active_entries
                .iter()
                .find(|e| e.start_token == r1.start_token && e.target_layer == r1.target_layer);
            assert!(
                survivor.is_some(),
                "round 1 entry (token={}, layer={}) should survive",
                r1.start_token,
                r1.target_layer
            );
            let s = survivor.unwrap();
            assert_eq!(s.bytes, r1.bytes, "bytes should be unchanged");
            assert_eq!(s.transfer_kind, r1.transfer_kind, "transfer_kind should be unchanged");
            assert!((s.estimated_latency_us - r1.estimated_latency_us).abs() < 1e-6,
                "latency should be unchanged");
        }
    }

    #[test]
    fn advance_with_large_num_centroids_scales_linearly() {
        // Arrange: 50 centroids, depth=1
        let centroids: Vec<usize> = (0..50).collect();
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&centroids, 1024);

        // Assert: exactly 50 entries (one per centroid)
        assert_eq!(
            pipeline.active_count(),
            50,
            "should have exactly 50 entries for 50 centroids with depth=1"
        );

        // All entries target the same layer
        let target_layers: std::collections::HashSet<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        assert_eq!(
            target_layers.len(),
            1,
            "all entries should target a single layer with depth=1"
        );
    }

    // ── 13 additional edge-case tests ──

    #[test]
    fn advance_beyond_last_layer_no_panic() {
        // Arrange: pipeline with 3 layers, advance past the end
        let mut pipeline = PrefetchPipeline::new(3)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance 5 times (goes past num_layers=3)
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: current_layer=5 but no entries should target layers >= 3
        assert_eq!(pipeline.current_layer(), 5);
        let has_out_of_range = pipeline
            .active_entries
            .iter()
            .any(|e| e.target_layer >= 3);
        assert!(
            !has_out_of_range,
            "no entry should target layer >= num_layers"
        );
    }

    #[test]
    fn first_advance_timestamp_is_compute_time() {
        // Arrange: pipeline with 500μs compute time
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(500.0);

        // Act: first advance (layer 0 → 1)
        pipeline.advance_layer(&[0], 1024);

        // Assert: submit_timestamp_us = current_layer * compute_time = 1 * 500 = 500
        let entry = &pipeline.active_entries[0];
        assert_eq!(
            entry.submit_timestamp_us, 500,
            "first advance timestamp should be 1 * compute_time"
        );
    }

    #[test]
    fn is_prefetch_completed_returns_true_for_all_layers_after_reset() {
        // Arrange: advance pipeline then reset
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_layer_compute_time(1000000.0);
        pipeline.advance_layer(&[0, 1, 2], 1024);
        // Before reset, layer 2 has active entries
        assert!(!pipeline.is_prefetch_completed(2));

        // Act
        pipeline.reset();

        // Assert: no active entries, so all layers should be "completed"
        assert!(pipeline.is_prefetch_completed(0));
        assert!(pipeline.is_prefetch_completed(2));
        assert!(pipeline.is_prefetch_completed(31));
        assert!(pipeline.is_prefetch_completed(999));
    }

    #[test]
    fn utilization_is_zero_immediately_after_reset() {
        // Arrange: build up state then reset
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        for _ in 0..10 {
            pipeline.advance_layer(&[0], 1024);
        }
        assert!(pipeline.utilization() > 0.0);

        // Act
        pipeline.reset();

        // Assert: current_layer=0, utilization returns 0.0
        assert_eq!(
            pipeline.utilization(),
            0.0,
            "utilization must be 0.0 after reset (current_layer=0)"
        );
    }

    #[test]
    fn can_hide_latency_with_zero_compute_time() {
        // Arrange: pipeline with zero compute time
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_layer_compute_time(0.0);

        // Act & Assert: available_time = 3 * 0.0 = 0.0
        assert!(
            !pipeline.can_hide_latency(1.0),
            "1.0μs cannot be hidden when compute time is 0"
        );
        assert!(
            pipeline.can_hide_latency(0.0),
            "0μs should be hideable when compute time is 0"
        );
    }

    #[test]
    fn entries_from_consecutive_rounds_target_overlapping_layers() {
        // Arrange: pipeline with depth=2, extremely slow bandwidth to keep entries alive
        // and tiny compute time so timestamps grow very slowly relative to latency.
        // With bandwidth=0.001 GB/s and block=1024: PCIe latency = (2048/1e9)/0.001 * 1e6 = 2048μs
        // With compute=1.0μs: after 2 rounds, elapsed for round 1 entries = 1μs << 2048μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_bandwidth(0.001, 0.001)
            .with_layer_compute_time(1.0);

        // Act: advance twice with same centroid
        pipeline.advance_layer(&[10], 1024); // current_layer=1, targets: 2(HBM,5μs), 3(PCIe,2048μs)
        pipeline.advance_layer(&[10], 1024); // current_layer=2, targets: 3(PCIe,2048μs), 4(PCIe,2048μs)

        // Assert: layer 3 should have two entries (one from each round)
        // Round 1 HBM entry (5μs) completes (elapsed=1μs < 5μs actually), but
        // round 1 PCIe entry (2048μs) survives (elapsed=1μs << 2048μs)
        let layer3_entries: Vec<&PipelinePrefetchEntry> = pipeline
            .active_entries
            .iter()
            .filter(|e| e.target_layer == 3)
            .collect();
        assert_eq!(
            layer3_entries.len(),
            2,
            "layer 3 should have 2 entries (one per round), got {}",
            layer3_entries.len()
        );
        // The two entries should have different submit timestamps
        assert_ne!(
            layer3_entries[0].submit_timestamp_us,
            layer3_entries[1].submit_timestamp_us,
            "entries from different rounds must have different timestamps"
        );
    }

    #[test]
    fn completed_count_tracks_multiple_entries_per_round() {
        // Arrange: depth=3, compute=100μs. HBM(5μs) and PCIe entries all complete quickly
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);

        // Act: advance twice. Round 1 submits 3 entries (targets 2,3,4).
        // After round 2 (timestamp=200μs), round 1 entries (timestamp=100μs, elapsed=100μs)
        // should all be completed since even PCIe latency for small blocks << 100μs.
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 3);
        assert_eq!(pipeline.completed_count(), 0);

        pipeline.advance_layer(&[0], 1024);

        // Assert: at least the 3 entries from round 1 should have completed
        assert!(
            pipeline.completed_count() >= 3,
            "all 3 entries from round 1 should complete, got {}",
            pipeline.completed_count()
        );
    }

    #[test]
    fn multiple_centroids_with_depth_two_produces_correct_count() {
        // Arrange: 4 centroids × depth 2 = 8 entries
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[10, 20, 30, 40], 1024);

        // Assert: 4 centroids × 2 depth = 8
        assert_eq!(
            pipeline.active_count(),
            8,
            "4 centroids × depth=2 should produce 8 entries"
        );

        // Verify layer distribution: 4 entries for target_layer=2, 4 for target_layer=3
        let layer2_count = pipeline
            .active_entries
            .iter()
            .filter(|e| e.target_layer == 2)
            .count();
        let layer3_count = pipeline
            .active_entries
            .iter()
            .filter(|e| e.target_layer == 3)
            .count();
        assert_eq!(layer2_count, 4, "4 entries should target layer 2");
        assert_eq!(layer3_count, 4, "4 entries should target layer 3");
    }

    #[test]
    fn pipeline_depth_clamped_to_one_for_single_layer_model() {
        // Arrange: model with only 1 layer, request depth=5
        let pipeline = PrefetchPipeline::new(1).with_pipeline_depth(5);

        // Assert: depth should be clamped to num_layers=1
        assert_eq!(
            pipeline.pipeline_depth, 1,
            "depth must be clamped to num_layers=1"
        );
    }

    #[test]
    fn advance_with_single_centroid_depth_three_all_transfer_kinds_present() {
        // Arrange: depth=6, should produce HBM, PCIe, and RDMA entries
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(6)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[0], 4096);

        // Assert: verify all three transfer kinds are present
        let kinds: std::collections::HashSet<PrefetchTransferKind> = pipeline
            .active_entries
            .iter()
            .map(|e| e.transfer_kind)
            .collect();
        assert_eq!(
            kinds.len(),
            3,
            "depth=6 should produce all 3 transfer kinds, got {:?}",
            kinds
        );
        assert!(kinds.contains(&PrefetchTransferKind::GpuHbmToL2));
        assert!(kinds.contains(&PrefetchTransferKind::PcieDma));
        assert!(kinds.contains(&PrefetchTransferKind::Rdma));
    }

    #[test]
    fn advance_at_last_layer_produces_no_new_entries() {
        // Arrange: 10-layer model, advance to the last layer
        let mut pipeline = PrefetchPipeline::new(10)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance 9 times to reach current_layer=9
        for _ in 0..9 {
            pipeline.advance_layer(&[0], 1024);
        }
        assert_eq!(pipeline.current_layer(), 9);

        // Assert: the latest advance (layer 8→9) targets layer 10 >= num_layers → no new entry
        // But entries from earlier may still be active or completed.
        // Advance one more time: layer 9→10, target=11 >= 10
        pipeline.advance_layer(&[0], 1024);
        let has_out_of_range = pipeline
            .active_entries
            .iter()
            .any(|e| e.target_layer >= 10);
        assert!(
            !has_out_of_range,
            "no entry should target layers >= num_layers after advancing past the end"
        );
    }

    #[test]
    fn entry_debug_output_contains_stage_field() {
        // Arrange: create an entry at each stage and verify Debug output
        for stage in [
            PrefetchStage::CentroidComputed,
            PrefetchStage::AddressTranslated,
            PrefetchStage::TransferSubmitted,
            PrefetchStage::TransferCompleted,
        ] {
            let entry = PipelinePrefetchEntry {
                target_layer: 0,
                start_token: 0,
                token_count: 1,
                transfer_kind: PrefetchTransferKind::GpuHbmToL2,
                stage,
                bytes: 0,
                submit_timestamp_us: 0,
                estimated_latency_us: 0.0,
            };
            let debug = format!("{entry:?}");
            assert!(
                debug.contains("stage"),
                "Debug output must contain 'stage' field for {:?}",
                stage
            );
        }
    }

    #[test]
    fn utilization_formula_matches_manual_calculation() {
        // Arrange: pipeline with depth=1, compute=100μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance exactly 4 times
        for _ in 0..4 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: after 4 advances, current_layer=4
        assert_eq!(pipeline.current_layer(), 4);
        // GpuHbmToL2 latency=5μs, compute=100μs. After each advance (except the first),
        // the previous entry completes. So completed_count should be 3.
        let completed = pipeline.completed_count();
        assert_eq!(completed, 3, "3 entries should have completed (rounds 1-3)");

        // utilization = completed / (current_layer * pipeline_depth) = 3 / (4 * 1) = 0.75
        let expected_util = 0.75f32;
        let actual_util = pipeline.utilization();
        assert!(
            (actual_util - expected_util).abs() < 1e-6,
            "utilization should be exactly 0.75, got {actual_util}"
        );
    }

    // ── 13 new edge-case tests (wave-13) ──

    #[test]
    fn advance_far_beyond_max_layers_current_layer_keeps_incrementing() {
        // Arrange: 4-layer model, advance well past the end
        let mut pipeline = PrefetchPipeline::new(4)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance 20 times (current_layer goes to 20)
        for _ in 0..20 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: current_layer increments freely but no entries target valid layers
        assert_eq!(pipeline.current_layer(), 20);
        assert_eq!(
            pipeline.active_count(),
            0,
            "no active entries when current_layer >> num_layers"
        );
    }

    #[test]
    fn utilization_at_boundary_with_known_completion_rate() {
        // Arrange: pipeline with depth=1, compute=100μs (GpuHbmToL2 latency=5μs << 100μs)
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance exactly 4 times
        for _ in 0..4 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: after 4 advances with fast completion (5μs latency << 100μs compute)
        // completed_count=3 (entries from rounds 1-3 complete by round 4)
        // utilization = 3 / (4 * 1) = 0.75
        let util = pipeline.utilization();
        assert!(
            util >= 0.0 && util <= 1.0,
            "utilization must be in [0,1], got {util}"
        );
        assert!(
            (util - 0.75).abs() < 1e-6,
            "utilization should be 0.75 after 4 rounds with depth=1, got {util}"
        );
    }

    #[test]
    fn depth_clamping_with_zero_layers() {
        // Arrange: 0-layer model, request depth=10
        let pipeline = PrefetchPipeline::new(0).with_pipeline_depth(10);

        // Assert: depth clamped to min(10, 0) = 0, but max(0, 1) = 1
        // Implementation: depth.max(1).min(self.num_layers) = max(1,0).min(0) = 1.min(0) = 0
        // Wait: 1.min(0) = 0, but max(1, 10) = 10, then min(10, 0) = 0
        // Actually depth=10, max(10,1)=10, min(10,0)=0
        assert_eq!(
            pipeline.pipeline_depth, 0,
            "depth should be clamped to num_layers=0"
        );
    }

    #[test]
    fn entry_debug_format_includes_all_seven_fields() {
        // Arrange: construct an entry and verify Debug output contains all field names
        let entry = PipelinePrefetchEntry {
            target_layer: 99,
            start_token: 0,
            token_count: 0,
            transfer_kind: PrefetchTransferKind::Rdma,
            stage: PrefetchStage::AddressTranslated,
            bytes: 0,
            submit_timestamp_us: 0,
            estimated_latency_us: 0.0,
        };

        // Act
        let debug = format!("{entry:?}");

        // Assert: all 8 fields must appear in Debug output
        assert!(debug.contains("target_layer"), "missing target_layer");
        assert!(debug.contains("start_token"), "missing start_token");
        assert!(debug.contains("token_count"), "missing token_count");
        assert!(debug.contains("transfer_kind"), "missing transfer_kind");
        assert!(debug.contains("stage"), "missing stage");
        assert!(debug.contains("bytes"), "missing bytes");
        assert!(debug.contains("submit_timestamp_us"), "missing submit_timestamp_us");
        assert!(debug.contains("estimated_latency_us"), "missing estimated_latency_us");
    }

    #[test]
    fn pipeline_config_fields_are_readable_after_construction() {
        // Arrange: construct pipeline with all builder options
        let pipeline = PrefetchPipeline::new(64)
            .with_pipeline_depth(4)
            .with_bandwidth(96.0, 300.0)
            .with_layer_compute_time(75.0);

        // Assert: all fields accessible and match config
        assert_eq!(pipeline.num_layers, 64);
        assert_eq!(pipeline.pipeline_depth, 4);
        assert!((pipeline.pcie_bandwidth_gbs - 96.0).abs() < 1e-6);
        assert!((pipeline.rdma_bandwidth_gbs - 300.0).abs() < 1e-6);
        assert!((pipeline.layer_compute_time_us - 75.0).abs() < 1e-6);
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
        assert_eq!(pipeline.completed_count(), 0);
    }

    #[test]
    fn completed_count_at_boundary_with_large_depth() {
        // Arrange: depth=10, compute=100μs. After 3 rounds with tiny block size
        // all entries (HBM=5μs, PCIe small, RDMA small) should complete
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(10)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(100.0);

        // Act: advance 3 times with small block
        pipeline.advance_layer(&[0], 256); // round 1: 10 entries submitted
        pipeline.advance_layer(&[0], 256); // round 2: 10 more, round 1 entries complete
        pipeline.advance_layer(&[0], 256); // round 3: 10 more, round 2 entries complete

        // Assert: round 1 entries (timestamp=100μs, elapsed after round 3 = 300-100=200μs)
        // HBM (5μs), PCIe (0.512μs for 512B/32GB/s), all << 200μs → all complete
        assert!(
            pipeline.completed_count() >= 10,
            "round 1 entries should all complete, got {}",
            pipeline.completed_count()
        );
    }

    #[test]
    fn entry_fields_are_independent_between_clones() {
        // Arrange: create an entry and clone it
        let original = PipelinePrefetchEntry {
            target_layer: 5,
            start_token: 100,
            token_count: 3,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferSubmitted,
            bytes: 8192,
            submit_timestamp_us: 5000,
            estimated_latency_us: 123.456,
        };
        let mut cloned = original.clone();

        // Act: mutate the clone's stage
        cloned.stage = PrefetchStage::TransferCompleted;
        cloned.bytes = 0;

        // Assert: original is unchanged
        assert_eq!(original.stage, PrefetchStage::TransferSubmitted);
        assert_eq!(original.bytes, 8192);
        assert_eq!(cloned.stage, PrefetchStage::TransferCompleted);
        assert_eq!(cloned.bytes, 0);
    }

    #[test]
    fn pipeline_state_transition_from_empty_to_active_to_completed() {
        // Arrange: single centroid, depth=1, fast completion
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // State: empty
        assert_eq!(pipeline.active_count(), 0);
        assert_eq!(pipeline.completed_count(), 0);

        // Act: transition to active
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(pipeline.completed_count(), 0);
        assert_eq!(pipeline.current_layer(), 1);

        // Act: transition to completed (previous entry completes)
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.completed_count(), 1);
        assert_eq!(pipeline.current_layer(), 2);
        // Still 1 active (new entry submitted)
        assert_eq!(pipeline.active_count(), 1);
    }

    #[test]
    fn empty_pipeline_utilization_is_zero() {
        // Arrange: fresh pipeline with no advances
        let pipeline = PrefetchPipeline::new(32);

        // Act: call utilization without any advance
        let util = pipeline.utilization();

        // Assert: should return 0.0 (guard: current_layer == 0)
        assert!(
            (util - 0.0).abs() < 1e-6,
            "empty pipeline utilization must be 0.0, got {util}"
        );
    }

    #[test]
    fn empty_pipeline_is_prefetch_completed_for_any_layer() {
        // Arrange: fresh pipeline
        let pipeline = PrefetchPipeline::new(32);

        // Act & Assert: no entries exist, so any layer is "completed"
        assert!(pipeline.is_prefetch_completed(0));
        assert!(pipeline.is_prefetch_completed(15));
        assert!(pipeline.is_prefetch_completed(31));
        assert!(pipeline.is_prefetch_completed(1000));
    }

    #[test]
    fn advance_with_zero_block_size_produces_zero_byte_pcie_and_rdma() {
        // Arrange: depth=6, block_size=0
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(6)
            .with_layer_compute_time(100.0);

        // Act
        pipeline.advance_layer(&[0], 0);

        // Assert: all entries have bytes=0 regardless of transfer kind
        assert_eq!(pipeline.active_count(), 6);
        for entry in &pipeline.active_entries {
            assert_eq!(
                entry.bytes, 0,
                "bytes should be 0 for zero block_size at layer {}",
                entry.target_layer
            );
        }
    }

    #[test]
    fn pipeline_depth_one_with_many_layers_only_targets_next_layer() {
        // Arrange: 100-layer model, depth=1
        let mut pipeline = PrefetchPipeline::new(100)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Act: advance 10 times
        for _ in 0..10 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: after 10 advances (current_layer=10), only layer 11 should be targeted
        // Earlier entries for layers 2-10 should have been completed or submitted earlier
        // but with compute=1e6 only the very latest entries survive (elapsed=1e6, latency=5μs)
        // Actually all entries complete because latency=5μs << 1e6μs, so only the last
        // round's entry is active. Let's check: latest entry targets layer 11
        let targets: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        // With depth=1 only one target per advance
        assert!(
            targets.contains(&11),
            "latest advance should target layer 11, got {:?}",
            targets
        );
    }

    #[test]
    fn utilization_after_single_advance_no_completions() {
        // Arrange: pipeline with very long compute so nothing completes
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Act: single advance
        pipeline.advance_layer(&[0], 1024);

        // Assert: completed=0, current_layer=1, expected=1*1=1, util=0.0
        assert_eq!(pipeline.completed_count(), 0);
        let util = pipeline.utilization();
        assert!(
            (util - 0.0).abs() < 1e-6,
            "utilization must be 0.0 with no completions, got {util}"
        );
    }

    // ── 13 new tests (wave-14) ──

    #[test]
    fn prefetch_transfer_kind_has_exactly_three_variants() {
        // Arrange: enumerate all known variants
        let all_variants = [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
        ];

        // Assert: exactly 3 distinct variants exist
        assert_eq!(all_variants.len(), 3, "PrefetchTransferKind must have exactly 3 variants");
        for (i, vi) in all_variants.iter().enumerate() {
            for (j, vj) in all_variants.iter().enumerate() {
                assert_eq!(i == j, vi == vj, "variant {i} and {j} must be distinct");
            }
        }
    }

    #[test]
    fn prefetch_stage_has_exactly_four_variants() {
        // Arrange: enumerate all known PrefetchStage variants in ordinal order
        let all_stages = [
            PrefetchStage::CentroidComputed,
            PrefetchStage::AddressTranslated,
            PrefetchStage::TransferSubmitted,
            PrefetchStage::TransferCompleted,
        ];

        // Assert: exactly 4 distinct variants exist
        assert_eq!(all_stages.len(), 4, "PrefetchStage must have exactly 4 variants");
        for (i, si) in all_stages.iter().enumerate() {
            for (j, sj) in all_stages.iter().enumerate() {
                assert_eq!(i == j, si == sj, "stage {i} and {j} must be distinct");
            }
        }
    }

    #[test]
    fn pipeline_depth_equals_num_layers_is_valid() {
        // Arrange: request depth exactly equal to num_layers
        let pipeline = PrefetchPipeline::new(16).with_pipeline_depth(16);

        // Assert: depth should be exactly 16 (not clamped down)
        assert_eq!(
            pipeline.pipeline_depth, 16,
            "depth equal to num_layers should not be clamped"
        );
    }

    #[test]
    fn with_bandwidth_zero_does_not_panic() {
        // Arrange: set zero bandwidth (degenerate case)
        let pipeline = PrefetchPipeline::new(32).with_bandwidth(0.0, 0.0);

        // Assert: construction succeeds and fields are zero
        assert!(
            (pipeline.pcie_bandwidth_gbs - 0.0).abs() < 1e-6,
            "PCIe bandwidth should be 0.0"
        );
        assert!(
            (pipeline.rdma_bandwidth_gbs - 0.0).abs() < 1e-6,
            "RDMA bandwidth should be 0.0"
        );
    }

    #[test]
    fn with_bandwidth_negative_stored_as_is() {
        // Arrange: set negative bandwidth (nonsensical but allowed by builder)
        let pipeline = PrefetchPipeline::new(32).with_bandwidth(-10.0, -20.0);

        // Assert: negative values are stored verbatim (builder has no validation)
        assert!(
            (pipeline.pcie_bandwidth_gbs - (-10.0)).abs() < 1e-6,
            "PCIe bandwidth should be -10.0"
        );
        assert!(
            (pipeline.rdma_bandwidth_gbs - (-20.0)).abs() < 1e-6,
            "RDMA bandwidth should be -20.0"
        );
    }

    #[test]
    fn with_layer_compute_time_negative_stored_as_is() {
        // Arrange: set negative compute time (nonsensical but allowed by builder)
        let pipeline = PrefetchPipeline::new(32).with_layer_compute_time(-50.0);

        // Assert: negative value is stored verbatim (builder has no validation)
        assert!(
            (pipeline.layer_compute_time_us - (-50.0)).abs() < 1e-6,
            "compute time should be -50.0"
        );
    }

    #[test]
    fn can_hide_latency_with_small_epsilon_below_boundary() {
        // Arrange: depth=3, compute=100μs → available = 300μs
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_layer_compute_time(100.0);

        // Act & Assert: latency just below 300μs should be hideable
        assert!(
            pipeline.can_hide_latency(299.999),
            "299.999μs should be hideable by 300μs available"
        );
        assert!(
            !pipeline.can_hide_latency(300.001),
            "300.001μs should NOT be hideable by 300μs available"
        );
    }

    #[test]
    fn entry_manual_construction_with_arbitrary_token_count() {
        // Arrange: manually construct an entry with token_count=8 (not the default 1)
        // The advance_layer always sets token_count=1, but the struct allows any value
        let entry = PipelinePrefetchEntry {
            target_layer: 10,
            start_token: 0,
            token_count: 8,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferCompleted,
            bytes: 16384,
            submit_timestamp_us: 5000,
            estimated_latency_us: 42.0,
        };

        // Assert: all fields are stored exactly as provided
        assert_eq!(entry.target_layer, 10);
        assert_eq!(entry.token_count, 8, "token_count should accept values other than 1");
        assert_eq!(entry.bytes, 16384);
        assert_eq!(entry.stage, PrefetchStage::TransferCompleted);
    }

    #[test]
    fn advance_creates_entries_for_each_depth_level_with_correct_targets() {
        // Arrange: depth=4, 1 centroid → should create 4 entries targeting layers 2,3,4,5
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(4)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Act: advance once (current_layer becomes 1)
        pipeline.advance_layer(&[0], 1024);

        // Assert: 4 entries with distinct target layers 2,3,4,5
        assert_eq!(pipeline.active_count(), 4);
        let targets: std::collections::HashSet<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        assert_eq!(targets.len(), 4, "should have 4 distinct target layers");
        for expected in 2..=5 {
            assert!(
                targets.contains(&expected),
                "target layer {expected} should be present"
            );
        }
    }

    #[test]
    fn advance_layer_increments_current_layer_by_exactly_one() {
        // Arrange: pipeline at layer 0
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);
        assert_eq!(pipeline.current_layer(), 0);

        // Act: advance 7 times
        for expected in 1..=7 {
            pipeline.advance_layer(&[0], 1024);
            assert_eq!(
                pipeline.current_layer(),
                expected,
                "current_layer should be {expected} after {expected} advances"
            );
        }

        // Assert: final layer is 7
        assert_eq!(pipeline.current_layer(), 7);
    }

    #[test]
    fn entry_construction_with_all_transfer_kinds() {
        // Arrange: construct entries with each transfer kind
        let kinds = [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
        ];

        for (idx, &kind) in kinds.iter().enumerate() {
            // Act: construct entry with this transfer kind
            let entry = PipelinePrefetchEntry {
                target_layer: idx,
                start_token: 0,
                token_count: 1,
                transfer_kind: kind,
                stage: PrefetchStage::CentroidComputed,
                bytes: 1024,
                submit_timestamp_us: 0,
                estimated_latency_us: 10.0,
            };

            // Assert: kind is preserved
            assert_eq!(
                entry.transfer_kind, kind,
                "transfer_kind should be preserved for variant {idx}"
            );
        }
    }

    #[test]
    fn advance_with_compute_time_one_microsecond_entries_complete_slowly() {
        // Arrange: very short compute time (1μs), depth=1
        // GpuHbmToL2 latency = 5μs. After 1 round (elapsed=1μs), entry is NOT done (1 < 5).
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1.0);

        // Act: advance once
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1, "1 entry submitted");
        assert_eq!(pipeline.completed_count(), 0, "nothing completed yet (elapsed=1μs < 5μs)");

        // Advance again: timestamp=2μs, elapsed for round 1 entry = 1μs < 5μs → still active
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(
            pipeline.completed_count(), 0,
            "entry should not complete: elapsed=1μs < latency=5μs"
        );

        // Advance 4 more times (total 6): timestamp=6μs, elapsed for round 1 entry = 5μs >= 5μs
        for _ in 0..4 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: now some entries should have completed
        assert!(
            pipeline.completed_count() > 0,
            "entries should start completing after enough rounds"
        );
    }

    #[test]
    fn reset_does_not_affect_builder_fields() {
        // Arrange: configure pipeline with specific builder values
        let mut pipeline = PrefetchPipeline::new(48)
            .with_pipeline_depth(5)
            .with_bandwidth(80.0, 250.0)
            .with_layer_compute_time(300.0);

        // Act: advance several times then reset
        for _ in 0..3 {
            pipeline.advance_layer(&[0, 1], 1024);
        }
        pipeline.reset();

        // Assert: all builder-configured fields remain unchanged after reset
        assert_eq!(pipeline.num_layers, 48, "num_layers must survive reset");
        assert_eq!(pipeline.pipeline_depth, 5, "pipeline_depth must survive reset");
        assert!(
            (pipeline.pcie_bandwidth_gbs - 80.0).abs() < 1e-6,
            "PCIe bandwidth must survive reset"
        );
        assert!(
            (pipeline.rdma_bandwidth_gbs - 250.0).abs() < 1e-6,
            "RDMA bandwidth must survive reset"
        );
        assert!(
            (pipeline.layer_compute_time_us - 300.0).abs() < 1e-6,
            "compute time must survive reset"
        );
    }

    // ── 13 new tests (wave-15) ──

    #[test]
    fn duplicate_centroid_tokens_produce_distinct_entries() {
        // Arrange: advance with duplicate centroid tokens
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Act: pass the same token index three times
        pipeline.advance_layer(&[7, 7, 7], 1024);

        // Assert: 3 entries created even though tokens are identical
        assert_eq!(pipeline.active_count(), 3, "duplicate centroids should each produce an entry");
        for entry in &pipeline.active_entries {
            assert_eq!(entry.start_token, 7);
        }
    }

    #[test]
    fn entry_bytes_identical_across_transfer_kinds_for_same_block_size() {
        // Arrange: depth=6 produces all 3 transfer kinds for the same block_size
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(6)
            .with_layer_compute_time(1000000.0);
        let block_size = 4096;

        // Act
        pipeline.advance_layer(&[0], block_size);

        // Assert: every entry has bytes = block_size * 2 regardless of transfer_kind
        assert!(pipeline.active_count() >= 3);
        for entry in &pipeline.active_entries {
            assert_eq!(
                entry.bytes,
                block_size * 2,
                "bytes must be 2*block_size regardless of transfer_kind {:?}",
                entry.transfer_kind
            );
        }
    }

    #[test]
    fn varying_centroid_counts_across_rounds() {
        // Arrange: pipeline with depth=1, fast completion
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: round 1 with 1 centroid, round 2 with 3 centroids, round 3 with 0 centroids
        pipeline.advance_layer(&[10], 1024);
        assert_eq!(pipeline.active_count(), 1, "round 1: 1 centroid → 1 entry");

        pipeline.advance_layer(&[20, 30, 40], 1024);
        // Round 1 entry completed (5μs << 100μs), 3 new entries submitted
        assert_eq!(pipeline.active_count(), 3, "round 2: previous completed, 3 new entries");
        assert_eq!(pipeline.completed_count(), 1, "round 1 entry should be completed");

        pipeline.advance_layer(&[], 1024);
        // Round 2 entries completed (5μs << 100μs each), 0 new entries
        assert_eq!(pipeline.active_count(), 0, "round 3: all completed, no new entries");
        assert_eq!(pipeline.completed_count(), 4, "total 4 completions (1+3)");
    }

    #[test]
    fn completion_when_elapsed_exactly_equals_latency() {
        // Arrange: GpuHbmToL2 latency = 5.0μs. We want elapsed == latency.
        // After first advance: timestamp = 5μs (compute_time=5.0), entry has latency=5.0μs
        // After second advance: timestamp = 10μs, elapsed = 10 - 5 = 5μs == 5.0μs → completed
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(5.0);

        // Act
        pipeline.advance_layer(&[0], 1024);
        assert_eq!(pipeline.active_count(), 1, "entry submitted after first advance");
        assert_eq!(pipeline.completed_count(), 0);

        pipeline.advance_layer(&[0], 1024);

        // Assert: elapsed = 5μs >= 5.0μs → entry completes exactly at the boundary
        assert_eq!(
            pipeline.completed_count(), 1,
            "entry should complete when elapsed == estimated_latency"
        );
    }

    #[test]
    fn consecutive_resets_produce_clean_state() {
        // Arrange: advance pipeline then double-reset
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        pipeline.advance_layer(&[0, 1], 1024);
        pipeline.advance_layer(&[2], 1024);

        // Act: reset twice
        pipeline.reset();
        pipeline.reset();

        // Assert: state is clean after double reset
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);
        assert_eq!(pipeline.completed_count(), 0);
        assert_eq!(pipeline.utilization(), 0.0);
        assert!(pipeline.is_prefetch_completed(5));
    }

    #[test]
    fn transfer_kind_boundary_current_plus_3_is_pcie_not_rdma() {
        // Arrange: current_layer+3 should still be PcieDma, not Rdma
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(4)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[0], 1024); // current_layer=1, targets: 2,3,4,5

        // Assert: target_layer=4 (= current+3) must be PcieDma, not Rdma
        let layer4_entry = pipeline
            .active_entries
            .iter()
            .find(|e| e.target_layer == 4);
        assert!(
            layer4_entry.is_some(),
            "should have entry for target layer 4"
        );
        assert_eq!(
            layer4_entry.unwrap().transfer_kind,
            PrefetchTransferKind::PcieDma,
            "current_layer+3 must be PcieDma (boundary)"
        );

        // And target_layer=5 (= current+4) must be Rdma
        let layer5_entry = pipeline
            .active_entries
            .iter()
            .find(|e| e.target_layer == 5);
        assert!(
            layer5_entry.is_some(),
            "should have entry for target layer 5"
        );
        assert_eq!(
            layer5_entry.unwrap().transfer_kind,
            PrefetchTransferKind::Rdma,
            "current_layer+4 must be Rdma (boundary)"
        );
    }

    #[test]
    fn utilization_tracks_correctly_after_reset_and_readvance() {
        // Arrange: build state, reset, re-advance
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Build initial state
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }
        let util_before_reset = pipeline.utilization();
        assert!(util_before_reset > 0.0, "should have non-zero utilization before reset");

        // Act: reset and re-advance 3 times
        pipeline.reset();
        for _ in 0..3 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: utilization reflects only the new advances (not pre-reset state)
        // completed=2 (rounds 1-2 entries complete), current_layer=3, depth=1
        let util = pipeline.utilization();
        assert!(
            (util - (2.0f32 / 3.0)).abs() < 1e-6,
            "utilization after reset+readvance should be 2/3, got {util}"
        );
    }

    #[test]
    fn advance_with_usize_max_num_layers_no_panic() {
        // Arrange: extremely large num_layers (no actual allocation)
        let mut pipeline = PrefetchPipeline::new(usize::MAX)
            .with_pipeline_depth(3)
            .with_layer_compute_time(100.0);

        // Act: advance a few times
        pipeline.advance_layer(&[0], 1024);
        pipeline.advance_layer(&[0], 1024);
        pipeline.advance_layer(&[0], 1024);

        // Assert: no panic, current_layer=3, entries created
        assert_eq!(pipeline.current_layer(), 3);
        assert!(pipeline.active_count() > 0, "should have active entries");
    }

    #[test]
    fn completed_count_includes_all_entries_when_latency_is_tiny() {
        // Arrange: depth=3, compute=100μs, tiny block so all latencies are tiny
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_bandwidth(1_000_000.0, 1_000_000.0) // absurdly fast bandwidth → near-zero latency
            .with_layer_compute_time(100.0);

        // Act: advance twice. Round 1 entries have latency ≈ 0μs.
        pipeline.advance_layer(&[0], 1);
        assert_eq!(pipeline.active_count(), 3, "3 entries submitted in round 1");

        pipeline.advance_layer(&[0], 1);
        // Round 1 entries (3 entries, latency ≈ 0) all complete by round 2

        // Assert: all 3 round-1 entries should be completed
        assert!(
            pipeline.completed_count() >= 3,
            "all 3 round-1 entries should complete, got {}",
            pipeline.completed_count()
        );
    }

    #[test]
    fn active_entries_capacity_preserved_after_reset() {
        // Arrange: fill active_entries then reset
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(5)
            .with_layer_compute_time(1000000.0); // prevent completion

        pipeline.advance_layer(&[0, 1, 2, 3, 4], 1024);
        assert_eq!(pipeline.active_count(), 25); // 5 centroids × 5 depth
        let capacity_before = pipeline.active_entries.capacity();
        assert!(capacity_before >= 25);

        // Act: reset clears entries but Vec is cleared (not shrunk)
        pipeline.reset();

        // Assert: capacity is preserved (no reallocation needed on next advance)
        assert_eq!(pipeline.active_count(), 0);
        assert!(
            pipeline.active_entries.capacity() >= capacity_before,
            "capacity should not decrease after reset"
        );
    }

    #[test]
    fn advance_then_reset_then_advance_uses_reset_centroids() {
        // Arrange: advance with centroid A, reset, advance with centroid B
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        pipeline.advance_layer(&[111], 512);
        assert_eq!(pipeline.current_layer(), 1);

        // Act: reset and advance with a different centroid
        pipeline.reset();
        assert_eq!(pipeline.current_layer(), 0);
        assert_eq!(pipeline.active_count(), 0);

        pipeline.advance_layer(&[222], 2048);

        // Assert: only the new centroid should be in active entries
        // current_layer=1 after advance, depth=1, target=2, 1 centroid → 1 entry
        assert_eq!(
            pipeline.active_count(), 1,
            "expected 1 entry after reset+readvance, got {} with pipeline_depth={}",
            pipeline.active_count(), pipeline.pipeline_depth
        );
        assert_eq!(pipeline.active_entries[0].start_token, 222);
        assert_eq!(pipeline.active_entries[0].bytes, 2048 * 2);
        assert_eq!(pipeline.active_entries[0].target_layer, 2);
    }

    #[test]
    fn pcie_latency_formula_matches_manual_computation() {
        // Arrange: depth=4, bandwidth=32 GB/s, block_size=4096 bytes
        // For target_layer = current+2 (PcieDma):
        //   bytes_gb = (4096 * 2) / 1e9 = 8192e-9
        //   latency = bytes_gb / 32.0 * 1e6 = 8192e-9 / 32.0 * 1e6 = 0.256μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(4)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[0], 4096);

        // Assert: find the first PCIe entry (target_layer = current+2 = 3)
        let pcie_entry = pipeline
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::PcieDma)
            .expect("should have at least one PCIe entry");

        let expected_latency = ((4096 * 2) as f32 / 1e9) / 32.0 * 1e6;
        assert!(
            (pcie_entry.estimated_latency_us - expected_latency).abs() < 1e-6,
            "PCIe latency should be {}, got {}",
            expected_latency,
            pcie_entry.estimated_latency_us
        );
    }

    // ── 13 new tests (wave-16) ──

    #[test]
    fn advance_with_depth_one_and_three_layers_exact_entry_count() {
        // Arrange: 3-layer model, depth=1
        let mut pipeline = PrefetchPipeline::new(3)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Act: advance from layer 0 → 1, target_layer=2 < 3 → 1 entry
        pipeline.advance_layer(&[5], 1024);
        assert_eq!(pipeline.active_count(), 1, "should target layer 2");

        // Advance from layer 1 → 2, target_layer=3 >= 3 → 0 new entries
        pipeline.advance_layer(&[5], 1024);

        // Assert: total active entries = 2 (layer 2 from round 1 still alive + 0 from round 2)
        // But round 1 entry (latency=5μs) completed after 2e6μs elapsed, so only new entries matter
        // With compute=1e6μs: round 1 timestamp=1e6, round 2 timestamp=2e6
        // Elapsed for round 1 entry = 2e6 - 1e6 = 1e6μs >= 5μs → completed
        // So only the second round's entry attempt (target=3 >= 3) produces nothing
        assert_eq!(
            pipeline.active_count(), 0,
            "round 1 entry completed, round 2 produces no entries (target >= num_layers)"
        );
    }

    #[test]
    fn completed_count_monotonically_increases_or_stays() {
        // Arrange: pipeline where some entries complete across rounds
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        let mut prev_completed = 0u64;
        // Act: advance 8 times, tracking completed_count
        for round in 0..8 {
            pipeline.advance_layer(&[0], 1024);
            let current = pipeline.completed_count();
            assert!(
                current >= prev_completed,
                "completed_count must not decrease: round {round}, prev={prev_completed}, current={current}"
            );
            prev_completed = current;
        }

        // Assert: final completed_count > 0 (entries completed across rounds)
        assert!(pipeline.completed_count() > 0, "some entries must have completed");
    }

    #[test]
    fn entry_debug_format_output_is_valid_struct_syntax() {
        // Arrange: construct entry with known values
        let entry = PipelinePrefetchEntry {
            target_layer: 42,
            start_token: 99,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferSubmitted,
            bytes: 8192,
            submit_timestamp_us: 12345,
            estimated_latency_us: 67.89,
        };

        // Act
        let debug = format!("{entry:?}");

        // Assert: Debug output must contain field names and values
        assert!(debug.contains("42"), "must contain target_layer value");
        assert!(debug.contains("99"), "must contain start_token value");
        assert!(debug.contains("8192"), "must contain bytes value");
        assert!(debug.contains("12345"), "must contain submit_timestamp_us value");
        assert!(debug.contains("67.89"), "must contain estimated_latency_us value");
    }

    #[test]
    fn advance_with_single_centroid_at_last_valid_layer() {
        // Arrange: 10-layer model, advance to layer 8 (last layer = 9)
        let mut pipeline = PrefetchPipeline::new(10)
            .with_pipeline_depth(1)
            .with_layer_compute_time(1000000.0);

        // Advance 8 times to reach current_layer=8
        for _ in 0..8 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Act: advance to layer 9, target_layer=10 >= 10 → no entry
        pipeline.advance_layer(&[0], 1024);

        // Assert: only entries from prior rounds targeting layer 9 may persist
        let has_layer_10 = pipeline
            .active_entries
            .iter()
            .any(|e| e.target_layer >= 10);
        assert!(
            !has_layer_10,
            "no entry should target layer >= num_layers after advancing to last valid layer"
        );
    }

    #[test]
    fn pipeline_active_entries_order_is_depth_then_centroid() {
        // Arrange: 3 centroids, depth=2
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[10, 20, 30], 1024);

        // Assert: 6 entries total, verify they alternate by depth level
        assert_eq!(pipeline.active_count(), 6);
        // The loop iterates depth first (1..=2), then centroids
        // So first 3 entries are depth=1 (target=2), next 3 are depth=2 (target=3)
        let layer2_count = pipeline
            .active_entries
            .iter()
            .take(3)
            .filter(|e| e.target_layer == 2)
            .count();
        assert_eq!(layer2_count, 3, "first 3 entries should target layer 2");
    }

    #[test]
    fn advance_with_usize_max_centroid_token_no_panic() {
        // Arrange: centroid token at usize::MAX
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        // Act: advance with usize::MAX as centroid token
        pipeline.advance_layer(&[usize::MAX], 1024);

        // Assert: entry created with exact token preserved
        assert_eq!(pipeline.active_count(), 1);
        assert_eq!(
            pipeline.active_entries[0].start_token,
            usize::MAX,
            "start_token must preserve usize::MAX"
        );
    }

    #[test]
    fn prefetch_stage_ord_implies_eq_consistency() {
        // Arrange: pairs of equal stages
        let pairs = [
            (PrefetchStage::CentroidComputed, PrefetchStage::CentroidComputed),
            (PrefetchStage::AddressTranslated, PrefetchStage::AddressTranslated),
            (PrefetchStage::TransferSubmitted, PrefetchStage::TransferSubmitted),
            (PrefetchStage::TransferCompleted, PrefetchStage::TransferCompleted),
        ];

        // Assert: a == b implies !(a < b) && !(a > b) and a.cmp(b) == Equal
        for (a, b) in &pairs {
            assert_eq!(a, b);
            assert!(!(a < b), "{:?} < {:?} should be false for equal values", a, b);
            assert!(!(a > b), "{:?} > {:?} should be false for equal values", a, b);
            assert_eq!(a.cmp(b), std::cmp::Ordering::Equal);
        }
    }

    #[test]
    fn entry_stage_is_mutable_after_construction() {
        // Arrange: construct entry with CentroidComputed stage
        let mut entry = PipelinePrefetchEntry {
            target_layer: 1,
            start_token: 0,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::GpuHbmToL2,
            stage: PrefetchStage::CentroidComputed,
            bytes: 2048,
            submit_timestamp_us: 0,
            estimated_latency_us: 5.0,
        };

        // Act: progress through all stages
        entry.stage = PrefetchStage::AddressTranslated;
        assert_eq!(entry.stage, PrefetchStage::AddressTranslated);

        entry.stage = PrefetchStage::TransferSubmitted;
        assert_eq!(entry.stage, PrefetchStage::TransferSubmitted);

        entry.stage = PrefetchStage::TransferCompleted;
        assert_eq!(entry.stage, PrefetchStage::TransferCompleted);
    }

    #[test]
    fn can_hide_latency_with_f32_max() {
        // Arrange: pipeline with maximum depth
        let pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(32)
            .with_layer_compute_time(100.0);

        // Act & Assert: available = 32 * 100 = 3200μs, f32::MAX >> 3200
        assert!(
            !pipeline.can_hide_latency(f32::MAX),
            "f32::MAX cannot be hidden by any realistic pipeline depth"
        );
    }

    #[test]
    fn entry_zero_bytes_with_all_transfer_kinds() {
        // Arrange: construct entries with bytes=0 for each transfer kind
        for kind in [
            PrefetchTransferKind::GpuHbmToL2,
            PrefetchTransferKind::PcieDma,
            PrefetchTransferKind::Rdma,
        ] {
            // Act: construct entry with bytes=0
            let entry = PipelinePrefetchEntry {
                target_layer: 0,
                start_token: 0,
                token_count: 0,
                transfer_kind: kind,
                stage: PrefetchStage::CentroidComputed,
                bytes: 0,
                submit_timestamp_us: 0,
                estimated_latency_us: 0.0,
            };

            // Assert: bytes is 0 and transfer_kind is preserved
            assert_eq!(entry.bytes, 0, "bytes should be 0 for {:?}", kind);
            assert_eq!(entry.transfer_kind, kind);
        }
    }

    #[test]
    fn advance_with_depth_three_and_three_layers_only_targets_remaining() {
        // Arrange: 5-layer model, advance to layer 3, depth=3
        let mut pipeline = PrefetchPipeline::new(5)
            .with_pipeline_depth(3)
            .with_layer_compute_time(1000000.0);

        // Advance to current_layer=3
        for _ in 0..3 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Act: advance to layer 4, depth=3 would target 5,6,7 but only 5 >= num_layers=5
        pipeline.advance_layer(&[0], 1024);

        // Assert: no entry targets layer >= 5
        let targets: Vec<usize> = pipeline
            .active_entries
            .iter()
            .map(|e| e.target_layer)
            .collect();
        for &t in &targets {
            assert!(t < 5, "target layer {t} must be < num_layers=5");
        }
    }

    #[test]
    fn utilization_denominator_uses_current_layer_times_depth() {
        // Arrange: depth=2, advance exactly 5 times
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        // Act: 5 advances
        for _ in 0..5 {
            pipeline.advance_layer(&[0], 1024);
        }

        // Assert: current_layer=5, denominator = 5 * 2 = 10
        let completed = pipeline.completed_count();
        let expected_util = completed as f32 / 10.0f32;
        let actual_util = pipeline.utilization();
        assert!(
            (actual_util - expected_util).abs() < 1e-6,
            "utilization should be {completed}/10 = {expected_util}, got {actual_util}"
        );
    }

    // ── 10 new tests (wave-17) ──

    #[test]
    fn rdma_latency_formula_matches_manual_computation() {
        // Arrange: depth=6, bandwidth=100 GB/s RDMA, block_size=4096
        // For target_layer = current+4 (Rdma):
        //   bytes_gb = (4096 * 2) / 1e9 = 8192e-9
        //   latency = bytes_gb / 100.0 * 1e6 = 8192e-9 / 100.0 * 1e6 = 0.08192μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(6)
            .with_bandwidth(32.0, 100.0)
            .with_layer_compute_time(1000000.0);

        // Act
        pipeline.advance_layer(&[0], 4096);

        // Assert: find the first RDMA entry (target_layer = current+4 = 5)
        let rdma_entry = pipeline
            .active_entries
            .iter()
            .find(|e| e.transfer_kind == PrefetchTransferKind::Rdma)
            .expect("should have at least one RDMA entry with depth=6");

        let expected_latency = ((4096 * 2) as f32 / 1e9) / 100.0 * 1e6;
        assert!(
            (rdma_entry.estimated_latency_us - expected_latency).abs() < 1e-6,
            "RDMA latency should be {}, got {}",
            expected_latency,
            rdma_entry.estimated_latency_us
        );
    }

    #[test]
    fn prefetch_stage_all_four_variants_as_hashmap_keys() {
        // Arrange: use all 4 PrefetchStage variants as HashMap keys
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(PrefetchStage::CentroidComputed, 0);
        map.insert(PrefetchStage::AddressTranslated, 1);
        map.insert(PrefetchStage::TransferSubmitted, 2);
        map.insert(PrefetchStage::TransferCompleted, 3);

        // Assert: all 4 entries retrievable
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&PrefetchStage::CentroidComputed), Some(&0));
        assert_eq!(map.get(&PrefetchStage::AddressTranslated), Some(&1));
        assert_eq!(map.get(&PrefetchStage::TransferSubmitted), Some(&2));
        assert_eq!(map.get(&PrefetchStage::TransferCompleted), Some(&3));
    }

    #[test]
    fn prefetch_stage_all_four_variants_in_hashset() {
        // Arrange: insert all 4 PrefetchStage variants into a HashSet
        use std::collections::HashSet;
        let set: HashSet<PrefetchStage> = [
            PrefetchStage::CentroidComputed,
            PrefetchStage::AddressTranslated,
            PrefetchStage::TransferSubmitted,
            PrefetchStage::TransferCompleted,
            PrefetchStage::TransferSubmitted, // duplicate
        ]
        .into_iter()
        .collect();

        // Assert: exactly 4 unique variants
        assert_eq!(set.len(), 4, "HashSet should deduplicate to 4 unique stages");
        assert!(set.contains(&PrefetchStage::CentroidComputed));
        assert!(set.contains(&PrefetchStage::AddressTranslated));
        assert!(set.contains(&PrefetchStage::TransferSubmitted));
        assert!(set.contains(&PrefetchStage::TransferCompleted));
    }

    #[test]
    fn advance_with_zero_latency_entries_complete_immediately() {
        // Arrange: extremely fast bandwidth so PCIe/RDMA latency rounds to ~0
        // block=1 byte, bandwidth=1e9 GB/s → latency ≈ 0μs
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(3)
            .with_bandwidth(1_000_000_000.0, 1_000_000_000.0)
            .with_layer_compute_time(100.0);

        // Act: advance once → 3 entries submitted
        pipeline.advance_layer(&[0], 1);
        assert_eq!(pipeline.active_count(), 3, "3 entries with near-zero latency");

        // Advance again: entries from round 1 (latency ≈ 0μs) should all complete
        pipeline.advance_layer(&[0], 1);

        // Assert: all 3 round-1 entries completed
        assert!(
            pipeline.completed_count() >= 3,
            "all 3 round-1 entries should complete immediately, got {}",
            pipeline.completed_count()
        );
    }

    #[test]
    fn advance_with_constant_empty_centroids_only_increments_layer() {
        // Arrange: always advance with empty centroids
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(2)
            .with_layer_compute_time(100.0);

        // Act: advance 5 times with no centroids
        for _ in 0..5 {
            pipeline.advance_layer(&[], 1024);
        }

        // Assert: layer increments but no entries ever created
        assert_eq!(pipeline.current_layer(), 5, "layer should increment despite empty centroids");
        assert_eq!(pipeline.active_count(), 0, "no entries with empty centroids");
        assert_eq!(pipeline.completed_count(), 0, "no completions without entries");
    }

    #[test]
    fn entries_from_same_advance_share_identical_timestamp() {
        // Arrange: multiple centroids with depth > 1
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(4)
            .with_layer_compute_time(1000000.0); // prevent completion

        // Act
        pipeline.advance_layer(&[10, 20, 30], 1024);

        // Assert: all 12 entries (3 centroids × 4 depth) share the same timestamp
        assert_eq!(pipeline.active_count(), 12);
        let first_ts = pipeline.active_entries[0].submit_timestamp_us;
        for (i, entry) in pipeline.active_entries.iter().enumerate() {
            assert_eq!(
                entry.submit_timestamp_us, first_ts,
                "entry {i} should have same timestamp as first entry"
            );
        }
    }

    #[test]
    fn pipeline_with_two_layers_and_depth_two_no_entries_after_first_advance() {
        // Arrange: 2-layer model with depth=2
        let mut pipeline = PrefetchPipeline::new(2)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1000000.0);

        // Act: advance from layer 0 to 1, targets would be 2 and 3, both >= num_layers=2
        pipeline.advance_layer(&[0], 1024);

        // Assert: no entries because all targets exceed num_layers
        assert_eq!(
            pipeline.active_count(), 0,
            "depth=2 on 2-layer model produces no entries after first advance"
        );
        assert_eq!(pipeline.current_layer(), 1);
    }

    #[test]
    fn interleaved_reset_advance_cycles_preserve_independence() {
        // Arrange: run 3 independent cycles of advance → check → reset
        let mut pipeline = PrefetchPipeline::new(32)
            .with_pipeline_depth(1)
            .with_layer_compute_time(100.0);

        for cycle in 0..3 {
            // Act: advance 3 times with a cycle-specific centroid
            let centroid = (cycle + 1) * 10;
            for _ in 0..3 {
                pipeline.advance_layer(&[centroid], 1024);
            }

            // Assert: pipeline state reflects only current cycle
            assert_eq!(pipeline.current_layer(), 3, "cycle {cycle}: current_layer should be 3");
            let has_centroid = pipeline
                .active_entries
                .iter()
                .any(|e| e.start_token == centroid);
            assert!(has_centroid, "cycle {cycle}: should have entry with centroid {centroid}");

            // Reset for next cycle
            pipeline.reset();
            assert_eq!(pipeline.current_layer(), 0, "cycle {cycle}: reset should clear layer");
            assert_eq!(pipeline.active_count(), 0, "cycle {cycle}: reset should clear entries");
        }
    }

    #[test]
    fn entry_with_extreme_estimated_latency_values() {
        // Arrange: construct entries with f32::INFINITY and f32::NAN for estimated_latency_us
        let inf_entry = PipelinePrefetchEntry {
            target_layer: 0,
            start_token: 0,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::Rdma,
            stage: PrefetchStage::CentroidComputed,
            bytes: 1024,
            submit_timestamp_us: 0,
            estimated_latency_us: f32::INFINITY,
        };
        let nan_entry = PipelinePrefetchEntry {
            target_layer: 0,
            start_token: 0,
            token_count: 1,
            transfer_kind: PrefetchTransferKind::PcieDma,
            stage: PrefetchStage::TransferSubmitted,
            bytes: 1024,
            submit_timestamp_us: 0,
            estimated_latency_us: f32::NAN,
        };

        // Assert: extreme float values are stored faithfully
        assert!(inf_entry.estimated_latency_us.is_infinite(), "INFINITY should be stored as infinite");
        assert!(inf_entry.estimated_latency_us.is_sign_positive(), "INFINITY should be positive");
        assert!(nan_entry.estimated_latency_us.is_nan(), "NAN should be stored as NaN");
    }

    #[test]
    fn advance_with_depth_two_and_three_layer_model_exhausts_all_targets() {
        // Arrange: exactly 3 layers, depth=2
        let mut pipeline = PrefetchPipeline::new(3)
            .with_pipeline_depth(2)
            .with_layer_compute_time(1000000.0);

        // Act: advance from layer 0 to 1; targets would be 2 and 3
        // target_layer=2 < 3 → valid; target_layer=3 >= 3 → invalid
        pipeline.advance_layer(&[5], 1024);

        // Assert: only 1 entry (target_layer=2); target_layer=3 is excluded
        assert_eq!(pipeline.active_count(), 1, "only target_layer=2 is valid");
        assert_eq!(pipeline.active_entries[0].target_layer, 2);
        assert_eq!(pipeline.active_entries[0].start_token, 5);
    }
}
