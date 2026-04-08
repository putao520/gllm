//! 空间异构 Sub-Batching (SPEC §12.1)
//!
//! 实现多流空间分发 (Multi-Stream Spatial Dispatch)。
//!
//! ## 核心思想
//!
//! 不同的 CPU/GPU 核心物理分区同时运行着**不同拓扑结构的 JIT 图**：
//! 1. 零散归类 (Sub-Batching): 按形状分类请求
//! 2. 多流分发 (Multi-Stream Spatial Dispatch): 切割为子批次并行发射
//! 3. 硬件资源软分区: GPU SM 分区 / CPU NUMA 绑核

use std::collections::HashMap;

use crate::scheduler::types::RequestId;
use crate::scheduler::chunked_prefill::BatchManifest;
use super::compiler_constraints::{CompilerConstraints, GpuSmPartition};

// ── 形状分类 (Shape Classification) ──

/// 请求的图形状分类
///
/// §12.1: 根据请求的运行时状态，将其归类到不同拓扑结构的 JIT 图
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphShape {
    /// 形状 A: 可跳过注意力 (gate_skip, 低激活度)
    SkipAttention,
    /// 形状 B: 死神经元走窄化 INT8 网络
    NarrowQuant,
    /// 形状 C: 完整 FP16/FP32 密集网络
    FullPrecision,
    /// 形状 D: MoE 稀疏路由 (仅激活部分专家)
    MoeSparse,
}

impl GraphShape {
    /// 获取该形状的建议硬件分区类型
    pub fn preferred_hardware(&self) -> HardwareKind {
        match self {
            GraphShape::SkipAttention => HardwareKind::LowComputeCore,
            GraphShape::NarrowQuant => HardwareKind::LowComputeCore,
            GraphShape::FullPrecision => HardwareKind::FullComputeUnit,
            GraphShape::MoeSparse => HardwareKind::FullComputeUnit,
        }
    }

    /// 获取该形状的相对计算强度 (用于资源分配)
    pub fn compute_intensity(&self) -> f32 {
        match self {
            GraphShape::SkipAttention => 0.3,
            GraphShape::NarrowQuant => 0.5,
            GraphShape::FullPrecision => 1.0,
            GraphShape::MoeSparse => 0.8,
        }
    }
}

/// 硬件分区类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareKind {
    /// 低计算核心: E-Core / 小 SM 分区
    LowComputeCore,
    /// 完整计算单元: P-Core / AMX 核心 / 大 SM 分区
    FullComputeUnit,
    /// GPU Tensor Core 分区
    TensorCorePartition,
}

// ── 子批次 (Sub-Batch) ──

/// 子批次 — 一组具有相同图形状的请求集合
///
/// §12.1: 调度器发现本批 128 个请求中：
/// 60 个可跳过注意力（Shape A），30 个死神经元走 INT8 窄网（Shape B），
/// 38 个需完整 FP16 密集网络（Shape C）
#[derive(Debug, Clone)]
pub struct SubBatch {
    /// 该子批次的图形状
    pub shape: GraphShape,
    /// 包含的请求 ID 列表
    pub request_ids: Vec<RequestId>,
    /// 目标硬件分区
    pub target_partition: Option<HardwarePartition>,
    /// 该子批次的预估计算量 (GFLOPS)
    pub estimated_gflops: f32,
    /// 该子批次的预估显存占用 (bytes)
    pub estimated_vram_bytes: usize,
}

impl SubBatch {
    /// 创建新的子批次
    pub fn new(shape: GraphShape) -> Self {
        Self {
            shape,
            request_ids: Vec::new(),
            target_partition: None,
            estimated_gflops: 0.0,
            estimated_vram_bytes: 0,
        }
    }

    /// 添加请求到子批次
    pub fn add_request(&mut self, request_id: RequestId) {
        self.request_ids.push(request_id);
    }

    /// 获取请求数量
    pub fn len(&self) -> usize {
        self.request_ids.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.request_ids.is_empty()
    }
}

// ── 硬件分区 (Hardware Partition) ──

/// 硬件分区 — 物理 SM 范围或 NUMA 节点绑定
///
/// §12.1:
/// - GPU 端: JIT 代码生成时锁死 gridDim 与 Block Size
/// - CPU 端: 利用 NUMA Node / Core Affinity 绑核
#[derive(Debug, Clone)]
pub struct HardwarePartition {
    /// 分区 ID
    pub partition_id: usize,
    /// 硬件类型
    pub kind: HardwareKind,
    /// GPU: SM 范围 [start_sm, end_sm)
    pub sm_range: Option<(usize, usize)>,
    /// CPU: NUMA 节点 ID
    pub numa_node: Option<usize>,
    /// CPU: 核心范围 [start_core, end_core)
    pub core_range: Option<(usize, usize)>,
    /// 该分区的计算能力权重 (0.0-1.0)
    pub compute_weight: f32,
}

impl HardwarePartition {
    /// 创建 GPU SM 分区
    pub fn gpu_sm_partition(partition_id: usize, start_sm: usize, end_sm: usize) -> Self {
        let total_sms = end_sm - start_sm;
        Self {
            partition_id,
            kind: HardwareKind::TensorCorePartition,
            sm_range: Some((start_sm, end_sm)),
            numa_node: None,
            core_range: None,
            compute_weight: total_sms as f32 / 80.0, // 相对于 80 SM 的权重
        }
    }

    /// 创建 CPU NUMA 分区
    pub fn cpu_numa_partition(partition_id: usize, numa_node: usize, cores: usize) -> Self {
        let is_amx = numa_node == 0; // 假设 node 0 有 AMX
        Self {
            partition_id,
            kind: if is_amx {
                HardwareKind::FullComputeUnit
            } else {
                HardwareKind::LowComputeCore
            },
            sm_range: None,
            numa_node: Some(numa_node),
            core_range: None,
            compute_weight: cores as f32 / 64.0,
        }
    }
}

// ── Sub-Batch 分发器 (Dispatcher) ──

/// Sub-Batch 分发决策结果
#[derive(Debug, Clone)]
pub struct DispatchPlan {
    /// 分发后的子批次列表
    pub sub_batches: Vec<SubBatch>,
    /// 是否需要 Ragged Compaction (§12.2 兜底)
    pub needs_ragged_compaction: bool,
    /// 被合并到兜底路径的孤立请求数量
    pub orphan_count: usize,
    /// 分发决策的原因说明
    pub reason: DispatchReason,
}

/// 分发决策原因
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchReason {
    /// 单一形状，无需分片
    SingleShapeUniform,
    /// 多形状分片分发
    MultiShapeDispatched {
        shape_counts: HashMap<GraphShape, usize>,
    },
    /// 所有请求归入兜底路径 (数量太少)
    AllOrphanFallback,
}

/// §12.1 Sub-Batch 分发器
///
/// 负责将 batch 中的请求按图形状分类，创建子批次，
/// 并将子批次分配到硬件分区。
pub struct SubBatchDispatcher {
    /// IR 约束变量
    constraints: CompilerConstraints,
    /// 最小子批次大小（低于此值的请求归入兜底路径）
    min_sub_batch_size: usize,
    /// 最大子批次数（防止过度分片）
    max_sub_batches: usize,
}

impl SubBatchDispatcher {
    /// 创建新的分发器
    pub fn new(constraints: CompilerConstraints) -> Self {
        Self {
            constraints,
            min_sub_batch_size: 4,
            max_sub_batches: 4,
        }
    }

    /// 设置最小子批次大小
    pub fn with_min_sub_batch_size(mut self, size: usize) -> Self {
        self.min_sub_batch_size = size;
        self
    }

    /// 对 batch manifest 进行形状分类和子批次分发
    ///
    /// §12.1 核心流程:
    /// 1. 零散归类: 对每个请求进行形状分类
    /// 2. 孤立检测: 不足 min_sub_batch_size 的形状归入兜底
    /// 3. 硬件分配: 为每个有效子批次分配硬件分区
    /// 4. 返回分发计划
    pub fn dispatch(&self, manifest: &BatchManifest, shape_map: &HashMap<RequestId, GraphShape>) -> DispatchPlan {
        // 1. 按形状分类
        let mut groups: HashMap<GraphShape, Vec<RequestId>> = HashMap::new();
        for slot in &manifest.slots {
            if let Some(shape) = shape_map.get(&slot.request_id) {
                groups.entry(*shape).or_default().push(slot.request_id);
            } else {
                // 未分类的请求默认走完整路径
                groups.entry(GraphShape::FullPrecision).or_default().push(slot.request_id);
            }
        }

        // 2. 分离有效子批次和孤立请求
        let mut sub_batches = Vec::new();
        let mut orphans = Vec::new();

        for (shape, request_ids) in groups {
            if request_ids.len() < self.min_sub_batch_size {
                // 孤立请求，归入兜底路径
                orphans.extend(request_ids);
            } else {
                let mut sub_batch = SubBatch::new(shape);
                sub_batch.request_ids = request_ids;
                sub_batches.push(sub_batch);
            }
        }

        // 3. 如果只有一个形状且无孤立请求，简化为单一分发
        if sub_batches.len() == 1 && orphans.is_empty() {
            return DispatchPlan {
                sub_batches,
                needs_ragged_compaction: false,
                orphan_count: 0,
                reason: DispatchReason::SingleShapeUniform,
            };
        }

        // 4. 如果所有请求都是孤立的，全部走兜底
        if sub_batches.is_empty() {
            let mut fallback = SubBatch::new(GraphShape::FullPrecision);
            fallback.request_ids = orphans;
            let orphan_count = fallback.len();
            return DispatchPlan {
                sub_batches: vec![fallback],
                needs_ragged_compaction: true,
                orphan_count,
                reason: DispatchReason::AllOrphanFallback,
            };
        }

        // 5. 为子批次分配硬件分区
        self.assign_hardware_partitions(&mut sub_batches);

        // 6. 孤立请求合并到最近形状的子批次（走 Ragged Compaction）
        let needs_ragged = !orphans.is_empty();
        let orphan_count = orphans.len();
        if !orphans.is_empty() {
            if sub_batches.is_empty() {
                // 所有请求都是孤立的 — 走兜底
                let mut fallback = SubBatch::new(GraphShape::FullPrecision);
                fallback.request_ids = orphans;
                sub_batches.push(fallback);
            } else if let Some(lightest) = sub_batches.iter_mut().min_by(|a, b| {
                a.shape.compute_intensity().partial_cmp(&b.shape.compute_intensity()).unwrap_or(std::cmp::Ordering::Less)
            }) {
                lightest.request_ids.extend(orphans);
            }
        }

        let shape_counts: HashMap<GraphShape, usize> = sub_batches.iter()
            .map(|sb| (sb.shape, sb.len()))
            .collect();

        DispatchPlan {
            sub_batches,
            needs_ragged_compaction: needs_ragged,
            orphan_count,
            reason: DispatchReason::MultiShapeDispatched { shape_counts },
        }
    }

    /// 为子批次分配硬件分区
    ///
    /// §12.1:
    /// GPU: JIT 代码生成时锁死 gridDim 与 Block Size
    /// CPU: NUMA Node / Core Affinity 绑核
    fn assign_hardware_partitions(&self, sub_batches: &mut [SubBatch]) {
        // GPU 分区策略
        if let Some(sm_count) = self.constraints.gpu_sm_count {
            let num_partitions = sub_batches.len().min(4);
            let sm_per_partition = sm_count / num_partitions.max(1);

            for (i, sub_batch) in sub_batches.iter_mut().enumerate() {
                let start_sm = i * sm_per_partition;
                let end_sm = start_sm + sm_per_partition;
                sub_batch.target_partition = Some(
                    HardwarePartition::gpu_sm_partition(i, start_sm, end_sm)
                );
            }
        } else {
            // CPU NUMA 分区策略
            let numa_nodes = self.constraints.numa_node_count.max(1);
            let core_bindings = self.constraints.numa_core_bindings();

            for (i, sub_batch) in sub_batches.iter_mut().enumerate() {
                let node_id = i % numa_nodes;
                let cores = core_bindings.get(node_id)
                    .map(|(_, range)| range.end - range.start)
                    .unwrap_or(4);
                sub_batch.target_partition = Some(
                    HardwarePartition::cpu_numa_partition(i, node_id, cores)
                );
            }
        }
    }

    /// 推导单个请求的图形状
    ///
    /// 基于请求的运行时遥测数据（注意力稀疏度、死神经元比例等）推导形状分类
    pub fn classify_request(
        &self,
        attention_sparsity: f32,  // 0.0 = 全有效, 1.0 = 全跳过
        dead_neuron_ratio: f32,    // 0.0 = 全活, 1.0 = 全死
        is_moe: bool,
        moe_active_experts: f32,   // 活跃专家比例 (0.0-1.0)
    ) -> GraphShape {
        // §12.1 分类规则
        if is_moe && moe_active_experts < 0.5 {
            GraphShape::MoeSparse
        } else if dead_neuron_ratio > 0.6 {
            GraphShape::NarrowQuant
        } else if attention_sparsity > 0.7 {
            GraphShape::SkipAttention
        } else {
            GraphShape::FullPrecision
        }
    }

    /// 获取约束变量引用
    pub fn constraints(&self) -> &CompilerConstraints {
        &self.constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::chunked_prefill::{BatchSlot, BatchManifest, SlotType};

    fn make_manifest_with_ids(ids: &[RequestId]) -> BatchManifest {
        let slots: Vec<BatchSlot> = ids.iter().enumerate().map(|(i, &id)| {
            BatchSlot {
                request_id: id,
                slot_type: SlotType::Decode,
                token_start: i * 10,
                token_end: i * 10 + 10,
                compact_target: i as i32,
            }
        }).collect();
        BatchManifest {
            slots,
            total_tokens: ids.len() * 10,
            decode_tokens: ids.len() * 10,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        }
    }

    #[test]
    fn test_graph_shape_classify() {
        let constraints = CompilerConstraints::default();
        let dispatcher = SubBatchDispatcher::new(constraints);

        // Full precision (default)
        assert_eq!(dispatcher.classify_request(0.1, 0.1, false, 0.0), GraphShape::FullPrecision);

        // Skip attention
        assert_eq!(dispatcher.classify_request(0.8, 0.1, false, 0.0), GraphShape::SkipAttention);

        // Narrow quant
        assert_eq!(dispatcher.classify_request(0.1, 0.7, false, 0.0), GraphShape::NarrowQuant);

        // MoE sparse
        assert_eq!(dispatcher.classify_request(0.1, 0.1, true, 0.3), GraphShape::MoeSparse);
    }

    #[test]
    fn test_single_shape_dispatch() {
        let constraints = CompilerConstraints::default();
        let dispatcher = SubBatchDispatcher::new(constraints);

        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        // All full precision
        let mut shape_map = HashMap::new();
        for &id in &ids {
            shape_map.insert(id, GraphShape::FullPrecision);
        }

        let plan = dispatcher.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 1);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
    }

    #[test]
    fn test_multi_shape_dispatch() {
        let constraints = CompilerConstraints::default();
        let dispatcher = SubBatchDispatcher::new(constraints);

        let ids: Vec<RequestId> = (1..=20).collect();
        let manifest = make_manifest_with_ids(&ids);

        // 8 full precision, 8 skip attention, 4 narrow quant
        let mut shape_map = HashMap::new();
        for &id in &ids[..8] {
            shape_map.insert(id, GraphShape::FullPrecision);
        }
        for &id in &ids[8..16] {
            shape_map.insert(id, GraphShape::SkipAttention);
        }
        for &id in &ids[16..20] {
            shape_map.insert(id, GraphShape::NarrowQuant);
        }

        let plan = dispatcher.dispatch(&manifest, &shape_map);
        assert!(plan.sub_batches.len() >= 2);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
    }

    #[test]
    fn test_orphan_fallback() {
        let constraints = CompilerConstraints::default();
        let dispatcher = SubBatchDispatcher::new(constraints);

        let ids: Vec<RequestId> = (1..=3).collect();
        let manifest = make_manifest_with_ids(&ids);

        // Only 3 requests with different shapes → all orphans
        let mut shape_map = HashMap::new();
        shape_map.insert(1, GraphShape::FullPrecision);
        shape_map.insert(2, GraphShape::SkipAttention);
        shape_map.insert(3, GraphShape::NarrowQuant);

        let plan = dispatcher.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
        assert!(plan.needs_ragged_compaction);
    }

    #[test]
    fn test_hardware_partition_gpu() {
        let partition = HardwarePartition::gpu_sm_partition(0, 0, 26);
        assert_eq!(partition.sm_range, Some((0, 26)));
        assert_eq!(partition.kind, HardwareKind::TensorCorePartition);
    }

    #[test]
    fn test_hardware_partition_cpu() {
        let partition = HardwarePartition::cpu_numa_partition(0, 1, 16);
        assert_eq!(partition.numa_node, Some(1));
    }

    #[test]
    fn test_graph_shape_compute_intensity() {
        assert!(GraphShape::FullPrecision.compute_intensity() > GraphShape::SkipAttention.compute_intensity());
        assert!(GraphShape::NarrowQuant.compute_intensity() > GraphShape::SkipAttention.compute_intensity());
    }
}
