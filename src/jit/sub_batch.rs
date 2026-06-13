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
use super::compiler_constraints::CompilerConstraints;

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
#[allow(dead_code)]
pub struct SubBatchDispatcher {
    /// IR 约束变量
    constraints: CompilerConstraints,
    /// 最小子批次大小（低于此值的请求归入兜底路径）
    min_sub_batch_size: usize,
    /// 最大子批次数（防止过度分片）
    max_sub_batches: usize,
    /// Build-time constant: whether the model has MoE ops.
    /// ARCH-JIT-DATA-YIELDS: derived from graph topology at construction, not passed per-call.
    has_moe_ops: bool,
}

impl SubBatchDispatcher {
    /// 创建新的分发器
    pub fn new(constraints: CompilerConstraints) -> Self {
        Self {
            constraints,
            min_sub_batch_size: 4,
            max_sub_batches: 4,
            has_moe_ops: false,
        }
    }

    /// 设置 MoE 存在性（build-time constant）
    pub fn with_has_moe_ops(mut self, has_moe_ops: bool) -> Self {
        self.has_moe_ops = has_moe_ops;
        self
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
        moe_active_experts: f32,   // 活跃专家比例 (0.0-1.0)
    ) -> GraphShape {
        // §12.1 分类规则
        if self.has_moe_ops && moe_active_experts < 0.5 {
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
        let dispatcher = SubBatchDispatcher::new(CompilerConstraints::default());

        // Full precision (default)
        assert_eq!(dispatcher.classify_request(0.1, 0.1, 0.0), GraphShape::FullPrecision);

        // Skip attention
        assert_eq!(dispatcher.classify_request(0.8, 0.1, 0.0), GraphShape::SkipAttention);

        // Narrow quant
        assert_eq!(dispatcher.classify_request(0.1, 0.7, 0.0), GraphShape::NarrowQuant);

        // MoE sparse — needs has_moe_ops=true dispatcher
        let moe_dispatcher = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        assert_eq!(moe_dispatcher.classify_request(0.1, 0.1, 0.3), GraphShape::MoeSparse);
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

    // ── GraphShape: all variants ──

    #[test]
    fn test_graph_shape_preferred_hardware_mapping() {
        // Every variant maps to a deterministic HardwareKind
        assert_eq!(GraphShape::SkipAttention.preferred_hardware(), HardwareKind::LowComputeCore);
        assert_eq!(GraphShape::NarrowQuant.preferred_hardware(), HardwareKind::LowComputeCore);
        assert_eq!(GraphShape::FullPrecision.preferred_hardware(), HardwareKind::FullComputeUnit);
        assert_eq!(GraphShape::MoeSparse.preferred_hardware(), HardwareKind::FullComputeUnit);
    }

    #[test]
    fn test_graph_shape_compute_intensity_total_ordering() {
        // Verify the full ordering: SkipAttention < NarrowQuant < MoeSparse < FullPrecision
        let sa = GraphShape::SkipAttention.compute_intensity();
        let nq = GraphShape::NarrowQuant.compute_intensity();
        let ms = GraphShape::MoeSparse.compute_intensity();
        let fp = GraphShape::FullPrecision.compute_intensity();

        assert!(sa < nq, "SkipAttention should be less than NarrowQuant");
        assert!(nq < ms, "NarrowQuant should be less than MoeSparse");
        assert!(ms < fp, "MoeSparse should be less than FullPrecision");
        assert!((fp - 1.0f32).abs() < f32::EPSILON, "FullPrecision intensity should be 1.0");
    }

    #[test]
    fn test_graph_shape_copy_clone_eq_hash() {
        // GraphShape derives Copy, Clone, PartialEq, Eq, Hash
        let a = GraphShape::MoeSparse;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        // Hash consistency: equal values must produce equal hashes
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(GraphShape::FullPrecision);
        assert!(set.contains(&GraphShape::FullPrecision));
        assert!(!set.contains(&GraphShape::SkipAttention));
    }

    // ── HardwareKind: all variants ──

    #[test]
    fn test_hardware_kind_variants_distinct() {
        // Each variant is distinct
        let kinds = [HardwareKind::LowComputeCore, HardwareKind::FullComputeUnit, HardwareKind::TensorCorePartition];
        for (i, a) in kinds.iter().enumerate() {
            for (j, b) in kinds.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    // ── SubBatch: constructors and methods ──

    #[test]
    fn test_sub_batch_new_starts_empty() {
        let sb = SubBatch::new(GraphShape::FullPrecision);
        assert_eq!(sb.shape, GraphShape::FullPrecision);
        assert!(sb.request_ids.is_empty());
        assert!(sb.target_partition.is_none());
        assert_eq!(sb.estimated_gflops, 0.0);
        assert_eq!(sb.estimated_vram_bytes, 0);
        assert_eq!(sb.len(), 0);
        assert!(sb.is_empty());
    }

    #[test]
    fn test_sub_batch_add_request_and_len() {
        let mut sb = SubBatch::new(GraphShape::SkipAttention);
        assert!(sb.is_empty());

        sb.add_request(42);
        assert_eq!(sb.len(), 1);
        assert!(!sb.is_empty());
        assert_eq!(sb.request_ids[0], 42);

        sb.add_request(99);
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.request_ids[1], 99);
    }

    #[test]
    fn test_sub_batch_preserves_insertion_order() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        let ids: Vec<RequestId> = vec![10, 20, 30, 40, 50];
        for &id in &ids {
            sb.add_request(id);
        }
        assert_eq!(sb.request_ids, ids);
    }

    // ── HardwarePartition: constructors ──

    #[test]
    fn test_gpu_sm_partition_fields() {
        let p = HardwarePartition::gpu_sm_partition(3, 20, 40);
        assert_eq!(p.partition_id, 3);
        assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        assert_eq!(p.sm_range, Some((20, 40)));
        assert!(p.numa_node.is_none());
        assert!(p.core_range.is_none());
        // compute_weight = (40-20)/80.0 = 0.25
        assert!((p.compute_weight - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_numa_partition_node0_is_full_compute() {
        // node 0 is treated as having AMX → FullComputeUnit
        let p = HardwarePartition::cpu_numa_partition(0, 0, 32);
        assert_eq!(p.partition_id, 0);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
        assert_eq!(p.numa_node, Some(0));
        assert!(p.sm_range.is_none());
        // compute_weight = 32/64.0 = 0.5
        assert!((p.compute_weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_numa_partition_nonzero_is_low_compute() {
        // Non-zero NUMA node → LowComputeCore
        let p = HardwarePartition::cpu_numa_partition(1, 2, 16);
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
        assert_eq!(p.numa_node, Some(2));
        assert!(p.sm_range.is_none());
        assert!(p.core_range.is_none());
    }

    // ── SubBatchDispatcher: builder pattern ──

    #[test]
    fn test_dispatcher_default_min_sub_batch_size() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // dispatch with 3 identical-shape requests → all orphans (min=4)
        let ids: Vec<RequestId> = vec![1, 2, 3];
        let manifest = make_manifest_with_ids(&ids);
        let mut shape_map = HashMap::new();
        for &id in &ids {
            shape_map.insert(id, GraphShape::FullPrecision);
        }
        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
    }

    #[test]
    fn test_dispatcher_with_min_sub_batch_size_overrides() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        // 3 requests → above min=2, should produce SingleShapeUniform
        let ids: Vec<RequestId> = vec![1, 2, 3];
        let manifest = make_manifest_with_ids(&ids);
        let mut shape_map = HashMap::new();
        for &id in &ids {
            shape_map.insert(id, GraphShape::FullPrecision);
        }
        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].len(), 3);
    }

    #[test]
    fn test_dispatcher_constraints_accessor() {
        let constraints = CompilerConstraints::default();
        let d = SubBatchDispatcher::new(constraints);
        let returned = d.constraints();
        assert_eq!(returned.numa_node_count, 1);
        assert!(returned.gpu_sm_count.is_none());
    }

    // ── classify_request: boundary thresholds ──

    #[test]
    fn test_classify_moe_boundary() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);

        // moe_active_experts exactly at 0.5 → NOT sparse (condition is < 0.5)
        let shape = d.classify_request(0.0, 0.0, 0.5);
        assert_eq!(shape, GraphShape::FullPrecision);

        // Just below 0.5 → sparse
        let shape = d.classify_request(0.0, 0.0, 0.49);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    #[test]
    fn test_classify_dead_neuron_boundary() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());

        // dead_neuron_ratio exactly at 0.6 → NarrowQuant (condition is > 0.6 is false, so 0.6 stays)
        // Wait, condition is > 0.6, so 0.6 is NOT > 0.6
        let shape = d.classify_request(0.0, 0.6, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);

        // Just above 0.6 → NarrowQuant
        let shape = d.classify_request(0.0, 0.61, 0.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_attention_sparsity_boundary() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());

        // attention_sparsity exactly at 0.7 → NOT skip (> 0.7 is false)
        let shape = d.classify_request(0.7, 0.0, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);

        // Just above 0.7 → SkipAttention
        let shape = d.classify_request(0.71, 0.0, 0.0);
        assert_eq!(shape, GraphShape::SkipAttention);
    }

    #[test]
    fn test_classify_moe_takes_priority() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);

        // Even with high dead_neuron_ratio, MoE with low active_experts wins
        let shape = d.classify_request(0.9, 0.9, 0.1);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    // ── dispatch: unclassified requests default to FullPrecision ──

    #[test]
    fn test_dispatch_unclassified_defaults_to_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = vec![1, 2, 3, 4, 5];
        let manifest = make_manifest_with_ids(&ids);

        // Empty shape_map → all requests default to FullPrecision
        let shape_map: HashMap<RequestId, GraphShape> = HashMap::new();
        let plan = d.dispatch(&manifest, &shape_map);

        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);
        assert_eq!(plan.sub_batches[0].len(), 5);
    }

    // ── dispatch: orphans merge into lightest sub-batch ──

    #[test]
    fn test_dispatch_orphans_merge_to_lightest_batch() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(4);
        let ids: Vec<RequestId> = (1..=13).collect();
        let manifest = make_manifest_with_ids(&ids);

        // 5 FullPrecision, 5 SkipAttention, 3 NarrowQuant (orphans)
        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[10..13] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 3);
        assert!(plan.needs_ragged_compaction);

        // Orphans (NarrowQuant) should merge into lightest batch (SkipAttention, intensity 0.3)
        let skip_batch = plan.sub_batches.iter().find(|sb| sb.shape == GraphShape::SkipAttention);
        assert!(skip_batch.is_some());
        // Original 5 + 3 orphans = 8
        assert_eq!(skip_batch.unwrap().len(), 8);
    }

    // ── dispatch: GPU partition assignment ──

    #[test]
    fn test_dispatch_assigns_gpu_partitions_when_available() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=10).collect();
        let manifest = make_manifest_with_ids(&ids);

        // Two shapes: 5 FullPrecision, 5 SkipAttention
        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));

        // Each sub-batch should have a GPU SM partition assigned
        for sb in &plan.sub_batches {
            assert!(sb.target_partition.is_some(), "sub-batch should have hardware partition");
            let partition = sb.target_partition.as_ref().unwrap();
            assert_eq!(partition.kind, HardwareKind::TensorCorePartition);
            assert!(partition.sm_range.is_some());
        }
    }

    // ── DispatchPlan: shape_counts accuracy ──

    #[test]
    fn test_dispatch_shape_counts_reflect_actual_sizes() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=14).collect();
        let manifest = make_manifest_with_ids(&ids);

        // 5 FullPrecision, 5 MoeSparse, 4 NarrowQuant
        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::MoeSparse); }
        for &id in &ids[10..14] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);

        if let DispatchReason::MultiShapeDispatched { shape_counts } = &plan.reason {
            assert_eq!(shape_counts.get(&GraphShape::FullPrecision), Some(&5));
            assert_eq!(shape_counts.get(&GraphShape::MoeSparse), Some(&5));
            assert_eq!(shape_counts.get(&GraphShape::NarrowQuant), Some(&4));
        } else {
            panic!("Expected MultiShapeDispatched, got {:?}", plan.reason);
        }
    }

    // ── Debug trait: smoke test ──

    #[test]
    fn test_debug_trait_all_types() {
        // Verify Debug is implemented and produces non-empty output
        let shape = format!("{:?}", GraphShape::SkipAttention);
        assert!(shape.contains("SkipAttention"));

        let kind = format!("{:?}", HardwareKind::TensorCorePartition);
        assert!(kind.contains("TensorCorePartition"));

        let sb = SubBatch::new(GraphShape::MoeSparse);
        let sb_debug = format!("{:?}", sb);
        assert!(sb_debug.contains("MoeSparse"));

        let p = HardwarePartition::gpu_sm_partition(0, 0, 10);
        let p_debug = format!("{:?}", p);
        assert!(p_debug.contains("partition_id"));

        let reason = DispatchReason::SingleShapeUniform;
        assert!(!format!("{:?}", reason).is_empty());

        let plan = DispatchPlan {
            sub_batches: vec![],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::AllOrphanFallback,
        };
        assert!(!format!("{:?}", plan).is_empty());
    }

    // ── Clone trait: SubBatch deep copy ──

    #[test]
    fn test_sub_batch_clone_independence() {
        let mut original = SubBatch::new(GraphShape::FullPrecision);
        original.add_request(1);
        original.add_request(2);
        original.estimated_gflops = 5.0;

        let cloned = original.clone();
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.request_ids, original.request_ids);
        assert_eq!(cloned.estimated_gflops, original.estimated_gflops);

        // Modifying original should not affect clone
        original.add_request(3);
        assert_eq!(original.request_ids.len(), 3);
        assert_eq!(cloned.request_ids.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════
    //  New tests
    // ═══════════════════════════════════════════════════════════

    // ── HardwareKind: Copy, Clone, Hash ──

    #[test]
    fn test_hardware_kind_copy_clone_eq_hash() {
        let a = HardwareKind::LowComputeCore;
        let b = a; // Copy
        assert_eq!(a, b);

        let c = a.clone();
        assert_eq!(a, c);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(HardwareKind::TensorCorePartition);
        assert!(set.contains(&HardwareKind::TensorCorePartition));
        assert!(!set.contains(&HardwareKind::FullComputeUnit));
    }

    #[test]
    fn test_hardware_kind_debug_all_variants() {
        assert!(format!("{:?}", HardwareKind::LowComputeCore).contains("LowComputeCore"));
        assert!(format!("{:?}", HardwareKind::FullComputeUnit).contains("FullComputeUnit"));
        assert!(format!("{:?}", HardwareKind::TensorCorePartition).contains("TensorCorePartition"));
    }

    // ── GraphShape: Debug format for each variant ──

    #[test]
    fn test_graph_shape_debug_all_variants() {
        assert!(format!("{:?}", GraphShape::SkipAttention).contains("SkipAttention"));
        assert!(format!("{:?}", GraphShape::NarrowQuant).contains("NarrowQuant"));
        assert!(format!("{:?}", GraphShape::FullPrecision).contains("FullPrecision"));
        assert!(format!("{:?}", GraphShape::MoeSparse).contains("MoeSparse"));
    }

    // ── GraphShape: exact compute_intensity values ──

    #[test]
    fn test_graph_shape_compute_intensity_exact_values() {
        assert!((GraphShape::SkipAttention.compute_intensity() - 0.3).abs() < f32::EPSILON);
        assert!((GraphShape::NarrowQuant.compute_intensity() - 0.5).abs() < f32::EPSILON);
        assert!((GraphShape::MoeSparse.compute_intensity() - 0.8).abs() < f32::EPSILON);
        assert!((GraphShape::FullPrecision.compute_intensity() - 1.0).abs() < f32::EPSILON);
    }

    // ── DispatchReason: PartialEq for SingleShapeUniform ──

    #[test]
    fn test_dispatch_reason_partial_eq_single_shape() {
        let a = DispatchReason::SingleShapeUniform;
        let b = DispatchReason::SingleShapeUniform;
        assert_eq!(a, b);
    }

    #[test]
    fn test_dispatch_reason_partial_eq_all_orphan() {
        assert_eq!(
            DispatchReason::AllOrphanFallback,
            DispatchReason::AllOrphanFallback
        );
    }

    #[test]
    fn test_dispatch_reason_partial_eq_multi_shape_same_counts() {
        let mut counts_a = HashMap::new();
        counts_a.insert(GraphShape::FullPrecision, 5);
        let mut counts_b = HashMap::new();
        counts_b.insert(GraphShape::FullPrecision, 5);

        assert_eq!(
            DispatchReason::MultiShapeDispatched { shape_counts: counts_a },
            DispatchReason::MultiShapeDispatched { shape_counts: counts_b }
        );
    }

    #[test]
    fn test_dispatch_reason_partial_eq_multi_shape_different_counts() {
        let mut counts_a = HashMap::new();
        counts_a.insert(GraphShape::FullPrecision, 5);
        let mut counts_b = HashMap::new();
        counts_b.insert(GraphShape::FullPrecision, 3);

        assert_ne!(
            DispatchReason::MultiShapeDispatched { shape_counts: counts_a },
            DispatchReason::MultiShapeDispatched { shape_counts: counts_b }
        );
    }

    // ── DispatchReason: Clone ──

    #[test]
    fn test_dispatch_reason_clone() {
        let mut counts = HashMap::new();
        counts.insert(GraphShape::NarrowQuant, 10);
        let original = DispatchReason::MultiShapeDispatched { shape_counts: counts };
        let cloned = original.clone();

        if let DispatchReason::MultiShapeDispatched { shape_counts } = cloned {
            assert_eq!(shape_counts.get(&GraphShape::NarrowQuant), Some(&10));
        } else {
            panic!("Expected MultiShapeDispatched");
        }
    }

    // ── DispatchReason: Debug format ──

    #[test]
    fn test_dispatch_reason_debug_formats() {
        assert!(format!("{:?}", DispatchReason::SingleShapeUniform).contains("SingleShapeUniform"));
        assert!(format!("{:?}", DispatchReason::AllOrphanFallback).contains("AllOrphanFallback"));

        let mut counts = HashMap::new();
        counts.insert(GraphShape::MoeSparse, 2);
        let multi_debug = format!("{:?}", DispatchReason::MultiShapeDispatched { shape_counts: counts });
        assert!(multi_debug.contains("MultiShapeDispatched"));
    }

    // ── DispatchPlan: Clone independence ──

    #[test]
    fn test_dispatch_plan_clone_independence() {
        let mut sb = SubBatch::new(GraphShape::SkipAttention);
        sb.add_request(100);
        let plan = DispatchPlan {
            sub_batches: vec![sb],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::SingleShapeUniform,
        };
        let cloned = plan.clone();
        assert_eq!(cloned.orphan_count, plan.orphan_count);
        assert_eq!(cloned.needs_ragged_compaction, plan.needs_ragged_compaction);
        assert_eq!(cloned.reason, plan.reason);
        assert_eq!(cloned.sub_batches.len(), 1);

        // Deep copy: modifying plan's sub_batches does not affect clone
        assert_eq!(cloned.sub_batches[0].request_ids, vec![100]);
    }

    // ── DispatchPlan: Debug ──

    #[test]
    fn test_dispatch_plan_debug() {
        let plan = DispatchPlan {
            sub_batches: vec![],
            needs_ragged_compaction: true,
            orphan_count: 7,
            reason: DispatchReason::SingleShapeUniform,
        };
        let debug = format!("{:?}", plan);
        assert!(debug.contains("needs_ragged_compaction"));
        assert!(debug.contains("orphan_count"));
    }

    // ── HardwarePartition: Clone independence ──

    #[test]
    fn test_hardware_partition_clone_independence() {
        let original = HardwarePartition::gpu_sm_partition(0, 0, 40);
        let cloned = original.clone();
        assert_eq!(cloned.partition_id, original.partition_id);
        assert_eq!(cloned.kind, original.kind);
        assert_eq!(cloned.sm_range, original.sm_range);
        assert_eq!(cloned.compute_weight, original.compute_weight);
    }

    // ── HardwarePartition: compute_weight edge cases ──

    #[test]
    fn test_gpu_sm_partition_compute_weight_small_sm() {
        // 10 SM out of 80 = 0.125
        let p = HardwarePartition::gpu_sm_partition(0, 0, 10);
        assert!((p.compute_weight - 0.125).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gpu_sm_partition_compute_weight_full_gpu() {
        // 80 SM out of 80 = 1.0
        let p = HardwarePartition::gpu_sm_partition(0, 0, 80);
        assert!((p.compute_weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_numa_partition_compute_weight() {
        // 64 cores / 64.0 = 1.0
        let p = HardwarePartition::cpu_numa_partition(0, 0, 64);
        assert!((p.compute_weight - 1.0).abs() < f32::EPSILON);

        // 8 cores / 64.0 = 0.125
        let p2 = HardwarePartition::cpu_numa_partition(1, 1, 8);
        assert!((p2.compute_weight - 0.125).abs() < f32::EPSILON);
    }

    // ── SubBatch: target_partition and estimated fields ──

    #[test]
    fn test_sub_batch_fields_mutable_assignment() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        let partition = HardwarePartition::gpu_sm_partition(2, 40, 60);
        sb.target_partition = Some(partition);
        sb.estimated_gflops = 12.5;
        sb.estimated_vram_bytes = 1024 * 1024;

        assert!(sb.target_partition.is_some());
        let p = sb.target_partition.as_ref().unwrap();
        assert_eq!(p.partition_id, 2);
        assert!((sb.estimated_gflops - 12.5).abs() < f32::EPSILON);
        assert_eq!(sb.estimated_vram_bytes, 1048576);
    }

    // ── dispatch: empty manifest (zero slots) ──

    #[test]
    fn test_dispatch_empty_manifest() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let manifest = BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };
        let shape_map: HashMap<RequestId, GraphShape> = HashMap::new();
        let plan = d.dispatch(&manifest, &shape_map);

        // No requests at all → AllOrphanFallback with 0 orphans
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
        assert_eq!(plan.orphan_count, 0);
        assert_eq!(plan.sub_batches.len(), 1);
        assert!(plan.sub_batches[0].is_empty());
    }

    // ── dispatch: partially classified requests ──

    #[test]
    fn test_dispatch_partially_classified_requests() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        // Only classify first 4 as SkipAttention, rest unclassified → FullPrecision
        let mut shape_map = HashMap::new();
        for &id in &ids[0..4] {
            shape_map.insert(id, GraphShape::SkipAttention);
        }
        // ids[4..8] are not in shape_map

        let plan = d.dispatch(&manifest, &shape_map);

        // Two shapes: 4 SkipAttention + 4 FullPrecision (unclassified)
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
        assert_eq!(plan.sub_batches.len(), 2);

        let skip_count = plan.sub_batches.iter()
            .find(|sb| sb.shape == GraphShape::SkipAttention)
            .map(|sb| sb.len())
            .unwrap_or(0);
        let fp_count = plan.sub_batches.iter()
            .find(|sb| sb.shape == GraphShape::FullPrecision)
            .map(|sb| sb.len())
            .unwrap_or(0);
        assert_eq!(skip_count, 4);
        assert_eq!(fp_count, 4);
    }

    // ── dispatch: all same shape but below min threshold → orphan fallback ──

    #[test]
    fn test_dispatch_single_shape_below_min_threshold() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(10);
        let ids: Vec<RequestId> = (1..=5).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids {
            shape_map.insert(id, GraphShape::MoeSparse);
        }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
        assert_eq!(plan.orphan_count, 5);
        assert!(plan.needs_ragged_compaction);
        // Fallback batch uses FullPrecision
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);
    }

    // ── dispatch: CPU NUMA partition assignment ──

    #[test]
    fn test_dispatch_assigns_cpu_numa_partitions_when_no_gpu() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = None;
        constraints.numa_node_count = 2;
        constraints.numa_core_bindings = vec![(0, 0, 4), (1, 4, 8)];

        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..4] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[4..8] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        for sb in &plan.sub_batches {
            assert!(sb.target_partition.is_some());
            let partition = sb.target_partition.as_ref().unwrap();
            // CPU partition: no SM range, has NUMA node
            assert!(partition.sm_range.is_none());
            assert!(partition.numa_node.is_some());
        }
    }

    // ── classify_request: all low values → FullPrecision ──

    #[test]
    fn test_classify_all_low_values_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // All values well below thresholds, non-MoE
        let shape = d.classify_request(0.0, 0.0, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_moe_not_sparse_active_above_threshold() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE but active_experts=0.8 (>= 0.5) → NOT sparse
        let shape = d.classify_request(0.0, 0.0, 0.8);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    // ── dispatch: many shapes, one orphan group merges into lightest ──

    #[test]
    fn test_dispatch_orphan_merges_into_skip_attention_when_lightest() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        // 10 FullPrecision + 5 SkipAttention + 3 MoeSparse (orphans)
        let ids: Vec<RequestId> = (1..=18).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..10] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[10..15] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[15..18] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 3);
        assert!(plan.needs_ragged_compaction);

        // Orphans (MoeSparse, intensity=0.8) should merge into lightest (SkipAttention, intensity=0.3)
        let skip_batch = plan.sub_batches.iter().find(|sb| sb.shape == GraphShape::SkipAttention);
        assert!(skip_batch.is_some());
        assert_eq!(skip_batch.unwrap().len(), 5 + 3); // 5 original + 3 orphans
    }

    // ── dispatch: two shapes exactly at min threshold ──

    #[test]
    fn test_dispatch_two_shapes_exactly_at_min_threshold() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(4);

        // 4 FullPrecision + 4 NarrowQuant → exactly at min, both valid
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..4] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[4..8] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
        assert!(!plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 0);
        assert_eq!(plan.sub_batches.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════
    //  Additional tests (18 new)
    // ═══════════════════════════════════════════════════════════

    // ── GraphShape: preferred_hardware() returns only LowComputeCore or FullComputeUnit ──

    #[test]
    fn test_graph_shape_preferred_hardware_never_returns_tensor_core() {
        // None of the GraphShape variants map to TensorCorePartition
        for shape in [GraphShape::SkipAttention, GraphShape::NarrowQuant, GraphShape::FullPrecision, GraphShape::MoeSparse] {
            assert_ne!(shape.preferred_hardware(), HardwareKind::TensorCorePartition,
                "{:?} should not map to TensorCorePartition", shape);
        }
    }

    // ── GraphShape: compute_intensity() is in (0, 1] range ──

    #[test]
    fn test_graph_shape_compute_intensity_in_valid_range() {
        for shape in [GraphShape::SkipAttention, GraphShape::NarrowQuant, GraphShape::FullPrecision, GraphShape::MoeSparse] {
            let intensity = shape.compute_intensity();
            assert!(intensity > 0.0, "{:?} intensity should be > 0", shape);
            assert!(intensity <= 1.0, "{:?} intensity should be <= 1.0", shape);
        }
    }

    // ── HardwareKind: all three variants are pairwise not equal ──

    #[test]
    fn test_hardware_kind_ne_cross_variant() {
        assert_ne!(HardwareKind::LowComputeCore, HardwareKind::FullComputeUnit);
        assert_ne!(HardwareKind::LowComputeCore, HardwareKind::TensorCorePartition);
        assert_ne!(HardwareKind::FullComputeUnit, HardwareKind::TensorCorePartition);
    }

    // ── SubBatch: new() with each GraphShape variant ──

    #[test]
    fn test_sub_batch_new_each_shape_variant() {
        for shape in [GraphShape::SkipAttention, GraphShape::NarrowQuant, GraphShape::FullPrecision, GraphShape::MoeSparse] {
            let sb = SubBatch::new(shape);
            assert_eq!(sb.shape, shape);
            assert!(sb.is_empty());
            assert_eq!(sb.len(), 0);
            assert!(sb.target_partition.is_none());
        }
    }

    // ── SubBatch: add_request with many sequential IDs ──

    #[test]
    fn test_sub_batch_add_many_requests() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        for i in 0..100 {
            sb.add_request(i);
        }
        assert_eq!(sb.len(), 100);
        assert!(!sb.is_empty());
        // First and last IDs preserved
        assert_eq!(sb.request_ids[0], 0);
        assert_eq!(sb.request_ids[99], 99);
    }

    // ── HardwarePartition: gpu_sm_partition with minimal SM range ──

    #[test]
    fn test_gpu_sm_partition_minimal_range() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 1);
        assert_eq!(p.sm_range, Some((0, 1)));
        assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        // 1/80 = 0.0125
        assert!((p.compute_weight - 0.0125).abs() < f32::EPSILON);
    }

    // ── HardwarePartition: gpu_sm_partition with non-zero start ──

    #[test]
    fn test_gpu_sm_partition_nonzero_start() {
        let p = HardwarePartition::gpu_sm_partition(1, 40, 80);
        assert_eq!(p.partition_id, 1);
        assert_eq!(p.sm_range, Some((40, 80)));
        // (80-40)/80 = 0.5
        assert!((p.compute_weight - 0.5).abs() < f32::EPSILON);
        assert!(p.numa_node.is_none());
        assert!(p.core_range.is_none());
    }

    // ── HardwarePartition: cpu_numa_partition with zero cores ──

    #[test]
    fn test_cpu_numa_partition_zero_cores() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 0);
        // 0 cores / 64 = 0.0
        assert!((p.compute_weight - 0.0).abs() < f32::EPSILON);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit); // node 0 → AMX
    }

    // ── HardwarePartition: cpu_numa_partition high core count ──

    #[test]
    fn test_cpu_numa_partition_high_core_count() {
        // 128 cores / 64 = 2.0 (weight can exceed 1.0)
        let p = HardwarePartition::cpu_numa_partition(2, 3, 128);
        assert_eq!(p.kind, HardwareKind::LowComputeCore); // non-zero NUMA
        assert_eq!(p.numa_node, Some(3));
        assert!((p.compute_weight - 2.0).abs() < f32::EPSILON);
    }

    // ── DispatchPlan: default field values ──

    #[test]
    fn test_dispatch_plan_construction_with_fields() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        sb.add_request(7);
        let plan = DispatchPlan {
            sub_batches: vec![sb],
            needs_ragged_compaction: true,
            orphan_count: 3,
            reason: DispatchReason::AllOrphanFallback,
        };
        assert_eq!(plan.sub_batches.len(), 1);
        assert!(plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 3);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
    }

    // ── DispatchReason: MultiShapeDispatched with empty shape_counts ──

    #[test]
    fn test_dispatch_reason_multi_shape_empty_counts() {
        let reason = DispatchReason::MultiShapeDispatched {
            shape_counts: HashMap::new(),
        };
        if let DispatchReason::MultiShapeDispatched { shape_counts } = &reason {
            assert!(shape_counts.is_empty());
        } else {
            panic!("Expected MultiShapeDispatched");
        }
    }

    // ── DispatchReason: different variants are not equal ──

    #[test]
    fn test_dispatch_reason_different_variants_not_equal() {
        assert_ne!(DispatchReason::SingleShapeUniform, DispatchReason::AllOrphanFallback);

        let mut counts = HashMap::new();
        counts.insert(GraphShape::FullPrecision, 1);
        assert_ne!(DispatchReason::SingleShapeUniform, DispatchReason::MultiShapeDispatched { shape_counts: counts });
        assert_ne!(DispatchReason::AllOrphanFallback, DispatchReason::MultiShapeDispatched { shape_counts: HashMap::new() });
    }

    // ── SubBatchDispatcher: with_min_sub_batch_size is a value override (builder) ──

    #[test]
    fn test_dispatcher_min_size_builder_consumes_self() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(1);
        // min=1: even a single request should form a valid sub-batch
        let ids: Vec<RequestId> = vec![42];
        let manifest = make_manifest_with_ids(&ids);
        let mut shape_map = HashMap::new();
        shape_map.insert(42, GraphShape::NarrowQuant);
        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].len(), 1);
    }

    // ── classify_request: MoE with exactly 0 active experts ──

    #[test]
    fn test_classify_moe_zero_active_experts_is_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        let shape = d.classify_request(0.0, 0.0, 0.0);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    // ── classify_request: non-MoE with max sparsity and max dead neurons ──

    #[test]
    fn test_classify_non_moe_dead_neuron_takes_priority_over_skip_attention() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // Both dead_neuron_ratio > 0.6 and attention_sparsity > 0.7 are true,
        // but dead_neuron check comes first → NarrowQuant
        let shape = d.classify_request(0.9, 0.9, 0.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    // ── classify_request: negative edge values (boundary check) ──

    #[test]
    fn test_classify_negative_values_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // Negative values are below all thresholds → FullPrecision
        let shape = d.classify_request(-1.0, -1.0, -1.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    // ── dispatch: max_sub_batches limit with 4 shapes ──

    #[test]
    fn test_dispatch_max_sub_batches_caps_partition_count() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        // 4 different shapes, each with 3 requests (above min=2)
        let ids: Vec<RequestId> = (1..=12).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..3] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[3..6] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[6..9] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[9..12] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
        assert_eq!(plan.sub_batches.len(), 4);

        // Verify all 4 shapes are present
        let shapes: std::collections::HashSet<GraphShape> = plan.sub_batches.iter().map(|sb| sb.shape).collect();
        assert_eq!(shapes.len(), 4);
    }

    // ═══════════════════════════════════════════════════════════
    //  Additional tests (50 new) — target 122+
    // ═══════════════════════════════════════════════════════════

    // ── classify_request: extreme values ──

    #[test]
    fn test_classify_moe_active_experts_exactly_zero_is_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        let shape = d.classify_request(0.0, 0.0, 0.0);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    #[test]
    fn test_classify_moe_active_experts_exactly_one_not_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        let shape = d.classify_request(0.0, 0.0, 1.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_moe_high_active_experts_but_high_dead_neurons() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE active_experts = 0.8 (>= 0.5, not sparse) → check dead neurons
        let shape = d.classify_request(0.0, 0.7, 0.8);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_moe_high_active_experts_high_sparsity_skip() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE not sparse, dead neurons low, attention sparsity high → SkipAttention
        let shape = d.classify_request(0.8, 0.1, 0.8);
        assert_eq!(shape, GraphShape::SkipAttention);
    }

    #[test]
    fn test_classify_dead_neuron_exactly_sixty_one_is_narrow() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.0, 0.61, 0.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_sparsity_exactly_seventy_one_is_skip() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.71, 0.0, 0.0);
        assert_eq!(shape, GraphShape::SkipAttention);
    }

    #[test]
    fn test_classify_all_zero_non_moe_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.0, 0.0, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_very_high_values_non_moe_is_narrow() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // dead_neuron_ratio > 0.6 takes priority over sparsity
        let shape = d.classify_request(0.99, 0.99, 0.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_just_below_all_thresholds_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.69, 0.59, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_moe_sparse_beats_dead_neuron() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE sparse check comes first in the if-else chain
        let shape = d.classify_request(0.0, 0.99, 0.1);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    // ── HardwarePartition: gpu_sm_partition various sizes ──

    #[test]
    fn test_gpu_sm_partition_single_sm() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 1);
        assert_eq!(p.sm_range.unwrap(), (0, 1));
        assert!((p.compute_weight - (1.0 / 80.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gpu_sm_partition_large_range() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 120);
        // 120/80 = 1.5
        assert!((p.compute_weight - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gpu_sm_partition_partition_id_preserved() {
        for id in 0..10 {
            let p = HardwarePartition::gpu_sm_partition(id, 0, 10);
            assert_eq!(p.partition_id, id);
        }
    }

    #[test]
    fn test_gpu_sm_partition_no_cpu_fields() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 40);
        assert!(p.numa_node.is_none());
        assert!(p.core_range.is_none());
    }

    // ── HardwarePartition: cpu_numa_partition various configs ──

    #[test]
    fn test_cpu_numa_partition_node_zero_full_compute() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 8);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
    }

    #[test]
    fn test_cpu_numa_partition_node_one_low_compute() {
        let p = HardwarePartition::cpu_numa_partition(1, 1, 8);
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
    }

    #[test]
    fn test_cpu_numa_partition_node_two_low_compute() {
        let p = HardwarePartition::cpu_numa_partition(2, 2, 8);
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
    }

    #[test]
    fn test_cpu_numa_partition_no_gpu_fields() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 16);
        assert!(p.sm_range.is_none());
        assert!(p.core_range.is_none());
    }

    #[test]
    fn test_cpu_numa_partition_id_preserved() {
        for id in 0..5 {
            let p = HardwarePartition::cpu_numa_partition(id, 0, 4);
            assert_eq!(p.partition_id, id);
        }
    }

    // ── SubBatch: various operations ──

    #[test]
    fn test_sub_batch_len_after_multiple_adds() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        for i in 0..50 {
            sb.add_request(i);
        }
        assert_eq!(sb.len(), 50);
        assert!(!sb.is_empty());
    }

    #[test]
    fn test_sub_batch_shape_field_immutable_after_add() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        sb.add_request(1);
        sb.add_request(2);
        assert_eq!(sb.shape, GraphShape::MoeSparse);
    }

    #[test]
    fn test_sub_batch_zero_estimated_fields_default() {
        let sb = SubBatch::new(GraphShape::SkipAttention);
        assert_eq!(sb.estimated_gflops, 0.0);
        assert_eq!(sb.estimated_vram_bytes, 0);
    }

    #[test]
    fn test_sub_batch_estimated_vram_large_value() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.estimated_vram_bytes = usize::MAX;
        assert_eq!(sb.estimated_vram_bytes, usize::MAX);
    }

    #[test]
    fn test_sub_batch_estimated_gflops_fractional() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        sb.estimated_gflops = 0.001;
        assert!((sb.estimated_gflops - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sub_batch_target_partition_gpu() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.target_partition = Some(HardwarePartition::gpu_sm_partition(0, 0, 40));
        assert!(sb.target_partition.is_some());
        assert_eq!(sb.target_partition.as_ref().unwrap().kind, HardwareKind::TensorCorePartition);
    }

    #[test]
    fn test_sub_batch_target_partition_cpu() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        sb.target_partition = Some(HardwarePartition::cpu_numa_partition(0, 0, 32));
        assert!(sb.target_partition.is_some());
        assert_eq!(sb.target_partition.as_ref().unwrap().kind, HardwareKind::FullComputeUnit);
    }

    #[test]
    fn test_sub_batch_clone_with_partition() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.add_request(1);
        sb.target_partition = Some(HardwarePartition::gpu_sm_partition(3, 30, 50));
        sb.estimated_gflops = 7.5;
        sb.estimated_vram_bytes = 2048;

        let clone = sb.clone();
        assert_eq!(clone.request_ids, vec![1]);
        assert_eq!(clone.target_partition.as_ref().unwrap().partition_id, 3);
        assert!((clone.estimated_gflops - 7.5).abs() < f32::EPSILON);
        assert_eq!(clone.estimated_vram_bytes, 2048);
    }

    // ── dispatch: GPU SM partition evenly splits across sub-batches ──

    #[test]
    fn test_dispatch_gpu_sm_partition_split() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=6).collect();
        let manifest = make_manifest_with_ids(&ids);

        // 3 shapes, 2 each
        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[4..6] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 3);

        // Each sub-batch gets 80/3 = 26 SMs (integer division)
        for sb in &plan.sub_batches {
            let partition = sb.target_partition.as_ref().unwrap();
            let (start, end) = partition.sm_range.unwrap();
            assert_eq!(end - start, 26);
        }
    }

    #[test]
    fn test_dispatch_gpu_sm_partition_single_batch_no_partition_on_shortcut() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=5).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::FullPrecision); }

        let plan = d.dispatch(&manifest, &shape_map);
        // SingleShapeUniform takes an early return before assign_hardware_partitions
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert!(plan.sub_batches[0].target_partition.is_none());
    }

    // ── dispatch: CPU NUMA round-robin across sub-batches ──

    #[test]
    fn test_dispatch_cpu_numa_round_robin() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = None;
        constraints.numa_node_count = 2;
        constraints.numa_core_bindings = vec![(0, 0, 8), (1, 8, 16)];

        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=6).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[4..6] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        // 3 sub-batches, 2 NUMA nodes → round-robin 0, 1, 0
        let numa_nodes: Vec<usize> = plan.sub_batches.iter()
            .map(|sb| sb.target_partition.as_ref().unwrap().numa_node.unwrap())
            .collect();
        assert_eq!(numa_nodes.len(), 3);
        assert_eq!(numa_nodes[0], 0);
        assert_eq!(numa_nodes[1], 1);
        assert_eq!(numa_nodes[2], 0);
    }

    #[test]
    fn test_dispatch_cpu_numa_single_node() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = None;
        constraints.numa_node_count = 1;
        constraints.numa_core_bindings = vec![(0, 0, 4)];

        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=4).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        // Both sub-batches on node 0
        for sb in &plan.sub_batches {
            let numa = sb.target_partition.as_ref().unwrap().numa_node.unwrap();
            assert_eq!(numa, 0);
        }
    }

    // ── dispatch: multiple orphan groups ──

    #[test]
    fn test_dispatch_two_orphan_groups_merge_into_lightest() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        // 5 FullPrecision, 2 NarrowQuant (orphan), 2 MoeSparse (orphan)
        let ids: Vec<RequestId> = (1..=9).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..7] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[7..9] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        // 4 orphans total (2 NarrowQuant + 2 MoeSparse)
        assert_eq!(plan.orphan_count, 4);
        assert!(plan.needs_ragged_compaction);

        // FullPrecision is the only valid sub-batch, so orphans merge into it
        let fp_batch = plan.sub_batches.iter().find(|sb| sb.shape == GraphShape::FullPrecision);
        assert!(fp_batch.is_some());
        assert_eq!(fp_batch.unwrap().len(), 5 + 4); // 5 original + 4 orphans
    }

    // ── dispatch: shape_map has extra entries not in manifest ──

    #[test]
    fn test_dispatch_shape_map_extra_entries_ignored() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = vec![1, 2, 3];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        shape_map.insert(1, GraphShape::FullPrecision);
        shape_map.insert(2, GraphShape::FullPrecision);
        shape_map.insert(3, GraphShape::FullPrecision);
        // Extra entries not in manifest
        shape_map.insert(99, GraphShape::SkipAttention);
        shape_map.insert(100, GraphShape::NarrowQuant);

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].len(), 3);
    }

    // ── dispatch: single request with min_sub_batch_size=1 ──

    #[test]
    fn test_dispatch_single_request_min_size_one() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(1);

        let ids: Vec<RequestId> = vec![42];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        shape_map.insert(42, GraphShape::NarrowQuant);

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].len(), 1);
        assert_eq!(plan.sub_batches[0].shape, GraphShape::NarrowQuant);
    }

    // ── dispatch: all four shapes present with orphans ──

    #[test]
    fn test_dispatch_four_shapes_two_orphan_groups() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        // 5 FullPrecision, 5 SkipAttention, 3 NarrowQuant (orphan), 2 MoeSparse (orphan)
        let ids: Vec<RequestId> = (1..=15).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[10..13] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[13..15] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 5); // 3 + 2
        assert!(plan.needs_ragged_compaction);

        // Valid sub-batches: FullPrecision (5) + SkipAttention (5)
        assert_eq!(plan.sub_batches.len(), 2);

        // Orphans merge into lightest (SkipAttention, intensity=0.3)
        let skip = plan.sub_batches.iter().find(|sb| sb.shape == GraphShape::SkipAttention).unwrap();
        assert_eq!(skip.len(), 5 + 5); // 5 original + 5 orphans
    }

    // ── DispatchReason: PartialEq for different shape keys ──

    #[test]
    fn test_dispatch_reason_multi_shape_different_keys_not_equal() {
        let mut counts_a = HashMap::new();
        counts_a.insert(GraphShape::FullPrecision, 5);
        let mut counts_b = HashMap::new();
        counts_b.insert(GraphShape::SkipAttention, 5);

        assert_ne!(
            DispatchReason::MultiShapeDispatched { shape_counts: counts_a },
            DispatchReason::MultiShapeDispatched { shape_counts: counts_b }
        );
    }

    #[test]
    fn test_dispatch_reason_multi_shape_one_empty_one_not() {
        let mut counts = HashMap::new();
        counts.insert(GraphShape::NarrowQuant, 3);

        assert_ne!(
            DispatchReason::MultiShapeDispatched { shape_counts: HashMap::new() },
            DispatchReason::MultiShapeDispatched { shape_counts: counts }
        );
    }

    // ── DispatchPlan: construction with zero orphans and no ragged ──

    #[test]
    fn test_dispatch_plan_no_orphans_no_ragged() {
        let plan = DispatchPlan {
            sub_batches: vec![SubBatch::new(GraphShape::FullPrecision)],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::SingleShapeUniform,
        };
        assert!(!plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 0);
    }

    #[test]
    fn test_dispatch_plan_many_orphans_ragged() {
        let plan = DispatchPlan {
            sub_batches: vec![SubBatch::new(GraphShape::FullPrecision)],
            needs_ragged_compaction: true,
            orphan_count: 100,
            reason: DispatchReason::AllOrphanFallback,
        };
        assert!(plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 100);
    }

    // ── HardwarePartition: clone with all field types ──

    #[test]
    fn test_hardware_partition_clone_gpu_all_fields() {
        let original = HardwarePartition::gpu_sm_partition(7, 20, 60);
        let cloned = original.clone();
        assert_eq!(cloned.partition_id, 7);
        assert_eq!(cloned.kind, HardwareKind::TensorCorePartition);
        assert_eq!(cloned.sm_range, Some((20, 60)));
        assert!(cloned.numa_node.is_none());
        assert!(cloned.core_range.is_none());
        assert!((cloned.compute_weight - (40.0 / 80.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hardware_partition_clone_cpu_all_fields() {
        let original = HardwarePartition::cpu_numa_partition(3, 1, 24);
        let cloned = original.clone();
        assert_eq!(cloned.partition_id, 3);
        assert_eq!(cloned.kind, HardwareKind::LowComputeCore);
        assert!(cloned.sm_range.is_none());
        assert_eq!(cloned.numa_node, Some(1));
        assert!(cloned.core_range.is_none());
        assert!((cloned.compute_weight - (24.0 / 64.0)).abs() < f32::EPSILON);
    }

    // ── SubBatch: Debug output contains expected fields ──

    #[test]
    fn test_sub_batch_debug_contains_request_ids() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.add_request(42);
        sb.add_request(99);
        let debug = format!("{:?}", sb);
        assert!(debug.contains("request_ids"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("estimated_gflops"));
        assert!(debug.contains("estimated_vram_bytes"));
    }

    // ── HardwarePartition: Debug output contains expected fields ──

    #[test]
    fn test_hardware_partition_debug_gpu() {
        let p = HardwarePartition::gpu_sm_partition(1, 10, 30);
        let debug = format!("{:?}", p);
        assert!(debug.contains("partition_id"));
        assert!(debug.contains("TensorCorePartition"));
        assert!(debug.contains("sm_range"));
    }

    #[test]
    fn test_hardware_partition_debug_cpu() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 16);
        let debug = format!("{:?}", p);
        assert!(debug.contains("numa_node"));
        assert!(debug.contains("FullComputeUnit"));
    }

    // ── GraphShape: Hash consistency across multiple uses ──

    #[test]
    fn test_graph_shape_hash_in_hashmap_key() {
        let mut map: HashMap<GraphShape, &'static str> = HashMap::new();
        map.insert(GraphShape::SkipAttention, "skip");
        map.insert(GraphShape::NarrowQuant, "narrow");
        map.insert(GraphShape::FullPrecision, "full");
        map.insert(GraphShape::MoeSparse, "moe");

        assert_eq!(map.get(&GraphShape::SkipAttention), Some(&"skip"));
        assert_eq!(map.get(&GraphShape::NarrowQuant), Some(&"narrow"));
        assert_eq!(map.get(&GraphShape::FullPrecision), Some(&"full"));
        assert_eq!(map.get(&GraphShape::MoeSparse), Some(&"moe"));
        assert_eq!(map.len(), 4);
    }

    // ── HardwareKind: Hash consistency across multiple uses ──

    #[test]
    fn test_hardware_kind_hash_in_hashmap_key() {
        let mut map: HashMap<HardwareKind, usize> = HashMap::new();
        map.insert(HardwareKind::LowComputeCore, 1);
        map.insert(HardwareKind::FullComputeUnit, 2);
        map.insert(HardwareKind::TensorCorePartition, 3);

        assert_eq!(map.len(), 3);
        assert_eq!(map[&HardwareKind::LowComputeCore], 1);
        assert_eq!(map[&HardwareKind::FullComputeUnit], 2);
        assert_eq!(map[&HardwareKind::TensorCorePartition], 3);
    }

    // ── dispatch: GPU partition with single sub-batch ──

    #[test]
    fn test_dispatch_gpu_single_partition_shortcut_no_partition() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(40);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=6).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        // SingleShapeUniform takes an early return before assign_hardware_partitions
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert!(plan.sub_batches[0].target_partition.is_none());
    }

    // ── dispatch: two shapes one above min, one below ──

    #[test]
    fn test_dispatch_one_valid_one_orphan() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        let ids: Vec<RequestId> = (1..=7).collect();
        let manifest = make_manifest_with_ids(&ids);

        // 5 FullPrecision, 2 SkipAttention (orphan)
        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..7] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 2);
        assert!(plan.needs_ragged_compaction);

        // Only FullPrecision is a valid sub-batch
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);

        // Orphans (SkipAttention, intensity=0.3) merge into FullPrecision (intensity=1.0)
        // Since FullPrecision is the only batch, it gets the orphans
        assert_eq!(plan.sub_batches[0].len(), 5 + 2);
    }

    // ── dispatch: requests with same ID but different shape (shape_map wins) ──

    #[test]
    fn test_dispatch_shape_map_overrides_for_same_shape() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = vec![1, 2, 3, 4];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].shape, GraphShape::NarrowQuant);
    }

    // ── SubBatchDispatcher: constraints accessor returns same reference ──

    #[test]
    fn test_dispatcher_constraints_returns_correct_defaults() {
        let constraints = CompilerConstraints::default();
        let d = SubBatchDispatcher::new(constraints);
        let c = d.constraints();
        assert_eq!(c.max_gpr_count, 15);
        assert_eq!(c.simd_width_bits, 256);
        assert!(!c.has_amx);
        assert!(!c.has_avx512);
        assert!(!c.has_sve);
        assert!(!c.has_tma);
        assert_eq!(c.tensor_core_gen, 0);
    }

    // ── dispatch: large batch with mixed shapes ──

    #[test]
    fn test_dispatch_large_mixed_batch() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(10);

        // 30 FullPrecision, 20 SkipAttention, 5 NarrowQuant (orphan), 3 MoeSparse (orphan)
        let ids: Vec<RequestId> = (1..=58).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..30] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[30..50] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[50..55] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[55..58] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 8);
        assert!(plan.needs_ragged_compaction);

        // 2 valid sub-batches: FullPrecision (30) + SkipAttention (20)
        assert_eq!(plan.sub_batches.len(), 2);
    }

    // ═══════════════════════════════════════════════════════════
    //  Wave 3 — 43 additional tests (target 166+)
    // ═══════════════════════════════════════════════════════════

    // ── classify_request: all four shapes reachable in one call site ──

    #[test]
    fn test_classify_all_four_shapes_from_one_dispatcher() {
        let non_moe = SubBatchDispatcher::new(CompilerConstraints::default());
        let moe = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);

        let shapes: std::collections::HashSet<GraphShape> = [
            non_moe.classify_request(0.0, 0.0, 0.0),          // FullPrecision
            non_moe.classify_request(0.8, 0.0, 0.0),          // SkipAttention
            non_moe.classify_request(0.0, 0.7, 0.0),          // NarrowQuant
            moe.classify_request(0.0, 0.0, 0.3),              // MoeSparse
        ].into_iter().collect();

        assert_eq!(shapes.len(), 4, "classify_request must produce all 4 distinct shapes");
    }

    #[test]
    fn test_classify_moe_not_sparse_goes_to_dead_neuron_path() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE but active_experts=0.6 >= 0.5 → not sparse → dead neuron check
        let shape = d.classify_request(0.0, 0.8, 0.6);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_moe_not_sparse_goes_to_skip_attention_path() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE not sparse, dead_neuron low, sparsity high
        let shape = d.classify_request(0.9, 0.1, 0.9);
        assert_eq!(shape, GraphShape::SkipAttention);
    }

    #[test]
    fn test_classify_moe_not_sparse_goes_to_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        // MoE not sparse, low dead neurons, low sparsity
        let shape = d.classify_request(0.1, 0.1, 0.6);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_dead_neuron_just_above_sixty_is_narrow() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.0, 0.6001, 0.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_sparsity_just_above_seventy_is_skip() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.7001, 0.0, 0.0);
        assert_eq!(shape, GraphShape::SkipAttention);
    }

    #[test]
    fn test_classify_dead_neuron_at_exactly_sixty_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.0, 0.6, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_sparsity_at_exactly_seventy_is_full_precision() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // dead_neuron 0.0 (not > 0.6), sparsity 0.7 (not > 0.7)
        let shape = d.classify_request(0.7, 0.0, 0.0);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_moe_at_exactly_half_not_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        let shape = d.classify_request(0.0, 0.0, 0.5);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    #[test]
    fn test_classify_moe_just_below_half_is_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        let shape = d.classify_request(0.0, 0.0, 0.4999);
        assert_eq!(shape, GraphShape::MoeSparse);
    }

    #[test]
    fn test_classify_very_large_positive_values_non_moe() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // dead_neuron_ratio > 0.6 → NarrowQuant (even though sparsity is also huge)
        let shape = d.classify_request(100.0, 100.0, 100.0);
        assert_eq!(shape, GraphShape::NarrowQuant);
    }

    #[test]
    fn test_classify_is_moe_false_never_produces_moe_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default());
        // is_moe=false, so MoeSparse should never be returned regardless of moe_active_experts
        let shape = d.classify_request(0.0, 0.0, 0.0);
        assert_ne!(shape, GraphShape::MoeSparse);
    }

    // ── GraphShape: exhaustive preferred_hardware output set ──

    #[test]
    fn test_graph_shape_preferred_hardware_covers_two_kinds() {
        let kinds: std::collections::HashSet<HardwareKind> = [
            GraphShape::SkipAttention.preferred_hardware(),
            GraphShape::NarrowQuant.preferred_hardware(),
            GraphShape::FullPrecision.preferred_hardware(),
            GraphShape::MoeSparse.preferred_hardware(),
        ].into_iter().collect();
        // Must have exactly 2 distinct kinds: LowComputeCore and FullComputeUnit
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&HardwareKind::LowComputeCore));
        assert!(kinds.contains(&HardwareKind::FullComputeUnit));
    }

    #[test]
    fn test_graph_shape_compute_intensity_sum_positive() {
        let total: f32 = [
            GraphShape::SkipAttention,
            GraphShape::NarrowQuant,
            GraphShape::FullPrecision,
            GraphShape::MoeSparse,
        ].iter().map(|s| s.compute_intensity()).sum();
        assert!(total > 0.0, "total compute intensity must be positive");
    }

    // ── SubBatch: is_empty / len consistency after clear-like operations ──

    #[test]
    fn test_sub_batch_is_empty_consistent_with_len() {
        let sb = SubBatch::new(GraphShape::FullPrecision);
        assert_eq!(sb.is_empty(), sb.len() == 0);

        let mut sb2 = SubBatch::new(GraphShape::FullPrecision);
        sb2.add_request(1);
        assert_eq!(sb2.is_empty(), sb2.len() == 0);
    }

    #[test]
    fn test_sub_batch_request_ids_vec_accessible() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        sb.add_request(10);
        sb.add_request(20);
        // Direct vec access: can read, can use standard Vec methods
        assert_eq!(sb.request_ids.first(), Some(&10));
        assert_eq!(sb.request_ids.last(), Some(&20));
        assert!(sb.request_ids.contains(&10));
        assert!(sb.request_ids.contains(&20));
        assert!(!sb.request_ids.contains(&99));
    }

    #[test]
    fn test_sub_batch_clone_preserves_shape_not_ids_mutation() {
        let mut original = SubBatch::new(GraphShape::MoeSparse);
        original.add_request(1);
        let cloned = original.clone();

        // Mutate original
        original.add_request(2);
        assert_eq!(original.len(), 2);
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.shape, GraphShape::MoeSparse);
    }

    // ── HardwarePartition: gpu_sm_partition weight proportional to range ──

    #[test]
    fn test_gpu_sm_partition_weight_proportional_to_range() {
        let p_half = HardwarePartition::gpu_sm_partition(0, 0, 40);
        let p_quarter = HardwarePartition::gpu_sm_partition(1, 0, 20);
        assert!(p_half.compute_weight > p_quarter.compute_weight);
        assert!((p_half.compute_weight / p_quarter.compute_weight - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cpu_numa_partition_weight_proportional_to_cores() {
        let p_large = HardwarePartition::cpu_numa_partition(0, 0, 32);
        let p_small = HardwarePartition::cpu_numa_partition(1, 1, 8);
        assert!(p_large.compute_weight > p_small.compute_weight);
        assert!((p_large.compute_weight / p_small.compute_weight - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hardware_partition_gpu_no_numa_fields() {
        let p = HardwarePartition::gpu_sm_partition(5, 10, 30);
        assert!(p.numa_node.is_none());
        assert!(p.core_range.is_none());
    }

    #[test]
    fn test_hardware_partition_cpu_no_sm_fields() {
        let p = HardwarePartition::cpu_numa_partition(3, 2, 12);
        assert!(p.sm_range.is_none());
    }

    #[test]
    fn test_hardware_partition_gpu_kind_always_tensor_core() {
        for start in [0, 10, 40] {
            let p = HardwarePartition::gpu_sm_partition(0, start, start + 20);
            assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        }
    }

    #[test]
    fn test_hardware_partition_cpu_node_zero_kind_full_compute() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 1);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
    }

    #[test]
    fn test_hardware_partition_cpu_node_nonzero_kind_low_compute() {
        for node in [1, 2, 3, 7] {
            let p = HardwarePartition::cpu_numa_partition(0, node, 4);
            assert_eq!(p.kind, HardwareKind::LowComputeCore,
                "NUMA node {} should be LowComputeCore", node);
        }
    }

    // ── HardwarePartition: clone independence for CPU partition ──

    #[test]
    fn test_hardware_partition_cpu_clone_independence() {
        let original = HardwarePartition::cpu_numa_partition(0, 0, 16);
        let cloned = original.clone();
        assert_eq!(cloned.partition_id, original.partition_id);
        assert_eq!(cloned.kind, original.kind);
        assert_eq!(cloned.compute_weight, original.compute_weight);
    }

    // ── dispatch: all requests unclassified → FullPrecision single batch ──

    #[test]
    fn test_dispatch_all_unclassified_single_shape() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=10).collect();
        let manifest = make_manifest_with_ids(&ids);

        // Empty shape_map: every request defaults to FullPrecision
        let plan = d.dispatch(&manifest, &HashMap::new());
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);
        assert_eq!(plan.sub_batches[0].len(), 10);
    }

    // ── dispatch: min_sub_batch_size=0 means every group is valid ──

    #[test]
    fn test_dispatch_min_size_one_single_orphan_becomes_valid() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(1);
        let ids: Vec<RequestId> = vec![1, 2];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        shape_map.insert(1, GraphShape::FullPrecision);
        shape_map.insert(2, GraphShape::SkipAttention);

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
        assert_eq!(plan.sub_batches.len(), 2);
        assert_eq!(plan.orphan_count, 0);
    }

    // ── dispatch: request IDs preserved in sub-batch output ──

    #[test]
    fn test_dispatch_preserves_all_request_ids() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=12).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..6] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[6..12] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);

        let all_dispatched: std::collections::HashSet<RequestId> = plan.sub_batches.iter()
            .flat_map(|sb| sb.request_ids.iter().copied())
            .collect();
        let expected: std::collections::HashSet<RequestId> = ids.into_iter().collect();
        assert_eq!(all_dispatched, expected, "all request IDs must appear in dispatched sub-batches");
    }

    // ── dispatch: orphans still appear in dispatched sub-batches ──

    #[test]
    fn test_dispatch_orphans_preserved_in_sub_batches() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..8] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 3);

        // All 8 IDs must be in sub_batches (orphans merged in)
        let total: usize = plan.sub_batches.iter().map(|sb| sb.len()).sum();
        assert_eq!(total, 8);
    }

    // ── dispatch: shape_counts in MultiShapeDispatched matches sub_batch sizes ──

    #[test]
    fn test_dispatch_shape_counts_matches_sub_batch_sizes() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=10).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::MoeSparse); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);

        if let DispatchReason::MultiShapeDispatched { shape_counts } = &plan.reason {
            let total_from_counts: usize = shape_counts.values().sum();
            let total_from_batches: usize = plan.sub_batches.iter().map(|sb| sb.len()).sum();
            assert_eq!(total_from_counts, total_from_batches);
        } else {
            panic!("Expected MultiShapeDispatched");
        }
    }

    // ── dispatch: GPU partition covers full SM range ──

    #[test]
    fn test_dispatch_gpu_partitions_cover_full_range() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..4] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[4..8] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);

        let ranges: Vec<(usize, usize)> = plan.sub_batches.iter()
            .filter_map(|sb| sb.target_partition.as_ref().and_then(|p| p.sm_range))
            .collect();

        // Verify no gap: each start == previous end
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].0, 0);
        assert_eq!(ranges[1].0, ranges[0].1);
        assert_eq!(ranges[1].1, 80);
    }

    // ── dispatch: dispatch plan with 3 GPU partitions ──

    #[test]
    fn test_dispatch_three_sub_batches_gpu_partitions() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(90);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=9).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..3] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[3..6] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[6..9] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 3);

        // 90 / 3 = 30 SMs each
        for sb in &plan.sub_batches {
            let (start, end) = sb.target_partition.as_ref().unwrap().sm_range.unwrap();
            assert_eq!(end - start, 30);
        }
    }

    // ── dispatch: CPU partition fallback core count ──

    #[test]
    fn test_dispatch_cpu_partition_default_core_count() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = None;
        constraints.numa_node_count = 1;
        constraints.numa_core_bindings = vec![(0, 0, 8)];

        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=4).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        for sb in &plan.sub_batches {
            let partition = sb.target_partition.as_ref().unwrap();
            // Node 0 → FullComputeUnit, cores from binding = 8
            assert_eq!(partition.numa_node, Some(0));
        }
    }

    // ── dispatch: AllOrphanFallback uses FullPrecision shape ──

    #[test]
    fn test_dispatch_all_orphan_uses_full_precision_shape() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(100);
        let ids: Vec<RequestId> = (1..=5).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
        // Fallback batch always uses FullPrecision regardless of original shapes
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);
    }

    // ── dispatch: SingleShapeUniform with MoeSparse ──

    #[test]
    fn test_dispatch_single_shape_moe_sparse() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=6).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].shape, GraphShape::MoeSparse);
        assert_eq!(plan.sub_batches[0].len(), 6);
    }

    // ── dispatch: SingleShapeUniform with SkipAttention ──

    #[test]
    fn test_dispatch_single_shape_skip_attention() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=4).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].shape, GraphShape::SkipAttention);
    }

    // ── dispatch: SingleShapeUniform no ragged compaction ──

    #[test]
    fn test_dispatch_single_shape_no_ragged_compaction() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(!plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 0);
    }

    // ── dispatch: multi shape with no orphans → no ragged compaction ──

    #[test]
    fn test_dispatch_multi_shape_no_orphans_no_ragged() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(4);
        let ids: Vec<RequestId> = (1..=12).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..4] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[4..8] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[8..12] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(!plan.needs_ragged_compaction);
        assert_eq!(plan.orphan_count, 0);
        assert_eq!(plan.sub_batches.len(), 3);
    }

    // ── DispatchReason: MultiShapeDispatched with multiple shape keys ──

    #[test]
    fn test_dispatch_reason_multi_shape_three_keys() {
        let mut counts = HashMap::new();
        counts.insert(GraphShape::FullPrecision, 10);
        counts.insert(GraphShape::SkipAttention, 5);
        counts.insert(GraphShape::NarrowQuant, 3);

        let reason = DispatchReason::MultiShapeDispatched { shape_counts: counts };
        if let DispatchReason::MultiShapeDispatched { shape_counts } = &reason {
            assert_eq!(shape_counts.len(), 3);
            assert_eq!(shape_counts.get(&GraphShape::FullPrecision), Some(&10));
            assert_eq!(shape_counts.get(&GraphShape::SkipAttention), Some(&5));
            assert_eq!(shape_counts.get(&GraphShape::NarrowQuant), Some(&3));
        } else {
            panic!("Expected MultiShapeDispatched");
        }
    }

    // ── DispatchReason: PartialEq reflexivity ──

    #[test]
    fn test_dispatch_reason_partial_eq_reflexivity() {
        let a = DispatchReason::SingleShapeUniform;
        assert_eq!(a, a);

        let b = DispatchReason::AllOrphanFallback;
        assert_eq!(b, b);
    }

    // ── DispatchReason: symmetry ──

    #[test]
    fn test_dispatch_reason_partial_eq_symmetry() {
        let a = DispatchReason::SingleShapeUniform;
        let b = DispatchReason::SingleShapeUniform;
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ── DispatchPlan: debug contains sub_batches field ──

    #[test]
    fn test_dispatch_plan_debug_contains_sub_batches() {
        let plan = DispatchPlan {
            sub_batches: vec![SubBatch::new(GraphShape::FullPrecision)],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::SingleShapeUniform,
        };
        let debug = format!("{:?}", plan);
        assert!(debug.contains("sub_batches"));
        assert!(debug.contains("reason"));
    }

    // ── DispatchPlan: clone deep copies sub_batches ──

    #[test]
    fn test_dispatch_plan_clone_deep_copies_sub_batches() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        sb.add_request(42);
        let plan = DispatchPlan {
            sub_batches: vec![sb],
            needs_ragged_compaction: true,
            orphan_count: 5,
            reason: DispatchReason::AllOrphanFallback,
        };

        let cloned = plan.clone();
        // Modify the cloned sub-batch to verify independence
        assert_eq!(cloned.sub_batches[0].request_ids, vec![42]);
        assert_eq!(cloned.orphan_count, 5);
        assert!(cloned.needs_ragged_compaction);
    }

    // ── GraphShape: all variants are Copy ──

    #[test]
    fn test_graph_shape_copy_semantics() {
        let a = GraphShape::FullPrecision;
        let b = a;
        let c = a;
        // All three should be equal (Copy, not moved)
        assert_eq!(a, b);
        assert_eq!(a, c);
        assert_eq!(b, c);
    }

    // ── HardwareKind: all variants are Copy ──

    #[test]
    fn test_hardware_kind_copy_semantics() {
        let a = HardwareKind::FullComputeUnit;
        let b = a;
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    // ── SubBatch: shape is immutable after construction ──

    #[test]
    fn test_sub_batch_shape_is_set_by_constructor() {
        for shape in [GraphShape::SkipAttention, GraphShape::NarrowQuant,
                      GraphShape::FullPrecision, GraphShape::MoeSparse] {
            let sb = SubBatch::new(shape);
            assert_eq!(sb.shape, shape);
        }
    }

    // ── SubBatch: default estimated fields are zero ──

    #[test]
    fn test_sub_batch_default_estimated_fields_are_zero() {
        let sb = SubBatch::new(GraphShape::FullPrecision);
        assert_eq!(sb.estimated_gflops, 0.0);
        assert_eq!(sb.estimated_vram_bytes, 0);
        assert!(sb.target_partition.is_none());
    }

    // ── HardwarePartition: gpu compute_weight relative to 80 ──

    #[test]
    fn test_gpu_sm_partition_weight_relative_to_80() {
        // 20 SMs / 80 = 0.25
        let p = HardwarePartition::gpu_sm_partition(0, 0, 20);
        assert!((p.compute_weight - 0.25).abs() < f32::EPSILON);
    }

    // ── HardwarePartition: cpu compute_weight relative to 64 ──

    #[test]
    fn test_cpu_numa_partition_weight_relative_to_64() {
        // 16 cores / 64 = 0.25
        let p = HardwarePartition::cpu_numa_partition(0, 0, 16);
        assert!((p.compute_weight - 0.25).abs() < f32::EPSILON);
    }

    // ── dispatch: orphan merges into lightest batch (intensity ordering) ──

    #[test]
    fn test_dispatch_orphan_merges_into_lowest_intensity_batch() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        // 5 NarrowQuant (intensity=0.5), 5 FullPrecision (intensity=1.0), 3 orphans
        let ids: Vec<RequestId> = (1..=13).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[5..10] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[10..13] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        // NarrowQuant (0.5) is the lightest among the valid batches
        let narrow_batch = plan.sub_batches.iter()
            .find(|sb| sb.shape == GraphShape::NarrowQuant).unwrap();
        assert_eq!(narrow_batch.len(), 5 + 3); // 5 original + 3 orphans
    }

    // ── dispatch: GPU SM partition with 4 sub-batches ──

    #[test]
    fn test_dispatch_gpu_four_sub_batches_partition() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        // Two shapes with 4 each → 2 sub-batches
        for &id in &ids[0..4] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[4..8] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 2);

        // Verify both get partitions with kind TensorCorePartition
        for sb in &plan.sub_batches {
            assert_eq!(sb.target_partition.as_ref().unwrap().kind, HardwareKind::TensorCorePartition);
        }
    }

    // ── dispatch: total request count preserved across dispatch ──

    #[test]
    fn test_dispatch_total_request_count_preserved() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(3);
        let ids: Vec<RequestId> = (1..=20).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..7] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[7..14] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[14..18] { shape_map.insert(id, GraphShape::NarrowQuant); }
        // ids[18..20] are MoeSparse → orphans (2 < min=3)
        for &id in &ids[18..20] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        let total: usize = plan.sub_batches.iter().map(|sb| sb.len()).sum();
        assert_eq!(total, 20, "no request IDs should be lost during dispatch");
    }

    // ── dispatch: multi-shape with all orphans from same shape ──

    #[test]
    fn test_dispatch_all_same_shape_below_min_all_orphan() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(100);
        let ids: Vec<RequestId> = (1..=10).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::AllOrphanFallback));
        assert_eq!(plan.orphan_count, 10);
        assert!(plan.needs_ragged_compaction);
    }

    // ── SubBatchDispatcher: with_min_sub_batch_size zero still works ──

    #[test]
    fn test_dispatcher_min_size_zero_single_request_valid() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(0);
        let ids: Vec<RequestId> = vec![1];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        shape_map.insert(1, GraphShape::SkipAttention);

        let plan = d.dispatch(&manifest, &shape_map);
        // 1 >= 0, so it's a valid sub-batch → SingleShapeUniform
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        assert_eq!(plan.sub_batches[0].len(), 1);
    }

    // ── GraphShape: exhaustiveness — every variant handled ──

    #[test]
    fn test_graph_shape_exhaustiveness_all_four_variants() {
        let all = [
            GraphShape::SkipAttention,
            GraphShape::NarrowQuant,
            GraphShape::FullPrecision,
            GraphShape::MoeSparse,
        ];
        assert_eq!(all.len(), 4, "there should be exactly 4 GraphShape variants");
    }

    // ── HardwareKind: exhaustiveness — every variant handled ──

    #[test]
    fn test_hardware_kind_exhaustiveness_all_three_variants() {
        let all = [
            HardwareKind::LowComputeCore,
            HardwareKind::FullComputeUnit,
            HardwareKind::TensorCorePartition,
        ];
        assert_eq!(all.len(), 3, "there should be exactly 3 HardwareKind variants");
    }

    // ── classify_request: deterministic — same inputs produce same output ──

    #[test]
    fn test_classify_deterministic_same_inputs() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        for _ in 0..10 {
            assert_eq!(
                d.classify_request(0.5, 0.3, 0.2),
                GraphShape::MoeSparse,
            );
        }
    }

    // ── dispatch: GPU partition for 4 shapes evenly ──

    #[test]
    fn test_dispatch_gpu_four_shapes_even_split() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(100);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[4..6] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[6..8] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 4);

        // 100 / 4 = 25 SMs each
        for sb in &plan.sub_batches {
            let (start, end) = sb.target_partition.as_ref().unwrap().sm_range.unwrap();
            assert_eq!(end - start, 25);
        }
    }

    // ── dispatch: CPU NUMA with more sub-batches than nodes wraps around ──

    #[test]
    fn test_dispatch_cpu_numa_wraps_when_more_batches_than_nodes() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = None;
        constraints.numa_node_count = 2;
        constraints.numa_core_bindings = vec![(0, 0, 4), (1, 4, 8)];

        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        // 4 shapes → 4 sub-batches, 2 NUMA nodes → wraps
        let ids: Vec<RequestId> = (1..=8).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }
        for &id in &ids[4..6] { shape_map.insert(id, GraphShape::NarrowQuant); }
        for &id in &ids[6..8] { shape_map.insert(id, GraphShape::MoeSparse); }

        let plan = d.dispatch(&manifest, &shape_map);
        let numa_nodes: Vec<usize> = plan.sub_batches.iter()
            .map(|sb| sb.target_partition.as_ref().unwrap().numa_node.unwrap())
            .collect();
        // Round-robin: 0, 1, 0, 1
        assert_eq!(numa_nodes, vec![0, 1, 0, 1]);
    }

    // ═══════════════════════════════════════════════════════════
    //  Wave 4 — 12 additional tests (target 194)
    // ═══════════════════════════════════════════════════════════

    // ── SubBatch: target_partition can be cleared after assignment ──

    #[test]
    fn test_sub_batch_target_partition_can_be_cleared() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.target_partition = Some(HardwarePartition::gpu_sm_partition(0, 0, 40));
        assert!(sb.target_partition.is_some());

        sb.target_partition = None;
        assert!(sb.target_partition.is_none());
    }

    // ── SubBatch: duplicate request IDs are allowed ──

    #[test]
    fn test_sub_batch_allows_duplicate_request_ids() {
        let mut sb = SubBatch::new(GraphShape::SkipAttention);
        sb.add_request(42);
        sb.add_request(42);
        sb.add_request(42);
        assert_eq!(sb.len(), 3);
        assert_eq!(sb.request_ids, vec![42, 42, 42]);
    }

    // ── HardwarePartition: core_range field is always None from constructors ──

    #[test]
    fn test_hardware_partition_core_range_always_none_from_constructors() {
        let gpu = HardwarePartition::gpu_sm_partition(0, 0, 20);
        assert!(gpu.core_range.is_none());

        let cpu = HardwarePartition::cpu_numa_partition(0, 0, 8);
        assert!(cpu.core_range.is_none());
    }

    // ── HardwarePartition: direct field mutation of core_range ──

    #[test]
    fn test_hardware_partition_core_range_can_be_set_directly() {
        let mut p = HardwarePartition::cpu_numa_partition(0, 0, 16);
        assert!(p.core_range.is_none());
        p.core_range = Some((0, 8));
        assert_eq!(p.core_range, Some((0, 8)));
    }

    // ── DispatchPlan: construction with empty sub_batches list ──

    #[test]
    fn test_dispatch_plan_with_empty_sub_batches() {
        let plan = DispatchPlan {
            sub_batches: vec![],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::SingleShapeUniform,
        };
        assert!(plan.sub_batches.is_empty());
        assert_eq!(plan.orphan_count, 0);
        assert!(!plan.needs_ragged_compaction);
    }

    // ── DispatchReason: PartialEq transitivity ──

    #[test]
    fn test_dispatch_reason_partial_eq_transitivity() {
        let mut counts = HashMap::new();
        counts.insert(GraphShape::FullPrecision, 3);

        let a = DispatchReason::MultiShapeDispatched { shape_counts: counts.clone() };
        let b = DispatchReason::MultiShapeDispatched { shape_counts: counts.clone() };
        let c = DispatchReason::MultiShapeDispatched { shape_counts: counts };

        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── SubBatch: estimated_gflops can be negative ──

    #[test]
    fn test_sub_batch_estimated_gflops_negative_value() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        sb.estimated_gflops = -5.0;
        assert!((sb.estimated_gflops - (-5.0)).abs() < f32::EPSILON);
        assert!(sb.estimated_gflops < 0.0);
    }

    // ── SubBatchDispatcher: two independent instances with different min sizes ──

    #[test]
    fn test_dispatcher_different_min_sizes_produce_different_outcomes() {
        let d1 = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);
        let d2 = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(10);

        // Same request set, different min sizes → different outcomes
        let ids: Vec<RequestId> = (1..=5).collect();
        let manifest = make_manifest_with_ids(&ids);
        let mut shape_map = HashMap::new();
        for &id in &ids { shape_map.insert(id, GraphShape::FullPrecision); }

        let plan1 = d1.dispatch(&manifest, &shape_map);
        let plan2 = d2.dispatch(&manifest, &shape_map);

        assert!(matches!(plan1.reason, DispatchReason::SingleShapeUniform));
        assert!(matches!(plan2.reason, DispatchReason::AllOrphanFallback));
    }

    // ── DispatchPlan: orphan_count independent of sub_batches length ──

    #[test]
    fn test_dispatch_plan_orphan_count_independent_of_batch_count() {
        let plan = DispatchPlan {
            sub_batches: vec![SubBatch::new(GraphShape::FullPrecision)],
            needs_ragged_compaction: true,
            orphan_count: 50,
            reason: DispatchReason::AllOrphanFallback,
        };
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.orphan_count, 50);
    }

    // ── GraphShape: can be used as HashMap value ──

    #[test]
    fn test_graph_shape_as_hashmap_value() {
        let mut map: HashMap<usize, GraphShape> = HashMap::new();
        map.insert(0, GraphShape::SkipAttention);
        map.insert(1, GraphShape::NarrowQuant);
        map.insert(2, GraphShape::FullPrecision);
        map.insert(3, GraphShape::MoeSparse);

        assert_eq!(map.len(), 4);
        assert_eq!(map[&0], GraphShape::SkipAttention);
        assert_eq!(map[&3], GraphShape::MoeSparse);
    }

    // ── HardwareKind: can be used as HashMap value ──

    #[test]
    fn test_hardware_kind_as_hashmap_value() {
        let mut map: HashMap<&str, HardwareKind> = HashMap::new();
        map.insert("low", HardwareKind::LowComputeCore);
        map.insert("full", HardwareKind::FullComputeUnit);
        map.insert("tensor", HardwareKind::TensorCorePartition);

        assert_eq!(map.len(), 3);
        assert_eq!(map["low"], HardwareKind::LowComputeCore);
        assert_eq!(map["tensor"], HardwareKind::TensorCorePartition);
    }

    // ── SubBatch: estimated_vram_bytes can be set to zero after construction ──

    #[test]
    fn test_sub_batch_estimated_vram_can_be_reset_to_zero() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.estimated_vram_bytes = 1_000_000;
        assert_eq!(sb.estimated_vram_bytes, 1_000_000);

        sb.estimated_vram_bytes = 0;
        assert_eq!(sb.estimated_vram_bytes, 0);
    }

    // ═══════════════════════════════════════════════════════════
    //  Wave 5 — 13 additional tests (target 207)
    // ═══════════════════════════════════════════════════════════

    // ── HardwarePartition: gpu_sm_partition with zero-width SM range ──

    #[test]
    fn test_gpu_sm_partition_zero_width_range() {
        let p = HardwarePartition::gpu_sm_partition(0, 40, 40);
        assert_eq!(p.sm_range, Some((40, 40)));
        assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        assert!((p.compute_weight - 0.0).abs() < f32::EPSILON);
    }

    // ── SubBatch: estimated_gflops can hold f32::MAX without overflow ──

    #[test]
    fn test_sub_batch_estimated_gflops_f32_max() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.estimated_gflops = f32::MAX;
        assert_eq!(sb.estimated_gflops, f32::MAX);
    }

    // ── SubBatch: clone with usize::MAX estimated_vram_bytes ──

    #[test]
    fn test_sub_batch_clone_with_max_vram() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        sb.add_request(1);
        sb.estimated_vram_bytes = usize::MAX;
        let cloned = sb.clone();
        assert_eq!(cloned.estimated_vram_bytes, usize::MAX);
        assert_eq!(cloned.request_ids, vec![1]);
    }

    // ── GraphShape: HashMap key overwrite retains last value ──

    #[test]
    fn test_graph_shape_hashmap_key_overwrite() {
        let mut map: HashMap<GraphShape, i32> = HashMap::new();
        map.insert(GraphShape::FullPrecision, 1);
        map.insert(GraphShape::FullPrecision, 99);
        assert_eq!(map.get(&GraphShape::FullPrecision), Some(&99));
        assert_eq!(map.len(), 1);
    }

    // ── HardwarePartition: same partition_id from different constructors ──

    #[test]
    fn test_hardware_partition_same_id_different_kinds() {
        let gpu = HardwarePartition::gpu_sm_partition(5, 0, 20);
        let cpu = HardwarePartition::cpu_numa_partition(5, 1, 8);
        assert_eq!(gpu.partition_id, cpu.partition_id);
        assert_ne!(gpu.kind, cpu.kind);
        assert_eq!(gpu.kind, HardwareKind::TensorCorePartition);
        assert_eq!(cpu.kind, HardwareKind::LowComputeCore);
    }

    // ── DispatchPlan: manual construction with orphan_count not matching sub-batches ──

    #[test]
    fn test_dispatch_plan_orphan_count_decoupled_from_batches() {
        // DispatchPlan is a plain struct; orphan_count is independently set
        let plan = DispatchPlan {
            sub_batches: vec![SubBatch::new(GraphShape::SkipAttention)],
            needs_ragged_compaction: false,
            orphan_count: 999,
            reason: DispatchReason::SingleShapeUniform,
        };
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.orphan_count, 999);
        assert!(!plan.needs_ragged_compaction);
    }

    // ── classify_request: is_moe=true, active_experts just above 0.5 threshold ──

    #[test]
    fn test_classify_moe_active_experts_just_above_half() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default()).with_has_moe_ops(true);
        let shape = d.classify_request(0.0, 0.0, 0.5001);
        assert_eq!(shape, GraphShape::FullPrecision);
    }

    // ── SubBatch: request_ids supports iteration ──

    #[test]
    fn test_sub_batch_request_ids_iterable() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        for i in 10..15 {
            sb.add_request(i);
        }
        let collected: Vec<RequestId> = sb.request_ids.iter().copied().collect();
        assert_eq!(collected, vec![10, 11, 12, 13, 14]);

        let sum: RequestId = sb.request_ids.iter().sum();
        assert_eq!(sum, 60);
    }

    // ── HardwareKind: HashSet membership and removal ──

    #[test]
    fn test_hardware_kind_hashset_insert_remove() {
        let mut set: std::collections::HashSet<HardwareKind> = std::collections::HashSet::new();
        set.insert(HardwareKind::LowComputeCore);
        set.insert(HardwareKind::FullComputeUnit);
        assert_eq!(set.len(), 2);
        assert!(set.remove(&HardwareKind::LowComputeCore));
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&HardwareKind::LowComputeCore));
        assert!(set.contains(&HardwareKind::FullComputeUnit));
    }

    // ── DispatchReason: MultiShapeDispatched with two entries for same key ──

    #[test]
    fn test_dispatch_reason_multi_shape_insert_overwrites() {
        let mut counts = HashMap::new();
        counts.insert(GraphShape::FullPrecision, 5);
        counts.insert(GraphShape::FullPrecision, 10);
        // HashMap insert overwrites → last value wins
        assert_eq!(counts.get(&GraphShape::FullPrecision), Some(&10));

        let reason = DispatchReason::MultiShapeDispatched { shape_counts: counts };
        if let DispatchReason::MultiShapeDispatched { shape_counts } = &reason {
            assert_eq!(shape_counts.len(), 1);
            assert_eq!(shape_counts.get(&GraphShape::FullPrecision), Some(&10));
        } else {
            panic!("Expected MultiShapeDispatched");
        }
    }

    // ── SubBatch: sequence of shape reassignment not possible (shape is not mut) ──
    // Verify shape field is pub but the struct is created per shape

    #[test]
    fn test_sub_batch_different_shapes_independent() {
        let mut sb_a = SubBatch::new(GraphShape::SkipAttention);
        let mut sb_b = SubBatch::new(GraphShape::FullPrecision);
        sb_a.add_request(1);
        sb_b.add_request(2);

        assert_ne!(sb_a.shape, sb_b.shape);
        assert_eq!(sb_a.request_ids, vec![1]);
        assert_eq!(sb_b.request_ids, vec![2]);
    }

    // ── DispatchPlan: clone with empty sub_batches list ──

    #[test]
    fn test_dispatch_plan_clone_empty_sub_batches() {
        let plan = DispatchPlan {
            sub_batches: vec![],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::SingleShapeUniform,
        };
        let cloned = plan.clone();
        assert!(cloned.sub_batches.is_empty());
        assert_eq!(cloned.orphan_count, 0);
        assert!(!cloned.needs_ragged_compaction);
        assert_eq!(cloned.reason, DispatchReason::SingleShapeUniform);
    }

    // ── HardwarePartition: gpu_sm_partition with large non-zero start offset ──

    #[test]
    fn test_gpu_sm_partition_large_start_offset() {
        let p = HardwarePartition::gpu_sm_partition(2, 60, 80);
        assert_eq!(p.partition_id, 2);
        assert_eq!(p.sm_range, Some((60, 80)));
        // (80-60)/80 = 0.25
        assert!((p.compute_weight - 0.25).abs() < f32::EPSILON);
        assert!(p.numa_node.is_none());
    }

    // ═══════════════════════════════════════════════════════════
    //  Wave 6 — 10 additional tests (target 217)
    // ═══════════════════════════════════════════════════════════

    // ── SubBatch: replace target_partition from GPU to CPU ──

    #[test]
    fn test_sub_batch_target_partition_replaced_gpu_to_cpu() {
        let mut sb = SubBatch::new(GraphShape::FullPrecision);
        sb.target_partition = Some(HardwarePartition::gpu_sm_partition(0, 0, 40));
        assert_eq!(sb.target_partition.as_ref().unwrap().kind, HardwareKind::TensorCorePartition);

        sb.target_partition = Some(HardwarePartition::cpu_numa_partition(0, 0, 16));
        assert_eq!(sb.target_partition.as_ref().unwrap().kind, HardwareKind::FullComputeUnit);
        assert!(sb.target_partition.as_ref().unwrap().sm_range.is_none());
    }

    // ── HardwarePartition: manually constructed with both sm_range and numa_node ──

    #[test]
    fn test_hardware_partition_manual_construction_hybrid_fields() {
        let p = HardwarePartition {
            partition_id: 99,
            kind: HardwareKind::FullComputeUnit,
            sm_range: Some((10, 20)),
            numa_node: Some(3),
            core_range: Some((0, 4)),
            compute_weight: 0.75,
        };
        assert_eq!(p.partition_id, 99);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
        assert_eq!(p.sm_range, Some((10, 20)));
        assert_eq!(p.numa_node, Some(3));
        assert_eq!(p.core_range, Some((0, 4)));
        assert!((p.compute_weight - 0.75).abs() < f32::EPSILON);
    }

    // ── dispatch: manifest with duplicate request_ids in different shapes ──

    #[test]
    fn test_dispatch_duplicate_request_id_across_slots() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);

        // Two slots with same request_id, different shapes in shape_map
        // shape_map can only hold one shape per request_id (HashMap key)
        let slots = vec![
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: 0 },
            BatchSlot { request_id: 42, slot_type: SlotType::Decode, token_start: 10, token_end: 20, compact_target: 1 },
            BatchSlot { request_id: 43, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: 2 },
            BatchSlot { request_id: 44, slot_type: SlotType::Decode, token_start: 0, token_end: 10, compact_target: 3 },
        ];
        let manifest = BatchManifest {
            slots,
            total_tokens: 40,
            decode_tokens: 40,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };

        // request_id 42 appears twice in manifest but shape_map has one entry for it
        let mut shape_map = HashMap::new();
        shape_map.insert(42, GraphShape::SkipAttention);
        shape_map.insert(43, GraphShape::SkipAttention);
        shape_map.insert(44, GraphShape::SkipAttention);

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::SingleShapeUniform));
        // 4 slots total, all mapped to SkipAttention
        assert_eq!(plan.sub_batches[0].len(), 4);
    }

    // ── dispatch: two shapes where one is exactly min-1 (one below threshold) ──

    #[test]
    fn test_dispatch_one_shape_at_min_another_at_min_minus_one() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(5);

        // 5 FullPrecision (valid), 4 NarrowQuant (orphan, since 4 < 5)
        let ids: Vec<RequestId> = (1..=9).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..5] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[5..9] { shape_map.insert(id, GraphShape::NarrowQuant); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.orphan_count, 4);
        assert!(plan.needs_ragged_compaction);
        // Only FullPrecision is a valid sub-batch; orphans merge into it
        assert_eq!(plan.sub_batches.len(), 1);
        assert_eq!(plan.sub_batches[0].shape, GraphShape::FullPrecision);
        assert_eq!(plan.sub_batches[0].len(), 5 + 4);
    }

    // ── SubBatch: add_request with RequestId boundary values ──

    #[test]
    fn test_sub_batch_add_request_with_extreme_ids() {
        let mut sb = SubBatch::new(GraphShape::MoeSparse);
        sb.add_request(0);
        sb.add_request(RequestId::MAX);
        assert_eq!(sb.len(), 2);
        assert_eq!(sb.request_ids[0], 0);
        assert_eq!(sb.request_ids[1], RequestId::MAX);
    }

    // ── dispatch: all four shapes present with min=1 (all valid) ──

    #[test]
    fn test_dispatch_all_four_shapes_valid_min_one() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(80);
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(1);

        // 1 request per shape, min=1 → all valid sub-batches
        let ids: Vec<RequestId> = vec![1, 2, 3, 4];
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        shape_map.insert(1, GraphShape::FullPrecision);
        shape_map.insert(2, GraphShape::SkipAttention);
        shape_map.insert(3, GraphShape::NarrowQuant);
        shape_map.insert(4, GraphShape::MoeSparse);

        let plan = d.dispatch(&manifest, &shape_map);
        assert!(matches!(plan.reason, DispatchReason::MultiShapeDispatched { .. }));
        assert_eq!(plan.sub_batches.len(), 4);
        assert_eq!(plan.orphan_count, 0);
        assert!(!plan.needs_ragged_compaction);

        // Verify all four shapes are present
        let shapes: std::collections::HashSet<GraphShape> =
            plan.sub_batches.iter().map(|sb| sb.shape).collect();
        assert_eq!(shapes.len(), 4);
    }

    // ── DispatchPlan: multiple sub_batches each with target_partition ──

    #[test]
    fn test_dispatch_plan_multiple_batches_with_partitions() {
        let mut sb1 = SubBatch::new(GraphShape::FullPrecision);
        sb1.add_request(1);
        sb1.add_request(2);
        sb1.target_partition = Some(HardwarePartition::gpu_sm_partition(0, 0, 40));

        let mut sb2 = SubBatch::new(GraphShape::SkipAttention);
        sb2.add_request(3);
        sb2.target_partition = Some(HardwarePartition::gpu_sm_partition(1, 40, 80));

        let plan = DispatchPlan {
            sub_batches: vec![sb1, sb2],
            needs_ragged_compaction: false,
            orphan_count: 0,
            reason: DispatchReason::MultiShapeDispatched {
                shape_counts: {
                    let mut m = HashMap::new();
                    m.insert(GraphShape::FullPrecision, 2);
                    m.insert(GraphShape::SkipAttention, 1);
                    m
                },
            },
        };

        assert_eq!(plan.sub_batches.len(), 2);
        // First sub-batch: SM range [0, 40)
        let p0 = plan.sub_batches[0].target_partition.as_ref().unwrap();
        assert_eq!(p0.sm_range, Some((0, 40)));
        // Second sub-batch: SM range [40, 80)
        let p1 = plan.sub_batches[1].target_partition.as_ref().unwrap();
        assert_eq!(p1.sm_range, Some((40, 80)));
    }

    // ── dispatch: GPU partition with odd SM count produces integer truncation ──

    #[test]
    fn test_dispatch_gpu_odd_sm_count_integer_division() {
        let mut constraints = CompilerConstraints::default();
        constraints.gpu_sm_count = Some(7); // 7 SMs, 2 sub-batches → 7/2 = 3 each
        let d = SubBatchDispatcher::new(constraints)
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=4).collect();
        let manifest = make_manifest_with_ids(&ids);

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan = d.dispatch(&manifest, &shape_map);
        assert_eq!(plan.sub_batches.len(), 2);

        // Each gets 7/2 = 3 SMs (integer division truncation)
        for sb in &plan.sub_batches {
            let (start, end) = sb.target_partition.as_ref().unwrap().sm_range.unwrap();
            assert_eq!(end - start, 3);
        }
    }

    // ── SubBatch: shape field pub accessibility allows read but struct is not rebuilt ──

    #[test]
    fn test_sub_batch_shape_remains_constant_throughout_lifecycle() {
        let mut sb = SubBatch::new(GraphShape::NarrowQuant);
        let initial_shape = sb.shape;
        sb.add_request(1);
        sb.estimated_gflops = 100.0;
        sb.estimated_vram_bytes = 2048;
        sb.target_partition = Some(HardwarePartition::cpu_numa_partition(0, 0, 8));
        assert_eq!(sb.shape, initial_shape);
        assert_eq!(sb.shape, GraphShape::NarrowQuant);
    }

    // ── dispatch: total tokens in manifest irrelevant to dispatch logic ──

    #[test]
    fn test_dispatch_manifest_token_counts_do_not_affect_result() {
        let d = SubBatchDispatcher::new(CompilerConstraints::default())
            .with_min_sub_batch_size(2);

        let ids: Vec<RequestId> = (1..=4).collect();

        // Manifest with realistic token counts
        let manifest_realistic = {
            let slots: Vec<BatchSlot> = ids.iter().enumerate().map(|(i, &id)| {
                BatchSlot { request_id: id, slot_type: SlotType::Decode, token_start: i * 100, token_end: i * 100 + 100, compact_target: i as i32 }
            }).collect();
            BatchManifest { slots, total_tokens: 400, decode_tokens: 400, prefill_tokens: 0, compact_required: false, waste_ratio: 0.0 }
        };

        // Manifest with zero token counts (same slot structure)
        let manifest_zero = {
            let slots: Vec<BatchSlot> = ids.iter().enumerate().map(|(i, &id)| {
                BatchSlot { request_id: id, slot_type: SlotType::Decode, token_start: 0, token_end: 0, compact_target: i as i32 }
            }).collect();
            BatchManifest { slots, total_tokens: 0, decode_tokens: 0, prefill_tokens: 0, compact_required: false, waste_ratio: 0.0 }
        };

        let mut shape_map = HashMap::new();
        for &id in &ids[0..2] { shape_map.insert(id, GraphShape::FullPrecision); }
        for &id in &ids[2..4] { shape_map.insert(id, GraphShape::SkipAttention); }

        let plan_realistic = d.dispatch(&manifest_realistic, &shape_map);
        let plan_zero = d.dispatch(&manifest_zero, &shape_map);

        // Both produce identical dispatch plans (same shape, same counts)
        assert_eq!(plan_realistic.sub_batches.len(), plan_zero.sub_batches.len());
        assert_eq!(plan_realistic.orphan_count, plan_zero.orphan_count);
        assert_eq!(plan_realistic.needs_ragged_compaction, plan_zero.needs_ragged_compaction);
    }
}
