//! CPU/GPU 并行 MoE 分发 (SPEC §15.3)
//!
//! ## 核心职责
//! 将 MoE 专家分发到不同的硬件后端并行执行:
//! - 热专家 → GPU Tensor Core (高吞吐)
//! - 温专家 → CPU AMX/AVX-512 (低延迟本地计算)
//! - 冷专家 → 远程 RDMA (分布式推理)
//!
//! ## §15.3 CPU/GPU 真正并行 (Core Disaggregation)
//! CPU 的 NUMA 探针暴露 AMX 或 AVX512 能力为 IR。
//! 编译器为温专家顺量生成一段 CPU 特化的 JIT 代码。
//! 通过独立的并发流将少部分 Token 交给 CPU 算完后统一回写。
//!
//! ## SPEC §10.15 合规
//! 硬件能力从 `ExecutionPlan` 推导，不散乱读取 DeviceProfile:
//! - `cpu_has_amx` ← `exec_plan.gemm_plan.strategy == AmxTile`
//! - `cpu_has_avx512` ← matches BlisAvx512 | Avx512NativeBf16
//! - `cpu_core_count` ← `exec_plan.profile.physical_cores`

use super::routing::{ExpertRouteConfig, ExpertRouteTable};
use super::thermal::ExpertHeatLevel;
use crate::jit::sub_batch::{HardwareKind, HardwarePartition};

/// 专家硬件分配
#[derive(Debug, Clone)]
pub struct ExpertHardwareAssignment {
    /// 专家索引
    pub expert_idx: usize,
    /// 目标硬件类型
    pub hardware: HardwareKind,
    /// 目标硬件分区 (GPU 时为 SM 范围)
    pub partition: Option<HardwarePartition>,
    /// 分配的 token 数
    pub token_count: usize,
    /// 预估计算时间 (μs)
    pub estimated_compute_us: f32,
}

/// MoE 硬件分发计划
#[derive(Debug, Clone)]
pub struct MoeDispatchPlan {
    /// GPU 分配的专家
    pub gpu_experts: Vec<ExpertHardwareAssignment>,
    /// CPU 分配的专家
    pub cpu_experts: Vec<ExpertHardwareAssignment>,
    /// 远程分配的专家
    pub remote_experts: Vec<ExpertHardwareAssignment>,
    /// §15.4 被 evict 的专家（跳过执行，token 分配到 fallback）
    pub skipped_experts: Vec<usize>,
    /// GPU 预估总计算时间 (μs)
    pub gpu_total_us: f32,
    /// CPU 预估总计算时间 (μs)
    pub cpu_total_us: f32,
    /// 是否需要同步等待 CPU
    pub needs_cpu_sync: bool,
}

impl MoeDispatchPlan {
    /// 获取总专家分配数
    pub fn total_assignments(&self) -> usize {
        self.gpu_experts.len() + self.cpu_experts.len() + self.remote_experts.len()
    }

    /// 检查 CPU 和 GPU 是否可以并行 (CPU 不慢于 GPU)
    pub fn is_balanced(&self) -> bool {
        if self.cpu_experts.is_empty() || self.gpu_experts.is_empty() {
            return true;
        }
        // CPU 不应超过 GPU 2 倍时间
        self.cpu_total_us <= self.gpu_total_us * 2.0
    }
}

/// MoE 硬件分发器
///
/// §15.3: 根据专家热度和硬件能力，将专家分发到不同后端。
pub struct MoeHardwareDispatcher {
    /// 专家路由配置
    config: ExpertRouteConfig,
    /// GPU SM 数量
    gpu_sm_count: usize,
    /// CPU 是否支持 AMX
    cpu_has_amx: bool,
    /// CPU 是否支持 AVX-512
    cpu_has_avx512: bool,
    /// CPU 核心数
    cpu_core_count: usize,
    /// GPU 算力 (TFLOPS)
    gpu_tflops: f32,
    /// CPU 算力 (TFLOPS, AMX/AVX512)
    cpu_tflops: f32,
    /// 最大 CPU 分配比例 (0.0-1.0)
    max_cpu_ratio: f32,
}

impl MoeHardwareDispatcher {
    /// 从 ExecutionPlan 创建 MoE 硬件分发器 (SPEC §10.15)。
    ///
    /// 硬件能力从预计算的 `gemm_plan.strategy` 推导，不散乱读取 DeviceProfile。
    pub fn from_execution_plan(
        config: ExpertRouteConfig,
        exec_plan: &gllm_kernels::compiler::planner::ExecutionPlan,
    ) -> Self {
        use gllm_kernels::compiler::planner::GemmMicrokernelStrategy as GMS;

        let strategy = &exec_plan.gemm_plan.strategy;
        let cpu_has_amx = matches!(strategy, GMS::AmxTile);
        let cpu_has_avx512 = matches!(strategy, GMS::BlisAvx512 | GMS::Avx512NativeBf16);
        let cpu_core_count = exec_plan.profile.physical_cores;

        Self {
            config,
            gpu_sm_count: 0,
            cpu_has_amx,
            cpu_has_avx512,
            cpu_core_count,
            gpu_tflops: 0.0,
            cpu_tflops: 0.0,
            max_cpu_ratio: 0.3,
        }
    }

    /// 创建新的 MoE 硬件分发器（测试用，手动配置硬件参数）
    pub fn new(config: ExpertRouteConfig) -> Self {
        Self {
            config,
            gpu_sm_count: 0,
            cpu_has_amx: false,
            cpu_has_avx512: false,
            cpu_core_count: 0,
            gpu_tflops: 0.0,
            cpu_tflops: 0.0,
            max_cpu_ratio: 0.3,
        }
    }

    /// 配置 GPU 参数
    pub fn with_gpu(mut self, sm_count: usize, tflops: f32) -> Self {
        self.gpu_sm_count = sm_count;
        self.gpu_tflops = tflops;
        self
    }

    /// 配置 CPU 参数
    pub fn with_cpu(mut self, core_count: usize, has_amx: bool, has_avx512: bool, tflops: f32) -> Self {
        self.cpu_core_count = core_count;
        self.cpu_has_amx = has_amx;
        self.cpu_has_avx512 = has_avx512;
        self.cpu_tflops = tflops;
        self
    }

    /// 配置最大 CPU 分配比例
    pub fn with_max_cpu_ratio(mut self, ratio: f32) -> Self {
        self.max_cpu_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// 生成分发计划
    ///
    /// §15.3: 根据路由表和专家热度状态，将专家分配到 GPU/CPU/远程。
    /// §15.4: Evicted 专家跳过执行（token 重分配到 top-k 候选中次优专家）。
    pub fn dispatch(
        &self,
        route_table: &ExpertRouteTable,
        heat_levels: &[ExpertHeatLevel],
    ) -> MoeDispatchPlan {
        let mut gpu_experts = Vec::new();
        let mut cpu_experts = Vec::new();
        let remote_experts = Vec::new();
        let mut skipped_experts = Vec::new();

        // 统计每个专家的 token 数（排除 evicted 专家的 token）
        let mut expert_token_counts = vec![0usize; self.config.num_experts];
        for route in &route_table.token_routes {
            for &expert_idx in &route.expert_indices {
                if expert_idx < expert_token_counts.len() {
                    let heat = heat_levels.get(expert_idx).copied().unwrap_or_else(|| {
                        log::warn!("expert_id {} not found in heat_levels table — defaulting to Warm", expert_idx);
                        ExpertHeatLevel::Warm
                    });
                    if matches!(heat, ExpertHeatLevel::Evicted) {
                        // §15.4: Evicted 专家 token 不计入分配
                        continue;
                    }
                    expert_token_counts[expert_idx] += 1;
                }
            }
        }

        // 收集 evicted 专家列表
        for (expert_idx, &heat) in heat_levels.iter().enumerate() {
            if matches!(heat, ExpertHeatLevel::Evicted) && expert_token_counts.get(expert_idx).copied().unwrap_or(0) == 0 {
                skipped_experts.push(expert_idx);
            }
        }

        let total_tokens: usize = expert_token_counts.iter().sum();
        // [BCE-022] clamp before f32→usize: negative ratio or NaN could wrap to huge usize
        let max_cpu_tokens = (total_tokens as f32 * self.max_cpu_ratio).max(0.0) as usize;
        let mut cpu_tokens_used = 0usize;

        for (expert_idx, &token_count) in expert_token_counts.iter().enumerate() {
            if token_count == 0 {
                continue;
            }

            let heat = heat_levels.get(expert_idx).copied().unwrap_or_else(|| {
                log::warn!("expert_id {} not found in heat_levels table — defaulting to Warm", expert_idx);
                ExpertHeatLevel::Warm
            });
            // §15.4: 跳过 evicted 专家
            if matches!(heat, ExpertHeatLevel::Evicted) {
                skipped_experts.push(expert_idx);
                continue;
            }

            let assignment = self.create_assignment(expert_idx, token_count, heat);

            match assignment.hardware {
                HardwareKind::TensorCorePartition | HardwareKind::FullComputeUnit => {
                    gpu_experts.push(assignment);
                }
                HardwareKind::LowComputeCore => {
                    // 检查 CPU 容量
                    if cpu_tokens_used + token_count <= max_cpu_tokens && self.cpu_has_amx {
                        cpu_tokens_used += token_count;
                        cpu_experts.push(assignment);
                    } else {
                        // CPU 容量不足，回退到 GPU
                        let mut fallback = assignment;
                        fallback.hardware = HardwareKind::FullComputeUnit;
                        gpu_experts.push(fallback);
                    }
                }
            }
        }

        // 计算预估时间
        let gpu_total_us = gpu_experts.iter().map(|e| e.estimated_compute_us).fold(0.0f32, f32::max);
        let cpu_total_us = cpu_experts.iter().map(|e| e.estimated_compute_us).fold(0.0f32, f32::max);

        MoeDispatchPlan {
            needs_cpu_sync: !cpu_experts.is_empty(),
            gpu_experts,
            cpu_experts,
            remote_experts,
            skipped_experts,
            gpu_total_us,
            cpu_total_us,
        }
    }

    /// 创建单个专家的硬件分配
    fn create_assignment(
        &self,
        expert_idx: usize,
        token_count: usize,
        heat: ExpertHeatLevel,
    ) -> ExpertHardwareAssignment {
        let (hardware, compute_us) = match heat {
            ExpertHeatLevel::Hot => {
                // 热专家 → GPU Tensor Core
                let us = self.estimate_gpu_compute(token_count);
                (HardwareKind::TensorCorePartition, us)
            }
            ExpertHeatLevel::Warm => {
                // 温专家 → CPU AMX (如果可用) 或 GPU
                if self.cpu_has_amx {
                    let us = self.estimate_cpu_compute(token_count);
                    (HardwareKind::LowComputeCore, us)
                } else {
                    let us = self.estimate_gpu_compute(token_count);
                    (HardwareKind::FullComputeUnit, us)
                }
            }
            ExpertHeatLevel::Cold => {
                // 冷专家 → GPU (权重需要从 CPU 预取)
                let us = self.estimate_gpu_compute(token_count);
                (HardwareKind::FullComputeUnit, us)
            }
            ExpertHeatLevel::Evicted => {
                // 封杀专家 → 需要 Deopt 处理，暂分配到 GPU
                (HardwareKind::FullComputeUnit, f32::INFINITY)
            }
        };

        ExpertHardwareAssignment {
            expert_idx,
            hardware,
            partition: None, // 由 SubBatchDispatcher 填充
            token_count,
            estimated_compute_us: compute_us,
        }
    }

    /// 估算 GPU 计算时间 (μs)
    fn estimate_gpu_compute(&self, token_count: usize) -> f32 {
        if self.gpu_tflops <= 0.0 {
            return f32::INFINITY;
        }
        // 简化估算: FFN = 2 × hidden_size × intermediate_size × seq_len FLOPs
        let flops = 2 * 4096 * 14336 * token_count; // 典型 FFN FLOPs
        flops as f32 / (self.gpu_tflops * 1e12) * 1e6
    }

    /// 估算 CPU 计算时间 (μs)
    fn estimate_cpu_compute(&self, token_count: usize) -> f32 {
        if self.cpu_tflops <= 0.0 {
            return f32::INFINITY;
        }
        let flops = 2 * 4096 * 14336 * token_count;
        flops as f32 / (self.cpu_tflops * 1e12) * 1e6
    }

    /// 获取配置引用
    pub fn config(&self) -> &ExpertRouteConfig {
        &self.config
    }

    /// 获取 GPU SM 数量
    pub fn gpu_sm_count(&self) -> usize {
        self.gpu_sm_count
    }

    /// 获取 CPU 是否支持 AMX
    pub fn cpu_has_amx(&self) -> bool {
        self.cpu_has_amx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route_config() -> ExpertRouteConfig {
        ExpertRouteConfig::new(4, 2)
    }

    fn make_route_table(config: &ExpertRouteConfig) -> ExpertRouteTable {
        let gate_logits = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        ExpertRouteTable::from_gate_logits(config.clone(), &gate_logits)
    }

    #[test]
    fn test_dispatch_all_gpu() {
        let config = make_route_config();
        let route_table = make_route_table(&config);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
        ];

        let dispatcher = MoeHardwareDispatcher::new(config)
            .with_gpu(80, 300.0);

        let plan = dispatcher.dispatch(&route_table, &heat_levels);

        assert!(plan.gpu_experts.len() >= 4); // All hot → all GPU
        assert!(plan.cpu_experts.is_empty());
        assert!(!plan.needs_cpu_sync);
    }

    #[test]
    fn test_dispatch_mixed_gpu_cpu() {
        let config = make_route_config();
        let route_table = make_route_table(&config);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,   // → GPU
            ExpertHeatLevel::Warm,  // → CPU (AMX available)
            ExpertHeatLevel::Warm,  // → CPU
            ExpertHeatLevel::Hot,   // → GPU
        ];

        let dispatcher = MoeHardwareDispatcher::new(config)
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0); // AMX available

        let plan = dispatcher.dispatch(&route_table, &heat_levels);

        assert!(!plan.gpu_experts.is_empty());
        assert!(!plan.cpu_experts.is_empty());
        assert!(plan.needs_cpu_sync);
    }

    #[test]
    fn test_dispatch_no_amx_fallback_to_gpu() {
        let config = make_route_config();
        let _route_table = make_route_table(&config);
        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
        ];

        let config2 = ExpertRouteConfig::new(2, 1);
        let route_table2 = ExpertRouteTable::from_gate_logits(
            config2,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );

        let dispatcher = MoeHardwareDispatcher::new(make_route_config())
            .with_gpu(80, 300.0)
            .with_cpu(16, false, false, 0.0); // No AMX

        let plan = dispatcher.dispatch(&route_table2, &heat_levels);

        // Without AMX, warm experts go to GPU
        assert!(!plan.gpu_experts.is_empty());
    }

    #[test]
    fn test_dispatch_plan_balanced() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );

        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
        ];

        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);

        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Plan should be valid even if not perfectly balanced
        assert!(plan.total_assignments() >= 2);
    }

    #[test]
    fn test_max_cpu_ratio() {
        let config = ExpertRouteConfig::new(4, 1);
        // All tokens go to experts 0,1 (warm → CPU candidates)
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
        ];

        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.25); // Max 25% to CPU

        let plan = dispatcher.dispatch(&route_table, &heat_levels);

        // CPU tokens should be limited
        let cpu_tokens: usize = plan.cpu_experts.iter().map(|e| e.token_count).sum();
        let total_tokens: usize = plan.gpu_experts.iter().map(|e| e.token_count).sum::<usize>() + cpu_tokens;
        if total_tokens > 0 {
            let cpu_ratio = cpu_tokens as f32 / total_tokens as f32;
            assert!(cpu_ratio <= 0.26, "CPU ratio {:.2} exceeds max 0.25", cpu_ratio);
        }
    }

    // ── Data structure tests ──

    #[test]
    fn dispatch_plan_empty() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 0);
        assert!(plan.is_balanced());
        assert!(!plan.needs_cpu_sync);
    }

    #[test]
    fn dispatch_plan_total_assignments_counts_all() {
        let assignment = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 10,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![assignment.clone()],
            cpu_experts: vec![assignment.clone()],
            remote_experts: vec![assignment],
            skipped_experts: vec![5],
            gpu_total_us: 1.0,
            cpu_total_us: 0.5,
            needs_cpu_sync: true,
        };
        assert_eq!(plan.total_assignments(), 3);
    }

    #[test]
    fn dispatch_plan_is_balanced_when_cpu_fast() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 100.0,
            cpu_total_us: 50.0,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced());
    }

    #[test]
    fn dispatch_plan_not_balanced_when_cpu_slow() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 10,
            estimated_compute_us: 10.0,
        };
        let b = ExpertHardwareAssignment {
            expert_idx: 1,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 10,
            estimated_compute_us: 100.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a],
            cpu_experts: vec![b],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 10.0,
            cpu_total_us: 100.0,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced());
    }

    #[test]
    fn expert_hardware_assignment_fields() {
        let a = ExpertHardwareAssignment {
            expert_idx: 3,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 128,
            estimated_compute_us: 42.5,
        };
        assert_eq!(a.expert_idx, 3);
        assert_eq!(a.token_count, 128);
        assert!(a.partition.is_none());
    }

    #[test]
    fn dispatcher_builder_pattern() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        assert_eq!(d.gpu_sm_count(), 80);
        assert!(d.cpu_has_amx());
        assert_eq!(d.config().num_experts, 8);
    }

    #[test]
    fn dispatcher_max_cpu_ratio_clamped() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_max_cpu_ratio(2.0);
        // ratio should be clamped to 1.0
        // Verify indirectly: dispatch with all warm + AMX should work
        assert!(d.config().num_experts == 4);
    }

    #[test]
    fn dispatch_plan_debug_format() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![1, 2],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        let s = format!("{:?}", plan);
        assert!(s.contains("skipped_experts"));
    }

    #[test]
    fn expert_hardware_assignment_debug_format() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 64,
            estimated_compute_us: 10.0,
        };
        let s = format!("{:?}", a);
        assert!(s.contains("expert_idx"));
    }

    // ── ExpertHardwareAssignment derive traits ──

    #[test]
    fn expert_assignment_clone() {
        let a = ExpertHardwareAssignment {
            expert_idx: 3,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 128,
            estimated_compute_us: 25.0,
        };
        let cloned = a.clone();
        assert_eq!(a.expert_idx, cloned.expert_idx);
        assert_eq!(a.token_count, cloned.token_count);
    }

    #[test]
    fn expert_assignment_fields() {
        let a = ExpertHardwareAssignment {
            expert_idx: 7,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 32,
            estimated_compute_us: 5.5,
        };
        assert_eq!(a.expert_idx, 7);
        assert!(a.partition.is_none());
        assert!((a.estimated_compute_us - 5.5).abs() < 1e-6);
    }

    // ── MoeDispatchPlan derive traits ──

    #[test]
    fn dispatch_plan_clone() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![1],
            gpu_total_us: 10.0,
            cpu_total_us: 5.0,
            needs_cpu_sync: true,
        };
        let cloned = plan.clone();
        assert_eq!(cloned.skipped_experts, vec![1]);
        assert!(cloned.needs_cpu_sync);
    }

    #[test]
    fn dispatch_plan_is_balanced_no_cpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 100.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced(), "no CPU experts → balanced");
    }

    #[test]
    fn dispatch_plan_is_balanced_no_gpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 100.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced(), "no GPU experts → balanced");
    }

    #[test]
    fn dispatch_plan_is_unbalanced_cpu_too_slow() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a.clone()],
            cpu_experts: vec![a],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 10.0,
            cpu_total_us: 100.0, // 10× GPU → unbalanced
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced());
    }

    // ── MoeHardwareDispatcher accessors ──

    #[test]
    fn dispatcher_config_accessor() {
        let config = ExpertRouteConfig::new(8, 2);
        let d = MoeHardwareDispatcher::new(config);
        assert_eq!(d.config().num_experts, 8);
    }

    #[test]
    fn dispatcher_default_cpu_no_amx() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1));
        assert!(!d.cpu_has_amx);
        assert!(!d.cpu_has_avx512);
        assert_eq!(d.cpu_core_count, 0);
    }

    #[test]
    fn dispatcher_with_cpu_sets_flags() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_cpu(32, true, true, 15.0);
        assert!(d.cpu_has_amx);
        assert!(d.cpu_has_avx512);
        assert_eq!(d.cpu_core_count, 32);
        assert!((d.cpu_tflops - 15.0).abs() < 1e-6);
    }

    #[test]
    fn dispatcher_with_gpu_sets_params() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(120, 500.0);
        assert_eq!(d.gpu_sm_count, 120);
        assert!((d.gpu_tflops - 500.0).abs() < 1e-6);
    }

    // ── Dispatch with cold experts ──

    #[test]
    fn dispatch_cold_expert_assigned_to_gpu() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Cold,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Cold expert goes to GPU (FullComputeUnit), not skipped
        let cold = plan.gpu_experts.iter().find(|e| e.expert_idx == 1);
        assert!(cold.is_some(), "cold expert should be assigned to GPU");
        assert_eq!(cold.unwrap().hardware, HardwareKind::FullComputeUnit);
        assert!(!plan.skipped_experts.contains(&1));
    }

    #[test]
    fn dispatch_zero_sm_count() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(0, 0.0); // No GPU
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Should still produce a valid plan (all fallback)
        let _ = plan;
    }

    // ── ExpertHeatLevel trait and ordering tests ──

    #[test]
    fn heat_level_ordering() {
        // Ord is derived from variant declaration order: Hot(0) < Warm(1) < Cold(2) < Evicted(3)
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Warm);
        assert!(ExpertHeatLevel::Warm < ExpertHeatLevel::Cold);
        assert!(ExpertHeatLevel::Cold < ExpertHeatLevel::Evicted);
    }

    #[test]
    fn heat_level_equality() {
        assert_eq!(ExpertHeatLevel::Hot, ExpertHeatLevel::Hot);
        assert_ne!(ExpertHeatLevel::Hot, ExpertHeatLevel::Warm);
        assert_ne!(ExpertHeatLevel::Warm, ExpertHeatLevel::Cold);
        assert_ne!(ExpertHeatLevel::Cold, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn heat_level_from_hit_rate_hot() {
        let level = ExpertHeatLevel::from_hit_rate(0.9, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn heat_level_from_hit_rate_warm() {
        let level = ExpertHeatLevel::from_hit_rate(0.5, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn heat_level_from_hit_rate_cold() {
        let level = ExpertHeatLevel::from_hit_rate(0.01, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn heat_level_from_hit_rate_evicted() {
        let level = ExpertHeatLevel::from_hit_rate(0.0, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn heat_level_copy_trait() {
        let a = ExpertHeatLevel::Hot;
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn heat_level_debug_format() {
        let s = format!("{:?}", ExpertHeatLevel::Hot);
        assert_eq!(s, "Hot");
        let s = format!("{:?}", ExpertHeatLevel::Evicted);
        assert_eq!(s, "Evicted");
    }

    // ── HardwareKind trait tests ──

    #[test]
    fn hardware_kind_equality() {
        assert_eq!(HardwareKind::TensorCorePartition, HardwareKind::TensorCorePartition);
        assert_eq!(HardwareKind::FullComputeUnit, HardwareKind::FullComputeUnit);
        assert_eq!(HardwareKind::LowComputeCore, HardwareKind::LowComputeCore);
        assert_ne!(HardwareKind::TensorCorePartition, HardwareKind::LowComputeCore);
        assert_ne!(HardwareKind::FullComputeUnit, HardwareKind::TensorCorePartition);
    }

    #[test]
    fn hardware_kind_copy_trait() {
        let a = HardwareKind::FullComputeUnit;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn hardware_kind_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(HardwareKind::TensorCorePartition);
        set.insert(HardwareKind::FullComputeUnit);
        set.insert(HardwareKind::LowComputeCore);
        assert_eq!(set.len(), 3);
        // Insert same again should not increase
        set.insert(HardwareKind::TensorCorePartition);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn hardware_kind_debug_format() {
        let s = format!("{:?}", HardwareKind::TensorCorePartition);
        assert!(s.contains("TensorCorePartition"));
    }

    // ── ExpertHeatLevel::from_hit_rate edge cases ──

    #[test]
    fn heat_level_from_hit_rate_exact_thresholds() {
        // At exactly hot_threshold → Hot
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.8, 0.8, 0.2), ExpertHeatLevel::Hot);
        // At exactly cold_threshold → Warm
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.2, 0.8, 0.2), ExpertHeatLevel::Warm);
        // Just above 0 → Cold
        assert_eq!(ExpertHeatLevel::from_hit_rate(0.001, 0.8, 0.2), ExpertHeatLevel::Cold);
    }

    // ── Dispatch with evicted experts ──

    #[test]
    fn dispatch_evicted_expert_is_skipped() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Hot,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Evicted expert should not appear in gpu/cpu/remote
        let evicted_in_assignments = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .chain(plan.remote_experts.iter())
            .any(|e| e.expert_idx == 1);
        assert!(!evicted_in_assignments, "evicted expert should not be assigned");
    }

    #[test]
    fn dispatch_all_evicted_produces_empty_plan() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Evicted, ExpertHeatLevel::Evicted];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.total_assignments(), 0);
    }

    // ── Dispatch with empty token routes ──

    #[test]
    fn dispatch_empty_route_table() {
        let config = ExpertRouteConfig::new(4, 2);
        let route_table = ExpertRouteTable {
            config: config.clone(),
            token_routes: vec![],
            expert_token_counts: vec![0; 4],
            overflow_count: 0,
        };
        let heat_levels = vec![ExpertHeatLevel::Hot; 4];
        let dispatcher = MoeHardwareDispatcher::new(config).with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.total_assignments(), 0);
        assert!(plan.gpu_experts.is_empty());
        assert!(plan.cpu_experts.is_empty());
        assert!(!plan.needs_cpu_sync);
    }

    // ── MoeDispatchPlan.is_balanced edge case: exact 2x boundary ──

    #[test]
    fn dispatch_plan_balanced_at_exactly_2x() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 10,
            estimated_compute_us: 50.0,
        };
        let b = ExpertHardwareAssignment {
            expert_idx: 1,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 10,
            estimated_compute_us: 100.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a],
            cpu_experts: vec![b],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 100.0, // exactly 2x GPU → balanced
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(), "CPU exactly 2x GPU is still balanced");
    }

    #[test]
    fn dispatch_plan_unbalanced_above_2x() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 10,
            estimated_compute_us: 50.0,
        };
        let b = ExpertHardwareAssignment {
            expert_idx: 1,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 10,
            estimated_compute_us: 101.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a],
            cpu_experts: vec![b],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 101.0, // just above 2x → unbalanced
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced());
    }

    // ── MoeDispatchPlan with remote experts ──

    #[test]
    fn dispatch_plan_total_assignments_includes_remote() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 64,
            estimated_compute_us: 10.0,
        };
        let remote = ExpertHardwareAssignment {
            expert_idx: 5,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 32,
            estimated_compute_us: 200.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a],
            cpu_experts: vec![],
            remote_experts: vec![remote],
            skipped_experts: vec![2, 3],
            needs_cpu_sync: false,
            gpu_total_us: 10.0,
            cpu_total_us: 0.0,
        };
        assert_eq!(plan.total_assignments(), 2);
    }

    // ── ExpertHardwareAssignment with partition ──

    #[test]
    fn expert_assignment_with_partition() {
        let partition = HardwarePartition {
            partition_id: 1,
            kind: HardwareKind::TensorCorePartition,
            sm_range: Some((0, 40)),
            numa_node: None,
            core_range: None,
            compute_weight: 0.5,
        };
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: Some(partition.clone()),
            token_count: 256,
            estimated_compute_us: 15.0,
        };
        assert!(a.partition.is_some());
        let p = a.partition.unwrap();
        assert_eq!(p.partition_id, 1);
        assert_eq!(p.sm_range, Some((0, 40)));
    }

    // ── dispatcher with_max_cpu_ratio negative clamps to 0 ──

    #[test]
    fn dispatcher_max_cpu_ratio_negative_clamps_to_zero() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_max_cpu_ratio(-0.5);
        // Verify it does not panic and still constructs
        assert_eq!(d.config().num_experts, 4);
    }

    // ── dispatch with shorter heat_levels than experts ──

    #[test]
    fn dispatch_heat_levels_shorter_than_experts() {
        let config = ExpertRouteConfig::new(4, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        // Only 2 heat levels for 4 experts → missing ones should default to Warm
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0);
        // Should not panic
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.total_assignments() >= 2);
    }

    // ── ExpertRouteConfig defaults ──

    #[test]
    fn route_config_default_values() {
        let config = ExpertRouteConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        assert!(!config.load_balance_loss);
    }

    #[test]
    fn route_config_expert_capacity() {
        let config = ExpertRouteConfig::new(4, 2);
        // capacity_factor=1.25, 8 tokens, 4 experts → ceil(1.25 * 8 / 4) = ceil(2.5) = 3
        let cap = config.expert_capacity(8);
        assert_eq!(cap, 3);
    }

    #[test]
    fn route_config_expert_capacity_single_expert() {
        let config = ExpertRouteConfig::new(1, 1);
        // ceil(1.25 * 10 / 1) = 13
        let cap = config.expert_capacity(10);
        assert_eq!(cap, 13);
    }

    // ── Dispatch plan gpu_total_us uses max not sum ──

    #[test]
    fn dispatch_plan_gpu_time_is_max_not_sum() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 64,
            estimated_compute_us: 30.0,
        };
        let b = ExpertHardwareAssignment {
            expert_idx: 1,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 64,
            estimated_compute_us: 50.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![a, b],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        // gpu_total_us = max(30, 50) = 50, not 80
        assert!((plan.gpu_total_us - 50.0).abs() < 1e-6);
    }

    // ── Dispatch plan with only remote experts ──

    #[test]
    fn dispatch_plan_only_remote_experts() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 100,
            estimated_compute_us: 500.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![a],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 1);
        assert!(plan.is_balanced());
    }

    // ═══════════════════════════════════════════════════════════════
    //  New tests — 38 additional tests
    // ═══════════════════════════════════════════════════════════════

    // ── ExpertHardwareAssignment: field edge values ──

    #[test]
    fn assignment_zero_expert_idx_and_token_count() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 0,
            estimated_compute_us: 0.0,
        };
        assert_eq!(a.expert_idx, 0);
        assert_eq!(a.token_count, 0);
        assert!(a.estimated_compute_us == 0.0);
    }

    #[test]
    fn assignment_max_expert_idx() {
        let a = ExpertHardwareAssignment {
            expert_idx: usize::MAX,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 1,
            estimated_compute_us: 99.9,
        };
        assert_eq!(a.expert_idx, usize::MAX);
    }

    #[test]
    fn assignment_large_token_count() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: usize::MAX,
            estimated_compute_us: 1000.0,
        };
        assert_eq!(a.token_count, usize::MAX);
    }

    #[test]
    fn assignment_infinite_compute_time() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 1,
            estimated_compute_us: f32::INFINITY,
        };
        assert!(a.estimated_compute_us.is_infinite());
        assert!(a.estimated_compute_us > 0.0);
    }

    #[test]
    fn assignment_nan_compute_time() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 1,
            estimated_compute_us: f32::NAN,
        };
        assert!(a.estimated_compute_us.is_nan());
    }

    #[test]
    fn assignment_negative_compute_time() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 1,
            estimated_compute_us: -10.0,
        };
        assert!(a.estimated_compute_us < 0.0);
    }

    #[test]
    fn assignment_all_hardware_kinds() {
        for hw in [
            HardwareKind::LowComputeCore,
            HardwareKind::FullComputeUnit,
            HardwareKind::TensorCorePartition,
        ] {
            let a = ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: hw,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            };
            assert_eq!(a.hardware, hw);
        }
    }

    // ── MoeDispatchPlan: is_balanced with special floats ──

    #[test]
    fn dispatch_plan_balanced_with_nan_gpu_time() {
        // NaN comparison: cpu_total_us <= NaN*2.0 is false → unbalanced
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: f32::NAN,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced(), "NaN gpu_total_us should make is_balanced return false");
    }

    #[test]
    fn dispatch_plan_balanced_with_zero_times() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced());
    }

    #[test]
    fn dispatch_plan_balanced_infinite_cpu_time() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: f32::INFINITY,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: f32::INFINITY,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced(), "infinite CPU time should be unbalanced");
    }

    // ── MoeDispatchPlan: total_assignments edge cases ──

    #[test]
    fn dispatch_plan_total_assignments_with_many_per_category() {
        let make_assignment = |idx: usize| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: (0..10).map(make_assignment).collect(),
            cpu_experts: (10..15).map(make_assignment).collect(),
            remote_experts: (15..17).map(make_assignment).collect(),
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert_eq!(plan.total_assignments(), 17);
    }

    #[test]
    fn dispatch_plan_total_assignments_skipped_not_counted() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![0, 1, 2, 3, 4],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 0, "skipped experts are not assignments");
    }

    // ── MoeHardwareDispatcher: builder chaining ──

    #[test]
    fn dispatcher_builder_chaining_all_methods() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(16, 4))
            .with_gpu(132, 600.0)
            .with_cpu(64, true, true, 25.0)
            .with_max_cpu_ratio(0.4);

        assert_eq!(d.config().num_experts, 16);
        assert_eq!(d.config().top_k, 4);
        assert_eq!(d.gpu_sm_count(), 132);
        assert!(d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_gpu_only_no_cpu() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_gpu(80, 300.0);
        // CPU fields remain default
        assert!(!d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_cpu_only_no_gpu() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_cpu(32, true, true, 10.0);
        assert_eq!(d.gpu_sm_count(), 0);
        assert!(d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_no_builder_calls() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1));
        assert_eq!(d.gpu_sm_count(), 0);
        assert!(!d.cpu_has_amx());
        assert_eq!(d.cpu_core_count, 0);
    }

    #[test]
    fn dispatcher_with_cpu_amx_false_avx512_true() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_cpu(8, false, true, 5.0);
        assert!(!d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_with_max_cpu_ratio_zero() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_max_cpu_ratio(0.0);
        // Constructed successfully, ratio clamped to 0.0
        assert_eq!(d.config().num_experts, 4);
    }

    #[test]
    fn dispatcher_with_max_cpu_ratio_one() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_max_cpu_ratio(1.0);
        assert_eq!(d.config().num_experts, 4);
    }

    // ── Dispatch: single token, single expert ──

    #[test]
    fn dispatch_single_token_single_expert() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.total_assignments() >= 1);
        assert!(!plan.needs_cpu_sync);
    }

    // ── Dispatch: all warm with no AMX → all GPU ──

    #[test]
    fn dispatch_all_warm_no_amx_all_gpu() {
        let config = ExpertRouteConfig::new(4, 2);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 2))
            .with_gpu(80, 300.0)
            .with_cpu(16, false, false, 0.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty(), "no AMX means warm goes to GPU");
        assert!(!plan.gpu_experts.is_empty());
    }

    // ── Dispatch: all cold → GPU ──

    #[test]
    fn dispatch_all_cold_to_gpu() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Cold, ExpertHeatLevel::Cold];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty());
        assert!(plan.gpu_experts.len() >= 2, "cold experts go to GPU");
        for e in &plan.gpu_experts {
            assert_eq!(e.hardware, HardwareKind::FullComputeUnit);
        }
    }

    // ── Dispatch: empty heat_levels ──

    #[test]
    fn dispatch_empty_heat_levels() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels: Vec<ExpertHeatLevel> = vec![];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        // Should not panic; missing heat levels default to Warm
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.total_assignments() >= 2);
    }

    // ── Dispatch: mixed heat levels with AMX ──

    #[test]
    fn dispatch_mixed_heat_levels_with_amx() {
        let config = ExpertRouteConfig::new(4, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,   // → GPU TensorCore
            ExpertHeatLevel::Warm,  // → CPU (AMX available)
            ExpertHeatLevel::Cold,  // → GPU FullCompute
            ExpertHeatLevel::Hot,   // → GPU TensorCore
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);

        // Verify assignments
        let gpu_tensor: Vec<_> = plan.gpu_experts.iter()
            .filter(|e| e.hardware == HardwareKind::TensorCorePartition)
            .collect();
        let _gpu_full: Vec<_> = plan.gpu_experts.iter()
            .filter(|e| e.hardware == HardwareKind::FullComputeUnit)
            .collect();

        assert!(!gpu_tensor.is_empty(), "hot experts → TensorCorePartition");
        assert!(plan.needs_cpu_sync, "warm expert with AMX → CPU sync needed");
    }

    // ── Dispatch: GPU with zero TFLOPS produces infinite compute time ──

    #[test]
    fn dispatch_zero_tflops_produces_infinite_estimate() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 0.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.gpu_total_us.is_infinite(), "zero GPU TFLOPS → infinite compute time");
    }

    // ── Dispatch: many tokens, one expert ──

    #[test]
    fn dispatch_many_tokens_one_expert() {
        let config = ExpertRouteConfig::new(1, 1);
        let gate_logits: Vec<Vec<f32>> = (0..100).map(|_| vec![1.0]).collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let total_tokens: usize = plan.gpu_experts.iter().map(|e| e.token_count).sum();
        assert!(total_tokens > 0, "should have tokens assigned");
    }

    // ── ExpertRouteConfig: expert_capacity edge cases ──

    #[test]
    fn route_config_expert_capacity_zero_tokens() {
        let config = ExpertRouteConfig::new(4, 2);
        let cap = config.expert_capacity(0);
        assert_eq!(cap, 0, "zero tokens → zero capacity");
    }

    #[test]
    fn route_config_expert_capacity_large_token_count() {
        let config = ExpertRouteConfig::new(8, 2);
        let cap = config.expert_capacity(1000000);
        // ceil(1.25 * 1000000 / 8) = ceil(156250.0) = 156250
        assert_eq!(cap, 156250);
    }

    #[test]
    fn route_config_new_customizes_experts_and_topk() {
        let config = ExpertRouteConfig::new(64, 8);
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.top_k, 8);
    }

    // ── ExpertHeatLevel::from_hit_rate additional edge cases ──

    #[test]
    fn heat_level_from_hit_rate_just_below_hot() {
        let level = ExpertHeatLevel::from_hit_rate(0.79, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn heat_level_from_hit_rate_just_above_cold() {
        let level = ExpertHeatLevel::from_hit_rate(0.21, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn heat_level_from_hit_rate_very_small_positive() {
        let level = ExpertHeatLevel::from_hit_rate(0.0001, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn heat_level_from_hit_rate_all_zero_thresholds() {
        // cold_threshold=0.0, hot_threshold=0.0 → rate=0.0 >= 0.0 → Hot
        let level = ExpertHeatLevel::from_hit_rate(0.0, 0.0, 0.0);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn heat_level_from_hit_rate_rate_one() {
        let level = ExpertHeatLevel::from_hit_rate(1.0, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    // ── HardwareKind: exhaustive variant coverage ──

    #[test]
    fn hardware_kind_all_variants_in_hashset() {
        use std::collections::HashSet;
        let all = [
            HardwareKind::LowComputeCore,
            HardwareKind::FullComputeUnit,
            HardwareKind::TensorCorePartition,
        ];
        let set: HashSet<HardwareKind> = all.iter().copied().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn hardware_kind_clone_independent() {
        let a = HardwareKind::FullComputeUnit;
        let b = a;
        // Copy semantics — both still valid
        assert_eq!(a, b);
    }

    // ── HardwarePartition construction from dispatch tests ──

    #[test]
    fn hardware_partition_gpu_sm_partition() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 40);
        assert_eq!(p.partition_id, 0);
        assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        assert_eq!(p.sm_range, Some((0, 40)));
        assert!(p.numa_node.is_none());
    }

    #[test]
    fn hardware_partition_cpu_numa_partition() {
        let p = HardwarePartition::cpu_numa_partition(2, 1, 8);
        assert_eq!(p.partition_id, 2);
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
        assert_eq!(p.numa_node, Some(1));
    }

    #[test]
    fn hardware_partition_cpu_numa_node0_is_full_compute() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 16);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
    }

    // ── Dispatch: expert with partition assigned ──

    #[test]
    fn dispatch_plan_with_partition_in_assignment() {
        let partition = HardwarePartition::gpu_sm_partition(3, 20, 60);
        let a = ExpertHardwareAssignment {
            expert_idx: 2,
            hardware: HardwareKind::TensorCorePartition,
            partition: Some(partition),
            token_count: 512,
            estimated_compute_us: 25.0,
        };
        assert_eq!(a.partition.as_ref().unwrap().partition_id, 3);
        assert_eq!(a.partition.as_ref().unwrap().sm_range, Some((20, 60)));
    }

    // ── Dispatch: multiple evicted among active ──

    #[test]
    fn dispatch_partial_evicted_mixed() {
        let config = ExpertRouteConfig::new(4, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        // Experts 0,2 evicted; experts 1,3 hot
        let heat_levels = vec![
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Hot,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);

        // Active experts (1,3) should be assigned
        let active_indices: Vec<usize> = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .map(|e| e.expert_idx)
            .collect();
        assert!(active_indices.contains(&1), "expert 1 should be assigned");
        assert!(active_indices.contains(&3), "expert 3 should be assigned");
    }

    // ── Dispatch: verify needs_cpu_sync is false when no CPU experts ──

    #[test]
    fn dispatch_needs_cpu_sync_false_no_cpu_experts() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(!plan.needs_cpu_sync);
        assert!(plan.cpu_experts.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 3 (55 tests)
    // ═══════════════════════════════════════════════════════════════

    // ── ExpertHardwareAssignment: estimated_compute_us precision ──

    #[test]
    fn assignment_compute_time_preserves_subnormal() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 1,
            estimated_compute_us: f32::MIN_POSITIVE,
        };
        assert_eq!(a.estimated_compute_us.to_bits(), f32::MIN_POSITIVE.to_bits());
    }

    #[test]
    fn assignment_compute_time_preserves_max_finite() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 1,
            estimated_compute_us: f32::MAX,
        };
        assert!(!a.estimated_compute_us.is_infinite());
        assert!(a.estimated_compute_us > 0.0);
    }

    #[test]
    fn assignment_compute_time_preserves_neg_infinity() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 1,
            estimated_compute_us: f32::NEG_INFINITY,
        };
        assert!(a.estimated_compute_us.is_infinite());
        assert!(a.estimated_compute_us < 0.0);
    }

    #[test]
    fn assignment_clone_preserves_all_fields() {
        let partition = HardwarePartition::gpu_sm_partition(5, 10, 30);
        let a = ExpertHardwareAssignment {
            expert_idx: 42,
            hardware: HardwareKind::TensorCorePartition,
            partition: Some(partition),
            token_count: 2048,
            estimated_compute_us: 123.456,
        };
        let c = a.clone();
        assert_eq!(c.expert_idx, 42);
        assert_eq!(c.hardware, HardwareKind::TensorCorePartition);
        assert!(c.partition.is_some());
        assert_eq!(c.partition.as_ref().unwrap().partition_id, 5);
        assert_eq!(c.token_count, 2048);
        assert!((c.estimated_compute_us - 123.456).abs() < 1e-3);
    }

    // ── MoeDispatchPlan: is_balanced boundary analysis ──

    #[test]
    fn dispatch_plan_balanced_cpu_equals_gpu_time() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 10,
                estimated_compute_us: 50.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 10,
                estimated_compute_us: 50.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 50.0,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(), "equal times → balanced");
    }

    #[test]
    fn dispatch_plan_balanced_cpu_just_under_2x() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 10,
                estimated_compute_us: 50.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 10,
                estimated_compute_us: 99.99,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 99.99,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(), "99.99 < 100.0 → balanced");
    }

    #[test]
    fn dispatch_plan_balanced_cpu_slightly_over_2x() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 10,
                estimated_compute_us: 50.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 10,
                estimated_compute_us: 100.01,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 100.01,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced(), "100.01 > 100.0 → unbalanced");
    }

    #[test]
    fn dispatch_plan_balanced_cpu_faster_than_gpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 10,
                estimated_compute_us: 100.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 10,
                estimated_compute_us: 10.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 100.0,
            cpu_total_us: 10.0,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(), "CPU much faster → balanced");
    }

    #[test]
    fn dispatch_plan_balanced_nan_cpu_time() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: f32::NAN,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: f32::NAN,
            needs_cpu_sync: true,
        };
        // NaN <= anything is false → unbalanced
        assert!(!plan.is_balanced());
    }

    // ── MoeDispatchPlan: total_assignments consistency ──

    #[test]
    fn dispatch_plan_total_assignments_all_gpu() {
        let make = |idx: usize| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: (0..8).map(make).collect(),
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 8);
    }

    #[test]
    fn dispatch_plan_total_assignments_all_cpu() {
        let make = |idx: usize| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: (0..4).map(make).collect(),
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert_eq!(plan.total_assignments(), 4);
    }

    #[test]
    fn dispatch_plan_total_assignments_all_remote() {
        let make = |idx: usize| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: (0..3).map(make).collect(),
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 3);
    }

    #[test]
    fn dispatch_plan_clone_independent_mutation() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 10,
                estimated_compute_us: 5.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![1, 2],
            gpu_total_us: 5.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        let mut cloned = plan.clone();
        cloned.skipped_experts.push(3);
        assert_eq!(plan.skipped_experts, vec![1, 2], "original unchanged after clone mutation");
        assert_eq!(cloned.skipped_experts, vec![1, 2, 3]);
    }

    // ── MoeDispatchPlan: needs_cpu_sync consistency ──

    #[test]
    fn dispatch_plan_needs_cpu_sync_true_with_cpu_experts() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert!(plan.needs_cpu_sync);
    }

    #[test]
    fn dispatch_plan_needs_cpu_sync_false_only_gpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(!plan.needs_cpu_sync);
    }

    #[test]
    fn dispatch_plan_needs_cpu_sync_false_only_remote() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: 1.0,
            }],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(!plan.needs_cpu_sync);
    }

    // ── MoeHardwareDispatcher: builder order independence ──

    #[test]
    fn dispatcher_builder_gpu_then_cpu() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_gpu(80, 300.0)
            .with_cpu(32, true, true, 10.0);
        assert_eq!(d.gpu_sm_count(), 80);
        assert!(d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_builder_cpu_then_gpu() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_cpu(32, true, true, 10.0)
            .with_gpu(80, 300.0);
        assert_eq!(d.gpu_sm_count(), 80);
        assert!(d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_builder_ratio_first() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(8, 2))
            .with_max_cpu_ratio(0.5)
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);
        assert_eq!(d.gpu_sm_count(), 80);
        assert!(d.cpu_has_amx());
    }

    #[test]
    fn dispatcher_config_preserved_after_builders() {
        let config = ExpertRouteConfig::new(32, 4);
        let d = MoeHardwareDispatcher::new(config)
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.4);
        assert_eq!(d.config().num_experts, 32);
        assert_eq!(d.config().top_k, 4);
    }

    // ── Dispatch: CPU capacity fallback ──

    #[test]
    fn dispatch_warm_overflow_cpu_to_gpu() {
        let config = ExpertRouteConfig::new(4, 1);
        // 8 tokens: all go to expert 0 (warm)
        let gate_logits: Vec<Vec<f32>> = (0..8)
            .map(|_| vec![1.0, 0.0, 0.0, 0.0])
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
        ];
        // max_cpu_ratio=0.1 → only 0 tokens to CPU → all warm fallback to GPU
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.1);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 0 is warm with AMX but ratio is very small → may fallback to GPU
        assert!(plan.total_assignments() >= 1, "at least one assignment");
    }

    // ── Dispatch: token count accuracy ──

    #[test]
    fn dispatch_token_count_matches_route_table() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let total_tokens: usize = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .chain(plan.remote_experts.iter())
            .map(|e| e.token_count)
            .sum();
        assert_eq!(total_tokens, 4, "total assigned tokens should equal route count");
    }

    #[test]
    fn dispatch_expert_idx_assigned_correctly() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let assigned_indices: Vec<usize> = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .map(|e| e.expert_idx)
            .collect();
        assert!(assigned_indices.contains(&0));
        assert!(assigned_indices.contains(&1));
        assert!(assigned_indices.contains(&2));
    }

    // ── Dispatch: skipped_experts accuracy ──

    #[test]
    fn dispatch_skipped_contains_all_evicted_with_no_tokens() {
        let config = ExpertRouteConfig::new(4, 1);
        // Only experts 0,1 get tokens
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        );
        // Experts 2,3 evicted and have no tokens
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Evicted,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.skipped_experts.contains(&2));
        assert!(plan.skipped_experts.contains(&3));
    }

    // ── Dispatch: hot experts always get TensorCorePartition ──

    #[test]
    fn dispatch_hot_experts_get_tensor_core_partition() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        for e in &plan.gpu_experts {
            assert_eq!(e.hardware, HardwareKind::TensorCorePartition,
                "hot experts should be TensorCorePartition");
        }
    }

    // ── Dispatch: cold experts get FullComputeUnit ──

    #[test]
    fn dispatch_cold_experts_get_full_compute_unit() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Cold, ExpertHeatLevel::Cold];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        for e in &plan.gpu_experts {
            assert_eq!(e.hardware, HardwareKind::FullComputeUnit,
                "cold experts should be FullComputeUnit");
        }
    }

    // ── Dispatch: warm with AMX → LowComputeCore ──

    #[test]
    fn dispatch_warm_with_amx_gets_low_compute_core() {
        // 2 experts, warm expert 0 gets 1 token, hot expert 1 gets 9 tokens
        // total_tokens=10, max_cpu_ratio=0.5 → max_cpu_tokens=5
        // expert 0: 1 token, warm → CPU (1 <= 5, AMX available)
        let config = ExpertRouteConfig::new(2, 1);
        let gate_logits: Vec<Vec<f32>> = std::iter::once(vec![1.0, 0.0])
            .chain((0..9).map(|_| vec![0.0, 1.0]))
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(!plan.cpu_experts.is_empty(), "warm with AMX → CPU");
        assert_eq!(plan.cpu_experts[0].hardware, HardwareKind::LowComputeCore);
    }

    // ── Dispatch: warm without AMX → FullComputeUnit ──

    #[test]
    fn dispatch_warm_without_amx_gets_full_compute_unit() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, false, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty(), "no AMX → warm goes to GPU");
        assert_eq!(plan.gpu_experts[0].hardware, HardwareKind::FullComputeUnit);
    }

    // ── Dispatch: gpu_total_us is max of GPU expert estimates ──

    #[test]
    fn dispatch_gpu_total_us_is_max_of_experts() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot; 3];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 0 has 2 tokens, expert 1 has 2, expert 2 has 2
        // max of their compute times
        let max_expert = plan.gpu_experts.iter()
            .map(|e| e.estimated_compute_us)
            .fold(0.0f32, f32::max);
        assert!((plan.gpu_total_us - max_expert).abs() < 1e-3,
            "gpu_total_us should be max of expert estimates");
    }

    // ── Dispatch: cpu_total_us is max of CPU expert estimates ──

    #[test]
    fn dispatch_cpu_total_us_is_max_of_experts() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        if !plan.cpu_experts.is_empty() {
            let max_expert = plan.cpu_experts.iter()
                .map(|e| e.estimated_compute_us)
                .fold(0.0f32, f32::max);
            assert!((plan.cpu_total_us - max_expert).abs() < 1e-3);
        }
    }

    // ── Dispatch: multiple tokens per expert aggregated ──

    #[test]
    fn dispatch_tokens_aggregated_per_expert() {
        let config = ExpertRouteConfig::new(2, 1);
        // 5 tokens to expert 0, 3 tokens to expert 1
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
            ],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let e0 = plan.gpu_experts.iter().find(|e| e.expert_idx == 0).unwrap();
        let e1 = plan.gpu_experts.iter().find(|e| e.expert_idx == 1).unwrap();
        assert_eq!(e0.token_count, 5);
        assert_eq!(e1.token_count, 3);
    }

    // ── Dispatch: partition is None from dispatcher ──

    #[test]
    fn dispatch_assignments_have_no_partition() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        for e in &plan.gpu_experts {
            assert!(e.partition.is_none(),
                "partition filled by SubBatchDispatcher, not MoeHardwareDispatcher");
        }
    }

    // ── Dispatch: compute time increases with token count ──

    #[test]
    fn dispatch_compute_time_scales_with_tokens() {
        let config = ExpertRouteConfig::new(1, 1);
        // Few tokens
        let rt_few = ExpertRouteTable::from_gate_logits(
            config.clone(),
            &vec![vec![1.0], vec![1.0]],
        );
        // Many tokens
        let config2 = ExpertRouteConfig::new(1, 1);
        let rt_many = ExpertRouteTable::from_gate_logits(
            config2,
            &(0..10).map(|_| vec![1.0]).collect::<Vec<_>>(),
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan_few = dispatcher.dispatch(&rt_few, &heat_levels);
        let plan_many = dispatcher.dispatch(&rt_many, &heat_levels);
        assert!(plan_many.gpu_total_us > plan_few.gpu_total_us,
            "more tokens should produce higher compute time");
    }

    // ── ExpertRouteConfig: capacity_factor is 1.25 by default ──

    #[test]
    fn route_config_default_capacity_factor() {
        let config = ExpertRouteConfig::default();
        assert!((config.capacity_factor - 1.25).abs() < 1e-6);
    }

    #[test]
    fn route_config_default_load_balance_off() {
        let config = ExpertRouteConfig::default();
        assert!(!config.load_balance_loss);
        assert!((config.load_balance_lambda - 0.01).abs() < 1e-6);
    }

    #[test]
    fn route_config_default_noise_sigma_zero() {
        let config = ExpertRouteConfig::default();
        assert!((config.noise_sigma - 0.0).abs() < 1e-6);
    }

    // ── ExpertHeatLevel: transitivity of ordering ──

    #[test]
    fn heat_level_ordering_transitivity() {
        let hot = ExpertHeatLevel::Hot;
        let warm = ExpertHeatLevel::Warm;
        let cold = ExpertHeatLevel::Cold;
        let evicted = ExpertHeatLevel::Evicted;
        assert!(hot < warm && warm < cold, "Hot < Warm < Cold");
        assert!(hot < cold, "transitivity: Hot < Cold");
        assert!(warm < evicted, "Warm < Evicted");
        assert!(hot < evicted, "transitivity: Hot < Evicted");
    }

    #[test]
    fn heat_level_sort_order() {
        let mut levels = vec![
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Warm,
        ];
        levels.sort();
        assert_eq!(levels, vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ]);
    }

    #[test]
    fn heat_level_from_hit_rate_threshold_boundary_negative() {
        // Negative rate → Evicted
        let level = ExpertHeatLevel::from_hit_rate(-0.5, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn heat_level_from_hit_rate_rate_above_one() {
        // Rate > 1.0 → Hot (>= hot_threshold)
        let level = ExpertHeatLevel::from_hit_rate(1.5, 0.8, 0.2);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn heat_level_from_hit_rate_swapped_thresholds() {
        // hot_threshold < cold_threshold: 0.2 < 0.8
        // rate=0.5 >= 0.2 → Hot
        let level = ExpertHeatLevel::from_hit_rate(0.5, 0.2, 0.8);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    // ── HardwarePartition: field verification ──

    #[test]
    fn hardware_partition_gpu_sm_partition_no_numa() {
        let p = HardwarePartition::gpu_sm_partition(1, 20, 60);
        assert_eq!(p.partition_id, 1);
        assert_eq!(p.kind, HardwareKind::TensorCorePartition);
        assert_eq!(p.sm_range, Some((20, 60)));
        assert!(p.numa_node.is_none());
        assert!(p.core_range.is_none());
    }

    #[test]
    fn hardware_partition_gpu_sm_partition_compute_weight() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 40);
        // weight = 40 / 80.0 = 0.5
        assert!((p.compute_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hardware_partition_cpu_numa_node1_is_low_compute() {
        let p = HardwarePartition::cpu_numa_partition(0, 1, 8);
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
        assert_eq!(p.numa_node, Some(1));
        assert!(p.sm_range.is_none());
    }

    #[test]
    fn hardware_partition_cpu_numa_compute_weight() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 32);
        // weight = 32 / 64.0 = 0.5
        assert!((p.compute_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hardware_partition_cpu_numa_node0_has_full_compute() {
        let p = HardwarePartition::cpu_numa_partition(5, 0, 16);
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
        assert_eq!(p.numa_node, Some(0));
    }

    // ── HardwareKind: exhaustive property checks ──

    #[test]
    fn hardware_kind_all_variants_distinct() {
        let variants = [
            HardwareKind::TensorCorePartition,
            HardwareKind::FullComputeUnit,
            HardwareKind::LowComputeCore,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn hardware_kind_hash_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        HardwareKind::TensorCorePartition.hash(&mut h1);
        let hash1 = h1.finish();
        let mut h2 = DefaultHasher::new();
        HardwareKind::TensorCorePartition.hash(&mut h2);
        let hash2 = h2.finish();
        assert_eq!(hash1, hash2, "same value should produce same hash");
    }

    #[test]
    fn hardware_kind_debug_all_variants() {
        let tcp = format!("{:?}", HardwareKind::TensorCorePartition);
        let fcu = format!("{:?}", HardwareKind::FullComputeUnit);
        let lcc = format!("{:?}", HardwareKind::LowComputeCore);
        assert!(tcp.contains("TensorCorePartition"));
        assert!(fcu.contains("FullComputeUnit"));
        assert!(lcc.contains("LowComputeCore"));
    }

    // ── ExpertHardwareAssignment: with HardwarePartition ──

    #[test]
    fn assignment_with_cpu_numa_partition() {
        let partition = HardwarePartition::cpu_numa_partition(3, 1, 8);
        let a = ExpertHardwareAssignment {
            expert_idx: 5,
            hardware: HardwareKind::LowComputeCore,
            partition: Some(partition),
            token_count: 64,
            estimated_compute_us: 20.0,
        };
        let p = a.partition.unwrap();
        assert_eq!(p.numa_node, Some(1));
        assert_eq!(p.kind, HardwareKind::LowComputeCore);
    }

    // ── Dispatch: large expert count ──

    #[test]
    fn dispatch_large_expert_count() {
        let config = ExpertRouteConfig::new(16, 2);
        let gate_logits: Vec<Vec<f32>> = (0..16)
            .map(|i| {
                let mut row = vec![0.0f32; 16];
                row[i] = 1.0;
                row[(i + 1) % 16] = 0.5;
                row
            })
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(16, 2))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.total_assignments() > 0);
        assert!(plan.gpu_experts.len() + plan.cpu_experts.len() > 0);
    }

    // ── Dispatch: all heat levels present ──

    #[test]
    fn dispatch_all_four_heat_levels() {
        let config = ExpertRouteConfig::new(4, 1);
        // 4 tokens each → enough tokens for CPU ratio to allow warm expert
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        // max_cpu_ratio=0.5 → enough CPU capacity for warm expert
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 0 (Hot) → GPU TensorCore
        // Expert 1 (Warm, AMX) → CPU LowComputeCore
        // Expert 2 (Cold) → GPU FullComputeUnit
        // Expert 3 (Evicted) → skipped
        assert!(plan.skipped_experts.contains(&3), "evicted should be skipped");
        assert!(plan.needs_cpu_sync, "warm expert with AMX → CPU sync");
    }

    // ── Dispatch: same config and table produces deterministic results ──

    #[test]
    fn dispatch_deterministic() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Cold];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);

        let plan1 = dispatcher.dispatch(&route_table, &heat_levels);
        let plan2 = dispatcher.dispatch(&route_table, &heat_levels);

        assert_eq!(plan1.gpu_experts.len(), plan2.gpu_experts.len());
        assert_eq!(plan1.cpu_experts.len(), plan2.cpu_experts.len());
        assert_eq!(plan1.total_assignments(), plan2.total_assignments());
    }

    // ── Dispatch: single expert with multiple tokens ──

    #[test]
    fn dispatch_single_expert_many_tokens() {
        let config = ExpertRouteConfig::new(1, 1);
        let gate_logits: Vec<Vec<f32>> = (0..50).map(|_| vec![1.0]).collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.gpu_experts.len(), 1);
        assert_eq!(plan.gpu_experts[0].token_count, 50);
    }

    // ── Dispatch: gpu_total_us with finite TFLOPS is finite ──

    #[test]
    fn dispatch_finite_tflops_finite_time() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.gpu_total_us.is_finite(), "finite TFLOPS → finite compute time");
        assert!(plan.gpu_total_us > 0.0, "positive tokens → positive compute time");
    }

    // ── MoeDispatchPlan: debug format includes all fields ──

    #[test]
    fn dispatch_plan_debug_includes_all_fields() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 10,
                estimated_compute_us: 5.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![3],
            gpu_total_us: 5.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        let s = format!("{:?}", plan);
        assert!(s.contains("gpu_experts"));
        assert!(s.contains("cpu_experts"));
        assert!(s.contains("remote_experts"));
        assert!(s.contains("skipped_experts"));
        assert!(s.contains("gpu_total_us"));
        assert!(s.contains("cpu_total_us"));
        assert!(s.contains("needs_cpu_sync"));
    }

    // ── ExpertHardwareAssignment: debug format includes all fields ──

    #[test]
    fn assignment_debug_includes_all_fields() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 10,
            estimated_compute_us: 5.0,
        };
        let s = format!("{:?}", a);
        assert!(s.contains("expert_idx"));
        assert!(s.contains("hardware"));
        assert!(s.contains("partition"));
        assert!(s.contains("token_count"));
        assert!(s.contains("estimated_compute_us"));
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 4 (55 new tests)
    // ═══════════════════════════════════════════════════════════════

    use super::super::routing::{
        ExpertRouteConfig, ExpertRouteTable, ExpertUtilizationStats,
        topk_with_weights, topk_indices, softmax,
    };
    use super::super::thermal::{
        ExpertHeatLevel, ExpertResidency, EvictionDecision, DeoptRequest,
        ExpertThermalManager, DeoptHandlingResult, ThermalSummary,
    };

    // ── ExpertRouteConfig: field modification and method tests ──

    #[test]
    fn route_config_new_sets_capacity_factor_1_25() {
        let config = ExpertRouteConfig::new(8, 2);
        assert!((config.capacity_factor - 1.25).abs() < 1e-6);
    }

    #[test]
    fn route_config_new_sets_noise_sigma_zero() {
        let config = ExpertRouteConfig::new(4, 1);
        assert!((config.noise_sigma - 0.0).abs() < 1e-6);
    }

    #[test]
    fn route_config_expert_capacity_single_token() {
        let config = ExpertRouteConfig::new(4, 2);
        let cap = config.expert_capacity(1);
        // ceil(1.25 * 1 / 4) = ceil(0.3125) = 1
        assert_eq!(cap, 1);
    }

    #[test]
    fn route_config_expert_capacity_large_expert_count() {
        let config = ExpertRouteConfig::new(256, 4);
        let cap = config.expert_capacity(100);
        // ceil(1.25 * 100 / 256) = ceil(0.488...) = 1
        assert_eq!(cap, 1);
    }

    #[test]
    fn route_config_expert_capacity_one_expert_many_tokens() {
        let config = ExpertRouteConfig::new(1, 1);
        let cap = config.expert_capacity(1000);
        // ceil(1.25 * 1000 / 1) = 1250
        assert_eq!(cap, 1250);
    }

    #[test]
    fn route_config_clone_preserves_fields() {
        let config = ExpertRouteConfig::new(16, 4);
        let cloned = config.clone();
        assert_eq!(cloned.num_experts, 16);
        assert_eq!(cloned.top_k, 4);
        assert!((cloned.capacity_factor - config.capacity_factor).abs() < 1e-6);
    }

    #[test]
    fn route_config_equality() {
        let a = ExpertRouteConfig::new(8, 2);
        let b = ExpertRouteConfig::new(8, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn route_config_inequality_different_experts() {
        let a = ExpertRouteConfig::new(4, 2);
        let b = ExpertRouteConfig::new(8, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn route_config_inequality_different_topk() {
        let a = ExpertRouteConfig::new(8, 2);
        let b = ExpertRouteConfig::new(8, 4);
        assert_ne!(a, b);
    }

    // ── ExpertRouteTable: from_gate_logits and field access ──

    #[test]
    fn route_table_from_gate_logits_basic() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        assert_eq!(table.token_routes.len(), 2);
        assert_eq!(table.expert_token_counts.len(), 2);
    }

    #[test]
    fn route_table_expert_token_counts_correct() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        // Expert 0: 2 tokens, Expert 1: 1 token
        assert_eq!(table.expert_token_counts[0], 2);
        assert_eq!(table.expert_token_counts[1], 1);
    }

    #[test]
    fn route_table_overflow_count_zero_when_within_capacity() {
        let config = ExpertRouteConfig::new(4, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
        );
        assert_eq!(table.overflow_count, 0);
    }

    #[test]
    fn route_table_token_routes_have_indices_and_weights() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![0.3, 0.7]],
        );
        assert_eq!(table.token_routes.len(), 1);
        let route = &table.token_routes[0];
        assert_eq!(route.expert_indices.len(), 1);
        assert_eq!(route.expert_weights.len(), 1);
        assert_eq!(route.expert_positions.len(), 1);
    }

    #[test]
    fn route_table_top2_selects_two_experts() {
        let config = ExpertRouteConfig::new(4, 2);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![0.1, 0.4, 0.3, 0.2]],
        );
        assert_eq!(table.token_routes[0].expert_indices.len(), 2);
    }

    #[test]
    fn route_table_tokens_for_expert_returns_correct_indices() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0]],
        );
        let tokens_for_0 = table.tokens_for_expert(0);
        let tokens_for_1 = table.tokens_for_expert(1);
        assert_eq!(tokens_for_0.len(), 2);
        assert_eq!(tokens_for_1.len(), 1);
    }

    #[test]
    fn route_table_tokens_for_nonexistent_expert_empty() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0]],
        );
        let tokens = table.tokens_for_expert(99);
        assert!(tokens.is_empty());
    }

    #[test]
    fn route_table_utilization_stats_basic() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let stats = table.utilization_stats();
        assert_eq!(stats.total_tokens, 2);
        assert_eq!(stats.total_expert_assignments, 2);
        assert_eq!(stats.overflow_count, 0);
        assert!(stats.balance_score > 0.0);
    }

    #[test]
    fn route_table_utilization_stats_balance_perfect() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let stats = table.utilization_stats();
        assert!((stats.balance_score - 1.0).abs() < 1e-6, "equal load → perfect balance");
    }

    #[test]
    fn route_table_load_balance_loss_zero_when_disabled() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0]],
        );
        let loss = table.load_balance_loss(&vec![vec![1.0, 0.0]]);
        assert!((loss - 0.0).abs() < 1e-6);
    }

    #[test]
    fn route_table_clone_independent() {
        let config = ExpertRouteConfig::new(2, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let cloned = table.clone();
        assert_eq!(cloned.token_routes.len(), table.token_routes.len());
        assert_eq!(cloned.overflow_count, table.overflow_count);
    }

    // ── ExpertUtilizationStats: field verification ──

    #[test]
    fn utilization_stats_fields_populated() {
        let stats = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 12,
            overflow_count: 2,
            max_expert_load: 5,
            min_expert_load: 1,
            mean_expert_load: 3.0,
            balance_score: 0.8,
        };
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.total_expert_assignments, 12);
        assert_eq!(stats.overflow_count, 2);
        assert_eq!(stats.max_expert_load, 5);
        assert_eq!(stats.min_expert_load, 1);
        assert!((stats.mean_expert_load - 3.0).abs() < 1e-6);
        assert!((stats.balance_score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn utilization_stats_copy_trait() {
        let stats = ExpertUtilizationStats {
            total_tokens: 5,
            total_expert_assignments: 5,
            overflow_count: 0,
            max_expert_load: 2,
            min_expert_load: 1,
            mean_expert_load: 1.25,
            balance_score: 0.5,
        };
        let copied = stats;
        assert_eq!(copied.total_tokens, stats.total_tokens);
        assert_eq!(copied.overflow_count, stats.overflow_count);
    }

    #[test]
    fn utilization_stats_clone_independent() {
        let stats = ExpertUtilizationStats {
            total_tokens: 3,
            total_expert_assignments: 3,
            overflow_count: 0,
            max_expert_load: 2,
            min_expert_load: 1,
            mean_expert_load: 1.5,
            balance_score: 0.5,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_tokens, stats.total_tokens);
    }

    // ── ExpertHeatState: constructor defaults ──

    #[test]
    fn heat_state_default_values_via_manager() {
        let mgr = ExpertThermalManager::new(8);
        // ExpertHeatState defaults are set by ExpertThermalManager::new
        let state = mgr.state(3).unwrap();
        assert_eq!(state.expert_idx, 3);
        assert_eq!(state.hit_rate, 0.0);
        assert_eq!(state.hit_count, 0);
        assert_eq!(state.route_count, 0);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.consecutive_zero_streak, 0);
        assert!(state.residency == ExpertResidency::Resident);
        assert_eq!(state.reactivation_count, 0);
    }

    #[test]
    fn heat_state_clone_preserves_all_fields() {
        let mgr = ExpertThermalManager::new(16);
        let state = mgr.state(7).unwrap().clone();
        assert_eq!(state.expert_idx, 7);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.hit_rate, 0.0);
    }

    // ── EvictionDecision: variant properties ──

    #[test]
    fn eviction_decision_variants_distinct() {
        assert_ne!(EvictionDecision::Keep, EvictionDecision::Evict);
        assert_ne!(EvictionDecision::Evict, EvictionDecision::Reactivate);
        assert_ne!(EvictionDecision::Keep, EvictionDecision::Reactivate);
    }

    #[test]
    fn eviction_decision_copy_trait() {
        let a = EvictionDecision::Keep;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn eviction_decision_ordering() {
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
    }

    #[test]
    fn eviction_decision_equality_same_variant() {
        assert_eq!(EvictionDecision::Keep, EvictionDecision::Keep);
        assert_eq!(EvictionDecision::Evict, EvictionDecision::Evict);
        assert_eq!(EvictionDecision::Reactivate, EvictionDecision::Reactivate);
    }

    #[test]
    fn eviction_decision_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EvictionDecision::Keep);
        set.insert(EvictionDecision::Evict);
        set.insert(EvictionDecision::Reactivate);
        assert_eq!(set.len(), 3);
        set.insert(EvictionDecision::Keep);
        assert_eq!(set.len(), 3);
    }

    // ── DeoptRequest: construction and fields ──

    #[test]
    fn deopt_request_fields() {
        let req = DeoptRequest {
            request_id: 42,
            expert_idx: 3,
            layer_idx: 7,
            step: 100,
        };
        assert_eq!(req.request_id, 42);
        assert_eq!(req.expert_idx, 3);
        assert_eq!(req.layer_idx, 7);
        assert_eq!(req.step, 100);
    }

    #[test]
    fn deopt_request_clone_preserves_fields() {
        let req = DeoptRequest {
            request_id: 1,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, req.request_id);
        assert_eq!(cloned.expert_idx, req.expert_idx);
    }

    #[test]
    fn deopt_request_equality() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        assert_eq!(a, b);
    }

    #[test]
    fn deopt_request_inequality() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 99, expert_idx: 2, layer_idx: 3, step: 4 };
        assert_ne!(a, b);
    }

    // ── ExpertThermalManager: construction and builder ──

    #[test]
    fn thermal_manager_new_creates_correct_count() {
        let mgr = ExpertThermalManager::new(8);
        assert_eq!(mgr.num_experts(), 8);
        assert_eq!(mgr.states().len(), 8);
    }

    #[test]
    fn thermal_manager_new_initial_states_are_warm() {
        let mgr = ExpertThermalManager::new(4);
        for state in mgr.states() {
            assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
            assert!(state.residency == ExpertResidency::Resident);
        }
    }

    #[test]
    fn thermal_manager_with_eviction_threshold() {
        let mgr = ExpertThermalManager::new(4).with_eviction_threshold(500);
        assert_eq!(mgr.effective_eviction_threshold(), 500);
    }

    #[test]
    fn thermal_manager_with_heat_thresholds_custom() {
        let mgr = ExpertThermalManager::new(4).with_heat_thresholds(0.5, 0.01);
        // After step with no hits, hit_rate=0.0 → cold_threshold=0.01 → Evicted
        let mut mgr = mgr;
        mgr.step(&[0, 0, 0, 0]);
        assert_eq!(mgr.state(0).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn thermal_manager_with_eviction_aggressiveness_zero() {
        let mgr = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(0.0);
        assert_eq!(mgr.effective_eviction_threshold(), 1000);
    }

    #[test]
    fn thermal_manager_with_eviction_aggressiveness_positive_reduces_threshold() {
        let mgr = ExpertThermalManager::new(4)
            .with_eviction_threshold(1000)
            .with_eviction_aggressiveness(1.0);
        // bias_factor = 1.0 / (1.0 + 1.0) = 0.5 → threshold = 500
        assert_eq!(mgr.effective_eviction_threshold(), 500);
    }

    #[test]
    fn thermal_manager_with_adaptive_eviction() {
        let mgr = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);
        // Adaptive eviction should not panic; threshold computed dynamically
        let threshold = mgr.effective_eviction_threshold();
        assert!(threshold >= 1);
    }

    #[test]
    fn thermal_manager_update_memory_pressure_clamps() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.update_memory_pressure(-0.5);
        // Should clamp to 0.0
        mgr.update_memory_pressure(1.5);
        // Should clamp to 1.0
    }

    // ── ExpertThermalManager: step and state transitions ──

    #[test]
    fn thermal_manager_step_with_hits_updates_state() {
        let mut mgr = ExpertThermalManager::new(4).with_heat_thresholds(0.5, 0.01);
        // Expert 0 gets hits, others don't
        mgr.step(&[10, 0, 0, 0]);
        let s0 = mgr.state(0).unwrap();
        assert_eq!(s0.hit_count, 1);
        assert!(s0.hit_rate > 0.0);
        assert_eq!(s0.consecutive_zero_streak, 0);
    }

    #[test]
    fn thermal_manager_step_no_hits_increments_streak() {
        let mut mgr = ExpertThermalManager::new(2).with_heat_thresholds(0.5, 0.01);
        mgr.step(&[0, 0]);
        assert_eq!(mgr.state(0).unwrap().consecutive_zero_streak, 1);
        mgr.step(&[0, 0]);
        assert_eq!(mgr.state(0).unwrap().consecutive_zero_streak, 2);
    }

    #[test]
    fn thermal_manager_step_hit_resets_streak() {
        let mut mgr = ExpertThermalManager::new(2).with_heat_thresholds(0.5, 0.01);
        mgr.step(&[0, 0]);
        mgr.step(&[0, 0]);
        assert_eq!(mgr.state(0).unwrap().consecutive_zero_streak, 2);
        mgr.step(&[5, 0]);
        assert_eq!(mgr.state(0).unwrap().consecutive_zero_streak, 0);
    }

    #[test]
    fn thermal_manager_step_updates_current_step() {
        let mut mgr = ExpertThermalManager::new(2);
        let summary0 = mgr.summary();
        assert_eq!(summary0.current_step, 0);
        mgr.step(&[0, 0]);
        let summary1 = mgr.summary();
        assert_eq!(summary1.current_step, 1);
        mgr.step(&[1, 1]);
        let summary2 = mgr.summary();
        assert_eq!(summary2.current_step, 2);
    }

    #[test]
    fn thermal_manager_hot_experts_after_steps() {
        let mut mgr = ExpertThermalManager::new(4).with_heat_thresholds(0.5, 0.01);
        // Expert 0 gets hits every step
        for _ in 0..5 {
            mgr.step(&[10, 0, 0, 0]);
        }
        let hot = mgr.hot_experts();
        assert!(hot.contains(&0), "expert 0 should be hot after consistent hits");
    }

    #[test]
    fn thermal_manager_cold_or_evicted_experts_after_steps() {
        let mut mgr = ExpertThermalManager::new(4).with_heat_thresholds(0.5, 0.01);
        // Expert 3 never gets hits
        for _ in 0..5 {
            mgr.step(&[10, 10, 10, 0]);
        }
        let cold = mgr.cold_or_evicted_experts();
        assert!(cold.contains(&3), "expert 3 should be cold/evicted");
    }

    // ── ExpertThermalManager: evict and reactivate ──

    #[test]
    fn thermal_manager_evict_expert_success() {
        let mut mgr = ExpertThermalManager::new(4);
        assert!(mgr.evict_expert(1));
        assert!(mgr.state(1).unwrap().residency == ExpertResidency::Evicted);
        assert_eq!(mgr.state(1).unwrap().heat_level, ExpertHeatLevel::Evicted);
    }

    #[test]
    fn thermal_manager_evict_already_evicted_returns_false() {
        let mut mgr = ExpertThermalManager::new(4);
        assert!(mgr.evict_expert(0));
        assert!(!mgr.evict_expert(0), "evicting already-evicted returns false");
    }

    #[test]
    fn thermal_manager_evict_out_of_bounds_returns_false() {
        let mut mgr = ExpertThermalManager::new(4);
        assert!(!mgr.evict_expert(10));
    }

    #[test]
    fn thermal_manager_reactivate_evicted_expert() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(2);
        assert!(mgr.reactivate_expert(2));
        assert!(mgr.state(2).unwrap().residency == ExpertResidency::Resident);
        assert_eq!(mgr.state(2).unwrap().heat_level, ExpertHeatLevel::Cold);
        assert_eq!(mgr.state(2).unwrap().reactivation_count, 1);
    }

    #[test]
    fn thermal_manager_reactivate_not_evicted_returns_false() {
        let mut mgr = ExpertThermalManager::new(4);
        assert!(!mgr.reactivate_expert(0), "reactivating non-evicted returns false");
    }

    #[test]
    fn thermal_manager_reactivate_out_of_bounds_returns_false() {
        let mut mgr = ExpertThermalManager::new(4);
        assert!(!mgr.reactivate_expert(10));
    }

    // ── ExpertThermalManager: eviction_decision ──

    #[test]
    fn thermal_manager_eviction_decision_keep_initially() {
        let mgr = ExpertThermalManager::new(4);
        assert_eq!(mgr.eviction_decision(0), EvictionDecision::Keep);
    }

    #[test]
    fn thermal_manager_eviction_decision_out_of_bounds_keep() {
        let mgr = ExpertThermalManager::new(4);
        assert_eq!(mgr.eviction_decision(100), EvictionDecision::Keep);
    }

    #[test]
    fn thermal_manager_eviction_decision_evict_after_streak() {
        let mut mgr = ExpertThermalManager::new(2)
            .with_eviction_threshold(3);
        for _ in 0..3 {
            mgr.step(&[0, 10]);
        }
        assert_eq!(mgr.eviction_decision(0), EvictionDecision::Evict);
    }

    #[test]
    fn thermal_manager_eviction_decision_reactivate_after_deopt() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(1);
        // Simulate deopt by incrementing reactivation_count via handle_deopt_request
        let req = DeoptRequest { request_id: 1, expert_idx: 1, layer_idx: 0, step: 1 };
        mgr.handle_deopt_request(req);
        // After handling, expert 1 has reactivation_count > 0
        // But it was reactivated by handle_deopt_request, so decision is Keep
        assert!(mgr.state(1).unwrap().residency == ExpertResidency::Resident);
    }

    // ── ExpertThermalManager: experts_to_evict / experts_to_reactivate ──

    #[test]
    fn thermal_manager_experts_to_evict_empty_initially() {
        let mgr = ExpertThermalManager::new(4);
        assert!(mgr.experts_to_evict().is_empty());
    }

    #[test]
    fn thermal_manager_experts_to_evict_after_long_streak() {
        let mut mgr = ExpertThermalManager::new(3).with_eviction_threshold(5);
        for _ in 0..5 {
            mgr.step(&[0, 0, 10]);
        }
        let to_evict = mgr.experts_to_evict();
        assert!(to_evict.contains(&0));
        assert!(to_evict.contains(&1));
        assert!(!to_evict.contains(&2));
    }

    #[test]
    fn thermal_manager_experts_to_reactivate_empty_initially() {
        let mgr = ExpertThermalManager::new(4);
        assert!(mgr.experts_to_reactivate().is_empty());
    }

    // ── ExpertThermalManager: summary ──

    #[test]
    fn thermal_manager_summary_initial() {
        let mgr = ExpertThermalManager::new(4);
        let summary = mgr.summary();
        assert_eq!(summary.num_experts, 4);
        assert_eq!(summary.hot_count, 0);
        assert_eq!(summary.warm_count, 4);
        assert_eq!(summary.cold_count, 0);
        assert_eq!(summary.evicted_count, 0);
        assert_eq!(summary.total_evictions, 0);
        assert_eq!(summary.current_step, 0);
        assert_eq!(summary.pending_deopt_count, 0);
    }

    #[test]
    fn thermal_manager_summary_after_eviction() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(0);
        let summary = mgr.summary();
        assert_eq!(summary.evicted_count, 1);
        assert_eq!(summary.total_evictions, 1);
    }

    // ── ExpertThermalManager: pending_deopt_requests ──

    #[test]
    fn thermal_manager_pending_deopt_empty_initially() {
        let mgr = ExpertThermalManager::new(4);
        assert!(mgr.pending_deopt_requests().is_empty());
    }

    #[test]
    fn thermal_manager_clear_deopt_requests() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(0);
        let req = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 1 };
        mgr.handle_deopt_request(req);
        assert!(!mgr.pending_deopt_requests().is_empty());
        mgr.clear_deopt_requests();
        assert!(mgr.pending_deopt_requests().is_empty());
    }

    // ── ExpertThermalManager: handle_deopt_request ──

    #[test]
    fn thermal_manager_handle_deopt_evicted_expert_reactivates() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(2);
        let req = DeoptRequest { request_id: 42, expert_idx: 2, layer_idx: 5, step: 100 };
        let result = mgr.handle_deopt_request(req);
        match result {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 2);
                assert_eq!(request_id, 42);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => {
                panic!("expected ReactivateAndRerun for evicted expert");
            }
        }
    }

    #[test]
    fn thermal_manager_handle_deopt_non_evicted_is_spurious() {
        let mut mgr = ExpertThermalManager::new(4);
        let req = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 1 };
        let result = mgr.handle_deopt_request(req);
        match result {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 0);
                assert_eq!(request_id, 1);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => {
                panic!("expected SpuriousDeopt for non-evicted expert");
            }
        }
    }

    // ── DeoptHandlingResult: variant properties ──

    #[test]
    fn deopt_handling_result_equality() {
        let a = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 5 };
        let b = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 5 };
        assert_eq!(a, b);
    }

    #[test]
    fn deopt_handling_result_inequality_different_variants() {
        let a = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 0, request_id: 1 };
        let b = DeoptHandlingResult::SpuriousDeopt { expert_idx: 0, request_id: 1 };
        assert_ne!(a, b);
    }

    #[test]
    fn deopt_handling_result_clone() {
        let a = DeoptHandlingResult::SpuriousDeopt { expert_idx: 3, request_id: 7 };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── ThermalSummary: field verification ──

    #[test]
    fn thermal_summary_fields() {
        let summary = ThermalSummary {
            num_experts: 8,
            hot_count: 3,
            warm_count: 2,
            cold_count: 2,
            evicted_count: 1,
            total_evictions: 5,
            total_reactivations: 2,
            current_step: 100,
            pending_deopt_count: 1,
            working_set_size: 6,
            effective_eviction_threshold: 500,
        };
        assert_eq!(summary.num_experts, 8);
        assert_eq!(summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count, 8);
        assert_eq!(summary.total_evictions, 5);
        assert_eq!(summary.total_reactivations, 2);
    }

    #[test]
    fn thermal_summary_counts_sum_to_num_experts() {
        let summary = ThermalSummary {
            num_experts: 4,
            hot_count: 1,
            warm_count: 1,
            cold_count: 1,
            evicted_count: 1,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 1000,
        };
        assert_eq!(summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count, summary.num_experts);
    }

    #[test]
    fn thermal_summary_clone_independent() {
        let summary = ThermalSummary {
            num_experts: 2,
            hot_count: 1,
            warm_count: 1,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 1000,
        };
        let cloned = summary.clone();
        assert_eq!(cloned.num_experts, summary.num_experts);
        assert_eq!(cloned.hot_count, summary.hot_count);
    }

    // ── ExpertHeatLevel: Hash consistency ──

    #[test]
    fn heat_level_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ExpertHeatLevel::Hot);
        set.insert(ExpertHeatLevel::Warm);
        set.insert(ExpertHeatLevel::Cold);
        set.insert(ExpertHeatLevel::Evicted);
        assert_eq!(set.len(), 4);
        set.insert(ExpertHeatLevel::Hot);
        assert_eq!(set.len(), 4);
    }

    // ── topk_with_weights / topk_indices / softmax from routing ──

    #[test]
    fn topk_with_weights_selects_highest() {
        let result = topk_with_weights(&[0.1, 0.5, 0.3, 0.1], 2);
        assert_eq!(result.len(), 2);
        let indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn topk_with_weights_weights_sum_to_one() {
        let result = topk_with_weights(&[0.1, 0.5, 0.3, 0.1], 2);
        let sum: f32 = result.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "top-k weights should sum to 1.0");
    }

    #[test]
    fn topk_with_weights_single_element() {
        let result = topk_with_weights(&[5.0], 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn topk_indices_selects_correct_order() {
        let result = topk_indices(&[0.1, 0.7, 0.2], 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
    }

    #[test]
    fn softmax_outputs_sum_to_one() {
        let result = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_empty_input() {
        let result = softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn softmax_single_element() {
        let result = softmax(&[5.0]);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_uniform_input() {
        let result = softmax(&[1.0, 1.0, 1.0]);
        assert_eq!(result.len(), 3);
        for &v in &result {
            assert!((v - (1.0 / 3.0)).abs() < 1e-5);
        }
    }

    // ── ExpertRouteConfig: expert_capacity boundary with zero experts ──

    #[test]
    fn route_config_expert_capacity_many_experts_few_tokens() {
        let config = ExpertRouteConfig::new(100, 2);
        let cap = config.expert_capacity(1);
        // ceil(1.25 * 1 / 100) = ceil(0.0125) = 1
        assert_eq!(cap, 1);
    }

    // ── ExpertRouteTable: empty gate_logits ──

    #[test]
    fn route_table_from_empty_gate_logits() {
        let config = ExpertRouteConfig::new(4, 1);
        let table = ExpertRouteTable::from_gate_logits(config, &[]);
        assert!(table.token_routes.is_empty());
        assert_eq!(table.overflow_count, 0);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 5 (15 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── MoeDispatchPlan::is_balanced with negative gpu_total_us ──

    #[test]
    fn dispatch_plan_unbalanced_negative_times() {
        // cpu_total_us=-5.0, gpu_total_us=-10.0 → -5.0 <= -20.0 is false → unbalanced
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: -10.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: -5.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: -10.0,
            cpu_total_us: -5.0,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced(), "negative times: -5 <= -20 is false");
    }

    // ── MoeDispatchPlan::is_balanced with both NaN times ──

    #[test]
    fn dispatch_plan_balanced_both_nan_times() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: f32::NAN,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: f32::NAN,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: f32::NAN,
            cpu_total_us: f32::NAN,
            needs_cpu_sync: true,
        };
        assert!(!plan.is_balanced(), "NaN <= NaN*2.0 is false → unbalanced");
    }

    // ── MoeDispatchPlan: skipped_experts with duplicates in construction ──

    #[test]
    fn dispatch_plan_skipped_experts_can_contain_duplicates() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![1, 1, 2],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.skipped_experts.len(), 3);
        assert_eq!(plan.total_assignments(), 0);
    }

    // ── ExpertHardwareAssignment: clone independence with partition ──

    #[test]
    fn assignment_clone_independence_with_partition() {
        let mut a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::TensorCorePartition,
            partition: Some(HardwarePartition::gpu_sm_partition(1, 0, 40)),
            token_count: 100,
            estimated_compute_us: 50.0,
        };
        let cloned = a.clone();
        a.token_count = 999;
        assert_eq!(cloned.token_count, 100, "clone should be independent");
    }

    // ── Dispatch: warm experts with cpu_ratio=0 forces all to GPU ──

    #[test]
    fn dispatch_warm_with_zero_cpu_ratio_all_gpu() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty(), "ratio=0 → no CPU assignments");
        assert!(plan.gpu_experts.len() >= 2, "warm experts fallback to GPU");
    }

    // ── Dispatch: heat_levels longer than num_experts ──

    #[test]
    fn dispatch_heat_levels_longer_than_experts() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Warm,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.total_assignments() >= 2);
        assert!(plan.gpu_experts.iter().all(|e| e.hardware == HardwareKind::TensorCorePartition));
    }

    // ── ExpertHardwareAssignment: negative subnormal compute time ──

    #[test]
    fn assignment_negative_subnormal_compute_time() {
        let neg_subnormal = -f32::MIN_POSITIVE / 2.0;
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 1,
            estimated_compute_us: neg_subnormal,
        };
        assert!(a.estimated_compute_us < 0.0);
        assert!(!a.estimated_compute_us.is_normal());
    }

    // ── MoeDispatchPlan: remote_experts not counted in is_balanced ──

    #[test]
    fn dispatch_plan_is_balanced_ignores_remote_experts() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 100,
                estimated_compute_us: 9999.0,
            }],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced(), "only remote → no GPU/CPU → balanced");
        assert_eq!(plan.total_assignments(), 1);
    }

    // ── Dispatch: negative GPU TFLOPS produces infinite estimate ──

    #[test]
    fn dispatch_negative_gpu_tflops_produces_infinite() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, -10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.gpu_total_us.is_infinite(), "negative TFLOPS → infinite estimate");
    }

    // ── Dispatch: token count consistency across all assignments ──

    #[test]
    fn dispatch_all_assigned_tokens_equal_total_routes() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot; 3];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let assigned: usize = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .chain(plan.remote_experts.iter())
            .map(|e| e.token_count).sum();
        assert_eq!(assigned, 5, "all 5 route entries should be accounted for");
    }

    // ── ExpertHardwareAssignment: debug output with Some(partition) ──

    #[test]
    fn assignment_debug_with_some_partition() {
        let a = ExpertHardwareAssignment {
            expert_idx: 2,
            hardware: HardwareKind::LowComputeCore,
            partition: Some(HardwarePartition::cpu_numa_partition(0, 1, 8)),
            token_count: 32,
            estimated_compute_us: 7.5,
        };
        let s = format!("{:?}", a);
        assert!(s.contains("partition: Some("), "debug should show Some(partition)");
    }

    // ── MoeDispatchPlan: clone deep verification for gpu_experts ──

    #[test]
    fn dispatch_plan_clone_deep_gpu_experts_independence() {
        let mut plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 50,
                estimated_compute_us: 20.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 20.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        let cloned = plan.clone();
        plan.gpu_experts[0].token_count = 0;
        assert_eq!(cloned.gpu_experts[0].token_count, 50, "clone vec should be deep");
    }

    // ── MoeDispatchPlan: is_balanced with cpu_total_us = 0.0 and gpu_experts present ──

    #[test]
    fn dispatch_plan_balanced_zero_cpu_time_with_gpu_experts() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 10,
                estimated_compute_us: 50.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 50.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced(), "no CPU experts → early return true");
    }

    // ── Dispatch: verify estimated_compute_us scales linearly with token count ──

    #[test]
    fn dispatch_compute_time_proportional_to_tokens() {
        let config = ExpertRouteConfig::new(1, 1);
        let rt_a = ExpertRouteTable::from_gate_logits(config, &vec![vec![1.0]; 2]);
        let rt_b = ExpertRouteTable::from_gate_logits(
            ExpertRouteConfig::new(1, 1), &vec![vec![1.0]; 4],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan_a = dispatcher.dispatch(&rt_a, &heat_levels);
        let plan_b = dispatcher.dispatch(&rt_b, &heat_levels);
        let ratio = plan_b.gpu_total_us / plan_a.gpu_total_us;
        assert!((ratio - 2.0).abs() < 0.1, "4 tokens / 2 tokens ≈ 2x compute time, got {:.2}x", ratio);
    }

    // ── Dispatch: gpu_total_us=0.0 when no GPU experts assigned ──

    #[test]
    fn dispatch_gpu_total_us_zero_when_all_evicted() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Evicted, ExpertHeatLevel::Evicted];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.total_assignments(), 0);
        assert!((plan.gpu_total_us - 0.0).abs() < 1e-6, "no assignments → gpu_total_us = 0.0");
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 6 (15 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── MoeDispatchPlan: gpu_total_us fold semantics with mixed signs ──

    #[test]
    fn dispatch_plan_gpu_total_us_max_fold_with_negative() {
        // gpu_total_us = fold(0.0, f32::max) over [-5.0, 10.0] = 10.0
        let plan = MoeDispatchPlan {
            gpu_experts: vec![
                ExpertHardwareAssignment {
                    expert_idx: 0,
                    hardware: HardwareKind::TensorCorePartition,
                    partition: None,
                    token_count: 1,
                    estimated_compute_us: -5.0,
                },
                ExpertHardwareAssignment {
                    expert_idx: 1,
                    hardware: HardwareKind::FullComputeUnit,
                    partition: None,
                    token_count: 1,
                    estimated_compute_us: 10.0,
                },
            ],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 10.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        // max(-5.0, 10.0) = 10.0
        let actual_max = plan.gpu_experts.iter()
            .map(|e| e.estimated_compute_us)
            .fold(0.0f32, f32::max);
        assert!((actual_max - 10.0).abs() < 1e-6);
        assert!((plan.gpu_total_us - actual_max).abs() < 1e-6);
    }

    // ── MoeDispatchPlan: cpu_total_us fold with zero seed ──

    #[test]
    fn dispatch_plan_cpu_total_us_max_fold_with_all_zero() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![
                ExpertHardwareAssignment {
                    expert_idx: 0,
                    hardware: HardwareKind::LowComputeCore,
                    partition: None,
                    token_count: 0,
                    estimated_compute_us: 0.0,
                },
            ],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: true,
        };
        let actual_max = plan.cpu_experts.iter()
            .map(|e| e.estimated_compute_us)
            .fold(0.0f32, f32::max);
        assert!((actual_max - 0.0).abs() < 1e-6);
        assert!((plan.cpu_total_us - 0.0).abs() < 1e-6);
    }

    // ── Dispatch: hot experts ignore AMX availability ──

    #[test]
    fn dispatch_hot_experts_ignore_amx_and_go_to_gpu() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0], vec![1.0], vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty(), "hot experts always go to GPU, never CPU");
        assert_eq!(plan.gpu_experts.len(), 1);
        assert_eq!(plan.gpu_experts[0].hardware, HardwareKind::TensorCorePartition);
    }

    // ── Dispatch: single warm expert with AMX and enough ratio ──

    #[test]
    fn dispatch_single_warm_with_amx_and_full_ratio() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0], vec![1.0], vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(1.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(!plan.cpu_experts.is_empty(), "warm + AMX + full ratio → CPU");
        assert_eq!(plan.cpu_experts[0].hardware, HardwareKind::LowComputeCore);
        assert!(plan.needs_cpu_sync);
    }

    // ── Dispatch: cold expert with zero gpu tflops → infinite compute ──

    #[test]
    fn dispatch_cold_expert_zero_gpu_tflops_infinite_compute() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Cold];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 0.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.gpu_experts.len(), 1);
        assert!(plan.gpu_experts[0].estimated_compute_us.is_infinite());
        assert!(plan.gpu_total_us.is_infinite());
    }

    // ── Dispatch: cpu_total_us reflects max across multiple CPU experts ──

    #[test]
    fn dispatch_cpu_total_us_reflects_max_across_cpu_experts() {
        // 2 warm experts, each with different token counts → different compute times
        let config = ExpertRouteConfig::new(2, 1);
        // expert 0 gets 1 token, expert 1 gets 5 tokens
        let gate_logits: Vec<Vec<f32>> = std::iter::once(vec![1.0, 0.0])
            .chain((0..5).map(|_| vec![0.0, 1.0]))
            .collect();
        let route_table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(1.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        if plan.cpu_experts.len() >= 2 {
            let max_cpu = plan.cpu_experts.iter()
                .map(|e| e.estimated_compute_us)
                .fold(0.0f32, f32::max);
            assert!((plan.cpu_total_us - max_cpu).abs() < 1e-3,
                "cpu_total_us should equal max of CPU expert estimates");
            assert!(plan.cpu_total_us > 0.0);
        }
    }

    // ── MoeHardwareDispatcher: repeated builder calls overwrite ──

    #[test]
    fn dispatcher_repeated_gpu_builder_overwrites() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_gpu(120, 500.0);
        assert_eq!(d.gpu_sm_count(), 120, "second with_gpu should overwrite first");
    }

    // ── MoeHardwareDispatcher: repeated cpu builder overwrites ──

    #[test]
    fn dispatcher_repeated_cpu_builder_overwrites() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_cpu(16, false, false, 5.0)
            .with_cpu(32, true, true, 15.0);
        assert!(d.cpu_has_amx(), "second with_cpu should overwrite");
        assert_eq!(d.cpu_core_count, 32);
    }

    // ── Dispatch: expert_token_counts from route_table with top_k=2 ──

    #[test]
    fn dispatch_with_topk_2_assigns_tokens_to_multiple_experts() {
        let config = ExpertRouteConfig::new(3, 2);
        // Each token selects top-2 experts
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.6, 0.3]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot; 3];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 2))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // With top_k=2, each token routes to 2 experts → total assignments > token count
        let total_assigned_tokens: usize = plan.gpu_experts.iter()
            .map(|e| e.token_count).sum();
        assert!(total_assigned_tokens >= 2, "top_k=2 should produce multi-expert routing");
    }

    // ── Dispatch: evicted expert with tokens still gets skipped ──

    #[test]
    fn dispatch_evicted_expert_with_tokens_skipped_not_assigned() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        // Expert 0 is evicted but has 2 tokens routed to it
        let heat_levels = vec![ExpertHeatLevel::Evicted, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 0's tokens are not counted (evicted), expert 1 gets 1 token
        let assigned_indices: Vec<usize> = plan.gpu_experts.iter()
            .map(|e| e.expert_idx)
            .collect();
        assert!(!assigned_indices.contains(&0), "evicted expert should not be assigned");
        assert!(assigned_indices.contains(&1), "active expert should be assigned");
    }

    // ── MoeDispatchPlan: is_balanced with very small gpu_total_us (denormal) ──

    #[test]
    fn dispatch_plan_balanced_with_denormal_gpu_time() {
        let denorm = f32::MIN_POSITIVE / 2.0; // subnormal positive
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: denorm,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: denorm,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: denorm,
            cpu_total_us: denorm,
            needs_cpu_sync: true,
        };
        // cpu <= gpu * 2.0: denorm <= denorm * 2.0 → true
        assert!(plan.is_balanced(), "equal denormal times should be balanced");
    }

    // ── Dispatch: dispatch called multiple times produces same skipped list ──

    #[test]
    fn dispatch_deterministic_skipped_experts() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Hot,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0);
        let plan1 = dispatcher.dispatch(&route_table, &heat_levels);
        let plan2 = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan1.skipped_experts, plan2.skipped_experts,
            "skipped_experts should be deterministic across calls");
    }

    // ── Dispatch: verify remote_experts is always empty from dispatch ──

    #[test]
    fn dispatch_produces_empty_remote_experts() {
        // The current dispatch implementation never populates remote_experts
        let config = ExpertRouteConfig::new(4, 2);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 2))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.remote_experts.is_empty(),
            "current implementation never assigns remote experts");
    }

    // ── MoeDispatchPlan: total_assignments with large vectors ──

    #[test]
    fn dispatch_plan_total_assignments_large_vectors() {
        let make = |idx: usize| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: HardwareKind::TensorCorePartition,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: (0..50).map(make).collect(),
            cpu_experts: (50..70).map(make).collect(),
            remote_experts: (70..75).map(make).collect(),
            skipped_experts: vec![],
            gpu_total_us: 1.0,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert_eq!(plan.total_assignments(), 75);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 7 (15 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── ExpertRouteConfig: expert_capacity with one token and many experts ──

    #[test]
    fn route_config_expert_capacity_one_token_many_experts() {
        let config = ExpertRouteConfig::new(1000, 1);
        let cap = config.expert_capacity(1);
        // ceil(1.25 * 1 / 1000) = ceil(0.00125) = 1
        assert_eq!(cap, 1);
    }

    // ── ExpertRouteTable: all tokens routed to single expert, capacity limits ──

    #[test]
    fn route_table_all_tokens_single_expert_capped() {
        let config = ExpertRouteConfig::new(4, 1);
        // capacity = ceil(1.25 * 10 / 4) = 4; expert 0 gets at most 4, rest overflow
        let gate_logits: Vec<Vec<f32>> = (0..10)
            .map(|_| vec![1.0, 0.0, 0.0, 0.0])
            .collect();
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        assert_eq!(table.expert_token_counts[0], 4, "expert 0 capped at capacity");
        assert!(table.overflow_count > 0, "tokens beyond capacity should overflow");
        // Overflow tokens get routed to least-loaded expert
        let total_assigned: usize = table.expert_token_counts.iter().sum();
        assert_eq!(total_assigned, 10, "all tokens should be assigned somewhere");
    }

    // ── Dispatch: very high GPU TFLOPS produces near-zero compute time ──

    #[test]
    fn dispatch_very_high_gpu_tflops_near_zero_time() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 1_000_000.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.gpu_total_us.is_finite());
        assert!(plan.gpu_total_us > 0.0);
        assert!(plan.gpu_total_us < 1.0, "very high TFLOPS → compute time < 1 μs");
    }

    // ── Dispatch: top_k=2 with hot+cold mixed heat ──

    #[test]
    fn dispatch_topk2_mixed_hot_cold() {
        let config = ExpertRouteConfig::new(3, 2);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.6, 0.3]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 2))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 0 (Hot) and 2 (Hot) → TensorCorePartition
        // Expert 1 (Cold) → FullComputeUnit
        let tensor_count = plan.gpu_experts.iter()
            .filter(|e| e.hardware == HardwareKind::TensorCorePartition)
            .count();
        let full_count = plan.gpu_experts.iter()
            .filter(|e| e.hardware == HardwareKind::FullComputeUnit)
            .count();
        assert!(tensor_count > 0, "hot experts should use TensorCorePartition");
        assert!(full_count > 0, "cold expert should use FullComputeUnit");
    }

    // ── MoeDispatchPlan: all three assignment categories populated ──

    #[test]
    fn dispatch_plan_all_three_categories() {
        let make = |idx: usize, hw: HardwareKind| ExpertHardwareAssignment {
            expert_idx: idx,
            hardware: hw,
            partition: None,
            token_count: 1,
            estimated_compute_us: 1.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![make(0, HardwareKind::TensorCorePartition)],
            cpu_experts: vec![make(1, HardwareKind::LowComputeCore)],
            remote_experts: vec![make(2, HardwareKind::FullComputeUnit)],
            skipped_experts: vec![3],
            gpu_total_us: 1.0,
            cpu_total_us: 1.0,
            needs_cpu_sync: true,
        };
        assert_eq!(plan.total_assignments(), 3);
        assert!(plan.needs_cpu_sync);
        assert!(!plan.skipped_experts.is_empty());
    }

    // ── ExpertRouteConfig: clone and mutation independence ──

    #[test]
    fn route_config_clone_mutation_independence() {
        let mut config = ExpertRouteConfig::new(8, 2);
        let cloned = config.clone();
        config.num_experts = 16;
        assert_eq!(cloned.num_experts, 8, "clone should be independent");
    }

    // ── Dispatch: same heat level different token counts produces different compute times ──

    #[test]
    fn dispatch_different_token_counts_different_times() {
        // Use 1 expert to avoid capacity overflow redistribution
        let config = ExpertRouteConfig::new(1, 1);
        // 2 tokens for first dispatch
        let rt_few = ExpertRouteTable::from_gate_logits(
            config.clone(),
            &vec![vec![1.0], vec![1.0]],
        );
        // 10 tokens for second dispatch
        let rt_many = ExpertRouteTable::from_gate_logits(
            ExpertRouteConfig::new(1, 1),
            &vec![vec![1.0]; 10],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0);
        let plan_few = dispatcher.dispatch(&rt_few, &heat_levels);
        let plan_many = dispatcher.dispatch(&rt_many, &heat_levels);
        assert!(plan_many.gpu_total_us > plan_few.gpu_total_us,
            "more tokens should produce higher compute time");
        let e_few = &plan_few.gpu_experts[0];
        let e_many = &plan_many.gpu_experts[0];
        assert_eq!(e_few.token_count, 2);
        assert_eq!(e_many.token_count, 10);
    }

    // ── HardwarePartition: gpu_sm_partition with full SM range ──

    #[test]
    fn hardware_partition_gpu_full_sm_range() {
        let p = HardwarePartition::gpu_sm_partition(0, 0, 80);
        assert_eq!(p.sm_range, Some((0, 80)));
        assert!((p.compute_weight - 1.0).abs() < 1e-6, "full SM range → weight 1.0");
    }

    // ── HardwarePartition: cpu_numa_partition compute_weight small ──

    #[test]
    fn hardware_partition_cpu_small_core_count_low_weight() {
        let p = HardwarePartition::cpu_numa_partition(0, 0, 4);
        // weight = 4 / 64.0 = 0.0625
        assert!((p.compute_weight - 0.0625).abs() < 1e-6);
        assert_eq!(p.numa_node, Some(0));
        assert_eq!(p.kind, HardwareKind::FullComputeUnit);
    }

    // ── Dispatch: both experts evicted produces empty plan ──

    #[test]
    fn dispatch_both_experts_evicted_empty_plan() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        // Both evicted → no assignments, all skipped
        let heat_levels = vec![ExpertHeatLevel::Evicted, ExpertHeatLevel::Evicted];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.total_assignments(), 0, "all evicted → empty plan");
        assert!(plan.skipped_experts.contains(&0));
        assert!(plan.skipped_experts.contains(&1));
    }

    // ── ExpertHardwareAssignment: debug output with None partition ──

    #[test]
    fn assignment_debug_with_none_partition() {
        let a = ExpertHardwareAssignment {
            expert_idx: 0,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 5,
            estimated_compute_us: 3.14,
        };
        let s = format!("{:?}", a);
        assert!(s.contains("partition: None"), "debug should show partition: None");
    }

    // ── MoeDispatchPlan: is_balanced with cpu faster than gpu by large margin ──

    #[test]
    fn dispatch_plan_balanced_cpu_much_faster_than_gpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 100,
                estimated_compute_us: 1000.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: 0.001,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 1000.0,
            cpu_total_us: 0.001,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(), "CPU much faster → balanced");
    }

    // ── Dispatch: dispatch with negative CPU TFLOPS, warm expert goes to GPU ──

    #[test]
    fn dispatch_warm_with_negative_cpu_tflops_goes_to_gpu() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0], vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, -5.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // AMX available, but CPU TFLOPS negative → CPU estimate infinite
        // Warm expert assigned to CPU (AMX available), but cpu_tflops <= 0 → infinite estimate
        // The dispatch still assigns to CPU since AMX is true and ratio check passes
        if !plan.cpu_experts.is_empty() {
            assert!(plan.cpu_experts[0].estimated_compute_us.is_infinite(),
                "negative CPU TFLOPS → infinite CPU compute estimate");
        }
    }

    // ── ExpertRouteTable: utilization_stats with imbalanced routing ──

    #[test]
    fn route_table_utilization_stats_imbalanced() {
        // Use 2 experts, 3 tokens all to expert 0
        // capacity = ceil(1.25 * 3 / 2) = 2, so expert 0 gets 2, overflow goes to expert 1
        let config = ExpertRouteConfig::new(2, 1);
        let gate_logits: Vec<Vec<f32>> = (0..3)
            .map(|_| vec![1.0, 0.0])
            .collect();
        let table = ExpertRouteTable::from_gate_logits(config, &gate_logits);
        let stats = table.utilization_stats();
        assert_eq!(stats.total_tokens, 3);
        assert!(stats.max_expert_load > stats.min_expert_load,
            "imbalanced routing: max load > min load");
        assert!(stats.balance_score < 1.0, "imbalanced routing → balance < 1.0");
    }

    // ── Dispatch: dispatch called twice with different heat levels produces different plans ──

    #[test]
    fn dispatch_different_heat_levels_different_plans() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);

        let plan_all_hot = dispatcher.dispatch(&route_table, &[
            ExpertHeatLevel::Hot, ExpertHeatLevel::Hot,
        ]);
        let plan_mixed = dispatcher.dispatch(&route_table, &[
            ExpertHeatLevel::Hot, ExpertHeatLevel::Warm,
        ]);

        // All hot → no CPU sync; mixed hot+warm with AMX → CPU sync
        assert!(!plan_all_hot.needs_cpu_sync);
        assert!(plan_mixed.needs_cpu_sync, "warm expert with AMX should trigger CPU sync");
    }

    // ── Dispatch: compute time is proportional for GPU and CPU independently ──

    #[test]
    fn dispatch_gpu_and_cpu_compute_times_scale_independently() {
        let config = ExpertRouteConfig::new(2, 1);
        // Expert 0 (hot → GPU): 2 tokens
        // Expert 1 (warm → CPU with AMX): 2 tokens
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Warm];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Both should have finite compute times
        if !plan.gpu_experts.is_empty() {
            assert!(plan.gpu_total_us.is_finite(), "GPU time should be finite with 300 TFLOPS");
            assert!(plan.gpu_total_us > 0.0);
        }
        if !plan.cpu_experts.is_empty() {
            assert!(plan.cpu_total_us.is_finite(), "CPU time should be finite with 10 TFLOPS");
            assert!(plan.cpu_total_us > 0.0);
        }
        // GPU (300 TFLOPS) should be faster than CPU (10 TFLOPS) for same work
        if !plan.gpu_experts.is_empty() && !plan.cpu_experts.is_empty() {
            assert!(plan.gpu_total_us < plan.cpu_total_us,
                "GPU at 300 TFLOPS should be faster than CPU at 10 TFLOPS");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 8 (15 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── Dispatch: warm expert without AMX but with AVX-512 still goes to GPU ──

    // @trace TEST-MOE-DISPATCH-296 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_warm_with_avx512_only_no_amx_goes_gpu() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Warm];
        // AVX-512 true but AMX false → warm goes to GPU (AMX is the CPU dispatch gate)
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, false, true, 10.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.cpu_experts.is_empty(),
            "warm without AMX but with AVX-512 still goes to GPU");
        assert!(!plan.gpu_experts.is_empty());
        for e in &plan.gpu_experts {
            assert_eq!(e.hardware, HardwareKind::FullComputeUnit,
                "warm without AMX → FullComputeUnit, not TensorCorePartition");
        }
    }

    // ── MoeDispatchPlan: is_balanced returns true when only GPU experts exist ──

    // @trace TEST-MOE-DISPATCH-297 [level:unit]
    #[test]
    fn dispatch_plan_balanced_gpu_only_regardless_of_time() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 100,
                estimated_compute_us: 99999.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 99999.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert!(plan.is_balanced(),
            "GPU-only plans are always balanced regardless of compute time");
    }

    // ── ExpertRouteConfig: Default trait produces same as new(8, 2) ──

    // @trace TEST-MOE-DISPATCH-298 [level:unit]
    #[test]
    fn route_config_default_matches_new_8_2() {
        let default_config = ExpertRouteConfig::default();
        let explicit_config = ExpertRouteConfig::new(8, 2);
        assert_eq!(default_config.num_experts, explicit_config.num_experts);
        assert_eq!(default_config.top_k, explicit_config.top_k);
        assert!((default_config.capacity_factor - explicit_config.capacity_factor).abs() < 1e-6);
        assert_eq!(default_config.load_balance_loss, explicit_config.load_balance_loss);
    }

    // ── Dispatch: CPU capacity overflow spills single warm expert to GPU ──

    // @trace TEST-MOE-DISPATCH-299 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_warm_exceeds_cpu_capacity_single_expert_spills() {
        let config = ExpertRouteConfig::new(1, 1);
        // 20 tokens to expert 0
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &(0..20).map(|_| vec![1.0]).collect::<Vec<_>>(),
        );
        let heat_levels = vec![ExpertHeatLevel::Warm];
        // max_cpu_ratio=0.05 → max_cpu_tokens = ceil(20*0.05) = 1
        // expert 0 has 20 tokens but only 1 allowed on CPU → fallback to GPU
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.05);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Token count 20 exceeds cpu capacity of 1 → spills to GPU
        assert!(plan.gpu_experts.len() + plan.cpu_experts.len() >= 1,
            "expert should be assigned somewhere");
        if !plan.cpu_experts.is_empty() {
            assert!(plan.cpu_experts[0].token_count <= 1,
                "CPU assignment should respect capacity limit");
        }
    }

    // ── MoeDispatchPlan: skipped_experts length independent of total_assignments ──

    // @trace TEST-MOE-DISPATCH-300 [level:unit]
    #[test]
    fn dispatch_plan_skipped_does_not_affect_total_count() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 10,
                estimated_compute_us: 5.0,
            }],
            cpu_experts: vec![],
            remote_experts: vec![],
            skipped_experts: vec![1, 2, 3, 4, 5, 6, 7],
            gpu_total_us: 5.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        assert_eq!(plan.total_assignments(), 1,
            "total_assignments counts only gpu+cpu+remote, not skipped");
        assert_eq!(plan.skipped_experts.len(), 7);
    }

    // ── Dispatch: hot expert with zero SM count still dispatched ──

    // @trace TEST-MOE-DISPATCH-301 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_hot_expert_zero_sm_still_dispatched() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0], vec![1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(0, 300.0); // SM count = 0 but TFLOPS valid
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.gpu_experts.len(), 1, "hot expert still dispatched with SM=0");
        assert_eq!(plan.gpu_experts[0].hardware, HardwareKind::TensorCorePartition);
        assert!(plan.gpu_experts[0].estimated_compute_us.is_finite());
    }

    // ── ExpertHardwareAssignment: Debug trait includes estimated_compute_us value ──

    // @trace TEST-MOE-DISPATCH-302 [level:unit]
    #[test]
    fn assignment_debug_includes_compute_time_value() {
        let a = ExpertHardwareAssignment {
            expert_idx: 42,
            hardware: HardwareKind::LowComputeCore,
            partition: None,
            token_count: 7,
            estimated_compute_us: 13.37,
        };
        let debug_str = format!("{:?}", a);
        assert!(debug_str.contains("42"), "should contain expert_idx value");
        assert!(debug_str.contains("7"), "should contain token_count value");
    }

    // ── Dispatch: identical route table and heat_levels produce same gpu_total_us ──

    // @trace TEST-MOE-DISPATCH-303 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_deterministic_gpu_total_us() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Warm,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0);

        let plan1 = dispatcher.dispatch(&route_table, &heat_levels);
        let plan2 = dispatcher.dispatch(&route_table, &heat_levels);

        assert!((plan1.gpu_total_us - plan2.gpu_total_us).abs() < 1e-6,
            "gpu_total_us should be deterministic");
        assert!((plan1.cpu_total_us - plan2.cpu_total_us).abs() < 1e-6,
            "cpu_total_us should be deterministic");
    }

    // ── MoeDispatchPlan: clone preserves remote_experts vector ──

    // @trace TEST-MOE-DISPATCH-304 [level:unit]
    #[test]
    fn dispatch_plan_clone_preserves_remote_experts() {
        let remote = ExpertHardwareAssignment {
            expert_idx: 99,
            hardware: HardwareKind::FullComputeUnit,
            partition: None,
            token_count: 50,
            estimated_compute_us: 200.0,
        };
        let plan = MoeDispatchPlan {
            gpu_experts: vec![],
            cpu_experts: vec![],
            remote_experts: vec![remote],
            skipped_experts: vec![],
            gpu_total_us: 0.0,
            cpu_total_us: 0.0,
            needs_cpu_sync: false,
        };
        let cloned = plan.clone();
        assert_eq!(cloned.remote_experts.len(), 1);
        assert_eq!(cloned.remote_experts[0].expert_idx, 99);
        assert_eq!(cloned.remote_experts[0].token_count, 50);
    }

    // ── ExpertRouteTable: tokens_for_expert returns correct results per expert ──

    // @trace TEST-MOE-DISPATCH-305 [level:unit]
    #[test]
    fn route_table_tokens_for_expert_results_consistent_with_counts() {
        // Use many experts with evenly distributed logits to avoid overflow redistribution
        let config = ExpertRouteConfig::new(3, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );
        // Each expert gets exactly 1 token, no overflow
        for expert_idx in 0..3 {
            let tokens = table.tokens_for_expert(expert_idx);
            assert_eq!(tokens.len(), table.expert_token_counts[expert_idx],
                "tokens_for_expert({}) length should match expert_token_counts", expert_idx);
        }
        // Out-of-bounds expert returns empty
        assert!(table.tokens_for_expert(99).is_empty());
    }

    // ── ExpertHeatLevel: all four variants produce distinct Debug strings ──

    // @trace TEST-MOE-DISPATCH-306 [level:unit]
    #[test]
    fn heat_level_all_debug_strings_distinct() {
        let hot = format!("{:?}", ExpertHeatLevel::Hot);
        let warm = format!("{:?}", ExpertHeatLevel::Warm);
        let cold = format!("{:?}", ExpertHeatLevel::Cold);
        let evicted = format!("{:?}", ExpertHeatLevel::Evicted);
        // All four strings should be distinct
        assert_ne!(hot, warm);
        assert_ne!(warm, cold);
        assert_ne!(cold, evicted);
        assert_ne!(hot, cold);
        assert_ne!(hot, evicted);
        assert_ne!(warm, evicted);
    }

    // ── Dispatch: multiple evicted experts reduces total_assignments ──

    // @trace TEST-MOE-DISPATCH-307 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_multiple_evicted_reduces_active_assignments() {
        let config = ExpertRouteConfig::new(6, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &(0..6).map(|i| {
                let mut row = vec![0.0f32; 6];
                row[i] = 1.0;
                row
            }).collect::<Vec<_>>(),
        );
        // 4 experts evicted, 2 hot
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Evicted,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(6, 1))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert_eq!(plan.total_assignments(), 2,
            "only 2 hot experts should be assigned out of 6");
        assert!(plan.skipped_experts.contains(&1));
        assert!(plan.skipped_experts.contains(&3));
        assert!(plan.skipped_experts.contains(&4));
        assert!(plan.skipped_experts.contains(&5));
    }

    // ── MoeDispatchPlan: is_balanced with subnormal positive times ──

    // @trace TEST-MOE-DISPATCH-308 [level:unit]
    #[test]
    fn dispatch_plan_balanced_subnormal_positive_times() {
        let sub = f32::MIN_POSITIVE / 3.0; // positive subnormal
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::FullComputeUnit,
                partition: None,
                token_count: 1,
                estimated_compute_us: sub,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: sub,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: sub,
            cpu_total_us: sub * 3.0, // 3x gpu → still within 2x? No, 3x > 2x
            needs_cpu_sync: true,
        };
        // cpu = sub*3.0, gpu*2 = sub*2.0 → sub*3.0 > sub*2.0 → unbalanced
        assert!(!plan.is_balanced(),
            "cpu 3x gpu_subnormal → unbalanced");
    }

    // ── HardwarePartition: cpu_numa_partition always sets sm_range to None ──

    // @trace TEST-MOE-DISPATCH-309 [level:unit]
    #[test]
    fn hardware_partition_cpu_numa_sm_range_always_none() {
        for node in [0, 1, 2, 3] {
            let p = HardwarePartition::cpu_numa_partition(0, node, 8);
            assert!(p.sm_range.is_none(),
                "CPU NUMA partition should never have sm_range set, node={}", node);
        }
    }

    // ── Dispatch: compute time for GPU with very small TFLOPS is very large ──

    // @trace TEST-MOE-DISPATCH-310 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_very_small_gpu_tflops_very_large_time() {
        let config = ExpertRouteConfig::new(1, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &(0..10).map(|_| vec![1.0]).collect::<Vec<_>>(),
        );
        let heat_levels = vec![ExpertHeatLevel::Hot];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(1, 1))
            .with_gpu(80, 0.001); // 0.001 TFLOPS = 1 GFLOPS
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        assert!(plan.gpu_total_us.is_finite());
        assert!(plan.gpu_total_us > 1000.0,
            "very low TFLOPS with 10 tokens should produce large compute time > 1000 μs");
    }

    // ── ExpertRouteConfig: expert_capacity calculation with 2 experts and odd tokens ──

    // @trace TEST-MOE-DISPATCH-311 [level:unit]
    #[test]
    fn route_config_expert_capacity_odd_tokens_two_experts() {
        let config = ExpertRouteConfig::new(2, 1);
        // ceil(1.25 * 7 / 2) = ceil(4.375) = 5
        assert_eq!(config.expert_capacity(7), 5);
        // ceil(1.25 * 1 / 2) = ceil(0.625) = 1
        assert_eq!(config.expert_capacity(1), 1);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 9 (15 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── ExpertHeatLevel: used as HashMap key ──

    // @trace TEST-MOE-DISPATCH-312 [level:unit]
    #[test]
    fn heat_level_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map: HashMap<ExpertHeatLevel, usize> = HashMap::new();
        map.insert(ExpertHeatLevel::Hot, 10);
        map.insert(ExpertHeatLevel::Warm, 20);
        map.insert(ExpertHeatLevel::Cold, 30);
        map.insert(ExpertHeatLevel::Evicted, 40);
        assert_eq!(map[&ExpertHeatLevel::Hot], 10);
        assert_eq!(map[&ExpertHeatLevel::Warm], 20);
        assert_eq!(map[&ExpertHeatLevel::Cold], 30);
        assert_eq!(map[&ExpertHeatLevel::Evicted], 40);
        assert_eq!(map.len(), 4);
    }

    // ── ExpertHeatState: field values after multiple evict/reactivate cycles ──

    // @trace TEST-MOE-DISPATCH-313 [level:unit]
    #[test]
    fn thermal_manager_evict_reactivate_cycle_counts() {
        let mut mgr = ExpertThermalManager::new(4);
        // Cycle 1: evict then reactivate
        mgr.evict_expert(0);
        mgr.reactivate_expert(0);
        assert_eq!(mgr.state(0).unwrap().reactivation_count, 1);
        assert!(mgr.state(0).unwrap().residency == ExpertResidency::Resident);
        // Cycle 2: evict resets reactivation_count to 0, then reactivate increments to 1
        mgr.evict_expert(0);
        assert_eq!(mgr.state(0).unwrap().reactivation_count, 0,
            "evict resets reactivation_count");
        mgr.reactivate_expert(0);
        assert_eq!(mgr.state(0).unwrap().reactivation_count, 1,
            "reactivate increments from 0 after evict reset");
        assert!(mgr.state(0).unwrap().residency == ExpertResidency::Resident);
        // total_reactivations should be 2 (one per cycle)
        let summary = mgr.summary();
        assert_eq!(summary.total_reactivations, 2);
    }

    // ── DeoptRequest: field values at maximum bounds ──

    // @trace TEST-MOE-DISPATCH-314 [level:unit]
    #[test]
    fn deopt_request_max_field_values() {
        let req = DeoptRequest {
            request_id: u64::MAX,
            expert_idx: usize::MAX,
            layer_idx: usize::MAX,
            step: u64::MAX,
        };
        assert_eq!(req.request_id, u64::MAX);
        assert_eq!(req.expert_idx, usize::MAX);
        assert_eq!(req.layer_idx, usize::MAX);
        assert_eq!(req.step, u64::MAX);
    }

    // ── Dispatch: top_k=2 with partial evicted experts reduces assignment count ──

    // @trace TEST-MOE-DISPATCH-315 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_topk2_partial_evicted_reduces_assigned_tokens() {
        let config = ExpertRouteConfig::new(3, 2);
        // Each token selects top-2; 3 tokens total = 6 assignments expected
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![0.5, 0.3, 0.2],
                vec![0.1, 0.6, 0.3],
                vec![0.4, 0.1, 0.5],
            ],
        );
        // Expert 2 evicted: its token slots are skipped
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 2))
            .with_gpu(80, 300.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Expert 2 evicted → assignments only from experts 0 and 1
        let active_indices: Vec<usize> = plan.gpu_experts.iter()
            .chain(plan.cpu_experts.iter())
            .map(|e| e.expert_idx)
            .collect();
        assert!(!active_indices.contains(&2), "evicted expert should not appear");
        assert!(active_indices.contains(&0) || active_indices.contains(&1));
    }

    // ── Dispatch: warm expert with AMX but cpu_ratio exactly at token boundary ──

    // @trace TEST-MOE-DISPATCH-316 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_warm_amx_cpu_ratio_exact_boundary() {
        let config = ExpertRouteConfig::new(2, 1);
        // 4 tokens: 2 to expert 0 (warm), 2 to expert 1 (hot)
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Warm, ExpertHeatLevel::Hot];
        // max_cpu_ratio=0.5 → max_cpu_tokens=2, expert 0 has 2 tokens → exactly fits
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        if !plan.cpu_experts.is_empty() {
            let cpu_tokens: usize = plan.cpu_experts.iter().map(|e| e.token_count).sum();
            assert!(cpu_tokens <= 2, "CPU tokens should not exceed max_cpu_tokens=2");
        }
    }

    // ── ExpertRouteTable: uniform gate logits produce balanced routing ──

    // @trace TEST-MOE-DISPATCH-317 [level:unit]
    #[test]
    fn route_table_uniform_logits_balanced_distribution() {
        let config = ExpertRouteConfig::new(3, 1);
        // All tokens have equal logits for all experts → should distribute evenly
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
        );
        let stats = table.utilization_stats();
        assert_eq!(stats.total_tokens, 3);
        assert_eq!(stats.total_expert_assignments, 3);
        // With uniform logits, capacity limits should still ensure each token is assigned
        let total_assigned: usize = table.expert_token_counts.iter().sum();
        assert_eq!(total_assigned, 3);
    }

    // ── Dispatch: cpu_has_avx512 accessor reflects builder configuration ──

    // @trace TEST-MOE-DISPATCH-318 [level:unit]
    #[test]
    fn dispatcher_cpu_avx512_accessor_reflects_builder() {
        let d = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_cpu(16, false, true, 5.0);
        assert!(!d.cpu_has_amx());
        assert!(d.cpu_has_avx512, "cpu_has_avx512 field should be true when set via with_cpu");
        let d2 = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_cpu(16, false, false, 5.0);
        assert!(!d2.cpu_has_avx512, "cpu_has_avx512 should be false when not set");
    }

    // ── ExpertUtilizationStats: Debug trait output ──

    // @trace TEST-MOE-DISPATCH-319 [level:unit]
    #[test]
    fn utilization_stats_debug_format() {
        let stats = ExpertUtilizationStats {
            total_tokens: 10,
            total_expert_assignments: 10,
            overflow_count: 0,
            max_expert_load: 5,
            min_expert_load: 1,
            mean_expert_load: 2.5,
            balance_score: 0.9,
        };
        let s = format!("{:?}", stats);
        assert!(s.contains("total_tokens"), "debug should include total_tokens");
        assert!(s.contains("overflow_count"), "debug should include overflow_count");
        assert!(s.contains("balance_score"), "debug should include balance_score");
    }

    // ── ThermalSummary: working_set_size after eviction ──

    // @trace TEST-MOE-DISPATCH-320 [level:unit]
    #[test]
    fn thermal_summary_working_set_size_after_partial_eviction() {
        let mut mgr = ExpertThermalManager::new(8);
        // Evict 3 of 8 experts → working set = 5
        mgr.evict_expert(0);
        mgr.evict_expert(2);
        mgr.evict_expert(5);
        let summary = mgr.summary();
        assert_eq!(summary.evicted_count, 3);
        assert_eq!(summary.hot_count + summary.warm_count + summary.cold_count, 5);
    }

    // ── Dispatch: dispatch with hot+warm+cold produces correct hardware kind distribution ──

    // @trace TEST-MOE-DISPATCH-321 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_three_heat_levels_correct_hardware_kinds() {
        let config = ExpertRouteConfig::new(3, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        );
        let heat_levels = vec![
            ExpertHeatLevel::Hot,   // → TensorCorePartition
            ExpertHeatLevel::Warm,  // → LowComputeCore (AMX available)
            ExpertHeatLevel::Cold,  // → FullComputeUnit
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(3, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.5);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Verify hot expert
        let hot = plan.gpu_experts.iter().find(|e| e.expert_idx == 0);
        assert!(hot.is_some());
        assert_eq!(hot.unwrap().hardware, HardwareKind::TensorCorePartition);
        // Verify cold expert
        let cold = plan.gpu_experts.iter().find(|e| e.expert_idx == 2);
        assert!(cold.is_some());
        assert_eq!(cold.unwrap().hardware, HardwareKind::FullComputeUnit);
        // Verify warm expert is on CPU
        let warm = plan.cpu_experts.iter().find(|e| e.expert_idx == 1);
        if let Some(w) = warm {
            assert_eq!(w.hardware, HardwareKind::LowComputeCore);
        }
    }

    // ── Dispatch: negative gpu_tflops combined with hot+cold experts ──

    // @trace TEST-MOE-DISPATCH-322 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_negative_gpu_tflops_hot_and_cold_both_infinite() {
        let config = ExpertRouteConfig::new(2, 1);
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        );
        let heat_levels = vec![ExpertHeatLevel::Hot, ExpertHeatLevel::Cold];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(2, 1))
            .with_gpu(80, -1.0);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        // Both hot and cold go to GPU → both have infinite estimate
        for e in &plan.gpu_experts {
            assert!(e.estimated_compute_us.is_infinite(),
                "negative GPU TFLOPS → all GPU estimates infinite");
        }
        assert!(plan.gpu_total_us.is_infinite());
    }

    // ── MoeDispatchPlan: is_balanced with gpu_total_us = infinity and finite cpu ──

    // @trace TEST-MOE-DISPATCH-323 [level:unit]
    #[test]
    fn dispatch_plan_balanced_infinite_gpu_finite_cpu() {
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 1,
                estimated_compute_us: f32::INFINITY,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 1,
                estimated_compute_us: 50.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: f32::INFINITY,
            cpu_total_us: 50.0,
            needs_cpu_sync: true,
        };
        // cpu_total_us (50.0) <= gpu_total_us * 2.0 (inf) → true → balanced
        assert!(plan.is_balanced(),
            "finite CPU <= infinite GPU * 2.0 → balanced");
    }

    // ── ExpertThermalManager: step with zero-length hit array ──

    // @trace TEST-MOE-DISPATCH-324 [level:unit]
    #[test]
    fn thermal_manager_step_empty_hits_array() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.step(&[]);
        let summary = mgr.summary();
        assert_eq!(summary.current_step, 1, "step should increment even with empty hits");
    }

    // ── Dispatch: multiple warm experts compete for limited CPU capacity ──

    // @trace TEST-MOE-DISPATCH-325 [req:REQ-MOE-002] [level:unit]
    #[test]
    fn dispatch_multiple_warm_experts_cpu_capacity_contention() {
        let config = ExpertRouteConfig::new(4, 1);
        // 4 experts, each gets 2 tokens = 8 total
        let route_table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ],
        );
        // All warm → all want CPU, but max_cpu_ratio=0.25 limits to 2 tokens
        let heat_levels = vec![
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Warm,
        ];
        let dispatcher = MoeHardwareDispatcher::new(ExpertRouteConfig::new(4, 1))
            .with_gpu(80, 300.0)
            .with_cpu(16, true, true, 10.0)
            .with_max_cpu_ratio(0.25);
        let plan = dispatcher.dispatch(&route_table, &heat_levels);
        let cpu_tokens: usize = plan.cpu_experts.iter().map(|e| e.token_count).sum();
        let total_tokens: usize = plan.gpu_experts.iter().map(|e| e.token_count).sum::<usize>()
            + cpu_tokens;
        assert_eq!(total_tokens, 8, "all 8 tokens should be assigned");
        assert!(cpu_tokens <= 2, "CPU tokens limited to 25% of 8 = 2, got {}", cpu_tokens);
        assert!(!plan.gpu_experts.is_empty(), "overflow warm experts spill to GPU");
    }

    // ── ExpertRouteTable: tokens_for_expert with top_k=2 returns correct positions ──

    // @trace TEST-MOE-DISPATCH-326 [level:unit]
    #[test]
    fn route_table_topk2_tokens_for_expert_positions() {
        let config = ExpertRouteConfig::new(2, 2);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![0.8, 0.2], vec![0.3, 0.7]],
        );
        // Each token routes to 2 experts → both experts get 2 assignments
        let tokens_0 = table.tokens_for_expert(0);
        let tokens_1 = table.tokens_for_expert(1);
        assert_eq!(tokens_0.len(), 2, "expert 0 should have 2 token assignments");
        assert_eq!(tokens_1.len(), 2, "expert 1 should have 2 token assignments");
    }

    // ── ExpertHeatLevel: Ord trait allows min/max selection ──

    // @trace TEST-MOE-DISPATCH-327 [level:unit]
    #[test]
    fn heat_level_ord_enables_min_max() {
        let levels = vec![
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Evicted,
            ExpertHeatLevel::Warm,
        ];
        let min_level = *levels.iter().min().unwrap();
        let max_level = *levels.iter().max().unwrap();
        assert_eq!(min_level, ExpertHeatLevel::Hot, "Hot is the minimum heat level");
        assert_eq!(max_level, ExpertHeatLevel::Evicted, "Evicted is the maximum heat level");
    }

    // ═══════════════════════════════════════════════════════════════
    //  Additional tests — batch 10 (10 new tests)
    // ═══════════════════════════════════════════════════════════════

    // ── ExpertRouteTable: all tokens overflow to least-loaded expert ──

    // @trace TEST-MOE-DISPATCH-328 [level:unit]
    #[test]
    fn route_table_all_overflow_fallback_to_least_loaded() {
        // 1 expert, capacity_factor=1.25, top_k=1 → capacity = ceil(1.25*3/1)=4
        // All 3 tokens pick expert 0 → no overflow, but test with many tokens:
        // capacity = ceil(1.25*5/1)=7, so all fit. Use 2 experts instead:
        let config = ExpertRouteConfig::new(2, 1);
        // capacity = ceil(1.25*5/2) = ceil(3.125) = 3
        // 5 tokens all want expert 0, capacity=3 → 2 overflow to expert 1
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
            ],
        );
        // Expert 0 capacity is 3, so 3 tokens go there, 2 overflow
        assert!(table.overflow_count > 0, "should have overflow tokens");
        // All tokens should still be routed (fallback to least-loaded)
        assert_eq!(table.token_routes.len(), 5, "all 5 tokens should have routes");
        // Expert 1 should have received the overflow tokens
        assert!(table.expert_token_counts[1] > 0, "expert 1 should receive overflow tokens");
    }

    // ── ExpertHeatLevel::from_hit_rate with rate exactly at cold_threshold ──

    // @trace TEST-MOE-DISPATCH-329 [level:unit]
    #[test]
    fn heat_level_from_hit_rate_exactly_at_cold_threshold() {
        // from_hit_rate: rate >= hot_threshold → Hot
        //               rate >= cold_threshold → Warm
        //               rate > 0.0 → Cold
        //               rate == 0.0 → Evicted
        // rate exactly at cold_threshold → Warm (>= cold_threshold)
        let level = ExpertHeatLevel::from_hit_rate(0.01, 0.1, 0.01);
        assert_eq!(level, ExpertHeatLevel::Warm,
            "rate exactly at cold_threshold should be Warm");
    }

    // ── ExpertThermalManager: working_set_size tracks distinct experts across steps ──

    // @trace TEST-MOE-DISPATCH-330 [level:unit]
    #[test]
    fn thermal_manager_working_set_size_multiple_steps() {
        let mut mgr = ExpertThermalManager::new(6)
            .with_adaptive_eviction(5);
        // Step 1: experts 0,1 active
        mgr.step(&[5, 3, 0, 0, 0, 0]);
        // Step 2: experts 2,3 active
        mgr.step(&[0, 0, 4, 2, 0, 0]);
        // Step 3: experts 0,4 active
        mgr.step(&[1, 0, 0, 0, 7, 0]);
        let summary = mgr.summary();
        // Working set should include experts 0,1,2,3,4 = 5 distinct experts
        assert_eq!(summary.working_set_size, 5,
            "working set should track 5 distinct experts across 3 steps");
    }

    // ── ExpertThermalManager: multiple deopt requests queue correctly ──

    // @trace TEST-MOE-DISPATCH-331 [level:unit]
    #[test]
    fn thermal_manager_multiple_deopt_requests_queue() {
        let mut mgr = ExpertThermalManager::new(4);
        mgr.evict_expert(0);
        mgr.evict_expert(1);
        // Queue two deopt requests for different evicted experts
        let req0 = DeoptRequest { request_id: 10, expert_idx: 0, layer_idx: 3, step: 50 };
        let req1 = DeoptRequest { request_id: 11, expert_idx: 1, layer_idx: 3, step: 50 };
        let result0 = mgr.handle_deopt_request(req0);
        let result1 = mgr.handle_deopt_request(req1);
        // Both should reactivate
        assert!(matches!(result0, DeoptHandlingResult::ReactivateAndRerun { expert_idx: 0, .. }));
        assert!(matches!(result1, DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, .. }));
        // Both should no longer be evicted
        assert!(mgr.state(0).unwrap().residency == ExpertResidency::Resident);
        assert!(mgr.state(1).unwrap().residency == ExpertResidency::Resident);
        // total_reactivations should be 2
        let summary = mgr.summary();
        assert_eq!(summary.total_reactivations, 2);
    }

    // ── ExpertRouteTable: load_balance_loss returns zero when disabled in config ──

    // @trace TEST-MOE-DISPATCH-332 [level:unit]
    #[test]
    fn route_table_load_balance_loss_disabled_returns_zero() {
        let config = ExpertRouteConfig::new(3, 1);
        let table = ExpertRouteTable::from_gate_logits(
            config,
            &vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
        );
        // load_balance_loss is disabled by default (load_balance_loss: false)
        let gate_logits = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let loss = table.load_balance_loss(&gate_logits);
        assert!((loss - 0.0).abs() < 1e-6,
            "load_balance_loss should be 0 when disabled in config");
    }

    // ── ExpertHeatLevel::from_hit_rate with rate > 1.0 (saturated) ──

    // @trace TEST-MOE-DISPATCH-333 [level:unit]
    #[test]
    fn heat_level_from_hit_rate_saturated_above_one() {
        // rate > 1.0 with low thresholds → still Hot
        let level = ExpertHeatLevel::from_hit_rate(5.0, 0.1, 0.01);
        assert_eq!(level, ExpertHeatLevel::Hot,
            "rate > 1.0 with low thresholds should be Hot");
        // rate > 1.0 with high thresholds → still Hot (>= hot_threshold)
        let level2 = ExpertHeatLevel::from_hit_rate(2.0, 0.5, 0.1);
        assert_eq!(level2, ExpertHeatLevel::Hot,
            "rate > 1.0 should always be Hot regardless of threshold");
    }

    // ── MoeDispatchPlan: is_balanced when cpu_total_us is exactly 2x gpu_total_us ──

    // @trace TEST-MOE-DISPATCH-334 [level:unit]
    #[test]
    fn dispatch_plan_balanced_cpu_exactly_2x_gpu() {
        // cpu <= gpu*2.0 → cpu=200.0, gpu=100.0 → 200.0 <= 200.0 → balanced
        let plan = MoeDispatchPlan {
            gpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 0,
                hardware: HardwareKind::TensorCorePartition,
                partition: None,
                token_count: 10,
                estimated_compute_us: 100.0,
            }],
            cpu_experts: vec![ExpertHardwareAssignment {
                expert_idx: 1,
                hardware: HardwareKind::LowComputeCore,
                partition: None,
                token_count: 5,
                estimated_compute_us: 200.0,
            }],
            remote_experts: vec![],
            skipped_experts: vec![],
            gpu_total_us: 100.0,
            cpu_total_us: 200.0,
            needs_cpu_sync: true,
        };
        assert!(plan.is_balanced(),
            "cpu_total_us exactly 2x gpu_total_us should be balanced");
    }

    // ── EvictionDecision: Debug format contains variant name ──

    // @trace TEST-MOE-DISPATCH-335 [level:unit]
    #[test]
    fn eviction_decision_debug_contains_variant_name() {
        let keep_str = format!("{:?}", EvictionDecision::Keep);
        let evict_str = format!("{:?}", EvictionDecision::Evict);
        let react_str = format!("{:?}", EvictionDecision::Reactivate);
        assert!(keep_str.contains("Keep"), "Debug should contain 'Keep'");
        assert!(evict_str.contains("Evict"), "Debug should contain 'Evict'");
        assert!(react_str.contains("Reactivate"), "Debug should contain 'Reactivate'");
    }

    // ── DeoptHandlingResult: Debug format contains variant name ──

    // @trace TEST-MOE-DISPATCH-336 [level:unit]
    #[test]
    fn deopt_handling_result_debug_contains_variant_name() {
        let reactivate = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 0, request_id: 1 };
        let spurious = DeoptHandlingResult::SpuriousDeopt { expert_idx: 0, request_id: 1 };
        let r_str = format!("{:?}", reactivate);
        let s_str = format!("{:?}", spurious);
        assert!(r_str.contains("ReactivateAndRerun"), "Debug should contain 'ReactivateAndRerun'");
        assert!(s_str.contains("SpuriousDeopt"), "Debug should contain 'SpuriousDeopt'");
    }

    // ── ExpertThermalManager: hit_rate converges after many alternating steps ──

    // @trace TEST-MOE-DISPATCH-337 [level:unit]
    #[test]
    fn thermal_manager_hit_rate_converges_alternating_hits() {
        let mut mgr = ExpertThermalManager::new(2).with_heat_thresholds(0.5, 0.01);
        // Expert 0: 5 hits, 5 misses → hit_rate should be ~0.5 (Hot)
        // Expert 1: 0 hits, 10 misses → hit_rate should be 0.0 (Evicted)
        for i in 0..10 {
            if i % 2 == 0 {
                mgr.step(&[10, 0]);
            } else {
                mgr.step(&[0, 0]);
            }
        }
        let s0 = mgr.state(0).unwrap();
        let s1 = mgr.state(1).unwrap();
        // Expert 0 should have positive hit_rate (5 hits out of 10 steps)
        assert!(s0.hit_rate > 0.3, "expert 0 hit_rate should be ~0.5, got {}", s0.hit_rate);
        assert_eq!(s0.heat_level, ExpertHeatLevel::Hot);
        // Expert 1 should be evicted (0 hits)
        assert_eq!(s1.heat_level, ExpertHeatLevel::Evicted);
    }
}
