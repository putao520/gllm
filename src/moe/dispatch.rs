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
    pub fn dispatch(
        &self,
        route_table: &ExpertRouteTable,
        heat_levels: &[ExpertHeatLevel],
    ) -> MoeDispatchPlan {
        let mut gpu_experts = Vec::new();
        let mut cpu_experts = Vec::new();
        let remote_experts = Vec::new();

        // 统计每个专家的 token 数
        let mut expert_token_counts = vec![0usize; self.config.num_experts];
        for route in &route_table.token_routes {
            for &expert_idx in &route.expert_indices {
                if expert_idx < expert_token_counts.len() {
                    expert_token_counts[expert_idx] += 1;
                }
            }
        }

        let total_tokens: usize = expert_token_counts.iter().sum();
        let max_cpu_tokens = (total_tokens as f32 * self.max_cpu_ratio) as usize;
        let mut cpu_tokens_used = 0usize;

        for (expert_idx, &token_count) in expert_token_counts.iter().enumerate() {
            if token_count == 0 {
                continue;
            }

            let heat = heat_levels.get(expert_idx).copied().unwrap_or(ExpertHeatLevel::Warm);

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
        let route_table = make_route_table(&config);
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
}
