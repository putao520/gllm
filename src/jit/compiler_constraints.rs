//! IR 约束变量体系 (SPEC §12.6)
//!
//! 将硬件探测结果严格坍缩为 JIT 编译器的强数学约束变量组。
//! 所有加载期的硬件探测结果必须通过此模块转化为 CompilerGraph 可直接消费的环境变量，
//! 确保 JIT 逻辑与物理实体芯片解耦。
//!
//! ## 核心法则
//!
//! 废弃散乱的指令集条件判断。
//! 所有硬件探测结果必须严格坍缩为对底层 JIT 编译器的**强数学约束变量组**。

use crate::sensors::{GpuPlatform, GpuTopology, MemoryNetworkSensors};
use super::profiler::ProbeResult;

// ── IR 约束变量 (§12.6 Target Execution Topology) ──

/// JIT 编译器约束变量组
///
/// 传感器数据转化为 CompilerGraph 直接可消费的环境变量。
/// §12.6 明确定义了这些变量及其典型值。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompilerConstraints {
    // ── 寄存器与平铺约束 ──

    /// 控制寄存器溢写（Spill）阈值
    ///
    /// 普通 CPU = 15 (x86_64 GPR), APX = 31 (扩展 GPR)
    pub max_gpr_count: usize,

    /// 决定 JIT 平铺展开的二维尺寸 (tile_m × tile_n)
    ///
    /// 启用 AMX/SME 阵列时极速扩大，否则受限于 SIMD 宽度
    pub optimal_tile_bits: TileBits,

    /// 是否可通过 VNNI/SVE2 直接下发硬件 INT4 解包
    ///
    /// true = 有 VNNI_INT8 / SVE2 dot 支持
    pub native_int4_dot: bool,

    // ── 缓存约束 ──

    /// L1 指令缓存大小 (bytes)
    ///
    /// 防止大 Bucket 的指令平铺导致 L1i 缓存颠簸。
    /// 触达 l1i_size * 0.8 时 JIT 退化为 jmp
    pub l1i_size: usize,

    /// L2 缓存大小 (bytes)
    ///
    /// GPU 分级驻留锚点
    pub l2_cache_size: usize,

    /// 共享内存大小 (bytes, GPU only)
    ///
    /// None for CPU backends
    pub smem_size: Option<usize>,

    // ── 拓扑约束 ──

    /// NUMA 节点数 (for §12.1 CPU Sub-Batch partitioning)
    pub numa_node_count: usize,

    /// GPU SM 版本 (e.g. 70, 80, 90, 100)
    ///
    /// None for CPU-only backends
    pub gpu_sm_version: Option<u32>,

    /// GPU SM 数量 (for §12.1 Sub-Batch partitioning)
    pub gpu_sm_count: Option<usize>,

    /// Warp/Wavefront 大小 (GPU)
    pub gpu_warp_size: Option<usize>,

    /// NUMA 节点到核心范围的绑定映射
    /// Vec of (node_id, start_core, end_core)
    pub numa_core_bindings: Vec<(usize, usize, usize)>,

    // ── 向量约束 ──

    /// SIMD 向量宽度 (bits)
    pub simd_width_bits: usize,

    /// 是否支持 AMX 矩阵运算 (x86)
    pub has_amx: bool,

    /// 是否支持 AVX-512
    pub has_avx512: bool,

    /// 是否支持 SVE (ARM)
    pub has_sve: bool,

    /// SVE 向量长度 (bytes, ARM only)
    pub sve_vl_bytes: Option<usize>,

    // ── 猜测量化约束 (§12.5) ──

    /// 是否支持 TMA (Tensor Memory Accelerator, Hopper SM90+)
    pub has_tma: bool,

    /// 是否支持 FP4 原生运算 (Blackwell SM100+)
    pub has_native_fp4: bool,

    /// 是否支持 FP6 原生运算
    pub has_native_fp6: bool,

    // ── RDMA 融合约束 (§12.6) ──

    /// RDMA Pipelining 的最小 Chunk 大小 (tokens)
    ///
    /// 满足约束: T_compute(chunk) >= T_rdma_transfer(chunk)
    pub rdma_min_chunk_tokens: Option<usize>,

    /// GPU Tensor Core 代数 (0=无, 2=wmma, 3=hma)
    pub tensor_core_gen: u32,
}

/// JIT 平铺尺寸参数
///
/// 决定 GEMM/Attention 平铺的 tile_m × tile_n 二维尺寸。
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct TileBits {
    /// tile_m 维度 (行方向)
    pub tile_m: usize,
    /// tile_n 维度 (列方向)
    pub tile_n: usize,
    /// tile_k (归约维度)
    pub tile_k: usize,
}

impl Default for CompilerConstraints {
    fn default() -> Self {
        Self {
            max_gpr_count: 15,
            optimal_tile_bits: TileBits::default(),
            native_int4_dot: false,
            l1i_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            smem_size: None,
            numa_node_count: 1,
            gpu_sm_version: None,
            gpu_sm_count: None,
            gpu_warp_size: None,
            numa_core_bindings: Vec::new(),
            simd_width_bits: 256,
            has_amx: false,
            has_avx512: false,
            has_sve: false,
            sve_vl_bytes: None,
            has_tma: false,
            has_native_fp4: false,
            has_native_fp6: false,
            rdma_min_chunk_tokens: None,
            tensor_core_gen: 0,
        }
    }
}

impl Default for TileBits {
    fn default() -> Self {
        Self {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
        }
    }
}

impl CompilerConstraints {
    /// 从 DeviceProfile 和硬件传感器数据推导 IR 约束变量
    ///
    /// §12.6 核心法则：废弃散乱的指令集条件判断，
    /// 所有硬件探测结果必须坍缩为此结构体。
    ///
    /// `gpu_topology` (SPEC/02-HARDWARE.md §2.2 Phase 4) 驱动 GPU 相关约束
    /// (SM 版本/数量、共享内存、L2 容量、warp_size、Tensor Core 代数、
    ///  TMA/FP4/FP6 特性)。当为 `None` 时系统为 CPU-only。
    pub fn derive(
        profile: &gllm_kernels::dispatch::DeviceProfile,
        sensors: &MemoryNetworkSensors,
        probe_result: Option<&ProbeResult>,
        gpu_topology: Option<&GpuTopology>,
    ) -> Self {
        let mut constraints = Self::default();

        let kc = &profile.kernel_config;

        // ── SIMD / 向量宽度 ──
        constraints.simd_width_bits = profile.simd_width_bytes() * 8; // bytes → bits
        constraints.has_amx = kc.has_amx;
        constraints.has_avx512 = kc.use_avx512;
        constraints.has_sve = kc.has_sve;
        constraints.sve_vl_bytes = if kc.has_sve {
            Some(kc.sve_vl_bytes.max(16))
        } else {
            None
        };

        // ── 寄存器约束 ──
        // APX (x86 扩展 GPR): 31 个通用寄存器
        constraints.max_gpr_count = if kc.has_amx || kc.use_avx512 {
            31 // APX implied
        } else {
            15
        };

        // ── 平铺约束 ──
        if kc.has_amx {
            // AMX: 1024-bit 阵列, 极大平铺
            constraints.optimal_tile_bits = TileBits {
                tile_m: 64,
                tile_n: 64,
                tile_k: 64,
            };
        } else if kc.use_avx512 {
            constraints.optimal_tile_bits = TileBits {
                tile_m: 32,
                tile_n: 32,
                tile_k: 32,
            };
        } else if kc.has_sve {
            let vl = kc.sve_vl_bytes.max(16);
            constraints.optimal_tile_bits = TileBits {
                tile_m: vl * 8 / 32, // 按 float 数
                tile_n: vl * 8 / 32,
                tile_k: 32,
            };
        } else {
            // AVX2 / NEON: 基础平铺
            constraints.optimal_tile_bits = TileBits {
                tile_m: 16,
                tile_n: 16,
                tile_k: 16,
            };
        };

        // ── INT4 硬件支持 ──
        // VNNI_INT8 / SVE2 dot product
        constraints.native_int4_dot = kc.has_amx || kc.use_avx512;

        // ── 缓存约束 ──
        let (l1d, l2, l3) = profile.cache_sizes();
        // L1i ≈ L1d on most architectures
        constraints.l1i_size = l1d;
        constraints.l2_cache_size = sensors.l2_cache_bytes.max(l2);
        constraints.numa_node_count = sensors.ccx_numa_topology
            .as_ref()
            .map(|t| t.nodes.len())
            .unwrap_or(1);

        // ── GPU 约束 (SPEC/02-HARDWARE.md §2.2 Phase 4) ──
        //
        // GpuTopology 由 `sensors::gpu::detect_gpu()` 在加载期一次性探测。
        // `None` 表示 CPU-only 系统（所有 GPU 字段清零）。
        if let Some(gpu) = gpu_topology {
            constraints.gpu_sm_count = Some(gpu.compute_unit_count);
            constraints.gpu_warp_size = Some(gpu.warp_size);
            constraints.smem_size = Some(gpu.shared_mem_per_sm_bytes);
            constraints.tensor_core_gen = gpu.tensor_core_gen;

            // L2 cache: GPU L2 优先于 CPU L2（gllm-kernels Compiler 在 GPU codegen
            // 时消费 l2_cache_size 作为分块锚点）。仅当 GPU 侧提供有效值时覆盖。
            if gpu.l2_bytes > 0 {
                constraints.l2_cache_size = gpu.l2_bytes;
            }

            // sm_version 仅在 CUDA 平台有意义（HIP 使用 gfx_arch，Metal 使用 gpu_family）。
            constraints.gpu_sm_version = match gpu.platform {
                GpuPlatform::Cuda { sm_version } => Some(sm_version),
                // HIP/Metal 不使用 SM 版本；这里返回 None 让 codegen 走各自的 platform 路径。
                GpuPlatform::Hip { .. } | GpuPlatform::Metal { .. } => None,
            };

            // ── Hopper/Blackwell 特性（仅 NVIDIA）──
            let sm = match gpu.platform {
                GpuPlatform::Cuda { sm_version } => sm_version,
                _ => 0,
            };
            // TMA (Tensor Memory Accelerator): Hopper SM90+
            constraints.has_tma = sm >= 90;
            // FP4 原生: Blackwell SM100+
            constraints.has_native_fp4 = sm >= 100;
            // FP6 原生: Blackwell SM100+ (与 FP4 同代)
            constraints.has_native_fp6 = sm >= 100;
        } else {
            constraints.gpu_sm_version = None;
            constraints.gpu_sm_count = None;
            constraints.gpu_warp_size = None;
            constraints.smem_size = None;
            constraints.tensor_core_gen = 0;
            constraints.has_tma = false;
            constraints.has_native_fp4 = false;
            constraints.has_native_fp6 = false;
        }

        // ── RDMA 约束 ──
        if let Some(_probe) = probe_result {
            // Probe 的 spill points 影响 tile_k 选择
            // 当 spill point 较小时, 说明寄存器紧张, 适当减小平铺
        }
        constraints.rdma_min_chunk_tokens = sensors.min_chunk_size_for_rdma(
            100.0,
        );

        constraints
    }

    /// 检查给定 seq_len 的 GEMM kernel 是否会超出 L1i 缓存
 ///
    /// §12.6: 触达 l1i_size * 0.8 时 JIT 退化为 jmp
    pub fn exceeds_l1i_budget(&self, kernel_code_bytes: usize) -> bool {
        kernel_code_bytes > (self.l1i_size as f64 * 0.8) as usize
    }

    /// 获取 GPU SM 分区大小 (用于 §12.1 Sub-Batch 空间分片)
    ///
    /// 将 SM 按等分划分为 n 个分区
    pub fn gpu_sm_partition(&self, num_partitions: usize) -> GpuSmPartition {
        let total_sm = self.gpu_sm_count.unwrap_or(1);
        let sm_per_partition = total_sm / num_partitions.max(1);

        GpuSmPartition {
            total_sm,
            num_partitions,
            sm_per_partition,
        }
    }

    /// 获取 NUMA 核绑定映射 (用于 §12.1 CPU 端异构)
    ///
    /// 返回 (node_id, core_range) 列表
    pub fn numa_core_bindings(&self) -> Vec<(usize, std::ops::Range<usize>)> {
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cores_per_node = if self.numa_node_count > 0 {
            total_cores / self.numa_node_count
        } else {
            total_cores
        };

        (0..self.numa_node_count)
            .map(|node_id| {
                let start = node_id * cores_per_node;
                let end = start + cores_per_node;
                (node_id, start..end)
            })
            .collect()
    }
}

/// GPU SM 分区信息
#[derive(Debug, Clone, Copy)]
pub struct GpuSmPartition {
    /// 总 SM 数量
    pub total_sm: usize,
    /// 分区数量
    pub num_partitions: usize,
    /// 每个分区的 SM 数量
    pub sm_per_partition: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_constraints_default() {
        let c = CompilerConstraints::default();
        assert_eq!(c.max_gpr_count, 15);
        assert!(!c.native_int4_dot);
        assert!(!c.has_tma);
    }

    #[test]
    fn test_compiler_constraints_derive() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let constraints = CompilerConstraints::derive(&profile, &sensors, None, None);

        // 基本合理性检查
        assert!(constraints.simd_width_bits > 0);
        assert!(constraints.l2_cache_size > 0);
        assert!(constraints.l1i_size > 0);
        // GPU 字段：无 GPU topology 必须全部为 None / 0
        assert_eq!(constraints.gpu_sm_count, None);
        assert_eq!(constraints.gpu_warp_size, None);
        assert_eq!(constraints.smem_size, None);
        assert_eq!(constraints.tensor_core_gen, 0);
        assert!(!constraints.has_tma);
        assert!(!constraints.has_native_fp4);
        assert!(!constraints.has_native_fp6);
    }

    #[test]
    fn test_compiler_constraints_derive_with_cuda_hopper() {
        // 模拟 Hopper (SM90) GPU topology
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 228 * 1024,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));

        assert_eq!(c.gpu_sm_version, Some(90));
        assert_eq!(c.gpu_sm_count, Some(132));
        assert_eq!(c.gpu_warp_size, Some(32));
        assert_eq!(c.smem_size, Some(228 * 1024));
        assert_eq!(c.tensor_core_gen, 3);
        assert!(c.has_tma, "Hopper SM90 must have TMA");
        assert!(!c.has_native_fp4, "Hopper SM90 does not have native FP4");
        assert_eq!(c.l2_cache_size, 50 * 1024 * 1024);
    }

    #[test]
    fn test_compiler_constraints_derive_with_cuda_blackwell() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 100 },
            compute_unit_count: 192,
            tensor_core_gen: 4,
            shared_mem_per_sm_bytes: 256 * 1024,
            l2_bytes: 96 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 10,
            compute_cap_minor: 0,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));

        assert_eq!(c.gpu_sm_version, Some(100));
        assert_eq!(c.tensor_core_gen, 4);
        assert!(c.has_tma);
        assert!(c.has_native_fp4, "Blackwell SM100 must have native FP4");
        assert!(c.has_native_fp6, "Blackwell SM100 must have native FP6");
    }

    #[test]
    fn test_compiler_constraints_derive_with_hip() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x942 },
            compute_unit_count: 228,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x42,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));

        assert_eq!(c.gpu_sm_version, None, "HIP does not use SM version");
        assert_eq!(c.gpu_sm_count, Some(228));
        assert_eq!(c.gpu_warp_size, Some(64));
        assert_eq!(c.tensor_core_gen, 3);
        // Hopper-only 特性在 HIP 路径上必须 false
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
    }

    #[test]
    fn test_l1i_budget_check() {
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        // 32KB L1i, 80% 阈值 = 25600 bytes
        assert!(c.exceeds_l1i_budget(30000));
        assert!(!c.exceeds_l1i_budget(20000));
    }

    #[test]
    fn test_gpu_sm_partition() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            ..Default::default()
        };
        let partition = c.gpu_sm_partition(3);
        assert_eq!(partition.total_sm, 80);
        assert_eq!(partition.num_partitions, 3);
        assert_eq!(partition.sm_per_partition, 26);
    }
}
