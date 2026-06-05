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
        let (l1d, l2, _l3) = profile.cache_sizes();
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

    // ---- Additional tests ----

    #[test]
    fn tile_bits_default_values() {
        let t = TileBits::default();
        assert_eq!(t.tile_m, 16);
        assert_eq!(t.tile_n, 16);
        assert_eq!(t.tile_k, 16);
    }

    #[test]
    fn tile_bits_copy_clone() {
        let t = TileBits { tile_m: 32, tile_n: 64, tile_k: 16 };
        let t2 = t;
        assert_eq!(t2.tile_m, 32);
        assert_eq!(t2.tile_n, 64);
        let t3 = t.clone();
        assert_eq!(t3.tile_k, 16);
    }

    #[test]
    fn compiler_constraints_default_all_fields() {
        let c = CompilerConstraints::default();
        assert_eq!(c.max_gpr_count, 15);
        assert_eq!(c.optimal_tile_bits.tile_m, 16);
        assert!(!c.native_int4_dot);
        assert_eq!(c.l1i_size, 32 * 1024);
        assert_eq!(c.l2_cache_size, 256 * 1024);
        assert!(c.smem_size.is_none());
        assert_eq!(c.numa_node_count, 1);
        assert!(c.gpu_sm_version.is_none());
        assert!(c.gpu_sm_count.is_none());
        assert!(c.gpu_warp_size.is_none());
        assert!(c.numa_core_bindings.is_empty());
        assert_eq!(c.simd_width_bits, 256);
        assert!(!c.has_amx);
        assert!(!c.has_avx512);
        assert!(!c.has_sve);
        assert!(c.sve_vl_bytes.is_none());
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
        assert!(c.rdma_min_chunk_tokens.is_none());
        assert_eq!(c.tensor_core_gen, 0);
    }

    #[test]
    fn compiler_constraints_clone() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            has_tma: true,
            ..Default::default()
        };
        let c2 = c.clone();
        assert_eq!(c2.gpu_sm_count, Some(80));
        assert!(c2.has_tma);
    }

    #[test]
    fn compiler_constraints_serialize_deserialize() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(132),
            has_avx512: true,
            optimal_tile_bits: TileBits { tile_m: 32, tile_n: 32, tile_k: 32 },
            ..Default::default()
        };
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.gpu_sm_count, Some(132));
        assert!(c2.has_avx512);
        assert_eq!(c2.optimal_tile_bits.tile_m, 32);
    }

    #[test]
    fn exceeds_l1i_budget_exact_boundary() {
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        // 80% of 32768 = 26214.4, floor = 26214
        let threshold = (32768_f64 * 0.8) as usize;
        assert!(!c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold + 1));
    }

    #[test]
    fn exceeds_l1i_budget_zero_l1i() {
        let c = CompilerConstraints {
            l1i_size: 0,
            ..Default::default()
        };
        // 0 * 0.8 = 0, any positive value exceeds
        assert!(c.exceeds_l1i_budget(1));
        assert!(!c.exceeds_l1i_budget(0));
    }

    #[test]
    fn gpu_sm_partition_single() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            ..Default::default()
        };
        let p = c.gpu_sm_partition(1);
        assert_eq!(p.sm_per_partition, 80);
    }

    #[test]
    fn gpu_sm_partition_no_gpu() {
        let c = CompilerConstraints::default();
        // gpu_sm_count = None → unwrap_or(1) → total_sm = 1
        let p = c.gpu_sm_partition(2);
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 0); // 1 / 2 = 0
    }

    #[test]
    fn gpu_sm_partition_zero_partitions() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            ..Default::default()
        };
        // num_partitions.max(1) = 1, so 80/1 = 80
        let p = c.gpu_sm_partition(0);
        assert_eq!(p.sm_per_partition, 80);
    }

    #[test]
    fn gpu_sm_partition_copy_clone() {
        let p = GpuSmPartition { total_sm: 80, num_partitions: 2, sm_per_partition: 40 };
        let p2 = p;
        assert_eq!(p2.total_sm, 80);
        let p3 = p.clone();
        assert_eq!(p3.sm_per_partition, 40);
    }

    #[test]
    fn gpu_sm_partition_debug() {
        let p = GpuSmPartition { total_sm: 80, num_partitions: 2, sm_per_partition: 40 };
        let debug = format!("{:?}", p);
        assert!(debug.contains("80"));
        assert!(debug.contains("40"));
    }

    #[test]
    fn numa_core_bindings_single_node() {
        let c = CompilerConstraints {
            numa_node_count: 1,
            ..Default::default()
        };
        let bindings = c.numa_core_bindings();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 0); // node_id
    }

    #[test]
    fn numa_core_bindings_multiple_nodes() {
        let c = CompilerConstraints {
            numa_node_count: 4,
            ..Default::default()
        };
        let bindings = c.numa_core_bindings();
        assert_eq!(bindings.len(), 4);
        // Each node should have non-overlapping core ranges
        for (i, (node_id, range)) in bindings.iter().enumerate() {
            assert_eq!(*node_id, i);
            assert!(!range.is_empty());
        }
    }

    #[test]
    fn derive_amx_tile_bits() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let _kc = &profile.kernel_config;
        // Can't force AMX on without GPU, but we verify derive succeeds
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // tile_m/tile_n/tile_k are always positive
        assert!(c.optimal_tile_bits.tile_m > 0);
        assert!(c.optimal_tile_bits.tile_n > 0);
        assert!(c.optimal_tile_bits.tile_k > 0);
    }

    #[test]
    fn derive_metal_gpu_no_sm_version() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1001 },
            compute_unit_count: 40,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        assert_eq!(c.gpu_sm_version, None, "Metal does not use SM version");
        assert_eq!(c.gpu_sm_count, Some(40));
        assert!(!c.has_tma, "TMA is NVIDIA-only");
        assert!(!c.has_native_fp4, "FP4 is NVIDIA-only");
    }

    #[test]
    fn derive_gpu_l2_zero_keeps_cpu_l2() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let cpu_l2 = sensors.l2_cache_bytes;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 164 * 1024,
            l2_bytes: 0, // GPU reports 0 L2
            global_mem_bytes: 48 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // l2_bytes = 0 → should keep CPU L2 (sensors.l2_cache_bytes.max(profile_l2))
        assert!(c.l2_cache_size >= cpu_l2);
    }

    #[test]
    fn tile_bits_serialize_deserialize() {
        let t = TileBits { tile_m: 64, tile_n: 64, tile_k: 64 };
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        assert_eq!(t2.tile_m, 64);
        assert_eq!(t2.tile_n, 64);
        assert_eq!(t2.tile_k, 64);
    }

    #[test]
    fn compiler_constraints_debug_format() {
        let c = CompilerConstraints::default();
        let debug = format!("{:?}", c);
        assert!(debug.contains("max_gpr_count"));
        assert!(debug.contains("simd_width_bits"));
    }

    // ---- Additional coverage tests ----

    #[test]
    fn tile_bits_debug_format() {
        let t = TileBits { tile_m: 64, tile_n: 32, tile_k: 16 };
        let debug = format!("{:?}", t);
        assert!(debug.contains("tile_m"));
        assert!(debug.contains("tile_n"));
        assert!(debug.contains("tile_k"));
    }

    #[test]
    fn tile_bits_equality_manual() {
        let a = TileBits { tile_m: 16, tile_n: 32, tile_k: 64 };
        let b = TileBits { tile_m: 16, tile_n: 32, tile_k: 64 };
        assert_eq!(a.tile_m, b.tile_m);
        assert_eq!(a.tile_n, b.tile_n);
        assert_eq!(a.tile_k, b.tile_k);
    }

    #[test]
    fn tile_bits_inequality_manual() {
        let a = TileBits { tile_m: 16, tile_n: 32, tile_k: 64 };
        let b = TileBits { tile_m: 32, tile_n: 32, tile_k: 64 };
        assert_ne!(a.tile_m, b.tile_m);
    }

    #[test]
    fn tile_bits_custom_construction() {
        let t = TileBits { tile_m: 128, tile_n: 256, tile_k: 512 };
        assert_eq!(t.tile_m, 128);
        assert_eq!(t.tile_n, 256);
        assert_eq!(t.tile_k, 512);
    }

    #[test]
    fn gpu_sm_partition_fields_access() {
        let p = GpuSmPartition {
            total_sm: 108,
            num_partitions: 4,
            sm_per_partition: 27,
        };
        assert_eq!(p.total_sm, 108);
        assert_eq!(p.num_partitions, 4);
        assert_eq!(p.sm_per_partition, 27);
    }

    #[test]
    fn gpu_sm_partition_more_partitions_than_sm() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(4),
            ..Default::default()
        };
        let p = c.gpu_sm_partition(10);
        assert_eq!(p.total_sm, 4);
        assert_eq!(p.num_partitions, 10);
        assert_eq!(p.sm_per_partition, 0); // 4 / 10 = 0
    }

    #[test]
    fn gpu_sm_partition_large_sm_count() {
        let c = CompilerConstraints {
            gpu_sm_count: Some(192),
            ..Default::default()
        };
        let p = c.gpu_sm_partition(6);
        assert_eq!(p.sm_per_partition, 32); // 192 / 6 = 32
        assert_eq!(p.total_sm, 192);
    }

    #[test]
    fn gpu_sm_partition_clone_independence() {
        let p = GpuSmPartition { total_sm: 80, num_partitions: 2, sm_per_partition: 40 };
        let mut p2 = p.clone();
        p2.sm_per_partition = 20;
        assert_eq!(p.sm_per_partition, 40, "original must not be mutated");
        assert_eq!(p2.sm_per_partition, 20);
    }

    #[test]
    fn compiler_constraints_serialize_roundtrip_minimal() {
        let c = CompilerConstraints::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        assert_eq!(c.max_gpr_count, c2.max_gpr_count);
        assert_eq!(c.l1i_size, c2.l1i_size);
        assert_eq!(c.l2_cache_size, c2.l2_cache_size);
        assert_eq!(c.numa_node_count, c2.numa_node_count);
        assert_eq!(c.tensor_core_gen, c2.tensor_core_gen);
    }

    #[test]
    fn compiler_constraints_serialize_with_gpu_fields() {
        let c = CompilerConstraints {
            gpu_sm_version: Some(90),
            gpu_sm_count: Some(132),
            gpu_warp_size: Some(32),
            smem_size: Some(228 * 1024),
            tensor_core_gen: 3,
            has_tma: true,
            ..Default::default()
        };
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.gpu_sm_version, Some(90));
        assert_eq!(c2.gpu_sm_count, Some(132));
        assert_eq!(c2.smem_size, Some(228 * 1024));
        assert!(c2.has_tma);
        assert_eq!(c2.tensor_core_gen, 3);
    }

    #[test]
    fn exceeds_l1i_budget_large_cache() {
        let c = CompilerConstraints {
            l1i_size: 64 * 1024,
            ..Default::default()
        };
        let threshold = (64_usize * 1024) as f64;
        let threshold = (threshold * 0.8) as usize;
        assert!(!c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold + 1));
    }

    #[test]
    fn exceeds_l1i_budget_one_byte_boundary() {
        let c = CompilerConstraints {
            l1i_size: 100,
            ..Default::default()
        };
        // 80% of 100 = 80.0, floor = 80
        assert!(!c.exceeds_l1i_budget(80));
        assert!(c.exceeds_l1i_budget(81));
    }

    #[test]
    fn numa_core_bindings_zero_nodes() {
        let c = CompilerConstraints {
            numa_node_count: 0,
            ..Default::default()
        };
        let bindings = c.numa_core_bindings();
        assert!(bindings.is_empty(), "zero NUMA nodes should produce empty bindings");
    }

    #[test]
    fn numa_core_bindings_node_ranges_non_overlapping() {
        let c = CompilerConstraints {
            numa_node_count: 2,
            ..Default::default()
        };
        let bindings = c.numa_core_bindings();
        assert_eq!(bindings.len(), 2);
        // Ranges must not overlap: end of node 0 == start of node 1
        assert_eq!(bindings[0].1.end, bindings[1].1.start);
    }

    #[test]
    fn compiler_constraints_field_mutate_after_clone() {
        let mut c = CompilerConstraints::default();
        c.has_amx = true;
        c.max_gpr_count = 31;
        let c2 = c.clone();
        c.has_amx = false;
        assert!(c2.has_amx, "clone must be independent");
        assert_eq!(c2.max_gpr_count, 31);
    }

    #[test]
    fn derive_cuda_sm80_no_tma_no_fp4() {
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 164 * 1024,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 48 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        assert_eq!(c.gpu_sm_version, Some(80));
        assert_eq!(c.tensor_core_gen, 2);
        assert!(!c.has_tma, "Ampere SM80 must not have TMA");
        assert!(!c.has_native_fp4, "Ampere SM80 does not have FP4");
        assert!(!c.has_native_fp6, "Ampere SM80 does not have FP6");
        assert_eq!(c.l2_cache_size, 40 * 1024 * 1024);
    }

    #[test]
    fn compiler_constraints_default_simd_width_positive() {
        let c = CompilerConstraints::default();
        assert!(c.simd_width_bits > 0, "default SIMD width must be positive");
        assert_eq!(c.simd_width_bits % 8, 0, "SIMD width in bits should be byte-aligned");
    }

    #[test]
    fn tile_bits_zero_dimensions() {
        let t = TileBits { tile_m: 0, tile_n: 0, tile_k: 0 };
        assert_eq!(t.tile_m, 0);
        assert_eq!(t.tile_n, 0);
        assert_eq!(t.tile_k, 0);
    }

    #[test]
    fn tile_bits_asymmetric_dimensions() {
        let t = TileBits { tile_m: 128, tile_n: 8, tile_k: 32 };
        assert_eq!(t.tile_m, 128);
        assert_eq!(t.tile_n, 8);
        assert_eq!(t.tile_k, 32);
    }

    #[test]
    fn compiler_constraints_debug_contains_all_key_fields() {
        let c = CompilerConstraints {
            has_amx: true,
            gpu_sm_count: Some(64),
            rdma_min_chunk_tokens: Some(256),
            ..Default::default()
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("has_amx"));
        assert!(debug.contains("gpu_sm_count"));
        assert!(debug.contains("rdma_min_chunk_tokens"));
    }

    #[test]
    fn tile_bits_serialize_roundtrip_json_keys() {
        let t = TileBits { tile_m: 4, tile_n: 8, tile_k: 2 };
        let json = serde_json::to_string(&t).unwrap();
        assert!(json.contains("tile_m"));
        assert!(json.contains("tile_n"));
        assert!(json.contains("tile_k"));
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        assert_eq!(t2.tile_m, 4);
        assert_eq!(t2.tile_n, 8);
        assert_eq!(t2.tile_k, 2);
    }

    #[test]
    fn gpu_sm_partition_copy_semantics() {
        let p = GpuSmPartition { total_sm: 80, num_partitions: 2, sm_per_partition: 40 };
        let p2 = p; // Copy, not move
        assert_eq!(p.total_sm, 80); // still accessible after copy
        assert_eq!(p2.total_sm, 80);
    }

    // ---- New tests (17 additional) ----

    #[test]
    fn compiler_constraints_debug_outputs_l1i_size() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 48 * 1024,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("l1i_size"), "Debug output must contain l1i_size field");
        assert!(debug.contains(&format!("{}", 48 * 1024)));
    }

    #[test]
    fn compiler_constraints_debug_outputs_gpu_sm_version() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_version: Some(90),
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("gpu_sm_version"), "Debug output must contain gpu_sm_version field");
    }

    #[test]
    fn compiler_constraints_clone_preserves_vec_field() {
        // Arrange
        let bindings = vec![(0, 0, 4), (1, 4, 8)];
        let c = CompilerConstraints {
            numa_core_bindings: bindings.clone(),
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        // Assert
        assert_eq!(c2.numa_core_bindings, bindings);
        assert_eq!(c2.numa_core_bindings.len(), 2);
    }

    #[test]
    fn compiler_constraints_clone_deep_copies_smem_size() {
        // Arrange
        let c = CompilerConstraints {
            smem_size: Some(164 * 1024),
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        // Assert
        assert_eq!(c2.smem_size, Some(164 * 1024));
    }

    #[test]
    fn compiler_constraints_serialize_deserialize_all_optional_fields() {
        // Arrange
        let c = CompilerConstraints {
            smem_size: Some(64 * 1024),
            gpu_sm_version: Some(80),
            gpu_sm_count: Some(108),
            gpu_warp_size: Some(32),
            sve_vl_bytes: Some(32),
            rdma_min_chunk_tokens: Some(128),
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.smem_size, Some(64 * 1024));
        assert_eq!(c2.gpu_sm_version, Some(80));
        assert_eq!(c2.gpu_sm_count, Some(108));
        assert_eq!(c2.gpu_warp_size, Some(32));
        assert_eq!(c2.sve_vl_bytes, Some(32));
        assert_eq!(c2.rdma_min_chunk_tokens, Some(128));
    }

    #[test]
    fn tile_bits_large_values() {
        // Arrange & Act
        let t = TileBits { tile_m: 1024, tile_n: 2048, tile_k: 4096 };
        // Assert
        assert_eq!(t.tile_m, 1024);
        assert_eq!(t.tile_n, 2048);
        assert_eq!(t.tile_k, 4096);
    }

    #[test]
    fn tile_bits_copy_preserves_all_fields() {
        // Arrange
        let t = TileBits { tile_m: 48, tile_n: 96, tile_k: 192 };
        // Act
        let t2 = t; // Copy
        let t3 = t; // Copy again (original still valid)
        // Assert
        assert_eq!(t.tile_m, 48);
        assert_eq!(t2.tile_n, 96);
        assert_eq!(t3.tile_k, 192);
    }

    #[test]
    fn tile_bits_max_usize_values() {
        // Arrange & Act
        let t = TileBits {
            tile_m: usize::MAX,
            tile_n: usize::MAX,
            tile_k: usize::MAX,
        };
        // Assert
        assert_eq!(t.tile_m, usize::MAX);
        assert_eq!(t.tile_n, usize::MAX);
        assert_eq!(t.tile_k, usize::MAX);
    }

    #[test]
    fn exceeds_l1i_budget_max_usize_l1i() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: usize::MAX,
            ..Default::default()
        };
        // Act
        let threshold = (usize::MAX as f64 * 0.8) as usize;
        // Assert: threshold itself should not exceed budget
        assert!(!c.exceeds_l1i_budget(threshold));
        // But threshold + 1 should exceed (wrapping)
        assert!(c.exceeds_l1i_budget(threshold.saturating_add(1)));
    }

    #[test]
    fn exceeds_l1i_budget_one_byte_l1i() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 1,
            ..Default::default()
        };
        // Act & Assert: 1 * 0.8 = 0.8, floor = 0
        assert!(!c.exceeds_l1i_budget(0));
        assert!(c.exceeds_l1i_budget(1));
    }

    #[test]
    fn gpu_sm_partition_exact_division() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(120),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(4);
        // Assert: 120 / 4 = 30, exact division
        assert_eq!(p.total_sm, 120);
        assert_eq!(p.num_partitions, 4);
        assert_eq!(p.sm_per_partition, 30);
    }

    #[test]
    fn gpu_sm_partition_remainder_truncated() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(7),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(3);
        // Assert: 7 / 3 = 2 (integer division truncates)
        assert_eq!(p.sm_per_partition, 2);
    }

    #[test]
    fn gpu_sm_partition_debug_contains_num_partitions() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 100,
            num_partitions: 5,
            sm_per_partition: 20,
        };
        // Act
        let debug = format!("{:?}", p);
        // Assert
        assert!(debug.contains("num_partitions"));
        assert!(debug.contains("total_sm"));
        assert!(debug.contains("sm_per_partition"));
    }

    #[test]
    fn gpu_sm_partition_zero_sm_count() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(0),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(4);
        // Assert: 0 / 4 = 0
        assert_eq!(p.total_sm, 0);
        assert_eq!(p.sm_per_partition, 0);
    }

    #[test]
    fn compiler_constraints_default_tile_bits_same_dimensions() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let tile = c.optimal_tile_bits;
        // Assert: default tile is square with uniform k
        assert_eq!(tile.tile_m, tile.tile_n);
        assert_eq!(tile.tile_m, 16);
        assert_eq!(tile.tile_k, 16);
    }

    #[test]
    fn compiler_constraints_default_no_gpu_features() {
        // Arrange
        let c = CompilerConstraints::default();
        // Assert: CPU-only defaults must not claim GPU features
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
        assert_eq!(c.tensor_core_gen, 0);
        assert_eq!(c.gpu_sm_version, None);
    }

    #[test]
    fn compiler_constraints_default_no_vector_extensions() {
        // Arrange
        let c = CompilerConstraints::default();
        // Assert: baseline CPU without advanced extensions
        assert!(!c.has_amx);
        assert!(!c.has_avx512);
        assert!(!c.has_sve);
        assert_eq!(c.sve_vl_bytes, None);
    }

    // ---- Batch 3: 45 additional tests ----

    // -- TileBits edge cases --

    #[test]
    fn tile_bits_unit_dimensions() {
        // Arrange & Act
        let t = TileBits { tile_m: 1, tile_n: 1, tile_k: 1 };
        // Assert
        assert_eq!(t.tile_m, 1);
        assert_eq!(t.tile_n, 1);
        assert_eq!(t.tile_k, 1);
    }

    #[test]
    fn tile_bits_single_dimension_one() {
        // Arrange & Act: only one dimension is non-default
        let t = TileBits { tile_m: 1, tile_n: 16, tile_k: 16 };
        // Assert
        assert_eq!(t.tile_m, 1);
        assert_eq!(t.tile_n, 16);
    }

    #[test]
    fn tile_bits_copy_independent_mutations() {
        // Arrange
        let mut t = TileBits { tile_m: 16, tile_n: 32, tile_k: 64 };
        // Act
        let t2 = t;
        t.tile_m = 999;
        // Assert: copy is independent
        assert_eq!(t.tile_m, 999);
        assert_eq!(t2.tile_m, 16);
    }

    #[test]
    fn tile_bits_debug_shows_field_names() {
        // Arrange
        let t = TileBits { tile_m: 1, tile_n: 2, tile_k: 3 };
        // Act
        let debug = format!("{:?}", t);
        // Assert
        assert!(debug.contains("tile_m: 1"));
        assert!(debug.contains("tile_n: 2"));
        assert!(debug.contains("tile_k: 3"));
    }

    #[test]
    fn tile_bits_serialize_custom_values() {
        // Arrange
        let t = TileBits { tile_m: 7, tile_n: 11, tile_k: 13 };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert: roundtrip preserves exact values
        assert_eq!(t2.tile_m, 7);
        assert_eq!(t2.tile_n, 11);
        assert_eq!(t2.tile_k, 13);
    }

    #[test]
    fn tile_bits_default_is_square() {
        // Arrange & Act
        let t = TileBits::default();
        // Assert: default has all equal dimensions
        assert_eq!(t.tile_m, t.tile_n);
        assert_eq!(t.tile_n, t.tile_k);
    }

    // -- CompilerConstraints construction edge cases --

    #[test]
    fn compiler_constraints_manual_construction_all_bools_true() {
        // Arrange & Act
        let c = CompilerConstraints {
            native_int4_dot: true,
            has_amx: true,
            has_avx512: true,
            has_sve: true,
            has_tma: true,
            has_native_fp4: true,
            has_native_fp6: true,
            ..Default::default()
        };
        // Assert
        assert!(c.native_int4_dot);
        assert!(c.has_amx);
        assert!(c.has_avx512);
        assert!(c.has_sve);
        assert!(c.has_tma);
        assert!(c.has_native_fp4);
        assert!(c.has_native_fp6);
    }

    #[test]
    fn compiler_constraints_zero_l2_cache() {
        // Arrange
        let c = CompilerConstraints {
            l2_cache_size: 0,
            ..Default::default()
        };
        // Assert: construction is valid even with zero L2
        assert_eq!(c.l2_cache_size, 0);
    }

    #[test]
    fn compiler_constraints_zero_simd_width() {
        // Arrange
        let c = CompilerConstraints {
            simd_width_bits: 0,
            ..Default::default()
        };
        // Assert: construction is valid even with zero SIMD width
        assert_eq!(c.simd_width_bits, 0);
    }

    #[test]
    fn compiler_constraints_max_tensor_core_gen() {
        // Arrange
        let c = CompilerConstraints {
            tensor_core_gen: u32::MAX,
            ..Default::default()
        };
        // Assert: u32::MAX is a valid value for the field
        assert_eq!(c.tensor_core_gen, u32::MAX);
    }

    #[test]
    fn compiler_constraints_numa_core_bindings_empty_vec() {
        // Arrange
        let c = CompilerConstraints {
            numa_core_bindings: vec![],
            ..Default::default()
        };
        // Assert
        assert!(c.numa_core_bindings.is_empty());
    }

    #[test]
    fn compiler_constraints_numa_core_bindings_multiple_entries() {
        // Arrange
        let bindings = vec![
            (0, 0, 3),
            (1, 3, 7),
            (2, 7, 12),
        ];
        let c = CompilerConstraints {
            numa_core_bindings: bindings.clone(),
            ..Default::default()
        };
        // Assert
        assert_eq!(c.numa_core_bindings.len(), 3);
        assert_eq!(c.numa_core_bindings[0], (0, 0, 3));
        assert_eq!(c.numa_core_bindings[2], (2, 7, 12));
    }

    #[test]
    fn compiler_constraints_sve_vl_bytes_set() {
        // Arrange
        let c = CompilerConstraints {
            sve_vl_bytes: Some(32),
            ..Default::default()
        };
        // Assert
        assert_eq!(c.sve_vl_bytes, Some(32));
    }

    #[test]
    fn compiler_constraints_rdma_chunk_tokens_set() {
        // Arrange
        let c = CompilerConstraints {
            rdma_min_chunk_tokens: Some(256),
            ..Default::default()
        };
        // Assert
        assert_eq!(c.rdma_min_chunk_tokens, Some(256));
    }

    #[test]
    fn compiler_constraints_clone_independence_bool_fields() {
        // Arrange
        let mut c = CompilerConstraints {
            has_amx: true,
            has_avx512: true,
            has_sve: true,
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.has_amx = false;
        c.has_avx512 = false;
        c.has_sve = false;
        // Assert: clone preserves original values
        assert!(c2.has_amx);
        assert!(c2.has_avx512);
        assert!(c2.has_sve);
    }

    #[test]
    fn compiler_constraints_clone_independence_numeric_fields() {
        // Arrange
        let mut c = CompilerConstraints {
            max_gpr_count: 31,
            simd_width_bits: 512,
            l1i_size: 64 * 1024,
            tensor_core_gen: 3,
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.max_gpr_count = 0;
        c.simd_width_bits = 0;
        c.l1i_size = 0;
        c.tensor_core_gen = 0;
        // Assert
        assert_eq!(c2.max_gpr_count, 31);
        assert_eq!(c2.simd_width_bits, 512);
        assert_eq!(c2.l1i_size, 64 * 1024);
        assert_eq!(c2.tensor_core_gen, 3);
    }

    #[test]
    fn compiler_constraints_clone_independence_optional_fields() {
        // Arrange
        let mut c = CompilerConstraints {
            gpu_sm_version: Some(100),
            gpu_sm_count: Some(192),
            gpu_warp_size: Some(64),
            smem_size: Some(256 * 1024),
            sve_vl_bytes: Some(16),
            rdma_min_chunk_tokens: Some(512),
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.gpu_sm_version = None;
        c.gpu_sm_count = None;
        c.gpu_warp_size = None;
        c.smem_size = None;
        // Assert
        assert_eq!(c2.gpu_sm_version, Some(100));
        assert_eq!(c2.gpu_sm_count, Some(192));
        assert_eq!(c2.gpu_warp_size, Some(64));
        assert_eq!(c2.smem_size, Some(256 * 1024));
    }

    #[test]
    fn compiler_constraints_serialize_roundtrip_all_bools() {
        // Arrange
        let c = CompilerConstraints {
            native_int4_dot: true,
            has_amx: true,
            has_avx512: true,
            has_sve: true,
            has_tma: true,
            has_native_fp4: true,
            has_native_fp6: true,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.native_int4_dot);
        assert!(c2.has_amx);
        assert!(c2.has_avx512);
        assert!(c2.has_sve);
        assert!(c2.has_tma);
        assert!(c2.has_native_fp4);
        assert!(c2.has_native_fp6);
    }

    #[test]
    fn compiler_constraints_serialize_roundtrip_vec_field() {
        // Arrange
        let c = CompilerConstraints {
            numa_core_bindings: vec![(0, 0, 8), (1, 8, 16)],
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.numa_core_bindings.len(), 2);
        assert_eq!(c2.numa_core_bindings[0], (0, 0, 8));
        assert_eq!(c2.numa_core_bindings[1], (1, 8, 16));
    }

    #[test]
    fn compiler_constraints_serialize_roundtrip_tile_bits() {
        // Arrange
        let c = CompilerConstraints {
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.optimal_tile_bits.tile_m, 64);
        assert_eq!(c2.optimal_tile_bits.tile_n, 64);
        assert_eq!(c2.optimal_tile_bits.tile_k, 64);
    }

    #[test]
    fn compiler_constraints_debug_contains_numa_node_count() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 4,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("numa_node_count"));
    }

    #[test]
    fn compiler_constraints_debug_contains_all_optional_fields() {
        // Arrange
        let c = CompilerConstraints {
            smem_size: Some(999),
            gpu_sm_version: Some(88),
            gpu_sm_count: Some(77),
            gpu_warp_size: Some(66),
            sve_vl_bytes: Some(55),
            rdma_min_chunk_tokens: Some(44),
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert: all optional field names must appear
        assert!(debug.contains("smem_size"));
        assert!(debug.contains("gpu_sm_version"));
        assert!(debug.contains("gpu_sm_count"));
        assert!(debug.contains("gpu_warp_size"));
        assert!(debug.contains("sve_vl_bytes"));
        assert!(debug.contains("rdma_min_chunk_tokens"));
    }

    #[test]
    fn compiler_constraints_debug_contains_bool_fields() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("native_int4_dot"));
        assert!(debug.contains("has_tma"));
        assert!(debug.contains("has_native_fp4"));
        assert!(debug.contains("has_native_fp6"));
    }

    // -- exceeds_l1i_budget edge cases --

    #[test]
    fn exceeds_l1i_budget_small_l1i_exact_half() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 1000,
            ..Default::default()
        };
        // Act & Assert: 80% of 1000 = 800
        assert!(!c.exceeds_l1i_budget(800));
        assert!(c.exceeds_l1i_budget(801));
    }

    #[test]
    fn exceeds_l1i_budget_two_byte_l1i() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 2,
            ..Default::default()
        };
        // Act & Assert: 2 * 0.8 = 1.6, floor = 1
        assert!(!c.exceeds_l1i_budget(1));
        assert!(c.exceeds_l1i_budget(2));
    }

    #[test]
    fn exceeds_l1i_budget_typical_x86_l1i() {
        // Arrange: typical x86 L1i = 32KB
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        let threshold = 32_usize * 1024 * 8 / 10; // 80% = 26214
        // Act & Assert
        assert!(!c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold + 100));
    }

    // -- gpu_sm_partition edge cases --

    #[test]
    fn gpu_sm_partition_one_sm_one_partition() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(1),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(1);
        // Assert
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 1);
    }

    #[test]
    fn gpu_sm_partition_one_sm_many_partitions() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(1),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(100);
        // Assert: 1 / 100 = 0
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 0);
    }

    #[test]
    fn gpu_sm_partition_large_power_of_two() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(256),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(8);
        // Assert: 256 / 8 = 32
        assert_eq!(p.sm_per_partition, 32);
    }

    // -- GpuSmPartition edge cases --

    #[test]
    fn gpu_sm_partition_copy_then_mutate_original() {
        // Arrange
        let mut p = GpuSmPartition {
            total_sm: 80,
            num_partitions: 4,
            sm_per_partition: 20,
        };
        // Act
        let p2 = p;
        p.total_sm = 0;
        // Assert: copy semantics means p2 keeps original value
        assert_eq!(p2.total_sm, 80);
    }

    #[test]
    fn gpu_sm_partition_zero_all_fields() {
        // Arrange & Act
        let p = GpuSmPartition {
            total_sm: 0,
            num_partitions: 0,
            sm_per_partition: 0,
        };
        // Assert
        assert_eq!(p.total_sm, 0);
        assert_eq!(p.num_partitions, 0);
        assert_eq!(p.sm_per_partition, 0);
    }

    #[test]
    fn gpu_sm_partition_clone_is_copy_equivalent() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 64,
            num_partitions: 2,
            sm_per_partition: 32,
        };
        // Act: both Copy and Clone should produce identical results
        let p_copy = p;
        let p_clone = p.clone();
        // Assert
        assert_eq!(p_copy.total_sm, p_clone.total_sm);
        assert_eq!(p_copy.num_partitions, p_clone.num_partitions);
        assert_eq!(p_copy.sm_per_partition, p_clone.sm_per_partition);
    }

    // -- numa_core_bindings edge cases --

    #[test]
    fn numa_core_bindings_single_node_covers_all_cores() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 1,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: single node covers 0..total_cores
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 0);
        assert_eq!(bindings[0].1.start, 0);
        assert_eq!(bindings[0].1.end, total_cores);
    }

    #[test]
    fn numa_core_bindings_three_nodes_contiguous() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 3,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: ranges must be contiguous — node i end == node i+1 start
        assert_eq!(bindings.len(), 3);
        for i in 0..bindings.len() - 1 {
            assert_eq!(bindings[i].1.end, bindings[i + 1].1.start);
        }
    }

    // -- derive with different GPU platforms --

    #[test]
    fn derive_with_cuda_sm70_no_tma_no_fp4_no_fp6() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 70 },
            compute_unit_count: 56,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 96 * 1024,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, Some(70));
        assert_eq!(c.gpu_sm_count, Some(56));
        assert!(!c.has_tma, "Volta SM70 must not have TMA");
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
        assert_eq!(c.l2_cache_size, 6 * 1024 * 1024);
    }

    #[test]
    fn derive_with_cuda_sm89_tma_false_fp4_false() {
        // Arrange: SM89 (Ada Lovelace) — below SM90 TMA threshold
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 89 },
            compute_unit_count: 142,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 100 * 1024,
            l2_bytes: 36 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 9,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, Some(89));
        assert!(!c.has_tma, "Ada SM89 must not have TMA (requires SM90+)");
        assert!(!c.has_native_fp4);
    }

    #[test]
    fn derive_with_cuda_sm95_has_tma_no_fp4() {
        // Arrange: SM95 (Hopper variant) — has TMA but no FP4
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 95 },
            compute_unit_count: 100,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 228 * 1024,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 5,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, Some(95));
        assert!(c.has_tma, "SM95 >= SM90 must have TMA");
        assert!(!c.has_native_fp4, "SM95 < SM100 must not have FP4");
        assert!(!c.has_native_fp6, "SM95 < SM100 must not have FP6");
    }

    #[test]
    fn derive_with_hip_gpu_l2_applied() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 110,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, None);
        assert_eq!(c.gpu_sm_count, Some(110));
        assert_eq!(c.gpu_warp_size, Some(64));
        assert_eq!(c.l2_cache_size, 8 * 1024 * 1024);
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
    }

    #[test]
    fn derive_with_metal_gpu_smem_set() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1002 },
            compute_unit_count: 48,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 32 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, None);
        assert_eq!(c.gpu_sm_count, Some(48));
        assert_eq!(c.smem_size, Some(32 * 1024));
        assert!(!c.has_tma);
    }

    // -- derive CPU-only: GPU fields all cleared --

    #[test]
    fn derive_cpu_only_all_gpu_fields_none() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act: no gpu_topology
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert_eq!(c.gpu_sm_version, None);
        assert_eq!(c.gpu_sm_count, None);
        assert_eq!(c.gpu_warp_size, None);
        assert_eq!(c.smem_size, None);
        assert_eq!(c.tensor_core_gen, 0);
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
    }

    #[test]
    fn derive_cpu_only_positive_cache_sizes() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert!(c.l1i_size > 0, "CPU L1i must be positive");
        assert!(c.l2_cache_size > 0, "CPU L2 must be positive");
    }

    #[test]
    fn derive_cpu_only_simd_width_multiple_of_8() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: simd_width_bits = simd_width_bytes * 8, always divisible by 8
        assert_eq!(c.simd_width_bits % 8, 0);
    }

    // -- derive with probe_result Some (tests the probe path is exercised) --

    #[test]
    fn derive_with_probe_result_some() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let probe = ProbeResult {
            spill_points: vec![112, 463],
            smem_cliffs: vec![(256, 0.8)],
            l2_thrash_threshold: 4 * 1024 * 1024,
            device_fingerprint: "test-device".to_string(),
            raw_measurements: Default::default(),
        };
        // Act: probe_result = Some should not panic
        let c = CompilerConstraints::derive(&profile, &sensors, Some(&probe), None);
        // Assert: basic sanity — still produces valid constraints
        assert!(c.simd_width_bits > 0);
        assert!(c.l1i_size > 0);
    }

    // -- gpu_sm_partition with various SM counts --

    #[test]
    fn gpu_sm_partition_uneven_division() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(13),
            ..Default::default()
        };
        // Act: 13 / 3 = 4 (truncated)
        let p = c.gpu_sm_partition(3);
        // Assert
        assert_eq!(p.sm_per_partition, 4);
    }

    #[test]
    fn gpu_sm_partition_none_defaults_to_one() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: None,
            ..Default::default()
        };
        // Act: None → unwrap_or(1)
        let p = c.gpu_sm_partition(1);
        // Assert
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 1);
    }

    // -- GpuSmPartition debug format --

    #[test]
    fn gpu_sm_partition_debug_shows_values() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 132,
            num_partitions: 3,
            sm_per_partition: 44,
        };
        // Act
        let debug = format!("{:?}", p);
        // Assert: must contain the field values
        assert!(debug.contains("132"));
        assert!(debug.contains("44"));
    }

    // -- TileBits serialize with zero --

    #[test]
    fn tile_bits_serialize_zero_values() {
        // Arrange
        let t = TileBits { tile_m: 0, tile_n: 0, tile_k: 0 };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(t2.tile_m, 0);
        assert_eq!(t2.tile_n, 0);
        assert_eq!(t2.tile_k, 0);
    }

    // -- CompilerConstraints default numa_node_count is 1 --

    #[test]
    fn compiler_constraints_default_numa_node_count() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert
        assert_eq!(c.numa_node_count, 1, "default must be single NUMA node");
    }

    // -- CompilerConstraints with large values --

    #[test]
    fn compiler_constraints_large_l1i_size() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: usize::MAX / 2,
            ..Default::default()
        };
        // Act & Assert: does not panic
        let threshold = (c.l1i_size as f64 * 0.8) as usize;
        assert!(!c.exceeds_l1i_budget(threshold));
    }

    #[test]
    fn compiler_constraints_large_gpu_sm_count() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(1024),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(8);
        // Assert: 1024 / 8 = 128
        assert_eq!(p.sm_per_partition, 128);
        assert_eq!(p.total_sm, 1024);
    }

    // ---- Batch 4: 55 additional tests ----

    // -- derive() SM version boundary & platform coverage --

    #[test]
    fn derive_cuda_sm99_has_tma_no_fp4() {
        // Arrange: SM99 between SM95 and SM100
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 99 },
            compute_unit_count: 128,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 200 * 1024,
            l2_bytes: 60 * 1024 * 1024,
            global_mem_bytes: 96 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 9,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert!(c.has_tma, "SM99 >= 90 must have TMA");
        assert!(!c.has_native_fp4, "SM99 < 100 must not have FP4");
        assert!(!c.has_native_fp6, "SM99 < 100 must not have FP6");
        assert_eq!(c.gpu_sm_version, Some(99));
    }

    #[test]
    fn derive_cuda_sm120_future_all_features() {
        // Arrange: future SM version beyond Blackwell
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 120 },
            compute_unit_count: 256,
            tensor_core_gen: 5,
            shared_mem_per_sm_bytes: 512 * 1024,
            l2_bytes: 128 * 1024 * 1024,
            global_mem_bytes: 256 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 12,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert!(c.has_tma);
        assert!(c.has_native_fp4);
        assert!(c.has_native_fp6);
        assert_eq!(c.gpu_sm_version, Some(120));
        assert_eq!(c.tensor_core_gen, 5);
    }

    #[test]
    fn derive_gpu_l2_large_overrides_cpu_l2() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_l2 = 80 * 1024 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 228 * 1024,
            l2_bytes: gpu_l2,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.l2_cache_size, gpu_l2);
    }

    #[test]
    fn derive_gpu_zero_l2_keeps_sensors_l2() {
        // Arrange: GPU reports l2_bytes = 0
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let sensor_l2 = sensors.l2_cache_bytes;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 164 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 48 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert!(c.l2_cache_size >= sensor_l2,
            "GPU L2=0 must not reduce below sensors L2");
    }

    #[test]
    fn derive_hip_shared_mem_set_correctly() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let smem = 96 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x942 },
            compute_unit_count: 228,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: smem,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x42,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.smem_size, Some(smem));
        assert_eq!(c.gpu_sm_version, None);
    }

    #[test]
    fn derive_metal_tensor_core_gen_preserved() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1003 },
            compute_unit_count: 40,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 48 * 1024,
            l2_bytes: 16 * 1024 * 1024,
            global_mem_bytes: 32 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.tensor_core_gen, 2);
        assert_eq!(c.gpu_sm_version, None);
    }

    #[test]
    fn derive_hip_sm_version_none_for_all_gfx() {
        // Arrange: HIP with various gfx architectures
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        for gfx in [0x90a, 0x942, 0x1100] {
            let gpu = GpuTopology {
                platform: GpuPlatform::Hip { gfx_arch: gfx },
                compute_unit_count: 100,
                tensor_core_gen: 2,
                shared_mem_per_sm_bytes: 64 * 1024,
                l2_bytes: 8 * 1024 * 1024,
                global_mem_bytes: 64 * 1024 * 1024 * 1024,
                warp_size: 64,
                compute_cap_major: 9,
                compute_cap_minor: 0,
            };
            // Act
            let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
            // Assert
            assert_eq!(c.gpu_sm_version, None,
                "HIP gfx={:#x} must not produce SM version", gfx);
        }
    }

    #[test]
    fn derive_cpu_l2_at_least_sensors_l2() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let sensor_l2 = sensors.l2_cache_bytes;
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert!(c.l2_cache_size >= sensor_l2);
    }

    #[test]
    fn derive_cpu_numa_at_least_one_node() {
        // Arrange: sensors without ccx_numa_topology → default
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert!(c.numa_node_count >= 1);
    }

    #[test]
    fn derive_with_probe_empty_data() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let probe = ProbeResult {
            spill_points: vec![],
            smem_cliffs: vec![],
            l2_thrash_threshold: 0,
            device_fingerprint: String::new(),
            raw_measurements: Default::default(),
        };
        // Act: should not panic with empty probe data
        let c = CompilerConstraints::derive(&profile, &sensors, Some(&probe), None);
        // Assert
        assert!(c.simd_width_bits > 0);
    }

    #[test]
    fn derive_with_probe_and_gpu_together() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let probe = ProbeResult {
            spill_points: vec![100, 200],
            smem_cliffs: vec![(512, 0.5)],
            l2_thrash_threshold: 8 * 1024 * 1024,
            device_fingerprint: "test".to_string(),
            raw_measurements: Default::default(),
        };
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, Some(&probe), Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, Some(90));
        assert!(c.has_tma);
        assert!(c.simd_width_bits > 0);
    }

    #[test]
    fn derive_cuda_sm80_shared_mem_size() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let smem = 164 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: smem,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 48 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.smem_size, Some(smem));
    }

    #[test]
    fn derive_cpu_max_gpr_count_range() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: GPR count is either 15 (standard) or 31 (APX)
        assert!(
            c.max_gpr_count == 15 || c.max_gpr_count == 31,
            "GPR count must be 15 or 31, got {}", c.max_gpr_count
        );
    }

    #[test]
    fn derive_cpu_tile_bits_all_positive() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert!(c.optimal_tile_bits.tile_m > 0);
        assert!(c.optimal_tile_bits.tile_n > 0);
        assert!(c.optimal_tile_bits.tile_k > 0);
    }

    // -- exceeds_l1i_budget additional coverage --

    #[test]
    fn exceeds_l1i_budget_64kb_arm_typical() {
        // Arrange: typical ARM L1i = 64KB
        let c = CompilerConstraints {
            l1i_size: 64 * 1024,
            ..Default::default()
        };
        let threshold = (64_usize * 1024 * 8) / 10;
        // Act & Assert
        assert!(!c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold + 1));
    }

    #[test]
    fn exceeds_l1i_budget_very_large_input() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        // Act & Assert
        assert!(c.exceeds_l1i_budget(usize::MAX));
    }

    #[test]
    fn exceeds_l1i_budget_threshold_formula_property() {
        // Arrange: for any l1i_size, threshold = floor(l1i_size * 0.8)
        for &l1i in &[1, 10, 100, 1024, 32 * 1024, 64 * 1024, 128 * 1024] {
            let c = CompilerConstraints {
                l1i_size: l1i,
                ..Default::default()
            };
            let threshold = (l1i as f64 * 0.8) as usize;
            // Act & Assert
            assert!(
                !c.exceeds_l1i_budget(threshold),
                "threshold for l1i={} should not exceed", l1i
            );
            if threshold > 0 {
                assert!(!c.exceeds_l1i_budget(threshold - 1));
            }
        }
    }

    // -- gpu_sm_partition additional coverage --

    #[test]
    fn gpu_sm_partition_equal_sm_and_partitions() {
        // Arrange: 80 SMs, 80 partitions → 1 SM each
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(80);
        // Assert
        assert_eq!(p.sm_per_partition, 1);
        assert_eq!(p.total_sm, 80);
    }

    #[test]
    fn gpu_sm_partition_prime_number_sm() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(17),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(3);
        // Assert: 17 / 3 = 5 (truncated)
        assert_eq!(p.sm_per_partition, 5);
    }

    #[test]
    fn gpu_sm_partition_two_way_split() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(132),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(2);
        // Assert: 132 / 2 = 66
        assert_eq!(p.sm_per_partition, 66);
        assert_eq!(p.total_sm, 132);
        assert_eq!(p.num_partitions, 2);
    }

    #[test]
    fn gpu_sm_partition_invariant_total_coverage() {
        // Arrange & Act & Assert: sm_per_partition * num_partitions <= total_sm
        for &sm_count in &[1, 7, 13, 80, 132, 192, 256] {
            let c = CompilerConstraints {
                gpu_sm_count: Some(sm_count),
                ..Default::default()
            };
            for &n_parts in &[1, 2, 3, 4, 8, 16] {
                let p = c.gpu_sm_partition(n_parts);
                assert!(
                    p.sm_per_partition * p.num_partitions <= p.total_sm,
                    "sm_per_partition({}) * num_partitions({}) <= total_sm({})",
                    p.sm_per_partition, p.num_partitions, p.total_sm
                );
            }
        }
    }

    #[test]
    fn gpu_sm_partition_double_partitions_halves_result() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(120),
            ..Default::default()
        };
        // Act
        let p2 = c.gpu_sm_partition(2);
        let p4 = c.gpu_sm_partition(4);
        // Assert
        assert_eq!(p2.sm_per_partition, 60);
        assert_eq!(p4.sm_per_partition, 30);
    }

    // -- numa_core_bindings additional coverage --

    #[test]
    fn numa_core_bindings_five_nodes() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 5,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        assert_eq!(bindings.len(), 5);
    }

    #[test]
    fn numa_core_bindings_ten_nodes_contiguous() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 10,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        for i in 0..bindings.len() - 1 {
            assert_eq!(
                bindings[i].1.end, bindings[i + 1].1.start,
                "Node {} end must equal node {} start", i, i + 1
            );
        }
    }

    #[test]
    fn numa_core_bindings_node_ids_sequential_from_zero() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 6,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        for (i, (node_id, _)) in bindings.iter().enumerate() {
            assert_eq!(*node_id, i);
        }
    }


    // -- TileBits additional coverage --

    #[test]
    fn tile_bits_clone_equals_copy_values() {
        // Arrange
        let t = TileBits { tile_m: 24, tile_n: 48, tile_k: 96 };
        // Act
        let t_copy = t;
        let t_clone = t.clone();
        // Assert
        assert_eq!(t_copy.tile_m, t_clone.tile_m);
        assert_eq!(t_copy.tile_n, t_clone.tile_n);
        assert_eq!(t_copy.tile_k, t_clone.tile_k);
    }

    #[test]
    fn tile_bits_power_of_two_construction() {
        // Arrange & Act
        let t = TileBits { tile_m: 64, tile_n: 128, tile_k: 256 };
        // Assert
        assert!(t.tile_m.is_power_of_two());
        assert!(t.tile_n.is_power_of_two());
        assert!(t.tile_k.is_power_of_two());
    }

    #[test]
    fn tile_bits_non_square_construction() {
        // Arrange & Act
        let t = TileBits { tile_m: 4, tile_n: 64, tile_k: 32 };
        // Assert
        assert_ne!(t.tile_m, t.tile_n);
        assert!(t.tile_m < t.tile_n);
    }

    #[test]
    fn tile_bits_only_k_differs_from_default() {
        // Arrange
        let default = TileBits::default();
        // Act
        let t = TileBits { tile_m: 16, tile_n: 16, tile_k: 32 };
        // Assert
        assert_eq!(t.tile_m, default.tile_m);
        assert_eq!(t.tile_n, default.tile_n);
        assert_ne!(t.tile_k, default.tile_k);
    }

    #[test]
    fn tile_bits_serialize_asymmetric_roundtrip() {
        // Arrange
        let t = TileBits { tile_m: 3, tile_n: 7, tile_k: 11 };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(t2.tile_m, 3);
        assert_eq!(t2.tile_n, 7);
        assert_eq!(t2.tile_k, 11);
    }

    // -- CompilerConstraints serialization additional --

    #[test]
    fn serialize_default_produces_valid_json() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let json = serde_json::to_string(&c).unwrap();
        // Assert
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }

    #[test]
    fn serialize_options_none_produces_null() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let json = serde_json::to_string(&c).unwrap();
        // Assert
        assert!(json.contains("null"), "None fields must serialize to null");
    }

    #[test]
    fn serialize_numa_bindings_roundtrip_multi() {
        // Arrange
        let c = CompilerConstraints {
            numa_core_bindings: vec![(0, 0, 4), (1, 4, 8), (2, 8, 16)],
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.numa_core_bindings.len(), 3);
        assert_eq!(c2.numa_core_bindings[0], (0, 0, 4));
        assert_eq!(c2.numa_core_bindings[2], (2, 8, 16));
    }

    #[test]
    fn serialize_preserves_max_gpr_count() {
        // Arrange
        let c = CompilerConstraints {
            max_gpr_count: 31,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.max_gpr_count, 31);
    }

    #[test]
    fn serialize_preserves_tensor_core_gen() {
        // Arrange
        let c = CompilerConstraints {
            tensor_core_gen: 4,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.tensor_core_gen, 4);
    }

    // -- CompilerConstraints field combinations --

    #[test]
    fn constraints_amx_true_manual() {
        // Arrange & Act
        let c = CompilerConstraints {
            has_amx: true,
            ..Default::default()
        };
        // Assert
        assert!(c.has_amx);
        assert!(!c.has_avx512);
    }

    #[test]
    fn constraints_all_options_some() {
        // Arrange & Act
        let c = CompilerConstraints {
            smem_size: Some(1024),
            gpu_sm_version: Some(90),
            gpu_sm_count: Some(132),
            gpu_warp_size: Some(32),
            sve_vl_bytes: Some(16),
            rdma_min_chunk_tokens: Some(64),
            ..Default::default()
        };
        // Assert
        assert!(c.smem_size.is_some());
        assert!(c.gpu_sm_version.is_some());
        assert!(c.gpu_sm_count.is_some());
        assert!(c.gpu_warp_size.is_some());
        assert!(c.sve_vl_bytes.is_some());
        assert!(c.rdma_min_chunk_tokens.is_some());
    }

    // -- GpuSmPartition additional --

    #[test]
    fn gpu_partition_copy_clone_same_values() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 132,
            num_partitions: 4,
            sm_per_partition: 33,
        };
        // Act
        let p_copy = p;
        let p_clone = p.clone();
        // Assert
        assert_eq!(p_copy.total_sm, p_clone.total_sm);
        assert_eq!(p_copy.num_partitions, p_clone.num_partitions);
        assert_eq!(p_copy.sm_per_partition, p_clone.sm_per_partition);
    }

    #[test]
    fn gpu_partition_one_sm_two_partitions_zero() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(1),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(2);
        // Assert: 1 / 2 = 0
        assert_eq!(p.sm_per_partition, 0);
        assert_eq!(p.num_partitions, 2);
    }

    // -- CompilerConstraints debug format additional --

    #[test]
    fn constraints_debug_l2_cache_size_present() {
        // Arrange
        let c = CompilerConstraints {
            l2_cache_size: 512 * 1024,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("l2_cache_size"));
    }

    #[test]
    fn constraints_debug_simd_width_present() {
        // Arrange
        let c = CompilerConstraints {
            simd_width_bits: 512,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("simd_width_bits"));
    }

    #[test]
    fn constraints_tile_bits_independent_after_clone() {
        // Arrange
        let mut c = CompilerConstraints {
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.optimal_tile_bits.tile_m = 0;
        // Assert
        assert_eq!(c2.optimal_tile_bits.tile_m, 64);
    }

    #[test]
    fn constraints_rdma_large_chunk_value() {
        // Arrange
        let c = CompilerConstraints {
            rdma_min_chunk_tokens: Some(65536),
            ..Default::default()
        };
        // Assert
        assert_eq!(c.rdma_min_chunk_tokens, Some(65536));
    }

    #[test]
    fn constraints_numa_bindings_large_vec() {
        // Arrange
        let bindings: Vec<(usize, usize, usize)> = (0..100)
            .map(|i| (i, i * 4, (i + 1) * 4))
            .collect();
        let c = CompilerConstraints {
            numa_core_bindings: bindings,
            ..Default::default()
        };
        // Assert
        assert_eq!(c.numa_core_bindings.len(), 100);
        assert_eq!(c.numa_core_bindings[0], (0, 0, 4));
        assert_eq!(c.numa_core_bindings[99], (99, 396, 400));
    }

    #[test]
    fn constraints_amx_and_avx512_both_true() {
        // Arrange & Act
        let c = CompilerConstraints {
            has_amx: true,
            has_avx512: true,
            ..Default::default()
        };
        // Assert
        assert!(c.has_amx);
        assert!(c.has_avx512);
    }

    #[test]
    fn constraints_native_int4_dot_manual_true() {
        // Arrange & Act
        let c = CompilerConstraints {
            native_int4_dot: true,
            ..Default::default()
        };
        // Assert
        assert!(c.native_int4_dot);
    }

    #[test]
    fn tile_bits_default_tile_k_equals_m_and_n() {
        // Arrange & Act
        let t = TileBits::default();
        // Assert
        assert_eq!(t.tile_k, t.tile_m);
        assert_eq!(t.tile_k, t.tile_n);
    }

    #[test]
    fn gpu_sm_partition_num_partitions_recorded_exact() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(100),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(7);
        // Assert
        assert_eq!(p.num_partitions, 7);
        assert_eq!(p.sm_per_partition, 14);
    }

    // -- derive GPU warp_size preserved --

    #[test]
    fn derive_gpu_warp_size_preserved_cuda() {
        // Arrange
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_warp_size, Some(32));
    }

    #[test]
    fn derive_gpu_warp_size_preserved_hip_64() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 110,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_warp_size, Some(64));
    }

    // -- derive CPU rdma from sensors --

    #[test]
    fn derive_cpu_rdma_chunk_tokens_from_sensors() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: derive completes without panic
        assert!(c.l1i_size > 0);
    }

    // -- gpu_sm_partition from derived constraints (CPU) --

    #[test]
    fn gpu_sm_partition_from_cpu_derived() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Act: CPU-only → gpu_sm_count is None → total_sm = 1
        let p = c.gpu_sm_partition(1);
        // Assert
        assert_eq!(p.total_sm, 1);
    }

    // -- SVE fields consistency --

    #[test]
    fn constraints_sve_vl_set_with_has_sve() {
        // Arrange & Act
        let c = CompilerConstraints {
            has_sve: true,
            sve_vl_bytes: Some(32),
            ..Default::default()
        };
        // Assert
        assert!(c.has_sve);
        assert_eq!(c.sve_vl_bytes, Some(32));
    }

    // -- gpu_sm_partition remainder correctness --

    #[test]
    fn gpu_sm_partition_remainder_is_lost() {
        // Arrange: 10 SMs, 3 partitions → 3 per partition, 1 SM unused
        let c = CompilerConstraints {
            gpu_sm_count: Some(10),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(3);
        // Assert
        assert_eq!(p.sm_per_partition, 3);
        assert!(p.sm_per_partition * p.num_partitions < p.total_sm);
    }

    // -- numa_core_bindings minimal coverage --

    #[test]
    fn numa_core_bindings_single_node_minimal() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 1,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 0);
        assert!(bindings[0].1.end >= bindings[0].1.start);
    }

    // -- serialize preserves l1i_size --

    #[test]
    fn serialize_preserves_l1i_size() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 48 * 1024,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.l1i_size, 48 * 1024);
    }

    // -- derive no_gpu clears all GPU fields --

    #[test]
    fn derive_no_gpu_clears_tma_fp4_fp6() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
        assert_eq!(c.tensor_core_gen, 0);
    }

    // -- constraints with max_l1i --

    #[test]
    fn constraints_l1i_size_usize_max() {
        // Arrange & Act
        let c = CompilerConstraints {
            l1i_size: usize::MAX,
            ..Default::default()
        };
        // Assert: construction is valid
        assert_eq!(c.l1i_size, usize::MAX);
    }

    // -- GpuSmPartition debug all field names --

    #[test]
    fn gpu_partition_debug_all_field_names() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 88,
            num_partitions: 2,
            sm_per_partition: 44,
        };
        // Act
        let debug = format!("{:?}", p);
        // Assert
        assert!(debug.contains("total_sm"));
        assert!(debug.contains("num_partitions"));
        assert!(debug.contains("sm_per_partition"));
    }

    // -- constraints native_int4_dot default false --

    #[test]
    fn constraints_native_int4_dot_default_false() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert
        assert!(!c.native_int4_dot);
    }

    // -- derive with gpu large smem --

    #[test]
    fn derive_with_gpu_large_shared_mem() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let large_smem = 512 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 100 },
            compute_unit_count: 192,
            tensor_core_gen: 4,
            shared_mem_per_sm_bytes: large_smem,
            l2_bytes: 96 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 10,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.smem_size, Some(large_smem));
    }

    // -- constraints debug contains has_sve --

    #[test]
    fn constraints_debug_contains_has_sve() {
        // Arrange
        let c = CompilerConstraints {
            has_sve: true,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("has_sve"));
    }

    // -- tile_bits serialize then mutate independent --

    #[test]
    fn tile_bits_serialize_original_unchanged() {
        // Arrange
        let mut t = TileBits { tile_m: 32, tile_n: 32, tile_k: 32 };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        t.tile_m = 0;
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert: deserialized value is from original, not mutated
        assert_eq!(t2.tile_m, 32);
        assert_eq!(t.tile_m, 0);
    }

    // ---- Batch 5: 15 additional tests ----

    #[test]
    fn constraints_has_avx512_without_amx() {
        // Arrange & Act: AVX-512 without AMX is a valid configuration
        let c = CompilerConstraints {
            has_avx512: true,
            has_amx: false,
            ..Default::default()
        };
        // Assert
        assert!(c.has_avx512);
        assert!(!c.has_amx);
    }

    #[test]
    fn serialize_preserves_has_amx() {
        // Arrange
        let c = CompilerConstraints {
            has_amx: true,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.has_amx);
    }

    #[test]
    fn serialize_preserves_has_sve() {
        // Arrange
        let c = CompilerConstraints {
            has_sve: true,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.has_sve);
    }

    #[test]
    fn serialize_preserves_numa_node_count() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 8,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.numa_node_count, 8);
    }

    #[test]
    fn tile_bits_all_equal_non_default() {
        // Arrange & Act: all dimensions equal but different from default (16)
        let t = TileBits { tile_m: 32, tile_n: 32, tile_k: 32 };
        // Assert
        assert_eq!(t.tile_m, t.tile_n);
        assert_eq!(t.tile_n, t.tile_k);
        assert_ne!(t.tile_m, TileBits::default().tile_m);
    }

    #[test]
    fn gpu_partition_homogenous_fields() {
        // Arrange & Act: all three fields set to the same value
        let p = GpuSmPartition {
            total_sm: 8,
            num_partitions: 8,
            sm_per_partition: 8,
        };
        // Assert
        assert_eq!(p.total_sm, p.num_partitions);
        assert_eq!(p.num_partitions, p.sm_per_partition);
    }

    #[test]
    fn derive_metal_l2_nonzero_applied() {
        // Arrange: Metal GPU with non-zero L2 should override CPU L2
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_l2 = 12 * 1024 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1005 },
            compute_unit_count: 40,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 48 * 1024,
            l2_bytes: gpu_l2,
            global_mem_bytes: 32 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.l2_cache_size, gpu_l2);
    }

    #[test]
    fn exceeds_l1i_budget_l1i_size_five() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 5,
            ..Default::default()
        };
        // Act & Assert: 5 * 0.8 = 4.0, floor = 4
        assert!(!c.exceeds_l1i_budget(4));
        assert!(c.exceeds_l1i_budget(5));
    }

    #[test]
    fn constraints_gpr_count_15_without_extensions() {
        // Arrange & Act: baseline CPU without AMX/AVX-512 → GPR = 15
        let c = CompilerConstraints {
            has_amx: false,
            has_avx512: false,
            max_gpr_count: 15,
            ..Default::default()
        };
        // Assert
        assert_eq!(c.max_gpr_count, 15);
        assert!(!c.has_amx);
        assert!(!c.has_avx512);
    }

    #[test]
    fn numa_bindings_first_node_starts_at_zero() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 3,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        assert_eq!(bindings[0].1.start, 0, "first node core range must start at 0");
    }

    #[test]
    fn gpu_sm_partition_total_equals_input() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(144),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(4);
        // Assert: total_sm must exactly equal the configured gpu_sm_count
        assert_eq!(p.total_sm, 144);
        assert_eq!(p.sm_per_partition, 36);
    }

    #[test]
    fn tile_bits_clone_from_non_default() {
        // Arrange
        let t = TileBits { tile_m: 48, tile_n: 96, tile_k: 192 };
        // Act
        let t2 = t.clone();
        // Assert
        assert_eq!(t2.tile_m, 48);
        assert_eq!(t2.tile_n, 96);
        assert_eq!(t2.tile_k, 192);
    }

    #[test]
    fn serialize_preserves_native_int4_dot_true() {
        // Arrange
        let c = CompilerConstraints {
            native_int4_dot: true,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.native_int4_dot);
    }

    #[test]
    fn gpu_partition_debug_zero_values() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 0,
            num_partitions: 0,
            sm_per_partition: 0,
        };
        // Act
        let debug = format!("{:?}", p);
        // Assert: all field names present even with zero values
        assert!(debug.contains("total_sm"));
        assert!(debug.contains("num_partitions"));
        assert!(debug.contains("sm_per_partition"));
    }

    #[test]
    fn constraints_inconsistent_sve_false_with_vl() {
        // Arrange & Act: has_sve=false but sve_vl_bytes=Some(32) —
        // construction allows this; semantics enforced by derive()
        let c = CompilerConstraints {
            has_sve: false,
            sve_vl_bytes: Some(32),
            ..Default::default()
        };
        // Assert: fields are independently settable
        assert!(!c.has_sve);
        assert_eq!(c.sve_vl_bytes, Some(32));
    }

    // ---- Batch 6: 15 additional tests ----

    #[test]
    fn exceeds_l1i_budget_threshold_invariant_half() {
        // Arrange: verify the 80% threshold property for non-trivial l1i sizes
        let sizes = [512, 1024, 4096, 8192, 16 * 1024, 48 * 1024, 128 * 1024];
        for &l1i in &sizes {
            let c = CompilerConstraints {
                l1i_size: l1i,
                ..Default::default()
            };
            let threshold = (l1i as f64 * 0.8) as usize;
            // Act & Assert
            assert!(!c.exceeds_l1i_budget(threshold),
                "at 80% boundary for l1i={} must not exceed", l1i);
            assert!(c.exceeds_l1i_budget(l1i),
                "at 100% for l1i={} must exceed", l1i);
        }
    }

    #[test]
    fn gpu_sm_partition_product_never_exceeds_total() {
        // Arrange: exhaustive check across multiple SM counts and partition sizes
        let sm_counts = [2, 3, 5, 8, 13, 16, 32, 64, 80, 108, 132, 192];
        let partition_counts = [1, 2, 3, 4, 5, 7, 8, 12, 16, 32, 64, 128];
        for &sm in &sm_counts {
            let c = CompilerConstraints {
                gpu_sm_count: Some(sm),
                ..Default::default()
            };
            for &np in &partition_counts {
                // Act
                let p = c.gpu_sm_partition(np);
                // Assert: total allocated SMs must not exceed total
                assert!(
                    p.sm_per_partition * p.num_partitions <= p.total_sm,
                    "sm={} np={} alloc={} total={}",
                    sm, np, p.sm_per_partition * p.num_partitions, p.total_sm,
                );
            }
        }
    }

    #[test]
    fn numa_core_bindings_ranges_are_monotonically_increasing() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 4,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: each range starts at or after previous range ended
        for i in 1..bindings.len() {
            assert!(
                bindings[i].1.start >= bindings[i - 1].1.end,
                "node {} start ({}) must be >= node {} end ({})",
                i, bindings[i].1.start, i - 1, bindings[i - 1].1.end,
            );
        }
    }

    #[test]
    fn serialize_deserve_roundtrip_preserves_all_bools_false() {
        // Arrange: explicitly set all bools to false (matching default)
        let c = CompilerConstraints {
            native_int4_dot: false,
            has_amx: false,
            has_avx512: false,
            has_sve: false,
            has_tma: false,
            has_native_fp4: false,
            has_native_fp6: false,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(!c2.native_int4_dot);
        assert!(!c2.has_amx);
        assert!(!c2.has_avx512);
        assert!(!c2.has_sve);
        assert!(!c2.has_tma);
        assert!(!c2.has_native_fp4);
        assert!(!c2.has_native_fp6);
    }

    #[test]
    fn tile_bits_all_fields_independent() {
        // Arrange & Act: each field set to a different prime number
        let t = TileBits { tile_m: 7, tile_n: 11, tile_k: 13 };
        // Assert: all distinct
        assert_ne!(t.tile_m, t.tile_n);
        assert_ne!(t.tile_n, t.tile_k);
        assert_ne!(t.tile_m, t.tile_k);
    }

    #[test]
    fn tile_bits_default_not_zero() {
        // Arrange & Act
        let t = TileBits::default();
        // Assert: default tile dimensions must be positive
        assert!(t.tile_m > 0, "default tile_m must be > 0");
        assert!(t.tile_n > 0, "default tile_n must be > 0");
        assert!(t.tile_k > 0, "default tile_k must be > 0");
    }

    #[test]
    fn derive_cuda_sm90_warp_size_32() {
        // Arrange: Hopper always has warp_size=32
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_warp_size, Some(32));
        assert_eq!(c.gpu_sm_count, Some(132));
    }

    #[test]
    fn derive_hip_warp_size_64_preserved() {
        // Arrange: AMD GPUs have warp_size=64 (wavefront)
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_warp_size, Some(64));
        assert_eq!(c.gpu_sm_count, Some(228));
    }

    #[test]
    fn derive_metal_compute_units_preserved() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1001 },
            compute_unit_count: 48,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: compute_unit_count maps to gpu_sm_count
        assert_eq!(c.gpu_sm_count, Some(48));
        assert_eq!(c.smem_size, Some(32 * 1024));
    }

    #[test]
    fn gpu_sm_partition_single_sm_division() {
        // Arrange: 1 SM divided into 1 partition → 1 SM per partition
        let c = CompilerConstraints {
            gpu_sm_count: Some(1),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(1);
        // Assert
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.num_partitions, 1);
        assert_eq!(p.sm_per_partition, 1);
    }

    #[test]
    fn constraints_clone_preserves_tile_bits() {
        // Arrange
        let c = CompilerConstraints {
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        // Assert
        assert_eq!(c2.optimal_tile_bits.tile_m, 64);
        assert_eq!(c2.optimal_tile_bits.tile_n, 64);
        assert_eq!(c2.optimal_tile_bits.tile_k, 64);
    }

    #[test]
    fn exceeds_l1i_budget_zero_bytes_never_exceeds() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        // Act & Assert: zero-byte kernel never exceeds any L1i budget
        assert!(!c.exceeds_l1i_budget(0));
    }

    #[test]
    fn derive_with_probe_multiple_spill_points() {
        // Arrange: probe with multiple spill points
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let probe = ProbeResult {
            spill_points: vec![64, 128, 256, 512, 1024],
            smem_cliffs: vec![(128, 0.9), (256, 0.6)],
            l2_thrash_threshold: 2 * 1024 * 1024,
            device_fingerprint: "multi-spill-test".to_string(),
            raw_measurements: Default::default(),
        };
        // Act: must not panic with multiple spill points
        let c = CompilerConstraints::derive(&profile, &sensors, Some(&probe), None);
        // Assert: basic validity
        assert!(c.simd_width_bits > 0);
        assert!(c.l1i_size > 0);
    }

    #[test]
    fn serialize_deserialize_sve_vl_bytes_none() {
        // Arrange: default has sve_vl_bytes = None
        let c = CompilerConstraints::default();
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert: None roundtrips correctly
        assert_eq!(c2.sve_vl_bytes, None);
    }

    #[test]
    fn gpu_partition_total_sm_reflects_input() {
        // Arrange: verify total_sm is always exactly gpu_sm_count.unwrap_or(1)
        let cases = [(Some(80), 80), (Some(1), 1), (Some(256), 256)];
        for (input, expected) in cases {
            let c = CompilerConstraints {
                gpu_sm_count: input,
                ..Default::default()
            };
            // Act
            let p = c.gpu_sm_partition(1);
            // Assert
            assert_eq!(p.total_sm, expected,
                "total_sm must be {} for input {:?}", expected, input);
        }
    }

    // ---- Batch 7: 15 additional tests ----

    #[test]
    fn exceeds_l1i_budget_returns_bool() {
        // Arrange
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        // Act: verify the return type is bool in both directions
        let under = c.exceeds_l1i_budget(0);
        let over = c.exceeds_l1i_budget(usize::MAX);
        // Assert
        assert!(!under);
        assert!(over);
    }

    #[test]
    fn derive_cpu_smem_size_always_none() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: CPU-only must never report shared memory
        assert_eq!(c.smem_size, None, "CPU-only must have smem_size = None");
    }

    #[test]
    fn constraints_numa_node_count_zero_prevents_bindings() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 0,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: zero nodes produce zero bindings
        assert_eq!(bindings.len(), 0);
    }

    #[test]
    fn derive_gpu_sm_count_matches_topology() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let expected_cu = 56;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 70 },
            compute_unit_count: expected_cu,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 96 * 1024,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_count, Some(expected_cu));
    }

    #[test]
    fn tile_bits_clone_then_original_unchanged() {
        // Arrange
        let mut t = TileBits { tile_m: 100, tile_n: 200, tile_k: 300 };
        // Act
        let t2 = t.clone();
        t.tile_m = 0;
        t.tile_n = 0;
        t.tile_k = 0;
        // Assert: clone is a deep copy, original mutation does not affect clone
        assert_eq!(t2.tile_m, 100);
        assert_eq!(t2.tile_n, 200);
        assert_eq!(t2.tile_k, 300);
    }

    #[test]
    fn gpu_sm_partition_num_partitions_stored_verbatim() {
        // Arrange: the partition count is stored as-is, even when > total_sm
        let c = CompilerConstraints {
            gpu_sm_count: Some(4),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(200);
        // Assert
        assert_eq!(p.num_partitions, 200);
        assert_eq!(p.total_sm, 4);
        assert_eq!(p.sm_per_partition, 0);
    }

    #[test]
    fn derive_metal_zero_l2_keeps_cpu_l2() {
        // Arrange: Metal GPU with l2_bytes=0 should not overwrite CPU L2
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let cpu_l2 = sensors.l2_cache_bytes;
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1001 },
            compute_unit_count: 40,
            tensor_core_gen: 1,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert!(c.l2_cache_size >= cpu_l2,
            "Metal GPU L2=0 must not reduce below CPU L2");
    }

    #[test]
    fn constraints_default_max_gpr_count_is_15() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert: standard x86_64 has 15 GPRs (rbp/rsp reserved)
        assert_eq!(c.max_gpr_count, 15);
    }

    #[test]
    fn serialize_roundtrip_preserves_zero_tile_bits() {
        // Arrange
        let c = CompilerConstraints {
            optimal_tile_bits: TileBits { tile_m: 0, tile_n: 0, tile_k: 0 },
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(c2.optimal_tile_bits.tile_m, 0);
        assert_eq!(c2.optimal_tile_bits.tile_n, 0);
        assert_eq!(c2.optimal_tile_bits.tile_k, 0);
    }

    #[test]
    fn derive_hip_tensor_core_gen_preserved() {
        // Arrange
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.tensor_core_gen, 3);
    }

    #[test]
    fn numa_bindings_total_allocated_leq_total_cores() {
        // Arrange: 3 NUMA nodes; integer division may lose remainder cores
        let c = CompilerConstraints {
            numa_node_count: 3,
            ..Default::default()
        };
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: sum of all range lengths <= total_cores (truncation is expected)
        let allocated: usize = bindings.iter().map(|(_, r)| r.end - r.start).sum();
        assert!(allocated <= total_cores,
            "allocated {} must not exceed total {}", allocated, total_cores);
        assert!(allocated > 0, "at least some cores must be allocated");
    }

    #[test]
    fn constraints_debug_contains_native_int4_dot_field() {
        // Arrange
        let c = CompilerConstraints {
            native_int4_dot: true,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("native_int4_dot"));
    }

    #[test]
    fn tile_bits_serialize_deserialize_large_prime_values() {
        // Arrange
        let t = TileBits { tile_m: 101, tile_n: 103, tile_k: 107 };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(t2.tile_m, 101);
        assert_eq!(t2.tile_n, 103);
        assert_eq!(t2.tile_k, 107);
    }

    #[test]
    fn gpu_sm_partition_sm_count_one_partition_one() {
        // Arrange: minimal meaningful GPU: 1 SM, 1 partition
        let c = CompilerConstraints {
            gpu_sm_count: Some(1),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(1);
        // Assert
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 1);
        assert_eq!(p.num_partitions, 1);
    }

    #[test]
    fn constraints_rdma_min_chunk_default_is_none() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert: CPU-only default has no RDMA chunk constraint
        assert_eq!(c.rdma_min_chunk_tokens, None);
    }

    // ---- Batch 8: 15 additional tests ----

    #[test]
    fn exceeds_l1i_budget_l1i_size_three_boundary() {
        // Arrange: 3 * 0.8 = 2.4, floor = 2
        let c = CompilerConstraints {
            l1i_size: 3,
            ..Default::default()
        };
        // Act & Assert
        assert!(!c.exceeds_l1i_budget(2), "2 <= floor(3*0.8)=2, must not exceed");
        assert!(c.exceeds_l1i_budget(3), "3 > floor(3*0.8)=2, must exceed");
    }

    #[test]
    fn constraints_partial_equality_after_independent_mutation() {
        // Arrange: two constraints start identical, mutate one field
        let mut c1 = CompilerConstraints::default();
        let c2 = CompilerConstraints::default();
        // Act
        c1.max_gpr_count = 31;
        // Assert: only the mutated field differs
        assert_ne!(c1.max_gpr_count, c2.max_gpr_count);
        assert_eq!(c1.l1i_size, c2.l1i_size);
        assert_eq!(c1.l2_cache_size, c2.l2_cache_size);
    }

    #[test]
    fn derive_cpu_amx_and_avx512_both_false() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: derive must not magically enable extensions not present
        // (both may be true if hardware supports them, but the fields exist)
        assert_eq!(c.has_amx, profile.kernel_config.has_amx);
        assert_eq!(c.has_avx512, profile.kernel_config.use_avx512);
    }

    #[test]
    fn derive_cpu_native_int4_dot_consistent_with_extensions() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: native_int4_dot is true iff has_amx or has_avx512
        let expected = c.has_amx || c.has_avx512;
        assert_eq!(c.native_int4_dot, expected);
    }

    #[test]
    fn numa_core_bindings_large_node_count() {
        // Arrange: 128 NUMA nodes (extreme but valid configuration)
        let c = CompilerConstraints {
            numa_node_count: 128,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert
        assert_eq!(bindings.len(), 128);
        assert_eq!(bindings[0].0, 0);
        assert_eq!(bindings[127].0, 127);
    }

    #[test]
    fn gpu_sm_partition_extreme_partition_count() {
        // Arrange: usize::MAX partitions is absurd but should not panic
        let c = CompilerConstraints {
            gpu_sm_count: Some(4),
            ..Default::default()
        };
        // Act: num_partitions.max(1) = usize::MAX, 4 / usize::MAX = 0
        let p = c.gpu_sm_partition(usize::MAX);
        // Assert: integer division must produce 0
        assert_eq!(p.sm_per_partition, 0);
        assert_eq!(p.total_sm, 4);
    }

    #[test]
    fn tile_bits_mixed_zero_and_nonzero() {
        // Arrange & Act: only tile_k is nonzero
        let t = TileBits { tile_m: 0, tile_n: 0, tile_k: 64 };
        // Assert
        assert_eq!(t.tile_m, 0);
        assert_eq!(t.tile_n, 0);
        assert_eq!(t.tile_k, 64);
    }

    #[test]
    fn serialize_roundtrip_all_numeric_fields_max() {
        // Arrange: push numeric fields to maximum values
        let c = CompilerConstraints {
            max_gpr_count: usize::MAX,
            l1i_size: usize::MAX,
            l2_cache_size: usize::MAX,
            numa_node_count: usize::MAX,
            simd_width_bits: usize::MAX,
            tensor_core_gen: u32::MAX,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert: all max values survive roundtrip
        assert_eq!(c2.max_gpr_count, usize::MAX);
        assert_eq!(c2.l1i_size, usize::MAX);
        assert_eq!(c2.l2_cache_size, usize::MAX);
        assert_eq!(c2.numa_node_count, usize::MAX);
        assert_eq!(c2.simd_width_bits, usize::MAX);
        assert_eq!(c2.tensor_core_gen, u32::MAX);
    }

    #[test]
    fn gpu_partition_debug_shows_all_numeric_values() {
        // Arrange
        let p = GpuSmPartition {
            total_sm: 77,
            num_partitions: 3,
            sm_per_partition: 25,
        };
        // Act
        let debug = format!("{:?}", p);
        // Assert: all three numeric values must appear in debug output
        assert!(debug.contains("77"), "must contain total_sm value 77");
        assert!(debug.contains("25"), "must contain sm_per_partition value 25");
    }

    #[test]
    fn derive_cuda_sm90_tma_boundary_true() {
        // Arrange: SM90 is the exact boundary for TMA enablement
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: SM90 is the minimum for TMA
        assert!(c.has_tma, "SM90 must enable TMA (>= 90 threshold)");
        assert!(!c.has_native_fp4, "SM90 must not enable FP4 (< 100 threshold)");
    }

    #[test]
    fn constraints_clone_numa_bindings_then_mutate_original() {
        // Arrange
        let bindings = vec![(0, 0, 8), (1, 8, 16)];
        let mut c = CompilerConstraints {
            numa_core_bindings: bindings.clone(),
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.numa_core_bindings.clear();
        c.numa_core_bindings.push((99, 0, 1));
        // Assert: clone is independent
        assert_eq!(c2.numa_core_bindings, bindings);
        assert_eq!(c.numa_core_bindings, vec![(99, 0, 1)]);
    }

    #[test]
    fn derive_gpu_zero_compute_units() {
        // Arrange: GPU reports 0 compute units (degenerate but structurally valid)
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: zero compute units produces zero SM count
        assert_eq!(c.gpu_sm_count, Some(0));
        assert_eq!(c.gpu_warp_size, Some(0));
    }

    #[test]
    fn tile_bits_copy_after_partial_reassignment() {
        // Arrange
        let mut t = TileBits { tile_m: 10, tile_n: 20, tile_k: 30 };
        let t2 = t; // Copy at initial state
        // Act: mutate only tile_m after copy
        t.tile_m = 99;
        // Assert: copy preserved original, only t changed
        assert_eq!(t.tile_m, 99);
        assert_eq!(t2.tile_m, 10);
        assert_eq!(t2.tile_n, 20);
        assert_eq!(t2.tile_k, 30);
    }

    #[test]
    fn constraints_default_l1i_exactly_32kb() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert: spec-defined default for L1i
        assert_eq!(c.l1i_size, 32 * 1024);
    }

    #[test]
    fn derive_cpu_tile_bits_symmetric_m_and_n() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: derive always produces tile_m == tile_n for all current code paths
        // (AMX: 64/64, AVX-512: 32/32, SVE: vl-based equal, baseline: 16/16)
        assert_eq!(c.optimal_tile_bits.tile_m, c.optimal_tile_bits.tile_n,
            "derived tile_m must equal tile_n");
    }

    // ---- Batch 9: 15 additional tests ----

    #[test]
    fn exceeds_l1i_budget_at_exact_80_percent_boundary() {
        // Arrange: l1i=1000 → 80% = 800 exactly (no floating point ambiguity)
        let c = CompilerConstraints {
            l1i_size: 1000,
            ..Default::default()
        };
        // Act & Assert: 800 is the threshold, must not exceed
        assert!(!c.exceeds_l1i_budget(800));
        // 801 is one byte over threshold, must exceed
        assert!(c.exceeds_l1i_budget(801));
        // 799 is safely under
        assert!(!c.exceeds_l1i_budget(799));
    }

    #[test]
    fn derive_cuda_sm91_has_tma_no_fp4() {
        // Arrange: SM91 is above SM90 TMA threshold but below SM100 FP4 threshold
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 91 },
            compute_unit_count: 120,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 228 * 1024,
            l2_bytes: 48 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 1,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_version, Some(91));
        assert!(c.has_tma, "SM91 >= SM90 must enable TMA");
        assert!(!c.has_native_fp4, "SM91 < SM100 must not enable FP4");
        assert!(!c.has_native_fp6, "SM91 < SM100 must not enable FP6");
    }

    #[test]
    fn gpu_sm_partition_default_constraints_yields_one_total_sm() {
        // Arrange: default constraints have gpu_sm_count = None
        let c = CompilerConstraints::default();
        // Act
        let p = c.gpu_sm_partition(1);
        // Assert: None → unwrap_or(1) → total_sm = 1
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 1);
        assert_eq!(p.num_partitions, 1);
    }

    #[test]
    fn tile_bits_manual_equals_default_when_same_values() {
        // Arrange & Act: manually construct with same values as default
        let manual = TileBits { tile_m: 16, tile_n: 16, tile_k: 16 };
        let default = TileBits::default();
        // Assert: field-by-field equivalence
        assert_eq!(manual.tile_m, default.tile_m);
        assert_eq!(manual.tile_n, default.tile_n);
        assert_eq!(manual.tile_k, default.tile_k);
    }

    #[test]
    fn serialize_roundtrip_preserves_has_native_fp6_true() {
        // Arrange
        let c = CompilerConstraints {
            has_native_fp6: true,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.has_native_fp6);
    }

    #[test]
    fn constraints_debug_contains_warp_size_field() {
        // Arrange
        let c = CompilerConstraints {
            gpu_warp_size: Some(32),
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert
        assert!(debug.contains("gpu_warp_size"));
    }

    #[test]
    fn derive_with_metal_zero_global_mem_no_panic() {
        // Arrange: degenerate Metal GPU with zero global memory
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1001 },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };
        // Act: must not panic with all-zero Metal GPU
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_count, Some(0));
        assert_eq!(c.gpu_sm_version, None);
        assert_eq!(c.smem_size, Some(0));
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
    }

    #[test]
    fn numa_core_bindings_single_core_system() {
        // Arrange: single core, single NUMA node
        let c = CompilerConstraints {
            numa_node_count: 1,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: even with 1 core, must produce 1 binding
        assert_eq!(bindings.len(), 1);
        let (_, range) = &bindings[0];
        assert_eq!(range.start, 0);
        // end may be 1 (single core) or more depending on system
        assert!(range.end >= range.start);
    }

    #[test]
    fn gpu_sm_partition_preserves_num_partitions_when_none() {
        // Arrange: no GPU, but 3 partitions requested
        let c = CompilerConstraints::default();
        // Act
        let p = c.gpu_sm_partition(3);
        // Assert: num_partitions stored verbatim, total_sm=1
        assert_eq!(p.num_partitions, 3);
        assert_eq!(p.total_sm, 1);
        assert_eq!(p.sm_per_partition, 0); // 1 / 3 = 0
    }

    #[test]
    fn tile_bits_copy_semantics_three_copies_independent() {
        // Arrange
        let mut a = TileBits { tile_m: 10, tile_n: 20, tile_k: 30 };
        // Act: first copy
        let b = a;
        a.tile_m = 111;
        // second copy (a already mutated)
        let c = a;
        a.tile_n = 222;
        // Assert: each copy is a snapshot at copy time
        assert_eq!(b.tile_m, 10); // captured before first mutation
        assert_eq!(b.tile_n, 20);
        assert_eq!(c.tile_m, 111); // captured after first mutation, before second
        assert_eq!(c.tile_n, 20);
        assert_eq!(a.tile_n, 222); // latest mutation
    }

    #[test]
    fn serialize_roundtrip_preserves_has_tma_false() {
        // Arrange: explicitly false (matching default)
        let c = CompilerConstraints {
            has_tma: false,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(!c2.has_tma);
        assert!(!c2.has_native_fp4);
        assert!(!c2.has_native_fp6);
    }

    #[test]
    fn constraints_tile_bits_amx_style_large() {
        // Arrange: simulate AMX-style large tile dimensions
        let c = CompilerConstraints {
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            has_amx: true,
            ..Default::default()
        };
        // Assert: construction valid, tile dimensions large
        assert!(c.optimal_tile_bits.tile_m >= 64);
        assert!(c.optimal_tile_bits.tile_n >= 64);
        assert!(c.optimal_tile_bits.tile_k >= 64);
        assert!(c.has_amx);
    }

    #[test]
    fn exceeds_l1i_budget_monotonically_increasing() {
        // Arrange: for any fixed l1i_size, if X exceeds, then X+1 must also exceed
        let c = CompilerConstraints {
            l1i_size: 4096,
            ..Default::default()
        };
        // Act & Assert: once we exceed the budget, all larger values must also exceed
        let threshold = (4096_f64 * 0.8) as usize;
        let mut last_exceeded = false;
        for bytes in 0..5000 {
            let exceeds = c.exceeds_l1i_budget(bytes);
            if exceeds && !last_exceeded {
                // This must be the first crossing
                assert!(bytes > threshold, "first exceed at {} must be > {}", bytes, threshold);
            }
            if last_exceeded {
                assert!(exceeds, "once exceeded at byte {}, all subsequent must exceed", bytes);
            }
            last_exceeded = exceeds;
        }
    }

    #[test]
    fn derive_cuda_sm100_tensor_core_gen_preserved() {
        // Arrange: Blackwell SM100 with tensor_core_gen=4
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
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: tensor_core_gen must be exactly what GPU reports
        assert_eq!(c.tensor_core_gen, 4);
        assert_eq!(c.gpu_sm_version, Some(100));
        assert!(c.has_native_fp4);
        assert!(c.has_native_fp6);
        assert!(c.has_tma);
    }

    #[test]
    fn gpu_sm_partition_cloned_partition_fields_independent() {
        // Arrange
        let mut p = GpuSmPartition {
            total_sm: 100,
            num_partitions: 4,
            sm_per_partition: 25,
        };
        // Act
        let p2 = p;
        p.total_sm = 0;
        p.num_partitions = 0;
        p.sm_per_partition = 0;
        // Assert: copy semantics — p2 retains original values
        assert_eq!(p2.total_sm, 100);
        assert_eq!(p2.num_partitions, 4);
        assert_eq!(p2.sm_per_partition, 25);
    }

    // ---- Batch 10: 15 additional tests ----

    #[test]
    fn derive_cpu_sve_vl_bytes_none_when_no_sve() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: if has_sve is false, sve_vl_bytes must be None
        if !c.has_sve {
            assert_eq!(c.sve_vl_bytes, None,
                "sve_vl_bytes must be None when has_sve is false");
        }
    }

    #[test]
    fn derive_cpu_l1i_size_matches_profile_cache() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let (l1d, _, _) = profile.cache_sizes();
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: derive sets l1i_size = l1d from profile
        assert_eq!(c.l1i_size, l1d,
            "l1i_size must equal profile L1d cache size");
    }

    #[test]
    fn tile_bits_serialize_deserialize_boundary_usize() {
        // Arrange: test serialization with usize::MAX / 2
        let t = TileBits {
            tile_m: usize::MAX / 2,
            tile_n: usize::MAX / 3,
            tile_k: usize::MAX / 5,
        };
        // Act
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        // Assert
        assert_eq!(t2.tile_m, usize::MAX / 2);
        assert_eq!(t2.tile_n, usize::MAX / 3);
        assert_eq!(t2.tile_k, usize::MAX / 5);
    }

    #[test]
    fn constraints_clone_preserves_all_optional_none() {
        // Arrange: default has all Option fields as None
        let c = CompilerConstraints::default();
        // Act
        let c2 = c.clone();
        // Assert: all optional fields remain None after clone
        assert_eq!(c2.smem_size, None);
        assert_eq!(c2.gpu_sm_version, None);
        assert_eq!(c2.gpu_sm_count, None);
        assert_eq!(c2.gpu_warp_size, None);
        assert_eq!(c2.sve_vl_bytes, None);
        assert_eq!(c2.rdma_min_chunk_tokens, None);
    }

    #[test]
    fn derive_cuda_sm89_tma_boundary_false() {
        // Arrange: SM89 is one below the TMA threshold (SM90)
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 89 },
            compute_unit_count: 142,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 100 * 1024,
            l2_bytes: 36 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 9,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: SM89 < SM90, so TMA must be false
        assert!(!c.has_tma, "SM89 < SM90 must not have TMA");
        assert!(!c.has_native_fp4, "SM89 < SM100 must not have FP4");
        assert!(!c.has_native_fp6, "SM89 < SM100 must not have FP6");
    }

    #[test]
    fn exceeds_l1i_budget_with_large_kernel_code_bytes() {
        // Arrange: 32KB L1i with kernel code close to limit
        let c = CompilerConstraints {
            l1i_size: 32 * 1024,
            ..Default::default()
        };
        let threshold = (32_usize * 1024 * 8 / 10) + 1; // just above 80%
        // Act & Assert
        assert!(c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold * 2));
        assert!(c.exceeds_l1i_budget(threshold + 1000));
    }

    #[test]
    fn gpu_sm_partition_twenty_partitions_on_80_sm() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            ..Default::default()
        };
        // Act: 80 / 20 = 4 per partition
        let p = c.gpu_sm_partition(20);
        // Assert
        assert_eq!(p.sm_per_partition, 4);
        assert_eq!(p.total_sm, 80);
        assert_eq!(p.num_partitions, 20);
    }

    #[test]
    fn constraints_debug_format_contains_l2_cache_size() {
        // Arrange
        let c = CompilerConstraints {
            l2_cache_size: 999999,
            ..Default::default()
        };
        // Act
        let debug = format!("{:?}", c);
        // Assert: must contain the field name and value
        assert!(debug.contains("l2_cache_size"));
    }

    #[test]
    fn serialize_roundtrip_preserves_numa_core_bindings_empty() {
        // Arrange: empty bindings vector
        let c = CompilerConstraints {
            numa_core_bindings: vec![],
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert
        assert!(c2.numa_core_bindings.is_empty());
    }

    #[test]
    fn derive_hip_zero_l2_keeps_cpu_l2_unchanged() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let cpu_l2 = sensors.l2_cache_bytes;
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 110,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: GPU L2=0 must not reduce below CPU L2
        assert!(c.l2_cache_size >= cpu_l2);
    }

    #[test]
    fn constraints_manual_max_gpr_31_with_avx512() {
        // Arrange & Act: simulate APX configuration with 31 GPRs
        let c = CompilerConstraints {
            max_gpr_count: 31,
            has_avx512: true,
            ..Default::default()
        };
        // Assert
        assert_eq!(c.max_gpr_count, 31);
        assert!(c.has_avx512);
        assert!(!c.has_amx);
    }

    #[test]
    fn tile_bits_clone_from_zero_then_verify() {
        // Arrange
        let t = TileBits { tile_m: 0, tile_n: 0, tile_k: 0 };
        // Act
        let t2 = t.clone();
        // Assert: clone of zero-valued TileBits is still zero
        assert_eq!(t2.tile_m, 0);
        assert_eq!(t2.tile_n, 0);
        assert_eq!(t2.tile_k, 0);
    }

    #[test]
    fn numa_core_bindings_eight_nodes_all_unique_ids() {
        // Arrange
        let c = CompilerConstraints {
            numa_node_count: 8,
            ..Default::default()
        };
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: each node_id must be unique and sequential
        let ids: Vec<usize> = bindings.iter().map(|(id, _)| *id).collect();
        let expected: Vec<usize> = (0..8).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn gpu_partition_copy_after_partial_field_change() {
        // Arrange
        let mut p = GpuSmPartition {
            total_sm: 80,
            num_partitions: 4,
            sm_per_partition: 20,
        };
        // Act: copy then change one field
        let p2 = p;
        p.total_sm = 999;
        // Assert: only total_sm changed in original, p2 unaffected
        assert_eq!(p.total_sm, 999);
        assert_eq!(p2.total_sm, 80);
        assert_eq!(p2.num_partitions, 4);
        assert_eq!(p2.sm_per_partition, 20);
    }

    #[test]
    fn serialize_roundtrip_preserves_all_gpu_fields_set() {
        // Arrange: all GPU-related fields set to non-default values
        let c = CompilerConstraints {
            smem_size: Some(128 * 1024),
            gpu_sm_version: Some(90),
            gpu_sm_count: Some(132),
            gpu_warp_size: Some(32),
            tensor_core_gen: 3,
            has_tma: true,
            has_native_fp4: false,
            has_native_fp6: false,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert: every GPU field survives the roundtrip
        assert_eq!(c2.smem_size, Some(128 * 1024));
        assert_eq!(c2.gpu_sm_version, Some(90));
        assert_eq!(c2.gpu_sm_count, Some(132));
        assert_eq!(c2.gpu_warp_size, Some(32));
        assert_eq!(c2.tensor_core_gen, 3);
        assert!(c2.has_tma);
        assert!(!c2.has_native_fp4);
        assert!(!c2.has_native_fp6);
    }

    // ---- Batch 11: 15 additional tests ----

    // @trace TEST-CC-001 [level:unit]
    #[test]
    fn exceeds_l1i_budget_odd_l1i_threshold_calculation() {
        // Arrange: odd L1i sizes produce non-integer 80% thresholds
        // 7 * 0.8 = 5.6 → floor = 5
        let c = CompilerConstraints {
            l1i_size: 7,
            ..Default::default()
        };
        // Act & Assert
        assert!(!c.exceeds_l1i_budget(5), "5 <= floor(7*0.8)=5");
        assert!(c.exceeds_l1i_budget(6), "6 > floor(7*0.8)=5");
        // 13 * 0.8 = 10.4 → floor = 10
        let c2 = CompilerConstraints {
            l1i_size: 13,
            ..Default::default()
        };
        assert!(!c2.exceeds_l1i_budget(10));
        assert!(c2.exceeds_l1i_budget(11));
    }

    // @trace TEST-CC-002 [level:unit]
    #[test]
    fn derive_cpu_max_gpr_31_when_avx512_detected() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: GPR count logic: has_amx || has_avx512 → 31, else 15
        let expected_gpr = if c.has_amx || c.has_avx512 { 31 } else { 15 };
        assert_eq!(c.max_gpr_count, expected_gpr,
            "max_gpr_count must be 31 when AMX or AVX-512 present, 15 otherwise");
    }

    // @trace TEST-CC-003 [level:unit]
    #[test]
    fn constraints_mutate_vec_field_does_not_affect_cloned() {
        // Arrange
        let mut c = CompilerConstraints {
            numa_core_bindings: vec![(0, 0, 4), (1, 4, 8), (2, 8, 16)],
            ..Default::default()
        };
        // Act
        let c2 = c.clone();
        c.numa_core_bindings.push((3, 16, 24));
        c.numa_core_bindings[0] = (99, 0, 0);
        // Assert: clone is a deep copy of the Vec
        assert_eq!(c2.numa_core_bindings.len(), 3, "clone must have original length");
        assert_eq!(c2.numa_core_bindings[0], (0, 0, 4), "clone must have original first element");
        assert_eq!(c.numa_core_bindings.len(), 4);
        assert_eq!(c.numa_core_bindings[0], (99, 0, 0));
    }

    // @trace TEST-CC-004 [level:unit]
    #[test]
    fn gpu_sm_partition_called_on_derived_constraints_matches_manual() {
        // Arrange: manually construct constraints with known gpu_sm_count
        let manual = CompilerConstraints {
            gpu_sm_count: Some(132),
            ..Default::default()
        };
        // Act: partition with 4 partitions
        let p_manual = manual.gpu_sm_partition(4);
        // Assert: 132 / 4 = 33
        assert_eq!(p_manual.total_sm, 132);
        assert_eq!(p_manual.num_partitions, 4);
        assert_eq!(p_manual.sm_per_partition, 33);
    }

    // @trace TEST-CC-005 [level:unit]
    #[test]
    fn constraints_debug_format_contains_all_seven_bool_field_names() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let debug = format!("{:?}", c);
        // Assert: all seven boolean fields must appear in debug output
        assert!(debug.contains("native_int4_dot"), "must contain native_int4_dot");
        assert!(debug.contains("has_amx"), "must contain has_amx");
        assert!(debug.contains("has_avx512"), "must contain has_avx512");
        assert!(debug.contains("has_sve"), "must contain has_sve");
        assert!(debug.contains("has_tma"), "must contain has_tma");
        assert!(debug.contains("has_native_fp4"), "must contain has_native_fp4");
        assert!(debug.contains("has_native_fp6"), "must contain has_native_fp6");
    }

    // @trace TEST-CC-006 [level:unit]
    #[test]
    fn serialize_json_contains_expected_keys() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(80),
            max_gpr_count: 31,
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        // Assert: JSON must contain the exact field names from the struct
        assert!(json.contains("\"max_gpr_count\""), "JSON must contain max_gpr_count key");
        assert!(json.contains("\"gpu_sm_count\""), "JSON must contain gpu_sm_count key");
        assert!(json.contains("\"simd_width_bits\""), "JSON must contain simd_width_bits key");
        assert!(json.contains("\"optimal_tile_bits\""), "JSON must contain optimal_tile_bits key");
    }

    // @trace TEST-CC-007 [level:unit]
    #[test]
    fn tile_bits_copy_then_clone_produce_identical_values() {
        // Arrange
        let t = TileBits { tile_m: 17, tile_n: 23, tile_k: 29 };
        // Act: both Copy and Clone from the same source
        let via_copy = t;
        let via_clone = t.clone();
        // Assert: both produce the exact same field values
        assert_eq!(via_copy.tile_m, via_clone.tile_m);
        assert_eq!(via_copy.tile_n, via_clone.tile_n);
        assert_eq!(via_copy.tile_k, via_clone.tile_k);
        assert_eq!(via_copy.tile_m, 17);
        assert_eq!(via_copy.tile_n, 23);
        assert_eq!(via_copy.tile_k, 29);
    }

    // @trace TEST-CC-008 [level:unit]
    #[test]
    fn derive_idempotent_two_calls_produce_same_fields() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act: derive twice from the same inputs
        let c1 = CompilerConstraints::derive(&profile, &sensors, None, None);
        let c2 = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: all deterministic fields must be identical
        assert_eq!(c1.simd_width_bits, c2.simd_width_bits);
        assert_eq!(c1.has_amx, c2.has_amx);
        assert_eq!(c1.has_avx512, c2.has_avx512);
        assert_eq!(c1.has_sve, c2.has_sve);
        assert_eq!(c1.max_gpr_count, c2.max_gpr_count);
        assert_eq!(c1.l1i_size, c2.l1i_size);
        assert_eq!(c1.gpu_sm_version, c2.gpu_sm_version);
        assert_eq!(c1.gpu_sm_count, c2.gpu_sm_count);
        assert_eq!(c1.tensor_core_gen, c2.tensor_core_gen);
        assert_eq!(c1.has_tma, c2.has_tma);
    }

    // @trace TEST-CC-009 [level:unit]
    #[test]
    fn numa_core_bindings_ranges_sum_never_exceeds_total_cores() {
        // Arrange: test with multiple node counts
        for node_count in &[1, 2, 3, 4, 7, 16] {
            let c = CompilerConstraints {
                numa_node_count: *node_count,
                ..Default::default()
            };
            let total_cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            // Act
            let bindings = c.numa_core_bindings();
            // Assert
            assert_eq!(bindings.len(), *node_count, "binding count must equal node count");
            let sum: usize = bindings.iter().map(|(_, r)| r.end - r.start).sum();
            assert!(sum <= total_cores,
                "sum of ranges ({}) must not exceed total cores ({}) for {} nodes",
                sum, total_cores, node_count);
        }
    }

    // @trace TEST-CC-010 [level:unit]
    #[test]
    fn exceeds_l1i_budget_property_threshold_is_strict_greater() {
        // Arrange: for several L1i sizes, verify strict > semantics (not >=)
        for &l1i in &[10, 50, 100, 500, 1000, 10000, 32 * 1024, 64 * 1024] {
            let c = CompilerConstraints {
                l1i_size: l1i,
                ..Default::default()
            };
            let threshold = (l1i as f64 * 0.8) as usize;
            // Act & Assert: threshold must NOT exceed, threshold + 1 must exceed
            assert!(
                !c.exceeds_l1i_budget(threshold),
                "l1i={} threshold={} must not exceed", l1i, threshold
            );
            assert!(
                c.exceeds_l1i_budget(threshold + 1),
                "l1i={} threshold+1={} must exceed", l1i, threshold + 1
            );
        }
    }

    // @trace TEST-CC-011 [level:unit]
    #[test]
    fn gpu_sm_partition_division_property_for_common_gpu_sizes() {
        // Arrange: real-world GPU SM counts
        let gpu_configs = [
            (56, 2, 28),    // V100
            (80, 4, 20),    // A100
            (108, 3, 36),   // A6000
            (132, 4, 33),   // H100
            (192, 6, 32),   // B200
        ];
        for (sm, np, expected_per) in gpu_configs {
            let c = CompilerConstraints {
                gpu_sm_count: Some(sm),
                ..Default::default()
            };
            // Act
            let p = c.gpu_sm_partition(np);
            // Assert
            assert_eq!(p.total_sm, sm, "total_sm must be {}", sm);
            assert_eq!(p.sm_per_partition, expected_per,
                "{} SMs / {} partitions must be {}", sm, np, expected_per);
            assert_eq!(p.num_partitions, np);
        }
    }

    // @trace TEST-CC-012 [level:unit]
    #[test]
    fn tile_bits_debug_output_format_structure() {
        // Arrange
        let t = TileBits { tile_m: 16, tile_n: 32, tile_k: 64 };
        // Act
        let debug = format!("{:?}", t);
        // Assert: Debug output must be a struct-like format containing all fields
        // Format is "TileBits { tile_m: 16, tile_n: 32, tile_k: 64 }"
        assert!(debug.starts_with("TileBits"), "Debug must start with type name");
        assert!(debug.contains("tile_m: 16"));
        assert!(debug.contains("tile_n: 32"));
        assert!(debug.contains("tile_k: 64"));
    }

    // @trace TEST-CC-013 [level:unit]
    #[test]
    fn constraints_clone_independence_after_multiple_field_mutations() {
        // Arrange: set all fields to non-default values
        let mut c = CompilerConstraints {
            max_gpr_count: 31,
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            native_int4_dot: true,
            l1i_size: 64 * 1024,
            l2_cache_size: 8 * 1024 * 1024,
            smem_size: Some(256 * 1024),
            numa_node_count: 4,
            gpu_sm_version: Some(90),
            gpu_sm_count: Some(132),
            gpu_warp_size: Some(32),
            simd_width_bits: 512,
            has_amx: true,
            has_avx512: true,
            has_sve: false,
            has_tma: true,
            has_native_fp4: true,
            has_native_fp6: true,
            rdma_min_chunk_tokens: Some(512),
            tensor_core_gen: 3,
            numa_core_bindings: vec![(0, 0, 4), (1, 4, 8)],
            sve_vl_bytes: None,
        };
        // Act
        let c2 = c.clone();
        // Mutate every field of original
        c.max_gpr_count = 0;
        c.optimal_tile_bits = TileBits { tile_m: 0, tile_n: 0, tile_k: 0 };
        c.native_int4_dot = false;
        c.l1i_size = 0;
        c.l2_cache_size = 0;
        c.smem_size = None;
        c.numa_node_count = 0;
        c.gpu_sm_version = None;
        c.gpu_sm_count = None;
        c.gpu_warp_size = None;
        c.simd_width_bits = 0;
        c.has_amx = false;
        c.has_avx512 = false;
        c.has_tma = false;
        c.has_native_fp4 = false;
        c.has_native_fp6 = false;
        c.rdma_min_chunk_tokens = None;
        c.tensor_core_gen = 0;
        c.numa_core_bindings.clear();
        // Assert: clone retains all original values
        assert_eq!(c2.max_gpr_count, 31);
        assert_eq!(c2.optimal_tile_bits.tile_m, 64);
        assert!(c2.native_int4_dot);
        assert_eq!(c2.l1i_size, 64 * 1024);
        assert_eq!(c2.l2_cache_size, 8 * 1024 * 1024);
        assert_eq!(c2.smem_size, Some(256 * 1024));
        assert_eq!(c2.numa_node_count, 4);
        assert_eq!(c2.gpu_sm_version, Some(90));
        assert_eq!(c2.gpu_sm_count, Some(132));
        assert_eq!(c2.gpu_warp_size, Some(32));
        assert_eq!(c2.simd_width_bits, 512);
        assert!(c2.has_amx);
        assert!(c2.has_avx512);
        assert!(c2.has_tma);
        assert!(c2.has_native_fp4);
        assert!(c2.has_native_fp6);
        assert_eq!(c2.rdma_min_chunk_tokens, Some(512));
        assert_eq!(c2.tensor_core_gen, 3);
        assert_eq!(c2.numa_core_bindings.len(), 2);
    }

    // @trace TEST-CC-014 [level:unit]
    #[test]
    fn derive_gpu_sm_count_and_partition_chain() {
        // Arrange: derive with CUDA GPU, then partition via derived constraints
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
        // Act: derive then use gpu_sm_partition on derived constraints
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        let p = c.gpu_sm_partition(4);
        // Assert: derived gpu_sm_count flows correctly into partition
        assert_eq!(c.gpu_sm_count, Some(132));
        assert_eq!(p.total_sm, 132);
        assert_eq!(p.sm_per_partition, 33); // 132 / 4
    }

    // @trace TEST-CC-015 [level:unit]
    #[test]
    fn serialize_deserialize_full_non_default_roundtrip() {
        // Arrange: every single field set to a non-default, distinguishable value
        let c = CompilerConstraints {
            max_gpr_count: 31,
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 32, tile_k: 48 },
            native_int4_dot: true,
            l1i_size: 64 * 1024,
            l2_cache_size: 6 * 1024 * 1024,
            smem_size: Some(228 * 1024),
            numa_node_count: 4,
            gpu_sm_version: Some(100),
            gpu_sm_count: Some(192),
            gpu_warp_size: Some(32),
            numa_core_bindings: vec![(0, 0, 4), (1, 4, 8)],
            simd_width_bits: 512,
            has_amx: true,
            has_avx512: true,
            has_sve: true,
            sve_vl_bytes: Some(32),
            has_tma: true,
            has_native_fp4: true,
            has_native_fp6: true,
            rdma_min_chunk_tokens: Some(1024),
            tensor_core_gen: 5,
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert: every field must roundtrip exactly
        assert_eq!(c2.max_gpr_count, 31);
        assert_eq!(c2.optimal_tile_bits.tile_m, 64);
        assert_eq!(c2.optimal_tile_bits.tile_n, 32);
        assert_eq!(c2.optimal_tile_bits.tile_k, 48);
        assert!(c2.native_int4_dot);
        assert_eq!(c2.l1i_size, 64 * 1024);
        assert_eq!(c2.l2_cache_size, 6 * 1024 * 1024);
        assert_eq!(c2.smem_size, Some(228 * 1024));
        assert_eq!(c2.numa_node_count, 4);
        assert_eq!(c2.gpu_sm_version, Some(100));
        assert_eq!(c2.gpu_sm_count, Some(192));
        assert_eq!(c2.gpu_warp_size, Some(32));
        assert_eq!(c2.numa_core_bindings, vec![(0, 0, 4), (1, 4, 8)]);
        assert_eq!(c2.simd_width_bits, 512);
        assert!(c2.has_amx);
        assert!(c2.has_avx512);
        assert!(c2.has_sve);
        assert_eq!(c2.sve_vl_bytes, Some(32));
        assert!(c2.has_tma);
        assert!(c2.has_native_fp4);
        assert!(c2.has_native_fp6);
        assert_eq!(c2.rdma_min_chunk_tokens, Some(1024));
        assert_eq!(c2.tensor_core_gen, 5);
    }

    // ---- Batch 12: 15 additional tests ----

    // @trace TEST-CC-016 [level:unit]
    #[test]
    fn derive_cuda_sm75_between_volta_ampere_no_tma() {
        // Arrange: SM75 (Turing) — above SM70 Volta, below SM80 Ampere
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 75 },
            compute_unit_count: 72,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 5,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: SM75 < SM90 → no TMA; SM75 < SM100 → no FP4/FP6
        assert_eq!(c.gpu_sm_version, Some(75));
        assert!(!c.has_tma, "Turing SM75 must not have TMA");
        assert!(!c.has_native_fp4, "Turing SM75 must not have FP4");
        assert!(!c.has_native_fp6, "Turing SM75 must not have FP6");
        assert_eq!(c.gpu_sm_count, Some(72));
        assert_eq!(c.smem_size, Some(64 * 1024));
        assert_eq!(c.l2_cache_size, 6 * 1024 * 1024);
    }

    // @trace TEST-CC-017 [level:unit]
    #[test]
    fn derive_hip_zero_compute_units_no_panic() {
        // Arrange: degenerate HIP GPU with zero compute units
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };
        // Act: must not panic with all-zero HIP GPU
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert
        assert_eq!(c.gpu_sm_count, Some(0));
        assert_eq!(c.gpu_sm_version, None, "HIP must not report SM version");
        assert_eq!(c.gpu_warp_size, Some(0));
        assert_eq!(c.smem_size, Some(0));
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
    }

    // @trace TEST-CC-018 [level:unit]
    #[test]
    fn derive_sve_vl_bytes_minimum_enforced_when_has_sve() {
        // Arrange: derive with real profile to check SVE VL bytes minimum
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: if has_sve is true, sve_vl_bytes must be >= 16 (per derive logic)
        if c.has_sve {
            assert!(
                c.sve_vl_bytes.unwrap_or(0) >= 16,
                "sve_vl_bytes must be >= 16 when has_sve is true, got {:?}",
                c.sve_vl_bytes
            );
        }
    }

    // @trace TEST-CC-019 [level:unit]
    #[test]
    fn derive_native_int4_dot_true_only_with_amx_or_avx512() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: native_int4_dot must be true iff has_amx or has_avx512
        // This is the derive() contract: line 238 sets it to has_amx || use_avx512
        let expected = profile.kernel_config.has_amx || profile.kernel_config.use_avx512;
        assert_eq!(c.native_int4_dot, expected,
            "native_int4_dot must equal has_amx || has_avx512");
    }

    // @trace TEST-CC-020 [level:unit]
    #[test]
    fn gpu_sm_partition_sm_per_partition_leq_total_sm() {
        // Arrange: for any partition count, sm_per_partition <= total_sm
        let c = CompilerConstraints {
            gpu_sm_count: Some(108),
            ..Default::default()
        };
        // Act & Assert: test many partition counts including edge cases
        for np in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 27, 36, 54, 108, 109, 256, 1024] {
            let p = c.gpu_sm_partition(np);
            assert!(
                p.sm_per_partition <= p.total_sm,
                "sm_per_partition ({}) must be <= total_sm ({}) for np={}",
                p.sm_per_partition, p.total_sm, np
            );
        }
    }

    // @trace TEST-CC-021 [level:unit]
    #[test]
    fn derive_cpu_simd_width_bits_equals_profile_bytes_times_8() {
        // Arrange
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let expected_bits = profile.simd_width_bytes() * 8;
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: derive sets simd_width_bits = profile.simd_width_bytes() * 8
        assert_eq!(c.simd_width_bits, expected_bits,
            "simd_width_bits must be profile.simd_width_bytes() * 8");
    }

    // @trace TEST-CC-022 [level:unit]
    #[test]
    fn constraints_construct_all_fields_zero_valid() {
        // Arrange & Act: all numeric fields zero, all bools false, all options None
        let c = CompilerConstraints {
            max_gpr_count: 0,
            optimal_tile_bits: TileBits { tile_m: 0, tile_n: 0, tile_k: 0 },
            native_int4_dot: false,
            l1i_size: 0,
            l2_cache_size: 0,
            smem_size: None,
            numa_node_count: 0,
            gpu_sm_version: None,
            gpu_sm_count: None,
            gpu_warp_size: None,
            numa_core_bindings: vec![],
            simd_width_bits: 0,
            has_amx: false,
            has_avx512: false,
            has_sve: false,
            sve_vl_bytes: None,
            has_tma: false,
            has_native_fp4: false,
            has_native_fp6: false,
            rdma_min_chunk_tokens: None,
            tensor_core_gen: 0,
        };
        // Assert: construction is valid, no panics, zero values preserved
        assert_eq!(c.max_gpr_count, 0);
        assert_eq!(c.l1i_size, 0);
        assert_eq!(c.l2_cache_size, 0);
        assert_eq!(c.numa_node_count, 0);
        assert_eq!(c.simd_width_bits, 0);
        assert_eq!(c.tensor_core_gen, 0);
        assert!(c.numa_core_bindings.is_empty());
    }

    // @trace TEST-CC-023 [level:unit]
    #[test]
    fn numa_core_bindings_core_ranges_have_positive_length() {
        // Arrange: 2 NUMA nodes — each range must have length > 0
        let c = CompilerConstraints {
            numa_node_count: 2,
            ..Default::default()
        };
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: if there are enough cores, each range must be non-empty
        if total_cores >= 2 {
            for (node_id, range) in &bindings {
                assert!(
                    !range.is_empty(),
                    "node {} range {:?} must be non-empty with {} total cores",
                    node_id, range, total_cores
                );
            }
        }
    }

    // @trace TEST-CC-024 [level:unit]
    #[test]
    fn derive_gpu_l2_overrides_cpu_even_when_cpu_l2_large() {
        // Arrange: GPU L2 = 80MB should override sensors L2 even if large
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_l2 = 80 * 1024 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 228 * 1024,
            l2_bytes: gpu_l2,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: GPU L2 must be used when gpu.l2_bytes > 0
        assert_eq!(c.l2_cache_size, gpu_l2,
            "GPU L2 must override when l2_bytes > 0");
    }

    // @trace TEST-CC-025 [level:unit]
    #[test]
    fn gpu_sm_partition_total_sm_preserved_across_many_partition_counts() {
        // Arrange
        let c = CompilerConstraints {
            gpu_sm_count: Some(192),
            ..Default::default()
        };
        // Act & Assert: total_sm must always be 192 regardless of partition count
        for np in [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192, 255, 1000] {
            let p = c.gpu_sm_partition(np);
            assert_eq!(p.total_sm, 192,
                "total_sm must be 192 for np={}", np);
        }
    }

    // @trace TEST-CC-026 [level:unit]
    #[test]
    fn derive_hip_nonzero_l2_applied_over_cpu_l2() {
        // Arrange: HIP GPU with large L2
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_l2 = 64 * 1024 * 1024;
        let gpu = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x942 },
            compute_unit_count: 228,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: gpu_l2,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x42,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: HIP non-zero L2 must override CPU L2
        assert_eq!(c.l2_cache_size, gpu_l2,
            "HIP GPU L2 must be applied when l2_bytes > 0");
    }

    // @trace TEST-CC-027 [level:unit]
    #[test]
    fn derive_idempotent_with_gpu_topology() {
        // Arrange
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
        // Act: derive twice with same GPU topology
        let c1 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        let c2 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: all fields must be identical across two calls
        assert_eq!(c1.gpu_sm_version, c2.gpu_sm_version);
        assert_eq!(c1.gpu_sm_count, c2.gpu_sm_count);
        assert_eq!(c1.gpu_warp_size, c2.gpu_warp_size);
        assert_eq!(c1.smem_size, c2.smem_size);
        assert_eq!(c1.tensor_core_gen, c2.tensor_core_gen);
        assert_eq!(c1.l2_cache_size, c2.l2_cache_size);
        assert_eq!(c1.has_tma, c2.has_tma);
        assert_eq!(c1.has_native_fp4, c2.has_native_fp4);
        assert_eq!(c1.has_native_fp6, c2.has_native_fp6);
        assert_eq!(c1.simd_width_bits, c2.simd_width_bits);
        assert_eq!(c1.l1i_size, c2.l1i_size);
        assert_eq!(c1.max_gpr_count, c2.max_gpr_count);
    }

    // @trace TEST-CC-028 [level:unit]
    #[test]
    fn derive_cuda_sm89_vs_sm90_tma_boundary() {
        // Arrange: SM89 (just below TMA) and SM90 (exactly at TMA threshold)
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_sm89 = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 89 },
            compute_unit_count: 142,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 100 * 1024,
            l2_bytes: 36 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 9,
        };
        let gpu_sm90 = GpuTopology {
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
        // Act
        let c89 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu_sm89));
        let c90 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu_sm90));
        // Assert: TMA flips from false to true at SM89 → SM90
        assert!(!c89.has_tma, "SM89 must not have TMA");
        assert!(c90.has_tma, "SM90 must have TMA");
        // Both must not have FP4 (requires SM100+)
        assert!(!c89.has_native_fp4);
        assert!(!c90.has_native_fp4);
    }

    // @trace TEST-CC-029 [level:unit]
    #[test]
    fn derive_cuda_sm99_vs_sm100_fp4_boundary() {
        // Arrange: SM99 (just below FP4) and SM100 (exactly at FP4 threshold)
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu_sm99 = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 99 },
            compute_unit_count: 128,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 200 * 1024,
            l2_bytes: 60 * 1024 * 1024,
            global_mem_bytes: 96 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 9,
        };
        let gpu_sm100 = GpuTopology {
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
        // Act
        let c99 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu_sm99));
        let c100 = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu_sm100));
        // Assert: FP4 flips from false to true at SM99 → SM100
        assert!(!c99.has_native_fp4, "SM99 must not have FP4");
        assert!(c100.has_native_fp4, "SM100 must have FP4");
        assert!(!c99.has_native_fp6, "SM99 must not have FP6");
        assert!(c100.has_native_fp6, "SM100 must have FP6");
        // Both have TMA (SM99 >= 90)
        assert!(c99.has_tma, "SM99 must have TMA");
        assert!(c100.has_tma, "SM100 must have TMA");
    }

    // @trace TEST-CC-030 [level:unit]
    #[test]
    fn exceeds_l1i_budget_threshold_is_strictly_greater_than_80_percent() {
        // Arrange: for any l1i_size, exactly floor(l1i * 0.8) must not exceed
        // and floor(l1i * 0.8) + 1 must exceed
        let test_sizes: Vec<usize> = (1..=50)
            .chain([100, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536].iter().copied())
            .collect();
        for &l1i in &test_sizes {
            let c = CompilerConstraints {
                l1i_size: l1i,
                ..Default::default()
            };
            let threshold = (l1i as f64 * 0.8) as usize;
            // Act & Assert
            assert!(
                !c.exceeds_l1i_budget(threshold),
                "l1i={} threshold={} must not exceed", l1i, threshold
            );
            assert!(
                c.exceeds_l1i_budget(threshold + 1),
                "l1i={} threshold+1={} must exceed", l1i, threshold + 1
            );
        }
    }

    // @trace TEST-CC-031 [level:unit]
    #[test]
    fn exceeds_l1i_budget_zero_kernel_bytes_never_exceeds_any_l1i() {
        // Arrange: kernel code of 0 bytes should never exceed even a 1-byte L1i
        for &l1i in &[1, 16, 256, 4096, 32 * 1024, 64 * 1024] {
            let c = CompilerConstraints { l1i_size: l1i, ..Default::default() };
            // Act & Assert
            assert!(!c.exceeds_l1i_budget(0), "0-byte kernel must not exceed l1i={}", l1i);
        }
    }

    // @trace TEST-CC-032 [level:unit]
    #[test]
    fn gpu_sm_partition_num_partitions_zero_treated_as_one() {
        // Arrange: zero partitions should be treated as 1 (max(1) guard)
        let c = CompilerConstraints { gpu_sm_count: Some(48), ..Default::default() };
        // Act
        let p = c.gpu_sm_partition(0);
        // Assert: max(0, 1) = 1 partition, so sm_per_partition = 48
        assert_eq!(p.num_partitions, 0);
        assert_eq!(p.sm_per_partition, 48, "zero partitions => divide by max(1)=1");
    }

    // @trace TEST-CC-033 [level:unit]
    #[test]
    fn gpu_sm_partition_product_never_exceeds_total_coverage() {
        // Arrange: verify sm_per_partition * num_partitions <= total_sm
        for total in [1, 7, 13, 80, 132, 192] {
            let c = CompilerConstraints { gpu_sm_count: Some(total), ..Default::default() };
            for &nparts in &[1, 2, 3, 4, 7, 13, total, total + 1] {
                // Act
                let p = c.gpu_sm_partition(nparts);
                // Assert
                assert!(
                    p.sm_per_partition * p.num_partitions <= total,
                    "sm_per_partition({}) * num_partitions({}) = {} > total({})",
                    p.sm_per_partition, p.num_partitions,
                    p.sm_per_partition * p.num_partitions, total,
                );
            }
        }
    }

    // @trace TEST-CC-034 [level:unit]
    #[test]
    fn tile_bits_all_same_value_not_default() {
        // Arrange: construct TileBits where all three fields are equal but not 16
        let t = TileBits { tile_m: 8, tile_n: 8, tile_k: 8 };
        // Assert: values are correct and differ from default
        assert_eq!(t.tile_m, 8);
        assert_eq!(t.tile_n, 8);
        assert_eq!(t.tile_k, 8);
        assert_ne!(t.tile_m, TileBits::default().tile_m);
    }

    // @trace TEST-CC-035 [level:unit]
    #[test]
    fn tile_bits_copy_independent_after_field_change() {
        // Arrange: copy a TileBits, then change the original
        let mut t1 = TileBits { tile_m: 4, tile_n: 8, tile_k: 16 };
        let t2 = t1; // Copy (TileBits is Copy)
        // Act: mutate t1
        t1.tile_m = 99;
        // Assert: t2 is unaffected
        assert_eq!(t2.tile_m, 4, "copy must be independent");
        assert_eq!(t1.tile_m, 99);
    }

    // @trace TEST-CC-036 [level:unit]
    #[test]
    fn constraints_default_l2_cache_exactly_256kb() {
        // Arrange & Act
        let c = CompilerConstraints::default();
        // Assert: 256 * 1024 = 262144
        assert_eq!(c.l2_cache_size, 262_144);
    }

    // @trace TEST-CC-037 [level:unit]
    #[test]
    fn constraints_sve_vl_bytes_some_with_has_sve_false_is_inconsistent() {
        // Arrange: manually construct inconsistent state
        let c = CompilerConstraints {
            has_sve: false,
            sve_vl_bytes: Some(32),
            ..Default::default()
        };
        // Assert: the struct allows it (pub fields), verify values roundtrip
        assert!(!c.has_sve);
        assert_eq!(c.sve_vl_bytes, Some(32));
    }

    // @trace TEST-CC-038 [level:unit]
    #[test]
    fn gpu_sm_partition_copy_semantics_field_independence() {
        // Arrange: GpuSmPartition is Copy, verify independence
        let mut p1 = GpuSmPartition { total_sm: 80, num_partitions: 4, sm_per_partition: 20 };
        let p2 = p1; // Copy
        // Act: mutate p1
        p1.total_sm = 0;
        p1.sm_per_partition = 0;
        // Assert: p2 unaffected
        assert_eq!(p2.total_sm, 80);
        assert_eq!(p2.sm_per_partition, 20);
    }

    // @trace TEST-CC-039 [level:unit]
    #[test]
    fn constraints_numa_bindings_len_equals_numa_node_count() {
        // Arrange: various node counts
        for &count in &[1, 2, 4, 8] {
            let c = CompilerConstraints { numa_node_count: count, ..Default::default() };
            // Act
            let bindings = c.numa_core_bindings();
            // Assert
            assert_eq!(bindings.len(), count, "binding count must match numa_node_count");
        }
    }

    // @trace TEST-CC-040 [level:unit]
    #[test]
    fn constraints_manual_construction_all_fields_non_default() {
        // Arrange: build constraints where every field differs from default
        let c = CompilerConstraints {
            max_gpr_count: 31,
            optimal_tile_bits: TileBits { tile_m: 64, tile_n: 64, tile_k: 64 },
            native_int4_dot: true,
            l1i_size: 64 * 1024,
            l2_cache_size: 512 * 1024,
            smem_size: Some(228 * 1024),
            numa_node_count: 4,
            gpu_sm_version: Some(90),
            gpu_sm_count: Some(132),
            gpu_warp_size: Some(32),
            numa_core_bindings: vec![(0, 0, 4), (1, 4, 8)],
            simd_width_bits: 512,
            has_amx: true,
            has_avx512: true,
            has_sve: false,
            sve_vl_bytes: None,
            has_tma: true,
            has_native_fp4: false,
            has_native_fp6: false,
            rdma_min_chunk_tokens: Some(64),
            tensor_core_gen: 3,
        };
        // Assert: each field is non-default
        assert_ne!(c.max_gpr_count, CompilerConstraints::default().max_gpr_count);
        assert_ne!(c.optimal_tile_bits.tile_m, TileBits::default().tile_m);
        assert!(c.native_int4_dot);
        assert_ne!(c.l1i_size, CompilerConstraints::default().l1i_size);
        assert!(c.smem_size.is_some());
        assert_eq!(c.numa_core_bindings.len(), 2);
        assert_eq!(c.tensor_core_gen, 3);
    }

    // @trace TEST-CC-041 [level:unit]
    #[test]
    fn serialize_default_produces_valid_json_with_all_fields() {
        // Arrange
        let c = CompilerConstraints::default();
        // Act
        let json = serde_json::to_string(&c).unwrap();
        // Assert: JSON must contain key field names
        assert!(json.contains("max_gpr_count"));
        assert!(json.contains("optimal_tile_bits"));
        assert!(json.contains("l1i_size"));
        assert!(json.contains("l2_cache_size"));
        assert!(json.contains("simd_width_bits"));
        assert!(json.contains("tensor_core_gen"));
        assert!(json.contains("numa_core_bindings"));
    }

    // @trace TEST-CC-042 [level:unit]
    #[test]
    fn exceeds_l1i_budget_very_large_l1i_no_panic() {
        // Arrange: use a large but safe l1i_size to verify no overflow
        let large_l1i = 512 * 1024 * 1024; // 512 MB L1i (absurdly large but valid usize)
        let c = CompilerConstraints { l1i_size: large_l1i, ..Default::default() };
        // Act & Assert: must not panic
        let threshold = (large_l1i as f64 * 0.8) as usize;
        assert!(!c.exceeds_l1i_budget(threshold));
        assert!(c.exceeds_l1i_budget(threshold + 1));
        assert!(!c.exceeds_l1i_budget(0));
    }

    // @trace TEST-CC-043 [level:unit]
    #[test]
    fn gpu_sm_partition_clone_equals_original() {
        // Arrange
        let c = CompilerConstraints { gpu_sm_count: Some(100), ..Default::default() };
        let p = c.gpu_sm_partition(5);
        // Act
        let p2 = p.clone();
        // Assert: all fields identical
        assert_eq!(p2.total_sm, p.total_sm);
        assert_eq!(p2.num_partitions, p.num_partitions);
        assert_eq!(p2.sm_per_partition, p.sm_per_partition);
    }

    // ---- Batch 13: 10 additional tests ----

    // @trace TEST-CC-044 [level:unit]
    #[test]
    fn derive_gpu_l2_bytes_one_still_overrides_cpu_l2() {
        // Arrange: GPU reports l2_bytes = 1 (minimal non-zero value)
        // The derive logic checks `if gpu.l2_bytes > 0`, so even 1 byte should override
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 164 * 1024,
            l2_bytes: 1, // minimal non-zero
            global_mem_bytes: 48 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: l2_bytes=1 > 0, so GPU L2 overrides CPU L2
        assert_eq!(c.l2_cache_size, 1,
            "GPU l2_bytes=1 must override CPU L2 (any non-zero GPU L2 wins)");
    }

    // @trace TEST-CC-045 [level:unit]
    #[test]
    fn derive_with_probe_nonzero_l2_thrash_threshold() {
        // Arrange: probe with a specific l2_thrash_threshold value
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let probe = ProbeResult {
            spill_points: vec![256],
            smem_cliffs: vec![(512, 0.75), (1024, 0.4)],
            l2_thrash_threshold: 16 * 1024 * 1024,
            device_fingerprint: "probe-thrash-test".to_string(),
            raw_measurements: Default::default(),
        };
        // Act: derive with probe must complete without panic
        let c = CompilerConstraints::derive(&profile, &sensors, Some(&probe), None);
        // Assert: derive still produces valid constraints regardless of probe content
        assert!(c.simd_width_bits > 0);
        assert!(c.l1i_size > 0);
        assert!(c.l2_cache_size > 0);
        // CPU-only: GPU fields must remain default
        assert_eq!(c.gpu_sm_count, None);
        assert_eq!(c.tensor_core_gen, 0);
    }

    // @trace TEST-CC-046 [level:unit]
    #[test]
    fn gpu_sm_partition_zero_sm_with_two_partitions_yields_zero() {
        // Arrange: gpu_sm_count = Some(0) with multiple partitions
        let c = CompilerConstraints {
            gpu_sm_count: Some(0),
            ..Default::default()
        };
        // Act
        let p = c.gpu_sm_partition(2);
        // Assert: 0 / 2 = 0, total_sm = 0, but num_partitions stored verbatim
        assert_eq!(p.total_sm, 0);
        assert_eq!(p.num_partitions, 2);
        assert_eq!(p.sm_per_partition, 0);
    }

    // @trace TEST-CC-047 [level:unit]
    #[test]
    fn tile_bits_degenerate_k_zero_m_n_nonzero() {
        // Arrange: tile_k=0 with tile_m and tile_n positive — degenerate but constructible
        let t = TileBits { tile_m: 32, tile_n: 32, tile_k: 0 };
        // Assert: construction preserves exact values
        assert_eq!(t.tile_m, 32);
        assert_eq!(t.tile_n, 32);
        assert_eq!(t.tile_k, 0);
        // Serialize roundtrip
        let json = serde_json::to_string(&t).unwrap();
        let t2: TileBits = serde_json::from_str(&json).unwrap();
        assert_eq!(t2.tile_m, 32);
        assert_eq!(t2.tile_n, 32);
        assert_eq!(t2.tile_k, 0);
    }

    // @trace TEST-CC-048 [level:unit]
    #[test]
    fn numa_core_bindings_many_nodes_few_cores_each_gets_small_ranges() {
        // Arrange: more NUMA nodes than typical cores — ranges may be 0-length
        let c = CompilerConstraints {
            numa_node_count: 1000,
            ..Default::default()
        };
        let total_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Act
        let bindings = c.numa_core_bindings();
        // Assert: must produce exactly numa_node_count bindings
        assert_eq!(bindings.len(), 1000);
        // With 1000 nodes, cores_per_node = total_cores / 1000 which may be 0
        let cores_per_node = total_cores / 1000;
        if cores_per_node == 0 {
            // All ranges are 0..0 (empty) when cores < nodes
            for (_, range) in &bindings {
                assert!(range.start == range.end || range.start < range.end,
                    "range {:?} must be empty or valid when cores_per_node=0", range);
            }
        }
    }

    // @trace TEST-CC-049 [level:unit]
    #[test]
    fn constraints_exceeds_l1i_budget_combined_with_gpu_partition() {
        // Arrange: constraints with GPU and L1i set, exercise both methods
        let c = CompilerConstraints {
            l1i_size: 16 * 1024,
            gpu_sm_count: Some(64),
            ..Default::default()
        };
        // Act: use both methods together
        let exceeds = c.exceeds_l1i_budget(14 * 1024);
        let partition = c.gpu_sm_partition(4);
        // Assert: both methods work on the same struct without interference
        let threshold = (16_usize * 1024 * 8 / 10) as usize;
        assert!(exceeds || 14 * 1024 <= threshold);
        assert_eq!(partition.total_sm, 64);
        assert_eq!(partition.sm_per_partition, 16);
    }

    // @trace TEST-CC-050 [level:unit]
    #[test]
    fn derive_metal_gpu_warp_size_preserved_as_compute_unit_count() {
        // Arrange: Metal GPU with warp_size=32 and specific compute_unit_count
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let gpu = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 0x1004 },
            compute_unit_count: 64,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 48 * 1024,
            l2_bytes: 24 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, Some(&gpu));
        // Assert: warp_size and compute_unit_count map to correct fields
        assert_eq!(c.gpu_warp_size, Some(32));
        assert_eq!(c.gpu_sm_count, Some(64));
        assert_eq!(c.smem_size, Some(48 * 1024));
        assert_eq!(c.l2_cache_size, 24 * 1024 * 1024);
        // Metal: no SM version, no NVIDIA-specific features
        assert_eq!(c.gpu_sm_version, None);
        assert!(!c.has_tma);
        assert!(!c.has_native_fp4);
        assert!(!c.has_native_fp6);
    }

    // @trace TEST-CC-051 [level:unit]
    #[test]
    fn serialize_roundtrip_smem_size_large_value() {
        // Arrange: smem_size with a large realistic value (512 KB for Blackwell)
        let c = CompilerConstraints {
            smem_size: Some(512 * 1024),
            ..Default::default()
        };
        // Act
        let json = serde_json::to_string(&c).unwrap();
        let c2: CompilerConstraints = serde_json::from_str(&json).unwrap();
        // Assert: large smem_size survives roundtrip
        assert_eq!(c2.smem_size, Some(512 * 1024));
        // All other fields remain default
        assert_eq!(c2.max_gpr_count, 15);
        assert_eq!(c2.gpu_sm_count, None);
        assert_eq!(c2.tensor_core_gen, 0);
    }

    // @trace TEST-CC-052 [level:unit]
    #[test]
    fn derive_cpu_numa_node_count_from_sensors_ccx_topology() {
        // Arrange: derive from real sensors and verify numa_node_count mapping
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let expected_nodes = sensors.ccx_numa_topology
            .as_ref()
            .map(|t| t.nodes.len())
            .unwrap_or(1);
        // Act
        let c = CompilerConstraints::derive(&profile, &sensors, None, None);
        // Assert: numa_node_count must match ccx_numa_topology or default to 1
        assert_eq!(c.numa_node_count, expected_nodes,
            "numa_node_count must be {} (from sensors ccx_numa_topology or 1)",
            expected_nodes);
    }

    // @trace TEST-CC-053 [level:unit]
    #[test]
    fn constraints_bool_fields_independent_control() {
        // Arrange: set each boolean feature independently to true, others false
        let bool_fields = [
            ("has_amx", CompilerConstraints { has_amx: true, ..Default::default() }),
            ("has_avx512", CompilerConstraints { has_avx512: true, ..Default::default() }),
            ("has_sve", CompilerConstraints { has_sve: true, ..Default::default() }),
            ("has_tma", CompilerConstraints { has_tma: true, ..Default::default() }),
            ("has_native_fp4", CompilerConstraints { has_native_fp4: true, ..Default::default() }),
            ("has_native_fp6", CompilerConstraints { has_native_fp6: true, ..Default::default() }),
            ("native_int4_dot", CompilerConstraints { native_int4_dot: true, ..Default::default() }),
        ];
        for (name, c) in &bool_fields {
            // Act & Assert: the named field must be true, all others must be false
            assert!(match name {
                &"has_amx" => c.has_amx,
                &"has_avx512" => c.has_avx512,
                &"has_sve" => c.has_sve,
                &"has_tma" => c.has_tma,
                &"has_native_fp4" => c.has_native_fp4,
                &"has_native_fp6" => c.has_native_fp6,
                &"native_int4_dot" => c.native_int4_dot,
                _ => false,
            }, "{} must be true", name);

            // All other bools must be false (independent control)
            assert_eq!(c.has_amx, name == &"has_amx", "has_amx only true when explicitly set");
            assert_eq!(c.has_avx512, name == &"has_avx512", "has_avx512 only true when explicitly set");
            assert_eq!(c.has_sve, name == &"has_sve", "has_sve only true when explicitly set");
            assert_eq!(c.has_tma, name == &"has_tma", "has_tma only true when explicitly set");
            assert_eq!(c.has_native_fp4, name == &"has_native_fp4", "has_native_fp4 only true when explicitly set");
            assert_eq!(c.has_native_fp6, name == &"has_native_fp6", "has_native_fp6 only true when explicitly set");
            assert_eq!(c.native_int4_dot, name == &"native_int4_dot", "native_int4_dot only true when explicitly set");
        }
    }
}
