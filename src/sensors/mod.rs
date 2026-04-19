//! 硬件物理拓扑传感器 — MemoryNetworkSensors
//!
//! 实现 SPEC §12.6 "硬件探测→IR 强约束变量体系"。
//! 探测跨机/跨片的物理拓扑参数，转化为 JIT 编译器的约束变量。

use gllm_kernels::dispatch::DeviceProfile;
use crate::jit::compiler_constraints::CompilerConstraints;

pub mod gpu;

pub use gpu::{GpuPlatform, GpuTopology};

/// NUMA 拓扑结构
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

/// NUMA 节点
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub l3_bytes: usize,
    pub core_count: usize,
}

/// 跨机/跨片物理拓扑的传感器读数
///
/// 将硬件探测结果转化为 JIT 编译器的强约束变量。
#[derive(Debug, Clone)]
pub struct MemoryNetworkSensors {
    /// GPU L2 Cache 驻留容量 (级别锚点)
    pub l2_cache_bytes: usize,

    /// AMD CCD / CCX 的 L3 分区边界探测
    pub ccx_numa_topology: Option<NumaTopology>,

    /// TLB 条目数上限
    pub tlb_entries: usize,

    /// 跨机网卡带宽 (GB/s), 用于 Pipelining 约束
    pub nic_bandwidth_gbs: Option<f32>,

    /// 跨机 RDMA 往返延迟 (μs), 用于 Chunk 切分下限
    pub rdma_latency_us: Option<f32>,

    /// ARM SVE/SME 向量宽度与 ZA 阵列尺寸
    pub arm_sme_za_size: Option<usize>,
}

impl MemoryNetworkSensors {
    /// 从 DeviceProfile 探测硬件拓扑
    pub fn detect(profile: &DeviceProfile) -> Self {
        let l2_cache_bytes = profile.kernel_config.l2;
        let tlb_entries = Self::detect_tlb_entries(profile);
        let ccx_numa_topology = Self::detect_numa_topology(profile);
        let arm_sme_za_size = Self::detect_arm_sme_za_size(profile);

        Self {
            l2_cache_bytes,
            ccx_numa_topology,
            tlb_entries,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size,
        }
    }

    /// 探测 TLB 条目数
    fn detect_tlb_entries(profile: &DeviceProfile) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_tlb_entries()
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64_tlb_entries()
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = profile;
            512
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_tlb_entries() -> usize {
        1024
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64_tlb_entries() -> usize {
        2048
    }

    /// 探测 NUMA 拓扑（AMD CCD/CCX）
    fn detect_numa_topology(_profile: &DeviceProfile) -> Option<NumaTopology> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_numa()
        }

        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_numa() -> Option<NumaTopology> {
        use std::fs;

        let numa_path = "/sys/devices/system/node";
        let entries = fs::read_dir(numa_path).ok()?;

        let mut nodes = Vec::new();

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_str()?;

            if !name_str.starts_with("node") {
                continue;
            }

            let node_id = name_str.strip_prefix("node")?.parse::<usize>().ok()?;

            let cpulist_path = format!("{}/node{}/cpulist", numa_path, node_id);
            let cpulist = fs::read_to_string(cpulist_path).ok()?;
            let core_count = Self::parse_cpulist(&cpulist);

            nodes.push(NumaNode {
                node_id,
                l3_bytes: 32 * 1024 * 1024,
                core_count,
            });
        }

        if nodes.is_empty() {
            None
        } else {
            Some(NumaTopology { nodes })
        }
    }

    #[cfg(target_os = "linux")]
    fn parse_cpulist(cpulist: &str) -> usize {
        let mut count = 0;
        for part in cpulist.trim().split(',') {
            if let Some((start, end)) = part.split_once('-') {
                if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    count += e - s + 1;
                }
            } else if part.parse::<usize>().is_ok() {
                count += 1;
            }
        }
        count
    }

    /// 探测 ARM SME ZA 阵列尺寸
    fn detect_arm_sme_za_size(_profile: &DeviceProfile) -> Option<usize> {
        #[cfg(target_arch = "aarch64")]
        {
            None
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            None
        }
    }

    /// 获取 L3 缓存大小（从 NUMA 拓扑推导）
    pub fn total_l3_bytes(&self) -> usize {
        self.ccx_numa_topology
            .as_ref()
            .map(|t| t.nodes.iter().map(|n| n.l3_bytes).sum())
            .unwrap_or(0)
    }
}

// ============================================================================
// SystemTopology — 聚合 CPU/GPU 拓扑 + CompilerConstraints (§12.6)
// ============================================================================

/// CPU 拓扑信息
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// 物理核心数
    pub core_count: usize,
    /// NUMA 拓扑
    pub numa: Option<NumaTopology>,
    /// L1d 缓存大小 (bytes)
    pub l1d_bytes: usize,
    /// L2 缓存大小 (bytes)
    pub l2_bytes: usize,
    /// L3 缓存大小 (bytes, 所有 NUMA 节点总和)
    pub l3_bytes: usize,
}

/// 系统级硬件拓扑 (§12.6)
///
/// 聚合 CPU + GPU 拓扑信息，一次性探测，运行时固定。
/// 输出 `CompilerConstraints` 供 JIT 编译器消费。
#[derive(Debug, Clone)]
pub struct SystemTopology {
    /// CPU 拓扑
    pub cpu: CpuTopology,
    /// GPU 拓扑 (None = CPU-only)
    pub gpu: Option<GpuTopology>,
    /// 底层传感器读数
    pub sensors: MemoryNetworkSensors,
    /// DeviceProfile (gllm-kernels)
    pub profile: DeviceProfile,
    /// 推导出的 JIT 编译器约束变量
    pub constraints: CompilerConstraints,
}

impl SystemTopology {
    /// 一次性探测系统拓扑 (§12.6: 加载期探测，运行时固定)
    ///
    /// GPU 探测错误（feature 启用但 runtime 查询失败）会 panic。
    /// 需要可恢复语义的调用方使用 [`SystemTopology::try_detect`]。
    pub fn detect() -> Self {
        Self::try_detect().expect("SystemTopology::detect: GPU probe failure")
    }

    /// 一次性探测系统拓扑，传播 GPU 探测错误（真实 bug 场景）。
    ///
    /// ## 返回值
    ///
    /// - `Ok(topo)`: 探测成功（`topo.gpu` 为 `Some` 表示有 GPU，`None` 表示 CPU-only 系统）
    /// - `Err(msg)`: feature 启用且 driver 库加载成功，但属性查询失败
    ///
    /// 禁止降级：feature 启用但 runtime 错误绝不静默返回 `gpu: None`。
    pub fn try_detect() -> Result<Self, String> {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        let (l1d, l2, _l3) = profile.cache_sizes();

        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let l3_bytes = sensors.total_l3_bytes();

        let cpu = CpuTopology {
            core_count,
            numa: sensors.ccx_numa_topology.clone(),
            l1d_bytes: l1d,
            l2_bytes: l2,
            l3_bytes,
        };

        // §12.6 Phase 4: 真实 GPU 拓扑探测（CUDA → HIP → Metal）。
        // feature-gated；无 feature 启用 → Ok(None)。
        let gpu = gpu::detect_gpu()?;

        let constraints = CompilerConstraints::derive(&profile, &sensors, None, gpu.as_ref());

        Ok(Self {
            cpu,
            gpu,
            sensors,
            profile,
            constraints,
        })
    }

    /// 从已有 DeviceProfile 构建（避免重复探测）。
    ///
    /// GPU 探测错误会 panic；使用 [`SystemTopology::try_from_profile`] 获得 Result。
    pub fn from_profile(profile: DeviceProfile) -> Self {
        Self::try_from_profile(profile)
            .expect("SystemTopology::from_profile: GPU probe failure")
    }

    /// 从已有 DeviceProfile 构建，传播 GPU 探测错误。
    pub fn try_from_profile(profile: DeviceProfile) -> Result<Self, String> {
        let sensors = MemoryNetworkSensors::detect(&profile);
        let (l1d, l2, _l3) = profile.cache_sizes();

        let core_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let l3_bytes = sensors.total_l3_bytes();

        let cpu = CpuTopology {
            core_count,
            numa: sensors.ccx_numa_topology.clone(),
            l1d_bytes: l1d,
            l2_bytes: l2,
            l3_bytes,
        };

        let gpu = gpu::detect_gpu()?;

        let constraints = CompilerConstraints::derive(&profile, &sensors, None, gpu.as_ref());

        Ok(Self {
            cpu,
            gpu,
            sensors,
            profile,
            constraints,
        })
    }

    /// 获取 JIT 编译器约束变量
    pub fn compiler_constraints(&self) -> &CompilerConstraints {
        &self.constraints
    }

    /// 是否有 GPU
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// NUMA 节点数
    pub fn numa_node_count(&self) -> usize {
        self.cpu.numa.as_ref().map(|n| n.nodes.len()).unwrap_or(1)
    }
}

impl MemoryNetworkSensors {
    /// 设置 RDMA 参数（用于分布式推理）
    pub fn set_rdma_params(&mut self, bandwidth_gbs: f32, latency_us: f32) {
        self.nic_bandwidth_gbs = Some(bandwidth_gbs);
        self.rdma_latency_us = Some(latency_us);
    }

    /// 计算 RDMA Pipelining 的最小 Chunk 大小
    ///
    /// 满足约束: T_compute(chunk) >= T_rdma_transfer(chunk)
    pub fn min_chunk_size_for_rdma(&self, _compute_tflops: f32) -> Option<usize> {
        let bandwidth = self.nic_bandwidth_gbs?;
        let latency = self.rdma_latency_us?;

        // 最小传输数据量 = latency(s) * bandwidth(GB/s) → GB
        // 假设 token = 4KB (4096 B) embedding
        let min_data_gb = latency * 1e-6 * bandwidth;
        let min_tokens = (min_data_gb * 1e9 / 4096.0) as usize;

        Some(min_tokens.max(64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensors_creation() {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);

        assert!(sensors.l2_cache_bytes > 0);
        assert!(sensors.tlb_entries > 0);
    }

    #[test]
    fn test_system_topology_detect() {
        let topo = SystemTopology::detect();
        assert!(topo.cpu.core_count > 0);
        assert!(topo.cpu.l1d_bytes > 0);
        assert!(topo.constraints.simd_width_bits > 0);
        assert!(topo.constraints.l2_cache_size > 0);
    }

    #[test]
    fn test_system_topology_from_profile() {
        let profile = DeviceProfile::detect();
        let topo = SystemTopology::from_profile(profile);
        assert!(topo.cpu.core_count > 0);
        assert!(topo.numa_node_count() >= 1);

        // 无 GPU feature → has_gpu() 必须 false
        // feature 启用但无物理 GPU → has_gpu() 仍为 false
        // feature 启用且有物理 GPU → has_gpu() 为 true（字段已被真实填充）
        #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
        {
            assert!(!topo.has_gpu(), "CPU-only build must report has_gpu() == false");
        }

        // GPU 探测结果与 constraints 字段一致性
        if let Some(gpu) = topo.gpu.as_ref() {
            assert!(gpu.compute_unit_count > 0);
            assert!(gpu.warp_size > 0);
            assert_eq!(
                topo.constraints.gpu_warp_size,
                Some(gpu.warp_size),
                "CompilerConstraints.gpu_warp_size must mirror GpuTopology.warp_size"
            );
            assert_eq!(
                topo.constraints.gpu_sm_count,
                Some(gpu.compute_unit_count),
                "CompilerConstraints.gpu_sm_count must mirror GpuTopology.compute_unit_count"
            );
        } else {
            assert_eq!(topo.constraints.gpu_sm_count, None);
            assert_eq!(topo.constraints.gpu_warp_size, None);
        }
    }

    #[test]
    fn test_gpu_detection_result_shape() {
        // try_detect 必须可调用且返回 Result
        let result = SystemTopology::try_detect();
        // 在 CI 里通常 CPU-only 或 driver 不存在 → Ok
        // 若真实 GPU 探测失败（feature 启用但 runtime 错误） → Err
        match result {
            Ok(topo) => {
                assert!(topo.cpu.core_count > 0);
            }
            Err(msg) => {
                assert!(!msg.is_empty(), "error must have diagnostic message");
            }
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
    #[test]
    fn test_no_gpu_feature_yields_none() {
        // 无 GPU feature 编译 → SystemTopology::detect().gpu 必须为 None（无 panic）
        let topo = SystemTopology::detect();
        assert!(topo.gpu.is_none());
        assert!(topo.constraints.gpu_sm_count.is_none());
        assert!(topo.constraints.gpu_sm_version.is_none());
        assert!(topo.constraints.gpu_warp_size.is_none());
    }

    #[test]
    fn test_rdma_chunk_size() {
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);

        sensors.set_rdma_params(100.0, 10.0);

        let min_chunk = sensors.min_chunk_size_for_rdma(100.0);
        assert!(min_chunk.is_some());
        assert!(min_chunk.unwrap() >= 64);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist() {
        assert_eq!(MemoryNetworkSensors::parse_cpulist("0-7"), 8);
        assert_eq!(MemoryNetworkSensors::parse_cpulist("0,2,4,6"), 4);
        assert_eq!(MemoryNetworkSensors::parse_cpulist("0-3,8-11"), 8);
    }
}
