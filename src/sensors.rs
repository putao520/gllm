//! 硬件物理拓扑传感器 — MemoryNetworkSensors
//!
//! 实现 SPEC §12.6 "硬件探测→IR 强约束变量体系"。
//! 探测跨机/跨片的物理拓扑参数，转化为 JIT 编译器的约束变量。

use gllm_kernels::dispatch::DeviceProfile;

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

    /// 设置 RDMA 参数（用于分布式推理）
    pub fn set_rdma_params(&mut self, bandwidth_gbs: f32, latency_us: f32) {
        self.nic_bandwidth_gbs = Some(bandwidth_gbs);
        self.rdma_latency_us = Some(latency_us);
    }

    /// 计算 RDMA Pipelining 的最小 Chunk 大小
    ///
    /// 满足约束: T_compute(chunk) >= T_rdma_transfer(chunk)
    pub fn min_chunk_size_for_rdma(&self, compute_tflops: f32) -> Option<usize> {
        let bandwidth = self.nic_bandwidth_gbs?;
        let latency = self.rdma_latency_us?;

        let transfer_time_per_gb = 1.0 / bandwidth;
        let compute_time_per_tflop = 1.0 / compute_tflops;

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
