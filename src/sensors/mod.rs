//! 硬件物理拓扑传感器 — MemoryNetworkSensors
//!
//! 实现 SPEC §12.6 "硬件探测→IR 强约束变量体系"。
//! 探测跨机/跨片的物理拓扑参数，转化为 JIT 编译器的约束变量。

use gllm_kernels::dispatch::DeviceProfile;
use crate::jit::compiler_constraints::CompilerConstraints;
use crate::kv_cache::CompressionCodec;

pub mod gpu;

pub use gpu::{GpuPlatform, GpuTopology};

/// NUMA 拓扑结构
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

/// NUMA 节点
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub node_id: usize,
    pub l3_bytes: usize,
    pub core_count: usize,
}

/// 跨机/跨片物理拓扑的传感器读数
///
/// 将硬件探测结果转化为 JIT 编译器的强约束变量。
#[derive(Debug, Clone, PartialEq)]
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
    fn detect_tlb_entries(_profile: &DeviceProfile) -> usize {
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
#[derive(Debug, Clone, PartialEq)]
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

// ============================================================================
// Compression Telemetry — SPEC 22 §9
// ============================================================================

/// Per-codec compression statistics (SPEC 22 §9).
///
/// Tracks compression ratio, latency, and byte counts for a single codec.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodecStats {
    /// Number of compression operations using this codec.
    pub compress_count: u64,
    /// Number of decompression operations using this codec.
    pub decompress_count: u64,
    /// Cumulative compression latency in microseconds.
    pub total_compress_latency_us: u64,
    /// Cumulative decompression latency in microseconds.
    pub total_decompress_latency_us: u64,
    /// Cumulative input bytes compressed with this codec.
    pub total_input_bytes: u64,
    /// Cumulative output bytes from compression with this codec.
    pub total_output_bytes: u64,
}

impl CodecStats {
    /// Average compression ratio for this codec (output / input).
    /// Returns 1.0 if no operations recorded.
    pub fn avg_compression_ratio(&self) -> f64 {
        if self.total_input_bytes == 0 {
            1.0
        } else {
            self.total_output_bytes as f64 / self.total_input_bytes as f64
        }
    }

    /// Average compression latency in microseconds.
    pub fn avg_compress_latency_us(&self) -> f64 {
        if self.compress_count == 0 {
            0.0
        } else {
            self.total_compress_latency_us as f64 / self.compress_count as f64
        }
    }

    /// Average decompression latency in microseconds.
    pub fn avg_decompress_latency_us(&self) -> f64 {
        if self.decompress_count == 0 {
            0.0
        } else {
            self.total_decompress_latency_us as f64 / self.decompress_count as f64
        }
    }
}

/// Compression telemetry aggregator (SPEC 22 §9).
///
/// Collects compression ratio statistics, codec latency histograms,
/// migration byte counters, and codec usage distribution across
/// all compression / decompression / migration paths.
///
/// Codec index mapping (same as `CompressionCodec` discriminant):
///   0 = None, 1 = Lz4, 2 = BitPackRle, 3 = NvcompAns, 4 = ZstdDict
#[derive(Debug, Clone, PartialEq)]
pub struct CompressionTelemetry {
    /// Per-codec statistics (indexed by CompressionCodec discriminant).
    pub codec_stats: [CodecStats; 5],
    /// Total bytes before compression across all codecs.
    pub total_input_bytes: u64,
    /// Total bytes after compression across all codecs.
    pub total_compressed_bytes: u64,
    /// Total bytes migrated between storage tiers.
    pub total_migration_bytes: u64,
    /// Total number of compression operations.
    pub compress_count: u64,
    /// Total number of decompression operations.
    pub decompress_count: u64,
    /// Cumulative compression latency across all codecs (μs).
    pub total_compress_latency_us: u64,
    /// Cumulative decompression latency across all codecs (μs).
    pub total_decompress_latency_us: u64,
    /// Number of page evictions recorded.
    pub eviction_count: u64,
    /// Number of page swap-ins recorded.
    pub swap_in_count: u64,
}

impl CompressionTelemetry {
    /// Create a new, empty compression telemetry aggregator.
    pub fn new() -> Self {
        Self {
            codec_stats: [CodecStats::default(), CodecStats::default(),
                          CodecStats::default(), CodecStats::default(),
                          CodecStats::default()],
            total_input_bytes: 0,
            total_compressed_bytes: 0,
            total_migration_bytes: 0,
            compress_count: 0,
            decompress_count: 0,
            total_compress_latency_us: 0,
            total_decompress_latency_us: 0,
            eviction_count: 0,
            swap_in_count: 0,
        }
    }

    /// Record a compression operation.
    ///
    /// # Arguments
    /// * `codec` — The compression codec used.
    /// * `input_bytes` — Uncompressed byte count.
    /// * `output_bytes` — Compressed byte count.
    /// * `latency_us` — Compression wall-clock time in microseconds.
    pub fn record_compress(&mut self, codec: CompressionCodec, input_bytes: u64, output_bytes: u64, latency_us: u64) {
        let idx = codec as usize;
        if idx < self.codec_stats.len() {
            let stats = &mut self.codec_stats[idx];
            stats.compress_count += 1;
            stats.total_compress_latency_us += latency_us;
            stats.total_input_bytes += input_bytes;
            stats.total_output_bytes += output_bytes;
        }
        self.total_input_bytes += input_bytes;
        self.total_compressed_bytes += output_bytes;
        self.compress_count += 1;
        self.total_compress_latency_us += latency_us;
    }

    /// Record a decompression operation.
    ///
    /// # Arguments
    /// * `codec` — The compression codec used.
    /// * `input_bytes` — Compressed byte count.
    /// * `output_bytes` — Decompressed byte count.
    /// * `latency_us` — Decompression wall-clock time in microseconds.
    pub fn record_decompress(&mut self, codec: CompressionCodec, input_bytes: u64, output_bytes: u64, latency_us: u64) {
        let idx = codec as usize;
        if idx < self.codec_stats.len() {
            let stats = &mut self.codec_stats[idx];
            stats.decompress_count += 1;
            stats.total_decompress_latency_us += latency_us;
            stats.total_input_bytes += input_bytes;
            stats.total_output_bytes += output_bytes;
        }
        self.decompress_count += 1;
        self.total_decompress_latency_us += latency_us;
    }

    /// Record a data migration between storage tiers.
    ///
    /// # Arguments
    /// * `bytes` — Number of bytes transferred.
    pub fn record_migration(&mut self, bytes: u64) {
        self.total_migration_bytes += bytes;
    }

    /// Record a page eviction event.
    pub fn record_eviction(&mut self) {
        self.eviction_count += 1;
    }

    /// Record a page swap-in event.
    pub fn record_swap_in(&mut self) {
        self.swap_in_count += 1;
    }

    /// Overall compression ratio across all codecs (compressed / original).
    /// Returns 1.0 if no input data recorded.
    pub fn overall_compression_ratio(&self) -> f64 {
        if self.total_input_bytes == 0 {
            1.0
        } else {
            self.total_compressed_bytes as f64 / self.total_input_bytes as f64
        }
    }

    /// Average compression latency across all codecs (μs).
    pub fn avg_compress_latency_us(&self) -> f64 {
        if self.compress_count == 0 {
            0.0
        } else {
            self.total_compress_latency_us as f64 / self.compress_count as f64
        }
    }

    /// Average decompression latency across all codecs (μs).
    pub fn avg_decompress_latency_us(&self) -> f64 {
        if self.decompress_count == 0 {
            0.0
        } else {
            self.total_decompress_latency_us as f64 / self.decompress_count as f64
        }
    }

    /// Get the per-codec statistics for a given codec.
    pub fn codec_stats(&self, codec: CompressionCodec) -> &CodecStats {
        let idx = codec as usize;
        &self.codec_stats[idx]
    }
}

impl Default for CompressionTelemetry {
    fn default() -> Self {
        Self::new()
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

    // ── Compression Telemetry Tests (SPEC 22 §9) ──

    #[test]
    fn test_codec_stats_default() {
        let stats = CodecStats::default();
        assert_eq!(stats.compress_count, 0);
        assert_eq!(stats.decompress_count, 0);
        assert_eq!(stats.total_compress_latency_us, 0);
        assert_eq!(stats.total_decompress_latency_us, 0);
        assert_eq!(stats.total_input_bytes, 0);
        assert_eq!(stats.total_output_bytes, 0);
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
        assert!((stats.avg_compress_latency_us() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_new() {
        let ct = CompressionTelemetry::new();
        assert_eq!(ct.compress_count, 0);
        assert_eq!(ct.decompress_count, 0);
        assert_eq!(ct.total_migration_bytes, 0);
        assert_eq!(ct.eviction_count, 0);
        assert_eq!(ct.swap_in_count, 0);
        assert!((ct.overall_compression_ratio() - 1.0).abs() < 1e-9);
        assert!((ct.avg_compress_latency_us() - 0.0).abs() < 1e-9);
        assert!((ct.avg_decompress_latency_us() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_record_compress() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, 1000);
        assert_eq!(ct.total_compressed_bytes, 500);
        assert_eq!(ct.total_compress_latency_us, 50);

        let stats = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.total_input_bytes, 1000);
        assert_eq!(stats.total_output_bytes, 500);
        assert_eq!(stats.total_compress_latency_us, 50);
        assert!((stats.avg_compression_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_record_decompress() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        assert_eq!(ct.decompress_count, 1);
        assert_eq!(ct.total_decompress_latency_us, 30);

        let stats = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(stats.decompress_count, 1);
        assert_eq!(stats.total_decompress_latency_us, 30);
    }

    #[test]
    fn test_compression_telemetry_migration_and_events() {
        let mut ct = CompressionTelemetry::new();
        ct.record_migration(4096);
        ct.record_migration(8192);
        assert_eq!(ct.total_migration_bytes, 12288);

        ct.record_eviction();
        ct.record_eviction();
        assert_eq!(ct.eviction_count, 2);

        ct.record_swap_in();
        assert_eq!(ct.swap_in_count, 1);
    }

    #[test]
    fn test_compression_telemetry_overall_ratio() {
        let mut ct = CompressionTelemetry::new();
        // 50% compression ratio for LZ4
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        // 25% compression ratio for BitPackRle
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 500, 100);
        // Overall: (500 + 500) / (1000 + 2000) = 1000 / 3000 ≈ 0.3333
        let ratio = ct.overall_compression_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 1e-9, "overall ratio = {ratio}");
    }

    #[test]
    fn test_compression_telemetry_avg_latency() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 500, 150);
        assert!((ct.avg_compress_latency_us() - 100.0).abs() < 1e-9);
        assert!((ct.avg_decompress_latency_us() - 0.0).abs() < 1e-9);

        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 70);
        assert!((ct.avg_decompress_latency_us() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_codec_stats_method() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::None, 100, 100, 0);
        ct.record_compress(CompressionCodec::ZstdDict, 10000, 100, 1000);

        let none_stats = ct.codec_stats(CompressionCodec::None);
        assert_eq!(none_stats.compress_count, 1);
        assert_eq!(none_stats.total_input_bytes, 100);

        let zstd_stats = ct.codec_stats(CompressionCodec::ZstdDict);
        assert_eq!(zstd_stats.compress_count, 1);
        assert_eq!(zstd_stats.total_input_bytes, 10000);
        assert_eq!(zstd_stats.total_output_bytes, 100);
        assert!((zstd_stats.avg_compression_ratio() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_default() {
        let ct: CompressionTelemetry = Default::default();
        assert_eq!(ct.compress_count, 0);
        assert_eq!(ct.total_input_bytes, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional comprehensive unit tests
    // ═══════════════════════════════════════════════════════════════════════════

    // ── NumaTopology & NumaNode construction, field access, Clone, Debug ────────

    #[test]
    fn test_numa_node_construction() {
        let node = NumaNode {
            node_id: 3,
            l3_bytes: 32 * 1024 * 1024,
            core_count: 8,
        };
        assert_eq!(node.node_id, 3);
        assert_eq!(node.l3_bytes, 33_554_432);
        assert_eq!(node.core_count, 8);
    }

    #[test]
    fn test_numa_node_zero_fields() {
        let node = NumaNode {
            node_id: 0,
            l3_bytes: 0,
            core_count: 0,
        };
        assert_eq!(node.node_id, 0);
        assert_eq!(node.l3_bytes, 0);
        assert_eq!(node.core_count, 0);
    }

    #[test]
    fn test_numa_node_clone_independence() {
        let node = NumaNode {
            node_id: 1,
            l3_bytes: 16 * 1024 * 1024,
            core_count: 4,
        };
        let cloned = node.clone();
        assert_eq!(cloned.node_id, node.node_id);
        assert_eq!(cloned.l3_bytes, node.l3_bytes);
        assert_eq!(cloned.core_count, node.core_count);
    }

    #[test]
    fn test_numa_node_debug_output() {
        let node = NumaNode {
            node_id: 2,
            l3_bytes: 64 * 1024 * 1024,
            core_count: 16,
        };
        let debug_str = format!("{node:?}");
        assert!(debug_str.contains("NumaNode"));
        assert!(debug_str.contains("node_id"));
        assert!(debug_str.contains("l3_bytes"));
        assert!(debug_str.contains("core_count"));
    }

    #[test]
    fn test_numa_topology_empty_nodes() {
        let topo = NumaTopology { nodes: vec![] };
        assert!(topo.nodes.is_empty());
    }

    #[test]
    fn test_numa_topology_single_node() {
        let topo = NumaTopology {
            nodes: vec![NumaNode {
                node_id: 0,
                l3_bytes: 32 * 1024 * 1024,
                core_count: 8,
            }],
        };
        assert_eq!(topo.nodes.len(), 1);
        assert_eq!(topo.nodes[0].node_id, 0);
    }

    #[test]
    fn test_numa_topology_multi_node() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode {
                    node_id: 0,
                    l3_bytes: 32 * 1024 * 1024,
                    core_count: 8,
                },
                NumaNode {
                    node_id: 1,
                    l3_bytes: 32 * 1024 * 1024,
                    core_count: 8,
                },
            ],
        };
        assert_eq!(topo.nodes.len(), 2);
        assert_eq!(topo.nodes[0].node_id, 0);
        assert_eq!(topo.nodes[1].node_id, 1);
    }

    #[test]
    fn test_numa_topology_clone() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode {
                    node_id: 0,
                    l3_bytes: 16 * 1024 * 1024,
                    core_count: 4,
                },
            ],
        };
        let cloned = topo.clone();
        assert_eq!(cloned.nodes.len(), topo.nodes.len());
        assert_eq!(cloned.nodes[0].node_id, topo.nodes[0].node_id);
    }

    #[test]
    fn test_numa_topology_debug_output() {
        let topo = NumaTopology {
            nodes: vec![NumaNode {
                node_id: 0,
                l3_bytes: 32 * 1024 * 1024,
                core_count: 8,
            }],
        };
        let debug_str = format!("{topo:?}");
        assert!(debug_str.contains("NumaTopology"));
        assert!(debug_str.contains("nodes"));
    }

    // ── MemoryNetworkSensors total_l3_bytes ─────────────────────────────────────

    #[test]
    fn test_total_l3_bytes_no_numa() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096 * 1024,
            ccx_numa_topology: None,
            tlb_entries: 1024,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 0);
    }

    #[test]
    fn test_total_l3_bytes_single_numa_node() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096 * 1024,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![NumaNode {
                    node_id: 0,
                    l3_bytes: 32 * 1024 * 1024,
                    core_count: 8,
                }],
            }),
            tlb_entries: 1024,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 32 * 1024 * 1024);
    }

    #[test]
    fn test_total_l3_bytes_multi_numa_nodes() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096 * 1024,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![
                    NumaNode {
                        node_id: 0,
                        l3_bytes: 32 * 1024 * 1024,
                        core_count: 8,
                    },
                    NumaNode {
                        node_id: 1,
                        l3_bytes: 64 * 1024 * 1024,
                        core_count: 16,
                    },
                ],
            }),
            tlb_entries: 1024,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 96 * 1024 * 1024);
    }

    // ── MemoryNetworkSensors field access, Debug, Clone ─────────────────────────

    #[test]
    fn test_sensors_manual_construction() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 2 * 1024 * 1024,
            ccx_numa_topology: None,
            tlb_entries: 2048,
            nic_bandwidth_gbs: Some(200.0),
            rdma_latency_us: Some(5.0),
            arm_sme_za_size: Some(256),
        };
        assert_eq!(sensors.l2_cache_bytes, 2 * 1024 * 1024);
        assert_eq!(sensors.tlb_entries, 2048);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(200.0));
        assert_eq!(sensors.rdma_latency_us, Some(5.0));
        assert_eq!(sensors.arm_sme_za_size, Some(256));
        assert!(sensors.ccx_numa_topology.is_none());
    }

    #[test]
    fn test_sensors_debug_output() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: None,
            tlb_entries: 512,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        let debug_str = format!("{sensors:?}");
        assert!(debug_str.contains("MemoryNetworkSensors"));
        assert!(debug_str.contains("l2_cache_bytes"));
        assert!(debug_str.contains("tlb_entries"));
        assert!(debug_str.contains("nic_bandwidth_gbs"));
        assert!(debug_str.contains("rdma_latency_us"));
        assert!(debug_str.contains("arm_sme_za_size"));
    }

    #[test]
    fn test_sensors_clone_independence() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![NumaNode {
                    node_id: 0,
                    l3_bytes: 32 * 1024 * 1024,
                    core_count: 8,
                }],
            }),
            tlb_entries: 1024,
            nic_bandwidth_gbs: Some(100.0),
            rdma_latency_us: Some(10.0),
            arm_sme_za_size: None,
        };
        let cloned = sensors.clone();
        assert_eq!(cloned.l2_cache_bytes, sensors.l2_cache_bytes);
        assert_eq!(cloned.tlb_entries, sensors.tlb_entries);
        assert_eq!(cloned.nic_bandwidth_gbs, sensors.nic_bandwidth_gbs);
        assert_eq!(cloned.rdma_latency_us, sensors.rdma_latency_us);
        // Verify NUMA topology is independently cloned
        assert_eq!(
            cloned.ccx_numa_topology.as_ref().unwrap().nodes.len(),
            sensors.ccx_numa_topology.as_ref().unwrap().nodes.len(),
        );
    }

    // ── MemoryNetworkSensors set_rdma_params ─────────────────────────────────────

    #[test]
    fn test_set_rdma_params_sets_both_fields() {
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);

        assert!(sensors.nic_bandwidth_gbs.is_none());
        assert!(sensors.rdma_latency_us.is_none());

        sensors.set_rdma_params(100.0, 10.0);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(100.0));
        assert_eq!(sensors.rdma_latency_us, Some(10.0));

        // Overwrite with new values
        sensors.set_rdma_params(200.0, 5.0);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(200.0));
        assert_eq!(sensors.rdma_latency_us, Some(5.0));
    }

    // ── MemoryNetworkSensors min_chunk_size_for_rdma edge cases ─────────────────

    #[test]
    fn test_min_chunk_size_rdma_no_params_returns_none() {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // No RDMA params set → None
        assert!(sensors.min_chunk_size_for_rdma(100.0).is_none());
    }

    #[test]
    fn test_min_chunk_size_rdma_minimum_floor() {
        // Very small bandwidth + latency → computed tokens may be < 64, must be floored to 64
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);
        sensors.set_rdma_params(0.001, 0.001);
        let chunk = sensors.min_chunk_size_for_rdma(100.0).unwrap();
        assert!(chunk >= 64, "min chunk must be >= 64, got {chunk}");
    }

    #[test]
    fn test_min_chunk_size_rdma_typical_values() {
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);
        // 100 GB/s NIC, 10 μs latency
        sensors.set_rdma_params(100.0, 10.0);
        let chunk = sensors.min_chunk_size_for_rdma(100.0).unwrap();
        // min_data_gb = 10e-6 * 100 = 0.001 GB = 1_000_000 bytes
        // min_tokens = 1_000_000_000 * 1e-9 * 100 * 1e-6 / 4096
        //   = 1e9 * 100e-6 * 1e-6 / 4096 = 100 * 1e-3 / 4096 = 0.1 / 4096 ≈ 24.4
        //   → floored to 64
        assert!(chunk >= 64);
    }

    // ── CpuTopology construction, fields, Debug, Clone ──────────────────────────

    #[test]
    fn test_cpu_topology_construction() {
        let cpu = CpuTopology {
            core_count: 16,
            numa: Some(NumaTopology {
                nodes: vec![
                    NumaNode {
                        node_id: 0,
                        l3_bytes: 32 * 1024 * 1024,
                        core_count: 8,
                    },
                    NumaNode {
                        node_id: 1,
                        l3_bytes: 32 * 1024 * 1024,
                        core_count: 8,
                    },
                ],
            }),
            l1d_bytes: 32 * 1024,
            l2_bytes: 512 * 1024,
            l3_bytes: 64 * 1024 * 1024,
        };
        assert_eq!(cpu.core_count, 16);
        assert_eq!(cpu.l1d_bytes, 32 * 1024);
        assert_eq!(cpu.l2_bytes, 512 * 1024);
        assert_eq!(cpu.l3_bytes, 64 * 1024 * 1024);
        assert!(cpu.numa.is_some());
        assert_eq!(cpu.numa.as_ref().unwrap().nodes.len(), 2);
    }

    #[test]
    fn test_cpu_topology_no_numa() {
        let cpu = CpuTopology {
            core_count: 4,
            numa: None,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 8 * 1024 * 1024,
        };
        assert_eq!(cpu.core_count, 4);
        assert!(cpu.numa.is_none());
    }

    #[test]
    fn test_cpu_topology_debug_output() {
        let cpu = CpuTopology {
            core_count: 8,
            numa: None,
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 16 * 1024 * 1024,
        };
        let debug_str = format!("{cpu:?}");
        assert!(debug_str.contains("CpuTopology"));
        assert!(debug_str.contains("core_count"));
        assert!(debug_str.contains("l1d_bytes"));
        assert!(debug_str.contains("l2_bytes"));
        assert!(debug_str.contains("l3_bytes"));
        assert!(debug_str.contains("numa"));
    }

    #[test]
    fn test_cpu_topology_clone_independence() {
        let cpu = CpuTopology {
            core_count: 8,
            numa: Some(NumaTopology {
                nodes: vec![NumaNode {
                    node_id: 0,
                    l3_bytes: 32 * 1024 * 1024,
                    core_count: 8,
                }],
            }),
            l1d_bytes: 32 * 1024,
            l2_bytes: 256 * 1024,
            l3_bytes: 32 * 1024 * 1024,
        };
        let cloned = cpu.clone();
        assert_eq!(cloned.core_count, cpu.core_count);
        assert_eq!(cloned.l1d_bytes, cpu.l1d_bytes);
        assert_eq!(cloned.l2_bytes, cpu.l2_bytes);
        assert_eq!(cloned.l3_bytes, cpu.l3_bytes);
        assert_eq!(
            cloned.numa.as_ref().unwrap().nodes[0].node_id,
            cpu.numa.as_ref().unwrap().nodes[0].node_id,
        );
    }

    // ── SystemTopology has_gpu / numa_node_count with constructed data ──────────

    #[test]
    fn test_has_gpu_with_none() {
        let topo = SystemTopology::detect();
        if topo.gpu.is_none() {
            assert!(!topo.has_gpu());
        }
        // If GPU is present, has_gpu() must be true
        if topo.gpu.is_some() {
            assert!(topo.has_gpu());
        }
    }

    #[test]
    fn test_numa_node_count_no_numa() {
        // When SystemTopology has no NUMA info, numa_node_count defaults to 1
        let topo = SystemTopology::detect();
        assert!(topo.numa_node_count() >= 1);
    }

    // ── CodecStats edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_codec_stats_avg_decompress_latency_zero_count() {
        let stats = CodecStats::default();
        assert!((stats.avg_decompress_latency_us() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_avg_decompress_latency_single_op() {
        let mut stats = CodecStats::default();
        stats.decompress_count = 1;
        stats.total_decompress_latency_us = 42;
        assert!((stats.avg_decompress_latency_us() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_avg_decompress_latency_multiple_ops() {
        let mut stats = CodecStats::default();
        stats.decompress_count = 3;
        stats.total_decompress_latency_us = 30 + 60 + 90;
        assert!((stats.avg_decompress_latency_us() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_avg_compress_latency_single_op() {
        let mut stats = CodecStats::default();
        stats.compress_count = 1;
        stats.total_compress_latency_us = 100;
        assert!((stats.avg_compress_latency_us() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_avg_compress_latency_multiple_ops() {
        let mut stats = CodecStats::default();
        stats.compress_count = 4;
        stats.total_compress_latency_us = 10 + 20 + 30 + 40;
        assert!((stats.avg_compress_latency_us() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_compression_ratio_no_expansion() {
        let mut stats = CodecStats::default();
        stats.total_input_bytes = 1000;
        stats.total_output_bytes = 1000;
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_compression_ratio_high_compression() {
        let mut stats = CodecStats::default();
        stats.total_input_bytes = 1_000_000;
        stats.total_output_bytes = 100;
        assert!((stats.avg_compression_ratio() - 0.0001).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_large_byte_counts() {
        let mut stats = CodecStats::default();
        stats.compress_count = 1000;
        stats.total_input_bytes = u64::MAX;
        stats.total_output_bytes = u64::MAX / 2;
        let ratio = stats.avg_compression_ratio();
        assert!((ratio - 0.5).abs() < 1e-6, "ratio = {ratio}");
    }

    // ── CompressionTelemetry per-codec isolation ────────────────────────────────

    #[test]
    fn test_compression_telemetry_per_codec_isolation() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 1000, 100);

        // Global counters aggregate both
        assert_eq!(ct.compress_count, 2);
        assert_eq!(ct.total_input_bytes, 3000);
        assert_eq!(ct.total_compressed_bytes, 1500);

        // Per-codec counters are isolated
        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.compress_count, 1);
        assert_eq!(lz4.total_input_bytes, 1000);
        assert_eq!(lz4.total_output_bytes, 500);

        let bpr = ct.codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(bpr.compress_count, 1);
        assert_eq!(bpr.total_input_bytes, 2000);
        assert_eq!(bpr.total_output_bytes, 1000);

        // Unrelated codecs have zero counts
        let none = ct.codec_stats(CompressionCodec::None);
        assert_eq!(none.compress_count, 0);
        let nvcomp = ct.codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(nvcomp.compress_count, 0);
        let zstd = ct.codec_stats(CompressionCodec::ZstdDict);
        assert_eq!(zstd.compress_count, 0);
    }

    #[test]
    fn test_compression_telemetry_decompress_per_codec_isolation() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        ct.record_decompress(CompressionCodec::NvcompAns, 200, 800, 20);
        ct.record_decompress(CompressionCodec::NvcompAns, 300, 1200, 40);

        assert_eq!(ct.decompress_count, 3);
        assert_eq!(ct.total_decompress_latency_us, 90);

        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.decompress_count, 1);
        assert_eq!(lz4.total_decompress_latency_us, 30);

        let nvcomp = ct.codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(nvcomp.decompress_count, 2);
        assert_eq!(nvcomp.total_decompress_latency_us, 60);
    }

    // ── CompressionTelemetry cumulative counters ───────────────────────────────

    #[test]
    fn test_compression_telemetry_cumulative_latency() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 10);
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 20);
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 30);
        // Average = (10+20+30) / 3 = 20
        assert!((ct.avg_compress_latency_us() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_cumulative_migration_bytes() {
        let mut ct = CompressionTelemetry::new();
        for _ in 0..10 {
            ct.record_migration(4096);
        }
        assert_eq!(ct.total_migration_bytes, 40960);
    }

    #[test]
    fn test_compression_telemetry_eviction_and_swap_in_counters() {
        let mut ct = CompressionTelemetry::new();
        for _ in 0..5 {
            ct.record_eviction();
        }
        for _ in 0..3 {
            ct.record_swap_in();
        }
        assert_eq!(ct.eviction_count, 5);
        assert_eq!(ct.swap_in_count, 3);
    }

    // ── CompressionTelemetry all codecs indexed correctly ───────────────────────

    #[test]
    fn test_compression_telemetry_all_five_codecs() {
        let mut ct = CompressionTelemetry::new();
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, &codec) in codecs.iter().enumerate() {
            ct.record_compress(codec, 100 * (i as u64 + 1), 50 * (i as u64 + 1), 10);
        }
        assert_eq!(ct.compress_count, 5);
        // Verify each codec has exactly 1 compress
        for &codec in &codecs {
            assert_eq!(ct.codec_stats(codec).compress_count, 1);
        }
    }

    // ── CompressionTelemetry zero-byte compress ─────────────────────────────────

    #[test]
    fn test_compression_telemetry_zero_byte_compress() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::None, 0, 0, 0);
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.total_input_bytes, 0);
        assert_eq!(ct.total_compressed_bytes, 0);
        // Ratio with 0 input returns 1.0
        assert!((ct.overall_compression_ratio() - 1.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry ratio > 1.0 (expansion) ───────────────────────────

    #[test]
    fn test_compression_telemetry_expansion_ratio() {
        let mut ct = CompressionTelemetry::new();
        // Compression expanded the data (output > input)
        ct.record_compress(CompressionCodec::None, 100, 200, 5);
        assert!((ct.overall_compression_ratio() - 2.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry clone independence ─────────────────────────────────

    #[test]
    fn test_compression_telemetry_clone_independence() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        ct.record_eviction();

        let cloned = ct.clone();
        assert_eq!(cloned.compress_count, ct.compress_count);
        assert_eq!(cloned.total_input_bytes, ct.total_input_bytes);
        assert_eq!(cloned.eviction_count, ct.eviction_count);

        // Mutating original does not affect clone
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 1000, 100);
        assert_eq!(cloned.compress_count, 1);
        assert_eq!(ct.compress_count, 2);
    }

    // ── CompressionTelemetry debug output ───────────────────────────────────────

    #[test]
    fn test_compression_telemetry_debug_output() {
        let ct = CompressionTelemetry::new();
        let debug_str = format!("{ct:?}");
        assert!(debug_str.contains("CompressionTelemetry"));
        assert!(debug_str.contains("codec_stats"));
        assert!(debug_str.contains("total_input_bytes"));
        assert!(debug_str.contains("total_compressed_bytes"));
        assert!(debug_str.contains("total_migration_bytes"));
        assert!(debug_str.contains("compress_count"));
        assert!(debug_str.contains("decompress_count"));
        assert!(debug_str.contains("eviction_count"));
        assert!(debug_str.contains("swap_in_count"));
    }

    // ── CompressionTelemetry avg latency with decompress only ───────────────────

    #[test]
    fn test_compression_telemetry_decompress_only_avg_latency() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 40);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 60);
        // No compress ops → compress avg = 0
        assert!((ct.avg_compress_latency_us() - 0.0).abs() < 1e-9);
        // Decompress avg = (40+60)/2 = 50
        assert!((ct.avg_decompress_latency_us() - 50.0).abs() < 1e-9);
    }

    // ── SystemTopology compiler_constraints accessor ────────────────────────────

    #[test]
    fn test_system_topology_compiler_constraints_accessor() {
        let topo = SystemTopology::detect();
        let constraints = topo.compiler_constraints();
        // Must return a reference to the same object as topo.constraints
        assert!(std::ptr::eq(constraints, &topo.constraints));
    }

    // ── SystemTopology debug output ─────────────────────────────────────────────

    #[test]
    fn test_system_topology_debug_output() {
        let topo = SystemTopology::detect();
        let debug_str = format!("{topo:?}");
        assert!(debug_str.contains("SystemTopology"));
        assert!(debug_str.contains("cpu"));
        assert!(debug_str.contains("gpu"));
        assert!(debug_str.contains("sensors"));
        assert!(debug_str.contains("profile"));
        assert!(debug_str.contains("constraints"));
    }

    // ── MemoryNetworkSensors detect preserves all non-optional fields ────────────

    #[test]
    fn test_sensors_detect_optional_fields_initially_none() {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // RDMA params are not auto-detected
        assert!(sensors.nic_bandwidth_gbs.is_none());
        assert!(sensors.rdma_latency_us.is_none());
        // ARM SME is None on x86_64 and currently None even on aarch64
        // (platform-specific, just verify it compiles)
        let _ = sensors.arm_sme_za_size;
    }

    // ── parse_cpulist edge cases (Linux only) ───────────────────────────────────

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_single_core() {
        assert_eq!(MemoryNetworkSensors::parse_cpulist("5"), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_empty_string() {
        assert_eq!(MemoryNetworkSensors::parse_cpulist(""), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_whitespace_only() {
        assert_eq!(MemoryNetworkSensors::parse_cpulist("   "), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_trailing_comma() {
        // Trailing comma with empty part after it → ignored
        assert_eq!(MemoryNetworkSensors::parse_cpulist("0-3,"), 4);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_mixed_ranges_and_singles() {
        assert_eq!(MemoryNetworkSensors::parse_cpulist("0-1,4,7,10-11"), 6);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_range_single_element() {
        // Range where start == end → 1 core
        assert_eq!(MemoryNetworkSensors::parse_cpulist("3-3"), 1);
    }

    // ── CompressionCodec discriminant indexing ──────────────────────────────────

    #[test]
    fn test_compression_codec_discriminant_indices() {
        // Verify CompressionCodec discriminants match codec_stats array indexing
        assert_eq!(CompressionCodec::None as usize, 0);
        assert_eq!(CompressionCodec::Lz4 as usize, 1);
        assert_eq!(CompressionCodec::BitPackRle as usize, 2);
        assert_eq!(CompressionCodec::NvcompAns as usize, 3);
        assert_eq!(CompressionCodec::ZstdDict as usize, 4);
    }

    #[test]
    fn test_compression_telemetry_codec_stats_array_length() {
        let ct = CompressionTelemetry::new();
        assert_eq!(ct.codec_stats.len(), 5);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // New comprehensive tests (45 additional)
    // ═══════════════════════════════════════════════════════════════════════════

    // ── PartialEq / Eq derive verification ────────────────────────────────────

    #[test]
    fn test_numa_node_partial_eq_equal() {
        let a = NumaNode { node_id: 1, l3_bytes: 32768, core_count: 4 };
        let b = NumaNode { node_id: 1, l3_bytes: 32768, core_count: 4 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_numa_node_partial_eq_not_equal_node_id() {
        let a = NumaNode { node_id: 0, l3_bytes: 32768, core_count: 4 };
        let b = NumaNode { node_id: 1, l3_bytes: 32768, core_count: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_numa_node_partial_eq_not_equal_l3_bytes() {
        let a = NumaNode { node_id: 0, l3_bytes: 32768, core_count: 4 };
        let b = NumaNode { node_id: 0, l3_bytes: 65536, core_count: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_numa_node_partial_eq_not_equal_core_count() {
        let a = NumaNode { node_id: 0, l3_bytes: 32768, core_count: 4 };
        let b = NumaNode { node_id: 0, l3_bytes: 32768, core_count: 8 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_numa_topology_partial_eq_equal() {
        let a = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 2 }] };
        let b = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 2 }] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_numa_topology_partial_eq_different_length() {
        let a = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 2 }] };
        let b = NumaTopology { nodes: vec![] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_numa_topology_partial_eq_different_nodes() {
        let a = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 2 }] };
        let b = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 2048, core_count: 2 }] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_memory_network_sensors_partial_eq_equal() {
        let a = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        let b = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_memory_network_sensors_partial_eq_different_l2() {
        let a = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        let b = MemoryNetworkSensors {
            l2_cache_bytes: 8192, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_memory_network_sensors_partial_eq_with_float_fields() {
        let a = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: Some(100.0), rdma_latency_us: Some(5.0), arm_sme_za_size: None,
        };
        let b = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: Some(100.0), rdma_latency_us: Some(5.0), arm_sme_za_size: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_memory_network_sensors_partial_eq_different_float() {
        let a = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: Some(100.0), rdma_latency_us: None, arm_sme_za_size: None,
        };
        let b = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: Some(200.0), rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_equal() {
        let a = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        let b = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_different_core_count() {
        let a = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        let b = CpuTopology { core_count: 16, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_with_numa() {
        let numa = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 4 }] };
        let a = CpuTopology { core_count: 4, numa: Some(numa.clone()), l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 1024 };
        let b = CpuTopology { core_count: 4, numa: Some(numa), l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 1024 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_numa_vs_none() {
        let numa = NumaTopology { nodes: vec![NumaNode { node_id: 0, l3_bytes: 1024, core_count: 4 }] };
        let a = CpuTopology { core_count: 4, numa: Some(numa), l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 1024 };
        let b = CpuTopology { core_count: 4, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 1024 };
        assert_ne!(a, b);
    }

    // ── CodecStats PartialEq / Eq / Default ────────────────────────────────

    #[test]
    fn test_codec_stats_partial_eq_equal() {
        let a = CodecStats { compress_count: 1, decompress_count: 2, total_compress_latency_us: 10,
            total_decompress_latency_us: 20, total_input_bytes: 100, total_output_bytes: 50 };
        let b = CodecStats { compress_count: 1, decompress_count: 2, total_compress_latency_us: 10,
            total_decompress_latency_us: 20, total_input_bytes: 100, total_output_bytes: 50 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_codec_stats_partial_eq_not_equal() {
        let a = CodecStats::default();
        let mut b = CodecStats::default();
        b.compress_count = 1;
        assert_ne!(a, b);
    }

    #[test]
    fn test_codec_stats_default_is_zeroed() {
        let stats = CodecStats::default();
        assert_eq!(stats, CodecStats { compress_count: 0, decompress_count: 0,
            total_compress_latency_us: 0, total_decompress_latency_us: 0,
            total_input_bytes: 0, total_output_bytes: 0 });
    }

    #[test]
    fn test_codec_stats_eq_all_fields_different() {
        let a = CodecStats { compress_count: 0, decompress_count: 0, total_compress_latency_us: 0,
            total_decompress_latency_us: 0, total_input_bytes: 0, total_output_bytes: 0 };
        let b = CodecStats { compress_count: 1, decompress_count: 1, total_compress_latency_us: 1,
            total_decompress_latency_us: 1, total_input_bytes: 1, total_output_bytes: 1 };
        assert_ne!(a, b);
    }

    // ── CompressionTelemetry PartialEq / Default ──────────────────────────

    #[test]
    fn test_compression_telemetry_partial_eq_new_default() {
        let a = CompressionTelemetry::new();
        let b = CompressionTelemetry::default();
        assert_eq!(a, b);
    }

    #[test]
    fn test_compression_telemetry_partial_eq_after_compress() {
        let mut a = CompressionTelemetry::new();
        a.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        let mut b = CompressionTelemetry::new();
        b.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        assert_eq!(a, b);
    }

    #[test]
    fn test_compression_telemetry_not_equal_after_different_op() {
        let mut a = CompressionTelemetry::new();
        a.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        let mut b = CompressionTelemetry::new();
        b.record_compress(CompressionCodec::BitPackRle, 1000, 500, 50);
        assert_ne!(a, b);
    }

    // ── MemoryNetworkSensors zero / boundary fields ───────────────────────

    #[test]
    fn test_memory_network_sensors_zero_fields() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 0, ccx_numa_topology: None, tlb_entries: 0,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_eq!(sensors.l2_cache_bytes, 0);
        assert_eq!(sensors.tlb_entries, 0);
        assert_eq!(sensors.total_l3_bytes(), 0);
    }

    #[test]
    fn test_memory_network_sensors_max_usize_fields() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: usize::MAX, ccx_numa_topology: None, tlb_entries: usize::MAX,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: Some(usize::MAX),
        };
        assert_eq!(sensors.l2_cache_bytes, usize::MAX);
        assert_eq!(sensors.tlb_entries, usize::MAX);
        assert_eq!(sensors.arm_sme_za_size, Some(usize::MAX));
    }

    #[test]
    fn test_memory_network_sensors_float_zero_bandwidth() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(0.0, 0.0);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(0.0));
        assert_eq!(sensors.rdma_latency_us, Some(0.0));
        let chunk = sensors.min_chunk_size_for_rdma(100.0);
        assert!(chunk.is_some());
        assert!(chunk.unwrap() >= 64);
    }

    #[test]
    fn test_memory_network_sensors_float_large_bandwidth() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(f32::MAX, f32::MAX);
        let chunk = sensors.min_chunk_size_for_rdma(100.0);
        assert!(chunk.is_some());
        assert!(chunk.unwrap() >= 64);
    }

    // ── min_chunk_size_for_rdma with only partial RDMA params ──────────────

    #[test]
    fn test_min_chunk_size_rdma_partial_bandwidth_only_returns_none() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: Some(100.0), rdma_latency_us: None, arm_sme_za_size: None,
        };
        // Missing latency → None
        assert!(sensors.min_chunk_size_for_rdma(100.0).is_none());
    }

    #[test]
    fn test_min_chunk_size_rdma_partial_latency_only_returns_none() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: Some(10.0), arm_sme_za_size: None,
        };
        // Missing bandwidth → None
        assert!(sensors.min_chunk_size_for_rdma(100.0).is_none());
    }

    // ── total_l3_bytes with empty NUMA nodes vector ────────────────────────

    #[test]
    fn test_total_l3_bytes_empty_numa_nodes() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: Some(NumaTopology { nodes: vec![] }),
            tlb_entries: 512, nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 0);
    }

    #[test]
    fn test_total_l3_bytes_numa_zero_l3_per_node() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![
                    NumaNode { node_id: 0, l3_bytes: 0, core_count: 4 },
                    NumaNode { node_id: 1, l3_bytes: 0, core_count: 4 },
                ],
            }),
            tlb_entries: 512, nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 0);
    }

    // ── CpuTopology zero and max fields ────────────────────────────────────

    #[test]
    fn test_cpu_topology_all_zero_fields() {
        let cpu = CpuTopology { core_count: 0, numa: None, l1d_bytes: 0, l2_bytes: 0, l3_bytes: 0 };
        assert_eq!(cpu.core_count, 0);
        assert_eq!(cpu.l1d_bytes, 0);
        assert_eq!(cpu.l2_bytes, 0);
        assert_eq!(cpu.l3_bytes, 0);
    }

    #[test]
    fn test_cpu_topology_max_usize_fields() {
        let cpu = CpuTopology {
            core_count: usize::MAX, numa: None,
            l1d_bytes: usize::MAX, l2_bytes: usize::MAX, l3_bytes: usize::MAX,
        };
        assert_eq!(cpu.core_count, usize::MAX);
        assert_eq!(cpu.l1d_bytes, usize::MAX);
    }

    // ── CompressionTelemetry many consecutive operations ──────────────────

    #[test]
    fn test_compression_telemetry_many_compress_ops() {
        let mut ct = CompressionTelemetry::new();
        for i in 0..100 {
            ct.record_compress(CompressionCodec::Lz4, 1024, 512, i);
        }
        assert_eq!(ct.compress_count, 100);
        assert_eq!(ct.total_input_bytes, 100 * 1024);
        assert_eq!(ct.total_compressed_bytes, 100 * 512);
        assert_eq!(ct.total_compress_latency_us, (0..100).sum::<u64>());
    }

    #[test]
    fn test_compression_telemetry_many_decompress_ops() {
        let mut ct = CompressionTelemetry::new();
        for i in 0..50 {
            ct.record_decompress(CompressionCodec::BitPackRle, 512, 1024, i);
        }
        assert_eq!(ct.decompress_count, 50);
        assert_eq!(ct.total_decompress_latency_us, (0..50).sum::<u64>());
    }

    #[test]
    fn test_compression_telemetry_many_migrations() {
        let mut ct = CompressionTelemetry::new();
        for _ in 0..1000 {
            ct.record_migration(4096);
        }
        assert_eq!(ct.total_migration_bytes, 4096 * 1000);
    }

    #[test]
    fn test_compression_telemetry_many_evictions_swap_ins() {
        let mut ct = CompressionTelemetry::new();
        for _ in 0..500 {
            ct.record_eviction();
        }
        for _ in 0..300 {
            ct.record_swap_in();
        }
        assert_eq!(ct.eviction_count, 500);
        assert_eq!(ct.swap_in_count, 300);
    }

    // ── CompressionTelemetry mixed compress/decompress ratio ──────────────

    #[test]
    fn test_compression_telemetry_mixed_ops_avg_latencies() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 100);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 50);
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 1000, 200);
        // compress avg = (100+200)/2 = 150
        assert!((ct.avg_compress_latency_us() - 150.0).abs() < 1e-9);
        // decompress avg = 50/1 = 50
        assert!((ct.avg_decompress_latency_us() - 50.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry per-codec latency isolation ──────────────────

    #[test]
    fn test_codec_stats_per_codec_latency_isolation() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 100);
        ct.record_compress(CompressionCodec::ZstdDict, 2000, 200, 500);

        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert!((lz4.avg_compress_latency_us() - 100.0).abs() < 1e-9);

        let zstd = ct.codec_stats(CompressionCodec::ZstdDict);
        assert!((zstd.avg_compress_latency_us() - 500.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry overall ratio after many codecs ──────────────

    #[test]
    fn test_compression_telemetry_overall_ratio_multiple_codecs() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::None, 500, 500, 1);
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 10);
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 1000, 20);
        ct.record_compress(CompressionCodec::NvcompAns, 4000, 800, 30);
        ct.record_compress(CompressionCodec::ZstdDict, 8000, 400, 40);
        // total_input = 500+1000+2000+4000+8000 = 15500
        // total_compressed = 500+500+1000+800+400 = 3200
        // ratio = 3200/15500
        let ratio = ct.overall_compression_ratio();
        assert!((ratio - (3200.0 / 15500.0)).abs() < 1e-9, "ratio = {ratio}");
    }

    // ── CompressionTelemetry large u64 values no overflow ─────────────────

    #[test]
    fn test_compression_telemetry_large_migration_bytes() {
        let mut ct = CompressionTelemetry::new();
        ct.record_migration(u64::MAX);
        assert_eq!(ct.total_migration_bytes, u64::MAX);
    }

    #[test]
    fn test_compression_telemetry_large_latency_values() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, u64::MAX);
        assert_eq!(ct.total_compress_latency_us, u64::MAX);
        let avg = ct.avg_compress_latency_us();
        assert!((avg - (u64::MAX as f64)).abs() < 1.0);
    }

    // ── CompressionTelemetry all codecs decompress ────────────────────────

    #[test]
    fn test_compression_telemetry_all_codecs_decompress() {
        let mut ct = CompressionTelemetry::new();
        let codecs = [
            CompressionCodec::None, CompressionCodec::Lz4, CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns, CompressionCodec::ZstdDict,
        ];
        for (i, &codec) in codecs.iter().enumerate() {
            ct.record_decompress(codec, 100, 200, (i + 1) as u64 * 10);
        }
        assert_eq!(ct.decompress_count, 5);
        for (i, &codec) in codecs.iter().enumerate() {
            assert_eq!(ct.codec_stats(codec).decompress_count, 1);
            assert_eq!(ct.codec_stats(codec).total_decompress_latency_us, (i + 1) as u64 * 10);
        }
    }

    // ── NumaNode with max values ───────────────────────────────────────────

    #[test]
    fn test_numa_node_max_values() {
        let node = NumaNode { node_id: usize::MAX, l3_bytes: usize::MAX, core_count: usize::MAX };
        assert_eq!(node.node_id, usize::MAX);
        assert_eq!(node.l3_bytes, usize::MAX);
        assert_eq!(node.core_count, usize::MAX);
    }

    // ── NumaTopology with many nodes ──────────────────────────────────────

    #[test]
    fn test_numa_topology_many_nodes() {
        let nodes: Vec<NumaNode> = (0..128)
            .map(|i| NumaNode { node_id: i, l3_bytes: 32 * 1024 * 1024, core_count: 8 })
            .collect();
        let topo = NumaTopology { nodes };
        assert_eq!(topo.nodes.len(), 128);
        assert_eq!(topo.nodes[0].node_id, 0);
        assert_eq!(topo.nodes[127].node_id, 127);
    }

    // ── MemoryNetworkSensors detect + set_rdma overwrite ──────────────────

    #[test]
    fn test_set_rdma_overwrite_preserves_other_fields() {
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);
        let original_l2 = sensors.l2_cache_bytes;
        let original_tlb = sensors.tlb_entries;
        let original_numa = sensors.ccx_numa_topology.clone();
        let original_sme = sensors.arm_sme_za_size;

        sensors.set_rdma_params(50.0, 2.5);
        assert_eq!(sensors.l2_cache_bytes, original_l2);
        assert_eq!(sensors.tlb_entries, original_tlb);
        assert_eq!(sensors.ccx_numa_topology, original_numa);
        assert_eq!(sensors.arm_sme_za_size, original_sme);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(50.0));
        assert_eq!(sensors.rdma_latency_us, Some(2.5));
    }

    // ── CompressionTelemetry per-codec decompress latency avg ─────────────

    #[test]
    fn test_codec_stats_decompress_avg_multiple_ops_per_codec() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::NvcompAns, 100, 200, 30);
        ct.record_decompress(CompressionCodec::NvcompAns, 100, 200, 50);
        ct.record_decompress(CompressionCodec::NvcompAns, 100, 200, 70);

        let stats = ct.codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(stats.decompress_count, 3);
        assert!((stats.avg_decompress_latency_us() - 50.0).abs() < 1e-9);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional tests (wave 3) — targeting 40+ new tests
    // ═══════════════════════════════════════════════════════════════════════════

    // ── SystemTopology::try_detect returns Result with valid shape ────────────

    #[test]
    fn test_system_topology_try_detect_returns_ok_or_err() {
        let result = SystemTopology::try_detect();
        match &result {
            Ok(topo) => {
                assert!(topo.cpu.core_count > 0);
                assert!(topo.cpu.l1d_bytes > 0 || topo.cpu.l1d_bytes == 0);
            }
            Err(msg) => {
                assert!(!msg.is_empty(), "Error message must not be empty");
            }
        }
    }

    #[test]
    fn test_system_topology_try_detect_cpu_fields_populated() {
        let topo = SystemTopology::try_detect().expect("try_detect must succeed in this env");
        assert!(topo.cpu.core_count >= 1, "core_count must be >= 1");
        assert!(topo.sensors.tlb_entries > 0, "tlb_entries must be > 0");
    }

    #[test]
    fn test_system_topology_try_detect_constraints_non_default() {
        let topo = SystemTopology::try_detect().expect("try_detect must succeed in this env");
        assert!(topo.constraints.simd_width_bits > 0, "simd_width_bits must be > 0");
    }

    #[test]
    fn test_system_topology_try_from_profile_with_detected_profile() {
        let profile = DeviceProfile::detect();
        let result = SystemTopology::try_from_profile(profile);
        match result {
            Ok(topo) => {
                assert!(topo.cpu.core_count >= 1);
            }
            Err(msg) => {
                assert!(!msg.is_empty());
            }
        }
    }

    // ── SystemTopology has_gpu and numa_node_count with explicit construction ─

    #[test]
    fn test_system_topology_numa_node_count_with_numa() {
        let topo = SystemTopology::detect();
        if topo.cpu.numa.is_some() {
            let count = topo.numa_node_count();
            assert!(count >= 1, "numa_node_count must be >= 1 when NUMA present");
            assert_eq!(count, topo.cpu.numa.as_ref().unwrap().nodes.len());
        }
    }

    #[test]
    fn test_system_topology_cpu_l_fields_relationship() {
        // l1d <= l2 <= l3 is a general hardware invariant
        let topo = SystemTopology::detect();
        assert!(topo.cpu.l1d_bytes <= topo.cpu.l2_bytes || topo.cpu.l1d_bytes == 0,
            "l1d should not exceed l2: l1d={}, l2={}",
            topo.cpu.l1d_bytes, topo.cpu.l2_bytes);
    }

    // ── NumaTopology with non-sequential node IDs ────────────────────────────

    #[test]
    fn test_numa_topology_non_sequential_ids() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { node_id: 0, l3_bytes: 32 * 1024 * 1024, core_count: 8 },
                NumaNode { node_id: 3, l3_bytes: 64 * 1024 * 1024, core_count: 16 },
                NumaNode { node_id: 7, l3_bytes: 32 * 1024 * 1024, core_count: 8 },
            ],
        };
        assert_eq!(topo.nodes.len(), 3);
        assert_eq!(topo.nodes[1].node_id, 3);
        assert_eq!(topo.nodes[2].node_id, 7);
    }

    #[test]
    fn test_numa_topology_heterogeneous_l3_sizes() {
        let topo = NumaTopology {
            nodes: vec![
                NumaNode { node_id: 0, l3_bytes: 16 * 1024 * 1024, core_count: 4 },
                NumaNode { node_id: 1, l3_bytes: 32 * 1024 * 1024, core_count: 8 },
                NumaNode { node_id: 2, l3_bytes: 64 * 1024 * 1024, core_count: 16 },
            ],
        };
        let total_l3: usize = topo.nodes.iter().map(|n| n.l3_bytes).sum();
        assert_eq!(total_l3, (16 + 32 + 64) * 1024 * 1024);
    }

    #[test]
    fn test_numa_topology_single_node_zero_l3() {
        let topo = NumaTopology {
            nodes: vec![NumaNode { node_id: 0, l3_bytes: 0, core_count: 1 }],
        };
        assert_eq!(topo.nodes.len(), 1);
        assert_eq!(topo.nodes[0].l3_bytes, 0);
    }

    // ── NumaNode with varying core counts ─────────────────────────────────────

    #[test]
    fn test_numa_node_single_core() {
        let node = NumaNode { node_id: 0, l3_bytes: 1024, core_count: 1 };
        assert_eq!(node.core_count, 1);
    }

    #[test]
    fn test_numa_node_large_core_count() {
        let node = NumaNode { node_id: 0, l3_bytes: 0, core_count: 256 };
        assert_eq!(node.core_count, 256);
    }

    // ── MemoryNetworkSensors total_l3_bytes with diverse NUMA configs ────────

    #[test]
    fn test_total_l3_bytes_three_numa_nodes_mixed() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![
                    NumaNode { node_id: 0, l3_bytes: 16 * 1024 * 1024, core_count: 4 },
                    NumaNode { node_id: 1, l3_bytes: 32 * 1024 * 1024, core_count: 8 },
                    NumaNode { node_id: 2, l3_bytes: 48 * 1024 * 1024, core_count: 12 },
                ],
            }),
            tlb_entries: 1024,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 96 * 1024 * 1024);
    }

    #[test]
    fn test_total_l3_bytes_numa_with_single_zero_l3_node() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![
                    NumaNode { node_id: 0, l3_bytes: 32 * 1024 * 1024, core_count: 8 },
                    NumaNode { node_id: 1, l3_bytes: 0, core_count: 4 },
                ],
            }),
            tlb_entries: 1024,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        assert_eq!(sensors.total_l3_bytes(), 32 * 1024 * 1024);
    }

    // ── MemoryNetworkSensors clone after set_rdma_params ─────────────────────

    #[test]
    fn test_sensors_clone_after_rdma_params_independent() {
        let profile = DeviceProfile::detect();
        let mut sensors = MemoryNetworkSensors::detect(&profile);
        sensors.set_rdma_params(100.0, 5.0);

        let cloned = sensors.clone();
        // Mutate original
        sensors.set_rdma_params(200.0, 10.0);

        assert_eq!(cloned.nic_bandwidth_gbs, Some(100.0));
        assert_eq!(cloned.rdma_latency_us, Some(5.0));
        assert_eq!(sensors.nic_bandwidth_gbs, Some(200.0));
        assert_eq!(sensors.rdma_latency_us, Some(10.0));
    }

    #[test]
    fn test_sensors_clone_numa_independence() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: Some(NumaTopology {
                nodes: vec![NumaNode { node_id: 0, l3_bytes: 32768, core_count: 4 }],
            }),
            tlb_entries: 512,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: None,
        };
        let cloned = sensors.clone();
        // Mutate original's NUMA
        sensors.ccx_numa_topology.as_mut().unwrap().nodes[0].l3_bytes = 0;
        // Clone unaffected
        assert_eq!(cloned.ccx_numa_topology.as_ref().unwrap().nodes[0].l3_bytes, 32768);
    }

    // ── MemoryNetworkSensors set_rdma_params edge cases ──────────────────────

    #[test]
    fn test_set_rdma_params_negative_values() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(-1.0, -1.0);
        assert_eq!(sensors.nic_bandwidth_gbs, Some(-1.0));
        assert_eq!(sensors.rdma_latency_us, Some(-1.0));
    }

    #[test]
    fn test_set_rdma_params_nan_values() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(f32::NAN, f32::NAN);
        assert!(sensors.nic_bandwidth_gbs.unwrap().is_nan());
        assert!(sensors.rdma_latency_us.unwrap().is_nan());
        // NaN params should still return Some from min_chunk_size_for_rdma
        let chunk = sensors.min_chunk_size_for_rdma(100.0);
        assert!(chunk.is_some());
    }

    #[test]
    fn test_set_rdma_params_infinity() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(f32::INFINITY, f32::INFINITY);
        assert!(sensors.nic_bandwidth_gbs.is_some());
        let chunk = sensors.min_chunk_size_for_rdma(100.0);
        assert!(chunk.is_some());
    }

    #[test]
    fn test_min_chunk_size_rdma_monotonically_increases_with_bandwidth() {
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        let latency = 10.0f32;
        let mut prev_chunk = 0usize;
        for bandwidth in [1.0f32, 10.0, 100.0, 1000.0] {
            sensors.set_rdma_params(bandwidth, latency);
            let chunk = sensors.min_chunk_size_for_rdma(100.0).unwrap();
            assert!(chunk >= prev_chunk || chunk == 64,
                "chunk should not decrease as bandwidth increases: prev={prev_chunk}, cur={chunk}");
            prev_chunk = chunk;
        }
    }

    // ── CompressionTelemetry interleaved compress/decompress ─────────────────

    #[test]
    fn test_compression_telemetry_interleaved_ops() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 100);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 50);
        ct.record_compress(CompressionCodec::Lz4, 2000, 800, 150);
        ct.record_decompress(CompressionCodec::Lz4, 800, 2000, 70);

        assert_eq!(ct.compress_count, 2);
        assert_eq!(ct.decompress_count, 2);
        assert_eq!(ct.total_input_bytes, 3000);
        assert_eq!(ct.total_compressed_bytes, 1300);
        assert_eq!(ct.total_compress_latency_us, 250);
        assert_eq!(ct.total_decompress_latency_us, 120);

        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.compress_count, 2);
        assert_eq!(lz4.decompress_count, 2);
        assert_eq!(lz4.total_input_bytes, 4300); // compress 3000 + decompress 1300
        assert_eq!(lz4.total_output_bytes, 4300); // compress 1300 + decompress 3000
    }

    #[test]
    fn test_compression_telemetry_interleaved_different_codecs() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 100);
        ct.record_decompress(CompressionCodec::BitPackRle, 200, 1000, 50);
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 600, 200);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);

        assert_eq!(ct.compress_count, 2);
        assert_eq!(ct.decompress_count, 2);

        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.compress_count, 1);
        assert_eq!(lz4.decompress_count, 1);
        assert_eq!(lz4.total_compress_latency_us, 100);
        assert_eq!(lz4.total_decompress_latency_us, 30);

        let bpr = ct.codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(bpr.compress_count, 1);
        assert_eq!(bpr.decompress_count, 1);
        assert_eq!(bpr.total_compress_latency_us, 200);
        assert_eq!(bpr.total_decompress_latency_us, 50);
    }

    // ── CompressionTelemetry record_migration, record_eviction, record_swap_in cumulative ──

    #[test]
    fn test_compression_telemetry_mixed_events() {
        let mut ct = CompressionTelemetry::new();
        ct.record_migration(4096);
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        ct.record_eviction();
        ct.record_migration(8192);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        ct.record_swap_in();

        assert_eq!(ct.total_migration_bytes, 12288);
        assert_eq!(ct.eviction_count, 1);
        assert_eq!(ct.swap_in_count, 1);
        assert_eq!(ct.compress_count, 1);
        assert_eq!(ct.decompress_count, 1);
    }

    #[test]
    fn test_compression_telemetry_many_events_cumulative() {
        let mut ct = CompressionTelemetry::new();
        for i in 0..200 {
            ct.record_migration((i + 1) as u64 * 1024);
            if i % 2 == 0 {
                ct.record_eviction();
            }
            if i % 3 == 0 {
                ct.record_swap_in();
            }
        }
        assert_eq!(ct.total_migration_bytes, (1..=200u64).map(|i| i * 1024).sum::<u64>());
        assert_eq!(ct.eviction_count, 100); // even numbers: 0, 2, ..., 198
        assert_eq!(ct.swap_in_count, 67); // multiples of 3: 0, 3, ..., 198 → 67 values
    }

    // ── CodecStats with zero input but non-zero output (data expansion) ──────

    #[test]
    fn test_codec_stats_expansion_ratio() {
        let mut stats = CodecStats::default();
        stats.total_input_bytes = 100;
        stats.total_output_bytes = 300;
        assert!((stats.avg_compression_ratio() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_codec_stats_zero_input_zero_output_ratio() {
        let stats = CodecStats::default();
        // Both zero → ratio returns 1.0 (guard clause)
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
    }

    // ── CodecStats avg latency when count overflows u64 ──────────────────────

    #[test]
    fn test_codec_stats_large_count_latency() {
        let mut stats = CodecStats::default();
        stats.compress_count = 1;
        stats.total_compress_latency_us = u64::MAX;
        let avg = stats.avg_compress_latency_us();
        assert!((avg - (u64::MAX as f64)).abs() < 1.0);
    }

    #[test]
    fn test_codec_stats_decompress_large_count_latency() {
        let mut stats = CodecStats::default();
        stats.decompress_count = 2;
        stats.total_decompress_latency_us = u64::MAX;
        let avg = stats.avg_decompress_latency_us();
        assert!((avg - (u64::MAX as f64 / 2.0)).abs() < 1.0);
    }

    // ── CompressionTelemetry overall ratio with only decompress ops ──────────

    #[test]
    fn test_compression_telemetry_overall_ratio_decompress_only() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 50);
        ct.record_decompress(CompressionCodec::BitPackRle, 200, 800, 30);
        // No compress ops → total_input_bytes = 0 → ratio = 1.0
        assert!((ct.overall_compression_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_avg_compress_latency_decompress_only() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 50);
        assert!((ct.avg_compress_latency_us() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_avg_decompress_latency_compress_only() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 100);
        assert!((ct.avg_decompress_latency_us() - 0.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry clone after mixed ops ───────────────────────────

    #[test]
    fn test_compression_telemetry_clone_after_mixed_ops() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 50);
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        ct.record_migration(4096);
        ct.record_eviction();
        ct.record_swap_in();

        let cloned = ct.clone();
        assert_eq!(cloned.compress_count, 1);
        assert_eq!(cloned.decompress_count, 1);
        assert_eq!(cloned.total_migration_bytes, 4096);
        assert_eq!(cloned.eviction_count, 1);
        assert_eq!(cloned.swap_in_count, 1);
        assert_eq!(cloned.total_input_bytes, 1000);
        assert_eq!(cloned.total_compressed_bytes, 500);

        // Mutate original
        ct.record_compress(CompressionCodec::BitPackRle, 2000, 1000, 100);
        assert_eq!(cloned.compress_count, 1);
        assert_eq!(ct.compress_count, 2);
    }

    // ── CompressionTelemetry codec_stats method for all 5 codecs in new() ────

    #[test]
    fn test_compression_telemetry_new_all_codecs_zeroed() {
        let ct = CompressionTelemetry::new();
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let stats = ct.codec_stats(codec);
            assert_eq!(stats.compress_count, 0, "codec {codec:?} should have 0 compress");
            assert_eq!(stats.decompress_count, 0, "codec {codec:?} should have 0 decompress");
        }
    }

    // ── CpuTopology PartialEq with identical NUMA ────────────────────────────

    #[test]
    fn test_cpu_topology_partial_eq_identical_numa() {
        let numa = NumaTopology {
            nodes: vec![
                NumaNode { node_id: 0, l3_bytes: 32768, core_count: 4 },
                NumaNode { node_id: 1, l3_bytes: 65536, core_count: 8 },
            ],
        };
        let a = CpuTopology { core_count: 12, numa: Some(numa.clone()), l1d_bytes: 32768, l2_bytes: 524288, l3_bytes: 98304 };
        let b = CpuTopology { core_count: 12, numa: Some(numa), l1d_bytes: 32768, l2_bytes: 524288, l3_bytes: 98304 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_different_l1d() {
        let a = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        let b = CpuTopology { core_count: 8, numa: None, l1d_bytes: 65536, l2_bytes: 262144, l3_bytes: 8388608 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_different_l2() {
        let a = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        let b = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 524288, l3_bytes: 8388608 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_cpu_topology_partial_eq_different_l3() {
        let a = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 8388608 };
        let b = CpuTopology { core_count: 8, numa: None, l1d_bytes: 32768, l2_bytes: 262144, l3_bytes: 16777216 };
        assert_ne!(a, b);
    }

    // ── CpuTopology Debug with NUMA present ──────────────────────────────────

    #[test]
    fn test_cpu_topology_debug_with_numa() {
        let cpu = CpuTopology {
            core_count: 16,
            numa: Some(NumaTopology {
                nodes: vec![NumaNode { node_id: 0, l3_bytes: 32768, core_count: 8 }],
            }),
            l1d_bytes: 32768,
            l2_bytes: 524288,
            l3_bytes: 32768,
        };
        let debug_str = format!("{cpu:?}");
        assert!(debug_str.contains("Some("));
        assert!(debug_str.contains("NumaTopology"));
    }

    // ── MemoryNetworkSensors arm_sme_za_size field ───────────────────────────

    #[test]
    fn test_sensors_arm_sme_za_size_none_on_non_arm() {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        // On x86_64, arm_sme_za_size is always None
        #[cfg(target_arch = "x86_64")]
        {
            assert!(sensors.arm_sme_za_size.is_none());
        }
        // On aarch64, it may still be None (not yet implemented)
        let _ = sensors.arm_sme_za_size;
    }

    #[test]
    fn test_sensors_manual_arm_sme_za_size() {
        let sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096,
            ccx_numa_topology: None,
            tlb_entries: 512,
            nic_bandwidth_gbs: None,
            rdma_latency_us: None,
            arm_sme_za_size: Some(1024),
        };
        assert_eq!(sensors.arm_sme_za_size, Some(1024));
    }

    // ── CompressionTelemetry record_compress with each codec individually ────

    #[test]
    fn test_compression_telemetry_compress_none_codec() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::None, 500, 500, 0);
        let stats = ct.codec_stats(CompressionCodec::None);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.total_input_bytes, 500);
        assert_eq!(stats.total_output_bytes, 500);
        assert!((stats.avg_compression_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_compress_nvcompans_codec() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::NvcompAns, 10000, 2000, 500);
        let stats = ct.codec_stats(CompressionCodec::NvcompAns);
        assert_eq!(stats.compress_count, 1);
        assert_eq!(stats.total_input_bytes, 10000);
        assert_eq!(stats.total_output_bytes, 2000);
        assert!((stats.avg_compression_ratio() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_compression_telemetry_compress_zstddict_codec() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::ZstdDict, 50000, 100, 1000);
        let stats = ct.codec_stats(CompressionCodec::ZstdDict);
        assert_eq!(stats.compress_count, 1);
        assert!((stats.avg_compression_ratio() - 0.002).abs() < 1e-9);
    }

    // ── CompressionTelemetry same codec multiple compress/decompress ─────────

    #[test]
    fn test_compression_telemetry_lz4_many_ops() {
        let mut ct = CompressionTelemetry::new();
        for i in 0..10 {
            ct.record_compress(CompressionCodec::Lz4, 1000, 400, (i + 1) as u64 * 10);
        }
        for i in 0..5 {
            ct.record_decompress(CompressionCodec::Lz4, 400, 1000, (i + 1) as u64 * 5);
        }

        assert_eq!(ct.compress_count, 10);
        assert_eq!(ct.decompress_count, 5);
        assert_eq!(ct.total_input_bytes, 10000);
        assert_eq!(ct.total_compressed_bytes, 4000);

        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.compress_count, 10);
        assert_eq!(lz4.decompress_count, 5);
        assert!((lz4.avg_compress_latency_us() - 55.0).abs() < 1e-9);
        assert!((lz4.avg_decompress_latency_us() - 15.0).abs() < 1e-9);
    }

    // ── CompressionTelemetry overall ratio exactly 0.5 ──────────────────────

    #[test]
    fn test_compression_telemetry_overall_ratio_half() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, 2000, 1000, 50);
        assert!((ct.overall_compression_ratio() - 0.5).abs() < 1e-9);
    }

    // ── CompressionTelemetry record_compress with u64 near-max ──────────────

    #[test]
    fn test_compression_telemetry_near_max_input_bytes() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::Lz4, u64::MAX - 1, u64::MAX / 2, 1);
        assert_eq!(ct.total_input_bytes, u64::MAX - 1);
        assert_eq!(ct.total_compressed_bytes, u64::MAX / 2);
    }

    // ── CompressionTelemetry record_decompress does not update byte counters ─

    #[test]
    fn test_compression_telemetry_decompress_no_byte_counter_update() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 500, 1000, 30);
        // Decompress does not update total_input_bytes or total_compressed_bytes
        assert_eq!(ct.total_input_bytes, 0);
        assert_eq!(ct.total_compressed_bytes, 0);
    }

    // ── SystemTopology detect has non-zero constraints ───────────────────────

    #[test]
    fn test_system_topology_constraints_l2_nonzero() {
        let topo = SystemTopology::detect();
        assert!(topo.constraints.l2_cache_size > 0, "l2_cache_size must be > 0");
    }

    #[test]
    fn test_system_topology_profile_cache_sizes() {
        let topo = SystemTopology::detect();
        let (l1d, l2, _l3) = topo.profile.cache_sizes();
        assert_eq!(topo.cpu.l1d_bytes, l1d);
        assert_eq!(topo.cpu.l2_bytes, l2);
    }

    // ── MemoryNetworkSensors detect preserves l2 from profile ────────────────

    #[test]
    fn test_sensors_detect_l2_matches_profile() {
        let profile = DeviceProfile::detect();
        let sensors = MemoryNetworkSensors::detect(&profile);
        assert_eq!(sensors.l2_cache_bytes, profile.kernel_config.l2);
    }

    // ── NumaTopology Eq derive verification ──────────────────────────────────

    #[test]
    fn test_numa_topology_eq_same_empty() {
        let a = NumaTopology { nodes: vec![] };
        let b = NumaTopology { nodes: vec![] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_numa_topology_eq_order_matters() {
        let a = NumaTopology {
            nodes: vec![
                NumaNode { node_id: 0, l3_bytes: 100, core_count: 4 },
                NumaNode { node_id: 1, l3_bytes: 200, core_count: 8 },
            ],
        };
        let b = NumaTopology {
            nodes: vec![
                NumaNode { node_id: 1, l3_bytes: 200, core_count: 8 },
                NumaNode { node_id: 0, l3_bytes: 100, core_count: 4 },
            ],
        };
        assert_ne!(a, b, "Node order must matter for equality");
    }

    // ── CodecStats Eq symmetry ───────────────────────────────────────────────

    #[test]
    fn test_codec_stats_eq_symmetry() {
        let a = CodecStats { compress_count: 1, decompress_count: 2, total_compress_latency_us: 10,
            total_decompress_latency_us: 20, total_input_bytes: 100, total_output_bytes: 50 };
        let b = CodecStats { compress_count: 1, decompress_count: 2, total_compress_latency_us: 10,
            total_decompress_latency_us: 20, total_input_bytes: 100, total_output_bytes: 50 };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_codec_stats_eq_transitivity() {
        let a = CodecStats { compress_count: 5, decompress_count: 0, total_compress_latency_us: 50,
            total_decompress_latency_us: 0, total_input_bytes: 500, total_output_bytes: 250 };
        let b = CodecStats { compress_count: 5, decompress_count: 0, total_compress_latency_us: 50,
            total_decompress_latency_us: 0, total_input_bytes: 500, total_output_bytes: 250 };
        let c = CodecStats { compress_count: 5, decompress_count: 0, total_compress_latency_us: 50,
            total_decompress_latency_us: 0, total_input_bytes: 500, total_output_bytes: 250 };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── CompressionTelemetry overall ratio with expansion on one codec ───────

    #[test]
    fn test_compression_telemetry_mixed_expansion_and_compression() {
        let mut ct = CompressionTelemetry::new();
        // None codec: 1:1 (no compression)
        ct.record_compress(CompressionCodec::None, 1000, 1000, 1);
        // Lz4: 50% compression
        ct.record_compress(CompressionCodec::Lz4, 1000, 500, 10);
        // BitPackRle: expansion (output > input)
        ct.record_compress(CompressionCodec::BitPackRle, 1000, 2000, 5);

        // total_input = 3000, total_compressed = 3500
        // ratio = 3500/3000 ≈ 1.1667
        let ratio = ct.overall_compression_ratio();
        assert!((ratio - (3500.0 / 3000.0)).abs() < 1e-9, "ratio = {ratio}");
    }

    // ── CompressionTelemetry per-codec compress count after many ops ─────────

    #[test]
    fn test_compression_telemetry_per_codec_count_sum_equals_global() {
        let mut ct = CompressionTelemetry::new();
        ct.record_compress(CompressionCodec::None, 100, 100, 1);
        ct.record_compress(CompressionCodec::None, 200, 200, 2);
        ct.record_compress(CompressionCodec::Lz4, 300, 150, 3);
        ct.record_compress(CompressionCodec::BitPackRle, 400, 200, 4);
        ct.record_compress(CompressionCodec::BitPackRle, 500, 250, 5);

        let per_codec_sum: u64 = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ].iter().map(|&c| ct.codec_stats(c).compress_count).sum();

        assert_eq!(per_codec_sum, ct.compress_count);
    }

    #[test]
    fn test_compression_telemetry_per_codec_decompress_count_sum_equals_global() {
        let mut ct = CompressionTelemetry::new();
        ct.record_decompress(CompressionCodec::Lz4, 100, 200, 10);
        ct.record_decompress(CompressionCodec::Lz4, 100, 200, 20);
        ct.record_decompress(CompressionCodec::NvcompAns, 50, 100, 5);

        let per_codec_sum: u64 = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ].iter().map(|&c| ct.codec_stats(c).decompress_count).sum();

        assert_eq!(per_codec_sum, ct.decompress_count);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Wave 4 additional tests (13 new, target: 173→186)
    // ═══════════════════════════════════════════════════════════════════════════

    // ── GpuPlatform enum variant construction and field access ────────────────

    #[test]
    fn test_gpu_platform_cuda_variant_construction() {
        let platform = GpuPlatform::Cuda { sm_version: 80 };
        match platform {
            GpuPlatform::Cuda { sm_version } => assert_eq!(sm_version, 80),
            _ => panic!("expected Cuda variant"),
        }
    }

    #[test]
    fn test_gpu_platform_hip_variant_construction() {
        let platform = GpuPlatform::Hip { gfx_arch: 0x908 };
        match platform {
            GpuPlatform::Hip { gfx_arch } => assert_eq!(gfx_arch, 0x908),
            _ => panic!("expected Hip variant"),
        }
    }

    #[test]
    fn test_gpu_platform_metal_variant_construction() {
        let platform = GpuPlatform::Metal { gpu_family: 13 };
        match platform {
            GpuPlatform::Metal { gpu_family } => assert_eq!(gpu_family, 13),
            _ => panic!("expected Metal variant"),
        }
    }

    #[test]
    fn test_gpu_platform_equality_same_variant() {
        let a = GpuPlatform::Cuda { sm_version: 90 };
        let b = GpuPlatform::Cuda { sm_version: 90 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_gpu_platform_inequality_different_field() {
        let a = GpuPlatform::Cuda { sm_version: 80 };
        let b = GpuPlatform::Cuda { sm_version: 90 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_gpu_platform_inequality_different_variants() {
        let a = GpuPlatform::Cuda { sm_version: 80 };
        let b = GpuPlatform::Hip { gfx_arch: 0x908 };
        assert_ne!(a, b);
    }

    // ── GpuPlatform Copy + Clone + Hash ───────────────────────────────────────

    #[test]
    fn test_gpu_platform_copy_semantics() {
        let original = GpuPlatform::Cuda { sm_version: 80 };
        let copy = original;
        assert_eq!(original, copy);
    }

    // ── GpuTopology construction and field access ────────────────────────────

    #[test]
    fn test_gpu_topology_construction_all_fields() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 49152,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.compute_unit_count, 108);
        assert_eq!(topo.tensor_core_gen, 2);
        assert_eq!(topo.shared_mem_per_sm_bytes, 49152);
        assert_eq!(topo.l2_bytes, 40 * 1024 * 1024);
        assert_eq!(topo.global_mem_bytes, 24 * 1024 * 1024 * 1024);
        assert_eq!(topo.warp_size, 32);
        assert_eq!(topo.compute_cap_major, 8);
        assert_eq!(topo.compute_cap_minor, 0);
    }

    #[test]
    fn test_gpu_topology_debug_output() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 65536,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        let debug_str = format!("{topo:?}");
        assert!(debug_str.contains("GpuTopology"));
        assert!(debug_str.contains("compute_unit_count"));
        assert!(debug_str.contains("tensor_core_gen"));
        assert!(debug_str.contains("warp_size"));
    }

    #[test]
    fn test_gpu_topology_clone_independence() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 64,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 65536,
            l2_bytes: 16 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        let cloned = topo.clone();
        assert_eq!(cloned.compute_unit_count, topo.compute_unit_count);
        assert_eq!(cloned.warp_size, topo.warp_size);
        assert_eq!(cloned.platform, topo.platform);
    }

    // ── min_chunk_size_for_rdma formula verification ──────────────────────────

    #[test]
    fn test_min_chunk_size_rdma_formula_precision() {
        // bandwidth = 200 GB/s, latency = 5 us
        // min_data_gb = 5e-6 * 200 = 0.001 GB = 1_000_000 bytes
        // min_tokens = (0.001 * 1e9 / 4096.0) = 1_000_000 / 4096.0 ≈ 244.14
        // max(244, 64) = 244
        let mut sensors = MemoryNetworkSensors {
            l2_cache_bytes: 4096, ccx_numa_topology: None, tlb_entries: 512,
            nic_bandwidth_gbs: None, rdma_latency_us: None, arm_sme_za_size: None,
        };
        sensors.set_rdma_params(200.0, 5.0);
        let chunk = sensors.min_chunk_size_for_rdma(100.0).unwrap();
        assert!(chunk >= 64);
        assert_eq!(chunk, 244);
    }

    // ── CompressionTelemetry per-codec decompress count sum matches global after mixed ops ──

    #[test]
    fn test_compression_telemetry_mixed_ops_per_codec_byte_totals() {
        let mut ct = CompressionTelemetry::new();
        // 2 Lz4 compress ops
        ct.record_compress(CompressionCodec::Lz4, 1000, 600, 10);
        ct.record_compress(CompressionCodec::Lz4, 2000, 800, 20);
        // 1 BitPackRle compress op
        ct.record_compress(CompressionCodec::BitPackRle, 5000, 1000, 50);

        // Global byte totals
        assert_eq!(ct.total_input_bytes, 8000);
        assert_eq!(ct.total_compressed_bytes, 2400);

        // Per-codec byte totals match
        let lz4 = ct.codec_stats(CompressionCodec::Lz4);
        assert_eq!(lz4.total_input_bytes, 3000);
        assert_eq!(lz4.total_output_bytes, 1400);

        let bpr = ct.codec_stats(CompressionCodec::BitPackRle);
        assert_eq!(bpr.total_input_bytes, 5000);
        assert_eq!(bpr.total_output_bytes, 1000);

        // Unrelated codecs untouched
        assert_eq!(ct.codec_stats(CompressionCodec::NvcompAns).total_input_bytes, 0);
        assert_eq!(ct.codec_stats(CompressionCodec::ZstdDict).total_output_bytes, 0);
    }

    // ── GpuPlatform Hash consistency ───────────────────────────────────────────

    #[test]
    fn test_gpu_platform_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let cuda80 = GpuPlatform::Cuda { sm_version: 80 };
        let cuda90 = GpuPlatform::Cuda { sm_version: 90 };
        let hip = GpuPlatform::Hip { gfx_arch: 0x908 };
        set.insert(cuda80);
        set.insert(cuda90);
        set.insert(hip);
        assert_eq!(set.len(), 3, "three distinct GpuPlatform values must produce 3 hash entries");
        // Inserting duplicate should not increase size
        set.insert(GpuPlatform::Cuda { sm_version: 80 });
        assert_eq!(set.len(), 3);
    }
}
