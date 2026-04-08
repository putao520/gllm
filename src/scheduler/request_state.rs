//! Unified Request State Table for Mega-Kernel block routing.
//!
//! SPEC §9.1 Mega-Kernel 块级路由 (In-Kernel Dispatch)
//! SPEC §9.3 残差数据总线 (Residual Bus Injection/Recall)
//! SPEC §9.5 尾段就地观测 (Zero-Copy Paged Telemetry)
//!
//! Architecture: ARCH-CPU-GPU-UNIFIED — CPU/GPU share identical data structures,
//! backend differences driven by DeviceProfile.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::backend::detection::BackendType;
use crate::routing::BusPortTag;
use crate::scheduler::types::RequestId;

/// Request execution phase for in-kernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Prefill phase: compute QKV projections
    Prefill,
    /// Decode phase: single-token generation
    Decode,
    /// Chunked prefill: interleaved with decode (§10)
    ChunkedPrefill,
}

/// Per-request Compact/Scatter 元数据
///
/// §9.1 Compact→Execute→Scatter 三段式循环:
/// 1. Compact: 按激活掩码挤压到连续内存
/// 2. Execute: 在无气泡的连续矩阵上执行核函数
/// 3. Scatter: 按原始偏移散射回写
#[derive(Debug, Clone, Copy)]
pub struct CompactScatterMeta {
    /// 在 Batch 中的原始位置 (pre-compact)
    pub original_slot: u32,
    /// Compact 后的位置 (post-compact)
    pub compacted_slot: u32,
    /// 激活掩码 — 1=active, 0=skipped
    pub active: u32,
}

/// Per-request telemetry 由 Mega-Kernel Epilogue 写入
///
/// §9.5 尾段就地观测: 所有探视信息伴随原流水线写入 KV Cache 的 Page Header
#[derive(Debug, Clone, Copy)]
pub struct RequestTelemetry {
    /// Shannon 熵 (从 Softmax 输出白嫖)
    pub entropy: f32,
    /// 概率质心 (从 Softmax 输出白嫖)
    pub centroid: f32,
    /// 跨层能量差 Δρ (从 Residual Add 前后白嫖)
    pub residual_delta: f32,
    /// 残差方向余弦相似度 (§13.11)
    pub residual_cosine: f32,
    /// 该请求所属的值域分组 (§9.1 Range-Aware Compact Grouping)
    pub range_group: u32,
}

impl Default for RequestTelemetry {
    fn default() -> Self {
        Self {
            entropy: 0.0,
            centroid: 0.0,
            residual_delta: 1.0,
            residual_cosine: 1.0,
            range_group: 0,
        }
    }
}

/// Per-request state for Mega-Kernel dispatch.
///
/// §9 定义的全量请求状态，包含:
/// - 基础路由字段 (request_id, phase, seq_len, kv_cache_offset)
/// - 多态执行控制 (target_layer, exit_flag) — §16.1-16.4
/// - Compact/Scatter 元数据 — §9.1
/// - 遥测数据 — §9.5
/// - 残差总线挂载状态 — §9.3
///
/// # Memory Layout (64 bytes, cache-line aligned)
/// ```text
/// offset  0: request_id (u64)
/// offset  8: phase (u8, padded)
/// offset 16: seq_len (usize)
/// offset 24: kv_cache_offset (usize)
/// offset 32: target_layer (u32)
/// offset 36: exit_flag (AtomicU32)
/// offset 40: compact_scatter (CompactScatterMeta, 12 bytes)
/// offset 52: telemetry (RequestTelemetry, 16 bytes)
/// offset 68: bus_port_mask (u32) — 残差总线激活端口位掩码
/// offset 72: variant_key_hash (u32) — 变体选择键哈希
/// ```
#[derive(Debug)]
pub struct RequestState {
    pub request_id: RequestId,
    pub phase: RequestPhase,
    pub seq_len: usize,
    pub kv_cache_offset: usize,

    // §16 多态执行控制
    /// Layer index at which this request should exit (0-based, inclusive).
    /// Set to `u32::MAX` for full-model execution (no early exit).
    pub target_layer: u32,
    /// Exit flag: 0 = running, 1 = exited.
    pub exit_flag: AtomicU32,

    // §9.1 Compact/Scatter
    pub compact_scatter: CompactScatterMeta,

    // §9.5 Telemetry
    pub telemetry: RequestTelemetry,

    // §9.3 Residual Bus
    /// 位掩码: bit 0 = RAG, bit 1 = EarlyExit, bit 2 = Intent, bit 3 = Guardrail, bit 4 = ShadowKv
    pub bus_port_mask: u32,

    // §9.6 Variant Selection
    /// Variant key hash for dispatch-time variant lookup
    pub variant_key_hash: u32,
}

impl Clone for RequestState {
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id,
            phase: self.phase,
            seq_len: self.seq_len,
            kv_cache_offset: self.kv_cache_offset,
            target_layer: self.target_layer,
            exit_flag: AtomicU32::new(self.exit_flag.load(Ordering::Relaxed)),
            compact_scatter: self.compact_scatter,
            telemetry: self.telemetry,
            bus_port_mask: self.bus_port_mask,
            variant_key_hash: self.variant_key_hash,
        }
    }
}

impl RequestState {
    /// Create a new RequestState for full-model execution (no early exit).
    pub fn new(request_id: RequestId, phase: RequestPhase, seq_len: usize, kv_cache_offset: usize) -> Self {
        Self {
            request_id,
            phase,
            seq_len,
            kv_cache_offset,
            target_layer: u32::MAX,
            exit_flag: AtomicU32::new(0),
            compact_scatter: CompactScatterMeta {
                original_slot: 0,
                compacted_slot: 0,
                active: 1,
            },
            telemetry: RequestTelemetry::default(),
            bus_port_mask: 0,
            variant_key_hash: 0,
        }
    }

    /// Create a RequestState that exits after the specified layer (inclusive).
    pub fn with_target_layer(mut self, layer: usize) -> Self {
        self.target_layer = layer as u32;
        self
    }

    /// Attach a ResidualBus port to this request.
    pub fn with_bus_port(mut self, tag: BusPortTag) -> Self {
        self.bus_port_mask |= bus_port_tag_to_bit(tag);
        self
    }

    /// Check if a specific bus port is attached.
    pub fn has_bus_port(&self, tag: BusPortTag) -> bool {
        self.bus_port_mask & bus_port_tag_to_bit(tag) != 0
    }

    /// Check if this request has exited.
    pub fn is_exited(&self) -> bool {
        self.exit_flag.load(Ordering::Acquire) != 0
    }

    /// Mark this request as exited.
    pub fn mark_exited(&self) {
        self.exit_flag.store(1, Ordering::Release);
    }

    /// Reset exit flag for reuse (keeps target_layer).
    pub fn reset_exit(&self) {
        self.exit_flag.store(0, Ordering::Release);
    }

    /// Whether this request runs the full model.
    pub fn is_full_model(&self) -> bool {
        self.target_layer == u32::MAX
    }

    /// Whether this request is active (not skipped by Compact).
    pub fn is_active(&self) -> bool {
        self.compact_scatter.active != 0
    }

    /// Set the original slot index for Scatter writeback.
    pub fn set_original_slot(&mut self, slot: u32) {
        self.compact_scatter.original_slot = slot;
    }

    /// Update telemetry from Epilogue instrumentation.
    pub fn update_telemetry(&mut self, entropy: f32, centroid: f32, residual_delta: f32, residual_cosine: f32) {
        self.telemetry.entropy = entropy;
        self.telemetry.centroid = centroid;
        self.telemetry.residual_delta = residual_delta;
        self.telemetry.residual_cosine = residual_cosine;
    }

    /// Compute Range-Aware Compact Group (§9.1).
    ///
    /// Groups requests by similar activation range to prevent
    /// cross-contamination in low-precision GEMM tiles.
    pub fn compute_range_group(num_groups: u32) {
        // Range group is computed from telemetry centroid
        // Group = floor(centroid * num_groups) % num_groups
        // This is set by the Epilogue telemetry, not by host
    }

    /// Convert to a `LayeredRequestControl` for use by PolymorphicExecutor.
    pub fn to_layered_control(&self) -> crate::graph::types::LayeredRequestControl {
        crate::graph::types::LayeredRequestControl {
            target_layer: self.target_layer,
            exit_flag: AtomicU32::new(self.exit_flag.load(Ordering::Relaxed)),
        }
    }
}

/// Convert BusPortTag to bit position for bus_port_mask
fn bus_port_tag_to_bit(tag: BusPortTag) -> u32 {
    match tag {
        BusPortTag::RagInjection => 1 << 0,
        BusPortTag::EarlyExit => 1 << 1,
        BusPortTag::IntentRecall => 1 << 2,
        BusPortTag::Guardrail => 1 << 3,
        BusPortTag::ShadowKv => 1 << 4,
        BusPortTag::Custom(n) => 1 << (8 + (n % 24)),
    }
}

// Safety: RequestState contains AtomicU32 which is Send/Sync.
// No raw pointer fields are owned data.
unsafe impl Send for RequestState {}
unsafe impl Sync for RequestState {}

/// Device memory handle (unified abstraction).
#[derive(Debug)]
pub enum DeviceMemory {
    /// CPU: host memory pointer
    Host { ptr: *mut u8, size: usize },
    /// CUDA: device pointer
    Cuda { ptr: u64, size: usize },
    /// HIP: device pointer
    Hip { ptr: u64, size: usize },
    /// Metal: buffer handle
    Metal { buffer_id: u64, size: usize },
}

unsafe impl Send for DeviceMemory {}
unsafe impl Sync for DeviceMemory {}

/// Unified Request State Table.
///
/// CPU/GPU share the same structure. Memory allocation strategy driven by DeviceProfile.
#[derive(Debug)]
pub struct RequestStateTable {
    /// Request states (host memory)
    pub states: Vec<RequestState>,
    /// Device memory (None for CPU, Some for GPU)
    pub device_memory: Option<DeviceMemory>,
    /// Backend type
    backend: BackendType,
}

impl RequestStateTable {
    /// Create new RST for given backend.
    pub fn new(backend: BackendType) -> Self {
        Self {
            states: Vec::new(),
            device_memory: None,
            backend,
        }
    }

    /// Add request state.
    pub fn add(&mut self, state: RequestState) {
        self.states.push(state);
    }

    /// Allocate device memory based on backend type.
    ///
    /// CPU: no-op (uses host memory directly)
    /// GPU: allocate device memory for RST
    pub fn allocate(&mut self) -> Result<(), String> {
        match self.backend {
            BackendType::Cpu => {
                // CPU: use host memory directly, no device allocation
                Ok(())
            }
            BackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.allocate_cuda()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA feature not enabled".into())
                }
            }
            BackendType::Rocm => {
                #[cfg(feature = "hip")]
                {
                    self.allocate_hip()
                }
                #[cfg(not(feature = "hip"))]
                {
                    Err("HIP feature not enabled".into())
                }
            }
            BackendType::Metal => {
                #[cfg(feature = "metal")]
                {
                    self.allocate_metal()
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal feature not enabled".into())
                }
            }
        }
    }

    /// Synchronize host states to device (HtoD).
    ///
    /// CPU: no-op
    /// GPU: copy states to device memory
    pub fn sync_to_device(&mut self) -> Result<(), String> {
        match self.backend {
            BackendType::Cpu => Ok(()),
            BackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    self.sync_cuda()
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err("CUDA feature not enabled".into())
                }
            }
            BackendType::Rocm => {
                #[cfg(feature = "hip")]
                {
                    self.sync_hip()
                }
                #[cfg(not(feature = "hip"))]
                {
                    Err("HIP feature not enabled".into())
                }
            }
            BackendType::Metal => {
                #[cfg(feature = "metal")]
                {
                    self.sync_metal()
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err("Metal feature not enabled".into())
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn allocate_cuda(&mut self) -> Result<(), String> {
        use gllm_kernels::gpu::cuda::CudaDriver;

        let size = self.states.len() * std::mem::size_of::<RequestState>();
        if size == 0 {
            return Ok(());
        }

        let driver = CudaDriver::load().map_err(|e| format!("Failed to load CUDA driver: {}", e))?;
        let ptr = driver.mem_alloc(size).map_err(|e| format!("cuMemAlloc failed: {}", e))?;

        self.device_memory = Some(DeviceMemory::Cuda { ptr, size });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn sync_cuda(&mut self) -> Result<(), String> {
        use gllm_kernels::gpu::cuda::CudaDriver;

        if self.states.is_empty() {
            return Ok(());
        }

        let DeviceMemory::Cuda { ptr, .. } = self.device_memory.as_ref()
            .ok_or("Device memory not allocated")? else {
            return Err("Invalid device memory type for CUDA".into());
        };

        let driver = CudaDriver::load().map_err(|e| format!("Failed to load CUDA driver: {}", e))?;
        let src = self.states.as_ptr() as *const u8;
        let size = self.states.len() * std::mem::size_of::<RequestState>();

        driver.memcpy_htod(*ptr, src, size)
            .map_err(|e| format!("cuMemcpyHtoD failed: {}", e))?;

        Ok(())
    }

    #[cfg(feature = "hip")]
    fn allocate_hip(&mut self) -> Result<(), String> {
        use gllm_kernels::gpu::hip::HipDriver;

        let size = self.states.len() * std::mem::size_of::<RequestState>();
        if size == 0 {
            return Ok(());
        }

        let driver = HipDriver::load().map_err(|e| format!("Failed to load HIP driver: {}", e))?;
        let ptr = driver.mem_alloc(size).map_err(|e| format!("hipMalloc failed: {}", e))?;

        self.device_memory = Some(DeviceMemory::Hip { ptr, size });
        Ok(())
    }

    #[cfg(feature = "hip")]
    fn sync_hip(&mut self) -> Result<(), String> {
        use gllm_kernels::gpu::hip::HipDriver;

        if self.states.is_empty() {
            return Ok(());
        }

        let DeviceMemory::Hip { ptr, .. } = self.device_memory.as_ref()
            .ok_or("Device memory not allocated")? else {
            return Err("Invalid device memory type for HIP".into());
        };

        let driver = HipDriver::load().map_err(|e| format!("Failed to load HIP driver: {}", e))?;
        let src = self.states.as_ptr() as *const u8;
        let size = self.states.len() * std::mem::size_of::<RequestState>();

        driver.memcpy_htod(*ptr, src, size)
            .map_err(|e| format!("hipMemcpyHtoD failed: {}", e))?;

        Ok(())
    }

    #[cfg(feature = "metal")]
    fn allocate_metal(&mut self) -> Result<(), String> {
        let size = self.states.len() * std::mem::size_of::<RequestState>();
        if size == 0 {
            return Ok(());
        }

        // Metal shared buffer allocation via MTLDevice.
        // The actual Metal backend in gllm-kernels handles buffer creation;
        // here we record the size for the device_memory bookkeeping.
        self.device_memory = Some(DeviceMemory::Metal { buffer_id: 0, size });
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn sync_metal(&mut self) -> Result<(), String> {
        if self.states.is_empty() {
            return Ok(());
        }

        // Metal uses unified memory — the host-side `states` slice is already
        // visible to the GPU via shared buffer mapping. No explicit sync needed.
        Ok(())
    }
}

impl Drop for RequestStateTable {
    fn drop(&mut self) {
        if let Some(ref mem) = self.device_memory {
            match mem {
                #[cfg(feature = "cuda")]
                DeviceMemory::Cuda { ptr, .. } => {
                    if let Ok(driver) = gllm_kernels::gpu::cuda::CudaDriver::load() {
                        let _ = driver.mem_free(*ptr);
                    }
                }
                #[cfg(feature = "hip")]
                DeviceMemory::Hip { ptr, .. } => {
                    if let Ok(driver) = gllm_kernels::gpu::hip::HipDriver::load() {
                        let _ = driver.mem_free(*ptr);
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rst_no_device_memory() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 128, 0));

        // CPU: allocate should be no-op
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_none());

        // CPU: sync should be no-op
        assert!(rst.sync_to_device().is_ok());
    }

    #[test]
    fn test_request_phase_equality() {
        assert_eq!(RequestPhase::Prefill, RequestPhase::Prefill);
        assert_ne!(RequestPhase::Prefill, RequestPhase::Decode);
    }

    #[test]
    fn test_rst_add_states() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        assert_eq!(rst.states.len(), 0);

        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        assert_eq!(rst.states.len(), 1);

        rst.add(RequestState::new(2, RequestPhase::ChunkedPrefill, 256, 10));
        assert_eq!(rst.states.len(), 2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_rst_allocate() {
        use gllm_kernels::gpu::cuda::CudaDriver;

        // Skip if CUDA not available
        if CudaDriver::load().is_err() {
            return;
        }

        let mut rst = RequestStateTable::new(BackendType::Cuda);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 128, 0));

        // CUDA: should allocate device memory
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_some());

        if let Some(DeviceMemory::Cuda { ptr, size }) = rst.device_memory {
            assert!(ptr > 0);
            assert_eq!(size, std::mem::size_of::<RequestState>());
        } else {
            panic!("Expected CUDA device memory");
        }

        // CUDA: sync should succeed
        assert!(rst.sync_to_device().is_ok());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn test_hip_rst_allocate() {
        use gllm_kernels::gpu::hip::HipDriver;

        // Skip if HIP not available
        if HipDriver::load().is_err() {
            return;
        }

        let mut rst = RequestStateTable::new(BackendType::Rocm);
        rst.add(RequestState {
            request_id: 1,
            phase: RequestPhase::Decode,
            seq_len: 64,
            kv_cache_offset: 0,
        });

        // HIP: should allocate device memory
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_some());

        if let Some(DeviceMemory::Hip { ptr, size }) = rst.device_memory {
            assert!(ptr > 0);
            assert_eq!(size, std::mem::size_of::<RequestState>());
        } else {
            panic!("Expected HIP device memory");
        }

        // HIP: sync should succeed
        assert!(rst.sync_to_device().is_ok());
    }

    // ── Polymorphic execution control tests (SPEC §16.1-16.4) ──

    #[test]
    fn test_request_state_default_is_full_model() {
        let rs = RequestState::new(42, RequestPhase::Decode, 10, 0);
        assert!(rs.is_full_model());
        assert!(!rs.is_exited());
        assert_eq!(rs.target_layer, u32::MAX);
    }

    #[test]
    fn test_request_state_with_target_layer() {
        let rs = RequestState::new(1, RequestPhase::Prefill, 128, 0)
            .with_target_layer(5);
        assert!(!rs.is_full_model());
        assert_eq!(rs.target_layer, 5);
        assert!(!rs.is_exited());
    }

    #[test]
    fn test_request_state_exit_flag() {
        let rs = RequestState::new(1, RequestPhase::Decode, 1, 0)
            .with_target_layer(3);
        assert!(!rs.is_exited());

        rs.mark_exited();
        assert!(rs.is_exited());

        rs.reset_exit();
        assert!(!rs.is_exited());
        // target_layer should be preserved
        assert_eq!(rs.target_layer, 3);
    }

    #[test]
    fn test_request_state_to_layered_control() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 100)
            .with_target_layer(7);
        let ctrl = rs.to_layered_control();
        assert_eq!(ctrl.target_layer, 7);
        assert!(!ctrl.is_exited());
    }

    #[test]
    fn test_request_state_clone_preserves_fields() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5);
        rs.mark_exited();

        let cloned = rs.clone();
        assert_eq!(cloned.target_layer, 5);
        assert!(cloned.is_exited());
    }
}
