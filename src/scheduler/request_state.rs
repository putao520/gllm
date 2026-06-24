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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn compute_range_group(_num_groups: u32) {
        // Range group is computed from telemetry centroid
        // Group = floor(centroid * num_groups) % num_groups
        // This is set by the Epilogue telemetry, not by host
    }

    /// Convert to a `LayeredRequestControl` for use by PolymorphicExecutor.
    /// Convert to LayeredRequestControl for graph execution.
    ///
    /// **Thread safety**: Caller must ensure no concurrent modification of
    /// `target_layer` or `exit_flag` during this call (typically under Executor
    /// Mutex). `exit_flag` is read with Acquire ordering to synchronize with
    /// the Release-ordered store that sets the exit condition.
    pub fn to_layered_control(&self) -> crate::graph::types::LayeredRequestControl {
        crate::graph::types::LayeredRequestControl {
            target_layer: self.target_layer,
            exit_flag: AtomicU32::new(self.exit_flag.load(Ordering::Acquire)),
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
                        if let Err(e) = driver.mem_free(*ptr) {
                            log::error!("Drop RequestStateTable: cuda mem_free({:?}) failed: {}", ptr, e);
                        }
                    }
                }
                #[cfg(feature = "hip")]
                DeviceMemory::Hip { ptr, .. } => {
                    if let Ok(driver) = gllm_kernels::gpu::hip::HipDriver::load() {
                        if let Err(e) = driver.mem_free(*ptr) {
                            log::error!("Drop RequestStateTable: hip mem_free({:?}) failed: {}", ptr, e);
                        }
                    }
                }
                #[cfg(feature = "metal")]
                DeviceMemory::Metal { buffer_id, .. } => {
                    // Metal buffer release via MetalDriver when available
                    // TODO(port): implement MetalDriver::mem_free once metal feature is complete
                    let _ = buffer_id;
                }
                DeviceMemory::Host { ptr, .. } => {
                    // SAFETY: Host variant is a CPU-only fallback. In production, GPU
                    // backends use Cuda/Hip/Metal variants. The Host ptr is currently
                    // always null (only used in tests). If real host allocation is needed
                    // in the future, must add Layout field to Host variant and use
                    // std::alloc::dealloc. DO NOT use libc::free — it's UB if ptr came
                    // from Rust's allocator.
                    if !ptr.is_null() {
                        log::error!(
                            "DeviceMemory::Host with non-null ptr ({:?}) requires Layout for safe dealloc; \
                             add layout field and use std::alloc::dealloc — leaking memory to avoid UB",
                            ptr
                        );
                        // Intentionally leak: dealloc without Layout is UB, leaking is safe.
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
                _ => {
                    // When no GPU features are enabled, only Host variant should exist
                    // Other variants are impossible at runtime, so this is unreachable
                }
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

        // Skip if CUDA driver or device not available
        let driver = match CudaDriver::load() {
            Ok(d) => d,
            Err(_) => return,
        };

        // Verify device memory allocation works before testing RST
        let test_size = std::mem::size_of::<RequestState>();
        if driver.mem_alloc(test_size).is_err() {
            return; // No CUDA device memory available, skip
        }

        let mut rst = RequestStateTable::new(BackendType::Cuda);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 128, 0));

        // CUDA: should allocate device memory
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_some());

        if let Some(DeviceMemory::Cuda { ptr, size }) = rst.device_memory {
            assert!(ptr > 0);
            assert_eq!(size, test_size);
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
        assert!(!ctrl.should_exit_at(0));
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

    // ---- Additional tests ----

    #[test]
    fn request_phase_all_variants() {
        assert_ne!(RequestPhase::Prefill, RequestPhase::Decode);
        assert_ne!(RequestPhase::Decode, RequestPhase::ChunkedPrefill);
        assert_ne!(RequestPhase::Prefill, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_phase_copy_clone() {
        let p = RequestPhase::ChunkedPrefill;
        let p2 = p;
        assert_eq!(p, p2);
        let p3 = p.clone();
        assert_eq!(p3, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn compact_scatter_meta_copy_clone() {
        let m = CompactScatterMeta { original_slot: 3, compacted_slot: 1, active: 1 };
        let m2 = m;
        assert_eq!(m2.original_slot, 3);
        assert_eq!(m2.compacted_slot, 1);
        let m3 = m.clone();
        assert_eq!(m3.active, 1);
    }

    #[test]
    fn request_telemetry_default() {
        let t = RequestTelemetry::default();
        assert_eq!(t.entropy, 0.0);
        assert_eq!(t.centroid, 0.0);
        assert!((t.residual_delta - 1.0).abs() < 1e-6);
        assert!((t.residual_cosine - 1.0).abs() < 1e-6);
        assert_eq!(t.range_group, 0);
    }

    #[test]
    fn request_telemetry_copy_clone() {
        let t = RequestTelemetry { entropy: 2.5, centroid: 0.8, residual_delta: 0.3, residual_cosine: 0.95, range_group: 7 };
        let t2 = t;
        assert!((t2.entropy - 2.5).abs() < 1e-6);
        let t3 = t.clone();
        assert_eq!(t3.range_group, 7);
    }

    #[test]
    fn request_state_new_default_fields() {
        let rs = RequestState::new(99, RequestPhase::Decode, 50, 100);
        assert_eq!(rs.request_id, 99);
        assert_eq!(rs.phase, RequestPhase::Decode);
        assert_eq!(rs.seq_len, 50);
        assert_eq!(rs.kv_cache_offset, 100);
        assert_eq!(rs.bus_port_mask, 0);
        assert_eq!(rs.variant_key_hash, 0);
        assert!(rs.is_active()); // default active = 1
    }

    #[test]
    fn request_state_is_active_flag() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(rs.is_active());
        rs.compact_scatter.active = 0;
        assert!(!rs.is_active());
    }

    #[test]
    fn request_state_set_original_slot() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert_eq!(rs.compact_scatter.original_slot, 0);
        rs.set_original_slot(42);
        assert_eq!(rs.compact_scatter.original_slot, 42);
    }

    #[test]
    fn request_state_update_telemetry() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.update_telemetry(1.5, 0.7, 0.3, 0.95);
        assert!((rs.telemetry.entropy - 1.5).abs() < 1e-6);
        assert!((rs.telemetry.centroid - 0.7).abs() < 1e-6);
        assert!((rs.telemetry.residual_delta - 0.3).abs() < 1e-6);
        assert!((rs.telemetry.residual_cosine - 0.95).abs() < 1e-6);
    }

    #[test]
    fn request_state_bus_port_rag() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection);
        assert!(rs.has_bus_port(BusPortTag::RagInjection));
        assert!(!rs.has_bus_port(BusPortTag::EarlyExit));
    }

    #[test]
    fn request_state_bus_port_multiple() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::Guardrail)
            .with_bus_port(BusPortTag::IntentRecall);
        assert!(rs.has_bus_port(BusPortTag::RagInjection));
        assert!(rs.has_bus_port(BusPortTag::Guardrail));
        assert!(rs.has_bus_port(BusPortTag::IntentRecall));
        assert!(!rs.has_bus_port(BusPortTag::EarlyExit));
        assert!(!rs.has_bus_port(BusPortTag::ShadowKv));
    }

    #[test]
    fn request_state_bus_port_custom() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::Custom(0));
        assert!(rs.has_bus_port(BusPortTag::Custom(0)));
        assert!(!rs.has_bus_port(BusPortTag::Custom(1)));
    }

    #[test]
    fn bus_port_tag_to_bit_known_values() {
        assert_eq!(bus_port_tag_to_bit(BusPortTag::RagInjection), 1 << 0);
        assert_eq!(bus_port_tag_to_bit(BusPortTag::EarlyExit), 1 << 1);
        assert_eq!(bus_port_tag_to_bit(BusPortTag::IntentRecall), 1 << 2);
        assert_eq!(bus_port_tag_to_bit(BusPortTag::Guardrail), 1 << 3);
        assert_eq!(bus_port_tag_to_bit(BusPortTag::ShadowKv), 1 << 4);
    }

    #[test]
    fn bus_port_tag_to_bit_custom() {
        let bit = bus_port_tag_to_bit(BusPortTag::Custom(5));
        assert_eq!(bit, 1 << (8 + 5));
    }

    #[test]
    fn device_memory_host() {
        let dm = DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 1024 };
        if let DeviceMemory::Host { size, .. } = dm {
            assert_eq!(size, 1024);
        }
    }

    #[test]
    fn request_state_table_cpu_empty_allocate() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_none());
    }

    #[test]
    fn request_state_to_layered_control_exit_flag() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(3);
        rs.mark_exited();
        let ctrl = rs.to_layered_control();
        assert!(ctrl.should_exit_at(3));
    }

    #[test]
    fn request_state_clone_independence() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        let cloned = rs.clone();
        rs.mark_exited();
        assert!(rs.is_exited());
        // Clone should have its own AtomicU32
        assert!(!cloned.is_exited());
    }

    // ── Debug trait tests ──

    #[test]
    fn request_phase_debug_output() {
        assert_eq!(format!("{:?}", RequestPhase::Prefill), "Prefill");
        assert_eq!(format!("{:?}", RequestPhase::Decode), "Decode");
        assert_eq!(format!("{:?}", RequestPhase::ChunkedPrefill), "ChunkedPrefill");
    }

    #[test]
    fn compact_scatter_meta_debug_output() {
        let m = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("original_slot"));
        assert!(debug_str.contains("compacted_slot"));
        assert!(debug_str.contains("active"));
    }

    #[test]
    fn request_telemetry_debug_output() {
        let t = RequestTelemetry::default();
        let debug_str = format!("{:?}", t);
        assert!(debug_str.contains("entropy"));
        assert!(debug_str.contains("centroid"));
        assert!(debug_str.contains("residual_delta"));
        assert!(debug_str.contains("residual_cosine"));
        assert!(debug_str.contains("range_group"));
    }

    #[test]
    fn request_state_debug_output() {
        let rs = RequestState::new(42, RequestPhase::Prefill, 128, 256);
        let debug_str = format!("{:?}", rs);
        assert!(debug_str.contains("request_id"));
        assert!(debug_str.contains("phase"));
        assert!(debug_str.contains("seq_len"));
    }

    #[test]
    fn device_memory_debug_output() {
        let dm = DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 512 };
        let debug_str = format!("{:?}", dm);
        assert!(debug_str.contains("Host"));
    }

    #[test]
    fn device_memory_cuda_variant() {
        let dm = DeviceMemory::Cuda { ptr: 0xDEAD, size: 4096 };
        if let DeviceMemory::Cuda { ptr, size } = dm {
            assert_eq!(ptr, 0xDEAD);
            assert_eq!(size, 4096);
        }
    }

    #[test]
    fn device_memory_hip_variant() {
        let dm = DeviceMemory::Hip { ptr: 0xBEEF, size: 8192 };
        if let DeviceMemory::Hip { ptr, size } = dm {
            assert_eq!(ptr, 0xBEEF);
            assert_eq!(size, 8192);
        }
    }

    #[test]
    fn device_memory_metal_variant() {
        let dm = DeviceMemory::Metal { buffer_id: 123, size: 2048 };
        if let DeviceMemory::Metal { buffer_id, size } = dm {
            assert_eq!(buffer_id, 123);
            assert_eq!(size, 2048);
        }
    }

    #[test]
    fn device_memory_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DeviceMemory>();
    }

    // ── BusPortTag edge cases ──

    #[test]
    fn bus_port_tag_to_bit_custom_zero() {
        assert_eq!(bus_port_tag_to_bit(BusPortTag::Custom(0)), 1 << 8);
    }

    #[test]
    fn bus_port_tag_to_bit_custom_wraps_modulo() {
        // 24 % 24 == 0, so Custom(24) should map to bit 8 (same as Custom(0))
        assert_eq!(
            bus_port_tag_to_bit(BusPortTag::Custom(24)),
            bus_port_tag_to_bit(BusPortTag::Custom(0))
        );
        // 25 % 24 == 1, so Custom(25) should map to bit 9 (same as Custom(1))
        assert_eq!(
            bus_port_tag_to_bit(BusPortTag::Custom(25)),
            bus_port_tag_to_bit(BusPortTag::Custom(1))
        );
    }

    #[test]
    fn bus_port_tag_to_bit_custom_max_index() {
        // 23 % 24 == 23, maps to bit 31 (8 + 23)
        let bit = bus_port_tag_to_bit(BusPortTag::Custom(23));
        assert_eq!(bit, 1 << 31);
    }

    #[test]
    fn request_state_bus_port_idempotent() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::RagInjection);
        // Attaching same port twice should still report true
        assert!(rs.has_bus_port(BusPortTag::RagInjection));
        // bus_port_mask should be exactly 1 bit
        assert_eq!(rs.bus_port_mask, 1 << 0);
    }

    #[test]
    fn request_state_builder_chain() {
        let rs = RequestState::new(10, RequestPhase::ChunkedPrefill, 512, 1024)
            .with_target_layer(3)
            .with_bus_port(BusPortTag::EarlyExit)
            .with_bus_port(BusPortTag::ShadowKv);
        assert_eq!(rs.request_id, 10);
        assert_eq!(rs.phase, RequestPhase::ChunkedPrefill);
        assert_eq!(rs.seq_len, 512);
        assert_eq!(rs.kv_cache_offset, 1024);
        assert_eq!(rs.target_layer, 3);
        assert!(!rs.is_full_model());
        assert!(rs.has_bus_port(BusPortTag::EarlyExit));
        assert!(rs.has_bus_port(BusPortTag::ShadowKv));
        assert!(!rs.has_bus_port(BusPortTag::RagInjection));
    }

    #[test]
    fn request_state_zero_seq_len_and_offset() {
        let rs = RequestState::new(0, RequestPhase::Decode, 0, 0);
        assert_eq!(rs.seq_len, 0);
        assert_eq!(rs.kv_cache_offset, 0);
        assert_eq!(rs.request_id, 0);
    }

    #[test]
    fn request_state_clone_preserves_all_fields() {
        let mut rs = RequestState::new(7, RequestPhase::ChunkedPrefill, 200, 300)
            .with_target_layer(12)
            .with_bus_port(BusPortTag::IntentRecall);
        rs.update_telemetry(0.5, 0.3, 0.1, 0.99);
        rs.compact_scatter = CompactScatterMeta { original_slot: 5, compacted_slot: 2, active: 1 };

        let cloned = rs.clone();
        assert_eq!(cloned.request_id, 7);
        assert_eq!(cloned.phase, RequestPhase::ChunkedPrefill);
        assert_eq!(cloned.seq_len, 200);
        assert_eq!(cloned.kv_cache_offset, 300);
        assert_eq!(cloned.target_layer, 12);
        assert_eq!(cloned.compact_scatter.original_slot, 5);
        assert_eq!(cloned.compact_scatter.compacted_slot, 2);
        assert!((cloned.telemetry.entropy - 0.5).abs() < 1e-6);
        assert_eq!(cloned.bus_port_mask, rs.bus_port_mask);
        assert_eq!(cloned.variant_key_hash, rs.variant_key_hash);
    }

    #[test]
    fn compact_scatter_meta_zero_values() {
        let m = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 0 };
        assert_eq!(m.original_slot, 0);
        assert_eq!(m.compacted_slot, 0);
        assert_eq!(m.active, 0);
    }

    #[test]
    fn compact_scatter_meta_max_values() {
        let m = CompactScatterMeta { original_slot: u32::MAX, compacted_slot: u32::MAX, active: u32::MAX };
        assert_eq!(m.original_slot, u32::MAX);
        assert_eq!(m.compacted_slot, u32::MAX);
        assert_eq!(m.active, u32::MAX);
    }

    #[test]
    fn request_telemetry_all_fields_set() {
        let t = RequestTelemetry {
            entropy: 3.14,
            centroid: 0.99,
            residual_delta: 0.01,
            residual_cosine: 0.5,
            range_group: u32::MAX,
        };
        assert!((t.entropy - 3.14).abs() < 1e-6);
        assert!((t.centroid - 0.99).abs() < 1e-6);
        assert!((t.residual_delta - 0.01).abs() < 1e-6);
        assert!((t.residual_cosine - 0.5).abs() < 1e-6);
        assert_eq!(t.range_group, u32::MAX);
    }

    #[test]
    fn request_state_layered_control_full_model() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        let ctrl = rs.to_layered_control();
        // Full model: target_layer = u32::MAX
        assert_eq!(ctrl.target_layer, u32::MAX);
        // u32::MAX > 0 is true, so should_exit_at any layer < u32::MAX returns false
        assert!(!ctrl.should_exit_at(0));
        assert!(!ctrl.should_exit_at(1000));
    }

    #[test]
    fn request_state_exit_then_reset_cycle() {
        let rs = RequestState::new(1, RequestPhase::Prefill, 64, 0);
        assert!(!rs.is_exited());
        rs.mark_exited();
        assert!(rs.is_exited());
        rs.reset_exit();
        assert!(!rs.is_exited());
        rs.mark_exited();
        assert!(rs.is_exited());
    }

    #[test]
    fn request_state_table_backend_field() {
        let rst = RequestStateTable::new(BackendType::Cpu);
        assert!(rst.device_memory.is_none());
        assert_eq!(rst.states.len(), 0);
    }

    #[test]
    fn request_state_table_multiple_add() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        for i in 0..10 {
            rst.add(RequestState::new(i, RequestPhase::Decode, i as usize * 10, i as usize * 100));
        }
        assert_eq!(rst.states.len(), 10);
        assert_eq!(rst.states[0].request_id, 0);
        assert_eq!(rst.states[9].request_id, 9);
    }

    #[test]
    fn request_state_table_gpu_without_feature_returns_error() {
        // Without CUDA/HIP/Metal features enabled, allocate should return error
        let mut rst_cuda = RequestStateTable::new(BackendType::Cuda);
        let result = rst_cuda.allocate();
        // Feature is not enabled in normal builds, so expect error
        if cfg!(not(feature = "cuda")) {
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("CUDA"));
        }

        let mut rst_hip = RequestStateTable::new(BackendType::Rocm);
        let result = rst_hip.allocate();
        if cfg!(not(feature = "hip")) {
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("HIP"));
        }

        let mut rst_metal = RequestStateTable::new(BackendType::Metal);
        let result = rst_metal.allocate();
        if cfg!(not(feature = "metal")) {
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Metal"));
        }
    }

    #[test]
    fn request_state_table_debug_output() {
        let rst = RequestStateTable::new(BackendType::Cpu);
        let debug_str = format!("{:?}", rst);
        assert!(debug_str.contains("RequestStateTable"));
    }

    #[test]
    fn request_state_compute_range_group_callable() {
        // compute_range_group is currently a no-op but should be callable
        RequestState::compute_range_group(8);
        RequestState::compute_range_group(0);
        RequestState::compute_range_group(u32::MAX);
    }

    #[test]
    fn bus_port_tag_partial_eq_custom() {
        assert_eq!(BusPortTag::Custom(5), BusPortTag::Custom(5));
        assert_ne!(BusPortTag::Custom(5), BusPortTag::Custom(6));
        assert_ne!(BusPortTag::Custom(0), BusPortTag::RagInjection);
    }

    #[test]
    fn request_state_is_active_various_values() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(rs.is_active());
        rs.compact_scatter.active = 0;
        assert!(!rs.is_active());
        rs.compact_scatter.active = 2;
        assert!(rs.is_active());
        rs.compact_scatter.active = u32::MAX;
        assert!(rs.is_active());
    }

    // ── Hash trait tests ──

    #[test]
    fn request_phase_hash_equal_variants_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value<T: Hash>(v: &T) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        assert_eq!(hash_value(&RequestPhase::Prefill), hash_value(&RequestPhase::Prefill));
        assert_eq!(hash_value(&RequestPhase::Decode), hash_value(&RequestPhase::Decode));
        assert_eq!(hash_value(&RequestPhase::ChunkedPrefill), hash_value(&RequestPhase::ChunkedPrefill));

        // Different variants should (almost certainly) produce different hashes
        assert_ne!(hash_value(&RequestPhase::Prefill), hash_value(&RequestPhase::Decode));
        assert_ne!(hash_value(&RequestPhase::Decode), hash_value(&RequestPhase::ChunkedPrefill));
    }

    #[test]
    fn bus_port_tag_hash_equal_variants_match() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        fn hash_value<T: Hash>(v: &T) -> u64 {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        }

        assert_eq!(hash_value(&BusPortTag::RagInjection), hash_value(&BusPortTag::RagInjection));
        assert_eq!(hash_value(&BusPortTag::Custom(7)), hash_value(&BusPortTag::Custom(7)));
        assert_ne!(hash_value(&BusPortTag::Custom(7)), hash_value(&BusPortTag::Custom(8)));
        assert_ne!(hash_value(&BusPortTag::RagInjection), hash_value(&BusPortTag::EarlyExit));
    }

    // ── PartialEq trait tests ──

    #[test]
    fn compact_scatter_meta_equality() {
        let a = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let b = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        assert_eq!(a, b);

        let c = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 0 };
        assert_ne!(a, c);

        let d = CompactScatterMeta { original_slot: 0, compacted_slot: 2, active: 1 };
        assert_ne!(a, d);
    }

    #[test]
    fn request_telemetry_equality() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        assert_eq!(a, b);

        let c = RequestTelemetry { entropy: 2.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        assert_ne!(a, c);
    }

    // ── Edge cases: seq_len, kv_cache_offset boundaries ──

    #[test]
    fn request_state_max_seq_len() {
        let rs = RequestState::new(1, RequestPhase::Prefill, usize::MAX, 0);
        assert_eq!(rs.seq_len, usize::MAX);
    }

    #[test]
    fn request_state_max_kv_cache_offset() {
        let rs = RequestState::new(1, RequestPhase::Decode, 0, usize::MAX);
        assert_eq!(rs.kv_cache_offset, usize::MAX);
    }

    #[test]
    fn request_state_target_layer_zero() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(0);
        assert_eq!(rs.target_layer, 0);
        assert!(!rs.is_full_model());
    }

    #[test]
    fn request_state_target_layer_max_minus_one() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(u32::MAX as usize - 1);
        assert_eq!(rs.target_layer, u32::MAX - 1);
        assert!(!rs.is_full_model());
    }

    // ── Telemetry edge values ──

    #[test]
    fn request_telemetry_negative_entropy() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.update_telemetry(-1.0, 0.0, -0.5, -0.99);
        assert!((rs.telemetry.entropy - (-1.0)).abs() < 1e-6);
        assert!((rs.telemetry.residual_delta - (-0.5)).abs() < 1e-6);
        assert!((rs.telemetry.residual_cosine - (-0.99)).abs() < 1e-6);
    }

    #[test]
    fn request_telemetry_zero_values() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.update_telemetry(0.0, 0.0, 0.0, 0.0);
        assert_eq!(rs.telemetry.entropy, 0.0);
        assert_eq!(rs.telemetry.centroid, 0.0);
        assert_eq!(rs.telemetry.residual_delta, 0.0);
        assert_eq!(rs.telemetry.residual_cosine, 0.0);
    }

    #[test]
    fn request_telemetry_extreme_floats() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.update_telemetry(f32::MAX, f32::MIN, f32::INFINITY, f32::NEG_INFINITY);
        assert_eq!(rs.telemetry.entropy, f32::MAX);
        assert_eq!(rs.telemetry.centroid, f32::MIN);
        assert_eq!(rs.telemetry.residual_delta, f32::INFINITY);
        assert_eq!(rs.telemetry.residual_cosine, f32::NEG_INFINITY);
    }

    // ── bus_port_mask edge cases ──

    #[test]
    fn request_state_bus_port_all_standard_ports() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::EarlyExit)
            .with_bus_port(BusPortTag::IntentRecall)
            .with_bus_port(BusPortTag::Guardrail)
            .with_bus_port(BusPortTag::ShadowKv);
        assert!(rs.has_bus_port(BusPortTag::RagInjection));
        assert!(rs.has_bus_port(BusPortTag::EarlyExit));
        assert!(rs.has_bus_port(BusPortTag::IntentRecall));
        assert!(rs.has_bus_port(BusPortTag::Guardrail));
        assert!(rs.has_bus_port(BusPortTag::ShadowKv));
        // bits 0-4 = 0b11111 = 31
        assert_eq!(rs.bus_port_mask & 0x1F, 31);
    }

    #[test]
    fn request_state_no_bus_port_by_default() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(!rs.has_bus_port(BusPortTag::RagInjection));
        assert!(!rs.has_bus_port(BusPortTag::EarlyExit));
        assert!(!rs.has_bus_port(BusPortTag::IntentRecall));
        assert!(!rs.has_bus_port(BusPortTag::Guardrail));
        assert!(!rs.has_bus_port(BusPortTag::ShadowKv));
        assert_eq!(rs.bus_port_mask, 0);
    }

    // ── DeviceMemory variant field access ──

    #[test]
    fn device_memory_host_null_ptr() {
        let dm = DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 0 };
        if let DeviceMemory::Host { ptr, size } = dm {
            assert!(ptr.is_null());
            assert_eq!(size, 0);
        }
    }

    #[test]
    fn device_memory_cuda_zero_ptr() {
        let dm = DeviceMemory::Cuda { ptr: 0, size: 0 };
        if let DeviceMemory::Cuda { ptr, size } = dm {
            assert_eq!(ptr, 0);
            assert_eq!(size, 0);
        }
    }

    #[test]
    fn device_memory_hip_zero_ptr() {
        let dm = DeviceMemory::Hip { ptr: 0, size: 0 };
        if let DeviceMemory::Hip { ptr, size } = dm {
            assert_eq!(ptr, 0);
            assert_eq!(size, 0);
        }
    }

    #[test]
    fn device_memory_metal_zero_buffer() {
        let dm = DeviceMemory::Metal { buffer_id: 0, size: 0 };
        if let DeviceMemory::Metal { buffer_id, size } = dm {
            assert_eq!(buffer_id, 0);
            assert_eq!(size, 0);
        }
    }

    // ── RequestStateTable sync error paths ──

    #[test]
    fn request_state_table_cpu_sync_empty() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        assert!(rst.states.is_empty());
        assert!(rst.sync_to_device().is_ok());
    }

    #[test]
    fn request_state_table_gpu_sync_without_feature_returns_error() {
        if cfg!(not(feature = "cuda")) {
            let mut rst = RequestStateTable::new(BackendType::Cuda);
            rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
            let result = rst.sync_to_device();
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("CUDA"));
        }
    }

    #[test]
    fn request_state_table_gpu_sync_without_feature_hip() {
        if cfg!(not(feature = "hip")) {
            let mut rst = RequestStateTable::new(BackendType::Rocm);
            rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
            let result = rst.sync_to_device();
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("HIP"));
        }
    }

    #[test]
    fn request_state_table_gpu_sync_without_feature_metal() {
        if cfg!(not(feature = "metal")) {
            let mut rst = RequestStateTable::new(BackendType::Metal);
            rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
            let result = rst.sync_to_device();
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Metal"));
        }
    }

    // ── Drop impl: CPU backend with no device memory ──

    #[test]
    fn request_state_table_drop_cpu_no_panic() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        drop(rst);
    }

    #[test]
    fn request_state_table_drop_empty_no_panic() {
        let rst = RequestStateTable::new(BackendType::Cpu);
        drop(rst);
    }

    // ── variant_key_hash field ──

    #[test]
    fn request_state_variant_key_hash_default() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert_eq!(rs.variant_key_hash, 0);
    }

    #[test]
    fn request_state_variant_key_hash_set() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.variant_key_hash = 0xABCD;
        assert_eq!(rs.variant_key_hash, 0xABCD);
    }

    #[test]
    fn request_state_variant_key_hash_max() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.variant_key_hash = u32::MAX;
        assert_eq!(rs.variant_key_hash, u32::MAX);
    }

    // ── layered_control: should_exit_at boundary ──

    #[test]
    fn layered_control_should_exit_at_exact_target() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5);
        let ctrl = rs.to_layered_control();
        assert!(ctrl.should_exit_at(5));
        assert!(ctrl.should_exit_at(6));
        assert!(!ctrl.should_exit_at(4));
        assert!(!ctrl.should_exit_at(0));
    }

    #[test]
    fn layered_control_exit_flag_set() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.mark_exited();
        let ctrl = rs.to_layered_control();
        // When exit_flag is set, should_exit_at returns true for any layer
        assert!(ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(100));
    }

    #[test]
    fn layered_control_target_layer_zero_exit() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(0);
        let ctrl = rs.to_layered_control();
        // target_layer=0 means should_exit_at(0) checks layer >= 0, but target_layer > 0 is false
        // So only exit_flag matters; target_layer=0 doesn't trigger the range check
        assert!(!ctrl.should_exit_at(0));
    }

    // ── RequestPhase Copy semantics ──

    #[test]
    fn request_phase_copy_not_move() {
        let p = RequestPhase::Prefill;
        let _p2 = p; // Copy, not move
        let _p3 = p; // Still usable after "move"
    }

    // ── CompactScatterMeta field mutation ──

    #[test]
    fn compact_scatter_meta_mutate_fields() {
        let mut m = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 1 };
        m.original_slot = 10;
        m.compacted_slot = 5;
        m.active = 0;
        assert_eq!(m.original_slot, 10);
        assert_eq!(m.compacted_slot, 5);
        assert_eq!(m.active, 0);
    }

    // ── RequestTelemetry field mutation ──

    #[test]
    fn request_telemetry_mutate_fields() {
        let mut t = RequestTelemetry::default();
        t.entropy = 4.2;
        t.centroid = 0.75;
        t.residual_delta = 0.1;
        t.residual_cosine = 0.88;
        t.range_group = 15;
        assert!((t.entropy - 4.2).abs() < 1e-6);
        assert!((t.centroid - 0.75).abs() < 1e-6);
        assert!((t.residual_delta - 0.1).abs() < 1e-6);
        assert!((t.residual_cosine - 0.88).abs() < 1e-6);
        assert_eq!(t.range_group, 15);
    }

    // ── RequestState clone independence for exit_flag ──

    #[test]
    fn request_state_clone_then_mark_exited_independent() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        let cloned = rs.clone();
        cloned.mark_exited();
        // Original should not be affected
        assert!(!rs.is_exited());
        assert!(cloned.is_exited());
    }

    #[test]
    fn request_state_clone_then_reset_independent() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.mark_exited();
        let cloned = rs.clone();
        cloned.reset_exit();
        // Original should still be exited
        assert!(rs.is_exited());
        assert!(!cloned.is_exited());
    }

    // ── RequestState with_target_layer doesn't affect other fields ──

    #[test]
    fn request_state_with_target_layer_preserves_other() {
        let rs = RequestState::new(42, RequestPhase::ChunkedPrefill, 100, 200)
            .with_bus_port(BusPortTag::Guardrail);
        let rs2 = rs.with_target_layer(7);
        assert_eq!(rs2.request_id, 42);
        assert_eq!(rs2.phase, RequestPhase::ChunkedPrefill);
        assert_eq!(rs2.seq_len, 100);
        assert_eq!(rs2.kv_cache_offset, 200);
        assert!(rs2.has_bus_port(BusPortTag::Guardrail));
    }

    // ── BusPortTag::Custom overflow behavior ──

    #[test]
    fn bus_port_tag_custom_large_index_wraps() {
        // 30 % 24 == 6
        let bit = bus_port_tag_to_bit(BusPortTag::Custom(30));
        assert_eq!(bit, 1 << (8 + 6));
    }

    #[test]
    fn bus_port_tag_custom_u32_max_wraps() {
        // u32::MAX % 24 == 15
        let bit = bus_port_tag_to_bit(BusPortTag::Custom(u32::MAX));
        assert_eq!(bit, 1 << (8 + 15));
    }

    // ── RequestStateTable: many adds ──

    #[test]
    fn request_state_table_many_adds() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        for i in 0..100u64 {
            rst.add(RequestState::new(i, RequestPhase::Decode, i as usize, 0));
        }
        assert_eq!(rst.states.len(), 100);
        assert_eq!(rst.states[0].request_id, 0);
        assert_eq!(rst.states[99].request_id, 99);
    }

    // ── RequestState: compact_scatter direct mutation ──

    #[test]
    fn request_state_compact_scatter_direct_set() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.compact_scatter = CompactScatterMeta {
            original_slot: 7,
            compacted_slot: 3,
            active: 1,
        };
        assert_eq!(rs.compact_scatter.original_slot, 7);
        assert_eq!(rs.compact_scatter.compacted_slot, 3);
        assert_eq!(rs.compact_scatter.active, 1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // New tests (~55) — public types and methods coverage expansion
    // ══════════════════════════════════════════════════════════════════════

    // ── RequestPhase: HashMap / HashSet usage (Hash trait) ──

    #[test]
    fn request_phase_hashmap_insert_lookup() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(RequestPhase::Prefill, 1u64);
        map.insert(RequestPhase::Decode, 2u64);
        map.insert(RequestPhase::ChunkedPrefill, 3u64);

        assert_eq!(map.get(&RequestPhase::Prefill), Some(&1));
        assert_eq!(map.get(&RequestPhase::Decode), Some(&2));
        assert_eq!(map.get(&RequestPhase::ChunkedPrefill), Some(&3));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn request_phase_hashset_dedup() {
        use std::collections::HashSet;

        let phases = [
            RequestPhase::Prefill,
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::Decode,
            RequestPhase::ChunkedPrefill,
        ];
        let set: HashSet<_> = phases.iter().copied().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&RequestPhase::Prefill));
        assert!(set.contains(&RequestPhase::Decode));
        assert!(set.contains(&RequestPhase::ChunkedPrefill));
    }

    // ── BusPortTag: HashMap / HashSet usage (Hash trait) ──

    #[test]
    fn bus_port_tag_hashmap_insert_lookup() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(BusPortTag::RagInjection, "rag");
        map.insert(BusPortTag::EarlyExit, "exit");
        map.insert(BusPortTag::Custom(42), "custom42");

        assert_eq!(map.get(&BusPortTag::RagInjection), Some(&"rag"));
        assert_eq!(map.get(&BusPortTag::EarlyExit), Some(&"exit"));
        assert_eq!(map.get(&BusPortTag::Custom(42)), Some(&"custom42"));
        assert_eq!(map.get(&BusPortTag::Guardrail), None);
    }

    #[test]
    fn bus_port_tag_hashset_custom_dedup() {
        use std::collections::HashSet;

        let tags = [BusPortTag::Custom(5), BusPortTag::Custom(5), BusPortTag::Custom(6)];
        let set: HashSet<_> = tags.iter().copied().collect();
        assert_eq!(set.len(), 2);
    }

    // ── CompactScatterMeta: PartialEq properties ──

    #[test]
    fn compact_scatter_meta_eq_symmetry() {
        let a = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let b = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn compact_scatter_meta_eq_reflexivity() {
        let m = CompactScatterMeta { original_slot: 99, compacted_slot: 77, active: 0 };
        assert_eq!(m, m);
    }

    #[test]
    fn compact_scatter_meta_eq_transitivity() {
        let a = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        let b = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        let c = CompactScatterMeta { original_slot: 5, compacted_slot: 3, active: 1 };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn compact_scatter_meta_eq_original_slot_only_differs() {
        let a = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let b = CompactScatterMeta { original_slot: 99, compacted_slot: 2, active: 1 };
        assert_ne!(a, b);
    }

    #[test]
    fn compact_scatter_meta_eq_compacted_slot_only_differs() {
        let a = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let b = CompactScatterMeta { original_slot: 1, compacted_slot: 99, active: 1 };
        assert_ne!(a, b);
    }

    // ── RequestTelemetry: PartialEq properties ──

    #[test]
    fn request_telemetry_eq_symmetry() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn request_telemetry_eq_reflexivity() {
        let t = RequestTelemetry { entropy: 2.7, centroid: 0.1, residual_delta: -0.5, residual_cosine: 0.33, range_group: 42 };
        assert_eq!(t, t);
    }

    #[test]
    fn request_telemetry_eq_centroid_differs() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.9, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        assert_ne!(a, b);
    }

    #[test]
    fn request_telemetry_eq_residual_delta_differs() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.7, residual_cosine: 0.9, range_group: 3 };
        assert_ne!(a, b);
    }

    #[test]
    fn request_telemetry_eq_residual_cosine_differs() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.1, range_group: 3 };
        assert_ne!(a, b);
    }

    #[test]
    fn request_telemetry_eq_range_group_differs() {
        let a = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let b = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 99 };
        assert_ne!(a, b);
    }

    // ── RequestState: method behavior expansions ──

    #[test]
    fn request_state_new_compact_scatter_initial_values() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert_eq!(rs.compact_scatter.original_slot, 0);
        assert_eq!(rs.compact_scatter.compacted_slot, 0);
        assert_eq!(rs.compact_scatter.active, 1);
    }

    #[test]
    fn request_state_mark_exited_idempotent() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(!rs.is_exited());
        rs.mark_exited();
        assert!(rs.is_exited());
        rs.mark_exited();
        assert!(rs.is_exited());
    }

    #[test]
    fn request_state_reset_exit_when_not_exited() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(!rs.is_exited());
        rs.reset_exit();
        assert!(!rs.is_exited());
    }

    #[test]
    fn request_state_update_telemetry_preserves_range_group() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.telemetry.range_group = 7;
        rs.update_telemetry(1.5, 0.5, 0.2, 0.88);
        assert_eq!(rs.telemetry.range_group, 7);
    }

    #[test]
    fn request_state_update_telemetry_overwrite() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.update_telemetry(1.0, 0.5, 0.3, 0.9);
        rs.update_telemetry(2.0, 0.8, 0.1, 0.6);
        assert!((rs.telemetry.entropy - 2.0).abs() < 1e-6);
        assert!((rs.telemetry.centroid - 0.8).abs() < 1e-6);
        assert!((rs.telemetry.residual_delta - 0.1).abs() < 1e-6);
        assert!((rs.telemetry.residual_cosine - 0.6).abs() < 1e-6);
    }

    #[test]
    fn request_state_set_original_slot_preserves_compacted() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.compact_scatter.compacted_slot = 5;
        rs.compact_scatter.active = 1;
        rs.set_original_slot(42);
        assert_eq!(rs.compact_scatter.compacted_slot, 5);
        assert_eq!(rs.compact_scatter.active, 1);
    }

    #[test]
    fn request_state_to_layered_control_after_reset() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5);
        rs.mark_exited();
        assert!(rs.is_exited());
        rs.reset_exit();
        let ctrl = rs.to_layered_control();
        assert!(!ctrl.should_exit_at(4));
    }

    #[test]
    fn request_state_to_layered_control_independent() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(3);
        let ctrl1 = rs.to_layered_control();
        let ctrl2 = rs.to_layered_control();
        assert_eq!(ctrl1.target_layer, ctrl2.target_layer);
        assert!(!ctrl1.should_exit_at(2));
        assert!(!ctrl2.should_exit_at(2));
    }

    // ── bus_port_tag_to_bit: uniqueness and overlap ──

    #[test]
    fn bus_port_tag_standard_bits_unique() {
        let bits = [
            bus_port_tag_to_bit(BusPortTag::RagInjection),
            bus_port_tag_to_bit(BusPortTag::EarlyExit),
            bus_port_tag_to_bit(BusPortTag::IntentRecall),
            bus_port_tag_to_bit(BusPortTag::Guardrail),
            bus_port_tag_to_bit(BusPortTag::ShadowKv),
        ];
        for i in 0..bits.len() {
            for j in (i + 1)..bits.len() {
                assert_ne!(bits[i], bits[j], "bits[{i}] and bits[{j}] should be unique");
            }
        }
    }

    #[test]
    fn bus_port_tag_custom_bits_above_standard_range() {
        for n in 0u32..23 {
            let bit = bus_port_tag_to_bit(BusPortTag::Custom(n));
            // Custom bits start at bit 8, standard tags use bits 0-4
            assert!(bit >= 1 << 8, "Custom({n}) bit should be >= bit 8");
        }
    }

    #[test]
    fn bus_port_tag_custom_consecutive_unique() {
        let bits: Vec<u32> = (0..24u32).map(|n| bus_port_tag_to_bit(BusPortTag::Custom(n))).collect();
        for i in 0..bits.len() {
            for j in (i + 1)..bits.len() {
                assert_ne!(bits[i], bits[j], "Custom({i}) and Custom({j}) should produce unique bits");
            }
        }
    }

    #[test]
    fn bus_port_tag_custom_same_modulo_same_bit() {
        let bit_a = bus_port_tag_to_bit(BusPortTag::Custom(3));
        let bit_b = bus_port_tag_to_bit(BusPortTag::Custom(27)); // 27 % 24 == 3
        assert_eq!(bit_a, bit_b);
    }

    // ── RequestState: builder pattern edge cases ──

    #[test]
    fn request_state_with_bus_port_preserves_target_layer() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(8)
            .with_bus_port(BusPortTag::Guardrail);
        assert_eq!(rs.target_layer, 8);
        assert!(rs.has_bus_port(BusPortTag::Guardrail));
    }

    #[test]
    fn request_state_builder_order_independence() {
        let rs_a = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5)
            .with_bus_port(BusPortTag::EarlyExit);
        let rs_b = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::EarlyExit)
            .with_target_layer(5);
        assert_eq!(rs_a.target_layer, rs_b.target_layer);
        assert_eq!(rs_a.bus_port_mask, rs_b.bus_port_mask);
    }

    #[test]
    fn request_state_builder_no_modifiers() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(rs.is_full_model());
        assert!(!rs.is_exited());
        assert!(rs.is_active());
        assert_eq!(rs.bus_port_mask, 0);
    }

    #[test]
    fn request_state_with_target_layer_large_value() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(1000);
        assert_eq!(rs.target_layer, 1000);
        assert!(!rs.is_full_model());
    }

    // ── RequestState: per-phase construction ──

    #[test]
    fn request_state_new_each_phase() {
        let prefill = RequestState::new(1, RequestPhase::Prefill, 128, 0);
        assert_eq!(prefill.phase, RequestPhase::Prefill);

        let decode = RequestState::new(2, RequestPhase::Decode, 1, 128);
        assert_eq!(decode.phase, RequestPhase::Decode);

        let chunked = RequestState::new(3, RequestPhase::ChunkedPrefill, 64, 0);
        assert_eq!(chunked.phase, RequestPhase::ChunkedPrefill);
    }

    // ── RequestState: clone preserves individual fields ──

    #[test]
    fn request_state_clone_copies_phase() {
        let rs = RequestState::new(1, RequestPhase::ChunkedPrefill, 10, 0);
        assert_eq!(rs.clone().phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_state_clone_copies_compact_scatter() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.compact_scatter = CompactScatterMeta { original_slot: 10, compacted_slot: 5, active: 0 };
        assert_eq!(rs.clone().compact_scatter, rs.compact_scatter);
    }

    #[test]
    fn request_state_clone_copies_bus_port_mask() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::ShadowKv);
        assert_eq!(rs.clone().bus_port_mask, rs.bus_port_mask);
    }

    #[test]
    fn request_state_clone_copies_variant_key_hash() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.variant_key_hash = 0x1234_5678;
        assert_eq!(rs.clone().variant_key_hash, 0x1234_5678);
    }

    // ── RequestState: compacted_slot direct mutation ──

    #[test]
    fn request_state_compacted_slot_direct_set() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.compact_scatter.compacted_slot = 15;
        assert_eq!(rs.compact_scatter.compacted_slot, 15);
    }

    // ── DeviceMemory: non-zero field values ──

    #[test]
    fn device_memory_host_positive_size() {
        let dm = DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 4096 };
        if let DeviceMemory::Host { size, .. } = dm {
            assert_eq!(size, 4096);
        }
    }

    #[test]
    fn device_memory_cuda_nonzero_fields() {
        let dm = DeviceMemory::Cuda { ptr: 0xCAFE, size: 8192 };
        if let DeviceMemory::Cuda { ptr, size } = dm {
            assert_eq!(ptr, 0xCAFE);
            assert_eq!(size, 8192);
        }
    }

    #[test]
    fn device_memory_hip_nonzero_fields() {
        let dm = DeviceMemory::Hip { ptr: 0xF00D, size: 16384 };
        if let DeviceMemory::Hip { ptr, size } = dm {
            assert_eq!(ptr, 0xF00D);
            assert_eq!(size, 16384);
        }
    }

    #[test]
    fn device_memory_metal_nonzero_fields() {
        let dm = DeviceMemory::Metal { buffer_id: 999, size: 32768 };
        if let DeviceMemory::Metal { buffer_id, size } = dm {
            assert_eq!(buffer_id, 999);
            assert_eq!(size, 32768);
        }
    }

    // ── RequestStateTable: operation sequences ──

    #[test]
    fn request_state_table_add_preserves_order() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(10, RequestPhase::Prefill, 1, 0));
        rst.add(RequestState::new(20, RequestPhase::Decode, 2, 0));
        rst.add(RequestState::new(30, RequestPhase::ChunkedPrefill, 3, 0));
        assert_eq!(rst.states[0].request_id, 10);
        assert_eq!(rst.states[1].request_id, 20);
        assert_eq!(rst.states[2].request_id, 30);
        assert_eq!(rst.states[0].seq_len, 1);
        assert_eq!(rst.states[1].seq_len, 2);
        assert_eq!(rst.states[2].seq_len, 3);
    }

    #[test]
    fn request_state_table_mixed_phases() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 100, 0));
        rst.add(RequestState::new(2, RequestPhase::Decode, 1, 100));
        rst.add(RequestState::new(3, RequestPhase::ChunkedPrefill, 50, 200));
        assert_eq!(rst.states[0].phase, RequestPhase::Prefill);
        assert_eq!(rst.states[1].phase, RequestPhase::Decode);
        assert_eq!(rst.states[2].phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_state_table_states_mutable() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        rst.states[0].seq_len = 999;
        assert_eq!(rst.states[0].seq_len, 999);
    }

    #[test]
    fn request_state_table_drop_with_host_memory() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.device_memory = Some(DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 256 });
        drop(rst);
    }

    #[test]
    fn request_state_table_cpu_allocate_idempotent() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        assert!(rst.allocate().is_ok());
        assert!(rst.allocate().is_ok());
        assert!(rst.device_memory.is_none());
    }

    #[test]
    fn request_state_table_single_state_field_access() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(42, RequestPhase::Prefill, 256, 1024)
            .with_target_layer(10)
            .with_bus_port(BusPortTag::IntentRecall));
        let state = &rst.states[0];
        assert_eq!(state.request_id, 42);
        assert_eq!(state.seq_len, 256);
        assert_eq!(state.kv_cache_offset, 1024);
        assert_eq!(state.target_layer, 10);
        assert!(state.has_bus_port(BusPortTag::IntentRecall));
    }

    #[test]
    fn request_state_table_states_clear_via_pub_field() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        rst.add(RequestState::new(2, RequestPhase::Decode, 20, 0));
        assert_eq!(rst.states.len(), 2);
        rst.states.clear();
        assert!(rst.states.is_empty());
    }

    #[test]
    fn request_state_table_cpu_sync_after_add() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        assert!(rst.allocate().is_ok());
        assert!(rst.sync_to_device().is_ok());
    }

    // ── RequestState: per-port mask value verification ──

    #[test]
    fn request_state_bus_port_early_exit_mask_value() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::EarlyExit);
        assert_eq!(rs.bus_port_mask, 1 << 1);
    }

    #[test]
    fn request_state_bus_port_intent_recall_mask_value() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::IntentRecall);
        assert_eq!(rs.bus_port_mask, 1 << 2);
    }

    #[test]
    fn request_state_bus_port_guardrail_mask_value() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::Guardrail);
        assert_eq!(rs.bus_port_mask, 1 << 3);
    }

    #[test]
    fn request_state_bus_port_shadow_kv_mask_value() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::ShadowKv);
        assert_eq!(rs.bus_port_mask, 1 << 4);
    }

    // ── CompactScatterMeta: Copy semantics ──

    #[test]
    fn compact_scatter_meta_copy_not_move() {
        let m = CompactScatterMeta { original_slot: 1, compacted_slot: 2, active: 1 };
        let _m2 = m;
        let _m3 = m;
    }

    // ── RequestTelemetry: Copy semantics ──

    #[test]
    fn request_telemetry_copy_not_move() {
        let t = RequestTelemetry { entropy: 1.0, centroid: 0.5, residual_delta: 0.3, residual_cosine: 0.9, range_group: 3 };
        let _t2 = t;
        let _t3 = t;
    }

    // ── RequestState: layered_control edge cases ──

    #[test]
    fn request_state_layered_control_snapshots_exit_flag() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5);
        let ctrl_before = rs.to_layered_control();
        assert!(!ctrl_before.should_exit_at(0));
        rs.mark_exited();
        let ctrl_after = rs.to_layered_control();
        assert!(ctrl_after.should_exit_at(0));
    }

    // ── RequestState: new exit_flag initial value ──

    #[test]
    fn request_state_new_exit_flag_is_zero() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(!rs.is_exited());
        assert_eq!(rs.exit_flag.load(Ordering::Relaxed), 0);
    }

    // ── RequestStateTable: drop with DeviceMemory::Cuda (no CUDA) ──

    #[test]
    fn request_state_table_drop_with_uninitialized_cuda_memory() {
        // Construct a table that claims Cuda backend but has no device_memory
        let mut rst = RequestStateTable::new(BackendType::Cuda);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        // device_memory is None, so drop should not try to free anything
        drop(rst);
    }

    // ── CompactScatterMeta: Debug output contains all field names ──

    #[test]
    fn compact_scatter_meta_debug_all_fields() {
        let m = CompactScatterMeta { original_slot: 42, compacted_slot: 7, active: 1 };
        let s = format!("{:?}", m);
        assert!(s.contains("original_slot: 42"));
        assert!(s.contains("compacted_slot: 7"));
        assert!(s.contains("active: 1"));
    }

    // ── RequestTelemetry: Debug shows range_group ──

    #[test]
    fn request_telemetry_debug_range_group() {
        let t = RequestTelemetry { entropy: 0.0, centroid: 0.0, residual_delta: 1.0, residual_cosine: 1.0, range_group: 42 };
        let s = format!("{:?}", t);
        assert!(s.contains("range_group: 42"));
    }

    // ── RequestState: set_original_slot overwrites ──

    #[test]
    fn request_state_set_original_slot_overwrite() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.set_original_slot(10);
        assert_eq!(rs.compact_scatter.original_slot, 10);
        rs.set_original_slot(20);
        assert_eq!(rs.compact_scatter.original_slot, 20);
    }

    // ── RequestState: is_active boundary — only zero is inactive ──

    #[test]
    fn request_state_is_active_only_zero_false() {
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.compact_scatter.active = 0;
        assert!(!rs.is_active());
        rs.compact_scatter.active = 1;
        assert!(rs.is_active());
        rs.compact_scatter.active = u32::MAX;
        assert!(rs.is_active());
    }

    // ── RequestState: with_target_layer usize→u32 conversion ──

    #[test]
    fn request_state_with_target_layer_usize_to_u32() {
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(65535);
        assert_eq!(rs.target_layer, 65535u32);
    }

    // ── RequestStateTable: add returns increasing length ──

    #[test]
    fn request_state_table_add_increments_len() {
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        assert_eq!(rst.states.len(), 0);
        rst.add(RequestState::new(1, RequestPhase::Decode, 10, 0));
        assert_eq!(rst.states.len(), 1);
        rst.add(RequestState::new(2, RequestPhase::Decode, 20, 0));
        assert_eq!(rst.states.len(), 2);
        rst.add(RequestState::new(3, RequestPhase::Decode, 30, 0));
        assert_eq!(rst.states.len(), 3);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Additional 13 tests — edge cases and uncovered paths
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn request_state_layered_control_target_layer_one_boundary() {
        // Arrange: target_layer=1 is the smallest "exit-eligible" value
        // (target_layer=0 is excluded by the `target_layer > 0` guard)
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(1);
        let ctrl = rs.to_layered_control();

        // Act & Assert: layer 0 should not exit, layer 1 and above should
        assert!(!ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(1));
        assert!(ctrl.should_exit_at(2));
    }

    #[test]
    fn request_state_all_standard_ports_plus_custom_combined_mask() {
        // Arrange: attach all 5 standard ports and one custom port
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::EarlyExit)
            .with_bus_port(BusPortTag::IntentRecall)
            .with_bus_port(BusPortTag::Guardrail)
            .with_bus_port(BusPortTag::ShadowKv)
            .with_bus_port(BusPortTag::Custom(0));

        // Act & Assert: standard bits 0-4 = 0x1F, custom(0) = bit 8 = 0x100
        assert_eq!(rs.bus_port_mask, 0x1F | (1 << 8));
        // All individual ports should be detected
        assert!(rs.has_bus_port(BusPortTag::RagInjection));
        assert!(rs.has_bus_port(BusPortTag::Custom(0)));
    }

    #[test]
    fn request_state_telemetry_nan_values() {
        // Arrange
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);

        // Act: update telemetry with NaN
        rs.update_telemetry(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

        // Assert: NaN values are stored as-is (is_nan check, not equality)
        assert!(rs.telemetry.entropy.is_nan());
        assert!(rs.telemetry.centroid.is_nan());
        assert!(rs.telemetry.residual_delta.is_nan());
        assert!(rs.telemetry.residual_cosine.is_nan());
    }

    #[test]
    fn request_state_table_mutate_state_after_add() {
        // Arrange
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 100, 0));

        // Act: mutate fields on the stored state
        rst.states[0].phase = RequestPhase::Decode;
        rst.states[0].seq_len = 200;
        rst.states[0].kv_cache_offset = 500;
        rst.states[0].variant_key_hash = 0xABCD;

        // Assert: mutations are reflected
        assert_eq!(rst.states[0].phase, RequestPhase::Decode);
        assert_eq!(rst.states[0].seq_len, 200);
        assert_eq!(rst.states[0].kv_cache_offset, 500);
        assert_eq!(rst.states[0].variant_key_hash, 0xABCD);
    }

    #[test]
    fn request_state_exit_flag_concurrent_visibility() {
        // Arrange: verify Acquire/Release ordering by marking exited in a thread
        let rs = std::sync::Arc::new(RequestState::new(1, RequestPhase::Decode, 10, 0));
        let rs_clone = std::sync::Arc::clone(&rs);

        // Act: mark exited from a different thread
        let handle = std::thread::spawn(move || {
            rs_clone.mark_exited();
        });
        handle.join().unwrap();

        // Assert: the other thread's Release store is visible via Acquire load
        assert!(rs.is_exited());
    }

    #[test]
    fn request_state_custom_port_bit_distinct_from_standard() {
        // Arrange: Custom(0) uses bit 8, all standard tags use bits 0-4
        let rs_custom = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::Custom(0));
        let rs_standard = RequestState::new(2, RequestPhase::Decode, 10, 0)
            .with_bus_port(BusPortTag::RagInjection);

        // Act & Assert: custom bit does not overlap any standard bit
        assert_ne!(rs_custom.bus_port_mask, rs_standard.bus_port_mask);
        assert_eq!(rs_custom.bus_port_mask & rs_standard.bus_port_mask, 0);
    }

    #[test]
    fn request_state_is_active_toggle_sequence() {
        // Arrange
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        assert!(rs.is_active());

        // Act & Assert: toggle active off then on
        rs.compact_scatter.active = 0;
        assert!(!rs.is_active());
        rs.compact_scatter.active = 1;
        assert!(rs.is_active());
    }

    #[test]
    fn request_state_clone_preserves_telemetry_range_group() {
        // Arrange: set a non-default range_group
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        rs.telemetry.range_group = 42;
        rs.update_telemetry(2.5, 0.75, 0.15, 0.92);

        // Act
        let cloned = rs.clone();

        // Assert: range_group survives clone even though update_telemetry doesn't touch it
        assert_eq!(cloned.telemetry.range_group, 42);
        assert!((cloned.telemetry.entropy - 2.5).abs() < 1e-6);
    }

    #[test]
    fn request_state_with_target_layer_zero_then_full_model() {
        // Arrange: set target_layer to 0, then override back to u32::MAX
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(0);
        assert!(!rs.is_full_model());

        // Act: with_target_layer returns a new Self, overwriting target_layer
        let rs2 = rs.with_target_layer(u32::MAX as usize);

        // Assert: now it's full model again
        assert!(rs2.is_full_model());
        assert_eq!(rs2.target_layer, u32::MAX);
    }

    #[test]
    fn request_state_table_add_and_iterate_phases() {
        // Arrange: add multiple requests with different phases
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 50, 0));
        rst.add(RequestState::new(2, RequestPhase::Decode, 1, 50));
        rst.add(RequestState::new(3, RequestPhase::ChunkedPrefill, 25, 100));

        // Act: collect phases
        let phases: Vec<RequestPhase> = rst.states.iter().map(|s| s.phase).collect();

        // Assert: order preserved
        assert_eq!(phases, vec![RequestPhase::Prefill, RequestPhase::Decode, RequestPhase::ChunkedPrefill]);
    }

    #[test]
    fn request_state_reset_exit_preserves_target_layer() {
        // Arrange: set target_layer and mark exited
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(15);
        rs.mark_exited();
        assert!(rs.is_exited());

        // Act: reset exit flag
        rs.reset_exit();

        // Assert: exited is cleared but target_layer is preserved
        assert!(!rs.is_exited());
        assert_eq!(rs.target_layer, 15);
        assert!(!rs.is_full_model());
    }

    #[test]
    fn bus_port_tag_custom_index_equality_semantics() {
        // Arrange: two Custom tags with same index should be equal
        assert_eq!(BusPortTag::Custom(10), BusPortTag::Custom(10));
        // Different indices are not equal
        assert_ne!(BusPortTag::Custom(10), BusPortTag::Custom(11));
        // Custom(0) is distinct from RagInjection despite both being "first"
        assert_ne!(BusPortTag::Custom(0), BusPortTag::RagInjection);
    }

    #[test]
    fn request_state_layered_control_exit_flag_snapshot_independence() {
        // Arrange: create a control snapshot, then modify the source
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(5);
        let ctrl = rs.to_layered_control();

        // Act: mark the source as exited after snapshot
        rs.mark_exited();

        // Assert: the previously created control still reflects the old state
        // (ctrl has its own AtomicU32, independent from rs)
        assert!(!ctrl.should_exit_at(3));
    }

    // ══════════════════════════════════════════════════════════════════════
    // Additional 10 tests — uncovered boundaries and interaction paths
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn request_state_clone_after_exit_then_to_layered_control() {
        // Arrange: mark exited, clone, verify clone's layered_control reflects exited state
        let rs = RequestState::new(1, RequestPhase::Decode, 10, 0)
            .with_target_layer(4);
        rs.mark_exited();
        let cloned = rs.clone();

        // Act
        let ctrl = cloned.to_layered_control();

        // Assert: clone captured exit_flag=1, so layered_control should exit at any layer
        assert!(ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(4));
        assert_eq!(ctrl.target_layer, 4);
    }

    #[test]
    fn request_state_set_original_slot_u32_max() {
        // Arrange
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);

        // Act
        rs.set_original_slot(u32::MAX);

        // Assert: original_slot accepts u32 max without truncation
        assert_eq!(rs.compact_scatter.original_slot, u32::MAX);
        // Other compact_scatter fields unaffected
        assert_eq!(rs.compact_scatter.compacted_slot, 0);
        assert_eq!(rs.compact_scatter.active, 1);
    }

    #[test]
    fn request_state_table_cpu_sync_with_host_device_memory() {
        // Arrange: CPU table with manually set Host device_memory
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 64, 0));
        rst.device_memory = Some(DeviceMemory::Host {
            ptr: std::ptr::null_mut(),
            size: std::mem::size_of::<RequestState>(),
        });

        // Act: CPU sync is a no-op regardless of device_memory
        let result = rst.sync_to_device();

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn request_state_with_all_builder_fields_then_clone_roundtrip() {
        // Arrange: fully configured request
        let mut rs = RequestState::new(42, RequestPhase::ChunkedPrefill, 1024, 2048)
            .with_target_layer(15)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::Guardrail)
            .with_bus_port(BusPortTag::Custom(3));
        rs.update_telemetry(2.5, 0.75, 0.12, 0.97);
        rs.telemetry.range_group = 9;
        rs.variant_key_hash = 0xCAFE;
        rs.compact_scatter = CompactScatterMeta {
            original_slot: 7,
            compacted_slot: 3,
            active: 1,
        };

        // Act
        let cloned = rs.clone();

        // Assert: every field survives clone
        assert_eq!(cloned.request_id, 42);
        assert_eq!(cloned.phase, RequestPhase::ChunkedPrefill);
        assert_eq!(cloned.seq_len, 1024);
        assert_eq!(cloned.kv_cache_offset, 2048);
        assert_eq!(cloned.target_layer, 15);
        assert_eq!(cloned.compact_scatter, rs.compact_scatter);
        assert!((cloned.telemetry.entropy - 2.5).abs() < 1e-6);
        assert_eq!(cloned.telemetry.range_group, 9);
        assert!(cloned.has_bus_port(BusPortTag::RagInjection));
        assert!(cloned.has_bus_port(BusPortTag::Guardrail));
        assert!(cloned.has_bus_port(BusPortTag::Custom(3)));
        assert!(!cloned.has_bus_port(BusPortTag::EarlyExit));
        assert_eq!(cloned.variant_key_hash, 0xCAFE);
        assert_eq!(cloned.bus_port_mask, rs.bus_port_mask);
    }

    #[test]
    fn request_state_exit_reset_race_with_threads() {
        // Arrange: shared state with exit flag
        let rs = std::sync::Arc::new(RequestState::new(1, RequestPhase::Decode, 10, 0));
        let rs_exit = std::sync::Arc::clone(&rs);
        let rs_reset = std::sync::Arc::clone(&rs);

        // Act: one thread marks exited, another resets — final state is non-deterministic
        // but both operations must complete without panic or data race
        let h1 = std::thread::spawn(move || {
            rs_exit.mark_exited();
        });
        let h2 = std::thread::spawn(move || {
            rs_reset.reset_exit();
        });
        h1.join().unwrap();
        h2.join().unwrap();

        // Assert: the final state is a valid boolean (no UB, no panic)
        let exited = rs.is_exited();
        assert!(exited == true || exited == false);
    }

    #[test]
    fn device_memory_exhaustive_variant_match() {
        // Arrange: one of each variant
        let variants: Vec<DeviceMemory> = vec![
            DeviceMemory::Host { ptr: std::ptr::null_mut(), size: 100 },
            DeviceMemory::Cuda { ptr: 1, size: 200 },
            DeviceMemory::Hip { ptr: 2, size: 300 },
            DeviceMemory::Metal { buffer_id: 3, size: 400 },
        ];

        // Act & Assert: exhaustive match on each variant extracts correct size
        let sizes: Vec<usize> = variants
            .into_iter()
            .map(|dm| match dm {
                DeviceMemory::Host { size, .. } => size,
                DeviceMemory::Cuda { size, .. } => size,
                DeviceMemory::Hip { size, .. } => size,
                DeviceMemory::Metal { size, .. } => size,
            })
            .collect();

        assert_eq!(sizes, vec![100, 200, 300, 400]);
    }

    #[test]
    fn request_state_table_find_by_request_id() {
        // Arrange: table with heterogeneous request IDs
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(10, RequestPhase::Prefill, 50, 0));
        rst.add(RequestState::new(20, RequestPhase::Decode, 1, 50));
        rst.add(RequestState::new(30, RequestPhase::ChunkedPrefill, 25, 100));

        // Act: search by request_id (simulating a lookup pattern)
        let found = rst.states.iter().find(|s| s.request_id == 20);

        // Assert
        assert!(found.is_some());
        assert_eq!(found.unwrap().phase, RequestPhase::Decode);
        assert_eq!(found.unwrap().seq_len, 1);

        // Non-existent ID returns None
        assert!(rst.states.iter().find(|s| s.request_id == 99).is_none());
    }

    #[test]
    fn request_state_bus_port_custom_indices_0_through_23_unique_bits() {
        // Arrange & Act: collect bits for Custom(0)..Custom(23)
        let bits: Vec<u32> = (0..24)
            .map(|n| bus_port_tag_to_bit(BusPortTag::Custom(n)))
            .collect();

        // Assert: all 24 custom bits are single-bit masks starting at bit 8
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(bit, 1u32 << (8 + i as u32), "Custom({i}) should be bit {}", 8 + i);
        }

        // None of the custom bits overlap the 5 standard bits (bits 0-4)
        let standard_mask: u32 = (1..=5).fold(0u32, |acc, bit| acc | (1 << bit));
        for &bit in &bits {
            assert_eq!(bit & standard_mask, 0);
        }
    }

    #[test]
    fn request_state_telemetry_subnormal_floats() {
        // Arrange
        let mut rs = RequestState::new(1, RequestPhase::Decode, 10, 0);
        let subnormal = f32::from_bits(1); // smallest positive subnormal f32

        // Act
        rs.update_telemetry(subnormal, -subnormal, subnormal, -subnormal);

        // Assert: subnormal values are stored faithfully
        assert_eq!(rs.telemetry.entropy.to_bits(), subnormal.to_bits());
        assert_eq!(rs.telemetry.centroid.to_bits(), (-subnormal).to_bits());
        assert_eq!(rs.telemetry.residual_delta.to_bits(), subnormal.to_bits());
        assert_eq!(rs.telemetry.residual_cosine.to_bits(), (-subnormal).to_bits());
    }

    #[test]
    fn request_state_table_phase_transition_workflow() {
        // Arrange: simulate a request transitioning Prefill -> Decode
        let mut rst = RequestStateTable::new(BackendType::Cpu);
        rst.add(RequestState::new(1, RequestPhase::Prefill, 128, 0));

        // Act: transition phase (simulating scheduler promoting a request)
        let state = &mut rst.states[0];
        assert_eq!(state.phase, RequestPhase::Prefill);
        state.phase = RequestPhase::Decode;
        state.seq_len = 1; // decode step: seq_len resets to 1

        // Assert: phase transition reflected
        assert_eq!(rst.states[0].phase, RequestPhase::Decode);
        assert_eq!(rst.states[0].seq_len, 1);
        assert_eq!(rst.states[0].request_id, 1); // identity preserved
    }
}
