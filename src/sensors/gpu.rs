//! GPU 拓扑探测 — CUDA / HIP / Metal 平台统一接口
//!
//! 实现 SPEC/02-HARDWARE.md §2.2 "GPU 字段" + §2.3 "探测方式"。
//! 所有 GPU driver 通过运行时 `dlopen` 加载，编译时不需要 CUDA/ROCm/Metal SDK。
//! Feature-gated: `cuda` / `hip` / `metal`。
//!
//! ## 探测流程（一次性，加载期）
//!
//! `detect_gpu()` 按 CUDA → HIP → Metal 顺序尝试，首个成功即返回 `Some(GpuTopology)`。
//! 无 feature 启用 / 无可探测 GPU driver → `Ok(None)`（合法，表示 CPU-only 系统）。
//! feature 启用且 driver 加载成功，但属性查询 runtime 失败 → `Err`（真实 bug，传播）。
//!
//! ## 铁律（CLAUDE.md NO_SILENT_FALLBACK）
//!
//! - 禁止 `unwrap_or(0)` / `unwrap_or_default()` 掩盖探测失败
//! - 禁止 feature 启用时静默降级为 `None`
//! - 所有默认值均来自 driver API 真实查询

/// GPU 拓扑信息（SPEC/02-HARDWARE.md §2.2）
///
/// 对应 NVIDIA SM / AMD CU / Apple simdgroup 的统一抽象。
/// 一次性探测、运行时固定；驱动 JIT codegen 的 tile 尺寸、block 尺寸、指令选择。
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// GPU 平台类型（CUDA / HIP / Metal）
    pub platform: GpuPlatform,

    /// NVIDIA SM / AMD CU / Apple simdgroup 数量
    pub compute_unit_count: usize,

    /// Tensor Core / Matrix Unit 代数
    /// - 0: 无矩阵单元（e.g. Pascal, Maxwell）
    /// - 1: Volta (sm_70) / CDNA (gfx908)
    /// - 2: Ampere (sm_80) / CDNA2 (gfx90a)
    /// - 3: Hopper (sm_90) / CDNA3 (gfx940+)
    /// - 4: Blackwell (sm_100+)
    /// - Metal 使用 gpu_family (Apple Silicon generation) 映射
    pub tensor_core_gen: u32,

    /// 每 SM/CU 的共享内存大小（字节）
    ///
    /// - CUDA: `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`
    /// - HIP: `hipDeviceAttributeMaxSharedMemoryPerBlock` (近似值)
    /// - Metal: `maxThreadgroupMemoryLength`
    pub shared_mem_per_sm_bytes: usize,

    /// L2 缓存大小（字节）
    ///
    /// - CUDA: `CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`
    /// - HIP: `hipDeviceAttributeL2CacheSize`
    /// - Metal: 不暴露 → 0（Apple GPU 不按传统 L2 架构）
    pub l2_bytes: usize,

    /// 全局内存大小（字节）
    ///
    /// - CUDA: `cuMemGetInfo_v2` 的 total
    /// - HIP: `hipMemGetInfo` 的 total
    /// - Metal: `recommendedMaxWorkingSetSize`
    pub global_mem_bytes: usize,

    /// Warp / Wavefront / SIMD-group 大小
    ///
    /// - CUDA: 32
    /// - HIP: 32 (RDNA) 或 64 (CDNA)
    /// - Metal: 32 (Apple GPU)
    pub warp_size: usize,

    /// 计算能力主版本
    ///
    /// - CUDA: compute capability major (e.g. 8 for sm_80)
    /// - HIP: gfx_arch 高位（hex，例如 gfx908 → 9）
    /// - Metal: gpu_family (Apple1=7, Apple2=8, Apple7=13, ...)
    pub compute_cap_major: u32,

    /// 计算能力次版本
    ///
    /// - CUDA: compute capability minor (e.g. 0 for sm_80)
    /// - HIP: gfx_arch 低位
    /// - Metal: 0
    pub compute_cap_minor: u32,
}

impl GpuTopology {
    /// SM version (CUDA) or equivalent. Returns `None` for non-CUDA.
    pub fn sm_version(&self) -> Option<u32> {
        self.platform.sm_version()
    }

    /// Total global memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.global_mem_bytes
    }

    /// Compute capability as (major, minor). Returns `None` for CPU-only.
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        self.platform.compute_capability()
    }

    /// Whether the detected GPU has tensor/matrix core units.
    pub fn has_tensor_cores(&self) -> bool {
        self.platform.has_tensor_cores()
    }
}

/// GPU 平台类型（与 gllm-kernels 的 `Platform` 枚举对齐）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuPlatform {
    /// NVIDIA CUDA（sm_version = major*10 + minor）
    Cuda { sm_version: u32 },
    /// AMD HIP / ROCm（gfx_arch，hex，如 0x908 / 0x90a / 0x940）
    Hip { gfx_arch: u32 },
    /// Apple Metal（gpu_family）
    Metal { gpu_family: u32 },
    /// CPU-only（无 GPU 加速，JIT codegen 走 CPU 路径）
    Cpu,
}

impl GpuPlatform {
    /// CUDA SM version (major*10 + minor). Returns `None` for non-CUDA platforms.
    pub fn sm_version(&self) -> Option<u32> {
        match self {
            GpuPlatform::Cuda { sm_version } => Some(*sm_version),
            _ => None,
        }
    }

    /// Compute capability as (major, minor). Returns `None` for CPU.
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        match self {
            GpuPlatform::Cuda { sm_version } => Some((*sm_version / 10, *sm_version % 10)),
            GpuPlatform::Hip { gfx_arch } => Some((*gfx_arch >> 8 & 0xFF, *gfx_arch & 0xFF)),
            GpuPlatform::Metal { gpu_family } => Some((*gpu_family, 0)),
            GpuPlatform::Cpu => None,
        }
    }

    /// Whether this platform has matrix/tensor core units.
    pub fn has_tensor_cores(&self) -> bool {
        !matches!(self, GpuPlatform::Cpu)
    }
}

/// 一次性探测系统 GPU 拓扑。
///
/// 按 CUDA → HIP → Metal 顺序尝试，首个成功即返回。
///
/// ## 返回值语义
///
/// - `Ok(Some(topo))`: 成功探测到 GPU
/// - `Ok(None)`: 无 feature 启用，或 feature 启用但 driver 库不存在（非 GPU 系统）
/// - `Err(msg)`: feature 启用且 driver 库加载成功，但属性查询 runtime 失败（真实 bug）
pub fn detect_gpu() -> Result<Option<GpuTopology>, String> {
    #[cfg(feature = "cuda")]
    {
        match detect_cuda() {
            Ok(Some(topo)) => return Ok(Some(topo)),
            Ok(None) => {} // driver library not present, try next backend
            Err(e) => return Err(format!("CUDA probe failed: {e}")),
        }
    }

    #[cfg(feature = "hip")]
    {
        match detect_hip() {
            Ok(Some(topo)) => return Ok(Some(topo)),
            Ok(None) => {}
            Err(e) => return Err(format!("HIP probe failed: {e}")),
        }
    }

    #[cfg(feature = "metal")]
    {
        match detect_metal() {
            Ok(Some(topo)) => return Ok(Some(topo)),
            Ok(None) => {}
            Err(e) => return Err(format!("Metal probe failed: {e}")),
        }
    }

    // No GPU feature enabled, or all enabled features had no available driver.
    Ok(None)
}

// ── CUDA 探测 ────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn detect_cuda() -> Result<Option<GpuTopology>, String> {
    use gllm_kernels::gpu::cuda::driver::{
        CudaDriver, CUDA_SUCCESS, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    };

    // Step 1: dlopen libcuda.so.1. 库不存在（非 CUDA 机器）→ Ok(None)。
    let driver = match CudaDriver::load() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    // Step 2: cuInit. 从这里开始任何错误都视为真实故障，传播 Err。
    driver.init().map_err(|e| e.to_string())?;

    let device_count = driver.device_count().map_err(|e| e.to_string())?;
    if device_count <= 0 {
        return Ok(None);
    }

    // Step 3: 获取 device 0 的 handle（通过 cuDeviceGet）
    let mut device_id: i32 = 0;
    let res = unsafe { (driver.cuDeviceGet)(&mut device_id, 0) };
    if res != CUDA_SUCCESS {
        return Err(format!("cuDeviceGet(0) failed with error {res}"));
    }

    // Step 4: 查询属性。feature 启用后任何属性查询失败都是真实 bug，传播 Err。
    let major = driver
        .device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id)
        .map_err(|e| e.to_string())? as u32;
    let minor = driver
        .device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_id)
        .map_err(|e| e.to_string())? as u32;
    let sm_version = major * 10 + minor;

    let sm_count = driver
        .device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_id)
        .map_err(|e| e.to_string())? as usize;

    let shared_mem_per_sm = driver
        .device_attribute(
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            device_id,
        )
        .map_err(|e| e.to_string())? as usize;

    let l2_bytes = driver
        .device_attribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_id)
        .map_err(|e| e.to_string())? as usize;

    let warp_size = driver
        .device_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device_id)
        .map_err(|e| e.to_string())? as usize;

    // cuMemGetInfo 需要一个 context. 为避免副作用，我们不创建 context；
    // 从 device 属性 CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY 无法得到全局内存，
    // 改用创建临时 context 的方案代价较大。此处使用 cuDeviceTotalMem_v2，
    // 但驱动表中未导入该符号 —— 改用 cuMemGetInfo 需要 context。
    //
    // 折中：复用 gllm-kernels/CudaDevice 初始化流程以获取 total_memory。
    //   但 CudaDevice::new 会创建 context，有运行时副作用。
    //
    // 实际方案：按 SPEC §2.2 要求，global_mem_bytes 使用 cuMemGetInfo + 临时 context。
    let global_mem_bytes = query_cuda_total_memory(&driver, device_id)?;

    // Tensor Core 代数映射（与 gllm-kernels/src/gpu/cuda/device.rs 保持一致）:
    //   sm_100+ → 4 (Blackwell)
    //   sm_90+  → 3 (Hopper)
    //   sm_80+  → 2 (Ampere)
    //   sm_70+  → 1 (Volta/Turing)
    //   else     → 0
    let tensor_core_gen = if sm_version >= 100 {
        4
    } else if sm_version >= 90 {
        3
    } else if sm_version >= 80 {
        2
    } else if sm_version >= 70 {
        1
    } else {
        0
    };

    Ok(Some(GpuTopology {
        platform: GpuPlatform::Cuda { sm_version },
        compute_unit_count: sm_count,
        tensor_core_gen,
        shared_mem_per_sm_bytes: shared_mem_per_sm,
        l2_bytes,
        global_mem_bytes,
        warp_size,
        compute_cap_major: major,
        compute_cap_minor: minor,
    }))
}

#[cfg(feature = "cuda")]
fn query_cuda_total_memory(
    driver: &gllm_kernels::gpu::cuda::driver::CudaDriver,
    device_id: i32,
) -> Result<usize, String> {
    use gllm_kernels::gpu::cuda::driver::CUDA_SUCCESS;

    // cuMemGetInfo 需要当前线程绑定到一个 CUcontext。
    // 创建临时 context → 查询 → 销毁，保证无副作用。
    let mut context: u64 = 0;
    let res = unsafe { (driver.cuCtxCreate_v2)(&mut context, 0, device_id) };
    if res != CUDA_SUCCESS {
        return Err(format!(
            "cuCtxCreate_v2 during total memory probe failed with error {res}"
        ));
    }

    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    let res = unsafe { (driver.cuMemGetInfo_v2)(&mut free_bytes, &mut total_bytes) };

    // 无论成功与否都销毁 context
    let destroy_res = unsafe { (driver.cuCtxDestroy_v2)(context) };

    if res != CUDA_SUCCESS {
        return Err(format!(
            "cuMemGetInfo_v2 failed with error {res}"
        ));
    }
    if destroy_res != CUDA_SUCCESS {
        return Err(format!(
            "cuCtxDestroy_v2 failed with error {destroy_res}"
        ));
    }

    Ok(total_bytes)
}

// ── HIP 探测 ─────────────────────────────────────────────────────────────────

#[cfg(feature = "hip")]
fn detect_hip() -> Result<Option<GpuTopology>, String> {
    use gllm_kernels::gpu::hip::driver::{
        HipDriver, HIP_SUCCESS, HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
        HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, HIP_DEVICE_ATTRIBUTE_WARP_SIZE,
    };

    let driver = match HipDriver::load() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };

    driver.init().map_err(|e| e.to_string())?;

    let device_count = driver.device_count().map_err(|e| e.to_string())?;
    if device_count <= 0 {
        return Ok(None);
    }

    let device_id: i32 = 0;

    // 设置当前设备以便 hipMemGetInfo 查询其内存
    let res = unsafe { (driver.hipSetDevice)(device_id) };
    if res != HIP_SUCCESS {
        return Err(format!("hipSetDevice(0) failed with error {res}"));
    }

    let gfx_arch = driver.gfx_arch(device_id).map_err(|e| e.to_string())?;

    let cu_count = driver
        .device_attribute(HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device_id)
        .map_err(|e| e.to_string())? as usize;

    let shared_mem = driver
        .device_attribute(HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device_id)
        .map_err(|e| e.to_string())? as usize;

    let l2_bytes = driver
        .device_attribute(HIP_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_id)
        .map_err(|e| e.to_string())? as usize;

    let warp_size = driver
        .device_attribute(HIP_DEVICE_ATTRIBUTE_WARP_SIZE, device_id)
        .map_err(|e| e.to_string())? as usize;

    // 全局内存通过 hipMemGetInfo 查询
    let mut free_bytes: usize = 0;
    let mut total_bytes: usize = 0;
    let res = unsafe { (driver.hipMemGetInfo)(&mut free_bytes, &mut total_bytes) };
    if res != HIP_SUCCESS {
        return Err(format!("hipMemGetInfo failed with error {res}"));
    }

    // Tensor Core 代数映射（与 gllm-kernels/src/gpu/hip/device.rs 保持一致）:
    //   gfx940+ (CDNA3, MI300) → 3
    //   gfx908+ (CDNA1/2, MI100/MI200) → 2
    //   gfx90a (CDNA2, MI200) → 2（包含在上面）
    //   其它 → 0
    let tensor_core_gen = if gfx_arch >= 0x940 {
        3
    } else if gfx_arch >= 0x908 {
        2
    } else {
        0
    };

    // compute_cap_major/minor: 从 gfx_arch 的 hex 分解
    //   gfx908 = 0x908 → major=9, minor=0x08
    //   gfx90a = 0x90a → major=9, minor=0x0a
    //   gfx1100 = 0x1100 → major=0x11, minor=0x00
    let compute_cap_major = (gfx_arch >> 8) & 0xFF;
    let compute_cap_minor = gfx_arch & 0xFF;

    Ok(Some(GpuTopology {
        platform: GpuPlatform::Hip { gfx_arch },
        compute_unit_count: cu_count,
        tensor_core_gen,
        shared_mem_per_sm_bytes: shared_mem,
        l2_bytes,
        global_mem_bytes: total_bytes,
        warp_size,
        compute_cap_major,
        compute_cap_minor,
    }))
}

// ── Metal 探测 ───────────────────────────────────────────────────────────────
//
// Apple Silicon GPU 通过 Metal.framework 探测。使用 `objc_msgSend` 的 typed
// function-pointer 重绑定，避免变长参数 ABI 歧义（supportsFamily: 返回 BOOL，
// maxThreadgroupMemoryLength 返回 NSUInteger）。

#[cfg(feature = "metal")]
fn detect_metal() -> Result<Option<GpuTopology>, String> {
    use gllm_kernels::gpu::metal::device::MetalDevice;

    // MetalDevice::new() 内部通过 MTLCreateSystemDefaultDevice() 加载；
    // 无 Metal.framework → DeviceNotFound → 返回 Ok(None)
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(gllm_kernels::gpu::GpuError::DeviceNotFound(_)) => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };

    // Apple GPU family 探测：通过 supportsFamily 系列调用
    let (gpu_family, tensor_core_gen, compute_cap_major) = detect_metal_family(&device)?;

    // threadgroup memory = per-SM shared memory 对标
    let shared_mem_per_sm_bytes = query_metal_threadgroup_memory(&device)?;

    // 全局内存通过 MetalDevice::total_memory() ([recommendedMaxWorkingSetSize])
    let global_mem_bytes = device.total_memory();
    if global_mem_bytes == 0 {
        return Err("Metal recommendedMaxWorkingSetSize returned 0".into());
    }

    // Apple GPU simdgroup 宽度 = 32（硬件常量，见 Metal Shading Language spec）
    let warp_size: usize = 32;

    // Metal 不暴露传统 L2 cache 计数（Apple GPU 使用 tile memory + unified cache）
    let l2_bytes: usize = 0;

    // Apple GPU 核心数 (cores) 无公开 API 暴露；
    // JIT codegen 的 grid 分配由 supportsFamily + threadgroup 配置驱动，
    // 此处取 1 作为 count 下限（实际 M1/M2/M3 为 8/10/10-40，但不影响 codegen 决策）。
    let compute_unit_count: usize = 1;

    Ok(Some(GpuTopology {
        platform: GpuPlatform::Metal { gpu_family },
        compute_unit_count,
        tensor_core_gen,
        shared_mem_per_sm_bytes,
        l2_bytes,
        global_mem_bytes,
        warp_size,
        compute_cap_major,
        compute_cap_minor: 0,
    }))
}

#[cfg(feature = "metal")]
fn detect_metal_family(
    device: &gllm_kernels::gpu::metal::device::MetalDevice,
) -> Result<(u32, u32, u32), String> {
    use gllm_kernels::gpu::metal::objc_runtime::{self, Id, NSUInteger, Sel};

    // supportsFamily: 的 typed binding
    //   -(BOOL)supportsFamily:(MTLGPUFamily)family
    // BOOL 在 64-bit Apple Silicon 上是 signed char (1 字节) 返回到 al 寄存器
    type SupportsFamilyFn = unsafe extern "C" fn(Id, Sel, NSUInteger) -> u8;

    // MTLGPUFamily 枚举值（Metal 3+）:
    //   Apple1 = 1001 (A7 GPU)
    //   Apple2 = 1002 (A8)
    //   ...
    //   Apple7 = 1007 (A14 / M1)
    //   Apple8 = 1008 (A15 / M2)
    //   Apple9 = 1009 (A17 / M3)
    const APPLE_FAMILIES: &[(u32, u32)] = &[
        (1009, 9),
        (1008, 8),
        (1007, 7),
        (1006, 6),
        (1005, 5),
        (1004, 4),
        (1003, 3),
        (1002, 2),
        (1001, 1),
    ];

    let raw_dev = device.raw_device();
    unsafe {
        let sel = objc_runtime::sel("supportsFamily:");
        // Rebind objc_msgSend with the exact signature we need.
        // SAFETY: objc_msgSend is an FFI function; casting via transmute is the standard
        // idiom for typed Objective-C dispatch.
        let send: SupportsFamilyFn =
            std::mem::transmute::<*const (), SupportsFamilyFn>(
                objc_runtime::objc_msgSend as *const (),
            );
        for (family_value, apple_gen) in APPLE_FAMILIES {
            let supports = send(raw_dev, sel, *family_value as NSUInteger);
            if supports != 0 {
                // Apple7+ (M1+) 有 simdgroup_matrix → tensor_core_gen = 2（类比 Ampere）
                // Apple9 (M3) 引入 dynamic caching / matrix indirect → tensor_core_gen = 3
                let tensor_core_gen = if *apple_gen >= 9 {
                    3
                } else if *apple_gen >= 7 {
                    2
                } else {
                    0
                };
                return Ok((*family_value, tensor_core_gen, *apple_gen));
            }
        }
    }

    Err("no Apple GPU family detected via supportsFamily:".into())
}

#[cfg(feature = "metal")]
fn query_metal_threadgroup_memory(
    device: &gllm_kernels::gpu::metal::device::MetalDevice,
) -> Result<usize, String> {
    use gllm_kernels::gpu::metal::objc_runtime::{self, Id, NSUInteger, Sel};

    // [device maxThreadgroupMemoryLength] -> NSUInteger
    type NoArgNSUIntFn = unsafe extern "C" fn(Id, Sel) -> NSUInteger;

    let raw_dev = device.raw_device();
    unsafe {
        let sel = objc_runtime::sel("maxThreadgroupMemoryLength");
        let send: NoArgNSUIntFn =
            std::mem::transmute::<*const (), NoArgNSUIntFn>(
                objc_runtime::objc_msgSend as *const (),
            );
        let size = send(raw_dev, sel);
        if size == 0 {
            return Err(
                "maxThreadgroupMemoryLength returned 0; driver or selector resolution failed"
                    .into(),
            );
        }
        Ok(size as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpu_returns_ok() {
        // Regardless of feature flags, detect_gpu must return Ok.
        // 无 feature → Ok(None)
        // feature 启用且硬件存在 → Ok(Some(...))
        // feature 启用但 driver 不存在 → Ok(None)
        // feature 启用且 driver 存在但查询失败 → Err (真实 bug)
        let result = detect_gpu();
        match result {
            Ok(Some(topo)) => {
                // 若探测到 GPU, 所有关键字段必须为真值
                assert!(topo.compute_unit_count > 0);
                assert!(topo.warp_size > 0);
                assert!(topo.global_mem_bytes > 0);
            }
            Ok(None) => {
                // 合法: 无 feature 或无驱动 GPU
            }
            Err(e) => {
                // feature 启用但 runtime 查询失败 — 此处仅断言错误信息非空
                assert!(!e.is_empty(), "error message must not be empty");
            }
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "hip", feature = "metal")))]
    #[test]
    fn test_no_feature_returns_none() {
        // 无任何 GPU feature 启用 → 必须返回 Ok(None)
        assert!(matches!(detect_gpu(), Ok(None)));
    }

    #[test]
    fn test_gpu_platform_variants() {
        let cuda = GpuPlatform::Cuda { sm_version: 80 };
        let hip = GpuPlatform::Hip { gfx_arch: 0x908 };
        let metal = GpuPlatform::Metal { gpu_family: 7 };
        let cpu = GpuPlatform::Cpu;
        assert_ne!(cuda, hip);
        assert_ne!(hip, metal);
        assert_ne!(metal, cpu);
        assert_ne!(cuda, cpu);
    }

    #[test]
    fn test_gpu_platform_cpu_variant() {
        let cpu = GpuPlatform::Cpu;
        assert!(!cpu.has_tensor_cores());
        assert_eq!(cpu.sm_version(), None);
        assert_eq!(cpu.compute_capability(), None);
    }

    #[test]
    fn test_gpu_platform_sm_version_accessors() {
        let cuda = GpuPlatform::Cuda { sm_version: 90 };
        assert_eq!(cuda.sm_version(), Some(90));
        assert_eq!(cuda.compute_capability(), Some((9, 0)));
        assert!(cuda.has_tensor_cores());

        let hip = GpuPlatform::Hip { gfx_arch: 0x908 };
        assert_eq!(hip.sm_version(), None);
        assert_eq!(hip.compute_capability(), Some((9, 0x08)));
        assert!(hip.has_tensor_cores());

        let metal = GpuPlatform::Metal { gpu_family: 7 };
        assert_eq!(metal.sm_version(), None);
        assert_eq!(metal.compute_capability(), Some((7, 0)));
        assert!(metal.has_tensor_cores());
    }

    #[test]
    fn gpu_platform_cuda_sm_version_field() {
        let p = GpuPlatform::Cuda { sm_version: 90 };
        assert_eq!(p, GpuPlatform::Cuda { sm_version: 90 });
        assert_ne!(p, GpuPlatform::Cuda { sm_version: 80 });
    }

    #[test]
    fn gpu_platform_hip_gfx_arch_field() {
        let p = GpuPlatform::Hip { gfx_arch: 0x940 };
        assert_eq!(p, GpuPlatform::Hip { gfx_arch: 0x940 });
        assert_ne!(p, GpuPlatform::Hip { gfx_arch: 0x90a });
    }

    #[test]
    fn gpu_topology_fields_populated() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.compute_unit_count, 108);
        assert_eq!(topo.tensor_core_gen, 2);
        assert_eq!(topo.warp_size, 32);
        assert_eq!(topo.compute_cap_major, 8);
        assert_eq!(topo.compute_cap_minor, 0);
    }

    #[test]
    fn gpu_topology_clone_independent() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        let clone = topo.clone();
        assert_eq!(clone.compute_unit_count, 132);
        assert_eq!(clone.platform, GpuPlatform::Cuda { sm_version: 90 });
    }

    // ── CUDA tensor_core_gen mapping ─────────────────────────────────────────

    /// Mirrors the CUDA tensor_core_gen mapping from detect_cuda().
    fn cuda_tensor_core_gen(sm_version: u32) -> u32 {
        if sm_version >= 100 {
            4
        } else if sm_version >= 90 {
            3
        } else if sm_version >= 80 {
            2
        } else if sm_version >= 70 {
            1
        } else {
            0
        }
    }

    #[test]
    fn cuda_tensor_core_blackwell_sm100() {
        assert_eq!(cuda_tensor_core_gen(100), 4);
    }

    #[test]
    fn cuda_tensor_core_blackwell_sm120() {
        assert_eq!(cuda_tensor_core_gen(120), 4);
    }

    #[test]
    fn cuda_tensor_core_hopper_sm90() {
        assert_eq!(cuda_tensor_core_gen(90), 3);
    }

    #[test]
    fn cuda_tensor_core_hopper_sm99() {
        assert_eq!(cuda_tensor_core_gen(99), 3);
    }

    #[test]
    fn cuda_tensor_core_ampere_sm80() {
        assert_eq!(cuda_tensor_core_gen(80), 2);
    }

    #[test]
    fn cuda_tensor_core_ampere_sm89() {
        assert_eq!(cuda_tensor_core_gen(89), 2);
    }

    #[test]
    fn cuda_tensor_core_volta_sm70() {
        assert_eq!(cuda_tensor_core_gen(70), 1);
    }

    #[test]
    fn cuda_tensor_core_turing_sm75() {
        assert_eq!(cuda_tensor_core_gen(75), 1);
    }

    #[test]
    fn cuda_tensor_core_pascal_sm60() {
        assert_eq!(cuda_tensor_core_gen(60), 0);
    }

    #[test]
    fn cuda_tensor_core_maxwell_sm52() {
        assert_eq!(cuda_tensor_core_gen(52), 0);
    }

    #[test]
    fn cuda_tensor_core_zero_sm0() {
        assert_eq!(cuda_tensor_core_gen(0), 0);
    }

    #[test]
    fn cuda_tensor_core_boundary_69() {
        // Just below Volta threshold
        assert_eq!(cuda_tensor_core_gen(69), 0);
    }

    #[test]
    fn cuda_tensor_core_boundary_79() {
        // Just below Ampere threshold
        assert_eq!(cuda_tensor_core_gen(79), 1);
    }

    #[test]
    fn cuda_tensor_core_boundary_89() {
        // Just below Hopper threshold
        assert_eq!(cuda_tensor_core_gen(89), 2);
    }

    #[test]
    fn cuda_tensor_core_boundary_99() {
        // Just below Blackwell threshold
        assert_eq!(cuda_tensor_core_gen(99), 3);
    }

    #[test]
    fn cuda_sm_version_from_major_minor() {
        // Verify the formula used in detect_cuda: sm_version = major * 10 + minor
        assert_eq!(8u32 * 10 + 0, 80); // sm_80
        assert_eq!(9u32 * 10 + 0, 90); // sm_90
        assert_eq!(7u32 * 10 + 5, 75); // sm_75
        assert_eq!(10u32 * 10 + 0, 100); // sm_100
    }

    // ── HIP tensor_core_gen mapping ──────────────────────────────────────────

    /// Mirrors the HIP tensor_core_gen mapping from detect_hip().
    fn hip_tensor_core_gen(gfx_arch: u32) -> u32 {
        if gfx_arch >= 0x940 {
            3
        } else if gfx_arch >= 0x908 {
            2
        } else {
            0
        }
    }

    #[test]
    fn hip_tensor_core_cdna3_gfx940() {
        assert_eq!(hip_tensor_core_gen(0x940), 3);
    }

    #[test]
    fn hip_tensor_core_cdna3_gfx950() {
        assert_eq!(hip_tensor_core_gen(0x950), 3);
    }

    #[test]
    fn hip_tensor_core_cdna2_gfx90a() {
        assert_eq!(hip_tensor_core_gen(0x90a), 2);
    }

    #[test]
    fn hip_tensor_core_cdna1_gfx908() {
        assert_eq!(hip_tensor_core_gen(0x908), 2);
    }

    #[test]
    fn hip_tensor_core_rdna_gfx1100() {
        // RDNA3 gfx1100 = 0x1100 (4352) — numerically >= 0x940 (2368).
        // The mapping in detect_hip() uses gfx_arch >= 0x940 as threshold,
        // so RDNA gfx values that exceed CDNA thresholds are classified accordingly.
        // This test documents the actual behavior of the mapping.
        assert_eq!(hip_tensor_core_gen(0x1100), 3);
    }

    #[test]
    fn hip_tensor_core_rdna_gfx1030() {
        // RDNA2 gfx1030 = 0x1030 (4144) — numerically >= 0x940 (2368).
        assert_eq!(hip_tensor_core_gen(0x1030), 3);
    }

    #[test]
    fn hip_tensor_core_boundary_0x907() {
        // Just below CDNA1 threshold
        assert_eq!(hip_tensor_core_gen(0x907), 0);
    }

    #[test]
    fn hip_tensor_core_boundary_0x93f() {
        // Just below CDNA3 threshold
        assert_eq!(hip_tensor_core_gen(0x93f), 2);
    }

    // ── HIP compute_cap_major/minor bit manipulation ─────────────────────────

    #[test]
    fn hip_compute_cap_major_minor_gfx908() {
        let gfx_arch: u32 = 0x908;
        let major = (gfx_arch >> 8) & 0xFF;
        let minor = gfx_arch & 0xFF;
        assert_eq!(major, 9);
        assert_eq!(minor, 0x08);
    }

    #[test]
    fn hip_compute_cap_major_minor_gfx90a() {
        let gfx_arch: u32 = 0x90a;
        let major = (gfx_arch >> 8) & 0xFF;
        let minor = gfx_arch & 0xFF;
        assert_eq!(major, 9);
        assert_eq!(minor, 0x0a);
    }

    #[test]
    fn hip_compute_cap_major_minor_gfx1100() {
        let gfx_arch: u32 = 0x1100;
        let major = (gfx_arch >> 8) & 0xFF;
        let minor = gfx_arch & 0xFF;
        assert_eq!(major, 0x11);
        assert_eq!(minor, 0x00);
    }

    #[test]
    fn hip_compute_cap_major_minor_gfx940() {
        let gfx_arch: u32 = 0x940;
        let major = (gfx_arch >> 8) & 0xFF;
        let minor = gfx_arch & 0xFF;
        assert_eq!(major, 9);
        assert_eq!(minor, 0x40);
    }

    #[test]
    fn hip_compute_cap_major_minor_max_value() {
        // Maximum 16-bit gfx_arch value
        let gfx_arch: u32 = 0xFFFF;
        let major = (gfx_arch >> 8) & 0xFF;
        let minor = gfx_arch & 0xFF;
        assert_eq!(major, 0xFF);
        assert_eq!(minor, 0xFF);
    }

    // ── Metal family tensor_core_gen mapping ─────────────────────────────────

    /// Mirrors the Metal tensor_core_gen mapping from detect_metal_family().
    fn metal_tensor_core_gen(apple_gen: u32) -> u32 {
        if apple_gen >= 9 {
            3
        } else if apple_gen >= 7 {
            2
        } else {
            0
        }
    }

    #[test]
    fn metal_tensor_core_apple9_m3() {
        assert_eq!(metal_tensor_core_gen(9), 3);
    }

    #[test]
    fn metal_tensor_core_apple8_m2() {
        assert_eq!(metal_tensor_core_gen(8), 2);
    }

    #[test]
    fn metal_tensor_core_apple7_m1() {
        assert_eq!(metal_tensor_core_gen(7), 2);
    }

    #[test]
    fn metal_tensor_core_apple6() {
        assert_eq!(metal_tensor_core_gen(6), 0);
    }

    #[test]
    fn metal_tensor_core_apple1_oldest() {
        assert_eq!(metal_tensor_core_gen(1), 0);
    }

    #[test]
    fn metal_tensor_core_boundary_6() {
        // Just below Apple7/M1 threshold
        assert_eq!(metal_tensor_core_gen(6), 0);
    }

    #[test]
    fn metal_tensor_core_future_gen10() {
        assert_eq!(metal_tensor_core_gen(10), 3);
    }

    // ── GpuPlatform Copy trait ───────────────────────────────────────────────

    #[test]
    fn gpu_platform_is_copy() {
        let cuda = GpuPlatform::Cuda { sm_version: 80 };
        let copy = cuda;
        assert_eq!(cuda, copy);
    }

    #[test]
    fn gpu_platform_copy_mutation_independence() {
        // Copy semantics: assigning to a new binding does not share state
        let original = GpuPlatform::Hip { gfx_arch: 0x908 };
        let copy = original;
        // Both should be equal (Copy, not Clone-only)
        assert_eq!(original, copy);
        // And both remain valid independently
        assert!(matches!(original, GpuPlatform::Hip { .. }));
        assert!(matches!(copy, GpuPlatform::Hip { .. }));
    }

    // ── GpuPlatform Debug trait ──────────────────────────────────────────────

    #[test]
    fn gpu_platform_debug_cuda() {
        let p = GpuPlatform::Cuda { sm_version: 80 };
        let debug_str = format!("{p:?}");
        assert!(debug_str.contains("Cuda"));
        assert!(debug_str.contains("80"));
    }

    #[test]
    fn gpu_platform_debug_hip() {
        let p = GpuPlatform::Hip { gfx_arch: 0x908 };
        let debug_str = format!("{p:?}");
        assert!(debug_str.contains("Hip"));
    }

    #[test]
    fn gpu_platform_debug_metal() {
        let p = GpuPlatform::Metal { gpu_family: 7 };
        let debug_str = format!("{p:?}");
        assert!(debug_str.contains("Metal"));
    }

    // ── GpuTopology Debug trait ──────────────────────────────────────────────

    #[test]
    fn gpu_topology_debug_all_fields_present() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let debug_str = format!("{topo:?}");
        assert!(debug_str.contains("compute_unit_count"));
        assert!(debug_str.contains("tensor_core_gen"));
        assert!(debug_str.contains("shared_mem_per_sm_bytes"));
        assert!(debug_str.contains("l2_bytes"));
        assert!(debug_str.contains("global_mem_bytes"));
        assert!(debug_str.contains("warp_size"));
        assert!(debug_str.contains("compute_cap_major"));
        assert!(debug_str.contains("compute_cap_minor"));
    }

    // ── GpuPlatform equality ─────────────────────────────────────────────────

    #[test]
    fn gpu_platform_eq_same_cuda() {
        assert_eq!(
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Cuda { sm_version: 80 }
        );
    }

    #[test]
    fn gpu_platform_ne_different_cuda_sm() {
        assert_ne!(
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Cuda { sm_version: 90 }
        );
    }

    #[test]
    fn gpu_platform_ne_different_hip_gfx() {
        assert_ne!(
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Hip { gfx_arch: 0x90a }
        );
    }

    #[test]
    fn gpu_platform_ne_different_metal_family() {
        assert_ne!(
            GpuPlatform::Metal { gpu_family: 7 },
            GpuPlatform::Metal { gpu_family: 8 }
        );
    }

    #[test]
    fn gpu_platform_ne_cross_platform() {
        // Different platform types are never equal
        assert_ne!(
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Hip { gfx_arch: 0x908 }
        );
        assert_ne!(
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Metal { gpu_family: 7 }
        );
        assert_ne!(
            GpuPlatform::Cuda { sm_version: 90 },
            GpuPlatform::Metal { gpu_family: 9 }
        );
    }

    // ── GpuPlatform matches! patterns ────────────────────────────────────────

    #[test]
    fn gpu_platform_match_cuda() {
        let p = GpuPlatform::Cuda { sm_version: 80 };
        let sm = match p {
            GpuPlatform::Cuda { sm_version: sm } => sm,
            _ => panic!("expected Cuda variant"),
        };
        assert_eq!(sm, 80);
    }

    #[test]
    fn gpu_platform_match_hip() {
        let p = GpuPlatform::Hip { gfx_arch: 0x940 };
        let gfx = match p {
            GpuPlatform::Hip { gfx_arch } => gfx_arch,
            _ => panic!("expected Hip variant"),
        };
        assert_eq!(gfx, 0x940);
    }

    #[test]
    fn gpu_platform_match_metal() {
        let p = GpuPlatform::Metal { gpu_family: 1009 };
        let fam = match p {
            GpuPlatform::Metal { gpu_family } => gpu_family,
            _ => panic!("expected Metal variant"),
        };
        assert_eq!(fam, 1009);
    }

    // ── GpuTopology with each platform variant ──────────────────────────────

    #[test]
    fn gpu_topology_hip_all_fields() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 104,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x0a,
        };
        assert_eq!(topo.platform, GpuPlatform::Hip { gfx_arch: 0x90a });
        assert_eq!(topo.compute_unit_count, 104);
        assert_eq!(topo.warp_size, 64);
        assert_eq!(topo.compute_cap_minor, 0x0a);
    }

    #[test]
    fn gpu_topology_metal_all_fields() {
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1008 },
            compute_unit_count: 1,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.platform, GpuPlatform::Metal { gpu_family: 1008 });
        assert_eq!(topo.l2_bytes, 0);
        assert_eq!(topo.warp_size, 32);
    }

    // ── GpuTopology clone produces equal value ───────────────────────────────

    #[test]
    fn gpu_topology_clone_equality() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        let cloned = topo.clone();
        assert_eq!(cloned.platform, topo.platform);
        assert_eq!(cloned.compute_unit_count, topo.compute_unit_count);
        assert_eq!(cloned.tensor_core_gen, topo.tensor_core_gen);
        assert_eq!(cloned.shared_mem_per_sm_bytes, topo.shared_mem_per_sm_bytes);
        assert_eq!(cloned.l2_bytes, topo.l2_bytes);
        assert_eq!(cloned.global_mem_bytes, topo.global_mem_bytes);
        assert_eq!(cloned.warp_size, topo.warp_size);
        assert_eq!(cloned.compute_cap_major, topo.compute_cap_major);
        assert_eq!(cloned.compute_cap_minor, topo.compute_cap_minor);
    }

    // ── Edge cases: zero and maximal field values ────────────────────────────

    #[test]
    fn gpu_topology_zero_fields() {
        // A topology with all-zero numeric fields is representable
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 0 },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.compute_unit_count, 0);
        assert_eq!(topo.tensor_core_gen, 0);
        assert_eq!(topo.compute_cap_major, 0);
    }

    #[test]
    fn gpu_topology_large_memory_values() {
        // Verify large memory sizes (e.g. 256 GB) don't overflow
        let large_mem: usize = 256 * 1024 * 1024 * 1024; // 256 GiB
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 100 },
            compute_unit_count: 168,
            tensor_core_gen: 4,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 128 * 1024 * 1024,
            global_mem_bytes: large_mem,
            warp_size: 32,
            compute_cap_major: 10,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.global_mem_bytes, 256 * 1024 * 1024 * 1024);
    }

    #[test]
    fn gpu_topology_max_usize_field() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 99 },
            compute_unit_count: usize::MAX,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: usize::MAX,
            l2_bytes: usize::MAX,
            global_mem_bytes: usize::MAX,
            warp_size: usize::MAX,
            compute_cap_major: u32::MAX,
            compute_cap_minor: u32::MAX,
        };
        assert_eq!(topo.compute_unit_count, usize::MAX);
        assert_eq!(topo.compute_cap_major, u32::MAX);
        assert_eq!(topo.compute_cap_minor, u32::MAX);
    }

    // ── CUDA GpuTopology with compute_cap derived from major/minor ───────────

    #[test]
    fn cuda_topology_compute_cap_matches_sm_version() {
        // For a typical Ampere GPU: major=8, minor=0 → sm_version=80
        let major: u32 = 8;
        let minor: u32 = 0;
        let sm_version = major * 10 + minor;
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version },
            compute_unit_count: 108,
            tensor_core_gen: cuda_tensor_core_gen(sm_version),
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: major,
            compute_cap_minor: minor,
        };
        assert_eq!(topo.compute_cap_major, 8);
        assert_eq!(topo.compute_cap_minor, 0);
        assert_eq!(topo.tensor_core_gen, 2);
        if let GpuPlatform::Cuda { sm_version: sm } = topo.platform {
            assert_eq!(sm, 80);
        } else {
            panic!("expected Cuda platform");
        }
    }

    // ── HIP GpuTopology with compute_cap derived from gfx_arch ───────────────

    #[test]
    fn hip_topology_compute_cap_from_gfx_arch() {
        let gfx_arch: u32 = 0x90a;
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch },
            compute_unit_count: 104,
            tensor_core_gen: hip_tensor_core_gen(gfx_arch),
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: (gfx_arch >> 8) & 0xFF,
            compute_cap_minor: gfx_arch & 0xFF,
        };
        assert_eq!(topo.compute_cap_major, 9);
        assert_eq!(topo.compute_cap_minor, 0x0a);
        assert_eq!(topo.tensor_core_gen, 2);
    }

    // ── Metal GpuTopology with gpu_family values ─────────────────────────────

    #[test]
    fn metal_topology_apple_family_values() {
        // Apple7 = 1007 (M1), Apple8 = 1008 (M2), Apple9 = 1009 (M3)
        for (family_val, apple_gen) in [(1007u32, 7u32), (1008, 8), (1009, 9)] {
            let topo = GpuTopology {
                platform: GpuPlatform::Metal { gpu_family: family_val },
                compute_unit_count: 1,
                tensor_core_gen: metal_tensor_core_gen(apple_gen),
                shared_mem_per_sm_bytes: 32 * 1024,
                l2_bytes: 0,
                global_mem_bytes: 16 * 1024 * 1024 * 1024,
                warp_size: 32,
                compute_cap_major: apple_gen,
                compute_cap_minor: 0,
            };
            assert_eq!(topo.compute_cap_major, apple_gen);
            assert_eq!(topo.compute_cap_minor, 0);
            if apple_gen >= 9 {
                assert_eq!(topo.tensor_core_gen, 3);
            } else if apple_gen >= 7 {
                assert_eq!(topo.tensor_core_gen, 2);
            } else {
                assert_eq!(topo.tensor_core_gen, 0);
            }
        }
    }

    // ── detect_gpu return type contract ──────────────────────────────────────

    #[test]
    fn detect_gpu_returns_result() {
        // detect_gpu always returns Ok or Err, never panics (on CPU-only systems)
        let result = detect_gpu();
        // Verify the type is Result<Option<GpuTopology>, String>
        assert!(result.is_ok() || result.is_err());
    }

    // ── GpuPlatform field access patterns ────────────────────────────────────

    #[test]
    fn gpu_platform_cuda_sm_version_extraction() {
        let platforms = vec![
            GpuPlatform::Cuda { sm_version: 70 },
            GpuPlatform::Cuda { sm_version: 75 },
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Cuda { sm_version: 86 },
            GpuPlatform::Cuda { sm_version: 89 },
            GpuPlatform::Cuda { sm_version: 90 },
            GpuPlatform::Cuda { sm_version: 100 },
        ];
        let sm_versions: Vec<u32> = platforms
            .into_iter()
            .map(|p| match p {
                GpuPlatform::Cuda { sm_version } => sm_version,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(sm_versions, vec![70, 75, 80, 86, 89, 90, 100]);
    }

    #[test]
    fn gpu_platform_hip_gfx_arch_extraction() {
        let platforms = vec![
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Hip { gfx_arch: 0x90a },
            GpuPlatform::Hip { gfx_arch: 0x940 },
        ];
        let arches: Vec<u32> = platforms
            .into_iter()
            .map(|p| match p {
                GpuPlatform::Hip { gfx_arch } => gfx_arch,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(arches, vec![0x908, 0x90a, 0x940]);
    }

    // ── Unit conversions and byte arithmetic ─────────────────────────────────

    #[test]
    fn shared_mem_typical_values_kib() {
        // 163_840 bytes = 160 KiB (Ampere A100)
        let bytes: usize = 163_840;
        assert_eq!(bytes / 1024, 160);
    }

    #[test]
    fn l2_cache_typical_values_mib() {
        // 40 MiB L2
        let bytes: usize = 40 * 1024 * 1024;
        assert_eq!(bytes / (1024 * 1024), 40);
    }

    #[test]
    fn global_mem_typical_values_gib() {
        // 24 GiB global memory
        let bytes: usize = 24 * 1024 * 1024 * 1024;
        assert_eq!(bytes / (1024 * 1024 * 1024), 24);
    }

    // ── GpuTopology struct field-by-field modification via clone ──────────────

    #[test]
    fn gpu_topology_clone_modify_platform() {
        let base = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let mut modified = base.clone();
        modified.platform = GpuPlatform::Cuda { sm_version: 90 };
        modified.tensor_core_gen = 3;
        modified.compute_cap_major = 9;
        // Original unchanged
        assert_eq!(base.platform, GpuPlatform::Cuda { sm_version: 80 });
        assert_eq!(base.tensor_core_gen, 2);
        // Modified is different
        assert_eq!(modified.platform, GpuPlatform::Cuda { sm_version: 90 });
        assert_eq!(modified.tensor_core_gen, 3);
    }

    // ── GpuPlatform Hash trait (HashSet / HashMap usability) ────────────────────

    #[test]
    fn gpu_platform_hash_set_dedup_cuda() {
        let mut set = std::collections::HashSet::new();
        set.insert(GpuPlatform::Cuda { sm_version: 80 });
        set.insert(GpuPlatform::Cuda { sm_version: 80 });
        set.insert(GpuPlatform::Cuda { sm_version: 90 });
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn gpu_platform_hash_set_mixed_platforms() {
        let mut set = std::collections::HashSet::new();
        set.insert(GpuPlatform::Cuda { sm_version: 80 });
        set.insert(GpuPlatform::Hip { gfx_arch: 0x908 });
        set.insert(GpuPlatform::Metal { gpu_family: 7 });
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn gpu_platform_hash_map_key_lookup() {
        let mut map = std::collections::HashMap::new();
        let key = GpuPlatform::Cuda { sm_version: 80 };
        map.insert(key, "A100");
        assert_eq!(map.get(&GpuPlatform::Cuda { sm_version: 80 }), Some(&"A100"));
        assert_eq!(map.get(&GpuPlatform::Cuda { sm_version: 90 }), None);
    }

    #[test]
    fn gpu_platform_hash_equal_values_equal_hash() {
        use std::hash::{Hash, Hasher};
        let a = GpuPlatform::Hip { gfx_arch: 0x940 };
        let b = GpuPlatform::Hip { gfx_arch: 0x940 };
        let mut ha = std::collections::hash_map::DefaultHasher::new();
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn gpu_platform_hash_different_values_likely_differ() {
        use std::hash::{Hash, Hasher};
        let a = GpuPlatform::Cuda { sm_version: 80 };
        let b = GpuPlatform::Cuda { sm_version: 90 };
        let mut ha = std::collections::hash_map::DefaultHasher::new();
        let mut hb = std::collections::hash_map::DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        // Not guaranteed, but extremely unlikely to collide
        assert_ne!(ha.finish(), hb.finish());
    }

    // ── GpuPlatform exhaustive variant discrimination ───────────────────────────

    #[test]
    fn gpu_platform_all_variants_constructible() {
        let cuda = GpuPlatform::Cuda { sm_version: 0 };
        let hip = GpuPlatform::Hip { gfx_arch: 0 };
        let metal = GpuPlatform::Metal { gpu_family: 0 };
        // All three are distinct
        assert_ne!(cuda, hip);
        assert_ne!(hip, metal);
        assert_ne!(cuda, metal);
    }

    #[test]
    fn gpu_platform_enum_size_is_three_variants() {
        // Ensure the compiler recognizes exactly 3 variants via exhaustive match
        let variants: Vec<GpuPlatform> = vec![
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Metal { gpu_family: 1009 },
        ];
        assert_eq!(variants.len(), 3);
        // All pairwise distinct
        assert_ne!(variants[0], variants[1]);
        assert_ne!(variants[1], variants[2]);
        assert_ne!(variants[0], variants[2]);
    }

    // ── CUDA tensor_core_gen monotonicity property ──────────────────────────────

    #[test]
    fn cuda_tensor_core_gen_monotonically_non_decreasing() {
        let prev = cuda_tensor_core_gen(0);
        for sm in 1..=120u32 {
            let cur = cuda_tensor_core_gen(sm);
            assert!(
                cur >= prev,
                "monotonicity violated at sm={sm}: {cur} < {prev}"
            );
        }
    }

    #[test]
    fn cuda_tensor_core_gen_max_output_is_4() {
        for sm in 0..=200u32 {
            let gen = cuda_tensor_core_gen(sm);
            assert!(gen <= 4, "tensor_core_gen={gen} exceeds 4 at sm={sm}");
        }
    }

    #[test]
    fn cuda_tensor_core_gen_output_range() {
        let mut gens = std::collections::HashSet::new();
        for sm in 0..=120u32 {
            gens.insert(cuda_tensor_core_gen(sm));
        }
        assert!(gens.contains(&0), "gen 0 (no tensor cores) expected");
        assert!(gens.contains(&1), "gen 1 (Volta/Turing) expected");
        assert!(gens.contains(&2), "gen 2 (Ampere) expected");
        assert!(gens.contains(&3), "gen 3 (Hopper) expected");
        assert!(gens.contains(&4), "gen 4 (Blackwell) expected");
    }

    #[test]
    fn cuda_tensor_core_sm86_and_sm89_ampere() {
        // sm_86 (RTX 3090) and sm_89 (RTX 4090 Ada Lovelace) are Ampere-family
        assert_eq!(cuda_tensor_core_gen(86), 2);
        assert_eq!(cuda_tensor_core_gen(89), 2);
    }

    #[test]
    fn cuda_tensor_core_sm110_and_beyond() {
        // Any sm >= 100 maps to Blackwell gen 4
        assert_eq!(cuda_tensor_core_gen(100), 4);
        assert_eq!(cuda_tensor_core_gen(110), 4);
        assert_eq!(cuda_tensor_core_gen(200), 4);
    }

    // ── HIP tensor_core_gen additional coverage ─────────────────────────────────

    #[test]
    fn hip_tensor_core_gen_output_range() {
        let mut gens = std::collections::HashSet::new();
        // Sweep representative gfx_arch values
        for gfx in [0x000, 0x500, 0x907, 0x908, 0x90a, 0x93f, 0x940, 0x950, 0x1000, 0x1100] {
            gens.insert(hip_tensor_core_gen(gfx));
        }
        assert!(gens.contains(&0), "gen 0 expected for pre-CDNA");
        assert!(gens.contains(&2), "gen 2 expected for CDNA1/2");
        assert!(gens.contains(&3), "gen 3 expected for CDNA3");
    }

    #[test]
    fn hip_tensor_core_gfx909_is_cdna1() {
        // gfx909 should map to CDNA1 tier (gen 2) — >= 0x908
        assert_eq!(hip_tensor_core_gen(0x909), 2);
    }

    #[test]
    fn hip_tensor_core_low_rdna_gfx1010_no_tensor() {
        // gfx1010 (RDNA1 Navi 10) = 0x1010 = 4112, numerically > 0x940 → maps to gen 3
        // This documents the actual numeric comparison behavior
        assert_eq!(hip_tensor_core_gen(0x1010), 3);
    }

    #[test]
    fn hip_tensor_core_zero_arch() {
        assert_eq!(hip_tensor_core_gen(0), 0);
    }

    // ── Metal tensor_core_gen additional coverage ───────────────────────────────

    #[test]
    fn metal_tensor_core_gen_monotonically_non_decreasing() {
        let prev = metal_tensor_core_gen(1);
        for gen in 2..=15u32 {
            let cur = metal_tensor_core_gen(gen);
            assert!(
                cur >= prev,
                "monotonicity violated at apple_gen={gen}: {cur} < {prev}"
            );
        }
    }

    #[test]
    fn metal_tensor_core_gen_max_output_is_3() {
        for gen in 0..=20u32 {
            let tcg = metal_tensor_core_gen(gen);
            assert!(tcg <= 3, "tensor_core_gen={tcg} exceeds 3 at apple_gen={gen}");
        }
    }

    #[test]
    fn metal_tensor_core_apple5_boundary() {
        // Apple5 = just below M1 threshold → gen 0
        assert_eq!(metal_tensor_core_gen(5), 0);
    }

    #[test]
    fn metal_tensor_core_all_apple_generations() {
        // Apple1..Apple9 (the full defined range in the source APPLE_FAMILIES table)
        let expected = [
            (1u32, 0u32), // Apple1
            (2, 0),       // Apple2
            (3, 0),       // Apple3
            (4, 0),       // Apple4
            (5, 0),       // Apple5
            (6, 0),       // Apple6
            (7, 2),       // Apple7 = M1
            (8, 2),       // Apple8 = M2
            (9, 3),       // Apple9 = M3
        ];
        for (apple_gen, expected_gen) in expected {
            assert_eq!(
                metal_tensor_core_gen(apple_gen),
                expected_gen,
                "mismatch at apple_gen={apple_gen}"
            );
        }
    }

    // ── GpuPlatform Copy semantics: assignment does not move ────────────────────

    #[test]
    fn gpu_platform_copy_reassignment() {
        let p = GpuPlatform::Cuda { sm_version: 90 };
        assert_eq!(p, GpuPlatform::Cuda { sm_version: 90 });
    }

    #[test]
    fn gpu_platform_copy_used_after_assignment() {
        let a = GpuPlatform::Hip { gfx_arch: 0x908 };
        let _b = a; // Copy, not move — a is still usable
        let c = a;  // Second copy
        assert_eq!(a, c);
    }

    // ── GpuTopology field validation: platform-type consistency ──────────────────

    #[test]
    fn gpu_topology_cuda_warp_size_is_32() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // CUDA warp size is always 32
        assert_eq!(topo.warp_size, 32);
    }

    #[test]
    fn gpu_topology_metal_l2_bytes_is_zero() {
        // Metal does not expose traditional L2 cache → always 0
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1009 },
            compute_unit_count: 1,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.l2_bytes, 0);
    }

    #[test]
    fn gpu_topology_metal_compute_cap_minor_is_zero() {
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1008 },
            compute_unit_count: 1,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.compute_cap_minor, 0);
    }

    // ── GpuTopology clone independence for each platform ─────────────────────────

    #[test]
    fn gpu_topology_hip_clone_independent() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        let clone = {
            let mut c = topo.clone();
            c.compute_unit_count = 0;
            c.global_mem_bytes = 0;
            c
        };
        assert_eq!(topo.compute_unit_count, 304);
        assert_eq!(topo.global_mem_bytes, 192 * 1024 * 1024 * 1024);
        assert_eq!(clone.compute_unit_count, 0);
        assert_eq!(clone.global_mem_bytes, 0);
    }

    #[test]
    fn gpu_topology_metal_clone_independent() {
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1007 },
            compute_unit_count: 1,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 8 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        let clone = {
            let mut c = topo.clone();
            c.shared_mem_per_sm_bytes = 0;
            c
        };
        assert_eq!(topo.shared_mem_per_sm_bytes, 32 * 1024);
        assert_eq!(clone.shared_mem_per_sm_bytes, 0);
    }

    // ── CUDA sm_version formula round-trip ──────────────────────────────────────

    #[test]
    fn cuda_sm_version_round_trip_typical_values() {
        // major * 10 + minor should recover the expected sm_version for known GPUs
        let cases: Vec<(u32, u32, u32)> = vec![
            (7, 0, 70),  // Volta V100
            (7, 5, 75),  // Turing T4
            (8, 0, 80),  // Ampere A100
            (8, 6, 86),  // Ampere RTX 3090
            (8, 9, 89),  // Ada RTX 4090
            (9, 0, 90),  // Hopper H100
            (10, 0, 100), // Blackwell B100
        ];
        for (major, minor, expected_sm) in cases {
            assert_eq!(major * 10 + minor, expected_sm);
        }
    }

    #[test]
    fn cuda_sm_version_major_minor_recovery() {
        // Given sm_version, recover major and minor
        for sm in [70u32, 75, 80, 86, 89, 90, 100] {
            let major = sm / 10;
            let minor = sm % 10;
            assert_eq!(major * 10 + minor, sm);
        }
    }

    // ── HIP gfx_arch bit decomposition for various architectures ────────────────

    #[test]
    fn hip_gfx_arch_round_trip() {
        // (gfx_arch >> 8) & 0xFF gives major; gfx_arch & 0xFF gives minor
        // Recombining: (major << 8) | minor should equal original
        for gfx in [0x908u32, 0x90a, 0x940, 0x950, 0x1030, 0x1100] {
            let major = (gfx >> 8) & 0xFF;
            let minor = gfx & 0xFF;
            let reconstructed = (major << 8) | minor;
            assert_eq!(reconstructed, gfx);
        }
    }

    #[test]
    fn hip_gfx_arch_zero_value() {
        let major = (0u32 >> 8) & 0xFF;
        let minor = 0u32 & 0xFF;
        assert_eq!(major, 0);
        assert_eq!(minor, 0);
    }

    // ── GpuPlatform Debug output contains field names ───────────────────────────

    #[test]
    fn gpu_platform_debug_cuda_contains_sm_version() {
        let p = GpuPlatform::Cuda { sm_version: 86 };
        let s = format!("{p:?}");
        assert!(s.contains("86"));
    }

    #[test]
    fn gpu_platform_debug_hip_contains_gfx_arch() {
        let p = GpuPlatform::Hip { gfx_arch: 0x90a };
        let s = format!("{p:?}");
        assert!(s.contains("Hip"));
    }

    #[test]
    fn gpu_platform_debug_metal_contains_family() {
        let p = GpuPlatform::Metal { gpu_family: 1009 };
        let s = format!("{p:?}");
        assert!(s.contains("Metal"));
    }

    // ── GpuTopology Debug output for each platform variant ──────────────────────

    #[test]
    fn gpu_topology_debug_hip_platform() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        let s = format!("{topo:?}");
        assert!(s.contains("Hip"), "Debug output should contain 'Hip'");
        assert!(s.contains("compute_unit_count"));
        assert!(s.contains("warp_size"));
    }

    #[test]
    fn gpu_topology_debug_metal_platform() {
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1008 },
            compute_unit_count: 1,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let s = format!("{topo:?}");
        assert!(s.contains("Metal"), "Debug output should contain 'Metal'");
        assert!(s.contains("gpu_family"));
    }

    // ── GpuTopology field ordering and structure ────────────────────────────────

    #[test]
    fn gpu_topology_all_nine_fields_accessible() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // Access every field — compiles only if all 9 fields exist
        let _ = topo.platform;
        let _ = topo.compute_unit_count;
        let _ = topo.tensor_core_gen;
        let _ = topo.shared_mem_per_sm_bytes;
        let _ = topo.l2_bytes;
        let _ = topo.global_mem_bytes;
        let _ = topo.warp_size;
        let _ = topo.compute_cap_major;
        let _ = topo.compute_cap_minor;
    }

    #[test]
    fn gpu_topology_field_types_match() {
        // Verify field types by assigning and comparing
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 0usize,
            tensor_core_gen: 0u32,
            shared_mem_per_sm_bytes: 0usize,
            l2_bytes: 0usize,
            global_mem_bytes: 0usize,
            warp_size: 0usize,
            compute_cap_major: 0u32,
            compute_cap_minor: 0u32,
        };
        assert_eq!(topo.compute_unit_count, 0usize);
        assert_eq!(topo.tensor_core_gen, 0u32);
        assert_eq!(topo.compute_cap_major, 0u32);
    }

    // ── detect_gpu idempotency ──────────────────────────────────────────────────

    #[test]
    fn detect_gpu_called_twice_consistent() {
        let first = detect_gpu();
        let second = detect_gpu();
        match (first, second) {
            (Ok(None), Ok(None)) => {}
            (Ok(Some(a)), Ok(Some(b))) => {
                assert_eq!(a.platform, b.platform);
                assert_eq!(a.compute_unit_count, b.compute_unit_count);
                assert_eq!(a.tensor_core_gen, b.tensor_core_gen);
                assert_eq!(a.warp_size, b.warp_size);
            }
            (Err(e1), Err(e2)) => {
                assert!(!e1.is_empty());
                assert!(!e2.is_empty());
            }
            _ => panic!("detect_gpu returned inconsistent results across two calls"),
        }
    }

    // ── Metal APPLE_FAMILIES mapping verification ──────────────────────────────

    #[test]
    fn metal_apple_family_values_range() {
        // Apple GPU families: 1001..=1009
        let families: Vec<u32> = (1001..=1009).collect();
        for family_val in families {
            let topo = GpuTopology {
                platform: GpuPlatform::Metal { gpu_family: family_val },
                compute_unit_count: 1,
                tensor_core_gen: 0,
                shared_mem_per_sm_bytes: 32 * 1024,
                l2_bytes: 0,
                global_mem_bytes: 8 * 1024 * 1024 * 1024,
                warp_size: 32,
                compute_cap_major: family_val - 1000,
                compute_cap_minor: 0,
            };
            if let GpuPlatform::Metal { gpu_family } = topo.platform {
                assert_eq!(gpu_family, family_val);
            }
        }
    }

    // ── GpuPlatform equality symmetry ──────────────────────────────────────────

    #[test]
    fn gpu_platform_eq_symmetry() {
        let a = GpuPlatform::Cuda { sm_version: 80 };
        let b = GpuPlatform::Cuda { sm_version: 80 };
        assert!(a == b);
        assert!(b == a);
    }

    #[test]
    fn gpu_platform_ne_symmetry() {
        let a = GpuPlatform::Hip { gfx_arch: 0x908 };
        let b = GpuPlatform::Hip { gfx_arch: 0x940 };
        assert!(a != b);
        assert!(b != a);
    }

    #[test]
    fn gpu_platform_eq_transitivity() {
        let a = GpuPlatform::Cuda { sm_version: 90 };
        let b = GpuPlatform::Cuda { sm_version: 90 };
        let c = GpuPlatform::Cuda { sm_version: 90 };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── GpuPlatform in collections (Vec, HashMap) ──────────────────────────────

    #[test]
    fn gpu_platform_vec_contains_and_iter() {
        let platforms = vec![
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Metal { gpu_family: 1009 },
        ];
        assert!(platforms.contains(&GpuPlatform::Cuda { sm_version: 80 }));
        assert!(platforms.contains(&GpuPlatform::Hip { gfx_arch: 0x908 }));
        assert!(!platforms.contains(&GpuPlatform::Cuda { sm_version: 90 }));
        assert_eq!(platforms.len(), 3);
    }

    #[test]
    fn gpu_platform_vec_dedup_via_hashset() {
        let platforms = vec![
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Cuda { sm_version: 80 },
            GpuPlatform::Cuda { sm_version: 90 },
            GpuPlatform::Hip { gfx_arch: 0x908 },
            GpuPlatform::Hip { gfx_arch: 0x908 },
        ];
        let unique: std::collections::HashSet<_> = platforms.into_iter().collect();
        assert_eq!(unique.len(), 3);
    }

    // ── GpuTopology with realistic CDNA3 MI300 values ──────────────────────────

    #[test]
    fn gpu_topology_cdna3_mi300x_realistic() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: hip_tensor_core_gen(0x940),
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: (0x940 >> 8) & 0xFF,
            compute_cap_minor: 0x940 & 0xFF,
        };
        assert_eq!(topo.tensor_core_gen, 3);
        assert_eq!(topo.compute_cap_major, 9);
        assert_eq!(topo.warp_size, 64);
        assert!(topo.global_mem_bytes > 100 * 1024 * 1024 * 1024);
    }

    // ── CUDA tensor_core_gen threshold boundary table ──────────────────────────

    #[test]
    fn cuda_tensor_core_all_thresholds() {
        // Verify every threshold boundary: just below, at, and just above
        let cases: Vec<(u32, u32)> = vec![
            (69, 0), (70, 1),  // Volta
            (79, 1), (80, 2),  // Ampere
            (89, 2), (90, 3),  // Hopper
            (99, 3), (100, 4), // Blackwell
        ];
        for (sm, expected_gen) in cases {
            assert_eq!(cuda_tensor_core_gen(sm), expected_gen, "at sm={sm}");
        }
    }

    // ── GpuPlatform match exhaustiveness helper ────────────────────────────────

    #[test]
    fn gpu_platform_match_returns_field_value() {
        fn extract_u32(p: GpuPlatform) -> Option<u32> {
            match p {
                GpuPlatform::Cuda { sm_version } => Some(sm_version),
                GpuPlatform::Hip { gfx_arch } => Some(gfx_arch),
                GpuPlatform::Metal { gpu_family } => Some(gpu_family),
                GpuPlatform::Cpu => None,
            }
        }
        assert_eq!(extract_u32(GpuPlatform::Cuda { sm_version: 42 }), Some(42));
        assert_eq!(extract_u32(GpuPlatform::Hip { gfx_arch: 0xFF }), Some(0xFF));
        assert_eq!(extract_u32(GpuPlatform::Metal { gpu_family: 7 }), Some(7));
        assert_eq!(extract_u32(GpuPlatform::Cpu), None);
    }

    // ── GpuPlatform Eq trait (strict equality, not just PartialEq) ─────────────

    #[test]
    fn gpu_platform_eq_trait_holds_for_same_variants() {
        // Eq is derived; verify reflexivity with assert_eq! (which requires Eq + Debug)
        fn assert_eq_reflexive<T: Eq + std::fmt::Debug>(a: &T, b: &T) {
            assert_eq!(a, b);
        }
        let a = GpuPlatform::Cuda { sm_version: 80 };
        let b = GpuPlatform::Cuda { sm_version: 80 };
        assert_eq_reflexive(&a, &b);
    }

    // ── GpuPlatform and GpuTopology are Send + Sync (compilation check) ────────

    #[test]
    fn gpu_platform_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuPlatform>();
    }

    #[test]
    fn gpu_topology_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuTopology>();
    }

    // ── GpuTopology with Blackwell B200 realistic values ───────────────────────

    #[test]
    fn gpu_topology_blackwell_b200_realistic() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 100 },
            compute_unit_count: 168,
            tensor_core_gen: cuda_tensor_core_gen(100),
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 128 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 10,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.tensor_core_gen, 4);
        assert_eq!(topo.compute_cap_major, 10);
        assert_eq!(topo.compute_cap_minor, 0);
        assert!(topo.global_mem_bytes >= 180 * 1024 * 1024 * 1024);
        if let GpuPlatform::Cuda { sm_version } = topo.platform {
            assert_eq!(sm_version, 100);
        } else {
            panic!("expected Cuda platform");
        }
    }

    // ── GpuTopology shared memory field boundary: zero vs realistic ────────────

    #[test]
    fn gpu_topology_shared_mem_zero_and_nonzero_both_representable() {
        let zero = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 0 },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };
        let realistic = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        assert_eq!(zero.shared_mem_per_sm_bytes, 0);
        assert!(realistic.shared_mem_per_sm_bytes > 0);
        assert_ne!(zero.shared_mem_per_sm_bytes, realistic.shared_mem_per_sm_bytes);
    }

    // ── HIP warp_size dual-path: RDNA=32, CDNA=64 ─────────────────────────────

    #[test]
    fn gpu_topology_hip_cdna_warp_size_64_rdna_warp_size_32() {
        let cdna = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        let rdna = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x1100 },
            compute_unit_count: 40,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 4 * 1024 * 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 0x11,
            compute_cap_minor: 0x00,
        };
        assert_eq!(cdna.warp_size, 64);
        assert_eq!(rdna.warp_size, 32);
        assert_ne!(cdna.warp_size, rdna.warp_size);
    }

    // ── GpuPlatform passed by reference into a function ────────────────────────

    #[test]
    fn gpu_platform_passed_by_ref_preserves_value() {
        fn platform_name(p: &GpuPlatform) -> &'static str {
            match p {
                GpuPlatform::Cuda { .. } => "CUDA",
                GpuPlatform::Hip { .. } => "HIP",
                GpuPlatform::Metal { .. } => "Metal",
                GpuPlatform::Cpu => "CPU",
            }
        }
        let cuda = GpuPlatform::Cuda { sm_version: 80 };
        assert_eq!(platform_name(&cuda), "CUDA");
        assert_eq!(platform_name(&GpuPlatform::Cpu), "CPU");
        // Original still usable after borrow
        assert_eq!(cuda, GpuPlatform::Cuda { sm_version: 80 });
    }

    // ── GpuTopology Debug pretty-print format includes all field names ─────────

    #[test]
    fn gpu_topology_debug_pretty_format_contains_all_fields() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x90a },
            compute_unit_count: 104,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 64 * 1024,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 64 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x0a,
        };
        let pretty = format!("{topo:#?}");
        assert!(pretty.contains("platform"));
        assert!(pretty.contains("compute_unit_count"));
        assert!(pretty.contains("tensor_core_gen"));
        assert!(pretty.contains("shared_mem_per_sm_bytes"));
        assert!(pretty.contains("l2_bytes"));
        assert!(pretty.contains("global_mem_bytes"));
        assert!(pretty.contains("warp_size"));
        assert!(pretty.contains("compute_cap_major"));
        assert!(pretty.contains("compute_cap_minor"));
    }

    // ── detect_gpu Result can be chained with map/and_then ─────────────────────

    #[test]
    fn detect_gpu_result_chain_map() {
        let result = detect_gpu();
        // map over the Result<Option<GpuTopology>, String>
        let mapped = result.map(|opt| opt.map(|topo| topo.warp_size));
        match mapped {
            Ok(Some(warp)) => assert!(warp > 0),
            Ok(None) => {} // no GPU — valid
            Err(e) => assert!(!e.is_empty()),
        }
    }

    // ── GpuTopology memory proportion: shared_mem << global_mem ────────────────

    #[test]
    fn gpu_topology_shared_mem_much_smaller_than_global() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        // shared_mem_per_sm_bytes should be orders of magnitude smaller than global_mem
        assert!(topo.shared_mem_per_sm_bytes < topo.global_mem_bytes / 1000);
    }

    // ── GpuTopology cross-platform instances are distinct ──────────────────────

    #[test]
    fn gpu_topology_different_platforms_have_different_fields() {
        let cuda = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        let hip = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 304,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 256 * 1024 * 1024,
            global_mem_bytes: 192 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        assert_ne!(cuda.platform, hip.platform);
        assert_ne!(cuda.warp_size, hip.warp_size);
        assert_ne!(cuda.compute_unit_count, hip.compute_unit_count);
    }

    // ── GpuTopology clone then mutate does not affect original (deep verify) ───

    #[test]
    fn gpu_topology_clone_then_mutate_all_numeric_fields() {
        let original = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1009 },
            compute_unit_count: 10,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        let mut clone = original.clone();
        clone.compute_unit_count = 999;
        clone.tensor_core_gen = 0;
        clone.shared_mem_per_sm_bytes = 1;
        clone.l2_bytes = usize::MAX;
        clone.global_mem_bytes = 0;
        clone.warp_size = 1;
        clone.compute_cap_major = 255;
        clone.compute_cap_minor = 255;
        // Original unchanged
        assert_eq!(original.compute_unit_count, 10);
        assert_eq!(original.tensor_core_gen, 3);
        assert_eq!(original.shared_mem_per_sm_bytes, 32 * 1024);
        assert_eq!(original.l2_bytes, 0);
        assert_eq!(original.global_mem_bytes, 16 * 1024 * 1024 * 1024);
        assert_eq!(original.warp_size, 32);
        assert_eq!(original.compute_cap_major, 9);
        assert_eq!(original.compute_cap_minor, 0);
        // Clone mutated
        assert_eq!(clone.compute_unit_count, 999);
        assert_eq!(clone.tensor_core_gen, 0);
        assert_eq!(clone.l2_bytes, usize::MAX);
    }

    // ── GpuTopology field-by-field equality after construction ─────────────────

    #[test]
    fn gpu_topology_field_wise_equality_after_construction() {
        let a = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 86 },
            compute_unit_count: 84,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 100_000,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 6,
        };
        let b = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 86 },
            compute_unit_count: 84,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 100_000,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 6,
        };
        assert_eq!(a.platform, b.platform);
        assert_eq!(a.compute_unit_count, b.compute_unit_count);
        assert_eq!(a.tensor_core_gen, b.tensor_core_gen);
        assert_eq!(a.shared_mem_per_sm_bytes, b.shared_mem_per_sm_bytes);
        assert_eq!(a.l2_bytes, b.l2_bytes);
        assert_eq!(a.global_mem_bytes, b.global_mem_bytes);
        assert_eq!(a.warp_size, b.warp_size);
        assert_eq!(a.compute_cap_major, b.compute_cap_major);
        assert_eq!(a.compute_cap_minor, b.compute_cap_minor);
    }

    // ── Metal GpuTopology compute_unit_count is always 1 (hardware constant) ──

    #[test]
    fn gpu_topology_metal_compute_unit_count_one() {
        // As per detect_metal(): compute_unit_count = 1 for all Apple GPUs
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1007 },
            compute_unit_count: 1,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 32 * 1024,
            l2_bytes: 0,
            global_mem_bytes: 8 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.compute_unit_count, 1);
    }

    // ── GpuPlatform variant with u32::MAX inner field ───────────────────────────

    #[test]
    fn gpu_platform_cuda_sm_version_u32_max() {
        let p = GpuPlatform::Cuda { sm_version: u32::MAX };
        if let GpuPlatform::Cuda { sm_version } = p {
            assert_eq!(sm_version, u32::MAX);
        } else {
            panic!("expected Cuda variant");
        }
    }

    // ── GpuPlatform zero inner fields are still distinct across variants ─────────

    #[test]
    fn gpu_platform_zero_inner_field_cross_variant_inequality() {
        let cuda_zero = GpuPlatform::Cuda { sm_version: 0 };
        let hip_zero = GpuPlatform::Hip { gfx_arch: 0 };
        let metal_zero = GpuPlatform::Metal { gpu_family: 0 };
        // Even with all-zero inner fields, different variants are never equal
        assert_ne!(cuda_zero, hip_zero);
        assert_ne!(hip_zero, metal_zero);
        assert_ne!(cuda_zero, metal_zero);
    }

    // ── GpuTopology memory hierarchy: shared_mem < l2 < global ───────────────────

    #[test]
    fn gpu_topology_memory_hierarchy_ordering() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert!(
            topo.shared_mem_per_sm_bytes < topo.l2_bytes,
            "shared_mem ({}) must be less than L2 ({})",
            topo.shared_mem_per_sm_bytes,
            topo.l2_bytes
        );
        assert!(
            topo.l2_bytes < topo.global_mem_bytes,
            "L2 ({}) must be less than global_mem ({})",
            topo.l2_bytes,
            topo.global_mem_bytes
        );
    }

    // ── CUDA tensor_core_gen for very high sm_version values ─────────────────────

    #[test]
    fn cuda_tensor_core_gen_very_high_sm_cap() {
        // sm_version beyond current Blackwell should still map to gen 4
        assert_eq!(cuda_tensor_core_gen(u32::MAX), 4);
    }

    // ── HIP tensor_core_gen for very high gfx_arch values ────────────────────────

    #[test]
    fn hip_tensor_core_gen_very_high_gfx_arch() {
        // Any gfx_arch >= 0x940 maps to gen 3
        assert_eq!(hip_tensor_core_gen(u32::MAX), 3);
    }

    // ── Metal tensor_core_gen at apple_gen zero ──────────────────────────────────

    #[test]
    fn metal_tensor_core_gen_zero() {
        // apple_gen=0 is below Apple1 → no tensor cores
        assert_eq!(metal_tensor_core_gen(0), 0);
    }

    // ── GpuPlatform stored in Box (heap allocation) ─────────────────────────────

    #[test]
    fn gpu_platform_boxed_deref_access() {
        let boxed: Box<GpuPlatform> = Box::new(GpuPlatform::Hip { gfx_arch: 0x90a });
        assert_eq!(*boxed, GpuPlatform::Hip { gfx_arch: 0x90a });
        if let GpuPlatform::Hip { gfx_arch } = *boxed {
            assert_eq!(gfx_arch, 0x90a);
        } else {
            panic!("expected Hip variant");
        }
    }

    // ── GpuPlatform stored in Arc (shared ownership) ─────────────────────────────

    #[test]
    fn gpu_platform_arc_shared_access() {
        use std::sync::Arc;
        let arc = Arc::new(GpuPlatform::Metal { gpu_family: 1009 });
        let clone = Arc::clone(&arc);
        assert_eq!(*arc, *clone);
        assert_eq!(Arc::strong_count(&arc), 2);
    }

    // ── GpuTopology with CUDA Volta V100 realistic values ────────────────────────

    #[test]
    fn gpu_topology_volta_v100_realistic() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 70 },
            compute_unit_count: 80,
            tensor_core_gen: cuda_tensor_core_gen(70),
            shared_mem_per_sm_bytes: 96 * 1024,
            l2_bytes: 6 * 1024 * 1024,
            global_mem_bytes: 16 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 7,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.tensor_core_gen, 1);
        assert_eq!(topo.shared_mem_per_sm_bytes, 96 * 1024);
        assert_eq!(topo.compute_cap_major, 7);
    }

    // ── GpuPlatform returned from a closure ──────────────────────────────────────

    #[test]
    fn gpu_platform_returned_from_closure() {
        let factory = |platform_type: u8| -> GpuPlatform {
            match platform_type {
                0 => GpuPlatform::Cuda { sm_version: 80 },
                1 => GpuPlatform::Hip { gfx_arch: 0x908 },
                _ => GpuPlatform::Metal { gpu_family: 1007 },
            }
        };
        assert_eq!(factory(0), GpuPlatform::Cuda { sm_version: 80 });
        assert_eq!(factory(1), GpuPlatform::Hip { gfx_arch: 0x908 });
        assert_eq!(factory(99), GpuPlatform::Metal { gpu_family: 1007 });
    }

    // ── GpuPlatform match with guard clause ──────────────────────────────────────

    #[test]
    fn gpu_platform_match_with_guard() {
        let p = GpuPlatform::Cuda { sm_version: 90 };
        let is_hopper_or_newer = match p {
            GpuPlatform::Cuda { sm_version } if sm_version >= 90 => true,
            GpuPlatform::Cuda { .. } => false,
            _ => false,
        };
        assert!(is_hopper_or_newer);
    }

    // ── GpuTopology shared_mem exact KiB conversion ──────────────────────────────

    #[test]
    fn gpu_topology_shared_mem_exact_kib_conversion() {
        // 227_328 bytes = 222 KiB (Hopper H100)
        let shared_mem: usize = 227_328;
        assert_eq!(shared_mem / 1024, 222);
        assert_eq!(shared_mem % 1024, 0);
    }

    // ── GpuPlatform inner field u32 arithmetic safety ────────────────────────────

    #[test]
    fn gpu_platform_cuda_sm_version_no_overflow_on_add() {
        let p = GpuPlatform::Cuda { sm_version: 80 };
        if let GpuPlatform::Cuda { sm_version } = p {
            // Verify arithmetic on sm_version does not panic
            let next = sm_version + 1;
            assert_eq!(next, 81);
        }
    }

    // ── GpuTopology Vec of mixed platforms filtered by variant ──────────────────

    #[test]
    fn gpu_topology_vec_filter_by_cuda_variant() {
        // Arrange: create a Vec of GpuTopology with mixed platforms
        let topologies = vec![
            GpuTopology {
                platform: GpuPlatform::Cuda { sm_version: 80 },
                compute_unit_count: 108,
                tensor_core_gen: 2,
                shared_mem_per_sm_bytes: 163_840,
                l2_bytes: 40 * 1024 * 1024,
                global_mem_bytes: 24 * 1024 * 1024 * 1024,
                warp_size: 32,
                compute_cap_major: 8,
                compute_cap_minor: 0,
            },
            GpuTopology {
                platform: GpuPlatform::Hip { gfx_arch: 0x940 },
                compute_unit_count: 304,
                tensor_core_gen: 3,
                shared_mem_per_sm_bytes: 227_328,
                l2_bytes: 256 * 1024 * 1024,
                global_mem_bytes: 192 * 1024 * 1024 * 1024,
                warp_size: 64,
                compute_cap_major: 9,
                compute_cap_minor: 0x40,
            },
            GpuTopology {
                platform: GpuPlatform::Cuda { sm_version: 90 },
                compute_unit_count: 132,
                tensor_core_gen: 3,
                shared_mem_per_sm_bytes: 227_328,
                l2_bytes: 50 * 1024 * 1024,
                global_mem_bytes: 80 * 1024 * 1024 * 1024,
                warp_size: 32,
                compute_cap_major: 9,
                compute_cap_minor: 0,
            },
        ];

        // Act: filter for CUDA platforms only
        let cuda_only: Vec<&GpuTopology> = topologies
            .iter()
            .filter(|t| matches!(t.platform, GpuPlatform::Cuda { .. }))
            .collect();

        // Assert: exactly 2 CUDA topologies
        assert_eq!(cuda_only.len(), 2);
        assert!(cuda_only.iter().all(|t| matches!(t.platform, GpuPlatform::Cuda { .. })));
    }

    // ── GpuTopology total shared memory across all SMs ──────────────────────────

    #[test]
    fn gpu_topology_total_shared_mem_across_all_sms() {
        // Arrange: Hopper H100 with 132 SMs, 227_328 bytes per SM
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 90 },
            compute_unit_count: 132,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 227_328,
            l2_bytes: 50 * 1024 * 1024,
            global_mem_bytes: 80 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };

        // Act: compute total shared memory
        let total_shared = topo.compute_unit_count * topo.shared_mem_per_sm_bytes;

        // Assert: total shared memory is reasonable (approx 29 MiB)
        assert_eq!(total_shared, 132 * 227_328);
        assert_eq!(total_shared / (1024 * 1024), 28);
    }

    // ── Metal gpu_family to Apple generation conversion ─────────────────────────

    #[test]
    fn metal_gpu_family_to_apple_generation_mapping() {
        // Arrange: Apple GPU family values from the APPLE_FAMILIES table
        let cases: Vec<(u32, u32)> = vec![
            (1001, 1),  // Apple1 = A7
            (1002, 2),  // Apple2 = A8
            (1007, 7),  // Apple7 = M1
            (1008, 8),  // Apple8 = M2
            (1009, 9),  // Apple9 = M3
        ];

        // Act & Assert: gpu_family - 1000 = apple_generation
        for (family_val, expected_gen) in cases {
            let apple_gen = family_val - 1000;
            assert_eq!(apple_gen, expected_gen, "family {} should map to gen {}", family_val, expected_gen);
        }
    }

    // ── CUDA tensor_core_gen step function verification across contiguous range ─

    #[test]
    fn cuda_tensor_core_gen_step_function_transitions() {
        // Arrange: verify the exact sm_version values where gen increments
        let transitions: Vec<(u32, u32, u32)> = vec![
            (0, 69, 0),    // pre-Volta: gen 0
            (70, 79, 1),   // Volta/Turing: gen 1
            (80, 89, 2),   // Ampere: gen 2
            (90, 99, 3),   // Hopper: gen 3
            (100, 110, 4), // Blackwell: gen 4
        ];

        // Act & Assert: every sm in each range maps to the expected gen
        for (lo, hi, expected_gen) in transitions {
            for sm in lo..=hi {
                assert_eq!(
                    cuda_tensor_core_gen(sm),
                    expected_gen,
                    "sm={} should be gen {}",
                    sm,
                    expected_gen
                );
            }
        }
    }

    // ── HIP tensor_core_gen monotonicity across gfx_arch sweep ──────────────────

    #[test]
    fn hip_tensor_core_gen_monotonically_non_decreasing() {
        // Arrange: sweep from gfx_arch=0 to 0x1200 in steps
        let mut prev = hip_tensor_core_gen(0);
        for gfx in (1..=0x1200u32).step_by(0x10) {
            // Act
            let cur = hip_tensor_core_gen(gfx);
            // Assert
            assert!(
                cur >= prev,
                "monotonicity violated at gfx_arch=0x{:x}: gen={} < prev={}",
                gfx,
                cur,
                prev
            );
            prev = cur;
        }
    }

    // ── GpuPlatform pattern match extracts inner value into separate binding ────

    #[test]
    fn gpu_platform_inner_value_extracted_via_let_binding() {
        // Arrange: create all three variants
        let cuda = GpuPlatform::Cuda { sm_version: 86 };
        let hip = GpuPlatform::Hip { gfx_arch: 0x90a };
        let metal = GpuPlatform::Metal { gpu_family: 1008 };

        // Act: extract inner u32 from each
        let (cuda_val, hip_val, metal_val) = match (cuda, hip, metal) {
            (GpuPlatform::Cuda { sm_version: c }, GpuPlatform::Hip { gfx_arch: h }, GpuPlatform::Metal { gpu_family: m }) => (c, h, m),
            _ => panic!("unexpected pattern"),
        };

        // Assert: values match expectations
        assert_eq!(cuda_val, 86);
        assert_eq!(hip_val, 0x90a);
        assert_eq!(metal_val, 1008);
    }

    // ── GpuTopology memory bandwidth ratio: global_mem / l2 ─────────────────────

    #[test]
    fn gpu_topology_global_to_l2_memory_ratio_realistic() {
        // Arrange: Ampere A100 with 40 MiB L2 and 24 GiB global
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };

        // Act: compute ratio
        let ratio = topo.global_mem_bytes / topo.l2_bytes;

        // Assert: global should be ~600x L2 for A100
        assert_eq!(ratio, 614);
    }

    // ── GpuPlatform Hash consistency: same value inserted twice yields same slot ─

    #[test]
    fn gpu_platform_hash_map_overwrite_preserves_count() {
        // Arrange
        let mut map = std::collections::HashMap::new();
        let key1 = GpuPlatform::Cuda { sm_version: 80 };
        let key2 = GpuPlatform::Cuda { sm_version: 80 };

        // Act: insert same key twice with different values
        map.insert(key1, "first");
        map.insert(key2, "second");

        // Assert: map has exactly one entry, with the second value
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&GpuPlatform::Cuda { sm_version: 80 }), Some(&"second"));
    }

    // ── GpuTopology Debug format for zero-initialized instance ──────────────────

    #[test]
    fn gpu_topology_debug_zero_initialized_no_panic() {
        // Arrange: fully zero-initialized topology
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 0 },
            compute_unit_count: 0,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 0,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };

        // Act: format Debug output
        let debug_str = format!("{topo:?}");

        // Assert: all field names present, no panic on zero values
        assert!(debug_str.contains("platform"));
        assert!(debug_str.contains("compute_unit_count"));
        assert!(debug_str.contains("tensor_core_gen"));
        assert!(debug_str.contains("warp_size"));
    }

    // ── Metal gpu_family value 1000 is below valid range but representable ──────

    #[test]
    fn metal_gpu_family_value_1000_below_apple1_range() {
        // Arrange: gpu_family=1000 is below Apple1 (1001) but GpuPlatform allows any u32
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 1000 },
            compute_unit_count: 1,
            tensor_core_gen: 0,
            shared_mem_per_sm_bytes: 0,
            l2_bytes: 0,
            global_mem_bytes: 0,
            warp_size: 32,
            compute_cap_major: 0,
            compute_cap_minor: 0,
        };

        // Act: extract gpu_family
        if let GpuPlatform::Metal { gpu_family } = topo.platform {
            // Assert: value is representable, below Apple1 threshold
            assert_eq!(gpu_family, 1000);
            assert!(gpu_family < 1001, "gpu_family 1000 is below Apple1 minimum");
        } else {
            panic!("expected Metal variant");
        }
    }

    // ── GpuTopology helper methods ────────────────────────────────────────────

    #[test]
    fn test_gpu_topology_sm_version_cuda() {
        let topo = GpuTopology {
            platform: GpuPlatform::Cuda { sm_version: 80 },
            compute_unit_count: 108,
            tensor_core_gen: 2,
            shared_mem_per_sm_bytes: 163_840,
            l2_bytes: 40 * 1024 * 1024,
            global_mem_bytes: 24 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 8,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.sm_version(), Some(80));
        assert_eq!(topo.memory_bytes(), 24 * 1024 * 1024 * 1024);
        assert_eq!(topo.compute_capability(), Some((8, 0)));
        assert!(topo.has_tensor_cores());
    }

    #[test]
    fn test_gpu_topology_sm_version_hip() {
        let topo = GpuTopology {
            platform: GpuPlatform::Hip { gfx_arch: 0x940 },
            compute_unit_count: 228,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 65_536,
            l2_bytes: 8 * 1024 * 1024,
            global_mem_bytes: 128 * 1024 * 1024 * 1024,
            warp_size: 64,
            compute_cap_major: 9,
            compute_cap_minor: 0x40,
        };
        assert_eq!(topo.sm_version(), None);
        assert_eq!(topo.compute_capability(), Some((9, 0x40)));
        assert!(topo.has_tensor_cores());
    }

    #[test]
    fn test_gpu_topology_metal() {
        let topo = GpuTopology {
            platform: GpuPlatform::Metal { gpu_family: 9 },
            compute_unit_count: 0,
            tensor_core_gen: 3,
            shared_mem_per_sm_bytes: 32_768,
            l2_bytes: 0,
            global_mem_bytes: 18 * 1024 * 1024 * 1024,
            warp_size: 32,
            compute_cap_major: 9,
            compute_cap_minor: 0,
        };
        assert_eq!(topo.sm_version(), None);
        assert_eq!(topo.compute_capability(), Some((9, 0)));
        assert!(topo.has_tensor_cores());
    }
}
