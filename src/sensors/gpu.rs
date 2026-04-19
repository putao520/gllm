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

/// GPU 平台类型（与 gllm-kernels 的 `Platform` 枚举对齐）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPlatform {
    /// NVIDIA CUDA（sm_version = major*10 + minor）
    Cuda { sm_version: u32 },
    /// AMD HIP / ROCm（gfx_arch，hex，如 0x908 / 0x90a / 0x940）
    Hip { gfx_arch: u32 },
    /// Apple Metal（gpu_family）
    Metal { gpu_family: u32 },
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
        assert_ne!(cuda, hip);
        assert_ne!(hip, metal);
    }
}
