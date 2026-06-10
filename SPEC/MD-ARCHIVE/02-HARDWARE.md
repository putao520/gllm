# 硬件适配层

> **SSOT**: 本文件定义 Platform 枚举、DeviceProfile 结构、ISV 检测、GPU SM 版本支持、硬件 Profile 矩阵、MemoryNetworkSensors、CompilerConstraints、LatencyProfiler、**HwOptEngine（统一硬件感知优化引擎）**。硬件参数→codegen 贯通链、融合策略硬件差异矩阵、算子实现硬件差异矩阵、统一分层 GEMM 模型见 `SPEC/04-OPERATORS.md`。JIT 编译管线如何消费 DeviceProfile 见 `SPEC/01-JIT-PIPELINE.md`。运行时编排如何消费 HwOptPlan 见 `SPEC/06-RUNTIME.md`。优化模块如何消费 FeatureRouter 见 `SPEC/05-OPTIMIZATIONS.md`。核心铁律见 `SPEC/00-PHILOSOPHY.md`。
>
> 交叉引用: `37-HARDWARE-ACCELERATION.md` 定义了 HardwareAcceleration 结构体和所有 IsaFeature 扩展（SparseTensorCore/TmaBulkCopy/NativeFp4/TmemAccess 等）。

## 1. Platform 枚举

```rust
pub enum Platform {
    X86_64 { avx512: bool, amx: bool },
    Aarch64 { sve: bool },
    Cuda { sm_version: u32 },
    Hip { gfx_arch: u32 },
    Metal { gpu_family: u32 },
}
```

AMX 不是独立 Platform variant，而是 CPU 平台的微内核加速能力。Phase 3 的 codegen 检测到 AMX 后，GEMM 微内核切换到 tile/block 指令（如 `TDPBF16PS`），一条指令完成整个 MR×NR 块的乘累加。

## 2. DeviceProfile

通过 driver API 一次性探测、运行时固定。Phase 2 融合决策和 Phase 3 代码生成完全由此结构驱动。

### 2.1 CPU 字段

| 字段 | 类型 | 说明 | codegen 影响 |
|------|------|------|-------------|
| `simd_width` | usize | SIMD 向量宽度（字节） | 循环步长、load/store 指令宽度 |
| `use_avx512` | bool | 是否使用 512-bit 指令 | ymm vs zmm 寄存器选择 |
| `has_amx` | bool | Intel AMX tile 指令 | GEMM 微内核切换到 AMX tile |
| `has_sve` | bool | ARM SVE 可变长向量 | predicated 向量操作 |
| `sve_vl_bytes` | usize | SVE 向量长度（字节） | SVE 循环步长 |
| `num_simd_regs` | usize | 可用 SIMD 寄存器数 | 微内核 MR×NR 规格、epilogue 深度 |
| `cache_sizes` | (usize, usize, usize) | L1/L2/L3 缓存大小（字节） | GEMM 分块 KC/MC/NC 参数 |
| `has_vnni` | bool | AVX-VNNI INT8 点积 | INT8 量化 GEMV 路径 |
| `has_bf16` | bool | AVX-512 BF16 原生 | VDPBF16PS 原生 BF16 GEMM |
| `has_avx512fp16` | bool | AVX-512 FP16 原生 | FP16 原生算术 |

### 2.2 GPU 字段

| 字段 | 类型 | 说明 | codegen 影响 |
|------|------|------|-------------|
| `sm_version` | u32 | SM 版本（70/80/90/100） | PTX 指令集选择（wmma/mma.sync/wgmma） |
| `shared_mem_per_block` | usize | 每 block 共享内存大小（字节） | Tile 尺寸决策（类比 CPU L1） |
| `tensor_core_gen` | u32 | Tensor Core 代数（0=无, 1=V1/WMMA, 2=V2, 3=V3/MMA.sync, 4=V4/WGMMA） | GEMM 策略选择（JitGpuTensorCore / JitGpu） |
| `compute_units` | usize | SM/CU 数量 | Grid 尺寸、并行度 |
| `warp_size` | usize | Warp/Wavefront 大小（32/64） | 线程组织 |

### 2.3 探测方式

| 平台 | 探测方法 |
|------|---------|
| x86_64 | `is_x86_feature_detected!()` 宏 + CPUID |
| AArch64 | `std::arch::is_aarch64_feature_detected!()` + 寄存器读取 |
| CUDA | `cuDeviceGetAttribute()` driver API（dlopen `libcuda.so.1`） |
| HIP | `hipDeviceGetAttribute()` driver API（dlopen `libamdhip64.so`） |
| Metal | `MTLDevice.supportsFamily()` Objective-C runtime |

所有 GPU driver 通过运行时 `dlopen` 加载，编译时不需要 CUDA SDK / ROCm SDK / Xcode。Feature-gated: `jit-cuda` / `jit-hip` / `jit-metal`。

## 3. ISV 检测

### 3.1 CPU ISV 检测

| ISV 特性 | 检测方式 | 对 GemmStrategy 的影响 |
|----------|---------|----------------------|
| VNNI (INT8 点积) | `is_x86_feature_detected!("avx512vnni")` 或 `avxvnni` | 量化 GEMV/GEMM 使用 vpdpbusd |
| BF16 (原生点积) | `is_x86_feature_detected!("avx512bf16")` | VDPBF16PS 原生 BF16 GEMM 路径 |
| FP16 (原生算术) | `is_x86_feature_detected!("avx512fp16")` | FP16 原生计算 |
| AMX (tile 指令) | `is_x86_feature_detected!("amx_tile")` + `amx_int8` + `amx_bf16` | tile GEMM 替代 BLIS 微内核 |
| SVE2 | `is_aarch64_feature_detected!("sve2")` | SVE2 整数/浮点指令 |

CPU ISV 策略链（`select_gemm_strategy()`）：

```
AMX → oneDNN → Accelerate → JIT BLIS
```

oneDNN 和 Accelerate 是 CPU ISV 的保留路径，因为 CPU 路径无 JIT GPU codegen。

### 3.2 GPU 禁止外部 BLAS

GPU 推理路径中禁止调用 cuBLAS/rocBLAS/cuDNN：

- GPU GEMM 全部走 JIT codegen 生成 PTX/HIP/MSL 原生二进制
- `GpuIsvCapabilities` 仅包含 `tensor_core_gen`（硬件矩阵单元代数）
- `GemmStrategy` GPU 路径为 `JitGpuTensorCore` / `JitGpu`，不存在 `CuBlas` / `RocBlas`

## 4. GPU SM 版本多支持

### 4.1 PtxKernelRegistry

按 SM 版本选择最优内核 emitter。禁止 Fallback。

```rust
pub struct PtxKernelRegistry {
    entries: Vec<(SmRange, Box<dyn Fn(...) -> Result<Vec<u8>>>)>,
}

pub struct SmRange {
    pub min: u32,  // 包含
    pub max: u32,  // 包含, u32::MAX 表示无上限
}
```

SM 版本选择逻辑：遍历 `entries`，找到第一个 `SmRange` 包含当前 `sm_version` 的条目。无匹配时返回 `Err`（不降级到低版本）。

### 4.2 FlashAttention 四版本

| SM 范围 | 名称 | 核心指令 | 数据搬运 |
|---------|------|---------|---------|
| sm_100+ | FA4 Block-Scaled | tcgen05.mma + block_scale + TMEM | TMEMLoad/TMEMScatter |
| sm_90 | FA3 Pipeline | WGMMA 16×16×64 + TMA | TMA 2D prefetch + Warp Specialization |
| sm_80-89 | FA2 Tiled | mma.sync 16×8×16 | cp.async 128B |
| sm_70-79 | wmma Tiled | wmma 16×16×16 | global_load（无异步） |

禁止 SM 版本 Fallback：
- SM70 没有 tensor core gating → codegen 生成 wmma 路径的 MoERouting（仍然融合）
- 标量 CPU → codegen 生成标量循环的 SwiGLU（仍然融合，不是拆成 3 个独立 op）

## 5. 硬件 Profile 矩阵

> **代码 SSOT**: `gllm-kernels/src/compiler/hardware_profile.rs::HardwareProfile` enum (14 变体含 Generic)
> **检测入口**: `HardwareProfile::detect(&DeviceProfile)`

14 个硬件 Profile（含 Generic fallback），每个定义了核心能力和对融合拓扑的影响。

### 5.1 GPU Profile (NVIDIA CUDA)

#### CudaSM100 (Blackwell, sm_100+)

- **核心能力**: FP4/FP6 原生 Tensor Core; Block-scaled GEMM (per-block 缩放因子内置); TMEM 256KB/SM; tcgen05.mma; 2-CTA 协同 MMA; Thread Block Cluster
- **拓扑影响**: 权重全程 FP4 无反量化节点; Block-scaled 消除独立 Scale 节点; TMEM 替代 shared memory 做 attention tiling; 2-CTA 协同产生跨 CTA 边

#### CudaSM90 (Hopper, sm_90-99)

- **核心能力**: TMA 2D/5D prefetch; WGMMA 16×16×64; Warp Specialization (producer/consumer); FP8 native; cuda::barrier; L2 multicast
- **拓扑影响**: TMA 替代 cp.async; WGMMA 替代 mma.sync; Warp spec 产生双线程组子图

#### CudaSM80 (Ampere/Ada, sm_80-89)

- **核心能力**: mma.sync 16×8×16; cp.async 128B; BF16/TF32 Tensor Core
- **拓扑影响**: mma.sync 主力; cp.async 异步预取; BF16 原生计算

### 5.2 GPU Profile (AMD ROCm)

#### RocmMI300 (CDNA3, MI300)

- **核心能力**: Matrix Cores; FP8 native; 64-wide wavefront; XCD/GCD chiplet
- **拓扑影响**: Matrix Core 融合; FP8 量化路径; chiplet-aware scheduling

#### RocmMI200 (CDNA2, MI200)

- **核心能力**: Matrix Cores; BF16; 64-wide wavefront
- **拓扑影响**: Matrix Core GEMM; 保守融合策略; 无 FP8

### 5.3 CPU Profile (x86_64)

#### CpuAvx10_2 (Intel AVX10.2 + APX)

- **核心能力**: 256-bit 统一 SIMD; P/E 核混合感知; VP2INTERSECT; 31 GPR; BF16 256-bit 原生; VNNI-INT8 256-bit
- **拓扑影响**: 31 GPR 支持最深 epilogue 链（≥8 ops 融合）; P-core 全速 / E-core 标量降级; VP2INTERSECT 硬件化 sparse mask

#### CpuAvx512 (Intel AVX-512, 含 AMX 子模式)

- **核心能力**: 512-bit SIMD (32 zmm); VNNI; VP2INTERSECT; BF16; AMX tile 8×8 BF16 (DeviceProfile.has_amx 驱动)
- **拓扑影响**: AMX 可用时 tile GEMM 替代 BLIS 微内核; 32 zmm 无溢出支持 8-op epilogue; 无 AMX 时 BLIS 微内核做 GEMM

#### CpuAvx2 (Intel/AMD AVX2)

- **核心能力**: 256-bit SIMD (16 ymm); FMA; F16C
- **拓扑影响**: 最保守拓扑: BLIS pack/unpack + 16 ymm epilogue 溢出到栈

### 5.4 CPU Profile (Apple Silicon)

#### AppleM1 / AppleM2 / AppleM3

- **核心能力**: AMX tiles; 统一内存架构; 128-bit NEON SIMD
- **拓扑影响**: AMX tile GEMM; 统一内存消除 htod/dtoh; 保守融合深度
- **差异**: M1 基础 AMX; M2 增强 AMX + 更宽内存; M3 动态缓存

### 5.5 CPU Profile (ARM Server)

#### ArmNeoverse (Neoverse V1/V2, NEON/SVE2)

- **核心能力**: 128-bit 固定 NEON SIMD; SVE2 可变长 (DeviceProfile.has_sve 驱动); ASIMD
- **拓扑影响**: BLIS 微内核 + NEON/SVE epilogue; 类似 AVX2 体验; SVE 可用时 predicated 向量操作

### 5.6 Fallback

#### Generic

- **核心能力**: 标量; 无 SIMD
- **拓扑影响**: 最低兼容; 禁止融合; 仅保证正确性

> **未实现的 SPEC Profile**（代码中无对应变体，待后续 Phase 补齐）:
> - AVX10.1 + APX (介于 CpuAvx2 和 CpuAvx10_2 之间)
> - SME2 + SVE2 (ARM outer product matrix)
> - SVE2 (ARM variable-length SIMD without ZA)
> - sm_70-79 (Volta/Turing wmma)

## 6. Driver API FFI

零外部依赖，运行时 dlopen。

| 后端 | Driver 库 | Feature gate |
|------|----------|-------------|
| CUDA | `libcuda.so.1` | `jit-cuda` |
| HIP | `libamdhip64.so` | `jit-hip` |
| Metal | `Metal.framework` | `jit-metal` |

HIP driver API 与 CUDA 几乎 1:1 映射（`cu` → `hip` 前缀），warp 大小差异：CDNA = 64 (wavefront)，RDNA = 32 (wave32)。

Metal 通过 Objective-C runtime 绑定，使用 `objc_msgSend` 调用。macOS 上 Metal framework 始终可用。

GPU 执行结果通过 CPU golden reference 验证：

| 精度 | Elementwise | Reduction | GEMM |
|------|------------|-----------|------|
| f32 | 1e-6 | 1e-5 | 1e-4 |
| f16 | 1e-3 | 1e-2 | 1e-2 |

## 7. MemoryNetworkSensors（NUMA / PCIe / RDMA / TLB 感知）

> **SSOT**: `src/sensors.rs`。探测跨机/跨片物理拓扑参数，转化为 JIT 编译器的约束变量。探测结果供 `CompilerConstraints::derive()` 消费（§8）。

### 7.1 核心结构

```rust
pub struct MemoryNetworkSensors {
    pub l2_cache_bytes: usize,
    pub ccx_numa_topology: Option<NumaTopology>,
    pub tlb_entries: usize,
    pub nic_bandwidth_gbs: Option<f32>,
    pub rdma_latency_us: Option<f32>,
    pub arm_sme_za_size: Option<usize>,
}

pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
}

pub struct NumaNode {
    pub node_id: usize,
    pub l3_bytes: usize,
    pub core_count: usize,
}
```

### 7.2 字段说明

| 字段 | 类型 | 探测目标 | JIT 影响 |
|------|------|---------|---------|
| `l2_cache_bytes` | usize | GPU L2 Cache 驻留容量 | GPU 分级驻留锚点，`CompilerConstraints.l2_cache_size` 取 `max(sensors.l2_cache_bytes, DeviceProfile.l2)` |
| `ccx_numa_topology` | Option\<NumaTopology\> | AMD CCD/CCX 的 L3 分区边界 | Sub-Batch 空间分片（§10 NUMA core bindings） |
| `tlb_entries` | usize | TLB 条目数上限 | 大页策略决策，GEMM 分块防 TLB miss |
| `nic_bandwidth_gbs` | Option\<f32\> | 跨机网卡带宽（GB/s） | RDMA Pipelining 约束 |
| `rdma_latency_us` | Option\<f32\> | 跨机 RDMA 往返延迟（μs） | Chunk 切分下限计算 |
| `arm_sme_za_size` | Option\<usize\> | ARM SME ZA 阵列尺寸 | ARM SME outer product 矩阵乘平铺决策 |

### 7.3 探测方法

#### NUMA 拓扑（Linux /sys 文件系统）

```
/sys/devices/system/node/
  node0/cpulist  →  "0-7"       → NumaNode { node_id: 0, core_count: 8 }
  node1/cpulist  →  "8-15"      → NumaNode { node_id: 1, core_count: 8 }
```

仅 Linux 平台启用（`#[cfg(target_os = "linux")"`）。非 Linux 平台返回 `None`，`CompilerConstraints.numa_node_count` 回退为 1。

cpulist 解析支持三种格式：范围 `"0-7"`、离散 `"0,2,4,6"`、混合 `"0-3,8-11"`。

#### TLB 条目检测

| 架构 | TLB 条目数 | 依据 |
|------|----------|------|
| x86_64 | 1024 | 现代 x86_64 L2 TLB 典型值 |
| AArch64 | 2048 | ARMv8 L2 TLB 典型值 |
| 其他 | 512 | 保守默认值 |

#### ARM SME ZA 阵列检测

当前阶段返回 `None`（预留接口）。实际检测需要读取 `SMCR_EL1.LEN` 寄存器获取 streaming SVE 向量长度，ZA 阵列尺寸 = `VL × VL`。

### 7.4 RDMA 参数配置

```rust
// 由上层调用方设置（分布式推理场景）
sensors.set_rdma_params(bandwidth_gbs: 100.0, latency_us: 10.0);
```

`set_rdma_params()` 设置网卡带宽和 RDMA 延迟，两者均为 `Option` 字段，本地推理场景保持 `None`。

### 7.5 RDMA 最小 Chunk 计算

```
min_chunk_size_for_rdma(compute_tflops) 计算：

  T_rdma_transfer(chunk) = latency + chunk_data / bandwidth
  T_compute(chunk)       = chunk_ops / compute_tflops

  约束: T_compute >= T_rdma_transfer
  min_data_gb = latency * 1e-6 * bandwidth
  min_tokens  = min_data_gb * 1e9 / hidden_dim_bytes
  结果取 max(min_tokens, 64)
```

返回 `None` 当 RDMA 参数未配置（本地推理场景）。

### 7.6 探测入口

```rust
let profile = DeviceProfile::detect();
let sensors = MemoryNetworkSensors::detect(&profile);
```

`detect()` 一次性完成所有探测。L2 Cache 直接从 `DeviceProfile.kernel_config.l2` 读取，TLB/NUMA/ZA 通过平台特定方法探测，RDMA 参数需后续通过 `set_rdma_params()` 显式配置。

## 8. CompilerConstraints（传感器 → JIT 约束桥）

> **SSOT**: `src/jit/compiler_constraints.rs`。将硬件探测结果严格坍缩为 JIT 编译器的强数学约束变量组。核心法则：废弃散乱的指令集条件判断，所有硬件探测结果必须坍缩为此结构体。

### 8.1 核心结构

```rust
pub struct CompilerConstraints {
    // 寄存器与平铺约束
    pub max_gpr_count: usize,
    pub optimal_tile_bits: TileBits,
    pub native_int4_dot: bool,

    // 缓存约束
    pub l1i_size: usize,
    pub l2_cache_size: usize,
    pub smem_size: Option<usize>,

    // 拓扑约束
    pub numa_node_count: usize,
    pub gpu_sm_version: Option<u32>,
    pub gpu_sm_count: Option<usize>,
    pub gpu_warp_size: Option<usize>,
    pub numa_core_bindings: Vec<(usize, usize, usize)>,

    // 向量约束
    pub simd_width_bits: usize,
    pub has_amx: bool,
    pub has_avx512: bool,
    pub has_sve: bool,
    pub sve_vl_bytes: Option<usize>,

    // 猜测量化约束
    pub has_tma: bool,
    pub has_native_fp4: bool,
    pub has_native_fp6: bool,

    // RDMA 融合约束
    pub rdma_min_chunk_tokens: Option<usize>,
    pub tensor_core_gen: u32,
}

pub struct TileBits {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}
```

### 8.2 derive() 推导方法

```rust
CompilerConstraints::derive(
    profile: &DeviceProfile,
    sensors: &MemoryNetworkSensors,
    probe_result: Option<&ProbeResult>,
) -> CompilerConstraints
```

三分量推导：

| 输入 | 推导内容 |
|------|---------|
| `DeviceProfile` | SIMD 宽度、ISA 特性（AMX/AVX-512/SVE）、缓存尺寸、GPU SM 版本 |
| `MemoryNetworkSensors` | L2 Cache 锚点、NUMA 节点数、RDMA 最小 Chunk |
| `ProbeResult`（可选） | Spill points 影响 tile_k 选择（寄存器紧张时减小平铺） |

### 8.3 字段推导规则

#### 寄存器约束

| 条件 | `max_gpr_count` | 推导逻辑 |
|------|-----------------|---------|
| `has_amx` 或 `use_avx512` | 31 | APX (x86 扩展 GPR) 隐含于 AMX/AVX-512 平台 |
| 其他 | 15 | x86_64 基线 GPR 数 |

#### TileBits 推导规则

| 硬件特征 | tile_m | tile_n | tile_k | 说明 |
|---------|--------|--------|--------|------|
| AMX | 64 | 64 | 64 | 1024-bit 阵列，极大平铺 |
| AVX-512 | 32 | 32 | 32 | 512-bit SIMD 充裕寄存器 |
| SVE | vl*8/32 | vl*8/32 | 32 | 按 SVE 向量长度动态计算 |
| AVX2 / NEON | 16 | 16 | 16 | 基础平铺 |

其中 SVE 的 `vl` = `sve_vl_bytes.max(16)`，`tile_m/tile_n = vl × 8 / 32`（按 float 数）。

#### INT4 硬件支持

```rust
native_int4_dot = has_amx || has_avx512
```

AMX tile 指令和 AVX-512 VNNI 均支持 INT4→INT8 解包后的硬件点积。

#### 缓存约束

| 字段 | 推导 |
|------|------|
| `l1i_size` | `DeviceProfile.cache_sizes().0`（L1d ≈ L1i） |
| `l2_cache_size` | `max(sensors.l2_cache_bytes, DeviceProfile.l2)` |
| `smem_size` | GPU 路径通过 feature-gated 检测设置；CPU 路径为 `None` |

#### 拓扑约束

| 字段 | 推导 |
|------|------|
| `numa_node_count` | `sensors.ccx_numa_topology.nodes.len()`，无 NUMA 时为 1 |
| `gpu_sm_version` / `gpu_sm_count` / `gpu_warp_size` | GPU 路径通过 feature-gated 检测设置 |
| `tensor_core_gen` | GPU Tensor Core 代数（0=无） |

#### Hopper/Blackwell 特性

| 字段 | 触发条件 |
|------|---------|
| `has_tma` | SM >= 90（Hopper TMA 2D/5D prefetch） |
| `has_native_fp4` | SM >= 100（Blackwell FP4 原生 Tensor Core） |
| `has_native_fp6` | SM >= 100（Blackwell FP6 原生 Tensor Core） |

以上特性仅通过 GPU feature-gated 路径设置，CPU 路径始终为 `false`。

#### RDMA 约束

```rust
rdma_min_chunk_tokens = sensors.min_chunk_size_for_rdma(100.0)
```

调用 `MemoryNetworkSensors::min_chunk_size_for_rdma()` 计算。未配置 RDMA 时为 `None`。

### 8.4 L1i 预算检查

```rust
fn exceeds_l1i_budget(&self, kernel_code_bytes: usize) -> bool
```

阈值 = `l1i_size × 80%`。当编译后的 kernel 二进制超过此阈值时，JIT 退化为 `jmp` 长跳转而非内联平铺，防止 L1i 缓存颠簸。

### 8.5 GPU SM 分区

```rust
fn gpu_sm_partition(&self, num_partitions: usize) -> GpuSmPartition
```

```rust
pub struct GpuSmPartition {
    pub total_sm: usize,
    pub num_partitions: usize,
    pub sm_per_partition: usize,
}
```

将 GPU SM 按等分划分为 `num_partitions` 个分区（§12.1 Sub-Batch 空间分片）。`sm_per_partition = total_sm / num_partitions`（整数除法）。

### 8.6 NUMA 核绑定映射

```rust
fn numa_core_bindings(&self) -> Vec<(usize, Range<usize>)>
```

根据 `numa_node_count` 和 `available_parallelism()` 将逻辑核心均匀分配到 NUMA 节点：

```
node 0: cores [0, cores_per_node)
node 1: cores [cores_per_node, 2 * cores_per_node)
...
```

每个分区绑定到一个 NUMA 节点，确保 L3 Cache 局部性。

## 9. LatencyProfiler（硬件物理拐点探测）

> **SSOT**: `src/jit/profiler.rs`。模型加载时通过真实 micro-benchmark 探测硬件物理拐点，实现 SPEC §12.4 "硬件感知型黄金装筒规则"。探测结果供 `GoldenBucketRegistry` 和 `CompilerConstraints` 消费。

### 9.1 核心原则

- **严禁预设硬编码数组**：禁止使用 `[128, 512, 1024, 2048]` 等静态 Bucket。采样点通过 `ProbeConfig.sample_points()` 按 2 的幂步进动态生成。
- **真实物理探测**：通过 micro-benchmark 测定寄存器溢出、SMEM 满载、L2 Thrashing 阈值。
- **黄金装筒塌缩**：将任意 SEQ 长度映射到探测出的"黄金尺寸"（Golden Sizes）。
- **缓存复用**：探测结果按 `device_fingerprint` 缓存，避免重复探测。

### 9.2 核心结构

```rust
pub struct ProbeResult {
    pub spill_points: Vec<usize>,
    pub smem_cliffs: Vec<(usize, f32)>,
    pub l2_thrash_threshold: usize,
    pub device_fingerprint: String,
    pub raw_measurements: HashMap<usize, u64>,
}

pub struct ProbeConfig {
    pub seq_range: (usize, usize),
    pub sample_density: usize,
    pub repeat_count: usize,
    pub hidden_size: usize,
    pub timeout_per_sample: Duration,
}
```

### 9.3 ProbeConfig 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seq_range` | (usize, usize) | (1, 4096) | 探测的 seq_len 范围 [min, max] |
| `sample_density` | usize | 1 | 2 的幂步进密度（1 = 每个幂次都采样，2 = 每 4 个采样一次） |
| `repeat_count` | usize | 5 | 每个采样点重复次数，取中位数降噪 |
| `hidden_size` | usize | 1024 | 构建微型 GEMM 的隐藏维度 |
| `timeout_per_sample` | Duration | 10s | 单个采样点超时 |

### 9.4 采样点生成算法

```rust
// sample_density = 1: 1, 2, 4, 8, 16, 32, 64, ..., max
// sample_density = 2: 1, 4, 16, 64, 256, 1024, ..., max

fn sample_points(&self) -> Vec<usize> {
    let mut current = 1;
    while current <= max {
        if current >= min { points.push(current); }
        current <<= sample_density;  // 2^n 步进
    }
    // 确保 max 被包含（如果不是 2 的幂）
    if last < max { points.push(max); }
}
```

### 9.5 CPU 探测算法

```
probe_cpu(config) → ProbeResult:
  1. DeviceProfile::detect() → 硬件指纹
  2. 对每个采样点 seq_len:
     a. compile_micro_gemm(seq_len, hidden) → MicroKernel
        图结构: A[seq_len, hidden] × B[hidden, hidden] → C[seq_len, hidden]
     b. benchmark_kernel(kernel, repeat_count, timeout) → Vec<u64>
        重复执行 repeat_count 次，取中位数
  3. detect_spill_points(measurements) → Vec<usize>
  4. detect_l2_thrash(measurements, profile) → usize
  5. CPU 无 SMEM → smem_cliffs = []
```

### 9.6 GPU 探测算法

GPU 拐点通过硬件参数解析推导，**无需编译 PTX kernel**（feature-gated: `cuda` / `hip` / `metal`）。

```
probe_gpu(config, sm_version, l2_size, smem_size) → ProbeResult:

  SMEM Cliffs 推导:
    default_tile (16×16 f32 = 1024B) 占用 > 80% smem → cliff at 16
    large_tile  (32×32 f32 = 4096B)  占用 > 80% smem → cliff at 32

  L2 Thrashing 推导:
    l2_thrash_threshold = l2_size / (hidden × 2 × elem_bytes)

  Spill Points 推导:
    1. L2 thrashing 拐点（seq 超出 L2 时）
    2. SMEM 满载拐点（KV cache 占 SMEM 80% 时）
    3. SM90+ Hopper 特性拐点（group-SMEM 228KB 阈值）
    去重排序
```

### 9.7 Spill Point 检测算法

```
detect_spill_points(measurements):
  1. 按 seq_len 排序
  2. 计算 times_per_element = time / seq_len
  3. 计算相邻点 delta = |tpe[i] - tpe[i-1]|
  4. median_delta = median(deltas)
  5. spill_point = delta > median_delta × 2.0 的点
```

**判定标准**：性能 delta 超过中位数 delta 的 2 倍，标志寄存器分配策略的切换点（从全寄存器保持到部分溢出）。

### 9.8 SMEM Cliff 检测

当 GEMM tile 占用超过 SMEM 的 80% 时，GPU occupancy 下降（如从 16 blocks 降到 8 blocks）。`smem_cliffs` 记录 `(seq_len, occupancy_ratio)` 对。

### 9.9 L2 Thrashing 检测

```
detect_l2_thrash(measurements, profile):
  1. estimated_l2 = l3_size / 4
  2. 遍历相邻采样点窗口:
     tpe2 > tpe1 × 1.2 且 working_set > estimated_l2
     → 返回该 seq_len
  3. 兜底: estimated_l2 / (hidden × 4)
```

**判定标准**：首个 per-element 性能下降 > 20% 且工作集超出 L2 估算大小的采样点。

### 9.10 设备指纹

| 平台 | 指纹格式 |
|------|---------|
| CPU | `cpu-{arch}-{isa}-l1d{KB}-l2{KB}-l3{KB}` |
| GPU | `gpu-sm{version}-l2{KB}-smem{KB}` |

指纹作为 `GoldenBucketRegistry` 缓存 key，相同硬件环境复用探测结果。

### 9.11 ProbeResult 下游消费

| 消费方 | 消费内容 |
|--------|---------|
| `CompilerConstraints::derive()` | `spill_points` 影响 `optimal_tile_bits.tile_k` 选择 |
| `GoldenBucketRegistry` | `spill_points` + `l2_thrash_threshold` 生成黄金装筒桶 |
| `ProbeResult.device_fingerprint` | 缓存 key，避免同设备重复探测 |

### 9.12 错误处理

所有探测错误必须传播，禁止使用默认值绕过：

| 错误类型 | 场景 |
|---------|------|
| `ProbeError::Compilation` | 微型 GEMM 编译失败 |
| `ProbeError::Timeout` | 单个采样点执行超时 |
| `ProbeError::Io` | 缓存读写失败 |
| `ProbeError::Serialization` | 缓存序列化/反序列化失败 |

## 10. HwOptEngine — 硬件感知统一优化引擎

> **SSOT**: 本节是所有硬件感知优化决策的唯一真源。JIT codegen 策略选择、融合深度决策、缓存预算分配、并行策略规划、批调度策略等不再散落在各模块独立判断，统一由本引擎一次性求解为不可变 `HwOptPlan`。

### 10.1 核心原则

| 原则 | 说明 |
|------|------|
| **Cost-Model 驱动** | 禁止硬编码 if/else 查表。每个决策通过数学成本模型评估候选策略，选最优解 |
| **全局最优** | 各子求解器的输出互相依赖，引擎按 DAG 拓扑序求解，保证跨决策一致性 |
| **一次性推导** | `HwOptPlan` 在模型加载时计算一次，推理全程只读。与 JIT 缓存协议一致 |
| **约束可溯** | 每个决策可追溯到 `DeviceProfile` 的具体字段和推导公式，无魔法数字 |
| **Feature-gated 兼容** | GPU 求解器仅在对应 feature gate 启用时编译，CPU-only 构建零开销 |

**替代关系**: 本引擎替代以下散落逻辑：
- `04-OPERATORS.md` §11 硬件→codegen 链 → `FeatureRouter` + `GemmSolver`
- `04-OPERATORS.md` §8 融合差异矩阵 → `FusionSolver`
- `04-OPERATORS.md` §13 GEMM 优化 → `GemmSolver` + `CacheBudgetSolver`
- `06-RUNTIME.md` §14 Multi-Wave → `ParallelismSolver` + `BatchSolver`
- `06-RUNTIME.md` §12 Golden Bucket → `BatchSolver`
- `06-RUNTIME.md` §13 VariantRegistry → `FeatureRouter`
- `01-JIT-PIPELINE.md` §4 融合决策 → `FusionSolver`（消费引擎输出，不再直接读 DeviceProfile）

### 10.2 引擎总览

```
                     ┌──────────────────────────────┐
                     │         输入三件套            │
                     │  DeviceProfile (§2)           │
                     │  MemoryNetworkSensors (§7)    │
                     │  ProbeResult (§9)             │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │         HwOptEngine                   │
                 │                                       │
                 │  ┌──────────────────────────────────┐ │
                 │  │    Solver DAG (拓扑序求解)        │ │
                 │  │                                    │ │
                 │  │  1. RooflineAnalyzer               │ │
                 │  │     ↓                              │ │
                 │  │  2. CacheBudgetSolver ←──┐        │ │
                 │  │     ↓                    │        │ │
                 │  │  3. GemmSolver ──────────┘        │ │
                 │  │     ↓ ↓                           │ │
                 │  │  4. FusionSolver                  │ │
                 │  │  5. AttentionSolver               │ │
                 │  │     ↓ ↓                           │ │
                 │  │  6. ParallelismSolver             │ │
                 │  │     ↓                             │ │
                 │  │  7. BatchSolver                   │ │
                 │  │     ↓                             │ │
                 │  │  8. FeatureRouter                 │ │
                 │  │                                    │ │
                 │  └──────────────────────────────────┘ │
                 │                                       │
                 │  CostModel — 贯穿所有求解器的统一估值  │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │    HwOptPlan (不可变)          │
                     │                               │
                     │  ├ gemm_plan                  │
                     │  ├ cache_plan                 │
                     │  ├ fusion_plan                │
                     │  ├ attention_plan             │
                     │  ├ parallel_plan              │
                     │  ├ batch_plan                 │
                     │  └ feature_plan               │
                     └──────────────────────────────┘
```

### 10.3 RooflineAnalyzer — 算子瓶颈分类器

**职责**: 为每种算子类型计算算术强度（Arithmetic Intensity），分类为 compute-bound / memory-bound / mixed，为下游求解器提供优化方向。

**无依赖**（直接读 DeviceProfile + ProbeResult）。

#### 算术强度公式

```
AI(op) = FLOPs(op) / bytes_accessed(op)

FLOPs(op):
  GEMM(M,N,K) = 2MNK
  Attention(Q,M,K,V,N) = 4MNK + 4MN² (M=seq_len, K=head_dim, N=seq_len)
  Norm(hidden) = 3 × hidden  (sum_sq + div + mul)
  Elementwise(hidden) = hidden

bytes_accessed(op):
  = inputs_bytes + outputs_bytes
  GEMM(M,N,K) = (M×K + K×N + M×N) × elem_bytes
  Attention = (MQ + NK + NV + MN) × elem_bytes
```

#### 瓶颈分类

```
roofline_point = peak_FLOPS / peak_bandwidth  (ops/byte)

if AI(op) > roofline_point × 1.5:
    ComputeBound    → 优化方向：最大化 FMA 利用率，增加 tile/寄存器使用
elif AI(op) < roofline_point × 0.67:
    MemoryBound     → 优化方向：最小化内存搬运，增加融合/预取
else:
    Mixed           → 优化方向：平衡计算与带宽，选择融合点
```

#### 硬件 Roofline Point

| 硬件 Profile | peak_FLOPS | peak_bandwidth | roofline_point |
|-------------|-----------|---------------|----------------|
| AVX2 (F32) | 0.5 TFLOPS | 50 GB/s | 10 ops/byte |
| AVX-512 (F32) | 1.0 TFLOPS | 70 GB/s | 14.3 ops/byte |
| AMX (BF16) | 2.0 TFLOPS | 70 GB/s | 28.6 ops/byte |
| GPU sm_80 (A100) | 312 TFLOPS | 2039 GB/s | 153 ops/byte |
| GPU sm_90 (H100) | 990 TFLOPS | 3350 GB/s | 296 ops/byte |
| GPU sm_100+ (B200) | 2250 TFLOPS | 8000 GB/s | 281 ops/byte |

#### 算子分类结果

| 算子 | AI (F32, hidden=4096) | GPU sm_80 | CPU AVX2 |
|------|----------------------|-----------|----------|
| GEMM(M≥128) | ~4096 | ComputeBound | ComputeBound |
| GEMM(M=1, decode) | ~1 | MemoryBound | MemoryBound |
| Attention (prefill) | ~128 | ComputeBound | Mixed |
| Attention (decode) | ~2 | MemoryBound | MemoryBound |
| RmsNorm | 3 | MemoryBound | MemoryBound |
| SiLU/SwiGLU | 2 | MemoryBound | MemoryBound |
| RoPE | 1 | MemoryBound | MemoryBound |

**关键结论**: decode 阶段几乎全 MemoryBound → 融合优先（减少内存搬运）；prefill 阶段 GEMM ComputeBound → tiling 优先（最大化 FMA 利用率）。

### 10.4 GemmSolver — GEMM 策略求解器

**职责**: 给定硬件寄存器预算和缓存层级，通过约束优化求解最优 GEMM 微内核参数。

**依赖**: `RooflineAnalyzer`（算子瓶颈分类）、`CacheBudgetSolver`（L1/L2 可用空间）。

#### 输入

```rust
struct GemmSolverInput {
    profile: &'a DeviceProfile,
    constraints: &'a CompilerConstraints,
    roofline: &'a RooflineClass,  // compute/memory/mixed
    cache_budget: &'a CacheBudgetPlan,
    elem_bytes: usize,             // 4(F32), 2(F16/BF16)
}
```

#### 候选空间枚举

```rust
struct GemmCandidate {
    mr: usize,
    nr: usize,
    k_depth: usize,        // K-loop 软件流水线深度
    pf_distance: usize,    // 预取距离（字节）
    nr_variant: Option<usize>,  // NR 缩减（牺牲吞吐换 epilogue）
}
```

候选 `MR×NR` 组合由寄存器约束生成：

```
for mr in [4, 6, 8, 10, 12, 14, 16]:
    for nr in [8, 12, 16, 24, 32]:
        nr_vecs = ceil(nr / simd_width_elems)
        acc = mr × nr_vecs
        if acc + 2(ptr) + 2(scratch_min) ≤ num_simd_regs:
            yield GemmCandidate { mr, nr, ... }
```

#### 成本评估

对每个候选计算预估延迟：

```
cost(candidate) = max(T_compute, T_memory) + T_overhead

T_compute = (2 × MR × NR × KC) / (peak_fma_throughput × simd_efficiency)
T_memory = ((MR × KC + KC × NR) × elem_bytes) / peak_bandwidth × (1 - l1_hit_rate)
T_overhead = k_depth × prefetch_latency  (软件流水线 setup)

simd_efficiency = (MR × nr_vecs) / (1 + nr_vecs)  // FMA/Load 比率
l1_hit_rate = min(1.0, tile_bytes / (l1_available × 0.75))
```

**关键**: 当 `RooflineClass = MemoryBound` 时，`T_memory` 主导，成本函数自动倾向更小的 tile（减少内存搬运）；当 `ComputeBound` 时，`T_compute` 主导，倾向最大化 FMA/Load。

#### 约束系统

```
硬约束:
  accumulator_regs(mr, nr) + pointer_regs(2) + scratch_min(2) ≤ num_simd_regs
  tile_a_bytes + tile_b_bytes ≤ l1_available × 0.75
  k_depth × 2 × cache_line_bytes ≤ l2_available  // 预取不溢出 L2

软约束（可违反但有惩罚）:
  scratch_available ≥ epilogue_required  // epilogue 深度需求
  k_depth ≥ 2                            // compute-bound 场景建议深度 ≥ 2
```

#### 输出

```rust
struct GemmPlan {
    mr: usize,
    nr: usize,
    nr_vecs: usize,
    k_depth: usize,
    pf_distance: usize,
    nr_variant: Option<usize>,
    max_epilogue_depth: usize,
    acc_regs: usize,
    scratch_regs: usize,
    strategy: GemmStrategy,  // AmxTile / Avx512NativeBF16 / BLIS / ...
}
```

| 字段 | 推导来源 |
|------|---------|
| `mr`, `nr` | 成本最优候选 |
| `k_depth` | 寄存器压力 vs 预取收益权衡 |
| `pf_distance` | `max(cache_line × k_depth, hidden × elem_bytes)` |
| `nr_variant` | 当 `scratch < epilogue_needed` 时缩减 NR |
| `max_epilogue_depth` | `(scratch_regs / avg_regs_per_op).floor()` |
| `strategy` | `has_amx → AmxTile`, `has_bf16 && avx512 → NativeBF16`, else `BLIS` |

#### 策略路由（Cost-Based，非 if/else）

```
策略候选池 = [
    (AmxTile,      条件: has_amx),
    (NativeBF16,   条件: has_bf16 && use_avx512),
    (TensorCore,   条件: tensor_core_gen >= 2),
    (BLIS_Avx512,  条件: use_avx512),
    (BLIS_Avx2,    条件: simd_width >= 32),
    (BLIS_Neon,    条件: platform == Aarch64),
]

→ 过滤: 移除不满足硬件前提的候选
→ 评估: 对每个候选调用 cost(candidate)
→ 选择: argmin(cost)
```

### 10.5 CacheBudgetSolver — 缓存预算分配器

**职责**: 将硬件缓存层级（L1/L2/L3/HBM）的物理容量，按优先级分配给 KV Cache、权重、激活值、Workspace，输出三级缓存预算方案。

**无依赖**（直接读 DeviceProfile + MemoryNetworkSensors）。

#### 输入

```rust
struct CacheBudgetInput {
    profile: &'a DeviceProfile,
    sensors: &'a MemoryNetworkSensors,
    model_bytes: usize,         // 模型权重总字节
    kv_bytes_per_token: usize,  // 每 token KV 字节数
    hidden_bytes: usize,        // hidden_dim × elem_bytes
}
```

#### CPU 缓存预算算法

```
L1 总量 = profile.cache_sizes.0
  GEMM tile_A + tile_B ≤ L1 × 75%
  可用于 TileLevelFusion scratch = L1 × 25%

L2 总量 = max(sensors.l2_cache_bytes, profile.cache_sizes.1)
  kv_budget     = L2 × 0.40   // Attention-bound 时 KV 热页
  weight_budget = L2 × 0.35   // GEMM-bound 时权重预取窗口
  activation    = L2 × 0.25   // Epilogue scratch / 前驱输出缓存

L3 总量 = profile.cache_sizes.2
  model_weights = min(model_bytes, L3 × 70%)
  kv_cold_pages = L3 × 20%
  workspace     = L3 × 10%
```

**动态调节**: 当 RooflineAnalyzer 表明当前工作负载为 MemoryBound 时，`kv_budget` 提升到 50%，`weight_budget` 降为 25%。

#### GPU 显存预算算法

```
HBM 总量 = profile.total_memory
model_weights = model_bytes × quant_ratio
kv_pool       = HBM × 0.60 - model_weights
workspace     = HBM × 0.10
reserved      = HBM × 0.05   // CUDA/驱动

SMEM 总量 = profile.shared_mem_per_block
  attention_tile = SMEM × 40%
  routing_table  = 0%        // 立即数常量，不占 SMEM
  compact_buffer = SMEM × 20%
  余量           = SMEM × 40%

max_pages = kv_pool / (page_size × 2 × num_layers × kv_dim × elem_bytes)
```

#### NUMA 感知分配

```
当 sensors.ccx_numa_topology.is_some():
  每个 NUMA 节点的 L3 预算 = node.l3_bytes
  每个 NUMA 节点的核心预算 = node.core_count
  → weights 按 NUMA 节点复制或按层交错放置
```

#### 输出

```rust
struct CacheBudgetPlan {
    l1_tile_budget: usize,          // L1 中 GEMM tile 可用字节
    l1_fusion_scratch: usize,       // L1 中 TileLevelFusion scratch
    l2_kv_budget: usize,            // L2 KV 热页预算
    l2_weight_budget: usize,        // L2 权重预取窗口
    l2_activation_budget: usize,    // L2 激活值缓存
    l3_model_budget: usize,         // L3 模型权重预算
    l3_kv_cold_budget: usize,       // L3 KV 冷页预算
    hbm_kv_pool: Option<usize>,     // GPU KV 池字节（CPU 为 None）
    hbm_max_pages: Option<usize>,   // GPU 最大页面数
    smem_attention: Option<usize>,  // GPU SMEM Attention 预算
    smem_compact: Option<usize>,    // GPU SMEM Compact 缓冲
    numa_nodes: Vec<NumaBudget>,    // 每 NUMA 节点预算
}
```

### 10.6 FusionSolver — 融合策略求解器

**职责**: 给定算子链 DAG 和寄存器/缓存约束，求解每个融合点的最优融合模式和最大 epilogue 深度。

**依赖**: `GemmSolver`（epilogue 寄存器预算）、`CacheBudgetSolver`（L1 scratch 预算）、`RooflineAnalyzer`（瓶颈分类）。

#### 输入

```rust
struct FusionSolverInput {
    gemm_plan: &'a GemmPlan,          // 来自 GemmSolver
    cache_plan: &'a CacheBudgetPlan,  // 来自 CacheBudgetSolver
    roofline: &'a RooflineResult,     // 来自 RooflineAnalyzer
    num_simd_regs: usize,
    l1_cache_bytes: usize,
}
```

#### 融合模式候选评估

对每个潜在的融合机会（后支配树上的生产者-消费者对），枚举所有可行融合模式并评估成本：

```
候选池 = [
    EpilogueInjection { depth: 1..max_epilogue },
    TileLevelFusion { tile_rows: MC },
    ComputeRoot,
    LoopFusion { chain_length: 2..N },
    QkvSharedInput,
    NormIntoGemm,
    FFNBlock,
    CrossLayerResidual,
]
```

#### EpilogueInjection 深度决策

```
max_depth = gemm_plan.max_epilogue_depth
available_scratch = gemm_plan.scratch_regs

for depth in (1..=max_depth).rev():
    required = sum(TraceOp::register_cost(&epilogue_ops[..depth]))
    if required ≤ available_scratch:
        selected_depth = depth
        break
```

**注意**: 当 `depth = 0` 时，epilogue 不注入，回退到 ComputeRoot 或 LoopFusion。这不是降级，而是成本最优解。

#### TileLevelFusion vs ComputeRoot 决策

```
predecessor_output_bytes = hidden_bytes × tile_rows

if predecessor_output_bytes > cache_plan.l1_tile_budget × 0.75:
    TileLevelFusion  // 输出太大无法驻留 L1，嵌入 MC 循环逐 tile 处理
else:
    ComputeRoot      // 输出可驻留 L1，先完整计算再消费
```

#### FFNBlock 融合路径选择

```
候选路径:
  A. EpilogueInjection(SiLU) + LoopFusion(Mul)  // Gate epilogue 内联 SiLU，结果与 Up 乘
  B. 分离 GEMM + LoopFusion(SiLU×Up)            // Gate/Up 各自独立 GEMM

选择: 取决于 scratch 寄存器是否足够注入 SiLU
  if scratch >= silu_cost(1 reg):
      → 路径 A（减少一次内存写回）
  else:
      → 路径 B（牺牲一次写回换取寄存器安全）
```

#### 输出

```rust
struct FusionPlan {
    max_epilogue_depth: usize,
    tile_fusion_threshold: usize,      // L1 字节阈值，超过此值用 TileLevelFusion
    ffn_strategy: FfnFusionStrategy,   // GateSiLUInject / SeparateGemm
    norm_into_gemm_enabled: bool,
    cross_layer_residual_enabled: bool,
    qkv_shared_input_enabled: bool,
    fusions: Vec<LocalFusionDecision>,  // 每个融合点的具体决策
}

enum FfnFusionStrategy {
    GateSiLUInject,     // SiLU 注入 Gate GEMM epilogue
    SeparateGemm,       // Gate/Up 独立 GEMM + LoopFusion
}

struct LocalFusionDecision {
    producer_idx: usize,
    consumer_idx: usize,
    strategy: FusionStrategy,
    cost_estimate_ns: u64,
}
```

### 10.7 AttentionSolver — 注意力策略求解器

**职责**: 根据 GPU SM 版本/SMEM/TMEM 或 CPU ISV 特性，求解最优注意力实现路径和 tiling 参数。

**依赖**: `CacheBudgetSolver`（SMEM/KV 预算）、`RooflineAnalyzer`（prefill vs decode 瓶颈分类）。

#### 候选路径枚举

```
GPU 候选池:
  FA4BlockScaled    条件: sm >= 100
  FA3Pipeline       条件: sm >= 90
  FA2Tiled          条件: sm >= 80
  WmmaTiled         条件: sm >= 70

CPU 候选池:
  AmxTileAttention  条件: has_amx
  Avx512Flash       条件: use_avx512
  NeonLoop          条件: platform == Aarch64
  ScalarLoop        条件: always (兜底)
```

#### GPU 路径成本模型

```
cost(path) = T_qk + T_softmax + T_av + T_kv_load

T_qk = (M × K × N) / (tc_throughput × efficiency)
  tc_throughput: FA4 > FA3 > FA2 > wmma (按 SM 版本递减)
  efficiency: warp_specialization ? 0.9 : 0.6

T_kv_load = (2 × N × kv_dim × elem_bytes) / peak_bandwidth
  N = kv_cache_seq_len, 受 SMEM 预算约束的分块大小影响

T_softmax = N × softmax_ops / (compute_throughput × softmax_efficiency)
  softmax_efficiency: online_softmax ? 0.95 : 0.5 (单遍 vs 三遍)
```

#### CPU 路径成本模型

```
cost(path) = T_qk_gemv + T_softmax + T_av_gemv

AMX: T_qk = (M × K × N) / (tdpbf16ps_throughput × tile_efficiency)
Avx512: T_qk = (M × K × N) / (vdpbf16ps_throughput)
Neon: T_qk = (M × K × N) / (neon_fmla_throughput)
```

#### Attention Tile 尺寸求解

```
SMEM 预算 = cache_plan.smem_attention

attention_tile_bytes = 3 × tile_size × head_dim × elem_bytes  // Q + K + V
约束: attention_tile_bytes ≤ smem_attention

tile_size = smem_attention / (3 × head_dim × elem_bytes)
tile_size = prev_power_of_2(tile_size)  // 对齐到 2 的幂
tile_size = max(tile_size, 16)          // 最小 tile
```

#### 输出

```rust
struct AttentionPlan {
    variant: AttentionVariant,
    tile_size: usize,
    smem_layout: Option<SmemLayout>,
    online_softmax: bool,
    warp_specialization: bool,   // SM90+ producer/consumer
    tma_enabled: bool,           // SM90+ TMA 2D prefetch
}

enum AttentionVariant {
    FA4BlockScaled,
    FA3Pipeline,
    FA2Tiled,
    WmmaTiled,
    AmxTile,
    Avx512Loop,
    NeonLoop,
    ScalarLoop,
}
```

### 10.8 ParallelismSolver — 并行策略求解器

**职责**: 根据 GPU SM 数量或 CPU NUMA 拓扑，求解 Multi-Wave 分区方案和硬件饱和目标。

**依赖**: `GemmSolver`（wave 内 GEMM 吞吐）、`CacheBudgetSolver`（NUMA 预算）。

#### GPU Wave 数量求解

```
sm_total = profile.compute_units
sm_per_wave = sm_total / wave_count

wave_count 候选 = [1, 2, 4]

for wc in wave_count_candidates:
    if wc > sm_total / min_sm_per_wave:  // min_sm_per_wave = 16
        break  // 分区太小无意义
    sm_per = sm_total / wc
    min_tokens = sm_per × warp_size × occupancy_target(0.5)
    // 评估: 当前 batch 中 decode + prefill 能否填满 wc 个 wave
```

#### 硬件饱和度计算

```
decode_per_wave = min_tokens / 1  // decode seq_len = 1
prefill_min_chunk = min_tokens    // prefill chunk 需要的 token 数

// sm_80 (108 SM, 2 waves):
//   sm_per_wave = 54, min_tokens = 54 × 32 × 0.5 = 864
//   → 需要 864 并发 decode 序列才能饱和一个 wave

// sm_90 (132 SM, 2 waves):
//   sm_per_wave = 66, min_tokens = 66 × 32 × 0.5 = 1056
```

#### CPU NUMA 绑定求解

```
numa_nodes = sensors.ccx_numa_topology
wave_count = numa_nodes.len()

for each node:
    node.wave binds to node.cores
    node.kv_cache 在 node.l3 本地
    → 零跨 NUMA 内存访问
```

#### 输出

```rust
struct ParallelPlan {
    wave_count: usize,
    gpu_sm_partition: Option<GpuSmPartition>,
    numa_bindings: Vec<NumaBinding>,
    min_batch_tokens_per_wave: usize,
    min_decode_seqs_per_wave: usize,
    occupancy_target: f32,
}

struct NumaBinding {
    node_id: usize,
    core_range: (usize, usize),
    l3_bytes: usize,
}
```

### 10.9 FeatureRouter — 硬件特性路由器

**职责**: 统一管理所有硬件特性的启用/禁用决策。替代散落在 codegen 各处的 `has_xxx` 条件检查。

**依赖**: 所有其他求解器（最终阶段执行，根据已求解的策略确认特性使用）。

#### 特性注册表

```rust
struct FeatureDecision {
    feature: HwFeature,
    enabled: bool,
    reason: &'static str,       // 启用/禁用的推导原因
    codegen_path: CodegenPath,  // 启用时使用的 codegen 路径
    impact: Vec<SolverId>,      // 影响哪些求解器
}

enum HwFeature {
    // x86
    Avx512Fma,
    Avx512NativeBf16,    // VDPBF16PS
    Avx512Vnni,          // vpdpbusd
    Avx512Fp16,          // vaddph
    AmxTile,             // TDPBF16PS tile
    // ARM
    NeonBf16,            // bfdot
    Sve2Predicated,      // whilelt + predicate
    Sme2OuterProduct,    // ZA array + SMOP
    // GPU NVIDIA
    Wmma,                // sm_70
    MmaSync,             // sm_80
    Wgmma,               // sm_90
    Tma2D,               // sm_90
    WarpSpecialization,  // sm_90
    Tcgen05Mma,          // sm_100+
    Tmem,                // sm_100+
    NativeFp4,           // sm_100+
    NativeFp6,           // sm_100+
    // GPU AMD
    WmmaRdna,            // gfx1100+
    WmmaCdna,            // gfx908+
}
```

#### 路由算法

```
for feature in all_features:
    // Step 1: 硬件前提检查
    if !feature.hw_prerequisite_satisfied(profile):
        feature.enabled = false
        feature.reason = "硬件不支持"
        continue

    // Step 2: 成本收益分析
    benefit = estimate_benefit(feature, solver_outputs)
    cost = estimate_cost(feature, solver_outputs)

    // cost 包括: 代码足迹增加、L1i 预算消耗、编译时间
    // benefit 包括: 延迟减少、吞吐提升、带宽节省

    // Step 3: L1i 预算检查
    if total_code_footprint + feature.code_footprint > l1i_budget * 0.8:
        feature.enabled = false
        feature.reason = "L1i 预算不足"
        continue

    // Step 4: 启用
    feature.enabled = true
    feature.reason = "成本收益比: {benefit/cost}"
```

#### 新硬件特性积极使用

FeatureRouter 内置"积极使用"策略——检测到新特性时自动切换更优路径，不需要手动添加 if/else：

| 特性检测 | 自动切换 | 收益估算 |
|---------|---------|---------|
| `has_amx` | `GemmSolver` → `AmxTile` | GEMM 2-4× |
| `has_native_bf16` | `GemmSolver` → `NativeBF16` | pack buffer -50% |
| `sm >= 90` | `AttentionSolver` → `FA3Pipeline` | TMA + WGMMA |
| `sm >= 100` | `AttentionSolver` → `FA4BlockScaled` | TMEM + FP4 |
| `has_sme2` | `GemmSolver` → `SmeOuterProduct` | outer product |

#### 输出

```rust
struct FeaturePlan {
    features: Vec<FeatureDecision>,
    l1i_budget_used: usize,          // 已用 L1i 指令缓存
    l1i_budget_total: usize,         // 总 L1i 预算
    code_sections: Vec<CodeSection>, // .text.hot / .text.warm / .text.cold 分配
    variant_keys: Vec<VariantKey>,   // 所有启用的编译变体
}
```

### 10.10 BatchSolver — 批策略求解器

**职责**: 根据 Golden Bucket、硬件饱和度、Prefill/Decode 混合比例，求解最优批组成策略。

**依赖**: `ParallelismSolver`（wave 分区）、`CacheBudgetSolver`（KV 预算）。

#### 批组成 Token Budget

```
total_budget = max_batch_tokens × memory_pressure_ratio

decode_budget = min(decode_ready_count, floor(total_budget × decode_ratio_cap))
prefill_budget = total_budget - decode_budget

decode_ratio_cap = 0.6  // decode 最多占 60%，保证 prefill 持续进展
```

#### 自适应 Chunk 大小

```
adaptive_chunk_size(l1_ratio, concurrent_reqs, remaining_tokens):

  候选 = GoldenBucketRegistry.golden_sizes()  // 仅限黄金尺寸

  // 按三个维度评分
  for size in 候选:
    score = 0.0
    score += 0.4 × cache_fit_score(size, l1_ratio)
    score += 0.3 × concurrency_score(size, concurrent_reqs)
    score += 0.3 × progress_score(size, remaining_tokens)
    candidates.push((size, score))

  → argmax(score)
```

#### 同号合并策略

```
1. 收集所有 ready 序列
2. 对每个序列: golden_size = registry.collapse(seq.seq_len)
3. 按 golden_size 分组
4. 过小分组（< min_batch_per_wave）合并到最近 bucket
5. 过大分组按 wave 容量拆分
6. 输出 Vec<BatchGroup>
```

#### Compact 决策

```
compact 触发条件 (全部满足):
  waste_ratio > 0.25
  active_count >= min_compact_threshold(4)
  当前 op 为 GEMM（非 Attention）

compact_cost = 2 × active_count × elem_size × cache_line_latency
saved_flops = waste_ratio × total_flops
decision = compact_cost < saved_flops × flops_to_mem_ratio
```

#### 输出

```rust
struct BatchPlan {
    decode_ratio_cap: f32,
    max_chunk_size: usize,
    golden_sizes: Vec<usize>,
    min_compact_threshold: usize,
    compact_waste_threshold: f32,
    decode_slots: usize,
    max_chunks_per_batch: usize,
}

struct BatchGroup {
    golden_size: usize,
    request_ids: Vec<RequestId>,
    slot_type: SlotType,  // Decode / PrefillChunk
}
```

### 10.11 CostModel — 统一成本估算框架

贯穿所有求解器的数学模型，统一估算任意策略配置的执行延迟。

#### 核心公式

```
T_estimated(strategy) = max(T_compute, T_memory) + T_launch + T_sync

T_compute = FLOPs / (peak_FLOPS × efficiency(reg_util, cache_hit))
T_memory  = bytes_accessed / (peak_bandwidth × (1 - cache_miss_rate))
T_launch  = kernel_launch_overhead × num_kernels
T_sync    = synchronization_cost(wave_count, barrier_type)
```

#### 效率因子推导

```
reg_util = accumulator_regs / total_regs  // 寄存器利用率
cache_hit = min(1.0, working_set / cache_size)  // 缓存命中率

efficiency(reg_util, cache_hit) = base_efficiency
    × reg_efficiency_curve(reg_util)
    × cache_efficiency_curve(cache_hit)

// reg_efficiency_curve:
//   reg_util < 0.5 → 线性增长
//   0.5 ~ 0.8 → 平稳高区
//   > 0.8 → 下降（寄存器溢出风险）

// cache_efficiency_curve:
//   cache_hit > 0.9 → 接近 1.0
//   0.5 ~ 0.9 → 线性
//   < 0.5 → 骤降（thrashing）
```

#### 融合收益估算

```
fusion_savings(fusion_mode, ops):
  savings = sum(individual_memory_access(op) for op in ops) - fused_memory_access(ops)

  EpilogueInjection:  savings = (N-1) × output_bytes  // N 个 op 的中间写回消除
  TileLevelFusion:    savings = (N-1) × output_bytes × (1 - L1_hit_rate)  // L1 命中部分无收益
  LoopFusion:         savings = (N-1) × output_bytes  // 逐元素链中间写回消除
  ComputeRoot:        savings = 0  // 无融合收益，但也不付出融合成本
```

#### Kernel Launch 开销模型

| 平台 | 单次 launch | 说明 |
|------|-----------|------|
| CPU | 0 ns | 函数调用，无 launch 开销 |
| GPU sm_80 | 2-5 μs | cuLaunchKernel driver 开销 |
| GPU sm_90 | 1-3 μs | persistent kernel 模式可分摊 |
| GPU sm_100+ | 0.5-2 μs | thread block cluster 降低调度开销 |

### 10.12 ConstraintSpace — 约束空间建模

所有求解器共享的约束系统，防止独立决策互相矛盾。

#### 约束类型

```
硬约束（违反则候选直接淘汰）:
  R1: acc_regs + ptr_regs(2) + scratch_min(2) ≤ num_simd_regs
  R2: tile_a + tile_b ≤ L1 × 75%
  R3: kv_pages ≤ max_pages
  R4: wave_count ≤ sm_partitions
  R5: l1i_code_footprint ≤ L1i × 80%
  R6: smem_usage ≤ smem_per_block

软约束（可违反，按惩罚系数计入成本）:
  S1: epilogue_depth ≥ required_depth (惩罚: ×1.5 per missing op)
  S2: k_depth ≥ recommended (惩罚: ×1.2)
  S3: wave_count ≥ 2 for compute_bound (惩罚: ×1.3 for single_wave)
```

#### 约束传播

当子求解器修改共享资源时，更新全局约束状态：

```
GemmSolver 选择 MR×NR → 消耗 accumulator_regs → 更新 FusionSolver 的 epilogue_budget
CacheBudgetSolver 分配 L2 → 更新 GemmSolver 的 l2_available
ParallelismSolver 分配 SM → 更新 BatchSolver 的 min_batch_tokens
```

### 10.13 Solver DAG — 求解依赖图与执行顺序

求解器之间形成 DAG，按拓扑序执行保证一致性：

```
层级 0 (无依赖):
  RooflineAnalyzer
  CacheBudgetSolver

层级 1 (依赖层级 0):
  GemmSolver ← RooflineAnalyzer + CacheBudgetSolver
  AttentionSolver ← CacheBudgetSolver + RooflineAnalyzer

层级 2 (依赖层级 0+1):
  FusionSolver ← GemmSolver + CacheBudgetSolver + RooflineAnalyzer
  ParallelismSolver ← GemmSolver + CacheBudgetSolver

层级 3 (依赖层级 0+1+2):
  BatchSolver ← ParallelismSolver + CacheBudgetSolver

层级 4 (依赖全部):
  FeatureRouter ← 全部求解器输出
```

```rust
impl HwOptEngine {
    pub fn solve(
        profile: &DeviceProfile,
        sensors: &MemoryNetworkSensors,
        probe: &ProbeResult,
        model: &ModelConfig,
    ) -> Result<HwOptPlan, OptError> {
        // Level 0
        let roofline = RooflineAnalyzer::analyze(profile, model);
        let cache_plan = CacheBudgetSolver::solve(profile, sensors, model);

        // Level 1
        let gemm_plan = GemmSolver::solve(profile, &roofline, &cache_plan);
        let attn_plan = AttentionSolver::solve(profile, &cache_plan, &roofline);

        // Level 2
        let fusion_plan = FusionSolver::solve(&gemm_plan, &cache_plan, &roofline);
        let parallel_plan = ParallelismSolver::solve(profile, &gemm_plan, &cache_plan);

        // Level 3
        let batch_plan = BatchSolver::solve(&parallel_plan, &cache_plan, model);

        // Level 4
        let feature_plan = FeatureRouter::route(
            profile, &gemm_plan, &fusion_plan, &attn_plan,
            &parallel_plan, &batch_plan,
        );

        Ok(HwOptPlan {
            roofline,
            gemm: gemm_plan,
            cache: cache_plan,
            fusion: fusion_plan,
            attention: attn_plan,
            parallel: parallel_plan,
            batch: batch_plan,
            features: feature_plan,
        })
    }
}
```

### 10.14 HwOptPlan — 不可变输出契约

所有子求解器输出的聚合体。模型加载时计算一次，推理全程只读。

```rust
pub struct HwOptPlan {
    pub roofline: RooflineResult,
    pub gemm: GemmPlan,
    pub cache: CacheBudgetPlan,
    pub fusion: FusionPlan,
    pub attention: AttentionPlan,
    pub parallel: ParallelPlan,
    pub batch: BatchPlan,
    pub features: FeaturePlan,
}
```

**不可变性保证**: `HwOptPlan` 的所有字段均为深不可变（无 `Cell`/`RefCell`/`Atomic`）。推理热路径只读访问，零同步开销。

**生命周期**: 与 `ModelJitCache`（§9）一致——模型加载时创建，模型卸载时销毁。

**缓存**: `HwOptPlan` 按 `ProbeResult.device_fingerprint + model_id` 缓存，相同硬件+模型组合不重复求解。

### 10.15 下游消费映射表

| 消费方 | 消费的 Plan 字段 | SPEC 位置 |
|--------|-----------------|----------|
| JIT FusionEngine | `fusion.max_epilogue_depth`, `fusion.ffn_strategy`, `fusion.fusions` | `01-JIT-PIPELINE.md` §4 |
| JIT Phase 3 Codegen (x86_64) | `gemm.mr/nr`, `gemm.k_depth`, `gemm.pf_distance`, `gemm.strategy`, `features[Avx512*]` | `01-JIT-PIPELINE.md` §5 |
| JIT Phase 3 Codegen (AArch64) | `gemm.mr/nr`, `attention.variant`, `features[Sve2/Sme2]` | `01-JIT-PIPELINE.md` §5 |
| JIT Phase 3 Codegen (GPU) | `attention.variant`, `attention.tile_size`, `features[Wgmma/Tma/etc]` | `01-JIT-PIPELINE.md` §5 |
| ContinuousBatcher | `batch.decode_ratio_cap`, `batch.max_chunk_size`, `batch.decode_slots` | `06-RUNTIME.md` §3 |
| ChunkedPrefillScheduler | `batch.golden_sizes`, `batch.max_chunks_per_batch` | `06-RUNTIME.md` §6 |
| AdaptiveChunkPolicy | `batch.max_chunk_size`, `cache.l2_kv_budget` | `06-RUNTIME.md` §6.4 |
| GoldenBucketRegistry | `roofline`, `cache.hbm_max_pages` | `06-RUNTIME.md` §12 |
| VariantRegistry | `features.features`, `features.l1i_budget_used` | `06-RUNTIME.md` §13 |
| WaveScheduler | `parallel.wave_count`, `parallel.gpu_sm_partition`, `parallel.numa_bindings` | `06-RUNTIME.md` §14 |
| HardwareSaturation | `parallel.min_batch_tokens_per_wave` | `06-RUNTIME.md` §14.2 |
| BatchSameLengthGrouping | `batch.golden_sizes`, `parallel.wave_count` | `06-RUNTIME.md` §14.3 |
| CompactDecision | `batch.min_compact_threshold`, `batch.compact_waste_threshold` | `06-RUNTIME.md` §6.5 |
| EpilogueBuilder | `fusion.max_epilogue_depth`, `gemm.scratch_regs` | `04-OPERATORS.md` §13.3 |
| GateSkipCallback | `roofline.op_class(FFN)`, `features.scratch_available` | `05-OPTIMIZATIONS.md` §2.1 |
| EarlyExitCallback | `roofline.op_class(Layer)`, `features[Avx512Fma]` | `05-OPTIMIZATIONS.md` §2.3 |
| MoEThermalManager | `cache.l2_kv_budget`, `parallel.wave_count` | `05-OPTIMIZATIONS.md` §9 |
| SpecDecodingState | `parallel.wave_count`, `batch.decode_slots` | `05-OPTIMIZATIONS.md` §2.8 |

**消费约束**: 下游模块**禁止**直接读取 `DeviceProfile` 做策略决策。所有策略决策必须通过 `HwOptPlan` 获取。`DeviceProfile` 仅作为探测数据源，不作为决策依据。

### 10.16 错误处理

所有求解错误必须传播，禁止使用默认值绕过：

| 错误类型 | 场景 |
|---------|------|
| `OptError::NoFeasibleCandidate(solver)` | 约束空间无可行解（硬件资源严重不足） |
| `OptError::L1iBudgetExceeded` | 启用的特性代码足迹超出 L1i 80% |
| `OptError::InconsistentPlan` | 子求解器输出互相矛盾（DAG 约束传播失败） |
| `OptError::CacheBudgetExhausted` | 缓存预算不足以放置模型权重 + KV |

### 10.17 禁止事项

| 禁止 | 原因 |
|------|------|
| 下游模块直接读取 `DeviceProfile` 做策略决策 | 绕过引擎统一求解，导致决策矛盾 |
| 硬编码 `if has_xxx → path_a else path_b` 查表 | 必须通过候选枚举 + CostModel 评估 |
| `HwOptPlan` 运行时修改 | 与 JIT 缓存协议一致，推理期间策略不可变 |
| 单个求解器不检查全局约束 | 约束传播保证一致性，独立决策可能越界 |
| FeatureRouter 在 Mega-Kernel 内执行 | 特性路由发生在 Dispatch-Time（build_batch），不在热路径 |
