# 4 项立即开发功能 — 详细架构设计

> **生成日期**: 2026-03-15
> **关联 REQ**: REQ-KV-EXT-001, REQ-KV-EXT-002, REQ-KERNELS-IQ-001~008, REQ-KERNELS-GPU-001~003

---

## 功能 1: 自适应 Chunk 大小 (REQ-KV-EXT-001)

### 问题分析

当前 `Executor::step()` 在 L840-857 使用 `kv_cache_config.max_seq_len` 作为 `plan_prefill()` 的 chunk_size。这意味着 prefill 实际上不分块（chunk_size = 4096/8192），`ChunkedConfig::chunk_size`（默认 64）完全未被使用。

高并发场景下，大 chunk 导致 L1 页面耗尽，触发不必要的 eviction。

### 架构方案

#### 新增结构体: `AdaptiveChunkPolicy`

**文件**: `src/scheduler/vllm2024.rs`

```rust
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveChunkPolicy {
    pub min_chunk: usize,   // 下界 = ChunkedConfig::chunk_size (64)
    pub max_chunk: usize,   // 上界 = max_seq_len
}

impl AdaptiveChunkPolicy {
    pub fn new(chunked: &ChunkedConfig, max_seq_len: usize) -> Self {
        Self {
            min_chunk: chunked.chunk_size.max(1),
            max_chunk: max_seq_len.max(chunked.chunk_size),
        }
    }

    /// 根据运行时负载计算最优 chunk_size
    pub fn compute(&self, l1_available_ratio: f32, concurrent_reqs: usize, prompt_len: usize) -> usize {
        // 短 prompt 直接返回
        if prompt_len <= self.min_chunk {
            return prompt_len.max(1);
        }

        let base = if l1_available_ratio < 0.25 {
            // 高负载: 最小 chunk
            self.min_chunk
        } else if l1_available_ratio > 0.75 {
            // 低负载: 最大 chunk
            self.max_chunk
        } else {
            // 中负载: 线性插值
            let t = (l1_available_ratio - 0.25) / 0.50;
            let range = self.max_chunk - self.min_chunk;
            self.min_chunk + (range as f32 * t) as usize
        };

        // 并发惩罚: 每增加一个并发请求，chunk 缩小 10%
        let penalty = 1.0_f32.max(1.0 - 0.1 * (concurrent_reqs.saturating_sub(1)) as f32);
        let adjusted = (base as f32 * penalty.max(0.2)) as usize;

        adjusted.clamp(self.min_chunk, self.max_chunk.min(prompt_len))
    }
}
```

#### 修改: `Executor::step()` 集成

**文件**: `src/engine/executor.rs` (L840-857)

```rust
// 替换:
// let chunk_size = self.kv_cache_config.max_seq_len.max(1);
// 为:
let adaptive = AdaptiveChunkPolicy::new(
    &self.vllm2024_state.config.chunked,
    self.kv_cache_config.max_seq_len,
);
let l1 = self.memory_manager.tier_usage(Tier::L1);
let l1_ratio = if l1.capacity > 0 {
    (l1.capacity - l1.used) as f32 / l1.capacity as f32
} else { 1.0 };
let concurrent = batch.requests.len();
// ... 在 for 循环内:
let chunk_size = adaptive.compute(l1_ratio, concurrent, prompt_len);
```

#### 修改: `Scheduler2024Config` 集成

**文件**: `src/scheduler/vllm2024.rs`

`Scheduler2024Config` 已包含 `chunked: ChunkedConfig`，无需新增字段。`AdaptiveChunkPolicy` 从 `ChunkedConfig` 构造。

### 测试策略

| TEST ID | 测试类型 | 描述 |
|---------|---------|------|
| TEST-KV-EXT-001 | 正向 | 高负载 (L1 < 25%) 返回 min_chunk |
| TEST-KV-EXT-002 | 正向 | 低负载 (L1 > 75%) 返回 max_chunk |
| TEST-KV-EXT-003 | 正向 | 中负载线性插值正确 |
| TEST-KV-EXT-004 | 边界 | prompt_len < min_chunk 返回 prompt_len |
| TEST-KV-EXT-005 | 边界 | 并发惩罚不低于 min_chunk |
| TEST-KV-EXT-006 | 正向 | Executor 集成后 plan_prefill 使用自适应值 |



---

## 功能 2: GPU TileLevelFusion / ComputeRoot (REQ-KERNELS-GPU-001~003)

### 问题分析

`plan_emitter.rs:62-69` 对 `FusionMode::TileLevelFusion` 和 `FusionMode::ComputeRoot` 返回 error。CPU 侧（x86_64.rs / aarch64_dynasm.rs）已完整实现这两种模式。

CPU 实现的核心是 L1 cache tiling：
- TileLevelFusion: Norm 输出按 MC strip 写入 scratchpad，GEMM 从 scratchpad 读取
- ComputeRoot: Norm 全量输出写入 scratchpad，GEMM 从 scratchpad 读取

GPU 需要将 L1 cache → shared memory 映射。

### 架构方案

#### 仓库: `/home/putao/code/rust/gllm-kernels/`

#### 核心映射: CPU L1 → GPU Shared Memory

| CPU 概念 | GPU 等价 |
|----------|---------|
| L1 cache (32-64KB) | Shared memory (48-96KB per SM) |
| Scratchpad buffer | `__shared__` / `threadgroup` 数组 |
| MC loop (row tiles) | Thread block Y 维度 |
| KC loop (col tiles) | Thread block X 维度内循环 |
| 75% L1 阈值 | 75% shared memory 阈值 |

#### 新增: `DeviceProfile::shared_memory_per_block()`

**文件**: `gllm-kernels/src/dispatch/mod.rs`

```rust
impl DeviceProfile {
    pub fn shared_memory_per_block(&self) -> usize {
        match self.backend {
            BackendKind::Cuda => 49152,  // 48KB (default, can query at runtime)
            BackendKind::Hip => 65536,   // 64KB (MI250/MI300)
            BackendKind::Metal => 32768, // 32KB (Apple GPU)
            BackendKind::Cpu => self.cache_sizes().0, // L1
        }
    }
}
```

#### 修改: `plan_emitter.rs` — TileLevelFusion

**文件**: `gllm-kernels/src/compiler/codegen/gpu_ir/plan_emitter.rs`

替换 L62-69 的 error 分支：

```rust
FusionMode::TileLevelFusion { predecessor, tile_rows } => {
    // 1. 找到 predecessor op (RmsNorm)
    let norm_op = graph.op(predecessor);
    let (m, n, k) = extract_gemm_dims(op);

    // 2. 分配 shared memory: tile_rows × k × byte_width
    let smem_bytes = tile_rows * k * byte_width;
    plan.shared_memory_bytes = plan.shared_memory_bytes.max(smem_bytes);

    // 3. 生成 kernel:
    //    Phase 1: 每个 thread block 计算 tile_rows 行 RmsNorm → shared memory
    //    Phase 2: __syncthreads()
    //    Phase 3: GEMM 从 shared memory 读取 A 矩阵
    plan.emit_tiled_norm_gemm(norm_op, op, tile_rows, k, smem_bytes);
}
```

#### 修改: `plan_emitter.rs` — ComputeRoot

```rust
FusionMode::ComputeRoot { predecessor } => {
    let norm_op = graph.op(predecessor);
    let (m, n, k) = extract_gemm_dims(op);
    let norm_bytes = m * k * byte_width;
    let smem_budget = profile.shared_memory_per_block() * 75 / 100;

    if norm_bytes <= smem_budget {
        // Norm 输出放 shared memory
        plan.shared_memory_bytes = plan.shared_memory_bytes.max(norm_bytes);
        plan.emit_compute_root_smem(norm_op, op, m, k, norm_bytes);
    } else {
        // Norm 输出放 global memory (device malloc)
        plan.emit_compute_root_gmem(norm_op, op, m, k, norm_bytes);
    }
}
```

#### 三后端 Shared Memory 语义

| 后端 | 声明 | 同步 |
|------|------|------|
| PTX (CUDA) | `.shared .align 16 .b8 smem[SIZE];` | `bar.sync 0;` |
| HIP C++ | `__shared__ char smem[SIZE];` | `__syncthreads();` |
| MSL (Metal) | `threadgroup char smem[SIZE];` | `threadgroup_barrier(mem_flags::mem_threadgroup);` |

#### Kernel 结构 (TileLevelFusion)

```
// Thread block: (BLOCK_X, BLOCK_Y) = (tile_cols, tile_rows)
// Grid: (ceil(n/BLOCK_X), ceil(m/tile_rows))

kernel void tiled_norm_gemm(
    global int* A,       // pre-norm input [m, k] (quantized)
    global int* B,       // weight [k, n] (quantized)
    global int* norm_w,  // norm weights [k] (quantized)
    global int* C,       // output accumulator [m, n]
    int m, int k, int n
) {
    shared int norm_tile[tile_rows * k];  // shared memory

    // Phase 1: RmsNorm for tile_rows rows
    int row = blockIdx.y * tile_rows + threadIdx.y;
    if (row < m) {
        // 每个 thread 计算一行的 RMS → normalize → 写入 norm_tile
        rms_norm_row(&A[row * k], &norm_w[0], &norm_tile[threadIdx.y * k], k);
    }
    __syncthreads();

    // Phase 2: GEMM tile
    // C[row, col] += sum_i(norm_tile[threadIdx.y * k + i] * B[i * n + col])
    gemm_tile(norm_tile, B, C, tile_rows, k, n);
}
```

### 测试策略

| TEST ID | 测试类型 | 描述 |
|---------|---------|------|
| TEST-KERNELS-GPU-001 | 正向 | TileLevelFusion PTX codegen 不返回 error |
| TEST-KERNELS-GPU-002 | 正向 | ComputeRoot PTX codegen 不返回 error |
| TEST-KERNELS-GPU-003 | 对齐 | GPU TileLevelFusion vs CPU TileLevelFusion 数值一致 (< 1e-5) |
| TEST-KERNELS-GPU-004 | 对齐 | GPU ComputeRoot vs CPU ComputeRoot 数值一致 (< 1e-5) |
| TEST-KERNELS-GPU-005 | 边界 | Norm 输出 > 75% shared memory 时正确选择 ComputeRoot gmem 路径 |
| TEST-KERNELS-GPU-006 | 正向 | HIP codegen 不返回 error |
| TEST-KERNELS-GPU-007 | 正向 | MSL codegen 不返回 error |



## 功能 3: 架构级 Zero-Fallback 原生直通设计 (ARCH-ZERO-FALLBACK)

### 设计目标与原则

Mega-Kernel 架构强制执行 **Fail-Fast** 与 **Zero-Compromise**，严禁在运行时构建幽灵控制流（Ghost Control Flow）或妥协性资源降级（如 GPU OOM 时自动切换 CPU 重试）。所有模块必须将底层执行异常以 `OomHaltError` 直接穿透至顶层客户端。

### 架构方案：直通传递与包装剥离

1. **零降级构建 (Zero-Fallback Context)**
   `BackendContext` 初始化链路中，底层编译器或后端探测若抛出 `ExecutorError`，必须即刻向上传递，绝对禁止基于 "OOM 错误" 执行重新配置和重试流程（如 GPU 构建失败回退至 CPU 构建）。

2. **客户端原生调用 (Client Direct Invocation)**
   客户端 (CLI / Server) 侧引擎调度流须直接与 `Backend::executor_mut()` 交互，原生获取 `generate` / `embed` / `rerank` 句柄完成调用。不再存在具有后台接管与降级逻辑的 `FallbackEmbedder`/`FallbackGenerator` 包装器。

---

## 功能 5: PTX 算法多版本支持 (REQ-KERNELS-PTX-MV-001~005)

### 问题分析

当前 PTX codegen 的 SM 版本分派存在两个结构性问题：

**问题 1: 硬编码 if-else 链**

`trace_emitter.rs:584-597` 中 GEMM 的 SM 分派是硬编码的 if-else：

```rust
if self.sm_version >= 89 {
    emit_gemm_tc_sm89_ptx(out, name, *m, *n, *k);
} else if self.sm_version >= 80 {
    emit_gemm_tc_sm80_ptx(out, name, *m, *n, *k);
} else if self.sm_version >= 70 {
    emit_gemm_tc_sm70_ptx(out, name, *m, *n, *k);
} else {
    // Fallback to tiled GEMM  ← 违反禁止 Fallback 铁律
    return super::kernel_builder::build_gemm_tiled_kernel(...);
}
```

每新增一个 SM 版本变体，都要修改 `emit_gemm_kernel` 函数体。且 `else` 分支是隐式 fallback。

**问题 2: FlashAttention 无 SM 特化**

`build_flash_attention_kernel` 是单一实现，对所有 SM 版本生成相同的 PTX 代码。但 FlashAttention 论文已演进 4 个版本，每个版本利用不同代际的硬件特性：

| FA 版本 | 目标架构 | SM 版本 | 关键硬件特性 | 性能提升 |
|---------|---------|---------|-------------|---------|
| v1 | Volta | sm_70-79 | wmma 16×16×16, tiled online softmax | 基线 |
| v2 | Ampere | sm_80-89 | mma.sync + cp.async, Split-Q 并行, warp partitioning | ~2× over v1 |
| v3 | Hopper | sm_90-99 | TMA + WGMMA, warp specialization (producer/consumer), FP8 | ~1.5-2× over v2 |
| v4 | Blackwell | sm_100+ | TMEM (256KB/SM) + tcgen05.mma, 2-CTA cooperative MMA, LPT scheduling | ~1.3× over v3 |

用同一份 PTX 代码覆盖所有架构，等于放弃了每代硬件 50-100% 的性能提升。

### 架构方案

#### 仓库: `/home/putao/code/rust/gllm-kernels/`

#### 层级关系: PTX 特化内核是 PTX codegen 的下层

```
codegen/
├── gpu_ir/                    # L0: GPU 统一抽象层 (GpuDialect trait)
│   ├── trace_emitter.rs       #     PtxDialect / HipDialect / MslDialect
│   ├── kernel_builder.rs      #     通用 kernel builder (所有后端共享)
│   └── plan_emitter.rs        #     gpu_emit_plan 统一分派
│
├── ptx/                       # L1: PTX codegen 子模块
│   ├── mod.rs                 #     子模块入口 + re-export
│   ├── codegen.rs             #     PtxCodeGen + PtxBackend (从 ptx.rs 迁入)
│   ├── registry.rs            #     L2: 多版本注册表 (SmRange + PtxKernelRegistry)
│   ├── gemm.rs                #     L2: SM 特化 GEMM (从 ptx_gemm.rs 迁入)
│   └── flash_attention.rs     #     L2: SM 特化 FlashAttention (4 版本)
│
├── hip.rs                     # L1: HIP codegen (AMD)
└── air.rs                     # L1: Metal AIR codegen (Apple)
```

**关键设计决策**: SM 特化内核（`gemm.rs`, `flash_attention.rs`）位于 `ptx/` 子目录下，
是 PTX codegen 的**内部实现细节**，而非与 `ptx.rs` 平级的独立模块。原因：

1. 特化内核本质上是普通算子的 SM 版本变体，仍然需要被融合管线（Phase 2）处理
2. `PtxDialect`（在 `gpu_ir/trace_emitter.rs`）通过 `ptx/registry.rs` 查询最优实现
3. `kernel_builder.rs` 提供通用骨架，`ptx/flash_attention.rs` 提供 SM 特化的内部实现
4. 未来 HIP/Metal 如需类似多版本支持，可在各自子目录下建立同构的 registry

#### 核心设计: PtxKernelRegistry + SmRange 调度

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PTX 多版本调度架构                                 │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ CudaDevice   │    │ PtxKernelRegistry│    │ PtxDialect       │  │
│  │              │    │  (ptx/registry)  │    │ (gpu_ir 层)      │  │
│  │ sm_version ──┼───►│ select(algo, sm) │───►│ emit_kernel()    │  │
│  │ (Driver API) │    │                  │    │ (选中的 emitter) │  │
│  └──────────────┘    │ ┌──────────────┐ │    └──────────────────┘  │
│                      │ │ FA-v1 sm70   │ │                          │
│                      │ │ FA-v2 sm80   │ │    ┌──────────────────┐  │
│                      │ │ FA-v3 sm90   │ │    │ 生成的 PTX       │  │
│                      │ │ FA-v4 sm100  │ │    │ .target sm_XX    │  │
│                      │ └──────────────┘ │    │ (版本特化指令)   │  │
│                      │                  │    └──────────────────┘  │
│                      │ 无匹配 → Err()   │                          │
│                      └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 新增类型: `SmRange`

**文件**: `gllm-kernels/src/compiler/codegen/ptx/registry.rs` (新增)

```rust
/// SM 版本范围 [min_sm, max_sm)，左闭右开。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmRange {
    pub min_sm: u32,  // 包含
    pub max_sm: u32,  // 不包含 (u32::MAX 表示无上界)
}

impl SmRange {
    pub const fn new(min_sm: u32, max_sm: u32) -> Self {
        Self { min_sm, max_sm }
    }

    /// 无上界范围: [min_sm, +∞)
    pub const fn from(min_sm: u32) -> Self {
        Self { min_sm, max_sm: u32::MAX }
    }

    pub fn contains(&self, sm: u32) -> bool {
        sm >= self.min_sm && sm < self.max_sm
    }

    pub fn overlaps(&self, other: &SmRange) -> bool {
        self.min_sm < other.max_sm && other.min_sm < self.max_sm
    }
}
```

#### 新增类型: `PtxAlgorithm`

```rust
/// PTX 多版本算法标识符。
/// 每个枚举变体代表一类可多版本特化的算法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxAlgorithm {
    /// FlashAttention (tiled fused QK^T·V with online softmax)
    FlashAttention,
    /// GEMM (矩阵乘法 — 已有 sm70/sm80/sm89 变体，迁移到注册表)
    Gemm,
    // 未来可扩展: Convolution, Reduction, Scan, ...
}
```

#### 核心结构: `PtxKernelRegistry`

```rust
/// PTX 内核 emitter 函数签名。
/// 接收 (out, kernel_name, dialect, params...) → Result<(), String>
pub type PtxKernelEmitter = fn(
    out: &mut String,
    kernel_name: &str,
    dialect: &PtxDialect,
    params: &PtxKernelParams,
) -> Result<(), String>;

/// 算法参数（按算法类型区分）。
#[derive(Debug, Clone)]
pub enum PtxKernelParams {
    FlashAttention {
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        block_n: usize,
    },
    Gemm {
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
    },
}

/// 注册表条目: SM 范围 + emitter 函数 + 版本标签。
struct PtxKernelEntry {
    sm_range: SmRange,
    emitter: PtxKernelEmitter,
    version_tag: &'static str,  // 如 "fa-v1-volta", "fa-v2-ampere"
}

/// PTX 内核多版本注册表。
///
/// 设计约束:
/// - 同一算法的 SM 范围不得重叠
/// - select() 无匹配时返回 Err（禁止 Fallback）
/// - 注册在编译期或 lazy_static 初始化时完成
pub struct PtxKernelRegistry {
    entries: HashMap<PtxAlgorithm, Vec<PtxKernelEntry>>,
}

impl PtxKernelRegistry {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// 注册一个算法的 SM 特化实现。
    /// 如果 SM 范围与已注册的条目重叠，返回 Err。
    pub fn register(
        &mut self,
        algorithm: PtxAlgorithm,
        sm_range: SmRange,
        version_tag: &'static str,
        emitter: PtxKernelEmitter,
    ) -> Result<(), String> {
        let entries = self.entries.entry(algorithm).or_default();
        // 检查 SM 范围重叠
        for existing in entries.iter() {
            if existing.sm_range.overlaps(&sm_range) {
                return Err(format!(
                    "{:?}: SM range [{}, {}) overlaps with existing '{}' [{}, {})",
                    algorithm,
                    sm_range.min_sm, sm_range.max_sm,
                    existing.version_tag,
                    existing.sm_range.min_sm, existing.sm_range.max_sm,
                ));
            }
        }
        entries.push(PtxKernelEntry { sm_range, emitter, version_tag });
        Ok(())
    }

    /// 根据 SM 版本选择最优内核。
    /// 无匹配时返回 Err（禁止 Fallback）。
    pub fn select(
        &self,
        algorithm: PtxAlgorithm,
        sm_version: u32,
    ) -> Result<(&PtxKernelEmitter, &str), String> {
        let entries = self.entries.get(&algorithm).ok_or_else(|| {
            format!("No implementations registered for {:?}", algorithm)
        })?;

        for entry in entries {
            if entry.sm_range.contains(sm_version) {
                return Ok((&entry.emitter, entry.version_tag));
            }
        }

        // 禁止 Fallback: 构造详细错误信息
        let ranges: Vec<String> = entries.iter()
            .map(|e| format!("'{}' [sm_{}, sm_{})", e.version_tag, e.sm_range.min_sm, e.sm_range.max_sm))
            .collect();
        Err(format!(
            "SM {} not supported for {:?}. Registered versions: {}",
            sm_version, algorithm, ranges.join(", ")
        ))
    }
}
```

#### 全局注册表初始化

```rust
use std::sync::OnceLock;

static PTX_REGISTRY: OnceLock<PtxKernelRegistry> = OnceLock::new();

/// 获取全局 PTX 内核注册表（首次调用时初始化）。
pub fn ptx_kernel_registry() -> &'static PtxKernelRegistry {
    PTX_REGISTRY.get_or_init(|| {
        let mut reg = PtxKernelRegistry::new();

        // ── FlashAttention 4 版本 ──
        reg.register(
            PtxAlgorithm::FlashAttention,
            SmRange::new(70, 80),
            "fa-v1-volta",
            emit_flash_attention_v1_volta,
        ).unwrap();
        reg.register(
            PtxAlgorithm::FlashAttention,
            SmRange::new(80, 90),
            "fa-v2-ampere",
            emit_flash_attention_v2_ampere,
        ).unwrap();
        reg.register(
            PtxAlgorithm::FlashAttention,
            SmRange::new(90, 100),
            "fa-v3-hopper",
            emit_flash_attention_v3_hopper,
        ).unwrap();
        reg.register(
            PtxAlgorithm::FlashAttention,
            SmRange::from(100),
            "fa-v4-blackwell",
            emit_flash_attention_v4_blackwell,
        ).unwrap();

        // ── GEMM 迁移（从硬编码 if-else 迁移到注册表）──
        reg.register(
            PtxAlgorithm::Gemm,
            SmRange::new(70, 80),
            "gemm-tc-sm70-wmma",
            emit_gemm_v_sm70,
        ).unwrap();
        reg.register(
            PtxAlgorithm::Gemm,
            SmRange::new(80, 90),
            "gemm-tc-sm80-mma",
            emit_gemm_v_sm80,
        ).unwrap();
        reg.register(
            PtxAlgorithm::Gemm,
            SmRange::from(90),
            "gemm-tc-sm89-fp8",
            emit_gemm_v_sm89,
        ).unwrap();

        reg
    })
}
```

#### FlashAttention 4 版本 Emitter 签名

**文件**: `gllm-kernels/src/compiler/codegen/ptx/flash_attention.rs` (新增)

```rust
//! FlashAttention PTX 多版本内核。
//! 每个版本针对特定 SM 架构深度优化，走完整 JIT 管线。

/// FA-v1 (Volta, sm_70-79): wmma 16×16×16 tiled attention + online softmax
///
/// 硬件特性: wmma.load/wmma.mma/wmma.store
/// Tile: block_m × block_n, QK^T 通过 wmma 计算
/// Softmax: online (running max + sum), 标量路径
pub fn emit_flash_attention_v1_volta(
    out: &mut String,
    kernel_name: &str,
    dialect: &PtxDialect,
    params: &PtxKernelParams,
) -> Result<(), String> { ... }

/// FA-v2 (Ampere, sm_80-89): mma.sync + cp.async + Split-Q 并行
///
/// 硬件特性:
/// - mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 (更灵活的极化 tile)
/// - cp.async.ca.shared.global (异步 global→shared 拷贝)
/// - Split-Q: 外层循环遍历 KV blocks，内层并行多个 Q rows
/// - Warp partitioning: 4 warps 分工 (2 compute + 2 prefetch)
pub fn emit_flash_attention_v2_ampere(
    out: &mut String,
    kernel_name: &str,
    dialect: &PtxDialect,
    params: &PtxKernelParams,
) -> Result<(), String> { ... }

/// FA-v3 (Hopper, sm_90-99): TMA + WGMMA + warp specialization
///
/// 硬件特性:
/// - TMA (Tensor Memory Accelerator): 硬件异步 global→shared 拷贝
/// - WGMMA (Warp Group MMA): 128 线程协作矩阵乘
/// - Warp specialization: producer warps (TMA load) / consumer warps (WGMMA compute)
/// - FP8 block quantization: e4m3/e5m2 输入，F32 累加
/// - Asynchronous barriers: cp.async.bulk + mbarrier
pub fn emit_flash_attention_v3_hopper(
    out: &mut String,
    kernel_name: &str,
    dialect: &PtxDialect,
    params: &PtxKernelParams,
) -> Result<(), String> { ... }

/// FA-v4 (Blackwell, sm_100+): TMEM + tcgen05.mma + 2-CTA cooperative
///
/// 硬件特性:
/// - TMEM (Tensor Memory, 256KB/SM): 片上 scratchpad 直连 Tensor Core
/// - tcgen05.mma: 第 5 代 Tensor Core 异步 MMA，累加到 TMEM
/// - 2-CTA cooperative MMA: 两个 CTA 共享 TMEM 执行单个大 MMA
/// - Software-emulated exponentials: 用 MMA 模拟 exp() 避免 SFU 瓶颈
/// - Conditional softmax rescaling: 仅在 max 变化时 rescale
/// - LPT (Longest Processing Time) scheduling: 非均匀 tile 调度
pub fn emit_flash_attention_v4_blackwell(
    out: &mut String,
    kernel_name: &str,
    dialect: &PtxDialect,
    params: &PtxKernelParams,
) -> Result<(), String> { ... }
```

#### JIT 管线集成点

**修改文件**: `gllm-kernels/src/compiler/codegen/gpu_ir/trace_emitter.rs`

将 `PtxDialect::emit_gemm_kernel` 中的 CachedGQA/FlashV2 分支改为查询注册表：

```rust
// 修改前 (硬编码):
AttentionStrategy::FlashV2 { block_m: _, block_n } => {
    super::kernel_builder::build_flash_attention_kernel(
        self, out, name, *total_seq, *num_heads, *head_dim, *block_n,
    )?;
}

// 修改后 (注册表查询):
AttentionStrategy::FlashV2 { block_m: _, block_n } => {
    let registry = ptx_kernel_registry();
    let (emitter, tag) = registry.select(
        PtxAlgorithm::FlashAttention, self.sm_version
    )?;
    let params = PtxKernelParams::FlashAttention {
        seq_len: *total_seq,
        num_heads: *num_heads,
        head_dim: *head_dim,
        block_n: *block_n,
    };
    log::debug!("FlashAttention: selected '{}' for sm_{}", tag, self.sm_version);
    emitter(out, name, self, &params)?;
}
```

同理，GEMM 分支也迁移到注册表（消除 if-else 链）：

```rust
// 修改前:
if self.sm_version >= 89 { emit_gemm_tc_sm89_ptx(...) }
else if self.sm_version >= 80 { emit_gemm_tc_sm80_ptx(...) }
...

// 修改后:
let registry = ptx_kernel_registry();
let (emitter, tag) = registry.select(PtxAlgorithm::Gemm, self.sm_version)?;
let params = PtxKernelParams::Gemm { m: *m, n: *n, k: *k, has_bias };
emitter(out, name, self, &params)?;
```

#### 错误处理: 禁止 Fallback 的执行路径

```
GPU 启动
  │
  ▼
CudaDevice::new() → sm_version = 75
  │
  ▼
JIT 编译 FlashAttention
  │
  ▼
ptx_kernel_registry().select(FlashAttention, 75)
  │
  ├─ sm_75 ∈ [70, 80) → Ok(emit_flash_attention_v1_volta)  ✅
  │
  └─ 假设 sm_60 (无注册):
     → Err("SM 60 not supported for FlashAttention.
            Registered versions:
            'fa-v1-volta' [sm_70, sm_80),
            'fa-v2-ampere' [sm_80, sm_90),
            'fa-v3-hopper' [sm_90, sm_100),
            'fa-v4-blackwell' [sm_100, +∞)")
     → 传播到调用方 → 用户看到明确错误  ❌ 不降级
```

#### 与现有 GEMM 多版本的关系

现有 `ptx_gemm.rs` 中的 `emit_gemm_tc_sm70_ptx` / `emit_gemm_tc_sm80_ptx` / `emit_gemm_tc_sm89_ptx` 三个函数保持不变，但包装为 `PtxKernelEmitter` 签名后注册到 `PtxKernelRegistry`。这是一次**无行为变更的重构**——调度逻辑从 if-else 迁移到注册表，但生成的 PTX 代码完全相同。

### 测试策略

| TEST ID | 测试类型 | 描述 |
|---------|---------|------|
| TEST-PTX-MV-001 | 正向 | `SmRange::contains()` 边界测试 (min/max/中间值) |
| TEST-PTX-MV-002 | 正向 | `SmRange::overlaps()` 检测重叠范围 |
| TEST-PTX-MV-003 | 负向 | 注册重叠 SM 范围时返回 Err |
| TEST-PTX-MV-004 | 正向 | `select(FlashAttention, 75)` → fa-v1-volta |
| TEST-PTX-MV-005 | 正向 | `select(FlashAttention, 80)` → fa-v2-ampere |
| TEST-PTX-MV-006 | 正向 | `select(FlashAttention, 90)` → fa-v3-hopper |
| TEST-PTX-MV-007 | 正向 | `select(FlashAttention, 100)` → fa-v4-blackwell |
| TEST-PTX-MV-008 | 负向 | `select(FlashAttention, 60)` → Err (禁止 Fallback) |
| TEST-PTX-MV-009 | 正向 | FA-v1 生成的 PTX 包含 `wmma.load` / `wmma.mma` 指令 |
| TEST-PTX-MV-010 | 正向 | FA-v2 生成的 PTX 包含 `mma.sync` / `cp.async` 指令 |
| TEST-PTX-MV-011 | 正向 | FA-v3 生成的 PTX 包含 `wgmma` / `cp.async.bulk` 指令 |
| TEST-PTX-MV-012 | 正向 | FA-v4 生成的 PTX 包含 `tcgen05.mma` 指令 |
| TEST-PTX-MV-013 | 回归 | GEMM 迁移到注册表后，sm70/sm80/sm89 生成的 PTX 与迁移前 byte-identical |
| TEST-PTX-MV-014 | 集成 | `PtxDialect::emit_gemm_kernel` CachedGQA/FlashV2 路径通过注册表分派 |



---

## 功能 6: 深度算子融合 (REQ-FUSION-DEEP-001~006)

### 问题分析

当前融合管线 (`fusion.rs`) 的融合深度为 2-5 ops，存在三个结构性瓶颈：

1. **单输出 ABI**: JIT ABI 只有一个 `output` 指针，无法生成多输出 kernel（如 Q/K/V 三路投影）
2. **Opaque 分类屏障**: `MHA`/`CachedGQA`/`RoPE` 被分类为 `Opaque`，融合引擎完全跳过
3. **寄存器压力限制**: AVX2 仅 16 个 ymm 寄存器，epilogue 链被限制在 ≤4 ops

目标：将典型 Transformer decoder layer 的融合深度从 2-5 ops 提升到 6-12 ops，消除 60%+ 的中间张量写回。

### 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deep Fusion Pipeline                         │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ FusionRule│──▶│ Pattern  │──▶│ HW-Aware │──▶│ FusionPlan  │ │
│  │ Registry  │   │ Matcher  │   │ Validator│   │ (deep groups)│ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘ │
│       ▲                                              │         │
│       │ rules: AttentionBlock / FFNBlock / ...        ▼         │
│  ┌──────────┐                              ┌─────────────────┐ │
│  │DeviceProf│                              │ Multi-Output    │ │
│  │ ile query│                              │ Codegen (ABI v2)│ │
│  └──────────┘                              └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### DEEP-001: 多输出 ABI



```rust
/// 扩展 JIT ABI (TurboQuant 泛化多输出签名):
extern "C" fn kernel_v2(
    input: *const u8,       // TurboQuant packed 或者是对应物理指针
    weight: *const u8,      // 量化权重
    output_ptrs: *const *mut u8,  // output_ptrs[0] = Q, [1] = K, [2] = V
    num_outputs: usize,
    dim0: usize,
    ...
)
```

**CompilerGraph 变更**:
- `CompilerOp.outputs: Vec<TensorId>` 已支持多输出（无需改动）
- 新增 `MultiOutputMarker` trait 标记需要多输出 codegen 的 FusionGroup

**Codegen 变更** (`x86_64.rs` / `aarch64_dynasm.rs` / `gpu_ir/`):
- `emit_prologue()`: 从 `output_ptrs` 基址 + 偏移加载各输出指针到寄存器
- `emit_store()`: 根据 output index 选择目标寄存器
- AVX2: 用 `r12`/`r13`/`r14` 保存额外输出指针（`r15` 已用于 scratch）
- AVX-512: 用 `r12`-`r15` 保存最多 4 个输出指针
- GPU: threadblock 内通过 shared memory 或直接 global store 到不同地址

**兼容性**: 单输出 kernel 仍使用旧 ABI（`output_ptrs[0]` 等价于原 `output`），零开销。

**文件**: `gllm-kernels/src/compiler/codegen/x86_64.rs`, `aarch64_dynasm.rs`, `gpu_ir/trace_emitter.rs`

### DEEP-002: Attention 可融合化

#### 现状

`semantics.rs:68`: `CachedGQA`/`MHA`/`RoPE` → `OpSemantics::Opaque`
`fusion.rs:527-539`: `Opaque → Standalone`（永远不融合）

#### 方案: Attention 分解 + 新语义分类

**Step 1**: 新增 `OpSemantics::Attention` 分类

```rust
pub enum OpSemantics {
    Elementwise,
    Gemm,
    Reduction,
    Attention,  // 新增: 可与前驱/后继有限融合
    Opaque,     // 仅保留 Transpose/Reshape
}
```

**Step 2**: RoPE 重分类为 `Elementwise`

RoPE 本质是逐元素旋转变换 `(x * cos - y * sin, x * sin + y * cos)`，不涉及 reduction。
```rust
OpKind::RoPE { .. } => OpSemantics::Elementwise,  // 从 Opaque 改为 Elementwise
```

**Step 3**: Attention 前驱融合规则

```
RmsNorm → Q_Gemm → RoPE(Q) ─┐
RmsNorm → K_Gemm → RoPE(K) ─┼─→ CachedGQA → O_Gemm → Residual
         V_Gemm ─────────────┘
```

融合策略:
- **Pre-Attention Block**: `[RmsNorm, Q_Gemm, K_Gemm, V_Gemm, RoPE_Q, RoPE_K]` → 6 ops 融合
  - 利用 QkvSharedInput 模式: 三个 GEMM 共享 RmsNorm 输出（单次 pack_a）
  - RoPE 作为 Q/K GEMM 的 epilogue 注入
- **Post-Attention Block**: `[CachedGQA, O_Gemm, Residual]` → 3 ops 融合
  - Attention 输出直接喂入 O_Gemm（无中间写回）
  - Residual Add 作为 O_Gemm epilogue

**fusion.rs 变更**:
- `fuse()` 的 `match sem` 新增 `OpSemantics::Attention` 分支
- Attention op 可被前驱 GEMM 组吸收（作为 epilogue 的 "consumer"）
- Attention op 的输出可喂入后继 GEMM（作为 NormIntoGemm 的变体）

**文件**: `semantics.rs`, `fusion.rs`, `hw_constraints.rs`

### DEEP-003: 深度 Epilogue 链

#### 现状

`hw_constraints.rs:267`: `epilogue_scratch = group.epilogue.len().min(4)` — 硬限制 4 ops
`hw_constraints.rs:83`: `max_depth = 16` — 理论上限 16，但寄存器压力在 4 ops 时已接近 AVX2 上限

#### 方案: 分层寄存器策略

**AVX2 (16 ymm)**:
- GEMM 微核占用 ~12 regs (6×2 累加器 + A panel + B broadcast)
- 剩余 4 regs → 4 ops epilogue（当前上限）
- **扩展**: epilogue 第 5-8 个 op 使用栈溢出 (spill/reload)
  - 每个溢出 op: `vmovaps [rsp+offset], ymm` + `vmovaps ymm, [rsp+offset]`
  - 代价: 每 op 额外 ~2 cycles（L1 命中），远低于独立 kernel 的内存往返

**AVX-512 (32 zmm)**:
- GEMM 微核占用 ~24 regs (6×4 累加器 + A panel + B broadcast)
- 剩余 8 regs → 8 ops epilogue 无溢出
- 利用 `zmm24`-`zmm31` 作为 epilogue scratch

**AArch64 NEON (32 v-regs)**:
- 类似 AVX-512 策略，8 ops 无溢出

**GPU**:
- 寄存器文件 ≥256 regs/thread → epilogue 深度不受限（由 occupancy 约束）

**Cost Model 更新** (`fusion.rs:estimate_fusion_cost`):
```rust
FusionMode::EpilogueInjection => {
    let nr = profile.gemm_blocking(0, 0, 0).nr;
    let mr = profile.gemm_blocking(0, 0, 0).mr;
    let acc = (mr * nr) / (profile.simd_width_bytes() / 4);
    let avail = profile.num_simd_regs();
    let free_regs = avail.saturating_sub(acc + 2); // A panel + B broadcast
    let no_spill = group.epilogue.len().min(free_regs);
    let spill_ops = group.epilogue.len().saturating_sub(free_regs);
    // 溢出 ops 的代价: 2 cycles × simd_width per spill
    no_spill + spill_ops  // 总 epilogue scratch
}
```

**文件**: `hw_constraints.rs`, `fusion.rs`, `codegen/x86_64.rs`, `codegen/aarch64_dynasm.rs`

### DEEP-004: FFN 全融合

#### 现状

SwiGLU FFN 当前被拆为 5 个独立 kernel:
```
Gate_Gemm → SiLU → Up_Gemm → Mul → Down_Gemm
```

每个 GEMM 之间有完整的中间张量写回（`seq_len × intermediate_size × 4 bytes`）。

#### 方案: 两阶段融合

**Stage 1: Gate+Up 共享输入融合**

复用 `QkvSharedInput` 模式:
```
Input ──┬── Gate_Gemm → SiLU ──┐
        └── Up_Gemm ───────────┼── Mul → (intermediate)
```
- Gate_Gemm 和 Up_Gemm 共享 `pack_a`（输入相同）
- SiLU 作为 Gate_Gemm 的 epilogue
- Mul 作为 Up_Gemm 的 epilogue（读取 Gate_Gemm+SiLU 的输出）
- 需要 DEEP-001 多输出 ABI: 一个 kernel 同时写出 `gate_silu` 和 `up` 两个张量

**Stage 2: Down_Gemm 吸收**

```
Mul_output → Down_Gemm → Residual_Add
```
- Down_Gemm 以 Mul 输出为输入
- Residual_Add 作为 Down_Gemm epilogue
- 如果 Mul 输出 ≤ 75% L1 → ComputeRoot（Mul 完整计算后驻留 L1）
- 如果 Mul 输出 > 75% L1 → TileLevelFusion（Mul 按 MC strip 分块）

**融合后**: 5 个 kernel → 2 个 kernel（Gate+Up+SiLU+Mul 和 Down+Residual）
**消除写回**: `2 × seq_len × intermediate_size × 4 bytes`（Gate 和 Up 的中间输出）

**新增 FusionMode**:
```rust
FusionMode::FFNBlock {
    gate_gemm: OpId,
    up_gemm: OpId,
    activation: OpId,  // SiLU/GeLU
    combine: OpId,     // Mul
}
```

**文件**: `fusion.rs`, `codegen/x86_64.rs`, `codegen/aarch64_dynasm.rs`, `gpu_ir/trace_emitter.rs`

### DEEP-005: 融合规则引擎

#### 现状

`fusion.rs:fuse()` 是一个 500+ 行的 greedy 函数，所有融合逻辑硬编码在 match 分支中。
新增融合模式需要修改核心函数，违反 OCP（开放/封闭原则）。

#### 方案: FusionRule trait + 优先级注册

```rust
/// 声明式融合规则 trait
pub trait FusionRule: Send + Sync {
    /// 规则名称（用于日志和调试）
    fn name(&self) -> &str;

    /// 优先级（数值越大越优先匹配）
    /// 深度融合规则 > 浅融合规则
    fn priority(&self) -> u32;

    /// 尝试在 subgraph 中匹配模式
    /// 返回 Some(FusionGroup) 如果匹配成功
    fn try_match(
        &self,
        graph: &CompilerGraph,
        anchor: OpId,
        profile: &DeviceProfile,
        claimed: &HashSet<OpId>,
    ) -> Option<FusionGroup>;
}
```

**内置规则（按优先级降序）**:

| 优先级 | 规则 | 匹配模式 | 融合深度 |
|--------|------|----------|----------|
| 100 | `AttentionBlockRule` | RmsNorm→QKV_Gemm→RoPE→Attn→O_Gemm→Residual | 9-12 ops |
| 90 | `FFNBlockRule` | Gate+Up_Gemm→SiLU→Mul→Down_Gemm→Residual | 5-6 ops |
| 80 | `CrossLayerResidualRule` | Residual_Add→RmsNorm (跨层) | 2 ops |
| 70 | `QkvSharedInputRule` | 3× GEMM 共享输入 | 3 ops |
| 60 | `NormIntoGemmRule` | RmsNorm→GEMM | 2 ops |
| 50 | `GemmEpilogueRule` | GEMM→Elementwise chain | 2-8 ops |
| 40 | `LoopFusionRule` | Elementwise chain | 2-8 ops |

**FusionEngine 替代 fuse()**:

```rust
pub struct FusionEngine {
    rules: Vec<Box<dyn FusionRule>>,
}

impl FusionEngine {
    pub fn with_defaults() -> Self { ... }

    pub fn fuse(&self, graph: &CompilerGraph, profile: &DeviceProfile) -> FusionPlan {
        let topo = graph.topological_sort();
        let mut claimed = HashSet::new();
        let mut groups = Vec::new();

        // 按优先级降序排列规则
        let sorted_rules = self.sorted_rules();

        for &op_id in &topo {
            if claimed.contains(&op_id) { continue; }
            // 尝试每条规则（高优先级先匹配）
            let matched = sorted_rules.iter().find_map(|rule| {
                rule.try_match(graph, op_id, profile, &claimed)
            });
            if let Some(group) = matched {
                for &oid in &group.ops { claimed.insert(oid); }
                groups.push(group);
            } else {
                // Standalone fallback
                claimed.insert(op_id);
                groups.push(standalone_group(op_id, groups.len()));
            }
        }
        build_plan(groups)
    }
}
```

**文件**: `fusion.rs`（重构）, 新增 `fusion_rules.rs`

### DEEP-006: 跨层残差融合

#### 现状

每个 decoder layer 的输出:
```
... → FFN_Residual_Add → [写回内存] → [下一层读取] → RmsNorm → ...
```

层间有一次完整的 `seq_len × hidden_size × 4 bytes` 写回+读取。

#### 方案: 跨层边 + 流水线融合

**Graph 构建变更** (`decoder_forward.rs`):

当前 `decoder_forward` 逐层构建独立 `CompilerGraph`。改为构建跨层图:

```rust
// 伪代码: 跨层图构建
fn build_cross_layer_graph(layers: &[LayerWeights], config: &ModelConfig) -> CompilerGraph {
    let mut graph = CompilerGraph::new();
    let mut prev_residual_output: Option<TensorId> = None;

    for (i, layer) in layers.iter().enumerate() {
        let input = if let Some(prev) = prev_residual_output {
            prev  // 直接引用上一层的残差输出（无中间写回）
        } else {
            graph.add_tensor("input", ...)
        };

        // Layer i 的 ops...
        let residual_out = build_layer_ops(&mut graph, input, layer);
        prev_residual_output = Some(residual_out);
    }
    graph
}
```

**融合效果**:
- `CrossLayerResidualRule` 匹配: Layer N 的 `Residual_Add` + Layer N+1 的 `RmsNorm`
- 融合为单 kernel: Add 结果直接在寄存器中喂入 RmsNorm（无内存写回）
- 每层节省: `seq_len × hidden_size × 4 × 2 bytes`（写+读）
- 32 层 LLaMA-7B (hidden=4096, seq=2048): 节省 `32 × 2048 × 4096 × 8 = 2 GB` 内存流量

**约束**:
- 跨层图可能过大 → 分段构建（每 4-8 层一个图）
- 需要 `FusionEngine` 支持跨层 op 的拓扑排序

**文件**: `gllm/src/decoder_forward.rs`, `gllm-kernels/src/compiler/fusion.rs`



---

## 功能 7: 全链路审计 — 设备参数 × JIT × 融合 (2026-03-17)

### 审计范围

验证三个维度：
1. ISV 格式全链路 JIT，无 fallback
2. CPU/GPU 设备参数（SM 级别、ISA 特性）影响 JIT 代码生成
3. 融合算法全覆盖，无 fallback

> **审计结果**: 全部通过。详见审计 commit: `f5b885a2`, `6dd2f3e`, `8621a1a7`。

---

## 功能 8：JIT 图通用化执行器 (REQ-JIT-GRAPH-001~003)

> **SSOT**: REQ-JIT-GRAPH-001 (SymDim Symbolic Shape) 和 REQ-JIT-GRAPH-003 (FusedGraph 执行器) 均已完成 🟢。
> SymDim 动态绑定机制的完整定义见 [SPEC/DOCS/scheduling/jit-cache-protocol.md](./DOCS/scheduling/jit-cache-protocol.md) §3。


---

## 功能 9：JIT 编译缓存（已重新设计）

> **⚠️ 2026-03-27 架构演进**: 本功能的原始设计（基于 `DecodeCachedJit` / `GraphKey.graph_type: QRope|Norm2` / `GraphKey.seq_len: usize` 的算子级缓存）已被推翻。
> **新设计 SSOT**: [SPEC/DOCS/scheduling/jit-cache-protocol.md](./DOCS/scheduling/jit-cache-protocol.md) (REQ-JIT-CACHE-001~007)

### 新旧对比

| 维度 | 旧设计（已删除） | 新设计（SSOT） |
|------|:---|:---|
| **缓存键** | `(ModelArchKey, GraphKey{graph_type: QRope\|Norm2, seq_len: usize})` | `ModelArchKey`（不含 seq_len，seq_len 通过 SymDim::Symbolic 运行时绑定） |
| **缓存粒度** | 单算子独立编译（QRope, CachedGQA, Norm2） | 全模型单体计算图（GraphExecutor 一次性编译整网） |
| **编译时机** | 每 decode step 每层重编译 | 模型加载时一次性编译 |
| **动态维度** | `Concrete(seq_len)` → 变化时重编译 | `Symbolic("seq_len")` + `ShapeBinding` |
| **GPU 缓存** | 无（零缓存，每步重编译 PTX/HIP/MSL） | 三级缓存（L1 模型级 → L2 全局 LRU → L3 磁盘） |

---

## 功能 10：量化算子符号化与 JIT 融合 (REQ-JIT-QUANT-001)

> **关联 REQ**: REQ-JIT-QUANT-001 (Symbolic Quantization IR)
> **状态**: 规划中 (📝 Planned)

当前 `gllm-kernels` 中针对 28 种复杂量化格式（如 `IQ2_XXS`, `AWQ4`, `IQ1_S`），JIT 中的 `SemanticDAG` 无法在内部原生表示量化解包逻辑。这导致每个后端的汇编发射器（PTX, x86_64, AMX, Metal）都必须在特征层分别手写 `dequantize_*`。这种实现引发了 O(M×N) 的巨量维护成本，并导致 GPU trait 层产生了大量的 `unimplemented!` fallback。

为了贯穿 **ARCH-JIT-FIRST** 原则，彻底消除硬编码的解包算子负担，我们将对 JIT Compiler 的中间表示 (IR) 进行升级：

### 核心设计 (Symbolic Quantization IR)

1. **IR 指令扩充 (TraceOp Upgrade - TurboQuant 版)**
   在现有图结构之外，为 `TraceOp` 引入底层针对特化精度内联积的硬件指令抽象（彻底消除运行时解包和提升）：
   - `TraceOp::BitAnd`, `TraceOp::BitOr`
   - `TraceOp::BitShiftRight`, `TraceOp::BitShiftLeft`
   - `TraceOp::Gather1D(table_ptr, index)`: 基于常量查表的零消耗掩码寻址。
   - `TraceOp::IntegerInnerProduct`: 张量定点内联积 (原生 VNNI/SDOT 重载支持)。

2. **查找表符号化 (Constant Graph Binding)**
   废弃现有 `traits.rs` 中繁冗的解包分支代码，转而在 JIT 图生成期（SemanticDAG 构图阶段），将诸如 `codebooks.rs` 内部的 `IQ1S_GRID`, `KMASK_IQ2XS` 等庞大表体，作为只读的 `ConstTensor` 节点直接注入计算图。

3. **量化逻辑等效为计算子图 (Sub-Graph Fusion)**
   反量化过程不再是外部黑盒方法，而是由上述 `TraceOp` 构建的一张子计算图。JIT 编译器会对“量化解包子图 + 矩阵乘 GEMM”进行真正的端到端算子融合 (Operation Fusion)。

### 收益
- **Write Once, Run Everywhere**: `gllm-kernels` 的各个后端 Emitter（PTX/x86/AMX）只需要增加对新指令（如 `vpgatherdd` 或 `ld.global`）的机器码翻译能力，28 种量化格式便可自动在所有计算平台上即时编译执行。
- 彻底消灭 `gllm-kernels` 中由 `unimplemented!` 引发的边缘架构量化缺失异常。
