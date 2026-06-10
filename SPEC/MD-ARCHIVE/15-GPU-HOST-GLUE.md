# GPU Mega-Kernel Host 执行胶水 (ARCH-GPU-HOST-GLUE)

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
GPU kernel launch 与分布式通信协同:
<a data-xref-id="REQ-DP-006" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-006">REQ-DP-006</a>
(P2P 页传输) |
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr 扩展) — 通信 VmInstr 通过本文件 GPU launch 路径执行 |
<a data-xref-id="REQ-SMPART-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-SMPART-001">REQ-SMPART-001</a>
(专用通信 SM 线程) — 通信 thread block 通过本文件 stream 编排 launch
</div>

## 定位

gllm 层 GPU 执行路径的实现。将已编译的 mega-kernel PTX/HIP/MSL 代码 launch 到 GPU 设备。GPU mega-kernel 与 CPU mega-kernel 共享同一 `MegaKernelExecutor` 编译产物，区别仅在执行层。

## 前置原则

- **ARCH-CPU-GPU-UNIFIED**: GPU 走与 CPU 相同的 `compile()` 编译路径（SPEC/39 统一编译器架构）。GPU PTX/HIP/MSL 由 gllm-kernels `gpu_lower.rs` 生成。编译器不假设图结构——喂什么编译什么
- **ARCH-RUST-IS-CODEGEN**: 推理时 Rust 只做一次 `cuLaunchKernel`。GPU 上一次 kernel launch 完成整个 generate loop
- **NO_SILENT_FALLBACK**: GPU 不可用时返回 `Err`，不降级到 CPU

## 架构

```
                    ┌─────────────────────────┐
                    │ MegaKernelExecutor       │
                    │ (compile, SPEC/39)       │
                    └─────────┬───────────────┘
                              │
                    ┌─────────▼───────────────┐
                    │ exec_code (CompiledLayer) │
                    │ .code_bytes() = PTX bin  │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │ CPU           │ GPU            │
              │ entry_fn()    │ cuLaunchKernel │
              │ 直接 CALL     │ weight_gpu_ptr │
              │               │ scratchpad_gpu │
              └───────────────┴───────────────┘
```

## 现有基础

| 组件 | 文件 | 状态 |
|------|------|------|
| PTX 编译缓存 | `cuda_backend.rs:compiled_ptx` | ✅ |
| GPU kernel launch | `cuda_backend.rs:gpu_launch_mega_kernel` | ✅ |
| GPU 权重上传 | `cuda_backend.rs:upload_weight_blob` | ✅ |
| GPU scratchpad/上传/下载 | `cuda_backend.rs` helper methods | ✅ |
| KV cache GPU 分配 | `gpu_helpers.rs:gpu_alloc_kv_cache` | ✅ |
| KV swap in/out | `gpu_helpers.rs:gpu_swap_out/in_pages` | ✅ |
| GPU 采样 | `gpu_helpers.rs:gpu_sample_from_tensor` | ✅ |
| Mega-kernel PTX 生成 | `gllm-kernels gpu_lower.rs` | ✅ |
| 6 个 gpu_pure 方法 | `gpu_backend_macro.rs` | ✅ 全部已实现 |
| prepare_gpu_mega_kernel | `Backend` trait + macro | ✅ executor 调用上传 |
| **GPU PTX 编译接入** | `mega_kernel.rs` | ✅ clone graph + GPU 编译 |
| **Forward-only GPU 编译** | 已物理删除 | 统一走 `compile()` GPU 变体 (SPEC/39) |
| **HIP/Metal launch** | `hip_backend.rs`/`metal_backend.rs` | ✅ HIP hipModuleLaunchKernel + Metal launch_compute |

## 关键设计

### GPU mega-kernel 执行 vs CPU mega-kernel 执行

```
CPU: entry_fn(input_ids_ptr, weight_blob_ptr, kv_cache, positions, ...,
              scratchpad_ptr, output_tokens_ptr, ...)
     → 直接函数调用，所有指针都是 host 内存

GPU: cuLaunchKernel(kernel_fn,
     grid_dim, block_dim, shared_mem_bytes, stream,
     [input_ids_gpu, weight_blob_gpu, kv_cache_gpu, positions_gpu, ...,
      scratchpad_gpu, output_tokens_gpu, ...])
     → GPU kernel launch，所有指针都是 device 内存
```

核心差异：
1. **weight_blob 必须上传到 GPU**：`htod(weight_blob) → weight_gpu_ptr`
2. **scratchpad 必须分配在 GPU**：`device.alloc(scratchpad_bytes) → scratchpad_gpu_ptr`
3. **input_ids 必须上传**：`htod(input_ids) → input_ids_gpu_ptr`
4. **output_tokens 必须从 GPU 取回**：`dtoh(output_tokens_gpu) → host_vec`
5. **positions 必须上传**：`htod(positions) → positions_gpu_ptr`

### 权重上传策略

- **首次调用时上传**：weight_blob 是只读的，首次 `generate`/`forward` 时 `htod` 一次，缓存 device ptr
- **后续调用复用**：device 端 weight_blob 指针不变，只需上传 input_ids 和分配 scratchpad
- **生命周期**：权重 GPU buffer 绑定到 `CudaBackend` 实例生命周期

---

## REQ 清单

### REQ-GPU-010: GPU PTX 编译接入 MegaKernelExecutor

在 mega-kernel 编译时同步生成 GPU PTX/HIP 代码，存入 `gpu_code` 字段。

**问题**: 统一 `compile()` 入口（SPEC/39）需要同时生成 CPU 和 GPU JIT 代码，存入 `CompileOutput`。

**设计**:
- `compile()`: 编译时同时调用 CPU 和 GPU 路径（通过 `CompileConfig` 的 `sm_version`/`gfx_arch` 字段决定是否生成 GPU 代码）
- encoder 统一走 `compile()`，GPU 编译使用同一入口（SPEC/39）
- SM version 从哪里来：executor 编译时从 `backend.device_info().sm_version()` 获取，传入 `CompileConfig`
- `CompileConfig` 包含 `sm_version: Option<u32>` 字段（替代 `MegaKernelConfig`，SPEC/39 §1.1.1）
- HIP: `gfx_arch` 和 `wave_size` 从 HipBackend.gpu_profile 获取
- Metal: 从 MetalBackend.gpu_profile 获取

**调用链**:
```
executor.rs compile
  → MegaKernelExecutor::compile(graph, ..., config)
    → graph.clone()
    → compiler.compile(graph, &config)          // CPU (SPEC/39 统一编译入口)
    → compiler.compile(graph_clone, &gpu_config) // GPU (SPEC/39 统一编译入口)
    → mega_compiled.gpu_code = Some(gpu_output.gpu_code)
  → backend.prepare_gpu_mega_kernel(wb, mk.gpu_code(), ...)
    → CudaBackend: upload weight_blob + cache PTX in compiled_ptx
```

**关键文件**:
- `gllm/mega_kernel.rs`: clone graph + GPU 编译 + set_gpu_code
- `gllm/executor.rs`: 传 sm_version 到 CompileConfig (SPEC/39 §1.1.1)
- `gllm-kernels/mod.rs`: 统一 `compile()` 入口，GPU 变体内置 (SPEC/39)

### ~~REQ-GPU-011: Forward-only GPU 编译~~ 已物理删除

> **已物理删除** (SPEC/39 统一编译器架构): encoder GPU 编译走同一 `compile()` 入口。编译器不假设图结构，不存在"forward-only"编译路径。10-param `CompiledLayerFn` ABI 已删除。

### REQ-GPU-012: HIP/Metal Mega-Kernel Launch ✅

实现 HIP 和 Metal 后端的 `gpu_launch_mega_kernel` 及 `GpuEncoderOps`。

**设计**:
- HIP: `HipDevice::load_hsaco` (HipModule + get_function) + `launch_kernel` (hipModuleLaunchKernel)
- Metal: `MetalDevice::load_library_data` (AIR bitcode) + `launch_compute` (MTLComputeCommandEncoder + dispatchThreadgroups)
- 23-param ABI 参数布局与 CUDA 完全一致（全为 device 指针）
- `GpuEncoderOps` trait 三后端统一实现 encoder forward

**关键文件**:
- `gllm-kernels/gpu/hip/device.rs`: HipModule + load_hsaco + launch_kernel
- `gllm-kernels/gpu/metal/device.rs`: launch_compute
- `gllm/hip_backend.rs:gpu_launch_mega_kernel`
- `gllm/metal_backend.rs:gpu_launch_mega_kernel`
- `gllm/gpu_backend_macro.rs:GpuEncoderOps` impl for HIP/Metal

> **执行路径说明**: GPU 后端通过这些 Backend 方法执行推理。CPU 后端由 Executor 直接调用 MegaKernelExecutor (mega-kernel single CALL)，绕过 Backend trait。

### REQ-GPU-001: batch_forward_gpu_pure ✅

实现 GPU decoder forward，用于 continuous batching 调度路径。

**设计**:
- `gpu_backend_macro.rs` 的 `batch_forward_gpu_pure` 方法
- Launch mega-kernel PTX 到 GPU，传入 batch 维度参数
- KV cache 使用 device ptr（`gpu_alloc_kv_cache` 已分配）
- 返回 `(Vec<LogitsHandle>, sparsity, telemetries)`
- LogitsHandle 中的 data 在 GPU 上，通过 `gpu_sample_from_tensor` 采样

**调用方**: `executor.rs:step()` L1591, `executor.rs` profiling L2656

### REQ-GPU-002: GPU Embedding Forward (Mega-Kernel) ✅

实现 GPU encoder forward，用于 embedding 推理。

**设计**:
- Launch mega-kernel PTX（encoder 图，无 Argmax/StoreToken ops，见 SPEC/39 §1.3）
- 上传 input_ids 到 GPU
- 执行 forward pass
- DtoH 取回 output buffer (hidden states)
- MeanPool 在 JIT 图内已完成，返回 `Vec<f32>`

**调用方**: `executor.rs` embedding 路径

### REQ-GPU-003: rerank_forward_gpu_pure ✅

实现 GPU reranker forward。

**设计**: 复用 REQ-GPU-002 架构，区别仅在：
- OutputMode = `ClsToken`（取 [CLS] token 位置的 hidden state）
- JIT 图内已包含 ClsToken pooling

### REQ-GPU-004: classify_forward_gpu_pure ✅

实现 GPU classifier forward。

**设计**: 复用 REQ-GPU-002 架构，区别仅在：
- OutputMode = `ClassifyMultiway`
- JIT 图内已包含 label token logits 提取

### REQ-GPU-005: score_tokens_forward_gpu_pure ✅

实现 HR score_tokens GPU 路径。

**设计**:
- 传入 tokens + target_token_ids
- Mega-kernel output_mode_selector = EncodeToLayer + score computation
- 返回 `Vec<f32>` (每个 target token 的 score)

**调用方**: `executor.rs:score_tokens()` L2920

### REQ-GPU-006: encode_at_layer_forward_gpu_pure ✅

实现 Intent/HR mid-layer encode GPU 路径。

**设计**:
- 传入 tokens + anchor_layer
- Mega-kernel 执行到 anchor_layer 截断
- DtoH 取回 hidden states
- 返回 `Vec<f32>`

**调用方**: `executor.rs:encode_at_layer()` L2957

### REQ-GPU-007: GPU 权重上传管理 ✅

一次性 htod 权重 blob + scratchpad 分配。

**设计**:
- `CudaBackend` 新增字段 `weight_blob_gpu: Option<(u64, usize)>` (device_ptr, bytes)
- 首次调用时：`device.alloc(weight_blob_bytes)` + `device.htod(weight_blob)`
- 后续调用：复用 device_ptr
- Scratchpad 每次调用重新分配（大小可能变化）

**关键文件**: `cuda_backend.rs`, `hip_backend.rs`, `metal_backend.rs`

### REQ-GPU-008: GPU Callback 传递 ✅

SG/Guardrail callback table 通过 GPU shared memory 传递。

**设计**:
- SG `SgSharedMemory` 已经是共享内存设计，GPU 上通过 device ptr 传递
- Callback table (`MegaKernelCallbackTable`) 在 GPU 上通过 `LoadCallbackEntry` + `NativeCall` VmInstr 调用
- **约束**: GPU callback 只能调用 `__device__` 函数
- 无 callback 时传 NULL ptr（零开销）

**关键文件**: `cuda_backend.rs`, `gpu_backend_macro.rs`, `mega_kernel.rs`

### REQ-GPU-009: HIP/Metal 对等实现

CUDA GPU launch 已实现。HIP/Metal 的 mega-kernel launch 返回 `Err("not yet implemented (REQ-GPU-009)")`。

**关键文件**: `hip_backend.rs`, `metal_backend.rs`

**关键文件**: `gpu_helpers.rs`, `cuda_backend.rs`, `hip_backend.rs`, `metal_backend.rs`

---

## ABI 映射

### CPU/GPU Mega-Kernel ABI (23 参数, 0-based 索引)

> **SSOT**: 完整 23 参数定义见 `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §1.5.5 (L1117-1141)`。本节为跨后端引用摘要。所有 arg 编号使用 0-based 逻辑索引。

```rust
entry_fn(
    // arg 0-5: 基础推理参数
    input_ids_ptr: *const u32,       // arg 0  — 输入 token IDs
    weight_blob_ptr: *const u8,      // arg 1  — 权重 blob
    kv_cache_ptr: *mut u8,           // arg 2  — KV cache buffer
    positions_ptr: *const u32,       // arg 3  — 位置编码表
    aux_ptr: *const u8,              // arg 4  — 辅助数据
    batch_size: usize,               // arg 5  — 批大小

    // arg 6-13: 序列控制参数
    prompt_len: usize,               // arg 6  — 输入长度
    scratchpad_ptr: *mut u8,         // arg 7  — 临时计算缓冲区
    output_tokens_ptr: *mut u32,     // arg 8  — 输出 token 缓冲区
    temperature_u32: usize,          // 8-byte aligned            // arg 9  — 采样温度 (IEEE 754)
    top_k: usize,                    // 8-byte aligned                      // arg 10 — top-k 参数
    top_p_u32: usize,                // 8-byte aligned                  // arg 11 — top-p 参数 (IEEE 754)
    max_new_tokens: usize,           // arg 12 — 最大生成 token 数
    eos_token_id: usize,             // 8-byte aligned               // arg 13 — 终止 token ID

    // arg 14-22: 高级功能参数
    output_mode_selector: usize,     // 8-byte aligned       // arg 14 — 输出模式 (0-5 JMP Table)
    hook_ctx_ptr: *mut u8,           // arg 15 — Hook/SG 共享内存
    telemetry_ptr: *mut u8,          // arg 16 — 遥测数据
    session_position: usize,         // arg 17 — Session KV 已处理位置
    fused_hidden_ptr: *const u8,     // void* 泛型指针    // arg 18 — 多模态 fused hidden
    num_mm_tokens: usize,            // arg 19 — 多模态 token 数
    callback_table_ptr: *const u8,   // arg 20 — Callback 函数指针表
    page_table_ptr: *const u32,      // arg 21 — PagedAttention 页表
    batch_ctx_ptr: *const u8,        // arg 22 — BatchContext 批量推理上下文
)
```

### GPU Kernel Launch ABI

```c
// CUDA / HIP / Metal — 所有 GPU 后端统一
cuLaunchKernel(
    kernel_fn,
    grid_dim_x, grid_dim_y, grid_dim_z,
    block_dim_x, block_dim_y, block_dim_z,
    shared_mem_bytes,
    stream,
    kernel_params: [*mut void; 23],        // 23 个指针/值参数 (0-based)
    extra: null
)
```

GPU launch 时 23 个参数全部转为 `void*` 数组，每个参数指向 device 内存地址。CPU 路径等价：`entry_fn` 直接按上述签名调用。

---

## 实施顺序

1. ~~REQ-GPU-007 (权重上传)~~ ✅
2. ~~REQ-GPU-008 (Callback 传递)~~ ✅
3. ~~REQ-GPU-001 (batch_forward)~~ ✅
4. ~~REQ-GPU-002 (embedding_forward)~~ ✅
5. ~~REQ-GPU-003/004 (rerank/classify)~~ ✅
6. ~~REQ-GPU-005/006 (score_tokens/encode_at_layer)~~ ✅
7. ~~REQ-GPU-010 (GPU PTX 编译接入)~~ ✅
8. ~~REQ-GPU-011 (Forward-only GPU 编译)~~ 废弃 → 统一到 `compile()` GPU 变体（见 SPEC/39）
9. ~~REQ-GPU-012 (HIP/Metal launch)~~ ✅

---

## 验证

```bash
# 编译检查
cd ../gllm && cargo check

# 单元测试
cargo test --lib

# E2E GPU 测试（需要 GPU 环境）
cargo test --test test_e2e_generator -- --test-threads=1
cargo test --test test_e2e_embedding -- --test-threads=1

# 验证 GPU 不降级
# mega-kernel PTX 编译成功 = GPU 路径可用
# cuLaunchKernel 成功 = GPU 执行正确
# output_tokens 与 CPU 路径数值对齐 = 结果正确
```
