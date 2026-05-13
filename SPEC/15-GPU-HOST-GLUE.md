# GPU Mega-Kernel Host 执行胶水 (ARCH-GPU-HOST-GLUE)

## 定位

gllm 层 GPU 执行路径的实现。将已编译的 mega-kernel PTX/HIP/MSL 代码 launch 到 GPU 设备。GPU mega-kernel 与 CPU mega-kernel 共享同一 `MegaKernelExecutor` 编译产物，区别仅在执行层。

## 前置原则

- **ARCH-CPU-GPU-UNIFIED**: GPU 走与 CPU 相同的 `compile_from_auto_graph` / `compile_forward_from_graph` 编译路径。GPU PTX/HIP/MSL 由 gllm-kernels `gpu_lower.rs` 生成
- **ARCH-RUST-IS-CODEGEN**: 推理时 Rust 只做一次 `cuLaunchKernel`。GPU 上一次 kernel launch 完成整个 generate loop
- **NO_SILENT_FALLBACK**: GPU 不可用时返回 `Err`，不降级到 CPU

## 架构

```
                    ┌─────────────────────────┐
                    │ MegaKernelExecutor       │
                    │ (compile_from_auto_graph) │
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
| **Forward-only GPU 编译** | `gllm-kernels/mod.rs` | ✅ compile_graph_to_gpu |
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

**问题**: `compile_from_auto_graph` 和 `compile_forward_from_graph` 中 `gpu_code: None`，导致 `prepare_gpu_mega_kernel` 传入的 `decoder_gpu_code` 始终为 None，GPU launch 拿不到 PTX。

**设计**:
- `compile_from_auto_graph`: 在 move graph 之前 clone，CPU 编译完后调 `compile_mega_kernel_to_gpu(graph_clone, &config, sm_version)` 获取 GPU 代码
- `compile_forward_from_graph`: graph 是 `&CompilerGraph`（引用），无需 clone。调 `compile_graph_to_gpu(graph, sm_version)` 获取 GPU 代码
- SM version 从哪里来：executor 编译时从 `backend.device_info().sm_version()` 获取，传入 mega-kernel 编译
- `MegaKernelConfig` 新增 `sm_version: Option<u32>` 字段
- HIP: `gfx_arch` 和 `wave_size` 从 HipBackend.gpu_profile 获取
- Metal: 从 MetalBackend.gpu_profile 获取

**调用链**:
```
executor.rs compile
  → MegaKernelExecutor::compile_from_auto_graph(graph, ..., config)
    → graph.clone()
    → compiler.compile_mega_kernel_from_graph(graph, &config, hetero)  // CPU
    → compiler.compile_mega_kernel_to_gpu(graph_clone, &config, sm_version)  // GPU
    → mega_compiled.gpu_code = Some(gpu_output.gpu_code)
  → backend.prepare_gpu_mega_kernel(wb, mk.gpu_code(), ...)
    → CudaBackend: upload weight_blob + cache PTX in compiled_ptx
```

**关键文件**:
- `gllm/mega_kernel.rs`: clone graph + GPU 编译 + set_gpu_code
- `gllm/executor.rs`: 传 sm_version 到 MegaKernelConfig
- `gllm-kernels/mod.rs`: `compile_mega_kernel_to_gpu` 已存在，无需修改

### REQ-GPU-011: Forward-only GPU 编译

gllm-kernels 新增 `compile_graph_to_gpu` 方法，为 encoder 路径（embedding/rerank/classify）生成 GPU kernel。

**问题**: `compile_mega_kernel_to_gpu` 只处理 decoder mega-kernel（21-param ABI + generate loop），encoder forward-only kernel 需要独立编译（10-param ABI）。

**设计**:
- `InferenceCompiler::compile_graph_to_gpu(&mut self, graph: &CompilerGraph, sm_version: u32) -> Result<GpuForwardOutput, InferenceError>`
- 复用 `compile_graph` 的 Phase 0-2 pipeline (fusion, buffer alloc)
- Phase 3 使用 `GpuLower` 替代 `X86CodeGen`
- 输出 `GpuForwardOutput { gpu_code: Vec<u8>, scratchpad_bytes: usize }`

**关键文件**:
- `gllm-kernels/src/compiler/mod.rs`: 新增方法

### REQ-GPU-012: HIP/Metal Mega-Kernel Launch ✅

实现 HIP 和 Metal 后端的 `gpu_launch_mega_kernel` 及 `GpuEncoderOps`。

**设计**:
- HIP: `HipDevice::load_hsaco` (HipModule + get_function) + `launch_kernel` (hipModuleLaunchKernel)
- Metal: `MetalDevice::load_library_data` (AIR bitcode) + `launch_compute` (MTLComputeCommandEncoder + dispatchThreadgroups)
- 21-param ABI 参数布局与 CUDA 完全一致（全为 device 指针）
- `GpuEncoderOps` trait 三后端统一实现 encoder forward

**关键文件**:
- `gllm-kernels/gpu/hip/device.rs`: HipModule + load_hsaco + launch_kernel
- `gllm-kernels/gpu/metal/device.rs`: launch_compute
- `gllm/hip_backend.rs:gpu_launch_mega_kernel`
- `gllm/metal_backend.rs:gpu_launch_mega_kernel`
- `gllm/gpu_backend_macro.rs:GpuEncoderOps` impl for HIP/Metal

### REQ-GPU-001: batch_forward_gpu_pure ✅

实现 GPU decoder forward，用于 continuous batching 调度路径。

**设计**:
- `gpu_backend_macro.rs` 的 `batch_forward_gpu_pure` 方法
- Launch mega-kernel PTX 到 GPU，传入 batch 维度参数
- KV cache 使用 device ptr（`gpu_alloc_kv_cache` 已分配）
- 返回 `(Vec<LogitsHandle>, sparsity, telemetries)`
- LogitsHandle 中的 data 在 GPU 上，通过 `gpu_sample_from_tensor` 采样

**调用方**: `executor.rs:step()` L1591, `executor.rs` profiling L2656

### REQ-GPU-002: embedding_forward_gpu_pure ✅

实现 GPU encoder forward，用于 embedding 推理。

**设计**:
- Launch forward-only mega-kernel PTX
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

### CPU Mega-kernel ABI (21 参数)

```rust
entry_fn(
    input_ids: *const u32,        // arg 1
    weight_ptr: *const u8,        // arg 2
    kv_cache: *mut u8,            // arg 3
    positions: *const u32,        // arg 4
    seq_lens: *const u32,         // arg 5
    batch_size: usize,            // arg 6
    seq_len: usize,               // arg 7
    scratchpad: *mut u8,          // arg 8
    output_tokens: *mut u32,      // arg 9
    temperature_bits: usize,      // arg 10
    top_k: usize,                 // arg 11
    top_p_bits: usize,            // arg 12
    max_new_tokens: usize,        // arg 13
    eos_token_id: usize,          // arg 14
    output_mode_selector: usize,  // arg 15
    hook_ctx_ptr: *const u8,      // arg 16
    telemetry: *mut u8,           // arg 17
    session_position: usize,      // arg 18
    fused_hidden_ptr: *const u8,  // arg 19
    num_mm_tokens: usize,         // arg 20
    callback_table_ptr: *const u8 // arg 21
)
```

### GPU Kernel Launch ABI

```c
// CUDA
cuLaunchKernel(
    kernel_fn,
    grid_dim_x, grid_dim_y, grid_dim_z,  // 1, 1, 1 (single sequence)
    block_dim_x, block_dim_y, block_dim_z, // threads per block
    shared_mem_bytes,
    stream,
    kernel_params: [*mut void; 21],        // 21 个指针/值参数
    extra: null
)
```

GPU launch 时 21 个参数全部转为 `void*` 数组，每个参数指向 device 内存地址。

---

## 实施顺序

1. ~~REQ-GPU-007 (权重上传)~~ ✅
2. ~~REQ-GPU-008 (Callback 传递)~~ ✅
3. ~~REQ-GPU-001 (batch_forward)~~ ✅
4. ~~REQ-GPU-002 (embedding_forward)~~ ✅
5. ~~REQ-GPU-003/004 (rerank/classify)~~ ✅
6. ~~REQ-GPU-005/006 (score_tokens/encode_at_layer)~~ ✅
7. ~~REQ-GPU-010 (GPU PTX 编译接入)~~ ✅
8. ~~REQ-GPU-011 (Forward-only GPU 编译)~~ ✅
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
