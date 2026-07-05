# CUDA Driver API（H2D/D2H/cuLaunchKernel）领域资料库

> 来源：CUDA Driver API 官方文档 + gllm 自研 GpuDevice trait + 5070Ti 实测
> 建库触发（C-9）：CUDA 相关代码反复出错 ≥2 次，根因都是对 host/device 地址空间 + ABI 的误解
> 最后验证：2026-07-05

## 反复踩坑记录（建库原因）

| 次数 | 错误 | 根因 | 代价 |
|------|------|------|------|
| 1 | GPU E2E argmax=38734（应 253） | launcher 闭包把 host 指针 `as usize` 传给 cuLaunchKernel，GPU 读显存随机数据 | 整个 ARCH-UNIFIED-EXEC 重构 |
| 2 | launcher 设计反复两次 | pub(super) 模块可见性 + trait object 调不到 inherent method | architect 两轮 consult |
| 3 | SSH 5070Ti 反复试 5 次 | 未建立「机器离线是基础设施阻塞」的认知，机械重试 | 5 轮 CRON 空转 |

## 核心机制（与出错相关的）

### 1. host 地址 vs device 地址（最易错点）

**CUDA 有两个独立地址空间**：
- **host 内存**：CPU 侧，`vec.as_mut_ptr()` 返回的地址，`0x7f...` 段
- **device 内存**（显存）：GPU 侧，`cuMemAlloc` 返回的地址，`0x...` 通过 driver handle 访问

**致命区别**：GPU kernel 只能读 device 地址。把 host 指针传给 cuLaunchKernel 的 kernelParams，GPU 拿到的是 host 地址数字，**在 device 地址空间读到的是随机显存**（不是 host 数据）——无报错，结果纯噪声（argmax=38734 就是这么来的）。

```
✅ 正确: cuLaunchKernel(..., [&device_ptr, ...])  // device 地址
❌ 错误: cuLaunchKernel(..., [&host_ptr, ...])    // host 地址数字, GPU 读随机显存
```

### 2. cuLaunchKernel 的 kernelParams ABI

```c
CUresult cuLaunchKernel(
    CUfunction f, ...,
    void **kernelParams  // 指向参数数组的指针数组
);
```

`kernelParams` 是 `void**`——每个元素是**指向某个参数存储位置的指针**。gllm 的 22-param mega-kernel：

```rust
let kernel_args: [*mut c_void; 22] = array::from_fn(|i| {
    &args[i] as *const usize as *mut c_void  // 指向 args[i] 的地址
});
```

`args: [usize; 22]`——**这里的 usize 必须是 device pointer 的数值**（对指针类参数）或标量值（batch_size/prompt_len 等）。

**22-param SSOT 顺序**（mega_kernel_abi.rs:159-210 + gpu_generate_single_sequence:340-369，绝不能改）：
```
0:input_ids_ptr  1:weight_blob_ptr  2:kv_cache_ptr  3:positions_ptr  4:aux_ptr
5:batch_size  6:prompt_len  7:scratchpad_ptr  8:output_tokens_ptr
9:temperature_u32  10:top_k  11:top_p_u32  12:max_new_tokens  13:eos_token_id
14:hook_ctx_ptr  15:telemetry_ptr  16:session_position  17:fused_hidden_ptr
18:num_mm_tokens  19:callback_table_ptr  20:page_table_ptr  21:batch_ctx_ptr
```

**类参数（0-4,7,8,14-21 的 ptr）**：GPU 模式下必须是 device pointer
**标量参数（5,6,9-13,16,18）**：usize 值直接传

### 3. H2D / D2H（host↔device 数据搬运）

GPU kernel 不能直接访问 host 内存（unified memory 除外，gllm 不用）。完整 GPU 推理流程：

```
1. H2D (host→device): cuMemcpyHtoD_v2(device_dst, host_src, bytes)
   - input_ids, positions, weight 等输入数据要先传到 device
2. launch: cuLaunchKernel(device args)  // kernel 在 GPU 跑
3. D2H (device→host): cuMemcpyDtoH_v2(host_dst, device_src, bytes)
   - output_tokens, logits 等 GPU 产出要拷回 host 给 CPU 读
```

**GpuDevice trait API**（gllm-kernels/src/gpu/mod.rs，三 backend 统一）：
```rust
fn alloc(bytes) -> Buffer                      // 分配 device 内存
fn htod(src: &[u8], dst: &mut Buffer, stream)  // host→device (需 Buffer 对象)
fn dtoh(src: &Buffer, dst: &mut [u8], stream)  // device→host (需 Buffer 对象)
```

**CudaBackend 的 u64 模式**（cuda_backend.rs，绕过 Buffer 对象用裸 device ptr）：
```rust
fn alloc_scratchpad_gpu(bytes) -> u64          // alloc + as_device_ptr + forget(Buffer)
fn upload_to_gpu<T>(data: &[T]) -> u64         // alloc + htod + forget → 返回 device ptr
fn download_from_gpu(src_ptr: u64, bytes) -> Vec<u8>  // cuMemcpyDtoH_v2 直接拷
```

**u64 模式的代价**：`std::mem::forget(Buffer)` 永不释放 → 每次 alloc 泄漏一块显存。热循环（generate 逐 token）调用会 OOM。这是预存 BUG（BCE-20260705-GPU-FORGET-LEAK 待横扫）。

### 4. PTX 加载 + cuLaunchKernel 流程

```rust
let module = device.load_ptx(ptx_code)?;           // cuModuleLoadData
let func = module.get_function(kernel_name)?;      // cuModuleGetFunction
let stream = device.default_stream();
device.launch_kernel(func, grid, block, &kernel_args, stream)?;  // cuLaunchKernel
```

PTX 是 NVIDIA 的并行线程执行 ISA（GPU 机器码的中间表示），由 gllm-kernels GPU codegen 生成。5070Ti SM 12.0 (Blackwell)。

## AI 易误判点

| ❌ 误判 | ✅ 正解 |
|--------|---------|
| 「host 指针 as usize 传给 GPU 能读」 | GPU 只读 device 地址，host 指针传过去读随机显存 |
| 「把 host 指针当 device 用，编译能过就行」 | 编译过 ≠ 对，host/device 是运行时地址空间区分，类型系统不挡 |
| 「u64 device ptr 和 Buffer 对象等价」 | u64 模式 forget 了 Buffer → 无法 htod/dtoh（需 Buffer），只能用裸 cuMemcpyHtoD_v2/cuMemcpyDtoH_v2 |
| 「D2H 拷贝量用 min(device, host) 防越界」 | 用 host 期望尺寸 args.scratchpad_bytes，加 debug_assert 防越界；min 掩盖尺寸不一致真 bug |
| 「闭包捕获 backend 用普通 Mutex」 | 必须 Arc<Mutex>——Clone 后普通 Mutex 重置，闭包看不到 prepare 缓存的 buffer |
| 「trait 加默认 Err 方法 = LSP 违规」 | 默认实现 + 子类可选覆盖 ≠ LSP 违规（与 prepare_gpu_mega_kernel 同模式）|

## 解决问题时参考

### 写 GPU launch 代码时必须确认
1. **每个传给 cuLaunchKernel 的指针类参数是 device ptr 还是 host ptr**——对照 gpu_generate_single_sequence 的 device ptr 模式
2. **H2D/D2H 是否齐全**——kernel 读取的每个 host 数据都要先 H2D，产出的每个 device 数据都要 D2H
3. **device buffer 生命周期**——热循环复用（缓存进 backend 字段）还是 per-call alloc（leak，仅 gate 用）
4. **闭包捕获 backend 的所有权**——Arc<Mutex> 共享，普通 Mutex 会 Clone 重置

### SSH 5070Ti 排障（基础设施，非代码）
- 离线（No route to host）= 物理机器关机/网络故障 → **不重试，报告用户启动机器**
- sshpass 命令：`sshpass -p '123456' ssh -o StrictHostKeyChecking=no putao@192.168.1.200`
- 重试阈值：**连续 2 次失败就停**（C-2），不机械重试 5 次

## 已知问题 / 边界

- **forget-leak**：CudaBackend 的 alloc_scratchpad_gpu/upload_to_gpu 用 forget(Buffer) 持裸 u64 ptr，永不释放。gate 阶段单次调用可接受，generate 热循环会 OOM。BCE-20260705-GPU-FORGET-LEAK 待横扫（缓存化 device buffer + RAII free）。
- **positions device buffer**：3C 实现把 positions H2D 进 input buffer 尾部偏移（input_bytes 之后），假设 input buffer 足够大（sp_bytes = scratchpad_bytes.max(1024)）。长 prompt 可能溢出，gate 用小 prompt 不触发。
- **aux/hook_ctx/telemetry 等 host ptr**：3C 实现仍传 host ptr（argv 4/14/15/17/19/20/21），假设 GPU kernel 不在 device 侧解引用这些（或走 zero-copy uniform memory）。若 kernel 真读会崩——gate 后需确认 kernel 对这些参数的访问语义。
