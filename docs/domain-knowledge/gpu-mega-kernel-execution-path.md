# GPU Mega-Kernel 执行路径领域资料库（C-9）

> 来源：gllm 源码（ARCH-UNIFIED-EXEC 重构后的实现，确定性，非猜）
> 建库触发：GPU 推理路径反复出错（≥3 轮重构 + 2 次 BCE + 2 轮 architect consult），5 个技术点分散未文档化，诊断时反复重新读源码
> 最后验证：2026-07-06

## 核心机制（源码确认）

### 1. CompiledExecutable 枚举 — CPU/GPU 统一可执行体

源码：`src/engine/mega_kernel/abi_types.inc.rs:289-306`

```rust
pub enum CompiledExecutable {
    Cpu {
        code: CompiledLayer,          // mmap'd JIT 机器码
        entry_fn: MegaKernelFn,       // 22-param 函数指针，单次 CALL
    },
    Gpu {
        ptx: Vec<u8>,                 // PTX/HIP/MSL 字节码
        kernel_name: String,
        launcher: Arc<dyn Fn(&MegaKernelArgs) -> Result<(), MegaKernelError> + Send + Sync>,
    },
}
```

**关键语义**：
- 编译时按 target **二选一 bake** 进 `MegaKernelCompiled.executable`，运行时零 dtype/target 分支
- `Cpu` 变体：直接 CALL `entry_fn` 函数指针（22-param flat ABI via KernelContext）
- `Gpu` 变体：通过 `launcher` 闭包 `cuLaunchKernel`，**闭包在编译时捕获 backend**
- `MegaKernelExecutor` struct **不持 backend 字段**（避免泛型传染）— backend 通过 launcher 闭包间接捕获

### 2. 单次编译 + target 派生（消灭套娃双编译）

源码：`src/engine/mega_kernel/executor_core.inc.rs:104-108`

```rust
let target = if let Some(sm) = gpu_sm_version {
    CompileTarget::Gpu { sm_version: sm }
} else {
    CompileTarget::Cpu
};
// 单次编译: target 决定走 CompileOutput::Cpu 还是 CompileOutput::Gpu
```

**已删除的套娃**（`executor_core.inc.rs:126, 374` 注释明确）：
- ~~`target: CompileTarget::Cpu` 硬编码~~（旧 line 64）
- ~~克隆 graph_for_gpu 单独编 GPU~~（旧 line 87，套娃双编译块）
- 现在：单次 `compile()`，target 按 `gpu_sm_version` 派生，Cpu 仅 fallback

### 3. launcher 闭包捕获 backend（architect sessionId 5d98f4f4）

**工厂注入链**（`src/engine/executor_compile.rs:614-636`）：

```rust
// compile_and_upload_mega 内（有 backend: &B）
let gpu_launcher_builder = if gpu_sm_version.is_some() {
    Some(&|ptx, kernel_name| backend.build_mega_launcher(ptx, kernel_name))
} else { None };
let mega = MegaKernelExecutor::compile_from_auto_graph(
    ..., gpu_sm_version, gpu_launcher_builder,  // 工厂注入
);
```

**闭包构造**（`src/compat/gpu_backend_macro.rs:568-599`，在 backend 模块内）：

```rust
fn build_mega_launcher(&self, ptx, kernel_name) -> Result<Arc<dyn Fn(...)>> {
    let backend = self.clone();  // Arc-share GPU device/driver handle
    Ok(Arc::new(move |args: &MegaKernelArgs| {
        backend.launch_mega_kernel_with_bridging(&ptx, &kernel_name, args)
    }))
}
```

**为什么传工厂而非 backend**：
- `compile_from_auto_graph` 是静态方法**不持 backend**
- `gpu_launch_mega_kernel` 是 `pub(super)` — 只在 backend 模块内可见
- 闭包**必须在 backend 模块内构造**才能调用 `pub(super)` 方法
- 工厂模式让 `compile_from_auto_graph` 不感知 backend 类型，无泛型传染

**Backend trait 默认实现返回 Err**（`src/compat/mod.rs:320-331`），GPU backend override。**这不是 LSP 违规**（与 `prepare_gpu_mega_kernel`/`gpu_sm_version` 同默认+覆盖模式）。

### 4. execute_* 分流（7 个 execute_* 统一 match）

源码：`src/engine/mega_kernel/executor_ops.inc.rs`（`CompiledExecutable::` 出现 14 处 = 6 execute_* × 2 arm + 2 查询 arm）

```rust
match &mega.executable {
    CompiledExecutable::Cpu { entry_fn, .. } => unsafe {
        (entry_fn)(input_ids.as_ptr(), weight_blob_ptr, kv_cache.as_mut_ptr(),
                   positions.as_ptr(), /* 17 more params */)
    },
    CompiledExecutable::Gpu { launcher, .. } => {
        let args = MegaKernelArgs { input_ids_ptr, weight_blob_ptr, ..., scratchpad_bytes };
        launcher(&args)?;   // → launch_mega_kernel_with_bridging → cuLaunchKernel
        0usize
    }
}
```

**CPU/GPU 参数展开差异**：
- CPU：22 个参数直接传给 `entry_fn` 函数指针（flat ABI）
- GPU：构造 `MegaKernelArgs` 结构体（22 字段），launcher 内部转 `[usize; 22]` 指针数组
- 类型系统保证字段不遗漏（结构体字段 vs 函数参数一一对应）

### 5. H2D → launch → D2H 三步内聚（launch_mega_kernel_with_bridging）

源码：`src/compat/cuda_backend.rs:525-636`

```rust
pub(super) fn launch_mega_kernel_with_bridging(&self, ptx, kernel_name, args) -> Result<()> {
    // 1. 锁 GpuMegaBuffers（prepare 时分配的 4 个 device buffer）
    let (scratchpad_dev, output_dev, kv_cache_dev, sp_alloc, out_alloc) = { ... };

    // 2. weight device ptr（从 weight_blob_gpu 缓存读）
    let weight_dev = self.weight_blob_gpu.lock()?.map(|(ptr,_)| ptr)?;

    // 3. H2D input_ids + positions（per-call alloc，gate 允许 leak）
    let input_dev = self.upload_to_gpu(input_host)?;
    let positions_dev = self.upload_to_gpu(positions_host)?;

    // 4. 构造 device argv（host ptr → device ptr 替换）
    let raw: [usize; 22] = [
        input_dev, weight_dev, kv_cache_dev, positions_dev,  // device ptrs
        args.aux_ptr as usize,                                // host ptr (kernel 少读)
        args.batch_size, args.prompt_len,
        scratchpad_dev, output_dev,                           // device ptrs
        args.temperature_u32, args.top_k, args.top_p_u32,
        args.max_new_tokens, args.eos_token_id,
        args.hook_ctx_ptr as usize,                           // host ptr
        args.telemetry_ptr as usize,                          // host ptr
        args.session_position, args.fused_hidden_ptr as usize,
        args.num_mm_tokens, args.callback_table_ptr as usize, args.page_table_ptr as usize,
        args.batch_ctx_ptr as usize,
    ];

    // 5. cuLaunchKernel
    self.gpu_launch_mega_kernel(ptx, kernel_name, &raw)?;  // → device.launch_kernel

    // 6. D2H scratchpad + output_tokens
    let sp_bytes = args.scratchpad_bytes.min(sp_alloc);  // BCE-20260705-GPUPTR-002
    self.download_from_gpu(scratchpad_dev, sp_bytes) → copy to args.scratchpad_ptr;
    let out_bytes = args.output_tokens_bytes.min(out_alloc);
    self.download_from_gpu(output_dev, out_bytes) → copy to args.output_tokens_ptr;
}
```

### 6. prepare_gpu_mega_kernel — 统一上传入口

源码：`src/compat/gpu_backend_macro.rs:54-96` + `src/engine/executor_compile.rs:638-646`

```rust
// executor_compile.rs: compile_and_upload_mega 末尾
let wb = mega.weight_blob();           // weight 原始字节
let sb = mega.scratchpad_bytes();      // scratchpad 大小
let decoder_gc = mega.gpu_code();      // PTX 字节
let kv_cb = mega.kv_cache_bytes(geometry.num_layers);  // kv_cache 大小（独立 buffer）
backend.prepare_gpu_mega_kernel(wb, decoder_gc, sb, kv_cb);
```

`prepare_gpu_mega_kernel` 一次做三件事：
1. **upload_weight_blob** → `weight_blob_gpu: Mutex<Option<(u64, usize)>>` 缓存
2. **compiled_ptx.insert** → `compiled_ptx: Mutex<HashMap<String, Vec<u8>>>` 缓存（key "mega_kernel" + "__scratchpad_bytes__"）
3. **alloc_gpu_mega_buffers** → `GpuMegaBuffers { scratchpad, output, kv_cache, scratchpad_bytes, output_bytes }`（4 个 device buffer 槽）

**契约**：`get_cached_ptx`/`get_cached_scratchpad_bytes` 用 `.expect("prepare_gpu_mega_kernel was not called")` — prepare 必须先于 launch 调用，否则 panic。

## GpuMegaBuffers 4 槽结构（BCE-20260705-GPUPTR-002 关键）

```rust
struct GpuMegaBuffers {
    scratchpad: u64,    // device ptr — kernel 读写激活/logits
    output: u64,        // device ptr — output_tokens
    kv_cache: u64,      // device ptr — 层内 attention K/V（必须与 scratchpad 分离！）
    scratchpad_bytes: usize,
    output_bytes: usize,
}
```

**kv_cache 与 scratchpad 必须分离**（BCE-20260705-GPUPTR-002）：
- 旧实现 kv_cache 别名 scratchpad → logits 污染
- 现在 argv slot 2 (kv_cache_dev) ≠ slot 7 (scratchpad_dev)，独立 device buffer
- `kv_cache_bytes(geometry.num_layers)` 独立传入 prepare，独立分配

## D2H size 不一致陷阱（BCE-20260705-GPUPTR-002 残留）

```rust
let sp_bytes_copy = args.scratchpad_bytes.min(scratchpad_alloc_bytes);
debug_assert!(args.scratchpad_bytes <= scratchpad_alloc_bytes, "size mismatch...");
```

- prepare 分配 `runtime_scratchpad_bytes(1)`（单 seq）
- runtime 要 copy `runtime_scratchpad_bytes(max_total)`（满 batch）
- `min` 防越界 + `debug_assert` 暴露不一致（**不静默 min 掩盖**）
- **这是预存 BUG**：长 batch 时 D2H 截断，logits 部分丢失。gate 用单 seq 不触发，生产 batch 会爆

## 22-param ABI SSOT 顺序（argv 索引）

| idx | 字段 | CPU 路径 | GPU 路径 |
|-----|------|---------|---------|
| 0 | input_ids | `input_ids.as_ptr()` | `input_dev` (H2D) |
| 1 | weight_blob | `weight_blob.as_ptr()` | `weight_dev` (cached) |
| 2 | kv_cache | `kv_cache.as_mut_ptr()` | `kv_cache_dev` (独立) |
| 3 | positions | `positions.as_ptr()` | `positions_dev` (H2D) |
| 4 | aux | `null()` | host ptr (kernel 少读) |
| 5 | batch_size | `1` | `args.batch_size` |
| 6 | prompt_len | `prompt_len` | `args.prompt_len` |
| 7 | scratchpad | `scratchpad.as_mut_ptr()` | `scratchpad_dev` (cached) |
| 8 | output_tokens | `output_tokens.as_mut_ptr()` | `output_dev` (cached) |
| 9-13 | temp/top_k/top_p/max_new/eos | 字面量 | `args.*` |
| 14 | hook_ctx | `null()` | host ptr |
| 15 | telemetry | `null_mut()` | host ptr |
| 16 | session_position | `0` | `args.session_position` |
| 17 | fused_hidden | `null()` | host ptr |
| 18 | num_mm_tokens | `0` | `args.num_mm_tokens` |
| 19 | callback_table | `null()` | host ptr |
| 20 | page_table | `null()` | host ptr |
| 21 | batch_ctx | `null()` (单 seq legacy) | host ptr |

**CPU/GPU 参数语义必须一一对应** — 任一错位 = 随机显存读取。

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| `MegaKernelExecutor` 持 backend 字段 | **不持** — backend 通过 `CompiledExecutable::Gpu.launcher` 闭包间接捕获（无泛型传染）|
| Backend trait 加 `build_mega_launcher` 是 LSP 违规 | **非违规** — 默认 Err + GPU override，与 `prepare_gpu_mega_kernel` 同模式 |
| 闭包用普通 Mutex 捕获 backend | **必须 `Arc::clone`** — 普通 Mutex Clone 后重置，闭包看不到 prepare 缓存的 buffer |
| GPU launch 不需要 D2H（kernel 写 host 内存） | **必须 D2H** — kernel 写 device 显存，CPU 读 host，`cuMemcpyDtoH` 拷回 |
| kv_cache 和 scratchpad 共用 device buffer 省 alloc | **必须分离** — 别名致 logits 污染（BCE-20260705-GPUPTR-002）|
| D2H 用 `min(device, host)` 静默防越界 | **`min` + `debug_assert`** — min 掩盖尺寸不一致真 bug，assert 暴露 |
| `gpu_launch_mega_kernel` 可在 backend 模块外调用 | **`pub(super)`** — 只在 backend 模块内可见，闭包必须在模块内构造 |
| prepare 和 launch 可任意顺序 | **prepare 必须先** — `get_cached_ptx`/`get_cached_scratchpad_bytes` `.expect` 未 prepare 会 panic |
| GPU PTX 编了不 launch = 死代码 | **已接通** — execute_* Gpu arm → launcher → launch_mega_kernel_with_bridging → cuLaunchKernel |
| `max_new_tokens=1` 在 execute_encode GPU arm 是 encode 语义 | ⚠️ **可疑** — encode 模式输出是 logits（logits_scratch_offset），不是 output_tokens；GPU arm 仍 D2H output_tokens_bytes，语义需 GPU E2E 验证 |

## 已知问题 / 边界（待 GPU E2E 验证聚焦）

### 1. forget-leak（BCE-20260705-GPU-FORGET-LEAK 待横扫）
- `alloc_scratchpad_gpu`/`upload_to_gpu` 用 `std::mem::forget(Buffer)` 持裸 u64 ptr，永不释放
- gate 阶段单次调用可接受，generate 热循环（逐 token）会 OOM
- **根治方向**：缓存化 device buffer + RAII free（复用 `GpuMegaBuffers` 模式）

### 2. D2H size 不一致（BCE-20260705-GPUPTR-002 残留）
- prepare 分配单 seq，runtime copy 满 batch → `min` 截断
- gate 单 seq 不触发，生产 batch 会丢 logits
- **根治方向**：prepare 用 `runtime_scratchpad_bytes(max_total)` 分配

### 3. host ptr 传 GPU kernel（argv 4/14/15/17/19/20/21）
- aux/hook_ctx/telemetry/fused_hidden/callback_table/page_table/batch_ctx 仍传 host ptr
- 假设 GPU kernel 不在 device 侧解引用这些（或走 zero-copy unified memory）
- 若 kernel 真读 host ptr 会崩 — **gate 后需确认 kernel 对这些参数的访问语义**

### 4. positions device buffer 溢出风险
- 3C 实现把 positions H2D 进 input buffer 尾部偏移（input_bytes 之后）
- 假设 input buffer 足够大（`sp_bytes = scratchpad_bytes.max(1024)`）
- 长 prompt 可能溢出，gate 用小 prompt 不触发

### 5. execute_encode GPU arm output_tokens 语义可疑
- `executor_ops.inc.rs:358-385` GPU arm 用 `output_tokens_ptr` + `output_tokens_bytes`
- 但 encode 模式输出是 logits（scratchpad + logits_scratch_offset），非 output_tokens
- GPU kernel 是否正确把 MeanPool/classifier 结果写到 scratchpad 的 logits offset？需 GPU E2E 验证

## 解决问题时参考

### 诊断 GPU 推理路径
1. 确认 `gpu_sm_version` 是否 `Some`（决定 target=Gpu 还是 Cpu）
2. 确认 `prepare_gpu_mega_kernel` 已调（否则 launch panic）
3. 确认 `CompiledExecutable::Gpu` arm 是否被命中（execute_* 分流）
4. 确认 launcher 闭包是否调 `launch_mega_kernel_with_bridging`
5. 确认 H2D/D2H 是否齐全（input_ids/positions H2D，scratchpad/output D2H）
6. 确认 kv_cache_dev ≠ scratchpad_dev（slot 2 ≠ slot 7）

### GPU E2E argmax 不对齐诊断
1. 先跑 CPU E2E 确认 argmax=253（基线，已 pass）
2. GPU E2E argmax≠253 → 检查是否真走 Gpu arm（加 eprintln 在 launcher）
3. 检查 D2H size 是否截断（`debug_assert` 触发？）
4. 检查 host ptr 是否被 kernel 解引用（崩 = 解引用了）
5. 检查 kv_cache_dev 是否别名 scratchpad_dev（logits 污染）

### SSH 5070Ti 排障（基础设施，非代码）
- 离线（No route to host）= 物理机关机/网络故障 → **不重试，报告用户启动机器**
- `sshpass -p '123456' ssh -o StrictHostKeyChecking=no putao@192.168.1.200`
- 重试阈值：**连续 2 次失败就停**（C-2）

## 关键代码位置

| 组件 | 位置 |
|------|------|
| CompiledExecutable enum | `src/engine/mega_kernel/abi_types.inc.rs:289-306` |
| MegaKernelExecutor struct | `src/engine/mega_kernel/abi_types.inc.rs:34-53` |
| compile_from_auto_graph target 派生 | `src/engine/mega_kernel/executor_core.inc.rs:104-108` |
| 套娃双编译删除声明 | `src/engine/mega_kernel/executor_core.inc.rs:126, 374` |
| gpu_launcher_builder 工厂注入 | `src/engine/executor_compile.rs:621-628` |
| prepare_gpu_mega_kernel 调用 | `src/engine/executor_compile.rs:644` |
| build_mega_launcher（闭包构造） | `src/compat/gpu_backend_macro.rs:568-599` |
| Backend trait 默认 Err | `src/compat/mod.rs:320-331` |
| launch_mega_kernel_with_bridging | `src/compat/cuda_backend.rs:525-636` |
| gpu_launch_mega_kernel（cuLaunchKernel） | `src/compat/cuda_backend.rs:456-481` |
| execute_* Cpu/Gpu 分流 | `src/engine/mega_kernel/executor_ops.inc.rs`（14 处 CompiledExecutable::） |
| execute_encode 分流（encode 模式） | `src/engine/mega_kernel/executor_ops.inc.rs:326-388` |
| GpuMegaBuffers 4 槽 | `src/compat/cuda_backend.rs`（alloc_gpu_mega_buffers:488） |

## 与其他资料库关系

- `cuda-driver-api.md`: CUDA Driver API 基础（H2D/D2H/cuLaunchKernel ABI + GpuDevice trait）— 本文件是其上层"mega-kernel 执行路径"的语义
- `mega-kernel-topology.md`: GenerateLoop M=1 拓扑 + ARCH-DECODE-LOGITS-ROW0 — 本文件是其在 GPU 侧的执行实现
- `vam-activation-pingpong.md`: ActivationPing/Pong 层间残差流 — line 91 引用 BCE-20260705-GPUPTR-002（kv_cache 分离），本文件展开该 BCE 的完整上下文
- `kv-cache-dtype-dual-layer.md`: KV cache dtype 双地层陷阱（JIT F32 vs buffer BF16）— 本文件覆盖 GPU 侧 kv_cache_dev 的 buffer 生命周期

## 建库缺口清零映射

本文件一次清零 5 个 GPU 执行路径知识库缺口：

| 缺口 | 覆盖 section | 反复出错依据 |
|------|-------------|-------------|
| D-1 CompiledExecutable Cpu/Gpu 分流语义 | §1, §4 | ≥3 轮 ARCH-UNIFIED-EXEC 重构 |
| A-1 GPU scratchpad 生命周期 | §5, §6, GpuMegaBuffers | 2 次 BCE（GPUPTR-001/002）|
| E-1 MegaKernelExecutor 闭包捕获方案 | §3 | 2 轮 architect consult（5d98f4f4）|
| B-1 PTX kernel 注册 + compiled_ptx 缓存 | §6（prepare 统一入口）| 1 轮重构阻塞 |
| C-1 GPU weight_blob 上传统一入口 | §6（prepare 三合一）| 1 轮重构阻塞 |
