# GPU Mega-Kernel Launch ABI 统一裁决

> 状态：架构裁决，修复预存缺陷（`cargo check --features cuda --lib` 5×E0308 + 三层 ABI 漂移）。
> 性质：BCE 闭环（C-7）——BUG 类 = "ABI arity drift across layers"，根因 R1→R2 迁移半途。根治非打补丁。
> 所有结论源码实证，引用见文末。

## 0. 纠正问题前提（关键）

**三层没有一层是 R2 单指针，全是漂移的 R1 多参数。**

| 层 | 实际形态 | 证据 |
|----|---------|------|
| CPU ABI（对齐基准） | **单指针** `fn(ctx: *const u8)`，22 **字段**按偏移读 | abi_types.inc.rs:274；prologue 偏移读 |
| PTX mega-kernel | 20 个 `.param`（input_ids…callback_table） | gpu_lower/prologue.inc.rs:294-316 |
| args builder | `[usize;22]` **flat 多参数**（含 batch_size/seq_len/temperature_bits 内联标量） | gpu_helpers.rs:381-430 |
| launch(cuda/hip/metal) | `&[usize;23]`，`kernel_args[i]=&args[i]` → 23 个独立 kernelParam | cuda_backend.rs:427,439-441 |

builder 的 22 ≠ KernelContext 的 22 字段：builder 是 `input_ids/weight/kv/positions/aux/batch_size/seq_len/scratchpad/output/temp/top_k/top_p/max_new/eos/hook/telem/session/fused/num_mm/callback/page_table/batch_ctx`；KernelContext 是 `weight_blob/kv_cache/output/hook/seq_len_ptr/rope_freqs/kv_page_table/batch_meta/kv_page_size/…/weight_offsets_ptr/…`。**两套字段集不同**。所以"选一层对齐"解决不了问题——必须真正迁移到 R2。

## 1. Q1 GPU kernel ABI 统一方向 → **选项 A（KernelContext 单指针，与 CPU 对齐）**

PTX mega-kernel 改为接收 **1 个 KernelContext device 指针**，prologue 内按偏移 `ld.global` 读 22 字段——与 CPU prologue 的偏移读逻辑同构。

**理由（SPEC 强制 + 铁律）**：
- **REQ-UMK-27 明文**："CPU/GPU 统一编译 — 共享 IR/GraphType + DeviceProfile 驱动 codegen"。CPU 是单指针 → GPU 必须单指针，ABI 相同，差异只落在 codegen（`ld.param` vs `ld.global`）。选 B（22 param）/ 现状（23）= 固化 CPU/GPU ABI 分叉，直接违反 REQ-UMK-27。
- **ARCH-CPU-GPU-UNIFIED**：共享 CompilerGraph/VmProgram，仅 ISA lowering 分叉（mod.rs:690 X86Lower vs 922 GpuLower）。ABI 是 IR 契约的一部分，不该按设备分叉。单指针让 GPU prologue 复用 CPU 的 KernelContext 偏移表（同一份 field-offset SSOT）。
- **偏移读机制已存在**：CPU prologue 已做 ctx+offset 字段读。GPU 只是把 `ld.param.u64` 换成从 ctx base + 偏移 `ld.global.u64`——codegen 层的指令选择差异，非 ABI 差异（符合 NO-HW-DEGRADATION：硬件差异落在 codegen）。

**否决 B（扩到 22 param）**：与 CPU 单指针对立，违反 REQ-UMK-27。且 KernelContext 后续加字段（现已含 weight_page_table/kv_page_header 等 REQ-WP-008/COMP11）→ 每加一个字段就要改 PTX 签名 + 三端 launch arity，正是当前漂移的病根。单指针则字段增减只改偏移表，签名恒定。

**否决 C（降到 5 param）**：5-param 无法承载 KernelContext 全部字段——KV paging / telemetry / callback / batch_ctx / sampling 会全部无处安放，交付范围塌缩，违反 C-5 范围守恒（SPEC 内元素平权，全部交付）+ ARCH-ROOT-CAUSE（禁降级思维）。simplified 5-param 只应留作无 mega-kernel 的 debug 简单 kernel，不作 mega-kernel ABI。

## 2. Q2 launch 层统一 → **三端统一为单指针**

cuda/hip/metal 的 `gpu_launch_mega_kernel` 签名统一为接收 **1 个 KernelContext device 指针**（`ctx_dev: u64`，launch 内构造 `[*mut c_void;1] = [&ctx_dev]`）：
- CUDA/HIP：`cuLaunchKernel`/`hipModuleLaunchKernel` kernelParams 数组长度 1（cuda/device.rs:491 机制不变，只是 1 个 param）。
- Metal：`MTLComputeCommandEncoder.setBuffer(ctx_buffer, offset:0, index:0)`——单 buffer 绑定 index 0。
- 消除 HIP `[*mut c_void;21]` vs `&[usize;23]` 自相矛盾（同归单指针）。

这直接消掉 5×E0308（gpu_backend_macro.rs:150/279/345/400/634 全部因 22 vs 23 而来）——但**不是把 22 改 23 凑数**（那是 NO-SILENT-FALLBACK 陷阱），而是两侧都归单指针。

## 3. Q3 KernelContext 在 GPU 如何传递 → **device global memory，每次 launch H2D 拷贝**

- host 侧把 22 字段按 **KernelContext repr(C) 布局**（abi_types.inc.rs:80-157 的 184 字节偏移）打包成 byte buffer，内部指针全部填 **device 地址**（weight_blob_gpu/kv_cache_gpu/… builder 已收集这些 device ptr）。
- `cuMemAlloc(184)` + H2D copy → 返回 device 指针，作唯一 kernelParam。PTX prologue 从该 base + 偏移 `ld.global` 读。
- **决策：global memory，不用 constant memory**（最终设计）。理由：KernelContext prologue 读一次入寄存器，无跨线程 broadcast 复用；constant memory 的优势（广播同址）在此不成立，且 `__constant__` 需模块内符号声明 + `cuMemcpyToSymbol`，JIT PTX 下徒增复杂度。global + L2 是正确且充分的选择。
- **决策：每 launch 重建 ctx buffer + H2D（正确性所需的完整设计）**。KernelContext 多数字段每 call 变化（seq_len/positions/output/kv 指针），必须每次重建才能保证正确性；跨 call 复用旧 buffer 会读到陈旧字段 = 正确性 bug。184B H2D 远小于 kernel 计算，无性能问题。
- **行为变更须注意**：现状 GPU 把 `seq_len/batch_size` 作内联 u32；单指针下 KernelContext 用 `seq_len_ptr`（device 指针指向值），与 CPU 语义一致。builder 需把这些标量落 device buffer 再存指针。

## 4. Q4 SPEC/40 ABI 是否适用 GPU → **适用，但 SPEC 当前 GPU 段过时/矛盾，须先修 SPEC**

- **适用**：REQ-UMK-20（统一 22 字段 ABI）+ REQ-UMK-27（CPU/GPU 统一）明确 GPU 用同一 KernelContext。
- **SPEC 缺陷（阻断，先修）**：
  - SPEC/40 REQ-E2E-007/008 定义 KernelContext 单指针，但**未说明 GPU 如何 marshal**（device memory / 拷贝时机 / 内部指针为 device 地址）——需补 GPU marshaling REQ。
  - SPEC/15-GPU-HOST-GLUE "ABI 映射" 段写 "CPU/GPU Mega-Kernel ABI (23 参数)"——**这是 R1 legacy，与 REQ-UMK-20 的单指针/22字段矛盾**，须整段替换为单指针 marshaling 规格。
  - → S2 SPEC 进化：新增/修订 REQ-GPU-ABI（KernelContext GPU marshaling：device global + 184B repr(C) + 内部 device 指针 + 单 kernelParam），废止 15 的 23 参数描述。**改代码前先落 SPEC。**

## 改动文件清单

| 领域 | 文件 | 动作 |
|------|------|------|
| SPEC(前置) | `SPEC/40-END-TO-END-DATA-FLOW.html` | 补 REQ：GPU KernelContext marshaling（device global/184B/device 指针/单 param） |
| SPEC(前置) | `SPEC/15-GPU-HOST-GLUE.html` | 废止 "23 参数" 段，替换为单指针 launch 规格 |
| PTX kernel | `gllm-kernels/.../gpu_lower/prologue.inc.rs:281-417` | mega-kernel 20-param → 1 个 ctx `.param .u64`，字段改 `ld.global` from ctx+offset |
| 偏移表(SSOT) | KernelContext field-offset 表（CPU/GPU 共享） | 抽为共享常量，GPU prologue 与 CPU 同源读 |
| args builder | `src/compat/gpu_helpers.rs:381-430` + `gpu_compile.rs` `to_mega_kernel_args` | `[usize;22]` flat → 打包 KernelContext repr(C) byte buf + H2D → 返回 `ctx_dev: u64`；标量落 device buf 存指针 |
| launch cuda | `src/compat/cuda_backend.rs:423-447` | `args:&[usize;23]` → `ctx_dev:u64`，kernelParams 长度 1 |
| launch hip | `src/compat/hip_backend.rs:212-` | 同上；消除 `[*mut c_void;21]` 矛盾 |
| launch metal | `src/compat/metal_backend.rs:360-` | `setBuffer(ctx,0,0)` 单绑定 |
| 调用点 | `src/compat/gpu_backend_macro.rs:150/279/345/400/634` | 传 ctx_dev 单指针（5×E0308 消除） |

## 与 SPEC/40 对齐方式

- KernelContext 字节布局 = abi_types.inc.rs:80-157 的 184B repr(C)（SPEC/40 REQ-E2E-007 SSOT），CPU/GPU **同一份偏移定义**。
- GPU = 把该 struct 放 device memory；CPU = 放栈上；**布局字节完全一致**，prologue 偏移读逻辑同构（唯 load 空间 param/stack vs global 不同）。
- 满足 REQ-UMK-20（统一 ABI）+ REQ-UMK-27（DeviceProfile 仅驱动 codegen 差异，不驱动 ABI 差异）。

## 实现优先级

- **P0**：修 SPEC（40 补 GPU marshaling + 15 废 23 参数）——SPEC-first 门控。
- **P1**：抽 KernelContext 偏移表为 CPU/GPU 共享 SSOT。
- **P2**：builder 改产 device ctx 指针（打包+H2D）。
- **P3**：PTX prologue 单指针偏移读。
- **P4**：三端 launch 单指针 + 调用点——消 5×E0308。
- **P5**：BCE 横扫——全仓搜其他 ABI arity 硬编码（20/21/22/23 魔法数），确认无残留同类漂移；GPU 数值对齐 E2E（5070Ti 验，本地 1060 可编译验证）。

## 待讨论 / 未决

1. **seq_len/batch_size 语义变更**：现内联标量 → 单指针下须走 device 指针（seq_len_ptr）。确认接受该行为变更（与 CPU 对齐所必需）。
2. **constant memory / ctx 缓存**：已在 §3 定为 global memory + 每 launch 重建（正确性所需的完整设计）。无未决。
3. **simplified 5-param kernel**：保留作 debug 路径，与 mega-kernel ABI 解耦。无未决。
4. GPU E2E 数值对齐验证需 5070Ti（192.168.1.200），本地 1060 仅够编译验证——这是验证环境约束，非交付范围裁剪。

## 证据表（file:line）

- CPU 单指针 ABI：`gllm/src/engine/mega_kernel/abi_types.inc.rs:274`；184B/22字段布局:80-157
- PTX mega-kernel 20 param：`gllm-kernels/src/compiler/codegen/vm/gpu_lower/prologue.inc.rs:294-316`；simple 5 param:226-232
- args builder [usize;22] flat：`gllm/src/compat/gpu_helpers.rs:381-430`
- launch [usize;23] + kernel_args[i]=&args[i]：`gllm/src/compat/cuda_backend.rs:427,439-441`；metal:364；hip:216
- 5×E0308（22 vs 23）：`gllm/src/compat/gpu_backend_macro.rs:150,279,345,400,634`
- cuLaunchKernel kernelParams 机制：`gllm-kernels/src/gpu/cuda/device.rs:482-507`；driver.rs:63-71
- CPU/GPU codegen 分叉点：`gllm-kernels/src/compiler/mod.rs:690(X86),922(Gpu)`；共享 IR:353,901
- REQ-UMK-20（统一22字段ABI）/ REQ-UMK-27（CPU/GPU统一，DeviceProfile驱动codegen）：SPEC/39-UNIFIED-MEGA-KERNEL.html
- SPEC/40 KernelContext 单指针（REQ-E2E-007/008）；SPEC/15 "23参数" 过时段
