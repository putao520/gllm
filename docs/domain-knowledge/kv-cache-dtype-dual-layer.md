# KV Cache dtype 双地层陷阱（C-9，堵 AI 幻觉）

> 来源：gllm 源码（abi_types.inc.rs + context.inc.rs + build_graph.inc.rs + lower_op.inc.rs）+ architect 归因（sessionId edb98acd-f28a-466b-ad97-863e9056c3b7）
> 建库触发：8 轮 CPU BUG 诊断 + SmolLM2 logits 发散（argmax=967, cosine=-0.465）根因——KV cache dtype 在 JIT 层与 buffer 层解耦不一致
> 最后验证：2026-07-05

## 核心陷阱：JIT 层 dtype 与 buffer 层 dtype 是两个独立来源

gllm 有**两个 dtype 来源**，它们解耦，必须保持一致否则致命：

| 层 | dtype 来源 | 决定什么 | SmolLM2 值 |
|----|----------|---------|-----------|
| **JIT 层** | `ctx.dtype = graph_dtype()` 硬编码 F32（context.inc.rs:179） | GEMM c_dtype、激活 dtype、AttentionSpec.dtype、MemCopy dtype + stride | **F32** |
| **buffer 层** | `compute_dtype = config.compute_dtype.unwrap_or(config.dtype)`（types.inc.rs:167） | KV cache / scratchpad buffer 的 elem_bytes + stride | **BF16**（torch_dtype） |

**陷阱**：JIT 层硬编码 F32，buffer 层从 config 推断 BF16 → 两者不一致 → KV cache buffer 按 BF16（384 bytes/行）分配，但 MemCopy 写/读按 F32（768 bytes/行）→ **越界踩踏**。

## 致命不一致链路（SmolLM2 实例）

```
config.json torch_dtype=bfloat16
  → config.dtype = BF16 (config_impl.inc.rs:401)
    → compute_dtype = unwrap_or(config.dtype) = BF16 (types.inc.rs:167)  [buffer 层]
      → MegaKernelCompiled.compute_dtype = BF16 (executor_core.inc.rs:371)
        → abi_types.inc.rs:395 elem_bytes() = compute_dtype.size_bytes() = 2
          → abi_types.inc.rs:469 kv_row_stride() = 3*64*2 = 384  [buffer 分配]
          → abi_types.inc.rs:485 kv_cache_bytes = num_layers*2*max_seq*384

但 JIT 层:
  context.inc.rs:179 ctx.dtype = graph_dtype() = F32 硬编码  [JIT 层]
    → build_graph.inc.rs:693 AttentionSpec.dtype = DType::F32 硬编码
      → lower_op.inc.rs:1488 dtype = spec.dtype = F32
        → lower_op.inc.rs:1521 kv_row_stride = 3*64*4 = 768  [MemCopy 写读]
        → lower_op.inc.rs:1541 MemCopy{bytes:768, dtype:F32}  [写 768 进 384 行 → 越界]
```

**结果**：每行 KV 写 768 字节进 384 字节 buffer → 溢出覆盖下一行 V / 下一 layer K → 30 层逐层 2× stride 越界踩踏 → attention Q·K 全错 → logits 发散（cosine=-0.465, argmax=967）。

## MemCopy 纯字节搬，忽略 dtype（源码确认）

**lower_instr_dispatch.inc.rs:1220-1242 `lower_mem_copy_x86`**：
```rust
VmInstr::MemCopy { dst, src, bytes, dtype: _, guard, effect } => {  // dtype 被忽略!
    for off in (0..b).step_by(8) {
        mov rax, [src+off]; mov [dst+off], rax;  // 纯 8 字节搬，不转换
    }
}
```
- `dtype: _` —— MemCopy lower **完全忽略 dtype**
- 逐 8 字节 mov，纯 memcpy，不做任何 dtype 转换
- dtype 字段只影响 emit 端的 stride 计算（lower_op.inc.rs:1521 `kv_row_stride = ... * dtype.elem_bytes()`）

**修正原描述**：
- ❌ "MemCopy dtype=F32 不 narrow → BF16 buffer 存 F32 字节" —— 错，MemCopy 不看 dtype
- ✅ MemCopy 按 `bytes=768`（F32 stride 算的）逐字节搬，但 KV cache buffer 每行只有 384 字节 → 768 字节搬进 384 字节行 = 越界覆盖下一行

bug 本质是 **stride 不一致（768 vs 384）导致越界**，不是 dtype 转换问题。

## attention 读 KV cache 用 F32（读写 stride 一致，但都 > buffer 分配）

**attention_emit.rs:122** `VecLoad { dst, base: k_row, offset: d_off, width, dtype }` —— dtype=F32，按 F32 读 KV cache。

**读写一致分析**：
- **写**：MemCopy 按 `bytes=768`（F32 stride）纯字节搬
- **读**：VecLoad 按 `dtype=F32` 读，每行 768 字节（F32 stride）
- 写读 stride 一致（都 768），**如果 buffer 够大**，数据格式自洽（F32 字节存 F32 读）
- 但 buffer 按 384 分配 → 写 768 溢出覆盖下一行，读 768 读到下一行数据

**结论**：bug 不是"读 F32 写 BF16 格式不匹配"，而是"buffer 分配 384 vs 写读 stride 768 越界"。若 buffer 按 768 分配（compute_dtype=F32），则全链自洽（虽然 KV cache 内存翻倍，但正确）。这就是方案 B（止血）的原理。

## 方案 A 的复杂性（不只是改 graph_dtype）

统一到 compute_dtype=BF16 后，需要：
- buffer 按 BF16（384）分配 ✅ 已是
- MemCopy 按 BF16 stride（384）搬 + **需要 narrow F32→BF16**（因为 k_out 是 F32）—— 但 MemCopy 当前 `dtype: _` 忽略 dtype，不会 narrow
- VecLoad 按 BF16 读 + widen BF16→F32 计算

**方案 A 不能只改 graph_dtype()**，还要：
1. MemCopy 改成支持 dtype 转换（看 dtype 字段，F32→BF16 时 narrow）
2. 或改用专门的 narrow op 替代 MemCopy
3. attention VecLoad dtype 改 compute_dtype

## 方案 A 会激活 VecNarrow lane-loss bug（次生风险，源码确认）

**gemm_emit.rs:317** `needs_narrow = c_dtype.needs_narrowing_from(acc_dtype)`，:393-396 如果 needs_narrow，emit `VecNarrow`。

| 配置 | c_dtype | acc_dtype | needs_narrow | VecNarrow |
|------|---------|-----------|-------------|-----------|
| 当前 | F32 | F32 | false | 不触发（lane-loss 是死代码） |
| 方案 A 后 | BF16 | F32 | **true** | **每个 GEMM store 都触发** |

方案 A 后每个 GEMM（q/k/v/o_proj + lm_head + FFN）的 store 都走 `VecNarrow { dst_dtype: BF16, src_dtype: F32 }` → AVX2 路径调 `emit_f32_to_bf16_ymm_to_xmm_avx2`（lane-loss bug：8 lanes F32 只窄化低 4 lanes，高 4 丢失）。

**lane-loss 修复方案**（与方案 A 同时做）：`emit_f32_to_bf16_ymm_to_xmm_avx2` 用 `vextracti128` 取 ymm 高半，正确 pack 8 lanes 成 BF16。当前只 pack 低 4 lanes（`ymm_to_xmm` 丢高半）。

**结论**：方案 A 必须同时做 4 项，否则激活 lane-loss：
1. graph_dtype() 返回 compute_dtype
2. MemCopy 支持 dtype 转换（或改用 narrow op）
3. attention VecLoad dtype 改 compute_dtype + widen
4. 修 emit_f32_to_bf16_ymm_to_xmm_avx2 lane-loss（vextracti128 取高半）

需 architect 给完整改动清单 + 评估全链影响。

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码 + architect 证明） |
|--------|---------|
| compute_dtype=BF16 → GEMM 输出 BF16 → 触发 VecNarrow | JIT 层 ctx.dtype=F32 硬编码，GEMM c_dtype=F32，激活 F32，**前向路径零 VecNarrow**（needs_narrowing_from(F32)=false） |
| VecNarrow lane-loss（8 lanes 只窄化低 4）是真因 | VecNarrow 根本不执行，lane-loss 是死代码（emit_f32_to_bf16_ymm_to_xmm_avx2 search_code profile: 0 consumers，实际在 VecNarrow 路径但 SmolLM2 不触发） |
| KV cache 是 F32（因为激活 F32） | KV cache 按 compute_dtype=BF16 分配（abi_types elem_bytes=compute_dtype.size_bytes） |
| AttentionSpec.dtype 跟 compute_dtype 走 | AttentionSpec.dtype 硬编码 F32（build_graph.inc.rs:693），与 compute_dtype 无关 |
| MemCopy dtype 字段会自动转换 | MemCopy 按 dtype 字节拷，dtype=F32 就 4 字节为单位搬，不 narrow（需查 lower 实现确认） |
| 7月3日"本地 argmax=253 正确"证明 bug 不在 KV cache | 该记录证据不足（未跑 gllm-vs-golden 端到端数值断言，同 BCE-20260704 重开原因） |

## 两个候选根因的区分（architect 裁决）

| 候选 | 描述 | 裁决 |
|------|------|------|
| A（KV cache dtype 双地层裂开） | JIT 层 F32 vs buffer 层 BF16 → stride 768 vs 384 越界踩踏 | **✅ 唯一致命根因**（architect 6 Agent 交叉验证） |
| B（VecNarrow lane-loss） | emit_f32_to_bf16_ymm_to_xmm_avx2 8 lanes 只窄化低 4 | ❌ 排除（VecNarrow 不执行，前向路径零调用） |

## 修复方案（architect 给出，待用户确认）

| 方案 | 内容 | 影响面 | 架构 |
|------|------|--------|------|
| **A（推荐）** | graph_dtype() 返回 compute_dtype（非硬编码 F32）→ 全链 BF16，SSOT 统一 | 所有模型（GEMM c_dtype + 激活全变）需全量回归 | 根治 |
| B（止血） | buffer 层 elem_bytes 硬编码 4（不读 compute_dtype）→ 全链 F32 自洽 | 仅 KV cache buffer | 局部 patch，违反 compute_dtype 语义 |

按 ARCH-JIT-DATA-YIELDS（代码顺从数据实际 dtype）+ C-3（根治优先）应选 A。

## 关键代码位置

- `gllm-kernels/src/compiler/codegen/vm/plan_lower/context.inc.rs:179`：`ctx.dtype = graph_dtype()` 硬编码 F32（JIT 层源头）
- `gllm/src/model_config_fragments/types.inc.rs:167`：`compute_dtype = config.compute_dtype.unwrap_or(config.dtype)`（buffer 层源头）
- `gllm/src/engine/mega_kernel/abi_types.inc.rs:395,469,485`：elem_bytes/kv_row_stride/kv_cache_bytes 按 compute_dtype（buffer 分配）
- `gllm/src/arch/auto_graph_fragments/build_graph.inc.rs:693`：`AttentionSpec.dtype = DType::F32` 硬编码（写读层）
- `gllm-kernels/src/compiler/codegen/vm/plan_lower/lower_op.inc.rs:1488,1521,1541,1556`：MemCopy 按 spec.dtype=F32（stride=768）
- `gllm/src/model_config_fragments/config_impl.inc.rs:401`：torch_dtype → config.dtype 映射

## 与其他资料库关系

- `dtype-propagation.md`：BF16↔F32 widen/narrow（本库是其 KV cache 应用场景的陷阱）
- `mega-kernel-topology.md`：GenerateLoop M=1（本库是 KV cache 在该拓扑下的 dtype 陷阱）
- `smollm2-135m-architecture.md`：SmolLM2 BF16 权重事实（本库根因的模型侧输入）
- 本文件：KV cache dtype 在 JIT 层与 buffer 层解耦不一致的陷阱
