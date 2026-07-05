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
