# SYMDIM 动态维度 + PagedAttention 集成 (ARCH-SYMDIM-PAGED-KV)

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
PagedAttention 页表结构与分布式分页协同:
<a data-xref-id="REQ-DP-001" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-001">REQ-DP-001</a>
(分布式页表数据结构) 扩展本文件本地页表到跨节点 |
<a data-xref-id="REQ-DP-002" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-002">REQ-DP-002</a>
(页位置解析) 本地→同节点→跨节点三级查找 |
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr RemotePageLookup 扩展 mega-kernel page table 间接寻址)
</div>

## 定位

修复 `SYMDIM_MAX_SEQ_LEN = 2048` 硬编码 bug，并将已实现的 PagedAttention 调度层接通到 mega-kernel 推理路径。两个问题紧密耦合——修 SYMDIM 不修 PA，大上下文模型会 OOM；修 PA 不修 SYMDIM，PA 没有 page table 传入 JIT。

## 前置原则

- **ARCH-SYMDIM-NO-CONST-DEGRADE**: SymDim::Symbolic 的 max_value 必须从模型配置 `max_position_embeddings` 来，不是引擎硬编码
- **ARCH-PA-END-TO-END**: PagedAttention 必须端到端接通：PagedScheduler → page table → mega-kernel JIT attention
- **ARCH-PA-BACKWARD-COMPAT**: `page_table_ptr == NULL` 时使用连续 stride 寻址（向后兼容小上下文模型）
- **NO_SILENT_FALLBACK**: 请求 seq_len > model max_position_embeddings 时返回错误，不截断

## 根因链条

```
SYMDIM_MAX_SEQ_LEN = 2048 (硬编码在 gllm-kernels/src/compiler/graph.rs:19)
  → SymDim::Symbolic { max_value: 2048 } (所有 graph builder)
    → scratchpad activation buffer = 2048 × hidden × 4 bytes
    → KV cache stride = 2048 × kv_dim × 4 bytes
    → effective_kv_max_seq_len() = min(geom, 2048) (executor.rs:38)
    → 请求 > 2048 tokens → 内存越界
```

## 架构现状

### Mega-kernel KV 管理（decode 阶段）

```
Generate Loop:
  seq_len = prompt_len + gen_counter (每步 +1)
  融合组序列由图拓扑推导 (SPEC/39 §1.3.2)，典型 decoder 图为: embed融合组 → 层循环融合组 → lm_head融合组 → argmax → store token

Attention K/V 来源:
  prefill: K/V = QKV GEMM 输出 → 写入 scratchpad activation buffer
  decode:  K/V = QKV GEMM 输出（仅 1 行）+ 历史 K/V（从 scratchpad 读取）

问题: 所有上下文的 K/V 都在 scratchpad → 超过 2048 就越界
解决: PagedAttention — 历史 K/V 从 page pool 读取，不占 scratchpad
```

### PagedAttention 半成品现状

| 层 | 组件 | 状态 | 连接到推理 |
|---|------|------|----------|
| 调度层 | PagedScheduler | ✅ 实现了 | ❌ 只做记账 |
| 内存层 | CpuKvCacheBuffer | ✅ 实际使用 | ✅ 连续分配 |
| GPU 层 | GpuPagedKvMeta | ✅ 代码写了 | ❌ 从未调用 |
| JIT 层 | mega-kernel attention | ✅ 实际运行 | ✅ 连续 stride |

---

## Part A: SYMDIM 修复 — REQ-SYMDIM-001~009

### REQ-SYMDIM-001: CompilerGraph 携带 max_seq_len

`CompilerGraph` 新增 `pub max_seq_len: usize` 字段。`new()` 默认 2048（向后兼容）。`weight_layout()` 使用 `self.max_seq_len` 替代 `SYMDIM_MAX_SEQ_LEN`。`SYMDIM_MAX_SEQ_LEN` 标记 `#[deprecated]`。

**文件**: `gllm-kernels/src/compiler/graph.rs`

### REQ-SYMDIM-002: buffer_alloc 使用 graph.max_seq_len

`buffer_alloc.rs` L260, L521: `SYMDIM_MAX_SEQ_LEN` → `graph.max_seq_len`。

**文件**: `gllm-kernels/src/compiler/buffer_alloc.rs`

### REQ-SYMDIM-003: gllm-kernels graph_builders 参数化

3 个 graph builder 函数新增 `max_seq_len: usize` 参数。`SymDim::Symbolic { max_value: Some(max_seq_len) }`。返回前设置 `g.max_seq_len = max_seq_len`。

**文件**: `gllm-kernels/src/compiler/graph_builders.rs`

### REQ-SYMDIM-004: gllm compat graph_builders 传入模型值

3 个 builder 从 `config.geometry.max_seq_len` 取值替代 `SYMDIM_MAX_SEQ_LEN`。

**文件**: `gllm/src/compat/graph_builders.rs`

### REQ-SYMDIM-005: auto_graph 传入模型值

`ResolvedConfig` 新增 `pub max_position_embeddings: usize`。`from_geometry()` 从 `g.max_seq_len` 填充。`build_compiler_graph()` 使用配置值。MoEGate/TopK 的 `seq_len` 字段也从配置取。

**文件**: `gllm/src/arch/auto_graph.rs`, `gllm/src/arch/resolve.rs`

### REQ-SYMDIM-006: ONNX graph_convert 传入 max_seq_len

`onnx_to_compiler_graph()` 新增 `max_seq_len: usize` 参数。`ConvertContext::new()` 用参数替代 `SYMDIM_MAX_SEQ_LEN`。

**文件**: `gllm/src/loader/onnx/graph_convert.rs`

### REQ-SYMDIM-007: Semantic Gatekeeper 使用模型值

`SmallGraphCompiler` 新增 `max_seq_len: usize` 字段。构造时从模型配置传入。`sym_seq()` 用 `self.max_seq_len`。运行时检查 `tokens.len() > self.max_seq_len` 替代硬编码 2048。

**文件**: `gllm/src/semantic_gatekeeper/small_graph.rs`

### REQ-SYMDIM-008: 移除 effective_kv_max_seq_len 截断

删除 `effective_kv_max_seq_len()` 函数。所有调用方直接使用 `self.geometry.max_seq_len`。

**文件**: `gllm/src/engine/executor.rs`

### REQ-SYMDIM-009: mega_kernel output buffer 使用模型值

`compile()` 中 `SYMDIM_MAX_SEQ_LEN` → 已有的 `max_seq_len` 参数（从 `geometry.max_seq_len` 传入，SPEC/39 统一编译入口）。

**文件**: `gllm/src/engine/mega_kernel.rs`

---

## Part B: PagedAttention 接通 — REQ-PA-001~007

### REQ-PA-001: Mega-kernel ABI 新增 page_table_ptr 参数

ABI 新增 `page_table_ptr` 参数（当前 23 参数 ABI 中的 arg 21）。`page_table_ptr` 指向 `u32[]` page table（`page_table[seq_pos] = physical_page_id`）。当 `NULL` 时使用连续 stride（向后兼容）。

**文件**: `gllm-kernels/src/compiler/mega_kernel_abi.rs`, `gllm/src/engine/mega_kernel.rs`, `gllm/src/compat/gpu_backend_macro.rs`

### REQ-PA-002: JIT attention 支持 page table 间接寻址

新增 VmInstr:
- `PageTableLookup { dst, page_table_ptr, seq_index, page_size, head_dim, kv_half, layer, num_kv_heads }` — 从 page table 读取物理 page id → 计算物理偏移
- `PageTableWrite { src, page_table_ptr, seq_index, page_size, head_dim, kv_half, layer, num_kv_heads }` — 写入 page pool

ISA Lowering:
- x86: `mov eax, [page_table + seq_index*4]; imul rax, page_stride; add rax, pool_base; add rax, offset_in_page`
- GPU PTX: `ld.global.u32 %r, [page_table + %seq*4]; mul.wide.u64 %addr, %r, page_stride; add.u64 %addr, pool_base`
- AArch64: LDR + UMADDL + ADD

修改 `emit_tiled_attention_inline` ki loop：当有 page table 时用 `PageTableLookup` 替代 `k_head + ki_off`。

**文件**: `gllm-kernels/src/compiler/codegen/vm/instr.rs`, `plan_lower.rs`, `x86_lower.rs`, `gpu_lower.rs`, `aarch64_lower.rs`

### REQ-PA-003: KV cache 写入支持 page table

decode 步骤的 K/V projection 结果写回 page pool（不是 scratchpad）。只写当前 token 的 K/V 行（`seq_index = total_seq - 1`）。写入位置 = `pool_base + phys_page_id * page_stride + offset_in_page`。

**文件**: 同 REQ-PA-002

### REQ-PA-004: Executor 接通 PagedScheduler → Mega-kernel

`generate_single_sequence()` 新增 `page_table: Option<Vec<u32>>` 参数。从 `PagedScheduler` 获取当前 sequence 的 page table。CPU 路径: `page_table.as_ptr()` 或 `null`。GPU 路径: `upload_to_gpu(&page_table)` 然后 device ptr。

**文件**: `gllm/src/engine/executor.rs`, `gllm/src/engine/mega_kernel.rs`

### REQ-PA-005: PagedScheduler 构建 page table

`PagedScheduler` 新增 `fn get_page_table(&self, seq_id: u64) -> Vec<u32>` 方法。遍历 sequence 的 block table: `page_table[i] = blocks[i / page_size]`。

**文件**: `gllm/src/scheduler/paged_scheduler.rs`

### REQ-PA-006: Paged KV pool 分配和管理

新增 `PagedKvPool` 结构体: `pool_ptr`, `page_stride`, `num_pages`。CPU 用 `vec![0u8; num_pages * page_stride]`。GPU 用 `device.alloc()`。替代 `CpuKvCacheBuffer`（当启用 PagedAttention 时）。内存计算: `page_stride = num_layers * 2 * num_kv_heads * page_size * head_dim * elem_bytes`。

**文件**: `gllm/src/compat/cpu_backend.rs`, `gllm/src/compat/gpu_helpers.rs`

### REQ-PA-007: 内存安全护栏

启动时: 计算连续模式和 Paged 模式的内存需求，如果连续模式 > 可用内存 80% 则强制 Paged。运行时: `prompt_tokens.len() + max_new_tokens <= geometry.max_seq_len`。

**文件**: `gllm/src/engine/executor.rs`

---

## 实施顺序

```
Phase 1 (gllm-kernels): REQ-SYMDIM-001~003
Phase 2 (gllm):         REQ-SYMDIM-004~009
Phase 3 (VmInstr):      REQ-PA-002~003
Phase 4 (ABI):          REQ-PA-001
Phase 5 (Scheduler):    REQ-PA-004~006
Phase 6 (验证):         REQ-PA-007
```

## 验证

```bash
cargo check && cargo test --lib
cargo test --test test_e2e_generator -- --test-threads=1
```
