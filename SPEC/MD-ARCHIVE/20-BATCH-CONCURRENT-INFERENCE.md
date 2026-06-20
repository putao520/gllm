# 批量并发推理 — M 维度统一架构 (Batch Concurrent Inference)

> **SSOT 声明**: 本文档定义 gllm 从 `batch_size=1` 扩展到 `batch_size=N` 的统一推理架构。
> 核心原则: **多序列 forward 是 M 维度更大的单序列 forward，不是另一种模型。**

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
批量并发与分布式通信协同:
<a data-xref-id="REQ-ALG-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-001">REQ-ALG-001</a>
<a data-xref-id="REQ-ALG-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-002">REQ-ALG-002</a>
<a data-xref-id="REQ-ALG-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-ALG-003">REQ-ALG-003</a>
(Ring/Tree/Pipeline AllReduce) 消费本文件 BatchContext per-seq_id 机制进行分布式 batch 归约 |
<a data-xref-id="REQ-FLUX-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-FLUX-001">REQ-FLUX-001</a>
<a data-xref-id="REQ-FLUX-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-FLUX-002">REQ-FLUX-002</a>
(Over-Decomposition + Pipeline) 与本文件 Prefill M=sum(prompt_lens) 分块策略协同
</div>

## 0. 核心洞察

### 0.1 batch_size=1 是 batch_size=N 的特例

当前 mega-kernel 的 forward pass 处理 `M × K` 矩阵乘法：

```
单序列 decode:  M = 1
单序列 prefill: M = prompt_len
多序列 decode:  M = num_active_seqs
多序列 prefill: M = sum(prompt_lens)
```

GEMM、FFN、Norm — 这些层对 M 维度 uniform，不关心 token 属于哪条序列。
只有 Attention 需要 per-token 查找所属序列的 KV cache。

**结论**: batch 并发不是新系统，是 M 维度从 1 变成 total_tokens + Attention 增加 per-token seq_id 查找。

### 0.2 与 vLLM/SGLang 的本质区别

| 功能 | vLLM/SGLang | gllm |
|------|------------|------|
| 推理调度 | Python 逐 step dispatch，每 step 独立 kernel launch | 单次 CALL 完成 M 条序列全部生成 |
| Prefill/Decode 混合 | 两个不同 kernel + 运行时切换 | 同一 mega-kernel，M 维度自然变化 |
| Variable-length Attn | FlashInfer CSR indptr 运行时数组 | JIT 编译的 per-token seq_id 查找指令 |
| PagedAttention | block_table 2D 数组 + Python 构建 | VmInstr::PageTableAddr 一条机器指令 |
| Per-seq sampling | Python 控制流 | JIT 编译的 per-seq argmax |
| KV 共享 | RadixAttention trie | KvPrefixIndex + CoW 页映射（已实现） |
| Stop condition | 外层 Python 循环检查 | JIT CheckStopCondition 内嵌 |

**核心差异**: 它们每 step 做 1 次 kernel dispatch × N 层 = 大量 host↔device 往返。
我们 1 次 CALL 完成全部序列的 prefill + decode 全生命周期，零 host 介入。

### 0.3 已有的 batch 基础设施

| 组件 | 已有能力 | 批量化改造 |
|------|---------|-----------|
| **MegaKernelFn ABI** | `batch_size` (arg 5), `page_table_ptr` (arg 21) | batch_size 从忽略变为驱动 forward M 维度 |
| **SymDim + emit_loop** | `BoundExpr::Symbolic` 完整符号维度循环 | batch_size 是另一个 SymDim，与 seq_len 同等 |
| **VmInstr** | `PageTableAddr`, `PageTableKVWrite`, `StoreToken`, `CheckStopCondition`, `Argmax` | `seq_index` 参数化（当前固定 0） |
| **MegaKernelBufferLayout** | ping/pong activation + logits + sampling | M 维度 sizing 改为 total_tokens |
| **ContinuousBatcher** | 多请求调度 + decode 优先 + prefill backfill | 返回扩展为 BatchContext |
| **PagedScheduler** | 多请求页管理 + `get_page_table()` | 批量导出拼接页表 |
| **RaggedCompaction** | 三阶段 compact/execute/scatter + RequestActiveMask | 接入 mega-kernel active mask |
| **EpilogueSubsystem** | 批量遥测 per-request decision array | 已是 batch-ready |
| **VariantRegistry** | 机制级编译分发 + constraint relaxation | 新增 batch_size 维度 |
| **KvPrefixIndex** | token 级 radix tree + prefix match + CoW | 已支持多请求 KV 共享 |
| **KV Optimizer** | per-page importance + PrecisionTier | 已是 batch-ready |
| **SessionKvCache** | per-session finalized_position + cross-turn reuse | per-seq session offset |

---

## 1. BatchContext — flat memory block

### 1.1 设计原则

JIT 代码只能访问 raw 指针 + 编译时已知 offset。BatchContext 是一块 **flat memory**，
不是 Rust struct（不能有 `Vec`、`String`、指针间接）。JIT 编译时知道每个字段的精确偏移。

### 1.2 内存布局

```
BatchContext flat memory layout (由 Executor 分配):

Offset  Field                      Size                   说明
──────  ──────────────────────────  ─────────────────────  ──────────────────────
0       num_seqs                   4 bytes (u32)          活跃序列数
4       max_decode_steps           4 bytes (u32)          最大剩余 decode 步数
8       total_prefill_tokens       4 bytes (u32)          sum(prompt_lens)
12      pad                        4 bytes
16      input_ids_flat_ptr         8 bytes (*const u32)   所有序列 prompt 拼接
24      output_tokens_flat_ptr     8 bytes (*mut u32)     所有序列输出拼接
32      positions_ptr              8 bytes (*const u32)   位置数组
40      page_table_flat_ptr        8 bytes (*const u32)   所有序列页表拼接
48      kv_pool_base               8 bytes (*const u8)    KV 物理池基址
56      sampling_params_ptr        8 bytes (*const u32)   packed: [temp, top_k, top_p, eos] × N
64      hook_ctx_ptr               8 bytes (*const u8)    SG shared memory (共享)
72      callback_table_ptr         8 bytes (*const u8)    callback table (共享)
80      seq_meta_base              ── per-seq 数组起始 ──
  +seq_idx × SEQ_META_STRIDE:
    +0   prompt_len                4 bytes (u32)          0 = 纯 decode
    +4   kv_len                    4 bytes (u32)          已有 KV token 数
    +8   rope_pos_offset           4 bytes (u32)          session resume 偏移
   +12   max_new_tokens            4 bytes (u32)          该序列最大生成 token 数 (>0)
   +16   session_position          4 bytes (u32)          0 = 新序列
   +20   page_table_offset         4 bytes (u32)          该序列在 page_table_flat 中的起始
   +24   page_table_len            4 bytes (u32)          该序列页表条目数
   +28   fused_hidden_offset       4 bytes (u32)          多模态注入偏移
   +32   num_mm_tokens             4 bytes (u32)          多模态 token 数
   +36   active_flag               4 bytes (u32)          1=活跃, 0=已完成
   +40   seq_position              4 bytes (u32)          当前 decode 位置 (运行时更新)
   +44   gen_count                 4 bytes (u32)          已生成 token 数 (运行时更新)
   +48   last_sampled_token        4 bytes (u32)          上次采样结果 (运行时更新)
   +52   output_offset             4 bytes (u32)          该序列在 output_tokens_flat 中的起始偏移
  SEQ_META_STRIDE = 56 bytes (对齐到 64)
```

**关键**: 所有 per-seq 元数据通过 `seq_meta_base + seq_idx × SEQ_META_STRIDE` 访问（SEQ_META_STRIDE = 56 bytes，见上方物理布局表推导：14 个 u32 字段 = 56 bytes）。
JIT 编译时知道 offset，运行时只读一个 u32。

### 1.3 构建流程

```
ContinuousBatcher.build_batch()
  → ScheduledBatch { requests, seq_offsets, draft_steps }
     ↓
Executor.build_batch_context(scheduled_batch):
  1. 分配 flat memory block (大小 = 固定头 + num_seqs × 52)
  2. PagedScheduler.get_page_table(req) × N → 拼接 page_table_flat + 填充 page_table_offset/len
  3. session_position[seq] → rope_pos_offset, session_position (复用 SessionKvCache)
  4. request.sampling_params → packed sampling_params
  5. seq_offsets → input_ids_flat 拼接
  6. waste_ratio → compact_required 标记 (驱动 RaggedCompaction)
     ↓
  BatchContext (flat memory block ptr 传入 mega-kernel ABI)
```

---

## 2. Mega-Kernel 统一 generate 循环

### 2.1 当前单序列结构

```
compile() 当前产出:
  Phase 0: Load ABI params
  Phase 1: Compute derived values (prompt_len_bytes, input_base)
  Phase 2: LoopBegin { bound: max_new_tokens }         ← generate 循环
  Phase 3: Compute input_ptr = input_base + gen_offset  ← 单序列: 固定偏移
  层循环融合组: emit_fusion_groups (融合组序列由图拓扑推导)
  Argmax/StoreToken 融合组: Argmax → sample
  StoreToken: StoreToken
  CheckStopCondition 融合组: CheckStopCondition
  generate loop end: LoopEnd
```

### 2.2 统一 batch 结构

```
编译时已知:
  - SEQ_META_STRIDE = 56
  - seq_meta_base offset in BatchContext
  - 各 per-seq field offset (prompt_len=0, kv_len=4, ...)

运行时生成:
Phase 0: Load ABI params (不变)
Phase 1: Compute derived values
         + 从 AbiArg(5) 加载 batch_size → SymDimSlotMap["batch_size"]
         + 从 arg 22 加载 batch_ctx_ptr → 基址寄存器
         + 如果 batch_ctx_ptr == NULL → 跳转到 Phase 1.Legacy (当前单序列逻辑)

Phase 1.Legacy: (向后兼容, batch_ctx_ptr == NULL 时执行)
         当前 Phase 0-3 + 融合组逻辑不变, goto DONE

Phase 2: Prefill — M = total_prefill_tokens
         + M_prefill = load [batch_ctx + 8] (total_prefill_tokens)
         + if M_prefill == 0 → 跳到层循环融合组 (全部 decode)
         + input_ptr = load [batch_ctx + 16] (input_ids_flat_ptr)
         + M = M_prefill → SymDim("prefill_M")
         + emit_fusion_groups (融合组序列由图拓扑推导)
             Attention 内部: per-token seq_id 查找 (§3)
         + per-seq argmax on last token of each sequence's prompt
         + per-seq store: first_sampled_token[seq] = argmax(logits[seq_last_token])
         + per-seq update: gen_count[seq] = 1, seq_position[seq] = prompt_len[seq]

Phase 3: Decode Step Loop
         + max_steps = load [batch_ctx + 4] (max_decode_steps)
         + LoopBegin { bound: max_steps }
         + num_active = compute from active_flag array
         + if num_active == 0 → break
         + M = num_active → SymDim("decode_M")
         + 构建 decode input_ids: last_sampled_token[seq] × num_active
         + emit_fusion_groups (融合组序列由图拓扑推导)
             Attention 内部: per-token seq_id 查找 (§3)
         + per-seq argmax, store, update seq_position/gen_count/last_sampled_token
         + per-seq CheckStopCondition (eos match or gen_count >= max_new_tokens)
         + per-seq: if stopped → active_flag[seq] = 0
         + LoopEnd

层循环融合组（decode 路径）: DONE — return total_generated_count
```

**关键**: Phase 2 (Prefill) 和 Phase 3 (Decode) 的 `emit_fusion_groups` 是**同一个 forward pass 函数**，
只是 M 维度不同。GEMM、FFN、Norm 完全不关心 M 里面是几条序列。

### 2.3 batch_size=1 时的行为

当 `batch_ctx_ptr == NULL`:
- 走 Phase 1.Legacy → 当前单序列代码路径
- 零行为变更，零性能影响

当 `batch_ctx_ptr != NULL` 但 `num_seqs == 1`:
- Phase 2: M = prompt_lens[0]（和当前一样）
- Phase 3: M = 1, step 循环跑 max_new_tokens 次（和当前一样）
- 等价于当前行为，仅多几条 seq_id 查找指令（可优化掉）

### 2.4 Scratchpad 布局 (M 维度参数化)

```
MegaKernelBufferLayout 扩展:

  Activation A (ping): max_M × hidden         ← M 维度参数化
  Activation B (pong): max_M × hidden         ← M 维度参数化
  Logits:              max_M × vocab_size     ← M 维度参数化
  Sampling workspace:  vocab_size × 4         ← 固定大小
  SG data:             hidden × 2             ← 固定大小
  RoPE cache:          max_total × head_dim × 2  ← 共享 (按最大序列)
```

`max_M` 在编译时由 Variant 的 `golden_size` 决定，运行时通过 SymDim 读取实际 M 值。
Prefill 阶段 M = total_prefill_tokens，Decode 阶段 M = num_active_seqs。
Activation 复用同一块 buffer，两个阶段不重叠（时序互斥）。

---

## 3. Attention Per-Token seq_id 查找

### 3.1 核心机制

GEMM / FFN / Norm 对 M 维度 uniform，不需要知道 token 属于哪条序列。
只有 Attention 需要 per-token 查找 seq_id 以定位该序列的 KV cache。

在 Prefill 阶段，token_i 属于哪条序列由 `prompt_lens` 累积推导:

```
seq_id[token_i] = 最小的 s 使得 cumsum(prompt_lens)[s] > token_i
```

在 Decode 阶段，M = num_active_seqs，token_i 直接对应第 i 条活跃序列。

### 3.2 JIT 编译的 seq_id 查找

Prefill 阶段 — cumsum 搜索:
```asm
; 输入: rax = token_index (0..M_prefill)
; 输出: rcx = seq_id
xor ecx, ecx                    ; seq_id = 0
mov edx, [batch_ctx + seq_meta_base + 0]  ; cumulative = prompt_lens[0]
.loop:
  cmp rax, rdx
  jl  .found
  inc ecx
  mov edx, [batch_ctx + seq_meta_base + ecx*52 + 0]  ; prompt_lens[seq_id+1]
  add edx, edx_prev              ; cumulative += prompt_lens[next]
  jmp .loop
.found:
; rcx = seq_id
```

Decode 阶段 — 直接映射:
```asm
; token_i → seq_id = active_seq_ids[token_i]
; (active_seq_ids 是 compact 后的活跃序列索引数组，在 scratchpad 中维护)
mov ecx, [active_seq_ids + rax * 4]
```

### 3.3 PageTableAddr 扩展

当前 `VmInstr::PageTableAddr` 的 `seq_index: Option<VRegId>` 固定为 `None`（单序列）。
改为接受 seq_id 寄存器后:

```asm
; 计算 seq_id 对应的 KV 物理地址
; rcx = seq_id, eax = token_pos (该序列内的逻辑位置)
mov edx, [batch_ctx + seq_meta_base + rcx*52 + 20]  ; pt_offset = page_table_offset[seq]
mov esi, eax
shr esi, PAGE_SHIFT                                 ; logical_page = pos >> shift
mov r8d, [page_table_flat + rdx + rsi*4]           ; phys_page = pt_flat[pt_offset + logical_page]
imul r8d, PAGE_BYTES
add r8, kv_pool_base                                ; final_addr
```

**不需要新增 VmInstr**。只需 `PageTableAddr.seq_index` 从 `None` 改为 `Some(VRegId)`，
JIT 生成多一条 pt_offset 读取指令。

### 3.4 KV 写入

每层 forward 需要将当前 token 的 K/V 写入该序列对应的 KV cache page。
Prefill 阶段每条序列写 prompt_len 条 KV，Decode 阶段每条写 1 条。
写入地址通过 `PageTableAddr` + `seq_id` 计算，和读取同一套寻址逻辑。

---

## 4. Per-Sequence 采样与停止

### 4.1 Per-Sequence Argmax

Prefill 阶段: 每条序列取自己 prompt 最后一行的 logits 做 argmax。
Decode 阶段: 每条序列取自己那行 logits 做 argmax。

Logits 布局: `[M, vocab_size]`。每条序列的 logits 行在 M 维度中有固定偏移。
argmax 只扫描该序列对应的行，结果写入 `last_sampled_token[seq]`。

### 4.2 Per-Sequence Stop Check

```
for seq in 0..num_seqs:
  if active_flag[seq] == 0: continue
  if last_sampled_token[seq] == eos_token_ids[seq]:
    active_flag[seq] = 0
  if gen_count[seq] >= max_new_tokens[seq]:
    active_flag[seq] = 0
```

全部在 JIT 机器码中完成。序列完成后从 active 集合移除，M 维度缩小。
全序列完成后 generate 循环自然终止。

### 4.3 Per-Sequence 采样参数

从 `sampling_params_ptr` 读取 packed 参数:
```
[temperature_u32, top_k, top_p_u32, eos_token_id] × num_seqs
stride = 16 bytes
```

JIT 编译时知道 stride，运行时 `mov edx, [sampling_params + seq * 16]` 读取。

---

## 5. ABI 扩展

### 5.1 当前 ABI (23 参数)

> **SSOT**: 完整 23 参数定义见 `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §1.5.5` 和 `SPEC/15-GPU-HOST-GLUE.md`。

```
arg 0:  input_ids_ptr      *const u32    → rdi
arg 1:  weight_blob_ptr    *const u8     → rsi
arg 2:  kv_cache_ptr       *mut u8       → rdx
arg 3:  positions_ptr      *const u32    → rcx
arg 4:  aux_ptr            *const u8     → r8
arg 5:  batch_size         usize         → r9
arg 6:  prompt_len         usize         → [rbp+16]
...
arg 20: callback_table_ptr *const u8     → [rbp+128]
arg 21: page_table_ptr     *const u32    → [rbp+136]
arg 22: batch_ctx_ptr      *const u8     → [rbp+144]
```

### 5.2 新增参数

`batch_ctx_ptr` 是 23 参数 ABI 的 arg 22（与 `page_table_ptr` arg 21 同批次扩展，见 SPEC/18 REQ-PA-001）。

- `batch_ctx_ptr == NULL` → 单序列模式，走 Phase 1.Legacy（当前逻辑不变）
- `batch_ctx_ptr != NULL` → batch 模式，从 flat memory 读取所有 per-seq 元数据

**向后兼容**: 现有所有调用点传 NULL 即可，零行为变更。

### 5.3 复用已有参数

| 需求 | 已有参数 | batch 模式下的用途 |
|------|---------|------------------|
| batch 循环 | arg 5 `batch_size` | 从忽略变为驱动 forward M 维度 |
| per-seq 状态 | arg 7 `scratchpad` | scratchpad 内 M 维度参数化 |
| 页表 | arg 21 `page_table_ptr` | 从单序列页表变为 batch 拼接页表的 fallback |
| per-seq sampling | batch_ctx 内 `sampling_params_ptr` | 替代 arg 9/10/11 的单序列参数 |
| per-seq prompt_len | batch_ctx 内 `seq_meta.prompt_len` | 替代 arg 6 的单序列参数 |

batch 模式下，arg 6/9/10/11/13/17 从直接使用变为忽略（从 batch_ctx 读取 per-seq 值）。

---

## 6. Variant 扩展

### 6.1 VariantKey 新增 batch 维度

```rust
pub struct VariantKey {
    // ... 已有字段 ...
    /// Batch M 维度上界 (决定 activation sizing)
    /// None = 单序列 (当前行为)
    /// Some(golden_batch) = batch 模式, M 上界 = golden_batch
    pub batch_golden_size: Option<usize>,
}
```

不同 `batch_golden_size` 编译出不同 scratchpad 布局。复用 `SeqHistogram + GoldenBucket`
对 batch 维度做装筒，与 seq_len 维度同等处理。

### 6.2 Variant 选择

```
Executor.select_variant_for_batch(scheduled_batch):
  golden_batch = GoldenBucket.collapse_to_golden(num_seqs)
  key = VariantKey { batch_golden_size: Some(golden_batch), ... }
  registry.find_closest(&key)
```

batch_size=1 时 `batch_golden_size = None` → 走当前 Single variant。

---

## 7. 组件连线

### 7.1 ContinuousBatcher → BatchContext

```rust
impl Executor {
    fn build_batch_context(&self, batch: &ScheduledBatch) -> Vec<u8> {
        // 分配 flat memory: 固定头 80 bytes + num_seqs × 52
        let layout_size = 80 + batch.requests.len() * 64; // 52 padded to 64
        let mut ctx = vec![0u8; layout_size];

        // 填充头部
        write_u32(&mut ctx, 0, batch.requests.len() as u32);    // num_seqs
        write_u32(&mut ctx, 4, max_decode_steps);                // max_decode_steps
        write_u32(&mut ctx, 8, total_prefill_tokens);            // total_prefill_tokens

        // 填充指针 (指向 caller 分配的 buffer)
        write_ptr(&mut ctx, 16, input_ids_flat.as_ptr());
        write_ptr(&mut ctx, 24, output_tokens_flat.as_mut_ptr());
        write_ptr(&mut ctx, 32, positions.as_ptr());
        write_ptr(&mut ctx, 40, page_table_flat.as_ptr());
        write_ptr(&mut ctx, 48, kv_pool_base);
        write_ptr(&mut ctx, 56, sampling_params.as_ptr());
        write_ptr(&mut ctx, 64, hook_ctx_ptr);
        write_ptr(&mut ctx, 72, callback_table_ptr);

        // 填充 per-seq 元数据 (从 PagedScheduler / SessionKvCache / running seqs 读取)
        for (i, req_id) in batch.requests.iter().enumerate() {
            let base = 80 + i * 64;
            write_u32(&mut ctx, base + 0, prompt_lens[i]);
            write_u32(&mut ctx, base + 4, kv_lens[i]);
            write_u32(&mut ctx, base + 8, rope_pos_offsets[i]);
            write_u32(&mut ctx, base + 12, max_new_tokens[i]);
            write_u32(&mut ctx, base + 16, session_positions[i]);
            write_u32(&mut ctx, base + 20, page_table_offsets[i]);
            write_u32(&mut ctx, base + 24, page_table_lens[i]);
            write_u32(&mut ctx, base + 28, fused_hidden_offsets[i]);
            write_u32(&mut ctx, base + 32, num_mm_tokens[i]);
            write_u32(&mut ctx, base + 36, 1); // active_flag = 1
            write_u32(&mut ctx, base + 40, prompt_lens[i]); // seq_position = prompt_len
            write_u32(&mut ctx, base + 44, 0); // gen_count = 0
            write_u32(&mut ctx, base + 48, 0); // last_sampled_token = placeholder
        }

        ctx
    }
}
```

### 7.2 Executor batch 推理路径

```rust
impl Executor {
    pub fn generate_batch(&self, requests: &[GenerateRequest]) -> Vec<GenerateResult> {
        let (batch, ctx_mem) = self.build_batch_context(&scheduled_batch);

        let generated = unsafe {
            (mega.entry_fn)(
                input_ids_flat.as_ptr(),      // arg 0: flat input
                mega.weight_blob.as_ptr(),     // arg 1: 权重
                kv_pool_base as *mut u8,       // arg 2: KV pool
                positions.as_ptr(),            // arg 3: 位置
                std::ptr::null(),              // arg 4: aux
                batch.requests.len(),          // arg 5: batch_size → 现在用于 forward
                0,                             // arg 6: prompt_len → 忽略 (从 ctx 读取)
                scratchpad.as_mut_ptr(),       // arg 7: scratchpad
                output_tokens_flat.as_mut_ptr(), // arg 8: flat output
                0, 0, 0,                       // arg 9-11: sampling → 忽略 (从 ctx 读取)
                0,                             // arg 12: max_new_tokens → 忽略
                0,                             // arg 13: eos_token_id → 忽略
                0,                             // arg 14: output_mode
                std::ptr::null(),              // arg 15: hook_ctx → 忽略 (从 ctx 读取)
                std::ptr::null_mut(),          // arg 16: telemetry
                0,                             // arg 17: session_position → 忽略
                std::ptr::null(),              // arg 18: fused_hidden → 忽略 (从 ctx 读取)
                0,                             // arg 19: num_mm_tokens → 忽略
                std::ptr::null(),              // arg 20: callback_table → 忽略 (从 ctx 读取)
                std::ptr::null(),              // arg 21: page_table → 忽略 (从 ctx 读取)
                ctx_mem.as_ptr(),              // arg 22: batch_ctx_ptr → NEW
            )
        };

        // 从 ctx_mem 中读取 per-seq 结果 (gen_count, output tokens)
        self.collect_batch_results(&ctx_mem, output_tokens_flat)
    }
}
```

### 7.3 compile() 改造

```
改造点:
1. Phase 0: 加载 arg 22 → if NULL → goto Legacy (当前逻辑分支)
2. Phase 1: 从 batch_ctx 读取 num_seqs, total_prefill_tokens, max_decode_steps
3. Phase 2 (新增): Prefill forward, M = total_prefill_tokens
   - embed: GatherLoad 从 input_ids_flat, 按 prompt_lens 分段
   - Attention: per-token seq_id 查找 (§3)
   - per-seq argmax + store + update seq_meta
4. Phase 3 (改造): Decode step loop
   - M = num_active (从 active_flag 数组计算)
   - 构建 decode input: last_sampled_token × num_active
   - forward (M 维度)
   - per-seq argmax + store + check stop + update active_flag
5. 层循环融合组（decode 路径）: Return
```

---

## 8. REQ 清单

> **实现状态 (2026-05-25 审计)**: 全部 10 个 REQ 已实现。
>
> | REQ | 状态 | 实现位置 |
> |-----|------|---------|
> | BCI-001 | ✅ | `src/engine/batch_context.rs` |
> | BCI-002 | ✅ | `mega_kernel.rs` KernelContext offset 0x88 + Phase 0.5 分支 |
> | BCI-003 | ✅ | `gllm-kernels/.../mega_kernel_emit.rs` Phase 2 (Prefill) + Phase 3 (Decode) |
> | BCI-004 | ✅ | `VmInstr::BatchSeqIdLookup` + `seq_mapping_ptr` (BCI6) |
> | BCI-005 | ✅ | `VmInstr::PageTableAddr.seq_pt_offset: Option<VRegId>` |
> | BCI-006 | ✅ | 温度分支 + Per-seq argmax/softmax + `BatchPerSeqStopCheck` |
> | BCI-007 | ✅ | `src/scheduler/batcher.rs` `build_batch_with_prep()` → `BatchPrepData` |
> | BCI-008 | ✅ | `Client::generate_batch()` → `Executor::generate_batch()` → `MegaKernelExecutor::generate_batch()` |
> | BCI-009 | ✅ | `VariantKey.batch_golden_size` + `find_closest()` |
> | BCI-010 | ✅ | `MegaKernelBufferLayout` M 维度参数化 |

### REQ-BCI-001: BatchContext flat memory 布局 ✅
- 定义 `BATCH_CTX_HEADER_SIZE = 80`, `SEQ_META_STRIDE = 56` (14 × u32 fields)
- 定义 per-seq 字段 offset: `PROMPT_LEN=0, KV_LEN=4, ROPE_POS_OFFSET=8, ...`
- Executor 构建函数: `build_batch_context(ScheduledBatch) -> Vec<u8>`
- 从 PagedScheduler 批量收集页表
- 从 SessionKvCache 收集 session_position
- 从 running sequences 收集 kv_len, prompt_len

### REQ-BCI-002: ABI arg 22 batch_ctx_ptr + 向后兼容 ✅
- `MegaKernelFn` 新增 arg 22: `*const u8` (batch_ctx_ptr)
- `MEGA_KERNEL_PARAMS` 和 `MEGA_KERNEL_STACK_OFFSETS` 追加 → `[rbp+144]`
- JIT 代码: `batch_ctx_ptr == NULL` → Phase 1.Legacy (当前单序列逻辑，零行为变更)

### REQ-BCI-003: Mega-Kernel generate 循环 M 维度统一 ✅
- Phase 2 (Prefill): `M = total_prefill_tokens` → SymDim("prefill_M")
- Phase 3 (Decode): step loop, `M = num_active` → SymDim("decode_M")
- Phase 1.Legacy: batch_ctx_ptr == NULL 时走当前逻辑
- emit_fusion_groups 不变 — forward pass 对 M uniform

### REQ-BCI-004: Attention per-token seq_id 查找 ✅
- Prefill 阶段: cumsum(prompt_lens) 搜索确定 token 属于哪条序列
- Decode 阶段: active_seq_ids 直接映射
- seq_id 用于定位 page_table_offset[seq] → 该序列的 KV cache

### REQ-BCI-005: PageTableAddr seq_index 参数化 ✅
- `VmInstr::PageTableAddr.seq_index` 从 `None` 改为 `Some(VRegId)`
- JIT 地址计算: `page_table_flat[pt_offsets[seq] + logical_page]`
- KV 写入同套寻址逻辑

### REQ-BCI-006: Per-Sequence 采样与停止 JIT 实现 ✅
- Logits 布局 `[M, vocab_size]` — per-seq 行偏移
- 采样管线 (per-seq, 从 `sampling_params_ptr + seq × 16` 读取参数):
  - temperature == 0: Argmax (greedy)
  - temperature > 0: TemperatureScale → Softmax (ReduceMax + ExpSum + Normalize) → TopK → TopP → Multinomial
- 采样结果写入两处:
  - `seq_meta[seq].last_sampled_token` (offset +48) — 下一步 decode 的输入
  - `output_tokens_flat[output_offset + gen_count]` — 最终输出 (output_offset 从 seq_meta +52 读取)
- CheckStopCondition: eos match or gen_count >= max_new_tokens
- active_flag 更新: 序列完成后从活跃集合移除

### REQ-BCI-007: ContinuousBatcher 产出扩展 ✅
- `build_batch()` 返回 `(ScheduledBatch, BatchPrepData)` — prep data 包含所有构建 BatchContext 所需的中间数据
- BatchPrepData: prompt_lens, kv_lens, session_positions, sampling_params, page_tables
- 复用 SeqHistogram + GoldenBucket 进行形状分组

### REQ-BCI-008: Executor batch 推理路径 ✅
- `generate_batch()` 公共 API (Client 层多请求并发)
- Scratchpad 分配: M 维度参数化，max_M × hidden
- RoPE cache: 按最大序列的 max_total 预填充
- Output tokens: flat 布局，per-seq 按 prompt_len + max_new_tokens 分段
- 从 ctx_mem 收集 per-seq 结果

### REQ-BCI-009: VariantKey batch 维度扩展 ✅
- `VariantKey` 新增 `batch_golden_size: Option<usize>`
- batch_size=1 → None (当前 Single variant)
- batch_size=N → Some(golden_bucket) (Batch variant)
- GoldenBucket 装筒与 seq_len 维度同等处理
- 复用 `VariantRegistry.find_closest()` constraint relaxation

### REQ-BCI-010: MegaKernelBufferLayout M 维度参数化 ✅
- `from_graph_geometry_batched(geo, max_M, max_seq_len)` — 新增 max_M 参数
- Activation sizing: `max_M × hidden` (替代 `max_seq_len × hidden`)
- Logits sizing: `max_M × vocab_size` (替代 `max_seq_len × vocab_size`)
- RoPE cache: max_seq_len × head_dim × 2 (不变，按最长序列)

---

## 9. 实施依赖图

```
REQ-BCI-002 (ABI arg 22)
    │
REQ-BCI-001 (BatchContext flat layout)
    │
REQ-BCI-010 (BufferLayout M 维度参数化)
    │
REQ-BCI-003 (generate 循环 M 维度统一)
    ├── REQ-BCI-004 (Attention seq_id 查找)
    ├── REQ-BCI-005 (PageTableAddr seq_index)
    └── REQ-BCI-006 (Per-Seq 采样停止)
         │
REQ-BCI-007 (Batcher 产出扩展)
REQ-BCI-009 (Variant batch 维度)
         │
REQ-BCI-008 (Executor batch 连线)
```

## 10. 与现有 SPEC 的交叉引用

| 本文档 REQ | 影响的已有组件 | 改造性质 |
|------------|-------------|---------|
| BCI-001 | Executor | 新增 `build_batch_context()` |
| BCI-002 | `MegaKernelFn`, `MEGA_KERNEL_PARAMS` | +1 ABI 参数 |
| BCI-003 | `compile()` Phase 0-3 + 融合组 | Phase 1.Legacy + Phase 2/3 新增 |
| BCI-004 | `compile()` Attention | 新增 seq_id 查找逻辑 |
| BCI-005 | `VmInstr::PageTableAddr` | seq_index 参数化 |
| BCI-006 | Argmax/StoreToken/CheckStopCondition 融合组 | per-seq 偏移 |
| BCI-007 | `ContinuousBatcher` | 返回类型扩展 |
| BCI-008 | `Executor`, `Client` | 新增 `generate_batch()` API |
| BCI-009 | `VariantKey`, `VariantRegistry` | 新增 batch_golden_size |
| BCI-010 | `MegaKernelBufferLayout` | max_M 参数化 |
