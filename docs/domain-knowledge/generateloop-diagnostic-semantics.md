# GenerateLoop 诊断语义资料库（C-9，堵 AI 幻觉）

> 来源：gllm mega_kernel_emit + executor_ops + topology.rs 源码 + Step7 实测（确定性）
> 建库触发：8 轮 CPU BUG 诊断反复被诊断工具语义错位误导（第6轮架构师纠错 + Step7 证实）
> 最后验证：2026-07-05

## 核心陷阱：GenerateLoop M=1 让诊断测试读多行错位

### GenerateLoop 每迭代只处理 1 个 token
- decoder 模型（SmolLM2/Llama/Qwen）→ `is_generate=true` → `LoopTopology::GenerateLoop`
- `seq_len_source = LoopCounterPlusOne`：每迭代 seq_len = gen_counter+1
- Gather/GEMM/lm_head 每迭代 M=1（只 1 token）
- **输出写 row 0（覆盖），不按 [seq_len, ...] 累加写**

### 诊断 harness 读多行 = 错位（第6轮架构师纠错）
| 错位类型 | harness 错法 | 伪信号 |
|---------|------------|--------|
| embedding 读多行 | `sp.read_dtype_aware(emb_off, seq*hidden)` 读 row0-4 | row1-4=0，cosine 低 |
| logits 读多行 | `read_dtype_aware(logits_off, seq*vocab)` | row1-4=0，cosine 低 |
| encode_at_layer 读多行 | `read_dtype_aware(0, seq*hidden)` | 同上 |
| 逐层 hidden 读多行 | encode_to_layer(N) 返回层输出，读多行 | 全错位 |

### 实测确证（Step7，commit 829ae2f6）
单 token prefill（方法 A）重建 embedding：
```
token 0-4 各自 cosine vs golden hidden_layer_0 row i = 1.0000
```
**embedding 完全正确**，之前 0.13/0.67 是 harness 读多行错位伪信号。

## 正确诊断方法（语义对齐）

### 方法 A: 单 token prefill 重建（embedding/中间层）
对每位置 i：
1. 用前缀 `&tokens[0..=i]` 调 `diagnostic_prefill_scratchpad`
2. 读 row 0（最后 token = token i 的输出）
3. 拼 [seq, hidden] vs golden hidden_layer_i
- **适用**：layer 0 ops（embedding/gather/q_proj）不依赖 KV cache，单 token 可重建
- **限制**：attention 跨 token 依赖 KV cache，深层需逐 token 累积

### 方法 B: 单次调用只比 row0 vs golden 最后 token
- 读 gllm row0（最后 prompt token 的输出）
- 比 golden hidden_layer_{N} 的最后行（row seq_len-1）
- 单 token 验证（弱，但语义对齐）

### logits 读取（ARCH-DECODE-LOGITS-ROW0，BCE-20260629-002）
- decode 内核把 last-token logits 写 **row 0**（非 row[prompt_len-1]）
- `diagnostic_prefill_logits` 读 row 0 → **正确**（这个信号可信）
- 读 row[prompt_len-1] = 读未写零内存 → 错

## encode_at_layer 的 session_position 复用（executor_ops.inc.rs:762）
```rust
// session_position 参数复用为 anchor_layer
session_position: anchor_layer,  // anchor layer N
```
- anchor_layer 通过 session_position 传给 mega-kernel
- 内核在第 N 层 early-exit，写该层 hidden 到 scratchpad
- **但仍是 GenerateLoop M=1**，只 row0 有效
- `read_dtype_aware(0, seq*hidden)` 读多行 → 错位

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码 + 实测证明） |
|--------|---------|
| prefill 一次写 [seq, hidden] | GenerateLoop 每迭代 M=1 覆盖 row0 |
| 诊断读 row0-4 比对 golden 多行 | 错位，row1-4=0 是设计 |
| encode_at_layer 返回完整 [seq, hidden] | 返回 M=1 输出，只 row0 有效 |
| embedding cosine 低 = embedding bug | Step7 证 embedding 100% 对，是 harness 错位 |
| logits 在 row[prompt_len-1] | ARCH-DECODE-LOGITS-ROW0：写 row0 |
| 单 token prefill 不能重建（缺 KV） | layer 0 ops 无 KV 依赖，可重建 |
| **encode_to_layer(LastToken) 返回 layer N 输出 row0** | **实测返回全零（Step8）— API 不可用于逐层 bisection** |
| **encode_to_layer(MeanPool) cosine 低 = 层错** | **错位伪信号：gllm mean=row0/5 vs golden mean=真实5token平均** |

## 与 BUG 诊断的关系
- ✅ embedding 对（Step7 方法 A）
- ✅ dtype 对（D0）
- ✅ 残差流映射对（H4）
- ✅ row1-4 零是设计（topology 库）
- ✅ dtype 链自洽（architect sessionId 401396fe 裁决，非发散根因）
- 🔄 **logits cosine=-0.465 是真信号**（diagnostic_prefill_logits 读 row0 正确）
- 根因在 logits 计算路径（final norm / lm_head / 某 layer GEMM），需用语义对齐方法诊断
- ❌ **encode_to_layer(LastToken) 返回全零**（Step8 实测，30 层全 cosine=0.0000）— API 不可用，需新诊断路径
- ❌ **encode_to_layer(MeanPool) cosine≈0**（Step5，错位伪信号：gllm mean=row0/5 vs golden mean=5token真实平均）

## 逐层 bisection 的困境（Step8 实测 + architect sessionId b2aff8e2 确认）

诊断 logits 发散需逐层定位首个发散算子，但现有诊断 API 均不可用：
- `encode_to_layer(LastToken)`: 返回全零（Step8 30 屄全 cosine=0.0000）
- `encode_to_layer(MeanPool)`: 错位伪信号（Step5 cosine≈-0.01，gllm mean=row0/5 vs golden mean=5token真实平均）
- `diagnostic_prefill_scratchpad`: 能读 embedding offset（Step7 证对），但 layer hidden 输出在 ActivationPing/Pong 区，不在 named_offsets，无法直读

**物理根因（architect 确认）**：层循环是**单模板×30次迭代**（NO-LAYER-EXPAND 铁律）。所有 30 层写同一对 ping/pong buffer，`ActivationSwap`（pipeline.inc.rs:384）只换指针。循环跑完，**只有 layer 29 的输出存活**，中间层物理上读不到。

**EarlyExit 双重未完成**：
1. EarlyExit op 未插入图（grep 全代码库无插入点，只定义未使用）
2. GprBranchAction::Exit x86 lowering 未实现（lower_instr_dispatch.inc.rs:2901-2903 CmpEq+Exit 返 Err）
3. ConditionalExit 已实现（:2995）但 EarlyExit op 用错了 GprCondAction::Exit

**结论**：逐层 bisection 需新机制。architect 推荐 Ring-Buffer 单遍捕获：
- 层循环末尾（ActivationSwap 前）插无条件 side-channel 拷贝
- 按 layer_loop_counter 缩放偏移到 capture 区
- 单次 forward 捕获全部 30 层
- 避开 CmpEq+Exit 未实现（用无条件拷贝 + counter 缩放，复用 AddPtr + emit_side_channel_copy）
- 诊断开关门控（ComputeProfile 字段，禁环境变量），生产零开销

**dtype 假说已排除**（architect 重提但已被运行时证伪）：
- KV cache 全 F32 自洽（运行时插桩 kv_row_stride=768=buffer=MemCopy，非越界）
- derive_compute_dtype 硬编码是宪法 -1 违宪（层2），但当前对 SmolLM2 数值自洽（层1），非发散根因

## Ring-Buffer 实现 + capture 全零卡点（2026-07-06）

Ring-Buffer 基础设施 + 暴露层全落地（gllm-kernels commit 0a710e86/59ad1e6a, gllm commit 37d9b312）：
- BufferAllocation/BufferLayout 加 layer_capture_offset/stride/bytes 字段
- pipeline.inc.rs close_layer_loop + handle_standard_layer_loop 在 ActivationSwap 前 emit capture copy
- named_offsets 注册 "layer_capture" + diagnostic_layer_capture_stride() getter
- diagnostic-layer-capture Cargo feature 门控（默认关，生产零开销）

**实测（diag_step10 启用 feature）**：
```
[RING-BUF] layer_capture registered: offset=245366784 stride=18874368 bytes=566231040
[RING-BUF] capture_off=245366784 stride=18874368 (feature 启用)
layer 0-29 row_last cosine = 0.0000 (nonzero=0)  ← capture 区全零!
```

**capture 区全零** = emit capture copy 没执行或写错位置。可能原因：
1. close_layer_loop 的 emit 条件 `locals.layer_capture` + `state.abi.layer_loop_counter` + `activation_swap_vregs` 某个为 None
2. SmolLM2 走的层循环路径没进 close_layer_loop（走 handle_standard_layer_loop 的 `!group.is_layer_group` 分支，但 is_layer_group=true 不进）
3. pong 指针在 emit 时指向错误（ActivationSwap 前 pong 不是当前层输出）
4. layer_loop_counter 在 emit 时值不对

**待调试**：加 eprintln 在 emit capture copy 处确认是否执行 + 各变量值。Ring-Buffer 基础设施正确，诊断测试待调试 capture 写入。

## Ring-Buffer capture emit 位置错（2026-07-06 调试确认）

加 eprintln 调试发现：
```
[CAPTURE-DBG] construct: alloc.layer_capture_bytes=566231040 stride=18874368 (构造成功)
[CAPTURE-DBG] NOT in_layer_loop (state.in_layer_loop=false) — close_layer_loop skipped
```

**根因**：capture emit 在 close_layer_loop（pipeline.inc.rs:440），但 SmolLM2 编译时 `state.in_layer_loop=false`——**没进层循环路径**，close_layer_loop 从不被调，capture copy 从不 emit。

handle_standard_layer_loop 的 emit 点（line 878）也要求 `state.in_layer_loop`，同样不进。

**问题**：SmolLM2 层循环 codegen 不走 emit_fusion_groups 的 in_layer_loop 路径。可能走 GroupMarker 或别的机制（SPEC 39 统一编译器，层循环由 GroupMarker 驱动）。

**需 architect 确认**：SmolLM2 层循环实际走哪条 codegen 路径？capture emit 应放在哪？

候选：
- GroupMarker 驱动的层循环（mega_kernel_emit.rs 或别处）
- handle_standard_layer_loop 但 in_layer_loop 状态没正确设置
- 层循环根本不在 emit_fusion_groups，而在 mega_kernel_emit 的统一 emit

Ring-Buffer 基础设施（字段/暴露/getter）全对，只差 emit 位置匹配 SmolLM2 实际层循环路径。

## is_layer_group 全 false 铁证（2026-07-06）

grep "is_layer_group" 全代码库：所有构造点（group_dep.rs:153/334/495/796/1088/...共 15+ 处）都是 `is_layer_group: false`。**没有任何地方设 is_layer_group: true**。

所以 pipeline.inc.rs:812 `if group.is_layer_group && !state.in_layer_loop` 永远不进 → in_layer_loop 恒 false → close_layer_loop 不调 → Ring-Buffer capture 不 emit。

**矛盾**：is_layer_group 全 false，但 SmolLM2 30 层能跑。说明层循环不在 emit_fusion_groups 的 in_layer_loop 路径。

**architect 分析中（sessionId ff2b4f63）**：
- SmolLM2 层 op label 是 "layer." 前缀（build_graph 用 ptname("layer.xxx")）
- assign_group_markers 的 fallback（label starts_with "layer."）理论上该标 is_layer_group=true
- 但实际全 false → fallback 没命中（group.ops[0] label 不是 "layer." 或没跑到这 plan）
- 正在查 fusion/pass.rs assign_group_markers + group.ops[0] 语义

**待确认**：SmolLM2 30 层循环实际 emit 位置（mega_kernel_emit.rs GenerateLoop 内？单模板只 emit 一层 + 运行时 LoopBegin/End 循环 30 次？）+ Ring-Buffer capture 正确位置。

## 正确的逐层 bisection（用方法 A 累积）
1. layer 0：单 token prefill（prefix=[tok0]）读 row0 vs golden hidden_layer_1 row0
2. layer N：prefix=tokens[0..N+1]，读 row0 vs golden hidden_layer_{N+1} row(seq_len-1)
3. 注意：attention 有 KV cache 累积，单 token prefill 的 prefix 含历史 token
4. 找首个 cosine<0.99 的层

## 关键代码位置
- `executor_ops.inc.rs:805-810`: encode_at_layer 读 `read_dtype_aware(0, seq*hidden)`（多行陷阱）
- `executor_ops.inc.rs:121-134`: ARCH-DECODE-LOGITS-ROW0 注释
- `executor_ops.inc.rs:762`: session_position 复用 anchor_layer
- `tests/test_diag_cpu_bisect.rs diag_step7`: 方法 A 实现（实测 cosine 1.0000）

## 与其他资料库关系
- `mega-kernel-topology.md`: GenerateLoop M=1 根因（本库是其诊断应用）
- `vam-activation-pingpong.md`: layer loop ActivationPing/Pong
- 本文件: 诊断 harness 怎么避开 M=1 错位陷阱
