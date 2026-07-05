# Mega-Kernel Topology (prefill vs decode, M 维) 资料库（C-9）

> 来源：gllm-kernels topology.rs + mega_kernel_emit.rs 源码（确定性，非猜）
> 建库触发：8 轮 CPU BUG 诊断核心疑点——SmolLM2 prefill 走 decode 还是 prefill 内核；row1-4 全零是设计还是 bug
> 最后验证：2026-07-05

## 核心机制（topology.rs:202-213，确定性）

### LoopTopology 判定
```rust
let is_generate = has_argmax && has_store_token && has_check_stop;
let loop_topology = if is_generate { LoopTopology::GenerateLoop } else { LoopTopology::SinglePass };
let outer_loop_bound = if is_generate {
    TopologyBound::DynamicTotalIters   // (prompt_len-1) + max_new_tokens
} else {
    TopologyBound::Const(1)
};
let seq_len_source = if is_generate {
    SeqLenSource::LoopCounterPlusOne   // 每迭代 seq_len = gen_counter + 1 (递增)
} else {
    SeqLenSource::PromptLen            // seq_len = prompt_len (一次)
};
```

### 模型 → Topology 映射
| 模型类型 | Op 含 Argmax/StoreToken/CheckStop | LoopTopology | 例子 |
|---------|------|------|------|
| decoder/generator (LlamaForCausalLM) | ✅ | **GenerateLoop** | SmolLM2, Llama, Qwen |
| encoder/embedding/reranker | ❌ | **SinglePass** | BERT, E5, BGE |

### SmolLM2 → GenerateLoop（确认）
SmolLM2-135M-Instruct = `LlamaForCausalLM`（decoder），有 Argmax+StoreToken+CheckStop → **is_generate=true → GenerateLoop**。

**关键**：即使 `diagnostic_prefill_logits`（max_new_tokens=0/1，只 prefill 求 logits）也走 **GenerateLoop** 内核（拓扑在编译时决定，不由运行时 max_new_tokens 切换）。

## GenerateLoop 的 M 维语义（ARCH-DECODE-LOGITS-ROW0，BCE-20260629-002）

### 每迭代只 embed 1 个 token
- `seq_len_source = LoopCounterPlusOne`：iter N 时 seq_len = N+1
- Gather 每迭代只处理 **1 个 token**（input_ids[gen_counter]），M=1
- 所有 per-token GEMM 的 M=1（q_proj/k_proj/v_proj/lm_head 等）

### logits 写 row 0（ARCH-DECODE-LOGITS-ROW0）
- decode 内核不按行累加写 [seq_len, vocab]，而是**每代覆盖 row 0**
- lm_head 把 last-token logits 写入 scratchpad logits 区 offset 0（不是 row[prompt_len-1]）
- `executor_ops.inc.rs:121-134` 注释确证

### prefill 区的 embedding 写入
- GenerateLoop 前 prompt_len-1 次迭代是 prefill（embed+layers，跳过 sampling）
- 每迭代 Gather 写 1 个 token 的 embedding，**覆盖同一 row 0**（因 M=1，output_ptr 每迭代重置到 base）

## 关键：row1-4 全零是设计行为，非 bug

### 实测（commit 790e6883）
```
gllm embedding per-row cosine vs golden hidden_layer_0:
- row 0: 0.1307 (有数据, 部分对)
- row 1-4: 0.0000 (全零)
```

### 解释（GenerateLoop M=1）
- GenerateLoop 每迭代 Gather 写 1 token 到 row 0（M=1 覆盖）
- 迭代完成后，row 0 = 最后一个 token 的 embedding
- row 1-4 从未被写（scratchpad 零初始化，executor_ops.inc.rs:166 `vec![0u8; ...]`）
- **这是设计行为**，不是 Gather bug

### 诊断测试读 row1-4 是错的
- golden hidden_layer_0 [5, 576]：PyTorch prefill 一次过 5 token
- gllm embedding：GenerateLoop 每迭代 1 token 覆盖 row0，最终 row0 = 最后 token
- 两者维度语义不同，读 row1-4 比对是**测试 harness 错**（架构师第6轮：诊断工具语义错位）

### row0 部分对 (0.13) 的含义
- row0 = 最后一个 prompt token 的 embedding（不是第一个）
- golden hidden_layer_0 row0 = 第一个 token 的 embedding
- 不同 token 的 embedding → cosine 低是正常（不同 token 查不同权重行）
- **row0 部分对可能是 token 错位 1（last vs first），或诊断读错行**

## 正确的 embedding 验证方法（绕开 GenerateLoop M=1）

### 方法 A: 单 token prefill（DIAG-015 模式）
- 对每个位置 i，用前缀 `&INPUT_IDS[0..i+1]` 调 diagnostic_prefill_scratchpad
- 读 row 0（最后 token）= token i 的 embedding
- 拼接 [seq, hidden] vs golden hidden_layer_0
- layer 0 ops（embedding/gather/q_proj）不依赖 KV cache（无跨 token attention），单 token 可重建

### 方法 B: 单次调用只比 row0 vs golden last token
- 读 gllm row0（最后 prompt token = "is"）的 embedding
- 比 golden hidden_layer_0 row4（最后 token = "is"）
- 单 token 验证（弱）

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| prefill 走 SinglePass（M=seq_len） | decoder 模型 prefill 也走 GenerateLoop（M=1 循环）|
| Gather 写 [seq_len, hidden] | GenerateLoop 每迭代 1 token，覆盖 row0 |
| row1-4 全零 = Gather bug | 设计行为（M=1 + scratchpad 零初始化）|
| logits 在 row[prompt_len-1] | ARCH-DECODE-LOGITS-ROW0：写 row 0 |
| GenerateLoop 由 max_new_tokens 切换 | 编译时拓扑决定，运行时不切换 |
| embedding 单次调用读多行 | M=1，单次只 row0 有数据 |

## 与 BUG 诊断的关系
- **块1 logits cosine=-0.465**：GenerateLoop M=1 正确，logits 在 row0，diagnostic_prefill_logits 读 row0 正确——这个信号可信，logits 确实错
- **embedding cosine=0.13/0.67**：诊断测试读多行错位，是 harness bug，非 embedding bug
- **逐层 bisection 全≈0**：encode_to_layer(N) 返回 layer N 输出，但内部也走 GenerateLoop（M=1），输出语义需重审
- 真根因仍在 logits 计算路径（lm_head/final norm/某层 GEMM），需用语义对齐的诊断（方法 A）

## 关键代码位置
- `topology.rs:38-45, 195-213`: LoopTopology 判定 + is_generate
- `topology.rs:97-101`: outer_loop_bound + seq_len_source 字段
- `mega_kernel_emit.rs:1158-1200`: GenerateLoop emit（统一 prefill+generate 循环）
- `mega_kernel_emit.rs:1275-1300`: seq_len_source 推导 + mega_decode_seq_len
- `executor_ops.inc.rs:121-134`: ARCH-DECODE-LOGITS-ROW0 注释
- `lower_op.inc.rs:785-789`: Gather seq_bound = Const(1)（当 mega_decode_seq_len=Some）

## 与其他资料库关系
- `smollm2-135m-architecture.md`: SmolLM2 是 LlamaForCausalLM（decoder → GenerateLoop）
- `vam-activation-pingpong.md`: layer loop ActivationPing/Pong（GenerateLoop 内）
- 本文件: GenerateLoop M=1 语义 + ARCH-DECODE-LOGITS-ROW0
