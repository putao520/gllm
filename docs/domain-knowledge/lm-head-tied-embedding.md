# lm_head Tied Embedding 权重解析资料库（C-9，堵 AI 幻觉）

> 来源：gllm name_map.rs + executor_compile.rs + build_graph.inc.rs + pack_observe.inc.rs + graph_impl.inc.rs（Explore 调研，源码事实）
> 建库触发：8 轮 SmolLM2 logits 发散诊断，lm_head（tied embedding）反复被怀疑，需确定性记录其解析链路正确性
> 最后验证：2026-07-05

## 核心机制（源码确认）

### tie 检测是 data-driven，不是 flag-driven

**name_map.rs:201-211**：
```rust
// Tied embeddings: when lm_head canonical doesn't exist (no separate lm_head
// tensor), map lm_head → embed's external name.
// This applies regardless of tie_word_embeddings config — if there's no separate
// lm_head tensor in the weight files, the model physically uses tied embeddings.
if !canonical_to_external.contains_key("lm_head") {
    if let Some(embed_ext) = canonical_to_external.get("embed").cloned() {
        canonical_to_external.insert("lm_head".to_string(), embed_ext.clone());
    }
}
```

**关键**：`tie_word_embeddings` config flag 仅作 hint 存储（loader_impl.inc.rs:612 `set_tie_word_embeddings_hint`），**实际别名行为完全由"权重文件中是否存在独立 lm_head tensor"决定**。

SmolLM2-135M-Instruct 的 safetensors 没有 `lm_head.weight`（tied），所以 `canonical_to_external` 无 `"lm_head"` 键 → 触发别名到 `model.embed_tokens.weight`。

### 权重名映射：图层面独立 TensorId，物理层面共享指针

**机制 (c)+(a) 组合，非 (b) 复制**：

1. **名称别名**（name_map.rs:206-211）：`canonical_to_external["lm_head"] = "model.embed_tokens.weight"`
2. **指针别名**（executor_compile.rs:266-273 `convert_ext_to_canonical`）：
   ```rust
   for (ext_name, &ptr) in ext_ptrs {
       for cn in name_map.all_canonical_for(ext_name) {  // 返回 ["embed", "lm_head"]
           weight_ptrs.entry(cn).or_insert(ptr);  // lm_head 和 embed 共享同一 ptr
       }
   }
   ```
3. **图层面独立 TensorId**（build_graph.inc.rs:2301）：
   ```rust
   let lm_head_w = g.add_tensor_concrete("lm_head", &[vocab_size, hidden], tdt("lm_head"));
   ```
   `add_tensor` 每次分配新 TensorId（graph_impl.inc.rs:33-36），所以 `embed_w`（行191）和 `lm_head_w`（行2301）是**不同 TensorId**，在 `g.inputs` 里是两个独立条目。
4. **dtype 独立查找但同值**（build_graph.inc.rs:86）：
   ```rust
   let tdt = |name| weight_dtypes.get(name).copied().unwrap_or(F32);
   ```
   `weight_dtypes["lm_head"]` 和 `weight_dtypes["embed"]` 通过 `all_canonical_for` 都映射到 BF16。

### weight_blob 布局：lm_head 是单独一份拷贝（双倍存储）

**graph_impl.inc.rs:233-250 `weight_layout()`** 按 `self.inputs` 顺序分配 offset。因 `lm_head_w` 和 `embed_w` 是不同 TensorId，**各自获得独立 offset**。

**pack_observe.inc.rs:204-339 `pack_weights_from_graph`**：
- 行205：`ext_name = name_map.resolve_external_to_string("lm_head")` → `"model.embed_tokens.weight"`（别名）
- 行206：`raw_floats.get(&ext_name)` → 找到同一份 raw 数据
- 行308-339：把 `raw.data` 复制到 `blob[blob_off..blob_off+copy_size]`

**结果**：
- `embed` 条目把 `model.embed_tokens.weight` 字节复制到 embed offset
- `lm_head` 条目**再次把同一份字节复制到 lm_head offset**
- blob 里 embed_tokens 权重存储两份（双倍内存，SmolLM2-135M 约 56MB 额外）
- **正确性无害**：lm_head GEMM 读 lm_head offset，内容是正确的 embed_tokens 权重字节

### graph 构建：lm_head = Op::Gemm

**build_graph.inc.rs:2299-2307**：
```rust
let lm_head_w = g.add_tensor_concrete("lm_head", &[vocab_size, hidden], tdt("lm_head"));  // [49152, 576]
let mut logits = g.add_tensor("logits", vec![s.clone(), SymDim::Concrete(vocab_size)], act_dt);
add_gemm_or_quant(&mut g, "lm_head", s.clone(), vocab_size, hidden,
    vec![final_normed, lm_head_w], vec![logits], "lm_head");
```

**add_gemm_or_quant（build_graph.inc.rs:99-114）**：
- op 类型 = `Op::Gemm`（SmolLM2 lm_head 未量化）
- weight 输入 tensor 名 = `"lm_head"`（非 `"embed"`）
- `GemmSpec { dtype: tdt("lm_head")=BF16, trans_b: true, has_bias: false }`

### GEMM 维度（M=seq_len, N=vocab, K=hidden, 权重 [vocab, hidden]）

| 维度 | 值 | 来源 |
|------|-----|------|
| M | seq_len（decode M=1） | `s.clone()` SymDim |
| N | vocab=49152 | build_graph.inc.rs:122 `vocab_size = embed_shape[0]` |
| K | hidden=576 | build_graph.inc.rs:123 `hidden = embed_shape[1]` |
| 权重 shape | [vocab, hidden] = [49152, 576] | build_graph.inc.rs:2301 `[vocab_size, hidden]` |
| trans_b | true | build_graph.inc.rs:110 硬编码 |

**trans_b=true 语义**（gemm_emit.rs:1302-1305）：B 是 [N,K] 行主序，GEMM 计算 `C[M,N] = A[M,K] × B[N,K]^T`。与 PyTorch `nn.Embedding.weight` 的 `[vocab, hidden]` 行主序一致，作为 lm_head 时数学上 `logits = hidden @ embed.weight.T`，**无需 transpose**，trans_b=true 正确。

### logits 写入（ARCH-DECODE-LOGITS-ROW0）

**mega_kernel_emit.rs:1136-1147**：`logits_scratch_offset` 动态计算（RoPE cache 之后，64 字节对齐）。

**mega_kernel_emit.rs:2247-2252**：decode 时 logits 写到 `scratchpad[logits_scratch_offset + compact_row * vocab_bytes]`，M=1 时 compact_row=0 → **row 0**，大小 vocab_size=49152 元素（F32，196608 bytes）。

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| tie_word_embeddings flag 驱动别名 | data-driven（无独立 lm_head tensor 即别名），flag 仅 hint |
| lm_head 复用 embed 的同一 offset | 独立 offset（不同 TensorId），但内容是同一份字节的拷贝 |
| lm_head weight dtype 是 F32 | BF16（通过 all_canonical_for 与 embed 同值） |
| 权重布局 [hidden, vocab] | [vocab, hidden] = [49152, 576] 行主序 |
| trans_b=true 是错误 transpose | 正确对应 [N,K] 行主序，无需 transpose |
| lm_head 独立权重 | tied（lm_head.weight == embed_tokens.weight，物理共享） |
| tie 时 blob 共享同一份字节 | 各自独立 offset 各存一份拷贝（双倍内存，正确性无害） |

## 排除结论（7 项全查，未发现 bug）

| 检查项 | 结论 |
|--------|------|
| tie 检测 | data-driven，正确 |
| 权重名映射 | 机制 (c)+(a)，图层面独立 TensorId + 物理共享指针，正确 |
| weight_blob 布局 | 单独拷贝，内容正确复制自 embed_tokens.weight，正确性 OK（内存浪费） |
| graph 构建 | Op::Gemm，weight tensor 名="lm_head"，正确 |
| GEMM 维度 | M=seq, N=vocab=49152, K=hidden=576, 权重 [vocab,hidden]，正确 |
| trans_b | true 对应 [N,K] 行主序，无错误 transpose，正确 |
| logits 写入 | scratchpad row 0（decode M=1），大小 49152，正确 |

**lm_head tied embedding 链路正确性完整**——dtype BF16 对、权重布局 [vocab, hidden] 对、trans_b=true 对、offset 独立但内容对。

**非 SmolLM2 logits 发散根因**。唯一问题是内存浪费（embed_tokens 权重存两份）。真因见 `kv-cache-dtype-dual-layer.md`（KV cache dtype 双地层裂开）。

## 关键代码位置

- `gllm/src/loader/name_map.rs:201-211, 257-272` — tie 检测 + all_canonical_for
- `gllm/src/engine/executor_compile.rs:266-273, 545-554` — 指针别名 + weight_dtypes 构建
- `gllm/src/arch/auto_graph_fragments/build_graph.inc.rs:86, 121-123, 2299-2307` — tdt + 维度推导 + lm_head graph 构建
- `gllm-kernels/src/compiler/graph_impl.inc.rs:33-36, 233-250` — add_tensor 新 TensorId + weight_layout
- `gllm/src/engine/mega_kernel/pack_observe.inc.rs:204-339` — pack_weights_from_graph（双倍拷贝）
- `gllm-kernels/src/compiler/codegen/vm/mega_kernel_emit.rs:1136-1147, 2247-2252` — logits 写入 row0
- `gllm/src/weight_names.rs` — decoder_final_norm_aliases（无 lm_head 别名，tied 由 name_map 处理）

## 与其他资料库关系

- `smollm2-135m-architecture.md`：SmolLM2 tied embedding + BF16 事实（本库是其 lm_head 解析应用）
- `kv-cache-dtype-dual-layer.md`：SmolLM2 logits 发散真因（本库排除 lm_head 嫌疑）
- `mega-kernel-topology.md`：ARCH-DECODE-LOGITS-ROW0（本库引用 logits 写入位置）
- 本文件：lm_head tied embedding 权重解析链路正确性证据
