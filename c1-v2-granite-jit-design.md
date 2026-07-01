# c1 v2_granite — JIT 图构建方案（架构提案，pre-SPEC）

> 状态：设计提案，**未落 SPEC**。落地前必须先走 S1 设计节点写 PRD + SPEC（见 §0 门控）。
> 所有结论基于 gllm / gllm-kernels 源码实证，非猜测。引用见文末证据表。

## 0. 前置门控（阻断项，先于任何编码）

1. **SPEC-first**：`granite` / `ModernBERT` / `c1` / 递归 tracker 在 SPEC 中零覆盖（现有 `INTENT-TRACKER.html` REQ-SIT-001..009 是 3-task/4-diff 门控聚合，**非递归、非同架构**）。c1 是全新 arch → 必须先写 PRD-REQ + SPEC（新建 `C1-GRANITE-TRACKER.html`）。禁止在 SPEC 未定义领域直接编码（C-1）。
2. **需要 c1 参考前向代码**：JIT 管线要求每个算子有 scalar 参考实现作 ground truth（ARCH-RUST-IS-CODEGEN / JIT 管线铁律）。best.pt 只给了 47 个权重名，**没给 cell 递推公式的精确接线**（α/i/c 三门如何组合成 h_t）。没有 PyTorch reference forward → 无法定义数值对齐基线 → 无法建图。**这是硬前置。**

## 1. 一个纠正你问题前提的关键实证

- **ABI 固定，不可扩展**：`KernelContext` = 22 参数 / 184 字节 repr(C)，偏移在编译期 bake 进机器码。新增顶层 ABI 参数 = 全量重编译 + 改所有偏移常量。
- **BatchContext 才是外部 I/O 扩展层**：token_ids / positions / output_tokens **都不是图 tensor**，是运行时经 `batch_ctx_ptr`(offset 0x88) 按固定偏移传入的外部 flat buffer。图本身**没有通用的"新增输入/输出 buffer"声明机制**。
- 推论：**跨 turn 状态不该走"新 ABI 参数"，应走 BatchContext 扩展字段**（与现有外部 I/O 同机制，零 ABI 重排）。这直接影响 Q2 的推荐。

---

## Q1. gated recurrent cell 如何在 graph IR 表达？

**推荐：选项 B（现有 op 图组合），不新增 Op。**

cell 权重分解 100% 落在现有算子：
- `W_int_x·x`, `W_int_h·h`, `W_α·x`, `U_α·h`, `W_i·x`, `U_i·h`, `W_c·x`, `U_c·h` → 全是 `Gemm`/`GemmBias`
- `α = sigmoid(...)`, `i = sigmoid(...)` → `Sigmoid`（已存在）
- `c = tanh(...)` → `Tanh`（已存在）
- 门控合成 `h_t = α⊙h + (1-α)⊙c` 类 → `Add`/`Mul`/`Sub`（BinaryElementwise，已存在）

**理由**：
- ARCH-UNIFIED-COMPILER：编译器=喂什么编译什么，不假设图结构。新 `GatedCell` Op = 编译器必须特判的结构算子 = 违反统一编译器。
- ARCH-AUTO-INSTR-SELECT 规则4：禁止创建 per-OpKind 手写函数。gated cell 分解为 Gemm + BinaryElementwise + Elementwise 三种**已有** ComputePattern，**无新 ComputePattern** → 不满足"TraceOp 语义扩展"的正当性门槛。
- 融合归融合层负责：图节点多不是问题，fusion pass（FusedRMSLinear / LoopFusion / EpilogueInjection）会收敛节点。intent_tracker_graph.rs 已用 ~60 节点验证这条路可行。

**否决 A（GatedCell Op）**：只有当 cell 存在 7 种 ComputePattern 无法表达的新计算模式时才成立——它没有。会引入 opaque 结构算子 + 手写 lowering。

**否决 C（RecurrentCell Op + JIT 循环）**：**基于误判**。c1 单次 forward = 单个 turn = 一次 encode + 一次 3 层 cell 栈传播，**forward 内无时间步循环**。递归是**跨 turn**（h 由调用者跨 turn 携带），不是 forward 内的 seq 循环。"生成循环的 RecurrentCell"在解一个不存在的问题。

**3 层 cell 栈 → 走层模板循环（NO-LAYER-EXPAND）**：
- 3 层结构同构、权重不同 = 标准层模板场景。用 `compile_layer_type_body` 编译单模板 + `append_with_mapping` 实例化 3 次，per-layer 权重偏移 = `layer_idx * layer_stride`，tensor 命名 `cn_layer(i, "cell.W_alpha")` → `"L{i}.cell.W_alpha"`。
- 栈语义：L0 输入 = `e_t + h_prev[0]`；L1 输入 = `h_next[0] + h_prev[1]`；每层自带 state slice，按 layer_idx 偏移寻址（与 KV cache 的 layer 索引同构）。

---

## Q2. 跨 turn 状态如何管理？

**推荐：选项 A 的变体——h_prev/h_next 为调用者所有的 buffer，经 BatchContext 扩展字段传入，非新顶层 ABI 参数。**

- 在 BatchContext（SPEC/20 扩展层）新增 `h_prev_ptr`(读) + `h_next_ptr`(写) 字段，调用者分配 + 拥有，跨 turn 存活——**复用 `kv_cache_ptr`(offset 0x08, 调用者所有、可变、跨 forward 存活) 的所有权模式**（借模式，不借 buffer）。
- cell 栈 op 从 `h_prev_ptr` 按 layer 偏移 load，向 `h_next_ptr` store。
- 运行时 API：`f(text, h_prev_buf) → (h_next_buf, intent_logits[7], difficulty_logits[3])`。调用者每 turn 交换 h_prev/h_next。Rust 侧只传指针，零计算（ARCH-RUST-IS-CODEGEN）。

**否决 B（KV cache 复用）**：KV cache 语义（per-head K/V、paged、seq 索引）与稠密 [3×768] hidden 不匹配。KV cache 是 attention 专用，强塞状态 = 数据迁就代码，语义违宪。**只借它的"调用者所有 + ABI 指针"所有权模式，不借 buffer 本身。**

**否决 C（weight blob）**：权重不可变、全调用共享/缓存；状态是 per-conversation 可变数据。状态入 blob 违反 ARCH-MEMORY-FIRST（状态≠权重）+ 破坏 blob 不可变性。明确 no。

---

## Q3. ModernBERT encoder 分支如何加？

**推荐：ARCH_TABLE 注册新 arch（encoder/embedding 族）+ 新建 encoder build 分支，组合现有算子。不要给 xlmr 路径挂 flag。**

实证：现有 encoder 路径 = 绝对位置 emb + type emb；**RoPE 是 decoder-only**（`features.has_rope`）；alternating attention 存在但仅 Gemma decoder（hetero 层模板 sliding/full）。ModernBERT = RoPE + 无 type_emb + local/global alternating **bidirectional** attention → 三者皆与 xlmr 不同 = 全新 encoder 拓扑。

**理由**：
- xlmr 与 ModernBERT 零共享位置/type 机制。复用 xlmr + flag = bool 门控的路径分叉，违反"build 路径顺从模型实际拓扑"。BUILD-COMPILE 边界**允许**按 family 在 BUILD 阶段选策略（`_family` 后缀），所以给 ModernBERT 独立 build strategy 是正确的 BUILD 阶段动作。
- 积木已存在：RoPE op（需在 encoder build 路径启用）、MultiHeadAttention（full + sliding via hetero 模板）、LayerNorm、GEMM。无 type_emb = 直接省略 `Gather(type_emb)`（已按 tensor 存在性条件化，天然可省）。
- 新增能力只有两点：(a) 在 encoder build 路径启用 RoPE；(b) **non-causal sliding-window attention**（bidirectional 局部窗口掩码）——需确认 MultiHeadAttention 是否支持"非因果+窗口"掩码，**这是最大未知（待验证）**。
- 复用 Gemma-4 hetero 层模板机制表达 local/global 交替（`global_attn_every_n_layers`）。granite-311m 冻结 = 只加载，无训练顾虑。

**建议**：encoder 单独建为可复用 embedding-family arch，再在 c1 图中 append tracker 子图 → 单 MegaKernel 单 CALL（ARCH-RUST-IS-CODEGEN），兼顾复用与效率。

---

## Q4. label-query attention head 如何建图？

**推荐：图组合，不新增 Op。**

label-query attention = cross-attention，Q 来自 label_queries(7×768 可学习常量)，K/V 来自序列。现有 attention op 取 `inputs[0]=Q / inputs[1]=K / inputs[2]=V` 为已物化 tensor 指针。c1 head 带 `q_proj` → **Q = GEMM(label_queries, q_proj)**，Q 是 GEMM 输出（满足"Q 来自 prior op"约束），label_queries 只是作为 GEMM 左操作数的常量 tensor。

建图：
- `Q = GEMM(label_queries[7×768], q_proj)`；`K/V = GEMM(seq, pool_proj/context_proj)`
- `MultiHeadAttention(Q,K,V)` → 7 个 attended 向量
- intent 头：`input_norm/context_proj/label_proj`(LayerNorm+GEMM) → `global_w1`(768→384)→act→`global_w2`(384→7)
- difficulty 头：`diff_norm`(1536)→`diff_w1`(1536→384)→act→`diff_w2`(384→3)

与 intent_tracker_graph.rs 双头分类器同构（GemmBias + SiLU + LayerNorm + MeanPool + MHA）。**无新 Op。**

**两个待验证点**：
1. **常量/权重 tensor 作 GEMM 左操作数**：build_graph 现在 GEMM 左操作数惯例是 activation。label_queries 需 `add_tensor_concrete` 后作 GEMM 输入——大概率可行（权重本就是 tensor），但需确认建图 API 不假设左操作数必为 activation。
2. **Concat op**：`diff_norm(1536)=2×768` 暗示 difficulty 路拼接两个 768 向量。算子清单**未见 Concat**——需确认存在，或用向 1536 buffer 的 strided 写入表达。**待验证。**

---

## 改动文件清单

| 领域 | 文件 | 动作 |
|------|------|------|
| SPEC(前置) | `SPEC/C1-GRANITE-TRACKER.html`(新) + PRD/ | 写 PRD-REQ + SPEC req(groundedIn)，spec_write |
| Q3 encoder | `src/arch/registry.rs` | ARCH_TABLE 加 `("modernbert"/"granite", ..., "encoder")` |
| Q3 encoder | `src/arch/auto_graph_fragments/build_modernbert_encoder.inc.rs`(新) | RoPE + 无 type_emb + local/global bidirectional attention 分支 |
| Q3 encoder | `src/model_config.rs` / `model_config_fragments/` | ModernBERT config: local_attention_window / global_attn_every_n / rope_theta(local,global) |
| Q3 encoder | gllm-kernels attention（待验证） | non-causal sliding-window 掩码变体（若缺） |
| Q1+Q4 图 | `src/arch/c1_granite_graph.rs`(新，仿 intent_tracker_graph.rs) | cell 栈组合 + label-query head + 双分类头 |
| Q1 循环 | 复用 gllm-kernels `template.inc.rs` / `append_with_mapping` | cell 层模板循环，`L{i}.cell.*` 命名，无需改 |
| Q2 状态 | `src/engine/batch_context.rs` | h_prev_ptr/h_next_ptr 字段 + setter |
| Q2 状态 | `src/engine/batch_executor.rs` | 运行时绑定 h_prev/h_next 指针（仿 set_input_ids_flat_ptr） |
| Q2 API | `src/` c1 模块 / client.rs | `f(text,h_prev)→(h_next,intent,difficulty)` stateful API |
| Q4 Concat | gllm-kernels（待验证） | 若无 Concat，补 op 或 strided 写 |

## 实现优先级

- **P0 门控**：SPEC-first（PRD+SPEC）+ 拿到 c1 reference forward（cell 精确公式）——不满足不动代码。
- **P1**：ModernBERT encoder arch（Q3）——基础、可复用、产出 e_t，解锁下游。
- **P2**：状态 BatchContext 管线（Q2）——递归基础设施。与 P1 可并行。
- **P3**：cell 栈图组合（Q1）——核心 tracker，依赖 P2。
- **P4**：label-query 双头（Q4）——输出层，依赖 P3。

## 待讨论 / 未决

1. **cell 递推精确公式**（无 reset gate 的 GRU 变体，α/i/c 三门如何合成 h_t）——需 reference forward。
2. **non-causal sliding-window attention** 在 MultiHeadAttention 是否支持。
3. **Concat op** 是否存在（difficulty 1536 路）。
4. **常量 tensor 作 GEMM 左操作数**是否被建图 API 支持。
5. **encoder pooling** 是 CLS 还是 mean（granite-embedding 惯例）。
6. **encoder 与 tracker 是否融合为单 MegaKernel**（推荐融合，单 CALL）还是两段图。
7. 状态承载：确认走 BatchContext 扩展（推荐）而非新顶层 ABI 槽。

## 证据表（file:line）

- ABI 22参/184B：`src/engine/mega_kernel/abi_types.inc.rs:80-157,274`；kv_cache_ptr offset 0x08 调用者所有:118,436-450
- 外部 I/O 经 BatchContext：`src/engine/batch_executor.rs:50-146`(set_input_ids_flat_ptr:137 等)
- OpKind/TraceOp：`gllm-kernels/src/compiler/trace.rs:174-402`；`registry_fragments/types.inc.rs:1-161`；Sigmoid trace.rs:202 / Tanh:196
- 7 ComputePattern：`gllm-kernels/src/compiler/trace.rs:26-87`；auto_select 分发:53+,131-134
- 无递归 op：全仓零 LSTM/GRU 实现（仅 onnx loader 测试 fixture）
- attention inputs[0..2]=Q/K/V 物化指针：`lower_op.inc.rs` lower_attention_v2；Q 经 GEMM `build_graph.inc.rs:~565-580`
- encoder 绝对pos+type-emb、RoPE decoder-only：`build_graph.inc.rs:212-273`(pos:223-230,type-emb 条件:236),RoPE:670-678
- alternating attention hetero 模板：`build_graph.inc.rs:369-402,1296-1509`；hetero config types.inc.rs:97-119
- 层模板循环：`gllm-kernels/.../plan_lower/template.inc.rs:20-99`；`hetero_emit.rs:24-127`；`append_with_mapping`(program.inc.rs)；cn_layer `build_graph.inc.rs:1-12`
- ARCH_TABLE：`src/arch/registry.rs:12,14-54`(xlmr encoder:~,resolve:71-85)
- intent_tracker_graph 双头结构：`src/arch/intent_tracker_graph.rs`（~60 op，GemmBias/SiLU/LayerNorm/MeanPool/MHA，task3+diff4）
