# G2b 根因判定 + act_dt 语义 + 根治方案（architect consult）

> 来源：architect consult sessionId=426a2014（G2b act_dt 根因判定）+ 项目代码分析
> 建库触发：BCE-20260708-G2b-ACCUM-AS-LOAD-STRIDE（GEMM A-load dtype 硬编码 ctx.accum_dtype）
> 最后验证：2026-07-08

> 结论先行：G2b 失败**不是** 5 个嫌疑点里的任何一个。真正根因是第 6 个、被历次方案漏掉的
> dtype 消费者——**GEMM 激活矩阵 A-load 的 dtype 硬编码为 `ctx.accum_dtype`（F32），从不跟随激活张量
> 的实际存储 dtype**。act_dt=BF16 时激活按 2B 存、按 4B 读，首个投影 GEMM 就 2× 越读 → 全程乱码。

---

## 1. 根因判定（单一，已代码验证）

**违宪点（新增，第 6 个）**：`gllm-kernels/src/compiler/codegen/vm/plan_lower/lower_op.inc.rs:1365`
```rust
let a_dtype = ctx.accum_dtype;   // ← 激活 A-load dtype 硬编码 = 累加精度 F32
```
下游 `gemm_emit.rs:313` 用它算 **load 步长**：`let a_elem = a_dtype.elem_bytes();`

- K1（K-stride:1488）、K3（C-store:1358）、K4（QuantGemm 输出:118）、executor KV-buffer(:95) 你都审了，
  它们**确实一致**——因为 KV/输出域本来就一致。
- 你漏审的是 **GEMM A 矩阵（激活输入）的 load dtype**。它不读 `op.inputs[0]` 张量的 dtype，
  而是钉死 `ctx.accum_dtype`。这是 K1/K3/K4 同一类违宪的**第 6 个孪生实例**，历次 D1–D5 计划全漏了它。

**失败机理**：act_dt=BF16 → 激活张量（normed / q_out / …）声明为 BF16，物理存 2B/elem。
首个投影 GEMM（q_proj）的 A = normed（BF16, 2B），但 `a_dtype=F32` → `a_elem=4` →
每行按 4B 步长读 2B 缓冲 → 2× 越读 → 从 attention 之前的**第一个 GEMM 就已污染**。
模型在进入 KV/attention 域之前就已经是垃圾——所以你在 KV 域的所有 stride 分析都"一致"却依然乱码。

---

## 2. PASS/FAIL 悖论解答（你问的核心矛盾）

**act_dt=F32（回滚，PASS）为什么 768 vs 384 不越界？**
- 768 是**激活空间** K 张量（k_out，F32）的行步长；384 是**独立的持久化 KV cache 缓冲**（按
  compute_dtype=BF16 分配）的行步长。这是**两个不同缓冲**。
- FromCache 写入是一次 **dtype 感知的窄化拷贝**（F32 激活源 → BF16 cache 目的），源用 768、目的用 384，
  逐元素转换，各自用**自己正确的步长**，无越界。你把它误读成"768 踩 384"，实则是合法的 narrowing copy。
- 全链投影 GEMM 的 A-load（F32）匹配 F32 激活存储 → 全对 → PASS。

**act_dt=BF16（FAIL）为什么 384==384 匹配反而崩？**
- KV 域确实变一致了（K 张量 BF16→stride 384 == buffer 384）。但**首个投影 GEMM 的 A-load 崩了**
  （2B 存储 vs 4B 读步长）。KV 域的"匹配"是真的，可上游 GEMM 早已产出垃圾。
- 一句话：你盯着 KV 域找 bug，真正的 bug 在它上游的投影 GEMM A-load，与 KV/attention 无关。

---

## 3. 5 个嫌疑点逐一裁决

| # | 嫌疑 | 裁决 |
|---|------|------|
| 1 | VecNarrow GEMM store | 否。C-store 窄化路径正确（acc=c_dtype.accumulator_dtype()=F32，VecStore 负责 narrow）。SmolLM2 投影无 bias，narrow 路径没参与失败 |
| 2 | Q·K 走 BF16 点积 | **架构上不可能**。`trace.rs:1103` BF16→WidenCompute→accumulator=F32。累加恒 F32 |
| 3 | KV FromCache MemCopy dtype 不匹配 | 否。那是合法窄化拷贝（见 §2），不是 bug |
| 4 | BF16 累加发散 | **不可能**，同 #2。累加器结构性 F32 |
| 5 | embedding BF16 widen | 否。Gather 是 dtype 无关的字节行拷贝，act_dt=BF16 时 embedding=BF16 自洽 |
| **6** | **GEMM A-load dtype=accum_dtype 硬编码，不顺激活张量** | **✅ 真根因** |

---

## 4. act_dt 正确语义（回答 Q3：它是存储还是计算精度？）

**act_dt 是「激活存储 dtype」，不是「计算/累加 dtype」。** 证据：`g.add_tensor(..., act_dt)` 决定的是
张量在图里的**存储 dtype**（内存字节布局）。而累加精度是**另一个独立量**，由
`QuantPrecision::accumulator_dtype()`（trace.rs:1100）在 emit 时导出，BF16→恒 F32。

build_graph.inc.rs:76-80 的注释"act_dt = 激活/计算精度…GEMM 的 A-load/FMA 累加/C-store 全程用此精度"
是**错误注释**（BUG-KNOWLEDGE:3257 已识破"act_dt 注释撒谎"）。真相：act_dt 只到**存储/load**，
**从不到累加器**；累加器永远走 `accumulator_dtype()`=F32。

因此需要区分三个正交 dtype，不能再混：
- **存储 dtype**（activation storage）= act_dt → 必须顺 config（宪法 -1 管辖）。A-load **步长**用它。
- **权重存储 dtype** = tdt(name) → 顺权重文件自描述。B-load 步长用它。
- **累加 dtype**（FMA 寄存器精度）= 恒 F32，由 accumulator_dtype() 导出 → **不是精度立场，是数值正确性
  约定（硬件惯例）**，合法地不受 config 驱动。

**「act_dt=F32 硬编码是否违宪」的裁决**：违宪。因为 act_dt 语义是**存储 dtype**，存储必须顺数据/配置。
但"累加器恒 F32"**不**违宪——它是独立量，已正确落在 accumulator_dtype()，与 act_dt 无关。
方案 C（"承认激活计算永远 F32"）混淆了这两者：把合法的"累加恒 F32"当借口保留非法的"存储写死 F32"。

---

## 5. 根治方案：选 B（act_dt 真顺 config），但必须先补第 6 个切除点

方案 A（act_dt 保持 F32、只让 KV 顺 config 解耦）= 放弃 BF16 激活、放弃宪法 -1，正是用户已否决的"回避"。**不选**。
方案 C = 语义混淆（见 §4）。**不选**。
**方案 B（选）**：act_dt 从 config 派生（G2b 方向正确），但 G2b 单独上必崩——因为它没修第 6 个 A-load 消费者。
顺序必须是：**先补 A-load 顺张量（记为 G4/D6），再上 G2b**。

### 切除点 G4/D6（新增，强制先于 G2b）
`lower_op.inc.rs:1365` 改为镜像 c_dtype/b_dtype 的派生方式（从输入张量 TensorMeta 读）：
```rust
// a_dtype 顺激活张量存储 dtype（op.inputs[0]），非硬编码累加精度。
// 累加器仍 F32：BLIS 内 acc = c_dtype.accumulator_dtype()；a_dtype 只管 load 步长 + WidenCompute。
let a_dtype = op.inputs.first().copied()
    .and_then(|tid| graph.tensor(tid))
    .map(|t| t.dtype.to_quant_precision())
    .unwrap_or(ctx.accum_dtype);   // fallback 仅防御
```
- act_dt=F32 时：输入张量 F32 → a_dtype=F32 → **零回归**（与现状字节完全一致）。
- act_dt=BF16 时：输入张量 BF16 → a_dtype=BF16 → VecLoad 读 2B + WidenCompute→F32 累加 → 正确。
- 落地建议：这是"改 dtype 来源"而非搬移代码，直接 Edit 即可；无需 refactor_code。

### 伴随必查项（同类违宪，需 Executor 用同一 pattern 审计，未全部验证）
1. **MHA attention Q/K/V load 路径**（lower_op.inc.rs MHA 分支）：其 Q·K 的激活 load 是否也钉死 accum/F32？
   投影 GEMM 先崩所以未观测到，但 act_dt=BF16 全链前必须确认 attention load 也顺张量 dtype。**待验证**。
2. **GemmBias 的 C load/store**（lower_op.inc.rs:1406 `spec.dtype.size_bytes()`）：用的是**权重 dtype** 当激活
   C 步长——同类 split-brain。SmolLM2 核心投影无 bias 未触发，但混合精度带 bias 模型会踩。应改用 c_dtype。**待修**。
3. 其余激活消费算子（Norm/Add/逐元素/Gather 之后的 load）全量扫一遍 "用 accum_dtype/spec.dtype/硬编码
   *4 当激活步长" 的点。建议 Executor 用 `search_code(action="scan")` 找 `elem_bytes\|size_bytes\|\* 4`
   与激活张量 load 的组合。

### 落地顺序
```
G4/D6（lower_op:1365 a_dtype 顺输入张量）  → 编译 + SmolLM2 E2E（act_dt 仍 F32，验零回归）
  → 伴随项 #1 attention load 审计/修复      → E2E 复验
  → G2b（build_graph:85 act_dt = config.compute_dtype）→ SmolLM2 E2E（此时 BF16 激活，验通过）
  → 删/改 build_graph:76-80 错误注释（标注 act_dt=存储 dtype，累加器独立 F32）
  → 伴随项 #2/#3 混合精度带 bias 模型的补丁（非 SmolLM2 blocker，可后续）
```
关键：**G4/D6 必须先于 G2b**，与 BUG-KNOWLEDGE"D3 先于 D1"同理——所有激活 load 消费者顺张量后，
act_dt 变 BF16 才是安全的。

---

## 6. 建议新增/修正的 SPEC 元素（方向，详义交 spec_write）

- **REQ-DTYPE-CHAIN-00x（新）**：GEMM/MHA 激活 A-load dtype 必须顺 `op.inputs[0]` TensorMeta.dtype，
  禁止用 `ctx.accum_dtype` 或任何硬编码值当激活 load 步长。关键约束：load 步长=存储 dtype，累加=F32，二者正交。
- **修正 build_graph.inc.rs:76-80 注释契约**：act_dt 定义收敛为「激活**存储** dtype」；显式声明
  "累加器恒 F32 由 accumulator_dtype() 导出，与 act_dt 解耦，不受 config 驱动"。
- **REQ-DTYPE-013 补强**：现有"GemmSpec.dtype 用 tdt(权重名)"已覆盖 B；需补一条对称的 A（激活）约束。

---

## 7. 待讨论 / 未决

- **伴随项 #1（attention load）未经代码验证**——我只确认了投影 GEMM 的 A-load 是首个断点。act_dt=BF16
  全链跑通前，Executor 必须先读 MHA lowering 分支确认其激活 load 同样顺张量 dtype，否则 G2b 会在
  attention 处二次崩。这是唯一的"待验证"项，建议优先。
- act_dt=BF16 通过后仍有**数值精度回归**（BF16 激活 vs F32 激活，输出会有差异但应可用）——这是预期的、
  合法的精度变化，不是 bug。需与调用方确认 SmolLM2 E2E 的通过判据是"输出连贯可读"而非"逐 bit 等于 F32 基线"。
