# 架构决策：GPU GEMM accs[] 数组设计与 warp 映射模型

> 关联：BCE-20260702-GPU-OOM 下游卡点。决策者：architect。状态：proposed（待 Commander/用户确认）。
> 宪法约束：ARCH-JIT-YIELDS P0（GPU 按 GPU 编程模型）、ARCH-NO-LOOP-UNROLL、ARCH-ROOT-CAUSE（治本）、NO_SILENT_FALLBACK。

---

## 结论先行

| 项 | 决策 |
|----|------|
| **唯一正确方案** | **B（重新设计 warp 并行分层 tile 映射）** —— 唯一治本且符合 GPU 编程模型的方案，覆盖全部 GPU 契约完整交付 |
| **否决 A** | **A（accs 搬 shared memory）** —— 反模式，把错误架构固化 |
| **否决 C** | **C（lm_head 走 CPU）** —— 会使 GPU logits 路径处于未实现状态，与「GPU 契约全覆盖」冲突（C-5 交付范围 ≡ SPEC 契约范围，双向零差集）。决定权在用户，architect 默认按全覆盖交付 |
| **范围** | **GPU codegen 架构级缺口（缺线程/lane 模型），非单 GEMM kernel** |
| **附带 BCE** | `acc_idx < accs.len()` 静默守卫 = 静默错误输出，违反 NO_SILENT_FALLBACK，须走 BCE 闭环 |

---

## 根因重定位（比 teammate 描述更深一层）

teammate 把卡点描述为「accs 16 槽装 2048 + VReg 爆炸」。这是**症状**。真正根因：

> **GPU codegen 根本没有线程/lane 模型。它把 CPU BLIS 的 row/col 累加逻辑原样搬到 GPU，让"一个线程"承担整个 `warp_m×warp_n` 输出 tile。**

证据（Explore 探查）：
- `warp_m`/`warp_n`（wi/wj=64/32）是 **Rust 编译期 unroll bound**，不是线程→元素映射（gemm_emit.rs:500-503）
- GEMM codegen 内**无 `lane_id`、无 `threadIdx` 参与计算、无 per-thread 寄存器分区**（Explore §4）
- 部分 GPU op 甚至是「`if (threadIdx.x==0){...} __syncthreads()`」单线程串行 —— 正确但灾难性慢
- `accs = (0..warp_m*warp_n.min(16))` + `if acc_idx < accs.len()` 守卫（gemm_emit.rs:445-448, 570/595）

**因此**：
1. 小 M（Concrete m=16）不 OOM 只是因为 acc_count 小，**输出本就不完整**（只填前 16 acc），测试没暴露是因为断言没覆盖 warp_m×warp_n>16 的输出正确性。
2. Symbolic M（lm_head vocab=49152）不是引入了新 bug，而是**把一个预存的正确性缺陷 + 架构错位放大成了 VReg 爆炸**。

结论：这不是「16 太小、调大就行」，调到 2048 会立刻 VReg 爆炸 + spill → GPU 无栈 → StackArg 报错。**任何"在单线程里塞下整个 tile"的路线都是死路**，无论 acc 放寄存器还是 shared memory。

---

## 四个问题的回答

### Q1：哪个选项治本且符合 GPU 编程模型？

**B。** GPU GEMM 的标准做法（CUTLASS/cuBLAS/所有高性能实现）是**分层 tile + warp 并行**，不是单线程顺序 row/col 迭代：

```
Block tile (CTA)  ─ 一个 thread block 负责的输出块，如 128×128
  └ Warp tile     ─ 一个 warp（32 lane）负责的子块，如 64×32
      └ Thread tile ─ 单个线程负责的寄存器 tile，如 8×8 = 64 acc（留寄存器）
```

- 2048 个输出单元**分布在 warp 的 32 lane × 每 lane 的 thread tile** 上协同覆盖，**不是一个线程扛 2048**。
- 单线程只持有自己的 thread tile（TM×TN，如 8×8=64 acc），远低于 PTX ~255 f32 寄存器上限 → 无 VReg 爆炸、无 spill、无 StackArg。
- 运行时并行来自 `threadIdx/laneId/blockIdx`，不是来自 Rust unroll。

顺序 row/col 迭代（当前实现）**根本不是 GPU 模型**，是 CPU 单核 SIMD 模型误植。

### Q2：accs 数组的设计正解 —— 寄存器 vs shared memory？

**寄存器，且尺寸 = thread tile（小、编译期常量），不是 warp tile。**

- **accumulator 必须全程留寄存器**（贯穿 K-loop）。这是高性能 GPU GEMM 的铁律 —— CUTLASS 的 fragment accumulator 就是寄存器。
- **shared memory 是给 A/B operand staging 用的**（smem tile + 双缓冲），**不是给 accumulator**。把 acc 放 smem = 每次 FMA 走 smem 读写 → 带宽/bank conflict/同步开销吃光收益，且 smem 是 per-block 资源（~48–228KB），扛不住 per-thread 大 acc。
- **关键洞察 —— 这直接化解 teammate 的「emit_loop counter 无法编译期索引 accs」冲突**：

  > 在正确模型里，accs **永远是小的、编译期索引的**（thread tile TM×TN 是小编译期常量，Rust unroll 进寄存器完全 OK）。运行时规模来自**线程/块并行**，不是来自「用运行时 counter 索引一个大 acc 数组」。所以 emit_loop 与 accs 编译期索引**根本不冲突** —— 冲突是错误 tiling 模型的伪命题。

  分工：
  - K-loop（reduction，运行时 bound）→ `emit_loop(Symbolic/Const)`
  - M/N grid 跨 tile（运行时）→ 由 `blockIdx/threadIdx` 映射，不是 unroll
  - Thread tile 内 TM×TN 累加 → Rust 编译期 unroll，accs[编译期 idx]，留寄存器 ✅

### Q3：lm_head 策略（M=seq_len 小, N=vocab=49152 大, K=hidden）？

lm_head 是 **logits projection**，按 M 分两种 regime，**都在 GPU 上完整实现**：

- **M=1（decode）**：这是**纯 GEMV / memory-bound**，正解是 **N 并行 GEMV** —— 沿 N=49152 切分，每线程/warp 负责一段输出，各自沿 K reduction。**根本没有 acc tile 问题**（每个输出 1 个 acc + K 归约）。当前 CPU 侧已有 M=1 streaming GEMV 路径的对应概念，GPU 侧应有等价 kernel。
- **M 中等（prefill, 5–8192）**：标准 tiled GEMM，但 **tile 配置随 shape 自适应** —— 小 M tile + 沿大 N 迭代。B 方案的 block/warp/thread tile 一旦参数化，lm_head 只是「M 小 N 大」的一组 tile 参数，不是特例。

**lm_head 不特殊到需要 CPU 特判**（C 方案）。它暴露问题只是因为 N 大把「单线程扛 tile」的错误放大了。B 落地后自然覆盖。把 lm_head 剥离到 CPU 会使 GPU logits 路径处于未实现状态，与 GPU 契约全覆盖冲突（C-5 双向零差集）。

### Q4：范围 —— 单 kernel 还是整体 accs/accumulator 抽象重构？

**GPU codegen 架构级缺口，非单 kernel。**

- 缺的是**线程/lane 模型 + thread-tile 抽象**，这是所有 GPU 累加类 kernel 的共同基础：GEMM、attention（QK^T/PV 累加）、任何 reduction。
- `acc_idx < accs.len()` 静默守卫 + `.min(16)` 的**同源缺陷会出现在每一个从 CPU BLIS 搬过来的 GPU 累加路径**（Explore §7 确认 CPU/GPU 共享 row/col+guard 结构）。
- 佐证：CLAUDE.md 标注 `04-GPU-BACKEND 🔴 未实现` —— GPU 后端本就是未成熟的部分移植，线程模型从没建起来。

**落地拆解（全量交付，三步是同一次完整交付的施工顺序，每步满格，交付边界恒定）**：
1. 在 GPU GEMM emitter 内引入 `GpuThreadTile`（TM×TN）+ lane/thread 映射抽象
2. 抽出通用 `gpu_ir` 层的线程分区原语，供 attention / reduction 复用
3. 用 `refactor_code(extract_function)` 把共享的 thread-tile 逻辑提取，避免 CPU/GPU guard 缺陷再扩散

---

## 附带必做：BCE 闭环（NO_SILENT_FALLBACK 违规）

`if acc_idx < accs.len() { emit FMA }` 是**静默丢弃输出**：编译成功、kernel 跑通、结果部分为零/垃圾，常规测试发现不了。这**正是项目自己 CLAUDE.md 里 NO_SILENT_FALLBACK 铁律禁止的模式**（同 `emit_nop_raw()` catch-all 一类）。

按 C-7 走 BCE：
- **泛化**：BUG 模式 = 「累加类 codegen 用 `idx < accs.len()` 静默守卫跳过应算的输出单元」
- **横扫**：`search_code(scan, focus="patterns")` + grep `acc_idx < accs.len()` / `.min(16)` / 类似守卫，全项目（CPU BLIS 路径同样命中）
- **根治**：正确路线下 accs 尺寸 = thread tile，天然 == 实际输出单元数，**守卫消失**；过渡期任何 acc 不足必须 `Err`，禁止静默跳过
- **沉淀**：回归测试须断言 warp_m×warp_n > 16 时输出**完整且数值正确**（现有测试缺这条，是缺陷被掩盖的根因）

---

## 待讨论 / 未决问题

1. **C 方案是否触发**：默认按 GPU 契约全覆盖交付（B），lm_head 在 GPU 上完整实现。C（lm_head 走 CPU）会使 GPU logits 路径处于未实现状态，此项决定权在用户，architect 不代为选择。
2. **GPU 线程模型的目标形态**：确认 B 是否引入完整 SIMT 线程抽象（threadIdx/laneId 参与 codegen），还是先做 GEMM 局部的 thread-tile。这决定重构是「GEMM 内」还是「gpu_ir 全层」。倾向 gpu_ir 全层（attention 迟早要），需 Commander 拍板范围。
3. **M=1 GEMV 是否独立 kernel**：lm_head decode 是否走独立 GPU GEMV 路径（而非 tiled GEMM 的退化配置）？影响 kernel 数量与分派逻辑。
4. **SPEC 落点**：本决策 + 线程模型契约应写入 `SPEC/04-GPU-BACKEND`（当前 🔴 未实现），并新增 domain/decision 元素。GPU 编程模型约束应升为 SPEC criterion，约束后续所有 GPU 累加 kernel。
5. **thread tile 尺寸参数化**：TM×TN 的默认值 + 随 shape/dtype 自适应策略，需结合 DeviceProfile（SM 寄存器预算）定，属 B 落地时的调参，非本决策范围。
