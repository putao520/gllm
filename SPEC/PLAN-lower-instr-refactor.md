# PLAN: lower_instr God-Match 根治 (BCE-20260630-LOWER-INSTR-GOD-MATCH)

> 范式缺陷根治计划。单 match 反模式 + OCP 违反,跨三 ISA 全量清零。
> 关联铁律: ARCH-AUTO-INSTR-SELECT / ARCH-ISA-PERF / NO_SILENT_FALLBACK / ARCH-JIT-GENERATOR / C-5 范围守恒。

---

## 1. 问题定义 (arch_insight 量化)

| 文件 | 函数 | 行数 | 圈复杂度 |
|------|------|------|---------|
| `vm/gpu_lower/lower_instr.inc.rs` | lower_instr | 5090 | 1131 |
| `vm/x86_lower/lower_instr.inc.rs` | lower_instr_inner | 3908 | 699 |
| `vm/aarch64_lower/lower_instr.inc.rs` | lower_instr | 3340 | 587 |

总计: 9 long_method (≥500 行) + 60 high_cyclomatic (≥10)。
根因: 153 个 VmInstr 变体的 lowering 逻辑(寄存器分配/偏移算术/SimdWidth×ElemStrategy×dtype 嵌套)全量内联在单 match 的 arm 体内。每新增 VmInstr,三个 lower_instr 同步膨胀(违反 OCP)。

---

## 2. 层次澄清 (关键裁决依据)

ARCH-AUTO-INSTR-SELECT 治理 **TraceOp → VmInstr** 选择层(`auto_select.rs`,已合规)。
本计划治理 **VmInstr → 机器码** 后端 ISA lowering 层(`lower_instr`)。两层不同关注点。

- `auto_select.rs` (3113 行): 已是 ComputePattern 分类 dispatch + emit_binop/unary/transcendental helper 复用,新增 TraceOp 仅一行。**不在本计划范围(已合规)**。
- `lower_instr` (×3): 本计划根治目标。

---

## 3. 路径裁决: A+C 两级组合,拒绝 B

**拒绝 B (per-VmInstr trait dispatch)**: 153 变体 × 3 ISA = 459 impl 块,违反 ARCH-ISA-PERF(各 ISA 算法结构本就不同),零复用,可维护性下降(三 ISA 实现散落)。

**采纳 A (类别 dispatch) + C (叶子抽取) 两级组合**:

```
L0 lower_instr(instr)              分类 dispatch: match instr.category(), arm 仅单表达式委托
L1 lower_<category>(instr)         变体路由:     match 该类变体,    arm 仅单表达式委托
L2 emit_<variant>(...)             叶子 emit:    单变体作用域胖逻辑,  logic CC≤10 / ≤80 行
```

依据: 纯 dispatch match(扁平、arm 单表达式、无嵌套分支)的圈复杂度非有害复杂度,是 OCP 扩展点。两级分类压 arm 数,叶子抽取压 logic CC。

---

## 4. 分类器设计

- `VmInstr::category(&self) -> InstrCategory` 定义在独立文件 `vm/vm_instr_category.rs`(与 `vm_instr_meta.rs` 并列;meta=fusion 语义,category=lowering 路由,关注点分离)。
- 分类器跨三 ISA 共享同一份定义。
- 初始类别(Phase 1 x86 实测后按 arm 分布定稿,Quant 先独立成类):
  - `Memory` (VecLoad/VecStore/Broadcast/Mov/LoadPtr/Gather/Scatter/TableLookup...)
  - `Arith` (VecBinOp/VecUnaryOp/Fma/HReduce/Accumulate/VecCmp/ConditionalSelect...)
  - `Control` (LoopBegin/LoopEnd/ScopeBegin/Branch/ConditionalSkip/MarkLabel...)
  - `Tile` (TileConfig/TileLoad/TileMma/TileStore/TileRelease/Tmem*...)
  - `Quant` (Quant*/Kivi*/Gguf*/Q3KDecodeStep/HwQuantDequant...)
  - `GpuComm` (WarpSync/AsyncCopy/SharedMem*/AllReduceChunk/Nvlink*/Rdma*... — GPU 专属)
  - `Sampling` (Argmax/Softmax*/Sample*/WarpPRNG/StoreToken/CheckStopCondition...)
  - `Misc` (Comment/Debug*/NativeCall/Prefetch...)
- ISA 不支持的类别: dispatch 走 `Err`(禁止静默 NOP,呼应 NO_SILENT_FALLBACK)。

---

## 5. 范围守恒 (C-5, 平权全量,零差集)

终态 ≡ 当前全部 god-match 清零。全部交付项平权,无主次:

- 6 处 match 全部根治: x86 / aarch64 / gpu 三 lower_instr + program + verify + reg_alloc。
- 153 个 VmInstr 变体在每个 ISA 全量 lowering,无遗漏,无静默 NOP。
- 60 处 high_cyclomatic + 9 处 long_method 全部清零。
- 三个 ISA 各实现完整 L0/L1/L2 三级,平权全量完成。
- 分 Phase 仅是 commit 执行顺序,非按重要性筛选,不缩减任何交付项。

---

## 6. 执行编排 (三 Phase,commit 顺序)

| Phase | 范围 | 文件域 | 验证门 |
|-------|------|--------|--------|
| P1 | x86 lower_instr 全量三级拆分,确立可复用实现模式 | x86_lower/ | golden test 全绿 + commit_gate + arch_insight(quality) x86 部分清零 |
| P2 | aarch64 + gpu lower_instr 按同模式全量拆分(撞不同文件域,可并行 Agent;gpu 启动前核 gpu_ir 方言边界) | aarch64_lower/ ‖ gpu_lower/ | 同上,三 ISA 全清零 |
| P3 | program/verify/reg_alloc 的 match 全量根治 | vm_state/verify/reg_alloc | 同上,60 处 high_cyclomatic 全清零 |

GPU 适配判定(P2 启动前): 若 `gpu_ir/` 已有 GpuDialect trait 层 → GPU dispatch 适配方言分发;否则同 x86 类别分发。

---

## 7. 验证红线 (JIT 行为保持)

静默错误机器码是最危险 bug(NO_SILENT_FALLBACK)。整个重构是行为保持变换:

- 每 Phase 前后跑全量 golden test (`cargo test --lib`),三 ISA lowering 正确性测试 diff 必须为零。
- L2 抽取优先用 `refactor_code(extract_function)` 机械抽取,非手写 Edit(防漏寄存器/偏移状态)。
- ARCH-JIT-GENERATOR 的 `CodegenContext`/`OffsetValidator` 状态在抽取后必须仍闭环,不跨函数边界丢失。

---

## 8. SPEC criterion 沉淀 (防复发,P3 完成时写入 SPEC/02-ARCHITECTURE.md §8)

**ARCH-LOWER-DISPATCH-LAYERING (VmInstr lowering 分层铁律)**

1. `lower_instr` 必须两级 dispatch: L0 按 `VmInstr::category()` 分类,L1 按变体路由,二者 arm 体只允许单表达式委托(无内联逻辑)。
2. L2 叶子 emit 函数: 单变体作用域,logic 圈复杂度 ≤10,行数 ≤80。
3. 新增 VmInstr 变体 = 1 行 dispatch + 1 个叶子 emit 函数,禁止在 dispatch/handler arm 内联逻辑(OCP)。
4. dispatch 禁止 catch-all 静默 NOP(NO_SILENT_FALLBACK),未实现变体返回 `Err`。

配套: 写回归断言 + 更新 `SPEC/BUG-KNOWLEDGE.md` (patternId: BCE-20260630-LOWER-INSTR-GOD-MATCH)。

---

## 9. 验收标准 (残留=0 才放行)

- [ ] 6 处 god-match 全部两级 dispatch 化
- [ ] 60 处 high_cyclomatic 清零 (arch_insight quality 复扫)
- [ ] 9 处 long_method 清零
- [ ] 三 ISA × 153 变体全量 lowering,无静默 NOP (grep 验证无 emit_nop catch-all)
- [ ] golden test 全绿 (三 ISA lowering 正确性 diff=0)
- [ ] commit_gate canCommit=true
- [ ] ARCH-LOWER-DISPATCH-LAYERING criterion 写入 SPEC + 回归断言入库 + BUG-KNOWLEDGE.md 沉淀
