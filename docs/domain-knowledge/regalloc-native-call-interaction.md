# RegAlloc × Native Call 交互 领域资料库（自研线性扫描 RegAlloc + native call 边界）

> 来源：System V AMD64 ABI（x86 calling conventions，Wikipedia 权威摘要 + Agner Fog calling_conventions）+ LLVM CodeGenerator 文档（register mask 机制）+ 项目源码验证（gllm-kernels reg_alloc.rs / isa_profile.rs / lower_instr_dispatch.inc.rs）
> 建库触发：BCE-20260712-Q5_K_M-REGALLOC-KNOWLEDGEGAP（architect consult agentId adecf1a90d91cbbd1 报告 knowledgeGap：RegAllocator 在混合 native-call 序列下的运行时行为无训练数据覆盖）
> 最后验证：2026-07-12

---

## 核心机制（与出错相关的）

### 1. SysV AMD64 ABI 寄存器约定（权威事实）

| 类别 | 寄存器 | 跨 call 行为 |
|------|--------|-------------|
| **Callee-saved（非易失）** | RBX, RSP, RBP, R12-R15 | callee 必须恢复原值；caller 可假定跨 call 持久 |
| **Caller-saved（易失）** | RAX, RCX, RDX, RSI, RDI, R8-R11, 所有 XMM/YMM/ZMM | callee 可自由修改；caller 须自己保存才跨 call |
| **参数传递（整数）** | RDI, RSI, RDX, RCX, R8, R9（前 6 个） | 第 7 个起栈传 |
| **返回值** | RAX（≤64bit）/ RAX+RDX（≤128bit） | — |
| **栈对齐** | call 前 16 字节对齐（call push 8B 返回地址 → callee 入口 16B 对齐） | — |

**关键规则**（SysV ABI §3.2 + Wikipedia x86 calling conventions）：native call（extern "C"）**clobber 所有 caller-saved 寄存器**，callee 不保证它们跨 call 持久。

### 2. LLVM 的 call clobber 建模（register mask）

LLVM CodeGenerator 用 **register mask** 机制建模 call：call 指令带一个 `MO_RegisterMask` 操作数，mask 标记 preserved 寄存器，**其余视为被 clobber**。这把"call clobber 所有 caller-saved"建模进干涉图，让 regalloc 把跨 call 的 live VReg 分到 callee-saved 或 spill。

**线性扫描限制**：LLVM 文档对线性扫描跨 call 的具体行为（live interval 跨 call 时的 spill vs callee-saved 决策）无详述——这正是 architect 报告的 knowledgeGap 来源。

### 3. gllm-kernels 自研 RegAlloc 的实际行为（源码验证）

**RegAlloc 不显式建模 call clobber**，而是依赖 **lower 层 save_gprs 全量保存**：

- `reg_alloc.rs:1222-1230` GPR 分配池选择：
  - `Counter/ByteOffset` 优先 `callee_saved`（跨循环生存，正确）
  - `Ptr/Scalar` 优先 `caller_saved`（非跨循环，含 block_base/lane_offset）
- **无 register mask / call clobber 建模**：RegAlloc 只看 VRegKind + occupied 干涉集，不感知 native call 边界
- block_base/lane_offset（Ptr kind）实测**走 Spilled 路径**（regalloc dump 确认），不持有物理 caller_saved → 跨 call 通过 spill slot reload（resolve_gpr_read 每次从 [rbp+off] 重载）

**save_gprs 全量保存兜底**（lower_instr_dispatch.inc.rs，4 个 decode step 对称）：
```
save_gprs = [rax, rbx, rcx, rdx, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15]  // 14 GPR
```
覆盖 RegAlloc 可分配池的全部 11 GPR（caller_saved ∪ callee_saved，scratch rax/r10/r11 虽在 save_gprs 但不在 RegAlloc 池）。**若 save_gprs 漏一个 caller_saved → 跨 native call 的 live VReg 被静默 clobber → 运行时值丢失**。

### 4. 关键不变量（源码 + dump 双验证）

| 不变量 | 验证 |
|--------|------|
| RegAlloc 可分配池 = caller_saved ∪ callee_saved = 11 GPR | isa_profile.rs:412-429 |
| scratch_gprs [rax, r10, r11] 不进 RegAlloc 池 | isa_profile.rs:419（ARCH-ISA-SCRATCH）|
| save_gprs 覆盖全部 11 可分配 GPR | regalloc dump Q5_K_M/Q6_K 都只用 PhysGpr(1,2,3,6,7,8,9,12,13,14,15)，全在 save_gprs |
| block_base/lane_offset Spilled，resolve_gpr_read 每次从 spill slot reload | helpers.inc.rs:287-318 + regalloc dump |
| 物理 GPR 使用 Q5_K_M == Q6_K（11 个，频次比例近同）| regalloc dump 对比 |
| spill offset 0 重复（无 slot 冲突）| regalloc dump uniq -d |

---

## AI 易误判点

- ❌ 误判："RegAlloc 应该像 LLVM 用 register mask 显式建模 call clobber" → 以为缺 mask 是 bug
- ✅ 正解：自研 RegAlloc 依赖 **lower 层 save_gprs 全量保存**兜底，不显式建模 call clobber。这是设计选择（非 LLVM 式），只要 save_gprs 覆盖完整就正确。源码验证覆盖完整。

- ❌ 误判："block_base 是 Ptr kind 优先 caller_saved，跨 native call 会被 clobber"
- ✅ 正解：regalloc dump 显示 block_base/lane_offset 实测 **Spilled**（不持有物理 caller_saved），跨 call 通过 spill slot reload（resolve_gpr_read 每次从 [rbp+off] 重载，不假设 scratch 持久）。

- ❌ 误判："混合 DecodeStep 序列（Q5K+Q6K）会让 RegAlloc 产生与纯序列（Q6K）不同的物理寄存器分配"
- ✅ 正解：VmProgram 结构对称（归一化后）→ regalloc 输入对称 → 线性扫描算法确定性 → 分配模式对称。regalloc dump 确证：Q5_K_M 与 Q6_K 用**完全相同的 11 物理 GPR**，spill 布局前 30 VReg 逐字节相同。27 vs 33 DecodeStep 差异纯量化参数，非 VReg 生命周期差异。

- ❌ 误判："caller_saved 寄存器跨 native call 一定丢失" → 以为所有 caller_saved 都危险
- ✅ 正解：SysV ABI 规定 caller 须自己保存才跨 call。gllm-kernels 的 save_gprs 在 native call 前保存、后恢复，覆盖全部 caller_saved。只要 save/restore LIFO 对称（callframe.inc.rs 已验证），caller_saved 跨 call 安全。

- ❌ 误判："线性扫描 regalloc 在 call 边界的行为有详尽文档" → 以为能查到权威规则
- ✅ 正解：LLVM 文档对线性扫描跨 call 的具体行为（live interval 跨 call 时的 spill vs callee-saved 决策）**无详述**。这是 architect 报告的 knowledgeGap 实质——训练数据/文档不足，须靠源码 + dump 验证，不能凭文档推断。

---

## 解决问题时参考

诊断"native call 跨层运行时值丢失"类 bug 时：

1. **先查 save_gprs 覆盖**：4 个 decode step 的 save_gprs 列表是否完整覆盖 RegAlloc 可分配池？漏一个 caller_saved → 跨 call clobber。验证方法：regalloc dump 的 PhysGpr 使用集 ⊆ save_gprs。
2. **查 block_base/lane_offset 是否 Spilled**：若分配到物理 caller_saved 且生命周期跨 call → 危险。regalloc dump 看 `MAP vNN → Gpr(PhysGpr(X))` vs `Spilled`。
3. **查 resolve_gpr_read 是否每次 reload**：helpers.inc.rs:287-318，Spilled 时 `mov scratch, [rbp+off]`，不假设 scratch 持久。
4. **查 push/restore LIFO**：callframe.inc.rs SymbolicSaveFrame，push 正序 + restore `.iter().rev()` 逆序。
5. **dump 对比**：`GLLM_REGALLOC_DEBUG=1` 跑 Q5_K_M vs Q6_K，对比 `/tmp/gllm_regalloc.log` 的 PhysGpr 使用 + spill 布局。

**关键认知**：自研 RegAlloc 不建模 call clobber，依赖 save_gprs 兜底。静态层面（VmProgram + regalloc dump）全对称 ≠ 运行时机器码执行后物理寄存器实际值正确。后者须 DAP runtime 调试（在 `call rax` 断点检查 rdi/rsi 物理值 + restore_all 后检查物理 GPR）。

---

## 已知问题 / 边界

- **knowledgeGap（未闭合）**：RegAlloc 在混合 native-call 序列下的**运行时机器码执行后物理寄存器实际值**无文档/训练数据覆盖。静态对称性已通过源码 + dump 确证，但运行时值须 DAP。
- **线性扫描 vs graph coloring 跨 call 的陷阱**：LLVM 文档无对比分析。自研 RegAlloc 用线性扫描 + 干涉图，跨 call 行为靠 save_gprs 兜底而非 register mask，若未来新增 native call 调用点且 save_gprs 未同步更新 → 新 bug。
- **SysV ABI 版本**：x86-64-abi-0.99（refspecs.linuxbase.org），稳定规范，无版本风险。
- **AVX-512 扩展**：本机 i9-10900KF 无 AVX-512（仅 16 YMM），BCE-20260702-REGALLOC-AVX2-OOB（32 ZMM OOB）不直接适用。但 RegAlloc 覆盖扩展寄存器的规则在 AVX-512 机器上仍需验证（见 x86-simd-isa.md）。

---

## 交叉引用

- `BUG-KNOWLEDGE.md` BCE-20260710-Q5_K_M（Q5_K_M 多层 E2E 乱码，14 静态方向排除）
- `docs/domain-knowledge/x86-simd-isa.md`（BCE-20260702-REGALLOC-AVX2-OOB + SysV 调用约定 caller/callee-saved）
- `docs/domain-knowledge/jit-numerical-debug-method.md`（JIT 数值诊断方法论，DAP runtime 调试）
