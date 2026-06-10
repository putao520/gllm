# 半VM 编译时符号执行跟踪系统 — 维度增强 SPEC

> **SSOT**: 半虚拟化 REGISTER-VM 的编译时符号执行能力完整规格。
> **原则**: 每个 BUG 都暴露一个跟踪维度缺失。补齐维度 = 系统性消灭整类 BUG。
> **关联**: `vm-state-tracking.md` (VmState ABI 跟踪), `instr.rs` (VmInstr 定义), `reg_alloc.rs` (寄存器分配)

## 0. 现状：已实现的跟踪维度

| 维度 | 实现位置 | 跟踪内容 |
|------|---------|---------|
| D0a: 寄存器活跃性 | `reg_alloc.rs` compute_intervals | VRegId → (def_point, last_use) 活跃区间 |
| D0b: 寄存器干涉 | `reg_alloc.rs` InterferenceGraph | 同类物理寄存器活跃区间重叠检测 |
| D0c: 循环变量延展 | `reg_alloc.rs` Pass 2/3 | Counter/ByteOffset 活跃到 LoopEnd + loop-invariant 检测 |
| D0d: Spill 写前读 | `reg_alloc.rs` validate_spill_safety | spilled VReg 首写位置 < 首读位置 |
| D0e: VReg 声明序 | `instr.rs` validate_declares_before_uses | DeclareVReg 在引用之前 |
| D0f: ABI 参数位置 | `vm_state.rs` VmState | 参数名 → ArgLocation(Register/Stack) 映射 |
| D0g: 栈帧布局 | `stack_frame.rs` StackFrame | callee_save + spill + scratchpad 总大小 |

**能捕获的 BUG**: 寄存器冲突复用、spill 写前读、VReg 未声明、ABI 偏移错误。

**不能捕获的 BUG**: SIGSEGV(指针无效)、类型混淆(标量当指针)、宽度不一致、内存越界、scratch 冲突。

---

## 1. 缺失维度清单

### D1: 值溯源 (Value Provenance Tracking)

**问题**: 不知道一个 VReg 在运行时到底"是什么"。VRegKind(Ptr/Vec/Scalar) 只声明了寄存器类型，没有跟踪值的来源和传播路径。

**CPU 需要知道**: 每条指令执行后，每个 GPR 里存的是什么值——是指针(哪个 buffer 的基址?)、偏移量(字节还是元素?)、计数器值、还是立即数?

**运行时等价**: CPU 硬件通过地址翻译 (CR3 → page table → physical frame) 验证每个内存访问的合法性。半VM 应该在编译时做等价验证。

**追踪内容**:
```
ValueProvenance {
    origin: ProvenanceRoot,        // 值的最初来源
    transforms: Vec<ValueTransform>, // 经过的一系列变换
    current_domain: ValueDomain,    // 当前的值域
}

ProvenanceRoot:
  - AbiLoad { name: "weight_blob_ptr" }     // 从 ABI 参数加载
  - StackLoad { offset: 24 }                 // 从栈参数加载
  - ConstImm { value: 0x1800 }              // 立即数加载
  - LoopCounter { loop_id: 0 }              // 循环计数器
  - LoopByteOffset { loop_id: 0 }           // 循环字节偏移
  - Computed { desc: "lea rax, [rbx + rcx]" } // 运行时计算结果

ValueTransform:
  - AddConst { bytes: 144 }         // 加了常量偏移
  - AddVReg { vreg: VRegId }        // 加了另一个 VReg
  - LoadFromMemory                  // 从内存加载了新值 (指针解引用)
  - IntToFloat                      // 整数转浮点
  - FloatToInt                      // 浮点转整数
  - Broadcast                       // 标量广播到向量
  - ExtractLane0                    // 提取向量 lane 0

ValueDomain:
  - Ptr { target: BufferTarget }    // 指向已知 buffer 的指针
  - ByteOffset { relative_to: VRegId } // 相对于某指针的字节偏移
  - ElemOffset { stride: usize }     // 元素偏移 (×stride = 字节偏移)
  - ScalarFloat                      // f32 标量值
  - ScalarInt                        // 整数值
  - CounterValue                     // 循环计数器值
  - VecData { lanes: usize }         // SIMD 向量数据
  - Unknown                          // 无法推导

BufferTarget:
  - InputIds
  - WeightBlob
  - Scratchpad
  - OutputBuffer
  - KVCache
  - Telemetry
  - HookContext
  - StackLocal { offset: i32 }       // 栈上临时变量
```

**能捕获的 BUG**:
- `fault_addr=0x8` — 用偏移量(0x8)当指针 → Ptr domain 不匹配
- 指针 + 偏移搞反 (base=offset, offset=ptr) → ByteOffset 被用作 VecLoad base
- 悬空指针 (已 ReleaseVReg 的指针被引用) → ProvenanceRoot 指向已释放 VReg
- 权重指针退化 (SIGSEGV 中 weight_ptr = input_ptr 副本) → 两个 VReg 有相同 ProvenanceRoot 但语义不同

### D2: 类型一致性 (Type Consistency Checking)

**问题**: VmInstr 的操作数只通过 VRegId 引用，不检查 VRegKind 是否匹配指令语义。VecLoad 的 base 可以是 Vec 类型(应该是 Ptr)，Fma 的 dst 可以是 GPR(应该是 Vec)。

**CPU 需要知道**: 每条指令的操作数类型约束。`vmovups ymm, [ptr]` 要求 ymm 是向量寄存器、ptr 是整数/指针寄存器。

**追踪内容**:
```
每条 VmInstr 定义操作数类型约束:
  VecLoad { dst: Vec, base: Ptr, offset: _, width: _ }
  VecStore { base: Ptr, offset: _, src: Vec, width: _ }
  LoadPtr { dst: Ptr, src: PtrExpr }
  Broadcast { dst: Vec, src: ScalarExpr, width: _ }
  Fma { dst: Vec, acc: Vec, a: Vec, b: Vec }
  HReduce { dst: Vec, src: Vec, op: _ }
  Accumulate { acc: Vec, src: Vec }
  LoopBegin { counter: Counter, byte_offset: ByteOffset, bound: _, step: _ }
  Argmax { dst: Scalar, logits_ptr: Ptr, vocab_bytes: _, width: _ }
  StoreToken { token_id: Scalar, output_buf: Ptr, counter: Counter, input_ids_ptr: Ptr, prompt_len_bytes: Scalar }
  AddPtr { dst: Ptr, base: Ptr, offset: _ }
  IntMulStride { dst: Ptr, src: Scalar, stride: _ }
  ScalarToIndex { dst: Ptr, src: Scalar, stride: _ }
  ScalarLoad { dst: Scalar, base: Ptr, offset: _ }
  ScalarStore { base: Ptr, src: Scalar, offset: _ }
```

**验证规则**:
1. 操作数 VRegId 的 VRegKind 必须匹配约束
2. VRegKind::Ptr ≈ VRegKind::Scalar ≈ VRegKind::Counter ≈ VRegKind::ByteOffset (都是 GPR 类)
3. 宽约束: Ptr/Scalar/Counter/ByteOffset 可以互换(都是 GPR)，但需要 WARNING
4. 窄约束: Vec 和 GPR 类绝对不能互换 → 编译错误

**能捕获的 BUG**:
- Vec 被当作 base 指针用于 VecLoad → 类型不匹配
- Ptr 被当作 Fma 输入 → 类型不匹配
- Counter 被当作 VecStore base → 可能的意图错误

### D3: 宽度一致性 (Width Consistency Checking)

**问题**: VmInstr 携带 SimdWidth，但不验证 dst/src 的 VReg width 是否一致。Fma { dst=W256, a=W256, b=W128 } 会生成错误的 vmulps ymm, ymm, xmm 指令。

**CPU 需要知道**: SIMD 指令的操作数宽度必须匹配。`vaddps ymm0, ymm1, ymm2` 三个操作数都是 256-bit。

**验证规则**:
1. 二元 VecBinOp: dst.width == a.width == b.width
2. Fma: dst.width == acc.width == a.width == b.width
3. Broadcast: dst.width == 参数 width
4. VecLoad: dst.width == 参数 width
5. HReduce: dst.width == src.width (归约结果仍以同宽度广播)
6. Accumulate: acc.width == src.width

**能捕获的 BUG**:
- AVX-512 zmm 和 AVX2 ymm 混用在同一指令 → SIGILL 或数据截断
- Scalar width 用于 VecBinOp → vmovss 用于需要 ymm 的操作

### D4: 内存访问安全 (Memory Access Validation)

**问题**: VecLoad/VecStore 的 `base + offset` 不验证是否在已知 buffer 范围内。offset 可以是任意 OffsetExpr，包括运行时值，但编译时仍可推导上界。

**CPU 需要知道**: 每个内存访问的目标地址是否在已分配的 buffer 内。硬件通过 page fault 检测越界，半VM 应在编译时静态检测。

**追踪内容**:
```
BufferDescriptor {
    name: String,              // "scratchpad", "weight_blob", ...
    base_vreg: VRegId,         // 基址寄存器
    size_bytes: usize,         // buffer 总大小
    element_size: usize,       // 元素大小 (f32=4, bf16=2)
    layout: BufferLayout,      // 行主序/列主序/分块
}

每次 VecLoad/VecStore:
  1. 解析 base VReg → 找到关联的 BufferDescriptor
  2. 估算 offset 上界 (OffsetExpr::eval_upper_bound)
  3. 验证: offset_upper_bound + access_size <= buffer_size
  4. 对齐验证: offset % alignment == 0 (vmovups 放松，但 AMX/Tensor Core 有严格对齐)
```

**能捕获的 BUG**:
- GEMM 输出写入超出 scratchpad 大小 → 段错误
- RoPE 越界访问 (seq_len > max_seq_len) → 覆盖相邻 buffer
- 权重访问超出 weight_blob 范围 → 读到垃圾数据

### D5: Scratch 寄存器冲突检测 (Scratch Conflict Detection)

**问题**: ISA Lower 使用 scratch_gprs[0..2] (rax, r10, r11) 作为临时寄存器，但多个 resolve_gpr_read 在同一 VmInstr 内可能争抢同一 scratch slot。当前通过 slot 参数区分(0/1/2)，但不验证 slot 分配的正确性。

**CPU 需要知道**: 在一条指令的 micro-op 序列中，临时寄存器不能被覆盖直到使用完毕。

**验证规则**:
```
对每条 VmInstr，跟踪 scratch 寄存器使用:

VecLoad { dst, base, offset, width }:
  slot 2 = base (resolve_gpr_read)
  slot 0/1 = eval_offset_to_rax (内部 s0/s1)
  slot 2 = base (加到 rax 后不再需要)
  → 约束: slot 2 和 slot 0/1 不冲突 ✓

Fma { dst, acc, a, b } (全 spilled):
  slot 0 = a load
  slot 1 = b load
  slot 2 = dst (acc 先 load, 写回 dst)
  → 约束: 三个 slot 互不冲突 ✓

自定义验证: 对于任意 VmInstr，跟踪 scratch slot 分配,
  如果同一 slot 被分配两次且中间没有 commit → 报错
```

**能捕获的 BUG**:
- resolve_gpr_read slot 冲突导致 base 指针被覆盖 → VecLoad 基址错误 → SIGSEGV
- eval_offset_to_rax 内部递归覆盖 scratch[1] → 偏移量计算错误

### D6: Spill 槽重叠检测 (Spill Overlap Detection)

**问题**: RegAllocator 分配 spill slots 时按顺序排列 (offset 递增)，但同一时刻多个 live VReg 的 spill slot 可能因 offset 计算错误而重叠。特别是 GPR(8B) 和 Vec(32B) 混合分配时。

**CPU 需要知道**: 每个栈帧位置的唯一拥有者。两个同时活跃的值不能驻留在重叠的内存位置。

**验证规则**:
```
1. 遍历 spill slots，验证 offset 严格递增且无重叠:
   for i in 1..spills.len():
     spills[i].offset >= spills[i-1].offset + spills[i-1].size

2. 验证同活跃区间内的 VReg spill slots 不重叠:
   for each pair (a, b) where both live at same point:
     if a.spill.offset < b.spill.offset + b.spill.size
     && b.spill.offset < a.spill.offset + a.spill.size:
       error!("spill overlap: v{} [{}, {}) vs v{} [{}, {})",
         a.id, a.offset, a.offset+a.size, b.id, b.offset, b.offset+b.size)

3. 验证 spill_offset() 和 gpr_spill_rbp_offset() 计算一致性:
   对每个 GPR spill: 两种公式给出的地址必须相同
```

**能捕获的 BUG**:
- Vec spill(32B) 覆盖相邻 GPR spill(8B) → 寄存器值损坏
- Spill 区域计算与 stack_frame 不一致 → 栈溢出

### D7: 指针算术正确性 (Pointer Arithmetic Validation)

**问题**: OffsetExpr 可以是任意嵌套的 Add/Mul/LoopOffset/Const，但不验证语义正确性。`Add(LoopOffset(counter), Const(0x1800))` 是否正确取决于 counter 的单位和 0x1800 是否是合法的字节偏移。

**CPU 需要知道**: LEA/MOV 地址计算的操作数必须是字节偏移。元素索引 × element_size = 字节偏移。

**验证规则**:
```
OffsetExpr 语义约束:
  Const(c):     c 必须是字节偏移，且 c % element_size == 0 (对齐)
  LoopOffset(v): v 是 ByteOffset VReg → 已是字节偏移 ✓
  Add(a, b):    a 和 b 都必须是字节偏移
  Mul(e, s):    e 可以是元素偏移，s = element_size → 结果是字节偏移 ✓
  ScalarVReg(v): v 的 domain 必须是 ByteOffset (不能是 ElemOffset)

Pointer 算术约束:
  AddPtr { dst, base, offset }: base 必须是 Ptr domain, offset 必须是字节常量
  VRegPlusConst(base, off): base 必须是 Ptr domain
  VRegPlusVReg(base, offset): base 必须是 Ptr, offset 必须是 ByteOffset
```

**能捕获的 BUG**:
- 元素偏移未乘 stride 直接用作字节偏移 → 越界访问
- LoopOffset 的 step_bytes 不匹配实际 stride → 地址错位
- 两层循环的偏移量加错顺序 → 跨层访问

### D8: 控制流完整性 (Control Flow Integrity)

**问题**: LoopBegin/LoopEnd 只做基本的配对检查，不验证循环语义的正确性。

**CPU 需要知道**: 每个条件跳转的目标地址是有效的指令边界，back-edge 不跳过活跃 VReg 的初始化。

**验证规则**:
```
1. LoopBegin/LoopEnd 严格嵌套 (已实现 ✓)
2. LoopBegin 的 counter 和 byte_offset 在 LoopEnd 之前不被覆盖写
   (除非是 LoopEnd 自动的 counter++/byte_offset += step)
3. BreakLoop 只出现在循环体内
4. ConditionalSkip skip_count 精确匹配跳过的指令数
5. OutputModeDispatch 的 paths 覆盖所有声明的模式
6. ScopeBegin/ScopeEnd 内不包含跨越边界的 VReg 引用
```

**能捕获的 BUG**:
- LoopEnd 被错误跳过 → 无限循环
- ConditionalSkip skip_count 偏差 → 跳到指令中间
- BreakLoop 在循环外 → 跳到随机地址

### D9: 数值域追踪 (Numerical Domain Tracking)

**问题**: 不跟踪 VReg 的数值范围。counter 从 0 开始递增到 bound，byte_offset = counter × step_bytes，但编译时不知道 counter 当前值是否可能导致溢出。

**CPU 需要知道**: 整数溢出导致地址回绕 → 访问到非法内存。

**追踪内容**:
```
NumericalDomain:
  Range { min: usize, max: usize }       // 确知范围
  NonNegative                            // ≥ 0 (计数器)
  ByteAligned { alignment: usize }       // 对齐约束
  PowerOfTwo                             // 2 的幂 (stride, tile size)

对每个循环:
  counter ∈ [0, bound.max_alloc)
  byte_offset = counter × step_bytes ∈ [0, bound.max_alloc × step_bytes)
  验证: byte_offset.max + access_size ≤ buffer_size

对每个算术操作:
  Add: result.range = [a.min + b.min, a.max + b.max]
  Mul: result.range = [a.min * b.min, a.max * b.max]
  溢出检测: result.max > i64::MAX → 报错
```

**能捕获的 BUG**:
- GEMM 大矩阵的偏移量溢出 32-bit → 地址截断 → 段错误
- 循环 bound × step_bytes 超过 buffer 大小 → 越界访问

### D10: ABI 契约完整验证 (ABI Contract Validation)

**问题**: VmState 追踪了参数位置，但不验证:
1. 所有需要的 ABI 参数是否已被加载
2. Callee-saved 寄存器是否在 epilogue 正确恢复
3. 栈对齐在 CALL 指令前是否满足 16B 约束
4. 返回值(rax)是否在 BreakLoop/epilogue 前被设置

**CPU 需要知道**: SysV ABI 的完整契约——哪些寄存器必须保存、栈如何对齐、参数如何传递。

**验证规则**:
```
1. ABI 参数使用完整性:
   对 MegaKernelFn 的 17 个参数，验证每个都被 LoadPtr 引用过
   (未使用的参数 = 可能的遗漏)

2. Callee-saved 恢复对称性:
   prologue push 的 callee-saved 序列 = epilogue pop 的逆序列
   顺序不一致 → 寄存器值互换 → 指针损坏

3. 栈对齐:
   在每个 CALL 指令前 (如果有), RSP 必须 16B 对齐
   prologue 后: RSP = RBP - frame_size
   frame_size 必须是 16 的倍数 (已在 stack_frame.rs 实现 ✓)

4. 返回值设置:
   在 epilogue 前, rax 必须被赋值 (通过 BreakLoop { return_value } 或显式 mov)

5. SysV 入参保存区完整性:
   abi_save_top_off 覆盖所有 6 个 SysV 寄存器
   LoadPtr { AbiArg(idx) } 的 idx ∈ 0..=5
```

**能捕获的 BUG**:
- callee-saved push/pop 不对称 → 寄存器值错乱
- 栈参数偏移 off-by-one → 读取错误参数
- 忘记设置返回值 → 返回垃圾值

---

## 2. 优先级排序

按"捕获 BUG 数量 × 实现难度"排序:

| 优先级 | 维度 | 预期捕获 BUG | 实现难度 | 依赖 |
|--------|------|-------------|---------|------|
| P0 | D2 类型一致性 | 3-5 | ★☆☆ 低 | 无 |
| P0 | D3 宽度一致性 | 2-3 | ★☆☆ 低 | 无 |
| P0 | D6 Spill 重叠 | 1-2 | ★☆☆ 低 | 无 |
| P1 | D1 值溯源 | 5-10 | ★★★ 高 | D2 |
| P1 | D5 Scratch 冲突 | 2-3 | ★★☆ 中 | 无 |
| P1 | D7 指针算术 | 2-3 | ★★☆ 中 | D1 |
| P2 | D4 内存安全 | 3-5 | ★★★ 高 | D1 |
| P2 | D9 数值域 | 1-2 | ★★☆ 中 | D1 |
| P2 | D8 控制流 | 1-2 | ★★☆ 中 | 无 |
| P2 | D10 ABI 契约 | 1-2 | ★☆☆ 低 | 无 |

---

## 3. 实施路线图

### Phase 1: 零依赖快速增益 (P0 — 立即实施)

**D2 + D3 + D6**: 三个维度互不依赖，全部在 `VmProgram` 上做静态分析 pass，不修改任何运行时代码。

实现方式:
1. 新增 `VmProgram::validate_type_consistency()` — 遍历所有 VmInstr，检查操作数 VRegKind
2. 新增 `VmProgram::validate_width_consistency()` — 遍历所有 VmInstr，检查 SimdWidth
3. 扩展 `RegAllocation::validate_spill_safety()` — 增加 spill 槽重叠检测

插入位置: `compile()` 的 RegAlloc 之后、StackFrame 之前。

### Phase 2: 值溯源核心 (P1 — 核心能力)

**D1**: 构建编译时值溯源图 (Value Provenance Graph)。

实现方式:
1. 新增 `ValueProvenanceTracker` struct — 维护 VRegId → ValueProvenance 映射
2. 遍历 VmProgram，对每条 VmInstr 更新值溯源:
   - LoadPtr: dst 的 origin = PtrExpr 解析的 ABI 参数
   - VecLoad: 验证 base 是 Ptr domain
   - AddPtr: dst 的 origin = base 的 origin, transform += AddConst(offset)
   - Fma: dst 的 domain = VecData
3. 在 VecLoad/VecStore 遇到 base 非 Ptr domain 时报错
4. 在 VecLoad 遇到 base = ProvenanceRoot::ConstImm{0} 时报错 (null pointer)

### Phase 3: 指针安全与算术验证 (P1 — 依赖 D1)

**D7 + D5**: 基于 D1 的值溯源信息，验证指针运算和 scratch 使用。

### Phase 4: 完整内存安全 (P2 — 依赖 D1)

**D4 + D9**: 基于 D1 的 buffer 归属信息，做边界检查和数值域分析。

### Phase 5: 控制流与 ABI (P2 — 独立)

**D8 + D10**: 不依赖 D1，可以并行实施。

---

## 4. 预期收益

完成所有 10 个维度后，半VM 编译时应能自动捕获:

- **所有 SIGSEGV**: null pointer、偏移量当指针、越界访问、spill 损坏
- **所有 SIGILL**: 类型混淆、宽度不匹配、非法指令编码
- **所有数值错误**: 整数溢出、精度丢失、对齐违反
- **所有控制流错误**: 无限循环、跳转错位、返回值缺失

推理管线从"编译通过但运行时崩溃"变为"编译时捕获所有静态可检测错误"。
