# 25-JIT-LIFECYCLE-INFRASTRUCTURE.md

**JIT 管线声明式生命周期管理基础设施**

> **SSOT**: 本文档是 JIT 虚拟寄存器生命周期、spill slot 管理、量化感知偏移计算、事后一致性验证的**唯一权威源**。

## §0 设计动机

### 问题诊断

Q4_0 NaN bug 的根因不是某条 VmInstr lowering 错误，而是 JIT codegen 存在四层系统性缺陷：

| 缺陷层 | 症状 | 根因 |
|--------|------|------|
| **VReg 无语义标注** | 寄存器分配器只知 VRegKind(Ptr/Vec/Scalar)，不知生命周期语义 | `LiveInterval` 只有 `def_point`/`last_use`，无 LoopInvariant/LoopCarried/BodyLocal 标签 |
| **Spill slot 无 ownership** | 顺序单调分配 `spill_offset += size`，永不回收 | `SpillSlot` 无 scope 标记，ScopeBegin/ScopeEnd 不参与 spill 回收 |
| **偏移计算 dtype 盲** | `block_size * elem` 中 `elem=dtype.elem_bytes()` 对量化格式返回 4 (F32) | `OffsetExpr` 无量化感知语义，不知道 Q4_0 的 block 内数据是 18 字节而非 32×4 |
| **无 post-hoc 验证** | NaN 在 JIT 执行后才发现，无编译期检查 | `verify.rs` 只有 5 条规则（loop pairing / def-before-use 等），无物理寄存器一致性或 dtype 偏移验证 |

### ARCH-LIFECYCLE-ROOT-CAUSE

**NaN 传播路径**: embed QuantGather 输出 NaN → input_norm 输入 NaN → 所有下游全 NaN。具体触发条件：量化偏移计算使用 compute_dtype elem_bytes 而非量化格式 block_bytes，或寄存器分配器在嵌套循环中将 loop-invariant 值覆盖。

### 解决策略

不是修单个 bug，而是构建四层声明式基础设施，从结构上消除所有此类 bug 的产生条件。

---

## §1 VReg 生命周期语义标注 (REQ-LC-001~003)

### §1.1 LifecycleTag 枚举 (REQ-LC-001)

在 `reg_alloc.rs` 的 `LiveInterval` 中新增语义标签：

```
LifecycleTag:
  | LoopInvariant     — 定义在循环外，每次迭代只读
                       例: weight_ptr, scale_ptr, output_base
  | LoopCarried       — 定义在循环外，循环内读+写，跨迭代传递
                       例: seq_counter, blk_counter, data_ptr
  | BodyLocal         — 定义在循环内，每次迭代重新覆盖
                       例: block_ptr, decoded_vec, tmp_vec
  | CrossScope        — 定义在 ScopeBegin 内，需在 ScopeEnd 后存活
                       例: GEMM 累加器跨 inner-K 和 outer-N scope
  | Global            — 全生命周期 (prologue 到 epilogue)
                       例: scratchpad_ptr, batch_ctx_ptr
```

**分配策略影响**:

| LifecycleTag | 物理寄存器策略 | Spill 策略 |
|-------------|--------------|-----------|
| LoopInvariant | 固定分配，循环内不释放 | 循环入口 spill，出口 restore |
| LoopCarried | 固定分配，跨迭代保留 | 循环入口 spill，出口 restore，循环内 spill 安全 |
| BodyLocal | 自由分配，迭代内复用 | 仅在迭代内 spill |
| CrossScope | 固定分配，跨 scope 保留 | ScopeBegin spill，ScopeEnd restore |
| Global | callee-save 或固定 GPR | prologue spill，epilogue restore |

### §1.2 自动推导规则 (REQ-LC-002)

`compute_intervals` 在现有三 pass 基础上新增 Pass 4 — 生命周期语义标注：

```
Pass 4: Lifecycle Tagging
  输入: Pass 3 的 live intervals + loop_ranges + scope_ranges

  对每个 LiveInterval:
    if def_point == 0 and last_use == program_end:
      → Global
    elif def_point 在所有 loop 外:
      if last_use 在 loop 内 and 首次 loop 内出现是 Read:
        → LoopInvariant
      elif last_use 在 loop 内 and 首次 loop 内出现是 Write:
        → BodyLocal (每迭代重定义)
    elif def_point 在 loop 外 and last_use 在 loop 内:
      if 存在循环内 Read 之前的 Write:
        → LoopCarried
      else:
        → LoopInvariant
    elif def_point 和 last_use 都在同一 loop 内:
      → BodyLocal
    elif 跨越 ScopeBegin/ScopeEnd 边界:
      → CrossScope
    else:
      → BodyLocal (默认安全)
```

**实现约束**:
- 禁止手工标注 — 全部由 `compute_intervals` 自动推导
- LifecycleTag 不改变 `LiveInterval` 的 def/last_use，只影响物理分配优先级和 spill 策略
- LoopInvariant 标签的 VReg：如果物理寄存器不足，优先 spill 到栈，保留给 BodyLocal

### §1.3 分配优先级调整 (REQ-LC-003)

当前 `allocate()` 仅有 Counter 的 no-spill 特殊政策。新增基于 LifecycleTag 的分配优先级：

```
分配优先级 (高→低):
  1. Global (callee-save 或固定分配)
  2. Counter (no-spill，已有)
  3. LoopInvariant / LoopCarried (跨迭代必须保留)
  4. CrossScope
  5. BodyLocal (自由分配)
```

**Spill 优先级** (谁先被 spill 到栈):

```
Spill 优先级 (先被 spill → 物理寄存器腾出):
  1. BodyLocal — 生命周期最短，重新加载代价最低
  2. CrossScope — 需跨 scope 但可 spill+restore
  3. LoopCarried — 跨迭代但可 spill 到循环入口
  4. LoopInvariant — spill 代价高 (循环内每次迭代都要 reload)
  5. Global — 几乎不 spill
  6. Counter — 永不 spill
```

---

## §2 基于 Scope 的 Spill Slot 管理 (REQ-LC-004~006)

### §2.1 ScopedSpillAllocator (REQ-LC-004)

替代当前的顺序单调分配 `spill_offset += size`：

```
ScopedSpillAllocator:
  slots: Vec<SpillSlotInfo>
  free_list: Vec<usize>           — 已释放 slot 索引
  scope_stack: Vec<ScopeId>       — 嵌套 scope 栈
  scope_slots: Map<ScopeId, Vec<usize>> — 每个 scope 拥有的 slot

SpillSlotInfo:
  offset: usize                   — 栈帧偏移
  size: usize                     — 字节数
  owner: Option<ScopeId>          — 所属 scope (None = global)
  vreg: Option<VRegId>            — 当前占用者
  state: SlotState                — Free / Occupied

SlotState:
  | Free        — 可分配
  | Occupied    — 被 VReg 占用
```

**分配算法**:

```
alloc(vreg, size, scope_id):
  // 优先从 free_list 找 size 完全匹配的 slot
  for idx in free_list:
    if slots[idx].size == size:
      slots[idx].state = Occupied
      slots[idx].vreg = vreg
      slots[idx].owner = scope_id
      scope_slots[scope_id].push(idx)
      free_list.remove(idx)
      return slots[idx].offset

  // 无匹配：分配新 slot
  offset = next_offset
  next_offset += size
  slots.push(SpillSlotInfo { offset, size, owner: scope_id, vreg, state: Occupied })
  scope_slots[scope_id].push(slots.len() - 1)
  return offset
```

**释放算法**:

```
scope_end(scope_id):
  for idx in scope_slots[scope_id]:
    slots[idx].state = Free
    slots[idx].vreg = None
    free_list.push(idx)
  scope_slots.remove(scope_id)
```

### §2.2 ScopeBegin/ScopeEnd 接入 (REQ-LC-005)

当前 ScopeBegin/ScopeEnd 是空操作（`referenced_vregs` 返回空 vec）。改为：

- ScopeBegin: 推入 `scope_stack`，分配 `ScopeId`，传递给 `ScopedSpillAllocator`
- ScopeEnd: 弹出 `scope_stack`，触发 `scope_end(scope_id)` 释放该 scope 的所有 spill slot

**ScopeId 分配**:
```
next_scope_id: usize = 0
// ScopeBegin: scope_stack.push(next_scope_id); next_scope_id += 1
// ScopeEnd: scope_stack.pop()
```

### §2.3 嵌套 Scope 规则 (REQ-LC-006)

- Scope 可嵌套（内层 scope 的 slot 在外层 scope 结束前不释放）
- 内层 scope 结束时，只释放内层独有的 slot，不释放外层的
- VReg 的 `LifecycleTag::CrossScope` 标注意味着它的 spill slot 不随 scope 结束释放

---

## §3 量化感知偏移计算 DSL (REQ-LC-007~009)

### §3.1 QuantLayoutExpr (REQ-LC-007)

在 `OffsetExpr` 基础上，新增量化感知偏移计算语义。不修改 `OffsetExpr` 枚举，而是在 `emit_quant_gather_trace_driven` 等函数中通过 `QuantFormatDescriptor` 自动推导偏移：

```
QuantOffsetDsl:
  // 从 QuantFormatDescriptor 推导所有偏移参数，消除手写公式

  derive_block_stride(desc: &QuantFormatDescriptor) -> usize:
    return desc.block_bytes   // Q4_0=18, Q8_0=34+2=34

  derive_data_offset(desc: &QuantFormatDescriptor, sub_block_idx: usize, lanes: usize) -> i64:
    match desc.data_layout:
      PackedNibbles { offset, .. } => offset as i64 + (sub_block_idx * (lanes / 2)) as i64
      Bytes { offset, .. } => offset as i64 + (sub_block_idx * lanes) as i64
      NibbleWithHighBits { offset, .. } => offset as i64 + (sub_block_idx * (lanes / 2)) as i64
      CodebookIndex { index_bits, .. } => offset as i64 + (sub_block_idx * ((lanes * index_bits + 7) / 8)) as i64

  derive_scale_offset(desc: &QuantFormatDescriptor) -> usize:
    match desc.scale_layout:
      BlockScalar { offset_bytes, .. } => offset_bytes
      BlockScalarWithMin { scale_offset, .. } => scale_offset
      PackedScale { .. } => 0  // packed scale 在 QuantScaleLoad VmInstr 内处理

  derive_output_stride(hidden_dim: usize, compute_dtype: QuantPrecision) -> usize:
    return hidden_dim * compute_dtype.elem_bytes()  // compute_dtype 由 TensorMeta.dtype × DeviceProfile 推导

  derive_output_block_offset(block_idx: usize, block_size: usize, sub_offset: usize, compute_dtype: QuantPrecision) -> OffsetExpr:
    // 输出缓冲区是 compute_dtype, elem_bytes 由 (模型 dtype, 硬件) 联合决定
    return Add(Mul(ScalarVReg(block_counter), block_size * compute_dtype.elem_bytes()), LoopOffset(sub_offset))
```

### §3.2 输入 vs 输出偏移分离 (REQ-LC-008)

当前 NaN 的关键混淆：**输入偏移**使用量化格式的 block_bytes (Q4_0=18)，**输出偏移**使用 compute_dtype 的 elem_bytes。两者语义不同但在代码中都使用 `block_size * elem` 计算。

**约束**:

| 偏移类型 | 计算方式 | 来源 |
|---------|---------|------|
| 输入 (量化数据) | `block_idx * block_bytes` | `QuantFormatDescriptor.block_bytes` |
| 输入 (scale) | `block_base + scale_offset_bytes` | `QuantFormatDescriptor.scale_layout` |
| 输入 (data) | `block_base + data_offset + sub_idx * data_byte_advance` | `QuantFormatDescriptor.data_layout` |
| 输出 (compute_dtype) | `block_idx * block_size * compute_dtype.elem_bytes()` | 输出缓冲区 dtype = compute_dtype |

**禁止规则**:
- 禁止用 `dtype.elem_bytes()` 计算量化数据的字节偏移
- 禁止用 `block_bytes` 计算输出缓冲区的偏移
- 量化数据的字节布局**必须**从 `QuantFormatDescriptor` 推导，不允许硬编码

### §3.3 emit_quant_gather 参数化约束 (REQ-LC-009)

`emit_quant_gather_trace_driven` 的所有偏移计算必须通过 `QuantOffsetDsl` 方法推导：

```
必须使用 QuantOffsetDsl 的位置:
  1. seq loop 的 step_bytes = 4 (固定: 每个索引 4 字节 u32)
  2. block loop 的 step_bytes = derive_block_stride(desc)
  3. sub_block loop 的 step_bytes = block_size * compute_dtype.elem_bytes() (输出行内步进)
  4. data_ptr 的初始偏移 = derive_data_offset(desc, 0, lanes)
  5. data_ptr 的步进 = derive_data_offset(desc, 1, lanes) - derive_data_offset(desc, 0, lanes)
  6. scale 加载偏移 = derive_scale_offset(desc) (在 DecodeTraceBuilder 内)
  7. 输出行步进 = derive_output_stride(hidden_dim, compute_dtype)
```

---

## §4 Post-Hoc 一致性验证 Pass (REQ-LC-010~012)

### §4.1 验证规则扩展 (REQ-LC-010)

在 `verify.rs` 的 `verify_vm_program` 中新增 4 条验证规则：

**规则 6: 物理寄存器冲突检测**

```
verify_no_physical_reg_conflict(prog, alloc):
  对每个相邻指令对 (instr[i], instr[i+1]):
    检查 instr[i] 的所有 src VReg 的物理寄存器
    是否被 instr[i+1] 的 dst VReg 覆盖
    但 src 在 i+1 仍有后续使用 (不是 dead)

  如果冲突 → Err("physical register conflict: ...")
```

**规则 7: Spill slot 一致性**

```
verify_spill_consistency(prog, alloc, frame):
  对每个 SpillSlot:
    1. offset + size 不超过 spill_area
    2. 同 scope 内两个 slot 的 [offset, offset+size) 不重叠
    3. LifecycleTag::LoopInvariant 的 VReg 在 loop 入口有 spill，出口有 reload
```

**规则 8: 量化偏移合理性**

```
verify_quant_offset_sanity(prog):
  对每个 VecLoad/VecStore 涉及量化数据的指令:
    如果 base 来自 QuantGather 的 embed_ptr 路径:
      检查 OffsetExpr 中的 Const 值是否为 block_bytes 的整数倍
      检查步进值是否与 QuantFormatDescriptor 一致

  如果不合理 → Err("quant offset sanity check failed: ...")
```

**规则 9: Loop 嵌套生命周期安全**

```
verify_loop_lifecycle_safety(prog, intervals):
  对每个 LoopBegin/LoopEnd 范围内的 VReg:
    如果 LifecycleTag == LoopInvariant:
      验证该 VReg 的物理寄存器在 loop body 内没有被其他 VReg 覆盖
    如果 LifecycleTag == LoopCarried:
      验证该 VReg 在 LoopEnd 前有写入 (确保下一次迭代有合法值)

  如果违反 → Err("loop lifecycle safety violation: ...")
```

### §4.2 编译时数值模拟 (REQ-LC-011)

新增 `verify_numerical_sanity` — 使用 `QuantFormatDescriptor` 的已知参数，在编译时模拟一个 block 的解量化：

```
verify_numerical_sanity(desc, width):
  // 构造一个已知 block: scale=1.0, data=[0,1,2,...,31], zero=bias
  // 运行 DecodeTraceBuilder 的 trace (标量模拟)
  // 验证输出:
  //   - 无 NaN
  //   - 无 Inf
  //   - 值域合理 (对于 Q4_0: (0-8)*1.0 到 (15-8)*1.0 = [-8, 7])

  如果输出异常 → Err("numerical sanity check failed: decode trace produces NaN/Inf")
```

**目的**: 在编译时就发现解量化 trace 的数学错误，而非等到运行时 JIT 产物出现 NaN。

### §4.3 分配后完整性检查 (REQ-LC-012)

在 `RegAllocator::allocate()` 完成后，自动执行：

```
post_alloc_verify(alloc, intervals, prog):
  1. 每个 VRegId 都有物理寄存器或 spill slot
  2. 两个冲突的 VReg 不在同一物理寄存器上
  3. Spill slot 无重叠
  4. LoopInvariant VReg 在循环内的所有引用点，其物理寄存器未被覆盖
  5. Counter 永远在物理寄存器上 (不 spill)
```

---

## §5 实施范围

### 文件修改清单

| 文件 | REQ | 改动 |
|------|-----|------|
| `codegen/vm/reg_alloc.rs` | LC-001~003, LC-012 | 新增 LifecycleTag + Pass 4 推导 + 分配优先级 + post_alloc_verify |
| `codegen/vm/instr.rs` | LC-005 | ScopeBegin 新增 scope_id 字段 |
| `codegen/vm/stack_frame.rs` | LC-004~006 | ScopedSpillAllocator 替代顺序分配 |
| `codegen/vm/plan_lower.rs` | LC-007~009 | QuantOffsetDsl 接入 emit_quant_gather |
| `codegen/vm/verify.rs` | LC-010~011 | 新增规则 6~9 + 数值模拟 |
| `codegen/vm/quant_decode.rs` | LC-011 | 标量模拟接口 |

### 不改动的文件

| 文件 | 理由 |
|------|------|
| `codegen/vm/x86_lower.rs` | ISA lowering 无需感知 LifecycleTag，物理分配已完成 |
| `codegen/vm/auto_select.rs` | TraceOp→VmInstr 映射与生命周期无关 |
| `quant_format.rs` | QuantFormatDescriptor 已完整，无需修改 |

---

## §6 REQ 清单

| REQ ID | 标题 | 验收标准 |
|--------|------|---------|
| REQ-LC-001 | LifecycleTag 枚举定义 | `LiveInterval` 包含 `lifecycle: LifecycleTag` 字段，5 种标签全部定义 |
| REQ-LC-002 | 自动推导 Pass 4 | `compute_intervals` 输出所有 VReg 的 LifecycleTag，零手工标注 |
| REQ-LC-003 | 分配优先级调整 | `allocate()` 按 Global > Counter > LoopInvariant > BodyLocal 优先级分配 |
| REQ-LC-004 | ScopedSpillAllocator | spill slot 在 ScopeEnd 时释放，free_list 支持复用 |
| REQ-LC-005 | ScopeBegin/End 接入 | ScopeBegin 分配 ScopeId，ScopeEnd 释放该 scope 的 spill slot |
| REQ-LC-006 | 嵌套 Scope 规则 | 内层 scope 结束不释放外层 slot，CrossScope slot 不随 scope 释放 |
| REQ-LC-007 | QuantOffsetDsl 定义 | 量化数据偏移从 QuantFormatDescriptor 自动推导，零硬编码 |
| REQ-LC-008 | 输入/输出偏移分离 | 输入偏移使用 block_bytes，输出偏移使用 compute_dtype.elem_bytes()，二者独立推导 |
| REQ-LC-009 | emit_quant_gather 参数化 | 所有偏移通过 QuantOffsetDsl 方法推导，删除手写公式 |
| REQ-LC-010 | 验证规则 6~9 | 物理寄存器冲突、spill 一致性、量化偏移、循环生命周期四条规则通过 |
| REQ-LC-011 | 编译时数值模拟 | DecodeTraceBuilder 标量模拟一个 block，输出无 NaN/Inf |
| REQ-LC-012 | 分配后完整性检查 | post_alloc_verify 检查 5 项不变量全部通过 |
