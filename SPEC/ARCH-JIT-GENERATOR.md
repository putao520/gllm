# JIT 代码生成器状态机 — 抽象架构设计

> **SSOT**: 本文档定义 JIT 代码生成器的层次关系、接口边界和数据流。
> 所有 ISV (x86_64/aarch64/PTX/HIP/Metal) 的代码生成器必须符合此架构。

## 1. 问题定义

当前 JIT codegen 是**无状态的**：每个 emit 函数独立发射汇编指令，不知道：
- 寄存器当前持有什么值
- scratchpad 已经分配了多少
- 上一个 emit 是否 clobber 了某个寄存器
- buffer offset 是否越界

导致的系统性 bug：
- 寄存器语义冲突（r14 同时被用作 weight_base 和 ic_counter）
- scratchpad 越界（BLIS offset 超出分配）
- 跨 kernel 状态泄漏（bias_saved 全局布尔）
- 偏移手算错误（hidden * 4 而非 layout.row_stride()）

## 2. 四层架构

```
┌─────────────────────────────────────────────────┐
│  Layer 4: Algorithm Emitters                     │
│  emit_blis_gemm / emit_rope / emit_mha / ...   │
│  · 接收 Layout 对象                              │
│  · 通过 Layer 3 发射指令                         │
│  · 声明 EmitContract (precondition/postcondition)│
└──────────────────────┬──────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────┐
│  Layer 3: Instruction Emitter                    │
│  emit_load_ptr / emit_store / emit_loop_header  │
│  · 接收 (Reg, PtrSource) 对                     │
│  · 发射 asm 指令到 CodeAssembler                │
│  · 自动更新 KernelState.ctx                     │
│  · 自动验证 OffsetValidator                     │
└──────────────────────┬──────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────┐
│  Layer 2: State Machine (KernelState)            │
│  ctx: CodegenContext (寄存器绑定)                 │
│  scratchpad: ScratchpadAllocator (buffer 分配)   │
│  stack: StackLayout (栈偏移推导)                  │
│  · per-kernel 生命周期                           │
│  · 每条指令后自动更新                            │
│  · final_audit() 最终验证                        │
└──────────────────────┬──────────────────────────┘
                       │ 查询
┌──────────────────────▼──────────────────────────┐
│  Layer 1: Layout Algebra (只读计算层)             │
│  GemmLayout / RopeLayout / MhaLayout / ...       │
│  · 从 (shape, dtype) 推导 stride/offset/size    │
│  · 纯函数，无副作用                              │
│  · 所有 byte offset 的唯一计算源                 │
└─────────────────────────────────────────────────┘
```

## 3. 数据流

```
CompilerGraph + FusionPlan + BufferAllocation
        │
        ▼
   emit_plan() 入口
        │
        ├─ 创建 KernelState::new(alloc.total_bytes, has_weight_layout)
        │   └─ 初始化 ctx, scratchpad, bias_slot
        │
        ├─ emit_prologue()
        │   └─ 发射 push rbp / push r12-r15 / sub rsp
        │      通过 StackLayout::compute() 确定 sub rsp 大小
        │
        ├─ for group in plan.groups:
        │   │
        │   ├─ compute_group_pointer_map(group, graph, alloc)
        │   │   └─ 返回 [(Reg, PtrSource)] 列表
        │   │
        │   ├─ for (reg, source) in pointer_map:
        │   │   └─ emit_load_ptr(reg, source)    ← Layer 3
        │   │       ├─ source.stack_slot() → asm.mov(reg, [rbp+slot])
        │   │       ├─ source.offset() > 0 → asm.add(reg, offset)
        │   │       └─ state.ctx.set(reg, source.to_binding())
        │   │
        │   ├─ save_callee_saved()
        │   │   └─ state.ctx.save_callee_saved() + asm.push(r13/r14)
        │   │
        │   ├─ emit_group(group, ...)            ← Layer 4
        │   │   ├─ contract = contract_for_op(op_kind)
        │   │   ├─ state.ctx.apply_contract(&contract)  ← precondition 验证
        │   │   ├─ layout = Layout::new(shape, dtype)    ← Layer 1
        │   │   ├─ ... emit algorithm instructions ...
        │   │   └─ state.ctx.apply_contract_post(...)    ← postcondition 更新
        │   │
        │   └─ restore_callee_saved()
        │       └─ asm.pop(r14/r13) + state.ctx.restore_callee_saved()
        │
        ├─ state.final_audit()
        │   └─ assert_no_violations() + scratchpad 交叉验证
        │
        └─ emit_epilogue()
```

## 4. 接口定义

### 4.1 Layer 3: Instruction Emitter（ISV 必须实现）

```rust
trait InstructionEmitter {
    /// 加载指针到寄存器。自动更新 state。
    fn emit_load_ptr(&mut self, reg: Reg, source: PtrSource, state: &mut KernelState) -> Result<()>;

    /// 保存 callee-saved 寄存器。自动更新 state。
    fn emit_save_callee_saved(&mut self, state: &mut KernelState) -> Result<CalleeSavedSnapshot>;

    /// 恢复 callee-saved 寄存器。自动更新 state。
    fn emit_restore_callee_saved(&mut self, snap: &CalleeSavedSnapshot, state: &mut KernelState) -> Result<()>;

    /// 发射 prologue。初始化 state。
    fn emit_prologue(&mut self, state: &mut KernelState) -> Result<()>;

    /// 发射 epilogue。
    fn emit_epilogue(&mut self) -> Result<()>;
}
```

### 4.2 Layer 4: Algorithm Emitter（ISV 必须实现）

```rust
trait AlgorithmEmitter: InstructionEmitter {
    /// BLIS GEMM。layout 提供所有维度和 offset。
    fn emit_blis_gemm(&mut self, layout: &GemmLayout, state: &mut KernelState) -> Result<()>;

    /// RoPE 旋转位置编码。
    fn emit_rope(&mut self, layout: &RopeLayout, state: &mut KernelState) -> Result<()>;

    /// Multi-Head Attention。
    fn emit_mha(&mut self, layout: &MhaLayout, scratch: &MhaScratchpad, state: &mut KernelState) -> Result<()>;

    /// RmsNorm / LayerNorm。
    fn emit_norm(&mut self, layout: &NormLayout, state: &mut KernelState) -> Result<()>;

    /// Elementwise (Add/Mul/SiLU/SwiGLU/...)。
    fn emit_elementwise(&mut self, layout: &ElementwiseLayout, trace: &[TraceOp], state: &mut KernelState) -> Result<()>;

    /// MeanPool。
    fn emit_mean_pool(&mut self, layout: &TensorStride, state: &mut KernelState) -> Result<()>;

    /// pack_a / pack_b (BLIS 内部使用)。
    fn emit_pack_a(&mut self, layout: &GemmLayout, mc: usize, kc: usize, state: &mut KernelState) -> Result<()>;
    fn emit_pack_b(&mut self, layout: &GemmLayout, kc: usize, nc: usize, state: &mut KernelState) -> Result<()>;
}
```

### 4.3 Layer 2: KernelState（共享，ISV 无关）

```rust
struct KernelState {
    ctx: CodegenContext,
    scratchpad: ScratchpadAllocator,
    has_bias_slot: bool,
    has_weight_layout: bool,
}
```

### 4.4 Layer 1: Layout Algebra（共享，ISV 无关）

所有 Layout struct 是纯计算类型，只有方法没有状态：
- `GemmLayout::new(m, n, k, dtype)` → strides, pack sizes
- `RopeLayout::new(num_heads, head_dim, dtype)` → pair offsets, token stride
- `MhaLayout::new(seq, heads, kv_heads, head_dim, dtype)` → q/k/v strides, v_offset
- `NormLayout::new(feature_dim, simd_width, dtype)` → vec_count, tail, row_bytes
- `ElementwiseLayout::new(elem_count, simd_width, dtype)` → simd_bytes, unroll stride

## 5. 状态隔离规则

| 状态 | 生命周期 | 存储位置 | 跨 kernel 行为 |
|------|---------|---------|---------------|
| 寄存器绑定 | per-kernel | `KernelState.ctx` | 重置 |
| scratchpad 水位 | per-kernel | `KernelState.scratchpad` | 重置 |
| bias_slot 写入标记 | per-kernel | `KernelState.has_bias_slot` | 重置 |
| DeviceProfile | per-session | `X86CodeGen` field | 不变 |
| ISA flags (AVX2/512) | per-session | `X86CodeGen` field | 不变 |
| const_pool | per-kernel | `X86CodeGen` (待迁移到 KernelState) | 需重置 |

## 6. 迁移路径

### Phase 1（当前完成）
- StackLayout: 栈偏移全部计算推导
- Layout Algebra: 7 个 Layout struct 覆盖所有算法
- EmitContract: 全 op + 全 fusion mode
- CodegenContext: 寄存器追踪 + contract 验证
- KernelState: per-kernel 状态容器

### Phase 2（下一步）
- 在 emit_plan 中创建 KernelState，替代 self.bias_saved/blis_scratchpad_offset
- 实现 emit_load_ptr 统一指令发射
- BLIS GEMM 的 pack_a/pack_b offset 通过 ScratchpadAllocator 分配

### Phase 3（最终形态）
- 所有 emit_* 函数改签名接收 &mut KernelState
- 禁止 self.asm.mov() 直接调用（通过 InstructionEmitter trait 方法）
- const_pool 迁移到 KernelState
- aarch64 后端实现相同的 trait
