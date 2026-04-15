# SymDim 穿透协议 (ARCH-SYMDIM-THREADING)

> **SSOT 声明**: 本文档定义 SymDim 从 FusedGraph 到 JIT 机器码的完整穿透机制。
> 所有动态维度（seq_len、batch_size、total_seq）的运行时绑定必须遵循本协议。
>
> **交叉引用**: `03-GRAPH-IR.md` §3（SymDim 定义）、`01-JIT-PIPELINE.md` §6.2（编译时机）、`08-EXECUTOR.md` §4（全 JIT 执行模型）

## 1. 问题陈述

SymDim::Symbolic 在管线中途被 `as_concrete().unwrap_or(默认值)` 丢弃，导致：
- JIT 内核按编译时上界（2048）生成循环 → 实际只有 6 个 token → 写越界 → 堆损坏
- buffer 按上界分配 → 每节点 3MB × 40 节点 = 120MB 浪费

**设计目标**: SymDim 信息**完整穿透**四阶段管线，到 VmInstr 层面自然转换为 `BoundExpr::Runtime`，JIT 内核在运行时只迭代实际 seq_len 次。

## 2. 穿透路径

```
gllm: build_node_graph()
  └─ CompilerGraph 中 OpKind 的 m/seq_len 维度 = SymDim::Symbolic("seq_len", max=2048)
      │
      ▼
gllm-kernels: compile_graph() → plan_lower::lower_fusion_plan()
  ├─ Phase 1-2: SemanticDAG + FusionPlan
  │   └─ SymDim 原样保留（不 concretize）
  │
  └─ Phase 3: plan_lower → lower.rs → VmInstr
      │
      ├─ 编译时路径 (Concrete):
      │   SymDim::Concrete(384) → usize → BoundExpr::Const(384) → 硬编码到机器码
      │
      └─ 运行时路径 (Symbolic):
          SymDim::Symbolic("seq_len", max=2048)
            → SymBound { name: "seq_len", max_alloc: 2048 }  ← 新类型
            → BoundExpr::Symbolic(SymBound)                    ← BoundExpr 新变体
            → x86_lower: cmp counter, [rbp + sym_offset_map["seq_len"]]
            → 运行时只迭代实际 seq_len 次
      │
      ▼
gllm: FusedGraphExecutor::execute()
  └─ CompiledLayerFn ABI 第 7 个参数 = seq_len (运行时值)
     executor 传入 effective_seq = 实际 token 数
```

## 3. 核心数据结构

### 3.1 SymBound（新类型）

替代在 lower 函数中直接使用 `usize`，保留符号维度的**名称**和**分配上界**。

```rust
/// 符号化的循环上界。
/// 携带符号名称（用于运行时绑定查找）和编译时分配上界。
pub struct SymBound {
    /// 符号维度名称，与 SymDim::Symbolic.name 一致
    pub name: String,
    /// 编译时分配上界（buffer 按此值分配，确保不越界）
    pub max_alloc: usize,
}
```

### 3.2 BoundExpr 扩展

```rust
pub enum BoundExpr {
    /// 编译时常量 — 硬编码到机器码（展开/优化目标）
    Const(usize),
    /// 运行时值 — 从栈参数/ABI 寄存器读取（现有，低级）
    Runtime(PtrExpr),
    /// 符号化运行时上界 — 通过 SymDimSlotMap 在 prologue 绑定到物理位置
    /// 编译时用 max_alloc 分配 buffer，运行时用 SymDimSlotMap 查找实际值
    Symbolic(SymBound),
}
```

### 3.3 SymDimSlotMap（新结构）

在 `plan_lower::compile_layer()` 的 prologue 阶段构建，将符号维度名称映射到 ABI 物理位置。

```rust
/// 符号维度 → ABI 物理位置的映射。
/// plan_lower prologue 阶段构建，传递给所有 lower 函数。
pub struct SymDimSlotMap {
    slots: HashMap<String, PtrExpr>,
}
```

**预定义映射**（与 CompiledLayerFn ABI 对齐）：

| 符号名称 | ABI 位置 | 物理寄存器/栈位 | CompiledLayerFn 参数 |
|---------|----------|---------------|---------------------|
| `"seq_len"` | `PtrExpr::StackArg(16)` | `[rbp+16]` | 第 7 个参数 |
| `"batch_size"` | `PtrExpr::AbiArg(5)` | `r9` | 第 6 个参数 |
| `"total_seq"` | `PtrExpr::StackArg(16)` | `[rbp+16]`（同 seq_len） | Decoder 模式复用 |

**构建时机**: `compile_layer()` 开头，一次构建，传递给所有 lower 调用。

### 3.4 SymDim → BoundExpr 转换函数

```rust
impl SymDimSlotMap {
    /// SymDim → BoundExpr，保留完整信息。
    /// Concrete → BoundExpr::Const（编译时优化）
    /// Symbolic → BoundExpr::Symbolic（运行时绑定）
    pub fn to_bound(&self, dim: &SymDim) -> BoundExpr {
        match dim {
            SymDim::Concrete(v) => BoundExpr::Const(*v),
            SymDim::Symbolic { name, max_value } => {
                BoundExpr::Symbolic(SymBound {
                    name: name.clone(),
                    max_alloc: max_value.unwrap_or(2048),
                })
            }
        }
    }

    /// SymDim → usize（仅用于编译时 buffer 分配，使用 max_alloc 上界）
    pub fn alloc_size(&self, dim: &SymDim) -> usize {
        match dim {
            SymDim::Concrete(v) => *v,
            SymDim::Symbolic { max_value, .. } => max_value.unwrap_or(2048),
        }
    }
}
```

## 4. lower 函数接口变更

### 4.1 当前接口（断裂的）

```rust
// plan_lower 调用:
let (m, n, k) = extract_gemm_dims(anchor_op)?;  // m: usize ← Symbolic 信息丢失
emit_gemm_inline(&mut prog, m, n, k, width, ...);

// lower.rs:
pub fn lower_elementwise(elem_count: usize, ...) {
    prog.emit_loop(BoundExpr::Const(vec_count), ...);  // 硬编码
}
```

### 4.2 新接口（完整穿透）

```rust
// plan_lower 调用:
let (m_sym, n, k) = extract_gemm_dims_sym(anchor_op)?;  // m_sym: SymDim ← 保留
let m_bound = slot_map.to_bound(&m_sym);                  // → BoundExpr::Symbolic 或 Const
emit_gemm_inline(&mut prog, m_bound, n, k, width, ...);

// lower.rs:
pub fn lower_elementwise(elem_bound: BoundExpr, feature_dim: usize, ...) {
    // 外层循环（seq_len 维度）用 elem_bound（可能是 Runtime）
    // 内层循环（feature_dim 维度）总是 BoundExpr::Const（编译时已知）
    let vec_count = feature_dim / lanes;
    prog.emit_loop(elem_bound, row_bytes, |prog, _, row_off| {
        prog.emit_loop(BoundExpr::Const(vec_count), vec_step, |prog, _, col_off| {
            // 内层 vectorized 循环
        });
    });
}
```

### 4.3 受影响的 lower 函数

| 函数 | 外层维度 | 当前 | 目标 |
|------|---------|------|------|
| `lower_elementwise` | elem_count | `Const(n)` | `elem_bound: BoundExpr` |
| `lower_gemm` / `emit_gemm_inline` | M 维度 | `Const(m)` | `m_bound: BoundExpr` |
| `lower_norm` | seq_len | ✅ 已用 `Runtime(StackArg(16))` | 改用 `slot_map.to_bound(seq_dim)` |
| `lower_mha` | seq_len | `Const(seq_len)` | `seq_bound: BoundExpr` |
| `lower_gather` | seq_len | `Const(total_vecs)` | `seq_bound: BoundExpr` |
| `lower_moe_gate` | seq_len | `Const(seq_len)` | `seq_bound: BoundExpr` |

**内层维度**（hidden_size、head_dim、embed_dim 等）始终是 `Concrete` → `BoundExpr::Const`，不受影响。

## 5. x86_lower BoundExpr::Symbolic 处理

### 5.1 ISA Lower 阶段的转换

`BoundExpr::Symbolic` 在 ISA Lower 阶段通过 `SymDimSlotMap` 解析为 `PtrExpr`：

```rust
// x86_lower.rs LoopBegin 处理:
match bound {
    BoundExpr::Const(n) => {
        self.asm.cmp(counter, *n as i32)?;  // cmp rax, <imm>
    }
    BoundExpr::Runtime(ptr_expr) => {
        // 现有路径
        let addr = self.resolve_ptr_expr(ptr_expr)?;
        self.asm.cmp(counter, addr)?;  // cmp rax, [rbp+16]
    }
    BoundExpr::Symbolic(sym) => {
        // 新路径: 通过 SymDimSlotMap 查找物理位置
        let ptr_expr = self.sym_slot_map.resolve(&sym.name)
            .ok_or(CompilerError::SymDimNotBound(sym.name.clone()))?;
        let addr = self.resolve_ptr_expr(&ptr_expr)?;
        self.asm.cmp(counter, addr)?;  // cmp rax, [rbp+16]
    }
}
```

### 5.2 SymDimSlotMap 传递

`SymDimSlotMap` 在 `compile_layer()` 中构建，通过 `X86Lower::new()` 传入：

```rust
// plan_lower.rs compile_layer():
let sym_slot_map = SymDimSlotMap::default_abi();  // 预定义 seq_len→StackArg(16) 等
let mut lowerer = X86Lower::with_sym_map(use_avx512, sym_slot_map);
```

## 6. Buffer 分配协议

### 6.1 编译时分配（executor.rs）

使用 `SymDim::max_for_allocation()` 或 `SymBound::max_alloc` 分配 buffer。
保证 buffer 足够容纳最大可能的写入量。

```rust
let output_bytes = cn.output_numel * cn.output_dtype.size_bytes();
// output_numel 用 max_alloc 计算：2048 × hidden_size
let mut output_buf = vec![0u8; output_bytes];  // 3MB for seq=2048, hidden=384
```

### 6.2 运行时迭代（JIT 内核）

JIT 内核的循环只迭代 `min(runtime_seq_len, max_alloc)` 次。
通过 `BoundExpr::Symbolic` → `cmp counter, [rbp+16]` 实现。

```
实际执行: seq_len=6 → 循环 6 次 → 写入 6×384×4 = 9216 bytes
buffer 大小: 2048×384×4 = 3145728 bytes
安全裕度: 3145728 - 9216 = 3136512 bytes 未使用（但不越界）
```

### 6.3 Scratchpad 分配

同理，scratchpad 按 `max_alloc` 上界分配。
JIT 内核的中间结果只写 `runtime_seq_len` 范围。

## 7. plan_lower 改造清单

### 7.1 extract_gemm_dims → extract_gemm_dims_sym

```rust
// 当前（断裂）:
fn extract_gemm_dims(op: &CompilerOp) -> Result<(usize, usize, usize)> {
    match &op.kind {
        OpKind::Gemm { m, n, k, .. } => Ok((m.as_concrete().unwrap_or(1), *n, *k)),
    }
}

// 新（完整穿透）:
fn extract_gemm_dims_sym(op: &CompilerOp) -> Result<(SymDim, usize, usize)> {
    match &op.kind {
        OpKind::Gemm { m, n, k, .. } => Ok((m.clone(), *n, *k)),
    }
}
```

### 7.2 infer_elem_count → infer_elem_count_sym

```rust
// 当前（断裂）:
fn infer_elem_count(op: &CompilerOp, graph: &CompilerGraph) -> usize {
    tensor.shape.iter().map(|d| d.as_concrete().unwrap_or(1)).product()
}

// 新（完整穿透）:
fn infer_output_sym(op: &CompilerOp, graph: &CompilerGraph, slot_map: &SymDimSlotMap) 
    -> (BoundExpr, usize)  // (外层 bound, 内层 feature_dim)
{
    // 从输出张量形状提取：
    // 第一个 Symbolic dim → BoundExpr::Symbolic（外层循环）
    // 剩余 Concrete dims 乘积 → usize（内层维度）
    let shape = &tensor.shape;
    let outer = shape.iter().find(|d| d.is_symbolic())
        .map(|d| slot_map.to_bound(d))
        .unwrap_or(BoundExpr::Const(1));
    let inner: usize = shape.iter()
        .filter(|d| d.is_concrete())
        .map(|d| d.as_concrete().unwrap())
        .product();
    (outer, inner)
}
```

### 7.3 emit_gemm_inline 签名变更

```rust
// 当前:
fn emit_gemm_inline(prog, m: usize, n: usize, k: usize, width, ...)

// 新:
fn emit_gemm_inline(prog, m_bound: BoundExpr, n: usize, k: usize, width, ...)
```

M 维度循环用 `m_bound`，N/K 维度循环用 `Const`（始终编译时已知）。

## 8. 铁律

| 铁律 | 说明 |
|------|------|
| **ARCH-SYMDIM-NO-UNWRAP** | 禁止 `as_concrete().unwrap_or(默认值)` 丢弃 Symbolic 信息。必须通过 `SymDimSlotMap::to_bound()` 转换。 |
| **ARCH-SYMDIM-OUTER-ONLY** | 只有外层维度（seq_len, batch_size）可以是 Symbolic。内层维度（hidden, head_dim, embed_dim, K, N）始终 Concrete。 |
| **ARCH-SYMDIM-MAX-ALLOC** | Buffer 分配用 `max_alloc` 上界。JIT 运行时迭代用实际值。两者解耦。 |
| **ARCH-SYMDIM-SLOT-MAP** | 符号名 → ABI 位置的映射集中定义在 `SymDimSlotMap`。lower 函数禁止硬编码栈偏移。 |

## 9. 废弃项

| 废弃 | 替代 |
|------|------|
| `as_concrete().unwrap_or(1)` | `slot_map.to_bound(&sym_dim)` |
| `as_concrete().unwrap_or(2048)` | `slot_map.alloc_size(&sym_dim)` |
| `BoundExpr::Runtime(PtrExpr::StackArg(16))` 硬编码 | `BoundExpr::Symbolic(SymBound { name: "seq_len", .. })` |
| `extract_gemm_dims` 返回 usize | `extract_gemm_dims_sym` 返回 SymDim |
| `infer_elem_count` 返回 usize | `infer_output_sym` 返回 (BoundExpr, usize) |
