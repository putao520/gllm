# ARCH-DATA-FLOW-CONTRACT — 数据流唯一来源契约

> **铁律：每个值只有一个来源。禁止独立计算、反推、硬编码任何已存在于数据源中的值。**
>
> **例外（行业标准默认值）**：在 `auto_graph` 构建器中，从模型配置读取参数时，
> 属性可能不存在（格式差异）。此时可使用 `unwrap_or(行业标准值)` 并标注 `// LEGAL: reason`。
> 此规则仅限图优化器层面，**lower/executor 层禁止使用默认值**。
>
> **例外（GGUF 可选元数据）**：GGUF 格式的可选字段在 `model_config.rs` 中读取时，
> 缺失的字段用行业标准默认值填充（如 `rope_theta=10000.0`, `num_kv_heads=num_heads`）。
> 标注 `// LEGAL: GGUF optional field` 后合法。
>
> **例外（硬件探测 fallback）**：CPU 核心数、NUMA 拓扑等系统信息读取失败时，
> 用安全下界（如 1 核、1 NUMA node）。标注 `// LEGAL: hw detection fallback` 后合法。

## §1 数据源层级

```
ModelGeometry (SSOT)
  ↓ Arc<> 共享
GeneratorForwardConfig
  ↓ 构建
CompilerGraph { ops: [CompilerOp], tensors: [TensorMeta] }
  ↓ plan_lower
VmProgram { instrs: [VmInstr] }
  ↓ ISA lower
CompiledLayer { code, scratchpad_bytes, weight_layout }
  ↓ 执行
CompiledNode { feature_dim, per_output_feature_dims, output_dtype, ... }
```

## §2 lower 函数数据来源契约

> **表格读取说明**：
> - **"唯一来源" 列** 是合法数据源（代码必须使用）
> - **"❌ 禁止" 列** 是**被禁止的反模式**（代码出现即视为 SPEC 违反，审计工具应标红）
> - 本节所有条目均为 "禁止列" 项，不是可接受的备选项

### §2.1 维度参数

| 消费点 | 值 | **唯一来源** | ❌ 禁止 |
|--------|-----|-------------|---------|
| GEMM m | `SymDim` | `OpKind::Gemm { m }` | 从张量 shape 反推 |
| GEMM n, k | `usize` | `OpKind::Gemm { n, k }` | 手动计算 |
| MHA num_heads | `usize` | `OpKind::MultiHeadAttention { num_heads }` | `infer_num_heads()` 反推 |
| MHA num_kv_heads | `usize` | `OpKind::MultiHeadAttention { num_kv_heads }` | 硬编码 |
| MHA head_dim | `usize` | `OpKind::MultiHeadAttention { head_dim }` | 硬编码 |
| MHA causal | `bool` | `OpKind::MultiHeadAttention { causal }` | 硬编码 true |
| RoPE num_heads | `usize` | `OpKind::RoPE { num_heads }` | 从张量 shape 反推 |
| RoPE head_dim | `usize` | `OpKind::RoPE { head_dim }` | 硬编码 |
| RoPE theta | `f64` | `OpKind::RoPE { theta }` | 硬编码 10000.0 |
| RoPE partial | `f32` | `OpKind::RoPE { partial }` | 硬编码 1.0 |
| Norm eps | `f32` | `OpKind::RmsNorm { eps }` / `OpKind::LayerNorm { eps }` | 硬编码 1e-5 |
| Norm feature_dim | `usize` | 输出张量 Concrete 维度之积 | 硬编码 |
| MoE num_experts | `usize` | `OpKind::MoEGate { num_experts }` | 硬编码 |
| MoE top_k | `usize` | `OpKind::MoEGate { top_k }` | 硬编码 2 |
| Gather embed_dim | `usize` | `OpKind::Gather { embed_dim }` | 硬编码 |
| MeanPool seq_len | `usize` | `OpKind::MeanPool { seq_len }` | 硬编码 |
| MeanPool hidden | `usize` | `OpKind::MeanPool { hidden }` | 硬编码 |
| QuantGemm bits | `usize` | `OpKind::QuantGemm { bits }` | 硬编码 |
| QuantGemm block_size | `usize` | `OpKind::QuantGemm { block_size }` | 硬编码 |
| QuantGemm scale 偏移 | `usize` | `OpKind::QuantGemm { scale_offset }` (来自 WeightLayout 元数据) | `k * n * bits / 8` 公式推导 |
| QuantGemm zero-point 偏移 | `Option<usize>` | `OpKind::QuantGemm { zero_point_offset }` | 硬编码或从 scale_offset 推导 |

### §2.2 元素字节数

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| JIT 计算路径 elem_bytes | `op_input_dtype(op, graph).elem_bytes()` — 从 tensor 元数据自动推断 | `computation_elem_bytes()` (硬编码 F32), 裸 `4`, `4usize`, `* 4` |
| JIT 计算 elem_bytes (无 op 上下文) | `graph.tensor(graph.inputs[0]).dtype.to_quant_precision().elem_bytes()` | `size_of::<f32>()` |
| Tile 步长 elem_bytes | `TileConfig.dtype.size_bytes()` | `* 4` |
| executor 输出 elem_bytes | `cn.output_dtype.size_bytes()` | `4usize`, `size_of::<f32>()` |
| KV cache elem_bytes | `DType::F32.size_bytes()` (TurboQuant: KV 统一 F32) | `size_of::<f32>()`, `4` |

### §2.3 行步长 (stride)

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| 张量行步长 | `inner_dim * op_input_dtype(op, graph).elem_bytes()` — dtype 从 tensor 元数据推断 | `row_stride_bytes(inner_dim)` (硬编码 F32), 手动 `k * elem`, `n * 4` |
| MHA Q stride | `num_heads * head_dim * dtype.elem_bytes()` — dtype 从 OpKind 上下文推断 | `num_heads * head_dim * 4` |
| MHA K stride | `num_kv_heads * head_dim * dtype.elem_bytes()` — dtype 从 OpKind 上下文推断 | 手动计算 |

### §2.4 循环边界

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| seq_len (运行时) | `BoundExpr::Symbolic(SymBound)` via `sym_map.to_bound(&sym_dim)` | `BoundExpr::Const(512)` |
| num_heads (编译时) | `BoundExpr::Const(num_heads)` — 从 OpKind 读取 | Rust `for h in 0..num_heads` 展开 |
| head_dim 向量化 | `BoundExpr::Const(hd_vecs)` — 从 head_dim / lanes 计算 | 硬编码 |
| feature_dim 向量化 | `BoundExpr::Const(feature_vecs)` — 从输出张量 Concrete 维度计算 | 硬编码 |
| causal attention ki 上界 | `BoundExpr::DynamicVRegPlusOne(qi_ctr)` — 绑定到外层 qi 计数器 (ARCH-CAUSAL) | 外层手动遮罩或硬编码上界 |

### §2.5 ABI 参数访问 (ARCH-VM-QUERY-NOT-ASSUME)

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| lower 阶段访问参数 | `sym_map.resolve("name")` → 具体 PtrExpr | 手动构造 `PtrExpr::StackArg(40)` |
| opt_pass 注入参数 | `PtrExpr::NamedArg("name")`（ISA Lower 时通过 SymDimSlotMap 解析） | 硬编码 `StackArg(N)` |
| ISA Lower 解析 NamedArg | `self.sym_slot_map.resolve(name)` → Err 若找不到 | 静默兜底到 rax/arg0 |

### §2.5 IsaHook（硬件策略）

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| FMA 策略 | `hook.select_fma(m, n, k)` — hook 必须存在 | `unwrap_or(Fma3)` |
| 微内核形状 | `hook.gemm_microkernel_shape()` — hook 必须存在 | `unwrap_or((6, 2))` |
| Attention 策略 | `hook.select_attention(seq, hd)` — hook 必须存在 | `unwrap_or(Naive)` |
| MoE dispatch | `hook.moe_dispatch(experts)` | 硬编码 |
| Tile config dtype | `TileConfig.dtype` / `WgmmaConfig.input_dtype` | `unwrap_or(BF16)` |

## §3 executor 数据来源契约

### §3.1 CompiledNode 字段

| 字段 | **唯一来源** | ❌ 禁止 |
|------|-------------|---------|
| `feature_dim` | `extract_tensor_feature_dim(graph, out_tid)` — 输出张量 Concrete 维度之积 | `output_numel / SYMDIM_MAX_SEQ_LEN` |
| `per_output_feature_dims` | 每个输出张量的 Concrete 维度之积 | `per_output_numel[i] / SYMDIM_MAX_SEQ_LEN` |
| `output_numel` | `SYMDIM_MAX_SEQ_LEN * feature_dim` — 仅用于 buffer 分配 | 用于计算 feature_dim（禁止反推） |
| `output_dtype` | `graph.tensors[out_tid].dtype` | 硬编码 `DType::F32` |

### §3.2 运行时值

| 消费点 | **唯一来源** | ❌ 禁止 |
|--------|-------------|---------|
| seq_len | `shape_bindings["seq_len"]` — 必须存在 | `unwrap_or(1)`, `unwrap_or(&1)` |
| layer_idx | `extract_layer_index(node_name)` 从节点名解析 | `node_idx / 2` |
| hidden_state (callback) | `tensors[cn.graph_input_names[0]]` 命名查找 | `tensors.values().next()` |
| cache_max_seq | `forward_config.max_seq_len()` — forward_config 必须存在 | `unwrap_or(SYMDIM_MAX_SEQ_LEN)` |
| elem_bytes (GQA expand) | `cn.output_dtype.size_bytes()` | `4usize` |
| elem_bytes (scatter) | `cn.output_dtype.size_bytes()` | `4usize` |
| elem_bytes (KV cache) | `DType::F32.size_bytes()` | `size_of::<f32>()` |

### §3.3 SYMDIM_MAX_SEQ_LEN 使用范围

| ✅ 合法 | ❌ 禁止 |
|---------|---------|
| `SymDim::Symbolic { max_value: Some(SYMDIM_MAX_SEQ_LEN) }` | `output_numel / SYMDIM_MAX_SEQ_LEN` |
| `output_numel = SYMDIM_MAX_SEQ_LEN * feature_dim` (buffer 分配) | `per_token = numel / SYMDIM_MAX_SEQ_LEN` |
| `alloc_rows = SYMDIM_MAX_SEQ_LEN` (KV cache buffer 分配) | 任何除法/取模运算 |

## §4 emit_loop 铁律 (ARCH-NO-LOOP-UNROLL)

| 维度 | 循环方式 | 原因 |
|------|---------|------|
| seq_len | `emit_loop(BoundExpr::Symbolic)` | 运行时维度 |
| num_heads | `emit_loop(BoundExpr::Const(num_heads))` | 可能很大(32/64/128)，展开爆炸 |
| head_dim/lanes | `emit_loop(BoundExpr::Const(hd_vecs))` | 编译时常量，适度大小 |
| k (GEMM) | `emit_loop(BoundExpr::Const(k_tiles))` | 编译时常量 |
| tail 元素 (< SIMD width) | `for t in 0..tail` Rust 展开 | 编译时常量且极小 (< lanes) |
| BLIS 微内核 mr/nr | `for r in 0..mr` Rust 展开 | mr ≤ 6, nr_vecs ≤ 4, 寄存器级完全展开是 BLIS 算法本质 |
| BLIS pack 向量数 | `for v in 0..(dim/lanes).min(MAX)` Rust 展开 | MAX_NR_VECS=8, MAX_MR_VECS=4, pack buffer 向量复制 |
| MoE top_k 选择 | `for _ki in 0..top_k` Rust 展开 | top_k ≤ 8 且是模型架构常量 |

**禁止**：`for h in 0..num_heads { prog.emit(...) }` — 生成 O(num_heads) 条 VmInstr。
**禁止**：`for i in 0..m { prog.emit(...) }` — M 是矩阵维度，可能很大。
**禁止**：`for p in 0..k { prog.emit(...) }` — K 是矩阵维度，可能很大。
**合法展开上界**：编译时常量且 ≤ MAX_NR_VECS (8)。超过此上界必须 emit_loop。

## §5 MegaKernelFn ABI（唯一入口，详见 08-EXECUTOR.md §4.1.3）

> **已物理删除** (SPEC/39): `CompiledLayerFn` 单节点 ABI。所有编译产物统一使用 `MegaKernelFn` ABI。编译器不假设图结构——喂什么编译什么。

```

arg[0] rdi    = input_ids_ptr  (prompt token ID 数组)
arg[1] rsi    = weight_blob_ptr (全部权重连续打包)
arg[2] rdx    = kv_cache_ptr
arg[3] rcx    = positions_ptr
arg[4] r8     = aux_ptr        (KV-V half 指针)
arg[5] r9     = batch_size
arg[6] [rsp+0]  = prompt_len    ← SymDim 运行时绑定点
arg[7] [rsp+8]  = scratchpad_ptr
arg[8] [rsp+16] = output_tokens_ptr
arg[9] [rsp+24] = temperature   ← 采样参数（运行时传入）
arg[10] [rsp+28] = top_k
arg[11] [rsp+32] = top_p
arg[12] [rsp+36] = max_new_tokens
arg[13] [rsp+40] = eos_token_id
arg[14] [rsp+48] = hook_ctx_ptr
arg[15] [rsp+56] = telemetry_ptr
→ rax: 实际生成的 token 数
```

**数据来源**：`SymDimSlotMap::mega_kernel_abi()` 将 `"seq_len"` 映射到 `PtrExpr::StackArg(16)`。
lower 函数通过 `sym_map.to_bound(&sym_dim)` 获取 `BoundExpr::Symbolic(SymBound)`，
x86_lower 将其翻译为 `mov rax, [rbp+16]`。

**禁止**：手动构造 `PtrExpr::StackArg(16)` 或 `BoundExpr::Symbolic(SymBound { name: "seq_len", ... })`。
必须通过 `sym_map` 查询。

## §6 Causal Attention VM 扩展 (ARCH-CAUSAL)

Causal attention 的关键约束 `ki ≤ qi` 通过 VM 指令集扩展实现，不使用运行时条件遮罩：

**BoundExpr 扩展**：
- `BoundExpr::DynamicVReg(VRegId)` — 循环上界绑定到外层循环计数器 VReg（`counter < vreg_value`）
- `BoundExpr::DynamicVRegPlusOne(VRegId)` — `counter < vreg_value + 1`（即 `counter ≤ vreg_value`）

**lower_mha_with_hook 中 causal 路径**：
```rust
let ki_bound = if causal {
    BoundExpr::DynamicVRegPlusOne(qi_ctr)  // ki ≤ qi
} else {
    seq_bound.clone()                       // ki < seq_len
};
prog.emit_loop(ki_bound, k_stride, |prog, _ki_ctr, ki_off| { ... });
```

**x86_lower 生成**：`lea tmp, [qi_reg + 1]; cmp ki_reg, tmp; jge done`

**禁止**：
- 使用 ScalarCmp + ConditionalMask 模拟遮罩（会引入额外指令开销）
- 在 ki 循环体内运行时检查 `if ki > qi continue`（破坏向量化）

## §7 TraceOp Input 索引契约 (NO_SILENT_FALLBACK)

TraceOp::Input(n) 必须映射到显式提供的 VReg 数组，越界必须报错：

```rust
fn lower_trace_body(inputs: &[VRegId], ...) -> Result<(), CompilerError> {
    match op {
        TraceOp::Input(n) => *inputs.get(*n as usize).ok_or_else(|| ...)?,
        ...
    }
}
```

**禁止**：
- `TraceOp::Input(1) => secondary.unwrap_or(primary)` — 静默兜底
- `TraceOp::Input(n) if n >= 2 => primary` — 硬编码兜底
- 调用方对 `is_binary` 硬编码 true/false，必须从 `op.inputs.len() > 1` 派生

## §8 GEMM Epilogue 通用性 (ARCH-GEMM-EPILOGUE)

GEMM epilogue 必须支持任意元数 (unary/binary/N-ary) 算子，通过 `EpilogueInputSource` 显式提供 `Input(1+)` 数据源：

```rust
pub enum EpilogueInputSource {
    BroadcastN(VRegId),                          // [n] 广播向量 (bias 布局)
    FullMatrix { ptr: VRegId, row_stride_bytes: usize },  // [m,n] 完整矩阵 (residual)
}
```

**lower_gemm 签名**：`(m, n, k, width, epilogue, aux_inputs: &[EpilogueInputSource], hook)`

**载入时机**：在 K 归约完成后、epilogue 执行前，根据每个 aux 的 layout 从指定位置 VecLoad 向量。

**调用方（plan_lower）的职责**：从图中 Add/Residual op 的 `inputs[1]` 张量推导 `EpilogueInputSource`，绝不传入 `None` 然后让 Input(1) 静默 fallback 到 primary。

**禁止**：
- `lower_trace_body_compat(prog, epi, acc, None, width)` 用于引用 Input(1+) 的 epilogue
- 硬编码 `is_binary: true/false`（应从 `op.inputs.len() > 1` 派生）

## §9 FFN Block Fusion 校验 (ARCH-FFN-SHAPE)

FFNBlock 融合要求 Gate/Up GEMM 形状严格匹配：

| 校验项 | 条件 | 违反后果 |
|--------|------|---------|
| `gate_gemm.n == up_gemm.n` | 相同中间维度 | 跳过融合 |
| `gate_gemm.k == up_gemm.k` | 相同输入 hidden | 跳过融合 |
| `gate_gemm.m == up_gemm.m` | 相同 seq 维度（SymDim 相等） | 跳过融合 |
| `gate_gemm.inputs[0] == up_gemm.inputs[0]` | 共享 pack_a | 跳过融合 |
| `activation.inputs.len() == 1` | activation 必须一元 | 跳过融合 |

**实现**: `detect_ffn_block()` 在 `fusion/helpers.rs`。
**豁免**: FFNBlock 不受 `max_fusion_depth` 约束（结构性融合，非 epilogue chain）。

## §10 审计命令

```bash
# 裸数字 4 (F32 硬编码)
grep -rn "= 4usize\|= 4;\|element_size = 4\|\* 4[,;)]" src/ | grep -v test | grep -v "//"

# unwrap_or 降级
grep -rn "unwrap_or(" src/compiler/codegen/vm/ | grep -v test | grep -v "unwrap_or_else"

# for 循环展开
grep -rn "for .* in 0\.\.num_heads\|for .* in 0\.\.seq" src/compiler/codegen/vm/

# SYMDIM_MAX_SEQ_LEN 计算使用
grep -rn "/ SYMDIM_MAX\|/ (SYMDIM" src/

# 手动 stride 计算
grep -rn "\* elem[;,]" src/compiler/codegen/vm/ | grep -v "row_stride_bytes\|computation_elem"

# 手动 SymBound 构造
grep -rn "SymBound {" src/compiler/codegen/vm/ | grep -v "sym_map\|to_bound"

# eprintln 在生产代码
grep -rn "eprintln\!" src/ | grep -v test | grep -v "e2e_tests"

# 硬编码 QuantPrecision::F32（dtype 必须从 tensor 元数据推断）
grep -rn "QuantPrecision::F32" src/compiler/codegen/vm/plan_lower.rs | grep -v "op_input_dtype\|unwrap_or"

# computation_elem_bytes 已废弃
grep -rn "computation_elem_bytes" src/compiler/codegen/vm/

# row_stride_bytes 已废弃
grep -rn "row_stride_bytes" src/compiler/codegen/vm/

# emit 函数内部 F32 硬编码（ARCH-EMIT-DTYPE）
grep -rn "QuantPrecision::F32" src/compiler/codegen/vm/plan_lower.rs | grep -v "op_input_dtype\|unwrap_or"

# auto_lower_trace_raw 应迁移到 typed 版本
grep -rn "auto_lower_trace_raw\b" src/compiler/codegen/vm/plan_lower.rs
```

## §11 emit 函数 dtype 传播契约 (ARCH-EMIT-DTYPE)

> **铁律**：plan_lower.rs 中所有 emit 函数必须接受 `dtype: QuantPrecision` 参数，
> dtype 由调用方从 `op_input_dtype(op, graph)` 推断并传入。
> emit 函数内部禁止硬编码 `QuantPrecision::F32`。

### §11.1 dtype 推断入口

```rust
fn op_input_dtype(op: &CompilerOp, graph: &CompilerGraph) -> QuantPrecision {
    op.inputs.first()
        .and_then(|&tid| graph.tensor(tid))
        .map(|t| t.dtype.to_quant_precision())
        .unwrap_or(QuantPrecision::F32)  // 仅此处允许 F32 默认值（无输入张量的边缘情况）
}
```

**合法调用点**：
- `dispatch_compute_pattern`：推断 dtype 后传入各 emit 函数
- `emit_standalone_op`：同上
- `emit_fusion_groups`：从 fusion group 的第一个 op 推断

**禁止**：emit 函数内部调用 `op_input_dtype()`（dtype 由调用方推断并传入）。

### §11.2 emit 函数签名规范

所有 emit 函数必须添加 `dtype: QuantPrecision` 参数：

| 函数 | dtype 用途 |
|------|-----------|
| `emit_elementwise_inline` | `dtype.elem_bytes()` 替换 `computation_elem_bytes()`, VmInstr.dtype |
| `emit_normlike_inline` | 同上 + `dim * dtype.elem_bytes()` 替换 `row_stride_bytes(dim)` |
| `emit_normlike_one_group` | 同上 |
| `emit_layernorm_auto` | 同上 |
| `emit_softmax_inline` | VmInstr.dtype |
| `emit_gemm_inline_with_hook` | VmInstr.dtype |
| `emit_gemm_naive_inline` | VmInstr.dtype |
| `emit_gemm_blis_inline` | VmInstr.dtype |
| `emit_gemm_inline_with_epilogue` | VmInstr.dtype |
| `emit_injective_inline` | 传入 `auto_lower_trace_typed` |
| `emit_ple_fused_elementwise` | 传入 `auto_lower_trace_typed` |
| `emit_ple_residual_add` | 传入 `auto_lower_trace_typed` |
| `emit_gather_inline` | VmInstr.dtype |
| `emit_column_slice_inline` | VmInstr.dtype |
| `emit_rope_inline` | VmInstr.dtype |
| `emit_tiled_attention_inline` | VmInstr.dtype |
| `emit_moe_router_gemv_inline` | VmInstr.dtype |
| `emit_moe_topk_dispatch_inline` | VmInstr.dtype |
| `emit_moe_packed_inline` | VmInstr.dtype |
| `emit_quant_gemm_inline` | VmInstr.dtype |
| `emit_row_copy` | VmInstr.dtype |

**豁免**：`emit_zero_fill_bytes` — 与 dtype 无关（纯字节清零）。

### §11.3 auto_select.rs 公共 API

`auto_lower_trace_raw` / `auto_lower_trace` / `auto_lower_trace_into` / `auto_lower_trace_multi`
均接受 `dtype: QuantPrecision` 参数。

`dispatch_trace_op` 接受 `dtype` 参数，所有 VmInstr 构造使用传入的 `dtype`。

`copy_vreg` / `emit_binop` / `emit_binop_into` / `emit_fwht` 等辅助函数
接受 `dtype: QuantPrecision` 参数，VmInstr 构造中使用 `dtype`。

**禁止**：auto_select.rs 内部出现 `QuantPrecision::F32` — dtype 始终从调用方传入。

### §11.4 删除目标

所有调用点替换完成后，删除 `lower.rs` 中的：
- `computation_elem_bytes()` — 硬编码 `size_of::<f32>()` = 4
- `row_stride_bytes(dim)` — 硬编码 `dim * 4`

## §12 Loader → Graph dtype 契约 (REQ-DTYPE-CHAIN)

> **SSOT**: `../gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md §0.9` 定义全链路 dtype 契约。
> 本节描述 gllm 侧（Loader → Graph）的 dtype 传播责任。

### §12.1 dtype 传播链

```
TensorMeta { name, shape, dtype }    ← Loader 层产出（SSOT）
  ↓
weight_ptrs: HashMap<String, *const u8>  ← 指向原始 dtype raw bytes（无转换）
weight_sizes: HashMap<String, usize>     ← 原始 dtype 的 byte 大小
  ↓
auto_graph::build_compiler_graph()
  每个 add_tensor_*() 使用 TensorMeta.dtype（per-tensor，非全局 config.dtype）
  ↓
CompilerGraph { tensors: Vec<TensorMeta { dtype: DType }> }
  ↓
GraphDerivedGeometry::from_graph() → compute_dtype: DType（用于 buffer/scratchpad）
graph.weight_layout() → per-tensor 偏移（按 per-tensor dtype.size_bytes()）
```

### §12.2 Loader 责任

| 格式 | 当前行为 | 目标行为 |
|------|---------|---------|
| SafeTensors | `convert_tensor_to_f32()` 全转 F32 | 保留原始 dtype raw bytes |
| GGUF float | F16/BF16 转 F32 | 保留原始 bytes + GgmlDType |
| GGUF 量化 | 已保留 raw bytes + dtype | 不变 |
| ONNX | 继承 provider 行为 | 跟随 provider 改进 |

### §12.3 禁止事项

- ❌ Loader 层 `convert_tensor_to_f32()` 对 BF16/F16 tensor 调用
- ❌ `auto_graph` 中 `let dt = match config.dtype` 全局覆盖 per-tensor dtype
- ❌ `GraphDerivedGeometry.dtype` 用于权重偏移计算（应用 `graph.weight_layout()`）
- ❌ executor 层对权重做 dtype 转换再传入 mega-kernel
