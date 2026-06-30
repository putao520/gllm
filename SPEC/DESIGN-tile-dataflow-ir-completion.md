# DESIGN: Tile 数据流 IR 补全 (CR-TIER-SOVEREIGNTY-004 根治)

> 状态: accepted (设计评审通过)
> 关联: CR-TIER-SOVEREIGNTY-004 / TASK-07 / ARCH-AUTO-INSTR-SELECT / NO-SILENT-FALLBACK
> 决策: 在 VmInstr IR 层补全 tile 数据流 (TileLoad/TileStore + TileMma 加 shape),
>       让 numerical_sim 自主模拟 13 后端数值对齐,不修正 SPEC,不依赖 Intel SDE。

---

## 0. 决策背景与裁决

### 0.1 根因 (代码核查确认)

`emit_tile_gemm` (gemm_emit.rs:13) 当前产出:

```
TileConfig { rows, cols, dtype }
Loop(k_tiles): TileMma { c: tile_c, a: tile_a, b: tile_b }   // a/b/c 是 dummy VRegId
TileRelease
```

致命缺陷: **没有 TileLoad 指令**。tile_a/tile_b 是空 tile vreg,真实矩阵数据加载:

- x86 (AMX): `tile_loadd` 硬编码 VEX 字节 `[rsi+rdi*1]` 直接寻址,绕过 VmInstr IR
- aarch64 (SME): TileMma 用 `resolve_vreg(a/b)` 跑 FMOPA,但 vreg 从未被灌数据 → **未初始化寄存器上做 MMA (潜伏 bug)**
- gpu (PTX/HIP): TileMma 用 `reg_name(c/a/b)` 跑 mma.sync,同样 vreg 从未灌数据 → **潜伏 bug**

numerical_sim 是 VReg 数据流解释器,TileMma 的 a/b vreg 空 → 只能 `Ok(())` NOP → 11 tile 后端无数值对齐。

### 0.2 三方案对比与裁决

| 方案 | 机制 | 覆盖 | 裁决 |
|------|------|------|------|
| B (SPEC 修正) | 改 CR 为 tile 用结构测试 | — | 否决 (SPEC 不为实现让步) |
| SDE | Intel 模拟器执行 x86 机器码 | 仅 3/10 Intel AMX,GPU/ARM 7 个无关 | 否决 (外部工具 + 覆盖不全) |
| **IR 补全** | tile 数据流提回 IR,numerical_sim 自主模拟 | **13/13 全后端同一解释器** | **采纳** |

裁决理由:
1. ARCH-AUTO-INSTR-SELECT 铁律明文"数据流必须在 IR 层,禁止 opaque 算子跳过 trace"。tile 数据流硬编码在 VEX 字节 = 历史 IR 欠债,补全 = 还债。
2. NO-ISLAND-MODULE: tile 数据流进 IR 不再是孤岛。
3. NO-SILENT-FALLBACK: 13 后端全在 numerical_sim 验证,无 NOP 静默。
4. 顺带根治 SME/GPU 未初始化寄存器 MMA 潜伏 bug。

### 0.3 验证边界 (诚实标注)

numerical_sim 验证 **IR 语义正确** (TileLoad/TileMma/TileStore 序列算出正确 GEMM),
**不验证** x86_lower 的 VEX 字节 / gpu_lower 的 PTX 文本是否忠实实现该 IR。
此 gap 对 FMA 路径同样存在 → tile 验证达到与 FMA **同等** tier。
ISA lowering 字节正确性由结构测试 + requires() 谓词测试保留并存 (不删)。

---

## 1. VmInstr IR 定义

### 1.1 新增/修改指令 (vminstr.inc.rs ~300 行附近)

```rust
// ── Tile/Matrix 操作 (数据流完整版) ──

/// 配置 tile 寄存器 (AMX TILECFG / SME SMSTART / GPU fragment decl)。不变。
TileConfig { rows: usize, cols: usize, dtype: DType },

/// 把内存中的 2D 矩阵块加载进 tile vreg。
/// 语义: dst_tile[r][col] = mem[base_ptr + k_offset + r*row_stride + col*elem_bytes]
///   for r in 0..rows, col in 0..cols
/// @trace REQ-HW-TIER-007 [req:VmInstr-TileLoad] tile 数据加载进 IR,消除 lowering 硬编码寻址
TileLoad {
    dst_tile: VRegId,   // 目标 tile vreg
    base_ptr: VRegId,   // 矩阵基址 (A 或 B 的 ptr vreg)
    k_offset: VRegId,   // K 循环偏移 (循环变量推进 base,= 当前 k 块起始字节偏移)
    row_stride: usize,  // 2D tile 内逐行字节跨度 (→ x86 rdi stride reg)
    rows: usize,
    cols: usize,
    dtype: DType,
},

/// Tile MMA: c += a × b,带完整 shape 让解释器跑 host FMA 累加。
/// @trace REQ-HW-TIER-008 [req:VmInstr-TileMma-Shape] TileMma 携带 shape,解释器可模拟累加语义
TileMma {
    c: VRegId, a: VRegId, b: VRegId,
    m: usize, n: usize, k: usize,   // tile MMA 形状 (a:m×k, b:k×n, c:m×n)
    dtype: DType,                    // 输入精度,决定 round-trip 窄化
},

/// 把 tile vreg 结果写回内存。
/// 语义: mem[base_ptr + out_offset + r*row_stride + col*elem_bytes] = src_tile[r][col]
/// @trace REQ-HW-TIER-009 [req:VmInstr-TileStore] tile 结果写回进 IR
TileStore {
    src_tile: VRegId,
    base_ptr: VRegId,    // 输出 C 的 ptr vreg
    out_offset: VRegId,  // C tile 在输出中的字节偏移
    row_stride: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
},

/// 释放 tile 资源。不变。
TileRelease,
```

### 1.2 寻址语义关键约束 (R1 决定性风险)

x86 AMX `TILELOADD tmm, [base + stride_reg]` 的 stride_reg 是**逐行跨度**,不是 K 偏移:

| IR 字段 | x86 物理映射 | 语义 |
|---------|-------------|------|
| `base_ptr` | rsi (A) / rdx (B) — 见 mega_kernel_abi | 矩阵基址 |
| `k_offset` | 循环变量值,加到 base | K 块推进 (每次迭代 += k*elem_bytes) |
| `row_stride` | rdi (stride reg) | tile 内逐行字节跨度 |

**铁律**: `k_offset` (循环推进) 与 `row_stride` (2D 行跨度) 是两个独立量,禁止合并成单一 offset。
合并 → x86 TILELOADD 误寻址,数值全错。

**Executor 前置确认**: 写 x86 TileLoad lowering 前,先用 debug_process 或读 emit_loop 实现,
确认循环计数器绑定的物理寄存器,与 rdi (stride) 不冲突。base_ptr+k_offset 的有效地址
计算落在哪个寄存器,TILELOADD 的 base/stride 两个寄存器操作数如何分配。

---

## 2. emit_tile_gemm 重写 (gemm_emit.rs:13)

### 2.1 重写后伪码

```rust
pub(crate) fn emit_tile_gemm(
    prog: &mut VmProgram, width: SimdWidth,
    rows: usize, cols: usize, kd: usize, k: usize, dt: DType,
    a_ptr: VRegId, b_ptr: VRegId, c_ptr: VRegId,   // ← 新增: 真实矩阵 ptr
) -> Result<(), CompilerError> {
    let tile_c = prog.alloc_vreg(VRegKind::Tile, width);
    let tile_a = prog.alloc_vreg(VRegKind::Tile, width);
    let tile_b = prog.alloc_vreg(VRegKind::Tile, width);
    let row_stride = cols * dt.size_bytes();  // 行跨度

    prog.emit(VmInstr::TileConfig { rows, cols, dtype: dt });
    let k_tiles = (k + kd - 1) / kd;
    prog.emit_loop(BoundExpr::Const(k_tiles), kd * dt.size_bytes(), |prog, ctr, off| {
        // off = 当前 K 块字节偏移 (循环变量推进)
        prog.emit(VmInstr::TileLoad {
            dst_tile: tile_a, base_ptr: a_ptr, k_offset: off,
            row_stride, rows, cols: kd, dtype: dt,
        });
        prog.emit(VmInstr::TileLoad {
            dst_tile: tile_b, base_ptr: b_ptr, k_offset: off,
            row_stride, rows: kd, cols, dtype: dt,
        });
        prog.emit(VmInstr::TileMma {
            c: tile_c, a: tile_a, b: tile_b,
            m: rows, n: cols, k: kd, dtype: dt,
        });
    });
    prog.emit(VmInstr::TileStore {
        src_tile: tile_c, base_ptr: c_ptr, out_offset: /* C tile 偏移 */,
        row_stride: cols * dt.size_bytes(), rows, cols, dtype: dt,
    });
    prog.emit(VmInstr::TileRelease);
    Ok(())
}
```

### 2.2 调用方适配

12 个 OpImpl 调 emit_tile_gemm 处 (gemm_impls.rs:93,123,144,161,189,206 等),
传入 lo.a_ptr/b_ptr/c_ptr (GemmOpLayout 已有这三个字段)。

---

## 3. numerical_sim tile 值模型 (numerical_sim.rs)

### 3.1 VmInterpState 扩展

当前 VmInterpState 只存 lane vector (Vec<f32>)。新增 tile 值表示:

```rust
// tile vreg 的值 = 2D 矩阵块 (row-major f32)
struct TileVal { rows: usize, cols: usize, data: Vec<f32> }  // data[r*cols + c]

// VmInterpState 加: tile_regs: HashMap<VRegId, TileVal>
```

### 3.2 三指令模拟逻辑

```rust
VmInstr::TileLoad { dst_tile, base_ptr, k_offset, row_stride, rows, cols, dtype } => {
    let buf = state.buffer_of(*base_ptr)?;        // 模拟内存 buffer
    let base_off = state.get_scalar(*k_offset)?;  // K 循环偏移
    let eb = dtype.size_bytes();
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..*rows {
        for c in 0..*cols {
            let byte = base_off + r*row_stride + c*eb;
            let v = read_elem(buf, byte, *dtype);    // 按 dtype 解码
            data.push(narrow_to_dtype(v, dtype.into())); // round-trip 窄化 (R4)
        }
    }
    state.tile_regs.insert(*dst_tile, TileVal { rows: *rows, cols: *cols, data });
    Ok(())
}

VmInstr::TileMma { c, a, b, m, n, k, dtype } => {
    let ta = state.tile_regs.get(a)?;  // m×k
    let tb = state.tile_regs.get(b)?;  // k×n
    let tc = state.tile_regs.entry(*c).or_insert(TileVal::zeros(*m, *n));
    for i in 0..*m {
        for j in 0..*n {
            let mut acc = tc.data[i*n + j];
            for kk in 0..*k {
                // host FMA 累加,输入按 dtype 窄化模拟硬件精度
                let av = narrow_to_dtype(ta.data[i*k + kk], (*dtype).into());
                let bv = narrow_to_dtype(tb.data[kk*n + j], (*dtype).into());
                acc += av * bv;   // F32 累加 (AMX/TC 累加器是 F32)
            }
            tc.data[i*n + j] = acc;
        }
    }
    Ok(())
}

VmInstr::TileStore { src_tile, base_ptr, out_offset, row_stride, rows, cols, dtype } => {
    let tile = state.tile_regs.get(src_tile)?;
    let buf = state.buffer_mut_of(*base_ptr)?;
    let off = state.get_scalar(*out_offset)?;
    let eb = dtype.size_bytes();
    for r in 0..*rows {
        for c in 0..*cols {
            write_elem(buf, off + r*row_stride + c*eb, tile.data[r*cols + c], *dtype);
        }
    }
    Ok(())
}
```

### 3.3 删除旧 NOP

删除 numerical_sim.rs:1860-1862 的 `TileConfig|TileRelease => Ok(())` (TileConfig/Release 仍 NOP,
但 TileMma 不再 NOP) 与 `TileMma { .. } => Ok(())`,换成上述三个完整 arm。
TileConfig/TileRelease 在解释器侧保持 NOP (抽象资源,无数值语义)。

---

## 4. ISA Lowering 映射表

| 指令 | x86_lower (AMX) | aarch64_lower (SME) | gpu_lower (PTX/HIP) |
|------|-----------------|---------------------|---------------------|
| TileLoad | `TILELOADD tmm, [base+stride]` (VEX 4B) | `LD1W {Za}, Pg/Z, [base, k_off]` + MOVA | `ldmatrix.sync` (PTX) / `global_load` (HIP) |
| TileMma | `TDP{BF16,FP16,HF8,BF8,TF32}PS` (现有) | `FMOPA ZA.S` (现有,但操作数现有真实数据) | `mma.sync`/`wgmma`/`wmma`/`v_mfma` (现有) |
| TileStore | `TILESTORED [base+stride], tmm` (VEX 4B) | `ST1W {Za}` / `MOVA Z←ZA` + ST | `st.global` (PTX) / `global_store` (HIP) |

**关键**: TileMma 的 lowering 已存在 (x86/aarch64/gpu 三处),本次只改"操作数现在指向已被 TileLoad
灌入数据的 tile vreg"。新增的是 TileLoad/TileStore 两条 lowering arm。

---

## 5. 完整改动范围 (用户清单 + 5 处补漏)

| # | 文件 | 改动 | 用户清单? |
|---|------|------|----------|
| 1 | vminstr.inc.rs:~300 | 新增 TileLoad/TileStore + TileMma 加 shape | ✅ |
| 2 | gemm_emit.rs:13 | emit_tile_gemm 重写 + 签名加 a/b/c_ptr | ✅ |
| 3 | gemm_impls.rs (12 调用点) | 调 emit_tile_gemm 传 ptr | ✅ (隐含) |
| 4 | numerical_sim.rs:1860 | 删旧 NOP,补 TileLoad/Mma/Store 模拟 + TileVal 类型 | ✅ |
| 5 | x86_lower/lower_instr.inc.rs:1332 | TileMma 完整解构加 shape;新增 TileLoad/Store→VEX | ✅ |
| 6 | aarch64_lower/lower_instr.inc.rs:1367 | 同上,SME load/store | ✅ |
| 7 | gpu_lower/lower_instr.inc.rs:440 | 同上,ldmatrix/st.global | ✅ |
| 8 | reg_alloc.rs:728 | TileMma 解构加 shape;TileLoad/Store 的 vreg 操作数 | ✅ |
| 9 | verify.rs:1201 | TileLoad/Mma/Store def-before-use ({..} 通配,加新 arm 验证) | ✅ |
| **10** | **x86_lower TileConfig + reg_alloc** | **物理 tmm 分配 (8 个上限),tile-vreg→tmm 映射** | **❌ 补漏 (R2)** |
| **11** | **trace.rs:1339,1408** | **dtype_of + for_each_value 加新指令 arm (否则活性漏 base_ptr use)** | **❌ 补漏** |
| **12** | **program.inc.rs:312,543** | **renumber_vregs 完整解构加 shape;TileLoad/Store renumber arm** | **❌ 补漏** |
| **13** | **auto_select.rs:1498** | **TraceOp::TileMma→VmInstr::TileMma 转换加 shape** | **❌ 补漏** |
| **14** | **trace.rs:780 TraceOp::TileMma** | **决策: TraceOp 是否同步加 shape/新增 TileLoad-Store TraceOp** | **❌ 补漏 (见 6.1)** |

---

## 6. 待 Executor 决策的子问题

### 6.1 TraceOp 层是否同步补全

两条路:
- **A (推荐)**: TraceOp 不动,emit_tile_gemm 独家产 VmInstr::TileLoad/Store。TraceOp::TileMma 仅加 shape
  供 auto_select 透传。理由: tile 数据加载是 lowering 期决策,不是 symexec 提取的标量语义。
- B: symexec 也产 TileLoad/Store TraceOp。理由: 更彻底。但 symexec 从标量 GEMM 提不出 tile 加载语义 (标量无 tile 概念) → 不可行。

**结论: 走 A。** TraceOp::TileMma 加 shape 透传,不新增 TraceOp::TileLoad/Store。

### 6.2 物理 tmm 分配策略 (R2)

AMX 8 个 tmm (tmm0-7),当前硬编码 c=0/a=1/b=2。显式 TileLoad 后,tile vreg 需映射物理 tmm。
最简: 沿用固定 c=tmm0/a=tmm1/b=tmm2 (单 MMA 链够用),reg_alloc 对 Tile-kind vreg 走独立小分配池。
SME ZA / GPU fragment 类比 (ZA 单累加器隐式,fragment 按 dialect)。
超 8 tmm → ResourceBudgetExceeded (JitContext 已有机制)。

---

## 7. 风险点 (给 Executor)

| ID | 风险 | 缓解 |
|----|------|------|
| **R1 决定性** | TileLoad 字段 row_stride(行跨度)vs k_offset(K推进)混淆 → x86 误寻址 | 写 x86 arm 前确认 emit_loop 计数器寄存器绑定 (6.2 前置) |
| R2 | tmm 物理分配 8 个上限 | 固定 c/a/b=tmm0/1/2,reg_alloc 独立 Tile 池 |
| R3 | numerical_sim 需 2D tile 值 + stride 读内存 | TileVal 类型 + read_elem 按 dtype 解码 |
| R4 | BF16/FP8 输入精度不窄化 → 对不上 ~1e-2 容差 | TileLoad + TileMma 输入走 narrow_to_dtype |
| R5 | numerical_sim 不验 VEX/PTX 字节 | 结构测试 + requires() 谓词测试保留并存 |

---

## 8. Executor TASK 拆分 (C-E-W-V)

文件域分析 (writes 集):

- **TASK-A** writes: {vminstr.inc.rs, trace.rs} — IR 定义 (TileLoad/Store + TileMma shape + TraceOp 透传)
- **TASK-B** writes: {gemm_emit.rs, gemm_impls.rs} — emit_tile_gemm 重写 + 调用点 — blockedBy A
- **TASK-C** writes: {numerical_sim.rs} — tile 模拟 + TileVal — blockedBy A
- **TASK-D** writes: {x86_lower/lower_instr.inc.rs, reg_alloc.rs} — x86 lowering + tmm 分配 — blockedBy A
- **TASK-E** writes: {aarch64_lower/lower_instr.inc.rs} — SME lowering — blockedBy A
- **TASK-F** writes: {gpu_lower/lower_instr.inc.rs} — PTX/HIP lowering — blockedBy A
- **TASK-G** writes: {program.inc.rs, verify.rs, auto_select.rs} — renumber + verify + 透传 — blockedBy A

A 是根 (定义),B-G 并行 (文件域不相交,各进 Watcher DAG)。
撞文件: 无 (各 TASK writes 两两不相交) → 可直调并行 Agent。

V (主会话批量): 13 后端 numerical_sim 数值对齐测试 (含 11 tile) + commit_gate + verify(alignment)
+ arch_insight(quality) + 结构测试 + requires() 谓词测试,全过 = TASK-07 闭环。

