# compute_dtype 宪法 -1 合规重构方案

> 承接 sessionId 401396fe（层1 裁决：SmolLM2 BF16 路径数值自洽，非发散根因）。
> 本方案处理**层2 违宪根治**：derive_compute_dtype 硬编码降级违反宪法 -1（ARCH-NO-PRECISION-ASSUMPTION）。
> **与发散诊断解耦**：本重构对当前 SmolLM2 发散零帮助，是纯合规 + 混合精度/NVFP4 铺路。
> 证据来源：4 个 explore agent 读源码交叉验证（gllm + gllm-kernels，HEAD gllm=d576b28f kernels=1d7eee3f）。

---

## 一、违宪的本质：compute_dtype 是过载概念

`compute_dtype` 把三个正交语义塞进同一字段，其中一个已剥离，剩两个耦合导致违宪：

| 语义 | 是什么 | 当前载体 | 宪法 -1 要顺从的数据源 | 现状 |
|------|--------|---------|----------------------|------|
| **A 累加器精度** | FMA acc 寄存器 dtype | compute_dtype | 硬件能力 + 配置，兜底 F32 | ❌ 硬编码 F32，device 被忽略 |
| **B KV cache 存储精度** | runtime 产生的 K/V | compute_dtype(buffer) + spec.dtype 硬编码 F32(stride/load) | K/V projection 输出张量 dtype | ❌ 主权分裂（split-brain 隐患）|
| **C 权重存储精度** | 权重文件字节 | ~~compute_dtype~~ | weight_dtypes/tdt/b_dtype | ✅ 已 per-tensor 化，合规 |

**关键认知**：语义 C（权重布局）已从 compute_dtype 剥离（build_graph.inc.rs:86 `tdt`、lower_op.inc.rs:1357-1362 三路 b_dtype、build_graph.inc.rs:357 weight_physical_bytes 逐张量）。NVFP4 原生寄存器解码已实现（emit_nvfp4_sub_block_dequant，无 F32 buffer 落地）。所以本重构**不碰权重侧**，只拆分 A+B。

---

## 二、compute_dtype 的 6 个消费点（agent 2 取证）

| # | 用途 | 位置 | 归属语义 | 重构后主权 |
|---|------|------|---------|-----------|
| 1 | scratchpad/logits/SG buffer 尺寸 | abi_types.inc.rs:395-465 | A（累加器/输出精度） | 保留 compute_dtype（=累加器精度）|
| 2 | KV cache 尺寸+stride | abi_types.inc.rs:469-489 | **B** | 改为 kv_dtype（K/V 输出张量 dtype）|
| 3 | 权重 dtype 转换目标 needs_dtype_conversion | executor_compile.rs:185-206 | 遗留 | 对 BF16 已是死代码（上轮确认），保留 |
| 4 | hetero 层维度反推 | executor_compile.rs:329-378 | 尺寸计算 | 保留（用累加器 elem_bytes 反推维度合理）|
| 5 | **TurboQuant 开关** `!= F32` | executor_builder.rs:219 | **耦合 bug** | 解耦：改看 storage_dtype 是否量化 |
| 6 | 诊断/logits 解码 | pack_observe.inc.rs:650-706 | A（输出精度） | 保留 |

---

## 三、重构方案（3 阶段，每阶段独立可验证）

### 阶段 1：derive_compute_dtype → 累加器精度，顺从硬件+配置（行为不变，逻辑合规）

**目标**：消除 `BF16 => F32` 恒等预设，改成 device 查询 + 配置 + 兜底分支。

**改动 1.1** — `gllm-kernels/src/compiler/dtype_chain.rs:195-210`
```
现状:
  match storage_dtype {
      BF16 | F16 => F32,          // 恒等预设，device 忽略（违宪）
      U8 | F8E4M3 | ... => F32,
      F32 => F32,
  }

改为（累加器精度语义，device 真正参与）:
  // 累加器精度顺从硬件能力：能原生累加则用原生，否则数值安全兜底 F32
  // 这不是"BF16 always => F32"预设，是"无原生累加支持时的显式兜底分支"
  match device.dot_product_cap() {
      DotProductCap::NativeBf16 if matches!(storage_dtype, BF16) => BF16,  // AMX/VDPBF16PS BF16 累加
      // NVFP4 + FP4 tensor core → acc F32（tensor core 输出 F32，非预设）
      _ => F32,  // 兜底：widen 到 F32 累加（数值安全，非恒等预设）
  }
```
**理由**：宪法 -1 禁止的是"无视硬件的恒等映射"。改后 device.dot_product_cap() 前置查询，F32 是兜底分支。当前硬件（i9-10900KF 无 AMX-BF16、5070Ti GPU 路径）→ 仍 F32，**行为零变化**，但逻辑合规。
**注**：语义从"storage/权重精度"正式改名为"累加器精度"，注释同步（graph_geometry.rs:26 注释已经是"accumulator precision"，与新语义一致，只是实现之前没兑现）。

**改动 1.2** — 统一双 compute_dtype 主权分裂（隐藏 bug #2）
- `ModelGeometry.compute_dtype`（types.inc.rs:76，用户 Option override，executor_builder.rs:95 用）
- `GraphDerivedGeometry.compute_dtype`（graph_geometry.rs:127 derive_compute_dtype，MegaKernelCompiled 用）
- **问题**：两条路径产出可能不同值。用户 `with_compute_dtype(BF16)` 设 ModelGeometry，但 MegaKernelCompiled 用 GraphDerived（硬编码 F32）→ 用户配置被忽略。
- **改**：derive_compute_dtype 增加 `config_override: Option<DType>` 参数，P0 优先用户配置。graph_geometry.rs:127 调用点传入 ModelGeometry.compute_dtype override。

**阶段 1 影响面**：累加器精度用途（#1 scratchpad #4 hetero #6 诊断）+ TurboQuant 开关（#5，见阶段 3 解耦）。当前硬件行为不变。
**阶段 1 回归门**：全量 `cargo test --lib` + SmolLM2/Qwen3 E2E golden 数值对齐（cosine 不退化）+ 确认所有本地硬件 acc_dtype 仍 F32。

---

### 阶段 2：KV cache dtype 主权归位到 K/V projection 输出张量（消除 split-brain）

**目标**：KV cache 三来源（buffer/MemCopy/VecLoad）统一到单一 SSOT = k_out/v_out 张量 dtype。

**当前分裂**（agent 1 取证）：
| 来源 | 位置 | 当前主权 | 当前值 |
|------|------|---------|--------|
| buffer 分配 | abi_types.inc.rs:469→396 | compute_dtype | F32/768 |
| MemCopy stride | lower_op.inc.rs:1521→1488 | spec.dtype 硬编码 F32 | F32/768 |
| attention VecLoad | attention_emit.rs:122 | spec.dtype 硬编码 F32 | F32/768 |
| **数据产生点** k_out/v_out | build_graph.inc.rs:590,609 | act_dt 硬编码 F32 | F32 |

**改动 2.1** — AttentionSpec.dtype 不再硬编码，从 K/V 输出张量推导
- `build_graph.inc.rs:693, 1316, 1523`（三处 AttentionSpec 构造）
- 现状：`dtype: DType::F32` 硬编码
- 改为：`dtype: <k_out 张量的 dtype>`（当前 = act_dt = F32，跟随数据产生点）
- 语义：AttentionSpec.dtype 表达"KV cache 里存的是什么 dtype"，应等于写入 KV cache 的 k_out/v_out dtype

**改动 2.2** — abi_types KV cache 尺寸用 kv_dtype 而非全局 compute_dtype
- `gllm/src/engine/mega_kernel/abi_types.inc.rs:469-489`
- 现状：kv_row_stride 用 `elem_bytes()` = compute_dtype.size_bytes()
- 改为：MegaKernelCompiled 新增 `kv_dtype: DType` 字段（从 graph 的 K/V projection 输出张量 dtype 推导，executor_core.inc.rs 填充），kv_row_stride 用 kv_dtype.size_bytes()
- 消除"buffer 绑 compute_dtype、stride 绑 spec.dtype"的双主权

**改动 2.3** — kv_bytes_per_token 用 kv_dtype（隐藏 bug #3）
- `gllm/src/model_config_fragments/types.inc.rs:211-217`
- 现状：用 self.dtype.size_bytes()（storage dtype）
- 改为：用 kv_dtype，与实际 buffer 分配一致

**阶段 2 影响面**：KV cache 尺寸/stride/读写。当前全 F32 → 行为不变，但消除 split-brain 隐患，且让"未来 K/V 输出 BF16 → KV cache 自动 BF16"成为正确路径（而非现在改一处就越界）。
**阶段 2 回归门**：全量测试 + E2E golden + 显式断言三来源 stride 一致（可加临时插桩验证 buffer/MemCopy/VecLoad stride 相等）。

---

### 阶段 3（收尾）：解耦 TurboQuant + 清理死标签 + NVFP4 W512

**改动 3.1** — TurboQuant 开关解耦（阶段 1 前置依赖）
- `gllm/src/engine/executor_builder.rs:219`
- 现状：`if compute_dtype != F32 { TurboQuant enabled }`
- 问题：重构后累加器若为 BF16 会误开 TurboQuant
- 改为：看 storage 是否量化 —— `if weight_dtypes.values().any(|d| d.is_quantized())` 或从 geometry.storage_dtype 判断
- **注**：此改动必须在阶段 1 合入前或同批，否则阶段 1 改累加器精度可能误触发 TurboQuant

**改动 3.2** — DequantMethod 死标签处理（NO-ISLAND 边缘）
- `gllm-kernels/src/compiler/trace.rs:1025-1036`
- agent 4 确认：DequantMethod(VNNI/AMX/ScalarLUT/BlockScale) 在 lowering 层无 match 消费，实际路径由 QuantGemmPlan/GemmKernel 决定
- 选择：接线（让 x86_elem_strategy 的 DequantMethod 真正驱动 emit）或删除（GemmKernel 已是真源）
- **低优先级**，独立评估，不阻塞 A/B

**改动 3.3** — NVFP4 W512(ZMM) 补齐
- `gllm-kernels/src/compiler/codegen/vm/x86_lower/lower_instr.inc.rs:1360-1362, 1572-1575`
- 现状：W512 NVFP4 命中 CodegenViolation
- 当前硬件不触发（i9 无 AVX-512、5070Ti GPU）→ **低优先级**，但宪法 -1 要求 NVFP4 全路径原生
- 补 ZMM 路径或至少留明确的"未实现"错误（当前已是 Err，符合 NO-SILENT-FALLBACK）

---

## 四、总影响面 + 风险

| 阶段 | 改动文件数 | 当前硬件行为 | 风险 | 收益 |
|------|-----------|-------------|------|------|
| 1 累加器精度 | 2-3（dtype_chain + graph_geometry + 调用点）| 不变（仍 F32）| 低（兜底 F32 保行为）| device 参与，配置生效，去恒等预设 |
| 2 KV cache 主权 | 4（build_graph×3 + abi_types + lower_op + types）| 不变（仍 F32）| 中（改 KV stride 主权，须验 stride 一致）| 消除 split-brain，混合精度 KV 铺路 |
| 3 收尾 | 3（executor_builder + trace + lower_instr）| TurboQuant 修正 | 低-中 | 解耦 + NVFP4 全路径 |

**核心风险警告**：
1. 本重构**当前对 SmolLM2 发散零帮助**（上轮已证 dtype 链自洽）。发散诊断须并行走别的方向（Gather/decode M=1/RoPE）。
2. 重构本身有回归风险（改 KV stride 主权、累加器推导）。**必须与发散诊断分开 commit/分支**。
3. 分阶段验证：阶段 1 行为不变最安全，先做+全量回归确认无退化，再做阶段 2。阶段 3.1（TurboQuant 解耦）须与阶段 1 同批。

---

## 五、宪法合规验证

| 宪法 | 重构后是否合规 | 证据 |
|------|--------------|------|
| **-1 ARCH-NO-PRECISION-ASSUMPTION** | ✅ | derive_compute_dtype 去恒等预设，device.dot_product_cap() 前置查询，F32 是兜底分支非映射 |
| **1 ARCH-BLOB-YIELDS-WEIGHT** | ✅（已合规，不动）| 权重侧 per-tensor（tdt/weight_dtypes），blob 保原始 dtype |
| **2 ARCH-MEMORY-FIRST** | ✅ | KV cache dtype 跟数据产生点（k_out 张量），内存布局顺从数据 |
| **ARCH-JIT-DATA-YIELDS** | ✅ | 累加器/KV dtype 顺从硬件+数据产生点，非代码预设 |
| **ARCH-JIT-YIELDS 四重信息源** | ✅ | P0 配置 > P1 硬件(dot_product_cap) > P2 兜底 |

---

## 六、执行建议（给 Commander 派 Executor）

- **DAG 顺序**：阶段 3.1（TurboQuant 解耦，前置）→ 阶段 1（累加器精度）→ 全量回归 → 阶段 2（KV cache 主权）→ 全量回归 → 阶段 3.2/3.3（收尾，可选）
- **每阶段 commit_gate**：@trace 覆盖 REQ-DTYPE-* + verify(alignment) + SmolLM2/Qwen3 E2E golden cosine 不退化
- **BCE 沉淀**：修正知识库 derive-compute-dtype-unconstitution.md（违宪链对 BF16 不可达的更正）+ BUG-KNOWLEDGE.md:2928/2998
- **发散诊断另开线**：dtype 链已排除，按 Gather seq 维 / decode M=1 / RoPE partial 方向，M=1 单 token prefill 逐算子 golden 对齐
