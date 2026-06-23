# BUG-KNOWLEDGE.md — BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。

---

## ⚡ 综合归因 — 范式级缺陷模式

### 范式：场景覆盖不全（Partial Scenario Coverage, PSC）

**定义**：函数/计算只覆盖部分合法场景，忽略其余合法场景，导致"已覆盖场景正确但未覆盖场景静默失效或崩溃"。

**认知根因**：锚定偏差（Anchoring Bias）— 开发者锚定在第一个/最常用的场景，其他合法场景未被纳入覆盖，导致"部分场景正确但其余场景静默失效或崩溃"。

**3 个实例的共性提取**：

| 维度 | BCE-20260622-001 | BCE-20260622-002 | BCE-20260623-001 |
|------|------------------|------------------|------------------|
| **已覆盖场景** | specDir（SPEC 目录） | JS/TS import | generate 图（有 Argmax） |
| **未覆盖场景** | sourceDir（源码目录） | Rust/Go/Python/Java | single-pass 图（无 Argmax） |
| **失效表现** | 项目根指向不存在目录 | 依赖矩阵全零 | SIGSEGV heap-buffer-overflow |
| **根因代码** | `d === specDir ? resolve(abs,'..') : abs` | `if (resolvedPath.startsWith('.'))` | `logits_end = offset + N * vocab_size` (vocab_size=0) |
| **修复模式** | 统一取 parent | 按语言分派 | 取 max(生成, 单遍) |

**PSC 检测签名**（可横扫新代码）：

1. **条件分支只处理部分枚举值，其余值走空/fallback**：`if X == A { handle } else { skip/0/null }` — 其中 B/C/D 也是合法值且需要正确处理
2. **计算公式中某项为 0 时整体失效**：`result = f(known) + g(unknown)` 其中 `g(unknown)=0` 不代表"无需此项"
3. **硬编码默认值覆盖动态推导**：`cwd = GSC`（硬编码）vs `cwd = _projectRoot`（动态推导）
4. **部分场景正常但其余场景静默失效**：已覆盖场景产出正确结果，未覆盖场景产出空/零/错误值且无报错

**PSC 根治模板**：

1. **枚举所有场景**：在函数/计算设计时显式列出所有合法输入场景（而非仅"已覆盖"+"其他"）
2. **取并集而非取部分**：`result = max(scenario_A_need, scenario_B_need, ...)` 而非 `result = scenario_A_need`
3. **断言守卫**：`debug_assert!(result >= all_scenario_needs)` — 编译时/运行时捕获覆盖遗漏
4. **零值 ≠ 无需求**：当某场景的参数为 0 时，不代表该场景不存在，需区分"参数=0"和"场景不适用"

**跨项目横扫指引**：用以下模式搜索 gllm/gllm-kernels/gsc 代码中的 PSC 嫌疑点：

```bash
# 1. 条件分支只处理部分枚举值
grep -rn 'if.*===.*specDir\|if.*startsWith.*\.\|if.*vocab_size.*>.*0' --include='*.rs' --include='*.mjs'

# 2. unwrap_or(0) 可能掩盖"场景不适用"（0 是合法值还是 fallback？）
grep -rn 'unwrap_or(0)' --include='*.rs' | grep -v test | grep -v counter

# 3. 硬编码路径/目录
grep -rn 'cwd.*=.*GSC\|cwd.*=.*"/home\|default.*=.*"/' --include='*.mjs'
```

**PSC 横扫结果（2026-06-23）**：gllm + gllm-kernels 双仓全量横扫，发现 33 个嫌疑点。

已根治（3 个）：

| patternId | 位置 | 修复 |
|-----------|------|------|
| BCE-20260623-001 | gllm-kernels/compiler/mod.rs logits_end | 取 max(generate, single_pass) |
| PSC-采样scratch | gllm/abi_types.inc.rs runtime_scratchpad_bytes | 补回 sampling_bytes |
| PSC-测试 | gllm/compat/sampling.rs top_p_one_is_no_op | greedy(T=0) 替代 stochastic(T=1) |

待根治（30 个，全部须根治，无遗漏）：

| # | 位置 | 模式 | 失效表现 |
|---|------|------|---------|
| 1 | executor_ops.inc.rs:676 mega_compiled.unwrap_or(0) | 未编译时 scratchpad=0 | buffer overrun |
| 2 | gpu_backend_macro.rs:131 kv_caches.get().unwrap_or(0) | KV 指针=null | GPU fault |
| 3 | mtp_executor.rs:145 / executor_step.rs:52 logits.max_by().unwrap_or(0) | 空 logits→token 0; NaN→Equal | 错误 token |
| 4 | request_state.rs:484 DeviceMemory Drop 只释放 Cuda/Hip | Metal/Host 变体未释放 | 内存泄漏 |
| 5 | gllm-kernels/compiler/mod.rs GPU 路径缺少 sg_end/dwc_end | CPU/GPU 不一致 | 有 SG/DWC 时越界 |
| 6 | gllm/convert.rs:112 / reader.rs:218 模型元数据 unwrap_or(0) | 缺失字段静默为 0 | 后续崩溃难定位 |
| 7 | mid_layer_encode.rs:170 match dtype 只处理 F16/BF16 | FP8/INT8 走空分支 | 数据静默丢弃 |
| 8 | hgal.rs:314 PagePayloadKind 只处理 2/5 变体 | KvContext/PromptSystem/RAG 优先级=0 | 逐出优先级错误 |
| 9 | safetensors.rs:681 AWQ group_size fallback=128 | 非标准 AWQ 模型 | 静默错误反量化 |
| 10 | executor_core.inc.rs:624 output_tokens[0]!=0 判断生成 | token 0 是合法 token | 合法 token 被丢弃 |
| 11 | graph/profile.rs:170 num_experts unwrap_or(0) | MoE 配置缺失 | 静默当 dense |
| 12 | hip_backend.rs:208 PTX cache unwrap_or(0)+.max(1024) | PTX 缓存缺失 | 1KB stub 替代正确大小 |
| 13 | executor_api.rs:416 session position unwrap_or(0) | 错误 session_id | 位置静默重置 |
| 14 | cpu_backend.rs:150 attention_pattern unwrap_or(0) | 配置不匹配 | 层类型错误 |
| 15 | weight_tier.rs:96 GPU capacity unwrap_or(0) | 无 GPU | 容量=0 |
| 16 | three_tier_swap.rs:822 StorageTier 只统计 4/6 对 | GPU↔NVMe 直接换页 | 统计缺失 |
| 17 | hgal.rs:193 PageState 只给 Protected/Warm 加分 | Active/Standby 与 Free 同权 | 逐出优先级错误 |
| 18 | abi_types.inc.rs:336 sampling_bytes=vocab_bytes*4 | 硬编码乘数 | 新采样策略溢出 |
| 19 | gpu_backend_macro.rs:335 out_bytes=(N*4).min(scratch) | scratch 错误时截断输出 | 输出截断 |
| 20 | mega_kernel_gpu.rs:627 PREFILL_CHUNK_SIZE=512 | 固定分块 | 设备差异未感知 |
| 21 | mega_kernel_gpu.rs:636 POOL_LOCAL_CAPACITY=32 | 固定池大小 | 并发不足/浪费 |
| 22 | batch_context.rs:24 MAX_DECODE_STEPS=4 | 固定步数上限 | MTP depth>4 截断 |
| 23 | mega_kernel_gpu.rs:565 BATCH_CTX_EXTENSION_SIZE=128 | 固定扩展区 | 新字段可能溢出 |
| 24 | weight_tier.rs:47/51 容量分数硬编码 70%/60% | 固定比例 | KV 小/大时浪费/不足 |
| 25 | executor_core.inc.rs:624 output_tokens[0]!=0 | token 0 判断 | 合法 token 丢弃（同 #10） |
| 26 | weight_tier.rs:99 L3*100 估算主机容量 | 启发式 | 偏差大 |
| 27 | gllm-kernels compiler/mod.rs kv_bytes=hidden*2 | 硬编码 hidden | GQA/MQA 浪费内存 |
| 28 | gllm-kernels compiler/mod.rs activation_bytes=hidden*4 | 硬编码 F32 | BF16/F16 浪费 |
| 29 | gllm-kernels BufferLayout 无条件分配 SG 空间 | 注释说 0 when disabled 实际 >0 | 非 SG 模型浪费 |
| 30 | gllm-kernels mega_kernel_emit.rs hdim*4 硬编码 F32 | 与同函数其他位置不一致 | BF16/F16 偏移错误 |

---

## BCE-20260622-001: WF SDK "exists but failed to launch" — cwd 指向不存在目录

### BUG 模式签名
- **patternId**: BCE-20260622-001
- **title**: resolveProjectRoot 对 sourceDir 未取 parent + execCmd 硬编码 cwd=GSC
- **layer**: 设计缺陷（逻辑/边界错误）
- **codePattern**:
  - resolveProjectRoot() 对 sourceDir 返回 abs 而非 resolve(abs,'..')，导致项目根被设为子目录
  - execCmd() 硬编码 cwd=GSC，导致子进程在错误目录下运行
- **triggerCondition**: WF 传入 sourceDir="./src" 或 specDir="./SPEC"（项目子目录）时触发
- **detectionSignatures**:
  - literal: `resolveProjectRoot` 中 `d === specDir ? _resolve(abs, '..') : abs`
  - literal: `execCmd(cmd, { timeout, cwd = GSC })`
- **sameClassCriterion**: 任何从子目录路径推导项目根时只对部分参数取 parent 的逻辑
- **fixTemplate**: 统一对所有子目录参数取 parent；execCmd 默认 cwd 用推导出的项目根而非硬编码
- **regressionAssertion**: `resolveProjectRoot({sourceDir:"./src"}) === resolve(process.cwd(),".")` 而非 `resolve(process.cwd(),"src")`

### 根因
wflib.mjs resolveProjectRoot() 对 sourceDir 直接返回 abs（如 /path/to/project/src），对 specDir 才取 parent。sourceDir 和 specDir 都是项目子目录，统一应该取 parent 得到项目根。同时 execCmd() 默认 cwd 硬编码 GSC，导致子进程从 GSC 目录启动而非用户项目根。

### 影响
- smartAgent() 传递 cwd=_projectRoot（指向不存在的 src/ 目录）给 Claude Agent SDK
- SDK spawn Claude Code binary 时 chdir 到不存在的目录 → "exists but failed to launch" 错误
- 所有 WF（six-node-dev, batch-execute, test-full 等）的 S4 阶段全部失败

### 根治
1. resolveProjectRoot() 统一返回 `resolve(abs, '..')` 对 sourceDir 和 specDir
2. execCmd() 默认 cwd 改为 `_projectRoot`（动态推导）而非 `GSC`（硬编码）

### 归因时间
2026-06-22

---

## BCE-20260622-002: LSP Coupling Matrix 对非 JS/TS 语言失效 — import 路径解析缺失

### BUG 模式签名
- **patternId**: BCE-20260622-002
- **title**: _normalizeImport 只识别 JS/TS 相对路径，Rust/Go/Python/Java 模块路径全部被丢弃
- **layer**: 范式缺陷（假设所有 import 都是文件系统路径）
- **codePattern**:
  - `_normalizeImport` 只处理 `.` 开头的相对路径和绝对路径
  - Rust `crate::`/`super::`/`self::` → resolvedPath=null → 被跳过
  - Go module path → resolvedPath=null → 被跳过
  - Python package path → resolvedPath=null → 被跳过
  - Java package path → resolvedPath=null → 被跳过
- **triggerCondition**: 任何非 JS/TS 项目的 LSP scan architecture/dep/coupling 分析
- **detectionSignatures**:
  - literal: `resolvedPath = null; // external package`
  - literal: `if (resolvedPath.startsWith('.'))`
  - structural: Dependency Matrix 全零但项目有多模块
- **sameClassCriterion**: 任何语言特定的模块路径格式未被 import 解析器识别
- **fixTemplate**: 在 _normalizeImport 中按文件扩展名分派到语言特定的路径解析器
- **regressionAssertion**: Rust 项目的 `lsp_query(scan, architecture)` 返回非零依赖矩阵

### 根因
`_normalizeImport` 假设所有 import 都是 JS/TS 风格的相对路径（`./foo`）或绝对路径。非 JS/TS 语言的模块路径（Rust 的 `crate::`/`super::`、Go 的 module path、Python 的 package、Java 的 FQCN）不匹配任何已知模式，被标记为 `resolvedPath=null`（外部包），然后在 `buildDependencyMatrix` 中被 `if (!imp.resolvedPath) continue` 跳过。

### 影响
- 所有 Rust/Go/Python/Java 项目的 LSP coupling matrix/deps scan/architecture map 完全失效
- 依赖矩阵全零，架构模式被判定为 "flat"
- 子系统边界、循环依赖、合并建议等功能全部基于错误数据

### 根治
1. `_normalizeImport` 增加按文件扩展名的语言分派：`.rs` → `_resolveRustImportPath`，`.go` → `_resolveGoImportPath`，`.py` → `_resolvePythonImportPath`，`.java` → `_resolveJavaImportPath`
2. Rust 路径解析：`crate::` → 从 srcDir 解析；`super::` → 从父目录解析；`self::` → 从当前目录解析；逐级回退处理 inline module
3. `_buildModuleMap` 增加 Rust workspace 和 Go workspace 检测
4. `_detectProjectType` 增加 Rust workspace（Cargo.toml `[workspace]`）和 Go workspace（go.work）识别
5. `buildDependencyMatrix` 在 `scanImports` 之前设置 `_cachedFiles`
6. 新增 `_inferSrcDirFromPath` 方法，从 sourcePath 推导 srcDir，不依赖 _cachedFiles

### 归因时间
2026-06-22

---

## BCE-20260623-001: cargo test SIGSEGV — vision_forward scratchpad logits 区域不足

### BUG 模式签名
- **patternId**: BCE-20260623-001
- **title**: compile_cpu logits_end 未覆盖 output_float_elems → scratchpad 越界读取
- **layer**: 设计缺陷（scratchpad 大小计算未覆盖非生成图 output tensor 场景）
- **codePattern**:
  - `logits_end = logits_scratch_offset + max_seq_len * vocab_size * elem_bytes`
  - 无 Argmax 图: vocab_size=0 → logits_end = logits_scratch_offset → scratchpad 不为 output 分配空间
  - `execute_as_mega_kernel` 中 `copy_nonoverlapping(src, dst, output_float_elems)` 从 scratchpad 越界读取
- **triggerCondition**: 任何无 Argmax 的图（vision encoder / embedding / reranker）通过 compile_cpu 编译后调用 execute_as_mega_kernel
- **detectionSignatures**:
  - structural: `logits_end = logits_scratch_offset + N * vocab_size * elem_bytes` 且 vocab_size=0 时 logits_end == logits_scratch_offset
  - literal: `copy_nonoveranking(src, dst, output_float_elems)` 且 output_float_elems > (scratchpad_bytes - logits_scratch_offset) / 4
  - antipattern: "scratchpad 大小计算仅考虑 vocab_size 而忽略 output_float_elems"
- **sameClassCriterion**: 任何 scratchpad/buffer 大小计算仅考虑部分使用场景（如仅 generate 图的 vocab_size）而忽略其他场景（如 single-pass 图的 output tensor）
- **fixTemplate**: logits_end = logits_scratch_offset + max(generate_logits_bytes, single_pass_output_bytes)；加 debug_assert! 防回归
- **regressionAssertion**: 对任何 compile_cpu 输出: scratchpad_bytes >= logits_scratch_offset + output_float_elems * elem_bytes

### 根因
`compile_cpu` 中 `logits_end` 计算仅考虑 `max_seq_len * vocab_size * elem_bytes`（生成图的 logits 空间需求），忽略了无 Argmax 图中 output tensor 的空间需求。当 vocab_size=0（无 Argmax 图如 vision encoder）时，`logits_end == logits_scratch_offset`，scratchpad 不为 output tensor 分配空间，但 `execute_as_mega_kernel` 仍从 `scratchpad[logits_scratch_offset]` 读取 `output_float_elems` 个 f32，导致 heap-buffer-overflow。

具体数值（vision encoder tiny_config）：scratchpad_bytes=960, logits_scratch_offset=896, output_float_elems=32。logits 区域 = 960-896 = 64 bytes = 16 f32，但需要 32 f32 = 128 bytes，越界 64 bytes。

GPU compile 路径有同类问题：`total_scratch` 计算同样只考虑 `vocab_size`，未考虑 `output_float_elems`。

### 影响
- 全量 cargo test --lib SIGSEGV（signal 11）
- compat::vision_forward 530 测试全量跑时崩溃
- 所有无 Argmax 图（vision encoder / embedding / reranker）均受影响
- 仅 --skip compat::vision_forward 可绕过

### 根治
1. CPU 路径：`logits_end = logits_scratch_offset + max(generate_logits_bytes, single_pass_output_bytes)`
2. GPU 路径：`total_scratch` 计算增加 `single_pass_output_bytes` 考虑
3. 加 `debug_assert!(total_scratch >= logits_scratch_offset + output_float_elems * elem_bytes)` 防回归
4. 修改位置：`gllm-kernels/src/compiler/mod.rs` compile_cpu 函数（~行 697）和 compile_for_gpu 函数（~行 858）

### 归因时间
2026-06-23
