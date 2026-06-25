# BUG-KNOWLEDGE.md — BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。

---

## BCE-042: 权重打包 +1.0 残差硬编码 F32 解读 — BF16/F16 权重数据损坏 ✅

| 字段 | 值 |
|------|-----|
| patternId | BCE-20260624-042 |
| title | Gemma norm residual +1.0 hetero 路径硬编码 f32 解读，BF16/F16 权重数据损坏 |
| layer | 设计 |
| codePattern | `as *mut f32` + `*v += 1.0` 硬编码 f32 字节解读做 Gemma residual +1.0，而非用 `add_one_inplace` dtype 感知函数 |
| triggerCondition | 权重文件中 q_norm/k_norm 为 BF16 或 F16 dtype（如 Gemma 4 BF16 模型） |
| detectionSignatures | literal: `as *mut f32` + `+= 1.0` in weight packing code |
| sameClassCriterion | 任何权重操作按硬编码 dtype 解读字节而非从 TensorMeta/dtype 参数推断 |
| fixTemplate | 替换为 `add_one_inplace(&mut blob[off..off+size], actual_dtype)` — dtype 感知函数 |
| regressionAssertion | BF16 q_norm 权重 +1.0 残差：`add_one_inplace` 后每元素正确，非 f32 误读 |
| 发现 | R9 代码审计，pack_observe.inc.rs L341-352 |
| 根治 | L341-352 整段替换为 `add_one_inplace(blob, norm_dtype)`，norm_dtype 从 raw_floats 解析 |

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

已根治（3 + 30 = 33 个，PSC 全部根治 ✅）：

| patternId | 位置 | 修复 |
|-----------|------|------|
| BCE-20260623-001 | gllm-kernels/compiler/mod.rs logits_end | 取 max(generate, single_pass) |
| PSC-采样scratch | gllm/abi_types.inc.rs runtime_scratchpad_bytes | 补回 sampling_bytes |
| PSC-测试 | gllm/compat/sampling.rs top_p_one_is_no_op | greedy(T=0) 替代 stochastic(T=1) |

待根治（0 个）— 全部已根治 ✅：

| # | 位置 | 模式 | 失效表现 | 状态 |
|---|------|------|---------|------|
| 1 | executor_ops.inc.rs:676 mega_compiled.unwrap_or(0) | 未编译时 scratchpad=0 | buffer overrun | ✅ |
| 2 | gpu_backend_macro.rs:131 kv_caches.get().unwrap_or(0) | KV 指针=null | GPU fault | ✅ |
| 3 | mtp_executor.rs:145 / executor_step.rs:52 logits.max_by().unwrap_or(0) | 空 logits→token 0; NaN→Equal | 错误 token | ✅ |
| 4 | request_state.rs:484 DeviceMemory Drop 只释放 Cuda/Hip | Metal/Host 变体未释放 | 内存泄漏 | ✅ |
| 5 | gllm-kernels/compiler/mod.rs GPU 路径缺少 sg_end/dwc_end | CPU/GPU 不一致 | 有 SG/DWC 时越界 | ✅ |
| 6 | gllm/convert.rs:112 / reader.rs:218 模型元数据 unwrap_or(0) | 缺失字段静默为 0 | 后续崩溃难定位 | ✅ |
| 7 | mid_layer_encode.rs:170 match dtype 只处理 F16/BF16 | FP8/INT8 走空分支 | 数据静默丢弃 | ✅ |
| 8 | hgal.rs:314 PagePayloadKind 只处理 2/5 变体 | KvContext/PromptSystem/RAG 优先级=0 | 逐出优先级错误 | ✅ |
| 9 | safetensors.rs:681 AWQ group_size fallback=128 | 非标准 AWQ 模型 | 静默错误反量化 | ✅ |
| 10 | executor_core.inc.rs:624 output_tokens[0]!=0 判断生成 | token 0 是合法 token | 合法 token 被丢弃 | ✅ BCE-20260624-001 扩展：移除哨兵，信任 generated_count |
| 11 | graph/profile.rs:170 num_experts unwrap_or(0) | MoE 配置缺失 | 静默当 dense | ✅ |
| 12 | hip_backend.rs:208 PTX cache unwrap_or(0)+.max(1024) | PTX 缓存缺失 | 1KB stub 替代正确大小 | ✅ |
| 13 | executor_api.rs:416 session position unwrap_or(0) | 错误 session_id | 位置静默重置 | ✅ |
| 14 | cpu_backend.rs:150 attention_pattern unwrap_or(0) | 配置不匹配 | 层类型错误 | ✅ |
| 15 | weight_tier.rs:96 GPU capacity unwrap_or(0) | 无 GPU | 容量=0 | ✅ |
| 16 | three_tier_swap.rs:822 StorageTier 只统计 4/6 对 | GPU↔NVMe 直接换页 | 统计缺失 | ✅ |
| 17 | hgal.rs:193 PageState 只给 Protected/Warm 加分 | Active/Standby 与 Free 同权 | 逐出优先级错误 | ✅ |
| 18 | abi_types.inc.rs:336 sampling_bytes=vocab_bytes*4 | 硬编码乘数 | 新采样策略溢出 | ✅ |
| 19 | gpu_backend_macro.rs:335 out_bytes=(N*4).min(scratch) | scratch 错误时截断输出 | 输出截断 | ✅ |
| 20 | mega_kernel_gpu.rs:627 PREFILL_CHUNK_SIZE=512 | 固定分块 | 设备差异未感知 | ✅ |
| 21 | mega_kernel_gpu.rs:636 POOL_LOCAL_CAPACITY=32 | 固定池大小 | 并发不足/浪费 | ✅ |
| 22 | batch_context.rs:24 MAX_DECODE_STEPS_OFFSET=4 | ~~固定步数上限~~ 误报：是字节偏移量，非值限制 | ~~MTP depth>4 截断~~ MTP 与 max_decode_steps 正交 | ✅ |
| 23 | mega_kernel_gpu.rs:565 BATCH_CTX_EXTENSION_SIZE=128 | 固定扩展区 | 新字段可能溢出 | ✅ |
| 24 | weight_tier.rs:47/51 容量分数硬编码 70%/60% | 固定比例 | KV 小/大时浪费/不足 | ✅ |
| 25 | executor_core.inc.rs:624 output_tokens[0]!=0 | token 0 判断 | 合法 token 丢弃（同 #10） | ✅ |
| 26 | weight_tier.rs:99 L3*100 估算主机容量 | 启发式 | 偏差大 | ✅ |
| 27 | gllm-kernels compiler/mod.rs kv_bytes=hidden*2 | 硬编码 hidden | GQA/MQA 浪费内存 | ✅ |
| 28 | gllm-kernels compiler/mod.rs activation_bytes=hidden*4 | 硬编码 F32 | BF16/F16 浪费 | ✅ |
| 29 | gllm-kernels BufferLayout 无条件分配 SG 空间 | 注释说 0 when disabled 实际 >0 | 非 SG 模型浪费 | ✅ |
| 30 | gllm-kernels mega_kernel_emit.rs hdim*4 硬编码 F32 | 与同函数其他位置不一致 | BF16/F16 偏移错误 | ✅ |

---

## BCE-20260622-001: WF SDK "exists but failed to launch" — cwd 指向不存在目录 ✅

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

## BCE-20260622-002: LSP Coupling Matrix 对非 JS/TS 语言失效 — import 路径解析缺失 ✅

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

## BCE-20260623-001: cargo test SIGSEGV — vision_forward scratchpad logits 区域不足 ✅

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

---

## BCE-20260623-004: tok==0 哨兵 — Token ID 0 被错误当作 EOS 终止符 ✅

- **patternId**: BCE-20260623-004
- **title**: Token ID 0 被错误当作 EOS 哨兵终止输出扫描
- **layer**: 设计缺陷
- **codePattern**:
  - `if tok == req.eos_token_id || tok == 0 { break; }` — 将 token ID 0 硬编码为 EOS 哨兵
  - Token ID 0 是合法 token（如 `<pad>` / `<unk>` 在许多 tokenizer 中），不应被特殊对待
- **triggerCondition**: 任何模型生成 token ID 0 的场景（如 pad token、unk token、或某些 tokenizer 的第一个 token）
- **detectionSignatures**:
  - literal: `|| tok == 0` 或 `== 0 { break }` 在 token 扫描循环中
  - structural: token 扫描循环中除 `tok == eos_token_id` 外的额外终止条件
- **sameClassCriterion**: 任何将特定 token ID（非 eos_token_id）硬编码为终止条件的代码
- **fixTemplate**: 移除 `|| tok == 0`，仅保留 `tok == req.eos_token_id` 作为唯一终止条件
- **regressionAssertion**: 构造 output 包含 token ID 0 的测试 → 0 必须被收集为合法输出 token

### 根因
`collect_results` 中 `if tok == req.eos_token_id || tok == 0 { break; }` 将 token ID 0 硬编码为 EOS 哨兵。这源于 mega-kernel 输出 buffer 初始化为 0 的实现细节——未生成的 slot 为 0，用 0 作为"无更多 token"的标记。但 token ID 0 是合法 token，此哨兵导致模型无法正确输出 token 0。

### 影响
- 任何生成 token ID 0 的模型输出被截断
- 测试中 4 个测试依赖 tok==0 哨兵行为，修复后需同步更新

### 根治
1. `src/engine/batch_executor.rs:320`: 移除 `|| tok == 0`，仅保留 `if tok == req.eos_token_id { break; }`
2. 更新 4 个依赖 tok==0 哨兵的测试：
   - `test_collect_results`: 输出数据改用 EOS=99 终止
   - `collect_results_zero_token_terminates`: 改为验证 token 0 被收集为合法 token
   - `collect_results_all_zeros_yields_empty`: EOS 改为 0（0 匹配 eos_token_id）
   - `collect_results_first_token_is_zero`: EOS 改为 0（0 匹配 eos_token_id）
   - `collect_results_one_seq_empty_generation_other_has_tokens`: seq 1 EOS 改为 0
3. 新增 `collect_results_zero_token_not_sentinel` 测试：验证 token 0 被正确收集

### 归因时间
2026-06-24

---

## BCE-20260624-001: 跨侧 dtype 硬编码 — `* 4` 假设 F32 elem_bytes ✅

- **patternId**: BCE-20260624-001
- **title**: Rust 侧 `* 4` 硬编码假设 compute dtype 为 F32，违反 ARCH-JIT-DATA-YIELDS 跨侧 dtype 独立推导铁律
- **layer**: 设计缺陷
- **codePattern**:
  - `vocab_size * 4` — 假设 logits 每元素 4 字节（F32）
  - `seq_len * hidden_size * 4` — 假设 activation 每元素 4 字节
  - `output_elems * 4` — 假设 output 每元素 4 字节
  - `DiagnosticScratchpad` 缺少 `elem_bytes` 字段，`last_token_logits()` 硬编码 `vocab_size * 4`
  - `bytes_to_f32_vec` 假设 GPU 下载数据为 F32 格式，BF16 下载后转 f32 错误
  - `sz / (hidden_size * 4)` 权重维度推导假设 F32 权重，BF16 权重推导出错
  - `output_tokens[0] != 0` tok==0 哨兵（与 BCE-20260623-004 同类）
- **triggerCondition**: 任何非 F32 compute dtype 的模型（如 BF16、F16、量化模型）使用这些代码路径
- **detectionSignatures**:
  - literal: `* 4` 在 buffer/stride/size 计算中（排除 RoPE cos/sin 的 `* 4`，RoPE 精度始终 F32）
  - structural: buffer 大小计算未使用 `elem_bytes` / `compute_dtype.size_bytes()`
  - literal: `bytes_to_f32_vec(&data)` 在 `elem_bytes != 4` 路径中
  - literal: `/ (hidden_size * 4)` 权重维度推导
  - literal: `output_tokens[0] != 0` token 0 哨兵
- **sameClassCriterion**: 任何 Rust 侧 buffer/stride/size 计算硬编码 `* 4` 而非从 compute dtype 推导 elem_bytes；任何将 dtype-aware 数据传给 F32-only 转换函数的代码
- **fixTemplate**: 用 `compute_dtype.size_bytes()` 或 `elem_bytes` 替代 `* 4`；`bytes_to_f32_vec_with_elem_bytes(&data, elem_bytes)` 替代 `bytes_to_f32_vec`；BF16→f32 转换用 `half::bf16::from_bits(bits).to_f32()`；`DiagnosticScratchpad` 增加 `elem_bytes` 字段
- **regressionAssertion**: 对任何 compute dtype: `buffer_size == count * compute_dtype.size_bytes()`; GPU 下载后 `result.len() == expected_elem_count`

### 根因
Rust 侧多处 buffer 大小/stride 计算硬编码 `* 4`（F32 elem_bytes），违反 ARCH-JIT-DATA-YIELDS 铁律"跨侧 dtype 硬编码对齐"禁令。JIT 侧已通过 `ctx.dtype.elem_bytes()` 正确感知 dtype，但 Rust 侧未同步。当 compute dtype 非 F32（如 BF16=2 bytes、F16=2 bytes）时，buffer 分配不足或 stride 错误。更深层：GPU 路径下载 BF16 字节后调用 `bytes_to_f32_vec` 将每 4 字节解释为 f32，返回半数错误值。权重维度推导 `sz / (hidden_size * 4)` 在 BF16 权重下返回维度/2。

### 影响
- `gpu_backend_macro.rs` 7 处 `* 4` 硬编码 + 4 处缺少 `let elem_bytes` 定义：GPU 路径 buffer 分配/stride 全部错误，4 处为潜伏编译错误
- `gpu_backend_macro.rs` 6 处 `bytes_to_f32_vec` 在 BF16 下载路径上返回错误元素数量
- `gpu_helpers.rs` 缺少 dtype-aware 字节→f32 转换函数
- `executor_ops.inc.rs` 5 处 `* 4` 硬编码：logits stride / output copy / rerank offset / score_tokens offset / diagnostic logits
- `executor_ops.inc.rs` 5 处 `copy_nonoverlapping` 读取 F32 但 scratchpad 可能为 BF16
- `pack_observe.inc.rs` `embedding()` 总是 `read_f32_at`，BF16 scratchpad 返回垃圾
- `pack_observe.inc.rs` `last_token_logits()`: logits row stride 硬编码 `vocab_size * 4`
- `executor_core.inc.rs` `output_tokens[0] != 0` tok==0 哨兵丢弃合法 token 0
- `executor_compile.rs` 2 处 `sz / (hidden_size * 4)` BF16 权重维度推导返回维度/2
- `mid_layer_encode.rs` 生产路径 `hidden_state: vec![0u8; hidden_size * 4]` 假设 F32
- `abi_types.inc.rs` `runtime_scratchpad_bytes()`: sampling workspace 硬编码 `vocab_bytes * 4`
- ~57 处 `DiagnosticScratchpad` 构造缺少 `elem_bytes` 字段

### 根治
1. `gpu_backend_macro.rs`: 4 处添加 `let elem_bytes = config.geometry.compute_dtype.size_bytes();`; 7 处 `* 4` → `* elem_bytes`; 6 处 `bytes_to_f32_vec` → `bytes_to_f32_vec_with_elem_bytes(&data, elem_bytes)`
2. `gpu_helpers.rs`: 新增 `bytes_to_f32_vec_with_elem_bytes(data, elem_bytes)` 函数，elem_bytes=2 时 BF16→f32 转换
3. `executor_ops.inc.rs`: 5 处 `* 4` → `* mega.elem_bytes`; 5 处 `copy_nonoverlapping` → `match elem_bytes { 4 => direct copy, 2 => BF16→f32 }`; `diagnostic_prefill_logits` 同样添加 BF16 处理
4. `pack_observe.inc.rs`: `embedding()` 添加 `match self.elem_bytes { 4 => read_f32_at, 2 => BF16→f32 }`; `last_token_logits()` 改用 `vocab_size * self.elem_bytes`
5. `executor_core.inc.rs`: 移除 `output_tokens[0] != 0` 哨兵，直接用 `generated_count`
6. `executor_compile.rs`: 2 处 `sz / (hidden_size * 4)` → `sz / (hidden_size * elem_bytes)`
7. `mid_layer_encode.rs`: `hidden_state: vec![0u8; hidden_size * 4]` → `hidden_size * compute_dtype.size_bytes()`
8. `abi_types.inc.rs`: 新增 `SAMPLING_WORKSPACE_MULTIPLIER = 4` 命名常量
9. ~57 处测试 `DiagnosticScratchpad` 构造补 `elem_bytes: 4`（测试中 F32 是合理的）
10. RoPE `* 4` 不修（RoPE cos/sin 精度始终 F32，与 compute dtype 无关）
11. SG 共享内存 `* 4` 不修（JIT 侧 SgDetect/SgInject 接口规范为 F32）
12. weight_blob 读取 `* 4` 不修（权重存储始终 F32）

### 归因时间
2026-06-24

---

## BCE-20260624-013 — 静默降级编解码不匹配 ✅

**patternId**: BCE-20260624-013
**title**: ZstdDict/NvcompAns 编解码静默降级到 LZ4
**layer**: 设计

**codePattern**:
- `CompressionCodec::ZstdDict` 空字典时 `lz4_compress(data)` fallback → 存储 ZstdDict codec tag + LZ4 数据
- `CompressionCodec::NvcompAns` CPU 不可用时 `lz4_compress(data)` fallback → 存储 NvcompAns codec tag + LZ4 数据
- `compress_weight` 用 `.ok().flatten()` 静默丢弃错误

**triggerCondition**: ZstdDict 字典未训练 / NvcompAns 在 CPU 环境

**detectionSignatures**:
- literal: `lz4_compress(data)` 在 ZstdDict/NvcompAns match arm 内
- literal: `.ok().flatten()` 在 compress_weight 函数

**sameClassCriterion**: 任何 codec 编码路径使用与 codec tag 不匹配的实际压缩算法

**fixTemplate**:
1. ZstdDict 空字典 → `Err(CodecError(...))` (NO-SILENT-FALLBACK)
2. NvcompAns 不可用 → `Err(CodecError(...))` (NO-FALLBACK)
3. `compress_weight` 用 `unwrap_or_else(|e| { log::warn!(...); None })` 替代 `.ok().flatten()`

**regressionAssertion**: compress_weight_page(ZstdDict, empty_dict) → is_err(); compress_weight_page(NvcompAns, cpu) → is_err()

**归因时间**: 2026-06-24

---

## BCE-20260624-014 — AllGather 缓冲区分配不足 ✅

**patternId**: BCE-20260624-014
**title**: AllGather 缓冲区分配 elem_count 但写入 world_size*elem_count
**layer**: 设计

**codePattern**: `vec![0.0f32; sendcount]` 然后 `all_gather_inplace(&mut buf, sendcount)` — all_gather_inplace 文档要求 buffer 容量 = world_size * sendcount

**triggerCondition**: 分布式推理，world_size > 1 时 OOB 写入

**detectionSignatures**:
- literal: `vec![0.0f32; elem_count]` 紧接 `all_gather_inplace`
- structural: buffer 分配大小不含 world_size 因子

**sameClassCriterion**: 任何 all_gather_inplace 调用的 buffer 分配大小不含 world_size 因子

**fixTemplate**: `vec![0.0f32; elem_count * world_size]` + 显式提取 world_size 局部变量

**regressionAssertion**: all_gather_inplace 调用时 buffer.len() >= world_size * sendcount

**归因时间**: 2026-06-24

---

## BCE-20260624-015 — AllGather sendcount 整数截断 ✅

**patternId**: BCE-20260624-015
**title**: buffer.len() / world_size 整数除法截断余数
**layer**: 设计

**codePattern**: `sendcount = buffer.len() / world_size` 无可整除性检查

**triggerCondition**: buffer 大小不是 world_size 整数倍时 sendcount 截断 → 数据丢失

**detectionSignatures**:
- literal: `buffer.len() / .*world_size` 无 assert
- structural: 除法前无可整除断言

**sameClassCriterion**: 任何整数除法计算分布式参数时缺少可整除性检查

**fixTemplate**: 除法前 `assert!(buffer.len() % world_size == 0, "...")`

**regressionAssertion**: 不可整除时 panic 而非静默截断

**归因时间**: 2026-06-24

---

## BCE-20260624-016 — PageAddrTable current_tier 迁移后不更新 ✅

**patternId**: BCE-20260624-016
**title**: 页面迁移完成后 addr_table.current_tier 未更新导致重复迁移
**layer**: 设计

**codePattern**: `drain_completions_and_update` 接收 `&PageAddrTable` 参数但以下划线前缀忽略，只更新 PageMetadata 不更新 PageAddrEntry

**triggerCondition**: 页面从 GpuHbm 驱逐到 CpuDram 后 addr_table 仍显示 GpuHbm → 重复驱逐；或换入后仍显示 CpuDram → 重复换入

**detectionSignatures**:
- literal: `_addr_table` 参数名（下划线前缀 = 未使用）
- structural: 迁移完成处理只更新 metadata 不更新 addr_table

**sameClassCriterion**: 任何迁移/状态变更操作只更新部分状态存储，导致状态不一致

**fixTemplate**: 移除下划线前缀，在 MigrationResult::Ok 处理中同时更新 `addr_table` 的 `entry.current_tier = done.to_tier`

**regressionAssertion**: 迁移后 addr_table[page_id].current_tier == new_tier

**归因时间**: 2026-06-24

---

## BCE-20260624-017 — poll_transfers 吞掉 KV transfer 失败 ✅

**patternId**: BCE-20260624-017
**title**: poll_transfers 失败时 log::warn 后丢弃，调用者无从得知
**layer**: 设计

**codePattern**: `future.wait()` 的 Err 分支只 `log::warn!`，不返回任何失败信息给调用者

**triggerCondition**: 分布式 KV transfer 失败（网络/NCCL 错误）

**detectionSignatures**:
- literal: `log::warn!("[poll_transfers] async transfer failed` 后无 return/push
- structural: 函数返回 `Vec<KvTransferResult>`（只有 Ok 结果，无 Err 空间）

**sameClassCriterion**: 任何异步操作结果收集只返回成功结果，失败被静默丢弃

**fixTemplate**: 返回 `Vec<Result<KvTransferResult, String>>`，让调用者决定如何处理失败

**regressionAssertion**: poll_transfers 返回值包含 Err variant，调用者必须处理

**归因时间**: 2026-06-24

---

## BCE-20260624-018 — victim_id wrapping_sub 整数下溢出 ✅

**patternId**: BCE-20260624-018
**title**: expert group ID wrapping_sub 无下界检查
**layer**: 设计

**codePattern**: `victim_id.wrapping_sub(1_000_000) as usize` — victim_id < 1_000_000 时下溢出产生巨大索引

**triggerCondition**: HGAL 返回非 expert group ID（ID < 1_000_000 基偏移）

**detectionSignatures**:
- literal: `wrapping_sub` 在 group ID 上
- structural: 减法无下界检查

**sameClassCriterion**: 任何从编码 ID 提取索引的减法缺少下界检查

**fixTemplate**: `checked_sub().expect("...")` 替代 `wrapping_sub()`

**regressionAssertion**: victim_id < base_offset 时 panic，不产生巨大索引

**归因时间**: 2026-06-24

---

## BCE-20260624-019 — Mutex poison 恢复掩盖数据不一致 ✅

**patternId**: BCE-20260624-019
**title**: Mutex poison 时 into_inner() 恢复数据继续推理，掩盖前一个线程 panic 导致的不一致
**layer**: 设计

**codePattern**: `.lock().unwrap_or_else(|e| e.into_inner())` — poison 时恢复内部数据继续使用

**triggerCondition**: 持有锁的线程在修改数据过程中 panic → 数据可能不一致 → into_inner() 恢复后继续推理产生错误结果

**detectionSignatures**:
- literal: `into_inner()` 在 Mutex lock 之后
- literal: `unwrap_or_else(|e| e.into_inner())` 或 `unwrap_or_else(|err| err.into_inner())`

**sameClassCriterion**: 任何 Mutex poison 时恢复数据而非终止操作的代码

**fixTemplate**: `.lock().expect("mutex poison — previous holder panicked, cannot continue inference")` — 推理引擎中任何 panic 都应终止推理

**regressionAssertion**: Mutex poison 时 panic 传播，不恢复数据继续推理

**归因时间**: 2026-06-24

---

## BCE-20260624-020 — MoE Custom mapping 静默 RoundRobin fallback ✅

**patternId**: BCE-20260624-020
**title**: Custom expert mapping 缺失时静默 fallback 到 RoundRobin
**layer**: 设计

**codePattern**: `mapping.get(id).copied().unwrap_or(id % world_size)` — 用户指定 Custom mapping 但未覆盖所有 expert 时，缺失项静默 RoundRobin 分配

**triggerCondition**: Custom mapping 长度 < num_experts

**detectionSignatures**:
- literal: `unwrap_or(expert_id % self.world_size)` 在 placement match 中
- structural: Option::unwrap_or 回退到不同分配策略

**sameClassCriterion**: 任何用户指定的映射/配置缺失项静默 fallback 到默认策略而非报错

**fixTemplate**: `.ok_or_else(|| format!("expert_id {} not found in Custom mapping...", id))` — Result 传播

**regressionAssertion**: Custom mapping 缺失 expert → Err，不静默 RoundRobin

**归因时间**: 2026-06-24

---

## BCE-20260624-021 — TieredCache migration plan 丢弃 no-op ✅

**patternId**: BCE-20260624-021
**title**: build_batch() 产出的 TierMigrationPlan 被 `_plan` 丢弃，tier migration 为空操作
**layer**: 设计

**codePattern**: `let _plan = coordinator.build_batch(&[], 0.5)` — 计算了迁移计划但不执行

**triggerCondition**: TieredCache 模式下有 page 需要迁移

**detectionSignatures**:
- literal: `let _plan =` 或 `let _ =` 丢弃非 trivial 返回值
- structural: 返回值包含 Vec/容器字段但被丢弃

**sameClassCriterion**: 任何计算结果被 `_` 丢弃（尤其包含待执行操作的容器）

**fixTemplate**: 执行 plan 或 log::warn! 标注为未执行（直到 scheduler 集成完成）

**regressionAssertion**: TierMigrationPlan 非空时必须至少有日志输出

**归因时间**: 2026-06-24

---

## BCE-20260624-022 — f32→usize 负值环绕 ✅

**patternId**: BCE-20260624-022
**title**: f32 转 usize 无负值保护，负 f32 环绕为接近 usize::MAX 的巨大值
**layer**: 设计

**codePattern**: `(f32_value) as usize` — f32 为负时环绕为巨大 usize → 缓冲区过分配/索引 OOB

**triggerCondition**: 算术运算产生负 f32（如 memory_pressure_ratio < 0、capacity_factor < 0、sparsity > 2.0）

**detectionSignatures**:
- literal: `as usize` 前置为 f32 表达式
- structural: f32 算术结果无范围检查直接转 usize

**sameClassCriterion**: 任何 f32→usize 转换无负值/上界保护

**fixTemplate**: `.clamp(0.0, usize::MAX as f32) as usize` 或返回 Result 传播错误

**regressionAssertion**: 负 f32 输入 → clamp 到 0 而非环绕到 usize::MAX

**归因时间**: 2026-06-24

---

## BCE-20260624-023 — GPU mem_free 错误静默丢弃 (N/A: 不存在生产代码中) ✅

**patternId**: BCE-20260624-023
**title**: `let _ = driver.mem_free()` 吞掉 GPU 内存释放错误，导致 GPU 内存泄漏不可观测
**layer**: 设计

**codePattern**: `let _ = driver.mem_free(*ptr)` 或 `let _ = backend.free_gpu_page(gpu_ptr)` — 释放 GPU 内存失败被静默忽略

**triggerCondition**: GPU 驱动返回内存释放错误

**detectionSignatures**:
- literal: `let _ =` 后跟 `mem_free`/`free_gpu_page`
- structural: 资源释放返回值被丢弃

**sameClassCriterion**: 任何资源释放错误被 `let _ =` 吞掉（GPU/CPU 内存、文件句柄、网络连接）

**fixTemplate**: Drop 中用 `if let Err(e) = ... { log::error!(...) }`；非 Drop 函数用 `?` 传播

**regressionAssertion**: GPU 内存释放失败时 log::error! 输出，不静默

**归因时间**: 2026-06-24

---

## BCE-20260624-024 — decode 热路径 .unwrap() 无上下文 panic ✅

**patternId**: BCE-20260624-024
**title**: 推理 decode 热路径中 .unwrap() 缺少诊断信息，panic 时无法定位根因
**layer**: 设计

**codePattern**: `self.eagle_config.as_ref().unwrap()` / `self.ngram_index.as_ref().unwrap()` — 推理路径 panic 无具体原因

**triggerCondition**: 配置/状态不一致导致 Option 为 None

**detectionSignatures**:
- literal: `.unwrap()` 在 `as_ref()`/`as_mut()` 之后
- structural: 推理热路径中 Option 解包无诊断

**sameClassCriterion**: 任何推理热路径中的 .unwrap() 缺少 expect 诊断

**fixTemplate**: `.expect("具体原因 — 调用什么方法修复")` 替代 `.unwrap()`

**regressionAssertion**: panic 消息包含具体原因和修复建议

---

## BCE-20260624-025 — match _ => {} 静默吞掉合法枚举值 ✅

**patternId**: BCE-20260624-025
**title**: match 通配符 `_ => {}` 静默跳过合法枚举值，无日志无错误
**layer**: 设计

**codePattern**: `match dtype { BF16 => ..., F16 => ..., F32 => ..., _ => {} }` — 通配符静默跳过量化 dtype (Q4_0/Q8_0 等)

**triggerCondition**: 量化模型触发 `_` 分支（量化 dtype 是合法值，不应被静默忽略）

**detectionSignatures**:
- literal: `_ => {}` 或 `_ => { }` 在 match 块末尾
- structural: match 枚举中通配符分支无日志/错误传播

**sameClassCriterion**: 任何 match 通配符 `_ => {}` 跳过合法枚举值且无日志

**fixTemplate**: `_ => { log::warn!("...: unsupported/unexpected value {:?}", value) }` 或 `_ => return Err(...)`

**regressionAssertion**: 触发 _ 分支时产生可观测日志

---

## BCE-20260624-026 — checked_mul().unwrap_or(usize::MAX) 产生荒谬分配大小 ✅

**patternId**: BCE-20260624-026
**title**: 整数溢出后 saturating 到 usize::MAX 产生荒谬分配大小，导致混淆 OOM
**layer**: 设计

**codePattern**: `a.checked_mul(b).unwrap_or(usize::MAX)` — 溢出时饱和到 usize::MAX，下游用于分配

**triggerCondition**: vocab_size * head_dim 等乘法溢出

**detectionSignatures**:
- literal: `unwrap_or(usize::MAX)`
- structural: checked_op 后 unwrap_or(usize::MAX) 用于大小计算

**sameClassCriterion**: 任何溢出后饱和到 usize::MAX 用于内存分配大小计算

**fixTemplate**: 返回 `Result<usize, String>` / `ok_or_else(|| format!("overflow: ..."))?`

**regressionAssertion**: 溢出时返回清晰错误而非分配 usize::MAX 字节

---

## BCE-20260624-027 — Drop impl 中 assert! 可致 double-panic abort (N/A: 不存在生产代码中) ✅

**patternId**: BCE-20260624-027
**title**: Drop trait impl 中 assert! 可在 panic-in-progress 时触发 double-panic 导致进程 abort
**layer**: 设计

**codePattern**: `impl Drop for T { fn drop(&mut self) { assert!(condition, ...) } }`

**triggerCondition**: Drop 期间 assert 失败，同时另一 panic 已在进行中

**detectionSignatures**:
- structural: Drop impl 内含 assert! / panic!

**sameClassCriterion**: 任何 Drop impl 中的 assert! 或 panic!

**fixTemplate**: Drop impl 中用 `log::error!` 替代 `assert!`；选择安全行为（如 leak）而非 panic

**regressionAssertion**: Drop impl 中不包含 assert!/panic!

---

## BCE-20260624-028 — 空 logits 静默选择 token 0 ✅

**patternId**: BCE-20260624-028
**title**: 空 logits vec 经 max_by 返回 None 后 .unwrap_or(0) 静默选择 token ID 0
**layer**: 设计

**codePattern**: `iter.max_by(...).map(...).unwrap_or(0)` — 空 iter 返回 None → 默认 0

**triggerCondition**: logits 向量为空（len=0），不是 NaN

**detectionSignatures**:
- literal: `.unwrap_or(0)` 在 max_by/map 链末端
- structural: argmax 链对空输入无前置守卫

**sameClassCriterion**: 任何 argmax 操作对空输入无守卫，静默选择默认 token

**fixTemplate**: 前置 `if logits.is_empty() { panic!("...") }` + `.expect("max_by on non-empty always returns Some")`

**regressionAssertion**: 空 logits 输入触发 panic 而非静默选 token 0

---

## BCE-20260624-029 — .take(N) 静默截断超长数据 ✅

**patternId**: BCE-20260624-029
**title**: .take(total_elements) 在源数据不足时静默截断，无错误无日志
**layer**: 设计

**codePattern**: `data.iter().take(total_elements)` 其中 total_elements 可能 > data.len()

**triggerCondition**: 部分填充的 KV cache 导致 k_data/v_data 比 total_elements 短

**detectionSignatures**:
- literal: `.take(` 后跟计算值（非编译时常量）
- structural: take 的 count 由 num_tokens * num_channels 计算，可能超过源切片长度

**sameClassCriterion**: 任何 .take(N) 中 N 可能 > source.len() 且无截断检测

**fixTemplate**: 前置 `if total_elements > data.len() { log::warn!("...: {} elements truncated, {} missing", ...) }`

**regressionAssertion**: 截断发生时产生可观测日志

**归因时间**: 2026-06-24

---

## BCE-20260624-030 — eviction_worker meta.access_count 误当 compressed_size (DEFERRED)

**patternId**: BCE-20260624-030
**title**: eviction_worker compute_importance_score 使用 meta.access_count（频率计数器）作为 compressed_size，产生错误压缩比评分
**layer**: 设计
**status**: DEFERRED — 测试用假 GPU 指针，修正评分后 DMA 触发 SIGSEGV；需先修复测试基础设施

**codePattern**: `meta.access_count` 在 eviction_worker::evict_round 中用作 `compressed_size` 参数传入 compute_importance_score

**triggerCondition**: 页面被频繁访问（access_count 高），被误判为"压缩后极小"（compression_ratio 极低），获得巨大 compression_ratio_bonus

**detectionSignatures**:
- literal: `meta.access_count` 赋值给 `compressed_size`
- structural: 频率计数器被用作字节大小

**sameClassCriterion**: 任何非字节量语义的字段被用作字节大小参数

**fixTemplate**: PageAddrEntry 新增 `compressed_bytes: Option<u32>` 字段，evict_round 优先使用该字段，fallback 到 original_bytes（ratio=1.0, 无 bonus）

**regressionAssertion**: compressed_size 参数值与页面字节数相关，与访问频率无关

**阻塞原因**: eviction 测试使用假 GPU 指针（`gpu_ptr: Some(0x1000)`），修正评分改变 evict 选中页面后 DMA 操作对假指针 SIGSEGV。需先为测试分配真实 GPU 内存或 mock DMA backend。

**归因时间**: 2026-06-24

---

## BCE-20260624-031 — ZSTD_LEN_MASK 截断 31 位压缩长度 ✅

**patternId**: BCE-20260624-031
**title**: NVMe slot 格式中 (compressed.len() as u32 & ZSTD_LEN_MASK) 截断高位，>2GB 压缩数据长度丢失导致永久页损坏
**layer**: 设计

**codePattern**: `(compressed.len() as u32 & MASK)` — usize 转 u32 截断高位

**triggerCondition**: 单页压缩后超过 2^31 字节（超大 FFN 层）

**detectionSignatures**:
- literal: `as u32 & ZSTD_LEN_MASK` 或类似位掩码截断
- structural: 压缩/编码长度用 u32 存储但源为 usize

**sameClassCriterion**: 任何长度字段从宽类型截断到窄类型无溢出检查

**fixTemplate**: 截断前检查 `compressed.len() > MASK as usize` → 返回 `MigrationResult::Failed`

**regressionAssertion**: 超大压缩数据返回明确错误而非静默截断

**归因时间**: 2026-06-24

---

## BCE-20260624-032 — u32→f32→u32 AllToAll 精度丢失 ✅

**patternId**: BCE-20260624-032
**title**: MoE AllToAll send_counts 经 f32 AllGather 往返后 >2^24 的计数值精度丢失
**layer**: 设计

**codePattern**: `u32 as f32` → AllGather → `f32 as u32` — f32 尾数仅 24 位

**triggerCondition**: send_count > 16,777,216（2^24）

**detectionSignatures**:
- literal: `as f32` 在 AllGather 通信路径
- structural: 整数量经浮点通信通道往返

**sameClassCriterion**: 任何整数量经浮点通道传输且值可能超过 2^24

**fixTemplate**: 添加 f32→u32 往返精度检测 + `log::warn!`；长期方案：实现 u32 AllGather

**regressionAssertion**: 往返差异 > 1.0 时产生可观测日志

**归因时间**: 2026-06-24

---

## BCE-20260624-033 — PromoteToDram 硬编码 codec=ZstdDict ✅

**patternId**: BCE-20260624-033
**title**: PromoteToDram 写入 addr_table 时硬编码 codec=ZstdDict，不反映实际解压后状态
**layer**: 设计

**codePattern**: `codec: CompressionCodec::ZstdDict` 在 PromoteToDram 的 or_insert_with 中

**triggerCondition**: 页面从 NVMe 提升到 DRAM 后 addr_table.codec 误标为 ZstdDict（实际已解压为 None）

**detectionSignatures**:
- literal: `codec: CompressionCodec::ZstdDict` 在 promote/decompress 路径
- structural: 解压后写入的 codec 不反映解压后状态

**sameClassCriterion**: 任何数据转换后元数据未同步更新

**fixTemplate**: 解压后设置 `codec: CompressionCodec::None`（数据已解压，无压缩格式）

**regressionAssertion**: DRAM 中页面的 codec 反映实际数据状态（解压后 = None）

**归因时间**: 2026-06-24

---

## BCE-20260624-034 — RwLock poison 静默丢弃错误 ✅

**patternId**: BCE-20260624-034
**title**: eviction_worker 中 RwLock poison 被静默丢弃（`Err(_) => return 0`），完全掩盖导致 poison 的 panic
**layer**: 设计

**codePattern**: `match lock.read() { Ok(g) => g, Err(_) => return 0 }` — 无日志无错误传播

**triggerCondition**: 另一线程持锁时 panic，锁被污染

**detectionSignatures**:
- literal: `Err(_) => return 0` 或 `Err(_) => default` 在 RwLock/Mutex read/write 之后
- structural: 锁 poison 静默丢弃

**sameClassCriterion**: 任何锁 poison 静默丢弃（无日志）

**fixTemplate**: `Err(e) => { log::error!("lock poisoned: {}", e); return 0 }`

**regressionAssertion**: 锁 poison 时产生 log::error

**归因时间**: 2026-06-24

---

## BCE-20260624-035 — TurboQuant 单标量广播到所有通道 ✅

**patternId**: BCE-20260624-035
**title**: record_turboquant_scales 将单一 per_channel_scale 广播到所有 kv_dim 通道，击败逐通道量化
**layer**: 设计

**codePattern**: `vec![per_ch_scale; kv_dim]` — 单标量扩展为全通道 scales

**triggerCondition**: 模型 K 通道幅度差异大（如 DeepSeek V3 / Qwen3）

**detectionSignatures**:
- literal: `vec![scalar; dimension]` 在量化 scales 上下文
- structural: 逐通道量化使用统一 scale

**sameClassCriterion**: 任何逐通道操作使用统一值

**fixTemplate**: 广播时 `log::warn!` 标注限制；长期：实现 per-channel telemetry

**regressionAssertion**: 广播时产生可观测日志

**归因时间**: 2026-06-24

---

## BCE-20260624-036 — sparsity 公式负值退化 ✅

**patternId**: BCE-20260624-036
**title**: 稀疏度调度公式 (1.0 - sparsity*0.5) 在 sparsity>2.0 时产生负值，clamp 到 0 导致批大小退化
**layer**: 设计

**codePattern**: `(1.0 - sparsity * factor).clamp(0.0, MAX)` — reduction_factor 可能为负

**triggerCondition**: attention_sparsity > 2.0（遥测 bug 或异常注意力模式）

**detectionSignatures**:
- literal: `1.0 -` 后跟变量乘以常数，结果可能为负
- structural: reduction factor 未限制在 [0, 1]

**sameClassCriterion**: 任何 reduction/ratio 公式未限制在合法范围

**fixTemplate**: `reduction_factor = (1.0 - sparsity * 0.5).clamp(0.0, 1.0)` 后再乘以 batch_safe

**regressionAssertion**: 极端 sparsity 输入不产生负 reduction factor

**归因时间**: 2026-06-24

---

## BCE-20260624-037 — ExpertWeight DRAM warm 分支死代码 ✅

**patternId**: BCE-20260624-037
**title**: weight_compress ExpertWeight DRAM 中 if/else 冷热分支返回相同 codec (BitPackRle)，warm 专家被过度压缩
**layer**: 设计

**codePattern**: `if is_cold { A } else { A }` — 两分支相同

**triggerCondition**: 频繁激活的 warm 专家权重在 DRAM 被压缩，精度损失

**detectionSignatures**:
- literal: if/else 两个分支返回相同值
- structural: 冷热区分存在但未产生差异化行为

**sameClassCriterion**: 任何条件分支的 then/else 分支完全相同

**fixTemplate**: warm 分支返回 `CompressionCodec::None`（SPEC §6.2：频繁激活专家不压缩）

**regressionAssertion**: warm 和 cold expert 在 DRAM 使用不同 codec

---

## BCE-20260624-038 — mutex poison 静默丢弃（同一结构体不一致） ✅

**patternId**: BCE-20260624-038
**title**: three_tier_swap 中 mutex poison 静默返回空 Vec/0，同一结构体 fault recovery 路径正确返回 Abort
**layer**: 设计

**codePattern**: `Err(_) => return Vec::new()` / `Err(_) => return 0` — 同一结构体中其他方法用 `Abort`

**triggerCondition**: coordinator 持有锁时 panic，后续 migration/defrag 调用静默跳过

**detectionSignatures**:
- literal: `Err(_) => return Vec::new()` / `Err(_) => return 0` 与 `Abort` 不一致
- structural: 同一 impl 块中 mutex lock 错误处理不一致

**sameClassCriterion**: 同一结构体中相同类型错误（mutex poison）的处理方式不一致

**fixTemplate**: 统一用 `Err(e) => { log::error!("...: {e}"); return default }` 至少记录错误

**regressionAssertion**: mutex poison 至少有 log::error! 记录

**归因时间**: 2026-06-24

---

## BCE-20260624-039 — 路由解析失败返回 false 导致跨节点读错误数据 (N/A: 仅测试代码) ✅

**patternId**: BCE-20260624-039
**title**: needs_cross_node_transfer 路由解析失败返回 false（本地页），导致失败页被当作本地页处理，可能在错误节点读数据
**layer**: 设计

**codePattern**: `Err(_) => false` — 解析失败 = 不需要跨节点传输 = 本地页

**triggerCondition**: 分布式推理中路由表损坏/不一致时，解析失败的页被当作本地页

**detectionSignatures**:
- literal: `Err(_) => false` 在路由/解析函数中
- structural: 错误路径返回"安全默认值"但实际不安全

**sameClassCriterion**: 任何路由/解析函数中错误路径返回"不需要处理"的默认值

**fixTemplate**: `Err(e) => { log::error!("...: {e}"); true }` — 保守返回需要处理

**regressionAssertion**: 路由解析失败时返回 true（需要跨节点传输），不返回 false

**归因时间**: 2026-06-24

---

## BCE-20260624-040 — usize::MAX as f32 = +inf 导致 clamp 上界无效 ✅

**patternId**: BCE-20260624-040
**title**: `(x * 3.0).ceil().clamp(0.0, usize::MAX as f32) as usize` — usize::MAX as f32 = +inf，clamp 上界无效；NaN 传播
**layer**: 设计

**codePattern**: `usize::MAX as f32` — 64 位平台上 f32 无法表示 usize::MAX，结果是 +inf

**triggerCondition**: memory_pressure 异常大或为 NaN 时，force_swap_out_count 可能为不合理大值

**detectionSignatures**:
- literal: `usize::MAX as f32`
- structural: f32 clamp 上界超过 f32 精度范围

**sameClassCriterion**: 任何 `usize::MAX as f32` 用作 clamp 上界

**fixTemplate**: 用合理业务上界替代 `usize::MAX as f32`，如 1000.0

**regressionAssertion**: NaN/Inf 输入不导致 force_swap_out_count 为不合理值

**归因时间**: 2026-06-24

---

## BCE-20260624-041 — mutex/RwLock poison 处理不完整（静默/panic） ✅

**patternId**: BCE-20260624-041
**title**: 多处 mutex/RwLock poison 静默丢弃错误或 panic，应统一用 log::error! + 防御性返回
**layer**: 设计

**codePattern**: `Err(_) => return default` 无日志 / `.lock().unwrap()` / `.unwrap()` 在 mutex 上

**triggerCondition**: 持锁线程 panic → poison → 后续调用静默失败或二次 panic

**detectionSignatures**:
- literal: `Err(_) =>` 在 Mutex/RwLock lock 结果上无日志
- literal: `.lock().unwrap()` / `.unwrap()` 在 Mutex 上（应改为 .expect() 或 if let Ok）
- structural: 同一模块中部分 poison 路径有日志、部分无

**sameClassCriterion**: 任何 Mutex/RwLock poison 处理中 `Err(_)` 无错误日志或 `.unwrap()` 无上下文

**fixTemplate**: `Err(e) => { log::error!("...lock poisoned: {e}"); 防御性返回 }` 或 `if let Ok(guard) = lock { ... } else { log::error!("..."); 防御性返回 }`

**regressionAssertion**: 所有 Mutex/RwLock poison 路径至少有 log::error! 记录

**根治实例**: executor.rs:1020, three_tier_swap.rs:512/523, mega_kernel_callback.rs:187, executor_builder.rs:756/791/807

**归因时间**: 2026-06-24

**归因时间**: 2026-06-24

**归因时间**: 2026-06-24
