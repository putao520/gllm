# BUG-KNOWLEDGE.md — BUG 模式知识库

> 每次 BCE 根治后沉淀，避免重复归因。按 patternId 倒序排列。

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

## BCE-20260623-001: cargo test SIGSEGV — vision_forward 模块全量测试时段错误

### BUG 模式签名
- **patternId**: BCE-20260623-001
- **title**: compat::vision_forward 全量测试时 SIGSEGV（单独跑通过，全量跑崩溃）
- **layer**: 设计缺陷（可能的内存累积/unsafe 越界）
- **codePattern**:
  - 单独跑 vision_forward 530 测试全通过
  - 全量跑 44485 测试时在 vision_forward 附近 SIGSEGV
  - --skip compat::vision_forward 后 43955 测试全通过
- **triggerCondition**: 全量 cargo test --lib -- --test-threads=1/2
- **detectionSignatures**:
  - literal: `signal: 11, SIGSEGV: invalid memory reference`
  - literal: 在 `compat::vision_forward::tests::single_layernorm_executes` 附近
- **sameClassCriterion**: 任何测试在单独运行通过但在全量运行时 SIGSEGV 的模式
- **fixTemplate**: 需 GDB 定位 SIGSEGV 精确地址 + 追溯前面测试的内存泄漏/越界
- **regressionAssertion**: 全量 cargo test --lib -- --test-threads=2 无 SIGSEGV

### 根因
待查。预存 BUG（在 f7785d5 原始 commit 也存在）。

### 影响
- 全量测试无法通过 CI
- 需 --skip compat::vision_forward 绕过

### 根治
待执行完整 BCE 闭环。

### 归因时间
2026-06-23
