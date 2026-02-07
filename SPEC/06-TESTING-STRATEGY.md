# gllm 测试策略

> **关联需求**: REQ-TEST-001 ~ REQ-TEST-010
>
> 本文档定义 gllm 项目的测试规范、测试类型、用例格式和执行规则。

---

## 1. TEST-XXX 规范

### 1.1 命名规则

| 格式 | 说明 | 示例 |
|------|------|------|
| `TEST-{类型}-{序号}` | 测试用例标识 | `TEST-BACKEND-001` |
| 类型取值 | BACKEND, LOADER, INFERENCE, SCHEDULER, QUANT, E2E, PERF | - |

### 1.2 文档格式

每个测试用例必须包含以下注释：

```rust
/// TEST-{类型}-{序号}: {测试名称}
///
/// **关联需求**: REQ-TEST-XXX
/// **测试类型**: 正向/负向/边界/安全
/// **前置条件**: {条件描述}
///
/// **测试步骤**:
/// 1. {步骤1}
/// 2. {步骤2}
/// 3. {步骤3}
///
/// **期望结果**: {预期结果}
#[test]
fn test_name() {
    // ...
}
```

### 1.3 REQ 关联要求

| 规则 | 说明 |
|------|------|
| 每个 TEST-XXX 必须关联至少一个 REQ-XXX | 追溯需求来源 |
| 每个 REQ-TEST-XXX 必须有对应的 TEST-XXX | 确保需求被测试覆盖 |
| 关联格式 | `**关联需求**: REQ-TEST-XXX` |

---

## 2. 测试类型定义

### 2.1 正向测试

**定义**: 验证功能在正常输入下正确工作

| 要素 | 说明 |
|------|------|
| 目的 | 验证核心功能正常执行 |
| 输入 | 合法、有效的输入数据 |
| 验证 | 输出符合预期，无错误或警告 |

**示例**: CPU 后端生成文本成功

### 2.2 负向测试

**定义**: 验证功能对异常输入的错误处理

| 要素 | 说明 |
|------|------|
| 目的 | 验证错误场景被正确处理 |
| 输入 | 非法、无效、边界外输入 |
| 验证 | 返回明确错误信息，不崩溃 |

**示例**: 无效模型 ID 返回错误

### 2.3 边界测试

**定义**: 验证功能在边界条件下的行为

| 要素 | 说明 |
|------|------|
| 目的 | 验证极限值、空值、临界值 |
| 输入 | 空字符串、0、最大值、最小值 |
| 验证 | 行为符合规范，无未定义行为 |

**示例**: 空输入文本的处理

### 2.4 安全测试

**定义**: 验证安全性相关场景

| 要素 | 说明 |
|------|------|
| 目的 | 验证安全机制正确工作 |
| 场景 | 权限控制、注入防护、数据验证 |
| 验证 | 拒绝非法访问，数据安全 |

---

## 3. REQ-TEST → TEST-XXX 映射

| REQ-TEST | 描述 | TEST-XXX | 测试文件 |
|----------|------|----------|----------|
| REQ-TEST-001 | 后端覆盖测试 | TEST-BACKEND-001 ~ -003 | `test_backend_compat.rs` |
| REQ-TEST-002 | Generator 模型矩阵 | TEST-MATRIX-GEN-001 ~ -007 | `test_complete_matrix.rs` |
| REQ-TEST-003 | Embedding 模型矩阵 | TEST-MATRIX-EMB-001 ~ -010 | `test_complete_matrix.rs` |
| REQ-TEST-004 | Reranker 模型矩阵 | TEST-MATRIX-RERANK-001 ~ -003 | `test_complete_matrix.rs` |
| REQ-TEST-005 | 功能模块覆盖 | TEST-LOADER-*, TEST-INFERENCE-*, TEST-SCHED-* | 各功能测试文件 |
| REQ-TEST-006 | 量化格式测试 | TEST-QUANT-001 ~ -004 | `test_quantization.rs` |
| REQ-TEST-007 | 错误处理测试 | TEST-ERROR-001 ~ -004 | `test_error_handling.rs` |
| REQ-TEST-008 | 性能基准测试 | TEST-PERF-001 ~ -004 | `test_performance.rs` |
| REQ-TEST-009 | MoE 专项测试 | TEST-MOE-001 ~ -003 | `test_moe_routing.rs` |
| REQ-TEST-010 | 后端一致性测试 | TEST-BACKEND-001 | `test_backend_compat.rs` |

---

## 4. 真实测试铁律

### 4.1 禁止 Mock (TEST-REAL-001)

> **违反 Ω1.2**: 禁止虚假测试

| 禁止行为 | 说明 |
|----------|------|
| ❌ Mock 权重 | 使用假权重无法验证真实推理 |
| ❌ Mock 模型 | 必须使用真实模型文件 |
| ❌ 硬编码输出 | 输出必须来自真实推理 |

**正确做法**:
- ✅ 使用真实模型文件 (safetensors/GGUF/ONNX)
- ✅ 执行真实推理验证输出正确性
- ✅ 验证生成的文本有意义（非空、非乱码）

### 4.2 真实推理验证

| 验证项 | 方法 |
|--------|------|
| 生成模型 | 验证输出非空、长度合理、内容有意义 |
| Embedding | 验证向量维度正确、值域合理 |
| Reranker | 验证分数为有限浮点数、排序合理 |

---

## 5. Skip 清零规则

### 5.1 禁止 #[ignore] 标记

> **违反 Ω1.6**: 禁止跳过测试

| 场景 | 规则 | 处理方式 |
|------|------|----------|
| 网络相关测试 | ❌ 禁止 #[ignore] "需要网络" | 必须执行，网络故障是失败 |
| CUDA 测试 | ⚠️ 允许 #[ignore] "需要 CUDA" | 条件忽略，CUDA 不可用时跳过 |
| 暂未实现 | ❌ 禁止 #[ignore] "TODO" | 未实现 = 失败，必须修复 |

### 5.2 条件忽略格式

```rust
#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "Requires CUDA backend")]
fn cuda_test() {
    // ...
}
```

---

## 6. 测试执行规则

### 6.1 单元测试 vs E2E 测试

| 测试类型 | 位置 | 执行方式 | 并发 |
|----------|------|----------|------|
| 单元测试 | `src/` 内 `#[cfg(test)]` | `cargo test --lib` | ✅ 允许并发 |
| 集成测试 | `tests/` 目录 | `cargo test --test xxx` | ⚠️ 取决于 I/O |
| E2E 测试 | `tests/test_e2e.rs` 等 | `cargo test --test xxx -- --test-threads=1` | ❌ 禁止并发 |

### 6.2 E2E 测试单线程要求

> **已在 CLAUDE.md 定义**: E2E 测试涉及真实模型下载、文件 I/O，必须串行执行

```bash
# 正确：E2E 测试单线程运行
cargo test --test test_e2e -- --test-threads=1
cargo test --test test_complete_matrix -- --test-threads=1
```

### 6.3 测试文件清单

| 测试文件 | 关联 REQ | 测试类型 | 状态 |
|----------|----------|----------|------|
| `test_backend_compat.rs` | REQ-TEST-001, REQ-TEST-010 | 单元测试 | 🟢 已实现 |
| `test_e2e.rs` | REQ-TEST-001 | E2E 测试 | 🟡 需优化 |
| `test_model_matrix.rs` | REQ-TEST-002/003/004 | 单元测试 (Mock) | ⚠️ 需改为真实测试 |
| `test_complete_matrix.rs` | REQ-TEST-002/003/004 | 真实推理测试 | 🟢 已实现 |
| `test_quantization.rs` | REQ-TEST-006 | 单元测试 | 🟢 已实现 |
| `test_error_handling.rs` | REQ-TEST-007 | 负向/边界测试 | 🟢 已实现 |
| `test_moe_routing.rs` | REQ-TEST-009 | 单元测试 | 🟢 已实现 |
| `test_performance.rs` | REQ-TEST-008 | 性能测试 | 🟢 已实现 |
| `test_kv_cache.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_vllm2024.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_paged_attention.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_gguf_loader.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_intelligent_loading.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_loader_modelscope.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_pytorch_loader.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_adapters.rs` | REQ-TEST-005 | 单元测试 | 🟢 已实现 |
| `test_multiple_architectures.rs` | REQ-TEST-002 | 单元测试 | 🟢 已实现 |
| `test_multi_model_summary.rs` | REQ-TEST-002 | 单元测试 | 🟢 已实现 |
| `test_smollm2_generation.rs` | REQ-TEST-002 | 单元测试 | 🟢 已实现 |
| `test_quantization_loader.rs` | REQ-TEST-006 | 单元测试 | 🟢 已实现 |

---

## 7. 测试用例模板

### 7.1 正向测试模板

```rust
/// TEST-{TYPE}-{NUM}: {测试名称}
///
/// **关联需求**: REQ-TEST-XXX
/// **测试类型**: 正向测试
/// **前置条件**: {模型已缓存/服务可用}
///
/// **测试步骤**:
/// 1. {准备操作}
/// 2. {执行操作}
/// 3. {验证结果}
///
/// **期望结果**: {具体期望值}
#[test]
fn test_name() {
    // Given
    let input = "valid input";

    // When
    let result = operation(input);

    // Then
    assert!(!result.is_empty(), "结果不应为空");
    assert!(result.len() > MIN_LENGTH, "结果长度应合理");
}
```

### 7.2 负向测试模板

```rust
/// TEST-{TYPE}-{NUM}: {测试名称}
///
/// **关联需求**: REQ-TEST-XXX
/// **测试类型**: 负向测试
///
/// **测试步骤**:
/// 1. 提供无效输入
/// 2. 验证返回错误
///
/// **期望结果**: 返回明确的错误类型
#[test]
fn test_invalid_input() {
    let invalid_input = "";

    let result = operation(invalid_input);

    assert!(result.is_err(), "应返回错误");
    assert!(matches!(result, Err(Error::InvalidInput)), "错误类型正确");
}
```

### 7.3 边界测试模板

```rust
/// TEST-{TYPE}-{NUM}: {测试名称}
///
/// **关联需求**: REQ-TEST-XXX
/// **测试类型**: 边界测试
///
/// **测试步骤**:
/// 1. 提供边界值输入 (空/0/最大值)
/// 2. 验证处理正确
///
/// **期望结果**: 行为符合规范
#[test]
fn test_empty_input() {
    let empty = "";

    let result = operation(empty);

    // 边界情况：空输入应有明确定义的行为
    assert!(result.is_ok() || matches!(result.unwrap_err(), Error::EmptyInput));
}
```

---

## 8. 测试质量标准

### 8.1 覆盖率要求

| 覆盖类型 | 要求 |
|----------|------|
| REQ-TEST 覆盖 | 100% (每个 REQ-TEST-XXX 都有对应 TEST) |
| 测试类型覆盖 | 正向 + 负向 + 边界 |
| 关键功能 | 四类全覆盖 (正向/负向/边界/安全) |

### 8.2 断言要求

| 要求 | 说明 |
|------|------|
| 明确断言 | 每个测试有明确的断言验证 |
| 有意义断言 | 不使用 `assert!(true)` |
| 错误信息 | 断言失败时提供有用信息 |

### 8.3 真实测试要求 (TEST-REAL-002)

| 检查项 | 验证方法 |
|--------|----------|
| 生成模型 | 输出非空 + 长度合理 + 内容有意义 |
| Embedding | 维度正确 + 值域合理 (非全零) |
| Reranker | 分数为有限浮点数 |

---

## 9. 待改进项

基于审计发现，以下测试需要改进：

| 文件 | 问题 | 优先级 |
|------|------|--------|
| `test_model_matrix.rs` | 使用 Mock 权重 | P0 - 改为真实测试 |
| `test_e2e.rs` | 依赖外网下载 | P1 - 使用本地缓存 |
| 所有测试文件 | 缺少 TEST-XXX 注释 | P1 - 补充文档 |
| 所有测试文件 | 缺少负向/边界测试 | P2 - 补充覆盖 |

## 10. 跨语言对齐测试策略 (REQ-TEST-011)

> **目标**: 消除 Rust (faer/ndarray) 与 Python (numpy/torch) 矩阵布局(Row/Col-Major)差异导致的隐蔽数值错误。

### 10.1 测试流程
1. **生成基准 (Python)**: 使用 `tests/e2e_alignment/generate_golden.py` 运行 HF Transformers 模型，保存中间层(Embedding/Attention)和最终输出到 `.safetensors`。
2. **执行比对 (Rust)**: 使用 `tests/e2e_alignment/test_alignment.rs` 加载相同模型和基准数据，执行推理。
3. **断言验证**: 逐元素比对数值差异，确保在容差范围内。

### 10.2 目录结构
```
tests/e2e_alignment/
├── README.md               # 操作指南
├── generate_golden.py      # PyTorch 基准生成脚本
├── requirements.txt        # Python 依赖
├── data/                   # .gitignore (不提交)
│   └── golden_e5_small.safetensors
└── test_alignment.rs       # Rust 对齐测试
```

### 10.3 容差标准
| 精度 | 允许误差 (Abs Diff) |
|------|--------------------|
| FP32 | < 1e-5             |
| FP16 | < 1e-3             |
| INT8 | < 1 (整数位一致)    |

---

## 11. GGUF Loader 测试策略 (TEST-GGUF)

> **关联需求**: REQ-LOADER-011, REQ-LOADER-014, REQ-LOADER-019
> **关联架构**: ARCH-GGUF-PARSER
> **核心目标**: 验证 ARRAY[STRING] bug 修复

### 11.1 测试用例清单

| TEST-ID | 测试名称 | 测试类型 | 关联 REQ | 重要性 |
|---------|----------|----------|----------|--------|
| TEST-GGUF-001 | 头部解析 (magic, version) | 正向 | REQ-LOADER-011 | P0 |
| TEST-GGUF-002 | ARRAY[STRING] 解析 | 正向 | REQ-LOADER-011 | 🔴 P0 |
| TEST-GGUF-003 | Tensor info 解析 | 正向 | REQ-LOADER-011 | P0 |
| TEST-GGUF-004 | Ω1: 元数据读取（禁止默认值） | 正向 | REQ-LOADER-019 | 🔴 P0 |
| TEST-GGUF-005 | 量化类型识别 (28 种) | 正向 | REQ-LOADER-014 | P0 |
| TEST-GGUF-006 | TensorSlice 零拷贝 | 正向 | REQ-LOADER-011 | P0 |
| TEST-GGUF-007 | 无效 magic 检测 | 负向 | REQ-LOADER-011 | P1 |
| TEST-GGUF-008 | 缺失元数据处理 | 负向 | REQ-LOADER-019 | P1 |
| TEST-GGUF-009 | Tensor 边界检查 | 边界 | REQ-LOADER-011 | P1 |

### 11.2 TEST-GGUF-002: ARRAY[STRING] 解析

**关联需求**: REQ-LOADER-011
**测试类型**: 正向测试
**重要性**: 🔴 P0 (gguf-rs bug 根源)

**测试步骤**:
1. 加载包含完整 tokenizer 的 GGUF 文件
2. 读取 `tokenizer.ggml.tokens`
3. 验证 token 数量正确 (49152 个，不是 3 个)

**期望结果**:
```rust
let reader = GgufReader::open("model.gguf")?;
let tokens = reader.tokenizer_tokens()?;

// 修复 bug: 应该返回 49152 个 token，而不是 3 个
assert_eq!(tokens.len(), 49152);
assert_eq!(tokens[0], "<unk>");
assert_eq!(tokens[1], "<s>");
```

### 11.3 TEST-GGUF-004: Ω1 真实性原则

**关联需求**: REQ-LOADER-019
**测试类型**: 正向测试
**重要性**: 🔴 P0

**测试步骤**:
1. 打开 GGUF 文件
2. 验证 `architecture()` 从 `general.architecture` 读取
3. 验证缺少元数据时返回错误（不使用默认值）

**期望结果**:
```rust
let reader = GgufReader::open("model.gguf")?;

// Ω1: 必须从元数据读取
let arch = reader.architecture()?;
assert_eq!(arch, "llama");

// Ω1: 缺失元数据必须报错，不能使用默认值
let result = reader.get_metadata_u64("nonexistent.key");
assert!(result.is_none());
```

### 11.4 测试数据

| 模型 | 格式 | 用途 |
|------|------|------|
| SmolLM2-135M-Instruct-GGUF | GGUF (BF16) | ARRAY[STRING] 完整测试 |
| Mistral-7B-Instruct-GGUF | GGUF (Q4_K_M) | 量化类型测试 |

### 11.5 测试文件

```
tests/
└── test_gguf_loader.rs         # GGUF loader 单元测试
```

### 11.6 TEST-GGUF-010: 量化类型映射测试

**关联需求**: REQ-LOADER-014
**关联架构**: ARCH-GGUF-PARSER (3.4 节), gllm-kernels ARCH-QUANT-GENERIC
**测试类型**: 正向测试
**重要性**: P0

**测试步骤**:
1. 加载包含多种量化类型的 GGUF 文件
2. 验证每个 Tensor 的 GgmlDType 正确识别
3. 验证适配层类型映射正确 (GgmlDType -> gllm-kernels 类型)

**期望结果**:
```rust
let adapter = GgufAdapter::open("model.gguf")?;

// 验证量化类型识别
let token_embd = adapter.tensor_info("token_embd.weight")?;
assert_eq!(token_embd.dtype, GgmlDType::Q4_0);

// 验证类型映射到 gllm-kernels
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;
assert!(matches!(kernel_tensor.dtype, gllm_kernels::DType::Q4_0));
```

**覆盖类型**:
- P0 类型 (已实现): Q4_0, Q8_0, Q5_K
- P1 类型 (待实现): Q4_K, Q6_K, Q8_K
- P2 类型 (待实现): Q2_K, Q3_K

### 11.7 TEST-GGUF-011: 泛型约束验证

**关联需求**: REQ-LOADER-014
**关联架构**: gllm-kernels ARCH-QUANT-GENERIC
**测试类型**: 边界测试
**重要性**: P0

**测试目的**: 确保适配层不违反 gllm-kernels 泛型约束

**测试步骤**:
1. 验证 GGUF 解析器返回原始字节 + 类型标识符
2. 验证适配层负责类型映射
3. 验证 gllm-kernels 只通过泛型参数接收数据

**期望结果**:
```rust
// GGUF 解析器接口 (返回原始数据)
let slice = reader.tensor("token_embd.weight")?;
assert_eq!(slice.dtype(), GgmlDType::Q4_0);
assert!(!slice.data().is_empty()); // 原始字节

// 适配层转换
let kernel_tensor = adapter.tensor_for_kernel("token_embd.weight")?;
// gllm-kernels 只接收泛型参数，不依赖 GGUF 类型
```
