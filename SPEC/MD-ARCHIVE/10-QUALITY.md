# 质量契约 (Quality Contract)

> **SSOT 声明**: 本文档是 gllm 测试策略、集成验证规则、Fallback 审计清单、错误处理契约、观测系统的唯一真源。

## 1. 测试策略

### 1.1 三维测试网格

后端 × 模型 × 功能 的完整覆盖矩阵。

| 维度 | 选项 |
|------|------|
| 后端 | `cpu`, `jit-cuda`（`jit-hip`/`jit-metal` 未来支持） |
| 模型类型 | `generator`, `embedding`, `rerank` |
| 功能模块 | `loader`, `inference`, `scheduler`, `quantization` |

### 1.2 命名规则

| 格式 | 说明 | 示例 |
|------|------|------|
| `TEST-{类型}-{序号}` | 测试用例标识 | `TEST-BACKEND-001` |
| 类型取值 | BACKEND, LOADER, INFERENCE, SCHEDULER, QUANT, E2E, PERF | — |

### 1.3 测试类型

| 类型 | 目的 | 输入 |
|------|------|------|
| 正向测试 | 验证核心功能正常执行 | 合法输入 |
| 负向测试 | 验证错误场景被正确处理 | 非法/无效输入 |
| 边界测试 | 验证极限值、空值、临界值 | 空字符串、0、最大值 |
| 安全测试 | 验证安全机制正确工作 | 权限控制、注入防护 |

### 1.4 测试执行规则

| 测试类型 | 位置 | 执行方式 | 并发 |
|----------|------|----------|------|
| 单元测试 | `src/` 内 `#[cfg(test)]` | `cargo test --lib` | 允许并发 |
| 集成测试 | `tests/` 目录 | `cargo test --test xxx` | 取决于 I/O |
| E2E 测试 | `tests/test_e2e_*.rs` | `cargo test --test xxx -- --test-threads=1` | **禁止并发** |

E2E 测试涉及真实模型下载、文件 I/O、CPU 推理，必须串行执行。

## 2. 真实测试铁律

### 2.1 禁止 Mock

| 禁止行为 | 说明 |
|----------|------|
| Mock 权重 | 使用假权重无法验证真实推理 |
| Mock 模型 | 必须使用真实模型文件 |
| 硬编码输出 | 输出必须来自真实推理 |
| `assert!(true)` | 无意义断言 |

**正确做法**：使用真实模型文件（safetensors / GGUF / ONNX），执行真实推理，验证输出正确性。

### 2.2 真实推理验证

| 验证项 | 方法 |
|--------|------|
| 生成模型 | 输出非空 + 长度合理 + 内容有意义 |
| Embedding | 维度正确 + 值域合理（非全零） |
| Reranker | 分数为有限浮点数 + 排序合理 |

### 2.3 Skip 清零规则

- 网络相关测试：禁止 `#[ignore]` "需要网络"，网络故障即为测试失败
- CUDA 测试：允许 `#[cfg_attr(not(feature = "jit-cuda"), ignore)]`
- 暂未实现：禁止 `#[ignore]` "TODO"，未实现即为失败

## 3. 集成验证规则

### 3.1 集成测试铁律

每个优化模块必须有集成测试证明"有模块"和"无模块"时执行行为不同。

| 铁律 | 说明 |
|------|------|
| 集成测试必须证明执行行为改变 | 单元测试不等于集成测试 |
| 集成测试禁止仅验证日志输出 | `assert!(log.contains("skip"))` 无效 |
| 集成测试禁止 Mock Executor | 必须使用真实 Executor + 真实 Graph |

### 3.2 集成测试模板

```rust
#[test]
fn test_module_integration() {
    // Baseline: 无优化
    let baseline_result = run_forward_without_optimization();

    // Optimized: 有优化
    let optimized_result = run_forward_with_optimization();

    // 证明优化改变了执行行为
    assert_ne!(
        baseline_result.execution_trace,
        optimized_result.execution_trace,
        "优化模块未改变执行行为 — 可能未接入执行路径"
    );
}
```

### 3.3 禁止的测试模式

| 禁止模式 | 原因 | 替代方案 |
|---------|------|---------|
| `assert!(log.contains("enabled"))` | 日志不等于执行 | 验证 forward 输出或执行次数 |
| 仅测试模块内部函数 | 内部正确不等于接入正确 | 测试完整 Executor 流程 |
| `assert_eq!(callback.priority(), 60)` | 只验证元数据不验证效果 | 验证回调后 Executor 行为改变 |
| Mock Executor | 绕过真实集成路径 | 使用真实 Executor + 真实 Graph |

## 4. Fallback 审计清单

### 4.1 SPEC 已授权 Fallback（5 个）

| Fallback | 授权来源 | 说明 |
|----------|---------|------|
| HF → ModelScope 下载源切换 | 下载管理 | HF 失败时自动尝试 ModelScope |
| ONNX Fusion → Atomic | 图优化 | Graph Pattern Matching 不匹配时降级 |
| HW Fusion → Standalone | 硬件约束 | 硬件约束违反时降级 |
| Reshape / Transpose NOP | 元数据操作 | 纯元数据 op 不生成指令 |
| GGUF 可选字段默认值 | GGUF 元数据 | 非必需字段缺失时使用行业标准默认值 |

### 4.2 SPEC 明确禁止的 Fallback（7 类）

| 禁止类型 | 说明 |
|---------|------|
| JIT Codegen 静默 NOP | 未实现的 op 必须返回 Err |
| Scalar 函数调用 | `scalar_ops.rs` 仅供 SymExec，运行时禁止调用 |
| PTX SM 版本 Fallback | SM < 70 必须返回 Err，禁止降级 |
| Mock / Stub 测试 | 必须使用真实模型文件 |
| 静默错误处理 | 禁止 `let _ =`, `Err(_)`, 无意义 `unwrap_or(default)` |
| 内存压力默认值 | `get_memory_pressure()` 失败必须传播错误 |
| GPU OOM → CPU 回退 | OOM 必须直接 Halt，禁止降级 |

## 5. 错误处理契约

### 5.1 消除静默失败

| 替换 | 说明 |
|------|------|
| `let _ = expr` → `?` 或 `log::warn!` | 所有返回值必须处理 |
| `Err(_)` → 具体错误匹配 | 禁止吞掉错误 |
| `unwrap_or(default)` → `?` 或显式错误 | 禁止默认值掩盖错误 |
| `expect()` → `Result` 返回 | 生产代码禁止 panic |

### 5.2 OOM Halt 截断

```rust
pub struct OomHaltError {
    pub message: String,
    pub fatal: bool,  // true = 硬件 OOM，进程必须 Halt
}
```

- 物理显存分配失败时必须 `log::error!("OOM Halt triggered")`
- 系统必须当场返回架构级硬件越界错误
- **禁止**退回到 CPU 或执行 FP16 → FP32 降级

### 5.3 Backend Detection 错误传播

- `detect_backend()` 探测失败返回 `Err(BackendContextError)`，不 panic
- 无 GPU 环境下返回 `Ok(CpuBackend)` 或 `Err`

## 6. 观测系统

### 6.1 Epilogue 遥测管线

零开销遥测通过 Epilogue（Kernel 尾部指令）寄生在计算结果上。

- **禁止**为主线程新开任何 `std::thread` 或异步任务做性能监控
- 所有遥测数据由 Kernel 的 Epilogue 生成，嵌在 KV Page Header
- 下次调度器回收页表时零反序列化读回

### 6.2 KvPageHeader 40B 设计

KvPageHeader 完整结构定义和字段说明见 `SPEC/06-RUNTIME.md` §8.3。

遥测数据质量验证规则：
- 所有字段由 Epilogue 尾段 STG 指令自动写入，禁止独立采集循环
- 下次调度器回收页表时零反序列化读回
- Epilogue 写入精度: f32 值域内的数值结果必须与标量参考实现一致（误差阈值见 `SPEC/02-HARDWARE.md` §6 精度表）

### 6.3 AbsolutePolicy 护栏

系统锁定在单轨 `AbsolutePolicy`，禁止运行时切换精度模式。

决策矩阵：

| 条件 | max_batch_size | admit_prefill | swap_out |
|------|---------------|--------------|---------|
| `memory_pressure > 0.9` | `current.max(1)` | false | `ceil(pressure * 3)` |
| `kv_fragmentation > 0.5` | `current.max(1)` | false | 1 |
| 正常 | `min(256, capacity)` | true | 0 |

核心系统中不存在 `kernel_strategy` 传参。QuantType 直接驱动 JIT 生成硬件原生内核，推理过程中无类型判断分支。

### 6.4 测试用例

| TEST-ID | 描述 | 验证内容 |
|---------|------|---------|
| TEST-OBS-001 | Epilogue 页头写入完整性 | `fragmentation_metric` + `logits_entropy` 为正确正浮点数 |
| TEST-OBS-002 | AbsolutePolicy 极限压力决策 | `memory_pressure=0.95` 时 `admit_new_prefill=false` |
| TEST-OBS-003 | 静态内核路径不可更改性 | 配置中不存在 `kernel_strategy` 字段 |
| TEST-ERR-001 | 内存压力采集失败传播 | 返回 `Err`，不返回 `memory_pressure=0.0` |
| TEST-ERR-002 | OOM Halt 验证 | 当场 Halt + 硬件越界错误，无降级 |
| TEST-ERR-003 | Backend Detection 不 panic | 无 GPU 环境返回 `Ok(Cpu)` 或 `Err` |
| TEST-ARCH-004 | KV Scatter Kernel 正确性 | scatter 与逐 head DtoD 写入 bit-exact 一致 |
| TEST-ARCH-005 | GPU 权重缓存命中 | 缓存命中后零 htod 调用，输出不变 |
| TEST-ARCH-006 | Metal KV 直写正确性 | 直写与 dtoh→repack→htod bit-exact 一致 |
| TEST-ARCH-007 | Paged Attention 三后端一致性 | 三后端 attention 输出容差 < 1e-5 |

## 7. 测试文件清单

| 测试文件 | 覆盖范围 |
|----------|---------|
| `test_e2e_embedding.rs` | Embedding E2E |
| `test_e2e_generator.rs` | Generator E2E |
| `test_e2e_reranker.rs` | Reranker E2E |
| `test_gguf_parser.rs` | GGUF 解析 |
| `test_model_config_gguf.rs` | GGUF 配置推导 |
| `test_kv_cache.rs` | KV Cache 管理 |
| `test_paged_attention.rs` | Paged Attention |
| `test_continuous_batching_flow.rs` | Continuous Batching |
| `test_client_lifecycle.rs` | Client 生命周期 |
| `test_moe_routing.rs` | MoE 路由 |
| `test_error_handling.rs` | 错误处理 |
| `test_performance.rs` | 性能基准 |
