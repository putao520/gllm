# 运行时观测与自适应调度 (Runtime Observability & Adaptive Scheduling)

> **📌 SSOT**: 本文档定义了 gllm 的可观测性架构、JIT 调度策略系统与错误处理铁律。

## 1. 核心理念

为了在保证 **AOT (Pre-compiled)** 架构性能的前提下，实现类似 **JIT (Runtime-compiled)** 的灵活性，我们引入 **"对象化观测与决策系统"**。

该系统由三个核心组件构成闭环控制回路：
1.  **Observer (感知)**: 零开销采集系统状态。
2.  **Policy (决策)**: 基于状态输出 JIT 指令。
3.  **Controller (执行)**: 将指令映射为底层算子调用。

## 2. 架构设计

### 2.1 RuntimeObserver (运行时观察者)

负责实时聚合系统指标，必须是无锁且零分配的。

```rust
pub struct SystemState {
    // 资源维度（Phase 1 — 实时采集）
    pub memory_pressure: f32,       // 显存占用率 [0.0, 1.0]，从 Backend::get_memory_pressure() 采集
    pub kv_fragmentation: f32,      // KV Cache 碎片率，从 PagedScheduler 计算
    pub swap_io_rate: f32,          // 当前 Swap 带宽使用率

    // 负载维度（Phase 1 — 实时采集）
    pub waiting_queue_len: usize,   // 等待队列长度，从 ContinuousBatcher 读取
    pub current_batch_size: usize,  // 当前批次大小
    pub current_running_len: usize, // 当前运行中序列数
    pub mean_context_len: usize,    // 平均上下文长度

    // 精度维度（Phase 2 — 预留）
    pub logits_entropy: f32,        // 输出分布熵 (不确定性指标)
    pub attention_sparsity: f32,    // 注意力稀疏度 (用于优化)
}

pub trait RuntimeObserver {
    fn capture(&self) -> Result<SystemState, ObserverError>;
}
```

**采集规范**：

| 指标 | 数据源 | 采集方式 | 失败处理 |
|------|--------|---------|---------|
| `memory_pressure` | `Backend::get_memory_pressure()` | 实时调用 | 返回 `Err(ObserverError::BackendUnavailable)` |
| `kv_fragmentation` | `PagedScheduler::fragmentation()` | 计算 `(total - active - free) / total` | 返回 0.0（无碎片是安全默认值） |
| `waiting_queue_len` | `ContinuousBatcher::waiting_len()` | 直接读取 | 返回 0 |
| `current_batch_size` | `ContinuousBatcher::running_len()` | 直接读取 | 返回 0 |
| `current_running_len` | `ContinuousBatcher::running_len()` | 直接读取 | 返回 0 |
| `mean_context_len` | `ContinuousBatcher::mean_context_len()` | 计算平均值 | 返回 0 |
| `logits_entropy` | Phase 2 实现 | Shannon 熵 | 返回 0.0 |
| `attention_sparsity` | Phase 2 实现 | 稀疏度计算 | 返回 0.0 |

**铁律**：`memory_pressure` 采集失败时**禁止**返回默认值 0.0（伪装无压力），必须返回 `Err` 让调用方决策。

### 2.2 JIT Scheduling Policy (JIT 调度策略)

策略层是纯逻辑计算，不涉及 IO 或 GPU 操作。

```rust
pub struct SchedulerDecision {
    pub max_batch_size: usize,
    pub admit_new_prefill: bool,
    pub force_swap_out_count: usize,
    pub kernel_strategy: KernelStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelStrategy {
    AccuracyFirst,      // FP32 Acc, Deterministic
    ThroughputFirst,    // BF16 Acc, Aggressive Fused
}

pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}
```

### 2.3 零成本抽象实现 (Zero-Cost Abstraction)

使用 **Enum Dispatch** 避免 `Box<dyn Policy>` 的动态分发开销。

```rust
enum PolicyVariant {
    Accuracy(AccuracyFirstPolicy),
    Throughput(ThroughputFirstPolicy),
    Balanced(BalancedPolicy),
}

impl PolicyVariant {
    #[inline(always)]
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Accuracy(p) => p.decide(state),
            Self::Throughput(p) => p.decide(state),
            Self::Balanced(p) => p.decide(state),
        }
    }
}
```

### 2.4 KernelStrategy 桥接规范

`SchedulerDecision.kernel_strategy` 必须端到端传递：

```
Policy.decide() → SchedulerDecision.kernel_strategy
    → Executor 存入 GeneratorForwardConfig.kernel_strategy
    → compat 层 forward 函数接收
    → 记录日志（当前 gllm-kernels 只有 AccuracyFirst 路径）
```

| KernelStrategy | 行为 |
|---------------|------|
| `AccuracyFirst` | FP32 累加器，确定性执行，禁止近似数学函数 |
| `ThroughputFirst` | 预留接口，当前等同 AccuracyFirst，记录 `log::info!` |

## 3. 预定义策略

### 3.1 AccuracyFirstPolicy (默认/保底)

**决策矩阵**：

| 条件 | max_batch_size | admit_prefill | swap_out | kernel_strategy |
|------|---------------|--------------|---------|----------------|
| `memory_pressure > 0.9` | `current_running.max(1)` | false | `ceil(pressure * 3)` | AccuracyFirst |
| `kv_fragmentation > 0.5` | `current_running.max(1)` | false | 1 | AccuracyFirst |
| 正常 | `min(32, capacity)` | true | 0 | AccuracyFirst |

### 3.2 ThroughputFirstPolicy (冲量模式)

**触发条件**: `waiting_queue_len > 50 AND memory_pressure < 0.8`

| 条件 | max_batch_size | admit_prefill | swap_out | kernel_strategy |
|------|---------------|--------------|---------|----------------|
| 触发 | `min(256, capacity)` | true | 0 | ThroughputFirst |
| 未触发 | 降级为 AccuracyFirst 决策 | - | - | AccuracyFirst |

### 3.3 BalancedPolicy (均衡模式)

**多指标加权决策**：

| 条件 | max_batch_size | admit_prefill | swap_out | kernel_strategy |
|------|---------------|--------------|---------|----------------|
| `memory_pressure > 0.85` | `current_running.max(1)` | false | 1 | AccuracyFirst |
| `waiting_queue_len > 20 AND memory_pressure < 0.7` | `min(64, capacity)` | true | 0 | AccuracyFirst |
| 正常 | `min(48, capacity)` | true | 0 | AccuracyFirst |

## 4. Phase 4: 策略热切换

- `Executor::set_policy(PolicyVariant)` 方法
- 切换在下一个 `step()` 生效，不中断当前批次
- 默认策略: `PolicyVariant::Accuracy`

## 5. 错误处理铁律 (ARCH-ERR)

所有调度/引擎/后端代码必须遵守：

### 5.1 禁止的错误处理模式

| 模式 | 状态 | 替代方案 |
|------|------|---------|
| `Err(_)` catch-all | **禁止** | 匹配具体错误类型，或 `Err(e) => log::warn!("{e}")` |
| `let _ = fallible_op()` | **禁止** | `fallible_op()?` 传播，或 `if let Err(e) = op() { log::warn!("{e}") }` |
| `unwrap_or(default)` 掩盖错误 | **禁止** | `?` 传播，或显式 `match` 处理 |
| `expect()` 在非初始化代码 | **禁止** | 返回 `Result`，让调用方处理 |
| GPU→CPU 静默 fallback | **禁止** | `log::warn!("OOM fallback: GPU→CPU")` + 返回标记 |

### 5.2 OOM Fallback 显式化

```rust
pub struct FallbackResult<T> {
    pub value: T,
    pub fallback_used: bool,  // true = GPU OOM, fell back to CPU
}
```

- OOM fallback 发生时必须 `log::warn!("OOM fallback triggered: GPU→CPU for {operation}")`
- 返回 `FallbackResult` 让调用方感知降级

### 5.3 Backend Detection 错误传播

- `detect_backend()` 中的 `expect()` 替换为 `Result` 返回
- 探测失败返回 `Err(BackendContextError)`，不 panic

## 6. KV Cache 增量持久化 (ARCH-KV-PERSIST)

### 6.1 问题

当前 `update_kv_cache()` 只更新 `seq_len` 计数器，K/V 值在 JIT graph 内部计算后未写入 cache buffer。增量 decode 每次重新计算全部 K/V。

### 6.2 方案

JIT graph 的 MHA op 通过额外的输出指针参数将 K/V 写入外部 buffer：

```
build_decoder_layer_graph() 添加 kv_cache_ptr 输出参数
    → MHA op 计算 K/V 后写入 buffer
    → update_kv_cache() 管理 buffer 生命周期和 seq_len
    → 增量 decode 时传入已缓存的 K/V 前缀
```

**约束**：
- 只修改 gllm 侧的 graph 构建和 buffer 管理

## 7. 演进路线

1.  **Phase 1** (当前): 实现 `SystemState` 实时采集 + 错误处理修复 + KV Cache 持久化
2.  **Phase 2**: 实现 `logits_entropy` / `attention_sparsity` 采集
3.  **Phase 3**: 实现完整策略矩阵 (Accuracy/Throughput/Balanced)
4.  **Phase 4**: 实现运行时策略热切换
