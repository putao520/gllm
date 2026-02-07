# 运行时观测与自适应调度 (Runtime Observability & Adaptive Scheduling)

> **📌 SSOT**: 本文档定义了 gllm 的可观测性架构与 JIT 调度策略系统。

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
    // 资源维度
    pub memory_pressure: f32,       // 显存占用率 [0.0, 1.0]
    pub kv_fragmentation: f32,      // KV Cache 碎片率
    pub swap_io_rate: f32,          // 当前 Swap 带宽使用率

    // 负载维度
    pub waiting_queue_len: usize,   // 等待队列长度
    pub current_batch_size: usize,  // 当前批次大小
    pub mean_context_len: usize,    // 平均上下文长度

    // 精度维度 (Data Observability)
    pub logits_entropy: f32,        // 输出分布熵 (不确定性指标)
    pub attention_sparsity: f32,    // 注意力稀疏度 (用于优化)
}

pub trait RuntimeObserver {
    fn capture(&self) -> SystemState;
}
```

### 2.2 JIT Scheduling Policy (JIT 调度策略)

策略层是纯逻辑计算，不涉及 IO 或 GPU 操作。

```rust
pub struct SchedulerDecision {
    // 动态参数
    pub max_batch_size: usize,
    pub admit_new_prefill: bool,
    pub force_swap_out_count: usize,

    // 算子策略选择
    pub kernel_strategy: KernelStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelStrategy {
    AccuracyFirst,      // FP32 Acc, Deterministic
    ThroughputFirst,    // BF16 Acc, Aggressive Fused
    LatencyFirst,       // Minimal Batching
}

pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}
```

### 2.3 零成本抽象实现 (Zero-Cost Abstraction)

为了避免 `Box<dyn Policy>` 的动态分发开销，我们使用 **Enum Dispatch** 或 **Static Trait**。

```rust
// 静态分发示例
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

## 3. 预定义策略

### 3.1 AccuracyFirstPolicy (默认/保底)

*   **触发条件**: `memory_pressure > 0.9` OR `logits_entropy > THRESHOLD` (数值不稳定)
*   **行为**:
    *   `max_batch_size` 降级，防止 OOM。
    *   禁止新 Prefill 进入，优先保证 Decode 完成。
    *   强制使用 `KernelStrategy::AccuracyFirst`。

### 3.2 ThroughputFirstPolicy (冲量模式)

*   **触发条件**: `waiting_queue_len > 50` AND `memory_pressure < 0.8`
*   **行为**:
    *   `max_batch_size` 提升至硬件极限。
    *   开启 Aggressive Batching。
    *   使用 `KernelStrategy::ThroughputFirst`。

## 4. 演进路线

1.  **Phase 1**: 实现 `SystemState` 采集基础设施 (集成在 `executor.step()` 中)。
2.  **Phase 2**: 实现 `SchedulingPolicy` 接口与基础策略 (`AccuracyFirst`)。
3.  **Phase 3**: 实现运行时策略热切换逻辑。
