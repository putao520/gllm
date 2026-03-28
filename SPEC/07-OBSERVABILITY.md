# 运行时观测与自适应调度 (Runtime Observability & Adaptive Scheduling)

> **📌 SSOT**: 本文档定义了 gllm 的可观测性架构、JIT 调度策略系统与错误处理铁律。

## 1. 核心理念 (Zero-Overhead Telemetry)

在极致的 Mega-Kernel 架构下，任何传统的**“旁路探测、CPU 定时轮询、无锁环形队列 (RingBuffer)”**都会引发致命的 PCIe 延迟与缓存击穿。
因此，gllm 的可观测性彻底抛弃外挂式的 `RuntimeObserver` 伴随线程，转而采用 **就地硬核写入（In-Place Logging）**。

其闭环构成为：
1.  **Epilogue 嗅探器 (In-Kernel Sniffer)**: 在 Mega-Kernel 算子执行的最后一条指令（Epilogue），将熵值等数据顺手算出。
2.  **物理页头烙印 (Page Header STG)**: 将由于计算衍生出的遥测数据，直接使用 `STG` (Store Global) 等写合并汇编，**硬贴**在当前 Request 正在写入的 KV Page 的前置字节（Header）里。
3.  **零编解码回调 (Zero-Cost Readback)**: 下次调度器回收/轮询页表状态时，顺便就将这些监控值读回，绝不发生二次通信。

## 2. 架构设计

### 2.1 物理页表遥测层 (Page Header Telemetry)

抛弃松散的结构体观测，将所有遥测数据强制压缩并寄生到 KV Cache 的内存版图中：

```rust
#[repr(C)]
pub struct KvPageHeader {
    // 基础管理 (8 Bytes)
    pub page_id: u32,
    pub ref_count: u32,

    // 负载与拓扑 (Phase 1 — Epilogue 自动写入)
    pub fragmentation_metric: f32,  // 该块剩余碎片的熵
    
    // 意图与安全护栏 (Phase 2 — 知识引擎注入写回)
    pub logits_entropy: f32,        // 输出分布熵（标定模型是否进入瞎猜断崖）
    pub guard_veto_flag: u32,       // 0=Safe, 1=Veto (触发硬件级熔断)
}
```

**采集硬性规范（SSOT）**：

| 指标 | 物理存放点 | 采集方式 (Kernel 执行期) | CPU 反馈成本 |
|------|-----------|-------------------------|-------------|
| `logits_entropy` | 分配给该 Request 的尾部页 | SVE/AMX 归约出 max_prob 时顺便用其计算 | 跟随 D2H (Device-to-Host) 同轴回送 |
| `guard_veto_flag`| Guardrail 专属拦截区 | 由最后置的分类小模型内核投管 | Kernel Launch 级别的分支断流 |

**铁律**：**禁止**为主线程新开任何 `std::thread` 或 `tokio::spawn` 来做性能监控！所有遥测数据只能由 Kernel 的下半场（Epilogue）生成并嵌在结果集或页表头。

### 2.2 确定性执行路由 (Deterministic Dispatch Routing)

Mega-Kernel 下不存在运行时精度决策（如 `AccuracyFirst` 与 `ThroughputFirst` 的切换）。全系统被 **TurboQuant 静态预置**为极简数学模型。调度层的决策只负责吞吐量的生命周期管理，不污染内核侧图拓扑。 

```rust
pub struct SchedulerDecision {
    pub max_batch_size: usize,
    pub admit_new_prefill: bool,
    pub force_swap_out_count: usize,
    // 删除了所有 kernel_strategy 字段！所有请求享有平等的静态内核路径。
}

pub trait SchedulingPolicy {
    fn decide(&self, state: &SystemState) -> SchedulerDecision;
}
```

### 2.3 单轨极致调度 (Single-Track Absolute Policy)

通过 `STG` 采集到的熵数据，系统唯一常态执行的是具有全局护栏保障的 `AbsolutePolicy`，去掉了原本复杂的 CPU 均衡博弈：

```rust
enum PolicyVariant {
    Absolute(AbsolutePolicy),
}

impl PolicyVariant {
    #[inline(always)]
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        match self {
            Self::Absolute(p) => p.decide(state),
        }
    }
}
```

## 3. AbsolutePolicy 护栏级决策矩阵 (SSOT Dispatch)

**决策矩阵**：

| 条件 | max_batch_size | admit_prefill | swap_out |
|------|---------------|--------------|---------|
| `memory_pressure > 0.9` | `current_running.max(1)` | false | `ceil(pressure * 3)` | 
| `kv_fragmentation > 0.5` | `current_running.max(1)` | false | 1 | 
| 正常 | `min(256, capacity)` | true | 0 | 

没有任何 `Throughput` 或 `Balanced` 的妥协路径，因为底层在遇到无效请求时直接交给**Mega-Kernel 内部硬件屏蔽掩码**来消化计算，无需上升到主机层面分流。

## 4. Phase 4: 热熔断切换 (Global DCE Handover)

如果捕获到长期静滞（长尾共识），JIT 直接在图编译上进行全局 DCE，调度层：
- Host 层仅用最简单的 `PolicyVariant::Absolute`。
- 将动态路由放权到底层 `jmp`！

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
3.  **Phase 3**: (取消) 策略矩阵 (Accuracy/Throughput/Balanced) 及运行时策略热切换已被 Architect 否决，全系统锁定在单轨 AbsolutePolicy 与 Mega-Kernel 静态块路由。
