# 运行时观测与自适应调度 — 实现架构 (Runtime Observability & Adaptive Scheduling)

> **📌 定位**: 本文档定义 gllm 可观测性与调度系统的**实现架构**。
> **SSOT 声明**: 需求定义（REQ-OBS-001~003, REQ-ERR-001~003）见 [01-REQUIREMENTS.md](./01-REQUIREMENTS.md)，本文档不重复需求条目，仅定义实现方案。

## 1. 核心理念 (Zero-Overhead Telemetry)

在极致的 Mega-Kernel 架构下，任何传统的**"旁路探测、CPU 定时轮询、无锁环形队列 (RingBuffer)"**都会引发致命的 PCIe 延迟与缓存击穿。
因此，gllm 的可观测性彻底抛弃外挂式的 `RuntimeObserver` 伴随线程，转而采用 **就地核心寄生（In-Place Piggybacking Logging）**。

> **🌋 物理灭顶禁令**：禁止使用任何形式的单生产者/单消费者 `RingBuffer` (如 `crossbeam_channel`, `flume`) 向主机侧汇报遥测状态。状态传递仅允许寄生于算力管道残存带宽。

其闭环构成为：
1.  **Epilogue 嗅探器 (In-Kernel Sniffer)**: 在 Mega-Kernel 算子执行的最后一条指令（Epilogue），将熵值等数据顺手算出。
2.  **物理页头烙印 (Page Header STG)**: 将由于计算衍生出的遥测数据，直接使用 `STG` (Store Global) 等写合并汇编，**硬贴机制 (Piggybacking)** 在当前 Request 正在写入的 KV Page 前置空白区 (`Header Padding`) 里，利用 DMA 写回数据段的顺风车带回主机。
3.  **零编解码回调 (Zero-Cost Readback)**: 下次调度器回收/轮询页表状态时，按固定偏移 (`Header Offset`) 零反序列化读回，绝不发生二次 PCIe 通信。

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

**采集硬性规范**：

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
pub struct AbsolutePolicy;

impl SchedulingPolicy for AbsolutePolicy {
    #[inline(always)]
    fn decide(&self, state: &SystemState) -> SchedulerDecision {
        // ... 基于 STG 矩阵直接输出唯一决策 ...
    }
}
```

## 3. AbsolutePolicy 护栏级决策矩阵

**决策矩阵**：

| 条件 | max_batch_size | admit_prefill | swap_out |
|------|---------------|--------------|---------|
| `memory_pressure > 0.9` | `current_running.max(1)` | false | `ceil(pressure * 3)` | 
| `kv_fragmentation > 0.5` | `current_running.max(1)` | false | 1 | 
| 正常 | `min(256, capacity)` | true | 0 | 

没有任何 `Throughput` 或 `Balanced` 的妥协路径，因为底层在遇到无效请求时直接交给**Mega-Kernel 内部硬件屏蔽掩码**来消化计算，无需上升到主机层面分流。

## 4. Phase 4: 热熔断切换 (Global DCE Handover)

如果捕获到长期静滞（长尾共识），JIT 直接在图编译上进行全局 DCE，调度层：
- Host 层仅用最简单的 `AbsolutePolicy`。
- 将动态路由放权到底层 `jmp`！

## 5. 错误处理实现架构 (ARCH-ERR)

> **需求 SSOT**: REQ-ERR-001~003 见 [01-REQUIREMENTS.md](./01-REQUIREMENTS.md)。本节定义实现方案。

### 5.1 OOM Halt 实现 (ARCH-ZERO-FALLBACK)

```rust
pub struct OomHaltError {
    pub message: String,
    pub fatal: bool,  // true = Hardware OOM, process must halt
}
```

- OOM 发生时必须 `log::error!("OOM Halt triggered: Physical memory limit exceeded for {operation}")`
- 架构底线：一旦触发硬件内存越界，系统必须当场截断 (Halt) 抛出严重错误，禁止框架层面自行容错。

### 5.2 Backend Detection 错误传播

- `detect_backend()` 中的 `expect()` 替换为 `Result` 返回
- 探测失败返回 `Err(BackendContextError)`，不 panic

## 6. 演进路线

1.  **Phase 1** (当前): 实现 `SystemState` 实时采集 + 错误处理修复
2.  **Phase 2**: 实现 `logits_entropy` / `attention_sparsity` 采集
3.  **Phase 3**: (取消) 策略矩阵 (Accuracy/Throughput/Balanced) 及运行时策略热切换已被 Architect 否决，全系统锁定在单轨 AbsolutePolicy 与 Mega-Kernel 静态块路由。

## 7. 全链路 Epilogue 白嫖遥测扩展 (ARCH-EPILOGUE-TELEMETRY)

> **SSOT**: 遥测点的物理实现位置、寄生采集方式、指令级开销、采集-消费闭环，均见 **02-ARCHITECTURE.md §13 (ARCH-EPILOGUE-FUSIONS + ARCH-EPILOGUE-EXTENDED)**。
> 本节仅定义观测系统的持久化数据结构。

### 7.1 KvPageHeader 扩展

Phase 2 扩展遥测字段直接写入现有 KvPageHeader 结构，保持紧凑布局：

```rust
#[repr(C)]
pub struct KvPageHeader {
    // 基础管理 (8 Bytes)
    pub page_id: u32,
    pub ref_count: u32,

    // Phase 1 — Epilogue 自动写入 (12 Bytes)
    pub fragmentation_metric: f32,
    pub logits_entropy: f32,
    pub guard_veto_flag: u32,

    // Phase 2 — 全链路白嫖扩展 (20 Bytes)
    pub softmax_max: f32,           // Attention Sink 信号 (§13.9)
    pub softmax_sharpness: f32,     // max/sum 比值 (§13.9)
    pub residual_delta_rho: f32,    // 跨层能量差 (§13.3)
    pub dead_neuron_ratio: f32,     // 死神经元占比 (§13.5)
    pub per_channel_scale: f32,     // per-channel scale (§13.8) — 仅 KIVI 路径写入
}
// 总计 40 Bytes, 对齐到 cache line padding 由分配器处理
```

**设计决策**:
- `rms_norm_norm` 和 `per_channel_scale` 为寄存器直传信号 (Tier A)，不持久化到 PageHeader
- `softmax_centroid` 质心坐标仅用于 RDMA 预取场景，作为后续扩展保留
- 保持 40B 而非 64B：减少 KV page metadata 内存开销

---
