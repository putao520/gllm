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

> **关联**: 02-ARCHITECTURE.md §13 (ARCH-EPILOGUE-FUSIONS + ARCH-EPILOGUE-EXTENDED)
> **核心原则**: 所有遥测数据均在 Epilogue 尾段寄生采集，零额外全局内存读写，零独立 Kernel Launch。
> **SSOT**: 遥测点的物理实现位置和指令级开销见 02-ARCHITECTURE.md §13.5-§13.11。

### 7.1 扩展遥测指标矩阵

| 指标 | 寄生位置 | 额外指令 | 写入目标 | 消费者 |
|------|---------|---------|---------|--------|
| **死神经元计数** | SiLU Epilogue (§13.5) | ~3 SIMD | 临时寄存器 → Gate-First 判断 | §13.1 Gate-First Skip |
| **MoE 命中计数** | TopK Epilogue (§13.6) | ~2 (atomic add) | 共享内存计数器 | §15.4 Deopt Daemon |
| **GEMM 行级范数** | GEMM Epilogue (§13.7) | ~10 SIMD | 临时寄存器 → 死神经元判断 | §13.5 前置信号 |
| **per-channel scale** | RmsNorm 规约 (§13.8) | 1 SIMD | 临时寄存器 → KV 写入 | §11.2 KIVI 量化 |
| **Attention Sink 标记** | Softmax Epilogue (§13.9) | ~2 SIMD | 临时寄存器 → Sink 保护 | §11.2 FP16 保护 |
| **Softmax 锐度** | Softmax Epilogue (§13.9) | 1 除法 | 临时寄存器 → 调度决策 | 自适应 chunk |
| **Embedding 范数** | Embedding copy (§13.10) | ~5 SIMD | 临时寄存器 → RmsNorm | §11.3 RaBitQ 初始值 |
| **残差方向余弦** | Residual Add (§13.11) | ~5 (FMA+div) | 临时寄存器 → 旁路判断 | §13.3 Early Exit |

### 7.2 KvPageHeader 扩展

> **SSOT**: KvPageHeader 最终设计见 §8.7（64B Cache-Line 对齐版）。本节仅描述 Phase 2 扩展的字段语义，不重复结构体定义。
> **设计决策**: `rms_norm_norm` 和 `per_channel_scale` 为寄存器直传信号 (Tier A)，不持久化到 PageHeader。它们在 RmsNorm Epilogue 内直接传递给下游消费者（KIVI 量化、RaBitQ 修正），无需落盘。

**Phase 2 新增字段语义**:

| 字段 | 来源 Epilogue | 语义 | 下游消费者 |
|------|-------------|------|-----------|
| `softmax_max` | Softmax (§13.9) | Attention Sink 信号 | §11.2 Sink FP16 保护 |
| `softmax_sharpness` | Softmax (§13.9) | max/sum 比值 | 自适应 chunk + 调度决策 |
| `softmax_centroid` | Softmax (§13.2) | 质心坐标 (position, entropy) | 跨 step / 跨卡 RDMA 预取 |
| `residual_delta_rho` | Residual Add (§13.3) | 跨层能量差 | §13.3 Early Exit |
| `dead_neuron_ratio` | SiLU (§13.5) | 死神经元占比 | §13.1 Gate-First Skip |
| `env_vector_snapshot` | SignalRouter (§8.4) | 写入时的 EnvVector 快照 | JIT Director 环境分布统计 |

### 7.3 采集-消费闭环

```
Epilogue 采集 (Kernel 内，寄存器级)
    │
    ├─ 死神经元计数 → Gate-First Skip 决策 → 跳过 Up/Down GEMM
    ├─ MoE 命中计数 → 共享内存原子写入 → JIT Director 轮询 → Deopt
    ├─ per-channel scale → KV Cache 写入时直接消费 → KIVI 量化
    ├─ Softmax max + 锐度 → Sink 保护 + 自适应 chunk → 调度决策
    ├─ RmsNorm ‖v‖ → PageHeader STG → RaBitQ 修正 → CPU 读回
    └─ 残差 Δρ + cosθ → Early Exit 决策 → 跳过后续层
```

### 7.4 与已有 SPEC 的交叉引用

| 本节指标 | 上游采集点 | 下游消费者 | SPEC 位置 |
|---------|-----------|-----------|----------|
| dead_neuron_ratio | SiLU Epilogue | Gate-First Skip | §13.1, §13.5 |
| MoE hit counter | TopK Epilogue | Deopt Daemon | §15.4, §13.6 |
| per_channel_scale | RmsNorm 规约 | KIVI KV 量化 | §11.2, §13.8 |
| softmax_max | Softmax Epilogue | Sink 保护 | §11.2, §13.9 |
| softmax_sharpness | Softmax Epilogue | 自适应调度 | §13.9 |
| rms_norm_norm | RmsNorm Epilogue | RaBitQ 修正 | §11.3, §13.8 |
| residual_delta_rho | Residual Add | Early Exit | §13.3, §13.11 |
| residual_cosine | Residual Add | 精确 Early Exit | §13.11 |

---

## 8. Epilogue 白嫖信号 → 环境感知变体路由 (ARCH-EPILOGUE-ENV-ROUTING)

> **关联**: 02-ARCHITECTURE.md §17 (ARCH-ENV-VARIANT), 03-DATA-STRUCTURE.md §16
> **核心使命**: 将 Epilogue 白嫖信号从"一次性消费"升级为"持续驱动算子变体选择"的运行时环境向量。

### 8.1 信号量化管道 (Signal Quantization Pipeline)

白嫖信号的连续值 (f32) 必须在 Epilogue 尾段就地量化为离散枚举，写入 `Request_State_Table` 的 `EnvVector` 字段。

```
Epilogue 连续信号 (f32 寄存器)
    │
    ▼ SignalRouter (Epilogue 内联，~3 条 SIMD)
EnvVector 枚举字段 (u8)
    │
    ▼ STG 写入 Request_State_Table[thread_block_idx].env_vector
下游算子变体选择 (下一 sub-step)
    │
    ▼ 完美哈希跳表 O(1) 查找
VariantId → jmp 到特化机器码
```

### 8.2 信号 → EnvVector 字段映射

| Epilogue 信号 | 量化指令 | EnvVector 字段 | 阈值 | 采样频率 |
|--------------|---------|---------------|------|---------|
| Softmax max/sum 比值 | `vcmpps` + `vpshufb` 查 LUT | `sharpness: SharpnessBin` | [0.1, 0.5] | 每 decode step |
| SiLU 死神经元计数 | `vpcmpd` + 计数 | `dead_neuron: DeadNeuronBin` | [10%, 50%] | 每 decode step |
| Δρ + cosθ 联合 | 2×`vcmpps` + 交叉编码 | `residual_energy: EnergyBin` | Δρ∈[0.001], cos∈[0.99] | 每 decode step |
| 调度器 phase 标记 | 枚举直传 | `phase: PhaseBin` | — | per-batch |
| batcher 统计 | 统计 + `vcmpps` | `batch_shape: BatchShapeBin` | skew > 0.3 | per-batch |

### 8.3 量化开销分析

每个 Epilogue 追加的量化指令：

| 信号 | 追加指令 | 延迟 | 寄存器消耗 |
|------|---------|------|-----------|
| sharpness | `vcmpps` + `vpshufb` (LUT) + `vpmovwb` (pack to u8) | ~3 cycles | 1 ymm/zmm |
| dead_neuron | `vpcmpd` + `vphaddd` + 比较阈值 | ~3 cycles | 1 GPR |
| residual_energy | 2×`vcmpps` + `vpshufb` | ~4 cycles | 1 ymm/zmm |
| batch_shape | Host 侧统计 | 0 (不在 kernel 内) | 0 |

**总计**: ~10 条 SIMD 指令，~10 cycles，消耗 2-3 个寄存器。完全在 Epilogue 的空闲寄存器预算内（AVX-512 有 12 个空闲 zmm，见 03-DATA-STRUCTURE.md §9.5）。

### 8.4 EnvVector 写入协议

`Request_State_Table` 新增 `env_vector` 字段：

```rust
/// 每个 Thread Block 的请求状态 (Mega-Kernel 内可见)
#[repr(C)]
pub struct RequestState {
    // 现有字段
    pub request_id: u32,
    pub layer_idx: u32,
    pub seq_len: u32,
    pub total_seq: u32,
    // ... 其他现有字段

    // 新增: 环境向量 (1 个 u32，bit-packed)
    pub env_vector: u32,
    // bit 0-1: sharpness (Sharp=0, Medium=1, Flat=2)
    // bit 2-3: dead_neuron (None=0, Low=1, High=2)
    // bit 4-5: residual_energy (Active=0, Dormant=1, Skip=2)
    // bit 6: phase (Prefill=0, Decode=1)
    // bit 7-8: batch_shape (Dense=0, Sparse=1, Mixed=2)
    // bit 9-31: reserved
}
```

**写入时序**:
- `sharpness`: Softmax Epilogue 写入
- `dead_neuron`: SiLU Epilogue 写入
- `residual_energy`: Residual Add Epilogue 写入
- `phase` + `batch_shape`: Mega-Kernel 启动前由 Host 写入 (通过 `cuMemcpyHtoD` 或 CPU memset)

**读取时序**:
- 下游算子 (RmsNorm / FFN / Attention) 在启动时通过 `LDG` 读取当前 Thread Block 的 `env_vector`
- 通过 `bfe.u32` (PTX) 或 `shr` + `and` (x86) 提取各字段
- 查找完美哈希跳表获得 VariantId
- `jmp` 到对应变体入口

### 8.5 信号级联规则

多个 Epilogue 信号可能写入同一个 `env_vector` 的不同位域。为确保一致性：

| 规则 | 说明 |
|------|------|
| **位域原子性** | 每个 Epilogue 只写入自己负责的 2-3 bit，不触碰其他位域 |
| **写后即视** | 当前 sub-step 写入的信号，同层后续 sub-step 可见（SMEM 延迟 < 30 cycles） |
| **跨层传递** | Residual Add Epilogue 写入的 `residual_energy` 影响下一层的算子变体选择 |
| **默认值** | Mega-Kernel 启动前，Host 将所有位域初始化为 0 (对应"最通用"变体) |

### 8.6 消费者矩阵

| 下游算子 | 消费的 EnvVector 字段 | 变体决策 |
|---------|---------------------|---------|
| Softmax (下一层) | `sharpness` (来自上层 Softmax) | 预取策略 / Sink 保护 |
| RmsNorm (下一层) | `residual_energy` (来自 Residual Add) | 融合模式选择 |
| FFN Gate/Up/Down | `dead_neuron` (来自 SiLU Epilogue) | Gate-First Skip 挤压比 |
| KV Write | `sharpness` + `phase` | 量化策略 + FWHT 旋转 |
| Attention | `sharpness` + `phase` | Flash 版本选择 / Paged 策略 |
| JIT Director | 全部字段 (汇总统计) | 冷热变体管理 / 热修补 |

### 8.7 KvPageHeader 最终设计 (64B Cache-Line 对齐)

综合 §7.2 和 §8.4 的需求，最终 KvPageHeader 对齐到一个 64 字节 cache line：

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

    // Phase 2 — 全链路白嫖扩展 (24 Bytes)
    pub softmax_max: f32,           // Attention Sink 信号 (§13.9)
    pub softmax_sharpness: f32,     // max/sum 比值 (§13.9)
    pub softmax_centroid: (f32, f32), // 质心坐标 (position, entropy) (§13.2)
    pub residual_delta_rho: f32,    // 跨层能量差 (§13.3)
    pub dead_neuron_ratio: f32,     // 死神经元占比 (§13.5)
    pub env_vector_snapshot: u32,   // 写入时的 EnvVector 快照 (§8.4)

    // Padding (20 Bytes → 对齐到 64B)
    _reserved: [u8; 20],
}
// static_assert!(size_of::<KvPageHeader>() == 64);
```

**设计决策**:
- `rms_norm_norm` 和 `per_channel_scale` 从 PageHeader 移除：它们是寄存器直传信号 (Tier A)，不需要持久化到 PageHeader
- 新增 `softmax_centroid`: 质心坐标用于跨 step / 跨卡 RDMA 预取 (§13.2)
- 新增 `env_vector_snapshot`: JIT Director 轮询 PageHeader 时可直接读取环境分布，无需额外采样
- 严格 64B 对齐：一个 cache line，STG 单次写入，CPU 侧零额外读开销
