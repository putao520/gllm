# 运行时编排层 (Runtime Orchestration)

> **SSOT 声明**: 本文档是 gllm 调度器、KV Cache 管理、Continuous Batching、内存管理、遥测管线的唯一真源。

## 1. HGAL 调度算法总览

HGAL (Hybrid Gang-Aware LIRS) 解决 VLLM 分页调度中的两个核心问题：页面序列错乱（同一 Sequence 的页面被分散换出/换入）和 Cache Thrashing（刚 swap-in 的页面立即被换出）。

### 1.1 组成

| 组件 | 职责 |
|------|------|
| Gang-Aware Eviction | 序列组（SequenceGroup）作为换出单位，保持页面连续性 |
| LIRS Priority | 使用 IRR（Inter-Reference Recency）替代纯 LRU |
| Warm-up Protection | 新换入页面保护期，禁止立即换出 |
| Working Set Detection | 自动检测热页并锁定保护 |
| CLOCK-Pro Approximation | 低成本的 LIRS 近似实现 |

### 1.2 核心约束

| 约束 | 说明 | 违规后果 |
|------|------|----------|
| 禁止序列内页面分散 | 必须以 SequenceGroup 为单位换出 | 序列错乱、内存碎片 |
| 禁止新换入页立即换出 | Warm-up 保护期内禁止换出 | Cache Thrashing |
| 禁止纯 LRU | 必须使用 LIRS 或 CLOCK-Pro | 无法区分访问模式 |
| 零拷贝原则 | Swap 不介入生成循环数据流 | 数据回流到 CPU |

### 1.3 调度器 API

| 接口 | 功能 |
|------|------|
| `select_victim_groups(count)` → `Vec<RequestId>` | Gang-Aware 受害者选择 |
| `handle_page_fault(page_id, seq_id)` → `PageLocation` | 页面错误处理 |
| `get_batch_page_table(request_ids)` → `BatchPageTable` | 批次页表获取 |
| `update_batch(batch, results)` → `BatchAction` | 批次状态更新 |

## 2. PagedAttention 页面管理

### 2.1 PageState 状态机

| 状态 | 说明 | 转换条件 |
|------|------|----------|
| **Active** | 在 GPU 且正在使用 | 变为 Standby |
| **Standby** | 在 GPU 但未使用 | 可被换出 |
| **Swapped** | 已换出到 CPU | swap-in 后转 Warm |
| **Warm** | 刚 swap-in，保护期 | 保护期结束且访问 >= 2 次后转 Active/Protected |
| **Protected** | 工作集保护（类似 LIR） | 超过 working_set_window 未访问则降级 |

**Warm-up 防护盾**: 新分配页面在 `warmup_duration = 100ms` 且被访问 >= 2 次之前，绝对不可被换出。

### 2.2 Gang-Aware 换出

Eviction 对象不能是单一离散 Page，必须是 SequenceGroup（序列组整体换出）。禁止破坏同一序列组中不同序列的 KV 数据完整性。

**受害者选择算法**:
1. 排除 Warm、Protected、Pinned 状态的页面
2. 对每个序列组计算 LIRS 优先级分数
3. 按优先级从低到高排序
4. 选择最少数量的序列组满足所需页面数

**优先级公式**:

```
priority(group) = time_penalty + recency_penalty - freq_bonus - pin_bonus
```

### 2.3 Working Set 自动检测

检测周期: 每 N 个 generation tick。

| 条件 | 阈值 |
|------|------|
| 访问频率 | `access_count >= hot_threshold` |
| 最近访问 | `now - last_access < working_set_window` |

保护解除: 热页在 `working_set_window` 时间内未被访问则解除保护。

### 2.4 CLOCK-Pro 近似实现

| 结构 | 说明 |
|------|------|
| `clock_hand` | 扫描指针位置 |
| `cold_pages` | 冷页面候选集合 |
| `hot_pages` | 热页面集合（类似 LIR） |
| `test_pages` | 测试页面（检测访问模式变化） |

扫描算法: 从 clock_hand 开始，被访问的页面 cold → hot 或保持 hot，未被访问的 hot → cold 或保持 cold。

## 3. Continuous Batching

### 3.1 ContinuousBatcher

```rust
pub enum BatchAction {
    /// 继续处理当前序列
    Continue,
    /// 序列已完成生成
    Complete,
    /// 序列暂停（内存不足）
    Pause,
}
```

ContinuousBatcher 在每个 `step()` 中动态管理 active sequences:
1. 执行当前 batch forward
2. 对每个序列检查 `BatchAction`（EOS → Complete，内存不足 → Pause）
3. 完成的序列移出，释放 KV 页面
4. 新序列通过 `admit_waiting()` 加入

### 3.2 企业级死锁防护（drain 模式）

**禁止 `while let` 无限循环**: 当所有序列都无法分配内存时，`while let` 循环会导致 CPU 100%。

**解决方案** (drain 模式):

```rust
fn admit_waiting(&mut self, scheduler: &mut PagedScheduler) {
    // 1. drain(..) 一次性清空等待队列
    let waiting_sequences: Vec<_> = self.waiting.drain(..).collect();
    // 2. 尝试 admit 每个序列
    for mut sequence in waiting_sequences {
        match scheduler.add_sequence(sequence.to_sequence_group()) {
            Ok(()) => { self.running.insert(sequence.id, sequence); }
            Err(_) => { self.waiting.push_back(sequence); }  // 放回末尾，不在本次调用中重试
        }
    }
}
```

关键设计: `drain(..)` 一次性清空等待队列避免迭代中修改；失败序列放回末尾下次 build_batch 重试；本次调用不重试避免无限循环。

## 4. KV Cache 管理

### 4.1 Paged KV Cache 物理布局

KV Cache 按 PagedAttention 方式管理:

```
全局布局: [K_all_layers | V_all_layers]
每半:      [layer][head][seq_pos][head_dim]
页面大小:  page_size tokens (默认 128)
```

每个 token 在每层占用 `2 × num_kv_heads × head_dim × elem_bytes` 字节的 KV 空间。

### 4.2 KvPrefixIndex 前缀复用

用于无 Session ID 场景下查找最长可复用 token 前缀。

```rust
pub struct KvPrefixIndex {
    pub root: TrieNode,
}

pub struct TrieNode {
    pub children: HashMap<TokenId, TrieNode>,
    pub page_ref: Option<PageRef>,
}
```

**集成约束**:
1. `GlobalMemoryManager` 负责维护索引生命周期，与页表一致更新
2. `prepare_prefill_with_auto_reuse(request_id, tokens)` 必须先做最长前缀匹配
3. 前缀命中后采用虚拟页映射复用语义，禁止直接共享可写物理页

### 4.3 SessionKvCache 会话级确定性复用

用于 AI 编程场景的确定性 append 复用。

> **数据结构 SSOT**: `SessionKvCache` 完整字段定义见 `03-DATA-STRUCTURE.md §8.2`。

**集成约束**:
1. `claim_session_prefix` 只能 claim `finalized_position` 范围内页面
2. `finalize_session_tokens` 只能单调递增
3. 禁止越界复用

### 4.4 KvPipeline 双管线隔离

```rust
pub enum KvPipeline {
    Conversation,  // 跨轮保留
    Working,       // 轮次结束可回收
}
```

| 管线 | 语义 | 生命周期 |
|------|------|----------|
| `Conversation` | 主对话上下文 | 跨轮保留 |
| `Working` | Thinking/Reasoning 临时上下文 | `prepare_next_turn` 全量释放 |

**隔离约束**: 页表、换出策略、预取逻辑均按 pipeline 维度隔离。Working 管线的释放不影响 Conversation 管线。

### 4.5 Dual-Track 量化池

```rust
pub struct DualTrackMemoryPool {
    pub main_pool: BlockAllocator,      // 3-4 bit 主数据流
    pub _xnor_pool: BitsetAllocator,    // 1-bit QJL 残差掩码
}
```

显存被物理隔离为双轨通道。多卡同步时仅需传输原 FP16 内存量纲的 25%（4x 压缩）。

## 5. GlobalMemoryManager

### 5.1 三级 Tier

> **跨 SPEC Tier 映射**: 本 Tier 是 `GlobalMemoryManager` 全局容量管理的 SSOT。21-WEIGHT-PAGING `WeightTier` 和 22-PAGE-COMPRESSION `StorageTier` 是同一物理层级的领域视图：
>
> | 本 Tier | 21-WEIGHT-PAGING `WeightTier` | 22-PAGE-COMPRESSION `StorageTier` | 物理位置 |
> |---|---|---|---|
> | `L1` | `DeviceLocal` | `GpuHbm` | GPU HBM |
> | `L2` | `HostLocal` | `CpuDram` | CPU DRAM |
> | `L3` | `DiskMmap` | `Nvme` | NVMe 磁盘 |

| Tier | 位置 | 用途 |
|------|------|------|
| L1 | GPU HBM | 热路径计算数据（KV Cache、激活值） |
| L2 | CPU RAM | 冷 KV Cache 换出目标 |
| L3 | 磁盘 | JIT 编译缓存（7 天 TTL） |

### 5.2 页面分配/释放 API

| 接口 | 功能 |
|------|------|
| `allocate_page_in_pipeline(pipeline, tier)` → `PhysicalId` | 在指定管线和 Tier 分配页面 |
| `plan_prefill(prompt_tokens, chunk_size)` → `PrefillPlan` | Prefill 页面规划 |
| `prepare_prefill_with_auto_reuse(request_id, tokens)` → `PrefillPlan` | 带前缀自动复用的 Prefill 规划 |
| `prepare_next_turn(session_id)` | 释放 Working 管线，保留 Conversation 管线 |

### 5.3 Unified Virtual Page

废弃按功能划分的 KvCacheConfig / MoESharedWeight，系统唯一物理原子级驻留容器:

```rust
pub struct UnifiedVirtualPage {
    pub page_id: VirtualPageId,
    pub payload_kind: PagePayloadKind,  // KvContext / ExpertWeight / PromptSystem / KnowledgeRAG
    pub residency: MemoryResidency,
    pub quant_type: QuantType,
}
```

无论内容是什么，对 GlobalMemoryManager 而言仅是一个固定尺寸的极化数组，使用同一套 LIRS 页面置换与锁定状态机。

## 6. Chunked Prefill + Adaptive Chunking

### 6.1 交织调度

将长 Prefill 切成固定大小的物理 Chunk，与 Decode Token 交织塞进同一个 Batch。Decode 请求永远零等待。

```
Batch: [Prefill-64, Decode-1, Decode-1, Decode-1, Decode-1, Prefill-64, ...]
```

**精度隔离壁**: Prefill Chunk 的 Softmax 分布平坦，Decode Token 的 Softmax 尖锐。在 Attention 阶段，两者必须物理分轨调度到不同 Thread Block 组和 SMEM 分区执行。FFN 阶段允许合流。

### 6.2 Batch Composition Token Budget

```
total_budget = max_batch_tokens × memory_pressure_ratio
decode_budget = min(decode_ready_count, floor(total_budget × 0.6))
prefill_budget = total_budget - decode_budget
```

**decode_ratio_cap = 0.6**: decode 最多占 60% 预算，保证 prefill 持续进展。

### 6.3 Batch Composition 五步流程

1. 收集 ready decode tokens，按 BatchOrderPolicy 排序
2. 填充 decode slots 直到 decode_budget 用尽
3. 计算 prefill_budget
4. 从 prefill_queue 填入 chunks，按 AdaptiveChunkSize 切分
5. 生成 BatchManifest（每个 slot 的 type、token_range、compact_required）

### 6.4 AdaptiveChunkPolicy

Chunk 大小由三个维度动态决定:

```rust
fn adaptive_chunk_size(
    l1_available_ratio: f32,    // L1 可用页比例
    concurrent_reqs: usize,     // 并发请求数
    remaining_prefill_tokens: usize,  // prompt 剩余 tokens
) -> usize  // 映射到最近的 Golden Size
```

Chunk size 限定为 Golden Size 集合（非任意值），编译为独立 stride 常量。

### 6.5 Compact 决策模型

**触发条件** (必须同时满足):
1. `waste_ratio > 0.25`
2. `active_count >= min_compact_threshold` (默认 4)
3. Compact 发生在 GEMM op 级别，禁止在 Attention op 上（memory-bound，compact 不节省带宽）

**开销模型**:
```
compact_cost = 2 × active_count × elem_size × cache_line_latency
saved_flops  = waste_ratio × total_flops
decision     = compact_cost < saved_flops × flops_to_mem_ratio
```

### 6.6 BatchManifest

```rust
pub struct BatchManifest {
    pub slots: Vec<BatchSlot>,
    pub total_decode_tokens: usize,
    pub total_prefill_tokens: usize,
    pub compact_required: bool,
    pub waste_ratio: f32,
}

pub enum SlotType {
    Decode,
    PrefillChunk { chunk_offset: usize, chunk_len: usize, remaining_after: usize },
}

pub struct BatchSlot {
    pub request_id: RequestId,
    pub slot_type: SlotType,
    pub token_range: (usize, usize),
    pub priority: f32,
    pub sub_batch_key: SubBatchKey,
}
```

### 6.7 SplitFuse 废弃声明

SplitFuse 混批路径已永久移除。`enable_splitfuse` 配置字段锁定为 `false`。

## 7. BatchOrderPolicy

```rust
pub enum BatchOrderPolicy {
    StrictRequestIdOrder,  // 默认：确定性优先
    FifoOrder,             // FIFO 排序
}
```

**约束**: 默认策略为 `StrictRequestIdOrder`。吞吐优先重排会改变浮点规约顺序，破坏可复现性。`Executor::run_batch_forward` 需校验输入序列严格单调递增。

## 8. 遥测管线

### 8.1 零额外计算原则

**禁止**: 独立的旁路探测、CPU 定时轮询、无锁环形队列（RingBuffer）。所有遥测数据只能由 Kernel 的下半场（Epilogue）生成并嵌在结果集或 KV Page Header。

### 8.2 采集 → 传输 → 消费闭环

**采集** (Epilogue, 寄存器内, ~3-10 条 SIMD):
- Softmax Entropy + Centroid + max → 预取 + Sink 检测
- RmsNorm per-channel scale → KIVI K 量化
- Residual Delta_rho → Early Exit 决策
- SiLU 死神经元计数 → Gate-First Skip
- MoE Gate 命中计数 → Deopt 信号

**传输** (Epilogue 尾段 STG 指令):
- Epilogue 尾段的 STG 指令 → 写入 KV Page Header padding bytes
- 宿主机低频轮询（后台 Daemon，不在热路径）
- L1i 影响: 零 — STG 不增加指令足迹

**消费**:
- JIT Director Daemon: 冷专家零命中 → Hot JMP 物理封杀
- Block Routing: Delta_rho < epsilon → Thread Block Thread Exit
- Spec Scheduling: Entropy 低 → 启用推测解码
- Adaptive Chunking: Softmax 锐度 → 动态 chunk size

### 8.3 KvPageHeader (40B)

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
    pub softmax_max: f32,
    pub softmax_sharpness: f32,
    pub residual_delta_rho: f32,
    pub dead_neuron_ratio: f32,
    pub per_channel_scale: f32,
}
```

### 8.4 Observer 观测

**BasicObserver**: 系统状态实时采集。不开启独立线程，不在热路径分配内存。

**SystemState**: 调度决策的输入快照，包含 memory_pressure、kv_fragmentation、running_count 等。

### 8.5 AbsolutePolicy 护栏

单轨调度策略，去掉了 Accuracy/Throughput/Balanced 妥协路径。底层在遇到无效请求时交给 Mega-Kernel 内部硬件屏蔽掩码消化，无需上升到主机层面分流。

| 条件 | max_batch_size | admit_prefill | swap_out |
|------|---------------|--------------|---------|
| `memory_pressure > 0.9` | `current_running.max(1)` | false | `ceil(pressure * 3)` |
| `kv_fragmentation > 0.5` | `current_running.max(1)` | false | 1 |
| 正常 | `min(256, capacity)` | true | 0 |

### 8.6 禁止事项

| 禁止 | 原因 |
|------|------|
| 主线程新开 `std::thread` 或 `tokio::spawn` 做性能监控 | Epilogue 寄生是唯一采集路径 |
| RingBuffer (crossbeam_channel / flume) 向主机侧汇报 | PCIe 延迟与缓存击穿 |
| 运行时策略热切换 | 全系统锁定在 AbsolutePolicy |

## 9. JIT 编译缓存协议

### 9.1 核心铁律

JIT 编译只允许在两个时间点发生:
1. **模型加载时** — 静态形状已知，一次性编译
2. **Autotuning 窗口期** — 首次推理前 3-5 秒

推理热路径中禁止出现任何编译行为。

### 9.2 缓存粒度

缓存粒度 = 全模型单体计算图 (Whole-Model Graph)。CPU 和 GPU 后端共享完全一致的执行图。禁止算子级/层级分离缓存。

**缓存键** = 模型结构签名，不包含运行时动态维度:

```rust
ModelArchKey {
    model_id: String,
    backend: BackendKind,
}
```

### 9.3 动态维度处理

| 维度性质 | 编码方式 | 绑定时机 |
|----------|---------|---------|
| 静态 (hidden_size) | `SymDim::Concrete(value)` | 编译时 |
| 动态 (seq_len, total_seq) | `SymDim::Symbolic("name")` | launch 时 ShapeBinding |

动态维度通过 ShapeBinding 运行时绑定，不触发重编译。

### 9.4 三级缓存

| 级别 | 存储 | TTL | 作用 |
|------|------|-----|------|
| L1 | 模型级内存 | 模型生命周期 | 单模型共享实例 |
| L2 | 全局内存 LRU | 进程生命周期 | 跨模型复用 |
| L3 | 磁盘 (`~/.gllm/jit_cache/`) | 7 天 | 跨进程复用 |

Debug 模式（`cargo test`）L3 磁盘缓存完全禁用。Release 模式正常工作，启动时自动清理超 7 天的缓存。

## 10. 配置参数汇总

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `page_size` | usize | 128 | 每页 token 数 |
| `total_pages` | usize | 2048 | 总页面数 |
| `max_batch` | usize | 8 | 最大批大小 |
| `max_batch_tokens` | usize | 4096 | 每步最大 token 数 |
| `warmup_duration` | Duration | 100ms | Warm-up 保护期 |
| `working_set_window` | Duration | 1s | 工作集时间窗 |
| `hot_threshold` | usize | 3 | 热页访问阈值 |
| `lir_ratio` | f32 | 0.3 | LIR 页面比例 |
| `chunk_size` | usize | 64 | Prefill Chunk 大小 |
| `decode_slots` | usize | 8 | 每批 Decode 插槽 |
| `max_chunks_per_batch` | usize | 4 | 每批最大 Chunk 数 |
| `decode_ratio_cap` | f32 | 0.6 | Decode 占比上限 |
| `min_compact_threshold` | usize | 4 | Compact 最小 batch |

## 11. SEQ 分布直方图 (SeqHistogram)

### 11.1 设计目标

在推理热路径中以极低开销 (<100ns/step) 采集当前 batch 的 `seq_len` 分布，为 GoldenBucketRegistry 的 `evolve()` 提供输入。所有计数器使用 `AtomicU64`，无锁并发更新，不在 Mega-Kernel 内引入条件分支。

### 11.2 数据结构

```rust
pub struct SeqHistogram {
    buckets: Vec<SeqBucket>,
    window_size: usize,
    total_samples: AtomicU64,
}

pub struct SeqBucket {
    pub start: usize,       // 区间闭左端
    pub end: usize,         // 区间闭右端
    pub count: AtomicU64,   // 无锁并发计数器
}

pub struct HistogramSnapshot {
    pub buckets: Vec<(usize, usize, u64)>,
    pub total_samples: u64,
    pub top_k: Vec<(usize, usize, u64)>,
}
```

### 11.3 分档规则

按 2 的幂分档，每个 bucket 的区间为 `(2^(n-1), 2^n]`：

| Bucket 索引 | 区间 | 示例 seq_len |
|-------------|------|-------------|
| 0 | [0, 1] | 0, 1 |
| 1 | (1, 2] | 2, 3 |
| 2 | (2, 4] | 4, 5, 6, 7 |
| 3 | (4, 8] | 8, 9, ..., 15 |
| ... | ... | ... |
| n | `(2^(n-1), 2^n]` | |
| 12 | `(2048, 4096]` | |

最后一个 bucket 扩展到 `max_seq`，确保所有 `seq_len` 均可映射。

### 11.4 核心算法

**`find_bucket(seq_len)`** — O(1) bucket 定位：

使用前导零计数 (`lzcnt`) 计算 `floor(log2(seq_len))`，直接映射到 bucket 索引。x86_64 上编译为单条 `lzcnt` 指令。

```
log2 = BITS - seq_len.leading_zeros() - 1
bucket_index = log2.min(buckets.len() - 1)
```

**`record(seq_len)`** — O(1) 无锁更新：

1. `find_bucket(seq_len)` 定位 bucket
2. `bucket.count.fetch_add(1, Relaxed)` 原子递增
3. `total_samples.fetch_add(1, Relaxed)` 原子递增

内存序使用 `Relaxed`，因为直方图仅用于统计决策，不参与同步。

### 11.5 滑动窗口与衰减

**`decay(factor)`** — 遗忘效果：

遍历所有 bucket，将计数器乘以 `factor`（如 0.9），同时衰减 `total_samples`。由 JIT Director Daemon 周期性调用（不在热路径），实现滑动窗口的时间衰减。

### 11.6 快照与查询

| 方法 | 说明 |
|------|------|
| `snapshot()` | 读取全部 bucket 计数器，返回 `HistogramSnapshot`（含 top_k 热门区间，默认 top 10） |
| `top_k(k)` | 返回最热门 k 个区间的 `(start, end, count)` |
| `is_gap(seq_len, tolerance)` | 检测 `seq_len` 是否远离 bucket 中心（`distance > tolerance`） |
| `gap_hit_rate(tolerance)` | 估算 bucket 边缘缝隙的累计命中率 |

### 11.7 与 GoldenBucketRegistry 的集成

`SeqHistogram` 是 `GoldenBucketRegistry::evolve()` 的唯一输入源：

1. JIT Director Daemon 周期性调用 `histogram.snapshot()`
2. `evolve(snapshot)` 扫描高流量缝隙区间（`find_high_traffic_gaps`）
3. 在缝隙中创建新中间态 GoldenSize（命中率 > 5%）
4. 详见 §12.4 运行时演化

## 12. 黄金装筒规则 (Golden Bucket)

### 12.1 核心原则

**严禁预设硬编码数组**：禁止使用 `[128, 512, 1024, 2048]` 等静态 Bucket。所有黄金尺寸从硬件物理探测推导，由 `ProbeResult.spill_points` 决定。

**零退化原则**：禁止 Padding 补零，使用 Ragged Compaction（§6.5）。

### 12.2 数据结构

```rust
pub struct GoldenSize {
    pub seq_len: usize,
    pub register_efficiency: f32,
    pub smem_efficiency: f32,
    pub l2_hit_rate: f32,
    pub performance_score: f32,
}

pub struct GoldenBucketRegistry {
    golden_sizes: Vec<GoldenSize>,
    constraints: CompilerConstraints,
    hit_stats: BTreeMap<usize, u64>,
    max_buckets: usize,         // 默认 32
    zombie_threshold: f64,       // 默认 0.001 (<0.1%)
}

pub enum EvolveDecision {
    InsufficientData,
    NoEvolutionNeeded,
    CapacityLimitReached { current: usize, max: usize },
    Evolved { new_bucket_count: usize },
}
```

### 12.3 GoldenSize 评分公式

```
performance_score = 0.4 × register_efficiency
                  + 0.3 × smem_efficiency
                  + 0.3 × l2_hit_rate
```

| 权重 | 指标 | 说明 |
|------|------|------|
| 0.4 | `register_efficiency` | 寄存器利用率 (0.0-1.0)，spill point 前最高 |
| 0.3 | `smem_efficiency` | 共享内存利用率 (0.0-1.0)，tile 面积 / SMEM 容量 |
| 0.3 | `l2_hit_rate` | L2 缓存命中率 (0.0-1.0)，距离 L2 thrash 阈值越近越低 |

### 12.4 生命周期

| 阶段 | 触发 | 行为 |
|------|------|------|
| **Load-Time** | 模型加载 | `from_probe_results(probe, constraints)` — `ProbeResult.spill_points` → GoldenSize 列表 |
| **Runtime** | JIT Director Daemon 周期调用 | `evolve(histogram)` — 高流量缝隙检测 → 创建中间态 Bucket (>5% 命中率) |
| **Hot-Swap** | L1i 缓存重排 | `evict_zombies()` — 原子覆写跳表，淘汰命中率 < 0.1% 的僵尸 Bucket |

**Load-Time 推导逻辑**：

1. 遍历 `ProbeResult.spill_points`，每个 spill point 代表寄存器溢出的物理拐点
2. GoldenSize = spill point 前的最大 seq_len（寄存器未溢出的最佳尺寸）
3. SMEM 利用率 = tile 面积 / `constraints.smem_size`
4. L2 命中率 = `1.0 - (seq_len / l2_thrash_threshold) × 0.3`
5. 如果无 spill points，回退到 L2 thrash 阈值的 25%/50%/75%/100% 比例推导
6. 最终兜底：至少一个默认 GoldenSize (seq_len=256)

### 12.5 collapse() — 任意 seq_len → 黄金尺寸

```rust
pub fn collapse(&mut self, seq_len: usize) -> (usize, &GoldenSize)
```

使用二分查找 (`partition_point`) 定位最近的黄金尺寸，同时记录命中统计到 `hit_stats`。算法复杂度 O(log n)。

**查找策略**：找到第一个 `seq_len >= golden.seq_len` 的位置，比较前后两个 GoldenSize 的距离，返回更近者。

### 12.6 evolve() — 运行时演化

| 条件 | 决策 |
|------|------|
| `total_samples < 100` | `InsufficientData` |
| 无高流量缝隙 | `NoEvolutionNeeded` |
| `golden_sizes.len() + gaps > max_buckets` | `CapacityLimitReached` |
| 缝隙区间命中率 > 5% | `Evolved` — 在缝隙中心创建新 GoldenSize |

**高流量缝隙检测** (`find_high_traffic_gaps`)：扫描 SeqHistogram 快照，找出落在两个相邻 GoldenSize 之间且累计命中数 > 0 的区间。

### 12.7 evict_zombies() — 僵尸淘汰

命中率 < `zombie_threshold` (默认 0.1%) 的 GoldenSize 被移除，同时清理对应的 `hit_stats` 条目。淘汰后保持 `golden_sizes` 升序不变。

### 12.8 与 AdaptiveChunkPolicy 的集成

§6.4 中 `adaptive_chunk_size()` 的返回值限定为 Golden Size 集合（非任意值）。Chunk size 映射为 `collapse(chunk_size).seq_len`，编译为独立 stride 常量。这确保了 Prefill Chunk 与 JIT 编译的 Tile 形状完全匹配。

## 13. 变体注册表 (VariantRegistry)

### 13.1 核心原则

**所有跨机制冲突通过编译时变体隔离消解，禁止 Mega-Kernel 内运行时 if**。

禁止模式：
- `if moe_enabled { ... }` 在 Mega-Kernel 内部
- `if guardrail_active { ... }` 在 Mega-Kernel 内部
- `if rag_enabled { ... }` 在 Mega-Kernel 内部

变体选择发生在 Dispatch-Time (`build_batch` 阶段)，不在 Mega-Kernel 执行时。

### 13.2 数据结构

```rust
pub struct VariantKey {
    pub arch: ModelArchitecture,
    pub moe_enabled: bool,
    pub guardrail_enabled: bool,
    pub spec_phase: Option<SpecPhase>,
    pub rag_enabled: bool,
    pub golden_size: usize,
    pub quant_type: Option<String>,
}

pub struct CompiledVariant {
    pub code: Vec<u8>,
    pub instruction_footprint_bytes: usize,
    pub mechanisms: Vec<MechanismId>,
    pub section: CodeSection,
    pub key: VariantKey,
}

pub enum CodeSection { Hot, Warm, Cold }
```

### 13.3 VariantKey 签名

每个 VariantKey 维度及其含义：

| 维度 | 类型 | 决定内容 |
|------|------|---------|
| `arch` | `ModelArchitecture` | 基础图结构 (RmsNorm → QKV → Attn → FFN → Residual) |
| `moe_enabled` | `bool` | 专家分发代码（`05-OPTIMIZATIONS.md` §2.7 MoE） |
| `guardrail_enabled` | `bool` | 安全探针代码（`05-OPTIMIZATIONS.md` §2.5 Guardrail） |
| `spec_phase` | `Option<SpecPhase>` | 推测解码阶段：Draft (浅层变体) / Verify (全量模型) / None (标准)（`05-OPTIMIZATIONS.md` §2.8） |
| `rag_enabled` | `bool` | Late-Fusion 知识注入残差代码（`05-OPTIMIZATIONS.md` §2.4 RAG） |
| `golden_size` | `usize` | 序列长度装筒（§12 黄金装筒） |
| `quant_type` | `Option<String>` | TurboQuant FWHT 旋转代码精度标识 |

### 13.4 L1i 预算检查

`register()` 时强制检查指令足迹：

```
instruction_footprint_bytes ≤ 80% × L1i_size
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `l1i_budget_bytes` | 32 KB | x86_64 主流 L1i 大小 |
| `L1I_BUDGET_RATIO` | 0.8 | 80% 预算上限 |
| `available_budget()` | `l1i_budget_bytes × 0.8` | 实际可用空间 |

超预算时返回 `L1iBudgetExceeded` 错误，建议减少 batch 并发或禁用部分机制。

### 13.5 代码段分层

| 代码段 | 缓存层级 | 说明 |
|--------|---------|------|
| `.text.hot` (`CodeSection::Hot`) | L1i 常驻 | 热路径代码，足迹 ≤ 80% L1i |
| `.text.warm` (`CodeSection::Warm`) | L2 常驻 | 温路径，通过 NOP Trampoline 按需拉入 L1i |
| `.text.cold` (`CodeSection::Cold`) | L3/DRAM | 冷路径，长跳转 (Long JMP)，几乎不执行 |

### 13.6 find_closest() — 逐步放松约束

当精确匹配失败时，按以下优先级逐步放松约束：

| 优先级 | 放松维度 | 说明 |
|--------|---------|------|
| 1 | `golden_size` | 找 ≤ 目标值的最大 GoldenSize |
| 2 | `spec_phase → None` | 放弃推测解码阶段特化 |
| 3 | `guardrail → false` | 放弃 Guardrail 探针代码 |
| 4 | `spec_phase → None` + `guardrail → false` | 组合放松 |
| 5 | `rag → false` | 放弃 RAG 注入代码 |
| 6 | 全部放松 | 仅保留 `arch` + `moe_enabled` |

放松后的变体仍可正确执行——被放松的机制在 Dispatch-Time 已被禁用。

### 13.7 MechanismId 枚举

12 种机制标识，每个 `CompiledVariant` 包含其所涉及的全部机制清单：

| MechanismId | 说明 | SPEC 引用 |
|-------------|------|----------|
| `Dense` | 基础稠密前向 (RmsNorm → QKV → Attn → FFN → Residual) | `05-OPTIMIZATIONS.md` |
| `MoeDispatch` | MoE 专家分发 | `05-OPTIMIZATIONS.md` §2.7 |
| `SpecDraft` | 推测解码 Draft 阶段 | `05-OPTIMIZATIONS.md` §2.8 |
| `SpecVerify` | 推测解码 Verify 阶段 | `05-OPTIMIZATIONS.md` §2.8 |
| `GuardrailProbe` | Guardrail 安全探针 | `05-OPTIMIZATIONS.md` §2.5 |
| `RagInjection` | Late-Fusion RAG 知识注入 | `05-OPTIMIZATIONS.md` §2.4 |
| `TurboQuantFwht` | TurboQuant FWHT 旋转 | `05-OPTIMIZATIONS.md` §4.2 |
| `GateFirstSkip` | Gate-First Skip 死神经元跳过 | `05-OPTIMIZATIONS.md` §2.1 |
| `ResidualBypass` | 残差旁路 Δρ 跳过 | `05-OPTIMIZATIONS.md` §2.2 |
| `Telemetry` | Epilogue 遥测 | `05-OPTIMIZATIONS.md` §5 |
| `EarlyExit` | Early-Exit 微型 lm_head | `05-OPTIMIZATIONS.md` §2.3 |
| `RaggedCompaction` | Compact→Execute→Scatter 三段式 | `05-OPTIMIZATIONS.md` §3.2 |

### 13.8 Dispatch-Time 选择

`build_batch()` 阶段调用 `derive_key()` 收集 batch 中所有请求属性，生成 VariantKey：

```rust
pub fn derive_key(
    arch: ModelArchitecture,
    has_moe_layers: bool,
    guardrail_active: bool,
    spec_phase: Option<SpecPhase>,
    rag_active: bool,
    golden_size: usize,
    quant_type: Option<String>,
) -> VariantKey
```

随后通过 `find_closest(&key)` 查找最优编译产物。整个过程发生在批构建阶段，不进入 Mega-Kernel 执行路径。

### 13.9 与 CompilerConstraints 的协作

L1i budget 来自 `CompilerConstraints.l1i_size`，通过 `VariantRegistry::with_l1i_budget(l1i_bytes)` 在构造时注入。`DeviceProfile` → `CompilerConstraints` → `VariantRegistry` 形成完整的硬件参数传递链。

## 14. Multi-Wave 空间并行与硬件饱和

### 14.1 Multi-Wave 空间并行

在单个 `step()` 内，多个计算 wave 在不同硬件单元上空间并行执行。每个 wave 是一个独立的 prefill chunk 或 decode token 集合，分配到不同的 SM 组或 NUMA 节点。

**Multi-Wave 是 Mega-Kernel 内部执行路径**：

```
Rust 调用: executor.step(requests)
  → 一次 CALL 进入 Mega-Kernel 入口函数
  → Mega-Kernel 内部:
      ├── 读取 WaveSchedule (已由 Scheduler 预计算)
      ├── GPU: 启动 grid launch，每个 Thread Block 处理一个 wave
      │         Thread Block 0: Wave 1 (SM 0-39)
      │         Thread Block 1: Wave 2 (SM 40-79)
      │         → 内部自行调度 wave partitioning，无需 Rust 介入
      ├── CPU: 每个 NUMA 节点的线程执行一个 wave
      │         Thread 0-7 (Node 0): Wave 1
      │         Thread 8-15 (Node 1): Wave 2
      └── 所有 wave 完成后返回 Rust

→ 调度器负责生成 WaveSchedule (请求→wave 分配)
→ Mega-Kernel 负责执行 WaveSchedule (内部 wave 并行)
→ Rust 层只做一次 CALL，不参与 wave 级编排
```

**架构原则**：
- Mega-Kernel 是单一的编译单元，内部包含所有 wave 的代码路径
- WaveSchedule 通过 ABI 参数传入 Mega-Kernel（非运行时动态调度）
- 每个 wave 的 KV cache 指针、权重指针、激活 buffer 在 ABI 中预分配
- wave 间无运行时同步需求（各自独立处理请求），仅在最终写回 output_buffer 时通过原子操作协调

**GPU Multi-Wave**:

```
Timeline (sm_90, 80 SM, 2 waves):
  SM 0-39:  Wave 1 Attention (WGMMA)  →  Wave 1 FFN (WGMMA)  →  ...
  SM 40-79: Wave 2 QKV Projection (TMA) →  Wave 2 Attn (WGMMA) →  ...

  → Wave 1 进入 FFN 时, SM 0-39 切换到 FFN 权重
  → Wave 2 进入 Attention 时, SM 40-79 开始 Wave 3 Prefill
  → 所有 SM 始终忙碌
```

**CPU NUMA Multi-Wave**:

```
Timeline (2 NUMA node, 16 cores):
  Node 0 (cores 0-7):  Wave 1 Attention (L3 local KV) → Wave 1 FFN (BLIS)
  Node 1 (cores 8-15): Wave 2 Attention (L3 local KV) → Wave 2 FFN (BLIS)

  → 每个 NUMA 节点的 KV cache 和权重数据在本地 L3
  → 零跨 NUMA 内存访问
```

**数据结构**:

```rust
pub struct WaveSchedule {
    pub waves: Vec<Wave>,
    pub sm_partition: GpuSmPartition,  // 见 02-HARDWARE.md §8.5
    pub numa_binding: Option<NumaBinding>,
}

pub struct Wave {
    pub request_ids: Vec<RequestId>,
    pub wave_type: WaveType,
    pub golden_size: usize,  // 本 wave 所有 seq 的装筒尺寸
}

pub enum WaveType {
    PrefillChunk { chunk_offset: usize, chunk_len: usize },
    Decode,
}
```

**调度约束**:

| 约束 | 说明 |
|------|------|
| 同 wave 内所有 seq 必须同一 golden_size | 保证 GEMM 形状一致，零 padding |
| GPU wave 数 ≤ SM 分区数 | 每个 wave 独占一个 SM 分区 |
| CPU wave 数 ≤ NUMA 节点数 | 每个 wave 绑定到一个 NUMA 节点 |
| Wave 间权重共享 | 同模型不同 wave 共享权重，只有 KV cache 独立 |

### 14.2 硬件饱和策略

确保每个计算单元在任何微秒都有工作：

**GPU 最小 batch tokens 计算**:

```
min_batch_tokens_per_wave = sm_per_partition × warp_size × occupancy_target

// GPU sm_80 (108 SM, 2 partitions, warp=32):
//   sm_per_partition = 54
//   min_tokens = 54 × 32 × 0.5 = 864 tokens per wave
//   decode (seq=1): 需要 864 个并发 decode 序列才能饱和一个 wave
//   prefill: chunk_size ≥ 864 时一个 wave 即可饱和

// GPU sm_90 (132 SM, 2 partitions, warp=32):
//   sm_per_partition = 66
//   min_tokens = 66 × 32 × 0.5 = 1056 tokens per wave
```

**CPU 最小并行度计算**:

```
min_parallel_gemms = num_cores / cores_per_blis_instance

// AVX-512 (16 cores):
//   cores_per_blis_instance = 2  (主线程 + 1 helper 做 pack)
//   min_parallel_gemms = 8
//   → 需要 8 个独立 GEMM 任务（不同 seq 或不同 head group）保持所有核心忙碌
```

**饱和度监控**:

```rust
pub struct SaturationMetrics {
    pub compute_utilization: f32,    // 理论 TFLOPS / 实际 TFLOPS
    pub memory_bandwidth_util: f32,  // 实际带宽 / 理论带宽
    pub sm_occupancy: f32,           // 活跃 SM 占比
    pub wave_pipeline_depth: usize,  // 当前并行 wave 数
}
```

### 14.3 Batch 同号合并（Same-Length Grouping）

将相似 seq_len 的序列合并到同一个 batch/wave，消除 padding 浪费。

**问题**: 不同 seq_len 的序列混在同一个 batch 中，短序列被 padding 到最长序列的长度：

```
seq_1: [tok × 2048] → GEMM(2048, hidden, hidden) → 有效
seq_2: [tok × 64]   → GEMM(2048, hidden, hidden) → 1984 行浪费 FLOPs
```

**解决方案**: 按 Golden Bucket 分组，同号序列合并：

```
1. 收集所有 ready 序列
2. 按 seq_len 分配到对应的 Golden Bucket (collapse())
3. 同 bucket 的序列打包到同一个 wave
4. 每个 wave 内所有 seq_len 相同 → GEMM 形状统一 → 零 padding
```

**算法**:

```rust
fn group_by_golden_size(
    sequences: &[SequenceInfo],
    registry: &GoldenBucketRegistry,
) -> Vec<Vec<RequestId>> {
    // 1. 每个序列映射到最近的 Golden Size
    let mut groups: BTreeMap<usize, Vec<RequestId>> = BTreeMap::new();
    for seq in sequences {
        let (golden_size, _) = registry.collapse(seq.seq_len);
        groups.entry(golden_size).or_default().push(seq.id);
    }

    // 2. 按 wave 容量拆分过大的组
    // 3. 合并过小的组到最近 bucket（允许少量 padding）
    groups.into_values().collect()
}
```

**与 Continuous Batching 的集成**:

`ContinuousBatcher::build_batch()` 流程更新：

```
1. 收集 ready decode tokens 和 prefill chunks
2. 按 golden_size 分组（decode 和 prefill 分开）
3. 为每个组分配 wave slot
4. 生成 WaveSchedule（包含 wave 数量、SM 分配、NUMA 绑定）
5. 构建 BatchManifest（每个 wave 的 slot 列表）
```

**收益量化**:

| 场景 | 无合并 (padding) | 同号合并 | FLOPs 节省 |
|------|-----------------|---------|-----------|
| 8 seq × [2048, 512, 64, 1024, 256, 128, 4096, 32] | 全 padding 到 4096 | 3 groups: [32,64,128], [256,512,1024], [2048,4096] | 55-70% |
| 4 decode seq × [1, 1, 1, 1] | 无 padding | 单组 | 0% (已均匀) |
| 混合 prefill+decode | decode 和 prefill 分离 | decode 一组, prefill 按 chunk 分组 | 30-50% |

### 14.4 资源分配算法总览

系统级资源分配的决策流程：

```
DeviceProfile + MemoryNetworkSensors
  → CompilerConstraints (02-HARDWARE.md §8)
    → 资源分配决策:
      ├── 寄存器: 微内核 MR×NR + epilogue 深度 (04-OPERATORS.md §13.3)
      ├── L1 Cache: GEMM tile 分块 KC × MC (04-OPERATORS.md §13.4)
      ├── L2 Cache: KV/权重/激活 三路预算 (04-OPERATORS.md §13.4)
      ├── L3/HBM: 全模型 + KV pool + workspace (04-OPERATORS.md §13.4)
      ├── SM/Core: Multi-Wave 分区 (本节 §14.1)
      └── Batch: 同号合并 + 硬件饱和 (本节 §14.2-14.3)
```

**分配优先级**:

| 优先级 | 资源 | 分配策略 | 不可压缩 |
|--------|------|---------|---------|
| 1 | 模型权重 | 常驻 HBM，按需 L2 预取 | 是 |
| 2 | KV Cache | PagedAttention 管理，L2 热页 + L3 冷页 | 否（可 swap） |
| 3 | Workspace | 区间图着色复用，最小化分配 | 否 |
| 4 | 寄存器 | 微内核累加器优先，epilogue 其次 | 是 |
| 5 | SM/Core | Multi-Wave 分区，按 batch 负载动态调整 | 否 |
