# HGAL 调度算法详细设计

> **关联需求**: REQ-SCHED-001, REQ-SCHED-002, REQ-SCHED-003, REQ-SCHED-007, REQ-SCHED-009
> **版本**: 2.0
> **状态**: 已实现 (v1.0), 扩展设计中 (v2.0 - 2024 优化)

---

## 1. 设计目标

解决 VLLM 分页调度中的两个核心问题：

| 问题 | 描述 | 影响 |
|------|------|------|
| **页面序列错乱** | 同一 Sequence 的页面被分散换出/换入 | 内存碎片、预取失效、多次 swap-in |
| **Cache Thrashing** | 刚 swap-in 的页面立即被换出 | 浪费 PCIe 带宽、端到端延迟增加 |

### 禁止行为

| 禁止项 | 原因 | 检测方法 |
|--------|------|----------|
| 禁止序列内页面分散换出 | 破坏 Gang Scheduling 一致性 | 检查 `select_victim_pages` 是否返回单个页面 |
| 禁止新换入页立即换出 | Cache Thrashing | 检查 Warm 状态页面是否被选中 |
| 禁止使用纯 LRU | 无法区分扫描型和重复型访问 | 检查优先级计算是否仅基于时间 |

---

## 2. 算法概述

### 2.1 HGAL (Hybrid Gang-Aware LIRS) 组成

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HGAL (Hybrid Gang-Aware LIRS)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Gang-Aware      │    │ LIRS Priority   │    │ Warm-up         │ │
│  │ Eviction        │ +  │ Calculation     │ +  │ Protection      │ │
│  │                 │    │                 │    │                 │ │
│  │ 序列组整体换出   │    │ IRR 优先级      │    │ 新页保护期      │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Working Set     │    │ CLOCK-Pro       │    │ Chunked         │ │
│  │ Detection       │    │ Approximation   │    │ Prefill         │ │
│  │                 │    │                 │    │                 │ │
│  │ 热页自动锁定     │    │ 低成本实现      │    │ 计算与内存平衡  │ │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 设计原则

| 原则 | 说明 | 来源 |
|------|------|------|
| **Gang-Aware** | 序列组作为换出单位，保持页面连续性 | vLLM Sequence Group 设计 |
| **LIRS 优先级** | 使用 IRR (Inter-Reference Recency) 替代纯 LRU | 操作系统 LIRS 算法 |
| **Warm-up 保护** | 新换入页面保护期，禁止立即换出 | CLOCK-Pro 热页保护 |
| **Working Set** | 自动检测热页并锁定保护 | 2Q/LIRS 工作集检测 |
| **CLOCK-Pro 近似** | 低成本的 LIRS 实现 | CLOCK-Pro 论文 |

---

## 3. 数据结构定义

### 3.1 页面状态机

```
                    ┌─────────────────┐
                    │   Swapped        │
                    └────────┬────────┘
                             │ swap-in
                             ▼
                    ┌─────────────────┐
                    │      Warm        │ ◄─── 保护期 (禁止换出)
                    └────────┬─────────┘
                             │ 访问 + 时间
                             ▼
┌───────────────┐   ┌─────────────────┐
│   Protected    │◄──│     Active       │
└───────┬───────┘   └────────┬─────────┘
        │                   │ 访问频率下降
        │                   ▼
        │             ┌─────────────────┐
        └────────────►│     Standby      │
                      └─────────────────┘
```

**状态定义表**：

| 状态 | 说明 | 换出条件 | 保护等级 |
|------|------|----------|----------|
| **Active** | 在 GPU 且正在使用 | 变为 Standby | 正常 |
| **Standby** | 在 GPU 但未使用 | 可被换出 | 正常 |
| **Swapped** | 已换出到 CPU | swap-in | 最高 |
| **Warm** | 刚 swap-in，保护期 | 保护期结束 | 高 |
| **Protected** | 工作集保护 | 访问频率下降 | 中 |

### 3.2 序列组 (SequenceGroup)

**设计依据**: vLLM 的 Sequence Group 概念

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | RequestId | 序列组唯一标识 |
| `pages` | Vec<PageId> | 该序列的所有页面 |
| `state` | GroupState | Running / Swapped / Paused |
| `access_count` | usize | 访问计数 (用于 LFU) |
| `last_access` | Instant | 最后访问时间 |
| `is_pinned` | bool | 是否锁定 (优先级高) |

### 3.3 页面元数据 (PageMetadata)

| 字段 | 类型 | 说明 | 用途 |
|------|------|------|------|
| `page_id` | PageId | 页面唯一标识 | 定位 |
| `sequence_id` | Option<RequestId> | 所属序列 | Gang-Aware |
| `state` | PageState | 当前状态 | 换出决策 |
| `recency` | usize | IRR 值 | LIRS 优先级 |
| `is_lir` | bool | 是否为 LIR 页面 | 长期驻留 |
| `swap_in_time` | Option<Instant> | 换入时间 | Warm-up 计算 |
| `warm_until` | Option<Instant> | 保护期结束 | 换出保护 |
| `access_count` | usize | 访问计数 | 频率检测 |
| `last_access` | Instant | 最后访问 | LRU 计算 |

---

## 4. 核心算法设计

### 4.1 Gang-Aware 受害者选择

**目标**: 选择要换出的序列组，而非单个页面

**输入**: 需要换出的页面数量 `count`
**输出**: 要换出的序列组 ID 列表

**算法步骤**:

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 排除筛选 | 排除 Warm、Protected、Pinned 状态的页面 |
| 2 | 计算优先级 | 对每个序列组计算优先级分数 |
| 3 | 排序 | 按优先级从低到高排序 |
| 4 | 选择 | 选择最少数量的序列组满足 `count` |

**优先级计算公式**:

```
priority(group) = time_penalty + recency_penalty - freq_bonus - pin_bonus

其中:
  time_penalty = now - group.last_access
  recency_penalty = sum(page.recency for page in group.pages)
  freq_bonus = group.access_count * FREQUENCY_WEIGHT
  pin_bonus = group.is_pinned ? PIN_BONUS : 0
```

### 4.2 LIRS 优先级计算

**目标**: 综合 IRR 和访问频率，识别真正的冷页面

**输入**: 页面元数据
**输出**: 优先级分数 (越低越应该被换出)

**计算逻辑**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 计算时间衰减 | `time_since_access = now - last_access` |
| 2 | 应用 IRR 惩罚 | `recency_penalty = page.recency` |
| 3 | 应用频率奖励 | `freq_bonus = access_count × 频率权重` |
| 4 | 应用保护加成 | Warm 状态 +1000，Protected 状态 +2000 |
| 5 | 计算最终优先级 | `time_since_access + recency_penalty - freq_bonus - protection_bonus` |

### 4.3 Warm-up 保护机制

**目标**: 防止刚 swap-in 的页面立即被换出

**触发时机**: 页面 swap-in 完成时

**保护期计算**:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `warmup_duration` | 100ms | 保护期时长 |
| `min_access_count` | 2 | 最小访问次数保护 |

**状态转换**:

```
Swapped --(swap-in)--> Warm --(时间到期 OR 访问2次)--> Active
                              └--(访问频率高)--> Protected
```

### 4.4 Working Set 自动检测

**目标**: 自动识别高频访问页面并锁定保护

**检测周期**: 每 N 个 generation tick

**热页判定条件**:

| 条件 | 说明 | 阈值 |
|------|------|------|
| 访问频率 | 单位时间内访问次数 | `access_count >= hot_threshold` |
| 最近访问 | 最后访问时间距离现在 | `now - last_access < working_set_window` |

**保护解除**: 热页在 `working_set_window` 时间内未被访问则解除保护

### 4.5 CLOCK-Pro 风格近似实现

**目标**: 低成本的 LIRS 近似实现

**数据结构**:

| 结构 | 说明 |
|------|------|
| `clock_hand` | 扫描指针位置 |
| `cold_pages` | 冷页面候选集合 |
| `hot_pages` | 热页面集合 (类似 LIR) |
| `test_pages` | 测试页面 (用于检测访问模式变化) |

**扫描算法**:

```
1. 从 clock_hand 开始扫描
2. 对于每个页面:
   a. 如果页面被访问:
      - 如果在 cold_pages → 移到 hot_pages
      - 如果在 hot_pages → 保持在 hot_pages
   b. 如果页面未被访问:
      - 如果在 hot_pages → 移到 cold_pages
      - 如果在 cold_pages → 保持在 cold_pages
3. clock_hand 前进
```

---

## 5. 与 VLLM 的对比

| 特性 | VLLM V1 (RECOMPUTE) | HGAL 方案 |
|------|---------------------|-----------|
| 换出粒度 | 序列级 (gang) | 序列组级 (gang) |
| 抢占模式 | Recompute (重算) | Swap (换出) + Warm-up 保护 |
| 优先级 | FCFS | IRR + 频率 + 状态 |
| 新页保护 | 无 | Warm-up 期 |
| 工作集 | 无 | 自动检测 |
| 序列错乱 | 禁止 | 禁止 |
| Cache Thrashing | 可能 | 避免 |

---

## 6. 配置参数

### 6.1 调度器配置 (SchedulerConfig)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `page_size` | usize | 128 | 每页 token 数 |
| `total_pages` | usize | 2048 | 总页面数 |
| `max_batch` | usize | 8 | 最大批大小 |
| `max_tokens` | usize | 4096 | 最大 token 数 |
| `warmup_duration` | Duration | 100ms | Warm-up 保护期 |
| `working_set_window` | Duration | 1s | 工作集时间窗 |
| `hot_threshold` | usize | 3 | 热页访问阈值 |
| `lir_ratio` | f32 | 0.3 | LIR 页面比例 |
| `enable_clock_pro` | bool | true | 启用 CLOCK-Pro 近似 |

### 6.2 性能调优建议

| 场景 | 建议配置 | 说明 |
|------|----------|------|
| 高并发短请求 | `warmup_duration = 50ms` | 缩短保护期 |
| 长上下文推理 | `hot_threshold = 5` | 提高热页阈值 |
| 内存受限 | `lir_ratio = 0.2` | 减少 LIR 页面 |
| 计算/内存混合 | `max_tokens` 适当调整 | 平衡吞吐和延迟 |

---

## 7. 验收标准

### 7.1 功能验收

| 标准 | 检测方法 |
|------|----------|
| 序列组整体换出 | 单元测试验证 `select_victim_groups` 返回序列组 ID |
| 新页不被立即换出 | 单元测试验证 Warm 状态页面不被选中 |
| 热页自动锁定 | 单元测试验证高频访问页面变为 Protected |
| 显存碎片率 < 5% | 性能测试验证 |

### 7.2 性能验收

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| Thrashing 率 | < 1% | 统计 swap-in 后立即 swap-out 的比例 |
| 平均换出延迟 | < 10ms | 基准测试 |
| Working Set 命中率 | > 90% | 热页访问命中率 |

---

## 8. 2024 vLLM 优化扩展 (v2.0)

> **关联需求**: REQ-SCHED-007, REQ-SCHED-008, REQ-SCHED-009
> **JIT 兼容性**: ✅ 所有优化均为调度/算法级别，不涉及 JIT 编译器管线

### 8.1 Chunked Prefill / SplitFuse 调度

#### 8.1.1 问题定义

**传统 Continuous Batching 阶段隔离问题**：

```
问题场景：同时存在 Prefill 请求和 Decode 请求

传统调度（阶段隔离）:
┌─────────────────────────────────────────────────────────────┐
│ Batch 1: [Prefill-512, Prefill-256]                         │
│   → GPU 利用率 ~60% (Memory Bound)                          │
│   → Decode 请求必须等待 (Tail Latency 恶化)                 │
├─────────────────────────────────────────────────────────────┤
│ Batch 2: [Decode-1, Decode-1, ..., Decode-1] (×32)         │
│   → GPU 利用率 ~95% (Compute Bound)                         │
│   → Prefill 请求必须等待 (吞吐浪费)                         │
└─────────────────────────────────────────────────────────────┘
```

**问题量化**：

| 指标 | 传统方式 | 期望目标 |
|------|----------|----------|
| Tail Latency (P99) | ~200ms | <50ms |
| GPU 利用率方差 | 35% | <15% |
| 批次切换开销 | 5-10ms | <1ms |

#### 8.1.2 Chunked Prefill 算法

**核心思想**：将长 Prefill 请求切分为多个 Chunk，与 Decode Token 交织调度。

```
Chunked 调度（交织）:
┌─────────────────────────────────────────────────────────────┐
│ Batch 1:                                                    │
│   [Prefill-64, Decode-1, Decode-1, Decode-1, Decode-1,      │
│    Prefill-64, Decode-1, Decode-1, Decode-1, Decode-1]      │
│    ↑ Chunk 1       ↑ Decode 插槽 (优先)    ↑ Chunk 2        │
└─────────────────────────────────────────────────────────────┘

优势:
  - Decode 请求无需等待 (低延迟保证)
  - GPU 利用率平滑 (Memory + Compute 混合)
  - 批次连续 (无切换开销)
```

**切分策略**：

| 策略 | Chunk Size | 适用场景 |
|------|------------|----------|
| **固定切分** | 64/128 tokens | 通用场景 |
| **自适应切分** | 根据内存压力动态调整 | 显存受限 |
| **优先级切分** | 高优先级请求更大 Chunk | SLA 保障 |

#### 8.1.3 SplitFuse 优化

**进一步细分 Prefill 阶段**：

```
SplitFuse = 分离 Q/K/V 计算 + 融合 Attention

阶段 A (并行计算，充分利用 Tensor Core):
  ┌─────────────────────────────────────────────────────────┐
  │ Chunk 1: Q_proj │ Chunk 2: Q_proj │ Chunk 3: Q_proj │  │
  │ Chunk 1: K_proj │ Chunk 2: K_proj │ Chunk 3: K_proj │  │
  │ Chunk 1: V_proj │ Chunk 2: V_proj │ Chunk 3: V_proj │  │
  └─────────────────────────────────────────────────────────┘
                              ↓
阶段 B (融合 Attention，内存优化):
  ┌─────────────────────────────────────────────────────────┐
  │ FlashAttention(Chunk1+Chunk2+Chunk3 KV Cache)          │
  │   → 单次大矩阵 Attention，最大化内存局部性               │
  └─────────────────────────────────────────────────────────┘
```

**数据流对比**：

```
传统方式:
  Token[0-511] → QKV Projection → Attention → Output

SplitFuse 方式:
  Token[0-63]   → QKV → [缓存]
  Token[64-127] → QKV → [缓存]
  Token[128-191] → QKV → [缓存]
  ...
  → Attention([合并后的 KV]) → Output
```

#### 8.1.4 调度算法

**数据结构**：

| 结构 | 字段 | 说明 |
|------|------|------|
| `ChunkedScheduler` | config, prefill_queue, decode_queue | 调度器主体 |
| `PrefillRequest` | id, total_tokens, completed_tokens, pending_chunks | Prefill 请求状态 |
| `ChunkedConfig` | chunk_size, decode_slots, enable_splitfuse, max_chunks_per_batch | 配置参数 |

**调度流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 收集 Decode 请求 | 填充 decode_slots 直到达到配置上限 |
| 2 | 计算剩余预算 | `remaining_budget = max_tokens - decode_tokens` |
| 3 | 填入 Prefill Chunk | 从 prefill_queue 取出请求，按 chunk_size 切分 |
| 4 | 更新请求进度 | 完成的 Chunk 从 pending 减少 |
| 5 | 请求状态转换 | Prefill 完成后转为 Decode 请求 |

**返回结果**：`ChunkedBatch` 包含 decode_slots 和 prefill_chunks 列表

#### 8.1.5 验收标准

| 标准 | 测试方法 | 目标值 |
|------|----------|--------|
| Tail Latency (P99) | 混合负载测试 | <50ms |
| GPU 利用率方差 | 性能监控 | <15% |
| 吞吐提升 | 对比纯 Decode Batch | +30-50% |
| 内存开销 | 显存监控 | <5% 额外 |

---

### 8.2 (已废弃) SwiftKV 算法

> 🚨 **架构合规性警报 (Architect Veto)**:
> 原始的 `SwiftKV 算法`（通过 `cosine_similarity` 做跨层共享和通过 Attention 权重做合并蒸馏）已被判定为 **违宪**，并将在此架构下永久失效。原因如下：
> 1. 原算法要求对连续时间步的 KV 缓存进行浮点强度的插值合并（Interpolation）。但在 `Mega-Kernel` 架构下，所有的 KV Cache 已被 **TurboQuant 静态锁定在 3-bit / 4-bit 的双轨分布**中，强行插值合并不仅需要巨大的反量化/重量化流水线开销，还会造成严重的重分配时戳延迟。
> 2. 原算法要求主机去测算 $\Delta$ 层距离矩阵（cosine_similarity 等）。这使得 CPU 端重新拿回了观测控制权，严重抵触了“消灭 CPU 控制台”与“In-Kernel Routing”法则。
>
> 故，HGAL 系统只能做**拓扑级的完整页框交换（Complete Page Swapping）**，禁止在调度中侵入修改数据的压缩或数值形式！

---

### 8.3 (已废弃) LMCache 跨请求 KV Cache

> 🚨 **架构合规性警报 (Architect Veto)**:
> 旧版通过引入第三方 LMCache 组件及复杂的 L1/L2/L3 分布式哈希缓存机制已被 **废弃并从 vllm2024.rs 中物理删除**。
> 本项目已转向使用 `GlobalMemoryManager` 配合 `PrefixIndex` 和 `SessionKvCache` 的双管线本地内存管理机制 (REQ-KV-001/002)。此处的旧规范不再具有 SSOT 效应，留作历史封存。

---

### 8.4 三项优化集成数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  新请求到达                                                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  SessionKvCache     │
                    │  前缀缓存查询       │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │ 命中                         │ 未命中
              ▼                               ▼
        ┌───────────┐                 ┌──────────────┐
        │ Prefix Hit│                 │ Chunked      │
        │ (部分跳过)│                 │ Prefill      │
        │  Prefix)  │                 │ (交织调度)   │
        └─────┬─────┘                 └──────┬───────┘
              │                               │
              ▼                               ▼
        ┌─────────────┐              ┌────────────────┐
        │ Decode      │              │ PageSwap       │
        │ Loop        │              │ (拓扑级页框交换)│
        └─────────────┘              └────────┬───────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │ KV Cache 写入  │
                                    │ (L1/L2/L3)     │
                                    └────────────────┘
```

### 8.5 配置参数汇总

#### 8.5.1 Chunked Prefill 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `chunk_size` | usize | 64 | Chunk 大小 (tokens) |
| `decode_slots` | usize | 8 | 每批保留的 Decode 插槽 |
| `enable_splitfuse` | bool | ~~true~~ | ⛔ 已废弃 (REQ-SCHED-007) — SplitFuse 混批路径已移除 |
| `max_chunks_per_batch` | usize | 4 | 每批最大 Chunk 数 |

#### 8.5.2 SwiftKV 配置 (⛔ 违宪/已废弃)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `window_size` | usize | 4 | SIKV 窗口大小 |
| `enable_across_kv` | bool | true | 启用 AKV |
| `similarity_threshold` | f32 | 0.9 | AKV 相似度标量阈值 |
| `precision_guard` | f32 | 0.1 | 精度损失阈值标量 (PPL) |

#### 8.5.3 LMCache 配置 (⛔ 违宪/已废弃物理删除)

> 已被删除。请使用 SessionKvCache 机制，不再支持 LMCache。

---

## 9. 参考文献

1. [CLOCK-Pro: An Effective Improvement of the CLOCK Replacement](https://www.usenix.org/legacyurl/clock-pro-effective-improvement-clock-replacement) - USENIX 2005
2. [vLLM 架构及源码 - Scheduling](https://zhuanlan.zhihu.com/p/28180951363) - Sequence Group 设计
3. [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115) - arXiv:2502.07115
4. [Clock2Q+: A Simple and Efficient Replacement Algorithm](https://www.arxiv.org/pdf/2511.21958) - arXiv:2511.21958
5. [2D Management of KV-Cache via Layer-wise Optimization](https://iclr.cc/virtual/2025/poster/30707) - ICLR 2025
