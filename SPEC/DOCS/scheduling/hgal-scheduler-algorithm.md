# HGAL 调度算法详细设计

> **关联需求**: REQ-SCHED-001, REQ-SCHED-002, REQ-SCHED-003, REQ-SCHED-007, REQ-SCHED-008, REQ-SCHED-009
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
> **AOT CUBIN 兼容性**: ✅ 所有优化均为调度/算法级别，无需修改内核编译流程

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

### 8.2 SwiftKV 算法

> **论文**: SwiftKV: Efficient KV Cache Compression for LLM (arXiv:2410.03960)

#### 8.2.1 问题定义

**传统 PagedAttention 的 KV Cache 增长**：

```
KV Cache 内存占用 = Num_Layers × Num_Tokens × Hidden_Dim × 2 (K+V) × Sizeof(f16)

示例 (Qwen3-7B, 32k 上下文):
  - Layers: 28
  - Tokens: 32,768
  - Hidden: 4096
  - 内存: 28 × 32768 × 4096 × 2 × 2 = 15 GB

问题: 内存随序列长度线性增长
```

**关键观察**：后续 Token 的 KV 贡献度递减

```
Attention Weight 分布:
  Token 0:    ████████████████ 0.25  (早的 Token)
  Token 100:  ██████░░░░░░░░░░ 0.12
  Token 500:  ███░░░░░░░░░░░░░ 0.05
  Token 1000: ██░░░░░░░░░░░░░░ 0.03
  Token 5000: █░░░░░░░░░░░░░░░ 0.01  (晚的 Token，贡献低)

→ 大量 KV Cache 用于低 Attention Score 的 Token
```

#### 8.2.2 SingleInputKV (SIKV) 蒸馏

**核心思想**：将连续的 N 个 KV 向量蒸馏为 1 个

```
算法流程:
  1. 将序列按窗口大小 W 分组 (W=2/4/8)
  2. 每组内: [K0, K1, K2, K3] → Attention 加权 → [K_merged]
  3. 只保留 merged KV

蒸馏公式:
  K_merged = Σ(Attention_Weight_i × K_i) / Σ(Attention_Weight_i)
  V_merged = Σ(Attention_Weight_i × V_i) / Σ(Attention_Weight_i)

其中 Attention_Weight_i 基于:
  - 该 Token 在历史中的平均 Attention Score
  - 或使用简单平均 (1/W)
```

**效果量化**：

| 窗口大小 W | KV Cache 减少 | 精度损失 (PPL) | 适用场景 |
|-----------|---------------|----------------|----------|
| 2 | 50% | <0.01% | 高精度要求 |
| 4 | 75% | <0.05% | 平衡 |
| 8 | 87.5% | <0.2% | 极限压缩 |

#### 8.2.3 AcrossKV (AKV) 跨层共享

**核心思想**：相邻层的 KV Cache 高度相关

```
相似度分析 (Qwen3-7B):
  Layer 0 & Layer 1:  cosine_similarity = 0.94
  Layer 1 & Layer 2:  cosine_similarity = 0.91
  Layer 5 & Layer 6:  cosine_similarity = 0.87
  ...

→ 相邻层 KV Cache 可共享
```

**共享策略**：

| 配置项 | 说明 |
|--------|------|
| `similarity_threshold` | 相似度阈值 (默认 0.9) |
| `check_interval` | 每隔 N 层检查一次 |

**决策流程**：

| 步骤 | 操作 | 条件 |
|------|------|------|
| 1 | 计算相邻层 KV 相似度 | cosine_similarity |
| 2 | 高相似度决策 | 相似度 > 阈值 → 共享前一层的 KV |
| 3 | 低相似度决策 | 相似度 ≤ 阈值 → 保留当前层 KV |

**收益**：

| 指标 | 无 AKV | 有 AKV |
|------|--------|--------|
| KV Cache 总量 | 100% | 50% |
| 精度损失 (PPL) | - | <0.1% |

#### 8.2.4 集成到 SwapManager

**SwapManager 扩展方法**：

| 方法 | 说明 |
|------|------|
| `swap_out_with_distillation(pages, swift_config)` | Swap-out 时执行蒸馏 |
| `distill_kv_pages(pages, window_size)` | SIKV 蒸馏（合并连续页面） |
| `share_across_layers(pages, config)` | AKV 跨层共享 |

**蒸馏流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 检查 SwiftKV 是否启用 | 未启用则直接普通 swap_out |
| 2 | SIKV 蒸馏 | 按 window_size 合并页面 |
| 3 | AKV 跨层共享（可选） | 计算相似度决定是否共享 |
| 4 | 实际 Swap-out | 使用压缩后的页面数 |

**跨层共享流程**：

| 步骤 | 操作 | 条件 |
|------|------|------|
| 1 | 遍历所有页面 | - |
| 2 | 计算与前一层的相似度 | cosine_similarity |
| 3 | 高相似度时跳过 | 相似度 > 阈值 |
| 4 | 低相似度时保留 | 添加到共享列表 |

#### 8.2.5 精度保障机制

**配置扩展**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `precision_guard` | 浮点数 | 精度损失阈值 (PPL 差值，默认 0.1) |
| `validation_interval` | 数值 | 定期验证（每 N 个请求） |

**验证流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 计算 PPL 差值 | 对比原始输出和蒸馏输出 |
| 2 | 阈值检查 | 差值 > precision_guard 则失败 |
| 3 | 自动禁用 | 精度损失超阈值时禁用蒸馏 |

#### 8.2.6 验收标准

| 标准 | 测试方法 | 目标值 |
|------|----------|--------|
| KV Cache 减少 | 长序列测试 (32k) | >50% |
| 精度损失 (PPL) | 对比完整 KV Cache | <0.1% |
| 蒸馏开销 | CPU 时间监控 | <5% Swap 时间 |

---

### 8.3 LMCache 跨请求 KV Cache

> **项目**: [LMCache](https://github.com/LMCache/LMCache) - Distributed KV Cache for LLM Serving

#### 8.3.1 问题定义

**多用户场景的重复计算**：

```
场景: 100 个用户使用相同的 System Prompt

System Prompt:
  "You are a helpful assistant. Please answer the following questions..."

Tokenized: [101, 102, ..., 612]  (512 tokens)

传统方式:
  请求 1: Prefill 512 tokens → 生成 KV Cache
  请求 2: Prefill 512 tokens → 生成 KV Cache  (重复！)
  请求 3: Prefill 512 tokens → 生成 KV Cache  (重复！)
  ...
  请求 100: Prefill 512 tokens → 生成 KV Cache  (重复！)

浪费: 51,200 tokens 的 Prefill 计算被重复 100 次
```

#### 8.3.2 三层缓存架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1: GPU In-Memory Cache (最快，最小)                              │
│  ────────────────────────────────────────────────────────────────  │
│  存储: 当前活跃请求的 KV Cache                                     │
│  命中率: ~5% (仅当前 Batch 重复)                                   │
│  容量: ~100MB (显存限制)                                           │
│  延迟: <1ms (GPU 内存复制)                                        │
├─────────────────────────────────────────────────────────────────────┤
│  L2: CPU Offload Cache (中等，中量)                               │
│  ────────────────────────────────────────────────────────────────  │
│  存储: 最近使用的 KV Cache (LRU 淘汰)                              │
│  命中率: ~30% (跨 Batch 重复)                                      │
│  容量: ~10GB (系统内存)                                            │
│  延迟: ~10ms (PCIe DMA 复制)                                      │
├─────────────────────────────────────────────────────────────────────┤
│  L3: Distributed Cache (慢，无限)                                  │
│  ────────────────────────────────────────────────────────────────  │
│  存储: Redis/S3/本地磁盘                                           │
│  命中率: ~65% (跨会话/跨实例重复)                                  │
│  容量: 无限                                                        │
│  延迟: ~50ms (网络/磁盘 I/O)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### 8.3.3 Cache Key 设计

**CacheKey 结构**：

| 字段 | 说明 |
|------|------|
| `model_id` | 模型标识 |
| `prompt_hash` | Prompt 前缀哈希 |
| `layer_indices` | 缓存哪些层 |
| `quantization` | 量化类型 |

**计算流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 截取前缀 | 只缓存前 N tokens (默认 512) |
| 2 | 组合哈希输入 | `model_id + prefix_tokens` |
| 3 | SHA256 哈希 | 生成唯一缓存键 |

#### 8.3.4 缓存查找流程

**LMCache 管理器结构**：

| 组件 | 说明 |
|------|------|
| `l1` | GPU L1 缓存 (LRU) |
| `l2` | CPU L2 缓存 (LRU) |
| `l3` | 分布式后端 (L3Backend) |

**三层查找流程**：

| 层级 | 操作 | 延迟 |
|------|------|------|
| **L1 查询** | 检查 GPU 缓存 | <1ms |
| **L2 查询** | Miss 时 DMA 复制到 GPU，回填 L1 | ~10ms |
| **L3 查询** | Miss 时加载到 L2，DMA 复制到 GPU，回填 L1/L2 | ~50ms |

**写入流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 写入 L1 | 直接存储到 GPU 缓存 |
| 2 | 异步写入 L2 | DMA 复制到 CPU，不阻塞 |
| 3 | 异步写入 L3 | 序列化后写入后端，不阻塞 |

#### 8.3.5 L3 Backend 接口

**L3Backend 接口**：

| 方法 | 说明 |
|------|------|
| `get(key)` | 从后端获取缓存条目 |
| `put(key, data)` | 写入数据到后端 |
| `invalidate(model_id)` | 失效指定模型的所有缓存 |

**后端实现类型**：

| 类型 | 说明 | 用途 |
|------|------|------|
| `RedisBackend` | Redis 连接 + TTL 配置 | 分布式缓存 |
| `LocalDiskBackend` | 本地磁盘路径 | 单机缓存 |
| `DisabledBackend` | 禁用 L3 | 仅 L1+L2 模式 |

#### 8.3.6 调度器集成

**调度方法**：`schedule_with_cache(request, lmcache)`

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 计算缓存键 | 基于 model_id + prompt_tokens |
| 2 | 查询缓存 | 调用 `lmcache.get(key, backend)` |
| 3a | 命中处理 | 返回 `SchedulingResult::CacheHit` |
| 3b | 未命中处理 | 正常 Prefill，完成后写入缓存 |
| 4 | 错误降级 | 缓存错误时降级为普通 Prefill |

#### 8.3.7 验收标准

| 标准 | 测试方法 | 目标值 |
|------|----------|--------|
| 缓存命中率 | 相同 Prompt 多请求 | >70% |
| 吞吐提升 | 对比无缓存 | 10×+ |
| 端到端延迟 | 缓存命中场景 | <10ms (跳过 Prefill) |
| 内存开销 | L2+L3 存储 | <20GB |

---

### 8.4 三项优化集成数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│  新请求到达                                                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  LMCache 查询       │
                    │  (L1 → L2 → L3)     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │ 命中                         │ 未命中
              ▼                               ▼
        ┌───────────┐                 ┌──────────────┐
        │ Cache Hit │                 │ Chunked      │
        │ (跳过     │                 │ Prefill      │
        │  Prefix)  │                 │ (交织调度)   │
        └─────┬─────┘                 └──────┬───────┘
              │                               │
              ▼                               ▼
        ┌─────────────┐              ┌────────────────┐
        │ Decode      │              │ SwiftKV 蒸馏   │
        │ Loop        │              │ (Swap-out 时)  │
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
| `enable_splitfuse` | bool | true | 启用 SplitFuse |
| `max_chunks_per_batch` | usize | 4 | 每批最大 Chunk 数 |

#### 8.5.2 SwiftKV 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `window_size` | usize | 4 | SIKV 窗口大小 |
| `enable_across_kv` | bool | true | 启用 AKV |
| `similarity_threshold` | f32 | 0.9 | AKV 相似度阈值 |
| `precision_guard` | f32 | 0.1 | 精度损失阈值 (PPL) |

#### 8.5.3 LMCache 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `l1_capacity_mb` | usize | 100 | L1 容量 (MB) |
| `l2_capacity_mb` | usize | 10000 | L2 容量 (MB) |
| `l3_backend` | L3Backend | Redis | L3 后端 |
| `cache_prefix_len` | usize | 512 | 缓存前缀长度 |

---

## 9. 参考文献

1. [CLOCK-Pro: An Effective Improvement of the CLOCK Replacement](https://www.usenix.org/legacyurl/clock-pro-effective-improvement-clock-replacement) - USENIX 2005
2. [vLLM 架构及源码 - Scheduling](https://zhuanlan.zhihu.com/p/28180951363) - Sequence Group 设计
3. [Online Scheduling for LLM Inference with KV Cache Constraints](https://arxiv.org/abs/2502.07115) - arXiv:2502.07115
4. [Clock2Q+: A Simple and Efficient Replacement Algorithm](https://www.arxiv.org/pdf/2511.21958) - arXiv:2511.21958
5. [2D Management of KV-Cache via Layer-wise Optimization](https://iclr.cc/virtual/2025/poster/30707) - ICLR 2025
