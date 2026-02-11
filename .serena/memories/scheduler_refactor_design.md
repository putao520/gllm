# 分页调度系统重构设计（待实施）

## 决策背景

### 现状分析（2026-02）

当前存在两套冲突的缓存/分页系统：

| 系统 | 位置 | 设计思想 |
|------|------|----------|
| **GlobalMemoryManager + HGAL** | `memory_manager.rs`, `hgal.rs` | OS 虚拟内存风格 |
| **vllm2024 LMCache** | `vllm2024.rs` | 简单 HashMap LRU |

### 为什么我们的设计更好

1. **虚拟→物理映射**：`VirtualPageId → PageTable → (Tier, PhysicalId)`
2. **多维淘汰策略**：HGAL = LIRS + 工作集检测 + Gang Scheduling
3. **统一三层抽象**：L1(GPU) ↔ L2(CPU) ↔ L3(Disk)
4. **Warm-up 保护**：防止刚 swap-in 的页面立即被淘汰

### vLLM 业界现状

- vLLM V1 已弃用 CPU swap，改用 prefix caching + 重算
- 没有虚拟→物理映射层
- 简单 LRU 淘汰

---

## 重构方向

**核心决策**：删除 `vllm2024.rs`，必要功能融合到 `GlobalMemoryManager`

### 从 vllm2024.rs 保留的功能

#### 1. Chunked Prefill（时间片调度）

```rust
// 配置
pub struct ChunkedConfig {
    pub chunk_size: usize,        // 64 tokens
    pub decode_slots: usize,      // 预留给 decode 的槽位
    pub enable_splitfuse: bool,   // prefill/decode 交织
}

// 状态追踪
pub struct ChunkTracker {
    pub prompt: String,
    pub total_tokens: usize,
    pub chunk_size: usize,
    pub completed_chunks: usize,
    pub pending_tokens: usize,
}
```

#### 2. SwiftKV（KV 蒸馏）- 需泛型化

```rust
// 当前：硬编码 f32
pub fn distill_cpu(&mut self, pages: &[Vec<f32>]) -> DistillOutcome

// 目标：泛型
pub fn distill_cpu<E: Element>(&mut self, pages: &[Vec<E>]) -> DistillOutcome
```

### 完全删除的功能

- `LMCacheConfig` / `LmcacheState` - 被 `GlobalMemoryManager` 取代
- `CacheEntry` / `CacheHit` / `CacheLevel` - 被 `Tier` 取代

---

## Prefill 与 OS 分页的分叉点

### 差异分析

| 维度 | OS 分页 | LLM 推理 |
|------|--------|----------|
| 页面生成 | 被动（page fault） | 主动（Prefill 批量） |
| 访问模式 | 不可预测 | 完全可预测（causal） |
| 写入模式 | 随机写 | Append-only |
| 读取模式 | 局部性 | 全量读之前所有 |
| 生命周期 | 未知 | 明确（请求结束即释放） |

### 融合设计

#### 1. Prefill 感知的页面预规划

```rust
impl GlobalMemoryManager {
    pub fn plan_prefill(
        &mut self, 
        prompt_tokens: usize,
        chunk_size: usize,
    ) -> PrefillPlan {
        let total_pages = prompt_tokens.div_ceil(self.page_size);
        let l1_available = self.tier_usage(Tier::L1).available();
        
        if total_pages <= l1_available {
            PrefillPlan::FullyResident { pages: total_pages }
        } else {
            PrefillPlan::Pipelined {
                l1_pages: l1_available,
                l2_prefetch: total_pages - l1_available,
                chunk_schedule: ...,
            }
        }
    }
}
```

#### 2. Chunk 完成后的页面状态更新

```rust
impl HGALScheduler {
    pub fn on_prefill_chunk_complete(
        &mut self,
        chunk_idx: usize,
        total_chunks: usize,
        pages: &[PageId],
    ) {
        for &page in pages {
            self.mark_accessed(page);
            // 早期 chunk 在整个 prefill 期间不能被淘汰
            if chunk_idx < total_chunks - 1 {
                self.update_page_state(page, None, PageState::Protected);
            }
        }
    }
    
    pub fn on_prefill_complete(&mut self, request_id: RequestId) {
        // 所有页面进入正常淘汰候选
        if let Some(group) = self.sequence_groups.get_mut(&request_id) {
            for &page in &group.pages {
                self.update_page_state(page, Some(request_id), PageState::Active);
            }
        }
    }
}
```

#### 3. Decode 阶段的 Append-Only 模式

```rust
impl GlobalMemoryManager {
    pub fn allocate_decode_token(
        &mut self,
        request_id: RequestId,
        current_tokens: usize,
    ) -> Result<Option<VirtualPageId>, MemoryManagerError> {
        let page_idx = current_tokens / self.page_size;
        let offset_in_page = current_tokens % self.page_size;
        
        if offset_in_page == 0 && page_idx > 0 {
            // 需要新页面
            let virtual_id = VirtualPageId::new(request_id, page_idx);
            let physical_id = self.allocate_page(Tier::L1)?;
            self.bind_virtual_page(virtual_id, Tier::L1, physical_id)?;
            Ok(Some(virtual_id))
        } else {
            Ok(None)
        }
    }
}
```

#### 4. 利用可预测性做预取

```rust
impl GlobalMemoryManager {
    pub fn prefetch_for_attention(
        &mut self,
        request_id: RequestId,
        current_position: usize,
    ) {
        // Decode 时要读取 [0, current_position] 的所有 KV
        let pages_needed = current_position.div_ceil(self.page_size);
        for logical_idx in 0..pages_needed {
            let vpid = VirtualPageId::new(request_id, logical_idx);
            if let Ok((tier, pid)) = self.resolve(vpid) {
                if tier != Tier::L1 {
                    self.schedule_prefetch(tier, Tier::L1, pid);
                }
            }
        }
    }
}
```

---

## 泛型违规修复清单

需要同时修复的 f32 硬编码问题：

1. `vllm2024.rs`: `distill_cpu(&mut self, pages: &[Vec<f32>])`
2. `vllm2024.rs`: `cosine_similarity(a: &[f32], b: &[f32])`
3. `vllm2024.rs`: `approx_ppl_delta(original: &[Vec<f32>], distilled: &[Vec<f32>])`

---

## 设计需求 3：无 Session ID 时的 KV Cache 复用

### 问题

vLLM 用 hash 方式有根本缺陷：hash 只能发现完全相同的 block，无法发现 append 关系。

```
请求A: [t1, t2, t3, t4, t5]     → hash = H1
请求B: [t1, t2, t3, t4, t5, t6] → hash = H2 ≠ H1
                                    ↑
                            即使是 append 也无法复用
```

### 设计：Token 序列前缀树 (Prefix Trie)

```rust
/// 用于快速查找最长公共前缀的索引结构
pub struct KvPrefixIndex {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<TokenId, TrieNode>,
    page_ref: Option<(VirtualPageId, usize)>,  // (页面, 页内 offset)
}

impl KvPrefixIndex {
    /// O(n) 查找最长匹配前缀
    pub fn find_longest_prefix(&self, tokens: &[TokenId]) -> Option<PrefixMatch>;
    
    /// 插入新的 token 序列
    pub fn insert(&mut self, tokens: &[TokenId], pages: &[VirtualPageId]);
}
```

### 融合到 GlobalMemoryManager

```rust
impl GlobalMemoryManager {
    prefix_index: KvPrefixIndex,
    
    /// 智能 prefill：自动检测可复用前缀
    pub fn prepare_prefill_with_auto_reuse(
        &mut self,
        request_id: RequestId,
        tokens: &[TokenId],
    ) -> PrefillPlan {
        // 1. 查找最长匹配前缀
        // 2. 验证页面仍然有效
        // 3. 建立虚拟页面映射（CoW 语义）
        // 4. 无匹配则全量 prefill
    }
}
```

---

## 设计需求 4：Session 级确定性复用

对于 AI 编程等场景，可以确定性知道后续请求是 append。

```rust
pub struct SessionKvCache {
    session_id: SessionId,
    pages: Vec<VirtualPageId>,
    finalized_position: usize,
}

impl GlobalMemoryManager {
    pub fn register_session(&mut self, session_id: SessionId) -> SessionKvCache;
    
    pub fn claim_session_prefix(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        prefix_tokens: usize,
    ) -> Result<Vec<VirtualPageId>, Error>;
    
    pub fn finalize_session_tokens(
        &mut self,
        session_id: SessionId,
        new_finalized_position: usize,
    );
}
```

---

## 设计需求 5：多管线 KV Cache（Thinking vs Content）

### 问题

DeepSeek R1 等 thinking 模型，reasoning 过程不应该污染会话 KV cache。

### 设计：双管线分离

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvPipeline {
    Conversation,  // 会话管线：System/User/Assistant
    Working,       // 工作管线：Thinking/Reasoning（可丢弃）
}

pub struct PipelinedVirtualPageId {
    pub pipeline: KvPipeline,
    pub sequence_id: RequestId,
    pub logical_index: usize,
}

impl GlobalMemoryManager {
    pub fn allocate_page_in_pipeline(
        &mut self,
        pipeline: KvPipeline,
        tier: Tier,
    ) -> Result<PhysicalId, Error>;
    
    pub fn release_working_pipeline(&mut self, request_id: RequestId);
    
    pub fn prepare_next_turn(&mut self, session_id: SessionId) {
        // 1. 释放 Working 管线所有页面
        // 2. 保留 Conversation 管线用于下一轮
    }
}
```

---

## 设计需求 6：Batch 确定性顺序（精度保证）

### 问题

vLLM 的 `reorder_batch_to_split_decodes_and_prefills()` 把 batch 内请求重排，
导致浮点累加顺序变化，破坏确定性和精度。

```
浮点数非结合律：
  Σ(a, b, c) = ((a + b) + c) ≠ ((c + a) + b) = Σ(c, a, b)
```

### gllm 核心原则

1. **准确度 > 吞吐量**：不为调度优化牺牲计算精度
2. **严格因果顺序**：batch 内 attention 必须保证严格因果掩码
3. **确定性调度**：batch 必须按 RequestId 严格排序
4. **优先确定性串行**：宁可串行也不乱序并行

### 设计：确定性批处理策略

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchOrderPolicy {
    /// 严格按 RequestId 排序（确定性，精度优先）
    StrictRequestIdOrder,
    /// 按入队时间排序（确定性，FIFO）
    FifoOrder,
    /// 允许 vLLM 风格重排（性能优先，不推荐）
    #[deprecated = "Breaks determinism"]
    ThroughputFirst,
}

impl ContinuousBatcher {
    pub fn build_batch(
        &mut self,
        scheduler: &mut PagedScheduler,
        max_batch_size: usize,
        policy: BatchOrderPolicy,
    ) -> ScheduledBatch {
        let mut requests = self.collect_runnable(max_batch_size);
        
        match policy {
            BatchOrderPolicy::StrictRequestIdOrder => {
                requests.sort_by_key(|r| r.id);
            }
            BatchOrderPolicy::FifoOrder => {
                requests.sort_by_key(|r| r.enqueue_time);
            }
            BatchOrderPolicy::ThroughputFirst => {
                // 警告：破坏确定性！
                requests.sort_by_key(|r| (r.is_prefill, r.id));
            }
        }
        
        ScheduledBatch { requests, ... }
    }
}
```

### Attention 确定性保证

```rust
impl<B: Backend<E>, E: Element> Executor<B, E> {
    fn run_batch_forward(&mut self, batch: &BatchInput) -> ExecutorResult<Vec<LogitsHandle>> {
        debug_assert!(
            batch.sequences.windows(2).all(|w| w[0].request_id < w[1].request_id),
            "Batch must be ordered by request_id for deterministic computation"
        );
        
        // 使用确定性 attention kernel
        // - 不使用 FlashAttention 的随机丢弃
        // - 不使用乱序归约
        // - 使用 Kahan 累加减少浮点误差
        self.backend.batch_forward_deterministic(batch, ...)
    }
}
```

---

## 实施计划

1. 删除 `vllm2024.rs` 中冗余的 LMCache 部分
2. 保留 ChunkedConfig/ChunkedState 并融合到页面调度
3. 泛型化 SwiftKV 的 distill 函数
4. 实现 KvPrefixIndex（前缀树索引）
5. 实现 SessionKvCache（会话级复用）
6. 实现 KvPipeline（多管线分离）
7. 强制 BatchOrderPolicy::StrictRequestIdOrder 为默认
8. 更新 Executor 使用确定性 attention kernel