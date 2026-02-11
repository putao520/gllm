# 调度器重构开发计划 (Development Plan)

> **基于 SPEC**: SPEC/01-REQUIREMENTS.md, SPEC/02-ARCHITECTURE.md, SPEC/03-DATA-STRUCTURE.md
> **设计文档**: `.serena/memories/scheduler_refactor_design.md`

## 阶段概览

| 阶段 | 任务 | 优先级 | 预估复杂度 |
|------|------|--------|------------|
| P0 | 删除 vllm2024.rs 冗余代码 | 🔴 高 | 低 |
| P1 | BatchOrderPolicy 确定性批处理 | 🔴 高 | 低 |
| P2 | KvPipeline 双管线分离 | 🟡 中 | 中 |
| P3 | SessionKvCache 会话级复用 | 🟡 中 | 中 |
| P4 | KvPrefixIndex 前缀树索引 | 🟡 中 | 高 |
| P5 | ChunkedConfig 融合 | 🟢 低 | 中 |
| P6 | SwiftKV 泛型化 | 🟢 低 | 低 |

---

## P0: 删除 vllm2024.rs 冗余代码

**目标**: REQ-SCHED-015

### 删除列表

```
src/scheduler/vllm2024.rs:
  ❌ LMCacheConfig
  ❌ LmcacheState
  ❌ CacheEntry
  ❌ CacheHit
  ❌ CacheLevel
  ❌ L3Backend (enum)
  ✅ ChunkedConfig (保留)
  ✅ ChunkedState (保留)
  ✅ SwiftKVConfig (保留)
  ✅ SwiftKvState (保留，需泛型化)
```

### 步骤

1. 搜索 LMCache 相关类型的引用
2. 确认无外部依赖后删除
3. 更新 mod.rs 导出
4. 运行 `cargo check` 确认编译
5. 运行 `cargo test --lib` 确认测试

### 验收

- `cargo check` 通过
- `cargo test --lib` 通过
- 无 LMCache 相关导出

---

## P1: BatchOrderPolicy 确定性批处理

**目标**: REQ-SCHED-017, ARCH-SCHED-BATCH-ORDER

### 新增文件/类型

```rust
// src/scheduler/types.rs 或新增 src/scheduler/batch_policy.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchOrderPolicy {
    #[default]
    StrictRequestIdOrder,
    FifoOrder,
    #[deprecated = "Breaks determinism, use only for benchmarking"]
    ThroughputFirst,
}
```

### 修改点

1. `ContinuousBatcher::build_batch()` 接收 `BatchOrderPolicy` 参数
2. 根据策略排序 `requests`
3. `Executor::run_batch_forward()` 添加顺序校验 `debug_assert!`
4. 默认策略强制为 `StrictRequestIdOrder`

### 验收

- 批内请求按 RequestId 严格升序
- 相同输入 + 不同负载 = 相同输出
- `cargo test` 通过

---

## P2: KvPipeline 双管线分离

**目标**: REQ-KV-003, REQ-SCHED-018, ARCH-SCHED-PIPELINE

### 新增类型

```rust
// src/scheduler/types.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvPipeline {
    #[default]
    Conversation,
    Working,
}
```

### 修改点

1. `VirtualPageId` 扩展为包含 `pipeline` 字段，或新增 `PipelinedVirtualPageId`
2. `GlobalMemoryManager` 按 pipeline 隔离页表
3. 新增 `release_working_pipeline(request_id)` 方法
4. 新增 `prepare_next_turn(session_id)` 方法
5. 更新 HGAL 淘汰策略：Working 优先淘汰

### 验收

- `Working` 页面可独立释放
- `Conversation` 页面跨轮保留
- 测试 Thinking 模型场景

---

## P3: SessionKvCache 会话级复用

**目标**: REQ-KV-002, ARCH-SCHED-SESSION-KV

### 新增类型

```rust
// src/scheduler/session.rs (新文件)

pub type SessionId = u64;

pub struct SessionKvCache {
    pub session_id: SessionId,
    pub pages: Vec<VirtualPageId>,
    pub finalized_position: usize,
}
```

### 新增方法 (GlobalMemoryManager)

```rust
impl GlobalMemoryManager {
    pub fn register_session(&mut self, session_id: SessionId) -> &mut SessionKvCache;
    
    pub fn claim_session_prefix(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        prefix_tokens: usize,
    ) -> Result<Vec<VirtualPageId>, MemoryManagerError>;
    
    pub fn finalize_session_tokens(
        &mut self,
        session_id: SessionId,
        new_finalized_position: usize,
    ) -> Result<(), MemoryManagerError>;
}
```

### 验收

- 会话注册/注销生命周期正确
- `finalized_position` 单调递增
- claim 不能越界
- 测试 AI 编程多轮场景

---

## P4: KvPrefixIndex 前缀树索引

**目标**: REQ-KV-001, ARCH-SCHED-PREFIX-INDEX

### 新增文件

```
src/scheduler/prefix_index.rs
```

### 核心实现

```rust
pub struct KvPrefixIndex {
    root: TrieNode,
}

struct TrieNode {
    children: HashMap<TokenId, Box<TrieNode>>,
    page_ref: Option<PageRef>,
}

struct PageRef {
    virtual_page_id: VirtualPageId,
    offset_in_page: usize,
}

impl KvPrefixIndex {
    pub fn new() -> Self;
    pub fn find_longest_prefix(&self, tokens: &[TokenId]) -> Option<PrefixMatch>;
    pub fn insert(&mut self, tokens: &[TokenId], pages: &[VirtualPageId]);
    pub fn remove_stale(&mut self, stale_pages: &HashSet<VirtualPageId>);
}
```

### 集成点

1. `GlobalMemoryManager` 持有 `KvPrefixIndex`
2. `prepare_prefill_with_auto_reuse()` 调用 `find_longest_prefix()`
3. prefill 完成后调用 `insert()`
4. 页面淘汰时调用 `remove_stale()`

### 验收

- O(n) 查找复杂度
- 能识别 append 关系
- 与 GlobalMemoryManager 正确联动
- 内存占用合理（大规模 token 测试）

---

## P5: ChunkedConfig 融合页面调度

**目标**: REQ-SCHED-016, ARCH-SCHED-GLOBAL-MEM-REFACTOR

### 修改点

1. `ChunkedConfig` 移动到 `memory_manager.rs` 或保留在 `vllm2024.rs`
2. 新增 `plan_prefill(prompt_tokens, chunk_size) -> PrefillPlan`
3. `PrefillPlan` 枚举（FullyResident / Pipelined）
4. 与 PagedScheduler 状态机一致更新

### 验收

- 长 prompt 正确分 chunk
- 页面状态正确更新
- 禁止与 Decode 混批

---

## P6: SwiftKV 泛型化

**目标**: REQ-KV-004

### 修改点

```rust
// 当前
pub fn distill_cpu(&mut self, pages: &[Vec<f32>]) -> DistillOutcome
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32
fn approx_ppl_delta(original: &[Vec<f32>], distilled: &[Vec<f32>]) -> f32

// 目标
pub fn distill_cpu<E: Element>(&mut self, pages: &[Vec<E>]) -> DistillOutcome
fn cosine_similarity<E: Element>(a: &[E], b: &[E]) -> E
fn approx_ppl_delta<E: Element>(original: &[Vec<E>], distilled: &[Vec<E>]) -> E
```

### 约束

- `Element` trait 需要提供 `from_f32()`、`to_f32()` 方法
- 或使用 `num_traits::Float` bound

### 验收

- 编译通过
- f16/bf16 测试通过
- 无 `Vec<f32>` 硬编码签名

---

## 执行顺序

```
P0 → P1 → P6 → P2 → P3 → P4 → P5
│     │     │     │     │     │
删除   确定性  泛型   双管线  会话   前缀树  Chunk融合
冗余   批处理  化     分离    复用   索引
```

**理由**:
1. P0 先清理，减少后续修改干扰
2. P1 是核心约束，影响后续所有批处理
3. P6 独立修改，不依赖其他
4. P2→P3→P4 依赖递增
5. P5 依赖前面所有

---

## 测试策略

### 单元测试（每阶段）

```bash
cargo test --lib
```

### 集成测试（P4 完成后）

```bash
cargo test --test test_executor_swap_flow -- --test-threads=1
cargo test --test test_kv_cache -- --test-threads=1
```

### E2E 测试（全部完成后）

```bash
cargo test --test test_e2e_generator -- --test-threads=1
```

---

## 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 前缀树内存占用 | 大规模 token 场景 OOM | 添加容量限制 + LRU 淘汰 |
| 双管线页表复杂度 | 淘汰逻辑 bug | 增加 invariant 断言 |
| Session 生命周期 | 内存泄漏 | 添加超时自动清理 |
