# 权重分页统一 — 现有组件补全与统一 (SSOT)

> **SSOT 声明**: 本文档定义 gllm 将现有分散的权重管理组件统一到 PagedAttention 分页系统的改造计划。
> 不是重新发明，而是将已有的 `WeightTierManager` + `ExpertWeightPrefetcher` + `ExpertThermalManager` + `GlobalMemoryManager` 融为一体。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
权重页跨设备迁移与分布式分页协同:
<a data-xref-id="REQ-DP-004" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-004">REQ-DP-004</a>
(迁移计划生成，权重页跨 Tier/HBM→DRAM→NVMe 迁移) |
<a data-xref-id="REQ-DP-005" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-005">REQ-DP-005</a>
(迁移成本估算) |
<a data-xref-id="REQ-DP-007" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-007">REQ-DP-007</a>
(RDMA 页传输)
</div>

## §0 现有系统全景

### §0.1 已有组件清单

| 组件 | 文件 | 行数 | 职责 | executor 接入状态 |
|------|------|------|------|-------------------|
| `WeightTierManager` | src/loader/weight_tier.rs | 253 | 加载时三级缓存决策 (DeviceLocal/HostLocal/DiskMmap) | ✅ loader/mod.rs:1077 调用 |
| `ExpertWeightPrefetcher` | src/moe/prefetch.rs | 401 | 热度感知预取 (GpuL2/GpuVram/CpuRam/RemoteNode/Evicted) | ⚠️ 被创建，step 中未消费 |
| `PrefetchPipeline` | src/moe/prefetch_pipeline.rs | 323 | 层间 KV block 预取编排 | ✅ executor:2625 |
| `ExpertThermalManager` | src/moe/thermal.rs | 717 | 热度追踪 (Hot/Warm/Cold/Evicted) + 自适应驱逐阈值 | ✅ executor 20+ 处 |
| `MoeHardwareDispatcher` | src/moe/dispatch.rs | 472 | 专家→GPU/CPU 硬件分发 | ✅ executor |
| `ExpertFaultHandler` | src/moe/fault_handler.rs | — | 缺页处理 (suspend/resume/recovery) | ✅ executor |
| `HotPatchManager` | src/moe/hot_patch.rs | — | 冷专家 NOP/Deopt JIT 代码修改 | ✅ executor |
| `UnifiedVirtualPage` | src/scheduler/types.rs | — | 统一页容器 (ExpertWeight 变体已定义) | ❌ ExpertWeight 无实际使用 |
| `GlobalMemoryManager` | src/scheduler/memory_manager.rs | — | KV 页三级 Tier 管理 | ✅ executor |
| `HGALScheduler` | src/scheduler/hgal.rs | — | KV 页 LIRS 驱逐 | ✅ executor |

### §0.2 当前架构图

```
加载时:
  Loader → WeightTierManager::decide() → 上传到对应 Tier
  WeightTierManager 和 GlobalMemoryManager 完全独立，不共享 Tier 追踪

运行时 (MoE):
  ExpertThermalManager 追踪热度 → MoeHardwareDispatcher 分发
  → ExpertFaultHandler 处理缺页 → HotPatchManager NOP/Deopt
  ExpertWeightPrefetcher 被创建但 schedule_prefetch() 在 step 中未被调用

运行时 (KV):
  GlobalMemoryManager 管理 Tier → PagedScheduler 分配/释放
  HGALScheduler LIRS 驱逐 → SequenceGroup Gang-Aware
```

**核心问题**: 权重管理和 KV 分页是两套独立系统，不共享 Tier 容量、不共享驱逐策略、不共享页容器。

## §1 需要补全的 Gap

### Gap 1: WeightTierManager 与 GlobalMemoryManager 的 Tier 统一

**现状**: `WeightTierManager` 有自己的 `device_used`/`host_used` 原子计数器。`GlobalMemoryManager` 的 `TierManager` 也有独立的 L1/L2/L3 容量追踪。两者互不知情。

**跨 SPEC Tier 枚举映射**（三套命名，同一物理层级）：

| 本 SPEC `WeightTier` | 06-RUNTIME `Tier` | 22-PAGE-COMPRESSION `StorageTier` | 物理位置 |
|---|---|---|---|
| `DeviceLocal` | `L1` | `GpuHbm` | GPU HBM（微秒级延迟） |
| `HostLocal` | `L2` | `CpuDram` | CPU DRAM（~10ms PCIe 换入换出） |
| `DiskMmap` | `L3` | `Nvme` | NVMe 磁盘（~100ms 文件 I/O） |

> **统一方向**: `GlobalMemoryManager::TierManager` 的 `Tier::L1/L2/L3` 是运行时全局 SSOT。
> `WeightTier` 和 `StorageTier` 是领域视图，通过 §2.1 的 `decide_via_gmm()` 统一到 GMM 的容量管理。
> 枚举值不同但物理层级 1:1 对应，未来实现时需保证三方 Tier 值的 `From/TryFrom` 转换。

**目标**: 权重页的 Tier 分配通过 `GlobalMemoryManager` 统一执行，`WeightTierManager` 变为查询/决策层（不自己分配容量）。

### Gap 2: ExpertWeightPrefetcher 在 executor step 中未被消费

**现状**: `moe_prefetcher` 字段在 executor 中被创建（executor.rs:571），但 `schedule_prefetch()` 在 `step()` 中从未被调用。

**目标**: 在 MoE dispatch plan 生成后、forward 执行前，调用 `schedule_prefetch()` 生成预取计划，通过 `GlobalMemoryManager::migrate_page` 执行实际迁移。

### Gap 3: UnifiedVirtualPage::ExpertWeight 无实际使用

**现状**: `PagePayloadKind::ExpertWeight` 枚举变体已定义，`UnifiedVirtualPage::expert()` 构造函数存在，但从未被调用。HGAL 中没有任何 ExpertWeight 类型的页被注册。

**目标**: 权重页在加载时注册到 HGAL 的 `page_metadata`，参与 LIRS 评分和驱逐决策。

### Gap 4: 非 MoE 模型（Dense Layer）的权重无管理

**现状**: 权重管理仅针对 MoE 专家权重。Dense 模型的权重（如 Qwen3、Llama）全部常驻 GPU，无换出能力。

**目标**: Dense 层权重也注册为 UnifiedVirtualPage（`PagePayloadKind::DenseLayerWeight`），在内存压力下可换出到 CPU/Disk。

## §2 改造方案

### §2.1 WeightTierManager → WeightTierPlanner（重命名 + 职责变更）

`WeightTierManager` 不再自己追踪容量，而是查询 `GlobalMemoryManager` 的 Tier 可用空间来做决策。

```rust
// 改造前: WeightTierManager 自带容量追踪
pub struct WeightTierManager {
    device_capacity: usize,
    host_capacity: usize,
    device_used: AtomicUsize,    // ← 独立计数，与 GMM 不共享
    host_used: AtomicUsize,      // ← 独立计数
    allocations: Mutex<HashMap<String, WeightAllocation>>,
}

// 改造后: 查询 GlobalMemoryManager 的 Tier 可用空间
impl WeightTierManager {
    pub fn decide_via_gmm(
        &self,
        name: &str,
        size: usize,
        gmm: &GlobalMemoryManager,  // ← 查询 GMM 的 TierUsage
    ) -> UploadDecision {
        let l1 = gmm.tier_usage(Tier::L1);
        if l1.available() >= size {
            return UploadDecision { tier: WeightTier::DeviceLocal, ... };
        }
        let l2 = gmm.tier_usage(Tier::L2);
        if l2.available() >= size {
            return UploadDecision { tier: WeightTier::HostLocal, ... };
        }
        UploadDecision { tier: WeightTier::DiskMmap, ... }
    }
}
```

保留原有的 `decide()` 方法作为无 GMM 时的 fallback（测试兼容），新增 `decide_via_gmm()` 作为主路径。

### §2.2 Executor step 中接入 ExpertWeightPrefetcher

在 `step()` 的 MoE dispatch plan 生成后，调用预取：

```rust
// executor.rs step() 中，dispatch plan 生成后
if let (Some(ref mut prefetcher), Some(ref thermal)) =
    (&mut self.moe_prefetcher, &self.moe_thermal)
{
    // 1. 从 thermal 获取各专家当前热度
    for expert_idx in 0..prefetcher.num_experts() {
        let heat = thermal.state(expert_idx).map(|s| s.heat_level);
        let location = ExpertWeightLocation::from_heat_level(heat);
        prefetcher.update_location(expert_idx, location);
    }

    // 2. 从 dispatch plan 获取本 step 被路由到的专家
    let routed_experts: Vec<usize> = moe_dispatch_plan
        .as_ref()
        .map(|p| p.gpu_experts.iter().map(|e| e.expert_idx).collect())
        .unwrap_or_default();

    // 3. 生成预取请求并执行
    let requests = prefetcher.schedule_prefetch(&routed_experts);
    for req in &requests {
        if req.destination == ExpertWeightLocation::GpuVram
            && req.source != ExpertWeightLocation::GpuL2
        {
            // 通过 GlobalMemoryManager 执行 Tier 迁移
            // self.weight_migrate_to_l1(req.expert_idx);
        }
    }
}
```

### §2.3 权重页注册到 HGAL

在权重加载完成后，为每个权重页注册 HGAL 元数据：

```rust
// loader 加载完成后，executor 初始化时
for (layer_idx, weight_pages) in &self.weight_page_table {
    for (page_idx, physical_id) in weight_pages.iter().enumerate() {
        let page_id = compose_weight_page_id(*layer_idx, page_idx);
        self.scheduler.hgal.update_page_state(
            page_id,
            None,  // 权重不属于任何 request
            PageState::Active,
        );
        // 注册为 UnifiedVirtualPage
        let uvp = UnifiedVirtualPage::expert(*physical_id, model_dtype);
        self.scheduler.hgal.upsert_group(SequenceGroup {
            id: weight_group_id(*layer_idx),
            pages: vec![page_id],
            state: GroupState::Running,
            is_pinned: true,  // Dense 层权重默认 pinned
            pipeline: KvPipeline::Conversation,
            ..
        });
    }
}
```

Dense 层权重 `is_pinned = true`（默认不可驱逐），MoE 专家权重 `is_pinned = false`（可被 HGAL 驱逐）。

### §2.4 HGAL 权重页驱逐策略

权重页按 Per-Page 驱逐（无 Gang 约束），与 KV 页的 Gang-Aware 驱逐共存。

**驱逐优先级**（从先驱逐到后驱逐）:
1. Cold Expert 权重页（`ExpertThermalManager` 标记 Evicted）
2. Standby KV 页
3. Warm Expert 权重页
4. Active KV 页
5. Hot Expert 权重页
6. Dense Layer 权重页（pinned，仅极端内存压力）

`compute_group_priority` 中增加 `payload_kind` 感知：

```rust
fn compute_group_priority(&self, group: &SequenceGroup) -> isize {
    let base_priority = /* 现有逻辑 */;
    let weight_bonus = match self.group_payload_kind(group) {
        Some(PagePayloadKind::ExpertWeight) => -200,  // 专家权重页优先级高于 KV
        Some(PagePayloadKind::DenseLayerWeight) => 5_000,  // Dense 层极难驱逐
        _ => 0,
    };
    base_priority + weight_bonus
}
```

## §3 数据结构

### §3.1 WeightPageId（复用现有 PageId）

不引入新的 ID 类型。通过 `UnifiedVirtualPage` 的 `logical_index` 字段编码 `(layer_idx << 16 | page_idx)`。

### §3.2 PagePayloadKind 扩展

```rust
pub enum PagePayloadKind {
    KvContext,          // 已有
    ExpertWeight,       // 已有，MoE 专家权重
    PromptSystem,       // 已有
    KnowledgeRAG,       // 已有
    DenseLayerWeight,   // 新增：Dense 层权重（Q/K/V/O/FFN/Grounding Norm）
}
```

`DenseLayerWeight` 的 `is_evictable()` 返回 `false`（默认 pinned），仅极端内存压力下由管理员 API 解除 pin。

### §3.2.1 UnifiedVirtualPage.dtype 字段类型修正

**问题**: `UnifiedVirtualPage.quant_type` 字段类型为 `QuantType`（量化格式），但权重页的精度信息是原始 dtype（F32/F16/BF16），不是量化格式。硬编码 `Q8_0` 作为占位值违反了 `ARCH-DTYPE-JIT-TYPED` 铁律。

**修复**: 将 `quant_type` 字段类型从 `QuantType` 改为 `DType`，与模型 `geometry.dtype` 统一。

```rust
// 修改前
pub struct UnifiedVirtualPage {
    pub quant_type: gllm_kernels::quant::QuantType,  // ← 错误：量化格式 ≠ 原始精度
    ...
}

// 修改后
pub struct UnifiedVirtualPage {
    pub dtype: gllm_kernels::types::DType,  // ← 正确：统一使用模型原始 dtype
    ...
}
```

所有构造函数参数从 `quant_type: QuantType` 改为 `dtype: DType`。`register_weight_pages()` 直接使用 `geometry.dtype`。

### §3.2.2 权重零拷贝直传 + 可选 dtype 转换

**核心设计原则**: "什么格式就跑什么格式" — 权重保持原始格式直传 JIT，不做任何中间转换。

**问题**: 旧代码将所有 GGUF 量化权重（Q4_0/Q8_0/Q4K 等）强制 dequantize 到 F32，无视：
1. JIT 已有原生 quantized GEMM 内核（`kquant_matmul`）可直接消费 Q4_0/Q8_0/Q4K 字节
2. 中间 F32 转换是精度损失 + 内存浪费 + 计算浪费
3. MXFP4 已经证明"跳过 dequantize，raw bytes 直传 JIT"路径可行

**双路径设计**:

```
路径 A: 零拷贝直传（默认，compute_dtype == dtype）
    GGUF Q4_0 → raw bytes → JIT quantized GEMM → 结果
    SafeTensors BF16 → raw bytes → JIT BF16 GEMM → 结果
    任何格式 → 原样传入 → JIT 原生内核 → 结果

路径 B: dtype 转换（用户主动覆盖，compute_dtype != dtype）
    GGUF Q4_0 → dequantize F32 → 转换 BF16 → JIT BF16 GEMM → 结果
    SafeTensors BF16 → 转换 FP8 → JIT FP8 GEMM → 结果
    用户指定 → 精度转换 → JIT 目标内核 → 结果
```

**执行逻辑**:

```rust
let needs_dtype_conversion = geometry.compute_dtype != geometry.dtype;

if needs_dtype_conversion {
    // 路径 B: 用户主动要求不同精度 — dequantize → 转换
    let target_dtype = geometry.compute_dtype;
    dequantize_weight_to_dtype(qt, backend, target_dtype)  // F32 → target
} else {
    // 路径 A: 默认 — 零拷贝直传，raw bytes 直接喂 JIT
    ext_ptrs[name] = qt.data.as_ptr();  // 原始字节，无转换
}
```

**支持的全部量化/精度格式**:

| 来源格式 | 直传路径 (默认) | JIT 内核 |
|---------|----------------|---------|
| GGUF Q4_0 | ✅ raw bytes 直传 | kquant_matmul Q4_0 |
| GGUF Q4_1 | ✅ raw bytes 直传 | kquant_matmul Q4_1 |
| GGUF Q5_0/Q5_1 | ✅ raw bytes 直传 | kquant_matmul Q5 |
| GGUF Q8_0/Q8_1 | ✅ raw bytes 直传 | kquant_matmul Q8 |
| GGUF Q2K~Q6K | ✅ raw bytes 直传 | kquant_matmul K-Quant |
| GGUF IQ 系列 | ✅ raw bytes 直传 | kquant_matmul IQ |
| MXFP4 | ✅ raw bytes 直传 | mxfp4 GEMM |
| SafeTensors BF16 | ✅ raw bytes 直传 | BF16 GEMM (VDPBF16PS) |
| SafeTensors F16 | ✅ raw bytes 直传 | F16 GEMM |
| SafeTensors F32 | ✅ raw bytes 直传 | F32 GEMM |

**dtype 转换目标（compute_dtype 覆盖时）**:

| 类别 | DType | 说明 |
|------|-------|------|
| 全精度 | F32 | 标准 32 位浮点 |
| 半精度 | F16, BF16 | 16 位浮点 |
| FP8 | F8E4M3, F8E5M2 | OCP 标准 8 位浮点 |
| FP6 | F6E3M2, F6E2M3 | 6 位浮点 (AMD CDNA4) |
| FP4 | F4E2M1 | 4 位浮点 (NVIDIA Blackwell/AMD CDNA4) |

**混合精度**（Phase 2）: 未来支持 per-layer compute_dtype，允许 attention 层用 BF16、FFN 层用 FP8。
当前 Phase 1: 全局统一 compute_dtype。

**状态**: ✅ Phase 1 已实现
- 默认零拷贝直传：`compute_dtype == dtype` 时量化权重原样传入 JIT
- 可选 dtype 转换：`compute_dtype != dtype` 时触发 dequantize → convert 路径
- `dequantize_weight_to_dtype()` 在 `src/compat/weight_helpers.rs:88-95`
- 799 单元测试全部通过

**数据流 (路径 A: 零拷贝)**:
```
GGUF Q4_0 权重 (raw blocks)
    ↓ qt.data.as_ptr() — 零拷贝
    ↓ ext_ptrs[name] = raw_ptr
    ↓ ext_sizes[name] = raw_len
    ↓
JIT 编译 → weight_blob 打包 → kquant_matmul(Q4_0) → 结果
```

**数据流 (路径 B: dtype 转换)**:
```
GGUF Q4_0 权重
    ↓ dequantize_weight_to_dtype(qt, backend, compute_dtype)
    ↓ 内部: _dequantize_to_f32 → f32_to_typed_bytes
    ↓
Vec<u8> (目标 dtype 字节)
    ↓ ext_ptrs[name] = typed_bytes.as_ptr()
    ↓
JIT 编译 → weight_blob → 目标 dtype GEMM → 结果
```

**dtype 传播链**:
```
模型文件格式 → Loader 自动检测
    ↓
ModelConfig::dtype (存储精度: Q4_0/BF16/F16/...)
ModelConfig::compute_dtype (计算精度: 默认 = dtype)
    ↓ compute_dtype == dtype?
    ├── 是 → 零拷贝直传 (路径 A)
    └── 否 → dequantize + convert (路径 B)
```

### §3.3 Executor 新增字段

```rust
pub struct Executor<B, E> {
    // 已有 ...
    moe_prefetcher: Option<ExpertWeightPrefetcher>,

    // 新增
    /// 权重页表: layer_idx → Vec<PhysicalId>
    weight_page_table: HashMap<usize, Vec<PhysicalId>>,
    /// 权重页是否已注册到 HGAL
    weight_pages_registered: bool,
}
```

## §4 数据流

### §4.1 加载时（改造后）

```
safetensors/GGUF/ONNX 文件
    │
    ├── TensorProvider → ResolvedConfig (已有)
    │
    ├── WeightTierPlanner::decide_via_gmm(name, size, gmm)
    │   ├── 查询 GMM Tier::L1 可用空间
    │   ├── 足够 → DeviceLocal
    │   ├── 不足 → 查 L2 → HostLocal
    │   └── 都不足 → DiskMmap
    │
    ├── Backend::upload_provider(tensor, placement)
    │
    └── Executor::register_weight_pages(layer_idx, physical_ids)
        ├── 创建 UnifiedVirtualPage::expert()/dense_layer()
        ├── HGAL.update_page_state() 注册元数据
        └── HGAL.upsert_group() 注册 group (Dense=pinned, MoE=unpinned)
```

### §4.2 运行时（MoE 推理 step）

```
step()
    │
    ├── ExpertThermalManager.step(route_counts)     // 更新专家热度
    │
    ├── MoeHardwareDispatcher.dispatch()            // 生成 dispatch plan
    │   └── skipped_experts → ExpertFaultHandler    // 缺页记录
    │
    ├── ExpertWeightPrefetcher.schedule_prefetch()  // 🆕 新接入
    │   ├── 输入: 路由到的专家列表 + 当前位置
    │   └── 输出: PrefetchRequest 列表
    │
    ├── PrefetchPipeline.advance_layer()            // KV block 预取（已有）
    │
    ├── run_batch_forward()                         // JIT 执行
    │
    └── HotPatchManager.apply()                     // NOP/Deopt（已有）
```

### §4.3 内存压力下的权重驱逐

```
check_memory_pressure()
    │
    ├── SwiftKV distill_pages()                     // KV 蒸馏（已有）
    │
    ├── HGAL.select_victim_groups(count)            // 统一驱逐
    │   ├── 优先驱逐 Cold Expert 权重 group
    │   ├── 然后 Standby KV group
    │   └── Dense Layer 权重 group (pinned, 最后手段)
    │
    └── GlobalMemoryManager.migrate_page(tier)      // 实际迁移
```

## §5 REQ 清单

### REQ-WP-001: WeightTierManager 新增 decide_via_gmm

**描述**: `WeightTierManager` 新增 `decide_via_gmm()` 方法，查询 `GlobalMemoryManager` 的 TierUsage 做决策，而非使用独立容量计数器。保留原有 `decide()` 方法用于测试。

**验收标准**:
1. `decide_via_gmm()` 正确读取 GMM 的 L1/L2/L3 可用空间
2. 原有 `decide()` 行为不变（测试兼容）
3. loader 加载路径可选使用 `decide_via_gmm()` 或 `decide()`

### REQ-WP-002: 权重页 HGAL 注册

**描述**: 在 executor 初始化完成后，将权重页注册到 HGAL 的 `page_metadata` 和 `sequence_groups`。MoE 专家权重 `is_pinned = false`，Dense 层权重 `is_pinned = true`。

**验收标准**:
1. 权重页出现在 `hgal.page_metadata` 中
2. 权重页 group 出现在 `hgal.sequence_groups` 中
3. Dense 权重 group 的 `is_pinned = true`
4. MoE 权重 group 的 `is_pinned = false`
5. HGAL 驱逐时正确跳过 pinned 权重 group

### REQ-WP-003: PagePayloadKind::DenseLayerWeight

**描述**: 新增 `DenseLayerWeight` 枚举变体，`is_evictable()` 返回 `false`。`UnifiedVirtualPage` 新增 `dense_layer()` 构造函数。

**验收标准**:
1. `PagePayloadKind::DenseLayerWeight` 编译通过
2. `UnifiedVirtualPage::dense_layer(page_id, quant_type)` 构造正确
3. `is_evictable()` 返回 `false`

### REQ-WP-004: HGAL 权重页驱逐优先级

**描述**: `compute_group_priority()` 中增加 `PagePayloadKind` 感知。权重页驱逐优先级低于 KV 页（先驱逐权重，保护 KV）。

**验收标准**:
1. Cold Expert 权重页优先级 < Standby KV 页
2. Dense Layer 权重页优先级 > Active KV 页（极难驱逐）
3. 单元测试验证优先级排序

### REQ-WP-005: Executor step 接入 ExpertWeightPrefetcher

**描述**: 在 `step()` 的 MoE dispatch plan 生成后，调用 `prefetcher.update_location()` + `schedule_prefetch()` 生成预取请求。

**验收标准**:
1. `schedule_prefetch()` 在每个 step 中被调用（当 MoE 模型时）
2. 预取请求的 `source` 和 `destination` 正确反映当前 Tier
3. 日志输出预取决策

### REQ-WP-006: 权重页 Tier 迁移执行

**描述**: 将 `ExpertWeightPrefetcher` 的预取请求转化为 `GlobalMemoryManager::migrate_page()` 调用，执行实际的权重页 Tier 迁移。

**验收标准**:
1. 预取请求触发 `GMM.migrate_page(Tier::L2 → Tier::L1)` 或 `L3 → L2`
2. 迁移后 HGAL 元数据更新（`update_page_state`）
3. 迁移失败时走 `ExpertFaultHandler` 缺页路径

### REQ-WP-007: weight_page_table 权重页表

**描述**: Executor 新增 `weight_page_table: HashMap<usize, Vec<PhysicalId>>`，记录每层的权重物理页映射。加载时填充，运行时由 Tier 迁移更新。

**验收标准**:
1. 加载完成后 `weight_page_table` 包含所有层的权重页
2. Tier 迁移后 `weight_page_table` 中的 PhysicalId 更新
3. mega-kernel 的 `weight_blob_ptr` 从 `weight_page_table` 正确获取

### REQ-WP-008: 内存压力下权重页驱逐

**描述**: `check_memory_pressure()` 在 KV 蒸馏和 swap 回收之前，先尝试驱逐 Cold Expert 权重页。

**验收标准**:
1. `check_memory_pressure()` 查询 HGAL 的 Cold Expert 权重 group
2. 驱逐权重页释放 L1 空间
3. 驱逐后 `weight_page_table` 中对应条目标记为 L2/L3
4. Dense Layer 权重页不被驱逐（pinned 保护）

### REQ-WP-009: 权重页缺页恢复

**描述**: 当推理需要已被驱逐到 L2/L3 的权重页时，通过 `ExpertFaultHandler` 路径恢复。

**验收标准**:
1. 访问 L2/L3 权重页触发 fault handler
2. fault handler 通过 GMM 执行 Tier 迁移（L2→L1 或 L3→L2→L1）
3. 恢复完成后 `weight_page_table` 更新
4. 热度追踪更新（`ExpertThermalManager.reactivate_expert()`）

### REQ-WP-010: 权重页遥测

**描述**: Observer 新增权重页遥测指标：权重页总数、各 Tier 分布、驱逐/恢复次数、平均恢复延迟。

**验收标准**:
1. `update_weight_metrics()` 记录权重页 L1/L2/L3 分布
2. 遥测指标被 `AbsolutePolicy` 消费（高权重驱逐率 → 减小 batch size）
3. 日志输出权重页 Tier 分布摘要

## §6 实施优先级

Phase 1（基础接通）:
- REQ-WP-003: DenseLayerWeight 枚举
- REQ-WP-002: 权重页 HGAL 注册
- REQ-WP-007: weight_page_table

Phase 2（预取接通）:
- REQ-WP-005: Executor step 接入 prefetcher
- REQ-WP-006: 预取 → GMM Tier 迁移

Phase 3（统一驱逐）:
- REQ-WP-001: WeightTierManager decide_via_gmm
- REQ-WP-004: HGAL 权重驱逐优先级
- REQ-WP-008: 内存压力下权重驱逐

Phase 4（缺页 + 遥测）:
- REQ-WP-009: 权重缺页恢复
- REQ-WP-010: 权重页遥测

## §7 与现有 SPEC 的关系

| 本文档 REQ | 依赖的已有 SPEC | 依赖的已有代码 |
|-----------|----------------|---------------|
| WP-001 | SPEC/06-RUNTIME §5.1 (三级 Tier) | WeightTierManager, GlobalMemoryManager |
| WP-002 | SPEC/03-DATA-STRUCTURE §11.2 (UnifiedVirtualPage) | HGALScheduler, PagedScheduler |
| WP-003 | SPEC/03-DATA-STRUCTURE §11.2 | PagePayloadKind, UnifiedVirtualPage |
| WP-004 | SPEC/DOCS/scheduling/hgal-scheduler-algorithm.md | HGAL::compute_group_priority |
| WP-005 | SPEC/02-ARCHITECTURE §15 (MoE Dispatch) | ExpertWeightPrefetcher |
| WP-006 | SPEC/06-RUNTIME §5.1 | GlobalMemoryManager::migrate_page |
| WP-007 | SPEC/08-EXECUTOR | Executor |
| WP-008 | SPEC/06-RUNTIME §5.2 (内存压力) | check_memory_pressure |
| WP-009 | SPEC/02-ARCHITECTURE §15.4 (Fault) | ExpertFaultHandler |
| WP-010 | SPEC/07-OBSERVABILITY | Observer, AbsolutePolicy |

## §8 量化权重页 (与 SPEC 23/35 协同)

> 权重压缩 = 量化。SPEC 23 定义 QuantFormatDescriptor 覆盖 22 种量化格式（AWQ4/GPTQ4/NVFP4 等），
> GEMM 的 JIT prologue（QuantGather，SPEC 24）负责解量化。本节定义量化权重页的分页集成。

### §8.1 量化权重页布局

量化权重页以量化格式存储，页内布局与 `QuantBlockLoad` VmInstr (SPEC 26 §1.2.1) 对齐：

```
QuantizedWeightPage:
  header:
    quant_format: QuantType         // AWQ4/GPTQ4/NVFP4/FP8/...
    block_size: u32                 // 量化块大小 (如 128/32)
    n_blocks: u32                   // 页内量化块数
    scale_dtype: DType              // 缩放因子精度 (FP16/FP32/UE4M3/E8M0)
  data:
    packed_weights: [u8; ...]       // 量化后权重 (bit-packed)
    scales: [scale_dtype; n_blocks] // per-block 缩放因子
    zero_points: [u8; ...]          // per-block 零点 (可选，取决于量化格式)
```

### §8.2 推理时量化权重使用流程

```
推理时量化权重使用流程 (GPU HBM 内):
  1. GEMM 微核需要权重 tile
  2. 从 WeightPage 读取量化数据 (packed_weights + scales + zero_points)
  3. GEMM JIT prologue (QuantGather, SPEC 24) 执行 on-the-fly 解量化
  4. 解量化后权重直接驻留寄存器，送入 Tensor Core GEMM
  5. 无需额外解压 pass — 解量化与 GEMM 融合为单一内核
```

**关键**: 量化权重的"解压"不是独立的解压步骤，而是 GEMM 的 prologue 融合。
这比 LZ4/BitPack 等通用压缩更高效：零额外内存占用，解量化在寄存器中完成。

### §8.3 .gllm 文件的量化权重页映射

从 `.gllm` 文件 (SPEC 36) 加载量化权重时的页映射：

```
.gllm 加载 → 量化权重页映射:
  1. 解析 Tensor Directory 获取每个张量的 quant_format + data_offset + compressed_size
  2. 按 block_size (来自 QuantFormatDescriptor) 计算页内块数
  3. 每个张量对应一个或多个 WeightPage:
     - page_id = sequential
     - file_offset = tensor_data_offset + page_index × page_aligned_bytes
     - codec = None (量化即压缩，无需额外页级压缩)
     - quant_format = from Tensor Directory
  4. 注册到 UnifiedVirtualPage 系统
  5. 推理时按需 mmap 读取 → GEMM prologue 解量化 → Tensor Core 计算
```

### §8.4 REQ 清单 (量化权重页)

| REQ ID | 描述 | 验收标准 | 依赖 |
|--------|------|---------|------|
| REQ-WP-011 | 量化权重页布局 | QuantizedWeightPage header + data 布局与 QuantBlockLoad VmInstr 对齐 | SPEC 23 |
| REQ-WP-012 | GEMM prologue 解量化集成 | QuantGather 从量化权重页解量化 + GEMM 融合，数值精度与 SPEC 10 一致 | SPEC 24 |
| REQ-WP-013 | .gllm 量化权重页映射 | 从 .gllm 文件正确映射量化 WeightPage，推理结果与原格式一致 | SPEC 36 REQ-GLF-003 |
