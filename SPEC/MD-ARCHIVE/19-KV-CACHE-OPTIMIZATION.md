# KV Cache 智能优化 — Epilogue 驱动的动态稀疏 + 混合精度 + 页级表达 (SSOT)

> **铁律**: 所有 KV Cache 优化信号由 Epilogue 白嫖网络零开销产出（§5）。不存在独立的信号采集通道。
> 所有优化决策在 Rust 调度器的 `build_batch` 阶段执行（非 JIT 内部），编译时按 precision tier 生成 Variant，运行时零 dtype 分支。
> **与现有设计的关系**: 本文档是 `06-RUNTIME.md §4 KV Cache 管理` 和 `05-OPTIMIZATIONS.md §4.3 KIVI / §5 Epilogue 白嫖` 的深化，不替代已有设计。

## 参考文献

| ID | 论文 | 会议 | gllm 借鉴点 |
|----|------|------|-------------|
| LeanKV | Unifying KV Cache Compression with Hetero-KV | 2024/2025 | per-head 动态稀疏 + K/V 异构量化 3-11x 压缩 |
| ChunkKV | Semantic-Preserving KV Cache Compression | NeurIPS 2025 | page 语义分组 + attention weight 重要性评分 + 跨层 index 复用 |
| MUSTAFAR | Promoting Unstructured Sparsity for KV Cache Pruning | NeurIPS 2025 | 非结构化稀疏 70% 零损失 + bitmap 格式 |
| KVTuner | Sensitivity-Aware Layer-Wise Mixed-Precision | ICML 2025 | 层间敏感度差异 → 混合精度 3.25-bit 近乎无损 |
| MiniKV | 2-Bit KV Cache via Compression and Token Eviction | ACL 2025 | 量化 + eviction 混合 >80% 压缩 |
| KVzip | Query-Agnostic KV Cache Compression | NeurIPS 2025 Oral | 查询无关压缩，system prompt 压缩一次复用 |
| Lexico | Extreme Compression via Sparse Coding | ICML 2025 | 通用字典稀疏编码 KV 极端压缩 |
| KVmix | Gradient-Based Layer Importance-Aware Mixed-Precision | 2025 | 梯度驱动层间精度分配 |
| MixKVQ | Query-Aware Mixed-Precision KV Quantization | 2025 | 查询感知 + 通道量化难度双因子 |
| PagedEviction | Structured Block-wise KV Cache Pruning | EACL 2026 | 块级结构化 eviction（与 gllm page 对齐） |
| CacheSlide | Cross Position-Aware KV Cache Reuse | FAST 2026 | 位置编码对齐解决前缀复用位置漂移 |
| IceCache | Semantic Token Clustering KV Management | ICLR 2026 | 语义聚类 token 组替代逐 token 管理 |
| SinkQ | Dynamic Sink Token Protection | 2025 | sink token 动态识别 + FP16 不可量化 |
| ThinKV | Thought-Adaptive KV Cache Compression | 2025 | 思维步重要性分级 + 量化 eviction 混合 |
| SHMQ+ | Token-Aware Static Hierarchical Mixed-Precision | 2025 | outlier 量级驱动的 token 级分层精度 |

## §1 核心洞察

### §1.1 三层正交压缩维度

最新研究（2025-2026）揭示 KV Cache 压缩存在三个正交维度，可以独立组合：

```
维度 1: Token 维度（哪些 token 保留）
  → Eviction: 丢弃不重要 token 的 KV
  → 稀疏: 保留但置零不重要通道
  → 来源: ChunkKV (语义分组), PagedEviction (块级), IceCache (聚类)

维度 2: 精度维度（保留的 token 用多少 bit）
  → 量化: FP16 / FP8 / 4-bit / 2-bit
  → 异构: K 和 V 不同量化策略
  → 来源: KVTuner (层间), KVmix (梯度), MixKVQ (查询感知), LeanKV (per-head)

维度 3: 表示维度（KV 向量的编码方式）
  → 原始: 直接存储
  → 稀疏编码: 字典 + 稀疏系数
  → 来源: Lexico (通用字典), MUSTAFAR (bitmap)
```

### §1.2 gllm 的 Epilogue 白嫖优势

所有论文的核心挑战是：**如何获取 token 重要性信号**。现有方案需要额外前向传播（KVzip）、离线校准（KVTuner）、训练（KVmix）。gllm 的 Epilogue 白嫖网络在推理过程中零开销产出了所有信号：

```
论文需要额外计算 ←→ gllm 已有信号（零开销）

ChunkKV: "计算 attention weight 之和评分"
  = gllm Epilogue softmax_max + centroid（已在 Softmax Epilogue 写入 KvPageHeader）

MUSTAFAR: "分析 channel 方差确定稀疏模式"
  = gllm Epilogue row_l1_norm + row_max（已在 FFN GEMM Epilogue 写入）

SinkQ: "识别 attention sink token"
  = gllm Epilogue entropy（低 entropy = sink） + softmax_max（高峰值 = sink）

KVTuner: "离线测算层间 KV 敏感度"
  = gllm Epilogue delta_rho（Residual Epilogue 跨层能量差）+ dead_neuron_count

LeanKV: "per-head 动态稀疏"
  = gllm Softmax Epilogue per-head entropy（已在 KvPageHeader 按页记录）
```

### §1.3 Page = 自然压缩单元

| 论文粒度 | gllm 对应 | 优势 |
|---------|----------|------|
| ChunkKV 10-token chunk | **page (默认 128 tokens)** | 更大的语义单元，重要性评估更稳定 |
| PagedEviction block | **page** | 完全对齐，块级 eviction = page 级淘汰 |
| MUSTAFAR per-element bitmap | **KvPageHeader.channel_bitmap** | page header 携带稀疏掩码 |
| Lexico dictionary coding | **PagedKvPool + 字典页** | pool 内可分配字典页 |

## §2 Page 级混合精度存储模型

### §2.1 KvPageHeader 遥测与量化语义

> **📌 物理布局 SSOT**: `KvPageHeader (56B)` 完整物理布局定义在 `03-DATA-STRUCTURE §8.0`。
> 本节仅描述 Epilogue 遥测区域和量化元数据区域的**语义**，不重复物理布局。

Epilogue 遥测区域 (offset 8..24, 16B) — Softmax/Residual/FFN Epilogue 白嫖写入：
- `entropy_avg`: 注意力分散度（Softmax Epilogue 写入）
- `centroid_pos`: 注意力重心位置（Softmax Epilogue 写入）
- `softmax_max_avg`: 注意力峰值（Softmax Epilogue 写入）
- `delta_rho_avg`: 跨层能量差（Residual Epilogue 写入）
- `dead_ratio`: 死神经元比例（FFN Epilogue 写入）
- `importance_score`: 综合重要性评分 0-255
- `head_entropy_max/min`: per-head 最大/最小 entropy

量化元数据区域 (offset 24..36, 12B) — Rust 调度器写入：
- `precision_tier`: 精度等级（见 §2.2）
- `sink_mask`: page 内 sink token 位掩码
- `channel_bitmap_lo`: 通道稀疏掩码低 32 位 (MUSTAFAR)
- `k_scale_offset`: per-channel K scale 在页内偏移
- `v_scale_factor`: per-token V scale 指数

调度元数据区域 (offset 36..44, 8B):
- `pipeline_id`: 所属管线 (Conversation/Working)
- `layer_mask`: 有效层位掩码（跨层共享页时标识）
- `tier_age`: 精度等级赋值后的 tick 计数
- `deopt_flags`: Deopt 标志

### §2.2 PrecisionTier — 页级精度等级

```rust
#[repr(u8)]
pub enum PrecisionTier {
    /// 全精度 — sink token / 刚 prefill 的页
    /// KV 数据: [f16; page_size * 2 * num_kv_heads * head_dim]
    FP16 = 0,

    /// 8-bit 量化 — 高重要性正常 token
    /// KV 数据: [u8; page_size * 2 * num_kv_heads * head_dim] + scale metadata
    FP8 = 1,

    /// 4-bit KIVI 量化 — 正常 token (默认)
    /// K: per-channel 4-bit + scale[f16; num_kv_heads * head_dim]
    /// V: per-token 4-bit + scale[f16; page_size]
    KIVI4 = 2,

    /// 2-bit 激进量化 — 低重要性 / Working 管线
    /// K: per-channel 2-bit + scale
    /// V: per-token 2-bit + scale
    KIVI2 = 3,

    /// 非结构化稀疏 (MUSTAFAR) — channel_bitmap 标记零通道
    /// 仅存储非零通道 (bitmap + compressed data)
    Sparse = 4,

    /// 字典稀疏编码 (Lexico) — KV 向量 = 字典系数
    /// 存储: [dict_id: u16; num_codes] + [coefficient: f16; num_codes]
    Dictionary = 5,

    /// 已 eviction — 页数据无效，可被回收
    Evicted = 6,
}
```

### §2.3 Page 内数据布局（按 tier 变体）

```
FP16 page (基准):
  [K_layer0_head0..headN | K_layer1_... | ... | V_layer0_head0..headN | ...]
  总大小 = page_size * num_layers * 2 * num_kv_heads * head_dim * 2 bytes

KIVI4 page (默认):
  [K_data: 4bit packed | K_scales: f16[num_channels] | V_data: 4bit packed | V_scales: f16[page_size]]
  总大小 ≈ FP16 * 0.3

KIVI2 page:
  [K_data: 2bit packed | K_scales: f16[num_channels] | V_data: 2bit packed | V_scales: f16[page_size]]
  总大小 ≈ FP16 * 0.18

Sparse page (MUSTAFAR):
  [channel_bitmap: u32 | nonzero_K_data | nonzero_V_data | offset_table]
  总大小 ≈ FP16 * (1 - sparsity_ratio) * 0.5

Dictionary page (Lexico):
  [dict_id: u16[num_codes] | coefficients: f16[num_codes]]
  总大小 ≈ FP16 * 0.05
```

## §3 动态稀疏 + 量化协同

### §3.1 Epilogue 信号 → Page 重要性评分

每个 decode step 结束后，Softmax Epilogue 已经将遥测数据写入该 page 对应的 KvPageHeader。Rust 调度器在 `build_batch` 时消费这些信号：

```
importance_score 计算 (0-255):

  // 信号来源: 全部来自 Epilogue 白嫖 (§5), 零额外计算
  let attention_concentration = 1.0 - (entropy_avg / max_entropy);  // 高集中度 = 重要
  let sink_indicator = softmax_max_avg > SINK_THRESHOLD;            // 峰值 = sink
  let stability = 1.0 - delta_rho_avg;                              // 低 Δρ = 稳定 = 可降级
  let active_heads = head_entropy_max - head_entropy_min;           // 头间差异 = 语义丰富

  importance_score = clamp(
    attention_concentration * 120       // 0-120: 注意力集中度权重最高
    + sink_indicator * 80               // +80: sink token 额外加成
    + active_heads * 30                 // +30: 语义丰富度
    - stability * 40,                   // -40: 过于稳定的可降级
    0, 255
  )
```

### §3.2 PrecisionTier 自动升降级

Rust 调度器在每个 wave 间执行 tier 升降决策：

```
tier_upgrade/page 降级决策 (在 build_batch 中执行):

  if importance_score > 200 || sink_mask != 0:
      target_tier = FP16      // 高重要性 / sink → 全精度
  else if importance_score > 150:
      target_tier = FP8       // 中高重要性 → 8-bit
  else if importance_score > 80:
      target_tier = KIVI4     // 正常 → 4-bit (默认)
  else if importance_score > 40:
      target_tier = KIVI2     // 低重要性 → 2-bit
  else if importance_score > 15:
      target_tier = Sparse    // 很低 → 稀疏
  else:
      target_tier = Evicted   // 极低 → eviction 候选

层间精度调制 (KVTuner/KVmix 启发):
  浅层 [0..L/3]:    tier 不可低于 FP8 (理解层对精度敏感)
  中层 [L/3..2L/3]: tier 不可低于 KIVI4
  深层 [2L/3..L]:   无下限 (生成层钝感)
```

### §3.3 Per-Head 动态稀疏 (LeanKV 启发)

```
head_entropy_max - head_entropy_min > HEAD_SPARSITY_THRESHOLD
  → 该 page 的某些 attention head 几乎无贡献
  → channel_bitmap 中标记这些 head 对应的通道
  → Attention decode 时跳过这些 head 的 KV 读取
  → 节省: 50-70% KV 读取带宽 (MUSTAFAR 验证)

JIT 实现:
  Compact→Execute→Scatter 分组时:
    Sparse tier 的 page 打包到同一 Mega-Kernel Variant
    该 Variant 的 Attention kernel 使用 channel_bitmap mask 跳过零通道
  非 Sparse tier 的 page 走标准 Variant
  → 运行时零分支，编译时 Variant 隔离
```

## §4 精度分级存储

### §4.1 四维交叉决策矩阵

gllm 已有的三个分类维度 + 新增的精度维度，形成四维决策：

```
维度 1: KvPipeline (§4.4 已有)
  Conversation → 跨轮保留 → 最低 FP8
  Working      → 单轮释放 → 可 Sparse/Evicted

维度 2: PageState (§2.1 已有)
  Active    → 正在参与计算 → FP16
  Standby   → 等待使用     → KIVI4 (节省内存)
  Protected → 工作集保护   → KIVI4
  Swapped   → 换出到 CPU   → KIVI2 (压缩传输)
  Warm      → 刚换入保护期 → FP16

维度 3: 层位置 (KVTuner/KVmix 启发)
  浅层理解区  → FP16/FP8
  中层过渡区  → FP8/KIVI4
  深层生成区  → KIVI4/KIVI2/Sparse

维度 4: Token 类型 (SinkQ/ChunkKV 启发)
  Sink token (Epilogue 检测)    → FP16 锁定
  Heavy hitter (centroid 检测)  → FP8
  Normal token                  → KIVI4
  Filler token (低 entropy)     → KIVI2/Sparse
```

**决策优先级**: Token 类型 > Pipeline > 层位置 > PageState

### §4.2 System Prompt 压缩复用 (KVzip + CacheSlide 启发)

```
场景: 100 个请求共享 system prompt "你是一个有用的AI助手..." (128 tokens)

传统: 每请求独立 prefill → 128 × 100 = 12,800 token 重复计算
KvPrefixIndex: 前缀命中 → 复用 page → 零重复计算

KVzip 增强:
  system prompt 页在首次 prefill 后标记为 query-agnostic
  → 执行一次 KVzip importance scoring
  → 保留高重要性 page (FP16)
  → 压缩低重要性 page (KIVI2/Sparse)
  → 压缩后 page 的 ref_count = 活跃请求数
  → 后续请求直接引用压缩后的 page

CacheSlide 增强:
  问题: 不同请求的 system prompt 位置不同 → RoPE 注入不同位置编码 → KV 不同
  解决: ChunkKV 式 CCPE (Chunked Contextual Position Encoding)
    → system prompt page 使用 position-agnostic 编码
    → Weighted Correction Attention 在 decode 时修正位置偏移
    → gllm 实现: RoPE Epilogue 检测到 system prompt range → 跳过 RoPE 注入
                  → 在 attention 计算时通过 Correction Add 补偿
```

### §4.3 跨层 Index 复用 (ChunkKV 启发)

```
观察: 相邻层的 attention pattern 高度相似 (相关性 > 0.95)
推论: layer N 的 page importance_score 近似等于 layer N+1

优化:
  1. 仅在关键层 (每 K 层) 完整评估 importance_score
  2. 中间层复用最近关键层的评分
  3. 减少 Epilogue → Rust 的遥测读取频率

关键层选择: 由 Epilogue inter-layer similarity 检测决定
  每个模型只需要一次离线校准，确定 K 值（通常 K=4，即每 4 层评估一次）
```

## §5 Epilogue 白嫖信号 → KV 优化消费链

### §5.1 信号产出到消费的完整数据流

```
JIT 推理 (每个 decode step):
  Softmax Epilogue 尾段 (~5 条 SIMD 指令):
    → 计算 entropy, centroid, softmax_max, head_entropy_range
    → STG 写入当前 page 的 KvPageHeader.telemetry 区域
  Residual Epilogue 尾段 (~3 条 SIMD 指令):
    → 计算 delta_rho, cosine_similarity
    → STG 写入 KvPageHeader.telemetry
  FFN Gate Epilogue 尾段 (~3 条 SIMD 指令):
    → 计算 dead_neuron_count, row_l1_norm, row_max
    → STG 写入 KvPageHeader.telemetry

Rust 调度器 (wave 间, build_batch 阶段):
  1. 读取所有 active page 的 KvPageHeader.telemetry
  2. 计算 importance_score (§3.1)
  3. 执行 tier 升降决策 (§3.2)
  4. 执行数据格式转换 (如 KIVI4 → KIVI2 需要 requantize)
  5. 更新 channel_bitmap (per-head 稀疏)
  6. 按 tier 分组 → 选择对应 Mega-Kernel Variant
  7. Compact→Execute→Scatter 启动下一 wave

JIT 消费 (下一 wave 的 Mega-Kernel 内):
  Attention decode 读取 KV page:
    → 根据 Variant (编译时 bake) 使用对应的 dequant 逻辑
    → FP16 Variant: 直接 load
    → KIVI4 Variant: load 4-bit + scale → dequant to f32 in registers
    → KIVI2 Variant: load 2-bit + scale → dequant to f32
    → Sparse Variant: load bitmap + nonzero data → scatter to f32
  所有 dequant 在 GEMM/Attention 计算的 load 微内核中完成 (寄存器内)
  → 零额外内存搬运
```

### §5.2 量化写回路径

```
Prefill / Decode KV 写入:
  QKV GEMM 输出 (f32 寄存器) → Epilogue 尾段:
    1. 计算 per-channel max (K) 或 per-token max (V)
    2. 量化到目标精度 (由当前页 tier 决定)
    3. STG 写入 page data 区域
    4. STG 写入 KvPageHeader.telemetry

关键: 量化在 Epilogue 尾段的寄存器中完成，不经过中间 buffer
  → 与现有 Epilogue 白嫖架构一致 (§5 核心原则)
  → 新增开销: ~10 条 SIMD 指令/层 (量化 + 写 scale)
```

## §6 Mega-Kernel Variant 编译策略

### §6.1 编译时 Variant 矩阵

每个 PrecisionTier 对应一个独立的 Mega-Kernel Variant。Variant 间的差异仅在 Attention 层的 KV load 微内核：

```
Variant 名称           | Attention KV load    | FFN/GEMM/其他 | 编译产物大小
FP16_VARIANT          | 直接 f16 load        | 不变          | 基准
FP8_VARIANT           | f16 load + dequant   | 不变          | 基准 + ~2KB
KIVI4_VARIANT         | 4-bit load + dequant | 不变          | 基准 + ~4KB
KIVI2_VARIANT         | 2-bit load + dequant | 不变          | 基准 + ~4KB
SPARSE_VARIANT        | bitmap + scatter     | 不变          | 基准 + ~6KB
```

**L1i 预算** (05-OPTIMIZATIONS.md §3.5):
- 所有 Variant 共享 FFN/Norm/RoPE/Embed/lm_head 代码段
- 仅 Attention KV load 部分不同 (~2-6KB 差异)
- 每次只执行一个 Variant → 不超出 L1i 80% 预算

### §6.2 Variant 选择 = Compact 分组

```
build_batch 阶段:
  let pages = collect_all_active_pages();
  let tiers: HashMap<PrecisionTier, Vec<PageId>> = group_by_tier(pages);

  // 按优先级选择 Variant (最多占 60% decode budget)
  // 混合 tier 的请求通过 Scatter 机制还原位置
  let variant = select_variant_by_majority(tiers);

  // Compact: 按 variant 的 KV load 逻辑重排 batch
  // Execute: 运行对应 Variant 的 Mega-Kernel
  // Scatter: 还原原始 batch 位置
```

### §6.3 混合 tier 请求处理

当同一 batch 内有不同 tier 的 page 时：

```
方案 A (简单): 多次 launch
  按 tier 分组 → 每个 tier 独立 Mega-Kernel launch
  → 简单但多次 launch 有驱动开销

方案 B (推荐): 主 Variant + 通用 dequant 桥接
  选择数量最多的 tier 作为主 Variant
  少数 tier 的 page 在 build_batch 阶段预 dequant 到临时 FP16 buffer
  → 单次 launch，少数 page 有额外搬运，但保持 L1i 预算

方案 C (远期): Variant 内嵌 tier switch
  JIT 编译的 Attention KV load 中嵌入有限 tier 分支
  通过 Compact 将相同 tier 的 page 连续排列 → cache line 友好
  → 复杂但最优，留作 Phase 2
```

## §7 REQ 清单

### REQ-KV-OPT-001: KvPageHeader 扩展 48B

KvPageHeader 从 40B 扩展为 48B，新增 quant_meta 区域（precision_tier, sink_mask, channel_bitmap_lo, k_scale_offset, v_scale_factor）和调度元数据区域（pipeline_id, layer_mask, tier_age, deopt_flags）。

**文件**: `src/kv_cache/mod.rs`

### REQ-KV-OPT-002: Epilogue 遥测 → importance_score 评分

Rust 调度器在 wave 间读取 KvPageHeader.telemetry，计算 0-255 的 importance_score，回写到 KvPageHeader.importance_score 字段。

**文件**: `src/scheduler/kv_optimizer.rs` (新增)

### REQ-KV-OPT-003: PrecisionTier 自动升降级

基于 importance_score + 层位置 + KvPipeline + PageState 四维交叉决策，自动升级/降级 page 的 precision_tier。升降级时执行 requantize 数据转换。

**文件**: `src/scheduler/kv_optimizer.rs`

### REQ-KV-OPT-004: KIVI4/2 量化写回 Epilogue

QKV GEMM Epilogue 尾段扩展：在 KV 写入 page 时根据当前页 precision_tier 执行 KIVI 异构量化（K per-channel + V per-token），量化在寄存器内完成，STG 写入压缩数据 + scale。

**文件**: `gllm-kernels/src/compiler/codegen/vm/plan_lower.rs` Epilogue 尾段

### REQ-KV-OPT-005: Per-Head 稀疏 Bitmap (MUSTAFAR)

Epilogue head_entropy_range 差异大于阈值时，设置 KvPageHeader.channel_bitmap 标记低活跃 head。Sparse Variant 的 Attention KV load 使用 bitmap 跳过零通道。

**文件**: `src/scheduler/kv_optimizer.rs` + `gllm-kernels/src/compiler/codegen/vm/plan_lower.rs`

### REQ-KV-OPT-006: Sink Token 动态保护 (SinkQ)

importance_score > 200 或 softmax_max_avg > SINK_THRESHOLD 的 page 标记 sink_mask。sink_mask 非零的 page precision_tier 锁定为 FP16，禁止降级。Epilogue 首次写入时检测。

**文件**: `src/scheduler/kv_optimizer.rs`

### REQ-KV-OPT-007: System Prompt 压缩复用 (KVzip)

KvPrefixIndex 匹配的 system prompt 页在首次 prefill 后执行 KVzip importance scoring。高重要性页保持 FP16，低重要性页降级到 KIVI2/Sparse。标记为 query-agnostic，ref_count 引用管理。

**文件**: `src/scheduler/prefix_index.rs` + `kv_optimizer.rs`

### REQ-KV-OPT-008: 跨层 Index 复用 (ChunkKV)

仅在关键层（每 K 层）完整评估 importance_score，中间层复用最近关键层评分。K 值由离线校准确定。

**文件**: `src/scheduler/kv_optimizer.rs`

### REQ-KV-OPT-009: Mega-Kernel Variant 矩阵编译

为每个 PrecisionTier 编译独立 Attention KV load Variant。Variant 间共享 FFN/Norm/RoPE/Embed/lm_head 代码。build_batch 阶段按 tier 分组选择 Variant。

**文件**: `gllm-kernels/src/compiler/codegen/vm/plan_lower.rs` + `src/engine/mega_kernel.rs`

### REQ-KV-OPT-010: Position-Agnostic System Prompt (CacheSlide)

System prompt 页使用 position-agnostic 编码（跳过 RoPE 注入），decode 时通过 Weighted Correction Attention 补偿位置偏移。实现跨请求 KV 复用不受位置编码影响。

**文件**: `gllm-kernels/src/compiler/codegen/vm/plan_lower.rs` RoPE Epilogue + Attention

## §8 实施阶段

```
Phase 1: 基础设施 (与 PagedAttention 接通并行)
  → REQ-KV-OPT-001: KvPageHeader 48B
  → REQ-KV-OPT-002: importance_score 评分
  → REQ-KV-OPT-006: Sink Token 保护

Phase 2: 量化写回 (REQ-PA 端到端接通后)
  → REQ-KV-OPT-004: KIVI4/2 Epilogue 量化写回
  → REQ-KV-OPT-009: Variant 矩阵编译

Phase 3: 动态稀疏 + 自动升降级
  → REQ-KV-OPT-003: PrecisionTier 自动升降级
  → REQ-KV-OPT-005: Per-Head 稀疏 Bitmap

Phase 4: 跨请求复用优化
  → REQ-KV-OPT-007: System Prompt 压缩复用
  → REQ-KV-OPT-008: 跨层 Index 复用
  → REQ-KV-OPT-010: Position-Agnostic 编码
```

## §9 验证

```bash
# Phase 1: KvPageHeader + importance_score
cargo test --lib scheduler::kv_optimizer

# Phase 2: KIVI 量化写回 + Variant
cargo test --test test_e2e_generator -- --test-threads=1

# Phase 3: 动态稀疏
cargo test --lib scheduler::kv_optimizer -- sparse_tier

# 全量回归
cargo test --lib && cargo test --test test_e2e_generator -- --test-threads=1
```

## §10 预期压缩效果

基于论文基准数据 + gllm 四维交叉决策的预期：

| 场景 | FP16 基准 | 预期压缩比 | 近似精度影响 |
|------|----------|-----------|------------|
| 短上下文 (<4K) | 1.0x | 1.0x (全 FP16) | 零损失 |
| 中上下文 (4K-32K) | 1.0x | 2-3x (KIVI4 默认) | 近乎无损 (KVTuner: <0.1 ppl) |
| 长上下文 (32K-128K) | 1.0x | 4-6x (KIVI4 + Sparse + Eviction) | 近乎无损 (MUSTAFAR: 70% 稀疏零损失) |
| 超长上下文 (>128K) | 1.0x | 8-12x (KIVI2 + Sparse + Dictionary) | <5% 质量退化 (LeanKV 验证) |
| System prompt 复用 | 1.0x | 3-4x (KVzip 压缩 + 复用) | 零损失 (query-agnostic) |
