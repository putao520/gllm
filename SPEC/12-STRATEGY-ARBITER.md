# 策略仲裁器 (Strategy Arbiter)

> **SSOT 声明**: 本文档是 gllm 推理引擎全局优化策略优先级决策的唯一真源。定义 InferenceMode（延迟/吞吐模式）、GraphProfile（模型图拓扑特征提取）、StrategyArbiter（策略优先级仲裁），及其与 HwOptEngine（`02-HARDWARE.md §10`）的集成关系。

## 1. 动机

### 1.1 问题

当前 HwOptEngine 的策略决策完全由**硬件参数**驱动（`DeviceProfile` + `MemoryNetworkSensors` + `ProbeResult`）。这导致两个盲区：

1. **使用场景盲区**: 同一硬件上，本地单请求低延迟推理与服务端高吞吐批处理的最优策略完全相反，但 HwOptEngine 无法区分。
2. **模型图拓扑盲区**: Dense 大模型（Llama-70B）与 MoE 稀疏模型（DeepSeek-V3）的最优资源分配策略截然不同，但 HwOptEngine 仅通过 `ModelConfig` 获取标量参数（hidden_dim、num_layers），未分析图级拓扑特征。

### 1.2 解决方案

引入 **Strategy Arbiter** 作为 HwOptEngine 的前置阶段。Arbiter 输出 `StrategyBias`——一组归一化的资源配额权重，注入 HwOptEngine 的 CostModel，使所有下游求解器的成本评估函数自动偏向场景最优策略。

### 1.3 决策链

```
InferenceMode (用户指定)
  └→ GraphProfiler (模型图静态分析)
       └→ StrategyArbiter (Mode × GraphProfile × DeviceProfile → StrategyBias)
            └→ HwOptEngine (StrategyBias 注入 CostModel → HwOptPlan)
                 └→ JIT Codegen (生成机器码)
```

## 2. InferenceMode — 推理模式

### 2.1 定义

```rust
/// 推理引擎的全局优化目标。模型加载时确定，推理全程不可变。
pub enum InferenceMode {
    /// 极致单请求延迟：batch=1，所有资源服务唯一请求。
    /// 适用场景：本地推理、实时交互、边缘设备。
    Latency,

    /// 极致吞吐：最大化 tokens/second/dollar。
    /// 适用场景：API 服务、批量处理、离线推理。
    Throughput,
}
```

### 2.2 API 入口

```rust
let client = Client::builder()
    .model("Qwen/Qwen3-7B-Instruct")
    .inference_mode(InferenceMode::Latency)  // 新增
    .build()?;
```

**默认值**: 不指定时默认 `InferenceMode::Latency`。

**不可变性**: `InferenceMode` 嵌入 `HwOptPlan`，模型加载时确定，推理期间不可切换。切换模式需 `swap_model()` 重新加载。

### 2.3 模式影响矩阵

| 维度 | Latency | Throughput |
|------|---------|------------|
| **Batch 上限** | 1（固定） | 动态（由 BatchSolver 求解） |
| **GEMM 形态** | M=1 GEMV（memory-bound） | M=batch_size GEMM（compute-bound） |
| **主瓶颈** | 内存带宽 | 计算单元 |
| **融合收益** | 极高（每字节 = 延迟） | 中等（计算主导时带宽非瓶颈） |
| **流水线收益** | 极高（隐藏 DRAM 延迟） | 低（计算足以掩盖访存） |
| **并行度** | 低（M=1 无法切 wave） | 高（大矩阵需饱和全部 SM） |
| **量化激进度** | 极高（4bit 带宽 ÷ 4） | 适度（精度影响批量质量） |
| **KV Cache 策略** | 全量驻留（单序列） | PagedAttention + 驱逐 |
| **Speculative Decoding** | 高价值（加速唯一请求） | 低价值（抢占 batch 容量） |
| **MoE 专家驻留** | 全部常驻（内存充裕） | 激进驱逐（省内存给更多请求） |
| **Tensor Parallelism** | 有价值（降单请求延迟） | 可能不如各卡独立 batch |
| **Prefill 策略** | 尽快完成切 decode | Chunked Prefill 交织 decode |

### 2.4 Roofline 翻转效应

同一模型、同一硬件，两种模式的算术强度分布截然不同：

```
GEMM Arithmetic Intensity:
  AI = 2MNK / ((MK + KN + MN) × elem_bytes)

  Latency  (M=1):  AI ≈ 2N/(K+N+1) ≈ 2    → MemoryBound
  Throughput (M=64): AI ≈ 128NK/(64K+KN+64N) → ComputeBound
```

**REQ-ARB-001**: RooflineAnalyzer 必须接收 `InferenceMode` 以确定 M 值参与 AI 计算。Latency 模式下 `effective_M = 1`；Throughput 模式下 `effective_M` 由 BatchSolver 的预期 batch 大小决定。

## 3. GraphProfile — 模型图拓扑特征

### 3.1 定义

```rust
/// 模型图的静态拓扑特征向量。
/// 由 GraphProfiler 在模型加载时从 OnnxGraph 一次性提取，不可变。
pub struct GraphProfile {
    // ── 计算特征 ──
    /// 单层总 FLOPs / 单层总参数字节 (ops/byte)
    pub compute_density: f64,
    /// 模型总参数量 (bytes)
    pub total_param_bytes: usize,

    // ── 融合机会特征 ──
    /// 可融合算子对数 / 总算子数 (0.0~1.0)
    pub fusion_opportunity: f64,
    /// 平均 epilogue 链长度 (GEMM 后跟的连续 elementwise op 数)
    pub avg_epilogue_chain_len: f64,
    /// FFN 类型
    pub ffn_kind: FfnKind,

    // ── 并行度特征 ──
    /// MoE 专家数 (Dense 模型 = 0)
    pub num_experts: usize,
    /// MoE top-k (Dense 模型 = 0)
    pub moe_top_k: usize,
    /// MoE 层占总层数比例 (0.0~1.0)
    pub moe_layer_ratio: f64,

    // ── 注意力特征 ──
    /// Attention 类型
    pub attention_kind: AttentionKind,
    /// KV head 数 / Q head 数 (MHA=1.0, GQA<1.0, MQA≈0)
    pub kv_q_head_ratio: f64,
    /// head_dim
    pub head_dim: usize,

    // ── 内存特征 ──
    /// 每 token KV cache 字节 = 2 × num_kv_heads × head_dim × num_layers × elem_bytes
    pub kv_bytes_per_token: usize,
    /// 权重复用率 = 每 token 访问的独立权重字节 / 总权重字节
    /// Dense=1.0 (每 token 访问全部权重), MoE<1.0 (只访问 top-k 专家)
    pub weight_reuse_ratio: f64,

    // ── 结构特征 ──
    /// 总层数
    pub num_layers: usize,
    /// hidden_dim
    pub hidden_dim: usize,
    /// 词表大小
    pub vocab_size: usize,
    /// 是否有 tied embeddings (input/output 共享权重)
    pub tied_embeddings: bool,
    /// 残差连接拓扑
    pub residual_kind: ResidualKind,
}
```

### 3.2 辅助枚举

```rust
pub enum FfnKind {
    SwiGLU,       // gate + up + silu + mul + down (Llama/Qwen/DeepSeek)
    GeGLU,        // gate + up + gelu + mul + down
    ReLU,         // up + relu + down (legacy)
    MoESwiGLU,    // MoE routing → per-expert SwiGLU
    MoEGeGLU,     // MoE routing → per-expert GeGLU
}

pub enum AttentionKind {
    MHA,          // Multi-Head Attention (KV heads = Q heads)
    GQA,          // Grouped-Query Attention (KV heads < Q heads)
    MQA,          // Multi-Query Attention (KV heads = 1)
    SlidingWindow { window_size: usize },
}

pub enum ResidualKind {
    PreNorm,      // Pre-LN: Norm → Attn → Add → Norm → FFN → Add
    PostNorm,     // Post-LN: Attn → Add → Norm → FFN → Add → Norm
    DeepNorm,     // DeepNorm: scaled residual
}
```

### 3.3 提取算法

```rust
impl GraphProfiler {
    /// 从模型图和配置一次性提取拓扑特征。
    pub fn profile(graph: &OnnxGraph, config: &ModelConfig) -> GraphProfile
}
```

**提取来源**:

| 特征 | 提取方式 |
|------|---------|
| `compute_density` | 遍历 graph 节点统计 FLOPs（GEMM=2MNK, Norm=3H, ElemWise=H）÷ 参数字节 |
| `fusion_opportunity` | 遍历 graph 边，统计 (producer, consumer) 匹配融合模式的对数 ÷ 总节点数 |
| `avg_epilogue_chain_len` | 对每个 GEMM 节点，沿输出边向前走到第一个非 elementwise 节点，取平均步数 |
| `ffn_kind` | 匹配 FFN 子图模式：检测 gate_proj+up_proj+activation+down_proj 的组合 |
| `num_experts` | `config.num_experts.unwrap_or(0)` |
| `moe_top_k` | `config.num_experts_per_tok.unwrap_or(0)` |
| `moe_layer_ratio` | 统计含 MoE routing 子图的层数 ÷ 总层数 |
| `attention_kind` | 从 `config.num_key_value_heads` 与 `config.num_attention_heads` 推导 |
| `kv_q_head_ratio` | `num_kv_heads / num_q_heads` |
| `kv_bytes_per_token` | `2 × num_kv_heads × head_dim × num_layers × elem_bytes` |
| `weight_reuse_ratio` | Dense: 1.0; MoE: `(shared_params + top_k × expert_params) / total_params` |
| `residual_kind` | 匹配残差连接拓扑：Norm 在 Attention 前（PreNorm）还是后（PostNorm） |

### 3.4 模型图原型

基于提取特征，将模型图归类为 **Graph Archetype**（图原型）。原型不是硬分类，而是一个连续空间中的锚点，用于成本模型插值。

```rust
/// 模型图在优化策略空间中的位置。
/// 每个维度是 [0.0, 1.0] 的归一化权重。
pub struct GraphArchetype {
    /// 计算密集度 (高 = 大 hidden_dim, 长 epilogue 链, Dense)
    pub compute_intensive: f64,
    /// 内存密集度 (高 = 大 KV cache, 低权重复用, MoE)
    pub memory_intensive: f64,
    /// 并行可开发度 (高 = 多专家, 大 batch 适应性)
    pub parallelism_exploitable: f64,
    /// 融合收益度 (高 = 长 epilogue 链, SwiGLU, QKV 可共享)
    pub fusion_profitable: f64,
    /// 流水线价值度 (高 = 小 GEMM, memory-bound, 深层数)
    pub pipeline_valuable: f64,
}
```

**REQ-ARB-002**: GraphArchetype 的五个维度之和不要求为 1.0。每个维度独立归一化，表示该维度的绝对强度。

### 3.5 原型推导算法

```
compute_intensive:
  base = sigmoid((hidden_dim - 2048) / 2048)  // 4096 → 0.73, 8192 → 0.95
  boost = avg_epilogue_chain_len / 8.0         // 长链 → 更多计算
  penalty = moe_layer_ratio × 0.5             // MoE 降低单路径计算密度
  result = clamp(base + boost - penalty, 0.0, 1.0)

memory_intensive:
  kv_factor = sigmoid((kv_bytes_per_token - 1024) / 4096)
  reuse_factor = 1.0 - weight_reuse_ratio     // 低复用 = 高内存压力
  result = clamp(kv_factor + reuse_factor, 0.0, 1.0)

parallelism_exploitable:
  moe_factor = if num_experts > 0 {
      sigmoid((num_experts × moe_top_k - 16) / 64)  // 256×6 → 0.99
  } else { 0.0 }
  batch_factor = 1.0 - compute_intensive × 0.3  // 大 GEMM 已天然并行，不需额外 wave
  result = clamp(moe_factor + batch_factor × 0.3, 0.0, 1.0)

fusion_profitable:
  epilogue = avg_epilogue_chain_len / 6.0
  swiglu = if ffn_kind.is_gated() { 0.3 } else { 0.0 }
  qkv = if kv_q_head_ratio < 1.0 { 0.2 } else { 0.1 }  // GQA QKV 共享 pack_a 更有价值
  norm = if residual_kind == PreNorm { 0.2 } else { 0.0 }  // PreNorm → NormIntoGemm
  result = clamp(epilogue + swiglu + qkv + norm, 0.0, 1.0)

pipeline_valuable:
  small_gemm = sigmoid((2048 - hidden_dim) / 2048)  // 小 hidden → 更需要 pipeline
  depth = sigmoid((num_layers - 16) / 32)            // 深层 → 更多 pipeline 机会
  memory_bound = memory_intensive × 0.5              // memory-bound → pipeline 价值高
  result = clamp(small_gemm + depth × 0.3 + memory_bound, 0.0, 1.0)
```

### 3.6 典型模型原型参考

| 模型 | compute | memory | parallel | fusion | pipeline |
|------|---------|--------|----------|--------|----------|
| Llama-70B (Dense, GQA, 80L) | 0.95 | 0.6 | 0.2 | 0.85 | 0.15 |
| Qwen3-7B (Dense, GQA, 32L) | 0.73 | 0.4 | 0.15 | 0.80 | 0.45 |
| DeepSeek-V3 (MoE 256×6, 60L) | 0.50 | 0.90 | 0.99 | 0.60 | 0.35 |
| Phi-4 (Dense, 3B, 32L) | 0.35 | 0.25 | 0.10 | 0.70 | 0.80 |
| BGE-M3 (Encoder, MHA, 24L) | 0.60 | 0.10 | 0.05 | 0.50 | 0.30 |

## 4. StrategyArbiter — 策略仲裁器

### 4.1 职责

接收 `InferenceMode` × `GraphArchetype` × `DeviceProfile`，输出 `StrategyBias`——一组成本模型调制系数，注入 HwOptEngine 使其自动偏向场景最优策略。

**核心理念**: Arbiter 不直接决定具体策略（那是 HwOptEngine 各 Solver 的职责），而是通过**调整成本函数的权重**间接影响所有决策。这保持了 HwOptEngine 的 Cost-Model 驱动架构不变。

### 4.2 StrategyBias 定义

```rust
/// 策略偏置系数。注入 HwOptEngine CostModel，调制成本评估。
/// 所有系数 > 0.0，1.0 = 中性，< 1.0 = 偏好（降低该策略成本），> 1.0 = 惩罚（提高成本）。
pub struct StrategyBias {
    // ── 资源配额偏置 ──

    /// 融合族偏置: < 1.0 时 CostModel 低估融合的代价（偏好融合）
    pub fusion_cost_scale: f64,
    /// 流水线族偏置: < 1.0 时 CostModel 低估 pipeline 的代价（偏好双缓冲/预取）
    pub pipeline_cost_scale: f64,
    /// 并行族偏置: < 1.0 时 CostModel 低估 wave 分裂的代价（偏好多 wave）
    pub parallelism_cost_scale: f64,

    // ── 寄存器预算偏置 ──

    /// epilogue 深度偏好: > 1.0 时 FusionSolver 更倾向深 epilogue（牺牲 k_depth）
    pub epilogue_depth_preference: f64,
    /// k_depth 偏好: > 1.0 时 GemmSolver 更倾向深 k pipeline（牺牲 epilogue）
    pub k_depth_preference: f64,

    // ── 缓存预算偏置 ──

    /// KV cache 预算比例调制 (> 1.0 = 给 KV 更多 L2)
    pub kv_cache_budget_scale: f64,
    /// 权重预取预算比例调制
    pub weight_prefetch_budget_scale: f64,

    // ── 批处理偏置 ──

    /// batch 大小偏好: 0.0 = 永远 batch=1, 1.0 = BatchSolver 自由决策
    pub batch_flexibility: f64,
    /// decode/prefill 混合比偏置
    pub decode_ratio_scale: f64,

    // ── MoE 专属偏置 ──

    /// 专家驻留激进度: 0.0 = 全部常驻, 1.0 = 激进驱逐
    pub expert_eviction_aggressiveness: f64,
    /// 专家权重预取优先级: > 1.0 = 更积极预取即将使用的专家
    pub expert_prefetch_priority: f64,

    // ── 特殊策略偏置 ──

    /// 投机解码价值: > 1.0 = 更倾向启用 speculative decoding
    pub speculative_decoding_value: f64,
    /// 量化激进度: > 1.0 = 更倾向低比特量化
    pub quantization_aggressiveness: f64,
}
```

### 4.3 仲裁算法

```rust
impl StrategyArbiter {
    pub fn arbitrate(
        mode: InferenceMode,
        archetype: &GraphArchetype,
        profile: &DeviceProfile,
    ) -> StrategyBias
}
```

#### 4.3.1 InferenceMode 基线偏置

首先根据模式生成基线偏置（与模型图无关）：

```
Latency 基线:
  fusion_cost_scale       = 0.5   // 强烈偏好融合（memory-bound 时融合 = 延迟下降）
  pipeline_cost_scale     = 0.6   // 偏好流水线（隐藏 DRAM 延迟）
  parallelism_cost_scale  = 1.5   // 惩罚多 wave（batch=1 切不动）
  epilogue_depth_preference = 1.5 // 深 epilogue（减少中间写回 = 减少延迟）
  k_depth_preference      = 1.3   // 偏好双缓冲（隐藏 pack 延迟）
  kv_cache_budget_scale   = 0.5   // KV cache 小（单序列）
  weight_prefetch_budget_scale = 1.5 // 权重预取重要（memory-bound）
  batch_flexibility       = 0.0   // batch=1 固定
  decode_ratio_scale      = 1.0   // 单请求无混合比概念
  expert_eviction_aggressiveness = 0.0 // 全部常驻
  expert_prefetch_priority = 0.5  // 不需要激进预取
  speculative_decoding_value = 1.5 // 高价值
  quantization_aggressiveness = 1.5 // 激进量化

Throughput 基线:
  fusion_cost_scale       = 1.0   // 中性（compute-bound 时融合收益有限）
  pipeline_cost_scale     = 1.3   // 轻微惩罚（计算足以掩盖）
  parallelism_cost_scale  = 0.5   // 强烈偏好并行（喂满硬件）
  epilogue_depth_preference = 0.8 // 浅 epilogue（省寄存器给 GEMM tile）
  k_depth_preference      = 0.8   // 轻度降低（compute-bound 不需要深 pipeline）
  kv_cache_budget_scale   = 1.5   // KV cache 大（多序列）
  weight_prefetch_budget_scale = 0.8 // 权重自然被 batch 复用
  batch_flexibility       = 1.0   // BatchSolver 自由决策
  decode_ratio_scale      = 1.0   // 混合比自由
  expert_eviction_aggressiveness = 0.8 // 适度驱逐
  expert_prefetch_priority = 1.5  // 激进预取（下一 batch 可能用）
  speculative_decoding_value = 0.3 // 低价值
  quantization_aggressiveness = 0.8 // 适度
```

#### 4.3.2 GraphArchetype 调制

在基线偏置上，根据模型图原型做乘法调制：

```
// 融合族调制
bias.fusion_cost_scale *= lerp(1.0, 0.6, archetype.fusion_profitable)
// fusion_profitable 高 → 融合代价进一步降低 → 更偏好融合

// 流水线族调制
bias.pipeline_cost_scale *= lerp(1.0, 0.6, archetype.pipeline_valuable)
// pipeline_valuable 高 → pipeline 代价进一步降低

// 并行族调制
bias.parallelism_cost_scale *= lerp(1.0, 0.5, archetype.parallelism_exploitable)
// parallelism_exploitable 高 → 并行代价进一步降低

// 寄存器竞争调制 (融合 vs 流水线的寄存器分配)
reg_tension = archetype.fusion_profitable - archetype.pipeline_valuable
if reg_tension > 0.0:
    // 融合更有价值 → 寄存器给 epilogue
    bias.epilogue_depth_preference *= 1.0 + reg_tension × 0.5
    bias.k_depth_preference *= 1.0 - reg_tension × 0.3
else:
    // 流水线更有价值 → 寄存器给 k_depth
    bias.k_depth_preference *= 1.0 + (-reg_tension) × 0.5
    bias.epilogue_depth_preference *= 1.0 - (-reg_tension) × 0.3

// MoE 调制
if archetype.parallelism_exploitable > 0.5:  // MoE 模型
    bias.expert_eviction_aggressiveness *= lerp(1.0, 1.5, archetype.memory_intensive)
    bias.expert_prefetch_priority *= lerp(1.0, 2.0, archetype.memory_intensive)
    // 内存密集的 MoE → 更激进驱逐 + 更激进预取

// KV cache 调制
bias.kv_cache_budget_scale *= lerp(1.0, 1.5, archetype.memory_intensive)

// 量化调制
bias.quantization_aggressiveness *= lerp(1.0, 1.3, archetype.memory_intensive)
// 内存密集 → 更激进量化
```

#### 4.3.3 硬件特征微调

最后根据硬件特征做最终微调：

```
// GPU 有海量寄存器 → 寄存器竞争缓解
if profile.is_gpu():
    bias.epilogue_depth_preference *= 1.2  // GPU 可以更深 epilogue
    bias.k_depth_preference *= 1.2         // GPU 可以更深 pipeline
    // 二者同时提升因为 GPU 寄存器充裕

// CPU 寄存器紧张 → 竞争加剧
if profile.num_simd_regs <= 16:
    reg_scarcity = 1.0 - (profile.num_simd_regs as f64 / 32.0)
    bias.epilogue_depth_preference *= 1.0 + reg_scarcity × 0.3
    bias.k_depth_preference *= 1.0 - reg_scarcity × 0.2
    // 寄存器少 → 更倾向 epilogue（融合直接省带宽）而非 pipeline（需要额外 buffer）

// 大 L1 → 融合更自由
l1_richness = (profile.cache_sizes.0 as f64 / 65536.0).min(2.0)
bias.fusion_cost_scale *= 1.0 / l1_richness.sqrt()
// 大 L1 → TileLevelFusion scratch 更充裕 → 融合代价更低

// 高带宽设备 → pipeline 价值降低
if profile.is_gpu():
    bias.pipeline_cost_scale *= 1.2  // HBM 带宽高，pipeline 隐藏的延迟占比小
```

### 4.4 完整仲裁签名

```rust
impl StrategyArbiter {
    /// 主入口。模型加载时调用一次。
    pub fn arbitrate(
        mode: InferenceMode,
        archetype: &GraphArchetype,
        profile: &DeviceProfile,
    ) -> StrategyBias {
        let mut bias = Self::mode_baseline(mode);
        Self::apply_archetype_modulation(&mut bias, archetype);
        Self::apply_hardware_adjustment(&mut bias, profile);
        bias.validate();  // 确保所有系数 > 0.0
        bias
    }
}
```

## 5. HwOptEngine 集成 — Solver Bias-Aware 完整算法

> **SSOT 声明**: 本节定义每个 Solver 的 bias-aware 完整算法。当本节与 `02-HARDWARE.md §10` 的原始 Solver 定义冲突时，以本节为准。本节是原始定义的**超集**——在原始算法的每个决策点插入 StrategyBias 调制，但不改变算法结构。

### 5.1 扩展后的 solve() 签名

```rust
impl HwOptEngine {
    pub fn solve(
        profile: &DeviceProfile,
        sensors: &MemoryNetworkSensors,
        probe: &ProbeResult,
        model: &ModelConfig,
        bias: &StrategyBias,    // 新增: Arbiter 输出
    ) -> Result<HwOptPlan, OptError> {
        // Level 0
        let effective_m = estimate_batch_size(bias.batch_flexibility, profile, model);
        let roofline = RooflineAnalyzer::analyze(profile, model, effective_m);
        let cache_plan = CacheBudgetSolver::solve(profile, sensors, model, bias);

        // Level 1
        let gemm_plan = GemmSolver::solve(profile, &roofline, &cache_plan, bias);
        let attn_plan = AttentionSolver::solve(profile, &cache_plan, &roofline);
        // AttentionSolver 不受 bias 影响（纯硬件决策）

        // Level 2
        let fusion_plan = FusionSolver::solve(&gemm_plan, &cache_plan, &roofline, bias);
        let parallel_plan = ParallelismSolver::solve(profile, &gemm_plan, &cache_plan, bias);

        // Level 3
        let batch_plan = BatchSolver::solve(&parallel_plan, &cache_plan, model, bias);

        // Level 4
        let feature_plan = FeatureRouter::route(
            profile, &gemm_plan, &fusion_plan, &attn_plan,
            &parallel_plan, &batch_plan, bias,
        );

        // Level 5: 循环依赖校正 (仅 Throughput 模式)
        let plan = Self::maybe_correct(roofline, cache_plan, gemm_plan, ...);

        Ok(plan)
    }
}
```

### 5.2 RooflineAnalyzer (Level 0) — bias-aware

**变更**: 输入增加 `effective_m: usize`。

```rust
struct RooflineInput {
    profile: &DeviceProfile,
    model: &ModelConfig,
    effective_m: usize,     // 新增: 来自 estimate_batch_size()
}
```

**算法变更点**: 算术强度计算中所有 M 替换为 `effective_m`。

```
AI(GEMM) = 2 × effective_m × N × K / ((effective_m × K + K × N + effective_m × N) × elem_bytes)

// Latency: effective_m = 1 → AI ≈ 2 → MemoryBound
// Throughput: effective_m = 64 → AI ≈ 128 → ComputeBound
```

**输出不变**: `RooflineResult` 结构体不变。

### 5.3 CacheBudgetSolver (Level 0) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
struct CacheBudgetInput {
    profile: &DeviceProfile,
    sensors: &MemoryNetworkSensors,
    model_bytes: usize,
    kv_bytes_per_token: usize,
    hidden_bytes: usize,
    bias: &StrategyBias,    // 新增
}
```

**完整 bias-aware 算法** (替代 `02-HARDWARE.md §10.5` 的 CPU 缓存预算算法):

```
// ── L1 分配 (不受 bias 影响) ──
l1_tile_budget    = L1 × 0.75    // GEMM tile
l1_fusion_scratch = L1 × 0.25    // TileLevelFusion scratch

// ── L2 分配 (bias 调制) ──
raw_kv     = 0.40 × bias.kv_cache_budget_scale
raw_weight = 0.35 × bias.weight_prefetch_budget_scale
raw_act    = 0.25  // activation 不受 bias 直接调制

total = raw_kv + raw_weight + raw_act

// 归一化: 总和 > 1.0 时按比例缩放; ≤ 1.0 时 activation 吃剩余
if total > 1.0:
    kv_ratio     = raw_kv / total
    weight_ratio = raw_weight / total
    act_ratio    = raw_act / total
else:
    kv_ratio     = raw_kv
    weight_ratio = raw_weight
    act_ratio    = 1.0 - raw_kv - raw_weight

l2_total = max(sensors.l2_cache_bytes, profile.cache_sizes.1)
kv_budget     = l2_total × kv_ratio
weight_budget = l2_total × weight_ratio
activation    = l2_total × act_ratio

// ── L3 分配 (不受 bias 影响) ──
// 原算法不变

// ── GPU HBM 分配 (不受 bias 影响) ──
// 原算法不变

// ── 动态调节: RooflineAnalyzer 结果叠加 ──
// 原逻辑: MemoryBound 时 kv_budget 提升到 50%
// 本调节在 bias 调制之后执行, 与 bias 效果叠加
```

**具体数值推演**:

```
Latency mode (bias: kv=0.5, weight=1.5):
  raw_kv     = 0.40 × 0.5  = 0.20
  raw_weight = 0.35 × 1.5  = 0.525
  raw_act    = 0.25
  total = 0.975 (< 1.0)
  → kv = 20%, weight = 52.5%, act = 1.0 - 0.20 - 0.525 = 27.5%

Throughput mode (bias: kv=1.5, weight=0.8):
  raw_kv     = 0.40 × 1.5  = 0.60
  raw_weight = 0.35 × 0.8  = 0.28
  raw_act    = 0.25
  total = 1.13 (> 1.0, 需归一化)
  → kv = 0.60/1.13 = 53.1%, weight = 0.28/1.13 = 24.8%, act = 0.25/1.13 = 22.1%
```

### 5.4 GemmSolver (Level 1) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
struct GemmSolverInput {
    profile: &DeviceProfile,
    constraints: &CompilerConstraints,
    roofline: &RooflineClass,
    cache_budget: &CacheBudgetPlan,
    elem_bytes: usize,
    bias: &StrategyBias,    // 新增
}
```

**完整 bias-aware 算法**:

```
// ── 步骤 1: 候选枚举 (原算法不变) ──
for mr in [4, 6, 8, 10, 12, 14, 16]:
    for nr in [8, 12, 16, 24, 32]:
        nr_vecs = ceil(nr / simd_width_elems)
        acc = mr × nr_vecs
        if acc + 2 + 2 ≤ num_simd_regs:
            candidates.push(GemmCandidate { mr, nr, ... })

// ── 步骤 2: k_depth 候选集合 (⚠️ 受 bias.k_depth_preference 影响) ──
k_depth_candidates =
    if bias.k_depth_preference >= 1.5:
        [1, 2, 4]              // 强偏好 → 扩展到 4（如果寄存器允许）
    elif bias.k_depth_preference >= 0.8:
        [1, 2]                 // 默认范围
    else:
        [1]                    // 弱偏好 → 固定为 1

// ── 步骤 3: 成本评估 (⚠️ T_overhead 受 bias.pipeline_cost_scale 调制) ──
for candidate in candidates:
    for k_depth in k_depth_candidates:
        // 检查寄存器硬约束: k_depth > 1 需要额外 double-buffer 寄存器
        if k_depth > 1:
            extra_regs = k_depth  // 每级流水线需要 1 个额外 buffer 寄存器
            if candidate.acc_regs + 2 + 2 + extra_regs > num_simd_regs:
                continue  // 寄存器不够，跳过此 k_depth

        // 成本公式 (bias-aware 版本)
        T_compute = (2 × mr × nr × KC) / (peak_fma × simd_efficiency)
        T_memory  = ((mr × KC + KC × nr) × elem_bytes) / peak_bw × (1 - l1_hit)
        T_overhead = k_depth × prefetch_latency × bias.pipeline_cost_scale  // ← bias 注入点
        cost = max(T_compute, T_memory) + T_overhead

        candidates_scored.push((candidate, k_depth, cost))

// ── 步骤 4: 选择最优候选 ──
best = candidates_scored.min_by(|a, b| a.cost.partial_cmp(&b.cost))

// ── 步骤 5: max_epilogue_depth 计算 (⚠️ 受 bias.epilogue_depth_preference 调制) ──
scratch_regs = num_simd_regs - best.acc_regs - 2 - 2 - best.k_depth
base_max_epilogue = (scratch_regs as f64 / 1.5).floor() as usize

// epilogue_depth_preference > 1.0 时：允许更深 epilogue，可能导致 NR 缩减
// epilogue_depth_preference < 1.0 时：限制 epilogue 深度，把寄存器留给其他用途
effective_max_epilogue = ((base_max_epilogue as f64) × bias.epilogue_depth_preference)
    .round() as usize
effective_max_epilogue = effective_max_epilogue.max(1).min(scratch_regs)
// 硬约束: 最小 1 (至少一层 epilogue)，最大 = scratch_regs (物理上限)

// ── 步骤 6: NR 缩减决策 (如果 epilogue 需求超出 scratch) ──
// 当 effective_max_epilogue > base_max_epilogue 且 scratch 不够时:
//   缩减 NR → 释放累加器寄存器 → 转为 scratch
// 原算法逻辑不变，只是 max_epilogue_depth 的值被 bias 调制了

// ── 步骤 7: 策略路由 (原算法不变) ──
strategy = select_gemm_strategy_cost_based(profile, best, ...)
```

**输出**: `GemmPlan` 结构体不变，但以下字段的数值被 bias 影响:

| 字段 | 受影响的 bias | 效果 |
|------|-------------|------|
| `k_depth` | `k_depth_preference` | pref↑ → 更深 k_depth |
| `pf_distance` | 间接 (通过 k_depth) | k_depth↑ → pf_distance↑ |
| `max_epilogue_depth` | `epilogue_depth_preference` | pref↑ → 允许更深 epilogue |
| `mr`, `nr` | `pipeline_cost_scale` | scale↓ → pipeline 便宜 → 可能选更小 tile 腾出寄存器给 pipeline |
| `scratch_regs` | 间接 (通过 k_depth + epilogue) | |

### 5.5 FusionSolver (Level 2) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
struct FusionSolverInput {
    gemm_plan: &GemmPlan,
    cache_plan: &CacheBudgetPlan,
    roofline: &RooflineResult,
    num_simd_regs: usize,
    l1_cache_bytes: usize,
    bias: &StrategyBias,    // 新增
}
```

**完整 bias-aware 算法**:

```
// ── 步骤 1: max_epilogue_depth (已被 GemmSolver 调制，直接读取) ──
max_epilogue = gemm_plan.max_epilogue_depth  // 已含 bias 效果

// ── 步骤 2: EpilogueInjection 深度决策 (原算法不变) ──
for depth in (1..=max_epilogue).rev():
    required = sum(TraceOp::register_cost(&epilogue_ops[..depth]))
    if required ≤ gemm_plan.scratch_regs:
        selected_depth = depth
        break

// ── 步骤 3: TileLevelFusion vs ComputeRoot (⚠️ 受 bias.fusion_cost_scale 调制) ──
// 原算法: threshold = cache_plan.l1_tile_budget × 0.75
// bias-aware: threshold 乘以 fusion_cost_scale
effective_tile_threshold = (cache_plan.l1_tile_budget as f64 × 0.75 × bias.fusion_cost_scale) as usize

predecessor_output_bytes = hidden_bytes × tile_rows

if predecessor_output_bytes > effective_tile_threshold:
    TileLevelFusion
    // fusion_cost_scale < 1.0 → threshold 降低 → 更多情况触发 TileLevelFusion
    // 效果: 偏好融合时，更小的输出也会嵌入 MC 循环而非 ComputeRoot
else:
    ComputeRoot

// ── 步骤 4: FFNBlock 融合路径选择 (原算法不变) ──
// scratch 寄存器够 → GateSiLUInject，不够 → SeparateGemm
// 注意: scratch_regs 已被 epilogue_depth_preference 间接影响
if gemm_plan.scratch_regs >= 1:
    ffn_strategy = GateSiLUInject
else:
    ffn_strategy = SeparateGemm

// ── 步骤 5: 融合收益评估 (⚠️ 受 bias.fusion_cost_scale 调制) ──
// 对每个潜在融合点, 计算调制后的净收益
for (producer, consumer) in fusion_candidates:
    raw_savings = fusion_savings(mode, ops)  // 原公式: 中间 tensor 写回字节数
    adjusted_savings = raw_savings / bias.fusion_cost_scale
    // fusion_cost_scale < 1.0 → savings 被 1/0.5 = 2× 放大

    raw_cost = fusion_overhead(mode, ops)    // 寄存器占用 + L1 压力
    adjusted_cost = raw_cost                 // cost 侧不调制 (只调制 savings 侧)

    if adjusted_savings > adjusted_cost:
        apply fusion
    else:
        keep standalone

// ── 步骤 6: CrossLayerResidual 和 QkvSharedInput (原算法不变) ──
cross_layer_residual_enabled = gemm_plan.scratch_regs >= 4
qkv_shared_input_enabled = true  // 始终启用（零额外寄存器开销）
norm_into_gemm_enabled = true    // 始终启用
```

**输出**: `FusionPlan` 结构体不变，但以下字段的数值被 bias 影响:

| 字段 | 受影响的 bias | 效果 |
|------|-------------|------|
| `max_epilogue_depth` | `epilogue_depth_preference` (通过 GemmPlan) | pref↑ → 更深 epilogue |
| `tile_fusion_threshold` | `fusion_cost_scale` | scale↓ → 阈值降低 → 更多 TileFusion |
| `ffn_strategy` | `epilogue_depth_preference` (通过 scratch_regs) | pref↑ → scratch 可能更少 → 可能 SeparateGemm |
| `fusions[].strategy` | `fusion_cost_scale` | scale↓ → 净收益放大 → 更多融合被接受 |

### 5.6 ParallelismSolver (Level 2) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
struct ParallelismSolverInput {
    profile: &DeviceProfile,
    gemm_plan: &GemmPlan,
    cache_plan: &CacheBudgetPlan,
    bias: &StrategyBias,    // 新增
}
```

**完整 bias-aware GPU 算法**:

```
sm_total = profile.compute_units
wave_count_candidates = [1, 2, 4]
best_wave = 1
best_cost = f64::MAX

for wc in wave_count_candidates:
    if wc > sm_total / 16:  // min 16 SM per wave
        break

    sm_per = sm_total / wc
    min_tokens = sm_per × warp_size × occupancy_target

    // 同步成本: wave 越多，barrier 越贵
    // 基础模型: 每个 wave 边界一次 barrier, 延迟 = wc × 2μs
    sync_cost = (wc as f64) × 2.0  // 微秒

    // ⚠️ bias 注入点: parallelism_cost_scale 调制同步成本
    adjusted_sync_cost = sync_cost × bias.parallelism_cost_scale
    // parallelism_cost_scale < 1.0 → 同步看起来更便宜 → 更倾向多 wave

    // 并行收益: wave 越多，SM 利用率越高（假设 batch 足够大）
    parallel_benefit = (wc as f64).ln() × sm_total as f64 × 0.1  // 对数收益递减

    wave_score = parallel_benefit - adjusted_sync_cost

    if wave_score > -best_cost:  // 注意: 最小化负 score = 最大化 score
        best_wave = wc
        best_cost = -wave_score

// Latency mode: parallelism_cost_scale = 1.5
//   → sync_cost × 1.5 → 同步更贵 → 倾向 wave=1
// Throughput + MoE: parallelism_cost_scale = 0.25
//   → sync_cost × 0.25 → 同步几乎免费 → 倾向 wave=4
```

**CPU NUMA**: 原算法不变。CPU 的 wave_count 直接等于 NUMA 节点数，不受 bias 影响（硬件物理拓扑决定）。

**输出**: `ParallelPlan` 结构体不变。

| 字段 | 受影响的 bias | 效果 |
|------|-------------|------|
| `wave_count` | `parallelism_cost_scale` | scale↓ → 更多 wave |
| `min_batch_tokens_per_wave` | 间接 (通过 wave_count) | wave↑ → 每 wave 需要更少 tokens |

### 5.7 BatchSolver (Level 3) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
struct BatchSolverInput {
    parallel_plan: &ParallelPlan,
    cache_plan: &CacheBudgetPlan,
    model: &ModelConfig,
    bias: &StrategyBias,    // 新增
}
```

**完整 bias-aware 算法**:

```
// ── 步骤 1: batch 大小上限 (⚠️ 受 bias.batch_flexibility 硬约束) ──
// batch_flexibility = 0.0 是唯一的硬约束切入点（Latency mode）
if bias.batch_flexibility == 0.0:
    max_batch_tokens = 1                // 严格 batch=1
    decode_slots = 1
    max_chunks_per_batch = 1
    → 直接返回 BatchPlan，跳过后续所有逻辑
else:
    // base_max 来自 KV cache 内存预算
    base_max = cache_plan.hbm_max_pages.unwrap_or(cache_plan.l3_kv_cold_budget / kv_per_page)
    max_batch_tokens = (base_max as f64 × bias.batch_flexibility).round() as usize
    max_batch_tokens = max_batch_tokens.max(1)

// ── 步骤 2: decode/prefill 混合比 (⚠️ 受 bias.decode_ratio_scale 调制) ──
base_ratio_cap = 0.6  // 原算法默认值
effective_ratio_cap = (base_ratio_cap × bias.decode_ratio_scale).min(1.0)
// decode_ratio_scale > 1.0 → decode 可占更大比例 → prefill 进展变慢

// ── 步骤 3: Chunk 大小和黄金尺寸 (原算法不变) ──
// adaptive_chunk_size() 和同号合并策略不受 bias 影响
// 这些决策由 ProbeResult 的黄金尺寸驱动

// ── 步骤 4: decode_slots (受 max_batch_tokens 间接影响) ──
decode_budget = min(max_batch_tokens, floor(max_batch_tokens × effective_ratio_cap))
decode_slots = decode_budget  // 每个 decode 序列 = 1 token
```

**输出**: `BatchPlan` 结构体不变。

| 字段 | 受影响的 bias | 效果 |
|------|-------------|------|
| `decode_ratio_cap` | `decode_ratio_scale` | scale↑ → decode 占比上限更高 |
| `decode_slots` | `batch_flexibility` + `decode_ratio_scale` | flex=0 → 1, flex=1 → 自由 |
| `max_chunk_size` | 不受影响 | 由 ProbeResult 驱动 |

### 5.8 FeatureRouter (Level 4) — bias-aware

**变更**: 输入增加 `bias: &StrategyBias`。

```rust
fn route(
    profile: &DeviceProfile,
    gemm_plan: &GemmPlan,
    fusion_plan: &FusionPlan,
    attn_plan: &AttentionPlan,
    parallel_plan: &ParallelPlan,
    batch_plan: &BatchPlan,
    bias: &StrategyBias,    // 新增
) -> FeaturePlan
```

**bias 影响的特性决策**:

```
// ── Speculative Decoding 启用决策 ──
// 原算法: benefit = estimate_spec_benefit(model, attn_plan)
//         cost = estimate_spec_cost(batch_plan)  // draft model 抢占 batch 容量
//         enabled = benefit > cost

// bias-aware 版本:
raw_benefit = estimate_spec_benefit(model, attn_plan)
adjusted_benefit = raw_benefit × bias.speculative_decoding_value
// speculative_decoding_value = 1.5 (Latency) → 收益放大 1.5× → 更倾向启用
// speculative_decoding_value = 0.3 (Throughput) → 收益缩小 → 不太启用

raw_cost = estimate_spec_cost(batch_plan)
// cost 不调制

spec_decoding_enabled = adjusted_benefit > raw_cost

// ── 量化策略决策 ──
// 原算法: 在候选量化方案中选 cost 最优
// bias-aware: 量化的收益乘以 quantization_aggressiveness

for quant_level in [INT8, INT4, FP4]:
    raw_quant_benefit = bandwidth_savings(quant_level)
    adjusted_quant_benefit = raw_quant_benefit × bias.quantization_aggressiveness
    quant_cost = precision_loss(quant_level)  // 不调制
    quant_candidates.push((quant_level, adjusted_quant_benefit - quant_cost))

selected_quant = quant_candidates.max_by(|a, b| a.1.partial_cmp(&b.1))

// ── 其他特性 (不受 bias 影响) ──
// Avx512Fma, NeonBf16, Wgmma, Tma2D 等硬件特性
// 纯硬件决策，不受 StrategyBias 影响
```

### 5.9 MoE 模块消费 — bias-aware

**ExpertThermalManager** 消费 `expert_eviction_aggressiveness`:

```rust
impl ExpertThermalManager {
    fn effective_eviction_threshold(&self) -> u64 {
        if self.adaptive_eviction {
            let adaptive = self.working_set.adaptive_threshold(self.memory_pressure);
            // ⚠️ bias 调制: aggressiveness 高 → 阈值低 → 更容易驱逐
            let bias_factor = 1.0 / (1.0 + self.eviction_aggressiveness);
            // aggressiveness = 0.0 → factor = 1.0 → 阈值不变（全部常驻）
            // aggressiveness = 1.0 → factor = 0.5 → 阈值减半（更激进驱逐）
            // aggressiveness = 2.0 → factor = 0.33 → 阈值缩到 1/3
            (adaptive as f64 × bias_factor) as u64
        } else {
            self.eviction_streak_threshold
        }
    }
}
```

**ExpertWeightPrefetcher** 消费 `expert_prefetch_priority`:

```rust
impl ExpertWeightPrefetcher {
    fn proactive_warmup(&mut self, candidates: &[(usize, f64)]) {
        // 原算法: 按 revival_probability 排序，预取 top-N
        // bias-aware: 预取数量乘以 expert_prefetch_priority
        let base_prefetch_count = 2;  // 默认预取 2 个候选
        let effective_count = ((base_prefetch_count as f64) × self.prefetch_priority)
            .round() as usize;
        // prefetch_priority = 0.5 → 预取 1 个
        // prefetch_priority = 2.85 → 预取 5~6 个

        for (expert_idx, _prob) in candidates.iter().take(effective_count) {
            self.initiate_async_prefetch(*expert_idx);
        }
    }
}
```

**如何传递 bias 到 MoE 模块**:

```rust
// 在 Executor 初始化时:
let moe_thermal = ExpertThermalManager::new(config)
    .with_adaptive_eviction(window_size)
    .with_eviction_aggressiveness(hw_opt_plan.strategy_bias.expert_eviction_aggressiveness);

let moe_prefetcher = ExpertWeightPrefetcher::new(config)
    .with_prefetch_priority(hw_opt_plan.strategy_bias.expert_prefetch_priority);
```

### 5.10 HwOptPlan 扩展

```rust
pub struct HwOptPlan {
    // 原有字段 (§10.14)
    pub roofline: RooflineResult,
    pub gemm: GemmPlan,
    pub cache: CacheBudgetPlan,
    pub fusion: FusionPlan,
    pub attention: AttentionPlan,
    pub parallel: ParallelPlan,
    pub batch: BatchPlan,
    pub features: FeaturePlan,

    // 新增: 溯源信息
    pub inference_mode: InferenceMode,
    pub graph_archetype: GraphArchetype,
    pub strategy_bias: StrategyBias,
}
```

**REQ-ARB-003**: `inference_mode`、`graph_archetype`、`strategy_bias` 嵌入 HwOptPlan 以便下游模块溯源决策依据。这三个字段均为只读。

## 6. 决策链完整流程

```
Client::builder()
    .model("DeepSeek/DeepSeek-V3")
    .inference_mode(InferenceMode::Throughput)      // ① 用户指定模式
    .build()?

  → Loader 加载模型文件
  → ModelConfig 从元数据提取标量参数                   // ② 标量配置
  → ArchitectureResolver 解析 OnnxGraph               // ③ 模型图构建
  → GraphProfiler::profile(graph, config)             // ④ 拓扑特征提取
      → GraphProfile { num_experts: 256, moe_top_k: 6, ... }
  → GraphArchetype::derive(profile)                    // ⑤ 原型推导
      → { compute: 0.50, memory: 0.90, parallel: 0.99, fusion: 0.60, pipeline: 0.35 }
  → StrategyArbiter::arbitrate(Throughput, archetype, device)  // ⑥ 仲裁
      → StrategyBias {
            fusion_cost_scale: 0.64,       // Throughput 中性 × MoE 融合较低
            pipeline_cost_scale: 1.17,     // Throughput 轻惩罚
            parallelism_cost_scale: 0.25,  // Throughput 强偏好 × MoE 极高并行
            expert_eviction: 1.2,          // 适度驱逐
            expert_prefetch: 2.7,          // 极激进预取
            batch_flexibility: 1.0,        // 自由
            ...
        }
  → HwOptEngine::solve(profile, sensors, probe, config, bias)  // ⑦ 参数求解
      → GemmSolver: 被 parallelism bias 影响 → 选择较小 tile 让出 SM 给更多 wave
      → ParallelismSolver: parallelism_cost_scale=0.25 → 选择 4 wave
      → FusionSolver: fusion_cost_scale=0.64 → 中等融合深度
      → BatchSolver: batch_flexibility=1.0 → 大 batch 自由决策
  → HwOptPlan { inference_mode: Throughput, ... }      // ⑧ 最终计划
  → JIT Codegen 生成机器码                              // ⑨ 编译
```

## 7. 约束与铁律

### REQ-ARB-004: InferenceMode 不可变

`InferenceMode` 在 `Client::builder().build()` 时确定，推理期间不可切换。切换模式需 `swap_model()` 重新加载，因为 JIT 编译产物因模式不同而不同。

### REQ-ARB-005: Arbiter 不直接决策

Arbiter 只输出 `StrategyBias`（成本调制系数），不直接选择具体策略（如"使用 FA3"或"k_depth=2"）。具体决策始终由 HwOptEngine 的各 Solver 通过 Cost-Model 做出。Arbiter 通过调制成本函数间接影响决策方向。

### REQ-ARB-006: 所有系数正数

`StrategyBias` 的所有字段必须 > 0.0。0.0 会导致 CostModel 除零或项消失。`batch_flexibility = 0.0` 是唯一例外，表示"硬约束 batch=1"。

### REQ-ARB-007: GraphProfile 一次性提取

`GraphProfiler::profile()` 在模型加载时调用一次，结果嵌入 `HwOptPlan`。推理期间不重新提取。图结构在推理期间不变。

### REQ-ARB-008: 禁止绕过 Arbiter

下游模块禁止根据 `InferenceMode` 做直接 if/else 分支决策。所有模式差异必须通过 `StrategyBias` 传导。

```rust
// ❌ 禁止
if hw_opt_plan.inference_mode == Latency {
    max_batch = 1;
}

// ✅ 正确
max_batch = (base_max as f64 × hw_opt_plan.strategy_bias.batch_flexibility) as usize;
```

**理由**: 直接 if/else 会绕过 CostModel，导致策略决策碎片化。StrategyBias 是策略影响的唯一通道。

### REQ-ARB-009: 与 JIT 缓存协议一致

StrategyBias 影响 JIT 编译产物，因此 `ModelJitCache` 的缓存 key 必须包含 `InferenceMode`：

```
cache_key = device_fingerprint + model_id + inference_mode
```

同一模型在 Latency 和 Throughput 模式下的 JIT 缓存互不复用。

## 8. 实现位置

| 组件 | 文件 | 职责 |
|------|------|------|
| `InferenceMode` | `src/client.rs` | 枚举定义 + Builder 集成 |
| `GraphProfile` | `src/graph/profile.rs` (新增) | 拓扑特征提取 |
| `GraphArchetype` | `src/graph/profile.rs` | 原型推导 |
| `StrategyArbiter` | `src/engine/arbiter.rs` (新增) | 仲裁算法 |
| `StrategyBias` | `src/engine/arbiter.rs` | 偏置结构 |
| `HwOptEngine::solve()` | gllm-kernels `src/compiler/planner.rs` | 扩展签名 + bias 注入 |
| `CostModel` | gllm-kernels `src/compiler/fusion/cost_model.rs` | bias 调制点 |

## 9. 下游消费映射表

| 消费方 | 消费的 StrategyBias 字段 | 影响 |
|--------|------------------------|------|
| RooflineAnalyzer | `InferenceMode` → effective_M | 算术强度分类翻转 |
| GemmSolver | `pipeline_cost_scale`, `k_depth_preference`, `epilogue_depth_preference` | GEMM 微内核参数 |
| CacheBudgetSolver | `kv_cache_budget_scale`, `weight_prefetch_budget_scale` | 缓存预算分配 |
| FusionSolver | `fusion_cost_scale`, `epilogue_depth_preference` | 融合深度和模式选择 |
| ParallelismSolver | `parallelism_cost_scale` | wave 数量 |
| BatchSolver | `batch_flexibility`, `decode_ratio_scale` | batch 大小和混合比 |
| FeatureRouter | `speculative_decoding_value`, `quantization_aggressiveness` | 特性启用决策 |
| ExpertThermalManager | `expert_eviction_aggressiveness` | 驱逐阈值 |
| ExpertWeightPrefetcher | `expert_prefetch_priority` | 预取优先级 |

## 10. 数学基础与辅助函数

本节定义 SPEC 中所有公式使用的辅助函数。实现必须严格使用这些定义，禁止替换为近似形式。

### 10.1 sigmoid (标准逻辑函数)

```rust
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

值域: (0.0, 1.0)。当 x=0 → 0.5，x=5 → 0.993，x=-5 → 0.007。

**注意**: §3.5 中部分 sigmoid 输入可能为负值（例如 `(2048 - hidden_dim) / 2048` 当 hidden_dim=4096 时输入为 -1.0 → 输出 0.27），这是预期行为。

### 10.2 lerp (线性插值)

```rust
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
```

当 t=0.0 → a，t=1.0 → b，t=0.5 → (a+b)/2。t 不要求在 [0,1] 内，但本 SPEC 中所有 t 均来自 GraphArchetype 维度（已 clamp 到 [0,1]）。

### 10.3 clamp

```rust
fn clamp(x: f64, min: f64, max: f64) -> f64 {
    x.max(min).min(max)
}
```

## 11. StrategyBias 字段值域与验证

### 11.1 字段合法范围

每个 StrategyBias 字段有明确的合法范围。超出范围的值在 `validate()` 中被 clamp（不是 panic/error），因为 Arbiter 是数值计算，浮点累积偏差是正常现象。

| 字段 | 最小值 | 最大值 | 含义 |
|------|--------|--------|------|
| `fusion_cost_scale` | 0.2 | 3.0 | 0.2 = 融合代价被低估 5×（极度偏好融合） |
| `pipeline_cost_scale` | 0.2 | 3.0 | 同上 |
| `parallelism_cost_scale` | 0.1 | 3.0 | 0.1 = MoE 极端场景可达 |
| `epilogue_depth_preference` | 0.3 | 3.0 | 0.3 = 极度抑制 epilogue |
| `k_depth_preference` | 0.3 | 3.0 | 0.3 = 极度抑制 pipeline |
| `kv_cache_budget_scale` | 0.2 | 3.0 | L2 预算倍率 |
| `weight_prefetch_budget_scale` | 0.2 | 3.0 | L2 预算倍率 |
| `batch_flexibility` | 0.0 | 1.0 | 0.0 = batch=1 硬约束 |
| `decode_ratio_scale` | 0.3 | 2.0 | decode/prefill 混合比倍率 |
| `expert_eviction_aggressiveness` | 0.0 | 2.0 | 0.0 = 全部常驻 |
| `expert_prefetch_priority` | 0.1 | 5.0 | 5.0 = 极激进预取 |
| `speculative_decoding_value` | 0.1 | 3.0 | 收益倍率 |
| `quantization_aggressiveness` | 0.3 | 3.0 | 量化倾向倍率 |

### 11.2 validate() 实现

```rust
impl StrategyBias {
    /// clamp 所有字段到合法范围。不 panic，不返回 error。
    pub fn validate(&mut self) {
        self.fusion_cost_scale = self.fusion_cost_scale.clamp(0.2, 3.0);
        self.pipeline_cost_scale = self.pipeline_cost_scale.clamp(0.2, 3.0);
        self.parallelism_cost_scale = self.parallelism_cost_scale.clamp(0.1, 3.0);
        self.epilogue_depth_preference = self.epilogue_depth_preference.clamp(0.3, 3.0);
        self.k_depth_preference = self.k_depth_preference.clamp(0.3, 3.0);
        self.kv_cache_budget_scale = self.kv_cache_budget_scale.clamp(0.2, 3.0);
        self.weight_prefetch_budget_scale = self.weight_prefetch_budget_scale.clamp(0.2, 3.0);
        self.batch_flexibility = self.batch_flexibility.clamp(0.0, 1.0);
        self.decode_ratio_scale = self.decode_ratio_scale.clamp(0.3, 2.0);
        self.expert_eviction_aggressiveness = self.expert_eviction_aggressiveness.clamp(0.0, 2.0);
        self.expert_prefetch_priority = self.expert_prefetch_priority.clamp(0.1, 5.0);
        self.speculative_decoding_value = self.speculative_decoding_value.clamp(0.1, 3.0);
        self.quantization_aggressiveness = self.quantization_aggressiveness.clamp(0.3, 3.0);
    }
}
```

**REQ-ARB-010**: `validate()` 使用 clamp 而非 panic。Arbiter 是数值管线，边界值是合法的极端策略，不是错误。

## 12. Throughput 模式 effective_M 循环依赖解决

### 12.1 问题

RooflineAnalyzer（Level 0）需要 `effective_M` 计算算术强度，但精确的 batch size 由 BatchSolver（Level 3）决定。这形成循环依赖。

### 12.2 解决方案: 两阶段求解

```
阶段 1: 使用估算 M 完成初始求解
  estimated_M = estimate_batch_size(mode, profile, model)
  roofline = RooflineAnalyzer::analyze(profile, model, estimated_M)
  → 完成 Level 0~4 全部求解 → 初始 HwOptPlan

阶段 2: 用初始 BatchPlan 的实际 batch size 校正（仅当偏差显著时）
  actual_M = initial_plan.batch.decode_slots
  if abs(actual_M - estimated_M) / estimated_M > 0.5:  // 偏差超过 50%
      重新用 actual_M 跑 RooflineAnalyzer → 重新求解（最多 1 轮校正）
```

### 12.3 estimate_batch_size() 估算函数

```rust
fn estimate_batch_size(mode: InferenceMode, profile: &DeviceProfile, model: &ModelConfig) -> usize {
    match mode {
        InferenceMode::Latency => 1,  // 永远是 1，无需估算
        InferenceMode::Throughput => {
            // 目标: GPU 内存的 60% 给 KV cache + 权重，剩余给 batch
            // batch_M ≈ remaining_memory / (kv_per_token × max_seq_len)
            // 保守估算: min(64, gpu_memory_gb × 4)
            // CPU: min(32, num_cores × 2)
            if profile.is_gpu() {
                let gpu_mem_gb = profile.total_memory_bytes() / (1 << 30);
                (gpu_mem_gb * 4).min(256).max(8) as usize
            } else {
                let cores = profile.num_cores();
                (cores * 2).min(64).max(4)
            }
        }
    }
}
```

**REQ-ARB-011**: Throughput 模式使用两阶段求解打破循环依赖。Latency 模式 effective_M=1，无循环。校正最多执行 1 轮，禁止迭代收敛。

## 13. CacheBudget 归一化公式

完整算法见 §5.3 CacheBudgetSolver bias-aware 算法。此处保留 REQ 索引。

**REQ-ARB-012**: activation 预算不受 bias 直接调制。activation 始终 = L2 × (1 - kv_ratio - weight_ratio)，作为 kv 和 weight 调制的被动结果。完整公式和数值推演见 §5.3。

## 14. CostModel 注入方向规则

**核心原则**: StrategyBias 的 `*_cost_scale` 系列字段遵循统一语义——**乘到代价侧，除到收益侧**。

```
代价侧 (越低越好):
  adjusted_cost = base_cost × bias.*_cost_scale
  // cost_scale < 1.0 → 代价降低 → Solver 更倾向选择该策略

收益侧 (越高越好):
  adjusted_benefit = base_benefit × (1.0 / bias.*_cost_scale)
  // cost_scale < 1.0 → 1/scale > 1.0 → 收益放大 → Solver 更倾向选择该策略
```

**方向一致性**: 对于同一个 bias 字段，代价侧和收益侧的调制方向必须一致。`fusion_cost_scale = 0.5` 意味着"融合的代价减半"和"融合的收益翻倍"，二者效果相同。

### 14.1 各 Solver 注入方向表

| Solver | 注入点 | bias 字段 | 注入方式 | 效果 |
|--------|-------|----------|---------|------|
| GemmSolver | T_overhead (pipeline setup) | `pipeline_cost_scale` | `T_overhead × scale` | scale↓ → pipeline 便宜 → 偏好深 k_depth |
| GemmSolver | k_depth 候选范围 | `k_depth_preference` | pref>1.0 → 扩展候选 | pref↑ → 更多 k_depth 选项 |
| GemmSolver | max_epilogue_depth | `epilogue_depth_preference` | `base × pref` | pref↑ → 允许更深 epilogue |
| FusionSolver | fusion_savings | `fusion_cost_scale` | `savings × (1/scale)` | scale↓ → 收益放大 → 偏好融合 |
| FusionSolver | TileFusion 阈值 | `fusion_cost_scale` | `threshold × scale` | scale↓ → 阈值降低 → 更多 TileFusion |
| CacheBudgetSolver | L2 kv 比例 | `kv_cache_budget_scale` | `0.40 × scale` | scale↑ → KV 预算更大 |
| CacheBudgetSolver | L2 weight 比例 | `weight_prefetch_budget_scale` | `0.35 × scale` | scale↑ → 权重预取预算更大 |
| ParallelismSolver | sync_cost | `parallelism_cost_scale` | `sync_cost × scale` | scale↓ → 同步便宜 → 更多 wave |
| BatchSolver | max_batch_size | `batch_flexibility` | `base × flex` | flex=0 → batch=1 |
| BatchSolver | decode_ratio_cap | `decode_ratio_scale` | `cap × scale` | scale↑ → decode 占比上限更高 |
| FeatureRouter | spec_decoding benefit | `speculative_decoding_value` | `benefit × value` | value↑ → 更倾向启用 |
| FeatureRouter | quant benefit | `quantization_aggressiveness` | `benefit × aggr` | aggr↑ → 更倾向低比特 |

**REQ-ARB-013**: 此表是 CostModel 注入的权威映射。实现时必须按此表的注入方式和方向执行，禁止自行设计注入点。

## 15. 黄金测试向量

以下测试向量用于验证 Arbiter 实现的正确性。每个向量给出输入和期望的 StrategyBias 关键字段值（±10% 容差）。

### 15.1 测试辅助: 标准 DeviceProfile

```
CPU_AVX2:    { is_gpu: false, num_simd_regs: 16, cache_sizes: (32768, 262144, 8388608) }
CPU_AVX512:  { is_gpu: false, num_simd_regs: 32, cache_sizes: (49152, 1048576, 33554432) }
GPU_A100:    { is_gpu: true,  num_simd_regs: 255, total_memory: 80GB }
```

### 15.2 向量 1: Llama-70B + Latency + CPU_AVX2

```
GraphArchetype: { compute: 0.95, memory: 0.60, parallel: 0.20, fusion: 0.85, pipeline: 0.15 }

阶段 1 — mode_baseline(Latency):
  fusion_cost_scale = 0.5
  pipeline_cost_scale = 0.6
  parallelism_cost_scale = 1.5
  epilogue_depth_preference = 1.5
  k_depth_preference = 1.3
  batch_flexibility = 0.0

阶段 2 — archetype 调制:
  fusion_cost_scale = 0.5 × lerp(1.0, 0.6, 0.85) = 0.5 × 0.66 = 0.33 → clamp → 0.33
  pipeline_cost_scale = 0.6 × lerp(1.0, 0.6, 0.15) = 0.6 × 0.94 = 0.564
  parallelism_cost_scale = 1.5 × lerp(1.0, 0.5, 0.20) = 1.5 × 0.90 = 1.35
  reg_tension = 0.85 - 0.15 = 0.70 > 0:
    epilogue_depth_preference = 1.5 × (1.0 + 0.70 × 0.5) = 1.5 × 1.35 = 2.025
    k_depth_preference = 1.3 × (1.0 - 0.70 × 0.3) = 1.3 × 0.79 = 1.027

阶段 3 — 硬件微调 (CPU_AVX2, num_simd_regs=16):
  reg_scarcity = 1.0 - 16/32 = 0.5
  epilogue_depth_preference = 2.025 × (1.0 + 0.5 × 0.3) = 2.025 × 1.15 = 2.329
  k_depth_preference = 1.027 × (1.0 - 0.5 × 0.2) = 1.027 × 0.90 = 0.924
  l1_richness = (32768 / 65536).min(2.0) = 0.5
  fusion_cost_scale = 0.33 × 1.0/sqrt(0.5) = 0.33 × 1.414 = 0.467
  not GPU → pipeline_cost_scale 不变 = 0.564

期望结果 (±10%):
  fusion_cost_scale ≈ 0.47
  pipeline_cost_scale ≈ 0.56
  parallelism_cost_scale ≈ 1.35
  epilogue_depth_preference ≈ 2.33
  k_depth_preference ≈ 0.92
  batch_flexibility = 0.0 (精确)

解读: 极度偏好融合 (0.47)，深 epilogue (2.33)，batch=1 固定。
符合预期: Llama-70B + Latency → memory-bound GEMV，融合每省一字节 = 省一 cycle。
```

### 15.3 向量 2: DeepSeek-V3 + Throughput + GPU_A100

```
GraphArchetype: { compute: 0.50, memory: 0.90, parallel: 0.99, fusion: 0.60, pipeline: 0.35 }

阶段 1 — mode_baseline(Throughput):
  fusion_cost_scale = 1.0
  pipeline_cost_scale = 1.3
  parallelism_cost_scale = 0.5
  epilogue_depth_preference = 0.8
  k_depth_preference = 0.8
  batch_flexibility = 1.0
  expert_eviction_aggressiveness = 0.8
  expert_prefetch_priority = 1.5

阶段 2 — archetype 调制:
  fusion_cost_scale = 1.0 × lerp(1.0, 0.6, 0.60) = 1.0 × 0.76 = 0.76
  pipeline_cost_scale = 1.3 × lerp(1.0, 0.6, 0.35) = 1.3 × 0.86 = 1.118
  parallelism_cost_scale = 0.5 × lerp(1.0, 0.5, 0.99) = 0.5 × 0.505 = 0.253
  reg_tension = 0.60 - 0.35 = 0.25 > 0:
    epilogue_depth_preference = 0.8 × (1.0 + 0.25 × 0.5) = 0.8 × 1.125 = 0.90
    k_depth_preference = 0.8 × (1.0 - 0.25 × 0.3) = 0.8 × 0.925 = 0.74
  MoE (parallel > 0.5):
    expert_eviction = 0.8 × lerp(1.0, 1.5, 0.90) = 0.8 × 1.45 = 1.16
    expert_prefetch = 1.5 × lerp(1.0, 2.0, 0.90) = 1.5 × 1.90 = 2.85
  kv_cache_budget_scale = 1.5 × lerp(1.0, 1.5, 0.90) = 1.5 × 1.45 = 2.175
  quantization_aggressiveness = 0.8 × lerp(1.0, 1.3, 0.90) = 0.8 × 1.27 = 1.016

阶段 3 — 硬件微调 (GPU_A100):
  is_gpu → epilogue_depth_preference = 0.90 × 1.2 = 1.08
  is_gpu → k_depth_preference = 0.74 × 1.2 = 0.888
  is_gpu → pipeline_cost_scale = 1.118 × 1.2 = 1.342
  not CPU (num_simd_regs > 16) → 无寄存器稀缺调制
  l1_richness: GPU 的 cache_sizes.0 视为 shared_mem, 这里假设 49152
    l1_richness = (49152 / 65536).min(2.0) = 0.75
    fusion_cost_scale = 0.76 / sqrt(0.75) = 0.76 / 0.866 = 0.878

期望结果 (±10%):
  fusion_cost_scale ≈ 0.88
  pipeline_cost_scale ≈ 1.34
  parallelism_cost_scale ≈ 0.25
  epilogue_depth_preference ≈ 1.08
  k_depth_preference ≈ 0.89
  batch_flexibility = 1.0 (精确)
  expert_eviction ≈ 1.16
  expert_prefetch ≈ 2.85

解读: 极度偏好并行 (0.25)，极激进专家预取 (2.85)，pipeline 惩罚 (1.34)。
符合预期: MoE 256 专家 + Throughput → wave 最大化，专家权重异步预取。
```

### 15.4 向量 3: Phi-4 (3B) + Latency + CPU_AVX512

```
GraphArchetype: { compute: 0.35, memory: 0.25, parallel: 0.10, fusion: 0.70, pipeline: 0.80 }

阶段 1 — mode_baseline(Latency):
  fusion_cost_scale = 0.5
  pipeline_cost_scale = 0.6
  k_depth_preference = 1.3
  epilogue_depth_preference = 1.5

阶段 2 — archetype 调制:
  fusion_cost_scale = 0.5 × lerp(1.0, 0.6, 0.70) = 0.5 × 0.72 = 0.36
  pipeline_cost_scale = 0.6 × lerp(1.0, 0.6, 0.80) = 0.6 × 0.68 = 0.408
  parallelism_cost_scale = 1.5 × lerp(1.0, 0.5, 0.10) = 1.5 × 0.95 = 1.425
  reg_tension = 0.70 - 0.80 = -0.10 < 0:
    k_depth_preference = 1.3 × (1.0 + 0.10 × 0.5) = 1.3 × 1.05 = 1.365
    epilogue_depth_preference = 1.5 × (1.0 - 0.10 × 0.3) = 1.5 × 0.97 = 1.455

阶段 3 — 硬件微调 (CPU_AVX512, num_simd_regs=32):
  num_simd_regs > 16 → 无寄存器稀缺调制
  l1_richness = (49152 / 65536).min(2.0) = 0.75
  fusion_cost_scale = 0.36 / sqrt(0.75) = 0.36 / 0.866 = 0.416
  not GPU → pipeline_cost_scale 不变 = 0.408

期望结果 (±10%):
  fusion_cost_scale ≈ 0.42
  pipeline_cost_scale ≈ 0.41
  k_depth_preference ≈ 1.37
  epilogue_depth_preference ≈ 1.46
  batch_flexibility = 0.0 (精确)

解读: 融合和流水线几乎等权偏好 (0.42 vs 0.41)，pipeline 略优先获得寄存器 (k_depth 1.37 > epilogue 1.46 的相对偏移较小)。
符合预期: 小模型 + Latency → memory-bound，GEMM 小所以 pipeline 和 fusion 都极重要。
```

### 15.5 向量 4: Qwen3-7B + Throughput + GPU_A100

```
GraphArchetype: { compute: 0.73, memory: 0.40, parallel: 0.15, fusion: 0.80, pipeline: 0.45 }

阶段 1 — mode_baseline(Throughput):
  fusion_cost_scale = 1.0
  parallelism_cost_scale = 0.5
  batch_flexibility = 1.0
  speculative_decoding_value = 0.3

阶段 2 — archetype 调制:
  fusion_cost_scale = 1.0 × lerp(1.0, 0.6, 0.80) = 1.0 × 0.68 = 0.68
  parallelism_cost_scale = 0.5 × lerp(1.0, 0.5, 0.15) = 0.5 × 0.925 = 0.463
  not MoE (parallel < 0.5) → 无专家调制

阶段 3 — 硬件微调 (GPU_A100):
  fusion_cost_scale: l1_richness = 0.75, 0.68 / 0.866 = 0.785

期望结果 (±10%):
  fusion_cost_scale ≈ 0.79
  parallelism_cost_scale ≈ 0.46
  batch_flexibility = 1.0 (精确)
  speculative_decoding_value ≈ 0.3

解读: 中度偏好融合 (0.79)，中度偏好并行 (0.46)。
符合预期: Dense 7B + Throughput + GPU → 平衡策略，GEMM 够大不需要极端偏向。
```

### 15.6 测试验证规则

**REQ-ARB-014**: 实现必须包含上述 4 个黄金测试向量作为单元测试。每个 StrategyBias 字段的实际值与期望值的偏差不超过 ±10%。测试失败意味着仲裁算法实现有误。

```rust
#[test]
fn test_golden_vector_llama70b_latency_cpu_avx2() {
    let archetype = GraphArchetype {
        compute_intensive: 0.95,
        memory_intensive: 0.60,
        parallelism_exploitable: 0.20,
        fusion_profitable: 0.85,
        pipeline_valuable: 0.15,
    };
    let profile = DeviceProfile::test_cpu_avx2();
    let bias = StrategyArbiter::arbitrate(InferenceMode::Latency, &archetype, &profile);

    assert_approx(bias.fusion_cost_scale, 0.47, 0.10);
    assert_approx(bias.pipeline_cost_scale, 0.56, 0.10);
    assert_approx(bias.parallelism_cost_scale, 1.35, 0.10);
    assert_approx(bias.epilogue_depth_preference, 2.33, 0.10);
    assert_approx(bias.k_depth_preference, 0.92, 0.10);
    assert_eq!(bias.batch_flexibility, 0.0);
}

fn assert_approx(actual: f64, expected: f64, tolerance_ratio: f64) {
    let diff = (actual - expected).abs();
    let max_diff = expected.abs() * tolerance_ratio;
    assert!(diff <= max_diff,
        "expected ≈{expected}, got {actual} (diff {diff} > tolerance {max_diff})");
}
```

## 16. 常见实现错误与反模式

### 16.1 ❌ 注入方向搞反

```rust
// ❌ 错误: fusion_cost_scale < 1.0 时反而提高了融合代价
effective_savings = fusion_savings × bias.fusion_cost_scale;  // 应该是 ÷

// ✅ 正确: fusion_cost_scale < 1.0 时融合收益放大
effective_savings = fusion_savings × (1.0 / bias.fusion_cost_scale);
// 或等价地: effective_savings = fusion_savings / bias.fusion_cost_scale;
```

### 16.2 ❌ 直接读 InferenceMode 做分支

```rust
// ❌ 错误: 绕过 StrategyBias
if plan.inference_mode == InferenceMode::Latency {
    self.max_batch = 1;
    self.disable_continuous_batching();
}

// ✅ 正确: 通过 bias 传导
self.max_batch = (base_max as f64 * plan.strategy_bias.batch_flexibility) as usize;
// Latency mode: batch_flexibility = 0.0 → max_batch = 0 → 实际 max(1, 0) = 1
```

**注意**: `batch_flexibility = 0.0` 会导致 `max_batch = 0`，调用方必须 `max(1, computed)`。

### 16.3 ❌ validate() 改为 panic

```rust
// ❌ 错误: 数值计算边界值不应 panic
fn validate(&self) -> Result<(), ArbiterError> {
    if self.fusion_cost_scale <= 0.0 { return Err(...); }
}

// ✅ 正确: clamp 到合法范围
fn validate(&mut self) {
    self.fusion_cost_scale = self.fusion_cost_scale.clamp(0.2, 3.0);
}
```

### 16.4 ❌ sigmoid 用错形式

```rust
// ❌ 错误: 用 tanh 代替 sigmoid
fn sigmoid(x: f64) -> f64 { x.tanh() }  // 值域 (-1, 1)，不是 (0, 1)

// ❌ 错误: 用 ReLU 代替 sigmoid
fn sigmoid(x: f64) -> f64 { x.max(0.0).min(1.0) }  // 线性 clamp，不是 S 曲线

// ✅ 正确: 标准逻辑函数
fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }
```

### 16.5 ❌ GraphArchetype 维度相加归一化

```rust
// ❌ 错误: 五维之和归一化为 1.0
let total = c + m + p + f + pl;
archetype.compute_intensive = c / total;
// 这会丢失绝对强度信息！五维独立，不要求和为 1.0

// ✅ 正确: 每个维度独立 clamp 到 [0, 1]
archetype.compute_intensive = clamp(compute_raw, 0.0, 1.0);
archetype.memory_intensive = clamp(memory_raw, 0.0, 1.0);
// 五维之和可以 > 1.0 也可以 < 1.0
```

## 17. 关联文档

| 文档 | 关系 |
|------|------|
| `02-HARDWARE.md §10` HwOptEngine | Arbiter 输出注入 HwOptEngine |
| `04-API-DESIGN.md §2` Client Builder | `InferenceMode` API 入口 |
| `05-OPTIMIZATIONS.md` | MoE/SpecDecoding 模块消费 StrategyBias |
| `06-RUNTIME.md §3` ContinuousBatcher | batch 策略受 `batch_flexibility` 影响 |
| `DOCS/scheduling/jit-cache-protocol.md` | JIT 缓存 key 包含 InferenceMode |

## 18. Integration Trace — 真实调用链追踪

> **NO_ISLAND_MODULE 合规**: 以下是从用户 API 到最终消费的完整调用链，每一跳标明文件和函数。

```
Client::builder()
    .model("Qwen/Qwen3-7B")
    .inference_mode(InferenceMode::Throughput)    ← src/client.rs:ClientBuilder::inference_mode()
    .build()                                       ← src/client.rs:ClientBuilder::build()
  │
  └→ ClientBuilder::build_state(model_id, kind, inference_mode)   ← src/client.rs
       │
       ├── [1] Loader 加载模型 → ModelConfig 提取
       │   └→ ModelConfig::from_loader()           ← src/model_config.rs
       │
       ├── [2] GraphProfiler::profile(&model_config)    ← src/graph/profile.rs
       │   └→ GraphProfile { hidden_dim, num_experts, ffn_kind, ... }
       │
       ├── [3] GraphArchetype::derive(&graph_profile)   ← src/graph/profile.rs
       │   └→ GraphArchetype { compute: 0.73, memory: 0.40, ... }
       │
       ├── [4] ArbiterHwView::from(device_profile)      ← src/engine/arbiter.rs
       │
       ├── [5] StrategyArbiter::arbitrate(mode, &archetype, &hw_view)  ← src/engine/arbiter.rs
       │   └→ StrategyBias { fusion_cost_scale: 0.79, parallelism: 0.46, ... }
       │
       ├── [6] 转换 gllm::StrategyBias → gllm_kernels::StrategyBias   ← src/client.rs
       │
       ├── [7] init_global_execution_plan_with_bias(&kernels_bias)
       │   └→ gllm-kernels/src/compiler/planner.rs:EXECUTION_PLAN.get_or_init()
       │       └→ HwOptEngine::solve(ir, profile, &bias)
       │           ├── solve_cache_budget(..., &bias)     // L2 kv/weight 比例调制
       │           ├── solve_gemm(..., &bias)              // k_depth + epilogue 调制
       │           ├── solve_fusion(..., &bias)            // tile_threshold 调制
       │           ├── solve_parallelism(..., &bias)       // wave_count 调制
       │           ├── solve_batch(..., &bias)             // batch_flexibility 调制
       │           └── solve_features(..., &bias)          // spec_decoding 调制
       │
       └── [8] BackendContext::new()                      ← 此后 JIT 编译使用已注入 bias 的 ExecutionPlan
```

**验证命令** (grep 确认非测试调用):
```bash
# StrategyArbiter::arbitrate 在真实路径被调用
grep -rn "StrategyArbiter::arbitrate\|arbitrate_cpu" src/ | grep -v test | grep -v "mod tests"
# 期望: src/client.rs 中有调用

# init_global_execution_plan_with_bias 在真实路径被调用
grep -rn "init_global_execution_plan_with_bias" src/ | grep -v test
# 期望: src/client.rs 中有调用

# StrategyBias 不仅存在于定义和测试
grep -rn "StrategyBias" src/ | grep -v test | grep -v "mod tests" | grep -v "///" | grep -v "pub struct\|pub use"
# 期望: src/client.rs 中有实例化和传递
```
