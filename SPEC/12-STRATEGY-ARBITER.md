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
    ReLU,         // up + relu + down (legacy GPT-2)
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

## 5. HwOptEngine 集成

### 5.1 扩展后的 solve() 签名

```rust
impl HwOptEngine {
    pub fn solve(
        profile: &DeviceProfile,
        sensors: &MemoryNetworkSensors,
        probe: &ProbeResult,
        model: &ModelConfig,
        bias: &StrategyBias,    // 新增: Arbiter 输出
    ) -> Result<HwOptPlan, OptError>
}
```

### 5.2 CostModel 注入点

StrategyBias 通过 CostModel 的调制系数影响所有求解器。注入方式为乘法调制——成本函数的各项开销乘以对应的 bias 系数。

#### 5.2.1 GemmSolver 注入

```
// 候选评估成本公式（原 §10.4）:
cost(candidate) = max(T_compute, T_memory) + T_overhead

// 注入后:
T_overhead_pipeline = k_depth × prefetch_latency × bias.pipeline_cost_scale
// pipeline_cost_scale < 1.0 → pipeline 开销被低估 → GemmSolver 更倾向选择深 k_depth

// k_depth 候选过滤:
if bias.k_depth_preference > 1.0:
    k_depth 候选范围从 [1, 2] 扩展到 [1, 2, 4]（如果寄存器允许）
if bias.k_depth_preference < 0.8:
    k_depth 固定为 1

// epilogue 深度调制:
effective_max_epilogue = (base_max_epilogue as f64 × bias.epilogue_depth_preference) as usize
// epilogue_depth_preference > 1.0 → 允许更深 epilogue（可能触发 NR 缩减）
```

#### 5.2.2 FusionSolver 注入

```
// 融合收益估算（原 §10.6）:
fusion_savings(mode, ops) = ...

// 注入后:
effective_savings = fusion_savings × (1.0 / bias.fusion_cost_scale)
// fusion_cost_scale < 1.0 → 融合收益被放大 → FusionSolver 更倾向选择融合

// TileLevelFusion vs ComputeRoot 阈值调制:
effective_threshold = base_threshold × bias.fusion_cost_scale
// fusion_cost_scale < 1.0 → 阈值降低 → 更多情况选择 TileLevelFusion
```

#### 5.2.3 CacheBudgetSolver 注入

```
// L2 预算分配（原 §10.5）:
kv_budget     = L2 × 0.40 × bias.kv_cache_budget_scale
weight_budget = L2 × 0.35 × bias.weight_prefetch_budget_scale
activation    = L2 × (1.0 - kv_ratio - weight_ratio)

// Latency mode: kv × 0.5 = 20%, weight × 1.5 = 52.5%, activation = 27.5%
//   → 权重预取占大头（memory-bound 下权重是主要带宽消费）
// Throughput mode: kv × 1.5 = 60%, weight × 0.8 = 28%, activation = 12%
//   → KV cache 占大头（多序列共存）

// 归一化: 总和超 100% 时按比例缩放
```

#### 5.2.4 ParallelismSolver 注入

```
// wave_count 候选评估（原 §10.8）:
wave_cost = synchronization_cost(wave_count) × bias.parallelism_cost_scale
// parallelism_cost_scale < 1.0 → 同步代价被低估 → 更倾向多 wave
```

#### 5.2.5 BatchSolver 注入

```
// batch 大小约束:
if bias.batch_flexibility == 0.0:
    max_batch_size = 1  // Latency mode: 固定 batch=1
else:
    max_batch_size = base_max × bias.batch_flexibility

// decode_ratio_cap 调制:
effective_ratio_cap = base_ratio_cap × bias.decode_ratio_scale
```

#### 5.2.6 FeatureRouter 注入

```
// Speculative Decoding 启用决策:
spec_benefit = base_benefit × bias.speculative_decoding_value
// speculative_decoding_value > 1.0 → 收益被放大 → 更倾向启用

// 量化决策:
quant_benefit = base_benefit × bias.quantization_aggressiveness
// quantization_aggressiveness > 1.0 → 更倾向低比特
```

### 5.3 HwOptPlan 扩展

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

## 10. 关联文档

| 文档 | 关系 |
|------|------|
| `02-HARDWARE.md §10` HwOptEngine | Arbiter 输出注入 HwOptEngine |
| `04-API-DESIGN.md §2` Client Builder | `InferenceMode` API 入口 |
| `05-OPTIMIZATIONS.md` | MoE/SpecDecoding 模块消费 StrategyBias |
| `06-RUNTIME.md §3` ContinuousBatcher | batch 策略受 `batch_flexibility` 影响 |
| `DOCS/scheduling/jit-cache-protocol.md` | JIT 缓存 key 包含 InferenceMode |
