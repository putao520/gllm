# Gemma 4 算子审计与优化方案

> Gemma 4 (2026-04-02) 支持所需的新增算子、执行路径变更、融合模式、JIT codegen 扩展全面审计。

## 1. 新增算子

### 1.1 DualRotaryEmbedding — 双轨 RoPE

**用途**: 每层根据注意力类型 (sliding/global) 选择不同的 RoPE 配置。

| 属性 | Sliding Window 层 | Global 层 |
|------|-------------------|-----------|
| θ | 10,000 | 1,000,000 |
| 旋转比例 | 100% (全维度) | **25% (p-RoPE)** — 仅旋转前 25% head_dim |
| head_dim | 256 | **512** (全局层加倍) |

**与现有 RotaryEmbedding 的关系**:
- 现有 `RotaryEmbedding` 是单 θ、单 partial 比例
- `DualRotaryEmbedding` 在运行时根据 `attention_pattern[layer_idx]` 选择配置
- JIT 编译时可生成两份代码（sliding 变体 + global 变体），dispatch-time 选择

**JIT codegen 扩展**:
- Phase 2 Fusion: 新增 `DualRotaryEmbedding` → 按层类型分裂为 `RotaryEmbedding(θ=10K, partial=1.0)` 或 `RotaryEmbedding(θ=1M, partial=0.25)`
- Phase 3 ISA: 复用现有 RoPE codegen，仅参数不同
- **优化**: p-RoPE (partial=0.25) 时 75% 维度跳过旋转 → 减少 75% 的 sin/cos 计算

**复杂度**: 中。核心 RoPE 计算复用，新增层类型分发 + partial 跳过逻辑。

---

### 1.2 QkNorm — Query-Key 归一化

**用途**: 替代 Gemma 2 的 Softcap。对 Q 和 K 向量做 L2 归一化后缩放。

**计算**:
```
Q_norm = Q / ‖Q‖₂ × √head_dim
K_norm = K / ‖K‖₂ × √head_dim
```

**与现有算子关系**:
- Gemma 2 使用 `LogitSoftCap` (注意力 logit 层面截断)
- QkNorm 在投影后、RoPE 前作用于 Q/K 向量 (不同位置、不同语义)

**JIT codegen**:
- Phase 0: Scalar 实现 = L2 norm + scale
- Phase 1: SemanticDAG → `OpClass::ElemWise` (逐元素归一化)
- Phase 2: 可融合到 `FusedQkvNormRope` (Q/K/V 投影 → QkNorm → ValueNorm → RoPE 一气呵成)
- Phase 3: SIMD 实现 = `rsqrt(dot(x,x)) * scale * x`，直接利用 AVX-512 的 `vrsqrt14ps`

**复杂度**: 低。纯 ElemWise 算子，融合收益大。

---

### 1.3 ValueNorm — Value 向量归一化

**用途**: 对 V 向量做无学习参数的 RMSNorm。

**计算**:
```
V_norm = V / √(mean(V²) + ε)    // 无 weight 参数
```

**与现有 SimplifiedLayerNormalization 的关系**:
- `SimplifiedLayerNormalization` = RMSNorm with learned scale weight
- `ValueNorm` = RMSNorm **without** learned scale (省一次乘法)

**JIT codegen**:
- 复用 RMSNorm codegen，skip weight multiplication
- 可融合到 `FusedQkvNormRope` 中

**复杂度**: 极低。现有 RMSNorm 的子集。

---

### 1.4 SharedKvRef — 层间 KV 复用 (调度器层)

**用途**: E2B 后 20 层、E4B 后 18 层不计算 K/V 投影，直接引用前一个非共享层的 KV cache page。

**影响范围**:
- **不是图算子**，而是**调度器/KV cache 层面的优化**
- PagedAttention 的 block table 需支持跨层引用
- 共享层的 K/V 投影权重不存在于模型文件中（无 `k_proj.weight`/`v_proj.weight`）

**实现方案**:
1. 模型加载时检测 `num_kv_shared_layers` 配置
2. 共享层的 K/V 投影 MatMul 节点在图中标记为 `skip_if_shared`
3. `PagedScheduler` 为共享层分配引用而非新 page
4. 前向传播时共享层只计算 Q 投影 + 注意力（K/V 从引用层读取）

**复杂度**: 高。涉及调度器、KV cache、权重加载三个子系统。

---

### 1.5 PerLayerEmbedding (PLE) — 并行条件注入 (仅 E2B/E4B)

**用途**: 每层注入额外的 token-identity + context-aware 信号，增强小模型表达能力。

**计算**:
```
ple_token = embed_per_layer[:, layer_i * dim : (layer_i+1) * dim]  // token identity
ple_context = linear_proj(main_embedding)                           // context-aware
ple_signal = (ple_context + ple_token * √dim) / √2
hidden = hidden + post_mlp_proj(ple_signal)                         // 残差注入
```

**新增权重**:
- `model.per_layer_embedding.embed_tokens.weight` — [vocab_size, num_layers × hidden_per_layer]
- `model.per_layer_embedding.per_layer_projection.weight` — context-aware 投影
- `model.layers.{i}.post_mlp_projection.weight` — 残差注入投影

**JIT codegen**:
- Phase 1: 独立子图，不参与 attention/FFN 融合
- Phase 2: 可与 post-FFN residual Add 融合
- Phase 3: 3 个 MatMul + 1 个 Add + scaling，标准 SIMD

**复杂度**: 中。新的权重加载路径 + 新的残差注入点。31B/26B 不需要此算子。

---

## 2. 现有算子扩展

### 2.1 Attention — per-layer 属性扩展

**变更**: 现有 `Attention` op 的 `sliding_window` 属性是全局常量。Gemma 4 需要 per-layer 动态选择。

**方案**: 新增 `per_layer_type: 1` 属性，指示执行器根据 `attention_pattern[layer_idx]` 选择:
- `attention_pattern[i] == 0` → sliding-window attention (window=512/1024)
- `attention_pattern[i] == 1` → global full attention (无窗口)

**JIT 影响**: 编译两份注意力变体（sliding + global），dispatch-time 选择。

### 2.2 GELU — 已存在

Gemma 4 使用标准 GELU (PyTorch tanh 近似)。现有 `GELU` op 已支持，无需扩展。

### 2.3 SimplifiedLayerNormalization — 已存在

ε=1e-6，与现有实现兼容。

---

## 3. 融合模式

### 3.1 FusedQkvNormRope (新)

```
Q_proj → K_proj → V_proj → QkNorm(Q,K) → ValueNorm(V) → DualRoPE(Q,K)
```

6 个算子融合为 1 个内核。收益：
- 省 5 次 intermediate 写回 (Q/K/V/Q_normed/K_normed)
- QkNorm 的 rsqrt 和 RoPE 的 sin/cos 可共享寄存器

### 3.2 FusedLinearGELU (已有，复用)

```
up_proj → GELU → down_proj
```

复用现有 `FusedLinearGELU`（up_proj + GELU + down_proj 的 epilogue fusion 模式）。

### 3.3 FlashAttention (已有，扩展)

需处理两种模式:
- Sliding-window: causal + 窗口裁剪 (现有 mistral3 已支持)
- Global: 标准 causal (现有 qwen3 已支持)

---

## 4. 实现优先级

| # | 项目 | 影响 | 复杂度 | 依赖 |
|---|------|------|--------|------|
| P0 | `DualRotaryEmbedding` | 所有 Gemma 4 模型 | 中 | 现有 RotaryEmbedding |
| P0 | `QkNorm` | 所有 Gemma 4 模型 | 低 | 无 |
| P0 | `ValueNorm` | 所有 Gemma 4 模型 | 极低 | 现有 RMSNorm |
| P0 | Attention per-layer type | 所有 Gemma 4 模型 | 低 | 现有 Attention |
| P0 | config.json 解析 (`global_rope_theta`, `attention_pattern`, `num_kv_shared_layers`) | 所有模型 | 低 | model_config.rs |
| P1 | `SharedKvRef` | E2B/E4B | 高 | PagedScheduler |
| P1 | `FusedQkvNormRope` 融合 | 性能优化 | 中 | P0 算子全部完成 |
| P2 | `PerLayerEmbedding` | 仅 E2B/E4B | 中 | 权重加载 |
| P3 | 多模态 (Vision/Audio) | 完整多模态 | 非常高 | 超出当前范围 |

---

## 5. 模型尺寸与 E2E 测试方案

| 模型 | 参数量 | E2E 可行性 |
|------|--------|-----------|
| gemma-4-E2B-it | 2.3B (effective) / 5.1B (total) | 边界可行 (需 PLE + SharedKV) |
| gemma-4-E4B-it | 4.5B (effective) / 8B (total) | 过大 |
| gemma-4-27B-it | 27B | 不可行 |
| gemma-4-26B-A4B-it | 26B MoE | 不可行 |

**E2E 代表模型**: `google/gemma-4-E2B-it` (最小，但需要 P0+P1+P2 全部完成)

**最小可行测试** (仅 P0): 若跳过 SharedKV 和 PLE，可先用 31B 的 GGUF 量化版验证纯 P0 路径。
