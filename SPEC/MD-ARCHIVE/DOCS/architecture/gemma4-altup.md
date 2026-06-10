# AltUp (Alternating Updates) + PerLayerEmbedding 技术协议

> **SSOT** — Gemma 4 E2B/E4B 的 PLE 注入机制基于 AltUp 框架。
> 现有 `gemma4-op-audit.md §1.5` 中的 PLE 公式为**已废弃近似**（无 AltUp、无 LAuReL、无门控乘法），
> 本文档完全取代 §1.5。

## 0. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 层间携带状态 | 单 `[S,H]` + AltUp 拆散 | **胖激活 `[P,S,H]`** | 不破坏单模板 ping-pong；宪法禁向后兼容 |
| per_layer_inputs 存储 | 每层重算 projection | **预计算 + 持久激活 buffer** | projection 是一次性全层大 GEMM，按层重算无法廉价切片 |
| per_layer_inputs 布局 | column-major `[S, L·hpl]` + ColumnSlice | **layer-major `[L, S, hpl]` + 步进指针** | cache 友好；与 weight stride 机制对称；无需动态 ColumnSlice |
| AltUp OpKind 粒度 | 单一复合 PerLayerEmbed | **3 个细粒度 OpKind** | predict/correct/inject 是不可再约的新计算模式；其余用现有算子组合 |
| 现有 PerLayerEmbed | 保留 + 扩展 | **物理删除** | 公式不等价于官方；宪法禁向后兼容 |

## 1. AltUp 架构概览

AltUp (Alternating Updates) 把每层的残差流从 1 路 `[S, H]` 扩展为 P 路并行预测 `[P, S, H]`，
每层执行："预测全部 P 路 → 仅对 active 路（index 0）跑真实 attention/FFN → 用 active 路结果修正全部 P 路"。

### 1.1 Config 参数

| 参数 | Gemma 4 E2B | Gemma 4 E4B | 说明 |
|------|-------------|-------------|------|
| `altup_num_inputs` (P) | 2 | 2 | 并行预测路数（官方默认 4，Gemma 4 E2B/E4B 为 2） |
| `altup_active_idx` | 0 | 0 | Active 预测索引 |
| `altup_coef_clip` | 120.0 | 120.0 | 预测/修正系数裁剪值 |
| `altup_correct_scale` | true | true | 是否对 active 预测应用 correct_output_scale |
| `hidden_size_per_layer_input` (hpl) | 256 | 256 | 每层 PLE 嵌入维度 |

> **注**：官方 Gemma3nConfig 默认 `altup_num_inputs=4`，但 Gemma 4 E2B/E4B 实际使用
> `altup_num_inputs=2`。需在模型 config.json 中确认具体值。若实际为 4，则本协议中
> 所有 `[2,S,H]` 应替换为 `[4,S,H]`，buffer 预算相应调整。

### 1.2 数据流全景

```
┌─────────────────────────────────────────────────────────────────┐
│ Prefill 阶段 (循环外，一次性计算)                                │
│                                                                 │
│  token_ids ──→ Gather(embed_per_layer) ──→ ple_tokens [S,L·hpl]│
│                                      reshape ──→ [L,S,hpl]      │
│                                                                 │
│  hidden_0 ──→ per_layer_model_projection(GEMM) ──→ proj [S,L·hpl]│
│                × hidden_size⁻⁰·⁵  →  reshape ──→ [L,S,hpl]     │
│                RMSNorm(hpl)                                     │
│                                                                 │
│  per_layer_inputs[L,S,hpl] = (projection + ple_tokens × √hpl)  │
│                               × 1/√2                            │
│  → 存入 PersistentActivation buffer                             │
│                                                                 │
│  AltUp init:                                                    │
│  hidden_0 [S,H] → stack [P,S,H]                                │
│    slice[0] = hidden_0                                          │
│    slice[1..P] = altup_projection[i](hidden_0) + 幅度归一化     │
│  → 存入 ping buffer [P,S,H]                                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 层循环 (每层执行，模板 × N)                                      │
│                                                                 │
│  ping [P,S,H] = 上层输出                                        │
│                                                                 │
│  ── AltUp Predict ──                                            │
│  modalities = tanh(router_norm(active) × H⁻¹ @ W_router)        │
│  coefs = prediction_coefs(modalities)  → [P,P] per position     │
│  predictions[p] = hidden[p] + Σ_q coefs[p,q]·hidden[q]         │
│                                                                 │
│  ── Active path (slice 0, [S,H]) ──                             │
│  active = predictions[0]                                        │
│  normed = input_norm(active)                                    │
│  laurel = LAuReL(normed)                                        │
│  attn = MHA(normed)                                             │
│  attn_laurel = (active + attn + laurel) / √2                    │
│  ffn = MLP(pre_ffn_norm(attn_laurel))                           │
│  gated = attn_laurel + ffn                                      │
│                                                                 │
│  ── AltUp Correct ──                                            │
│  innovation = gated - predictions[0]                            │
│  corrected_coefs = correction_coefs(modalities) + 1.0           │
│  corrected[p] = predictions[p] + corrected_coefs[p]·innovation  │
│  corrected[0] = gated  (active 路直接更新)                       │
│  if altup_correct_scale:                                        │
│    corrected[0] ×= correct_output_scale                         │
│                                                                 │
│  ── PLE 门控注入 ──                                              │
│  ple_input = per_layer_inputs[layer_idx]  [S,hpl] (步进指针)    │
│  gate = GELU(per_layer_input_gate(corrected[0]))  [S,hpl]       │
│  gated_ple = gate × ple_input                                   │
│  projected = per_layer_projection(gated_ple)  [S,H]             │
│  normalized = post_per_layer_input_norm(projected)  RMSNorm     │
│  corrected[1:] += normalized  (注入到非 active 预测)             │
│                                                                 │
│  pong [P,S,H] = corrected                                       │
│  ActivationSwap → 下层 ping = 本层 pong                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ AltUp Unembed (循环后)                                           │
│                                                                 │
│  对于 i=1..P-1:                                                 │
│    hidden[i] = altup_unembed_projection[i](hidden[i])           │
│    magnitude normalize to match hidden[0]                       │
│  result = mean(hidden[0..P-1], dim=0)  [S,H]                   │
│  result = final_norm(result)                                    │
│  → 继续到 lm_head                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 权重清单

### 2.1 AltUp 核心权重 (模型级，循环外)

| 权重名 | 形状 | 说明 |
|--------|------|------|
| `altup.correct_output_scale` | `[H]` | Active 预测修正缩放 |
| `altup.correction_coefs.weight` | `[P, P]` | 修正系数矩阵 |
| `altup.prediction_coefs.weight` | `[P², P]` | 预测系数矩阵 (reshape 为 [P,P]) |
| `altup.modality_router.weight` | `[P, H]` | 模态路由器 |
| `altup.router_norm.weight` | `[H]` | 路由器前 RMSNorm |

### 2.2 AltUp 初始化/反嵌权重 (模型级，循环外)

| 权重名 | 形状 | 说明 |
|--------|------|------|
| `altup_projections.{i}.weight` | `[H, H]` × (P-1) | 初始化：将 hidden_0 投影到非 active 槽 |
| `altup_unembed_projections.{i}.weight` | `[H, H]` × (P-1) | 反嵌：将非 active 槽投影回 |

### 2.3 PLE 权重 (模型级，循环外)

| 权重名 | 形状 | 说明 |
|--------|------|------|
| `per_layer_embedding.embed_tokens.weight` | `[vocab_size, L·hpl]` | PLE token embedding 表 |
| `per_layer_embedding.per_layer_projection.weight` | `[L·hpl, H]` | Context-aware 投影 |

### 2.4 PLE 权重 (层循环内，per-layer)

| 权重名 | 形状 | 说明 |
|--------|------|------|
| `layers.{i}.per_layer_input_gate.weight` | `[hpl, H]` | 门控线性投影 H→hpl |
| `layers.{i}.per_layer_projection.weight` | `[H, hpl]` | PLE 投影 hpl→H |
| `layers.{i}.post_per_layer_input_norm.weight` | `[H]` | PLE 后 RMSNorm |

### 2.5 LAuReL 权重 (层循环内，per-layer)

| 权重名 | 形状 | 说明 |
|--------|------|------|
| `layers.{i}.laurel_up.weight` | `[laurel_rank, H]` | LAuReL 升维投影 |
| `layers.{i}.laurel_down.weight` | `[H, laurel_rank]` | LAuReL 降维投影 |
| `layers.{i}.laurel_norm.weight` | `[H]` | LAuReL 后 RMSNorm |

> `laurel_rank` = 64 (Gemma 4 E2B/E4B)

## 3. 新增 OpKind

### 3.1 AltUpPredict

```
AltUpPredict {
    num_preds: usize,     // P = altup_num_inputs (编译期常量)
    hidden: usize,        // H
}
```

**语义**：对 `[P,S,H]` 胖激活做 altup 维混合预测。

```
输入: stacked [P,S,H], router_coefs [S, P²] (由外部 GEMM+Tanh 产出)
输出: predictions [P,S,H]

predictions[p][s][h] = stacked[p][s][h] + Σ_q coefs[p][q] · stacked[q][s][h]
```

**OpClass**: Injective（多输入多输出逐元素，P≤4 ≤ UNROLL_THRESHOLD 可内层展开）

**TraceOp 表达**：
- 外层 `emit_loop(s ∈ [0, seq_len) Symbolic)`
- 内层 `for p in 0..P` (Const 展开, P≤4)
- 逐 h_vec: VecLoad stacked[p], ScalarLoad coefs[p][q], Fma 链, VecStore predictions[p]

### 3.2 AltUpCorrect

```
AltUpCorrect {
    num_preds: usize,     // P
    hidden: usize,        // H
}
```

**语义**：用 active 路的 innovation 修正所有 P 路预测。

```
输入: predictions [P,S,H], corrected_coefs [S, P], activated [S,H]
输出: corrected [P,S,H]

innovation[s][h] = activated[s][h] - predictions[0][s][h]
corrected[p][s][h] = predictions[p][s][h] + corrected_coefs[s][p] × innovation[s][h]
corrected[0] = activated  (active 路直接覆盖)
```

**OpClass**: Injective

**TraceOp 表达**：与 AltUpPredict 类似，但多一个 innovation 计算。

### 3.3 AltUpInject

```
AltUpInject {
    num_preds: usize,     // P
    hidden: usize,        // H
}
```

**语义**：将 PLE 门控结果注入到非 active 预测。

```
输入: corrected [P,S,H], ple_projected [S,H]
输出: corrected [P,S,H] (in-place 修改 slice 1..P)

corrected[p][s][h] += ple_projected[s][h]   for p = 1..P-1
```

**OpClass**: Injective（多输出写回）

**TraceOp 表达**：外层 Symbolic s 循环 + 内层 `for p in 1..P` (Const 展开) + VecLoad+VecAdd+VecStore。

## 4. Buffer 生命周期

### 4.1 胖激活 Ping-Pong

现有 `ActivationSwap` 机制零改动复用，仅将 buffer 尺寸从 `[S,H]` 扩展为 `[P,S,H]`。

| Buffer | 尺寸 | 生命周期 |
|--------|------|---------|
| `altup_ping` | `P × S × H × 4B` | 层循环全程，ping-pong 交替 |
| `altup_pong` | `P × S × H × 4B` | 层循环全程，ping-pong 交替 |

E2B (H=2048, S=512, P=2): 每个 buffer 8MB，共 16MB。

### 4.2 PersistentActivation: per_layer_inputs

新增 buffer 类：全 Mega-Kernel 生命周期，可写一次读多次。

| Buffer | 尺寸 | 生命周期 |
|--------|------|---------|
| `per_layer_inputs` | `L × S × hpl × 4B` | Mega-Kernel 全程，层循环只读 |

E2B (L=35, S=512, hpl=256): ~18MB。

### 4.3 Scratch buffer (层内临时)

| Buffer | 尺寸 | 生命周期 |
|--------|------|---------|
| `active_path` | `S × H × 4B` | 层内，attention/FFN/LAuReL scratch |
| `ple_ctx` | `S × hpl × 4B` | 层内，PLE 投影中间结果 |
| `innovation` | `S × H × 4B` | 层内，AltUp correct 中间结果 |

## 5. 与 SharedKvRef 的交互

两者正交共存，无需特殊协调：

- **SharedKvRef GprCondAction** (`LayerIdxLt(num_layers - num_kv_shared_layers)`)：
  守护 K/V 子 op（k_proj/v_proj/k_norm/v_norm/rope_k）
- **AltUp ops**：全部 `guard=Always`（每层无条件执行）
- op 集不相交，互不干扰

## 6. 与 LAuReL 的关系

LAuReL (Learned Augmented Residual Layer) 是 Gemma 4 E2B/E4B 的另一个新子系统，
与 AltUp 正交但协同工作。

**计算**：
```
laurel_hidden = GEMM(normed, laurel_up)     // [S, H] → [S, 64]
laurel_hidden = GELU(laurel_hidden)
laurel_output = GEMM(laurel_hidden, laurel_down)  // [S, 64] → [S, H]
laurel_output = RMSNorm(laurel_output, laurel_norm)
```

**融合点**：`attn_laurel = (active + attn + laurel_output) / √2`
这是 3 路加法 + 缩放，可融合为单个 elementwise op。

## 7. 废弃声明

- **`OpKind::PerLayerEmbed`**：已废弃，物理删除。其 scalar 实现 (`scalar_per_layer_embed`)
  和专用 lower (`lower_per_layer_embed`) 一并删除。
- **`PleScratchRequirement`**：已废弃，被 `PersistentActivation` 取代。
- **`gemma4-op-audit.md §1.5`**：已废弃，以本文档为 SSOT。

## 8. 实施依赖 DAG

```
P0 SPEC 建档 (本文档) ──┬─→ P1a PersistentActivation buffer 类
                        ├─→ P1b LayerLoopConfig 扩展: per_layer_input_stride
                        ├─→ P1c 新增 3 OpKind scalar 实现
                        └─→ P1d 删除废弃 PerLayerEmbed + lower + scratch
                                    │
P1c ──→ P2 SymExec/registry + TraceOp dispatch (auto_select.rs)
                                    │
P1a,P1b,P2 ──→ P3 build_graph 重构:
  ├─ 删除 PLE 块 (:245-297)
  ├─ 插入 AltUp init + PLE precompute (循环前)
  ├─ 层模板改胖 buffer [P,S,H] + predict/LAuReL/correct/gate/inject
  └─ 插入 AltUp unembed + mean (循环后)
                                    │
P3 ──→ P4 ISA lowering 验证 + 数值对齐
                                    │
P4 ──→ P5 E2E: gemma-4-E2B dry-run + cos_sim ≥ 0.999
```
