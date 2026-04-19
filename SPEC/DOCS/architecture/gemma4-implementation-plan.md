# Gemma 4 实现计划

> P0-P3 详细执行计划,每个任务标注影响文件、依赖关系、验证方法和当前状态。
> 状态图例: ✅ 已完成 | 🟡 进行中 | ⏸ 未启动

## 依赖图

```
P0.1 config解析 ─────────────────────────────────────┐
  │                                                   │
  ├──→ P0.2 QkNorm ──┐                               │
  ├──→ P0.3 ValueNorm ┼──→ P1.2 FusedQkvNormRope     │
  ├──→ P0.4 DualRoPE ─┘                              │
  ├──→ P0.5 Attention per-layer ──→ P0.6 E2E验证(31B) │
  │                                                   │
  ├──→ P1.1 SharedKvRef ──→ P2.1 PLE ──→ P2.2 E2E验证(E2B)
  │
  └──→ P3.1 Vision ──→ P3.2 Audio ──→ P3.3 多模态路由
```

---

## P0 — 核心算子 (所有 Gemma 4 模型)

### P0.1 config.json 解析扩展  ✅ 已完成 (T21)

**目标**: 从模型元数据提取 Gemma 4 特有的配置字段。

**文件**:
| 文件 | 变更 |
|------|------|
| `gllm/src/model_config.rs` | `ModelGeometry` 新增 `global_rope_theta: f64`, `attention_pattern: Vec<u8>`, `num_kv_shared_layers: usize`, `hidden_size_per_layer_input: Option<usize>`, `global_head_dim: Option<usize>` |
| `gllm/src/model_config.rs` | GGUF/JSON 解析逻辑提取新字段 |
| `gllm/src/arch/resolve.rs` | `ResolvedConfig` 新增对应字段 |

**验证**: 单元测试 — 构造 Gemma 4 config.json，断言解析出正确的 `attention_pattern` 和 `global_rope_theta=1e6`。

**依赖**: 无

---

### P0.2 QkNorm 算子  ✅ 已完成 (T22)

**目标**: Q/K 向量 L2 归一化 + √head_dim 缩放。

**计算**: `Q_out = Q / ‖Q‖₂ × √head_dim`

**文件**:
| 文件 (gllm-kernels) | 变更 |
|---------------------|------|
| `src/compiler/registry.rs` | `OpKindKey` 新增 `QkNorm` |
| `src/compiler/graph.rs` | `OpKind` 新增 `QkNorm { head_dim: usize }` |
| `src/compat/scalar_ops.rs` | 标量参考实现 `scalar_qk_norm()` |
| `src/compiler/semantic_dag.rs` | `OpClass::ElemWise` 分类 |
| `src/compiler/codegen/x86_64/jit/` | SIMD: `vrsqrt14ps` + `vmulps` |
| `src/compiler/codegen/aarch64_dynasm/` | NEON: `frsqrte` + `fmul` |
| `src/compiler/codegen/gpu_ir/` | GPU: rsqrt + mul |

| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/decoder_forward.rs` | Gemma4 forward 调用 QkNorm |
| `src/graph/types.rs` | OnnxGraph op 映射 |

**验证**: JIT 编译 QkNorm → 与 scalar 参考对比，误差 < 1e-5。

**依赖**: 无

---

### P0.3 ValueNorm 算子  ✅ 已完成 (T23)

**目标**: V 向量无学习参数 RMSNorm。

**计算**: `V_out = V / √(mean(V²) + ε)` (无 weight 乘法)

**文件**:
| 文件 (gllm-kernels) | 变更 |
|---------------------|------|
| `src/compiler/registry.rs` | `OpKindKey` 新增 `ValueNorm` |
| `src/compiler/graph.rs` | `OpKind` 新增 `ValueNorm { eps: f32 }` |
| `src/compat/scalar_ops.rs` | 标量参考实现（RmsNorm 去掉 weight mul） |
| `src/compiler/codegen/*/` | 复用 RmsNorm codegen，skip weight step |

| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/decoder_forward.rs` | Gemma4 forward 调用 ValueNorm |
| `src/graph/types.rs` | OnnxGraph op 映射 |

**验证**: JIT 编译 ValueNorm → 与 scalar 参考对比。

**依赖**: 无（可与 P0.2 并行）

---

### P0.4 DualRotaryEmbedding 算子  ✅ 已完成 (T24)

**目标**: 根据层类型选择不同的 RoPE 配置。

| 层类型 | θ | partial | head_dim |
|--------|------|---------|----------|
| Sliding | 10,000 | 1.0 (全维度) | 256 |
| Global | 1,000,000 | 0.25 (p-RoPE) | 512 |

**文件**:
| 文件 (gllm-kernels) | 变更 |
|---------------------|------|
| `src/compiler/graph.rs` | `OpKind::RoPE` 新增 `partial: f32` 字段 (0.0~1.0) |
| `src/compat/scalar_ops.rs` | `scalar_rope()` 增加 partial 参数 — 仅旋转前 `partial × head_dim` 维度 |
| `src/compiler/codegen/*/` | RoPE codegen 增加 partial 跳过逻辑 (75% 维度跳过 = 75% 减少 sin/cos) |

| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/decoder_forward.rs` | 根据 `attention_pattern[layer_idx]` 选择 θ 和 partial |
| `src/graph/executor.rs` | 模板展开时按层类型设置 RoPE 参数 |

**验证**:
1. 单元测试: partial=1.0 → 全维度旋转 (现有行为)
2. 单元测试: partial=0.25 → 仅前 25% 维度旋转
3. 数值对比: p-RoPE 输出的后 75% 维度应与输入相同

**依赖**: P0.1 (需要 `attention_pattern` 和 `global_rope_theta`)

---

### P0.5 Attention per-layer type  ✅ 已完成 (T25/T35)

**目标**: 同一模型中不同层使用不同注意力模式 (sliding-window / global)。

**方案**: 现有 `Attention` op 已支持 `sliding_window` 属性。扩展为 per-layer:
- 执行器根据 `attention_pattern[layer_idx]` 决定:
  - `0` → sliding-window (使用 config.sliding_window)
  - `1` → global (无窗口限制)

**文件**:
| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/decoder_forward.rs` | `attention_step()` 接收 `layer_type` 参数 |
| `src/compat/cpu_backend.rs` | sliding-window attention mask 条件生成 |
| `src/engine/executor.rs` | 传递 `attention_pattern` 到前向传播 |

**验证**: 构造 4 层模型 (3 sliding + 1 global)，验证 sliding 层输出受窗口约束、global 层不受约束。

**依赖**: P0.1

---

### P0.6 P0 集成验证  ✅ 已完成 (T26/T27,覆盖于 T40 全管线 dry-run)

**目标**: 使用 Gemma 4 31B GGUF 量化版验证纯 P0 路径 (无 SharedKV/PLE)。

**文件**:
| 文件 | 变更 |
|------|------|
| `tests/test_e2e_generator.rs` | 新增 `e2e_generator_gemma4` 测试 |

**验证**: 加载 GGUF 模型 → generate → 反退化检查 + 语义正确性。

**依赖**: P0.1 ~ P0.5 全部完成

---

## P1 — KV 共享 + 融合优化

### P1.1 SharedKvRef — 层间 KV 复用  ✅ page 层已完成 (T39) | 🟡 graph 层进行中 (T43)

**目标**: E2B 后 20 层、E4B 后 18 层不计算 K/V，引用前一个非共享同类层的 KV。

**当前状态**:
- ✅ Scheduler/KvCache/WeightLoader page 级共享已接入 (commit `f5acc67`, T39)
- 🟡 graph executor 层跳过 K/V MatMul 的路径并行实现中 (T43)

**核心逻辑**:
```
for layer_i in 0..num_layers:
    if layer_i >= (num_layers - num_kv_shared_layers):
        # 共享层: 找到最近的非共享同类层
        donor = find_donor(layer_i, attention_pattern)
        K, V = kv_cache[donor]  # 引用，不拷贝
    else:
        # 非共享层: 正常计算 K/V
        K = k_proj(hidden)
        V = v_proj(hidden)
        kv_cache[layer_i] = (K, V)
```

**文件**:
| 文件 (gllm) | 变更 |
|-------------|------|
| `src/scheduler/paged_scheduler.rs` | 共享层分配 page 引用 (非新 page)，引用计数 |
| `src/kv_cache.rs` | `KvPageHeader` 新增 `donor_layer: Option<usize>` 字段 |
| `src/compat/decoder_forward.rs` | 共享层跳过 K/V MatMul，从 donor 层读取 |
| `src/weight_loader.rs` | 共享层不加载 `k_proj.weight` / `v_proj.weight` (权重文件中不存在) |
| `src/model_config.rs` | `ModelGeometry.num_kv_shared_layers` 驱动共享决策 |

**验证**:
1. 单元测试: 共享层的 KV 与 donor 层完全相同 (pointer equality)
2. 单元测试: 非共享层正常计算 KV
3. 内存测试: 共享层不分配新 KV page

**依赖**: P0.1

---

### P1.2 FusedQkvNormRope 融合模式  ✅ 已完成 (T29/T36/T41/T42)

**目标**: 6 个算子融合为 1 个内核。

**当前状态**: pattern_fusion 识别 (commit `9697fb8`, T29) + atomic_op_to_kind 契约 (T36) + 两处根治性 bug 修复 (commit `dedd86d`, T41/T42) 全部完成。

```
Q_proj → K_proj → V_proj → QkNorm(Q,K) → ValueNorm(V) → DualRoPE(Q,K)
                                    ↓
                        FusedQkvNormRope (单内核)
```

**收益**: 省 5 次 intermediate buffer 写回 (Q/K/V/Q_normed/K_normed)。

**文件**:
| 文件 (gllm-kernels) | 变更 |
|---------------------|------|
| `src/compiler/fusion/mod.rs` | 新增 `FusionRule::QkvNormRope` |
| `src/compiler/codegen/*/` | 融合内核 codegen: 3×Gemm + 2×L2Norm + 1×RmsNorm + RoPE |

| 文件 (gllm) | 变更 |
|-------------|------|
| `src/graph/optimizer/pattern_fusion.rs` | 新增 `FusedQkvNormRope` pattern |

**验证**: 融合前后数值输出误差 < 1e-5。

**依赖**: P0.2 + P0.3 + P0.4

---

## P2 — Per-Layer Embedding

### P2.1 PerLayerEmbedding 算子  ✅ 已完成 (JIT: T30/T33/T37/T38 | atomic 契约: T36)

**目标**: E2B/E4B 每层注入 token-identity + context-aware 信号。

**当前状态**:
- ✅ 权重加载 alias 解析 + bind_weight_shapes_fuzzy (commit `6f6b648`, T28.2)
- ✅ YAML `only_if` 条件展开机制 (commit `774a51f`, T28.3)
- ✅ PLE 5-input 跨层一致签名 (commit `9dc283b`, T33)
- ✅ `PleSlice` → `OpKind::ColumnSlice` 真实列切片 JIT (commit `bab213c`, T37)
- ✅ GPU codegen (PTX/HIP/MSL) 接入 (T38)
- ✅ atomic_op_to_kind 契约测试 + 白名单移除 (commits `7dd6520`/`c0d0382`, T36)

**计算**:
```
ple_token = per_layer_embed[:, i*d : (i+1)*d]   // 按层切片
ple_ctx   = linear(main_embed)                    // context-aware 投影
signal    = (ple_ctx + ple_token × √d) / √2
hidden    = hidden + post_mlp_proj(signal)        // 残差注入
```

**新增权重**:
- `model.per_layer_embedding.embed_tokens.weight` — `[vocab_size, num_layers × dim_per_layer]`
- `model.per_layer_embedding.per_layer_projection.weight`
- `model.layers.{i}.post_mlp_projection.weight`

**文件**:
| 文件 (gllm-kernels) | 变更 |
|---------------------|------|
| `src/compiler/registry.rs` | `OpKindKey::PerLayerEmbed` |
| `src/compiler/graph.rs` | `OpKind::PerLayerEmbed { layer_idx, dim_per_layer }` |

| 文件 (gllm) | 变更 |
|-------------|------|
| `src/weight_loader.rs` | 加载 PLE 权重 (新的权重路径) |
| `src/compat/decoder_forward.rs` | 每层 FFN 后注入 PLE signal |
| `src/graph/types.rs` | 新 op 映射 |
| `src/arch/templates/gemma4.yaml` | 条件启用 PLE 节点 |

**验证**: PLE 输出 ≠ 0 且不同层的 PLE 信号不同。

**依赖**: P0.1

---

### P2.2 E2B E2E 验证  🟡 进行中 (T47: 模拟加载 + compile dry-run ✅,真实 HF 下载 + 数值对齐 ⏸)

**当前状态**: T40 全管线 dry-run (commit `99a09ad`) 已验证 compile 路径通,T47 真实 HF 下载和数值对齐仍在处理中。

**文件**:
| 文件 | 变更 |
|------|------|
| `tests/test_e2e_generator.rs` | 新增 `e2e_generator_gemma4_e2b` (需 PLE + SharedKV) |

**依赖**: P0 全部 + P1.1 + P2.1

---

## P3 — 多模态

### P3.1 Vision Encoder (SigLIP)  🟡 骨架就位 (T44 进行中)

**目标**: 图像 → 视觉 token 序列。

**当前状态**: `src/compat/audio_forward.rs` 骨架 + `PatchEmbed` / `LearnedPos2D` OpKind 占位完成,SigLIP 前向传播和权重加载并行实现中。

**架构**: SigLIP ViT — Patch embedding → Transformer encoder → 2D 学习位置编码。

**文件**:
| 文件 (gllm) | 变更 |
|-------------|------|
| `src/arch/templates/gemma4.yaml` | 新增 vision encoder 子图 |
| `src/compat/vision_forward.rs` | **新文件**: SigLIP 前向传播 |
| `src/loader/mod.rs` | 加载 vision encoder 权重 |

**新增算子**:
| 算子 | 用途 |
|------|------|
| `PatchEmbed` | 图像 → patch 序列 (Conv2D + Reshape) |
| `LearnedPos2D` | 2D 学习位置编码 (非 RoPE) |

**依赖**: P0 全部

---

### P3.2 Audio Encoder (USM Conformer)  🟡 骨架就位 (T45 进行中)

**目标**: 音频 → 音频 token 序列。

**当前状态**: `src/compat/audio_forward.rs` 骨架 + `DepthwiseConv1D` OpKind 占位完成,Conformer 前向传播和权重加载并行实现中。

**架构**: USM-style Conformer — 卷积 + 自注意力混合编码器。

**文件**:
| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/audio_forward.rs` | **新文件**: Conformer 前向传播 |
| `src/loader/mod.rs` | 加载 audio encoder 权重 |

**新增算子**:
| 算子 | 用途 |
|------|------|
| `DepthwiseConv1D` | Conformer 的深度卷积模块 |
| `ConformerBlock` | 卷积 + 注意力混合块 |

**依赖**: P0 全部

---

### P3.3 多模态 Token 路由  ⏸ 未启动

**目标**: 文本 / 图像 / 音频 token 的统一序列化和路由。

**逻辑**:
```
input_ids 中:
  258880 (image_token_id)  → 插入 vision encoder 输出的 token 序列
  258881 (audio_token_id)  → 插入 audio encoder 输出的 token 序列
  其他                     → 正常文本 embedding
```

**文件**:
| 文件 (gllm) | 变更 |
|-------------|------|
| `src/compat/decoder_forward.rs` | embedding 层插入多模态 token |
| `src/client.rs` | API 支持多模态输入 (image path / audio path) |
| `src/generation.rs` | `GenerationBuilder` 支持 `.image()` / `.audio()` |

**依赖**: P3.1 + P3.2

---

## 时间线摘要

| 阶段 | 内容 | 任务数 | 核心复杂度 | 状态 |
|------|------|--------|-----------|------|
| **P0** | 核心算子 + E2E 验证 | 6 | DualRoPE (中), 其余低 | ✅ 全部完成 (T21–T27) |
| **P1** | SharedKvRef + 融合 | 2 | SharedKvRef (**高**) | ✅ FusedQkvNormRope 完成 (T29/T36/T41/T42);SharedKvRef page 层 ✅ (T39),graph 层 🟡 (T43) |
| **P2** | PLE + E2B 验证 | 2 | 中 | ✅ PLE 全链路完成 (T28.2/T28.3/T30/T33/T36/T37/T38);E2B E2E 验证 🟡 (T47 dry-run ✅,数值对齐待) |
| **P3** | 多模态 | 3 | Vision/Audio encoder (高), 路由 (中) | 🟡 Vision 骨架 (T44) + Audio 骨架 (T45) 并行中;多模态路由 ⏸ 未启动 |

**里程碑**:
- ✅ P0.6 完成 → Gemma 4 31B GGUF 路径通 (含 T40 全管线 dry-run)
- 🟡 P2.2 完成中 → Gemma 4 E2B compile dry-run ✅,数值对齐待 T47
- ⏸ P3.3 完成 → Gemma 4 多模态全功能可用
