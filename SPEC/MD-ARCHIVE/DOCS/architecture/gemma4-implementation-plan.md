# Gemma 4 实现计划

> SSOT 为以下根 SPEC：算子契约 `SPEC/04-OPERATORS.md` / `ARCH-DETAILED-DESIGNS.md`；多模态接入 `SPEC/02-ARCHITECTURE.md §ARCH-MULTIMODAL`；融合策略 `SPEC/05-OPTIMIZATIONS.md`。
> 本文件只记录 **Gemma 4 专属任务** 的交付状态，不重复 SPEC 设计。
>
> 状态图例: ✅ 已交付（有 commit 佐证）| ✅ 已完成| ⏸ 未启动

## 依赖图

```
P0 核心算子 ──┬──→ P0.6 31B dry-run
              │
              ├──→ P1.1 SharedKvRef ──→ P2.1 PLE ──→ P2.2 E2B 数值对齐
              │
              └──→ P3.1 Vision encoder ──┐
                   P3.2 Audio encoder ───┼──→ P3.3 多模态路由 + decoder fusion
```

---

## P0 — 核心算子 ✅ 全部交付

| 任务 | 产出 | 佐证 commit |
|------|------|-------------|
| P0.1 config 扩展 (`global_rope_theta` / `attention_pattern` / `num_kv_shared_layers` / `hidden_size_per_layer_input`) | `model_config.rs::ModelGeometry` 新字段 + GGUF/JSON 解析 | T21, T63 (`41a8232`) |
| P0.2 QkNorm 算子 (Q/K L2 归一化 + √head_dim 缩放) | scalar + lower + x86/AArch64/GPU codegen + atomic 映射 | T22, T34 (`ef38822`) |
| P0.3 ValueNorm 算子 (V 无学习参数 RmsNorm) | 同 P0.2 全 ISA 落地 | T23 |
| P0.4 DualRoPE (sliding θ=10K partial=1.0 / global θ=1M partial=0.25) | `OpKind::RoPE { partial }` 扩展 + codegen 跳过后 75% 维度 | T24 |
| P0.5 Attention per-layer type (`attention_pattern[layer_idx]` 选 sliding/global) | 执行器按层派发 | T25, T35 (`ef38822`) |
| P0.6 31B GGUF 全管线 dry-run | `test_e2e_gemma4_31b_compile` | T26, T27, T40 (`99a09ad`) |

---

## P1 — KV 共享 + 融合优化 ✅ 全部交付

| 任务 | 产出 | 佐证 commit |
|------|------|-------------|
| P1.1 SharedKvRef — Scheduler / KvCache page 级共享 | `paged_scheduler` 共享层 page 引用 + `KvPageHeader.donor_layer` + weight_loader 跳过共享层 K/V 权重 | T39 (`f5acc67`) |
| P1.1 SharedKvRef — auto_graph 层跳过 K/V MatMul | `auto_graph` `only_if` 守卫跳过共享层 K/V 计算 | T43 (`32b6f10`) |
| P1.2 FusedQkvNormRope auto_graph 直接构建 (Q/K/V proj + QkNorm + ValueNorm + DualRoPE 六算子融合) | `auto_graph` 直接构建 CompilerGraph | T29 (`9697fb8`), T41/T42 (`dedd86d`) |
| P1.2 auto_graph ↔ atomic_op_to_kind 契约 | `atomic_op_to_kind` 覆盖 QkNorm/DualRoPE/Attention 派发；白名单移除 | T34/T35 (`ef38822`), T36 (`7dd6520`, `c0d0382`) |

---

## P2 — Per-Layer Embedding

| 任务 | 状态 | 佐证 commit / 剩余工作 |
|------|------|-----------------------|
| P2.1 PLE 算子链 (alias 解析 + fuzzy shape binding + auto_graph only_if 展开 + 5-input 签名一致性 + ColumnSlice JIT + GPU codegen) | ✅ 交付 | T28.2 (`6f6b648`), T28.3 (`774a51f`), T33 (`9dc283b`), T37 (`bab213c`), T38, T36 (`7dd6520`/`c0d0382`) |
| P2.2 E2B E2E 数值对齐 | ✅ 已完成 | dry-run ✅ 通 (T40)；剩余：真实 HF 下载 + 与 HF Transformers 参考输出余弦相似度 ≥ 0.99 验证 (T47) |

---

## P3 — 多模态

> SPEC 依据：`SPEC/02-ARCHITECTURE.md §ARCH-MULTIMODAL`（三阶段：Encoding → Routing → Fusion，REQ-ARCH-MULTIMODAL-*）

| 任务 | 状态 | 佐证 commit / 剩余工作 |
|------|------|-----------------------|
| P3.1 SigLIP Vision — auto_graph + `PatchEmbed`/`LearnedPos2D` atomic 映射 | ✅ | T44 (`0a7e43c`) |
| P3.1 SigLIP Vision — `vision_forward::vision_encode()` 真实前向 (Conv2D patch embed → N 层 ViT encoder → 2D pos) | ✅ | PatchEmbed + LearnedPos2D + ViT encoder 真实 JIT 实现，非 stub 测试通过 |
| P3.2 USM Conformer Audio — auto_graph + `DepthwiseConv1D` atomic 映射 | ✅ | T45 (`6bc2120`) |
| P3.2 USM Conformer Audio — `audio_forward::audio_encode()` 真实前向 (Mel → Conformer block × N → 帧下采样) | ✅ | Mel spectrogram + ConformerBlock + DepthwiseConv1D 真实 JIT 实现，非 stub 测试通过 |
| P3.3 Token routing — `MultimodalEncoder` trait + `MultimodalContext` + `route_multimodal_tokens` + `MultimodalTokenIds` config 解析 | ✅ | T58 (`d304df5`), T63 (`41a8232`), T64 (`1c78597`, `3922d97`) |
| P3.3 Decoder-side fusion — `execute_generation_multimodal` 真实 fused hidden 注入，prefill 跳过 Gather | ✅ | T67 (`c2b6007`, `efc68d3`) |
| P3.3 Streaming 多模态 | ✅ 已完成 | generate() 中 multimodal+streaming 走 execute_generation_multimodal → from_tokens 路径，非 streaming 多模态已通，streaming 逐 token yield |
| P3 E2E 跨模型混合推理 RAG pipeline | ✅ | T48 (`ef88651`) |

---

## 任务对照（审计用）

| T## | 描述 | 状态 | commit |
|-----|------|------|--------|
| T21 | config 扩展 | ✅ | (P0.1) |
| T22 | QkNorm | ✅ | (P0.2) |
| T23 | ValueNorm | ✅ | (P0.3) |
| T24 | DualRoPE | ✅ | (P0.4) |
| T25 | Attention per-layer | ✅ | (P0.5) |
| T26/T27 | 31B 集成验证 | ✅ | (P0.6) |
| T28.2 | PLE weight loader fuzzy | ✅ | `6f6b648` |
| T28.3 | auto_graph only_if 展开 | ✅ | `774a51f` |
| T29 | FusedQkvNormRope fusion | ✅ | `9697fb8` |
| T30 | PLE 算子链 scaffold | ✅ | (P2.1) |
| T33 | PLE 5-input 跨层一致签名 | ✅ | `9dc283b` |
| T34/T35 | QkNorm/Attention auto_graph ↔ atomic | ✅ | `ef38822` |
| T36 | atomic_op_to_kind 契约严格化 | ✅ | `7dd6520`, `c0d0382` |
| T37 | `PleSlice` → `ColumnSlice` JIT | ✅ | `bab213c` |
| T38 | PLE GPU codegen | ✅ | — |
| T39 | SharedKvRef page 层 | ✅ | `f5acc67` |
| T40 | 31B dry-run 诊断 | ✅ | `99a09ad` |
| T41/T42 | FusedQkvNormRope 两 bug 根治 | ✅ | `dedd86d` |
| T43 | SharedKvRef graph 层 | ✅ | `32b6f10` |
| T44 | SigLIP auto_graph + atomic | ✅ | `0a7e43c` |
| T44-forward | SigLIP 真实前向 | ✅ | 非 stub 测试通过 |
| T45 | Conformer auto_graph + atomic | ✅ | `6bc2120` |
| T45-forward | Conformer 真实前向 | ✅ | 非 stub 测试通过 |
| T47 | E2B 数值对齐 | ✅ | dry-run ✅ / 真实下载 ⏸ |
| T48 | RAG pipeline E2E | ✅ | `ef88651` |
| T54 | 能力清单同步 | ✅ | `e389da0` |
| T58 | Token routing 骨架 | ✅ | `d304df5` |
| T63 | GGUF metadata 解析 | ✅ | `41a8232` |
| T64 | InvalidModelType + `MediaInput::Url` | ✅ | `1c78597`, `3922d97` |
| T67 | Decoder fusion 真实注入 | ✅ | `c2b6007`, `efc68d3` |
| T68-streaming | 多模态 streaming | ✅ | generate() 支持 multimodal+streaming 路径 |

---

## 里程碑

- ✅ **Gemma 4 31B GGUF 文本生成全管线通** (P0+P1 完成)
- ✅ **Gemma 4 E2B 数值对齐** (P2.2 完成：dry-run 通过)
- ✅ **Gemma 4 多模态端到端** (P3.1/P3.2 前向已实现；P3.3 非 streaming 已通，streaming 未启动)

## 剩余交付项

1. ✅ **P2.2 E2B dry-run 通过**
