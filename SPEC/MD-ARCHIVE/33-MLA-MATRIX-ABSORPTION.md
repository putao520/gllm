# Multi-head Latent Attention (MLA) 矩阵吸收 (ARCH-MLA)

> **实现状态**: ✅ 全部完成 — REQ-MLA-001~008 (MlaConfig + 特征检测 + OpKind + TraceOp + JIT dispatch + PagedKV + 模型注册) + REQ-MLA-009 (合成数据数值验证: standard vs absorbed cos_sim >= 0.999, KV 压缩 readback < 1e-5, 56.9× 压缩比)

## 定位

DeepSeek V3/R1/Kimi-K2 的 MLA 低秩 KV 压缩与矩阵吸收实现。通过低秩分解将 KV cache 压缩 56.9×，再利用矩阵乘法结合律（Matrix Absorption）避免运行时 KV 还原，在 decode 阶段实现极致带宽优化。

> **SSOT**: 本 SPEC 定义 MLA 的完整实现规范。数学推导见 `DOCS/architecture/fwht-mla-pagedaddr-reference.md §2`。
> 分段路由策略见 `DOCS/architecture/2026-frontier-reference.md §2`。

## 前置原则

- **ARCH-MLA-UNIFIED**: MLA 是 Attention 的变体，不是独立后端路径。MLA 模型走与标准 MHA 相同的 `compile_from_auto_graph` 编译路径，差异仅在 `auto_graph` 的图构建层
- **ARCH-MLA-ABSORB-DEFAULT**: Absorbed 路径是默认路径（decode + 长序列 prefill），Un-absorbed 仅用于短序列 prefill 的算力饱和场景
- **ARCH-MLA-KV-LATENT**: PagedKV-Cache 存储 `c_KV`（低秩向量）+ `k_pe`（解耦 RoPE key），不存储全量 K/V
- **ARCH-MLA-JIT-ABSORPTION**: `W_UK^T` 吸收到 Q 的矩阵乘在 JIT 编译时 bake 为 GEMM 节点，运行时零分支

## 维度定义

| 符号 | 含义 | DeepSeek V3 值 |
|------|------|---------------|
| `d` | 每 Head 维度 (head_dim) | 128 / 192 |
| `n_h` | Query 头数 (num_attention_heads) | 128 |
| `n_kv` | KV 头数 (num_key_value_heads) | 128 (MHA, MLA 下无独立 KV 头概念) |
| `d_c` | KV 压缩低秩维度 (latent_dim) | 512 |
| `d_rope` | 解耦 RoPE 专属维度 (rope_dim) | 64 |
| `d_model` | 模型隐藏层维度 | 7168 |

### 每-Token 缓存物理量

| 方案 | 缓存维度 | 缓存字节数 (FP16) |
|------|---------|------------------|
| **MLA (c_KV + k_pe)** | `d_c + d_rope = 512 + 64 = 576` | 1152 B |
| **标准 MHA (K + V)** | `2 × n_h × d = 2 × 128 × 128 = 32768` | 65536 B |
| **压缩比** | **56.9×** | — |

## 架构

```
                    标准 MHA 路径
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Q_proj   │    │ K_proj   │    │ V_proj   │
│ X·W_Q    │    │ X·W_K    │    │ X·W_V    │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │          KV Cache 存储         │
     │          全量 K/V             │
     │               │               │
     └───────────────┼───────────────┘
                     │
              FlashAttention
              Q × K^T → scores → × V

                    MLA 路径
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Q_proj   │    │ KV 压缩  │    │ RoPE key │
│ X·W_Q    │    │ X·W_DKV  │    │ X·W_KR   │
└────┬─────┘    │ → c_KV   │    │ → k_pe   │
     │          └────┬─────┘    └────┬─────┘
     │               │               │
     │          KV Cache 存储         │
     │          低秩 c_KV + k_pe     │
     │               │               │
     ├──── Absorbed ─┤               │
     │ Q·W_UK^T     │               │
     │ → Q_absorbed │               │
     └───────────────┼───────────────┘
                     │
              MLA Attention
              Q_absorbed × c_KV^T → scores → × V_up
              (k_pe 参与 RoPE 后与 c_KV 拼接)
```

## MLA 数学模型

### KV 压缩 (降维)

```
c_KV = X · W_DKV    (W_DKV ∈ R^{d_model × d_c})
k_pe = X · W_KR     (W_KR ∈ R^{d_model × d_rope})
```

PagedKV-Cache 存储: `c_KV` (d_c=512) + `k_pe` (d_rope=64) = 576 维。

### Absorbed Pathway (矩阵吸收)

传统解压: `K = c_KV · W_UK`  → 标准注意力 `Score = Q · K^T`

利用结合律: `Score = Q · (c_KV · W_UK)^T = (Q · W_UK^T) · c_KV^T`

1. **吸收到 Q**: `Q_absorbed = Q · W_UK^T` (维度: [M, d] × [d, d_c] → [M, d_c])
2. **压缩注意力**: `Score = Q_absorbed · c_KV^T` (维度: [M, d_c] × [d_c, kv_len] → [M, kv_len])
3. **V 还原**: `V = c_KV · W_UV` (维度: [kv_len, d_c] × [d_c, d] → [kv_len, d])

### 解耦 RoPE

RoPE 不参与矩阵吸收（位置信息必须在 attention score 计算时施加）:

```
拼接: key_for_attn = concat(c_KV_no_rope, rope(k_pe))
         维度: [d_c - d_rope, d_rope]  →  c_KV 后 d_rope 维替换为 RoPE(k_pe)
```

**关键**: DeepSeek MLA 的 RoPE 处理方式是将 `k_pe` 通过 RoPE 后替换 `c_KV` 的后 `d_rope` 维，而非标准 RoPE 作用于全维度。

### Un-absorbed Pathway (短文本 Prefill)

```
K = c_KV · W_UK     (还原完整 K, 维度 [kv_len, n_h × d])
V = c_KV · W_UV     (还原完整 V, 维度 [kv_len, n_h × d])
→ 标准 FlashAttention(Q, K, V)
```

**适用场景**: Prefill 短序列 (≤ threshold)，此时算力饱和，还原 K/V 的 FLOPs 开销可接受，
标准 FlashAttention 的大 tile 利用率更高。threshold 由 `seq_len × n_h × d` vs SM 算力预算动态决定。

## 依赖 SPEC

| SPEC | 依赖点 |
|------|--------|
| `SPEC/02-ARCHITECTURE.md §11` | TurboQuant 运行时优化框架 |
| `SPEC/08-EXECUTOR.md` | Executor 执行模型 |
| `SPEC/11-MODELS.md` | DeepSeek V3/R1 架构描述（需补充 MLA 字段） |
| `SPEC/03-DATA-STRUCTURE.md` | KV Cache 数据结构 |
| `SPEC/20-BATCH-CONCURRENT-INFERENCE.md` | BatchContext per-seq KV 维度 |
| `DOCS/architecture/fwht-mla-pagedaddr-reference.md §2` | MLA 数学推导 SSOT |

## REQ 清单

### REQ-MLA-001: MlaConfig 配置扩展

`ModelGeometry` / `ResolvedConfig` 新增 MLA 配置字段。

**设计**:
```rust
/// MLA (Multi-head Latent Attention) 配置
pub struct MlaConfig {
    /// KV 压缩低秩维度 (latent_dim, DeepSeek V3 = 512)
    pub d_c: usize,
    /// 解耦 RoPE 维度 (rope_dim, DeepSeek V3 = 64)
    pub d_rope: usize,
    /// W_DKV 权重名模式 (如 "kv_b_proj")
    pub w_dkv_pattern: String,
    /// W_UK 权重名模式 (如 "k_b_proj") — 用于 Matrix Absorption
    pub w_uk_pattern: String,
    /// W_UV 权重名模式 (如 "v_b_proj") — 用于 V 还原
    pub w_uv_pattern: String,
    /// W_KR 权重名模式 (如 "k_pe_proj") — 解耦 RoPE key
    pub w_kr_pattern: String,
    /// Un-absorbed 阈值 (token 数，短 prefill ≤ 此值走还原路径)
    pub unabsorbed_threshold: usize,
}
```

**GGUF 元数据映射**:
- `deepseek_mla.d_c` → `d_c`
- `deepseek_mla.d_rope` → `d_rope`
- GGUF 没有 `unabsorbed_threshold`，默认值从 `DeviceProfile` 算力推导

**关键文件**:
- `gllm/src/model_config.rs`: `ResolvedConfig` 新增 `mla_config: Option<MlaConfig>`
- `gllm/src/loader/gguf/reader.rs`: GGUF 元数据解析
- `gllm-kernels/SPEC/GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md`: ABI 参数布局（KV cache 维度参数化）

### REQ-MLA-002: ArchitectureFeatures MLA 检测

`auto_graph.rs::analyze_architecture()` 自动检测 MLA 架构。

**设计**:
- `ArchitectureFeatures` 新增字段:
  ```rust
  pub is_mla: bool,                    // MLA 模型
  pub mla_latent_dim: usize,           // d_c
  pub mla_rope_dim: usize,             // d_rope
  ```
- **检测逻辑**: 权重名包含 `kv_b_proj` / `k_b_proj` / `v_b_proj` (DeepSeek MLA 权重命名模式)
  ```rust
  let is_mla = role_index.keys().any(|(role, _)| {
      matches!(role, TensorRole::MlaKvCompress | TensorRole::MlaKeyAbsorb)
  });
  ```
- 新增 `TensorRole` 变体:
  ```rust
  MlaKvCompress,   // W_DKV — KV 压缩权重
  MlaKeyAbsorb,    // W_UK  — K 吸收权重
  MlaValueAbsorb,  // W_UV  — V 还原权重
  MlaRopeKey,      // W_KR  — 解耦 RoPE key 权重
  ```

**关键文件**:
- `gllm/src/arch/auto_graph.rs`: `ArchitectureFeatures` + `analyze_architecture()`
- `gllm/src/manifest/types.rs`: `TensorRole` 新增变体

### REQ-MLA-003: MLA Absorbed Attention Graph 构建

`auto_graph.rs::build_compiler_graph()` 构建 MLA absorbed 路径的 CompilerGraph。

**设计**:
当 `features.is_mla` 时，attention 子图替换为 MLA 路径:

```
标准 MHA:  input_norm → [Q_proj, K_proj, V_proj] → RoPE → Attention
MLA:       input_norm → [Q_proj, KV_compress, RoPE_key] → Absorb(Q) → MLA_Attention
```

**Absorbed 路径图节点**:
```
1. RmsNorm(input, norm_weight) → normed

2. Q_proj:    Gemm(normed, W_Q) → Q_raw          [M, n_h × d]
3. KV压缩:    Gemm(normed, W_DKV) → c_KV          [M, d_c]
4. RoPE key:  Gemm(normed, W_KR) → k_pe_raw       [M, d_rope]

5. Q reshape: Reshape(Q_raw, [M, n_h, d])
6. Q 吸收:   Gemm(Q_reshaped, W_UK^T) → Q_absorbed  [n_h, M, d_c]
   (W_UK 分 per-head: W_UK_m ∈ R^{d_c × d}, 每个 head 独立 GEMM)

7. KV cache 写: c_KV + k_pe 拼接 → 写入 PagedKV (维度 d_c + d_rope)

8. MLA Attention:
   a. 从 KV cache 读取: c_KV_history [kv_len, d_c] + k_pe_history [kv_len, d_rope]
   b. RoPE(k_pe_history) → k_pe_rope [kv_len, d_rope]
   c. 替换 c_KV_history 的后 d_rope 维: key = concat(c_KV[:, :d_c-d_rope], k_pe_rope)
   d. Score = Q_absorbed_m · key^T  (per-head, 维度 [M, d_c] × [d_c, kv_len])
   e. Score_scaled = Score / sqrt(d_c)  (缩放因子是 d_c 而非 d，因为 score 在压缩空间计算)
   f. Softmax(Score_scaled) → attn_weights [M, kv_len]
   g. V 还原: V_m = c_KV · W_UV_m  [kv_len, d_c] × [d_c, d] → [kv_len, d]
   h. Output = attn_weights × V_m → [n_h, M, d] → reshape → [M, n_h × d]
```

**关键约束**:
- Q 吸收 `Gemm(Q, W_UK^T)` 和 V 还原 `Gemm(c_KV, W_UV)` 是额外 GEMM，但维度小 (d × d_c = 128 × 512)，开销可忽略
- W_UK / W_UV 权重常驻 GPU/CPU，不参与 KV cache
- MLA Attention 的 score 计算在压缩空间 (d_c=512) 而非原始空间 (d=128)，减少了 softmax 计算量

**关键文件**:
- `gllm/src/arch/auto_graph.rs`: `build_compiler_graph()` MLA 分支
- `gllm-kernels/src/compiler/`: JIT 编译管线（新 OpKind 可能需要）

### REQ-MLA-004: MLA Un-absorbed Prefill 路径

短序列 Prefill 时走 KV 还原路径（标准 FlashAttention），优化算力利用率。

**设计**:
```
Un-absorbed 路径:
1. c_KV = Gemm(X, W_DKV)       [M, d_c]
2. K = Gemm(c_KV, W_UK)         [M, n_h × d]  — 还原完整 K
3. V = Gemm(c_KV, W_UV)         [M, n_h × d]  — 还原完整 V
4. Q = Gemm(X, W_Q)             [M, n_h × d]
5. RoPE(Q, K)
6. 标准 FlashAttention(Q, K, V)

KV Cache 仍然存储 c_KV + k_pe（不存全量 K/V）
```

**路由条件** (JIT 编译时确定):
```rust
// Prefill 路径选择: absorbed vs un-absorbed
// Absorbed: Q·W_UK^T 然后点积 c_KV (带宽优)
// Un-absorbed: 还原 K/V 然后标准 MHA (算力优)
fn select_prefill_path(seq_len: usize, device_profile: &DeviceProfile) -> MlaPrefillPath {
    // 短序列: 算力饱和, 还原 K/V 的 FLOPs 可接受, 大 tile FlashAttention 效率更高
    // 长序列: 带宽瓶颈, c_KV 加载量 56.9× 更少
    let compute_budget = device_profile.sm_count * device_profile.clock_mhz * 2; // 简化
    let seq_compute = seq_len * 128 * 128; // n_h × d × seq_len
    if seq_compute < compute_budget * 3 {
        MlaPrefillPath::Unabsorbed  // 算力有余, 走标准路径
    } else {
        MlaPrefillPath::Absorbed    // 带宽瓶颈, 走压缩路径
    }
}
```

**Mega-Kernel 中的实现**: Prefill 阶段的 PhaseDispatch（SPEC/39 §1.3.3 ForwardPhaseDispatch）根据 `total_prefill_tokens` 与阈值比较，选择 Absorbed 或 Un-absorbed 路径。两条路径编译为同一 Mega-Kernel 内的不同 device function，运行时零开销分支。

**关键文件**:
- `gllm/src/arch/auto_graph.rs`: `build_compiler_graph()` Un-absorbed 分支
- `SPEC/32-MEGA-KERNEL-ENHANCEMENT.md §1`: PhaseDispatch 集成

### REQ-MLA-005: MLA PagedKV-Cache 维度适配

KV Cache 按低秩维度 `d_c + d_rope` 存储，替代标准 `2 × n_kv × d`。

**设计**:

当前 PagedKV-Cache 页维度:
```
标准: page_data = [page_size, 2, num_kv_heads, head_dim]  // K + V
MLA:  page_data = [page_size, d_c + d_rope]                // c_KV + k_pe
```

**影响**:
- `PagedScheduler`: page 大小计算改为 `page_size × (d_c + d_rope) × elem_bytes`
- `KvPageHeader`: 无需修改，MLA 的 page 元数据与标准 KV 共用 56B header
- `BatchContext`: per-seq 的 `kv_dim` 字段从 `2 × n_kv × d` 变为 `d_c + d_rope`
- `gpu_alloc_kv_cache()`: 分配大小按 `d_c + d_rope` 计算

**约束**:
- MLA 模型不使用标准 GQA 的 `num_key_value_heads` 参数（KV 压缩后无独立 KV 头）
- `d_c` 和 `d_rope` 从 `MlaConfig` 读取，不从 `num_key_value_heads × head_dim` 推导

**关键文件**:
- `gllm/src/kv_cache.rs`: page 大小计算参数化
- `gllm/src/scheduler/paged_scheduler.rs`: MLA page 分配逻辑
- `gllm/src/compat/gpu_helpers.rs`: GPU KV cache 分配维度适配
- `SPEC/03-DATA-STRUCTURE.md`: KV Cache 数据结构描述更新
- `SPEC/20-BATCH-CONCURRENT-INFERENCE.md`: BatchContext kv_dim 字段语义

#### MLA ↔ SPEC 32 (MKO) 集成要点

| MKO 阶段 | MLA 特殊处理 |
|----------|------------|
| PhaseDispatch | Absorbed/Un-absorbed 路径选择基于 `total_prefill_tokens` vs `unabsorbed_threshold` |
| 层循环融合组（decode 路径） | 固定走 Absorbed 路径（decode 是 memory-bound，吸收后 KV 维度 d_c=512 远小于原始 2×n_kv×d） |
| 层循环融合组（prefill 路径） | 短文本走 Un-absorbed（算力饱和时 3.36× FLOPs 节省 > 带宽优化）；长文本走 Absorbed |
| Argmax/StoreToken 融合组（stream output） | 无特殊处理——MLA 不影响 token 输出格式 |
| Compact + Refill 融合组 | KV page 大小按 `d_c + d_rope` 计算，page 回收与标准 KV 相同逻辑 |
| SM 分区 Ping-Pong | Absorbed decode 的 GEMM tile 与标准 decode 不同（score 在 d_c 空间而非 d 空间），编译时参数化 |

**关键约束**:
- Absorbed/Un-absorbed 路径编译为同一 Mega-Kernel 内的不同 JMP 分支（SPEC 32 §1.1 的 PhaseDispatch）
- MLA 的 score 缩放因子是 `1/sqrt(d_c)` 而非标准 attention 的 `1/sqrt(d)`（影响 GEMM 参数化）
- Decode 阶段的 KV cache 读写维度是 `d_c + d_rope`（56.9× 压缩），不影响 MKO 的 compact/refill 逻辑

### REQ-MLA-006: MLA OpKind 扩展

新增 MLA 专用 OpKind，用于 JIT 编译管线识别和优化。

**设计**:
```rust
/// MLA KV 压缩: X · W_DKV → c_KV
/// 等价于 GEMM，但语义标记用于 KV cache 写入路径选择
/// 参数从 MlaConfig (d_c) 读取
MlaKvCompress,

/// MLA Q 吸收: Q · W_UK^T → Q_absorbed (per head)
/// Per-head GEMM, W_UK 分块为 [d_c × d] per head
/// 参数从 MlaConfig + ModelGeometry 读取
MlaQAbsorb,

/// MLA V 还原: c_KV · W_UV → V (per head)
/// Per-head GEMM, W_UV 分块为 [d_c × d] per head
/// 参数从 MlaConfig + ModelGeometry 读取
MlaVRestore,

/// MLA Attention: Q_absorbed × concat(c_KV_no_rope, RoPE(k_pe))^T → scores → × V
/// score 空间在 d_c 而非 d, softmax 输入维度不同
/// 参数从 MlaConfig + ModelGeometry 读取
MlaAttention,

/// MLA 解耦 RoPE: 替换 c_KV 的后 d_rope 维为 RoPE(k_pe)
/// 参数从 MlaConfig (d_c, d_rope) 读取
MlaRopeMerge,
```

**JIT 编译管线映射**:

| OpKind | ComputePattern | auto_select 路径 |
|--------|---------------|------------------|
| `MlaKvCompress` | Gemm | 标准降低 Gemm |
| `MlaQAbsorb` | Gemm (batched per-head) | 批处理降低 Gemm |
| `MlaVRestore` | Gemm (batched per-head) | 批处理降低 Gemm |
| `MlaAttention` | Structural | TraceOp 扩展 (MlaAttnScore + MlaRopeMerge) |
| `MlaRopeMerge` | Injective | 多输入多输出自动分发 |

**关键**: `MlaKvCompress`、`MlaQAbsorb`、`MlaVRestore` 本质是 GEMM，复用现有 JIT GEMM 路径。只有 `MlaAttention` 和 `MlaRopeMerge` 需要新的 TraceOp 语义。

**关键文件**:
- `gllm-kernels/src/compiler/graph_ir.rs`: `OpKind` 新增变体
- `gllm-kernels/src/compiler/codegen/vm/plan_lower.rs`: ComputePattern 分类
- `gllm-kernels/src/compiler/codegen/vm/auto_select.rs`: TraceOp → VmInstr 映射

### REQ-MLA-007: MLA TraceOp 语义扩展

为 `MlaAttention` 和 `MlaRopeMerge` 定义 TraceOp 语义，纳入自动指令选择。

**设计**:

```rust
/// MLA Attention score 计算 (压缩空间)
/// Trace: for each head m:
///   load Q_absorbed_m [1, d_c] from register
///   for each kv_pos:
///     load key[pos] [d_c] (concat c_KV + RoPE(k_pe))
///     score = dot_product(Q_absorbed_m, key[pos])
///   softmax(scores)
///   for each kv_pos:
///     load c_KV[pos] [d_c]
///     V_m[pos] = gemv(W_UV_m, c_KV[pos])  // per-head V 还原
///     weighted_sum += scores[pos] × V_m[pos]
TraceOp::MlaAttnScore,
// 参数从 MlaConfig + ModelGeometry 读取: num_heads, d_c, d_rope
// Trace: for each head m:
//   load Q_absorbed_m [1, d_c] from register
//   for each kv_pos:
//     load key[pos] [d_c] (concat c_KV + RoPE(k_pe))
//     score = dot_product(Q_absorbed_m, key[pos])
//   softmax(scores)
//   for each kv_pos:
//     load c_KV[pos] [d_c]
//     V_m[pos] = gemv(W_UV_m, c_KV[pos])  // per-head V 还原
//     weighted_sum += scores[pos] × V_m[pos]

/// MLA RoPE merge: 替换 c_KV 后 d_rope 维为 RoPE(k_pe)
/// Trace: concat(c_KV[:, :d_c-d_rope], RoPE(k_pe))
/// 参数从 MlaConfig 读取: d_c, d_rope
TraceOp::MlaRopeMerge,
```

**auto_select 映射**:
- `MlaAttnScore` → VmInstr 组合: `VecDotProduct` + `Softmax` + `Gemv` + `VecFma`（d_c 维度从配置读取）
- `MlaRopeMerge` → VmInstr 组合: `VecLoad` + `VecStore` + `RoPE` + `VecConcat`

**关键文件**:
- `gllm-kernels/src/compiler/codegen/vm/trace.rs`: TraceOp 新增变体
- `gllm-kernels/src/compiler/codegen/vm/auto_select.rs`: dispatch_trace_op 新增 arm

### REQ-MLA-008: MLA 模型注册

将 MLA 支持注册到模型架构系统，覆盖 DeepSeek V3/R1 和 Kimi-K2。

**设计**:

**SPEC/11-MODELS.md 更新**:
```
| `deepseek-v3` | DeepSeek (MLA) | 671B MoE (37B Active) | MLA: d_c=512, d_rope=64 | 旗舰 MoE + MLA |
| `deepseek-r1` | DeepSeek (MLA) | 671B MoE | MLA: d_c=512, d_rope=64 | 推理强化 + MLA |
| `kimi-k2` | DeepSeek (MLA) | MoE | MLA: d_c=512, d_rope=64 | DeepSeek 架构变体 |
```

**GGUF 权重名映射**:
```rust
// DeepSeek MLA 权重名 → TensorRole 映射
"kv_b_proj"   → TensorRole::MlaKvCompress   // W_DKV
"k_b_proj"    → TensorRole::MlaKeyAbsorb     // W_UK
"v_b_proj"    → TensorRole::MlaValueAbsorb   // W_UV
"k_pe_proj"   → TensorRole::MlaRopeKey       // W_KR (部分 GGUF 无此名,从 kv_b_proj 推导)
```

**关键文件**:
- `gllm/src/arch/registry.rs`: 架构注册表
- `gllm/src/loader/gguf/reader.rs`: GGUF 权重名 → TensorRole 映射
- `gllm/SPEC/11-MODELS.md`: 模型描述更新

### REQ-MLA-009: MLA 数值对齐验证

MLA 路径与标准 MHA 路径的数值对齐验证。

**设计**:
- 给定相同输入 X 和权重 W_DKV/W_UK/W_UV/W_KR:
  1. **标准路径**: K = c_KV · W_UK, V = c_KV · W_UV, Attention(Q, K, V)
  2. **Absorbed 路径**: Q_absorbed = Q · W_UK^T, MLA_Attention(Q_absorbed, c_KV)
  3. **数值一致性**: 两条路径输出余弦相似度 ≥ 0.999
- 验证 KV cache 压缩: 存储 c_KV + k_pe (576 维) 后读回还原 K/V，与直接计算 K/V 的误差 < 1e-5
- 验证 RoPE merge: 替换 c_KV 后 d_rope 维后，attention score 与标准 RoPE K 的误差 < 1e-4

**关键文件**:
- `gllm/tests/` 或 `gllm-kernels/tests/`: E2E 数值验证测试

## 实施顺序

```
REQ-MLA-001 (MlaConfig) ───→ REQ-MLA-002 (特征检测) ───→ REQ-MLA-003 (Absorbed graph)
         │                                                    │
         │                                                    ├──→ REQ-MLA-004 (Un-absorbed)
         │                                                    │
         ├──→ REQ-MLA-005 (PagedKV 适配) ←───────────────────┘
         │
         └──→ REQ-MLA-006 (OpKind) ──→ REQ-MLA-007 (TraceOp) ──→ REQ-MLA-009 (数值验证)
                                                         │
                                       REQ-MLA-008 (模型注册) ←─┘
```

1. **REQ-MLA-001**: MlaConfig — 配置基础
2. **REQ-MLA-002**: 特征检测 — auto_graph 识别 MLA 模型
3. **REQ-MLA-006**: OpKind — 算子注册
4. **REQ-MLA-007**: TraceOp — 自动指令选择
5. **REQ-MLA-003**: Absorbed graph — 核心图构建
6. **REQ-MLA-005**: PagedKV 适配 — KV cache 维度
7. **REQ-MLA-004**: Un-absorbed — 短序列优化
8. **REQ-MLA-008**: 模型注册 — 端到端接入
9. **REQ-MLA-009**: 数值验证 — E2E 对齐

## 验证

```bash
# 编译检查
cd ../gllm && cargo check
cd ../gllm-kernels && cargo check

# 单元测试
cargo test --lib

# E2E MLA 测试 (需要 DeepSeek V3 GGUF)
cargo test --test test_e2e_generator -- --test-threads=1

# 数值对齐
# Absorbed vs Un-absorbed 输出余弦相似度 ≥ 0.999
# c_KV 还原 K/V 误差 < 1e-5
# RoPE merge 误差 < 1e-4
```

## 性能预期

| 场景 | 标准 MHA | MLA Absorbed | 加速比 |
|------|---------|-------------|--------|
| Decode KV cache 带宽 | 65536 B/token | 1152 B/token | **56.9×** |
| Decode Attention score | Q × K^T (d=128) | Q_abs × c_KV^T (d_c=512) | 4× FLOPs ↑ |
| Prefill 短序列 | FlashAttention 大 tile | Un-absorbed (等同) | 1× |
| Prefill 长序列 | KV 加载瓶颈 | Absorbed 56.9× 带宽节省 | 显著 |

**关键洞察**: Decode 阶段的瓶颈是 KV cache 带宽（memory-bound），56.9× 压缩直接消除瓶颈。Absorbed 路径的额外 4× score FLOPs 远小于节省的带宽开销。
