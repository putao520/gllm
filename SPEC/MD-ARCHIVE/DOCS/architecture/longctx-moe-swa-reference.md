# 长上下文 / 稀疏注意力 / MoE 共享专家技术参考

> **用途**: 本文档为 gllm JIT codegen 提供 YaRN RoPE Scaling、Sliding Window Attention、
> MoE Shared Experts 三项关键技术的底层实现细节。
> 实现规范见 `SPEC/23-QUANT-CODEGEN-ALGO.md` 和 `SPEC/04-OPERATORS.md`。

## §1 YaRN RoPE Scaling

### §1.1 数学公式

核心思想: 基于波长的分频处理 — 高频外推、低频内插、中间渐变。

**频域划分基准**:
- 维度波长: `λ_i = 2π × 10000^(2i/d)`
- 缩放系数: `s = L_test / L_train`

**三分频域**:

| 区域 | 条件 | 混合系数 γ_i | 角速度处理 |
|------|------|-------------|-----------|
| 低频区 | `r(i) < α` | `γ_i = 0` | PI 线性内插: `θ_i / s` |
| 高频区 | `r(i) > β` | `γ_i = 1` | 保持外推: `θ_i` 不变 |
| 渐变区 | `α ≤ r(i) ≤ β` | `γ_i = (r(i) - α) / (β - α)` | 线性平滑过渡 |

**最终角速度**:
```
θ_i' = (1 - γ_i) × (θ_i / s) + γ_i × θ_i
```

**熵稳定因子 (mscale)**:
```
mscale = 0.1 × ln(s) + 1
```
在向量旋转时直接作为系数乘在 Query 和 Key 上, 维持 Attention 熵稳定。

### §1.2 JIT 编译策略

| 优化 | 实现 |
|------|------|
| 常量折叠 | 编译期折叠 `s, α, β, mscale` 等所有已知超参到立即数 |
| 分支消除 | 静态展开每个维度的 θ_i', 移除运行时 if-else |
| 流水融合 | rotate_half + mscale 缩放在单算子寄存器内完成 (零额外 load/store) |

### §1.3 partial (p-RoPE) 与 YaRN 组合

Gemma 4 使用 partial=0.25 (仅 25% 维度施加 RoPE, 其余 75% 不旋转)。
组合公式: 对维度 `d_i`:
```
if d_i < partial × head_dim:
    θ_i'' = YaRN_scaled(θ_i)    // 应用 YaRN 公式
else:
    θ_i'' = 1.0                   // 不旋转
```

编译期已知 `partial`, 可完全静态化每个维度的处理方式。

### §1.4 gllm 代码映射

- `OpKind::RoPE { theta, partial, rope_scaling: Some(RopeScaling::Yarn { factor, beta_fast, beta_slow, original_max_position }) }`
- JIT codegen: 对每个 `dim_idx` 计算 `θ_i'`, 编译期 bake 进 cos/sin 表或立即数

## §2 Sliding Window Attention (SWA)

### §2.1 核心策略: KV Range Limitation (禁止 Tensor Mask)

| 策略 | 做法 | 复杂度 | 评估 |
|------|------|--------|------|
| Tensor Mask | 全量 Q×K^T 计算, Softmax 前超范围 Score 置 -∞ | O(N²) | ❌ 无效空算, 浪费带宽 |
| **KV Range Limitation** | 动态裁剪 FlashAttention 块级循环区间 | O(N×W) | ✅ 完全跳过窗口外 KV 加载 |

### §2.2 KV Range Limitation 实现

对给定 Query Tile 区间 `[Q_start, Q_end]`, Key Block 循环索引限制为:

```
KV_loop_start = max(0, floor((Q_start - W) / B_kv))
KV_loop_end   = ceil(Q_end / B_kv)

其中:
  W     = sliding_window 窗口大小
  B_kv  = KV block size (Tile 大小)
```

FlashAttention 外层循环边界直接改写, 不加载、不计算窗口外 KV。

### §2.3 Decoding 阶段环形缓冲区

解码阶段 KV Cache 物理显存固定为 W:

```
写入地址: kv_ptr = base + (token_idx % W) × stride
```

**位置解耦**: 地址循环复用, 必须将绝对位置 `token_idx` 作为元数据传入内核,
用于计算正确的 RoPE 相对距离。

### §2.4 GPT-OSS Sliding/Full 交替层

GPT-OSS 每 N 层交替使用 sliding (W=4096) 和 full attention:
- Sliding 层: KV Range Limitation 生效, O(N×W) 复杂度
- Full 层: 循环边界不限制, 标准 O(N²) FlashAttention
- JIT 策略: 编译期根据 `model_adapter.rs` 的 `sliding_window: Option<usize>` 为每层生成不同的循环边界

## §3 MoE Shared Experts JIT 路径

### §3.1 数据流

```
Y = FFN_shared(X) + Σ(i ∈ TopK) g_i × FFN_i(X)
```

核心冲突:
- Shared Expert: 接收**全量** Token, Dense GEMM
- Routed Experts: 接收**部分被分流** Token, Sparse GEMM

### §3.2 方案 A: Grouped GEMM 融合 (推荐)

```
[输入 Token M 个]
    ↓ 分流编排
    ├── Group 0: 共享专家 (全量 M)     ← 伪装为 Expert 0
    ├── Group 1: 路由专家 1 (稀疏)
    ├── Group 2: 路由专家 2 (稀疏)
    └── ...
    ↓ 单内核一次发射
Grouped GEMM → 硬件 SM 自动切片并发
```

实现:
- 编译期将全量 Token 索引无条件硬编码指派给 Group 0
- 路由专家分发至 Group 1..E
- 发射单个 Grouped GEMM 内核, 由硬件调度器自动切片 SM 并发
- 规避多次 Kernel 启动的流阻塞

### §3.3 方案 B: 双流交错发射

```
[输入 Token M 个]
    ├── 默认 CUDA 流 ────────→ Shared FFN Kernel (Dense)
    └── 异步高优先级流 ───→ MoE Router + Grouped GEMM (Sparse)
    ↓ 流级同步
    ↓ Fused Epilogue Aggregation
```

利用硬件 SM 级并发能力 (Concurrent Kernels), 稠密与稀疏矩阵乘法空间重叠。

### §3.4 Fused Epilogue Aggregation (两种方案通用)

```
Y_final = Y_shared + Y_routed

累加操作直接融合进 JIT 算子的 Epilogue 写回阶段:
  acc += shared_out × 1.0  (shared expert 恒定权重 1.0)
  acc += routed_out × g_i  (routed expert 带 gate 权重)
```

彻底消除一次 Global Memory 的中间读写 Roundtrip。

### §3.5 gllm 实现指引

| 组件 | 当前状态 | 需要做的 |
|------|---------|---------|
| `MoEDispatchPacked` | 支持 routed experts | 扩展支持 shared expert (Group 0 全量 Token) |
| `model_adapter.rs` | `moe_config` | 添加 `num_shared_experts` 字段 |
| `OpKind::MoEGate` / `MoERouter` | 不变 | Shared expert 不经过 Router |
| GEMM codegen | 标准 GEMM | Grouped GEMM 支持 (多 expert 共享一次 kernel launch) |

## §4 参考

- YaRN 论文: "YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al., 2023)
- Sliding Window Attention: Mistral paper "A Trick of Attention" (Child et al.)
- DeepSeek V3: "DeepSeek-V3 Technical Report" — Shared Experts + Multi-head Latent Attention
- Kimi-K2: MoE with shared experts architecture
- FlashAttention-2/3: Tiled attention with KV range limitation
- CUTLASS Grouped GEMM: Multi-expert batched matrix multiplication
