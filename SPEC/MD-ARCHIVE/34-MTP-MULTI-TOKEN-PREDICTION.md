# Multi-Token Prediction (MTP) 模型内置多 Token 预测 (ARCH-MTP)

## 定位

MTP 是模型级推测解码机制。训练时使用 MTP loss 的模型（DeepSeek V3、Qwen3）在推理时一次前向输出 K 个未来 token 的候选，相当于内置 draft model，零额外权重开销。MTP 候选 token 经过标准推测解码验证流程后提交。

> **SSOT**: 本 SPEC 定义 MTP 的完整实现规范。推测解码通用框架见 `SPEC/05-OPTIMIZATIONS.md §2.8`。
> 开发清单见 `SPEC/DOCS/scheduling/thinking-temperature-speculative-plan.md S3`。

## 前置原则

- **ARCH-MTP-BUILTIN**: MTP 是模型内置能力，不需要额外 draft model。MTP projections 权重来自模型文件，pack 进 weight_blob
- **ARCH-MTP-JIT**: MTP 投影和 argmax 在 JIT mega-kernel 内部完成（OutputModeDispatch (SPEC/39 §1.3.3) 内），不经过 Rust。一次 CALL 输出 main token + K 个 MTP 候选
- **ARCH-MTP-VERIFY**: MTP 候选必须经过 Verify 阶段验证，不直接提交。Verify 使用全量模型 forward
- **ARCH-MTP-EMA**: 接受率通过 EMA 追踪，连续低接受率时自动回退到标准解码

## 架构

```
                    MTP 推测解码流程
┌──────────────────────────────────────────────────────┐
│ Mega-Kernel Forward                                  │
│                                                      │
│  prefill 层循环融合组: 标准前向 (embed → L×(attn+ffn) → norm)│
│  MTP 融合组（per-depth 投影 + argmax 循环）                      │
│    ├── lm_head → main token logits → argmax          │
│    └── MTP 候选生成 (JIT 内联, MTP 融合组子阶段) │
│         for k in 0..depth:                           │
│           hidden_proj = hidden · W_mtp[k]^T          │
│           argmax(hidden_proj) → candidate_token[k]   │
│                                                      │
│  Output: [main_token, candidate_0, ..., candidate_K] │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│ Verify Phase (全量模型)                               │
│                                                      │
│  1. 将 K 个候选 token 拼接到序列                     │
│  2. 全量模型 forward → 每 token 的 logits            │
│  3. 逐 token 验证:                                   │
│     if candidate[k] == argmax(verify_logits[k]):     │
│       accept → commit token                          │
│     else:                                            │
│       reject → 使用 verify_logits 重新采样            │
│       break (后续候选全部丢弃)                        │
│  4. KV cache: 被拒绝的 token 的 KV 条目回滚          │
└──────────────────────────────────────────────────────┘
```

## 维度定义

| 符号 | 含义 | DeepSeek V3 值 | Qwen3 值 |
|------|------|---------------|----------|
| `K` (depth) | MTP 预测深度 | 2-4 | 1-2 |
| `d_model` | 隐藏层维度 | 7168 | 4096/5120 |
| `V` | 词表大小 | 129280 | 151936 |
| `W_mtp[k]` | 第 k 层投影权重 | `[d_model, V]` | `[d_model, V]` |

### MTP 投影权重布局

```
Weight Blob:
┌──────────────────────────────────┐
│ ... (标准模型权重)                │
├──────────────────────────────────┤
│ lm_head.weight [d_model, V]      │  ← main token
├──────────────────────────────────┤
│ W_mtp[0] [d_model, V]            │  ← MTP depth 0
│ W_mtp[1] [d_model, V]            │  ← MTP depth 1
│ ...                               │
│ W_mtp[K-1] [d_model, V]          │  ← MTP depth K-1
└──────────────────────────────────┘

每个 depth 的投影: logits[k] = hidden · W_mtp[k]^T
即 hidden (1×d_model) × W_mtp[k]^T (d_model × V) → logits (1×V)
```

## 已实现组件

| 组件 | 文件 | 状态 |
|------|------|------|
| `MtpConfig` (depth, vocab_size, hidden_size) | `src/speculative/mtp.rs` | ✅ |
| `MtpHead` (projections 权重) | `src/speculative/mtp.rs` | ✅ |
| `mtp_draft()` 标量参考实现 | `src/speculative/mtp.rs` | ✅ |
| `mtp_candidates()` argmax 提取 | `src/speculative/mtp.rs` | ✅ |
| `MtpKernelConfig` (JIT ABI) | `gllm-kernels/mega_kernel_abi.rs` | ✅ |
| `TraceOp::MtpDraft` | `gllm-kernels/vm/trace.rs` | ✅ |
| `auto_select` MtpDraft dispatch | `gllm-kernels/vm/auto_select.rs` | ✅ |
| `emit_mtp_draft_inline()` JIT codegen | `gllm-kernels/vm/mega_kernel_emit.rs` | ✅ |
| `emit_mtp_gemv()` per-depth GEMV | `gllm-kernels/vm/mega_kernel_emit.rs` | ✅ |
| `SpecDecodingMode::Mtp` 模式枚举 | `src/speculative/engine.rs` | ✅ |
| `MtpEmaTracker` 接受率追踪 | `src/speculative/adapter.rs` | ✅ |
| MTP 融合组（per-depth 投影 + argmax 循环） MTP 路由 | `gllm-kernels/vm/mega_kernel_emit.rs` | ✅ |

## REQ 清单

### REQ-MTP-001: MTP 模型权重注册

将 MTP 投影权重从模型文件加载并映射到 weight_blob。

**设计**:

**GGUF 权重名映射**:
```
DeepSeek V3 MTP 权重命名:
  "model.layers.{N}.mtp_proj.{k}.weight"  → W_mtp[k]
  或 "model.mtp_head.{k}.weight"           → W_mtp[k] (非分层)

Qwen3 MTP 权重命名:
  "model.layers.{N}.mtp_proj.{k}.weight"  → W_mtp[k]
  或 "model.mtp.{k}.weight"                → W_mtp[k]
```

**Weight Blob 打包**:
- MTP 权重紧跟 `lm_head.weight` 之后
- `MtpKernelConfig.depth` 和 `MtpKernelConfig.hidden_size`/`vocab_size` 从 `MtpConfig` 传入
- JIT codegen 从 weight_blob 的 `lm_head_offset + lm_head_bytes` 处读取 MTP 权重基址

**GGUF 元数据**:
```
deepseek.mtp_depth: uint32  → MtpConfig.depth
qwen3.mtp_depth: uint32     → MtpConfig.depth
```

**关键文件**:
- `gllm/src/loader/gguf/reader.rs`: GGUF MTP 元数据解析
- `gllm/src/weight_loader.rs`: MTP 权重加载到 weight_blob
- `gllm/src/arch/auto_graph.rs`: MTP 权重名 → TensorRole 映射

### REQ-MTP-002: MTP ↔ Mega-Kernel 编排集成 (SPEC 32 联动)

MTP 候选生成嵌入 SPEC 32 的 decode loop，通过 MegaKernelBusinessConfig.mtp_config 控制。

**与 SPEC 32 的集成点**:

```
SPEC 32 §1 Decode Loop:
  prefill 层循环融合组: Forward pass → hidden_state
  MTP 融合组（per-depth 投影 + argmax 循环） (现有 4 模式 + MTP 扩展)
    ├── LABEL_GENERATE: main_token + sample (现有)
    └── LABEL_MTP_GENERATE (新增，当 mtp_config.is_some()):
         main_token = argmax(logits)
         for k in 0..depth:
           candidate[k] = emit_mtp_draft(k, hidden_state)  // REQ-MTP-003
         写入 output: [main_token, cand_0, ..., cand_K]
  decode 层循环融合组（MTP verify 路径）: Sample (MTP 模式跳过——已在 MTP 融合组完成 argmax)
  CheckStopCondition 融合组（MTP verify + KV rollback）: Stream output (MTP 模式只流式输出 main_token)
  epilogue: Compact + Refill
```

**MegaKernelBusinessConfig 扩展** (已在 `mega_kernel_abi.rs` 实现):
```rust
pub struct MegaKernelBusinessConfig {
    // ...现有字段...
    pub mtp_config: Option<MtpKernelConfig>,  // ✅ 已实现
}

pub struct MtpKernelConfig {  // ✅ 已实现
    pub depth: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}
```

**MTP 在 decode loop 中的行为**:
- 每次 decode step 输出 `1 + K` 个 token（main + K 候选）
- Streaming output (SPEC 32 §4) 只流式 main_token（候选不立即提交）
- Verify 阶段在 Mega-Kernel 返回后由 Rust 端执行（需要全量模型 forward，不在单次 CALL 内）
- 验证通过后的 token 提交 + KV 回滚由 Rust 端驱动
- MTP 不影响 SPEC 32 的 SM 分区和 compact/refill 逻辑

### REQ-MTP-002: MTP Executor 集成路径

将 MTP 推测解码接入 generate 循环。

**设计**:

当前 generate 循环:
```
loop {
  main_token = mega_kernel_forward(input_ids)
  if main_token == EOS: break
  input_ids = [main_token]
}
```

MTP 模式 generate 循环:
```
loop {
  [main_token, cand_0, ..., cand_K] = mega_kernel_forward(input_ids, mtp_enabled=true)
  // main_token 已确认, 候选待验证
  if main_token == EOS: break

  // Verify phase: 全量 forward 验证候选
  verified = verify_candidates([cand_0, ..., cand_K])

  // 提交已验证的 token, 回滚被拒绝的 KV
  commit_count = verified.accepted_count
  rollback_kv(verified.rejected_positions)

  if verified.last_is_eos: break
  input_ids = [verified.last_token]
}
```

**关键约束**:
- MTP 候选验证必须走全量模型 forward（不是浅层变体）
- Verify 阶段 K 个候选 token 拼成 batch 一次 forward（batch_size=K）
- KV cache 回滚: 被拒绝的 token 的 KV 条目标记为无效，page ref_count 减 1
- **ARCH-MTP-VERIFY 保证**: Verify 输出与标准逐 token 解码的 logits 完全一致

**关键文件**:
- `gllm/src/engine/executor.rs`: generate 循环 MTP 路径
- `gllm/src/speculative/engine.rs`: `SpecDecodingState` MTP 模式执行
- `gllm/src/speculative/verify.rs`: MTP 候选验证逻辑

### REQ-MTP-003: MTP Mega-Kernel MTP 融合组（per-depth 投影 + argmax 循环） 完善

完善 Mega-Kernel 内 MTP 候选生成（MTP 融合组（per-depth 投影 + argmax 循环））的完整实现。

**设计**:

MTP 融合组（per-depth 投影 + argmax 循环） 当前实现（`mega_kernel_emit.rs`）:
1. 从 weight_blob 读取 MTP 权重基址
2. 获取 final_normed tensor（forward 最后一层 hidden state）
3. 对每个 depth 执行 `emit_mtp_gemv()`:
   - `hidden_state @ W_mtp[k]^T` → logits
   - argmax(logits) → candidate_token[k]
4. 写入 output buffer: `[main_token, cand_0, ..., cand_K]`

**需要完善**:
- **GPU codegen**: `emit_mtp_gemv()` 当前只有 x86_64 实现，需要 GPU (PTX/HIP/MSL) 路径
- **SymDim 穿透**: hidden_size 和 vocab_size 应从编译时常量改为 SymDim 参数化
- **温度采样**: MTP 候选当前走 argmax（top-1），需要支持 temperature-based 采样（与 main token 采样策略一致）
- **per-depth 独立投影**: DeepSeek V3 的 MTP 每层有独立的 embedding norm + projection，不是简单共享 hidden_state

**DeepSeek V3 MTP 完整架构**:
```
MTP depth k:
  1. embed_norm_k = RMSNorm(hidden_from_layer_k)
  2. projected_k = embed_norm_k @ W_mtp[k]^T
  3. logits_k = projected_k (shape [1, V])
  4. candidate_k = sample(logits_k)
```

当前实现假设所有 depth 共享同一个 hidden_state（最后一层输出）。DeepSeek V3 实际上每层 MTP 有独立的 RMSNorm。需要为 per-depth 独立处理。

**关键文件**:
- `gllm-kernels/src/compiler/codegen/vm/mega_kernel_emit.rs`: MTP 融合组（per-depth 投影 + argmax 循环） 完善
- `gllm-kernels/src/compiler/codegen/vm/auto_select.rs`: MtpDraft GPU 路径
- `gllm-kernels/src/compiler/codegen/gpu_ir/`: GPU MTP GEMV

### REQ-MTP-004: MTP KV Cache 回滚

MTP 候选验证失败时回滚 KV cache 条目。

**设计**:
```
Verify 结果: accepted_count = N (0 ≤ N ≤ K)

KV cache 状态:
  Before verify: KV 包含 main_token + K 个候选 token 的条目
  After verify:
    - main_token + cand_0 ~ cand_{N-1}: 保留 (已验证)
    - cand_N ~ cand_{K-1}: 回滚 (被拒绝)

回滚操作:
  1. 将被拒绝 token 的 KV page 引用计数减 1
  2. 如果 page ref_count == 0, 释放 page 到 free pool
  3. 更新 seq 的 kv_len = original_kv_len + 1 (main) + N (accepted)
  4. 被拒绝 token 从 output buffer 中移除
```

**关键约束**:
- KV 回滚必须在 Verify 完成后立即执行，不能延迟
- 回滚是 PagedAttention 级别的操作（page ref_count 原子减）
- 不需要物理清除 KV 数据（只需更新逻辑长度）
- **正确性保证**: 回滚后的 KV cache 状态与"从未生成被拒绝 token"完全一致

**关键文件**:
- `gllm/src/scheduler/paged_scheduler.rs`: KV page 回滚
- `gllm/src/kv_cache.rs`: KV cache 逻辑长度更新
- `gllm/src/speculative/verify.rs`: `KvCommitInstruction` 生成

### REQ-MTP-005: MTP 自适应回退

基于 EMA 接受率的 MTP 自适应启用/禁用。

**设计**:
```
MTP 决策状态机:
  ACTIVE → (连续 3 轮 acceptance_rate < 0.3) → DISABLED
  DISABLED → (连续 5 轮 标准解码 无压力) → ACTIVE

EMA 追踪 (MtpEmaTracker):
  - per-depth 接受率: ema[k] = α × accept[k] + (1-α) × ema[k]
  - 默认参数（可通过 MtpConfig 运行时配置）:
    - α = 0.1 (平滑因子)
    - disable_threshold = 0.3 (连续 N 轮低于此值 → 禁用 MTP)
    - enable_threshold = 0.5 (标准解码连续 M 轮后尝试重新启用)
    - disable_patience = 3 (连续低接受率轮数)
    - enable_patience = 5 (标准解码稳定轮数)
  - 整体接受率: avg(ema[0..K-1])

自适应 depth 调节:
  - 如果 ema[0] > 0.8 且 ema[1] > 0.6: depth = 3 (激进)
  - 如果 ema[0] > 0.5 且 ema[1] > 0.3: depth = 2 (适中)
  - 如果 ema[0] < 0.3: depth = 0 (禁用)
  - depth 调整不需要重编译（MTP weights 全部打包，运行时选择使用前 N 个）
```

**关键文件**:
- `gllm/src/speculative/adapter.rs`: `MtpEmaTracker` 完善
- `gllm/src/speculative/engine.rs`: 自适应决策逻辑
- `gllm/src/engine/executor.rs`: depth 运行时传递

### REQ-MTP-006: MTP 模型注册与 E2E 验证

将 MTP 支持注册到模型架构系统，覆盖 DeepSeek V3 和 Qwen3。

**设计**:

**SPEC/11-MODELS.md 更新**:
```
| `deepseek-v3` | DeepSeek (MLA+MTP) | 671B MoE | MTP: depth=2 | 旗舰 MoE |
| `qwen3` | Qwen3 | 0.6B-235B | MTP: depth=1 | 通用标杆 |
```

**模型检测逻辑**:
```rust
// GGUF: 检测 MTP 权重
let has_mtp = weight_names.iter().any(|n| n.contains("mtp_proj") || n.contains("mtp_head"));
if has_mtp {
    let depth = count_mtp_weights(weight_names);
    mtp_config = Some(MtpConfig { depth, vocab_size, hidden_size });
}
```

**E2E 验证**:
1. 标准 decode: 逐 token 生成 → baseline output
2. MTP decode: 启用 MTP → 验证 main token 与 baseline 一致
3. 接受率: MTP 候选的接受率 > 0 (至少部分正确)
4. 输出一致性: MTP 模式最终输出与标准模式完全一致（已验证 token 相同）
5. KV 一致性: MTP 模式回滚后的 KV cache 与标准模式等价

**关键文件**:
- `gllm/src/arch/registry.rs`: MTP 模型注册
- `gllm/SPEC/11-MODELS.md`: 模型描述更新
- `gllm/tests/`: E2E MTP 测试

## 实施顺序

```
REQ-MTP-001 (权重注册) ──→ REQ-MTP-003 (MTP 融合组（per-depth 投影 + argmax 循环） 完善)
         │                          │
         │                          ├──→ REQ-MTP-002 (Executor 集成)
         │                          │           │
         │                          │           └──→ REQ-MTP-004 (KV 回滚)
         │                          │
         └──→ REQ-MTP-005 (自适应回退)
                          │
         REQ-MTP-006 (模型注册 + E2E) ←───┘
```

1. **REQ-MTP-001**: 权重注册 — MTP 权重从模型文件加载
2. **REQ-MTP-003**: MTP 融合组（per-depth 投影 + argmax 循环） — JIT MTP 候选生成完善（GPU 路径、per-depth 独立处理）
3. **REQ-MTP-002**: Executor 集成 — generate 循环 MTP 路径
4. **REQ-MTP-004**: KV 回滚 — 候选验证失败的 KV 处理
5. **REQ-MTP-005**: 自适应回退 — EMA 接受率追踪
6. **REQ-MTP-006**: 模型注册 + E2E 验证

## 性能预期

| 场景 | 标准 Decode | MTP Decode | 加速比 |
|------|-----------|-----------|--------|
| DeepSeek V3 (K=2, acceptance=60%) | 1 token/step | 1 + 0.6×2 = 2.2 tokens/step | **2.2×** |
| DeepSeek V3 (K=4, acceptance=40%) | 1 token/step | 1 + 0.4×4 = 2.6 tokens/step | **2.6×** |
| Qwen3 (K=1, acceptance=70%) | 1 token/step | 1 + 0.7 = 1.7 tokens/step | **1.7×** |
| 极端低接受率 (< 30%) | 1 token/step | ≈1.0 (回退到标准) | 1.0× |

**关键**: MTP 的 ROI 取决于接受率。EMA 自适应确保低接受率时自动回退，零开销损失。DeepSeek V3 论文报告 MTP acceptance rate ≈ 60-80%，预期 2-3× 加速。

## 验证

```bash
# 编译检查
cd ../gllm && cargo check
cd ../gllm-kernels && cargo check

# 单元测试
cargo test --lib

# MTP 组件测试
cargo test -- mtp
cargo test -- speculative

# E2E MTP 测试 (需要 DeepSeek V3 / Qwen3 GGUF)
cargo test --test test_e2e_generator -- --test-threads=1

# 验证项:
# 1. MTP main token 与标准 decode 一致
# 2. MTP 候选接受率 > 0
# 3. KV 回滚后 cache 状态正确
# 4. 自适应回退在低接受率时触发
# 5. GPU 路径 MTP 输出与 CPU 路径一致
```
