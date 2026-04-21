# Guardrail SDK — In-Flight 安全 Veto 探针协议

> **执行模型**: Hook/Callback 在 mega-kernel 架构下通过 JIT 内嵌条件 JMP 实现（详见 `08-EXECUTOR.md` §4.1.5）。无 hook 注册时不生成跳转代码。Hook 通信通过共享内存，不经过 Rust 函数调用。


> **SSOT**: 本文档定义 gllm Guardrail SDK 的技术协议, 提供推理**前向中途**插入安全分类线性探针 + 多档策略响应(HaltAndVeto / LogOnly / SampleDowngrade)的机制。
>
> **需求 SSOT**: `SPEC/01-REQUIREMENTS.md §14` REQ-GR-001..005
>
> **API 定义 SSOT**: `SPEC/04-API-DESIGN.md §3.9`
>
> **遵守铁律**: ARCH-FULL-JIT、ARCH-CPU-GPU-UNIFIED、NO_SILENT_FALLBACK、NO_ISLAND_MODULE、CallbackChain 正交性

---

## §1 动机与定位

### 1.1 为什么需要 in-flight 安全探针

主流 LLM 安全方案有三类:

1. **Prompt-level filter**: 输入敏感词黑名单 — 粒度粗, 易被改写绕过, 且已损耗用户上下文。
2. **Output-level moderation**: 生成后再分类一次 — 延迟最差(完整生成 + 再跑分类模型), 且浪费已生成 tokens 的算力。
3. **Fine-tuned 安全头**: RLHF / DPO 把安全策略烘进权重 — 模型升级即失效, 策略不可热切换。

Guardrail SDK 采用**第四条路径**: 在**同一前向过程**中, 用一份独立训练的小线性探针对**中间层 hidden state** 做分类, 探针触发立即影响当前推理的策略 (veto / downgrade / log)。

### 1.2 与 Semantic Gatekeeper / Head Routing 的正交关系

| 维度 | Semantic Gatekeeper | Head Routing | Guardrail |
|------|---------------------|--------------|-----------|
| 作用 | 注入知识残差 (InjectHidden) | 读最终 hidden 投 lm_head | 读中间 hidden 做二元分类 + veto |
| 优先级 | 90 (pre_node) | 无 Callback, 直接读 forward 结果 | 40 (post_node) |
| 修改 hidden? | ✅ 是 (InjectHidden) | ❌ 只读 | ❌ 只读 + 可 ExitEarly |
| 粒度 | 每次 decode step 每检测层 | 每次 API 调用一次 | 每次 forward 一次 |
| 生效开销 | 检索 + 编码 + 残差加 | 零 (最终层读出) | 单次矩阵乘 (hidden · w + b) |

三者**可共存**。Guardrail 在 post_node 执行, SG 在 pre_node 执行, HR 读最终结果 —— 互不阻塞。

---

## §2 架构总览

### 2.1 数据流

```
 ┌─ Client::classify_binary / encode_intent / generate ─────────────────────┐
 │                                                                          │
 │  1. tokens = tokenizer.encode(prompt)                                     │
 │                                                                          │
 │  2. Client::build_guardrail_chain():                                      │
 │     ├─ 遍历 Client.guardrails (HashMap<id, registration>)                  │
 │     ├─ 为每个 registration 构造 GuardrailProbeCallback { target_layer,    │
 │     │    weights, policy, hidden_size, shared (Arc) }                    │
 │     ├─ 加上用户自定 Extra callbacks (MidLayerEncodeCallback for Intent)    │
 │     └─ 返回 CallbackChain (按 priority 排序)                              │
 │                                                                          │
 │  3. forward_config.callback_chain_ptr = &mut chain                        │
 │                                                                          │
 │  4. backend.score_tokens_forward_gpu_pure(...)                            │
 │     ↓                                                                    │
 │  5. FusedGraphExecutor::run_with_callbacks():                             │
 │     for each compiled node:                                               │
 │       ... execute JIT kernel ...                                          │
 │       if layer_idx == target_layer:                                       │
 │         score = sigmoid(W · h_last + b)                                   │
 │         shared.record_score(score)                                        │
 │         match policy:                                                     │
 │           HaltAndVeto { thr } if score > thr → ExitEarly + veto flag      │
 │           LogOnly → Continue                                              │
 │           SampleDowngrade { min_temp } → record + Continue                │
 │                                                                          │
 │  6. Client 读 logits: 若为空 → Err("guardrail vetoed")                     │
 │       否则 softmax 归一化返回分数                                          │
 │  7. 用户通过 attachment.is_vetoed() / last_score() 查询状态               │
 └──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心不变量

1. **零权重重载**: attach_guardrail 仅在 Client 内部 HashMap 新增条目, 不触发 JIT 重编译或权重重装。
2. **ARCH-FULL-JIT 合规**: 前向仍走 JIT 管线 (Scalar → SymExec → IR → Lowering), 探针 sigmoid 只是 callback 层的轻量计算, 不绕开 JIT。
3. **NO_SILENT_FALLBACK**: 权重加载失败 / 形状不匹配 / anchor 越界 → 显式 `GuardrailError::...` (包装为 `ClientError::RuntimeError`)。
4. **NO_ISLAND_MODULE**: `GuardrailProbeCallback` 必须通过 `Client::build_guardrail_chain` 进入真实前向路径, 且 `classify_binary` / `encode_intent` 在构造前向时会消费注册表。
5. **状态共享**: 每个 attachment 持有 `Arc<GuardrailSharedState>`, 回调侧写入 score/veto, API 侧只读 — 双向隔离, 无锁竞争。

---

## §3 Client API

### 3.1 API 表面

```rust
use gllm::{Client, GuardProbe, GuardProbeWeights, LayerAnchor, SafetyPolicy};

let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")?;

// 从 safetensors 加载探针
let attachment = client.attach_guardrail(
    GuardProbe::from_safetensors("toxicity_probe.safetensors"),
    LayerAnchor::Relative(0.5),
    SafetyPolicy::HaltAndVeto { threshold: 0.95 },
)?;

// 或 inline 权重 (测试 / 程序生成的探针)
let attachment = client.attach_guardrail_inline(
    GuardProbeWeights { weight: vec![0.0; 576], bias: 0.0 },
    LayerAnchor::Relative(0.5),
    SafetyPolicy::LogOnly,
)?;

let _score = client.classify_binary("question", ...)?;

if attachment.is_vetoed() {
    eprintln!("Blocked: {:?}", attachment.last_veto_reason());
}
println!("score: {:?}", attachment.last_score());

client.detach_guardrail(attachment.id)?;
```

### 3.2 方法签名

```rust
impl Client {
    pub fn attach_guardrail(
        &self,
        probe: GuardProbe,
        anchor: LayerAnchor,
        policy: SafetyPolicy,
    ) -> Result<GuardrailAttachment, ClientError>;

    pub fn attach_guardrail_inline(
        &self,
        weights: GuardProbeWeights,
        anchor: LayerAnchor,
        policy: SafetyPolicy,
    ) -> Result<GuardrailAttachment, ClientError>;

    pub fn detach_guardrail(&self, id: u64) -> Result<(), ClientError>;
}
```

`attach_guardrail` 自动完成权重加载 + 解析 LayerAnchor + 校验 SafetyPolicy。

---

## §4 GuardProbe 与 GuardProbeWeights

### 4.1 GuardProbe 枚举

```rust
pub enum GuardProbe {
    /// 从本地 safetensors 文件加载探针权重。
    FromSafetensors { path: String },
}
```

未来扩展点: HuggingFace Hub 下载、内存字节加载等。**当前仅支持 safetensors 文件**。

### 4.2 Safetensors 格式约定

| Tensor 名称 (按优先级) | 形状 | 含义 |
|------------------------|------|------|
| `weight` / `classifier.weight` / `guard_probe.weight` | `[H]` / `[1, H]` / `[N, H]` | 线性分类器权重 (N≥1 时取第一行) |
| `bias` / `classifier.bias` / `guard_probe.bias` | `[1]` / `[N]` | 偏置 (缺失时默认 0.0) |

`H` 必须与模型 `hidden_size` 一致, 或小于 `hidden_size` (降维探针, 使用部分点积)。

### 4.3 GuardProbeWeights struct

```rust
pub struct GuardProbeWeights {
    pub weight: Vec<f32>,
    pub bias: f32,
}

impl GuardProbeWeights {
    pub fn input_dim(&self) -> usize;
    pub fn score(&self, hidden: &[f32]) -> f32; // sigmoid(w · h + b)
}
```

---

## §5 SafetyPolicy

```rust
pub enum SafetyPolicy {
    HaltAndVeto { threshold: f32 },       // score > threshold → veto + ExitEarly
    LogOnly,                              // 只 record_score, 不改变前向
    SampleDowngrade { min_temperature: f32 }, // 记录 min_temp, 由上层采样器响应
}
```

### 5.1 HaltAndVeto 语义

- `threshold` 必须有限且在 `[0.0, 1.0]` 之间, 否则 `GuardrailError::InvalidPolicy`。
- `score > threshold` 时:
  1. `shared.trigger_veto(reason)` 写入 veto 标志 + 原因字符串
  2. Callback 返回 `CallbackAction::ExitEarly { logits: Vec::new() }`
  3. `FusedGraphExecutor` 立即退出节点循环, 返回空 `outputs`
  4. `backend::score_tokens_forward_gpu_pure` 看到 logits 异常 → 返回 `Ok(vec![])`
  5. Client 层检查 `logits.is_empty()` → 返回 `ClientError::RuntimeError("guardrail vetoed ...")`

### 5.2 LogOnly 语义

- `shared.record_score(score)` 写入最近分数; 回调返回 `Continue`。
- 前向**不变**, 用户可后续查询 `attachment.last_score()`。

### 5.3 SampleDowngrade 语义

- `min_temperature` 必须有限且 > 0, 否则 `InvalidPolicy`。
- 回调返回 `Continue` (不中断前向), 但 `shared.record_downgrade(min_temperature)` 写入共享状态。
- 采样器 (generation loop) 可通过 `attachment.downgraded_temperature()` 读取, 对当次采样调低温度。
- **当前实现**: Client-level sampler 钩子尚未完全集成, SampleDowngrade 仅保证"记录"语义; 将来采样器消费路径落地后该字段自动影响生成。

---

## §6 LayerAnchor 复用

Guardrail 直接复用 `crate::head_routing::LayerAnchor` 定义:

```rust
pub enum LayerAnchor {
    Relative(f32),   // r ∈ [0.0, 1.0], 映射 round(r × (num_layers - 1))
    Absolute(usize), // 绝对层索引, 必须 < num_layers
}
```

解析函数 `guardrail::resolve_anchor(anchor, num_layers) -> Result<usize, GuardrailError>` 包装底层 `LayerAnchor::resolve`, 把 `HeadRoutingError::InvalidLayerAnchor` 转为 `GuardrailError::InvalidAnchor`。

**铁律**: **禁止**在 guardrail.rs 中重新定义 `LayerAnchor` enum, 必须复用 head_routing 定义。

---

## §7 Integration Trace (NO_ISLAND_MODULE 合规证据)

| 步骤 | 文件:函数:行号(开发后审计) |
|------|---------------------------|
| 用户 API | `src/client.rs::Client::attach_guardrail` / `attach_guardrail_inline` / `detach_guardrail` |
| 注册表 | `src/client.rs::Client::guardrails: Arc<Mutex<HashMap<u64, GuardrailRegistration>>>` |
| 构造 CallbackChain | `src/client.rs::Client::build_guardrail_chain` — 被 `classify_binary` / 未来 generate 调用 |
| 运行时触发 | `src/engine/callbacks/guardrail_probe.rs::GuardrailProbeCallback::post_node` |
| Executor 接入 | `src/engine/executor.rs::Executor::score_tokens_for_prompt_with_callbacks` 设置 `forward_config.callback_chain_ptr` |
| Backend 消费 | `src/compat/cpu_backend.rs::score_tokens_forward_gpu_pure` → `run_with_optional_callbacks` → `FusedGraphExecutor::run_with_callbacks` |
| 状态返回 | `GuardrailSharedState` (Arc) + `GuardrailAttachment` 方法 (`is_vetoed` / `last_score` / `last_veto_reason` / `downgraded_temperature`) |

审计命令:
```bash
# 非测试代码调用点
grep -rn "build_guardrail_chain\|GuardrailProbeCallback::new" src/ --include="*.rs" | grep -v "#\[cfg(test)\]" | grep -v "mod tests"
# 期待: 至少 1 条命中 (client.rs 真实路径)
```

---

## §8 E2E 验收 (REQ-GR-001..005)

### REQ-GR-001: 零探针不触发 veto

```
输入: SmolLM2-135M-Instruct, GuardProbeWeights { weight=[0...], bias=0 },
      HaltAndVeto { threshold: 0.99 }
操作: classify_binary(...)
验收:
  (a) 返回 Ok(f32 ∈ [0, 1]) (无 veto)
  (b) attachment.is_vetoed() == false
  (c) attachment.last_score() ≈ 0.5 (零探针 sigmoid(0) = 0.5)
  (d) actual_layer ∈ [0, num_layers)
```

### REQ-GR-002: HaltAndVeto 触发

```
输入: weight=[0...], bias=+20 (sigmoid(20) ≈ 1.0),
      HaltAndVeto { threshold: 0.5 }
验收:
  (a) classify_binary 返回 Err(guardrail vetoed)
  (b) attachment.is_vetoed() == true
  (c) last_veto_reason() 包含 "vetoed" / "score" 子串
  (d) last_score() > 0.99
```

### REQ-GR-003: LogOnly 不改变生成

```
输入: weight=[0...], bias=+100, LogOnly
基准: 无 guardrail 时 classify_binary 分数 = baseline
验收:
  (a) 加 guardrail 后 classify_binary 仍成功返回
  (b) 两次分数完全相等 (LogOnly 不影响前向, 浮点误差 < 1e-4)
  (c) attachment.is_vetoed() == false
  (d) last_score() > 0.99
```

### REQ-GR-004: SampleDowngrade 记录温度

```
输入: SampleDowngrade { min_temperature: 0.3 }
验收:
  (a) classify_binary 成功返回
  (b) attachment.downgraded_temperature() == Some(0.3)
  (c) attachment.is_vetoed() == false
```

### REQ-GR-005: attach / detach 生命周期

```
验收:
  (a) 多次 attach 返回单调递增 id
  (b) detach 未知 id 返回 Err
  (c) detach 后再次 detach 同一 id 返回 Err
  (d) detach 另一个 id 独立成功
```

---

## §9 与 SG / HR 的正交性

### 9.1 共存

同一 Client 可同时挂 SG + HR 召回 + Guardrail:

```rust
client.register_semantic_gatekeeper(sg_config)?;
let ga = client.attach_guardrail(probe, anchor, policy)?;
let score = client.classify_binary("prompt", cfg)?;  // 三者全部触发
let emb = client.encode_intent("intent", LayerAnchor::Relative(0.5), PoolMode::MeanPool)?;
```

- SG 在 pre_node 注入知识残差 (优先级 90)
- Guardrail 在 post_node 评估 hidden 做安全检测 (优先级 40)
- HR 读最终 hidden, 不影响 SG / Guardrail 的 callback 流

### 9.2 失败隔离

- Guardrail 的 `ExitEarly` 只影响**本次前向**; 对下一次 `classify_binary` 调用无影响 (`shared.reset()` 在 `build_guardrail_chain` 时调用)。
- Guardrail 解绑 (detach) 后, `Arc<GuardrailSharedState>` 仍由 `GuardrailAttachment` 持有, 不会泄漏。

---

## §10 实现映射

| SPEC 条目 | 代码位置 |
|-----------|----------|
| `GuardProbe` / `GuardProbeWeights` / `SafetyPolicy` / `GuardrailError` | `src/guardrail.rs` |
| `GuardrailSharedState` / `GuardrailAttachment` | `src/guardrail.rs` |
| `load_probe_weights` / `validate_policy` / `resolve_anchor` | `src/guardrail.rs` |
| `GuardrailProbeCallback` (`LayerCallback::post_node` 实现) | `src/engine/callbacks/guardrail_probe.rs` |
| `Client::attach_guardrail` / `attach_guardrail_inline` / `detach_guardrail` | `src/client.rs` |
| `Client::guardrails` / `build_guardrail_chain` | `src/client.rs` |
| `Backend::apply_guardrail_probe` (CPU 实现 + GPU Unimplemented) | `src/compat/cpu_backend.rs`, `src/compat/gpu_backend_macro.rs` |
| `Executor::score_tokens_for_prompt_with_callbacks` | `src/engine/executor.rs` |
| 公共导出 | `src/lib.rs` |
| E2E 测试 | `tests/test_e2e_guardrail.rs` |

---

## §11 变更历史

SPEC 文件不维护变更历史 (CLAUDE.md §6 铁律 4: SPEC 防腐化)。版本历史由 git 管理。本 SPEC 当前描述的是**唯一正确设计**, 如需更新设计, 整段重写对应章节。
