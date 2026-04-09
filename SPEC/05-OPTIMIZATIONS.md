# 运行时优化全景 (Runtime Optimizations)

> **SSOT 声明**: 本文档是 gllm 推理引擎所有运行时优化机制的统一描述。优化模块的执行路径集成状态由本文件唯一记录。

## 1. Per-Node Callback 架构

所有需要在中间层介入的优化模块通过 Per-Node Callback 接入 `FusedGraphExecutor` 的节点执行循环。

### 1.1 CallbackAction

```rust
pub enum CallbackAction {
    /// 继续正常执行当前节点
    Continue,
    /// 跳过当前节点（Gate Skip / Dead Code Elimination）
    SkipThisNode,
    /// 提前终止层循环（Early Exit / Guardrail Veto）
    ExitEarly { logits: Option<Vec<f32>> },
    /// 注入外部隐藏状态（RAG / Knowledge Injection）
    InjectHidden { data: Vec<f32> },
}
```

### 1.2 LayerContext

```rust
pub struct LayerContext<'a> {
    /// 当前 hidden_state 的可变引用
    pub hidden_state: &'a mut [f32],
    /// 当前层号 (0-based)
    pub layer_idx: usize,
    /// 模型总层数
    pub total_layers: usize,
    /// 当前请求 ID
    pub request_id: RequestId,
}
```

### 1.3 LayerCallback trait + CallbackChain

```rust
pub trait LayerCallback {
    /// 节点执行前介入
    fn pre_node(&mut self, ctx: &LayerContext, node_idx: usize) -> CallbackAction;
    /// 节点执行后介入（遥测、探针）
    fn post_node(&mut self, ctx: &LayerContext, node_idx: usize, output: &[f32]);
}

pub struct CallbackChain {
    callbacks: Vec<Box<dyn LayerCallback>>,
}
```

### 1.4 FusedGraphExecutor 接入点

```
FusedGraphExecutor::run_with_kv_cache_with_callbacks()
  for (node_idx, node) in graph.nodes.iter().enumerate() {
    → callback_chain.pre_node(ctx, node_idx)
    → match action:
        Continue     → execute JIT kernel
        SkipThisNode → skip
        ExitEarly    → break loop
        InjectHidden → replace hidden_state
    → callback_chain.post_node(ctx, node_idx, output)
  }
```

**零开销保证**: 当 CallbackChain 为空时，`run_with_kv_cache()` 走原始路径，无额外分支或函数调用。

## 2. 优化模块清单

### 2.1 Gate Skip (SiLU 死神经元跳过)

**机制**: FFN 阶段 Gate GEMM 后，Epilogue 评估 SiLU(g)，统计死神经元（sigma(x) < epsilon 的列）。死神经元 > 50% 时，硬件谓词掩码（AVX-512 vcompressps / GPU Prefix Sum）物理挤压 inactive lanes，跳过 Up/Down GEMM 通道。

**接入**: `GateSkipCallback` → `pre_node(ffn_nodes)` → `SkipThisNode`。FLOPs 当场砍掉 ~40%。

**铁律**: 不跳远只挤压。死神经元在 SMEM 挤成最小密实体，不破坏全局机器码。

### 2.2 Residual Bypass (残差旁路)

**机制**: Residual Add 的 Epilogue 免费测算跨层能量差 Delta_rho 和方向余弦 cos(theta)。当 Delta_rho < 0.001 且 cos(theta) > 0.99 时，Thread Block 触发 Thread Exit，输入原封不动抛给输出。

**接入**: `pre_node` → 读取 Epilogue Paged Header 的遥测信号 → Thread Block 级路由。全局省力 ~15%。

**铁律**: 绝对不能用宏观 Hot JMP 修补。Continuous Batching 下 Request A 可跳、Request B 不能跳，修改汇编 jmp 会导致 B 产出乱码。

### 2.3 Early Exit (PGSLE)

**机制**: JIT 在特定层（如 Layer 20, 28, 35）的残差 Add 后附带微型 lm_head。当中间 logits 概率逼近阈值（如 99.9%）时，该层 Thread Block 发射控制信标，物理切断后续层计算。

**接入**: `EarlyExitCallback` → `post_node(exit_layers)` → `ExitEarly { logits }`。

**ADEPT 阴影 KV**: Early-Exit token 在 exit 层后的 KV 缺口由 ShadowProjector（低秩投影 rank=64）近似填充。参数量约占模型 0.05%，perplexity 下降 < 0.1。

### 2.4 Late-Fusion RAG (外部知识注入)

**机制**: 外部检索向量通过残差总线的 Vector Add 指令（LDG.E）直接注入指定知识融合层（如第 15 层），跳过前半段网络的无效计算。外部知识用小模型（BERT / 2B）预算出高维语义向量。

**接入**: `RagInjectCallback` → `pre_node(fusion_layer)` → `InjectHidden { data }`。

**铁律**: 注入点语义锚定。残差旁路跳过某层时，该层上的注入点自动迁移到同语义区最近的未跳过层。

### 2.5 Guardrail (零延迟飞行护栏)

**机制**: 在模型深层物理强插入极小安全审查头（Safety Head），寄生在残差总线上。生成过程中每吐出一个新词时，Guard 探针不断嗅探高维语义流中的危险概念特征。一旦概率超标，Mega-Kernel 内 Thread Block 直接抛出 Safety Veto 中断信号，在危险词吐出之前当场熔断。

**接入**: `GuardProbeRunner` → `post_step()` → `HookDecision::Veto / Terminate`。

**铁律**: Safety Veto = Block Routing 标记，非代码修改。探针在 .text.cold 中执行（不占热路径 L1i）。探针结果写入 `SafetyFlags[request_id]`，Scatter 阶段按 flag 排除该 request。

### 2.6 Intent NLU (纯解码意图识别)

**机制**: `Pure_Decode` API 模式下，JIT 仅选取前 15-20 层"理解区"（剥离后续语言生成层），残差总线截留后送入轻量级多分类探针（Linear Probe）。模型化身为超高速意图分类与特征提取器。

**接入**: `IntentRecallCallback` → `post_node(target_layer)` → Recall extraction。

### 2.7 MoE 专家系统

**核心**: 每一轮 MoE 层仅 Launch 1 个 Mega-Kernel，Thread Block 内部汇编 jmp 直接跃迁到对应专家权重读取区，零 Driver 启动开销。

**子系统**:
- **路由** (routing.rs): Gate 概率 → TopK → 专家选择
- **热度追踪** (thermal.rs): Epilogue 白嫖 Gate 命中计数 → 冷板凳专家检测
- **预取** (prefetch.rs / prefetch_pipeline.rs): TurboQuant 4-bit 压缩后 cuMemPrefetchAsync 预加载冷专家权重
- **热补** (hot_patch.rs): JIT Director Daemon 检测全域零命中专家 → Hot JMP 物理抹除其代码分支
- **Deopt** (Uncommon Trap): 被封杀专家被覆写为 Deopt Handler 跳转。触发时 Thread Block 写 DEOPT_REQUEST 并挂起，主机端微冻结后恢复

**接入**: `MoeDispatchCallback` → `pre_node(moe_layers)` → Routing setup。

### 2.8 Speculative Decoding (推测解码)

**EESD 单 GPU 模式**: 复用模型自身的浅层变体（L2_hot ≈ L/3 层）充当 Draft Model，零额外权重。Draft phase（浅层）→ Verify phase（全层，单次 batched forward）→ Compact→Execute→Scatter。

**SAGUARO 多 GPU 模式** (≥2 GPU): Draft GPU + Target GPU 独立部署，Draft 和 Verify 并行执行（流水线重叠）。三个数学最优策略：几何扇出（Geometric Fan-Out）、缓存感知采样（Cache-Aware Sampling）、自适应回退（Adaptive Fallback）。

**共同基础设施**: EqSpec 三不变量（共享拓扑、单次验证、原子 KV Commit）、各向异性推测树（PLD spine + n-gram branches）、硬件级 Compact→Execute→Scatter（x86 vcompressps / GPU Warp Prefix Sum / ARM SVE predicate）。

**接入**: `SpecDecodingState` → `step()` draft/verify phase → `draft_budget` per sequence。

### 2.9 Knowledge Injection (知识注入 SDK)

**机制**: 外部知识（文档/向量/结构化数据）通过 `InjectHidden` 注入指定层。支持 Static（加载时注入）、Dynamic（运行时注入）、LoRA Adapter 等注入方式。

**接入**: `pre_node(target_layer)` → `InjectHidden { data }`。框架就绪，具体注入内容由 SDK 用户定义。

## 3. Mega-Kernel 块级路由

### 3.1 单一内核发射

每一轮 Decode 或 Chunked Prefill，全系统仅 Launch 唯一一个 Mega-Kernel。取消主机条件网，禁止在 CPU Host 为 Gate-First-Skip 等建立多线程调度。

### 3.2 Compact→Execute→Scatter 三段式

1. **Compact**: SM 核心内 Thread Block 读取 Request_State_Table，通过向量外设掩码（AVX-512 vcompress / GPU Prefix Sum）将 active lanes 物理挤压为无 Padding 的连续稠密矩阵
2. **Execute**: 在稠密数据上执行核函数运算
3. **Scatter**: 按原始 Request 偏移原位散射回写，还原到初始 Batch 位置

整个流程在单次 Kernel Launch 内闭环。

### 3.3 Range-Aware Compact Grouping

极低精度（如 W4A4）下，Compact 过程严禁单纯按 Batch 顺序挤压。利用 Epilogue 观测到的 Entropy 和 Residual Delta 指标，将激活值域相近的请求聚集在同一 GEMM Tile 内，值域悬殊的请求互相物理隔离。

### 3.4 硬件指令级实现

| 平台 | 指令 | 说明 |
|------|------|------|
| x86_64 | `vcompressps` / `vpscatterdd` | AVX-512 物理挤压 + 散射 |
| GPU | `__ballot_sync` + `__popc` Prefix Sum | Warp 级挤压 |
| AArch64 SVE | `whilelt` + predicate register | 无需显式 compact，predicate 自动 mask |

### 3.5 L1i 指令缓存预算协议

运行期热路径的指令足迹 必须 <= 80% L1i 容量。剩余 20% 留给硬件预取和 alignment gap。

| 平台 | L1i 大小 | 可用预算 (80%) |
|------|---------|---------------|
| x86_64 (Intel/AMD) | 32 KB | 25.6 KB |
| Apple M-series | 64-128 KB | 51.2-102.4 KB |
| ARM Neoverse V2 | 64 KB | 51.2 KB |
| GPU SM | ~4-8 KB | 3.2-6.4 KB |

**三段式代码段分割**:
- `.text.hot` — L1i 常驻，每个 Variant 仅包含该场景的必要指令（18-26 KB）
- `.text.warm` — L2 常驻，通过 NOP Trampoline 按需拉入 L1i
- `.text.cold` — L3/DRAM，长跳转，仅 Deopt/Guardrail/SAGUARO NCCL 等冷路径

**变体选择发生在 Dispatch-Time**（build_batch 阶段），不在 Mega-Kernel 执行时。每个 Variant 编译时计算 instruction footprint，超 80% L1i 预算则自动降级低优先级机制到 .text.warm。

## 4. TurboQuant 运行时数学优化

### 4.1 定位

TurboQuant 不是权重量化工具，而是推理过程中执行的一组运行时数学优化，使推理精度在任意量化权重格式上逼近数学无损。"无损"不是"权重逼近原值"，而是"推理过程中内积/输出的期望与全精度一致"。

### 4.2 在线 FWHT 旋转插入点

前向传播中 3 个非线性边界（Softmax、SwiGLU、RoPE）各有一个在线 FWHT 旋转：

| 位置 | 白嫖路径 |
|------|---------|
| Softmax(QK^T) V 输出之后 | Attention Epilogue 尾段内联，数据在寄存器/SMEM |
| SwiGLU(Gate) Up 输出之后 | FFN Epilogue 内联，数据在寄存器 |
| RoPE(K) 存入 KV Cache 之前 | KV Write 阶段内联，与 Epilogue STG 共享同一指令流 |

净开销：3 个 O(d log d) 算术 + 每层若干条 max/FMA 指令，远低于 GEMM 的 O(d^2)。

### 4.3 KV Cache 非对称量化 (KIVI)

| 维度 | 离群点特征 | 量化粒度 | Scale 来源 |
|------|-----------|---------|-----------|
| Key | 集中在特定通道（跨 Token 稳定） | Per-Channel | 离线校准常数 |
| Value | 集中在特定 Token（跨通道稳定） | Per-Token | KV 写入时寄存器内 reduce_max |

**Attention Sink 保护**: 前 N 个 Token（默认 N=4）保留 FP16 全精度。Sink 判定从 Epilogue Telemetry 的 Entropy/Centroid 数据推导。

### 4.4 RaBitQ 无偏修正

在 Attention QK^T 计算中引入修正因子：widehat(QK^T) = QK^T_quant * C1 + C0。
- ||v|| 从 RMSNorm 白嫖（||v|| = RMS * sqrt(d)）
- 量化前后内积：量化循环中追加 1 条 FMA 指令
- 理论误差界：O(1/sqrt(D))，D=4096 时约 1.5%

### 4.5 Dual-Track 显存池

| 轨道 | 物理精度 | 职能 |
|------|---------|------|
| 主池 | 3-4 bit | 组级缩放由 Epilogue 快递，KV 按非对称量化 |
| 校验池 (QJL) | 1-bit | XNOR 残差掩码阵列 |

多卡同步时仅需传输原 FP16 内存量纲的 25%（4x 压缩），突破总线墙。

## 5. Epilogue 白嫖网络

### 5.1 核心原则

所有特征检测必须寄生在上游核函数的数学尾段（Epilogue），严禁单独发起采集循环。

### 5.2 白嫖信号流

```
Embedding Lookup ─── ‖embed‖₂ → RaBitQ 初始修正
  ↓
Layer Loop (×N):
  RmsNorm ─────── per-channel scale → KIVI K 量化
  Q/K/V GEMM ──── 行级 ‖row‖₁ + max → 死神经元检测
  RoPE ────────── FWHT 在线旋转
  Softmax ─────── Entropy + Centroid + max + 锐度 → 预取 + Sink 检测
  Residual Add ── Δρ + 方向余弦 → Early Exit
  MoE Gate ────── 路由熵 + 命中计数 → Deopt 信号
  FFN Gate SiLU ─ 死神经元掩码 → Gate-First Skip (40% FLOPs)
  ↓
lm_head GEMM ──── logits 范数 → 采样策略调整
```

### 5.3 零额外计算原则

12 个融合点的指令开销均为 ~1-10 条 SIMD 指令/层。信号传输通过 Epilogue 尾段 STG 指令写入 KV Page Header padding bytes，宿主机后台低频轮询拾取。不存在独立的机制间通信通道。

## 6. 交叉协调约束

### 6.1 Critical 冲突消解

| # | 冲突 | 消解方案 |
|---|------|---------|
| C1 | 单 Kernel 宪法 vs 空间异构分治 | 放松为"同构子批单 Kernel"。同一 Golden Size 的请求打包进同一个 Mega-Kernel launch |
| C2 | 残差旁路 vs 残差总线注入点 | 注入点语义锚定 + 最近邻迁移。跳过某层时其注入点自动迁移到同语义区最近未跳过层 |
| C3 | MoE 核内分发 vs CPU/GPU 并行 | Mega-Kernel 仅执行 GPU 热专家，冷专家标记 DeferredCpuExpert，层结束后 micro-freeze 等待 CPU async |
| C4 | Hot JMP 修补 vs Block Routing 并发安全 | Hot JMP 严格在 step 间隙应用。GPU 双缓冲代码页，x86 5-byte atomic JMP |
| C5 | Block Routing vs 128+ 专家 SMEM 爆炸 | 路由表寄存器化（立即数常量，不占 SMEM），专家权重仅 L2 预取 |
| C6 | Guardrail Safety Veto 作用域 | Safety Veto = Block Routing 标记，非代码修改。Scatter index 中排除该 request |

### 6.2 High 冲突消解

| # | 冲突 | 消解方案 |
|---|------|---------|
| H1 | Chunked Prefill + MoE 的 SMEM 三重压力 | SMEM 硬分区: Attention 40% + Routing 立即数(0%) + Compact 20% + 余量 40% |
| H2 | Telemetry + KV Quant 争抢 Page Header 字节 | Header 扩展为 40B: telemetry + quant_meta + deopt_flags |
| H3 | Chunked Prefill 中 K per-channel scale 校准不完整 | Prefill 阶段用 per-token scale，完成后重算 per-channel scale 并固化 |
| H4 | 4-bit 量化 vs 残差总线注入精度损失 | 注入点前后局部 FP16 切换（独立 Variant） |
| H5 | Spec Decode 的 SM 分区预算不足 | 优先级: Prefill > Verify > Draft；Draft 可降级到 CPU |
| H6 | constexpr strides vs 动态 chunk size | Chunk size 限定为 Golden Size 集合 |
| H7 | EqSpec KV rollback vs Scatter 已写入 | Tree KV 使用临时 buffer (L2)，仅 accepted tokens commit |
| H8 | Shadow KV + TurboQuant 量化范围不匹配 | Shadow projector 输出 clamp 到 TurboQuant 量化范围 |
| H9 | Guardrail vs Spec Decode 树状 token 回滚 | Guard probe 在 draft phase 执行，veto 时整棵树丢弃 |
| H10 | CPU 专家 + Chunked Prefill 层同步 | CPU 专家仅处理 decode token (seq=1) |
| H11 | Draft phase 跳过深层注入点 | Draft phase 注入点前置到浅层 |

### 6.3 禁止的冲突模式

- Mega-Kernel 内运行时 `if moe_enabled / if guardrail_active / if spec_phase` — 编译为独立 Variant
- 热路径包含 Guardrail probe / Shadow KV projector / Late-Fusion RAG 注入的代码体 — 放 .text.cold
- Hot JMP 在 step 执行期间触发 — 严格 step 间隙
- 单个 Variant 包含所有机制代码 — 变体特化，每场景 ~20 KB

## 7. 集成路径注册表

每个优化模块的执行路径集成状态。状态只有两种：Integrated（决策通过 Callback 改变执行行为）和 Not Integrated（等同于未实现）。

| 模块 | 回调实现 | Hook Point | Decision Type | 硬件前提 | 状态 |
|------|----------|------------|---------------|---------------|------|
| Early Exit (PGSLE) | `EarlyExitCallback` | `post_node(exit_layers)` | `ExitEarly { logits }` | `num_simd_regs` ≥ 20; logits 概率分布集中度可检测 | Integrated |
| Gate Skip | `GateSkipCallback` | `pre_node(ffn_nodes)` | `SkipThisNode` | `num_simd_regs` ≥ 16 + 4 scratch; 检测成本 < 脳过收益 | Integrated |
| Late-Fusion RAG | `RagInjectCallback` | `pre_node(fusion_layer)` | `InjectHidden { data }` | 需调用 `set_rag_system()` | Integrated |
| Guardrail Probe | `GuardrailProbeCallback` (via Callback Chain) | `post_node(all_layers)` | `Veto / Terminate` | 需调用 `add_guardrail_runner()` | Integrated |
| Intent NLU | `IntentRecallCallback` | `post_node(target_layer)` | Recall extraction | — | Integrated |
| MoE Dispatch | `MoeDispatchCallback` | `pre_node(moe_layers)` | Routing setup | MoE 模型 | Integrated |
| Speculative Decoding | `SpecDecodingState` | `step()` advice only | `should_speculate()` 建议 | — | Integrated (状态机; draft/verify 管线未实现) |
| Knowledge Injection | `KnowledgeInjectCallback` | `pre_node(target_layer)` | `InjectHidden { data }` | 需调用 `inject_knowledge()` → `set_knowledge_payload()` | Integrated |
| Embed+Rerank Fusion | `EmbeddingsBuilder.rerank_query()` → `Client::execute_embed_rerank_pipeline()` | `generate()` 内部 | `ReorderByScore` | 多模型管线 (02-ARCHITECTURE.md ARCH-MULTI-MODEL-PIPELINE) | Integrated |

**铁律**: SPEC 审计时，任何 Not Integrated 的模块等同于未实现，不接受"逻辑正确但未接入"作为完成状态。

**禁止"日志即丢弃"反模式**: 优化模块的决策产出禁止仅通过日志输出消费。决策必须通过结构化路径（Callback/Event/State）改变执行行为。

## 8. 执行优先级表

| 优先级 | 回调 | 触发时机 |
|--------|------|---------|
| 100 | Prefetch | `pre_node` |
| 90 | Knowledge Inject | `pre_node` |
| 80 | RAG Inject | `pre_node` |
| 70 | MoE Dispatch | `pre_node` |
| 60 | Gate Skip | `pre_node` |
| 50 | Early Exit | `post_node` |
| 40 | Guardrail Probe | `post_step` |
| 30 | Intent Recall | `post_node` |
| 20 | Residual Bypass | `pre_node` |
| 10 | Telemetry | `post_node` |

高优先级先执行。`SkipThisNode` / `ExitEarly` 等终止性 action 会跳过后续低优先级回调。

## 9. MoE 专家热度管理 (Thermal Management)

### 9.1 定位

运行时追踪 MoE 专家的命中率与热度状态，实现冷专家封杀（Eviction）与 Deopt 恢复（Reactivation）。封杀决策由 `ExpertThermalManager` 驱动，实际 `.text` 回写由 JIT Director 执行（Hot JMP Patching，`src/moe/hot_patch.rs`）。

**核心原则**: 门控（Gate Router）计算永远保留（开销 < 1%）。冷专家权重被 NOP/Deopt 跳转替换。请求触发冷专家时走 Uncommon Trap → DEOPT_REQUEST → 微冷冻 → 回写 `.text` → 重算。

### 9.2 四级热度状态机

```rust
pub enum ExpertHeatLevel {
    Hot,     // 活跃使用中，权重常驻 GPU L2
    Warm,    // 偶尔使用，权重在 CPU RAM
    Cold,    // 长时间未使用，可能被封杀
    Evicted, // 已被 NOP/Deopt 替换，需 OSR Bailout 恢复
}
```

**状态转移**: Hot → Warm → Cold → Evicted（单向降级）。Evicted → Cold 仅通过 `reactivate_expert()` 恢复（Deopt 触发）。

| 转移 | 触发条件 | 执行者 |
|------|---------|--------|
| Hot → Warm | `hit_rate < hot_threshold` | `step()` 自动推导 |
| Warm → Cold | `hit_rate < cold_threshold` | `step()` 自动推导 |
| Cold → Evicted | `consecutive_zero_streak >= eviction_streak_threshold` | `evict_expert()` |
| Evicted → Cold | Deopt 触发恢复 | `reactivate_expert()` |

**热度推导函数**: `ExpertHeatLevel::from_hit_rate(rate, hot_threshold, cold_threshold)`

| hit_rate 范围 | 热度级别 |
|--------------|---------|
| `>= hot_threshold` | Hot |
| `[cold_threshold, hot_threshold)` | Warm |
| `(0, cold_threshold)` | Cold |
| `== 0.0` | Evicted |

### 9.3 ExpertHeatState

```rust
pub struct ExpertHeatState {
    pub expert_idx: usize,
    pub hit_rate: f64,                  // 历史命中率 (0.0-1.0)
    pub hit_count: u64,                 // 累计命中次数
    pub route_count: u64,               // 累计路由次数
    pub heat_level: ExpertHeatLevel,    // 当前热度级别
    pub consecutive_zero_streak: u64,   // 连续零命中次数 (封杀决策依据)
    pub last_hit_step: u64,             // 最近一次命中的时间步
    pub is_evicted: bool,               // 是否已被封杀
    pub reactivation_count: u64,        // 封杀后被触发的次数
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `hit_rate` | `f64` | 滑动窗口近似: `hit_count / route_count` |
| `consecutive_zero_streak` | `u64` | 每步 route_count > 0 但 count == 0 时递增；命中时归零 |
| `is_evicted` | `bool` | 封杀标记，由 `evict_expert()` 设为 `true` |
| `reactivation_count` | `u64` | 每次 Deopt 恢复递增，用于 `experts_to_reactivate()` 批量恢复判断 |

### 9.4 ExpertThermalManager

```rust
pub struct ExpertThermalManager {
    num_experts: usize,
    states: Vec<ExpertHeatState>,
    eviction_streak_threshold: u64,   // 默认 1,000,000
    hot_threshold: f64,               // 默认 0.1 (>10%)
    cold_threshold: f64,              // 默认 0.001 (<0.1%)
    current_step: u64,
    pending_deopt_requests: Vec<DeoptRequest>,
}
```

| 配置参数 | 默认值 | 说明 |
|---------|--------|------|
| `eviction_streak_threshold` | `1_000_000` | 连续零命中次数达到此值则可封杀 |
| `hot_threshold` | `0.1` | 命中率 ≥ 10% → Hot |
| `cold_threshold` | `0.001` | 命中率 < 0.1% → Cold |

**核心方法**:

| 方法 | 说明 |
|------|------|
| `step(route_counts)` | 推进一步，更新所有专家的 hit_rate / streak / heat_level |
| `eviction_decision(idx)` | 返回 `Keep / Evict / Reactivate` |
| `evict_expert(idx)` | 标记专家为 Evicted，JIT Director 用 NOP/Deopt 替换访存分支 |
| `reactivate_expert(idx)` | 恢复专家至 Cold，重置 streak |
| `handle_deopt_request(req)` | 处理 Uncommon Trap 触发的 Deopt 请求 |
| `experts_to_evict()` | 批量返回所有需要封杀的专家索引（供 JIT Director 消费） |
| `experts_to_reactivate()` | 批量返回所有需要恢复的专家索引（供 JIT Director 消费） |
| `summary()` | 返回 `ThermalSummary` 统计 |

### 9.5 Deopt 处理流程

```
Thread Block 撞进 Uncommon Trap
  → 写下 DEOPT_REQUEST (request_id, expert_idx, layer_idx, step)
  → Thread Block 挂起
  → 引擎主循环发现 DEOPT_REQUEST
  → ExpertThermalManager::handle_deopt_request()
  → reactivate_expert() → JIT Director 回写 .text 恢复专家代码
  → 异步唤回主存的 4-bit 权重
  → 挂起的 Request 走一遍回炉重造 (Re-evaluate)
```

**Deopt 处理结果**:

```rust
pub enum DeoptHandlingResult {
    ReactivateAndRerun { expert_idx: usize, request_id: u64 },
    SpuriousDeopt { expert_idx: usize, request_id: u64 },
}
```

| 变体 | 触发条件 |
|------|---------|
| `ReactivateAndRerun` | 专家确实处于 Evicted 状态，执行恢复并重算 |
| `SpuriousDeopt` | 专家未被封杀，Deopt 为误触发（可能是竞态或旧代码页） |

### 9.6 ThermalSummary 统计

```rust
pub struct ThermalSummary {
    pub num_experts: usize,
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub evicted_count: usize,
    pub total_evictions: u64,
    pub total_reactivations: u64,
    pub current_step: u64,
    pub pending_deopt_count: usize,
}
```

### 9.7 与其他模块的联动

| 联动模块 | 接口 | 说明 |
|---------|------|------|
| MoE Routing (§2.7) | `step(route_counts)` | 每步 Gate Router 的路由结果驱动热度更新 |
| Epilogue 白嫖 (§5) | Gate 命中计数 | Epilogue 免费提供 Gate 命中计数，零额外计算 |
| JIT Director (§14.4) | `experts_to_evict()` / `experts_to_reactivate()` | 批量操作，Hot JMP Patching 实际修改 `.text` |
| MoE Prefetch (§2.7) | `hot_experts()` | 热专家列表驱动权重预取策略 |

## 10. Speculative DraftAdapter (零参数投影头)

### 10.1 定位

附在 Speculative Decoding L2_hot 浅层变体末尾的微型投影图，将中间 hidden state 映射到 vocab 空间。通过共享 lm_head.weight 实现零额外参数（Phase A），可选在线蒸馏学习残差修正（Phase B）。

### 10.2 两阶段设计

| 阶段 | 额外参数 | 说明 |
|------|---------|------|
| Phase A | 0 | 直接复用 lm_head.weight，零额外存储 |
| Phase B | `vocab_size × hidden_size × sizeof(f32)` | 添加 `residual_delta` 通过在线蒸馏学习（≈0.1% 模型大小） |

**Phase A 图结构**:

```
RmsNorm(hidden_state, norm_weight) → MatMul(normed, lm_head.weight^T) → logits
```

**Phase B 图结构**:

```
RmsNorm(hidden_state, norm_weight) → MatMul(normed, (lm_head.weight + residual_delta)^T) → logits
```

### 10.3 AdapterConfig

```rust
pub struct AdapterConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub enable_distillation: bool,
    pub distillation_steps: usize,  // 默认 100
}
```

| 字段 | 说明 |
|------|------|
| `hidden_size` | 模型 hidden 维度（来自 ModelConfig） |
| `vocab_size` | 词表大小（来自 ModelConfig） |
| `rms_norm_eps` | LayerNorm epsilon |
| `enable_distillation` | 是否启用 Phase B 蒸馏残差 |
| `distillation_steps` | Phase B 蒸馏步数（前 N 步用 full model logits 蒸馏） |

### 10.4 DraftAdapter

```rust
pub struct DraftAdapter {
    config: AdapterConfig,
    weight: Arc<Vec<f32>>,              // 共享 lm_head.weight [vocab_size, hidden_size]
    residual_delta: Option<Vec<f32>>,   // Phase B: 蒸馏残差 [vocab_size, hidden_size]
    distillation_step: usize,           // 已完成的蒸馏步数
    norm_weight: Arc<Vec<f32>>,         // 共享最后一层 norm 权重 [hidden_size]
}
```

| 方法 | 说明 |
|------|------|
| `new_phase_a(config, lm_head_weight, norm_weight)` | 创建 Phase A Adapter（零额外参数） |
| `new_phase_b(config, lm_head_weight, norm_weight)` | 创建 Phase B Adapter（初始化 `residual_delta` 为零向量） |
| `forward(hidden)` | 单个 hidden → logits |
| `forward_batch(hiddens)` | 批量 draft（EqSpec 场景，多个 sequence 同时 draft） |
| `distill_step(draft_logits, target_logits, hidden, lr)` | Phase B SGD 更新 delta，返回 MSE loss |
| `is_distillation_complete()` | 蒸馏步数 >= `distillation_steps` |
| `distillation_progress()` | 返回 `(current_step, total_steps)` |
| `parameter_bytes()` | Phase A = 0；Phase B = `vocab_size × hidden_size × 4` |

### 10.5 Phase B 在线蒸馏

```
每步 forward() 产生 draft_logits
  → 如果 distillation_step < distillation_steps:
      → full model 同步产生 target_logits (ground truth)
      → distill_step(): softmax_diff(draft, target) → gradient
      → SGD: delta[v,h] -= lr × grad[v] × normed[h]
      → distillation_step += 1
  → else:
      → 蒸馏完成，delta 固定
```

**梯度信号**: `softmax(draft_logits) - softmax(target_logits)`，即 draft 分布与 full model 分布的差异。

### 10.6 与 Speculative Decoding 的集成

| 场景 | 使用的 Adapter 方法 | 说明 |
|------|-------------------|------|
| EqSpec 单 GPU | `forward()` | 浅层变体 draft → 全层 verify |
| EqSpec batch | `forward_batch()` | 多 sequence 并行 draft |
| 蒸馏训练阶段 | `distill_step()` | 前 `distillation_steps` 步收集 full model logits |
| 蒸馏完成 | `is_distillation_complete()` | 停止收集 full model logits，节省开销 |

## 11. Centroid-Guided KV Prefetch

### 11.1 定位

利用 Softmax Epilogue 免费提取的 attention probability centroid（argmax token position），为下一层的 KV cache block 发出异步预取指令。数据来自 Epilogue 遥测，零额外计算。

### 11.2 机制

```
Layer N Attention Softmax Epilogue
  → 免费计算 attention probability centroid (argmax token position)
  → PrefetchQueue::enqueue(layer=N, centroid_token_idx)
  → 计算 KV cache 物理地址: offset = (N+1) × kv_stride + centroid_token_idx × block_size
  → issue_prefetch(): 发出硬件预取指令
```

**直觉**: Softmax 的 attention centroid 指示了模型在 layer N 最关注的历史 token 位置。Layer N+1 极大概率也需要该位置的 KV cache block，因此提前预取到 L2 cache。

### 11.3 PrefetchRequest

```rust
pub struct PrefetchRequest {
    pub layer: usize,       // 目标层 = 当前层 + 1
    pub token_idx: usize,   // Centroid token position (argmax)
}
```

### 11.4 PrefetchQueue

```rust
pub struct PrefetchQueue {
    queue: VecDeque<PrefetchRequest>,
    block_size: usize,      // KV cache block 物理大小 (bytes)
}
```

| 方法 | 说明 |
|------|------|
| `new(block_size)` | 创建队列，block_size 为 KV cache block 的物理大小 |
| `enqueue(layer, centroid_token_idx)` | 入队 `layer + 1` 层的预取请求 |
| `issue_prefetch(kv_cache_ptr, kv_stride)` | 批量发出所有队列中的预取指令并清空队列 |

### 11.5 硬件特化预取指令

| 平台 | 指令 | 目标缓存级别 | 说明 |
|------|------|-------------|------|
| x86_64 | `_mm_prefetch` (`_MM_HINT_T1`) | L2 | Intel/AMD prefetch to L2 |
| AArch64 | `PRFM PLDL2KEEP` | L2 | ARM prefetch to L2, keep policy |
| 其他 | no-op | — | 不支持的架构直接清空队列 |

**地址计算**: `ptr = kv_cache_ptr + layer × kv_stride + token_idx × block_size`

### 11.6 与 Epilogue 白嫖网络的集成

本模块是 §5 Epilogue 白嫖网络的直接消费者之一:

| Epilogue 信号 | 消费者 | 用途 |
|--------------|--------|------|
| Softmax Centroid (argmax) | `PrefetchQueue::enqueue()` | 确定 layer N+1 需要预取的 KV block 位置 |
| Softmax Entropy | TurboQuant Attention Sink 保护 (§4.3) | 判断是否保留 FP16 全精度 |
| Softmax max / 锐度 | 采样策略调整 (§5.2) | logits 范数 → 温度补偿 |

Centroid 数据从 Epilogue STG 指令写入的 KV Page Header padding bytes 中读取，无独立通信通道。
