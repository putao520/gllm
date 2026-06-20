# 32 — Mega-Kernel 性能增强：Prefill/Decode 差异化 + SM 分区 Ping-Pong + 自治批调度

> **Status**: 设计完成
> **SSOT**: 本文档是 Mega-Kernel 性能增强的唯一真源。**不替代** `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md`（23 参数 ABI + JMP Table SSOT），而是作为其补充扩展。
> **前置 SPEC**: `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md` §1.5.5（23 参数 ABI + OutputModeDispatch JMP Table）; `20-BATCH-CONCURRENT-INFERENCE.md`（BatchContext flat memory）; `jit-cache-protocol.md` §2.2（单一 Mega-Kernel 缓存粒度铁律）
> **演进自**: `12-STRATEGY-ARBITER.md`（StrategyBias/HwOptEngine → SM 分区比例/编译参数差异化）; `02-HARDWARE.md`（DeviceProfile → 硬件感知 prefill/decode 策略选择）; `02-ARCHITECTURE.md §13.12`（12 Profile 硬件感知融合拓扑 → Mega-Kernel 内 code path 差异化）
> **基础设施依赖**: CUDA Runtime API `cudaOccupancyMaxPotentialClusterSize()` 查询硬件实际支持的 cluster 大小；`gllm-nccl/src/lower/` mbarrier + cp.async.bulk PTX 模板（已存在）

---

## §0 核心设计原则

1. **单一 Mega-Kernel 不变**：保持 `MegaKernelFn = unsafe extern "C" fn(ctx: *const u8) -> u32` + 23 参数 ABI + JIT Cache 单实例铁律
2. **JMP Table 内差异化**：prefill（compute-bound）和 decode（memory-bound）作为同一 Mega-Kernel 内的不同 JMP 分支，各自使用最优编译参数（GEMM tile、attention 策略、KV cache 访问模式）
3. **SM 分区并行**：单 Mega-Kernel 内通过 thread role + **GPC 对齐**的 CTA 分区实现 decode∥prefill 重叠执行。同步原语按硬件分档：`cluster.sync`（SM90+）/ `cooperative_groups::grid_group::sync()`（SM80）/ persistent ring barrier（SM70-/CPU）。`bar.sync` 仅用于 CTA 内 16-warp 同步，**禁止跨 CTA 使用**
4. **自治连续批调度**：Mega-Kernel decode loop 内 GPU 原子操作完成 page alloc/free + 序列压缩 + 请求补充，零 CPU 参与
5. **编译时硬件感知**：DeviceProfile 驱动 SM 分区比例（沿 GPC 边界对齐）、tile 策略、prefill/decode code path 参数选择，零运行时退化
6. **推理热路径边界**：推理热路径 = `input_ids_ptr` 到 `output_tokens_ptr`。Tokenize（text→IDs）和 detokenize（IDs→text）是请求级预/后处理，不在 Mega-Kernel 内执行，不违反 ARCH-RUST-IS-CODEGEN

---

## §1 Prefill/Decode JMP Table 差异化 (REQ-MKO-001)

### §1.1 Prologue / ForwardPhaseDispatch 嵌套关系

```
compile() 扩展布局:

Phase 0 (prologue): Load ABI params (23 参数)
  Prologue 子阶段: Load batch_ctx_ptr (arg 22)
  ├── NULL → Legacy 单序列路径（现有逻辑不变，无 ForwardPhaseDispatch）
  └── non-NULL → Batch mode
      ForwardPhaseDispatch (SPEC/39 §1.3.3): (新增 JMP Table)
        ├── Load total_prefill_tokens from BatchContext[8]
        ├── CMP total_prefill_tokens, 0
        │   ├── > 0 → JE .prefill_path       ← compute-bound 优化
        │   └── == 0 → JE .decode_path        ← memory-bound 优化
        └── (注：仅 batch mode 执行 ForwardPhaseDispatch)

.prefill_path:
  prefill 路径首融合组: Compute prefill-derived values (M=sum(prompt_lens))
  层循环融合组（prefill 路径）: Forward pass (embed 融合组 → 层循环融合组 → lm_head 融合组) ← prefill 参数化
  OutputModeDispatch (SPEC/39 §1.3.3): 现有 6 模式 JMP Table
  Argmax/StoreToken 融合组: Sample 首 token
  Argmax/StoreToken 融合组（BatchContext 状态更新）: 写 total_prefill_tokens = 0 到 BatchContext[8]
  → JMP .decode_entry                         ← prefill 后直接进 decode loop

.decode_entry:
.decode_path:
  decode 路径首融合组: Compute decode-derived values (M=num_active_seqs)
  层循环融合组（decode 路径）: Forward pass (embed 融合组 → 层循环融合组 → lm_head 融合组) ← decode 参数化
  OutputModeDispatch (SPEC/39 §1.3.3): 现有 6 模式 JMP Table
  decode 层循环融合组（SM 分区）: Sample next token (per-seq, §2.4)
  Argmax/StoreToken 融合组（streaming output）: Stop condition check
  CheckStopCondition 融合组（streaming）: Stream output token (§4)
  generate loop end: Compact + Refill (§3)
  epilogue: 检查 refill 是否引入新 prefill 序列
    ├── 新序列 total_prefill_tokens > 0 → JMP .prefill_path   ← 回到 prefill
    └── 无新序列 → LoopEnd → JMP .decode_entry

.phase_done:
  Function epilogue → RET
```

**关键**：prefill_path 末尾 JMP 到 `.decode_entry`（不是 `.phase_done`），refill 后如检测到新 prefill 序列则 JMP 回 `.prefill_path`。形成 `prefill → decode loop → [refill 引入新序列 → prefill → decode loop]` 循环。

**Legacy 单序列**：`batch_ctx_ptr=NULL` 时不执行 ForwardPhaseDispatch，走现有单次 forward + OutputModeDispatch 路径，不受本 SPEC 影响。

### §1.2 Legacy 模式的 prefill/decode

Legacy 单序列模式（`batch_ctx_ptr=NULL`）通过现有 ABI 参数 `prompt_len` (arg 6) 和 `max_new_tokens` (arg 12) 控制：
- `prompt_len > 0`：执行完整 prefill + sample 首 token → 进入 decode loop
- decode loop 重复执行直到 `max_new_tokens` 或 EOS
- 无需 ForwardPhaseDispatch，现有逻辑已覆盖

### §1.3 差异化编译参数

| 编译参数 | prefill_path | decode_path |
|---------|-------------|-------------|
| GEMM tile | 大 M×N×K（compute-bound 优化） | 小 M=1, 优化 N×K（memory-bound 优化） |
| Attention | FlashAttention / full KV 写入 | Incremental KV（读历史 + 写 1 行）+ GQA 共享 |
| KV cache | 写入完整 KV entries | 读取历史 + 写入新 1 行 |
| Forward pass M | `total_prefill_tokens` (Symbolic) | `num_active_seqs` (Symbolic) |
| Sampling | 首 token argmax/sample | 每 step per-seq sample |
| 外层 generate loop | 无（prefill 后 JMP decode_entry） | decode loop + refill 循环 |

### §1.4 Chunked Prefill 混合路径（默认路径）

**问题**：refill 引入少量新序列时，立即跳 `.prefill_path` 导致全 SM 切路径，产生抖动。1 个新序列独占 25% SM 执行 prefill，剩余 decode SM 空闲浪费。

**解决**：默认走 `.mixed_path`（prefill chunk + decode 合并同一 forward pass），仅当 prefill chunk 过大时走 dedicated `.prefill_path`。

```
ForwardPhaseDispatch (SPEC/39 §1.3.3) 扩展:
  total_prefill = BatchContext[8]
  if total_prefill == 0:
    → .decode_path
  elif total_prefill <= PREFILL_CHUNK_THRESHOLD:       // 编译时常量，默认 512
    → .mixed_path                                      // 推荐：混合路径
  else:
    → .prefill_path                                    // 大 chunk 独占 prefill

.mixed_path:
  M = num_active_decode_seqs + prefill_chunk_tokens    // 合并 M
  层循环融合组（混合路径）: Forward pass（混合 attention mask）
    decode 部分: 增量 KV attention (memory-bound 参数)
    prefill chunk: FlashAttention (compute-bound 参数)
    → attention mask 按 seq_id 区分 decode/prefill 段
  Argmax/StoreToken 融合组: Sample（decode seqs） + Prefill output（prefill seqs 首 token logits）
  → JMP .decode_entry
```

**编译时参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `PREFILL_CHUNK_THRESHOLD` | 512 tokens | 超过此值走 dedicated prefill_path |
| `PREFILL_CHUNK_SIZE` | 512 tokens | 每次 mixed_path 消化的 prefill token 上限 |
| attention mask | 混合 mask | per-token seq_id 区分 decode( causal ) / prefill( full causal ) |

**优势**：消除路径切换抖动；vLLM V1 / SGLang 默认策略。编译时为 `.mixed_path` 生成第三条优化路径的指令序列。

两条基础路径走**同一个 `emit_fusion_groups` 函数**，SymDim bound 参数不同。编译时为三条路径分别生成最优指令序列，运行时条件 JMP 平均开销 < 5 cycle（branch predictor 命中时 1-2 cycle）。

---

## §2 SM 分区 Ping-Pong 重叠执行 (REQ-MKO-002)

### §2.1 设计原理

decode 是 memory-bound，只用部分 SM 带宽。剩余 SM 空闲。在单 Mega-Kernel 内，通过 thread role 分区让 decode 和 prefill 重叠执行。

```
时间线 →
CTA 0-80 (decode):    [Decode_step_N] ───→ [Decode_step_N+1] ───→ ...
CTA 81-107 (prefill):      [Prefill_new_seq_A] ──→ ...
```

### §2.2 Thread Role 分区与多级同步

GPU 路径：Mega-Kernel launch 时 CTA 总数 = `decode_cta_count + prefill_cta_count`（编译时 bake 的常量）。每个 CTA 根据 blockIdx 判断角色。

**关键语义**：`bar.sync N` 是 CTA 内 16-warp 同步指令，**不能跨 CTA**。跨 CTA 同步需要多级原语链（编译时根据 DeviceProfile 选择，NO_HW_DEGRADATION）。

#### §2.2.1 概念澄清：Cluster vs Grid

- **Thread Block Cluster**：SM90+ 协同执行单元。Portable cluster size = **8 CTA**（跨 SM90/SM100 通用）。Non-portable cluster size = **16 CTA**（需 `cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` 显式 opt-in）。**一个 Mega-Kernel grid 可以包含多个 cluster**。
- **Cluster 内同步**：`cluster.sync`（延迟 ~20 cycle）。仅同步同一 cluster 内的 CTA。
- **跨 Cluster 同步**：需要 `mbarrier.arrive/wait`（全局内存中的 barrier 对象）+ `__threadfence_system()`（全 grid 内存可见性）。
- **Grid 级同步**：`cooperative_groups::this_grid().sync()`（要求 cooperative launch，SM80 兜底方案）。

**本 SPEC 的设计**：整个 Mega-Kernel grid 包含多个 cluster（每个 cluster 由 decode + prefill CTA 组成）。`cluster.sync` 用于 cluster 内同步；跨 cluster 通过 mbarrier 多级汇合同步。

#### §2.2.2 同步原语分硬件档

| 硬件 | Cluster 内同步 | 跨 Cluster 同步 | 说明 |
|------|--------------|----------------|------|
| **SM90+** | `cluster.sync`（≤8 CTA portable / ≤16 CTA opt-in） | `mbarrier.arrive` + `mbarrier.test_wait` + `__threadfence_system()` | 每个 cluster 选一个 representative CTA 参与 mbarrier 汇合 |
| **SM80** | 无 cluster 概念 | `cooperative_groups::this_grid().sync()` | 要求 cooperative launch；性能约 200-500 cycle |
| **SM70- / CPU** | 无 cluster 概念 | Persistent ring barrier（`atom.global.add.u32` + 自旋） | 最简实现，延迟最高 |

#### §2.2.3 SM90+ 多级同步伪代码

```
// 编译时通过 cudaOccupancyMaxPotentialClusterSize() 查询实际支持值
// 确定 CLUSTER_SIZE (8 或 16, 取决于是否 opt-in non-portable)
// Grid 包含 NUM_CLUSTERS 个 cluster
// 每个 cluster: 前 DECODE_PER_CLUSTER 个 CTA = decode, 后 PREFILL_PER_CLUSTER = prefill

setp.lt.u32  %is_decode, %local_ctaidx, DECODE_PER_CLUSTER

@%is_decode:
  call decode_forward, (decode_ctx)
  bra .cluster_barrier_0

@%is_prefill:
  call prefill_forward, (prefill_ctx)
  bra .cluster_barrier_0

.cluster_barrier_0:
  // Level 1: Cluster 内同步（cluster.sync, ~20 cycle）
  cluster.sync;

  // Level 2: 跨 Cluster 同步（仅 representative CTA 执行）
  // 每个 cluster 的 CTA 0 作为 representative
  if %local_ctaidx == 0:
    // arrive at global mbarrier（所有 cluster 的 representative 汇合）
    mbarrier.arrive.shared::cta.mbarrier_ptr, %representative_tid;
    // 等待所有 cluster representative 到达
    mbarrier.test_wait.shared::cta.mbarrier_ptr, %state, 1;
    // 全 grid 内存可见性
    __threadfence_system();

  // Level 3: Cluster 内广播（representative 通知同 cluster 其他 CTA）
  cluster.sync;    // 确保 fence 对同 cluster 所有 CTA 可见
```

**编译时参数**：

| 参数 | 来源 | 说明 |
|------|------|------|
| `CLUSTER_SIZE` | `cudaOccupancyMaxPotentialClusterSize()` 运行时查询 | Portable=8, non-portable opt-in=16 |
| `DECODE_PER_CLUSTER` | 编译时 bake | `DECODE_CTA_COUNT / NUM_CLUSTERS` |
| `PREFILL_PER_CLUSTER` | 编译时 bake | `PREFILL_CTA_COUNT / NUM_CLUSTERS` |
| `NUM_CLUSTERS` | 编译时 bake | `total_cta_count / CLUSTER_SIZE` |

**Cluster 配置选择**：
- 默认使用 portable cluster size (8 CTA)，零额外配置
- 用户显式启用 non-portable (16 CTA) 时，编译变体名追加 `_NP16` 标记

CPU 路径：无 SM 分区，串行执行（prefill 先，decode 后）。见 §2.5。

### §2.3 DualBatchMeta

```rust
/// Ping/Pong 双缓冲元数据
/// 位于 BatchContext 扩展区（§6.2），不侵入现有字段
#[repr(C)]
struct DualBatchMeta {
    /// Ping：当前 decode batch 的 seq_meta 起始偏移（相对 seq_meta_base，单位：SEQ_META_STRIDE）
    ping_seq_offset: u32,
    ping_seq_count: u32,
    /// Pong：compact+refill 后新 batch 的 seq_meta 起始偏移
    pong_seq_offset: u32,
    pong_seq_count: u32,
    /// Barrier epoch：每步递增。SM90+ cluster.sync 隐式同步；SM80 grid_sync 需要 epoch 防止跨步同步；SM70- ring barrier 用作 arrival count
    step_epoch: u32,
    /// SM70-/CPU ring barrier：每步到达计数器
    epoch_arrival_count: u32,
}
```

**注意**：`decode_cta_count` 和 `total_cta_count` 不在 DualBatchMeta 中——它们是编译时 bake 的常量，不是运行时字段。

### §2.4 重叠执行流程与并发安全

```
Mega-Kernel 入口 → ForwardPhaseDispatch (SPEC/39 §1.3.3):

  // ===== Step 0: 初始 Prefill =====
  if total_prefill_tokens > 0:
    // SharedKvRef 两阶段 prefill（SPEC 03 §1.3.2）
    if num_kv_shared_layers > 0:
      // 阶段 1: Donor pass — 完整计算，KV cache 写入
      for layer_idx in [0, num_layers - num_kv_shared_layers):
        forward_layer(batch_ctx, layer_idx)       // 全 op 执行
      // 阶段 2: Consumer pass — 跳过 K/V GEMM，MHA 读 donor KV
      for layer_idx in [num_layers - num_kv_shared_layers, num_layers):
        forward_layer(batch_ctx, layer_idx)       // GprCondAction 跳过 k/v
    else:
      prefill_path(batch_ctx)                     // 非 SharedKvRef：标准 prefill
    // 写 BatchContext[8] total_prefill_tokens = 0
  GLOBAL_SYNC B0                                  // 等 prefill 完成

  // ===== Decode Loop =====
  .decode_entry:
    decode_path(batch_ctx)                        // decode CTA 组
    // ↑ 此时 prefill CTA 组空闲等待 GLOBAL_SYNC

    GLOBAL_SYNC B0                                // 等 decode 完成

    // ===== Sampling（decode CTA 组分摊）=====
    // CTA[i] 负责 seq[i], seq[i + decode_cta_count], seq[i + 2*decode_cta_count], ...
    for seq_idx in range(cta_id, ping_seq_count, decode_cta_count):
      sample(batch_ctx, seq_idx)
      stop_check(batch_ctx, seq_idx)
      stream_output_token(batch_ctx, seq_idx)     // §4 per-token streaming

    // ===== Compact + Refill =====
    // Stage 1-2: 并行标记 + prefix-scan (§3.3, all decode CTAs)
    // Stage 3: 并行搬移 (§3.3, all decode CTAs)
    // Stage 4: Refill (仅 CTA 0，RequestQueue 入队顺序敏感)
    parallel_compact(batch_ctx)                   // §3.3 三阶段并行
    if cta_id == 0:
      refill_from_queue(batch_ctx)                // §3.3 Stage 4
      // 如 refill 引入新 prefill 序列：
      //   st.global.release BatchContext[8] = total_prefill_tokens（release 语义）
      //   __threadfence_system() 保证所有 CTA 在 GLOBAL_SYNC B1 后可见
      swap(ping, pong)
      step_epoch += 1

    GLOBAL_SYNC B1                                // 等 compact+refill 完成
    // 所有 CTA 读 swap 后的 ping meta，开始下一步

    // ===== 如有新 prefill 序列，prefill CTA 组执行 =====
    if total_prefill_tokens > 0:
      prefill_path(batch_ctx)
      total_prefill_tokens = 0
    GLOBAL_SYNC B0                                // 等 prefill 完成

    if active_seqs > 0: JMP .decode_entry

  // ===== Phase DONE =====
  function_epilogue → RET
```

**`GLOBAL_SYNC` 语义**（编译时根据 DeviceProfile 选择，§2.2.2）：
- **SM90+**: `cluster.sync`（cluster 内同步 ~20 cycle）+ mbarrier（跨 cluster，representative CTA 汇合）+ `__threadfence_system()`（全 grid 内存可见性）
- **SM80**: `cooperative_groups::this_grid().sync()`（延迟 ~200-500 cycle）
- **SM70-/CPU**: `atom.global.add.u32(epoch_arrival_count, 1)` + 自旋等待 `epoch_arrival_count == total_cta_count`

**并发安全保证**：
- `GLOBAL_SYNC B0`：隔离 decode forward 与 sampling/compact，确保 decode 不读半写入的 seq_meta
- `GLOBAL_SYNC B1`：隔离 compact 与下一步 decode，确保所有 CTA 读到 swap 后的一致视图
- Compact 三阶段并行执行（§3.3），仅 Refill 阶段由 CTA 0 独占（入队顺序敏感）
- prefill CTA 组在 `GLOBAL_SYNC B0` 后才启动，不会与 compact 并行
- 所有子系统共享内存（SG hook_ctx、Guardrail callback_table）在 `GLOBAL_SYNC` 后全局可见

### §2.5 CPU 路径

CPU 无 SM 分区概念，无跨 CTA 同步。Mega-Kernel 串行执行：

```
prefill_path → .decode_entry:
  decode_path → sample → stop_check → stream_output → compact_and_refill
  → (如有新 prefill → prefill_path → .decode_entry)
  → LoopEnd → JMP .decode_entry
```

权重预取和 tokenize 在 Mega-Kernel 外部由独立线程完成（不属于推理热路径）。

### §2.6 SM 分区：Cluster-Size 对齐 + 多变体编译

**问题**：SM 分区不沿物理 cluster 边界会导致跨 cluster L2/DSMEM 访问低效。

**方案**：编译时通过 `cudaOccupancyMaxPotentialClusterSize()` 查询硬件实际支持的 cluster 大小，以此为单位对齐 decode/prefill CTA 分配。

```
Cluster-Size 对齐分区:
  // 运行时查询: cluster_size = cudaOccupancyMaxPotentialClusterSize(kernel, ...)
  // 编译时: bake DECODE_CTA_COUNT, PREFILL_CTA_COUNT, CLUSTER_SIZE, NUM_CLUSTERS

  H100 (132 SM, cluster_size=8 portable / 16 opt-in):
    编译变体 A (portable): cluster=8, NUM_CLUSTERS=16
      decode=6×8=48 CTA, prefill=2×8=16 CTA (per-cluster: 6 decode + 2 prefill)
    编译变体 B (opt-in 16): cluster=16, NUM_CLUSTERS=8
      decode=12×16=96 CTA, prefill=4×16=64 CTA (非典型，仅 MoE 密集)
  A100 (108 SM, 无 cluster → grid_sync):
    编译变体: decode=81 CTA, prefill=27 CTA (grid_sync 模式)
  L40 (76 SM):
    编译变体: decode=57 CTA, prefill=19 CTA (grid_sync 模式)
  小卡 < 60 SM: 全部 decode (MK_SERIAL 变体)
```

**Cluster 大小发现**：使用 CUDA Runtime API `cudaOccupancyMaxPotentialClusterSize(&cluster_size, kernel, block_size)`，返回硬件实际支持的最大 cluster CTA 数。无需依赖 GPC 元数据。

**MK_SERIAL 变体说明**：SM<60 的编译变体（prefill 和 decode 在同一组 CTA 内串行执行）是**编译时选择的不同 codegen 路径**（融合算子仍然完整融合），非运行时降级，与 NO_HW_DEGRADATION 铁律一致。

**编译变体选择策略**（编译时 bake，运行时一次选择）：

| 变体名 | 同步模式 | decode:prefill | 适用场景 |
|--------|---------|---------------|---------|
| `MK_CLUSTER_6_2` | cluster.sync + mbarrier | 6:2 per cluster | SM90+ 通用 |
| `MK_CLUSTER_5_3` | cluster.sync + mbarrier | 5:3 per cluster | SM90+ prefill 密集 |
| `MK_GRID_SYNC` | cooperative grid_sync | 75%:25% 总量 | SM80 |
| `MK_SERIAL` | 串行 | 100%:0% | SM<60 |

编译时为每个变体生成独立 `DECODE_CTA_COUNT` + `CLUSTER_SIZE` 常量 bake 进机器码。运行时由 Rust 端根据 batch 特征选择变体（一次 CALL 前选择，非热路径决策）。

**与 StrategyBias 联动**：演进 `12-STRATEGY-ARBITER.md` 的 `StrategyBias`，新增 `decode_cluster_ratio_scale: f32` 字段，由 `GraphArchetype` 调制（MoE 模型偏向 decode → scale > 1.0；Dense 模型平衡 → scale ≈ 1.0）。

**NUMA 多 GPU**：代码对象共享（JIT Cache 存一份编译产物），per-NUMA 数据隔离（各自 BatchContext + KV Pool + RequestQueue）。不违反 jit-cache 单实例铁律——单实例指编译产物，不是数据上下文。

---

## §3 自治连续批调度 (REQ-MKO-003)

### §3.1 Request Queue（设备内存 ring buffer）

```
RequestQueue {
    ring_buffer_ptr: *const u8,    // 设备内存 ring buffer 基址
    ring_capacity: u32,            // 最大请求数
    write_idx_ptr: *mut u64,       // Rust 写入位置（Rust 端 atom.global.add.u64）
    read_idx_ptr: *mut u64,        // Mega-Kernel 读取位置（GPU 端 atom.global.add.u64）
}

RequestQueueEntry {
    input_ids_ptr: *const u32,     // tokenized IDs（Rust tokenizer 写入 pinned memory）
    prompt_len: u32,
    sampling_params: [u32; 4],     // packed: temperature_u32, top_k, top_p_u32, eos_token_id
    max_new_tokens: u32,
    fused_hidden_offset: u32,      // 多模态注入（0=无）
    num_mm_tokens: u32,            // 多模态 token 数
    session_position: u32,         // Session KV 复用（0=新序列）
}
```

Tokenize 在 Rust 端完成（请求级预处理，不属于推理热路径，见 §0 原则 6）。

### §3.2 Page 分配：三层池 + 单页 CAS

PagedAttention 的页表已提供虚拟→物理映射，**无需连续物理页**。每页独立分配。

> **命名约定**：本节三层池命名为 `pool_local` / `pool_cluster` / `pool_global`，避免与 `gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md` 的 L0(本地 GPU HBM) / L1(NVLink 节点内) / L2(RDMA 跨节点) 分布式分页层级冲突。

**三层池设计**（消除 90% 原子开销）：

```
pool_local: Per-CTA Private Pool (32 page slots, register/local memory)
  ├─ 命中：无 atomic 操作（最快路径）
  └─ 空 → 从 pool_cluster 批量取 64 页

pool_cluster: Per-Cluster Shared Pool (DSMEM, SM90+, 1024 page slots)
  ├─ cluster shared atom（cluster 内同步，避免跨 cluster L2 traffic）
  └─ 空 → 从 pool_global 批量取 256 页

pool_global: Global Bitmap (DRAM, 全局)
  ├─ atom.global.exch.b32（一次抢整个 32-bit word = 32 页）
  └─ 失败 → OOM
```

**Fallback 路径**（SM80/SM70 无 DSMEM）：
- pool_local 直接从 pool_global 批量取（`atom.global.exch.b32`）
- 性能差异：SM80 `atom.global.exch` 约 100-200ns vs SM90 DSMEM `atom.shared.cluster` 约 10-20ns
- 竞争退避：每个 CTA 从 `word_idx = hash(cta_id) % bitmap_word_count` 开始扫描，分散热点 word 竞争

```
alloc_single_page():
  // pool_local: Per-CTA private pool（无 atom）
  if cta_private_count > 0:
    cta_private_count -= 1
    return cta_private_pool[cta_private_count]

  // pool_local 空 → 从 pool_cluster 批量补充 (SM90+ DSMEM)
  if use_dsmem:
    claimed = atom.shared.cluster.exch.b32(pool_cluster_count_ptr, 0)
    if claimed > 0:
      transfer = min(claimed, 64)
      refill_cta_private(pool_cluster_pool, transfer)
      atom.shared.cluster.add.b32(pool_cluster_count_ptr, claimed - transfer)
      return alloc_from_private()
    // pool_cluster 空 → 从 pool_global 批量取

  // pool_global: Global bitmap (atom.global.exch 抢整个 word)
  start_word = hash(cta_id) % bitmap_word_count              // 分散竞争
  for offset in 0..bitmap_word_count:
    word_idx = (start_word + offset) % bitmap_word_count
    old_word = atom.global.exch.b32(kv_free_bitmap + word_idx * 4, 0)
    if old_word != 0:
      refill_cta_private_from_word(old_word, word_idx)
      return alloc_from_private()
  return PAGE_ALLOC_OOM

alloc_pages_for_seq(prompt_len):
  pages_needed = ceil(prompt_len / kv_page_size)
  for i in 0..pages_needed:
    page_id = alloc_single_page()
    if page_id == PAGE_ALLOC_OOM:
      // OOM 回滚：直接还 pool_local（同 CTA 同 step，当步可复用）
      for j in 0..i:
        cta_private_pool[cta_private_count++] = allocated[j]
      return OOM
    page_table[i] = page_id
  return OK
```

**释放路径**（延迟批量释放 + Mega-Kernel 退出前 drain）：

```
free_single_page(page_id):
  // 写入 per-CTA pending-free list（无 atom）
  cta_pending_free[cta_pending_free_count++] = page_id

batch_free_at_compact():
  // Compact 阶段（GLOBAL_SYNC B1 前），CTA 0 批量释放
  for cta in 0..total_cta_count:
    for j in 0..cta_pending_free_count[cta]:
      page_id = cta_pending_free[cta][j]
      word_idx = page_id / 32
      bit = page_id % 32
      mask = 1u32 << bit
      atom.global.or.b32(kv_free_bitmap + word_idx * 4, mask)
    cta_pending_free_count[cta] = 0

drain_pools_to_global():
  // Mega-Kernel 退出前（Phase DONE 前），所有 CTA 执行
  // 确保 pool_local + pool_cluster 余量全部归还 pool_global
  // 任何时刻所有页 ∈ pool_local ∪ pool_cluster ∪ pool_global ∪ in-use seq_meta

  // 1. pool_local 余量归还 pool_cluster (SM90+) 或直接归还 pool_global
  while cta_private_count > 0:
    page_id = cta_private_pool[--cta_private_count]
    if use_dsmem:
      // 推入 pool_cluster DSMEM
      pool_cluster_pool[atom.shared.cluster.add.b32(pool_cluster_count_ptr, 1) - 1] = page_id
    else:
      // 直接归还 pool_global bitmap
      word_idx = page_id / 32
      bit = page_id % 32
      atom.global.or.b32(kv_free_bitmap + word_idx * 4, 1u32 << bit)

  // 2. pool_cluster 余量归还 pool_global (仅 CTA 0 per cluster，避免重复)
  if use_dsmem && local_ctaidx == 0:
    count = atom.shared.cluster.exch.b32(pool_cluster_count_ptr, 0)
    for j in 0..count:
      page_id = pool_cluster_pool[j]
      word_idx = page_id / 32
      bit = page_id % 32
      atom.global.or.b32(kv_free_bitmap + word_idx * 4, 1u32 << bit)

  // 3. pending-free list 也全部归还
  for j in 0..cta_pending_free_count:
    page_id = cta_pending_free[j]
    word_idx = page_id / 32
    bit = page_id % 32
    atom.global.or.b32(kv_free_bitmap + word_idx * 4, 1u32 << bit)
  cta_pending_free_count = 0
```

**完整性不变量**：`总页数 = pool_local 余量 + pool_cluster 余量 + pool_global bitmap 置 0 位数 + in-use seq_meta 持有页数`。Mega-Kernel 退出时 pool_local 和 pool_cluster 余量必须为 0。

**优势**：
- 典型场景 90%+ 命中 pool_local，零 atomic 开销
- OOM 回滚直接还 pool_local，当步可复用，避免假性 OOM 雪崩
- Mega-Kernel 退出前 drain 确保零泄漏
- 释放延迟一个 step（可接受：free 的 page 在下一 step 才被复用）

### §3.3 Compact + Refill（三阶段并行 + CTA 0 Refill）

**三阶段并行 compact**（所有 decode CTA 参与，从 ~10μs 降至 ~1μs）：

```
compact_parallel(batch_ctx):

  // ===== Stage 1: 标记（并行，all decode CTAs）=====
  // 每 CTA 处理 ⌈ping_seq_count / decode_cta_count⌉ 个 seq
  for seq in range(cta_id, ping_seq_count, decode_cta_count):
    meta = seq_meta[ping_seq_offset + seq]
    survivor_flag[seq] = meta.active_flag          // 写 DSMEM/SMEM
    // 回收完成序列的 KV pages（写入 pending-free list，§3.2）
    if meta.active_flag == 0:
      for page in meta.page_table_offset .. meta.page_table_offset + meta.page_table_len:
        free_single_page(page)

  bar.sync                                        // CTA 内 warp 同步（非跨 CTA）

  // ===== Stage 2: Prefix-Scan（warp-level shuffle Hillis-Steele）=====
  // 所有 decode CTA 的 warp 0 并行执行 block-level prefix-sum
  // 256 元素 = 256/32 = 8 warp-iterations + log2(32)=5 shuffle ≈ 40 cycle
  if warp_id == 0:
    // 每个 CTA 处理自己的 survivor_flag 段
    for my_seq in range(cta_id, ping_seq_count, decode_cta_count):
      local_offset = my_seq - cta_id                          // CTA 内偏移
      my_flag = survivor_flag[my_seq]
      // warp-level prefix-sum: __shfl_up_sync 累加
      for stride in [1, 2, 4, 8, 16]:
        neighbor = __shfl_up_sync(0xFFFFFFFF, my_flag, stride)
        if lane_id >= stride: my_flag += neighbor
      // my_flag 现在是 CTA 内前缀和
      if survivor_flag[my_seq]:
        target_offset[my_seq] = my_flag - 1                   // 0-indexed
    // CTA 0 warp 0 汇总各 CTA 的 survivor_count，计算跨 CTA 偏移
    if cta_id == 0:
      per_cta_count[0..decode_cta_count] = 各 CTA warp 0 广播的 survivor_count
      cross_cta_offset = exclusive_prefix_sum(per_cta_count)
      // 修正 target_offset: target_offset[seq] += cross_cta_offset[cta_of_seq]

  cluster.sync                                    // 等 scan 完成

  // ===== Stage 3: 并行搬移（all decode CTAs）=====
  for seq in range(cta_id, ping_seq_count, decode_cta_count):
    if survivor_flag[seq]:
      target = pong_seq_offset + target_offset[seq]
      seq_meta[target] = seq_meta[ping_seq_offset + seq]
      page_table_flat[target * pages_per_seq ..] = page_table_flat[seq * pages_per_seq ..]
      sampling_params[target] = sampling_params[seq]

  // ===== Stage 4: Refill（仅 CTA 0，入队顺序敏感）=====
  if cta_id == 0:
    newly_added = 0
    free_slots = max_batch_size - survivor_count
    for i in 0..free_slots:
      idx = atom.global.add.u64(read_idx_ptr, 1)
      if idx >= ld.global.u64(write_idx_ptr): break  // queue 空
      entry = ring_buffer[idx % ring_capacity]

      rc = alloc_pages_for_seq(entry.prompt_len)
      if rc == OOM: break

      target = pong_seq_offset + survivor_count + newly_added
      seq_meta[target] = SeqMeta {
          prompt_len: entry.prompt_len,
          kv_len: 0,
          active_flag: 1,
          page_table_offset: target * pages_per_seq,
          page_table_len: pages_needed,
          session_position: entry.session_position,
          fused_hidden_offset: entry.fused_hidden_offset,
          num_mm_tokens: entry.num_mm_tokens,
      }
      sampling_params[target] = entry.sampling_params
      atom.global.add.u32(BatchContext + 8, entry.prompt_len)
      newly_added += 1

    // 更新 pong 并 swap
    pong_seq_count = survivor_count + newly_added
    swap(ping_seq_offset, pong_seq_offset)
    swap(ping_seq_count, pong_seq_count)
    step_epoch += 1

    // 批量释放 pending-free pages（§3.2）
    batch_free_at_compact()
```

**DSMEM 开销**：`survivor_flag[max_batch_size]` + `target_offset[max_batch_size]` ≈ max_batch_size × 8B。256 序列 = 2KB，远在 DSMEM 限制内。

### §3.4 与现有调度栈的关系

| 现有 CPU 端模块 | 角色 | 说明 |
|----------------|------|------|
| `PagedScheduler` | 初始 batch 构建 + RequestQueue 入队 | 初始调度由 CPU 完成，运行中由 Mega-Kernel 自治 |
| `HGAL` | 初始 batch 分配策略 | 初始 seq 选择仍由 HGAL 打分 |
| `KvPrefixIndex` | CPU 端 prefix 共享 | Mega-Kernel 内不做 trie 查找 |
| `GlobalMemoryManager` | KV pool 一次性分配 | 加载期分配 KV pool + free bitmap，运行时 Mega-Kernel 从 bitmap 分配，CPU 端不再参与 page 管理 |
| `ContinuousBatcher` | 入队职责保留 | CPU 端仅负责 tokenize + 入队，不再做 compact/refill |

---

## §4 Streaming Output（per-token）(REQ-MKO-004)

### §4.1 设计：Per-CTA Sub-Ring + Doorbell

每个采样的 token 立即写入 **per-CTA sub-ring**（消除全局 atomic 串行化），通过 doorbell 通知 Rust 端。

```
OutputRingBuffer {
    sub_ring_base_ptr: *mut u8,    // sub-ring 数组基址（pinned memory）
    cta_sub_ring_size: u32,        // 每个 sub-ring 容量（条目数）
    num_sub_rings: u32,            // sub-ring 数量 = total_cta_count
    per_cta_doorbell_ptr: *mut u64, // Per-CTA doorbell 数组（Rust 端 per-CTA 阻塞等待）
    epoch_flag_ptr: *mut u32,      // 全局 epoch flag（Rust 端可选的 epoch 粒度消费）
}

// 每个 CTA 独占一个 sub-ring + 一个 doorbell slot，零跨 CTA atomic
// sub_ring[i] = sub_ring_base_ptr + cta_id * cta_sub_ring_size * sizeof(OutputTokenEntry)
// per_cta_doorbell[i] = per_cta_doorbell_ptr + cta_id * sizeof(u64)
// per-CTA write_idx 放在 sub-ring 头部（local write，无 atom）
// cta_sub_ring_size = max_batch_size × max_new_tokens / num_decode_ctas + slack
```

```
OutputTokenEntry {                  // 每个 token 一条，非每序列一条
    seq_id: u32,                    // 序列标识（对应 seq_meta 索引）
    token_id: u32,                  // 采样的 token ID
    is_final: u32,                  // 0=中间 token, 1=EOS/stop/max_tokens
    finish_reason: u32,             // 0=中间, 1=EOS, 2=max_tokens, 3=stop_word
    gen_idx: u32,                   // 该序列已生成 token 计数（Rust 端排序用）
}
```

**Mega-Kernel 写入**（每 CTA 零跨 CTA atomic）：

```
// CTA 内写入（decode CTA 负责 sampling 的 seq）
for seq_idx in range(cta_id, ping_seq_count, decode_cta_count):
  // 写入 per-CTA sub-ring（local write，无 atom）
  local_idx = cta_local_write_idx++
  sub_ring[cta_id][local_idx % cta_sub_ring_size] = OutputTokenEntry {
      seq_id, sampled_token_id, is_final=0, finish_reason=0, gen_idx
  }

// CTA 完成所有 seq 后，更新 per-CTA doorbell（写自己的 slot，无跨 CTA 竞争）
st.global.release.u64 per_cta_doorbell[cta_id] = cta_local_write_idx
```

**Rust 端消费**：

```
// 阻塞等待任一 per-CTA doorbell 更新（cuStreamWaitValue64 per slot，或轮询 epoch）
// 方案 A（推荐）: epoch 粒度 — Mega-Kernel GLOBAL_SYNC B0 后写 epoch_flag，Rust 端等 epoch 变化
wait_for_epoch(epoch_flag_ptr, last_consumed_epoch)
// 扫描所有 per-CTA sub-ring
for cta_id in 0..num_sub_rings:
  entries = read sub_ring[cta_id] since last_read[cta_id]
  // 按 (seq_id, gen_idx) 排序合并 → detokenize → yield
```

**容量计算**：`cta_sub_ring_size = ceil(max_batch_size × max_new_tokens / num_decode_ctas) + 64`（64 条目 slack 防溢出）。Mega-Kernel 不等 Rust 消费，Rust 端必须保证消费速率 ≥ 生产速率（否则 ring overflow 返回错误）。

### §4.2 数据流

```
Rust: tokenize(request) → input_ids → pinned memory
Rust: enqueue(RequestQueue, entry)
Rust: CALL MegaKernelFn(ctx)

Mega-Kernel decode loop 内，每个 decode CTA sample 后:
  // 写入 per-CTA sub-ring（local write，零跨 CTA atomic）
  local_idx = cta_local_write_idx++
  sub_ring[cta_id][local_idx % cta_sub_ring_size] = OutputTokenEntry {
      seq_id, sampled_token_id, is_final=0, finish_reason=0, gen_idx
  }
  // 序列完成时额外写一条 is_final=1 的条目

Mega-Kernel 返回 (RET)

Rust: wait_for_epoch → 扫描 per-CTA sub-rings → 按 (seq_id, gen_idx) 排序 → detokenize → yield
```

**流式支持**：Rust 端在 Mega-Kernel 执行期间即可通过 epoch_flag 阻塞等待消费 sub-rings，实现逐 token 流式输出。Mega-Kernel 不需要等 Rust 消费，sub-ring 容量足够大时不会溢出。

---

## §5 硬件感知编译策略 (REQ-MKO-005)

> **演进自**：`12-STRATEGY-ARBITER.md`; `02-HARDWARE.md` DeviceProfile; `02-ARCHITECTURE.md §13.12`

### §5.1 编译时参数决策

```
DeviceProfile::detect() → compile_params:

prefill_params = FusionParams {
    gemm_tile: match profile {
        SM100  => TileSize(128, 256, 64),    // tcgen05.mma
        SM90   => TileSize(128, 256, 64),    // WGMMA
        SM80   => TileSize(64, 128, 32),     // mma.sync
        SM70   => TileSize(32, 64, 32),      // wmma
        AVX10  => TileSize(16, 64, 64),      // APX + AVX10.2
        AVX512 => TileSize(16, 16, 64),      // AMX tile
        AVX2   => TileSize(6, 64, 256),      // BLIS
        SME2   => TileSize(8, 16, 64),       // ZA outer product
        SVE2   => TileSize(8, 64, 128),      // variable-length SIMD
        NEON   => TileSize(4, 64, 128),      // NEON
    },
    attention: FlashAttention,
    kv_mode: KvWriteFull,
    symdim_m: Symbolic("total_prefill_tokens"),
}

decode_params = FusionParams {
    gemm_tile: match profile {
        SM100  => TileSize(1, 256, 64),
        SM90   => TileSize(1, 256, 64),
        SM80   => TileSize(1, 128, 32),
        SM70   => TileSize(1, 64, 32),
        AVX10  => TileSize(1, 64, 64),
        AVX512 => TileSize(1, 16, 64),
        AVX2   => TileSize(1, 64, 256),
        SME2   => TileSize(1, 16, 64),
        SVE2   => TileSize(1, 64, 128),
        NEON   => TileSize(1, 64, 128),
    },
    attention: IncrementalKvAttention,
    kv_mode: KvReadHistoryWriteOne,
    symdim_m: Symbolic("num_active_seqs"),
    // Decode 带宽优化参数
    kv_pipeline_stages: match profile {
        SM90  => 4,    // 4-stage TMA pipeline（Ping/Pong/Ping2/Pong2）
        SM80  => 2,    // 2-stage cp.async pipeline
        _     => 1,    // 无 pipeline
    },
    use_dsmem_kv_share: match profile {
        SM90  => true, // 同 GPC CTA 复用 KV（省 30% HBM 流量）
        _     => false,
    },
    use_ld_nc: true,                              // ld.global.nc 绕过 L1 一致性开销
    use_tensor_core_gemv: match profile {         // M=1 GEMV 仍用 TC (warp-broadcast B)
        SM80..SM100 => true,
        _           => false,
    },
}

// SM 分区：Cluster-Size 对齐 + 多变体编译（§2.6）
cluster_size = cudaOccupancyMaxPotentialClusterSize(kernel, block_size)  // 运行时查询
sm_partition = {
    cluster_size: u32,           // 8 (portable) 或 16 (opt-in non-portable)
    total_cta_count: u32,        // = sm_count (1 CTA/SM)
    decode_cta_count: u32,       // = total_cta_count × decode_ratio
    prefill_cta_count: u32,      // = total_cta_count - decode_cta_count
    num_clusters: u32,           // = total_cta_count / cluster_size
    decode_per_cluster: u32,     // = decode_cta_count / num_clusters
    prefill_per_cluster: u32,    // = prefill_cta_count / num_clusters
}
compile_variants = [
    (decode_ratio=0.75, cluster_mode="portable"),    // 默认：decode 密集
    (decode_ratio=0.625, cluster_mode="portable"),   // 可选：prefill 密集
    (decode_ratio=1.0, cluster_mode="serial"),       // SM<60: 串行
]
for each variant: bake DECODE_CTA_COUNT, CLUSTER_SIZE, NUM_CLUSTERS, DECODE_PER_CLUSTER
```

### §5.2 运行时 AutoTune（soft 参数）

AutoTune 仅调整 **soft 参数**（通过 BatchContext 字段传递，不需要重编译）。**Hard 参数**（SM 分区比例、tile size、编译时常量）不可运行时调整。

```
Soft 参数（AutoTune 可调）:
  - actual_batch_size: ≤ max_batch_size（编译时上界）
  - KV pool free bitmap（运行时 page 管理）
  - sampling temperature/top_k/top_p（per-seq 可变）

Hard 参数（编译时确定）:
  - DECODE_CTA_COUNT（SM 分区边界）
  - GEMM tile size（prefill/decode 各自的 tile）
  - max_batch_size（影响 buffer layout 上界）
  - KV pool total size（加载期确定）

AutoTune 反馈循环:
  遥测数据（decode_step_latency, prefill_latency, page_alloc_failures）
  → Rust 端分析 → 调整 soft 参数 → 下次 CALL 时生效
  例: actual_batch_size = min(max_batch_size, optimal_batch_from_telemetry)
```

### §5.3 NUMA 感知

```
多 NUMA 节点:
  NUMA 0: CPU 核 0-63 + GPU 0 → Mega-Kernel code object (共享) + per-NUMA BatchContext
  NUMA 1: CPU 核 64-127 + GPU 1 → 同一 code object + per-NUMA BatchContext

代码共享: JIT Cache 存一份编译产物（jit-cache 单实例铁律满足）
数据隔离: 每 NUMA 节点独立的 BatchContext + KV Pool + RequestQueue + OutputRingBuffer

多卡分布式: 显式引用 ../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md（三层分页 L0/L1/L2）
```

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
<a data-xref-id="REQ-DP-001" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-001">REQ-DP-001</a>~<a data-xref-id="REQ-DP-014" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-014">REQ-DP-014</a>
(分布式分页 L0/L1/L2) |
<a data-xref-id="REQ-DP-006" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-006">REQ-DP-006</a>
(P2P 页传输) |
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr 扩展) |
<a data-xref-id="REQ-SMPART-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-SMPART-001">REQ-SMPART-001</a>
<a data-xref-id="REQ-SMPART-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-SMPART-002">REQ-SMPART-002</a>
(SM 分区) |
<a data-xref-id="REQ-FLUX-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-FLUX-001">REQ-FLUX-001</a>
<a data-xref-id="REQ-FLUX-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-FLUX-002">REQ-FLUX-002</a>
(Pipeline 缓冲)
</div>

---

## §6 与已实装子系统的集成 (REQ-MKO-006)

### §6.1 全量子系统矩阵

| 子系统 | ABI 参数 | 触发位置 | prefill_path | decode_path | GLOBAL_SYNC 可见性 | 影响 |
|--------|---------|---------|-------------|-------------|-------------------|------|
| Semantic Gatekeeper | arg 15 `hook_ctx_ptr` | 层循环融合组 CallbackChain | ✅ | ✅ | hook_ctx 写入在 forward pass 内，GLOBAL_SYNC B0 后对所有 CTA 可见 | 不变 |
| Guardrail | arg 20 `callback_table_ptr` | OutputModeDispatch (SPEC/39 §1.3.3) post_node | ✅ | ✅ | callback 返回值在 GLOBAL_SYNC B0 后可见 | 不变 |
| Head Routing | arg 14 `output_mode_selector` | OutputModeDispatch (SPEC/39 §1.3.3) JMP Table | ✅ (6模式) | ✅ (6模式) | 纯 CTA-local 决策，无跨 CTA 状态 | 不变 |
| Intent Recall | arg 14 mode=5 | OutputModeDispatch (SPEC/39 §1.3.3) encode_to_layer | ✅ | — | 纯 CTA-local | 不变 |
| CoT Reasoner | arg 15 hook_ctx | decode 层循环融合组（SM 分区） Step Hook | — | ✅ | hook_ctx 写入在 decode forward 内，GLOBAL_SYNC B0 后可见 | 不变 |
| Intent Tracker | 独立 Mega-Kernel | 独立 CALL | — | — | 独立实例，无共享状态 | 不变 |
| Session KV | arg 17 `session_position` | 层循环融合组 SessionKvRestore | ✅ | ✅ | KV cache 写入在 forward pass 内，GLOBAL_SYNC B0 后可见 | 不变 |
| Multimodal | arg 18-19 | 层循环融合组 MmHiddenInject | ✅ | ✅ | fused hidden 写入在 forward pass 内，GLOBAL_SYNC B0 后可见 | 不变 |
| Speculative Decoding | 08-EXECUTOR spec_state | decode 层循环融合组（SM 分区） verify | — | ✅ | 纯 CTA-local | 不变 |
| Early Exit | 08-EXECUTOR early_exit | 层循环融合组 | ✅ | ✅ | 纯 CTA-local | 不变 |
| MoE Hot-Patch | 08-EXECUTOR moe.hot_patch | 层循环融合组 FFN | ✅ | ✅ | weight_patch 写入在 forward pass 内，GLOBAL_SYNC B0 后可见 | 不变 |
| Page Compression | SPEC/22 codec | KV page 读写 | ✅ | ✅ | page 数据在 GLOBAL_SYNC B0 后一致 | 不变 |
| Weight Paging | SPEC/21 WeightTierManager | 层循环融合组 权重加载 | ✅ | ✅ | weight_page 在 GLOBAL_SYNC B0 后可见 | 不变 |
| KV Optimization | SPEC/19 importance | 层循环融合组 KV 访问 | ✅ | ✅ | importance score 在 GLOBAL_SYNC B0 后可见 | 不变 |

**不变性保证**：ForwardPhaseDispatch (SPEC/39 §1.3.3) 只在 forward pass **入口**分流，两条路径各自包含完整的层循环融合组 + OutputModeDispatch (SPEC/39 §1.3.3)。所有子系统的触发点在 forward pass 内部，不受入口分流影响。

### §6.2 BatchContext 扩展字段

新增字段追加在 **seq_meta 数组之后**（不侵入 SPEC/20 现有布局）：

```
BatchContext (SPEC/20 SSOT):
  offset 0-79:  header (12 个指针/标量字段)     // 不变
  offset 80:    seq_meta_base                    // 不变，指向 seq_meta 数组起始
  seq_meta 数组: max_batch_size × SEQ_META_STRIDE (56 bytes each)

新增扩展区 (offset = 80 + max_batch_size × 56):
  +0   request_queue_ptr         *const u8    // RequestQueue 基址（§3.1）
  +8   output_ring_ptr           *const u8    // OutputRingBuffer sub-ring 基址（§4.1）
  +16  kv_free_bitmap_ptr        *mut u32     // KV page 全局空闲 bitmap（§3.2 L2）
  +24  kv_pool_total_pages       u32          // KV pool 总页数
  +28  max_batch_size            u32          // 最大同时活跃序列数（编译时上界）
  +32  [DualBatchMeta, 24 bytes]              // §2.3（含 epoch_arrival_count）
  +56  autotune_actual_batch     u32          // AutoTune soft 参数：实际 batch size
  +60  page_alloc_pool_cluster_dsmem_ptr *mut u8  // 三层池 pool_cluster DSMEM 基址（SM90+，§3.2）
  +68  pending_free_list_ptr     *mut u32     // Per-CTA pending-free list 基址（§3.2）
  +76  pending_free_count_ptr    *mut u32     // Per-CTA pending-free 计数基址
  +80  output_per_cta_doorbell_ptr *mut u64   // Per-CTA doorbell 数组（§4.1，每 CTA 一个 u64 slot）
  +88  output_epoch_flag_ptr     *mut u32     // Output epoch flag（§4.1，Rust 端 epoch 粒度消费）
  +92  [reserved, 4 bytes]                    // 对齐到 96
```

**SEQ_META_STRIDE**：SPEC/20 定义为 56 bytes（14 × u32），非 64。本 SPEC 遵循此定义。

---

## §7 REQ 清单

| REQ ID | 描述 | 依赖 | 验收标准 |
|--------|------|------|---------|
| REQ-MKO-001 | Prefill/Decode JMP Table 差异化（ForwardPhaseDispatch (SPEC/39 §1.3.3) + batch mode only + Chunked Prefill 混合路径 + prefill→decode→refill→prefill 循环 + SharedKvRef 两阶段 prefill 加速） | 无 | 单 Mega-Kernel 内 prefill_path/decode_path/mixed_path 三条路径各自 GEMM tile 不同；prefill 后 JMP decode 无重编译；legacy 单序列不受影响；默认走 mixed_path 消除路径切换抖动；SharedKvRef 模型 prefill 分 donor pass + consumer pass 两阶段执行（SPEC 03 §1.3.2），consumer pass 跳过 K/V GEMM 省 ~30% prefill FLOPs |
| REQ-MKO-002 | SM 分区 Ping-Pong（thread role + cluster-size 对齐 CTA 分区 + 多级同步 cluster.sync/mbarrier/grid_sync/ring-barrier + DualBatchMeta + 多变体编译） | MKO-001 | SM90+ decode∥prefill 用 cluster.sync(cluster 内) + mbarrier(跨 cluster) + __threadfence_system()；SM80 用 cooperative grid_sync；SM<60 编译时生成 MK_SERIAL 变体（非运行时降级）；cluster size 通过 cudaOccupancyMaxPotentialClusterSize 查询（portable=8/opt-in=16）；GLOBAL_SYNC 保证 seq_meta 无 data race |
| REQ-MKO-003 | 自治连续批调度（RequestQueue ring buffer + 三层池 page allocator pool_local/pool_cluster/pool_global + warp-level prefix-scan 并行 compact + CTA 0 refill + 退出前 drain） | MKO-002 | Mega-Kernel decode loop 内自治 page alloc/free；pool_local 命中率 > 90%；OOM 回滚直接还 pool_local（当步可复用）；Mega-Kernel 退出前 drain 确保零泄漏；compact warp-level prefix-scan ≈ 40 cycle |
| REQ-MKO-004 | Streaming Output（per-token OutputTokenEntry + per-CTA sub-ring + per-CTA doorbell array + epoch flag + Rust 端流式消费） | MKO-003 | 每 token 写入 per-CTA sub-ring 零全局 atomic；per-CTA doorbell slot 消除跨 CTA 竞争；Rust 端通过 epoch_flag 阻塞等待消费；支持 SSE/WebSocket 逐 token yield |
| REQ-MKO-005 | 硬件感知编译（12 Profile × prefill/decode tile + cluster-size 对齐 SM 分区 + decode 带 4-stage TMA pipeline + DSMEM KV 共享 + soft AutoTune + NUMA code 共享） | MKO-001 | DeviceProfile 驱动编译时参数选择；SM90+ decode 使用 4-stage TMA pipeline + DSMEM KV 共享；AutoTune 仅调 soft 参数；NUMA 代码共享不违反 jit-cache 铁律 |
| REQ-MKO-006 | 全量子系统集成（14 子系统 prefill/decode 路径 + GLOBAL_SYNC(含 __threadfence_system) 可见性验证 + BatchContext 尾部扩展 + seq_meta 不侵入 + SEQ_META_STRIDE=56 回写 SPEC/20） | MKO-001 | 所有已实装子系统在 prefill_path 和 decode_path 下行为不变；子系统共享内存在 GLOBAL_SYNC(含 __threadfence_system) 后全局可见；BatchContext 扩展区在 seq_meta 数组之后 |
