# 统一编译器架构 (SSOT)

> **核心定位**: gllm-kernels 是一个**通用编译器**，不是模型推理框架。编译器不假设图的结构——喂什么编译什么。1 个 op 的 Gather 图和 35 层 decoder 图走同一管线，产出同一 ABI 的 JIT 机器码。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
统一编译器与通信指令融合:
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr 扩展: RemotePageLookup/P2pPageFetch 等通过本文件 compile() 统一管线编译) |
<a data-xref-id="REQ-DP-011" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-011">REQ-DP-011</a>
(CommInstr 扩展: gllm-nccl CommInstr→VmInstr 映射后接入本文件 Phase 3 lowering) |
<a data-xref-id="REQ-SMPART-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-SMPART-002">REQ-SMPART-002</a>
(SM 分区配置算法: 通信 SM 分配通过本文件 ResourcePlanner 统一预算规划)
</div>

## §0 设计宪法

**编译器 = 喂什么编译什么。**

这条原则贯穿所有设计决策。任何违反此原则的现状（代码或 SPEC）都是 bug，必须修正。

推论：
- 编译器不知道"encoder"和"decoder"的区别——它只看到一个 CompilerGraph
- 编译器不知道"embed 阶段"和"lm_head 阶段"——它只看到 ops 的拓扑排序
- 编译器不知道"采样循环"——它只看到图中有没有 Argmax/StoreToken/CheckStopCondition
- 编译器不知道"权重布局是 embed→layers→lm_head"——它只看到 named_offsets 映射表
- 模型形态差异 100% 由 CompilerGraph 拓扑编码，编译器零假设

## §1 编译管线

```
CompilerGraph (任意拓扑)
  ↓
compile(graph, config)
  ↓
[Phase 0: Scalar → SymExec]
[Phase 1: SemanticDAG]
[Phase 2: Fusion + HW + VTM + LayoutNegotiate + BufferAlloc]
[Phase 3: VmInstr emit → RegAlloc → ISA Lowering]
  ↓
CompiledLayer (MegaKernelFn ABI)
```

### §1.1 唯一编译入口

```rust
impl InferenceCompiler {
    pub fn compile(
        &mut self,
        graph: CompilerGraph,
        config: &CompileConfig,
    ) -> Result<CompileOutput, InferenceError>;
}
```

- **一个函数**。不存在 `compile_graph`、`compile_mega_kernel_from_graph`、`compile_forward` 等独立路径
- **任意 CompilerGraph**：1 个 Gather op 的 SG 小图、RmsNorm+Gemm 的 K 投影图、完整 35 层 decoder 图——全部走这个入口
- `CompileConfig` 携带编译器需要的**非图推导**参数（max_seq_len、output_modes、SG/hook 配置等），但**不携带任何图结构假设**

### §1.1.1 CompileConfig 定义

```rust
/// 编译配置 — 携带编译器需要的非图推导参数。
///
/// 设计原则：CompileConfig 不携带任何图结构假设。
/// 编译器从 CompilerGraph 推导一切结构信息。
/// CompileConfig 只提供"图本身无法推导的外部输入"。
pub struct CompileConfig {
    // ── 内存上界 ──
    /// Buffer 分配上界（替代 SYMDIM_MAX_SEQ_LEN 硬编码）。
    /// 来自模型配置 max_position_embeddings，不是编译器硬编码。
    /// 仅用于 scratchpad/KV buffer 分配大小，不影响循环 bound。
    pub max_seq_len: usize,

    // ── 权重寻址 ──
    /// 权重偏移映射表（通用路径）。
    /// JIT 机器码通过 weight_ptr + named_offsets["layer.0"] 访问权重。
    pub named_offsets: HashMap<String, usize>,

    // ── 采样/输出配置 ──
    /// EOS token ID 数量。从模型配置推导，非硬编码。
    /// 图无 Argmax 算子时此字段忽略（encoder/小图）。
    pub num_eos_tokens: usize,

    /// 业务配置（output modes、guardrail、SG、intent、CoT）。
    /// 图无对应算子时这些字段忽略。
    pub business_config: MegaKernelBusinessConfig,

    // ── 异构层配置 ──
    /// 异构层布局（Gemma 4 E2B 等）。None = 均匀层。
    /// 编译器通过 GprCondAction 在 JIT 内条件分支，不由 CompileConfig 驱动。
    pub hetero: Option<HeteroLayerConfig>,

    // ── 调试 ──
    /// 启用 JIT 调试输出。
    pub debug_jit: bool,

    // ── 设备能力（GPU 编译路径） ──
    /// CUDA SM 版本（如 80, 90, 100）。None = CPU-only 编译。
    /// 从 backend.device_info().sm_version() 获取 (SPEC/15 REQ-GPU-010)。
    pub sm_version: Option<u32>,

    /// AMD GPU 架构标识 + wave 大小。None = 非 HIP 编译。
    /// 从 HipBackend.gpu_profile 获取 (SPEC/15 REQ-GPU-010)。
    pub gfx_arch: Option<(String, u32)>,

    /// Apple Metal GPU 配置。None = 非 Metal 编译。
    /// 从 MetalBackend.gpu_profile 获取 (SPEC/15 REQ-GPU-010)。
    pub metal_profile: Option<MetalGpuProfile>,
}
```

**字段来源规则**（ARCH-DATA-FLOW-CONTRACT 兼容）：

| 字段 | 来源 | 禁止 |
|------|------|------|
| `max_seq_len` | 模型配置 `max_position_embeddings` | 编译器硬编码 2048 |
| `named_offsets` | `CompilerGraph.weight_layout()` | 假设 embed→layers→lm_head 顺序 |
| `num_eos_tokens` | 模型配置 `eos_token_id` 数量 | 硬编码 1 |
| `business_config` | Client API 配置 | 默认值 fallback |
| `hetero` | 模型配置层结构分析 | 编译器假设层类型 |
| `debug_jit` | 调用方传入 | 总是 true/false |
| `sm_version` | backend.device_info().sm_version() | 编译器硬编码 SM 版本 |
| `gfx_arch` | HipBackend.gpu_profile | 编译器硬编码 GPU 架构 |
| `metal_profile` | MetalBackend.gpu_profile | 编译器硬编码 Metal 配置 |

**不存在的字段**：
- ❌ `is_encoder: bool` — 编译器不区分模型形态，图拓扑自动推导
- ❌ `is_decoder: bool` — 同上
- ❌ `inference_mode: InferenceMode` — 同上
- ❌ `vocab_size: usize` — 从图的 lm_head OpKind 推导（`OpKind::Gemm { n }`）
- ❌ `num_layers: usize` — 从图的层循环结构推导
- ❌ `hidden: usize` — 从图的 tensor shape 推导

### §1.2 Phase 0-2：图结构无关

Phase 0（Scalar + SymExec）、Phase 1（SemanticDAG）、Phase 2（Fusion + HW + BufferAlloc）**不假设图的结构**。它们分析的是 op 语义、数据依赖、融合机会、buffer 生命周期——这些概念对 1-op 图和 100-op 图完全相同。

当前实现已满足此要求。两条旧路径（`compile_graph` vs `compile_mega_kernel_from_graph`）的 Phase 0-2 代码完全相同，证明了这一点。

### §1.3 Phase 3 必须图结构无关 — 详细重构规范

**当前（违反设计）**：`compile_mega_kernel_vm()` 硬编码 Phase 0-8：
```
Phase 0: prologue + weight_ptr + scratchpad
Phase 1: embed (Gather op)
Phase 2-3: prefill (N 层循环)
Phase 4-5: decode (N 层循环 + 采样)
Phase 6: Argmax + StoreToken
Phase 7: CheckStopCondition + 循环回边
Phase 8: epilogue + ret
```
→ 编译器**假设**图一定有 embed→layers→lm_head→sampling→token_store→EOS_check 结构。
→ encoder/SG 小图不符合此假设 → 需要第二条路径 `compile_graph()`。

**目标（正确设计）**：编译器不假设图结构，从 `plan.groups` 迭代发射：
```
Phase 0: prologue（栈帧、weight_ptr、scratchpad_ptr、output_ptr）
Phase 0.5: ForwardPhaseDispatch（仅多步生成图存在）
for group in plan.groups:
    emit_group(group)  ← 图有什么算子就发射什么
Phase N: epilogue + ret
```

#### §1.3.1 通用 emit_group 规范

```rust
/// 发射一个融合组的 VmInstr。
/// 组内所有 op 已在 Phase 2 融合决策中确定。
/// emit_group 不关心组属于"第几层"或"什么 Phase"——
/// 它只关心组内的 op 类型和融合模式。
fn emit_group(prog: &mut VmProgram, group: &FusionGroup, ctx: &GroupEmitContext) {
    match group.fusion_mode {
        FusionMode::GemmEpilogue => emit_gemm_with_epilogue(prog, group, ctx),
        FusionMode::LoopFusion => emit_elementwise_fused(prog, group, ctx),
        FusionMode::TileLevelFusion => emit_tile_fused(prog, group, ctx),
        FusionMode::ComputeRoot => emit_compute_root(prog, group, ctx),
        FusionMode::Standalone => emit_standalone(prog, group, ctx),
    }
}
```

**关键约束**：
- `emit_group` 不引用 "layer_idx"、"phase_id" 等位置概念
- 组内 op 通过 `GroupEmitContext` 获取输入/输出 VReg 和指针偏移
- 层循环由 `plan.groups` 中的 `GroupMarker` 触发，不在 emit_group 内部硬编码

#### §1.3.2 GroupMark 触发机制

编译器需要知道"哪些组属于同一层循环"来插入 `LayerLoopBegin`/`LayerLoopEnd`。
这不是硬编码——而是从图的**层结构**推导：

```rust
/// Phase 2 融合引擎在组序列中插入结构标记。
/// 标记位置由 CompilerGraph 的层拓扑推导，不由编译器假设。
pub enum GroupMarker {
    /// 层循环开始。num_iterations 从图拓扑推导。
    LayerLoopBegin { num_iterations: usize },
    /// 层循环结束。
    LayerLoopEnd,
    /// ForwardPhaseDispatch 三路分支（仅多步生成图）。
    PhaseDispatch,
    /// 无标记——普通融合组。
    None,
}
```

**推导规则**：
1. 图中连续 N 个同构子结构（相同 op 模式重复） → `LayerLoopBegin { num_iterations: N }`
2. 单次出现的子结构 → 无循环标记，直接发射
3. 图无 Argmax/StoreToken/CheckStopCondition → 无 PhaseDispatch
4. 图有 Argmax → PhaseDispatch 插入在首次层循环前

**不同图类型的标记序列**：

| 图类型 | Group 序列 | 标记 |
|--------|-----------|------|
| 35 层 decoder | embed + N×[attn+ffn] + lm_head + argmax + store + stop_check | `PhaseDispatch` + `LayerLoopBegin{35}` + `LayerLoopEnd` |
| 12 层 encoder | embed + 12×[attn+ffn] + pool/classify | `LayerLoopBegin{12}` + `LayerLoopEnd` |
| 24 层 reranker | embed + 24×[attn+ffn] + classify + score | `LayerLoopBegin{24}` + `LayerLoopEnd` |
| SG K-投影图 | rmsnorm + gemm | 无标记 |
| SG Gather 图 | gather | 无标记 |
| Gemma 4 E2B | embed + N×[altup+attn+ffn_sliding] + M×[altup+attn+ffn_global] + lm_head + ... | `PhaseDispatch` + `LayerLoopBegin{N}` + `LayerLoopEnd` + `LayerLoopBegin{M}` + `LayerLoopEnd` |
| MoE decoder | embed + N×[attn+moe_router+moe_ffn] + lm_head + argmax + store + stop_check | `PhaseDispatch` + `LayerLoopBegin{N}` + `LayerLoopEnd` |
| DeepSeek V3 (MLA+MoE) | embed + N×[mla_attn+moe_router+moe_ffn+shared_experts] + lm_head + mtp + argmax + stop_check | `PhaseDispatch` + `LayerLoopBegin{N}` + `LayerLoopEnd` + MTP 融合组 |

#### §1.3.3 ForwardPhaseDispatch 详细行为

`ForwardPhaseDispatch` 是 JIT 内的三路分支，由 `phase_id` 寄存器控制：

```
JIT 内部:
  mov phase_id, [rbp + phase_offset]     ; ABI 参数传入
  cmp phase_id, 0
  je prefill_path
  cmp phase_id, 1
  je decode_path
  ; else: encoder_path (直接 forward)

prefill_path:
  <emit prefill groups>
  jmp after_layer_loop

decode_path:
  <emit decode groups + sampling + token_store + stop_check>
  jmp loop_back_or_exit

encoder_path:
  <emit encoder groups (pool/classify/rerank)>
  jmp epilogue
```

**关键**：三路分支的代码存在性由图拓扑决定：
- 图无 Argmax → 无 decode_path 分支
- 图无 Sampling ops → 无 decode_path
- 图只有小图 ops → 无 PhaseDispatch，线性执行

#### §1.3.4 废弃的硬编码 Phase 编号

| 废弃编号 | 原含义 | 替代机制 |
|---------|--------|---------|
| Phase 0 | prologue | 保留（所有图都需要） |
| Phase 0.5 | (无) | `PhaseDispatch`（条件性，仅多步生成图） |
| Phase 1 | embed Gather | `plan.groups` 中第一个融合组（无特殊编号） |
| Phase 2-3 | prefill layers | `LayerLoopBegin` + `emit_group` 迭代 |
| Phase 4-5 | decode layers | 同上（PhaseDispatch 分支后） |
| Phase 6 | Argmax/StoreToken | 融合组中的 Standalone op |
| Phase 7 | CheckStopCondition + loop back | 融合组 + 回边 JMP 指令 |
| Phase 8 | epilogue | 保留（所有图都需要） |

**设计要点**：Phase 0 和 Phase N（prologue/epilogue）保留，因为它们是 JIT 函数帧的固定结构，与图内容无关。中间的 Phase 1-7 被通用 `for group in plan.groups` 迭代 + `GroupMarker` 触发替代。

### §1.4 模型拓扑 → JIT 机器码的映射规则

编译器不知道模型形态，但编译产物的行为由图拓扑天然决定：

| 图拓扑特征 | JIT 机器码自然包含 | JIT 机器码自然不包含 |
|-----------|-------------------|---------------------|
| 有 N 个 decoder 层 + KV ops | 层循环 + KV cache 读写 | — |
| 有 Argmax + StoreToken + CheckStopCondition | 采样 + token 存储 + EOS 检查 + 生成循环 | — |
| 无 Argmax/StoreToken/CheckStopCondition | — | 采样、token 存储、生成循环 |
| 有 MeanPool/Classify 输出 op | encoder 输出逻辑 | 采样、生成循环 |
| 有 Gather(embed_weight) | Embedding lookup | — |
| 无 Gather(embed_weight)（如 SG 小图） | — | Embedding lookup |
| 有 RoPE ops | RoPE cache 预计算 | — |
| 有 MtpConfig + 额外投影 | MTP 候选 token 生成 | — |

**不存在的概念**："encoder 模式"、"decoder 模式"、"forward-only"、"独立编译路径"。

### §1.5 辅助小图（SG Level Keys 预计算、Vision/Audio 编码）

SG 的 `EmbedLookupOnlyGraph`（Gather-only）、`KProjOnlyGraph`（RmsNorm+Gemm）等辅助图也通过同一个 `compile()` 入口编译，使用同一个 MegaKernelFn ABI。

调用时：
- `input_ids_ptr` → 指向索引数据（u32 token IDs 或 u32 编码的索引）
- `weight_blob_ptr` → 指向小图独立的权重 blob
- `kv_cache_ptr` → NULL
- `max_new_tokens` → 0
- `output_mode_selector` → 0（Generate，但不影响无采样图的执行）
- 其余采样/KV/session 参数 → NULL/0

小图编译产物与完整模型编译产物拥有相同的 ABI 签名、相同的 prologue/epilogue 结构、相同的 `CompiledLayer` 存储。区别仅在于图里的 ops 不同 → JIT 机器码不同。

## §2 ABI 统一

### §2.1 唯一函数签名：MegaKernelFn

所有编译产物（1-op 小图、encoder、decoder、vision、audio）使用 `MegaKernelFn` ABI：

```rust
pub type MegaKernelFn = unsafe extern "C" fn(
    *const u32,   // arg 0:  input_ids_ptr
    *const u8,    // arg 1:  weight_blob_ptr
    *mut u8,      // arg 2:  kv_cache_ptr
    *const u32,   // arg 3:  positions_ptr
    *const u8,    // arg 4:  aux_ptr
    usize,        // arg 5:  batch_size
    usize,        // arg 6:  prompt_len
    *mut u8,      // arg 7:  scratchpad_ptr
    *mut u32,     // arg 8:  output_tokens_ptr
    usize,        // arg 9:  temperature_u32
    usize,        // arg 10: top_k
    usize,        // arg 11: top_p_u32
    usize,        // arg 12: max_new_tokens
    usize,        // arg 13: eos_token_id
    usize,        // arg 14: output_mode_selector
    *const u8,    // arg 15: hook_ctx_ptr
    *mut u8,      // arg 16: telemetry_ptr
    usize,        // arg 17: session_position
    *const u8,    // arg 18: fused_hidden_ptr
    usize,        // arg 19: num_mm_tokens
    *const u8,    // arg 20: callback_table_ptr
    *const u32,   // arg 21: page_table_ptr
    *const u8,    // arg 22: batch_ctx_ptr
) -> usize;
```

### §2.2 参数语义：图拓扑决定用途

编译器 prologue 从 ABI 参数加载通用指针和值。具体哪个参数被使用、如何使用，由图拓扑决定：

| ABI 参数 | Decoder 图 | Encoder 图 | SG 小图 (Gather) | SG 小图 (Norm+Gemm) |
|----------|-----------|-----------|-----------------|-------------------|
| `input_ids_ptr` | prompt token IDs | 输入 token IDs | 索引数据 (u32) | 输入激活 (reinterpret) |
| `weight_blob_ptr` | 模型完整权重 | 模型完整权重 | embed table 权重 | norm+gemm 权重 |
| `kv_cache_ptr` | KV buffer | NULL | NULL | NULL |
| `positions_ptr` | position IDs | position IDs | NULL | NULL |
| `scratchpad_ptr` | JIT 临时空间 | JIT 临时空间 | JIT 临时空间 | JIT 临时空间 |
| `prompt_len` | prompt 长度 | 全输入长度 | 索引数量 | seq_len |
| `max_new_tokens` | ≥1 | 0 | 0 | 0 |
| `output_tokens_ptr` | 输出 token 缓冲 | NULL | NULL | NULL |
| `output_mode_selector` | 0 (Generate) | 3 (Encode) | 0 | 0 |
| 采样参数 (9-11) | 温度/Top-K/P | 0 | 0 | 0 |
| 其余参数 | 按需 | NULL/0 | NULL/0 | NULL/0 |

### §2.3 不存在的 ABI

`CompiledLayerFn`（10-param ABI）已物理删除。不存在任何 10 参数编译产物。

## §3 权重布局

### §3.1 通用权重布局：named_offsets 映射表

编译器不假设权重是"embed→layers→lm_head"布局。权重布局由 `CompilerGraph.weight_layout()` 返回的 `named_offsets` 映射表描述：

```
named_offsets: HashMap<String, usize>
  "embed"     → 0
  "layer.0"   → 262144
  "layer.1"   → 524288
  ...
  "lm_head"   → 8388608
```

JIT 机器码通过 `weight_ptr + named_offsets["layer.0"]` 访问权重。偏移值 bake 为立即数。

**对于小图**：`named_offsets` 只包含小图需要的权重条目（如 `"sg_embed_table" → 0`）。

### §3.2 模型级权重布局（MegaKernelWeightLayout）

完整模型编译时，`CompileOutput` 额外提供 `MegaKernelWeightLayout` 作为 pack 阶段的约定格式：

```
embed_offset → layer_0_offset → [layer_stride × N] → final_norm_offset → lm_head_offset
```

这是**pack 阶段的便利结构**，不是编译器的假设。编译器只读 `named_offsets`，不读 `MegaKernelWeightLayout`。

### §3.3 异构权重布局（HeteroWeightLayout）

Gemma 4 E2B 等模型有 4 种层类型（sliding_small/full_small/sliding_large/full_large），每层权重大小不同。`HeteroWeightLayout` 描述 per-segment per-type 的权重 stride。

这同样是**pack 阶段的便利结构**。编译器通过 `GprCondAction` 在 JIT 内按层索引条件跳过/执行不同层类型的 ops，权重偏移由 pack 阶段写入 `named_offsets`。

## §4 输出重定向

### §4.1 图输出 → scratchpad output region

编译器将 `CompilerGraph.outputs[0]` 重定向到 `scratchpad + logits_scratch_offset`（紧接在所有中间 tensor 和 RoPE cache 之后）。调用方从该区域读取结果。

- Decoder：lm_head 输出 logits → scratchpad 中间区域（prefill）或直接采样
- Encoder：MeanPool/Classify/Rerank 输出 → scratchpad 输出区域，调用方读取
- SG 小图：Gather/Gemm 输出 → scratchpad 输出区域，调用方读取

### §4.2 output_tokens_ptr 仅用于 token 生成

`output_tokens_ptr`（ABI arg 8）只在图包含 `StoreToken` 算子时被使用。encoder/小图传 NULL，JIT 不访问该参数。

## §5 编译产物

### §5.1 CompiledLayer

所有编译产物统一存储为 `CompiledLayer`：

```rust
pub struct CompiledLayer {
    code: ExecutableBuffer,          // JIT 机器码 (mmap executable)
    entry_offset: usize,             // 入口偏移
    pub scratchpad_bytes: usize,     // 所需 scratchpad 大小
    pub config_hash: u64,            // 缓存校验 hash
    pub weight_layout: Option<WeightLayout>,  // 权重映射 (可省)
    pub rope_cache: Option<RopeCacheRequirement>, // RoPE 预计算需求 (可省)
    hotpatch_registry: Option<HotPatchRegistry>, // 热修补注册 (可省)
}
```

`CompiledLayer` 是编译器唯一产出类型。不存在 `CompiledModel`、`CompiledSubGraph` 等变体。

### §5.2 CompileOutput

完整模型编译产出 `CompileOutput`（包含 `CompiledLayer` + 额外元数据）：

```rust
pub struct CompileOutput {
    pub layer_code: CompiledLayer,
    pub weight_layout: MegaKernelWeightLayout,   // pack 阶段便利结构
    pub buffer_layout: MegaKernelBufferLayout,    // scratchpad 布局
    pub num_layers: usize,
    pub vocab_size: usize,
    pub hidden: usize,
    pub rope_cache: Option<RopeCacheRequirement>,
    pub total_scratchpad_bytes: usize,
    pub logits_scratch_offset: usize,
    pub hetero_layout: Option<HeteroWeightLayout>,
    pub source_map: Option<JitSourceMap>,
}
```

**小图编译**产出 `CompiledLayer`（不含额外元数据），因为小图不需要 `weight_layout`/`buffer_layout` 等模型级信息。

### §5.3 入口获取

```rust
impl CompiledLayer {
    /// 唯一入口获取方式：MegaKernelFn ABI
    pub unsafe fn entry_fn(&self) -> MegaKernelFn;
}
```

`entry_point()`（返回 `CompiledLayerFn`）和 `execute()`（10-param 调用）已物理删除。

## §6 废弃清单

以下 API 为目标架构中应删除的废弃项。标注 `[已删除]` 的已从代码库物理删除；标注 `[待删除]` 的仍存在于代码库中，需在后续代码清理阶段删除：

| 废弃项 | 原位置 | 替代 | 废弃原因 | 状态 |
|--------|--------|------|---------|------|
| `CompiledLayerFn` 类型 | `executable.rs` | `MegaKernelFn` | 10-param ABI 与统一 23-param ABI 矛盾 | [待删除] |
| `compile_graph()` | `mod.rs` | `compile()` | 独立编译路径违反"唯一入口" | [待删除] |
| `compile_graph_with_quant()` | `mod.rs` | `compile()` | 同上 | [已删除] |
| `compile_graph_to_gpu()` | `mod.rs` | `compile()` (GPU 变体) | 同上 | [已删除] |
| `compile_mega_kernel_from_graph()` | `mod.rs` | `compile()` | 合并为统一入口 | [待删除] |
| `compile_forward_from_graph()` | `executor_core.inc.rs` | `compile()` | encoder 独立编译路径 | [已删除] |
| `execute_forward()` | `executor_core.inc.rs` | `generate_single_sequence()` | encoder 独立执行路径 | [已删除] |
| `forward_compiled` 字段 | `executor_core.inc.rs` | 统一的 `compiled` 字段 | 第二编译产物存储 | [已删除] |
| `CompiledLayer::entry_point()` | `executable.rs` | `entry_fn()` | 返回废弃的 CompiledLayerFn | [待删除] |
| `CompiledLayer::execute()` | `executable.rs` | 直接调用 `entry_fn()` | 10-param 调用封装 | [待删除] |
| `X86CodeGen::emit_plan()` | `emitter.rs` | `compile_mega_kernel_vm` 管线 | 10-param prologue 发射 | [待删除] |
| `SymDimSlotMap::default_abi()` | `plan_lower/context.inc.rs` | `SymDimSlotMap::mega_kernel_abi()` | 10-param ABI 映射 | [待删除] |
| `COMPILED_LAYER_PARAMS` | `vm_state.rs` | `MEGA_KERNEL_PARAMS` | 10-param 参数名表 | [待删除] |
| `is_encoder` 编译参数 | `compile_mega_kernel_vm` | 图拓扑自动推导 | 编译器不假设模型形态 | [待删除] — 代码中已不再分支（`let _ = is_encoder`），但参数仍存在 |
| `MegaKernelWeightLayout` 硬编码结构 | `mega_kernel_abi.rs` | `named_offsets` + 可选便利结构 | 编译器不假设 embed→layers→lm_head | [待删除] — 仅 pack/observe 层使用，编译器不引用 |

## §7 REQ 清单

| ID | 要求 | 验证 |
|----|------|------|
| REQ-UMK-001 | `compile()` 是 `InferenceCompiler` 的唯一公开编译方法 | `grep -rn "compile_graph\|compile_forward\|compile_mega_kernel_from_graph" src/compiler/` 无结果（除 `compile()` 实现内部） |
| REQ-UMK-002 | 所有编译产物使用 `MegaKernelFn` ABI | `entry_fn()` 类型统一为 `MegaKernelFn`，不存在 `CompiledLayerFn` 引用 |
| REQ-UMK-003 | 编译器不假设图结构 | `compile_mega_kernel_vm` 无硬编码 Phase 编号、无 `is_encoder` 参数、无 `lm_head` 搜索、无 embed→layers→lm_head 假设 |
| REQ-UMK-004 | 小图（SG/Vision/Audio 辅助图）与完整模型走同一 `compile()` 入口 | `grep -rn "compile_graph\|InferenceCompiler::new.*compile_graph" src/semantic_gatekeeper/ src/compat/` 无结果 |
| REQ-UMK-005 | encoder 模型走同一编译管线，无 `is_encoder` 分支 | `grep -rn "is_encoder" src/compiler/codegen/vm/mega_kernel_emit.rs` 无结果；图拓扑不含 Argmax/StoreToken → JIT 自然无采样循环 |
| REQ-UMK-006 | encoder 输出通过 scratchpad output tensor 读取 | `grep -rn "logits_scratch_offset\|output_tensor" src/engine/mega_kernel/` 有结果；encoder 调用后从 `scratchpad + logits_scratch_offset` 读取结果 |
| REQ-UMK-007 | N 个模型 = N 个独立编译实例 | `grep -rn "entry_fn\|compiled:" src/engine/mega_kernel/executor_core.inc.rs` 每实例独立 `entry_fn` |
| REQ-UMK-008 | `MegaKernelWeightLayout` 是 pack 阶段便利结构，编译器只读 `named_offsets` | `grep -rn "MegaKernelWeightLayout" src/compiler/codegen/` 无结果（仅 pack/observe 层使用） |
| REQ-UMK-009 | `CompiledLayerFn` 类型、`execute()` 方法、10-param ABI 相关代码物理删除 | `grep -rn "CompiledLayerFn\|COMPILED_LAYER_PARAMS\|default_abi" src/compiler/` 无结果；参见 §6 废弃清单状态列 |
| REQ-UMK-010 | `compile_mega_kernel_vm` Phase 3 按融合组顺序 emit，不硬编码执行阶段 | `grep -rn "Phase 4\|Phase 5\|Phase 6\|Phase 7" src/compiler/codegen/vm/` 无 JIT 发射阶段编号；仅有 prologue + `for group in groups { emit }` + epilogue |
| REQ-UMK-011 | `CompileConfig` 不携带图结构假设字段（`is_encoder`/`inference_mode`/`vocab_size`/`num_layers`/`hidden`） | `CompileConfig` 结构体定义中无上述字段；`vocab_size` 从 `OpKind::Gemm { n }` 推导；`num_layers` 从图拓扑推导 |
| REQ-UMK-012 | `GroupMarker` 机制替代硬编码层循环 | `grep -rn "LayerLoopBegin\|LayerLoopEnd\|GroupMarker" src/compiler/codegen/vm/` 有结果；Phase 2 融合引擎根据图同构子结构插入标记 |
| REQ-UMK-013 | `PhaseDispatch` 仅当图含 Argmax 算子时存在 | `if graph.ops.iter().any(\|op\| matches!(&op.kind, OpKind::Argmax { .. }))` → 插入 PhaseDispatch；否则无分支 |
| REQ-UMK-014 | `MegaKernelConfig`（旧）重命名为 `CompileConfig`，字段重新分类 | `grep -rn "MegaKernelConfig" src/compiler/mega_kernel_abi.rs` 仅保留兼容别名；新 `CompileConfig` 不含 `vocab_size` 等图推导字段 |

## §8 受影响的跨仓库 SPEC

以下 SPEC 文档已同步更新以消除对废弃 API 的引用：

| SPEC 文档 | 更新内容 | 状态 |
|-----------|---------|------|
| `01-JIT-PIPELINE.md` | §977 将 `compile_graph()` 替换为 `compile()` | ✅ 已完成 |
| `01-REQUIREMENTS.md` | REQ-JIT-CACHE-004 将 `compile_graph` 替换为 `compile` | ✅ 已完成 |
| `08-EXECUTOR.md` | §4.7 强化"唯一 ABI"声明，`CompiledLayerFn` 标注"已物理删除 (SPEC/39)" | ✅ 已完成 |
| `08-EXECUTOR.md` | §4.1.4 添加 SPEC/39 注释：模型类型差异由图拓扑自然决定 | ✅ 已完成 |
| `15-GPU-HOST-GLUE.md` | 删除 REQ-GPU-011（Forward-only GPU 编译），encoder GPU 走同一 `compile()` | ✅ 已完成 |
| `15-GPU-HOST-GLUE.md` | ARCH-CPU-GPU-UNIFIED 原则更新为 `compile()` (SPEC/39) | ✅ 已完成 |
| `15-GPU-HOST-GLUE.md` | ABI 参数类型统一为与 SPEC/39 §2.1 一致（usize 替代 u32） | ✅ 已完成 |
| `18-SYMDIM-PAGED-KV.md` | §105 将 `compile_forward_from_graph()` 替换为 `compile()` | ✅ 已完成 |
| `ARCH-DATA-FLOW-CONTRACT.md` | `CompiledLayerFn` 标注"已物理删除 (SPEC/39)" | ✅ 已完成 |
| `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md` | `PainPointAnalyzer` 移除 `InferenceMode`，`gemm_shapes(graph)` 替代 `gemm_shapes(mode)` | ✅ 已完成 |
| `GRAPH-SHAPE-DRIVEN-MEGA-KERNEL.md` | Phase 4-8 → 融合组视角，"embed 阶段" → "embed 融合组" | ✅ 已完成 |
| `12-STRATEGY-ARBITER.md` | `InferenceMode` → `SchedulingMode` (从 batch_size 推导) | ✅ 已完成 |
| `04-API-DESIGN.md` | 删除 `.inference_mode(InferenceMode::Latency)` builder 方法 | ✅ 已完成 |
| `20-BATCH-CONCURRENT-INFERENCE.md` | Phase 4-8 → 融合组视角，图假设术语 → 拓扑无关 | ✅ 已完成 |
| `32-MEGA-KERNEL-ENHANCEMENT.md` | Phase → 融合组视角，Phase 4.5 → PhaseDispatch | ✅ 已完成 |
| `33-MLA-MATRIX-ABSORPTION.md` | Phase 0.7/2'-4'/7'/8' → 融合组视角 | ✅ 已完成 |
| `34-MTP-MULTI-TOKEN-PREDICTION.md` | Phase 4.5'/5'/7' → MTP 融合组视角 | ✅ 已完成 |
| `26-VMINSTR-RATIONALIZATION.md` | Phase 4.5' → MTP 融合组 | ✅ 已完成 |
| `DOCS/scheduling/jit-cache-protocol.md` | `compile_graph()` → `compile()` | ✅ 已完成 |
| `DOCS/scheduling/symdim-threading-protocol.md` | `compile_graph()` → `compile()` | ✅ 已完成 |
