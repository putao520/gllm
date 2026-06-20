# 28 — 全图资源预规划 (Graph Resource Planner)

> **状态**: 实验性 — 代码已实现（`resource_planner.rs`），但未集成到编译管线。当前编译路径使用 reg_alloc + stack_frame 事后分配。集成路线图见下方 §Integration。

> **铁律 ARCH-GRAPH-RESOURCE-PLAN**: 单一 mega-kernel = 单一资源规划。发射任何 VmInstr 之前，必须完成整图资源预规划。禁止边发代码边算资源。

## §0 问题与设计哲学

### §0.1 约束：单一 mega-kernel

gllm 架构铁律：整个模型推理编译为**一个函数**（mega-kernel）。一次 CALL 完成所有层的计算。

这意味着：
- **一个栈帧** — 所有层的 spill、临时存储、ABI args 在同一个栈上
- **一个寄存器文件** — 所有层共享同一组物理寄存器，层间复用
- **一个 scratchpad** — 所有中间 buffer 在同一块内存中，按生命周期复用
- **无函数调用边界** — 层间没有 caller/callee save/restore，寄存器状态连续传递

在这种约束下，**不可能逐层独立规划资源**——每层独立分配会严重浪费（N 层 × 每层最大寄存器需求 vs 全局分配一次）。必须**整图一次性规划**。

### §0.2 现状分析

| 组件 | 位置 | 做什么 | 问题 |
|------|------|--------|------|
| `BufferAllocation` | `buffer_alloc.rs` | 张量生命周期分析 + 区间图着色 scratchpad 分配 | 仅管内存，不管寄存器 |
| `PressureModel` | `stack_frame.rs` | 缓存感知 GEMM 分块 (mc/nc/kc) | 仅管分块，不管全局压力 |
| `LifecycleTag` | `reg_alloc.rs` | VReg 生命周期标签 | 事后打标，不影响发射决策 |
| `ScopedSpillAllocator` | `reg_alloc.rs` | 作用域 spill slot 回收 | 事后回收，非预规划 |
| `StackFrame` | `stack_frame.rs` | 栈帧计算 | 依赖 reg_alloc 结果，非独立规划 |

**核心问题**：这些组件各自独立、部分事后执行。没有统一的预规划阶段在发射代码前告诉发射器"资源长什么样"。

### §0.3 设计目标

```
CompilerGraph + FusionPlan + DeviceProfile
    │
    ▼
┌──────────────────────────────────────────────┐
│          Graph Resource Planner              │
│  (整图资源预规划 — 发射任何代码之前)           │
│                                              │
│  1. 张量生命周期分析 → 哪些 buffer 可复用      │
│  2. 寄存器压力曲线 → 哪里高压需降展开度        │
│  3. 栈帧蓝图 → spill/temp/ABI 统一布局        │
│  4. 循环不变量检测 → 哪些值提升到层循环外       │
│  5. 并发资源隔离 → per-sequence vs shared      │
│                                              │
│  输出: GraphResourcePlan (纯数据)             │
└──────────────────────────────────────────────┘
    │
    ▼
SPEC 27 模板解释器 / auto_select (按 ResourcePlan 发射代码)
    │
    ▼
VmInstr → ISA Lowering → 机器码
```

**关键原则**：`GraphResourcePlan` 是**只读数据**，在代码发射阶段不修改。发射器查询它做决策，就像编译器后端查询寄存器分配结果一样。

## §1 GraphResourcePlan 数据模型

### §1.1 顶层结构

```rust
/// 全图资源预规划结果 — 在发射任何 VmInstr 之前一次性计算。
///
/// 包含整个 mega-kernel 的资源布局：
/// - 每个中间 buffer 的物理位置和生命周期
/// - 每个图节点的寄存器压力估计
/// - 栈帧的完整蓝图
/// - 循环不变量集合
/// - 并发资源分区
///
/// 发射阶段只读查询此结构，不修改。
pub struct GraphResourcePlan {
    /// 张量 → scratchpad 偏移映射 (含生命周期复用)
    pub buffers: BufferLayout,
    /// 每个融合组的寄存器压力估计
    pub pressure: Vec<GroupPressure>,
    /// 栈帧蓝图
    pub stack: StackBlueprint,
    /// 循环不变量集合
    pub loop_invariants: Vec<LoopInvariant>,
    /// 并发资源分区
    pub concurrency: ConcurrencyPartition,
    /// 全局常量 (total_scratchpad_bytes, total_stack_bytes, etc.)
    pub summary: ResourceSummary,
}
```

### §1.2 BufferLayout — 张量内存布局

```rust
/// 张量内存布局 — 扩展现有 BufferAllocation。
///
/// 在现有 lifetime interval graph coloring 基础上增加:
/// - 层循环维度的生命周期扩展 (N 层 × 相同生命周期)
/// - Ping-pong 双 buffer 规划
/// - 跨层 buffer 复用 (层 N 的 output buffer 可复用层 M 的 expired buffer)
pub struct BufferLayout {
    /// 现有 BufferAllocation (保持兼容)
    pub base: BufferAllocation,

    /// 张量生命周期 — 每个张量的 [first_use_step, last_use_step]
    pub lifetimes: HashMap<TensorId, TensorLifetime>,

    /// Scratchpad 内存映射 — 区间图着色后的结果
    /// 相同 offset 的不同张量 = 生命周期不重叠的复用
    pub memory_map: Vec<MemoryRegion>,

    /// Ping-pong buffer 对 — 层循环中的双 buffer
    pub ping_pong: Option<PingPongLayout>,

    /// 总 scratchpad 字节数
    pub total_bytes: usize,
}

/// 张量生命周期 — 扩展到层循环维度
pub struct TensorLifetime {
    pub tensor_id: TensorId,
    /// 单层内的生命周期 [first_step, last_step]
    pub intra_layer: (usize, usize),
    /// 跨层生命周期
    pub cross_layer: CrossLayerLifetime,
    /// buffer 字节数
    pub size_bytes: usize,
}

pub enum CrossLayerLifetime {
    /// 仅当层使用 (如 FFN 中间结果) — 每层独立分配
    PerLayer,
    /// 跨层连续存活 (如 activation, KV cache) — 层循环外分配
    SpanningAllLayers,
    /// N 层内有效 (如 sliding window KV)
    SpanningNLayers(usize),
}

/// Scratchpad 内存区域
pub struct MemoryRegion {
    pub offset: usize,
    pub size_bytes: usize,
    /// 占用此区域的张量列表 (生命周期不重叠)
    pub tenants: Vec<TensorId>,
}

/// Ping-pong 双 buffer 布局
pub struct PingPongLayout {
    /// 输入 buffer offset (层 N 读)
    pub ping_offset: usize,
    /// 输出 buffer offset (层 N 写)
    pub pong_offset: usize,
    /// 每个 buffer 的大小
    pub buffer_bytes: usize,
}
```

### §1.3 GroupPressure — 融合组寄存器压力

```rust
/// 融合组的寄存器压力估计 — 指导代码发射策略。
///
/// 在发射每个融合组的代码之前查询此结构，决定:
/// - GEMM 展开度 (高压力 → 降低 mr/nr)
/// - 是否需要额外的 spill/reload
/// - 微核选择 (高压力用更小微核)
pub struct GroupPressure {
    /// 融合组 ID
    pub group_id: usize,
    /// 该组峰值向量寄存器需求
    pub peak_vec_regs: usize,
    /// 该组峰值 GPR 需求
    pub peak_gpr_regs: usize,
    /// 该组可用的向量寄存器 (从总可用减去跨组活跃的)
    pub available_vec_regs: usize,
    /// 该组可用的 GPR (从总可用减去跨组活跃的)
    pub available_gpr_regs: usize,
    /// 建议的 GEMM 展开策略
    pub suggested_blocking: Option<GemmBlocking>,
    /// 是否需要该组前后插入 spill/reload
    pub needs_spill_fence: bool,
}
```

### §1.4 StackBlueprint — 栈帧蓝图

```rust
/// 栈帧蓝图 — 在代码发射前预计算完整栈布局。
///
/// 替代现有事后计算的 StackLayout，提供发射阶段可查询的栈规划。
pub struct StackBlueprint {
    /// 总栈帧大小 (sub rsp, N 中的 N)
    pub total_frame_bytes: usize,

    /// ABI 参数 slots: [rbp + offset]
    pub abi_arg_slots: [Option<i32>; 6],

    /// Callee-save 寄存器 slots
    pub callee_save_slots: Vec<(AsmReg64, i32)>,

    /// Spill 区域起始偏移
    pub spill_base_rbp_off: i32,

    /// Spill slot 分配 — 每个 slot 的 (rbp_offset, size_bytes, lifecycle)
    pub spill_slots: Vec<SpillSlot>,

    /// MXCSR 保存位置
    pub mxcsr_rsp_off: i32,

    /// Debug 探针 buffer (可选)
    pub debug_probe_region: Option<DebugProbeRegion>,
}

pub struct SpillSlot {
    pub rbp_offset: i32,
    pub size_bytes: usize,
    /// 哪些 VReg 在什么区间使用此 slot
    pub lifecycle: LifecycleTag,
}

pub struct DebugProbeRegion {
    pub rbp_offset: i32,
    pub size_bytes: usize,
}
```

### §1.5 LoopInvariant — 循环不变量

```rust
/// 循环不变量 — 在层循环外计算一次，循环内只读。
///
/// 由 Graph Resource Planner 从 CompilerGraph 推导:
/// - 哪些值在所有层都相同 (如 cos/sin 表)
/// - 哪些权重偏移是常量 (如 PackMap 索引)
/// - 哪些标量参数每层相同 (如 eps, hidden_dim)
pub struct LoopInvariant {
    /// 不变量描述
    pub kind: InvariantKind,
    /// 存放位置 (寄存器 or 栈)
    pub location: InvariantLocation,
    /// 计算方式 (在层循环外用什么指令算)
    pub computation: InvariantComputation,
}

pub enum InvariantKind {
    /// RoPE cos/sin 表指针 (所有层共享)
    RopeTablePtr,
    /// RMSNorm scale/gamma 指针 (按层偏移)
    NormGammaPtr { layer_stride: usize },
    /// 模型配置常量 (hidden_dim, num_heads, etc.)
    ModelConfig { name: String, value: usize },
    /// PackMap 索引基址
    PackMapBase,
}

pub enum InvariantLocation {
    /// 固定 GPR (整个层循环生命周期)
    Gpr(usize),
    /// 栈 slot (rbp 偏移)
    Stack(i32),
}

pub enum InvariantComputation {
    /// 从 ABI arg 加载
    LoadAbiArg(u8),
    /// 立即数加载
    LoadImm(u64),
    /// 指针运算: base + layer_stride * counter
    PtrArithmetic { base: u8, stride: usize },
}
```

### §1.6 ConcurrencyPartition — 并发资源隔离

```rust
/// 并发资源分区 — 哪些资源是 per-sequence 的，哪些是全局共享的。
///
/// 在 batch_size > 1 时，mega-kernel 并发处理多条序列。
/// 资源必须明确分区，否则会冲突。
pub struct ConcurrencyPartition {
    /// Per-sequence 资源 (每条序列独占)
    pub per_sequence: PerSequenceResources,
    /// 共享资源 (所有序列共享)
    pub shared: SharedResources,
}

pub struct PerSequenceResources {
    /// 每 sequence 的 scratchpad 区域大小
    pub scratchpad_bytes_per_seq: usize,
    /// 每 sequence 的 spill slot 数量
    pub spill_slots_per_seq: usize,
    /// KV cache 页表区域
    pub kv_cache_region: MemoryRegion,
    /// BatchSeqId → per-sequence data 的偏移映射
    pub seq_offset_map: SeqOffsetMap,
}

pub struct SharedResources {
    /// 权重指针 (所有序列读同一份权重)
    pub weight_ptr: InvariantLocation,
    /// 共享常量 (cos/sin 表, config 等)
    pub constants: Vec<InvariantLocation>,
    /// 全局 barrier/counter 区域 (如 MoE expert 命中计数)
    pub global_counters: Vec<MemoryRegion>,
}

pub struct SeqOffsetMap {
    /// 从 per-sequence scratchpad 基址到各 buffer 的偏移
    pub activation_offset: usize,
    pub kv_cache_offset: usize,
    pub temp_buffer_offset: usize,
}
```

### §1.7 ResourceSummary

```rust
pub struct ResourceSummary {
    pub total_scratchpad_bytes: usize,
    pub total_stack_bytes: usize,
    pub peak_vec_regs: usize,
    pub peak_gpr_regs: usize,
    pub num_layers: usize,
    pub num_buffer_reuse_slots: usize,
    pub bytes_saved_by_reuse: usize,
    pub num_loop_invariants: usize,
    pub batch_size: usize,
}
```

## §2 预规划管线

### §2.1 管线阶段

```
CompilerGraph + FusionPlan + DeviceProfile + BufferAllocation
    │
    ▼ Phase 1: 张量生命周期分析 (扩展现有 analyze_lifetimes)
    │
    ▼ Phase 2: 内存区间图着色 (扩展现有 interval_coloring)
    │
    ▼ Phase 3: 寄存器压力曲线计算 (新增)
    │
    ▼ Phase 4: 栈帧蓝图计算 (新增，替代事后 StackFrame::compute)
    │
    ▼ Phase 5: 循环不变量推导 (新增)
    │
    ▼ Phase 6: 并发资源分区 (新增)
    │
    ▼ Phase 7: 汇总 + 一致性验证
    │
    GraphResourcePlan (不可变数据)
```

### §2.2 Phase 1: 张量生命周期分析

扩展现有 `analyze_lifetimes()`：

```rust
/// Phase 1: 张量生命周期分析
///
/// 在现有 analyze_lifetimes 基础上增加:
/// 1. 层循环维度扩展 (每层重复相同生命周期模式)
/// 2. 跨层存活检测 (activation, KV cache)
/// 3. Output-alias 解析 (output 复用 input slot)
fn analyze_tensor_lifetimes_extended(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    vtm: Option<&VirtualTensorMap>,
    vam: Option<&VirtualActivationMap>,
) -> Vec<TensorLifetime> {
    let base = analyze_lifetimes(graph, plan, vtm, vam);

    base.into_iter().map(|lt| {
        let cross_layer = determine_cross_layer_lifetime(lt.tensor_id, graph);
        TensorLifetime {
            tensor_id: lt.tensor_id,
            intra_layer: (lt.first_use, lt.last_use),
            cross_layer,
            size_bytes: lt.size_bytes,
        }
    }).collect()
}
```

### §2.3 Phase 2: 内存区间图着色

扩展现有 interval coloring 增加 ping-pong 双 buffer：

```rust
/// Phase 2: 内存区间图着色
///
/// 算法: 按 last_use 排序 → 贪心分配到最左可用 slot
/// 新增: ping-pong buffer 对单独规划
fn plan_memory_layout(
    lifetimes: &[TensorLifetime],
    graph: &CompilerGraph,
) -> BufferLayout {
    let base_alloc = interval_coloring(lifetimes);
    let ping_pong = plan_ping_pong(graph, &base_alloc);
    let memory_map = build_memory_map(&base_alloc, lifetimes);
    let total_bytes = base_alloc.total_bytes;

    BufferLayout {
        base: base_alloc,
        lifetimes: lifetimes.iter().map(|lt| (lt.tensor_id, lt.clone())).collect(),
        memory_map,
        ping_pong,
        total_bytes,
    }
}
```

### §2.4 Phase 3: 寄存器压力曲线

**新增 Pass** — 估计每个融合组的峰值寄存器需求：

```rust
/// Phase 3: 寄存器压力曲线
///
/// 对每个融合组:
/// 1. 分析组内所有 Op 的输入/输出数量
/// 2. 估计 GEMM 累加器需求 (mr × nr)
/// 3. 估计临时向量寄存器 (中间值)
/// 4. 估计 GPR 需求 (指针、循环计数器)
/// 5. 减去跨组活跃的寄存器 (前组输出仍活)
fn estimate_register_pressure(
    graph: &CompilerGraph,
    plan: &FusionPlan,
    lifetimes: &[TensorLifetime],
    profile: &IsaProfile,
) -> Vec<GroupPressure> {
    let total_vec = profile.vec_regs.len();
    let total_gpr = profile.gpr_regs.len();

    plan.groups.iter().enumerate().map(|(gid, group)| {
        let live_across = count_live_across_groups(gid, plan, lifetimes);
        let inner_demand = estimate_inner_demand(group, graph, profile);

        let available_vec = total_vec.saturating_sub(live_across.vec_regs);
        let available_gpr = total_gpr.saturating_sub(live_across.gpr_regs);

        let suggested_blocking = if inner_demand.has_gemm {
            let blocking = derive_blocking_from_budget(available_vec, profile);
            Some(blocking)
        } else {
            None
        };

        GroupPressure {
            group_id: gid,
            peak_vec_regs: inner_demand.vec_regs,
            peak_gpr_regs: inner_demand.gpr_regs,
            available_vec_regs: available_vec,
            available_gpr_regs: available_gpr,
            suggested_blocking,
            needs_spill_fence: inner_demand.vec_regs > available_vec,
        }
    }).collect()
}
```

### §2.5 Phase 4: 栈帧蓝图

```rust
/// Phase 4: 栈帧蓝图
///
/// 从寄存器压力曲线推导 spill slot 需求，统一规划栈帧。
fn plan_stack_blueprint(
    pressure: &[GroupPressure],
    num_abi_args: usize,
    callee_saves: &[AsmReg64],
    profile: &IsaProfile,
) -> StackBlueprint {
    // 1. ABI arg slots: 固定 rbp 偏移
    // 2. Callee-save slots: 紧接 ABI args
    // 3. Spill slots: 峰值压力 > 可用寄存器 的组需要 spill
    // 4. Debug probe (可选)

    let max_spill_vec = pressure.iter()
        .map(|g| g.peak_vec_regs.saturating_sub(g.available_vec_regs))
        .max()
        .unwrap_or(0);

    let max_spill_gpr = pressure.iter()
        .map(|g| g.peak_gpr_regs.saturating_sub(g.available_gpr_regs))
        .max()
        .unwrap_or(0);

    // ... 计算 rbp 偏移 ...

    StackBlueprint { /* ... */ }
}
```

### §2.6 Phase 5: 循环不变量推导

```rust
/// Phase 5: 循环不变量推导
///
/// 分析 CompilerGraph，找出在层循环中不变的值:
/// 1. RoPE cos/sin 表 (所有层共享同一份)
/// 2. 模型配置常量 (hidden_dim, num_heads)
/// 3. PackMap 索引基址
/// 4. Scratchpad 基址
fn derive_loop_invariants(
    graph: &CompilerGraph,
    buffer_layout: &BufferLayout,
) -> Vec<LoopInvariant> {
    let mut invariants = Vec::new();

    // RoPE: 所有 RoPE 层使用相同 theta/partial → 共享 cos/sin 表
    if graph.has_rope() {
        invariants.push(LoopInvariant {
            kind: InvariantKind::RopeTablePtr,
            location: InvariantLocation::Gpr(/* 固定 GPR */),
            computation: InvariantComputation::LoadAbiArg(/* rope_table arg */),
        });
    }

    // 模型配置常量
    invariants.push(LoopInvariant {
        kind: InvariantKind::ModelConfig { name: "hidden_dim".into(), value: graph.hidden_dim() },
        location: InvariantLocation::Stack(/* rbp offset */),
        computation: InvariantComputation::LoadImm(graph.hidden_dim() as u64),
    });

    // PackMap 基址
    if graph.has_packed_weights() {
        invariants.push(LoopInvariant {
            kind: InvariantKind::PackMapBase,
            location: InvariantLocation::Gpr(/* 固定 GPR */),
            computation: InvariantComputation::LoadAbiArg(/* pack_map arg */),
        });
    }

    invariants
}
```

### §2.7 Phase 6: 并发资源分区

```rust
/// Phase 6: 并发资源分区
///
/// 根据 batch_size 决定资源分区策略:
/// - batch=1: 所有资源全局唯一
/// - batch>1: 区分 per-sequence (activation, KV) vs shared (weights, config)
fn partition_concurrency(
    graph: &CompilerGraph,
    buffer_layout: &BufferLayout,
    stack_blueprint: &StackBlueprint,
    batch_size: usize,
) -> ConcurrencyPartition {
    if batch_size <= 1 {
        // 无并发: 全局唯一
        return ConcurrencyPartition::single_sequence(buffer_layout, stack_blueprint);
    }

    // Per-sequence: activation buffer, KV cache, 临时 buffer
    let per_seq = PerSequenceResources {
        scratchpad_bytes_per_seq: buffer_layout.ping_pong.as_ref()
            .map(|pp| pp.buffer_bytes * 2)
            .unwrap_or(buffer_layout.total_bytes),
        spill_slots_per_seq: stack_blueprint.spill_slots.len(),
        kv_cache_region: buffer_layout.kv_cache_region(),
        seq_offset_map: SeqOffsetMap {
            activation_offset: 0,
            kv_cache_offset: buffer_layout.activation_bytes(),
            temp_buffer_offset: buffer_layout.activation_bytes() + buffer_layout.kv_bytes(),
        },
    };

    // Shared: weights, constants, global counters
    let shared = SharedResources {
        weight_ptr: InvariantLocation::Gpr(/* ABI arg weight_ptr */),
        constants: derive_loop_invariants(graph, buffer_layout)
            .into_iter()
            .map(|li| li.location)
            .collect(),
        global_counters: vec![],
    };

    ConcurrencyPartition { per_sequence: per_seq, shared }
}
```

### §2.8 Phase 7: 汇总 + 一致性验证

```rust
/// Phase 7: 汇总 + 一致性验证
///
/// 验证:
/// 1. 所有 buffer lifetime 无重叠冲突 (相同 offset 的张量不并发存活)
/// 2. 寄存器压力 ≤ 物理寄存器数 + spill slots
/// 3. 栈帧总大小 ≤ 合理上界 (如 64KB)
/// 4. 循环不变量未引用 per-sequence 资源
/// 5. 并发分区无资源冲突
fn verify_resource_plan(plan: &GraphResourcePlan, profile: &IsaProfile) -> Result<(), CompilerError> {
    // Buffer overlap check
    for region in &plan.buffers.memory_map {
        verify_no_temporal_overlap(&region.tenants, &plan.buffers.lifetimes)?;
    }

    // Register pressure check
    for gp in &plan.pressure {
        if gp.peak_vec_regs > profile.vec_regs.len() + plan.stack.spill_slots.len() * profile.optimal_simd_width().f32_lanes() {
            return Err(CompilerError::ResourceExhaustion(
                format!("Group {}: peak vec regs {} exceeds capacity", gp.group_id, gp.peak_vec_regs)
            ));
        }
    }

    // Stack size check
    const MAX_STACK_BYTES: usize = 65536;
    if plan.stack.total_frame_bytes > MAX_STACK_BYTES {
        return Err(CompilerError::ResourceExhaustion(
            format!("Stack frame {} exceeds {} limit", plan.stack.total_frame_bytes, MAX_STACK_BYTES)
        ));
    }

    // Loop invariant check
    for inv in &plan.loop_invariants {
        verify_invariant_not_per_sequence(inv, &plan.concurrency)?;
    }

    Ok(())
}
```

## §3 与 SPEC 27 模板的集成

### §3.1 模板解释器查询 ResourcePlan

模板解释器在实例化时查询 `GraphResourcePlan`，做出资源感知的决策：

```rust
/// 资源感知的模板实例化
pub fn instantiate_template_with_plan(
    template: &AlgoTemplate,
    ctx: &AlgoContext,
    plan: &GraphResourcePlan,  // ← 新增: 全图资源规划
) -> Result<Vec<TraceOp>, CompilerError> {
    let resolved = resolve_params(template.params, ctx)?;

    // 查询当前融合组的寄存器压力
    let group_pressure = &plan.pressure[ctx.current_group_id];
    let blocking = group_pressure.suggested_blocking.unwrap_or_else(|| {
        plan.pressure[ctx.current_group_id].derive_default_blocking(ctx.device)
    });

    // 查询 buffer 布局
    let buffer_for = |tensor_id: TensorId| -> usize {
        plan.buffers.base.offset_of(tensor_id).unwrap_or(0)
    };

    // 查询循环不变量
    let invariants = &plan.loop_invariants;

    // 按资源约束实例化模板
    let mut builder = TraceBuilder::new()
        .with_blocking(blocking)
        .with_buffer_map(buffer_for)
        .with_invariants(invariants);

    for step in template.steps {
        emit_step_with_resources(step, &resolved, ctx, &mut builder, template.micro_kernel, plan)?;
    }

    Ok(builder.finish())
}
```

### §3.2 寄存器压力影响 GEMM 策略

```
GroupPressure.suggested_blocking 影响:

高压力 (available_vec_regs < 16):
  → 降低 mr/nr (如 3×4 替代 6×4)
  → 减少 FMA 展开
  → 插入 spill fence 前后保存/恢复

中等压力 (16..24):
  → 标准展开 (6×4 BLIS)

低压力 (>24):
  → 激进展开 (6×8 或更大)
  → 更多 epilogue 融合
```

### §3.3 Buffer 布局影响面板加载

```
BufferLayout 影响:

LoadPanel 步骤:
  → A 面板地址 = weight_ptr + PackMap[layer][tile]  (从 loop_invariants 查)
  → B 面板地址 = pack_b_scratch + offset             (从 buffers 查)
  → C 输出地址 = activation_pong + tile_offset       (从 ping_pong 查)

PackBuffer 步骤:
  → 源 = weight_ptr + B 偏移
  → 目标 = scratch[pack_region]                       (从 buffers 查)
```

## §4 与现有组件的关系

### §4.1 组件演进

| 现有组件 | SPEC 28 中的角色 | 改动 |
|---------|----------------|------|
| `BufferAllocation` | Phase 1-2 的基础 | 扩展 `TensorLifetime` 增加跨层维度 |
| `PressureModel` | Phase 3 的输入 | 不变，作为 per-group blocking 的默认 |
| `LifecycleTag` | Phase 3 寄存器压力估算的输入 | 不变 |
| `ScopedSpillAllocator` | Phase 4 的参考 | 替换为 `StackBlueprint` 的预计算 spill |
| `StackFrame` | Phase 4 的产出 | 从 `compute(post_hoc)` 改为 `build(pre_plan)` |
| `reg_alloc.rs` | 发射后的验证 | 保留 post-hoc reg_alloc，但用 pre-plan 指导 |

### §4.2 两阶段分配模型

```
Pre-plan (SPEC 28):           Post-verify (SPEC 25):
估算压力 → 指导发射策略        发射后 → 精确 reg_alloc → 验证与 pre-plan 一致

Phase 3: "这个组大约需要      reg_alloc: "实际用了 22 个 vec reg，
18 个 vec reg, 用 3×4 微核"   预估 18, 误差 +4, 但可用寄存器有 28, OK"

如果 post-verify 发现冲突 → 报错, 回退到更保守策略重新编译
```

**Pre-plan 不替代 post-hoc reg_alloc**，而是：
1. 指导发射策略（在发射前就知道大概的压力）
2. 减少 post-hoc 分配的 spill（因为发射时已经考虑了压力）
3. 提供验证基准（post-hoc 结果应该 ≤ pre-plan 估计）

## §5 REQ 列表

### REQ-GRP-001: TensorLifetime 跨层扩展

扩展 `analyze_lifetimes()` 增加 `CrossLayerLifetime` 维度。检测 per-layer vs spanning-all-layers 张量。扩展 `BufferAllocation` 增加 `lifetimes` 字段。

**文件**: `buffer_alloc.rs`

### REQ-GRP-002: MemoryLayout 内存区间图着色

将 interval coloring 结果组织为 `MemoryRegion` 列表，包含复用租户信息。规划 ping-pong 双 buffer 布局。

**文件**: `buffer_alloc.rs`

### REQ-GRP-003: GroupPressure 寄存器压力曲线

对每个融合组估计峰值寄存器需求。根据压力推导 `suggested_blocking`。标记需要 spill fence 的组。

**文件**: `resource_planner.rs`（新）

### REQ-GRP-004: StackBlueprint 栈帧蓝图

预计算完整栈布局: ABI args → callee saves → spill slots → debug region。从 GroupPressure 推导 spill slot 需求。

**文件**: `resource_planner.rs`（新）

### REQ-GRP-005: LoopInvariant 循环不变量推导

从 CompilerGraph 推导层循环不变量 (RoPE 表、Norm gamma、配置常量、PackMap 基址)。分配固定 GPR 或栈 slot。

**文件**: `resource_planner.rs`（新）

### REQ-GRP-006: ConcurrencyPartition 并发资源分区

batch > 1 时区分 per-sequence (activation, KV, temp) vs shared (weights, config) 资源。计算 per-sequence scratchpad 布局。

**文件**: `resource_planner.rs`（新）

### REQ-GRP-007: GraphResourcePlan 汇总 + 验证

整合 Phase 1-6 产出为 `GraphResourcePlan`。执行一致性验证: buffer 无时序重叠、寄存器压力不超限、栈帧大小合理、不变量不引用 per-seq 资源。

**文件**: `resource_planner.rs`（新）

### REQ-GRP-008: plan_mega_kernel_resources() 编排入口

实现 `plan_mega_kernel_resources(graph, plan, profile) -> GraphResourcePlan` 编排函数。接入 `compile()` 编译入口，在 VmInstr 发射之前调用。

**文件**: `mod.rs`（修改编译入口）

### REQ-GRP-009: 模板解释器资源感知集成

`instantiate_template_with_plan()` 查询 `GraphResourcePlan`。GEMM 模板根据 `suggested_blocking` 调整展开度。面板加载根据 `BufferLayout` 计算地址。

**文件**: `algo_interpreter.rs`（SPEC 27）

### REQ-GRP-010: E2E 验证 — 资源规划正确性

验证:
1. 单层模型: buffer 布局正确，无重叠
2. 多层模型: ping-pong 正确，层间数据传递正确
3. batch > 1: per-sequence 资源隔离正确
4. 数值对齐: 资源规划不改变计算结果

**文件**: 测试

## §6 实施顺序

```
REQ-GRP-001 (TensorLifetime 扩展)
    ↓
REQ-GRP-002 (MemoryLayout) + REQ-GRP-005 (LoopInvariant — 可并行)
    ↓
REQ-GRP-003 (GroupPressure) + REQ-GRP-006 (Concurrency — 可并行)
    ↓
REQ-GRP-004 (StackBlueprint)
    ↓
REQ-GRP-007 (汇总 + 验证) + REQ-GRP-008 (编排入口 — 可并行)
    ↓
REQ-GRP-009 (模板集成)
    ↓
REQ-GRP-010 (E2E 验证)
```

## §7 文件结构

```
gllm-kernels/src/compiler/
├── buffer_alloc.rs          — [REQ-GRP-001, 002] 扩展 TensorLifetime + MemoryLayout
├── resource_planner.rs      — [REQ-GRP-003~008] 全图资源预规划 (新)
├── mod.rs                   — [REQ-GRP-008] compile() 接入
├── codegen/vm/
│   ├── algo_interpreter.rs  — [REQ-GRP-009] 资源感知模板实例化 (SPEC 27)
│   ├── stack_frame.rs       — 从 post-hoc 改为查询 StackBlueprint
│   └── ...
```

## §8 与其他 SPEC 的关系

```
SPEC 25 (JIT Lifecycle) — 提供生命周期标签体系，SPEC 28 在其基础上做预规划
SPEC 26 (VmInstr 去类型化) — 无关，纯指令层面
SPEC 27 (算法模板) — 模板解释器消费 GraphResourcePlan 做资源感知发射
SPEC 28 (资源预规划) — 在模板实例化之前提供全局资源约束
```

```
调用顺序:
compile(graph, config)
  → SPEC 28: plan_mega_kernel_resources(graph) → GraphResourcePlan
  → SPEC 27: emit_algo_from_template(template, plan) → TraceOp
  → auto_select: auto_lower_trace(trace_ops) → VmInstr
  → SPEC 25: reg_alloc(vm_program) → 精确分配
  → 验证: reg_alloc 结果 ≤ pre-plan 估计
  → ISA Lowering → 机器码
```
