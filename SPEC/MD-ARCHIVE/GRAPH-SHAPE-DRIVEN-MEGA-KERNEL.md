# 图形状驱动的 Mega-Kernel 编译算法 (SSOT)

> **元抽象**: 整个编译管线围绕一个统一原则——**编译时映射函数替代运行时物理操作**。
> 半 VM (VmInstr) 和 VTC (VirtualTensor) 是这个元抽象在不同维度的实例化。
> Phase 3 ISA Lowering 是唯一物化点，之前一切皆为虚拟资源。

> **铁律：mega-kernel 的结构（循环、算子排列、融合分组）完全由 CompilerGraph 拓扑
> 经融合管线自动推导。禁止在 `compile()` 中硬编码 VmInstr 序列。**

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
Mega-Kernel 编译管线与通信指令集成:
<a data-xref-id="REQ-DP-010" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-010">REQ-DP-010</a>
(VmInstr 扩展) 定义 RemotePageLookup/P2pPageFetch 等，通过本文件 Phase 3 lowering 集成 |
<a data-xref-id="REQ-DP-011" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-011">REQ-DP-011</a>
(CommInstr 扩展) 由 gllm-nccl 定义 CommInstr 序列 → 本文件 VmInstr 转换器消费 |
<a data-xref-id="REQ-SMPART-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-SMPART-002">REQ-SMPART-002</a>
(SM 分区配置) 与本文件 RegAllocator + ResourcePlanner 预算规划协同 |
<a data-xref-id="REQ-VENDOR-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-001">REQ-VENDOR-001</a>~<a data-xref-id="REQ-VENDOR-003" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-VENDOR-003">REQ-VENDOR-003</a>
(三厂商后端) 消费本文件 GpuBackendDialect lowering
</div>

## 参考文献

| ID | 论文 | 场景 | gllm 借鉴点 |
|----|------|------|-------------|
| **VTC** | Muyan Hu et al., "VTC: DNN Compilation with Virtual Tensors for Data Movement Elimination", OSDI 2026, arXiv:2604.09558 | 虚拟 tensor 消除数据搬移 | §0.3 统一 tensor 抽象 + §3.3 Round 2 数据流消除 |
| **TVM** | T. Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018 | OpPatternKind 分类 + Post-Dominator Tree 融合 | §3.2 Round 1 PDT 拓扑融合 + OpClass 分类 |
| **Inductor** | A. Paszke et al., "PyTorch Inductor: A PyTorch Native Compiler", 2023 | score_fusion 融合收益评分 | §4 融合收益评分 |
| **BLIS** | F. Van Zee et al., "The BLIS Framework", TOMS 2016 | KC/MC/NC 三级分块 + pack_a/pack_b | GEMM 分块策略基础 |

## §0 核心哲学：全虚拟化编译管线 — 编译时映射替代运行时物理操作

### §0.1 元抽象：Virtual Resource = (标识, 映射函数) → 物理实体

整个编译管线围绕**一个**统一原则构建：

```
每个编译时实体都是一个虚拟资源 (Virtual Resource):
  VirtualResource = (Id, MapFn) → PhysicalEntity

其中:
  Id       = 唯一标识 (TensorId, VRegId, LoopScopeId, GroupId, ...)
  MapFn    = 编译时确定的映射函数 (索引映射 / 寄存器分配 / 偏移计算 / ...)
  物理实体 = Phase 3 ISA Lowering 的唯一输出 (机器码指令)

关键约束: 所有 MapFn 在编译时确定，运行时零开销。
物化 (Materialization) = MapFn 的求值，仅发生在 Phase 3。
```

**半 VM（VmProgram）和 VTC（VirtualTensor）是这个元抽象的两个实例化**：
- 半 VM = 虚拟资源在**计算/控制/寄存器/内存**维度的实例化
- VTC  = 虚拟资源在**数据流**维度的实例化
- 两者共享同一个元抽象：**编译时映射函数替代运行时物理操作**

### §0.2 全虚拟化图谱 — 十维 (含 GPU 特性)

**核心原则**：MapFn 参数化于 `DeviceProfile`。同一个虚拟资源在不同硬件上物化为完全不同的
物理实现。虚拟层编码**意图**（what），物化层选择**实现**（how）。

```
虚拟维度          编译时映射函数 (MapFn)               物理实体 (Phase 3 输出)
─────────────────────────────────────────────────────────────────────────────
§0.2.1 虚拟数据   TensorId → (source, IndexMap)       实际内存地址 + load/store 指令
  (VTC)           IndexMap::Identity/Offset/Permute    指令中的地址计算

§0.2.2 虚拟寄存器 VRegId → PhysReg                    物理寄存器号 (ymm0-rmm31)
  (RegAlloc)      无限命名空间 → 图着色分配             spill 时的栈偏移

§0.2.3 虚拟内存   TensorId → scratch_offset           [rbp - N] 栈帧偏移
  (BufferAlloc)   生命周期区间着色 + 对齐约束           实际 byte offset

§0.2.4 虚拟控制   LoopScopeId → JMP target address    实际分支指令 (jge .loop_end)
  (LoopBegin/End) 作用域嵌套 + BoundExpr              CMP + 条件跳转

§0.2.5 虚拟维度   SymDim::Symbolic("seq_len")         运行时值读取 (ABI 参数位置)
  (SymDim)        Symbolic → BoundExpr::Symbolic       实际立即数或内存读取

§0.2.6 虚拟计算   FusionGroupId → composed_op_chain   单段连续机器码 (无 CALL)
  (Fusion)        算子组合 → 中间值在寄存器中流转      累加器直接消费

§0.2.7 虚拟权重   WeightTensorId → (raw_source, PackMap)   带 stride 的 load 指令
  (PackLayout)    PackMap(MR, NR, KC) 索引映射        无需物理 pack 的访存模式
                  [Phase D 深化]
                  **与 §3.10 布局协商的关系**: §0.2.7 管权重**加载侧**
                  (raw weights → 按 PackMap 索引读取), §3.10 管权重**输出侧**
                  (GEMM 输出 → 按下游偏好布局写入 scratch)。同一个 pack 布局问题
                  的两个视角: 加载时用 PackMap 虚拟化避免物理 pack, 输出时用
                  布局协商在自然搬运点免费切换到下游最优布局。

§0.2.8 虚拟激活   LayerActivationId → (buffer_id, IterOffset)  同一 buffer 的不同指针
  (Activation)    跨迭代身份映射: output_N = input_{N+1}       原地加法 + ptr 交换
                  [Phase D 深化]

§0.2.9 虚拟执行   ExecutionPattern → DeviceProfile → PhysicalSchedule
  (ExecPattern)   硬件无关的执行意图 → 硬件特定的执行模式
                  编译时: "这个 GEMM 需要 tile 并行"     运行时: 选择最优 tile 策略
                  [Phase D 深化]

§0.2.10 虚拟并行  ParallelismDesc → DeviceProfile → ParallelSchedule
  (Parallelism)   逻辑并行度 → 物理 SIMD/warp/wavefront 映射
                  编译时: "这个循环向量化宽度 = W"      运行时: W = simd_width
                  编译时: "Multi-Wave 并行度 = N"       运行时: N = num_waves
                  [Phase D 深化]
```

**§0.2.10 虚拟并行 — Multi-Wave 内部调度机制**

虚拟并行编码并行度意图，由 DeviceProfile 决定物化方式。Multi-Wave 是 Mega-Kernel 内部执行路径——Rust 只做一次 CALL，Mega-Kernel 内部自行调度 wave partitioning。

```
意图 (Virtual)             CPU 物化                    GPU 物化
────────────────────────────────────────────────────────────────────────
SimdLoop {                  for lane in 0..simd_width   PTX: 单条 vector 指令
  width: Symbolic("W")      向量化循环 (ymm/zmm)        HIP: 单条 DS 指令
}                           NEON/SVE 向量寄存器         MSL: simd_vector

WaveParallel {              NUMA 绑定多线程              Grid launch 多 Thread Block
  num_waves: usize          每个线程处理一个 wave        每个 TB 处理一个 wave
  partition: PartitionDesc  跨 NUMA 节点独立执行         跨 SM 分区独立执行
}                           wave 间共享权重 (本地 L3)    wave 间共享权重 (global mem)

WarpCooperative {           N/A (单线程)                 Thread Block Cluster
  producer: WarpRole        —                           SM 0-1: producer (TMA)
  consumer: WarpRole        —                           SM 2-3: consumer (WGMMA)
}                           —                           warp barrier 同步
```
```

**§0.2.9 虚拟执行模式 — 硬件感知物化的关键**

虚拟执行模式编码**计算策略的意图**，由 DeviceProfile 决定具体物化方式：

```
意图 (Virtual)                CPU AVX2 物化              GPU SM90 物化
────────────────────────────────────────────────────────────────────────
TileGemm {                     BLIS KC/MC/NC 三级分块     TMA + wgmma 异步流水线
  tile_m, tile_n, tile_k       6×8 微核 + pack_a/b       16×16×8 tensor core tile
}                              FMA 向量乘累加              WMMA MMA 指令

SharedMemTile {                L1 cache 驻留 (64KB)       Shared memory 显式分块
  size_bytes,                  无需显式管理               -barrier + shared memory
  access_pattern               依赖 cache line 对齐       手动 prefetch + sync
}

AsyncPipeline {                软件预取 (PREFETCHT0)      TMA async memcpy
  producer, consumer           __builtin_prefetch          cp.async.bulk
}                              无等待 (顺序执行)           barrier.wait

WarpReduce {                   SIMD horizontal reduce     Warp shuffle reduce
  width                        hadd across lanes           __shfl_down_sync
}
```

**关键**: §0.2.9 使得融合决策和代码生成在虚拟层是硬件无关的。
`score_fusion()` 和 `FusionEngine` 产出的是虚拟执行模式，
Phase 3 根据 DeviceProfile 选择物化策略。这不是"为每种硬件写一套 codegen"，
而是"一套虚拟规划 + 多种物化策略"。

**与 R0 PainPointAnalyzer (§3.9) 的关系**: GemmBottleneck.exec_pattern 字段是
§0.2.9 虚拟执行模式的具体产出。R0 分析阶段根据 (ModelProfile × DeviceProfile)
为每个 GEMM 推导出最优 ExecutionPattern（如 TileGemm / SharedMemTile / AsyncPipeline），
Phase 3 codegen 消费此字段选择物化策略。虚拟执行模式是"意图"层 (§0.2.9)，
PainPointAnalyzer 是"推导"层 (§3.9)，两者共同构成感知→规划的桥梁。

**§0.2.10 虚拟并行策略**

```
意图 (Virtual)                CPU 物化                    GPU 物化
────────────────────────────────────────────────────────────────────────
SimdVectorize {                YMM (256-bit) / ZMM (512)  Warp (32 threads)
  element_width,               向量寄存器 lane 并行        SIMT 线程并行
  unroll_factor                循环展开 + 指令级并行       warp 级 ILP
}

ThreadParallel {               Rayon MC 循环并行           CUDA block 级并行
  parallel_dim,                多核线程池                   Grid/Block 调度
  granularity                  cache-line 粒度             warp 粒度
}

MemoryHierarchy {              L1/L2/L3 自动              Shared/Global/Register 显式
  level,                       预取 + cache 友好布局       手动 staging + bank 冲突避免
  strategy
}
```

§0.2.11 虚拟布局   TensorId → LayoutConstraint → PhysicalLayout
  (Layout)        加速指令布局约束 → 动态协商 → 最大公约布局
                  编译时: 收集所有加速指令的布局需求     运行时: 零变换消费
                  [核心新增, R1.5 布局协商]
```

**§0.2.11 虚拟布局 — 流水线时序级联协商**

> **关键洞察**: 不是找"一个全局最大公约布局"。每个加速指令有自己的布局偏好,
> 但流水线中相邻阶段之间**本来就要发生数据搬运**（store/load/TMA prefetch 等），
> 每个搬运点都是免费的布局变换窗口——直接按下一个阶段的需求布局搬过去。
> 协商器识别这些自然搬运点，在正确时刻零成本切换布局。

```
加速指令的布局需求是"声明"而非"规则":
├── AVX-512 GEMM:  输入 B 需要 PanelPacked(mr=14, nr=32, kc) 格式
├── AMX tile:      输入 A 需要 AmxTile(16×32 BF16), 输入 B 需要 K-pair interleaved
├── VNNI:          输入需要 u8×i8 32-bit lane packing
├── GPU WMMA:      输入需要 128-byte aligned shared memory tile
├── GPU wgmma:     输入需要 TMA 2D prefetch compatible layout
├── FlashAttention: Q/K/V 需要 [seq, heads, head_dim] head-split layout
├── RoPE:          需要 [seq, heads, head_dim] 以便 per-head rotation
└── SwiGLU:        可以消费 interleaved (gate[i], up[i]) pair layout

当前实现: 每个 codegen 路径自己硬编码 pack/transform → 重复 pack, 冗余变换
正确设计: R1.5 布局协商器动态求解 → 一次变换满足整条链 → 零额外变换
```

**当前实现状态**：
- §0.2.1-§0.2.6：已虚拟化（编译时映射，Phase 3 物化）
- §0.2.7 权重布局：当前为物理 pack，Phase D 深化为虚拟索引映射
- §0.2.8 激活流：当前每层独立分配，Phase D 深化为虚拟身份映射
- §0.2.9 执行模式：当前 CPU/GPU codegen 分离，Phase D 统一为虚拟执行意图
- §0.2.10 并行策略：当前硬编码在 codegen，Phase D 提升为虚拟并行描述符
- **§0.2.11 虚拟布局：当前无协商机制（每个 codegen 路径独立 pack），Phase C 引入动态布局协商**

### §0.3 零数据搬移目标 — 全虚拟化的推论

**终极目标：完全消除所有不必要的数据搬移。** 这是全虚拟化原则的直接推论——
如果每个资源都有编译时映射函数，那么运行时就不需要任何数据搬运来"准备"数据。

```
可消除的物理操作                虚拟化替代
──────────────────────────────────────────────────────
Tensor 物理复制 (memcpy)       → VirtualTensor IndexMap (§0.2.1)
权重物理重排 (pack_a/pack_b)   → VirtualPack PackMap (§0.2.7)
中间结果写回内存               → EpilogueInjection 寄存器直传 (§0.2.6)
层间 activation 拷贝           → VirtualActivation 身份映射 (§0.2.8)
维度信息运行时传递             → SymDim Symbolic 绑定 (§0.2.5)
寄存器 spill/reload            → 全局 RegAlloc 跨组复用 (§0.2.2)
循环计数器管理                 → LoopScope 自动递增 (§0.2.4)
GPU shared memory 显式拷贝     → VirtualExecPattern SharedMemTile (§0.2.9)
GPU global↔shared 同步等待     → VirtualExecPattern AsyncPipeline (§0.2.9)
硬件特定循环展开策略           → VirtualParallel SimdVectorize (§0.2.10)
加速指令间的数据布局变换       → VirtualLayout 动态协商最大公约布局 (§0.2.11)
重复 pack (多 GEMM 分别 pack)  → VirtualLayout 协商统一 pack 格式 (§0.2.11)
业务功能运行时分发 (if-else)   → 编译时条件图构建 + JMP table (§1.5)
Head Routing 重编译            → 一次编译包含所有 output_modes (§1.5.5)
Guardrail 轮询检查             → 共享内存 veto 标志 + 条件 JMP (§1.5.3)
SG 知识注入数据搬移            → 共享内存残差向量 + ADD 指令 (§1.5.3)
```

每个寄存器 load/store、内存复制都必须被证明**不可消除**才允许存在。
证明不了必要性 → 编译时映射函数替代之。

### §0.4 虚拟权重布局 — VTC 思想在权重维度的深化

> **核心洞察**：权重 pack 是 VTC 的特例。pack_a 将 A 矩阵从行-major 重排为
> MR-wide column panel，本质上是 `(i, j) → (i/MR * K + pc * MR + i%MR, j)` 的
> 索引映射。当前这个映射通过物理数据搬移实现，但完全可以表达为 VirtualTensor。

```
当前实现 (物理 pack):
  raw_weights[i][j] → pack_a → packed_weights[panel_idx][i_local][j]
  ├── 运行时: memcpy(MR × KC 面板) 到临时缓冲区
  ├── 额外内存: packed buffer 大小 = M × K (等于原始权重大小)
  └── 额外延迟: O(M × K) 数据搬移

深化方向 (虚拟 pack):
  VirtualWeight {
      source: raw_weights TensorId,
      index_map: PackMap { mr, nr, kc, stride_k },
  }
  ├── 编译时: PackMap 计算出每个 (i,j) 的物理偏移
  ├── ISA Lowering: 生成带 stride 的 SIMD load 指令
  │   vmovups ymm0, [weight_ptr + panel_offset + row_inner * sizeof(f32)]
  │   其中 panel_offset = (ic/mr) * kc * mr + pc * mr
  ├── 消除: pack buffer 分配 + pack 计算时间
  └── 代价: load 指令地址计算更复杂 (1-2 条额外 LEA/IMUL)

适用性分析:
├── M=1 (decode GEMV): pack 几乎无用 (只有 1 行) → 虚拟 pack 无损
├── M≥16 (prefill GEMM): pack 对 L2 友好 → 虚拟 pack 需要评估 cache 行为
└── 结论: decode 路径立即虚拟化, prefill 路径保留物理 pack 但决策是编译时的
```

**PackMap 数据结构**：

```rust
/// 权重 pack 的索引映射 (VTC 思想在权重维度的实例化)
pub struct PackMap {
    /// 微核行分块大小 (MR, 硬件相关: AVX2=6, AVX-512=16)
    pub mr: usize,
    /// 微核列分块大小 (NR, 硬件相关: AVX2=8, AVX-512=16)
    pub nr: usize,
    /// K 方向分块大小 (KC, 缓存约束: B strip fits in L1 × 0.50)
    pub kc: usize,
    /// 原始矩阵的 K 维度 stride (bytes)
    pub stride_k: usize,
}

impl PackMap {
    /// 将逻辑 (i, j) 映射到物理地址偏移
    /// pack_a 的索引公式: packed[pc*mr + i_local][j] where i_local = i % mr
    fn logical_to_physical(&self, i: usize, j: usize, pc: usize) -> usize {
        let panel = (i / self.mr) * self.kc * self.mr;
        let inner = pc * self.mr + (i % self.mr);
        (panel + inner) * self.stride_k + j
    }
}
```

**Phase 化**：
- Phase A-C: 维持物理 pack（已有实现，正确且高效）
- Phase D: 引入 `TensorPtrSource::VirtualPack { source, pack_map }`，decode 路径虚拟化

### §0.5 虚拟激活流 — VTC 思想在层间维度的深化

> **核心洞察**：层循环中 `activation_in == activation_out` 是一个恒等映射。
> 当前实现中每层独立分配 buffer，但激活流应该是 VirtualTensor 的身份映射。

```
当前实现 (每层独立 buffer):
  Layer 0: input_buf_A → [compute] → output_buf_B
  Layer 1: input_buf_B → [compute] → output_buf_C  (B 是新分配的)
  Layer 2: input_buf_C → [compute] → output_buf_D
  ├── N 层需要 ⌈N/2⌉ 个 activation buffer
  └── 残差连接: output = input + delta → 写入 output buffer

深化方向 (虚拟激活流):
  Layer 0: buf → [compute] → buf (原地, 虚拟 output = buf)
  Layer 1: buf → [compute] → buf (原地, 虚拟 output = buf)
  Layer 2: buf → [compute] → buf (原地, 虚拟 output = buf)
  ├── 只需 1 个 activation buffer
  ├── VirtualActivation: output_tensor.id → VirtualTensor { source: input_tensor.id, Identity, 0 }
  ├── 残差连接: output = input + delta → 直接 ADD [buf], delta → 写回 [buf]
  └── 消除: (⌈N/2⌉ - 1) × activation_bytes 的 buffer 分配

实现约束:
├── 要求: 融合组内的残差加法 (O_proj output + residual) 必须在 EpilogueInjection 中
│   → GEMM 累加器直接加 residual，结果写回 activation buffer
├── 要求: Buffer allocator 识别跨迭代生命周期 = 整个循环
│   → 分配固定 slot，不参与 interval coloring 复用
└── 要求: 循环末尾 ActivationSwap { primary, secondary } 交换 ptr (Phase D VM 指令)
```

### §0.6 多后端强一致性

- CPU x86 和 GPU CUDA/HIP/Metal 共享**完全一致**的 CompilerGraph IR + ComputePattern
- FusionRule 在所有后端上产生**相同**的融合决策
- 后端差异**仅在 Phase 3 ISA Lowering**（指令选择、SIMD 宽度、分块策略）
- **禁止** FusionRule 层出现 `if backend == GPU { ... }` 分支
- 全虚拟化的映射函数是后端无关的：
  - VirtualTensor IndexMap 在 CPU 和 GPU 语义相同
  - PackMap 在 CPU (LEA+load) 和 GPU (计算+load) 都有效
  - Activation 身份映射在所有后端都成立
  - 后端差异仅体现在 Phase 3 物化时的指令选择

### §0.7 唯一物化点与管线不变量

```
管线不变量: Phase 0-2 的输出全部是虚拟资源，Phase 3 是唯一物化点。

Phase 0-1: 语义虚拟化 (OpTrace → OpClass + AI + Bottleneck)
  ├── 输出: SemanticDAG (虚拟分类，无物理含义)
  └── 映射: OpId → OpClass + arithmetic_intensity + bottleneck_type

Phase 2 R0: 瓶颈虚拟化 (PainPointAnalyzer → OpBottleneckMap) [§3.9]
  ├── 输出: OpBottleneckMap (编译时推导的瓶颈分析，零运行时依赖)
  ├── 映射: GemmOp → GemmBottleneck { AI, ridge, MemoryBound/ComputeBound, optimal_fusion }
  └── 输入: ModelProfile × DeviceProfile × CompilerGraph → 纯静态计算（InferenceMode 已移除，见 SPEC/39 §0：编译器不假设模型形态，GEMM shapes 从图拓扑推导）

Phase 2 R1: 分组虚拟化 (PDT → FusionPlan)
  ├── 输出: FusionPlan (虚拟融合组，中间值不物化)
  └── 映射: OpId → FusionGroupId (评分消费 R0 的 OpBottleneckMap)

Phase 2 R1.5: 布局虚拟化 (LayoutNegotiator → LayoutAssignment) [§3.10]
  ├── 输出: LayoutAssignment (每个流水线阶段的协商布局 + 变换代价)
  ├── 映射: StageId × TensorRole → LayoutConstraint (流水线级联协商)
  └── 输入: FusionPlan(R1) + AccelerationRegistry(DeviceProfile) + OpBottleneckMap(R0)

Phase 2 R2: 数据流虚拟化 (VTOG → VirtualTensorMap) [布局感知]
  ├── 输出: VirtualTensorMap (哪些 tensor 是虚拟的)
  ├── 映射: TensorId → VirtualTensor(source, IndexMap) | Physical
  └── IndexMap 考虑 R1.5 协商布局 (如 HeadSplit = IndexMap::Reshape)

Phase 2 R3: 空间虚拟化 (GlobalBufferLayout)
  ├── 输出: scratch offset 规划 (虚拟偏移，未分配物理内存)
  └── 映射: TensorId → scratch_offset (布局约束影响对齐和 padding)

Phase 2 R4: 寄存器虚拟化 (RegAllocResult)
  ├── 输出: VReg → PhysReg 映射 (虚拟→物理)
  └── 映射: VRegId → PhysReg | SpillSlot

Phase 2 R5: 时序虚拟化 (StackFrame)
  ├── 输出: 最终栈帧布局 (虚拟偏移物化为 [rbp-N])
  └── 映射: 所有虚拟资源的最终物理位置 + 布局变换时序调度

Phase 3: 唯一物化点
  ├── 输入: 所有虚拟资源的映射函数 + LayoutAssignment (只读消费)
  ├── 操作: 求值每个 MapFn → 按 R1.5 协商布局生成物理机器码
  └── 输出: 单一 x86_64 / AArch64 / PTX / HIP / MSL 函数
```

**不变量验证**: 任何 Phase 0-2 的输出如果包含物理机器码字节，则违反管线不变量。

### §0.8 dtype 感知编译与去数据搬移

> **实现状态**: ✅ REQ-DTYPE-001~008 全部完成 — LoweringContext.dtype 穿透 + emit_* 参数化 + 融合组 dtype 连续性 + 布局协商 dtype 感知 + GEMM 累加策略 + Epilogue 窄化写回 + 代数重关联 + 交换律判定

> **核心原则**: dtype 从模型 TensorMeta 自动推断，编译时穿透所有 emit 函数，
> 运行时零 dtype 分支。融合组内通过代数重关联消除不必要的 dtype 变换，
> 最小化数据搬移。

#### §0.8.1 问题诊断

当前实现存在两类架构缺陷：

**缺陷 A — dtype 穿透断裂**: `LoweringContext` 无 dtype 字段，所有 32 个 `emit_*` 函数
内部硬编码 `QuantPrecision::F32`（plan_lower.rs 568 处）。dtype 传播链在 GEMM 入口处
断裂，导致 BF16/F16 模型权重被无条件 widen 到 F32 计算。

```
正确传播链 (auto_select.rs 已实现):
  TensorMeta.dtype → TypedSlot.dtype → infer_result_dtype() → VmInstr.dtype → ISA Lowering

断裂点 (plan_lower.rs):
  emit_gemm_inline_with_hook(prog, m, n, k, ctx, ...)
    ├── ctx 无 dtype 字段
    ├── 函数签名无 dtype 参数
    └── 内部 568 处 QuantPrecision::F32 硬编码
```

**缺陷 B — 重复 pack/transform**: 每个 codegen 路径独立决定数据格式，缺少全局布局协商。
融合组内存在冗余的 widen→narrow→widen 链，本可通过算子重排消除。

#### §0.8.2 dtype 穿透设计 — REQ-DTYPE-001 ~ REQ-DTYPE-002

**REQ-DTYPE-001: LoweringContext.dtype 穿透**

`LoweringContext` 新增 dtype 字段，从 `op_input_dtype(op, graph)` 自动推断：

```
传播路径:
  CompilerGraph.tensor(op.inputs[0]).dtype
    → op_input_dtype(op, graph)       [已实现]
      → LoweringContext.dtype          [新增字段]
        → emit_gemm_*(ctx)            [从 ctx.dtype 读取]
          → VmInstr { ..., dtype: ctx.dtype }
            → ISA Lowering 用 dtype.x86_elem_strategy() 选择指令
```

**约束**:
- dtype 推断方向严格单向: TensorMeta → emit 函数 → VmInstr → ISA Lowering
- 禁止反向推断或独立计算
- 每个 emit 函数的 dtype 从调用者传入，禁止在函数内部硬编码
- `op_input_dtype()` 的 `unwrap_or(QuantPrecision::F32)` 仅用于无输入 tensor 的安全回退

**REQ-DTYPE-002: 所有 emit_* 函数签名参数化**

所有 `emit_*` 函数从 `LoweringContext` 读取 dtype，替代内部硬编码：

```
变更前:
  emit_gemm_naive_inline(prog, m_dim, n, k, width, a, b, c, sym_map, seq_bound)
  内部: 567 处 QuantPrecision::F32

变更后:
  emit_gemm_naive_inline(prog, m_dim, n, k, ctx, a, b, c, seq_bound)
  内部: 用 ctx.dtype 替代所有 QuantPrecision::F32
```

**影响范围**: plan_lower.rs 全部 32 个 emit_* 函数。

#### §0.8.3 融合组内 dtype 连续性 — REQ-DTYPE-003

**REQ-DTYPE-003: 融合组内 dtype 连续性约束**

融合规则增加约束: 相邻算子 dtype 相同时，零变换直传。dtype 不同时按以下策略处理:

```
dtype 边界处理策略:

widening (存储 dtype → 累加 dtype):
  BF16 weights → F32 累加器
  ├── 在 GEMM 内部 load 时完成 (VDPBF16PS / BF16→F32 widen)
  └── 无额外 pack/transform

narrowing (累加 dtype → 存储 dtype):
  F32 累加器 → BF16 写回
  ├── 在 Epilogue 最后一步完成 (§0.8.6)
  └── 无中间变换

量化边界 (F32/BF16 → INT8/INT4):
  ├── 仅在量化算子入口发生
  └── 算子间的中间值保持原生 dtype
```

**约束**:
- 禁止融合组内部出现 widen→compute→narrow→widen 链（应通过重排消除，见 §0.8.7）
- 中间 tensor 的 dtype 等于前驱算子的输出 dtype（零拷贝直传）
- pack/transform 仅在 dtype 真正需要改变时发生

#### §0.8.4 布局协商器 dtype 感知 — REQ-DTYPE-004

**REQ-DTYPE-004: R1.5 布局协商器消费 dtype 信息**

`LayoutNegotiator` (§3.10) 收集每个加速指令的 `(dtype, layout_preference)` 对:

```
协商输入:
  ├── GEMM(BF16 weights):     layout = PanelPack(mr, nr, kc), dtype = BF16
  ├── AMX tile:               layout = AmxTile(16×32),        dtype = BF16
  ├── VNNI:                   layout = VnniPack,              dtype = U8/I8
  ├── GPU WMMA:               layout = SharedMemTile,         dtype = BF16/F16
  ├── FlashAttention:         layout = HeadSplit,             dtype = F32
  ├── RmsNorm:                layout = RowMajor,              dtype = F32(累加)
  └── SwiGLU:                 layout = Interleaved,           dtype = BF16

协商结果:
  ├── 相同 dtype + 相同布局 → 零变换
  ├── 相同 dtype + 不同布局 → PackMap 虚拟索引 (无物理 pack)
  └── 不同 dtype → 在自然搬运点零成本切换 (GEMM 累加器写回时顺便窄化)
```

**约束**: 协商器不得引入仅因 dtype 不同的额外 pack 操作。dtype 变换在计算内部完成，
不作为独立数据搬移步骤。

#### §0.8.5 GEMM 累加器 dtype 策略 — REQ-DTYPE-005

**REQ-DTYPE-005: GEMM 累加器 dtype 选择**

GEMM 累加器 dtype 由 `(输入 dtype × 硬件能力)` 联合决定:

```
累加策略决策:
  ├── 输入 BF16 + AVX-512 VDPBF16PS → BF16 load + F32 累加 (硬件原生)
  ├── 输入 BF16 + AVX2              → F32 widen + F32 累加 (软件 widen)
  ├── 输入 BF16 + GPU tensor core   → BF16 load + F32 累加 (wgmma/mma.sync)
  ├── 输入 F16 + GPU tensor core    → F16 load + F32 累加 (同上)
  ├── 输入 F32 + 任意硬件            → F32 load + F32 累加 (直通)
  └── 输入 INT8/INT4 (量化)         → Dequant → F32 累加 (量化算子路径)

累加器 dtype 传播到 Epilogue:
  Epilogue 链在累加 dtype 下执行全部后继算子 (§3.2.1 P0: EpilogueInjection)
  最后一步窄化到存储 dtype (§0.8.6)
```

**约束**: 累加 dtype 由 `dtype.x86_elem_strategy()` / `dtype.gpu_elem_strategy()` 属性方法
决定，禁止 `match dtype { BF16 => ..., F32 => ... }` 身份匹配。

#### §0.8.6 Epilogue 窄化写回 — REQ-DTYPE-006

**REQ-DTYPE-006: Epilogue 最后一步窄化**

EpilogueInjection (§3.2.1 P0) 的写回阶段将累加器 dtype 窄化为存储 dtype:

```
GEMM + Epilogue 全链路:

  累加阶段:   load(input_dtype) → FMA(accumulate_dtype) → 累加器 [累加 dtype]
  Epilogue:   累加器 → Op1(累加 dtype) → Op2(累加 dtype) → ... → 最后一步 [全在累加 dtype]
  写回阶段:   narrow(累加 dtype → 存储 dtype) → VecStore [存储 dtype]
```

**约束**:
- 窄化仅发生在 Epilogue 链的最末端
- Epilogue 内部所有算子在累加 dtype 下执行（零中间变换）
- 窄化操作通过 `dtype.narrow_strategy()` 属性方法选择，禁止手写 match

#### §0.8.7 dtype 感知的代数重关联 — REQ-DTYPE-007, REQ-DTYPE-008

**REQ-DTYPE-007: dtype 感知的代数重关联**

融合组构建时（R1 PDT 融合），当融合组内存在 dtype 边界，若相邻算子满足
交换律，重排顺序将同 dtype 算子聚集，消除中间 widen/narrow 变换:

```
原始算子链 (按图拓扑序):
  Op_A(BF16) → Cast(BF16→F32) → Op_B(F32) → Cast(F32→BF16) → Op_C(BF16)
      ╰── dtype 边界 ──╯                    ╰── dtype 边界 ──╯
  = 2 次 dtype 变换, 2 个融合断点

重排后 (交换律成立时):
  Op_A(BF16) → Op_C(BF16) → Cast(BF16→F32) → Op_B(F32)
      ╰── 同 dtype 融合 ──╯    ╰── 单次变换 ──╯
  = 1 次 dtype 变换, 1 个融合断点
```

**重排条件**:
1. 算子对满足交换律: f(g(x)) = g(f(x))
2. 重排后数值等价: 输出在目标精度范围内一致
3. 重排不破坏数据依赖: 无跨算子的值依赖

**REQ-DTYPE-008: 交换律与 dtype 无损判定矩阵**

| 算子对 | 交换律 | dtype 无损 | 可重排 |
|--------|--------|-----------|--------|
| ElemWise × ElemWise (Mul, Add, Sub) | 成立 | BF16/MXFP 精度足够 | 可重排聚集 |
| ElemWise × ElemWise (SiLU, GELU) | 成立 | BF16 精度足够 | 可重排聚集 |
| Norm → ElemWise | Norm 先执行 | Norm 输出窄化回原生 dtype 后 ElemWise 可执行 | 可保持序 |
| ElemWise → Norm | 可交换 | ElemWise 在原生 dtype 下执行后 Norm 内部 widen | 可重排 |
| GEMM 累加 → ElemWise | 累加器内执行 | 全在累加 dtype (EpilogueInjection) | 已融合 |
| Softmax → ElemWise | 不交换 | Softmax 内部需要 F32 exp/sum | 不可重排 |
| Attention → RoPE | 不交换 | RoPE 修改 Q/K 位置编码 | 不可重排 |

**判定规则**:
- 默认不可重排（保守策略）
- 仅当判定矩阵明确标注"可重排"时才允许重排
- 新增算子必须显式声明其交换律属性

#### §0.8.9 全链路数据流不变量

dtype 贯穿后的完整数据流约束:

```
模型加载:   TensorMeta.dtype (BF16/F16/F32/MXFP4/INT8/INT4)
              ↓
图构建:     OpKind × TensorMeta.dtype → 计算图节点
              ↓
SymExec:    OpTrace × TypedSlot.dtype → ComputePattern (dtype 感知)
              ↓
融合 (R1):  PDT 融合 + dtype 代数重关联 → FusionPlan (dtype 连续性约束)
              ↓
布局 (R1.5): (dtype, layout_preference) 协商 → LayoutAssignment
              ↓
数据流 (R2): VirtualTensor IndexMap 尊重 dtype + layout
              ↓
Plan Lower:  LoweringContext.dtype → emit_* 函数 → VmInstr { dtype }
              ↓
ISA Lower:   dtype.x86_elem_strategy() / dtype.gpu_elem_strategy()
              ↓
机器码:      VDPBF16PS / VMULPS / wmma.mma.sync.bf16 / v_mfma_bf16
```

**不变量验证**:
- 任何 emit 函数内部出现 `QuantPrecision::F32` 硬编码 = 违反 REQ-DTYPE-002
- 任何 dtype 变换出现在融合组内部（非 Epilogue 末端）= 违反 REQ-DTYPE-003
- 任何 pack/transform 仅因 dtype 不同而引入 = 违反 REQ-DTYPE-004

### §0.9 全链路 dtype 传播契约

> **核心原则**: dtype 传播是单向管线，每个阶段只读前一阶段的输出。
> 任何阶段不得将原始 dtype 转换为"更方便"的中间表示、独立计算/反推/硬编码 dtype、
> 或跳过 dtype 传播链中的任何环节。

```
dtype 单向管线:

  Loader (raw bytes + per-tensor DType)
    → Graph (per-tensor DType in TensorMeta)
      → JIT (per-op QuantPrecision via op_input_dtype)
        → ISA (per-instruction 策略 via dtype.x86_elem_strategy())
```

#### 三种 dtype 概念

| 概念 | 来源 | 用途 | 存储位置 |
|------|------|------|---------|
| **存储 dtype** | Loader per-tensor | 权重 blob 偏移计算 | `graph.tensor(id).dtype` |
| **计算 dtype** | DeviceProfile × 存储 dtype | GEMM 累加器 / 激活 buffer | `GraphDerivedGeometry.compute_dtype` |
| **指令 dtype** | 计算dtype × ISA 能力 | 每条 VM 指令的寄存器/指令选择 | `VmInstr.dtype` |

#### REQ-DTYPE-CHAIN-001: Loader 保留原始 dtype

SafeTensors / GGUF / ONNX loader 必须保留每个 tensor 的原始 dtype 和 raw bytes。
**禁止加载时无条件转换到 F32。**

- 量化 tensor（Q4_K / MXFP4 等）：保留 raw bytes + `GgmlDType` 元数据
- 浮点 tensor（BF16 / F16 / F32）：保留 raw bytes，不转换
- 仅以下场景允许 dtype 转换：
  1. GGUF 量化 → JIT 时 dequant（在 GEMM load 微内核内完成）
  2. BF16 权重 → F32 累加（在 GEMM `VDPBF16PS` 指令内完成）
  3. 累加结果 → 存储写回（在 Epilogue 最后一步完成）

#### REQ-DTYPE-CHAIN-002: Graph per-tensor dtype 源自 Loader

CompilerGraph 中每个 tensor 的 dtype 必须来自 Loader 的 `TensorMeta.dtype`。
`auto_graph::build_compiler_graph()` 不得用全局 `config.dtype` 覆盖 per-tensor dtype。

**约束**:
- 每个 `add_tensor_*()` 调用使用该 tensor 在 Loader 中的实际 dtype
- 图构建时不做 dtype 转换（转换在 JIT 指令层完成）

#### REQ-DTYPE-CHAIN-003: 权重布局 per-tensor dtype 感知

`MegaKernelWeightLayout` 通过 `graph.weight_layout()` 获取每个 tensor 的精确偏移。
`graph.weight_layout()` 内部按 per-tensor `dtype.size_bytes()` 计算偏移。
`GraphDerivedGeometry` 不承担权重 dtype 责任——权重布局从 graph 直接派生。

**约束**:
- 权重 blob 中不同 dtype 的 tensor 按各自 `size_bytes()` 排列
- Norm 权重（通常 F32）与 GEMM 权重（可能 BF16）的偏移各自正确

#### REQ-DTYPE-CHAIN-004: Buffer 布局用计算 dtype

`MegaKernelBufferLayout` 使用 **compute_dtype** 计算 activation / logits / sampling 大小。
compute_dtype 由 `(TensorMeta.dtype, DeviceProfile)` 推导，非硬编码。

**compute_dtype 推导规则** (当前硬件约束下的实际映射):
- BF16 模型 + AVX-512 VDPBF16PS → `compute_dtype = F32`（BF16×BF16→F32 累加）
- BF16 模型 + GPU tensor core WGMMA → `compute_dtype = F32`（BF16×BF16→F32 累加）
- BF16 模型 + AMX TDPBF16PS → `compute_dtype = F32`（BF16×BF16→F32 累加）
- F32 模型 + 任意硬件 → `compute_dtype = F32`
- FP16 模型 + GPU tensor core → `compute_dtype = F32`（FP16×FP16→F32 累加）
- 量化模型 (Q4_K 等) → `compute_dtype = F32`（dequant 后 F32 累加）
- **未来**: BF16 残差路径可能保持 `compute_dtype = BF16`（硬件 BF16 累加器成熟后）

> **架构约束**: compute_dtype 是编译时确定的常量，由 `(model TensorMeta.dtype, DeviceProfile)` 联合推导。SPEC 不硬编码为 F32 — F32 是当前所有硬件路径的实际结果，但推导逻辑允许未来演进。

#### REQ-DTYPE-CHAIN-005: GraphDerivedGeometry 演进

`GraphDerivedGeometry.dtype` 语义演化为 `compute_dtype`：
- 来自 graph 输入 tensor 的 dtype，但经过 DeviceProfile 提升
- 仅用于 buffer sizing 和 scratchpad 计算
- 权重偏移计算不依赖此字段（使用 `graph.weight_layout()`）

#### §0.9 审计命令

```bash
# Loader 层: 检查是否有 convert_to_f32 调用在非测试路径
grep -rn "convert.*to_f32\|convert_tensor_to_f32" src/loader/ --include="*.rs" | grep -v test

# Graph 层: 检查 auto_graph 是否使用全局 dt 而非 per-tensor
grep -n "config\.dtype\|let dt = " src/arch/auto_graph.rs

# Geometry 层: 检查 compute_dtype 是否只用于 buffer/scratchpad
grep -rn "compute_dtype\|geometry\.dtype" src/compiler/ --include="*.rs" | grep -v "weight_layout"

# 权重布局: 检查是否使用单一 elem_bytes
grep -rn "elem_bytes.*geo\|geo.*elem_bytes" src/compiler/mega_kernel_abi.rs
```

## §1 核心原则

```
原则：图包含一切 → 多轮虚拟化求解 → Phase 3 唯一物化

错误: compile() 手动 emit LoopBegin → emit_fusion_groups → emit Argmax → ...
正确: auto_graph 构建完整图 → 7 轮虚拟化求解 → RegAlloc → Phase 3 物化

每轮优化求解一个虚拟维度的映射函数:
  R0   → §0.2.9+§3.9 瓶颈推导 (ModelProfile × DeviceProfile → PainPoint)
  R1   → §0.2.6 虚拟计算 (OpId → FusionGroupId)
  R1.5 → §0.2.7+§0.2.11 布局协商 (流水线级联, 自然搬运点免费变换)
  R2   → §0.2.1 虚拟数据 (布局感知 VTC: TensorId → VirtualTensor | Physical)
  R3   → §0.2.3 虚拟内存 (布局约束影响对齐和 padding)
  R4   → §0.2.2 虚拟寄存器 (跨组全局 VReg → PhysReg)
  R5   → §0.2.4+§0.2.8+§0.2.9 时序规划 (PING/PONG + activation 原地 + 布局变换级联)
```

**CompilerGraph 是唯一的结构来源。** 融合管线从图拓扑自动推导：
1. 算子分类（ElemWise/Injective/Reduction/Gemm/Opaque）
2. 融合分组（哪些算子可以融合 + 融合收益评分）
3. 循环结构（层循环、生成循环）
4. 数据流（虚拟 tensor 索引映射 + 物理布局）
5. 寄存器分配（跨融合组全局规划）
6. 时序重叠（PING/PONG 延迟隐藏、WAVE 流水线）

`compile()` **仅**负责 MegaKernelFn ABI 初始化（加载参数、计算派生值），
然后委托给 `emit_fusion_groups()` 处理全部算子。

## §1.5 业务配置 → 图构建映射

> **定位**: 业务配置是"感知→规划→生成"链的最前端输入。所有业务功能遵循同一模式：
> **加载时声明 → 影响 CompilerGraph 构建 → 编译为条件分支嵌入 mega-kernel**。
> 不同业务功能映射到图的不同位置（图头部/层内/图尾部）。

### §1.5.1 配置驱动的图变体

```rust
/// mega-kernel 编译的业务配置 (从 Client 构建时收集)
pub struct MegaKernelBusinessConfig {
    // ── 输出模式 (HEAD-ROUTING.md) ──
    /// 需要编译的输出路径 (可同时编译多个, 运行时切换)
    pub output_modes: Vec<OutputMode>,

    // ── Session KV Cache 复用 (REQ-MEGA-SESSION-001) ──
    /// 是否编译 session 模式 (KV cache 跨轮次复用)
    /// 启用后: embed 融合组检查 session_position > 0 → 跳过已处理 tokens
    pub session_enabled: bool,

    // ── Multimodal Fused Hidden 注入 (REQ-MEGA-MM-001) ──
    /// 是否编译 multimodal 模式 (预计算 fused hidden state 注入)
    /// 启用后: embed 融合组读取 fused_hidden_ptr → ADD 到 token embedding
    pub multimodal_enabled: bool,

    // ── Guardrail (GUARDRAIL.md) ──
    /// 是否编译 post_node veto 探针 (每层 GEMM 后的条件 JMP)
    pub guardrail_enabled: bool,

    // ── Semantic Gatekeeper (SEMANTIC-GATEKEEPER.md) ──
    /// SG 检测层索引 + 知识注入配置
    pub semantic_gatekeeper: Option<SgConfig>,

    // ── Intent Recall (INTENT.md) ──
    /// encode_intent 的 anchor 层索引 (影响 EarlyExit 分支)
    pub intent_anchor_layer: Option<usize>,

    // ── CoT Reasoner (COT-REASONER.md) ──
    /// Step Hook 配置 (推理步骤间的思考控制)
    pub cot_step_hook: Option<CotStepConfig>,
}

/// 输出模式 — 同一模型的多种输出头形态
pub enum OutputMode {
    /// 自回归生成: lm_head → argmax → store → check → loop
    Generate {
        max_new_tokens: usize,
        eos_token_id: u32,
    },

    /// 二元分类: lm_head → 读取 pos/neg token logits → Rust softmax
    /// mega-kernel 只需将 logits 写出到 output buffer, 不做 argmax
    ClassifyBinary {
        positive_token_id: u32,
        negative_token_id: u32,
    },

    /// 多类分类: lm_head → 读取 N 个 label token logits → Rust softmax
    ClassifyMultiway {
        label_token_ids: Vec<u32>,
    },

    /// 中间层编码: 截断至 anchor 层 → pool hidden → 返回句向量
    /// 需要 EarlyExit 分支 (见 §1.5.2)
    EncodeToLayer {
        anchor_layer: usize,
        pool_mode: PoolMode,
    },
}

pub enum PoolMode {
    LastToken,
    MeanPool,
    ClsToken,
}
```

### §1.5.2 业务功能在图中的插入位置

```
图结构 (带业务功能标注):

embed_gather                                                    ← 全局权重
  │
  ├─ [if session_enabled] session_kv_restore(SessionKvRestore) ← §1.5.3 Session
  │    └── 检查 session_position > 0 → 恢复 KV cache 位置指针
  │
  ├─ [if multimodal_enabled] fused_hidden_add(MmHiddenInject)  ← §1.5.3 Multimodal
  │    └── 读取 fused_hidden_ptr → ADD 到 token embedding
  │
  ├─ [if sg_enabled] inject_knowledge(SgInject)               ← §1.5.3 SG
  │
  └→ [层循环: layer.attn_rms_norm → QKV → RoPE → MHA → O → Residual₁
       │
       ├─ [if guardrail] post_node_veto(GuardrailCheck)       ← §1.5.3 Guardrail
       │
       ├─ [if layer_idx == intent_anchor] EarlyExit(anchor)   ← §1.5.3 Intent
       │
       ├─ [if sg_enabled && layer_idx == sg_detect_layer]
       │    post_node_hidden_extract(SgDetect)                 ← §1.5.3 SG
       │
       → layer.ffn_rms_norm → Gate/Up → SwiGLU → Down → Residual₂
       │
       └─ [if cot_step_hook] step_hook_check(CotStepCheck)    ← §1.5.3 CoT
  ]  × N 层
  │
  └→ final_norm                                                ← 全局权重
      │
      ├── [generate]     → lm_head → argmax → store → check → loop
      ├── [classify_*]   → lm_head → write_logits_to_output
      └── [encode_layer] → (已由 EarlyExit 截断, 不走此路径)
```

**关键设计原则**:
- **条件编译**: `if business_feature_enabled` 在图构建时判断。未启用的功能 **完全不生成对应 op**
- **零开销保证**: 未启用的功能在 JIT 产物中不存在任何指令痕迹 (不是运行时 if-else)
- **多头共存**: `output_modes` 可包含多个变体, 图尾部为每个变体生成独立路径,
  运行时由 MegaKernelFn ABI 参数 `output_mode_selector` 决定走哪条

### §1.5.3 业务功能与图的映射细节

**Head Routing (HEAD-ROUTING.md)**:

| 输出模式 | 图构建变化 | mega-kernel 行为 |
|---------|-----------|-----------------|
| `Generate` | 图尾: lm_head → Argmax → StoreToken → CheckStopCondition → LoopBegin | 自回归循环 |
| `ClassifyBinary` | 图尾: lm_head → WriteLogits(只写 pos/neg 两个位置) | 单次前向, logits 写入 output buffer |
| `ClassifyMultiway` | 图尾: lm_head → WriteLogits(写 N 个 label 位置) | 单次前向, logits 写入 output buffer |
| `EncodeToLayer` | 层循环内: 在 anchor 层插入 EarlyExit op | 截断前向, pool hidden 写入 output |

**WriteLogits op** (新增 OpKind, 仅 classify 模式):
```
OpKind::WriteLogits { target_indices: Vec<u32> }
  ├── OpClass: Opaque
  ├── 输入: logits[1, vocab_size] (从 lm_head GEMM 累加器直接消费)
  ├── 输出: selected_logits[len(target_indices)] 写入 output buffer
  └── 与 Argmax 互斥: generate 模式用 Argmax, classify 模式用 WriteLogits
```

**EarlyExit op** (编码模式):
```
OpKind::EarlyExit { anchor_layer: usize }
  ├── OpClass: Opaque
  ├── 输入: current hidden state + loop counter (layer_idx)
  ├── 行为: if layer_idx == anchor_layer → JMP 到 pool+output 代码
  │   否则 → 继续下一层
  └── 编译为: CMP layer_counter, anchor → JE .early_exit_path
```

**Guardrail (GUARDRAIL.md)**:
```
在层循环内, 每个 GEMM 输出后插入:
  OpKind::GuardrailCheck { probe_offset: usize }
  ├── OpClass: Opaque
  ├── 编译为: 从共享内存读 veto 标志 → CMP → JE .veto_handler
  ├── veto_handler: 写 NaN 哨兵到 output → JMP 到函数末尾
  └── guardrail_enabled=false → 此 op 不插入图 → 零指令开销
```

**Semantic Gatekeeper (SEMANTIC-GATEKEEPER.md)**:
```
两层干预 (通过 hook_ctx_ptr 共享内存通信):

  1. 注入层 (embed 后): OpKind::SgInject { knowledge_offset, dim }
     ├── 将预计算的知识残差加到 hidden state
     ├── 编译为:
     │   1. 从 hook_ctx_ptr + knowledge_offset 读 knowledge_vector
     │   2. 从 hook_ctx_ptr 读 confidence
     │   3. alpha = 常量 (编译时从 MegaKernelBusinessConfig 读取)
     │   4. activation[last_token] += alpha * confidence * knowledge_vector
     │   5. 零向量检测: if confidence == 0.0 → NOP (无注入)
     └── sg_enabled=false → 此 op 不插入图 → 零指令开销

  2. 检测层 (指定层 GEMM 后): OpKind::SgDetect { detect_offset }
     ├── 提取当前层 hidden state 最后 token → 写入共享内存
     ├── 编译为:
     │   1. 从 activation + (seq_len - 1) * hidden_size 读最后 token
     │   2. STORE 到 hook_ctx_ptr + detect_offset
     │   3. release 内存序确保 Rust 侧可见
     └── sg_enabled=false → 此 op 不插入图 → 零指令开销

数据流 (每 decode step):
  generate loop 迭代:
    ├── 层循环 → SgDetect 写入 detect_hidden
    ├── Rust 通过 hook_ctx_ptr 读 detect_hidden → KnowledgeProvider.retrieve()
    ├── Rust 写入 knowledge_vector + confidence 到 hook_ctx_ptr
    └── 下一层循环 → SgInject 读取并注入 knowledge_vector

注意: mega-kernel 的 generate loop 在 JIT 内部循环，每 decode step
不返回到 Rust。SG 的 knowledge 更新需要在 generate loop 的每轮迭代
开始时通过 hook_ctx_ptr 交换数据。Rust 侧在 SgSharedMemory 中
预计算 knowledge_vector（基于上一步的 detect_hidden），JIT 侧在
当前步的 SgInject 中消费。
```

**Intent Recall (INTENT.md)**:
```
复用 EncodeToLayer 的 EarlyExit 机制:
  encode_intent(anchor_layer) ≡ encode_to_layer(anchor_layer, LastToken)
  → 同一个 JIT 编译的 EarlyExit 路径, 只是调用方式不同
  → DRY: Intent 不需要独立的图变体
```

**CoT Step Hook (COT-REASONER.md)**:
```
层循环末尾:
  OpKind::CotStepCheck { shared_mem_offset: usize }
  ├── 每个推理步骤完成后检查共享内存中的 step 控制标志
  ├── 编译为: 从共享内存 load step_flag → CMP → JE .step_handler
  └── cot_step_hook=None → 此 op 不插入图 → 零指令开销
```

**Session KV Cache 复用 (REQ-MEGA-SESSION-001)**:
```
embed 后, 层循环前:
  OpKind::SessionKvRestore
  ├── OpClass: Opaque
  ├── 输入: session_position (从 ABI StackArg 读取)
  ├── 行为:
  │   ├── session_position > 0:
  │   │   ├── 跳过 embed 融合组: input_ids[0..session_position] 已在 KV cache
  │   │   ├── 调整 input_ids 指针: input_ids_ptr += session_position * sizeof(u32)
  │   │   ├── 调整 prompt_len: prompt_len -= session_position
  │   │   └── 设置 KV cache 起始位置为 session_position
  │   └── session_position == 0: NOP (全新生成)
  ├── 编译为: CMP session_position, 0 → JE .skip_restore → 调整指针运算
  └── session_enabled=false → 此 op 不插入图 → 零指令开销
```

**Multimodal Fused Hidden 注入 (REQ-MEGA-MM-001)**:
```
embed 后, 层循环前:
  OpKind::MmHiddenInject { hidden_dim: usize }
  ├── OpClass: Opaque
  ├── 输入: token_embedding [seq, hidden_dim] + fused_hidden_ptr (ABI 参数 #18) + num_mm_tokens (ABI 参数 #19)
  ├── 行为:
  │   ├── 从 fused_hidden_ptr 读取预计算的 fused hidden state [num_mm_tokens, hidden_dim]
  │   ├── ADD 到 token embedding 的对应位置 (position-based 替换)
  │   └── 结果: 融合了文本+多模态信息的 hidden state
  ├── 编译为:
  │   ├── 从 ABI 参数 #18 加载 fused_hidden_ptr
  │   ├── 从 ABI 参数 #19 加载 num_mm_tokens
  │   ├── 循环: for i in 0..num_mm_tokens { embedding[i] += fused_hidden[i] }
  │   └── 向量化 ADD (各后端 simd_width 并行)
  └── multimodal_enabled=false → 此 op 不插入图 → 零指令开销
```

### §1.5.4 配置到编译管线的完整数据流

```
Client::new_chat(model, guardrail=Some(...), sg=Some(...))
  ↓ 收集业务配置
MegaKernelBusinessConfig {
  output_modes: [Generate, ClassifyBinary, EncodeToLayer],
  guardrail_enabled: true,
  semantic_gatekeeper: Some(SgConfig { ... }),
  intent_anchor_layer: Some(12),
  cot_step_hook: None,
}
  ↓ 传入图构建器
auto_graph::build_compiler_graph(&config)
  ↓ 根据 config 决定图的形状
  ├── embed_gather
  ├── [if sg] SgInject
  ├── [层循环: ... + guardrail checks + early exit + sg detect + cot hooks]
  ├── [generate path]: final_norm → lm_head → argmax → store → check
  ├── [classify path]: final_norm → lm_head → write_logits
  └── [encode path]: (early exit at anchor → pool → output)
  ↓
7 轮虚拟化求解 (R0-R5)
  ↓ 所有业务 op 与计算 op 统一参与融合规划和寄存器分配
Phase 3 物化 → 单一推理函数 (包含所有业务功能的条件分支)
  ↓
运行时: MegaKernelFn ABI 参数 `output_mode` 选择走哪条尾部路径
```

### §1.5.5 Output Mode Selector + JMP Table — 零重编译多头分发

> HEAD-ROUTING.md §2.2 铁律: **零权重重载、零 JIT 重编译**。
> ARCH-RUST-IS-CODEGEN 铁律: 推理时 Rust 只做一次 CALL。

#### 设计原理

所有 `output_modes` 在一次编译中全部包含到单一 mega-kernel 函数。
运行时通过 ABI 参数 `output_mode_selector` 选择执行哪条尾部路径——零重编译。

**核心问题**: generate 模式需要 **generate loop**（多次前向 + argmax + store），而
classify 模式只需 **单次前向** + write_logits。两种模式的控制流结构根本不同。

**解决方案**: generate loop **包裹整个前向 + 尾部分发**。JMP table 在 **循环体内**
的 lm_head 后面，根据 `output_mode_selector` 选择走 argmax 路径还是 write_logits
路径。classify 模式在第一次循环迭代后就 break（CheckStopCondition 的 max_tokens=1
分支），等效于单次前向。

#### ABI 变更: output_mode_selector

```
MegaKernelFn 原始签名 (16 参数):
  fn(input_ids_ptr, weight_blob_ptr, kv_cache_ptr, positions_ptr,
     aux_ptr, batch_size,                                    ← register params
     prompt_len, scratchpad_ptr, output_tokens_ptr,
     temperature_u32, top_k, top_p_u32, max_new_tokens, eos_token_id,
     hook_ctx_ptr, telemetry_ptr)                            ← stack params
    → rax: generated token count

MegaKernelFn 扩展签名 (17 参数):
  fn(input_ids_ptr, weight_blob_ptr, kv_cache_ptr, positions_ptr,
     aux_ptr, batch_size,                                    ← 寄存器参数 (数量由 ISA 决定)
     prompt_len, scratchpad_ptr, output_tokens_ptr,
     temperature_u32, top_k, top_p_u32, max_new_tokens, eos_token_id,
     output_mode_selector,                                   ← 新增 (插入 hook_ctx_ptr 前)
     hook_ctx_ptr,
     telemetry_ptr)
    → 返回值: 语义随 output_mode 变化 (见下表)

MegaKernelFn 完整签名 (19 参数, 含 Session + Multimodal):
  fn(input_ids_ptr, weight_blob_ptr, kv_cache_ptr, positions_ptr,
     aux_ptr, batch_size,                                    ← 寄存器参数 (数量由 ISA 决定)
     prompt_len, scratchpad_ptr, output_tokens_ptr,
     temperature_u32, top_k, top_p_u32, max_new_tokens, eos_token_id,
     output_mode_selector,
     hook_ctx_ptr,
     telemetry_ptr,
     session_position,                                       ← 新增: session KV cache 已处理位置
     fused_hidden_ptr,                                       ← 新增: 多模态 fused hidden state 指针
     num_mm_tokens)                                          ← 新增: 多模态 token 数量
    → 返回值: 语义随 output_mode 变化 (见下表)

MegaKernelFn 当前完整签名 (23 参数, 含 Callback + Paging + Batch):
  fn(input_ids_ptr, weight_blob_ptr, kv_cache_ptr, positions_ptr,
     aux_ptr, batch_size,                                    ← 寄存器参数 (6 个, x86_64: rdi..r9)
     prompt_len, scratchpad_ptr, output_tokens_ptr,
     temperature_u32, top_k, top_p_u32, max_new_tokens, eos_token_id,
     output_mode_selector,
     hook_ctx_ptr,
     telemetry_ptr,
     session_position,
     fused_hidden_ptr,
     num_mm_tokens,
     callback_table_ptr,                                     ← 新增: SG/Guardrail callback 函数指针表
     page_table_ptr,                                         ← 新增: PagedAttention 页表
     batch_ctx_ptr)                                          ← 新增: BatchContext 批量推理上下文
    → 返回值: 语义随 output_mode 变化 (见下表)

参数语义:
  output_mode_selector: u32
  ├── 0 = Generate (默认, 兼容旧 ABI)
  ├── 1 = ClassifyBinary
  ├── 2 = ClassifyMultiway
  ├── 3 = EncodeToLayer
  ├── 4 = ScoreTokens
  └── 5 = EncodeToLayer (anchor layer 截断, 用于 Intent SDK)

返回值语义 (rax):
  ├── Generate:         生成的 token 数量 (现有行为不变)
  ├── ClassifyBinary:   0 (2 个 compute_dtype logits 已写入 output_tokens_ptr)
  ├── ClassifyMultiway: 0 (N 个 compute_dtype logits 已写入 output_tokens_ptr)
  ├── EncodeToLayer:    0 (hidden_size 个 compute_dtype hidden 已写入 output_tokens_ptr)
  ├── ScoreTokens:      0 (target_token_ids 个 compute_dtype scores 已写入 scratchpad)
  └── EncodeAtLayer:    0 (anchor 层 compute_dtype hidden state 已写入 scratchpad)
```

**ABI 演进策略**: `output_mode_selector` 插入到 `eos_token_id` 之后、`hook_ctx_ptr` 之前。
旧调用方传 16 参数时，原 `hook_ctx_ptr` 位置被新 `output_mode_selector` 占据，
值为 NULL (0) → 被解释为 Generate 模式 → 兼容。各后端 calling convention 负责具体参数位置映射。

**ABI 参数布局 (完整 23 参数, 后端无关)**:

```
参数传递由各后端 ISA 的 calling convention 决定:
  - x86_64 SystemV: rdi, rsi, rdx, rcx, r8, r9 (6 寄存器) + 栈
  - AArch64 AAPCS: x0-x7 (8 寄存器) + 栈
  - GPU PTX/HIP/MSL: 参数 buffer 或专用寄存器传递

逻辑参数顺序 (后端无关):
  # 寄存器参数 (x86_64: 6个, AArch64: 8个, GPU: 参数 buffer)
  0. input_ids_ptr:     *const u32     — 输入 token IDs
  1. weight_blob_ptr:   *const u8      — 权重 blob
  2. kv_cache_ptr:      *mut u8        — KV cache buffer
  3. positions_ptr:     *const i32     — 位置编码表
  4. aux_ptr:           *const u8      — 辅助数据
  5. batch_size:        usize          — 批大小

  # 栈/后续参数
  6.  prompt_len:           usize      — 输入长度
  7.  scratchpad_ptr:       *mut u8    — 临时计算缓冲区
  8.  output_tokens_ptr:    *mut u32   — 输出 token 缓冲区
  9.  temperature_u32:      u32        — 采样温度 (IEEE 754 位模式)
  10. top_k:                u32        — top-k 采样参数
  11. top_p_u32:            u32        — top-p 采样参数 (IEEE 754 位模式)
  12. max_new_tokens:       usize      — 最大生成 token 数
  13. eos_token_id:         u32        — 终止 token ID
  14. output_mode_selector: u32        — 输出模式 (0=generate, 1=classify_binary, 2=classify_multiway, 3=encode, 4=score_tokens, 5=encode_to_layer)
  15. hook_ctx_ptr:         *mut u8    — Hook/SG 共享内存指针 (SEMANTIC-GATEKEEPER.md §7.4.2 SgSharedMemory 布局)
  16. telemetry_ptr:        *mut u8    — 遥测数据指针
  17. session_position:     usize      — Session KV cache 已处理位置 (0=全新)
  18. fused_hidden_ptr:     *const u8  — 多模态 fused hidden state (NULL=禁用, 实际元素类型 = compute_dtype)
  19. num_mm_tokens:        usize      — 多模态 token 数量 (0=纯文本)
  20. callback_table_ptr:   *const u8  — Callback 函数指针表 (NULL=无回调, C 风格 fn_ptr 数组, SEMANTIC-GATEKEEPER.md §7.4 + GUARDRAIL.md §7)
  21. page_table_ptr:       *const u32 — PagedAttention 页表 (NULL=连续 stride 寻址, u32[] page_table[seq_pos]=physical_page_id, 18-SYMDIM-PAGED-KV.md)
  22. batch_ctx_ptr:        *const u8  — BatchContext 批量推理上下文 (NULL=单序列 legacy 模式, 20-BATCH-CONCURRENT-INFERENCE.md)
```

**后端 codegen 职责**: 各 ISA codegen 根据自身 calling convention 将逻辑参数映射到物理位置:
- `x86_64_lower.rs`: SystemV ABI (6 寄存器 + `[rbp+16]` 起始栈参数)
- `aarch64_lower.rs`: AAPCS (x0-x7 + 栈)
- `ptx_lower.rs`: GPU 参数 buffer
- `hip_lower.rs`: AMDGPU 参数 buffer
- `msl_lower.rs`: Metal 参数 buffer

#### JMP Table 结构

```
compile() 融合组布局 (SPEC/39 §1.3):

Prologue: Load ABI params (23 参数: 6 寄存器 + 17 栈)
Prologue+: Load batch_ctx_ptr (arg 22) → non-NULL 时跳转 batch mode
Prologue+: Compute derived values
Prologue+: Generate loop begin (LoopBegin) [仅当图含 Argmax/StoreToken]
Prologue+: Compute per-iteration input_ptr
Fusion Groups: for group in plan.groups { emit_group(group) } ← 图有什么算子就发射什么
  ├── 融合组序列由 Phase 2 FusionEngine + GroupMarker 决定 (SPEC/39 §1.3.2)
  ├── LayerLoopBegin { N } 标记 → 插入层循环 (从图同构子结构推导)
  ├── PhaseDispatch 标记 → 仅当图含 Argmax 算子 (SPEC/39 §1.3.3)
  └── LayerLoopEnd 标记 → 结束层循环
OutputModeDispatch (JMP table) [仅当图含输出模式 ops]
  ├── Load output_mode_selector from ABI arg 14
  ├── CMP output_mode_selector, 0
  │   JE .generate_path
  ├── CMP output_mode_selector, 1
  │   JE .classify_binary_path
  ├── CMP output_mode_selector, 2
  │   JE .classify_multiway_path
  ├── CMP output_mode_selector, 3
  │   JE .encode_path
  ├── CMP output_mode_selector, 4
  │   JE .score_tokens_path
  └── CMP output_mode_selector, 5
      JE .encode_to_layer_path
.generate_path:                           ← Argmax/StoreToken/CheckStopCondition 融合组
  Argmax → StoreToken → CheckStopCondition
  JMP .loop_end                           ← 跳到 generate loop end
.classify_binary_path:
  WriteLogits(pos, neg) → JMP .classify_break
.classify_multiway_path:
  WriteLogits(label_0, ..., label_N) → JMP .classify_break
.classify_break:
  MOV rax, 0                              ← 返回值 = 0
  JMP .function_epilogue                  ← 直接跳出 generate loop 到函数尾部
.encode_path:
  Pool hidden state → MOV rax, 0          ← 返回值 = 0
  JMP .function_epilogue
.score_tokens_path:                       ← HEAD-ROUTING.md output_mode=4
  提取 target_token_ids 对应 logits → WriteScores → JMP .classify_break
.encode_to_layer_path:                    ← INTENT.md output_mode=5
  截断前向至 anchor 层 → 复制 hidden state 到 scratchpad → JMP .classify_break
.loop_end:
  LoopEnd                                 ← generate loop end
.function_epilogue:
  RET
```

#### VM IR 扩展: OutputModeDispatch

```rust
/// JMP table: 根据 output_mode_selector 分发到不同尾部路径。
/// HEAD-ROUTING 铁律: 零重编译 → 必须无条件发射（始终包含所有 4 条路径）。
/// 即使模型只配置了 generate 模式，JMP table 仍然存在，条件分支开销可忽略。
///
/// ISA lowering 示例 (x86_64):
///   Load selector → CMP + JE 链 → 跳转到 MarkLabel 标记的路径入口
///   使用 dispatch_labels HashMap (label_id → CodeLabel) 机制，
///   MarkLabel 在运行时 set_label 到实际代码位置。
///
/// 寄存器需求: 1 GPR (selector value), 不使用 SIMD
VmInstr::OutputModeDispatch {
    /// 从 ABI stack 读取的 selector VReg
    selector: VRegId,
    /// 每条路径对应的 label ID (通过 MarkLabel 机制绑定到代码位置)
    /// paths[0] = generate_path label ID
    /// paths[1] = classify_binary_path label ID
    /// paths[2] = classify_multiway_path label ID
    /// paths[3] = encode_path label ID
    paths: Vec<usize>,
}
```

#### VmInstr 扩展: BreakLoop

```rust
/// 跳出当前 generate loop 到函数尾部。
/// classify/encode 模式在完成 write_logits/pool 后使用。
/// 等效于: MOV rax, 0; JMP .function_epilogue
///
/// x86 lower: 设置 rax=0 后 JMP 到 LoopEnd 之后的指令位置
VmInstr::BreakLoop {
    return_value: u32,  // rax 返回值 (classify=0, encode=0)
}
```

#### compile() OutputModeDispatch 伪代码 (SPEC/39 §1.3.3)

```rust
// OutputModeDispatch (JMP table) — 仅当图含 Argmax/StoreToken 算子时存在
// HEAD-ROUTING 铁律: 零重编译 → 无条件发射，所有 4 条路径始终存在
const LABEL_GENERATE: usize = 0;
const LABEL_CLASSIFY_BINARY: usize = 1;
const LABEL_CLASSIFY_MULTIWAY: usize = 2;
const LABEL_ENCODE: usize = 3;

let selector = prog.alloc_vreg(VRegKind::Scalar, SimdWidth::Scalar);
prog.emit(VmInstr::LoadPtr {
    dst: selector,
    src: PtrExpr::StackArg(80), // output_mode_selector (具体偏移由 calling convention 决定)
});

prog.emit(VmInstr::OutputModeDispatch {
    selector,
    paths: vec![LABEL_GENERATE, LABEL_CLASSIFY_BINARY,
                LABEL_CLASSIFY_MULTIWAY, LABEL_ENCODE],
});

// .generate_path
prog.emit(VmInstr::MarkLabel { label_id: LABEL_GENERATE });
// ... emit argmax + store + check ...
prog.emit(VmInstr::LoopEnd); // generate loop end

// .classify_binary_path
prog.emit(VmInstr::MarkLabel { label_id: LABEL_CLASSIFY_BINARY });
prog.emit(VmInstr::BreakLoop { return_value: 0 });

// .classify_multiway_path
prog.emit(VmInstr::MarkLabel { label_id: LABEL_CLASSIFY_MULTIWAY });
prog.emit(VmInstr::BreakLoop { return_value: 0 });

// .encode_path
prog.emit(VmInstr::MarkLabel { label_id: LABEL_ENCODE });
prog.emit(VmInstr::BreakLoop { return_value: 0 });
```

#### 编译时条件 vs 运行时分发

**条件编译** (加载时确定, 零指令开销):
- SG / Guardrail / CoT / Intent — `if config.xxx.is_some()` 在图构建时判断
- 未启用 = 图中无对应 op = JIT 产物中零指令痕迹

**运行时分发** (推理时切换, JMP table 开销):
- OutputMode — `output_mode_selector` 参数驱动 JMP table
- 有开销: 1 次 stack load + N 次 CMP+JE (N ≤ 4, 约 3-5 条 x86 指令)
- 开销量级: < 1ns (微不足道, 相比 GEMM 的 μs 级)

**为什么 OutputMode 是运行时分发而非条件编译?**
- HEAD-ROUTING.md 铁律: 零重编译 → 必须一次编译包含所有模式
- 用户可能在同一 client 上交替调用 generate / classify_binary
- 3-5 条 CMP+JE 指令的开销可忽略不计

#### Head Routing 切换 = 改变 `output_mode_selector` 参数 = 零重编译

## §2 全模型图构建 (auto_graph)

### §2.1 图中算子

`auto_graph::build_compiler_graph(business_config)` 根据 §1.5 业务配置构建模型图。
基础骨架 + 条件插入的业务 ops:

```
embed_gather(Gather)                                        ← 全局权重
  └→ [if sg] inject_knowledge(SgInject)                     ← SG 知识注入
  └→ [层循环: layer.attn_rms_norm → QKV → RoPE → MHA → O → Residual₁
       │   [if guardrail] post_node_veto(GuardrailCheck)
       │   [if intent && layer==anchor] EarlyExit(anchor)
       │   [if sg && layer==detect] SgDetect(hidden_extract)
       → layer.ffn_rms_norm → Gate/Up → SwiGLU → Down → Residual₂
       │   [if cot_hook] CotStepCheck
  ]  × N 层
  └→ final_norm(RmsNorm)                                    ← 全局权重
  └→ lm_head(Gemm)                                          ← 全局权重
  │
  ├── [generate path]:    argmax → store_token → check_stop → LoopBegin
  ├── [classify path]:    write_logits(selected_indices)
  └── [encode path]:      (由层循环内 EarlyExit 截断 → pool → output)
```

**条件编译**: 未启用的业务功能在图中**完全不出现**，零指令开销。

### §2.2 OpKind 变体

**基础算子** (所有模式共用):

| OpKind | OpClass | 描述 | 输入 | 输出 |
|--------|---------|------|------|------|
| `Argmax { vocab_size }` | Reduction | logits 向量找最大值索引 | `logits[1, vocab_size]` | `token_id[1]` |
| `StoreToken` | Opaque | token_id 写入输出缓冲区（副作用） | `token_id[1]` | 哨兵张量 |
| `CheckStopCondition` | Opaque | 检查 EOS / max_tokens（控制流） | `token_id[1]` | 哨兵张量 |

**业务功能算子** (条件插入, §1.5):

| OpKind | OpClass | 插入位置 | 描述 |
|--------|---------|---------|------|
| `WriteLogits { target_indices }` | Opaque | lm_head 后 (classify 模式) | 从 GEMM 累加器选写指定 token logits |
| `EarlyExit { anchor_layer }` | Opaque | 层循环内 (encode 模式) | 层计数器 == anchor → JMP pool 路径 |
| `GuardrailCheck { probe_offset }` | Opaque | 层内 GEMM 后 | 读共享内存 veto 标志 → 条件 JMP |
| `SgInject { knowledge_offset, dim }` | Opaque | embed 后 | 知识残差 ADD 到 hidden state |
| `SgDetect { detect_offset }` | Opaque | 指定层 GEMM 后 | 提取 hidden 写入共享内存 |
| `CotStepCheck { shared_mem_offset }` | Opaque | 层循环末尾 | 读 step 控制标志 → 条件 JMP |
| `SessionKvRestore` | Opaque | embed 后, 层循环前 | 恢复 session KV cache 位置指针 |
| `MmHiddenInject { hidden_dim }` | Opaque | embed 后, 层循环前 | 预计算 fused hidden ADD 到 embedding |

### §2.3 层循环配置

层循环是图级配置（权重布局由图决定），生成循环是 ABI 特定包装：

```rust
pub struct LayerLoopConfig {
    pub num_layers: usize,
    pub weight_stride: usize,           // 每层权重字节跨度
    pub layer_blob_base_offset: usize,  // 层权重 blob 起始偏移
    pub layer_weight_input_indices: Vec<usize>,  // 层内权重的 graph.inputs 索引
}
```

生成循环由 `compile()` 在 `emit_fusion_groups()` 前后发射。

### §2.4 图来源：auto_graph 驱动 (ARCH-UNIFIED-GRAPH-SOURCE)

**铁律：CompilerGraph 的唯一来源是 gllm auto_graph 系统。禁止在 `graph_builders.rs` 中独立手写图构建函数。**

#### §2.4.1 数据流

```
模型文件 (safetensors/GGUF/ONNX)
    │
    ├─ config.json → ModelConfig → ModelGeometry (几何常量)
    │
    └─ auto_graph (gllm/src/arch/auto_graph.rs)
        │
        ├─ 从 tensor names 推导架构特征
        ├─ 条件节点: 按模型配置选择性启用
        ├─ 层循环展开
        │
        └─ auto_graph::build_compiler_graph(config, tensors) → CompilerGraph
              │
              ├─ op_type 字符串 → OpKind 枚举映射
              ├─ 重复层模式检测 → 层模板折叠 (§2.5)
              │
              └─ CompilerGraph (4 模板 + hetero_loop 或 均匀 loop)
```

#### §2.4.2 op_type → OpKind 映射表

转换器的核心职责是将 OnnxGraph 的 `op_type` 字符串映射为 `OpKind` 枚举：

| YAML op_type / OnnxGraph op_type | OpKind | 分类 |
|----------------------------------|--------|------|
| `Gather` | `OpKind::Gather` | Structural |
| `SimplifiedLayerNormalization` | `OpKind::RmsNorm` | Norm |
| `MatMul` | `OpKind::Gemm` | Complex |
| `QkNorm` | `OpKind::QkNorm` | Norm |
| `ValueNorm` | `OpKind::ValueNorm` | Norm |
| `DualRotaryEmbedding` | `OpKind::RoPE` | Complex |
| `Attention` | `OpKind::MultiHeadAttention` | Complex |
| `Add` (残差) | `OpKind::Residual` | Structural |
| `GELU` | `OpKind::Gelu` | Elementwise |
| `GeGLU` | `OpKind::GeGlu` | Elementwise |
| `SwiGLU` | `OpKind::SwiGlu` | Elementwise |
| `Mul` (layer_scalar) | `OpKind::ElementwiseMul` | Elementwise |
| `LogitSoftcap` | `OpKind::LogitSoftcap` | Elementwise |

**未识别的 op_type → 编译报错 (NO_SILENT_FALLBACK)**。转换器不做猜测，新增算子必须在此表注册。

#### §2.4.3 Gemma-4 层结构 (参考实现)

Gemma-4 E2B 的 auto_graph 定义了以下层结构：

```
input_layernorm → Q_proj → K_proj → V_proj → QkNorm → ValueNorm
    → DualRoPE → Attention → O_proj → Residual₁ (post-attention)
    → post_attention_layernorm → pre_feedforward_layernorm
    → gate_proj → GELU → up_proj → GeGLU → down_proj
    → post_feedforward_layernorm → Residual₂ → layer_scalar
```

与通用 Llama/Qwen 模型的差异：
- 4 个 norm（vs 2 个）：input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm
- layer_scalar：每层乘性缩放因子
- GeGLU（vs SwiGLU）：gate_proj 用 GELU 激活
- QkNorm + ValueNorm：Q/K L2 归一化 + V 无参数 RMSNorm

#### §2.4.4 ModelConfig 缺失字段补充

当前 `ModelConfig` 未解析以下 config.json 字段，必须补全：

| config.json 字段 | ModelConfig 新增字段 | 说明 |
|------------------|---------------------|------|
| `use_double_wide_mlp` | `use_double_wide_mlp: Option<bool>` | Gemma-4 segments ≥3 用 2× intermediate |
| `final_logit_softcapping` | `final_logit_softcapping: Option<f32>` | Gemma-4 logit 缩放 (30.0) |
| `hidden_activation` | 已有 `hidden_act`，需映射到 FfnActivation | GeGLU vs SwiGLU 选择 |

### §2.5 异构层模板折叠

异构模型（如 Gemma-4 E2B）有多种层类型。auto_graph 展开为 35 个独立层节点，
编译器通过模式检测折叠为 K 个模板 + 异构层循环：

#### §2.5.1 折叠算法

```
输入: OnnxGraph (35 层展开)
输出: CompilerGraph (4 模板 + HeteroLayerLoopConfig)

1. 按层索引分组，检测层内算子序列是否相同
2. 按 (attention_type × ffn_size) 分类为 4 种模板
3. 对每种模板，取第一个实例的图结构作为模板图
4. 生成 HeteroLayerLoopConfig 描述循环结构
```

Gemma-4 E2B 的 4 种模板：

| 模板 | attention | FFN intermediate | 层数 | 权重布局 |
|------|-----------|------------------|------|---------|
| sliding_small | sliding (head_dim=256) | 6144 | 4×5=20 | PerLayerWeightLayout (sliding_small) |
| full_small | global (head_dim=512) | 6144 | 1×5=5 | PerLayerWeightLayout (full_small) |
| sliding_large | sliding (head_dim=256) | 12288 | 4×2=8 | PerLayerWeightLayout (sliding_large) |
| full_large | global (head_dim=512) | 12288 | 1×2=2 | PerLayerWeightLayout (full_large) |

#### §2.5.2 HeteroLayerLoopConfig

```rust
pub struct HeteroLayerLoopConfig {
    pub num_segments: usize,
    pub sliding_per_segment: usize,
    pub templates: Vec<LayerTemplate>,
    pub segment_schedule: Vec<SegmentDescriptor>,
    pub activation_aliases: Vec<(TensorId, TensorId)>,
}

pub struct LayerTemplate {
    pub attention_type: AttentionType,  // Sliding / Global
    pub ffn_size: FfnSize,              // Small / Large
    pub ops: Vec<OpId>,                 // 模板图内的算子 ID
    pub weight_layout: PerLayerWeightLayout,
    pub weight_stride: usize,
}

pub struct SegmentDescriptor {
    pub template_indices: Vec<usize>,   // 本段内的模板序列索引
}
```

#### §2.5.3 权重布局同步

`PerLayerWeightLayout` 的字段数量和顺序**完全由 auto_graph 推导**：
- 通用 Llama: 11 个权重 (attn_norm, w_q, w_k, w_v, w_o, ffn_norm, w_gate, w_up, w_down)
- Gemma-4: 14 个权重 (+w_q_norm, w_k_norm, pre_ffn_norm, post_ffn_norm, layer_scalar)

转换器从模板的 `tensor_patterns` + `nodes` 自动推导权重布局，不需要手写 `compute_per_layer_bytes`。

### §2.6 废弃代码清除

以下代码在 SPEC 统一后**必须删除**：

| 文件 | 删除内容 | 原因 |
|------|---------|------|
| `graph_builders.rs` | `decoder_model()`, `decoder_model_hetero()`, `build_layer_body()` | 被 OnnxGraphConverter 替代 |
| `mega_kernel_abi.rs` | `compute_per_layer_bytes()` 手写字节计算 | 由模板自动推导权重布局替代 |
| `mega_kernel_abi.rs` | `MegaKernelWeightLayout::from_config()` 手写偏移计算 | 由 HeteroWeightLayout 从模板自动生成替代 |

**保留的代码**：
- `mega_kernel_abi.rs`: `ModelMegaConfig`, `HeteroLayerConfig`, ABI 参数布局定义
- `mega_kernel_abi.rs`: `MegaKernelBufferLayout` (运行时 scratchpad 布局)
- `mod.rs`: `compile()` (JIT 编译入口，改为消费 OnnxGraphConverter 产出的 CompilerGraph)

## §3 多轮全局优化算法 — 7 轮虚拟化求解 (R0-R5 + R1.5)

### §3.1 管线总览

```
auto_graph 构建图
  ↓
Phase 0: ScalarOpRegistry (OpTrace + ComputePattern)
  ↓
Phase 1: SemanticDAG (OpClass 自动分类 + 算术强度 + 瓶颈)
  ↓
Phase 2: 7 轮虚拟化求解 (每轮求解一个虚拟维度的映射函数)
  ├── R0: 瓶颈推导 → §3.9 性能建模
  │   PainPointAnalyzer: (ModelProfile × DeviceProfile) → OpBottleneckMap
  │   编译时推导每个 GEMM 的瓶颈位置, 零运行时依赖
  │
  ├── R1: 虚拟计算求解 → §0.2.6 虚拟计算
  │   PDT 拓扑融合: OpId → FusionGroupId
  │   中间结果在寄存器中流转，不物理化
  │
  ├── R1.5: 虚拟布局求解 → §0.2.11 虚拟布局 [新增]
  │   LayoutNegotiator: 加速指令布局约束 → 动态协商最大公约布局
  │   考虑所有加速指令需求, 适配出满足最多约束的布局
  │   尽可能保持数据不变换基础上应用各种加速指令
  │
  ├── R2: 虚拟数据求解 → §0.2.1 虚拟数据
  │   VTC VTOG + Greedy: TensorId → VirtualTensor(source, IndexMap) | Physical
  │   布局感知: 虚拟 tensor 的 IndexMap 考虑协商布局
  │
  ├── R3: 虚拟内存求解 → §0.2.3 虚拟内存
  │   全局 scratch 规划: TensorId → scratch_offset
  │   布局约束影响对齐和 padding
  │
  ├── R4: 虚拟寄存器求解 → §0.2.2 虚拟寄存器
  │   全局 RegAlloc: VRegId → PhysReg | SpillSlot
  │   跨融合组复用，消除不必要的 spill
  │
  └── R5: 虚拟时序+激活+布局变换求解 → §0.2.4+§0.2.8+§8.2
      StackFrame + 时序布局变换: 所有虚拟资源 → 最终物理位置
      零成本虚拟变换 + PING/PONG 异步重叠 + activation 原地更新
  ↓
Phase 3: 唯一物化点 — 按协商布局生成指令, 不做布局决策
```

**每轮的输入/输出形成增强线性链**（§0.7 管线不变量）：
```
R0(OpBottleneckMap) → R1(+FusionPlan) → R1.5(+LayoutAssignment) → R2(+VirtualTensorMap)
  → R3(+GlobalBufferLayout) → R4(+RegAllocResult) → R5(StackFrame) → Phase 3(机器码)
```

后续轮次可以回溯修改前序轮次的决策（如 R3 发现 scratch 不足 → R2 取消某些虚拟化
→ 该 tensor 回退为物理分配），但 Phase 3 不可修改任何决策。

### §3.2 Round 1: 虚拟计算求解 — OpId → FusionGroupId

> **对应**: §0.2.6 虚拟计算 | **参考**: TVM (OSDI 2018) Post-Dominator Tree 融合算法

**后支配树（Post-Dominator Tree）驱动的拓扑融合**，替代当前 7 条 FusionRule 模式匹配。
映射函数：`OpId → FusionGroupId`，同组内的中间 tensor 不物理化（在寄存器中流转）。

```
算法:
1. 构建 SemanticDAG 的 PDT (post-dominator tree)
   a. 对每个节点计算其 post-dominator（所有路径必须经过的下一个汇合点）
   b. PDT 边: node → ipostdom(node)
   c. 数据结构: Vec<OpId> ipostdom (index = node_id, value = post-dominator node_id)

2. 拓扑序遍历节点

3. 对每个节点，检查其 PDT 子树中的可融合消费者:
   a. 若 node 是 Gemm → 收集 PDT 子树中的 Reduction/ElemWise 消费者
   b. 若 node 是 ElemWise → 收集 PDT 子树中的 ElemWise/Injective 消费者
   c. 若 node 是 Opaque → 不融合

4. 基于 OpClass 层级决策融合模式:

   OpClass 融合层级（优先级从高到低）:
   ├── Gemm → anchor, 融合 Reduction/ElemWise/Injective 作为 epilogue
   ├── Reduction → 可作为 Gemm epilogue (如 Argmax + lm_head)
   ├── ElemWise → 可链接为 LoopFusion 或作为 Gemm epilogue
   ├── Injective → 可链接为 LoopFusion
   └── Opaque → 永不融合 (Standalone)

5. 融合收益评分 (§4) 决定是否实际融合
```

**PDT 构建算法选择**：

```
选择: Lengauer-Tarjan 算法 (O(n α(n))) — 标准高效算法
替代: 简单迭代法 (O(n²)) — 对小图 (<100 节点) 足够

gllm 场景: auto_graph 单层模板 ≈ 20 ops, 全模型 ≈ 30 ops
→ 简单迭代法即可, O(n²) = O(900) ≈ 微秒级
```

### §3.2.1 GEMM 融合策略侧重

GEMM 是融合系统的**核心锚点**。不同 GEMM 场景有明确的融合策略优先级：

```
GEMM 融合策略优先级 (按收益排序):

P0: EpilogueInjection (最高优先 — 直接消除内存写回)
├── GEMM 累加器 → 后续 ElemWise/Reduction 直接在寄存器中完成
├── 收益: 消除 output tensor 的 1 次写 + N 次读 (N = epilogue ops 数)
├── 典型: lm_head + Argmax, QKV + SiLU, Gate + SwiGLU
├── 约束: 累加器寄存器 + epilogue 临时寄存器 ≤ 可用 SIMD 寄存器
└── Roofline 影响: memory-bound GEMM (AI < ridge) → 收益最大 (1.0×)
                    compute-bound GEMM (AI ≥ ridge) → 收益缩减 (ridge/AI ×)

P1: NormIntoGemm / TileLevelFusion / ComputeRoot
├── 前驱 Norm(RmsNorm) 输出直接喂入 GEMM
├── 决策阈值: predecessor_output_bytes vs L1 × 75%
│   ├── > 75% L1 → TileLevelFusion (嵌入 MC 循环, 按 MC 行切分)
│   └── ≤ 75% L1 → ComputeRoot (完整计算后驻留 L1/L2)
├── 收益: 消除 norm output tensor 的写回 + 重读
└── 典型: attn_rms_norm → QKV, ffn_rms_norm → Gate/Up

P2: QkvSharedInput / FFNBlock
├── 结构性融合 (共享 pack_a, 消除重复 pack)
├── QkvSharedInput: Q/K/V 三个 GEMM 共享同一输入 → 单次 pack_a
├── FFNBlock: Gate + Up GEMM (共享输入) → activation → Mul → Down
└── 收益: pack_a 从 3 次降为 1 次, 内存流量减少 ~3×

P3: LoopFusion
├── 逐元素算子链合并为单循环
├── 收益: 消除中间 tensor 的写回 + 重读
├── 约束: 累计中间 tensor ≤ 75% L1
└── 典型: SiLU → Mul, Residual Add
```

**GEMM 融合与推理模式的关系**：

```
Latency 模式 (batch=1, M=1, GEMV):
├── AI ≈ 2N/(K+N+1) ≈ 2 → MemoryBound
├── EpilogueInjection 收益最大 (消除内存瓶颈)
├── TileLevelFusion 通常不适用 (M=1 无 MC 循环可嵌入)
└── ComputeRoot 是 Norm→GEMV 的主要路径

Throughput 模式 (batch=64, M=64):
├── AI ≈ 128NK/(64K+KN+64N) → ComputeBound
├── EpilogueInjection 收益缩减但仍正 (score > 0)
├── TileLevelFusion 对大 norm 输出更有效 (MC 有意义)
└── QkvSharedInput 收益显著 (3× pack_a 节省在大 batch 下放大)
```

### §3.2.2 关键融合规则矩阵

| 生产者 | 消费者 | 融合规则 | 收益 | 优先级 |
|--------|--------|----------|------|--------|
| Gemm (lm_head) | Reduction (Argmax) | EpilogueInjection | logits 不写回内存 | P0 |
| Gemm (Gate/Up) | ElemWise (SwiGLU) | EpilogueInjection | 累加器直接激活 | P0 |
| Gemm (O_proj) | ElemWise (Residual) | EpilogueInjection | O 累加器直接加残差 | P0 |
| RmsNorm | Gemm (QKV) | NormIntoGemm | 归一化输出直接喂 GEMM | P1 |
| RmsNorm (大输出) | Gemm | TileLevelFusion | 按 MC 行嵌入 GEMM 循环 | P1 |
| 3× Gemm (Q/K/V) | — | QkvSharedInput | 共享 pack_a | P2 |
| Gemm(Gate)+Gemm(Up) | SwiGLU+Mul | FFNBlock | 结构性融合 | P2 |
| ElemWise (SiLU) | ElemWise (Mul) | LoopFusion | 消除中间写回 | P3 |

**Argmax 特殊处理**：`Gemm + Reduction → EpilogueInjection` 是一条通用规则。
当 lm_head（Gemm）后接 Argmax（Reduction）时，PDT 融合自动将其归为
EpilogueInjection——logits 永不写回内存，argmax 直接在 GEMM 累加器寄存器中完成。

### §3.3 Round 2: 虚拟数据求解 — 布局感知的 TensorId → VirtualTensor | Physical

> **对应**: §0.2.1 虚拟数据 | **参考**: VTC (OSDI'26) VTOG + Global Greedy
> **关键联动**: 消费 R1.5 LayoutAssignment — 虚拟 tensor 的 IndexMap 必须尊重协商布局。
> 例如: R1.5 协商出 GEMM 输出为 HeadSplit，R2 将其建模为 VirtualTensor { IndexMap::Reshape },
> 无需物理变换（3×hidden = num_heads × head_dim，纯 stride 重计算）。

> **参考**: VTC (OSDI'26) — Virtual Tensor Opportunity Graph (VTOG) + Global Greedy Algorithm

**消除所有可消除的中间 tensor 物理化**。

```
VTC 算法映射到 gllm:

1. 构建 VTOG (Virtual Tensor Opportunity Graph):
   ├── 节点 = 图中所有 tensor
   ├── 边 = 两个 tensor 之间可以建立虚拟映射的机会
   ├── 边权重 = 虚拟化后节省的延迟 (消除一次物理写 + 一次物理读)
   └── 冲突检测: 互斥的虚拟化策略 (一个 tensor 只能有一个虚拟来源)

2. Global Greedy 选择:
   ├── 计算每条边的离散导数: w(e) = ℓ(C ∪ {e}) - ℓ(C)
   ├── 贪心选择收益最大的非冲突边
   └── 复杂度 O(|V|²), gllm 全模型 ≈ 30 ops → 微秒级

3. VTC 优化类型:
   ├── Type I (始终有利): 全连续虚拟 tensor (Reshape, Slice)
   │   → 零成本索引变换, 总是虚拟化
   ├── Type II (需要分析): 部分连续 (Transpose, Permute)
   │   → 需评估 memory coalescing 影响
   └── 不可虚拟化: 跨融合组边界, 多消费者外部引用
```

**def-use 分析规则**：

```
分析每个 tensor 的 def-use 链:
├── 单消费者 + 下游融合 → 虚拟化 (Type I, 零物理数据)
│   例: Reshape/Transpose/Slice → 不产生物理数据, 只记录索引变换
│   例: GEMM 输出 → epilogue → 下游 op → 全在寄存器中
├── 多消费者 + 融合组内 → 部分虚拟化 (一次物理化, 多消费点虚拟引用)
├── 跨融合组 → 必须物理化 (写入 scratch buffer)
└── 跨循环迭代 (activation) → 原地更新 buffer
```

**虚拟 Tensor 数据结构**：

```rust
/// 索引变换类型 (VTC §3.2 mapping function F)
#[derive(Debug, Clone)]
pub enum IndexMap {
    /// f(i) = i — 零开销
    Identity,
    /// f(i) = i + offset — Reshape, Slice
    Offset(isize),
    /// f(i) = perm[i] — Transpose (需要维度重排)
    Permute(Vec<usize>),
    /// f(i, j) = (j, i) — 矩阵转置
    Transpose2D,
    /// f(i) = scale * i — 广播/重复
    Broadcast { factor: usize },
}

/// 虚拟 tensor (VTC §3.1: (F, P₁, ..., Pₙ))
#[derive(Debug, Clone)]
pub struct VirtualTensor {
    /// 物理来源 tensor
    pub source: TensorId,
    /// 索引变换函数
    pub index_map: IndexMap,
    /// 字节偏移 (叠加在 source offset 之上)
    pub byte_offset: usize,
    /// 元素类型
    pub dtype: DType,
    /// 逻辑形状
    pub shape: Vec<SymDim>,
}

/// VTC Round 2 输出: 虚拟化决策映射
pub struct VirtualTensorMap {
    /// TensorId → VirtualTensor (如果虚拟化)
    pub virtual_map: HashMap<TensorId, VirtualTensor>,
    /// 必须物理化的 tensor (跨组/多消费者)
    pub physical_set: HashSet<TensorId>,
    /// 总节省字节数 (用于反馈到 Round 3 内存规划)
    pub bytes_saved: usize,
}
```

**与现有 TensorPtrResolver 整合**：

```
现有: TensorPtrResolver { map: HashMap<TensorId, TensorPtrSource> }
TensorPtrSource = Activation | Weight { offset } | Intermediate { offset } | Output { offset }

扩展:
TensorPtrSource = Activation
               | Weight { offset }
               | Intermediate { offset }
               | Output { offset }
               | Virtual { source: TensorId, index_map: IndexMap, byte_offset: usize }

materialize() 行为:
├── Activation/Weight/Intermediate/Output → 与现有相同 (base + offset)
├── Virtual { source, Identity, 0 } → 直接返回 source 的 materialize 结果
├── Virtual { source, Offset(n), 0 } → materialize(source) + n
└── Virtual { source, Permute, 0 } → 需要 ISA Lowering 生成索引重算指令
```

**收益量化**：每个虚拟化的 tensor 节省 `tensor_bytes × 2`（一次写 + 一次读）的内存带宽。

### §3.4 Round 3: 虚拟内存求解 — TensorId → scratch_offset

> **对应**: §0.2.3 虚拟内存 | 输入: FusionPlan(R1) + VirtualTensorMap(R2)

**整个模型的 scratch buffer 作为单一资源全局优化**（超越当前逐 tensor interval coloring）。

```
输入: Round 1 的 FusionPlan + Round 2 的 VirtualTensorMap
算法:
1. 构建 tensor 生命周期区间 (从定义点到最后消费点)
2. Virtual tensor 不参与物理内存分配 (已在 Round 2 虚拟化)
3. 跨层循环迭代的 activation tensor 分配到固定 buffer (原地更新)
4. 层内中间 tensor 通过 interval coloring 复用空间
5. 全局权重 tensor (embed, lm_head, final_norm) 不分配 scratch

优化目标: 最小化 scratchpad 总大小
约束:
├── 同一时刻活跃的 tensor 不能重叠
├── 对齐要求 (cacheline_bytes 边界)
├── 跨迭代 tensor 必须有独立 slot
└── 虚拟 tensor 的物理来源必须已分配

与现有 buffer_alloc.rs 的关系:
├── Phase A-C: 现有 interval coloring + cacheline 对齐 (已实现)
└── Phase D: 升级为全局感知 (考虑虚拟 tensor 跳过 + 跨迭代 activation 固定 slot)
```

**数据流传递结构**：

```rust
/// Round 3 输出: 全局 scratch 布局
pub struct GlobalBufferLayout {
    /// 物理分配的 tensor slots
    pub slots: Vec<BufferSlot>,
    /// scratch 总大小 (字节)
    pub total_bytes: usize,
    /// 虚拟 tensor 的物理来源映射 (VirtualTensor.source → slot)
    pub virtual_source_map: HashMap<TensorId, BufferSlot>,
    /// 跨迭代 activation 固定 slot (原地更新)
    pub activation_slots: HashMap<TensorId, usize>,
}
```

### §3.5 Round 4: 虚拟寄存器求解 — VRegId → PhysReg | SpillSlot

> **对应**: §0.2.2 虚拟寄存器 | 输入: VmProgram + GlobalBufferLayout(R3)

**跨融合组的 VReg → 物理寄存器全局映射**。

```
当前问题: 每个融合组独立分配寄存器 → 跨组切换时全部 spill/reload
正确做法: 整个 VmProgram 全局分配

算法:
1. 分析所有 VmInstr 的 VReg def/use
2. 构建 interference graph (VReg 生命周期重叠 = 干扰)
3. 图着色分配 (GPR / YMM / ZMM 分离分配)
4. 跨融合组保持公共 VReg 映射 (如 weight_ptr, activation_ptr)
5. spill 只在物理寄存器不够时发生 (而非每次组切换)

半 VM 的角色: VmInstr 序列是全局寄存器分配器的输入
RegAllocator 消费完整 VmProgram, 而非逐组消费

与现有 RegAllocator 的关系:
├── Phase A-C: 现有全局 RegAlloc (已消费完整 VmProgram) — 无需修改
└── Phase D: 优化跨组 VReg 复用 (识别跨组活跃的公共 VReg, 避免重复分配)
```

### §3.6 Round 5: 虚拟时序+激活求解 — 所有资源 → 物理位置

> **对应**: §0.2.4+§0.2.8 虚拟控制+激活 | 输入: RegAllocResult(R4) + GlobalBufferLayout(R3)

**PING/PONG 延迟隐藏 + WAVE 流水线 + 最终栈帧**。

```
时序规划层:
├── 识别可重叠的计算:
│   ├── 层 N 计算 + 层 N+1 权重预取 (PING/PONG)
│   └── GEMM 计算 + 下游 epilogue 计算 (WAVE pipeline)
├── 计算最终栈帧布局:
│   ├── callee-save 区
│   ├── spill 区 (Round 4 分配器决定)
│   ├── scratch 区 (Round 3 布局决定)
│   └── 对齐约束
└── 输出 StackFrame → Phase 3 消费

Round 间数据流传递结构 (7 轮增强线性链):
┌─────────────────────────────────────────────────────────────────┐
│ R0: OpBottleneckMap                                            │
│    ↓ (每个 GEMM 的瓶颈 + 推荐融合策略 + exec_pattern)          │
│ R1: FusionPlan                                                 │
│    ↓ (fusion groups + op_to_group mapping)                     │
│ R1.5: LayoutAssignment [新增]                                  │
│    ↓ (每个 Stage 的协商布局 + 变换代价 + 自然搬运点分类)        │
│ R2: VirtualTensorMap (布局感知)                                 │
│    ↓ (virtual_map + physical_set + bytes_saved)                │
│ R3: GlobalBufferLayout                                         │
│    ↓ (slots + total_bytes + activation_slots + layout padding)  │
│ R4: RegAllocResult (现有, 不变)                                │
│    ↓ (全局 VReg → 物理寄存器映射 + spill 决策)                  │
│ R5: StackFrame (现有, 不变)                                    │
│    ↓ (callee-save + spill + scratch 布局 + 布局变换时序)        │
│ Phase 3: ISA Lowering (按 LayoutAssignment 生成指令)            │
└─────────────────────────────────────────────────────────────────┘
```

### §3.7 半 VM 指令集扩展

当前 VmInstr 需要以下扩展以支持虚拟 tensor 和全局优化：

```
现有指令 (无需修改):
├── LoadPtr { dst, src: PtrExpr }      — ptr 加载
├── Gemm { ... }                       — GEMM 计算
├── Elementwise { ... }                — 逐元素操作
├── Norm { ... }                       — 归一化操作
├── Argmax { ... }                     — argmax (可融入 GEMM)
├── LoopBegin/LoopEnd                  — 循环控制
└── StoreToken / CheckStopCondition    — mega-kernel 专用

新增指令 (Phase D):
├── VirtualTensorDeclare { tid, source, index_map }
│   — 声明虚拟 tensor, 不产生物理加载, 仅记录索引映射
│   — RegAllocator 追踪: source 和 virtual 共享生命周期
├── MaterializeVirtual { dst, virtual_tid }
│   — 将虚拟 tensor 物理化到 scratch (仅在跨组边界需要时)
│   — 等价于: dst = LoadPtr(source_base + compute_offset(index_map))
└── ActivationSwap { primary, secondary }
    — 层循环 activation 原地更新的双缓冲 ping-pong
    — 仅交换 ptr, 不复制数据
```

**Phase A-C 不需要新增指令**——虚拟 tensor 在 Phase A-C 阶段通过
`TensorPtrSource::Virtual` 在 `materialize()` 中静默处理,
不需要显式 VmInstr。

### §3.8 compile() 全链路

```
compile(config)
  ├── 0. 性能建模 (Phase 1.5: R0)
  │     PerformanceModel::analyze(&graph, &device_profile)
  │     └── 每个 GEMM 的瓶颈分析 + 最优策略推荐
  ├── 1. auto_graph::build_compiler_graph()
  │     └── 融合组序列由图拓扑推导 (SPEC/39 §1.3.2)
  ├── 2. Phase 0: ScalarOpRegistry::with_defaults()
  ├── 3. Phase 1: SemanticDAG::from_graph(&graph, &registry)
  │     └── OpClass 自动分类 + AI + Bottleneck
  ├── 4. Phase 2: 多轮全局优化 (由性能模型 + 布局协商驱动)
  │     ├── R0:  PainPointAnalyzer → OpBottleneckMap [§3.9]
  │     ├── R1:  FusionEngine::fuse_pdt() → FusionPlan [§3.2]
  │     ├── R1.5: LayoutNegotiator::negotiate() → LayoutAssignment [§3.10, 新增]
  │     ├── R2:  DataFlowOptimizer::eliminate() → VirtualTensorMap [§3.3, 布局感知]
  │     ├── R3:  GlobalMemoryPlanner::plan() → GlobalBufferLayout [§3.4]
  │     ├── R4:  GlobalRegPlanner::plan() → RegAllocResult [§3.5]
  │     └── R5:  TemporalPlanner::plan() → StackFrame [§3.6, 时序布局变换]
  ├── 5. Phase 3: 虚拟资源 → ISA Lowering (唯一物化点)
  │     ├── emit_fusion_groups() → VmProgram (层循环 + 全部算子)
  │     ├── RegAllocator::allocate(&program) → RegAllocResult
  │     ├── StackFrame::compute(&alloc_result, &profile, total_scratch)
  │     └── X86Lower / AArch64Lower / PtxLower / HipLower / MslLower
  │         (按 LayoutAssignment 协商布局生成指令, 不做布局决策)
  └── 6. 输出 MegaKernelCompileOutput
```

### §3.9 性能建模子系统 — (模型×硬件) 瓶颈分析驱动融合

> **对应**: §0.2.9 虚拟执行模式 + §4 融合评分 | **定位**: R1 之前的 Round 0 (R0)

**问题**：不同模型在不同硬件下的性能瓶颈不同，融合策略必须据此自适应选择：

```
模型 × 硬件 × 模式 → 瓶颈 → 最优融合策略

SmolLM-135M / AVX2 / Decode:
  GEMV (M=1):  AI ≈ 2, bandwidth-bound
  → EpilogueInjection 收益最大 (消除带宽瓶颈)
  → NormIntoGemm 无额外收益 (norm 输出已在 L1)
  → TileLevelFusion 不适用 (M=1 无 MC 循环)

Llama-70B / AVX-512 / Decode:
  量化 GEMV (M=1):  AI ≈ 2, bandwidth-bound (量化降低带宽需求但仍为瓶颈)
  → EpilogueInjection 收益最大
  → QkvSharedInput 收益显著 (QKV GEMV 共享 pack_a)
  → 虚拟权重布局收益最大 (pack_a 对 M=1 无意义)

Llama-8B / AVX-512 / Prefill:
  GEMM (M=512):  AI ≈ 200+, compute-bound
  → EpilogueInjection 收益缩减但仍正
  → TileLevelFusion 对 Norm→GEMM 有效 (MC 有意义)
  → 物理权重 pack 有价值 (提升 L2 命中率)

同一模型 decode vs prefill:
  M=1: memory-bound → 融合重点 = 消除内存访问
  M=512: compute-bound → 融合重点 = 减少计算浪费 (寄存器压力平衡)
```

**性能建模数据结构**：

```rust
/// 单个 GEMM 的瓶颈分析结果
pub struct GemmBottleneck {
    /// 哪个 GEMM (QKV / O_proj / Gate / Up / Down / lm_head)
    pub gemm_role: GemmRole,

    /// 算术强度 (FLOPS / bytes_accessed)
    pub arithmetic_intensity: f64,

    /// 硬件 ridge point (peak_flops / peak_bandwidth)
    pub ridge_point: f64,

    /// 瓶颈类型
    pub bottleneck: BottleneckType,

    /// 推荐的 GEMM 融合策略 (§3.2.1 P0-P3)
    pub optimal_fusion: FusionPriority,

    /// 预估每种融合策略的收益倍率
    pub fusion_benefits: HashMap<FusionMode, f64>,

    /// 推荐的执行模式 (§0.2.9)
    pub exec_pattern: ExecutionPattern,
}

/// 瓶颈类型
pub enum BottleneckType {
    /// AI < ridge: 带宽瓶颈 → 融合消除内存访问收益最大
    MemoryBound { bandwidth_utilization: f64 },
    /// AI ≥ ridge: 计算瓶颈 → 融合收益缩减但寄存器效率仍有益
    ComputeBound { compute_utilization: f64 },
    /// 延迟瓶颈 (小矩阵, kernel launch overhead 等)
    LatencyBound { estimated_latency_ns: f64 },
}

/// GEMM 在模型中的角色 (影响融合策略选择)
pub enum GemmRole {
    QkvProjection,    // QKV → QkvSharedInput 候选
    OutputProjection, // O_proj → EpilogueInjection + ResidualAdd
    GateUpProjection, // Gate+Up → FFNBlock + SwiGLU
    DownProjection,   // Down → EpilogueInjection
    LmHead,           // lm_head → EpilogueInjection + Argmax
}

/// 全模型性能分析结果 (R0 输出)
pub struct OpBottleneckMap {
    /// 每个 GEMM op 的瓶颈分析
    pub gemm_bottlenecks: HashMap<OpId, GemmBottleneck>,
    /// 每个 Reduction/ElemWise op 的开销估计
    pub op_costs: HashMap<OpId, OpCost>,
    /// 设备 profile (缓存大小、带宽、FLOPS)
    pub device: DeviceProfile,
    /// 模型 profile (维度、层数、head 数)
    pub model: ModelProfile,
}
```

**编译时痛点推导 — 从 (模型图 + 设备参数) 直接推导瓶颈**：

> **关键**: 痛点不是运行时测量发现的，而是**编译时从静态信息推导的**。
> 模型结构 (维度/层数/head数/vocab) 和设备参数 (带宽/FLOPS/缓存/SIMD) 都是已知的。
> 将两者结合就能精确计算出每个算子的瓶颈位置，不需要任何运行时测量。

```
编译时已知输入:
├── 模型图: CompilerGraph (每个 op 的 M/N/K/SymDim)
├── 模型配置: ModelProfile { hidden_dim, num_heads, num_layers, vocab_size, intermediate_size }
├── 设备参数: DeviceProfile { peak_flops, peak_bandwidth, l1_bytes, l2_bytes, simd_width, ... }
└── 推理模式: Latency (M=1) / Throughput (M=seq_len)

推导链:
  GEMM 形状 (M, N, K) × DeviceProfile
    → FLOPS = 2MKN, bytes = (MK+KN+MN)×sizeof
    → AI = FLOPS / bytes
    → ridge = peak_flops / peak_bandwidth
    → AI vs ridge → MemoryBound / ComputeBound
    → 每种融合策略的收益 = f(bottleneck, bytes_saved, device_constraints)

痛点矩阵 (模型 × 硬件):
────────────────────────────────────────────────────────────────────────
模型\硬件      AVX2         AVX-512      GPU SM80     GPU SM90
────────────────────────────────────────────────────────────────────────
SmolLM decode  带宽瓶颈     带宽瓶颈     带宽瓶颈     带宽瓶颈
  QKV GEMV     AI=0.5       AI=0.5       AI=0.5       AI=0.5
  lm_head      带宽瓶颈     带宽瓶颈     带宽瓶颈     带宽瓶颈
               AI=0.5       AI=0.5       AI=0.5       AI=0.5
               vocab 极大   vocab 极大   vocab 极大   vocab 极大
               → Epilogue   → Epilogue   → Epilogue   → Epilogue

Llama-70B decode 带宽瓶颈   带宽瓶颈     带宽瓶颈     带宽瓶颈
  量化 GEMV    量化降低带宽  量化降低带宽  量化降低带宽  量化降低带宽
               → 虚拟权重   → 虚拟权重   → 虚拟权重   → 虚拟权重

Llama-8B prefill 计算瓶颈   计算瓶颈     计算瓶颈     计算瓶颈
  QKV GEMM     AI=200+      AI=200+      AI=200+      AI=200+
               → TileLevel  → TileLevel  → TMA流水    → TMA+wgmma
               → 物理pack   → 物理pack   → shared mem → 异步流水
────────────────────────────────────────────────────────────────────────

GPU 特有痛点 (从 DeviceProfile 推导):
├── SM90 (H100): TMA 带宽 ~3.35 TB/s, FP16 ~990 TFLOPS
│   → 小 GEMM: 延迟瓶颈 (kernel launch overhead)
│   → 大 GEMM: 计算瓶颈 → wgmma 异步流水 + 双缓冲
│   → Attention: shared memory bank 冲突 → 需 PaddingMap
│
├── SM80 (A100): 带宽 ~2 TB/s, FP16 ~312 TFLOPS
│   → 大多数推理: 带宽瓶颈 → 融合消除内存访问
│   → 大 batch prefill: 计算瓶颈 → CP_ASYNC + Stage
│
├── Wave scheduling:
│   → occupancy = f(register_pressure, shared_memory_per_block)
│   → 融合增加寄存器压力 → occupancy 降低 → wave 减少 → 吞吐下降
│   → 评分函数惩罚: reg_pressure × wave_cost(device.waves_per_sm)
│
└── PING/PONG 延迟隐藏:
    → 层 N 计算 + 层 N+1 权重预取 (CPU: PREFETCHT0, GPU: cp.async.bulk)
    → 需要: 权重加载延迟 > 计算延迟 × overlap_ratio
    → 从 DeviceProfile 计算: weight_load_latency = weight_bytes / bandwidth
    → compute_latency = FLOPS / peak_flops
    → 可重叠 iff weight_load_latency > compute_latency × 0.5
```

**ModelProfile 与 DeviceProfile 联合推导**：

```rust
/// 模型配置 (从模型文件加载时获取, 编译时已知)
pub struct ModelProfile {
    pub hidden_dim: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl ModelProfile {
    /// 从 CompilerGraph 推导每个 GEMM 的形状 (SPEC/39: 编译器不假设模型形态)
    /// GEMM M 维度从图的 OpKind::Gemm { m: SymDim } 推导（Symbolic → 运行时绑定）
    pub fn gemm_shapes(&self, graph: &CompilerGraph) -> Vec<(GemmRole, (SymDim, usize, usize))> {
        // 从图拓扑推导：遍历图中所有 Gemm ops，读取 m/n/k SymDim
        // M 维度保持 Symbolic（运行时通过 ShapeBinding 绑定）
        graph.iter_gemm_ops().map(|op| {
            let (m, n, k) = op.gemm_dims(); // SymDim × usize × usize
            (op.gemm_role(), (m, n, k))
        }).collect()
    }
}

/// 编译时痛点分析 (零运行时依赖)
pub struct PainPointAnalyzer;

impl PainPointAnalyzer {
    /// 纯静态分析: (ModelProfile × DeviceProfile × CompilerGraph) → 痛点列表
    /// InferenceMode 参数已移除 (SPEC/39 §0: 编译器不假设模型形态)
    /// GEMM shapes 从 CompilerGraph 的 OpKind.Gemm.m (SymDim) 推导
    pub fn analyze(
        model: &ModelProfile,
        device: &DeviceProfile,
        graph: &CompilerGraph,
    ) -> Vec<PainPoint> {
        let ridge = device.peak_flops() as f64 / device.peak_bandwidth() as f64;

        model.gemm_shapes(graph).into_iter().map(|(role, (m, n, k))| {
            // M 维度为 SymDim，max_for_allocation 仅用于 buffer 上界
            let m_val = m.max_for_allocation();
            let flops = 2.0 * m_val as f64 * n as f64 * k as f64;
            let bytes = (m*k + k*n + m*n) as f64 * 4.0; // F32
            let ai = flops / bytes;
            let bottleneck = if ai < ridge {
                MemoryBound { bandwidth_util: ai / ridge }
            } else {
                ComputeBound { compute_util: ridge / ai }
            };

            PainPoint {
                gemm_role: role,
                shape: (m, n, k),
                arithmetic_intensity: ai,
                bottleneck,
                optimal_strategy: Self::pick_strategy(role, &bottleneck, m, device),
            }
        }).collect()
    }
}
```

**瓶颈分析算法**：

```
PerformanceModel::analyze(graph, device_profile):
  1. 对图中每个 Gemm op:
     a. 计算 FLOPS = 2 × M × N × K
        M = SymDim 解析 (decode: 1, prefill: seq_len)
        N, K = 从图结构推导 (hidden_dim, intermediate_size, vocab_size)
     b. 计算 bytes_accessed = (M×K + K×N + M×N) × sizeof(dtype)
     c. 计算 AI = FLOPS / bytes_accessed
     d. 计算 ridge_point = device.peak_flops / device.peak_bandwidth
     e. 判定瓶颈: AI < ridge → MemoryBound, else → ComputeBound
     f. 估算每种融合策略的收益:
        - EpilogueInjection: 消除 M×N bytes 写回 → 按瓶颈缩放
        - NormIntoGemm: 消除 M×K bytes norm 输出 → 按 L1 容量缩放
        - QkvSharedInput: 消除 2× pack_a → pack 字节数 × 2
        - TileLevelFusion: 仅当 M > mc_min 时有效
     g. 选择最高收益策略为 optimal_fusion

  2. 对图中每个 Reduction/ElemWise op:
     a. 计算 FLOPS (算子特定: Argmax = 2×N, SiLU = 4×N, ...)
     b. 计算 bytes_accessed = input_bytes + output_bytes
     c. 估算融合后的收益 (消除 input_bytes 的写回+重读)

  3. 输出 OpBottleneckMap → 传入 R1 FusionEngine
```

**与融合引擎的集成**：

```
R1 FusionEngine 消费 OpBottleneckMap:

  score_fusion(producer, consumer, bottleneck_map):
    bottleneck = bottleneck_map.gemm_bottlenecks[producer]
    bytes_saved = compute_bytes_saved(producer, consumer)

    // 瓶颈感知缩放 (替代通用 roofline 缩放)
    scale = match bottleneck.bottleneck {
      MemoryBound { .. } => 1.0,  // 消除带宽瓶颈 → 全额收益
      ComputeBound { util } => ridge / AI,  // 计算瓶颈 → 缩减收益
      LatencyBound { .. } => 0.5,  // 延迟瓶颈 → 适度收益
    };

    // 融合策略特定加权 (§4.3)
    strategy_weight = bottleneck.fusion_benefits[fusion_mode];

    // 寄存器压力惩罚 (从 device_profile 获取可用寄存器数)
    reg_penalty = estimate_reg_pressure(producer, consumer, device_profile);

    bytes_saved as f64 * scale * strategy_weight - reg_penalty
```

**测量校准**（Phase D 深化）：

```
编译时分析提供初始估算。Phase D 引入运行时测量校准:

  1. 首次编译: 使用分析模型 (屋顶线估算)
  2. 运行一次: 测量每个 GEMM 的实际耗时和带宽利用率
  3. 校准: 用实测数据修正 OpBottleneckMap
  4. 重编译: 用校准后的瓶颈分析驱动融合

校准数据存储在 ModelJitCache 中 (与编译产物关联):
  CalibrationData {
    gemm_timings: HashMap<GemmRole, Duration>,
    bandwidth_utilization: f64,
    compute_utilization: f64,
  }

这实现了"测量 → 计算 → 优化"的闭环，而非纯静态启发式。
```

### §3.10 布局协商子系统 — 动态约束求解最大公约布局

> **对应**: §0.2.11 虚拟布局 | **定位**: R1.5 (R1 之后, R2 之前)
> **核心原则**: 不是写死的规则表，是融合规划时的**动态约束协商器**。
> 每个加速指令**声明**自己的布局需求，协商器**动态求解**满足最多约束的布局。

#### §3.10.1 问题定义

```
问题: 一条融合链中的每个算子可能被不同加速指令加速,
      而不同加速指令要求不同的数据布局。

例: QKV GEMM → SiLU → ResidualAdd
  AVX-512 GEMM:    需要 B 为 PanelPacked(14, 32, kc) 格式
  AVX-512 VNNI:    需要 B 为 u8×i8 lane-packed 格式 (量化时)
  SiLU:            可以消费 row-major 或 interleaved
  ResidualAdd:     可以消费任意布局 (element-wise 无偏好)

当前做法 (错误):
  GEMM 输出 row-major → SiLU 读取 row-major → 写回 row-major
  → 没有利用任何布局优化, 数据在 row-major 和 packed 之间反复变换

正确做法 (动态协商):
  1. 收集所有加速指令的布局约束
  2. 协商出一个同时满足 GEMM 输出和 SiLU 输入的布局
  3. 如果 GEMM 可以直接输出 interleaved, SiLU 也消费 interleaved
     → 零变换! 整条链在同一布局中完成
```

#### §3.10.2 加速指令布局注册表

每个加速指令是一个**声明式条目**，不是硬编码的代码路径：

```rust
/// 加速指令的布局声明 (注册到 AccelerationRegistry)
pub struct AccelerationDecl {
    /// 唯一标识
    pub id: AccelerationId,
    /// 硬件要求 (哪些设备支持)
    pub hw_req: HardwareRequirement,
    /// 可加速的算子模式
    pub applicable_patterns: Vec<OpPattern>,
    /// 输入布局约束 (可以声明多种可接受布局)
    pub input_layouts: Vec<LayoutConstraint>,
    /// 输出布局 (使用此指令后输出是什么布局)
    pub output_layout: LayoutConstraint,
    /// 性能收益函数 (由 §3.9 PainPointAnalyzer 驱动)
    pub benefit_fn: fn(&OpBottleneck) -> f64,
}

/// 布局约束 — 不是一种布局, 是一组可接受的布局
pub enum LayoutConstraint {
    /// 标准 row-major (对齐到 align_bytes)
    RowMajor { align_bytes: usize },
    /// 列优先
    ColMajor { align_bytes: usize },
    /// GEMM panel pack (mr×kc 块)
    PanelPacked { mr: usize, nr: usize, kc: usize },
    /// Head-split: [seq, heads, head_dim] 而非 [seq, hidden]
    HeadSplit { num_heads: usize, head_dim: usize },
    /// Interleaved pair: (gate[i], up[i]) 交替排列
    InterleavedPairs,
    /// AMX tile: 16×32 BF16 格式
    AmxTileBF16 { rows: usize, cols: usize },
    /// GPU shared memory tile (padding 避免银行冲突)
    SharedMemTile { tile_rows: usize, tile_cols: usize, padding_bytes: usize },
    /// GPU TMA 2D 兼容布局 (128-byte aligned)
    TmaAligned2D { tile_m: usize, tile_n: usize },
    /// VNNI u8×i8 packed
    VnniPacked4,
    /// 任意布局均可 (element-wise 无偏好)
    Any,
}
```

**注册表示例** (不是写死规则, 是声明式注册):

```
AccelerationRegistry::register(AccelerationDecl {
    id: "avx512_gemm_f32",
    hw_req: AVX512,
    applicable_patterns: [Gemm(M, N, K)],
    input_layouts: [
        // A 可以 row-major (微核 broadcast)
        RowMajor { align: 64 },
        // B 需要 panel-packed 以最大化 L1 命中
        PanelPacked { mr: 14, nr: 32, kc: dynamic },
    ],
    output_layout: RowMajor { align: 64 },  // GEMM 输出是 row-major
    benefit_fn: |bottleneck| bottleneck.memory_bound_benefit(),
});

AccelerationRegistry::register(AccelerationDecl {
    id: "amx_tile_bf16",
    hw_req: AMX,
    applicable_patterns: [Gemm(M, N, K) where K >= 32],
    input_layouts: [
        AmxTileBF16 { rows: 16, cols: 32 },  // A 必须 tile 格式
        KPairInterleaved,                      // B 必须 K-pair interleaved
    ],
    output_layout: RowMajor { align: 64 },
    benefit_fn: |bottleneck| bottleneck.compute_bound_benefit() * 2.0,
});

AccelerationRegistry::register(AccelerationDecl {
    id: "gpu_sm90_wgmma",
    hw_req: SM90,
    applicable_patterns: [Gemm(M, N, K)],
    input_layouts: [
        TmaAligned2D { tile_m: 64, tile_n: dynamic },  // TMA 2D prefetch
    ],
    output_layout: RegisterTile { rows: 64, cols: dynamic },  // wgmma 输出在寄存器
    benefit_fn: |bottleneck| bottleneck.tensor_core_benefit(),
});

AccelerationRegistry::register(AccelerationDecl {
    id: "flash_attn_tile",
    hw_req: Any,
    applicable_patterns: [Attention(seq, heads, kv_len)],
    input_layouts: [
        // Q/K/V 需要 head-split 以便 per-head tile
        HeadSplit { num_heads, head_dim },
        // 或者 shared memory tile (GPU)
        SharedMemTile { tile_rows: 256, tile_cols: head_dim, padding: 16 },
    ],
    output_layout: RowMajor { align: 64 },
    benefit_fn: |bottleneck| 2.0,  // FlashAttention 总是 2× 以上收益
});
```

#### §3.10.3 动态协商算法 — 流水线时序级联

> **核心原则**: 不是找"一个全局最大公约布局"。是**流水线时序级联协商**。
> A→B 之间本来就要发生数据搬运（寄存器→内存、全局→共享内存、GEMM store 等），
> 这个搬运本身就是**免费的布局变换机会**——直接按 B 的需求布局搬过去。
>
> 每个自然数据搬运点都是一个"免费变换窗口"，协商器识别并利用这些窗口。

```
流水线中的自然数据搬运点 (Natural Movement Points):

┌─────────────────────────────────────────────────────────────────┐
│ 搬运类型              发生场景                变换机会         │
├─────────────────────────────────────────────────────────────────┤
│ 寄存器 → 内存 (store)  GEMM 结果写回 scratch   写入时换布局     │
│ 内存 → 寄存器 (load)   下一个 op 加载输入       加载时换布局     │
│ 内存 → 内存 (copy)     activation 跨组传递     拷贝时换布局     │
│ 全局 → 共享 (GPU)      TMA prefetch            预取时 swizzle   │
│ 共享 → 寄存器 (GPU)    warp load               直接消费         │
│ 寄存器 → 寄存器        EpilogueInjection       无搬运=无变换    │
└─────────────────────────────────────────────────────────────────┘
```

```
LayoutNegotiator::negotiate(
    fusion_plan: &FusionPlan,
    registry: &AccelerationRegistry,
    device: &DeviceProfile,
    pain_points: &OpBottleneckMap,
) -> LayoutAssignment

算法:

对 FusionPlan 中每个融合组 group:
  // === Step 1: 识别流水线阶段和自然搬运点 ===
  // 融合组内部不是扁平的算子列表, 而是流水线阶段序列
  // 每两个相邻阶段之间存在自然数据搬运

  stages = group.pipeline_stages()  // 按 R1 融合决策划分的阶段
  // 例: [Stage0: Norm→GEMM(fused), Stage1: SiLU(fused), Stage2: ResidualAdd(fused)]
  // 每两个 Stage 之间 = 一个自然搬运点

  // === Step 2: 查询每个阶段的加速指令和布局偏好 ===
  for stage in stages:
    stage.accel_features = registry.query(stage.pattern, device.hw_caps)
    stage.preferred_output = best_output_layout(stage.accel_features, pain_points)
    stage.preferred_input = best_input_layout(stage.accel_features, pain_points)

  // === Step 3: 流水线级联协商 ===
  // 核心逻辑: 利用自然搬运点做免费布局变换
  //
  // 思路: 不是让所有 op 共享一个布局
  // 而是: 每个 op 用自己最优的布局, 在自然搬运点免费切换到下一个 op 的布局

  layout_plan = LayoutPlan::new()

  for i in 0..stages.len() - 1:
    (stage_A, stage_B) = (stages[i], stages[i+1])
    movement = classify_movement(stage_A, stage_B)
    // movement = Store | Load | Copy | RegisterDirect | GpuPrefetch | ...

    match movement:
      // 情况 1: 寄存器直传 (EpilogueInjection)
      // 数据在寄存器中, 不经过内存 → 无法变换布局
      // 必须: stage_A 的输出布局与 stage_B 的输入布局兼容
      RegisterDirect =>
        if stage_A.preferred_output compatible_with stage_B.preferred_input:
          layout_plan.set(stage_B.input, stage_A.preferred_output)  // 不变
        else:
          // 不兼容 + 无搬运点 → 无法免费变换
          // 解决: 退回 stage_A 的 Any 布局, 或接受 stage_B 的次优布局
          compromise = find_compromise(stage_A.output_options, stage_B.input_options)
          layout_plan.set_with_cost(stage_B.input, compromise, penalty)

      // 情况 2: 寄存器 → 内存 (store) — 免费变换窗口!
      // stage_A 计算结果在寄存器中, 要写回 scratch buffer
      // 写回时可以按 stage_B 的偏好布局写入!
      RegisterToMemory =>
        // 写回操作本来就要发生, 写入 stage_B 偏好的布局 = 零额外成本
        layout_plan.set(stage_B.input, stage_B.preferred_input)
        layout_plan.set_transform(stage_A→stage_B, FreeStoreTransform {
          source_layout: stage_A.output_in_registers,
          target_layout: stage_B.preferred_input,
          cost: 0,  // store 本来就要做, 换布局不增加开销
        })

      // 情况 3: 内存 → 内存 (copy) — 免费变换窗口!
      // activation 跨融合组传递时本来就要拷贝
      MemoryToMemory =>
        layout_plan.set(stage_B.input, stage_B.preferred_input)
        layout_plan.set_transform(stage_A→stage_B, FreeCopyTransform {
          source_layout: stage_A.preferred_output,
          target_layout: stage_B.preferred_input,
          cost: 0,  // 拷贝本来就要做, 换布局只是改变 stride
        })

      // 情况 4: GPU 全局 → 共享内存 (TMA prefetch) — 免费变换窗口!
      GpuGlobalToShared =>
        layout_plan.set(stage_B.input, stage_B.preferred_input)
        layout_plan.set_transform(stage_A→stage_B, FreePrefetchTransform {
          source_layout: stage_A.preferred_output,
          target_layout: SharedMemTile { padding: bank_conflict_free },
          cost: 0,  // TMA 预取本来就要做, swizzle 在 prefetch 中完成
        })

  // === Step 4: 计算收益和变换代价 ===
  total_benefit = 0
  total_transform_cost = 0
  for stage in stages:
    // 该阶段使用的加速指令 → 收益
    satisfied = stage.accel_features.filter(|f| f.layout_satisfied(layout_plan[stage]))
    total_benefit += satisfied.map(|f| f.benefit).sum()

    // 该阶段的变换代价
    transform = layout_plan.get_transform(stage)
    total_transform_cost += transform.cost

  return LayoutAssignment {
    stage_layouts: Map<StageId, LayoutConstraint>,
    transforms: Map<(StageId, StageId), Transform>,
    total_benefit,
    total_transform_cost,
  }
```

**关键洞察——"免费变换窗口"**:

```
错误理解: 找一个全局最大公约布局, 所有 op 都用这个布局
  → 不可能! GEMM 要 PanelPacked, Attention 要 HeadSplit, 矛盾

正确理解: 流水线时序级联, 每个自然搬运点都是免费变换机会

  Stage A (GEMM)                自然搬运点              Stage B (RoPE)
  ┌──────────┐                  ┌─────────┐             ┌──────────┐
  │ 寄存器中   │ ──── store ──── │ scratch │ ─── load ─→ │ 消费      │
  │ RowMajor  │   写入时换成     │ buffer  │   直接读    │ HeadSplit│
  │ 累加结果   │   HeadSplit!    │HeadSplit│             │ 直接用    │
  └──────────┘   成本=0         └─────────┘   成本=0     └──────────┘
       ↑                                          ↑
  GEMM 用自己的                               RoPE 用自己的
  最优布局计算                                 最优布局消费

  中间的 store/load 本来就要发生!
  把"按 RowMajor 写"换成"按 HeadSplit 写" = 改变 store stride = 零额外指令!
```

**示例: 完整解码层的流水线级联协商**

```
流水线阶段 (AVX-512):

  [RmsNorm] --RegisterDirect→ [QKV GEMM] --Store→ [RoPE] --RegisterDirect→ [Attn]

  RmsNorm:
    无特殊布局偏好 → 输出 RowMajor
    → RegisterDirect 到 QKV: GEMM 接受 RowMajor 输入 ✅ 零变换

  QKV GEMM:
    AVX-512 加速: B 需要 PanelPacked(14,32,kc) → 权重提前 pack
    输出在寄存器: RowMajor [seq, 3×hidden]
    → Store 到 scratch: 免费变换窗口!
    → 按 RoPE 偏好写入 HeadSplit [seq, heads, head_dim]
    → 只需改变 store stride: 从 stride=3H 改为 stride=head_dim
    → 零额外指令! 本来就要 store, 改 stride 不增加成本

  RoPE:
    消费 HeadSplit → per-head rotation → 直接在 HeadSplit 布局上操作
    输出: 仍然是 HeadSplit (rotation 不改变布局)
    → RegisterDirect 到 Attention: Attn 需要 HeadSplit ✅ 零变换

  Attention:
    消费 HeadSplit → per-head tile → Q/K/V 已经按 head 切分
    无需 reshape/transpose! 直接按 [heads, seq, head_dim] 遍历

  整条链: 1 个免费变换 (GEMM store 时改变 stride), 其余零变换!
```

#### §3.10.4 协商结果示例 — 流水线级联

```
示例 1: CPU AVX-512 解码层 — GEMM store 免费变换

  流水线: RmsNorm --RegDirect→ QKV_GEMM --Store→ RoPE --RegDirect→ Attention

  级联协商:
  ┌──────────┐  RegDirect   ┌──────────┐   Store(免费!)   ┌──────────┐  RegDirect  ┌──────────┐
  │ RmsNorm  │ ───────────→ │ QKV GEMM │ ────────────────→ │ scratch  │ ──────────→ │ RoPE     │
  │ RowMajor │   零变换     │ RowMajor │  改 stride 写入   │HeadSplit │  零变换    │HeadSplit │
  │ 输出     │   ✅         │ 累加器   │  成本=0!          │ buffer   │  ✅        │ 消费     │
  └──────────┘              └──────────┘                   └──────────┘             └──────────┘
                                                                                      │
                                                               RegDirect (零变换) ✅
                                                                                      ↓
                                                                              ┌──────────┐
                                                                              │Attention │
                                                                              │HeadSplit │
                                                                              │ 直接消费  │
                                                                              └──────────┘

  总变换: 1 次免费 (GEMM store 时改 stride), 0 次付费
  总收益: QKV 用 AVX-512(14×32) + RoPE 用 HeadSplit rotation + Attn 用 HeadSplit tile


示例 2: GPU SM90 解码层 — TMA prefetch 免费变换

  流水线: RmsNorm --TMA→ QKV_GEMM --Reg→ RoPE --Reg→ Attn --Reg→ O_proj --Store→ ...

  级联协商:
  ┌──────────┐   TMA prefetch   ┌──────────┐   RegisterDirect   ┌──────────┐
  │ 权重 B   │ ────────────────→ │ 共享内存  │ ─────────────────→ │ wgmma    │
  │ RowMajor │  免费变换!       │ TmaTile  │  零变换 ✅         │ 寄存器中  │
  │ 在全局内存 │ swizzle+align   │ bank-free│                    │ RegisterTile│
  └──────────┘  成本=0!         └──────────┘                    └──────────┘
                                                                                    │
                    RegisterDirect (EpilogueInjection) — 零变换 ✅                  ↓
  ┌──────────┐   Store 到 scratch   ┌──────────┐   TMA prefetch   ┌──────────┐
  │ RoPE     │ ───────────────────→ │ scratch  │ ────────────────→ │ 共享内存  │
  │ 寄存器中  │  免费变换!          │ HeadSplit│  免费变换!       │ AttnTile │
  │ rotation │  改 stride 写入      │ 布局     │  swizzle+pad     │ bank-free│
  └──────────┘  成本=0!            └──────────┘  成本=0!         └──────────┘

  整条链: 4 次自然搬运点 = 4 次免费变换, 0 次付费显式变换


示例 3: CPU AVX-512 Gate/Up FFN — InterleavedPairs 免费变换

  流水线: RmsNorm --RegDirect→ GateUp_GEMM --Store→ SwiGLU --RegDirect→ Down_GEMM

  关键洞察: GateUp GEMM 输出 [seq, 2×intermediate] = gate 和 up 交织排列
  SwiGLU = silu(gate[i]) * up[i] — 天然消费 InterleavedPairs 布局!

  ┌──────────┐  RegDirect  ┌──────────┐    Store(免费!)    ┌──────────┐
  │ RmsNorm  │ ──────────→ │GateUp    │ ──────────────────→ │ scratch  │
  │ RowMajor │  零变换 ✅  │ GEMM     │  直接写入           │Interleave│
  │          │             │ 累加器中  │  InterleavedPairs! │ Pairs    │
  └──────────┘             └──────────┘  成本=0!            └──────────┘
                                                               │
                          RegisterDirect (零变换) ✅            ↓
  ┌──────────┐  Store(免费!)   ┌──────────┐
  │ SwiGLU   │ ──────────────→ │ scratch  │
  │ silu*g   │  结果写回       │ RowMajor │  → Down GEMM 消费 RowMajor ✅
  │ pairs!   │  RowMajor      │          │
  └──────────┘  成本=0!       └──────────┘

  GateUp GEMM 的累加器按 [gate0, up0, gate1, up1, ...] 排列
  → SwiGLU 直接消费这个布局 → silu(gate[i]) * up[i] = 一条 SIMD 指令
  → store 时改回 RowMajor → Down GEMM 消费 RowMajor
  → 全链 2 次免费变换, 0 次付费
```

#### §3.10.5 布局协商与融合评分的联动

```
融合评分 (§4) 现在考虑布局协商结果:

score_fusion_with_layout(producer, consumer, layout_assignment):
  base_score = score_fusion(producer, consumer, bottleneck_map)  // §4.1

  layout_benefit = layout_assignment.total_benefit  // 满足的加速指令总收益
  transform_penalty = layout_assignment.transform_penalty  // 需要的布局变换代价

  adjusted_score = base_score + layout_benefit - transform_penalty

  // 如果布局协商发现整条链可以零变换 → 额外奖励
  if layout_assignment.transform_penalty == 0:
    adjusted_score *= ZERO_TRANSFORM_BONUS  // 1.2× 奖励

  return adjusted_score
```

**这意味着**: 融合决策不再只看"算子是否可融合"，还要看"融合后的布局协商是否零变换"。
两条同样可融合的链，布局协商收益更高的那条胜出。

#### §3.10.6 与管线的集成位置

```
R0: PainPointAnalyzer → 每个 GEMM 的瓶颈 + 加速指令收益 (§3.9)
  ↓
R1: FusionEngine → PDT 拓扑融合 + 评分 → FusionPlan
  ↓
R1.5: LayoutNegotiator → 动态布局协商 (本节) [新增]
  ↓ 输入: FusionPlan + AccelerationRegistry(DeviceProfile) + PainPoints
  ↓ 输出: LayoutAssignment (每个 tensor 的协商布局 + 变换代价)
  ↓
R2: DataFlowOptimizer → VTC 虚拟化 (现在是布局感知的)
  ↓ 虚拟 tensor 的 IndexMap 考虑协商布局
  ↓ 例: HeadSplit 作为 VirtualTensor IndexMap::Reshape 实现 (零物理变换)
  ↓
R3-R5: 内存/寄存器/时序 (布局约束影响对齐和 padding)
  ↓
Phase 3: 仅物化 (按协商布局生成指令, 不做布局决策)
```

## §4 融合收益评分 — 性能模型驱动

### §4.1 评分函数 — 瓶颈感知

> **核心变化**: 评分不再依赖通用启发式，而是消费 §3.9 OpBottleneckMap 的分析结果。
> 每个 (producer, consumer) 对的评分基于其**在当前模型+硬件组合下的真实瓶颈位置**。

```rust
fn score_fusion(
    producer: &SemanticNode,
    consumer: &SemanticNode,
    graph: &CompilerGraph,
    bottleneck_map: &OpBottleneckMap,  // [新增] R0 输出
) -> f64 {
    // 基础分: 消除的中间 tensor 大小
    let bytes_saved: usize = compute_bytes_saved(producer, consumer, graph);

    // 瓶颈感知缩放 (从 R0 性能模型获取, 非通用公式)
    let scale = match bottleneck_map.get_bottleneck(producer) {
        MemoryBound { bandwidth_util } => {
            // 带宽越接近饱和, 融合消除内存访问收益越大
            // util=0.9 → scale=1.0 (全额), util=0.3 → scale=0.3 (收益有限)
            bandwidth_util.max(0.1)
        }
        ComputeBound { compute_util } => {
            // 计算瓶颈下, 消除内存访问帮助有限
            // 但仍正: 减少内存压力 → 更多 L1/L2 给计算 tile
            let ridge = bottleneck_map.device.ridge_point();
            let ai = bottleneck_map.arithmetic_intensity(producer);
            (ridge / ai).min(1.0)
        }
        LatencyBound { .. } => {
            // 小矩阵延迟瓶颈 → 融合减少 kernel launch / 分支开销
            0.5
        }
    };

    // 融合策略特定加权 (§4.3, 从 R0 GemmBottleneck.fusion_benefits 获取)
    let strategy_weight = bottleneck_map
        .get_fusion_benefit(producer, consumer)
        .unwrap_or(1.0);

    // 寄存器压力惩罚 (从 DeviceProfile 获取可用寄存器数)
    let available_regs = bottleneck_map.device.simd_register_count();
    let reg_penalty = estimate_reg_pressure(producer, consumer, available_regs);

    // 最终分
    (bytes_saved as f64 * scale * strategy_weight) - (reg_penalty as f64 * REG_COST_FACTOR)
}
```

**决策**：`score > 0` → 融合有利；`score <= 0` → Standalone。
瓶颈类型决定 scale 的取值范围，策略权重决定融合模式的优先级。

### §4.2 屋顶线缩放 — 由 DeviceProfile 参数化

```
ridge_point = device.peak_flops / device.peak_bandwidth

每个硬件的 ridge_point 不同:
├── AVX2 (4.5 GHz, 8 核, ~50 GFLOPS, ~40 GB/s):  ridge ≈ 1.25
├── AVX-512 (3.5 GHz, 8 核, ~90 GFLOPS, ~50 GB/s): ridge ≈ 1.8
├── GPU SM80 (A100, 312 TFLOPS FP16, 2 TB/s):      ridge ≈ 156
└── GPU SM90 (H100, 990 TFLOPS FP16, 3.35 TB/s):    ridge ≈ 295

同一模型在不同硬件上的瓶颈可能不同:
  SmolLM GEMV (AI ≈ 2):
    AVX2 (ridge=1.25): AI > ridge → ComputeBound (scale < 1.0)
    GPU SM80 (ridge=156): AI < ridge → MemoryBound (scale = 1.0)
  → 同一个 GEMV 在 CPU 上不急于融合, 在 GPU 上融合收益最大
```

### §4.3 GEMM 特殊评分规则 — 模型+硬件自适应

```
GEMM 融合策略加权 (由 §3.9 性能模型计算, 非硬编码):

EpilogueInjection:
├── 基础: bytes_saved × 3.0 (寄存器级消除)
├── MemoryBound: × 1.5 (消除带宽瓶颈 → 收益放大)
├── ComputeBound: × 1.0 (收益缩减但无损失)
└── 自动检测: M=1 (decode) → MemoryBound; M≥64 (prefill) → 可能 ComputeBound

NormIntoGemm:
├── 基础: bytes_saved × 2.0 (L1 级消除)
├── 计算: norm_output_bytes vs L1 × 75% (阈值来自 DeviceProfile)
│   > 75% L1 → TileLevelFusion (权重 1.5)
│   ≤ 75% L1 → ComputeRoot (权重 1.0)
└── 与 GEMM 角色相关:
    QKV_proj: norm 输出小 → ComputeRoot
    Gate/Up:  norm 输出同 hidden_dim → 可能 TileLevelFusion

QkvSharedInput:
├── 基础: pack_bytes × 3.0 (3 次 pack → 1 次)
├── 仅在 QKV GEMM 角色时激活 (GemmRole::QkvProjection)
├── M=1 decode: pack 无意义 → 虚拟权重布局 (§0.2.7) 替代
└── M≥16 prefill: 物理pack 有价值 → 权重 3.0

LoopFusion:
├── 基础: bytes_saved × 1.0
├── 约束: 累计中间 tensor ≤ L1 × 75% (从 DeviceProfile 获取 L1 大小)
└── 仅 ElemWise → ElemWise 链 (不涉及 GEMM)
```

### §4.4 融合决策示例 — 同模型不同硬件

```
SmolLM-135M 解码层 (hidden=576, intermediate=1536, heads=9, vocab=49152):

GEMM QKV (M=1, N=3×576=1728, K=576):
  FLOPS = 2 × 1 × 1728 × 576 = 1.99M
  bytes = (576 + 995328 + 1728) × 4 = 3.99MB
  AI = 1.99M / 3.99M = 0.50

  AVX2 (ridge=1.25): AI < ridge → MemoryBound
    → EpilogueInjection 权重 3.0 × 1.5 = 4.5
    → QkvSharedInput: M=1, pack 无意义 → 0 (虚拟化替代)
    → NormIntoGemm: norm 输出 2.25KB << L1 → ComputeRoot 权重 1.0

  GPU SM80 (ridge=156): AI << ridge → 极度 MemoryBound
    → EpilogueInjection 权重 3.0 × 1.5 = 4.5
    → QkvSharedInput: M=1, 仍然无意义 → 0
    → NormIntoGemm: shared memory 足够 → TileLevelFusion 权重 1.5

GEMM lm_head (M=1, N=49152, K=576):
  FLOPS = 2 × 1 × 49152 × 576 = 56.6M
  bytes = (576 + 28311552 + 49152) × 4 = 113.5MB
  AI = 56.6M / 113.5M = 0.50

  任何硬件: 极度 MemoryBound (vocab 大, AI 极低)
    → Argmax EpilogueInjection: 消除 113.5MB logits 写回 → 收益极大
    → 这是最重要的融合点: logits 不写回 = 节省 113.5MB 带宽
```

## §5 半 VM 资源规划契约

### §5.1 VmProgram 结构

```
VmProgram = 全局资源规划记录
├── instrs: Vec<VmInstr>                    // 状态转移序列
├── vreg_map: HashMap<VRegId, VRegInfo>     // VReg 元信息
├── tensor_map: HashMap<TensorId, TensorInfo> // Tensor 布局信息
├── loop_scopes: Vec<LoopScope>             // 循环作用域 (嵌套层级)
├── virtual_tensors: HashMap<TensorId, VirtualTensor> // Phase D: 虚拟 tensor
└── abi: Option<MegaKernelAbi>              // ABI 约束
```

### §5.2 VmInstr 分类

| 类别 | 指令 | 资源影响 |
|------|------|----------|
| 数据加载 | LoadPtr, LoadConst | 消耗 GPR + 产生 VReg |
| 计算 | Gemm, Elementwise, Norm, Attention | 消耗/产生多个 VReg |
| 控制 | LoopBegin, LoopEnd, CondBranch | 开辟/关闭作用域 |
| 存储 | StorePtr, StoreToken | 消耗 VReg |
| 采样 | Argmax | 消耗 VReg (可融入 GEMM 累加器) |
| 退出 | CheckStopCondition | 消耗 GPR + 条件跳转 |
| 虚拟 (Phase D) | VirtualTensorDeclare, MaterializeVirtual | 仅追踪, 无物理消耗 |

### §5.3 编译时全局规划流

```
VmProgram 构建完成后:

RegAllocator (Phase 3, Stage 2):
├── 输入: 完整 VmProgram (所有融合组 + 层循环 + 生成循环)
├── 分析: 全局 VReg 生命周期 + interference graph
├── 分配: 图着色 → VReg → 物理寄存器
└── 输出: RegAllocResult (全局映射 + spill 决策)

StackFrame (Phase 3, Stage 3):
├── 输入: RegAllocResult + GlobalBufferLayout
├── 计算: callee-save + spill + scratch 对齐布局
└── 输出: StackFrame (每个 slot 的 rbp 偏移量)

IsaLower (Phase 3, Stage 4):
├── 输入: VmProgram + RegAllocResult + StackFrame
├── 翻译: 每条 VmInstr → 物理指令
├── 虚拟 tensor: VirtualTensorDeclare → 不生成指令 (仅 offset 计算)
├── 约束: 半 VM 的 VReg 映射在此步物化
└── 输出: 机器码字节流
```

## §6 数据流契约

### §6.1 Argmax 数据流 (EpilogueInjection)

```
lm_head GEMM 累加器 → argmax 直接在寄存器中完成
├── 不产生 logits 物理写回
├── Argmax 消费 GEMM 累加器行向量
└── 输出: token_id (Scalar VReg)

Round 2 数据流消除: logits tensor 标记为 VirtualTensor
Round 3 内存规划: logits 不分配 scratch slot
Round 4 寄存器规划: GEMM 累加器 → Argmax 保持同一组物理寄存器
```

### §6.2 StoreToken 数据流

```
Argmax 输出: argmax_token (Scalar VReg)
    ↓ materialize(token_id_tid, &abi)
StoreToken 输入: token_id_ptr (VReg)
    ↓ 需要 gen_loop_counter (AbiPtrs.gen_loop_counter)
    ↓ 需要 output_tokens_ptr (ABI 参数 #8: output_tokens_ptr)
```

### §6.3 CheckStopCondition 数据流

```
Argmax 输出: argmax_token (Scalar VReg)
    ↓ materialize(token_id_tid, &abi)
CheckStopCondition 输入: token_id_ptr (VReg)
    ↓ 需要 gen_loop_counter (AbiPtrs.gen_loop_counter)
    ↓ 需要 eos_token_id (ABI 参数 #13: eos_token_id)
    ↓ 需要 max_new_tokens (ABI 参数 #12: max_new_tokens)
```

### §6.4 生成循环 VReg 传递

```rust
pub struct AbiPtrs {
    pub input_ptr: VRegId,
    pub weight_ptr: Option<VRegId>,
    pub output_ptr: VRegId,
    pub scratch_ptr: Option<VRegId>,
    pub gen_loop_counter: Option<VRegId>,
    pub gen_byte_offset: Option<VRegId>,
}
```

### §6.5 SessionKvRestore 数据流

```
ABI 输入: session_position (ABI 参数 #17)
    ↓ Load to VReg (各后端根据 calling convention 读取)
SessionKvRestore 行为:
    ├── CMP session_position, 0
    │   ├── == 0 → JMP .skip_restore (NOP, 全新生成)
    │   └── > 0 → 执行恢复:
    │       ├── 调整 input_ids_ptr += session_position * 4
    │       ├── 调整 prompt_len -= session_position
    │       └── 设置 KV cache write_offset = session_position
    └── 输出: 调整后的 input_ptr 和 prompt_len (原地修改 VReg)
```

### §6.6 MmHiddenInject 数据流

```
ABI 输入: fused_hidden_ptr (ABI 参数 #18), num_mm_tokens (ABI 参数 #19)
    ↓ Load to VRegs (各后端根据 calling convention 读取)
MmHiddenInject 行为:
    ├── CMP num_mm_tokens, 0
    │   ├── == 0 → JMP .skip_inject (NOP, 纯文本)
    │   └── > 0 → 执行注入:
    │       ├── LoopBegin(num_mm_tokens)
    │       │   ├── Load embedding[i*hidden_dim .. (i+1)*hidden_dim]
    │       │   ├── Load fused_hidden[i*hidden_dim .. (i+1)*hidden_dim]
    │       │   ├── VADD (向量化 ADD, simd_width 并行)
    │       │   └── Store result back to embedding buffer
    │       └── LoopEnd
    └── 输出: 修改后的 embedding buffer (原地 ADD)
```

## §7 compile() 职责边界

**仅**负责 MegaKernelFn ABI 特定的初始化：

| 职责 | 理由 |
|------|------|
| 加载 ABI 参数 (input_ids_ptr, weight_ptr, scratchpad) | ABI 特定 |
| 计算 prompt_len_bytes, input_base, output_ptr | ABI 特定 |
| Store seq_len=1 到运行时绑定位置 (各后端 ABI 决定) | ABI 特定 |
| 生成循环 LoopBegin/LoopEnd | ABI 特定 |
| 调用 emit_fusion_groups() | 共享管线 |

**禁止**：
- ❌ 手动 emit 任何计算型 VmInstr (Argmax/StoreToken/CheckStopCondition)
- ❌ 硬编码采样/生成循环控制流
- ❌ 绕过融合管线直接发射算子

## §8 关键文件

**业务配置层 (§1.5)**:

| 文件 | 变更 | 交叉引用 |
|------|------|---------|
| `graph_builders.rs` | **待删除** — 被 OnnxGraphConverter 替代 (§2.4, §2.6) | — |
| **`onnx_to_compiler.rs`** | **OnnxGraph → CompilerGraph 转换器 (§2.4)** | gllm `arch/auto_graph.rs` |
| `graph.rs` | 新增 OpKind: `WriteLogits` / `EarlyExit` / `GuardrailCheck` / `SgInject` / `SgDetect` / `CotStepCheck` | — |

**编译管线 (§3-§5)**:

| 文件 | 变更 |
|------|------|
| `semantic_dag.rs` | `fallback_op_class()` 新 OpKind 分类 (WriteLogits→Opaque, EarlyExit→Opaque, 等) |
| `fusion/pass.rs` | 重写: PDT 拓扑融合 + 融合收益评分 + Argmax EpilogueInjection |
| `fusion/types.rs` | 扩展: VirtualTensor, VirtualTensorMap, IndexMap |
| `fusion/cost_model.rs` | 增强: 全局收益评分函数 + GEMM 特殊加权 |
| `fusion/dataflow.rs` | 新增: DataFlowOptimizer (VTOG + Global Greedy) |
| `codegen/vm/plan_lower.rs` | `emit_standalone_op()` 新 OpKind lowering + Argmax EpilogueInjection 路径 |
| `codegen/vm/vm_state.rs` | 虚拟 tensor 追踪状态 |
| `codegen/vm/instr.rs` | Phase D: VirtualTensorDeclare / MaterializeVirtual / ActivationSwap |
| `mod.rs` | `compile()` 传递 LayerLoopConfig + 多轮优化 |
| `buffer_alloc.rs` | Phase D: 全局内存布局规划 (虚拟 tensor 跳过 + activation 固定 slot) |
| **`accel_registry.rs`** | **新增: 加速指令布局注册表 (SSOT, 含全部硬件加速指令声明)** [§8.1] |
| **`layout_negotiator.rs`** | **新增: 动态布局协商器 (R1.5, 约束求解最大公约布局)** [§3.10] |
| **`pain_point.rs`** | **新增: PainPointAnalyzer (R0, 编译时瓶颈推导)** [§3.9] |

### §8.1 `accel_registry.rs` — 全硬件加速指令布局声明 (SSOT)

> **定位**: 这个文件是所有硬件加速指令的**声明式注册表**。不是代码实现，是数据声明。
> 新增硬件/指令 = 在此文件添加声明条目，布局协商器自动发现并使用。

**文件结构**：

```rust
// src/compiler/accel_registry.rs
//
// 全硬件加速指令布局注册表 (SSOT)
// 每个条目声明一种加速指令的: 硬件要求 + 算子模式 + 布局约束 + 收益函数
// 布局协商器 (layout_negotiator.rs) 消费此注册表进行动态约束求解
//
// 维护规则:
// 1. 新增硬件/ISA = 新增 register() 调用, 不修改协商器代码
// 2. 每个条目必须含完整注释: 指令手册链接 + 布局原理 + 性能数据
// 3. LayoutConstraint 必须精确 (不精确 = 协商结果错误 = 运行时数据错乱)

// ═══════════════════════════════════════════════════
// §A Intel x86 — AVX-512
// ═══════════════════════════════════════════════════

// AVX-512 F32 GEMM: 14×32 microkernel (28 ZMM registers)
// Ref: Intel® 64 and IA-32 Architectures Optimization Reference Manual §5.7
// Layout: B 需要 PanelPacked(14, 32, kc) 以最大化 L1 TLB 命中
// Benefit: decode GEMV (M=1) → bandwidth bound → pack 无意义 → 虚拟化
//          prefill GEMM (M≥64) → compute bound → pack 有价值 → 59% peak
registry.register(AccelerationDecl {
    id: "x86_avx512_gemm_f32",
    hw_req: AVX512,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::RowMajor { align: 64 }]),
        (TensorRole::B, vec![Layout::PanelPacked { mr: 14, nr: 32, kc: Dyn }]),
    ],
    output_layout: Layout::RowMajor { align: 64 },
    benefit_fn: gemm_f32_avx512_benefit,
});

// AVX-512 VNNI: _mm512_dpbusd_epi32 — 4× throughput vs F32 GEMM for INT8
// Ref: Intel VNNI whitepaper — u8×i8 dot product in 32-bit accumulator
// Layout: A/B 都需要 VnniPacked4 (4个 u8/i8 packed 在 32-bit lane)
// Benefit: 量化模型 decode → bandwidth 降低 4× → memory bound 益处更大
registry.register(AccelerationDecl {
    id: "x86_avx512_vnni",
    hw_req: AVX512_VNNI,
    applicable: [OpPattern::QuantizedGemm(Int8)],
    input_layouts: [
        (TensorRole::A, vec![Layout::VnniPacked4]),
        (TensorRole::B, vec![Layout::VnniPacked4]),
    ],
    output_layout: Layout::RowMajor { align: 64 },
    benefit_fn: vnni_int8_benefit,
});

// AVX-512 BF16: _mm512_dpbf16_ps — native BF16 dot product
// Ref: Intel® BF16 ISA — 2× FLOPS vs F32 (16-bit vs 32-bit)
// Layout: A/B 需要 BF16 format (16-bit bfloat)
// Benefit: 大模型 decode → bandwidth 降低 2×
registry.register(AccelerationDecl {
    id: "x86_avx512_bf16",
    hw_req: AVX512_BF16,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::RowMajor { align: 64 }]),
        (TensorRole::B, vec![Layout::RowMajor { align: 64 }]),
    ],
    output_layout: Layout::RowMajor { align: 64 },
    benefit_fn: bf16_gemm_benefit,
});

// ═══════════════════════════════════════════════════
// §B Intel x86 — AMX (Advanced Matrix Extensions)
// ═══════════════════════════════════════════════════

// AMX tile: 16×32×32 BF16 tile multiply
// Ref: Intel® AMX Programming Guide — TMM0-7 tile registers
// Layout: A 必须 AmxTileBF16(16, 32), B 必须 K-pair interleaved
// Benefit: compute-bound GEMM → 接近理论峰值 (~90%)
registry.register(AccelerationDecl {
    id: "x86_amx_tile_bf16",
    hw_req: AMX,
    applicable: [OpPattern::Gemm where K >= 32],
    input_layouts: [
        (TensorRole::A, vec![Layout::AmxTileBF16 { rows: 16, cols: 32 }]),
        (TensorRole::B, vec![Layout::KPairInterleaved]),
    ],
    output_layout: Layout::RowMajor { align: 64 },
    benefit_fn: amx_tile_benefit,
});

// ═══════════════════════════════════════════════════
// §C AMD x86 — Zen4 AVX-512
// ═══════════════════════════════════════════════════

// Zen4 AVX-512: double-pumped 256-bit → 6×16 microkernel (避免降频)
// Ref: AMD64 Architecture Programmer's Manual Vol 2 §3.4
// Layout: 与 Intel AVX-512 兼容但 MR/NR 不同 → PanelPacked(6, 16, kc)
// Note: Zen4 double-pump → FMA 吞吐与 AVX2 相同, 优势在 VNNI/BF16
registry.register(AccelerationDecl {
    id: "amd_zen4_avx512_gemm_f32",
    hw_req: AMD_Zen4_AVX512,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::RowMajor { align: 64 }]),
        (TensorRole::B, vec![Layout::PanelPacked { mr: 6, nr: 16, kc: Dyn }]),
    ],
    output_layout: Layout::RowMajor { align: 64 },
    benefit_fn: zen4_gemm_benefit,
});

// ═══════════════════════════════════════════════════
// §D ARM — NEON / SVE / SVE2
// ═══════════════════════════════════════════════════

// NEON: 128-bit SIMD — 6×8 F32 microkernel
// Ref: ARM Cortex-X4 Optimization Guide §4.3
// Layout: 与 AVX2 类似但 MR=6, NR=8
registry.register(AccelerationDecl {
    id: "arm_neon_gemm_f32",
    hw_req: NEON,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::RowMajor { align: 16 }]),
        (TensorRole::B, vec![Layout::PanelPacked { mr: 6, nr: 8, kc: Dyn }]),
    ],
    output_layout: Layout::RowMajor { align: 16 },
    benefit_fn: neon_gemm_benefit,
});

// SVE/SVE2: 可变长度 SIMD (128-2048 bit)
// Ref: ARM SVE Architecture Specification
// Layout: 动态 SIMD 宽度 → MR/NR 运行时参数化
registry.register(AccelerationDecl {
    id: "arm_sve2_gemm",
    hw_req: SVE2,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::RowMajor { align: 16 }]),
        (TensorRole::B, vec![Layout::PanelPacked { mr: Dyn, nr: Dyn, kc: Dyn }]),
    ],
    output_layout: Layout::RowMajor { align: 16 },
    benefit_fn: sve2_gemm_benefit,
});

// ═══════════════════════════════════════════════════
// §E NVIDIA GPU — SM70/SM80/SM90/SM100+
// ═══════════════════════════════════════════════════

// SM80 WMMA: 16×8×16 BF16/TF32 tensor core
// Ref: CUDA Programming Guide §16.3 — wmma::fragment layout
// Layout: shared memory 需要 128-byte aligned + bank-conflict-free padding
registry.register(AccelerationDecl {
    id: "nvidia_sm80_wmma",
    hw_req: SM80,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::SharedMemTile { rows: 16, cols: Dyn, padding: 16 }]),
        (TensorRole::B, vec![Layout::SharedMemTile { rows: Dyn, cols: 16, padding: 16 }]),
    ],
    output_layout: Layout::RegisterTile { rows: 16, cols: 8 },
    benefit_fn: sm80_wmma_benefit,
});

// SM90 wgmma: 64×N×K warpgroup MMA + TMA 2D prefetch
// Ref: Hopper Tuning Guide §3 — TMA descriptor + wgmma.async
// Layout: TMA 2D 要求 128-byte aligned swizzled layout
// 时序: TMA async prefetch → wgmma compute → 完全重叠
registry.register(AccelerationDecl {
    id: "nvidia_sm90_wgmma",
    hw_req: SM90,
    applicable: [OpPattern::Gemm],
    input_layouts: [
        (TensorRole::A, vec![Layout::TmaAligned2D { tile_m: 64, tile_n: Dyn }]),
        (TensorRole::B, vec![Layout::TmaAligned2D { tile_m: Dyn, tile_n: 16 }]),
    ],
    output_layout: Layout::RegisterTile { rows: 64, cols: Dyn },
    benefit_fn: sm90_wgmma_benefit,
});

// SM100+ tcgen05.mma: block-scaled FP4/FP6 tensor core
// Ref: Blackwell Architecture Guide §4 — TMEM + tcgen05
// Layout: TMEM (Tensor Memory) + block scaling factors
registry.register(AccelerationDecl {
    id: "nvidia_sm100_tcgen05",
    hw_req: SM100Plus,
    applicable: [OpPattern::QuantizedGemm(FP4)],
    input_layouts: [
        (TensorRole::A, vec![Layout::TmemTile { rows: 128, cols: Dyn }]),
        (TensorRole::B, vec![Layout::TmaAligned2D { tile_m: Dyn, tile_n: 32 }]),
    ],
    output_layout: Layout::RegisterTile { rows: 128, cols: Dyn },
    benefit_fn: sm100_tcgen_benefit,
});
```

**维护约定**：
- 每个条目必须包含: 指令参考文档链接、布局原理注释、性能预期
- 新增硬件 = 新增 `register()` 调用 + 对应 `HardwareRequirement` 枚举变体
- 布局约束修改 = 只改此文件，协商器和 codegen 自动适应
- CLAUDE.md 记录此文件位置，后续完善加速指令只需编辑此文件

### §8.2 推理时序布局变换 — 多步骤零成本级联

> **核心洞察**: 不是"找全局最大公约布局"。数据在流水线中自然流动,
> **每个自然搬运点都是免费变换窗口**。协商器利用时序理解,
> 让布局变换恰好发生在本来就要搬数据的时刻, 零额外成本。

```
推理时序中的布局变换策略:

情况 1: 无变换 (最理想)
  GEMM output [RowMajor] → SiLU input accepts [RowMajor] → 直接消费
  → 布局协商器匹配: GEMM 输出 = SiLU 可接受输入之一

情况 2: 零成本虚拟变换 (VTC 思想)
  QKV GEMM output [RowMajor, shape=(seq, 3*hidden)]
    → RoPE 需要 [HeadSplit, shape=(seq, heads, head_dim)]
  → 不是物理变换! 只是 VirtualTensor IndexMap::Reshape
  → 3*hidden = num_heads * head_dim, 纯索引重计算
  → 内存布局完全相同, 只是 stride 不同 → 零拷贝

情况 3: 利用计算时序隐藏变换
  层 N 最后一步: GEMM output 在寄存器中 [RegisterTile]
  层间: activation 写回 scratch + 下一步 Norm 读
  → GEMM epilogue 阶段: 从 RegisterTile 转换为 RowMajor 写回
  → 这个"转换"实际上就是 GEMM 正常的 store 操作 (不是额外变换)

情况 4: 不可避免的物理变换 (最差)
  GPU: Global memory [RowMajor] → Shared memory [SharedMemTile]
  → cp.async.bulk 搬运 + swizzle → 利用 TMA 异步隐藏延迟
  → 变换与计算重叠: 层 N 计算 + 层 N+1 数据搬运变换并行

时序规划器 (R5) 的布局感知调度:

  R5 现在考虑 LayoutAssignment 中的变换代价:
  1. 零成本变换 (VirtualTensor): 不占时序, 无需规划
  2. 计算内嵌变换 (Epilogue store): 不占额外时序
  3. 异步可重叠变换: 安排在计算时隙中并行 (PING/PONG)
  4. 不可避免的同步变换: 计入关键路径, 最小化次数

  调度策略:
  ├── 尽可能让布局变换成为"零成本虚拟变换" (情况 2)
  ├── 其次让变换内嵌在计算操作中 (情况 3)
  ├── 最后才使用异步重叠 (情况 4)
  └── 禁止: 同步阻塞的显式 memcpy 作为布局变换
```

### §9.1 当前状态 (v1: 模式匹配 7 规则)

```
7 条 FusionRule (pattern matching):
├── detect_qkv_norm_rope (Gemma 4)
├── detect_qkv_shared_input
├── detect_ffn_block
├── detect_norm_into_gemm
├── collect_epilogue (Gemm + ElemWise)
├── collect_elementwise_chain
└── detect_tile_vs_compute_root
```

### §9.2 目标状态 (v2: PDT 拓扑融合 + 多轮全局优化)

```
PDT 拓扑融合 (topology-driven):
├── 构建 PDT → 拓扑序遍历
├── OpClass 层级决策 (Gemm > Reduction > ElemWise > Injective > Opaque)
├── 融合收益评分 (bytes_saved × roofline_scale - reg_penalty)
└── 自动发现可融合组合 (不限于预定义模式)

7 轮全局优化:
├── R0: 瓶颈推导 (PainPointAnalyzer) → OpBottleneckMap [§3.9]
├── R1: PDT 拓扑融合 → FusionPlan
├── R1.5: 布局协商 (LayoutNegotiator) → LayoutAssignment [§3.10, 新增]
├── R2: 数据流消除 (VTC VTOG + Greedy, 布局感知) → VirtualTensorMap
├── R3: 全局内存布局 → GlobalBufferLayout
├── R4: 全局寄存器规划 → RegAllocResult
└── R5: 时序规划 + 布局变换级联 → StackFrame
```

### §9.3 演进阶段

**禁止一步跳到 v2**。必须逐 phase 演进，每个 phase 可独立验证：

| Phase | 虚拟化维度 (§0.2.x) | 内容 | 验证方法 |
|-------|---------------------|------|----------|
| A | §1.5 业务配置 + §0.2.6 虚拟计算 | MegaKernelBusinessConfig → 条件图构建 + Argmax EpilogueInjection + WriteLogits/EarlyExit/业务 ops | `cargo test --lib` fusion 测试 + Head Routing E2E |
| B | §0.2.6 虚拟计算 + §3.9 性能建模 | score_fusion 由 PainPointAnalyzer 驱动 + 编译时瓶颈推导 | E2E SmolLM2-135M |
| C | §0.2.6 虚拟计算 + §0.2.11 虚拟布局 | PDT 拓扑融合 + 加速指令注册表 + 动态布局协商 | 全模型 E2E + 性能无退化 |
| D | §0.2.1 数据 + §0.2.3 内存 + §0.2.7 权重 + §0.2.8 激活 + §0.2.9 执行 + §8.2 时序 | VTC 数据流 + 虚拟权重 + 虚拟激活 + 虚拟执行 + 时序布局变换 | 性能基准验证收益 |

**Phase B 深化路线**（编译时瓶颈推导）：
```
Phase B.1: PainPointAnalyzer (§3.9 R0)
  ├── 编译时从 ModelProfile + DeviceProfile 推导每个 GEMM 瓶颈
  ├── 零运行时依赖: 纯静态 (M, N, K) × (peak_flops, peak_bandwidth) 计算
  └── 验证: 不同模型在不同硬件上的瓶颈分析正确

Phase B.2: score_fusion 瓶颈感知 (§4.1)
  ├── 替换通用 roofline 缩放为 PainPointAnalyzer 结果
  ├── MemoryBound → scale=1.0, ComputeBound → scale=ridge/AI
  └── 验证: E2E SmolLM2-135M 推理正确
```

**Phase C 深化路线**（动态布局协商）：
```
Phase C.1: 加速指令注册表 (§8.1 accel_registry.rs)
  ├── 创建声明式注册表: 所有 Intel/AMD/ARM/NV 加速指令的布局约束
  ├── 每个条目含: 硬件要求 + 算子模式 + 布局约束 + 收益函数
  └── 验证: 注册表条目覆盖当前支持的所有硬件

Phase C.2: LayoutNegotiator (§3.10 R1.5)
  ├── 动态约束求解: 收集布局约束 → 求最大公约布局
  ├── 布局兼容性计算: 交集/冲突/收益权衡
  └── 验证: 协商结果满足最多加速指令, 变换代价最小

Phase C.3: 推理时序布局变换 (§8.2)
  ├── 零成本虚拟变换 (VirtualTensor IndexMap)
  ├── 计算内嵌变换 (Epilogue store)
  ├── 异步可重叠变换 (PING/PONG)
  └── 验证: 全模型 E2E 无退化
```

**Phase D 的全虚拟化深化路线**：
```
Phase D.1: VTC 数据流消除 (§0.2.1 虚拟数据)
  ├── DataFlowOptimizer::build_vtog + eliminate_greedy
  └── 验证: 中间 tensor 物理分配减少

Phase D.2: 虚拟激活流 (§0.2.8 虚拟激活)
  ├── 跨层 activation 身份映射 (1 buffer 替代 ⌈N/2⌉)
  └── 验证: scratch buffer 大小减少 ~50%

Phase D.3: 虚拟权重布局 (§0.2.7 虚拟权重)
  ├── PackMap 替代 decode 路径物理 pack_a
  └── 验证: decode 延迟减少 (消除 pack 开销)

Phase D.4: 虚拟执行模式 (§0.2.9 + §0.2.10)
  ├── GPU shared memory tiling / TMA / warp reduce 作为虚拟执行意图
  └── 验证: GPU 路径使用协商布局生成正确代码

Phase D.5: 全局内存/寄存器联合优化
  ├── R3 虚拟内存 + R4 虚拟寄存器 联合求解
  └── 验证: spill 减少 + scratch 布局 cache 友好
```

### §9.4 Phase A 实施细节

Phase A 不修改现有融合架构，仅在 `fuse_with_dag_prebuilt` 中增加：
1. `collect_epilogue` 后检查下游 Reduction (如 Argmax)
2. 单消费者 + 单输入 Reduction → 追加到 epilogue
3. 新增 `try_collect_reduction_epilogue()` 辅助函数
4. `emit_standalone_op()` 新增 Argmax/StoreToken/CheckStopCondition lowering

### §9.5 Phase B: Output Mode Selector + JMP Table

> 依赖: Phase A 完成 (业务 ops OpKind + 条件图构建 + NOP passthrough)

**目标**: 多 output mode 编译为单一 mega-kernel，运行时通过 `output_mode_selector` ABI 参数切换。

#### B.1 ABI 扩展 (mega_kernel_abi.rs)

```
新增 ABI 参数: output_mode_selector (u32)
位置: 逻辑参数 #14 (位于 eos_token_id 之后, hook_ctx_ptr 之前)

具体物理偏移由各后端 calling convention 决定:
  - x86_64 SystemV: 栈参数, 具体偏移取决于前 6 个寄存器参数后的栈布局
  - AArch64 AAPCS: 栈参数 (超出 x0-x7 的部分)
  - GPU: 参数 buffer 中的顺序位置

MEGA_KERNEL_PARAMS 新增 "output_mode_selector" 名称
各后端 AbiPtrs 结构体中添加 output_mode_selector 字段
```

#### B.2 VmInstr 扩展 (instr.rs)

```
新增:
  OutputModeDispatch { selector: VRegId, paths: Vec<usize> }
    — 后端无关的条件分发指令
    — x86_64: CMP+JE 链
    — AArch64: CBZ/CBNZ + B.EQ
    — GPU: 分支/谓词执行

  BreakLoop { return_value: u32 }
    — 后端无关的循环退出指令
    — x86_64: 设置 rax + JMP 到 epilogue
    — AArch64: 设置 x0 + B 到 epilogue
    — GPU: 设置返回值 + 退出 kernel
```

#### B.3 compile() OutputModeDispatch (SPEC/39 §1.3.3)

在融合组序列发射之后 (for group in plan.groups { emit_group(group) }):
1. 如果 `output_modes.len() > 1`: 发射 OutputModeDispatch + 各路径指令
2. 如果 `output_modes.len() == 1`: 走单一路径 (现有逻辑不变)
3. generate 路径: Argmax + StoreToken + CheckStopCondition 融合组
4. classify 路径: WriteLogits + BreakLoop(0)
5. encode 路径: Pool + BreakLoop(0)

#### B.4 各后端 ISA Lowering

**VmInstr → ISA 映射 (后端无关)**:

| VmInstr | x86_64 | AArch64 | GPU |
|---------|--------|---------|-----|
| `OutputModeDispatch` | CMP+JE 链 | CBZ/CBNZ + B.EQ | 分支/谓词 |
| `BreakLoop` | MOV rax + JMP epilogue | MOV x0 + B epilogue | 设置返回值 + 退出 |
| `SessionKvRestore` | CMP + 条件指针运算 | CBZ + 条件指针运算 | 条件偏移计算 |
| `MmHiddenInject` | 向量 ADD 循环 (AVX2/AVX-512) | 向量 ADD 循环 (NEON/SVE) | 并行 kernel |

**各后端 codegen 负责**: 将 VmInstr 的语义映射为本 ISA 的指令序列。SPEC 只定义语义，不写死具体指令。

#### B.5 gllm 侧 MegaKernelExecutor 更新

```
MegaKernelFn 调用签名 (23 参数, 含 Session + Multimodal + Callback + Paging + Batch)
output_mode_selector (arg 14) 传入:
  generate → 0
  classify_binary → 1
  classify_multiway → 2
  encode_to_layer → 3
  score_tokens → 4
  encode_at_layer → 5

返回值语义:
  > 0 → generate: token 数量
  == 0 → classify/encode/score: output buffer 已填充
```

## §10 编译管线三层并行化

> **核心洞察**: mega-kernel 编译完全串行处理所有 fusion groups。
> 异构模型（Gemma-4）有 4 种层类型（ss/fs/sl/fl），每种类型有独立的 fusion group 序列，
> 它们之间编译完全独立。三层并行化将编译时间从 O(总 groups) 降低到 O(最慢层类型)。

### §10.1 基础设施：LoweringContext

**问题**: `emit_fusion_groups` 有 16 个参数，大量参数在调用链中重复传递。
`emit_standalone_op`、`emit_gemm_inline_with_hook` 等函数同样参数膨胀。
新增 `ResourceBudget` 等参数会进一步恶化。

**方案**: `LoweringContext<'a>` 持有编译会话的所有不可变共享状态：

```rust
pub struct LoweringContext<'a> {
    pub width: SimdWidth,
    pub sym_map: &'a SymDimSlotMap,
    pub registry: Option<&'a ScalarOpRegistry>,
    pub hook: Option<&'a dyn IsaHook>,
    pub budget: Option<ResourceBudget>,
    pub rope_req: Option<&'a RopeCacheRequirement>,
    pub ple_req: Option<&'a PleScratchRequirement>,
    pub dwc_req: Option<&'a DwcScratchRequirement>,
}
```

**约束**:
- `LoweringContext` 一旦构造就不可变 — 编译会话期间 `width`/`sym_map`/`hook`/`budget` 不变
- 通过 `&LoweringContext` 传递，不消耗所有权
- 所有 emit 函数签名简化为 `ctx: &LoweringContext` + 必要的局部参数

### §10.2 EmitState — emit_fusion_groups 可变状态抽取

**问题**: `emit_fusion_groups` 内部有 5 个跨迭代可变状态变量：

```
current_abi: &mut AbiPtrs           — ABI 指针集 (层间 output→input 交换)
hetero_phase: HeteroPhase           — 异构层模板阶段追踪
in_layer_loop: bool                 — 是否已进入层循环
hetero_seg_byte_offset: Option<VRegId> — 异构段偏移
hetero_seg_weight_base: Option<VRegId> — 异构段权重基址
```

这些状态变量阻止了 emit_fusion_groups 被拆分为可并行的子函数。

**方案**: 抽取为 `EmitState` 结构体：

```rust
pub struct EmitState {
    /// ABI 指针集 — 层间 output→input 交换
    pub abi: AbiPtrs,
    /// 异构层模板阶段追踪
    pub hetero_phase: HeteroPhase,
    /// 是否已进入层循环
    pub in_layer_loop: bool,
    /// 异构段偏移 VReg
    pub hetero_seg_byte_offset: Option<VRegId>,
    /// 异构段权重基址 VReg
    pub hetero_seg_weight_base: Option<VRegId>,
}
```

**状态机模型**: `emit_fusion_groups` 的执行是一个有限状态机：

```
EmitState {
    hetero_phase: HeteroPhase (5 状态)
    ├── BeforeLayers      — 层循环前融合组（embed 等），非 layer ops
    ├── SlidingSmall(ss)  — 在 ss 层循环内
    ├── FullSmall(fs)     — fs 层模板
    ├── SlidingLarge(sl)  — sl 层循环内
    └── FullLarge(fl)     — fl 层模板

    in_layer_loop: bool
    ├── false → 层循环外 (embed/lm_head)
    └── true  → 层循环内 (op 已包在 LoopBegin/LoopEnd 中)
}
```

**状态转移**:

```
BeforeLayers --(遇到 "layer_sliding_small.*" op)--> SlidingSmall(0)
SlidingSmall(i) --(LoopEnd + 段内迭代完成)--> SlidingSmall(i+1) | FullSmall
FullSmall --(遇到 "layer_sliding_large.*" op)--> SlidingLarge(0)
SlidingLarge(i) --(LoopEnd + 段内迭代完成)--> SlidingLarge(i+1) | FullLarge
FullLarge --(遇到非 layer op)--> BeforeLayers (层循环后融合组，lm_head 等)
```

**并行化关键**: 每种层类型（ss/fs/sl/fl）的编译拥有独立的 `EmitState` 快照。
层模板编译函数 `compile_layer_type_body` 从快照初始化自己的 `EmitState`，
编译完成后返回 `LayerTemplate`（包含 VmProgram + ABI 映射）。

### §10.3 第一层：层模板并行编译（最大收益）

**核心洞察**: 异构模型（Gemma-4）的 4 种层类型各自有一组 fusion groups。
4 种类型之间编译完全独立，可以用 rayon 并行编译。

**数据结构**:

```rust
/// 编译后的层模板 — 独立 VmProgram + ABI 指针映射
pub struct LayerTemplate {
    pub body: VmProgram,
    pub abi_map: LayerAbiMap,
}

pub struct LayerAbiMap {
    pub input_ptr: VRegId,
    pub weight_ptr: VRegId,
    pub output_ptr: VRegId,
    pub scratch_base: VRegId,
}
```

**编译函数**:

```rust
fn compile_layer_type_body(
    ctx: &LoweringContext,
    group_range: Range<usize>,
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    state: &EmitState,
    resolver: &TensorPtrResolver,
) -> Result<LayerTemplate, CompilerError>
```

**当前状态**: 仅支持 `Standalone` / `LoopFusion`。

**扩展计划**: 逐步支持所有 FusionMode：

| FusionMode | 模板编译可行性 | 实现策略 |
|------------|---------------|---------|
| `Standalone` / `LoopFusion` | ✅ 已支持 | 直接 emit |
| `EpilogueInjection` | 可行 | GEMM + epilogue trace 整体编译为模板 |
| `NormIntoGemm` | 可行 | Norm→GEMM 融合整体编译 |
| `QkvSharedInput` | 可行 | 3× GEMM + shared pack_a 整体编译 |
| `TileLevelFusion` | 需评估 | 嵌入 GEMM MC 循环的前驱算子需要 scratch 布局 |
| `ComputeRoot` | 需评估 | 前驱完整计算 → GEMM，scratch 生命周期跨组 |
| `FFNBlock` | 可行 | Gate+Up+Act+Mul 整体编译 |
| `CrossLayerResidual` | 不可行 | 跨层残差依赖外部层输出 |
| `FusedQkvNormRope` | 可行 | 6 算子链整体编译 |

**并行实例化**: 在 `emit_fusion_groups` 中：

```
1. 预分析 FusionPlan，识别每种层类型的 group 边界
   ├── 分析 anchor op label 前缀 ("layer_sliding_small.*", "layer_full_small.*", ...)
   ├── 对每种层类型收集 group_range
   └── 构造每种层类型的 EmitState 快照

2. 使用 rayon 并行编译 N 个模板
   templates: Vec<LayerTemplate> = layer_types
       .par_iter()
       .map(|(group_range, state)| compile_layer_type_body(ctx, group_range, ...))
       .collect()

3. 在主 VmProgram 中按顺序实例化模板
   ├── LoopBegin(outer_segment_loop)
   │   ├── LoopBegin(inner_sliding_loop)
   │   │   └── prog.append_with_mapping(ss_template, &ss_abi_subst)
   │   └── LoopEnd
   │   └── prog.append_with_mapping(fs_template, &fs_abi_subst)
   └── LoopEnd
```

**VmProgram 模板合并**:

```rust
impl VmProgram {
    /// 合并模板：VReg 重映射 + ABI 指针替换。
    /// subst: [(template_vreg, main_prog_vreg)] 替换表
    pub fn append_with_mapping(&mut self, other: VmProgram, subst: &[(VRegId, VRegId)]);
}
```

**替换语义**:
- 模板内 `DeclareVReg { id }` → 跳过声明，用 subst 映射的 VRegId 替代
- 模板内 `VRegId` 引用 → 全部替换为主程序对应的 VRegId
- 模板内 `LoopBegin`/`LoopEnd` → 保留循环结构，VReg 映射后嵌入
- 新分配的 VReg（非 ABI 指针）→ 主程序分配新的 VRegId，保持独立性

### §10.4 第二层：融合组级并行（中等收益）

**核心洞察**: FusionPlan 的 groups 形成生产-消费 DAG。
同一拓扑层级的 groups 之间无数据依赖，可以并行 lower。

**依赖分析器**:

```rust
struct GroupDependencyAnalyzer;

impl GroupDependencyAnalyzer {
    /// 分析 FusionPlan 的 group 间数据依赖。
    /// 返回拓扑层级列表：同一层级的 groups 可并行处理。
    fn analyze(plan: &FusionPlan, graph: &CompilerGraph) -> Vec<Vec<usize>>;
}
```

**依赖分析算法**:

```
1. 对每个 group gi，收集其所有 op 的 input TensorId 集合
2. 对每个其他 group gj，检查是否有 op 的 output TensorId ∈ gi 的 input 集合
   ├── 是 → gj blocks gi (gi 依赖 gj)
   └── 否 → 无依赖
3. 拓扑排序：按依赖关系分层，每层的 groups 无互相依赖
   ├── level 0: 无依赖的 groups (如 embed 融合组)
   ├── level 1: 仅依赖 level 0 的 groups
   └── level N: 依赖 level 0..N-1 的 groups
4. 循环依赖检测：无剩余 group 可分配时，将剩余 group 合入同一层
```

**并行 lower**:

```rust
fn lower_parallel_groups(
    ctx: &LoweringContext,
    groups: &[usize],
    plan: &FusionPlan,
    graph: &CompilerGraph,
    alloc: &BufferAllocation,
    resolver: &TensorPtrResolver,
) -> Vec<(usize, VmProgram)>
```

**约束**:
- 同一拓扑层级内无依赖的 groups 可并行编译到独立 VmProgram
- 层内 groups 有顺序依赖（attn → ffn），不能并行
- `emit_inter_group_casts` 需要 producer group 先完成才能更新 resolver
- 组间 cast 在所有 groups 编译完成后统一处理
- 跨层残差（`CrossLayerResidual`）的 group 不能与其他组并行

### §10.5 第三层：自调优并行（精确优化）

**核心洞察**: 每个 GEMM 算子的 FMA 策略选择可以并行评估多种候选方案。

**成本模型**:

```rust
/// 硬件资源预算 — 从 IsaProfile 推导
pub struct ResourceBudget {
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub avail_regs: usize,
}

/// FMA 候选策略 + 估算成本
pub struct FmaCandidate {
    pub strategy: FmaStrategy,
    pub estimated_cost: f64,
}
```

**成本评估维度**:

```
estimate_strategy_cost(strategy, m, n, k, budget):
  ├── 计算吞吐: FMA 单元利用率 (utilization rate)
  ├── 寄存器压力: 累加器占用 vs 可用寄存器预算 (spill penalty)
  └── Cache 友好性: 分块是否 fit L1/L2
```

**并行评估**:

```rust
pub fn select_fma_best(
    hook: &dyn IsaHook, m: usize, n: usize, k: usize,
    dtype: DType, budget: &ResourceBudget,
) -> FmaStrategy {
    let candidates = select_fma_candidates(hook, m, n, k, dtype);
    candidates.into_par_iter()
        .map(|c| evaluate_cost(c, m, n, k, budget))
        .min_by(|a, b| a.estimated_cmp(b))
        .map(|c| c.strategy)
        .unwrap_or_else(|| hook.select_fma(m, n, k, dtype))
}
```

### §10.6 实施顺序

```
1. 基础设施 (已完成): LoweringContext + 函数签名重构
2. 第三层 (已完成): select_fma_best + ResourceBudget + 成本模型
3. EmitState 抽取: 从 emit_fusion_groups 提取可变状态 → EmitState 结构体
4. compile_layer_type_body 扩展: 支持 EpilogueInjection / TileLevelFusion / ComputeRoot
5. 第一层 (层模板并行): LayerTemplate + append_with_mapping + rayon 并行编译
6. 第二层 (融合组并行): GroupDependencyAnalyzer + 拓扑序并行 lower
```

### §10.7 关键文件

| 文件 | 改动 |
|------|------|
| `codegen/vm/plan_lower.rs` | `EmitState`, `LoweringContext`, `compile_layer_type_body` 扩展, rayon 并行路径 |
| `codegen/vm/isa_hook.rs` | `FmaCandidate`, `ResourceBudget`, `estimate_strategy_cost`, `select_fma_best` (已完成) |
| `codegen/vm/instr.rs` | `LayerTemplate`, `LayerAbiMap`, `append_with_mapping` (已完成) |

### §10.8 验证

```bash
# 编译正确性
cd gllm-kernels && cargo test --lib

# gllm E2E 测试
cd gllm && cargo test --lib -p gllm

# 异构模型并行编译验证
cd gllm && cargo test test_e2e_gemma -- --nocapture --test-threads=1

# 编译时间对比
GLLM_DEBUG_RESOURCE=1 cargo test test_e2e_llama_7b -- --nocapture 2>&1 | grep "compile\|time"
```
