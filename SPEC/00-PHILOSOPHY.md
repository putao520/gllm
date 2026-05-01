# 核心铁律与约束

> **SSOT**: 本文件是 gllm + gllm-kernels 统一项目的所有不可违反的架构铁律的唯一真源。

## 1. 系统定位

gllm 是基于 JIT 编译的硬件适配推理引擎。核心理念：根据 (模型结构 × 硬件能力) 的笛卡尔积，在模型加载时生成最优机器码，推理热路径零编译、零分发、零分支。

**两个姊妹项目**：
- **gllm**: 推理客户端库 — 模型管理、调度编排、公共 API
- **gllm-kernels**: JIT 编译器与算子库 — 四阶段编译管线、算子定义、硬件适配

## 2. 铁律清单

### JIT-FIRST (ARCH-JIT-FIRST)

所有算子必须走完整四阶段 JIT 编译管线，无例外：

```
算法 (Scalar Rust) → Lifting (SymExec) → IR (TraceOp SSA) → ISA Lowering (DeviceProfile) → 机器码
```

- 禁止跳过任何阶段
- 禁止在 JIT 代码中 `call` 预编译的 Rust/C 函数
- 禁止把 JIT 当"调度器"只负责串联预编译函数
- 禁止写死特定 ISA，必须通过 DeviceProfile 参数化

### PEAK-PERF (ARCH-PEAK-PERF)

硬件参数在 DeviceProfile::detect() 时一次性探测、运行时固定。JIT codegen 为当前硬件生成最优路径。

- 禁止硬件降级（如 FlashAttention → Atomic）
- 禁止 "硬件能力不足所以降级" 作为拆分融合算子的理由
- 硬件差异体现在 codegen 层的指令选择，不是 fusion 层的算子拆分

### NO_SCALAR (零容忍)

scalar_ops.rs 中的函数禁止在任何代码路径中被调用（含测试）。

- 仅作为 Phase 0 的算子语义定义，供 SymExec trace 提取使用
- 测试必须通过 JIT 编译图验证正确性

### NO_SILENT_FALLBACK

JIT codegen 遇到无法生成代码的 OpKind 必须返回 Err，禁止静默 NOP。

- 禁止 emit_nop() / emit_nop_placeholder() 作为未实现 op 的 catch-all
- 禁止 match _ => Ok(()) 吞掉未知 OpKind
- 仅 Reshape / Transpose（纯元数据 op）允许 NOP

### NO_HW_DEGRADATION

禁止任何形式的硬件降级/退化。FusionRule 不因硬件能力不足而拆分融合算子。

```
错误: FusionRule 看到硬件不足 → 降级为 Atomic → executor 拆分执行
正确: FusionRule 生成融合图 → JIT codegen 根据 DeviceProfile 生成最优机器码
```

### ARCH-CPU-GPU-UNIFIED

CPU 和 GPU 共享完全一致的 CompilerGraph IR。同一个融合图通过 DeviceProfile 驱动 Phase 3 codegen。

- 禁止以 Cpu/Gpu/后端名称为前缀的 GraphType 变体
- 禁止 CPU 专用图构建器
- 禁止子算子级 GraphType（如 KvProjection、QRope）
- CPU 和 GPU 给定相同输入和权重，输出必须在浮点精度范围内一致

### ZERO-FALLBACK (ARCH-ZERO-FALLBACK)

GPU OOM 必须直接引发全进程 Halt，禁止退回 CPU。

- OOM 时直接返回硬件越界错误
- 无降级、无 fallback、无静默处理

### 禁止外部推理引擎/BLAS

- 禁止引入 candle、tch、ort、tract、burn 等推理框架
- 禁止在 GPU 推理路径中调用 cuBLAS/rocBLAS/cuDNN
- GPU GEMM 全部走 JIT codegen 生成 PTX/HIP/MSL 原生二进制
- CPU ISV（oneDNN/Accelerate）保留

### JIT 缓存协议

- 编译只发生在模型加载时
- 推理热路径（decode step 层循环）中禁止任何编译行为
- 动态维度（seq_len, total_seq）通过 SymDim::Symbolic + ShapeBinding 运行时绑定
- 缓存粒度 = 全层融合图

## 3. 硬性约束

### NO_DYNAMIC_LOADING

gllm 自身的所有模块必须静态编译并链接。禁止 dlopen/libloading。

例外：gllm-kernels GPU 后端通过 dlopen 加载 GPU driver API（libcuda.so / libamdhip64.so / Metal.framework）是内部实现细节。

### TRUTH (Ω1: 真实性原则)

所有架构/量化/精度信息必须从模型文件自身的 metadata 读取。

- 禁止基于 Model ID 推断架构类型
- 禁止基于文件名推测量化类型或精度
- 禁止使用 contains() 模糊匹配进行架构推测
- 禁止使用硬编码默认值代替模型元数据

### 纯 Rust

- 禁止引入 candle、tch 等重量级深度学习框架依赖
- 仅使用 safetensors、half、prost 等底层工具库
- GGUF 解析器完全自研
- 计算核心完全由 gllm-kernels 提供

### 单一后端原则

全程在单一后端执行。禁止中途 GPU→CPU→GPU 往返。

### 环境变量铁律

禁止擅自添加环境变量。新增必须由用户明确要求且场景不可替代。

## 4. DType 架构决定 — 编译时 dtype 感知 + 原生混合精度

**核心原则**：dtype 在模型加载时确定（TensorMeta.dtype），JIT 编译时根据 dtype 生成特化机器码，运行时零分支。

### 4.1 dtype 传播链

```
模型文件 (safetensors/GGUF)
  → loader 读取 DType
    → CompilerGraph.TensorMeta.dtype
      → JIT 编译时: op_input_dtype(op, graph) 从 tensor 元数据推断
        → TraceOp body 通过 TypedSlot 携带推断的 dtype
          → VmInstr 携带 dtype 字段
            → ISA Lowering 根据 dtype 选择原生硬件指令
              → 单态化机器码（每种 dtype 组合一份）
```

### 4.2 编译时单态化

- dtype 在 JIT 编译时已知（不是运行时变量）
- 对每种 dtype 组合生成特化机器码
- ISA Lowering 根据 dtype 分支选指令：BF16→VDPBF16PS, F32→VMULPS, F16→VFMULCPH
- 这是**编译时分支**（类似 LLVM SelectionDAG），不是运行时分支

### 4.3 双路径共存

| 路径 | 触发条件 | 行为 |
|------|---------|------|
| **原生混合精度** | 模型权重以 BF16/F16/FP8 格式保留（不经 dequant） | JIT 直接生成对应 dtype 的原生指令（VDPBF16PS/mfma.bf16） |
| **TurboQuant 量化** | 模型使用 TurboQuant 量化格式（INT4/FP4/FP6） | TurboQuant 解量化管线 + 定点/微浮点硬派算子 |

两种路径互不冲突：原生混合精度路径处理模型原始 dtype，TurboQuant 处理额外量化。

### 4.4 禁止项

- 禁止硬编码 `QuantPrecision::F32` 作为计算 dtype（必须从 TensorMeta 推断）
- 禁止 `computation_elem_bytes()` 硬编码返回 4（必须用推断的 dtype.elem_bytes()）
- 禁止硬件不支持时静默 fallback 到 F32（必须报错）
- 禁止运行时 `match dtype { ... }` 分支（编译时确定，运行时零分支）

### 4.5 运行时行为

- 运行时零 dtype 分支：dtype 在 JIT 编译时已 bake 进机器码
- `<E: Element>` 编译时单态化保留（与 dtype 推断兼容）
- executor 通过 `cn.output_dtype` 读取 dtype（来源：graph.tensor.dtype）

## 5. 禁止模式索引

| ID | 禁止模式 | 正确替代 |
|----|---------|---------|
| LOG-AND-DISCARD | 优化决策只进日志不改变执行 | 通过 Callback 直接改变 FusedGraphExecutor 行为 |
| 渐进式开发 | "保留旧实现，添加新功能" | 销毁重建，从零实现符合 SPEC 的新代码 |
| 静默降级 | 运行时条件判断降级到次优路径 | 编译时变体特化，运行时零分支 |
| Mock/Stub 测试 | 使用假权重/硬编码输出 | 必须使用真实模型文件执行真实推理 |
| 暴力 Padding | 对齐批次时补零浪费 FLOPs | Ragged Compaction + 硬件谓词掩码 |
| 死板 Bucket | 预设 [128,512,1024] 等静态 JIT Bucket | 硬件探测 → 黄金装筒 → 运行时演化 |

## 6. SPEC 维护规则

- SPEC 只表达当前正确设计，版本历史由 Git 管理
- 修改时整段重写受影响章节，禁止在末尾追加
- 同一概念只在一处定义（SSOT）
- SPEC 文件不包含版本号、日期、变更历史
- 不在 SPEC 中放临时文件（审计报告/草稿/TODO）
