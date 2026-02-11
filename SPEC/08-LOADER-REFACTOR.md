# SPEC: 加载器架构重构 - 统一 DAG 表示

## 版本信息
- **文档版本**: 1.0
- **日期**: 2026-02-11
- **状态**: 设计完成，待实施

---

## 1. 背景与动机

### 1.1 当前架构问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **config.json 依赖** | 配置回退到外部 config.json | 违反 Ω1 真实性原则 |
| **精度转换** | f16→f32→E 中间转换 | 性能损失、精度损失 |
| **硬编码 Adapter** | 12 个 adapter 文件硬编码 DAG | 维护成本高、不可扩展 |
| **三格式分裂** | GGUF/SafeTensors/ONNX 走不同路径 | 代码重复、优化分散 |

### 1.2 设计目标

| 原则 | 描述 |
|------|------|
| **Ω1 无配置** | 100% 从模型文件推导配置 |
| **Ω2 无转换** | 读什么精度用什么精度推理 |
| **Ω3 统一 DAG** | 所有格式统一到 OnnxGraph 表示 |
| **Ω4 算子融合** | 动态 DAG 优化，最大化性能 |

---

## 2. 架构设计

### 2.1 统一数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         统一加载流水线                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ONNX 文件 ────────► OnnxLoader ────────────────────► OnnxGraph         │
│                         │                                               │
│                         ├─► 从文件解析 graph                            │
│                         └─► 权重在 initializers                         │
│                                                                         │
│  GGUF 文件组 ──────► GgufLoader ────► 架构模板(YAML)                    │
│                         │                + 权重绑定 ────► OnnxGraph     │
│                         │                                               │
│                         ├─► metadata 推导 config                        │
│                         └─► 权重零拷贝绑定到 initializers               │
│                                                                         │
│  SafeTensors 文件组 ► SafeTensorsLoader ► 架构模板(YAML)                │
│                         │                  + 权重绑定 ──► OnnxGraph     │
│                         │                                               │
│                         ├─► 张量形状推导 config                         │
│                         └─► 权重零拷贝绑定到 initializers               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         OnnxGraph (统一 DAG 表示)                        │
│                                   │                                     │
│                                   ▼                                     │
│                         ┌─────────────────────┐                         │
│                         │   Graph Optimizer   │                         │
│                         │                     │                         │
│                         │  Pass 1: 模式融合   │                         │
│                         │  Pass 2: 硬件融合   │                         │
│                         │  Pass 3: 布局优化   │                         │
│                         │  Pass 4: 死代码消除 │                         │
│                         └─────────────────────┘                         │
│                                   │                                     │
│                                   ▼                                     │
│                         FusedGraph (优化后执行计划)                      │
│                                   │                                     │
│                                   ▼                                     │
│                         FusedGraphExecutor (执行)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
src/
├── loader/
│   ├── mod.rs              # 统一加载入口
│   ├── gguf/               # GGUF 解析（零转换）
│   ├── safetensors.rs      # SafeTensors 解析（零转换）
│   ├── onnx/               # ONNX 解析
│   └── arch_template.rs    # YAML 模板 → OnnxGraph
│
├── arch/                   # 架构 YAML 模板
│   ├── mod.rs              # 模板注册表
│   ├── qwen3.yaml
│   ├── qwen3_moe.yaml
│   ├── llama.yaml
│   ├── mistral.yaml
│   └── ...
│
├── graph/                  # DAG 处理
│   ├── mod.rs
│   ├── types.rs            # OnnxGraph 扩展类型
│   ├── optimizer/          # 优化器
│   │   ├── mod.rs
│   │   ├── pass.rs         # Pass trait
│   │   ├── pattern_fusion.rs
│   │   ├── hardware_fusion.rs
│   │   └── dead_code.rs
│   └── executor.rs         # FusedGraph 执行
│
├── engine/
│   └── executor.rs         # 重写：使用 graph 模块
│
└── backend/                # 保留不变
```

---

## 3. 需求规格

### 3.1 Phase 1: 清理冲突代码 ✅

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-REFACTOR-001 | 删除 adapter 模块 | `src/adapter/` 目录不存在 | ✅ [3cd8da8] |
| REQ-REFACTOR-002 | 删除 config.rs | `src/loader/config.rs` 不存在 | ✅ [3cd8da8] |
| REQ-REFACTOR-003 | 删除精度转换函数 | `convert_to_f32`, `convert_to_element` 不存在 | ✅ [3cd8da8] |
| REQ-REFACTOR-004 | 删除 config.json 回退 | `model_config.rs` 无 config.json 回退逻辑 | ✅ [3cd8da8] |
| REQ-REFACTOR-005 | 移除 executor 的 adapter 依赖 | `executor.rs` 不引用 adapter 模块 | ✅ [3cd8da8] |

### 3.2 Phase 2: 架构模板系统 ✅

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-ARCH-001 | YAML 模板格式定义 | 支持 nodes、repeat、config 占位符 | ✅ [3cd8da8] |
| REQ-ARCH-002 | YAML → OnnxGraph 解析器 | `arch_template::parse()` 返回 OnnxGraph | ✅ [3cd8da8] |
| REQ-ARCH-003 | 权重绑定机制 | `OnnxGraph::bind_weights()` 零拷贝绑定 | ✅ [3cd8da8] |
| REQ-ARCH-004 | Qwen3 架构模板 | `arch/qwen3.yaml` 完整可用 | ✅ [3cd8da8] |
| REQ-ARCH-005 | 配置推导 | 从 GGUF metadata / 张量形状推导 config 占位符 | ✅ [3cd8da8] |

### 3.3 Phase 3: DAG 优化器 ✅

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-OPT-001 | OptimizationPass trait | 支持 name(), run(), enabled() | ✅ [3cd8da8] |
| REQ-OPT-002 | 模式融合 Pass | FlashAttention, SwiGLU, RoPE 融合 | ✅ [3cd8da8] |
| REQ-OPT-003 | 硬件感知融合 | 根据 CUDA SM 版本选择策略 | ✅ [3cd8da8] |
| REQ-OPT-004 | 优化器管道 | `GraphOptimizer::optimize()` 执行所有 Pass | ✅ [3cd8da8] |
| REQ-OPT-005 | FusedGraph 生成 | 优化后输出 FusedGraph | ✅ [3cd8da8] |

### 3.4 Phase 4: 执行器重写 ✅

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-EXEC-001 | 统一加载入口 | `Loader::load() -> OnnxGraph` | ✅ [3cd8da8] |
| REQ-EXEC-002 | FusedGraph 执行 | `FusedGraphExecutor::run()` 执行融合图 | ✅ [3cd8da8] |
| REQ-EXEC-003 | 零转换推理 | 权重原生精度直接参与计算 | ✅ [3cd8da8] |
| REQ-EXEC-004 | 端到端测试 | Qwen3-0.5B 推理正确 | ✅ [3cd8da8] |

### 3.5 Phase 5: 执行器接入后端（待实施）

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-EXEC-005 | 接入 gllm-kernels | FusedOp 调用对应内核执行 | 📋 待 gllm-kernels 就绪 |
| REQ-EXEC-006 | 中间张量内存管理 | 分配/释放中间计算结果 | 📋 依赖 REQ-EXEC-005 |
| REQ-EXEC-007 | 并行执行 | 无依赖节点并行执行 | 📋 依赖 REQ-EXEC-005 |

### 3.6 Phase 6: OnnxGraph 完整性补齐

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-ONNX-001 | 稀疏张量支持 | `SparseTensor` 加载并转换为 FusedGraph | 📋 待实现 |
| REQ-ONNX-002 | 量化标注支持 | `QuantizationAnnotation` 传递到优化器 | 📋 待实现 |
| REQ-ONNX-003 | 节点属性传递 | `OnnxNode.attributes` 完整传递到 `FusedNode` | 📋 待实现 |

### 3.7 Phase 7: 更多融合模式

| REQ ID | 需求 | 验收标准 | 状态 |
|--------|------|----------|------|
| REQ-OPT-006 | GQA 融合 | Grouped Query Attention 模式识别和融合 | 📋 待实现 |
| REQ-OPT-007 | MoE routing 融合 | TopK + Softmax + Dispatch 模式融合 | 📋 待实现 |
| REQ-OPT-008 | 常量折叠 Pass | 编译期常量表达式求值 | 📋 待实现 |

---

## 4. 架构模板 YAML 规格

### 4.1 模板格式

```yaml
# 架构名称和版本
name: qwen3
version: "1.0"

# 配置占位符（从模型文件推导）
config:
  num_layers: ${num_hidden_layers}
  hidden_size: ${hidden_size}
  num_heads: ${num_attention_heads}
  num_kv_heads: ${num_key_value_heads}
  head_dim: ${head_dim}
  intermediate_size: ${intermediate_size}
  vocab_size: ${vocab_size}
  rope_theta: ${rope_theta}
  dtype: ${dtype}  # f16/bf16/f32

# 图定义
graph:
  inputs:
    - name: input_ids
      dtype: int64
      shape: [batch, seq_len]

  outputs:
    - name: logits
      dtype: ${dtype}
      shape: [batch, seq_len, ${vocab_size}]

  nodes:
    - name: embed
      op_type: Gather
      inputs: [model.embed_tokens.weight, input_ids]
      outputs: [hidden_states]

    - repeat: ${num_layers}
      var: i
      nodes:
        - name: layer_${i}_norm
          op_type: SimplifiedLayerNormalization
          inputs: [hidden_${i}, model.layers.${i}.input_layernorm.weight]
          outputs: [normed_${i}]
        # ... 更多节点

# 融合提示（可选，帮助优化器）
fusion_hints:
  - pattern: [q_proj, k_proj, v_proj, rope]
    target: FusedQkvRope
```

### 4.2 占位符解析

| 占位符 | 来源 |
|--------|------|
| `${num_hidden_layers}` | GGUF: `llama.block_count` / SafeTensors: 张量名最大层号 |
| `${hidden_size}` | GGUF: `llama.embedding_length` / SafeTensors: embed 张量形状 |
| `${num_attention_heads}` | GGUF: `llama.attention.head_count` / SafeTensors: q_proj 形状推导 |
| `${head_dim}` | 计算: hidden_size / num_heads |
| `${dtype}` | 权重张量实际 dtype |

---

## 5. 优化器 Pass 规格

### 5.1 Pass 接口

```rust
pub trait OptimizationPass {
    fn name(&self) -> &'static str;
    fn run(&self, graph: OnnxGraph, ctx: &OptimizationContext) -> Result<OnnxGraph>;
    fn enabled(&self, ctx: &OptimizationContext) -> bool { true }
}

pub struct OptimizationContext {
    pub backend_type: BackendType,
    pub cuda_sm_version: Option<(u32, u32)>,
    pub available_memory: usize,
    pub dtype: Dtype,
}
```

### 5.2 融合模式

| 模式名 | 输入节点序列 | 输出融合算子 |
|--------|--------------|-------------|
| FlashAttention | MatMul(Q,K) → Scale → Softmax → MatMul(_,V) | `FusedKernel::FlashAttention` |
| SwiGLU | Linear(gate) → SiLU → Mul(_, Linear(up)) | `FusedKernel::SwiGlu` |
| FusedQkvRope | Linear(Q) + Linear(K) + Linear(V) + RoPE | `FusedKernel::FusedQkvRope` |
| FusedRMSLinear | RMSNorm → Linear | `FusedKernel::FusedRMSLinear` |

---

## 6. 删除清单

### 6.1 文件删除

```
删除目录:
- src/adapter/                    # 整个目录

删除文件:
- src/loader/config.rs            # config.json 解析
```

### 6.2 函数删除

```rust
// src/loader/mod.rs
- fn convert_to_f32(dtype: Dtype, data: &[u8]) -> Result<Cow<'_, [f32]>>
- fn convert_to_element<E: Element>(dtype: Dtype, data: &[u8]) -> Result<Vec<E>>

// src/model_config.rs
- 所有 config.json 回退逻辑
- fn dtype_size_from_config()
- fn rope_scaling_from_json()
```

### 6.3 依赖移除

```rust
// src/engine/executor.rs
- use crate::adapter::*;
- adapter: &'static dyn ModelAdapter<B, E>
- adapter.load_weights()

// src/lib.rs
- pub mod adapter;
```

---

## 7. 实施计划

### Phase 1: 清理 (估时 2h) ✅ 已完成 [commit: 3cd8da8]
- [x] T1.1: 删除 `src/adapter/` 目录
- [x] T1.2: 删除 `src/loader/config.rs`
- [x] T1.3: 删除转换函数
- [x] T1.4: 删除 config.json 回退
- [x] T1.5: 移除 executor adapter 依赖
- [x] T1.6: 修复编译错误

### Phase 2: 架构模板 (估时 4h) ✅ 已完成 [commit: 3cd8da8]
- [x] T2.1: 创建 `src/arch/` 目录
- [x] T2.2: 实现 YAML 解析器
- [x] T2.3: 实现权重绑定
- [x] T2.4: 编写 qwen3.yaml
- [x] T2.5: 单元测试

### Phase 3: 优化器 (估时 4h) ✅ 已完成 [commit: 3cd8da8]
- [x] T3.1: 创建 `src/graph/` 模块
- [x] T3.2: 实现 Pass trait
- [x] T3.3: 实现模式融合 Pass
- [x] T3.4: 实现优化器管道
- [x] T3.5: 单元测试

### Phase 4: 集成 (估时 3h) ✅ 已完成 [commit: 3cd8da8]
- [x] T4.1: 重写 Executor
- [x] T4.2: 端到端测试
- [ ] T4.3: 性能验证 (待后续补充)

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 删除 adapter 导致大量编译错误 | 高 | 中 | 分阶段删除，先注释再删 |
| YAML 模板表达能力不足 | 低 | 高 | 预留扩展点，支持自定义 op |
| 融合模式匹配失败 | 中 | 中 | 保留 Atomic fallback |
| 零转换后精度问题 | 低 | 高 | 保留可选转换路径用于调试 |

---

## 9. 验收标准

1. **编译通过**: `cargo check` 无错误
2. **测试通过**: `cargo test --lib` 通过
3. **无 config.json**: 删除所有 config.json 依赖
4. **无 adapter**: `src/adapter/` 目录不存在
5. **零转换**: 权重原生精度直接使用
6. **端到端**: Qwen3-0.5B 推理输出正确
