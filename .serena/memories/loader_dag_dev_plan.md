# Loader DAG 重构开发计划

> **基于 SPEC**: SPEC/08-LOADER-REFACTOR.md

## 目标

将 GGUF/SafeTensors/ONNX 三种格式统一到 OnnxGraph 表示，通过 DAG 优化器实现算子融合，最终由 FusedGraphExecutor 执行。

## 当前已实现

### 1. 架构模板系统 (`src/arch/`)
- ✅ `ArchTemplate` 结构体（name, version, config, graph, fusion_hints, tensor_patterns）
- ✅ `registry.rs` - 模板注册表
- ✅ `resolve.rs` - 配置解析
- ✅ `template.rs` - 模板定义

### 2. Graph 模块 (`src/graph/`)
- ✅ `FusedGraph` - 融合图结构
- ✅ `FusedNode` - 融合节点
- ✅ `FusedOp` - 融合算子枚举
- ✅ `GraphOptimizer` - 优化器框架
- ✅ `OptimizationPass` trait
- ✅ 模式融合 Pass (`pattern_fusion.rs`)
- ✅ 死代码消除 Pass (`dead_code.rs`)

### 3. Loader 模块 (`src/loader/`)
- ✅ `TensorProvider` trait - 统一张量提供接口
- ✅ GGUF/SafeTensors/ONNX 各自实现 `TensorProvider`
- ✅ `to_unified_graph()` 方法 - 转换为 OnnxGraph

## 待实现

### Phase 1: 清理冲突代码 (REQ-REFACTOR-001~005)

| REQ ID | 任务 | 状态 |
|--------|------|------|
| REQ-REFACTOR-001 | 删除 adapter 模块 | 🟢 待验证 |
| REQ-REFACTOR-002 | 删除 loader/config.rs | 🟢 待验证 |
| REQ-REFACTOR-003 | 删除精度转换函数 | 📋 待实现 |
| REQ-REFACTOR-004 | 删除 config.json 回退 | 📋 待实现 |
| REQ-REFACTOR-005 | 移除 executor 的 adapter 依赖 | 🟢 待验证 |

### Phase 2: 架构模板完善 (REQ-ARCH-001~005)

| REQ ID | 任务 | 状态 |
|--------|------|------|
| REQ-ARCH-001 | YAML 模板格式支持 repeat/占位符 | 📋 待验证 |
| REQ-ARCH-002 | YAML → OnnxGraph 解析器 | 📋 待实现 |
| REQ-ARCH-003 | 权重绑定机制 | 📋 待实现 |
| REQ-ARCH-004 | Qwen3 架构模板 | 📋 待实现 |
| REQ-ARCH-005 | 配置推导完善 | 🟢 部分实现 |

### Phase 3: DAG 优化器完善 (REQ-OPT-001~005)

| REQ ID | 任务 | 状态 |
|--------|------|------|
| REQ-OPT-001 | OptimizationPass trait | ✅ 已实现 |
| REQ-OPT-002 | 模式融合 Pass | 🟡 需完善 |
| REQ-OPT-003 | 硬件感知融合 | 📋 待实现 |
| REQ-OPT-004 | 优化器管道 | ✅ 已实现 |
| REQ-OPT-005 | FusedGraph 生成 | ✅ 已实现 |

### Phase 4: 执行器重写 (REQ-EXEC-001~004)

| REQ ID | 任务 | 状态 |
|--------|------|------|
| REQ-EXEC-001 | 统一加载入口 `Loader::load() -> OnnxGraph` | 🟡 部分实现 |
| REQ-EXEC-002 | FusedGraph 执行 | 📋 待实现 |
| REQ-EXEC-003 | 零转换推理 | 📋 待实现 |
| REQ-EXEC-004 | 端到端测试 | 📋 待实现 |

## 执行顺序

```
1. 验证现有实现状态
2. Phase 1: 清理冲突代码
3. Phase 2: 完善架构模板系统
4. Phase 3: 完善 DAG 优化器
5. Phase 4: 重写执行器
6. 端到端验证
```

## 关键数据结构

### FusedOp 枚举
```rust
pub enum FusedOp {
    FlashAttention(FlashAttentionConfig),
    SwiGLU(SwiGLUConfig),
    RoPE(RoPEConfig),
    FusedQkvRope(FusedQkvRopeConfig),
    FusedRMSLinear(FusedRMSLinearConfig),
    Atomic(AtomicOp),
}
```

### 融合模式
| 模式名 | 输入节点序列 | 输出融合算子 |
|--------|--------------|-------------|
| FlashAttention | MatMul(Q,K) → Scale → Softmax → MatMul(_,V) | `FusedOp::FlashAttention` |
| SwiGLU | Linear(gate) → SiLU → Mul(_, Linear(up)) | `FusedOp::SwiGLU` |
| FusedQkvRope | Linear(Q) + Linear(K) + Linear(V) + RoPE | `FusedOp::FusedQkvRope` |
| FusedRMSLinear | RMSNorm → Linear | `FusedOp::FusedRMSLinear` |
