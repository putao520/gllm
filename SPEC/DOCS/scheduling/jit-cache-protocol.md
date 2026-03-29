# JIT 编译缓存协议 (JIT Cache Protocol)

> **SSOT 更新**: 2026-03-28
> **关联需求**: REQ-JIT-CACHE-001~007, ARCH-JIT-FIRST, ARCH-CPU-GPU-UNIFIED
> **关联文档**: [p4-p5-next-gen-optimizations.md](./p4-p5-next-gen-optimizations.md) §6, [optimization_strategy_master.md](./optimization_strategy_master.md) Tier V

---

## 1. 核心铁律

> **JIT 编译只允许在两个时间点发生**：
> 1. **模型加载时** — 静态形状已知，基于 `GraphExecutor` 一次性编译整个模型的全网计算图。
> 2. **Autotuning 窗口期** — 首次推理的前 3-5 秒，执行微基准测试搜索最优 tile 配置。
>
> **推理热路径中禁止出现任何编译行为。**
>
> **CPU 与 GPU 后端共享完全一致的全模型执行图。**

### 禁止行为

| 禁止 | 原因 | 检测方法 |
|------|------|---------|
| 热路径中 `InferenceCompiler::new()` | 每步重建编译器 = 50-500× 编译开销 | `grep -rn "InferenceCompiler::new()" src/compat/decoder_forward.rs` |
| 热路径中 `compile_graph()` | 每步重编译 = PTX/HIP 编译 5-50ms/次 | `grep -rn "compile_graph" src/compat/decoder_forward.rs` |
| 算子级/层级分离缓存 | 阻断了跨层与全网层面的调度与融合 | 禁止出现定义特定层（如 FusedAttentionLayer 等 GraphType）的离散块缓存。 |

---

## 2. 缓存架构

### 2.1 缓存键 (Cache Key)

缓存键 = **模型结构签名**，不包含任何运行时动态维度：

```rust
ModelArchKey {
    model_id: String,       // "meta-llama/Llama-3-70B" 或模型路径 hash
    backend: BackendKind,   // Cuda / Hip / Metal / Cpu
    // ... 其他全模型级静态参数
}
```

### 2.2 缓存粒度 (Graph Executor)

> **铁律**：缓存粒度为 **全模型单体计算图 (Whole-Model Graph)**。
> `gllm` 彻底抛弃基于 `GraphType` 分离算子或分离网络层的缓存体制（摒弃诸如 FusedAttentionLayer、FusedFfnLayer 的碎块化管理）。整个架构的计算流必须被 `GraphExecutor` 于加载期统一解析为**唯一的执行结构实例**。
>
> 内存中仅存在单一共享实例。禁止任何层级、算子级的局部后备缓存。所有前向计算在执行时直接在同一个预编译大图结构内调度。
>
> **在线旋转矩阵归属**: TurboQuant 2.0 的在线 FWHT 旋转矩阵 $R_{online}^{(l)}$ 是**模型级静态常数**（每层一个，不随请求变化）。它作为模型加载时的附属元数据与权重一起注入 `GraphExecutor`，在 JIT 编译时作为立即数常量嵌入 FWHT 指令序列。不新建独立缓存。

### 2.3 CPU / GPU 后端统一要求

无论是 CPU 还是 GPU，模型加载时必须只生成且只缓存一条 `GraphExecutor` 执行流水线。不允许 GPU 使用一种按层编译的 HashMap，而 CPU 使用另一种独立拓扑。
所有后端通过统一的上下文环境，共同依赖底层单一真源的大图调度。

---

## 3. 动态维度处理：SymDim::Symbolic + ShapeBinding

### 3.1 核心原理

**静态维度**（模型加载时已知）→ `SymDim::Concrete(value)` → 编译进 kernel
**动态维度**（每步变化）→ `SymDim::Symbolic("name")` → 运行时 `ShapeBinding` 绑定

| 维度 | 性质 | 编码方式 | 绑定时机 |
|:---|:---|:---|:---|
| `hidden_size` | 静态 | `Concrete(4096)` | 编译时 |
| `seq_len` | **动态** | `Symbolic("seq_len")` | launch 时 |
| `total_seq` | **动态** | `Symbolic("total_seq")` | launch 时 |

---

## 4. 模型加载时编译流程

```text
Client::load_model(model_id)
    → Loader 加载权重 + config.json
    → 构造模型级的 ModelArchKey
    → 查询磁盘/内存级缓存
    → 没有命中时: 解析 YAML 模板 (如 llama.yaml) 构造全网结构图
    → GraphExecutor 接管进行全局统一编译（仅在加载时发生一次）
    → 将执行器本身作为唯一的缓存实体写回
    → 推理阶段全程由持有的单体缓存环境接管前向传播
```
