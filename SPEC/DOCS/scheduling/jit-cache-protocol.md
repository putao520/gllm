# JIT 编译缓存协议 (JIT Cache Protocol)

> **SSOT 更新**: 2026-03-28
> **关联需求**: REQ-JIT-CACHE-001~007, ARCH-JIT-FIRST, ARCH-CPU-GPU-UNIFIED

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

缓存键 = **模型结构签名** × **环境维度模板**，不包含任何运行时动态维度：

```rust
ModelArchKey {
    model_id: String,       // "meta-llama/Llama-3-70B" 或模型路径 hash
    backend: BackendKind,   // Cuda / Hip / Metal / Cpu
    // 环境感知扩展 (ARCH-ENV-VARIANT):
    // 描述该模型需要编译哪些环境维度及其合法枚举值
    env_schema: EnvSchema,
    // ... 其他全模型级静态参数
}
```

**EnvSchema** 定义见 03-DATA-STRUCTURE.md §16.2。它不包含运行时值，仅描述"该模型支持哪些环境维度"。
例如，MoE 模型比稠密模型多一个 `moe_load_balance` 维度。

**L3 磁盘缓存文件名格式**:
```
{model_hash}_{backend}_{env_schema_hash}.bin
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
    → 构造模型级的 ModelArchKey (含 EnvSchema)
    → 查询磁盘/内存级缓存
    → 没有命中时: 解析 YAML 模板 (如 llama.yaml) 构造全网结构图
    → GraphExecutor 接管进行全局统一编译（仅在加载时发生一次）
    → 【新增】枚举 EnvSchema 的合法 EnvVector 笛卡尔积
    → 【新增】对每个 (OpKind, EnvVector) 组合编译物理变体
    → 【新增】构建 EnvVariantRegistry 完美哈希跳表
    → 将执行器本身 (含变体注册表) 作为唯一缓存实体写回
    → 推理阶段全程由持有的单体缓存环境接管前向传播
```

---

## 5. 环境感知变体编译协议 (ARCH-ENV-VARIANT-CACHE)

> **关联**: 02-ARCHITECTURE.md §17, 03-DATA-STRUCTURE.md §16

### 5.1 变体编译时机

变体编译发生在**模型加载时**，与主图编译同批次。不违反"推理热路径零编译"铁律。

### 5.2 变体笛卡尔积剪枝

EnvVector 的理论组合数为 `3 × 3 × 3 × 2 × 3 = 162`，但经过以下剪枝后每个 OpKind 仅 4-8 个有效变体：

| 剪枝规则 | 示例 |
|---------|------|
| 互斥约束 | Phase=Prefill 时 Sharpness 不适用 (固定为 Flat) |
| 架构约束 | 非 MoE 模型不编译 dead_neuron 维度 |
| 性能约束 | 某些变体性能在最优 80% 以内，合并为一个通用变体 |
| Autotuning 剔除 | 首次运行微基准测试后，剔除冷变体 |

### 5.3 变体缓存结构

```text
L1 (内存): EnvVariantRegistry (完美哈希表 + 机器码偏移)
    └── 每个 (OpKind, EnvVector) → VariantId → code_offset

L2 (全局 LRU): 跨模型共享的变体机器码页 (4KB 对齐)
    └── key: (OpKind, EnvVector, IsaLevel, QuantType)
    └── value: 编译后的机器码字节

L3 (磁盘): 模型级二进制
    └── 文件: {model_hash}_{backend}_{env_schema_hash}.bin
    └── 内含: 所有变体机器码 + 注册表序列化
    └── TTL: 7 天自动清理
```

### 5.4 变体冷热分级

| 级别 | 条件 | 存储 | 恢复机制 |
|------|------|------|---------|
| **热** | hit_rate > 10% | L1 + L2 + L3 | 直接跳表命中 |
| **温** | 1% < hit_rate <= 10% | L2 + L3 | LRU 缓存命中 |
| **冷** | hit_rate <= 1% | L3 磁盘 | Deopt → 沙盒编译 → 热插入 |
| **僵尸** | 0 hit in 100K steps | L3 或剔除 | Uncommon Trap → §15.4 恢复 |
