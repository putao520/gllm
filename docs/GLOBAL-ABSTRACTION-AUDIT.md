# 全局工程抽象审计报告 (Global Abstraction Audit)

> **目标**: 确保 `gllm` (Client) 与 `gllm-kernels` (Backend) 之间的工程抽象达到完美 SSOT 状态，无逻辑断层，无性能陷阱。

## 1. 跨层数据流审计 (Layer Interaction)

我们定义的 4 层架构：**Manifest (L1) -> Adapter (L2) -> Engine (L3) -> Driver (L4)**。

### 关键路径验证

| 路径 | 数据对象 | 传输方式 | 审计结论 | 潜在风险 |
|---|---|---|---|---|
| **L1 -> L2** | `ModelManifest` | `&'static` 引用 | ✅ Pass | 无，Rust 静态生命周期完美契合。 |
| **L2 -> L3** | `Weights` | Mmap / GPU Ptr | ✅ Pass | 需确保 Adapter 不产生中间 `Vec<f32>` 拷贝。 |
| **L3 -> L4** | `TokenTree` | `&AttentionTopology` | ✅ Pass | L3 负责构建拓扑，L4 负责只读访问。 |
| **L4 -> L3** | `Logits` | `LogitsTensor` Handle | ✅ Pass | **物理阻断**了数据回流，设计完美。 |

### 生命周期 (Lifetime) 约束

*   **Weights**: 必须活得比 Engine 久 (`Arc<Weights>`)。
*   **KV Cache**: 由 Engine 管理生命周期，Driver 仅借用 (`&mut`)。
*   **Graph**: CUDA Graph 录制期间，所有指针必须固定 (Pinning)。

---

## 2. Client 侧抽象审计 (gllm)

### Manifest (SSOT)
*   **现状**: `ModelManifest` 包含 Repo, Arch, TensorRules。
*   **缺口**: 缺少 **MoE 路由策略** 配置 (Top-K, Capacity Factor)。
*   **修正**: 需要在 `ModelManifest` 中增加 `MoEConfig` 字段。

### Adapter (Logic)
*   **现状**: `ModelAdapter` 处理权重加载。
*   **缺口**: 缺少 **Tokenizer 适配**。Qwen3 和 Llama4 的 Chat Template 处理逻辑不同，这也属于 Adapter 职责。
*   **修正**: `ModelAdapter` 需包含 `apply_chat_template` 方法。

---

## 3. Backend 侧抽象审计 (gllm-kernels)

### Driver API (Interface)
*   **现状**: `Backend` Trait 定义了 L3 接口。
*   **挑战**: 如何在 Trait 中表达 **AOT Binary** 的加载？
*   **方案**: Backend 初始化时 (`new()`) 接收设备 ID，内部自动完成 AOT 加载。Trait 接口只需暴露 `forward`。

### Tensor System (Type)
*   **现状**: `LogitsTensor`, `QuantizedTensor`。
*   **挑战**: 如何统一 CPU 和 GPU 的 `QuantizedTensor`？
*   **方案**: 定义 `trait QuantizedStorage`，分别由 `CudaSlice<u8>` 和 `Vec<u8>` 实现。

---

## 4. 最终契约定义 (The Final Contract)

基于审计，我们将核心契约细化为以下 3 个 Rust 文件结构：

### A. `gllm/src/manifest.rs` (L1)
```rust
pub struct ModelManifest {
    pub arch: ModelArchitecture,
    pub moe_config: Option<MoEConfig>, // 新增
    // ...
}
```

### B. `gllm/src/adapter.rs` (L2)
```rust
pub trait ModelAdapter: Send + Sync {
    fn load_weights(&self, ...) -> Result<Weights>;
    fn chat_template(&self) -> &str; // 新增
}
```

### C. `gllm-kernels/src/backend_trait.rs` (L4)
```rust
pub trait Backend {
    type Tensor; // 关联类型，支持 GPU/CPU 差异

    fn generator_forward_gpu_pure(
        &self,
        topology: &AttentionTopology,
        // ...
    ) -> Result<Self::Tensor, Error>;
}
```

## 5. 结论

**整体架构逻辑自洽，闭环完整。**
唯一的修补项是在 `ModelManifest` 中增加 MoE 配置，以及在 `ModelAdapter` 中增加 Tokenizer 支持。

**状态**: **Audit Passed. Ready for Code.**
