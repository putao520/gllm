# SPEC-07: 极致性能方案 (Extreme Performance Scheme)

## 1. 背景与目标

当前 `gllm` 在小参数模型（如 SmolLM2-135M）推理时，性能受限于以下三个核心因素：
1. **混合推理延迟**：关注点在 CPU-GPU 间的频繁同步与数据拷贝 (H2D/D2H)。
2. **权重加载开销**：每次推理前可能涉及权重的重新验证或局部拷贝，而非显存常驻。
3. **计算密度低**：CPU 线性层在 batch=1 时无法充分利用片上缓存与并行能力。

**目标**：实现全路径 GPU 推理缓存与计算，消除热路径上的 CPU 介入。

## 2. 核心设计

### 2.1 权重常驻 (Weight Pinning)

目前 `LinearWeights` 仅存储在内存中。我们将引入 `GpuLinearWeights` 机制：
- **Lazy Loading**: 第一次使用 GPU 推理时，将权重上传至 GPU VRAM。
- **Buffer Reuse**: 权重以 `wgpu::Buffer` 形式常驻，生命周期与 `GeneratorModel` 绑定。
- **Zero Copy**: 推理时仅传递 GPU Buffer 引用，而非数据切片。

### 2.2 GPU 线性算子 (GEMV)

在 `gllm-kernels` 中引入针对 Decode 阶段优化的 GEMV (Matrix-Vector) 算子：
- **WGSL 实现**：优化工作组（Workgroup）大小以适应不同维度的隐藏层（如 576, 1536）。
- **精度支持**：支持 f32 和 f16（通过 `f16` 扩展或手动打包）。

### 2.3 零拷贝推理路径 (Zero-copy Path)

改造 `forward_inplace` 流程：
1. **Input**: 首个 Token 的 Embedding 在 GPU 上生成。
2. **Layers**: 
   - 所有的 Norm (RMSNorm) 在 GPU 上执行。
   - QKV 投影通过 GPU GEMV 执行。
   - Attention (Flash Attention) 直接读取 GPU 上的 QKV 缓冲区。
   - FFN (Gate/Up/Down) 在 GPU 上执行 GEMV。
3. **Output**: 仅将最终的 Logits 拷贝回 CPU 进行采样。

## 3. 架构演进

### 3.1 gllm-kernels 接口变更
```rust
impl KernelDispatcher {
    pub fn linear_forward<T>(
        &self,
        input: &GpuTensor,  // 新增：支持 GPU Tensor 引用
        weight: &GpuWeight, 
        output: &mut GpuTensor,
        config: LinearConfig
    );
}
```

### 3.2 路由逻辑调整
`Device::Auto` 将根据显存余量和计算单元密度进行更有侵略性的 GPU 路由。

## 4. 预期性能指标 (SmolLM2-135M / RTX 3060)

| 场景 | 当前速度 | 目标速度 | 提升 |
|------|----------|----------|------|
| CPU (6-core) | 6.7 t/s | 50+ t/s | 7x |
| GPU (RTX 3060) | 6.9 t/s | 200+ t/s | 30x |
