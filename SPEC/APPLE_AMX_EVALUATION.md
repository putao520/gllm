# Apple M 系列 AMX 技术评估

## 概述

Apple M 系列芯片内置的 **AMX (Apple Matrix coprocessor)** 是一个不公开的矩阵加速单元。本文档评估其在 Rust LLM 推理中的可行性。

## AMX 现状

### 已知事实
- **无公开 API**: Apple 未提供任何官方文档或 API
- **Accelerate.framework**: Apple 的 BLAS/LAPACK 实现会自动利用 AMX
- **逆向工程项目**: [dougallj/applegpu](https://github.com/dougallj/applegpu) 有部分 AMX 指令逆向

### 性能参考
| 芯片 | AMX 通道 | 理论峰值 (FP32) |
|------|---------|----------------|
| M1 | 2 | ~2 TFLOPS |
| M2 | 2 | ~3 TFLOPS |
| M3 | 2 | ~3.5 TFLOPS |
| M4 | 2 | ~4 TFLOPS |

## 可行策略

### 策略 1: FFI 调用 Accelerate.framework ✅ 推荐
```rust
// 通过 cblas_sgemm 间接利用 AMX
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: c_int, transA: c_int, transB: c_int,
        M: c_int, N: c_int, K: c_int,
        alpha: f32, A: *const f32, lda: c_int,
        B: *const f32, ldb: c_int,
        beta: f32, C: *mut f32, ldc: c_int
    );
}
```

**优点**: 
- 稳定、官方支持
- 自动利用 AMX 加速
- 无需逆向工程

**缺点**:
- 额外 FFI 开销
- 无法针对特定用例优化

### 策略 2: 直接使用 AMX 指令 ❌ 不推荐
- 需要逆向工程
- 可能因 macOS 更新而失效
- 存在法律风险 (DMCA)

### 策略 3: Metal GPU 计算 ⚠️ 考虑
对于大型模型，Metal GPU 可能比 CPU+AMX 更快。

## 实施建议

### 短期方案 (已实现)
继续使用 NEON 内核 + `gemv_dispatch` 并行化：
- SmolLM2-135M (576): ~4 GFLOPS
- Llama-7B (4096): ~5.6 GFLOPS

### 中期方案 (可选)
添加 Accelerate.framework 后端:
```rust
#[cfg(target_os = "macos")]
fn linear_forward_accelerate(...) {
    // 调用 cblas_sgemm
}
```

### 长期方案
若 Apple 开放 AMX API，立即集成。

## 结论

**当前推荐**: 不直接使用 AMX，而是通过 Accelerate.framework FFI 间接利用。这提供了最佳的稳定性和性能平衡。

对于 gllm 项目，现有的 NEON 实现已经充分利用了 ARM64 SIMD 能力，对于 Batch=1 推理场景已接近最优。
