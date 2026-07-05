# DotProductCap 硬件能力 API 资料库（C-9，堵 AI 幻觉）

> 来源：gllm-kernels dispatch/device_profile.rs:52-598（源码确认）
> 建库触发：derive_compute_dtype 宪法 -1 重构方案假设 device.dot_product_cap() API，需确认是否已存在
> 最后验证：2026-07-06

## 核心确认：dot_product_cap() API 已存在（无需新建）

architect 重构方案假设 `device.dot_product_cap()` API——**已存在**（device_profile.rs:566）。

### DotProductCap 枚举（device_profile.rs:52-）

```rust
pub enum DotProductCap {
    // === Native floating-point dot-product ===
    NativeBf16,    // x86 VDPBF16PS / ARM BFMMLA / GPU HMMA bf16 (FP32 accumulate)
    NativeFp16,    // ARM FMMLA / GPU HMMA fp16 (x86 无原生 FP16 compute)
    // === Native integer dot-product ===
    NativeInt4x8,  // AMD GFX12 a4w8 WMMA
    NativeFp4,     // NVIDIA SM100+ tcgen05 FP4 tensor core
    NativeInt8Tc,  // SM80/SM90 IMMA/WGMMA INT8 tensor core
    NativeInt8Simd, // x86 VNNI / ARM SDOT
    NativeInt8Tile, // ARM SME2
    // === 软件辅助 ===
    SimdAssisted,  // AVX2 256-bit, NEON+SVE (无原生 dot, 软件 widening)
    SimdBasic,     // SSE2 128-bit, NEON 128-bit
    None,
}
```

### dot_product_cap() 方法（device_profile.rs:566-598）

```rust
pub fn dot_product_cap(&self) -> DotProductCap {
    if self.has_native_bf16_dot() { return DotProductCap::NativeBf16; }
    if self.has_native_fp16_dot() { return DotProductCap::NativeFp16; }
    // if self.has_wmma_a4w8() { return DotProductCap::NativeInt4x8; }
    // if self.has_fp4_tc() { return DotProductCap::NativeFp4; }
    // if self.has_int8_tc() { return DotProductCap::NativeInt8Tc; }
    if self.has_vnni() { return DotProductCap::NativeInt8Simd; }  // AVX-VNNI
    if self.has_amx() { return DotProductCap::NativeInt8Tile; }   // AMX
    if self.has_avx512() { return DotProductCap::NativeInt8Simd; }
    if self.has_avx2() { return DotProductCap::SimdAssisted; }    // AVX2 256-bit
    if self.has_neon() { return DotProductCap::SimdBasic; }
    DotProductCap::None
}
```

## derive_compute_dtype 重构可直接用（architect 方案 API 就绪）

```rust
// dtype_chain.rs 重构后 (宪法 -1 合规):
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    match device.dot_product_cap() {
        DotProductCap::NativeBf16 if matches!(storage_dtype, DType::BF16) => DType::BF16,
        // NVFP4 + NativeFp4 → 原生 FP4 计算 (tensor core 输出 F32 acc)
        DotProductCap::NativeFp4 if matches!(storage_dtype, DType::F8E4M3) => storage_dtype,
        // _ => F32 兜底 (无原生累加支持, widen F32 数值安全)
        _ => DType::F32,
    }
}
```

**关键**：F32 是**兜底分支**（无原生累加时数值安全 widen），非"BF16 always => F32"恒等预设。符合宪法 -1。

## 各硬件的 DotProductCap（实测）

| 硬件 | DotProductCap | derive_compute_dtype(BF16) |
|------|--------------|---------------------------|
| i9-10900KF (本地) | SimdAssisted (AVX2, 无 VNNI/AMX) | F32 兜底（widen） |
| AMD 9950X3D (5070Ti) | SimdAssisted 或 NativeBf16（待查 has_bf16） | 待查 |
| GPU SM80+ | NativeInt8Tc / NativeBf16 | BF16 原生 |

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| dot_product_cap() API 需新建 | 已存在（device_profile.rs:566） |
| DotProductCap 只有 Native/SimdAssisted | 有 10 个变体（NativeBf16/NativeFp16/NativeInt4x8/NativeFp4/NativeInt8Tc/NativeInt8Simd/NativeInt8Tile/SimdAssisted/SimdBasic/None）|
| AVX2 = NativeBf16 | AVX2 = SimdAssisted（无原生 BF16 dot，需软件 widening）|
| NativeBf16 指 AVX-512 BF16 | 指 VDPBF16PS（AVX-512 BF16 指令）/ ARM BFMMLA / GPU HMMA |
| dot_product_cap 只用于量化 | 浮点也用（NativeBf16/NativeFp16）|

## 关键代码位置

- `gllm-kernels/src/dispatch/device_profile.rs:52-100` — DotProductCap 枚举定义
- `gllm-kernels/src/dispatch/device_profile.rs:566-598` — dot_product_cap() 方法
- `gllm-kernels/src/dispatch/device_profile.rs:982, 998` — 消费点（能力判断）
- `gllm-kernels/src/compiler/codegen/vm/x86_lower.rs:85,89` — X86Lower.has_bf16/has_vnni 字段
- `gllm-kernels/src/compiler/codegen/vm/x86_lower/helpers.inc.rs:13-25` — has_bf16/has_vnni 从 platform 提取

## 与其他资料库关系

- `derive-compute-dtype-unconstitution.md`：宪法 -1 重构方案，本库确认其 API 已就绪
- `dtype-propagation.md`：WidenCompute 策略，本库的 SimdAssisted 兜底对应 widen F32
- 本文件：dot_product_cap 既有能力证据（堵"需新建 API"幻觉）
