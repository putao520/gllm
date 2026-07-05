# derive_compute_dtype 违宪铁证（C-9，堵 AI 幻觉）

> 来源：gllm-kernels dtype_chain.rs:195-210 + graph_geometry.rs:127 + executor_compile.rs:185-206 + 运行时插桩实测
> 建库触发：运行时钉死 SmolLM2 compute_dtype=F32（config 是 BF16）→ 追出 derive_compute_dtype 硬编码 BF16→F32 降级，宪法1（ARCH-BLOB-YIELDS-WEIGHT）违宪
> 最后验证：2026-07-05

## 违宪铁证（源码 + 运行时双确认）

### 运行时实测（commit 本资料库前插桩）
```
[DIAG-KV-BUF] compute_dtype=F32 elem_bytes=4 ... (SmolLM2, config torch_dtype=bfloat16)
[DIAG-GEO] geometry.compute_dtype=F32 (GraphDerivedGeometry, 非 ModelGeometry)
```
SmolLM2 config.json `torch_dtype=bfloat16`，权重实际 BF16（safetensors 实测），但运行时 compute_dtype=**F32**。

### 违宪源码（dtype_chain.rs:195-210）
```rust
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    match storage_dtype {
        // BF16/F16 always widen to F32 on current hardware
        DType::BF16 | DType::F16 => DType::F32,  // ← 硬编码 BF16→F32 降级!
        // Quantized types always dequant to F32
        DType::U8 | DType::F8E4M3 | ... => DType::F32,
        // F32 stays F32
        DType::F32 => DType::F32,
    }
    // Note: device parameter is reserved for future hardware that supports
    // native BF16 accumulation. Currently all paths result in F32.
}
```

**违宪点**：line 198 `DType::BF16 | DType::F16 => DType::F32` —— 硬编码把 BF16 compute 降级成 F32，**无视 device 参数**（函数收了 device 但 match 里不用）。

## 违宪因果链（BF16 数据 → F32 代码）

```
config.json torch_dtype=bfloat16 (数据是 BF16)
  → 权重 safetensors 实际 BF16 (实测确认)
    → ModelGeometry.dtype = BF16 (config_impl.inc.rs:401)
      → ModelGeometry.compute_dtype = unwrap_or(dtype) = BF16 (types.inc.rs:167) [纸面正确]
        但实际 MegaKernelCompiled.compute_dtype 来自 GraphDerivedGeometry:
          → graph_geometry.rs:127 compute_dtype = derive_compute_dtype(storage_dtype, device)
            → dtype_chain.rs:198 BF16 => F32  [违宪! 硬编码降级]
              → GraphDerivedGeometry.compute_dtype = F32
                → executor_core.inc.rs:371 MegaKernelCompiled.compute_dtype = F32
                  → abi_types.inc.rs:395 elem_bytes() = 4 (F32)
                  → KV cache 按 F32(768/行) 分配
                  → executor_compile.rs:185 needs_dtype_conversion = (F32 != BF16) = true
                    → executor_compile.rs:193 dequantize_weight_to_dtype(BF16 → F32)
                      → BF16 权重被转成 F32 字节存进 blob [宪法1违宪! blob 应保留 BF16]
```

## 宪法1违宪（ARCH-BLOB-YIELDS-WEIGHT）

**宪法1**：Weight blob 的内存布局必须与权重文件中的原始布局完全一致，禁止任何格式转换。Blob 保留原始 dtype 原始字节。

**违宪**：`derive_compute_dtype` 返回 F32 → `executor_compile.rs:193 dequantize_weight_to_dtype` 把 BF16 权重转成 F32 字节存进 blob。**blob 不再保留原始 BF16 字节**，而是 F32 字节。

**宪法3违宪传染形态**：
- 代码方（derive_compute_dtype）被实现成"BF16 必须降级 F32"（硬编码 match arm）
- 为了喂给这个违宪的 compute_dtype，loader 把 BF16 权重转 F32（dequantize_weight_to_dtype）
- 转换层存在的"理由"被记为"compute_dtype 是 F32" —— 但 compute_dtype=F32 本身就是违宪现状
- 结果：违宪从 derive_compute_dtype 传染到 loader，正确数据（BF16）被改错（转 F32）

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码 + 运行时证明） |
|--------|---------|
| compute_dtype 从 config.compute_dtype 来（ModelGeometry） | MegaKernelCompiled 用 GraphDerivedGeometry.compute_dtype（from_graph 推导），非 ModelGeometry |
| BF16 权重保留原始 BF16 字节进 blob | derive_compute_dtype 返回 F32 → loader dequantize BF16→F32 → blob 存 F32 字节（违宪） |
| BF16→F32 是 JIT 层 WidenCompute（正确） | WidenCompute 在 SIMD 指令层 widen，但这里是 loader 层把权重字节转 F32（违宪，不是 widen） |
| compute_dtype=F32 自洽（KV cache 读写都 F32） | 表面自洽，但底层违宪：BF16 数据被降级 F32，代码没顺从数据 |
| device 参数驱动 compute_dtype | device 参数被忽略（match 不用），硬编码 BF16→F32 |
| derive_compute_dtype 注释"future hardware"是待办 | 这是违宪现状，非待办——当前就错 |

## 与 SmolLM2 logits 发散的关系

derive_compute_dtype 违宪（BF16→F32）**理论应更精确**（F32 > BF16），不该直接致发散。但：
1. **dequantize_weight_to_dtype(BF16→F32) 过程可能出错**（cast/round/字节序）—— 需查
2. **KV cache 按 F32 分配但 attention 某处按 BF16 假设**（若仍有残留 BF16 假设）—— 需查
3. **derive_compute_dtype 违宪本身需根治**（不管是否发散根因）—— 用户明确要求代码顺从数据

**候选根因 A（KV cache dtype 双地层裂开）已被运行时证伪**：KV cache 实际全 F32 自洽，无越界。真因在别处，但 derive_compute_dtype 违宪是必须根治的预存问题（C-7 BCE：发现即收集）。

## 根治方案（用户要求：代码顺从数据/配置）

**方案（根治）**：`derive_compute_dtype` 顺从 storage_dtype，不硬编码降级
```rust
pub fn derive_compute_dtype(storage_dtype: DType, device: &DeviceProfile) -> DType {
    // 代码顺从数据：compute_dtype = storage_dtype（BF16 权重就用 BF16 compute）
    // JIT 层 WidenCompute 在 SIMD 指令层 widen BF16→F32 累加（正确路径）
    // 不在 loader 层把权重字节转 F32（宪法1：blob 保留原始 dtype）
    match storage_dtype {
        DType::BF16 | DType::F16 => storage_dtype,  // 顺从, 不降级
        DType::F32 => DType::F32,
        // 量化类型仍 dequant（合法，量化本身需解码）
        DType::U8 | DType::F8E4M3 | ... => DType::F32,
    }
}
```

**配套**：
- `executor_compile.rs:185 needs_dtype_conversion` 变 false（compute_dtype==dtype==BF16）→ 不再 dequantize BF16→F32 → blob 保留 BF16 字节（宪法1恢复）
- KV cache 按 BF16(384/行) 分配 → MemCopy 需 narrow F32→BF16（k_out 是 F32）→ 触发方案 A 的 4 项联动（见 kv-cache-dtype-dual-layer.md）
- attention VecLoad 按 BF16 读 + widen
- GEMM c_dtype=BF16 → needs_narrow=true → 触发 VecNarrow → **必须先修 lane-loss bug**（emit_f32_to_bf16_ymm_to_xmm_avx2 vextracti128 取高半）

**影响面**：所有 BF16 模型（SmolLM2/Llama/Qwen 等）。需全量回归 + 5070Ti 验证。

## 关键代码位置

- `gllm-kernels/src/compiler/dtype_chain.rs:195-210` — derive_compute_dtype（违宪源头，BF16→F32 硬编码）
- `gllm-kernels/src/compiler/graph_geometry.rs:64,127,137` — from_graph 调 derive_compute_dtype
- `gllm/src/engine/mega_kernel/executor_core.inc.rs:371` — MegaKernelCompiled.compute_dtype 来自 GraphDerivedGeometry
- `gllm/src/engine/executor_compile.rs:185-206` — needs_dtype_conversion + dequantize_weight_to_dtype（宪法1违宪执行点）
- `gllm/src/model_config_fragments/types.inc.rs:167` — ModelGeometry.compute_dtype（纸面正确但被 GraphDerivedGeometry 覆盖）
- `gllm/src/loader/safetensors.rs:780` — cast_or_copy_f32（BF16→F32 转换实现）

## 与其他资料库关系

- `kv-cache-dtype-dual-layer.md`：候选根因 A 运行时证伪（KV cache 全 F32 自洽），但方案 A 4 项联动仍需做（derive_compute_dtype 修复后 KV cache 变 BF16）
- `dtype-propagation.md`：WidenCompute 是 JIT 层正确 widen，本库指出 loader 层 dequantize 是违宪（非 WidenCompute）
- `smollm2-135m-architecture.md`：SmolLM2 BF16 权重事实（本库是违宪检测的输入）
- 本文件：derive_compute_dtype 硬编码 BF16→F32 降级违宪铁证
