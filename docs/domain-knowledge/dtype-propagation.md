# dtype 传播链资料库（C-9，堵 AI 幻觉）

> 来源：gllm-kernels trace.rs + emit_helpers.inc.rs 源码（确定性，非猜）
> 建库触发：8 轮 CPU BUG 诊断反复涉及 dtype 传播（BF16 权重 / F32 激活 / widen / narrow），从未完整文档化
> 最后验证：2026-07-05

## 核心机制（源码确认）

### X86ElemStrategy（trace.rs:1016-1022, 1130-1146）
```rust
pub enum X86ElemStrategy {
    Native,                              // F32/TF32 原生 SIMD
    WidenCompute,                        // BF16/F16 widen 到 F32 计算
    DequantCompute(DequantMethod),       // INT8 VNNI / GGML ScalarLUT / MXBlock BlockScale ...
}

pub fn x86_elem_strategy(&self) -> X86ElemStrategy {
    match self.kind {
        F32 | TF32 => Native,
        BF16 | F16 => WidenCompute,      // ← SmolLM2 BF16 权重走这
        INT8 => DequantCompute(VNNI),
        INT4/INT2/INT1/FP8/FP6/FP4 => DequantCompute(ScalarLUT),
    }
}
```

**关键**：BF16 在 x86 上**总是** WidenCompute（无论有无 AVX-512 BF16）。无 AVX-512 BF16 时软件 widen，有时也用硬件 vcvtneps2bf16（但 emit 路径仍 widen 到 F32 累加）。

### BF16 → F32 widen（零扩展保符号）
BF16 = F32 的高 16 位。widen：取 BF16 16 位 << 16 拼成 F32（零扩展低 16 位）。
- AVX2 软件路径：`vpmovzxwd`（16→32 零扩展）+ `vpslld 16`（左移16位到F32高位）
- 这**保留符号**（BF16 的符号位 → F32 符号位），指数位对齐

### F32 → BF16 narrow（emit_helpers.inc.rs）
F32 累加后 → BF16 store。取 F32 高 16 位 + round。
- AVX-512：`vcvtneps2bf16`（原生）
- AVX2 软件：`vpackusdw` + 预移位（**lane-loss bug** — BCE-20260705 待修，漏 vextracti128 取高半，但给 +0.5 非 -0.5）

## 完整 dtype 传播链（混合精度，SmolLM2）

```
权重文件 BF16 (config.json torch_dtype=bfloat16)
  → loader 保留原始 BF16 (ARCH-BLOB-YIELDS-WEIGHT, 禁止 BF16→F32 转换)
    → weight_blob raw-pack (原始字节)
      → weight_dtypes map: name → DType::BF16 (executor_compile.rs build_weight_dtype_maps)
        → build_graph: tdt("embed") = weight_dtypes["embed"] = BF16 → graph tensor "embed" dtype=BF16
          → op_input_dtype(op, graph): 从 op.inputs[i].dtype 推断每个输入 (plan_lower.rs)
            → auto_lower_trace(prog, body, inputs, width, dtype): dtype 参数 = 输入 dtype
              → VmInstr { ..., dtype } (instr.rs, 携带 dtype)
                → dtype.x86_elem_strategy() → WidenCompute (BF16)
                  → ISA lowering: BF16 VecLoad → widen F32 / F32 VecLoad → Native
                    → FMA 累加 F32 (accumulator_dtype)
                      → needs_narrowing_from(F32): 若 output dtype=BF16 → VecNarrow+VecStore
```

## D0 实测确认（commit b672c6ed）
```
weight_dtypes["embed"] = BF16 ✓
weight_dtypes["lm_head"] = BF16 ✓
embed weight graph dtype = BF16 ✓
act_dt = F32 ✓ (激活 F32, 混合精度 A=F32 + B=BF16 正确)
```
**dtype 传播链无退化**（D0 证伪"embed dtype 静默退化 F32"假设）。

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| BF16 权重被转 F32 | blob 保留 BF16，widen 在 SIMD 指令层（非存储层）|
| BF16 用 Native 指令（无 AVX-512 BF16 时）| 总是 WidenCompute（即使有 AVX-512 BF16）|
| BF16→F32 widen 丢符号 | vpmovzxwd+vpslld 零扩展保符号 |
| dtype 从激活推断权重 | 各输入独立推断（op_input_dtype 从 op.inputs[i].dtype）|
| elem_bytes 硬编码 4（F32）| 必须 dtype.size_bytes()（BF16=2）|
| single dtype 覆盖 A/B/C | 多路 dtype 传播（a_dtype / weight_dtype / acc_dtype / c_dtype 独立）|

## 关键代码位置
- `trace.rs:1016-1146`: X86ElemStrategy + x86_elem_strategy()
- `trace.rs:1076-1090`: elem_bytes()
- `emit_helpers.inc.rs`: BF16↔F32 widen/narrow 指令
- `plan_lower.rs op_input_dtype`: 每输入 dtype 独立推断
- `executor_compile.rs build_weight_dtype_maps:543-562`: weight_dtypes 从 safetensors meta 构建
- `build_graph.inc.rs:191`: tdt("embed") = weight_dtypes["embed"]

## 与 BUG 诊断的关系
- 块1 cosine=-0.465：**不是** dtype 退化（D0 证伪）
- embedding cosine=0.13/0.67：dtype 对（BF16 权重 + F32 激活），根因在别处（Gather seq 维 / decode M=1，待 Explore 确定）
- lane-loss bug（emit_helpers narrow）：真 bug 但给 +0.5 非 -0.5，独立 BCE

## 与其他资料库关系
- `smollm2-135m-architecture.md`: SmolLM2 BF16 权重事实
- 本文件: BF16 怎么在 x86 JIT 传播（widen/narrow）
