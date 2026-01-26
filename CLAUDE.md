# gllm

Pure Rust LLM inference library with embedding, reranking, and text generation.

## SPEC 位置

- `./SPEC/`

## 依赖项目

- gllm-kernels: `/home/putao/code/rust/gllm-kernels`
  - 必须遵守 gllm-kernels 的所有 FROZEN 约束
  - 参见 gllm-kernels/CLAUDE.md

## 核心架构约束（🚨 FROZEN - 铁律）

### 纯 GPU 数据流原则（ARCH-GPU-001 🚨 最高优先级）

> 📌 SSOT: 详见 `SPEC/02-ARCHITECTURE.md` ARCH-ADR-009

**所有热路径推理必须保持纯 GPU 数据流**：

| 约束ID | 约束内容 | 违规示例 |
|--------|----------|----------|
| ARCH-GPU-001-A | 热路径 API 必须接受 GpuTensor | 接受 `&[u32]`/`&[f32]` host slices |
| ARCH-GPU-001-B | 管线内禁止 GPU→CPU→GPU 往返 | routing readback 后 re-upload |
| ARCH-GPU-001-C | 只在最终输出时 readback | 每层都 readback hidden states |

**正确的数据流**：

```
输入 → upload一次 → [Layer1 GPU] → [Layer2 GPU] → ... → [LayerN GPU] → readback一次 → 输出
                    ↑                                                    ↑
                    └───────────── 全程 GPU 显存 ─────────────────────────┘

❌ 禁止：
输入 → upload → [GPU] → readback → [CPU处理] → upload → [GPU] → readback → 输出
```

### 类型安全 Tensor 操作（ARCH-TYPE-001 🚨 FROZEN）

**整数类型 tensor 必须使用类型安全的 readback 方法**：

| Tensor 类型 | 正确方法 | 禁止的 hack |
|-------------|----------|-------------|
| U32 | `backend.readback_u32()` | ❌ f32 读取 + `to_bits()` |
| I32 | `backend.readback_i32()` | ❌ f32 读取 + `to_bits() as i32` |
| U64 | `backend.readback_u64()` | ❌ 两次 f32 读取拼接 |

**禁止的代码模式**：

```rust
// ❌ 严重违规：类型不安全的 hack
let mut bits = vec![0.0f32; count];
backend.readback(&gpu_tensor, &mut bits)?;  // U32 用 f32 读！
let indices: Vec<u32> = bits.iter().map(|v| v.to_bits()).collect();

// ✅ 正确：类型安全
let mut indices = vec![0u32; count];
backend.readback_u32(&gpu_tensor, &mut indices)?;
```

---

## 已完成的 GPU Pure 优化

| 组件 | 状态 | 说明 |
|------|------|------|
| `moe_forward_gpu_pure` | ✅ 完成 | MoE 管线完全 GPU 化，routing→forward 无 readback |
| `readback_u32` | ✅ 完成 | 类型安全的 U32 tensor readback |
| `readback_i32` | ✅ 完成 | 类型安全的 I32 tensor readback |
| `readback_u64` | ✅ 完成 | 类型安全的 U64 tensor readback |
| `paged_attention_gpu_pure` | ✅ 完成 | P0 优先级，接受 GpuTensor 的 page_table/block_offsets |
| `flash_tree_attention_gpu_pure` | ✅ 完成 | P0 优先级，tree_mask 使用 I32 GpuTensor |

## 需要优化的 API 清单（ARCH-OPTIMIZE-001）

以下 gllm-kernels API 仍使用 host slices，需要按 ARCH-ADR-009 模式重构：

| 优先级 | API | Host Slices 参数 | 状态 |
|--------|-----|------------------|------|
| P0 | `paged_attention` | `page_table: &[u32]`, `seq_lens: &[u32]` | ✅ `paged_attention_gpu_pure` 已添加 |
| P0 | `flash_tree_attention` | `tree_mask: &[i32]` | ✅ `flash_tree_attention_gpu_pure` 已添加 |
| P1 | `medusa_verify` | `candidate_tokens: &[i32]` | 🚧 待重构 |
| P1 | `evict_press_evict` | `token_ages: &[i32]`, `current_zones: &[i32]` | 🚧 待重构 |
| P2 | `prompt_cache_lookup` | `tokens: &[i32]`, `cache_hashes: &[u64]`, `cache_lengths: &[u32]` | 🚧 待重构 |
| P2 | `rerank_pipeline` | 多个 `&[u32]` 参数 | 🚧 待重构 |
| P3 | `binary_ip_*` | `queries`, `database` | 🚧 待重构 |

**重构模式**（参考 MoE/PagedAttention 实现）：

1. 添加 `*_gpu_pure` 版本接受 `&GpuTensor`
2. 添加对应的类型安全 readback（如需 `readback_i32`/`readback_u64`）
3. 保留旧 API 用于需要 host 控制的场景
4. 新代码优先使用 `*_gpu_pure` 版本

**参考实现**（已验证模式）：
- `backend.rs:L3569-3648` - `moe_forward_gpu_pure` 完整实现
- `backend.rs:L1835+` - `paged_attention_gpu_pure` WgpuBackend 实现

---

---

## 待实现需求（性能优化）

| 需求 ID | 描述 | 架构设计 | 状态 |
|---------|------|----------|------|
| REQ-LOAD-001 | 异步并行模型加载 | ARCH-ADR-010 | ✅ 架构设计完成，🔲 待实现 |
| REQ-QUANT-001 | 原生量化推理 Kernel | ARCH-ADR-011 | ✅ 架构设计完成，🔲 待实现 |

**依赖关系**：
- REQ-QUANT-001 依赖 gllm-kernels 的量化 kernel 实现（Backend trait 扩展 q4/q8/awq_matmul）
- REQ-LOAD-001 独立，可并行开发

**实现优先级**：
1. REQ-LOAD-001（异步并行加载）- 独立，无外部依赖
2. REQ-QUANT-001（原生量化 Kernel）- 依赖 gllm-kernels Backend trait 扩展

---

## 常用命令

```bash
cargo check                    # 语法检查
cargo test                     # 运行测试
cargo test --features quantized # 量化模型测试
cargo bench                    # 性能基准
```

## 模型目录

```
~/.gllm/models/                # 模型缓存
```
