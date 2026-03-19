# TASKLIST: REQ-JIT-GRAPH-001~003 实现计划

## 阶段 1：SymDim 基础设施 (REQ-JIT-GRAPH-001)

### T1.1 gllm-kernels: 新增 SymDim + ShapeBinding ✅
- 文件: `gllm-kernels/src/compiler/graph.rs`
- 内容:
  - 新增 `SymDim` 枚举 (`Concrete(usize)` / `Symbolic(String)`)
  - 新增 `ShapeBinding` 结构体 (HashMap<String, usize> + builder API)
  - `TensorMeta.shape` 从 `Vec<usize>` 改为 `Vec<SymDim>`
  - 新增 `TensorMeta::concrete_numel()` / `concrete_bytes()` 辅助方法
  - `CompilerGraph::add_tensor` 接受 `Vec<SymDim>`
  - 新增 `CompilerGraph::add_tensor_concrete(&[usize])` 便利方法
  - `tensor_numel` / `tensor_numel_resolved` 更新为处理 SymDim
  - 修复所有 `from_layer_ir` 内部调用改用 `add_tensor_concrete`

### T1.2 gllm-kernels: codegen/fusion/semantic_dag 适配 ✅
- 文件: `gllm-kernels/src/compiler/codegen/x86_64.rs`, `emitter.rs`, `completeness_test.rs`
- 文件: `gllm-kernels/src/compiler/fusion.rs`, `fusion_rules.rs`, `semantic_dag.rs`
- 文件: `gllm-kernels/src/compiler/buffer_alloc.rs`, `parallel.rs`, `hw_constraints.rs`
- 内容:
  - 所有 `t.shape.iter().product::<usize>() * t.dtype.size_bytes()` → `t.concrete_bytes()`
  - 所有 `t.shape.iter().product::<usize>()` → `t.concrete_numel()`
  - `out_shape.last()` 解包 SymDim → usize
  - 所有测试代码 `add_tensor(vec![usize])` → `add_tensor_concrete(&[usize])`

### T1.3 gllm-kernels: 导出 SymDim + ShapeBinding ✅
- 文件: `gllm-kernels/src/compiler/mod.rs`
- 内容: `pub use graph::{..., SymDim, ShapeBinding}`

### T1.4 gllm: jit_helpers.rs 使用 SymDim::Symbolic ✅
- 文件: `src/compat/jit_helpers.rs`
- 内容:
  - `build_cached_gqa_graph` 中 k_cache/v_cache 的 `total_seq` 维度改为 `SymDim::Symbolic("total_seq")`
  - 注释说明：图编译一次，每步通过 ShapeBinding 传入当前 total_seq

### T1.5 gllm: 其他 compat 文件适配 ✅
- 文件: `src/compat/bert_forward.rs`, `src/compat/decoder_forward.rs`, `src/graph/executor.rs`
- 内容: 所有 `add_tensor(vec![usize])` → `add_tensor_concrete(&[usize])`

### T1.6 验证 ✅
- `cargo check` (gllm-kernels): 通过
- `cargo check` (gllm): 通过
- `cargo test --lib` (gllm-kernels): 853 passed, 0 failed
- `cargo test --lib` (gllm): 161 passed, 0 failed

---

## 阶段 2：GPT-2 路径 JIT 化 (REQ-JIT-GRAPH-002)

### T2.1 gllm-kernels: 新增 GeluNew OpKind 变体
- 文件: `gllm-kernels/src/compiler/graph.rs`
- 内容: `OpKind::GeluNew` — GPT-2 tanh 近似 GELU

### T2.2 gllm-kernels: GeluNew codegen 实现
- 文件: `gllm-kernels/src/compiler/codegen/x86_64.rs`
- 内容: Phase 0 scalar → SymExec → Phase 3 AVX2/AVX-512 codegen

### T2.3 gllm: Gpt2CachedJit 结构体 + 四图编译
- 文件: `src/compat/decoder_forward.rs`
- 内容:
  - `Gpt2CachedJit { ln1_qkv, cached_attn, o_proj, ln2_mlp }`
  - `compile_gpt2_jit()` 编译四个图（含 SymDim::Symbolic("total_seq")）
  - 替换 `gpt2_forward_sequence` 中所有 scalar GEMM

### T2.4 验证
- `grep -rn "for.*hidden_size\|for.*qkv_dim" src/compat/decoder_forward.rs` 返回 0
- `cargo test --lib` 两个 crate 全部通过

---

## 阶段 3：图执行器打通 (REQ-JIT-GRAPH-003)

### T3.1 gllm: FusedGraph 执行器支持完整 decoder forward ✅
- 文件: `src/graph/executor.rs`
- 内容: 支持 KV cache、symbolic shape binding、完整 decoder 前向

### T3.2 gllm: YAML → OnnxGraph → JIT 端到端链路 ✅
- 文件: `src/arch/mod.rs`, `src/graph/executor.rs`
- 内容:
  - `FusedGraphExecutor::from_graph(onnx_graph, seq_len, hidden)` — 优化 + JIT 编译
  - `build_executor_from_yaml(arch_name, config, seq_len, hidden)` — YAML 注册表 → 执行器
  - 3 个新单元测试: `test_fused_graph_executor_from_simple_graph`, `test_weight_binding_resolve`, `test_shape_binding_resolve`

### T3.3 缺口 A-E 全部实现 ✅ (2026-03-19)
- **缺口 A**: `FusedGraphExecutor::run_with_kv_cache()` — 含 KV cache 指针的完整 decoder forward
- **缺口 B**: `WeightBinding.ptr: Option<*const f32>` — 运行时权重指针注入，优先于 `data`
- **缺口 C**: `Executor.graph_executor: Option<FusedGraphExecutor>` — `from_loader` 自动构建；`GeneratorForwardConfig.graph_executor_ptr` 在每次 forward 前注入
- **缺口 D/E**: `decoder_forward()` 优先走图执行器路径，手写路径保留为 fallback
- **新增测试**: `test_fused_executor_with_kv_cache` + `test_weight_binding_ptr_injection`

### T3.4 验证 ✅
- `cargo check`: 通过
- `cargo test --lib`: 177 passed, 0 failed (新增 2 个测试)

---

## 阶段 4：JIT 编译三级缓存 (REQ-JIT-CACHE-001/002/003) ✅

### T4.1 gllm-kernels: CompiledLayer::code_bytes() ✅
- 文件: `src/compiler/executable.rs`
- 内容: 新增 `pub fn code_bytes(&self) -> &[u8]` — 暴露 mmap 机器码字节，供磁盘序列化使用

### T4.2 gllm: src/compat/jit_cache.rs ✅
- 文件: `src/compat/jit_cache.rs` (新增)
- 内容:
  - `ModelArchKey` / `GraphType` / `JitCacheKey` — 复合缓存键
  - `ModelJitCache` — Level 1 模型实例级缓存结构体
  - `LruCache` — 自实现 LRU（HashMap + VecDeque，容量 512，无外部依赖）
  - `GlobalJitCache` — Level 2 进程级单例（`OnceLock<GlobalJitCache>`）
  - `disk_write` / `disk_read` — Level 3 磁盘持久化（magic + version + cpu_fingerprint + code bytes）
  - `global_jit_cache()` — 全局访问入口
  - 6 个单元测试: dedup / LRU eviction / disk round-trip / version mismatch / disk failure silent

### T4.3 gllm: decoder_forward.rs 接入全局缓存 ✅
- 文件: `src/compat/decoder_forward.rs`
- 内容:
  - `compile_gpt2_jit` 调用 → `global_jit_cache().get_or_compile(...)` (4 个图: LnQkv/OProj/LnMlp/FinalLnLmHead)
  - `compile_decode_jit` 调用 → `global_jit_cache().get_or_compile(...)` (2 个图: QRope/Norm2)
  - `kv_proj_decode` 编译 → `global_jit_cache().get_or_compile(...)` (KvProjection)
  - `compile_moe_decode_jit` 调用 → `global_jit_cache().get_or_compile(...)` (3 个图: MoePreAttn/MoeOGemm/MoeNorm2)

### T4.4 gllm: mod.rs 注册模块 ✅
- 文件: `src/compat/mod.rs`
- 内容: `pub(crate) mod jit_cache;`

### T4.5 验证 ✅
- `cargo check`: 通过（0 errors）
- `cargo test --lib`: 175 passed, 0 failed（新增 6 个 jit_cache 测试）
