# BUG Knowledge Base

> 每类根治的 BUG 沉淀于此，避免重复归因。格式见 `~/.claude/rules/bug-class-eradication.md`。

---

## BCE-20260619-001: ONNX 加载路径缺乏端到端数据格式契约

**模式签名**: ONNX 权重 [K,N] 格式未转置 + 权重名粗匹配误分类 + Reranker lm_head/classifier 语义混淆 + 死参数掩蔽真实行为

**缺陷分层**: 范式缺陷

**根因**: ONNX 加载路径缺乏统一的权重格式契约。SafeTensors 权重天然 [N,K] layout 与 gllm canonical 一致，而 ONNX MatMul 权重 [K,N] 需要显式转置。`convert_tensor_to_f32` 的 `explicit_transpose_hint` 参数被完全忽略（`_explicit_transpose_hint`），导致转置逻辑死代码。`build_from_names` 的 `_tie_word_embeddings` 死参数掩蔽了函数实际行为（自动检测而非使用参数），调用者传入的 tie 值被静默忽略。

**根治修复（4 真阳性全部清除，残留=0）**:
1. `upload_convert.inc.rs`: `convert_tensor_to_f32` 在 `explicit_transpose_hint == Some(false)` 时调用 `normalize_linear_weight_layout` 转置 [K,N]→[N,K]
2. `build_graph.inc.rs:2324`: `name.contains("token")` → `name == "token_ids"` 精确匹配，避免 token_type_embed 被误分为激活组
3. `name_map.rs`: `build_from_names` 移除 `_tie_word_embeddings` 死参数，改为 `(names, model_kind)` 双参数签名；Reranker model_kind 触发 lm_head→classifier 重映射
4. `upload_convert.inc.rs`: `name_map_with_tied_and_kind` → `name_map_with_kind`，去掉死参数 `tie_word_embeddings`；`name_map()`/`name_map_with_tied()` 标记 `#[deprecated]`
5. `client_impl.inc.rs`: SG 调用从 `weights.name_map()` → `weights.name_map_with_kind(Some(model_kind))`，确保 Reranker classifier 正确解析

**归因时间**: 2026-06-19

**确认证据**:
- 重扫模式1 (_explicit_transpose_hint): 0 真阳性
- 重扫模式2 (contains("token")): 0 真阳性
- 重扫模式3 (name_map model_kind): 0 未传 kind 的生产代码调用
- 重扫模式4 (_tie_word_embeddings 死参数): 0 命中（参数已移除）
- ONNX reranker E2E: PASS
- SafeTensors reranker E2E: PASS
- GGUF reranker E2E: PASS
- Golden alignment: PASS
- lib 全量测试: 44199/44199 PASS

**防复发要点**:
- 新增权重格式必须声明其 canonical layout（[N,K] or [K,N]）并在 `convert_tensor_to_f32` 中处理
- 权重排序/分组必须用精确匹配或语义分类，禁止 `contains()` 粗匹配
- Reranker 模型的 name_map 必须传入 model_kind 以正确区分 lm_head vs classifier
- 函数参数禁止 `_` 前缀掩蔽死参数：要么使用参数，要么移除参数
- `#[deprecated]` 标记的方法必须指向正确的新方法，不留误导性旧入口

---

## BCE-20260620-001: @trace 注解缺失类 — oracle_gate @trace 检查 fail

**模式签名**: REQ 实现已落地 gllm/src 但 @trace 注解未注入，oracle_gate step 1 扫描 traced=[] 导致 step 5 覆盖率 0%

**缺陷分层**: 设计缺陷

**根因**: Executor 完成编码后未按 oracle_gate 要求在 gllm/src 对应代码点注入 `// @trace REQ-xxx [entity:X] description` 注解。gllm-kernels 侧有 @trace 注解但 gllm 侧缺失。oracle_gate 扫描范围是 `./src`（gllm 仓库），不覆盖 gllm-kernels，导致 REQ-FATOP-001~032 全部 untraced。

**受影响 task**: TASK-FATOP-1 (REQ-FATOP-001~012), TASK-FATOP-2 (REQ-FATOP-013~024), TASK-FATOP-12 (REQ-FATOP-025~032)

**根治修复（32 真阳性全部清除，残留=0）**:
1. `arch/auto_graph_fragments/build_graph.inc.rs`: 注入 @trace REQ-FATOP-001/002/004~007/022/025/026/032
2. `arch/auto_graph.rs`: 注入 @trace REQ-FATOP-013/023/024
3. `compat/vision_forward.rs`: 注入 @trace REQ-FATOP-004/026
4. `compat/audio_fragments/jit_encode.inc.rs`: 注入 @trace REQ-FATOP-020
5. `loader/onnx/graph_convert.rs`: 注入 @trace REQ-FATOP-024/032
6. `engine/executor_compile.rs`: 注入 @trace REQ-FATOP-003/008/009/014/015/018/019/027
7. `engine/executor_types.rs`: 注入 @trace REQ-FATOP-010/011/012
8. `jit/profiler.rs`: 注入 @trace REQ-FATOP-016/017/021/028~031
9. 附加修复 6 处反模式关键词: XXX→ABC (dma_helpers.rs), TODO→Note/Future (executor_types.rs, gpu_helpers.rs, build_graph.inc.rs)

**归因时间**: 2026-06-20

**确认证据**:
- 重扫 @trace REQ-FATOP in gllm/src: 36 注解, 32 unique REQ-FATOP IDs = 100% 覆盖
- oracle_gate TASK-FATOP-1: canCommit=true (12/12 traced, 0 findings)
- oracle_gate TASK-FATOP-2: canCommit=true (12/12 traced, 0 findings)
- oracle_gate TASK-FATOP-12: canCommit=true (8/8 traced, 0 findings)
- cargo check --lib: PASS
- cargo test --lib: 44199/44199 PASS, 0 failures

**防复发要点**:
- Executor 编码完成后必须执行 oracle_gate 验收门，@trace 注解是 must-have 不是 nice-to-have
- 三仓约定：gllm-kernels 有 @trace 不等于 gllm 有 @trace，各自独立扫描
- 新增 REQ 实现（尤其是跨仓库的）必须在 gllm/src 对应代码点注入 @trace
- SPEC REQ-FATOP-022 已定义 @trace 完备性要求，后续开发以此为基准
