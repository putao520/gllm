# VAM ActivationPing/Pong + activation_alias 语义资料库（C-9）

> 来源：gllm-kernels VAM + buffer_alloc + context.inc.rs 源码（确定性，非猜）
> 建库触发：8 轮 CPU BUG 诊断反复涉及 ActivationPing/alias，从未确定性文档化
> 最后验证：2026-07-05

## 核心机制（源码确认）

### activation_alias = (layer_input, layer_output)
源码：`gllm-kernels/src/compiler/virtual_activation.rs:149-167`

```
activation_alias = (input_tid, output_tid):
- layer_input: 层循环的输入 tensor (embedding → 第一层 → 第二层 → ...)
- layer_output: 层循环的输出 tensor (最后一层的残差输出)

两 tensor 共享 ping-pong buffer，层循环每轮迭代后 ActivationSwap 交换指针。
其他 "layer." 组的中间 tensor (q_proj, k_proj, attn_out 等) 是层内临时变量，
需通过正常 lifetime coloring 获取独立 buffer slot，不能放入 ping-pong buffer。
```

### SmolLM2 实例（实测 RESOLVER 日志，commit b672c6ed）
```
[VAM] analyze: layer_loop_config=Some((30, 7080192)) activation_alias=Some((TensorId(2), TensorId(26)))
```
- `TensorId(2)` = embedding（Gather 输出，layer_input）
- `TensorId(26)` = layer_output（最后一层残差输出）
- 层循环 30 轮（num_hidden_layers=30），activation_buffer 7080192 bytes

### Ping-Pong buffer（vm_state.rs:350-356）
```
activation_ping_ptr: scratch_base + ping_offset
activation_pong_ptr: scratch_base + pong_offset
ActivationSwap: 每层迭代边界交换这两个指针（零数据拷贝，只换指针）
```

## BCE-20260629-005 → BCE-20260706-ACTSWAP-FIX 根治（关键转折）

**旧 BCE-005（已废止）**：排除 Gather 输出（embedding 走 Intermediate{37748736}），为防 NaN。
**问题**：layer hidden input = embedding = activation_alias.in_tid。走 Intermediate → 不随 ActivationSwap 切换 → layer1+ 读 embedding 非 layer0 输出 → logits 发散。

**根治（BCE-20260706-ACTSWAP-FIX）**：in_tid 强制 ActivationPing（含 gather 输出），三处重复逻辑统一消除：
1. `buffer_alloc.rs build_tensor_sources()` — `map.insert(in_tid, ActivationPing)`（不跳过 gather 输出，根源）
2. `context.inc.rs build()` — 删除运行时 gather 强制 Intermediate（DRY，依赖 tensor_sources）
3. `mod.rs compile_cpu` — 删除 meta 构建时的 gather 强制 Intermediate（DRY）

**根治后**：
- embedding (tid=2) → ActivationPing（off=0, ping buffer）
- gather 写 ping, layer0 读 ping=embedding, ActivationSwap 后 layer1 读 ping=layer0_out
- named_offset("embedding")=0（正确, 旧值 37748736 已废）
- BCE-005 的 NaN 不复现（gather 循环前写 ping, layer0 读到正确 embedding）

## H4 确认（旧日志, 已被根治推翻）

~~[RESOLVER] Gather output tid=2 → Intermediate{offset=37748736}~~
旧 H4 认为"映射无断链"是基于 BCE-005 的 Intermediate 映射。但 ActivationSwap 因此失效（layer1+ 读固定 offset 非 ping/pong）。
根治后: embedding → ActivationPing, ActivationSwap 正确切换, layer1+ 读上一层输出。

## TensorPtrSource 枚举（context.inc.rs:273-289）
```rust
match source {
    ActivationPing => activation_ping_ptr (scratch_base + ping_offset)
    ActivationPong => activation_pong_ptr
    Intermediate { offset } => scratch_base + offset
    Activation => 普通激活 (lifetime coloring)
    Output { offset } => output 区
    Weight { offset } => weight_blob + offset
}
```

## AI 易误判点

| ❌ 误判 | ✅ 正解（源码证明） |
|--------|---------|
| ~~embedding 分配 ActivationPing slot（BCE-005 排除）~~ | **embedding → ActivationPing**（BCE-20260706-ACTSWAP-FIX 根治, in_tid 强制 ping）|
| ~~layer loop 读 Intermediate{37748736}~~ | layer loop 读 ping（off=0），ActivationSwap 切换 layer1+ 读 pong→ping=上一层输出 |
| activation_alias = (任意 tensor) | = (layer_input=embedding, layer_output) 特定 |
| ActivationSwap 拷贝数据 | 只交换指针（零拷贝）|
| gather_outs 排除 input_tid | **已删**（in_tid 强制 ActivationPing, 不跳过 gather 输出）|

## 解决问题时参考

### 诊断 ActivationPing 问题
1. 跑 diag 拿 `[RESOLVER]` + `[VAM]` 日志（代码已埋 eprintln）
2. 看目标 tensor 的 source：ActivationPing / Intermediate{offset}
3. offset=0 且读零 → 落 ping sentinel slot（未写）
4. offset≠0 → 读 Intermediate 区，映射对

### SmolLM2 KV cache 与 ActivationPing 区分
- ActivationPing/Pong: 层间残差流（embedding → layer0 → layer1 → ...）
- kv_cache: 层内 attention K/V 存储（独立 buffer，scratchpad 别名致 logits 污染是 BCE-20260705-GPUPTR-002）
- 两者不同！诊断时别混淆

## 与其他资料库关系
- `smollm2-135m-architecture.md`: SmolLM2 架构事实
- `cuda-driver-api.md`: GPU launch（host/device ptr）
- 本文件: VAM 层间 activation 语义（CPU/GPU 共用）

## 重大陷阱：execute_encode_at_layer 读 offset 0 = ActivationPing 全零（architect sessionId aa9aee8e）

**根因**：execute_encode_at_layer（executor_ops.inc.rs:801-812）读 scratchpad offset 0（ActivationPing），但 layer hidden 写在 ActivationPong（activation_b_offset）。读从未写入的 ping buffer → 全零。

**实测**：diag_step8 encode_to_layer(LastToken) 30 层全 cosine=0.0000（全零返回值）。

**修复尝试失败**（2026-07-06 实测）：读 activation_a_offset(=0)/activation_b_offset(=9437184) 两 buffer 都零。

**更深发现**：layer output 根本不写到 activation buffer。EarlyExit（lower_op.inc.rs:894）`Exit(input_ptr)` 的 input_ptr 是 layer input tensor 指针（如 embedding 在 Intermediate 区 offset=37748736），不是 activation buffer。EarlyExit 语义是"跳转/返回 input_ptr"，不是"写 output 到 activation"。

**根因铁证（2026-07-06 源码确认）**：`GprBranchAction::Exit` 在 x86 lowering **完全未实现**！

lower_instr_dispatch.inc.rs:2901-2903:
```rust
GprCondition::CmpEq(vreg, imm) => {
    ...
    match action {
        GprBranchAction::Exit(_) => {
            return Err("GprCondAction: CmpEq + Exit not yet supported".into());
        }
        ...
    }
}
```

**所有条件 + Exit 都返 Err**（IsNull/BitClear/BitSet/IsNonNull/CmpEq/CmpLtU/CmpGeU 全 "not yet supported"）。

EarlyExit op（lower_op.inc.rs:894）用 `CmpEq(layer_ctr, anchor_layer)` + `Exit(input_ptr)`——这个组合 x86 lowering **报错未实现**。

**推断**：SmolLM2 能编译说明 EarlyExit op 没被插入图（否则编译失败），encode_to_layer 走完整 generate loop（max_new_tokens=1），不 early-exit。layer N 输出从未被捕获到 activation buffer → encode_to_layer 返回全零（读 activation 区零内存）。

**这是真 bug**：EarlyExit CmpEq+Exit 未实现，encode_to_layer 功能损坏。但**非 logits 发散根因**（diagnostic_prefill_logits 不依赖 EarlyExit，走完整 generate loop 写 logits row0）。

**实测数据**（diag_step8 修复后）:
```
[ENCODE-AT-LAYER-DIAG] anchor=0 seq_len=5 hidden=576 ping_off=0 pong_off=9437184 scratchpad.len=249233408 compute_dtype=F32 target=Cpu
ping_off=0, pong_off=9437184 两 buffer 都零 (nonzero=0)
```

**结论**：encode_to_layer 不可用于逐层 bisection（EarlyExit 未实现）。需新诊断路径：
- 实现 CmpEq+Exit 的 x86 lowering（让 EarlyExit 真正 early-exit + 写 output）
- 或用别的方式捕获 layer N 输出（如 hook callback）
- 或用 diagnostic_prefill_scratchpad 在 generate loop 中插桩

**与 logits 发散的关系**：此 bug 是诊断工具 bug（execute_encode_at_layer 读错），**非 logits 发散根因**（diagnostic_prefill_logits 读 logits_scratch_offset 正确，cosine=-0.465 真信号）。但阻断逐层 bisection。
