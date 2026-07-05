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

## BCE-20260629-005 排除 Gather 输出（关键，反复踩坑）

源码：`virtual_activation.rs:166-179` + `buffer_alloc.rs:614-660` + `context.inc.rs:246-257`

**为什么排除 Gather 输出**（BCE-20260629-005 修的 NaN）：
- 不排除：Gather 输出（embedding）被 VAM 当 activation → 分配 ping/pong slot
- → resolver materialize 返 activation_ping_ptr
- → Gather 写 ping buffer（而非 scratchpad intermediate 区）
- → DIAG 读 scratchpad offset 0 读不到 → NaN

**排除后**（当前行为）：
- Gather 输出走 `Intermediate{offset}`（resolver 强制 `context.inc.rs:246-257`）
- 不分配 ping/pong slot
- 写入位置：`alloc.offset_of(out_tid)`（动态查，非硬编码）

## H4 确认：映射层全对（commit b672c6ed 日志）
```
[RESOLVER] Gather output tid=2 → Intermediate{offset=37748736}
```
- embedding (tid=2) → Intermediate{37748736}（非 ActivationPing，offset≠0）
- layer_input 读 Intermediate{37748736} = Gather 写的位置
- **映射无断链**（BCE-20260705-RESIDUAL-STREAM-DISCONNECT 假设证伪）

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
| embedding 分配 ActivationPing slot | BCE-20260629-005 排除，走 Intermediate |
| layer loop 读空 ping buffer | 实际读 Intermediate{37748736}（embedding 写的位置）|
| activation_alias = (任意 tensor) | = (layer_input=embedding, layer_output) 特定 |
| ActivationSwap 拷贝数据 | 只交换指针（零拷贝）|
| gather_outs 排除 input_tid 和 output_tid | 实际只排除 output（input_tid 继承 offset，H4）|

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
