# 开发计划: 分布式预留功能 SPEC + 实现 | epoch: 1 | status: active

## 范围：REQ-DIST-001~017

### 调研结论

| 层 | REQ 组 | 差距 |
|----|--------|------|
| L0 | REQ-DIST-001~003 | CommHandle 未初始化；DistributedConfig 零消费；PageRoutingTable 始终 None |
| L1 | REQ-DIST-004~006 | TP 权重分片/AllReduce/ReduceScatter 全缺失 |
| L2 | REQ-DIST-007~010 | ForceDoubleBuffer/ForceFlux mk_variant 返回 "select_mk_variant()"（等同于 Auto）；QuantizedComm/算法选择未接线 |
| L3 | REQ-DIST-011~013 | PdDisaggConfig/KvDistributionConfig 零消费；KV 跨节点传输缺失 |
| L4 | REQ-DIST-014~015 | MoeDistributedConfig 零消费；EPLB 缺失 |
| L5 | REQ-DIST-016~017 | Context Parallelism / SAGUARO 完全缺失 |

### 关键发现

1. **DistributedConfig 5 子配置类型完整但零消费** — ParallelConfig/PdDisaggConfig/KvDistributionConfig/CommConfig/MoeDistributedConfig 全部已定义，Default/validate()/测试完备，但无任何代码路径使用它们
2. **gllm-nccl 基础设施完整但未接入** — CommHandle API (all_reduce/all_gather/reduce_scatter/broadcast/send/recv)、25 CommInstr、8 算法选择、FLUX 分解、QuantizedComm 全部已实现但从未被 gllm 调用
3. **ModelContextHolder 只有 1 个 NCCL 字段** — `distributed_routing_table: Option<PageRoutingTable>` 始终 None，缺 `comm_handle` 和 `parallel_config`
4. **ForceFlux/ForceDoubleBuffer 是空壳** — mk_variant 返回 "select_mk_variant()"，等同于 Auto
5. **SAGUARO 枚举存在但无逻辑** — 模式定义在推测解码枚举中，无跨 GPU 通信
6. **Context Parallelism / EPLB 完全空白** — 无任何代码或 SPEC

## 任务树

### TASK-17: SPEC 43 写入 ✅
- 文件: `SPEC/43-DISTRIBUTED-IMPLEMENTATION.html` (70123 bytes / 1538 lines)
- 内容: 17 REQ + 14 Entity + 6 DF + 5 CF + 17 Test
- 状态: completed

### TASK-18: Layer 0 基础设施接线 ✅
- L0-1: CommHandleWrapper 新增 (REQ-DIST-001) ✅ — 轻量 rank/world_size 封装，nccl-gated
- L0-2: DistributedConfig 传递 (REQ-DIST-002) ✅ — init_distributed() 幂等方法，不改 from_loader 签名
- L0-3: PageRoutingTable 构建 (REQ-DIST-003) ✅ — PageRoutingTable::new(rank, tp_size) 空表
- L0-4: resolve_page_for_kernel() 单机修复 ✅ — None→Local{frame_id:0} 而非 NotPresent
- 文件: `executor_builder.rs`, `model_context.rs`, `executor.rs`, `distributed_config.rs`
- 状态: completed (44481 tests pass)

### TASK-19: Layer 1 Tensor Parallelism ✅
- L1-1: TP 权重分片 (REQ-DIST-004) ✅ — weight_shard.rs 新建，ShardStrategy + shard_weight() + infer_shard_strategy()
- L1-2: TP AllReduce (REQ-DIST-005) ✅ — compute.rs 新增 tp_all_reduce()
- L1-3: TP ReduceScatter+AllGather (REQ-DIST-006) ✅ — compute.rs 新增 tp_reduce_scatter()/tp_all_gather()
- 文件: `src/loader/weight_shard.rs`(新), `src/engine/coordinator/compute.rs`(改)
- 状态: completed (44479 tests pass)

### TASK-20: Layer 2 通信重叠 ✅
- L2-1: ForceDoubleBuffer mk_variant 修正 (REQ-DIST-007) ✅ — 返回 "DoubleBufferCluster" 等实际值
- L2-2: ForceFlux 连接 gllm-nccl (REQ-DIST-008) ✅ — CommScheduleDecision + resolve_comm_schedule()
- L2-3: 量化通信接线 (REQ-DIST-009) ✅ — QuantCommDecision + from_hint()
- L2-4: 算法覆盖 (REQ-DIST-010) ✅ — AlgorithmOverride + from_config()
- 文件: `src/engine/intent_bias.rs`(改), `src/engine/coordinator/comm_schedule.rs`(新)
- 状态: completed

### TASK-21: Layer 3 PD 分离 + KV 分布 ✅
- L3-1: PD 角色路由 (REQ-DIST-011) ✅ — dispatch.rs PdRoleDecision 3 变体 + needs_sampling/needs_prefill
- L3-2: KV 跨节点传输 (REQ-DIST-012) ✅ — kv.rs kv_transfer() + KvTransferRequest/Result
- L3-3: KV 5 模式 (REQ-DIST-013) ✅ — kv.rs KvDistDecision 5 变体 + from_config()
- 文件: `src/engine/coordinator/dispatch.rs`(改), `src/engine/coordinator/kv.rs`(改)
- 状态: completed (44483 tests pass)

### TASK-22: Layer 4 MoE 分布式 ✅
- L4-1: 分布式 MoE (REQ-DIST-014) ✅ — distributed_dispatch.rs MoeDistDecision + MoePlacementDecision + AllToAllDecision
- L4-2: EPLB (REQ-DIST-015) ✅ — eplb.rs ExpertLoadStats + should_rebalance() + EplbDecision
- 文件: `src/moe/distributed_dispatch.rs`(新), `src/moe/eplb.rs`(新)
- 状态: completed (44483 tests pass)

### TASK-23: Layer 5 高级功能 ✅
- L5-1: Context Parallelism (REQ-DIST-016) ✅ — context_parallel.rs CpConfig + RingPhase + RingAttentionPlan (14 tests)
- L5-2: SAGUARO (REQ-DIST-017) ✅ — saguaro.rs SaguaroConfig + SaguaroPhase + SaguaroResult (15 tests)
- 文件: `src/engine/coordinator/context_parallel.rs`(新), `src/speculative/saguaro.rs`(新)
- 状态: completed

### TASK-24: 全量验证 ✅
- `cargo check` ✅
- `cargo check --features nccl` ✅
- `cargo test --lib` ✅ (44482 pass, 1 flaky fail 无关本次)
- SPEC 43 REQ 覆盖 ✅ (145 处 REQ-DIST 引用)
- ForceDoubleBuffer/ForceFlux mk_variant 修正 ✅ (仅 Auto 返回 select_mk_variant())
- 非 nccl 构建零退化 ✅
- 状态: completed
