# 开发计划: 补齐 gllm 分布式推理所有缺失功能：PP 微批次交错调度 SPEC+代码、REQ-DIST 全状态升级

## epoch
1

## status
in_progress

## reqLedger
REQ-DIST-001, REQ-DIST-002, REQ-DIST-003, REQ-DIST-004, REQ-DIST-005,
REQ-DIST-006, REQ-DIST-007, REQ-DIST-008, REQ-DIST-009, REQ-DIST-010,
REQ-DIST-011, REQ-DIST-012, REQ-DIST-013, REQ-DIST-014, REQ-DIST-015,
REQ-DIST-016, REQ-DIST-017, REQ-DIST-018, REQ-DIST-019, REQ-DIST-020,
REQ-DIST-021, REQ-DIST-022, REQ-DIST-023, REQ-DIST-024, REQ-DIST-025,
REQ-DIST-026, REQ-DIST-027, REQ-DIST-028, REQ-DIST-029, REQ-DIST-030,
REQ-DIST-031, REQ-DIST-032, REQ-DIST-033, REQ-DIST-034

## 范围

REQ-DIST-001~034 | ENT-DIST-COMMHANDLE / ENT-DIST-ROUTING-TABLE / ENT-DIST-TP-SHARD / ENT-DIST-TP-COMM / ENT-DIST-DOUBLE-BUFFER / ENT-DIST-FLUX / ENT-DIST-QUANT-COMM / ENT-DIST-PD-ROLE / ENT-DIST-KV-XFER / ENT-DIST-KV-MODE / ENT-DIST-MOE-DIST / ENT-DIST-EPLB / ENT-DIST-CP / ENT-DIST-SAGUARO / ENT-DIST-PP（新增）| TEST-DIST-001~034 | CF-DIST-001~005 / CF-DIST-006~010（新增）| DF-DIST-001~006 / DF-DIST-007~009（新增）

## 影响矩阵

| SPEC ID | 关联 TASK | 文件 |
|---------|-----------|------|
| REQ-DIST-001 | TASK-03 | src/engine/distributed_config.rs, src/engine/executor_builder.rs |
| REQ-DIST-002 | TASK-04 | src/engine/distributed_config.rs, src/client_fragments/builder.inc.rs, src/engine/executor.rs |
| REQ-DIST-003 | TASK-05 | src/engine/executor_builder.rs, src/engine/coordinator/model_context.rs, src/scheduler/paged_scheduler.rs |
| REQ-DIST-004 | TASK-06 | src/loader/weight_shard.rs, src/loader/mod.rs |
| REQ-DIST-005 | TASK-06 | src/engine/coordinator/comm_schedule.rs, src/engine/coordinator/compute.rs |
| REQ-DIST-006 | TASK-07 | src/engine/coordinator/comm_schedule.rs, src/engine/coordinator/compute.rs |
| REQ-DIST-007 | TASK-08 | src/engine/coordinator/comm_schedule.rs, src/engine/executor_step.rs |
| REQ-DIST-008 | TASK-08 | src/engine/coordinator/comm_schedule.rs, src/engine/executor_step.rs |
| REQ-DIST-009 | TASK-09 | src/engine/coordinator/comm_schedule.rs |
| REQ-DIST-010 | TASK-09 | src/engine/coordinator/comm_schedule.rs |
| REQ-DIST-011 | TASK-10 | src/engine/coordinator/dispatch.rs, src/engine/executor_step.rs |
| REQ-DIST-012 | TASK-11 | src/engine/coordinator/kv.rs, src/engine/executor_step.rs |
| REQ-DIST-013 | TASK-11 | src/engine/coordinator/kv.rs |
| REQ-DIST-014 | TASK-12 | src/moe/distributed_dispatch.rs, src/engine/executor_step.rs |
| REQ-DIST-015 | TASK-12 | src/moe/eplb.rs, src/moe/distributed_dispatch.rs |
| REQ-DIST-016 | TASK-13 | src/engine/coordinator/context_parallel.rs, src/engine/executor_step.rs |
| REQ-DIST-017 | TASK-14 | src/speculative/saguaro.rs, src/speculative/engine.rs |
| REQ-DIST-018~020 | TASK-15 | src/engine/pipeline/ (新建), src/loader/weight_shard.rs |
| REQ-DIST-021~023 | TASK-16 | src/engine/pipeline/micro_batch.rs (新建), src/engine/pipeline/activation_xfer.rs (新建) |
| REQ-DIST-024~027 | TASK-17 | src/engine/pipeline/scheduler.rs (新建), src/engine/pipeline/interleaved.rs (新建) |
| REQ-DIST-028~031 | TASK-18 | src/engine/pipeline/hybrid.rs (新建), src/engine/coordinator/kv.rs, src/engine/coordinator/dispatch.rs |
| REQ-DIST-032~034 | TASK-19 | src/engine/pipeline/comm.rs (新建), src/engine/pipeline/observability.rs (新建) |
| TEST-DIST-001~017 | TASK-20 | tests/dist/ (新建) |
| TEST-DIST-018~034 | TASK-21 | tests/dist/pipeline/ (新建) |

---

## 任务树（扁平列表）

### TASK-01: SPEC 修复 — 补全 REQ-DIST-001~017 description 字段，状态升至 approved
- SPEC: REQ-DIST-001~017 | 文件: SPEC/43-DISTRIBUTED-IMPLEMENTATION.html
- 实现: spec_write(crudAction="update") 为每个 REQ 补充 description（当前为空）；spec_govern(auditAction="status", status="approved")
- 复用锚点: spec:REQ-DIST-001~017
- 依赖: 无
- 状态: pending

### TASK-02: SPEC 扩展 — 新增 REQ-DIST-018~034（PP 微批次交错调度层 17 个 REQ）
- SPEC: REQ-DIST-018~034 | 文件: SPEC/43-DISTRIBUTED-IMPLEMENTATION.html
- 实现:
  - REQ-DIST-018: PipelineConfig（pp_size, stage_id, num_virtual_stages）数据结构
  - REQ-DIST-019: PP stage 权重分区加载（按 stage_id 只加载对应层权重）
  - REQ-DIST-020: MicroBatchScheduler（将 batch 切成 num_microbatches 个 chunk）
  - REQ-DIST-021: 1F1B 流水线调度（warmup→稳定态→cooldown）
  - REQ-DIST-022: 交错 1F1B 虚拟 stage（virtual_stage 数 = num_virtual_stages）
  - REQ-DIST-023: Stage 间激活传输（send_activation / recv_activation via p2p）
  - REQ-DIST-024: PP+TP 混合并行（pp_size×tp_size rank 映射，group 划分）
  - REQ-DIST-025: PP 感知 KV Cache 分配（仅 last stage 分配 KV；中间 stage 分配 activation buffer）
  - REQ-DIST-026: PP zero-bubble 变体（prefill-decode micro-batch 交错填充 bubble）
  - REQ-DIST-027: PP warmup/cooldown 调度
  - REQ-DIST-028: PP 与 PD 分离集成（prefill stage → Prefill 节点，decode stage → Decode 节点）
  - REQ-DIST-029: PP activation ring buffer（容量 = num_microbatches）
  - REQ-DIST-030: PP 动态 micro-batch 数量（min(pp_size*2, batch_tokens/min_micro_tokens)）
  - REQ-DIST-031: PP 跨 stage 通信重叠（overlap activation xfer 与下一 micro-batch compute）
  - REQ-DIST-032: PP fault tolerance（StageFailed → 终止 in-flight micro-batches）
  - REQ-DIST-033: PP observability（bubble_ratio, activation_xfer_latency_ms, stage_throughput）
  - REQ-DIST-034: PP 测试 fixture（MockPipelineEnv，loopback send/recv）
- 复用锚点: spec:REQ-DIST-001, code:CommHandleWrapper, code:ParallelConfig
- 依赖: TASK-01
- 状态: pending

### TASK-03: REQ-DIST-001 — CommHandle 完整生命周期接线至 Executor
- SPEC: REQ-DIST-001 [验收: CommHandleWrapper 在 executor build 时 init_nccl; Drop 时释放] | TDD: TEST-DIST-001 | 文件: src/engine/executor_builder.rs, src/engine/executor.rs, src/engine/distributed_config.rs
- 实现: executor_builder 检测 parallel.is_distributed() → init_nccl()；executor Drop 显式 cleanup；错误传播到 ExecutorBuildError
- 复用锚点: code:CommHandleWrapper::from_config, code:ParallelConfig::validate
- 依赖: TASK-01
- 状态: pending

### TASK-04: REQ-DIST-002 — DistributedConfig 完整传递链 Client → Executor
- SPEC: REQ-DIST-002 [验收: ClientConfig.distributed 全链路传播到 ExecutorConfig] | TDD: TEST-DIST-002 | 文件: src/client_fragments/builder.inc.rs, src/engine/executor_builder.rs, src/engine/executor.rs
- 实现: ClientConfig.distributed → ExecutorConfig.distributed；executor_builder 读取 distributed 字段配置 CommHandleWrapper + KvDistributionConfig；executor_step 持有 comm_handle 引用
- 复用锚点: code:DistributedConfig, code:builder.inc.rs distributed() method
- 依赖: TASK-03
- 状态: pending

### TASK-05: REQ-DIST-003 — 分布式页路由表完整构建与调度器接线
- SPEC: REQ-DIST-003 [验收: PageRoutingTable 注入 PagedScheduler; 跨 rank KV 查询命中正确 rank] | TDD: TEST-DIST-003 | 文件: src/engine/executor_builder.rs, src/engine/coordinator/model_context.rs, src/scheduler/paged_scheduler.rs
- 实现: build_distributed_routing_table 返回值注入 paged_scheduler.set_routing_table(); scheduler 分配页按 rank 映射；kv_coordinator 跨节点路由查询走 routing_table lookup
- 复用锚点: code:build_distributed_routing_table, code:gllm_kernels::PageRoutingTable
- 依赖: TASK-04
- 状态: pending

### TASK-06: REQ-DIST-004/005 — TP 权重分片接线 + AllReduce JIT stub 集成
- SPEC: REQ-DIST-004 [验收: QKV/O 按 head 维度切，FFN Gate/Up 按行切，lm_head 按 vocab 切] REQ-DIST-005 [验收: CompilerGraph 含 AllReduce VmInstr stub；executor step 触发 all_reduce_inplace] | TDD: TEST-DIST-004, TEST-DIST-005 | 文件: src/loader/weight_shard.rs, src/loader/mod.rs, src/engine/coordinator/compute.rs, src/engine/coordinator/comm_schedule.rs
- 实现: loader 按 infer_shard_strategy + shard_weight 加载分片权重；compute coordinator step 后注入 CommScheduleDecision::AllReduce 调用 comm_handle.all_reduce_inplace；JIT AllReduce 走 gllm-nccl VmInstr call stub
- 复用锚点: code:shard_weight, code:infer_shard_strategy, code:CommScheduleDecision, code:all_reduce_inplace
- 依赖: TASK-04
- 状态: pending

### TASK-07: REQ-DIST-006 — TP 列并行通信（ReduceScatter+AllGather）
- SPEC: REQ-DIST-006 [验收: DownProj 后 ReduceScatter；AllGather 前置到下一层 QKV] | TDD: TEST-DIST-006 | 文件: src/engine/coordinator/comm_schedule.rs, src/engine/coordinator/compute.rs
- 实现: CommScheduleDecision::ColumnParallel 分支；compute coordinator step 在 down_proj 后调 reduce_scatter_inplace，在 up_proj 前调 all_gather_inplace
- 复用锚点: code:reduce_scatter_inplace, code:all_gather_inplace, code:CommScheduleDecision
- 依赖: TASK-06
- 状态: pending

### TASK-08: REQ-DIST-007/008 — 双缓冲重叠 + FLUX 分解调度接线
- SPEC: REQ-DIST-007 [验收: 通信 stream 与计算 stream 并行；bubble < 5%] REQ-DIST-008 [验收: FLUX flux_decompose 调用成功] | TDD: TEST-DIST-007, TEST-DIST-008 | 文件: src/engine/coordinator/comm_schedule.rs, src/engine/executor_step.rs
- 实现: CommScheduleDecision::build_comm_plan 返回 DoubleBuffer 或 FluxDecompose；executor_step 按方案在 CUDA multi-stream 异步通信；FLUX 调用 gllm_nccl::flux_decompose
- 复用锚点: code:CommScheduleDecision::build_comm_plan, code:resolve_comm_schedule
- 依赖: TASK-07
- 状态: pending

### TASK-09: REQ-DIST-009/010 — 量化通信 + 算法选择接线
- SPEC: REQ-DIST-009 [验收: CommCompressHint 驱动 QuantizedComm；FP8 带宽节省 ≥ 50%] REQ-DIST-010 [验收: AlgorithmOverride 覆盖自动算法选择] | TDD: TEST-DIST-009, TEST-DIST-010 | 文件: src/engine/coordinator/comm_schedule.rs
- 实现: QuantCommDecision::from_hint 接入 comm_handle 传输前量化压缩；AlgorithmOverride::select_or_auto 接入 CommHandle.set_algorithm；executor_step 读取 comm_config 应用策略
- 复用锚点: code:QuantCommDecision, code:AlgorithmOverride, code:build_quantized_comm
- 依赖: TASK-08
- 状态: pending

### TASK-10: REQ-DIST-011 — PD 分离角色路由完整实现
- SPEC: REQ-DIST-011 [验收: PdRoleDecision 驱动 executor step 裁剪；PrefillOnly 跳过 decode ops] | TDD: TEST-DIST-011 | 文件: src/engine/coordinator/dispatch.rs, src/engine/executor_step.rs
- 实现: executor_step 检查 needs_prefill/needs_decode 控制 graph portion 执行；PrefillOnly 角色执行后触发 KV 传输到 Decode 节点
- 复用锚点: code:PdRoleDecision, code:PdDisaggConfig
- 依赖: TASK-05
- 状态: pending

### TASK-11: REQ-DIST-012/013 — 跨节点 KV 传输 + 5 模式 KV 分布
- SPEC: REQ-DIST-012 [验收: send_kv_pages/recv_kv_pages 全链路通；Decode 节点收到后可立即使用] REQ-DIST-013 [验收: Local/Remote/Replicated/Sharded/Tiered 5 种模式按配置切换] | TDD: TEST-DIST-012, TEST-DIST-013 | 文件: src/engine/coordinator/kv.rs, src/engine/executor_step.rs
- 实现: KvCoordinator::kv_transfer 接入 executor step PD 切换点；KvDistDecision::from_config 5 分支实现；Tiered 对接 three_tier_swap
- 复用锚点: code:kv_transfer, code:KvDistDecision, code:send_kv_pages, code:recv_kv_pages
- 依赖: TASK-10
- 状态: pending

### TASK-12: REQ-DIST-014/015 — 分布式 MoE AllToAll + EPLB 接线
- SPEC: REQ-DIST-014 [验收: Expert 分布 3 策略按配置切换；AllToAll dispatch 路由正确] REQ-DIST-015 [验收: EPLB 运行时重均衡；热 expert 迁移到多 GPU] | TDD: TEST-DIST-014, TEST-DIST-015 | 文件: src/moe/distributed_dispatch.rs, src/moe/eplb.rs, src/engine/executor_step.rs
- 实现: MoeDistDecision::dispatch_experts + all_to_all_exchange_counts 接入 moe coordinator；eplb 热度统计接入 expert dispatch result；update_hot_cold_counts 驱动 EPLB 重均衡
- 复用锚点: code:MoeDistDecision, code:dispatch_experts, code:all_to_all_exchange_counts
- 依赖: TASK-06
- 状态: pending

### TASK-13: REQ-DIST-016 — Ring Attention CP 完整链路
- SPEC: REQ-DIST-016 [验收: execute_ring_step 全 cp_size 轮；输出数值与单卡 FlashAttention 对齐 atol=1e-3] | TDD: TEST-DIST-016 | 文件: src/engine/coordinator/context_parallel.rs, src/engine/executor_step.rs
- 实现: RingAttentionPlan::phases_for_step + execute_ring_step 接入 executor step attention 阶段；send/recv KV chunks 走 CommHandleWrapper p2p；kv_source_rank 映射正确
- 复用锚点: code:RingAttentionPlan, code:execute_ring_step, code:CpConfig
- 依赖: TASK-05
- 状态: pending

### TASK-14: REQ-DIST-017 — SAGUARO 分布式推测解码接线
- SPEC: REQ-DIST-017 [验收: Draft GPU 发 draft_tokens；Verify GPU 接收验证；acceptance_rate 正确] | TDD: TEST-DIST-017 | 文件: src/speculative/saguaro.rs, src/speculative/engine.rs
- 实现: SaguaroConfig 接入 speculative engine；is_draft_gpu/is_verify_gpu 路由执行路径；transfer_draft_tokens/receive_verify_result 全链路测试
- 复用锚点: code:SaguaroConfig, code:transfer_draft_tokens, code:receive_verify_result
- 依赖: TASK-04
- 状态: pending

### TASK-15: REQ-DIST-018/019/029 — PP 基础设施（PipelineConfig + stage 权重加载 + activation buffer）
- SPEC: REQ-DIST-018, REQ-DIST-019, REQ-DIST-029 | TDD: TEST-DIST-018 | 文件: src/engine/pipeline/mod.rs（新建）, src/engine/pipeline/config.rs（新建）, src/loader/weight_shard.rs
- 实现:
  - PipelineConfig { pp_size: u32, stage_id: u32, num_virtual_stages: u32, micro_batch_size: usize }
  - ParallelConfig 扩展 pp_size 字段
  - weight loader: 按 stage_id 计算层范围 [stage_id*layers_per_stage..(stage_id+1)*layers_per_stage)，仅加载该范围层权重
  - ActivationRingBuffer { capacity: usize, buffers: Vec<Vec<f32>> }，capacity = num_microbatches
- 复用锚点: code:ParallelConfig, code:shard_weight, code:ShardStrategy
- 依赖: TASK-02
- 状态: pending

### TASK-16: REQ-DIST-020/023/030 — MicroBatchScheduler + stage 间激活传输 + 动态 chunk 分配
- SPEC: REQ-DIST-020, REQ-DIST-023, REQ-DIST-030 | TDD: TEST-DIST-019 | 文件: src/engine/pipeline/micro_batch.rs（新建）, src/engine/pipeline/activation_xfer.rs（新建）
- 实现:
  - MicroBatchScheduler::split(batch, micro_batch_size) → Vec<MicroBatch>
  - send_activation(comm: &CommHandleWrapper, next_rank: u32, data: &[f32]) → comm.send_f32
  - recv_activation(comm: &CommHandleWrapper, prev_rank: u32, count: usize) → comm.recv_f32
  - dynamic_num_microbatches(pp_size, batch_tokens, min_micro_tokens) → u32
- 复用锚点: code:CommHandleWrapper::send_f32, code:CommHandleWrapper::recv_f32
- 依赖: TASK-15
- 状态: pending

### TASK-17: REQ-DIST-021/022/026/027 — 1F1B + 交错 1F1B + zero-bubble + warmup/cooldown
- SPEC: REQ-DIST-021, REQ-DIST-022, REQ-DIST-026, REQ-DIST-027 | TDD: TEST-DIST-020 | 文件: src/engine/pipeline/scheduler.rs（新建）, src/engine/pipeline/interleaved.rs（新建）
- 实现:
  - PipelineScheduler::schedule_1f1b(stage_id, num_microbatches) → Vec<PipelineOp>（Forward/Recv/Send/Bubble）
  - schedule_interleaved(stage_id, num_virtual_stages, num_microbatches) → Vec<PipelineOp>
  - warmup_steps = pp_size - stage_id - 1；cooldown_steps = stage_id
  - zero_bubble_variant: prefill micro-batch 与 decode micro-batch 交错填充 bubble slot
- 复用锚点: code:MicroBatchScheduler, code:send_activation, code:recv_activation
- 依赖: TASK-16
- 状态: pending

### TASK-18: REQ-DIST-024/025/028/031 — PP+TP 混合 + PP 感知 KV + PP+PD 集成 + comm 重叠
- SPEC: REQ-DIST-024, REQ-DIST-025, REQ-DIST-028, REQ-DIST-031 | TDD: TEST-DIST-021 | 文件: src/engine/pipeline/hybrid.rs（新建）, src/engine/coordinator/kv.rs, src/engine/coordinator/dispatch.rs
- 实现:
  - rank_to_tp_group(rank, pp_size, tp_size) → tp_rank：混合并行 group 划分
  - is_last_pp_stage(stage_id, pp_size) → bool：仅最后 stage 分配 KV pages
  - PD 分离时：stage 0..N-1 在 Prefill 节点，decode stage 在 Decode 节点
  - activation xfer overlap: 发送 activation 时异步启动下一 micro-batch compute
- 复用锚点: code:KvDistDecision, code:PdRoleDecision, code:PipelineConfig
- 依赖: TASK-17, TASK-11, TASK-10
- 状态: pending

### TASK-19: REQ-DIST-032/033/034 — PP fault tolerance + observability + test fixture
- SPEC: REQ-DIST-032, REQ-DIST-033, REQ-DIST-034 | TDD: TEST-DIST-022 | 文件: src/engine/pipeline/observability.rs（新建）, src/engine/pipeline/mod.rs
- 实现:
  - PipelineError::StageFailed { stage_id } → 终止 in-flight micro-batches，返回 Err 到 executor
  - PipelineMetrics { bubble_ratio: f32, activation_xfer_latency_ms: f64, stage_throughput_tokens_per_sec: f64 }
  - PipelineMetrics::record_step 接入 executor observability 遥测 hook
  - MockPipelineEnv: loopback CommHandleWrapper（CommHandleWrapper::new_for_test）+ PipelineConfig
- 复用锚点: code:CommHandleWrapper::new_for_test, code:PipelineScheduler
- 依赖: TASK-18
- 状态: pending

### TASK-20: TEST-DIST-001~017 — 现有功能单测实现 ✅ COMPLETED
- SPEC: TEST-DIST-001~017 | 文件: tests/dist/mod.rs（新建）
- 实现: 每个 REQ 对应一个 #[test]；使用 CommHandleWrapper::new_for_test(rank=0, world_size=1) 作 mock；TEST-DIST-003 验证 PageRoutingTable rank 映射；TEST-DIST-004 验证 shard_weight 切分维度；TEST-DIST-016 验证 ring_step 数值（loopback）；TEST-DIST-017 验证 SAGUARO transfer round-trip
- 复用锚点: code:CommHandleWrapper::new_for_test, code:shard_weight
- 依赖: TASK-03, TASK-04, TASK-05, TASK-06, TASK-07, TASK-08, TASK-09, TASK-10, TASK-11, TASK-12, TASK-13, TASK-14
- 状态: completed

### TASK-21: TEST-DIST-018~034 — PP 新功能测试实现 ✅ COMPLETED
- SPEC: TEST-DIST-018~034 | 文件: tests/dist/pipeline/（新建）
- 实现: MockPipelineEnv 驱动；TEST-DIST-018 验证 stage 权重加载层范围；TEST-DIST-019 验证 micro-batch 切分 + activation send/recv loopback；TEST-DIST-020 验证 1F1B schedule 序列（forward/bubble 顺序）；TEST-DIST-021 验证交错 1F1B virtual stage 数；TEST-DIST-022 验证 bubble_ratio 指标采集
- 复用锚点: code:MockPipelineEnv, code:PipelineScheduler, code:MicroBatchScheduler
- 依赖: TASK-15, TASK-16, TASK-17, TASK-18, TASK-19
- 状态: completed

### TASK-22: oracle_gate 验收 + REQ-DIST-001~034 状态升级 implemented
- SPEC: REQ-DIST-001~034 | 文件: SPEC/43-DISTRIBUTED-IMPLEMENTATION.html
- 实现:
  - cargo test --lib tests/dist/ 全部绿灯
  - oracle_gate(sourceDir="./src", dir="./SPEC", taskId="TASK-22", reqIds=[REQ-DIST-001..034]) → canCommit=true
  - spec_govern(auditAction="status", ids=[REQ-DIST-001..034], status="implemented")
- 复用锚点: spec:REQ-DIST-001~034
- 依赖: TASK-20, TASK-21
- 状态: pending
