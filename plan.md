# 开发计划: IntentBias + DistributedConfig 实现 | epoch: 1 | status: active

## 范围：REQ-IB-001~014

### 调研结论

| REQ 组 | SPEC 状态 | 代码位置 | 复用探测 | 差距 |
|--------|----------|---------|---------|------|
| REQ-IB-001~003 | approved | 新增 `src/engine/intent_bias.rs` | pattern=partial_match (StrategyBias已存在) | IntentBias/ScenarioHint/OverlapHint 全新 struct + Default |
| REQ-IB-004~005 | approved | `src/engine/arbiter.rs` | code=partial_match (StrategyArbiter已存在, 需扩展resolve) | StrategyBiasResolver 三阶段合并管线全新 |
| REQ-IB-006~012 | approved | 新增 `src/engine/distributed_config.rs` | code=no_match | DistributedConfig + 5子配置 + 6枚举 全新, nccl feature-gated |
| REQ-IB-013~014 | approved | `src/engine/arbiter.rs` + DF/CF | code=no_match | DF-IB-001/CF-IB-001 数据流+控制流标注 |

### 关键发现

1. **StrategyBias 已在 gllm-kernels**: `gllm_kernels::compiler::planner::StrategyBias` 是 canonical SSOT，gllm 通过 `pub use` 重导出
2. **StrategyArbiter 已在 gllm**: `src/engine/arbiter.rs` 有 `arbitrate()` 入口，但缺少 IntentBias 集成
3. **IntentBias 全新**: 5 字段 + ScenarioHint + OverlapHint 枚举，需新增文件
4. **DistributedConfig 全新**: 5 子配置 + 6 枚举，nccl feature-gated，需新增文件
5. **StrategyBiasResolver 全新**: 三阶段合并管线 (auto_bias → scenario_override → clamp)，扩展 StrategyArbiter

## 影响矩阵

| SPEC ID | 关联 TASK | 文件 |
|---------|----------|------|
| REQ-IB-001 | TASK-1 | `src/engine/intent_bias.rs` |
| REQ-IB-002 | TASK-2 | `src/engine/intent_bias.rs` |
| REQ-IB-003 | TASK-3 | `src/engine/intent_bias.rs` |
| REQ-IB-004 | TASK-4 | `src/engine/arbiter.rs` |
| REQ-IB-005 | TASK-5 | `src/engine/arbiter.rs` |
| REQ-IB-006 | TASK-6 | `src/engine/distributed_config.rs` |
| REQ-IB-007 | TASK-7 | `src/engine/distributed_config.rs` |
| REQ-IB-008 | TASK-8 | `src/engine/distributed_config.rs` |
| REQ-IB-009 | TASK-9 | `src/engine/distributed_config.rs` |
| REQ-IB-010 | TASK-10 | `src/engine/distributed_config.rs` |
| REQ-IB-011 | TASK-11 | `src/engine/distributed_config.rs` |
| REQ-IB-012 | TASK-12 | `src/engine/distributed_config.rs` |
| REQ-IB-013 | TASK-13 | `src/engine/arbiter.rs` |
| REQ-IB-014 | TASK-14 | `src/engine/arbiter.rs` |

## 任务树（扁平列表，禁止分 Phase/阶段/分期）

### TASK-1: REQ-IB-001 — IntentBias 5 字段 struct + Default
- SPEC: REQ-IB-001 [IntentBias: scenario(ScenarioHint), comm_overlap(OverlapHint), decode_sm_ratio(Option<f32>), kv_budget_scale(Option<f32>), quant_aggression(Option<f32>); Default 实现; 全部字段必填+有默认值] | TDD: TEST-IB-001 | 文件: `src/engine/intent_bias.rs` | 实现: 创建 IntentBias struct, 5 字段全部 pub, derive(Debug,Clone,PartialEq), Default trait 返回 Auto/Auto/None/None/None; 新增 mod 声明到 src/engine/mod.rs; pub use 到 src/lib.rs
- 复用锚点: spec=full_match, code=no_match, pattern=partial_match(StrategyBias field pattern)
- 依赖: TASK-2, TASK-3 (枚举类型先定义)
- 状态: pending

### TASK-2: REQ-IB-002 — ScenarioHint 6 变体 + 5 调制系数
- SPEC: REQ-IB-002 [ScenarioHint: Auto/LatencyCritical/ThroughputOptimal/LongContext/DistributedHeavy/MemoryConstrained; 每个变体 5 调制系数: inference_mode_baseline, kv_cache_budget_scale_mod, quantization_aggressiveness_mod, expert_prefetch_priority_mod, pipeline_cost_scale_mod; 映射表如 SPEC 定义] | TDD: TEST-IB-002 | 文件: `src/engine/intent_bias.rs` | 实现: ScenarioHint enum 6 变体, 每变体 5 个 f64 方法返回调制系数, inference_mode_baseline() 返回 InferenceMode
- 复用锚点: spec=full_match, code=partial_match(InferenceMode已存在), pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-3: REQ-IB-003 — OverlapHint 5 变体 + 硬件映射
- SPEC: REQ-IB-003 [OverlapHint: Auto/PreferOverlap/PreferIsolated/ForceDoubleBuffer/ForceFlux; mk_variant_sm90plus/sm70_89/sm_below60 映射; overlap_mode 映射到 gllm-nccl OverlapMode] | TDD: TEST-IB-003 | 文件: `src/engine/intent_bias.rs` | 实现: OverlapHint enum 5 变体, 3 个硬件映射方法返回 String/enum, overlap_mode() 返回 OverlapMode (nccl feature-gated)
- 复用锚点: spec=full_match, code=no_match, pattern=partial_match(MkVariant已存在)
- 依赖: 无
- 状态: pending

### TASK-4: REQ-IB-004 — StrategyBiasResolver 三阶段合并管线
- SPEC: REQ-IB-004 [StrategyBiasResolver: stage1_auto_bias(StrategyArbiter::arbitrate), stage2_scenario_override(IntentBias字段覆盖), stage3_clamp_result(bias.validate() clamp); resolve(intent_bias, batch_size, archetype, hw) → StrategyBias] | TDD: TEST-IB-004 | 文件: `src/engine/arbiter.rs` | 实现: StrategyBiasResolver struct, resolve() 三阶段: (1) auto_bias = StrategyArbiter::arbitrate(mode_baseline, archetype, hw) (2) scenario 覆盖 inference_mode → 重新 arbitrate + comm_overlap 覆盖 mk_variant + decode_sm_ratio/kv_budget_scale/quant_aggression 覆盖 StrategyBias 字段 (3) bias.validate() clamp 合法范围
- 复用锚点: spec=full_match, code=partial_match(StrategyArbiter::arbitrate已存在), pattern=full_match
- 依赖: TASK-1, TASK-5
- 状态: pending

### TASK-5: REQ-IB-005 — StrategyArbiter::arbitrate 扩展接受 IntentBias
- SPEC: REQ-IB-005 [StrategyArbiter::arbitrate 扩展: 当 IntentBias 存在时走 StrategyBiasResolver.resolve(); 无 IntentBias 时走原有 arbitrate() 路径] | TDD: TEST-IB-005 | 文件: `src/engine/arbiter.rs` | 实现: 新增 arbitrate_with_bias(mode, archetype, hw, intent_bias) 方法, 内部调 StrategyBiasResolver::resolve(); 原有 arbitrate() 保持不变(零 IntentBias 退化为默认路径)
- 复用锚点: spec=full_match, code=full_match(StrategyArbiter已存在), pattern=full_match
- 依赖: TASK-1
- 状态: pending

### TASK-6: REQ-IB-006 — DistributedConfig 聚合体
- SPEC: REQ-IB-006 [DistributedConfig: parallel/pd_disagg/kv_distribution/comm/moe 5 子配置; nccl feature-gated; Default = 单机模式] | TDD: TEST-IB-006 | 文件: `src/engine/distributed_config.rs` | 实现: DistributedConfig struct, 5 字段, derive(Debug,Clone,PartialEq), Default 返回全单机默认值; #[cfg(feature = "nccl")] 门控整个模块; 非 nccl 构建零开销
- 复用锚点: spec=full_match, code=no_match, pattern=partial_match(feature gating pattern)
- 依赖: TASK-7, TASK-8, TASK-9, TASK-10, TASK-11
- 状态: pending

### TASK-7: REQ-IB-007 — ParallelConfig
- SPEC: REQ-IB-007 [ParallelConfig: tp_size(u32,Default=1), pp_size(u32,Default=1), ep_size(u32,Default=1), rank(u32,Default=0), world_size(u32,Default=1), unique_id(String,Default=""); 约束 tp*pp*ep==world_size] | TDD: TEST-IB-007 | 文件: `src/engine/distributed_config.rs` | 实现: ParallelConfig struct, 6 字段, validate() 方法检查 tp*pp*ep==world_size
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-8: REQ-IB-008 — PdDisaggConfig + PdDisaggMode + NodeRole
- SPEC: REQ-IB-008 [PdDisaggConfig: mode(PdDisaggMode,Default=Collocated), role(NodeRole,Default=Auto); PdDisaggMode: Collocated/Disaggregated; NodeRole: Auto/PrefillOnly/DecodeOnly/Mixed] | TDD: TEST-IB-008 | 文件: `src/engine/distributed_config.rs` | 实现: 3 个类型, PdDisaggMode 2 变体, NodeRole 4 变体
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-9: REQ-IB-009 — KvDistributionConfig + KvDistMode
- SPEC: REQ-IB-009 [KvDistributionConfig: mode(KvDistMode,Default=Local), mirror_heads(u32,Default=0); KvDistMode: Local/OnDemand/Mirror/PartialHeadMirror/TieredCache] | TDD: TEST-IB-009 | 文件: `src/engine/distributed_config.rs` | 实现: KvDistMode 5 变体, KvDistributionConfig 2 字段
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-10: REQ-IB-010 — CommConfig + CommCompressHint
- SPEC: REQ-IB-010 [CommConfig: overlap(OverlapHint,Default=Auto,复用IntentBias.comm_overlap), compress(CommCompressHint,Default=Auto), algorithm_override(String,Default=""); CommCompressHint: Auto/AlwaysCompress/NeverCompress/ForceQuant] | TDD: TEST-IB-010 | 文件: `src/engine/distributed_config.rs` | 实现: CommCompressHint 4 变体, CommConfig 3 字段, overlap 字段复用 OverlapHint
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: TASK-3 (OverlapHint)
- 状态: pending

### TASK-11: REQ-IB-011 — MoeDistributedConfig + ExpertPlacement + AllToAllStrategy
- SPEC: REQ-IB-011 [MoeDistributedConfig: expert_placement(ExpertPlacement,Default=Auto), all_to_all(AllToAllStrategy,Default=Auto); ExpertPlacement: Auto/RoundRobin/HotCold/Custom; AllToAllStrategy: Auto/NvlinkAllToAll/RdmaAllToAll/HierarchicalAllToAll] | TDD: TEST-IB-011 | 文件: `src/engine/distributed_config.rs` | 实现: ExpertPlacement 4 变体, AllToAllStrategy 4 变体, MoeDistributedConfig 2 字段
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-12: REQ-IB-012 — ClientConfig 聚合体
- SPEC: REQ-IB-012 [ClientConfig: intent_bias(IntentBias,Default=Default), distributed(DistributedConfig,Default=Default); nccl feature-gated on distributed] | TDD: TEST-IB-012 | 文件: `src/engine/distributed_config.rs` | 实现: ClientConfig struct, 2 字段; distributed 字段 #[cfg(feature = "nccl")]; Default = Default IntentBias + Default DistributedConfig; 集成到 ClientBuilder
- 复用锚点: spec=full_match, code=no_match, pattern=partial_match(ClientBuilder已存在)
- 依赖: TASK-1, TASK-6
- 状态: pending

### TASK-13: REQ-IB-013 — DF-IB-001 数据流标注
- SPEC: REQ-IB-013 [DF-IB-001: IntentBias → StrategyBiasResolver → StrategyBias; data-flow-from=API-CLIENT-INTENT, data-flow-to=ENT-STRATEGY-BIAS; transform=StrategyBiasResolver::resolve()] | TDD: TEST-IB-013 | 文件: `src/engine/arbiter.rs` | 实现: 在 StrategyBiasResolver::resolve() 上添加 @trace REQ-IB-013 标注, 在 lib.rs 添加 pub use 导出
- 复用锚点: spec=full_match, code=partial_match(@trace pattern已存在), pattern=full_match
- 依赖: TASK-4
- 状态: pending

### TASK-14: REQ-IB-014 — CF-IB-001 控制流标注
- SPEC: REQ-IB-014 [CF-IB-001: StrategyArbiter 调度控制流; data-controlflow-source=API-STRATEGY-ARBITRATE, data-controlflow-target=ENT-STRATEGY-BIAS; relation=invokes] | TDD: TEST-IB-014 | 文件: `src/engine/arbiter.rs` | 实现: 在 StrategyArbiter::arbitrate_with_bias() 上添加 @trace REQ-IB-014 控制流标注
- 复用锚点: spec=full_match, code=partial_match(@trace pattern已存在), pattern=full_match
- 依赖: TASK-5
- 状态: pending
