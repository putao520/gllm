# 开发计划: IntentBias + DistributedConfig 实现 | epoch: 1 | status: active

## 范围：REQ-IB-001~014

### 调研结论

| REQ 组 | SPEC 状态 | 代码位置 | 复用探测 | 差距 |
|--------|----------|---------|---------|------|
| REQ-IB-001 | approved | `src/engine/intent_bias.rs` — **已实现** IntentBias struct 5 字段 + Default | code=full_match | 无差距，已交付 |
| REQ-IB-002 | approved | `src/engine/intent_bias.rs` — **已实现** ScenarioHint 6 变体 + 5 调制系数方法 | code=full_match | 无差距，已交付 |
| REQ-IB-003 | approved | `src/engine/intent_bias.rs` — **已实现** OverlapHint 5 变体 + 3 SM 映射方法 | code=partial_match | **缺 SM60-69 映射方法** (SPEC 要求 4 层: SM90+/SM70-89/SM60-69/SM<60，代码只有 3 层: SM90+/SM70-89/SM<60)；缺 ForceFlux 单 GPU 降级逻辑 |
| REQ-IB-004 | approved | 无 | code=no_match | **StrategyBiasResolver 三阶段合并管线全新** |
| REQ-IB-005 | approved | `src/client_fragments/builder.inc.rs` — 有 inference_mode() 但**无 .intent() setter**；`src/client_fragments/error_config.inc.rs` — ClientConfig 只有 weight_paging_enabled | code=partial_match | **缺 .intent() setter**；**缺 ClientConfig.intent_bias 字段**；**缺 inference_mode() deprecated 标注** |
| REQ-IB-006 | approved | 无 | code=no_match | **DistributedConfig 聚合体全新** |
| REQ-IB-007 | approved | 无 | code=no_match | **ParallelConfig 全新** |
| REQ-IB-008 | approved | 无 | code=no_match | **PdDisaggConfig + PdDisaggMode + NodeRole 全新** |
| REQ-IB-009 | approved | 无 | code=no_match | **KvDistributionConfig + KvDistMode 全新** |
| REQ-IB-010 | approved | 无 | code=no_match | **CommConfig + CommCompressHint 全新** |
| REQ-IB-011 | approved | 无 | code=no_match | **MoeDistributedConfig + ExpertPlacement + AllToAllStrategy 全新** |
| REQ-IB-012 | approved | `src/client_fragments/error_config.inc.rs` — ClientConfig 只有 weight_paging_enabled | code=partial_match | **缺 .distributed() setter**；**缺 ClientConfig.distributed 字段** |
| REQ-IB-013 | approved | 无 | code=no_match | **DF-IB-001 数据流标注全新** |
| REQ-IB-014 | approved | 无 | code=no_match | **CF-IB-001 控制流标注全新** |

### 关键发现

1. **REQ-IB-001~002 已完整实现**: IntentBias + ScenarioHint 在 `src/engine/intent_bias.rs` 中，struct 字段、Default、调制系数方法全部就位，测试覆盖完整
2. **REQ-IB-003 部分实现**: OverlapHint 5 变体已定义，3 个 SM 映射方法已有，但**缺 SM60-69 层级**（SPEC 要求 4 层 SM 映射）和**缺 ForceFlux 单 GPU 降级逻辑**
3. **StrategyBiasResolver 全新**: 三阶段合并管线 (auto_bias → scenario_override → clamp) 需新增
4. **StrategyArbiter 已有基础**: `src/engine/arbiter.rs` 有 `arbitrate()` 入口，需新增 `arbitrate_with_bias()` 方法
5. **ClientBuilder 需扩展**: 新增 `.intent()` 和 `.distributed()` setter，`inference_mode()` 标 deprecated
6. **ClientConfig 需扩展**: 新增 `intent_bias: IntentBias` 和 `distributed: DistributedConfig` 字段
7. **DistributedConfig 全新**: 5 子配置 + 6 枚举，nccl feature-gated，需新增 `src/engine/distributed_config.rs`
8. **StrategyBias 在 gllm-kernels**: `gllm_kernels::compiler::planner::StrategyBias` 是 canonical SSOT，12 个 f64 字段，`validate()` clamp 方法已存在
9. **SmPartitionConfig 已在 gllm**: `src/engine/mega_kernel_gpu.rs` 有 SmPartitionConfig，decode_sm_ratio 可直接覆盖其 decode_ctas/total_ctas

## 影响矩阵

| SPEC ID | 关联 TASK | 文件 |
|---------|----------|------|
| REQ-IB-001 | (已实现) | `src/engine/intent_bias.rs` |
| REQ-IB-002 | (已实现) | `src/engine/intent_bias.rs` |
| REQ-IB-003 | TASK-1 | `src/engine/intent_bias.rs` |
| REQ-IB-004 | TASK-2 | `src/engine/arbiter.rs` |
| REQ-IB-005 | TASK-3 | `src/client_fragments/builder.inc.rs`, `src/client_fragments/error_config.inc.rs` |
| REQ-IB-006 | TASK-5 | `src/engine/distributed_config.rs` |
| REQ-IB-007 | TASK-4 | `src/engine/distributed_config.rs` |
| REQ-IB-008 | TASK-4 | `src/engine/distributed_config.rs` |
| REQ-IB-009 | TASK-4 | `src/engine/distributed_config.rs` |
| REQ-IB-010 | TASK-4 | `src/engine/distributed_config.rs` |
| REQ-IB-011 | TASK-4 | `src/engine/distributed_config.rs` |
| REQ-IB-012 | TASK-6 | `src/client_fragments/builder.inc.rs`, `src/client_fragments/error_config.inc.rs` |
| REQ-IB-013 | TASK-7 | `src/engine/arbiter.rs` |
| REQ-IB-014 | TASK-7 | `src/engine/arbiter.rs` |

## 任务树（扁平列表，禁止分 Phase/阶段/分期）

### TASK-1: REQ-IB-003 补全 — OverlapHint SM60-69 映射 + ForceFlux 单 GPU 降级
- SPEC: REQ-IB-003 [OverlapHint: 5 变体已有; 补 mk_variant_sm60_69() 方法 (SM60-69 层级映射: PreferOverlap→GridSync, PreferIsolated→GridSync, Auto/ForceDoubleBuffer/ForceFlux→select_mk_variant()); 补 ForceFlux 单 GPU 降级逻辑 (tracing::warn + 退回 Auto)] | TDD: TEST-IB-003 | 文件: `src/engine/intent_bias.rs` | 实现: (1) 新增 mk_variant_sm60_69() 方法，SM60-69 映射: PreferOverlap→GridSync, PreferIsolated→GridSync, Auto/ForceDoubleBuffer/ForceFlux→select_mk_variant() (2) 新增 resolve_overlap(is_single_gpu: bool) -> OverlapHint 方法，ForceFlux + 单 GPU → 降级 Auto + tracing::warn (3) 补全测试
- 复用锚点: spec=full_match, code=partial_match(OverlapHint已有3层映射), pattern=full_match
- 依赖: 无
- 状态: pending

### TASK-2: REQ-IB-004 — StrategyBiasResolver 三阶段合并管线
- SPEC: REQ-IB-004 [StrategyBiasResolver: resolve(intent_bias, batch_size, archetype, hw) → StrategyBias; stage1: auto_bias = StrategyArbiter::arbitrate(mode_baseline, archetype, hw); stage2: scenario 覆盖 → 重跑 arbitrate(scenario_baseline, archetype, hw) + scenario 调制系数乘到 bias 字段 + comm_overlap 覆盖 MkVariant + decode_sm_ratio 覆盖 SmPartitionConfig + kv_budget_scale 覆盖 StrategyBias.kv_cache_budget_scale + quant_aggression 覆盖 StrategyBias.quantization_aggressiveness; stage3: bias.validate() clamp] | TDD: TEST-IB-004 | 文件: `src/engine/arbiter.rs` | 实现: (1) 新增 StrategyBiasResolver struct (2) resolve() 方法三阶段: stage1 调用 StrategyArbiter::arbitrate() 获取 auto_bias; stage2 根据 IntentBias 字段覆盖: scenario!=Auto 时重跑 arbitrate(scenario.inference_mode_baseline(), archetype, hw) 再乘 scenario 调制系数; comm_overlap 影响 MkVariant 选择; decode_sm_ratio Some 时覆盖 decode_ratio_scale; kv_budget_scale Some 时覆盖 kv_cache_budget_scale; quant_aggression Some 时覆盖 quantization_aggressiveness; stage3 调用 bias.validate() (3) 新增 arbitrate_with_bias(mode, archetype, hw, intent_bias) 便捷入口 (4) 补全测试覆盖所有 scenario 变体 + Option Some/None 路径
- 复用锚点: spec=full_match, code=partial_match(StrategyArbiter::arbitrate已存在), pattern=full_match
- 依赖: TASK-1
- 状态: pending

### TASK-3: REQ-IB-005 — ClientBuilder.intent() setter + ClientConfig.intent_bias 字段 + inference_mode() deprecated
- SPEC: REQ-IB-005 [ClientBuilder 新增 .intent(IntentBias) setter; IntentBias 存入 ClientConfig.intent_bias; build_state() 传 IntentBias 到 StrategyBiasResolver; inference_mode() 标 #[deprecated]; 不设 .intent() 行为与现有完全一致] | TDD: TEST-IB-005 | 文件: `src/client_fragments/builder.inc.rs`, `src/client_fragments/error_config.inc.rs` | 实现: (1) ClientConfig 新增 intent_bias: IntentBias 字段 (Default = IntentBias::default()) (2) ClientBuilder 新增 intent_bias: IntentBias 字段 (3) ClientBuilder 新增 .intent(IntentBias) setter 方法 (4) build_state() 中 StrategyArbiter::arbitrate() 调用替换为 StrategyBiasResolver::resolve(intent_bias, ...) (5) inference_mode() setter 标注 #[deprecated(note="use .intent(IntentBias{ scenario: ScenarioHint::..., ..Default::default() })")] (6) 补全测试
- 复用锚点: spec=full_match, code=partial_match(ClientBuilder/ClientConfig已存在), pattern=full_match
- 依赖: TASK-2
- 状态: pending

### TASK-4: REQ-IB-007~011 — DistributedConfig 5 子配置 + 6 枚举
- SPEC: REQ-IB-007 [ParallelConfig: tp_size/pp_size/ep_size/rank/world_size(u32) + unique_id([u8;128]); 约束 tp*pp*ep==world_size; validate()] + REQ-IB-008 [PdDisaggConfig: mode(PdDisaggMode)/role(NodeRole); PdDisaggMode: Collocated/Disaggregated; NodeRole: Auto/PrefillOnly/DecodeOnly/Mixed] + REQ-IB-009 [KvDistributionConfig: mode(KvDistMode)/mirror_heads(u32); KvDistMode: Local/OnDemand/Mirror/PartialHeadMirror/TieredCache] + REQ-IB-010 [CommConfig: overlap(OverlapHint)/compress(CommCompressHint)/algorithm_override(Option<CollectiveAlgorithm>); CommCompressHint: Auto/AlwaysCompress/NeverCompress/ForceQuant(QuantScheme)] + REQ-IB-011 [MoeDistributedConfig: expert_placement(ExpertPlacement)/all_to_all(AllToAllStrategy); ExpertPlacement: Auto/RoundRobin/HotCold/Custom(Vec<u32>); AllToAllStrategy: Auto/NvlinkAllToAll/RdmaAllToAll/HierarchicalAllToAll] | TDD: TEST-IB-007~011 | 文件: `src/engine/distributed_config.rs` | 实现: (1) 新建 src/engine/distributed_config.rs (2) ParallelConfig struct 6 字段 + validate() (3) PdDisaggMode 2 变体 + NodeRole 4 变体 + PdDisaggConfig struct 2 字段 (4) KvDistMode 5 变体 + KvDistributionConfig struct 2 字段 (5) CommCompressHint 4 变体 (Auto/AlwaysCompress/NeverCompress/ForceQuant(String)) + CommConfig struct 3 字段 (6) ExpertPlacement 4 变体 + AllToAllStrategy 4 变体 + MoeDistributedConfig struct 2 字段 (7) 所有类型 derive(Debug,Clone,PartialEq), Default 实现 (8) nccl feature-gated: 非 nccl 构建 DistributedConfig 仅允许 Default 值 (9) 在 src/engine/mod.rs 添加 mod distributed_config (10) 在 src/lib.rs 添加 pub use (11) 补全测试
- 复用锚点: spec=full_match, code=no_match, pattern=full_match
- 依赖: TASK-1 (OverlapHint 复用)
- 状态: pending

### TASK-5: REQ-IB-006 — DistributedConfig 聚合体 + nccl feature-gated
- SPEC: REQ-IB-006 [DistributedConfig: parallel/pd_disagg/kv_distribution/comm/moe 5 子配置; Default = 单机零分布式; nccl feature-gated: 非 nccl 构建仅允许 Default 值; 设置非 Default 值返回编译错误] | TDD: TEST-IB-006 | 文件: `src/engine/distributed_config.rs` | 实现: (1) DistributedConfig struct 5 字段 (2) Default 返回全单机默认值 (3) validate() 方法: 非 nccl 构建检查是否全为 Default 值，非 Default 返回 Err (4) #[cfg(feature = "nccl")] 门控分布式功能 (5) 补全测试
- 复用锚点: spec=full_match, code=no_match, pattern=partial_match(feature gating pattern)
- 依赖: TASK-4
- 状态: pending

### TASK-6: REQ-IB-012 — ClientBuilder.distributed() setter + ClientConfig.distributed 字段
- SPEC: REQ-IB-012 [ClientBuilder 新增 .distributed(DistributedConfig) setter; DistributedConfig 存入 ClientConfig.distributed; 不设 .distributed() 行为与现有完全一致 (单机模式); distributed 字段优先级高于 intent 对应字段 (distributed.comm.overlap > intent.comm_overlap); nccl feature-gated] | TDD: TEST-IB-012 | 文件: `src/client_fragments/builder.inc.rs`, `src/client_fragments/error_config.inc.rs` | 实现: (1) ClientConfig 新增 distributed: DistributedConfig 字段 (Default = DistributedConfig::default()) (2) ClientBuilder 新增 distributed: DistributedConfig 字段 (3) ClientBuilder 新增 .distributed(DistributedConfig) setter 方法 (4) build_state() 中集成 DistributedConfig: ParallelConfig 影响 CompilerGraph 构建和权重加载; PdDisaggConfig 影响 MegaKernel 编译路径; KvDistributionConfig 影响 VmInstr 生成; CommConfig 影响 SmPartitionConfig + CommScheduleHint; MoeDistributedConfig 影响 ExpertWeightPrefetcher + AllToAll (5) distributed.comm.overlap 优先级高于 intent.comm_overlap (6) nccl feature-gated: 非 nccl 构建设置非 Default DistributedConfig 返回编译错误 (7) 补全测试
- 复用锚点: spec=full_match, code=partial_match(ClientBuilder/ClientConfig已存在), pattern=full_match
- 依赖: TASK-3, TASK-5
- 状态: pending

### TASK-7: REQ-IB-013~014 — DF-IB-001 数据流 + CF-IB-001 控制流标注
- SPEC: REQ-IB-013 [DF-IB-001: IntentBias → StrategyBiasResolver → StrategyBias; data-flow-from=API-CLIENT-INTENT, data-flow-to=ENT-STRATEGY-BIAS; transform=StrategyBiasResolver::resolve()] + REQ-IB-014 [CF-IB-001: StrategyArbiter 调度控制流; data-controlflow-source=API-STRATEGY-ARBITRATE, data-controlflow-target=ENT-STRATEGY-BIAS; relation=invokes] | TDD: TEST-IB-013~014 | 文件: `src/engine/arbiter.rs`, `src/lib.rs` | 实现: (1) StrategyBiasResolver::resolve() 上添加 @trace REQ-IB-013 [entity:API-CLIENT-INTENT] [api:POST /internal/strategy/resolve_bias] 标注 (2) StrategyArbiter::arbitrate_with_bias() 上添加 @trace REQ-IB-014 [entity:ENT-STRATEGY-ARBITER] [api:POST /internal/strategy/arbitrate] 控制流标注 (3) src/lib.rs 添加 pub use engine::arbiter::StrategyBiasResolver 导出 (4) 补全测试验证标注存在
- 复用锚点: spec=full_match, code=partial_match(@trace pattern已存在), pattern=full_match
- 依赖: TASK-2
- 状态: pending
