# 开发计划: 硬件能力等级主权（Tier Sovereignty）— IsaHook → OpImpl 重构

## epoch
1

## status
completed

## taskTree
- TASK-03: OpImpl trait + select 两阶段 + 13 后端 GEMM-FMA 迁移 [completed]
  - TASK-04: 硬件特性矩阵细粒度化 [completed, blockedBy TASK-03]
  - TASK-05: BF16 各硬件等级 OpImpl 实现 [completed, blockedBy TASK-03, TASK-04]
    - TASK-07: numerical_sim 跨等级等价验证 [completed, blockedBy TASK-05]
  - TASK-06: VmInstr 指令名语义化重命名 [completed, blockedBy TASK-04, 串行文件域 isa_profile.rs]
    - TASK-08: SmolLM2 logits 对齐 golden 验收 [completed, blockedBy TASK-05, TASK-06]

## reqLedger
REQ-HW-TIER-001 ✅, REQ-HW-TIER-002 ✅, REQ-HW-TIER-003 ✅, REQ-HW-TIER-004 ✅, REQ-HW-TIER-005 ✅, REQ-HW-TIER-006 ✅,
REQ-TIER-SOVEREIGNTY-001 ✅, REQ-TIER-SOVEREIGNTY-002 ✅

## 范围
ARCH-HW-TIER-SOVEREIGNTY（6 REQ-HW-TIER + 4 criterion）| 10-QUALITY REQ-TIER-SOVEREIGNTY-001/002 | GEMM-FMA 算子族 13 后端全量迁移 | VmInstr 语义化重命名

## 影响矩阵
| SPEC ID | 关联 TASK | 文件 |
|---------|-----------|------|
| REQ-HW-TIER-002/003 | TASK-03 | src/compiler/codegen/vm/isa_hook.rs, gemm_emit.rs, isa_profile.rs, 新建 op_impl.rs |
| REQ-HW-TIER-001 | TASK-04 | src/compiler/codegen/vm/isa_profile.rs |
| REQ-HW-TIER-004 | TASK-05 | src/compiler/codegen/vm/gemm_emit.rs (BF16 各 OpImpl) |
| REQ-HW-TIER-005 | TASK-06 | src/compiler/codegen/vm/instr_fragments/vminstr.inc.rs + 全 lower 路径 |
| REQ-HW-TIER-006 | TASK-07 | src/compiler/codegen/vm/numerical_sim.rs |

## TASK-03: OpImpl trait + select 两阶段框架 + 13 后端 GEMM-FMA 全量迁移
- SPEC: REQ-HW-TIER-002/003 + CR-TIER-SOVEREIGNTY-001~004 + REQ-TIER-SOVEREIGNTY-001/002
- 设计来源: /tmp/gsc-arch-9599774a-cf78-4009-ba07-902a72a0b872.md（architect sessionId 9599774a）
- 交付物（全量，无缩减）:
  1. 新建 op_impl.rs: FeatureSet bitflags（从 IsaProfile.features 派生）+ EmitCtx + GemmOpLayout + trait OpImpl<L>
  2. 13 后端 GEMM-FMA OpImpl registry: GemmFmaBlis/GemmAmx{Bf16,Fp16,Fp8}Tile/GemmWgmma/GemmTcgen05/GemmMfmaV1V2/GemmSmeTile/GemmTcSm70/80/GemmScalar
  3. select_gemm_impl(feats, dtype) 选择器: filter(supports_dtype).filter(requires⊆feats).max_by(throughput)
  4. gemm_emit.rs 改造: select-then-emit 两阶段，调用 .emit()
  5. FmaStrategy/select_fma/select_fma_best/select_fma_candidates/estimate_strategy_cost 全删，全项目零残留
  6. 13 个 numerical_sim 对齐测试（每个 OpImpl 对齐 GemmScalar oracle）
  7. IsaProfile::feature_set() 派生方法 + CompileSession.feature_set 字段
- 3 个已拍板决策:
  - GemmLayout → GemmOpLayout（避免与 jit/layout.rs 同名）
  - estimate_strategy_cost → throughput_refine（保留 budget 调优）
  - can_blis 放 GemmFmaBlis.emit 内部（selector 纯净）
- 铁律: 禁止并行新旧 trait（ARCH-UNCONSTITUTION-CONTAGION）；select_fma 删除是原子操作，13后端+33mock同批迁移
- 调度: L2 的 13 个 impl 同写 registry 文件 → 单 Agent 串行，禁止并行撞文件
- 验收: cargo check --workspace 通过 + 13 后端 numerical_sim 全绿 + commit_gate
- 复用锚点: code:emit_tile_gemm, code:emit_gemm_blis_inline, code:simulate_trace, code:simulate_compile
- 依赖: 无（C/E 节点已完成）
- 状态: completed

## TASK-04: 硬件特性矩阵细粒度化
- SPEC: REQ-HW-TIER-001 + CR-TIER-SOVEREIGNTY-002
- 现状: IsaProfile 已有 has_avx512_bf16/has_f16c/has_vnni/has_vp2intersect/has_avx10_2/has_apx 等
- 交付物: 验证 13 后端的 FeatureSet 声明完整覆盖各自 ISA；补齐缺失 flag（如 has_amx_fp16 已在 X86AmxHook struct 但未进 IsaProfile）；DeviceProfile detection 对齐
- 依赖: TASK-03
- 状态: completed

## TASK-05: BF16 各硬件等级 OpImpl 实现
- SPEC: REQ-HW-TIER-004
- 交付物: BF16 在各 ISA 的 OpImpl 真实 emit 实现（AVX512_BF16 vcvtneps2bf16 / AVX2 向量化序列 / Scalar）
- 依赖: TASK-03, TASK-04
- 状态: completed

## TASK-06: VmInstr 指令名语义化重命名
- SPEC: REQ-HW-TIER-005 + CR-VMINSTR-SEMANTIC-001
- 交付物: Vp2Intersect → SparseMaskIntersect（首发，先 audit 再决定 vpdpbusd 等后续）；全 lower 路径映射（vminstr.inc.rs + x86/aarch64/gpu_lower + reg_alloc + verify + program.inc + isa_profile.rs IsaFeature enum + KernelCapabilities）
- 依赖: TASK-04（**串行：#4 和 #6 都改 isa_profile.rs，共享 git 工作树禁止并行**，按 C-E-W-V.1 撞文件域路由）
- 文件域冲突: isa_profile.rs（#4 补 flag 行 vs #6 改 Vp2Intersect enum 行，不同行但共享工作树 → 必须串行）
- 状态: completed

## TASK-07: numerical_sim 跨等级等价验证
- SPEC: REQ-HW-TIER-006 + CR-TIER-SOVEREIGNTY-004
- 交付物: 所有 OpImpl 的 VmInstr 序列跑 numerical_sim 对齐 scalar reference
- 依赖: TASK-05
- 状态: completed

## TASK-08: SmolLM2 logits 对齐 golden 验收
- 交付物: SmolLM2-135M BF16 E2E logits 对齐 PyTorch golden（argmax=253）
- 依赖: TASK-05, TASK-06
- 状态: completed
