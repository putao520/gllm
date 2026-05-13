# 设备特化融合 PASS (ARCH-DEVICE-FUSION)

> **实现状态**: ✅ REQ-FUS-001~009 全部完成 — `HardwareProfile` 4 个新方法 (max_epilogue_depth/supports_quant_epilogue/compute_roi_weight/cache_roi_weight) + `FusionCostModel` 代价模型 + `HwOptEngine::solve_fusion()` 使用 profile 驱动融合决策

## 定位

gllm-kernels Phase 2 融合决策层的设备适配。10 条通用 FusionRule 已实现，但 12 个 HardwareProfile 的特化融合能力未完全利用。本 SPEC 定义每个 Profile 的融合 PASS 特化策略。

## 前置原则

- **ARCH-CPU-GPU-UNIFIED**: FusionRule 层禁止 `if backend == GPU { ... }` 分支。设备差异通过 FusionRule 参数化（register limit、cache size、tensor core gen）驱动
- **ARCH-NO-HW-DEGRADATION**: 硬件不支持某融合模式 → codegen 生成该硬件的最优路径（仍然是融合的），不是拆分融合图
- **FusionRule 参数化**: 每条规则通过 `DeviceProfile` 字段（寄存器数、SIMD 宽度、cache 层级、tensor core 代数）自动适配，不硬编码 ISA 名

## 现有基础

### 通用融合规则 (10 条)

| 规则 | 融合模式 | 设备感知 | 状态 |
|------|---------|---------|------|
| QkvSharedInput | Q/K/V 三 GEMM 共享 pack_a | register limit | ✅ |
| FusedQkvNormRope | QKV + QkNorm + ValueNorm + RoPE | register limit | ✅ |
| FFNBlock | Gate+Up → SiLU → Mul → Down | register limit | ✅ |
| EpilogueInjection | GEMM + elementwise 到累加器 | register limit + L1 | ✅ |
| LoopFusion | 连续 elementwise 合并单循环 | N/A | ✅ |
| NormIntoGemm | RmsNorm → GEMM | register limit | ✅ |
| TileLevelFusion | 前驱嵌入 GEMM MC 循环 | L1 capacity (75%) | ✅ |
| ComputeRoot | 前驱完整计算驻留 L1 | L1 capacity (75%) | ✅ |
| CrossLayerResidual | Residual Add + RmsNorm | register limit | ✅ |
| Standalone | 不融合 | N/A | ✅ |

### 12 HardwareProfile 定义

| Profile | SIMD | 寄存器 | 融合积极度 | 特化能力 |
|---------|------|--------|-----------|---------|
| CudaSM100 | FP4 native | 256 TMEM | 1.0 | Block Scale + FP4 MMA + TMEM |
| CudaSM90 | WGMMA | 256 regs | 0.95 | TMA + WGMMA + Cluster |
| CudaSM80 | mma.sync | 255 regs | 0.9 | cp.async 双缓冲 |
| CpuAvx10_2 | 256-bit | 31 GPR | 0.85 | VP2INTERSECT + 深链 epilogue |
| CpuAvx512Amx | 512-bit | 32 zmm | 0.85 | AMX tile + AVX-512 epilogue |
| CpuAvx512 | 512-bit | 32 zmm | 0.8 | VNNI + 32 zmm epilogue |
| CpuAvx2 | 256-bit | 16 ymm | 0.6 | 最保守 BLIS |
| AppleM1/M2 | 128-bit + AMX | 32 NEON | 0.8 | NeonAMX tile |
| ArmSME2 | SVE streaming + ZA | 32 SVE | 0.9 | outer product attention |
| ArmSVE2 | 可变长 128-2048 | 32 SVE | 0.75 | predicated 向量 |
| ArmNeon | 128-bit | 32 NEON | 0.6 | 最保守 BLIS |
| Generic | 标量 | — | 0.3 | 无融合 |

---

## REQ 清单

### REQ-FUS-001: Profile 驱动融合分发

FusionEngine 根据当前 HardwareProfile 选择特化融合规则集。

**设计**:
- `FusionEngine::fuse()` 接收 `&DeviceProfile`，通过 `profile.hardware_tier()` 获取设备分级
- 每条 FusionRule 的 `applicable()` 方法查询 profile 能力，而非硬编码 ISA 名
- 融合积极度（0.3~1.0）影响规则触发阈值（如 epilogue 深度、tile 大小）

**关键文件**: `gllm-kernels/src/compiler/fusion.rs`, `hardware_profile.rs`

### REQ-FUS-002: GPU SM80+ Tensor Core 融合

SM80/SM90/SM100 共享 Tensor Core GEMM 融合基础。

**特化策略**:
- GEMM 融合使用 `TileMma` VmInstr（已实现），gpu_lower 生成 wgmma/mma.sync/wmma
- Epilogue 写回：shared memory → global memory 时融合 activation/bias/residual
- FFNBlock 融合：Gate+Up GEMM 共享输入 → SiLU → elementwise mul → Down GEMM 全部在同一 kernel
- `cp.async` 双缓冲：GEMM 计算与下一块 weight prefetch 重叠

**DeviceProfile 驱动参数**:
- `tensor_core_gen`: wgmma vs mma.sync vs wmma
- `shared_memory_bytes`: 影响 tile 大小
- `max_threads_per_block`: 影响 occupancy

**关键文件**: `gpu_lower.rs`, `fusion.rs`

### REQ-FUS-003: GPU SM90 TMA 融合

Hopper 特有 TMA (Tensor Memory Accelerator) 融合。

**特化策略**:
- KV Cache 离散读取 → TMA 2D prefetch（禁止 `LDG`）
- Thread Block Cluster → L2 multicast（Q/K/V 共享 weight）
- WGMMA 异步矩阵乘 → 与 epilogue 计算重叠
- TMA 2D 兼容布局：128-byte aligned shared memory tile

**DeviceProfile 条件**: `tensor_core_gen >= 90 && has_tma == true`

**关键文件**: `gpu_lower.rs`, `fusion.rs`, `hw_constraints.rs`

### REQ-FUS-004: GPU SM100 FP4 融合

Blackwell FP4/F6 原生精度融合。

**特化策略**:
- Block Scale + FP4 native MMA（不模拟、不溢出防范）
- TMEM (Tensor Memory) 寄存器文件用于矩阵累加
- `tcgen05.mma` 指令直接操作 FP4/F6 weight

**DeviceProfile 条件**: `tensor_core_gen >= 100 && has_native_fp4 == true`

**关键文件**: `gpu_lower.rs`, `fusion.rs`

### REQ-FUS-005: SME2 outer product 融合

ARM SME2 ZA array outer product attention。

**特化策略**:
- ZA array 2D 存储：消除显式 tile 管理节点
- Outer product 直接做 attention：`FMOPA` (outer product accumulate)
- Streaming SVE 模式切换：在层边界切换（层内保持 streaming 模式）
- 融合链：`SVE2_RmsNorm → SME2_tile_load_QKV → RoPE(SVE2) → SME2_outer_product_QK → Softmax → SME2_outer_product_AV → Epilogue → Direct_KV_Write`

**DeviceProfile 条件**: `isa == Sve2 && has_sme2 == true`

**关键文件**: `aarch64_lower.rs`, `fusion.rs`

### REQ-FUS-006: AVX-512+AMX tile 融合

Intel Sapphire Rapids+ AMX tile GEMM。

**特化策略**:
- AMX `TDPBF16PS` 替代 BLIS 微内核（BF16 8×8 tile GEMM）
- AVX-512 做 epilogue（32 zmm 无溢出，支持 8-op epilogue）
- `TILECONFIG` → `TDPBF16PS` → `TILESTORE` 流水线
- 量化 GEMM：`TDPBSSD` (INT8) / `TDPBF16PS` (BF16)

**DeviceProfile 条件**: `isa == Avx512Amx`

**关键文件**: `x86_lower.rs`, `fusion.rs`, `hw_constraints.rs`

### REQ-FUS-007: AVX10.2 深链 epilogue

AVX10.2 + APX (31 GPR) 的深链融合。

**特化策略**:
- 31 GPR 允许 ≥8 ops epilogue（无溢出到栈）
- `VPCOMPRESSD` 硬件化 sparse mask（VP2INTERSECT 替代软件实现）
- P-core 全速 / E-core 降级为标量
- 融合链：`RmsNorm → Gate_GEMM(VNNI-256) → SiLU → GateSkip(vcompressps,31GPR) → Up_GEMM → FusedMulAdd → Down_GEMM → Residual_Add`

**DeviceProfile 条件**: `isa == Avx10_2 && gpr_count >= 31`

**关键文件**: `x86_lower.rs`, `fusion.rs`

### REQ-FUS-008: 量化感知融合

VNNI/SVE2 dot product + 量化 GEMM 融合链。

**特化策略**:
- x86: `VPDPBF16PS` / `VPDPBUSD` (VNNI) 量化 GEMM → 累加到 F32 → epilogue 融合
- ARM: `SDOT` / `UDOT` (SVE2) 量化 GEMM
- GPU: tensor core 自动处理 W4A4/W4A8 精度
- 量化 GEMM 的 epilogue 需要 dequant step（累加后乘 scale + zero_point）

**DeviceProfile 驱动参数**:
- `has_vnni`: x86 VNNI 支持
- `has_sve2_dot`: ARM SVE2 integer dot
- `tensor_core_gen`: GPU 量化精度

**关键文件**: `x86_lower.rs`, `aarch64_lower.rs`, `gpu_lower.rs`, `fusion.rs`

### REQ-FUS-009: 融合代价模型

Roofline + register pressure + cache hierarchy 综合评估。

**设计**:
- 当前 `RooflineModel` 只考虑 compute/memory bound
- 扩展为 `FusionCostModel`：
  - **Compute ROI**: 融合节省的 load/store 指令数 vs 增加的 register pressure
  - **Cache ROI**: 融合后的 working set 是否 fit L1/L2
  - **Latency ROI**: 融合是否减少 kernel launch overhead（GPU）或 pipeline stall（CPU）
- 每个 HardwareProfile 有不同的 cost 权重

**关键文件**: `fusion.rs`, `hardware_profile.rs`, `hw_constraints.rs`

---

## 融合拓扑矩阵 (02-ARCHITECTURE.md §13.12)

### Attention 融合拓扑

| Profile | GEMM 方法 | Attention 方法 | Epilogue |
|---------|----------|---------------|----------|
| SM100+ | tcgen05.mma + block_scale | TMA + TMEM prefetch | Epilogue 内联 |
| SM90 | WGMMA 16×16×64 | TMA + Thread Block Cluster | Epilogue 内联 |
| SM80 | mma.sync 16×8×16 | cp.async 双缓冲 | Epilogue 内联 |
| SM70 | wmma 16×16×16 | shared memory | 可选跳过 |
| SME2 | SME2 outer product | FMOPA ZA array | SVE2 内联 |
| SVE2 | BLIS SVE | predicated 向量 | SVE2 内联 |
| AVX10.2+APX | BLIS VNNI-256 | Naive Attn | 31GPR 深链 |
| AVX-512+AMX | AMX tdpbssd | AMX Attn GEMM | 32 zmm |
| AVX-512 | BLIS FP32 | Naive Attn O(n²) | 32 zmm |
| AVX2 | BLIS FP32 | Naive Attn | 16 ymm (栈溢出) |
| NEON | BLIS NEON | Naive Attn | 128-bit |

### FFN 融合拓扑

| Profile | 融合链 | 关键特性 |
|---------|--------|---------|
| SM100+ | Gate_GEMM → SiLU → Up → Mul → Down (全 tensor core) | FP4 native + Block Scale |
| SME2 | RmsNorm → Gate(SME2) → SiLU → Up(SME2) → Down(SME2) | ZA array tile 管理 |
| AVX10.2 | RmsNorm → Gate(VNNI-256) → SiLU → GateSkip(vcompressps) → Up → Down | 31 GPR 8-op epilogue |
| AVX-512+AMX | RmsNorm → AMX_Gate → SiLU → AMX_Up → Mul → AMX_Down | AMX tile 替代 BLIS |
| AVX2 | Gate(BLIS) → SiLU → Up(BLIS) → Mul → Down(BLIS) | 16 ymm 最保守 |

---

## 验证

```bash
# 融合决策测试：给定 HardwareProfile，验证融合规则选择
cd ../gllm-kernels && cargo test --lib fusion

# 硬件约束测试
cargo test --lib hw_constraints

# Roofline 代价模型测试
cargo test --lib hardware_profile

# E2E 融合正确性：每个 Profile 的融合图 vs scalar reference
# (需要对应硬件或模拟器)
```
