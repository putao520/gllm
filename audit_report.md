
# ISA Lowering 层 dtype 审计报告

## 1. VmInstr 定义情况

在  中：

- ✅  (第 326-332 行) - 包含  字段
- ✅  (第 334-340 行) - 包含  字段  
- ✅  (第 342-347 行) - 包含  字段
- ✅  (第 452-458 行) - 包含  字段
- ✅  (第 491-497 行) - 包含  字段

## 2. x86_lower.rs 审计结果

### 完全忽略 dtype 的指令 (生成固定 F32 指令):

#### VecBinOp (第 980-1000 行)

**问题**: 所有  完全忽略 ，无论 BF16、F16 还是 F32 都生成相同的 F32 指令（VMULPS, VADDPS 等）

#### VecLoad (第 876-892 行)

**问题**: 使用  (F32 加载指令)，忽略 

#### VecStore (第 920-940 行)

**问题**: 使用  (F32 存储指令)，忽略 

#### Fma (第 1114-1138 行)

**问题**: 使用  (F32 FMA 指令)，忽略 

#### Broadcast (第 948-960 行)

**问题**: 使用  (F32 广播指令)，忽略 

### 正确使用 dtype 的指令:

#### TileMma (第 1456-1516 行)

**正确**: AMX TileMma 根据不同的 dtype 生成对应的 TDP* 指令

## 3. aarch64_lower.rs 审计结果

### 完全忽略 dtype 的指令:

#### VecBinOp (第 1438-1450 行)

**问题**: 所有 VecBinOp 使用固定的 F32 指令

#### VecLoad (第 1345-1365 行)

**问题**: 使用  (F32 加载指令)，忽略 

#### VecStore (第 1370-1380 行)

**问题**: 使用  (F32 存储指令)，忽略 

#### Fma (第 1510-1529 行)

**问题**: 使用固定的 F32 FMA 指令

#### Broadcast (第 1395-1405 行)

**问题**: 使用  (F32 广播指令)，忽略 

## 4. gpu_lower.rs 审计结果

### 完全忽略 dtype 的指令:

#### VecBinOp (第 480-504 行)

**问题**: 所有 VecBinOp 生成固定的  操作，忽略 

#### VecLoad (第 335-350 行)

**问题**: 使用  加载，忽略 

#### VecStore (第 357-370 行)

**问题**: 使用  存储，忽略 

#### Fma (第 556-566 行)

**问题**: 使用  FMA，忽略 

#### Broadcast (第 443-457 行)

**问题**: 使用  广播，忽略 

## 5. 总结

### 严重问题: 大部分 ISA lowering 层完全忽略 dtype

**x86_lower.rs:**
- ✅ TileMma - 正确使用 dtype
- ❌ VecBinOp, VecLoad, VecStore, Fma, Broadcast - 完全忽略 dtype，始终生成 F32 指令

**aarch64_lower.rs:**
- ❌ VecBinOp, VecLoad, VecStore, Fma, Broadcast - 完全忽略 dtype，始终生成 F32 指令

**gpu_lower.rs:**
- ❌ VecBinOp, VecLoad, VecStore, Fma, Broadcast - 完全忽略 dtype，始终生成 F32 指令

### 具体问题:

1. **BF16 数据处理**：即使输入是 BF16，ISA 层也会使用 F32 指令（如 VMULPS），导致不必要的精度转换
2. **F16 数据处理**：F16 数据会被错误地当作 F32 处理
3. **性能损失**：无法利用 BF16 专用指令（如 VDPBF16PS）或 F16 专用指令（如 VFMULCPH）

### 建议:

需要在各个 lowering 文件中添加对  的检查和对应的指令选择：

- x86: BF16 → VDPBF16PS, VDPBF16PS, VDPBF16PS
- x86: F16 → VFMULCPH, VFADDNHPS, VFCVTN2PS2PH
- ARM: BF16 → VFMLA/VFMLAL, VFMLAL, VFMUL
- ARM: F16 → VFMLA/VFMLAL, VFMLAL, VFMUL
- GPU: 支持 bf16.f16 类型转换和相应操作

