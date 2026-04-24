# 自动指令选择器架构设计 (Auto Instruction Selector)

## 1. 背景与问题

### 当前架构问题

当前 gllm-kernels 的 codegen 路径中存在**两层手写指令选择**：

1. **plan_lower.rs**：31 个 OpKind match arms，每个 op 调用专门的 lower 函数
2. **lower.rs**：25 个专用 lower 函数（lower_gemm, lower_rope, lower_mha 等）
3. **lower_trace_body**：TraceOp → VmInstr 的大型 match 语句（约 40 个 arms）

**问题**：
- 手写 match arms 容易出错（bug 的源头）
- 难以维护和扩展
- 无法自动优化指令选择
- 不符合现代编译器架构（LLVM SelectionDAG/GlobalISel）

### 正确架构

```
Scalar → SymExec → TraceOp → [自动指令选择] → VmInstr → ISA Lowering → Machine Code
                                    ↑
                            Auto Instruction Selector
```

## 2. 设计目标

1. **消除手写 match arms**：从 TraceOp 到 VmInstr 的映射由算法自动完成
2. **模式匹配**：识别 TraceOp 序列中的模式，选择最优 VmInstr 序列
3. **硬件感知**：根据 SIMD 宽度、缓存层级等硬件特性优化
4. **可扩展**：新增 TraceOp 或 VmInstr 不需要修改核心选择器逻辑

## 3. 核心概念

### 3.1 TraceOp 属性

每个 TraceOp 具有以下属性：

```rust
pub enum TraceOp {
    // 算术
    Add(u32, u32),    // Binary, Commutative, Associative
    Mul(u32, u32),    // Binary, Commutative, Associative
    Fma(u32, u32, u32), // Ternary, Associative in accumulation
    
    // 一元
    Exp(u32),         // Unary, Monotonic
    Sqrt(u32),        // Unary, Monotonic
    
    // 内存
    Input(u32),       // Source operand
    Const(f64),       // Constant folding
}
```

### 3.2 VmInstr 属性

每个 VmInstr 具有以下属性：

```rust
pub enum VmInstr {
    VecBinOp { op: VecOp },       // Add, Sub, Mul, Div, Max, Min
    VecUnaryOp { op: VecUnaryOp }, // Neg, Abs, Sqrt, Rsqrt
    Fma { .. },                    // Fused multiply-add
    Transcendental { .. },         // Exp, Log, Tanh
}
```

### 3.3 模式匹配规则

**Pattern → VmInstr 映射表**：

| Pattern | VmInstr | 优化条件 |
|---------|---------|----------|
| `Add(x, Const(0))` | NOP | 常量折叠 |
| `Mul(x, Const(1))` | NOP | 恒等变换 |
| `Mul(Add(a, b), c)` | FMA(a, c, b) | FMA 融合 |
| `Exp(x) + Exp(y)` | FMA path | 指数优化 |
| `Sqrt(Mul(x, x))` | Abs(x) | 代数简化 |

## 4. 架构设计

### 4.1 SelectionDAG 风格

```
TraceOp 序列 → DAG 构建 → 模式匹配 → VmInstr 选择 → 代码生成
```

**阶段 1：DAG 构建**
- TraceOp 序列转换为 DAG
- 识别数据依赖关系
- 计算每个节点的属性（commutative, associative 等）

**阶段 2：模式匹配**
- 自底向上遍历 DAG
- 对每个节点尝试匹配已知模式
- 选择最优的 VmInstr 序列

**阶段 3：代码生成**
- 生成 VmInstr 序列
- 应用局部优化（常量折叠、死代码消除）

### 4.2 Pattern Matcher

```rust
pub struct PatternMatcher {
    rules: Vec<PatternRule>,
}

pub struct PatternRule {
    pattern: Pattern,
    replacement: Replacement,
    condition: Option<Condition>,
}

pub enum Pattern {
    Leaf(OpKind),
    Binary(Box<Pattern>, Box<Pattern>),
    Ternary(Box<Pattern>, Box<Pattern>, Box<Pattern>),
    Wildcard,
}

pub enum Replacement {
    Single(VmInstr),
    Sequence(Vec<VmInstr>),
    Transform(Box<dyn Fn(&Context) -> Vec<VmInstr>>),
}
```

### 4.3 规则示例

```rust
// FMA 融合规则
let fma_rule = PatternRule {
    pattern: Pattern::Binary(
        Box::new(Pattern::Binary(
            Box::new(Pattern::Leaf(OpKind::Mul)),
            Box::new(Pattern::Wildcard),
        )),
        Box::new(Pattern::Wildcard),
    ),
    replacement: Replacement::Single(VmInstr::Fma { .. }),
    condition: Some(Condition::HasFmaHardware),
};

// 常量折叠规则
let const_fold_rule = PatternRule {
    pattern: Pattern::Binary(
        Box::new(Pattern::Wildcard),
        Box::new(Pattern::Leaf(OpKind::Const(0.0))),
    ),
    replacement: Replacement::Nop,
    condition: None,
};
```

## 5. 实施路径

### Phase 1：Pattern Matcher 基础设施
- [ ] 定义 Pattern 和 Replacement 类型
- [ ] 实现 DAG 构建器
- [ ] 实现基础模式匹配算法

### Phase 2：核心规则库
- [ ] 算术规则（常量折叠、结合律重排）
- [ ] FMA 融合规则
- [ ] 指数/对数优化规则
- [ ] 内存访问优化规则

### Phase 3：集成到现有管线
- [ ] 替换 lower_trace_body 中的 match 语句
- [ ] 集成到 plan_lower.rs
- [ ] 保留手写规则作为 fallback

### Phase 4：验证与优化
- [ ] 单元测试（每个规则）
- [ ] 集成测试（现有 E2E 测试）
- [ ] 性能测试（codegen 时间）
- [ ] 数值对齐测试（与手写版本比较）

## 6. 关键挑战

### 6.1 SSA 形式处理
- TraceOp 是 SSA 形式（每个操作定义一个新值）
- VmInstr 也是 SSA（VRegId）
- 需要保持 SSA 不变式

### 6.2 循环处理
- TraceOp 可能在循环内部
- 需要区分循环不变量和循环变量
- 循环展开优化

### 6.3 硬件特性
- 不同 ISA 的指令集差异（AVX2 vs AVX-512）
- SIMD 宽度变化（W128, W256, W512）
- 需要硬件感知的模式匹配

## 7. 参考实现

### 7.1 LLVM SelectionDAG
- DAG 构建：SDNode
- 模式匹配：MatcherTable
- 代码生成：ISel

### 7.2 LLVM GlobalISel
- 更模块化的设计
- 规则可以独立定义和测试
- 更容易扩展

## 8. 验证标准

1. **正确性**：自动选择器生成的 VmInstr 与手写版本数值一致
2. **性能**：不增加 codegen 时间（< 10% overhead）
3. **可维护性**：新增规则不需要修改核心代码
4. **可测试性**：每个规则都有独立的单元测试

## 9. 里程碑

- [M1] Pattern Matcher 原型（1-2 周）
- [M2] 核心规则库（2-3 周）
- [M3] 集成测试通过（1 周）
- [M4] 性能对齐（1 周）
- [M5] 完全替换手写 match arms（2-3 周）

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模式匹配错误 | 正确性 | 保留手写版本作为 reference |
| 性能回归 | 用户体验 | 性能测试 + fallback 机制 |
| 开发时间过长 | 进度 | 分阶段实施，逐步替换 |
