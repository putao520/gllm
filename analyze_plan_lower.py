#!/usr/bin/env python3
import re

# 读取文件内容
with open('.claude/worktrees/gllm-kernels/src/compiler/codegen/vm/plan_lower.rs', 'r') as f:
    content = f.read()

# 获取所有emit函数
emit_functions = re.findall(r'^fn emit_\w+\(', content, re.MULTILINE)

# 存储结果
results = []

# 检查每个emit函数
for func in emit_functions:
    func_name = func[3:-1]  # 去掉 "fn " 和 "("
    pattern = r'fn ' + re.escape(func_name) + r'\([^)]*\)\s*[^{]*\{'
    match = re.search(pattern, content)
    if not match:
        continue
    
    func_start = match.end()
    # 找到下一个函数定义或文件结尾作为函数结束
    next_func = re.search(r'^fn \w+\(', content[func_start:], re.MULTILINE)
    if next_func:
        func_end = func_start + next_func.start()
    else:
        func_end = len(content)
    
    func_body = content[func_start:func_end]
    
    # 检查是否包含auto_lower_trace
    has_auto_lower_trace = 'auto_lower_trace' in func_body
    
    # 检查是否包含手写prog.emit
    has_manual_prog_emit = 'prog.emit(' in func_body
    
    # 统计手写prog.emit数量
    manual_prog_emit_count = func_body.count('prog.emit(')
    
    # 检查是否包含注释（用于判断是否死代码）
    has_dead_code_comment = '// dead code' in func_body or '死代码' in func_body
    
    results.append({
        'function': func_name,
        'has_auto_lower_trace': has_auto_lower_trace,
        'has_manual_prog_emit': has_manual_prog_emit,
        'manual_prog_emit_count': manual_prog_emit_count,
        'has_dead_code_comment': has_dead_code_comment,
        'func_start': func_start,
        'func_end': func_end
    })

# 输出结果
print("=== 问题1: 手写VmInstr的旧代码是否被替换但没删除？ ===\n")
both_count = 0
dead_code_count = 0

for result in results:
    if result['has_auto_lower_trace'] and result['has_manual_prog_emit']:
        both_count += 1
        if result['has_dead_code_comment']:
            dead_code_count += 1
        print(f"函数: {result['function']}")
        print(f"  - 同时使用 auto_lower_trace 和手写 prog.emit")
        print(f"  - 手写 prog.emit 数量: {result['manual_prog_emit_count']}")
        print(f"  - 有死代码注释: {result['has_dead_code_comment']}")
        print()

print(f"总结:")
print(f"- 共有 {both_count} 个emit函数同时使用 auto_lower_trace 和手写 prog.emit")
print(f"- 其中有 {dead_code_count} 个包含死代码注释")

# 现在按功能分类所有手写 VmInstr
print("\n=== 问题2: 460处手写VmInstr的分类 ===\n")

# 读取所有prog.emit调用
prog_emits = re.finditer(r'prog\.emit\((VmInstr::\w+)[^)]*\)', content)

categories = {
    'A类': { 'name': '纯算术运算', 'ops': [], 'count': 0 },  # VecBinOp, Fma, Broadcast等
    'B类': { 'name': '内存操作', 'ops': [], 'count': 0 },   # VecLoad, VecStore等
    'C类': { 'name': '控制流', 'ops': [], 'count': 0 },     # LoopBegin, LoopEnd等
    'D类': { 'name': '指针计算', 'ops': [], 'count': 0 }     # LoadPtr, IntMulStride, ScalarLoad等
}

# 定义各类别的操作
arithmetic_ops = {'VecBinOp', 'Fma', 'Broadcast', 'VecUnaryOp', 'HReduce', 'ScalarUnaryOp'}
memory_ops = {'VecLoad', 'VecStore', 'ScalarLoad'}
control_ops = {'LoopBegin', 'LoopEnd'}
pointer_ops = {'LoadPtr', 'IntMulStride', 'ScalarLoad', 'StorePtr'}

for match in prog_emits:
    instr_type = match.group(1)
    if instr_type in arithmetic_ops:
        categories['A类']['ops'].append(instr_type)
        categories['A类']['count'] += 1
    elif instr_type in memory_ops:
        categories['B类']['ops'].append(instr_type)
        categories['B类']['count'] += 1
    elif instr_type in control_ops:
        categories['C类']['ops'].append(instr_type)
        categories['C类']['count'] += 1
    elif instr_type in pointer_ops:
        categories['D类']['ops'].append(instr_type)
        categories['D类']['count'] += 1

# 输出分类结果
for category, data in categories.items():
    print(f"{category} ({data['name']}): {data['count']} 处")
    if data['ops']:
        op_counts = {}
        for op in data['ops']:
            op_counts[op] = op_counts.get(op, 0) + 1
        for op, count in op_counts.items():
            print(f"  - {op}: {count} 处")
    print()

print("分析建议:")
print("- A类: 应该用 auto_lower_trace 替代")
print("- B类: 需要 dtype 但无法用 TraceOp 表达")
print("- C类: TraceOp 无法表达，必须在 emit 层手写 emit_loop")
print("- D类: 部分已有 TraceOp 语义，部分缺失")
