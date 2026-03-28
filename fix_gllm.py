import json
import subprocess
import os

def run_cargo():
    res = subprocess.run(["cargo", "check", "-p", "gllm", "--message-format=json"], capture_output=True, text=True)
    return res.stdout

stdout = run_cargo()
replacements = {}
for line in stdout.splitlines():
    if not line.strip(): continue
    try:
        msg = json.loads(line)
        if msg.get("reason") == "compiler_message" and msg.get("message", {}).get("level") == "error":
            err = msg["message"]
            msg_text = err.get("message", "")
            for span in err.get("spans", []):
                if span.get("is_primary"):
                    file_name = span.get("file_name")
                    line_num = span.get("line_start")
                    c1 = span.get("column_start")
                    c2 = span.get("column_end")
                    if file_name not in replacements: replacements[file_name] = {}
                    if line_num not in replacements[file_name]: replacements[file_name][line_num] = []
                    
                    replacements[file_name][line_num].append({
                        "c1": c1 - 1, 
                        "c2": c2 - 1,
                        "msg": msg_text,
                        "text": span.get("text", [{}])[0].get("text", "")
                    })
    except Exception as e: pass

for file, lines in replacements.items():
    if not os.path.exists(file): continue
    with open(file, 'r') as f: file_lines = f.readlines()
    
    for line_num, edits in lines.items():
        idx = line_num - 1
        v_line = file_lines[idx]
        
        for edit in edits:
            msg = edit["msg"]
            if "expected `SymDim`, found" in msg or "expected `SymDim`" in msg:
                # Basic literal match
                v_line = v_line.replace("m: seq_len,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(seq_len),")
                v_line = v_line.replace("m: b_size,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(b_size),")
                v_line = v_line.replace("m: b,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(b),")
                v_line = v_line.replace("m: s,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(s),")
                v_line = v_line.replace("m: b * s,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(b * s),")
                v_line = v_line.replace("m: seq,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(seq),")
                v_line = v_line.replace("m,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(m),")

                v_line = v_line.replace("seq_len: b_size,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(b_size),")
                v_line = v_line.replace("seq_len: seq,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(seq),")
                v_line = v_line.replace("seq_len: _seq,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(_seq),")
                v_line = v_line.replace("seq_len: b,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(b),")
                v_line = v_line.replace("seq_len: s,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(s),")
                v_line = v_line.replace("seq_len: seq_len,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(seq_len),")
                v_line = v_line.replace("seq_len,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(seq_len),")

                v_line = v_line.replace("total_seq: max_seq,", "total_seq: gllm_kernels::compiler::graph::SymDim::Concrete(max_seq),")
                v_line = v_line.replace("total_seq: total_seq,", "total_seq: gllm_kernels::compiler::graph::SymDim::Concrete(total_seq),")
                v_line = v_line.replace("total_seq: total_seq + b_size,", "total_seq: gllm_kernels::compiler::graph::SymDim::Concrete(total_seq + b_size),")
                v_line = v_line.replace("total_seq: total_seq_len,", "total_seq: gllm_kernels::compiler::graph::SymDim::Concrete(total_seq_len),")
                
                v_line = v_line.replace("m: 1,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(1),")
                v_line = v_line.replace("m: 0,", "m: gllm_kernels::compiler::graph::SymDim::Concrete(0),")
                v_line = v_line.replace("seq_len: 1,", "seq_len: gllm_kernels::compiler::graph::SymDim::Concrete(1),")
                v_line = v_line.replace("total_seq: 1,", "total_seq: gllm_kernels::compiler::graph::SymDim::Concrete(1),")

        file_lines[idx] = v_line

    with open(file, 'w') as f: f.writelines(file_lines)
    print(f"Patched {file}")

