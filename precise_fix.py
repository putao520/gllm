import json
import subprocess
import os

res = subprocess.run(["cargo", "check", "-p", "gllm", "--message-format=json"], capture_output=True, text=True)
edits = {}

for line in res.stdout.splitlines():
    if not line.strip(): continue
    try:
        msg = json.loads(line)
        if msg.get("reason") == "compiler_message" and msg.get("message", {}).get("level") == "error":
            err = msg["message"]
            msg_text = err.get("message", "")
            if "expected `SymDim`" in msg_text or "expected `gllm_kernels::compiler::SymDim`" in msg_text or "expected struct `gllm_kernels::compiler::graph::SymDim`" in msg_text:
                for span in err.get("spans", []):
                    if span.get("is_primary"):
                        file = span["file_name"]
                        line_idx = span["line_start"] - 1
                        # We want to replace the exact text
                        text = span["text"][0]["text"] if span.get("text") else None
                        
                        col_start = span["column_start"] - 1
                        col_end = span["column_end"] - 1

                        if file not in edits: edits[file] = {}
                        if line_idx not in edits[file]: edits[file][line_idx] = []
                        # Store in reverse order to not mess up indices
                        edits[file][line_idx].append({
                            "start": col_start,
                            "end": col_end,
                        })
    except Exception as e: pass

for file, lines in edits.items():
    if not os.path.exists(file): continue
    with open(file, 'r') as f: content = f.readlines()
    
    for l_idx, mod_list in lines.items():
        # sort by start desc so we can insert without shifting previous indices
        mod_list.sort(key=lambda x: x["start"], reverse=True)
        line_str = content[l_idx]
        
        for mod in mod_list:
            start = mod["start"]
            end = mod["end"]
            
            # Extract the raw snippet
            snippet = line_str[start:end]
            
            # If it already has .into(), skip
            if ".into()" in snippet: continue
            
            # If it's a field initialization shorthand like `seq_len,`, we need `seq_len: seq_len.into(),`
            # Wait! The span might cover just `seq_len`.
            if line_str[end] == ',' and line_str[start-2:start] != ': ':
                # Field shorthand!
                line_str = line_str[:start] + f"{snippet}: {snippet}.into()" + line_str[end:]
            else:
                line_str = line_str[:start] + f"{snippet}.into()" + line_str[end:]
                
        content[l_idx] = line_str

    with open(file, 'w') as f: f.writelines(content)
    print(f"Patched {file}")

