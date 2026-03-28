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
            err = msg.get("message")
            for span in err.get("spans", []):
                if span.get("is_primary"):
                    label = span.get("label") or ""
                    text = span.get("text", [{}])[0].get("text", "") if span.get("text") else ""
                    if "SymDim" in label or "SymDim" in err.get("message", "") or "expected `SymDim`" in label or "SymDim" in text or "mismatched types" in err.get("message", ""):
                        # Ensure we are actually dealing with our target
                        file = span["file_name"]
                        line_idx = span["line_start"] - 1
                        col_start = span["column_start"] - 1
                        col_end = span["column_end"] - 1

                        if file not in edits: edits[file] = {}
                        if line_idx not in edits[file]: edits[file][line_idx] = []
                        edits[file][line_idx].append({
                            "start": col_start,
                            "end": col_end,
                            "label": label,
                        })
    except Exception as e: pass

for file, lines in edits.items():
    if not os.path.exists(file): continue
    with open(file, 'r') as f: content = f.readlines()
    
    for l_idx, mod_list in lines.items():
        # sort by start time descending
        mod_list.sort(key=lambda x: x["start"], reverse=True)
        line_str = content[l_idx]
        
        for mod in mod_list:
            start = mod["start"]
            end = mod["end"]
            label = mod["label"]
            
            # Use safety checks to ensure we only touch expected SymDim errors
            if "SymDim" not in label and "SymDim" not in "".join(content):
                continue

            snippet = line_str[start:end]
            if ".into()" in snippet or "SymDim" in snippet: continue
            
            if end < len(line_str) and line_str[end] == ',' and line_str[start-2:start] != ': ':
                line_str = line_str[:start] + f"{snippet}: {snippet}.into()" + line_str[end:]
            else:
                line_str = line_str[:start] + f"{snippet}.into()" + line_str[end:]
                
        content[l_idx] = line_str

    with open(file, 'w') as f: f.writelines(content)
    print(f"Patched {file}")

