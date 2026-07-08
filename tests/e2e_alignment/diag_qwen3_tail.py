#!/usr/bin/env python3
"""对比 gllm Q4_0 layer27/tail dump vs golden full(28层+logits).
定位 layer0(cosine=1.0)之后的发散点: layer循环? final_norm? lm_head?
"""
import numpy as np

golden = np.load("tests/e2e_alignment/data/golden_qwen3_full.npz")
# golden: hs_0(embed)..hs_28(layer27 out) + logits
# gllm layer.* 是 layer27(最后循环)的值, final_normed 是 final norm 输出

def load_gllm(name):
    raw = open(f"/tmp/qwen3_gllm_{name}.bin","rb").read()
    a = np.frombuffer(raw, dtype=np.float32)
    # header 2 f32 (seq=5, dim), 然后 5*dim
    seq = int(a[0]); dim = int(a[1])
    data = a[2:2+seq*dim]
    return data.reshape(seq, dim)

def cos(a, b):
    a=a.flatten().astype(np.float64); b=b.flatten().astype(np.float64)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return float('nan')
    return float(np.dot(a,b)/(na*nb))

# gllm layer.* = layer27(最后循环)输出. golden hs_28 = layer27 out
# layer.ffn_resid (attn+ffn 后, 下一层输入) 对应 golden hs_28
# final_normed 对应 golden final_norm 输出(=lm_head 输入)
print("=== gllm layer27(最后循环) tail vs golden ===")
ref_hs28 = golden['hs_28'][0]  # (5,1024)
for gllm_name, desc in [("layer_ffn_resid","layer27 ffn_resid(=layer27 out)"),("final_normed","final norm out"),("layer_normed","layer27 input_normed"),("layer_o","layer27 attn o"),("layer_attn_resid","layer27 attn residual")]:
    try:
        g = load_gllm(gllm_name)
        # gllm dim=2048(含相邻), 取前 1024
        g1024 = g[:, :1024] if g.shape[1] >= 1024 else g
        c = cos(g1024, ref_hs28)
        print(f"{desc:<35} vs golden hs_28: cosine={c:.6f} {'PASS' if c>0.99 else '*** DIVERGE ***'}")
    except Exception as e:
        print(f"{desc}: {e}")

print("\n=== gllm final_normed vs golden final(需 hook final_norm) ===")
# golden 无 final_norm 单独, 但 hs_28 = pre-final, logits=post-final+lm_head
# gllm final_normed 应是 hs_28 经 final_norm 后(=lm_head 输入)
# 用 golden logits 反推不出 final_normed, 跳过(需 golden 加 final hook)

print("\n=== gllm logits vs golden logits ===")
g_logits = np.frombuffer(open("/tmp/qwen3_gllm_logits.bin","rb").read(),dtype=np.float32)
g_logits = g_logits[2:]  # skip header
ref_logits = golden['logits'][0,-1]  # last token
print(f"gllm logits len={len(g_logits)} argmax={int(g_logits.argmax())}")
print(f"golden logits len={len(ref_logits)} argmax={int(ref_logits.argmax())}")
c = cos(g_logits, ref_logits)
print(f"logits cosine={c:.6f} {'PASS' if c>0.99 else '*** DIVERGE ***'}")

# top5 对比
g_top5 = np.argsort(g_logits)[::-1][:5]
r_top5 = np.argsort(ref_logits)[::-1][:5]
print(f"gllm top5={g_top5.tolist()}")
print(f"golden top5={r_top5.tolist()}")
