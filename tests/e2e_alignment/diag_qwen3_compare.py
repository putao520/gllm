#!/usr/bin/env python3
"""对比 gllm Q4_0 layer0 dump (/tmp/qwen3_golden_layer0/*.bin) vs transformers golden (.npz).
逐算子 cosine, 定位首个发散(根因算子).

gllm dump: 每 .bin 是 [seq_len, dim] f32 (可能有 8 字节 header, 需探测).
golden: golden_qwen3_0.6b.npz, [1, seq_len, dim].
"""
import numpy as np

golden = np.load("tests/e2e_alignment/data/golden_qwen3_0.6b.npz")
DUMP = "/tmp/qwen3_golden_layer0"

def load_dump(name):
    """gllm dump f32, 探测 header(8B) vs 无 header."""
    raw = open(f"{DUMP}/{name}.bin","rb").read()
    n_f32 = len(raw) // 4
    a = np.frombuffer(raw, dtype=np.float32)
    # seq_len=5. 试无 header: 5×dim; 试 8B header(2 f32): skip 2
    # 探测: 找 dim 使 n_f32 == 5*dim 或 n_f32-2 == 5*dim
    for skip in [0, 2]:
        sub = a[skip:]
        if len(sub) % 5 == 0:
            dim = len(sub) // 5
            if dim in [1024, 2048]:
                return sub.reshape(5, dim)
    return a  # fallback

def cos(a, b):
    a = a.flatten().astype(np.float64); b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return float('nan')
    return float(np.dot(a, b) / (na * nb))

# 算子映射: dump name → golden key
probes = [
    ("embedding", "post_embed"),
    ("input_norm", "post_input_norm"),
    ("q_proj", "post_q_proj"),
    ("q_norm", "post_q_norm"),
    ("k_proj", "post_k_proj"),
    ("k_norm", "post_k_norm"),
    ("o_proj", "post_o_proj"),
    ("layer0_out", "post_layer0"),
]

print(f"{'probe':<16} {'cosine':<12} {'gllm_norm':<12} {'golden_norm':<12} {'verdict'}")
print("-"*70)
first_bad = None
for dump_name, golden_key in probes:
    g = load_dump(dump_name)
    ref = golden[golden_key][0]  # [seq,dim]
    if g.shape != ref.shape:
        print(f"{dump_name:<16} SHAPE MISMATCH gllm={g.shape} golden={ref.shape}")
        continue
    c = cos(g, ref)
    gn = np.linalg.norm(g); rn = np.linalg.norm(ref)
    verdict = "PASS" if c > 0.99 else "*** DIVERGE ***"
    if c <= 0.99 and first_bad is None:
        first_bad = dump_name
    print(f"{dump_name:<16} {c:<12.6f} {gn:<12.4f} {rn:<12.4f} {verdict}")

print(f"\n=== 首个发散算子: {first_bad} ===")
if first_bad:
    g = load_dump(first_bad); ref = golden[ {"embedding":"post_embed","input_norm":"post_input_norm","q_proj":"post_q_proj","q_norm":"post_q_norm","k_proj":"post_k_proj","k_norm":"post_k_norm","o_proj":"post_o_proj","layer0_out":"post_layer0"}[first_bad] ][0]
    print(f"gllm {first_bad} row0 first5:  {g[0,:5]}")
    print(f"golden {first_bad} row0 first5:{ref[0,:5]}")
    print(f"gllm norm={np.linalg.norm(g[0]):.4f} golden norm={np.linalg.norm(ref[0]):.4f}")
