#!/usr/bin/env python3
"""SmolLM2 layer0 发散诊断: 导出 layer0 中间结果, 与 gllm capture 对比.

gllm capture layer0 row0 = [1.94, 2.38, -0.60, 0.99, -1.49] norm=60.7
ref/golden layer0 row0   = [2.44, 0.34, -0.36, 0.79, 1.25]  norm=34.6
→ gllm 发散, norm 偏大 1.75x.

假设测试: 缺 softmax / 缺 attention scale / 缺 RMSNorm / 残差错位.
"""
import torch, json, struct, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

model_path = "/home/putao/.gllm/models/huggingface/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac"
with open("tests/e2e_alignment/data/golden_smollm2_135m.safetensors","rb") as f:
    data = f.read()
n = struct.unpack("<Q", data[:8])[0]
h = json.loads(data[8:8+n].decode()); base = 8+n
def get_golden(name):
    info = h[name]; off = info["data_offsets"][0]+base; cnt = 1
    for s in info["shape"]: cnt*=s
    return np.array([struct.unpack("<f", data[off+i*4:off+i*4+4])[0] for i in range(cnt)]).reshape(info["shape"])
golden_h0 = get_golden("hidden_layer_0")
golden_h1 = get_golden("hidden_layer_1")

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16); model.eval()
ids = tok("The meaning of life is", return_tensors="pt")["input_ids"]
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
l0_ref = out.hidden_states[1].to(torch.float32).detach().numpy()[0]
print("ref l0 row0 first5:", l0_ref[0,:5], "norm:", np.linalg.norm(l0_ref[0]))
print("gllm l0 row0 first5: [1.9386488, 2.383808, -0.6042177, 0.9909464, -1.4935191] norm: 60.7")

# Manual layer0 with hooks to dump intermediates
layer0 = model.model.layers[0]
hidden = torch.tensor(golden_h0, dtype=model.dtype).unsqueeze(0)  # (1,5,576) BF16
position_ids = torch.arange(5).unsqueeze(0)

# Get rotary embedding
rot = model.model.rotary_emb(hidden, position_ids)
cos, sin = rot
cos = cos.unsqueeze(1)  # (1,1,5,64) for head broadcast
sin = sin.unsqueeze(1)
print("rotary cos shape:", cos.shape, "theta check: cos.flatten()[0]=", float(cos.flatten()[0]))

# RMSNorm1
w_in = layer0.input_layernorm.weight.to(model.dtype)
rms1 = torch.sqrt((hidden.float()**2).mean(-1, keepdim=True) + 1e-5).to(model.dtype)
normed1 = hidden / rms1 * w_in
print("rmsnorm1 row0 norm:", float(torch.norm(normed1[0].float())))

# q/k/v proj
q = layer0.self_attn.q_proj(normed1)  # (5, 576) = (5, 9*64)
k = layer0.self_attn.k_proj(normed1)  # (5, 192) = (5, 3*64)
v = layer0.self_attn.v_proj(normed1)  # (5, 192)
print("q row0 norm:", float(torch.norm(q[0].float())), "k row0 norm:", float(torch.norm(k[0].float())), "v row0 norm:", float(torch.norm(v[0].float())))

# reshape to heads
bsz, q_len, _ = q.shape
q = q.view(1, 5, 9, 64).transpose(1, 2)  # (1,9,5,64)
k = k.view(1, 5, 3, 64).transpose(1, 2)  # (1,3,5,64)
v = v.view(1, 5, 3, 64).transpose(1, 2)  # (1,3,5,64)
# repeat k,v for GQA (9/3=3)
k_rep = k.repeat(1, 3, 1, 1)  # (1,9,5,64)
v_rep = v.repeat(1, 3, 1, 1)  # (1,9,5,64)
# RoPE
q_rot, k_rot = apply_rotary_pos_emb(q, k_rep, cos, sin)
print("q_rot row0 head0 first4:", q_rot[0,0,0,:4].float().detach().numpy())

# attention scores
scale = 1.0 / (64.0 ** 0.5)
scores = torch.matmul(q_rot, k_rot.transpose(-2,-1)) * scale  # (1,9,5,5)
print("attn scores shape:", scores.shape, "row0 head0:", scores[0,0,0].float().detach().numpy())
attn_weights = F.softmax(scores, dim=-1)
print("attn weights row0 head0:", attn_weights[0,0,0].float().detach().numpy())
attn_out = torch.matmul(attn_weights, v_rep)  # (1,9,5,64)
print("attn_out row0 head0 first4:", attn_out[0,0,0,:4].float().detach().numpy())

# o_proj
attn_out = attn_out.transpose(1,2).reshape(1,5,576)
o = layer0.self_attn.o_proj(attn_out)  # (1,5,576)
print("o_proj out row0 first5:", o[0,0,:5].float().detach().numpy(), "norm:", float(torch.norm(o[0,0].float())))

# residual
resid1 = hidden + o
print("resid1 (after attn) row0 first5:", resid1[0,0,:5].float().detach().numpy(), "norm:", float(torch.norm(resid1[0,0].float())))

# Test hypothesis: missing softmax (raw scores · V)
attn_no_sm = torch.matmul(scores, v_rep)  # no softmax
o_no_sm = layer0.self_attn.o_proj(attn_no_sm.transpose(1,2).reshape(1,5,576))
print("\n[HYPO] no-softmax o_proj row0 first5:", o_no_sm[0,0,:5].float().detach().numpy(), "norm:", float(torch.norm(o_no_sm[0,0].float())))

# Test hypothesis: missing scale
scores_noscale = torch.matmul(q_rot, k_rot.transpose(-2,-1))
attn_noscale = torch.matmul(F.softmax(scores_noscale, dim=-1), v_rep)
o_noscale = layer0.self_attn.o_proj(attn_noscale.transpose(1,2).reshape(1,5,576))
print("[HYPO] no-scale o_proj row0 first5:", o_noscale[0,0,:5].float().detach().numpy(), "norm:", float(torch.norm(o_noscale[0,0].float())))

# Test hypothesis: no RoPE
attn_norope = torch.matmul(F.softmax(torch.matmul(q, k_rep.transpose(-2,-1))*scale, dim=-1), v_rep)
o_norope = layer0.self_attn.o_proj(attn_norope.transpose(1,2).reshape(1,5,576))
print("[HYPO] no-rope o_proj row0 first5:", o_norope[0,0,:5].float().detach().numpy(), "norm:", float(torch.norm(o_norope[0,0].float())))

print("\ngllm layer0 (full) row0 first5: [1.9386488, 2.383808, -0.6042177, 0.9909464, -1.4935191] norm: 60.7")

# ★ 关键验证: gllm l0 norm (60.7) ≈ ref q_proj norm (61.22) — gllm 是否捕获了 q_proj 而非 layer0?
print("\n=== gllm l0 vs ref q_proj (norm 匹配检查) ===")
# ref q row0 first5 (需要导出)
# 重新算 q_proj row0
normed1_f = normed1[0].float().detach().numpy()
q_row0 = layer0.self_attn.q_proj(normed1[0,0]).float().detach().numpy()
print("ref q_proj row0 first5:", q_row0[:5], "norm:", np.linalg.norm(q_row0))
print("gllm l0 row0 first5: [1.9386488, 2.383808, -0.6042177, 0.9909464, -1.4935191]")
gllm_l0 = np.fromfile('/tmp/gllm_capture_layer0_5token.bin', dtype=np.float32)  # (576,)
# cosine ref q_proj row0 vs gllm l0 row0
cos = float(np.dot(q_row0, gllm_l0) / (np.linalg.norm(q_row0) * np.linalg.norm(gllm_l0)))
print(f"cosine(ref q_proj row0, gllm l0 row0) = {cos:.4f}")
# also k_proj, v_proj
k_row0 = layer0.self_attn.k_proj(normed1[0,0]).float().detach().numpy()
v_row0 = layer0.self_attn.v_proj(normed1[0,0]).float().detach().numpy()
# k_proj/v_proj are 192 elem (3*64), not 576 — skip

# o_proj, resid1, full l0
print(f"cosine(ref o_proj row0, gllm l0 row0) = {float(np.dot(o[0,0].float().detach().numpy(), gllm_l0)/(np.linalg.norm(o[0,0].float().detach().numpy())*np.linalg.norm(gllm_l0))):.4f}")
print(f"cosine(ref resid1 row0, gllm l0 row0) = {float(np.dot(resid1[0,0].float().detach().numpy(), gllm_l0)/(np.linalg.norm(resid1[0,0].float().detach().numpy())*np.linalg.norm(gllm_l0))):.4f}")
print(f"cosine(ref l0 row0, gllm l0 row0) = {float(np.dot(l0_ref[0], gllm_l0)/(np.linalg.norm(l0_ref[0])*np.linalg.norm(gllm_l0))):.4f}")

# gllm capture (5-token) = 最后 token = row4. 比 ref row4.
print("\n=== gllm l0 (5-token last) vs ref row4 ===")
print(f"cosine(ref l0 row4, gllm l0) = {float(np.dot(l0_ref[4], gllm_l0)/(np.linalg.norm(l0_ref[4])*np.linalg.norm(gllm_l0))):.4f}")
print(f"cosine(ref resid1 row4, gllm l0) = {float(np.dot(resid1[0,4].float().detach().numpy(), gllm_l0)/(np.linalg.norm(resid1[0,4].float().detach().numpy())*np.linalg.norm(gllm_l0))):.4f}")
print(f"ref l0 row4 first5: {l0_ref[4,:5]}")
print(f"gllm l0 first5: {gllm_l0[:5]}")
# 也试: gllm l0 是否 = ref l0 row0 的某种变换? 检查 norm
print(f"ref l0 row0 norm: {np.linalg.norm(l0_ref[0])}, row4 norm: {np.linalg.norm(l0_ref[4])}, gllm norm: {np.linalg.norm(gllm_l0)}")
