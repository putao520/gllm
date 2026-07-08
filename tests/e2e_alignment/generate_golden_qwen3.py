#!/usr/bin/env python3
"""Qwen3-0.6B layer0 golden: hook 各算子输出导出, 供 gllm Q4_0 对比定位发散点.

Qwen3 架构(0.6B): hidden=1024, 16 q-heads, 8 kv-heads(GQA 2:1), head_dim=128,
q_out=2048(!=hidden), k/v_out=1024, rope_theta=1M, 无 partial_rotary(全RoPE),
q_norm/k_norm=HeadRmsNorm(逐head RMS + learned weight[128]).

head_dim 解耦(q=2048≠hidden=1024)是 SmolLM2 没有的, 嫌疑最高.
"""
import torch, numpy as np, json, struct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

model_path = "/home/putao/.gllm/models/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.eval()

# 用与 gllm 相同 prompt
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt")["input_ids"]
print(f"prompt ids: {ids[0].tolist()}, len={ids.shape[1]}")

# hook layer0 各算子
layer0 = model.model.layers[0]
captures = {}
def mk_hook(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captures[name] = o.detach().to(torch.float32).cpu().numpy()
    return hook

# 注册 hook 到关键模块
layer0.input_layernorm.register_forward_hook(mk_hook("post_input_norm"))
layer0.self_attn.q_proj.register_forward_hook(mk_hook("post_q_proj"))
layer0.self_attn.k_proj.register_forward_hook(mk_hook("post_k_proj"))
layer0.self_attn.v_proj.register_forward_hook(mk_hook("post_v_proj"))
# Qwen3 q_norm/k_norm (HeadRmsNorm)
if hasattr(layer0.self_attn, 'q_norm'):
    layer0.self_attn.q_norm.register_forward_hook(mk_hook("post_q_norm"))
if hasattr(layer0.self_attn, 'k_norm'):
    layer0.self_attn.k_norm.register_forward_hook(mk_hook("post_k_norm"))
layer0.self_attn.o_proj.register_forward_hook(mk_hook("post_o_proj"))
# 整层输出
layer0.register_forward_hook(mk_hook("post_layer0"))

with torch.no_grad():
    out = model(ids, output_hidden_states=True)

# hidden_states[0]=embed, [1]=layer0 out
captures["post_embed"] = out.hidden_states[0].to(torch.float32).cpu().numpy()

# 打印各 capture shape + norm(row0 = 第一个 token)
print("\n=== Qwen3 layer0 golden (transformers BF16) ===")
for name in ["post_embed","post_input_norm","post_q_proj","post_q_norm","post_k_proj","post_k_norm","post_o_proj","post_layer0"]:
    if name in captures:
        a = captures[name]
        print(f"{name}: shape={a.shape} row0_norm={np.linalg.norm(a[0,0]):.4f} row0_first5={a[0,0,:5]}")

# 保存 golden 到 npz (numpy, 兼容旧 safetensors)
import os
os.makedirs("tests/e2e_alignment/data", exist_ok=True)
np.savez("tests/e2e_alignment/data/golden_qwen3_0.6b.npz", **captures)
print(f"\nSaved {len(captures)} captures to golden_qwen3_0.6b.npz")

# argmax (对比 gllm prefill_argmax=121034 错误值)
logits = out.logits[0, -1].to(torch.float32).cpu().numpy()
argmax = int(logits.argmax())
print(f"\nprefill last-token argmax={argmax} (期望 Paris~7310)")
top5 = np.argsort(logits)[::-1][:5]
print(f"top5={top5.tolist()}")
