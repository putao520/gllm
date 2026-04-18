#!/usr/bin/env python3
"""
SmolLM2 generator PyTorch 参考 dump — 捕获每层 hidden state,方便和 gllm 逐层对比。
用法: python3 tools/pytorch_ref_smollm2.py --prompt "The capital of France is" --out /tmp/ref
"""
import argparse
import os
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_bin(path, arr, seq_len, feature_dim):
    import numpy as np
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<II", seq_len, feature_dim))
        f.write(arr.tobytes())


def save_tensor(path, t):
    t = t.detach().cpu().numpy()
    if t.ndim == 3:
        _, seq, feat = t.shape
        arr = t[0]
    elif t.ndim == 2:
        seq, feat = t.shape
        arr = t
    elif t.ndim == 1:
        seq, feat = 1, t.shape[0]
        arr = t.reshape(1, -1)
    else:
        return
    write_bin(path, arr, seq, feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="/tmp/pytorch_ref_smollm2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    ids = tok.encode(args.prompt, return_tensors="pt", add_special_tokens=False)
    seq_len = ids.shape[1]
    print(f"[ref] seq_len={seq_len}, ids={ids[0].tolist()}")

    captures = {}

    def hook_factory(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().cpu()
        return hook

    hooks = []
    # Llama-style: model.model is LlamaModel with embed_tokens + layers[] + norm
    llm = model.model
    hooks.append(llm.embed_tokens.register_forward_hook(hook_factory("embed")))
    for i, layer in enumerate(llm.layers):
        hooks.append(layer.input_layernorm.register_forward_hook(hook_factory(f"layer_{i}_input_norm")))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_factory(f"layer_{i}_q_proj")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_factory(f"layer_{i}_k_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_factory(f"layer_{i}_v_proj")))
        hooks.append(layer.self_attn.register_forward_hook(hook_factory(f"layer_{i}_attn")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_factory(f"layer_{i}_o_proj")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(hook_factory(f"layer_{i}_post_norm")))
        hooks.append(layer.mlp.register_forward_hook(hook_factory(f"layer_{i}_mlp_out")))
        hooks.append(layer.register_forward_hook(hook_factory(f"layer_{i}_output")))
    hooks.append(llm.norm.register_forward_hook(hook_factory("final_norm")))
    hooks.append(model.lm_head.register_forward_hook(hook_factory("lm_head")))

    with torch.no_grad():
        out = model(ids)

    for h in hooks:
        h.remove()

    for name, t in captures.items():
        path = os.path.join(args.out, f"{name}.bin")
        save_tensor(path, t)

    # Also save logits last row
    logits = out.logits[0, -1]
    top = torch.topk(logits, 10)
    print("Top-10 logits at last position:")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {i} [{tok.decode([i])!r}] = {v:.4f}")

    # Greedy decode 5 steps for reference
    cur = ids.clone()
    for step in range(5):
        with torch.no_grad():
            o = model(cur)
        nxt = o.logits[0, -1].argmax().item()
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
        print(f"  step {step}: next={nxt} [{tok.decode([nxt])!r}]")

    print(f"[ref] dumped {len(captures)} tensors to {args.out}")


if __name__ == "__main__":
    main()
