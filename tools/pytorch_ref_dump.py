#!/usr/bin/env python3
"""
PyTorch 参考 dump: 跑 BGE-reranker-v2-m3 在给定 query+doc 下的完整 forward pass,
通过 forward hook 捕获每层每个子模块的 output, dump 到二进制文件。

格式: 4B u32 seq_len + 4B u32 feature_dim + seq*feat*4B f32 raw (与 gllm 一致)

命名对齐 gllm xlmr-reranker YAML 的 node name:
  embed_tok / embed_pos_out / embed_type_out / hidden_0_init
  layer_{i}_q / layer_{i}_q_biased / layer_{i}_k_biased / layer_{i}_v_biased
  layer_{i}_attn_out / layer_{i}_attn_proj / layer_{i}_attn_biased
  layer_{i}_attn_residual / layer_{i}_attn_normed
  layer_{i}_inter / layer_{i}_inter_biased / layer_{i}_act / layer_{i}_out_proj
  layer_{i}_out_biased / layer_{i}_ffn_residual_out / layer_{i}_output_norm (=hidden_0)
  classifier_dense_biased / classifier_tanh_out / rerank_logit
"""
import argparse
import os
import struct
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def write_bin(path: str, arr, seq_len: int, feature_dim: int):
    import numpy as np
    arr = np.ascontiguousarray(arr.astype(np.float32))
    with open(path, "wb") as f:
        f.write(struct.pack("<II", seq_len, feature_dim))
        f.write(arr.tobytes())


def save_tensor(path: str, t):
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
    ap.add_argument("--model", default="/home/putao/.gllm/models/huggingface/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
    ap.add_argument("--out", default="/tmp/pytorch_ref")
    ap.add_argument("--query", required=True)
    ap.add_argument("--doc", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    enc = tok(args.query, args.doc, return_tensors="pt", add_special_tokens=True)
    seq_len = enc["input_ids"].shape[1]
    print(f"[ref] seq_len={seq_len}, ids={enc['input_ids'][0].tolist()}", flush=True)

    # 用 forward hooks 捕获每个 submodule 的 output
    captures = {}

    def hook_factory(name: str):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                t = out[0]
            else:
                t = out
            captures[name] = t.detach().cpu()
        return hook

    hooks = []
    roberta = model.roberta  # XLMRobertaModel

    # Embeddings
    hooks.append(roberta.embeddings.register_forward_hook(hook_factory("hidden_0_init")))

    # 每层
    for i, layer in enumerate(roberta.encoder.layer):
        # layer.attention.self = XLMRobertaSelfAttention; output = (attn_output, ...)
        hooks.append(layer.attention.self.query.register_forward_hook(hook_factory(f"layer_{i:02d}_q")))
        hooks.append(layer.attention.self.key.register_forward_hook(hook_factory(f"layer_{i:02d}_k")))
        hooks.append(layer.attention.self.value.register_forward_hook(hook_factory(f"layer_{i:02d}_v")))
        hooks.append(layer.attention.self.register_forward_hook(hook_factory(f"layer_{i:02d}_attn_out")))
        hooks.append(layer.attention.output.dense.register_forward_hook(hook_factory(f"layer_{i:02d}_attn_proj")))
        hooks.append(layer.attention.output.LayerNorm.register_forward_hook(hook_factory(f"layer_{i:02d}_attn_normed")))
        hooks.append(layer.intermediate.dense.register_forward_hook(hook_factory(f"layer_{i:02d}_inter_biased")))
        hooks.append(layer.intermediate.register_forward_hook(hook_factory(f"layer_{i:02d}_act")))
        hooks.append(layer.output.dense.register_forward_hook(hook_factory(f"layer_{i:02d}_out_biased")))
        hooks.append(layer.output.LayerNorm.register_forward_hook(hook_factory(f"layer_{i:02d}_output_norm")))

    # Classifier
    cls = model.classifier
    hooks.append(cls.dense.register_forward_hook(hook_factory("classifier_dense_biased")))
    hooks.append(cls.out_proj.register_forward_hook(hook_factory("rerank_logit")))

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    # Add classifier_tanh_out separately (no module for tanh)
    captures["classifier_tanh_out"] = torch.tanh(captures["classifier_dense_biased"])

    for h in hooks:
        h.remove()

    print(f"[ref] captured {len(captures)} tensors", flush=True)
    for name, t in captures.items():
        save_tensor(f"{args.out}/{name}.bin", t)

    print(f"[ref] final logit = {out.logits[0].item():.6f}", flush=True)


if __name__ == "__main__":
    main()
