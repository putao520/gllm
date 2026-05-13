#!/usr/bin/env python3
"""Generate golden reference for layer 0 per-op verification.

Outputs:
- hidden_layer_0 (after embedding): input to layer 0
- L0_input_norm: output of RmsNorm(hidden_layer_0)
- L0_q_proj: output of q_proj(L0_input_norm) — raw GEMM result
- L0_k_proj: output of k_proj(L0_input_norm)
- L0_v_proj: output of v_proj(L0_input_norm)
"""

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The meaning of life is"
OUT_DIR = Path(__file__).parent / "data"


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"Token IDs: {input_ids.tolist()[0]}")

    tensors = {}
    tensors["input_ids"] = input_ids.squeeze(0).float()

    # Hook into layer 0 to capture intermediate activations
    layer0 = model.model.layers[0]
    captures = {}

    def make_hook(name):
        def hook(module, input, output):
            captures[name] = output.detach()
        return hook

    # Register hooks
    layer0.input_layernorm.register_forward_hook(make_hook("L0_input_norm"))
    layer0.self_attn.q_proj.register_forward_hook(make_hook("L0_q_proj"))
    layer0.self_attn.k_proj.register_forward_hook(make_hook("L0_k_proj"))
    layer0.self_attn.v_proj.register_forward_hook(make_hook("L0_v_proj"))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # hidden_layer_0 = embedding output (before any transformer layer)
    tensors["hidden_layer_0"] = outputs.hidden_states[0].squeeze(0)

    # Captured intermediate activations
    for name, tensor in captures.items():
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        t = tensor.squeeze(0)
        tensors[name] = t
        print(f"  {name}: shape={t.shape}, first4={t[0, :4].tolist()}")

    # Also save raw weight matrices for direct GEMM verification
    q_weight = layer0.self_attn.q_proj.weight.data  # [576, 576]
    tensors["L0_q_weight"] = q_weight
    print(f"  L0_q_weight shape={q_weight.shape}")

    k_weight = layer0.self_attn.k_proj.weight.data  # [192, 576]
    tensors["L0_k_weight"] = k_weight
    print(f"  L0_k_weight shape={k_weight.shape}")

    v_weight = layer0.self_attn.v_proj.weight.data  # [192, 576]
    tensors["L0_v_weight"] = v_weight
    print(f"  L0_v_weight shape={v_weight.shape}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_layer0_ops.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")

    # Manual GEMM verification: C = A @ W^T
    hidden = outputs.hidden_states[0].squeeze(0)  # [5, 576]
    normed = captures["L0_input_norm"].squeeze(0)  # [5, 576]
    q_manual = normed @ q_weight.T  # [5, 576]
    q_hook = captures["L0_q_proj"].squeeze(0)  # [5, 576]
    diff = (q_manual - q_hook).abs().max().item()
    print(f"Manual GEMM vs hook max_diff: {diff:.2e}")

    # hidden_layer_1 (after layer 0 transformer block)
    tensors["hidden_layer_1"] = outputs.hidden_states[1].squeeze(0)

    # Resave with hidden_layer_1
    save_file(tensors, str(out_path))

    # Verify
    for name, t in tensors.items():
        assert t.isfinite().all(), f"Non-finite values in {name}"
    print("All tensors finite ✓")


if __name__ == "__main__":
    main()
