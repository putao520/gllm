#!/usr/bin/env python3
"""Generate golden reference for e5-small-v2 layer 0 per-op verification.

Captures embedding output, attention intermediates, and layer 0 output.
"""
import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import numpy as np

MODEL = "intfloat/e5-small-v2"
TEXT = "Hello, world!"
OUT_DIR = Path(__file__).parent / "data"


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    encoded = tokenizer(TEXT, padding=False, truncation=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"Token IDs: {input_ids.tolist()[0]} (len={input_ids.shape[1]})")

    tensors = {}
    tensors["input_ids"] = input_ids.squeeze(0).float()
    tensors["token_type_ids"] = encoded.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0).float()

    # Hook into layer 0
    layer0 = model.encoder.layer[0]
    captures = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captures[name] = output[0].detach()
            else:
                captures[name] = output.detach()
        return hook

    # Attention self projections
    layer0.attention.self.register_forward_hook(make_hook("attn_self_output"))
    # Attention output dense
    layer0.attention.output.dense.register_forward_hook(make_hook("attn_output_dense"))
    # Attention output LayerNorm
    layer0.attention.output.LayerNorm.register_forward_hook(make_hook("attn_output_layernorm"))
    # Intermediate dense
    layer0.intermediate.dense.register_forward_hook(make_hook("intermediate_dense"))
    # Output dense (FFN down)
    layer0.output.dense.register_forward_hook(make_hook("output_dense"))
    # Output LayerNorm
    layer0.output.LayerNorm.register_forward_hook(make_hook("output_layernorm"))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # Embedding output (before any encoder layer)
    embed_out = outputs.hidden_states[0].squeeze(0)
    tensors["embed_output"] = embed_out
    print(f"  embed_output: shape={embed_out.shape}, first4={embed_out[0, :4].tolist()}")
    print(f"  embed_output norm per row: {[f'{embed_out[i].norm():.4f}' for i in range(embed_out.shape[0])]}")

    # Layer 0 output
    layer0_out = outputs.hidden_states[1].squeeze(0)
    tensors["layer0_output"] = layer0_out
    print(f"  layer0_output: shape={layer0_out.shape}, first4={layer0_out[0, :4].tolist()}")
    print(f"  layer0_output norm: {layer0_out.norm():.4f}")

    # Captured intermediates
    for name, t in captures.items():
        tensors[name] = t
        print(f"  {name}: shape={t.shape}, first3={t[0, :3].tolist()}")

    # Also save raw attention weights for GEMM verification
    q_w = layer0.attention.self.query.weight.data
    k_w = layer0.attention.self.key.weight.data
    v_w = layer0.attention.self.value.weight.data
    o_w = layer0.attention.output.dense.weight.data
    tensors["L0_q_weight"] = q_w
    tensors["L0_k_weight"] = k_w
    tensors["L0_v_weight"] = v_w
    tensors["L0_o_weight"] = o_w

    # Biases
    tensors["L0_q_bias"] = layer0.attention.self.query.bias.data
    tensors["L0_k_bias"] = layer0.attention.self.key.bias.data
    tensors["L0_v_bias"] = layer0.attention.self.value.bias.data
    tensors["L0_o_bias"] = layer0.attention.output.dense.bias.data

    # LayerNorm weights
    tensors["L0_attn_ln_weight"] = layer0.attention.output.LayerNorm.weight.data
    tensors["L0_attn_ln_bias"] = layer0.attention.output.LayerNorm.bias.data
    tensors["L0_output_ln_weight"] = layer0.output.LayerNorm.weight.data
    tensors["L0_output_ln_bias"] = layer0.output.LayerNorm.bias.data

    # FFN weights
    tensors["L0_up_weight"] = layer0.intermediate.dense.weight.data
    tensors["L0_up_bias"] = layer0.intermediate.dense.bias.data
    tensors["L0_down_weight"] = layer0.output.dense.weight.data
    tensors["L0_down_bias"] = layer0.output.dense.bias.data

    # Manual Q projection verification: Q = embed_output @ q_w^T + q_bias
    q_manual = embed_out @ q_w.T + layer0.attention.self.query.bias.data
    print(f"\nManual Q projection: shape={q_manual.shape}, first3={q_manual[0, :3].tolist()}")

    # Embedding details
    embeddings = model.embeddings
    print(f"\nEmbedding details:")
    print(f"  word_embeddings weight shape: {embeddings.word_embeddings.weight.shape}")
    if hasattr(embeddings, 'position_embeddings') and embeddings.position_embeddings is not None:
        print(f"  position_embeddings weight shape: {embeddings.position_embeddings.weight.shape}")
    if hasattr(embeddings, 'token_type_embeddings') and embeddings.token_type_embeddings is not None:
        print(f"  token_type_embeddings weight shape: {embeddings.token_type_embeddings.weight.shape}")
    print(f"  LayerNorm weight shape: {embeddings.LayerNorm.weight.shape}")

    # Manual embedding verification
    word_emb = embeddings.word_embeddings(input_ids).squeeze(0)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    pos_emb = embeddings.position_embeddings(position_ids).squeeze(0) if embeddings.position_embeddings else 0
    type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))
    type_emb = embeddings.token_type_embeddings(type_ids).squeeze(0) if embeddings.token_type_embeddings else 0

    combined = word_emb + pos_emb + type_emb
    ln_out = torch.nn.functional.layer_norm(combined, [combined.shape[-1]],
        embeddings.LayerNorm.weight.data, embeddings.LayerNorm.bias.data, eps=1e-12)

    tensors["manual_embed_combined"] = combined
    tensors["manual_embed_layernorm"] = ln_out

    print(f"\nManual embedding:")
    print(f"  word_emb[0] first3: {word_emb[0, :3].tolist()}")
    print(f"  pos_emb[0] first3: {pos_emb[0, :3].tolist() if not isinstance(pos_emb, int) else 'N/A'}")
    print(f"  type_emb[0] first3: {type_emb[0, :3].tolist() if not isinstance(type_emb, int) else 'N/A'}")
    print(f"  combined[0] first3: {combined[0, :3].tolist()}")
    print(f"  ln_out[0] first3: {ln_out[0, :3].tolist()}")
    print(f"  model embed_output[0] first3: {embed_out[0, :3].tolist()}")
    diff = (ln_out - embed_out).abs().max().item()
    print(f"  max diff (manual vs model): {diff:.2e}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_e5_layer0.safetensors"
    save_file(tensors, str(out_path))
    print(f"\nSaved {len(tensors)} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
