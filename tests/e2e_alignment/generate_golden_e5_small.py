#!/usr/bin/env python3
"""Generate golden embeddings for e5-small-v2 (intfloat/e5-small-v2).

Uses MeanPool (same as gllm) over last_hidden_state to produce pooled embeddings.
Golden data used by test_e2e_embedding.rs for numerical alignment verification.
"""

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

MODEL = "intfloat/e5-small-v2"
OUT_DIR = Path(__file__).parent / "data"

# Same inputs as test_e2e_embedding.rs (processed individually, no padding)
SENTENCES = ["Hello, world!", "Test sentence"]


def mean_pool_unmasked(last_hidden_state):
    """MeanPool over all token positions (no padding mask needed for single-sequence)."""
    return last_hidden_state.mean(dim=1)


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    tensors = {}

    for i, sent in enumerate(SENTENCES):
        # Tokenize individually (no padding) — same as gllm's per-sentence embed
        encoded = tokenizer(sent, padding=False, truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        print(f"  [{i}] {sent!r} → token IDs: {input_ids.tolist()[0]} (len={input_ids.shape[1]})")

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            # MeanPool: average over seq dimension (no mask needed, no padding)
            pooled = mean_pool_unmasked(outputs.last_hidden_state)  # (1, 384)

        tensors[f"embedding_{i}"] = pooled.squeeze(0)  # (384,)

        emb = pooled.squeeze(0).tolist()
        print(f"       first5: {emb[:5]}")
        print(f"       norm:   {sum(x**2 for x in emb)**0.5:.4f}")

    tensors["num_sentences"] = torch.tensor([len(SENTENCES)], dtype=torch.int32)
    tensors["embedding_dim"] = torch.tensor([tensors["embedding_0"].shape[0]], dtype=torch.int32)

    # Self-check
    for name, t in tensors.items():
        assert t.isfinite().all(), f"Non-finite values in {name}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_e5_small_v2.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print first few values for quick sanity check
    for i in range(len(SENTENCES)):
        emb = tensors[f"embedding_{i}"].tolist()
        print(f"  emb[{i}] first5: {emb[:5]}")
        print(f"  emb[{i}] norm: {sum(x**2 for x in emb)**0.5:.4f}")


if __name__ == "__main__":
    main()
