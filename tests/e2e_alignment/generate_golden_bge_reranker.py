#!/usr/bin/env python3
"""Generate golden rerank scores for bge-reranker-v2-m3 (BAAI/bge-reranker-v2-m3).

Uses the cross-encoder model to produce relevance scores for query-document pairs.
Golden data used by test_e2e_reranker.rs for numerical alignment verification.
"""

import torch
from safetensors.torch import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

MODEL = "BAAI/bge-reranker-v2-m3"
OUT_DIR = Path(__file__).parent / "data"

# Same query-doc pairs as test_e2e_reranker.rs
PAIRS = [
    ("What is the capital of France?", "Paris is the capital and most populous city of France."),
    ("What is the capital of France?", "Berlin is the capital and largest city of Germany."),
    ("What is the capital of France?", "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
]


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, torch_dtype=torch.float32
    )
    model.eval()

    scores = []
    tensors = {}

    for i, (query, doc) in enumerate(PAIRS):
        encoded = tokenizer(
            query, doc, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encoded)
            # Cross-encoder: single logit score
            score = outputs.logits.squeeze(-1)  # (1,) or scalar
            if score.dim() == 0:
                score = score.unsqueeze(0)
            scores.append(score.item())

        tensors[f"score_{i}"] = score  # (1,)
        tensors[f"input_ids_{i}"] = encoded["input_ids"].squeeze(0).float()

    tensors["num_pairs"] = torch.tensor([len(PAIRS)], dtype=torch.int32)

    # Self-check
    for name, t in tensors.items():
        assert t.isfinite().all(), f"Non-finite values in {name}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_bge_reranker_v2_m3.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    print("\nGolden scores:")
    for i, (query, doc) in enumerate(PAIRS):
        print(f"  [{i}] score={scores[i]:.6f}")
        print(f"       query: {query}")
        print(f"       doc:   {doc[:80]}...")

    # Sanity: relevant pair should score higher than irrelevant
    print(f"\nScore[0] (relevant) > Score[1] (irrelevant): {scores[0] > scores[1]}")


if __name__ == "__main__":
    main()
