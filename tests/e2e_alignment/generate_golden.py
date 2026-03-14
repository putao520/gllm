#!/usr/bin/env python3
"""Generate golden embeddings from PyTorch/HuggingFace for cross-language alignment testing.

Usage:
    python generate_golden.py --model intfloat/multilingual-e5-small --output data/golden.safetensors
"""

import argparse
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer


def generate_golden(model_name: str, output_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    sentences = [
        "Hello world",
        "This is a test sentence for alignment verification",
        "跨语言对齐测试",
    ]

    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoded)
        # Use CLS token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    tensors = {}
    for i, sent in enumerate(sentences):
        tensors[f"embedding_{i}"] = embeddings[i]
    # Store sentences as metadata via a dummy tensor (sentence count)
    tensors["num_sentences"] = torch.tensor([len(sentences)], dtype=torch.int32)

    save_file(tensors, output_path)
    print(f"Saved {len(sentences)} golden embeddings to {output_path}")
    print(f"Model: {model_name}")
    print(f"Embedding dim: {embeddings.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate golden embeddings for alignment testing")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small", help="HuggingFace model name")
    parser.add_argument("--output", default="data/golden.safetensors", help="Output safetensors path")
    args = parser.parse_args()
    generate_golden(args.model, args.output)


if __name__ == "__main__":
    main()
