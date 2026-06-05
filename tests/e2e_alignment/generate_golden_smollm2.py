#!/usr/bin/env python3
"""Generate golden reference data for SmolLM2-135M numerical alignment.

Outputs per-layer hidden states + final logits as safetensors for gllm comparison.
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
    print(f"Prompt: {PROMPT!r}")
    print(f"Token IDs: {input_ids.tolist()[0]}")

    # Forward with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    tensors = {}

    # 1. Input token IDs (for reproducibility)
    tensors["input_ids"] = input_ids.squeeze(0).float()  # (seq_len,)

    # 2. Per-layer hidden states: shape (seq_len, hidden_size)
    num_layers = len(outputs.hidden_states)
    print(f"Layers: {num_layers}")

    for i, hs in enumerate(outputs.hidden_states):
        tensors[f"hidden_layer_{i}"] = hs.squeeze(0)  # (seq_len, hidden_size)

    # 3. Final logits: shape (seq_len, vocab_size) — only last token
    last_logits = outputs.logits.squeeze(0)  # (seq_len, vocab_size)
    tensors["logits"] = last_logits

    # 4. Ground truth next token
    next_token_id = last_logits[-1].argmax().item()
    tensors["next_token_id"] = torch.tensor([float(next_token_id)])
    print(f"Next token ID: {next_token_id} ({tokenizer.decode([next_token_id])!r})")

    # 5. Metadata
    config = model.config
    tensors["meta_hidden_size"] = torch.tensor([float(config.hidden_size)])
    tensors["meta_num_layers"] = torch.tensor([float(config.num_hidden_layers)])
    tensors["meta_vocab_size"] = torch.tensor([float(config.vocab_size)])
    tensors["meta_seq_len"] = torch.tensor([float(input_ids.shape[1])])

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "golden_smollm2_135m.safetensors"
    save_file(tensors, str(out_path))
    print(f"Saved {len(tensors)} tensors to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify self-consistency
    for name, t in tensors.items():
        assert t.isfinite().all(), f"Non-finite values in {name}"
    print("All tensors finite ✓")


if __name__ == "__main__":
    main()
