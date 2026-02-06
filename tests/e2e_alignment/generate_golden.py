import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file

OUTPUT_DIR = "tests/e2e_alignment/data"
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"

def generate_embedding_golden():
    print(f"Processing Embedding: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    text = "Hello world from Rust alignment test"
    # No padding needed for single sequence
    inputs = tokenizer(text, return_tensors="pt")

    print(f"Embedding Tokens: {inputs['input_ids'][0].tolist()}")

    with torch.no_grad():
        # Use the base model to get the last_hidden_state (which includes final layernorm)
        # This matches gllm's forward_hidden output
        outputs = model.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        # Get last token
        # shape: [1, seq_len, hidden]
        last_token_embedding = last_hidden_state[0, -1, :]

    save_file({
        "embeddings": last_token_embedding.float().contiguous(),
    }, os.path.join(OUTPUT_DIR, "golden_embedding.safetensors"))

def generate_generation_golden():
    print(f"Processing Generation: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"Generation Tokens: {inputs['input_ids'][0].tolist()}")

    with torch.no_grad():
        outputs = model(**inputs)
        # GLLM computes logits for the LAST token only
        next_token_logits = outputs.logits[0, -1, :]

    save_file({
        "logits": next_token_logits.float().contiguous(),
    }, os.path.join(OUTPUT_DIR, "golden_generation.safetensors"))

def generate_rerank_golden():
    # Use SmolLM as a "fallback" reranker to verify the data flow.
    # Since SmolLM has no score head, gllm falls back to returning the last hidden state (Embedding).
    # We verify that this fallback path is numerically aligned.
    print(f"Processing Rerank (Fallback): {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    # Rerank usually takes query + doc. gllm joins them.
    # We simulate the input "query" + " " + "doc"
    text = "query hello world doc this is a test"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # Get last token embedding
        last_token_embedding = last_hidden_state[0, -1, :]

    save_file({
        "scores": last_token_embedding.float().contiguous(),
    }, os.path.join(OUTPUT_DIR, "golden_rerank.safetensors"))

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        generate_embedding_golden()
        generate_generation_golden()
        generate_rerank_golden()
        print("All golden data generated successfully!")
    except Exception as e:
        print(f"Error generating golden data: {e}")
        exit(1)
