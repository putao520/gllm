#!/usr/bin/env python3
"""Check HuggingFace model file structures for all 26 supported models."""

import json
import urllib.request
import urllib.error
import sys

# All 26 models from registry.rs
MODELS = {
    # Embedding models
    "bge-small-zh": "BAAI/bge-small-zh-v1.5",
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "bge-base-en": "BAAI/bge-base-en-v1.5",
    "bge-large-en": "BAAI/bge-large-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "e5-large": "intfloat/e5-large",
    "e5-base": "intfloat/e5-base",
    "e5-small": "intfloat/e5-small",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-embeddings-v2-small-en": "jinaai/jina-embeddings-v2-small-en",
    "m3e-base": "moka-ai/m3e-base",
    "multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "distiluse-base-multilingual-cased-v1": "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",
    # Reranker models
    "bge-reranker-v2": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "ms-marco-electra-base": "cross-encoder/ms-marco-electra-base",
    "quora-distilroberta-base": "cross-encoder/quora-distilroberta-base",
}

def get_model_files(repo_id: str) -> list:
    """Fetch file list from HuggingFace API."""
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return [f for f in data if f["type"] == "file"]
    except urllib.error.HTTPError as e:
        return [{"error": f"HTTP {e.code}"}]
    except Exception as e:
        return [{"error": str(e)}]

def analyze_model(alias: str, repo_id: str) -> dict:
    """Analyze model file structure."""
    files = get_model_files(repo_id)

    if files and "error" in files[0]:
        return {"error": files[0]["error"]}

    result = {
        "repo_id": repo_id,
        "has_safetensors": False,
        "has_pytorch_bin": False,
        "has_tokenizer_json": False,
        "has_sentencepiece": False,
        "has_vocab_txt": False,
        "weight_files": [],
        "tokenizer_files": [],
        "all_files": [],
    }

    for f in files:
        name = f["path"]
        size_mb = f.get("lfs", {}).get("size", f.get("size", 0)) / 1024 / 1024
        result["all_files"].append(name)

        # Check weight files
        if name == "model.safetensors":
            result["has_safetensors"] = True
            result["weight_files"].append(f"model.safetensors ({size_mb:.1f}MB)")
        elif name == "pytorch_model.bin":
            result["has_pytorch_bin"] = True
            result["weight_files"].append(f"pytorch_model.bin ({size_mb:.1f}MB)")
        elif name.startswith("model-") and name.endswith(".safetensors"):
            # Sharded safetensors
            result["weight_files"].append(f"{name} ({size_mb:.1f}MB)")
        elif name.startswith("pytorch_model-") and name.endswith(".bin"):
            # Sharded pytorch
            result["weight_files"].append(f"{name} ({size_mb:.1f}MB)")

        # Check tokenizer files
        if name == "tokenizer.json":
            result["has_tokenizer_json"] = True
            result["tokenizer_files"].append("tokenizer.json")
        elif "sentencepiece" in name.lower() or name.endswith(".model"):
            result["has_sentencepiece"] = True
            result["tokenizer_files"].append(name)
        elif name == "vocab.txt":
            result["has_vocab_txt"] = True
            result["tokenizer_files"].append("vocab.txt")

    return result

def main():
    print("=" * 80)
    print("HuggingFace Model File Structure Analysis")
    print("=" * 80)

    # Summary counters
    has_safetensors = 0
    needs_pytorch = 0
    has_tokenizer_json = 0
    needs_sentencepiece = 0
    errors = 0

    issues = []

    for alias, repo_id in MODELS.items():
        print(f"\n📦 {alias}")
        print(f"   Repo: {repo_id}")

        result = analyze_model(alias, repo_id)

        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            errors += 1
            issues.append(f"{alias}: API error - {result['error']}")
            continue

        # Weight analysis
        if result["has_safetensors"]:
            print(f"   ✅ Weights: model.safetensors")
            has_safetensors += 1
        elif result["has_pytorch_bin"]:
            print(f"   ⚠️  Weights: pytorch_model.bin (no safetensors!)")
            needs_pytorch += 1
            issues.append(f"{alias}: No safetensors, only pytorch_model.bin")
        elif result["weight_files"]:
            print(f"   ⚠️  Weights: {', '.join(result['weight_files'][:3])}")
            issues.append(f"{alias}: Unusual weight format - {result['weight_files']}")
        else:
            print(f"   ❌ Weights: NONE FOUND!")
            issues.append(f"{alias}: No weight files found")

        # Tokenizer analysis
        if result["has_tokenizer_json"]:
            print(f"   ✅ Tokenizer: tokenizer.json")
            has_tokenizer_json += 1
        elif result["has_sentencepiece"]:
            sp_files = [f for f in result["tokenizer_files"] if "sentence" in f.lower() or f.endswith(".model")]
            print(f"   ⚠️  Tokenizer: {', '.join(sp_files)} (no tokenizer.json!)")
            needs_sentencepiece += 1
            issues.append(f"{alias}: SentencePiece tokenizer, not tokenizer.json")
        elif result["has_vocab_txt"]:
            print(f"   ⚠️  Tokenizer: vocab.txt (needs WordPiece)")
            issues.append(f"{alias}: vocab.txt tokenizer")
        else:
            print(f"   ❌ Tokenizer: NONE FOUND!")
            issues.append(f"{alias}: No tokenizer files found")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(MODELS)
    print(f"\nTotal models: {total}")
    print(f"✅ Has model.safetensors: {has_safetensors}/{total}")
    print(f"⚠️  Only pytorch_model.bin: {needs_pytorch}/{total}")
    print(f"✅ Has tokenizer.json: {has_tokenizer_json}/{total}")
    print(f"⚠️  Needs SentencePiece: {needs_sentencepiece}/{total}")
    print(f"❌ API Errors: {errors}/{total}")

    if issues:
        print("\n" + "=" * 80)
        print("ISSUES TO ADDRESS")
        print("=" * 80)
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDED DOWNLOAD STRATEGY")
    print("=" * 80)
    print("""
1. Weight files (in order of preference):
   a. model.safetensors (preferred - native format)
   b. pytorch_model.bin (fallback - needs conversion or direct loading)
   c. Sharded files (model-*.safetensors or pytorch_model-*.bin)

2. Tokenizer files (check in order):
   a. tokenizer.json (HuggingFace tokenizers format)
   b. sentencepiece.bpe.model / *.model (SentencePiece format)
   c. vocab.txt + tokenizer_config.json (WordPiece/BERT format)

3. Required config files:
   a. config.json (model architecture)
   b. tokenizer_config.json (tokenizer settings)
   c. special_tokens_map.json (special tokens)
""")

if __name__ == "__main__":
    main()
