# Cross-Language Alignment Tests (REQ-TEST-011)

Validates that gllm inference output matches PyTorch/HuggingFace reference output within FP32 tolerance.

## Setup

```bash
cd tests/e2e_alignment
pip install -r requirements.txt
python generate_golden.py --model intfloat/multilingual-e5-small --output data/golden.safetensors
```

## Run

```bash
# From project root
cargo test --test e2e_alignment -- --test-threads=1
```

## Tolerance

- FP32 element-wise: < 1e-5
- Cosine similarity: > 0.9999

## Notes

- Golden data is NOT checked into git (see .gitignore)
- Tests are `#[ignore]` by default — run with `--ignored` after generating golden data
