#!/bin/bash
# Regression Test Script for gllm (CUDA Backend)
# Tests all supported models on CUDA backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  gllm Regression Test (CUDA Backend)"
echo "════════════════════════════════════════════════════════════════"
echo ""

cargo run --release --example regression -- --cuda "$@"
