#!/bin/bash
set -e

echo "===================================================="
echo "      GLLM ONE-CLICK REGRESSION TEST SUITE          "
echo "===================================================="

# Set release mode for realistic performance measurement
FLAGS="--release --features tokio"

echo "Phase 1: CPU Matrix Regression (Stability & Accuracy)"
export GLLM_FORCE_CPU=1
export GLLM_SKIP_LARGE=0
cargo run $FLAGS --example matrix_test

echo ""
echo "Phase 2: GPU Matrix Regression (Performance)"
export GLLM_FORCE_CPU=0
# Large models might need GPU, skip if not enough VRAM in CI but here we assume user has it
cargo run $FLAGS --example matrix_test

echo ""
echo "Phase 3: Integration Test Plan"
cargo test --test integration model_test_plan -- --nocapture

echo ""
echo "===================================================="
echo "          REGRESSION TESTING COMPLETED              "
echo "===================================================="
