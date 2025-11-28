#!/bin/bash

# Real Model Testing Script
#
# This script runs tests with actual model downloads from HuggingFace Hub.
# It will download real models and perform genuine inference.
#
# Usage: ./scripts/test_real_models.sh [test_name]
# If no test_name is provided, it runs all real model tests

set -e

echo "🌍 GLLM Real Model Testing Script"
echo "================================="
echo "⚠️  WARNING: This script will download actual models from HuggingFace Hub"
echo "⚠️  This requires internet connection and may take several minutes"
echo ""

# Default test
TEST_NAME="${1:-test_real_model_download_and_inference}"

# Check if we have internet connection
if ! ping -c 1 huggingface.co > /dev/null 2>&1; then
    echo "❌ Error: No internet connection to HuggingFace Hub"
    echo "   Please check your network connection and try again"
    exit 1
fi

echo "✅ Internet connection to HuggingFace Hub available"
echo ""

# Set up environment for real model testing
export HF_HUB_DOWNLOAD_TIMEOUT="600"  # 10 minutes timeout

# Create a temporary directory for model downloads if not set
if [ -z "$GLLM_MODEL_CACHE_DIR" ]; then
    export GLLM_MODEL_CACHE_DIR="/tmp/gllm-real-model-cache"
    mkdir -p "$GLLM_MODEL_CACHE_DIR"
    echo "📁 Using model cache directory: $GLLM_MODEL_CACHE_DIR"
fi

echo ""
echo "🧪 Running real model test: $TEST_NAME"
echo "======================================"

# Run the specified test
if cargo test "integration::$TEST_NAME" --features cpu,wgpu,async -- --nocapture; then
    echo ""
    echo "🎉 Real model test PASSED!"
    echo ""
    echo "📊 Model cache usage:"
    du -sh "$GLLM_MODEL_CACHE_DIR"/* 2>/dev/null || echo "   No models cached yet"
else
    echo ""
    echo "❌ Real model test FAILED!"
    echo ""
    echo "🔍 Debugging information:"
    echo "   - Check internet connection"
    echo "   - Verify HuggingFace Hub is accessible"
    echo "   - Check available disk space in $GLLM_MODEL_CACHE_DIR"
    echo "   - Try running with RUST_LOG=debug for more details"
    echo ""
    echo "   Cache directory contents:"
    ls -la "$GLLM_MODEL_CACHE_DIR" 2>/dev/null || echo "   Cache directory not found"
    exit 1
fi

echo ""
echo "🧹 Cleaning up temporary files..."
# We don't remove the cache directory to allow reuse in subsequent tests
# Uncomment the line below if you want to clean up after each test
# rm -rf "$GLLM_MODEL_CACHE_DIR"

echo ""
echo "✅ Real model testing completed!"