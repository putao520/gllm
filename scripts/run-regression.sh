#!/bin/bash
# gllm 回归测试脚本
# 按架构分组测试，避免重复测试同类模型

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "gllm 回归测试"
echo "========================================="
echo ""
echo "测试策略：按架构分组，每组测试一个代表模型"
echo "优先使用公开模型，减少对 HF_TOKEN 的依赖"
echo ""

case "${1:-arch}" in
    ci|quick)
        echo "⚡ CI 快速测试 (2个最小模型)"
        cargo test --test test_real_models regression_ci_quick -- --nocapture --test-threads=1
        ;;
    arch|all)
        echo "🚀 架构回归测试 (公开模型)"
        cargo test --test test_real_models regression_all_architectures -- --nocapture --test-threads=1
        ;;
    gated)
        echo "🔒 Gated 模型测试 (需要 HF_TOKEN)"
        if [ -z "$HF_TOKEN" ] && [ ! -f ~/.huggingface/token ]; then
            echo "⚠️  请先设置 HF_TOKEN 或 ~/.huggingface/token"
            exit 1
        fi
        cargo test --test test_real_models regression_gated_models -- --nocapture --test-threads=1
        ;;
    local)
        echo "📦 本地缓存模型测试"
        cargo test --test test_real_models real_models_batch_test_all_available -- --nocapture
        ;;
    matrix)
        echo "🧪 虚拟模型矩阵测试"
        cargo test --test test_model_matrix -- --nocapture
        ;;
    *)
        echo "用法: $0 [ci|arch|gated|local|matrix]"
        echo ""
        echo "选项:"
        echo "  ci       - CI 快速测试 (2个最小模型，默认)"
        echo "  arch     - 架构回归测试 (公开模型)"
        echo "  gated    - Gated 模型测试 (需要 HF_TOKEN)"
        echo "  local    - 本地缓存模型测试"
        echo "  matrix   - 虚拟模型矩阵测试"
        echo ""
        echo "公开模型覆盖 (无需 token):"
        echo "  Generator: Qwen3-1.7B, SmolLM, Phi4, Gemma2, InternLM3"
        echo "  Embedding: BGE-M3, E5-Small, M3E"
        echo ""
        echo "Gated 模型 (需要 HF_TOKEN):"
        echo "  Generator: Llama4, GPT-OSS, GLM4, Mistral3"
        echo "  Embedding: Qwen3-Embed"
        echo "  Reranker:  Qwen3-Rerank, BGE-Reranker"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
