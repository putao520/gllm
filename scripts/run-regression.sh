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
echo "覆盖率：11个架构 → 32个模型"
echo ""

case "${1:-arch}" in
    ci|quick)
        echo "⚡ CI 快速测试 (2个最小模型)"
        cargo test --test test_real_models regression_ci_quick -- --nocapture --test-threads=1
        ;;
    arch|all)
        echo "🚀 架构回归测试 (11个架构代表)"
        cargo test --test test_real_models regression_all_architectures -- --nocapture --test-threads=1
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
        echo "用法: $0 [ci|arch|local|matrix]"
        echo ""
        echo "选项:"
        echo "  ci       - CI 快速测试 (2个最小模型，默认)"
        echo "  arch     - 架构回归测试 (11个架构代表)"
        echo "  local    - 本地缓存模型测试"
        echo "  matrix   - 虚拟模型矩阵测试"
        echo ""
        echo "架构覆盖:"
        echo "  Generator: Qwen3, Llama4, SmolLM, Phi4, InternLM3, GPT-OSS, GLM4, Mistral3"
        echo "  Embedding: Qwen3-Embed, BGE-XlmR, E5-XlmR"
        echo "  Reranker:  Qwen3-Rerank, BGE-Rerank"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
