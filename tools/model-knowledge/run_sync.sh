#!/usr/bin/env bash
# model-knowledge 一键同步入口
#
# 流程: venv 检查 → 上游 sparse-checkout → extract_llama.py + extract_hf.py → validate.py → 报告路径
#
# 所有路径相对脚本目录解析, 不依赖 cwd。
# 退出码: 0=成功(可能有 gap, gap 信息在报告里), 非 0=流程故障

set -euo pipefail

# ── 解析脚本所在目录为绝对路径 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_ROOT="$SCRIPT_DIR"
UPSTREAM_DIR="$TOOL_ROOT/.upstream"
VENV_DIR="$TOOL_ROOT/.venv"
REPO_ROOT="$(cd "$TOOL_ROOT/../.." && pwd)"
OUT_DIR="$REPO_ROOT/generated/model-knowledge"

LLAMA_DIR="$UPSTREAM_DIR/llama.cpp"
HF_DIR="$UPSTREAM_DIR/transformers"

LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"
HF_REPO="https://github.com/huggingface/transformers.git"

# sparse-checkout 清单
LLAMA_SPARSE=(
    "gguf-py/"
    "src/llama-arch.cpp"
    "src/llama-arch.h"
    "src/llama-model.cpp"
    "src/llama-hparams.h"
)
HF_SPARSE=(
    "src/transformers/models/"
    "src/transformers/configuration_utils.py"
)

log() { printf '[run_sync] %s\n' "$*" >&2; }
die() { printf '[run_sync] ERROR: %s\n' "$*" >&2; exit 1; }

# ── Step 1: venv ──
ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log "creating venv at $VENV_DIR"
        python3 -m venv "$VENV_DIR" || die "venv creation failed"
        "$VENV_DIR/bin/pip" install --upgrade pip >/dev/null || die "pip upgrade failed"
    fi
    if [[ ! -f "$VENV_DIR/.deps-installed" ]] \
       || [[ "$TOOL_ROOT/requirements.txt" -nt "$VENV_DIR/.deps-installed" ]]; then
        log "installing dependencies"
        "$VENV_DIR/bin/pip" install -r "$TOOL_ROOT/requirements.txt" \
            || die "pip install failed"
        touch "$VENV_DIR/.deps-installed"
    fi
}

# ── Step 2: 上游 sparse-checkout ──
# 用法: sync_upstream <repo_url> <target_dir> <sparse_paths...>
sync_upstream() {
    local repo="$1"; shift
    local dir="$1"; shift
    local paths=("$@")

    if [[ ! -d "$dir/.git" ]]; then
        log "cloning $repo (sparse) → $dir"
        mkdir -p "$dir"
        git clone --no-checkout --depth 1 "$repo" "$dir" \
            || die "clone failed: $repo"
        git -C "$dir" sparse-checkout init --cone \
            || die "sparse-checkout init failed"
        git -C "$dir" sparse-checkout set --no-cone "${paths[@]}" \
            || die "sparse-checkout set failed"
        git -C "$dir" checkout \
            || die "checkout failed: $repo"
    else
        log "updating $repo"
        git -C "$dir" fetch --depth 1 origin HEAD \
            || die "fetch failed: $repo"
        git -C "$dir" reset --hard origin/HEAD \
            || die "reset failed: $repo"
        # 确保 sparse 配置覆盖最新清单（清单变化时同步）
        git -C "$dir" sparse-checkout set --no-cone "${paths[@]}" \
            || die "sparse-checkout re-set failed"
    fi
}

# ── 主流程 ──
main() {
    log "tool root: $TOOL_ROOT"
    log "output dir: $OUT_DIR"
    mkdir -p "$OUT_DIR"

    ensure_venv

    log "sync upstream: llama.cpp"
    sync_upstream "$LLAMA_REPO" "$LLAMA_DIR" "${LLAMA_SPARSE[@]}"

    log "sync upstream: transformers"
    sync_upstream "$HF_REPO" "$HF_DIR" "${HF_SPARSE[@]}"

    log "extract: llama.cpp"
    "$VENV_DIR/bin/python" "$TOOL_ROOT/extract_llama.py" \
        --upstream "$LLAMA_DIR" \
        --out "$OUT_DIR"

    log "extract: transformers"
    "$VENV_DIR/bin/python" "$TOOL_ROOT/extract_hf.py" \
        --upstream "$HF_DIR" \
        --out "$OUT_DIR"

    log "validate: gap report"
    # validate.py 退出码: 0=无 gap, 1=有 gap (两者都是流程成功, 不触发 set -e)
    set +e
    "$VENV_DIR/bin/python" "$TOOL_ROOT/validate.py" \
        --gllm-root "$REPO_ROOT" \
        --knowledge-dir "$OUT_DIR"
    local rc=$?
    set -e
    if [[ $rc -ne 0 && $rc -ne 1 ]]; then
        die "validate.py failed with exit code $rc"
    fi

    log "done. report: $OUT_DIR/GAP_REPORT.md"
    if [[ $rc -eq 1 ]]; then
        log "WARNING: gaps detected — review the report"
    fi
    return 0
}

main "$@"
