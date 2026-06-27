# model-knowledge

从上游生态（llama.cpp / HF transformers）自动提取模型知识，校验 gllm `FIELD_DEFS` 的覆盖度，输出 gap 报告供人工 review。

## 目的

gllm 的 `ModelConfig` 解析依赖 4 张手写知识表：

- `src/model_config_fragments/field_registry.inc.rs` — `FIELD_DEFS`（canonical 字段 + json_keys + gguf_keys）
- `src/arch/registry.rs` — `ARCH_TABLE`（架构 token 别名）
- `src/loader/name_map.rs` — tensor 名映射

上游生态规模远超手写覆盖：

| 上游 | 规模 |
|------|------|
| llama.cpp | 132 架构 / 281 GGUF key / 395 tensor 枚举 |
| HF transformers | 664 model_type / 13240+ config 字段 |

本工具集定时从上游提取知识到 TOML 数据文件，与 gllm 现有知识表做差集，输出 gap 报告。人工 review diff 后补齐 `FIELD_DEFS`（补齐动作不在本工具范围内）。

## 同步流程（3 步）

```bash
cd tools/model-knowledge

# 1. 准备 venv + 依赖（首次或依赖变更时）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. 一键提取 + 校验（包含 1+2+3+校验）
./run_sync.sh

# 报告输出到 ../../generated/model-knowledge/GAP_REPORT.md
```

`run_sync.sh` 内部完成：

1. 检查/创建 `.venv`（不存在则 `python3 -m venv` + `pip install -r requirements.txt`）
2. sparse-checkout 拉取/更新上游到 `.upstream/`：
   - `llama.cpp`: `gguf-py/` + `src/llama-arch.*` + `src/llama-model.cpp` + `src/llama-hparams.h`
   - `transformers`: `src/transformers/models/` + `src/transformers/configuration_utils.py`
3. 运行 `extract_llama.py` + `extract_hf.py` 生成 TOML
4. 运行 `validate.py` 输出 gap 报告（退出码 0=无 gap / 1=有 gap）

所有路径相对脚本目录解析，不依赖 cwd。

## 定时运行

本工具是纯脚本，不绑定定时器。由外部 cron / CI 调度。

### cron 示例

每天 03:17 跑一次（避开整点舰队高峰），有 gap 时邮件通知：

```cron
17 3 * * * cd /home/putao/code/rust/gllm/tools/model-knowledge && ./run_sync.sh >> /var/log/gllm-model-knowledge.log 2>&1 || mail -s "[gllm] model-knowledge gap detected" team@example.com < /home/putao/code/rust/gllm/generated/model-knowledge/GAP_REPORT.md
```

### GitHub Actions 示例

每周一 03:17 UTC 跑，有 gap（退出码 1）时开 issue：

```yaml
# .github/workflows/model-knowledge-sync.yml
name: model-knowledge-sync
on:
  schedule:
    - cron: '17 3 * * 1'   # 每周一 03:17 UTC
  workflow_dispatch:
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -r tools/model-knowledge/requirements.txt
      - run: tools/model-knowledge/run_sync.sh
        continue-on-error: true
        id: sync
      - if: steps.sync.outcome == 'failure'
        uses: peter-evans/create-issue-from-file@v5
        with:
          title: "[model-knowledge] upstream gap detected"
          content-filepath: generated/model-knowledge/GAP_REPORT.md
```

## 上游版本锁定

每次提取记录上游 commit hash 到 `generated/model-knowledge/UPSTREAM_COMMIT.lock`，格式：

```
# Updated: 2026-06-27T10:00:00Z
llama.cpp=<40-char-sha>
transformers=<40-char-sha>
```

TOML 产物提交到 git，便于 diff review。`.upstream/` 工作目录被 `.gitignore` 忽略。

## 产物清单

`generated/model-knowledge/` 下：

| 文件 | 内容 |
|------|------|
| `llama-arch.toml` | llama.cpp 架构名列表 + per-arch tensor 列表 |
| `llama-kv.toml` | 281 GGUF key + 语义分类 |
| `llama-tensor.toml` | tensor 名模板（如 `blk.%d.attn_q`） |
| `llama-semantics.toml` | `get_key(LLM_KV_X, hparams.Y)` 提取的 KV→field 语义映射 |
| `hf-config-fields.toml` | transformers 每 model_type 的 config 字段 + 全局 attribute_map |
| `GAP_REPORT.md` | gap 校验报告（MISSING GGUF KEYS / MISSING ARCHITECTURES / HF ALIASES NOT COVERED / SEMANTIC MAPPING COVERAGE） |
| `UPSTREAM_COMMIT.lock` | 上游 commit hash 锁定 |

## License attribution

提取的上游知识分别遵循其原始 license：

- **llama.cpp**: MIT License — https://github.com/ggml-org/llama.cpp
- **transformers**: Apache License 2.0 — https://github.com/huggingface/transformers

提取的 TOML 产物是上游元数据（字段名 / 架构名 / key 字符串）的事实记录，不包含上游源代码。产物随 gllm 仓库 license 分布，但上游元数据的著作权属上游项目；引用上游具体字段名时建议保留来源注释。
