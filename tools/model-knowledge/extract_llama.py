"""
从 llama.cpp 提取 GGUF 模型知识。

两层提取:

名称层 (gguf-py/gguf/constants.py SSOT, C++ 是镜像):
    from gguf.constants import MODEL_ARCH, Keys, MODEL_TENSOR, TENSOR_NAMES, MODEL_TENSORS
    反射提取架构 / KV key / tensor 名模板 / per-arch tensor 列表。

语义层 (src/llama-model.cpp load_hparams):
    tree-sitter-cpp 解析, 找 ml.get_key(LLM_KV_X, hparams.Y) 调用,
    得到 (KV enum, hparams 字段) 对。再 join Keys 得到 GGUF key 字符串。

输出 TOML:
    llama-arch.toml       架构名 + per-arch tensor 列表
    llama-kv.toml         GGUF key + 语义分类
    llama-tensor.toml     tensor 名模板
    llama-semantics.toml  KV→hparams field 语义映射
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # py311+
    _dump_toml = None  # 见下方自定义 writer
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# tree-sitter-cpp 在 venv 里安装; 解析延迟到调用时
_CPP_PARSER = None


def _cpp_parser():
    global _CPP_PARSER
    if _CPP_PARSER is None:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser
        lang = Language(tscpp.language())
        _CPP_PARSER = Parser(lang)
    return _CPP_PARSER


# ─────────────────────────────────────────────────────────────
# TOML writer (避免依赖 tomli_w; 产物是简单结构)
# ─────────────────────────────────────────────────────────────

def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _dump_table(name: str, rows: list[dict], headers: list[str]) -> str:
    """渲染 [[name]] 数组 of tables。"""
    out = [f"# {name}: {len(rows)} entries"]
    for r in rows:
        out.append(f"\n[[{name}]]")
        for h in headers:
            v = r.get(h)
            if v is None:
                out.append(f"{h} = \"\"")
            elif isinstance(v, bool):
                out.append(f"{h} = {'true' if v else 'false'}")
            elif isinstance(v, int):
                out.append(f"{h} = {v}")
            elif isinstance(v, float):
                out.append(f"{h} = {v}")
            elif isinstance(v, list):
                items = ", ".join(f'"{_toml_escape(str(x))}"' for x in v)
                out.append(f"{h} = [{items}]")
            else:
                out.append(f'{h} = "{_toml_escape(str(v))}"')
    return "\n".join(out) + "\n"


# ─────────────────────────────────────────────────────────────
# 名称层: gguf.constants 反射
# ─────────────────────────────────────────────────────────────

def _enum_values(enum_cls) -> list[str]:
    """枚举成员名列表 (不含 alias / value 重复)。"""
    seen, names = set(), []
    for name in dir(enum_cls):
        if name.startswith("_"):
            continue
        val = getattr(enum_cls, name)
        # 过滤方法/嵌套类: 只收 enum 成员
        if not hasattr(val, "name") or not hasattr(val, "value"):
            continue
        # 去重 value (alias 指向同 value 时只留首个)
        try:
            key = val.value
        except Exception:
            continue
        if isinstance(key, (str, int)) and key not in seen:
            seen.add(key)
            names.append(name)
    return names


def _kv_keys(keys_cls) -> dict[str, str]:
    """从 gguf.constants.Keys (及其嵌套类) 递归收集 key 路径 → 字面量。

    Keys 的成员是 'attention.head_count' 这种带占位符或点分路径的字符串字面量。
    """
    result: dict[str, str] = {}

    def _walk(obj, prefix: str):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            val = getattr(obj, name)
            if isinstance(val, str):
                # 字面量 key
                result[f"{prefix}{name}" if not prefix else f"{prefix}.{name}"] = val
            elif hasattr(val, "__dict__") or isinstance(val, type):
                # 嵌套 namespace 类
                _walk(val, f"{prefix}{name}." if prefix else f"{name}.")

    _walk(keys_cls, "")
    return result


def _classify_key(key: str) -> str:
    """对 GGUF key 做粗语义分类 (基于点分前缀)。"""
    k = key.lower()
    # 含 {arch} 占位符 (gguf-py 用 {arch}, 不是 %s)
    if "{arch}" in key:
        return "arch_specific"
    if k.startswith("general."):
        return "general"
    if ".attention." in k or k.startswith("attention."):
        return "attention"
    if ".rope." in k or k.startswith("rope."):
        return "rope"
    if ".feed_forward" in k or "ffn" in k:
        return "ffn"
    if ".block_count" in k or ".expert" in k or "moe" in k:
        return "moe_or_block"
    if ".norm" in k or "rms" in k or "layernorm" in k:
        return "norm"
    if "vocab" in k or "token" in k or "tokenizer" in k:
        return "vocab"
    return "other"


# ─────────────────────────────────────────────────────────────
# 语义层: tree-sitter-cpp 解析 load_hparams
# ─────────────────────────────────────────────────────────────

def _node_text(node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _iter_call_expressions(root):
    """递归 yield 所有 call_expression 节点。"""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "call_expression":
            yield n
        for c in n.children:
            stack.append(c)


def _extract_get_key_calls(root, source_bytes: bytes):
    """提取形如 ml.get_key(LLM_KV_X, hparams.Y, ...) 的调用。

    返回 [(kv_enum_name, hparams_field), ...]
    """
    results: list[tuple[str, str]] = []
    for call in _iter_call_expressions(root):
        func = call.child_by_field_name("function")
        if func is None:
            continue
        # function 可能是 identifier (get_key) 或 field_expression (ml.get_key / ml->get_key)
        if func.type == "field_expression":
            fname = func.child_by_field_name("field")
            if fname is None or fname.text != b"get_key":
                continue
        elif func.type == "identifier":
            if func.text != b"get_key":
                continue
        else:
            continue

        args = call.child_by_field_name("arguments")
        if args is None:
            continue
        # arguments 节点: ( expr, expr, ... )
        arg_nodes = [c for c in args.children
                     if c.type not in ("(", ")", ",", "comment")]
        if len(arg_nodes) < 2:
            continue
        kv_node = arg_nodes[0]
        hp_node = arg_nodes[1]

        # arg0 期望 LLM_KV_X (identifier 或 :: qualified)
        kv_text = kv_node.text.decode("utf-8", errors="replace").strip()
        kv_name = kv_text.split("::")[-1].split(".")[-1] if kv_text else ""
        if not kv_name.startswith("LLM_KV_"):
            continue

        # arg1 期望 hparams.Y (field_expression) — 取 field 名
        if hp_node.type == "field_expression":
            field = hp_node.child_by_field_name("field")
            if field is None:
                continue
            hp_field = field.text.decode("utf-8", errors="replace")
        else:
            # 形如 (*tensor).field 或其它表达式, 取末尾 token
            hp_field = hp_node.text.decode("utf-8", errors="replace").strip()
            hp_field = re.split(r"[.\s\[\]()]+", hp_field)[-1] if hp_field else ""

        if kv_name and hp_field:
            results.append((kv_name, hp_field))
    return results


def parse_load_hparams(upstream: Path) -> list[tuple[str, str]]:
    """解析 src/llama-model.cpp 的 load_hparams, 提取 get_key 调用。"""
    target = upstream / "src" / "llama-model.cpp"
    if not target.exists():
        print(f"[extract_llama] WARN: {target} not found, skipping semantic layer",
              file=sys.stderr)
        return []
    source = target.read_bytes()
    tree = _cpp_parser().parse(source)
    calls = _extract_get_key_calls(tree.root_node, source)
    # 去重保序
    seen, uniq = set(), []
    for c in calls:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def _git_commit(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def extract(upstream: Path, out: Path) -> None:
    # 延迟 import: gguf 来自 venv
    from gguf import constants as gc

    # ── 名称层 ──
    arch_names = _enum_values(gc.MODEL_ARCH)
    kv_map = _kv_keys(gc.Keys)
    # TENSOR_NAMES 在新版 gguf-py 是 dict[MODEL_TENSOR enum → string]
    tensor_names: dict[str, str] = {}
    tn = getattr(gc, "TENSOR_NAMES", None)
    if isinstance(tn, dict):
        for mt_enum, name_str in tn.items():
            enum_name = getattr(mt_enum, "name", str(mt_enum))
            tensor_names[enum_name] = name_str
    elif tn is not None:
        # 旧版可能是 namespace
        for name in dir(tn):
            if name.startswith("_"):
                continue
            val = getattr(tn, name)
            if isinstance(val, str):
                tensor_names[name] = val

    # per-arch tensor list: MODEL_TENSORS dict[ARCH -> list[MODEL_TENSOR]]
    per_arch_tensors: dict[str, list[str]] = {}
    mt = getattr(gc, "MODEL_TENSORS", None)
    if mt is not None:
        for arch_member in gc.MODEL_ARCH:
            try:
                lst = mt.get(arch_member, []) if hasattr(mt, "get") else mt[arch_member]
            except Exception:
                lst = []
            names = []
            for t in lst:
                tname = getattr(t, "name", str(t))
                names.append(tname)
            per_arch_tensors[arch_member.name] = names

    # ── 语义层 ──
    semantic_calls = parse_load_hparams(upstream)

    # join: KV enum → GGUF key string
    # gguf Keys 是 'attention.head_count' 字面量, enum 名是 'ATTN_HEAD_COUNT'
    # 反向映射: enum_name → key_literal (粗略, 不一定一一对应)
    enum_to_key: dict[str, str] = {}
    if hasattr(gc, "MODEL_TENSOR"):
        pass  # tensor enum 反向不在此处理
    # LLM_KV_* 在新版 llama.cpp 是 llama 内部 enum, 不在 gguf.constants。
    # 我们直接记录 enum 名 + 让 validate.py 后续对照 gguf key 文本。
    # 同时尝试: 把 enum_name 里 LLM_KV_ 前缀去掉后, 在 kv_map values 里找含该片段的 key
    def _lookup_key(enum_name: str) -> str:
        # LLM_KV_ATTN_HEAD_COUNT → attn.head_count
        stem = enum_name[len("LLM_KV_"):] if enum_name.startswith("LLM_KV_") else enum_name
        parts = stem.lower().split("_")
        # 在 kv_map value 里找包含所有 part 的
        for kstr in kv_map.values():
            ks = kstr.lower()
            if all(p in ks for p in parts):
                return kstr
        # fallback: 不含 arch 占位符的通用形态
        return ""

    semantic_rows = []
    for kv_enum, hp_field in semantic_calls:
        semantic_rows.append({
            "kv_enum": kv_enum,
            "gguf_key": _lookup_key(kv_enum),
            "hparams_field": hp_field,
        })

    # ── 输出 TOML ──
    out.mkdir(parents=True, exist_ok=True)

    arch_rows = []
    for a in arch_names:
        arch_rows.append({
            "arch_enum": a,
            "tensors": per_arch_tensors.get(a, []),
        })
    (out / "llama-arch.toml").write_text(
        _dump_table("arch", arch_rows, ["arch_enum", "tensors"]),
        encoding="utf-8",
    )

    kv_rows = []
    for kname, kstr in sorted(kv_map.items()):
        kv_rows.append({
            "key_name": kname,
            "key_string": kstr,
            "has_arch_placeholder": "{arch}" in kstr,
            "category": _classify_key(kstr),
        })
    (out / "llama-kv.toml").write_text(
        _dump_table("kv", kv_rows, ["key_name", "key_string", "has_arch_placeholder", "category"]),
        encoding="utf-8",
    )

    tensor_rows = []
    for tname, tmpl in sorted(tensor_names.items()):
        tensor_rows.append({
            "tensor_enum": tname,
            "name_template": tmpl,
            "has_index_placeholder": "%d" in tmpl or "%i" in tmpl,
        })
    (out / "llama-tensor.toml").write_text(
        _dump_table("tensor", tensor_rows, ["tensor_enum", "name_template", "has_index_placeholder"]),
        encoding="utf-8",
    )

    (out / "llama-semantics.toml").write_text(
        _dump_table("semantic", semantic_rows, ["kv_enum", "gguf_key", "hparams_field"]),
        encoding="utf-8",
    )

    # ── commit lock ──
    commit = _git_commit(upstream)
    lock = (
        f"# Updated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n"
        f"llama.cpp={commit}\n"
    )
    _merge_lock(out / "UPSTREAM_COMMIT.lock", "llama.cpp", commit)

    print(f"[extract_llama] arch={len(arch_rows)} kv={len(kv_rows)} "
          f"tensor={len(tensor_rows)} semantic={len(semantic_rows)} "
          f"→ {out}", file=sys.stderr)


def _merge_lock(lock_path: Path, key: str, value: str) -> None:
    """读改写 UPSTREAM_COMMIT.lock, 保留其它 key。"""
    lines = []
    existing = {}
    if lock_path.exists():
        for ln in lock_path.read_text(encoding="utf-8").splitlines():
            if ln.startswith("#") or "=" not in ln:
                lines.append(ln)
                continue
            k, _, v = ln.partition("=")
            existing[k.strip()] = v.strip()
    existing[key] = value
    header = [ln for ln in lines if ln.startswith("#")]
    body = [f"{k}={existing[k]}" for k in sorted(existing)]
    out = []
    if header:
        out.extend(header)
    else:
        out.append(f"# Updated: {_dt.datetime.now(_dt.timezone.utc).isoformat()}")
    out.extend(body)
    lock_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--upstream", type=Path,
                   default=here / ".upstream" / "llama.cpp",
                   help="path to llama.cpp checkout (default: .upstream/llama.cpp)")
    p.add_argument("--out", type=Path,
                   default=repo_root / "generated" / "model-knowledge",
                   help="output dir for TOML (default: ../../generated/model-knowledge/)")
    args = p.parse_args(argv)
    if not args.upstream.exists():
        print(f"[extract_llama] ERROR: upstream not found: {args.upstream}",
              file=sys.stderr)
        print("  run ./run_sync.sh first, or pass --upstream <llama.cpp path>",
              file=sys.stderr)
        return 2
    extract(args.upstream, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
