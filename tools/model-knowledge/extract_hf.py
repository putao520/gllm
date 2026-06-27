"""
从 HF transformers 提取 JSON config 字段知识。

tree-sitter-python AST 解析 src/transformers/models/*/configuration_*.py,
提取每个 Config 类的:
  - model_type (类属性赋值 `model_type = "xxx"`)
  - AnnAssign 字段 (name, type_annotation, default_value)
  - attribute_map dict (别名→规范名)
  - __post_init__ / __init__ 派生字段标记 (无法静态提取默认值的)

新版 transformers 用 @strict + 类级 annotated assignment:
    class LlamaConfig(PretrainedConfig):
        vocab_size: int = 32000
        hidden_size: int = 4096
        model_type = "llama"
        attribute_map = {"hidden_size": "dim"}

输出: hf-config-fields.toml + attribute_map 汇总
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# TOML writer (与 extract_llama.py 一致风格, 自包含避免依赖 tomli_w)
# ─────────────────────────────────────────────────────────────

def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _dump_table(name: str, rows: list[dict], headers: list[str]) -> str:
    out = [f"# {name}: {len(rows)} entries"]
    for r in rows:
        out.append(f"\n[[{name}]]")
        for h in headers:
            v = r.get(h)
            if v is None:
                out.append(f'{h} = ""')
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
# tree-sitter-python
# ─────────────────────────────────────────────────────────────

_PY_PARSER = None


def _py_parser():
    global _PY_PARSER
    if _PY_PARSER is None:
        import tree_sitter_python as tspy
        from tree_sitter import Language, Parser
        lang = Language(tspy.language())
        _PY_PARSER = Parser(lang)
    return _PY_PARSER


def _node_text(node) -> str:
    return node.text.decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────
# 解析单个 configuration_*.py
# ─────────────────────────────────────────────────────────────

def _string_literal_value(node):
    """从 string 节点取字面量 (去引号 + 去前缀 u/r/b)。"""
    if node.type != "string":
        return None
    text = node.text.decode("utf-8", errors="replace")
    # 去前缀 (b/u/r/f 组合)
    while text and text[0] in "bBurRfF":
        text = text[1:]
    if len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        # 三引号
        if text.startswith(text[0] * 3):
            return text[3:-3]
        return text[1:-1]
    return text


def _parse_dict_literal(node) -> dict[str, str] | None:
    """解析 dict 字面量节点 → {key: value}。仅处理 string key/value。"""
    if node.type != "dictionary":
        return None
    result: dict[str, str] = {}
    for pair in node.children:
        if pair.type != "pair":
            continue
        k = pair.child_by_field_name("key")
        v = pair.child_by_field_name("value")
        if k is None or v is None:
            continue
        ks = _string_literal_value(k) if k.type == "string" else _node_text(k)
        vs = _string_literal_value(v) if v.type == "string" else _node_text(v)
        if ks is not None:
            result[ks] = vs if vs is not None else ""
    return result


def _extract_class_fields(class_node):
    """从 class_definition 节点提取 (model_type, fields, attribute_map, derived)。

    fields: [{name, type, default, has_default}]
    """
    model_type = None
    fields: list[dict] = []
    attribute_map: dict[str, str] = {}
    has_post_init = False

    body = class_node.child_by_field_name("body")
    if body is None:
        return model_type, fields, attribute_map, has_post_init

    for stmt in body.children:
        # 旧版 ts-py: class body 直接是 assignment/annotated_assignment
        # 新版 ts-py (0.25): 都被包在 expression_statement 里
        target = stmt
        if stmt.type == "expression_statement" and stmt.children:
            target = stmt.children[0]

        if target.type == "assignment":
            # model_type = "xxx"  或  attribute_map = {...}  或  a: int = 1 (新版 ts-py)
            lhs = target.child_by_field_name("left")
            rhs = target.child_by_field_name("right")
            ann = target.child_by_field_name("type")  # annotation, None if plain
            if lhs is None or lhs.type != "identifier":
                continue
            name = lhs.text.decode("utf-8", errors="replace")
            if name == "model_type" and rhs is not None and rhs.type == "string":
                model_type = _string_literal_value(rhs)
            elif name == "attribute_map" and rhs is not None:
                parsed = _parse_dict_literal(rhs)
                if parsed is not None:
                    attribute_map = parsed
            elif ann is not None and not name.startswith("_"):
                # annotated field: a: int = 1  (新版 ts-py grammar)
                ann_text = _node_text(ann)
                default_text = None
                has_default = False
                if rhs is not None:
                    has_default = True
                    default_text = _node_text(rhs).strip()
                    if len(default_text) > 80:
                        default_text = default_text[:80] + "…"
                fields.append({
                    "name": name,
                    "type": ann_text,
                    "default": default_text or "",
                    "has_default": has_default,
                })

        elif target.type == "annotated_assignment":
            # 旧版 ts-py: name: type = default
            lhs = target.child_by_field_name("left")
            ann = target.child_by_field_name("type")
            rhs = target.child_by_field_name("right")
            if lhs is None or lhs.type != "identifier":
                continue
            name = lhs.text.decode("utf-8", errors="replace")
            if name.startswith("_"):
                continue
            ann_text = _node_text(ann) if ann is not None else ""
            default_text = None
            has_default = False
            if rhs is not None:
                has_default = True
                default_text = _node_text(rhs).strip()
                if len(default_text) > 80:
                    default_text = default_text[:80] + "…"
            fields.append({
                "name": name,
                "type": ann_text,
                "default": default_text or "",
                "has_default": has_default,
            })

        elif target.type == "function_definition":
            name_node = target.child_by_field_name("name")
            if name_node is not None and name_node.text == b"__post_init__":
                has_post_init = True

    return model_type, fields, attribute_map, has_post_init


def parse_config_file(path: Path) -> list[dict]:
    """解析一个 configuration_*.py, 返回其中的 Config 类记录。"""
    source = path.read_bytes()
    tree = _py_parser().parse(source)
    root = tree.root_node

    # 递归收集所有 class_definition（含嵌套/带装饰器/被简单语句包裹的情况）。
    # 只查 root 直接子节点会漏掉大量带 @dataclass 装饰器的类。
    class_nodes: list = []
    def _collect(node):
        if node.type == "class_definition":
            class_nodes.append(node)
        for ch in node.children:
            _collect(ch)
    _collect(root)

    classes: list[dict] = []
    for node in class_nodes:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        class_name = name_node.text.decode("utf-8", errors="replace")
        # 只关注 Config 类 (命名约定)
        if not (class_name.endswith("Config") or "Config" in class_name):
            continue
        model_type, fields, attr_map, has_post = _extract_class_fields(node)
        super_node = node.child_by_field_name("superclasses")
        bases = []
        if super_node is not None:
            for arg in super_node.children:
                if arg.type in ("identifier", "attribute"):
                    bases.append(arg.text.decode("utf-8", errors="replace"))
        classes.append({
            "class_name": class_name,
            "file": str(path),
            "model_type": model_type or "",
            "bases": ",".join(bases),
            "fields": fields,
            "attribute_map": attr_map,
            "has_post_init": has_post,
            "is_strict": any("strict" in b for b in bases),
        })
    return classes


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


def _merge_lock(lock_path: Path, key: str, value: str) -> None:
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
    out_list = header if header else [f"# Updated: {datetime.now(_dt.timezone.utc).isoformat()}"]
    out_list.extend(body)
    lock_path.write_text("\n".join(out_list) + "\n", encoding="utf-8")


def extract(upstream: Path, out: Path) -> None:
    models_dir = upstream / "src" / "transformers" / "models"
    if not models_dir.exists():
        print(f"[extract_hf] ERROR: {models_dir} not found", file=sys.stderr)
        return

    all_classes: list[dict] = []
    config_files = sorted(models_dir.glob("*/configuration_*.py"))
    for cf in config_files:
        try:
            classes = parse_config_file(cf)
        except Exception as e:  # 单文件失败不阻断
            print(f"[extract_hf] WARN: parse failed {cf.name}: {e}", file=sys.stderr)
            continue
        all_classes.extend(classes)

    out.mkdir(parents=True, exist_ok=True)

    # ── 字段记录: 一个 model_type 的所有字段展开 ──
    field_rows: list[dict] = []
    for cls in all_classes:
        mt = cls["model_type"]
        for f in cls["fields"]:
            field_rows.append({
                "model_type": mt,
                "class_name": cls["class_name"],
                "field": f["name"],
                "type": f["type"],
                "has_default": f["has_default"],
                "default": f["default"],
                "derived": cls["has_post_init"],
            })

    (out / "hf-config-fields.toml").write_text(
        _dump_table(
            "config_field",
            field_rows,
            ["model_type", "class_name", "field", "type", "has_default", "default", "derived"],
        ),
        encoding="utf-8",
    )

    # ── 全局 attribute_map 汇总 (别名 → 规范名) ──
    alias_rows: list[dict] = []
    for cls in all_classes:
        for alias, canonical in cls["attribute_map"].items():
            alias_rows.append({
                "model_type": cls["model_type"],
                "alias": alias,
                "canonical": canonical,
                "class_name": cls["class_name"],
            })
    (out / "hf-attribute-map.toml").write_text(
        _dump_table("attribute_alias", alias_rows,
                    ["model_type", "alias", "canonical", "class_name"]),
        encoding="utf-8",
    )

    # ── model_type 索引 ──
    mt_rows = []
    for cls in all_classes:
        if cls["model_type"]:
            mt_rows.append({
                "model_type": cls["model_type"],
                "class_name": cls["class_name"],
                "field_count": len(cls["fields"]),
                "is_strict": cls["is_strict"],
                "derived": cls["has_post_init"],
            })
    (out / "hf-model-types.toml").write_text(
        _dump_table("model_type", mt_rows,
                    ["model_type", "class_name", "field_count", "is_strict", "derived"]),
        encoding="utf-8",
    )

    commit = _git_commit(upstream)
    _merge_lock(out / "UPSTREAM_COMMIT.lock", "transformers", commit)

    print(f"[extract_hf] classes={len(all_classes)} model_types={len(mt_rows)} "
          f"fields={len(field_rows)} aliases={len(alias_rows)} → {out}",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--upstream", type=Path,
                   default=here / ".upstream" / "transformers",
                   help="path to transformers checkout")
    p.add_argument("--out", type=Path,
                   default=repo_root / "generated" / "model-knowledge",
                   help="output dir for TOML")
    args = p.parse_args(argv)
    if not args.upstream.exists():
        print(f"[extract_hf] ERROR: upstream not found: {args.upstream}",
              file=sys.stderr)
        print("  run ./run_sync.sh first, or pass --upstream <transformers path>",
              file=sys.stderr)
        return 2
    extract(args.upstream, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
