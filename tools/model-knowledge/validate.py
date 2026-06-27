"""
校验 gllm FIELD_DEFS 对上游知识的覆盖, 输出 gap 报告。

退出码:
    0 = 无 gap
    1 = 有 gap (供 cron 判断是否通知)
    2 = 流程故障 (输入缺失/解析失败)

输入:
    --gllm-root      gllm 仓库根目录 (解析 Rust 知识表)
    --knowledge-dir  extract_*.py 的 TOML 输出目录

gap 维度:
    [MISSING GGUF KEYS]        llama.cpp 有但 FIELD_DEFS 的 gguf_keys 没用到的 key
    [MISSING ARCHITECTURES]    llama.cpp 有但 registry.rs ARCH_TABLE 没的架构
    [HF ALIASES NOT COVERED]   transformers attribute_map 里 gllm json_keys 没覆盖的别名
    [SEMANTIC MAPPING COVERAGE] llama.cpp get_key KV→field, gllm 是否有对应 canonical 字段
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# tree-sitter-rust (复用 ast_tools.py 的范式)
# ─────────────────────────────────────────────────────────────

_RUST_PARSER = None


def _rust_parser():
    global _RUST_PARSER
    if _RUST_PARSER is None:
        import tree_sitter_rust as tsrust
        from tree_sitter import Language, Parser
        _RUST_PARSER = Parser(Language(tsrust.language()))
    return _RUST_PARSER


def _find_all(node, type_name: str, out: list):
    if node.type == type_name:
        out.append(node)
    for c in node.children:
        _find_all(c, type_name, out)


def _fi_name_value(fi):
    """field_initializer → (name, value_node), 不依赖 field name binding。"""
    name = None
    val = None
    for c in fi.children:
        if c.type == "field_identifier":
            name = c.text.decode("utf-8", errors="replace")
        elif c.type == ":":
            pass
        else:
            val = c
    return name, val


def _strings_in_array(value_node) -> list[str]:
    """从 json_keys/gguf_keys 的 value 节点 (含 reference_expression 包裹) 提取字符串。"""
    arrs: list = []
    _find_all(value_node, "array_expression", arrs)
    result: list[str] = []
    for a in arrs:
        for c in a.children:
            if c.type == "string_literal":
                txt = c.text.decode("utf-8", errors="replace")
                # 去引号 (支持普通字符串, 不含 raw/byte 前缀因 static &str)
                if len(txt) >= 2 and txt[0] == '"' and txt[-1] == '"':
                    result.append(txt[1:-1])
                else:
                    result.append(txt)
    return result


# ─────────────────────────────────────────────────────────────
# 解析 FIELD_DEFS
# ─────────────────────────────────────────────────────────────

def parse_field_defs(path: Path) -> list[dict]:
    """解析 src/model_config_fragments/field_registry.inc.rs 的 FIELD_DEFS static。

    返回: [{canonical, json_keys: [str], gguf_keys: [str]}]
    """
    if not path.exists():
        return []
    src = path.read_bytes()
    tree = _rust_parser().parse(src)
    root = tree.root_node

    # 定位 static FIELD_DEFS
    target_static = None
    statics: list = []
    _find_all(root, "static_item", statics)
    for st in statics:
        for c in st.children:
            if c.type == "identifier" and c.text == b"FIELD_DEFS":
                target_static = st
                break
        if target_static:
            break
    if target_static is None:
        return []

    # 找所有 FieldDef struct_expression (type_identifier == FieldDef)
    structs: list = []
    _find_all(target_static, "struct_expression", structs)
    field_defs: list[dict] = []
    for s in structs:
        type_ids = [c for c in s.children if c.type == "type_identifier"]
        if not type_ids or type_ids[0].text != b"FieldDef":
            continue
        fil = [c for c in s.children if c.type == "field_initializer_list"]
        if not fil:
            continue
        canonical = ""
        json_keys: list[str] = []
        gguf_keys: list[str] = []
        kind_node = None
        for fi in fil[0].children:
            if fi.type != "field_initializer":
                continue
            nm, val = _fi_name_value(fi)
            if nm == "canonical" and val is not None:
                canonical = val.text.decode("utf-8", errors="replace").strip().strip('"')
            elif nm == "kind" and val is not None:
                kind_node = val
        # kind 内部 Alias { json_keys, gguf_keys }
        if kind_node is not None:
            inner_fils: list = []
            _find_all(kind_node, "field_initializer_list", inner_fils)
            for ifil in inner_fils:
                for ifi in ifil.children:
                    if ifi.type != "field_initializer":
                        continue
                    inm, inval = _fi_name_value(ifi)
                    if inm == "json_keys" and inval is not None:
                        json_keys = _strings_in_array(inval)
                    elif inm == "gguf_keys" and inval is not None:
                        gguf_keys = _strings_in_array(inval)
        if canonical:
            field_defs.append({
                "canonical": canonical,
                "json_keys": json_keys,
                "gguf_keys": gguf_keys,
            })
    return field_defs


# ─────────────────────────────────────────────────────────────
# 解析 ARCH_TABLE
# ─────────────────────────────────────────────────────────────

def parse_arch_table(path: Path) -> list[str]:
    """解析 src/arch/registry.rs 的 ARCH_TABLE, 返回 canonical 名集合。

    ARCH_TABLE: &[(&str, &str, &str, Option<&str>], 第 2 列是 canonical。
    每条形如 ("token", "canonical", "family", None)。
    """
    if not path.exists():
        return []
    src = path.read_bytes()
    tree = _rust_parser().parse(src)
    root = tree.root_node

    # 找 ARCH_TABLE (可能是 const_item 或 static_item)
    target = None
    decls: list = []
    _find_all(root, "const_item", decls)
    _find_all(root, "static_item", decls)
    for st in decls:
        for c in st.children:
            if c.type == "identifier" and c.text == b"ARCH_TABLE":
                target = st
                break
        if target:
            break
    if target is None:
        return []

    # 提取所有 tuple_expression (token, canonical, family, ...)
    canonicals: list[str] = []
    tuples: list = []
    _find_all(target, "tuple_expression", tuples)
    for tp in tuples:
        # tuple children: '(', string_literal, ',', string_literal, ',', ...
        strs = [c.text.decode("utf-8", errors="replace").strip('"')
                for c in tp.children if c.type == "string_literal"]
        if len(strs) >= 2:
            canonicals.append(strs[1])  # 第 2 列 canonical
    return canonicals


# ─────────────────────────────────────────────────────────────
# TOML 读取 (py311+ tomllib)
# ─────────────────────────────────────────────────────────────

def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


# ─────────────────────────────────────────────────────────────
# Gap 报告
# ─────────────────────────────────────────────────────────────

def build_report(field_defs: list[dict], arch_canonicals: list[str],
                 knowledge_dir: Path) -> tuple[str, bool]:
    """生成 GAP_REPORT.md, 返回 (报告文本, has_gap)。"""
    sections: list[str] = []
    has_gap = False

    # ── 汇总 gllm 已覆盖的 key 集合 ──
    gllf_json_keys: set[str] = set()
    gllm_gguf_keys: set[str] = set()
    for fd in field_defs:
        gllf_json_keys.update(fd["json_keys"])
        gllm_gguf_keys.update(fd["gguf_keys"])
    gllf_json_keys_norm = {k.lower() for k in gllf_json_keys}
    gllm_canonicals = {fd["canonical"].lower() for fd in field_defs}

    sections.append("# model-knowledge GAP REPORT")
    sections.append("")
    sections.append(f"- gllm FIELD_DEFS entries: {len(field_defs)}")
    sections.append(f"- gllm unique json_keys: {len(gllf_json_keys)}")
    sections.append(f"- gllm unique gguf_keys: {len(gllm_gguf_keys)}")
    sections.append(f"- gllm ARCH_TABLE canonicals: {len(arch_canonicals)}")
    sections.append("")

    # ── [MISSING GGUF KEYS] ──
    llama_kv = _load_toml(knowledge_dir / "llama-kv.toml")
    kv_entries = llama_kv.get("kv", []) if isinstance(llama_kv.get("kv"), list) else []
    # tomllib 把 [[kv]] 解析成 list[dict]; 我们的 writer 没用 [[kv]] 语法而是 [[kv]]?
    # 实际 _dump_table 写的是 [[name]] 格式, tomllib 会读成 dict{"kv":[...]}
    llama_keys = [e.get("key_string", "") for e in kv_entries
                  if isinstance(e, dict) and not e.get("has_arch_placeholder", False)]
    missing_kv = [k for k in llama_keys if k and k.lower() not in {g.lower() for g in gllm_gguf_keys}]
    # 注意: gguf_keys 在 FIELD_DEFS 是无 arch 前缀的, 运行时 format!("{arch}.{key}")
    # 这里我们比对 key 本体 (不含 arch 前缀)
    sections.append("## [MISSING GGUF KEYS]")
    sections.append("")
    sections.append(f"llama.cpp GGUF keys (non-arch-specific) not covered by FIELD_DEFS.gguf_keys.")
    sections.append("")
    if missing_kv:
        has_gap = True
        sections.append(f"**{len(missing_kv)} keys missing:**")
        sections.append("")
        for k in sorted(set(missing_kv)):
            sections.append(f"- `{k}`")
    else:
        sections.append("_(none — full coverage)_")
    sections.append("")

    # ── [MISSING ARCHITECTURES] ──
    llama_arch = _load_toml(knowledge_dir / "llama-arch.toml")
    arch_entries = llama_arch.get("arch", []) if isinstance(llama_arch.get("arch"), list) else []
    llama_arch_names = [e.get("arch_enum", "") for e in arch_entries if isinstance(e, dict)]
    arch_canonicals_lower = {c.lower() for c in arch_canonicals}
    missing_arch = [a for a in llama_arch_names
                    if a and a.lower() not in arch_canonicals_lower]
    sections.append("## [MISSING ARCHITECTURES]")
    sections.append("")
    sections.append(f"llama.cpp MODEL_ARCH names not in gllm ARCH_TABLE (canonical set).")
    sections.append("")
    if missing_arch:
        has_gap = True
        sections.append(f"**{len(missing_arch)} architectures missing:**")
        sections.append("")
        for a in sorted(set(missing_arch)):
            sections.append(f"- `{a}`")
    else:
        sections.append("_(none — full coverage)_")
    sections.append("")

    # ── [HF ALIASES NOT COVERED] ──
    hf_attr = _load_toml(knowledge_dir / "hf-attribute-map.toml")
    attr_entries = hf_attr.get("attribute_alias", []) if isinstance(hf_attr.get("attribute_alias"), list) else []
    aliases = [(e.get("alias", ""), e.get("canonical", ""), e.get("model_type", ""))
               for e in attr_entries if isinstance(e, dict)]
    # gllm 已覆盖: json_keys 集合 + canonical 集合
    not_covered = [(alias, canon, mt) for (alias, canon, mt) in aliases
                   if alias and alias.lower() not in gllf_json_keys_norm
                   and alias.lower() not in gllm_canonicals]
    sections.append("## [HF ALIASES NOT COVERED]")
    sections.append("")
    sections.append("transformers attribute_map aliases not in gllm FIELD_DEFS.json_keys.")
    sections.append("")
    if not_covered:
        has_gap = True
        sections.append(f"**{len(not_covered)} aliases not covered:**")
        sections.append("")
        sections.append("| model_type | alias | canonical |")
        sections.append("|------------|-------|-----------|")
        for alias, canon, mt in sorted(set(not_covered)):
            sections.append(f"| {mt} | `{alias}` | `{canon}` |")
    else:
        sections.append("_(none — full coverage)_")
    sections.append("")

    # ── [SEMANTIC MAPPING COVERAGE] ──
    llama_sem = _load_toml(knowledge_dir / "llama-semantics.toml")
    sem_entries = llama_sem.get("semantic", []) if isinstance(llama_sem.get("semantic"), list) else []
    # hparams field 名 → gllm canonical 大致映射靠名字相似度
    # 这里只列出 llama.cpp 提取的 (KV→field), 标注 gllm 是否有同名/相似 canonical
    sections.append("## [SEMANTIC MAPPING COVERAGE]")
    sections.append("")
    sections.append("llama.cpp get_key(KV, hparams.X) — X covered by a gllm canonical field?")
    sections.append("")
    if sem_entries:
        sections.append("| KV enum | hparams field | gllm canonical match |")
        sections.append("|---------|---------------|----------------------|")
        for e in sorted({(d.get("kv_enum",""), d.get("hparams_field","")) for d in sem_entries if isinstance(d, dict)}):
            kv, hp = e
            matched = hp.lower() in gllm_canonicals
            mark = f"`{hp}`" if matched else f"`{hp}` _(no exact match)_"
            sections.append(f"| {kv} | {hp} | {'yes' if matched else 'no'} |")
        # 统计覆盖率
        covered = sum(1 for d in sem_entries if isinstance(d, dict)
                      and d.get("hparams_field","").lower() in gllm_canonicals)
        if covered < len(sem_entries):
            has_gap = True
    else:
        sections.append("_(no semantic data extracted)_")
    sections.append("")

    report = "\n".join(sections) + "\n"
    return report, has_gap


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gllm-root", type=Path, default=repo_root,
                   help="gllm repo root (default: ../../)")
    p.add_argument("--knowledge-dir", type=Path,
                   default=repo_root / "generated" / "model-knowledge",
                   help="TOML knowledge dir from extract_*.py")
    args = p.parse_args(argv)

    field_defs_path = args.gllm_root / "src" / "model_config_fragments" / "field_registry.inc.rs"
    arch_path = args.gllm_root / "src" / "arch" / "registry.rs"

    field_defs = parse_field_defs(field_defs_path)
    arch_canonicals = parse_arch_table(arch_path)
    if not field_defs:
        print(f"[validate] ERROR: failed to parse FIELD_DEFS from {field_defs_path}",
              file=sys.stderr)
        return 2

    report, has_gap = build_report(field_defs, arch_canonicals, args.knowledge_dir)
    out_path = args.knowledge_dir / "GAP_REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[validate] report → {out_path}", file=sys.stderr)
    print(f"[validate] {'GAP DETECTED (exit 1)' if has_gap else 'no gap (exit 0)'}",
          file=sys.stderr)
    return 1 if has_gap else 0


if __name__ == "__main__":
    sys.exit(main())
