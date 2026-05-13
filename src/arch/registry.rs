//! 架构别名注册表 — 纯 const 查找表
//!
//! 提供 GGUF arch token / HF `architectures` 值 → canonical name 的映射。
//! 不依赖 YAML 模板。auto_graph 从 tensor names 推导架构，此处仅提供 token 别名。

use crate::manifest::ArchFamily;

/// (token, canonical_name, family, moe_router)
///
/// token 已归一化（小写，`-`/`.` → `_`）。
/// 查找时对输入做相同归一化。
const ARCH_TABLE: &[(&str, &str, &str, Option<&str>)] = &[
    // Decoder family
    ("qwen3",           "qwen3",      "decoder", Some("qwen")),
    ("qwen3moe",        "qwen3",      "decoder", Some("qwen")),
    ("qwen2_5",         "qwen3",      "decoder", Some("qwen")),
    ("qwen2",           "qwen3",      "decoder", Some("qwen")),
    ("llama",           "llama",      "decoder", None),
    ("smollm",          "llama",      "decoder", None),
    ("internlm",        "llama",      "decoder", None),
    ("mistral3",        "mistral3",   "decoder", None),
    ("ministral",       "mistral3",   "decoder", None),
    ("phi4",            "phi4",       "decoder", None),
    ("phi4_mini",       "phi4",       "decoder", None),
    ("phi3",            "phi4",       "decoder", None),
    ("glm4",            "glm4",       "decoder", Some("mixtral")),
    ("glm5",            "glm4",       "decoder", Some("mixtral")),
    ("chatglm",         "glm4",       "decoder", Some("mixtral")),
    ("gemma4",          "gemma4",     "decoder", None),
    ("deepseek",        "deepseek",   "decoder", Some("deepseek")),
    ("deepseek_v3",     "deepseek",   "decoder", Some("deepseek")),
    ("gptoss",          "gptoss",     "decoder", Some("gptoss")),
    // Encoder family
    ("xlmr",            "xlmr",       "encoder", None),
    ("xlm_roberta",     "xlmr",       "encoder", None),
    ("bert",            "xlmr",       "encoder", None),
    ("roberta",         "xlmr",       "encoder", None),
    // Audio/Vision
    ("usm_conformer",   "usm_conformer", "encoder", None),
    ("siglip",          "siglip",     "encoder", None),
    ("siglip_vision_model", "siglip", "encoder", None),
];

/// 归一化 architecture token: 小写 + `-`/`.` → `_`
fn normalize_token(token: &str) -> String {
    token
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
        .map(|ch| match ch {
            '-' | '.' => '_',
            _ => ch.to_ascii_lowercase(),
        })
        .collect()
}

/// 按 architecture token 查找 canonical name。
pub fn resolve_template_name(token: &str) -> Option<&'static str> {
    let norm = normalize_token(token);
    // 直接匹配
    for &(t, name, _, _) in ARCH_TABLE {
        if t == norm { return Some(name); }
    }
    // 自动派生 ForCausalLM 后缀
    if norm.ends_with("forcausallm") {
        let base = &norm[..norm.len() - "forcausallm".len()];
        for &(t, name, _, _) in ARCH_TABLE {
            if t == base { return Some(name); }
        }
    }
    None
}

/// 按 architecture token 查找 ArchFamily。
pub fn resolve_family(token: &str) -> Option<ArchFamily> {
    let norm = normalize_token(token);
    for &(t, _, family, _) in ARCH_TABLE {
        if t == norm {
            return Some(match family {
                "encoder" => ArchFamily::Encoder,
                _ => ArchFamily::Decoder,
            });
        }
    }
    // 尝试 ForCausalLM 后缀
    if norm.ends_with("forcausallm") {
        let base = &norm[..norm.len() - "forcausallm".len()];
        for &(t, _, family, _) in ARCH_TABLE {
            if t == base {
                return Some(match family {
                    "encoder" => ArchFamily::Encoder,
                    _ => ArchFamily::Decoder,
                });
            }
        }
    }
    None
}

/// 按 canonical name 查找 MoE 路由器类型。
pub fn resolve_moe_router(name: &str) -> Option<crate::manifest::RouterType> {
    for &(_, n, _, router) in ARCH_TABLE {
        if n == name {
            return router.map(|r| match r {
                "deepseek" => crate::manifest::RouterType::DeepSeek,
                "qwen" => crate::manifest::RouterType::Qwen,
                "mixtral" => crate::manifest::RouterType::Mixtral,
                "gptoss" => crate::manifest::RouterType::GptOss,
                _ => crate::manifest::RouterType::Unknown,
            });
        }
    }
    None
}

/// 校验 name 是否为已知 canonical name。
pub fn is_valid_template(name: &str) -> bool {
    ARCH_TABLE.iter().any(|&(_, n, _, _)| n == name)
}

/// 按模板名获取模板 (兼容旧签名，不再返回 ArchTemplate)。
/// 返回 None — auto_graph 已替代 template 系统。
pub fn get_template(_name: &str) -> Option<&'static ()> {
    None
}

/// 初始化 (兼容旧签名，空操作)。
pub fn register_builtin_templates() {}

/// 兼容旧导出
pub type ArchRegistry = ();
pub static REGISTRY: std::sync::OnceLock<()> = std::sync::OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_tokens() {
        assert_eq!(resolve_template_name("qwen3"), Some("qwen3"));
        assert_eq!(resolve_template_name("LlamaForCausalLM"), Some("llama"));
        assert_eq!(resolve_template_name("xlmr"), Some("xlmr"));
        assert_eq!(resolve_template_name("deepseek"), Some("deepseek"));
        assert_eq!(resolve_template_name("glm4"), Some("glm4"));
        assert_eq!(resolve_template_name("chatglm"), Some("glm4"));
        assert_eq!(resolve_template_name("phi3"), Some("phi4"));
        assert_eq!(resolve_template_name("Phi3ForCausalLM"), Some("phi4"));
        assert_eq!(resolve_template_name("nonexistent"), None);
    }

    #[test]
    fn family_resolution() {
        assert_eq!(resolve_family("qwen3"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("xlmr"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("siglip"), Some(ArchFamily::Encoder));
    }

    #[test]
    fn moe_router_resolution() {
        assert_eq!(resolve_moe_router("deepseek"), Some(crate::manifest::RouterType::DeepSeek));
        assert_eq!(resolve_moe_router("qwen3"), Some(crate::manifest::RouterType::Qwen));
        assert_eq!(resolve_moe_router("llama"), None);
    }

    #[test]
    fn is_valid() {
        assert!(is_valid_template("qwen3"));
        assert!(is_valid_template("xlmr"));
        assert!(!is_valid_template("nonexistent"));
    }
}
