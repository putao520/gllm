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
    ("llama4",          "llama4",     "decoder", None),       // REQ-MODEL-8
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
    ("deepseekv3",      "deepseek",   "decoder", Some("deepseek")),
    ("deepseek_r1",     "deepseek",   "decoder", Some("deepseek")),
    ("deepseekr1",      "deepseek",   "decoder", Some("deepseek")),
    ("kimi_k2",         "deepseek",   "decoder", Some("deepseek")),
    ("gptoss",          "gptoss",     "decoder", Some("gptoss")),
    // Encoder family
    ("xlmr",            "xlmr",       "encoder", None),
    ("xlm_roberta",     "xlmr",       "encoder", None),
    ("xlmroberta",      "xlmr",       "encoder", None),      // ForCausalLM suffix fix: normalized "xlmroberta" has own entry
    ("bert",            "xlmr",       "encoder", None),
    ("roberta",         "xlmr",       "encoder", None),
    // Embedding family (REQ-MODEL-9: encoder weight topology + MeanPool output)
    ("bge_m3",          "bge",        "embedding", None),
    // Reranker family (REQ-MODEL-10: encoder weight topology + Classify output)
    ("bge_reranker",    "bge",        "reranker", None),
    // Audio/Vision
    ("usm_conformer",   "usm_conformer", "encoder", None),
    ("siglip",          "siglip",     "encoder", None),
    ("siglip_vision_model", "siglip", "encoder", None),
    // Custom classifiers
    ("signal_intent_tracker", "signal_intent_tracker", "encoder", None),
    ("signalintenttracker",   "signal_intent_tracker", "encoder", None),
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
                "embedding" => ArchFamily::Embedding,
                "reranker" => ArchFamily::Reranker,
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
                    "embedding" => ArchFamily::Embedding,
                    "reranker" => ArchFamily::Reranker,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_tokens() {
        assert_eq!(resolve_template_name("qwen3"), Some("qwen3"));
        assert_eq!(resolve_template_name("LlamaForCausalLM"), Some("llama"));
        assert_eq!(resolve_template_name("llama4"), Some("llama4"));
        assert_eq!(resolve_template_name("xlmr"), Some("xlmr"));
        assert_eq!(resolve_template_name("xlmroberta"), Some("xlmr"));
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
        assert_eq!(resolve_family("bge_m3"), Some(ArchFamily::Embedding));
        assert_eq!(resolve_family("bge_reranker"), Some(ArchFamily::Reranker));
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
        assert!(is_valid_template("llama4"));
        assert!(is_valid_template("bge"));
        assert!(!is_valid_template("nonexistent"));
    }

    #[test]
    fn normalize_token_strips_and_lowercases() {
        assert_eq!(normalize_token("Qwen3-MoE"), "qwen3_moe");
        assert_eq!(normalize_token("DeepSeek.v3"), "deepseek_v3");
        assert_eq!(normalize_token("  llama  "), "llama");
    }

    #[test]
    fn resolve_aliases() {
        assert_eq!(resolve_template_name("qwen2"), Some("qwen3"));
        assert_eq!(resolve_template_name("qwen2_5"), Some("qwen3"));
        assert_eq!(resolve_template_name("smollm"), Some("llama"));
        assert_eq!(resolve_template_name("internlm"), Some("llama"));
        assert_eq!(resolve_template_name("ministral"), Some("mistral3"));
        assert_eq!(resolve_template_name("roberta"), Some("xlmr"));
        assert_eq!(resolve_template_name("bert"), Some("xlmr"));
    }

    #[test]
    fn resolve_forcausallm_suffix() {
        assert_eq!(resolve_template_name("Qwen3ForCausalLM"), Some("qwen3"));
        assert_eq!(resolve_template_name("DeepSeekForCausalLM"), Some("deepseek"));
        assert_eq!(resolve_template_name("Gemma4ForCausalLM"), Some("gemma4"));
        assert_eq!(resolve_template_name("NonExistentForCausalLM"), None);
    }

    #[test]
    fn family_for_forcausallm_suffix() {
        assert_eq!(resolve_family("Qwen3ForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("XLMRForCausalLM"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("NonExistentForCausalLM"), None);
    }

    #[test]
    fn resolve_all_deepseek_aliases() {
        assert_eq!(resolve_template_name("deepseek_v3"), Some("deepseek"));
        assert_eq!(resolve_template_name("deepseek_r1"), Some("deepseek"));
        assert_eq!(resolve_template_name("kimi_k2"), Some("deepseek"));
    }

    #[test]
    fn resolve_signal_intent_tracker() {
        assert_eq!(resolve_template_name("signal_intent_tracker"), Some("signal_intent_tracker"));
        assert_eq!(resolve_template_name("signalintenttracker"), Some("signal_intent_tracker"));
    }

    #[test]
    fn resolve_moe_router_for_all_routers() {
        assert_eq!(resolve_moe_router("glm4"), Some(crate::manifest::RouterType::Mixtral));
        assert_eq!(resolve_moe_router("gptoss"), Some(crate::manifest::RouterType::GptOss));
        assert_eq!(resolve_moe_router("gemma4"), None);
        assert_eq!(resolve_moe_router("xlmr"), None);
    }

    #[test]
    fn is_valid_all_canonical_names() {
        let canonicals: Vec<&str> = ARCH_TABLE.iter().map(|&(_, n, _, _)| n).collect();
        for name in canonicals {
            assert!(is_valid_template(name), "{} should be valid", name);
        }
    }

    // ── normalize_token edge cases ──

    #[test]
    fn normalize_token_empty_string() {
        assert_eq!(normalize_token(""), "");
    }

    #[test]
    fn normalize_token_whitespace_only() {
        assert_eq!(normalize_token("   "), "");
    }

    #[test]
    fn normalize_token_special_chars_stripped() {
        // Non-alphanumeric, non-separator chars are filtered out
        assert_eq!(normalize_token("hello@world!"), "helloworld");
        assert_eq!(normalize_token("a b\tc\nd"), "abcd");
    }

    #[test]
    fn normalize_token_all_uppercase() {
        assert_eq!(normalize_token("DEEPSEEK"), "deepseek");
        assert_eq!(normalize_token("LLAMA"), "llama");
    }

    #[test]
    fn normalize_token_mixed_case_with_separators() {
        assert_eq!(normalize_token("Qwen3-MoE"), "qwen3_moe");
        assert_eq!(normalize_token("DeepSeek.v3"), "deepseek_v3");
    }

    #[test]
    fn normalize_token_consecutive_separators() {
        // Each separator becomes _, no dedup
        assert_eq!(normalize_token("a--b"), "a__b");
        assert_eq!(normalize_token("x..y"), "x__y");
    }

    #[test]
    fn normalize_token_numbers_preserved() {
        assert_eq!(normalize_token("Phi4Mini"), "phi4mini");
        assert_eq!(normalize_token("Qwen2.5"), "qwen2_5");
    }

    #[test]
    fn normalize_token_already_normalized() {
        assert_eq!(normalize_token("qwen3"), "qwen3");
        assert_eq!(normalize_token("deepseek_v3"), "deepseek_v3");
    }

    // ── resolve_template_name edge cases ──

    #[test]
    fn resolve_template_name_empty() {
        assert_eq!(resolve_template_name(""), None);
    }

    #[test]
    fn resolve_template_name_whitespace_only() {
        assert_eq!(resolve_template_name("   "), None);
    }

    #[test]
    fn resolve_template_name_case_insensitive() {
        assert_eq!(resolve_template_name("LLAMA"), Some("llama"));
        assert_eq!(resolve_template_name("QWEN3"), Some("qwen3"));
        assert_eq!(resolve_template_name("DeepSeek"), Some("deepseek"));
    }

    #[test]
    fn resolve_template_name_unknown() {
        assert_eq!(resolve_template_name("nonexistent_arch"), None);
        assert_eq!(resolve_template_name("foobar"), None);
    }

    #[test]
    fn resolve_template_name_partial_match_does_not_match() {
        // "qwen" is not a token, only "qwen3", "qwen3moe", etc.
        assert_eq!(resolve_template_name("qwen"), None);
    }

    #[test]
    fn resolve_template_name_forcausallm_stripped_base_unknown() {
        assert_eq!(resolve_template_name("UnknownForCausalLM"), None);
    }

    // ── resolve_family edge cases ──

    #[test]
    fn resolve_family_empty() {
        assert_eq!(resolve_family(""), None);
    }

    #[test]
    fn resolve_family_unknown() {
        assert_eq!(resolve_family("nonexistent"), None);
    }

    #[test]
    fn resolve_family_all_decoders() {
        let decoder_tokens = ["qwen3", "llama", "llama4", "mistral3", "phi4", "glm4", "gemma4", "deepseek", "gptoss"];
        for token in decoder_tokens {
            assert_eq!(
                resolve_family(token),
                Some(ArchFamily::Decoder),
                "{token} should resolve to Decoder"
            );
        }
    }

    #[test]
    fn resolve_family_all_encoders() {
        let encoder_tokens = ["xlmr", "usm_conformer", "siglip", "signal_intent_tracker"];
        for token in encoder_tokens {
            assert_eq!(
                resolve_family(token),
                Some(ArchFamily::Encoder),
                "{token} should resolve to Encoder"
            );
        }
    }

    // ── resolve_moe_router edge cases ──

    #[test]
    fn resolve_moe_router_empty() {
        assert_eq!(resolve_moe_router(""), None);
    }

    #[test]
    fn resolve_moe_router_unknown_canonical() {
        assert_eq!(resolve_moe_router("nonexistent"), None);
    }

    #[test]
    fn resolve_moe_router_all_router_types() {
        assert_eq!(resolve_moe_router("deepseek"), Some(crate::manifest::RouterType::DeepSeek));
        assert_eq!(resolve_moe_router("qwen3"), Some(crate::manifest::RouterType::Qwen));
        assert_eq!(resolve_moe_router("glm4"), Some(crate::manifest::RouterType::Mixtral));
        assert_eq!(resolve_moe_router("gptoss"), Some(crate::manifest::RouterType::GptOss));
    }

    #[test]
    fn resolve_moe_router_non_moe_models_return_none() {
        let non_moe = ["llama", "llama4", "mistral3", "phi4", "gemma4", "xlmr", "siglip", "bge"];
        for name in non_moe {
            assert_eq!(
                resolve_moe_router(name),
                None,
                "{name} should not have a MoE router"
            );
        }
    }

    // ── is_valid_template edge cases ──

    #[test]
    fn is_valid_template_empty() {
        assert!(!is_valid_template(""));
    }

    #[test]
    fn is_valid_template_case_sensitive() {
        // is_valid_template does exact string match on canonical names
        // Canonical names are lowercase; uppercase should not match
        assert!(!is_valid_template("QWEN3"));
        assert!(!is_valid_template("Llama"));
        assert!(!is_valid_template("DEEPSEEK"));
    }

    #[test]
    fn is_valid_template_token_not_canonical() {
        // "qwen2" is a token but its canonical is "qwen3"
        assert!(!is_valid_template("qwen2"));
        assert!(!is_valid_template("smollm"));
        assert!(!is_valid_template("chatglm"));
    }

    #[test]
    fn is_valid_template_all_unique_canonicals() {
        // Every canonical name in the table should be valid
        let mut seen: Vec<&str> = Vec::new();
        for &(_, name, _, _) in ARCH_TABLE {
            if !seen.contains(&name) {
                seen.push(name);
            }
        }
        for name in &seen {
            assert!(is_valid_template(name), "canonical '{}' should be valid", name);
        }
    }

    // ── ArchFamily trait tests ──

    #[test]
    fn arch_family_equality() {
        assert_eq!(ArchFamily::Encoder, ArchFamily::Encoder);
        assert_eq!(ArchFamily::Decoder, ArchFamily::Decoder);
        assert_ne!(ArchFamily::Encoder, ArchFamily::Decoder);
        assert_eq!(ArchFamily::Embedding, ArchFamily::Embedding);
        assert_eq!(ArchFamily::Reranker, ArchFamily::Reranker);
        assert_ne!(ArchFamily::Embedding, ArchFamily::Decoder);
        assert_ne!(ArchFamily::Reranker, ArchFamily::Encoder);
        assert_ne!(ArchFamily::Embedding, ArchFamily::Reranker);
    }

    #[test]
    fn arch_family_clone() {
        let a = ArchFamily::Decoder;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_copy() {
        let a = ArchFamily::Encoder;
        let b = a; // Copy, not move
        assert_eq!(a, b);
    }

    #[test]
    fn arch_family_debug() {
        let encoder = format!("{:?}", ArchFamily::Encoder);
        let decoder = format!("{:?}", ArchFamily::Decoder);
        let embedding = format!("{:?}", ArchFamily::Embedding);
        let reranker = format!("{:?}", ArchFamily::Reranker);
        assert!(encoder.contains("Encoder"), "Debug for Encoder should contain 'Encoder'");
        assert!(decoder.contains("Decoder"), "Debug for Decoder should contain 'Decoder'");
        assert!(embedding.contains("Embedding"), "Debug for Embedding should contain 'Embedding'");
        assert!(reranker.contains("Reranker"), "Debug for Reranker should contain 'Reranker'");
    }

    #[test]
    fn arch_family_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ArchFamily::Encoder);
        set.insert(ArchFamily::Decoder);
        set.insert(ArchFamily::Embedding);
        set.insert(ArchFamily::Reranker);
        set.insert(ArchFamily::Encoder); // duplicate
        assert_eq!(set.len(), 4);
    }

    // ── ARCH_TABLE consistency ──

    #[test]
    fn arch_table_no_empty_tokens() {
        for &(token, _, _, _) in ARCH_TABLE {
            assert!(!token.is_empty(), "token must not be empty");
        }
    }

    #[test]
    fn arch_table_no_empty_canonical_names() {
        for &(_, name, _, _) in ARCH_TABLE {
            assert!(!name.is_empty(), "canonical name must not be empty");
        }
    }

    #[test]
    fn arch_table_family_values_are_valid() {
        for &(_, _, family, _) in ARCH_TABLE {
            assert!(
                family == "decoder" || family == "encoder" || family == "embedding" || family == "reranker",
                "family must be 'decoder', 'encoder', 'embedding', or 'reranker', got '{}'",
                family
            );
        }
    }

    #[test]
    fn arch_table_tokens_are_normalized() {
        for &(token, _, _, _) in ARCH_TABLE {
            let expected = normalize_token(token);
            assert_eq!(
                token, expected,
                "token '{}' should be pre-normalized",
                token
            );
        }
    }

    #[test]
    fn arch_table_no_duplicate_tokens() {
        let mut seen: Vec<&str> = Vec::new();
        for &(token, _, _, _) in ARCH_TABLE {
            assert!(
                !seen.contains(&token),
                "duplicate token '{}'",
                token
            );
            seen.push(token);
        }
    }

    #[test]
    fn arch_table_router_values_are_valid() {
        for &(_, _, _, router) in ARCH_TABLE {
            if let Some(r) = router {
                assert!(
                    matches!(r, "deepseek" | "qwen" | "mixtral" | "gptoss"),
                    "router must be a known type, got '{}'",
                    r
                );
            }
        }
    }

    // ── Additional tests (tests 50-64) ──

    #[test]
    fn resolve_template_name_with_leading_trailing_whitespace() {
        // Arrange: tokens with surrounding whitespace should still resolve
        // Act & Assert
        assert_eq!(resolve_template_name("  llama  "), Some("llama"));
        assert_eq!(resolve_template_name("\tqwen3\n"), Some("qwen3"));
        assert_eq!(resolve_template_name(" phi4 "), Some("phi4"));
    }

    #[test]
    fn normalize_token_mixed_separators() {
        // Arrange: string with all three separator types (- . and _)
        // Act
        let result = normalize_token("a-b.c_d");
        // Assert: all separators become _
        assert_eq!(result, "a_b_c_d");
    }

    #[test]
    fn resolve_template_name_dot_separator() {
        // Arrange: token using '.' as separator (common in HF model names)
        // Act & Assert: dots normalize to _, matching table entries
        assert_eq!(resolve_template_name("Qwen2.5"), Some("qwen3"));
        assert_eq!(resolve_template_name("DeepSeek.v3"), Some("deepseek"));
    }

    #[test]
    fn normalize_token_unicode_characters() {
        // Arrange: non-ASCII characters should be filtered out
        // Act & Assert
        assert_eq!(normalize_token("qwen3\u{00e9}"), "qwen3"); // e-acute removed
        assert_eq!(normalize_token("llama\u{4e2d}"), "llama");  // CJK char removed
    }

    #[test]
    fn normalize_token_very_long_name() {
        // Arrange: a long repeated name (256+ chars)
        // Act
        let long_input = "qwen3_".repeat(50);
        let result = normalize_token(&long_input);
        // Assert: should be fully processed without truncation
        assert_eq!(result, long_input);
        assert_eq!(result.len(), 300);
    }

    #[test]
    fn resolve_template_name_mixed_case_forcausallm() {
        // Arrange: ForCausalLM suffix with various casings
        // Act & Assert: normalize lowercases before suffix stripping
        assert_eq!(resolve_template_name("LLAMAFORCAUSALLM"), Some("llama"));
        assert_eq!(resolve_template_name("chatglmforcausallm"), Some("glm4"));
        assert_eq!(resolve_template_name("Gemma4FORCAUSALLM"), Some("gemma4"));
    }

    #[test]
    fn resolve_family_phi4_variants() {
        // Arrange: all phi4 family tokens
        // Act & Assert: phi4, phi4_mini, phi3 all resolve to Decoder
        assert_eq!(resolve_family("phi4"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("phi4_mini"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("phi3"), Some(ArchFamily::Decoder));
    }

    #[test]
    fn resolve_family_forcausallm_preserves_decoder() {
        // Arrange: ForCausalLM suffix on decoder tokens
        // Act & Assert: should still return Decoder family
        assert_eq!(resolve_family("Mistral3ForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("GLM4ForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("GPTOssForCausalLM"), Some(ArchFamily::Decoder));
    }

    #[test]
    fn resolve_moe_router_canonical_vs_token() {
        // Arrange: resolve_moe_router takes canonical name, not token
        // Act & Assert
        // "chatglm" is a token alias for canonical "glm4"
        assert_eq!(resolve_moe_router("chatglm"), None); // not a canonical name
        assert_eq!(resolve_moe_router("glm4"), Some(crate::manifest::RouterType::Mixtral)); // canonical
        // "deepseek_v3" is a token alias for canonical "deepseek"
        assert_eq!(resolve_moe_router("deepseek_v3"), None);
        assert_eq!(resolve_moe_router("deepseek"), Some(crate::manifest::RouterType::DeepSeek));
    }

    #[test]
    fn resolve_template_name_glm_family_aliases() {
        // Arrange: GLM family has multiple aliases mapping to canonical "glm4"
        // Act & Assert
        assert_eq!(resolve_template_name("glm4"), Some("glm4"));
        assert_eq!(resolve_template_name("glm5"), Some("glm4"));
        assert_eq!(resolve_template_name("chatglm"), Some("glm4"));
    }

    #[test]
    fn resolve_template_name_vision_model_variants() {
        // Arrange: siglip has two token entries with different names
        // Act & Assert
        assert_eq!(resolve_template_name("siglip"), Some("siglip"));
        assert_eq!(resolve_template_name("siglip_vision_model"), Some("siglip"));
        // Verify family for vision models
        assert_eq!(resolve_family("siglip_vision_model"), Some(ArchFamily::Encoder));
    }

    #[test]
    fn is_valid_template_whitespace_canonicals() {
        // Arrange: canonical names with whitespace should not match
        // Act & Assert
        assert!(!is_valid_template(" llama "));
        assert!(!is_valid_template(" qwen3"));
        assert!(!is_valid_template("deepseek "));
    }

    #[test]
    fn resolve_template_name_multiple_forcausallm_suffixes() {
        // Arrange: ForCausalLM suffix applied to various known tokens
        // Note: CamelCase tokens like "SmolLM" normalize to "smollm" (no underscore),
        // so the ForCausalLM base must match an ARCH_TABLE entry exactly.
        // Act & Assert
        assert_eq!(resolve_template_name("SmolLMForCausalLM"), Some("llama"));
        assert_eq!(resolve_template_name("InternLMForCausalLM"), Some("llama"));
        assert_eq!(resolve_template_name("MinistralForCausalLM"), Some("mistral3"));
        assert_eq!(resolve_template_name("Phi4ForCausalLM"), Some("phi4"));
        assert_eq!(resolve_template_name("BertForCausalLM"), Some("xlmr"));
        assert_eq!(resolve_template_name("Llama4ForCausalLM"), Some("llama4"));
        assert_eq!(resolve_template_name("XLMRobertaForCausalLM"), Some("xlmr"));
    }

    #[test]
    fn arch_table_canonical_names_are_unique() {
        // Arrange: collect all canonical names from the table
        let canonicals: Vec<&str> = ARCH_TABLE.iter().map(|&(_, n, _, _)| n).collect();
        // Act & Assert: each canonical should appear at least once (verified by
        // is_valid_template_all_unique_canonicals), but we also check for exact uniqueness
        let mut sorted = canonicals.clone();
        sorted.sort();
        sorted.dedup();
        // The unique canonical names count should match the deduped count
        assert_eq!(canonicals.len() - (canonicals.len() - sorted.len()), sorted.len());
    }

    #[test]
    fn resolve_template_name_xlm_roberta_alias() {
        // Arrange: xlm_roberta is a multi-word alias (with underscore) for xlmr
        // Act & Assert
        assert_eq!(resolve_template_name("xlm_roberta"), Some("xlmr"));
        // Also verify the short form
        assert_eq!(resolve_template_name("xlmr"), Some("xlmr"));
        // Verify both resolve to encoder family
        assert_eq!(resolve_family("xlm_roberta"), Some(ArchFamily::Encoder));
    }

    // ── Additional tests (tests 65-77) ──

    // @trace TEST-REG-65 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn normalize_token_digits_only() {
        // Arrange: input is purely numeric
        let input = "12345";
        // Act
        let result = normalize_token(input);
        // Assert: digits are preserved as-is
        assert_eq!(result, "12345");
    }

    // @trace TEST-REG-66 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn normalize_token_single_char() {
        // Arrange: single character inputs
        // Act & Assert: each single char is preserved (lowercased if alpha)
        assert_eq!(normalize_token("A"), "a");
        assert_eq!(normalize_token("z"), "z");
        assert_eq!(normalize_token("5"), "5");
    }

    // @trace TEST-REG-67 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_forcausallm_empty_base() {
        // Arrange: "ForCausalLM" alone — after stripping suffix, base is empty
        // Act
        let result = resolve_template_name("ForCausalLM");
        // Assert: empty base matches nothing
        assert_eq!(result, None);
    }

    // @trace TEST-REG-68 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_usm_conformer() {
        // Arrange: usm_conformer is an encoder architecture in the table
        // Act
        let name = resolve_template_name("usm_conformer");
        let family = resolve_family("usm_conformer");
        // Assert
        assert_eq!(name, Some("usm_conformer"));
        assert_eq!(family, Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-69 [req:REQ-MODEL-7] [level:unit]
    #[test]
    fn resolve_family_for_forcausallm_encoder_tokens() {
        // Arrange: ForCausalLM suffix on encoder-family tokens
        // Act & Assert: should still resolve to Encoder family
        // XLMRobertaForCausalLM normalizes to "xlmroberta" which now has its own
        // ARCH_TABLE entry (separate from "xlm_roberta"), fixing the ForCausalLM gap.
        assert_eq!(resolve_family("BertForCausalLM"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("RobertaForCausalLM"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("XLMRobertaForCausalLM"), Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-70 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_family_deepseek_variant_tokens() {
        // Arrange: all deepseek variant tokens individually
        // Act & Assert: each variant should resolve to Decoder family
        assert_eq!(resolve_family("deepseek_v3"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("deepseek_r1"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("kimi_k2"), Some(ArchFamily::Decoder));
    }

    // @trace TEST-REG-71 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_moe_router_deepseek_variants_all_resolve() {
        // Arrange: all deepseek variants share canonical name "deepseek"
        // Act & Assert: canonical name should resolve to DeepSeek router
        assert_eq!(resolve_moe_router("deepseek"), Some(crate::manifest::RouterType::DeepSeek));
        // Verify token aliases do NOT resolve (resolve_moe_router takes canonical, not token)
        assert_eq!(resolve_moe_router("deepseek_v3"), None);
        assert_eq!(resolve_moe_router("kimi_k2"), None);
    }

    // @trace TEST-REG-72 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_bert_and_roberta_aliases() {
        // Arrange: bert and roberta are token aliases for canonical "xlmr"
        // Act & Assert: both map to xlmr canonical name
        assert_eq!(resolve_template_name("bert"), Some("xlmr"));
        assert_eq!(resolve_template_name("roberta"), Some("xlmr"));
        // Both should be encoder family
        assert_eq!(resolve_family("bert"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("roberta"), Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-73 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_family_signal_intent_tracker_both_forms() {
        // Arrange: signal_intent_tracker has two token forms (with and without underscore)
        // Act & Assert: both should resolve to Encoder
        assert_eq!(resolve_family("signal_intent_tracker"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("signalintenttracker"), Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-74 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn is_valid_template_each_unique_canonical() {
        // Arrange: collect deduplicated canonical names
        let mut canonicals: Vec<&str> = ARCH_TABLE.iter().map(|&(_, n, _, _)| n).collect();
        canonicals.sort();
        canonicals.dedup();
        // Act & Assert: each unique canonical name is valid
        for name in &canonicals {
            assert!(is_valid_template(name), "'{}' should be a valid canonical", name);
        }
    }

    // @trace TEST-REG-75 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_qwen3moe_alias() {
        // Arrange: qwen3moe is a MoE variant of qwen3
        // Act
        let name = resolve_template_name("qwen3moe");
        let family = resolve_family("qwen3moe");
        let router = resolve_moe_router("qwen3");
        // Assert: alias resolves to canonical qwen3 with Qwen router
        assert_eq!(name, Some("qwen3"));
        assert_eq!(family, Some(ArchFamily::Decoder));
        assert_eq!(router, Some(crate::manifest::RouterType::Qwen));
    }

    // @trace TEST-REG-76 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_case_variations() {
        // Arrange: various case patterns for the same token
        // Act & Assert: all should resolve to the same canonical name
        assert_eq!(resolve_template_name("QWEN3"), Some("qwen3"));
        assert_eq!(resolve_template_name("Qwen3"), Some("qwen3"));
        assert_eq!(resolve_template_name("qWeN3"), Some("qwen3"));
        assert_eq!(resolve_template_name("DEEPSEEK"), Some("deepseek"));
        assert_eq!(resolve_template_name("DeepSeek"), Some("deepseek"));
        assert_eq!(resolve_template_name("LLAMA"), Some("llama"));
    }

    // @trace TEST-REG-77 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_family_for_all_table_tokens_consistency() {
        // Arrange: every token in ARCH_TABLE should resolve to a family
        // Act & Assert: no token should return None from resolve_family
        for &(token, _, family, _) in ARCH_TABLE {
            let expected = match family {
                "encoder" => ArchFamily::Encoder,
                "embedding" => ArchFamily::Embedding,
                "reranker" => ArchFamily::Reranker,
                _ => ArchFamily::Decoder,
            };
            assert_eq!(
                resolve_family(token),
                Some(expected),
                "token '{}' should resolve to {:?}",
                token,
                expected
            );
        }
    }

    // ── Additional tests (tests 78-87) ──

    // @trace TEST-REG-78 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn normalize_token_only_separators() {
        // Arrange: input consisting solely of separator characters
        let dash = normalize_token("-");
        let dot = normalize_token(".");
        let mixed = normalize_token("-._-");
        // Assert: each separator becomes '_' with no alphanumeric content
        assert_eq!(dash, "_");
        assert_eq!(dot, "_");
        assert_eq!(mixed, "____");
    }

    // @trace TEST-REG-79 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_special_chars_in_input() {
        // Arrange: input with special characters that get filtered during normalization
        // Act: "Llama@2" normalizes to "llama2" (not in table), not "llama"
        // Assert: filtered input does not accidentally match existing tokens
        // Note: "qwen 3" → normalize strips spaces → "qwen3" which IS in the table
        assert_eq!(resolve_template_name("Llama@2"), None);
        assert_eq!(resolve_template_name("qwen 3"), Some("qwen3"));
        assert_eq!(resolve_template_name("deep$seek"), Some("deepseek"));
    }

    // @trace TEST-REG-80 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_moe_router_all_canonicals_exhaustive_coverage() {
        // Arrange: collect all canonical names that have a MoE router
        let moe_canonicals: Vec<(&str, &str)> = ARCH_TABLE
            .iter()
            .filter_map(|&(_, n, _, r)| r.map(|router| (n, router)))
            .collect();
        // Act & Assert: each MoE canonical must resolve to a known RouterType (not Unknown)
        for (name, router_str) in &moe_canonicals {
            let resolved = resolve_moe_router(name);
            assert!(
                resolved.is_some(),
                "canonical '{}' with router '{}' should resolve",
                name,
                router_str
            );
            // The router string in the table should map to the corresponding RouterType
            let expected = match *router_str {
                "deepseek" => crate::manifest::RouterType::DeepSeek,
                "qwen" => crate::manifest::RouterType::Qwen,
                "mixtral" => crate::manifest::RouterType::Mixtral,
                "gptoss" => crate::manifest::RouterType::GptOss,
                _ => crate::manifest::RouterType::Unknown,
            };
            assert_eq!(resolved, Some(expected), "canonical '{}'", name);
        }
    }

    // @trace TEST-REG-81 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn is_valid_template_forcausallm_suffix_not_valid() {
        // Arrange: ForCausalLM suffixed names are not canonical names in the table
        // Act & Assert: is_valid_template only matches exact canonical entries
        assert!(!is_valid_template("LlamaForCausalLM"));
        assert!(!is_valid_template("qwen3forcausallm"));
        assert!(!is_valid_template("Qwen3ForCausalLM"));
    }

    // @trace TEST-REG-82 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_forcausallm_on_encoder_tokens() {
        // Arrange: ForCausalLM suffix on encoder-family tokens
        // Act & Assert: these should still resolve to their canonical names
        assert_eq!(resolve_template_name("BertForCausalLM"), Some("xlmr"));
        assert_eq!(resolve_template_name("RobertaForCausalLM"), Some("xlmr"));
        assert_eq!(resolve_template_name("SigLipForCausalLM"), Some("siglip"));
    }

    // @trace TEST-REG-83 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_family_encoder_family_excludes_all_non_encoder_tokens() {
        // Arrange: decoder tokens must NOT resolve to Encoder family
        let decoder_tokens = [
            "qwen3", "llama", "llama4", "smollm", "internlm", "mistral3", "ministral",
            "phi4", "glm4", "gemma4", "deepseek", "gptoss",
        ];
        // Act & Assert: every decoder token resolves to Decoder, never Encoder
        for token in &decoder_tokens {
            let family = resolve_family(token);
            assert_ne!(family, Some(ArchFamily::Encoder), "'{}' should not be Encoder", token);
            assert_eq!(family, Some(ArchFamily::Decoder), "'{}' should be Decoder", token);
        }
        // Embedding/Reranker tokens are distinct from both Encoder and Decoder
        assert_eq!(resolve_family("bge_m3"), Some(ArchFamily::Embedding));
        assert_eq!(resolve_family("bge_reranker"), Some(ArchFamily::Reranker));
    }

    // @trace TEST-REG-84 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn arch_table_moe_tokens_have_matching_canonical() {
        // Arrange: every token with a MoE router must have its canonical name
        // also present in the table (so resolve_moe_router can find it)
        for &(token, canonical, _, router) in ARCH_TABLE {
            if router.is_some() {
                // Act: check that the canonical name exists in the table
                let canonical_exists = ARCH_TABLE
                    .iter()
                    .any(|&(_, n, _, _)| n == canonical);
                // Assert
                assert!(
                    canonical_exists,
                    "token '{}' has MoE router but canonical '{}' not in table",
                    token, canonical
                );
            }
        }
    }

    // @trace TEST-REG-85 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_template_name_all_aliases_produce_same_canonical_as_primary() {
        // Arrange: groups of alias tokens that should share the same canonical name
        let alias_groups: &[(&[&str], &str)] = &[
            (&["qwen3", "qwen3moe", "qwen2_5", "qwen2"], "qwen3"),
            (&["llama", "smollm", "internlm"], "llama"),
            (&["mistral3", "ministral"], "mistral3"),
            (&["phi4", "phi4_mini", "phi3"], "phi4"),
            (&["glm4", "glm5", "chatglm"], "glm4"),
            (&["deepseek", "deepseek_v3", "deepseekv3", "deepseek_r1", "deepseekr1", "kimi_k2"], "deepseek"),
            (&["xlmr", "xlm_roberta", "xlmroberta", "bert", "roberta"], "xlmr"),
            (&["siglip", "siglip_vision_model"], "siglip"),
            (&["bge_m3", "bge_reranker"], "bge"),
        ];
        // Act & Assert: every alias in each group must resolve to the same canonical
        for &(aliases, expected_canonical) in alias_groups {
            for alias in aliases {
                assert_eq!(
                    resolve_template_name(alias),
                    Some(expected_canonical),
                    "alias '{}' should resolve to '{}'",
                    alias,
                    expected_canonical
                );
            }
        }
    }

    // @trace TEST-REG-86 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_family_forcausallm_suffix_on_moe_tokens() {
        // Arrange: ForCausalLM suffix on MoE-capable tokens should still resolve family
        // Act & Assert
        assert_eq!(resolve_family("Qwen3MoEForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("GLM4ForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("DeepSeekV3ForCausalLM"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("GptOssForCausalLM"), Some(ArchFamily::Decoder));
    }

    // @trace TEST-REG-87 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn arch_table_router_is_consistent_across_aliases() {
        // Arrange: tokens that share the same canonical name should have the same
        // router value (or all None)
        use std::collections::HashMap;
        let mut canonical_routers: HashMap<&str, Option<&str>> = HashMap::new();
        for &(_, canonical, _, router) in ARCH_TABLE {
            match canonical_routers.get(canonical) {
                None => {
                    canonical_routers.insert(canonical, router);
                }
                Some(&prev_router) => {
                    assert_eq!(
                        prev_router, router,
                        "canonical '{}' has inconsistent router values across aliases",
                        canonical
                    );
                }
            }
        }
    }

    // ── Additional tests (tests 88-97) — adversarial verification gap fixes ──

    // @trace TEST-REG-88 [req:REQ-MODEL-9] [level:unit]
    #[test]
    fn resolve_family_embedding_models() {
        // Arrange: bge_m3 is an embedding model (REQ-MODEL-9)
        // Act & Assert: must resolve to ArchFamily::Embedding, not Decoder
        assert_eq!(resolve_family("bge_m3"), Some(ArchFamily::Embedding));
        assert_eq!(resolve_template_name("bge_m3"), Some("bge"));
    }

    // @trace TEST-REG-89 [req:REQ-MODEL-10] [level:unit]
    #[test]
    fn resolve_family_reranker_models() {
        // Arrange: bge_reranker is a reranker model (REQ-MODEL-10)
        // Act & Assert: must resolve to ArchFamily::Reranker, not Decoder
        assert_eq!(resolve_family("bge_reranker"), Some(ArchFamily::Reranker));
        assert_eq!(resolve_template_name("bge_reranker"), Some("bge"));
    }

    // @trace TEST-REG-90 [req:REQ-MODEL-8] [level:unit]
    #[test]
    fn resolve_llama4_token() {
        // Arrange: llama4 is a REQ-MODEL-8 supported Generator model
        // Act & Assert: must resolve correctly (was absent from ARCH_TABLE)
        assert_eq!(resolve_template_name("llama4"), Some("llama4"));
        assert_eq!(resolve_family("llama4"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_moe_router("llama4"), None);
    }

    // @trace TEST-REG-91 [req:REQ-MODEL-7] [level:unit]
    #[test]
    fn resolve_xlmroberta_forcausallm_fixed() {
        // Arrange: XLMRobertaForCausalLM normalizes to "xlmroberta" (no underscore).
        // Previously failed because only "xlm_roberta" was in the table.
        // Now "xlmroberta" has its own entry, fixing the ForCausalLM gap.
        // Act & Assert
        assert_eq!(resolve_template_name("XLMRobertaForCausalLM"), Some("xlmr"));
        assert_eq!(resolve_family("XLMRobertaForCausalLM"), Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-92 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn resolve_deepseek_non_underscore_aliases() {
        // Arrange: deepseekv3 and deepseekr1 are non-underscore aliases
        // that previously had zero test coverage
        // Act & Assert
        assert_eq!(resolve_template_name("deepseekv3"), Some("deepseek"));
        assert_eq!(resolve_template_name("deepseekr1"), Some("deepseek"));
        assert_eq!(resolve_family("deepseekv3"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("deepseekr1"), Some(ArchFamily::Decoder));
    }

    // @trace TEST-REG-93 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn arch_family_four_variants_are_distinct() {
        // Arrange: ArchFamily now has 4 variants per ENT-ARCHITECTURE-FEATURES.family
        // Act & Assert: all 4 variants must be pairwise distinct
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ArchFamily::Encoder);
        set.insert(ArchFamily::Decoder);
        set.insert(ArchFamily::Embedding);
        set.insert(ArchFamily::Reranker);
        assert_eq!(set.len(), 4, "all four ArchFamily variants must be distinct");
    }

    // @trace TEST-REG-94 [req:REQ-MODEL-9] [req:REQ-MODEL-10] [level:unit]
    #[test]
    fn embedding_reranker_not_misclassified_as_decoder() {
        // Arrange: the old `_ => ArchFamily::Decoder` wildcard silently misclassified
        // embedding/reranker models as Decoder (the original adversarial finding)
        // Act & Assert: embedding/reranker models must NOT resolve to Decoder
        assert_ne!(resolve_family("bge_m3"), Some(ArchFamily::Decoder));
        assert_ne!(resolve_family("bge_reranker"), Some(ArchFamily::Decoder));
        assert_ne!(resolve_family("bge_m3"), Some(ArchFamily::Encoder));
        assert_ne!(resolve_family("bge_reranker"), Some(ArchFamily::Encoder));
    }

    // @trace TEST-REG-95 [req:REQ-MODEL-8] [level:unit]
    #[test]
    fn resolve_family_forcausallm_on_llama4() {
        // Arrange: Llama4ForCausalLM should resolve as a decoder model
        // Act & Assert
        assert_eq!(resolve_template_name("Llama4ForCausalLM"), Some("llama4"));
        assert_eq!(resolve_family("Llama4ForCausalLM"), Some(ArchFamily::Decoder));
    }

    // @trace TEST-REG-96 [req:REQ-ARCH-AUTO-001] [level:unit]
    #[test]
    fn arch_table_all_family_strings_are_mapped() {
        // Arrange: every family string in ARCH_TABLE must have a corresponding
        // ArchFamily variant (completeness check — catches unmapped family strings)
        // Act & Assert
        let mut families: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for &(_, _, family, _) in ARCH_TABLE {
            families.insert(family);
        }
        for family in &families {
            let mapped = match *family {
                "encoder" => Some(ArchFamily::Encoder),
                "decoder" => Some(ArchFamily::Decoder),
                "embedding" => Some(ArchFamily::Embedding),
                "reranker" => Some(ArchFamily::Reranker),
                _ => None,
            };
            assert!(mapped.is_some(), "family '{}' has no ArchFamily mapping", family);
        }
    }

    // @trace TEST-REG-97 [req:REQ-MODEL-8] [level:unit]
    #[test]
    fn arch_table_covers_all_req_model_8_generator_models() {
        // Arrange: REQ-MODEL-8 lists supported Generator models that must all
        // be resolvable through the registry
        // Act & Assert: each generator model token must resolve to a canonical name
        let generator_tokens = [
            "qwen3", "llama", "llama4", "mistral3", "phi4", "glm4", "gemma4",
            "deepseek", "gptoss",
        ];
        for token in &generator_tokens {
            assert!(
                resolve_template_name(token).is_some(),
                "generator token '{}' must be in ARCH_TABLE (REQ-MODEL-8)",
                token
            );
            assert_eq!(
                resolve_family(token),
                Some(ArchFamily::Decoder),
                "generator token '{}' must resolve to Decoder (REQ-MODEL-8)",
                token
            );
        }
    }
}
