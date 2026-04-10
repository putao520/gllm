//! 架构模板注册表 (REQ-ARCH-002)
//!
//! 从 YAML 模板内容自动注册。每个模板通过 `aliases` 字段声明自己匹配
//! 哪些 HuggingFace architecture token，通过 `family` 字段声明编码器/解码器族。
//!
//! 新增架构 = 新增 YAML 文件 + 在 `ALL_TEMPLATES` 数组添加一行 `include_str!`。
//! 无需修改枚举、match 语句或手工映射。

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::manifest::ArchFamily;

use super::template::ArchTemplate;

/// 全局模板注册表
static REGISTRY: OnceLock<ArchRegistry> = OnceLock::new();

/// 架构注册表
///
/// 两种查询路径：
/// - `get(name)`: 按模板名精确查找 (e.g., "qwen3")
/// - `resolve_token(token)`: 按架构 token 模糊查找 (e.g., "LlamaForCausalLM")
#[derive(Debug, Default)]
pub struct ArchRegistry {
    /// 模板名 → 模板
    templates: HashMap<String, ArchTemplate>,
    /// 归一化别名 → 模板名
    alias_map: HashMap<String, String>,
}

impl ArchRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册模板：按 name 存储，按 aliases 建索引
    pub fn register(&mut self, template: ArchTemplate) {
        // 模板名本身也是别名
        self.alias_map
            .insert(template.name.clone(), template.name.clone());
        // 注册所有声明的别名
        for alias in &template.aliases {
            self.alias_map
                .insert(alias.clone(), template.name.clone());
        }
        self.templates.insert(template.name.clone(), template);
    }

    /// 按模板名精确查找
    pub fn get(&self, name: &str) -> Option<&ArchTemplate> {
        self.templates.get(name)
    }

    /// 按架构 token 查找模板 (自动归一化)
    ///
    /// 支持 HuggingFace config.json 的 `architectures` 字段值
    /// (e.g., "LlamaForCausalLM") 和 GGUF metadata 的架构字符串。
    pub fn resolve_token(&self, token: &str) -> Option<&ArchTemplate> {
        let normalized = normalize_token(token);
        let template_name = self.alias_map.get(&normalized)?;
        self.templates.get(template_name)
    }

    /// 按架构 token 查找模板名
    pub fn resolve_token_to_name(&self, token: &str) -> Option<&str> {
        let normalized = normalize_token(token);
        self.alias_map.get(&normalized).map(|s| s.as_str())
    }

    /// 按架构 token 查找 ArchFamily
    pub fn resolve_family(&self, token: &str) -> Option<ArchFamily> {
        let template = self.resolve_token(token)?;
        Some(parse_family(&template.family))
    }

    /// 按模板名查找 MoE 路由器类型
    pub fn resolve_moe_router(&self, template_name: &str) -> Option<crate::manifest::RouterType> {
        let template = self.get(template_name)?;
        let router_str = template.moe_router.as_deref()?;
        Some(match router_str {
            "deepseek" => crate::manifest::RouterType::DeepSeek,
            "qwen" => crate::manifest::RouterType::Qwen,
            "mixtral" => crate::manifest::RouterType::Mixtral,
            _ => crate::manifest::RouterType::Unknown,
        })
    }

    /// 校验模板名是否已注册
    pub fn is_valid(&self, name: &str) -> bool {
        self.templates.contains_key(name)
    }

    /// 列出所有模板名
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }
}

/// 归一化 architecture token: 小写 + 去除 - . 转 _
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

/// 解析 family 字符串
fn parse_family(family: &str) -> ArchFamily {
    match family.to_ascii_lowercase().as_str() {
        "encoder" => ArchFamily::Encoder,
        _ => ArchFamily::Decoder,
    }
}

// ============================================================================
// 公共 API（全局注册表操作）
// ============================================================================

/// 按模板名获取模板
pub fn get_template(name: &str) -> Option<&'static ArchTemplate> {
    REGISTRY.get().and_then(|r| r.get(name))
}

/// 按架构 token 获取模板 (自动归一化)
pub fn resolve_template(token: &str) -> Option<&'static ArchTemplate> {
    REGISTRY.get().and_then(|r| r.resolve_token(token))
}

/// 按架构 token 获取模板名
pub fn resolve_template_name(token: &str) -> Option<&'static str> {
    REGISTRY.get().and_then(|r| r.resolve_token_to_name(token))
}

/// 按架构 token 获取 ArchFamily
pub fn resolve_family(token: &str) -> Option<ArchFamily> {
    REGISTRY.get().and_then(|r| r.resolve_family(token))
}

/// 按模板名获取 MoE 路由器类型
pub fn resolve_moe_router(template_name: &str) -> Option<crate::manifest::RouterType> {
    REGISTRY.get().and_then(|r| r.resolve_moe_router(template_name))
}

/// 校验模板名是否已注册
pub fn is_valid_template(name: &str) -> bool {
    REGISTRY.get().is_some_and(|r| r.is_valid(name))
}

/// 初始化全局注册表（幂等，多次调用安全）
pub fn register_builtin_templates() {
    let _ = REGISTRY.get_or_init(|| {
        let mut registry = ArchRegistry::new();
        for yaml_str in SCANNED_TEMPLATES {
            if let Ok(template) = ArchTemplate::from_yaml(yaml_str) {
                registry.register(template);
            }
        }
        registry
    });
}

// ============================================================================
// 模板列表 — build.rs 扫描 src/arch/templates/*.yaml 自动生成
// ============================================================================

// build.rs 生成的 SCANNED_TEMPLATES 常量
include!(concat!(env!("OUT_DIR"), "/template_list.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_templates_parse() {
        register_builtin_templates();
        let registry = REGISTRY.get().unwrap();
        assert!(
            registry.templates.len() >= 9,
            "expected at least 9 templates, got {}",
            registry.templates.len()
        );
    }

    #[test]
    fn resolve_token_works() {
        register_builtin_templates();
        // HuggingFace config.json 格式
        assert!(resolve_template("LlamaForCausalLM").is_some());
        assert_eq!(resolve_template("LlamaForCausalLM").unwrap().name, "llama");
        assert!(resolve_template("MistralForCausalLM").is_some());
        assert_eq!(resolve_template("MistralForCausalLM").unwrap().name, "mistral3");
        assert!(resolve_template("Gemma2ForCausalLM").is_some());
        assert_eq!(resolve_template("Gemma2ForCausalLM").unwrap().name, "gemma2");
        assert!(resolve_template("GPTOSSForCausalLM").is_some());
        assert_eq!(resolve_template("GPTOSSForCausalLM").unwrap().name, "gpt2next");
        // 不存在的
        assert!(resolve_template("custom-llama-adapter").is_none());
    }

    #[test]
    fn resolve_family_works() {
        register_builtin_templates();
        assert_eq!(resolve_family("llama"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("xlmr"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("bert"), Some(ArchFamily::Encoder));
        assert_eq!(resolve_family("qwen3"), Some(ArchFamily::Decoder));
        assert_eq!(resolve_family("gptoss"), Some(ArchFamily::Decoder));
    }

    #[test]
    fn template_lookup_by_name() {
        register_builtin_templates();
        for name in ["qwen3", "llama", "mistral3", "glm4", "phi4", "gemma2", "xlmr", "deepseek", "gpt2next"] {
            assert!(get_template(name).is_some(), "template '{name}' not found");
        }
        assert!(get_template("nonexistent").is_none());
    }

    #[test]
    fn aliases_cover_all_known_tokens() {
        register_builtin_templates();
        // 验证所有已知的 HuggingFace architecture token 都能解析
        let known_tokens = [
            ("LlamaForCausalLM", "llama"),
            ("Qwen3ForCausalLM", "qwen3"),
            ("Qwen2ForCausalLM", "qwen3"),
            ("MistralForCausalLM", "mistral3"),
            ("MinistralForCausalLM", "mistral3"),
            ("Phi4ForCausalLM", "phi4"),
            ("Phi3ForCausalLM", "phi4"),
            ("Gemma2ForCausalLM", "gemma2"),
            ("DeepseekV3ForCausalLM", "deepseek"),
            ("SmolLM2ForCausalLM", "llama"),
            ("GPTOSSForCausalLM", "gpt2next"),
            ("GPT2NextForCausalLM", "gpt2next"),
            ("bert", "xlmr"),
            ("roberta", "xlmr"),
        ];
        for (token, expected_template) in known_tokens {
            let resolved = resolve_template_name(token);
            assert_eq!(
                resolved,
                Some(expected_template),
                "token '{token}' should resolve to '{expected_template}', got {resolved:?}"
            );
        }
    }

}
