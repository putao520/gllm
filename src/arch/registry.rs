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

/// 全局模板注册表 (pub 以支持上层测试按 YAML 驱动反查)。
pub static REGISTRY: OnceLock<ArchRegistry> = OnceLock::new();

/// 架构注册表
///
/// 两种查询路径：
/// - `get(name)`: 按模板名精确查找 (e.g., "qwen3")
/// - `resolve_token(token)`: 按架构 token 模糊查找 (e.g., "LlamaForCausalLM")
#[derive(Debug, Default)]
pub struct ArchRegistry {
    /// 模板名 → 模板 (pub 以支持上层测试按 YAML 驱动迭代)
    pub templates: HashMap<String, ArchTemplate>,
    /// 归一化别名 → 模板名
    alias_map: HashMap<String, String>,
}

impl ArchRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册模板：自动派生标准别名 + 注册例外别名
    ///
    /// 自动派生规则（从模板名 `name` 生成）：
    /// - `{name}` — 模板名本身
    /// - `{name}forcausallm` — HuggingFace `architectures` 字段标准格式
    ///
    /// `extra_aliases` 仅用于声明无法从模板名推导的例外映射
    /// （如 `chatglm` → `glm4`，`qwen2` → `qwen3`）。
    pub fn register(&mut self, template: ArchTemplate) {
        let name = &template.name;
        // 自动派生标准别名
        self.insert_with_suffix(name, name);
        // 对每个例外别名也派生标准后缀
        for alias in &template.extra_aliases {
            let normalized = normalize_token(alias);
            self.insert_with_suffix(&normalized, name);
        }
        self.templates.insert(name.clone(), template);
    }

    /// 插入别名及其 `{alias}ForCausalLM` 派生形式。
    /// register/lookup 对称:alias 一律先 `normalize_token` 再入表,保证含连字符
    /// 的模板名(如 `qwen3-reranker`)能通过 `resolve_token("qwen3-reranker")` 反查。
    fn insert_with_suffix(&mut self, alias: &str, template_name: &str) {
        let norm = normalize_token(alias);
        self.alias_map.insert(norm.clone(), template_name.to_string());
        self.alias_map.insert(format!("{norm}forcausallm"), template_name.to_string());
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

    /// YAML 模板目录扫描结果非空 — build.rs 填充 SCANNED_TEMPLATES。
    /// 无硬编码模板名列表:测试只校验"扫描路径有效、所有模板 YAML 均可解析"。
    #[test]
    fn all_templates_parse() {
        register_builtin_templates();
        let registry = REGISTRY.get().unwrap();
        assert!(
            !registry.templates.is_empty(),
            "build.rs 扫描 src/arch/templates/*.yaml 必须至少注册一个模板"
        );
        assert_eq!(
            registry.templates.len(),
            SCANNED_TEMPLATES.len(),
            "已扫描 YAML 数 ({}) 与注册模板数 ({}) 必须一致",
            SCANNED_TEMPLATES.len(),
            registry.templates.len()
        );
    }

    /// 每个 YAML 模板的 `name` 本身 + 自动派生的 `{name}ForCausalLM` 都必须可解析。
    /// 不硬编码 LlamaForCausalLM 等字符串 — 直接从已加载模板反查。
    #[test]
    fn yaml_name_and_standard_token_resolve() {
        register_builtin_templates();
        let registry = REGISTRY.get().unwrap();
        for name in registry.templates.keys() {
            assert_eq!(resolve_template_name(name), Some(name.as_str()),
                "template name '{name}' 必须解析为自身");
            let forcausallm = format!("{name}ForCausalLM");
            assert_eq!(resolve_template_name(&forcausallm), Some(name.as_str()),
                "自动派生 token '{forcausallm}' 必须解析为 '{name}'");
        }
    }

    /// 每个 YAML 模板声明的 `extra_aliases` 必须反向解析为该模板。
    /// SSOT:别名完全由 YAML 源驱动,测试仅校验"YAML 声明 ↔ 注册表注册"一致性。
    #[test]
    fn extra_aliases_are_registered() {
        register_builtin_templates();
        let registry = REGISTRY.get().unwrap();
        for (name, template) in &registry.templates {
            for alias in &template.extra_aliases {
                assert_eq!(resolve_template_name(alias), Some(name.as_str()),
                    "extra_alias '{alias}' (YAML {}) 未能反查到模板 '{name}'", name);
                let alias_forcausallm = format!("{alias}ForCausalLM");
                assert_eq!(resolve_template_name(&alias_forcausallm), Some(name.as_str()),
                    "派生 token '{alias_forcausallm}' 未能反查到 '{name}'");
            }
        }
    }

    /// 每个模板的 YAML `family` 字段能正确解析为 ArchFamily。
    /// 不硬编码 llama/xlmr 等名称 — 仅校验每个已注册模板的 family 都有合法解析。
    #[test]
    fn every_template_family_resolves() {
        register_builtin_templates();
        let registry = REGISTRY.get().unwrap();
        for name in registry.templates.keys() {
            assert!(resolve_family(name).is_some(),
                "模板 '{name}' 的 family 字段必须解析为 ArchFamily");
        }
    }

    /// 未知 token 必须返回 None — 唯一允许的硬编码字符串是"故意不存在的 token"。
    #[test]
    fn unknown_token_returns_none() {
        register_builtin_templates();
        assert!(resolve_template("custom-llama-adapter-that-does-not-exist").is_none());
        assert!(get_template("nonexistent-template-name").is_none());
    }
}
