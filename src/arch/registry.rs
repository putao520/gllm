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
            "gptoss" => crate::manifest::RouterType::GptOss,
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

    /// T45: USM Conformer YAML 解析 + to_onnx_graph 展开。
    /// 给定 mock config (num_layers=2), 验证节点总数与 DepthwiseConv1D 出现次数。
    /// 不硬编码模板名 (若未来重命名, SCANNED_TEMPLATES 扫描自动跟进),
    /// 直接基于注册表反查确认模板存在。
    #[test]
    fn builtin_usm_conformer_parses_and_expands() {
        use super::super::resolve::ResolvedConfig;

        register_builtin_templates();
        let template = get_template("usm_conformer")
            .expect("usm_conformer.yaml 必须被 build.rs 扫描并注册");
        assert_eq!(template.name, "usm_conformer");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.tensor_patterns.layer_prefix.is_some());

        // Mock Conformer 配置: 2 层 Conformer, channels=512, heads=8, inter=2048
        let config = ResolvedConfig {
            num_hidden_layers: 2,
            hidden_size: 512,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            head_dim: 64,
            intermediate_size: Some(2048),
            vocab_size: 1, // 音频编码器不涉及 vocab, 占位
            rope_theta: 10000.0,
            dtype: "f32".to_string(),
            ..Default::default()
        };

        let graph = template
            .to_onnx_graph(&config)
            .expect("to_onnx_graph 必须成功");

        // 每个 Conformer block 展开为:
        //   FF1: norm + matmul + silu + matmul + add  (5)
        //   MHA: norm + q + k + v + mha + o + add    (7)
        //   Conv: norm + pw1 + glu + dw + bn + act + pw2 + add (8)
        //   FF2: norm + matmul + silu + matmul + add  (5)
        //   Block-end: norm                           (1)
        // 每层 26 个节点; 2 层 → 52, 加顶层 final_norm → 53
        let expected_per_layer = 26;
        let expected_total = expected_per_layer * config.num_hidden_layers + 1;
        assert_eq!(
            graph.nodes.len(),
            expected_total,
            "USM Conformer 展开节点数应为 {expected_total} (每层 {expected_per_layer} + 1 final_norm)",
        );

        // 每层必须包含 1 个 DepthwiseConv1D
        let dw_nodes: Vec<_> = graph
            .nodes
            .iter()
            .filter(|n| n.op_type == "DepthwiseConv1D")
            .collect();
        assert_eq!(
            dw_nodes.len(),
            config.num_hidden_layers,
            "每层 Conformer 必须包含 1 个 DepthwiseConv1D 节点, 共 {} 层",
            config.num_hidden_layers,
        );

        // 验证 DepthwiseConv1D 节点命名与权重引用 (第 0 层)
        let dw0 = &dw_nodes[0];
        assert_eq!(dw0.name, "layer_0_conv_dw");
        assert_eq!(dw0.inputs.len(), 2, "DepthwiseConv1D 需要 2 个输入 (x, weight)");
        assert_eq!(
            dw0.inputs[1],
            "audio_tower.encoder.layers.0.conv_module.depthwise_conv.weight",
        );
    }

    /// T45: 注册表按模板名 "usm_conformer" 反查成功。
    #[test]
    fn builtin_usm_conformer_registered() {
        register_builtin_templates();
        assert!(
            get_template("usm_conformer").is_some(),
            "USM Conformer 必须注册到全局模板表",
        );
    }

    /// T44: SigLIP ViT YAML 解析 + to_onnx_graph 展开。
    /// 给定 mock config (num_layers=2, patch=14, image=28 → num_patches=4),
    /// 验证节点总数 / PatchEmbed / LearnedPos2D 节点存在 + 命名 + 权重引用。
    #[test]
    fn builtin_siglip_parses_and_expands() {
        use super::super::resolve::ResolvedConfig;
        use std::collections::HashMap;

        register_builtin_templates();
        let template = get_template("siglip")
            .expect("siglip.yaml 必须被 build.rs 扫描并注册");
        assert_eq!(template.name, "siglip");
        assert!(template.config.contains_key("num_layers"));
        assert!(template.config.contains_key("patch_size"));
        assert!(template.config.contains_key("image_size"));
        assert!(template.tensor_patterns.layer_prefix.is_some());

        // Mock SigLIP 配置: 2 层 ViT, embed=128, heads=4, patch=14, image=28,
        // in_channels=3 → num_patches = (28/14)^2 = 4
        let mut extra: HashMap<String, i64> = HashMap::new();
        extra.insert("patch_size".into(), 14);
        extra.insert("image_size".into(), 28);
        extra.insert("num_patches".into(), 4);
        extra.insert("in_channels".into(), 3);
        let config = ResolvedConfig {
            num_hidden_layers: 2,
            hidden_size: 128,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 32,
            intermediate_size: Some(512),
            vocab_size: 1, // 视觉 encoder 不涉及 vocab
            rope_theta: 10000.0,
            dtype: "f32".to_string(),
            extra,
            ..Default::default()
        };

        let graph = template
            .to_onnx_graph(&config)
            .expect("to_onnx_graph 必须成功");

        // 每层 ViT block 展开:
        //   attn_norm + q + k + v + attn + o + attn_residual = 7
        //   ffn_norm + fc1 + gelu + fc2 + ffn_residual       = 5
        // 每层 12 个节点, 2 层 → 24,
        // 顶层: patch_embed + pos_embed + final_norm        = 3
        // 总计 24 + 3 = 27
        let expected_per_layer = 12;
        let expected_top = 3;
        let expected_total = expected_per_layer * config.num_hidden_layers + expected_top;
        assert_eq!(
            graph.nodes.len(),
            expected_total,
            "SigLIP 展开节点数应为 {expected_total} \
             (每层 {expected_per_layer} + 顶层 {expected_top})",
        );

        // PatchEmbed 节点存在 + 命名 + 输入/权重引用
        let patch_nodes: Vec<_> = graph.nodes.iter()
            .filter(|n| n.op_type == "PatchEmbed")
            .collect();
        assert_eq!(patch_nodes.len(), 1, "SigLIP 必须有且仅有 1 个 PatchEmbed 节点");
        let patch = patch_nodes[0];
        assert_eq!(patch.name, "patch_embed");
        assert_eq!(patch.inputs.len(), 2, "PatchEmbed 需要 2 个输入 (image, kernel)");
        assert_eq!(patch.inputs[1], "vision_tower.patch_embed.proj.weight");

        // LearnedPos2D 节点存在 + 命名 + 权重引用
        let pos_nodes: Vec<_> = graph.nodes.iter()
            .filter(|n| n.op_type == "LearnedPos2D")
            .collect();
        assert_eq!(pos_nodes.len(), 1, "SigLIP 必须有且仅有 1 个 LearnedPos2D 节点");
        let pos = pos_nodes[0];
        assert_eq!(pos.name, "pos_embed");
        assert_eq!(pos.inputs.len(), 2, "LearnedPos2D 需要 2 个输入 (patches, pos_table)");
        assert_eq!(
            pos.inputs[1],
            "vision_tower.embeddings.position_embedding.weight",
        );
    }

    /// T44: SigLIP 按模板名 / extra_aliases 多种 token 反查都能命中。
    #[test]
    fn builtin_siglip_registered_and_resolvable() {
        register_builtin_templates();
        assert!(get_template("siglip").is_some(),
            "SigLIP 必须注册到全局模板表");
        // 自动派生别名 + extra_aliases 统一校验
        for token in ["siglip", "SiglipForCausalLM", "SiglipVisionModel",
                      "siglip_vision_model", "SiglipModel"] {
            assert!(
                resolve_template(token).is_some(),
                "token '{token}' 应解析到 SigLIP 模板",
            );
        }
    }
}
