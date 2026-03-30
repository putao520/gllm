//! Knowledge Injection API (per SPEC 04-API-DESIGN §7, §8)

use std::collections::HashMap;
use std::path::PathBuf;

/// 语义锚点 (per SPEC 04-API-DESIGN §7.1)
///
/// 摒弃死板的层号（如 `layer=15`），由引擎动态测算映射层深。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTarget {
    /// 浅层词法区
    ShallowSyntax,
    /// 中层语义区
    MidSemantic,
    /// 深层逻辑区（爆词前夕）
    DeepLogic,
}

impl LayerTarget {
    /// 返回归一化深度 (0.0 到 1.0)
    pub fn normalized_depth(&self) -> f32 {
        match self {
            LayerTarget::ShallowSyntax => 0.125,
            LayerTarget::MidSemantic => 0.5,
            LayerTarget::DeepLogic => 0.875,
        }
    }

    /// 映射到物理层号
    pub fn to_physical_layer(&self, total_layers: usize) -> usize {
        let depth = self.normalized_depth();
        (depth * total_layers as f32).floor() as usize
    }
}

/// 注入类型标识 (per SPEC 04-API-DESIGN §8.1, §8.6)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionKind {
    /// 侧载 KV：预存的 KV cache 片段
    FrozenKvChunk,
    /// 晚期插入：密实特征向量列
    LateFusionVector,
    /// 领域特征挂载：LoRA 权重片
    DynamicLoRA,
}

/// 物理化载荷 (per SPEC 04-API-DESIGN §8.1)
#[derive(Debug, Clone)]
pub struct MaterializedPayload {
    pub kind: InjectionKind,
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub metadata: HashMap<String, String>,
}

/// 知识注入错误
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("source not found: {0}")]
    SourceNotFound(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("invalid layer target")]
    InvalidLayerTarget,
    #[error("data format error: {0}")]
    DataFormatError(String),
}

/// 知识注入结果 (per SPEC 04-API-DESIGN §7.2)
#[derive(Debug, Clone)]
pub struct KnowledgeInjectionResult {
    /// 注入的实际物理层
    pub actual_layer: usize,
    /// 注入的数据大小（字节）
    pub data_size_bytes: usize,
}

/// 知识数据源的多态抽象 (per SPEC 04-API-DESIGN §8.1)
pub trait KnowledgeDataSource {
    /// 返回注入类型标识
    fn injection_kind(&self) -> InjectionKind;

    /// 将数据物理化至引擎可感知的格式
    fn materialize(&self) -> Result<MaterializedPayload, KnowledgeError>;
}

/// 知识注入配置 (per SPEC 04-API-DESIGN §7.2)
pub struct KnowledgeInjectionConfig {
    pub target: LayerTarget,
    pub source: Box<dyn KnowledgeDataSource>,
}

impl KnowledgeInjectionConfig {
    pub fn new(target: LayerTarget, source: Box<dyn KnowledgeDataSource>) -> Self {
        Self { target, source }
    }
}

/// 冻结 KV 数据源 (per SPEC 04-API-DESIGN §8.4)
///
/// 从预存的 KV cache 文件加载数据。
#[derive(Debug, Clone)]
pub struct FrozenKvSource {
    pub path: PathBuf,
}

impl FrozenKvSource {
    /// 从冻结 KV 文件创建数据源
    pub fn from_frozen_kv(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl KnowledgeDataSource for FrozenKvSource {
    fn injection_kind(&self) -> InjectionKind {
        InjectionKind::FrozenKvChunk
    }

    fn materialize(&self) -> Result<MaterializedPayload, KnowledgeError> {
        // Skeleton: requires actual KV file loading implementation
        Err(KnowledgeError::NotImplemented(
            "FrozenKvSource::materialize requires executor integration".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_target_discriminant() {
        assert_ne!(LayerTarget::ShallowSyntax as u8, LayerTarget::MidSemantic as u8);
        assert_ne!(LayerTarget::MidSemantic as u8, LayerTarget::DeepLogic as u8);
    }

    #[test]
    fn test_layer_target_normalized_depth() {
        assert_eq!(LayerTarget::ShallowSyntax.normalized_depth(), 0.125);
        assert_eq!(LayerTarget::MidSemantic.normalized_depth(), 0.5);
        assert_eq!(LayerTarget::DeepLogic.normalized_depth(), 0.875);
    }

    #[test]
    fn test_layer_target_to_physical_layer() {
        assert_eq!(LayerTarget::ShallowSyntax.to_physical_layer(32), 4);
        assert_eq!(LayerTarget::MidSemantic.to_physical_layer(32), 16);
        assert_eq!(LayerTarget::DeepLogic.to_physical_layer(32), 28);
    }
}
