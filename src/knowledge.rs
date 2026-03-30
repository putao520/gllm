//! Knowledge Injection API (per SPEC 04-API-DESIGN §7-§8)
//!
//! 提供知识注入、语义锚点定位和多态数据源抽象。
//! 当前为骨架实现，底层注入机制依赖 SPEC §9-§16 的 Mega-Kernel 架构。

/// 语义锚点 — 指定知识注入的物理层目标。
///
/// 对应模型 Transformer 层的不同语义深度：
/// - `Embedding`: 浅层嵌入区（token 表征注入）
/// - `MidSemantic`: 中层语义区（概念级知识注入）
/// - `DeepLogic`: 深层逻辑区（推理级知识注入）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTarget {
    Embedding,
    MidSemantic,
    DeepLogic,
}

/// 知识注入方式标识，供编译器分流处理。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionKind {
    /// 侧载 KV：业务端传入预存的 KV cache 片段
    FrozenKvChunk,
    /// 晚期插入：上游小模型算好的密实特征向量列
    LateFusionVector,
    /// 领域特征挂载：带有特定领域特征缩放因子的极小权重片
    DynamicLoRA,
}

/// 知识数据源的多态抽象 (per SPEC 04-API-DESIGN §8.1)。
///
/// 开发者通过实现此 trait 定义自己的知识数据来源。
/// 引擎根据 `injection_kind()` 返回值选择对应的注入路径。
pub trait KnowledgeDataSource: Send + Sync {
    /// 返回注入类型标识，供编译器分流。
    fn injection_kind(&self) -> InjectionKind;

    /// 返回数据源的描述信息（用于日志和调试）。
    fn describe(&self) -> String;
}

/// 内置知识数据源：从冻结的 KV cache 文件加载。
#[derive(Debug, Clone)]
pub struct FrozenKvSource {
    pub path: String,
}

impl FrozenKvSource {
    pub fn from_path(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl KnowledgeDataSource for FrozenKvSource {
    fn injection_kind(&self) -> InjectionKind {
        InjectionKind::FrozenKvChunk
    }

    fn describe(&self) -> String {
        format!("FrozenKvSource({})", self.path)
    }
}

/// 内置知识数据源：从文本向量注入。
#[derive(Debug, Clone)]
pub struct VectorSource {
    pub vectors: Vec<Vec<f32>>,
    pub dimension: usize,
}

impl VectorSource {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        let dimension = vectors.first().map(|v| v.len()).unwrap_or(0);
        Self { vectors, dimension }
    }
}

impl KnowledgeDataSource for VectorSource {
    fn injection_kind(&self) -> InjectionKind {
        InjectionKind::LateFusionVector
    }

    fn describe(&self) -> String {
        format!("VectorSource(count={}, dim={})", self.vectors.len(), self.dimension)
    }
}

/// 知识注入配置 (per SPEC 04-API-DESIGN §7.2)。
pub struct KnowledgeInjectionConfig {
    /// 注入目标层
    pub target: LayerTarget,
    /// 数据源
    pub source: Box<dyn KnowledgeDataSource>,
}

impl KnowledgeInjectionConfig {
    pub fn new(target: LayerTarget, source: impl KnowledgeDataSource + 'static) -> Self {
        Self {
            target,
            source: Box::new(source),
        }
    }
}

/// 知识注入结果
#[derive(Debug, Clone)]
pub struct KnowledgeInjectionResult {
    /// 注入的目标层
    pub target: LayerTarget,
    /// 注入的 token 数量（若适用）
    pub tokens_injected: usize,
    /// 注入状态描述
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_target_values() {
        assert_ne!(LayerTarget::Embedding, LayerTarget::MidSemantic);
        assert_ne!(LayerTarget::MidSemantic, LayerTarget::DeepLogic);
    }

    #[test]
    fn test_frozen_kv_source() {
        let source = FrozenKvSource::from_path("test.kv");
        assert_eq!(source.injection_kind(), InjectionKind::FrozenKvChunk);
        assert!(source.describe().contains("test.kv"));
    }

    #[test]
    fn test_vector_source() {
        let source = VectorSource::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(source.injection_kind(), InjectionKind::LateFusionVector);
        assert_eq!(source.dimension, 3);
        assert_eq!(source.vectors.len(), 2);
    }

    #[test]
    fn test_injection_config() {
        let config = KnowledgeInjectionConfig::new(
            LayerTarget::MidSemantic,
            FrozenKvSource::from_path("logs.kv"),
        );
        assert_eq!(config.target, LayerTarget::MidSemantic);
    }
}
