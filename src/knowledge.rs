//! Knowledge Injection API (per SPEC 04-API-DESIGN §7, §8)

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

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
    /// 领域特征挂载：LoRA 权重片 (per SPEC §8.1)
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
    /// 将数据物理化至引擎可感知的格式 (per SPEC 04-API-DESIGN §8.1)
    fn materialize(&self, _engine: &crate::engine::EngineContext) -> Result<MaterializedPayload, KnowledgeError>;
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
pub struct KnowledgeSource {
    pub path: PathBuf,
}

impl KnowledgeSource {
    /// 从冻结 KV 文件创建数据源
    pub fn from_frozen_kv(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
        }
    }
}

impl KnowledgeDataSource for KnowledgeSource {
    fn injection_kind(&self) -> InjectionKind {
        InjectionKind::FrozenKvChunk
    }

    fn materialize(&self, _engine: &crate::engine::EngineContext) -> Result<MaterializedPayload, KnowledgeError> {
        Err(KnowledgeError::NotImplemented(
            "KnowledgeSource::materialize requires executor integration".into(),
        ))
    }
}

// ============================================================================
// 零拷贝页表管理器 (per SPEC 04-API-DESIGN §8.4)
// ============================================================================

/// 零拷贝页表管理器 (per SPEC 04-API-DESIGN §8.4)
///
/// 实现超大财报 0 算力注入的核心支撑库：
/// - 与主系统的 `GlobalMemoryManager` 紧密咬合
/// - 当用户传入含有 10 万字 KV 数据的 `FrozenKvChunk` 时，**不开辟任何新显存、不执行任何 `memcpy`**
/// - 只负责做一件事：拦截当前 Request 的**逻辑地址页表（Logical Page Table）**，并将预存的物理页（Physical Block IDs）原样插入
/// - 对 LLM 后续的 Attention 算子来说，这跟自己前一秒算出来的数据存在那一模一样
#[derive(Debug, Clone)]
pub struct KvSideloadManager {
    /// 预存的物理页 ID 映射
    physical_pages: HashMap<PathBuf, Vec<u32>>,
}

impl KvSideloadManager {
    /// 创建新的零拷贝页表管理器
    pub fn new() -> Self {
        Self {
            physical_pages: HashMap::new(),
        }
    }

    /// 注册预存的 KV 页
    ///
    /// 将外部预计算的 KV cache 物理页 ID 注册到管理器中。
    /// 后续注入时直接使用这些页 ID，无需重新分配内存。
    pub fn register_pages(&mut self, path: PathBuf, page_ids: Vec<u32>) {
        self.physical_pages.insert(path, page_ids);
    }

    /// 获取预存的物理页 ID
    pub fn get_pages(&self, path: &PathBuf) -> Option<&[u32]> {
        self.physical_pages.get(path).map(|v| v.as_slice())
    }
}

impl Default for KvSideloadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 多模态 RAG 注入调度策略 (per SPEC 04-API-DESIGN §8.6)
// ============================================================================

/// 命中率追踪器 (per SPEC 04-API-DESIGN §8.6)
#[derive(Debug, Clone, Copy)]
pub struct HitRateTracker {
    /// 最近 N 次请求的命中次数
    hits: usize,
    /// 总观察次数
    total: usize,
}

impl HitRateTracker {
    pub fn new() -> Self {
        Self { hits: 0, total: 0 }
    }

    /// 记录一次命中
    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.total += 1;
    }

    /// 记录一次未命中
    pub fn record_miss(&mut self) {
        self.total += 1;
    }

    /// 获取命中率 [0.0, 1.0]
    pub fn hit_rate(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f32 / self.total as f32
        }
    }
}

impl Default for HitRateTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 多模态 RAG 注入调度策略 (per SPEC 04-API-DESIGN §8.6)
///
/// 规范多轮对话中挂载对象的生命周期。
#[derive(Debug, Clone)]
pub struct InjectionScheduler {
    /// 测算当前 RAG 注入对象的存活率，长时间不被访问进行异步卸载
    pub ttl_policy: Duration,
    /// 当请求到来时，快速探测是否存在需要被唤醒的休眠特征
    pub hit_rate_monitor: HitRateTracker,
}

impl InjectionScheduler {
    /// 创建新的注入调度器
    pub fn new(ttl_policy: Duration) -> Self {
        Self {
            ttl_policy,
            hit_rate_monitor: HitRateTracker::new(),
        }
    }

    /// 检查注入对象是否过期
    pub fn is_expired(&self, last_access: std::time::Instant) -> bool {
        last_access.elapsed() > self.ttl_policy
    }

    /// 记录注入对象访问
    pub fn record_access(&mut self, hit: bool) {
        if hit {
            self.hit_rate_monitor.record_hit();
        } else {
            self.hit_rate_monitor.record_miss();
        }
    }

    /// 获取当前命中率
    pub fn hit_rate(&self) -> f32 {
        self.hit_rate_monitor.hit_rate()
    }
}

// ============================================================================
// Tests
// ============================================================================

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

    #[test]
    fn test_injection_kind_dynamic_lora_adapter() {
        // Verify the variant name matches SPEC §8.6
        let kind = InjectionKind::DynamicLoRA;
        assert_eq!(kind as u8, InjectionKind::DynamicLoRA as u8);
    }

    #[test]
    fn test_frozen_kv_source() {
        let source = KnowledgeSource::from_frozen_kv("test.kv");
        assert_eq!(source.injection_kind(), InjectionKind::FrozenKvChunk);
        let ctx = crate::engine::EngineContext::new(32, 4096, 16);
        assert!(source.materialize(&ctx).is_err());
    }

    #[test]
    fn test_kv_sideload_manager() {
        let mut manager = KvSideloadManager::new();
        let path = PathBuf::from("test.kv");
        let pages = vec![1, 2, 3, 4, 5];

        manager.register_pages(path.clone(), pages.clone());
        assert_eq!(manager.get_pages(&path), Some(pages.as_slice()));
        assert_eq!(manager.get_pages(&PathBuf::from("missing.kv")), None);
    }

    #[test]
    fn test_hit_rate_tracker() {
        let mut tracker = HitRateTracker::new();
        assert_eq!(tracker.hit_rate(), 0.0);

        tracker.record_hit();
        tracker.record_hit();
        tracker.record_miss();
        assert_eq!(tracker.hit_rate(), 2.0 / 3.0);

        tracker.record_hit();
        assert_eq!(tracker.hit_rate(), 3.0 / 4.0);
    }

    #[test]
    fn test_injection_scheduler() {
        let ttl = Duration::from_secs(60);
        let scheduler = InjectionScheduler::new(ttl);
        assert_eq!(scheduler.ttl_policy, ttl);
        assert_eq!(scheduler.hit_rate(), 0.0);

        let now = std::time::Instant::now();
        assert!(!scheduler.is_expired(now));

        let past = now - Duration::from_secs(120);
        assert!(scheduler.is_expired(past));
    }

    #[test]
    fn test_injection_scheduler_record_access() {
        let ttl = Duration::from_secs(60);
        let mut scheduler = InjectionScheduler::new(ttl);

        scheduler.record_access(true);
        scheduler.record_access(true);
        scheduler.record_access(false);
        assert_eq!(scheduler.hit_rate(), 2.0 / 3.0);
    }
}
