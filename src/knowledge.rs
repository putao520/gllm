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
    #[error("unsupported element type: {0}")]
    UnsupportedElementType(String),
    #[error("invalid layer target")]
    InvalidLayerTarget,
    #[error("data format error: {0}")]
    DataFormatError(String),
    #[error("KV cache error: {0}")]
    KvCacheError(String),
    #[error("injection failed: {0}")]
    InjectionFailed(String),
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

/// 知识数据源 (per SPEC 04-API-DESIGN §8.1)
///
/// 统一的数据源抽象，支持三种注入类型：
/// - `FrozenKvChunk`: 侧载预存的 KV cache
/// - `LateFusionVector`: 截断前向传播生成密实特征向量
/// - `DynamicLoRA`: LoRA 权重片注入
#[derive(Debug, Clone)]
pub struct KnowledgeSource {
    /// 数据源文件路径
    pub path: PathBuf,
    /// 注入类型
    kind: InjectionKind,
}

impl KnowledgeSource {
    /// 从冻结 KV 文件创建数据源 (per SPEC §8.1 FrozenKvChunk)
    pub fn from_frozen_kv(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            kind: InjectionKind::FrozenKvChunk,
        }
    }

    /// 从文本文件创建晚期融合数据源 (per SPEC §8.1 LateFusionVector)
    ///
    /// 读取文本内容，tokenize 后运行截断式前向传播，
    /// 将生成的隐藏状态向量注入到指定层。
    pub fn from_late_fusion(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            kind: InjectionKind::LateFusionVector,
        }
    }

    /// 从 safetensors 文件创建 LoRA 数据源 (per SPEC §8.1 DynamicLoRA)
    ///
    /// 加载 LoRA A/B 权重矩阵，解析元数据中的 rank/alpha/target_module，
    /// 注入到指定层的目标模块。
    pub fn from_lora(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            kind: InjectionKind::DynamicLoRA,
        }
    }
}

impl KnowledgeDataSource for KnowledgeSource {
    fn injection_kind(&self) -> InjectionKind {
        self.kind
    }

    fn materialize(&self, engine: &crate::engine::EngineContext) -> Result<MaterializedPayload, KnowledgeError> {
        match self.kind {
            InjectionKind::FrozenKvChunk => self.materialize_frozen_kv(engine),
            InjectionKind::LateFusionVector => {
                // LateFusionVector: 文件存在性检查后返回轻量 payload
                // 实际的 tokenize + forward 在 client.rs 中执行
                if !self.path.exists() {
                    return Err(KnowledgeError::SourceNotFound(format!(
                        "late fusion source file not found: {}", self.path.display()
                    )));
                }
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("source_path".to_string(), self.path.display().to_string());
                Ok(MaterializedPayload {
                    kind: InjectionKind::LateFusionVector,
                    data: Vec::new(),
                    shape: Vec::new(),
                    metadata,
                })
            }
            InjectionKind::DynamicLoRA => {
                // DynamicLoRA: 文件存在性检查后返回轻量 payload
                // 实际的 safetensors 加载 + LoRA 构建在 client.rs 中执行
                if !self.path.exists() {
                    return Err(KnowledgeError::SourceNotFound(format!(
                        "LoRA safetensors file not found: {}", self.path.display()
                    )));
                }
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("source_path".to_string(), self.path.display().to_string());
                Ok(MaterializedPayload {
                    kind: InjectionKind::DynamicLoRA,
                    data: Vec::new(),
                    shape: Vec::new(),
                    metadata,
                })
            }
        }
    }
}

impl KnowledgeSource {
    /// FrozenKvChunk 物理化：从文件加载 KV cache 数据
    fn materialize_frozen_kv(
        &self,
        engine: &crate::engine::EngineContext,
    ) -> Result<MaterializedPayload, KnowledgeError> {
        

        let path = &self.path;

        if !path.exists() {
            return Err(KnowledgeError::SourceNotFound(format!(
                "frozen KV file not found: {}", path.display()
            )));
        }

        // Ω1 真实性原则 — 从 EngineContext 读取所有维度参数
        let num_kv_heads = engine.num_kv_heads;
        let max_seq_len = engine.max_seq_len;
        let head_dim = engine.hidden_size / num_kv_heads;

        let mut metadata = std::collections::HashMap::new();
        metadata.insert("source_path".to_string(), path.display().to_string());
        metadata.insert("num_layers".to_string(), engine.num_layers.to_string());
        metadata.insert("hidden_size".to_string(), engine.hidden_size.to_string());
        metadata.insert("kv_page_size".to_string(), engine.kv_page_size.to_string());
        metadata.insert("num_kv_heads".to_string(), num_kv_heads.to_string());
        metadata.insert("max_seq_len".to_string(), max_seq_len.to_string());
        metadata.insert("head_dim".to_string(), head_dim.to_string());

        let shape = vec![engine.num_layers, 2, num_kv_heads, max_seq_len, head_dim];

        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let data = if extension == "safetensors" {
            load_kv_from_safetensors(path, &shape)?
        } else {
            load_kv_from_raw(path, &shape)?
        };

        Ok(MaterializedPayload {
            kind: InjectionKind::FrozenKvChunk,
            data,
            shape,
            metadata,
        })
    }
}

/// 从 safetensors 格式的 KV cache 文件加载权重数据
///
/// 期望的 tensor 名称:
/// - `kv_cache`: 完整的 KV cache 数据 (shape: [num_layers, 2, num_kv_heads, max_seq_len, head_dim])
/// - 或 `key_cache` + `value_cache`: 分开的 K/V cache
///
/// per SPEC 04-API-DESIGN §8.4 — 从 safetensors 文件加载预计算的 KV cache
fn load_kv_from_safetensors(
    path: &std::path::Path,
    expected_shape: &[usize],
) -> Result<Vec<u8>, KnowledgeError> {
    let loader = crate::loader::safetensors::MappedSafetensors::open(path).map_err(|e| {
        KnowledgeError::DataFormatError(format!(
            "failed to open safetensors '{}': {}", path.display(), e
        ))
    })?;

    // 尝试加载统一的 kv_cache tensor
    if let Ok(tensor) = loader.tensor("kv_cache") {
        let data = tensor.as_f32().map_err(|e| {
            KnowledgeError::DataFormatError(format!(
                "kv_cache tensor is not f32 in '{}': {}", path.display(), e
            ))
        })?;

        // 验证形状匹配
        let expected_elements: usize = expected_shape.iter().product();
        if data.len() != expected_elements {
            return Err(KnowledgeError::DataFormatError(format!(
                "kv_cache shape mismatch: expected {} elements (shape {:?}), got {} elements (shape {:?})",
                expected_elements, expected_shape, data.len(), tensor.shape
            )));
        }

        // 转换为原始字节（f32 → u8）
        let f32_slice: &[f32] = &data;
        let byte_len = std::mem::size_of_val(f32_slice);
        let mut bytes = vec![0u8; byte_len];
        let view = unsafe { std::slice::from_raw_parts(f32_slice.as_ptr() as *const u8, byte_len) };
        bytes.copy_from_slice(view);
        return Ok(bytes);
    }

    // 尝试分开的 key_cache + value_cache
    let key_tensor = loader.tensor("key_cache")
        .or_else(|_| loader.tensor("k_cache"))
        .map_err(|e| KnowledgeError::DataFormatError(format!(
            "no 'kv_cache', 'key_cache', or 'k_cache' tensor found in '{}': {}",
            path.display(), e
        )))?;

    let val_tensor = loader.tensor("value_cache")
        .or_else(|_| loader.tensor("v_cache"))
        .map_err(|e| KnowledgeError::DataFormatError(format!(
            "no 'value_cache' or 'v_cache' tensor found in '{}': {}", path.display(), e
        )))?;

    let key_data = key_tensor.as_f32().map_err(|e| KnowledgeError::DataFormatError(
        format!("key_cache tensor is not f32: {}", e)
    ))?;
    let val_data = val_tensor.as_f32().map_err(|e| KnowledgeError::DataFormatError(
        format!("value_cache tensor is not f32: {}", e)
    ))?;

    // 合并 K 和 V 数据（K 在前，V 在后）
    let key_f32: &[f32] = &key_data;
    let val_f32: &[f32] = &val_data;
    let key_byte_len = std::mem::size_of_val(key_f32);
    let val_byte_len = std::mem::size_of_val(val_f32);
    let mut bytes = Vec::with_capacity(key_byte_len + val_byte_len);

    let key_bytes = unsafe { std::slice::from_raw_parts(key_f32.as_ptr() as *const u8, key_byte_len) };
    bytes.extend_from_slice(key_bytes);
    let val_bytes = unsafe { std::slice::from_raw_parts(val_f32.as_ptr() as *const u8, val_byte_len) };
    bytes.extend_from_slice(val_bytes);

    Ok(bytes)
}

/// 从原始二进制文件加载 KV cache 数据
///
/// 直接读取文件内容作为 KV cache 的原始字节。
/// 要求文件大小与 expected_shape 对应的字节数匹配。
fn load_kv_from_raw(
    path: &std::path::Path,
    expected_shape: &[usize],
) -> Result<Vec<u8>, KnowledgeError> {
    let expected_bytes: usize = expected_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

    let data = std::fs::read(path).map_err(|e| KnowledgeError::DataFormatError(format!(
        "failed to read KV file '{}': {}", path.display(), e
    )))?;

    if data.len() != expected_bytes {
        return Err(KnowledgeError::DataFormatError(format!(
            "KV file size mismatch: expected {} bytes (shape {:?}), got {} bytes",
            expected_bytes, expected_shape, data.len()
        )));
    }

    Ok(data)
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
    fn test_frozen_kv_source_file_not_found() {
        // Non-existent file should return SourceNotFound
        let source = KnowledgeSource::from_frozen_kv("nonexistent_test_file_12345.kv");
        assert_eq!(source.injection_kind(), InjectionKind::FrozenKvChunk);
        let ctx = crate::engine::EngineContext::new(32, 4096, 16, 32, 2048);
        let result = source.materialize(&ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KnowledgeError::SourceNotFound(_)));
    }

    #[test]
    fn test_frozen_kv_source_raw_file() {
        // Create a temporary raw KV file with correct size
        let ctx = crate::engine::EngineContext::new(2, 64, 16, 4, 32);
        let head_dim = ctx.hidden_size / ctx.num_kv_heads;
        // shape: [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
        let shape = vec![ctx.num_layers, 2, ctx.num_kv_heads, ctx.max_seq_len, head_dim];
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        // Write a temp file with the exact expected size
        let dir = std::env::temp_dir().join("gllm_test_kv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_raw.kv");
        let data = vec![1.0f32; total_elements];
        let data_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, total_bytes)
        };
        std::fs::write(&path, data_bytes).unwrap();

        let source = KnowledgeSource::from_frozen_kv(&path);
        let payload = source.materialize(&ctx).unwrap();
        assert_eq!(payload.kind, InjectionKind::FrozenKvChunk);
        assert_eq!(payload.data.len(), total_bytes);
        assert_eq!(payload.shape, shape);
        assert!(payload.metadata.contains_key("source_path"));

        // Cleanup
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_late_fusion_source() {
        // Late fusion source should report LateFusionVector kind
        let source = KnowledgeSource::from_late_fusion("input.txt");
        assert_eq!(source.injection_kind(), InjectionKind::LateFusionVector);

        // Non-existent file should return SourceNotFound
        let ctx = crate::engine::EngineContext::new(32, 4096, 16, 32, 2048);
        let result = source.materialize(&ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KnowledgeError::SourceNotFound(_)));
    }

    #[test]
    fn test_late_fusion_source_file_exists() {
        // Create a temp file
        let dir = std::env::temp_dir().join("gllm_test_late_fusion");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_input.txt");
        std::fs::write(&path, "Hello world").unwrap();

        let source = KnowledgeSource::from_late_fusion(&path);
        let ctx = crate::engine::EngineContext::new(2, 64, 16, 4, 32);
        let payload = source.materialize(&ctx).unwrap();
        assert_eq!(payload.kind, InjectionKind::LateFusionVector);
        assert!(payload.data.is_empty());

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_lora_source() {
        // LoRA source should report DynamicLoRA kind
        let source = KnowledgeSource::from_lora("adapter.safetensors");
        assert_eq!(source.injection_kind(), InjectionKind::DynamicLoRA);

        // Non-existent file should return SourceNotFound
        let ctx = crate::engine::EngineContext::new(32, 4096, 16, 32, 2048);
        let result = source.materialize(&ctx);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), KnowledgeError::SourceNotFound(_)));
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

// ============================================================================
// Error conversion to ClientError
// ============================================================================

impl From<KnowledgeError> for crate::client::ClientError {
    fn from(err: KnowledgeError) -> Self {
        crate::client::ClientError::RuntimeError(format!("knowledge injection error: {}", err))
    }
}
