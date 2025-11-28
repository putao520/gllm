# 企业级BERT推理引擎实现指南

## 概述

本文档提供gllm库企业级BERT推理引擎的完整实现指导，确保支持26个主流嵌入模型和重排序模型的真实推理能力。

## 企业级要求

### 架构完整性
- 支持所有主流BERT变体：BERT、RoBERTa、DistilBERT、ELECTRA、ALBERT等
- 完整的多头注意力、前馈网络、层归一化实现
- 支持不同的激活函数：GELU、ReLU、SiLU、Swish等
- 动态层数、头数、隐藏层维度配置

### 配置驱动
- 基于HuggingFace config.json构建模型架构
- 支持26个模型的完整配置映射
- 配置验证、默认值填充、兼容性检查
- 模型类型自动检测（Embedding vs Rerank）

### 权重加载
- 完整的SafeTensors文件解析和验证
- 支持分片模型、压缩模型、量化模型
- Tensor名称映射和权重验证
- 企业级错误处理和降级策略

### 性能优化
- 高效的内存管理和批量处理
- GPU利用率优化和计算图优化
- 模型缓存和预加载机制
- 推理延迟和吞吐量优化

## 技术架构

### 核心组件

```
src/
├── dynamic_bert.rs          # 企业级动态BERT模型
├── model_config.rs          # 扩展的模型配置管理
├── weight_loader.rs         # SafeTensors权重加载器
├── engine.rs               # 推理引擎核心
├── model_registry.rs        # 26个模型注册表
└── bert_variants.rs         # BERT变体适配器
```

### 企业级DynamicBert架构

```rust
// 支持所有BERT变体的企业级实现
pub struct EnterpriseBertModel<B: Backend> {
    // 核心组件
    embeddings: BertEmbeddings<B>,
    encoder: BertEncoder<B>,
    pooler: BertPooler<B>,

    // 配置
    config: BertConfig,
    variant: BertVariant,

    // 性能优化
    cache: ModelCache<B>,
    profiler: Profiler,
}
```

## 实现任务块

### Block A: 企业级BERT架构重构 (4-6小时)

#### A1: 修复编译错误，升级到最新Burn API
**目标**: 解决所有15个编译错误，确保代码使用最新的Burn框架API

**关键修复点**:
```rust
// 修复Dropout API
use burn::nn::{Dropout, DropoutConfig};

// 修复ReLU实现
pub fn relu_forward<B: Backend>(input: Tensor<B, 3>) -> Tensor<B, 3> {
    burn::tensor::activation::relu(input)
}

// 修复tensor操作
tensor.repeat(&[batch_size])  // 而非 tensor.repeat(0, batch_size)
tensor.squeeze::<2>()            // 而非 tensor.squeeze(1)
```

**文件**: `src/dynamic_bert.rs`

#### A2: 实现完整的EnterpriseBertModel
**目标**: 支持所有BERT变体和企业级特性

**核心实现**:
```rust
impl<B: Backend> EnterpriseBertModel<B> {
    pub fn new(device: &B::Device, config: BertConfig) -> Result<Self> {
        let variant = BertVariant::detect(&config);

        // 动态构建embedding层
        let embeddings = BertEmbeddings::new(device, &config, &variant)?;

        // 动态构建encoder
        let encoder = BertEncoder::new(device, &config, &variant)?;

        // 动态构建pooler
        let pooler = BertPooler::new(device, &config, &variant)?;

        // 性能优化组件
        let cache = ModelCache::new(config.cache_size);
        let profiler = Profiler::new();

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            config,
            variant,
            cache,
            profiler,
        })
    }
}
```

#### A3: 支持所有BERT架构变体
**目标**: 实现BERT、RoBERTa、DistilBERT、ELECTRA等变体的差异处理

**变体适配**:
```rust
pub enum BertVariant {
    Bert,
    Roberta,
    DistilBert,
    Electra,
    Albert,
    // 更多变体...
}

impl BertVariant {
    pub fn detect(config: &BertConfig) -> Self {
        if config.model_type.as_ref().map_or(false, |t| t.contains("roberta")) {
            BertVariant::Roberta
        } else if config.model_type.as_ref().map_or(false, |t| t.contains("distilbert")) {
            BertVariant::DistilBert
        } else {
            BertVariant::Bert
        }
    }

    pub fn embedding_config(&self, base: &EmbeddingConfig) -> EmbeddingConfig {
        match self {
            BertVariant::Bert => base.clone(),
            BertVariant::Roberta => base.clone(),
            BertVariant::DistilBert => base.clone(),
            // 变体特定配置
        }
    }
}
```

**文件**: `src/dynamic_bert.rs`, `src/bert_variants.rs`

#### A4: 企业级性能优化
**目标**: 实现高效的内存管理、批量处理和GPU优化

**性能优化策略**:
```rust
pub struct PerformanceOptimizer {
    batch_size: usize,
    max_sequence_length: usize,
    memory_limit: usize,
    gpu_memory_fraction: f32,
}

impl PerformanceOptimizer {
    pub fn optimize_batch_size(&self, input_length: usize) -> usize {
        let optimal_batch = (self.memory_limit / input_length).min(self.batch_size);
        optimal_batch.max(1)
    }

    pub fn optimize_sequence_length(&self, inputs: &[&str]) -> usize {
        let max_len = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
        max_len.min(self.max_sequence_length)
    }
}
```

**文件**: `src/performance_optimizer.rs`

### Block B: 企业级配置系统 (3-4小时)

#### B1: 扩展ModelConfig支持所有HuggingFace配置
**目标**: 支持完整的HuggingFace配置规范

**扩展配置**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnterpriseModelConfig {
    // 核心BERT配置
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,

    // 高级配置
    pub attention_probs_dropout_prob: f32,
    pub hidden_dropout_prob: f32,
    pub classifier_dropout: Option<f32>,
    pub layer_norm_eps: f64,

    // 模型变体特定配置
    pub position_embedding_type: Option<String>,
    pub type_vocab_size: Option<usize>,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,

    // 性能配置
    pub use_cache: bool,
    pub gradient_checkpointing: bool,
    pub use_flash_attention: Option<bool>,

    // 量化配置
    pub quantization_config: Option<QuantizationConfig>,

    // 企业级扩展
    pub enterprise_config: EnterpriseExtensionConfig,
}
```

**文件**: `src/model_config.rs`

#### B2: 实现26个模型的完整配置映射
**目标**: 为README中所有26个模型提供准确的配置

**模型配置映射**:
```rust
impl EnterpriseModelConfig {
    pub fn for_model(repo_id: &str) -> Self {
        match repo_id {
            // BGE系列
            "BAAI/bge-m3" => Self {
                hidden_size: 1024,
                num_hidden_layers: 12,
                num_attention_heads: 16,
                intermediate_size: 4096,
                hidden_act: "gelu".to_string(),
                max_position_embeddings: 8192,
                vocab_size: 30522,
                // BGE-M3特定配置...
                enterprise_config: EnterpriseExtensionConfig {
                    pooling_strategy: PoolingStrategy::Cls,
                    normalize_embeddings: true,
                    sentence_pooling: true,
                },
            },

            // E5系列
            "intfloat/e5-large" => Self {
                hidden_size: 1024,
                num_hidden_layers: 24,
                num_attention_heads: 16,
                intermediate_size: 4096,
                hidden_act: "gelu".to_string(),
                // E5特定配置...
                enterprise_config: EnterpriseExtensionConfig {
                    pooling_strategy: PoolingStrategy::Mean,
                    normalize_embeddings: true,
                    instruction_format: "passage: ".to_string(),
                },
            },

            // 所有其他模型...
            _ => Self::default_for_embedding(),
        }
    }
}
```

**文件**: `src/model_config.rs`

#### B3: 企业级配置验证和错误处理
**目标**: 提供完善的配置验证和用户友好的错误信息

**配置验证**:
```rust
impl EnterpriseModelConfig {
    pub fn validate(&self) -> Result<Vec<ConfigWarning>> {
        let mut warnings = Vec::new();

        // 基础验证
        if self.hidden_size == 0 {
            return Err(Error::InvalidConfig("hidden_size must be > 0".to_string()));
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(Error::InvalidConfig(
                format!("hidden_size {} must be divisible by num_attention_heads {}",
                       self.hidden_size, self.num_attention_heads)
            ));
        }

        // 企业级警告
        if self.hidden_dropout_prob > 0.5 {
            warnings.push(ConfigWarning::HighDropout(self.hidden_dropout_prob));
        }

        if self.max_position_embeddings < 512 {
            warnings.push(ConfigWarning::ShortSequenceLimit(self.max_position_embeddings));
        }

        Ok(warnings)
    }

    pub fn auto_fix(&mut self) -> Vec<ConfigAutoFix> {
        let mut fixes = Vec::new();

        // 自动修复常见配置问题
        if self.intermediate_size.is_none() {
            self.intermediate_size = Some(self.hidden_size * 4);
            fixes.push(ConfigAutoFix::SetIntermediateSize(self.hidden_size * 4));
        }

        if self.layer_norm_eps.is_none() {
            self.layer_norm_eps = 1e-12;
            fixes.push(ConfigAutoFix::SetLayerNormEps(1e-12));
        }

        fixes
    }
}
```

**文件**: `src/model_config.rs`

### Block C: 企业级SafeTensors加载系统 (5-7小时)

#### C1: 完整的SafeTensors解析和验证
**目标**: 实现生产级SafeTensors文件处理

**SafeTensors加载器**:
```rust
pub struct SafeTensorsLoader {
    cache: LruCache<String, SafeTensorsFile>,
    validation_strict: bool,
    compression: CompressionType,
}

impl SafeTensorsLoader {
    pub fn load_file(&mut self, path: &Path) -> Result<SafeTensorsFile> {
        let cache_key = path.to_string_lossy().to_string();

        // 检查缓存
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // 验证文件
        self.validate_file(path)?;

        // 解析文件
        let file = SafeTensorsFile::open(path)?;

        // 企业级验证
        if self.validation_strict {
            self.validate_tensors(&file)?;
        }

        // 缓存文件
        self.cache.put(cache_key, file.clone());

        Ok(file)
    }

    fn validate_file(&self, path: &Path) -> Result<()> {
        // 文件存在性检查
        if !path.exists() {
            return Err(Error::WeightFileNotFound(path.to_path_buf()));
        }

        // 文件完整性检查
        let metadata = fs::metadata(path)
            .map_err(|e| Error::IoError(e))?;

        if metadata.len() == 0 {
            return Err(Error::CorruptedWeightFile(path.to_path_buf()));
        }

        // 文件格式验证
        let magic = self.read_magic_bytes(path)?;
        if magic != b"<>" {
            return Err(Error::InvalidWeightFormat(path.to_path_buf()));
        }

        Ok(())
    }
}
```

**文件**: `src/weight_loader.rs`

#### C2: 支持分片、压缩、量化模型
**目标**: 支持各种模型存储格式

**分片模型支持**:
```rust
pub struct ShardedReader {
    base_path: PathBuf,
    num_shards: usize,
    current_shard: usize,
}

impl ShardedReader {
    pub fn load_sharded_weights(&mut self, pattern: &str) -> Result<HashMap<String, TensorData>> {
        let mut all_weights = HashMap::new();

        for shard_idx in 0..self.num_shards {
            let shard_path = self.base_path.join(format!("{}-{:05}-of-{:05}.safetensors",
                                                   pattern, shard_idx + 1, self.num_shards));

            let shard_file = SafeTensorsFile::open(&shard_path)?;
            for (name, tensor) in shard_file tensors() {
                all_weights.insert(name.clone(), tensor.clone());
            }
        }

        Ok(all_weights)
    }
}

// 量化支持
pub struct QuantizedTensorLoader {
    quantization_type: QuantizationType,
    calibration_data: Option<TensorData>,
}

impl QuantizedTensorLoader {
    pub fn load_quantized_weights(&self, safetensors: &SafeTensorsFile) -> Result<HashMap<String, TensorData>> {
        let mut weights = HashMap::new();

        for (name, tensor) in safetensors tensors() {
            let loaded_tensor = match self.quantization_type {
                QuantizationType::Int8 => self.dequantize_int8(tensor)?,
                QuantizationType::Fp16 => self.dequantize_fp16(tensor)?,
                QuantizationType::Fp32 => tensor.clone(),
            };

            weights.insert(name.clone(), loaded_tensor);
        }

        Ok(weights)
    }
}
```

**文件**: `src/weight_loader.rs`, `src/quantization.rs`

#### C3: 企业级Tensor名称映射
**目标**: 智能映射不同模型的tensor名称

**Tensor映射器**:
```rust
pub struct TensorMapper {
    mapping_rules: HashMap<String, String>,
    model_type: ModelType,
}

impl TensorMapper {
    pub fn new(model_config: &EnterpriseModelConfig) -> Self {
        let mut rules = HashMap::new();

        // 通用BERT映射规则
        rules.insert("embeddings.word_embeddings.weight".to_string(),
                     "embeddings.weight".to_string());
        rules.insert("embeddings.position_embeddings.weight".to_string(),
                     "position_embeddings.weight".to_string());
        rules.insert("embeddings.token_type_embeddings.weight".to_string(),
                     "token_type_embeddings.weight".to_string());

        // 模型特定规则
        match model_config.model_type.as_deref() {
            Some("roberta") => {
                rules.insert("embeddings.LayerNorm.weight".to_string(),
                             "embeddings.LayerNorm.weight".to_string());
                rules.insert("embeddings.LayerNorm.bias".to_string(),
                             "embeddings.LayerNorm.bias".to_string());
            },
            Some("distilbert") => {
                rules.insert("vocab_transform.weight".to_string(),
                             "vocab_projector.weight".to_string());
                rules.insert("vocab_transform.bias".to_string(),
                             "vocab_projector.bias".to_string());
            },
            _ => {}
        }

        Self {
            mapping_rules: rules,
            model_type: ModelType::from_config(model_config),
        }
    }

    pub fn map_tensor_name(&self, hf_name: &str) -> Option<String> {
        // 直接匹配
        if let Some(mapped) = self.mapping_rules.get(hf_name) {
            return Some(mapped.clone());
        }

        // 模式匹配
        for (pattern, replacement) in &self.mapping_rules {
            if hf_name.contains(pattern) {
                return Some(hf_name.replace(pattern, replacement));
            }
        }

        // 层级映射
        self.map_layer_tensors(hf_name)
    }

    fn map_layer_tensors(&self, name: &str) -> Option<String> {
        let layer_re = Regex::new(r"encoder\.layer\.(\d+)\.(.+)")
            .map_err(|_| ())
            .ok()?;

        if let Some(captures) = layer_re.captures(name) {
            let layer_idx = captures.get(1)?.as_str();
            let component = captures.get(2)?.as_str();

            Some(format!("layers.{}.{}", layer_idx, component))
        } else {
            None
        }
    }
}
```

**文件**: `src/tensor_mapper.rs`

### Block D: 企业级推理引擎优化 (4-6小时)

#### D1: 动态输出维度和Pooling策略
**目标**: 支持不同模型的正确embedding维度和pooling

**动态Pooling**:
```rust
pub enum PoolingStrategy {
    Cls,
    Mean,
    Max,
    WeightedMean,
    LastToken,
}

pub struct DynamicPooler<B: Backend> {
    strategy: PoolingStrategy,
    config: PoolingConfig,
    attention_mask: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> DynamicPooler<B> {
    pub fn new(strategy: PoolingStrategy, config: PoolingConfig) -> Self {
        Self {
            strategy,
            config,
            attention_mask: None,
        }
    }

    pub fn pool(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();

        match self.strategy {
            PoolingStrategy::Cls => {
                hidden_states.clone().slice([0..batch_size, 0..1, 0..hidden_size])
                    .squeeze::<2>()
            },

            PoolingStrategy::Mean => {
                if let Some(ref mask) = self.attention_mask {
                    // Masked mean pooling
                    let expanded_mask = mask.unsqueeze_dim(2)
                        .expand([batch_size, seq_len, hidden_size]);
                    let masked = hidden_states * expanded_mask;
                    let sum = masked.sum_dim(1);
                    let count = expanded_mask.sum_dim(1);
                    sum / count.clamp(1e-9, f32::MAX)
                } else {
                    hidden_states.mean_dim(1)
                }
            },

            PoolingStrategy::Max => {
                hidden_states.max_dim(1)
            },

            PoolingStrategy::WeightedMean => {
                self.weighted_mean_pooling(hidden_states)
            },

            PoolingStrategy::LastToken => {
                let last_indices = self.get_last_token_indices();
                hidden_states.gather(1, last_indices.unsqueeze_dim(1))
            }
        }
    }
}
```

**文件**: `src/engine.rs`, `src/pooling.rs`

#### D2: 高性能批量处理
**目标**: 优化批量推理性能

**批量优化器**:
```rust
pub struct BatchOptimizer {
    max_batch_size: usize,
    max_sequence_length: usize,
    memory_limit_mb: usize,
}

impl BatchOptimizer {
    pub fn optimize_batch<T>(&self, inputs: Vec<T>) -> Vec<Vec<T>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_tokens = 0;

        for item in inputs {
            let item_tokens = self.estimate_tokens(&item);

            // 检查是否需要新的batch
            if self.should_start_new_batch(&current_batch, item_tokens, current_tokens) {
                if !current_batch.is_empty() {
                    batches.push(current_batch);
                }
                current_batch = Vec::new();
                current_tokens = 0;
            }

            current_batch.push(item);
            current_tokens += item_tokens;
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    fn should_start_new_batch<T>(&self, batch: &[T], new_tokens: usize, current_tokens: usize) -> bool {
        if batch.len() >= self.max_batch_size {
            return true;
        }

        if current_tokens + new_tokens > self.max_sequence_length {
            return true;
        }

        false
    }
}
```

**文件**: `src/batch_optimizer.rs`

#### D3: GPU内存优化
**目标**: 高效的GPU内存管理和利用

**GPU优化器**:
```rust
pub struct GpuOptimizer<B: Backend> {
    device: B::Device,
    memory_pool: MemoryPool<B>,
    tensor_cache: LruCache<String, Tensor<B, 3>>,
}

impl<B: Backend> GpuOptimizer<B> {
    pub fn allocate_tensor(&mut self, shape: Shape, dtype: DType) -> Result<Tensor<B, 3>> {
        // 尝试从缓存获取
        let cache_key = format!("{:?}_{:?}", shape, dtype);
        if let Some(cached) = self.tensor_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // 分配新tensor
        let tensor = Tensor::zeros(shape, &self.device);

        // 缓存tensor
        self.tensor_cache.put(cache_key, tensor.clone());

        Ok(tensor)
    }

    pub fn optimize_memory_layout(&mut self, tensors: &mut [Tensor<B, 3>]) -> Result<()> {
        // 内存布局优化
        for tensor in tensors {
            if tensor.can_be_contiguous() {
                *tensor = tensor.contiguous();
            }
        }

        Ok(())
    }

    pub fn clear_cache(&mut self) {
        self.tensor_cache.clear();
        self.memory_pool.clear();
    }
}
```

**文件**: `src/gpu_optimizer.rs`

### Block E: 企业级测试和验证 (4-5小时)

#### E1: 26个模型功能验证
**目标**: 确保所有26个模型能正确工作

**模型验证框架**:
```rust
pub struct ModelValidator {
    test_models: Vec<String>,
    tolerance: f32,
    quick_mode: bool,
}

impl ModelValidator {
    pub fn validate_all_models(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        for model_id in &self.test_models {
            let result = self.validate_single_model(model_id)?;
            report.add_model_result(model_id.clone(), result);
        }

        Ok(report)
    }

    fn validate_single_model(&self, model_id: &str) -> Result<ModelValidationResult> {
        // 创建测试client
        let client = Client::new(model_id)?;

        // 配置验证
        let config_result = self.validate_model_config(model_id)?;

        // Embedding验证
        let embedding_result = self.validate_embeddings(&client)?;

        // 维度验证
        let dimension_result = self.validate_dimensions(&client, model_id)?;

        // 性能验证
        let performance_result = if !self.quick_mode {
            Some(self.validate_performance(&client)?)
        } else {
            None
        };

        Ok(ModelValidationResult {
            model_id: model_id.to_string(),
            config_valid: config_result,
            embedding_valid: embedding_result,
            dimension_correct: dimension_result,
            performance_acceptable: performance_result,
        })
    }

    fn validate_embeddings(&self, client: &Client) -> Result<bool> {
        let test_texts = vec![
            "This is a test sentence.",
            "Another test sentence for validation.",
            "Machine learning is fascinating!",
        ];

        let response = client.embeddings(&test_texts).generate()?;

        // 验证返回结果
        if response.embeddings.len() != test_texts.len() {
            return Ok(false);
        }

        // 验证embedding维度
        let first_dim = response.embeddings[0].embedding.len();
        for embedding in &response.embeddings {
            if embedding.embedding.len() != first_dim {
                return Ok(false);
            }

            // 验证向量归一化
            let norm: f32 = embedding.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if (norm - 1.0).abs() > self.tolerance {
                return Ok(false);
            }
        }

        Ok(true)
    }
}
```

**文件**: `tests/validation/model_validator.rs`

## 验收标准

### 功能验收
1. ✅ 所有26个模型能正常加载和推理
2. ✅ Embedding输出维度与模型规格匹配
3. ✅ Rerank功能正常，评分范围合理
4. ✅ 配置系统能正确解析所有模型config.json
5. ✅ SafeTensors权重加载完整可用

### 性能验收
1. ✅ 内存使用合理，无内存泄漏
2. ✅ 批量处理性能满足生产要求
3. ✅ GPU利用率优化合理
4. ✅ 冷启动和热启动性能差异可接受

### 企业级验收
1. ✅ 错误处理完善，用户友好的错误信息
2. ✅ 日志和监控功能完整
3. ✅ 配置验证和自动修复机制
4. ✅ 完整的文档和使用示例

## 实施优先级

1. **P0 (必须)**: 编译错误修复 + 基础BERT架构
2. **P1 (重要)**: 配置系统 + 26个模型支持
3. **P2 (推荐)**: SafeTensors加载 + 性能优化
4. **P3 (可选)**: 高级特性 + 监控功能

这个实施计划确保gllm库达到企业级标准，真正支持26个模型的完整推理能力。