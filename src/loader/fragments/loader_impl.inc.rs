#[derive(Debug)]
pub struct Loader {
    manifest: ModelManifest,
    weight_paths: Vec<PathBuf>,
    config_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    format: WeightFormat,
    tie_word_embeddings_hint: Option<bool>,
    /// User-specified compute dtype. Overrides model native dtype for weight dequantization.
    compute_dtype: Option<gllm_kernels::types::DType>,

    /// COMP12: Weight page compression configuration (SPEC 22 §6).
    /// Controls which codec is used at each tier for weight pages.
    weight_compress_config: Option<weight_compress::WeightCompressionConfig>,

    // Internal loaders (only one is active)
    safetensors: Option<safetensors::SafeTensorsLoader>,
    gguf: Option<gguf::GgufReader>,
    gllm: Option<gllm::GllmReader>,
    onnx: Option<onnx::OnnxLoader>,
    pytorch: Option<pytorch::PytorchLoader>,

    // #6: Configurable tensor skip strategy.
    tensor_skip_config: TensorSkipConfig,
    // #5: Runtime-extensible suffix patterns for match_tensor_role().
    extra_suffix_patterns: Vec<(Vec<String>, TensorRole, bool)>,
    // #14: Cached config.json Value to avoid repeated IO.
    config_json: std::sync::OnceLock<serde_json::Value>,
}

/// Per-tensor processing result for concurrent upload pipeline.
enum TensorProcessResult<B: Backend<E>, E: Element> {
    Native {
        name: String,
        meta: TensorMeta,
        tensor: B::Tensor,
        placement: crate::compat::backend_trait::WeightPlacement,
        sp_meta: Option<Vec<Vec<u16>>>,
    },
    RawFloat {
        name: String,
        meta: TensorMeta,
        data: RawFloatTensor,
    },
    Quantized {
        name: String,
        meta: TensorMeta,
        data: QuantizedTensor,
    },
    Skipped,
}

impl Loader {
    pub fn new(manifest: ModelManifest) -> Self {
        Self {
            manifest,
            weight_paths: Vec::new(),
            config_path: None,
            tokenizer_path: None,
            format: WeightFormat::SafeTensors, // Default, will be detected
            tie_word_embeddings_hint: None,
            compute_dtype: None,
            weight_compress_config: None,
            safetensors: None,
            gguf: None,
            gllm: None,
            onnx: None,
            pytorch: None,
            tensor_skip_config: TensorSkipConfig::default(),
            extra_suffix_patterns: Vec::new(),
            config_json: std::sync::OnceLock::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        Ok(Self::new(ModelManifest::default()))
    }

    pub fn from_env_with_manifest(manifest: ModelManifest) -> Result<Self> {
        Ok(Self::new(manifest))
    }

    pub fn from_source_with_config(model_id: String, config: LoaderConfig) -> Result<Self> {
        let local_path = Path::new(&model_id);
        if local_path.is_file() {
            return Self::from_local_file(local_path);
        }
        if local_path.is_dir() {
            return Self::from_local_dir(local_path);
        }

        let cache = CacheLayout::new(config.cache_dir.clone())
            .map_err(|e| LoaderError::Cache(e.to_string()))?;
        cache.ensure()?;

        let (weights, format, aux_files) = match config.source {
            ModelSource::HuggingFace => {
                let api = HfHubClient::with_endpoint_and_token_path(
                    cache.hf_cache_dir(),
                    None,
                    config.hf_token_path.clone(),
                )?;
                let parallel = ParallelLoader::new(true);

                // Try HF first
                match api.download_model_files_filtered(
                    &model_id,
                    EMPTY_FILE_MAP,
                    parallel,
                    config.gguf_file_filter.as_deref(),
                ) {
                    Ok(files) => {
                        let fmt = match files.format {
                            hf_hub::WeightFormat::SafeTensors => WeightFormat::SafeTensors,
                            hf_hub::WeightFormat::Gguf => WeightFormat::Gguf,
                            hf_hub::WeightFormat::Onnx => WeightFormat::Onnx,
                            hf_hub::WeightFormat::PyTorch => WeightFormat::PyTorch,
                        };
                        (files.weights, fmt, files.aux_files)
                    }
                    Err(err) => {
                        // Fallback to ModelScope if enabled and error is recoverable
                        if config.enable_fallback && is_recoverable_error(&err) {
                            log::warn!(
                                "HuggingFace download failed, falling back to ModelScope: model_id={}, error={}",
                                model_id, err
                            );
                            let ms_api = ModelScopeClient::new(cache.modelscope_cache_dir())?;
                            let ms_files = ms_api.download_model_files(
                                &model_id,
                                EMPTY_FILE_MAP,
                                ParallelLoader::new(true),
                                config.gguf_file_filter.as_deref(),
                            )?;
                            (ms_files.weights, ms_files.format, ms_files.aux_files)
                        } else {
                            return Err(err);
                        }
                    }
                }
            }
            ModelSource::ModelScope => {
                let api = ModelScopeClient::new(cache.modelscope_cache_dir())?;
                let files =
                    api.download_model_files(&model_id, EMPTY_FILE_MAP, ParallelLoader::new(true), config.gguf_file_filter.as_deref())?;
                (files.weights, files.format, files.aux_files)
            }
        };

        let mut loader = Self::new(ModelManifest::default());
        loader.weight_paths = weights;
        loader.format = format;
        loader.tensor_skip_config = config.tensor_skip_config;
        loader.extra_suffix_patterns = config.extra_suffix_patterns;

        // Populate config/tokenizer paths from aux_files
        for path in aux_files {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path.clone());
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path.clone());
                }
            }
        }

        // #5: Inject runtime-extensible suffix patterns into global registry.
        // OnceLock is safe under ARCH-SINGLE-MODEL-INSTANCE (one model per process).
        if !loader.extra_suffix_patterns.is_empty() {
            let _ = set_extra_suffix_patterns(
                loader.extra_suffix_patterns.clone()
            );
        }

        Ok(loader)
    }

    /// 从本地单文件加载模型（支持 GGUF / ONNX / .gllm 等自包含格式）
    fn from_local_file(file: &Path) -> Result<Self> {
        let mut loader = Self::new(ModelManifest::default());
        loader.weight_paths = vec![file.to_path_buf()];
        loader.detect_format();

        // GGUF 文件内嵌 tokenizer，不需要外部 tokenizer.json
        // 对于 ONNX/safetensors 单文件场景，可能需要同目录的 config.json
        if let Some(parent) = file.parent() {
            for entry in std::fs::read_dir(parent).map_err(LoaderError::Io)? {
                let entry = entry.map_err(LoaderError::Io)?;
                if let Some(name) = entry.file_name().to_str() {
                    if name == "config.json" {
                        loader.config_path = Some(entry.path());
                    } else if name == "tokenizer.json" {
                        loader.tokenizer_path = Some(entry.path());
                    }
                }
            }
        }

        Ok(loader)
    }

    /// 从本地目录加载模型文件
    fn from_local_dir(dir: &Path) -> Result<Self> {
        let local = format_detector::collect_local_files(dir, None)?;

        let mut loader = Self::new(ModelManifest::default());
        loader.weight_paths = local.weights;
        loader.format = local.format;

        for path in local.aux_files {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path);
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path);
                }
            }
        }

        Ok(loader)
    }

    pub fn with_weights(mut self, paths: Vec<PathBuf>) -> Self {
        self.weight_paths = paths;
        self.detect_format();
        self
    }

    pub fn weight_paths(&self) -> &[PathBuf] {
        &self.weight_paths
    }

    /// Returns user-specified compute dtype (for overriding model native dtype).
    pub fn compute_dtype(&self) -> Option<gllm_kernels::types::DType> {
        self.compute_dtype
    }

    pub fn with_config(mut self, path: PathBuf) -> Self {
        self.config_path = Some(path);
        self
    }

    pub fn with_tokenizer(mut self, path: PathBuf) -> Self {
        self.tokenizer_path = Some(path);
        self
    }

    /// Override the compute dtype for inference.
    ///
    /// By default, weights are dequantized to the model's native dtype (from config.json).
    /// Setting this allows native mixed precision: e.g., BF16 model → FP8 compute,
    /// F32 model → BF16 compute, or any combination.
    pub fn with_compute_dtype(mut self, dtype: gllm_kernels::types::DType) -> Self {
        self.compute_dtype = Some(dtype);
        self
    }

    /// COMP12: Set weight page compression configuration (SPEC 22 §6).
    ///
    /// Controls which compression codec is used at each storage tier for weight
    /// pages. When set, the loader applies compression during weight loading
    /// according to the tier and weight class (DenseLayerWeight vs ExpertWeight).
    pub fn with_weight_compress_config(
        mut self,
        config: weight_compress::WeightCompressionConfig,
    ) -> Self {
        self.weight_compress_config = Some(config);
        self
    }

    fn detect_format(&mut self) {
        if let Some(first) = self.weight_paths.first() {
            if let Some(ext) = first.extension() {
                if ext == "gguf" {
                    self.format = WeightFormat::Gguf;
                    return;
                }
                if ext == "onnx" {
                    self.format = WeightFormat::Onnx;
                    return;
                }
                if ext == "pt" || ext == "bin" || ext == "pth" {
                    self.format = WeightFormat::PyTorch;
                    return;
                }
            }
        }
        self.format = WeightFormat::SafeTensors;
    }

    pub fn load(mut self) -> Result<Self> {
        match self.format {
            WeightFormat::SafeTensors => {
                let loader = safetensors::SafeTensorsLoader::from_files(
                    &self.weight_paths,
                    parallel::ParallelLoader::new(true),
                )?;
                self.safetensors = Some(loader);
            }
            WeightFormat::Gguf => {
                let reader = gguf::GgufReader::from_files(&self.weight_paths)?;
                self.gguf = Some(reader);
            }
            WeightFormat::Onnx => {
                // ONNX usually single file for now
                if let Some(path) = self.weight_paths.first() {
                    let loader = onnx::OnnxLoader::from_path(path)?;
                    self.onnx = Some(loader);
                }
            }
            WeightFormat::PyTorch => {
                let loader = pytorch::PytorchLoader::from_files(&self.weight_paths)?;
                self.pytorch = Some(loader);
            }
            WeightFormat::Gllm => {
                let reader = gllm::GllmReader::from_files(&self.weight_paths)?;
                self.gllm = Some(reader);
            }
        }
        Ok(self)
    }

    pub fn weight_format(&self) -> WeightFormat {
        self.format
    }

    pub fn config_path(&self) -> Option<&Path> {
        self.config_path.as_deref()
    }

    /// #14: Lazily read and cache config.json. Returns None if no config path or parse failure.
    pub fn config_json(&self) -> Option<&serde_json::Value> {
        let val = self.config_json.get_or_init(|| {
            let path = match self.config_path.as_deref() {
                Some(p) if p.exists() => p,
                _ => return serde_json::Value::Null,
            };
            match std::fs::read_to_string(path) {
                Ok(content) => serde_json::from_str(&content).unwrap_or(serde_json::Value::Null),
                Err(_) => serde_json::Value::Null,
            }
        });
        if val.is_null() { None } else { Some(val) }
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer_path.as_deref()
    }

    pub fn safetensors_loader(&mut self) -> Result<&mut safetensors::SafeTensorsLoader> {
        self.safetensors.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn safetensors_ref(&self) -> Option<&safetensors::SafeTensorsLoader> {
        self.safetensors.as_ref()
    }

    pub fn gguf_reader(&mut self) -> Result<&mut gguf::GgufReader> {
        self.gguf.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn gguf_ref(&self) -> Option<&gguf::GgufReader> {
        self.gguf.as_ref()
    }

    pub fn gllm_reader(&mut self) -> Result<&mut gllm::GllmReader> {
        self.gllm.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn gllm_ref(&self) -> Option<&gllm::GllmReader> {
        self.gllm.as_ref()
    }

    pub fn onnx_loader(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx.as_mut().ok_or(LoaderError::MissingWeights)
    }

    pub fn onnx_ref(&self) -> Option<&onnx::OnnxLoader> {
        self.onnx.as_ref()
    }

    pub fn onnx(&mut self) -> Result<&mut onnx::OnnxLoader> {
        self.onnx_loader()
    }

    /// 获取统一的 OnnxGraph 表示 (REQ-EXEC-001)
    ///
    /// 无论原始格式是 ONNX、SafeTensors 还是 GGUF，都转换为统一的 OnnxGraph。
    /// 检测模型架构（统一入口）
    ///
    /// 优先级：GGUF metadata > config.json model_type > 张量名称模式匹配 > manifest fallback
    pub fn detect_architecture(&self) -> String {
        use crate::manifest::map_architecture_token;

        // 1. GGUF metadata
        if let Some(gguf) = &self.gguf {
            if let Ok(arch_str) = gguf.architecture() {
                if let Some(arch) = map_architecture_token(arch_str) {
                    return arch;
                }
            }
        }

        // 1.5 .gllm metadata (SPEC 36 §4 REQ-GLF-006)
        if let Some(gllm) = &self.gllm {
            if let Some(arch_str) = gllm.architecture() {
                if let Some(arch) = map_architecture_token(&arch_str) {
                    return arch;
                }
            }
        }

        // 2. config.json model_type / architectures (SafeTensors/ONNX) — #14: uses cached config
        if let Some(json) = self.config_json() {
            if let Some(arr) = json.get("architectures").and_then(|v| v.as_array()) {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        if let Some(arch) = map_architecture_token(s) {
                            return arch;
                        }
                    }
                }
            }
            if let Some(s) = json.get("model_type").and_then(|v| v.as_str()) {
                if let Some(arch) = map_architecture_token(s) {
                    return arch;
                }
            }
        }

        // 3. 张量名称模式匹配
        if let Some(arch) = self.detect_architecture_from_tensors() {
            return arch;
        }

        // 4. manifest fallback
        self.manifest.arch.clone()
    }

    /// #4: Validate that detected architecture has key tensor roles recognized.
    /// If a known architecture is detected but no critical tensor roles are found,
    /// log a warning — this likely means tensor naming is non-standard.
    pub fn validate_architecture_tensor_roles(&self, arch: &str) {
        let critical_roles = [
            TensorRole::AttentionQuery,
            TensorRole::AttentionKey,
            TensorRole::AttentionValue,
        ];
        let mut found_any = false;

        macro_rules! check_provider {
            ($provider:expr) => {
                if let Some(p) = $provider {
                    for meta in p.iter_tensors() {
                        if let Some((role, _)) = match_tensor_role(&meta.name) {
                            if critical_roles.contains(&role) {
                                found_any = true;
                                break;
                            }
                        }
                    }
                }
            };
        }

        check_provider!(self.safetensors.as_ref());
        if !found_any { check_provider!(self.onnx.as_ref()); }
        if !found_any { check_provider!(self.gguf.as_ref()); }
        if !found_any { check_provider!(self.pytorch.as_ref()); }

        if !found_any {
            log::warn!(
                "Architecture '{}' detected but no critical tensor roles (AttentionQuery/Key/Value) found. \
                 Tensor naming may be non-standard or match_tensor_role patterns need updating.",
                arch
            );
        }
    }

    /// 从张量名称推断架构
    ///
    /// REQ-ARCH-Ω1: 禁止使用 contains() 模糊匹配，必须使用前缀匹配或张量形状推导
    fn detect_architecture_from_tensors(&self) -> Option<String> {
        // 检查单个张量名称是否匹配特定架构模式
        // 使用前缀匹配而非 contains() 避免模糊匹配
        let check_name = |name: &str| -> Option<String> {
            let lower = name.to_ascii_lowercase();

            // 将名称按 '.' 分割进行前缀匹配
            let parts: Vec<&str> = lower.split('.').collect();

            // BERT/RoBERTa/XLMR 风格: 前缀匹配
            // "bert.embeddings", "roberta.encoder", "xlmr."
            if parts.first().is_some_and(|p| {
                matches!(*p, "bert" | "roberta" | "xlmr" | "encoder")
            }) {
                return Some("xlmr".to_string());
            }

            // Mistral 风格: 前缀匹配 "model.layers" 或 "mistral."
            if parts.first().is_some_and(|p| {
                matches!(*p, "mistral" | "model")
            }) && parts.get(1).is_some_and(|p| {
                matches!(*p, "layers" | "embeddings")
            }) {
                return Some("mistral3".to_string());
            }

            // BERT encoder 模式: "encoder.layer.{N}.{...}" 或 "bert.encoder.layer.{N}"
            // 使用精确路径匹配而非 contains
            if parts.len() >= 3
                && ((parts[0] == "encoder" && parts[1] == "layer")
                    || (parts[0] == "bert" && parts[1] == "encoder" && parts[2] == "layer"))
                {
                    return Some("xlmr".to_string());
                }

            // BERT attention 模式: "attention.self.query" 精确路径匹配
            if parts.len() >= 3 && parts[1] == "attention" && parts[2] == "self" {
                return Some("xlmr".to_string());
            }

            None
        };

        // GPT-OSS (openai/gpt-oss-20b): 独有特征是 `self_attn.sinks` (attention sinks)
        // 和 `mlp.experts.gate_up_proj_blocks` (packed mxfp4 expert weights)。
        // 张量前缀 `model.layers.*` 与 Mistral 相同,但 GPT-OSS 有 MoE packed layout。
        // 必须在通用 check_name 之前检测,否则会被误识别为 mistral3。
        let is_gptoss_name = |name: &str| -> bool {
            let lower = name.to_ascii_lowercase();
            lower.contains("self_attn.sinks")
                || lower.contains("mlp.experts.gate_up_proj_blocks")
        };

        // Gemma 4 multi-modal SafeTensors 的张量布局是
        //   model.{audio_tower,vision_tower,embed_audio,embed_vision}.*
        //   model.language_model.{layers.*,embed_tokens,...}
        // 与单模态 LLM (model.layers.*) 完全不同。这种 multi-modal nesting 在所有
        // `check_name` 规则下都会落空,因为没有任何分支能把
        // `model.language_model` 识别成 decoder family。先扫一遍找
        // `model.language_model.` / `language_model.` 前缀,命中即返回 `gemma4`。
        // 优先级高于通用 check_name,但不会影响其他 SafeTensors 模型 (它们没有
        // language_model 这一层 nesting)。
        let is_gemma4_name = |name: &str| -> bool {
            let lower = name.to_ascii_lowercase();
            lower.starts_with("model.language_model.")
                || lower.starts_with("language_model.")
        };

        if let Some(st) = self.safetensors.as_ref() {
            if st.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if st.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in st.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(onnx) = self.onnx.as_ref() {
            if onnx.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if onnx.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in onnx.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(gguf) = self.gguf.as_ref() {
            if gguf.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if gguf.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in gguf.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        if let Some(pytorch) = self.pytorch.as_ref() {
            if pytorch.iter_tensors().any(|m| is_gptoss_name(&m.name)) {
                return Some("gptoss".to_string());
            }
            if pytorch.iter_tensors().any(|m| is_gemma4_name(&m.name)) {
                return Some("gemma4".to_string());
            }
            for meta in pytorch.iter_tensors() {
                if let Some(arch) = check_name(&meta.name) {
                    return Some(arch);
                }
            }
        }
        None
    }

    pub fn set_manifest_if_missing(&mut self, manifest: &ModelManifest) {
        // Simple overwrite or check if empty?
        // Assuming override for now as `from_env` creates a default one.
        self.manifest = manifest.clone();
    }

    pub fn set_tie_word_embeddings_hint(&mut self, hint: Option<bool>) {
        self.tie_word_embeddings_hint = hint;
    }

    pub fn gguf_architecture(&self) -> Result<&str> {
        if let Some(reader) = &self.gguf {
            reader
                .architecture()
                .map_err(|e| LoaderError::Gguf(e.to_string()))
        } else {
            Err(LoaderError::MissingWeights)
        }
    }

    // Helper for config derivation
    pub fn safetensors_gllm_config(&self) -> Result<Option<&Value>> {
        if let Some(loader) = &self.safetensors {
            Ok(loader.gllm_config())
        } else {
            Ok(None)
        }
    }

    /// Detect the dominant weight dtype from loaded tensors.
    /// Returns `DType` enum for type-safe dtype handling.
    pub fn detect_weight_dtype(&self) -> Result<Option<gllm_kernels::types::DType>> {
        use gllm_kernels::types::DType;
        if let Some(loader) = &self.safetensors {
            Ok(loader.detect_weight_dtype())
        } else if let Some(reader) = &self.gguf {
            Ok(reader.floating_point_dtype())
        } else if let Some(loader) = &self.onnx {
            let precisions = loader.unique_precisions();
            for dtype in precisions {
                match dtype {
                    Dtype::BF16 => return Ok(Some(DType::BF16)),
                    Dtype::F16 => return Ok(Some(DType::F16)),
                    Dtype::F32 => return Ok(Some(DType::F32)),
                    Dtype::F64 => return Ok(Some(DType::F32)), // f64 降级到 f32
                    _ => continue,
                }
            }
            Ok(None)
        } else if let Some(loader) = &self.pytorch {
            for meta in loader.iter_tensors() {
                match meta.dtype {
                    Dtype::BF16 => return Ok(Some(DType::BF16)),
                    Dtype::F16 => return Ok(Some(DType::F16)),
                    Dtype::F32 => return Ok(Some(DType::F32)),
                    Dtype::F64 => return Ok(Some(DType::F32)),
                    _ => continue,
                }
            }
            Ok(None)
        } else {
            Ok(None)
        }
    }

    pub fn upload_weights<B: Backend<E>, E: Element>(
        &mut self,
        backend: &B,
    ) -> Result<WeightsHandle<B, E>> {
        let format = self.format;
        match format {
            WeightFormat::SafeTensors => {
                let provider = self
                    .safetensors
                    .as_ref()
                    .ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::Gguf => {
                let provider = self.gguf.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::Onnx => {
                let provider = self.onnx.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::Gllm => {
                let provider = self.gllm.as_ref().ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
            WeightFormat::PyTorch => {
                let provider = self.pytorch.as_ref()
                    .ok_or(LoaderError::MissingWeights)?;
                self.upload_provider(provider, backend, format)
            }
        }
    }

    fn upload_provider<P: TensorProvider + Sync, B: Backend<E>, E: Element>(
        &self,
        provider: &P,
        backend: &B,
        format: WeightFormat,
    ) -> Result<WeightsHandle<B, E>> {
        use crate::compat::backend_trait::WeightPlacement;

        // Pass 1: collect + filter + sort (back-to-front priority)
        let mut tensor_metas: Vec<TensorMeta> = provider
            .iter_tensors()
            .filter(|m| !should_skip_tensor(&m.name, &self.tensor_skip_config))
            .collect();
        tensor_metas.sort_by(|a, b| {
            tensor_load_priority(&b.name).cmp(&tensor_load_priority(&a.name))
        });

        // Pass 2: concurrent processing (CPU-side parallelism)
        let tier_manager = weight_tier::WeightTierManager::from_backend(backend);

        let results: Vec<TensorProcessResult<B, E>> = tensor_metas
            .par_iter()
            .map(|meta| {
                Self::process_single_tensor(provider, backend, meta, format, &tier_manager)
                    .unwrap_or_else(|e| {
                        log::error!("failed to process tensor '{}': {}", meta.name, e);
                        TensorProcessResult::Skipped
                    })
            })
            .collect();

        // Pass 3: sequential fold into HashMaps
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        let mut meta_map = HashMap::new();
        let mut quantized = HashMap::new();
        let mut raw_floats = HashMap::new();
        let mut sparse_24 = HashMap::new();
        let mut placements = HashMap::new();

        for result in results {
            match result {
                TensorProcessResult::Native { name, meta, tensor, placement, sp_meta } => {
                    tensors.insert(name.clone(), tensor);
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name.clone(), meta);
                    placements.insert(name.clone(), placement);
                    if let Some(sp) = sp_meta {
                        sparse_24.insert(name, sp);
                    }
                }
                TensorProcessResult::RawFloat { name, meta, data } => {
                    // #11: Verify dtype consistency between TensorMeta and RawFloatTensor.
                    assert_eq!(meta.dtype, data.dtype,
                        "RawFloatTensor dtype mismatch for '{}': meta={:?}, raw={:?}",
                        name, meta.dtype, data.dtype);
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name.clone(), meta);
                    raw_floats.insert(name, data);
                }
                TensorProcessResult::Quantized { name, meta, data } => {
                    quantized.insert(name.clone(), data);
                    shapes.insert(name.clone(), meta.shape.clone());
                    meta_map.insert(name, meta);
                }
                TensorProcessResult::Skipped => {}
            }
        }

        // Log tier distribution
        let (dev_used, dev_cap) = tier_manager.usage(weight_tier::WeightTier::DeviceLocal);
        let (host_used, host_cap) = tier_manager.usage(weight_tier::WeightTier::HostLocal);
        let device_count = placements.values().filter(|p| **p == WeightPlacement::DeviceLocal).count();
        let host_count = placements.values().filter(|p| **p == WeightPlacement::HostLocal).count();
        let mmap_count = tier_manager.tensor_count() - device_count - host_count;
        log::info!(
            "upload_provider: {} tensors loaded, device={}/{}B, host={}/{}B",
            placements.len(), dev_used, dev_cap, host_used, host_cap,
        );
        if mmap_count > 0 {
            log::info!("upload_provider: {} tensors degraded to mmap", mmap_count);
        }

        // COMP12: Compress raw float weights for cold-storage tiers (SPEC 22 §6)
        let compressed_weights = if let Some(ref comp_config) = self.weight_compress_config {
            if comp_config.enabled {
                let mut comp_map = HashMap::new();
                for (name, raw) in &raw_floats {
                    let layer_idx = extract_layer_index(name);
                    // Skip hot layers (always uncompressed per SPEC §6.2)
                    if comp_config.is_hot_layer(layer_idx) {
                        continue;
                    }
                    // Classify weight type for codec selection
                    let weight_class = if comp_config.has_moe_experts > 0 {
                        let lower = name.to_ascii_lowercase();
                        if lower.contains("experts") || lower.contains("shared_expert") {
                            weight_compress::WeightClass::ExpertWeight
                        } else {
                            weight_compress::WeightClass::DenseLayerWeight
                        }
                    } else {
                        weight_compress::WeightClass::DenseLayerWeight
                    };
                    // For loading, we target HBM tier; the compressed copies serve
                    // as cold-storage seeds for later eviction. HBM tier uses None
                    // or BitPackRle (SPEC §6.2).
                    let codec = weight_compress::select_weight_codec(
                        weight_tier::WeightTier::DeviceLocal,
                        weight_class,
                        comp_config,
                        layer_idx,
                        false, // not quantized
                    );
                    if codec != crate::kv_cache::CompressionCodec::None {
                        if let Ok(Some(compressed)) =
                            weight_compress::compress_weight_page(&raw.data, codec, comp_config)
                        {
                            comp_map.insert(name.clone(), compressed);
                        }
                    }
                }
                if !comp_map.is_empty() {
                    log::info!(
                        "COMP12: compressed {} raw float weight pages for cold storage",
                        comp_map.len()
                    );
                }
                comp_map
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        Ok(WeightsHandle::new_with_compressed(
            tensors, shapes, meta_map, quantized, raw_floats, sparse_24, placements,
            compressed_weights,
        ))
    }

    /// Process a single tensor in the concurrent upload pipeline.
    fn process_single_tensor<P, B, E>(
        provider: &P,
        backend: &B,
        meta: &TensorMeta,
        format: WeightFormat,
        tier_manager: &weight_tier::WeightTierManager,
    ) -> Result<TensorProcessResult<B, E>>
    where
        P: TensorProvider + Sync,
        B: Backend<E>,
        E: Element,
    {
        // Debug: trace F16 norm tensor processing
        if meta.name.contains("attn_norm") || meta.name.contains("ffn_norm") {
            let ggml_dt = provider.ggml_dtype(&meta.name);
            log::debug!("[TENSOR-PROC] name='{}' dtype={:?} ggml_dt={:?}", meta.name, meta.dtype, ggml_dt);
        }
        // Quantized tensor — store raw bytes, no tier decision needed
        if let Some(ggml_dt) = provider.ggml_dtype(&meta.name) {
            if let Some(qt) = adapter::ggml_dtype_to_quant_type(ggml_dt) {
                let data = provider.load_tensor_data(&meta.name)?;

                // AWQ/GPTQ: repack separate qweight/scales/qzeros tensors into the
                // unified 72-byte interleaved block layout consumed by JIT QuantGemm.
                // `meta.shape` comes from `iter_tensors()` which already exposes element-level
                // [N, K] for AWQ/GPTQ base_name (see SafeTensorsLoader::iter_tensors).
                let packed_data = if let Some((scales, zeros, g_idx, _gs)) =
                    provider.awq_gptq_aux_data(&meta.name)
                {
                    let n = meta.shape.first().copied().unwrap_or(1);
                    let k = meta.shape.get(1).copied().unwrap_or(1);
                    repack_awq_gptq_blocks(&data, &scales, &zeros, g_idx.as_deref(), qt, n, k)
                } else {
                    data.into_owned()
                };

                return Ok(TensorProcessResult::Quantized {
                    name: meta.name.clone(),
                    meta: meta.clone(),
                    data: QuantizedTensor {
                        data: packed_data,
                        quant_type: qt,
                        shape: meta.shape.clone(),
                        ggml_dtype: ggml_dt,
                    },
                });
            }
        }

        // Float tensor — preserve original dtype for BF16/F16, convert F32/F64 normally
        match meta.dtype {
            Dtype::BF16 | Dtype::F16 => {
                let data = provider.load_tensor_data(&meta.name)?;
                let cloned_meta = meta.clone();
                let raw = data.into_owned();

                // ARCH-WEIGHT-NO-TRANSPOSE: Linear weights kept in original [out, in] layout.
                // GEMM lowering handles trans_b=true for HF [N,K] row-major weights.

                Ok(TensorProcessResult::RawFloat {
                    name: cloned_meta.name.clone(),
                    meta: cloned_meta.clone(),
                    data: RawFloatTensor {
                        data: raw,
                        dtype: meta.dtype,
                        shape: cloned_meta.shape.clone(),
                    },
                })
            }
            Dtype::F32 | Dtype::F64 => {
                let data = provider.load_tensor_data(&meta.name)?;
                let explicit_hint = provider.weight_layout_hint(&meta.name);
                let (cloned_meta, converted_f32, sp_meta_opt) =
                    convert_tensor_preserve_dtype(meta, data.as_ref(), format, explicit_hint)?;

                // Tier decision: DeviceLocal → HostLocal → DiskMmap
                let tensor_size = converted_f32.len() * std::mem::size_of::<f32>();
                let decision = tier_manager.decide(&cloned_meta.name, tensor_size);

                let (tensor, placement) = backend
                    .upload_weights_with_placement(converted_f32, decision.placement)
                    .map_err(|e| LoaderError::Backend(e.to_string()))?;

                Ok(TensorProcessResult::Native {
                    name: cloned_meta.name.clone(),
                    meta: cloned_meta,
                    tensor,
                    placement,
                    sp_meta: sp_meta_opt,
                })
            }
            _ => Ok(TensorProcessResult::Skipped),
        }
    }

    pub fn from_local_files_with_manifest(
        _model_id: &str,
        weight_paths: Vec<PathBuf>,
        aux_paths: Vec<PathBuf>,
        manifest: Option<&ModelManifest>,
    ) -> Result<Self> {
        let mut loader = if let Some(m) = manifest {
            Self::new(m.clone())
        } else {
            Self::new(ModelManifest::default())
        };
        loader.weight_paths = weight_paths;
        loader.detect_format();

        for path in aux_paths {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name == "config.json" {
                    loader.config_path = Some(path);
                } else if name == "tokenizer.json" {
                    loader.tokenizer_path = Some(path);
                }
            }
        }
        Ok(loader)
    }
}
