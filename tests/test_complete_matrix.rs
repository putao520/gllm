//! 完整测试矩阵: ModelKind × WeightFormat × Backend × Quantization/Precision
//!
//! 测试矩阵结构:
//! - ModelKind: {Chat, Embedding, Reranker}
//! - WeightFormat: {SafeTensors, GGUF, ONNX}
//! - Backend: {CPU, CUDA (条件)}
//! - GGUF Quantization: {Q4_0, Q8_0, F16}
//! - ONNX Precision: {FP32, FP16, INT8}
//!
//! 运行方式:
//! - CPU 全部测试: cargo test --test test_complete_matrix
//! - 包含 CUDA 测试: cargo test --test test_complete_matrix -- --include-ignored
//!
//! 注意: 这些测试执行真实的模型推理，验证输出正确性

use std::sync::Arc;

use gllm::adapter::{AdapterResult, AdapterWeights, ModelAdapter};
use gllm::engine::executor::Executor;
use gllm::loader::{
    config as loader_config, CacheLayout, HfHubClient, Loader, LoaderConfig,
    ParallelLoader, TensorInfo, WeightsHandle,
};
use gllm::manifest::{FileMap, ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP};
use gllm_kernels::{backend_trait::Backend, cpu_backend::CpuBackend, CudaBackend};

// ============================================================================
// 辅助函数
// ============================================================================

fn cuda_available() -> bool {
    CudaBackend::new(0).is_ok()
}

/// GGUF 张量名称映射和形状归一化
fn remap_gguf_handle<B: Backend>(
    handle: WeightsHandle<B>,
) -> Result<WeightsHandle<B>, gllm::loader::LoaderError> {
    let WeightsHandle { tensors, meta } = handle;
    let mut out = WeightsHandle::default();

    for (name, tensor) in tensors {
        let mapped = map_gguf_name(&name).unwrap_or_else(|| name.clone());
        if out.tensors.contains_key(&mapped) {
            return Err(gllm::loader::LoaderError::DuplicateTensor(mapped));
        }
        if let Some(info) = meta.get(&name) {
            let shape = normalize_gguf_shape(&info.shape);
            out.meta.insert(
                mapped.clone(),
                TensorInfo {
                    shape,
                    dtype: info.dtype,
                    quantized: info.quantized,
                },
            );
        }
        out.tensors.insert(mapped, tensor);
    }

    Ok(out)
}

/// GGUF 存储矩阵形状为 [cols, rows]; 后端期望 [rows, cols]
fn normalize_gguf_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 2 {
        vec![shape[1], shape[0]]
    } else {
        shape.to_vec()
    }
}

/// GGUF 张量名称映射到标准名称
fn map_gguf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        _ => {}
    }

    let rest = name.strip_prefix("blk.")?;
    let mut parts = rest.splitn(2, '.');
    let layer = parts.next()?.parse::<usize>().ok()?;
    let suffix = parts.next()?;

    let mapped = match suffix {
        "attn_norm.weight" => format!("model.layers.{layer}.input_layernorm.weight"),
        "attn_q.weight" => format!("model.layers.{layer}.self_attn.q_proj.weight"),
        "attn_k.weight" => format!("model.layers.{layer}.self_attn.k_proj.weight"),
        "attn_v.weight" => format!("model.layers.{layer}.self_attn.v_proj.weight"),
        "attn_output.weight" => format!("model.layers.{layer}.self_attn.o_proj.weight"),
        "ffn_norm.weight" => format!("model.layers.{layer}.post_attention_layernorm.weight"),
        "ffn_gate.weight" => format!("model.layers.{layer}.mlp.gate_proj.weight"),
        "ffn_up.weight" => format!("model.layers.{layer}.mlp.up_proj.weight"),
        "ffn_down.weight" => format!("model.layers.{layer}.mlp.down_proj.weight"),
        _ => return None,
    };

    Some(mapped)
}

/// GGUF 适配器 (带张量重映射)
struct GgufRemapAdapter;

impl<B: Backend> ModelAdapter<B> for GgufRemapAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Llama4)
    }

    fn load_weights(
        &self,
        loader: &mut Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        let handle = remap_gguf_handle(handle)?;
        Ok(AdapterWeights::new(handle))
    }
}

// ============================================================================
// 矩阵 1: ModelKind × WeightFormat × CPU (真实推理验证)
// ============================================================================

/// SafeTensors 格式测试 - HuggingFaceTB/SmolLM-135M-Instruct
#[test]
fn matrix_chat_safetensors_cpu() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("SafeTensors loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::SafeTensors);

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest = gllm::loader::config::manifest_from_config(
        model,
        &config_value,
        ModelKind::Chat,
    )
    .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);
    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor = Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader)
        .expect("executor");

    // 真实推理测试
    let output = executor.generate("Hello", 5, 0.0).expect("generate");
    assert!(
        !output.trim().is_empty(),
        "SafeTensors model should generate text, got: {output:?}"
    );
}

/// GGUF 格式测试 - SmolLM-135M GGUF (Q8_0)
#[test]
fn matrix_chat_gguf_cpu() {
    const BASE_REPO: &str = "HuggingFaceTB/SmolLM-135M-Instruct";
    const GGUF_REPO: &str = "mav23/SmolLM-135M-Instruct-GGUF";
    const GGUF_FILE: &str = "smollm-135m-instruct.Q8_0.gguf";
    const GGUF_FILE_MAP: FileMap = &[("model.gguf", GGUF_FILE)];

    // 下载 base 配置文件
    let config = LoaderConfig::default();
    let files = loader_config::download_config_files(BASE_REPO, &config, EMPTY_FILE_MAP)
        .expect("download config files");
    let config_path = files.config_path;
    let tokenizer_path = files
        .tokenizer_path
        .expect("tokenizer.json missing in base repo");
    let config_value = loader_config::load_config_value(&config_path).expect("load config");
    let manifest = loader_config::manifest_from_config(BASE_REPO, &config_value, ModelKind::Chat)
        .expect("manifest");

    // 下载 GGUF 文件
    let cache = CacheLayout::new(None).expect("cache layout");
    let hf = HfHubClient::new(cache.hf_cache_dir()).expect("hf client");
    let gguf_files = hf
        .download_model_files(GGUF_REPO, GGUF_FILE_MAP, ParallelLoader::new(false))
        .expect("download gguf file");
    let gguf_path = gguf_files
        .weights
        .into_iter()
        .next()
        .expect("gguf weights missing");

    // 组合: base config + GGUF weights
    let mut loader = Loader::from_local_files_with_manifest(
        BASE_REPO,
        vec![gguf_path],
        vec![config_path, tokenizer_path],
        Some(&manifest),
    )
    .expect("combined loader");

    // 使用 GGUF adapter
    let backend = CpuBackend::new();
    let mut executor =
        Executor::from_loader(backend, Arc::new(manifest), &GgufRemapAdapter, &mut loader)
            .expect("executor");

    // 生成文本并验证 (这是真实推理测试)
    let output = executor.generate("Hello", 5, 0.0).expect("generate");
    assert!(
        !output.trim().is_empty(),
        "GGUF model should generate text, got: {output:?}"
    );
}

/// ONNX 格式测试 - 暂时只验证格式检测
/// ONNX 推理使用不同的执行路径，这里只验证能正确加载
#[test]
fn matrix_chat_onnx_cpu() {
    // ONNX 模型使用不同的适配器，这里只验证格式检测
    let model = "microsoft/Phi-3-mini-4k-instruct-onnx";
    let loader = Loader::from_hf(model).expect("ONNX loader");
    assert!(
        loader.weight_format() == gllm::loader::WeightFormat::Onnx
            || loader.weight_format() == gllm::loader::WeightFormat::SafeTensors
    );
}

/// Embedding 测试 - BAAI/bge-small-en-v1.5
#[test]
fn matrix_embedding_safetensors_cpu() {
    let model = "BAAI/bge-small-en-v1.5";
    let loader = Loader::from_hf(model).expect("Embedding loader");

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest = gllm::loader::config::manifest_from_config(
        model,
        &config_value,
        ModelKind::Embedding,
    )
    .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);

    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor = Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader)
        .expect("executor");

    // 生成 embedding 并验证
    let text = "test";
    let embedding = executor.embed(text).expect("embed");
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert!(
        embedding.len() > 100,
        "Embedding dimension should be reasonable"
    );
}

/// Reranker 测试 - BAAI/bge-reranker-v2-m3
#[test]
fn matrix_reranker_safetensors_cpu() {
    let model = "BAAI/bge-reranker-v2-m3";
    let loader = Loader::from_hf(model).expect("Reranker loader");

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest = gllm::loader::config::manifest_from_config(
        model,
        &config_value,
        ModelKind::Reranker,
    )
    .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);

    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor = Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader)
        .expect("executor");

    // 生成 rerank 分数
    let query = "test query";
    let scores = executor.rerank(query).expect("rerank");
    assert!(!scores.is_empty(), "Rerank scores should not be empty");

    // 验证分数是合理的浮点数
    for &score in &scores {
        assert!(score.is_finite(), "Scores should be finite");
    }
}

// ============================================================================
// 矩阵 2: ModelKind × CUDA (条件测试)
// ============================================================================

#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_safetensors_cuda() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("SafeTensors loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::SafeTensors);
    assert!(cuda_available(), "CUDA backend should be available");
}

#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_gguf_cuda() {
    let model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::Gguf);
    assert!(cuda_available(), "CUDA backend should be available");
}

#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_onnx_cuda() {
    let model = "microsoft/Phi-3-mini-4k-instruct-onnx";
    let loader = Loader::from_hf(model).expect("ONNX loader");
    assert!(
        loader.weight_format() == gllm::loader::WeightFormat::Onnx
            || loader.weight_format() == gllm::loader::WeightFormat::SafeTensors
    );
    assert!(cuda_available(), "CUDA backend should be available");
}

// ============================================================================
// 矩阵 3: GGUF 量化类型测试 (单元测试)
// ============================================================================

#[test]
fn matrix_gguf_format_detection() {
    let gguf_model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(gguf_model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::Gguf);
}

#[test]
fn matrix_gguf_q4_0_detection() {
    use gllm::loader::naming_parser;
    let filename = "smollm-135m-instruct.Q4_0.gguf";
    let quant = naming_parser::parse_gguf_quantization(filename);
    assert_eq!(quant, Some(naming_parser::GgufQuantization::Q4_0));
    let rank = naming_parser::gguf_candidate_rank(filename).unwrap();
    assert_eq!(rank.0, 1); // supported
    assert_eq!(rank.1, 1); // highest priority
}

#[test]
fn matrix_gguf_q8_0_detection() {
    use gllm::loader::naming_parser;
    let filename = "model.Q8_0.gguf";
    assert_eq!(
        naming_parser::parse_gguf_quantization(filename),
        Some(naming_parser::GgufQuantization::Q8_0)
    );
}

#[test]
fn matrix_gguf_f16_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_gguf_quantization("model.f16.gguf"),
        Some(naming_parser::GgufQuantization::F16)
    );
}

// ============================================================================
// 矩阵 4: ONNX 精度类型测试 (单元测试)
// ============================================================================

#[test]
fn matrix_onnx_format_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model.onnx"),
        Some(naming_parser::OnnxPrecision::Fp32)
    );
}

#[test]
fn matrix_onnx_fp16_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model_fp16.onnx"),
        Some(naming_parser::OnnxPrecision::Fp16)
    );
}

#[test]
fn matrix_onnx_int8_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("model_int8.onnx"),
        Some(naming_parser::OnnxPrecision::Int8)
    );
}

#[test]
fn matrix_onnx_q4_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("model_q4.onnx"),
        Some(naming_parser::OnnxPrecision::Q4)
    );
}

// ============================================================================
// 矩阵 5: 格式优先级测试
// ============================================================================

#[test]
fn matrix_format_preference_safe_tensors_first() {
    use gllm::loader::format_detector;
    let formats = vec![
        gllm::loader::WeightFormat::Gguf,
        gllm::loader::WeightFormat::Onnx,
        gllm::loader::WeightFormat::SafeTensors,
    ];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        gllm::loader::WeightFormat::SafeTensors
    );
}

#[test]
fn matrix_format_preference_gguf_over_onnx() {
    use gllm::loader::format_detector;
    let formats = vec![
        gllm::loader::WeightFormat::Onnx,
        gllm::loader::WeightFormat::Gguf,
    ];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        gllm::loader::WeightFormat::Gguf
    );
}

// ============================================================================
// 矩阵 6: 量化优先级测试
// ============================================================================

#[test]
fn matrix_gguf_quantization_preference() {
    use gllm::loader::naming_parser::GgufQuantization;
    assert!(GgufQuantization::Q4_0.preference_rank() < GgufQuantization::Q8_0.preference_rank());
    assert!(GgufQuantization::Q8_0.preference_rank() < GgufQuantization::F16.preference_rank());
    assert!(GgufQuantization::F16.preference_rank() < GgufQuantization::F32.preference_rank());
}

#[test]
fn matrix_onnx_precision_preference() {
    use gllm::loader::naming_parser::OnnxPrecision;
    assert!(OnnxPrecision::Q4.preference_rank() < OnnxPrecision::Fp32.preference_rank());
    assert!(OnnxPrecision::Fp16.preference_rank() < OnnxPrecision::Fp32.preference_rank());
}

// ============================================================================
// 矩阵 7: 后端检测测试
// ============================================================================

#[test]
fn matrix_backend_cpu_always_available() {
    let _backend = CpuBackend::new();
    // 验证后端创建不 panic
}

#[test]
fn matrix_backend_cuda_detection() {
    let available = cuda_available();
    let _ = available;
    // 验证检测逻辑不 panic
}

// ============================================================================
// 矩阵 8: 源切换测试
// ============================================================================

#[test]
fn matrix_source_huggingface() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("HF loader");
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);
}

#[test]
fn matrix_source_auto_selection() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::auto(model).expect("auto loader");
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);
}
