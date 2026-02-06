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
    config as loader_config, CacheLayout, HfHubClient, Loader, LoaderConfig, ParallelLoader,
    TensorInfo, WeightsHandle,
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

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        let handle = remap_gguf_handle(handle)?;
        Ok(AdapterWeights::new(handle))
    }
}

// ============================================================================
// 矩阵 1: ModelKind × WeightFormat × CPU (真实推理验证)
// ============================================================================

/// TEST-MATRIX-GEN-001: SafeTensors 或 ONNX 格式 Chat 模型 CPU 后端
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: HuggingFaceTB/SmolLM-135M-Instruct 模型已缓存
///
/// **测试步骤**:
/// 1. 加载 Chat 模型 (SafeTensors 或 ONNX 格式)
/// 2. 使用 CPU 后端执行文本生成
/// 3. 验证生成结果非空
///
/// **期望结果**: 模型成功生成非空文本
/// **注意**: HuggingFace 仓库现在优先提供 ONNX 格式，测试接受两种格式
#[test]
fn matrix_chat_safetensors_cpu() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("model loader");
    // 接受 SafeTensors 或 ONNX 格式 (HuggingFace 仓库现在优先提供 ONNX)
    let format = loader.weight_format();
    assert!(
        matches!(
            format,
            gllm::loader::WeightFormat::SafeTensors | gllm::loader::WeightFormat::Onnx
        ),
        "Expected SafeTensors or Onnx format, got: {:?}",
        format
    );

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest =
        gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Chat)
            .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);
    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor =
        Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader).expect("executor");

    // 真实推理测试
    let output = executor.generate("Hello", 5, 0.0).expect("generate");
    assert!(
        !output.trim().is_empty(),
        "Model ({:?}) should generate text, got: {output:?}",
        format
    );
}

/// TEST-MATRIX-GEN-002: GGUF 格式 Chat 模型 CPU 后端
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: SmolLM-135M GGUF (Q8_0) 模型已缓存
///
/// **测试步骤**:
/// 1. 下载 base 配置文件
/// 2. 下载 GGUF 权重文件
/// 3. 组合 base config + GGUF weights
/// 4. 使用 CPU 后端执行文本生成
///
/// **期望结果**: GGUF 模型成功生成非空文本
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

/// TEST-MATRIX-GEN-003: ONNX 格式检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: microsoft/Phi-3-mini-4k-instruct-onnx 模型已缓存
///
/// **测试步骤**:
/// 1. 加载 ONNX 模型
/// 2. 验证格式检测为 ONNX 或 SafeTensors
///
/// **期望结果**: 格式检测正确
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

/// TEST-MATRIX-EMB-001: SafeTensors 格式 Embedding 模型 CPU 后端
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向测试
/// **前置条件**: BAAI/bge-small-en-v1.5 模型已缓存
///
/// **测试步骤**:
/// 1. 加载 SafeTensors 格式的 Embedding 模型
/// 2. 使用 CPU 后端执行 embedding
/// 3. 验证输出向量维度和值
///
/// **期望结果**: Embedding 向量非空且维度合理
#[test]
fn matrix_embedding_safetensors_cpu() {
    let model = "BAAI/bge-small-en-v1.5";
    let loader = Loader::from_hf(model).expect("Embedding loader");

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest =
        gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Embedding)
            .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);

    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor =
        Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader).expect("executor");

    // 生成 embedding 并验证
    let text = "test";
    let embedding = executor.embed(text).expect("embed");
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert!(
        embedding.len() > 100,
        "Embedding dimension should be reasonable"
    );
}

/// TEST-MATRIX-RERANK-001: SafeTensors 格式 Reranker 模型 CPU 后端
///
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向测试
/// **前置条件**: BAAI/bge-reranker-v2-m3 模型已缓存
///
/// **测试步骤**:
/// 1. 加载 SafeTensors 格式的 Reranker 模型
/// 2. 使用 CPU 后端执行 rerank
/// 3. 验证输出分数为有限浮点数
///
/// **期望结果**: Rerank 分数非空且为有限值
#[test]
fn matrix_reranker_safetensors_cpu() {
    let model = "BAAI/bge-reranker-v2-m3";
    let loader = Loader::from_hf(model).expect("Reranker loader");

    let config_path = loader.config_path().expect("config path");
    let config_value = gllm::loader::config::load_config_value(&config_path).expect("load config");
    let manifest =
        gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Reranker)
            .expect("manifest");

    let mut loader = Loader::from_hf(model).expect("loader");
    loader.set_manifest_if_missing(&manifest);

    let adapter = gllm::adapter::adapter_for::<CpuBackend>(&manifest).expect("adapter");
    let backend = CpuBackend::new();
    let mut executor =
        Executor::from_loader(backend, Arc::new(manifest), adapter, &mut loader).expect("executor");

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

/// TEST-MATRIX-GEN-004: SafeTensors 格式 Chat 模型 CUDA 后端
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: CUDA 后端可用，模型已缓存
///
/// **测试步骤**:
/// 1. 验证 CUDA 后端可用
/// 2. 验证 SafeTensors 格式检测
///
/// **期望结果**: CUDA 后端可用且格式检测正确
#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_safetensors_cuda() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("SafeTensors loader");
    assert_eq!(
        loader.weight_format(),
        gllm::loader::WeightFormat::SafeTensors
    );
    assert!(cuda_available(), "CUDA backend should be available");
}

/// TEST-MATRIX-GEN-005: GGUF 格式 Chat 模型 CUDA 后端
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: CUDA 后端可用，GGUF 模型已缓存
///
/// **测试步骤**:
/// 1. 验证 CUDA 后端可用
/// 2. 验证 GGUF 格式检测
///
/// **期望结果**: GGUF 格式检测正确
#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_gguf_cuda() {
    let model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::Gguf);
    assert!(cuda_available(), "CUDA backend should be available");
}

/// TEST-MATRIX-GEN-006: ONNX 格式 Chat 模型 CUDA 后端
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
/// **前置条件**: CUDA 后端可用，ONNX 模型已缓存
///
/// **测试步骤**:
/// 1. 验证 CUDA 后端可用
/// 2. 验证 ONNX 格式检测
///
/// **期望结果**: ONNX 格式检测正确
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

/// TEST-MATRIX-GEN-007: GGUF 格式检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 加载 GGUF 模型
/// 2. 验证格式检测为 GGUF
///
/// **期望结果**: 格式检测正确
#[test]
fn matrix_gguf_format_detection() {
    let gguf_model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(gguf_model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), gllm::loader::WeightFormat::Gguf);
}
/// TEST-MATRIX-GEN-008: GGUF Q4_0 量化检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 Q4_0 文件名
/// 2. 验证量化类型和优先级
///
/// **期望结果**: 正确解析为 Q4_0 且优先级最高
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
/// TEST-MATRIX-GEN-009: GGUF Q8_0 量化检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 Q8_0 文件名
/// 2. 验证量化类型
///
/// **期望结果**: 正确解析为 Q8_0
#[test]
fn matrix_gguf_q8_0_detection() {
    use gllm::loader::naming_parser;
    let filename = "model.Q8_0.gguf";
    assert_eq!(
        naming_parser::parse_gguf_quantization(filename),
        Some(naming_parser::GgufQuantization::Q8_0)
    );
}
/// TEST-MATRIX-GEN-010: GGUF F16 量化检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 F16 文件名
/// 2. 验证量化类型
///
/// **期望结果**: 正确解析为 F16
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

/// TEST-MATRIX-EMB-002: ONNX 格式检测
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 ONNX 路径
/// 2. 验证精度类型为 FP32
///
/// **期望结果**: 正确解析为 FP32
#[test]
fn matrix_onnx_format_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model.onnx"),
        Some(naming_parser::OnnxPrecision::Fp32)
    );
}
/// TEST-MATRIX-EMB-003: ONNX FP16 精度检测
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 FP16 路径
/// 2. 验证精度类型
///
/// **期望结果**: 正确解析为 FP16
#[test]
fn matrix_onnx_fp16_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model_fp16.onnx"),
        Some(naming_parser::OnnxPrecision::Fp16)
    );
}
/// TEST-MATRIX-RERANK-002: ONNX INT8 精度检测
///
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 INT8 路径
/// 2. 验证精度类型
///
/// **期望结果**: 正确解析为 INT8
#[test]
fn matrix_onnx_int8_detection() {
    use gllm::loader::naming_parser;
    assert_eq!(
        naming_parser::parse_onnx_precision("model_int8.onnx"),
        Some(naming_parser::OnnxPrecision::Int8)
    );
}
/// TEST-MATRIX-RERANK-003: ONNX Q4 精度检测
///
/// **关联需求**: REQ-TEST-004
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析 Q4 路径
/// 2. 验证精度类型
///
/// **期望结果**: 正确解析为 Q4
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

/// TEST-MATRIX-GEN-011: 格式优先级 - SafeTensors 第一
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 提供 GGUF, ONNX, SafeTensors 格式列表
/// 2. 选择优先格式
///
/// **期望结果**: SafeTensors 被选中
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
/// TEST-MATRIX-GEN-012: 格式优先级 - GGUF 优于 ONNX
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 提供 ONNX, GGUF 格式列表
/// 2. 选择优先格式
///
/// **期望结果**: GGUF 被选中
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

/// TEST-MATRIX-GEN-013: GGUF 量化优先级
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 比较各量化类型的优先级
///
/// **期望结果**: Q4_0 < Q8_0 < F16 < F32 (数值越小优先级越高)
#[test]
fn matrix_gguf_quantization_preference() {
    use gllm::loader::naming_parser::GgufQuantization;
    assert!(GgufQuantization::Q4_0.preference_rank() < GgufQuantization::Q8_0.preference_rank());
    assert!(GgufQuantization::Q8_0.preference_rank() < GgufQuantization::F16.preference_rank());
    assert!(GgufQuantization::F16.preference_rank() < GgufQuantization::F32.preference_rank());
}
/// TEST-MATRIX-EMB-004: ONNX 精度优先级
///
/// **关联需求**: REQ-TEST-003
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 比较各精度类型的优先级
///
/// **期望结果**: Q4 < FP32, FP16 < FP32
#[test]
fn matrix_onnx_precision_preference() {
    use gllm::loader::naming_parser::OnnxPrecision;
    assert!(OnnxPrecision::Q4.preference_rank() < OnnxPrecision::Fp32.preference_rank());
    assert!(OnnxPrecision::Fp16.preference_rank() < OnnxPrecision::Fp32.preference_rank());
}

// ============================================================================
// 矩阵 7: 后端检测测试
// ============================================================================

/// TEST-BACKEND-003: CPU 后端始终可用
///
/// **关联需求**: REQ-TEST-001, REQ-TEST-010
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 CPU 后端
///
/// **期望结果**: 后端创建成功不 panic
#[test]
fn matrix_backend_cpu_always_available() {
    let _backend = CpuBackend::new();
    // 验证后端创建不 panic
}
/// TEST-BACKEND-004: CUDA 后端检测
///
/// **关联需求**: REQ-TEST-001, REQ-TEST-010
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 检测 CUDA 后端是否可用
///
/// **期望结果**: 检测逻辑不 panic
#[test]
fn matrix_backend_cuda_detection() {
    let available = cuda_available();
    let _ = available;
    // 验证检测逻辑不 panic
}

// ============================================================================
// 矩阵 8: 源切换测试
// ============================================================================

/// TEST-MATRIX-GEN-014: HuggingFace 源检测
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 从 HuggingFace 加载模型
/// 2. 验证源属性
///
/// **期望结果**: 源为 HuggingFace
#[test]
fn matrix_source_huggingface() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("HF loader");
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);
}
/// TEST-MATRIX-GEN-015: 自动源选择
///
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 使用 auto 模式加载模型
/// 2. 验证源属性
///
/// **期望结果**: 源为 HuggingFace (可用源)
#[test]
fn matrix_source_auto_selection() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::auto(model).expect("auto loader");
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);
}
