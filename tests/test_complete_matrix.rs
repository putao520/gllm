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
//! - 包含网络/CUDA 测试: cargo test --test test_complete_matrix -- --include-ignored

use gllm::loader::{Loader, WeightFormat, ModelSource};
use gllm::manifest::ModelKind;
use gllm_kernels::{CpuBackend, CudaBackend};

// ============================================================================
// 辅助函数
// ============================================================================

/// 检测 CUDA 是否可用
fn cuda_available() -> bool {
    CudaBackend::new(0).is_ok()
}

// ============================================================================
// 矩阵 1: ModelKind × WeightFormat × CPU
// ============================================================================

#[test]
fn matrix_chat_safetensors_cpu() {
    // 使用 HuggingFaceTB/SmolLM-135M-Instruct，明确指定 SafeTensors 格式
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::auto_with_format(model, WeightFormat::SafeTensors)
        .expect("HF loader should succeed with SafeTensors format");
    assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);

    // 验证 manifest 创建
    if let Some(config_path) = loader.config_path() {
        let config_value = gllm::loader::config::load_config_value(config_path)
            .expect("load config");
        let manifest = gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Chat)
            .expect("create manifest");
        assert_eq!(manifest.kind, ModelKind::Chat);
    }
}

#[test]
fn matrix_chat_gguf_cpu() {
    let model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), WeightFormat::Gguf);
}

#[test]
fn matrix_embedding_safetensors_cpu() {
    // 使用公开的 Embedding 模型，明确指定 SafeTensors 格式
    let model = "sentence-transformers/all-MiniLM-L6-v2";
    let loader = Loader::auto_with_format(model, WeightFormat::SafeTensors)
        .expect("HF loader should succeed for public model");
    assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);

    // 验证 manifest 类型
    if let Some(config_path) = loader.config_path() {
        let config_value = gllm::loader::config::load_config_value(config_path)
            .expect("load config");
        let manifest = gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Embedding)
            .expect("create manifest");
        assert_eq!(manifest.kind, ModelKind::Embedding);
    }
}

#[test]
fn matrix_reranker_safetensors_cpu() {
    // 使用公开的 Reranker 模型，明确指定 SafeTensors 格式
    let model = "BAAI/bge-reranker-v2-m3";
    let loader = Loader::auto_with_format(model, WeightFormat::SafeTensors)
        .expect("HF loader should succeed for public model");
    assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);

    // 验证 manifest 类型
    if let Some(config_path) = loader.config_path() {
        let config_value = gllm::loader::config::load_config_value(config_path)
            .expect("load config");
        let manifest = gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Reranker)
            .expect("create manifest");
        assert_eq!(manifest.kind, ModelKind::Reranker);
    }
}

// ============================================================================
// 矩阵 2: ModelKind × CUDA (条件测试)
// ============================================================================

#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_chat_safetensors_cuda() {
    // 使用 SmolLM，明确指定 SafeTensors 格式
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::auto_with_format(model, WeightFormat::SafeTensors)
        .expect("HF loader should succeed");
    assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);
    assert!(cuda_available(), "CUDA backend should be available");

    // 验证 manifest
    if let Some(config_path) = loader.config_path() {
        let config_value = gllm::loader::config::load_config_value(config_path)
            .expect("load config");
        let manifest = gllm::loader::config::manifest_from_config(model, &config_value, ModelKind::Chat)
            .expect("create manifest");
        assert_eq!(manifest.kind, ModelKind::Chat);
    }
}

#[test]
#[ignore = "Requires CUDA backend"]
fn matrix_embedding_safetensors_cuda() {
    let model = "sentence-transformers/all-MiniLM-L6-v2";
    let loader = Loader::auto_with_format(model, WeightFormat::SafeTensors)
        .expect("HF loader should succeed");
    assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);
    assert!(cuda_available(), "CUDA backend should be available");
}

// ============================================================================
// 矩阵 3: GGUF 量化类型测试
// ============================================================================

#[test]
fn matrix_gguf_format_detection() {
    // 测试 GGUF 格式自动检测
    let gguf_model = "mav23/SmolLM-135M-Instruct-GGUF";
    let loader = Loader::from_hf(gguf_model).expect("GGUF loader");
    assert_eq!(loader.weight_format(), WeightFormat::Gguf);
}

#[test]
fn matrix_gguf_q4_0_detection() {
    // 测试 Q4_0 量化类型检测
    use gllm::loader::naming_parser;

    let filename = "smollm-135m-instruct.Q4_0.gguf";
    let quant = naming_parser::parse_gguf_quantization(filename);
    assert_eq!(quant, Some(naming_parser::GgufQuantization::Q4_0));

    // Q4_0 应该是优先级最高的支持的量化类型
    let rank = naming_parser::gguf_candidate_rank(filename).unwrap();
    assert_eq!(rank.0, 1); // supported
    assert_eq!(rank.1, 1); // highest priority
}

#[test]
fn matrix_gguf_q8_0_detection() {
    use gllm::loader::naming_parser;

    let filename = "model.Q8_0.gguf";
    let quant = naming_parser::parse_gguf_quantization(filename);
    assert_eq!(quant, Some(naming_parser::GgufQuantization::Q8_0));
}

#[test]
fn matrix_gguf_f16_detection() {
    use gllm::loader::naming_parser;

    let filename = "model.f16.gguf";
    let quant = naming_parser::parse_gguf_quantization(filename);
    assert_eq!(quant, Some(naming_parser::GgufQuantization::F16));
}

// ============================================================================
// 矩阵 4: ONNX 精度类型测试
// ============================================================================

#[test]
fn matrix_onnx_format_detection() {
    use gllm::loader::naming_parser;

    // ONNX 格式检测
    let filename = "onnx/model.onnx";
    let precision = naming_parser::parse_onnx_precision(filename);
    assert_eq!(precision, Some(naming_parser::OnnxPrecision::Fp32));
}

#[test]
fn matrix_onnx_fp16_detection() {
    use gllm::loader::naming_parser;

    let filename = "onnx/model_fp16.onnx";
    let precision = naming_parser::parse_onnx_precision(filename);
    assert_eq!(precision, Some(naming_parser::OnnxPrecision::Fp16));
}

#[test]
fn matrix_onnx_int8_detection() {
    use gllm::loader::naming_parser;

    let filename = "model_int8.onnx";
    let precision = naming_parser::parse_onnx_precision(filename);
    assert_eq!(precision, Some(naming_parser::OnnxPrecision::Int8));
}

#[test]
fn matrix_onnx_q4_detection() {
    use gllm::loader::naming_parser;

    let filename = "model_q4.onnx";
    let precision = naming_parser::parse_onnx_precision(filename);
    assert_eq!(precision, Some(naming_parser::OnnxPrecision::Q4));
}

// ============================================================================
// 矩阵 5: 格式优先级测试
// ============================================================================

#[test]
fn matrix_format_preference_safe_tensors_first() {
    use gllm::loader::format_detector;

    // SafeTensors 应该是最高优先级
    let formats = vec![
        WeightFormat::Gguf,
        WeightFormat::Onnx,
        WeightFormat::SafeTensors,
    ];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        WeightFormat::SafeTensors
    );
}

#[test]
fn matrix_format_preference_gguf_over_onnx() {
    use gllm::loader::format_detector;

    // GGUF 应该优先于 ONNX
    let formats = vec![WeightFormat::Onnx, WeightFormat::Gguf];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        WeightFormat::Gguf
    );
}

// ============================================================================
// 矩阵 6: 量化优先级测试
// ============================================================================

#[test]
fn matrix_gguf_quantization_preference() {
    use gllm::loader::naming_parser::GgufQuantization;

    // Q4_0 > Q8_0 > F16 > F32 (preference rank 越小越好)
    assert!(GgufQuantization::Q4_0.preference_rank() < GgufQuantization::Q8_0.preference_rank());
    assert!(GgufQuantization::Q8_0.preference_rank() < GgufQuantization::F16.preference_rank());
    assert!(GgufQuantization::F16.preference_rank() < GgufQuantization::F32.preference_rank());
}

#[test]
fn matrix_onnx_precision_preference() {
    use gllm::loader::naming_parser::OnnxPrecision;

    // Q4 > Int8 > Uint8 > FP16 > FP32 (preference rank 越小越好)
    assert!(OnnxPrecision::Q4.preference_rank() < OnnxPrecision::Int8.preference_rank());
    assert!(OnnxPrecision::Int8.preference_rank() < OnnxPrecision::Fp16.preference_rank());
    assert!(OnnxPrecision::Fp16.preference_rank() < OnnxPrecision::Fp32.preference_rank());
}

// ============================================================================
// 矩阵 7: 后端检测测试
// ============================================================================

#[test]
fn matrix_backend_cpu_always_available() {
    // CPU 后端应该始终可用
    let _backend = CpuBackend::new();
    // 验证后端创建不 panic
}

#[test]
fn matrix_backend_cuda_detection() {
    // CUDA 可用性检测
    let available = cuda_available();
    // 只验证检测逻辑不 panic，不要求 CUDA 必须存在
    let _ = available;
}

// ============================================================================
// 矩阵 8: 源切换测试
// ============================================================================

#[test]
fn matrix_source_huggingface() {
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::from_hf(model).expect("HF loader");
    assert_eq!(loader.source(), ModelSource::HuggingFace);
}

#[test]
fn matrix_source_auto_selection() {
    // auto() 应该选择 HuggingFace 作为默认源
    let model = "HuggingFaceTB/SmolLM-135M-Instruct";
    let loader = Loader::auto(model).expect("auto loader");
    assert_eq!(loader.source(), ModelSource::HuggingFace);
}
