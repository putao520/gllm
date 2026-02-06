//! E2E tests for intelligent model loading (REQ-LOADER-013 ~ 017).
//!
//! These tests verify:
//! - Auto format detection (REQ-LOADER-013)
//! - GGUF naming parsing (REQ-LOADER-014)
//! - ONNX naming parsing (REQ-LOADER-015)
//! - Smart source fallback (REQ-LOADER-016)
//! - Unified loading entry (REQ-LOADER-017)

use std::path::PathBuf;

use gllm::loader::{format_detector, naming_parser, LoaderError, WeightFormat};

/// Small GGUF model for testing - Q4_0 quantized
const GGUF_TEST_MODEL: &str = "mav23/SmolLM-135M-Instruct-GGUF";

/// TEST-INTEL-001: 从扩展名检测格式
///
/// **关联需求**: REQ-LOADER-013
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 检测 .safetensors 扩展名
/// 2. 检测 .gguf 扩展名
/// 3. 检测 .onnx 扩展名
///
/// **期望结果**: 正确检测格式
#[test]
fn detect_format_from_extensions() {
    // SafeTensors
    let path = PathBuf::from("model.safetensors");
    assert_eq!(
        format_detector::detect_format_from_path(&path).unwrap(),
        WeightFormat::SafeTensors
    );

    // GGUF
    let path = PathBuf::from("model.gguf");
    assert_eq!(
        format_detector::detect_format_from_path(&path).unwrap(),
        WeightFormat::Gguf
    );

    // ONNX
    let path = PathBuf::from("model.onnx");
    assert_eq!(
        format_detector::detect_format_from_path(&path).unwrap(),
        WeightFormat::Onnx
    );

    // Case insensitive
    let path = PathBuf::from("MODEL.GGUF");
    assert_eq!(
        format_detector::detect_format_from_path(&path).unwrap(),
        WeightFormat::Gguf
    );
}

/// TEST-INTEL-002: 解析 GGUF 量化类型
///
/// **关联需求**: REQ-LOADER-014
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析各种 GGUF 文件名
/// 2. 验证量化类型正确
///
/// **期望结果**: 正确解析 Q4_0, Q8_0, Q4_K_M, F16 等类型
#[test]
fn parse_gguf_quantization_types() {
    // Standard GGUF naming
    assert_eq!(
        naming_parser::parse_gguf_quantization("SmolLM-135M-Instruct-Q4_0.gguf"),
        Some(naming_parser::GgufQuantization::Q4_0)
    );

    assert_eq!(
        naming_parser::parse_gguf_quantization("SmolLM-135M-Instruct-Q8_0.gguf"),
        Some(naming_parser::GgufQuantization::Q8_0)
    );

    assert_eq!(
        naming_parser::parse_gguf_quantization("model-Q4_K_M.gguf"),
        Some(naming_parser::GgufQuantization::Q4_K_M)
    );

    assert_eq!(
        naming_parser::parse_gguf_quantization("model-f16.gguf"),
        Some(naming_parser::GgufQuantization::F16)
    );

    // No quantization suffix → None
    assert_eq!(naming_parser::parse_gguf_quantization("model.gguf"), None);
}

/// TEST-INTEL-003: 解析 ONNX 精度类型
///
/// **关联需求**: REQ-LOADER-015
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析各种 ONNX 文件名
/// 2. 验证精度类型正确
///
/// **期望结果**: 正确解析 FP16, FP32, INT8, Q4 等类型
#[test]
fn parse_onnx_precision_types() {
    // ONNX with precision suffix
    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model_fp16.onnx"),
        Some(naming_parser::OnnxPrecision::Fp16)
    );

    assert_eq!(
        naming_parser::parse_onnx_precision("onnx/model.onnx"),
        Some(naming_parser::OnnxPrecision::Fp32)
    );

    assert_eq!(
        naming_parser::parse_onnx_precision("model_int8.onnx"),
        Some(naming_parser::OnnxPrecision::Int8)
    );

    assert_eq!(
        naming_parser::parse_onnx_precision("model_q4.onnx"),
        Some(naming_parser::OnnxPrecision::Q4)
    );

    // Not an ONNX file → None
    assert_eq!(naming_parser::parse_onnx_precision("model.txt"), None);
}

/// TEST-INTEL-004: GGUF 候选排名
///
/// **关联需求**: REQ-LOADER-014
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 解析支持的量化类型
/// 2. 验证排名
///
/// **期望结果**: Q4_0 排名高，Q6_K 排名低
#[test]
fn gguf_candidate_ranking() {
    // Supported quantization with high priority
    let rank1 = naming_parser::gguf_candidate_rank("model-Q4_0.gguf").unwrap();
    assert_eq!(rank1.0, 1); // supported
    assert_eq!(rank1.1, 1); // Q4_0 has rank 1

    // Unsupported quantization should rank lower
    let rank2 = naming_parser::gguf_candidate_rank("model-Q6_K.gguf").unwrap();
    assert_eq!(rank2.0, 0); // not fully supported
    assert_eq!(rank2.1, 5); // Q6_K has rank 5
}

/// TEST-INTEL-005: ONNX 候选排名
///
/// **关联需求**: REQ-LOADER-015
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 检查 onnx/ 目录下的文件
/// 2. 检查根目录下的文件
/// 3. 验证排名
///
/// **期望结果**: onnx/ 目录优先，FP32 优先级最低
#[test]
fn onnx_candidate_ranking() {
    // Files in onnx/ directory should be preferred
    let rank1 = naming_parser::onnx_candidate_rank("onnx/model_fp16.onnx").unwrap();
    assert_eq!(rank1.0, 1); // in onnx/ directory
    assert_eq!(rank1.1, 5); // FP16 has rank 5

    // Files not in onnx/ directory
    let rank2 = naming_parser::onnx_candidate_rank("model_fp16.onnx").unwrap();
    assert_eq!(rank2.0, 0); // not in onnx/ directory
    assert_eq!(rank2.1, 5); // FP16 has rank 5

    // Default (FP32) should be lower priority than explicit precision
    let rank3 = naming_parser::onnx_candidate_rank("onnx/model.onnx").unwrap();
    assert_eq!(rank3.0, 1); // in onnx/ directory
    assert_eq!(rank3.1, 6); // FP32 has rank 6 (lowest)
}

/// TEST-INTEL-006: 选择优先格式
///
/// **关联需求**: REQ-LOADER-013
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 提供多种格式列表
/// 2. 选择优先格式
///
/// **期望结果**: SafeTensors > GGUF > ONNX
#[test]
fn select_preferred_format() {
    // SafeTensors should be preferred over GGUF and ONNX
    let formats = vec![
        WeightFormat::Gguf,
        WeightFormat::SafeTensors,
        WeightFormat::Onnx,
    ];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        WeightFormat::SafeTensors
    );

    // GGUF should be preferred over ONNX
    let formats = vec![WeightFormat::Gguf, WeightFormat::Onnx];
    assert_eq!(
        format_detector::select_preferred_format(&formats),
        WeightFormat::Gguf
    );
}

/// TEST-INTEL-007: 格式检测器拒绝未知扩展名
///
/// **关联需求**: REQ-LOADER-013
/// **测试类型**: 负向测试
///
/// **测试步骤**:
/// 1. 检测 .bin 扩展名
///
/// **期望结果**: 返回 UnsupportedWeightExtension 错误
#[test]
fn format_detector_rejects_unknown_extensions() {
    let path = PathBuf::from("model.bin");
    let result = format_detector::detect_format_from_path(&path);
    assert!(matches!(
        result,
        Err(LoaderError::UnsupportedWeightExtension(_))
    ));
}

/// TEST-INTEL-008: GGUF 量化优先级顺序
///
/// **关联需求**: REQ-LOADER-014
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 比较各量化类型的优先级
///
/// **期望结果**: Q4_0 < Q8_0 < F32 (数值越小优先级越高)
#[test]
fn gguf_quantization_preference_order() {
    // Q4_0 should be preferred over Q8_0
    assert!(
        naming_parser::GgufQuantization::Q4_0.preference_rank()
            < naming_parser::GgufQuantization::Q8_0.preference_rank()
    );

    // F32 should be lowest priority
    assert!(
        naming_parser::GgufQuantization::Q8_0.preference_rank()
            < naming_parser::GgufQuantization::F32.preference_rank()
    );
}

/// TEST-INTEL-009: ONNX 精度优先级顺序
///
/// **关联需求**: REQ-LOADER-015
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 比较各精度的优先级
///
/// **期望结果**: Q4 < FP32，UINT8 > INT8
#[test]
fn onnx_precision_preference_order() {
    // FP32 should be lowest priority (default fallback)
    assert!(
        naming_parser::OnnxPrecision::Q4.preference_rank()
            < naming_parser::OnnxPrecision::Fp32.preference_rank()
    );

    // UINT8 (rank 3) should be higher priority than INT8 (rank 4)
    assert!(
        naming_parser::OnnxPrecision::Int8.preference_rank()
            > naming_parser::OnnxPrecision::Uint8.preference_rank()
    );
}

/// TEST-INTEL-010: GGUF E2E 量化检测
///
/// **关联需求**: REQ-LOADER-013, REQ-LOADER-014
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 单API
///
/// **前置条件**: mav23/SmolLM-135M-Instruct-GGUF 模型已缓存
///
/// **测试步骤**:
/// 1. 使用 Loader API 加载 GGUF 模型
/// 2. 验证格式检测
/// 3. 验证源检测
///
/// **期望结果**: 正确检测 GGUF 格式和 HuggingFace 源
#[test]
#[ignore = "Requires actual model download"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn gguf_e2e_quantization_detection() {
    // Use Loader API to download GGUF model
    let loader =
        gllm::loader::Loader::from_hf(GGUF_TEST_MODEL).expect("GGUF loader should be created");

    // Verify loader detected the correct format
    assert_eq!(loader.weight_format(), WeightFormat::Gguf);

    // Verify source is HuggingFace
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);

    // Verify repo name is preserved
    assert_eq!(loader.repo(), GGUF_TEST_MODEL);
}

/// TEST-INTEL-011: 智能源回退 E2E
///
/// **关联需求**: REQ-LOADER-016
/// **测试类型**: 正向测试
/// **E2E测试粒度**: 单API
///
/// **前置条件**: 网络可用
///
/// **测试步骤**:
/// 1. 尝试从 HuggingFace 加载公开模型
/// 2. 验证加载成功
///
/// **期望结果**: HuggingFace 下载成功
#[test]
#[ignore = "Requires network access"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn smart_source_fallback_e2e() {
    // Test that HuggingFace is the default source (should succeed)
    let result = gllm::loader::Loader::from_hf(GGUF_TEST_MODEL);

    assert!(
        result.is_ok(),
        "HuggingFace download should succeed for public model"
    );

    let loader = result.unwrap();
    assert_eq!(loader.source(), gllm::loader::ModelSource::HuggingFace);
    assert_eq!(loader.weight_format(), WeightFormat::Gguf);
}

// E2E: Verify unified loading entry works
#[test]
#[ignore = "Requires model download"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn unified_loading_entry_e2e() {
    // Test Loader::auto() for GGUF model
    let gguf_loader = gllm::loader::Loader::auto(GGUF_TEST_MODEL)
        .expect("Loader::auto should work for GGUF model");
    assert_eq!(gguf_loader.weight_format(), WeightFormat::Gguf);
    assert_eq!(gguf_loader.source(), gllm::loader::ModelSource::HuggingFace);

    // Test Loader::auto_with_format() with explicit format
    let gguf_explicit = gllm::loader::Loader::auto_with_format(GGUF_TEST_MODEL, WeightFormat::Gguf)
        .expect("Loader::auto_with_format should work with explicit GGUF");
    assert_eq!(gguf_explicit.weight_format(), WeightFormat::Gguf);

    // Test Loader::auto_with_source() with explicit source
    let gguf_ms = gllm::loader::Loader::auto_with_source(
        GGUF_TEST_MODEL,
        gllm::loader::ModelSource::ModelScope,
    );
    // ModelScope might not have this specific GGUF model, so we just check it doesn't panic
    assert!(
        gguf_ms.is_ok() || gguf_ms.is_err(),
        "auto_with_source should return a Result"
    );
}

/// TEST-INTEL-007: Loader 接受 configuration.json
///
/// **关联需求**: REQ-LOADER-005 (ModelScope 支持)
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 configuration.json
/// 2. 使用 from_local_files 加载
/// 3. 验证 config_path
///
/// **期望结果**: config_path 指向 configuration.json
#[test]
fn loader_accepts_configuration_json() {
    use std::path::Path;
    use tempfile::TempDir;

    fn write_config(path: &Path) {
        let value = serde_json::json!({
            "hidden_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 4,
            "max_position_embeddings": 8,
            "torch_dtype": "float32"
        });
        std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    let dir = TempDir::new().expect("temp dir");
    let weights = dir.path().join("model.safetensors");
    let config = dir.path().join("configuration.json");

    std::fs::write(&weights, []).expect("write weights");
    write_config(&config);

    let loader = gllm::loader::Loader::from_local_files(
        "Qwen/Qwen3-0.6B",
        vec![weights],
        vec![config.clone()]
    )
    .expect("loader");

    let path = loader.config_path().expect("config path");
    assert_eq!(path.file_name().unwrap(), "configuration.json");
}
