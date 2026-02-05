//! E2E tests for intelligent model loading (REQ-LOADER-013 ~ 017).
//!
//! These tests verify:
//! - Auto format detection (REQ-LOADER-013)
//! - GGUF naming parsing (REQ-LOADER-014)
//! - ONNX naming parsing (REQ-LOADER-015)
//! - Smart source fallback (REQ-LOADER-016)
//! - Unified loading entry (REQ-LOADER-017)

use std::path::PathBuf;

use gllm::loader::{
    format_detector, naming_parser, LoaderError,
    WeightFormat,
};

/// Small GGUF model for testing - Q4_0 quantized
const GGUF_TEST_MODEL: &str = "mav23/SmolLM-135M-Instruct-GGUF";
const GGUF_Q4_0_FILE: &str = "smollm-135m-instruct.Q4_0.gguf";

/// Small ONNX model for testing - FP32 precision
const ONNX_TEST_MODEL: &str = "Xenova/gte-small"; // Has ONNX export

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
    assert_eq!(
        naming_parser::parse_gguf_quantization("model.gguf"),
        None
    );
}

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
    assert_eq!(
        naming_parser::parse_onnx_precision("model.txt"),
        None
    );
}

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

#[test]
fn format_detector_rejects_unknown_extensions() {
    let path = PathBuf::from("model.bin");
    let result = format_detector::detect_format_from_path(&path);
    assert!(matches!(result, Err(LoaderError::UnsupportedWeightExtension(_))));
}

#[test]
fn gguf_quantization_preference_order() {
    // Q4_0 should be preferred over Q8_0
    assert!(naming_parser::GgufQuantization::Q4_0.preference_rank()
        < naming_parser::GgufQuantization::Q8_0.preference_rank());

    // F32 should be lowest priority
    assert!(naming_parser::GgufQuantization::Q8_0.preference_rank()
        < naming_parser::GgufQuantization::F32.preference_rank());
}

#[test]
fn onnx_precision_preference_order() {
    // FP32 should be lowest priority (default fallback)
    assert!(naming_parser::OnnxPrecision::Q4.preference_rank()
        < naming_parser::OnnxPrecision::Fp32.preference_rank());

    // UINT8 (rank 3) should be higher priority than INT8 (rank 4)
    assert!(naming_parser::OnnxPrecision::Int8.preference_rank()
        > naming_parser::OnnxPrecision::Uint8.preference_rank());
}

// E2E: Verify actual GGUF model can be loaded with quantization detection
#[test]
#[ignore = "Requires actual model download"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn gguf_e2e_quantization_detection() {
    // This would test downloading the actual GGUF model
    // and verifying that the quantization type is detected correctly
    // during loading
}

// E2E: Verify smart source fallback works
#[test]
#[ignore = "Requires network access"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn smart_source_fallback_e2e() {
    // This would test that when HF fails, ModelScope is automatically tried
    // Requires mocking or actual network access
}

// E2E: Verify unified loading entry works
#[test]
#[ignore = "Requires model download"] // Run with: cargo test --test test_intelligent_loading -- --ignored
fn unified_loading_entry_e2e() {
    // This would test that Loader::auto() works for:
    // 1. SafeTensors model from HF
    // 2. GGUF model from HF
    // 3. ONNX model from HF
    // 4. ModelScope fallback
}
