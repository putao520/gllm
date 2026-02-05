use std::collections::HashMap;

use gllm::loader::{Loader, UploadedTensor};
use gllm::quantization::{dequantize_int8_with_zero, BlockQuantization};
use gllm_kernels::cpu_backend::CpuBackend;
use half::f16;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use tempfile::TempDir;

fn f16_bytes(values: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    out
}

/// TEST-QUANT-001: GPTQ signed int4 反量化
///
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 packed int4 权重文件
/// 2. 加载并反量化
/// 3. 验证反量化结果
///
/// **期望结果**: 正确反量化为 [-1.0, 0.5]
#[test]
fn gptq_signed_int4_dequantizes_with_metadata_bits() {
    let dir = TempDir::new().expect("temp dir");
    let weights_path = dir.path().join("model.safetensors");

    // Packed int4 values: [1, -2] after signed decode.
    let qweight = vec![0b0001_1110u8];
    let scales = vec![f16::from_f32(0.5)];
    let zeros = vec![0u8];
    let scale_bytes = f16_bytes(&scales);

    let qweight_view = TensorView::new(Dtype::U8, vec![1, 1], &qweight).expect("qweight view");
    let scales_view = TensorView::new(Dtype::F16, vec![1], &scale_bytes).expect("scales view");
    let zeros_view = TensorView::new(Dtype::U8, vec![1], &zeros).expect("zeros view");

    let mut metadata = HashMap::new();
    metadata.insert(
        "gllm.packed_bits".to_string(),
        serde_json::json!({"linear.qweight_s4": 4}).to_string(),
    );

    serialize_to_file(
        vec![
            ("linear.qweight_s4".to_string(), qweight_view),
            ("linear.scales_s4".to_string(), scales_view),
            ("linear.qzeros_s4".to_string(), zeros_view),
        ],
        &Some(metadata),
        &weights_path,
    )
    .expect("serialize safetensors");

    let mut loader =
        Loader::from_local_files("microsoft/Phi-4-mini-instruct", vec![weights_path], vec![])
            .expect("loader");
    let backend = CpuBackend::new();
    let handle = loader.upload_weights(&backend).expect("upload weights");

    let tensor = handle
        .get("linear.weight_s4")
        .expect("linear.weight_s4 missing");
    match tensor {
        UploadedTensor::F32(values) => {
            // Signed int4 unpacked as [-2, 1] -> scaled by 0.5 => [-1.0, 0.5]
            assert_eq!(values.as_slice(), &[-1.0, 0.5]);
        }
    }
}

/// TEST-QUANT-002: SmoothQuant 块缩放减少激活范围
///
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 BlockQuantization
/// 2. 缩放激活值
/// 3. 验证峰值被裁剪
///
/// **期望结果**: 缩放后峰值 <= 4.0
#[test]
fn smoothquant_block_scales_reduce_activation_range() {
    let quant = BlockQuantization::new(2, vec![f16::from_f32(0.5), f16::from_f32(0.25)]);
    let activations = vec![4.0f32, 2.0, 8.0, 4.0];
    let mut scaled = Vec::with_capacity(activations.len());
    for (idx, value) in activations.iter().enumerate() {
        let block = idx / 2;
        scaled.push(value * quant.scale_for_block(block));
    }
    let peak = scaled.iter().copied().fold(f32::MIN, f32::max);
    assert!(peak <= 4.0, "expected scaled activations to be clipped");
    assert_eq!(scaled[0], 2.0);
}

/// TEST-QUANT-003: 动态 INT8 量化往返误差容忍
///
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 量化浮点数为 INT8
/// 2. 反量化回浮点数
/// 3. 验证误差 < 0.05
///
/// **期望结果**: 误差在容忍范围内
#[test]
fn dynamic_int8_quantization_round_trips_within_tolerance() {
    let source = [0.5f32, -1.5, 3.25, -0.75];
    let max = source
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let encode_scale = 127.0 / max;
    let decode_scale = 1.0 / encode_scale;

    let mut quantized = Vec::with_capacity(source.len());
    for &value in source.iter() {
        let clipped = (value * encode_scale).round().clamp(-127.0, 127.0);
        quantized.push(clipped as i8);
    }

    let recovered = dequantize_int8_with_zero(&quantized, decode_scale, 0.0);
    for (orig, deq) in source.iter().zip(recovered.iter()) {
        assert!(
            (orig - deq).abs() < 0.05,
            "dynamic quantization drift too large: {orig} vs {deq}"
        );
    }
}
