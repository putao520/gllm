use gllm::loader::Loader;
use gllm::loader::UploadedTensor;
use gllm_kernels::cpu_backend::CpuBackend;
use half::f16;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use std::collections::HashMap;
use tempfile::TempDir;

fn f16_bytes(values: &[f16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    out
}

#[test]
fn quantized_awq_like_weights_are_dequantized() {
    let dir = TempDir::new().expect("temp dir");
    let weights_path = dir.path().join("model.safetensors");

    let qweight = vec![0x21u8, 0x43u8];
    let scales = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let zeros = vec![0u8, 1u8];

    let qweight_view = TensorView::new(Dtype::U8, vec![1, 2], &qweight).expect("qweight view");
    let scales_bytes = f16_bytes(&scales);
    let scales_view = TensorView::new(Dtype::F16, vec![2], &scales_bytes).expect("scales view");
    let zeros_view = TensorView::new(Dtype::U8, vec![2], &zeros).expect("zeros view");

    let mut metadata = HashMap::new();
    metadata.insert(
        "gllm.packed_bits".to_string(),
        serde_json::json!({"linear.qweight": 4}).to_string(),
    );

    serialize_to_file(
        vec![
            ("linear.qweight".to_string(), qweight_view),
            ("linear.scales".to_string(), scales_view),
            ("linear.qzeros".to_string(), zeros_view),
        ],
        &Some(metadata),
        &weights_path,
    )
    .expect("serialize safetensors");

    let loader =
        Loader::from_local_files("Qwen/Qwen3-0.6B", vec![weights_path], vec![]).expect("loader");
    let mut loader = loader;
    let backend = CpuBackend::new();
    let handle = loader.upload_weights(&backend).expect("upload weights");

    let tensor = handle.get("linear.weight").expect("linear.weight missing");
    match tensor {
        UploadedTensor::F32(values) => {
            assert_eq!(values.as_slice(), &[1.0, 2.0, 4.0, 6.0]);
        }
    }

    assert!(handle.get("linear.qweight").is_none());
}
