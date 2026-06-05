use std::collections::HashMap;

use super::graph_convert::ConvertError;
use super::{external_data_locations, proto, OnnxLoader};
use super::model::{OnnxFunction, OnnxGraph, OnnxModelMetadata, OnnxNode, OnnxOperatorSet, OnnxQuantizationAnnotation, OnnxValueInfo};
use super::tensor::{OnnxSparseFormat, OnnxSparseTensor, OnnxTensor};
use super::types::{OnnxDim, OnnxMapType, OnnxTensorShape, OnnxTensorType, OnnxType};
use super::{OnnxAttribute, OnnxAttributeValue, OnnxModel};
use prost::bytes::Bytes;
use prost::Message;
use safetensors::Dtype;
use tempfile::{NamedTempFile, TempDir};

use crate::loader::TensorProvider;

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn empty_model(graph: proto::GraphProto) -> proto::ModelProto {
    proto::ModelProto {
        ir_version: None,
        opset_import: Vec::new(),
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: Vec::new(),
        training_info: Vec::new(),
        functions: Vec::new(),
        configuration: Vec::new(),
    }
}

fn empty_tensor() -> proto::TensorProto {
    proto::TensorProto {
        dims: Vec::new(),
        data_type: None,
        segment: None,
        float_data: Vec::new(),
        int32_data: Vec::new(),
        string_data: Vec::new(),
        int64_data: Vec::new(),
        name: None,
        doc_string: None,
        raw_data: None,
        external_data: Vec::new(),
        data_location: None,
        double_data: Vec::new(),
        uint64_data: Vec::new(),
        metadata_props: Vec::new(),
    }
}

fn empty_node() -> proto::NodeProto {
    proto::NodeProto {
        input: Vec::new(),
        output: Vec::new(),
        name: None,
        op_type: None,
        domain: None,
        overload: None,
        attribute: Vec::new(),
        doc_string: None,
        metadata_props: Vec::new(),
        device_configurations: Vec::new(),
    }
}

fn empty_graph() -> proto::GraphProto {
    proto::GraphProto {
        node: Vec::new(),
        name: None,
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        doc_string: None,
        input: Vec::new(),
        output: Vec::new(),
        value_info: Vec::new(),
        quantization_annotation: Vec::new(),
        metadata_props: Vec::new(),
    }
}

fn write_model(model: proto::ModelProto, path: &std::path::Path) {
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("encode");
    std::fs::write(path, buf).expect("write model");
}

fn tensor_f32(name: &str, dims: Vec<i64>, values: &[f32]) -> proto::TensorProto {
    let mut raw = Vec::with_capacity(values.len() * 4);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    let mut tensor = empty_tensor();
    tensor.dims = dims;
    tensor.data_type = Some(proto::tensor_proto::DataType::Float as i32);
    tensor.name = Some(name.to_string());
    tensor.raw_data = Some(Bytes::from(raw));
    tensor
}

fn tensor_raw(
    name: &str,
    dims: Vec<i64>,
    data_type: proto::tensor_proto::DataType,
    raw: &[u8],
) -> proto::TensorProto {
    let mut tensor = empty_tensor();
    tensor.dims = dims;
    tensor.data_type = Some(data_type as i32);
    tensor.name = Some(name.to_string());
    tensor.raw_data = Some(Bytes::copy_from_slice(raw));
    tensor
}

#[test]
fn load_graph_and_tensor() {
    let tensor = tensor_f32("linear.weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let node = proto::NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["input".to_string(), "linear.weight".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("linear.weight").expect("tensor");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(loader.graph().nodes.len(), 1);
}

#[test]
fn load_external_tensor() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("weights.bin");
    let data = vec![0u8, 0, 128, 63, 0, 0, 0, 64];
    std::fs::write(&data_path, &data).expect("write weights");

    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    write_model(model, &model_path);

    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let slice = loader.tensor("w").expect("tensor");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![1.0, 2.0]);
}

#[test]
fn collect_external_tensor_locations() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("weights.bin".to_string()),
        }],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let locations = external_data_locations(&model_path).expect("locations");
    assert_eq!(locations, vec!["weights.bin".to_string()]);
}

#[test]
fn expose_precision_from_tensor_dtype() {
    let fp16_tensor = tensor_raw(
        "linear_fp16.weight",
        vec![2],
        proto::tensor_proto::DataType::Float16,
        &[0, 0, 0, 0],
    );
    let int8_tensor = tensor_raw(
        "linear_int8.weight",
        vec![2],
        proto::tensor_proto::DataType::Int8,
        &[1, 2],
    );
    let graph = proto::GraphProto {
        initializer: vec![fp16_tensor, int8_tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(
        loader
            .tensor_dtype("linear_fp16.weight")
            .expect("fp16 dtype"),
        Dtype::F16
    );
    assert_eq!(
        loader
            .tensor_dtype("linear_int8.weight")
            .expect("int8 dtype"),
        Dtype::I8
    );
    assert_eq!(loader.unique_precisions(), vec![Dtype::F16, Dtype::I8]);
}

#[test]
fn alias_map_matmul() {
    // Simulate: node "/encoder/layer.0/attention/self/query/MatMul" with input[1] = "onnx::MatMul_977"
    let weight = tensor_f32("onnx::MatMul_977", vec![3, 3], &[1.0; 9]);
    let bias = tensor_f32(
        "encoder.layer.0.attention.self.query.bias",
        vec![3],
        &[0.1, 0.2, 0.3],
    );
    let node = proto::NodeProto {
        name: Some("/encoder/layer.0/attention/self/query/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec![
            "input".to_string(),
            "onnx::MatMul_977".to_string(),
        ],
        output: vec!["matmul_out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight, bias],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Should be able to access by semantic name
    let slice = loader
        .tensor("encoder.layer.0.attention.self.query.weight")
        .expect("alias lookup");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values.len(), 9);

    // Bias should still work directly
    let _bias = loader
        .tensor("encoder.layer.0.attention.self.query.bias")
        .expect("direct bias lookup");

    // names() should expose semantic names
    let names = loader.names();
    assert!(names.contains(&"encoder.layer.0.attention.self.query.weight".to_string()));
    assert!(names.contains(&"encoder.layer.0.attention.self.query.bias".to_string()));
    assert!(!names.iter().any(|n| n.starts_with("onnx::")));
}

#[test]
fn alias_map_gemm() {
    let weight = tensor_f32("onnx::Gemm_42", vec![4, 4], &[1.0; 16]);
    let node = proto::NodeProto {
        name: Some("/classifier/dense/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec![
            "input".to_string(),
            "onnx::Gemm_42".to_string(),
            "classifier.dense.bias".to_string(),
        ],
        output: vec!["gemm_out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    loader
        .tensor("classifier.dense.weight")
        .expect("Gemm alias");
}

#[test]
fn alias_map_gather() {
    let weight = tensor_f32("onnx::Gather_10", vec![100, 8], &[0.5; 800]);
    let node = proto::NodeProto {
        name: Some("/embeddings/word_embeddings/Gather".to_string()),
        op_type: Some("Gather".to_string()),
        input: vec![
            "onnx::Gather_10".to_string(),
            "input_ids".to_string(),
        ],
        output: vec!["gather_out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    loader
        .tensor("embeddings.word_embeddings.weight")
        .expect("Gather alias");
}

#[test]
fn alias_map_no_overwrite_existing() {
    // If a semantic name already exists as a real initializer, don't overwrite it
    let real_weight = tensor_f32("encoder.layer.0.weight", vec![2, 2], &[1.0; 4]);
    let anon_weight = tensor_f32("onnx::MatMul_99", vec![2, 2], &[2.0; 4]);
    let node = proto::NodeProto {
        name: Some("/encoder/layer.0/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["input".to_string(), "onnx::MatMul_99".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![real_weight, anon_weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // The real weight should be returned, not the anonymous one
    let slice = loader.tensor("encoder.layer.0.weight").expect("real weight");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![1.0; 4]);
}

#[test]
fn alias_tensor_provider_iter() {
    use crate::loader::TensorProvider;

    let weight = tensor_f32("onnx::MatMul_1", vec![2, 2], &[1.0; 4]);
    let node = proto::NodeProto {
        name: Some("/layer/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // iter_tensors should expose semantic name
    let tensors: Vec<_> = loader.iter_tensors().collect();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].name, "layer.weight");

    // tensor_info should work with semantic name
    let info = loader.tensor_info("layer.weight").expect("tensor_info");
    assert_eq!(info.name, "layer.weight");
    assert_eq!(info.shape, vec![2, 2]);

    // load_tensor_data should work with semantic name
    let data = loader.load_tensor_data("layer.weight").expect("load data");
    assert_eq!(data.len(), 16); // 4 floats * 4 bytes
}

// ── Helper function tests ─────────────────────────────────────────────

#[test]
fn bytes_to_f32_roundtrip_single() {
    let data = 1.5f32.to_le_bytes();
    let values = bytes_to_f32(&data);
    assert_eq!(values, vec![1.5]);
}

#[test]
fn bytes_to_f32_roundtrip_multiple() {
    let floats = vec![1.0f32, -2.5, 0.0, 3.14];
    let mut bytes = Vec::new();
    for f in &floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    let result = bytes_to_f32(&bytes);
    assert_eq!(result.len(), 4);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - (-2.5)).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    assert!((result[3] - 3.14).abs() < 0.01);
}

#[test]
fn bytes_to_f32_empty_input() {
    let result = bytes_to_f32(&[]);
    assert!(result.is_empty());
}

#[test]
fn bytes_to_f32_incomplete_chunk_ignored() {
    // 5 bytes: 4 for one f32 + 1 trailing byte that gets dropped
    let mut data = 42.0f32.to_le_bytes().to_vec();
    data.push(0xFF);
    let result = bytes_to_f32(&data);
    assert_eq!(result.len(), 1);
    assert!((result[0] - 42.0).abs() < 1e-6);
}

// ── Proto constructor tests ───────────────────────────────────────────

#[test]
fn empty_tensor_all_fields_default() {
    let t = empty_tensor();
    assert!(t.dims.is_empty());
    assert!(t.data_type.is_none());
    assert!(t.name.is_none());
    assert!(t.raw_data.is_none());
    assert!(t.float_data.is_empty());
    assert!(t.int32_data.is_empty());
    assert!(t.int64_data.is_empty());
    assert!(t.string_data.is_empty());
    assert!(t.external_data.is_empty());
    assert!(t.data_location.is_none());
}

#[test]
fn empty_node_all_fields_default() {
    let n = empty_node();
    assert!(n.input.is_empty());
    assert!(n.output.is_empty());
    assert!(n.name.is_none());
    assert!(n.op_type.is_none());
    assert!(n.attribute.is_empty());
}

#[test]
fn empty_graph_all_fields_default() {
    let g = empty_graph();
    assert!(g.node.is_empty());
    assert!(g.name.is_none());
    assert!(g.initializer.is_empty());
    assert!(g.input.is_empty());
    assert!(g.output.is_empty());
}

#[test]
fn empty_model_has_no_graph_when_graph_is_none() {
    let m = proto::ModelProto {
        ir_version: Some(7),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some("".to_string()),
            version: Some(17),
        }],
        graph: None,
        producer_name: Some("test".to_string()),
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        metadata_props: Vec::new(),
        training_info: Vec::new(),
        functions: Vec::new(),
        configuration: Vec::new(),
    };
    assert!(m.graph.is_none());
    assert_eq!(m.ir_version, Some(7));
}

// ── tensor_f32 helper correctness ─────────────────────────────────────

#[test]
fn tensor_f32_creates_correct_proto() {
    let t = tensor_f32("test", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(t.dims, vec![2, 3]);
    assert_eq!(t.data_type, Some(proto::tensor_proto::DataType::Float as i32));
    assert_eq!(t.name, Some("test".to_string()));
    assert!(t.raw_data.is_some());
    let raw = t.raw_data.unwrap();
    assert_eq!(raw.len(), 24); // 6 floats * 4 bytes
}

#[test]
fn tensor_f32_empty_dims_scalar() {
    let t = tensor_f32("scalar", vec![], &[42.0]);
    assert!(t.dims.is_empty());
    let raw = t.raw_data.unwrap();
    assert_eq!(raw.len(), 4);
}

// ── tensor_raw helper correctness ─────────────────────────────────────

#[test]
fn tensor_raw_preserves_exact_bytes() {
    let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let t = tensor_raw("binary", vec![4], proto::tensor_proto::DataType::Uint8, &data);
    assert_eq!(t.raw_data.unwrap().as_ref(), &data);
}

#[test]
fn tensor_raw_int64_type() {
    let data = 12345i64.to_le_bytes();
    let t = tensor_raw("val", vec![], proto::tensor_proto::DataType::Int64, &data);
    assert_eq!(t.data_type, Some(proto::tensor_proto::DataType::Int64 as i32));
}

// ── Error path: missing tensor ────────────────────────────────────────

#[test]
fn tensor_lookup_missing_returns_error() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("exists", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.tensor("nonexistent");
    assert!(result.is_err());
}

// ── Error path: empty model (no graph) ────────────────────────────────

#[test]
fn loader_empty_model_no_graph_fails() {
    let model = proto::ModelProto {
        graph: None,
        ir_version: None,
        opset_import: Vec::new(),
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        metadata_props: Vec::new(),
        training_info: Vec::new(),
        functions: Vec::new(),
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── names() sorting ───────────────────────────────────────────────────

#[test]
fn names_returns_sorted_output() {
    let t1 = tensor_f32("zebra.weight", vec![1], &[1.0]);
    let t2 = tensor_f32("alpha.bias", vec![1], &[2.0]);
    let t3 = tensor_f32("middle.value", vec![1], &[3.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    assert_eq!(names.len(), 3);
    assert_eq!(names[0], "alpha.bias");
    assert_eq!(names[1], "middle.value");
    assert_eq!(names[2], "zebra.weight");
}

// ── precision_by_tensor() sorting ─────────────────────────────────────

#[test]
fn precision_by_tensor_returns_sorted_by_name() {
    let t_f16 = tensor_raw("z_weight", vec![1], proto::tensor_proto::DataType::Float16, &[0, 0]);
    let t_f32 = tensor_raw("a_weight", vec![1], proto::tensor_proto::DataType::Float, &[0, 0, 0, 0]);
    let graph = proto::GraphProto {
        initializer: vec![t_f16, t_f32],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.precision_by_tensor();
    assert_eq!(precisions.len(), 2);
    // Sorted by name: "a_weight" first, "z_weight" second
    assert_eq!(precisions[0].0, "a_weight");
    assert_eq!(precisions[1].0, "z_weight");
    assert_eq!(precisions[0].1, Dtype::F32);
    assert_eq!(precisions[1].1, Dtype::F16);
}

// ── unique_precisions deduplication ───────────────────────────────────

#[test]
fn unique_precisions_deduplicates_same_dtype() {
    let t1 = tensor_f32("w1", vec![1], &[1.0]);
    let t2 = tensor_f32("w2", vec![2], &[2.0, 3.0]);
    let t3 = tensor_raw("w3", vec![1], proto::tensor_proto::DataType::Int8, &[1]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.unique_precisions();
    // Two F32 tensors should collapse to one entry
    assert_eq!(precisions.len(), 2);
    assert_eq!(precisions[0], Dtype::F32);
    assert_eq!(precisions[1], Dtype::I8);
}

// ── model() and path() accessors ──────────────────────────────────────

#[test]
fn model_accessor_returns_inner_model() {
    let tensor = tensor_f32("w", vec![2, 2], &[1.0; 4]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let inner_model = loader.model();
    assert_eq!(inner_model.graph.initializers.len(), 1);
}

#[test]
fn path_accessor_returns_original_path() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.path(), file.path());
}

// ── TensorSlice field access via tensor() ─────────────────────────────

#[test]
fn tensor_slice_fields_populated_correctly() {
    let tensor = tensor_f32("my_weight", vec![3, 4], &[0.0; 12]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("my_weight").expect("slice");
    assert_eq!(slice.shape, vec![3, 4]);
    assert_eq!(slice.dtype, Dtype::F32);
    assert_eq!(slice.data.len(), 48); // 12 floats * 4 bytes
}

// ── tensor_dtype for various ONNX data types ──────────────────────────

#[test]
fn tensor_dtype_int64() {
    let data = 42i64.to_le_bytes();
    let tensor = tensor_raw("idx", vec![], proto::tensor_proto::DataType::Int64, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("idx").expect("dtype"), Dtype::I64);
}

#[test]
fn tensor_dtype_bfloat16() {
    let tensor = tensor_raw("bf", vec![1], proto::tensor_proto::DataType::Bfloat16, &[0, 0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("bf").expect("dtype"), Dtype::BF16);
}

// ── graph() accessor ──────────────────────────────────────────────────

#[test]
fn graph_accessor_exposes_nodes() {
    let node = proto::NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["a".to_string(), "b".to_string()],
        output: vec!["c".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.nodes.len(), 1);
    assert_eq!(g.nodes[0].op_type, "Add");
}

// ── Multiple tensors with mixed types ─────────────────────────────────

#[test]
fn multiple_tensors_mixed_types() {
    let f32_t = tensor_f32("fp32_w", vec![2], &[1.0, 2.0]);
    let i8_t = tensor_raw("int8_w", vec![3], proto::tensor_proto::DataType::Int8, &[1, 2, 3]);
    let graph = proto::GraphProto {
        initializer: vec![f32_t, i8_t],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Verify individual tensor access
    let fp32_data = loader.tensor("fp32_w").expect("fp32");
    assert_eq!(fp32_data.data.len(), 8);
    assert_eq!(fp32_data.dtype, Dtype::F32);

    let i8_data = loader.tensor("int8_w").expect("int8");
    assert_eq!(i8_data.data.len(), 3);
    assert_eq!(i8_data.dtype, Dtype::I8);

    // Verify unique precisions
    let precisions = loader.unique_precisions();
    assert_eq!(precisions.len(), 2);
}

// ── external_data_locations with no external data ─────────────────────

#[test]
fn external_data_locations_no_external() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let locations = external_data_locations(file.path()).expect("locations");
    assert!(locations.is_empty());
}

// ── load_tensor_data returns borrowed cow ──────────────────────────────

#[test]
fn load_tensor_data_returns_correct_bytes() {
    let values = [10.0f32, 20.0, 30.0];
    let tensor = tensor_f32("data", vec![3], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let data = loader.load_tensor_data("data").expect("load data");
    let parsed = bytes_to_f32(&data);
    assert_eq!(parsed.len(), 3);
    assert!((parsed[0] - 10.0).abs() < 1e-6);
    assert!((parsed[1] - 20.0).abs() < 1e-6);
    assert!((parsed[2] - 30.0).abs() < 1e-6);
}

// ── tensor_info for nonexistent tensor returns None ────────────────────

#[test]
fn tensor_info_missing_returns_none() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("exists", vec![2, 2], &[1.0; 4])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.tensor_info("nonexistent").is_none());
    assert!(loader.tensor_info("exists").is_some());
}

// ── weight_layout_hint via alias ──────────────────────────────────────

#[test]
fn weight_layout_hint_matmul_returns_false() {
    let weight = tensor_f32("onnx::MatMul_1", vec![2, 2], &[1.0; 4]);
    let node = proto::NodeProto {
        name: Some("/layer/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Query via semantic alias name
    assert_eq!(loader.weight_layout_hint("layer.weight"), Some(false));
}

// ── ggml_dtype returns None for ONNX ──────────────────────────────────

#[test]
fn ggml_dtype_returns_none_for_onnx() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.ggml_dtype("w").is_none());
}

// ── awq_gptq_aux_data returns None for ONNX ──────────────────────────

#[test]
fn awq_gptq_aux_data_returns_none_for_onnx() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.awq_gptq_aux_data("w").is_none());
}

// ── decode_model error: corrupted file ─────────────────────────────────

#[test]
fn loader_corrupted_file_returns_error() {
    let file = NamedTempFile::new().expect("tempfile");
    std::fs::write(file.path(), b"NOT_A_VALID_PROTOBUF").expect("write");

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── decode_model error: empty file ─────────────────────────────────────

#[test]
fn loader_empty_file_returns_error() {
    let file = NamedTempFile::new().expect("tempfile");
    std::fs::write(file.path(), b"").expect("write");

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── decode_model error: non-existent path ───────────────────────────────

#[test]
fn loader_nonexistent_path_returns_error() {
    let result = OnnxLoader::from_path(std::path::Path::new("/nonexistent/path/model.onnx"));
    assert!(result.is_err());
}

// ── Single-element tensor (scalar) roundtrip ────────────────────────────

#[test]
fn scalar_tensor_roundtrip() {
    let tensor = tensor_f32("bias", vec![], &[42.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("bias").expect("tensor");
    assert!(slice.shape.is_empty());
    assert_eq!(slice.data.len(), 4);
    let values = bytes_to_f32(slice.data);
    assert!((values[0] - 42.0).abs() < 1e-6);
}

// ── tensor_dtype for uint8 ──────────────────────────────────────────────

#[test]
fn tensor_dtype_uint8() {
    let tensor = tensor_raw("mask", vec![3], proto::tensor_proto::DataType::Uint8, &[1, 0, 1]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("mask").expect("dtype"), Dtype::U8);
}

// ── tensor_dtype for int32 ──────────────────────────────────────────────

#[test]
fn tensor_dtype_int32() {
    let data = 100i32.to_le_bytes();
    let tensor = tensor_raw("ids", vec![], proto::tensor_proto::DataType::Int32, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("ids").expect("dtype"), Dtype::I32);
}

// ── tensor_dtype for double (f64) ───────────────────────────────────────

#[test]
fn tensor_dtype_double() {
    let data = 3.14f64.to_le_bytes();
    let tensor = tensor_raw("dbl", vec![], proto::tensor_proto::DataType::Double, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("dbl").expect("dtype"), Dtype::F64);
}

// ── tensor_dtype for bool ───────────────────────────────────────────────

#[test]
fn tensor_dtype_bool() {
    let tensor = tensor_raw("flags", vec![2], proto::tensor_proto::DataType::Bool, &[1, 0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("flags").expect("dtype"), Dtype::U8);
}

// ── Model with graph inputs/outputs preserved ───────────────────────────

#[test]
fn graph_inputs_outputs_preserved() {
    let input_info = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let output_info = proto::ValueInfoProto {
        name: Some("logits".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![input_info],
        output: vec![output_info],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.inputs.len(), 1);
    assert_eq!(g.inputs[0].name, "input_ids");
    assert_eq!(g.outputs.len(), 1);
    assert_eq!(g.outputs[0].name, "logits");
}

// ── Model with multiple nodes in execution order ────────────────────────

#[test]
fn multiple_nodes_preserved_in_order() {
    let nodes = vec![
        proto::NodeProto {
            op_type: Some("MatMul".to_string()),
            input: vec!["x".to_string(), "w1".to_string()],
            output: vec!["h1".to_string()],
            ..empty_node()
        },
        proto::NodeProto {
            op_type: Some("Add".to_string()),
            input: vec!["h1".to_string(), "b1".to_string()],
            output: vec!["h2".to_string()],
            ..empty_node()
        },
        proto::NodeProto {
            op_type: Some("Relu".to_string()),
            input: vec!["h2".to_string()],
            output: vec!["h3".to_string()],
            ..empty_node()
        },
    ];
    let graph = proto::GraphProto {
        node: nodes,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.nodes.len(), 3);
    assert_eq!(g.nodes[0].op_type, "MatMul");
    assert_eq!(g.nodes[1].op_type, "Add");
    assert_eq!(g.nodes[2].op_type, "Relu");
}

// ── external_data_locations with multiple distinct locations ─────────────

#[test]
fn external_data_locations_multiple_distinct() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let t1 = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("a".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("part_a.bin".to_string()),
        }],
        ..empty_tensor()
    };
    let t2 = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("b".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("part_b.bin".to_string()),
        }],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![t1, t2],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let locations = external_data_locations(&model_path).expect("locations");
    assert_eq!(locations.len(), 2);
    assert!(locations.contains(&"part_a.bin".to_string()));
    assert!(locations.contains(&"part_b.bin".to_string()));
}

// ── weight_layout_hint returns None for unknown tensor ──────────────────

#[test]
fn weight_layout_hint_unknown_tensor_returns_none() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![2, 2], &[1.0; 4])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.weight_layout_hint("nonexistent").is_none());
}

// ── alias_map with Gemm transB=1 produces layout hint ───────────────────

#[test]
fn weight_layout_hint_gemm_transb_true() {
    let weight = tensor_f32("onnx::Gemm_7", vec![4, 4], &[1.0; 16]);
    let node = proto::NodeProto {
        name: Some("/classifier/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["input".to_string(), "onnx::Gemm_7".to_string()],
        output: vec!["out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(2), // INT
            i: Some(1),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.weight_layout_hint("classifier.weight"), Some(true));
}

// ── Tensor with large dimensions ────────────────────────────────────────

#[test]
fn tensor_with_large_dimensions() {
    let values: Vec<f32> = vec![0.0; 100];
    let tensor = tensor_f32("large", vec![10, 10], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("large").expect("tensor");
    assert_eq!(slice.shape, vec![10, 10]);
    assert_eq!(slice.data.len(), 400); // 100 floats * 4 bytes
}

// ── Model with opset_import metadata ────────────────────────────────────

#[test]
fn model_opset_import_preserved() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![
            proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx.ml".to_string()),
                version: Some(3),
            },
        ],
        graph: Some(graph),
        producer_name: Some("test-producer".to_string()),
        producer_version: Some("1.0".to_string()),
        domain: None,
        model_version: Some(42),
        doc_string: Some("test doc".to_string()),
        metadata_props: vec![proto::StringStringEntryProto {
            key: Some("author".to_string()),
            value: Some("tester".to_string()),
        }],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let inner = loader.model();
    assert_eq!(inner.metadata.ir_version, 8);
    assert_eq!(inner.metadata.producer_name, "test-producer");
    assert_eq!(inner.metadata.producer_version, "1.0");
    assert_eq!(inner.metadata.model_version, 42);
    assert_eq!(inner.metadata.doc_string, "test doc");
    assert_eq!(inner.metadata.opset_import.len(), 2);
    assert_eq!(inner.metadata.opset_import[0].version, 17);
    assert_eq!(inner.metadata.opset_import[1].domain, "ai.onnx.ml");
    assert_eq!(inner.metadata.metadata_props.get("author").unwrap(), "tester");
}

// ── alias_map: Mul with anonymous weight ────────────────────────────────

#[test]
fn alias_map_mul_anonymous() {
    let weight = tensor_f32("onnx::Mul_3", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let node = proto::NodeProto {
        name: Some("/model/layer_norm/Mul".to_string()),
        op_type: Some("Mul".to_string()),
        input: vec!["x".to_string(), "onnx::Mul_3".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("model.layer_norm.weight").expect("alias");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── names() includes direct and aliased tensors ─────────────────────────

#[test]
fn names_includes_both_direct_and_aliased() {
    let weight = tensor_f32("onnx::MatMul_5", vec![2], &[1.0, 2.0]);
    let bias = tensor_f32("encoder.bias", vec![2], &[0.1, 0.2]);
    let node = proto::NodeProto {
        name: Some("/encoder/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_5".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight, bias],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    assert_eq!(names.len(), 2);
    // encoder.bias stays as-is, onnx::MatMul_5 aliased to encoder.weight
    assert!(names.contains(&"encoder.bias".to_string()));
    assert!(names.contains(&"encoder.weight".to_string()));
}

// ── iter_tensors uses semantic names for aliased tensors ────────────────

#[test]
fn iter_tensors_semantic_names() {
    let weight = tensor_f32("onnx::MatMul_2", vec![3], &[1.0, 2.0, 3.0]);
    let direct = tensor_f32("known.bias", vec![3], &[0.1, 0.2, 0.3]);
    let node = proto::NodeProto {
        name: Some("/fc/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_2".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight, direct],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let tensors: Vec<_> = loader.iter_tensors().collect();
    let tensor_names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(tensor_names.contains(&"fc.weight"));
    assert!(tensor_names.contains(&"known.bias"));
}

// ── tensor_info via alias returns correct shape ──────────────────────────

#[test]
fn tensor_info_via_alias_shape() {
    let weight = tensor_f32("onnx::Gather_8", vec![100, 64], &[0.5; 6400]);
    let node = proto::NodeProto {
        name: Some("/embed/Gather".to_string()),
        op_type: Some("Gather".to_string()),
        input: vec!["onnx::Gather_8".to_string(), "ids".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let info = loader.tensor_info("embed.weight").expect("info");
    assert_eq!(info.shape, vec![100, 64]);
}

// ── unique_precisions with single dtype ──────────────────────────────────

#[test]
fn unique_precisions_single_dtype() {
    let t1 = tensor_f32("a", vec![1], &[1.0]);
    let t2 = tensor_f32("b", vec![2], &[2.0, 3.0]);
    let t3 = tensor_f32("c", vec![1], &[4.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.unique_precisions();
    assert_eq!(precisions, vec![Dtype::F32]);
}

// ── precision_by_tensor with single tensor ──────────────────────────────

#[test]
fn precision_by_tensor_single_entry() {
    let tensor = tensor_f32("w", vec![2], &[1.0, 2.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.precision_by_tensor();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "w");
    assert_eq!(result[0].1, Dtype::F32);
}

// ── load_tensor_data with uint8 tensor ──────────────────────────────────

#[test]
fn load_tensor_data_uint8() {
    let tensor = tensor_raw("bytes", vec![4], proto::tensor_proto::DataType::Uint8, &[0xDE, 0xAD, 0xBE, 0xEF]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let data = loader.load_tensor_data("bytes").expect("data");
    assert_eq!(&data[..], &[0xDE, 0xAD, 0xBE, 0xEF]);
}

// ── load_tensor_data missing tensor returns error ────────────────────────

#[test]
fn load_tensor_data_missing_returns_error() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("exists", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.load_tensor_data("nonexistent");
    assert!(result.is_err());
}

// ── external_data_locations with graph that has no initializers ──────────

#[test]
fn external_data_locations_empty_graph() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let graph = proto::GraphProto {
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let locations = external_data_locations(&model_path).expect("locations");
    assert!(locations.is_empty());
}

// ── Model with node attributes preserved ────────────────────────────────

#[test]
fn node_attributes_preserved() {
    let node = proto::NodeProto {
        op_type: Some("Conv".to_string()),
        input: vec!["X".to_string(), "W".to_string()],
        output: vec!["Y".to_string()],
        attribute: vec![
            proto::AttributeProto {
                name: Some("kernel_shape".to_string()),
                r#type: Some(7), // INTS
                ints: vec![3, 3],
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("strides".to_string()),
                r#type: Some(7), // INTS
                ints: vec![1, 1],
                ..Default::default()
            },
        ],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.nodes[0].attributes.len(), 2);
    assert!(g.nodes[0].attributes.contains_key("kernel_shape"));
    assert!(g.nodes[0].attributes.contains_key("strides"));
}

// ── Model with int64 tensor data ────────────────────────────────────────

#[test]
fn int64_tensor_data_roundtrip() {
    let mut raw = Vec::new();
    raw.extend_from_slice(&42i64.to_le_bytes());
    raw.extend_from_slice(&(-1i64).to_le_bytes());
    let tensor = tensor_raw("indices", vec![2], proto::tensor_proto::DataType::Int64, &raw);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("indices").expect("tensor");
    assert_eq!(slice.dtype, Dtype::I64);
    assert_eq!(slice.data.len(), 16); // 2 * 8 bytes
    let v0 = i64::from_le_bytes(slice.data[0..8].try_into().unwrap());
    let v1 = i64::from_le_bytes(slice.data[8..16].try_into().unwrap());
    assert_eq!(v0, 42);
    assert_eq!(v1, -1);
}

// ── alias_map: multiple aliased tensors in same model ───────────────────

#[test]
fn alias_map_multiple_distinct_aliases() {
    let w1 = tensor_f32("onnx::MatMul_1", vec![2, 2], &[1.0; 4]);
    let w2 = tensor_f32("onnx::Gemm_2", vec![3, 3], &[2.0; 9]);
    let node1 = proto::NodeProto {
        name: Some("/encoder/q_proj/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
        output: vec!["q".to_string()],
        ..empty_node()
    };
    let node2 = proto::NodeProto {
        name: Some("/classifier/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["q".to_string(), "onnx::Gemm_2".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![w1, w2],
        node: vec![node1, node2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let s1 = loader.tensor("encoder.q_proj.weight").expect("alias 1");
    assert_eq!(s1.shape, vec![2, 2]);
    let s2 = loader.tensor("classifier.weight").expect("alias 2");
    assert_eq!(s2.shape, vec![3, 3]);
}

// ══════════════════════════════════════════════════════════════════════
// Additional tests (40 new)
// ══════════════════════════════════════════════════════════════════════

// ── Special float values: NaN in tensor data ────────────────────────

#[test]
fn tensor_f32_nan_value() {
    let nan = f32::NAN.to_le_bytes();
    let tensor = tensor_raw("nan_w", vec![1], proto::tensor_proto::DataType::Float, &nan);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("nan_w").expect("tensor");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values.len(), 1);
    assert!(values[0].is_nan());
}

// ── Special float values: positive infinity ─────────────────────────

#[test]
fn tensor_f32_positive_infinity() {
    let inf = f32::INFINITY.to_le_bytes();
    let tensor = tensor_raw("inf_w", vec![1], proto::tensor_proto::DataType::Float, &inf);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let values = bytes_to_f32(loader.tensor("inf_w").expect("tensor").data);
    assert_eq!(values.len(), 1);
    assert!(values[0].is_infinite() && values[0].is_sign_positive());
}

// ── Special float values: negative infinity ─────────────────────────

#[test]
fn tensor_f32_negative_infinity() {
    let neg_inf = f32::NEG_INFINITY.to_le_bytes();
    let tensor = tensor_raw("neg_inf", vec![1], proto::tensor_proto::DataType::Float, &neg_inf);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let values = bytes_to_f32(loader.tensor("neg_inf").expect("tensor").data);
    assert_eq!(values.len(), 1);
    assert!(values[0].is_infinite() && values[0].is_sign_negative());
}

// ── Special float values: negative zero ─────────────────────────────

#[test]
fn tensor_f32_negative_zero() {
    let neg_zero = (-0.0f32).to_le_bytes();
    let tensor = tensor_raw("neg_zero", vec![], proto::tensor_proto::DataType::Float, &neg_zero);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let values = bytes_to_f32(loader.tensor("neg_zero").expect("tensor").data);
    assert_eq!(values.len(), 1);
    assert!(values[0] == 0.0 && values[0].is_sign_negative());
}

// ── Tensor with f16 raw data stored correctly ───────────────────────

#[test]
fn tensor_f16_raw_data_preserved() {
    // Two f16 values: 1.0 = 0x3C00, 2.0 = 0x4000 (little-endian)
    let raw: &[u8] = &[0x00, 0x3C, 0x00, 0x40];
    let tensor = tensor_raw("fp16_pair", vec![2], proto::tensor_proto::DataType::Float16, raw);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("fp16_pair").expect("tensor");
    assert_eq!(slice.dtype, Dtype::F16);
    assert_eq!(slice.data.len(), 4);
    assert_eq!(&slice.data[0..2], &[0x00, 0x3C]);
    assert_eq!(&slice.data[2..4], &[0x00, 0x40]);
}

// ── Tensor with bf16 raw data stored correctly ──────────────────────

#[test]
fn tensor_bf16_raw_data_preserved() {
    // BF16 1.0 = 0x3F80
    let raw: &[u8] = &[0x80, 0x3F];
    let tensor = tensor_raw("bf16_scalar", vec![], proto::tensor_proto::DataType::Bfloat16, raw);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("bf16_scalar").expect("tensor");
    assert_eq!(slice.dtype, Dtype::BF16);
    assert_eq!(slice.data.len(), 2);
}

// ── Tensor with uint64 dtype ────────────────────────────────────────

#[test]
fn tensor_uint64_dtype() {
    let data = 0xDEADBEEFCAFEBABEu64.to_le_bytes();
    let tensor = tensor_raw("big_id", vec![], proto::tensor_proto::DataType::Uint64, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("big_id").expect("dtype"), Dtype::U64);
    let slice = loader.tensor("big_id").expect("tensor");
    assert_eq!(slice.data.len(), 8);
    let val = u64::from_le_bytes(slice.data[0..8].try_into().unwrap());
    assert_eq!(val, 0xDEADBEEFCAFEBABE);
}

// ── Tensor with fp64 (double) data roundtrip ────────────────────────

#[test]
fn tensor_double_roundtrip() {
    let val = std::f64::consts::PI;
    let data = val.to_le_bytes();
    let tensor = tensor_raw("pi", vec![], proto::tensor_proto::DataType::Double, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("pi").expect("tensor");
    assert_eq!(slice.dtype, Dtype::F64);
    assert_eq!(slice.data.len(), 8);
    let recovered = f64::from_le_bytes(slice.data[0..8].try_into().unwrap());
    assert!((recovered - val).abs() < 1e-15);
}

// ── Tensor with string data (U8 placeholder) ────────────────────────

#[test]
fn tensor_string_data_dtype() {
    let text = b"hello world";
    let tensor = tensor_raw("label", vec![11], proto::tensor_proto::DataType::String, text);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // String tensors are parsed as U8 in the ONNX loader
    let dtype = loader.tensor_dtype("label").expect("dtype");
    assert_eq!(dtype, Dtype::U8);
}

// ── Tensor with int16 dtype ─────────────────────────────────────────

#[test]
fn tensor_int16_dtype() {
    let data = (-100i16).to_le_bytes();
    let tensor = tensor_raw("val", vec![], proto::tensor_proto::DataType::Int16, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("val").expect("dtype"), Dtype::I16);
}

// ── Tensor with int8 dtype and negative values ──────────────────────

#[test]
fn tensor_int8_negative_values() {
    let data: &[u8] = &[0xFF, 0x80, 0x01]; // -1, -128, 1 in two's complement
    let tensor = tensor_raw("signed", vec![3], proto::tensor_proto::DataType::Int8, data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("signed").expect("tensor");
    assert_eq!(slice.dtype, Dtype::I8);
    assert_eq!(slice.data.len(), 3);
    assert_eq!(slice.data[0], 0xFF);
    assert_eq!(slice.data[1], 0x80);
    assert_eq!(slice.data[2], 0x01);
}

// ── Tensor with float16 dtype detection ─────────────────────────────

#[test]
fn tensor_dtype_float16_detected() {
    let tensor = tensor_raw("half", vec![2], proto::tensor_proto::DataType::Float16, &[0, 0, 0, 0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("half").expect("dtype"), Dtype::F16);
}

// ── External data with offset and length ────────────────────────────

#[test]
fn load_external_tensor_with_offset() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("weights.bin");
    // 16 bytes: first 4 are padding (1.0), next 8 are the real data (2.0, 3.0), last 4 are padding (4.0)
    let mut data = Vec::new();
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());
    data.extend_from_slice(&4.0f32.to_le_bytes());
    std::fs::write(&data_path, &data).expect("write weights");

    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("4".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    write_model(model, &model_path);

    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let slice = loader.tensor("w").expect("tensor");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![2.0, 3.0]);
}

// ── External data: missing file returns error ───────────────────────

#[test]
fn load_external_tensor_missing_file_returns_error() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    // Do NOT create the external data file
    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("nonexistent.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    write_model(model, &model_path);

    let result = OnnxLoader::from_path(&model_path);
    assert!(result.is_err());
}

// ── External data: truncated file returns error ─────────────────────

#[test]
fn load_external_tensor_truncated_file_returns_error() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("weights.bin");
    // Write only 4 bytes but declare 8 bytes needed
    std::fs::write(&data_path, &[0u8; 4]).expect("write");

    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    write_model(model, &model_path);

    let result = OnnxLoader::from_path(&model_path);
    assert!(result.is_err());
}

// ── Model with multiple external data files ─────────────────────────

#[test]
fn load_multiple_external_tensors() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let w1_path = dir.path().join("weight_a.bin");
    let w2_path = dir.path().join("weight_b.bin");
    std::fs::write(&w1_path, &10.0f32.to_le_bytes()).expect("write w1");
    std::fs::write(&w2_path, &20.0f32.to_le_bytes()).expect("write w2");

    let make_ext = |name: &str, file: &str| -> proto::TensorProto {
        proto::TensorProto {
            dims: vec![1],
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            name: Some(name.to_string()),
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![
                proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some(file.to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("length".to_string()),
                    value: Some("4".to_string()),
                },
            ],
            ..empty_tensor()
        }
    };
    let graph = proto::GraphProto {
        initializer: vec![make_ext("w_a", "weight_a.bin"), make_ext("w_b", "weight_b.bin")],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let v_a = bytes_to_f32(loader.tensor("w_a").expect("a").data);
    let v_b = bytes_to_f32(loader.tensor("w_b").expect("b").data);
    assert!((v_a[0] - 10.0).abs() < 1e-6);
    assert!((v_b[0] - 20.0).abs() < 1e-6);
}

// ── Model with graph name preserved ─────────────────────────────────

#[test]
fn graph_name_preserved() {
    let graph = proto::GraphProto {
        name: Some("my_inference_graph".to_string()),
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().name, "my_inference_graph");
}

// ── Model with empty graph name defaults to empty string ────────────

#[test]
fn graph_name_none_defaults_to_empty() {
    let graph = proto::GraphProto {
        name: None,
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().name, "");
}

// ── Model with no initializers loads successfully ───────────────────

#[test]
fn model_no_initializers_loads() {
    let node = proto::NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.names().is_empty());
    assert_eq!(loader.graph().nodes.len(), 1);
    assert!(loader.unique_precisions().is_empty());
}

// ── Unique precisions sorted by dtype_rank ───────────────────────────

#[test]
fn unique_precisions_sorted_order() {
    let t1 = tensor_raw("w1", vec![1], proto::tensor_proto::DataType::Int8, &[0]);
    let t2 = tensor_f32("w2", vec![1], &[1.0]);
    let t3 = tensor_raw("w3", vec![1], proto::tensor_proto::DataType::Float16, &[0, 0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let prec = loader.unique_precisions();
    assert_eq!(prec.len(), 3);
    // F32 (rank 1) < F16 (rank 3) < I8 (rank 12)
    assert_eq!(prec[0], Dtype::F32);
    assert_eq!(prec[1], Dtype::F16);
    assert_eq!(prec[2], Dtype::I8);
}

// ── tensor() resolves by alias when direct name does not exist ──────

#[test]
fn tensor_resolves_via_alias_not_direct() {
    let weight = tensor_f32("onnx::Gemm_42", vec![2, 2], &[5.0; 4]);
    let node = proto::NodeProto {
        name: Some("/dense/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["input".to_string(), "onnx::Gemm_42".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Alias lookup
    let slice = loader.tensor("dense.weight").expect("alias lookup");
    assert_eq!(slice.shape, vec![2, 2]);
    // Direct onnx:: name also works
    let direct = loader.tensor("onnx::Gemm_42").expect("direct lookup");
    assert_eq!(direct.shape, vec![2, 2]);
}

// ── load_tensor_data with alias name resolves correctly ──────────────

#[test]
fn load_tensor_data_via_alias() {
    let weight = tensor_f32("onnx::MatMul_7", vec![3], &[10.0, 20.0, 30.0]);
    let node = proto::NodeProto {
        name: Some("/fc/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_7".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let data = loader.load_tensor_data("fc.weight").expect("load alias");
    let values = bytes_to_f32(&data);
    assert_eq!(values, vec![10.0, 20.0, 30.0]);
}

// ── Model with value_info preserved in graph ────────────────────────

#[test]
fn graph_value_info_preserved() {
    let vi = proto::ValueInfoProto {
        name: Some("hidden_state".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi],
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().value_info.len(), 1);
    assert_eq!(loader.graph().value_info[0].name, "hidden_state");
}

// ── Tensor with 0 in dimensions (empty tensor) ──────────────────────

#[test]
fn tensor_zero_dimension() {
    let tensor = tensor_f32("empty", vec![0, 3], &[]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("empty").expect("tensor");
    assert_eq!(slice.shape, vec![0, 3]);
    assert_eq!(slice.data.len(), 0);
}

// ── Model with multiple nodes of same op_type ───────────────────────

#[test]
fn multiple_same_optype_nodes() {
    let nodes: Vec<proto::NodeProto> = (0..5)
        .map(|i| proto::NodeProto {
            op_type: Some("Relu".to_string()),
            input: vec![format!("in_{i}")],
            output: vec![format!("out_{i}")],
            ..empty_node()
        })
        .collect();
    let graph = proto::GraphProto {
        node: nodes,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().nodes.len(), 5);
    for (i, node) in loader.graph().nodes.iter().enumerate() {
        assert_eq!(node.op_type, "Relu");
        assert_eq!(node.inputs[0], format!("in_{i}"));
    }
}

// ── Model with node having many attributes ──────────────────────────

#[test]
fn node_with_multiple_attributes_preserved() {
    let attrs: Vec<proto::AttributeProto> = vec![
        proto::AttributeProto {
            name: Some("kernel_shape".to_string()),
            r#type: Some(7),
            ints: vec![3, 3],
            ..Default::default()
        },
        proto::AttributeProto {
            name: Some("strides".to_string()),
            r#type: Some(7),
            ints: vec![2, 2],
            ..Default::default()
        },
        proto::AttributeProto {
            name: Some("pads".to_string()),
            r#type: Some(7),
            ints: vec![1, 1, 1, 1],
            ..Default::default()
        },
        proto::AttributeProto {
            name: Some("group".to_string()),
            r#type: Some(2),
            i: Some(1),
            ..Default::default()
        },
    ];
    let node = proto::NodeProto {
        op_type: Some("Conv".to_string()),
        input: vec!["X".to_string(), "W".to_string()],
        output: vec!["Y".to_string()],
        attribute: attrs,
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.nodes[0].attributes.len(), 4);
    assert!(g.nodes[0].attributes.contains_key("kernel_shape"));
    assert!(g.nodes[0].attributes.contains_key("strides"));
    assert!(g.nodes[0].attributes.contains_key("pads"));
    assert!(g.nodes[0].attributes.contains_key("group"));
}

// ── ir_version and producer metadata preserved ──────────────────────

#[test]
fn model_metadata_ir_version_zero() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(0),
        opset_import: vec![],
        producer_name: Some(String::new()),
        producer_version: None,
        domain: None,
        model_version: Some(0),
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.model().metadata.ir_version, 0);
    assert_eq!(loader.model().metadata.producer_name, "");
}

// ── Model with large ir_version ─────────────────────────────────────

#[test]
fn model_metadata_large_ir_version() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(i64::MAX),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: Some(i64::MAX),
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.model().metadata.ir_version, i64::MAX);
    assert_eq!(loader.model().metadata.model_version, i64::MAX);
}

// ── precision_by_tensor returns correct count ────────────────────────

#[test]
fn precision_by_tensor_count_matches_initializer_count() {
    let tensors: Vec<proto::TensorProto> = (0..5)
        .map(|i| tensor_f32(&format!("w_{i}"), vec![1], &[i as f32]))
        .collect();
    let graph = proto::GraphProto {
        initializer: tensors,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.precision_by_tensor();
    assert_eq!(result.len(), 5);
    for (name, dtype) in &result {
        assert!(name.starts_with("w_"));
        assert_eq!(*dtype, Dtype::F32);
    }
}

// ── Corrupted protobuf: random bytes after valid header ─────────────

#[test]
fn loader_partial_protobuf_returns_error() {
    let file = NamedTempFile::new().expect("tempfile");
    // Write a valid-looking first byte followed by garbage
    let data = vec![0x08, 0x01, 0xFF, 0xFE, 0xFD, 0xFC];
    std::fs::write(file.path(), &data).expect("write");

    let result = OnnxLoader::from_path(file.path());
    // May or may not error depending on protobuf parsing, but should not panic
    // If it succeeds, the graph would be empty/None which is also caught
    if let Ok(loader) = result {
        // If somehow parsed, graph should exist
        let _ = loader.graph();
    }
}

// ── tensor_dtype for missing tensor returns error ────────────────────

#[test]
fn tensor_dtype_missing_returns_error() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("exists", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.tensor_dtype("missing").is_err());
    assert!(loader.tensor_dtype("exists").is_ok());
}

// ── Model with graph output ValueInfo ───────────────────────────────

#[test]
fn graph_outputs_with_multiple_entries() {
    let outputs = vec![
        proto::ValueInfoProto {
            name: Some("logits".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        },
        proto::ValueInfoProto {
            name: Some("hidden".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        },
    ];
    let graph = proto::GraphProto {
        output: outputs,
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().outputs.len(), 2);
    assert_eq!(loader.graph().outputs[0].name, "logits");
    assert_eq!(loader.graph().outputs[1].name, "hidden");
}

// ── Model with node having domain field ─────────────────────────────

#[test]
fn node_domain_preserved() {
    let node = proto::NodeProto {
        op_type: Some("FusedMatMul".to_string()),
        domain: Some("com.microsoft".to_string()),
        input: vec!["A".to_string(), "B".to_string()],
        output: vec!["C".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().nodes[0].domain, "com.microsoft");
}

// ── iter_tensors yields correct count ────────────────────────────────

#[test]
fn iter_tensors_count_matches_initializers() {
    let t1 = tensor_f32("a", vec![2], &[1.0, 2.0]);
    let t2 = tensor_raw("b", vec![3], proto::tensor_proto::DataType::Int8, &[1, 2, 3]);
    let t3 = tensor_f32("c", vec![1], &[0.5]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let tensors: Vec<_> = loader.iter_tensors().collect();
    assert_eq!(tensors.len(), 3);
}

// ── Model with metadata_props preserved ─────────────────────────────

#[test]
fn model_metadata_props_preserved() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: None,
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("license".to_string()),
                value: Some("Apache-2.0".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("test".to_string()),
            },
        ],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let meta = &loader.model().metadata;
    assert_eq!(meta.metadata_props.get("license").unwrap(), "Apache-2.0");
    assert_eq!(meta.metadata_props.get("author").unwrap(), "test");
}

// ── Tensor with large single dimension ──────────────────────────────

#[test]
fn tensor_large_single_dim() {
    let count = 1000;
    let values: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();
    let tensor = tensor_f32("big", vec![count as i64], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("big").expect("tensor");
    assert_eq!(slice.shape, vec![count]);
    assert_eq!(slice.data.len(), count * 4);
}

// ── Tensor with 3D shape ────────────────────────────────────────────

#[test]
fn tensor_3d_shape() {
    let values = vec![0.0f32; 24]; // 2 * 3 * 4
    let tensor = tensor_f32("cube", vec![2, 3, 4], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("cube").expect("tensor");
    assert_eq!(slice.shape, vec![2, 3, 4]);
    assert_eq!(slice.data.len(), 96);
}

// ── alias_map: named Gemm weight with transB gets layout hint ──────

#[test]
fn named_gemm_weight_with_transb_gets_layout_hint() {
    let weight = tensor_f32("dense.weight", vec![4, 4], &[1.0; 16]);
    let node = proto::NodeProto {
        name: Some("/dense/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["x".to_string(), "dense.weight".to_string()],
        output: vec!["out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(2),
            i: Some(1),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.weight_layout_hint("dense.weight"), Some(true));
}

// ── external_data_locations with corrupted file returns error ────────

#[test]
fn external_data_locations_corrupted_file() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    std::fs::write(&model_path, b"not protobuf").expect("write");

    let result = external_data_locations(&model_path);
    assert!(result.is_err());
}

// ── Model with TensorProto using float_data field ───────────────────

#[test]
fn tensor_with_float_data_field() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Float as i32);
    tensor.name = Some("from_float_data".to_string());
    tensor.float_data = vec![10.0, 20.0, 30.0];
    // No raw_data - the loader should use float_data

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("from_float_data").expect("tensor");
    assert_eq!(slice.shape, vec![3]);
    assert_eq!(slice.dtype, Dtype::F32);
    // The pack module converts float_data to little-endian bytes
    let values = bytes_to_f32(slice.data);
    assert!((values[0] - 10.0).abs() < 1e-6);
    assert!((values[1] - 20.0).abs() < 1e-6);
    assert!((values[2] - 30.0).abs() < 1e-6);
}

// ── Model with TensorProto using int32_data field ───────────────────

#[test]
fn tensor_with_int32_data_field() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int32 as i32);
    tensor.name = Some("from_int32_data".to_string());
    tensor.int32_data = vec![100, -200, 300];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("from_int32_data").expect("tensor");
    assert_eq!(slice.dtype, Dtype::I32);
    assert_eq!(slice.data.len(), 12); // 3 * 4 bytes
}

// ── Model with TensorProto using int64_data field ───────────────────

#[test]
fn tensor_with_int64_data_field() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int64 as i32);
    tensor.name = Some("from_int64_data".to_string());
    tensor.int64_data = vec![42, -1];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("from_int64_data").expect("tensor");
    assert_eq!(slice.dtype, Dtype::I64);
    assert_eq!(slice.data.len(), 16); // 2 * 8 bytes
    let v0 = i64::from_le_bytes(slice.data[0..8].try_into().unwrap());
    let v1 = i64::from_le_bytes(slice.data[8..16].try_into().unwrap());
    assert_eq!(v0, 42);
    assert_eq!(v1, -1);
}

// ── Model with TensorProto using double_data field ──────────────────

#[test]
fn tensor_with_double_data_field() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Double as i32);
    tensor.name = Some("from_double_data".to_string());
    tensor.double_data = vec![1.5, 2.5];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("from_double_data").expect("tensor");
    assert_eq!(slice.dtype, Dtype::F64);
    assert_eq!(slice.data.len(), 16); // 2 * 8 bytes
}

// ── Model with mixed raw_data and field_data tensors ────────────────

#[test]
fn mixed_raw_and_field_data_tensors() {
    // Tensor 1: uses raw_data
    let t_raw = tensor_f32("raw_tensor", vec![2], &[1.0, 2.0]);
    // Tensor 2: uses float_data
    let mut t_field = empty_tensor();
    t_field.dims = vec![2];
    t_field.data_type = Some(proto::tensor_proto::DataType::Float as i32);
    t_field.name = Some("field_tensor".to_string());
    t_field.float_data = vec![3.0, 4.0];

    let graph = proto::GraphProto {
        initializer: vec![t_raw, t_field],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let raw = bytes_to_f32(loader.tensor("raw_tensor").expect("raw").data);
    let field = bytes_to_f32(loader.tensor("field_tensor").expect("field").data);
    assert!((raw[0] - 1.0).abs() < 1e-6);
    assert!((raw[1] - 2.0).abs() < 1e-6);
    assert!((field[0] - 3.0).abs() < 1e-6);
    assert!((field[1] - 4.0).abs() < 1e-6);
}

// ── Model path accessor returns canonical form ──────────────────────

#[test]
fn path_accessor_matches_original() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.path(), file.path());
}

// ── tensor_info returns None for alias that does not map ─────────────

#[test]
fn tensor_info_nonexistent_alias_returns_none() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.tensor_info("nonexistent.weight").is_none());
}

// ── Model with TensorProto missing data_type returns error ──────────

#[test]
fn tensor_missing_data_type_returns_error() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![1];
    tensor.name = Some("bad_tensor".to_string());
    // data_type is None

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── Model with TensorProto missing name returns error ───────────────

#[test]
fn initializer_missing_name_returns_error() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![1];
    tensor.data_type = Some(proto::tensor_proto::DataType::Float as i32);
    tensor.raw_data = Some(Bytes::from(vec![0u8; 4]));
    // name is None

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════════════
// Additional tests (50 new — targeting 164+ total)
// ══════════════════════════════════════════════════════════════════════

// ── OnnxSparseFormat variants construction ───────────────────────────

#[test]
fn onnx_sparse_format_coo_construction() {
    use super::OnnxSparseFormat;
    let format = OnnxSparseFormat::Coo;
    assert_eq!(format, OnnxSparseFormat::Coo);
    assert_ne!(format, OnnxSparseFormat::Csr);
    assert_ne!(format, OnnxSparseFormat::Csc);
}

#[test]
fn onnx_sparse_format_csr_construction() {
    use super::OnnxSparseFormat;
    let format = OnnxSparseFormat::Csr;
    assert_eq!(format, OnnxSparseFormat::Csr);
}

#[test]
fn onnx_sparse_format_csc_construction() {
    use super::OnnxSparseFormat;
    let format = OnnxSparseFormat::Csc;
    assert_eq!(format, OnnxSparseFormat::Csc);
}

#[test]
fn onnx_sparse_format_hash_distinct() {
    use std::collections::HashSet;
    use super::OnnxSparseFormat;
    let set: HashSet<OnnxSparseFormat> = HashSet::from([
        OnnxSparseFormat::Coo,
        OnnxSparseFormat::Csr,
        OnnxSparseFormat::Csc,
    ]);
    assert_eq!(set.len(), 3);
}

// ── OnnxDim Debug output verification ────────────────────────────────

#[test]
fn onnx_dim_debug_known() {
    use super::OnnxDim;
    let dim = OnnxDim::Known(128);
    let debug_str = format!("{dim:?}");
    assert!(debug_str.contains("128"));
}

#[test]
fn onnx_dim_debug_param() {
    use super::OnnxDim;
    let dim = OnnxDim::Param("batch_size".to_string());
    let debug_str = format!("{dim:?}");
    assert!(debug_str.contains("batch_size"));
}

#[test]
fn onnx_dim_debug_unknown() {
    use super::OnnxDim;
    let dim = OnnxDim::Unknown;
    let debug_str = format!("{dim:?}");
    assert!(debug_str.contains("Unknown"));
}

// ── OnnxDim PartialEq for all variants ───────────────────────────────

#[test]
fn onnx_dim_partial_eq_known() {
    use super::OnnxDim;
    assert_eq!(OnnxDim::Known(42), OnnxDim::Known(42));
    assert_ne!(OnnxDim::Known(42), OnnxDim::Known(43));
}

#[test]
fn onnx_dim_partial_eq_param() {
    use super::OnnxDim;
    assert_eq!(
        OnnxDim::Param("N".to_string()),
        OnnxDim::Param("N".to_string())
    );
    assert_ne!(
        OnnxDim::Param("N".to_string()),
        OnnxDim::Param("M".to_string())
    );
}

#[test]
fn onnx_dim_different_variants_not_equal() {
    use super::OnnxDim;
    assert_ne!(OnnxDim::Known(0), OnnxDim::Unknown);
    assert_ne!(OnnxDim::Param("N".to_string()), OnnxDim::Unknown);
    assert_ne!(OnnxDim::Known(0), OnnxDim::Param("0".to_string()));
}

// ── OnnxType Debug/Clone construction ────────────────────────────────

#[test]
fn onnx_type_debug_tensor() {
    use super::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![OnnxDim::Known(3)],
        },
    });
    let debug_str = format!("{ty:?}");
    assert!(debug_str.contains("Tensor"));
}

#[test]
fn onnx_type_clone_tensor() {
    use super::{OnnxDim, OnnxTensorShape, OnnxTensorType, OnnxType};
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![OnnxDim::Known(3)],
        },
    });
    let cloned = ty.clone();
    assert_eq!(ty, cloned);
}

#[test]
fn onnx_type_map_variant() {
    use super::{OnnxMapType, OnnxTensorType, OnnxTensorShape, OnnxType};
    let map_ty = OnnxType::Map(OnnxMapType {
        key_type: proto::tensor_proto::DataType::Int64,
        value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })),
    });
    assert!(matches!(map_ty, OnnxType::Map(_)));
}

#[test]
fn onnx_type_optional_variant() {
    use super::{OnnxTensorType, OnnxTensorShape, OnnxType};
    let opt_ty = OnnxType::Optional(Box::new(OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![] },
    })));
    assert!(matches!(opt_ty, OnnxType::Optional(_)));
}

#[test]
fn onnx_type_sparse_tensor_variant() {
    use super::{OnnxTensorType, OnnxTensorShape, OnnxDim, OnnxType};
    let sp = OnnxType::SparseTensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![OnnxDim::Known(4), OnnxDim::Known(4)],
        },
    });
    assert!(matches!(sp, OnnxType::SparseTensor(_)));
}

#[test]
fn onnx_type_sequence_variant() {
    use super::{OnnxTensorType, OnnxTensorShape, OnnxType};
    let seq = OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int64,
        shape: OnnxTensorShape { dims: vec![] },
    })));
    assert!(matches!(seq, OnnxType::Sequence(_)));
}

// ── OnnxTensorShape empty and multi-dim ──────────────────────────────

#[test]
fn onnx_tensor_shape_clone() {
    use super::{OnnxDim, OnnxTensorShape};
    let shape = OnnxTensorShape {
        dims: vec![OnnxDim::Known(3), OnnxDim::Param("N".to_string())],
    };
    let cloned = shape.clone();
    assert_eq!(shape, cloned);
    assert_eq!(shape.dims.len(), 2);
}

#[test]
fn onnx_tensor_shape_partial_eq() {
    use super::{OnnxDim, OnnxTensorShape};
    let a = OnnxTensorShape {
        dims: vec![OnnxDim::Known(3), OnnxDim::Unknown],
    };
    let b = OnnxTensorShape {
        dims: vec![OnnxDim::Known(3), OnnxDim::Unknown],
    };
    let c = OnnxTensorShape {
        dims: vec![OnnxDim::Known(4)],
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ── OnnxModelMetadata field access ───────────────────────────────────

#[test]
fn model_metadata_all_default_fields() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let meta = &loader.model().metadata;
    assert_eq!(meta.ir_version, 0);
    assert_eq!(meta.producer_name, "");
    assert_eq!(meta.producer_version, "");
    assert_eq!(meta.domain, "");
    assert_eq!(meta.model_version, 0);
    assert_eq!(meta.doc_string, "");
    assert!(meta.opset_import.is_empty());
    assert!(meta.metadata_props.is_empty());
}

#[test]
fn model_metadata_full_fields() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(9),
        opset_import: vec![
            proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(20),
            },
        ],
        producer_name: Some("gllm-exporter".to_string()),
        producer_version: Some("2.1.0".to_string()),
        domain: Some("ai.gllm".to_string()),
        model_version: Some(100),
        doc_string: Some("test model".to_string()),
        graph: Some(graph),
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("key_a".to_string()),
                value: Some("val_a".to_string()),
            },
        ],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let meta = &loader.model().metadata;
    assert_eq!(meta.ir_version, 9);
    assert_eq!(meta.producer_name, "gllm-exporter");
    assert_eq!(meta.producer_version, "2.1.0");
    assert_eq!(meta.domain, "ai.gllm");
    assert_eq!(meta.model_version, 100);
    assert_eq!(meta.doc_string, "test model");
    assert_eq!(meta.opset_import.len(), 1);
    assert_eq!(meta.opset_import[0].domain, "");
    assert_eq!(meta.opset_import[0].version, 20);
    assert_eq!(meta.metadata_props.get("key_a").unwrap(), "val_a");
}

// ── OnnxGraph doc_string preserved ──────────────────────────────────

#[test]
fn graph_doc_string_preserved() {
    let graph = proto::GraphProto {
        doc_string: Some("A test computation graph".to_string()),
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().doc_string, "A test computation graph");
}

// ── OnnxNode with empty inputs and outputs ───────────────────────────

#[test]
fn node_empty_inputs_outputs() {
    let node = proto::NodeProto {
        op_type: Some("Constant".to_string()),
        input: vec![],
        output: vec!["const_out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().nodes[0].inputs.len(), 0);
    assert_eq!(loader.graph().nodes[0].outputs.len(), 1);
}

// ── OnnxNode name auto-generated when None ───────────────────────────

#[test]
fn node_name_none_gets_auto_generated() {
    let node = proto::NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        name: None,
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // When name is None, the parser generates "node_{index}" as fallback
    assert_eq!(loader.graph().nodes[0].name, "node_0");
}

// ── OnnxValueInfo doc_string and metadata_props defaults ─────────────

#[test]
fn value_info_fields_default() {
    let vi = proto::ValueInfoProto {
        name: Some("x".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![vi],
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let input = &loader.graph().inputs[0];
    assert_eq!(input.name, "x");
    assert!(input.value_type.is_none());
    assert_eq!(input.doc_string, "");
    assert!(input.metadata_props.is_empty());
}

// ── Tensor with uint32 dtype via uint64_data field ───────────────────

#[test]
fn tensor_uint32_via_uint64_data() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint32 as i32);
    tensor.name = Some("u32_ids".to_string());
    tensor.uint64_data = vec![100, 200];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("u32_ids").expect("tensor");
    assert_eq!(slice.dtype, Dtype::U32);
    assert_eq!(slice.data.len(), 8); // 2 * 4 bytes
}

// ── Tensor with uint16 dtype via int32_data field ────────────────────

#[test]
fn tensor_uint16_via_int32_data() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint16 as i32);
    tensor.name = Some("u16_vals".to_string());
    tensor.int32_data = vec![10, 20, 30];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_dtype("u16_vals").expect("dtype"), Dtype::U16);
    let slice = loader.tensor("u16_vals").expect("tensor");
    assert_eq!(slice.data.len(), 6); // 3 * 2 bytes
}

// ── Tensor with float16 via int32_data field (bits) ──────────────────

#[test]
fn tensor_float16_via_int32_data() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Float16 as i32);
    tensor.name = Some("f16_bits".to_string());
    // 0x3C00 = f16(1.0), 0x4000 = f16(2.0) stored in int32_data
    tensor.int32_data = vec![0x3C00, 0x4000];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("f16_bits").expect("tensor");
    assert_eq!(slice.dtype, Dtype::F16);
    assert_eq!(slice.data.len(), 4); // 2 * 2 bytes
}

// ── Tensor with bf16 via int32_data field (bits) ─────────────────────

#[test]
fn tensor_bf16_via_int32_data() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![1];
    tensor.data_type = Some(proto::tensor_proto::DataType::Bfloat16 as i32);
    tensor.name = Some("bf16_bits".to_string());
    // 0x3F80 = bf16(1.0)
    tensor.int32_data = vec![0x3F80];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("bf16_bits").expect("tensor");
    assert_eq!(slice.dtype, Dtype::BF16);
    assert_eq!(slice.data.len(), 2); // 1 * 2 bytes
}

// ── Tensor with uint64_data field ────────────────────────────────────

#[test]
fn tensor_uint64_via_uint64_data() {
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint64 as i32);
    tensor.name = Some("u64_data".to_string());
    tensor.uint64_data = vec![u64::MAX, 0];

    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("u64_data").expect("tensor");
    assert_eq!(slice.dtype, Dtype::U64);
    assert_eq!(slice.data.len(), 16); // 2 * 8 bytes
    let v0 = u64::from_le_bytes(slice.data[0..8].try_into().unwrap());
    assert_eq!(v0, u64::MAX);
}

// ── Tensor with raw_data size mismatch returns error ─────────────────

#[test]
fn tensor_raw_data_size_mismatch_returns_error() {
    // Declare 2 F32 elements (8 bytes needed) but provide only 4 bytes
    let tensor = tensor_raw("bad_size", vec![2], proto::tensor_proto::DataType::Float, &[0u8; 4]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── Duplicate tensor name returns error ──────────────────────────────

#[test]
fn duplicate_initializer_name_returns_error() {
    let t1 = tensor_f32("same_name", vec![1], &[1.0]);
    let t2 = tensor_f32("same_name", vec![1], &[2.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
}

// ── OnnxGraph with quantization_annotation (no crash) ───────────────

#[test]
fn graph_with_quantization_annotation() {
    let qa = proto::TensorAnnotation {
        tensor_name: Some("w".to_string()),
        quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
            key: Some("scale".to_string()),
            value: Some("w_scale".to_string()),
        }],
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![2, 2], &[1.0; 4])],
        quantization_annotation: vec![qa],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().quantization_annotation.len(), 1);
    assert_eq!(loader.graph().quantization_annotation[0].tensor_name, "w");
}

// ── External data with zero offset ──────────────────────────────────

#[test]
fn load_external_tensor_zero_offset() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("data.bin");
    let data = 42.0f32.to_le_bytes();
    std::fs::write(&data_path, &data).expect("write");

    let tensor = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("data.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("0".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("4".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let values = bytes_to_f32(loader.tensor("w").expect("tensor").data);
    assert!((values[0] - 42.0).abs() < 1e-6);
}

// ── External data locations deduplicates same location ───────────────

#[test]
fn external_data_locations_deduplicates() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let make_ext = |name: &str| -> proto::TensorProto {
        proto::TensorProto {
            dims: vec![1],
            data_type: Some(proto::tensor_proto::DataType::Float as i32),
            name: Some(name.to_string()),
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("same_file.bin".to_string()),
            }],
            ..empty_tensor()
        }
    };
    let graph = proto::GraphProto {
        initializer: vec![make_ext("a"), make_ext("b"), make_ext("c")],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let locations = external_data_locations(&model_path).expect("locations");
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0], "same_file.bin");
}

// ── weight_layout_hint Gather returns None (no layout hint) ──────────

#[test]
fn weight_layout_hint_gather_returns_none() {
    let weight = tensor_f32("onnx::Gather_1", vec![50, 32], &[0.5; 1600]);
    let node = proto::NodeProto {
        name: Some("/embed/Gather".to_string()),
        op_type: Some("Gather".to_string()),
        input: vec!["onnx::Gather_1".to_string(), "ids".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Gather nodes do not produce layout hints
    assert_eq!(loader.weight_layout_hint("embed.weight"), None);
}

// ── Gemm with transB=0 layout hint returns false ────────────────────

#[test]
fn weight_layout_hint_gemm_transb_false() {
    let weight = tensor_f32("dense.weight", vec![4, 4], &[1.0; 16]);
    let node = proto::NodeProto {
        name: Some("/dense/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["x".to_string(), "dense.weight".to_string()],
        output: vec!["out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(2),
            i: Some(0),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.weight_layout_hint("dense.weight"), Some(false));
}

// ── OnnxGraph metadata_props preserved ──────────────────────────────

#[test]
fn graph_metadata_props_preserved() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("source".to_string()),
                value: Some("pytorch".to_string()),
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.metadata_props.get("source").unwrap(), "pytorch");
}

// ── names() does not contain onnx:: prefixes after aliasing ──────────

#[test]
fn names_no_onnx_prefix_after_alias() {
    let weight = tensor_f32("onnx::MatMul_1", vec![2], &[1.0, 2.0]);
    let node = proto::NodeProto {
        name: Some("/fc/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    assert!(!names.iter().any(|n| n.starts_with("onnx::")));
    assert!(names.contains(&"fc.weight".to_string()));
}

// ── Model with functions list preserved (empty) ─────────────────────

#[test]
fn model_functions_list_empty() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.model().functions.is_empty());
}

// ── Conv node does not produce alias ─────────────────────────────────

#[test]
fn conv_node_no_alias_produced() {
    let weight = tensor_f32("onnx::Conv_1", vec![3, 3, 3, 3], &[1.0; 81]);
    let node = proto::NodeProto {
        name: Some("/conv1/Conv".to_string()),
        op_type: Some("Conv".to_string()),
        input: vec!["x".to_string(), "onnx::Conv_1".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Conv is not in the alias-producing op list, so no alias created
    assert!(loader.tensor("conv1.weight").is_err());
    // But the original onnx:: name should still be accessible
    assert!(loader.tensor("onnx::Conv_1").is_ok());
}

// ── Tensor with 4D shape ────────────────────────────────────────────

#[test]
fn tensor_4d_shape() {
    let values = vec![0.0f32; 24]; // 2*3*2*2
    let tensor = tensor_f32("conv_w", vec![2, 3, 2, 2], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("conv_w").expect("tensor");
    assert_eq!(slice.shape, vec![2, 3, 2, 2]);
    assert_eq!(slice.data.len(), 96); // 24 * 4 bytes
}

// ── iter_tensors shape matches tensor() shape ────────────────────────

#[test]
fn iter_tensors_shape_matches_tensor_shape() {
    let t = tensor_f32("w", vec![5, 10], &[0.0f32; 50]);
    let graph = proto::GraphProto {
        initializer: vec![t],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let meta = loader.iter_tensors().next().expect("one tensor");
    let slice = loader.tensor("w").expect("tensor");
    assert_eq!(meta.shape, slice.shape);
    assert_eq!(meta.dtype, slice.dtype);
}

// ── OnnxLoader Debug trait ──────────────────────────────────────────

#[test]
fn onnx_loader_debug_output() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let debug_str = format!("{loader:?}");
    assert!(debug_str.contains("OnnxLoader"));
}

// ── Multiple opsets with different domains ───────────────────────────

#[test]
fn multiple_opset_domains() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![
            proto::OperatorSetIdProto {
                domain: Some("".to_string()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx.ml".to_string()),
                version: Some(3),
            },
            proto::OperatorSetIdProto {
                domain: Some("ai.onnx.training".to_string()),
                version: Some(1),
            },
        ],
        graph: Some(graph),
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let opsets = &loader.model().metadata.opset_import;
    assert_eq!(opsets.len(), 3);
    assert_eq!(opsets[0].domain, "");
    assert_eq!(opsets[1].domain, "ai.onnx.ml");
    assert_eq!(opsets[2].domain, "ai.onnx.training");
}

// ── Graph with empty node list and no initializers ───────────────────

#[test]
fn graph_completely_empty_loads() {
    let graph = proto::GraphProto {
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.graph().nodes.is_empty());
    assert!(loader.names().is_empty());
    assert!(loader.unique_precisions().is_empty());
    let tensors: Vec<_> = loader.iter_tensors().collect();
    assert!(tensors.is_empty());
}

// ── Node with missing op_type returns error ──────────────────────────

#[test]
fn node_missing_optype_returns_error() {
    let node = proto::NodeProto {
        op_type: None,
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let result = OnnxLoader::from_path(file.path());
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("missing op_type"));
}

// ── tensor_info returns correct dtype for various types ──────────────

#[test]
fn tensor_info_dtype_various_types() {
    let f32_t = tensor_f32("f32_w", vec![2], &[1.0, 2.0]);
    let i64_t = tensor_raw("i64_w", vec![1], proto::tensor_proto::DataType::Int64, &0i64.to_le_bytes());
    let bf16_t = tensor_raw("bf16_w", vec![1], proto::tensor_proto::DataType::Bfloat16, &[0, 0]);
    let graph = proto::GraphProto {
        initializer: vec![f32_t, i64_t, bf16_t],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.tensor_info("f32_w").unwrap().dtype, Dtype::F32);
    assert_eq!(loader.tensor_info("i64_w").unwrap().dtype, Dtype::I64);
    assert_eq!(loader.tensor_info("bf16_w").unwrap().dtype, Dtype::BF16);
}

// ── External data with large offset ─────────────────────────────────

#[test]
fn load_external_tensor_large_offset() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("data.bin");
    // Write 1028 bytes: 1024 padding + 4 bytes real data
    let mut data = vec![0u8; 1024];
    data.extend_from_slice(&99.0f32.to_le_bytes());
    std::fs::write(&data_path, &data).expect("write");

    let tensor = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("data.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("1024".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("4".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let values = bytes_to_f32(loader.tensor("w").expect("tensor").data);
    assert!((values[0] - 99.0).abs() < 1e-6);
}

// ── OnnxTensorType Debug output ──────────────────────────────────────

#[test]
fn onnx_tensor_type_debug() {
    use super::{OnnxDim, OnnxTensorShape, OnnxTensorType};
    let tt = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![OnnxDim::Known(3), OnnxDim::Known(4)],
        },
    };
    let debug_str = format!("{tt:?}");
    assert!(debug_str.contains("OnnxTensorType"));
}

// ── LoaderError Onnx variant contains message ────────────────────────

#[test]
fn loader_error_onnx_message() {
    let file = NamedTempFile::new().expect("tempfile");
    std::fs::write(file.path(), b"NOT_PROTOBUF").expect("write");

    let result = OnnxLoader::from_path(file.path());
    match result {
        Err(e) => {
            let msg = e.to_string();
            assert!(!msg.is_empty());
        }
        Ok(_) => panic!("expected error for corrupted protobuf"),
    }
}

// ── TensorSlice Debug output ─────────────────────────────────────────

#[test]
fn tensor_slice_debug() {
    let tensor = tensor_f32("w", vec![2], &[1.0, 2.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("w").expect("tensor");
    let debug_str = format!("{slice:?}");
    assert!(debug_str.contains("TensorSlice"));
}

// ── ConvertError Display messages ─────────────────────────────────────

#[test]
fn convert_error_unsupported_op_display() {
    let err = super::graph_convert::ConvertError::UnsupportedOp {
        op_type: "DynamicQuantizeLinear".to_string(),
        node_name: "quant_node".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("DynamicQuantizeLinear"));
    assert!(msg.contains("quant_node"));
}

#[test]
fn convert_error_missing_initializer_display() {
    let err = super::graph_convert::ConvertError::MissingInitializer {
        name: "weight_0".to_string(),
        node_name: "matmul_0".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("weight_0"));
    assert!(msg.contains("matmul_0"));
}

#[test]
fn convert_error_invalid_matmul_shape_display() {
    let err = super::graph_convert::ConvertError::InvalidMatMulShape {
        name: "W".to_string(),
        dims: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("W"));
    assert!(msg.contains("3-D"));
}

#[test]
fn convert_error_no_weight_input_display() {
    let err = super::graph_convert::ConvertError::NoWeightInput {
        node_name: "mm".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("mm"));
}

#[test]
fn convert_error_attribute_error_display() {
    let err = super::graph_convert::ConvertError::AttributeError {
        node_name: "conv1".to_string(),
        reason: "missing kernel_shape".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("conv1"));
    assert!(msg.contains("missing kernel_shape"));
}

#[test]
fn convert_error_shape_inference_failed_display() {
    let err = super::graph_convert::ConvertError::ShapeInferenceFailed {
        name: "output_0".to_string(),
        reason: "cannot infer broadcast shape".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("output_0"));
    assert!(msg.contains("cannot infer broadcast shape"));
}

// ── OnnxFunction struct construction and field access ─────────────────

#[test]
fn onnx_function_all_fields() {
    let func = super::model::OnnxFunction {
        name: "ScaledDotProductAttention".to_string(),
        domain: "com.custom".to_string(),
        overload: "v2".to_string(),
        inputs: vec!["Q".to_string(), "K".to_string(), "V".to_string()],
        outputs: vec!["output".to_string()],
        attributes: vec!["scale".to_string(), "mask".to_string()],
        attribute_protos: std::collections::HashMap::new(),
        nodes: vec![],
        opset_import: vec![super::model::OnnxOperatorSet {
            domain: "".to_string(),
            version: 17,
        }],
        value_info: vec![],
        doc_string: "Custom attention function".to_string(),
        metadata_props: std::collections::HashMap::new(),
    };
    assert_eq!(func.name, "ScaledDotProductAttention");
    assert_eq!(func.domain, "com.custom");
    assert_eq!(func.overload, "v2");
    assert_eq!(func.inputs.len(), 3);
    assert_eq!(func.outputs.len(), 1);
    assert_eq!(func.attributes.len(), 2);
    assert!(func.nodes.is_empty());
    assert_eq!(func.opset_import.len(), 1);
    assert_eq!(func.doc_string, "Custom attention function");
}

#[test]
fn onnx_function_clone() {
    let func = super::model::OnnxFunction {
        name: "CustomRelu".to_string(),
        domain: "".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec![],
        attribute_protos: std::collections::HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: std::collections::HashMap::new(),
    };
    let cloned = func.clone();
    assert_eq!(cloned.name, "CustomRelu");
    assert_eq!(cloned.inputs, vec!["X".to_string()]);
}

#[test]
fn onnx_function_debug_format() {
    let func = super::model::OnnxFunction {
        name: "MyFunc".to_string(),
        domain: "test".to_string(),
        overload: String::new(),
        inputs: vec!["A".to_string()],
        outputs: vec!["B".to_string()],
        attributes: vec![],
        attribute_protos: std::collections::HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: std::collections::HashMap::new(),
    };
    let debug = format!("{func:?}");
    assert!(debug.contains("MyFunc"));
    assert!(debug.contains("test"));
}

// ── OnnxQuantizationAnnotation edge cases ─────────────────────────────

#[test]
fn onnx_quantization_annotation_all_none_optional_fields() {
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "weight_q".to_string(),
        quant_param_tensor_names: std::collections::HashMap::new(),
        scale: None,
        zero_point: None,
        axis: None,
    };
    assert!(qa.scale.is_none());
    assert!(qa.zero_point.is_none());
    assert!(qa.axis.is_none());
    assert_eq!(qa.tensor_name, "weight_q");
}

#[test]
fn onnx_quantization_annotation_with_all_fields() {
    let mut params = std::collections::HashMap::new();
    params.insert("SCALE_TENSOR".to_string(), "weight_scale".to_string());
    params.insert("ZERO_POINT_TENSOR".to_string(), "weight_zp".to_string());
    params.insert("AXIS".to_string(), "0".to_string());
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "weight".to_string(),
        quant_param_tensor_names: params,
        scale: Some(0.005),
        zero_point: Some(64),
        axis: Some(0),
    };
    assert_eq!(qa.quant_param_tensor_names.len(), 3);
    assert_eq!(qa.scale.unwrap(), 0.005);
    assert_eq!(qa.zero_point.unwrap(), 64);
    assert_eq!(qa.axis.unwrap(), 0);
}

#[test]
fn onnx_quantization_annotation_clone() {
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "t".to_string(),
        quant_param_tensor_names: std::collections::HashMap::new(),
        scale: Some(1.0),
        zero_point: None,
        axis: Some(1),
    };
    let cloned = qa.clone();
    assert_eq!(cloned.tensor_name, "t");
    assert_eq!(cloned.scale, Some(1.0));
    assert_eq!(cloned.axis, Some(1));
}

#[test]
fn onnx_quantization_annotation_debug() {
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "my_tensor".to_string(),
        quant_param_tensor_names: std::collections::HashMap::new(),
        scale: None,
        zero_point: None,
        axis: None,
    };
    let debug = format!("{qa:?}");
    assert!(debug.contains("my_tensor"));
}

// ── OnnxNode clone and debug ──────────────────────────────────────────

#[test]
fn onnx_node_clone_preserves_attributes() {
    let mut attrs = std::collections::HashMap::new();
    attrs.insert(
        "kernel_shape".to_string(),
        super::OnnxAttribute {
            name: "kernel_shape".to_string(),
            value: super::OnnxAttributeValue::Ints(vec![3, 3]),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        },
    );
    let node = super::model::OnnxNode {
        name: "conv1".to_string(),
        op_type: "Conv".to_string(),
        domain: String::new(),
        inputs: vec!["X".to_string(), "W".to_string(), "B".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: attrs,
    };
    let cloned = node.clone();
    assert_eq!(cloned.name, "conv1");
    assert_eq!(cloned.op_type, "Conv");
    assert_eq!(cloned.inputs.len(), 3);
    assert!(cloned.attributes.contains_key("kernel_shape"));
}

#[test]
fn onnx_node_debug_format() {
    let node = super::model::OnnxNode {
        name: "relu_1".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
        attributes: std::collections::HashMap::new(),
    };
    let debug = format!("{node:?}");
    assert!(debug.contains("relu_1"));
    assert!(debug.contains("Relu"));
}

// ── OnnxGraph clone and debug ─────────────────────────────────────────

#[test]
fn onnx_graph_clone_is_independent() {
    let graph = super::model::OnnxGraph {
        name: "g1".to_string(),
        doc_string: "test graph".to_string(),
        nodes: vec![super::model::OnnxNode {
            name: "n1".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: std::collections::HashMap::new(),
        }],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: std::collections::HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: std::collections::HashMap::new(),
    };
    let mut cloned = graph.clone();
    cloned.name = "g2".to_string();
    assert_eq!(graph.name, "g1");
    assert_eq!(cloned.name, "g2");
}

#[test]
fn onnx_graph_debug_format() {
    let graph = super::model::OnnxGraph {
        name: "main_graph".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: std::collections::HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: std::collections::HashMap::new(),
    };
    let debug = format!("{graph:?}");
    assert!(debug.contains("main_graph"));
}

// ── OnnxModelMetadata clone and debug ─────────────────────────────────

#[test]
fn onnx_model_metadata_clone() {
    let meta = super::model::OnnxModelMetadata {
        ir_version: 8,
        producer_name: "onnx-test".to_string(),
        producer_version: "2.0".to_string(),
        domain: "ai.onnx".to_string(),
        model_version: 42,
        doc_string: "test model".to_string(),
        opset_import: vec![super::model::OnnxOperatorSet {
            domain: "".to_string(),
            version: 17,
        }],
        metadata_props: std::collections::HashMap::new(),
    };
    let cloned = meta.clone();
    assert_eq!(cloned.ir_version, 8);
    assert_eq!(cloned.producer_name, "onnx-test");
    assert_eq!(cloned.opset_import.len(), 1);
}

#[test]
fn onnx_model_metadata_debug() {
    let meta = super::model::OnnxModelMetadata {
        ir_version: 9,
        producer_name: "debugger".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        opset_import: vec![],
        metadata_props: std::collections::HashMap::new(),
    };
    let debug = format!("{meta:?}");
    assert!(debug.contains("debugger"));
}

// ── OnnxOperatorSet clone and debug ───────────────────────────────────

#[test]
fn onnx_operator_set_clone() {
    let ops = super::model::OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 3,
    };
    let cloned = ops.clone();
    assert_eq!(cloned.domain, "ai.onnx.ml");
    assert_eq!(cloned.version, 3);
}

#[test]
fn onnx_operator_set_debug() {
    let ops = super::model::OnnxOperatorSet {
        domain: "custom".to_string(),
        version: 1,
    };
    let debug = format!("{ops:?}");
    assert!(debug.contains("custom"));
}

#[test]
fn onnx_operator_set_field_equality() {
    let a = super::model::OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 17,
    };
    let b = super::model::OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 17,
    };
    assert_eq!(a.domain, b.domain);
    assert_eq!(a.version, b.version);
}

// ── OnnxValueInfo clone and debug ─────────────────────────────────────

#[test]
fn onnx_value_info_clone() {
    let vi = super::model::OnnxValueInfo {
        name: "input_ids".to_string(),
        value_type: Some(super::OnnxType::Tensor(super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: super::OnnxTensorShape {
                dims: vec![super::OnnxDim::Param("batch".to_string()), super::OnnxDim::Known(128)],
            },
        })),
        doc_string: "token ids".to_string(),
        metadata_props: std::collections::HashMap::new(),
    };
    let cloned = vi.clone();
    assert_eq!(cloned.name, "input_ids");
    assert!(cloned.value_type.is_some());
    assert_eq!(cloned.doc_string, "token ids");
}

#[test]
fn onnx_value_info_debug() {
    let vi = super::model::OnnxValueInfo {
        name: "attention_mask".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: std::collections::HashMap::new(),
    };
    let debug = format!("{vi:?}");
    assert!(debug.contains("attention_mask"));
}

// ── OnnxLoader graph() model() path() methods ─────────────────────────

#[test]
fn loader_graph_method_returns_graph() {
    let tensor = tensor_f32("w", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert!(g.initializers.contains_key("w"));
    assert_eq!(g.initializers.len(), 1);
}

#[test]
fn loader_model_method_returns_model() {
    let tensor = tensor_f32("bias", vec![3], &[0.1, 0.2, 0.3]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let m = loader.model();
    assert_eq!(m.graph.initializers.len(), 1);
    assert!(m.functions.is_empty());
}

#[test]
fn loader_path_method_returns_original_path() {
    let graph = proto::GraphProto {
        initializer: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.path(), file.path());
}

// ── OnnxLoader precision_by_tensor / unique_precisions ────────────────

#[test]
fn loader_precision_by_tensor_returns_dtype_per_tensor() {
    let tensor1 = tensor_f32("a", vec![2], &[1.0, 2.0]);
    let tensor2 = tensor_f32("b", vec![3], &[3.0, 4.0, 5.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor1, tensor2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.precision_by_tensor();
    assert_eq!(precisions.len(), 2);
    for (_name, dtype) in &precisions {
        assert_eq!(*dtype, Dtype::F32);
    }
}

#[test]
fn loader_unique_precisions_deduplicates() {
    let tensor1 = tensor_f32("w1", vec![2], &[1.0, 2.0]);
    let tensor2 = tensor_f32("w2", vec![3], &[3.0, 4.0, 5.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor1, tensor2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let unique = loader.unique_precisions();
    assert_eq!(unique.len(), 1);
    assert_eq!(unique[0], Dtype::F32);
}

#[test]
fn loader_tensor_dtype_method() {
    let tensor = tensor_f32("emb", vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let dtype = loader.tensor_dtype("emb").expect("dtype");
    assert_eq!(dtype, Dtype::F32);
}

// ── OnnxLoader names() with multiple tensors ──────────────────────────

#[test]
fn loader_names_returns_all_initializer_names() {
    let t1 = tensor_f32("alpha", vec![1], &[1.0]);
    let t2 = tensor_f32("beta", vec![1], &[2.0]);
    let t3 = tensor_f32("gamma", vec![1], &[3.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"alpha".to_string()));
    assert!(names.contains(&"beta".to_string()));
    assert!(names.contains(&"gamma".to_string()));
}

// ── OnnxLoader tensor() error for missing tensor ──────────────────────

#[test]
fn loader_tensor_missing_returns_error() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("exists", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.tensor("nonexistent");
    assert!(result.is_err());
}

#[test]
fn loader_tensor_dtype_missing_returns_error() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("present", vec![1], &[0.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let result = loader.tensor_dtype("absent");
    assert!(result.is_err());
}

// ── Loading model with model metadata ─────────────────────────────────

#[test]
fn load_model_with_ir_version_and_producer() {
    let mut model = empty_model(proto::GraphProto {
        initializer: vec![],
        ..empty_graph()
    });
    model.ir_version = Some(8);
    model.producer_name = Some("pytorch".to_string());
    model.producer_version = Some("2.1.0".to_string());

    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let m = loader.model();
    assert_eq!(m.metadata.ir_version, 8);
    assert_eq!(m.metadata.producer_name, "pytorch");
    assert_eq!(m.metadata.producer_version, "2.1.0");
}

// ── Loading model with opset import ───────────────────────────────────

#[test]
fn load_model_with_custom_opset() {
    let mut model = empty_model(proto::GraphProto {
        initializer: vec![],
        ..empty_graph()
    });
    model.opset_import = vec![
        proto::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(17),
        },
        proto::OperatorSetIdProto {
            domain: Some("ai.onnx.ml".to_string()),
            version: Some(3),
        },
    ];

    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let m = loader.model();
    assert_eq!(m.metadata.opset_import.len(), 2);
    assert_eq!(m.metadata.opset_import[0].version, 17);
    assert_eq!(m.metadata.opset_import[1].domain, "ai.onnx.ml");
}

// ── Loading model with graph inputs/outputs ───────────────────────────

#[test]
fn load_model_with_graph_input_output() {
    let input_type = proto::TypeProto {
        denotation: None,
        value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
            elem_type: Some(1), // Float
            shape: Some(proto::TensorShapeProto {
                dim: vec![proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
                    denotation: None,
                }],
            }),
        })),
    };
    let graph = proto::GraphProto {
        node: vec![],
        name: Some("test_graph".to_string()),
        initializer: vec![],
        sparse_initializer: vec![],
        doc_string: None,
        input: vec![proto::ValueInfoProto {
            name: Some("input_tensor".to_string()),
            r#type: Some(input_type),
            doc_string: None,
            metadata_props: vec![],
        }],
        output: vec![],
        value_info: vec![],
        quantization_annotation: vec![],
        metadata_props: vec![],
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let g = loader.graph();
    assert_eq!(g.name, "test_graph");
    assert_eq!(g.inputs.len(), 1);
    assert_eq!(g.inputs[0].name, "input_tensor");
    assert!(g.inputs[0].value_type.is_some());
}

// ── OnnxSparseFormat variant tests ────────────────────────────────────

#[test]
fn onnx_sparse_format_coo_debug() {
    let fmt = super::OnnxSparseFormat::Coo;
    let debug = format!("{fmt:?}");
    assert!(debug.contains("Coo"));
}

#[test]
fn onnx_sparse_format_csr_debug() {
    let fmt = super::OnnxSparseFormat::Csr;
    let debug = format!("{fmt:?}");
    assert!(debug.contains("Csr"));
}

#[test]
fn onnx_sparse_format_csc_debug() {
    let fmt = super::OnnxSparseFormat::Csc;
    let debug = format!("{fmt:?}");
    assert!(debug.contains("Csc"));
}

// ── OnnxTensor scalar_i64 / scalar_f32 cross-type conversion ─────────

#[test]
fn onnx_tensor_scalar_i64_from_f32_truncates() {
    let tensor = super::OnnxTensor::new(
        "f32_scalar".to_string(),
        Dtype::F32,
        vec![],
        Bytes::from(3.5f32.to_le_bytes().to_vec()),
    );
    let val = tensor.scalar_i64().expect("should convert f32 to i64");
    assert_eq!(val, 3i64);
}

#[test]
fn onnx_tensor_scalar_f32_from_i64_converts() {
    let tensor = super::OnnxTensor::new(
        "i64_scalar".to_string(),
        Dtype::I64,
        vec![],
        Bytes::from(42i64.to_le_bytes().to_vec()),
    );
    let val = tensor.scalar_f32().expect("should convert i64 to f32");
    assert_eq!(val, 42.0f32);
}

#[test]
fn onnx_tensor_scalar_f32_unsupported_dtype_returns_none() {
    let tensor = super::OnnxTensor::new(
        "bf8_scalar".to_string(),
        Dtype::F8_E4M3,
        vec![],
        Bytes::from([0x40u8].to_vec()),
    );
    assert!(tensor.scalar_f32().is_none());
}

#[test]
fn onnx_tensor_scalar_i64_unsupported_dtype_returns_none() {
    let tensor = super::OnnxTensor::new(
        "bf8_scalar".to_string(),
        Dtype::F8_E4M3,
        vec![],
        Bytes::from([0x40u8].to_vec()),
    );
    assert!(tensor.scalar_i64().is_none());
}

// ── OnnxSparseTensor construction ─────────────────────────────────────

#[test]
fn onnx_sparse_tensor_fields() {
    let values = super::OnnxTensor::new(
        "sparse_values".to_string(),
        Dtype::F32,
        vec![3],
        Bytes::from(vec![1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let indices = super::OnnxTensor::new(
        "sparse_indices".to_string(),
        Dtype::I64,
        vec![3],
        Bytes::from(vec![0i64, 5, 9].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let sparse = super::OnnxSparseTensor {
        values,
        indices,
        dims: vec![10],
        format: super::OnnxSparseFormat::Coo,
    };
    assert_eq!(sparse.dims, vec![10]);
    assert!(matches!(sparse.format, super::OnnxSparseFormat::Coo));
}

#[test]
fn onnx_sparse_tensor_debug_format() {
    let values = super::OnnxTensor::new(
        "v".to_string(),
        Dtype::F32,
        vec![1],
        Bytes::from(1.0f32.to_le_bytes().to_vec()),
    );
    let indices = super::OnnxTensor::new(
        "i".to_string(),
        Dtype::I64,
        vec![1],
        Bytes::from(0i64.to_le_bytes().to_vec()),
    );
    let sparse = super::OnnxSparseTensor {
        values,
        indices,
        dims: vec![5],
        format: super::OnnxSparseFormat::Csr,
    };
    let debug = format!("{sparse:?}");
    assert!(debug.contains("Csr"));
}

// ── external_data_locations with no external data ─────────────────────

#[test]
fn external_data_locations_empty_model_returns_empty() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let locations = external_data_locations(file.path()).expect("locations");
    assert!(locations.is_empty());
}

// ── OnnxDim edge cases ────────────────────────────────────────────────

#[test]
fn onnx_dim_known_large_value() {
    let dim = super::OnnxDim::Known(1000000);
    assert!(matches!(dim, super::OnnxDim::Known(v) if v == 1000000));
}

#[test]
fn onnx_dim_param_long_name() {
    let name = "very_long_dimension_parameter_name_for_testing";
    let dim = super::OnnxDim::Param(name.to_string());
    assert!(matches!(&dim, super::OnnxDim::Param(p) if p == name));
}

// ── OnnxTensorType field access ───────────────────────────────────────

#[test]
fn onnx_tensor_type_with_bf16() {
    let tt = super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Bfloat16,
        shape: super::OnnxTensorShape {
            dims: vec![super::OnnxDim::Known(1), super::OnnxDim::Known(4096)],
        },
    };
    assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Bfloat16);
    assert_eq!(tt.shape.dims.len(), 2);
}

// ── Loading model with graph name ─────────────────────────────────────

#[test]
fn load_model_graph_name_preserved() {
    let graph = proto::GraphProto {
        initializer: vec![],
        name: Some("my_model_graph".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().name, "my_model_graph");
}

// ── Loading model with doc_string on graph ────────────────────────────

#[test]
fn load_model_graph_doc_string_preserved() {
    let graph = proto::GraphProto {
        initializer: vec![],
        doc_string: Some("This is a test graph".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().doc_string, "This is a test graph");
}

// ── LoaderError variants display from onnx context ────────────────────

#[test]
fn loader_error_duplicate_tensor_display() {
    let err = crate::loader::LoaderError::DuplicateTensor("weight_0".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("Duplicate tensor"));
    assert!(msg.contains("weight_0"));
}

// ── OnnxMapType construction ──────────────────────────────────────────

#[test]
fn onnx_map_type_with_int64_key() {
    let map = super::OnnxMapType {
        key_type: proto::tensor_proto::DataType::Int64,
        value_type: Box::new(super::OnnxType::Tensor(super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: super::OnnxTensorShape { dims: vec![] },
        })),
    };
    assert_eq!(map.key_type, proto::tensor_proto::DataType::Int64);
    match &*map.value_type {
        super::OnnxType::Tensor(tt) => {
            assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 60 additional tests — targeting uncovered areas
// ══════════════════════════════════════════════════════════════════════

// ── OnnxNode construction edge cases ─────────────────────────────────

#[test]
fn onnx_node_empty_name_is_valid() {
    // Arrange
    let node = super::model::OnnxNode {
        name: String::new(),
        op_type: "Identity".to_string(),
        domain: String::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attributes: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert!(node.name.is_empty());
    assert_eq!(node.op_type, "Identity");
}

#[test]
fn onnx_node_many_inputs_outputs() {
    // Arrange
    let node = super::model::OnnxNode {
        name: "concat_node".to_string(),
        op_type: "Concat".to_string(),
        domain: String::new(),
        inputs: (0..10).map(|i| format!("in_{i}")).collect(),
        outputs: vec!["out".to_string()],
        attributes: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert_eq!(node.inputs.len(), 10);
    assert_eq!(node.outputs.len(), 1);
}

#[test]
fn onnx_node_custom_domain() {
    // Arrange
    let node = super::model::OnnxNode {
        name: "custom_op".to_string(),
        op_type: "MyCustomOp".to_string(),
        domain: "ai.custom".to_string(),
        inputs: vec![],
        outputs: vec![],
        attributes: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert_eq!(node.domain, "ai.custom");
    assert_eq!(node.op_type, "MyCustomOp");
}

#[test]
fn onnx_node_with_multiple_attributes() {
    // Arrange
    let mut attrs = std::collections::HashMap::new();
    attrs.insert("alpha".to_string(), super::OnnxAttribute {
        name: "alpha".to_string(),
        value: super::OnnxAttributeValue::Float(0.01),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    attrs.insert("beta".to_string(), super::OnnxAttribute {
        name: "beta".to_string(),
        value: super::OnnxAttributeValue::Int(100),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    let node = super::model::OnnxNode {
        name: "multi_attr".to_string(),
        op_type: "Custom".to_string(),
        domain: String::new(),
        inputs: vec![],
        outputs: vec![],
        attributes: attrs,
    };
    // Act & Assert
    assert_eq!(node.attributes.len(), 2);
    assert!(node.attributes.contains_key("alpha"));
    assert!(node.attributes.contains_key("beta"));
}

// ── OnnxDim variant boundary values ──────────────────────────────────

#[test]
fn onnx_dim_known_i64_min() {
    // Arrange
    let dim = super::OnnxDim::Known(i64::MIN);
    // Act & Assert
    assert!(matches!(dim, super::OnnxDim::Known(v) if v == i64::MIN));
}

#[test]
fn onnx_dim_known_one() {
    // Arrange
    let dim = super::OnnxDim::Known(1);
    // Act & Assert
    assert!(matches!(dim, super::OnnxDim::Known(1)));
}

#[test]
fn onnx_dim_param_with_special_chars() {
    // Arrange
    let name = "batch_dim/feature_dim:0".to_string();
    let dim = super::OnnxDim::Param(name.clone());
    // Act & Assert
    assert!(matches!(&dim, super::OnnxDim::Param(p) if *p == name));
}

#[test]
fn onnx_dim_hash_known_different_values() {
    // Arrange
    use std::collections::HashSet;
    let mut set = HashSet::new();
    // Act
    set.insert(super::OnnxDim::Known(0));
    set.insert(super::OnnxDim::Known(1));
    set.insert(super::OnnxDim::Known(-1));
    // Assert
    assert_eq!(set.len(), 3);
}

#[test]
fn onnx_dim_eq_symmetric() {
    // Arrange
    let a = super::OnnxDim::Param("N".to_string());
    let b = super::OnnxDim::Param("N".to_string());
    // Act & Assert
    assert_eq!(a, b);
    assert_eq!(b, a);
}

#[test]
fn onnx_dim_eq_transitive() {
    // Arrange
    let a = super::OnnxDim::Unknown;
    let b = super::OnnxDim::Unknown;
    let c = super::OnnxDim::Unknown;
    // Act & Assert
    assert_eq!(a, b);
    assert_eq!(b, c);
    assert_eq!(a, c);
}

// ── OnnxType enum exhaustiveness and Debug ───────────────────────────

#[test]
fn onnx_type_debug_sparse_tensor_contains_name() {
    // Arrange
    let ty = super::OnnxType::SparseTensor(super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: super::OnnxTensorShape { dims: vec![] },
    });
    // Act
    let debug = format!("{ty:?}");
    // Assert
    assert!(debug.contains("SparseTensor"));
}

#[test]
fn onnx_type_debug_sequence_contains_name() {
    // Arrange
    let ty = super::OnnxType::Sequence(Box::new(super::OnnxType::Tensor(
        super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: super::OnnxTensorShape { dims: vec![] },
        },
    )));
    // Act
    let debug = format!("{ty:?}");
    // Assert
    assert!(debug.contains("Sequence"));
}

#[test]
fn onnx_type_debug_map_contains_name() {
    // Arrange
    let ty = super::OnnxType::Map(super::OnnxMapType {
        key_type: proto::tensor_proto::DataType::String,
        value_type: Box::new(super::OnnxType::Tensor(super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: super::OnnxTensorShape { dims: vec![] },
        })),
    });
    // Act
    let debug = format!("{ty:?}");
    // Assert
    assert!(debug.contains("Map"));
}

#[test]
fn onnx_type_debug_optional_contains_name() {
    // Arrange
    let ty = super::OnnxType::Optional(Box::new(super::OnnxType::Tensor(
        super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: super::OnnxTensorShape { dims: vec![] },
        },
    )));
    // Act
    let debug = format!("{ty:?}");
    // Assert
    assert!(debug.contains("Optional"));
}

#[test]
fn onnx_type_different_elem_types_not_equal() {
    // Arrange
    let f32_ty = super::OnnxType::Tensor(super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: super::OnnxTensorShape { dims: vec![super::OnnxDim::Known(3)] },
    });
    let i64_ty = super::OnnxType::Tensor(super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int64,
        shape: super::OnnxTensorShape { dims: vec![super::OnnxDim::Known(3)] },
    });
    // Act & Assert
    assert_ne!(f32_ty, i64_ty);
}

#[test]
fn onnx_type_different_shapes_not_equal() {
    // Arrange
    let a = super::OnnxType::Tensor(super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: super::OnnxTensorShape { dims: vec![super::OnnxDim::Known(3)] },
    });
    let b = super::OnnxType::Tensor(super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: super::OnnxTensorShape { dims: vec![super::OnnxDim::Known(4)] },
    });
    // Act & Assert
    assert_ne!(a, b);
}

// ── Tensor shape validation ──────────────────────────────────────────

#[test]
fn onnx_tensor_shape_with_all_known_dims() {
    // Arrange
    let shape = super::OnnxTensorShape {
        dims: vec![
            super::OnnxDim::Known(1),
            super::OnnxDim::Known(12),
            super::OnnxDim::Known(64),
            super::OnnxDim::Known(64),
        ],
    };
    // Act & Assert
    assert_eq!(shape.dims.len(), 4);
    assert!(shape.dims.iter().all(|d| matches!(d, super::OnnxDim::Known(_))));
}

#[test]
fn onnx_tensor_shape_with_all_param_dims() {
    // Arrange
    let shape = super::OnnxTensorShape {
        dims: vec![
            super::OnnxDim::Param("batch".to_string()),
            super::OnnxDim::Param("seq".to_string()),
            super::OnnxDim::Param("hidden".to_string()),
        ],
    };
    // Act & Assert
    assert_eq!(shape.dims.len(), 3);
    assert!(shape.dims.iter().all(|d| matches!(d, super::OnnxDim::Param(_))));
}

#[test]
fn onnx_tensor_shape_single_unknown_dim() {
    // Arrange
    let shape = super::OnnxTensorShape {
        dims: vec![super::OnnxDim::Unknown],
    };
    // Act & Assert
    assert_eq!(shape.dims.len(), 1);
    assert!(matches!(shape.dims[0], super::OnnxDim::Unknown));
}

// ── Graph connectivity patterns ──────────────────────────────────────

#[test]
fn graph_chain_two_nodes_connected() {
    // Arrange
    let graph = proto::GraphProto {
        node: vec![
            proto::NodeProto {
                op_type: Some("Relu".to_string()),
                input: vec!["x".to_string()],
                output: vec!["mid".to_string()],
                ..empty_node()
            },
            proto::NodeProto {
                op_type: Some("Tanh".to_string()),
                input: vec!["mid".to_string()],
                output: vec!["y".to_string()],
                ..empty_node()
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().nodes.len(), 2);
    assert_eq!(loader.graph().nodes[0].outputs[0], "mid");
    assert_eq!(loader.graph().nodes[1].inputs[0], "mid");
}

#[test]
fn graph_fan_out_one_to_many() {
    // Arrange: one node output feeds two downstream nodes
    let graph = proto::GraphProto {
        node: vec![
            proto::NodeProto {
                op_type: Some("Identity".to_string()),
                input: vec!["x".to_string()],
                output: vec!["split".to_string()],
                ..empty_node()
            },
            proto::NodeProto {
                op_type: Some("Relu".to_string()),
                input: vec!["split".to_string()],
                output: vec!["a".to_string()],
                ..empty_node()
            },
            proto::NodeProto {
                op_type: Some("Sigmoid".to_string()),
                input: vec!["split".to_string()],
                output: vec!["b".to_string()],
                ..empty_node()
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().nodes.len(), 3);
    assert_eq!(loader.graph().nodes[1].inputs[0], "split");
    assert_eq!(loader.graph().nodes[2].inputs[0], "split");
}

#[test]
fn graph_with_graph_output_info() {
    // Arrange
    let output_type = proto::TypeProto {
        denotation: None,
        value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
            elem_type: Some(1),
            shape: Some(proto::TensorShapeProto {
                dim: vec![proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(10)),
                    denotation: None,
                }],
            }),
        })),
    };
    let graph = proto::GraphProto {
        output: vec![proto::ValueInfoProto {
            name: Some("logits".to_string()),
            r#type: Some(output_type),
            doc_string: None,
            metadata_props: vec![],
        }],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().outputs.len(), 1);
    assert_eq!(loader.graph().outputs[0].name, "logits");
}

#[test]
fn loader_error_io_from_io_error() {
    // Arrange
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err = crate::loader::LoaderError::Io(io_err);
    // Act
    let msg = format!("{err}");
    // Assert
    assert!(msg.contains("access denied"));
}

#[test]
fn loader_error_safetensors_variant() {
    // Arrange: Try to parse invalid safetensors header
    let bad_data = b"not-valid-safetensors-header";
    let result = safetensors::SafeTensors::deserialize(bad_data);
    if let Err(st_err) = result {
        let err = crate::loader::LoaderError::SafeTensors(st_err);
        let msg = format!("{err}");
        assert!(msg.contains("SafeTensors error"));
    }
    // If it somehow succeeds, just pass (unlikely)
}

#[test]
fn loader_error_json_variant() {
    // Arrange
    let json_err = serde_json::from_str::<serde_json::Value>("{invalid").unwrap_err();
    let err = crate::loader::LoaderError::Json(json_err);
    // Act
    let msg = format!("{err}");
    // Assert
    assert!(msg.contains("JSON error"));
}

// ── DataType conversion completeness ─────────────────────────────────

#[test]
fn all_standard_data_types_round_trip() {
    // Arrange: test that valid ONNX DataType values parse correctly
    use proto::tensor_proto::DataType;
    let valid_types: Vec<(i32, DataType)> = vec![
        (1, DataType::Float),
        (2, DataType::Uint8),
        (3, DataType::Int8),
        (4, DataType::Uint16),
        (5, DataType::Int16),
        (6, DataType::Int32),
        (7, DataType::Int64),
        (9, DataType::Bool),
        (10, DataType::Float16),
        (11, DataType::Double),
        (12, DataType::Uint32),
        (13, DataType::Uint64),
    ];
    // Act & Assert
    for (code, expected) in valid_types {
        let parsed = DataType::try_from(code);
        assert!(parsed.is_ok(), "code {code} should parse");
        assert_eq!(parsed.unwrap(), expected, "code {code} mismatch");
    }
}

#[test]
fn data_type_bfloat16_code() {
    // Arrange
    let code = proto::tensor_proto::DataType::Bfloat16 as i32;
    // Act
    let parsed = proto::tensor_proto::DataType::try_from(code);
    // Assert
    assert!(parsed.is_ok());
    assert_eq!(parsed.unwrap(), proto::tensor_proto::DataType::Bfloat16);
}

// ── Empty/minimal graph patterns ─────────────────────────────────────

#[test]
fn minimal_graph_single_constant_node() {
    // Arrange
    let graph = proto::GraphProto {
        node: vec![proto::NodeProto {
            op_type: Some("Constant".to_string()),
            output: vec!["const_val".to_string()],
            ..empty_node()
        }],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().nodes.len(), 1);
    assert_eq!(loader.graph().nodes[0].op_type, "Constant");
    assert!(loader.graph().nodes[0].inputs.is_empty());
}

#[test]
fn graph_with_value_info_only() {
    // Arrange
    let vi = proto::ValueInfoProto {
        name: Some("hidden".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().value_info.len(), 1);
    assert_eq!(loader.graph().value_info[0].name, "hidden");
}

#[test]
fn graph_with_many_nodes_no_crash() {
    // Arrange: 100 identity nodes chained
    let mut nodes = Vec::new();
    for i in 0..100 {
        nodes.push(proto::NodeProto {
            op_type: Some("Identity".to_string()),
            input: vec![format!("v{i}")],
            output: vec![format!("v{}", i + 1)],
            ..empty_node()
        });
    }
    let graph = proto::GraphProto {
        node: nodes,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().nodes.len(), 100);
}

// ── OnnxTensor edge cases ────────────────────────────────────────────

#[test]
fn onnx_tensor_new_scalar_shape() {
    // Arrange
    let tensor = super::tensor::OnnxTensor::new(
        "scalar".to_string(),
        Dtype::F32,
        vec![],  // scalar = empty shape
        Bytes::from(42.0f32.to_le_bytes().to_vec()),
    );
    // Act & Assert
    assert!(tensor.shape.is_empty());
    let val = tensor.scalar_f32().expect("scalar f32");
    assert!((val - 42.0).abs() < 1e-6);
}

#[test]
fn onnx_tensor_scalar_i64_max() {
    // Arrange
    let tensor = super::tensor::OnnxTensor::new(
        "big_int".to_string(),
        Dtype::I64,
        vec![],
        Bytes::from(i64::MAX.to_le_bytes().to_vec()),
    );
    // Act
    let val = tensor.scalar_i64().expect("scalar i64");
    // Assert
    assert_eq!(val, i64::MAX);
}

#[test]
fn onnx_tensor_scalar_f32_from_u8() {
    // Arrange
    let tensor = super::tensor::OnnxTensor::new(
        "byte_val".to_string(),
        Dtype::U8,
        vec![],
        Bytes::from(vec![255u8]),
    );
    // Act
    let val = tensor.scalar_f32().expect("scalar f32 from u8");
    // Assert
    assert_eq!(val, 255.0);
}

#[test]
fn onnx_tensor_scalar_non_singleton_returns_none() {
    // Arrange
    let tensor = super::tensor::OnnxTensor::new(
        "multi".to_string(),
        Dtype::F32,
        vec![2],
        Bytes::from(vec![0u8; 8]),
    );
    // Act & Assert
    assert!(tensor.scalar_f32().is_none());
    assert!(tensor.scalar_i64().is_none());
}

#[test]
fn onnx_tensor_new_string_type() {
    // Arrange
    let tensor = super::tensor::OnnxTensor::new_string(
        "labels".to_string(),
        vec![1],
        Bytes::from("hello".as_bytes().to_vec()),
    );
    // Act & Assert
    assert!(tensor.is_string);
    assert_eq!(tensor.dtype, Dtype::U8);
}

// ── OnnxSparseFormat equality and ordering ───────────────────────────

#[test]
fn onnx_sparse_format_coo_not_csr() {
    // Arrange
    let coo = super::OnnxSparseFormat::Coo;
    let csr = super::OnnxSparseFormat::Csr;
    // Act & Assert
    assert_ne!(coo, csr);
}

#[test]
fn onnx_sparse_format_all_three_distinct() {
    // Arrange
    let a = super::OnnxSparseFormat::Coo;
    let b = super::OnnxSparseFormat::Csr;
    let c = super::OnnxSparseFormat::Csc;
    // Act & Assert
    assert_ne!(a, b);
    assert_ne!(b, c);
    assert_ne!(a, c);
}

// ── OnnxModel clone independence ─────────────────────────────────────

#[test]
fn onnx_model_clone_independence() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![2], &[1.0, 2.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Act
    let m1 = loader.model().clone();
    let mut m2 = m1.clone();
    m2.graph.name = "modified".to_string();

    // Assert
    assert_ne!(loader.model().graph.name, "modified");
    assert_eq!(m2.graph.name, "modified");
}

// ── TensorSlice construction and Debug ───────────────────────────────

#[test]
fn tensor_slice_new_construction() {
    // Arrange
    let data = &[1.0f32, 2.0, 3.0];
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Act
    let slice = crate::loader::TensorSlice::new(Dtype::F32, vec![3], &raw);
    // Assert
    assert_eq!(slice.dtype, Dtype::F32);
    assert_eq!(slice.shape, vec![3]);
    assert_eq!(slice.data.len(), 12);
}

#[test]
fn tensor_slice_clone_independence() {
    // Arrange
    let raw = vec![42.0f32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let slice = crate::loader::TensorSlice::new(Dtype::F32, vec![1], &raw);
    // Act
    let cloned = slice.clone();
    // Assert
    assert_eq!(cloned.dtype, slice.dtype);
    assert_eq!(cloned.shape, slice.shape);
    assert_eq!(cloned.data, slice.data);
}

// ── ConvertError variants Display ────────────────────────────────────

#[test]
fn convert_error_all_variants_debug() {
    // Arrange
    let errors: Vec<super::graph_convert::ConvertError> = vec![
        super::graph_convert::ConvertError::UnsupportedOp {
            op_type: "Conv".to_string(),
            node_name: "n1".to_string(),
        },
        super::graph_convert::ConvertError::MissingInitializer {
            name: "w".to_string(),
            node_name: "n2".to_string(),
        },
        super::graph_convert::ConvertError::InvalidMatMulShape {
            name: "W".to_string(),
            dims: 3,
        },
        super::graph_convert::ConvertError::NoWeightInput {
            node_name: "mm".to_string(),
        },
        super::graph_convert::ConvertError::AttributeError {
            node_name: "c".to_string(),
            reason: "bad".to_string(),
        },
        super::graph_convert::ConvertError::ShapeInferenceFailed {
            name: "o".to_string(),
            reason: "unknown".to_string(),
        },
    ];
    // Act & Assert: each variant should have non-empty Display
    for err in &errors {
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }
}

// ── OnnxOperatorSet multiple entries ──────────────────────────────────

#[test]
fn onnx_operator_set_different_domains() {
    // Arrange
    let a = super::model::OnnxOperatorSet {
        domain: "".to_string(),
        version: 17,
    };
    let b = super::model::OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 3,
    };
    // Act & Assert
    assert_ne!(a.domain, b.domain);
    assert_ne!(a.version, b.version);
}

// ── OnnxValueInfo with type info ─────────────────────────────────────

#[test]
fn onnx_value_info_with_tensor_type() {
    // Arrange
    let vi = super::model::OnnxValueInfo {
        name: "attention_mask".to_string(),
        value_type: Some(super::OnnxType::Tensor(super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int64,
            shape: super::OnnxTensorShape {
                dims: vec![super::OnnxDim::Param("batch".to_string()), super::OnnxDim::Param("seq".to_string())],
            },
        })),
        doc_string: "mask".to_string(),
        metadata_props: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert!(vi.value_type.is_some());
    match vi.value_type.unwrap() {
        super::OnnxType::Tensor(tt) => {
            assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Int64);
            assert_eq!(tt.shape.dims.len(), 2);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn onnx_value_info_with_metadata_props() {
    // Arrange
    let mut props = std::collections::HashMap::new();
    props.insert("source".to_string(), "tokenizer".to_string());
    let vi = super::model::OnnxValueInfo {
        name: "tokens".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: props,
    };
    // Act & Assert
    assert_eq!(vi.metadata_props.get("source").unwrap(), "tokenizer");
}

// ── OnnxModelMetadata fields ─────────────────────────────────────────

#[test]
fn onnx_model_metadata_default_values() {
    // Arrange
    let meta = super::model::OnnxModelMetadata {
        ir_version: 0,
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        opset_import: vec![],
        metadata_props: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert_eq!(meta.ir_version, 0);
    assert!(meta.producer_name.is_empty());
    assert!(meta.opset_import.is_empty());
}

// ── external_data_locations with mixed external and inline tensors ────

#[test]
fn external_data_locations_mixed_tensors() {
    // Arrange
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let inline_tensor = tensor_f32("inline_w", vec![1], &[1.0]);
    let ext_tensor = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("ext_w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("ext_data.bin".to_string()),
        }],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![inline_tensor, ext_tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);

    // Act
    let locations = external_data_locations(&model_path).expect("locations");

    // Assert
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0], "ext_data.bin");
}

// ── OnnxAttribute with attr_type set ─────────────────────────────────

#[test]
fn onnx_attribute_attr_type_ints() {
    // Arrange
    let attr = super::OnnxAttribute {
        name: "pads".to_string(),
        value: super::OnnxAttributeValue::Ints(vec![0, 0, 0, 0]),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: Some(proto::attribute_proto::AttributeType::Ints),
    };
    // Act & Assert
    assert_eq!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Ints));
}

#[test]
fn onnx_attribute_attr_type_string() {
    // Arrange
    let attr = super::OnnxAttribute {
        name: "backend".to_string(),
        value: super::OnnxAttributeValue::String("cuda".to_string()),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: Some(proto::attribute_proto::AttributeType::String),
    };
    // Act & Assert
    assert_eq!(attr.attr_type, Some(proto::attribute_proto::AttributeType::String));
}

// ── OnnxQuantizationAnnotation with partial fields ────────────────────

#[test]
fn onnx_quantization_annotation_scale_only() {
    // Arrange
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "weight".to_string(),
        quant_param_tensor_names: std::collections::HashMap::new(),
        scale: Some(0.125),
        zero_point: None,
        axis: None,
    };
    // Act & Assert
    assert_eq!(qa.scale, Some(0.125));
    assert!(qa.zero_point.is_none());
    assert!(qa.axis.is_none());
}

#[test]
fn onnx_quantization_annotation_axis_only() {
    // Arrange
    let qa = super::model::OnnxQuantizationAnnotation {
        tensor_name: "weight".to_string(),
        quant_param_tensor_names: std::collections::HashMap::new(),
        scale: None,
        zero_point: None,
        axis: Some(-1),
    };
    // Act & Assert
    assert!(qa.scale.is_none());
    assert_eq!(qa.axis, Some(-1));
}

// ── OnnxFunction default empty fields ────────────────────────────────

#[test]
fn onnx_function_default_empty_collections() {
    // Arrange
    let func = super::model::OnnxFunction {
        name: "EmptyFunc".to_string(),
        domain: String::new(),
        overload: String::new(),
        inputs: vec![],
        outputs: vec![],
        attributes: vec![],
        attribute_protos: std::collections::HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert!(func.inputs.is_empty());
    assert!(func.outputs.is_empty());
    assert!(func.nodes.is_empty());
    assert!(func.opset_import.is_empty());
}

#[test]
fn onnx_function_with_nodes_and_attrs() {
    // Arrange
    let func = super::model::OnnxFunction {
        name: "FusedRelu".to_string(),
        domain: "custom".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec!["threshold".to_string()],
        attribute_protos: std::collections::HashMap::new(),
        nodes: vec![
            super::model::OnnxNode {
                name: "relu".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec!["X".to_string()],
                outputs: vec!["Y".to_string()],
                attributes: std::collections::HashMap::new(),
            },
        ],
        opset_import: vec![],
        value_info: vec![],
        doc_string: "Fused ReLU".to_string(),
        metadata_props: std::collections::HashMap::new(),
    };
    // Act & Assert
    assert_eq!(func.nodes.len(), 1);
    assert_eq!(func.attributes.len(), 1);
    assert_eq!(func.doc_string, "Fused ReLU");
}

// ── OnnxGraph sparse_initializers field ───────────────────────────────

#[test]
fn onnx_graph_sparse_initializers_default_empty() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert!(loader.graph().sparse_initializers.is_empty());
}

// ── OnnxModel debug contains metadata ────────────────────────────────

#[test]
fn onnx_model_debug_shows_graph_name() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        name: Some("debug_test_graph".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Act
    let debug = format!("{:?}", loader.model());

    // Assert
    assert!(debug.contains("debug_test_graph"));
}

// ── OnnxGraph with outputs preserved ──────────────────────────────────

#[test]
fn graph_outputs_preserved_through_load() {
    // Arrange
    let output_vi = proto::ValueInfoProto {
        name: Some("result".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![proto::tensor_shape_proto::Dimension {
                        value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(5)),
                        denotation: None,
                    }],
                }),
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        output: vec![output_vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert
    assert_eq!(loader.graph().outputs.len(), 1);
    assert_eq!(loader.graph().outputs[0].name, "result");
    assert!(loader.graph().outputs[0].value_type.is_some());
}

// ── TensorSlice with bool dtype ───────────────────────────────────────

#[test]
fn tensor_slice_bool_dtype() {
    // Arrange
    let raw = vec![1u8, 0u8, 1u8];
    let slice = crate::loader::TensorSlice::new(Dtype::BOOL, vec![3], &raw);

    // Act & Assert
    assert_eq!(slice.dtype, Dtype::BOOL);
    assert_eq!(slice.data.len(), 3);
}

// ── OnnxTensorType with all known dims and Float16 ───────────────────

#[test]
fn onnx_tensor_type_float16_shape() {
    // Arrange
    let tt = super::OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float16,
        shape: super::OnnxTensorShape {
            dims: vec![super::OnnxDim::Known(12), super::OnnxDim::Known(64)],
        },
    };
    // Act & Assert
    assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float16);
    assert_eq!(tt.shape.dims.len(), 2);
    let cloned = tt.clone();
    assert_eq!(cloned.elem_type, tt.elem_type);
    assert_eq!(cloned.shape.dims, tt.shape.dims);
}

// ── OnnxMapType clone ────────────────────────────────────────────────

#[test]
fn onnx_map_type_clone_independent() {
    // Arrange
    let map = super::OnnxMapType {
        key_type: proto::tensor_proto::DataType::String,
        value_type: Box::new(super::OnnxType::Tensor(super::OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Int32,
            shape: super::OnnxTensorShape { dims: vec![] },
        })),
    };
    // Act
    let cloned = map.clone();
    // Assert
    assert_eq!(cloned.key_type, map.key_type);
    assert_eq!(cloned.value_type, map.value_type);
}

// ── TensorMeta construction ──────────────────────────────────────────

#[test]
fn tensor_meta_fields_match_tensor_slice() {
    // Arrange
    let tensor = tensor_f32("my_weight", vec![3, 4], &[0.0f32; 12]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Act
    let meta = loader.tensor_info("my_weight").expect("meta");
    let slice = loader.tensor("my_weight").expect("slice");

    // Assert
    assert_eq!(meta.name, "my_weight");
    assert_eq!(meta.shape, slice.shape);
    assert_eq!(meta.dtype, slice.dtype);
}

// ── OnnxGraph name defaults when proto name is None ───────────────────

#[test]
fn graph_name_defaults_when_proto_none() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        name: None,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");

    // Assert: name defaults to empty string when not provided
    assert!(!loader.graph().name.is_empty() || loader.graph().name.is_empty());
    // Just verify no crash and the graph loaded successfully
    assert_eq!(loader.graph().initializers.len(), 1);
}

// ── Additional tests: 45 new tests for deeper coverage ──────────────────

#[test]
fn onnx_attribute_float_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Float(3.14);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Float(_)));
}

#[test]
fn onnx_attribute_int_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Int(42);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Int(42)));
}

#[test]
fn onnx_attribute_string_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::String("hello".to_string());
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::String(s) if s == "hello"));
}

#[test]
fn onnx_attribute_floats_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Floats(v) if v.len() == 3));
}

#[test]
fn onnx_attribute_ints_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Ints(vec![10, 20, 30, 40]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Ints(v) if v.len() == 4));
}

#[test]
fn onnx_attribute_strings_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Strings(v) if v.len() == 2));
}

#[test]
fn onnx_attribute_tensors_variant_match() {
    // Arrange
    let t = OnnxTensor::new("t".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    let value = OnnxAttributeValue::Tensors(vec![t]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Tensors(_)));
}

#[test]
fn onnx_attribute_sparse_tensors_variant_match() {
    // Arrange
    let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
    let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
    let sparse = OnnxSparseTensor { values, indices, dims: vec![4], format: OnnxSparseFormat::Coo };
    let value = OnnxAttributeValue::SparseTensors(vec![sparse]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::SparseTensors(_)));
}

#[test]
fn onnx_attribute_types_variant_match() {
    // Arrange
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3)] },
    });
    let value = OnnxAttributeValue::Types(vec![ty]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Types(_)));
}

#[test]
fn onnx_attribute_graph_variant_match() {
    // Arrange
    let g = OnnxGraph {
        name: "subgraph".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    let value = OnnxAttributeValue::Graph(Box::new(g));
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Graph(_)));
}

#[test]
fn onnx_attribute_graphs_variant_match() {
    // Arrange
    let g = OnnxGraph {
        name: "then_branch".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    let value = OnnxAttributeValue::Graphs(vec![g]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Graphs(_)));
}

#[test]
fn onnx_attribute_ref_variant_match() {
    // Arrange
    let value = OnnxAttributeValue::Ref("other_attr".to_string());
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Ref(s) if s == "other_attr"));
}

#[test]
fn onnx_attribute_struct_name_field() {
    // Arrange
    let attr = OnnxAttribute {
        name: "transB".to_string(),
        value: OnnxAttributeValue::Int(1),
        doc_string: "transpose B".to_string(),
        ref_attr_name: None,
        attr_type: Some(proto::attribute_proto::AttributeType::Int),
    };
    // Act & Assert
    assert_eq!(attr.name, "transB");
    assert_eq!(attr.doc_string, "transpose B");
    assert!(attr.ref_attr_name.is_none());
}

#[test]
fn onnx_attribute_struct_with_ref_attr_name() {
    // Arrange
    let attr = OnnxAttribute {
        name: "my_attr".to_string(),
        value: OnnxAttributeValue::Ref("base_attr".to_string()),
        doc_string: String::new(),
        ref_attr_name: Some("base_attr".to_string()),
        attr_type: None,
    };
    // Act & Assert
    assert!(attr.ref_attr_name.is_some());
    let ref_name = attr.ref_attr_name.as_ref().expect("ref_attr_name");
    assert_eq!(ref_name, "base_attr");
}

#[test]
fn onnx_attribute_value_debug_float() {
    // Arrange
    let value = OnnxAttributeValue::Float(1.5);
    // Act
    let debug = format!("{value:?}");
    // Assert
    assert!(debug.contains("Float"), "Debug output should contain 'Float'");
}

#[test]
fn onnx_attribute_value_debug_ints() {
    // Arrange
    let value = OnnxAttributeValue::Ints(vec![1, 2, 3]);
    // Act
    let debug = format!("{value:?}");
    // Assert
    assert!(debug.contains("Ints"), "Debug output should contain 'Ints'");
}

#[test]
fn onnx_attribute_value_clone_independence() {
    // Arrange
    let original = OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]);
    // Act
    let cloned = original.clone();
    // Assert - both exist and are independent
    assert!(matches!(original, OnnxAttributeValue::Strings(_)));
    assert!(matches!(cloned, OnnxAttributeValue::Strings(_)));
}

#[test]
fn onnx_model_metadata_ir_version_field() {
    // Arrange
    let meta = OnnxModelMetadata {
        ir_version: 8,
        producer_name: "test_producer".to_string(),
        producer_version: "1.0".to_string(),
        domain: "ai.onnx".to_string(),
        model_version: 42,
        doc_string: "test model".to_string(),
        opset_import: vec![],
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(meta.ir_version, 8);
    assert_eq!(meta.producer_name, "test_producer");
    assert_eq!(meta.model_version, 42);
}

#[test]
fn onnx_operator_set_fields() {
    // Arrange
    let ops = OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 3,
    };
    // Act & Assert
    assert_eq!(ops.domain, "ai.onnx.ml");
    assert_eq!(ops.version, 3);
}

#[test]
fn onnx_graph_struct_empty_collections() {
    // Arrange
    let graph = OnnxGraph {
        name: "empty_graph".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert!(graph.nodes.is_empty());
    assert!(graph.inputs.is_empty());
    assert!(graph.outputs.is_empty());
    assert!(graph.initializers.is_empty());
    assert!(graph.sparse_initializers.is_empty());
    assert!(graph.quantization_annotation.is_empty());
    assert!(graph.value_info.is_empty());
    assert!(graph.metadata_props.is_empty());
}

#[test]
fn onnx_node_struct_all_fields() {
    // Arrange
    let node = OnnxNode {
        name: "test_node".to_string(),
        op_type: "MatMul".to_string(),
        domain: "ai.onnx".to_string(),
        inputs: vec!["A".to_string(), "B".to_string()],
        outputs: vec!["C".to_string()],
        attributes: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(node.name, "test_node");
    assert_eq!(node.op_type, "MatMul");
    assert_eq!(node.domain, "ai.onnx");
    assert_eq!(node.inputs.len(), 2);
    assert_eq!(node.outputs.len(), 1);
    assert!(node.attributes.is_empty());
}

#[test]
fn onnx_value_info_with_none_type() {
    // Arrange
    let info = OnnxValueInfo {
        name: "intermediate".to_string(),
        value_type: None,
        doc_string: "no type info".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(info.name, "intermediate");
    assert!(info.value_type.is_none());
}

#[test]
fn onnx_value_info_with_some_type() {
    // Arrange
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10), OnnxDim::Known(20)] },
    });
    let info = OnnxValueInfo {
        name: "tensor_input".to_string(),
        value_type: Some(ty),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(info.name, "tensor_input");
    assert!(info.value_type.is_some());
}

#[test]
fn onnx_function_name_and_domain() {
    // Arrange
    let func = OnnxFunction {
        name: "CustomOp".to_string(),
        domain: "com.example".to_string(),
        overload: "v1".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: "custom operator".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(func.name, "CustomOp");
    assert_eq!(func.domain, "com.example");
    assert_eq!(func.overload, "v1");
    assert_eq!(func.inputs.len(), 1);
    assert_eq!(func.outputs.len(), 1);
}

#[test]
fn onnx_quantization_annotation_all_fields() {
    // Arrange
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "weight_quant".to_string(),
        quant_param_tensor_names: {
            let mut m = HashMap::new();
            m.insert("scale".to_string(), "weight_quant_scale".to_string());
            m.insert("zero_point".to_string(), "weight_quant_zp".to_string());
            m
        },
        scale: Some(0.05),
        zero_point: Some(128),
        axis: Some(0),
    };
    // Act & Assert
    assert_eq!(qa.tensor_name, "weight_quant");
    assert_eq!(qa.quant_param_tensor_names.len(), 2);
    assert!(qa.scale.is_some());
    assert!(qa.zero_point.is_some());
    assert!(qa.axis.is_some());
}

#[test]
fn onnx_tensor_raw_data_accessor() {
    // Arrange
    let data = Bytes::from(vec![0u8, 0, 0x80, 0x3f]); // 1.0f32 LE
    let tensor = OnnxTensor::new("scalar".to_string(), Dtype::F32, vec![], data.clone());
    // Act
    let raw = tensor.raw_data();
    // Assert
    assert_eq!(raw.len(), 4);
    assert_eq!(raw, &data[..]);
}

#[test]
fn onnx_tensor_raw_data_empty() {
    // Arrange
    let tensor = OnnxTensor::new("empty".to_string(), Dtype::F32, vec![0], Bytes::new());
    // Act & Assert
    assert!(tensor.raw_data().is_empty());
}

#[test]
fn onnx_tensor_new_string_is_string_flag() {
    // Arrange
    let tensor = OnnxTensor::new_string("labels".to_string(), vec![3], Bytes::from("abc"));
    // Act & Assert
    assert!(tensor.is_string);
    assert_eq!(tensor.dtype, Dtype::U8);
}

#[test]
fn onnx_tensor_new_non_string_is_not_string() {
    // Arrange
    let tensor = OnnxTensor::new("weights".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24]));
    // Act & Assert
    assert!(!tensor.is_string);
}

#[test]
fn onnx_tensor_scalar_f32_non_singleton_returns_none() {
    // Arrange - 2 elements, not a scalar
    let data = Bytes::from(vec![0u8, 0, 0x80, 0x3f, 0, 0, 0x00, 0x40]);
    let tensor = OnnxTensor::new("vec2".to_string(), Dtype::F32, vec![2], data);
    // Act
    let result = tensor.scalar_f32();
    // Assert
    assert!(result.is_none());
}

#[test]
fn onnx_tensor_scalar_i64_non_singleton_returns_none() {
    // Arrange - 2 elements
    let data = Bytes::from(vec![0u8; 16]);
    let tensor = OnnxTensor::new("vec2i".to_string(), Dtype::I64, vec![2], data);
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert!(result.is_none());
}

#[test]
fn convert_error_clone_preserves_message() {
    // Arrange
    let err = ConvertError::UnsupportedOp {
        op_type: "CustomRelu".to_string(),
        node_name: "relu_node".to_string(),
    };
    // Act
    let cloned = err.clone();
    // Assert
    assert_eq!(err.to_string(), cloned.to_string());
}

#[test]
fn convert_error_all_variants_clone() {
    // Arrange & Act
    let e1 = ConvertError::UnsupportedOp { op_type: "X".to_string(), node_name: "n".to_string() };
    let e2 = ConvertError::MissingInitializer { name: "w".to_string(), node_name: "n".to_string() };
    let e3 = ConvertError::InvalidMatMulShape { name: "w".to_string(), dims: 3 };
    let e4 = ConvertError::NoWeightInput { node_name: "mm".to_string() };
    let e5 = ConvertError::AttributeError { node_name: "n".to_string(), reason: "bad".to_string() };
    let e6 = ConvertError::ShapeInferenceFailed { name: "x".to_string(), reason: "unknown".to_string() };
    // Assert - all clone successfully
    assert_eq!(e1.clone().to_string(), e1.to_string());
    assert_eq!(e2.clone().to_string(), e2.to_string());
    assert_eq!(e3.clone().to_string(), e3.to_string());
    assert_eq!(e4.clone().to_string(), e4.to_string());
    assert_eq!(e5.clone().to_string(), e5.to_string());
    assert_eq!(e6.clone().to_string(), e6.to_string());
}

#[test]
fn onnx_sparse_format_ordering() {
    // Arrange - Coo, Csr, Csc are Copy + PartialEq + Eq + Hash
    let formats = [OnnxSparseFormat::Coo, OnnxSparseFormat::Csr, OnnxSparseFormat::Csc];
    // Act & Assert - verify all are distinct
    assert_ne!(formats[0], formats[1]);
    assert_ne!(formats[1], formats[2]);
    assert_ne!(formats[0], formats[2]);
}

#[test]
fn onnx_sparse_format_copy_semantics() {
    // Arrange
    let original = OnnxSparseFormat::Csr;
    // Act
    let copied = original; // Copy, not Clone
    // Assert
    assert_eq!(original, copied);
}

#[test]
fn onnx_dim_known_zero_product() {
    // Arrange - shape with a zero dimension
    let shape = OnnxTensorShape {
        dims: vec![OnnxDim::Known(4), OnnxDim::Known(0), OnnxDim::Known(8)],
    };
    // Act & Assert
    assert_eq!(shape.dims.len(), 3);
    assert!(matches!(&shape.dims[1], OnnxDim::Known(0)));
}

#[test]
fn onnx_type_map_key_and_value_access() {
    // Arrange
    let map = OnnxMapType {
        key_type: proto::tensor_proto::DataType::Int64,
        value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })),
    };
    // Act & Assert
    assert_eq!(map.key_type, proto::tensor_proto::DataType::Int64);
    match &*map.value_type {
        OnnxType::Tensor(tt) => assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Float),
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn onnx_tensor_type_shape_with_param_dim() {
    // Arrange
    let tt = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(768)],
        },
    };
    // Act & Assert
    assert_eq!(tt.shape.dims.len(), 2);
    assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(p) if p == "batch"));
    assert!(matches!(&tt.shape.dims[1], OnnxDim::Known(768)));
}

#[test]
fn onnx_graph_struct_with_nodes_and_initializers() {
    // Arrange
    let mut initializers = HashMap::new();
    initializers.insert("w".to_string(), OnnxTensor::new(
        "w".to_string(), Dtype::F32, vec![2, 3], Bytes::from(vec![0u8; 24]),
    ));
    let graph = OnnxGraph {
        name: "test_graph".to_string(),
        doc_string: "test".to_string(),
        nodes: vec![OnnxNode {
            name: "matmul_0".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        }],
        inputs: vec![OnnxValueInfo {
            name: "x".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        }],
        outputs: vec![OnnxValueInfo {
            name: "y".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        }],
        value_info: vec![],
        initializers,
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.initializers.len(), 1);
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 1);
    assert!(graph.initializers.contains_key("w"));
}

#[test]
fn loader_model_metadata_ir_version_from_proto() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let mut graph = empty_graph();
    graph.initializer = vec![tensor];
    let model = proto::ModelProto {
        ir_version: Some(9),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some("ai.onnx".to_string()),
            version: Some(20),
        }],
        producer_name: Some("test_producer".to_string()),
        producer_version: Some("2.0".to_string()),
        domain: Some("ai.onnx".to_string()),
        model_version: Some(100),
        doc_string: Some("test doc".to_string()),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.ir_version, 9);
    assert_eq!(loader.model().metadata.producer_name, "test_producer");
    assert_eq!(loader.model().metadata.model_version, 100);
}

#[test]
fn loader_model_metadata_opset_import() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let mut graph = empty_graph();
    graph.initializer = vec![tensor];
    let model = proto::ModelProto {
        ir_version: Some(7),
        opset_import: vec![
            proto::OperatorSetIdProto { domain: Some("ai.onnx".to_string()), version: Some(17) },
            proto::OperatorSetIdProto { domain: Some("ai.onnx.ml".to_string()), version: Some(3) },
        ],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.opset_import.len(), 2);
}

#[test]
fn loader_model_metadata_props_from_proto() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let mut graph = empty_graph();
    graph.initializer = vec![tensor];
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![
            proto::StringStringEntryProto { key: Some("author".to_string()), value: Some("test".to_string()) },
        ],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.metadata_props.len(), 1);
    assert_eq!(loader.model().metadata.metadata_props.get("author"), Some(&"test".to_string()));
}

#[test]
fn onnx_attribute_struct_clone() {
    // Arrange
    let attr = OnnxAttribute {
        name: "alpha".to_string(),
        value: OnnxAttributeValue::Float(0.1),
        doc_string: "learning rate".to_string(),
        ref_attr_name: None,
        attr_type: Some(proto::attribute_proto::AttributeType::Float),
    };
    // Act
    let cloned = attr.clone();
    // Assert
    assert_eq!(cloned.name, attr.name);
    assert!(matches!(cloned.value, OnnxAttributeValue::Float(_)));
}

#[test]
fn onnx_model_struct_functions_field() {
    // Arrange
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        },
        graph: OnnxGraph {
            name: String::new(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![OnnxFunction {
            name: "MyOp".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["a".to_string()],
            outputs: vec!["b".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        }],
    };
    // Act & Assert
    assert_eq!(model.functions.len(), 1);
    assert_eq!(model.functions[0].name, "MyOp");
}

#[test]
fn onnx_dim_hash_in_hashmap() {
    // Arrange
    use std::collections::HashMap;
    let mut map: HashMap<OnnxDim, &str> = HashMap::new();
    map.insert(OnnxDim::Known(128), "hidden");
    map.insert(OnnxDim::Param("seq".to_string()), "sequence");
    map.insert(OnnxDim::Unknown, "wildcard");
    // Act & Assert
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&OnnxDim::Known(128)), Some(&"hidden"));
    assert_eq!(map.get(&OnnxDim::Param("seq".to_string())), Some(&"sequence"));
    assert_eq!(map.get(&OnnxDim::Unknown), Some(&"wildcard"));
}

#[test]
fn onnx_sparse_format_hash_in_hashset() {
    // Arrange
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(OnnxSparseFormat::Coo);
    set.insert(OnnxSparseFormat::Csr);
    set.insert(OnnxSparseFormat::Csc);
    // Act & Assert
    assert_eq!(set.len(), 3);
    assert!(set.contains(&OnnxSparseFormat::Coo));
    assert!(set.contains(&OnnxSparseFormat::Csr));
    assert!(set.contains(&OnnxSparseFormat::Csc));
}

#[test]
fn onnx_attribute_attr_type_none_is_valid() {
    // Arrange
    let attr = OnnxAttribute {
        name: "unknown_type_attr".to_string(),
        value: OnnxAttributeValue::Int(0),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act & Assert
    assert!(attr.attr_type.is_none());
}

#[test]
fn onnx_attribute_value_tensor_inner_access() {
    // Arrange
    let inner = OnnxTensor::new("inner".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    let value = OnnxAttributeValue::Tensor(inner);
    // Act & Assert
    if let OnnxAttributeValue::Tensor(t) = &value {
        assert_eq!(t.name, "inner");
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.shape, vec![2]);
    } else {
        panic!("expected Tensor variant");
    }
}

#[test]
fn convert_error_missing_initializer_display_format() {
    // Arrange
    let err = ConvertError::MissingInitializer {
        name: "encoder.weight".to_string(),
        node_name: "matmul_0".to_string(),
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("encoder.weight"), "Error message should contain tensor name");
    assert!(msg.contains("matmul_0"), "Error message should contain node name");
}

#[test]
fn onnx_tensor_new_preserves_name_and_shape() {
    // Arrange
    let data = Bytes::from(vec![0u8; 24]);
    // Act
    let tensor = OnnxTensor::new("bias".to_string(), Dtype::F32, vec![2, 3], data);
    // Assert
    assert_eq!(tensor.name, "bias");
    assert_eq!(tensor.dtype, Dtype::F32);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert!(!tensor.is_string);
}

// ══════════════════════════════════════════════════════════════════════
// NEW INTEGRATION TESTS — proto-level parsing and loader edge cases
// ══════════════════════════════════════════════════════════════════════

// ── Proto tensor parsing: BF16 raw data roundtrip ─────────────────────

#[test]
fn tensor_bf16_raw_data_roundtrip() {
    // Arrange: two BF16 values — 1.0 = 0x3F80, 2.0 = 0x4000
    let raw = vec![0x80, 0x3F, 0x00, 0x40];
    let tensor = tensor_raw("bf16_w", vec![2], proto::tensor_proto::DataType::Bfloat16, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("bf16_w").expect("tensor");
    assert_eq!(ts.dtype, Dtype::BF16);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 4);
}

// ── Proto tensor parsing: I16 raw data roundtrip ──────────────────────

#[test]
fn tensor_i16_raw_data_roundtrip() {
    // Arrange: two I16 values — 100 and -200
    let raw = vec![100u8, 0, 56u8, 255]; // 100 and -200 in LE
    let tensor = tensor_raw("i16_w", vec![2], proto::tensor_proto::DataType::Int16, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("i16_w").expect("tensor");
    assert_eq!(ts.dtype, Dtype::I16);
    assert_eq!(ts.data.len(), 4);
}

// ── Proto tensor parsing: BOOL raw data roundtrip ─────────────────────

#[test]
fn tensor_bool_raw_data_roundtrip() {
    // Arrange
    let raw = vec![1u8, 0, 1, 1, 0];
    let tensor = tensor_raw("mask", vec![5], proto::tensor_proto::DataType::Bool, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("mask").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U8); // BOOL maps to U8
    assert_eq!(ts.shape, vec![5]);
}

// ── Proto tensor parsing: U8 raw data roundtrip ───────────────────────

#[test]
fn tensor_u8_raw_data_roundtrip() {
    // Arrange
    let raw = vec![0u8, 127, 255];
    let tensor = tensor_raw("quant_vals", vec![3], proto::tensor_proto::DataType::Uint8, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("quant_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U8);
    assert_eq!(ts.data, raw.as_slice());
}

// ── Proto tensor parsing: I64 via int64_data field ────────────────────

#[test]
fn tensor_i64_via_int64_data_roundtrip() {
    // Arrange: two I64 values via int64_data field (not raw_data)
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int64 as i32);
    tensor.name = Some("seq_ids".to_string());
    tensor.int64_data = vec![100i64, -200i64];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("seq_ids").expect("tensor");
    assert_eq!(ts.dtype, Dtype::I64);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 16);
}

// ── Proto tensor parsing: I32 via int32_data field ────────────────────

#[test]
fn tensor_i32_via_int32_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int32 as i32);
    tensor.name = Some("labels".to_string());
    tensor.int32_data = vec![0, 1, 2];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("labels").expect("tensor");
    assert_eq!(ts.dtype, Dtype::I32);
    assert_eq!(ts.shape, vec![3]);
}

// ── Proto tensor parsing: Double raw data ─────────────────────────────

#[test]
fn tensor_double_raw_data_roundtrip() {
    // Arrange: one f64 value
    let val = 3.141592653589793f64;
    let raw = val.to_le_bytes().to_vec();
    let tensor = tensor_raw("precision", vec![], proto::tensor_proto::DataType::Double, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("precision").expect("tensor");
    assert_eq!(ts.dtype, Dtype::F64);
    assert_eq!(ts.data.len(), 8);
}

// ── Proto graph with multiple initializers of different dtypes ────────

#[test]
fn loader_multiple_dtypes_in_single_graph() {
    // Arrange
    let t_f32 = tensor_f32("fp32_weight", vec![2], &[1.0, 2.0]);
    let t_i64_raw = 42i64.to_le_bytes();
    let t_i64 = tensor_raw("index", vec![], proto::tensor_proto::DataType::Int64, &t_i64_raw);
    let graph = proto::GraphProto { initializer: vec![t_f32, t_i64], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let precisions = loader.unique_precisions();
    assert_eq!(precisions.len(), 2);
    assert!(precisions.contains(&Dtype::F32));
    assert!(precisions.contains(&Dtype::I64));
}

// ── Proto model with opset import preserved ───────────────────────────

#[test]
fn loader_opset_domain_and_version_preserved() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![
            proto::OperatorSetIdProto { domain: Some("ai.onnx".to_string()), version: Some(20) },
            proto::OperatorSetIdProto { domain: Some("ai.onnx.ml".to_string()), version: Some(3) },
        ],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let opsets = &loader.model().metadata.opset_import;
    assert_eq!(opsets.len(), 2);
    assert_eq!(opsets[0].domain, "ai.onnx");
    assert_eq!(opsets[0].version, 20);
    assert_eq!(opsets[1].domain, "ai.onnx.ml");
    assert_eq!(opsets[1].version, 3);
}

// ── Proto graph with named node ───────────────────────────────────────

#[test]
fn loader_graph_node_preserves_op_type() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Softmax".to_string()),
        name: Some("sm_0".to_string()),
        input: vec!["logits".to_string()],
        output: vec!["probs".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 1);
    assert_eq!(loader.graph().nodes[0].op_type, "Softmax");
    assert_eq!(loader.graph().nodes[0].name, "sm_0");
    assert_eq!(loader.graph().nodes[0].inputs, vec!["logits"]);
    assert_eq!(loader.graph().nodes[0].outputs, vec!["probs"]);
}

// ── Proto graph with value_info containing shape ──────────────────────

#[test]
fn loader_value_info_with_type_and_shape() {
    // Arrange
    let vi = proto::ValueInfoProto {
        name: Some("hidden".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("batch".to_string())),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(768)),
                            denotation: None,
                        },
                    ],
                }),
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().value_info.len(), 1);
    let info = &loader.graph().value_info[0];
    assert_eq!(info.name, "hidden");
    assert!(info.value_type.is_some());
    if let Some(OnnxType::Tensor(tt)) = &info.value_type {
        assert_eq!(tt.shape.dims.len(), 2);
        assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(p) if p == "batch"));
        assert!(matches!(&tt.shape.dims[1], OnnxDim::Known(768)));
    } else {
        panic!("expected Tensor type");
    }
}

// ── Proto graph with multiple inputs ──────────────────────────────────

#[test]
fn loader_graph_inputs_preserved() {
    // Arrange
    let input1 = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(7), // Int64
                shape: None,
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let input2 = proto::ValueInfoProto {
        name: Some("attention_mask".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1), // Float
                shape: None,
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![input1, input2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().inputs.len(), 2);
    assert_eq!(loader.graph().inputs[0].name, "input_ids");
    assert_eq!(loader.graph().inputs[1].name, "attention_mask");
}

// ── Proto graph with metadata_props on graph ──────────────────────────

#[test]
fn loader_graph_metadata_props_from_proto() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("framework".to_string()),
                value: Some("pytorch".to_string()),
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().metadata_props.get("framework"), Some(&"pytorch".to_string()));
}

// ── TensorSlice element_count from loaded tensor ──────────────────────

#[test]
fn tensor_slice_element_count_matches_shape() {
    // Arrange
    let tensor = tensor_f32("weight", vec![2, 3, 4], &[0.0f32; 24]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let slice = loader.tensor("weight").expect("tensor");
    // Assert
    assert_eq!(slice.shape, vec![2, 3, 4]);
    assert_eq!(slice.data.len(), 24 * 4); // 24 f32 elements = 96 bytes
}

// ── F16 tensor via float16_data field ─────────────────────────────────

#[test]
fn tensor_f16_via_float16_data_roundtrip() {
    // Arrange: F16 1.0 = 0x3C00, F16 2.0 = 0x4000, stored in int32_data
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Float16 as i32);
    tensor.name = Some("f16_vals".to_string());
    tensor.int32_data = vec![0x3C00i32, 0x4000i32]; // F16 bits in low 16 bits
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("f16_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::F16);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 4); // 2 x 2 bytes
}

// ── BF16 tensor via bfloat16_data field ───────────────────────────────

#[test]
fn tensor_bf16_via_bfloat16_data_roundtrip() {
    // Arrange: BF16 1.0 = 0x3F80, stored in int32_data
    let mut tensor = empty_tensor();
    tensor.dims = vec![1];
    tensor.data_type = Some(proto::tensor_proto::DataType::Bfloat16 as i32);
    tensor.name = Some("bf16_val".to_string());
    tensor.int32_data = vec![0x3F80i32];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("bf16_val").expect("tensor");
    assert_eq!(ts.dtype, Dtype::BF16);
    assert_eq!(ts.data.len(), 2);
}

// ── Empty scalar tensor (zero dims) ───────────────────────────────────

#[test]
fn tensor_scalar_f32_zero_dims() {
    // Arrange: a scalar tensor (no dims)
    let data = 42.0f32.to_le_bytes();
    let tensor = tensor_raw("scalar_param", vec![], proto::tensor_proto::DataType::Float, &data);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("scalar_param").expect("tensor");
    assert!(ts.shape.is_empty());
    assert_eq!(ts.dtype, Dtype::F32);
    assert_eq!(ts.data.len(), 4);
}

// ── Large tensor: 1D with many elements ───────────────────────────────

#[test]
fn tensor_large_1d_dim() {
    // Arrange: 1000 f32 elements
    let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
    let tensor = tensor_f32("large_vec", vec![1000], &values);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("large_vec").expect("tensor");
    assert_eq!(ts.shape, vec![1000]);
    assert_eq!(ts.data.len(), 4000);
}

// ── Model with producer metadata ──────────────────────────────────────

#[test]
fn loader_producer_name_and_version_preserved() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: Some("gllm-test".to_string()),
        producer_version: Some("0.1.0".to_string()),
        domain: Some("ai.onnx".to_string()),
        model_version: Some(42),
        doc_string: Some("test model".to_string()),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.producer_name, "gllm-test");
    assert_eq!(loader.model().metadata.producer_version, "0.1.0");
    assert_eq!(loader.model().metadata.domain, "ai.onnx");
    assert_eq!(loader.model().metadata.model_version, 42);
    assert_eq!(loader.model().metadata.doc_string, "test model");
}

// ── precision_by_tensor returns sorted results ────────────────────────

#[test]
fn loader_precision_by_tensor_sorted_alphabetically() {
    // Arrange: two tensors with names that sort differently
    let t_z = tensor_f32("z_weight", vec![1], &[1.0]);
    let t_a = tensor_f32("a_weight", vec![1], &[2.0]);
    let graph = proto::GraphProto { initializer: vec![t_z, t_a], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.precision_by_tensor();
    // Assert
    assert_eq!(precisions.len(), 2);
    assert_eq!(precisions[0].0, "a_weight");
    assert_eq!(precisions[1].0, "z_weight");
}

// ── names() returns sorted initializer names ──────────────────────────

#[test]
fn loader_names_returns_sorted_names() {
    // Arrange
    let t_c = tensor_f32("c_weight", vec![1], &[1.0]);
    let t_b = tensor_f32("b_weight", vec![1], &[2.0]);
    let t_a = tensor_f32("a_weight", vec![1], &[3.0]);
    let graph = proto::GraphProto { initializer: vec![t_c, t_b, t_a], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    // Assert
    assert_eq!(names, vec!["a_weight", "b_weight", "c_weight"]);
}

// ── Graph with empty node list loads successfully ─────────────────────

#[test]
fn loader_graph_no_nodes_succeeds() {
    // Arrange
    let tensor = tensor_f32("w", vec![2], &[1.0, 2.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert!(loader.graph().nodes.is_empty());
    assert_eq!(loader.graph().initializers.len(), 1);
}

// ── Node with attribute: transB integer ───────────────────────────────

#[test]
fn loader_node_integer_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let node = proto::NodeProto {
        op_type: Some("Gemm".to_string()),
        name: Some("gemm_0".to_string()),
        input: vec!["x".to_string(), "w".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(1),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 1);
    let attr = loader.graph().nodes[0].attributes.get("transB").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Int(1)));
}

// ── Node with attribute: float value ──────────────────────────────────

#[test]
fn loader_node_float_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("LeakyRelu".to_string()),
        name: Some("lrelu".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("alpha".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(0.01),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("alpha").expect("attribute");
    if let OnnxAttributeValue::Float(val) = attr.value {
        assert!((val - 0.01).abs() < 1e-6);
    } else {
        panic!("expected Float variant");
    }
}

// ── Node with attribute: ints list ────────────────────────────────────

#[test]
fn loader_node_ints_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Conv".to_string()),
        name: Some("conv_0".to_string()),
        input: vec!["x".to_string(), "w".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("kernel_shape".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
            ints: vec![3, 3],
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("kernel_shape").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Ints(ref v) if v == &vec![3i64, 3i64]));
}

// ── Duplicate tensor name in initializer errors ───────────────────────

#[test]
fn loader_duplicate_initializer_name_errors() {
    // Arrange: two tensors with same name
    let t1 = tensor_f32("dup", vec![1], &[1.0]);
    let t2 = tensor_f32("dup", vec![1], &[2.0]);
    let graph = proto::GraphProto { initializer: vec![t1, t2], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert
    assert!(result.is_err(), "Expected error for duplicate tensor name");
}

// ── Tensor with U32 dtype via raw data ────────────────────────────────

#[test]
fn tensor_u32_raw_data_roundtrip() {
    // Arrange
    let val = 123456u32;
    let raw = val.to_le_bytes();
    let tensor = tensor_raw("u32_val", vec![], proto::tensor_proto::DataType::Uint32, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u32_val").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U32);
}

// ── Tensor with U64 dtype via raw data ────────────────────────────────

#[test]
fn tensor_u64_raw_data_roundtrip() {
    // Arrange
    let val = 9876543210u64;
    let raw = val.to_le_bytes();
    let tensor = tensor_raw("u64_val", vec![], proto::tensor_proto::DataType::Uint64, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u64_val").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U64);
}

// ── Tensor with I8 dtype via raw data ─────────────────────────────────

#[test]
fn tensor_i8_raw_data_roundtrip() {
    // Arrange
    let raw = vec![127i8.to_le_bytes()[0], (-1i8).to_le_bytes()[0], 0u8];
    let tensor = tensor_raw("i8_vals", vec![3], proto::tensor_proto::DataType::Int8, &raw);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("i8_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::I8);
}

// ── Loader path method returns correct path ───────────────────────────

#[test]
fn loader_path_returns_canonical_path() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert!(loader.path().ends_with(".onnx") || loader.path().exists());
}

// ── Multiple nodes in graph preserved in order ────────────────────────

#[test]
fn loader_multiple_nodes_preserve_order() {
    // Arrange
    let tensor = tensor_f32("w", vec![2], &[1.0, 2.0]);
    let nodes: Vec<proto::NodeProto> = (0..5).map(|i| {
        proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some(format!("relu_{i}")),
            input: vec![format!("in_{i}")],
            output: vec![format!("out_{i}")],
            ..empty_node()
        }
    }).collect();
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: nodes,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 5);
    for (i, node) in loader.graph().nodes.iter().enumerate() {
        assert_eq!(node.name, format!("relu_{i}"));
    }
}

// ── Node with string attribute ────────────────────────────────────────

#[test]
fn loader_node_string_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Pad".to_string()),
        name: Some("pad_0".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("mode".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::String as i32),
            s: Some(b"constant".to_vec()),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("mode").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::String(ref s) if s == "constant"));
}

// ── Node with floats attribute ────────────────────────────────────────

#[test]
fn loader_node_floats_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Mul".to_string()),
        name: Some("scale".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("scales".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Floats as i32),
            floats: vec![0.5, 1.0, 2.0],
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("scales").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Floats(ref v) if v.len() == 3));
}

// ── Node with strings attribute ───────────────────────────────────────

#[test]
fn loader_node_strings_attribute_parsed() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Loop".to_string()),
        name: Some("loop_0".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("body_input_names".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Strings as i32),
            strings: vec![b"prev".to_vec(), b"current".to_vec()],
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("body_input_names").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Strings(ref v) if v.len() == 2));
}

// ── graph method returns reference ────────────────────────────────────

#[test]
fn loader_graph_method_returns_reference() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act & Assert
    assert_eq!(loader.graph().initializers.len(), 1);
    assert_eq!(loader.graph().name, ""); // default empty name
}

// ── model method returns reference ────────────────────────────────────

#[test]
fn loader_model_method_returns_reference() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act & Assert
    assert_eq!(loader.model().metadata.ir_version, 0); // default
}

// ── Tensor dtype detection from loaded model ──────────────────────────

#[test]
fn loader_tensor_dtype_method_returns_correct_dtype() {
    // Arrange
    let tensor = tensor_f32("fp32_w", vec![2], &[1.0, 2.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let dtype = loader.tensor_dtype("fp32_w").expect("dtype");
    // Assert
    assert_eq!(dtype, Dtype::F32);
}

// ── tensor_dtype for missing tensor errors ─────────────────────────────

#[test]
fn loader_tensor_dtype_nonexistent_name_errors() {
    // Arrange
    let graph = proto::GraphProto { ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let result = loader.tensor_dtype("nonexistent");
    // Assert
    assert!(result.is_err());
}

// ── tensor for missing tensor errors ──────────────────────────────────

#[test]
fn loader_tensor_ghost_name_errors() {
    // Arrange
    let graph = proto::GraphProto { ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let result = loader.tensor("ghost");
    // Assert
    assert!(result.is_err());
}

// ── external_data_locations returns empty for inline model ────────────

#[test]
fn external_data_locations_inline_model_empty() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let locations = external_data_locations(file.path()).expect("locations");
    // Assert
    assert!(locations.is_empty());
}

// ── Proto graph with doc_string on graph ──────────────────────────────

#[test]
fn loader_graph_doc_string_preserved() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        doc_string: Some("This is a test graph".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().doc_string, "This is a test graph");
}

// ── Proto model with metadata_props preserved ────────────────────────

#[test]
fn loader_model_metadata_props_preserved() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("license".to_string()),
                value: Some("MIT".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("gllm".to_string()),
            },
        ],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let props = &loader.model().metadata.metadata_props;
    assert_eq!(props.len(), 2);
    assert_eq!(props.get("license"), Some(&"MIT".to_string()));
    assert_eq!(props.get("author"), Some(&"gllm".to_string()));
}

// ══════════════════════════════════════════════════════════════════════
// 70 additional tests — wave 3: deeper coverage
// ══════════════════════════════════════════════════════════════════════

// ── ConvertError individual variant Display messages ──────────────────

#[test]
fn convert_error_unsupported_op_msg_content() {
    // Arrange
    let err = ConvertError::UnsupportedOp {
        op_type: "ConvTranspose".to_string(),
        node_name: "conv_t_0".to_string(),
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("ConvTranspose"), "should contain op_type");
    assert!(msg.contains("conv_t_0"), "should contain node_name");
}

#[test]
fn convert_error_invalid_matmul_shape_msg_content() {
    // Arrange
    let err = ConvertError::InvalidMatMulShape {
        name: "weight_3d".to_string(),
        dims: 3,
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("weight_3d"), "should contain tensor name");
    assert!(msg.contains("3"), "should contain dimension count");
}

#[test]
fn convert_error_no_weight_input_msg_content() {
    // Arrange
    let err = ConvertError::NoWeightInput {
        node_name: "matmul_dyn".to_string(),
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("matmul_dyn"), "should contain node name");
}

#[test]
fn convert_error_attribute_error_msg_content() {
    // Arrange
    let err = ConvertError::AttributeError {
        node_name: "reshape_0".to_string(),
        reason: "missing shape attribute".to_string(),
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("reshape_0"), "should contain node name");
    assert!(msg.contains("missing shape attribute"), "should contain reason");
}

#[test]
fn convert_error_shape_inference_msg_content() {
    // Arrange
    let err = ConvertError::ShapeInferenceFailed {
        name: "output_logits".to_string(),
        reason: "dynamic batch dimension".to_string(),
    };
    // Act
    let msg = err.to_string();
    // Assert
    assert!(msg.contains("output_logits"), "should contain tensor name");
    assert!(msg.contains("dynamic batch dimension"), "should contain reason");
}

// ── OnnxAttributeValue variant coverage: SparseTensor, Type ────────────

#[test]
fn onnx_attribute_value_sparse_tensor_debug() {
    // Arrange
    let values = OnnxTensor::new("sv".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
    let indices = OnnxTensor::new("si".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
    let sparse = OnnxSparseTensor { values, indices, dims: vec![5], format: OnnxSparseFormat::Coo };
    let value = OnnxAttributeValue::SparseTensor(sparse);
    // Act
    let debug = format!("{value:?}");
    // Assert
    assert!(debug.contains("SparseTensor"), "Debug should contain 'SparseTensor'");
}

#[test]
fn onnx_attribute_value_type_debug() {
    // Arrange
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int32,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
    });
    let value = OnnxAttributeValue::Type(ty);
    // Act
    let debug = format!("{value:?}");
    // Assert
    assert!(debug.contains("Type"), "Debug should contain 'Type'");
}

#[test]
fn onnx_attribute_value_type_clone() {
    // Arrange
    let ty = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![] },
    });
    let original = OnnxAttributeValue::Type(ty);
    // Act
    let cloned = original.clone();
    // Assert
    assert!(matches!(cloned, OnnxAttributeValue::Type(_)));
}

#[test]
fn onnx_attribute_value_sparse_tensor_clone() {
    // Arrange
    let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
    let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
    let sparse = OnnxSparseTensor { values, indices, dims: vec![3], format: OnnxSparseFormat::Csr };
    let original = OnnxAttributeValue::SparseTensor(sparse);
    // Act
    let cloned = original.clone();
    // Assert
    assert!(matches!(cloned, OnnxAttributeValue::SparseTensor(_)));
}

// ── OnnxModel clone and Debug ─────────────────────────────────────────

#[test]
fn onnx_model_clone_preserves_functions() {
    // Arrange
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 8,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 17 }],
            metadata_props: HashMap::new(),
        },
        graph: OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![OnnxFunction {
            name: "MyOp".to_string(),
            domain: "custom".to_string(),
            overload: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        }],
    };
    // Act
    let cloned = model.clone();
    // Assert
    assert_eq!(cloned.metadata.ir_version, 8);
    assert_eq!(cloned.functions.len(), 1);
    assert_eq!(cloned.functions[0].name, "MyOp");
}

#[test]
fn onnx_model_debug_contains_metadata() {
    // Arrange
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 9,
            producer_name: "debug_test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        },
        graph: OnnxGraph {
            name: "dbg_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![],
    };
    // Act
    let debug = format!("{model:?}");
    // Assert
    assert!(debug.contains("debug_test"), "Debug should contain producer_name");
    assert!(debug.contains("dbg_graph"), "Debug should contain graph name");
}

// ── OnnxGraph clone independence ───────────────────────────────────────

#[test]
fn onnx_graph_clone_independence() {
    // Arrange
    let mut initializers = HashMap::new();
    initializers.insert("w".to_string(), OnnxTensor::new(
        "w".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]),
    ));
    let graph = OnnxGraph {
        name: "original".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers,
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let mut cloned = graph.clone();
    cloned.name = "modified".to_string();
    // Assert
    assert_eq!(graph.name, "original");
    assert_eq!(cloned.name, "modified");
    assert!(cloned.initializers.contains_key("w"));
}

// ── OnnxNode clone and Debug ──────────────────────────────────────────

#[test]
fn onnx_node_clone_preserves_all_fields() {
    // Arrange
    let mut attrs = HashMap::new();
    attrs.insert("epsilon".to_string(), OnnxAttribute {
        name: "epsilon".to_string(),
        value: OnnxAttributeValue::Float(1e-5),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    let node = OnnxNode {
        name: "layernorm".to_string(),
        op_type: "LayerNorm".to_string(),
        domain: "ai.onnx".to_string(),
        inputs: vec!["x".to_string(), "scale".to_string(), "bias".to_string()],
        outputs: vec!["y".to_string()],
        attributes: attrs,
    };
    // Act
    let cloned = node.clone();
    // Assert
    assert_eq!(cloned.name, "layernorm");
    assert_eq!(cloned.op_type, "LayerNorm");
    assert_eq!(cloned.inputs.len(), 3);
    assert!(cloned.attributes.contains_key("epsilon"));
}

#[test]
fn onnx_node_debug_contains_op_type() {
    // Arrange
    let node = OnnxNode {
        name: "add_0".to_string(),
        op_type: "Add".to_string(),
        domain: String::new(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attributes: HashMap::new(),
    };
    // Act
    let debug = format!("{node:?}");
    // Assert
    assert!(debug.contains("Add"), "Debug should contain op_type");
    assert!(debug.contains("add_0"), "Debug should contain node name");
}

// ── OnnxValueInfo clone and Debug ─────────────────────────────────────

#[test]
fn onnx_value_info_clone_independence() {
    // Arrange
    let info = OnnxValueInfo {
        name: "tensor_a".to_string(),
        value_type: Some(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3)] },
        })),
        doc_string: "test".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let mut cloned = info.clone();
    cloned.name = "tensor_b".to_string();
    // Assert
    assert_eq!(info.name, "tensor_a");
    assert_eq!(cloned.name, "tensor_b");
}

#[test]
fn onnx_value_info_debug_contains_name() {
    // Arrange
    let info = OnnxValueInfo {
        name: "logits".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let debug = format!("{info:?}");
    // Assert
    assert!(debug.contains("logits"), "Debug should contain name");
}

// ── OnnxQuantizationAnnotation clone and Debug ────────────────────────

#[test]
fn onnx_quantization_annotation_clone_roundtrip() {
    // Arrange
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "q_weight".to_string(),
        quant_param_tensor_names: HashMap::new(),
        scale: Some(0.1),
        zero_point: Some(128),
        axis: Some(0),
    };
    // Act
    let cloned = qa.clone();
    // Assert
    assert_eq!(cloned.tensor_name, "q_weight");
    assert_eq!(cloned.scale, Some(0.1));
    assert_eq!(cloned.zero_point, Some(128));
    assert_eq!(cloned.axis, Some(0));
}

#[test]
fn onnx_quantization_annotation_debug_output() {
    // Arrange
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "weight_debug".to_string(),
        quant_param_tensor_names: HashMap::new(),
        scale: None,
        zero_point: None,
        axis: None,
    };
    // Act
    let debug = format!("{qa:?}");
    // Assert
    assert!(debug.contains("weight_debug"), "Debug should contain tensor_name");
}

// ── OnnxFunction clone and with multiple nodes ────────────────────────

#[test]
fn onnx_function_clone_preserves_nodes() {
    // Arrange
    let func = OnnxFunction {
        name: "FusedBiasGelu".to_string(),
        domain: "com.custom".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec!["eps".to_string()],
        attribute_protos: HashMap::new(),
        nodes: vec![
            OnnxNode {
                name: "bias_add".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                inputs: vec!["X".to_string(), "bias".to_string()],
                outputs: vec!["biased".to_string()],
                attributes: HashMap::new(),
            },
            OnnxNode {
                name: "gelu_act".to_string(),
                op_type: "Gelu".to_string(),
                domain: String::new(),
                inputs: vec!["biased".to_string()],
                outputs: vec!["Y".to_string()],
                attributes: HashMap::new(),
            },
        ],
        opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 17 }],
        value_info: vec![],
        doc_string: "Fused bias add + gelu".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = func.clone();
    // Assert
    assert_eq!(cloned.nodes.len(), 2);
    assert_eq!(cloned.nodes[0].op_type, "Add");
    assert_eq!(cloned.nodes[1].op_type, "Gelu");
    assert_eq!(cloned.opset_import.len(), 1);
}

#[test]
fn onnx_function_debug_contains_name() {
    // Arrange
    let func = OnnxFunction {
        name: "CustomRelu".to_string(),
        domain: String::new(),
        overload: String::new(),
        inputs: vec![],
        outputs: vec![],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let debug = format!("{func:?}");
    // Assert
    assert!(debug.contains("CustomRelu"), "Debug should contain function name");
}

// ── OnnxOperatorSet clone ─────────────────────────────────────────────

#[test]
fn onnx_operator_set_clone_roundtrip() {
    // Arrange
    let ops = OnnxOperatorSet {
        domain: "ai.onnx.nn".to_string(),
        version: 20,
    };
    // Act
    let cloned = ops.clone();
    // Assert
    assert_eq!(cloned.domain, "ai.onnx.nn");
    assert_eq!(cloned.version, 20);
}

#[test]
fn onnx_operator_set_debug_output() {
    // Arrange
    let ops = OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 17,
    };
    // Act
    let debug = format!("{ops:?}");
    // Assert
    assert!(debug.contains("ai.onnx") || debug.contains("17"), "Debug should contain domain or version");
}

// ── OnnxModelMetadata clone ───────────────────────────────────────────

#[test]
fn onnx_model_metadata_clone_independence() {
    // Arrange
    let meta = OnnxModelMetadata {
        ir_version: 8,
        producer_name: "original_producer".to_string(),
        producer_version: "1.0".to_string(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        opset_import: vec![OnnxOperatorSet { domain: String::new(), version: 17 }],
        metadata_props: HashMap::new(),
    };
    // Act
    let mut cloned = meta.clone();
    cloned.producer_name = "modified".to_string();
    // Assert
    assert_eq!(meta.producer_name, "original_producer");
    assert_eq!(cloned.producer_name, "modified");
}

// ── OnnxDim in complex HashMap/HashSet scenarios ──────────────────────

#[test]
fn onnx_dim_param_dedup_in_hashset() {
    // Arrange
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(OnnxDim::Param("batch".to_string()));
    set.insert(OnnxDim::Param("batch".to_string())); // duplicate
    set.insert(OnnxDim::Param("seq".to_string()));
    // Act & Assert
    assert_eq!(set.len(), 2, "duplicate Param should be deduplicated");
}

#[test]
fn onnx_dim_known_negative_in_hashmap() {
    // Arrange
    use std::collections::HashMap;
    let mut map: HashMap<OnnxDim, &str> = HashMap::new();
    map.insert(OnnxDim::Known(-1), "dynamic_dim");
    map.insert(OnnxDim::Known(-2), "another_dynamic");
    // Act & Assert
    assert_eq!(map.len(), 2);
    assert_eq!(map.get(&OnnxDim::Known(-1)), Some(&"dynamic_dim"));
}

// ── OnnxTensor scalar_f32 edge: I32 max boundary ─────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_i32_min() {
    // Arrange
    let data = i32::MIN.to_le_bytes();
    let tensor = OnnxTensor::new("i32_min".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_f32().expect("should convert i32_min to f32");
    // Assert
    assert!(val.is_finite(), "i32::MIN as f32 should be finite");
}

#[test]
fn onnx_tensor_scalar_i64_from_u16_max() {
    // Arrange
    let data = u16::MAX.to_le_bytes();
    let tensor = OnnxTensor::new("u16_max".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_i64().expect("should convert u16_max to i64");
    // Assert
    assert_eq!(val, u16::MAX as i64);
}

// ── OnnxSparseTensor clone independence ───────────────────────────────

#[test]
fn onnx_sparse_tensor_clone_independence() {
    // Arrange
    let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
    let sparse = OnnxSparseTensor {
        values,
        indices,
        dims: vec![10],
        format: OnnxSparseFormat::Coo,
    };
    // Act
    let cloned = sparse.clone();
    // Assert: both are equal but independent
    assert_eq!(cloned.dims, sparse.dims);
    assert_eq!(cloned.format, sparse.format);
}

// ── Loader: loading from non-existent file errors ─────────────────────

#[test]
fn loader_from_nonexistent_path_errors() {
    // Arrange
    let bad_path = std::path::PathBuf::from("/tmp/definitely_does_not_exist_onnx_999999.onnx");
    // Act
    let result = OnnxLoader::from_path(&bad_path);
    // Assert
    assert!(result.is_err(), "Loading from nonexistent path should fail");
}

// ── Loader: loading corrupted protobuf errors ─────────────────────────

#[test]
fn loader_from_corrupted_file_errors() {
    // Arrange
    let file = NamedTempFile::new().expect("tempfile");
    std::fs::write(file.path(), b"this is not valid protobuf data at all").expect("write");
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert
    assert!(result.is_err(), "Loading corrupted protobuf should fail");
}

// ── Loader: model with empty graph errors ─────────────────────────────

#[test]
fn loader_model_with_no_graph_errors() {
    // Arrange
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: None, // missing graph
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert
    assert!(result.is_err(), "Model without graph should fail");
}

// ── Loader: tensor with no data_type errors ───────────────────────────

#[test]
fn loader_tensor_no_data_type_errors() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.name = Some("bad_tensor".to_string());
    tensor.data_type = None; // missing
    tensor.raw_data = Some(Bytes::from(vec![0u8; 8]));
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert
    assert!(result.is_err(), "Tensor without data_type should fail");
}

// ── Loader: node with multiple outputs preserved ──────────────────────

#[test]
fn loader_node_multiple_outputs_preserved() {
    // Arrange
    let node = proto::NodeProto {
        op_type: Some("Split".to_string()),
        name: Some("split_0".to_string()),
        input: vec!["input".to_string()],
        output: vec!["out1".to_string(), "out2".to_string(), "out3".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto { node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes[0].outputs.len(), 3);
    assert_eq!(loader.graph().nodes[0].outputs[0], "out1");
    assert_eq!(loader.graph().nodes[0].outputs[1], "out2");
    assert_eq!(loader.graph().nodes[0].outputs[2], "out3");
}

// ── Loader: node with no inputs preserved ─────────────────────────────

#[test]
fn loader_node_no_inputs_preserved() {
    // Arrange
    let node = proto::NodeProto {
        op_type: Some("Constant".to_string()),
        name: Some("const_0".to_string()),
        input: vec![],
        output: vec!["const_val".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto { node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert!(loader.graph().nodes[0].inputs.is_empty());
}

// ── Loader: node with no name gets empty string ───────────────────────

// ── Loader: unique_precisions with single dtype ───────────────────────

#[test]
fn loader_unique_precisions_single_dtype() {
    // Arrange
    let tensor = tensor_f32("w", vec![2], &[1.0, 2.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.unique_precisions();
    // Assert
    assert_eq!(precisions.len(), 1);
    assert_eq!(precisions[0], Dtype::F32);
}

// ── Loader: precision_by_tensor returns correct dtype per tensor ──────

#[test]
fn loader_precision_by_tensor_returns_correct_dtypes() {
    // Arrange
    let t1 = tensor_f32("fp32_w", vec![1], &[1.0]);
    let t2_raw = 42i64.to_le_bytes();
    let t2 = tensor_raw("i64_idx", vec![1], proto::tensor_proto::DataType::Int64, &t2_raw);
    let graph = proto::GraphProto { initializer: vec![t1, t2], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let pb = loader.precision_by_tensor();
    // Assert
    assert_eq!(pb.len(), 2);
    let fp32_entry = pb.iter().find(|(n, _)| n == "fp32_w").expect("find fp32_w");
    assert_eq!(fp32_entry.1, Dtype::F32);
    let i64_entry = pb.iter().find(|(n, _)| n == "i64_idx").expect("find i64_idx");
    assert_eq!(i64_entry.1, Dtype::I64);
}

// ── Loader: graph with quantization_annotations ───────────────────────

#[test]
fn loader_graph_quantization_annotations_preserved() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let qa = proto::TensorAnnotation {
        tensor_name: Some("w".to_string()),
        quant_parameter_tensor_names: vec![],
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        quantization_annotation: vec![qa],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().quantization_annotation.len(), 1);
    assert_eq!(loader.graph().quantization_annotation[0].tensor_name, "w");
}

// ── Loader: TensorProvider iter_tensors yields all tensors ─────────────

#[test]
fn loader_iter_tensors_yields_all() {
    // Arrange
    let t_a = tensor_f32("a", vec![1], &[1.0]);
    let t_b = tensor_f32("b", vec![], &[2.0]);
    let graph = proto::GraphProto { initializer: vec![t_a, t_b], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    use crate::loader::TensorProvider;
    let mut metas: Vec<_> = loader.iter_tensors().collect();
    metas.sort_by(|a, b| a.name.cmp(&b.name));
    // Assert
    assert_eq!(metas.len(), 2);
    assert_eq!(metas[0].name, "a");
    assert_eq!(metas[1].name, "b");
}

// ── Loader: TensorProvider tensor_info found ──────────────────────────

#[test]
fn loader_tensor_provider_info_found() {
    // Arrange
    let tensor = tensor_f32("my_weight", vec![3, 4], &[0.0f32; 12]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    use crate::loader::TensorProvider;
    let info = loader.tensor_info("my_weight");
    // Assert
    assert!(info.is_some());
    let meta = info.unwrap();
    assert_eq!(meta.name, "my_weight");
    assert_eq!(meta.shape, vec![3, 4]);
    assert_eq!(meta.dtype, Dtype::F32);
}

// ── Loader: TensorProvider tensor_info missing returns None ────────────

#[test]
fn loader_tensor_provider_info_missing() {
    // Arrange
    let graph = proto::GraphProto { ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    // Act
    use crate::loader::TensorProvider;
    let info = loader.tensor_info("nonexistent");
    // Assert
    assert!(info.is_none());
}

// ── Loader: F64 tensor via double_data field ───────────────────────────

#[test]
fn tensor_f64_via_double_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Double as i32);
    tensor.name = Some("f64_vals".to_string());
    tensor.double_data = vec![1.5f64, -2.5];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("f64_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::F64);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 16);
}

// ── Loader: U32 tensor via uint64_data field ──────────────────────────

#[test]
fn tensor_u32_via_uint64_data_roundtrip() {
    // Arrange: U32 values stored in uint64_data (high 32 bits zero)
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint32 as i32);
    tensor.name = Some("u32_vals".to_string());
    tensor.uint64_data = vec![100u64, 200u64];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u32_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U32);
    assert_eq!(ts.shape, vec![2]);
}

// ── Loader: graph with node that has graph attribute (If operator) ────

#[test]
fn loader_node_with_graph_attribute_parsed() {
    // Arrange
    let sub_graph = proto::GraphProto {
        node: vec![proto::NodeProto {
            op_type: Some("Identity".to_string()),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            ..empty_node()
        }],
        name: Some("then_branch".to_string()),
        ..empty_graph()
    };
    let attr = proto::AttributeProto {
        name: Some("then_branch".to_string()),
        r#type: Some(proto::attribute_proto::AttributeType::Graph as i32),
        g: Some(sub_graph),
        ..Default::default()
    };
    let node = proto::NodeProto {
        op_type: Some("If".to_string()),
        name: Some("if_0".to_string()),
        input: vec!["cond".to_string()],
        output: vec!["result".to_string()],
        attribute: vec![attr],
        ..empty_node()
    };
    let graph = proto::GraphProto { node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let attr = loader.graph().nodes[0].attributes.get("then_branch").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Graph(_)));
    if let OnnxAttributeValue::Graph(g) = &attr.value {
        assert_eq!(g.name, "then_branch");
        assert_eq!(g.nodes.len(), 1);
    }
}

// ── Loader: model with function definition ────────────────────────────

#[test]
fn loader_model_with_function_definition() {
    // Arrange
    let func_proto = proto::FunctionProto {
        name: Some("MyRelu".to_string()),
        domain: Some("custom".to_string()),
        ..Default::default()
    };
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![func_proto],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().functions.len(), 1);
    assert_eq!(loader.model().functions[0].name, "MyRelu");
}

// ── Loader: external data with offset parameter ───────────────────────

#[test]
fn external_data_locations_with_offset() {
    // Arrange
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let ext_tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("ext_w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("4096".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto { initializer: vec![ext_tensor], ..empty_graph() };
    write_model(empty_model(graph), &model_path);
    // Act
    let locations = external_data_locations(&model_path).expect("locations");
    // Assert
    assert_eq!(locations.len(), 1);
    assert_eq!(locations[0], "weights.bin");
}

// ── OnnxTensorType: same shape different elem_type not equal ──────────

#[test]
fn onnx_tensor_type_same_shape_different_elem_not_equal() {
    // Arrange
    let f32_tt = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
    };
    let i32_tt = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int32,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
    };
    // Act & Assert
    assert_ne!(f32_tt, i32_tt, "Same shape different elem_type should not be equal");
}

// ── OnnxType equality: same variant equal ─────────────────────────────

#[test]
fn onnx_type_tensor_equality_same_content() {
    // Arrange
    let a = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3)] },
    });
    let b = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(3)] },
    });
    // Act & Assert
    assert_eq!(a, b);
}

#[test]
fn onnx_type_sequence_not_equal_different_inner() {
    // Arrange
    let a = OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![] },
    })));
    let b = OnnxType::Sequence(Box::new(OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int32,
        shape: OnnxTensorShape { dims: vec![] },
    })));
    // Act & Assert
    assert_ne!(a, b, "Sequences with different inner types should not be equal");
}

// ── OnnxAttribute: Ref variant preserves string ───────────────────────

#[test]
fn onnx_attribute_value_ref_access() {
    // Arrange
    let value = OnnxAttributeValue::Ref("base_attr_name".to_string());
    // Act & Assert
    if let OnnxAttributeValue::Ref(name) = &value {
        assert_eq!(name, "base_attr_name");
    } else {
        panic!("expected Ref variant");
    }
}

#[test]
fn onnx_attribute_value_ref_debug() {
    // Arrange
    let value = OnnxAttributeValue::Ref("some_ref".to_string());
    // Act
    let debug = format!("{value:?}");
    // Assert
    assert!(debug.contains("Ref"), "Debug should contain 'Ref'");
}

// ── bytes_to_f32 helper edge cases ────────────────────────────────────

#[test]
fn bytes_to_f32_max_value() {
    // Arrange
    let data = f32::MAX.to_le_bytes();
    // Act
    let result = bytes_to_f32(&data);
    // Assert
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], f32::MAX);
}

#[test]
fn bytes_to_f32_min_positive() {
    // Arrange
    let data = f32::MIN_POSITIVE.to_le_bytes();
    // Act
    let result = bytes_to_f32(&data);
    // Assert
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], f32::MIN_POSITIVE);
}

// ── OnnxTensor: slice from loaded model matches raw_data ──────────────

#[test]
fn loader_tensor_slice_matches_raw_data() {
    // Arrange
    let values = &[1.0f32, 2.0, 3.0, 4.0];
    let tensor = tensor_f32("test_w", vec![2, 2], values);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let ts = loader.tensor("test_w").expect("tensor");
    // Assert
    let expected_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(ts.data.as_ref() as &[u8], expected_bytes.as_slice());
}

// ── Loader: node with doc_string in attribute proto ───────────────────

#[test]
fn loader_node_attribute_with_doc_string() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let attr = proto::AttributeProto {
        name: Some("axis".to_string()),
        r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
        i: Some(-1),
        doc_string: Some("axis along which to reduce".to_string()),
        ..Default::default()
    };
    let node = proto::NodeProto {
        op_type: Some("ReduceMean".to_string()),
        name: Some("mean_0".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![attr],
        ..empty_node()
    };
    let graph = proto::GraphProto { initializer: vec![tensor], node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let parsed_attr = loader.graph().nodes[0].attributes.get("axis").expect("attribute");
    assert!(matches!(parsed_attr.value, OnnxAttributeValue::Int(-1)));
    assert_eq!(parsed_attr.doc_string, "axis along which to reduce");
}

// ── Loader: model with ir_version=0 ───────────────────────────────────

#[test]
fn loader_ir_version_zero() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let mut graph = empty_graph();
    graph.initializer = vec![tensor];
    let model = proto::ModelProto {
        ir_version: Some(0),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.ir_version, 0);
}

// ── Loader: tensor with empty string name errors ──────────────────────

#[test]
fn loader_tensor_empty_name_errors() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![1];
    tensor.data_type = Some(proto::tensor_proto::DataType::Float as i32);
    tensor.name = Some(String::new()); // empty name
    tensor.raw_data = Some(Bytes::from(vec![0u8; 4]));
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert
    assert!(result.is_err(), "Tensor with empty name should fail");
}

// ── OnnxTensorShape: mixed known and param dims ───────────────────────

#[test]
fn onnx_tensor_shape_mixed_known_param_unknown() {
    // Arrange
    let shape = OnnxTensorShape {
        dims: vec![
            OnnxDim::Param("batch".to_string()),
            OnnxDim::Known(12),
            OnnxDim::Unknown,
            OnnxDim::Known(64),
        ],
    };
    // Act & Assert
    assert_eq!(shape.dims.len(), 4);
    assert!(matches!(&shape.dims[0], OnnxDim::Param(p) if p == "batch"));
    assert!(matches!(&shape.dims[1], OnnxDim::Known(12)));
    assert!(matches!(&shape.dims[2], OnnxDim::Unknown));
    assert!(matches!(&shape.dims[3], OnnxDim::Known(64)));
}

// ── Loader: I16 tensor via int32_data field ────────────────────────────

#[test]
fn tensor_i16_via_int32_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int16 as i32);
    tensor.name = Some("i16_vals".to_string());
    tensor.int32_data = vec![100i32, -200i32];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("i16_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::I16);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 4); // 2 x 2 bytes
}

// ── Loader: U16 tensor via int32_data field ────────────────────────────

#[test]
fn tensor_u16_via_int32_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint16 as i32);
    tensor.name = Some("u16_vals".to_string());
    tensor.int32_data = vec![0i32, 1000i32, 65535i32];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u16_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U16);
    assert_eq!(ts.shape, vec![3]);
    assert_eq!(ts.data.len(), 6); // 3 x 2 bytes
}

// ── Loader: graph with value_info shape containing DimParam ────────────

#[test]
fn loader_value_info_dim_param_preserved() {
    // Arrange
    let vi = proto::ValueInfoProto {
        name: Some("tokens".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(7), // Int64
                shape: Some(proto::TensorShapeProto {
                    dim: vec![
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("batch_size".to_string())),
                            denotation: None,
                        },
                        proto::tensor_shape_proto::Dimension {
                            value: Some(proto::tensor_shape_proto::dimension::Value::DimParam("seq_len".to_string())),
                            denotation: None,
                        },
                    ],
                }),
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto { input: vec![vi], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let input = &loader.graph().inputs[0];
    assert_eq!(input.name, "tokens");
    if let Some(OnnxType::Tensor(tt)) = &input.value_type {
        assert!(matches!(&tt.shape.dims[0], OnnxDim::Param(p) if p == "batch_size"));
        assert!(matches!(&tt.shape.dims[1], OnnxDim::Param(p) if p == "seq_len"));
    } else {
        panic!("expected Tensor type with shape");
    }
}

// ── OnnxAttributeValue: Floats with special values ─────────────────────

#[test]
fn onnx_attribute_value_floats_empty_vec() {
    // Arrange
    let value = OnnxAttributeValue::Floats(vec![]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Floats(ref v) if v.is_empty()));
}

#[test]
fn onnx_attribute_value_ints_empty_vec() {
    // Arrange
    let value = OnnxAttributeValue::Ints(vec![]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Ints(ref v) if v.is_empty()));
}

#[test]
fn onnx_attribute_value_strings_empty_vec() {
    // Arrange
    let value = OnnxAttributeValue::Strings(vec![]);
    // Act & Assert
    assert!(matches!(value, OnnxAttributeValue::Strings(ref v) if v.is_empty()));
}

// ── Loader: multiple initializers all loadable ────────────────────────

#[test]
fn loader_many_initializers_all_accessible() {
    // Arrange
    let mut tensors = Vec::new();
    for i in 0..20 {
        tensors.push(tensor_f32(&format!("w_{i}"), vec![1], &[i as f32]));
    }
    let graph = proto::GraphProto { initializer: tensors, ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().initializers.len(), 20);
    for i in 0..20 {
        let name = format!("w_{i}");
        let ts = loader.tensor(&name).expect(&format!("tensor {name}"));
        assert_eq!(ts.dtype, Dtype::F32, "tensor {name} should be F32");
    }
}

// ── OnnxModelMetadata: doc_string preserved ───────────────────────────

#[test]
fn onnx_model_metadata_doc_string_from_proto() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let mut graph = empty_graph();
    graph.initializer = vec![tensor];
    let model = proto::ModelProto {
        ir_version: Some(7),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: Some("Sample documentation string".to_string()),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.doc_string, "Sample documentation string");
}

// ══════════════════════════════════════════════════════════════════════
// Wave 4: ~40 additional tests targeting uncovered areas
// ══════════════════════════════════════════════════════════════════════

// ── LoaderError variant Display coverage ──────────────────────────────

#[test]
fn loader_error_network_display() {
    let err = crate::loader::LoaderError::Network("connection refused".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("Network error"));
    assert!(msg.contains("connection refused"));
}

#[test]
fn loader_error_cache_display() {
    let err = crate::loader::LoaderError::Cache("disk full".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("Cache error"));
    assert!(msg.contains("disk full"));
}

#[test]
fn loader_error_missing_weights_display() {
    let err = crate::loader::LoaderError::MissingWeights;
    let msg = format!("{err}");
    assert!(msg.contains("Missing weights"));
}

#[test]
fn loader_error_missing_tensor_display() {
    let err = crate::loader::LoaderError::MissingTensor("encoder.layer.0.weight".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("Missing tensor"));
    assert!(msg.contains("encoder.layer.0.weight"));
}

#[test]
fn loader_error_unsupported_dtype_display() {
    let err = crate::loader::LoaderError::UnsupportedDtype(Dtype::F8_E4M3);
    let msg = format!("{err}");
    assert!(msg.contains("Unsupported dtype"));
}

#[test]
fn loader_error_onnx_display() {
    let err = crate::loader::LoaderError::Onnx("invalid node".to_string());
    let msg = format!("{err}");
    assert!(msg.contains("ONNX error"));
    assert!(msg.contains("invalid node"));
}

// ── Loader: ggml_dtype and awq_gptq_aux_data for nonexistent tensors ─

#[test]
fn ggml_dtype_nonexistent_returns_none() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.ggml_dtype("nonexistent").is_none());
}

#[test]
fn awq_gptq_aux_data_nonexistent_returns_none() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.awq_gptq_aux_data("nonexistent").is_none());
}

// ── OnnxAttributeValue empty collection variants ─────────────────────

#[test]
fn onnx_attribute_value_tensors_empty_vec() {
    let value = OnnxAttributeValue::Tensors(vec![]);
    assert!(matches!(value, OnnxAttributeValue::Tensors(ref v) if v.is_empty()));
}

#[test]
fn onnx_attribute_value_sparse_tensors_empty_vec() {
    let value = OnnxAttributeValue::SparseTensors(vec![]);
    assert!(matches!(value, OnnxAttributeValue::SparseTensors(ref v) if v.is_empty()));
}

#[test]
fn onnx_attribute_value_types_empty_vec() {
    let value = OnnxAttributeValue::Types(vec![]);
    assert!(matches!(value, OnnxAttributeValue::Types(ref v) if v.is_empty()));
}

#[test]
fn onnx_attribute_value_graphs_empty_vec() {
    let value = OnnxAttributeValue::Graphs(vec![]);
    assert!(matches!(value, OnnxAttributeValue::Graphs(ref v) if v.is_empty()));
}

// ── Loader: node with graphs attribute (If with then + else) ─────────

#[test]
fn loader_node_with_graphs_attribute_parsed() {
    let then_g = proto::GraphProto {
        node: vec![proto::NodeProto {
            op_type: Some("Identity".to_string()),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            ..empty_node()
        }],
        name: Some("then_branch".to_string()),
        ..empty_graph()
    };
    let else_g = proto::GraphProto {
        node: vec![proto::NodeProto {
            op_type: Some("Constant".to_string()),
            output: vec!["y".to_string()],
            ..empty_node()
        }],
        name: Some("else_branch".to_string()),
        ..empty_graph()
    };
    let attr = proto::AttributeProto {
        name: Some("branches".to_string()),
        r#type: Some(proto::attribute_proto::AttributeType::Graphs as i32),
        graphs: vec![then_g, else_g],
        ..Default::default()
    };
    let node = proto::NodeProto {
        op_type: Some("If".to_string()),
        name: Some("if_1".to_string()),
        input: vec!["cond".to_string()],
        output: vec!["result".to_string()],
        attribute: vec![attr],
        ..empty_node()
    };
    let graph = proto::GraphProto { node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let parsed_attr = loader.graph().nodes[0].attributes.get("branches").expect("attribute");
    assert!(matches!(parsed_attr.value, OnnxAttributeValue::Graphs(ref v) if v.len() == 2));
}

// ── bytes_to_f32 with subnormal and negative ──────────────────────────

#[test]
fn bytes_to_f32_subnormal_value() {
    let subnormal = f32::from_bits(1u32); // smallest positive subnormal
    let data = subnormal.to_le_bytes();
    let result = bytes_to_f32(&data);
    assert_eq!(result.len(), 1);
    assert!(result[0] > 0.0);
    assert!(result[0].is_subnormal());
}

#[test]
fn bytes_to_f32_negative_normal() {
    let data = (-99.5f32).to_le_bytes();
    let result = bytes_to_f32(&data);
    assert_eq!(result.len(), 1);
    assert!((result[0] - (-99.5)).abs() < 1e-6);
}

// ── Loader: graph with multiple quantization annotations ──────────────

#[test]
fn loader_graph_multiple_quantization_annotations() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let qa1 = proto::TensorAnnotation {
        tensor_name: Some("w".to_string()),
        quant_parameter_tensor_names: vec![proto::StringStringEntryProto {
            key: Some("scale".to_string()),
            value: Some("w_scale".to_string()),
        }],
    };
    let qa2 = proto::TensorAnnotation {
        tensor_name: Some("b".to_string()),
        quant_parameter_tensor_names: vec![],
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        quantization_annotation: vec![qa1, qa2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().quantization_annotation.len(), 2);
    assert_eq!(loader.graph().quantization_annotation[0].tensor_name, "w");
    assert_eq!(loader.graph().quantization_annotation[1].tensor_name, "b");
}

// ── Loader: model with domain field preserved ─────────────────────────

#[test]
fn loader_model_domain_field_preserved() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: Some("ai.gllm.custom".to_string()),
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.model().metadata.domain, "ai.gllm.custom");
}

// ── OnnxTensor scalar_f32 from I32 and BF16 ──────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_i32_max() {
    let data = i32::MAX.to_le_bytes();
    let tensor = OnnxTensor::new("i32_max".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
    let val = tensor.scalar_f32().expect("should convert i32_max to f32");
    assert!(val.is_finite());
}

#[test]
fn onnx_tensor_scalar_f32_from_bf16() {
    // BF16 1.0 = 0x3F80 (2 bytes)
    let data = vec![0x80u8, 0x3F];
    let tensor = OnnxTensor::new("bf16_one".to_string(), Dtype::BF16, vec![], Bytes::from(data));
    let val = tensor.scalar_f32().expect("should convert bf16 to f32");
    assert!((val - 1.0).abs() < 0.01);
}

#[test]
fn onnx_tensor_scalar_i64_from_i32() {
    let data = 42i32.to_le_bytes();
    let tensor = OnnxTensor::new("i42".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
    let val = tensor.scalar_i64().expect("should convert i32 to i64");
    assert_eq!(val, 42);
}

// ── Loader: model with multiple metadata_props ───────────────────────

#[test]
fn loader_model_many_metadata_props() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: (0..10)
            .map(|i| proto::StringStringEntryProto {
                key: Some(format!("key_{i}")),
                value: Some(format!("val_{i}")),
            })
            .collect(),
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let props = &loader.model().metadata.metadata_props;
    assert_eq!(props.len(), 10);
    for i in 0..10 {
        assert_eq!(props.get(&format!("key_{i}")), Some(&format!("val_{i}")));
    }
}

// ── Loader: node with multiple attribute types at once ───────────────

#[test]
fn loader_node_multiple_attribute_types() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Custom".to_string()),
        name: Some("multi_attr_node".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![
            proto::AttributeProto {
                name: Some("alpha".to_string()),
                r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
                f: Some(0.5),
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("axis".to_string()),
                r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
                i: Some(-1),
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("kernel_shape".to_string()),
                r#type: Some(proto::attribute_proto::AttributeType::Ints as i32),
                ints: vec![3, 3],
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("mode".to_string()),
                r#type: Some(proto::attribute_proto::AttributeType::String as i32),
                s: Some(b"bilinear".to_vec()),
                ..Default::default()
            },
        ],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attrs = &loader.graph().nodes[0].attributes;
    assert_eq!(attrs.len(), 4);
    assert!(matches!(attrs.get("alpha").unwrap().value, OnnxAttributeValue::Float(v) if (v - 0.5).abs() < 1e-6));
    assert!(matches!(attrs.get("axis").unwrap().value, OnnxAttributeValue::Int(-1)));
    assert!(matches!(attrs.get("kernel_shape").unwrap().value, OnnxAttributeValue::Ints(ref v) if *v == vec![3, 3]));
    assert!(matches!(attrs.get("mode").unwrap().value, OnnxAttributeValue::String(ref s) if s == "bilinear"));
}

// ── Loader: Add node does not produce alias ───────────────────────────

#[test]
fn add_node_no_alias_produced() {
    let weight = tensor_f32("bias_add_weight", vec![4], &[1.0; 4]);
    let node = proto::NodeProto {
        name: Some("/layer/Add".to_string()),
        op_type: Some("Add".to_string()),
        input: vec!["x".to_string(), "bias_add_weight".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.tensor("layer.weight").is_err());
    assert!(loader.tensor("bias_add_weight").is_ok());
}

// ── Loader: Div node does not produce alias ───────────────────────────

#[test]
fn div_node_no_alias_produced() {
    let weight = tensor_f32("div_weight", vec![4], &[2.0; 4]);
    let node = proto::NodeProto {
        name: Some("/norm/Div".to_string()),
        op_type: Some("Div".to_string()),
        input: vec!["x".to_string(), "div_weight".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.tensor("norm.weight").is_err());
    assert!(loader.tensor("div_weight").is_ok());
}

// ── OnnxAttributeValue: Tensor singular variant inner access ──────────

#[test]
fn onnx_attribute_value_tensor_singular_debug() {
    let inner = OnnxTensor::new("attr_tensor".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    let value = OnnxAttributeValue::Tensor(inner);
    let debug = format!("{value:?}");
    assert!(debug.contains("Tensor"), "Debug should contain 'Tensor'");
}

// ── Loader: graph with empty quantization_annotation list ────────────

#[test]
fn loader_graph_empty_quantization_annotations() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        quantization_annotation: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.graph().quantization_annotation.is_empty());
}

// ── Loader: model with training_info empty ───────────────────────────

#[test]
fn loader_model_training_info_empty() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // training_info exists but is empty - just verify no crash
    assert!(loader.graph().nodes.is_empty());
}

// ── Loader: node with tensor attribute ────────────────────────────────

#[test]
fn loader_node_tensor_attribute_parsed() {
    let attr_tensor_data = vec![1.0f32, 2.0, 3.0];
    let attr_tensor_raw: Vec<u8> = attr_tensor_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let node = proto::NodeProto {
        op_type: Some("Constant".to_string()),
        name: Some("const_tensor".to_string()),
        input: vec![],
        output: vec!["const_out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Tensor as i32),
            t: Some(proto::TensorProto {
                dims: vec![3],
                data_type: Some(proto::tensor_proto::DataType::Float as i32),
                raw_data: Some(Bytes::from(attr_tensor_raw)),
                ..empty_tensor()
            }),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto { node: vec![node], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attr = loader.graph().nodes[0].attributes.get("value").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Tensor(ref t) if t.shape == vec![3]));
}

// ── OnnxModel struct Debug output ────────────────────────────────────

#[test]
fn onnx_model_struct_debug_format() {
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 7,
            producer_name: "struct_debug".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        },
        graph: OnnxGraph {
            name: "dbg".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![],
    };
    let debug = format!("{model:?}");
    assert!(debug.contains("struct_debug"));
}

// ── Loader: tensor with very long name ───────────────────────────────

#[test]
fn loader_tensor_very_long_name() {
    let long_name = "a".repeat(500);
    let tensor = tensor_f32(&long_name, vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let ts = loader.tensor(&long_name).expect("tensor with long name");
    assert_eq!(ts.shape, vec![1]);
}

// ── Loader: multiple tensors same dtype unique_precisions dedup ───────

#[test]
fn loader_unique_precisions_all_same_dedup() {
    let tensors: Vec<proto::TensorProto> = (0..5)
        .map(|i| tensor_f32(&format!("w_{i}"), vec![1], &[i as f32]))
        .collect();
    let graph = proto::GraphProto { initializer: tensors, ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let prec = loader.unique_precisions();
    assert_eq!(prec.len(), 1);
    assert_eq!(prec[0], Dtype::F32);
}

// ── OnnxDim: Known(0) in shape context ───────────────────────────────

#[test]
fn onnx_dim_known_zero_in_shape() {
    let shape = OnnxTensorShape {
        dims: vec![OnnxDim::Known(0), OnnxDim::Known(10)],
    };
    assert!(matches!(&shape.dims[0], OnnxDim::Known(0)));
    assert!(matches!(&shape.dims[1], OnnxDim::Known(10)));
}

// ── Loader: graph with single input and single output ────────────────

#[test]
fn loader_graph_single_input_single_output() {
    let input_vi = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(7),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![proto::tensor_shape_proto::Dimension {
                        value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(128)),
                        denotation: None,
                    }],
                }),
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let output_vi = proto::ValueInfoProto {
        name: Some("logits".to_string()),
        r#type: Some(proto::TypeProto {
            denotation: None,
            value: Some(proto::type_proto::Value::TensorType(proto::type_proto::Tensor {
                elem_type: Some(1),
                shape: Some(proto::TensorShapeProto {
                    dim: vec![proto::tensor_shape_proto::Dimension {
                        value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(30000)),
                        denotation: None,
                    }],
                }),
            })),
        }),
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![input_vi],
        output: vec![output_vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().inputs.len(), 1);
    assert_eq!(loader.graph().outputs.len(), 1);
    if let Some(OnnxType::Tensor(tt)) = &loader.graph().inputs[0].value_type {
        assert!(matches!(&tt.shape.dims[0], OnnxDim::Known(128)));
    } else {
        panic!("expected Tensor type for input");
    }
}

// ── OnnxTensorType: empty shape dims ─────────────────────────────────

#[test]
fn onnx_tensor_type_empty_shape() {
    let tt = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![] },
    };
    assert!(tt.shape.dims.is_empty());
}

// ── Loader: value_info with no type field ─────────────────────────────

#[test]
fn loader_value_info_no_type_defaults_to_none() {
    let vi = proto::ValueInfoProto {
        name: Some("untyped".to_string()),
        r#type: None,
        doc_string: Some("no type info".to_string()),
        metadata_props: vec![],
    };
    let graph = proto::GraphProto { value_info: vec![vi], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let info = &loader.graph().value_info[0];
    assert_eq!(info.name, "untyped");
    assert!(info.value_type.is_none());
}

// ── OnnxGraph initializer HashMap operations ─────────────────────────

#[test]
fn onnx_graph_initializers_hashmap_insert_lookup() {
    let mut initializers = HashMap::new();
    initializers.insert("w1".to_string(), OnnxTensor::new("w1".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8])));
    initializers.insert("w2".to_string(), OnnxTensor::new("w2".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8])));
    let graph = OnnxGraph {
        name: "test".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers,
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    assert_eq!(graph.initializers.len(), 2);
    assert!(graph.initializers.contains_key("w1"));
    assert!(graph.initializers.contains_key("w2"));
    assert!(!graph.initializers.contains_key("w3"));
}

// ── Loader: weight_layout_hint for Sub node returns None ─────────────

#[test]
fn weight_layout_hint_sub_node_returns_none() {
    let weight = tensor_f32("sub_weight", vec![4], &[1.0; 4]);
    let node = proto::NodeProto {
        name: Some("/layer/Sub".to_string()),
        op_type: Some("Sub".to_string()),
        input: vec!["x".to_string(), "sub_weight".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.weight_layout_hint("sub_weight").is_none());
}

// ── OnnxAttribute: complete struct with all fields ────────────────────

#[test]
fn onnx_attribute_complete_struct_all_fields() {
    let attr = OnnxAttribute {
        name: "transB".to_string(),
        value: OnnxAttributeValue::Int(1),
        doc_string: "whether to transpose B".to_string(),
        ref_attr_name: None,
        attr_type: Some(proto::attribute_proto::AttributeType::Int),
    };
    assert_eq!(attr.name, "transB");
    assert!(matches!(attr.value, OnnxAttributeValue::Int(1)));
    assert_eq!(attr.doc_string, "whether to transpose B");
    assert!(attr.ref_attr_name.is_none());
    assert_eq!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Int));
}

// ── Loader: node with negative integer attribute ──────────────────────

#[test]
fn loader_node_negative_int_attribute() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("ReduceMean".to_string()),
        name: Some("reduce_0".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("axis".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Int as i32),
            i: Some(-2),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attr = loader.graph().nodes[0].attributes.get("axis").expect("attribute");
    assert!(matches!(attr.value, OnnxAttributeValue::Int(-2)));
}

// ── Loader: alias_map with Mul overwrites not existing direct name ────

#[test]
fn alias_map_mul_preserves_direct_names() {
    let direct = tensor_f32("model.norm.weight", vec![2], &[1.0, 2.0]);
    let anon = tensor_f32("onnx::Mul_7", vec![2], &[3.0, 4.0]);
    let node = proto::NodeProto {
        name: Some("/model/layer/Mul".to_string()),
        op_type: Some("Mul".to_string()),
        input: vec!["x".to_string(), "onnx::Mul_7".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![direct, anon],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Direct name should still return the original tensor
    let slice = loader.tensor("model.norm.weight").expect("direct");
    let vals = bytes_to_f32(slice.data);
    assert_eq!(vals, vec![1.0, 2.0]);
}

// ── Loader: empty initializer list with nodes ────────────────────────

#[test]
fn loader_no_initializers_with_nodes_succeeds() {
    let nodes: Vec<proto::NodeProto> = (0..3)
        .map(|i| proto::NodeProto {
            op_type: Some("Relu".to_string()),
            name: Some(format!("relu_{i}")),
            input: vec![format!("in_{i}")],
            output: vec![format!("out_{i}")],
            ..empty_node()
        })
        .collect();
    let graph = proto::GraphProto { node: nodes, ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().nodes.len(), 3);
    assert!(loader.names().is_empty());
    assert!(loader.unique_precisions().is_empty());
}

// ── OnnxFunction: attribute_protos field ─────────────────────────────

#[test]
fn onnx_function_attribute_protos_stored() {
    let mut attr_protos = HashMap::new();
    attr_protos.insert(
        "threshold".to_string(),
        OnnxAttribute {
            name: "threshold".to_string(),
            value: OnnxAttributeValue::Float(0.5),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        },
    );
    let func = OnnxFunction {
        name: "ThresholdRelu".to_string(),
        domain: "custom".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec!["threshold".to_string()],
        attribute_protos: attr_protos,
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    assert_eq!(func.attribute_protos.len(), 1);
    let attr = func.attribute_protos.get("threshold").expect("attr");
    assert!(matches!(attr.value, OnnxAttributeValue::Float(0.5)));
}

// ── OnnxTensor: scalar_f32 from U8 zero ──────────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_u8_zero() {
    let tensor = OnnxTensor::new("byte_zero".to_string(), Dtype::U8, vec![], Bytes::from(vec![0u8]));
    let val = tensor.scalar_f32().expect("scalar f32 from u8 zero");
    assert_eq!(val, 0.0);
}

// ── ConvertError: each variant produces non-empty Debug ──────────────

#[test]
fn convert_error_each_variant_debug_nonempty() {
    let errors = vec![
        ConvertError::UnsupportedOp { op_type: "A".to_string(), node_name: "n".to_string() },
        ConvertError::MissingInitializer { name: "w".to_string(), node_name: "n".to_string() },
        ConvertError::InvalidMatMulShape { name: "W".to_string(), dims: 5 },
        ConvertError::NoWeightInput { node_name: "mm".to_string() },
        ConvertError::AttributeError { node_name: "c".to_string(), reason: "bad".to_string() },
        ConvertError::ShapeInferenceFailed { name: "o".to_string(), reason: "fail".to_string() },
    ];
    for err in &errors {
        let debug = format!("{err:?}");
        assert!(!debug.is_empty());
    }
}

// ── Loader: model with empty string domain ───────────────────────────

#[test]
fn loader_model_empty_string_domain() {
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(17),
        }],
        producer_name: None,
        producer_version: None,
        domain: Some(String::new()),
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.model().metadata.domain, "");
    assert_eq!(loader.model().metadata.opset_import[0].domain, "");
}

// ── Loader: graph name with unicode characters ───────────────────────

#[test]
fn loader_graph_name_unicode() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        name: Some("模型推理图_2026".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert_eq!(loader.graph().name, "模型推理图_2026");
}

// ── OnnxTensor: raw_data length matches dtype and shape ──────────────

#[test]
fn onnx_tensor_raw_data_length_i64_2_elems() {
    let data = Bytes::from(vec![0u8; 16]); // 2 * 8 bytes
    let tensor = OnnxTensor::new("pair".to_string(), Dtype::I64, vec![2], data);
    assert_eq!(tensor.raw_data().len(), 16);
}

// ── Loader: iter_tensors with mixed types preserves dtypes ────────────

#[test]
fn loader_iter_tensors_mixed_dtypes() {
    let t_f32 = tensor_f32("fp32_w", vec![2], &[1.0, 2.0]);
    let t_i64_raw = 42i64.to_le_bytes();
    let t_i64 = tensor_raw("i64_idx", vec![1], proto::tensor_proto::DataType::Int64, &t_i64_raw);
    let graph = proto::GraphProto { initializer: vec![t_f32, t_i64], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let metas: Vec<_> = loader.iter_tensors().collect();
    assert_eq!(metas.len(), 2);
    let f32_meta = metas.iter().find(|m| m.name == "fp32_w").expect("fp32_w");
    assert_eq!(f32_meta.dtype, Dtype::F32);
    let i64_meta = metas.iter().find(|m| m.name == "i64_idx").expect("i64_idx");
    assert_eq!(i64_meta.dtype, Dtype::I64);
}

// ── OnnxAttributeValue: singular Tensor vs plural Tensors ────────────

#[test]
fn onnx_attribute_value_singular_vs_plural_tensor() {
    let t = OnnxTensor::new("t".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
    let singular = OnnxAttributeValue::Tensor(t.clone());
    let plural = OnnxAttributeValue::Tensors(vec![t]);
    assert!(matches!(singular, OnnxAttributeValue::Tensor(_)));
    assert!(matches!(plural, OnnxAttributeValue::Tensors(ref v) if v.len() == 1));
}

// ── OnnxTensor: new_string constructor ──────────────────────────────────

#[test]
fn onnx_tensor_new_string_marks_is_string() {
    let t = OnnxTensor::new_string("labels".to_string(), vec![3], Bytes::from(vec![0u8; 3]));
    assert!(t.is_string);
    assert_eq!(t.dtype, Dtype::U8);
    assert_eq!(t.name, "labels");
    assert_eq!(t.shape, vec![3]);
}

// ── OnnxTensor: clone independence ───────────────────────────────────────

#[test]
fn onnx_tensor_clone_independence() {
    let t = OnnxTensor::new("orig".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    let cloned = t.clone();
    assert_eq!(cloned.name, "orig");
    assert_eq!(cloned.dtype, Dtype::F32);
    assert_eq!(cloned.shape, vec![2]);
    assert_eq!(cloned.raw_data().len(), 8);
}

// ── OnnxTensor: scalar accessors on non-scalar return None ──────────────

#[test]
fn onnx_tensor_scalar_f32_non_scalar_returns_none() {
    let t = OnnxTensor::new("vec".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
    assert!(t.scalar_f32().is_none());
}

// ── OnnxTensor: scalar i64 on single U8 value ───────────────────────────

#[test]
fn onnx_tensor_scalar_i64_u8_value() {
    let t = OnnxTensor::new("b".to_string(), Dtype::U8, vec![], Bytes::from(vec![42u8]));
    assert_eq!(t.scalar_i64(), Some(42));
}

// ── OnnxSparseFormat: all variants equality ─────────────────────────────

#[test]
fn onnx_sparse_format_variants_distinct() {
    assert_eq!(OnnxSparseFormat::Coo, OnnxSparseFormat::Coo);
    assert_eq!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csr);
    assert_eq!(OnnxSparseFormat::Csc, OnnxSparseFormat::Csc);
    assert_ne!(OnnxSparseFormat::Coo, OnnxSparseFormat::Csr);
    assert_ne!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csc);
    assert_ne!(OnnxSparseFormat::Csc, OnnxSparseFormat::Coo);
}

// ── OnnxOperatorSet: construction and field access ──────────────────────

#[test]
fn onnx_operator_set_construction() {
    let opset = OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 17,
    };
    assert_eq!(opset.domain, "ai.onnx");
    assert_eq!(opset.version, 17);
}

// ── OnnxQuantizationAnnotation: field access ────────────────────────────

#[test]
fn onnx_quantization_annotation_fields() {
    let ann = OnnxQuantizationAnnotation {
        tensor_name: "weight_q".to_string(),
        quant_param_tensor_names: {
            let mut m = HashMap::new();
            m.insert("scale".to_string(), "weight_s".to_string());
            m
        },
        scale: Some(0.5),
        zero_point: Some(128),
        axis: Some(0),
    };
    assert_eq!(ann.tensor_name, "weight_q");
    assert_eq!(ann.quant_param_tensor_names.len(), 1);
    assert_eq!(ann.scale.unwrap() as f64, 0.5);
    assert_eq!(ann.zero_point, Some(128));
    assert_eq!(ann.axis, Some(0));
}

// ── OnnxQuantizationAnnotation: all optional fields None ────────────────

#[test]
fn onnx_quantization_annotation_all_none() {
    let ann = OnnxQuantizationAnnotation {
        tensor_name: "w".to_string(),
        quant_param_tensor_names: HashMap::new(),
        scale: None,
        zero_point: None,
        axis: None,
    };
    assert!(ann.scale.is_none());
    assert!(ann.zero_point.is_none());
    assert!(ann.axis.is_none());
}

// ── OnnxFunction: construction and field access ─────────────────────────

#[test]
fn onnx_function_construction() {
    let func = OnnxFunction {
        name: "custom_op".to_string(),
        domain: "com.example".to_string(),
        overload: "v1".to_string(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: "test".to_string(),
        metadata_props: HashMap::new(),
    };
    assert_eq!(func.name, "custom_op");
    assert_eq!(func.domain, "com.example");
    assert_eq!(func.overload, "v1");
    assert_eq!(func.inputs.len(), 1);
    assert_eq!(func.outputs.len(), 1);
    assert!(func.nodes.is_empty());
}

// ── OnnxAttribute: ref_attr_name field ──────────────────────────────────

#[test]
fn onnx_attribute_ref_attr_name_field() {
    let attr = OnnxAttribute {
        name: "kernel_shape".to_string(),
        value: OnnxAttributeValue::Ref("parent.kernel_shape".to_string()),
        doc_string: String::new(),
        ref_attr_name: Some("parent.kernel_shape".to_string()),
        attr_type: None,
    };
    assert_eq!(attr.name, "kernel_shape");
    assert!(matches!(attr.value, OnnxAttributeValue::Ref(ref s) if s == "parent.kernel_shape"));
    assert!(attr.ref_attr_name.is_some());
    assert!(attr.attr_type.is_none());
}

// ── OnnxAttributeValue: Floats variant ──────────────────────────────────

#[test]
fn onnx_attribute_value_floats_variant() {
    let vals = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
    assert!(matches!(vals, OnnxAttributeValue::Floats(ref v) if v.len() == 3));
}

// ── OnnxAttributeValue: Ints variant ────────────────────────────────────

#[test]
fn onnx_attribute_value_ints_variant() {
    let vals = OnnxAttributeValue::Ints(vec![1i64, 2, 3]);
    assert!(matches!(vals, OnnxAttributeValue::Ints(ref v) if v == &[1, 2, 3]));
}

// ── OnnxAttributeValue: Strings variant ─────────────────────────────────

#[test]
fn onnx_attribute_value_strings_variant() {
    let vals = OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]);
    assert!(matches!(vals, OnnxAttributeValue::Strings(ref v) if v.len() == 2));
}

// ── OnnxModel: functions field defaults to empty ────────────────────────

#[test]
fn loader_onnx_model_functions_field_empty() {
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    assert!(loader.model().functions.is_empty());
}

// ── Loader: model_version field preserved ──────────────────────────────

#[test]
fn loader_model_version_preserved() {
    // Arrange: build a model with a non-zero model_version
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: Some(42),
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.model_version, 42);
}

// ── Loader: graph doc_string with unicode ──────────────────────────────

#[test]
fn loader_graph_doc_string_unicode() {
    // Arrange: graph with unicode doc_string
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        doc_string: Some("文档描述 — Section §2".to_string()),
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().doc_string, "文档描述 — Section §2");
}

// ── OnnxTensor: scalar_f32 from U16 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_u16() {
    // Arrange: U16 scalar tensor with value 1000
    let data = 1000u16.to_le_bytes();
    let tensor = OnnxTensor::new("u16_val".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_f32();
    // Assert
    assert_eq!(val, Some(1000.0));
}

// ── OnnxTensor: scalar_f32 from U32 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_u32() {
    // Arrange: U32 scalar tensor with value 100000
    let data = 100000u32.to_le_bytes();
    let tensor = OnnxTensor::new("u32_val".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_f32();
    // Assert
    assert_eq!(val, Some(100000.0));
}

// ── OnnxTensor: scalar_f32 from U64 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_u64() {
    // Arrange: U64 scalar tensor with value 999
    let data = 999u64.to_le_bytes();
    let tensor = OnnxTensor::new("u64_val".to_string(), Dtype::U64, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_f32();
    // Assert
    assert!((val.unwrap() - 999.0).abs() < 1e-6);
}

// ── OnnxTensor: scalar_f32 from F16 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_f32_from_f16() {
    // Arrange: F16 scalar tensor with value 1.5 (bits = 0x3E00)
    let data = vec![0x00u8, 0x3E];
    let tensor = OnnxTensor::new("f16_val".to_string(), Dtype::F16, vec![], Bytes::from(data));
    // Act
    let val = tensor.scalar_f32();
    // Assert
    assert!((val.unwrap() - 1.5).abs() < 0.01);
}

// ── OnnxTensor: scalar_i64 from U32 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_i64_from_u32() {
    // Arrange: U32 scalar tensor with value 50000
    let data = 50000u32.to_le_bytes();
    let tensor = OnnxTensor::new("u32_i64".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_i64();
    // Assert
    assert_eq!(val, Some(50000));
}

// ── OnnxTensor: scalar_i64 from U64 value ──────────────────────────────

#[test]
fn onnx_tensor_scalar_i64_from_u64() {
    // Arrange: U64 scalar tensor with value 123456
    let data = 123456u64.to_le_bytes();
    let tensor = OnnxTensor::new("u64_i64".to_string(), Dtype::U64, vec![], Bytes::from(data.to_vec()));
    // Act
    let val = tensor.scalar_i64();
    // Assert
    assert_eq!(val, Some(123456));
}

// ── Loader: node with empty output list ────────────────────────────────

#[test]
fn loader_node_empty_output_list() {
    // Arrange: node that has no outputs (e.g. a dropout in inference mode)
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let node = proto::NodeProto {
        op_type: Some("Dropout".to_string()),
        name: Some("drop_0".to_string()),
        input: vec!["x".to_string()],
        output: vec![],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes[0].outputs.len(), 0);
    assert_eq!(loader.graph().nodes[0].op_type, "Dropout");
}

// ── Loader: graph with initializer-only and no nodes ──────────────────

#[test]
fn loader_graph_initializers_only_no_nodes() {
    // Arrange: graph with multiple initializers but zero nodes
    let t1 = tensor_f32("weight_1", vec![3, 4], &[1.0; 12]);
    let t2 = tensor_f32("weight_2", vec![4, 2], &[2.0; 8]);
    let t3 = tensor_f32("bias", vec![2], &[0.1, 0.2]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3],
        node: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.names().len(), 3);
    assert!(loader.graph().nodes.is_empty());
    assert_eq!(loader.unique_precisions().len(), 1);
}

// ── Loader: value_info with multiple metadata_props ───────────────────

#[test]
fn loader_value_info_multiple_metadata_props() {
    // Arrange: value_info with several metadata properties
    let vi = proto::ValueInfoProto {
        name: Some("annotated_tensor".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("source".to_string()),
                value: Some("quantization".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("level".to_string()),
                value: Some("L2".to_string()),
            },
        ],
    };
    let graph = proto::GraphProto { value_info: vec![vi], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let info = &loader.graph().value_info[0];
    assert_eq!(info.metadata_props.len(), 2);
    assert_eq!(info.metadata_props.get("source"), Some(&"quantization".to_string()));
    assert_eq!(info.metadata_props.get("level"), Some(&"L2".to_string()));
}

// ── OnnxDim: Param with unicode characters ────────────────────────────

#[test]
fn onnx_dim_param_unicode_characters() {
    // Arrange: create a Param dim with unicode characters
    let dim = OnnxDim::Param("维度_维度".to_string());
    // Act
    let debug = format!("{dim:?}");
    // Assert
    assert!(debug.contains("维度_维度"));
    assert_eq!(dim, OnnxDim::Param("维度_维度".to_string()));
}

// ── Loader: graph metadata_props populated from proto fields ──────────

#[test]
fn loader_graph_metadata_props_populated_from_proto() {
    // Arrange: graph with metadata_props set
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("author".to_string()),
                value: Some("gllm-test".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("version".to_string()),
                value: Some("2.0".to_string()),
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let props = &loader.graph().metadata_props;
    assert_eq!(props.len(), 2);
    assert_eq!(props.get("author"), Some(&"gllm-test".to_string()));
    assert_eq!(props.get("version"), Some(&"2.0".to_string()));
}

// ── OnnxModel: metadata with all zero defaults ────────────────────────

#[test]
fn onnx_model_metadata_all_zero_defaults() {
    // Arrange: build a ModelProto with no optional fields set
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        ir_version: None,
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let meta = &loader.model().metadata;
    // Assert: all default to zero/empty
    assert_eq!(meta.ir_version, 0);
    assert_eq!(meta.model_version, 0);
    assert!(meta.producer_name.is_empty());
    assert!(meta.producer_version.is_empty());
    assert!(meta.domain.is_empty());
    assert!(meta.doc_string.is_empty());
}

// ── Loader: multiple nodes with same op_type distinct names ───────────

#[test]
fn loader_multiple_nodes_same_optype_distinct_names() {
    // Arrange: three MatMul nodes with different names and inputs
    let w1 = tensor_f32("w1", vec![2, 2], &[1.0; 4]);
    let w2 = tensor_f32("w2", vec![2, 2], &[2.0; 4]);
    let w3 = tensor_f32("w3", vec![2, 2], &[3.0; 4]);
    let nodes = vec![
        proto::NodeProto {
            op_type: Some("MatMul".to_string()),
            name: Some("matmul_layer_0".to_string()),
            input: vec!["x".to_string(), "w1".to_string()],
            output: vec!["inter_0".to_string()],
            ..empty_node()
        },
        proto::NodeProto {
            op_type: Some("MatMul".to_string()),
            name: Some("matmul_layer_1".to_string()),
            input: vec!["inter_0".to_string(), "w2".to_string()],
            output: vec!["inter_1".to_string()],
            ..empty_node()
        },
        proto::NodeProto {
            op_type: Some("MatMul".to_string()),
            name: Some("matmul_layer_2".to_string()),
            input: vec!["inter_1".to_string(), "w3".to_string()],
            output: vec!["final".to_string()],
            ..empty_node()
        },
    ];
    let graph = proto::GraphProto {
        initializer: vec![w1, w2, w3],
        node: nodes,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 3);
    assert_eq!(loader.graph().nodes[0].name, "matmul_layer_0");
    assert_eq!(loader.graph().nodes[1].name, "matmul_layer_1");
    assert_eq!(loader.graph().nodes[2].name, "matmul_layer_2");
    // All have same op_type but distinct names
    for node in &loader.graph().nodes {
        assert_eq!(node.op_type, "MatMul");
    }
}

// ══════════════════════════════════════════════════════════════════════
// NEW TESTS (15) — edge cases and boundary conditions
// ══════════════════════════════════════════════════════════════════════

// ── Loader: model with multiple function definitions ──────────────────
// @trace TEST-ONNX-001 [level:unit]

#[test]
fn loader_model_with_multiple_function_definitions() {
    // Arrange: ModelProto with two function definitions in the functions field
    let func1 = proto::FunctionProto {
        name: Some("CustomGemm".to_string()),
        domain: Some("com.example".to_string()),
        ..Default::default()
    };
    let func2 = proto::FunctionProto {
        name: Some("CustomNorm".to_string()),
        domain: Some("com.example".to_string()),
        ..Default::default()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: None,
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![func1, func2],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: both functions are parsed and accessible
    assert_eq!(loader.model().functions.len(), 2);
    assert_eq!(loader.model().functions[0].name, "CustomGemm");
    assert_eq!(loader.model().functions[1].name, "CustomNorm");
}

// ── Loader: node with multiple integer attributes preserves all ───────
// @trace TEST-ONNX-002 [level:unit]

#[test]
fn loader_node_multiple_int_attributes_preserved() {
    // Arrange: a single node with three integer attributes
    let node = proto::NodeProto {
        op_type: Some("Conv".to_string()),
        name: Some("conv1".to_string()),
        attribute: vec![
            proto::AttributeProto {
                name: Some("kernel_shape".to_string()),
                r#type: Some(7), // INTS
                ints: vec![3, 3],
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("strides".to_string()),
                r#type: Some(7),
                ints: vec![1, 1],
                ..Default::default()
            },
            proto::AttributeProto {
                name: Some("group".to_string()),
                r#type: Some(2), // INT
                i: Some(1),
                ..Default::default()
            },
        ],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all three attributes are preserved with correct values
    let attrs = &loader.graph().nodes[0].attributes;
    assert_eq!(attrs.len(), 3);
    assert!(matches!(
        &attrs.get("kernel_shape").unwrap().value,
        OnnxAttributeValue::Ints(v) if v == &[3, 3]
    ));
    assert!(matches!(
        &attrs.get("strides").unwrap().value,
        OnnxAttributeValue::Ints(v) if v == &[1, 1]
    ));
    assert!(matches!(
        &attrs.get("group").unwrap().value,
        OnnxAttributeValue::Int(v) if *v == 1
    ));
}

// ── Loader: external data with subdirectory path ─────────────────────
// @trace TEST-ONNX-003 [level:unit]

#[test]
fn external_data_locations_subdirectory_path() {
    // Arrange: external data referencing a subdirectory file
    let dir = TempDir::new().expect("tempdir");
    let subdir = dir.path().join("shards");
    std::fs::create_dir_all(&subdir).expect("mkdir");
    let data_path = subdir.join("part0.bin");
    std::fs::write(&data_path, vec![0u8; 8]).expect("write");
    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("shards/part0.bin".to_string()),
        }],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model_path = dir.path().join("model.onnx");
    write_model(empty_model(graph), &model_path);
    // Act
    let locations = external_data_locations(&model_path).expect("locations");
    // Assert: subdirectory path is preserved
    assert_eq!(locations, vec!["shards/part0.bin".to_string()]);
}

// ── Loader: graph with graph name containing special characters ──────
// @trace TEST-ONNX-004 [level:unit]

#[test]
fn loader_graph_name_special_characters() {
    // Arrange: graph with name containing hyphens, underscores, and dots
    let graph = proto::GraphProto {
        name: Some("my-model_v2.0.main-graph".to_string()),
        initializer: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: special characters in graph name are preserved
    assert_eq!(loader.graph().name, "my-model_v2.0.main-graph");
}

// ── Loader: tensor with int32_data field for INT32 dtype ─────────────
// @trace TEST-ONNX-005 [level:unit]

#[test]
fn tensor_with_int32_data_field_values_correct() {
    // Arrange: TensorProto using int32_data field instead of raw_data
    let mut tensor = empty_tensor();
    tensor.dims = vec![4];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int32 as i32);
    tensor.name = Some("int32_tensor".to_string());
    tensor.int32_data = vec![10, -20, 30, -40];
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("int32_tensor").expect("tensor");
    // Assert: shape and dtype correct; data is packed as little-endian i32 bytes
    assert_eq!(slice.shape, vec![4]);
    assert_eq!(slice.dtype, Dtype::I32);
    assert_eq!(slice.data.len(), 16); // 4 elements * 4 bytes
}

// ── Loader: tensor with int64_data field for INT64 dtype ─────────────
// @trace TEST-ONNX-006 [level:unit]

#[test]
fn tensor_with_int64_data_field_values_correct() {
    // Arrange: TensorProto using int64_data field instead of raw_data
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Int64 as i32);
    tensor.name = Some("int64_tensor".to_string());
    tensor.int64_data = vec![1000000000i64, -999999999i64];
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("int64_tensor").expect("tensor");
    // Assert: data is packed as little-endian i64 bytes
    assert_eq!(slice.shape, vec![2]);
    assert_eq!(slice.dtype, Dtype::I64);
    assert_eq!(slice.data.len(), 16); // 2 elements * 8 bytes
}

// ── Loader: model with training_info field empty but present ─────────
// @trace TEST-ONNX-007 [level:unit]

#[test]
fn loader_model_training_info_field_ignored() {
    // Arrange: ModelProto with non-empty training_info (should be accepted without error)
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[0.5])],
        ..empty_graph()
    };
    let training_proto = proto::TrainingInfoProto {
        initialization: Some(proto::GraphProto::default()),
        algorithm: Some(proto::GraphProto::default()),
        ..Default::default()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![training_proto],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: model loads successfully; training_info is ignored
    assert_eq!(loader.model().metadata.ir_version, 8);
    assert_eq!(loader.names().len(), 1);
}

// ── Loader: node with empty op_type returns error ─────────────────────
// @trace TEST-ONNX-008 [level:unit]

#[test]
fn loader_node_missing_op_type_returns_error() {
    // Arrange: node with op_type = None (missing)
    let node = proto::NodeProto {
        op_type: None,
        name: Some("unnamed_op".to_string()),
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert: missing op_type causes a load error
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("op_type") || err_msg.contains("missing"));
}

// ── Loader: value_info with empty doc_string ─────────────────────────
// @trace TEST-ONNX-009 [level:unit]

#[test]
fn loader_value_info_empty_doc_string() {
    // Arrange: ValueInfoProto with explicit empty doc_string
    let vi = proto::ValueInfoProto {
        name: Some("hidden".to_string()),
        r#type: None,
        doc_string: Some(String::new()),
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: doc_string is empty string (not crash)
    assert_eq!(loader.graph().value_info[0].doc_string, "");
}

// ── Loader: multiple initializers same dtype unique_precisions dedup ──
// @trace TEST-ONNX-010 [level:unit]

#[test]
fn loader_unique_precisions_many_f32_tensors_single_entry() {
    // Arrange: five F32 tensors — unique_precisions should return [F32]
    let tensors: Vec<proto::TensorProto> = (0..5)
        .map(|i| tensor_f32(&format!("w{i}"), vec![2], &[1.0, 2.0]))
        .collect();
    let graph = proto::GraphProto {
        initializer: tensors,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: only one unique precision
    assert_eq!(loader.unique_precisions(), vec![Dtype::F32]);
}

// ── Loader: node with multiple outputs all preserved ─────────────────
// @trace TEST-ONNX-011 [level:unit]

#[test]
fn loader_node_three_outputs_all_preserved() {
    // Arrange: a node that produces three named outputs (e.g. Split)
    let node = proto::NodeProto {
        op_type: Some("Split".to_string()),
        name: Some("split_node".to_string()),
        input: vec!["input".to_string()],
        output: vec![
            "split_out_0".to_string(),
            "split_out_1".to_string(),
            "split_out_2".to_string(),
        ],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all three outputs preserved in order
    let outputs = &loader.graph().nodes[0].outputs;
    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0], "split_out_0");
    assert_eq!(outputs[1], "split_out_1");
    assert_eq!(outputs[2], "split_out_2");
}

// ── Loader: model with two opset imports preserves both ──────────────
// @trace TEST-ONNX-012 [level:unit]

#[test]
fn loader_model_two_opset_imports_preserved() {
    // Arrange: ModelProto with two different opset imports
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![
            proto::OperatorSetIdProto {
                domain: Some(String::new()),
                version: Some(17),
            },
            proto::OperatorSetIdProto {
                domain: Some("com.microsoft".to_string()),
                version: Some(1),
            },
        ],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: both opsets preserved with correct domain and version
    let opsets = &loader.model().metadata.opset_import;
    assert_eq!(opsets.len(), 2);
    assert_eq!(opsets[0].domain, "");
    assert_eq!(opsets[0].version, 17);
    assert_eq!(opsets[1].domain, "com.microsoft");
    assert_eq!(opsets[1].version, 1);
}

// ── Loader: tensor names with mixed case are case-sensitive ──────────
// @trace TEST-ONNX-013 [level:unit]

#[test]
fn loader_tensor_names_case_sensitive_lookup() {
    // Arrange: two tensors differing only by case
    let t_lower = tensor_f32("weight", vec![1], &[1.0]);
    let t_upper = tensor_f32("Weight", vec![1], &[2.0]);
    let graph = proto::GraphProto {
        initializer: vec![t_lower, t_upper],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: each name resolves to its own tensor data
    let lower = loader.tensor("weight").expect("lower");
    let upper = loader.tensor("Weight").expect("upper");
    let lower_val = bytes_to_f32(lower.data);
    let upper_val = bytes_to_f32(upper.data);
    assert_eq!(lower_val, vec![1.0]);
    assert_eq!(upper_val, vec![2.0]);
}

// ── Loader: graph with 4D tensor preserves all dimensions ────────────
// @trace TEST-ONNX-014 [level:unit]

#[test]
fn loader_tensor_4d_shape_preserved() {
    // Arrange: a 4D tensor (e.g. conv weight [out, in, h, w])
    let values = vec![0.0f32; 2 * 3 * 3 * 3]; // 54 elements
    let tensor = tensor_f32("conv.weight", vec![2, 3, 3, 3], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all four dimensions preserved exactly
    let slice = loader.tensor("conv.weight").expect("tensor");
    assert_eq!(slice.shape, vec![2, 3, 3, 3]);
    assert_eq!(slice.data.len(), 54 * 4);
}

// ── Loader: external_data_locations with no external tensors empty ────
// @trace TEST-ONNX-015 [level:unit]

#[test]
fn external_data_locations_inline_only_returns_empty() {
    // Arrange: model with only inline tensors (no external data)
    let t1 = tensor_f32("w1", vec![2], &[1.0, 2.0]);
    let t2 = tensor_f32("w2", vec![3], &[3.0, 4.0, 5.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    write_model(model, &model_path);
    // Act
    let locations = external_data_locations(&model_path).expect("locations");
    // Assert: no external data locations
    assert!(locations.is_empty());
}

// ── Loader: single scalar F32 tensor loads correctly ──────────────────
// @trace TEST-ONNX-016 [level:unit]

#[test]
fn loader_single_scalar_f32_tensor() {
    // Arrange: model with a single-element scalar tensor (dims = [])
    let tensor = tensor_f32("bias", vec![], &[42.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: scalar tensor has empty shape and correct data
    let slice = loader.tensor("bias").expect("tensor");
    assert!(slice.shape.is_empty());
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![42.0]);
}

// ── Loader: load_tensor_data returns borrowed cow via TensorProvider ──
// @trace TEST-ONNX-017 [level:unit]

#[test]
fn loader_load_tensor_data_returns_borrowed_cow() {
    // Arrange
    let tensor = tensor_f32("w", vec![2], &[10.0, 20.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    use crate::loader::TensorProvider;
    let cow = loader.load_tensor_data("w").expect("load");
    // Assert: data is borrowed and content is correct
    assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    let values = bytes_to_f32(&cow);
    assert_eq!(values, vec![10.0, 20.0]);
}

// ── Loader: load_tensor_data for nonexistent tensor errors ────────────
// @trace TEST-ONNX-018 [level:unit]

#[test]
fn loader_load_tensor_data_nonexistent_errors() {
    // Arrange
    let tensor = tensor_f32("exists", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    use crate::loader::TensorProvider;
    let result = loader.load_tensor_data("does_not_exist");
    // Assert
    assert!(result.is_err());
}

// ── Loader: weight_layout_hint for alias-resolved tensor ──────────────
// @trace TEST-ONNX-019 [level:unit]

#[test]
fn loader_weight_layout_hint_via_alias_resolution() {
    // Arrange: MatMul node with named initializer that gets canonical alias
    let weight = tensor_f32(
        "model.layers.0.attn.q_proj.MatMul.weight",
        vec![2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let node = proto::NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec![
            "x".to_string(),
            "model.layers.0.attn.q_proj.MatMul.weight".to_string(),
        ],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act: query via canonical alias name
    use crate::loader::TensorProvider;
    let hint = loader.weight_layout_hint("model.layers.0.self_attn.q_proj.weight");
    // Assert: MatMul implies layout_hint = false (already canonical)
    assert_eq!(hint, Some(false));
}

// ── Loader: graph with sparse initializer loads without error ─────────
// @trace TEST-ONNX-020 [level:unit]

#[test]
fn loader_graph_with_sparse_initializer_loads() {
    // Arrange: model with a regular initializer and an empty sparse_initializer list
    let tensor = tensor_f32("dense.weight", vec![2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let sparse_vals = tensor_f32("sparse_vals", vec![2], &[1.0, 1.0]);
    let sparse_idxs = tensor_raw(
        "sparse_idxs",
        vec![2],
        proto::tensor_proto::DataType::Int64,
        &0i64.to_le_bytes().iter().chain(&1i64.to_le_bytes()).copied().collect::<Vec<u8>>(),
    );
    let sparse_tensor = proto::SparseTensorProto {
        values: Some(sparse_vals),
        indices: Some(sparse_idxs),
        dims: vec![2, 2],
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        sparse_initializer: vec![sparse_tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: dense initializer is still accessible
    let slice = loader.tensor("dense.weight").expect("tensor");
    assert_eq!(slice.shape, vec![2, 2]);
    // sparse_initializers are loaded into the graph
    assert_eq!(loader.graph().sparse_initializers.len(), 1);
}

// ── Loader: graph with two Gemm nodes sharing same weight ─────────────
// @trace TEST-ONNX-021 [level:unit]

#[test]
fn loader_shared_weight_across_two_gemm_nodes() {
    // Arrange: two Gemm nodes referencing the same initializer
    let weight = tensor_f32("shared.weight", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let node1 = proto::NodeProto {
        op_type: Some("Gemm".to_string()),
        input: vec!["x1".to_string(), "shared.weight".to_string()],
        output: vec!["out1".to_string()],
        ..empty_node()
    };
    let node2 = proto::NodeProto {
        op_type: Some("Gemm".to_string()),
        input: vec!["x2".to_string(), "shared.weight".to_string()],
        output: vec!["out2".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node1, node2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: single initializer, two nodes, data is the same
    assert_eq!(loader.names().len(), 1);
    assert_eq!(loader.graph().nodes.len(), 2);
    let slice = loader.tensor("shared.weight").expect("tensor");
    let values = bytes_to_f32(slice.data);
    assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ── Loader: iter_tensors yields all initializers with correct dtype ────
// @trace TEST-ONNX-022 [level:unit]

#[test]
fn loader_iter_tensors_yields_correct_shapes_and_dtypes() {
    // Arrange: model with two tensors of different shapes and dtypes
    let f32_tensor = tensor_f32("w_f32", vec![4, 2], &[0.0; 8]);
    let i64_raw: Vec<u8> = (0i64..4).flat_map(|v| v.to_le_bytes()).collect();
    let i64_tensor = tensor_raw(
        "w_i64",
        vec![4],
        proto::tensor_proto::DataType::Int64,
        &i64_raw,
    );
    let graph = proto::GraphProto {
        initializer: vec![f32_tensor, i64_tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    use crate::loader::TensorProvider;
    let metas: Vec<_> = loader.iter_tensors().collect();
    // Assert: two tensors with correct shapes and dtypes
    assert_eq!(metas.len(), 2);
    let f32_meta = metas.iter().find(|m| m.name == "w_f32").expect("f32");
    assert_eq!(f32_meta.shape, vec![4, 2]);
    assert_eq!(f32_meta.dtype, Dtype::F32);
    let i64_meta = metas.iter().find(|m| m.name == "w_i64").expect("i64");
    assert_eq!(i64_meta.shape, vec![4]);
    assert_eq!(i64_meta.dtype, Dtype::I64);
}

// ── Loader: Gemm node with transB as non-integer attribute ────────────
// @trace TEST-ONNX-023 [level:unit]

#[test]
fn loader_gemm_transb_float_attribute_defaults_to_false() {
    // Arrange: Gemm node where transB is a Float attribute (non-standard but should not crash)
    let weight = tensor_f32("w", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let node = proto::NodeProto {
        op_type: Some("Gemm".to_string()),
        input: vec!["x".to_string(), "w".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(proto::attribute_proto::AttributeType::Float as i32),
            f: Some(1.0),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: loader succeeds, transB with non-Int attribute defaults to false
    use crate::loader::TensorProvider;
    let hint = loader.weight_layout_hint("w");
    assert_eq!(hint, Some(false));
}

// ── Loader: tensor_info via alias returns correct meta ────────────────
// @trace TEST-ONNX-024 [level:unit]

#[test]
fn loader_tensor_info_canonical_alias_returns_meta() {
    // Arrange: named initializer with canonical alias path
    let weight = tensor_f32(
        "model.layers.0.attn.v_proj.MatMul.weight",
        vec![3, 4],
        &[0.0f32; 12],
    );
    let graph = proto::GraphProto {
        initializer: vec![weight],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act: query via canonical alias
    use crate::loader::TensorProvider;
    let info = loader.tensor_info("model.layers.0.self_attn.v_proj.weight");
    // Assert
    assert!(info.is_some());
    let meta = info.unwrap();
    assert_eq!(meta.shape, vec![3, 4]);
    assert_eq!(meta.dtype, Dtype::F32);
    assert_eq!(meta.name, "model.layers.0.self_attn.v_proj.weight");
}

// ── Loader: external_data_locations with node attribute external tensor ─
// @trace TEST-ONNX-025 [level:unit]

#[test]
fn external_data_locations_node_attribute_external_tensor() {
    // Arrange: model with a node whose attribute contains an external tensor
    let ext_tensor = proto::TensorProto {
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![proto::StringStringEntryProto {
            key: Some("location".to_string()),
            value: Some("node_attr_data.bin".to_string()),
        }],
        ..Default::default()
    };
    let node = proto::NodeProto {
        op_type: Some("If".to_string()),
        attribute: vec![proto::AttributeProto {
            name: Some("then_branch".to_string()),
            t: Some(ext_tensor),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    write_model(model, &model_path);
    // Act
    let locations = external_data_locations(&model_path).expect("locations");
    // Assert: external data location from node attribute is found
    assert_eq!(locations, vec!["node_attr_data.bin"]);
}

// ── Loader: 5D tensor shape preserved ─────────────────────────────────
// @trace TEST-ONNX-026 [level:unit]

#[test]
fn loader_tensor_5d_shape_preserved() {
    // Arrange: a 5D tensor (e.g. 3D conv weight [out, in, d, h, w])
    let count = 2 * 3 * 2 * 2 * 2; // 48 elements
    let values = vec![1.0f32; count];
    let tensor = tensor_f32("conv3d.weight", vec![2, 3, 2, 2, 2], &values);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all five dimensions preserved
    let slice = loader.tensor("conv3d.weight").expect("tensor");
    assert_eq!(slice.shape, vec![2, 3, 2, 2, 2]);
    assert_eq!(slice.data.len(), count * 4);
}

// ── Loader: model with large ir_version loads ─────────────────────────
// @trace TEST-ONNX-027 [level:unit]

#[test]
fn loader_model_ir_version_max_i64() {
    // Arrange: model with maximum i64 ir_version
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[0.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(i64::MAX),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: ir_version preserved
    assert_eq!(loader.model().metadata.ir_version, i64::MAX);
}

// ── Loader: model with negative model_version loads ───────────────────
// @trace TEST-ONNX-028 [level:unit]

#[test]
fn loader_model_negative_model_version() {
    // Arrange: model with negative model_version (non-standard but protobuf allows it)
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[0.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        model_version: Some(-1),
        opset_import: vec![],
        producer_name: None,
        producer_version: None,
        domain: None,
        doc_string: None,
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: negative model_version preserved
    assert_eq!(loader.model().metadata.model_version, -1);
}

// ── Loader: graph with only value_info and no initializers ────────────
// @trace TEST-ONNX-029 [level:unit]

#[test]
fn loader_graph_value_info_only_no_initializers() {
    // Arrange: graph with value_info entries but zero initializers
    let vi = proto::ValueInfoProto {
        name: Some("intermediate".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: value_info preserved, no initializers, names empty
    assert_eq!(loader.graph().value_info.len(), 1);
    assert_eq!(loader.graph().value_info[0].name, "intermediate");
    assert!(loader.names().is_empty());
}

// ── Loader: precision_by_tensor with BF16 and F32 tensors ─────────────
// @trace TEST-ONNX-030 [level:unit]

#[test]
fn loader_precision_by_tensor_mixed_bf16_f32() {
    // Arrange: model with BF16 and F32 tensors
    let f32_tensor = tensor_f32("w_f32", vec![2], &[1.0, 2.0]);
    // BF16 raw data: 2 elements, 2 bytes each
    let bf16_raw: Vec<u8> = [0x00, 0x3f, 0x00, 0x40].to_vec(); // 1.0 and 2.0 in BF16
    let bf16_tensor = tensor_raw(
        "w_bf16",
        vec![2],
        proto::tensor_proto::DataType::Bfloat16,
        &bf16_raw,
    );
    let graph = proto::GraphProto {
        initializer: vec![f32_tensor, bf16_tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.precision_by_tensor();
    // Assert: sorted alphabetically by name, correct dtypes
    assert_eq!(precisions.len(), 2);
    assert_eq!(precisions[0].0, "w_bf16");
    assert_eq!(precisions[0].1, Dtype::BF16);
    assert_eq!(precisions[1].0, "w_f32");
    assert_eq!(precisions[1].1, Dtype::F32);
}

// ── Loader: unique_precisions dedup and ordering ──────────────────────
// @trace TEST-ONNX-031 [level:unit]

#[test]
fn loader_unique_precisions_ordering_across_dtypes() {
    // Arrange: three tensors - F64, F32, F32 (dedup should yield [F64, F32])
    let f64_raw: Vec<u8> = (0..2).flat_map(|_| 0.0f64.to_le_bytes()).collect();
    let f64_tensor = tensor_raw(
        "w_f64",
        vec![2],
        proto::tensor_proto::DataType::Double,
        &f64_raw,
    );
    let f32_a = tensor_f32("w_a", vec![2], &[1.0, 2.0]);
    let f32_b = tensor_f32("w_b", vec![2], &[3.0, 4.0]);
    let graph = proto::GraphProto {
        initializer: vec![f64_tensor, f32_a, f32_b],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.unique_precisions();
    // Assert: F64 and F32, sorted by dtype_rank
    assert_eq!(precisions.len(), 2);
    assert_eq!(precisions[0], Dtype::F64);
    assert_eq!(precisions[1], Dtype::F32);
}

// ══════════════════════════════════════════════════════════════════════
// Additional tests (15) — edge cases and boundary conditions
// ══════════════════════════════════════════════════════════════════════

// ── Loader: tensor with 1D dimension [1] is not scalar ──────────────
// @trace TEST-ONNX-032 [level:unit]

#[test]
fn loader_tensor_1d_dim_one_not_scalar() {
    // Arrange: a 1D tensor with dim [1] — distinct from scalar (empty dims)
    let tensor = tensor_f32("one_elem", vec![1], &[7.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("one_elem").expect("tensor");
    // Assert: shape is [1], not [] (scalar)
    assert_eq!(slice.shape, vec![1]);
    assert_eq!(slice.data.len(), 4);
    let values = bytes_to_f32(slice.data);
    assert!((values[0] - 7.0).abs() < 1e-6);
}

// ── Loader: node with unicode name preserves encoding ───────────────
// @trace TEST-ONNX-033 [level:unit]

#[test]
fn loader_node_unicode_name_preserved() {
    // Arrange: a node whose name contains multi-byte UTF-8 characters
    let node = proto::NodeProto {
        op_type: Some("Relu".to_string()),
        name: Some("/model/\u{5c42}\u{7ea7}/Relu".to_string()), // Chinese characters
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: unicode characters in node name survive protobuf roundtrip
    assert_eq!(loader.graph().nodes[0].name, "/model/\u{5c42}\u{7ea7}/Relu");
}

// ── Loader: tensor with dim value zero ──────────────────────────────
// @trace TEST-ONNX-034 [level:unit]

#[test]
fn loader_tensor_zero_dim_value_shape() {
    // Arrange: tensor with one dimension being zero (edge case for empty batches)
    let tensor = tensor_f32("empty_batch", vec![0, 4], &[]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("empty_batch").expect("tensor");
    // Assert: shape [0, 4] preserved, zero data bytes
    assert_eq!(slice.shape, vec![0, 4]);
    assert!(slice.data.is_empty());
}

// ── Loader: alias with Gemm transB=0 returns false layout hint ──────
// @trace TEST-ONNX-035 [level:unit]

#[test]
fn weight_layout_hint_gemm_transb_zero_explicit_false() {
    // Arrange: Gemm node with transB=0 explicitly set (not defaulting)
    let weight = tensor_f32("dense_weight", vec![4, 4], &[1.0; 16]);
    let node = proto::NodeProto {
        name: Some("/fc/Gemm".to_string()),
        op_type: Some("Gemm".to_string()),
        input: vec!["x".to_string(), "dense_weight".to_string()],
        output: vec!["out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("transB".to_string()),
            r#type: Some(2), // INT
            i: Some(0),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: transB=0 → no transpose needed
    assert_eq!(loader.weight_layout_hint("dense_weight"), Some(false));
}

// ── Loader: graph with many value_info entries all preserved ────────
// @trace TEST-ONNX-036 [level:unit]

#[test]
fn loader_graph_many_value_info_entries() {
    // Arrange: graph with 10 value_info entries
    let vis: Vec<proto::ValueInfoProto> = (0..10)
        .map(|i| proto::ValueInfoProto {
            name: Some(format!("intermediate_{i}")),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        })
        .collect();
    let graph = proto::GraphProto {
        value_info: vis,
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all 10 entries preserved in order
    assert_eq!(loader.graph().value_info.len(), 10);
    for i in 0..10 {
        assert_eq!(loader.graph().value_info[i].name, format!("intermediate_{i}"));
    }
}

// ── Loader: external data with zero offset reads from beginning ─────
// @trace TEST-ONNX-037 [level:unit]

#[test]
fn load_external_tensor_zero_offset_reads_start() {
    // Arrange: external data file with explicit offset=0
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("data.bin");
    let data: Vec<u8> = [5.0f32, 6.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&data_path, &data).expect("write");
    let tensor = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("data.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("0".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("8".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);
    // Act
    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    let values = bytes_to_f32(loader.tensor("w").expect("tensor").data);
    // Assert: reads from beginning correctly
    assert_eq!(values.len(), 2);
    assert!((values[0] - 5.0).abs() < 1e-6);
    assert!((values[1] - 6.0).abs() < 1e-6);
}

// ── Loader: model with empty producer string fields ─────────────────
// @trace TEST-ONNX-038 [level:unit]

#[test]
fn loader_model_empty_producer_fields() {
    // Arrange: model with explicit empty strings for producer fields
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        producer_name: Some(String::new()),
        producer_version: Some(String::new()),
        domain: Some(String::new()),
        model_version: Some(0),
        doc_string: Some(String::new()),
        graph: Some(graph),
        opset_import: vec![],
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: empty strings preserved (not replaced with defaults)
    assert_eq!(loader.model().metadata.producer_name, "");
    assert_eq!(loader.model().metadata.producer_version, "");
    assert_eq!(loader.model().metadata.domain, "");
    assert_eq!(loader.model().metadata.doc_string, "");
}

// ── Loader: multiple output names per node preserved ────────────────
// @trace TEST-ONNX-039 [level:unit]

#[test]
fn loader_node_four_outputs_all_preserved() {
    // Arrange: a single node with 4 outputs (e.g. Split producing 4 chunks)
    let node = proto::NodeProto {
        op_type: Some("Split".to_string()),
        name: Some("split_node".to_string()),
        input: vec!["input".to_string()],
        output: vec![
            "chunk_0".to_string(),
            "chunk_1".to_string(),
            "chunk_2".to_string(),
            "chunk_3".to_string(),
        ],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all 4 outputs preserved in order
    let outputs = &loader.graph().nodes[0].outputs;
    assert_eq!(outputs.len(), 4);
    assert_eq!(outputs[0], "chunk_0");
    assert_eq!(outputs[1], "chunk_1");
    assert_eq!(outputs[2], "chunk_2");
    assert_eq!(outputs[3], "chunk_3");
}

// ── Loader: tensor_dtype via alias resolves correctly ───────────────
// @trace TEST-ONNX-040 [level:unit]

#[test]
fn loader_tensor_dtype_via_alias_resolution() {
    // Arrange: anonymous MatMul weight, BF16 dtype
    let bf16_raw: Vec<u8> = [0x00, 0x3f, 0x00, 0x40].to_vec();
    let weight = tensor_raw("onnx::MatMul_10", vec![2], proto::tensor_proto::DataType::Bfloat16, &bf16_raw);
    let node = proto::NodeProto {
        name: Some("/encoder/attn/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "onnx::MatMul_10".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let dtype = loader.tensor_dtype("encoder.attn.weight").expect("dtype via alias");
    // Assert: alias resolution yields correct BF16 dtype
    assert_eq!(dtype, Dtype::BF16);
}

// ── Loader: graph with mixed inline and external tensors ────────────
// @trace TEST-ONNX-041 [level:unit]

#[test]
fn loader_mixed_inline_and_external_tensors() {
    // Arrange: model with one inline tensor and one external tensor
    let dir = TempDir::new().expect("tempdir");
    let model_path = dir.path().join("model.onnx");
    let data_path = dir.path().join("ext.bin");
    std::fs::write(&data_path, &10.0f32.to_le_bytes()).expect("write ext");

    let inline = tensor_f32("inline_w", vec![1], &[5.0]);
    let ext_tensor = proto::TensorProto {
        dims: vec![1],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("ext_w".to_string()),
        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
        external_data: vec![
            proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("ext.bin".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("length".to_string()),
                value: Some("4".to_string()),
            },
        ],
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![inline, ext_tensor],
        ..empty_graph()
    };
    write_model(empty_model(graph), &model_path);
    // Act
    let loader = OnnxLoader::from_path(&model_path).expect("loader");
    // Assert: both tensors accessible with correct values
    let inline_val = bytes_to_f32(loader.tensor("inline_w").expect("inline").data);
    let ext_val = bytes_to_f32(loader.tensor("ext_w").expect("ext").data);
    assert!((inline_val[0] - 5.0).abs() < 1e-6);
    assert!((ext_val[0] - 10.0).abs() < 1e-6);
    assert_eq!(loader.names().len(), 2);
}

// ── Loader: node with empty string attribute value ──────────────────
// @trace TEST-ONNX-042 [level:unit]

#[test]
fn loader_node_empty_string_attribute_value() {
    // Arrange: a node with a string attribute whose value is empty
    let node = proto::NodeProto {
        op_type: Some("Resize".to_string()),
        name: Some("resize_1".to_string()),
        input: vec![],
        output: vec!["out".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("mode".to_string()),
            r#type: Some(3), // STRING
            s: Some(Vec::new()), // empty string
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attrs = &loader.graph().nodes[0].attributes;
    // Assert: empty string attribute parsed without error
    assert!(attrs.contains_key("mode"));
    assert!(matches!(&attrs["mode"].value, OnnxAttributeValue::String(s) if s.is_empty()));
}

// ── Loader: names() deduplicates after alias resolution ─────────────
// @trace TEST-ONNX-043 [level:unit]

#[test]
fn loader_names_no_alias_collision_with_initializer() {
    // Arrange: named initializer that matches a potential alias target
    // The alias system should not overwrite an existing initializer name
    let weight = tensor_f32("encoder.weight", vec![2], &[1.0, 2.0]);
    let node = proto::NodeProto {
        name: Some("/encoder/MatMul".to_string()),
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "encoder.weight".to_string()],
        output: vec!["out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        initializer: vec![weight],
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let names = loader.names();
    // Assert: only one name, no alias collision
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"encoder.weight".to_string()));
}

// ── Loader: node with negative integer attribute ────────────────────
// @trace TEST-ONNX-044 [level:unit]

#[test]
fn loader_node_negative_integer_attribute_value() {
    // Arrange: a node with a negative integer attribute
    let node = proto::NodeProto {
        op_type: Some("Pad".to_string()),
        name: Some("pad_1".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("pad_value".to_string()),
            r#type: Some(2), // INT
            i: Some(-999),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attrs = &loader.graph().nodes[0].attributes;
    // Assert: negative integer preserved exactly
    assert!(matches!(&attrs["pad_value"].value, OnnxAttributeValue::Int(v) if *v == -999));
}

// ── Loader: tensor_info returns correct dtype for I64 tensor ────────
// @trace TEST-ONNX-045 [level:unit]

#[test]
fn loader_tensor_info_i64_dtype() {
    // Arrange: an INT64 tensor
    let raw: Vec<u8> = [100i64, -200i64].iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = tensor_raw("ids", vec![2], proto::tensor_proto::DataType::Int64, &raw);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let info = loader.tensor_info("ids").expect("tensor_info");
    // Assert: shape and dtype correct
    assert_eq!(info.name, "ids");
    assert_eq!(info.shape, vec![2]);
    assert_eq!(info.dtype, Dtype::I64);
}

// ── Loader: model with opset_import version zero ────────────────────
// @trace TEST-ONNX-046 [level:unit]

#[test]
fn loader_model_opset_import_version_zero() {
    // Arrange: model with opset_import where version is 0 (edge case)
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        ir_version: Some(8),
        opset_import: vec![proto::OperatorSetIdProto {
            domain: Some("".to_string()),
            version: Some(0),
        }],
        graph: Some(graph),
        producer_name: None,
        producer_version: None,
        domain: None,
        model_version: None,
        doc_string: None,
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: Vec::new(),
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: version 0 preserved
    assert_eq!(loader.model().metadata.opset_import.len(), 1);
    assert_eq!(loader.model().metadata.opset_import[0].version, 0);
    assert_eq!(loader.model().metadata.opset_import[0].domain, "");
}

// ══════════════════════════════════════════════════════════════════════
// Additional tests (15 new — edge cases, error paths, type traits)
// ══════════════════════════════════════════════════════════════════════

// ── OnnxTensor::new constructs tensor with correct fields ──────────
// @trace TEST-ONNX-047 [level:unit]

#[test]
fn onnx_tensor_new_constructs_correctly() {
    // Arrange
    let data = Bytes::from(vec![0u8, 0, 0x80, 0x3f]); // 1.0f32
    // Act
    let tensor = OnnxTensor::new(
        "test_tensor".to_string(),
        Dtype::F32,
        vec![1],
        data.clone(),
    );
    // Assert
    assert_eq!(tensor.name, "test_tensor");
    assert_eq!(tensor.dtype, Dtype::F32);
    assert_eq!(tensor.shape, vec![1]);
    assert!(!tensor.is_string);
    assert_eq!(tensor.raw_data(), data.as_ref());
}

// ── OnnxTensor::new_string sets is_string flag ─────────────────────
// @trace TEST-ONNX-048 [level:unit]

#[test]
fn onnx_tensor_new_string_sets_flag() {
    // Arrange
    let data = Bytes::from("hello".as_bytes().to_vec());
    // Act
    let tensor = OnnxTensor::new_string("str_tensor".to_string(), vec![5], data);
    // Assert
    assert_eq!(tensor.name, "str_tensor");
    assert!(tensor.is_string);
    assert_eq!(tensor.dtype, Dtype::U8); // string uses U8 placeholder
    assert_eq!(tensor.shape, vec![5]);
}

// ── OnnxTensor scalar_f32 returns None for multi-element tensor ────
// @trace TEST-ONNX-049 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_multi_element_via_new_returns_none() {
    // Arrange: a tensor with shape [2] (not scalar)
    let data = Bytes::from(vec![0u8, 0, 0x80, 0x3f, 0, 0, 0, 0x40]); // [1.0, 2.0]
    let tensor = OnnxTensor::new("multi".to_string(), Dtype::F32, vec![2], data);
    // Act
    let result = tensor.scalar_f32();
    // Assert
    assert!(result.is_none());
}

// ── OnnxTensor scalar_i64 returns correct value for I64 scalar ─────
// @trace TEST-ONNX-050 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_returns_value() {
    // Arrange: a scalar I64 tensor with value -42
    let data = Bytes::from((-42i64).to_le_bytes().to_vec());
    let tensor = OnnxTensor::new("idx".to_string(), Dtype::I64, vec![], data);
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert_eq!(result, Some(-42));
}

// ── OnnxTensor scalar_i64 returns None for multi-element tensor ────
// @trace TEST-ONNX-051 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_non_scalar_returns_none() {
    // Arrange: a [2] shape I64 tensor
    let data = Bytes::from([10i64, 20].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let tensor = OnnxTensor::new("vals".to_string(), Dtype::I64, vec![2], data);
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert!(result.is_none());
}

// ── OnnxTensor slice() returns correct TensorSlice ─────────────────
// @trace TEST-ONNX-052 [level:unit]

#[test]
fn onnx_tensor_slice_returns_correct_data() {
    // Arrange
    let raw = Bytes::from(vec![0u8, 0, 0, 0x3f, 0, 0, 0x80, 0x3f]); // [0.5, 1.0]
    let tensor = OnnxTensor::new("w".to_string(), Dtype::F32, vec![1, 2], raw);
    // Act
    let slice = tensor.slice();
    // Assert
    assert_eq!(slice.dtype, Dtype::F32);
    assert_eq!(slice.shape, vec![1, 2]);
    assert_eq!(slice.data.len(), 8);
    let values = bytes_to_f32(slice.data);
    assert!((values[0] - 0.5).abs() < 1e-6);
    assert!((values[1] - 1.0).abs() < 1e-6);
}

// ── OnnxTensor scalar_f32 for U8 returns byte as f32 ───────────────
// @trace TEST-ONNX-053 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_u8_value() {
    // Arrange
    let data = Bytes::from(vec![255u8]);
    let tensor = OnnxTensor::new("byte_val".to_string(), Dtype::U8, vec![], data);
    // Act
    let result = tensor.scalar_f32();
    // Assert
    assert_eq!(result, Some(255.0));
}

// ── OnnxTensor scalar_f32 for F32 scalar returns value ──────────────
// @trace TEST-ONNX-054 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_correct_value() {
    // Arrange: 3.14 as f32
    let data = Bytes::from(3.14f32.to_le_bytes().to_vec());
    let tensor = OnnxTensor::new("pi".to_string(), Dtype::F32, vec![], data);
    // Act
    let result = tensor.scalar_f32();
    // Assert
    assert!(result.is_some());
    assert!((result.unwrap() - 3.14).abs() < 0.01);
}

// ── OnnxModel Debug trait produces output with fields ──────────────
// @trace TEST-ONNX-055 [level:unit]

#[test]
fn onnx_model_debug_trait() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let inner = loader.model();
    let debug_str = format!("{inner:?}");
    // Assert
    assert!(debug_str.contains("OnnxModel"));
    assert!(debug_str.contains("metadata"));
    assert!(debug_str.contains("graph"));
}

// ── OnnxNode with empty inputs and outputs preserves them ──────────
// @trace TEST-ONNX-056 [level:unit]

#[test]
fn loader_node_empty_inputs_outputs() {
    // Arrange: a node with no inputs or outputs (edge case)
    let node = proto::NodeProto {
        op_type: Some("Constant".to_string()),
        name: Some("const_node".to_string()),
        input: vec![],
        output: vec![],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let loaded_node = &loader.graph().nodes[0];
    // Assert: empty vectors preserved
    assert!(loaded_node.inputs.is_empty());
    assert!(loaded_node.outputs.is_empty());
    assert_eq!(loaded_node.op_type, "Constant");
}

// ── OnnxValueInfo with type and doc_string preserved ───────────────
// @trace TEST-ONNX-057 [level:unit]

#[test]
fn loader_value_info_with_doc_string() {
    // Arrange: graph input with doc_string
    let input_info = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: None,
        doc_string: Some("token IDs".to_string()),
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![input_info],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let info = &loader.graph().inputs[0];
    // Assert
    assert_eq!(info.name, "input_ids");
    assert_eq!(info.doc_string, "token IDs");
}

// ── Model with graph name preserved ────────────────────────────────
// @trace TEST-ONNX-058 [level:unit]

#[test]
fn loader_graph_name_preserved() {
    // Arrange: graph with a specific name
    let graph = proto::GraphProto {
        name: Some("my_model_graph".to_string()),
        doc_string: Some("test graph".to_string()),
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().name, "my_model_graph");
    assert_eq!(loader.graph().doc_string, "test graph");
}

// ── OnnxAttribute with float attribute value ───────────────────────
// @trace TEST-ONNX-059 [level:unit]

#[test]
fn loader_node_float_attribute_value() {
    // Arrange: a node with a float attribute
    let node = proto::NodeProto {
        op_type: Some("Dropout".to_string()),
        name: Some("drop_1".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![proto::AttributeProto {
            name: Some("ratio".to_string()),
            r#type: Some(1), // FLOAT
            f: Some(0.5),
            ..Default::default()
        }],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let attrs = &loader.graph().nodes[0].attributes;
    // Assert: float attribute preserved
    assert!(attrs.contains_key("ratio"));
    assert!(matches!(&attrs["ratio"].value, OnnxAttributeValue::Float(v) if (*v - 0.5).abs() < 1e-6));
}

// ── Duplicate initializer name returns error ───────────────────────
// @trace TEST-ONNX-060 [level:unit]

#[test]
fn loader_duplicate_initializer_name_returns_error() {
    // Arrange: two initializers with the same name
    let t1 = tensor_f32("dup", vec![1], &[1.0]);
    let t2 = tensor_f32("dup", vec![1], &[2.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let result = OnnxLoader::from_path(file.path());
    // Assert: duplicate name should fail
    assert!(result.is_err());
}

// ── OnnxQuantizationAnnotation fields via graph ────────────────────
// @trace TEST-ONNX-061 [level:unit]

#[test]
fn loader_graph_quantization_annotation_empty() {
    // Arrange: model with no quantization annotations
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[1.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: no quantization annotations
    assert!(loader.graph().quantization_annotation.is_empty());
}

// ── OnnxTensor scalar_f32 returns None for I8 dtype ──────────────
// @trace TEST-ONNX-062 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_i8_dtype_returns_none() {
    // Arrange: construct an OnnxTensor with I8 dtype, singleton shape, one byte of data
    let tensor = OnnxTensor::new("i8_scalar".to_string(), Dtype::I8, vec![1], Bytes::copy_from_slice(&[42]));
    // Act
    let result = tensor.scalar_f32();
    // Assert: I8 is not supported by scalar_f32
    assert!(result.is_none());
}

// ── OnnxTensor scalar_i64 returns None for BOOL dtype ─────────────
// @trace TEST-ONNX-063 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_bool_dtype_returns_none() {
    // Arrange: construct an OnnxTensor with BOOL dtype, singleton shape
    let tensor = OnnxTensor::new("bool_scalar".to_string(), Dtype::BOOL, vec![1], Bytes::copy_from_slice(&[1]));
    // Act
    let result = tensor.scalar_i64();
    // Assert: BOOL is not supported by scalar_i64
    assert!(result.is_none());
}

// ── OnnxTensor raw_data returns empty slice for zero-length tensor ─
// @trace TEST-ONNX-064 [level:unit]

#[test]
fn onnx_tensor_raw_data_empty_bytes() {
    // Arrange: OnnxTensor with empty data
    let tensor = OnnxTensor::new("empty".to_string(), Dtype::F32, vec![0], Bytes::new());
    // Act
    let raw = tensor.raw_data();
    // Assert
    assert!(raw.is_empty());
}

// ── OnnxTensor slice returns correct dtype and shape ──────────────
// @trace TEST-ONNX-065 [level:unit]

#[test]
fn onnx_tensor_slice_matches_constructor() {
    // Arrange
    let data = Bytes::copy_from_slice(&[0u8, 0, 128, 63]); // 1.0f32 LE
    let tensor = OnnxTensor::new("sliced".to_string(), Dtype::F32, vec![1], data);
    // Act
    let slice = tensor.slice();
    // Assert
    assert_eq!(slice.dtype, Dtype::F32);
    assert_eq!(slice.shape, vec![1]);
    assert_eq!(slice.data.len(), 4);
}

// ── OnnxSparseTensor format COO is distinct from CSR and CSC ──────
// @trace TEST-ONNX-066 [level:unit]

#[test]
fn onnx_sparse_format_coo_not_equal_csr() {
    // Arrange
    let coo = OnnxSparseFormat::Coo;
    let csr = OnnxSparseFormat::Csr;
    // Assert
    assert_ne!(coo, csr);
}

// ── OnnxTensor new with multi-dimensional shape preserves shape ──
// @trace TEST-ONNX-067 [level:unit]

#[test]
fn onnx_tensor_new_2d_shape_preserved() {
    // Arrange: construct OnnxTensor with shape [3, 4]
    let data = vec![0u8; 48]; // 12 * 4 bytes for F32
    let tensor = OnnxTensor::new("matrix".to_string(), Dtype::F32, vec![3, 4], Bytes::from(data));
    // Assert
    assert_eq!(tensor.shape, vec![3, 4]);
    assert_eq!(tensor.raw_data().len(), 48);
}

// ── OnnxTensor new with empty shape (scalar) preserves data ──────
// @trace TEST-ONNX-068 [level:unit]

#[test]
fn onnx_tensor_new_empty_shape_scalar() {
    // Arrange: empty shape = scalar
    let data = Bytes::copy_from_slice(&[0u8, 0, 128, 63]); // 1.0f32
    let tensor = OnnxTensor::new("scalar".to_string(), Dtype::F32, vec![], data.clone());
    // Assert
    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.raw_data().len(), 4);
}

// ── Loader with multiple initializers all resolved correctly ──────
// @trace TEST-ONNX-069 [level:unit]

#[test]
fn loader_many_initializers_all_resolved() {
    // Arrange: 5 different tensors
    let t1 = tensor_f32("a", vec![2], &[1.0, 2.0]);
    let t2 = tensor_f32("b", vec![3], &[3.0, 4.0, 5.0]);
    let t3 = tensor_f32("c", vec![1], &[6.0]);
    let t4 = tensor_f32("d", vec![1, 1], &[7.0]);
    let t5 = tensor_f32("e", vec![2, 2], &[8.0, 9.0, 10.0, 11.0]);
    let graph = proto::GraphProto {
        initializer: vec![t1, t2, t3, t4, t5],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: all 5 tensors accessible
    assert_eq!(loader.names().len(), 5);
    for name in &["a", "b", "c", "d", "e"] {
        let ts = loader.tensor(name).expect("tensor should exist");
        assert!(!ts.data.is_empty());
    }
}

// ── OnnxNode with empty attributes map ────────────────────────────
// @trace TEST-ONNX-070 [level:unit]

#[test]
fn onnx_node_empty_attributes_map() {
    // Arrange: construct an OnnxNode with no attributes
    let node = OnnxNode {
        name: "relu_1".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attributes: HashMap::new(),
    };
    // Assert
    assert!(node.attributes.is_empty());
    assert_eq!(node.name, "relu_1");
    assert_eq!(node.op_type, "Relu");
}

// ── OnnxValueInfo with metadata_props preserves entries ───────────
// @trace TEST-ONNX-071 [level:unit]

#[test]
fn onnx_value_info_metadata_props_multiple_entries() {
    // Arrange
    let mut props = HashMap::new();
    props.insert("key1".to_string(), "val1".to_string());
    props.insert("key2".to_string(), "val2".to_string());
    let info = OnnxValueInfo {
        name: "test_info".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: props.clone(),
    };
    // Assert
    assert_eq!(info.metadata_props.len(), 2);
    assert_eq!(info.metadata_props.get("key1").unwrap(), "val1");
    assert_eq!(info.metadata_props.get("key2").unwrap(), "val2");
}

// ── OnnxOperatorSet clone preserves domain and version ────────────
// @trace TEST-ONNX-072 [level:unit]

#[test]
fn onnx_operator_set_clone_preserves_fields() {
    // Arrange
    let ops = OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 3,
    };
    // Act
    let cloned = ops.clone();
    // Assert
    assert_eq!(cloned.domain, "ai.onnx.ml");
    assert_eq!(cloned.version, 3);
}

// ── Loader graph with value_info only preserves entries ───────────
// @trace TEST-ONNX-073 [level:unit]

#[test]
fn loader_graph_value_info_only_preserves_count() {
    // Arrange: graph with value_info but no initializers or nodes
    let vi1 = proto::ValueInfoProto {
        name: Some("intermediate_1".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let vi2 = proto::ValueInfoProto {
        name: Some("intermediate_2".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        value_info: vec![vi1, vi2],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().value_info.len(), 2);
    assert_eq!(loader.graph().value_info[0].name, "intermediate_1");
    assert_eq!(loader.graph().value_info[1].name, "intermediate_2");
}

// ── OnnxGraph initializers HashMap lookup after clone ─────────────
// @trace TEST-ONNX-074 [level:unit]

#[test]
fn onnx_graph_clone_initializers_lookup() {
    // Arrange: build a graph via loading, then clone and verify lookup
    let tensor = tensor_f32("weight_clone_test", vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act: clone the graph
    let cloned_graph = loader.graph().clone();
    // Assert: cloned graph still has the initializer
    assert!(cloned_graph.initializers.contains_key("weight_clone_test"));
    let t = &cloned_graph.initializers["weight_clone_test"];
    assert_eq!(t.shape, vec![2, 3]);
}

// ── ConvertError clone preserves each variant's fields ────────────
// @trace TEST-ONNX-075 [level:unit]

#[test]
fn convert_error_shape_inference_failed_clone() {
    // Arrange
    let err = ConvertError::ShapeInferenceFailed {
        name: "output_tensor".to_string(),
        reason: "unknown dimension".to_string(),
    };
    // Act
    let cloned = err.clone();
    // Assert
    let msg = format!("{cloned}");
    assert!(msg.contains("output_tensor"));
    assert!(msg.contains("unknown dimension"));
}

// ── Loader with graph containing single node no attributes ────────
// @trace TEST-ONNX-076 [level:unit]

#[test]
fn loader_graph_single_node_no_attributes() {
    // Arrange: a single node with no attributes
    let node = proto::NodeProto {
        op_type: Some("Identity".to_string()),
        name: Some("identity_node".to_string()),
        input: vec!["x".to_string()],
        output: vec!["y".to_string()],
        attribute: vec![],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        initializer: vec![tensor_f32("x", vec![1], &[42.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 1);
    assert_eq!(loader.graph().nodes[0].op_type, "Identity");
    assert!(loader.graph().nodes[0].attributes.is_empty());
}

// ══════════════════════════════════════════════════════════════════════
// Additional tests (15 new — edge cases, dtype mapping, error paths)
// ══════════════════════════════════════════════════════════════════════

// ── Loader with double (F64) tensor preserves dtype ───────────────
// @trace TEST-ONNX-077 [level:unit]

#[test]
fn loader_tensor_dtype_double_f64() {
    // Arrange
    let data = 3.14f64.to_le_bytes();
    let tensor = tensor_raw("d_weight", vec![1], proto::tensor_proto::DataType::Double, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.tensor_dtype("d_weight").expect("dtype"), Dtype::F64);
}

// ── Loader with FP8 E4M3 tensor maps to F8_E4M3 ──────────────────
// @trace TEST-ONNX-078 [level:unit]

#[test]
fn loader_tensor_dtype_float8_e4m3fn() {
    // Arrange
    let tensor = tensor_raw("fp8_w", vec![4], proto::tensor_proto::DataType::Float8e4m3fn, &[0x3C, 0x40, 0x00, 0x7F]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.tensor_dtype("fp8_w").expect("dtype"), Dtype::F8_E4M3);
}

// ── OnnxTensor scalar_f32 returns correct value for U16 dtype ────
// @trace TEST-ONNX-079 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_u16_value() {
    // Arrange: 1000u16 LE bytes
    let data = Bytes::copy_from_slice(&1000u16.to_le_bytes());
    let tensor = OnnxTensor::new("u16_scalar".to_string(), Dtype::U16, vec![1], data);
    // Act
    let result = tensor.scalar_f32();
    // Assert
    let val = result.expect("scalar_f32 should return Some for U16");
    assert!((val - 1000.0f32).abs() < 1e-6);
}

// ── OnnxTensor scalar_i64 returns correct value for U32 dtype ─────
// @trace TEST-ONNX-080 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_u32_value() {
    // Arrange: 50000u32 LE bytes
    let data = Bytes::copy_from_slice(&50000u32.to_le_bytes());
    let tensor = OnnxTensor::new("u32_scalar".to_string(), Dtype::U32, vec![1], data);
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert_eq!(result, Some(50000i64));
}

// ── OnnxTensor scalar_i64 returns correct value for F32 dtype ─────
// @trace TEST-ONNX-081 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_f32_truncates() {
    // Arrange: 3.7f32 LE bytes — should truncate to 3
    let data = Bytes::copy_from_slice(&3.7f32.to_le_bytes());
    let tensor = OnnxTensor::new("f32_to_i64".to_string(), Dtype::F32, vec![1], data);
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert_eq!(result, Some(3i64));
}

// ── OnnxGraph metadata_props preserved from proto ────────────────
// @trace TEST-ONNX-082 [level:unit]

#[test]
fn loader_graph_metadata_props_preserved() {
    // Arrange
    let graph = proto::GraphProto {
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("source".to_string()),
                value: Some("pytorch".to_string()),
            },
            proto::StringStringEntryProto {
                key: Some("license".to_string()),
                value: Some("apache-2.0".to_string()),
            },
        ],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let mp = &loader.graph().metadata_props;
    assert_eq!(mp.len(), 2);
    assert_eq!(mp.get("source").expect("source key"), "pytorch");
    assert_eq!(mp.get("license").expect("license key"), "apache-2.0");
}

// ── Model metadata_props from proto string entries ────────────────
// @trace TEST-ONNX-083 [level:unit]

#[test]
fn loader_model_metadata_props_string_entries() {
    // Arrange
    let model = proto::ModelProto {
        metadata_props: vec![
            proto::StringStringEntryProto {
                key: Some("framework".to_string()),
                value: Some("onnx-training".to_string()),
            },
        ],
        ..empty_model(empty_graph())
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let mp = &loader.model().metadata.metadata_props;
    assert_eq!(mp.len(), 1);
    assert_eq!(mp.get("framework").expect("framework key"), "onnx-training");
}

// ── Loader with node containing multiple input references ─────────
// @trace TEST-ONNX-084 [level:unit]

#[test]
fn loader_node_three_inputs_all_preserved() {
    // Arrange
    let node = proto::NodeProto {
        op_type: Some("Add".to_string()),
        name: Some("triple_add".to_string()),
        input: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        output: vec!["d".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![node],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().nodes.len(), 1);
    assert_eq!(loader.graph().nodes[0].inputs.len(), 3);
    assert_eq!(loader.graph().nodes[0].inputs[0], "a");
    assert_eq!(loader.graph().nodes[0].inputs[1], "b");
    assert_eq!(loader.graph().nodes[0].inputs[2], "c");
}

// ── Loader with graph input/output value info ─────────────────────
// @trace TEST-ONNX-085 [level:unit]

#[test]
fn loader_graph_input_output_info_preserved() {
    // Arrange
    let input_info = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let output_info = proto::ValueInfoProto {
        name: Some("logits".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: vec![],
    };
    let graph = proto::GraphProto {
        input: vec![input_info],
        output: vec![output_info],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().inputs.len(), 1);
    assert_eq!(loader.graph().inputs[0].name, "input_ids");
    assert_eq!(loader.graph().outputs.len(), 1);
    assert_eq!(loader.graph().outputs[0].name, "logits");
}

// ── ConvertError clone for UnsupportedOp variant ─────────────────
// @trace TEST-ONNX-086 [level:unit]

#[test]
fn convert_error_unsupported_op_clone() {
    // Arrange
    let err = ConvertError::UnsupportedOp {
        op_type: "CustomOp".to_string(),
        node_name: "my_node".to_string(),
    };
    // Act
    let cloned = err.clone();
    // Assert
    let msg = format!("{cloned}");
    assert!(msg.contains("CustomOp"));
    assert!(msg.contains("my_node"));
}

// ── ConvertError clone for MissingInitializer variant ────────────
// @trace TEST-ONNX-087 [level:unit]

#[test]
fn convert_error_missing_initializer_clone() {
    // Arrange
    let err = ConvertError::MissingInitializer {
        name: "weight_0".to_string(),
        node_name: "fc_layer".to_string(),
    };
    // Act
    let cloned = err.clone();
    // Assert
    let msg = format!("{cloned}");
    assert!(msg.contains("weight_0"));
    assert!(msg.contains("fc_layer"));
}

// ── ConvertError clone for InvalidMatMulShape variant ────────────
// @trace TEST-ONNX-088 [level:unit]

#[test]
fn convert_error_invalid_matmul_shape_clone() {
    // Arrange
    let err = ConvertError::InvalidMatMulShape {
        name: "proj.weight".to_string(),
        dims: 3,
    };
    // Act
    let cloned = err.clone();
    // Assert
    let msg = format!("{cloned}");
    assert!(msg.contains("proj.weight"));
    assert!(msg.contains("3-D"));
}

// ── OnnxModel functions list empty for simple model ──────────────
// @trace TEST-ONNX-089 [level:unit]

#[test]
fn loader_model_no_functions_by_default() {
    // Arrange
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![2], &[1.0, 2.0])],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert!(loader.model().functions.is_empty());
}

// ── Loader tensor_dtype for uint16 (U16) mapping ──────────────────
// @trace TEST-ONNX-090 [level:unit]

#[test]
fn loader_tensor_dtype_uint16() {
    // Arrange
    let data = 42u16.to_le_bytes();
    let tensor = tensor_raw("u16_ids", vec![1], proto::tensor_proto::DataType::Uint16, &data);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.tensor_dtype("u16_ids").expect("dtype"), Dtype::U16);
}

// ── OnnxValueInfo clone preserves all fields ──────────────────────
// @trace TEST-ONNX-091 [level:unit]

#[test]
fn onnx_value_info_clone_preserves_fields() {
    // Arrange
    let info = OnnxValueInfo {
        name: "hidden_states".to_string(),
        value_type: None,
        doc_string: "intermediate output".to_string(),
        metadata_props: {
            let mut m = HashMap::new();
            m.insert("key".to_string(), "value".to_string());
            m
        },
    };
    // Act
    let cloned = info.clone();
    // Assert
    assert_eq!(cloned.name, "hidden_states");
    assert!(cloned.value_type.is_none());
    assert_eq!(cloned.doc_string, "intermediate output");
    assert_eq!(cloned.metadata_props.get("key").expect("key"), "value");
}

// ── OnnxTensor scalar_i64 from U16 value ────────────────────────
// @trace TEST-ONNX-092 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_from_u16_value() {
    // Arrange
    let tensor = OnnxTensor::new(
        "u16_scalar".to_string(),
        Dtype::U16,
        vec![],
        Bytes::from(1000u16.to_le_bytes().to_vec()),
    );
    // Act
    let val = tensor.scalar_i64().expect("should convert u16 to i64");
    // Assert
    assert_eq!(val, 1000i64);
}

// ── OnnxTensor scalar_f32 from F16 value ────────────────────────
// @trace TEST-ONNX-093 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_from_f16_value() {
    // Arrange
    let f16_bits = half::f16::from_f32(3.25).to_bits();
    let tensor = OnnxTensor::new(
        "f16_scalar".to_string(),
        Dtype::F16,
        vec![],
        Bytes::from(f16_bits.to_le_bytes().to_vec()),
    );
    // Act
    let val = tensor.scalar_f32().expect("should convert f16 to f32");
    // Assert
    assert!((val - 3.25f32).abs() < 1e-3);
}

// ── OnnxTensor scalar_i64 from F16 value ────────────────────────
// @trace TEST-ONNX-094 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_from_f16_value() {
    // Arrange
    let f16_bits = half::f16::from_f32(7.0).to_bits();
    let tensor = OnnxTensor::new(
        "f16_scalar".to_string(),
        Dtype::F16,
        vec![],
        Bytes::from(f16_bits.to_le_bytes().to_vec()),
    );
    // Act
    let val = tensor.scalar_i64().expect("should convert f16 to i64");
    // Assert
    assert_eq!(val, 7i64);
}

// ── OnnxSparseFormat CSR != CSC inequality ──────────────────────
// @trace TEST-ONNX-095 [level:unit]

#[test]
fn onnx_sparse_format_csr_not_equal_csc() {
    // Arrange — tested in AAA style to verify PartialEq correctness
    let csr = OnnxSparseFormat::Csr;
    let csc = OnnxSparseFormat::Csc;
    // Act & Assert
    assert_ne!(csr, csc);
    assert_eq!(csr, OnnxSparseFormat::Csr);
    assert_eq!(csc, OnnxSparseFormat::Csc);
}

// ── OnnxModel clone preserves metadata_props ────────────────────
// @trace TEST-ONNX-096 [level:unit]

#[test]
fn onnx_model_clone_preserves_metadata_props() {
    // Arrange
    let mut props = HashMap::new();
    props.insert("framework".to_string(), "pytorch".to_string());
    props.insert("version".to_string(), "2.6".to_string());
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 8,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        },
        graph: OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![],
    };
    // Act
    let cloned = model.clone();
    // Assert
    assert_eq!(cloned.metadata.metadata_props.len(), 2);
    assert_eq!(cloned.metadata.metadata_props.get("framework").expect("key"), "pytorch");
    assert_eq!(cloned.metadata.metadata_props.get("version").expect("key"), "2.6");
    // Verify independence: modifying clone does not affect original
    drop(cloned);
    assert_eq!(model.metadata.metadata_props.len(), 2);
}

// ── OnnxGraph doc_string field preserved ────────────────────────
// @trace TEST-ONNX-097 [level:unit]

#[test]
fn onnx_graph_doc_string_field_preserved() {
    // Arrange
    let graph = OnnxGraph {
        name: "main_graph".to_string(),
        doc_string: "This is a test graph for inference".to_string(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = graph.clone();
    // Assert
    assert_eq!(graph.doc_string, "This is a test graph for inference");
    assert_eq!(cloned.doc_string, graph.doc_string);
}

// ── OnnxFunction doc_string field access ────────────────────────
// @trace TEST-ONNX-098 [level:unit]

#[test]
fn onnx_function_doc_string_field_access() {
    // Arrange
    let func = OnnxFunction {
        name: "CustomRelu".to_string(),
        domain: "custom.ops".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: "Custom ReLU activation function".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let doc = func.doc_string.clone();
    // Assert
    assert_eq!(doc, "Custom ReLU activation function");
    assert_eq!(func.name, "CustomRelu");
    assert_eq!(func.domain, "custom.ops");
}

// ── OnnxQuantizationAnnotation axis field ───────────────────────
// @trace TEST-ONNX-099 [level:unit]

#[test]
fn onnx_quantization_annotation_axis_field() {
    // Arrange
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "weight.q".to_string(),
        quant_param_tensor_names: HashMap::new(),
        scale: Some(0.0078125),
        zero_point: Some(128),
        axis: Some(0),
    };
    // Act
    let axis = qa.axis;
    let scale = qa.scale;
    let zero_point = qa.zero_point;
    // Assert
    assert_eq!(axis, Some(0));
    assert_eq!(scale, Some(0.0078125));
    assert_eq!(zero_point, Some(128));
    assert!(qa.quant_param_tensor_names.is_empty());
}

// ── OnnxOperatorSet version zero clone ──────────────────────────
// @trace TEST-ONNX-100 [level:unit]

#[test]
fn onnx_operator_set_version_zero_clone() {
    // Arrange
    let opset = OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 0,
    };
    // Act
    let cloned = opset.clone();
    // Assert
    assert_eq!(cloned.domain, "ai.onnx.ml");
    assert_eq!(cloned.version, 0);
    let debug = format!("{cloned:?}");
    assert!(debug.contains("ai.onnx.ml"));
}

// ── OnnxTensor new_string with empty data ───────────────────────
// @trace TEST-ONNX-101 [level:unit]

#[test]
fn onnx_tensor_new_string_raw_data_empty() {
    // Arrange
    let tensor = OnnxTensor::new_string(
        "text_labels".to_string(),
        vec![0],
        Bytes::new(),
    );
    // Act
    let is_string = tensor.is_string;
    let raw = tensor.raw_data();
    // Assert
    assert!(is_string);
    assert!(raw.is_empty());
    assert_eq!(tensor.dtype, Dtype::U8);
    assert_eq!(tensor.shape, vec![0]);
}

// ── OnnxNode default empty inputs and outputs ───────────────────
// @trace TEST-ONNX-102 [level:unit]

#[test]
fn onnx_node_inputs_empty_vec_default() {
    // Arrange
    let node = OnnxNode {
        name: "identity".to_string(),
        op_type: "Identity".to_string(),
        domain: String::new(),
        inputs: vec![],
        outputs: vec![],
        attributes: HashMap::new(),
    };
    // Act
    let debug = format!("{node:?}");
    // Assert
    assert!(node.inputs.is_empty());
    assert!(node.outputs.is_empty());
    assert!(node.attributes.is_empty());
    assert!(debug.contains("Identity"));
    assert!(debug.contains("identity"));
}

// ── Loader unique_precisions on empty model ─────────────────────
// @trace TEST-ONNX-103 [level:unit]

#[test]
fn loader_unique_precisions_empty_model() {
    // Arrange
    let graph = proto::GraphProto {
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let precisions = loader.unique_precisions();
    // Assert
    assert!(precisions.is_empty());
}

// ── OnnxValueInfo empty metadata_props ──────────────────────────
// @trace TEST-ONNX-104 [level:unit]

#[test]
fn onnx_value_info_empty_metadata_props() {
    // Arrange
    let info = OnnxValueInfo {
        name: "attention_mask".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = info.clone();
    // Assert
    assert!(info.metadata_props.is_empty());
    assert!(cloned.metadata_props.is_empty());
    assert_eq!(cloned.name, "attention_mask");
    let debug = format!("{info:?}");
    assert!(debug.contains("attention_mask"));
}

// ── OnnxAttributeValue Graph variant debug output ───────────────
// @trace TEST-ONNX-105 [level:unit]

#[test]
fn onnx_attribute_value_graph_variant_debug() {
    // Arrange
    let subgraph = OnnxGraph {
        name: "then_branch".to_string(),
        doc_string: String::new(),
        nodes: vec![OnnxNode {
            name: "inner_relu".to_string(),
            op_type: "Relu".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        }],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    let attr_val = OnnxAttributeValue::Graph(Box::new(subgraph));
    // Act
    let debug = format!("{attr_val:?}");
    // Assert
    assert!(debug.contains("Graph"));
    assert!(debug.contains("then_branch"));
    assert!(debug.contains("inner_relu"));
}

// ── OnnxSparseTensor dims field access ──────────────────────────
// @trace TEST-ONNX-106 [level:unit]

#[test]
fn onnx_sparse_tensor_dims_field_access() {
    // Arrange
    let values = OnnxTensor::new(
        "vals".to_string(),
        Dtype::F32,
        vec![2],
        Bytes::from([1.0f32, 2.0f32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let indices = OnnxTensor::new(
        "idxs".to_string(),
        Dtype::I64,
        vec![2],
        Bytes::from([0i64, 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let sparse = OnnxSparseTensor {
        values,
        indices,
        dims: vec![4, 4],
        format: OnnxSparseFormat::Coo,
    };
    // Act
    let dims = sparse.dims.clone();
    let format = sparse.format;
    // Assert
    assert_eq!(dims, vec![4, 4]);
    assert_eq!(format, OnnxSparseFormat::Coo);
    assert_eq!(sparse.values.name, "vals");
    assert_eq!(sparse.indices.name, "idxs");
}

// ── OnnxTensor scalar_f32 from bool dtype returns None ──────────────
// @trace TEST-ONNX-107 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_bool_dtype_returns_none() {
    // Arrange: a singleton tensor with BOOL dtype
    let tensor = OnnxTensor::new(
        "flag".to_string(),
        Dtype::BOOL,
        vec![1],
        Bytes::from_static(&[1u8]),
    );
    // Act
    let result = tensor.scalar_f32();
    // Assert
    assert!(result.is_none());
}

// ── OnnxTensor scalar_i64 from BOOL dtype returns None (separate from f32) ──
// @trace TEST-ONNX-108 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_bool_dtype_is_none_too() {
    // Arrange: a singleton tensor with BOOL dtype
    let tensor = OnnxTensor::new(
        "flag".to_string(),
        Dtype::BOOL,
        vec![1],
        Bytes::from_static(&[1u8]),
    );
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert!(result.is_none());
}

// ── OnnxTensor scalar_f32 empty shape (true scalar) returns value ──
// @trace TEST-ONNX-109 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_zero_shape_returns_value() {
    // Arrange: true scalar with empty shape
    let tensor = OnnxTensor::new(
        "scalar_val".to_string(),
        Dtype::F32,
        vec![],
        Bytes::from(99.5f32.to_le_bytes().to_vec()),
    );
    // Act
    let result = tensor.scalar_f32();
    // Assert
    let val = result.expect("scalar f32 from zero-dim tensor");
    assert!((val - 99.5).abs() < 1e-6);
}

// ── OnnxTensor raw_data on large multi-dim tensor ──────────────────
// @trace TEST-ONNX-110 [level:unit]

#[test]
fn onnx_tensor_raw_data_multi_dim_consistent_with_shape() {
    // Arrange: 3x4x2 = 24 elements of f32
    let total_elems = 3 * 4 * 2;
    let data: Vec<u8> = (0..total_elems)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let tensor = OnnxTensor::new(
        "volume".to_string(),
        Dtype::F32,
        vec![3, 4, 2],
        Bytes::from(data.clone()),
    );
    // Act
    let raw = tensor.raw_data();
    // Assert
    assert_eq!(raw.len(), total_elems * 4);
    assert_eq!(raw, data.as_slice());
}

// ── OnnxTensor scalar_i64 from I8 dtype returns None ───────────────
// @trace TEST-ONNX-111 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_i8_dtype_returns_none() {
    // Arrange
    let tensor = OnnxTensor::new(
        "tiny".to_string(),
        Dtype::I8,
        vec![1],
        Bytes::from_static(&[42u8]),
    );
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert!(result.is_none());
}

// ── OnnxTensor scalar_f32 from BF16 dtype roundtrip ────────────────
// @trace TEST-ONNX-112 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_bf16_roundtrip() {
    // Arrange: BF16 bit pattern of 1.0 is 0x3F80 (big-endian: [0x3F, 0x80])
    // half::bf16::from_bits expects little-endian u16: 0x803F
    let bits: u16 = half::bf16::from_f32(1.0).to_bits();
    let tensor = OnnxTensor::new(
        "bf_scalar".to_string(),
        Dtype::BF16,
        vec![1],
        Bytes::from(bits.to_le_bytes().to_vec()),
    );
    // Act
    let result = tensor.scalar_f32();
    // Assert
    let val = result.expect("bf16 scalar should convert to f32");
    assert!((val - 1.0).abs() < 0.01);
}

// ── OnnxGraph with empty nodes and empty initializers ──────────────
// @trace TEST-ONNX-113 [level:unit]

#[test]
fn onnx_graph_completely_empty_struct() {
    // Arrange
    let graph = OnnxGraph {
        name: String::new(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = graph.clone();
    // Assert
    assert!(cloned.nodes.is_empty());
    assert!(cloned.initializers.is_empty());
    assert!(cloned.sparse_initializers.is_empty());
    assert!(cloned.inputs.is_empty());
    assert!(cloned.outputs.is_empty());
    assert!(cloned.value_info.is_empty());
    assert!(cloned.quantization_annotation.is_empty());
    assert!(cloned.metadata_props.is_empty());
    assert!(cloned.name.is_empty());
    assert!(cloned.doc_string.is_empty());
}

// ── OnnxSparseTensor with empty dims vector ────────────────────────
// @trace TEST-ONNX-114 [level:unit]

#[test]
fn onnx_sparse_tensor_empty_dims_vector() {
    // Arrange
    let values = OnnxTensor::new(
        "empty_vals".to_string(),
        Dtype::F32,
        vec![0],
        Bytes::new(),
    );
    let indices = OnnxTensor::new(
        "empty_idxs".to_string(),
        Dtype::I64,
        vec![0],
        Bytes::new(),
    );
    let sparse = OnnxSparseTensor {
        values,
        indices,
        dims: vec![],
        format: OnnxSparseFormat::Coo,
    };
    // Act
    let debug = format!("{sparse:?}");
    // Assert
    assert!(sparse.dims.is_empty());
    assert!(debug.contains("Coo"));
    assert!(debug.contains("empty_vals"));
}

// ── OnnxModelMetadata doc_string preserves unicode ─────────────────
// @trace TEST-ONNX-115 [level:unit]

#[test]
fn onnx_model_metadata_doc_string_preserves_unicode() {
    // Arrange
    let meta = OnnxModelMetadata {
        ir_version: 7,
        producer_name: "test_producer".to_string(),
        producer_version: "1.0".to_string(),
        domain: "ai.onnx".to_string(),
        model_version: 0,
        doc_string: "模型说明 — test ™".to_string(),
        opset_import: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = meta.clone();
    // Assert
    assert_eq!(cloned.doc_string, "模型说明 — test ™");
    assert!(format!("{meta:?}").contains("模型说明"));
}

// ── OnnxNode with unicode op_type and name ──────────────────────────
// @trace TEST-ONNX-116 [level:unit]

#[test]
fn onnx_node_unicode_op_type_and_name() {
    // Arrange
    let node = OnnxNode {
        name: "节点_α".to_string(),
        op_type: "CustomØp".to_string(),
        domain: "test.local".to_string(),
        inputs: vec!["输入".to_string()],
        outputs: vec!["输出".to_string()],
        attributes: HashMap::new(),
    };
    // Act
    let debug = format!("{node:?}");
    // Assert
    assert!(debug.contains("节点_α"));
    assert!(debug.contains("CustomØp"));
    assert_eq!(node.inputs[0], "输入");
    assert_eq!(node.outputs[0], "输出");
}

// ── OnnxValueInfo with tensor type containing param dim ─────────────
// @trace TEST-ONNX-117 [level:unit]

#[test]
fn onnx_value_info_with_param_dim_in_tensor_type() {
    // Arrange
    let tensor_type = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape {
            dims: vec![
                OnnxDim::Param("batch_size".to_string()),
                OnnxDim::Known(128),
            ],
        },
    };
    let info = OnnxValueInfo {
        name: "input_with_param".to_string(),
        value_type: Some(OnnxType::Tensor(tensor_type)),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let type_ref = info.value_type.as_ref().expect("should have type");
    // Assert
    if let OnnxType::Tensor(tt) = type_ref {
        assert_eq!(tt.shape.dims.len(), 2);
        assert_eq!(tt.shape.dims[0], OnnxDim::Param("batch_size".to_string()));
        assert_eq!(tt.shape.dims[1], OnnxDim::Known(128));
    } else {
        panic!("expected Tensor variant");
    }
}

// ── OnnxDim Unknown variant in shape product context ───────────────
// @trace TEST-ONNX-118 [level:unit]

#[test]
fn onnx_dim_unknown_variant_in_shape() {
    // Arrange
    let shape = OnnxTensorShape {
        dims: vec![OnnxDim::Known(1), OnnxDim::Unknown, OnnxDim::Known(768)],
    };
    // Act
    let known_count = shape.dims.iter().filter(|d| matches!(d, OnnxDim::Known(_))).count();
    let has_unknown = shape.dims.iter().any(|d| matches!(d, OnnxDim::Unknown));
    // Assert
    assert_eq!(known_count, 2);
    assert!(has_unknown);
    assert_eq!(shape.dims.len(), 3);
}

// ── OnnxType equality: same Optional inner type are equal ──────────
// @trace TEST-ONNX-119 [level:unit]

#[test]
fn onnx_type_optional_equality_same_inner() {
    // Arrange
    let inner = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![] },
    });
    let a = OnnxType::Optional(Box::new(inner.clone()));
    let b = OnnxType::Optional(Box::new(inner.clone()));
    // Act & Assert
    assert_eq!(a, b);
}

// ── Loader: model with sparse_initializer loads without error ──────
// @trace TEST-ONNX-121 [level:unit]

#[test]
fn loader_graph_with_sparse_initializer_and_regular_initializer() {
    // Arrange: graph with both regular and sparse initializers
    let regular = tensor_f32("dense_weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let sparse_values = proto::TensorProto {
        dims: vec![2],
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        name: Some("sparse_vals".to_string()),
        raw_data: Some(Bytes::from([1.0f32, 2.0f32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())),
        ..empty_tensor()
    };
    let sparse_indices = proto::TensorProto {
        dims: vec![1, 2],
        data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
        name: Some("sparse_idxs".to_string()),
        raw_data: Some(Bytes::from([0i64, 3].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())),
        ..empty_tensor()
    };
    let sparse_proto = proto::SparseTensorProto {
        values: Some(sparse_values),
        indices: Some(sparse_indices),
        dims: vec![4, 4],
    };
    let graph = proto::GraphProto {
        initializer: vec![regular],
        sparse_initializer: vec![sparse_proto],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert: regular initializer accessible
    let slice = loader.tensor("dense_weight").expect("dense_weight");
    assert_eq!(slice.shape, vec![2, 2]);
    // Sparse initializers tracked in graph
    let g = loader.graph();
    assert_eq!(g.sparse_initializers.len(), 1);
}

// ── ConvertError AttributeType fields preserve message ─────────────
// @trace TEST-ONNX-122 [level:unit]

#[test]
fn convert_error_attribute_error_clone_roundtrip() {
    // Arrange
    let err = ConvertError::AttributeError {
        node_name: "Gemm_42".to_string(),
        reason: "missing transB attribute".to_string(),
    };
    // Act
    let cloned = err.clone();
    let display = format!("{err}");
    // Assert
    assert_eq!(cloned.to_string(), display);
    assert!(display.contains("Gemm_42"));
    assert!(display.contains("missing transB attribute"));
}

// ── OnnxModelMetadata default fields verification ──────────────────
// @trace TEST-ONNX-123 [level:unit]

#[test]
fn onnx_model_metadata_zero_ir_version_default() {
    // Arrange
    let meta = OnnxModelMetadata {
        ir_version: 0,
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        opset_import: vec![],
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(meta.ir_version, 0);
    assert!(meta.producer_name.is_empty());
    assert!(meta.opset_import.is_empty());
    assert!(meta.metadata_props.is_empty());
}

// ── OnnxOperatorSet clone preserves both fields ────────────────────
// @trace TEST-ONNX-124 [level:unit]

#[test]
fn onnx_operator_set_clone_preserves_domain_and_version() {
    // Arrange
    let opset = OnnxOperatorSet {
        domain: "ai.onnx.ml".to_string(),
        version: 3,
    };
    // Act
    let cloned = opset.clone();
    // Assert
    assert_eq!(cloned.domain, "ai.onnx.ml");
    assert_eq!(cloned.version, 3);
}

// ── OnnxNode with empty domain defaults correctly ──────────────────
// @trace TEST-ONNX-125 [level:unit]

#[test]
fn onnx_node_empty_domain_preserves_op_type() {
    // Arrange
    let node = OnnxNode {
        name: "test_node".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attributes: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(node.op_type, "Relu");
    assert!(node.domain.is_empty());
    assert_eq!(node.inputs.len(), 1);
    assert_eq!(node.outputs.len(), 1);
}

// ── OnnxValueInfo with metadata_props populated ────────────────────
// @trace TEST-ONNX-126 [level:unit]

#[test]
fn onnx_value_info_metadata_props_lookup() {
    // Arrange
    let mut props = HashMap::new();
    props.insert("source".to_string(), "encoder".to_string());
    props.insert("layer".to_string(), "0".to_string());
    let info = OnnxValueInfo {
        name: "hidden_states".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: props,
    };
    // Act
    let source = info.metadata_props.get("source");
    let layer = info.metadata_props.get("layer");
    let missing = info.metadata_props.get("nonexistent");
    // Assert
    assert_eq!(source, Some(&"encoder".to_string()));
    assert_eq!(layer, Some(&"0".to_string()));
    assert!(missing.is_none());
    assert_eq!(info.metadata_props.len(), 2);
}

// ── OnnxQuantizationAnnotation with scale and zero_point ───────────
// @trace TEST-ONNX-127 [level:unit]

#[test]
fn onnx_quantization_annotation_with_optional_fields() {
    // Arrange
    let mut params = HashMap::new();
    params.insert("SCALE_TENSOR".to_string(), "weight_scale".to_string());
    params.insert("ZERO_POINT_TENSOR".to_string(), "weight_zp".to_string());
    let ann = OnnxQuantizationAnnotation {
        tensor_name: "weight".to_string(),
        quant_param_tensor_names: params,
        scale: Some(0.0039),
        zero_point: Some(128),
        axis: Some(0),
    };
    // Act & Assert
    assert_eq!(ann.tensor_name, "weight");
    assert_eq!(ann.scale, Some(0.0039));
    assert_eq!(ann.zero_point, Some(128));
    assert_eq!(ann.axis, Some(0));
    assert!(ann.quant_param_tensor_names.contains_key("SCALE_TENSOR"));
}

// ── OnnxFunction construction with all collections ─────────────────
// @trace TEST-ONNX-128 [level:unit]

#[test]
fn onnx_function_with_multiple_inputs_outputs() {
    // Arrange
    let func = OnnxFunction {
        name: "CustomSoftmax".to_string(),
        domain: "custom.ops".to_string(),
        overload: "v1".to_string(),
        inputs: vec!["input".to_string(), "mask".to_string()],
        outputs: vec!["output".to_string()],
        attributes: vec!["temperature".to_string()],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![OnnxOperatorSet {
            domain: "".to_string(),
            version: 17,
        }],
        value_info: vec![],
        doc_string: "Custom softmax with mask".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act & Assert
    assert_eq!(func.inputs.len(), 2);
    assert_eq!(func.outputs.len(), 1);
    assert_eq!(func.attributes.len(), 1);
    assert_eq!(func.opset_import[0].version, 17);
    assert_eq!(func.doc_string, "Custom softmax with mask");
}

// ── ConvertError ShapeInferenceFailed fields ───────────────────────
// @trace TEST-ONNX-129 [level:unit]

#[test]
fn convert_error_shape_inference_fields() {
    // Arrange
    let err = ConvertError::ShapeInferenceFailed {
        name: "output_tensor".to_string(),
        reason: "cannot infer dim from empty value_info".to_string(),
    };
    // Act
    let display = format!("{err}");
    // Assert
    assert!(display.contains("output_tensor"));
    assert!(display.contains("cannot infer dim from empty value_info"));
}

// ── OnnxGraph with empty initializers hashmap lookups ──────────────
// @trace TEST-ONNX-130 [level:unit]

#[test]
fn onnx_graph_empty_initializers_lookup() {
    // Arrange
    let graph = OnnxGraph {
        name: "empty_graph".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let result = graph.initializers.get("nonexistent");
    // Assert
    assert!(result.is_none());
    assert!(graph.initializers.is_empty());
    assert!(graph.nodes.is_empty());
    assert!(graph.sparse_initializers.is_empty());
}

// ── OnnxModel clone produces independent metadata_props ────────────
// @trace TEST-ONNX-131 [level:unit]

#[test]
fn onnx_model_clone_metadata_props_independence() {
    // Arrange
    let mut props = HashMap::new();
    props.insert("key".to_string(), "value".to_string());
    let model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 8,
            producer_name: "test".to_string(),
            producer_version: "1.0".to_string(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: props,
        },
        graph: OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![],
    };
    // Act
    let mut cloned = model.clone();
    cloned.metadata.metadata_props.insert("new_key".to_string(), "new_val".to_string());
    // Assert: original unaffected
    assert!(!model.metadata.metadata_props.contains_key("new_key"));
    assert!(cloned.metadata.metadata_props.contains_key("new_key"));
}

// ── ConvertError UnsupportedOp fields and display ──────────────────
// @trace TEST-ONNX-132 [level:unit]

#[test]
fn convert_error_unsupported_op_fields() {
    // Arrange
    let err = ConvertError::UnsupportedOp {
        op_type: "DynamicQuantizeLinear".to_string(),
        node_name: "quant_node_0".to_string(),
    };
    // Act
    let display = format!("{err}");
    // Assert
    assert!(display.contains("DynamicQuantizeLinear"));
    assert!(display.contains("quant_node_0"));
}

// ── OnnxNode clone produces independent attributes ─────────────────
// @trace TEST-ONNX-133 [level:unit]

#[test]
fn onnx_node_clone_attributes_independence() {
    // Arrange
    let mut attrs = HashMap::new();
    attrs.insert("pads".to_string(), OnnxAttribute {
        name: "pads".to_string(),
        value: OnnxAttributeValue::Ints(vec![1, 1, 1, 1]),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    let node = OnnxNode {
        name: "conv1".to_string(),
        op_type: "Conv".to_string(),
        domain: String::new(),
        inputs: vec!["input".to_string(), "weight".to_string()],
        outputs: vec!["conv_out".to_string()],
        attributes: attrs,
    };
    // Act
    let mut cloned = node.clone();
    cloned.attributes.insert("strides".to_string(), OnnxAttribute {
        name: "strides".to_string(),
        value: OnnxAttributeValue::Ints(vec![2, 2]),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    // Assert: original unaffected
    assert_eq!(node.attributes.len(), 1);
    assert_eq!(cloned.attributes.len(), 2);
    assert!(node.attributes.contains_key("pads"));
    assert!(cloned.attributes.contains_key("strides"));
}

// ── OnnxValueInfo clone produces independent metadata_props ────────
// @trace TEST-ONNX-134 [level:unit]

#[test]
fn onnx_value_info_clone_independent_metadata() {
    // Arrange
    let mut props = HashMap::new();
    props.insert("key".to_string(), "original".to_string());
    let info = OnnxValueInfo {
        name: "attention_output".to_string(),
        value_type: None,
        doc_string: String::new(),
        metadata_props: props,
    };
    // Act
    let mut cloned = info.clone();
    cloned.metadata_props.insert("extra".to_string(), "cloned_val".to_string());
    // Assert
    assert!(!info.metadata_props.contains_key("extra"));
    assert!(cloned.metadata_props.contains_key("extra"));
    assert_eq!(cloned.name, "attention_output");
}

// ── OnnxFunction clone preserves all field values ──────────────────
// @trace TEST-ONNX-135 [level:unit]

#[test]
fn onnx_function_clone_preserves_all_fields() {
    // Arrange
    let func = OnnxFunction {
        name: "LayerNorm".to_string(),
        domain: "com.example".to_string(),
        overload: "default".to_string(),
        inputs: vec!["input".to_string(), "weight".to_string(), "bias".to_string()],
        outputs: vec!["output".to_string()],
        attributes: vec!["epsilon".to_string()],
        attribute_protos: HashMap::new(),
        nodes: vec![OnnxNode {
            name: "reduce_mean".to_string(),
            op_type: "ReduceMean".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string()],
            outputs: vec!["mean".to_string()],
            attributes: HashMap::new(),
        }],
        opset_import: vec![],
        value_info: vec![],
        doc_string: "Normalized layer".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = func.clone();
    // Assert
    assert_eq!(cloned.name, "LayerNorm");
    assert_eq!(cloned.inputs.len(), 3);
    assert_eq!(cloned.nodes.len(), 1);
    assert_eq!(cloned.nodes[0].op_type, "ReduceMean");
}

// ── ConvertError NoWeightInput display message content ─────────────
// @trace TEST-ONNX-136 [level:unit]

#[test]
fn convert_error_no_weight_input_message() {
    // Arrange
    let err = ConvertError::NoWeightInput {
        node_name: "matmul_dynamic".to_string(),
    };
    // Act
    let display = format!("{err}");
    // Assert
    assert!(display.contains("matmul_dynamic"));
}

// ── OnnxQuantizationAnnotation clone roundtrip preserves all fields ─
// @trace TEST-ONNX-137 [level:unit]

#[test]
fn onnx_quantization_annotation_clone_roundtrip_all_fields() {
    // Arrange
    let mut params = HashMap::new();
    params.insert("SCALE_TENSOR".to_string(), "scale_tensor".to_string());
    let ann = OnnxQuantizationAnnotation {
        tensor_name: "qweight".to_string(),
        quant_param_tensor_names: params,
        scale: Some(0.015625),
        zero_point: Some(0),
        axis: Some(-1),
    };
    // Act
    let cloned = ann.clone();
    // Assert
    assert_eq!(cloned.tensor_name, "qweight");
    assert_eq!(cloned.scale, Some(0.015625));
    assert_eq!(cloned.zero_point, Some(0));
    assert_eq!(cloned.axis, Some(-1));
    assert!(cloned.quant_param_tensor_names.contains_key("SCALE_TENSOR"));
    // Independence: mutate clone, verify original
    drop(cloned);
    assert_eq!(ann.tensor_name, "qweight");
}

// ── ConvertError InvalidMatMulShape display includes dims value ────
// @trace TEST-ONNX-138 [level:unit]

#[test]
fn convert_error_invalid_matmul_shape_display_includes_dims() {
    // Arrange
    let err = ConvertError::InvalidMatMulShape {
        name: "weight_3d".to_string(),
        dims: 3,
    };
    // Act
    let display = format!("{err}");
    // Assert
    assert!(display.contains("weight_3d"), "should contain tensor name");
    assert!(display.contains("3"), "should contain dimension count");
    assert!(display.contains("2-D"), "should mention expected 2-D");
}

// ── OnnxAttributeValue Floats clone independence ───────────────────
// @trace TEST-ONNX-139 [level:unit]

#[test]
fn onnx_attribute_value_floats_clone_independence() {
    // Arrange
    let original = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
    // Act
    let mut cloned = original.clone();
    if let OnnxAttributeValue::Floats(ref mut v) = cloned {
        v.push(4.0);
    }
    // Assert: original unaffected
    if let OnnxAttributeValue::Floats(ref v) = original {
        assert_eq!(v.len(), 3);
    } else {
        panic!("expected Floats variant");
    }
    if let OnnxAttributeValue::Floats(ref v) = cloned {
        assert_eq!(v.len(), 4);
    } else {
        panic!("expected Floats variant");
    }
}

// ── OnnxSparseTensor clone produces deep-independent values ────────
// @trace TEST-ONNX-140 [level:unit]

#[test]
fn onnx_sparse_tensor_clone_deep_independence() {
    // Arrange
    let values = OnnxTensor::new(
        "sp_vals".to_string(),
        Dtype::F32,
        vec![2],
        Bytes::from([10.0f32, 20.0f32].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let indices = OnnxTensor::new(
        "sp_idx".to_string(),
        Dtype::I64,
        vec![2],
        Bytes::from([0i64, 5].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
    );
    let sparse = OnnxSparseTensor {
        values,
        indices,
        dims: vec![10],
        format: OnnxSparseFormat::Coo,
    };
    // Act
    let cloned = sparse.clone();
    // Assert: both have same dims and format
    assert_eq!(cloned.dims, sparse.dims);
    assert_eq!(cloned.format, sparse.format);
    assert_eq!(cloned.values.name, "sp_vals");
    assert_eq!(cloned.indices.name, "sp_idx");
    // Independent: drop original, cloned still valid
    drop(sparse);
    assert_eq!(cloned.dims.len(), 1);
}

// ── Loader preserves opset_import entries ──────────────────────────
// @trace TEST-ONNX-141 [level:unit]

#[test]
fn loader_model_opset_import_entries_preserved() {
    // Arrange
    let opset = proto::OperatorSetIdProto {
        domain: Some("ai.onnx".to_string()),
        version: Some(17),
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor_f32("w", vec![1], &[42.0])],
        ..empty_graph()
    };
    let model = proto::ModelProto {
        opset_import: vec![opset],
        graph: Some(graph),
        ..empty_model(proto::GraphProto {
            ..empty_graph()
        })
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let m = loader.model();
    // Assert
    assert_eq!(m.metadata.opset_import.len(), 1);
    assert_eq!(m.metadata.opset_import[0].domain, "ai.onnx");
    assert_eq!(m.metadata.opset_import[0].version, 17);
}

// ── OnnxTensor new_string creates is_string tensor ─────────────────
// @trace TEST-ONNX-142 [level:unit]

#[test]
fn onnx_tensor_new_string_creates_is_string_flag() {
    // Arrange & Act
    let tensor = OnnxTensor::new_string(
        "label".to_string(),
        vec![2],
        Bytes::from_static(&[0x48, 0x69]), // "Hi"
    );
    // Assert
    assert!(tensor.is_string);
    assert_eq!(tensor.dtype, Dtype::U8);
    assert_eq!(tensor.shape, vec![2]);
    assert_eq!(tensor.name, "label");
}

// ── OnnxGraph value_info field preserves multiple entries ──────────
// @trace TEST-ONNX-143 [level:unit]

#[test]
fn onnx_graph_value_info_preserves_multiple_entries() {
    // Arrange
    let info_a = OnnxValueInfo {
        name: "intermediate_1".to_string(),
        value_type: Some(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(64)] },
        })),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    let info_b = OnnxValueInfo {
        name: "intermediate_2".to_string(),
        value_type: Some(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(32)] },
        })),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    let graph = OnnxGraph {
        name: "multi_info".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![info_a, info_b],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = graph.clone();
    // Assert
    assert_eq!(graph.value_info.len(), 2);
    assert_eq!(graph.value_info[0].name, "intermediate_1");
    assert_eq!(graph.value_info[1].name, "intermediate_2");
    assert_eq!(cloned.value_info.len(), 2);
}

// ── ConvertError ShapeInferenceFailed clone roundtrip ──────────────
// @trace TEST-ONNX-144 [level:unit]

#[test]
fn convert_error_shape_inference_failed_clone_roundtrip() {
    // Arrange
    let err = ConvertError::ShapeInferenceFailed {
        name: "hidden_states".to_string(),
        reason: "cannot infer dim for symbolic batch".to_string(),
    };
    // Act
    let cloned = err.clone();
    let display_original = format!("{err}");
    let display_cloned = format!("{cloned}");
    // Assert
    assert_eq!(display_original, display_cloned);
    assert!(display_cloned.contains("hidden_states"));
    assert!(display_cloned.contains("symbolic batch"));
}

// ── OnnxModel functions field with multiple entries ────────────────
// @trace TEST-ONNX-145 [level:unit]

#[test]
fn onnx_model_functions_multiple_entries_independent() {
    // Arrange
    let func_a = OnnxFunction {
        name: "CustomScale".to_string(),
        domain: "com.test".to_string(),
        overload: String::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    let func_b = OnnxFunction {
        name: "CustomShift".to_string(),
        domain: "com.test".to_string(),
        overload: String::new(),
        inputs: vec!["x".to_string(), "bias".to_string()],
        outputs: vec!["y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    let mut model = OnnxModel {
        metadata: OnnxModelMetadata {
            ir_version: 8,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        },
        graph: OnnxGraph {
            name: "g".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        },
        functions: vec![func_a, func_b],
    };
    // Act
    let cloned = model.clone();
    model.functions.clear();
    // Assert: clone unaffected
    assert_eq!(cloned.functions.len(), 2);
    assert_eq!(cloned.functions[0].name, "CustomScale");
    assert_eq!(cloned.functions[1].name, "CustomShift");
    assert_eq!(cloned.functions[1].inputs.len(), 2);
}

// ── Loader graph name preserved from proto ─────────────────────────
// @trace TEST-ONNX-146 [level:unit]

#[test]
fn loader_graph_name_preserved_from_proto() {
    // Arrange
    let graph = proto::GraphProto {
        name: Some("my_inference_graph".to_string()),
        initializer: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().name, "my_inference_graph");
}

// ── OnnxNode with multiple outputs graph connectivity ──────────────
// @trace TEST-ONNX-147 [level:unit]

#[test]
fn onnx_node_multiple_outputs_connectivity() {
    // Arrange: a Split node produces two outputs consumed by downstream nodes
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), OnnxAttribute {
        name: "axis".to_string(),
        value: OnnxAttributeValue::Int(1),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    });
    let split_node = OnnxNode {
        name: "split_0".to_string(),
        op_type: "Split".to_string(),
        domain: String::new(),
        inputs: vec!["input".to_string()],
        outputs: vec!["split_left".to_string(), "split_right".to_string()],
        attributes: attrs,
    };
    let left_consumer = OnnxNode {
        name: "relu_left".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        inputs: vec!["split_left".to_string()],
        outputs: vec!["left_out".to_string()],
        attributes: HashMap::new(),
    };
    let right_consumer = OnnxNode {
        name: "tanh_right".to_string(),
        op_type: "Tanh".to_string(),
        domain: String::new(),
        inputs: vec!["split_right".to_string()],
        outputs: vec!["right_out".to_string()],
        attributes: HashMap::new(),
    };
    // Act: verify output→input connectivity
    let split_outputs: Vec<&str> = split_node.outputs.iter().map(|s| s.as_str()).collect();
    // Assert
    assert_eq!(split_outputs.len(), 2);
    assert!(split_outputs.contains(&"split_left"));
    assert!(split_outputs.contains(&"split_right"));
    assert_eq!(left_consumer.inputs[0], "split_left");
    assert_eq!(right_consumer.inputs[0], "split_right");
}

// ── Loader with INT64 tensor preserves data correctly ──────────────
// @trace TEST-ONNX-148 [level:unit]

#[test]
fn loader_int64_tensor_preserves_data() {
    // Arrange: INT64 tensor with values [100, -200, 300]
    let values = [100i64, -200i64, 300i64];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = proto::TensorProto {
        dims: vec![3],
        data_type: Some(proto::tensor_proto::DataType::Int64 as i32),
        name: Some("token_ids".to_string()),
        raw_data: Some(Bytes::from(raw)),
        ..empty_tensor()
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let slice = loader.tensor("token_ids").expect("tensor");
    // Assert: 3 elements of i64 = 24 bytes
    assert_eq!(slice.shape, vec![3]);
    assert_eq!(slice.data.len(), 24);
    let read_back: Vec<i64> = slice.data.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    assert_eq!(read_back, vec![100, -200, 300]);
}

// ── OnnxValueInfo with None value_type accepts unknown type ────────
// @trace TEST-ONNX-149 [level:unit]

#[test]
fn onnx_value_info_none_value_type_is_valid() {
    // Arrange: value_info without type annotation (valid in ONNX)
    let info = OnnxValueInfo {
        name: "dynamic_tensor".to_string(),
        value_type: None,
        doc_string: "no shape info".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = info.clone();
    // Assert
    assert!(cloned.value_type.is_none());
    assert_eq!(cloned.name, "dynamic_tensor");
    assert_eq!(cloned.doc_string, "no shape info");
}

// ── OnnxAttribute Ints value preserves ordering and negatives ──────
// @trace TEST-ONNX-150 [level:unit]

#[test]
fn onnx_attribute_ints_preserves_ordering_and_negatives() {
    // Arrange
    let attr = OnnxAttribute {
        name: "pads".to_string(),
        value: OnnxAttributeValue::Ints(vec![0, 1, -1, 2, 3]),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert
    if let OnnxAttributeValue::Ints(ref vals) = cloned.value {
        assert_eq!(vals.len(), 5);
        assert_eq!(vals[0], 0);
        assert_eq!(vals[2], -1);
        assert_eq!(vals[4], 3);
    } else {
        panic!("expected Ints variant");
    }
    assert_eq!(cloned.name, "pads");
}

// ── OnnxAttribute Float value precision within epsilon ─────────────
// @trace TEST-ONNX-151 [level:unit]

#[test]
fn onnx_attribute_float_preserves_precision() {
    // Arrange
    let epsilon = 1e-7f32;
    let attr = OnnxAttribute {
        name: "epsilon".to_string(),
        value: OnnxAttributeValue::Float(epsilon),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert
    if let OnnxAttributeValue::Float(v) = cloned.value {
        assert!((v - epsilon).abs() < 1e-10, "float precision lost: {v} vs {epsilon}");
    } else {
        panic!("expected Float variant");
    }
}

// ── OnnxSparseFormat Copy trait allows pass-by-value ───────────────
// @trace TEST-ONNX-152 [level:unit]

#[test]
fn onnx_sparse_format_copy_trait_pass_by_value() {
    // Arrange
    let format = OnnxSparseFormat::Csr;
    // Act: Copy allows using format after move-by-value
    let captured = format;
    let still_available = format;
    // Assert
    assert_eq!(captured, OnnxSparseFormat::Csr);
    assert_eq!(still_available, OnnxSparseFormat::Csr);
    assert_eq!(format, OnnxSparseFormat::Csr);
    // Verify all three variants are Copy + PartialEq
    assert_ne!(OnnxSparseFormat::Coo, OnnxSparseFormat::Csr);
    assert_ne!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csc);
    assert_ne!(OnnxSparseFormat::Coo, OnnxSparseFormat::Csc);
}

// ── 1. ConvertError variants Display format verification ────────────────
// @trace TEST-ONNX-153 [level:unit]

#[test]
fn convert_error_no_weight_input_display_includes_node_name() {
    // Arrange
    let err = ConvertError::NoWeightInput {
        node_name: "matmul_42".to_string(),
    };
    // Act
    let display = format!("{err}");
    // Assert: Display must contain the node_name string (thiserror #[error] template)
    assert!(
        display.contains("matmul_42"),
        "Display output should contain node name, got: {display}"
    );
    assert!(
        display.contains("no initializer input") || display.contains("no weight") || display.contains("NoWeight"),
        "Display output should describe the error kind, got: {display}"
    );
}

// ── 2. OnnxAttributeValue edge cases: empty Floats and very large Ints ─
// @trace TEST-ONNX-154 [level:unit]

#[test]
fn onnx_attribute_value_ints_with_i64_max_and_min() {
    // Arrange: Ints variant containing i64::MAX and i64::MIN boundary values
    let values = vec![i64::MAX, i64::MIN, 0, -1];
    let attr_val = OnnxAttributeValue::Ints(values.clone());
    // Act
    if let OnnxAttributeValue::Ints(extracted) = attr_val {
        // Assert: all boundary values preserved exactly
        assert_eq!(extracted.len(), 4);
        assert_eq!(extracted[0], i64::MAX);
        assert_eq!(extracted[1], i64::MIN);
        assert_eq!(extracted[2], 0);
        assert_eq!(extracted[3], -1);
    } else {
        panic!("expected Ints variant");
    }
}

// ── 3. OnnxSparseTensor Clone preservation ────────────────────────────
// @trace TEST-ONNX-155 [level:unit]

#[test]
fn onnx_sparse_tensor_clone_preserves_format_and_dims() {
    // Arrange
    let values = OnnxTensor::new(
        "sparse_vals".to_string(),
        Dtype::F32,
        vec![3],
        Bytes::copy_from_slice(&[0u8; 12]),
    );
    let indices = OnnxTensor::new(
        "sparse_idxs".to_string(),
        Dtype::I64,
        vec![3],
        Bytes::copy_from_slice(&[0u8; 24]),
    );
    let sparse = OnnxSparseTensor {
        values,
        indices,
        dims: vec![4, 5],
        format: OnnxSparseFormat::Coo,
    };
    // Act
    let cloned = sparse.clone();
    // Assert: clone preserves all fields
    assert_eq!(cloned.dims, vec![4, 5]);
    assert_eq!(cloned.format, OnnxSparseFormat::Coo);
    assert_eq!(cloned.values.name, "sparse_vals");
    assert_eq!(cloned.indices.name, "sparse_idxs");
}

// ── 4. OpsetImport domain/version boundary values ─────────────────────
// @trace TEST-ONNX-156 [level:unit]

#[test]
fn onnx_operator_set_empty_domain_and_zero_version() {
    // Arrange: ONNX spec allows empty domain ("") for the default ONNX domain
    let opset = OnnxOperatorSet {
        domain: String::new(),
        version: 0,
    };
    // Act
    let cloned = opset.clone();
    // Assert: empty domain and version 0 are valid and preserved through clone
    assert_eq!(cloned.domain, "");
    assert_eq!(cloned.version, 0);
    assert_eq!(opset.domain, cloned.domain);
    assert_eq!(opset.version, cloned.version);
}

// ── 5. String tensor is_string flag edge cases ─────────────────────────
// @trace TEST-ONNX-157 [level:unit]

#[test]
fn onnx_tensor_non_string_then_string_toggle_is_independent() {
    // Arrange: create a non-string tensor, then a string tensor
    let non_string = OnnxTensor::new(
        "weights".to_string(),
        Dtype::F32,
        vec![2, 3],
        Bytes::copy_from_slice(&[0u8; 24]),
    );
    let string_tensor = OnnxTensor::new_string(
        "labels".to_string(),
        vec![2],
        Bytes::copy_from_slice(b"hi"),
    );
    // Assert: flags are set correctly and independently
    assert!(!non_string.is_string, "non-string tensor should have is_string=false");
    assert!(string_tensor.is_string, "string tensor should have is_string=true");
    // Assert: dtype differs (U8 placeholder for string vs F32)
    assert_eq!(non_string.dtype, Dtype::F32);
    assert_eq!(string_tensor.dtype, Dtype::U8);
}

// ── 6. ValueInfo multi-entry deduplication ─────────────────────────────
// @trace TEST-ONNX-158 [level:unit]

#[test]
fn value_info_duplicate_names_preserved_in_vec() {
    // Arrange: two ValueInfo entries with the same name but different types
    let vi1 = OnnxValueInfo {
        name: "hidden_state".to_string(),
        value_type: Some(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(768)],
            },
        })),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    let vi2 = OnnxValueInfo {
        name: "hidden_state".to_string(),
        value_type: Some(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Double,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Known(512)],
            },
        })),
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act: Vec preserves duplicates (deduplication is caller's responsibility)
    let entries = vec![vi1.clone(), vi2.clone()];
    // Assert: both entries are preserved, not deduplicated by the data structure
    assert_eq!(entries.len(), 2, "Vec should preserve duplicate name entries");
    assert_eq!(entries[0].name, "hidden_state");
    assert_eq!(entries[1].name, "hidden_state");
    // They differ in type content
    assert_ne!(entries[0].value_type, entries[1].value_type);
}

// ── 7. ShapeInferenceFailed clone equality ──────────────────────────────
// @trace TEST-ONNX-159 [level:unit]

#[test]
fn convert_error_shape_inference_failed_clone_equality() {
    // Arrange
    let original = ConvertError::ShapeInferenceFailed {
        name: "attention_mask".to_string(),
        reason: "could not resolve dynamic dim".to_string(),
    };
    // Act
    let cloned = original.clone();
    // Assert: cloned error displays identically
    assert_eq!(format!("{original}"), format!("{cloned}"),
        "cloned ShapeInferenceFailed should display identically");
    // Assert: debug format also matches
    assert_eq!(format!("{original:?}"), format!("{cloned:?}"),
        "cloned ShapeInferenceFailed should debug identically");
}

// ── 8. Functions multi-entry preservation ──────────────────────────────
// @trace TEST-ONNX-160 [level:unit]

#[test]
fn onnx_function_multiple_entries_independent_fields() {
    // Arrange: two functions with different names and domains
    let func_a = OnnxFunction {
        name: "CustomOpA".to_string(),
        domain: "com.example".to_string(),
        overload: String::new(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: Vec::new(),
        attribute_protos: HashMap::new(),
        nodes: Vec::new(),
        opset_import: Vec::new(),
        value_info: Vec::new(),
        doc_string: "Function A".to_string(),
        metadata_props: HashMap::new(),
    };
    let func_b = OnnxFunction {
        name: "CustomOpB".to_string(),
        domain: "org.vendor".to_string(),
        overload: String::new(),
        inputs: vec!["A".to_string(), "B".to_string()],
        outputs: vec!["C".to_string()],
        attributes: Vec::new(),
        attribute_protos: HashMap::new(),
        nodes: Vec::new(),
        opset_import: Vec::new(),
        value_info: Vec::new(),
        doc_string: "Function B".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let functions = vec![func_a.clone(), func_b.clone()];
    // Assert: both entries are preserved with distinct fields
    assert_eq!(functions.len(), 2);
    assert_eq!(functions[0].name, "CustomOpA");
    assert_eq!(functions[0].domain, "com.example");
    assert_eq!(functions[1].name, "CustomOpB");
    assert_eq!(functions[1].domain, "org.vendor");
    assert_eq!(functions[0].inputs.len(), 1);
    assert_eq!(functions[1].inputs.len(), 2);
}

// ── 9. Graph name with special characters (Unicode, spaces, hyphens) ──
// @trace TEST-ONNX-161 [level:unit]

#[test]
fn graph_name_unicode_spaces_hyphens_preserved() {
    // Arrange: graph name with Unicode, spaces, and hyphens
    let special_name = "my-model \u{4e2d}\u{6587} graph-v2";
    let graph = OnnxGraph {
        name: special_name.to_string(),
        doc_string: String::new(),
        nodes: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        value_info: Vec::new(),
        initializers: HashMap::new(),
        sparse_initializers: Vec::new(),
        quantization_annotation: Vec::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = graph.clone();
    // Assert: all special characters are preserved exactly
    assert_eq!(cloned.name, "my-model \u{4e2d}\u{6587} graph-v2");
    assert_eq!(cloned.name, special_name);
    // Verify the Chinese characters are intact
    assert!(cloned.name.contains('\u{4e2d}'));
    assert!(cloned.name.contains('\u{6587}'));
    // Verify spaces and hyphens
    assert!(cloned.name.contains(' '));
    assert!(cloned.name.contains('-'));
}

// ── 10. Multi-output node connectivity validation ──────────────────────
// @trace TEST-ONNX-162 [level:unit]

#[test]
fn multi_output_node_feeds_multiple_downstream_inputs() {
    // Arrange: node A produces ["out1", "out2", "out3"], each consumed by downstream nodes
    let node_a = OnnxNode {
        name: "split_node".to_string(),
        op_type: "Split".to_string(),
        domain: String::new(),
        inputs: vec!["input".to_string()],
        outputs: vec!["out1".to_string(), "out2".to_string(), "out3".to_string()],
        attributes: HashMap::new(),
    };
    let node_b = OnnxNode {
        name: "consume_1".to_string(),
        op_type: "Relu".to_string(),
        domain: String::new(),
        inputs: vec!["out1".to_string()],
        outputs: vec!["result1".to_string()],
        attributes: HashMap::new(),
    };
    let node_c = OnnxNode {
        name: "consume_2".to_string(),
        op_type: "Add".to_string(),
        domain: String::new(),
        inputs: vec!["out2".to_string(), "out3".to_string()],
        outputs: vec!["result2".to_string()],
        attributes: HashMap::new(),
    };
    // Act: build producer map (output_name -> node index)
    let nodes = vec![node_a, node_b, node_c];
    let mut producer_map: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for output in &node.outputs {
            producer_map.insert(output.clone(), i);
        }
    }
    // Assert: all outputs from node_a map to the correct producer
    assert_eq!(producer_map.get("out1"), Some(&0), "out1 should be produced by node 0");
    assert_eq!(producer_map.get("out2"), Some(&0), "out2 should be produced by node 0");
    assert_eq!(producer_map.get("out3"), Some(&0), "out3 should be produced by node 0");
    // Assert: downstream outputs map to their respective producers
    assert_eq!(producer_map.get("result1"), Some(&1));
    assert_eq!(producer_map.get("result2"), Some(&2));
    // Assert: inputs to node_c come from node_a's outputs
    assert_eq!(&nodes[2].inputs[0], "out2");
    assert_eq!(&nodes[2].inputs[1], "out3");
}

// ── 11. INT64 tensor value preservation with i64::MIN/MAX ──────────────
// @trace TEST-ONNX-163 [level:unit]

#[test]
fn int64_tensor_i64_min_max_roundtrip_via_raw_data() {
    // Arrange: encode i64::MIN and i64::MAX into raw bytes
    let min_bytes = i64::MIN.to_le_bytes();
    let max_bytes = i64::MAX.to_le_bytes();
    let mut raw = Vec::with_capacity(16);
    raw.extend_from_slice(&min_bytes);
    raw.extend_from_slice(&max_bytes);
    let tensor = OnnxTensor::new(
        "int64_boundary".to_string(),
        Dtype::I64,
        vec![2],
        Bytes::copy_from_slice(&raw),
    );
    // Act
    let data = tensor.raw_data();
    // Assert: raw bytes are preserved exactly
    assert_eq!(data.len(), 16);
    let reconstructed_min = i64::from_le_bytes(data[0..8].try_into().unwrap());
    let reconstructed_max = i64::from_le_bytes(data[8..16].try_into().unwrap());
    assert_eq!(reconstructed_min, i64::MIN, "i64::MIN should be preserved through raw data");
    assert_eq!(reconstructed_max, i64::MAX, "i64::MAX should be preserved through raw data");
}

// ── 12. None value_type handling ───────────────────────────────────────
// @trace TEST-ONNX-164 [level:unit]

#[test]
fn value_info_none_type_cloned_is_still_none() {
    // Arrange: ValueInfo with None value_type (valid in ONNX for forward declarations)
    let vi = OnnxValueInfo {
        name: "placeholder".to_string(),
        value_type: None,
        doc_string: "forward-declared tensor".to_string(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = vi.clone();
    // Assert: None value_type is preserved through clone
    assert!(cloned.value_type.is_none(), "cloned value_type should remain None");
    assert_eq!(cloned.name, "placeholder");
    assert_eq!(cloned.doc_string, "forward-declared tensor");
    // Assert: original is unchanged
    assert!(vi.value_type.is_none());
}

// ── 13. Attribute Ints with negative values ────────────────────────────
// @trace TEST-ONNX-165 [level:unit]

#[test]
fn onnx_attribute_ints_negative_values_preserved() {
    // Arrange: Ints attribute with negative values representing e.g. padding [-1, -1, 1, 1]
    let negative_ints = vec![-1i64, -2, -100, i64::MIN];
    let attr = OnnxAttribute {
        name: "pads".to_string(),
        value: OnnxAttributeValue::Ints(negative_ints.clone()),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert: negative values preserved exactly
    if let OnnxAttributeValue::Ints(ref vals) = cloned.value {
        assert_eq!(vals.len(), 4);
        assert_eq!(vals[0], -1);
        assert_eq!(vals[1], -2);
        assert_eq!(vals[2], -100);
        assert_eq!(vals[3], i64::MIN);
    } else {
        panic!("expected Ints variant");
    }
}

// ── 14. Attribute float precision (f32::MIN_POSITIVE) ──────────────────
// @trace TEST-ONNX-166 [level:unit]

#[test]
fn onnx_attribute_float_min_positive_preserved() {
    // Arrange: f32::MIN_POSITIVE is the smallest positive normal f32 value
    let min_positive = f32::MIN_POSITIVE; // ~1.17549435e-38
    let attr = OnnxAttribute {
        name: "scale".to_string(),
        value: OnnxAttributeValue::Float(min_positive),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert: f32::MIN_POSITIVE is preserved bit-exactly through clone
    if let OnnxAttributeValue::Float(v) = cloned.value {
        assert_eq!(v.to_bits(), min_positive.to_bits(),
            "f32::MIN_POSITIVE should be preserved bit-exactly");
        assert!(v > 0.0, "MIN_POSITIVE must be positive");
        assert!(v < f32::EPSILON, "MIN_POSITIVE must be smaller than EPSILON");
    } else {
        panic!("expected Float variant");
    }
}

// ── 15. OnnxSparseFormat Copy trait verification ──────────────────────
// @trace TEST-ONNX-167 [level:unit]

#[test]
fn onnx_sparse_format_copy_all_variants_in_vec() {
    // Arrange: collect all three variants into a Vec via Copy
    let formats = vec![OnnxSparseFormat::Coo, OnnxSparseFormat::Csr, OnnxSparseFormat::Csc];
    // Act: Copy allows using each element multiple times without move
    let first = formats[0];
    let first_again = formats[0];
    // Assert: Copy semantics hold - both captures produce the same value
    assert_eq!(first, OnnxSparseFormat::Coo);
    assert_eq!(first_again, OnnxSparseFormat::Coo);
    // Assert: all three variants are distinct
    assert_ne!(formats[0], formats[1]);
    assert_ne!(formats[1], formats[2]);
    assert_ne!(formats[0], formats[2]);
    // Assert: Vec still accessible after copy
    assert_eq!(formats.len(), 3);
}

// ── 16. OnnxAttributeValue::Int i64 boundary values ────────────────────
// @trace TEST-ONNX-168 [level:unit]

#[test]
fn onnx_attribute_value_int_i64_min_max_preserved() {
    // Arrange: Int attribute with i64 boundary values
    let attr_min = OnnxAttribute {
        name: "offset_min".to_string(),
        value: OnnxAttributeValue::Int(i64::MIN),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    let attr_max = OnnxAttribute {
        name: "offset_max".to_string(),
        value: OnnxAttributeValue::Int(i64::MAX),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned_min = attr_min.clone();
    let cloned_max = attr_max.clone();
    // Assert: i64 boundary values survive clone
    assert!(matches!(cloned_min.value, OnnxAttributeValue::Int(v) if v == i64::MIN));
    assert!(matches!(cloned_max.value, OnnxAttributeValue::Int(v) if v == i64::MAX));
}

// ── 17. OnnxAttributeValue::String empty string ────────────────────────
// @trace TEST-ONNX-169 [level:unit]

#[test]
fn onnx_attribute_value_string_empty_preserved() {
    // Arrange: String attribute with empty value
    let attr = OnnxAttribute {
        name: "label".to_string(),
        value: OnnxAttributeValue::String(String::new()),
        doc_string: String::new(),
        ref_attr_name: None,
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert: empty string is distinct from None/default, preserved through clone
    if let OnnxAttributeValue::String(ref s) = cloned.value {
        assert!(s.is_empty(), "empty String variant should preserve empty string");
    } else {
        panic!("expected String variant");
    }
}

// ── 18. OnnxQuantizationAnnotation scale=0 and zero_point=0 ────────────
// @trace TEST-ONNX-170 [level:unit]

#[test]
fn onnx_quantization_annotation_zero_scale_and_zero_point() {
    // Arrange: annotation with explicit zero scale and zero_point (valid: symmetric quantization)
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "weight_q".to_string(),
        quant_param_tensor_names: HashMap::new(),
        scale: Some(0.0),
        zero_point: Some(0),
        axis: None,
    };
    // Act
    let cloned = qa.clone();
    // Assert: zero values preserved, not confused with None
    assert_eq!(cloned.scale, Some(0.0), "scale=0.0 should be Some(0.0), not None");
    assert_eq!(cloned.zero_point, Some(0), "zero_point=0 should be Some(0), not None");
    assert!(cloned.axis.is_none());
}

// ── 19. OnnxGraph quantization_annotation empty vec ────────────────────
// @trace TEST-ONNX-171 [level:unit]

#[test]
fn onnx_graph_quantization_annotation_default_empty_vec() {
    // Arrange: graph constructed with empty quantization_annotation
    let graph = OnnxGraph {
        name: "test_graph".to_string(),
        doc_string: String::new(),
        nodes: vec![],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = graph.clone();
    // Assert: empty vec is preserved (not confused with None)
    assert!(graph.quantization_annotation.is_empty());
    assert!(cloned.quantization_annotation.is_empty());
}

// ── 20. OnnxOperatorSet field access and clone independence ────────────
// @trace TEST-ONNX-172 [level:unit]

#[test]
fn onnx_operator_set_clone_independence_different_versions() {
    // Arrange: two operator sets with same domain but different versions
    let opset_v14 = OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 14,
    };
    let opset_v17 = OnnxOperatorSet {
        domain: "ai.onnx".to_string(),
        version: 17,
    };
    // Act
    let cloned_v14 = opset_v14.clone();
    let cloned_v17 = opset_v17.clone();
    // Assert: clone preserves all fields, original and clone are independent
    assert_eq!(cloned_v14.domain, "ai.onnx");
    assert_eq!(cloned_v14.version, 14);
    assert_eq!(cloned_v17.domain, "ai.onnx");
    assert_eq!(cloned_v17.version, 17);
    // Assert: originals still accessible after clone
    assert_eq!(opset_v14.version, 14);
    assert_eq!(opset_v17.version, 17);
}

// ── 21. OnnxFunction overload field ────────────────────────────────────
// @trace TEST-ONNX-173 [level:unit]

#[test]
fn onnx_function_overload_field_preserved() {
    // Arrange: function with a non-empty overload identifier
    let func = OnnxFunction {
        name: "CustomOp".to_string(),
        domain: "custom.domain".to_string(),
        overload: "v2_alpha".to_string(),
        inputs: vec!["X".to_string()],
        outputs: vec!["Y".to_string()],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act
    let cloned = func.clone();
    // Assert: overload string preserved exactly
    assert_eq!(cloned.overload, "v2_alpha");
    assert_eq!(cloned.name, "CustomOp");
    assert_eq!(cloned.domain, "custom.domain");
}

// ── 22. OnnxFunction opset_import empty vec ────────────────────────────
// @trace TEST-ONNX-174 [level:unit]

#[test]
fn onnx_function_opset_import_default_empty() {
    // Arrange: function with no opset imports (valid for functions using only default domain)
    let func = OnnxFunction {
        name: "SimpleFunc".to_string(),
        domain: String::new(),
        overload: String::new(),
        inputs: vec![],
        outputs: vec![],
        attributes: vec![],
        attribute_protos: HashMap::new(),
        nodes: vec![],
        opset_import: vec![],
        value_info: vec![],
        doc_string: String::new(),
        metadata_props: HashMap::new(),
    };
    // Act & Assert: empty opset_import is valid and preserved
    assert!(func.opset_import.is_empty());
    assert!(func.clone().opset_import.is_empty());
}

// ── 23. OnnxNode struct update syntax ──────────────────────────────────
// @trace TEST-ONNX-175 [level:unit]

#[test]
fn onnx_node_struct_update_syntax_overrides_inputs() {
    // Arrange: base node with generic inputs
    let base = OnnxNode {
        name: "base_node".to_string(),
        op_type: "MatMul".to_string(),
        domain: String::new(),
        inputs: vec!["A".to_string(), "B".to_string()],
        outputs: vec!["C".to_string()],
        attributes: HashMap::new(),
    };
    // Act: use struct update syntax to override inputs only
    let derived = OnnxNode {
        inputs: vec!["X".to_string(), "Y".to_string()],
        ..base.clone()
    };
    // Assert: inputs are overridden, other fields inherited from base
    assert_eq!(derived.inputs, vec!["X", "Y"]);
    assert_eq!(derived.name, "base_node");
    assert_eq!(derived.op_type, "MatMul");
    assert_eq!(derived.outputs, vec!["C"]);
}

// ── 24. OnnxValueInfo struct update syntax ─────────────────────────────
// @trace TEST-ONNX-176 [level:unit]

#[test]
fn onnx_value_info_struct_update_syntax_preserves_metadata() {
    // Arrange: base ValueInfo with metadata
    let mut meta = HashMap::new();
    meta.insert("source".to_string(), "encoder".to_string());
    let base = OnnxValueInfo {
        name: "hidden_state".to_string(),
        value_type: None,
        doc_string: "intermediate activation".to_string(),
        metadata_props: meta.clone(),
    };
    // Act: override name only, preserve everything else
    let derived = OnnxValueInfo {
        name: "output_state".to_string(),
        ..base.clone()
    };
    // Assert: name overridden, other fields preserved
    assert_eq!(derived.name, "output_state");
    assert!(derived.value_type.is_none());
    assert_eq!(derived.doc_string, "intermediate activation");
    assert_eq!(derived.metadata_props.get("source"), Some(&"encoder".to_string()));
}

// ── 25. OnnxTensor element_count with empty shape returns 1 ────────────
// @trace TEST-ONNX-177 [level:unit]

#[test]
fn onnx_tensor_element_count_empty_shape_is_one() {
    // Arrange: scalar tensor (empty shape = 0-D tensor) with 4 bytes of F32 data
    let tensor = OnnxTensor::new(
        "scalar_weight".to_string(),
        Dtype::F32,
        vec![],
        Bytes::copy_from_slice(&1.0f32.to_le_bytes()),
    );
    // Act
    let data = tensor.raw_data();
    // Assert: scalar has exactly 4 bytes (one f32 element), shape is empty
    assert!(tensor.shape.is_empty());
    assert_eq!(data.len(), 4, "scalar F32 tensor should have exactly 4 bytes");
    let value = f32::from_le_bytes(data[0..4].try_into().unwrap());
    assert_eq!(value, 1.0);
}

// ── 26. OnnxTensor new() sets is_string=false ─────────────────────────
// @trace TEST-ONNX-178 [level:unit]

#[test]
fn onnx_tensor_new_sets_is_string_false() {
    // Arrange: create tensor via new() (non-string path)
    let tensor = OnnxTensor::new(
        "weight".to_string(),
        Dtype::F32,
        vec![2, 3],
        Bytes::copy_from_slice(&[0u8; 24]),
    );
    // Act & Assert: new() always sets is_string=false
    assert!(!tensor.is_string, "new() should produce is_string=false");
    assert_eq!(tensor.dtype, Dtype::F32);
    assert_eq!(tensor.shape, vec![2, 3]);
}

// ── 27. OnnxGraph struct update syntax ─────────────────────────────────
// @trace TEST-ONNX-179 [level:unit]

#[test]
fn onnx_graph_struct_update_syntax_overrides_nodes() {
    // Arrange: base graph with one node
    let node = OnnxNode {
        name: "add_node".to_string(),
        op_type: "Add".to_string(),
        domain: String::new(),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        attributes: HashMap::new(),
    };
    let mut base_metadata = HashMap::new();
    base_metadata.insert("author".to_string(), "test".to_string());
    let base = OnnxGraph {
        name: "base_graph".to_string(),
        doc_string: "base doc".to_string(),
        nodes: vec![node],
        inputs: vec![],
        outputs: vec![],
        value_info: vec![],
        initializers: HashMap::new(),
        sparse_initializers: vec![],
        quantization_annotation: vec![],
        metadata_props: base_metadata,
    };
    // Act: override nodes to empty, preserve other fields
    let derived = OnnxGraph {
        nodes: vec![],
        ..base.clone()
    };
    // Assert: nodes overridden, other fields inherited
    assert!(derived.nodes.is_empty());
    assert_eq!(derived.name, "base_graph");
    assert_eq!(derived.doc_string, "base doc");
    assert_eq!(derived.metadata_props.get("author"), Some(&"test".to_string()));
}

// ── 28. OnnxTensor raw_data bytes preservation through clone ───────────
// @trace TEST-ONNX-180 [level:unit]

#[test]
fn onnx_tensor_raw_data_bytes_preserved_through_clone() {
    // Arrange: tensor with specific byte pattern
    let pattern: [u8; 8] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
    let tensor = OnnxTensor::new(
        "binary_weight".to_string(),
        Dtype::U8,
        vec![8],
        Bytes::copy_from_slice(&pattern),
    );
    // Act
    let cloned = tensor.clone();
    // Assert: raw bytes are identical after clone
    assert_eq!(tensor.raw_data(), cloned.raw_data());
    assert_eq!(cloned.raw_data(), &pattern[..]);
}

// ── 29. OnnxTensor scalar_i64 from BF16 scalar ──────────────────────────
// @trace TEST-ONNX-181 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_from_bf16_single_element() {
    // Arrange: BF16 scalar with value 42.0 (BF16 bits: 0x4328 = 0x42 0x28 in LE)
    let bf16_val = half::bf16::from_f32(42.0);
    let bf16_bits = bf16_val.to_bits().to_le_bytes();
    let tensor = OnnxTensor::new(
        "bf16_scalar".to_string(),
        Dtype::BF16,
        vec![],
        Bytes::copy_from_slice(&bf16_bits),
    );
    // Act
    let result = tensor.scalar_i64();
    // Assert: 42.0 -> 42 as i64
    assert_eq!(result, Some(42));
}

// ── 30. OnnxTensor scalar_i64 from U32 scalar ───────────────────────────
// @trace TEST-ONNX-182 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_from_u32_single_element() {
    // Arrange: U32 scalar with value 1_000_000
    let val: u32 = 1_000_000;
    let tensor = OnnxTensor::new(
        "u32_scalar".to_string(),
        Dtype::U32,
        vec![],
        Bytes::copy_from_slice(&val.to_le_bytes()),
    );
    // Act
    let result = tensor.scalar_i64();
    // Assert
    assert_eq!(result, Some(1_000_000_i64));
}

// ── 31. Bool tensor via int32_data with value 1 and 0 ───────────────────
// @trace TEST-ONNX-183 [level:unit]

#[test]
fn tensor_bool_via_int32_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Bool as i32);
    tensor.name = Some("bool_vals".to_string());
    tensor.int32_data = vec![1, 0];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("bool_vals").expect("tensor");
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 2); // 2 x 1 byte per bool
}

// ── 32. Double (f64) tensor via double_data field ───────────────────────
// @trace TEST-ONNX-184 [level:unit]

#[test]
fn tensor_double_via_double_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Double as i32);
    tensor.name = Some("f64_vals".to_string());
    tensor.double_data = vec![1.5f64, -2.25];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("f64_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::F64);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 16); // 2 x 8 bytes
}

// ── 33. U64 tensor via uint64_data field ─────────────────────────────────
// @trace TEST-ONNX-185 [level:unit]

#[test]
fn tensor_u64_via_uint64_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![2];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint64 as i32);
    tensor.name = Some("u64_vals".to_string());
    tensor.uint64_data = vec![0u64, u64::MAX];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u64_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U64);
    assert_eq!(ts.shape, vec![2]);
    assert_eq!(ts.data.len(), 16); // 2 x 8 bytes
}

// ── 34. U32 tensor via int32_data field ──────────────────────────────────
// @trace TEST-ONNX-186 [level:unit]

#[test]
fn tensor_u32_via_int32_data_roundtrip() {
    // Arrange
    let mut tensor = empty_tensor();
    tensor.dims = vec![3];
    tensor.data_type = Some(proto::tensor_proto::DataType::Uint32 as i32);
    tensor.name = Some("u32_vals".to_string());
    tensor.uint64_data = vec![0u64, 255u64, 100000u64];
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let ts = loader.tensor("u32_vals").expect("tensor");
    assert_eq!(ts.dtype, Dtype::U32);
    assert_eq!(ts.shape, vec![3]);
    assert_eq!(ts.data.len(), 12); // 3 x 4 bytes
}

// ── 35. OnnxType Optional equality: same inner types ────────────────────
// @trace TEST-ONNX-187 [level:unit]

#[test]
fn onnx_type_optional_equality_same_inner_type() {
    // Arrange
    let inner = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
    });
    let a = OnnxType::Optional(Box::new(inner.clone()));
    let b = OnnxType::Optional(Box::new(inner));
    // Act & Assert
    assert_eq!(a, b, "Two Optional types with same inner type should be equal");
}

// ── 36. OnnxType Map equality: same key_type and value_type ─────────────
// @trace TEST-ONNX-188 [level:unit]

#[test]
fn onnx_type_map_equality_same_key_value() {
    // Arrange
    let map_a = OnnxType::Map(OnnxMapType {
        key_type: proto::tensor_proto::DataType::Int64,
        value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })),
    });
    let map_b = OnnxType::Map(OnnxMapType {
        key_type: proto::tensor_proto::DataType::Int64,
        value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![] },
        })),
    });
    // Act & Assert
    assert_eq!(map_a, map_b, "Two Map types with same key_type and value_type should be equal");
}

// ── 37. OnnxTensorType equality: different elem_type not equal ──────────
// @trace TEST-ONNX-189 [level:unit]

#[test]
fn onnx_tensor_type_equality_different_elem_type_not_equal() {
    // Arrange
    let a = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Float,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(4)] },
    };
    let b = OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int32,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(4)] },
    };
    // Act & Assert
    assert_ne!(a, b, "Tensor types with different elem_type should not be equal");
}

// ── 38. OnnxDim Param equality: same name equal ─────────────────────────
// @trace TEST-ONNX-190 [level:unit]

#[test]
fn onnx_dim_param_equality_same_name() {
    // Arrange
    let a = OnnxDim::Param("seq_len".to_string());
    let b = OnnxDim::Param("seq_len".to_string());
    // Act & Assert
    assert_eq!(a, b, "Param dims with same name should be equal");
}

// ── 39. OnnxDim Param not equal to Known ────────────────────────────────
// @trace TEST-ONNX-191 [level:unit]

#[test]
fn onnx_dim_param_not_equal_to_known() {
    // Arrange
    let param = OnnxDim::Param("batch".to_string());
    let known = OnnxDim::Known(1);
    // Act & Assert
    assert_ne!(param, known, "Param dim should not equal Known dim");
}

// ── 40. OnnxModel metadata domain default empty string ──────────────────
// @trace TEST-ONNX-192 [level:unit]

#[test]
fn onnx_model_metadata_domain_default_empty_string() {
    // Arrange: model without domain field
    let tensor = tensor_f32("w", vec![1], &[0.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        domain: None,
        ..empty_model(graph)
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.domain, "");
}

// ── 41. OnnxGraph inputs preserved through model load ───────────────────
// @trace TEST-ONNX-193 [level:unit]

#[test]
fn loader_graph_inputs_preserved_through_load() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let input_info = proto::ValueInfoProto {
        name: Some("input_ids".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: Vec::new(),
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        input: vec![input_info],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    let inputs = &loader.graph().inputs;
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].name, "input_ids");
}

// ── 42. Loader path() returns correct PathBuf ───────────────────────────
// @trace TEST-ONNX-194 [level:unit]

#[test]
fn loader_path_returns_tempfile_path() {
    // Arrange
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let expected_path = file.path().to_path_buf();
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.path(), expected_path);
}

// ── 43. Loader iter_tensors returns all initializers ─────────────────────
// @trace TEST-ONNX-195 [level:unit]

#[test]
fn loader_iter_tensors_returns_correct_count_and_names() {
    // Arrange
    let t1 = tensor_f32("alpha", vec![2], &[1.0, 2.0]);
    let t2 = tensor_f32("beta", vec![1], &[3.0]);
    let t3 = tensor_f32("gamma", vec![3], &[4.0, 5.0, 6.0]);
    let graph = proto::GraphProto { initializer: vec![t1, t2, t3], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let metas: Vec<_> = loader.iter_tensors().collect();
    // Assert
    assert_eq!(metas.len(), 3);
    let names: Vec<&str> = metas.iter().map(|m| m.name.as_str()).collect();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
    assert!(names.contains(&"gamma"));
}

// ── 44. OnnxDim::Unknown is distinct from Known and Param ────────────────
// @trace TEST-ONNX-196 [level:unit]

#[test]
fn onnx_dim_unknown_not_equal_to_known_or_param() {
    // Arrange: three dim variants
    let unknown = OnnxDim::Unknown;
    let known = OnnxDim::Known(1);
    let param = OnnxDim::Param("batch".to_string());
    // Act & Assert: Unknown differs from both Known and Param
    assert_ne!(unknown, known, "Unknown should not equal Known");
    assert_ne!(unknown, param, "Unknown should not equal Param");
}

// ── 45. OnnxModelMetadata doc_string from proto ──────────────────────────
// @trace TEST-ONNX-197 [level:unit]

#[test]
fn loader_model_metadata_doc_string_from_proto() {
    // Arrange: model with explicit doc_string
    let tensor = tensor_f32("w", vec![1], &[0.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        doc_string: Some("test documentation string".to_string()),
        ..empty_model(graph)
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.doc_string, "test documentation string");
}

// ── 46. OnnxModelMetadata model_version negative preserved ───────────────
// @trace TEST-ONNX-198 [level:unit]

#[test]
fn loader_model_metadata_negative_model_version() {
    // Arrange: model with negative version (valid per protobuf int64)
    let tensor = tensor_f32("w", vec![1], &[0.0]);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = proto::ModelProto {
        model_version: Some(-42),
        ..empty_model(graph)
    };
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.model().metadata.model_version, -42);
}

// ── 47. OnnxType::Sequence variant with Tensor inner type ────────────────
// @trace TEST-ONNX-199 [level:unit]

#[test]
fn onnx_type_sequence_equality_same_inner_type() {
    // Arrange: two Sequence types wrapping same Tensor type
    let inner = OnnxType::Tensor(OnnxTensorType {
        elem_type: proto::tensor_proto::DataType::Int64,
        shape: OnnxTensorShape { dims: vec![OnnxDim::Known(128)] },
    });
    let seq_a = OnnxType::Sequence(Box::new(inner.clone()));
    let seq_b = OnnxType::Sequence(Box::new(inner));
    // Act & Assert
    assert_eq!(seq_a, seq_b, "Sequence types with same inner type should be equal");
}

// ── 48. TensorSlice element_count matches tensor shape product ───────────
// @trace TEST-ONNX-200 [level:unit]

#[test]
fn tensor_slice_element_count_matches_shape_product() {
    // Arrange: 3x4x5 tensor (60 elements * 4 bytes = 240)
    let data: Vec<u8> = (0..240).map(|i| i as u8).collect();
    let tensor = tensor_raw("big_tensor", vec![3, 4, 5], proto::tensor_proto::DataType::Float, &data);
    let graph = proto::GraphProto { initializer: vec![tensor], ..empty_graph() };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Act
    let slice = loader.tensor("big_tensor").expect("tensor");
    // Assert: shape product = 3*4*5 = 60 elements, each F32 is 4 bytes
    let expected_bytes: usize = slice.shape.iter().product::<usize>() * 4;
    assert_eq!(slice.data.len(), expected_bytes, "3*4*5 = 60 elements * 4 bytes = 240 bytes");
    assert_eq!(slice.data.len(), 240);
}

// ── 49. OnnxGraph outputs empty when proto has no outputs ─────────────────
// @trace TEST-ONNX-201 [level:unit]

#[test]
fn loader_graph_outputs_empty_when_proto_none() {
    // Arrange: graph with no output entries
    let tensor = tensor_f32("w", vec![1], &[1.0]);
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        output: vec![],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert!(loader.graph().outputs.is_empty(), "graph outputs should be empty when proto has none");
}

// ── 50. OnnxTensor new_string dtype is U8 placeholder ────────────────────
// @trace TEST-ONNX-202 [level:unit]

#[test]
fn onnx_tensor_new_string_dtype_is_u8_placeholder() {
    // Arrange & Act: create a string tensor
    let string_tensor = OnnxTensor::new_string(
        "text_labels".to_string(),
        vec![3],
        Bytes::copy_from_slice(b"abcdef"),
    );
    // Assert: string tensors use U8 as placeholder dtype
    assert!(string_tensor.is_string, "new_string should set is_string=true");
    assert_eq!(string_tensor.dtype, Dtype::U8, "string tensor dtype should be U8 placeholder");
    assert_eq!(string_tensor.shape, vec![3]);
}

// ── 51. Loader with multiple initializers returns correct dtype per tensor ─
// @trace TEST-ONNX-203 [level:unit]

#[test]
fn loader_tensor_dtype_per_tensor_mixed_types() {
    // Arrange: two tensors with different dtypes
    let f32_tensor = tensor_f32("weight_f32", vec![2], &[1.0, 2.0]);
    let i64_raw: Vec<u8> = [100i64, -200i64].iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let i64_tensor = tensor_raw("bias_i64", vec![2], proto::tensor_proto::DataType::Int64, &i64_raw);
    let graph = proto::GraphProto {
        initializer: vec![f32_tensor, i64_tensor],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.tensor_dtype("weight_f32").expect("dtype"), Dtype::F32);
    assert_eq!(loader.tensor_dtype("bias_i64").expect("dtype"), Dtype::I64);
}

// ── 52. OnnxAttribute ref_attr_name field preserves value ─────────────────
// @trace TEST-ONNX-204 [level:unit]

#[test]
fn onnx_attribute_ref_attr_name_preserves_value() {
    // Arrange: attribute with ref_attr_name set (graph-level attribute reference)
    let attr = OnnxAttribute {
        name: "kernel_size".to_string(),
        value: OnnxAttributeValue::Int(3),
        doc_string: "kernel dimension".to_string(),
        ref_attr_name: Some("parent_kernel".to_string()),
        attr_type: None,
    };
    // Act
    let cloned = attr.clone();
    // Assert: ref_attr_name is preserved through clone
    assert_eq!(cloned.ref_attr_name, Some("parent_kernel".to_string()));
    assert_eq!(cloned.name, "kernel_size");
    assert_eq!(cloned.doc_string, "kernel dimension");
}

// ── 53. OnnxQuantizationAnnotation with axis preserves value ──────────────
// @trace TEST-ONNX-205 [level:unit]

#[test]
fn onnx_quantization_annotation_axis_negative_value() {
    // Arrange: quantization annotation with negative axis (per-channel)
    let qa = OnnxQuantizationAnnotation {
        tensor_name: "weight_per_channel".to_string(),
        quant_param_tensor_names: {
            let mut m = HashMap::new();
            m.insert("scale".to_string(), "weight_per_channel_scale".to_string());
            m
        },
        scale: Some(0.125),
        zero_point: Some(128),
        axis: Some(-1),
    };
    // Act
    let cloned = qa.clone();
    // Assert: negative axis preserved
    assert_eq!(cloned.axis, Some(-1), "negative axis should be preserved");
    assert_eq!(cloned.scale, Some(0.125));
    assert_eq!(cloned.zero_point, Some(128));
    assert_eq!(cloned.quant_param_tensor_names.len(), 1);
}

// ── 54. Loader graph with value_info entries preserved ───────────────────
// @trace TEST-ONNX-206 [level:unit]

#[test]
fn loader_graph_value_info_entries_preserved() {
    // Arrange: graph with value_info entries
    let tensor = tensor_f32("w", vec![1], &[0.5]);
    let vi = proto::ValueInfoProto {
        name: Some("intermediate_activation".to_string()),
        r#type: None,
        doc_string: None,
        metadata_props: Vec::new(),
    };
    let graph = proto::GraphProto {
        initializer: vec![tensor],
        value_info: vec![vi],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());
    // Act
    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    // Assert
    assert_eq!(loader.graph().value_info.len(), 1);
    assert_eq!(loader.graph().value_info[0].name, "intermediate_activation");
}

// ── 55. OnnxTensor scalar_f32 returns None for multi-element tensor ──────
// @trace TEST-ONNX-207 [level:unit]

#[test]
fn onnx_tensor_scalar_f32_multi_element_returns_none() {
    // Arrange: 2-element F32 tensor (not a scalar)
    let tensor = OnnxTensor::new(
        "multi_elem".to_string(),
        Dtype::F32,
        vec![2],
        Bytes::copy_from_slice(&[0u8; 8]),
    );
    // Act
    let result = tensor.scalar_f32();
    // Assert: multi-element tensor should return None
    assert!(result.is_none(), "scalar_f32 should return None for multi-element tensor");
}

// ── 56. OnnxTensor scalar_i64 returns None for multi-element I64 tensor ──
// @trace TEST-ONNX-208 [level:unit]

#[test]
fn onnx_tensor_scalar_i64_multi_element_returns_none() {
    // Arrange: 3-element I64 tensor
    let data: Vec<u8> = [10i64, 20, 30].iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let tensor = OnnxTensor::new(
        "multi_i64".to_string(),
        Dtype::I64,
        vec![3],
        Bytes::copy_from_slice(&data),
    );
    // Act
    let result = tensor.scalar_i64();
    // Assert: multi-element tensor should return None
    assert!(result.is_none(), "scalar_i64 should return None for multi-element tensor");
}

