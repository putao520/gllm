use super::{external_data_locations, proto, OnnxLoader};
use prost::bytes::Bytes;
use prost::Message;
use safetensors::Dtype;
use tempfile::{NamedTempFile, TempDir};

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
