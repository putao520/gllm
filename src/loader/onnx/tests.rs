use super::{external_data_locations, proto, OnnxLoader};
use prost::bytes::Bytes;
use prost::Message;
use safetensors::Dtype;
use tempfile::{NamedTempFile, TempDir};

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
    let values = slice.as_f32().expect("f32");
    assert_eq!(values.as_ref(), &[1.0, 2.0, 3.0, 4.0]);
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
    let values = slice.as_f32().expect("f32");
    assert_eq!(values.as_ref(), &[1.0, 2.0]);
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
fn match_attention_swiglu_rope() {
    let attention = proto::NodeProto {
        op_type: Some("Attention".to_string()),
        input: vec!["q".to_string(), "k".to_string(), "v".to_string()],
        output: vec!["attn_out".to_string()],
        ..empty_node()
    };
    let gate = proto::NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "gate_w".to_string()],
        output: vec!["gate".to_string()],
        ..empty_node()
    };
    let up = proto::NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["x".to_string(), "up_w".to_string()],
        output: vec!["up".to_string()],
        ..empty_node()
    };
    let silu = proto::NodeProto {
        op_type: Some("Silu".to_string()),
        input: vec!["gate".to_string()],
        output: vec!["gate_act".to_string()],
        ..empty_node()
    };
    let mul = proto::NodeProto {
        op_type: Some("Mul".to_string()),
        input: vec!["gate_act".to_string(), "up".to_string()],
        output: vec!["mlp_out".to_string()],
        ..empty_node()
    };
    let rope = proto::NodeProto {
        op_type: Some("RotaryEmbedding".to_string()),
        input: vec!["rope_in".to_string()],
        output: vec!["rope_out".to_string()],
        ..empty_node()
    };
    let graph = proto::GraphProto {
        node: vec![attention, gate, up, silu, mul, rope],
        ..empty_graph()
    };
    let model = empty_model(graph);
    let file = NamedTempFile::new().expect("tempfile");
    write_model(model, file.path());

    let loader = OnnxLoader::from_path(file.path()).expect("loader");
    let fused = loader.fused_graph();
    let mut has_attention = false;
    let mut has_swiglu = false;
    let mut has_rope = false;
    for op in &fused.ops {
        match op.kind {
            super::FusedKernel::FlashAttention(_) => has_attention = true,
            super::FusedKernel::SwiGlu(_) => has_swiglu = true,
            super::FusedKernel::Rope(_) | super::FusedKernel::FusedQkvRope(_) => has_rope = true,
            _ => {}
        }
    }
    assert!(has_attention);
    assert!(has_swiglu);
    assert!(has_rope);
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
