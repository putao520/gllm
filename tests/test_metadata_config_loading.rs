use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;

use gllm::loader::onnx::proto;
use gllm::loader::Loader;
use gllm::manifest::{
    ModelArchitecture, ModelKind, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};
use gllm::model_config::ModelConfig;
use gllm_kernels::cpu_backend::CpuBackend;
use prost::bytes::Bytes;
use prost::Message;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use tempfile::TempDir;

fn test_manifest() -> ModelManifest {
    ModelManifest {
        model_id: Cow::Borrowed("test/model"),
        file_map: EMPTY_FILE_MAP,
        arch: ModelArchitecture::Llama4,
        tensor_rules: TensorNamingRule::Llama4,
        kind: ModelKind::Chat,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: HashMap::new(),
    }
}

#[test]
fn safetensors_gllm_config_metadata_is_used() {
    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("model.safetensors");

    let data = vec![0u8; 8];
    let view = TensorView::new(Dtype::F16, vec![2, 2], &data).expect("tensor view");

    let mut metadata = HashMap::new();
    metadata.insert(
        "_gllm_config".to_string(),
        serde_json::json!({
            "architecture": "LlamaForCausalLM",
            "hidden_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "head_dim": 32,
            "num_experts": 16,
            "expert_intermediate_size": 2048
        })
        .to_string(),
    );
    serialize_to_file(
        vec![("model.embed_tokens.weight", view)],
        &Some(metadata),
        &path,
    )
    .expect("write safetensors");

    let manifest = test_manifest();
    let mut loader = Loader::new(manifest.clone())
        .with_weights(vec![path])
        .load()
        .expect("load");

    let config = ModelConfig::from_loader(&manifest, &mut loader).expect("model config");
    assert_eq!(config.hidden_size, 256);
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.num_attention_heads, 8);
    assert_eq!(config.num_key_value_heads, 4);
    assert_eq!(config.head_dim, 32);
    assert_eq!(config.dtype_size, 2);
    assert_eq!(config.num_experts, Some(16));
    assert_eq!(config.expert_intermediate_size, Some(2048));
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
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    fs::write(path, bytes).expect("write model");
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
fn onnx_dtype_metadata_is_used_for_model_config_dtype_size() {
    let dir = TempDir::new().expect("temp dir");
    let onnx_path = dir.path().join("model.onnx");
    let config_path = dir.path().join("config.json");

    let graph = proto::GraphProto {
        initializer: vec![tensor_raw(
            "linear.weight",
            vec![2],
            proto::tensor_proto::DataType::Float16,
            &[0, 0, 0, 0],
        )],
        ..empty_graph()
    };
    write_model(empty_model(graph), &onnx_path);

    fs::write(
        &config_path,
        serde_json::json!({
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 6,
            "vocab_size": 32000,
            "max_position_embeddings": 1024,
            "head_dim": 16,
            "rope_theta": 10000.0
        })
        .to_string(),
    )
    .expect("write config");

    let manifest = test_manifest();
    let mut loader = Loader::new(manifest.clone())
        .with_weights(vec![onnx_path])
        .with_config(config_path)
        .load()
        .expect("load");
    // Explicitly detect format since we are manually constructing
    // The builder `with_weights` calls `detect_format`, so we should be good.

    let config = ModelConfig::from_loader(&manifest, &mut loader).expect("model config");
    assert_eq!(config.dtype_size, 2);
}

#[test]
fn onnx_weights_are_uploadable_for_reranker() {
    let dir = TempDir::new().expect("temp dir");
    let onnx_path = dir.path().join("model.onnx");

    let emb_values = [1.0f32, 2.0, 3.0, 4.0];
    let mut emb_raw = Vec::with_capacity(emb_values.len() * 4);
    for value in emb_values {
        emb_raw.extend_from_slice(&value.to_le_bytes());
    }

    let cls_values = [0.5f32, -0.25f32];
    let mut cls_raw = Vec::with_capacity(cls_values.len() * 4);
    for value in cls_values {
        cls_raw.extend_from_slice(&value.to_le_bytes());
    }

    let graph = proto::GraphProto {
        initializer: vec![
            tensor_raw(
                "roberta.embeddings.word_embeddings.weight",
                vec![2, 2],
                proto::tensor_proto::DataType::Float,
                &emb_raw,
            ),
            tensor_raw(
                "classifier.weight",
                vec![1, 2],
                proto::tensor_proto::DataType::Float,
                &cls_raw,
            ),
        ],
        ..empty_graph()
    };
    write_model(empty_model(graph), &onnx_path);

    let manifest = ModelManifest {
        model_id: Cow::Borrowed("test/reranker"),
        file_map: EMPTY_FILE_MAP,
        arch: ModelArchitecture::XlmR,
        tensor_rules: TensorNamingRule::XlmR,
        kind: ModelKind::Reranker,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: HashMap::new(),
    };

    let mut loader = Loader::new(manifest)
        .with_weights(vec![onnx_path])
        .load()
        .expect("load");
    let backend = CpuBackend::<f32>::new();
    let weights = loader
        .upload_weights(&backend)
        .expect("upload onnx weights");

    assert!(
        weights
            .tensor("roberta.embeddings.word_embeddings.weight")
            .is_some(),
        "embedding tensor should be uploaded"
    );
    assert!(weights.tensor("classifier.weight").is_some());
    assert_eq!(
        weights
            .meta
            .get("classifier.weight")
            .expect("classifier meta")
            .shape,
        vec![1, 2]
    );
}
