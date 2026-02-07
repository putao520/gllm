use std::borrow::Cow;
use std::fs;

use gllm::loader::gguf::{
    tensor_nbytes, GgmlDType, GgufValueType, GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};
use gllm::loader::Loader;
use gllm::manifest::{
    ModelArchitecture, ModelKind, ModelManifest, TensorNamingRule, EMPTY_FILE_MAP,
};
use gllm::model_config::{ModelConfig, ModelConfigError};
use tempfile::TempDir;

#[derive(Debug, Clone)]
enum MetaValue {
    Str(String),
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
}

#[derive(Debug, Clone)]
struct MetaEntry {
    key: String,
    value: MetaValue,
}

#[derive(Debug, Clone)]
struct TensorEntry {
    name: String,
    dtype: GgmlDType,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_string(out: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    write_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn write_meta(out: &mut Vec<u8>, entry: &MetaEntry) {
    write_string(out, &entry.key);
    match &entry.value {
        MetaValue::Str(v) => {
            write_u32(out, GgufValueType::String as u32);
            write_string(out, v);
        }
        MetaValue::U32(v) => {
            write_u32(out, GgufValueType::Uint32 as u32);
            write_u32(out, *v);
        }
        MetaValue::U64(v) => {
            write_u32(out, GgufValueType::Uint64 as u32);
            write_u64(out, *v);
        }
        MetaValue::F32(v) => {
            write_u32(out, GgufValueType::Float32 as u32);
            write_u32(out, v.to_bits());
        }
        MetaValue::Bool(v) => {
            write_u32(out, GgufValueType::Bool as u32);
            out.push(u8::from(*v));
        }
    }
}

fn write_tensor_info(out: &mut Vec<u8>, tensor: &TensorEntry, offset: u64) {
    write_string(out, &tensor.name);
    write_u32(out, tensor.shape.len() as u32);
    for &dim in &tensor.shape {
        write_u64(out, dim);
    }
    write_u32(out, tensor.dtype as u32);
    write_u64(out, offset);
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn build_gguf(metadata: Vec<MetaEntry>, tensors: Vec<TensorEntry>, alignment: usize) -> Vec<u8> {
    let mut out = Vec::new();
    write_u32(&mut out, GGUF_MAGIC);
    write_u32(&mut out, GGUF_SUPPORTED_VERSION);
    write_u64(&mut out, tensors.len() as u64);
    write_u64(&mut out, metadata.len() as u64);

    for entry in &metadata {
        write_meta(&mut out, entry);
    }

    let mut running_offset = 0u64;
    for tensor in &tensors {
        write_tensor_info(&mut out, tensor, running_offset);
        running_offset += tensor.data.len() as u64;
    }

    let data_start = align_up(out.len(), alignment);
    out.resize(data_start, 0);
    for tensor in tensors {
        out.extend_from_slice(&tensor.data);
    }
    out
}

fn write_temp_gguf(bytes: &[u8]) -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("create temp gguf dir");
    let path = dir.path().join("model.gguf");
    fs::write(&path, bytes).expect("write gguf");
    (dir, path)
}

fn make_tensor(name: &str, dtype: GgmlDType, shape: Vec<u64>) -> TensorEntry {
    let size = tensor_nbytes(dtype, &shape).expect("valid tensor shape");
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push(((i * 13) % 251) as u8);
    }
    TensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        data,
    }
}

fn make_manifest() -> ModelManifest {
    ModelManifest {
        model_id: Cow::Borrowed("test/gguf"),
        file_map: EMPTY_FILE_MAP,
        arch: ModelArchitecture::Llama4,
        tensor_rules: TensorNamingRule::Llama4,
        kind: ModelKind::Chat,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
    }
}

fn base_metadata() -> Vec<MetaEntry> {
    vec![
        MetaEntry {
            key: "general.alignment".to_string(),
            value: MetaValue::U32(32),
        },
        MetaEntry {
            key: "general.architecture".to_string(),
            value: MetaValue::Str("llama".to_string()),
        },
    ]
}

fn full_llama_config_metadata() -> Vec<MetaEntry> {
    let mut metadata = base_metadata();
    metadata.extend([
        MetaEntry {
            key: "llama.vocab_size".to_string(),
            value: MetaValue::U64(32_000),
        },
        MetaEntry {
            key: "llama.embedding_length".to_string(),
            value: MetaValue::U64(4096),
        },
        MetaEntry {
            key: "llama.block_count".to_string(),
            value: MetaValue::U64(32),
        },
        MetaEntry {
            key: "llama.attention.head_count".to_string(),
            value: MetaValue::U64(32),
        },
        MetaEntry {
            key: "llama.attention.head_count_kv".to_string(),
            value: MetaValue::U64(8),
        },
        MetaEntry {
            key: "llama.context_length".to_string(),
            value: MetaValue::U64(8192),
        },
        MetaEntry {
            key: "llama.rope.dimension_count".to_string(),
            value: MetaValue::U64(128),
        },
        MetaEntry {
            key: "llama.rope.freq_base".to_string(),
            value: MetaValue::F32(500_000.0),
        },
        MetaEntry {
            key: "tokenizer.ggml.bos_token_id".to_string(),
            value: MetaValue::U32(1),
        },
        MetaEntry {
            key: "tokenizer.ggml.eos_token_id".to_string(),
            value: MetaValue::U32(2),
        },
        MetaEntry {
            key: "tokenizer.ggml.add_bos_token".to_string(),
            value: MetaValue::Bool(true),
        },
        MetaEntry {
            key: "tokenizer.ggml.add_eos_token".to_string(),
            value: MetaValue::Bool(false),
        },
    ]);
    metadata
}

#[test]
fn model_config_reads_from_gguf_metadata_without_config_json() {
    let metadata = full_llama_config_metadata();
    let tensors = vec![make_tensor("token_embd.weight", GgmlDType::F16, vec![8])];
    let (_gguf_dir, gguf_path) = write_temp_gguf(&build_gguf(metadata, tensors, 32));

    let manifest = make_manifest();
    let mut loader = Loader::from_local_files_with_manifest(
        "test/gguf",
        vec![gguf_path],
        vec![],
        Some(&manifest),
    )
    .expect("loader");

    let config = ModelConfig::from_loader(&manifest, &mut loader).expect("model config");

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.vocab_size, 32_000);
    assert_eq!(config.max_position_embeddings, 8192);
    assert_eq!(config.rope_theta, 500_000.0);
    assert_eq!(config.head_dim, 128);
    assert_eq!(config.dtype_size, 2);
    assert_eq!(config.bos_token_id, Some(1));
    assert_eq!(config.eos_token_id, Some(2));
}

#[test]
fn model_config_falls_back_to_config_json_when_gguf_metadata_is_incomplete() {
    let mut metadata = full_llama_config_metadata();
    metadata.retain(|entry| entry.key != "llama.attention.head_count_kv");
    let tensors = vec![make_tensor("token_embd.weight", GgmlDType::F16, vec![8])];
    let (_gguf_dir, gguf_path) = write_temp_gguf(&build_gguf(metadata, tensors, 32));

    let dir = TempDir::new().expect("temp dir");
    let config_path = dir.path().join("config.json");
    let config_json = r#"{
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_hidden_layers": 24,
        "vocab_size": 30000,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "head_dim": 128,
        "torch_dtype": "float16",
        "bos_token_id": 11,
        "eos_token_id": 12
    }"#;
    fs::write(&config_path, config_json).expect("write config.json");

    let manifest = make_manifest();
    let mut loader = Loader::from_local_files_with_manifest(
        "test/gguf",
        vec![gguf_path],
        vec![config_path],
        Some(&manifest),
    )
    .expect("loader");

    let config = ModelConfig::from_loader(&manifest, &mut loader).expect("model config");

    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_hidden_layers, 24);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_key_value_heads, 16);
    assert_eq!(config.vocab_size, 30_000);
    assert_eq!(config.max_position_embeddings, 4096);
    assert_eq!(config.bos_token_id, Some(11));
    assert_eq!(config.eos_token_id, Some(12));
}

#[test]
fn model_config_prefers_gguf_metadata_over_config_json_when_both_exist() {
    let metadata = full_llama_config_metadata();
    let tensors = vec![make_tensor("token_embd.weight", GgmlDType::F16, vec![8])];
    let (_gguf_dir, gguf_path) = write_temp_gguf(&build_gguf(metadata, tensors, 32));

    let dir = TempDir::new().expect("temp dir");
    let config_path = dir.path().join("config.json");
    let config_json = r#"{
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_hidden_layers": 24,
        "vocab_size": 30000,
        "max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "head_dim": 128,
        "torch_dtype": "float16",
        "bos_token_id": 11,
        "eos_token_id": 12
    }"#;
    fs::write(&config_path, config_json).expect("write config.json");

    let manifest = make_manifest();
    let mut loader = Loader::from_local_files_with_manifest(
        "test/gguf",
        vec![gguf_path],
        vec![config_path],
        Some(&manifest),
    )
    .expect("loader");

    let config = ModelConfig::from_loader(&manifest, &mut loader).expect("model config");

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_hidden_layers, 32);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, 8);
    assert_eq!(config.vocab_size, 32_000);
    assert_eq!(config.max_position_embeddings, 8192);
    assert_eq!(config.bos_token_id, Some(1));
    assert_eq!(config.eos_token_id, Some(2));
}

#[test]
fn model_config_returns_error_when_gguf_and_config_json_are_both_missing() {
    let metadata = base_metadata();
    let tensors = vec![make_tensor("token_embd.weight", GgmlDType::F16, vec![8])];
    let (_gguf_dir, gguf_path) = write_temp_gguf(&build_gguf(metadata, tensors, 32));

    let manifest = make_manifest();
    let mut loader = Loader::from_local_files_with_manifest(
        "test/gguf",
        vec![gguf_path],
        vec![],
        Some(&manifest),
    )
    .expect("loader");

    let err = ModelConfig::from_loader(&manifest, &mut loader).expect_err("missing config");
    assert!(matches!(err, ModelConfigError::MissingConfigAndMetadata(_)));
}
