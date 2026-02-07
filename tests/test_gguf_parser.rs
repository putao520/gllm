use std::collections::BTreeSet;
use std::fs;

use gllm::loader::adapter::GgufAdapter;
use gllm::loader::gguf::{
    tensor_nbytes, GgmlDType, GgufError, GgufReader, GgufValueType, GGUF_MAGIC,
    GGUF_SUPPORTED_VERSION,
};
use gllm_kernels::{DType, PackedBits};
use tempfile::NamedTempFile;

#[derive(Debug, Clone)]
enum MetaValue {
    Str(String),
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    ArrayString(Vec<String>),
    ArrayF32(Vec<f32>),
    ArrayU32(Vec<u32>),
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
        MetaValue::ArrayString(values) => {
            write_u32(out, GgufValueType::Array as u32);
            write_u32(out, GgufValueType::String as u32);
            write_u64(out, values.len() as u64);
            for value in values {
                write_string(out, value);
            }
        }
        MetaValue::ArrayF32(values) => {
            write_u32(out, GgufValueType::Array as u32);
            write_u32(out, GgufValueType::Float32 as u32);
            write_u64(out, values.len() as u64);
            for value in values {
                write_u32(out, value.to_bits());
            }
        }
        MetaValue::ArrayU32(values) => {
            write_u32(out, GgufValueType::Array as u32);
            write_u32(out, GgufValueType::Uint32 as u32);
            write_u64(out, values.len() as u64);
            for value in values {
                write_u32(out, *value);
            }
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

fn write_temp_gguf(bytes: &[u8]) -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp gguf");
    fs::write(file.path(), bytes).expect("write gguf");
    file
}

fn make_metadata(include_architecture: bool) -> Vec<MetaEntry> {
    let mut metadata = vec![MetaEntry {
        key: "general.alignment".to_string(),
        value: MetaValue::U32(32),
    }];

    if include_architecture {
        metadata.push(MetaEntry {
            key: "general.architecture".to_string(),
            value: MetaValue::Str("llama".to_string()),
        });
    }

    metadata
}

fn make_tensor(name: &str, dtype: GgmlDType, shape: Vec<u64>) -> TensorEntry {
    let size = tensor_nbytes(dtype, &shape).expect("valid tensor shape");
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        data.push(((i * 17) % 251) as u8);
    }

    TensorEntry {
        name: name.to_string(),
        dtype,
        shape,
        data,
    }
}

#[test]
fn test_gguf_001_header_parse() {
    let metadata = make_metadata(true);
    let tensors = vec![make_tensor("weight", GgmlDType::F32, vec![4])];
    let file = write_temp_gguf(&build_gguf(metadata, tensors, 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");

    assert_eq!(reader.version(), 3);
    assert_eq!(reader.tensor_count(), 1);
    assert_eq!(reader.kv_count(), 2);
}

#[test]
fn test_gguf_002_array_string_parse_49152_tokens() {
    let mut metadata = make_metadata(true);
    let mut tokens = Vec::with_capacity(49_152);
    tokens.push("<unk>".to_string());
    tokens.push("<s>".to_string());
    for i in 2..49_152 {
        tokens.push(format!("tok_{i}"));
    }
    metadata.push(MetaEntry {
        key: "tokenizer.ggml.tokens".to_string(),
        value: MetaValue::ArrayString(tokens),
    });

    let file = write_temp_gguf(&build_gguf(metadata, Vec::new(), 32));
    let reader = GgufReader::open(file.path()).expect("open gguf");

    let parsed = reader.tokenizer_tokens().expect("parse tokenizer tokens");
    assert_eq!(parsed.len(), 49_152);
    assert_eq!(parsed[0], "<unk>");
    assert_eq!(parsed[1], "<s>");
}

#[test]
fn test_gguf_003_tensor_info_parse() {
    let metadata = make_metadata(true);
    let t0 = make_tensor("token_embd.weight", GgmlDType::F32, vec![4]);
    let t1 = make_tensor("blk.0.attn_q.weight", GgmlDType::Q4_0, vec![32]);
    let file = write_temp_gguf(&build_gguf(metadata, vec![t0, t1], 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");
    let info0 = reader
        .tensor_info("token_embd.weight")
        .expect("tensor info token_embd.weight");
    let info1 = reader
        .tensor_info("blk.0.attn_q.weight")
        .expect("tensor info blk.0.attn_q.weight");

    assert_eq!(info0.dtype, GgmlDType::F32);
    assert_eq!(info0.shape, vec![4]);
    assert_eq!(info0.size, 16);

    assert_eq!(info1.dtype, GgmlDType::Q4_0);
    assert_eq!(info1.shape, vec![32]);
    assert_eq!(info1.size, 18);
}

#[test]
fn test_gguf_004_omega1_architecture_from_metadata() {
    let metadata = make_metadata(true);
    let file = write_temp_gguf(&build_gguf(metadata, Vec::new(), 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");
    assert_eq!(reader.architecture().expect("architecture"), "llama");

    let metadata_missing_arch = make_metadata(false);
    let file_missing = write_temp_gguf(&build_gguf(metadata_missing_arch, Vec::new(), 32));
    let reader_missing = GgufReader::open(file_missing.path()).expect("open gguf");
    let err = reader_missing
        .architecture()
        .expect_err("missing architecture should fail");

    match err {
        GgufError::MissingMetadata(key) => assert_eq!(key, "general.architecture"),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_gguf_005_quantized_type_recognition() {
    let metadata = make_metadata(true);

    let tensors: Vec<TensorEntry> = GgmlDType::all()
        .iter()
        .enumerate()
        .map(|(idx, &dtype)| {
            let shape = vec![dtype.block_size() as u64];
            make_tensor(&format!("tensor_{idx}"), dtype, shape)
        })
        .collect();

    let file = write_temp_gguf(&build_gguf(metadata, tensors, 32));
    let reader = GgufReader::open(file.path()).expect("open gguf");

    for (idx, &dtype) in GgmlDType::all().iter().enumerate() {
        let info = reader
            .tensor_info(&format!("tensor_{idx}"))
            .expect("tensor info exists");
        assert_eq!(info.dtype, dtype);
    }

    let quantized: BTreeSet<String> = reader.quantization_types().iter().cloned().collect();
    assert!(quantized.contains("Q4_0"));
    assert!(quantized.contains("Q8_0"));
    assert!(!quantized.contains("F32"));
}

#[test]
fn test_gguf_006_tensorslice_zero_copy() {
    let metadata = make_metadata(true);
    let tensor = make_tensor("weight", GgmlDType::F32, vec![4]);
    let expected = tensor.data.clone();
    let file = write_temp_gguf(&build_gguf(metadata, vec![tensor], 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");
    let a = reader.tensor("weight").expect("tensor a");
    let b = reader.tensor("weight").expect("tensor b");

    assert_eq!(a.as_bytes(), expected.as_slice());
    assert_eq!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());
}

#[test]
fn test_gguf_007_invalid_magic_detection() {
    let mut bytes = Vec::new();
    write_u32(&mut bytes, 0x1234_5678);
    write_u32(&mut bytes, 3);
    write_u64(&mut bytes, 0);
    write_u64(&mut bytes, 0);

    let file = write_temp_gguf(&bytes);
    let err = GgufReader::open(file.path()).expect_err("invalid magic should fail");

    match err {
        GgufError::InvalidMagic(magic) => assert_eq!(magic, 0x1234_5678),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_gguf_008_missing_metadata_handling() {
    let metadata = make_metadata(false);
    let file = write_temp_gguf(&build_gguf(metadata, Vec::new(), 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");
    assert!(reader.get_metadata_u64("nonexistent.key").is_none());

    let err = reader
        .architecture()
        .expect_err("missing architecture should fail");
    match err {
        GgufError::MissingMetadata(key) => assert_eq!(key, "general.architecture"),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_gguf_009_tensor_boundary_check() {
    let metadata = make_metadata(true);
    let broken = TensorEntry {
        name: "broken.weight".to_string(),
        dtype: GgmlDType::F32,
        shape: vec![4], // requires 16 bytes
        data: vec![0u8; 8],
    };

    let file = write_temp_gguf(&build_gguf(metadata, vec![broken], 32));
    let err = GgufReader::open(file.path()).expect_err("tensor bounds should fail");
    assert!(matches!(err, GgufError::TensorOutOfBounds(_)));
}

#[test]
fn test_gguf_010_quantized_type_mapping() {
    let metadata = make_metadata(true);
    let tensors = vec![
        make_tensor("q4", GgmlDType::Q4_0, vec![32]),
        make_tensor("f32", GgmlDType::F32, vec![4]),
        make_tensor("q5", GgmlDType::Q5_0, vec![32]),
    ];

    let file = write_temp_gguf(&build_gguf(metadata, tensors, 32));
    let reader = GgufReader::open(file.path()).expect("open gguf");
    let adapter = GgufAdapter::new(reader);

    let q4 = adapter.tensor_for_kernel("q4").expect("map q4");
    assert!(matches!(q4.dtype, DType::PackedU8(PackedBits::Int4)));

    let f32 = adapter.tensor_for_kernel("f32").expect("map f32");
    assert!(matches!(f32.dtype, DType::F32));

    let err = adapter
        .tensor_for_kernel("q5")
        .expect_err("q5_0 mapping should fail with current kernels dtypes");
    assert!(matches!(err, GgufError::UnsupportedType(GgmlDType::Q5_0)));
}

#[test]
fn test_gguf_011_generic_constraint_verification() {
    let metadata = make_metadata(true);
    let tensor = make_tensor("q8", GgmlDType::Q8_0, vec![32]);
    let file = write_temp_gguf(&build_gguf(metadata, vec![tensor], 32));

    let reader = GgufReader::open(file.path()).expect("open gguf");
    let slice = reader.tensor("q8").expect("tensor slice");

    assert_eq!(slice.dtype(), GgmlDType::Q8_0);
    assert!(!slice.as_bytes().is_empty());
    let bytes_len = slice.as_bytes().len();
    drop(slice);

    let adapter = GgufAdapter::new(reader);
    let kernel = adapter.tensor_for_kernel("q8").expect("tensor for kernel");

    assert!(matches!(kernel.dtype, DType::U8));
    assert_eq!(kernel.data.len(), bytes_len);
}

#[test]
fn test_gguf_012_metadata_helper_accessors() {
    let mut metadata = make_metadata(true);
    metadata.extend([
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
            key: "general.file_type".to_string(),
            value: MetaValue::U32(7),
        },
        MetaEntry {
            key: "llama.rope.scale".to_string(),
            value: MetaValue::F32(8.0),
        },
        MetaEntry {
            key: "llama.rope.scaling.type".to_string(),
            value: MetaValue::Str("yarn".to_string()),
        },
        MetaEntry {
            key: "llama.rope.scaling.factor".to_string(),
            value: MetaValue::F32(8.0),
        },
        MetaEntry {
            key: "llama.rope.scaling.factors".to_string(),
            value: MetaValue::ArrayF32(vec![8.0, 4.0]),
        },
        MetaEntry {
            key: "llama.rope.ext_factor".to_string(),
            value: MetaValue::F32(1.5),
        },
        MetaEntry {
            key: "llama.rope.attn_factor".to_string(),
            value: MetaValue::F32(1.2),
        },
        MetaEntry {
            key: "llama.rope.beta_fast".to_string(),
            value: MetaValue::F32(32.0),
        },
        MetaEntry {
            key: "llama.rope.beta_slow".to_string(),
            value: MetaValue::F32(1.0),
        },
        MetaEntry {
            key: "llama.attention.head_dim".to_string(),
            value: MetaValue::U64(128),
        },
        MetaEntry {
            key: "llama.attention.dropout".to_string(),
            value: MetaValue::F32(0.1),
        },
        MetaEntry {
            key: "llama.feed_forward.activation".to_string(),
            value: MetaValue::Str("silu".to_string()),
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
            key: "tokenizer.hf.name".to_string(),
            value: MetaValue::Str("meta-llama/Llama-3.1-8B-Instruct".to_string()),
        },
        MetaEntry {
            key: "tokenizer.hf.pretrained_name".to_string(),
            value: MetaValue::Str("meta-llama/Llama-3.1-8B".to_string()),
        },
        MetaEntry {
            key: "tokenizer.ggml.add_bos_token".to_string(),
            value: MetaValue::Bool(true),
        },
        MetaEntry {
            key: "tokenizer.ggml.add_eos_token".to_string(),
            value: MetaValue::Bool(false),
        },
        MetaEntry {
            key: "tokenizer.ggml.scores".to_string(),
            value: MetaValue::ArrayF32(vec![0.1, 0.2, 0.3]),
        },
        MetaEntry {
            key: "tokenizer.ggml.token_type".to_string(),
            value: MetaValue::ArrayU32(vec![1, 2, 3]),
        },
        MetaEntry {
            key: "llama.num_experts".to_string(),
            value: MetaValue::U64(64),
        },
        MetaEntry {
            key: "llama.expert_intermediate_size".to_string(),
            value: MetaValue::U64(14336),
        },
    ]);
    let tensors = vec![
        make_tensor("f16_weight", GgmlDType::F16, vec![8]),
        make_tensor("f32_weight", GgmlDType::F32, vec![4]),
    ];

    let file = write_temp_gguf(&build_gguf(metadata, tensors, 32));
    let reader = GgufReader::open(file.path()).expect("open gguf");

    assert_eq!(reader.embedding_length(), Some(4096));
    assert_eq!(reader.block_count(), Some(32));
    assert_eq!(reader.head_count(), Some(32));
    assert_eq!(reader.head_count_kv(), Some(8));
    assert_eq!(reader.context_length(), Some(8192));
    assert_eq!(reader.rope_dimension_count(), Some(128));
    assert_eq!(reader.rope_freq_base(), Some(500_000.0));
    assert_eq!(reader.file_type(), Some(7));
    assert_eq!(reader.rope_scale(), Some(8.0));
    assert_eq!(reader.rope_scaling_type(), Some("yarn"));
    assert_eq!(reader.rope_scaling_factor(), Some(8.0));
    assert_eq!(reader.rope_scaling_factors(), Some(vec![8.0, 4.0]));
    assert_eq!(reader.rope_ext_factor(), Some(1.5));
    assert_eq!(reader.rope_attn_factor(), Some(1.2));
    assert_eq!(reader.rope_beta_fast(), Some(32.0));
    assert_eq!(reader.rope_beta_slow(), Some(1.0));
    assert_eq!(reader.attention_head_dim(), Some(128));
    assert_eq!(reader.attention_dropout(), Some(0.1));
    assert_eq!(reader.feed_forward_activation(), Some("silu"));
    assert_eq!(reader.bos_token_id(), Some(1));
    assert_eq!(reader.eos_token_id(), Some(2));
    assert_eq!(
        reader.hf_tokenizer_name(),
        Some("meta-llama/Llama-3.1-8B-Instruct")
    );
    assert_eq!(reader.hf_pretrained_name(), Some("meta-llama/Llama-3.1-8B"));
    assert_eq!(reader.num_experts(), Some(64));
    assert_eq!(reader.expert_intermediate_size(), Some(14336));
    assert!(reader.add_bos_token());
    assert!(!reader.add_eos_token());
    assert_eq!(
        reader.tokenizer_scores().expect("scores"),
        vec![0.1, 0.2, 0.3]
    );
    assert_eq!(
        reader.tokenizer_token_types().expect("token types"),
        vec![1, 2, 3]
    );

    assert!(reader.get("general.architecture").is_some());
    assert_eq!(reader.floating_point_dtype_size(), Some(2));
}
