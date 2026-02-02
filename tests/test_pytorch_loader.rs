use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use candle_core::pickle::{Object, Stack};
use candle_core::{Layout, Shape};
use gllm::loader::pytorch::convert_bins_to_safetensors;
use gllm::loader::{LoaderError, PytorchConversionConfig};
use hf_hub::api::sync::ApiBuilder;
use safetensors::{Dtype, SafeTensors};
use zip::ZipArchive;

const F32_REPO: &str = "hf-internal-testing/tiny-random-bert";
const F16_REPO: &str = "sanchit-gandhi/tiny-random-bart-fp16";
const BF16_REPO: &str = "Rocketknight1/tiny-random-gpt2-bfloat16-pt";

#[derive(Debug)]
struct TestError(String);

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TestError {}

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PytorchDtype {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl PytorchDtype {
    fn size_in_bytes(self) -> usize {
        match self {
            PytorchDtype::Bool => 1,
            PytorchDtype::U8 => 1,
            PytorchDtype::I8 => 1,
            PytorchDtype::I16 => 2,
            PytorchDtype::I32 => 4,
            PytorchDtype::I64 => 8,
            PytorchDtype::F16 => 2,
            PytorchDtype::BF16 => 2,
            PytorchDtype::F32 => 4,
            PytorchDtype::F64 => 8,
        }
    }

    fn as_safetensors(self) -> Dtype {
        match self {
            PytorchDtype::Bool => Dtype::BOOL,
            PytorchDtype::U8 => Dtype::U8,
            PytorchDtype::I8 => Dtype::I8,
            PytorchDtype::I16 => Dtype::I16,
            PytorchDtype::I32 => Dtype::I32,
            PytorchDtype::I64 => Dtype::I64,
            PytorchDtype::F16 => Dtype::F16,
            PytorchDtype::BF16 => Dtype::BF16,
            PytorchDtype::F32 => Dtype::F32,
            PytorchDtype::F64 => Dtype::F64,
        }
    }
}

#[derive(Debug, Clone)]
struct PytorchTensorInfo {
    name: String,
    dtype: PytorchDtype,
    layout: Layout,
    path: String,
}

static DOWNLOAD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn download_lock() -> &'static Mutex<()> {
    DOWNLOAD_LOCK.get_or_init(|| Mutex::new(()))
}

fn hf_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("GLLM_TEST_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-cache")
}

fn download_bin(repo: &str) -> TestResult<PathBuf> {
    let _guard = download_lock().lock().expect("download lock poisoned");
    let cache_dir = hf_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;
    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .map_err(|err| TestError(err.to_string()))?;
    let model = api.model(repo.to_string());
    let path = model
        .get("pytorch_model.bin")
        .map_err(|err| TestError(err.to_string()))?;
    Ok(path)
}

fn run_roundtrip(repo: &str, expected_dtype: PytorchDtype) -> TestResult<()> {
    let bin_path = download_bin(repo)?;
    let config = PytorchConversionConfig {
        force: true,
        ..PytorchConversionConfig::default()
    };
    let output = convert_bins_to_safetensors(&[bin_path.clone()], None, &config)?;
    let safe_path = output
        .safetensors
        .first()
        .ok_or_else(|| TestError("missing safetensors output".to_string()))?;

    let infos = read_tensor_infos(&bin_path, None)?;
    if !infos.values().any(|info| info.dtype == expected_dtype) {
        return Err(TestError(format!(
            "expected dtype {:?} not found in {repo}",
            expected_dtype
        ))
        .into());
    }

    let safe_bytes = std::fs::read(safe_path)?;
    let safe = SafeTensors::deserialize(&safe_bytes)?;
    assert_eq!(safe.names().len(), infos.len());

    let file = File::open(&bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader)?;

    for info in infos.values() {
        let tensor = safe.tensor(&info.name)?;
        assert_eq!(tensor.dtype(), info.dtype.as_safetensors());
        assert_eq!(tensor.shape(), info.layout.shape().dims());
        let data = read_tensor_bytes(&mut zip, info)?;
        assert_eq!(
            tensor.data(),
            data.as_slice(),
            "tensor {} differs",
            info.name
        );
    }

    Ok(())
}

#[test]
fn pytorch_bin_roundtrip_f32() {
    run_roundtrip(F32_REPO, PytorchDtype::F32).expect("f32 roundtrip");
}

#[test]
fn pytorch_bin_roundtrip_f16() {
    run_roundtrip(F16_REPO, PytorchDtype::F16).expect("f16 roundtrip");
}

#[test]
fn pytorch_bin_roundtrip_bf16() {
    run_roundtrip(BF16_REPO, PytorchDtype::BF16).expect("bf16 roundtrip");
}

#[test]
fn pytorch_conversion_rejects_empty_input() {
    let err = convert_bins_to_safetensors(&[], None, &PytorchConversionConfig::default())
        .expect_err("expected error");
    assert!(matches!(err, LoaderError::MissingWeights));
}

#[test]
fn pytorch_conversion_rejects_missing_state_dict_key() {
    let bin_path = download_bin(F32_REPO).expect("download bin");
    let config = PytorchConversionConfig {
        state_dict_key: Some("missing_key".to_string()),
        force: true,
        ..PytorchConversionConfig::default()
    };
    let err = convert_bins_to_safetensors(&[bin_path], None, &config)
        .expect_err("expected state dict error");
    match err {
        LoaderError::Pytorch(msg) => {
            assert!(msg.contains("state dict"), "unexpected error: {msg}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

fn read_tensor_infos(
    bin_path: &Path,
    key: Option<&str>,
) -> TestResult<HashMap<String, PytorchTensorInfo>> {
    let file = File::open(bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader)?;
    let file_names: Vec<String> = zip.file_names().map(|f| f.to_string()).collect();

    let mut infos = HashMap::new();
    for file_name in file_names {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = PathBuf::from(
            file_name
                .strip_suffix(".pkl")
                .ok_or_else(|| TestError("invalid pkl entry".to_string()))?,
        );
        let reader = zip
            .by_name(&file_name)
            .map_err(|err| TestError(err.to_string()))?;
        let mut reader = BufReader::new(reader);
        let mut stack = Stack::empty();
        stack
            .read_loop(&mut reader)
            .map_err(|err| TestError(err.to_string()))?;
        let obj = stack
            .finalize()
            .map_err(|err| TestError(err.to_string()))?;
        let obj = unwrap_module(obj);
        let obj = resolve_state_dict(obj, key)?;
        let dict = match obj {
            Object::Dict(dict) => dict,
            _ => continue,
        };
        for (name, value) in dict {
            if let Some(info) = tensor_info_from_object(value, name, &dir_name)? {
                if infos.contains_key(&info.name) {
                    return Err(TestError(format!("duplicate tensor name: {}", info.name)).into());
                }
                infos.insert(info.name.clone(), info);
            }
        }
    }

    if infos.is_empty() {
        return Err(TestError("missing weights".to_string()).into());
    }
    Ok(infos)
}

fn unwrap_module(obj: Object) -> Object {
    match obj {
        Object::Build { callable, args } => match *callable {
            Object::Reduce { callable, args: _ } => match *callable {
                Object::Class {
                    module_name,
                    class_name,
                } if module_name == "__torch__" && class_name == "Module" => *args,
                _ => Object::Build { callable, args },
            },
            _ => Object::Build { callable, args },
        },
        other => other,
    }
}

fn resolve_state_dict(obj: Object, key: Option<&str>) -> TestResult<Object> {
    match obj {
        Object::Dict(dict) => {
            if let Some(key) = key {
                for (k, v) in dict {
                    if let Object::Unicode(name) = k {
                        if name == key {
                            return Ok(v);
                        }
                    }
                }
                return Err(TestError(format!(
                    "state dict key '{key}' not found"
                ))
                .into());
            }
            for (k, v) in dict.iter() {
                if let Object::Unicode(name) = k {
                    if name == "state_dict" {
                        return Ok(v.clone());
                    }
                }
            }
            Ok(Object::Dict(dict))
        }
        other => Ok(other),
    }
}

fn tensor_info_from_object(
    value: Object,
    name: Object,
    dir_name: &Path,
) -> TestResult<Option<PytorchTensorInfo>> {
    let name = match name.unicode() {
        Ok(name) => name,
        Err(_) => return Ok(None),
    };
    let (callable, args) = match value.reduce() {
        Ok(callable_args) => callable_args,
        _ => return Ok(None),
    };
    let (callable, args) = match callable {
        Object::Class {
            module_name,
            class_name,
        } if module_name == "torch._tensor" && class_name == "_rebuild_from_type_v2" => {
            let mut args =
                args.tuple()
                    .map_err(|err| TestError(format!("{err:?}")))?;
            let callable = args.remove(0);
            let args = args.remove(1);
            (callable, args)
        }
        Object::Class {
            module_name,
            class_name,
        } if module_name == "torch._utils" && class_name == "_rebuild_parameter" => {
            let mut args =
                args.tuple()
                    .map_err(|err| TestError(format!("{err:?}")))?;
            args.remove(0)
                .reduce()
                .map_err(|err| TestError(format!("{err:?}")))?
        }
        other => (other, args),
    };
    match callable {
        Object::Class {
            module_name,
            class_name,
        } if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2" => {}
        _ => return Ok(None),
    };
    let (layout, dtype, file_path) = rebuild_args(args)?;
    Ok(Some(PytorchTensorInfo {
        name,
        dtype,
        layout,
        path: format!("{}/{}", dir_name.to_string_lossy(), file_path),
    }))
}

fn rebuild_args(args: Object) -> TestResult<(Layout, PytorchDtype, String)> {
    let mut args = args
        .tuple()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let stride = Vec::<usize>::try_from(args.remove(3))
        .map_err(|err| TestError(format!("invalid stride {err:?}")))?;
    let size = Vec::<usize>::try_from(args.remove(2))
        .map_err(|err| TestError(format!("invalid size {err:?}")))?;
    let offset = args
        .remove(1)
        .int_or_long()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let offset = usize::try_from(offset)
        .map_err(|_| TestError("negative storage offset".to_string()))?;
    let storage = args
        .remove(0)
        .persistent_load()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let mut storage = storage
        .tuple()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let _storage_size = storage
        .remove(4)
        .int_or_long()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let path = storage
        .remove(2)
        .unicode()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let (_module_name, class_name) = storage
        .remove(1)
        .class()
        .map_err(|err| TestError(format!("{err:?}")))?;
    let dtype = match class_name.as_str() {
        "FloatStorage" => PytorchDtype::F32,
        "DoubleStorage" => PytorchDtype::F64,
        "HalfStorage" => PytorchDtype::F16,
        "BFloat16Storage" => PytorchDtype::BF16,
        "ByteStorage" => PytorchDtype::U8,
        "CharStorage" => PytorchDtype::I8,
        "ShortStorage" => PytorchDtype::I16,
        "IntStorage" => PytorchDtype::I32,
        "LongStorage" => PytorchDtype::I64,
        "BoolStorage" => PytorchDtype::Bool,
        other => {
            return Err(TestError(format!(
                "unsupported storage type {other}"
            ))
            .into())
        }
    };
    let layout = Layout::new(
        Shape::from(size),
        stride,
        offset.saturating_mul(dtype.size_in_bytes()),
    );
    Ok((layout, dtype, path))
}

fn read_tensor_bytes(
    zip: &mut ZipArchive<BufReader<File>>,
    info: &PytorchTensorInfo,
) -> TestResult<Vec<u8>> {
    let mut entry = zip
        .by_name(&info.path)
        .map_err(|err| TestError(err.to_string()))?;

    let start_offset = info.layout.start_offset();
    if start_offset > 0 {
        std::io::copy(
            &mut entry.by_ref().take(start_offset as u64),
            &mut std::io::sink(),
        )?;
    }

    let elem_count: usize = info.layout.shape().dims().iter().product();
    let byte_len = elem_count
        .checked_mul(info.dtype.size_in_bytes())
        .ok_or_else(|| TestError("tensor size overflow".to_string()))?;
    let mut data = vec![0u8; byte_len];
    entry.read_exact(&mut data)?;

    if info.layout.is_contiguous() || elem_count <= 1 {
        return Ok(data);
    }
    if info.layout.is_fortran_contiguous() {
        return Ok(reorder_fortran_to_c(
            &data,
            info.layout.shape().dims(),
            info.dtype.size_in_bytes(),
        ));
    }
    Err(TestError(format!(
        "non-contiguous tensor layout for {}",
        info.name
    ))
    .into())
}

fn reorder_fortran_to_c(data: &[u8], shape: &[usize], elem_size: usize) -> Vec<u8> {
    let rank = shape.len();
    if rank <= 1 {
        return data.to_vec();
    }
    let elem_count: usize = shape.iter().product();
    let mut out = vec![0u8; elem_count * elem_size];

    let mut c_strides = vec![1usize; rank];
    for idx in (0..rank.saturating_sub(1)).rev() {
        c_strides[idx] = c_strides[idx + 1].saturating_mul(shape[idx + 1]);
    }
    let mut f_strides = vec![1usize; rank];
    for idx in 1..rank {
        f_strides[idx] = f_strides[idx - 1].saturating_mul(shape[idx - 1]);
    }

    for linear_c in 0..elem_count {
        let mut rem = linear_c;
        let mut linear_f = 0usize;
        for axis in 0..rank {
            let stride = c_strides[axis];
            let idx = rem / stride;
            rem %= stride;
            linear_f = linear_f.saturating_add(idx.saturating_mul(f_strides[axis]));
        }
        let src = linear_f * elem_size;
        let dst = linear_c * elem_size;
        out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
    }
    out
}
