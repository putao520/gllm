//! PyTorch .bin (zip+pickle) loader and safetensors conversion.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, Stack};
use candle_core::{Layout, Shape};
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use super::{LoaderError, Result};

#[derive(Debug, Clone)]
pub struct PytorchConversionConfig {
    pub state_dict_key: Option<String>,
    pub int4_name_hints: Vec<String>,
    pub force: bool,
}

impl Default for PytorchConversionConfig {
    fn default() -> Self {
        Self {
            state_dict_key: None,
            int4_name_hints: vec!["qweight".into(), "int4".into(), "q4".into()],
            force: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PytorchConversionOutput {
    pub safetensors: Vec<PathBuf>,
    pub index: Option<PathBuf>,
}

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

#[derive(Debug, Clone)]
struct PytorchTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
    packed_bits: Option<u8>,
}

#[derive(Debug, Deserialize, Serialize)]
struct BinIndex {
    weight_map: HashMap<String, String>,
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

pub fn convert_bins_to_safetensors(
    bin_paths: &[PathBuf],
    index_path: Option<&Path>,
    config: &PytorchConversionConfig,
) -> Result<PytorchConversionOutput> {
    if bin_paths.is_empty() {
        return Err(LoaderError::MissingWeights);
    }

    let mut safetensors_paths = Vec::with_capacity(bin_paths.len());
    for bin_path in bin_paths {
        let safe_path = bin_to_safetensors_path(bin_path)?;
        let tensors = load_pytorch_tensors(bin_path, config)?;
        if config.force || !safe_path.exists() {
            write_safetensors(&safe_path, &tensors)?;
        }
        safetensors_paths.push(safe_path);
    }

    let index = if let Some(index_path) = index_path {
        Some(write_safetensors_index(index_path, &safetensors_paths)?)
    } else {
        None
    };

    Ok(PytorchConversionOutput {
        safetensors: safetensors_paths,
        index,
    })
}

fn load_pytorch_tensors(
    bin_path: &Path,
    config: &PytorchConversionConfig,
) -> Result<Vec<PytorchTensor>> {
    let infos = read_tensor_infos(bin_path, config.state_dict_key.as_deref())?;
    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos.values() {
        let data = read_tensor_bytes(bin_path, info)?;
        let shape = info.layout.shape().dims().to_vec();
        let dtype = info.dtype.as_safetensors();
        let packed_bits = packed_bits_hint(&info.name, info.dtype, config);
        tensors.push(PytorchTensor {
            name: info.name.clone(),
            dtype,
            shape,
            data,
            packed_bits,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

fn read_tensor_infos(
    bin_path: &Path,
    key: Option<&str>,
) -> Result<HashMap<String, PytorchTensorInfo>> {
    let file = File::open(bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|err| LoaderError::Pytorch(err.to_string()))?;
    let file_names: Vec<String> = zip.file_names().map(|f| f.to_string()).collect();

    let mut infos = HashMap::new();
    for file_name in file_names {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = PathBuf::from(
            file_name
                .strip_suffix(".pkl")
                .ok_or_else(|| LoaderError::Pytorch("invalid pkl entry".into()))?,
        );
        let reader = zip
            .by_name(&file_name)
            .map_err(|err| LoaderError::Pytorch(err.to_string()))?;
        let mut reader = BufReader::new(reader);
        let mut stack = Stack::empty();
        stack
            .read_loop(&mut reader)
            .map_err(|err| LoaderError::Pytorch(err.to_string()))?;
        let obj = stack
            .finalize()
            .map_err(|err| LoaderError::Pytorch(err.to_string()))?;
        let obj = unwrap_module(obj);
        let obj = resolve_state_dict(obj, key)?;
        let dict = match obj {
            Object::Dict(dict) => dict,
            _ => continue,
        };
        for (name, value) in dict {
            if let Some(info) = tensor_info_from_object(value, name, &dir_name)? {
                if infos.contains_key(&info.name) {
                    return Err(LoaderError::DuplicateTensor(info.name));
                }
                infos.insert(info.name.clone(), info);
            }
        }
    }

    if infos.is_empty() {
        return Err(LoaderError::MissingWeights);
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

fn resolve_state_dict(obj: Object, key: Option<&str>) -> Result<Object> {
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
                return Err(LoaderError::Pytorch(format!(
                    "state dict key '{key}' not found"
                )));
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
) -> Result<Option<PytorchTensorInfo>> {
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
            let mut args = args
                .tuple()
                .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
            let callable = args.remove(0);
            let args = args.remove(1);
            (callable, args)
        }
        Object::Class {
            module_name,
            class_name,
        } if module_name == "torch._utils" && class_name == "_rebuild_parameter" => {
            let mut args = args
                .tuple()
                .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
            args.remove(0)
                .reduce()
                .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?
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

fn rebuild_args(args: Object) -> Result<(Layout, PytorchDtype, String)> {
    let mut args = args
        .tuple()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let stride = Vec::<usize>::try_from(args.remove(3))
        .map_err(|err| LoaderError::Pytorch(format!("invalid stride {err:?}")))?;
    let size = Vec::<usize>::try_from(args.remove(2))
        .map_err(|err| LoaderError::Pytorch(format!("invalid size {err:?}")))?;
    let offset = args
        .remove(1)
        .int_or_long()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let offset = usize::try_from(offset)
        .map_err(|_| LoaderError::Pytorch("negative storage offset".into()))?;
    let storage = args
        .remove(0)
        .persistent_load()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let mut storage = storage
        .tuple()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let _storage_size = storage
        .remove(4)
        .int_or_long()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let path = storage
        .remove(2)
        .unicode()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
    let (_module_name, class_name) = storage
        .remove(1)
        .class()
        .map_err(|err| LoaderError::Pytorch(format!("{err:?}")))?;
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
            return Err(LoaderError::Pytorch(format!(
                "unsupported storage type {other}"
            )))
        }
    };
    let layout = Layout::new(
        Shape::from(size),
        stride,
        offset.saturating_mul(dtype.size_in_bytes()),
    );
    Ok((layout, dtype, path))
}

fn read_tensor_bytes(bin_path: &Path, info: &PytorchTensorInfo) -> Result<Vec<u8>> {
    let file = File::open(bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|err| LoaderError::Pytorch(err.to_string()))?;
    let mut entry = zip
        .by_name(&info.path)
        .map_err(|err| LoaderError::Pytorch(err.to_string()))?;

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
        .ok_or_else(|| LoaderError::Pytorch("tensor size overflow".into()))?;
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
    Err(LoaderError::Pytorch(format!(
        "non-contiguous tensor layout for {}",
        info.name
    )))
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

fn packed_bits_hint(
    name: &str,
    dtype: PytorchDtype,
    config: &PytorchConversionConfig,
) -> Option<u8> {
    if dtype != PytorchDtype::U8 {
        return None;
    }
    let name = name.to_ascii_lowercase();
    for hint in &config.int4_name_hints {
        if name.contains(&hint.to_ascii_lowercase()) {
            return Some(4);
        }
    }
    None
}

fn write_safetensors(path: &Path, tensors: &[PytorchTensor]) -> Result<()> {
    let mut views = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
            .map_err(|err| LoaderError::Pytorch(err.to_string()))?;
        views.push((tensor.name.clone(), view));
    }
    let metadata = build_metadata(tensors)?;
    serialize_to_file(views, &metadata, path).map_err(|err| LoaderError::Pytorch(err.to_string()))
}

fn build_metadata(tensors: &[PytorchTensor]) -> Result<Option<HashMap<String, String>>> {
    let mut packed = HashMap::new();
    for tensor in tensors {
        if let Some(bits) = tensor.packed_bits {
            packed.insert(tensor.name.clone(), bits);
        }
    }
    if packed.is_empty() {
        return Ok(None);
    }
    let json = serde_json::to_string(&packed)?;
    let mut meta = HashMap::new();
    meta.insert("gllm.packed_bits".to_string(), json);
    Ok(Some(meta))
}

fn write_safetensors_index(
    bin_index_path: &Path,
    safetensors_paths: &[PathBuf],
) -> Result<PathBuf> {
    let bytes = std::fs::read(bin_index_path)?;
    let mut index: BinIndex = serde_json::from_slice(&bytes)?;
    for value in index.weight_map.values_mut() {
        *value = bin_name_to_safetensors(value);
    }
    let total_size: u64 = safetensors_paths
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|m| m.len()))
        .sum();
    index.metadata.insert(
        "total_size".to_string(),
        serde_json::Value::Number(total_size.into()),
    );

    let file_name = bin_index_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| LoaderError::Pytorch("invalid index filename".into()))?;
    let output_name = bin_index_name_to_safetensors(file_name);
    let output_path = bin_index_path.with_file_name(output_name);
    let data = serde_json::to_vec_pretty(&index)?;
    std::fs::write(&output_path, data)?;
    Ok(output_path)
}

fn bin_to_safetensors_path(bin_path: &Path) -> Result<PathBuf> {
    let file_name = bin_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| LoaderError::Pytorch("invalid bin filename".into()))?;
    let safe_name = bin_name_to_safetensors(file_name);
    Ok(bin_path.with_file_name(safe_name))
}

fn bin_name_to_safetensors(file_name: &str) -> String {
    let mut name = if let Some(rest) = file_name.strip_prefix("pytorch_model") {
        format!("model{rest}")
    } else {
        file_name.to_string()
    };
    if let Some(stripped) = name.strip_suffix(".bin") {
        name = format!("{stripped}.safetensors");
    } else if !name.ends_with(".safetensors") {
        name.push_str(".safetensors");
    }
    name
}

fn bin_index_name_to_safetensors(file_name: &str) -> String {
    let mut name = if let Some(rest) = file_name.strip_prefix("pytorch_model") {
        format!("model{rest}")
    } else {
        file_name.to_string()
    };
    if let Some(stripped) = name.strip_suffix(".bin.index.json") {
        name = format!("{stripped}.safetensors.index.json");
    } else if let Some(stripped) = name.strip_suffix(".bin") {
        name = format!("{stripped}.safetensors.index.json");
    } else if !name.ends_with(".safetensors.index.json") {
        name.push_str(".safetensors.index.json");
    }
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_name_mapping() {
        assert_eq!(
            bin_name_to_safetensors("pytorch_model.bin"),
            "model.safetensors"
        );
        assert_eq!(
            bin_name_to_safetensors("pytorch_model-00001-of-00002.bin"),
            "model-00001-of-00002.safetensors"
        );
    }

    #[test]
    fn bin_index_name_mapping() {
        assert_eq!(
            bin_index_name_to_safetensors("pytorch_model.bin.index.json"),
            "model.safetensors.index.json"
        );
    }

    #[test]
    fn fortran_reorder_roundtrip() {
        let data: Vec<u8> = (0u8..8).collect();
        let shape = vec![2, 4];
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out.len(), 8);
    }
}
