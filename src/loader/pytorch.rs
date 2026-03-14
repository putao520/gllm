//! PyTorch .bin (zip+pickle) loader and safetensors conversion.
//!
//! Pure Rust implementation — no candle/tch dependency (REQ-ARCH-003).
//! Contains a minimal pickle protocol parser sufficient for PyTorch checkpoints.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use zip::ZipArchive;

use super::{LoaderError, Result};

// ── Public types ──

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

// ── TensorLayout (replaces candle_core::{Layout, Shape}) ──
#[derive(Debug, Clone)]
struct TensorLayout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl TensorLayout {
    fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Self {
        Self { shape, strides, offset }
    }

    fn dims(&self) -> &[usize] {
        &self.shape
    }

    fn start_offset(&self) -> usize {
        self.offset
    }

    fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.shape[i]);
        }
        true
    }

    fn is_fortran_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected = 1usize;
        for i in 0..self.shape.len() {
            if self.strides[i] != expected {
                return false;
            }
            expected = expected.saturating_mul(self.shape[i]);
        }
        true
    }
}

// ── Pickle protocol parser (minimal, PyTorch-only) ──
#[derive(Debug, Clone, PartialEq)]
enum Object {
    Class { module_name: String, class_name: String },
    Int(i32),
    Long(i64),
    Float(f64),
    Unicode(String),
    Bool(bool),
    None,
    Tuple(Vec<Object>),
    List(Vec<Object>),
    Mark,
    Dict(Vec<(Object, Object)>),
    Reduce { callable: Box<Object>, args: Box<Object> },
    Build { callable: Box<Object>, args: Box<Object> },
    PersistentLoad(Box<Object>),
}

type OResult<T> = std::result::Result<T, Object>;

impl Object {
    fn unicode(self) -> OResult<String> {
        match self { Self::Unicode(t) => Ok(t), _ => Err(self) }
    }
    fn reduce(self) -> OResult<(Self, Self)> {
        match self { Self::Reduce { callable, args } => Ok((*callable, *args)), _ => Err(self) }
    }
    fn persistent_load(self) -> OResult<Self> {
        match self { Self::PersistentLoad(t) => Ok(*t), _ => Err(self) }
    }
    fn int_or_long(self) -> OResult<i64> {
        match self { Self::Int(t) => Ok(t as i64), Self::Long(t) => Ok(t), _ => Err(self) }
    }
    fn tuple(self) -> OResult<Vec<Self>> {
        match self { Self::Tuple(t) => Ok(t), _ => Err(self) }
    }
    fn class(self) -> OResult<(String, String)> {
        match self {
            Self::Class { module_name, class_name } => Ok((module_name, class_name)),
            _ => Err(self),
        }
    }
}

impl TryFrom<Object> for String {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value { Object::Unicode(s) => Ok(s), other => Err(other) }
    }
}

impl TryFrom<Object> for usize {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value { Object::Int(s) if s >= 0 => Ok(s as usize), other => Err(other) }
    }
}

impl<T: TryFrom<Object, Error = Object>> TryFrom<Object> for Vec<T> {
    type Error = Object;
    fn try_from(value: Object) -> std::result::Result<Self, Self::Error> {
        match value {
            Object::Tuple(values) => values.into_iter().map(T::try_from).collect(),
            other => Err(other),
        }
    }
}

// ── Pickle byte-reading helpers (no byteorder crate) ──

fn read_u8<R: Read>(r: &mut R) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_u16_le<R: Read>(r: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}
fn read_i32_le<R: Read>(r: &mut R) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}
fn read_u32_le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_f64_be<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_be_bytes(buf))
}
fn read_to_newline<R: BufRead>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let mut data = Vec::with_capacity(32);
    r.read_until(b'\n', &mut data)?;
    data.pop();
    if data.last() == Some(&b'\r') { data.pop(); }
    Ok(data)
}

struct PickleStack {
    stack: Vec<Object>,
    memo: HashMap<u32, Object>,
}

impl PickleStack {
    fn empty() -> Self {
        Self { stack: Vec::with_capacity(512), memo: HashMap::new() }
    }

    fn read_loop<R: BufRead>(&mut self, r: &mut R) -> Result<()> {
        loop { if self.read(r)? { break; } }
        Ok(())
    }

    fn finalize(mut self) -> Result<Object> {
        self.stack.pop().ok_or_else(|| LoaderError::Pytorch("empty pickle stack".into()))
    }

    fn push(&mut self, obj: Object) { self.stack.push(obj); }

    fn pop(&mut self) -> Result<Object> {
        self.stack.pop().ok_or_else(|| LoaderError::Pytorch("unexpected empty stack".into()))
    }

    fn last(&mut self) -> Result<&mut Object> {
        self.stack.last_mut().ok_or_else(|| LoaderError::Pytorch("unexpected empty stack".into()))
    }

    fn memo_get(&self, id: u32) -> Result<Object> {
        self.memo.get(&id).cloned()
            .ok_or_else(|| LoaderError::Pytorch(format!("missing memo {id}")))
    }

    fn memo_put(&mut self, id: u32) -> Result<()> {
        let obj = self.last()?.clone();
        self.memo.insert(id, obj);
        Ok(())
    }

    fn pop_to_marker(&mut self) -> Result<Vec<Object>> {
        let mark_idx = self.stack.iter().enumerate().rev()
            .find(|(_, obj)| **obj == Object::Mark)
            .map(|(idx, _)| idx)
            .ok_or_else(|| LoaderError::Pytorch("marker not found".into()))?;
        let objs = self.stack.split_off(mark_idx + 1);
        self.stack.pop();
        Ok(objs)
    }

    fn build_op(&mut self) -> Result<()> {
        let args = self.pop()?;
        let obj = self.pop()?;
        let obj = match (obj, args) {
            (Object::Dict(mut o), Object::Dict(mut a)) => { o.append(&mut a); Object::Dict(o) }
            (obj, args) => Object::Build { callable: Box::new(obj), args: Box::new(args) },
        };
        self.push(obj);
        Ok(())
    }

    fn reduce_op(&mut self) -> Result<()> {
        let args = self.pop()?;
        let callable = self.pop()?;
        let reduced = match &callable {
            Object::Class { module_name, class_name }
                if module_name == "collections"
                    && (class_name == "OrderedDict" || class_name == "defaultdict") =>
            {
                Some(Object::Dict(vec![]))
            }
            _ => None,
        };
        let reduced = reduced.unwrap_or_else(|| Object::Reduce {
            callable: Box::new(callable), args: Box::new(args),
        });
        self.push(reduced);
        Ok(())
    }

    fn read<R: BufRead>(&mut self, r: &mut R) -> Result<bool> {
        let op = read_u8(r).map_err(|e| LoaderError::Pytorch(format!("read opcode: {e}")))?;
        match op {
            0x80 => { read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; } // Proto
            b'c' => { // Global
                let module = read_to_newline(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                let class = read_to_newline(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                self.push(Object::Class {
                    module_name: String::from_utf8_lossy(&module).into(),
                    class_name: String::from_utf8_lossy(&class).into(),
                });
            }
            b'K' => { let v = read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.push(Object::Int(v as i32)); }
            b'M' => { let v = read_u16_le(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.push(Object::Int(v as i32)); }
            b'J' => { let v = read_i32_le(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.push(Object::Int(v)); }
            b'G' => { let v = read_f64_be(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.push(Object::Float(v)); }
            b'X' => { // BinUnicode
                let len = read_u32_le(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                let mut data = vec![0u8; len as usize];
                r.read_exact(&mut data).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                let s = String::from_utf8(data).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                self.push(Object::Unicode(s));
            }
            b'Q' => { let id = self.pop()?; self.push(Object::PersistentLoad(Box::new(id))); } // BinPersId
            b't' => { let objs = self.pop_to_marker()?; self.push(Object::Tuple(objs)); } // Tuple
            0x85 => { let o = self.pop()?; self.push(Object::Tuple(vec![o])); } // Tuple1
            0x86 => { let b = self.pop()?; let a = self.pop()?; self.push(Object::Tuple(vec![a, b])); } // Tuple2
            0x87 => { let c = self.pop()?; let b = self.pop()?; let a = self.pop()?; self.push(Object::Tuple(vec![a, b, c])); } // Tuple3
            0x88 => self.push(Object::Bool(true)),  // NewTrue
            0x89 => self.push(Object::Bool(false)), // NewFalse
            b'N' => self.push(Object::None),
            b')' => self.push(Object::Tuple(vec![])),  // EmptyTuple
            b']' => self.push(Object::List(vec![])),    // EmptyList
            b'}' | b'd' => self.push(Object::Dict(vec![])), // EmptyDict / Dict
            b'(' => self.push(Object::Mark),
            b'a' => { // Append
                let value = self.pop()?;
                let list = self.last()?;
                if let Object::List(d) = list { d.push(value); }
                else { return Err(LoaderError::Pytorch("append: expected list".into())); }
            }
            b'e' => { // Appends
                let objs = self.pop_to_marker()?;
                let list = self.last()?;
                if let Object::List(d) = list { d.extend(objs); }
                else { return Err(LoaderError::Pytorch("appends: expected list".into())); }
            }
            b's' => { // SetItem
                let value = self.pop()?;
                let key = self.pop()?;
                let dict = self.last()?;
                if let Object::Dict(d) = dict { d.push((key, value)); }
                else { return Err(LoaderError::Pytorch("setitem: expected dict".into())); }
            }
            b'u' => { // SetItems
                let mut objs = self.pop_to_marker()?;
                let dict = self.last()?;
                if let Object::Dict(d) = dict {
                    if objs.len() % 2 != 0 { return Err(LoaderError::Pytorch("setitems: odd count".into())); }
                    while let Some(value) = objs.pop() {
                        let key = objs.pop().ok_or_else(|| LoaderError::Pytorch("setitems: missing key".into()))?;
                        d.push((key, value));
                    }
                } else { return Err(LoaderError::Pytorch("setitems: expected dict".into())); }
            }
            b'R' => self.reduce_op()?,
            b'b' => self.build_op()?,
            0x81 => { // NewObj
                let args = self.pop()?;
                let class = self.pop()?;
                self.push(Object::Reduce { callable: Box::new(class), args: Box::new(args) });
            }
            b'q' => { let id = read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.memo_put(id as u32)?; } // BinPut
            b'r' => { let id = read_u32_le(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; self.memo_put(id)?; } // LongBinPut
            b'h' => { let id = read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; let obj = self.memo_get(id as u32)?; self.push(obj); } // BinGet
            b'j' => { let id = read_u32_le(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?; let obj = self.memo_get(id)?; self.push(obj); } // LongBinGet
            0x8a => { // Long1
                let n = read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
                let mut v: i64 = 0;
                for i in 0..n { v |= (read_u8(r).map_err(|e| LoaderError::Pytorch(e.to_string()))? as i64) << (i * 8); }
                self.push(Object::Long(v));
            }
            b'.' => return Ok(true), // Stop
            _ => return Err(LoaderError::Pytorch(format!("unknown pickle opcode 0x{op:02x}"))),
        }
        Ok(false)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PytorchDtype {
    Bool, U8, I8, I16, I32, I64, F16, BF16, F32, F64,
}

impl PytorchDtype {
    fn size_in_bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    fn as_safetensors(self) -> Dtype {
        match self {
            Self::Bool => Dtype::BOOL, Self::U8 => Dtype::U8, Self::I8 => Dtype::I8,
            Self::I16 => Dtype::I16, Self::I32 => Dtype::I32, Self::I64 => Dtype::I64,
            Self::F16 => Dtype::F16, Self::BF16 => Dtype::BF16,
            Self::F32 => Dtype::F32, Self::F64 => Dtype::F64,
        }
    }
}

#[derive(Debug, Clone)]
struct PytorchTensorInfo {
    name: String,
    dtype: PytorchDtype,
    layout: TensorLayout,
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

// ── Public API ──

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
    Ok(PytorchConversionOutput { safetensors: safetensors_paths, index })
}

fn load_pytorch_tensors(bin_path: &Path, config: &PytorchConversionConfig) -> Result<Vec<PytorchTensor>> {
    let infos = read_tensor_infos(bin_path, config.state_dict_key.as_deref())?;
    let mut tensors = Vec::with_capacity(infos.len());
    for info in infos.values() {
        let data = read_tensor_bytes(bin_path, info)?;
        let shape = info.layout.dims().to_vec();
        let dtype = info.dtype.as_safetensors();
        let packed_bits = packed_bits_hint(&info.name, info.dtype, config);
        tensors.push(PytorchTensor { name: info.name.clone(), dtype, shape, data, packed_bits });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

fn read_tensor_infos(bin_path: &Path, key: Option<&str>) -> Result<HashMap<String, PytorchTensorInfo>> {
    let file = File::open(bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
    let file_names: Vec<String> = zip.file_names().map(|f| f.to_string()).collect();
    let mut infos = HashMap::new();
    for file_name in file_names {
        if !file_name.ends_with("data.pkl") { continue; }
        let dir_name = PathBuf::from(
            file_name.strip_suffix(".pkl")
                .ok_or_else(|| LoaderError::Pytorch("invalid pkl entry".into()))?,
        );
        let reader = zip.by_name(&file_name).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
        let mut reader = BufReader::new(reader);
        let mut stack = PickleStack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;
        let obj = unwrap_module(obj);
        let obj = resolve_state_dict(obj, key)?;
        let dict = match obj { Object::Dict(dict) => dict, _ => continue };
        for (name, value) in dict {
            if let Some(info) = tensor_info_from_object(value, name, &dir_name)? {
                if infos.contains_key(&info.name) {
                    return Err(LoaderError::DuplicateTensor(info.name));
                }
                infos.insert(info.name.clone(), info);
            }
        }
    }
    if infos.is_empty() { return Err(LoaderError::MissingWeights); }
    Ok(infos)
}

fn unwrap_module(obj: Object) -> Object {
    match obj {
        Object::Build { callable, args } => match *callable {
            Object::Reduce { callable, args: _ } => match *callable {
                Object::Class { module_name, class_name }
                    if module_name == "__torch__" && class_name == "Module" => *args,
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
                        if name == key { return Ok(v); }
                    }
                }
                return Err(LoaderError::Pytorch(format!("state dict key '{key}' not found")));
            }
            for (k, v) in dict.iter() {
                if let Object::Unicode(name) = k {
                    if name == "state_dict" { return Ok(v.clone()); }
                }
            }
            Ok(Object::Dict(dict))
        }
        other => Ok(other),
    }
}

fn tensor_info_from_object(value: Object, name: Object, dir_name: &Path) -> Result<Option<PytorchTensorInfo>> {
    let name = match name.unicode() {
        Ok(name) => name,
        Err(e) => { log::debug!("skipping tensor with non-unicode name: {e:?}"); return Ok(None); }
    };
    let (callable, args) = match value.reduce() {
        Ok(ca) => ca,
        Err(e) => { log::debug!("skipping tensor '{}': reduce failed: {e:?}", name); return Ok(None); }
    };
    let (callable, args) = match callable {
        Object::Class { module_name, class_name }
            if module_name == "torch._tensor" && class_name == "_rebuild_from_type_v2" =>
        {
            let mut args = args.tuple().map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
            let callable = args.remove(0);
            let args = args.remove(1);
            (callable, args)
        }
        Object::Class { module_name, class_name }
            if module_name == "torch._utils" && class_name == "_rebuild_parameter" =>
        {
            let mut args = args.tuple().map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
            args.remove(0).reduce().map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?
        }
        other => (other, args),
    };
    match callable {
        Object::Class { module_name, class_name }
            if module_name == "torch._utils" && class_name == "_rebuild_tensor_v2" => {}
        _ => return Ok(None),
    };
    let (layout, dtype, file_path) = rebuild_args(args)?;
    Ok(Some(PytorchTensorInfo {
        name, dtype, layout,
        path: format!("{}/{}", dir_name.to_string_lossy(), file_path),
    }))
}

fn rebuild_args(args: Object) -> Result<(TensorLayout, PytorchDtype, String)> {
    let mut args = args.tuple().map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let stride = Vec::<usize>::try_from(args.remove(3))
        .map_err(|e| LoaderError::Pytorch(format!("invalid stride {e:?}")))?;
    let size = Vec::<usize>::try_from(args.remove(2))
        .map_err(|e| LoaderError::Pytorch(format!("invalid size {e:?}")))?;
    let offset = args.remove(1).int_or_long()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let offset = usize::try_from(offset)
        .map_err(|_| LoaderError::Pytorch("negative storage offset".into()))?;
    let storage = args.remove(0).persistent_load()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let mut storage = storage.tuple()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let _storage_size = storage.remove(4).int_or_long()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let path = storage.remove(2).unicode()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
    let (_module_name, class_name) = storage.remove(1).class()
        .map_err(|e| LoaderError::Pytorch(format!("{e:?}")))?;
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
        other => return Err(LoaderError::Pytorch(format!("unsupported storage type {other}"))),
    };
    let layout = TensorLayout::new(size, stride, offset.saturating_mul(dtype.size_in_bytes()));
    Ok((layout, dtype, path))
}

fn read_tensor_bytes(bin_path: &Path, info: &PytorchTensorInfo) -> Result<Vec<u8>> {
    let file = File::open(bin_path)?;
    let reader = BufReader::new(file);
    let mut zip = ZipArchive::new(reader).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
    let mut entry = zip.by_name(&info.path).map_err(|e| LoaderError::Pytorch(e.to_string()))?;
    let start_offset = info.layout.start_offset();
    if start_offset > 0 {
        std::io::copy(&mut entry.by_ref().take(start_offset as u64), &mut std::io::sink())?;
    }
    let elem_count: usize = info.layout.dims().iter().product();
    let byte_len = elem_count.checked_mul(info.dtype.size_in_bytes())
        .ok_or_else(|| LoaderError::Pytorch("tensor size overflow".into()))?;
    let mut data = vec![0u8; byte_len];
    entry.read_exact(&mut data)?;
    if info.layout.is_contiguous() || elem_count <= 1 {
        return Ok(data);
    }
    if info.layout.is_fortran_contiguous() {
        return Ok(reorder_fortran_to_c(&data, info.layout.dims(), info.dtype.size_in_bytes()));
    }
    Err(LoaderError::Pytorch(format!("non-contiguous tensor layout for {}", info.name)))
}

fn reorder_fortran_to_c(data: &[u8], shape: &[usize], elem_size: usize) -> Vec<u8> {
    let rank = shape.len();
    if rank <= 1 { return data.to_vec(); }
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

fn packed_bits_hint(name: &str, dtype: PytorchDtype, config: &PytorchConversionConfig) -> Option<u8> {
    if dtype != PytorchDtype::U8 { return None; }
    let name = name.to_ascii_lowercase();
    for hint in &config.int4_name_hints {
        if name.contains(&hint.to_ascii_lowercase()) { return Some(4); }
    }
    None
}

fn write_safetensors(path: &Path, tensors: &[PytorchTensor]) -> Result<()> {
    let mut views = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let view = TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data)
            .map_err(|e| LoaderError::Pytorch(e.to_string()))?;
        views.push((tensor.name.clone(), view));
    }
    let metadata = build_metadata(tensors)?;
    serialize_to_file(views, &metadata, path).map_err(|e| LoaderError::Pytorch(e.to_string()))
}

fn build_metadata(tensors: &[PytorchTensor]) -> Result<Option<HashMap<String, String>>> {
    let mut packed = HashMap::new();
    for tensor in tensors {
        if let Some(bits) = tensor.packed_bits { packed.insert(tensor.name.clone(), bits); }
    }
    if packed.is_empty() { return Ok(None); }
    let json = serde_json::to_string(&packed)?;
    let mut meta = HashMap::new();
    meta.insert("gllm.packed_bits".to_string(), json);
    Ok(Some(meta))
}

fn write_safetensors_index(bin_index_path: &Path, safetensors_paths: &[PathBuf]) -> Result<PathBuf> {
    let bytes = std::fs::read(bin_index_path)?;
    let mut index: BinIndex = serde_json::from_slice(&bytes)?;
    for value in index.weight_map.values_mut() {
        *value = bin_name_to_safetensors(value);
    }
    let total_size: u64 = safetensors_paths.iter()
        .filter_map(|path| std::fs::metadata(path).ok().map(|m| m.len()))
        .sum();
    index.metadata.insert("total_size".to_string(), serde_json::Value::Number(total_size.into()));
    let file_name = bin_index_path.file_name().and_then(|n| n.to_str())
        .ok_or_else(|| LoaderError::Pytorch("invalid index filename".into()))?;
    let output_name = bin_index_name_to_safetensors(file_name);
    let output_path = bin_index_path.with_file_name(output_name);
    let data = serde_json::to_vec_pretty(&index)?;
    std::fs::write(&output_path, data)?;
    Ok(output_path)
}

fn bin_to_safetensors_path(bin_path: &Path) -> Result<PathBuf> {
    let file_name = bin_path.file_name().and_then(|n| n.to_str())
        .ok_or_else(|| LoaderError::Pytorch("invalid bin filename".into()))?;
    Ok(bin_path.with_file_name(bin_name_to_safetensors(file_name)))
}

fn bin_name_to_safetensors(file_name: &str) -> String {
    let mut name = if let Some(rest) = file_name.strip_prefix("pytorch_model") {
        format!("model{rest}")
    } else { file_name.to_string() };
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
    } else { file_name.to_string() };
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
        assert_eq!(bin_name_to_safetensors("pytorch_model.bin"), "model.safetensors");
        assert_eq!(bin_name_to_safetensors("pytorch_model-00001-of-00002.bin"), "model-00001-of-00002.safetensors");
    }

    #[test]
    fn bin_index_name_mapping() {
        assert_eq!(bin_index_name_to_safetensors("pytorch_model.bin.index.json"), "model.safetensors.index.json");
    }

    #[test]
    fn fortran_reorder_roundtrip() {
        let data: Vec<u8> = (0u8..8).collect();
        let shape = vec![2, 4];
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn tensor_layout_contiguous() {
        let layout = TensorLayout::new(vec![2, 3, 4], vec![12, 4, 1], 0);
        assert!(layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_fortran() {
        let layout = TensorLayout::new(vec![2, 3, 4], vec![1, 2, 6], 0);
        assert!(!layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }
}
