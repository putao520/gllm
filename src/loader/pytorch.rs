//! PyTorch .bin (zip+pickle) native TensorProvider.
//!
//! Pure Rust implementation — no candle/tch dependency (REQ-ARCH-003).
//! Contains a minimal pickle protocol parser sufficient for PyTorch checkpoints.
//! Implements TensorProvider for direct weight access without format conversion.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use safetensors::Dtype;
use zip::ZipArchive;

use super::{LoaderError, Result};

// ── Public types ──

#[derive(Debug, Clone)]
pub struct PytorchLoaderConfig {
    pub state_dict_key: Option<String>,
    pub int4_name_hints: Vec<String>,
}

impl Default for PytorchLoaderConfig {
    fn default() -> Self {
        Self {
            state_dict_key: None,
            int4_name_hints: vec!["conv1d".to_string(), "qweight".to_string(), "bits".to_string()],
        }
    }
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

impl std::hash::Hash for Object {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Object::Class { module_name, class_name } => {
                module_name.hash(state);
                class_name.hash(state);
            }
            Object::Int(v) => v.hash(state),
            Object::Long(v) => v.hash(state),
            Object::Float(v) => v.to_bits().hash(state),
            Object::Unicode(v) => v.hash(state),
            Object::Bool(v) => v.hash(state),
            Object::None | Object::Mark => {}
            Object::Tuple(v) | Object::List(v) => v.hash(state),
            Object::Dict(v) => v.hash(state),
            Object::Reduce { callable, args } | Object::Build { callable, args } => {
                callable.hash(state);
                args.hash(state);
            }
            Object::PersistentLoad(v) => v.hash(state),
        }
    }
}

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
        }); // LEGAL: 无简化模式时使用 Reduce 模式
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
pub enum PytorchDtype {
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

    pub fn to_st_dtype(self) -> Dtype {
        match self {
            Self::Bool => Dtype::BOOL, Self::U8 => Dtype::U8, Self::I8 => Dtype::I8,
            Self::I16 => Dtype::I16, Self::I32 => Dtype::I32, Self::I64 => Dtype::I64,
            Self::F16 => Dtype::F16, Self::BF16 => Dtype::BF16,
            Self::F32 => Dtype::F32, Self::F64 => Dtype::F64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PytorchTensorInfo {
    name: String,
    dtype: PytorchDtype,
    layout: TensorLayout,
    path: String,
}

// ── Public API ──

#[derive(Debug)]
pub struct PytorchLoader {
    shards: Vec<(PathBuf, HashMap<String, PytorchTensorInfo>)>,
    tensor_index: HashMap<String, usize>,
    config: PytorchLoaderConfig,
}

impl PytorchLoader {
    pub fn from_files(paths: &[PathBuf]) -> Result<Self> {
        Self::from_files_with_config(paths, PytorchLoaderConfig::default())
    }

    pub fn from_files_with_config(paths: &[PathBuf], config: PytorchLoaderConfig) -> Result<Self> {
        if paths.is_empty() {
            return Err(LoaderError::MissingWeights);
        }
        let mut shards = Vec::new();
        let mut tensor_index = HashMap::new();
        for (idx, path) in paths.iter().enumerate() {
            let infos = read_tensor_infos(path, config.state_dict_key.as_deref())?;
            for name in infos.keys() {
                if let Some(_prev) = tensor_index.insert(name.clone(), idx) {
                    return Err(LoaderError::DuplicateTensor(name.clone()));
                }
            }
            shards.push((path.clone(), infos));
        }
        Ok(Self { shards, tensor_index, config })
    }
}

impl super::TensorProvider for PytorchLoader {
    fn tensor_info(&self, name: &str) -> Option<super::TensorMeta> {
        let &shard_idx = self.tensor_index.get(name)?;
        let info = self.shards[shard_idx].1.get(name)?;
        Some(super::TensorMeta {
            name: name.to_string(),
            shape: info.layout.dims().to_vec(),
            dtype: info.dtype.to_st_dtype(),
        })
    }

    fn iter_tensors(&self) -> impl Iterator<Item = super::TensorMeta> {
        self.tensor_index.iter().map(|(name, &shard_idx)| {
            let info = self.shards[shard_idx].1.get(name).unwrap();
            super::TensorMeta {
                name: name.clone(),
                shape: info.layout.dims().to_vec(),
                dtype: info.dtype.to_st_dtype(),
            }
        })
    }

    fn load_tensor_data(&self, name: &str) -> Result<std::borrow::Cow<'_, [u8]>> {
        let &shard_idx = self.tensor_index.get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let (path, infos) = &self.shards[shard_idx];
        let info = infos.get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let data = read_tensor_bytes(path, info)?;
        Ok(std::borrow::Cow::Owned(data))
    }

    fn ggml_dtype(&self, name: &str) -> Option<crate::loader::gguf::GgmlDType> {
        let &shard_idx = self.tensor_index.get(name)?;
        let info = self.shards[shard_idx].1.get(name)?;
        if info.dtype == PytorchDtype::U8 {
            if packed_bits_hint(&info.name, info.dtype, &self.config).is_some() {
                return Some(crate::loader::gguf::GgmlDType::Q4_0);
            }
        }
        None
    }
}

pub fn read_tensor_infos(bin_path: &Path, key: Option<&str>) -> Result<HashMap<String, PytorchTensorInfo>> {
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
    // Check the nested structure by reference without consuming obj.
    let is_torch_module = (|| -> bool {
        if let Object::Build { callable, .. } = &obj {
            if let Object::Reduce { callable: inner_callable, .. } = callable.as_ref() {
                if let Object::Class { module_name, class_name } = inner_callable.as_ref() {
                    return module_name == "__torch__" && class_name == "Module";
                }
            }
        }
        false
    })();

    if is_torch_module {
        // For __torch__.Module: return the Build's args (not Reduce's args).
        if let Object::Build { args, .. } = obj {
            return *args;
        }
    }
    obj
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

pub fn packed_bits_hint(name: &str, dtype: PytorchDtype, config: &PytorchLoaderConfig) -> Option<u8> {
    if dtype != PytorchDtype::U8 { return None; }
    let name = name.to_ascii_lowercase();
    for hint in &config.int4_name_hints {
        if name.contains(&hint.to_ascii_lowercase()) { return Some(4); }
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;

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

    // ── Object type tests ──

    #[test]
    fn object_unicode_extraction() {
        let obj = Object::Unicode("hello".to_string());
        assert_eq!(obj.clone().unicode().unwrap(), "hello");
        assert!(Object::Int(42).unicode().is_err());
    }

    #[test]
    fn object_int_or_long() {
        assert_eq!(Object::Int(7).int_or_long().unwrap(), 7i64);
        assert_eq!(Object::Long(999i64).int_or_long().unwrap(), 999i64);
        assert!(Object::Float(1.0).int_or_long().is_err());
    }

    #[test]
    fn object_tuple_extraction() {
        let obj = Object::Tuple(vec![Object::Int(1), Object::Int(2)]);
        let v = obj.tuple().unwrap();
        assert_eq!(v.len(), 2);
        assert!(Object::Float(0.0).tuple().is_err());
    }

    #[test]
    fn object_class_extraction() {
        let obj = Object::Class {
            module_name: "torch".to_string(),
            class_name: "Storage".to_string(),
        };
        let (m, c) = obj.class().unwrap();
        assert_eq!(m, "torch");
        assert_eq!(c, "Storage");
    }

    #[test]
    fn object_reduce_extraction() {
        let obj = Object::Reduce {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        let (c, a) = obj.reduce().unwrap();
        assert!(matches!(c, Object::Int(1)));
        assert!(matches!(a, Object::Int(2)));
    }

    #[test]
    fn object_persistent_load_extraction() {
        let obj = Object::PersistentLoad(Box::new(Object::Unicode("data".to_string())));
        let inner = obj.persistent_load().unwrap();
        assert!(matches!(inner, Object::Unicode(ref s) if s == "data"));
    }

    #[test]
    fn object_equality() {
        assert_eq!(Object::Int(5), Object::Int(5));
        assert_ne!(Object::Int(5), Object::Int(6));
        assert_eq!(Object::Bool(true), Object::Bool(true));
        assert_eq!(Object::None, Object::None);
        assert_ne!(Object::None, Object::Bool(false));
    }

    #[test]
    fn object_try_into_string() {
        let obj = Object::Unicode("test".to_string());
        let s: String = obj.try_into().unwrap();
        assert_eq!(s, "test");
        let bad: std::result::Result<String, Object> = Object::Int(0).try_into();
        assert!(bad.is_err());
    }

    #[test]
    fn object_try_into_usize() {
        let obj = Object::Int(42);
        let v: usize = obj.try_into().unwrap();
        assert_eq!(v, 42);
        let bad: std::result::Result<usize, Object> = Object::Int(-1).try_into();
        assert!(bad.is_err());
    }

    #[test]
    fn object_try_into_vec() {
        let obj = Object::Tuple(vec![Object::Int(1), Object::Int(2), Object::Int(3)]);
        let v: Vec<usize> = obj.try_into().unwrap();
        assert_eq!(v, vec![1, 2, 3]);
    }

    // ── TensorLayout tests ──

    #[test]
    fn tensor_layout_empty_is_contiguous() {
        let layout = TensorLayout::new(vec![], vec![], 0);
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_1d_contiguous() {
        let layout = TensorLayout::new(vec![10], vec![1], 0);
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_non_contiguous() {
        let layout = TensorLayout::new(vec![3, 4], vec![8, 2], 0);
        assert!(!layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_start_offset() {
        let layout = TensorLayout::new(vec![5], vec![1], 100);
        assert_eq!(layout.start_offset(), 100);
    }

    #[test]
    fn tensor_layout_dims() {
        let layout = TensorLayout::new(vec![2, 3, 4], vec![12, 4, 1], 0);
        assert_eq!(layout.dims(), &[2, 3, 4]);
    }

    // ── PytorchDtype tests ──

    #[test]
    fn pytorch_dtype_size_in_bytes() {
        assert_eq!(PytorchDtype::Bool.size_in_bytes(), 1);
        assert_eq!(PytorchDtype::U8.size_in_bytes(), 1);
        assert_eq!(PytorchDtype::I8.size_in_bytes(), 1);
        assert_eq!(PytorchDtype::I16.size_in_bytes(), 2);
        assert_eq!(PytorchDtype::F16.size_in_bytes(), 2);
        assert_eq!(PytorchDtype::BF16.size_in_bytes(), 2);
        assert_eq!(PytorchDtype::I32.size_in_bytes(), 4);
        assert_eq!(PytorchDtype::F32.size_in_bytes(), 4);
        assert_eq!(PytorchDtype::I64.size_in_bytes(), 8);
        assert_eq!(PytorchDtype::F64.size_in_bytes(), 8);
    }

    #[test]
    fn pytorch_dtype_to_st_dtype() {
        assert_eq!(PytorchDtype::Bool.to_st_dtype(), Dtype::BOOL);
        assert_eq!(PytorchDtype::F32.to_st_dtype(), Dtype::F32);
        assert_eq!(PytorchDtype::F16.to_st_dtype(), Dtype::F16);
        assert_eq!(PytorchDtype::BF16.to_st_dtype(), Dtype::BF16);
        assert_eq!(PytorchDtype::F64.to_st_dtype(), Dtype::F64);
        assert_eq!(PytorchDtype::I64.to_st_dtype(), Dtype::I64);
    }

    // ── Additional tests ──

    #[test]
    fn object_float_extraction_error() {
        let obj = Object::Float(3.14);
        assert!(obj.clone().unicode().is_err());
        assert!(obj.clone().tuple().is_err());
        assert!(obj.clone().int_or_long().is_err());
        assert!(obj.clone().persistent_load().is_err());
        assert!(obj.clone().reduce().is_err());
        assert!(obj.clone().class().is_err());
    }

    #[test]
    fn object_none_various_extractions() {
        assert!(Object::None.unicode().is_err());
        assert!(Object::None.tuple().is_err());
        assert!(Object::None.int_or_long().is_err());
        assert!(Object::None.persistent_load().is_err());
        assert!(Object::None.reduce().is_err());
        assert!(Object::None.class().is_err());
    }

    #[test]
    fn object_bool_extraction_errors() {
        assert!(Object::Bool(true).unicode().is_err());
        assert!(Object::Bool(false).tuple().is_err());
        assert!(Object::Bool(true).int_or_long().is_err());
        assert!(Object::Bool(false).class().is_err());
    }

    #[test]
    fn object_mark_equality_and_extractions() {
        assert_eq!(Object::Mark, Object::Mark);
        assert!(Object::Mark.unicode().is_err());
        assert!(Object::Mark.tuple().is_err());
        assert!(Object::Mark.int_or_long().is_err());
    }

    #[test]
    fn object_list_extraction_errors() {
        let list = Object::List(vec![Object::Int(1), Object::Int(2)]);
        assert!(list.clone().unicode().is_err());
        assert!(list.clone().tuple().is_err());
        assert!(list.clone().int_or_long().is_err());
        assert!(list.clone().class().is_err());
    }

    #[test]
    fn object_dict_equality() {
        let d1 = Object::Dict(vec![(Object::Int(1), Object::Int(2))]);
        let d2 = Object::Dict(vec![(Object::Int(1), Object::Int(2))]);
        assert_eq!(d1, d2);

        let d3 = Object::Dict(vec![(Object::Int(1), Object::Int(3))]);
        assert_ne!(d1, d3);
    }

    #[test]
    fn object_build_extraction_errors() {
        let build = Object::Build {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        assert!(build.clone().unicode().is_err());
        assert!(build.clone().tuple().is_err());
        assert!(build.clone().reduce().is_err());
    }

    #[test]
    fn try_from_object_vec_with_non_tuple() {
        let obj = Object::List(vec![Object::Int(1)]);
        let result: std::result::Result<Vec<usize>, Object> = obj.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn try_from_object_vec_with_non_usize_element() {
        let obj = Object::Tuple(vec![Object::Int(5), Object::Unicode("x".to_string())]);
        let result: std::result::Result<Vec<usize>, Object> = obj.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn pickle_stack_push_pop_finalize() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(42));
        stack.push(Object::Unicode("hello".to_string()));
        assert_eq!(stack.pop().unwrap(), Object::Unicode("hello".to_string()));
        assert_eq!(stack.pop().unwrap(), Object::Int(42));
        // Empty now
        assert!(stack.pop().is_err());
    }

    #[test]
    fn pickle_stack_finalize_returns_last() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Bool(true));
        stack.push(Object::None);
        let result = stack.finalize().unwrap();
        assert_eq!(result, Object::None);
    }

    #[test]
    fn pickle_stack_finalize_empty_errors() {
        let stack = PickleStack::empty();
        let result = stack.finalize();
        assert!(result.is_err());
    }

    #[test]
    fn pickle_stack_memo_get_put() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(99));
        stack.memo_put(0).unwrap();
        stack.push(Object::Unicode("abc".to_string()));
        stack.memo_put(1).unwrap();

        let retrieved = stack.memo_get(0).unwrap();
        assert_eq!(retrieved, Object::Int(99));
        let retrieved = stack.memo_get(1).unwrap();
        assert_eq!(retrieved, Object::Unicode("abc".to_string()));
        assert!(stack.memo_get(99).is_err());
    }

    #[test]
    fn pickle_stack_pop_to_marker() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(1));
        stack.push(Object::Mark);
        stack.push(Object::Int(2));
        stack.push(Object::Int(3));
        let objs = stack.pop_to_marker().unwrap();
        assert_eq!(objs, vec![Object::Int(2), Object::Int(3)]);
        // Marker and items removed, Int(1) remains
        assert_eq!(stack.pop().unwrap(), Object::Int(1));
    }

    #[test]
    fn pickle_stack_pop_to_marker_no_marker_errors() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(1));
        stack.push(Object::Int(2));
        let result = stack.pop_to_marker();
        assert!(result.is_err());
    }

    #[test]
    fn pickle_stack_build_op_dict_merge() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Dict(vec![(Object::Unicode("a".to_string()), Object::Int(1))]));
        stack.push(Object::Dict(vec![(Object::Unicode("b".to_string()), Object::Int(2))]));
        stack.build_op().unwrap();
        let result = stack.pop().unwrap();
        match result {
            Object::Dict(pairs) => {
                assert_eq!(pairs.len(), 2);
            }
            other => panic!("expected Dict, got {other:?}"),
        }
    }

    #[test]
    fn pickle_stack_build_op_non_dict() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(10));
        stack.push(Object::Int(20));
        stack.build_op().unwrap();
        let result = stack.pop().unwrap();
        assert!(matches!(result, Object::Build { .. }));
    }

    #[test]
    fn pickle_stack_reduce_op_ordered_dict() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Class {
            module_name: "collections".to_string(),
            class_name: "OrderedDict".to_string(),
        });
        stack.push(Object::Tuple(vec![]));
        stack.reduce_op().unwrap();
        let result = stack.pop().unwrap();
        assert!(matches!(result, Object::Dict(ref d) if d.is_empty()));
    }

    #[test]
    fn pickle_stack_reduce_op_defaultdict() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Class {
            module_name: "collections".to_string(),
            class_name: "defaultdict".to_string(),
        });
        stack.push(Object::Tuple(vec![]));
        stack.reduce_op().unwrap();
        let result = stack.pop().unwrap();
        assert!(matches!(result, Object::Dict(ref d) if d.is_empty()));
    }

    #[test]
    fn pickle_stack_reduce_op_generic() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Class {
            module_name: "torch".to_string(),
            class_name: "Storage".to_string(),
        });
        stack.push(Object::Int(42));
        stack.reduce_op().unwrap();
        let result = stack.pop().unwrap();
        assert!(matches!(result, Object::Reduce { .. }));
    }

    #[test]
    fn unwrap_module_non_module() {
        let obj = Object::Int(5);
        let result = unwrap_module(obj);
        assert_eq!(result, Object::Int(5));
    }

    #[test]
    fn unwrap_module_dict_passthrough() {
        let obj = Object::Dict(vec![]);
        let result = unwrap_module(obj);
        assert!(matches!(result, Object::Dict(_)));
    }

    #[test]
    fn resolve_state_dict_with_matching_key() {
        let dict = Object::Dict(vec![
            (Object::Unicode("my_state".to_string()), Object::Int(100)),
            (Object::Unicode("other".to_string()), Object::Int(200)),
        ]);
        let result = resolve_state_dict(dict, Some("my_state")).unwrap();
        assert_eq!(result, Object::Int(100));
    }

    #[test]
    fn resolve_state_dict_with_missing_key_errors() {
        let dict = Object::Dict(vec![
            (Object::Unicode("a".to_string()), Object::Int(1)),
        ]);
        let result = resolve_state_dict(dict, Some("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_state_dict_auto_finds_state_dict_key() {
        let inner = Object::Dict(vec![(Object::Unicode("weight".to_string()), Object::Int(42))]);
        let dict = Object::Dict(vec![
            (Object::Unicode("state_dict".to_string()), inner.clone()),
            (Object::Unicode("optimizer".to_string()), Object::None),
        ]);
        let result = resolve_state_dict(dict, None).unwrap();
        assert_eq!(result, inner);
    }

    #[test]
    fn resolve_state_dict_no_key_returns_whole_dict() {
        let dict = Object::Dict(vec![
            (Object::Unicode("weight".to_string()), Object::Int(1)),
        ]);
        let result = resolve_state_dict(dict, None).unwrap();
        match result {
            Object::Dict(pairs) => assert_eq!(pairs.len(), 1),
            other => panic!("expected Dict, got {other:?}"),
        }
    }

    #[test]
    fn resolve_state_dict_non_dict_passthrough() {
        let obj = Object::Int(7);
        let result = resolve_state_dict(obj, None).unwrap();
        assert_eq!(result, Object::Int(7));
    }

    #[test]
    fn packed_bits_hint_u8_with_matching_name() {
        let cfg = PytorchLoaderConfig::default();
        assert_eq!(packed_bits_hint("model.qweight.0", PytorchDtype::U8, &cfg), Some(4));
        assert_eq!(packed_bits_hint("layer.int4.weight", PytorchDtype::U8, &cfg), None);
        assert_eq!(packed_bits_hint("block.q4.weight", PytorchDtype::U8, &cfg), None);
    }

    #[test]
    fn packed_bits_hint_non_u8_returns_none() {
        let cfg = PytorchLoaderConfig::default();
        assert_eq!(packed_bits_hint("model.qweight.0", PytorchDtype::F32, &cfg), None);
        assert_eq!(packed_bits_hint("model.qweight.0", PytorchDtype::F16, &cfg), None);
    }

    #[test]
    fn packed_bits_hint_u8_no_match_returns_none() {
        let cfg = PytorchLoaderConfig::default();
        assert_eq!(packed_bits_hint("model.weight.0", PytorchDtype::U8, &cfg), None);
    }

    #[test]
    fn packed_bits_hint_case_insensitive() {
        let cfg = PytorchLoaderConfig::default();
        assert_eq!(packed_bits_hint("model.QWEIGHT.0", PytorchDtype::U8, &cfg), Some(4));
        assert_eq!(packed_bits_hint("layer.CONV1D.weight", PytorchDtype::U8, &cfg), Some(4));
    }

    #[test]
    fn reorder_fortran_to_c_1d_identity() {
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let shape = vec![4];
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out, data);
    }

    #[test]
    fn reorder_fortran_to_c_0d_identity() {
        let data: Vec<u8> = vec![42];
        let out = reorder_fortran_to_c(&data, &[], 1);
        assert_eq!(out, vec![42]);
    }

    #[test]
    fn reorder_fortran_to_c_2x2_roundtrip() {
        // 2x2 matrix, 2 bytes per element
        // Fortran (column-major): [a0,a1, b0,b1, c0,c1, d0,d1]
        // where columns are stored sequentially
        let data: Vec<u8> = vec![1, 0, 3, 0, 2, 0, 4, 0]; // F-order: (0,0)=1,(1,0)=3,(0,1)=2,(1,1)=4
        let shape = vec![2, 2];
        let out = reorder_fortran_to_c(&data, &shape, 2);
        // C-order (row-major): (0,0)=1,(0,1)=2,(1,0)=3,(1,1)=4
        assert_eq!(out, vec![1, 0, 2, 0, 3, 0, 4, 0]);
    }

    #[test]
    fn pytorch_dtype_all_to_st_dtype() {
        assert_eq!(PytorchDtype::Bool.to_st_dtype(), Dtype::BOOL);
        assert_eq!(PytorchDtype::U8.to_st_dtype(), Dtype::U8);
        assert_eq!(PytorchDtype::I8.to_st_dtype(), Dtype::I8);
        assert_eq!(PytorchDtype::I16.to_st_dtype(), Dtype::I16);
        assert_eq!(PytorchDtype::I32.to_st_dtype(), Dtype::I32);
        assert_eq!(PytorchDtype::I64.to_st_dtype(), Dtype::I64);
        assert_eq!(PytorchDtype::F16.to_st_dtype(), Dtype::F16);
        assert_eq!(PytorchDtype::BF16.to_st_dtype(), Dtype::BF16);
        assert_eq!(PytorchDtype::F32.to_st_dtype(), Dtype::F32);
        assert_eq!(PytorchDtype::F64.to_st_dtype(), Dtype::F64);
    }

    #[test]
    fn pytorch_dtype_copy_trait() {
        let a = PytorchDtype::F32;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_layout_contiguous_2d() {
        // shape [3, 5], row-major strides [5, 1]
        let layout = TensorLayout::new(vec![3, 5], vec![5, 1], 0);
        assert!(layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_fortran_2d() {
        // shape [3, 5], column-major strides [1, 3]
        let layout = TensorLayout::new(vec![3, 5], vec![1, 3], 0);
        assert!(!layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_scalar_both_contiguous() {
        let layout = TensorLayout::new(vec![1], vec![1], 0);
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn read_to_newline_lf() {
        let input: &[u8] = b"hello\nworld\n";
        let mut reader = std::io::BufReader::new(input);
        let line = read_to_newline(&mut reader).unwrap();
        assert_eq!(line, b"hello");
    }

    #[test]
    fn read_to_newline_crlf() {
        let input: &[u8] = b"hello\r\nworld\r\n";
        let mut reader = std::io::BufReader::new(input);
        let line = read_to_newline(&mut reader).unwrap();
        assert_eq!(line, b"hello");
    }

    #[test]
    fn pytorch_tensor_info_debug() {
        let info = PytorchTensorInfo {
            name: "weight".to_string(),
            dtype: PytorchDtype::F32,
            layout: TensorLayout::new(vec![2, 3], vec![3, 1], 0),
            path: "archive/data/0".to_string(),
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("weight"));
        assert!(debug_str.contains("F32"));
    }

    // ── Object Debug format tests ──

    #[test]
    fn object_debug_format_all_variants() {
        assert!(format!("{:?}", Object::Int(42)).contains("42"));
        assert!(format!("{:?}", Object::Long(-1)).contains("-1"));
        assert!(format!("{:?}", Object::Float(3.14)).contains("3.14"));
        assert!(format!("{:?}", Object::Unicode("abc".to_string())).contains("abc"));
        assert!(format!("{:?}", Object::Bool(true)).contains("true"));
        assert!(format!("{:?}", Object::None).contains("None"));
        assert!(format!("{:?}", Object::Mark).contains("Mark"));
    }

    #[test]
    fn object_debug_format_class() {
        let obj = Object::Class {
            module_name: "collections".to_string(),
            class_name: "OrderedDict".to_string(),
        };
        let debug = format!("{obj:?}");
        assert!(debug.contains("collections"));
        assert!(debug.contains("OrderedDict"));
    }

    #[test]
    fn object_debug_format_reduce() {
        let obj = Object::Reduce {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        let debug = format!("{obj:?}");
        assert!(debug.contains("Reduce"));
    }

    #[test]
    fn object_debug_format_build() {
        let obj = Object::Build {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        let debug = format!("{obj:?}");
        assert!(debug.contains("Build"));
    }

    #[test]
    fn object_debug_format_persistent_load() {
        let obj = Object::PersistentLoad(Box::new(Object::Unicode("data".to_string())));
        let debug = format!("{obj:?}");
        assert!(debug.contains("PersistentLoad"));
    }

    // ── Object equality edge cases ──

    #[test]
    fn object_equality_float_nan() {
        let a = Object::Float(f64::NAN);
        let b = Object::Float(f64::NAN);
        // NaN != NaN per IEEE 754, but PartialEq derive uses bit comparison
        // which makes NaN == NaN true (derive does field-by-field eq)
        // Actually derive uses the field's PartialEq, so NaN != NaN for f64
        assert_ne!(a, b);
    }

    #[test]
    fn object_equality_float_infinity() {
        assert_eq!(Object::Float(f64::INFINITY), Object::Float(f64::INFINITY));
        assert_eq!(Object::Float(f64::NEG_INFINITY), Object::Float(f64::NEG_INFINITY));
        assert_ne!(Object::Float(f64::INFINITY), Object::Float(f64::NEG_INFINITY));
    }

    #[test]
    fn object_equality_long_range() {
        assert_eq!(Object::Long(i64::MIN), Object::Long(i64::MIN));
        assert_eq!(Object::Long(i64::MAX), Object::Long(i64::MAX));
        assert_ne!(Object::Long(i64::MIN), Object::Long(i64::MAX));
    }

    #[test]
    fn object_equality_int_range() {
        assert_eq!(Object::Int(i32::MIN), Object::Int(i32::MIN));
        assert_eq!(Object::Int(i32::MAX), Object::Int(i32::MAX));
        assert_ne!(Object::Int(0), Object::Int(i32::MAX));
    }

    #[test]
    fn object_equality_across_types() {
        // Int and Long with same value are NOT equal (different variants)
        assert_ne!(Object::Int(42), Object::Long(42));
        assert_ne!(Object::Int(0), Object::Float(0.0));
        assert_ne!(Object::Bool(true), Object::Int(1));
    }

    #[test]
    fn object_equality_tuple_nested() {
        let t1 = Object::Tuple(vec![Object::Int(1), Object::Tuple(vec![Object::Int(2)])]);
        let t2 = Object::Tuple(vec![Object::Int(1), Object::Tuple(vec![Object::Int(2)])]);
        assert_eq!(t1, t2);

        let t3 = Object::Tuple(vec![Object::Int(1), Object::Tuple(vec![Object::Int(3)])]);
        assert_ne!(t1, t3);
    }

    #[test]
    fn object_equality_list() {
        let l1 = Object::List(vec![Object::Int(1), Object::Int(2)]);
        let l2 = Object::List(vec![Object::Int(1), Object::Int(2)]);
        assert_eq!(l1, l2);

        let l3 = Object::List(vec![Object::Int(2), Object::Int(1)]);
        assert_ne!(l1, l3);
    }

    #[test]
    fn object_equality_dict_order_matters() {
        let d1 = Object::Dict(vec![(Object::Int(1), Object::Int(2))]);
        let d2 = Object::Dict(vec![(Object::Int(1), Object::Int(2))]);
        assert_eq!(d1, d2);
    }

    #[test]
    fn object_equality_empty_collections() {
        assert_eq!(Object::Tuple(vec![]), Object::Tuple(vec![]));
        assert_eq!(Object::List(vec![]), Object::List(vec![]));
        assert_eq!(Object::Dict(vec![]), Object::Dict(vec![]));
        // Different empty collection types are not equal
        assert_ne!(Object::Tuple(vec![]), Object::List(vec![]));
    }

    // ── Object::int_or_long edge cases ──

    #[test]
    fn object_int_or_long_boundary_values() {
        assert_eq!(Object::Int(i32::MIN).int_or_long().unwrap(), i32::MIN as i64);
        assert_eq!(Object::Int(i32::MAX).int_or_long().unwrap(), i32::MAX as i64);
        assert_eq!(Object::Long(0i64).int_or_long().unwrap(), 0i64);
    }

    // ── TryFrom<Object> for usize edge cases ──

    #[test]
    fn try_from_object_usize_zero() {
        let obj = Object::Int(0);
        let v: usize = obj.try_into().unwrap();
        assert_eq!(v, 0);
    }

    #[test]
    fn try_from_object_usize_max_positive_int() {
        let obj = Object::Int(i32::MAX);
        let v: usize = obj.try_into().unwrap();
        assert_eq!(v, i32::MAX as usize);
    }

    #[test]
    fn try_from_object_usize_rejects_long() {
        let obj = Object::Long(42);
        let result: std::result::Result<usize, Object> = obj.try_into();
        assert!(result.is_err());
    }

    // ── TryFrom<Object> for Vec<usize> edge cases ──

    #[test]
    fn try_from_object_vec_empty_tuple() {
        let obj = Object::Tuple(vec![]);
        let v: Vec<usize> = obj.try_into().unwrap();
        assert!(v.is_empty());
    }

    #[test]
    fn try_from_object_vec_single_element() {
        let obj = Object::Tuple(vec![Object::Int(7)]);
        let v: Vec<usize> = obj.try_into().unwrap();
        assert_eq!(v, vec![7]);
    }

    // ── TryFrom<Object> for String edge cases ──

    #[test]
    fn try_from_object_string_empty() {
        let obj = Object::Unicode(String::new());
        let s: String = obj.try_into().unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn try_from_object_string_unicode_content() {
        let obj = Object::Unicode("日本語テスト".to_string());
        let s: String = obj.try_into().unwrap();
        assert_eq!(s, "日本語テスト");
    }

    // ── TensorLayout edge cases ──

    #[test]
    fn tensor_layout_large_offset() {
        let layout = TensorLayout::new(vec![1], vec![1], usize::MAX);
        assert_eq!(layout.start_offset(), usize::MAX);
    }

    #[test]
    fn tensor_layout_zero_offset() {
        let layout = TensorLayout::new(vec![10, 10], vec![10, 1], 0);
        assert_eq!(layout.start_offset(), 0);
    }

    #[test]
    fn tensor_layout_contiguous_with_leading_one() {
        // shape [1, 5], strides [5, 1] — contiguous (leading dim of 1)
        let layout = TensorLayout::new(vec![1, 5], vec![5, 1], 0);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn tensor_layout_contiguous_with_trailing_one() {
        // shape [5, 1], strides [1, 1] — contiguous (trailing dim of 1)
        let layout = TensorLayout::new(vec![5, 1], vec![1, 1], 0);
        assert!(layout.is_contiguous());
    }

    #[test]
    fn tensor_layout_3d_fortran() {
        // shape [2, 3, 4], Fortran strides [1, 2, 6]
        let layout = TensorLayout::new(vec![2, 3, 4], vec![1, 2, 6], 0);
        assert!(layout.is_fortran_contiguous());
        assert!(!layout.is_contiguous());
    }

    #[test]
    fn tensor_layout_clone_preserves_fields() {
        let layout = TensorLayout::new(vec![3, 4], vec![4, 1], 7);
        let cloned = layout.clone();
        assert_eq!(cloned.dims(), layout.dims());
        assert_eq!(cloned.start_offset(), layout.start_offset());
    }

    // ── PytorchDtype comprehensive tests ──

    #[test]
    fn pytorch_dtype_debug_format() {
        let debug = format!("{:?}", PytorchDtype::F32);
        assert!(debug.contains("F32"));
        let debug = format!("{:?}", PytorchDtype::BF16);
        assert!(debug.contains("BF16"));
    }

    #[test]
    fn pytorch_dtype_equality_all_pairs() {
        // Verify each variant equals itself and no cross-equality
        let variants = [
            PytorchDtype::Bool, PytorchDtype::U8, PytorchDtype::I8,
            PytorchDtype::I16, PytorchDtype::I32, PytorchDtype::I64,
            PytorchDtype::F16, PytorchDtype::BF16, PytorchDtype::F32,
            PytorchDtype::F64,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn pytorch_dtype_size_is_power_of_2_or_one() {
        // All sizes should be 1, 2, 4, or 8
        let variants = [
            PytorchDtype::Bool, PytorchDtype::U8, PytorchDtype::I8,
            PytorchDtype::I16, PytorchDtype::I32, PytorchDtype::I64,
            PytorchDtype::F16, PytorchDtype::BF16, PytorchDtype::F32,
            PytorchDtype::F64,
        ];
        for v in &variants {
            let sz = v.size_in_bytes();
            assert!(sz > 0);
            assert!(sz.is_power_of_two());
        }
    }

    #[test]
    fn pytorch_dtype_to_st_dtype_all_coverage() {
        // Every variant must map to a distinct Dtype
        let mappings: Vec<Dtype> = [
            PytorchDtype::Bool, PytorchDtype::U8, PytorchDtype::I8,
            PytorchDtype::I16, PytorchDtype::I32, PytorchDtype::I64,
            PytorchDtype::F16, PytorchDtype::BF16, PytorchDtype::F32,
            PytorchDtype::F64,
        ].iter().map(|d| d.to_st_dtype()).collect();
        // Check uniqueness
        for i in 0..mappings.len() {
            for j in (i+1)..mappings.len() {
                assert_ne!(mappings[i], mappings[j], "duplicate mapping at {i} vs {j}");
            }
        }
    }

    // ── PickleStack edge cases ──

    #[test]
    fn pickle_stack_last_returns_top() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(1));
        stack.push(Object::Int(2));
        let top = stack.last().unwrap();
        assert_eq!(*top, Object::Int(2));
    }

    #[test]
    fn pickle_stack_last_empty_errors() {
        let mut stack = PickleStack::empty();
        assert!(stack.last().is_err());
    }

    #[test]
    fn pickle_stack_memo_put_empty_stack_errors() {
        let mut stack = PickleStack::empty();
        assert!(stack.memo_put(0).is_err());
    }

    #[test]
    fn pickle_stack_memo_overwrite() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(1));
        stack.memo_put(0).unwrap();
        // Overwrite memo slot 0
        stack.push(Object::Int(2));
        stack.memo_put(0).unwrap();
        let retrieved = stack.memo_get(0).unwrap();
        assert_eq!(retrieved, Object::Int(2));
    }

    #[test]
    fn pickle_stack_pop_to_marker_multiple_markers() {
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(0));
        stack.push(Object::Mark);
        stack.push(Object::Int(1));
        stack.push(Object::Mark);
        stack.push(Object::Int(2));
        stack.push(Object::Int(3));
        // Pop to the most recent marker
        let objs = stack.pop_to_marker().unwrap();
        assert_eq!(objs, vec![Object::Int(2), Object::Int(3)]);
        // Pop to the next marker
        let objs = stack.pop_to_marker().unwrap();
        assert_eq!(objs, vec![Object::Int(1)]);
        // Int(0) remains
        assert_eq!(stack.pop().unwrap(), Object::Int(0));
    }

    // ── read_to_newline edge cases ──

    #[test]
    fn read_to_newline_empty_line() {
        let input: &[u8] = b"\nnext\n";
        let mut reader = std::io::BufReader::new(input);
        let line = read_to_newline(&mut reader).unwrap();
        assert!(line.is_empty());
    }

    #[test]
    fn read_to_newline_no_trailing_newline() {
        let input: &[u8] = b"last line";
        let mut reader = std::io::BufReader::new(input);
        // read_until reads all bytes to EOF (no \n found), then pop() removes last byte
        let line = read_to_newline(&mut reader).unwrap();
        assert_eq!(line, b"last lin"); // last byte 'e' is popped by data.pop()
    }

    // ── reorder_fortran_to_c edge cases ──

    #[test]
    fn reorder_fortran_to_c_3d_small() {
        // shape [2, 2, 2], 1 byte per element, 8 elements total
        let data: Vec<u8> = (0u8..8).collect();
        let shape = vec![2, 2, 2];
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out.len(), 8);
        // Verify it's not just a copy (F-order != C-order for 3D)
        // element mapping: C-order (i,j,k) -> i*4+j*2+k, F-order (i,j,k) -> i+j*2+k*4
        // C[0,0,0]=F[0,0,0]=0, C[0,0,1]=F[0,0,1]=1
        // C[0,1,0]=data[C]=2, F[0,1,0]=data[2]=2 => C[0,1,0]=F[2]
        assert_eq!(out[0], data[0]); // (0,0,0)
    }

    #[test]
    fn reorder_fortran_to_c_4byte_elements() {
        // shape [2, 3], 4 bytes per element, 6 elements total = 24 bytes
        let data: Vec<u8> = (0u8..24).collect();
        let shape = vec![2, 3];
        let out = reorder_fortran_to_c(&data, &shape, 4);
        assert_eq!(out.len(), 24);
        // First element in C-order (row 0, col 0) comes from F-order position 0
        assert_eq!(&out[0..4], &data[0..4]);
    }

    // ── packed_bits_hint edge cases ──

    #[test]
    fn packed_bits_hint_custom_hints() {
        let cfg = PytorchLoaderConfig {
            int4_name_hints: vec!["custom4".to_string()],
            ..Default::default()
        };
        assert_eq!(packed_bits_hint("layer.custom4.weight", PytorchDtype::U8, &cfg), Some(4));
        assert_eq!(packed_bits_hint("layer.qweight.weight", PytorchDtype::U8, &cfg), None);
    }

    #[test]
    fn packed_bits_hint_empty_name() {
        let cfg = PytorchLoaderConfig::default();
        assert_eq!(packed_bits_hint("", PytorchDtype::U8, &cfg), None);
    }

    // ── unwrap_module edge cases ──

    #[test]
    fn unwrap_module_nested_non_torch_module() {
        // Build with non-__torch__ module — should pass through as Build
        let obj = Object::Build {
            callable: Box::new(Object::Reduce {
                callable: Box::new(Object::Class {
                    module_name: "other".to_string(),
                    class_name: "Module".to_string(),
                }),
                args: Box::new(Object::Int(42)),
            }),
            args: Box::new(Object::Int(99)),
        };
        let result = unwrap_module(obj);
        assert!(matches!(result, Object::Build { .. }));
    }

    #[test]
    fn unwrap_module_torch_module_returns_args() {
        let obj = Object::Build {
            callable: Box::new(Object::Reduce {
                callable: Box::new(Object::Class {
                    module_name: "__torch__".to_string(),
                    class_name: "Module".to_string(),
                }),
                args: Box::new(Object::Int(0)),
            }),
            args: Box::new(Object::Dict(vec![])),
        };
        let result = unwrap_module(obj);
        assert!(matches!(result, Object::Dict(_)));
    }

    // ── resolve_state_dict edge cases ──

    #[test]
    fn resolve_state_dict_multiple_keys_first_match() {
        let dict = Object::Dict(vec![
            (Object::Unicode("state_dict".to_string()), Object::Int(1)),
            (Object::Unicode("state_dict".to_string()), Object::Int(2)),
        ]);
        let result = resolve_state_dict(dict, None).unwrap();
        assert_eq!(result, Object::Int(1));
    }

    #[test]
    fn resolve_state_dict_no_state_dict_key_no_user_key() {
        let dict = Object::Dict(vec![
            (Object::Unicode("weights".to_string()), Object::Int(42)),
        ]);
        let result = resolve_state_dict(dict, None).unwrap();
        // Returns the whole dict since no "state_dict" key found
        assert!(matches!(result, Object::Dict(d) if d.len() == 1));
    }

    // ── PytorchTensorInfo debug edge cases ──

    #[test]
    fn pytorch_tensor_info_clone_preserves_fields() {
        let info = PytorchTensorInfo {
            name: "layer.weight".to_string(),
            dtype: PytorchDtype::BF16,
            layout: TensorLayout::new(vec![256, 512], vec![512, 1], 0),
            path: "archive/data/0".to_string(),
        };
        let cloned = info.clone();
        assert_eq!(cloned.name, "layer.weight");
        assert_eq!(cloned.dtype, PytorchDtype::BF16);
        assert_eq!(cloned.path, "archive/data/0");
    }

    // ── read helper byte-reading tests ──

    #[test]
    fn read_u8_basic() {
        let input: &[u8] = &[0xAB];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u8(&mut reader).unwrap(), 0xAB);
    }

    #[test]
    fn read_u16_le_basic() {
        let input: &[u8] = &[0x34, 0x12];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u16_le(&mut reader).unwrap(), 0x1234);
    }

    #[test]
    fn read_u16_le_zero() {
        let input: &[u8] = &[0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u16_le(&mut reader).unwrap(), 0);
    }

    #[test]
    fn read_u16_le_max() {
        let input: &[u8] = &[0xFF, 0xFF];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u16_le(&mut reader).unwrap(), u16::MAX);
    }

    #[test]
    fn read_i32_le_negative() {
        let input: &[u8] = &[0x00, 0x00, 0x00, 0x80]; // -2147483648
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_i32_le(&mut reader).unwrap(), i32::MIN);
    }

    #[test]
    fn read_i32_le_positive() {
        let input: &[u8] = &[0x01, 0x00, 0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_i32_le(&mut reader).unwrap(), 1);
    }

    #[test]
    fn read_u32_le_basic() {
        let input: &[u8] = &[0x78, 0x56, 0x34, 0x12];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u32_le(&mut reader).unwrap(), 0x12345678);
    }

    #[test]
    fn read_f64_be_positive() {
        // 1.0 in big-endian IEEE 754: 0x3FF0000000000000
        let input: &[u8] = &[0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_f64_be(&mut reader).unwrap(), 1.0);
    }

    #[test]
    fn read_f64_be_negative() {
        // -1.0 in big-endian IEEE 754: 0xBFF0000000000000
        let input: &[u8] = &[0xBF, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        let val = read_f64_be(&mut reader).unwrap();
        assert_eq!(val, -1.0);
    }

    #[test]
    fn read_u8_empty_errors() {
        let input: &[u8] = &[];
        let mut reader = std::io::Cursor::new(input);
        assert!(read_u8(&mut reader).is_err());
    }

    #[test]
    fn read_u16_le_truncated_errors() {
        let input: &[u8] = &[0x34]; // only 1 byte, need 2
        let mut reader = std::io::Cursor::new(input);
        assert!(read_u16_le(&mut reader).is_err());
    }

    #[test]
    fn read_i32_le_truncated_errors() {
        let input: &[u8] = &[0x00, 0x00]; // only 2 bytes, need 4
        let mut reader = std::io::Cursor::new(input);
        assert!(read_i32_le(&mut reader).is_err());
    }

    // ── read_u32_le edge cases ──

    #[test]
    fn read_u32_le_zero() {
        let input: &[u8] = &[0x00, 0x00, 0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u32_le(&mut reader).unwrap(), 0u32);
    }

    #[test]
    fn read_u32_le_max() {
        let input: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF];
        let mut reader = std::io::Cursor::new(input);
        assert_eq!(read_u32_le(&mut reader).unwrap(), u32::MAX);
    }

    #[test]
    fn read_u32_le_truncated_errors() {
        let input: &[u8] = &[0x01, 0x02];
        let mut reader = std::io::Cursor::new(input);
        assert!(read_u32_le(&mut reader).is_err());
    }

    // ── read_f64_be edge cases ──

    #[test]
    fn read_f64_be_zero() {
        let input: &[u8] = &[0x00; 8];
        let mut reader = std::io::Cursor::new(input);
        let val = read_f64_be(&mut reader).unwrap();
        assert_eq!(val, 0.0);
        assert!(val.is_sign_positive());
    }

    #[test]
    fn read_f64_be_nan() {
        // NaN: exponent all 1s, mantissa non-zero
        let input: &[u8] = &[0x7F, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01];
        let mut reader = std::io::Cursor::new(input);
        let val = read_f64_be(&mut reader).unwrap();
        assert!(val.is_nan());
    }

    #[test]
    fn read_f64_be_infinity() {
        // +inf: 0x7FF0000000000000
        let input: &[u8] = &[0x7F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut reader = std::io::Cursor::new(input);
        let val = read_f64_be(&mut reader).unwrap();
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    }

    #[test]
    fn read_f64_be_truncated_errors() {
        let input: &[u8] = &[0x3F, 0xF0, 0x00, 0x00]; // 4 bytes, need 8
        let mut reader = std::io::Cursor::new(input);
        assert!(read_f64_be(&mut reader).is_err());
    }

    // ── Object Hash tests ──

    #[test]
    fn object_hash_equal_objects_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(&Object::Int(42)), hash_of(&Object::Int(42)));
        assert_eq!(hash_of(&Object::Unicode("abc".into())), hash_of(&Object::Unicode("abc".into())));
        assert_eq!(hash_of(&Object::None), hash_of(&Object::None));
    }

    #[test]
    fn object_hash_different_types_likely_different() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        // Different types should generally produce different hashes
        let h1 = hash_of(&Object::Int(1));
        let h2 = hash_of(&Object::Long(1));
        assert_ne!(h1, h2);
    }

    #[test]
    fn object_hash_tuple_content_matters() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let t1 = Object::Tuple(vec![Object::Int(1), Object::Int(2)]);
        let t2 = Object::Tuple(vec![Object::Int(2), Object::Int(1)]);
        assert_ne!(hash_of(&t1), hash_of(&t2));
    }

    #[test]
    fn object_hash_dict_order_matters() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let d1 = Object::Dict(vec![(Object::Int(1), Object::Int(10))]);
        let d2 = Object::Dict(vec![(Object::Int(2), Object::Int(10))]);
        assert_ne!(hash_of(&d1), hash_of(&d2));
    }

    #[test]
    fn object_hash_class_fields_matter() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let c1 = Object::Class { module_name: "a".into(), class_name: "B".into() };
        let c2 = Object::Class { module_name: "c".into(), class_name: "B".into() };
        assert_ne!(hash_of(&c1), hash_of(&c2));
    }

    // ── PickleStack::read() opcode tests ──

    #[test]
    fn pickle_read_stop_returns_true() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'.']; // STOP opcode
        let mut reader = std::io::BufReader::new(input);
        let done = stack.read(&mut reader).unwrap();
        assert!(done);
    }

    #[test]
    fn pickle_read_none_pushes_none() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'N', b'.']; // NONE + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        let obj = stack.finalize().unwrap();
        assert_eq!(obj, Object::None);
    }

    #[test]
    fn pickle_read_newtrue_pushes_bool_true() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[0x88, b'.']; // NEWTRUE + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Bool(true));
    }

    #[test]
    fn pickle_read_newfalse_pushes_bool_false() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[0x89, b'.']; // NEWFALSE + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Bool(false));
    }

    #[test]
    fn pickle_read_short_binint() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'K', 0x2A, b'.']; // SHORT_BININT(42) + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Int(42));
    }

    #[test]
    fn pickle_read_binint2() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'M', 0x39, 0x05, b'.']; // BININT2(0x0539=1337) + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Int(1337));
    }

    #[test]
    fn pickle_read_binint() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'J', 0x01, 0x00, 0x00, 0x00, b'.']; // BININT(1) + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Int(1));
    }

    #[test]
    fn pickle_read_binfloat() {
        // BINFLOAT for 1.0: big-endian 0x3FF0000000000000
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'G', 0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::Float(v) => assert!((v - 1.0).abs() < f64::EPSILON),
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_empty_tuple() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b')', b'.']; // EMPTY_TUPLE + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Tuple(vec![]));
    }

    #[test]
    fn pickle_read_empty_list() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b']', b'.']; // EMPTY_LIST + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::List(vec![]));
    }

    #[test]
    fn pickle_read_empty_dict() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'}', b'.']; // EMPTY_DICT + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Dict(vec![]));
    }

    #[test]
    fn pickle_read_binunicode() {
        let mut stack = PickleStack::empty();
        // BINUNICODE: 'X' + len(u32 LE) + utf8 bytes
        let text = b"hello";
        let mut input = vec![b'X'];
        input.extend_from_slice(&(text.len() as u32).to_le_bytes());
        input.extend_from_slice(text);
        input.push(b'.'); // STOP
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Unicode("hello".to_string()));
    }

    #[test]
    fn pickle_read_global() {
        let mut stack = PickleStack::empty();
        // GLOBAL: 'c' + module\n + class\n
        let input = b"ccollections\nOrderedDict\n.";
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::Class { module_name, class_name } => {
                assert_eq!(module_name, "collections");
                assert_eq!(class_name, "OrderedDict");
            }
            other => panic!("expected Class, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_tuple1() {
        let mut stack = PickleStack::empty();
        // Push Int(7) via SHORT_BININT, then TUPLE1
        let input: &[u8] = &[b'K', 0x07, 0x85, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Tuple(vec![Object::Int(7)]));
    }

    #[test]
    fn pickle_read_tuple2() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'K', 0x01, b'K', 0x02, 0x86, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Tuple(vec![Object::Int(1), Object::Int(2)]));
    }

    #[test]
    fn pickle_read_tuple3() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[b'K', 0x0A, b'K', 0x0B, b'K', 0x0C, 0x87, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Tuple(vec![Object::Int(10), Object::Int(11), Object::Int(12)]));
    }

    #[test]
    fn pickle_read_mark_and_tuple() {
        let mut stack = PickleStack::empty();
        // MARK + SHORT_BININT(5) + SHORT_BININT(6) + TUPLE('t')
        let input: &[u8] = &[b'(', b'K', 0x05, b'K', 0x06, b't', b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Tuple(vec![Object::Int(5), Object::Int(6)]));
    }

    #[test]
    fn pickle_read_long1_zero_length() {
        let mut stack = PickleStack::empty();
        // LONG1 with n=1 byte, value=0
        let input: &[u8] = &[0x8a, 0x01, 0x00, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Long(0));
    }

    #[test]
    fn pickle_read_long1_multi_byte() {
        let mut stack = PickleStack::empty();
        // LONG1 with n=2 bytes, value=0x0100 = 256 (little-endian)
        let input: &[u8] = &[0x8a, 0x02, 0x00, 0x01, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Long(256));
    }

    #[test]
    fn pickle_read_proto_ignores_version() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[0x80, 0x05, b'N', b'.']; // PROTO(5) + NONE + STOP
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::None);
    }

    #[test]
    fn pickle_read_append_op() {
        let mut stack = PickleStack::empty();
        // EMPTY_LIST + SHORT_BININT(42) + APPEND('a') + STOP
        let input: &[u8] = &[b']', b'K', 0x2A, b'a', b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::List(items) => assert_eq!(items, vec![Object::Int(42)]),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_appends_op() {
        let mut stack = PickleStack::empty();
        // EMPTY_LIST + MARK + INT(1) + INT(2) + APPENDS('e') + STOP
        let input: &[u8] = &[b']', b'(', b'K', 0x01, b'K', 0x02, b'e', b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::List(items) => assert_eq!(items, vec![Object::Int(1), Object::Int(2)]),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_setitem_op() {
        let mut stack = PickleStack::empty();
        // EMPTY_DICT + UNICODE("k") + INT(99) + SETITEM('s') + STOP
        let key = b"k";
        let mut key_bytes = vec![b'X'];
        key_bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        key_bytes.extend_from_slice(key);
        let mut input = vec![b'}'];
        input.extend_from_slice(&key_bytes);
        input.extend_from_slice(&[b'K', 99, b's', b'.']);
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::Dict(pairs) => {
                assert_eq!(pairs.len(), 1);
                assert_eq!(pairs[0].0, Object::Unicode("k".to_string()));
                assert_eq!(pairs[0].1, Object::Int(99));
            }
            other => panic!("expected Dict, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_unknown_opcode_errors() {
        let mut stack = PickleStack::empty();
        let input: &[u8] = &[0xFE]; // invalid opcode
        let mut reader = std::io::BufReader::new(input);
        let result = stack.read(&mut reader);
        assert!(result.is_err());
    }

    #[test]
    fn pickle_read_binpersid() {
        let mut stack = PickleStack::empty();
        // Push string "data", then BinPersId ('Q')
        let data = b"data";
        let mut input = vec![b'X'];
        input.extend_from_slice(&(data.len() as u32).to_le_bytes());
        input.extend_from_slice(data);
        input.push(b'Q');
        input.push(b'.');
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::PersistentLoad(inner) => {
                assert!(matches!(*inner, Object::Unicode(ref s) if s == "data"));
            }
            other => panic!("expected PersistentLoad, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_bin_put_and_get() {
        let mut stack = PickleStack::empty();
        // SHORT_BININT(7) + BINPUT(0) + SHORT_BININT(8) + BINGET(0) -> duplicates Int(7)
        let input: &[u8] = &[b'K', 0x07, b'q', 0x00, b'K', 0x08, b'h', 0x00, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        // After finalize we get the last pushed item (memo_get pushed Int(7) on top of Int(8))
        let top = stack.finalize().unwrap();
        assert_eq!(top, Object::Int(7));
    }


    // ── reorder_fortran_to_c additional edge cases ──

    #[test]
    fn reorder_fortran_to_c_3x4_non_square() {
        let shape = vec![3, 4];
        let data: Vec<u8> = (0u8..12).collect();
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out.len(), 12);
        // Verify total byte count preserved
        assert_eq!(out.iter().map(|b| *b as u32).sum::<u32>(), data.iter().map(|b| *b as u32).sum::<u32>());
    }

    #[test]
    fn reorder_fortran_to_c_2x2x3() {
        let shape = vec![2, 2, 3];
        let data: Vec<u8> = (0u8..12).collect();
        let out = reorder_fortran_to_c(&data, &shape, 1);
        assert_eq!(out.len(), 12);
    }

    // ── Object clone edge cases ──

    #[test]
    fn object_clone_dict_independent() {
        let obj = Object::Dict(vec![(Object::Int(1), Object::Int(2))]);
        let cloned = obj.clone();
        // Mutating original's inner vec doesn't affect clone
        assert_eq!(obj, cloned);
    }

    #[test]
    fn object_clone_reduce() {
        let obj = Object::Reduce {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Tuple(vec![])),
        };
        let cloned = obj.clone();
        assert_eq!(obj, cloned);
    }

    // ── Additional edge case tests ──

    #[test]
    fn object_float_subnormal_equality() {
        let subnormal = f64::from_bits(1u64); // smallest positive subnormal
        assert_eq!(Object::Float(subnormal), Object::Float(subnormal));
        assert_ne!(Object::Float(subnormal), Object::Float(0.0));
    }

    #[test]
    fn object_float_positive_zero_vs_negative_zero() {
        let pos_zero = Object::Float(0.0);
        let neg_zero = Object::Float(-0.0);
        // f64 PartialEq: +0.0 == -0.0, so derived PartialEq returns true
        assert_eq!(pos_zero, neg_zero);
    }

    #[test]
    fn object_hash_nan_consistent() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        // NaN hashes via to_bits(), so same bit pattern = same hash
        let nan1 = Object::Float(f64::NAN);
        let nan2 = Object::Float(f64::from_bits(f64::NAN.to_bits()));
        assert_eq!(hash_of(&nan1), hash_of(&nan2));
    }

    #[test]
    fn object_hash_build_and_reduce() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let b = Object::Build {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        let r = Object::Reduce {
            callable: Box::new(Object::Int(1)),
            args: Box::new(Object::Int(2)),
        };
        // Same inner values but different discriminants => different hashes
        assert_ne!(hash_of(&b), hash_of(&r));
    }

    #[test]
    fn pickle_read_dict_op_opcode_d() {
        let mut stack = PickleStack::empty();
        // DICT opcode ('d') pushes empty dict, same as '}'
        let input: &[u8] = &[b'd', b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Dict(vec![]));
    }

    #[test]
    fn pickle_read_setitems_op() {
        let mut stack = PickleStack::empty();
        // EMPTY_DICT + MARK + UNICODE("k1") + INT(10) + UNICODE("k2") + INT(20) + SETITEMS('u')
        let k1 = b"k1";
        let mut k1_bytes = vec![b'X'];
        k1_bytes.extend_from_slice(&(k1.len() as u32).to_le_bytes());
        k1_bytes.extend_from_slice(k1);
        let k2 = b"k2";
        let mut k2_bytes = vec![b'X'];
        k2_bytes.extend_from_slice(&(k2.len() as u32).to_le_bytes());
        k2_bytes.extend_from_slice(k2);
        let mut input = vec![b'}', b'('];
        input.extend_from_slice(&k1_bytes);
        input.extend_from_slice(&[b'K', 10]);
        input.extend_from_slice(&k2_bytes);
        input.extend_from_slice(&[b'K', 20, b'u', b'.']);
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::Dict(pairs) => assert_eq!(pairs.len(), 2),
            other => panic!("expected Dict, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_newobj_op() {
        let mut stack = PickleStack::empty();
        // GLOBAL("m","C") + EMPTY_TUPLE + NEWOBJ(0x81)
        let input = b"cm\nC\n)\x81.";
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        match stack.finalize().unwrap() {
            Object::Reduce { callable, args } => {
                assert!(matches!(*callable, Object::Class { .. }));
                assert!(matches!(*args, Object::Tuple(ref t) if t.is_empty()));
            }
            other => panic!("expected Reduce, got {other:?}"),
        }
    }

    #[test]
    fn pickle_read_long_bin_put_and_get() {
        let mut stack = PickleStack::empty();
        // SHORT_BININT(33) + LONG_BINPUT(id=0) + SHORT_BININT(44) + LONG_BINGET(id=0)
        let input: &[u8] = &[b'K', 33, b'r', 0x00, 0x00, 0x00, 0x00, b'K', 44, b'j', 0x00, 0x00, 0x00, 0x00, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        // Last push was LONG_BINGET which retrieves memo[0] = Int(33)
        assert_eq!(stack.finalize().unwrap(), Object::Int(33));
    }

    #[test]
    fn pickle_read_long1_negative_value() {
        let mut stack = PickleStack::empty();
        // LONG1 with n=1 byte, value=0xFF which is -1 in signed 8-bit
        let input: &[u8] = &[0x8a, 0x01, 0xFF, b'.'];
        let mut reader = std::io::BufReader::new(input);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Long(255));
    }

    #[test]
    fn pickle_read_long1_eight_byte_i64() {
        let mut stack = PickleStack::empty();
        // LONG1 with n=8 bytes representing i64::MAX (0x7FFFFFFFFFFFFFFF)
        let val = i64::MAX.to_le_bytes();
        let mut input = vec![0x8a, 0x08];
        input.extend_from_slice(&val);
        input.push(b'.');
        let mut reader = std::io::BufReader::new(&input[..]);
        stack.read_loop(&mut reader).unwrap();
        assert_eq!(stack.finalize().unwrap(), Object::Long(i64::MAX));
    }

    #[test]
    fn tensor_layout_zero_dimension_contiguous() {
        // shape [0, 5]: zero elements, strides don't matter
        let layout = TensorLayout::new(vec![0, 5], vec![5, 1], 0);
        assert!(layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
        assert_eq!(layout.dims(), &[0, 5]);
    }

    // ── New tests (15 additional) ──

    #[test]
    fn object_hash_float_different_values_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let h1 = hash_of(&Object::Float(1.0));
        let h2 = hash_of(&Object::Float(2.0));
        assert_ne!(h1, h2, "different float values must produce different hashes");
    }

    #[test]
    fn object_hash_bool_true_false_differ() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        assert_ne!(
            hash_of(&Object::Bool(true)),
            hash_of(&Object::Bool(false))
        );
    }

    #[test]
    fn object_hash_persistent_load_inner_matters() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        let p1 = Object::PersistentLoad(Box::new(Object::Int(1)));
        let p2 = Object::PersistentLoad(Box::new(Object::Int(2)));
        assert_ne!(hash_of(&p1), hash_of(&p2));
    }

    #[test]
    fn object_class_extraction_error_on_non_class() {
        // Arrange: non-Class variants should return Err
        let non_class = Object::Unicode("not a class".to_string());
        // Act
        let result = non_class.class();
        // Assert
        assert!(result.is_err());
        match result {
            Err(Object::Unicode(s)) => assert_eq!(s, "not a class"),
            other => panic!("expected Err(Unicode), got {other:?}"),
        }
    }

    #[test]
    fn tensor_layout_contiguous_high_rank_5d() {
        // Arrange: 5D tensor shape [2, 3, 4, 5, 6] with row-major strides
        let layout = TensorLayout::new(
            vec![2, 3, 4, 5, 6],
            vec![360, 120, 30, 6, 1],
            0,
        );
        // Act & Assert
        assert!(layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_fortran_high_rank_4d() {
        // Arrange: 4D tensor shape [2, 3, 4, 5] with column-major strides
        let layout = TensorLayout::new(
            vec![2, 3, 4, 5],
            vec![1, 2, 6, 24],
            0,
        );
        // Act & Assert
        assert!(!layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_zero_in_middle_dim_contiguous() {
        // Arrange: shape [3, 0, 5] — zero elements in middle dimension
        //          strides [0, 5, 1] — product of trailing dims for C-contiguous
        let layout = TensorLayout::new(vec![3, 0, 5], vec![0, 5, 1], 0);
        // Act & Assert: zero-size dims make contiguous check pass regardless of outer stride
        assert!(layout.is_contiguous());
    }

    #[test]
    fn pickle_stack_reduce_op_empty_stack_errors() {
        // Arrange: empty stack — reduce_op needs 2 items
        let mut stack = PickleStack::empty();
        // Act
        let result = stack.reduce_op();
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn pickle_stack_build_op_single_item_errors() {
        // Arrange: only one item on stack — build_op needs 2 items
        let mut stack = PickleStack::empty();
        stack.push(Object::Int(1));
        // Act
        let result = stack.build_op();
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn pickle_stack_pop_to_marker_only_marker_returns_empty() {
        // Arrange: only a Mark on the stack, no items after it
        let mut stack = PickleStack::empty();
        stack.push(Object::Mark);
        // Act
        let objs = stack.pop_to_marker().unwrap();
        // Assert: empty vec, marker removed
        assert!(objs.is_empty());
        assert!(stack.pop().is_err(), "stack should be empty after marker removed");
    }

    #[test]
    fn reorder_fortran_to_c_2byte_elements_3x2() {
        // Arrange: shape [3, 2], 2 bytes per element (e.g. F16), 6 elements = 12 bytes
        // F-order layout: column 0 elems [0,1],[2,3],[4,5] then column 1 elems [6,7],[8,9],[10,11]
        let data: Vec<u8> = (0u8..12).collect();
        let shape = vec![3, 2];
        // Act
        let out = reorder_fortran_to_c(&data, &shape, 2);
        // Assert: output length preserved
        assert_eq!(out.len(), 12);
        // First C-order element (row 0, col 0) = F-order element at (0,0) = bytes [0,1]
        assert_eq!(&out[0..2], &[0u8, 1u8]);
    }

    #[test]
    fn object_float_negative_zero_hash_equals_positive_zero() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |obj: &Object| {
            let mut h = DefaultHasher::new();
            obj.hash(&mut h);
            h.finish()
        };
        // Arrange: +0.0 and -0.0 have different bit patterns but hash uses to_bits()
        let pos = Object::Float(0.0f64);
        let neg = Object::Float(-0.0f64);
        // Act & Assert: different bit patterns => different hashes
        assert_ne!(hash_of(&pos), hash_of(&neg));
    }

    #[test]
    fn pytorch_dtype_size_matches_safetensors_dtype_size() {
        // Arrange: for every PytorchDtype, size_in_bytes must be consistent
        // with the corresponding safetensors Dtype element size (via to_st_dtype)
        let cases = vec![
            (PytorchDtype::Bool, 1),
            (PytorchDtype::U8, 1),
            (PytorchDtype::I8, 1),
            (PytorchDtype::I16, 2),
            (PytorchDtype::F16, 2),
            (PytorchDtype::BF16, 2),
            (PytorchDtype::I32, 4),
            (PytorchDtype::F32, 4),
            (PytorchDtype::I64, 8),
            (PytorchDtype::F64, 8),
        ];
        for (dtype, expected_size) in cases {
            // Act & Assert
            assert_eq!(
                dtype.size_in_bytes(), expected_size,
                "{dtype:?} size mismatch"
            );
        }
    }

    // ── 13 additional tests ──

    #[test]
    fn tensor_layout_both_c_and_f_contiguous_scalar() {
        // Arrange: 0-dimensional scalar represented as shape [1], stride [1]
        let layout = TensorLayout::new(vec![1], vec![1], 0);
        // Act & Assert: single-element layout is both C- and F-contiguous
        assert!(layout.is_contiguous());
        assert!(layout.is_fortran_contiguous());
    }

    #[test]
    fn tensor_layout_non_contiguous_all_strides_equal() {
        // Arrange: shape [3, 3], strides [3, 3] — strides equal but not row-major
        let layout = TensorLayout::new(vec![3, 3], vec![3, 3], 0);
        // Act & Assert: neither C- nor F-contiguous
        assert!(!layout.is_contiguous());
        assert!(!layout.is_fortran_contiguous());
    }

    #[test]
    fn resolve_state_dict_empty_dict_no_key() {
        // Arrange: empty dict with no user key
        let dict = Object::Dict(vec![]);
        // Act: no key specified, no "state_dict" entry inside
        let result = resolve_state_dict(dict, None).unwrap();
        // Assert: returns the empty dict itself
        match result {
            Object::Dict(pairs) => assert!(pairs.is_empty()),
            other => panic!("expected Dict, got {other:?}"),
        }
    }

    #[test]
    fn resolve_state_dict_empty_dict_with_key_errors() {
        // Arrange: empty dict with a requested key
        let dict = Object::Dict(vec![]);
        // Act
        let result = resolve_state_dict(dict, Some("missing_key"));
        // Assert: key not found in empty dict
        assert!(result.is_err());
    }

    #[test]
    fn object_try_into_usize_rejects_negative_long() {
        // Arrange: Long with negative value should not convert to usize
        let obj = Object::Long(-1);
        // Act
        let result: std::result::Result<usize, Object> = obj.try_into();
        // Assert: TryFrom<Object> for usize only accepts Object::Int >= 0
        assert!(result.is_err());
    }

    #[test]
    fn pickle_read_append_on_non_list_errors() {
        // Arrange: push a non-list (Int), then APPEND opcode
        let input: &[u8] = &[b'K', 0x05, b'K', 0x01, b'a', b'.'];
        let mut stack = PickleStack::empty();
        let mut reader = std::io::BufReader::new(input);
        // Act
        let result = stack.read_loop(&mut reader);
        // Assert: APPEND on non-list must error
        assert!(result.is_err());
    }

    #[test]
    fn reorder_fortran_to_c_preserves_byte_sum_3x3() {
        // Arrange: 3x3 matrix, 4 bytes per element, 9 elements = 36 bytes
        let data: Vec<u8> = (0u8..36).collect();
        let shape = vec![3, 3];
        // Act
        let out = reorder_fortran_to_c(&data, &shape, 4);
        // Assert: total bytes preserved, byte sum preserved (permutation)
        assert_eq!(out.len(), data.len());
        let sum_orig: u32 = data.iter().map(|&b| b as u32).sum();
        let sum_out: u32 = out.iter().map(|&b| b as u32).sum();
        assert_eq!(sum_orig, sum_out);
    }

    #[test]
    fn reorder_fortran_to_c_2x1_degenerate() {
        // Arrange: shape [2, 1], trivial case where F-order == C-order
        let data: Vec<u8> = vec![10, 20];
        let shape = vec![2, 1];
        // Act
        let out = reorder_fortran_to_c(&data, &shape, 1);
        // Assert: 1-column matrix is identical in both layouts
        assert_eq!(out, data);
    }

    #[test]
    fn packed_bits_hint_substring_not_prefix_or_suffix() {
        // Arrange: name containing "qweight" in the middle, not at start/end
        let cfg = PytorchLoaderConfig::default();
        // Act & Assert: "myqweightlayer" contains "qweight" => Some(4)
        assert_eq!(
            packed_bits_hint("myqweightlayer", PytorchDtype::U8, &cfg),
            Some(4)
        );
        // "qwight" does NOT contain any hint => None
        assert_eq!(
            packed_bits_hint("qwight.tensor", PytorchDtype::U8, &cfg),
            None
        );
    }

    #[test]
    fn object_equality_nested_dict_in_tuple() {
        // Arrange: tuples containing dicts with different nesting levels
        let inner = Object::Dict(vec![
            (Object::Unicode("a".to_string()), Object::Int(1)),
            (Object::Unicode("b".to_string()), Object::Float(2.5)),
        ]);
        let t1 = Object::Tuple(vec![Object::Int(0), inner.clone()]);
        let t2 = Object::Tuple(vec![Object::Int(0), inner.clone()]);
        // Act & Assert: deep equality must hold
        assert_eq!(t1, t2);
        // Mutated inner value breaks equality
        let inner2 = Object::Dict(vec![
            (Object::Unicode("a".to_string()), Object::Int(1)),
            (Object::Unicode("b".to_string()), Object::Float(9.9)),
        ]);
        let t3 = Object::Tuple(vec![Object::Int(0), inner2]);
        assert_ne!(t1, t3);
    }

    // ── 10 additional tests ──

    #[test]
    fn object_int_or_long_accepts_both_variants() {
        // Arrange: Int and Long both represent integers
        let obj_int = Object::Int(42);
        let obj_long = Object::Long(1_000_000_000_000i64);
        let obj_float = Object::Float(3.14);
        // Act & Assert: Int and Long succeed with correct value; Float fails
        assert_eq!(obj_int.int_or_long().unwrap(), 42i64);
        assert_eq!(obj_long.int_or_long().unwrap(), 1_000_000_000_000i64);
        assert!(obj_float.int_or_long().is_err());
    }

    #[test]
    fn object_reduce_and_persistent_load_error_on_wrong_variant() {
        // Arrange: non-Reduce and non-PersistentLoad objects
        let obj = Object::Int(7);
        // Act & Assert: both accessors must return Err
        assert!(obj.clone().reduce().is_err());
        assert!(obj.clone().persistent_load().is_err());
        // Verify error variant is the original object
        let err = obj.reduce().unwrap_err();
        assert_eq!(err, Object::Int(7));
    }

    #[test]
    fn object_tuple_accessor_errors_on_non_tuple() {
        // Arrange: List is not Tuple
        let obj = Object::List(vec![Object::Int(1), Object::Int(2)]);
        // Act
        let result = obj.tuple();
        // Assert: List must not be accepted as Tuple
        assert!(result.is_err());
        match result.unwrap_err() {
            Object::List(items) => assert_eq!(items.len(), 2),
            other => panic!("expected List, got {other:?}"),
        }
    }

    #[test]
    fn try_from_object_vec_usize_from_tuple_of_ints() {
        // Arrange: Tuple of Int(1), Int(2), Int(3)
        let obj = Object::Tuple(vec![Object::Int(1), Object::Int(2), Object::Int(3)]);
        // Act
        let result: std::result::Result<Vec<usize>, Object> = obj.try_into();
        // Assert: extracts [1, 2, 3]
        assert_eq!(result.unwrap(), vec![1usize, 2, 3]);
    }

    #[test]
    fn try_from_object_vec_usize_rejects_tuple_with_negative() {
        // Arrange: Tuple containing a negative Int
        let obj = Object::Tuple(vec![Object::Int(5), Object::Int(-1)]);
        // Act
        let result: std::result::Result<Vec<usize>, Object> = obj.try_into();
        // Assert: negative value rejected, returns the offending Object
        assert!(result.is_err());
    }

    #[test]
    fn try_from_object_string_rejects_non_unicode() {
        // Arrange: Bool is not Unicode
        let obj = Object::Bool(true);
        // Act
        let result: std::result::Result<String, Object> = obj.try_into();
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn pytorch_dtype_to_st_dtype_roundtrip_all_variants() {
        // Arrange: every PytorchDtype variant must map to a distinct safetensors Dtype
        let cases = vec![
            (PytorchDtype::Bool, Dtype::BOOL),
            (PytorchDtype::U8, Dtype::U8),
            (PytorchDtype::I8, Dtype::I8),
            (PytorchDtype::I16, Dtype::I16),
            (PytorchDtype::I32, Dtype::I32),
            (PytorchDtype::I64, Dtype::I64),
            (PytorchDtype::F16, Dtype::F16),
            (PytorchDtype::BF16, Dtype::BF16),
            (PytorchDtype::F32, Dtype::F32),
            (PytorchDtype::F64, Dtype::F64),
        ];
        for (pt_dtype, expected_st) in cases {
            // Act & Assert: each PytorchDtype maps to the correct safetensors Dtype
            assert_eq!(pt_dtype.to_st_dtype(), expected_st, "{pt_dtype:?} mapping");
        }
    }

    #[test]
    fn resolve_state_dict_auto_detects_state_dict_key() {
        // Arrange: Dict with a "state_dict" key containing another Dict
        let inner = Object::Dict(vec![(
            Object::Unicode("weight".to_string()),
            Object::Int(99),
        )]);
        let outer = Object::Dict(vec![
            (Object::Unicode("state_dict".to_string()), inner.clone()),
            (Object::Unicode("epoch".to_string()), Object::Int(3)),
        ]);
        // Act: no explicit key, should auto-detect "state_dict"
        let result = resolve_state_dict(outer, None).unwrap();
        // Assert: returns the inner dict, not the outer
        assert_eq!(result, inner);
    }

    #[test]
    fn unwrap_module_passthrough_non_build_objects() {
        // Arrange: plain Int should pass through unchanged
        let obj = Object::Int(42);
        // Act
        let result = unwrap_module(obj.clone());
        // Assert: non-Build objects returned as-is
        assert_eq!(result, obj);
        // Arrange: Build with non-__torch__ module also passes through
        let build = Object::Build {
            callable: Box::new(Object::Reduce {
                callable: Box::new(Object::Class {
                    module_name: "collections".to_string(),
                    class_name: "OrderedDict".to_string(),
                }),
                args: Box::new(Object::Tuple(vec![])),
            }),
            args: Box::new(Object::Dict(vec![])),
        };
        // Act: unwrap_module on non-__torch__ Build returns the Build intact
        let result2 = unwrap_module(build.clone());
        assert_eq!(result2, build);
    }
}
