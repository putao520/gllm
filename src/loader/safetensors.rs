//! SafeTensors loader (memory mapped).

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use half::{bf16, f16};
use memmap2::MmapOptions;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use super::{parallel::ParallelLoader, LoaderError, Result};

#[derive(Debug, Clone)]
pub struct TensorLocation {
    pub file_idx: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorSlice<'a> {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_f16(&self) -> Result<Cow<'a, [f16]>> {
        if self.dtype != Dtype::F16 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_f16(self.data)
    }

    pub fn as_bf16(&self) -> Result<Cow<'a, [bf16]>> {
        if self.dtype != Dtype::BF16 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_bf16(self.data)
    }

    pub fn as_f32(&self) -> Result<Cow<'a, [f32]>> {
        if self.dtype != Dtype::F32 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_f32(self.data)
    }

    pub fn as_f64(&self) -> Result<Cow<'a, [f64]>> {
        if self.dtype != Dtype::F64 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_f64(self.data)
    }

    pub fn as_i8(&self) -> Result<Cow<'a, [i8]>> {
        if self.dtype != Dtype::I8 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_i8(self.data)
    }

    pub fn as_u8(&self) -> Result<Cow<'a, [u8]>> {
        if self.dtype != Dtype::U8 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        Ok(Cow::Borrowed(self.data))
    }

    pub fn as_i16(&self) -> Result<Cow<'a, [i16]>> {
        if self.dtype != Dtype::I16 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_i16(self.data)
    }

    pub fn as_u16(&self) -> Result<Cow<'a, [u16]>> {
        if self.dtype != Dtype::U16 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_u16(self.data)
    }

    pub fn as_i32(&self) -> Result<Cow<'a, [i32]>> {
        if self.dtype != Dtype::I32 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_i32(self.data)
    }

    pub fn as_u32(&self) -> Result<Cow<'a, [u32]>> {
        if self.dtype != Dtype::U32 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_u32(self.data)
    }

    pub fn as_i64(&self) -> Result<Cow<'a, [i64]>> {
        if self.dtype != Dtype::I64 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_i64(self.data)
    }

    pub fn as_u64(&self) -> Result<Cow<'a, [u64]>> {
        if self.dtype != Dtype::U64 {
            return Err(LoaderError::UnsupportedDtype(self.dtype));
        }
        cast_or_copy_u64(self.data)
    }
}

#[derive(Debug)]
pub struct MappedSafetensors {
    path: PathBuf,
    mmap: Arc<memmap2::Mmap>,
    tensors: SafeTensors<'static>,
    metadata: Option<HashMap<String, String>>,
}

impl MappedSafetensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let arc = Arc::new(mmap);
        let slice = &arc[..];
        let (_, metadata) = SafeTensors::read_metadata(slice)?;
        let metadata = metadata.metadata().clone();
        let tensors = SafeTensors::deserialize(slice)?;
        // SAFETY: mmap is stored in the struct, so the backing bytes outlive SafeTensors.
        let tensors =
            unsafe { std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(tensors) };
        Ok(Self {
            path: path.to_path_buf(),
            mmap: arc,
            tensors,
            metadata,
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.tensors.names().into_iter().map(|s| s.to_string()).collect()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let view = self
            .tensors
            .tensor(name)
            .map_err(|err| LoaderError::SafeTensors(err))?;
        let dtype = view.dtype();
        let shape = view.shape().to_vec();
        let data = view.data();
        Ok(TensorSlice { dtype, shape, data })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mmap(&self) -> &Arc<memmap2::Mmap> {
        &self.mmap
    }

    pub fn metadata(&self) -> Option<&std::collections::HashMap<String, String>> {
        self.metadata.as_ref()
    }
}

#[derive(Debug)]
pub struct SafeTensorsLoader {
    files: Vec<MappedSafetensors>,
    index: HashMap<String, TensorLocation>,
}

impl SafeTensorsLoader {
    pub fn from_files(paths: &[PathBuf], parallel: ParallelLoader) -> Result<Self> {
        if paths.is_empty() {
            return Err(LoaderError::MissingWeights);
        }

        let files = parallel.map_paths(paths, |path| MappedSafetensors::open(path))?;
        let mut index = HashMap::new();

        for (file_idx, file) in files.iter().enumerate() {
            for name in file.names() {
                if index.contains_key(&name) {
                    return Err(LoaderError::DuplicateTensor(name));
                }
                let view = file
                    .tensors
                    .tensor(&name)
                    .map_err(|err| LoaderError::SafeTensors(err))?;
                index.insert(
                    name,
                    TensorLocation {
                        file_idx,
                        dtype: view.dtype(),
                        shape: view.shape().to_vec(),
                    },
                );
            }
        }

        Ok(Self { files, index })
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let location = self
            .index
            .get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let file = &self.files[location.file_idx];
        let mut tensor = file.tensor(name)?;
        tensor.dtype = location.dtype;
        tensor.shape = location.shape.clone();
        Ok(tensor)
    }

    pub fn tensor_meta(&self, name: &str) -> Option<&TensorLocation> {
        self.index.get(name)
    }

    pub fn prefetch_parallel(&self, parallel: ParallelLoader) -> Result<()> {
        if !parallel.enabled() {
            return Ok(());
        }
        let results: Vec<Result<()>> = self
            .index
            .par_iter()
            .map(|(name, _)| self.tensor(name).map(|_| ()))
            .collect();
        for result in results {
            result?;
        }
        Ok(())
    }

    pub fn packed_bits(&self) -> HashMap<String, u8> {
        let mut out = HashMap::new();
        for file in &self.files {
            let Some(meta) = file.metadata() else {
                continue;
            };
            let Some(encoded) = meta.get("gllm.packed_bits") else {
                continue;
            };
            if let Ok(map) = serde_json::from_str::<HashMap<String, u8>>(encoded) {
                out.extend(map);
                continue;
            }
            if let Ok(map) = serde_json::from_str::<HashMap<String, Value>>(encoded) {
                for (key, value) in map {
                    if let Some(bits) = value.as_u64().and_then(|v| u8::try_from(v).ok()) {
                        out.insert(key, bits);
                    }
                }
            }
        }
        out
    }
}

fn cast_or_copy_f16(data: &[u8]) -> Result<Cow<'_, [f16]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<f16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_bf16(data: &[u8]) -> Result<Cow<'_, [bf16]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<bf16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16::from_bits(bits));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_f32(data: &[u8]) -> Result<Cow<'_, [f32]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<f32>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_f64(data: &[u8]) -> Result<Cow<'_, [f64]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<f64>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let value = f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i8(data: &[u8]) -> Result<Cow<'_, [i8]>> {
    let mut out = Vec::with_capacity(data.len());
    out.extend(data.iter().map(|value| *value as i8));
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i16(data: &[u8]) -> Result<Cow<'_, [i16]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<i16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_u16(data: &[u8]) -> Result<Cow<'_, [u16]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<u16>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i32(data: &[u8]) -> Result<Cow<'_, [i32]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<i32>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_u32(data: &[u8]) -> Result<Cow<'_, [u32]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<u32>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i64(data: &[u8]) -> Result<Cow<'_, [i64]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<i64>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        out.push(i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_u64(data: &[u8]) -> Result<Cow<'_, [u64]>> {
    let (prefix, body, suffix) = unsafe { data.align_to::<u64>() };
    if prefix.is_empty() && suffix.is_empty() {
        return Ok(Cow::Borrowed(body));
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        out.push(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(Cow::Owned(out))
}
