//! GGUF loader (memory mapped, header parsed via gguf-rs).

use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use gguf_rs::{ByteOrder, GGMLType, GGUFContainer, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE};
use half::{bf16, f16};
use memmap2::MmapOptions;

use super::{LoaderError, Result};
use gllm_kernels::QuantizedType;

const GGUF_MAX_ARRAY: u64 = 3;

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub kind: u32,
    pub shape: Vec<usize>,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug)]
pub struct GgufTensorSlice<'a> {
    pub kind: GGMLType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
    pub byte_order: ByteOrder,
}

impl<'a> GgufTensorSlice<'a> {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn quantized_type(&self) -> Option<QuantizedType> {
        match &self.kind {
            GGMLType::Q4_0 => Some(QuantizedType::Q4_0),
            GGMLType::Q8_0 => Some(QuantizedType::Q8_0),
            GGMLType::Q5_K => Some(QuantizedType::Q5_K),
            _ => None,
        }
    }

    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match &self.kind {
            GGMLType::F16 => {
                let data = cast_or_copy_f16(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>())
            }
            GGMLType::BF16 => {
                let data = cast_or_copy_bf16(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| v.to_f32()).collect::<Vec<f32>>())
            }
            GGMLType::F32 => {
                let data = cast_or_copy_f32(self.data, &self.byte_order)?;
                Ok(data.iter().copied().collect())
            }
            GGMLType::F64 => {
                let data = cast_or_copy_f64(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| *v as f32).collect())
            }
            GGMLType::I8 => {
                let data = cast_or_copy_i8(self.data)?;
                Ok(data.iter().map(|v| *v as f32).collect())
            }
            GGMLType::I16 => {
                let data = cast_or_copy_i16(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| *v as f32).collect())
            }
            GGMLType::I32 => {
                let data = cast_or_copy_i32(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| *v as f32).collect())
            }
            GGMLType::I64 => {
                let data = cast_or_copy_i64(self.data, &self.byte_order)?;
                Ok(data.iter().map(|v| *v as f32).collect())
            }
            other => Err(LoaderError::Gguf(format!(
                "unsupported gguf tensor type {other:?}"
            ))),
        }
    }
}

#[derive(Debug)]
struct MappedGguf {
    #[allow(dead_code)]
    path: PathBuf,
    mmap: Arc<memmap2::Mmap>,
    byte_order: ByteOrder,
    data_start: usize,
    alignment: usize,
}

impl MappedGguf {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let arc = Arc::new(mmap);
        let bytes = &arc[..];

        if bytes.len() < 8 {
            return Err(LoaderError::Gguf("gguf file too small".into()));
        }

        let magic = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let byte_order = match magic {
            FILE_MAGIC_GGUF_LE => ByteOrder::LE,
            FILE_MAGIC_GGUF_BE => ByteOrder::BE,
            _ => return Err(LoaderError::Gguf("invalid gguf magic number".into())),
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader::new(MmapReader::new(arc.clone(), 4), counter.clone());
        let mut container =
            GGUFContainer::new(byte_order.clone(), Box::new(reader), GGUF_MAX_ARRAY);
        let model = container
            .decode()
            .map_err(|err| LoaderError::Gguf(err.to_string()))?;

        let header_len = 4 + counter.load(Ordering::Relaxed);
        let alignment = model
            .metadata()
            .get("general.alignment")
            .and_then(|value| value.as_u64())
            .unwrap_or(32) as usize;
        let alignment = alignment.max(1);
        let data_start = align_up(header_len, alignment);

        if data_start > bytes.len() {
            return Err(LoaderError::Gguf("gguf header exceeds file size".into()));
        }

        Ok(Self {
            path: path.to_path_buf(),
            mmap: arc,
            byte_order,
            data_start,
            alignment,
        })
    }

    fn byte_order(&self) -> &ByteOrder {
        &self.byte_order
    }

    fn data_start(&self) -> usize {
        self.data_start
    }

    #[allow(dead_code)]
    fn alignment(&self) -> usize {
        self.alignment
    }
}

#[derive(Debug)]
pub struct GgufLoader {
    file: MappedGguf,
    index: HashMap<String, GgufTensorInfo>,
}

impl GgufLoader {
    pub fn from_files(paths: &[PathBuf]) -> Result<Self> {
        if paths.len() != 1 {
            return Err(LoaderError::Gguf(
                "gguf loader expects a single weight file".into(),
            ));
        }
        Self::from_file(&paths[0])
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let file = MappedGguf::open(path)?;
        let mut index = HashMap::new();

        let counter = Arc::new(AtomicUsize::new(0));
        let reader = CountingReader::new(MmapReader::new(file.mmap.clone(), 4), counter.clone());
        let mut container =
            GGUFContainer::new(file.byte_order.clone(), Box::new(reader), GGUF_MAX_ARRAY);
        let model = container
            .decode()
            .map_err(|err| LoaderError::Gguf(err.to_string()))?;

        for tensor in model.tensors() {
            let name = tensor.name.clone();
            if index.contains_key(&name) {
                return Err(LoaderError::DuplicateTensor(name));
            }
            let _ = GGMLType::try_from(tensor.kind)
                .map_err(|err| LoaderError::Gguf(err.to_string()))?;
            let shape = normalize_shape(&tensor.shape)?;
            let offset = usize::try_from(tensor.offset)
                .map_err(|_| LoaderError::Gguf("tensor offset overflow".into()))?;
            let size = usize::try_from(tensor.size)
                .map_err(|_| LoaderError::Gguf("tensor size overflow".into()))?;
            let absolute = file
                .data_start()
                .checked_add(offset)
                .ok_or_else(|| LoaderError::Gguf("tensor offset overflow".into()))?;
            if absolute.checked_add(size).map(|end| end <= file.mmap.len()) != Some(true) {
                return Err(LoaderError::Gguf(format!(
                    "tensor {name} exceeds file bounds"
                )));
            }

            index.insert(
                name,
                GgufTensorInfo {
                    kind: tensor.kind,
                    shape,
                    offset: absolute as u64,
                    size: size as u64,
                },
            );
        }

        Ok(Self { file, index })
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Result<GgufTensorSlice<'_>> {
        let info = self
            .index
            .get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let start = usize::try_from(info.offset)
            .map_err(|_| LoaderError::Gguf("tensor offset overflow".into()))?;
        let size = usize::try_from(info.size)
            .map_err(|_| LoaderError::Gguf("tensor size overflow".into()))?;
        let end = start + size;
        let data = &self.file.mmap[start..end];
        let kind =
            GGMLType::try_from(info.kind).map_err(|err| LoaderError::Gguf(err.to_string()))?;
        Ok(GgufTensorSlice {
            kind,
            shape: info.shape.clone(),
            data,
            byte_order: self.file.byte_order().clone(),
        })
    }

    pub fn tensor_meta(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.index.get(name)
    }
}

fn normalize_shape(shape: &[u64]) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(shape.len());
    for &dim in shape {
        let value =
            usize::try_from(dim).map_err(|_| LoaderError::Gguf("tensor shape overflow".into()))?;
        out.push(value);
    }
    while out.len() > 1 && out.last() == Some(&1) {
        out.pop();
    }
    Ok(out)
}

fn align_up(value: usize, alignment: usize) -> usize {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) / alignment * alignment
}

struct MmapReader {
    mmap: Arc<memmap2::Mmap>,
    pos: usize,
}

impl MmapReader {
    fn new(mmap: Arc<memmap2::Mmap>, start: usize) -> Self {
        Self { mmap, pos: start }
    }
}

impl Read for MmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.mmap.len() {
            return Ok(0);
        }
        let remaining = &self.mmap[self.pos..];
        let count = remaining.len().min(buf.len());
        buf[..count].copy_from_slice(&remaining[..count]);
        self.pos += count;
        Ok(count)
    }
}

struct CountingReader<R> {
    inner: R,
    count: Arc<AtomicUsize>,
}

impl<R> CountingReader<R> {
    fn new(inner: R, count: Arc<AtomicUsize>) -> Self {
        Self { inner, count }
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.inner.read(buf)?;
        self.count.fetch_add(bytes, Ordering::Relaxed);
        Ok(bytes)
    }
}

fn cast_or_copy_f16<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [f16]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<f16>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = match bo {
            ByteOrder::LE => u16::from_le_bytes([chunk[0], chunk[1]]),
            ByteOrder::BE => u16::from_be_bytes([chunk[0], chunk[1]]),
        };
        out.push(f16::from_bits(bits));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_bf16<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [bf16]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<bf16>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = match bo {
            ByteOrder::LE => u16::from_le_bytes([chunk[0], chunk[1]]),
            ByteOrder::BE => u16::from_be_bytes([chunk[0], chunk[1]]),
        };
        out.push(bf16::from_bits(bits));
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_f32<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [f32]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<f32>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let value = match bo {
            ByteOrder::LE => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            ByteOrder::BE => f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        };
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_f64<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [f64]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<f64>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let value = match bo {
            ByteOrder::LE => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            ByteOrder::BE => f64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
        };
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i8(data: &[u8]) -> Result<Cow<'_, [i8]>> {
    let mut out = Vec::with_capacity(data.len());
    out.extend(data.iter().map(|value| *value as i8));
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i16<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [i16]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<i16>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let value = match bo {
            ByteOrder::LE => i16::from_le_bytes([chunk[0], chunk[1]]),
            ByteOrder::BE => i16::from_be_bytes([chunk[0], chunk[1]]),
        };
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i32<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [i32]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<i32>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let value = match bo {
            ByteOrder::LE => i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            ByteOrder::BE => i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        };
        out.push(value);
    }
    Ok(Cow::Owned(out))
}

fn cast_or_copy_i64<'a>(data: &'a [u8], bo: &ByteOrder) -> Result<Cow<'a, [i64]>> {
    if matches!(bo, ByteOrder::LE) {
        let (prefix, body, suffix) = unsafe { data.align_to::<i64>() };
        if prefix.is_empty() && suffix.is_empty() {
            return Ok(Cow::Borrowed(body));
        }
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let value = match bo {
            ByteOrder::LE => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
            ByteOrder::BE => i64::from_be_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]),
        };
        out.push(value);
    }
    Ok(Cow::Owned(out))
}
