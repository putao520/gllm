//! GGUF v3 file parser (pure Rust).
//!
//! This parser focuses on the parts required to load quantized GGUF tensors:
//! - Header parsing
//! - Metadata key/value decoding
//! - Tensor info table
//! - Tensor byte slicing with alignment handling

use crate::quantized::GgmlDType;
use crate::types::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Default alignment when GGUF metadata does not specify one.
const DEFAULT_ALIGNMENT: u64 = 32;
/// GGUF magic number in little-endian.
const MAGIC_GGUF_LE: u32 = 0x4655_4747;
/// GGUF magic number in big-endian (accepted for robustness).
const MAGIC_GGUF_BE: u32 = 0x4747_5546;

/// GGUF file header.
pub struct GgufHeader {
    /// Magic number (0x46554747).
    pub magic: u32,
    /// Format version (expected 3).
    pub version: u32,
    /// Number of tensors stored in the file.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs.
    pub metadata_kv_count: u64,
}

/// Parsed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// Tensor metadata extracted from GGUF.
pub struct TensorInfo {
    /// Tensor name as stored in the file.
    pub name: String,
    /// Number of dimensions in the tensor.
    pub n_dims: u32,
    /// Dimension sizes (already reversed into row-major order).
    pub dims: Vec<u64>,
    /// GGML dtype encoding for this tensor.
    pub dtype: GgmlDType,
    /// Offset into the tensor data section (relative to data start).
    pub offset: u64,
}

/// GGUF loader with parsed metadata and tensor table.
pub struct GgufLoader {
    #[allow(dead_code)]
    header: GgufHeader,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, TensorInfo>,
    data: Mmap,
    tensor_data_offset: u64,
}

impl GgufLoader {
    /// Load and parse a GGUF v3 file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open GGUF file {}: {err}",
                path.display()
            ))
        })?;
        // Safety: the file is not mutated while the mmap is alive.
        let data = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to memory-map GGUF file {}: {err}",
                path.display()
            ))
        })?;

        let mut reader = GgufReader::new(&data);
        let magic = reader.read_u32()?;
        if magic != MAGIC_GGUF_LE && magic != MAGIC_GGUF_BE {
            return Err(Error::LoadError(format!(
                "Invalid GGUF magic 0x{magic:08x}"
            )));
        }
        let version = reader.read_u32()?;
        if version != 3 {
            return Err(Error::LoadError(format!(
                "Unsupported GGUF version {version} (expected 3)"
            )));
        }
        let tensor_count = reader.read_u64()?;
        let metadata_kv_count = reader.read_u64()?;

        let header = GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        };

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let value_type = reader.read_u32()?;
            let value_type = GgufValueType::from_u32(value_type)?;
            let value = reader.read_value(value_type)?;
            metadata.insert(key, value);
        }

        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.read_u64()?);
            }
            // GGUF stores dims in reverse order.
            dims.reverse();
            let dtype_raw = reader.read_u32()?;
            let dtype = GgmlDType::from_u32(dtype_raw)?;
            let offset = reader.read_u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    n_dims,
                    dims,
                    dtype,
                    offset,
                },
            );
        }

        let alignment = alignment_from_metadata(&metadata);
        let position = reader.position() as u64;
        let tensor_data_offset = align_up(position, alignment);

        Ok(Self {
            header,
            metadata,
            tensors,
            data,
            tensor_data_offset,
        })
    }

    /// Look up a tensor by name.
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get raw tensor data bytes for the specified tensor.
    pub fn get_tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let byte_len = match tensor_byte_len(info) {
            Some(len) => len,
            None => return &[],
        };
        let start = match self.tensor_data_offset.checked_add(info.offset) {
            Some(value) => value as usize,
            None => return &[],
        };
        let end = match start.checked_add(byte_len) {
            Some(value) => value,
            None => return &[],
        };
        if end > self.data.len() {
            return &[];
        }
        &self.data[start..end]
    }

    /// Fetch a metadata value as u64 if possible.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        let value = self.metadata.get(key)?;
        match value {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Fetch a metadata value as string if it is a GGUF string.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }
}

/// GGUF value type tags.
#[derive(Debug, Clone, Copy)]
enum GgufValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Bool,
    String,
    Array,
}

impl GgufValueType {
    fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(Error::LoadError(format!(
                "Unknown GGUF value type {value}"
            ))),
        }
    }
}

/// Cursor-style reader for a memory-mapped GGUF file.
struct GgufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> GgufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            Error::LoadError("GGUF reader overflow while slicing bytes".into())
        })?;
        if end > self.data.len() {
            return Err(Error::LoadError(
                "Unexpected EOF while reading GGUF data".into(),
            ));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let mut bytes = self.read_bytes(len)?.to_vec();
        // GGUF sometimes stores null-terminated strings; trim them defensively.
        while matches!(bytes.last(), Some(0)) {
            bytes.pop();
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn read_value(&mut self, value_type: GgufValueType) -> Result<GgufValue> {
        match value_type {
            GgufValueType::U8 => Ok(GgufValue::U8(self.read_u8()?)),
            GgufValueType::I8 => Ok(GgufValue::I8(self.read_i8()?)),
            GgufValueType::U16 => Ok(GgufValue::U16(self.read_u16()?)),
            GgufValueType::I16 => Ok(GgufValue::I16(self.read_i16()?)),
            GgufValueType::U32 => Ok(GgufValue::U32(self.read_u32()?)),
            GgufValueType::I32 => Ok(GgufValue::I32(self.read_i32()?)),
            GgufValueType::U64 => Ok(GgufValue::U64(self.read_u64()?)),
            GgufValueType::I64 => Ok(GgufValue::I64(self.read_i64()?)),
            GgufValueType::F32 => Ok(GgufValue::F32(self.read_f32()?)),
            GgufValueType::F64 => Ok(GgufValue::F64(self.read_f64()?)),
            GgufValueType::Bool => {
                let value = match self.read_u8()? {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(Error::LoadError(format!(
                            "Invalid GGUF bool value {other}"
                        )))
                    }
                };
                Ok(GgufValue::Bool(value))
            }
            GgufValueType::String => Ok(GgufValue::String(self.read_string()?)),
            GgufValueType::Array => {
                let value_type_raw = self.read_u32()?;
                let value_type = GgufValueType::from_u32(value_type_raw)?;
                let len = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value(value_type)?);
                }
                Ok(GgufValue::Array(items))
            }
        }
    }
}

/// Compute aligned start offset for the tensor data section.
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) / alignment * alignment
}

/// Infer tensor-data alignment from GGUF metadata.
fn alignment_from_metadata(metadata: &HashMap<String, GgufValue>) -> u64 {
    let value = match metadata.get("general.alignment") {
        Some(value) => value,
        None => return DEFAULT_ALIGNMENT,
    };
    let alignment = match value {
        GgufValue::U8(v) => *v as u64,
        GgufValue::U16(v) => *v as u64,
        GgufValue::U32(v) => *v as u64,
        GgufValue::I8(v) if *v >= 0 => *v as u64,
        GgufValue::I16(v) if *v >= 0 => *v as u64,
        GgufValue::I32(v) if *v >= 0 => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 {
        DEFAULT_ALIGNMENT
    } else {
        alignment
    }
}

/// Compute the byte length for a tensor payload.
fn tensor_byte_len(info: &TensorInfo) -> Option<usize> {
    let elem_count = info
        .dims
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))?;
    let block_size = info.dtype.block_size() as u64;
    if block_size == 0 || elem_count % block_size != 0 {
        return None;
    }
    let blocks = elem_count / block_size;
    let bytes = blocks.checked_mul(info.dtype.type_size() as u64)?;
    usize::try_from(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_gguf_header_parse() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC_GGUF_LE.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&bytes).expect("write gguf");

        let loader = GgufLoader::load(file.path()).expect("load gguf");
        assert_eq!(loader.header.magic, MAGIC_GGUF_LE);
        assert_eq!(loader.header.version, 3);
        assert_eq!(loader.header.tensor_count, 0);
        assert_eq!(loader.header.metadata_kv_count, 0);
    }
}
