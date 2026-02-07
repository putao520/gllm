use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};

use super::{
    tensor_nbytes, GgmlDType, GgufArray, GgufError, GgufValue, GgufValueType, TensorSlice,
    GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: Arc<str>,
    pub dtype: GgmlDType,
    pub shape: Vec<u64>,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug)]
pub struct GgufReader {
    mmap: Arc<Mmap>,
    version: u32,
    tensor_count: usize,
    kv_count: usize,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    data_offset: usize,
    quantization_types: Vec<String>,
}

impl GgufReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        // SAFETY: mapping read-only model file for zero-copy tensor access.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::parse(Arc::new(mmap))
    }

    pub fn from_file(path: &Path) -> Result<Self, GgufError> {
        Self::open(path)
    }

    pub fn from_files(paths: &[PathBuf]) -> Result<Self, GgufError> {
        if paths.len() != 1 {
            return Err(GgufError::ParseError(
                "gguf loader expects a single weight file".to_string(),
            ));
        }
        Self::open(&paths[0])
    }

    fn parse(mmap: Arc<Mmap>) -> Result<Self, GgufError> {
        let bytes = &mmap[..];
        let mut pos = 0usize;

        let magic = read_u32(bytes, &mut pos)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32(bytes, &mut pos)?;
        if version != GGUF_SUPPORTED_VERSION {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count_u64 = read_u64(bytes, &mut pos)?;
        let kv_count_u64 = read_u64(bytes, &mut pos)?;
        let tensor_count = usize::try_from(tensor_count_u64)
            .map_err(|_| GgufError::ParseError("tensor_count overflow".to_string()))?;
        let kv_count = usize::try_from(kv_count_u64)
            .map_err(|_| GgufError::ParseError("kv_count overflow".to_string()))?;

        let mut metadata = BTreeMap::new();
        for _ in 0..kv_count {
            let key = read_string(bytes, &mut pos)?;
            let value_type = GgufValueType::try_from(read_u32(bytes, &mut pos)?)?;
            let value = parse_value(bytes, &mut pos, value_type)?;
            metadata.insert(key, value);
        }

        let mut raw_tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = Arc::<str>::from(read_string(bytes, &mut pos)?);
            let n_dims = usize::try_from(read_u32(bytes, &mut pos)?)
                .map_err(|_| GgufError::ParseError("tensor n_dims overflow".to_string()))?;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(bytes, &mut pos)?);
            }
            let dtype = GgmlDType::try_from(read_u32(bytes, &mut pos)?)?;
            let rel_offset = read_u64(bytes, &mut pos)?;
            raw_tensors.push((name, dtype, shape, rel_offset));
        }

        let data_offset = if let Some(alignment) = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
        {
            if alignment == 0 {
                return Err(GgufError::InvalidMetadata(
                    "general.alignment must be > 0".to_string(),
                ));
            }
            align_up(
                pos,
                usize::try_from(alignment)
                    .map_err(|_| GgufError::ParseError("alignment overflow".to_string()))?,
            )?
        } else {
            pos
        };
        if data_offset > bytes.len() {
            return Err(GgufError::ParseError(
                "tensor data offset exceeds file size".to_string(),
            ));
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);

        for (name, dtype, shape, rel_offset_u64) in raw_tensors {
            let rel_offset = usize::try_from(rel_offset_u64)
                .map_err(|_| GgufError::ParseError("tensor offset overflow".to_string()))?;
            let size = tensor_nbytes(dtype, &shape)?;

            let offset = data_offset.checked_add(rel_offset).ok_or_else(|| {
                GgufError::ParseError("tensor absolute offset overflow".to_string())
            })?;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| GgufError::ParseError("tensor end offset overflow".to_string()))?;

            if end > bytes.len() {
                return Err(GgufError::TensorOutOfBounds(format!(
                    "{} [{}..{}) exceeds file size {}",
                    name,
                    offset,
                    end,
                    bytes.len()
                )));
            }

            if tensor_index
                .insert(name.to_string(), tensors.len())
                .is_some()
            {
                return Err(GgufError::ParseError(format!(
                    "duplicate tensor name: {}",
                    name
                )));
            }

            tensors.push(TensorInfo {
                name,
                dtype,
                shape,
                offset,
                size,
            });
        }

        let mut quantization_types = BTreeSet::new();
        for tensor in &tensors {
            if tensor.dtype.is_quantized() {
                quantization_types.insert(tensor.dtype.as_str().to_string());
            }
        }

        Ok(Self {
            mmap,
            version,
            tensor_count,
            kv_count,
            metadata,
            tensors,
            tensor_index,
            data_offset,
            quantization_types: quantization_types.into_iter().collect(),
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    pub fn kv_count(&self) -> usize {
        self.kv_count
    }

    pub fn data_offset(&self) -> usize {
        self.data_offset
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.iter().map(|t| t.name.to_string()).collect();
        names.sort();
        names
    }

    pub fn architecture(&self) -> Result<&str, GgufError> {
        self.get_metadata_str("general.architecture")
            .ok_or_else(|| GgufError::MissingMetadata("general.architecture".to_string()))
    }

    pub fn architecture_name(&self) -> Option<&str> {
        self.get_metadata_str("general.architecture")
    }

    pub fn quantization_version(&self) -> Option<u64> {
        self.get_metadata_u64("general.quantization_version")
    }

    pub fn quantization_types(&self) -> &[String] {
        &self.quantization_types
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    fn get_metadata_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(GgufValue::as_bool)
    }

    pub fn get_metadata_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufValue::as_u64)
    }

    pub fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(GgufValue::as_f32)
    }

    pub fn get_metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    pub fn get_metadata_array(&self, key: &str) -> Option<&GgufArray> {
        self.metadata.get(key).and_then(GgufValue::as_array)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.get_metadata_u64("tokenizer.ggml.bos_token_id")
            .and_then(|v| u32::try_from(v).ok())
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.get_metadata_u64("tokenizer.ggml.eos_token_id")
            .and_then(|v| u32::try_from(v).ok())
    }

    pub fn add_bos_token(&self) -> bool {
        self.get_metadata_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false)
    }

    pub fn add_eos_token(&self) -> bool {
        self.get_metadata_bool("tokenizer.ggml.add_eos_token")
            .unwrap_or(false)
    }

    fn get_arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.architecture_name()?;
        let key = format!("{arch}.{suffix}");
        self.get_metadata_u64(&key)
    }

    fn get_arch_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.architecture_name()?;
        let key = format!("{arch}.{suffix}");
        self.get_metadata_f32(&key)
    }

    pub fn embedding_length(&self) -> Option<u64> {
        self.get_arch_u64("embedding_length")
    }

    pub fn block_count(&self) -> Option<u64> {
        self.get_arch_u64("block_count")
    }

    pub fn head_count(&self) -> Option<u64> {
        self.get_arch_u64("attention.head_count")
    }

    pub fn head_count_kv(&self) -> Option<u64> {
        self.get_arch_u64("attention.head_count_kv")
    }

    pub fn context_length(&self) -> Option<u64> {
        self.get_arch_u64("context_length")
    }

    pub fn rope_dimension_count(&self) -> Option<u64> {
        self.get_arch_u64("rope.dimension_count")
    }

    pub fn rope_freq_base(&self) -> Option<f32> {
        self.get_arch_f32("rope.freq_base")
    }

    pub fn tokenizer_tokens(&self) -> Result<Vec<&str>, GgufError> {
        let tokens = self
            .get_metadata_array("tokenizer.ggml.tokens")
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".to_string()))?;

        if tokens.item_type != GgufValueType::String {
            return Err(GgufError::InvalidMetadata(
                "tokenizer.ggml.tokens must be ARRAY[STRING]".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(tokens.items.len());
        for item in &tokens.items {
            out.push(item.as_str().ok_or_else(|| {
                GgufError::InvalidMetadata(
                    "tokenizer.ggml.tokens contains non-string item".to_string(),
                )
            })?);
        }
        Ok(out)
    }

    pub fn tensor_info(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        let idx = self
            .tensor_index
            .get(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        Ok(&self.tensors[*idx])
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>, GgufError> {
        let info = self.tensor_info(name)?;
        let data = self.tensor_bytes(name)?;
        Ok(TensorSlice::new(info.dtype, info.shape.clone(), data))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let info = self.tensor_info(name)?;
        let end = info
            .offset
            .checked_add(info.size)
            .ok_or_else(|| GgufError::ParseError("tensor end offset overflow".to_string()))?;

        if end > self.mmap.len() {
            return Err(GgufError::TensorOutOfBounds(name.to_string()));
        }

        Ok(&self.mmap[info.offset..end])
    }

    pub fn floating_point_dtype_size(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for tensor in &self.tensors {
            let size = match tensor.dtype {
                GgmlDType::F16 | GgmlDType::BF16 => 2,
                GgmlDType::F32 => 4,
                GgmlDType::F64 => 8,
                _ => continue,
            };
            best = Some(best.map_or(size, |current| current.min(size)));
            if best == Some(2) {
                break;
            }
        }
        best
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, GgufError> {
    if alignment == 0 {
        return Err(GgufError::ParseError("alignment must be > 0".to_string()));
    }
    let add = alignment - 1;
    let rounded = value
        .checked_add(add)
        .ok_or_else(|| GgufError::ParseError("alignment overflow".to_string()))?;
    Ok((rounded / alignment) * alignment)
}

fn parse_value(
    data: &[u8],
    pos: &mut usize,
    value_type: GgufValueType,
) -> Result<GgufValue, GgufError> {
    match value_type {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(read_u8(data, pos)?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(read_u8(data, pos)? as i8)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(read_u16(data, pos)?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(read_u16(data, pos)? as i16)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(read_u32(data, pos)?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(read_u32(data, pos)? as i32)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(f32::from_bits(read_u32(data, pos)?))),
        GgufValueType::Bool => Ok(GgufValue::Bool(read_u8(data, pos)? != 0)),
        GgufValueType::String => Ok(GgufValue::String(Arc::<str>::from(read_string(data, pos)?))),
        GgufValueType::Array => parse_array(data, pos),
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(read_u64(data, pos)?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(read_u64(data, pos)? as i64)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(f64::from_bits(read_u64(data, pos)?))),
    }
}

fn parse_array(data: &[u8], pos: &mut usize) -> Result<GgufValue, GgufError> {
    let item_type = GgufValueType::try_from(read_u32(data, pos)?)?;
    let count = usize::try_from(read_u64(data, pos)?)
        .map_err(|_| GgufError::ParseError("array length overflow".to_string()))?;

    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(parse_value(data, pos, item_type)?);
    }

    Ok(GgufValue::Array(GgufArray { item_type, items }))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = usize::try_from(read_u64(data, pos)?)
        .map_err(|_| GgufError::ParseError("string length overflow".to_string()))?;
    let bytes = read_bytes(data, pos, len)?;
    Ok(std::str::from_utf8(bytes)?.to_string())
}

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, GgufError> {
    Ok(read_bytes(data, pos, 1)?[0])
}

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16, GgufError> {
    let bytes = read_bytes(data, pos, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    let bytes = read_bytes(data, pos, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    let bytes = read_bytes(data, pos, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8], GgufError> {
    let end = pos
        .checked_add(len)
        .ok_or_else(|| GgufError::ParseError("offset overflow".to_string()))?;
    if end > data.len() {
        return Err(GgufError::ParseError("unexpected end of file".to_string()));
    }
    let out = &data[*pos..end];
    *pos = end;
    Ok(out)
}
