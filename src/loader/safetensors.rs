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

// Sibling module in `src/loader/` — `mod.rs` is owned by other agents, so we
// declare it inline here via `#[path]` to avoid touching that file.
#[path = "mxfp4_pairing.rs"]
pub mod mxfp4_pairing;

use mxfp4_pairing::{
    scan_mxfp4_pairs, CandidateTensor, Mxfp4Pair, Mxfp4PairMap, Mxfp4ScalesSidecarSet,
};
use crate::loader::gguf::GgmlDType;

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
        self.tensors
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let view = self
            .tensors
            .tensor(name)
            .map_err(LoaderError::SafeTensors)?;
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
    gllm_config: Option<Value>,
    gllm_tokenizer_config: Option<Value>,
    /// MXFP4 pair map — logical `_blocks` name → `Mxfp4Pair` metadata.
    ///
    /// Built once at load time by scanning the tensor index for
    /// `*_blocks` / `*_scales` sibling pairs (OpenAI gpt-oss layout).
    mxfp4_pairs: Mxfp4PairMap,
    /// `_scales` tensor names consumed as mxfp4 sidecars. These are
    /// hidden from `iter_tensors()` / `tensor_info()` so the upload path
    /// does not treat them as standalone U8 weights.
    mxfp4_sidecars: Mxfp4ScalesSidecarSet,
    /// `_blocks` logical name → `_scales` sidecar name. Spec-mandated
    /// explicit mapping for diagnostic / template-binding consumers.
    mxfp4_scale_map: HashMap<String, String>,
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
                    .map_err(LoaderError::SafeTensors)?;
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

        let gllm_config = parse_namespace_metadata(&files, &["gllm.config", "_gllm_config"])?;
        let gllm_tokenizer_config =
            parse_namespace_metadata(&files, &["gllm.tokenizer", "_gllm_tokenizer"])?;

        // Scan for mxfp4 `_blocks` / `_scales` pairs (OpenAI gpt-oss layout).
        // The scanner is purely structural — it pairs tensors by name suffix +
        // byte-length invariant, no format-level metadata required.
        let candidates: Vec<CandidateTensor> = index
            .iter()
            .map(|(name, loc)| CandidateTensor {
                name: name.clone(),
                dtype: loc.dtype,
                shape: loc.shape.clone(),
                byte_len: byte_len_of(loc.dtype, &loc.shape),
            })
            .collect();
        let scan = scan_mxfp4_pairs(candidates);

        Ok(Self {
            files,
            index,
            gllm_config,
            gllm_tokenizer_config,
            mxfp4_pairs: scan.pairs,
            mxfp4_sidecars: scan.sidecars,
            mxfp4_scale_map: scan.blocks_to_scales,
        })
    }

    /// Returns the mxfp4 pair metadata for a logical `_blocks` tensor, if any.
    ///
    /// Used by diagnostic tooling and by the in-process upload path to locate
    /// the companion `_scales` tensor and repack bytes into the GGUF-style
    /// interleaved layout expected by the cpu_backend Mxfp4 dequantize path.
    pub fn mxfp4_pair(&self, blocks_name: &str) -> Option<&Mxfp4Pair> {
        self.mxfp4_pairs.get(blocks_name)
    }

    /// Returns the full `blocks_name → scales_name` map (spec-mandated).
    pub fn mxfp4_scale_map(&self) -> &HashMap<String, String> {
        &self.mxfp4_scale_map
    }

    /// Returns `true` if `name` is a `_scales` tensor claimed as an mxfp4
    /// sidecar (hidden from the regular enumeration / upload path).
    pub fn is_mxfp4_sidecar(&self, name: &str) -> bool {
        self.mxfp4_sidecars.contains(name)
    }

    /// Read raw bytes for a tensor ignoring the mxfp4 pairing rewrite.
    ///
    /// Used by [`Self::load_mxfp4_repacked_bytes`] to fetch the underlying
    /// physical `_blocks` and `_scales` byte slices before repacking into
    /// the GGUF-style interleaved layout. Kept private — external callers
    /// should go through the `TensorProvider` API.
    fn raw_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let location = self
            .index
            .get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        let file = &self.files[location.file_idx];
        let view = file
            .tensors
            .tensor(name)
            .map_err(LoaderError::SafeTensors)?;
        let data: &[u8] = view.data();
        Ok(data)
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

    /// Ω1: 读取量化位宽（保留向后兼容）
    ///
    /// 推荐使用 `quantization_metadata()` 获取完整的量化信息
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

    /// Ω1: 读取完整的量化元数据
    ///
    /// 从 safetensors 元数据中读取 `gllm.quantization` 字段
    pub fn quantization_metadata(
        &self,
    ) -> super::Result<Option<HashMap<String, super::QuantizationMetadata>>> {
        let mut out = HashMap::new();
        let mut has_metadata = false;

        for file in &self.files {
            let Some(meta) = file.metadata() else {
                continue;
            };

            // 尝试读取新的量化元数据格式
            if let Some(metadata_map) = super::QuantizationMetadata::from_metadata(meta)? {
                out.extend(metadata_map);
                has_metadata = true;
            }
        }

        if has_metadata {
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    pub fn gllm_config(&self) -> Option<&Value> {
        self.gllm_config.as_ref()
    }

    pub fn gllm_tokenizer_config(&self) -> Option<&Value> {
        self.gllm_tokenizer_config.as_ref()
    }

    /// Ω1: 从实际张量中检测 dtype。
    ///
    /// 优先检测权重张量的实际 dtype，而非依赖 config.json。
    /// 返回 `DType` 枚举而非原始字节数。
    pub fn detect_weight_dtype(&self) -> Option<gllm_kernels::types::DType> {
        use gllm_kernels::types::DType;
        // 优先查找模型权重张量（排除量化张量）
        let weight_names = self
            .names()
            .into_iter()
            .filter(|name| {
                // 排除量化相关张量
                !name.contains("qweight")
                && !name.contains("qzeros")
                && !name.contains("scales")
                && !name.contains("g_idx")
                // 排除量化张量名称模式
                && !name.contains(".q")
                && !name.contains("_q4")
                && !name.contains("_q8")
                && !name.contains("_q2")
                && !name.contains("_q3")
                && !name.contains("_q5")
                && !name.contains("_q6")
            })
            .collect::<Vec<_>>();

        for name in weight_names {
            if let Some(meta) = self.tensor_meta(&name) {
                return match meta.dtype {
                    safetensors::Dtype::BF16 => Some(DType::BF16),
                    safetensors::Dtype::F16 => Some(DType::F16),
                    safetensors::Dtype::F32 => Some(DType::F32),
                    safetensors::Dtype::F64 => Some(DType::F32), // f64 降级到 f32
                    _ => None,
                };
            }
        }
        None
    }
}

impl super::TensorProvider for SafeTensorsLoader {
    fn tensor_info(&self, name: &str) -> Option<super::TensorMeta> {
        // ARCH-MXFP4-SEPARATE: _scales tensors are NOT hidden — both _blocks
        // and _scales are exposed as independent U8 tensors for the JIT path.
        let meta = self.tensor_meta(name)?;
        Some(super::TensorMeta {
            name: name.to_string(),
            shape: meta.shape.clone(),
            dtype: meta.dtype,
        })
    }

    fn iter_tensors(&self) -> impl Iterator<Item = super::TensorMeta> {
        // ARCH-MXFP4-SEPARATE: all tensors visible, including _scales.
        self.index
            .iter()
            .map(|(name, meta)| super::TensorMeta {
                name: name.clone(),
                shape: meta.shape.clone(),
                dtype: meta.dtype,
            })
    }

    fn load_tensor_data(&self, name: &str) -> super::Result<Cow<'_, [u8]>> {
        // ARCH-MXFP4-SEPARATE: both _blocks and _scales serve their raw bytes.
        // No repacking to GGUF interleaved format — JIT consumes separate arrays.
        let tensor = self.tensor(name)?;
        Ok(Cow::Borrowed(tensor.data))
    }

    /// ARCH-MXFP4-SEPARATE: both _blocks and _scales report GgmlDType::MXFP4
    /// so they route through the quantized upload path (storing raw bytes in
    /// WeightsHandle.quantized). The JIT lower consumes them as separate pointers.
    /// No repacking — raw bytes stored as-is.
    fn ggml_dtype(&self, name: &str) -> Option<GgmlDType> {
        if self.mxfp4_pairs.contains_key(name) || self.mxfp4_sidecars.contains(name) {
            Some(GgmlDType::MXFP4)
        } else {
            None
        }
    }
}

/// Byte length of a dense tensor stored in the safetensors file, computed
/// from dtype × product of shape dims. Used by the mxfp4 pair scanner.
fn byte_len_of(dtype: Dtype, shape: &[usize]) -> usize {
    let elem_size = match dtype {
        Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
        Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        // Any exotic dtype falls back to 1 — the pair scanner treats mismatched
        // byte ratios as non-pairs, so an over-conservative size just means the
        // candidate is rejected.
        _ => 1,
    };
    shape.iter().product::<usize>() * elem_size
}

fn parse_namespace_metadata(files: &[MappedSafetensors], keys: &[&str]) -> Result<Option<Value>> {
    let mut merged: Option<Value> = None;
    for file in files {
        let Some(meta) = file.metadata() else {
            continue;
        };

        for key in keys {
            let Some(encoded) = meta.get(*key) else {
                continue;
            };
            let parsed: Value = serde_json::from_str(encoded).map_err(|err| {
                LoaderError::InvalidQuantization(format!(
                    "invalid metadata json for {key} in {}: {err}",
                    file.path().display()
                ))
            })?;

            if let Some(existing) = &merged {
                if existing != &parsed {
                    return Err(LoaderError::InvalidQuantization(format!(
                        "conflicting metadata value for {key} across safetensors shards"
                    )));
                }
            } else {
                merged = Some(parsed);
            }
            break;
        }
    }
    Ok(merged)
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

#[cfg(test)]
mod mxfp4_integration_tests {
    //! End-to-end tests for mxfp4 pair detection + bias/sinks loading through
    //! the real `SafeTensorsLoader` + `TensorProvider` surface.
    //!
    //! Exercises the full rewire: load a file with `<prefix>_blocks` /
    //! `<prefix>_scales` / `<prefix>_bias` / `self_attn.sinks` + `*.bias`
    //! weights, then verify that:
    //! - `_scales` is hidden from enumeration,
    //! - `_blocks` is exposed as MXFP4 via `ggml_dtype()` and repacked into
    //!   GGUF-style interleaved layout on load,
    //! - bias / sinks tensors load unchanged through the regular path,
    //! - the explicit `blocks → scales` map is populated for consumers.

    use super::*;
    use crate::loader::TensorProvider;
    use ::safetensors::tensor::{serialize_to_file, TensorView};
    use ::safetensors::Dtype;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Build a minimal gpt-oss-style single-layer safetensors file on disk.
    ///
    /// Emits (for layer 0):
    ///   - `mlp.experts.gate_up_proj_blocks` (U8)
    ///   - `mlp.experts.gate_up_proj_scales` (U8)
    ///   - `mlp.experts.gate_up_proj_bias`   (BF16)
    ///   - `self_attn.q_proj.weight`         (BF16)
    ///   - `self_attn.q_proj.bias`           (BF16)
    ///   - `self_attn.sinks`                 (BF16, [num_heads])
    ///
    /// Returns (path, blocks_bytes, scales_bytes, bias_bytes, sinks_bytes).
    fn write_gpt_oss_like_fixture(
        dir: &std::path::Path,
        num_experts: usize,
        num_blocks_per_expert: usize,
        block_size: usize,
        num_heads: usize,
    ) -> (std::path::PathBuf, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        let bytes_per_block = block_size / 2;
        let total_blocks = num_experts * num_blocks_per_expert;
        let blocks_bytes: Vec<u8> = (0..total_blocks * bytes_per_block)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(7))
            .collect();
        let scales_bytes: Vec<u8> = (0..total_blocks)
            .map(|i| (i as u8).wrapping_mul(13).wrapping_add(128))
            .collect();

        let bias_elems = num_experts * (2 * num_blocks_per_expert).max(1);
        let bias_bytes: Vec<u8> = (0..bias_elems * 2)
            .map(|i| (i as u8).wrapping_mul(17))
            .collect();

        let qw_rows = 16usize;
        let qw_cols = 16usize;
        let qw_bytes: Vec<u8> = (0..qw_rows * qw_cols * 2).map(|i| (i as u8) ^ 0x5A).collect();
        let qb_bytes: Vec<u8> = (0..qw_cols * 2).map(|i| (i as u8).wrapping_add(3)).collect();
        let sinks_bytes: Vec<u8> = (0..num_heads * 2).map(|i| (i as u8).wrapping_add(9)).collect();

        let blocks_view = TensorView::new(
            Dtype::U8,
            vec![num_experts, num_blocks_per_expert, bytes_per_block],
            &blocks_bytes,
        )
        .expect("blocks view");
        let scales_view =
            TensorView::new(Dtype::U8, vec![num_experts, num_blocks_per_expert], &scales_bytes)
                .expect("scales view");
        let bias_view = TensorView::new(
            Dtype::BF16,
            vec![num_experts, (2 * num_blocks_per_expert).max(1)],
            &bias_bytes,
        )
        .expect("bias view");
        let qw_view =
            TensorView::new(Dtype::BF16, vec![qw_rows, qw_cols], &qw_bytes).expect("qw view");
        let qb_view = TensorView::new(Dtype::BF16, vec![qw_cols], &qb_bytes).expect("qb view");
        let sinks_view =
            TensorView::new(Dtype::BF16, vec![num_heads], &sinks_bytes).expect("sinks view");

        let path = dir.join("model.safetensors");
        let metadata: Option<HashMap<String, String>> = None;
        serialize_to_file(
            vec![
                ("model.layers.0.mlp.experts.gate_up_proj_blocks", blocks_view),
                ("model.layers.0.mlp.experts.gate_up_proj_scales", scales_view),
                ("model.layers.0.mlp.experts.gate_up_proj_bias", bias_view),
                ("model.layers.0.self_attn.q_proj.weight", qw_view),
                ("model.layers.0.self_attn.q_proj.bias", qb_view),
                ("model.layers.0.self_attn.sinks", sinks_view),
            ],
            &metadata,
            &path,
        )
        .expect("write safetensors");

        (path, blocks_bytes, scales_bytes, bias_bytes, sinks_bytes)
    }

    #[test]
    fn mxfp4_pair_detected_and_both_tensors_visible() {
        let dir = TempDir::new().expect("temp dir");
        let (path, _blocks, _scales, _bias, _sinks) =
            write_gpt_oss_like_fixture(dir.path(), 2, 2, 32, 4);
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        let map = loader.mxfp4_scale_map();
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get("model.layers.0.mlp.experts.gate_up_proj_blocks")
                .map(String::as_str),
            Some("model.layers.0.mlp.experts.gate_up_proj_scales"),
        );

        let pair = loader
            .mxfp4_pair("model.layers.0.mlp.experts.gate_up_proj_blocks")
            .expect("pair present");
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.num_blocks, 2 * 2);
        assert_eq!(
            pair.bias_name.as_deref(),
            Some("model.layers.0.mlp.experts.gate_up_proj_bias"),
        );

        // ARCH-MXFP4-SEPARATE: _scales IS visible in enumeration.
        let enumerated: Vec<String> = loader.iter_tensors().map(|m| m.name).collect();
        assert!(enumerated
            .iter()
            .any(|n| n == "model.layers.0.mlp.experts.gate_up_proj_scales"));
        assert!(enumerated
            .iter()
            .any(|n| n == "model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(enumerated
            .iter()
            .any(|n| n == "model.layers.0.mlp.experts.gate_up_proj_bias"));
    }

    #[test]
    fn mxfp4_blocks_and_scales_serve_raw_bytes_no_repack() {
        let dir = TempDir::new().expect("temp dir");
        let (path, expected_blocks, expected_scales, _bias, _sinks) =
            write_gpt_oss_like_fixture(dir.path(), 2, 2, 32, 4);
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        let blocks_name = "model.layers.0.mlp.experts.gate_up_proj_blocks";
        let scales_name = "model.layers.0.mlp.experts.gate_up_proj_scales";

        // ARCH-MXFP4-SEPARATE: both _blocks and _scales report MXFP4 for quantized upload routing.
        assert_eq!(loader.ggml_dtype(blocks_name), Some(GgmlDType::MXFP4));
        assert_eq!(loader.ggml_dtype(scales_name), Some(GgmlDType::MXFP4));
        assert_eq!(loader.ggml_dtype("model.layers.0.self_attn.q_proj.weight"), None);

        // _blocks returns raw nibbles bytes (not GGUF interleaved).
        let blocks_data = loader.load_tensor_data(blocks_name).expect("load blocks");
        let blocks_got: &[u8] = blocks_data.as_ref();
        assert_eq!(blocks_got, expected_blocks.as_slice());

        // _scales returns raw scale bytes (not hidden, not error).
        let scales_data = loader.load_tensor_data(scales_name).expect("load scales");
        let scales_got: &[u8] = scales_data.as_ref();
        assert_eq!(scales_got, expected_scales.as_slice());
    }

    #[test]
    fn attention_bias_and_sinks_load_through_regular_path() {
        let dir = TempDir::new().expect("temp dir");
        let num_heads = 4usize;
        let (path, _blocks, _scales, bias_bytes, sinks_bytes) =
            write_gpt_oss_like_fixture(dir.path(), 2, 2, 32, num_heads);
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        let qb = loader
            .tensor_info("model.layers.0.self_attn.q_proj.bias")
            .expect("q_proj.bias must be visible");
        assert_eq!(qb.dtype, Dtype::BF16);
        assert_eq!(qb.shape.len(), 1, "1-D bias");

        let sinks_meta = loader
            .tensor_info("model.layers.0.self_attn.sinks")
            .expect("sinks must be visible");
        assert_eq!(sinks_meta.dtype, Dtype::BF16);
        assert_eq!(sinks_meta.shape, vec![num_heads]);

        let sinks_data = loader
            .load_tensor_data("model.layers.0.self_attn.sinks")
            .expect("load sinks");
        let sinks_got: &[u8] = sinks_data.as_ref();
        assert_eq!(sinks_got, sinks_bytes.as_slice());

        let bias_data = loader
            .load_tensor_data("model.layers.0.mlp.experts.gate_up_proj_bias")
            .expect("load expert bias");
        let bias_got: &[u8] = bias_data.as_ref();
        assert_eq!(bias_got, bias_bytes.as_slice());
        assert_eq!(
            loader
                .tensor_info("model.layers.0.mlp.experts.gate_up_proj_bias")
                .expect("bias visible")
                .dtype,
            Dtype::BF16,
        );
    }

    #[test]
    fn mxfp4_blocks_and_scales_upload_via_quantized_path() {
        // ARCH-MXFP4-SEPARATE: both _blocks and _scales route through the
        // quantized upload path (ggml_dtype=MXFP4), storing raw separated bytes.
        use crate::compat::cpu_backend::CpuBackend;
        use crate::loader::{Loader, WeightFormat};
        use crate::manifest::ModelManifest;

        let dir = TempDir::new().expect("temp dir");
        let (path, expected_blocks, expected_scales, _bias, _sinks) =
            write_gpt_oss_like_fixture(dir.path(), 2, 2, 32, 4);
        let mut loader = Loader::new(ModelManifest::default())
            .with_weights(vec![path])
            .load()
            .expect("loader load");
        assert_eq!(loader.weight_format(), WeightFormat::SafeTensors);

        let backend = CpuBackend::<f32>::new();
        let handle = loader.upload_weights::<_, f32>(&backend).expect("upload");

        let blocks_name = "model.layers.0.mlp.experts.gate_up_proj_blocks";
        let scales_name = "model.layers.0.mlp.experts.gate_up_proj_scales";

        // Both _blocks and _scales are quantized (routed via MXFP4 ggml_dtype).
        assert!(handle.is_quantized(blocks_name));
        assert!(handle.is_quantized(scales_name));

        // _blocks stores raw nibbles bytes (NOT GGUF interleaved).
        let blocks_qt = handle.quantized_tensor(blocks_name).expect("blocks in quantized");
        assert_eq!(blocks_qt.data, expected_blocks);

        // _scales stores raw scale bytes (separate from blocks).
        let scales_qt = handle.quantized_tensor(scales_name).expect("scales in quantized");
        assert_eq!(scales_qt.data, expected_scales);

        // Both have shape info.
        assert!(handle.tensor_shape(blocks_name).is_some());
        assert!(handle.tensor_shape(scales_name).is_some());

        // Bias uploaded as regular BF16 tensor.
        let bias_name = "model.layers.0.mlp.experts.gate_up_proj_bias";
        assert!(!handle.is_quantized(bias_name));
        assert!(handle.tensor_shape(bias_name).is_some());

        // Sinks also uploaded.
        let sinks_name = "model.layers.0.self_attn.sinks";
        assert!(!handle.is_quantized(sinks_name));
        assert!(handle.tensor_shape(sinks_name).is_some());
    }
}
