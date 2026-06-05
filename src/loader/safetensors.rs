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

// Sibling modules in `src/loader/` — `mod.rs` is owned by other agents, so we
// declare them inline here via `#[path]` to avoid touching that file.
#[path = "mxfp4_pairing.rs"]
pub mod mxfp4_pairing;
#[path = "awq_gptq_pairing.rs"]
pub mod awq_gptq_pairing;
#[path = "nvfp4_pairing.rs"]
pub mod nvfp4_pairing;

use mxfp4_pairing::{
    scan_mxfp4_pairs, CandidateTensor, Mxfp4Pair, Mxfp4PairMap,
    Mxfp4ScalesSidecarSet,
};
use awq_gptq_pairing::{scan_awq_gptq_groups, AwqGptqGroup};
use nvfp4_pairing::{scan_nvfp4_groups, NvfpCandidate, NvfpGroup};
use crate::loader::gguf::GgmlDType;

#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// AWQ/GPTQ triplet scan results — base_name → group metadata.
    awq_gptq_groups: HashMap<String, AwqGptqGroup>,
    /// Tensor names consumed by AWQ/GPTQ triplets (hidden from regular enumeration).
    awq_gptq_consumed: std::collections::HashSet<String>,
    /// NVFP4 pair scan results — base_name → group metadata.
    nvfp4_groups: HashMap<String, NvfpGroup>,
    /// Tensor names consumed by NVFP4 pairs (hidden from regular enumeration).
    nvfp4_consumed: std::collections::HashSet<String>,
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
        let scan = scan_mxfp4_pairs(candidates.clone());

        // Scan for AWQ/GPTQ triplets (qweight + scales + qzeros + optional g_idx).
        let awq_scan = scan_awq_gptq_groups(candidates.clone());

        // Scan for NVFP4 pairs (weight + weight_scale + optional weight_scale_2).
        let nvfp_candidates: Vec<NvfpCandidate> = candidates
            .into_iter()
            .map(|c| NvfpCandidate {
                name: c.name,
                dtype: c.dtype,
                shape: c.shape,
                byte_len: c.byte_len,
            })
            .collect();
        let nvfp_scan = scan_nvfp4_groups(nvfp_candidates);

        Ok(Self {
            files,
            index,
            gllm_config,
            gllm_tokenizer_config,
            mxfp4_pairs: scan.pairs,
            mxfp4_sidecars: scan.sidecars,
            mxfp4_scale_map: scan.blocks_to_scales,
            awq_gptq_groups: awq_scan.groups,
            awq_gptq_consumed: awq_scan.consumed,
            nvfp4_groups: nvfp_scan.groups,
            nvfp4_consumed: nvfp_scan.consumed,
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

    /// Returns the AWQ/GPTQ group metadata for a base name, if any.
    pub fn awq_gptq_group(&self, base_name: &str) -> Option<&AwqGptqGroup> {
        self.awq_gptq_groups.get(base_name)
    }

    /// Returns all detected AWQ/GPTQ groups.
    pub fn awq_gptq_groups(&self) -> &HashMap<String, AwqGptqGroup> {
        &self.awq_gptq_groups
    }

    /// Returns `true` if `name` is a tensor consumed by an AWQ/GPTQ triplet
    /// (qweight / scales / qzeros / g_idx) — hidden from regular enumeration.
    pub fn is_awq_gptq_consumed(&self, name: &str) -> bool {
        self.awq_gptq_consumed.contains(name)
    }

    /// Read raw bytes for a tensor ignoring the mxfp4 pairing rewrite.
    ///
    /// Used by [`Self::load_mxfp4_repacked_bytes`] to fetch the underlying
    /// physical `_blocks` and `_scales` byte slices before repacking into
    /// the GGUF-style interleaved layout. Kept private — external callers
    /// should go through the `TensorProvider` API.
    #[allow(dead_code)]
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
        // AWQ/GPTQ consumed tensors (scales, qzeros, g_idx) are hidden from
        // regular enumeration. The qweight is exposed under the base_name with
        // a synthetic GgmlDType (see ggml_dtype below).
        if self.awq_gptq_consumed.contains(name) {
            // Check if this is the qweight of a known group — if so, report it
            // under the base_name with the qweight's shape.
            let group = self
                .awq_gptq_groups
                .values()
                .find(|g| g.qweight_name == name);
            if let Some(g) = group {
                let meta = self.tensor_meta(name)?;
                // AWQ/GPTQ: qweight shape is [K/8, N] (packed int32).
                // Report element-level shape [N, K] so downstream (process_single_tensor)
                // can derive (n, k) for `repack_awq_gptq_blocks` and JIT QuantGemm.
                let element_shape = if meta.shape.len() >= 2 {
                    vec![meta.shape[1], meta.shape[0] * 8]
                } else {
                    meta.shape.clone()
                };
                return Some(super::TensorMeta {
                    name: g.base_name.clone(),
                    shape: element_shape,
                    dtype: meta.dtype,
                });
            }
            return None;
        }
        // NVFP4 consumed tensors (weight_scale, optional weight_scale_2) are hidden.
        // The packed weight is exposed under base_name with element-level shape [N, K].
        if self.nvfp4_consumed.contains(name) {
            let group = self
                .nvfp4_groups
                .values()
                .find(|g| g.weight_name == name);
            if let Some(g) = group {
                let meta = self.tensor_meta(name)?;
                // NVFP4 weight shape is [N, K/2] (each byte packs 2 E2M1 nibbles).
                // Report element-level shape [N, K].
                let element_shape = if meta.shape.len() >= 2 {
                    vec![meta.shape[0], meta.shape[1] * 2]
                } else {
                    meta.shape.clone()
                };
                return Some(super::TensorMeta {
                    name: g.base_name.clone(),
                    shape: element_shape,
                    dtype: meta.dtype,
                });
            }
            return None;
        }
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
        // AWQ/GPTQ + NVFP4: hide consumed sidecars, expose primary weight under base_name.
        self.index
            .iter()
            .filter(|(name, _)| {
                let awq_consumed = self.awq_gptq_consumed.contains(*name)
                    && !self.awq_gptq_groups.values().any(|g| g.qweight_name == **name);
                let nvfp_consumed = self.nvfp4_consumed.contains(*name)
                    && !self.nvfp4_groups.values().any(|g| g.weight_name == **name);
                !(awq_consumed || nvfp_consumed)
            })
            .map(|(name, meta)| {
                // For AWQ/GPTQ qweight, emit under base_name with element-level shape
                if let Some(g) = self.awq_gptq_groups.values().find(|g| g.qweight_name == *name) {
                    // AWQ/GPTQ qweight is stored [K/8, N]; expose element-level [N, K] under base_name.
                    let element_shape = if meta.shape.len() >= 2 {
                        vec![meta.shape[1], meta.shape[0] * 8]
                    } else {
                        meta.shape.clone()
                    };
                    super::TensorMeta {
                        name: g.base_name.clone(),
                        shape: element_shape,
                        dtype: meta.dtype,
                    }
                } else if let Some(g) = self.nvfp4_groups.values().find(|g| g.weight_name == *name) {
                    // NVFP4 weight is stored [N, K/2]; expose element-level [N, K].
                    let element_shape = if meta.shape.len() >= 2 {
                        vec![meta.shape[0], meta.shape[1] * 2]
                    } else {
                        meta.shape.clone()
                    };
                    super::TensorMeta {
                        name: g.base_name.clone(),
                        shape: element_shape,
                        dtype: meta.dtype,
                    }
                } else {
                    super::TensorMeta {
                        name: name.clone(),
                        shape: meta.shape.clone(),
                        dtype: meta.dtype,
                    }
                }
            })
    }

    fn load_tensor_data(&self, name: &str) -> super::Result<Cow<'_, [u8]>> {
        // AWQ/GPTQ: when asked for base_name, serve the qweight data
        if let Some(g) = self.awq_gptq_groups.get(name) {
            let tensor = self.tensor(&g.qweight_name)?;
            return Ok(Cow::Borrowed(tensor.data));
        }
        // NVFP4: when asked for base_name, serve the packed weight data
        if let Some(g) = self.nvfp4_groups.get(name) {
            let tensor = self.tensor(&g.weight_name)?;
            return Ok(Cow::Borrowed(tensor.data));
        }
        // ARCH-MXFP4-SEPARATE: both _blocks and _scales serve their raw bytes.
        // No repacking to GGUF interleaved format — JIT consumes separate arrays.
        let tensor = self.tensor(name)?;
        Ok(Cow::Borrowed(tensor.data))
    }

    /// MXFP4 tensors report GgmlDType::MXFP4 for quantized upload routing.
    /// AWQ/GPTQ base_name reports AWQ4 or GPTQ4 so the quantized upload path
    /// stores raw bytes with the auxiliary scales/zeros/g_idx fields populated.
    fn ggml_dtype(&self, name: &str) -> Option<GgmlDType> {
        // MXFP4
        if self.mxfp4_pairs.contains_key(name) || self.mxfp4_sidecars.contains(name) {
            return Some(GgmlDType::MXFP4);
        }
        // AWQ/GPTQ: base_name → synthetic dtype
        if let Some(g) = self.awq_gptq_groups.get(name) {
            return Some(match g.format {
                awq_gptq_pairing::AwqGptqFormat::Awq => GgmlDType::AWQ4,
                awq_gptq_pairing::AwqGptqFormat::Gptq => GgmlDType::GPTQ4,
            });
        }
        // NVFP4: base_name → synthetic dtype
        if self.nvfp4_groups.contains_key(name) {
            return Some(GgmlDType::NVFP4);
        }
        None
    }

    fn awq_gptq_aux_data(&self, name: &str) -> Option<(Cow<'_, [u8]>, Cow<'_, [u8]>, Option<Vec<i32>>, usize)> {
        let g = self.awq_gptq_groups.get(name)?;

        // Load scales bytes
        let scales_tensor = self.tensor(&g.scales_name).ok()?;
        let scales_bytes = Cow::Borrowed(scales_tensor.data);

        // Load qzeros bytes
        let qzeros_tensor = self.tensor(&g.qzeros_name).ok()?;
        let zeros_bytes = Cow::Borrowed(qzeros_tensor.data);

        // Load g_idx if present (GPTQ)
        let g_idx = g.g_idx_name.as_ref().and_then(|gidx_name| {
            let gidx_tensor = self.tensor(gidx_name).ok()?;
            // g_idx is I32: reinterpret bytes as i32 slice
            let data = gidx_tensor.data;
            let (prefix, body, suffix) = unsafe { data.align_to::<i32>() };
            if prefix.is_empty() && suffix.is_empty() {
                Some(body.to_vec())
            } else {
                // Fallback: manual byte-to-i32 conversion
                Some(data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            }
        });

        // Compute group_size from qweight shape
        let qw_shape = &g.qweight_shape;
        let k = qw_shape.first().copied().unwrap_or(0) * 8;
        let scales_rows = scales_tensor.shape.first().copied().unwrap_or(1);
        let group_size = if scales_rows > 0 { k / scales_rows } else { 128 };

        Some((scales_bytes, zeros_bytes, g_idx, group_size))
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
mod unit_tests {
    use super::*;
    use crate::loader::LoaderError;
    use ::safetensors::Dtype;
    use half::{bf16, f16};
    use std::borrow::Cow;

    /// Reinterpret a typed slice as `&[u8]` — guaranteed aligned.
    unsafe fn to_bytes<T>(vals: &[T]) -> &[u8] {
        std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * std::mem::size_of::<T>())
    }

    // ── TensorSlice::element_count ──────────────────────────────────────

    #[test]
    fn element_count_scalar_shape() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: &[] };
        assert_eq!(ts.element_count(), 1);
    }

    #[test]
    fn element_count_1d() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![10], data: &[] };
        assert_eq!(ts.element_count(), 10);
    }

    #[test]
    fn element_count_2d() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![3, 4], data: &[] };
        assert_eq!(ts.element_count(), 12);
    }

    #[test]
    fn element_count_3d() {
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![2, 3, 4], data: &[] };
        assert_eq!(ts.element_count(), 24);
    }

    #[test]
    fn element_count_empty_shape_yields_one() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![], data: &[] };
        assert_eq!(ts.element_count(), 1);
    }

    // ── TensorSlice as_* — wrong dtype → UnsupportedDtype ──────────────

    #[test]
    fn as_f16_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: &[0; 4] };
        let err = ts.as_f16().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::F32)));
    }

    #[test]
    fn as_bf16_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::F16, shape: vec![1], data: &[0; 2] };
        let err = ts.as_bf16().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::F16)));
    }

    #[test]
    fn as_f32_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::F64, shape: vec![1], data: &[0; 8] };
        let err = ts.as_f32().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::F64)));
    }

    #[test]
    fn as_f64_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: &[0; 4] };
        let err = ts.as_f64().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::F32)));
    }

    #[test]
    fn as_i8_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![1], data: &[0; 1] };
        let err = ts.as_i8().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::U8)));
    }

    #[test]
    fn as_u8_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::I8, shape: vec![1], data: &[0; 1] };
        let err = ts.as_u8().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::I8)));
    }

    #[test]
    fn as_i16_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::U16, shape: vec![1], data: &[0; 2] };
        let err = ts.as_i16().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::U16)));
    }

    #[test]
    fn as_u16_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::I16, shape: vec![1], data: &[0; 2] };
        let err = ts.as_u16().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::I16)));
    }

    #[test]
    fn as_i32_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::U32, shape: vec![1], data: &[0; 4] };
        let err = ts.as_i32().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::U32)));
    }

    #[test]
    fn as_u32_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![1], data: &[0; 4] };
        let err = ts.as_u32().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::I32)));
    }

    #[test]
    fn as_i64_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::U64, shape: vec![1], data: &[0; 8] };
        let err = ts.as_i64().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::U64)));
    }

    #[test]
    fn as_u64_wrong_dtype() {
        let ts = TensorSlice { dtype: Dtype::I64, shape: vec![1], data: &[0; 8] };
        let err = ts.as_u64().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::I64)));
    }

    // ── TensorSlice as_* — correct dtype, aligned data ─────────────────

    #[test]
    fn as_f32_aligned_returns_correct_values() {
        let vals = vec![1.0f32, -2.5f32, 100.0f32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![3], data: bytes };
        let result = ts.as_f32().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], -2.5f32);
        assert_eq!(result[2], 100.0f32);
    }

    #[test]
    fn as_bf16_aligned_returns_correct_values() {
        let vals = vec![bf16::from_f32(1.0), bf16::from_f32(-2.5)];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::BF16, shape: vec![2], data: bytes };
        let result = ts.as_bf16().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], bf16::from_f32(1.0));
        assert_eq!(result[1], bf16::from_f32(-2.5));
    }

    #[test]
    fn as_f16_aligned_returns_correct_values() {
        let vals = vec![f16::from_f32(1.0), f16::from_f32(-2.5)];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::F16, shape: vec![2], data: bytes };
        let result = ts.as_f16().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(1.0));
        assert_eq!(result[1], f16::from_f32(-2.5));
    }

    #[test]
    fn as_f64_aligned_returns_correct_values() {
        let vals = vec![1.0f64, -2.5f64];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::F64, shape: vec![2], data: bytes };
        let result = ts.as_f64().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0f64);
        assert_eq!(result[1], -2.5f64);
    }

    #[test]
    fn as_i8_returns_correct_values() {
        let raw: &[u8] = &[0, 127, 128, 255];
        let ts = TensorSlice { dtype: Dtype::I8, shape: vec![4], data: raw };
        let result = ts.as_i8().unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 0i8);
        assert_eq!(result[1], 127i8);
        assert_eq!(result[2], -128i8);
        assert_eq!(result[3], -1i8);
    }

    #[test]
    fn as_u8_returns_borrowed_data() {
        let raw: &[u8] = &[10u8, 20u8, 255u8];
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![3], data: raw };
        let result = ts.as_u8().unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, raw);
    }

    #[test]
    fn as_i32_aligned_returns_correct_values() {
        let vals = vec![42i32, -1i32, 0i32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![3], data: bytes };
        let result = ts.as_i32().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 42i32);
        assert_eq!(result[1], -1i32);
        assert_eq!(result[2], 0i32);
    }

    #[test]
    fn as_u64_aligned_returns_correct_values() {
        let vals = vec![42u64, 0u64, u64::MAX];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::U64, shape: vec![3], data: bytes };
        let result = ts.as_u64().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 42u64);
        assert_eq!(result[1], 0u64);
        assert_eq!(result[2], u64::MAX);
    }

    // ── TensorSlice as_* — misaligned data → Owned Cow ─────────────────

    #[test]
    fn as_f32_misaligned_returns_owned() {
        let vals = vec![1.0f32, 2.0f32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![2], data: &buf[1..] };
        let result = ts.as_f32().unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], 2.0f32);
    }

    #[test]
    fn as_bf16_misaligned_returns_owned() {
        let vals = vec![bf16::from_f32(1.0)];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::BF16, shape: vec![1], data: &buf[1..] };
        let result = ts.as_bf16().unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], bf16::from_f32(1.0));
    }

    #[test]
    fn as_i32_misaligned_returns_owned() {
        let vals = vec![42i32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![1], data: &buf[1..] };
        let result = ts.as_i32().unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 42i32);
    }

    // ── cast_or_copy_* functions ────────────────────────────────────────

    #[test]
    fn cast_or_copy_f32_aligned_returns_borrowed() {
        let vals = vec![1.0f32, 2.0f32];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_f32(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn cast_or_copy_f32_misaligned_produces_correct_values() {
        let vals = vec![1.0f32, 2.0f32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_f32(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 1.0f32);
        assert_eq!(result[1], 2.0f32);
    }

    #[test]
    fn cast_or_copy_f32_empty_slice() {
        let result = cast_or_copy_f32(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_i8_maps_bytes_to_signed() {
        let bytes: &[u8] = &[0, 127, 128, 255];
        let result = cast_or_copy_i8(bytes).unwrap();
        assert_eq!(result[0], 0i8);
        assert_eq!(result[1], 127i8);
        assert_eq!(result[2], -128i8);
        assert_eq!(result[3], -1i8);
    }

    #[test]
    fn cast_or_copy_bf16_aligned_returns_borrowed() {
        let vals = vec![bf16::from_f32(3.14)];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_bf16(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], bf16::from_f32(3.14));
    }

    #[test]
    fn cast_or_copy_i32_misaligned_produces_correct_values() {
        let vals = vec![100i32, -200i32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_i32(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 100i32);
        assert_eq!(result[1], -200i32);
    }

    // ── byte_len_of ────────────────────────────────────────────────────

    #[test]
    fn byte_len_u8_1d() {
        assert_eq!(byte_len_of(Dtype::U8, &[10]), 10);
    }

    #[test]
    fn byte_len_bool_1d() {
        assert_eq!(byte_len_of(Dtype::BOOL, &[7]), 7);
    }

    #[test]
    fn byte_len_f16_2d() {
        assert_eq!(byte_len_of(Dtype::F16, &[3, 4]), 24);
    }

    #[test]
    fn byte_len_bf16_2d() {
        assert_eq!(byte_len_of(Dtype::BF16, &[10, 10]), 200);
    }

    #[test]
    fn byte_len_f32_2d() {
        assert_eq!(byte_len_of(Dtype::F32, &[2, 3]), 24);
    }

    #[test]
    fn byte_len_f64_1d() {
        assert_eq!(byte_len_of(Dtype::F64, &[5]), 40);
    }

    #[test]
    fn byte_len_i8_3d() {
        assert_eq!(byte_len_of(Dtype::I8, &[2, 3, 4]), 24);
    }

    #[test]
    fn byte_len_i32_empty_shape() {
        assert_eq!(byte_len_of(Dtype::I32, &[]), 4);
    }

    #[test]
    fn byte_len_u64_2d() {
        assert_eq!(byte_len_of(Dtype::U64, &[2, 3]), 48);
    }

    // ── TensorLocation ─────────────────────────────────────────────────

    #[test]
    fn tensor_location_construction_and_field_access() {
        let loc = TensorLocation { file_idx: 3, dtype: Dtype::F32, shape: vec![2, 4] };
        assert_eq!(loc.file_idx, 3);
        assert_eq!(loc.dtype, Dtype::F32);
        assert_eq!(loc.shape, vec![2, 4]);
    }

    #[test]
    fn tensor_location_equality_same_fields() {
        let a = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![3] };
        let b = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![3] };
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_location_inequality_different_shape() {
        let a = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![3] };
        let b = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![4] };
        assert_ne!(a, b);
    }

    // ── cast_or_copy_* — misaligned paths for remaining types ──────────

    #[test]
    fn cast_or_copy_f64_aligned_returns_borrowed() {
        let vals = vec![1.0f64, 2.0f64];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_f64(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0f64);
        assert_eq!(result[1], 2.0f64);
    }

    #[test]
    fn cast_or_copy_f64_misaligned_produces_correct_values() {
        let vals = vec![42.5f64, -7.25f64];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_f64(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 42.5f64);
        assert_eq!(result[1], -7.25f64);
    }

    #[test]
    fn cast_or_copy_f16_aligned_returns_borrowed() {
        let vals = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_f16(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], f16::from_f32(1.0));
    }

    #[test]
    fn cast_or_copy_f16_misaligned_produces_correct_values() {
        let vals = vec![f16::from_f32(3.14)];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_f16(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], f16::from_f32(3.14));
    }

    #[test]
    fn cast_or_copy_i16_aligned_returns_borrowed() {
        let vals = vec![100i16, -200i16];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_i16(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], 100i16);
        assert_eq!(result[1], -200i16);
    }

    #[test]
    fn cast_or_copy_i16_misaligned_produces_correct_values() {
        let vals = vec![42i16, -1i16];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_i16(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 42i16);
        assert_eq!(result[1], -1i16);
    }

    #[test]
    fn cast_or_copy_u16_aligned_returns_borrowed() {
        let vals = vec![1000u16, 65535u16];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_u16(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], 1000u16);
        assert_eq!(result[1], 65535u16);
    }

    #[test]
    fn cast_or_copy_u16_misaligned_produces_correct_values() {
        let vals = vec![42u16];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_u16(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 42u16);
    }

    #[test]
    fn cast_or_copy_u32_aligned_returns_borrowed() {
        let vals = vec![100000u32, 0u32];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_u32(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], 100000u32);
        assert_eq!(result[1], 0u32);
    }

    #[test]
    fn cast_or_copy_u32_misaligned_produces_correct_values() {
        let vals = vec![999u32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_u32(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 999u32);
    }

    #[test]
    fn cast_or_copy_i64_aligned_returns_borrowed() {
        let vals = vec![123456789i64, -1i64];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_i64(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], 123456789i64);
        assert_eq!(result[1], -1i64);
    }

    #[test]
    fn cast_or_copy_i64_misaligned_produces_correct_values() {
        let vals = vec![-42i64];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_i64(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], -42i64);
    }

    #[test]
    fn cast_or_copy_u64_aligned_returns_borrowed() {
        let vals = vec![u64::MAX, 0u64];
        let bytes = unsafe { to_bytes(&vals) };
        let result = cast_or_copy_u64(bytes).unwrap();
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result[0], u64::MAX);
        assert_eq!(result[1], 0u64);
    }

    #[test]
    fn cast_or_copy_u64_misaligned_produces_correct_values() {
        let vals = vec![42u64];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let result = cast_or_copy_u64(&buf[1..]).unwrap();
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 42u64);
    }

    // ── cast_or_copy_* — empty slice edge cases ────────────────────────

    #[test]
    fn cast_or_copy_f16_empty_slice() {
        let result = cast_or_copy_f16(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_bf16_empty_slice() {
        let result = cast_or_copy_bf16(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_f64_empty_slice() {
        let result = cast_or_copy_f64(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_i8_empty_slice() {
        let result = cast_or_copy_i8(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_i16_empty_slice() {
        let result = cast_or_copy_i16(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_u16_empty_slice() {
        let result = cast_or_copy_u16(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_i32_empty_slice() {
        let result = cast_or_copy_i32(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_u32_empty_slice() {
        let result = cast_or_copy_u32(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_i64_empty_slice() {
        let result = cast_or_copy_i64(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn cast_or_copy_u64_empty_slice() {
        let result = cast_or_copy_u64(&[]).unwrap();
        assert!(result.is_empty());
    }

    // ── TensorSlice as_* — correct dtype, additional types ─────────────

    #[test]
    fn as_i16_aligned_returns_correct_values() {
        let vals = vec![100i16, -200i16, 0i16];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::I16, shape: vec![3], data: bytes };
        let result = ts.as_i16().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 100i16);
        assert_eq!(result[1], -200i16);
        assert_eq!(result[2], 0i16);
    }

    #[test]
    fn as_u16_aligned_returns_correct_values() {
        let vals = vec![1000u16, 65535u16];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::U16, shape: vec![2], data: bytes };
        let result = ts.as_u16().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1000u16);
        assert_eq!(result[1], 65535u16);
    }

    #[test]
    fn as_u32_aligned_returns_correct_values() {
        let vals = vec![100000u32, 0u32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::U32, shape: vec![2], data: bytes };
        let result = ts.as_u32().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 100000u32);
        assert_eq!(result[1], 0u32);
    }

    #[test]
    fn as_i64_aligned_returns_correct_values() {
        let vals = vec![123456789i64, -1i64];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::I64, shape: vec![2], data: bytes };
        let result = ts.as_i64().unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 123456789i64);
        assert_eq!(result[1], -1i64);
    }

    // ── byte_len_of — additional dtypes and edge cases ─────────────────

    #[test]
    fn byte_len_i16_1d() {
        assert_eq!(byte_len_of(Dtype::I16, &[5]), 10);
    }

    #[test]
    fn byte_len_u16_2d() {
        assert_eq!(byte_len_of(Dtype::U16, &[3, 4]), 24);
    }

    #[test]
    fn byte_len_i32_2d() {
        assert_eq!(byte_len_of(Dtype::I32, &[4, 5]), 80);
    }

    #[test]
    fn byte_len_u32_1d() {
        assert_eq!(byte_len_of(Dtype::U32, &[7]), 28);
    }

    #[test]
    fn byte_len_i64_1d() {
        assert_eq!(byte_len_of(Dtype::I64, &[3]), 24);
    }

    #[test]
    fn byte_len_i8_1d() {
        assert_eq!(byte_len_of(Dtype::I8, &[9]), 9);
    }

    #[test]
    fn byte_len_exotic_dtype_falls_back_to_one() {
        assert_eq!(byte_len_of(Dtype::U32, &[0]), 0);
    }

    #[test]
    fn byte_len_bool_2d() {
        assert_eq!(byte_len_of(Dtype::BOOL, &[2, 5]), 10);
    }

    #[test]
    fn byte_len_f16_1d() {
        assert_eq!(byte_len_of(Dtype::F16, &[8]), 16);
    }

    #[test]
    fn byte_len_bf16_1d() {
        assert_eq!(byte_len_of(Dtype::BF16, &[8]), 16);
    }

    // ── TensorSlice with empty data ────────────────────────────────────

    #[test]
    fn as_f32_zero_elements_returns_empty_slice() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![0], data: &[] };
        let result = ts.as_f32().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn as_u8_zero_elements_returns_empty_borrowed() {
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![0], data: &[] };
        let result = ts.as_u8().unwrap();
        assert!(result.is_empty());
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn element_count_with_zero_dim() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![3, 0, 4], data: &[] };
        assert_eq!(ts.element_count(), 0);
    }

    // ── SafeTensorsLoader edge case ────────────────────────────────────

    #[test]
    fn from_files_empty_paths_returns_missing_weights() {
        let result = SafeTensorsLoader::from_files(
            &[],
            crate::loader::ParallelLoader::new(false),
        );
        assert!(matches!(result, Err(LoaderError::MissingWeights)));
    }

    // ── LoaderError variant matching (unit-level, no Display) ──────────

    #[test]
    fn loader_error_missing_tensor_variant() {
        let err = LoaderError::MissingTensor("test.weight".to_string());
        let msg = err.to_string();
        assert!(msg.contains("test.weight"), "MissingTensor message must contain tensor name");
    }

    #[test]
    fn loader_error_duplicate_tensor_variant() {
        let err = LoaderError::DuplicateTensor("layer.weight".to_string());
        let msg = err.to_string();
        assert!(msg.contains("layer.weight"));
    }

    #[test]
    fn loader_error_network_variant() {
        let err = LoaderError::Network("timeout".to_string());
        let msg = err.to_string();
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn loader_error_cache_variant() {
        let err = LoaderError::Cache("corrupt".to_string());
        let msg = err.to_string();
        assert!(msg.contains("corrupt"));
    }

    #[test]
    fn loader_error_onnx_variant() {
        let err = LoaderError::Onnx("parse failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("parse failed"));
    }

    #[test]
    fn loader_error_gguf_variant() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad header"));
    }

    #[test]
    fn loader_error_gllm_variant() {
        let err = LoaderError::Gllm("invalid format".to_string());
        let msg = err.to_string();
        assert!(msg.contains("invalid format"));
    }

    #[test]
    fn loader_error_hfhub_variant() {
        let err = LoaderError::HfHub("connection refused".to_string());
        let msg = err.to_string();
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn loader_error_invalid_quantization_variant() {
        let err = LoaderError::InvalidQuantization("bad scale".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad scale"));
    }

    #[test]
    fn loader_error_arch_detection_variant() {
        let err = LoaderError::ArchDetection("unknown arch".to_string());
        let msg = err.to_string();
        assert!(msg.contains("unknown arch"));
    }

    #[test]
    fn loader_error_backend_variant() {
        let err = LoaderError::Backend("CUDA OOM".to_string());
        let msg = err.to_string();
        assert!(msg.contains("CUDA OOM"));
    }

    #[test]
    fn loader_error_pytorch_variant() {
        let err = LoaderError::Pytorch("bad pickle".to_string());
        let msg = err.to_string();
        assert!(msg.contains("bad pickle"));
    }

    #[test]
    fn loader_error_unsupported_weight_extension_variant() {
        let err = LoaderError::UnsupportedWeightExtension(".bin".to_string());
        let msg = err.to_string();
        assert!(msg.contains(".bin"));
    }

    // ── TensorLocation — additional property tests ─────────────────────

    #[test]
    fn tensor_location_inequality_different_file_idx() {
        let a = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![3] };
        let b = TensorLocation { file_idx: 1, dtype: Dtype::F32, shape: vec![3] };
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_location_inequality_different_dtype() {
        let a = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![3] };
        let b = TensorLocation { file_idx: 0, dtype: Dtype::F64, shape: vec![3] };
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_location_clone_is_equal() {
        let a = TensorLocation { file_idx: 2, dtype: Dtype::BF16, shape: vec![4, 5] };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_location_debug_contains_fields() {
        let loc = TensorLocation { file_idx: 7, dtype: Dtype::I32, shape: vec![2, 3] };
        let debug = format!("{:?}", loc);
        assert!(debug.contains("file_idx"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("shape"));
    }

    // ── CandidateTensor (from mxfp4_pairing) ────────────────────────────

    #[test]
    fn candidate_tensor_construction_and_field_access() {
        let ct = mxfp4_pairing::CandidateTensor {
            name: "layer.weight".to_string(),
            dtype: Dtype::F32,
            shape: vec![64, 128],
            byte_len: 64 * 128 * 4,
        };
        assert_eq!(ct.name, "layer.weight");
        assert_eq!(ct.dtype, Dtype::F32);
        assert_eq!(ct.shape, vec![64, 128]);
        assert_eq!(ct.byte_len, 32768);
    }

    #[test]
    fn candidate_tensor_clone_is_equal() {
        let a = mxfp4_pairing::CandidateTensor {
            name: "test".to_string(),
            dtype: Dtype::BF16,
            shape: vec![10],
            byte_len: 20,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn candidate_tensor_equality_different_name() {
        let a = mxfp4_pairing::CandidateTensor {
            name: "alpha".to_string(),
            dtype: Dtype::F32,
            shape: vec![3],
            byte_len: 12,
        };
        let b = mxfp4_pairing::CandidateTensor {
            name: "beta".to_string(),
            dtype: Dtype::F32,
            shape: vec![3],
            byte_len: 12,
        };
        assert_ne!(a, b);
    }

    // ── Mxfp4Pair construction & field access ──────────────────────────

    #[test]
    fn mxfp4_pair_construction_with_all_fields() {
        let pair = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "expert_blocks".to_string(),
            scales_name: "expert_scales".to_string(),
            block_size: 32,
            num_blocks: 64,
            blocks_shape: vec![2, 32, 16],
            bias_name: Some("expert_bias".to_string()),
        };
        assert_eq!(pair.blocks_name, "expert_blocks");
        assert_eq!(pair.scales_name, "expert_scales");
        assert_eq!(pair.block_size, 32);
        assert_eq!(pair.num_blocks, 64);
        assert_eq!(pair.blocks_shape, vec![2, 32, 16]);
        assert_eq!(pair.bias_name.as_deref(), Some("expert_bias"));
    }

    #[test]
    fn mxfp4_pair_without_bias() {
        let pair = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 10,
            blocks_shape: vec![10, 16],
            bias_name: None,
        };
        assert!(pair.bias_name.is_none());
    }

    #[test]
    fn mxfp4_pair_clone_is_equal() {
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "b".to_string(),
            scales_name: "s".to_string(),
            block_size: 32,
            num_blocks: 5,
            blocks_shape: vec![5, 16],
            bias_name: None,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn mxfp4_pair_debug_contains_fields() {
        let pair = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "my_blocks".to_string(),
            scales_name: "my_scales".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: Some("my_bias".to_string()),
        };
        let debug = format!("{:?}", pair);
        assert!(debug.contains("blocks_name"));
        assert!(debug.contains("scales_name"));
        assert!(debug.contains("block_size"));
    }

    // ── Mxfp4PairScan ──────────────────────────────────────────────────

    #[test]
    fn mxfp4_pair_scan_default_is_empty() {
        let scan = mxfp4_pairing::Mxfp4PairScan::default();
        assert!(scan.pairs.is_empty());
        assert!(scan.blocks_to_scales.is_empty());
        assert!(scan.sidecars.is_empty());
    }

    #[test]
    fn mxfp4_pair_scan_equality() {
        let a = mxfp4_pairing::Mxfp4PairScan::default();
        let b = mxfp4_pairing::Mxfp4PairScan::default();
        assert_eq!(a, b);
    }

    #[test]
    fn mxfp4_pair_scan_manual_construction() {
        let mut scan = mxfp4_pairing::Mxfp4PairScan::default();
        scan.blocks_to_scales.insert("blocks".to_string(), "scales".to_string());
        scan.sidecars.insert("scales".to_string());
        assert_eq!(scan.blocks_to_scales.len(), 1);
        assert_eq!(scan.sidecars.len(), 1);
        assert_eq!(scan.blocks_to_scales.get("blocks"), Some(&"scales".to_string()));
    }

    // ── Mxfp4 public constants ─────────────────────────────────────────

    #[test]
    fn mxfp4_default_block_size_is_32() {
        assert_eq!(mxfp4_pairing::DEFAULT_MXFP4_BLOCK_SIZE, 32);
    }

    #[test]
    fn mxfp4_suffix_constants() {
        assert_eq!(mxfp4_pairing::MXFP4_BLOCKS_SUFFIX, "_blocks");
        assert_eq!(mxfp4_pairing::MXFP4_SCALES_SUFFIX, "_scales");
        assert_eq!(mxfp4_pairing::MXFP4_BIAS_SUFFIX, "_bias");
    }

    // ── AwqGptqFormat ──────────────────────────────────────────────────

    #[test]
    fn awq_gptq_format_awq_variant() {
        let fmt = awq_gptq_pairing::AwqGptqFormat::Awq;
        assert_eq!(fmt, awq_gptq_pairing::AwqGptqFormat::Awq);
        assert_ne!(fmt, awq_gptq_pairing::AwqGptqFormat::Gptq);
    }

    #[test]
    fn awq_gptq_format_gptq_variant() {
        let fmt = awq_gptq_pairing::AwqGptqFormat::Gptq;
        assert_eq!(fmt, awq_gptq_pairing::AwqGptqFormat::Gptq);
    }

    #[test]
    fn awq_gptq_format_is_copy() {
        let a = awq_gptq_pairing::AwqGptqFormat::Awq;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn awq_gptq_format_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(awq_gptq_pairing::AwqGptqFormat::Awq);
        set.insert(awq_gptq_pairing::AwqGptqFormat::Gptq);
        set.insert(awq_gptq_pairing::AwqGptqFormat::Awq);
        assert_eq!(set.len(), 2);
    }

    // ── AwqGptqGroup construction & field access ───────────────────────

    #[test]
    fn awq_gptq_group_construction_awq_format() {
        let group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer.gate".to_string(),
            qweight_name: "layer.gate.qweight".to_string(),
            scales_name: "layer.gate.scales".to_string(),
            qzeros_name: "layer.gate.qzeros".to_string(),
            g_idx_name: None,
            format: awq_gptq_pairing::AwqGptqFormat::Awq,
            qweight_shape: vec![16, 64],
        };
        assert_eq!(group.base_name, "layer.gate");
        assert_eq!(group.format, awq_gptq_pairing::AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());
        assert_eq!(group.qweight_shape, vec![16, 64]);
    }

    #[test]
    fn awq_gptq_group_construction_gptq_format() {
        let group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer.up".to_string(),
            qweight_name: "layer.up.qweight".to_string(),
            scales_name: "layer.up.scales".to_string(),
            qzeros_name: "layer.up.qzeros".to_string(),
            g_idx_name: Some("layer.up.g_idx".to_string()),
            format: awq_gptq_pairing::AwqGptqFormat::Gptq,
            qweight_shape: vec![32, 128],
        };
        assert_eq!(group.format, awq_gptq_pairing::AwqGptqFormat::Gptq);
        assert_eq!(group.g_idx_name.as_deref(), Some("layer.up.g_idx"));
    }

    // ── AwqGptqScan ────────────────────────────────────────────────────

    #[test]
    fn awq_gptq_scan_default_is_empty() {
        let scan = awq_gptq_pairing::AwqGptqScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn awq_gptq_group_debug_contains_fields() {
        let group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer.gate".to_string(),
            qweight_name: "layer.gate.qweight".to_string(),
            scales_name: "layer.gate.scales".to_string(),
            qzeros_name: "layer.gate.qzeros".to_string(),
            g_idx_name: Some("layer.gate.g_idx".to_string()),
            format: awq_gptq_pairing::AwqGptqFormat::Gptq,
            qweight_shape: vec![16, 64],
        };
        let debug = format!("{:?}", group);
        assert!(debug.contains("base_name"));
        assert!(debug.contains("qweight_name"));
        assert!(debug.contains("format"));
    }

    #[test]
    fn awq_gptq_group_clone_preserves_all_fields() {
        let group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer.up".to_string(),
            qweight_name: "layer.up.qweight".to_string(),
            scales_name: "layer.up.scales".to_string(),
            qzeros_name: "layer.up.qzeros".to_string(),
            g_idx_name: None,
            format: awq_gptq_pairing::AwqGptqFormat::Awq,
            qweight_shape: vec![8, 32],
        };
        let cloned = group.clone();
        assert_eq!(cloned.base_name, "layer.up");
        assert_eq!(cloned.qweight_name, "layer.up.qweight");
        assert_eq!(cloned.scales_name, "layer.up.scales");
        assert_eq!(cloned.qzeros_name, "layer.up.qzeros");
        assert!(cloned.g_idx_name.is_none());
        assert_eq!(cloned.format, awq_gptq_pairing::AwqGptqFormat::Awq);
        assert_eq!(cloned.qweight_shape, vec![8, 32]);
    }

    // ── NvfpCandidate construction & field access ──────────────────────

    #[test]
    fn nvfp_candidate_construction_and_field_access() {
        let c = nvfp4_pairing::NvfpCandidate {
            name: "model.layer.weight".to_string(),
            dtype: Dtype::U8,
            shape: vec![64, 128],
            byte_len: 64 * 128,
        };
        assert_eq!(c.name, "model.layer.weight");
        assert_eq!(c.dtype, Dtype::U8);
        assert_eq!(c.shape, vec![64, 128]);
        assert_eq!(c.byte_len, 8192);
    }

    #[test]
    fn nvfp_candidate_clone_is_equal() {
        let a = nvfp4_pairing::NvfpCandidate {
            name: "w".to_string(),
            dtype: Dtype::U8,
            shape: vec![10],
            byte_len: 10,
        };
        let b = a.clone();
        assert_eq!(a.name, b.name);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.byte_len, b.byte_len);
    }

    #[test]
    fn nvfp_candidate_debug_contains_fields() {
        let c = nvfp4_pairing::NvfpCandidate {
            name: "test.weight".to_string(),
            dtype: Dtype::U8,
            shape: vec![2, 3],
            byte_len: 6,
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("name"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("byte_len"));
    }

    // ── NvfpGroup construction & field access ──────────────────────────

    #[test]
    fn nvfp_group_construction_without_global_scale() {
        let g = nvfp4_pairing::NvfpGroup {
            base_name: "model.layer".to_string(),
            weight_name: "model.layer.weight".to_string(),
            scale_name: "model.layer.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![64, 128],
        };
        assert_eq!(g.base_name, "model.layer");
        assert_eq!(g.weight_name, "model.layer.weight");
        assert_eq!(g.scale_name, "model.layer.weight_scale");
        assert!(g.global_scale_name.is_none());
        assert_eq!(g.weight_shape, vec![64, 128]);
    }

    #[test]
    fn nvfp_group_construction_with_global_scale() {
        let g = nvfp4_pairing::NvfpGroup {
            base_name: "model.layer".to_string(),
            weight_name: "model.layer.weight".to_string(),
            scale_name: "model.layer.weight_scale".to_string(),
            global_scale_name: Some("model.layer.weight_scale_2".to_string()),
            weight_shape: vec![32, 64],
        };
        assert_eq!(g.global_scale_name.as_deref(), Some("model.layer.weight_scale_2"));
    }

    #[test]
    fn nvfp_group_equality_same_fields() {
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.weight".to_string(),
            scale_name: "x.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![4, 8],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.weight".to_string(),
            scale_name: "x.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![4, 8],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn nvfp_group_inequality_different_base_name() {
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "a".to_string(),
            weight_name: "a.weight".to_string(),
            scale_name: "a.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "b".to_string(),
            weight_name: "b.weight".to_string(),
            scale_name: "b.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn nvfp_group_clone_is_equal() {
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "base".to_string(),
            weight_name: "base.weight".to_string(),
            scale_name: "base.weight_scale".to_string(),
            global_scale_name: Some("base.weight_scale_2".to_string()),
            weight_shape: vec![16, 32],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn nvfp_group_hash_in_set() {
        use std::collections::HashSet;
        let g = nvfp4_pairing::NvfpGroup {
            base_name: "layer".to_string(),
            weight_name: "layer.weight".to_string(),
            scale_name: "layer.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![8],
        };
        let mut set = HashSet::new();
        set.insert(g.clone());
        set.insert(g.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn nvfp_group_debug_contains_fields() {
        let g = nvfp4_pairing::NvfpGroup {
            base_name: "blk".to_string(),
            weight_name: "blk.weight".to_string(),
            scale_name: "blk.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![2, 4],
        };
        let debug = format!("{:?}", g);
        assert!(debug.contains("base_name"));
        assert!(debug.contains("weight_name"));
        assert!(debug.contains("scale_name"));
        assert!(debug.contains("weight_shape"));
    }

    // ── NvfpScan default & manual construction ─────────────────────────

    #[test]
    fn nvfp_scan_default_is_empty() {
        let scan = nvfp4_pairing::NvfpScan::default();
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn nvfp_scan_manual_construction() {
        let mut scan = nvfp4_pairing::NvfpScan::default();
        let group = nvfp4_pairing::NvfpGroup {
            base_name: "layer".to_string(),
            weight_name: "layer.weight".to_string(),
            scale_name: "layer.weight_scale".to_string(),
            global_scale_name: None,
            weight_shape: vec![4, 8],
        };
        scan.groups.insert("layer".to_string(), group);
        scan.consumed.insert("layer.weight".to_string());
        scan.consumed.insert("layer.weight_scale".to_string());
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 2);
    }

    // ── Mxfp4PairMap & Mxfp4ScalesSidecarSet type aliases ─────────────

    #[test]
    fn mxfp4_pair_map_insert_and_lookup() {
        let mut map = mxfp4_pairing::Mxfp4PairMap::new();
        let pair = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        map.insert("blk".to_string(), pair);
        assert!(map.contains_key("blk"));
        assert_eq!(map["blk"].block_size, 32);
    }

    #[test]
    fn mxfp4_scales_sidecar_set_insert_and_check() {
        let mut set = mxfp4_pairing::Mxfp4ScalesSidecarSet::new();
        set.insert("expert_scales".to_string());
        assert!(set.contains("expert_scales"));
        assert!(!set.contains("expert_blocks"));
    }

    // ── LoaderError — uncovered variants ──────────────────────────────

    #[test]
    fn loader_error_missing_weights_variant() {
        let err = LoaderError::MissingWeights;
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn loader_error_authentication_error_variant() {
        let err = LoaderError::AuthenticationError { hint: "token expired".to_string() };
        let msg = err.to_string();
        assert!(msg.contains("token expired"));
    }

    #[test]
    fn loader_error_format_not_found_variant() {
        use crate::loader::WeightFormat;
        let err = LoaderError::FormatNotFound(WeightFormat::SafeTensors);
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn loader_error_multiple_weight_formats_variant() {
        use crate::loader::WeightFormat;
        let err = LoaderError::MultipleWeightFormats(vec![WeightFormat::SafeTensors, WeightFormat::Gguf]);
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn loader_error_unsupported_dtype_variant() {
        let err = LoaderError::UnsupportedDtype(Dtype::F32);
        let msg = err.to_string();
        assert!(msg.contains("F32"));
    }

    // ── WeightFormat enum variant existence ───────────────────────────

    #[test]
    fn weight_format_variants_are_distinct() {
        use crate::loader::WeightFormat;
        let variants = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "WeightFormat variants must be distinct");
            }
        }
    }

    // ── TensorSlice field accessors via construction ──────────────────

    #[test]
    fn tensor_slice_fields_match_construction() {
        let data: &[u8] = &[1, 2, 3, 4];
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![1], data };
        assert_eq!(ts.dtype, Dtype::I32);
        assert_eq!(ts.shape, vec![1]);
        assert_eq!(ts.data.len(), 4);
    }

    #[test]
    fn tensor_slice_element_count_consistent_with_shape() {
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![4, 5, 6], data: &[] };
        assert_eq!(ts.element_count(), 4 * 5 * 6);
    }

    // ── Mxfp4Pair inequality ──────────────────────────────────────────

    #[test]
    fn mxfp4_pair_inequality_different_blocks_name() {
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "a_blocks".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "b_blocks".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        assert_ne!(a, b);
    }

    // ── NvfpGroup inequality for remaining fields ─────────────────────

    #[test]
    fn nvfp_group_inequality_different_weight_name() {
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w1".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w2".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        assert_ne!(a, b);
    }

    // ── AwqGptqScan manual construction with data ────────────────────

    #[test]
    fn awq_gptq_scan_manual_insertion() {
        let mut scan = awq_gptq_pairing::AwqGptqScan::default();
        let group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer.gate".to_string(),
            qweight_name: "layer.gate.qweight".to_string(),
            scales_name: "layer.gate.scales".to_string(),
            qzeros_name: "layer.gate.qzeros".to_string(),
            g_idx_name: None,
            format: awq_gptq_pairing::AwqGptqFormat::Awq,
            qweight_shape: vec![16, 64],
        };
        scan.groups.insert("layer.gate".to_string(), group);
        scan.consumed.insert("layer.gate.qweight".to_string());
        scan.consumed.insert("layer.gate.scales".to_string());
        scan.consumed.insert("layer.gate.qzeros".to_string());
        assert_eq!(scan.groups.len(), 1);
        assert_eq!(scan.consumed.len(), 3);
    }

    // ── Mxfp4PairMap empty lookup ─────────────────────────────────────

    #[test]
    fn mxfp4_pair_map_empty_lookup_returns_none() {
        let map = mxfp4_pairing::Mxfp4PairMap::new();
        assert!(map.get("nonexistent").is_none());
        assert!(map.is_empty());
    }

    // ── byte_len_of with large shape ──────────────────────────────────

    #[test]
    fn byte_len_f32_large_2d() {
        assert_eq!(byte_len_of(Dtype::F32, &[1024, 1024]), 1024 * 1024 * 4);
    }

    #[test]
    fn byte_len_exotic_dtype_falls_back() {
        // Exotic dtypes not in the match arms fall back to elem_size=1
        assert_eq!(byte_len_of(Dtype::F64, &[0, 5]), 0);
    }

    // ── Additional unit tests ──────────────────────────────────────────

    #[test]
    fn tensor_location_empty_shape_is_equal() {
        // Arrange: two locations with empty shapes (scalar tensors)
        let a = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![] };
        let b = TensorLocation { file_idx: 0, dtype: Dtype::F32, shape: vec![] };
        // Assert: equal even with empty shape
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_location_shape_field_is_vec_usize() {
        // Arrange: construct with specific shape values
        let loc = TensorLocation { file_idx: 5, dtype: Dtype::U8, shape: vec![1, 2, 3, 4] };
        // Assert: shape retains exact values
        assert_eq!(loc.shape.len(), 4);
        assert_eq!(loc.shape[2], 3);
    }

    #[test]
    fn tensor_slice_debug_format_contains_dtype_and_shape() {
        // Arrange: a TensorSlice with known fields
        let data: &[u8] = &[0xAB, 0xCD, 0xEF, 0x01];
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data };
        // Act
        let debug = format!("{:?}", ts);
        // Assert: Debug output includes the field names
        assert!(debug.contains("dtype"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("data"));
    }

    #[test]
    fn tensor_slice_element_count_single_element() {
        // Arrange: shape is [1, 1, 1]
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1, 1, 1], data: &[0; 4] };
        // Assert: product is 1
        assert_eq!(ts.element_count(), 1);
    }

    #[test]
    fn byte_len_of_u8_3d_shape() {
        // Arrange: 3D shape with U8
        // Act
        let size = byte_len_of(Dtype::U8, &[7, 8, 9]);
        // Assert: 7*8*9 * 1 = 504
        assert_eq!(size, 504);
    }

    #[test]
    fn byte_len_of_bf16_1d_shape() {
        // Arrange: 1D shape with BF16 (2 bytes per element)
        // Act
        let size = byte_len_of(Dtype::BF16, &[100]);
        // Assert: 100 * 2 = 200
        assert_eq!(size, 200);
    }

    #[test]
    fn weight_format_clone_preserves_variant() {
        // Arrange
        use crate::loader::WeightFormat;
        let original = WeightFormat::Gguf;
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(original, cloned);
    }

    #[test]
    fn weight_format_debug_contains_variant_name() {
        // Arrange
        use crate::loader::WeightFormat;
        // Act
        let debug = format!("{:?}", WeightFormat::SafeTensors);
        // Assert: Debug output includes the variant name
        assert!(debug.contains("SafeTensors"));
    }

    #[test]
    fn weight_format_hash_in_set() {
        // Arrange: insert same variant twice
        use std::collections::HashSet;
        use crate::loader::WeightFormat;
        let mut set = HashSet::new();
        set.insert(WeightFormat::Gguf);
        set.insert(WeightFormat::Gguf);
        // Assert: deduplication works
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn awq_gptq_scan_default_is_empty_both_fields() {
        // Arrange: a default instance
        let scan = awq_gptq_pairing::AwqGptqScan::default();
        // Assert: both collections start empty
        assert!(scan.groups.is_empty());
        assert!(scan.consumed.is_empty());
    }

    #[test]
    fn nvfp_group_inequality_different_scale_name() {
        // Arrange: two groups differing only in scale_name
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s1".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s2".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn nvfp_group_inequality_different_global_scale_name() {
        // Arrange: one with None, one with Some for global_scale_name
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: None,
            weight_shape: vec![4],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: Some("x.gs".to_string()),
            weight_shape: vec![4],
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn nvfp_group_inequality_different_weight_shape() {
        // Arrange: two groups differing only in weight_shape
        let a = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: None,
            weight_shape: vec![4, 8],
        };
        let b = nvfp4_pairing::NvfpGroup {
            base_name: "x".to_string(),
            weight_name: "x.w".to_string(),
            scale_name: "x.s".to_string(),
            global_scale_name: None,
            weight_shape: vec![4, 16],
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn mxfp4_pair_inequality_different_scales_name() {
        // Arrange: two pairs differing only in scales_name
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc_a".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc_b".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn mxfp4_pair_inequality_different_block_size() {
        // Arrange: two pairs differing only in block_size
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 64,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        // Assert
        assert_ne!(a, b);
    }

    // ── 15 new tests: edge cases, error handling, derive traits, boundaries ──

    #[test]
    fn mxfp4_pair_inequality_different_num_blocks() {
        // Arrange: two pairs differing only in num_blocks
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 16,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn mxfp4_pair_inequality_different_bias_name() {
        // Arrange: one with None, one with Some for bias_name
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: Some("blk_bias".to_string()),
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn mxfp4_pair_inequality_different_blocks_shape() {
        // Arrange: two pairs differing only in blocks_shape
        let a = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![8, 16],
            bias_name: None,
        };
        let b = mxfp4_pairing::Mxfp4Pair {
            blocks_name: "blk".to_string(),
            scales_name: "sc".to_string(),
            block_size: 32,
            num_blocks: 8,
            blocks_shape: vec![4, 32],
            bias_name: None,
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_tensor_inequality_different_dtype() {
        // Arrange: same name/shape/byte_len but different dtype
        let a = mxfp4_pairing::CandidateTensor {
            name: "w".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            byte_len: 16,
        };
        let b = mxfp4_pairing::CandidateTensor {
            name: "w".to_string(),
            dtype: Dtype::F16,
            shape: vec![4],
            byte_len: 16,
        };
        // Assert: dtype differs
        assert_ne!(a, b);
    }

    #[test]
    fn candidate_tensor_inequality_different_byte_len() {
        // Arrange: same name/dtype/shape but different byte_len
        let a = mxfp4_pairing::CandidateTensor {
            name: "w".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            byte_len: 16,
        };
        let b = mxfp4_pairing::CandidateTensor {
            name: "w".to_string(),
            dtype: Dtype::F32,
            shape: vec![4],
            byte_len: 32,
        };
        // Assert: byte_len differs
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_slice_as_f32_single_element_roundtrip() {
        // Arrange: a single f32 value
        let vals = vec![-3.14f32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: bytes };
        // Act
        let result = ts.as_f32().unwrap();
        // Assert: roundtrip preserves value
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], -3.14f32);
    }

    #[test]
    fn tensor_slice_as_i32_max_and_min_values() {
        // Arrange: i32 extremal values
        let vals = vec![i32::MIN, i32::MAX, 0i32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![3], data: bytes };
        // Act
        let result = ts.as_i32().unwrap();
        // Assert: exact roundtrip for extremal values
        assert_eq!(result[0], i32::MIN);
        assert_eq!(result[1], i32::MAX);
        assert_eq!(result[2], 0i32);
    }

    #[test]
    fn byte_len_of_f32_zero_elements() {
        // Arrange: shape [0] means zero elements
        // Act
        let size = byte_len_of(Dtype::F32, &[0]);
        // Assert: 0 elements * 4 bytes = 0
        assert_eq!(size, 0);
    }

    #[test]
    fn byte_len_of_f64_2d_shape() {
        // Arrange: 2D shape with F64 (8 bytes per element)
        // Act
        let size = byte_len_of(Dtype::F64, &[3, 4]);
        // Assert: 3 * 4 * 8 = 96
        assert_eq!(size, 96);
    }

    #[test]
    fn byte_len_of_i64_2d_shape() {
        // Arrange: 2D shape with I64 (8 bytes per element)
        // Act
        let size = byte_len_of(Dtype::I64, &[2, 5]);
        // Assert: 2 * 5 * 8 = 80
        assert_eq!(size, 80);
    }

    #[test]
    fn tensor_slice_as_f64_misaligned_returns_owned() {
        // Arrange: f64 data with 1-byte offset to force misalignment
        let vals = vec![std::f64::consts::PI, std::f64::consts::E];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xAA];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::F64, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_f64().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], std::f64::consts::PI);
        assert_eq!(result[1], std::f64::consts::E);
    }

    #[test]
    fn tensor_slice_as_i16_misaligned_returns_owned() {
        // Arrange: i16 data with 1-byte offset to force misalignment
        let vals = vec![100i16, -100i16];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::I16, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_i16().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 100i16);
        assert_eq!(result[1], -100i16);
    }

    #[test]
    fn tensor_slice_as_u32_misaligned_returns_owned() {
        // Arrange: u32 data with 1-byte offset to force misalignment
        let vals = vec![0xDEADBEEFu32, 12345u32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0x00];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::U32, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_u32().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 0xDEADBEEFu32);
        assert_eq!(result[1], 12345u32);
    }

    #[test]
    fn tensor_slice_as_u64_misaligned_returns_owned() {
        // Arrange: u64 data with 1-byte offset to force misalignment
        let vals = vec![u64::MAX, 0u64];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::U64, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_u64().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], u64::MAX);
        assert_eq!(result[1], 0u64);
    }

    #[test]
    fn awq_gptq_group_format_field_distinguishes_awq_vs_gptq() {
        // Arrange: two groups differing only in format (Awq vs Gptq)
        let awq_group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer".to_string(),
            qweight_name: "layer.qweight".to_string(),
            scales_name: "layer.scales".to_string(),
            qzeros_name: "layer.qzeros".to_string(),
            g_idx_name: None,
            format: awq_gptq_pairing::AwqGptqFormat::Awq,
            qweight_shape: vec![16, 64],
        };
        let gptq_group = awq_gptq_pairing::AwqGptqGroup {
            base_name: "layer".to_string(),
            qweight_name: "layer.qweight".to_string(),
            scales_name: "layer.scales".to_string(),
            qzeros_name: "layer.qzeros".to_string(),
            g_idx_name: None,
            format: awq_gptq_pairing::AwqGptqFormat::Gptq,
            qweight_shape: vec![16, 64],
        };
        // Assert: format field distinguishes the two groups
        assert_ne!(awq_group.format, gptq_group.format);
        assert_eq!(awq_group.base_name, gptq_group.base_name);
        assert_eq!(awq_group.qweight_shape, gptq_group.qweight_shape);
    }

    // ── 15 additional new tests ──────────────────────────────────────────

    #[test]
    fn as_f16_misaligned_returns_owned_with_correct_values() {
        // Arrange: f16 data with 1-byte offset to force misalignment
        let vals = vec![f16::from_f32(0.0), f16::from_f32(-7.5)];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xCD];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::F16, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_f16().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], f16::from_f32(0.0));
        assert_eq!(result[1], f16::from_f32(-7.5));
    }

    #[test]
    fn as_u16_misaligned_returns_owned_with_correct_values() {
        // Arrange: u16 data with 1-byte offset to force misalignment
        let vals = vec![4660u16, 65534u16];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xFF];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::U16, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_u16().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 4660u16);
        assert_eq!(result[1], 65534u16);
    }

    #[test]
    fn as_i64_misaligned_returns_owned_with_correct_values() {
        // Arrange: i64 data with 1-byte offset to force misalignment
        let vals = vec![i64::MIN + 1, i64::MAX - 1];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xAA];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::I64, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_i64().unwrap();
        // Assert: Owned variant with correct extremal values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], i64::MIN + 1);
        assert_eq!(result[1], i64::MAX - 1);
    }

    #[test]
    fn as_u8_correct_dtype_returns_borrowed_with_exact_content() {
        // Arrange: specific byte values that differ from simple 0x00/0xFF
        let raw: &[u8] = &[0x0A, 0x5B, 0xF0, 0x7C];
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![4], data: raw };
        // Act
        let result = ts.as_u8().unwrap();
        // Assert: borrowed Cow preserving exact bytes
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, &[0x0A, 0x5B, 0xF0, 0x7C]);
    }

    #[test]
    fn as_i8_correct_dtype_negative_values_roundtrip() {
        // Arrange: i8 data with negative values
        let raw: &[u8] = &[0x80, 0xFF, 0x01, 0x7F]; // -128, -1, 1, 127
        let ts = TensorSlice { dtype: Dtype::I8, shape: vec![4], data: raw };
        // Act
        let result = ts.as_i8().unwrap();
        // Assert: exact signed roundtrip
        assert_eq!(result[0], -128i8);
        assert_eq!(result[1], -1i8);
        assert_eq!(result[2], 1i8);
        assert_eq!(result[3], 127i8);
    }

    #[test]
    fn byte_len_of_zero_in_middle_dimension() {
        // Arrange: shape [3, 0, 5] — any zero dimension means zero elements
        // Act
        let size = byte_len_of(Dtype::F32, &[3, 0, 5]);
        // Assert: 3 * 0 * 5 * 4 = 0
        assert_eq!(size, 0);
    }

    #[test]
    fn byte_len_of_i8_2d_shape() {
        // Arrange: 2D shape with I8 (1 byte per element)
        // Act
        let size = byte_len_of(Dtype::I8, &[11, 13]);
        // Assert: 11 * 13 * 1 = 143
        assert_eq!(size, 143);
    }

    #[test]
    fn tensor_location_with_large_file_idx() {
        // Arrange: file_idx larger than typical (multi-shard models can have many files)
        let loc = TensorLocation { file_idx: 999, dtype: Dtype::BF16, shape: vec![4096, 4096] };
        // Assert: fields preserved exactly
        assert_eq!(loc.file_idx, 999);
        assert_eq!(loc.dtype, Dtype::BF16);
        assert_eq!(loc.shape, vec![4096, 4096]);
    }

    #[test]
    fn tensor_slice_u8_data_pointer_equality() {
        // Arrange: construct TensorSlice with a known byte slice
        let raw: &[u8] = &[42, 84, 126, 168];
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![4], data: raw };
        // Act: as_u8 returns a Borrowed variant
        let result = ts.as_u8().unwrap();
        // Assert: the Borrowed pointer refers to the same underlying memory
        if let Cow::Borrowed(slice) = result {
            assert!(std::ptr::eq(slice.as_ptr(), raw.as_ptr()));
        } else {
            panic!("expected Cow::Borrowed for aligned u8 data");
        }
    }

    #[test]
    fn loader_error_io_variant_display() {
        // Arrange: construct an IO error from a concrete io::ErrorKind
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = LoaderError::Io(io_err);
        // Act
        let msg = err.to_string();
        // Assert: Display contains the underlying message
        assert!(msg.contains("file missing"), "IO variant message must contain the io error text");
    }

    #[test]
    fn loader_error_json_variant_display() {
        // Arrange: construct a JSON error from an invalid JSON string
        let json_err = serde_json::from_str::<serde_json::Value>("{invalid").unwrap_err();
        let err = LoaderError::Json(json_err);
        // Act
        let msg = err.to_string();
        // Assert: Display contains context about JSON
        assert!(!msg.is_empty(), "JSON variant message must not be empty");
    }

    #[test]
    fn tensor_slice_shape_product_matches_element_count_for_4d() {
        // Arrange: 4D tensor shape
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![2, 3, 4, 5], data: &[] };
        // Assert: element_count = 2*3*4*5 = 120
        assert_eq!(ts.element_count(), 120);
    }

    #[test]
    fn tensor_slice_same_data_different_dtype_accessors_error() {
        // Arrange: the same 4-byte slice interpreted as F32 succeeds but as I32 fails
        let raw: &[u8] = &[0x00, 0x00, 0x80, 0x3F]; // 1.0f32 in LE
        let ts_f32 = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: raw };
        let ts_i32 = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: raw };
        // Act & Assert: correct dtype yields success, wrong dtype yields error
        assert!(ts_f32.as_f32().is_ok());
        let err = ts_i32.as_i32().unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedDtype(Dtype::F32)));
    }

    #[test]
    fn parse_namespace_metadata_empty_files_returns_ok_none() {
        // Arrange: empty files slice, no metadata keys to find
        let files: Vec<MappedSafetensors> = vec![];
        // Act
        let result = parse_namespace_metadata(&files, &["gllm.config"]);
        // Assert: Ok(None) — no error, no metadata found
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn cast_or_copy_f16_misaligned_single_element() {
        // Arrange: a single f16 value misaligned by 1 byte
        let vals = vec![f16::from_f32(42.0)];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0x00];
        buf.extend_from_slice(aligned);
        // Act
        let result = cast_or_copy_f16(&buf[1..]).unwrap();
        // Assert: Owned with correct value
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], f16::from_f32(42.0));
        assert_eq!(result.len(), 1);
    }

    // ── 13 additional edge case tests ─────────────────────────────────────

    #[test]
    fn loader_error_from_io_error() {
        // Arrange: create an io::Error and convert via From
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        // Act: From trait converts io::Error → LoaderError::Io
        let loader_err: LoaderError = io_err.into();
        // Assert: Display contains the original message
        let msg = loader_err.to_string();
        assert!(msg.contains("access denied"));
    }

    #[test]
    fn loader_error_from_json_error() {
        // Arrange: create a JSON parse error
        let json_err = serde_json::from_str::<serde_json::Value>("not json at all").unwrap_err();
        // Act: From trait converts serde_json::Error → LoaderError::Json
        let loader_err: LoaderError = json_err.into();
        // Assert: it is the Json variant
        assert!(matches!(loader_err, LoaderError::Json(_)));
    }

    #[test]
    fn tensor_location_partial_eq_with_large_shape() {
        // Arrange: two locations with large multidimensional shapes
        let shape = vec![128, 256, 512, 1024];
        let a = TensorLocation { file_idx: 0, dtype: Dtype::BF16, shape: shape.clone() };
        let b = TensorLocation { file_idx: 0, dtype: Dtype::BF16, shape: shape.clone() };
        // Assert: equality holds for large shapes
        assert_eq!(a, b);
        assert_eq!(a.shape.len(), 4);
    }

    #[test]
    fn tensor_slice_as_f32_aligned_returns_borrowed_cow() {
        // Arrange: properly aligned f32 data
        let vals = vec![42.0f32];
        let bytes = unsafe { to_bytes(&vals) };
        let ts = TensorSlice { dtype: Dtype::F32, shape: vec![1], data: bytes };
        // Act
        let result = ts.as_f32().unwrap();
        // Assert: aligned data returns Borrowed, not Owned
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn tensor_slice_as_bf16_misaligned_returns_owned() {
        // Arrange: bf16 data misaligned by 1 byte
        let vals = vec![bf16::from_f32(99.5)];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0x00];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::BF16, shape: vec![1], data: &buf[1..] };
        // Act
        let result = ts.as_bf16().unwrap();
        // Assert: misaligned data returns Owned with correct value
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], bf16::from_f32(99.5));
    }

    #[test]
    fn byte_len_of_u32_zero_in_first_dimension() {
        // Arrange: shape with zero in the first dimension
        // Act
        let size = byte_len_of(Dtype::U32, &[0, 100]);
        // Assert: 0 * 100 * 4 = 0
        assert_eq!(size, 0);
    }

    #[test]
    fn byte_len_of_i16_2d_shape() {
        // Arrange: 2D shape with I16 (2 bytes per element)
        // Act
        let size = byte_len_of(Dtype::I16, &[5, 6]);
        // Assert: 5 * 6 * 2 = 60
        assert_eq!(size, 60);
    }

    #[test]
    fn tensor_slice_element_count_high_dimensional() {
        // Arrange: 5D tensor shape
        let ts = TensorSlice { dtype: Dtype::U8, shape: vec![2, 3, 4, 5, 6], data: &[] };
        // Assert: 2*3*4*5*6 = 720
        assert_eq!(ts.element_count(), 720);
    }

    #[test]
    fn candidate_tensor_debug_format_contains_all_fields() {
        // Arrange
        let ct = mxfp4_pairing::CandidateTensor {
            name: "layer.weight".to_string(),
            dtype: Dtype::F32,
            shape: vec![64],
            byte_len: 256,
        };
        // Act
        let debug = format!("{:?}", ct);
        // Assert: Debug output includes all field names
        assert!(debug.contains("name"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("byte_len"));
    }

    #[test]
    fn mxfp4_scales_sidecar_set_rejects_nonexistent_key() {
        // Arrange: empty sidecar set
        let set = mxfp4_pairing::Mxfp4ScalesSidecarSet::new();
        // Assert: any key lookup returns false
        assert!(!set.contains("anything"));
        assert!(set.is_empty());
    }

    #[test]
    fn awq_gptq_format_debug_output() {
        // Arrange: both format variants
        // Act
        let debug_awq = format!("{:?}", awq_gptq_pairing::AwqGptqFormat::Awq);
        let debug_gptq = format!("{:?}", awq_gptq_pairing::AwqGptqFormat::Gptq);
        // Assert: Debug output contains the variant name
        assert!(debug_awq.contains("Awq"));
        assert!(debug_gptq.contains("Gptq"));
    }

    #[test]
    fn nvfp_candidate_fields_differ_when_shape_differs() {
        // Arrange: two candidates with same name/dtype but different shapes
        let a = nvfp4_pairing::NvfpCandidate {
            name: "w".to_string(),
            dtype: Dtype::U8,
            shape: vec![10, 20],
            byte_len: 200,
        };
        let b = nvfp4_pairing::NvfpCandidate {
            name: "w".to_string(),
            dtype: Dtype::U8,
            shape: vec![10, 40],
            byte_len: 400,
        };
        // Assert: shapes and byte_len differ (NvfpCandidate has no PartialEq)
        assert_ne!(a.shape, b.shape);
        assert_ne!(a.byte_len, b.byte_len);
        assert_eq!(a.name, b.name);
        assert_eq!(a.dtype, b.dtype);
    }

    #[test]
    fn tensor_slice_as_i32_misaligned_returns_owned() {
        // Arrange: i32 data with 1-byte offset
        let vals = vec![7i32, -13i32];
        let aligned = unsafe { to_bytes(&vals) };
        let mut buf = vec![0xEE];
        buf.extend_from_slice(aligned);
        let ts = TensorSlice { dtype: Dtype::I32, shape: vec![2], data: &buf[1..] };
        // Act
        let result = ts.as_i32().unwrap();
        // Assert: Owned variant with correct values
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result[0], 7i32);
        assert_eq!(result[1], -13i32);
    }

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

#[cfg(test)]
mod awq_gptq_integration_tests {
    //! End-to-end tests for AWQ/GPTQ triplet detection + loading through
    //! the real `SafeTensorsLoader` + `TensorProvider` surface.

    use super::*;
    use crate::loader::TensorProvider;
    use ::safetensors::tensor::{serialize_to_file, TensorView};
    use ::safetensors::Dtype;
    use tempfile::TempDir;

    /// Build a minimal AWQ-style safetensors file with one qweight+scales+qzeros triplet.
    ///
    /// K=128, N=64, group_size=128:
    ///   qweight: [K/8, N] = [16, 64] I32
    ///   scales:  [K/group_size, N] = [1, 64] F16
    ///   qzeros:  [K/group_size, ceil(N/8)] = [1, 8] I32
    fn write_awq_fixture(dir: &std::path::Path) -> (std::path::PathBuf, Vec<u8>, Vec<u8>, Vec<u8>) {
        let n: usize = 64;
        let packed_rows: usize = 16; // K/8 = 128/8

        let qw_bytes: Vec<u8> = (0..packed_rows * n * 4)
            .map(|i| (i as u8).wrapping_mul(31))
            .collect();
        let scales_bytes: Vec<u8> = (0..n * 2)
            .map(|i| (i as u8).wrapping_mul(17))
            .collect();
        let qz_packed_cols = (n + 7) / 8; // 8
        let qzeros_bytes: Vec<u8> = (0..qz_packed_cols * 4)
            .map(|i| (i as u8).wrapping_mul(13))
            .collect();

        // A regular non-AWQ tensor
        let bias_bytes: Vec<u8> = (0..n * 4)
            .map(|i| (i as u8).wrapping_mul(7))
            .collect();

        let qw_view = TensorView::new(Dtype::I32, vec![packed_rows, n], &qw_bytes).expect("qw view");
        let scales_view = TensorView::new(Dtype::F16, vec![1, n], &scales_bytes).expect("scales view");
        let qzeros_view = TensorView::new(Dtype::I32, vec![1, qz_packed_cols], &qzeros_bytes).expect("qzeros view");
        let bias_view = TensorView::new(Dtype::F32, vec![n], &bias_bytes).expect("bias view");

        let path = dir.join("model.safetensors");
        let metadata: Option<HashMap<String, String>> = None;
        serialize_to_file(
            vec![
                ("model.layers.0.mlp.gate_proj.qweight", qw_view),
                ("model.layers.0.mlp.gate_proj.scales", scales_view),
                ("model.layers.0.mlp.gate_proj.qzeros", qzeros_view),
                ("model.layers.0.mlp.gate_proj.bias", bias_view),
            ],
            &metadata,
            &path,
        )
        .expect("write safetensors");

        (path, qw_bytes, scales_bytes, qzeros_bytes)
    }

    #[test]
    fn awq_triplet_detected_qweight_exposed_under_base_name() {
        let dir = TempDir::new().expect("temp dir");
        let (path, _qw, _scales, _qzeros) = write_awq_fixture(dir.path());
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        // AWQ group detected
        let groups = loader.awq_gptq_groups();
        assert_eq!(groups.len(), 1);
        let group = groups.get("model.layers.0.mlp.gate_proj").expect("group");
        assert_eq!(group.format, awq_gptq_pairing::AwqGptqFormat::Awq);
        assert!(group.g_idx_name.is_none());

        // Consumed tensors include all three
        assert!(loader.is_awq_gptq_consumed("model.layers.0.mlp.gate_proj.qweight"));
        assert!(loader.is_awq_gptq_consumed("model.layers.0.mlp.gate_proj.scales"));
        assert!(loader.is_awq_gptq_consumed("model.layers.0.mlp.gate_proj.qzeros"));

        // qweight exposed under base_name via iter_tensors
        let enumerated: Vec<String> = loader.iter_tensors().map(|m| m.name).collect();
        assert!(enumerated.iter().any(|n| n == "model.layers.0.mlp.gate_proj"), "base_name should be in iter_tensors");
        assert!(!enumerated.iter().any(|n| n == "model.layers.0.mlp.gate_proj.qweight"), "qweight should be hidden");
        assert!(!enumerated.iter().any(|n| n == "model.layers.0.mlp.gate_proj.scales"), "scales should be hidden");
        assert!(!enumerated.iter().any(|n| n == "model.layers.0.mlp.gate_proj.qzeros"), "qzeros should be hidden");

        // Regular tensor not affected
        assert!(enumerated.iter().any(|n| n == "model.layers.0.mlp.gate_proj.bias"));
    }

    #[test]
    fn awq_base_name_reports_synthetic_ggml_dtype() {
        let dir = TempDir::new().expect("temp dir");
        let (path, _qw, _scales, _qzeros) = write_awq_fixture(dir.path());
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        // base_name maps to AWQ4
        assert_eq!(loader.ggml_dtype("model.layers.0.mlp.gate_proj"), Some(GgmlDType::AWQ4));
        // Regular tensor has no ggml_dtype
        assert_eq!(loader.ggml_dtype("model.layers.0.mlp.gate_proj.bias"), None);
    }

    #[test]
    fn awq_load_tensor_data_serves_qweight_under_base_name() {
        let dir = TempDir::new().expect("temp dir");
        let (path, expected_qw, _scales, _qzeros) = write_awq_fixture(dir.path());
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        // load_tensor_data for base_name returns qweight bytes
        let data = loader.load_tensor_data("model.layers.0.mlp.gate_proj").expect("load qweight");
        let data_ref: &[u8] = data.as_ref();
        assert_eq!(data_ref, expected_qw.as_slice());
    }

    #[test]
    fn awq_aux_data_returns_scales_zeros_and_group_size() {
        let dir = TempDir::new().expect("temp dir");
        let (path, _qw, expected_scales, expected_qzeros) = write_awq_fixture(dir.path());
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        let (scales, zeros, g_idx, group_size) = loader
            .awq_gptq_aux_data("model.layers.0.mlp.gate_proj")
            .expect("aux data");

        let scales_ref: &[u8] = scales.as_ref();
        let zeros_ref: &[u8] = zeros.as_ref();
        assert_eq!(scales_ref, expected_scales.as_slice());
        assert_eq!(zeros_ref, expected_qzeros.as_slice());
        assert!(g_idx.is_none(), "AWQ has no g_idx");
        assert_eq!(group_size, 128, "group_size = K/scales_rows = 128/1");
    }

    /// REQ-QCG-010 验收:safetensors AWQ4 三元组 → loader 内部 repack →
    /// 暴露的 element-level shape 为 [N, K] 而非 packed [N, K/8],
    /// 这是 process_single_tensor 推导 (n, k) 与 JIT QuantGemm 对接的前提。
    #[test]
    fn awq_iter_tensors_reports_element_level_shape() {
        let dir = TempDir::new().expect("temp dir");
        let (path, _qw, _scales, _qzeros) = write_awq_fixture(dir.path());
        let loader = SafeTensorsLoader::from_files(
            &[path],
            crate::loader::ParallelLoader::new(false),
        )
        .expect("load");

        let base = "model.layers.0.mlp.gate_proj";
        // Fixture: K=128, N=64, packed_rows=K/8=16, n=64
        // iter_tensors must expose base_name with element-level shape [N, K] = [64, 128]
        let listed = loader.iter_tensors().find(|m| m.name == base).expect("base in iter_tensors");
        assert_eq!(listed.shape, vec![64, 128],
            "AWQ4 base_name shape must be element-level [N, K], got {:?}", listed.shape);

        // tensor_info on qweight_name also rewrites shape to element-level
        let qw_name = format!("{}.qweight", base);
        let info = loader.tensor_info(&qw_name).expect("qweight meta");
        assert_eq!(info.shape, vec![64, 128],
            "qweight tensor_info must return element-level [N, K], got {:?}", info.shape);
        assert_eq!(info.name, base, "qweight info name must be rewritten to base_name");
    }
}

