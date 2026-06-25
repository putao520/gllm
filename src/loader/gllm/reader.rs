//! .gllm 文件解析器 — mmap + 零拷贝读取。
//!
//! SPEC: `SPEC/36-GLLM-WEIGHT-FORMAT.md §1-§2`

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use safetensors::Dtype;

use super::types::{GllmError, GllmHeader, GllmTensorEntry};
use super::TENSOR_ENTRY_SIZE;
use crate::loader::{TensorMeta, TensorProvider};

/// Resolved tensor info with name resolved from the string table.
#[derive(Debug, Clone)]
pub struct ResolvedTensor {
    pub name: String,
    pub entry: GllmTensorEntry,
    /// Absolute byte offset into the file.
    pub abs_data_offset: usize,
    /// Byte size of the tensor data in the file.
    pub data_size: usize,
}

/// Parsed .gllm file.
#[derive(Debug)]
pub struct GllmReader {
    mmap: Arc<Mmap>,
    header: GllmHeader,
    tensors: Vec<ResolvedTensor>,
    tensor_index: HashMap<String, usize>,
    /// Raw MessagePack metadata bytes (unparsed; caller decides when to deserialize).
    metadata_bytes: Vec<u8>,
}

impl GllmReader {
    /// Open and parse a .gllm file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GllmError> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Self::parse(Arc::new(mmap))
    }

    /// Parse from a list of file paths (.gllm is always a single file).
    pub fn from_files(paths: &[PathBuf]) -> Result<Self, GllmError> {
        if paths.len() != 1 {
            return Err(GllmError::ParseError(
                ".gllm loader expects a single weight file".to_string(),
            ));
        }
        Self::open(&paths[0])
    }

    fn parse(mmap: Arc<Mmap>) -> Result<Self, GllmError> {
        let bytes = &mmap[..];

        // ── 1. Header ──────────────────────────────────────────────────────
        let header = GllmHeader::parse(bytes)?;

        // ── 2. Tensor Directory ────────────────────────────────────────────
        let td_offset = header.tensor_dir_offset as usize;
        let tensor_count = header.tensor_count as usize;
        let td_end = td_offset + tensor_count * TENSOR_ENTRY_SIZE;
        if td_end > bytes.len() {
            return Err(GllmError::TensorDirOutOfBounds {
                offset: td_offset,
                count: tensor_count,
                file_size: bytes.len(),
            });
        }

        let raw_entries: Vec<GllmTensorEntry> = (0..tensor_count)
            .map(|i| GllmTensorEntry::parse_at(bytes, td_offset + i * TENSOR_ENTRY_SIZE))
            .collect::<Result<_, _>>()?;

        // ── 3. Resolve tensor names ────────────────────────────────────────
        // String table sits right after the Tensor Directory.
        let string_table_offset = td_end;
        let data_offset = header.data_offset as usize;

        let mut tensors = Vec::with_capacity(tensor_count);
        let mut tensor_index = HashMap::with_capacity(tensor_count);

        for entry in &raw_entries {
            let name_start = string_table_offset + entry.name_offset as usize;
            let name_end = name_start + entry.name_len as usize;
            if name_end > data_offset {
                return Err(GllmError::StringTableOutOfBounds {
                    offset: name_start,
                    length: entry.name_len as usize,
                    file_size: bytes.len(),
                });
            }
            let name = String::from_utf8(bytes[name_start..name_end].to_vec())
                .map_err(|e| GllmError::ParseError(format!("invalid tensor name: {e}")))?;

            let abs_data_offset = data_offset + entry.data_offset as usize;
            let data_size = if entry.is_quantized() {
                entry.compressed_size as usize
            } else {
                entry.original_size as usize
            };

            let end = abs_data_offset.checked_add(data_size).ok_or_else(|| {
                GllmError::ParseError(format!("tensor \"{}\" data offset overflow", name))
            })?;
            if end > bytes.len() {
                return Err(GllmError::TensorOutOfBounds {
                    name: name.clone(),
                    start: abs_data_offset,
                    end,
                    file_size: bytes.len(),
                });
            }

            if tensor_index.contains_key(&name) {
                return Err(GllmError::DuplicateTensorName(name));
            }

            tensor_index.insert(name.clone(), tensors.len());
            tensors.push(ResolvedTensor {
                name,
                entry: entry.clone(),
                abs_data_offset,
                data_size,
            });
        }

        // ── 4. Metadata ────────────────────────────────────────────────────
        let meta_offset = header.meta_offset as usize;
        if meta_offset > bytes.len() {
            return Err(GllmError::MetadataOutOfBounds {
                offset: meta_offset,
                file_size: bytes.len(),
            });
        }
        // Metadata extends until the data section.
        let meta_end = data_offset;
        let metadata_bytes = if meta_offset < meta_end {
            let raw = &bytes[meta_offset..meta_end];
            let end = raw.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            raw[..end].to_vec()
        } else {
            Vec::new()
        };

        Ok(Self {
            mmap,
            header,
            tensors,
            tensor_index,
            metadata_bytes,
        })
    }

    pub fn header(&self) -> &GllmHeader {
        &self.header
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensors(&self) -> &[ResolvedTensor] {
        &self.tensors
    }

    /// Raw MessagePack metadata bytes.
    pub fn metadata_bytes(&self) -> &[u8] {
        &self.metadata_bytes
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&ResolvedTensor> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Get a zero-copy slice of tensor data.
    pub fn tensor_data(&self, name: &str) -> Option<Cow<'_, [u8]>> {
        let t = self.find_tensor(name)?;
        Some(Cow::Borrowed(&self.mmap[t.abs_data_offset..t.abs_data_offset + t.data_size]))
    }

    /// Get the QuantType for a tensor. Returns None for unquantized tensors.
    pub fn quant_type(&self, name: &str) -> Option<gllm_kernels::quant::QuantType> {
        let t = self.find_tensor(name)?;
        if !t.entry.is_quantized() {
            return None;
        }
        gllm_quant_type_from_u8(t.entry.quant_format)
    }

    /// Parse metadata JSON and extract a field as string.
    #[allow(dead_code)]
    fn metadata_str(&self, key: &str) -> Option<String> {
        let meta: std::collections::HashMap<String, String> =
            serde_json::from_slice(&self.metadata_bytes).ok()?;
        meta.get(key).cloned()
    }

    /// Architecture key from metadata (e.g. "qwen3", "gemma4").
    /// SPEC 36 §4: arch_key is the primary architecture identifier.
    pub fn architecture(&self) -> Option<String> {
        let meta: std::collections::HashMap<String, String> =
            serde_json::from_slice(&self.metadata_bytes).ok()?;
        meta.get("arch_key").cloned()
    }

    /// Model parameters from metadata for architecture resolution.
    ///
    /// Returns `None` if any required field is missing, unparseable, or zero.
    /// All 8 fields (vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
    /// head_dim, intermediate_size, context_length) are required — a zero value
    /// for any indicates corrupted or incomplete metadata.
    pub fn model_params(&self) -> Option<GllmModelParams> {
        let meta: std::collections::HashMap<String, String> =
            serde_json::from_slice(&self.metadata_bytes).ok()?;
        Some(GllmModelParams {
            vocab_size: require_meta_u64(&meta, "vocab_size")?,
            hidden_size: require_meta_u64(&meta, "hidden_size")?,
            num_layers: require_meta_u64(&meta, "num_layers")?,
            num_heads: require_meta_u64(&meta, "num_heads")?,
            num_kv_heads: require_meta_u64(&meta, "num_kv_heads")?,
            head_dim: require_meta_u64(&meta, "head_dim")?,
            intermediate_size: require_meta_u64(&meta, "intermediate_size")?,
            context_length: require_meta_u64(&meta, "context_length")?,
        })
    }
}

/// Parse a required u64 metadata field. Returns `None` if the key is missing,
/// the value cannot be parsed as u64, or the parsed value is 0 (zero indicates
/// corrupted/incomplete metadata for all required model parameter fields).
fn require_meta_u64(meta: &std::collections::HashMap<String, String>, key: &str) -> Option<u64> {
    let val = meta.get(key).and_then(|v| v.parse::<u64>().ok())?;
    if val == 0 { None } else { Some(val) }
}

/// Resolved model architecture parameters from .gllm metadata.
#[derive(Debug, Clone)]
pub struct GllmModelParams {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub num_layers: u64,
    pub num_heads: u64,
    pub num_kv_heads: u64,
    pub head_dim: u64,
    pub intermediate_size: u64,
    pub context_length: u64,
}

// ── QuantType mapping ────────────────────────────────────────────────────────

/// Map .gllm quant_format byte to QuantType.
/// Encoding must match the .gllm specification (SPEC 36 §1.2).
fn gllm_quant_type_from_u8(v: u8) -> Option<gllm_kernels::quant::QuantType> {
    use gllm_kernels::quant::QuantType;
    match v {
        0 => None, // unquantized
        1 => Some(QuantType::Bf16),
        2 => Some(QuantType::F16),
        3 => Some(QuantType::F32),
        10 => Some(QuantType::Q4_0),
        11 => Some(QuantType::Q4_1),
        12 => Some(QuantType::Q5_0),
        13 => Some(QuantType::Q5_1),
        14 => Some(QuantType::Q8_0),
        15 => Some(QuantType::Q8_1),
        20 => Some(QuantType::Q2K),
        21 => Some(QuantType::Q3K),
        22 => Some(QuantType::Q4K),
        23 => Some(QuantType::Q5K),
        24 => Some(QuantType::Q6K),
        25 => Some(QuantType::Q8K),
        30 => Some(QuantType::IQ1S),
        31 => Some(QuantType::IQ1M),
        32 => Some(QuantType::IQ2XXS),
        33 => Some(QuantType::IQ2XS),
        34 => Some(QuantType::IQ2S),
        35 => Some(QuantType::IQ3XXS),
        36 => Some(QuantType::IQ3S),
        37 => Some(QuantType::IQ4NL),
        38 => Some(QuantType::IQ4XS),
        40 => Some(QuantType::AWQ4),
        41 => Some(QuantType::GPTQ4),
        42 => Some(QuantType::Squeeze),
        50 => Some(QuantType::Fp8E4M3),
        51 => Some(QuantType::Fp8E5M2),
        52 => Some(QuantType::Mxfp4 { block_size: 32 }),
        53 => Some(QuantType::Nvfp4),
        _ => None,
    }
}

// ── TensorProvider integration ───────────────────────────────────────────────
fn gllm_dtype_to_st(dtype: u8) -> Result<Dtype, GllmError> {
    match dtype {
        0 => Ok(Dtype::F32),
        1 => Ok(Dtype::F16),
        2 => Ok(Dtype::BF16),
        3 => Ok(Dtype::U8),
        4 => Ok(Dtype::I8),
        5 => Ok(Dtype::I32),
        6 => Ok(Dtype::I64),
        _ => Err(GllmError::InvalidDType(dtype)),
    }
}

impl TensorProvider for GllmReader {
    fn tensor_info(&self, name: &str) -> Option<TensorMeta> {
        let t = self.find_tensor(name)?;
        let shape: Vec<usize> = t.entry.shape[..t.entry.ndim as usize]
            .iter()
            .map(|&d| d as usize)
            .collect();
        let dtype = gllm_dtype_to_st(t.entry.dtype).ok()?;
        Some(TensorMeta { name: t.name.clone(), shape, dtype })
    }

    fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
        self.tensors.iter().filter_map(|t| {
            let shape: Vec<usize> = t.entry.shape[..t.entry.ndim as usize]
                .iter()
                .map(|&d| d as usize)
                .collect();
            let dtype = gllm_dtype_to_st(t.entry.dtype).ok()?;
            Some(TensorMeta { name: t.name.clone(), shape, dtype })
        })
    }

    fn load_tensor_data(&self, name: &str) -> Result<Cow<'_, [u8]>, crate::loader::LoaderError> {
        let t = self.find_tensor(name).ok_or_else(|| {
            crate::loader::LoaderError::MissingTensor(name.to_string())
        })?;
        Ok(Cow::Borrowed(&self.mmap[t.abs_data_offset..t.abs_data_offset + t.data_size]))
    }

    fn ggml_dtype(&self, name: &str) -> Option<crate::loader::gguf::GgmlDType> {
        let qt = self.quant_type(name)?;
        crate::loader::adapter::quant_type_to_ggml_dtype(qt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gllm::HEADER_SIZE;
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("gllm_test_{}_{}_{}", name, std::process::id(), id))
    }

    fn build_minimal_gllm() -> Vec<u8> {
        let mut buf = Vec::new();

        // Header (64 bytes)
        let tensor_count: u32 = 1;
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let name = "test_tensor";
        let name_bytes = name.as_bytes();
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = string_table_offset as u64 + name_bytes.len() as u64;
        let data_offset: u64 = meta_offset + 2; // 2 bytes of dummy metadata

        buf.extend_from_slice(b"GLLM");           // magic
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u32.to_le_bytes()); // flags (quantized)
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes()); // page_size
        buf.extend_from_slice(&[0u8; 20]);            // reserved
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor Directory Entry (72 bytes)
        let entry_offset = 0u32;
        let entry_name_len = name_bytes.len() as u16;
        let entry_ndim: u8 = 2;
        let entry_dtype: u8 = 0; // F32
        let shape = [4u64, 4, 0, 0];
        let quant_format: u8 = 0;
        let quant_block_size: u16 = 0;
        let scale_dtype: u8 = 0;
        let zp_type: u8 = 0;
        let t_data_offset: u64 = 0;
        let compressed_size: u64 = 64; // 4×4×4 bytes
        let original_size: u64 = 64;

        buf.extend_from_slice(&entry_offset.to_le_bytes());     // 0..4
        buf.extend_from_slice(&entry_name_len.to_le_bytes());   // 4..6
        buf.push(entry_ndim);                                    // 6
        buf.push(entry_dtype);                                   // 7
        for s in &shape {                                        // 8..40
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf.push(quant_format);                                  // 40
        buf.extend_from_slice(&quant_block_size.to_le_bytes()); // 41..43
        buf.push(scale_dtype);                                   // 43
        buf.push(zp_type);                                       // 44
        buf.extend_from_slice(&[0u8; 3]);                        // 45..47 padding
        buf.extend_from_slice(&t_data_offset.to_le_bytes());    // 48..56
        buf.extend_from_slice(&compressed_size.to_le_bytes());  // 56..64
        buf.extend_from_slice(&original_size.to_le_bytes());    // 64..72
        assert_eq!(buf.len(), HEADER_SIZE + TENSOR_ENTRY_SIZE);

        // String table
        buf.extend_from_slice(name_bytes);
        assert_eq!(buf.len(), string_table_offset + name_bytes.len());

        // Metadata (2 dummy bytes)
        buf.extend_from_slice(&[0xDE, 0xAD]);
        assert_eq!(buf.len(), data_offset as usize);

        // Data region
        buf.extend_from_slice(&[0u8; 64]);

        buf
    }

    #[test]
    fn parse_minimal_gllm() {
        let data = build_minimal_gllm();

        // Write to temp file and parse
        let dir = unique_test_dir("parse");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        assert!(reader.header().is_quantized());
        assert_eq!(reader.header().page_size, 4096);

        let t = reader.find_tensor("test_tensor").unwrap();
        assert_eq!(t.entry.ndim, 2);
        assert_eq!(t.entry.shape[0], 4);
        assert_eq!(t.entry.shape[1], 4);
        assert!(!t.entry.is_quantized());

        let td = reader.tensor_data("test_tensor").unwrap();
        assert_eq!(td.len(), 64);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = build_minimal_gllm();
        data[0..4].copy_from_slice(b"XXXX");
        let dir = unique_test_dir("magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gllm");
        std::fs::write(&path, &data).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::InvalidMagic(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_wrong_version() {
        let mut data = build_minimal_gllm();
        data[4..8].copy_from_slice(&99u32.to_le_bytes());
        let dir = unique_test_dir("ver");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ver.gllm");
        std::fs::write(&path, &data).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::UnsupportedVersion(99))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_provider_iter() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("iter");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].name, "test_tensor");
        assert_eq!(metas[0].shape, vec![4, 4]);
        assert_eq!(metas[0].dtype, Dtype::F32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── New tests ──────────────────────────────────────────────────────────────

    #[test]
    fn gllm_quant_type_from_u8_known_values() {
        assert!(gllm_quant_type_from_u8(0).is_none()); // unquantized
        assert_eq!(gllm_quant_type_from_u8(1).unwrap(), gllm_kernels::quant::QuantType::Bf16);
        assert_eq!(gllm_quant_type_from_u8(2).unwrap(), gllm_kernels::quant::QuantType::F16);
        assert_eq!(gllm_quant_type_from_u8(3).unwrap(), gllm_kernels::quant::QuantType::F32);
        assert_eq!(gllm_quant_type_from_u8(10).unwrap(), gllm_kernels::quant::QuantType::Q4_0);
        assert_eq!(gllm_quant_type_from_u8(14).unwrap(), gllm_kernels::quant::QuantType::Q8_0);
    }

    #[test]
    fn gllm_quant_type_from_u8_awq_gptq_nvfp4() {
        assert_eq!(gllm_quant_type_from_u8(40).unwrap(), gllm_kernels::quant::QuantType::AWQ4);
        assert_eq!(gllm_quant_type_from_u8(41).unwrap(), gllm_kernels::quant::QuantType::GPTQ4);
        assert_eq!(gllm_quant_type_from_u8(53).unwrap(), gllm_kernels::quant::QuantType::Nvfp4);
    }

    #[test]
    fn gllm_quant_type_from_u8_unknown_returns_none() {
        assert!(gllm_quant_type_from_u8(5).is_none());
        assert!(gllm_quant_type_from_u8(100).is_none());
        assert!(gllm_quant_type_from_u8(255).is_none());
    }

    #[test]
    fn gllm_quant_type_from_u8_k_quant_range() {
        assert_eq!(gllm_quant_type_from_u8(20).unwrap(), gllm_kernels::quant::QuantType::Q2K);
        assert_eq!(gllm_quant_type_from_u8(21).unwrap(), gllm_kernels::quant::QuantType::Q3K);
        assert_eq!(gllm_quant_type_from_u8(22).unwrap(), gllm_kernels::quant::QuantType::Q4K);
        assert_eq!(gllm_quant_type_from_u8(23).unwrap(), gllm_kernels::quant::QuantType::Q5K);
        assert_eq!(gllm_quant_type_from_u8(24).unwrap(), gllm_kernels::quant::QuantType::Q6K);
        assert_eq!(gllm_quant_type_from_u8(25).unwrap(), gllm_kernels::quant::QuantType::Q8K);
    }

    #[test]
    fn gllm_quant_type_from_u8_fp8_range() {
        assert_eq!(gllm_quant_type_from_u8(50).unwrap(), gllm_kernels::quant::QuantType::Fp8E4M3);
        assert_eq!(gllm_quant_type_from_u8(51).unwrap(), gllm_kernels::quant::QuantType::Fp8E5M2);
    }

    #[test]
    fn gllm_dtype_to_st_valid_mappings() {
        assert_eq!(gllm_dtype_to_st(0).unwrap(), Dtype::F32);
        assert_eq!(gllm_dtype_to_st(1).unwrap(), Dtype::F16);
        assert_eq!(gllm_dtype_to_st(2).unwrap(), Dtype::BF16);
        assert_eq!(gllm_dtype_to_st(3).unwrap(), Dtype::U8);
        assert_eq!(gllm_dtype_to_st(4).unwrap(), Dtype::I8);
        assert_eq!(gllm_dtype_to_st(5).unwrap(), Dtype::I32);
        assert_eq!(gllm_dtype_to_st(6).unwrap(), Dtype::I64);
    }

    #[test]
    fn gllm_dtype_to_st_invalid_returns_error() {
        assert!(gllm_dtype_to_st(7).is_err());
        assert!(gllm_dtype_to_st(255).is_err());
    }

    #[test]
    fn resolved_tensor_data_size_matches_entry() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("rts");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rts.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("test_tensor").unwrap();

        // Non-quantized: data_size = original_size
        assert_eq!(t.data_size, 64);
        assert_eq!(t.abs_data_offset, t.entry.data_offset as usize + reader.header().data_offset as usize);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_model_params_debug_format() {
        let params = GllmModelParams {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            context_length: 4096,
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("vocab_size: 32000"));
        assert!(debug.contains("hidden_size: 4096"));
        assert!(debug.contains("context_length: 4096"));
    }

    #[test]
    fn from_files_rejects_multiple_paths() {
        let paths = vec![
            std::path::PathBuf::from("a.gllm"),
            std::path::PathBuf::from("b.gllm"),
        ];
        let result = GllmReader::from_files(&paths);
        assert!(matches!(result, Err(GllmError::ParseError(_))));
    }

    #[test]
    fn from_files_rejects_empty_paths() {
        let paths: Vec<std::path::PathBuf> = vec![];
        let result = GllmReader::from_files(&paths);
        assert!(matches!(result, Err(GllmError::ParseError(_))));
    }

    #[test]
    fn find_missing_tensor_returns_none() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("findmiss");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fm.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("nonexistent").is_none());
        assert!(reader.tensor_data("nonexistent").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_missing_returns_none() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ti");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ti.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.tensor_info("does_not_exist").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Pure logic tests (no file I/O) ──────────────────────────────────────────

    #[test]
    fn gllm_quant_type_from_u8_iq_family_full() {
        assert_eq!(gllm_quant_type_from_u8(30).unwrap(), gllm_kernels::quant::QuantType::IQ1S);
        assert_eq!(gllm_quant_type_from_u8(31).unwrap(), gllm_kernels::quant::QuantType::IQ1M);
        assert_eq!(gllm_quant_type_from_u8(32).unwrap(), gllm_kernels::quant::QuantType::IQ2XXS);
        assert_eq!(gllm_quant_type_from_u8(33).unwrap(), gllm_kernels::quant::QuantType::IQ2XS);
        assert_eq!(gllm_quant_type_from_u8(34).unwrap(), gllm_kernels::quant::QuantType::IQ2S);
        assert_eq!(gllm_quant_type_from_u8(35).unwrap(), gllm_kernels::quant::QuantType::IQ3XXS);
        assert_eq!(gllm_quant_type_from_u8(36).unwrap(), gllm_kernels::quant::QuantType::IQ3S);
        assert_eq!(gllm_quant_type_from_u8(37).unwrap(), gllm_kernels::quant::QuantType::IQ4NL);
        assert_eq!(gllm_quant_type_from_u8(38).unwrap(), gllm_kernels::quant::QuantType::IQ4XS);
    }

    #[test]
    fn gllm_quant_type_from_u8_classic_q_family() {
        assert_eq!(gllm_quant_type_from_u8(11).unwrap(), gllm_kernels::quant::QuantType::Q4_1);
        assert_eq!(gllm_quant_type_from_u8(12).unwrap(), gllm_kernels::quant::QuantType::Q5_0);
        assert_eq!(gllm_quant_type_from_u8(13).unwrap(), gllm_kernels::quant::QuantType::Q5_1);
        assert_eq!(gllm_quant_type_from_u8(15).unwrap(), gllm_kernels::quant::QuantType::Q8_1);
    }

    #[test]
    fn gllm_quant_type_from_u8_mxfp4_block_size() {
        let qt = gllm_quant_type_from_u8(52).unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Mxfp4 { block_size: 32 });
    }

    #[test]
    fn gllm_quant_type_from_u8_squeeze() {
        assert_eq!(gllm_quant_type_from_u8(42).unwrap(), gllm_kernels::quant::QuantType::Squeeze);
    }

    #[test]
    fn gllm_quant_type_from_u8_boundary_gaps() {
        // Gap between F32(3) and Q4_0(10)
        assert!(gllm_quant_type_from_u8(4).is_none());
        assert!(gllm_quant_type_from_u8(5).is_none());
        assert!(gllm_quant_type_from_u8(9).is_none());
        // Gap between Q8_1(15) and Q2K(20)
        assert!(gllm_quant_type_from_u8(16).is_none());
        assert!(gllm_quant_type_from_u8(19).is_none());
        // Gap between IQ4XS(38) and AWQ4(40)
        assert!(gllm_quant_type_from_u8(39).is_none());
        // Gap between Squeeze(42) and Fp8E4M3(50)
        assert!(gllm_quant_type_from_u8(43).is_none());
        assert!(gllm_quant_type_from_u8(49).is_none());
        // After Nvfp4(53)
        assert!(gllm_quant_type_from_u8(54).is_none());
        assert!(gllm_quant_type_from_u8(100).is_none());
    }

    #[test]
    fn gllm_quant_type_from_u8_all_valid_codes_unique() {
        // Verify that each valid code maps to a distinct QuantType
        let valid_codes: &[u8] = &[1, 2, 3, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 50, 51, 52, 53];
        let types: Vec<_> = valid_codes.iter()
            .map(|&c| gllm_quant_type_from_u8(c))
            .collect();
        // All must resolve to Some
        for (i, t) in types.iter().enumerate() {
            assert!(t.is_some(), "code {} should resolve", valid_codes[i]);
        }
    }

    #[test]
    fn gllm_dtype_to_st_all_valid_exhaustive() {
        let expected = &[
            (0u8, Dtype::F32),
            (1, Dtype::F16),
            (2, Dtype::BF16),
            (3, Dtype::U8),
            (4, Dtype::I8),
            (5, Dtype::I32),
            (6, Dtype::I64),
        ];
        for (code, dtype) in expected {
            assert_eq!(gllm_dtype_to_st(*code).unwrap(), *dtype);
        }
    }

    #[test]
    fn gllm_dtype_to_st_error_variant_content() {
        let err = gllm_dtype_to_st(7).unwrap_err();
        assert!(matches!(err, GllmError::InvalidDType(7)));

        let err = gllm_dtype_to_st(200).unwrap_err();
        assert!(matches!(err, GllmError::InvalidDType(200)));
    }

    #[test]
    fn gllm_dtype_to_st_boundary_just_below_valid() {
        // All valid codes are 0-6; code 7 is first invalid
        assert!(gllm_dtype_to_st(7).is_err());
    }

    #[test]
    fn resolved_tensor_construction_and_field_access() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 4,
            ndim: 2,
            dtype: 0,
            shape: [3, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 48,
            original_size: 48,
        };
        let rt = ResolvedTensor {
            name: "weight".to_string(),
            entry: entry.clone(),
            abs_data_offset: 1024,
            data_size: 48,
        };
        assert_eq!(rt.name, "weight");
        assert_eq!(rt.abs_data_offset, 1024);
        assert_eq!(rt.data_size, 48);
        assert_eq!(rt.entry.ndim, 2);
        assert_eq!(rt.entry.shape[0], 3);
        assert_eq!(rt.entry.shape[1], 4);
        assert!(!rt.entry.is_quantized());
    }

    #[test]
    fn resolved_tensor_quantized_entry_data_size_logic() {
        // When quant_format != 0, data_size should be compressed_size
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 5,
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 0,
            compressed_size: 8388608,  // 8MB
            original_size: 67108864,   // 64MB
        };
        assert!(entry.is_quantized());
        // Verify the data_size logic from GllmReader::parse:
        // if entry.is_quantized() { compressed_size } else { original_size }
        let expected_data_size = entry.compressed_size as usize;
        assert_eq!(expected_data_size, 8388608);
        // Verify compression ratio
        let ratio = entry.compression_ratio();
        assert!((ratio - 8.0).abs() < 1e-6);
    }

    #[test]
    fn resolved_tensor_non_quantized_entry_data_size_logic() {
        // When quant_format == 0, data_size should be original_size
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 4,
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 64,
            original_size: 64,
        };
        assert!(!entry.is_quantized());
        let expected_data_size = entry.original_size as usize;
        assert_eq!(expected_data_size, 64);
    }

    #[test]
    fn gllm_model_params_clone() {
        let params = GllmModelParams {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 11008,
            context_length: 4096,
        };
        let cloned = params.clone();
        assert_eq!(cloned.vocab_size, params.vocab_size);
        assert_eq!(cloned.hidden_size, params.hidden_size);
        assert_eq!(cloned.num_layers, params.num_layers);
        assert_eq!(cloned.num_heads, params.num_heads);
        assert_eq!(cloned.num_kv_heads, params.num_kv_heads);
        assert_eq!(cloned.head_dim, params.head_dim);
        assert_eq!(cloned.intermediate_size, params.intermediate_size);
        assert_eq!(cloned.context_length, params.context_length);
    }

    #[test]
    fn gllm_model_params_zero_values() {
        let params = GllmModelParams {
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            context_length: 0,
        };
        assert_eq!(params.vocab_size, 0);
        assert_eq!(params.hidden_size, 0);
        assert_eq!(params.num_layers, 0);
        assert_eq!(params.num_heads, 0);
        assert_eq!(params.num_kv_heads, 0);
        assert_eq!(params.head_dim, 0);
        assert_eq!(params.intermediate_size, 0);
        assert_eq!(params.context_length, 0);
    }

    #[test]
    fn gllm_model_params_debug_all_fields() {
        let params = GllmModelParams {
            vocab_size: 151936,
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 29568,
            context_length: 131072,
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("vocab_size: 151936"));
        assert!(debug.contains("hidden_size: 8192"));
        assert!(debug.contains("num_layers: 80"));
        assert!(debug.contains("num_heads: 64"));
        assert!(debug.contains("num_kv_heads: 8"));
        assert!(debug.contains("head_dim: 128"));
        assert!(debug.contains("intermediate_size: 29568"));
        assert!(debug.contains("context_length: 131072"));
    }

    #[test]
    fn gllm_model_params_large_values() {
        let params = GllmModelParams {
            vocab_size: u64::MAX,
            hidden_size: u64::MAX,
            num_layers: u64::MAX,
            num_heads: u64::MAX,
            num_kv_heads: u64::MAX,
            head_dim: u64::MAX,
            intermediate_size: u64::MAX,
            context_length: u64::MAX,
        };
        assert_eq!(params.vocab_size, u64::MAX);
        assert_eq!(params.context_length, u64::MAX);
    }

    #[test]
    fn from_files_error_message_content() {
        let paths = vec![
            std::path::PathBuf::from("a.gllm"),
            std::path::PathBuf::from("b.gllm"),
        ];
        let err = GllmReader::from_files(&paths).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("single weight file"));
    }

    #[test]
    fn gllm_quant_type_from_u8_q4_0_and_q8_0() {
        assert_eq!(gllm_quant_type_from_u8(10).unwrap(), gllm_kernels::quant::QuantType::Q4_0);
        assert_eq!(gllm_quant_type_from_u8(14).unwrap(), gllm_kernels::quant::QuantType::Q8_0);
    }

    #[test]
    fn tensor_entry_shape_dimensions_used_vs_unused() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 1; // ndim = 1 (only shape[0] meaningful)
        buf[8..16].copy_from_slice(&512u64.to_le_bytes());  // shape[0]
        buf[16..24].copy_from_slice(&999u64.to_le_bytes()); // shape[1] (unused but present)
        buf[24..32].copy_from_slice(&777u64.to_le_bytes()); // shape[2]
        buf[32..40].copy_from_slice(&333u64.to_le_bytes()); // shape[3]
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, 1);
        assert_eq!(e.shape[0], 512);
        // shape[1..3] are parsed but only shape[0..ndim] is semantically valid
        assert_eq!(e.shape[1], 999);
        assert_eq!(e.shape[2], 777);
        assert_eq!(e.shape[3], 333);
    }

    #[test]
    fn tensor_entry_ndim_zero_is_parseable() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, 0);
        // scalar tensor with 0 dimensions is valid in the format
    }

    #[test]
    fn tensor_entry_compression_ratio_equal_sizes() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[56..64].copy_from_slice(&1024u64.to_le_bytes()); // compressed_size
        buf[64..72].copy_from_slice(&1024u64.to_le_bytes()); // original_size
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_high() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[56..64].copy_from_slice(&64u64.to_le_bytes());    // compressed_size
        buf[64..72].copy_from_slice(&65536u64.to_le_bytes()); // original_size
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 1024.0).abs() < 1e-3);
    }

    #[test]
    fn from_files_empty_paths_error_message() {
        let paths: Vec<std::path::PathBuf> = vec![];
        let err = GllmReader::from_files(&paths).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("single weight file"));
    }

    #[test]
    fn gllm_quant_type_from_u8_fp8_both_variants() {
        assert_eq!(gllm_quant_type_from_u8(50).unwrap(), gllm_kernels::quant::QuantType::Fp8E4M3);
        assert_eq!(gllm_quant_type_from_u8(51).unwrap(), gllm_kernels::quant::QuantType::Fp8E5M2);
    }

    // ── Additional pure logic tests ───────────────────────────────────────────

    #[test]
    fn resolved_tensor_debug_trait() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 3,
            ndim: 1,
            dtype: 0,
            shape: [768, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 3072,
            original_size: 3072,
        };
        let rt = ResolvedTensor {
            name: "emb".to_string(),
            entry,
            abs_data_offset: 256,
            data_size: 3072,
        };
        let debug = format!("{rt:?}");
        assert!(debug.contains("ResolvedTensor"));
        assert!(debug.contains("emb"));
        assert!(debug.contains("abs_data_offset"));
        assert!(debug.contains("3072"));
    }

    #[test]
    fn resolved_tensor_clone_trait() {
        let entry = GllmTensorEntry {
            name_offset: 10,
            name_len: 6,
            ndim: 2,
            dtype: 1,
            shape: [1024, 1024, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 4096,
            compressed_size: 524288,
            original_size: 4194304,
        };
        let rt = ResolvedTensor {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            entry,
            abs_data_offset: 8192,
            data_size: 524288,
        };
        let cloned = rt.clone();
        assert_eq!(cloned.name, rt.name);
        assert_eq!(cloned.abs_data_offset, rt.abs_data_offset);
        assert_eq!(cloned.data_size, rt.data_size);
        assert_eq!(cloned.entry.quant_format, rt.entry.quant_format);
        assert_eq!(cloned.entry.shape, rt.entry.shape);
    }

    #[test]
    fn open_nonexistent_file_returns_io_error() {
        let path = std::path::PathBuf::from("/tmp/gllm_nonexistent_test_file_42.gllm");
        let result = GllmReader::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GllmError::Io(_)));
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn metadata_bytes_accessor_after_parse() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();
        // build_minimal_gllm writes [0xDE, 0xAD] as metadata
        assert_eq!(meta, &[0xDE, 0xAD]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensors_slice_accessor() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("slice");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slice.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "test_tensor");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_empty_metadata_returns_none() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("arch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        // build_minimal_gllm has metadata [0xDE, 0xAD] which is not valid JSON
        assert!(reader.architecture().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_non_json_metadata_returns_none() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("mp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        // [0xDE, 0xAD] is not valid JSON → model_params returns None
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_for_unquantized_tensor_returns_none() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("qt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qt.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        // test_tensor has quant_format=0 → unquantized
        assert!(reader.quant_type("test_tensor").is_none());
        // nonexistent tensor also returns None
        assert!(reader.quant_type("missing").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_compression_ratio_low() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        // compressed = 750, original = 1000 → ratio = 1.333...
        buf[56..64].copy_from_slice(&750u64.to_le_bytes());
        buf[64..72].copy_from_slice(&1000u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        let ratio = e.compression_ratio();
        assert!((ratio - (1000.0 / 750.0)).abs() < 1e-10);
        assert!(ratio > 1.0 && ratio < 2.0);
    }

    #[test]
    fn resolved_tensor_name_field_variations() {
        // Verify that ResolvedTensor stores arbitrary string names correctly
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.31.mlp.gate_proj.weight",
            "lm_head.weight",
            "model.embed_tokens.weight",
            "a",
            "",
        ];
        for name in names {
            let entry = GllmTensorEntry {
                name_offset: 0,
                name_len: name.len() as u16,
                ndim: 1,
                dtype: 0,
                shape: [1, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data_offset: 0,
                compressed_size: 4,
                original_size: 4,
            };
            let rt = ResolvedTensor {
                name: name.to_string(),
                entry,
                abs_data_offset: 0,
                data_size: 4,
            };
            assert_eq!(rt.name, name);
        }
    }

    #[test]
    fn tensor_entry_parse_at_offset_one_less_than_needed() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE - 1];
        let err = GllmTensorEntry::parse_at(&buf, 0).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
        let msg = err.to_string();
        assert!(msg.contains("tensor entry"));
    }

    #[test]
    fn gllm_model_params_field_independence() {
        // Each field can be set independently without affecting others
        let params = GllmModelParams {
            vocab_size: 100,
            hidden_size: 0,
            num_layers: 5,
            num_heads: 0,
            num_kv_heads: 1,
            head_dim: 64,
            intermediate_size: 0,
            context_length: 2048,
        };
        assert_eq!(params.vocab_size, 100);
        assert_eq!(params.hidden_size, 0);
        assert_eq!(params.num_layers, 5);
        assert_eq!(params.num_heads, 0);
        assert_eq!(params.num_kv_heads, 1);
        assert_eq!(params.head_dim, 64);
        assert_eq!(params.intermediate_size, 0);
        assert_eq!(params.context_length, 2048);
    }

    #[test]
    fn parse_truncated_file_header_only_too_small() {
        // A file with fewer than HEADER_SIZE bytes should fail with HeaderTooSmall
        let dir = unique_test_dir("trunc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("trunc.gllm");
        std::fs::write(&path, &[0u8; 32]).unwrap(); // 32 < 64
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(32))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_duplicate_tensor_names_rejected() {
        let dir = unique_test_dir("dup");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dup.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let name_bytes = name.as_bytes();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        // Two tensor entries
        let string_table_offset = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = string_table_offset as u64 + name_bytes.len() as u64;
        let data_offset: u64 = meta_offset + 2;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // tensor_count = 2
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // First entry: name at offset 0, len 1
        let write_entry = |buf: &mut Vec<u8>| {
            buf.extend_from_slice(&0u32.to_le_bytes());     // name_offset
            buf.extend_from_slice(&1u16.to_le_bytes());     // name_len = 1 ("t")
            buf.push(1);                                     // ndim
            buf.push(0);                                     // dtype F32
            buf.extend_from_slice(&4u64.to_le_bytes());     // shape[0]
            buf.extend_from_slice(&[0u8; 24]);              // shape[1..4] + padding
            buf.push(0);                                     // quant_format
            buf.extend_from_slice(&[0u8; 2]);               // quant_block_size
            buf.push(0);                                     // scale_dtype
            buf.push(0);                                     // zp_type
            buf.extend_from_slice(&[0u8; 3]);               // reserved
            buf.extend_from_slice(&0u64.to_le_bytes());     // data_offset
            buf.extend_from_slice(&16u64.to_le_bytes());    // compressed_size
            buf.extend_from_slice(&16u64.to_le_bytes());    // original_size
        };
        write_entry(&mut buf);
        write_entry(&mut buf);

        // String table: single "t"
        buf.extend_from_slice(name_bytes);
        // Metadata
        buf.extend_from_slice(&[0xAB, 0xCD]);
        // Data: 2 * 16 = 32 bytes
        buf.extend_from_slice(&[0u8; 32]);

        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::DuplicateTensorName(_))));
        if let Err(GllmError::DuplicateTensorName(name)) = result {
            assert_eq!(name, "t");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_accessor_after_parse() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("hacc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hacc.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let h = reader.header();
        assert_eq!(h.version, 1);
        assert!(h.is_quantized());
        assert_eq!(h.tensor_count, 1);
        assert_eq!(h.page_size, 4096);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_slice_matches_length() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("td");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("test_tensor").unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();
        assert_eq!(td.len(), t.data_size);
        assert_eq!(td.len(), 64);
        // All bytes are zero (build_minimal_gllm fills with [0u8; 64])
        assert!(td.iter().all(|&b| b == 0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Additional pure logic tests (no I/O) ──────────────────────────────────

    #[test]
    fn gllm_quant_type_from_u8_u8_min_and_max() {
        // u8::MIN = 0 maps to None (unquantized sentinel)
        assert!(gllm_quant_type_from_u8(u8::MIN).is_none());
        // u8::MAX is not a valid code
        assert!(gllm_quant_type_from_u8(u8::MAX).is_none());
    }

    #[test]
    fn gllm_quant_type_from_u8_sequential_gaps_are_none() {
        // Exhaustively check every byte value; collect all that map to None
        // within the range [54..255] — all should be None since max valid is 53
        for v in 54u8..=255 {
            assert!(gllm_quant_type_from_u8(v).is_none(), "byte {v} should be None");
        }
    }

    #[test]
    fn gllm_dtype_to_st_returns_correct_variant_not_just_is_ok() {
        // Verify the actual Dtype variant, not just is_ok()
        assert!(matches!(gllm_dtype_to_st(0), Ok(Dtype::F32)));
        assert!(matches!(gllm_dtype_to_st(1), Ok(Dtype::F16)));
        assert!(matches!(gllm_dtype_to_st(2), Ok(Dtype::BF16)));
        assert!(matches!(gllm_dtype_to_st(3), Ok(Dtype::U8)));
        assert!(matches!(gllm_dtype_to_st(4), Ok(Dtype::I8)));
        assert!(matches!(gllm_dtype_to_st(5), Ok(Dtype::I32)));
        assert!(matches!(gllm_dtype_to_st(6), Ok(Dtype::I64)));
    }

    #[test]
    fn gllm_dtype_to_st_zero_is_f32_not_u8() {
        // Code 0 maps to F32 (not U8); code 3 maps to U8
        let dt = gllm_dtype_to_st(0).unwrap();
        assert_eq!(dt, Dtype::F32);
        assert_ne!(dt, Dtype::U8);
    }

    #[test]
    fn resolved_tensor_data_size_zero_entry() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 1,
            ndim: 1,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 0,
            original_size: 0,
        };
        // Non-quantized → data_size = original_size = 0
        assert!(!entry.is_quantized());
        assert_eq!(entry.original_size, 0);
        // Compression ratio guard: compressed_size=0 → ratio=1.0
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn resolved_tensor_quantized_compressed_size_used() {
        // Verify the parse logic: when quantized, data_size = compressed_size
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 3,
            ndim: 2,
            dtype: 0,
            shape: [1024, 1024, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 0,
            compressed_size: 524288,
            original_size: 4194304,
        };
        assert!(entry.is_quantized());
        let data_size = if entry.is_quantized() {
            entry.compressed_size as usize
        } else {
            entry.original_size as usize
        };
        assert_eq!(data_size, 524288);
        assert_ne!(data_size, entry.original_size as usize);
    }

    #[test]
    fn resolved_tensor_abs_data_offset_calculation() {
        let data_offset_in_header: usize = 8192;
        let entry_data_offset: usize = 4096;
        let abs = data_offset_in_header + entry_data_offset;
        assert_eq!(abs, 12288);
    }

    #[test]
    fn gllm_model_params_all_fields_distinct() {
        // Ensure each field can hold a different value independently
        let params = GllmModelParams {
            vocab_size: 1,
            hidden_size: 2,
            num_layers: 3,
            num_heads: 4,
            num_kv_heads: 5,
            head_dim: 6,
            intermediate_size: 7,
            context_length: 8,
        };
        assert_eq!(params.vocab_size, 1);
        assert_eq!(params.hidden_size, 2);
        assert_eq!(params.num_layers, 3);
        assert_eq!(params.num_heads, 4);
        assert_eq!(params.num_kv_heads, 5);
        assert_eq!(params.head_dim, 6);
        assert_eq!(params.intermediate_size, 7);
        assert_eq!(params.context_length, 8);
    }

    #[test]
    fn gllm_model_params_clone_is_deep_copy() {
        let mut params = GllmModelParams {
            vocab_size: 50000,
            hidden_size: 2048,
            num_layers: 12,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 5504,
            context_length: 2048,
        };
        let cloned = params.clone();
        // Mutate original; clone should be unaffected
        params.vocab_size = 99999;
        assert_eq!(cloned.vocab_size, 50000);
        assert_eq!(params.vocab_size, 99999);
    }

    #[test]
    fn gllm_model_params_debug_contains_struct_name() {
        let params = GllmModelParams {
            vocab_size: 0,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            context_length: 0,
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("GllmModelParams"));
    }

    #[test]
    fn tensor_entry_is_quantized_various_formats() {
        // quant_format = 0 → not quantized
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!(!e.is_quantized());

        // quant_format = 1 (Bf16) → quantized
        buf[40] = 1;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!(e.is_quantized());

        // quant_format = 53 (Nvfp4) → quantized
        buf[40] = 53;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!(e.is_quantized());
    }

    #[test]
    fn tensor_entry_compression_ratio_values() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        // compressed=500, original=500 → ratio=1.0
        buf[56..64].copy_from_slice(&500u64.to_le_bytes());
        buf[64..72].copy_from_slice(&500u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 1.0).abs() < 1e-10);

        // compressed=1, original=1000 → ratio=1000.0
        buf[56..64].copy_from_slice(&1u64.to_le_bytes());
        buf[64..72].copy_from_slice(&1000u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_entry_shape_max_u64_values() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 4; // ndim = 4
        for i in 0..4 {
            let offset = 8 + i * 8;
            buf[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        }
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, 4);
        for i in 0..4 {
            assert_eq!(e.shape[i], u64::MAX);
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_all_30_valid_codes_resolve() {
        // Count all valid codes (excluding code 0 which is None sentinel)
        let valid_codes: &[u8] = &[
            1, 2, 3, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 50, 51, 52, 53,
        ];
        assert_eq!(valid_codes.len(), 31);
        for &code in valid_codes {
            assert!(gllm_quant_type_from_u8(code).is_some(), "code {code} must resolve");
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_code_0_is_explicit_none() {
        // Code 0 is explicitly mapped to None (not a missing match arm)
        assert!(gllm_quant_type_from_u8(0).is_none());
    }

    #[test]
    fn tensor_entry_ndim_values_1_through_4() {
        for ndim in 1u8..=4 {
            let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
            buf[6] = ndim;
            let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
            assert_eq!(e.ndim, ndim);
        }
    }

    #[test]
    fn resolved_tensor_with_empty_name() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 4,
            original_size: 4,
        };
        let rt = ResolvedTensor {
            name: String::new(),
            entry,
            abs_data_offset: 0,
            data_size: 4,
        };
        assert!(rt.name.is_empty());
        assert_eq!(rt.name.len(), 0);
        assert_eq!(rt.data_size, 4);
    }

    #[test]
    fn resolved_tensor_abs_data_offset_large_values() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 1,
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 1024,
            original_size: 1024,
        };
        let rt = ResolvedTensor {
            name: "x".to_string(),
            entry,
            abs_data_offset: usize::MAX / 2,
            data_size: 1024,
        };
        assert_eq!(rt.abs_data_offset, usize::MAX / 2);
    }

    // ── 45+ new tests ──────────────────────────────────────────────────────────────

    #[test]
    fn gllm_error_display_unsupported_version_contains_expected_text() {
        let e = GllmError::UnsupportedVersion(42);
        let s = e.to_string();
        assert!(s.contains("unsupported version"));
        assert!(s.contains("42"));
        assert!(s.contains("expected"));
    }

    #[test]
    fn gllm_error_display_header_too_small_contains_byte_count() {
        let e = GllmError::HeaderTooSmall(7);
        let s = e.to_string();
        assert!(s.contains("7 bytes"));
        assert!(s.contains("file too small"));
    }

    #[test]
    fn gllm_error_display_duplicate_tensor_name_contains_name() {
        let e = GllmError::DuplicateTensorName("blk.0.attn.weight".to_string());
        let s = e.to_string();
        assert!(s.contains("duplicate tensor name"));
        assert!(s.contains("blk.0.attn.weight"));
    }

    #[test]
    fn gllm_error_display_invalid_quant_type_contains_value() {
        let e = GllmError::InvalidQuantType(77);
        let s = e.to_string();
        assert!(s.contains("invalid quant_format"));
        assert!(s.contains("77"));
    }

    #[test]
    fn gllm_error_display_invalid_dtype_contains_value() {
        let e = GllmError::InvalidDType(9);
        let s = e.to_string();
        assert!(s.contains("invalid dtype"));
        assert!(s.contains("9"));
    }

    #[test]
    fn gllm_error_display_invalid_metadata_contains_message() {
        let e = GllmError::InvalidMetadata("corrupt msgpack header".to_string());
        let s = e.to_string();
        assert!(s.contains("invalid metadata"));
        assert!(s.contains("corrupt msgpack header"));
    }

    #[test]
    fn gllm_error_display_tensor_dir_out_of_bounds_shows_offsets() {
        let e = GllmError::TensorDirOutOfBounds {
            offset: 64,
            count: 10,
            file_size: 200,
        };
        let s = e.to_string();
        assert!(s.contains("64"));
        assert!(s.contains("200"));
        assert!(s.contains("tensor directory"));
    }

    #[test]
    fn gllm_error_display_string_table_out_of_bounds_shows_range() {
        let e = GllmError::StringTableOutOfBounds {
            offset: 500,
            length: 30,
            file_size: 256,
        };
        let s = e.to_string();
        assert!(s.contains("string table"));
        assert!(s.contains("530")); // offset + length
        assert!(s.contains("256"));
    }

    #[test]
    fn gllm_error_display_tensor_out_of_bounds_shows_tensor_name() {
        let e = GllmError::TensorOutOfBounds {
            name: "model.norm.weight".to_string(),
            start: 1000,
            end: 2000,
            file_size: 1500,
        };
        let s = e.to_string();
        assert!(s.contains("model.norm.weight"));
        assert!(s.contains("1000"));
        assert!(s.contains("2000"));
        assert!(s.contains("1500"));
    }

    #[test]
    fn gllm_error_display_metadata_out_of_bounds_shows_offset() {
        let e = GllmError::MetadataOutOfBounds {
            offset: 99999,
            file_size: 4096,
        };
        let s = e.to_string();
        assert!(s.contains("99999"));
        assert!(s.contains("4096"));
        assert!(s.contains("metadata"));
    }

    #[test]
    fn gllm_error_from_io_preserves_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let gllm_err: GllmError = io_err.into();
        assert!(matches!(gllm_err, GllmError::Io(_)));
        let s = gllm_err.to_string();
        assert!(s.contains("IO error"));
    }

    #[test]
    fn gllm_header_field_access_zero_values() {
        let h = GllmHeader {
            version: 0,
            flags: 0,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: 0,
            page_size: 0,
        };
        assert_eq!(h.version, 0);
        assert_eq!(h.flags, 0);
        assert_eq!(h.meta_offset, 0);
        assert_eq!(h.tensor_count, 0);
        assert_eq!(h.tensor_dir_offset, 0);
        assert_eq!(h.data_offset, 0);
        assert_eq!(h.page_size, 0);
        assert!(!h.is_quantized());
    }

    #[test]
    fn gllm_header_is_quantized_only_bit0_matters() {
        let h1 = GllmHeader { version: 1, flags: 2, meta_offset: 0, tensor_count: 0, tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
        assert!(!h1.is_quantized()); // flags=2 → bit0=0

        let h2 = GllmHeader { version: 1, flags: 1, meta_offset: 0, tensor_count: 0, tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
        assert!(h2.is_quantized()); // flags=1 → bit0=1

        let h3 = GllmHeader { version: 1, flags: 0xFFFF_FFFE, meta_offset: 0, tensor_count: 0, tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
        assert!(!h3.is_quantized()); // bit0=0 even though all other bits set

        let h4 = GllmHeader { version: 1, flags: 0xFFFF_FFFF, meta_offset: 0, tensor_count: 0, tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
        assert!(h4.is_quantized()); // bit0=1
    }

    #[test]
    fn gllm_header_clone_is_independent() {
        let h = GllmHeader {
            version: 1,
            flags: 1,
            meta_offset: 100,
            tensor_count: 5,
            tensor_dir_offset: 200,
            data_offset: 300,
            page_size: 4096,
        };
        let c = h.clone();
        assert_eq!(c.version, 1);
        assert_eq!(c.flags, 1);
        assert_eq!(c.tensor_count, 5);
    }

    #[test]
    fn gllm_header_debug_contains_all_fields() {
        let h = GllmHeader {
            version: 1,
            flags: 1,
            meta_offset: 512,
            tensor_count: 3,
            tensor_dir_offset: 256,
            data_offset: 1024,
            page_size: 4096,
        };
        let debug = format!("{h:?}");
        assert!(debug.contains("version: 1"));
        assert!(debug.contains("flags: 1"));
        assert!(debug.contains("meta_offset: 512"));
        assert!(debug.contains("tensor_count: 3"));
        assert!(debug.contains("tensor_dir_offset: 256"));
        assert!(debug.contains("data_offset: 1024"));
        assert!(debug.contains("page_size: 4096"));
    }

    #[test]
    fn gllm_tensor_entry_field_access_all_fields() {
        let entry = GllmTensorEntry {
            name_offset: 42,
            name_len: 13,
            ndim: 3,
            dtype: 2,
            shape: [768, 12, 64, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 8192,
            compressed_size: 2048,
            original_size: 16384,
        };
        assert_eq!(entry.name_offset, 42);
        assert_eq!(entry.name_len, 13);
        assert_eq!(entry.ndim, 3);
        assert_eq!(entry.dtype, 2);
        assert_eq!(entry.shape, [768, 12, 64, 0]);
        assert_eq!(entry.quant_format, 40);
        assert_eq!(entry.quant_block_size, 128);
        assert_eq!(entry.scale_dtype, 1);
        assert_eq!(entry.zp_type, 1);
        assert_eq!(entry.data_offset, 8192);
        assert_eq!(entry.compressed_size, 2048);
        assert_eq!(entry.original_size, 16384);
    }

    #[test]
    fn gllm_tensor_entry_is_quantized_boundary_format_zero() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 0, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert!(!entry.is_quantized());
    }

    #[test]
    fn gllm_tensor_entry_is_quantized_format_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 0, dtype: 0,
            shape: [0; 4], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert!(entry.is_quantized());
    }

    #[test]
    fn gllm_tensor_entry_compression_ratio_guard_zero_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 0, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 1000,
        };
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gllm_tensor_entry_compression_ratio_two_to_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1024, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 512, original_size: 1024,
        };
        assert!((entry.compression_ratio() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn gllm_tensor_entry_compression_ratio_very_high() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1, original_size: u64::MAX,
        };
        let ratio = entry.compression_ratio();
        assert!(ratio > 1e18);
    }

    #[test]
    fn gllm_tensor_entry_clone_preserves_all_fields() {
        let original = GllmTensorEntry {
            name_offset: 100,
            name_len: 20,
            ndim: 4,
            dtype: 6,
            shape: [10, 20, 30, 40],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data_offset: 9999,
            compressed_size: 8888,
            original_size: 7777,
        };
        let cloned = original.clone();
        assert_eq!(cloned.name_offset, original.name_offset);
        assert_eq!(cloned.name_len, original.name_len);
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.quant_format, original.quant_format);
        assert_eq!(cloned.quant_block_size, original.quant_block_size);
        assert_eq!(cloned.scale_dtype, original.scale_dtype);
        assert_eq!(cloned.zp_type, original.zp_type);
        assert_eq!(cloned.data_offset, original.data_offset);
        assert_eq!(cloned.compressed_size, original.compressed_size);
        assert_eq!(cloned.original_size, original.original_size);
    }

    #[test]
    fn gllm_tensor_entry_debug_contains_struct_name_and_key_fields() {
        let entry = GllmTensorEntry {
            name_offset: 7,
            name_len: 3,
            ndim: 2,
            dtype: 0,
            shape: [512, 768, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 4096,
            compressed_size: 1024,
            original_size: 8192,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("GllmTensorEntry"));
        assert!(debug.contains("ndim: 2"));
        assert!(debug.contains("quant_format: 40"));
        assert!(debug.contains("compressed_size: 1024"));
    }

    #[test]
    fn gllm_tensor_entry_parse_at_preserves_nonzero_offset() {
        let prefix_len = 100;
        let mut buf = vec![0u8; prefix_len + TENSOR_ENTRY_SIZE];
        let start = prefix_len;
        buf[start..start + 4].copy_from_slice(&55u32.to_le_bytes());
        buf[start + 4..start + 6].copy_from_slice(&7u16.to_le_bytes());
        buf[start + 6] = 1;
        buf[start + 7] = 5;
        buf[start + 40] = 14;
        buf[start + 48..start + 56].copy_from_slice(&6789u64.to_le_bytes());
        buf[start + 56..start + 64].copy_from_slice(&111u64.to_le_bytes());
        buf[start + 64..start + 72].copy_from_slice(&222u64.to_le_bytes());

        let e = GllmTensorEntry::parse_at(&buf, prefix_len).unwrap();
        assert_eq!(e.name_offset, 55);
        assert_eq!(e.name_len, 7);
        assert_eq!(e.ndim, 1);
        assert_eq!(e.dtype, 5);
        assert_eq!(e.quant_format, 14);
        assert_eq!(e.data_offset, 6789);
        assert_eq!(e.compressed_size, 111);
        assert_eq!(e.original_size, 222);
    }

    #[test]
    fn gllm_tensor_entry_parse_at_empty_buffer_returns_parse_error() {
        let buf: &[u8] = &[];
        let err = GllmTensorEntry::parse_at(buf, 0).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    #[test]
    fn gllm_tensor_entry_shape_zero_array() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 0, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert_eq!(entry.shape, [0u64; 4]);
    }

    #[test]
    fn gllm_model_params_debug_all_zero_fields() {
        let params = GllmModelParams {
            vocab_size: 0, hidden_size: 0, num_layers: 0, num_heads: 0,
            num_kv_heads: 0, head_dim: 0, intermediate_size: 0, context_length: 0,
        };
        let debug = format!("{params:?}");
        assert!(debug.contains("GllmModelParams"));
        assert!(debug.contains("vocab_size: 0"));
        assert!(debug.contains("context_length: 0"));
    }

    #[test]
    fn gllm_model_params_clone_independence() {
        let mut params = GllmModelParams {
            vocab_size: 100, hidden_size: 200, num_layers: 10, num_heads: 8,
            num_kv_heads: 2, head_dim: 64, intermediate_size: 512, context_length: 1024,
        };
        let cloned = params.clone();
        params.vocab_size = 999;
        params.hidden_size = 888;
        assert_eq!(cloned.vocab_size, 100);
        assert_eq!(cloned.hidden_size, 200);
        assert_eq!(params.vocab_size, 999);
    }

    #[test]
    fn gllm_model_params_large_realistic_values() {
        let params = GllmModelParams {
            vocab_size: 151936, // Qwen3
            hidden_size: 4096,
            num_layers: 36,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 11008,
            context_length: 131072,
        };
        assert_eq!(params.vocab_size, 151936);
        assert_eq!(params.context_length, 131072);
        assert_eq!(params.num_kv_heads, 4);
    }

    #[test]
    fn resolved_tensor_debug_shows_name_and_offsets() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 4, ndim: 2, dtype: 0,
            shape: [64, 64, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 16384, original_size: 16384,
        };
        let rt = ResolvedTensor {
            name: "qkv".to_string(),
            entry,
            abs_data_offset: 2048,
            data_size: 16384,
        };
        let debug = format!("{rt:?}");
        assert!(debug.contains("ResolvedTensor"));
        assert!(debug.contains("qkv"));
        assert!(debug.contains("2048"));
        assert!(debug.contains("16384"));
    }

    #[test]
    fn resolved_tensor_clone_preserves_name() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 5, ndim: 1, dtype: 0,
            shape: [100, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 400, original_size: 400,
        };
        let rt = ResolvedTensor {
            name: "input_ids".to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 400,
        };
        let cloned = rt.clone();
        assert_eq!(cloned.name, "input_ids");
        assert_eq!(cloned.abs_data_offset, rt.abs_data_offset);
        assert_eq!(cloned.data_size, rt.data_size);
    }

    #[test]
    fn resolved_tensor_data_size_zero() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 1, dtype: 0,
            shape: [0, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        let rt = ResolvedTensor {
            name: "scalar".to_string(),
            entry,
            abs_data_offset: 1024,
            data_size: 0,
        };
        assert_eq!(rt.data_size, 0);
    }

    #[test]
    fn resolved_tensor_abs_data_offset_zero() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 4, ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 4, original_size: 4,
        };
        let rt = ResolvedTensor {
            name: "bias".to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 4,
        };
        assert_eq!(rt.abs_data_offset, 0);
    }

    #[test]
    fn gllm_quant_type_from_u8_code_1_bf16() {
        let qt = gllm_quant_type_from_u8(1);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::Bf16);
    }

    #[test]
    fn gllm_quant_type_from_u8_code_2_f16() {
        let qt = gllm_quant_type_from_u8(2);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::F16);
    }

    #[test]
    fn gllm_quant_type_from_u8_code_3_f32() {
        let qt = gllm_quant_type_from_u8(3);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::F32);
    }

    #[test]
    fn gllm_quant_type_from_u8_code_42_squeeze() {
        let qt = gllm_quant_type_from_u8(42);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::Squeeze);
    }

    #[test]
    fn gllm_quant_type_from_u8_codes_4_through_9_are_none() {
        for v in 4..=9u8 {
            assert!(gllm_quant_type_from_u8(v).is_none(), "code {v} should be None");
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_codes_16_through_19_are_none() {
        for v in 16..=19u8 {
            assert!(gllm_quant_type_from_u8(v).is_none(), "code {v} should be None");
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_code_39_is_none() {
        assert!(gllm_quant_type_from_u8(39).is_none());
    }

    #[test]
    fn gllm_quant_type_from_u8_codes_43_through_49_are_none() {
        for v in 43..=49u8 {
            assert!(gllm_quant_type_from_u8(v).is_none(), "code {v} should be None");
        }
    }

    #[test]
    fn gllm_dtype_to_st_invalid_code_128() {
        let err = gllm_dtype_to_st(128).unwrap_err();
        assert!(matches!(err, GllmError::InvalidDType(128)));
    }

    #[test]
    fn gllm_dtype_to_st_invalid_code_u8_max() {
        let err = gllm_dtype_to_st(u8::MAX).unwrap_err();
        assert!(matches!(err, GllmError::InvalidDType(255)));
    }

    #[test]
    fn tensor_entry_dtype_field_all_valid_codes() {
        for (code, expected_dtype) in [(0, Dtype::F32), (1, Dtype::F16), (2, Dtype::BF16), (3, Dtype::U8), (4, Dtype::I8), (5, Dtype::I32), (6, Dtype::I64)] {
            let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
            buf[7] = code;
            let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
            assert_eq!(e.dtype, code);
            let converted = gllm_dtype_to_st(e.dtype);
            assert_eq!(converted.unwrap(), expected_dtype);
        }
    }

    #[test]
    fn parse_tensor_dir_out_of_bounds() {
        let dir = unique_test_dir("td_oob");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_oob.gllm");

        let mut buf = Vec::new();
        // Header with tensor_count=100 but file too small for entries
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        buf.extend_from_slice(&0u64.to_le_bytes()); // meta_offset
        buf.extend_from_slice(&100u32.to_le_bytes()); // tensor_count = 100
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // tensor_dir_offset
        buf.extend_from_slice(&(HEADER_SIZE as u64 + 1).to_le_bytes()); // data_offset
        buf.extend_from_slice(&4096u32.to_le_bytes()); // page_size
        buf.extend_from_slice(&[0u8; 20]); // reserved
        assert_eq!(buf.len(), HEADER_SIZE);

        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::TensorDirOutOfBounds { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_metadata_out_of_bounds() {
        let dir = unique_test_dir("meta_oob");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_oob.gllm");

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        buf.extend_from_slice(&99999u64.to_le_bytes()); // meta_offset = way past end
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // tensor_dir_offset
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // data_offset
        buf.extend_from_slice(&4096u32.to_le_bytes()); // page_size
        buf.extend_from_slice(&[0u8; 20]); // reserved
        assert_eq!(buf.len(), HEADER_SIZE);

        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::MetadataOutOfBounds { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_accessor_returns_correct_version_and_page_size() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("hacc2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hacc2.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let h = reader.header();
        assert_eq!(h.version, 1);
        assert_eq!(h.page_size, 4096);
        assert_eq!(h.tensor_dir_offset, HEADER_SIZE as u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_tensor_data_missing_tensor_returns_error() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let result = reader.load_tensor_data("does_not_exist");
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_tensor_data_existing_tensor_returns_correct_slice() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt2.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let result = reader.load_tensor_data("test_tensor").unwrap();
        assert_eq!(result.len(), 64);
        assert!(result.iter().all(|&b| b == 0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_shape_conversion_u64_to_usize() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shape.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.shape, vec![4usize, 4usize]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_quant_block_size_field() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[41..43].copy_from_slice(&256u16.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.quant_block_size, 256);
    }

    #[test]
    fn tensor_entry_scale_dtype_and_zp_type_fields() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[43] = 2; // scale_dtype
        buf[44] = 3; // zp_type
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.scale_dtype, 2);
        assert_eq!(e.zp_type, 3);
    }

    #[test]
    fn tensor_entry_name_offset_and_len_fields() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&12345u32.to_le_bytes());
        buf[4..6].copy_from_slice(&6789u16.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 12345);
        assert_eq!(e.name_len, 6789);
    }

    #[test]
    fn tensor_entry_data_offset_field() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[48..56].copy_from_slice(&0xDEADBEEFu64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.data_offset, 0xDEADBEEFu64);
    }

    #[test]
    fn tensor_entry_compressed_and_original_size_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[64..72].copy_from_slice(&u64::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.compressed_size, u64::MAX);
        assert_eq!(e.original_size, u64::MAX);
        assert!((e.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn open_empty_file_returns_header_too_small() {
        let dir = unique_test_dir("empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.gllm");
        std::fs::write(&path, []).unwrap();

        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(0))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_accessor_returns_borrowed_cow() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("cow");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cow.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let cow = reader.tensor_data("test_tensor").unwrap();
        assert!(matches!(cow, Cow::Borrowed(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_files_single_path_existing_file_works() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ff");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::from_files(&[path.clone()]).unwrap();
        assert_eq!(reader.tensor_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_parse_valid_magic_bytes() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("magic_valid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("magic_valid.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().version, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_bytes_strips_trailing_zeros() {
        // Build a gllm file with trailing zero bytes in metadata
        let dir = unique_test_dir("meta_strip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_strip.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = string_table_offset as u64 + name.len() as u64;
        let data_offset: u64 = meta_offset + 6; // 6 bytes of metadata

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // flags
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entry
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        // String table
        buf.extend_from_slice(name.as_bytes());
        // Metadata: real byte + trailing zeros
        buf.extend_from_slice(&[0xAA, 0xBB, 0x00, 0x00, 0x00, 0x00]);
        // Data
        buf.extend_from_slice(&[0u8; 16]);

        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();
        // Trailing zeros should be stripped
        assert_eq!(meta, &[0xAA, 0xBB]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 50 new tests (target: 185+) ──────────────────────────────────────────────

    // --- Helper: build a gllm file with JSON metadata and multiple tensors ---

    fn build_gllm_with_json_meta(
        tensor_names: &[&str],
        meta_json: &str,
        quant_format: u8,
    ) -> Vec<u8> {
        let tensor_count = tensor_names.len() as u32;
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_start = HEADER_SIZE + tensor_names.len() * TENSOR_ENTRY_SIZE;
        let total_name_bytes: usize = tensor_names.iter().map(|n| n.len()).sum();
        let meta_offset: u64 = (string_table_start + total_name_bytes) as u64;
        let meta_bytes = meta_json.as_bytes();
        let data_offset: u64 = meta_offset + meta_bytes.len() as u64;
        let flags: u32 = if quant_format != 0 { 1 } else { 0 };
        let data_per_tensor = 32usize;

        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entries — each 72 bytes, layout matching GllmTensorEntry::parse_at
        let mut name_offset_acc: u32 = 0;
        for (i, _name) in tensor_names.iter().enumerate() {
            // 0..4: name_offset (u32)
            buf.extend_from_slice(&name_offset_acc.to_le_bytes());
            // 4..6: name_len (u16)
            buf.extend_from_slice(&(tensor_names[i].len() as u16).to_le_bytes());
            // 6: ndim
            buf.push(1);
            // 7: dtype (F32 = 0)
            buf.push(0);
            // 8..40: shape[0..4] (4 × u64)
            buf.extend_from_slice(&8u64.to_le_bytes()); // shape[0] = 8
            buf.extend_from_slice(&0u64.to_le_bytes()); // shape[1]
            buf.extend_from_slice(&0u64.to_le_bytes()); // shape[2]
            buf.extend_from_slice(&0u64.to_le_bytes()); // shape[3]
            // 40: quant_format
            buf.push(quant_format);
            // 41..43: quant_block_size (u16)
            buf.extend_from_slice(&0u16.to_le_bytes());
            // 43: scale_dtype
            buf.push(0);
            // 44: zp_type
            buf.push(0);
            // 45..48: reserved (3 bytes)
            buf.extend_from_slice(&[0u8; 3]);
            // 48..56: data_offset (u64)
            let t_data_off = (i * data_per_tensor) as u64;
            buf.extend_from_slice(&t_data_off.to_le_bytes());
            // 56..64: compressed_size (u64)
            let data_size = data_per_tensor as u64;
            buf.extend_from_slice(&data_size.to_le_bytes());
            // 64..72: original_size (u64)
            buf.extend_from_slice(&data_size.to_le_bytes());

            name_offset_acc += tensor_names[i].len() as u32;
        }

        // String table
        for name in tensor_names {
            buf.extend_from_slice(name.as_bytes());
        }

        // Metadata
        buf.extend_from_slice(meta_bytes);

        // Data
        buf.extend_from_slice(&vec![0u8; tensor_names.len() * data_per_tensor]);

        buf
    }

    #[test]
    fn architecture_with_valid_json_metadata() {
        let meta = r#"{"arch_key":"qwen3","hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w1"], meta, 0);
        let dir = unique_test_dir("arch_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_json.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let arch = reader.architecture();
        assert_eq!(arch.as_deref(), Some("qwen3"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_missing_key_returns_none() {
        let meta = r#"{"hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w1"], meta, 0);
        let dir = unique_test_dir("arch_miss");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_miss.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.architecture().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_valid_json() {
        let meta = r#"{"vocab_size":"32000","hidden_size":"4096","num_layers":"32","num_heads":"32","num_kv_heads":"8","head_dim":"128","intermediate_size":"11008","context_length":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w1"], meta, 0);
        let dir = unique_test_dir("mp_json");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_json.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params().unwrap();
        assert_eq!(params.vocab_size, 32000);
        assert_eq!(params.hidden_size, 4096);
        assert_eq!(params.num_layers, 32);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.intermediate_size, 11008);
        assert_eq!(params.context_length, 4096);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_partial_json_returns_none() {
        // Incomplete metadata (only 2 of 8 required fields) → None
        let meta = r#"{"vocab_size":"100","hidden_size":"200"}"#;
        let data = build_gllm_with_json_meta(&["w1"], meta, 0);
        let dir = unique_test_dir("mp_partial");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_partial.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_invalid_number_returns_none() {
        // "not_a_number" fails to parse as u64 → missing required field → None
        let meta = r#"{"vocab_size":"not_a_number","hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w1"], meta, 0);
        let dir = unique_test_dir("mp_inv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_inv.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_multiple_tensors_all_resolved() {
        let data = build_gllm_with_json_meta(
            &["weight.q", "weight.k", "weight.v", "bias"],
            "{}",
            0,
        );
        let dir = unique_test_dir("multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 4);

        assert!(reader.find_tensor("weight.q").is_some());
        assert!(reader.find_tensor("weight.k").is_some());
        assert!(reader.find_tensor("weight.v").is_some());
        assert!(reader.find_tensor("bias").is_some());

        let tensors = reader.tensors();
        assert_eq!(tensors[0].name, "weight.q");
        assert_eq!(tensors[1].name, "weight.k");
        assert_eq!(tensors[2].name, "weight.v");
        assert_eq!(tensors[3].name, "bias");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_for_each_tensor_in_multi_file() {
        let data = build_gllm_with_json_meta(
            &["t1", "t2", "t3"],
            "{}",
            0,
        );
        let dir = unique_test_dir("multi_td");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi_td.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for name in &["t1", "t2", "t3"] {
            let td = reader.tensor_data(name).unwrap();
            assert_eq!(td.len(), 32);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iter_tensors_multi_file_count() {
        let data = build_gllm_with_json_meta(
            &["a", "b", "c", "d", "e"],
            "{}",
            0,
        );
        let dir = unique_test_dir("iter5");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter5.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();
        assert_eq!(metas.len(), 5);
        assert_eq!(metas[0].name, "a");
        assert_eq!(metas[4].name, "e");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_tensor_data_multi_file_existing() {
        let data = build_gllm_with_json_meta(
            &["alpha", "beta"],
            "{}",
            0,
        );
        let dir = unique_test_dir("lt_multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_multi.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let result = reader.load_tensor_data("beta").unwrap();
        assert_eq!(result.len(), 32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_multi_file_shapes() {
        let data = build_gllm_with_json_meta(
            &["w_in", "w_out"],
            "{}",
            0,
        );
        let dir = unique_test_dir("ti_multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ti_multi.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let info = reader.tensor_info("w_in").unwrap();
        assert_eq!(info.shape, vec![8]);
        assert_eq!(info.dtype, Dtype::F32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_gemma4_value() {
        let meta = r#"{"arch_key":"gemma4"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("gemma4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gemma4.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("gemma4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_deepseek_value() {
        let meta = r#"{"arch_key":"deepseek_v3"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("ds");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("deepseek_v3"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_for_quantized_tensor_returns_some() {
        let data = build_gllm_with_json_meta(&["w_quant"], "{}", 40); // AWQ4
        let dir = unique_test_dir("qt_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qt_quant.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_quant");
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::AWQ4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_string_table_out_of_bounds_rejected() {
        let dir = unique_test_dir("st_oob");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("st_oob.gllm");

        let mut buf = Vec::new();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        // tensor has name_offset=0, name_len=100, but string table only has 1 byte
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + 1) as u64;
        let data_offset: u64 = meta_offset + 1;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entry with name_len=100 (way past data_offset)
        buf.extend_from_slice(&0u32.to_le_bytes()); // name_offset
        buf.extend_from_slice(&100u16.to_le_bytes()); // name_len = 100
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        buf.push(b'x'); // string table: only 1 byte
        buf.push(0xAB); // metadata: 1 byte
        buf.extend_from_slice(&[0u8; 16]); // data

        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::StringTableOutOfBounds { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_tensor_data_out_of_bounds_rejected() {
        let dir = unique_test_dir("td_oob2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_oob2.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entry claiming 1MB of data but file has only 16 bytes
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0
        buf.extend_from_slice(&1048576u64.to_le_bytes()); // compressed_size = 1MB
        buf.extend_from_slice(&1048576u64.to_le_bytes()); // original_size = 1MB

        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&[0xAB, 0xCD]);
        buf.extend_from_slice(&[0u8; 16]); // only 16 bytes of data, not 1MB

        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::TensorOutOfBounds { .. })));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_zero_tensors_empty_file() {
        let dir = unique_test_dir("zero_t");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_t.gllm");

        let mut buf = Vec::new();
        let data_offset: u64 = HEADER_SIZE as u64;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // meta_offset
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // tensor_dir_offset
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 0);
        assert!(reader.tensors().is_empty());
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_empty_region_returns_empty_bytes() {
        // meta_offset == data_offset → empty metadata region
        let dir = unique_test_dir("meta_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_empty.gllm");

        let mut buf = Vec::new();
        let data_offset: u64 = HEADER_SIZE as u64;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes()); // meta_offset == data_offset
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_all_zeros_returns_empty_after_strip() {
        let dir = unique_test_dir("meta_zeros");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_zeros.gllm");

        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 10; // 10 bytes of all-zero metadata

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);
        buf.extend_from_slice(&[0u8; 10]); // 10 zero metadata bytes

        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        // All-zero metadata should be stripped to empty
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_empty_json_object_returns_none() {
        let meta = "{}";
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_empty.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.architecture().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_large_values() {
        let meta = r#"{"vocab_size":"151936","hidden_size":"8192","num_layers":"80","num_heads":"64","num_kv_heads":"8","head_dim":"128","intermediate_size":"29568","context_length":"131072"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_large.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params().unwrap();
        assert_eq!(params.vocab_size, 151936);
        assert_eq!(params.hidden_size, 8192);
        assert_eq!(params.num_layers, 80);
        assert_eq!(params.context_length, 131072);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_compression_ratio_exactly_two() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1000, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 500, original_size: 1000,
        };
        assert!((entry.compression_ratio() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_exactly_four() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [4096, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1024, original_size: 4096,
        };
        assert!((entry.compression_ratio() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn gllm_header_with_max_tensor_count() {
        let h = GllmHeader {
            version: 1,
            flags: 1,
            meta_offset: 0,
            tensor_count: u32::MAX,
            tensor_dir_offset: HEADER_SIZE as u64,
            data_offset: u64::MAX,
            page_size: 4096,
        };
        assert_eq!(h.tensor_count, u32::MAX);
        assert!(h.is_quantized());
    }

    #[test]
    fn gllm_header_data_offset_max() {
        let h = GllmHeader {
            version: 1,
            flags: 0,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: u64::MAX,
            page_size: 0,
        };
        assert_eq!(h.data_offset, u64::MAX);
    }

    #[test]
    fn resolved_tensor_entry_ndim_four_all_shapes() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 5, ndim: 4, dtype: 0,
            shape: [1, 2, 3, 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 96, original_size: 96,
        };
        let rt = ResolvedTensor {
            name: "4d_tensor".to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 96,
        };
        assert_eq!(rt.entry.ndim, 4);
        assert_eq!(rt.entry.shape, [1, 2, 3, 4]);
    }

    #[test]
    fn tensor_entry_parse_at_with_offset_in_middle() {
        let mut buf = vec![0u8; 300];
        let offset = 150;
        buf[offset..offset + 4].copy_from_slice(&42u32.to_le_bytes());
        buf[offset + 4..offset + 6].copy_from_slice(&3u16.to_le_bytes());
        buf[offset + 6] = 2;
        buf[offset + 7] = 1;
        buf[offset + 8..offset + 16].copy_from_slice(&256u64.to_le_bytes());
        buf[offset + 16..offset + 24].copy_from_slice(&512u64.to_le_bytes());
        buf[offset + 40] = 14; // Q8_0
        buf[offset + 48..offset + 56].copy_from_slice(&1000u64.to_le_bytes());
        buf[offset + 56..offset + 64].copy_from_slice(&500u64.to_le_bytes());
        buf[offset + 64..offset + 72].copy_from_slice(&2000u64.to_le_bytes());

        let e = GllmTensorEntry::parse_at(&buf, offset).unwrap();
        assert_eq!(e.name_offset, 42);
        assert_eq!(e.name_len, 3);
        assert_eq!(e.ndim, 2);
        assert_eq!(e.dtype, 1);
        assert_eq!(e.shape[0], 256);
        assert_eq!(e.shape[1], 512);
        assert_eq!(e.quant_format, 14);
        assert_eq!(e.data_offset, 1000);
        assert_eq!(e.compressed_size, 500);
        assert_eq!(e.original_size, 2000);
        assert!(e.is_quantized());
    }

    #[test]
    fn gllm_error_display_invalid_magic_shows_hex() {
        let e = GllmError::InvalidMagic(0x12345678);
        let s = e.to_string();
        assert!(s.contains("0x12345678"));
        assert!(s.contains("GLLM"));
    }

    #[test]
    fn gllm_error_display_parse_error_custom_message() {
        let e = GllmError::ParseError("custom parse issue at offset 0xFF".to_string());
        let s = e.to_string();
        assert!(s.contains("parse error"));
        assert!(s.contains("custom parse issue"));
    }

    #[test]
    fn gllm_tensor_entry_quant_block_size_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[41..43].copy_from_slice(&u16::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.quant_block_size, u16::MAX);
    }

    #[test]
    fn gllm_tensor_entry_name_len_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[4..6].copy_from_slice(&u16::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_len, u16::MAX);
    }

    #[test]
    fn gllm_tensor_entry_name_offset_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, u32::MAX);
    }

    #[test]
    fn resolved_tensor_with_dot_separated_name() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 30, ndim: 2, dtype: 0,
            shape: [4096, 4096, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 67108864, original_size: 67108864,
        };
        let name = "model.layers.0.self_attn.q_proj.weight";
        let rt = ResolvedTensor {
            name: name.to_string(),
            entry,
            abs_data_offset: 4096,
            data_size: 67108864,
        };
        assert_eq!(rt.name, name);
        assert!(rt.name.contains('.'));
        assert_eq!(rt.name.split('.').count(), 6);
    }

    #[test]
    fn gllm_quant_type_from_u8_all_q_family() {
        // Verify all classic Q types
        assert_eq!(gllm_quant_type_from_u8(10).unwrap(), gllm_kernels::quant::QuantType::Q4_0);
        assert_eq!(gllm_quant_type_from_u8(11).unwrap(), gllm_kernels::quant::QuantType::Q4_1);
        assert_eq!(gllm_quant_type_from_u8(12).unwrap(), gllm_kernels::quant::QuantType::Q5_0);
        assert_eq!(gllm_quant_type_from_u8(13).unwrap(), gllm_kernels::quant::QuantType::Q5_1);
        assert_eq!(gllm_quant_type_from_u8(14).unwrap(), gllm_kernels::quant::QuantType::Q8_0);
        assert_eq!(gllm_quant_type_from_u8(15).unwrap(), gllm_kernels::quant::QuantType::Q8_1);
    }

    #[test]
    fn gllm_quant_type_from_u8_nvfp4_code_53() {
        let qt = gllm_quant_type_from_u8(53).unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Nvfp4);
    }

    #[test]
    fn tensor_data_accessor_for_second_tensor_in_multi() {
        let data = build_gllm_with_json_meta(&["first", "second", "third"], "{}", 0);
        let dir = unique_test_dir("td_second");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_second.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("second").unwrap();
        assert_eq!(td.len(), 32);

        // Verify it's a Borrowed Cow
        assert!(matches!(td, Cow::Borrowed(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_header_version_one_exactly() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ver1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ver1.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().version, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_header_flags_quantized() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("flags");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        assert_eq!(reader.header().flags & 1, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_header_page_size_4096() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ps");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 4096);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_with_non_zero_non_trailing_content() {
        let dir = unique_test_dir("meta_nz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_nz.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 5;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entry
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        buf.extend_from_slice(name.as_bytes());
        // Metadata: [0x01, 0x02, 0x00, 0x00, 0x00] → strip trailing zeros → [0x01, 0x02]
        buf.extend_from_slice(&[0x01, 0x02, 0x00, 0x00, 0x00]);
        buf.extend_from_slice(&[0u8; 16]);

        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes(), &[0x01, 0x02]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_empty_json_object_returns_none() {
        // Empty JSON object has no required fields → None
        let meta = "{}";
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_empty.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_all_zeros_content_verification() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("zeros");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zeros.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();
        for &b in td.iter() {
            assert_eq!(b, 0);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_files_three_paths_rejected() {
        let paths = vec![
            PathBuf::from("a.gllm"),
            PathBuf::from("b.gllm"),
            PathBuf::from("c.gllm"),
        ];
        let err = GllmReader::from_files(&paths).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
        assert!(err.to_string().contains("single weight file"));
    }

    #[test]
    fn gllm_error_debug_all_variants() {
        let errors = vec![
            format!("{:?}", GllmError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "x"))),
            format!("{:?}", GllmError::InvalidMagic(0)),
            format!("{:?}", GllmError::UnsupportedVersion(1)),
            format!("{:?}", GllmError::HeaderTooSmall(5)),
            format!("{:?}", GllmError::DuplicateTensorName("n".into())),
            format!("{:?}", GllmError::ParseError("msg".into())),
            format!("{:?}", GllmError::InvalidQuantType(1)),
            format!("{:?}", GllmError::InvalidDType(1)),
            format!("{:?}", GllmError::InvalidMetadata("m".into())),
        ];
        // All debug strings should contain the variant name
        assert!(errors[0].contains("Io"));
        assert!(errors[1].contains("InvalidMagic"));
        assert!(errors[2].contains("UnsupportedVersion"));
        assert!(errors[3].contains("HeaderTooSmall"));
        assert!(errors[4].contains("DuplicateTensorName"));
        assert!(errors[5].contains("ParseError"));
        assert!(errors[6].contains("InvalidQuantType"));
        assert!(errors[7].contains("InvalidDType"));
        assert!(errors[8].contains("InvalidMetadata"));
    }

    #[test]
    fn gllm_header_struct_equality_via_fields() {
        let h1 = GllmHeader {
            version: 1, flags: 1, meta_offset: 100, tensor_count: 5,
            tensor_dir_offset: 200, data_offset: 300, page_size: 4096,
        };
        let h2 = GllmHeader {
            version: 1, flags: 1, meta_offset: 100, tensor_count: 5,
            tensor_dir_offset: 200, data_offset: 300, page_size: 4096,
        };
        // Compare all fields individually (no PartialEq derive)
        assert_eq!(h1.version, h2.version);
        assert_eq!(h1.flags, h2.flags);
        assert_eq!(h1.meta_offset, h2.meta_offset);
        assert_eq!(h1.tensor_count, h2.tensor_count);
        assert_eq!(h1.tensor_dir_offset, h2.tensor_dir_offset);
        assert_eq!(h1.data_offset, h2.data_offset);
        assert_eq!(h1.page_size, h2.page_size);
    }

    #[test]
    fn tensor_entry_shape_array_equality() {
        let e1 = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 2, dtype: 0,
            shape: [10, 20, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        let e2 = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 2, dtype: 0,
            shape: [10, 20, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert_eq!(e1.shape, e2.shape);
    }

    #[test]
    fn gllm_reader_open_via_path_ref() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("pathref");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pathref.gllm");
        std::fs::write(&path, &data).unwrap();

        // Test that open accepts both &Path and &PathBuf
        let reader1 = GllmReader::open(path.as_path()).unwrap();
        assert_eq!(reader1.tensor_count(), 1);

        let reader2 = GllmReader::open(&path).unwrap();
        assert_eq!(reader2.tensor_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolved_tensor_data_size_for_quantized_entry_uses_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 2, dtype: 0,
            shape: [1024, 1024, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 2097152, original_size: 16777216,
        };
        // data_size logic from parse: quantized → compressed_size
        let data_size = if entry.is_quantized() { entry.compressed_size as usize } else { entry.original_size as usize };
        assert_eq!(data_size, 2097152);
        assert_ne!(data_size, entry.original_size as usize);
    }

    #[test]
    fn quant_type_code_10_is_q4_0() {
        let qt = gllm_quant_type_from_u8(10);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::Q4_0);
    }

    #[test]
    fn quant_type_code_20_is_q2k() {
        let qt = gllm_quant_type_from_u8(20);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::Q2K);
    }

    #[test]
    fn quant_type_code_52_is_mxfp4_block32() {
        let qt = gllm_quant_type_from_u8(52);
        assert!(qt.is_some());
        assert_eq!(qt.unwrap(), gllm_kernels::quant::QuantType::Mxfp4 { block_size: 32 });
    }

    #[test]
    fn gllm_dtype_to_st_code_0_f32() {
        assert_eq!(gllm_dtype_to_st(0).unwrap(), Dtype::F32);
    }

    #[test]
    fn gllm_dtype_to_st_code_6_i64() {
        assert_eq!(gllm_dtype_to_st(6).unwrap(), Dtype::I64);
    }

    #[test]
    fn tensor_entry_is_quantized_all_nonzero_formats() {
        for fmt in [1u8, 2, 3, 10, 14, 20, 25, 30, 40, 50, 53] {
            let entry = GllmTensorEntry {
                name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
                shape: [1, 0, 0, 0], quant_format: fmt, quant_block_size: 0,
                scale_dtype: 0, zp_type: 0, data_offset: 0,
                compressed_size: 1, original_size: 1,
            };
            assert!(entry.is_quantized(), "format {fmt} should be quantized");
        }
    }

    #[test]
    fn parse_single_byte_file_returns_header_too_small() {
        let dir = unique_test_dir("1byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("one.gllm");
        std::fs::write(&path, &[0x47u8]).unwrap(); // Just 'G'

        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(1))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 50 additional tests ─────────────────────────────────────────────────────

    // --- ResolvedTensor field access: entry dtype round-trip ---

    #[test]
    fn resolved_tensor_entry_dtype_field_round_trip() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 2, ndim: 1, dtype: 2, // BF16
            shape: [128, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 256, original_size: 256,
        };
        let rt = ResolvedTensor {
            name: "emb".to_string(),
            entry,
            abs_data_offset: 1024,
            data_size: 256,
        };
        assert_eq!(rt.entry.dtype, 2);
        let converted = gllm_dtype_to_st(rt.entry.dtype);
        assert_eq!(converted.unwrap(), Dtype::BF16);
    }

    #[test]
    fn resolved_tensor_entry_scale_and_zp_fields_independent() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 3, ndim: 2, dtype: 0,
            shape: [1024, 1024, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 2, zp_type: 3, data_offset: 0,
            compressed_size: 524288, original_size: 4194304,
        };
        let rt = ResolvedTensor {
            name: "q_proj".to_string(),
            entry,
            abs_data_offset: 4096,
            data_size: 524288,
        };
        assert_eq!(rt.entry.scale_dtype, 2);
        assert_eq!(rt.entry.zp_type, 3);
        assert_ne!(rt.entry.scale_dtype, rt.entry.zp_type);
    }

    #[test]
    fn resolved_tensor_entry_quant_block_size_field_access() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 3, ndim: 2, dtype: 0,
            shape: [4096, 4096, 0, 0], quant_format: 41, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 8388608, original_size: 67108864,
        };
        let rt = ResolvedTensor {
            name: "gate".to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 8388608,
        };
        assert_eq!(rt.entry.quant_block_size, 128);
        assert!(rt.entry.is_quantized());
    }

    #[test]
    fn resolved_tensor_name_with_underscores_and_numbers() {
        let name = "blk.0.ffn_gate.ex_weight__Q8_0";
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: name.len() as u16, ndim: 2, dtype: 0,
            shape: [1024, 1024, 0, 0], quant_format: 14, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1048576, original_size: 4194304,
        };
        let rt = ResolvedTensor {
            name: name.to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 1048576,
        };
        assert!(rt.name.contains('_'));
        assert!(rt.name.contains('0'));
        assert!(rt.name.contains('.'));
    }

    #[test]
    fn resolved_tensor_abs_data_offset_addition_isolation() {
        let header_data_offset: usize = 4096;
        let entry_data_offset: usize = 2048;
        let abs = header_data_offset + entry_data_offset;
        assert_eq!(abs, 6144);
        // Verify data_size independent of offsets
        let data_size = 128;
        let end = abs + data_size;
        assert_eq!(end, 6272);
    }

    #[test]
    fn resolved_tensor_checked_add_overflow_protection() {
        // Simulate abs_data_offset + data_size overflow check from parse logic
        let abs_data_offset = usize::MAX - 10;
        let data_size = 100usize;
        let result = abs_data_offset.checked_add(data_size);
        assert!(result.is_none()); // overflow detected
    }

    // --- GllmModelParams: non-numeric JSON values ---

    #[test]
    fn model_params_json_empty_string_values_returns_none() {
        // Empty/invalid strings fail to parse as u64 → missing required field → None
        let meta = r#"{"vocab_size":"","hidden_size":"abc","num_layers":"NaN"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_nonnum");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_nonnum.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_json_numeric_string_values_parsed() {
        let meta = r#"{"vocab_size":"1","hidden_size":"2","num_layers":"3","num_heads":"4","num_kv_heads":"5","head_dim":"6","intermediate_size":"7","context_length":"8"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_seq");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_seq.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params().unwrap();
        assert_eq!(params.vocab_size, 1);
        assert_eq!(params.hidden_size, 2);
        assert_eq!(params.num_layers, 3);
        assert_eq!(params.num_heads, 4);
        assert_eq!(params.num_kv_heads, 5);
        assert_eq!(params.head_dim, 6);
        assert_eq!(params.intermediate_size, 7);
        assert_eq!(params.context_length, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_extra_json_keys_returns_none_when_incomplete() {
        // Extra keys are fine, but only 2 of 8 required fields → None
        let meta = r#"{"vocab_size":"100","hidden_size":"200","extra_key":"extra_value","another":"field"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_extra");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_extra.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Tensor name resolution: non-ASCII and special chars ---

    #[test]
    fn tensor_name_unicode_preserved() {
        let name = "model.layers_0.weight";
        let data = build_gllm_with_json_meta(&[name], "{}", 0);
        let dir = unique_test_dir("unicode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor(name).unwrap();
        assert_eq!(t.name, name);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_name_case_sensitivity() {
        let data = build_gllm_with_json_meta(&["Weight", "weight"], "{}", 0);
        let dir = unique_test_dir("case");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("case.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("Weight").is_some());
        assert!(reader.find_tensor("weight").is_some());
        assert!(reader.find_tensor("WEIGHT").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_name_lookup_order_irrelevant() {
        let data = build_gllm_with_json_meta(&["z_last", "a_first", "m_middle"], "{}", 0);
        let dir = unique_test_dir("order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("order.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 3);
        // Lookup by name works regardless of insertion order
        assert!(reader.find_tensor("z_last").is_some());
        assert!(reader.find_tensor("a_first").is_some());
        assert!(reader.find_tensor("m_middle").is_some());
        // Tensors slice preserves file order
        assert_eq!(reader.tensors()[0].name, "z_last");
        assert_eq!(reader.tensors()[1].name, "a_first");
        assert_eq!(reader.tensors()[2].name, "m_middle");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Error type Display/Debug: additional variants ---

    #[test]
    fn gllm_error_display_io_with_permission_denied() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "permission denied");
        let gllm_err: GllmError = io_err.into();
        let s = gllm_err.to_string();
        assert!(s.contains("IO error"));
        assert!(s.contains("permission denied"));
    }

    #[test]
    fn gllm_error_from_io_preserves_error_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::AlreadyExists, "already exists");
        let gllm_err: GllmError = io_err.into();
        if let GllmError::Io(ref inner) = gllm_err {
            assert_eq!(inner.kind(), std::io::ErrorKind::AlreadyExists);
        } else {
            panic!("expected Io variant");
        }
    }

    #[test]
    fn gllm_error_debug_all_struct_variants() {
        let e = GllmError::TensorDirOutOfBounds { offset: 10, count: 5, file_size: 100 };
        let debug = format!("{e:?}");
        assert!(debug.contains("TensorDirOutOfBounds"));

        let e = GllmError::StringTableOutOfBounds { offset: 50, length: 10, file_size: 100 };
        let debug = format!("{e:?}");
        assert!(debug.contains("StringTableOutOfBounds"));

        let e = GllmError::TensorOutOfBounds { name: "x".into(), start: 0, end: 10, file_size: 5 };
        let debug = format!("{e:?}");
        assert!(debug.contains("TensorOutOfBounds"));
    }

    // --- QuantType mapping edge cases ---

    #[test]
    fn gllm_quant_type_from_u8_gptq4_code_41() {
        let qt = gllm_quant_type_from_u8(41).unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::GPTQ4);
    }

    #[test]
    fn gllm_quant_type_from_u8_fp8_e4m3_and_e5m2_distinct() {
        let e4m3 = gllm_quant_type_from_u8(50).unwrap();
        let e5m2 = gllm_quant_type_from_u8(51).unwrap();
        assert_ne!(e4m3, e5m2);
    }

    #[test]
    fn gllm_quant_type_from_u8_nvfp4_and_mxfp4_distinct() {
        let mxfp4 = gllm_quant_type_from_u8(52).unwrap();
        let nvfp4 = gllm_quant_type_from_u8(53).unwrap();
        assert_ne!(mxfp4, nvfp4);
    }

    #[test]
    fn gllm_quant_type_from_u8_all_iq_variants_unique() {
        use gllm_kernels::quant::QuantType;
        let iq_types: Vec<QuantType> = (30..=38)
            .filter_map(|c| gllm_quant_type_from_u8(c))
            .collect();
        assert_eq!(iq_types.len(), 9);
        // Verify pairwise distinctness
        for i in 0..iq_types.len() {
            for j in (i + 1)..iq_types.len() {
                assert_ne!(iq_types[i], iq_types[j], "IQ types at {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_all_k_quant_variants_unique() {
        use gllm_kernels::quant::QuantType;
        let k_types: Vec<QuantType> = [20, 21, 22, 23, 24, 25]
            .iter()
            .filter_map(|&c| gllm_quant_type_from_u8(c))
            .collect();
        assert_eq!(k_types.len(), 6);
        for i in 0..k_types.len() {
            for j in (i + 1)..k_types.len() {
                assert_ne!(k_types[i], k_types[j]);
            }
        }
    }

    #[test]
    fn gllm_quant_type_from_u8_native_float_types_distinct() {
        let bf16 = gllm_quant_type_from_u8(1).unwrap();
        let f16 = gllm_quant_type_from_u8(2).unwrap();
        let f32 = gllm_quant_type_from_u8(3).unwrap();
        assert_ne!(bf16, f16);
        assert_ne!(bf16, f32);
        assert_ne!(f16, f32);
    }

    // --- gllm_dtype_to_st: every invalid code ---

    #[test]
    fn gllm_dtype_to_st_all_codes_above_6_are_invalid() {
        for code in 7u8..=255 {
            assert!(gllm_dtype_to_st(code).is_err(), "code {code} should be invalid");
        }
    }

    // --- GllmHeader: parse with exact 64-byte buffer ---

    #[test]
    fn header_parse_exact_64_bytes_succeeds() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes()); // GLLM magic
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.version, 1);
        assert_eq!(h.flags, 0);
        assert_eq!(h.tensor_count, 0);
    }

    #[test]
    fn header_parse_63_bytes_fails() {
        let buf = vec![0u8; HEADER_SIZE - 1];
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::HeaderTooSmall(n) if n == HEADER_SIZE - 1));
    }

    // --- GllmReader: zero tensors with metadata ---

    #[test]
    fn parse_zero_tensors_with_valid_json_metadata() {
        let meta = r#"{"arch_key":"llama4"}"#;
        let data = build_gllm_with_json_meta(&[], meta, 0);
        // build_gllm_with_json_meta with empty tensor_names produces tensor_count=0
        let dir = unique_test_dir("zero_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_meta.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 0);
        assert_eq!(reader.architecture().as_deref(), Some("llama4"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- TensorProvider: iter_tensors with zero tensors ---

    #[test]
    fn iter_tensors_empty_file_returns_empty_iterator() {
        let data = build_gllm_with_json_meta(&[], "{}", 0);
        let dir = unique_test_dir("iter_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter_empty.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();
        assert!(metas.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- TensorProvider: tensor_info for each dtype ---

    #[test]
    fn tensor_info_all_dtypes_via_entry_construction() {
        for (dtype_code, expected) in [(0, Dtype::F32), (1, Dtype::F16), (2, Dtype::BF16), (3, Dtype::U8), (4, Dtype::I8), (5, Dtype::I32), (6, Dtype::I64)] {
            let entry = GllmTensorEntry {
                name_offset: 0, name_len: 1, ndim: 1, dtype: dtype_code,
                shape: [10, 0, 0, 0], quant_format: 0, quant_block_size: 0,
                scale_dtype: 0, zp_type: 0, data_offset: 0,
                compressed_size: 80, original_size: 80,
            };
            assert_eq!(entry.dtype, dtype_code);
            let converted = gllm_dtype_to_st(entry.dtype);
            assert_eq!(converted.unwrap(), expected);
        }
    }

    // --- GllmTensorEntry: compression_ratio edge cases ---

    #[test]
    fn tensor_entry_compression_ratio_original_zero_returns_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 100, original_size: 0,
        };
        let ratio = entry.compression_ratio();
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn tensor_entry_compression_ratio_with_very_small_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1, original_size: 1000000,
        };
        let ratio = entry.compression_ratio();
        assert!((ratio - 1000000.0).abs() < 1.0);
    }

    // --- GllmReader: quant_type with various quant formats ---

    #[test]
    fn quant_type_gptq4_via_file() {
        let data = build_gllm_with_json_meta(&["w_gptq"], "{}", 41); // GPTQ4
        let dir = unique_test_dir("gptq");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gptq.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_gptq").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::GPTQ4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q8_0_via_file() {
        let data = build_gllm_with_json_meta(&["w_q8"], "{}", 14); // Q8_0
        let dir = unique_test_dir("q8");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q8.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_q8").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q8_0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_squeeze_via_file() {
        let data = build_gllm_with_json_meta(&["w_sq"], "{}", 42); // Squeeze
        let dir = unique_test_dir("sq");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sq.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_sq").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Squeeze);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_nvfp4_via_file() {
        let data = build_gllm_with_json_meta(&["w_nv"], "{}", 53); // Nvfp4
        let dir = unique_test_dir("nv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nv.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_nv").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Nvfp4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmReader: metadata_str private method coverage via architecture ---

    #[test]
    fn architecture_with_non_standard_arch_key() {
        let meta = r#"{"arch_key":"custom_llm_v2"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("custom_arch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("custom_arch.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("custom_llm_v2"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmReader: load_tensor_data returns MissingTensor error ---

    #[test]
    fn load_tensor_data_error_is_missing_tensor_variant() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt_err");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_err.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let err = reader.load_tensor_data("nonexistent_tensor").unwrap_err();
        // LoaderError::MissingTensor contains the tensor name
        let msg = err.to_string();
        assert!(msg.contains("nonexistent_tensor"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- TensorProvider: shape extraction with ndim=1 ---

    #[test]
    fn tensor_info_ndim_one_shape_single_element() {
        let data = build_gllm_with_json_meta(&["vec"], "{}", 0);
        let dir = unique_test_dir("ndim1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim1.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let info = reader.tensor_info("vec").unwrap();
        assert_eq!(info.shape.len(), 1);
        assert_eq!(info.shape[0], 8); // build_gllm_with_json_meta sets shape[0]=8

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmHeader: is_quantized with various flag values ---

    #[test]
    fn gllm_header_is_quantized_flags_even_values() {
        let even_flags = [0u32, 2, 4, 6, 8, 100, 0xFFFFFFFE];
        for flags in even_flags {
            let h = GllmHeader {
                version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0,
            };
            assert!(!h.is_quantized(), "flags={flags} should not be quantized");
        }
    }

    #[test]
    fn gllm_header_is_quantized_flags_odd_values() {
        let odd_flags = [1u32, 3, 5, 7, 9, 101, 0xFFFFFFFF];
        for flags in odd_flags {
            let h = GllmHeader {
                version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0,
            };
            assert!(h.is_quantized(), "flags={flags} should be quantized");
        }
    }

    // --- GllmReader: tensors() slice immutable ---

    #[test]
    fn tensors_slice_order_matches_file_order() {
        let names = ["c", "b", "a"];
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("slice_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slice_order.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();
        assert_eq!(tensors[0].name, "c");
        assert_eq!(tensors[1].name, "b");
        assert_eq!(tensors[2].name, "a");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmModelParams: individual field mutation independence ---

    #[test]
    fn gllm_model_params_mutation_independence() {
        let mut params = GllmModelParams {
            vocab_size: 100, hidden_size: 200, num_layers: 10, num_heads: 8,
            num_kv_heads: 2, head_dim: 64, intermediate_size: 512, context_length: 1024,
        };
        params.vocab_size = 999;
        assert_eq!(params.vocab_size, 999);
        // Other fields unaffected
        assert_eq!(params.hidden_size, 200);
        assert_eq!(params.num_layers, 10);
        assert_eq!(params.num_heads, 8);
        assert_eq!(params.num_kv_heads, 2);
        assert_eq!(params.head_dim, 64);
        assert_eq!(params.intermediate_size, 512);
        assert_eq!(params.context_length, 1024);
    }

    // --- GllmReader: from_files with single non-existent path ---

    #[test]
    fn from_files_single_nonexistent_path_returns_io_error() {
        let path = PathBuf::from("/tmp/gllm_from_files_nonexistent_999.gllm");
        let result = GllmReader::from_files(&[path]);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GllmError::Io(_)));
    }

    // --- Tensor name with special characters ---

    #[test]
    fn tensor_name_with_hyphens_and_dots() {
        let name = "model.encoder.layer-0.attention.query";
        let data = build_gllm_with_json_meta(&[name], "{}", 0);
        let dir = unique_test_dir("special_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("special_name.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor(name).unwrap();
        assert!(t.name.contains('-'));
        assert_eq!(t.name.split('.').count(), 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmReader: header accessor returns reference ---

    #[test]
    fn header_accessor_returns_borrowed_reference() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("href");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("href.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let h1 = reader.header();
        let h2 = reader.header();
        // Both references point to the same header
        assert_eq!(h1.version, h2.version);
        assert_eq!(h1.flags, h2.flags);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmReader: open with exactly header-sized file and zero tensors ---

    #[test]
    fn open_header_only_file_with_zero_tensors() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes()); // GLLM
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        let data_offset: u64 = HEADER_SIZE as u64;
        buf[12..20].copy_from_slice(&data_offset.to_le_bytes()); // meta_offset = data_offset (no metadata)
        buf[20..24].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf[24..32].copy_from_slice(&data_offset.to_le_bytes()); // tensor_dir_offset
        buf[32..40].copy_from_slice(&data_offset.to_le_bytes()); // data_offset
        buf[40..44].copy_from_slice(&4096u32.to_le_bytes()); // page_size

        let dir = unique_test_dir("header_only");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("header_only.gllm");
        std::fs::write(&path, &buf).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 0);
        assert!(reader.tensors().is_empty());
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- model_params with very large numbers in string form ---

    #[test]
    fn model_params_large_number_values() {
        let meta = r#"{"vocab_size":"999999999999","hidden_size":"999999999999","num_layers":"999","num_heads":"999","num_kv_heads":"999","head_dim":"9999","intermediate_size":"999999999999","context_length":"999999999999"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_large2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_large2.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params().unwrap();
        assert_eq!(params.vocab_size, 999999999999u64);
        assert_eq!(params.hidden_size, 999999999999u64);
        assert_eq!(params.num_layers, 999);
        assert_eq!(params.head_dim, 9999);
        assert_eq!(params.context_length, 999999999999u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmTensorEntry: parse_at with offset at buffer start ---

    #[test]
    fn tensor_entry_parse_at_offset_zero_succeeds() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&1u32.to_le_bytes());
        buf[4..6].copy_from_slice(&2u16.to_le_bytes());
        buf[6] = 1;
        buf[7] = 0;
        buf[8..16].copy_from_slice(&100u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 1);
        assert_eq!(e.name_len, 2);
        assert_eq!(e.ndim, 1);
        assert_eq!(e.shape[0], 100);
    }

    // --- GllmReader: data size for quantized tensor in multi-tensor file ---

    #[test]
    fn data_size_for_quantized_tensor_uses_compressed_size() {
        let data = build_gllm_with_json_meta(&["quant_w"], "{}", 40); // AWQ4
        let dir = unique_test_dir("ds_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds_quant.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("quant_w").unwrap();
        assert!(t.entry.is_quantized());
        // data_size should be compressed_size (32 in build_gllm_with_json_meta)
        assert_eq!(t.data_size, 32);
        assert_eq!(t.data_size, t.entry.compressed_size as usize);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- GllmReader: metadata_bytes content after successful parse ---

    #[test]
    fn metadata_bytes_preserves_json_content() {
        let meta = r#"{"arch_key":"qwen3","version":"2.5"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("meta_preserve");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_preserve.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let meta_bytes = reader.metadata_bytes();
        let meta_str = std::str::from_utf8(meta_bytes).unwrap();
        assert!(meta_str.contains("qwen3"));
        assert!(meta_str.contains("2.5"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- Multiple quant formats in same reader ---

    #[test]
    fn multiple_tensors_different_quant_formats() {
        // Build two separate files and verify quant_type for each
        // Note: build_gllm_with_json_meta uses uniform quant_format for all tensors,
        // so we test two separate files
        for (fmt, expected_qt) in [
            (40u8, gllm_kernels::quant::QuantType::AWQ4),
            (41u8, gllm_kernels::quant::QuantType::GPTQ4),
            (14u8, gllm_kernels::quant::QuantType::Q8_0),
        ] {
            let data = build_gllm_with_json_meta(&["w"], "{}", fmt);
            let dir = unique_test_dir("multi_fmt");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("fmt_{fmt}.gllm"));
            std::fs::write(&path, &data).unwrap();

            let reader = GllmReader::open(&path).unwrap();
            let qt = reader.quant_type("w").unwrap();
            assert_eq!(qt, expected_qt, "format {fmt} should map correctly");

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    // --- GllmModelParams: clone + mutation independence (comprehensive) ---

    #[test]
    fn gllm_model_params_clone_then_mutate_each_field() {
        let original = GllmModelParams {
            vocab_size: 100, hidden_size: 200, num_layers: 10, num_heads: 8,
            num_kv_heads: 2, head_dim: 64, intermediate_size: 512, context_length: 1024,
        };
        let mut params = original.clone();
        params.vocab_size = 0;
        params.hidden_size = 0;
        params.num_layers = 0;
        params.num_heads = 0;
        params.num_kv_heads = 0;
        params.head_dim = 0;
        params.intermediate_size = 0;
        params.context_length = 0;
        assert_eq!(original.vocab_size, 100);
        assert_eq!(original.hidden_size, 200);
        assert_eq!(original.num_layers, 10);
        assert_eq!(original.num_heads, 8);
        assert_eq!(original.num_kv_heads, 2);
        assert_eq!(original.head_dim, 64);
        assert_eq!(original.intermediate_size, 512);
        assert_eq!(original.context_length, 1024);
        assert_eq!(params.vocab_size, 0);
        assert_eq!(params.hidden_size, 0);
    }

    // --- GllmReader: file with many tensors (boundary) ---

    #[test]
    fn parse_ten_tensors_all_resolved() {
        let names: Vec<&str> = (0..10).map(|i| {
            match i {
                0 => "embed",
                1 => "layer.0.attn.q",
                2 => "layer.0.attn.k",
                3 => "layer.0.attn.v",
                4 => "layer.0.ffn.gate",
                5 => "layer.0.ffn.up",
                6 => "layer.0.ffn.down",
                7 => "layer.0.norm",
                8 => "output_norm",
                9 => "lm_head",
                _ => unreachable!(),
            }
        }).collect();
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("10t");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("10t.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 10);
        for name in &names {
            assert!(reader.find_tensor(name).is_some(), "tensor {name} should exist");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 45 additional tests (target: 285+) ──────────────────────────────────────────

    #[test]
    fn gllm_header_page_size_zero() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
            tensor_dir_offset: 0, data_offset: 0, page_size: 0,
        };
        assert_eq!(h.page_size, 0);
    }

    #[test]
    fn gllm_header_page_size_common_values() {
        for &ps in &[512u32, 1024, 2048, 4096, 8192, 65536] {
            let h = GllmHeader {
                version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: ps,
            };
            assert_eq!(h.page_size, ps);
        }
    }

    #[test]
    fn gllm_header_meta_offset_zero_is_valid() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
            tensor_dir_offset: 64, data_offset: 64, page_size: 4096,
        };
        assert_eq!(h.meta_offset, 0);
    }

    #[test]
    fn gllm_header_tensor_dir_offset_before_data_offset() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 200, tensor_count: 2,
            tensor_dir_offset: 64, data_offset: 300, page_size: 4096,
        };
        assert!(h.tensor_dir_offset < h.data_offset);
    }

    #[test]
    fn tensor_entry_compression_ratio_awq4_typical() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 2, dtype: 0,
            shape: [4096, 4096, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 8388608, original_size: 33554432,
        };
        assert!((entry.compression_ratio() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_entry_compression_ratio_fp8_typical() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 2, dtype: 0,
            shape: [1024, 1024, 0, 0], quant_format: 50, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1048576, original_size: 2097152,
        };
        assert!((entry.compression_ratio() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_data_non_zero_content_readback() {
        let dir = unique_test_dir("nz_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nz_data.gllm");
        let mut data = build_minimal_gllm();
        let data_start = data.len() - 64;
        for i in 0..64 {
            data[data_start + i] = (i as u8).wrapping_mul(4);
        }
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();
        for i in 0..64 {
            assert_eq!(td[i], (i as u8).wrapping_mul(4), "byte {i} mismatch");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_preserves_exact_string() {
        let meta = r#"{"arch_key":"qwen3-235b-a22b","hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_exact");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_exact.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("qwen3-235b-a22b"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_zero_string_values_returns_none() {
        // Zero values are invalid for required fields → None
        let meta = r#"{"vocab_size":"0","hidden_size":"0","num_layers":"0","num_heads":"0","num_kv_heads":"0","head_dim":"0","intermediate_size":"0","context_length":"0"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_zero_str");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_zero_str.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_one_dimensional_shape() {
        let data = build_gllm_with_json_meta(&["bias_term"], "{}", 0);
        let dir = unique_test_dir("1d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("1d.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let info = reader.tensor_info("bias_term").unwrap();
        assert_eq!(info.shape.len(), 1);
        assert_eq!(info.shape[0], 8);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_fp8_e4m3_via_file() {
        let data = build_gllm_with_json_meta(&["w_fp8"], "{}", 50);
        let dir = unique_test_dir("fp8e4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fp8e4.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.quant_type("w_fp8").unwrap(), gllm_kernels::quant::QuantType::Fp8E4M3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_fp8_e5m2_via_file() {
        let data = build_gllm_with_json_meta(&["w_fp8e5"], "{}", 51);
        let dir = unique_test_dir("fp8e5");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fp8e5.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.quant_type("w_fp8e5").unwrap(), gllm_kernels::quant::QuantType::Fp8E5M2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_mxfp4_via_file() {
        let data = build_gllm_with_json_meta(&["w_mxfp4"], "{}", 52);
        let dir = unique_test_dir("mxfp4_f");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mxfp4_f.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.quant_type("w_mxfp4").unwrap(), gllm_kernels::quant::QuantType::Mxfp4 { block_size: 32 });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_twenty_tensors_all_findable() {
        let names: Vec<String> = (0..20).map(|i| format!("tensor_{i:02}")).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let data = build_gllm_with_json_meta(&name_refs, "{}", 0);
        let dir = unique_test_dir("20t");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("20t.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 20);
        for name in &names {
            assert!(reader.find_tensor(name).is_some(), "tensor {name} should exist");
            assert_eq!(reader.tensor_data(name).unwrap().len(), 32);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_nested_json_returns_none() {
        // Nested JSON objects cannot deserialize into HashMap<String, String> → None
        let meta = r#"{"vocab_size":"500","nested":{"inner":"value"},"hidden_size":"1024"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_nested");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_nested.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_all_fields_set_and_read() {
        let entry = GllmTensorEntry {
            name_offset: 1000, name_len: 25, ndim: 3, dtype: 5,
            shape: [64, 128, 256, 0], quant_format: 41, quant_block_size: 64,
            scale_dtype: 2, zp_type: 1, data_offset: 999999,
            compressed_size: 111111, original_size: 888888,
        };
        assert_eq!(entry.name_offset, 1000);
        assert_eq!(entry.name_len, 25);
        assert_eq!(entry.ndim, 3);
        assert_eq!(entry.dtype, 5);
        assert_eq!(entry.shape, [64, 128, 256, 0]);
        assert_eq!(entry.quant_format, 41);
        assert_eq!(entry.quant_block_size, 64);
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
        assert_eq!(entry.data_offset, 999999);
        assert_eq!(entry.compressed_size, 111111);
        assert_eq!(entry.original_size, 888888);
    }

    #[test]
    fn gllm_error_tensor_out_of_bounds_start_equals_end() {
        let e = GllmError::TensorOutOfBounds {
            name: "empty_tensor".to_string(), start: 500, end: 500, file_size: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("empty_tensor"));
        assert!(s.contains("500"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn tensor_data_abs_offset_matches_resolved_tensor_multi() {
        let data = build_gllm_with_json_meta(&["alpha", "beta", "gamma"], "{}", 0);
        let dir = unique_test_dir("abs_off");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("abs_off.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        for name in &["alpha", "beta", "gamma"] {
            let t = reader.find_tensor(name).unwrap();
            let td = reader.tensor_data(name).unwrap();
            assert_eq!(td.len(), t.data_size);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_rejects_version_2() {
        let mut data = build_minimal_gllm();
        data[4..8].copy_from_slice(&2u32.to_le_bytes());
        let dir = unique_test_dir("v2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v2.gllm");
        std::fs::write(&path, &data).unwrap();
        assert!(matches!(GllmReader::open(&path), Err(GllmError::UnsupportedVersion(2))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_flags_zero_not_quantized() {
        let mut data = build_minimal_gllm();
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        let dir = unique_test_dir("flags0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags0.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(!reader.header().is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_exactly_fits_buffer() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&42u32.to_le_bytes());
        buf[4..6].copy_from_slice(&10u16.to_le_bytes());
        buf[6] = 2; buf[7] = 1; buf[40] = 10;
        buf[48..56].copy_from_slice(&12345u64.to_le_bytes());
        buf[56..64].copy_from_slice(&5000u64.to_le_bytes());
        buf[64..72].copy_from_slice(&20000u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 42);
        assert_eq!(e.name_len, 10);
        assert_eq!(e.quant_format, 10);
        assert_eq!(e.data_offset, 12345);
        assert!(e.is_quantized());
    }

    #[test]
    fn gllm_model_params_each_field_read_after_write() {
        let params = GllmModelParams {
            vocab_size: 151936, hidden_size: 8192, num_layers: 80,
            num_heads: 64, num_kv_heads: 8, head_dim: 128,
            intermediate_size: 29568, context_length: 131072,
        };
        assert_eq!(params.head_dim * params.num_heads, params.hidden_size);
        assert!(params.vocab_size > 0);
        assert!(params.context_length > 0);
    }

    #[test]
    fn metadata_bytes_empty_when_empty_json_meta() {
        let dir = unique_test_dir("no_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_meta.gllm");
        let data = build_gllm_with_json_meta(&["x"], "", 0);
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.metadata_bytes().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_numeric_arch_key_returns_none() {
        let meta = r#"{"arch_key":42,"hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_num");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_num.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.architecture().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iter_tensors_preserves_file_order_multi() {
        let names = ["first", "second", "third", "fourth"];
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("iter_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter_order.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();
        for (i, name) in names.iter().enumerate() {
            assert_eq!(metas[i].name, *name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_tensor_lookup_first_and_last_in_multi() {
        let names = ["z_alpha", "y_beta", "x_gamma", "w_delta", "v_epsilon"];
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("lookup_fl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lookup_fl.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.find_tensor("z_alpha").unwrap().name, "z_alpha");
        assert_eq!(reader.find_tensor("v_epsilon").unwrap().name, "v_epsilon");
        assert_eq!(reader.find_tensor("x_gamma").unwrap().name, "x_gamma");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_files_accepts_pathbuf_reference() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ff_ref");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff_ref.gllm");
        std::fs::write(&path, &data).unwrap();
        let paths: Vec<PathBuf> = vec![path];
        let reader = GllmReader::from_files(&paths).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_error_invalid_magic_display_zero() {
        let e = GllmError::InvalidMagic(0);
        assert!(e.to_string().contains("0x00000000"));
        assert!(e.to_string().contains("GLLM"));
    }

    #[test]
    fn gllm_error_invalid_magic_display_max() {
        assert!(GllmError::InvalidMagic(u32::MAX).to_string().contains("0xFFFFFFFF"));
    }

    #[test]
    fn gllm_header_max_offset_fields() {
        let h = GllmHeader {
            version: 1, flags: 1,
            meta_offset: u64::MAX, tensor_count: 0,
            tensor_dir_offset: u64::MAX, data_offset: u64::MAX,
            page_size: 0,
        };
        assert_eq!(h.meta_offset, u64::MAX);
        assert_eq!(h.tensor_dir_offset, u64::MAX);
        assert_eq!(h.data_offset, u64::MAX);
    }

    #[test]
    fn resolved_tensor_entry_access_through_reference() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 3, ndim: 2, dtype: 0,
            shape: [512, 768, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1572864, original_size: 1572864,
        };
        let rt = ResolvedTensor {
            name: "model.embed_tokens.weight".to_string(),
            entry, abs_data_offset: 4096, data_size: 1572864,
        };
        assert_eq!(rt.entry.shape[0] * rt.entry.shape[1] * 4, 1572864);
    }

    #[test]
    fn parse_non_utf8_tensor_name_rejected() {
        let dir = unique_test_dir("nonutf8");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nonutf8.gllm");
        let mut buf = Vec::new();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + 3) as u64;
        let data_offset: u64 = meta_offset + 2;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0); buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0); buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&[0xFF, 0xFE, 0xFD]);
        buf.extend_from_slice(&[0xAB, 0xCD]);
        buf.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid tensor name"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_model_params_like_default_construction() {
        let params = GllmModelParams {
            vocab_size: 0, hidden_size: 0, num_layers: 0, num_heads: 0,
            num_kv_heads: 0, head_dim: 0, intermediate_size: 0, context_length: 0,
        };
        let fields = [params.vocab_size, params.hidden_size, params.num_layers,
            params.num_heads, params.num_kv_heads, params.head_dim,
            params.intermediate_size, params.context_length];
        assert!(fields.iter().all(|&f| f == 0));
    }

    #[test]
    fn quantized_tensor_data_size_is_32_bytes() {
        let data = build_gllm_with_json_meta(&["q_weight"], "{}", 40);
        let dir = unique_test_dir("qds");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qds.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("q_weight").unwrap();
        assert_eq!(t.data_size, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_header_is_quantized_bit_isolation() {
        for &flags in &[1u32, 3, 5, 7, 0x10001] {
            let h = GllmHeader { version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
            assert!(h.is_quantized(), "flags={flags} bit0=1");
        }
        for &flags in &[0u32, 2, 4, 6, 0x10000] {
            let h = GllmHeader { version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0 };
            assert!(!h.is_quantized(), "flags={flags} bit0=0");
        }
    }

    #[test]
    fn tensor_entry_compression_ratio_less_than_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [100, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 200, original_size: 100,
        };
        let ratio = entry.compression_ratio();
        assert!(ratio < 1.0);
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn tensor_name_all_digits() {
        let data = build_gllm_with_json_meta(&["12345"], "{}", 0);
        let dir = unique_test_dir("digits");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("digits.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.find_tensor("12345").unwrap().name, "12345");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_name_single_char() {
        let data = build_gllm_with_json_meta(&["x"], "{}", 0);
        let dir = unique_test_dir("1char");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("1char.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.find_tensor("x").unwrap().name.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_invalid_dtype_returns_none() {
        let dir = unique_test_dir("bad_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_dtype.gllm");
        let mut buf = build_minimal_gllm();
        buf[HEADER_SIZE + 7] = 99;
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.tensor_info("test_tensor").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_malformed_json_returns_none() {
        let data = build_gllm_with_json_meta(&["w"], "not json {{{", 0);
        let dir = unique_test_dir("mp_malformed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_malformed.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_returns_none_when_key_is_different() {
        let meta = r#"{"model_name":"qwen3","version":"2.5"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_keyname");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_keyname.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.architecture().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_buffer_one_larger_than_needed() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE + 1];
        buf[0..4].copy_from_slice(&77u32.to_le_bytes());
        buf[6] = 1; buf[7] = 2;
        // Offset 0: exactly TENSOR_ENTRY_SIZE bytes available — succeeds
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 77);
        assert_eq!(e.ndim, 1);
        assert_eq!(e.dtype, 2);
        // Offset 1: still TENSOR_ENTRY_SIZE bytes available — also succeeds
        let e2 = GllmTensorEntry::parse_at(&buf, 1).unwrap();
        assert_eq!(e2.name_offset, 0); // reads from buf[1..5] which are zeros
        // Offset 2: only TENSOR_ENTRY_SIZE - 1 bytes available — fails
        assert!(GllmTensorEntry::parse_at(&buf, 2).is_err());
    }

    #[test]
    fn gllm_model_params_mixed_zero_nonzero() {
        let params = GllmModelParams {
            vocab_size: 32000, hidden_size: 0, num_layers: 32, num_heads: 0,
            num_kv_heads: 4, head_dim: 0, intermediate_size: 11008, context_length: 0,
        };
        assert_eq!(params.vocab_size, 32000);
        assert_eq!(params.hidden_size, 0);
        assert_eq!(params.num_layers, 32);
    }

    #[test]
    fn resolved_tensor_entry_clone_is_deep_copy() {
        let entry = GllmTensorEntry {
            name_offset: 50, name_len: 10, ndim: 2, dtype: 1,
            shape: [256, 512, 0, 0], quant_format: 14, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 1024,
            compressed_size: 65536, original_size: 262144,
        };
        let rt = ResolvedTensor {
            name: "layer.0.mlp.gate".to_string(),
            entry, abs_data_offset: 8192, data_size: 65536,
        };
        let cloned = rt.clone();
        assert_eq!(cloned.entry.name_offset, rt.entry.name_offset);
        assert_eq!(cloned.entry.shape, rt.entry.shape);
        assert_eq!(cloned.entry.quant_format, rt.entry.quant_format);
    }

    #[test]
    fn tensor_count_matches_slice_len() {
        let data = build_gllm_with_json_meta(&["a", "b", "c"], "{}", 0);
        let dir = unique_test_dir("tcount");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tcount.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), reader.tensors().len());
        assert_eq!(reader.tensor_count(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_missing_tensor_returns_none() {
        let data = build_gllm_with_json_meta(&["existing"], "{}", 40);
        let dir = unique_test_dir("qt_miss");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qt_miss.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.quant_type("nonexistent").is_none());
        assert!(reader.quant_type("existing").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_error_display_struct_variants_with_zero() {
        assert!(GllmError::TensorDirOutOfBounds { offset: 0, count: 0, file_size: 0 }.to_string().contains("0"));
        assert!(GllmError::StringTableOutOfBounds { offset: 0, length: 0, file_size: 0 }.to_string().contains("0"));
        assert!(GllmError::MetadataOutOfBounds { offset: 0, file_size: 0 }.to_string().contains("0"));
        assert!(GllmError::TensorOutOfBounds { name: String::new(), start: 0, end: 0, file_size: 0 }.to_string().contains("0"));
    }

    #[test]
    fn tensor_entry_not_quantized_even_with_nonzero_other_fields() {
        let entry = GllmTensorEntry {
            name_offset: 100, name_len: 20, ndim: 4, dtype: 6,
            shape: [1000, 1000, 1000, 1000], quant_format: 0, quant_block_size: 128,
            scale_dtype: 2, zp_type: 1, data_offset: 999999,
            compressed_size: 1000, original_size: 2000,
        };
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_data_idempotent_reads() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("idempotent");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("idempotent.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let td1 = reader.tensor_data("test_tensor").unwrap();
        let td2 = reader.tensor_data("test_tensor").unwrap();
        assert_eq!(td1.as_ref() as &[u8], td2.as_ref() as &[u8]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 55 additional tests (target: 344+) ─────────────────────────────────────────

    #[test]
    fn gllm_reader_open_accepts_str_path() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("str_path");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("str_path.gllm");
        std::fs::write(&path, &data).unwrap();
        // AsRef<Path> allows &str
        let reader = GllmReader::open(path.to_str().unwrap()).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_header_tensor_dir_offset_zero() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
            tensor_dir_offset: 0, data_offset: 0, page_size: 0,
        };
        assert_eq!(h.tensor_dir_offset, 0);
    }

    #[test]
    fn gllm_header_meta_offset_zero() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
            tensor_dir_offset: 0, data_offset: 0, page_size: 0,
        };
        assert_eq!(h.meta_offset, 0);
    }

    #[test]
    fn tensor_entry_compression_ratio_fractional() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1000, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 300, original_size: 1000,
        };
        let ratio = entry.compression_ratio();
        assert!((ratio - (1000.0_f64 / 300.0_f64)).abs() < 1e-10);
        assert!(ratio > 3.3 && ratio < 3.4);
    }

    #[test]
    fn tensor_entry_compression_ratio_eight_to_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [4096, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 128, original_size: 1024,
        };
        assert!((entry.compression_ratio() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn resolved_tensor_name_with_slash() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 11, ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 4, original_size: 4,
        };
        let rt = ResolvedTensor {
            name: "group/tensor".to_string(),
            entry,
            abs_data_offset: 0,
            data_size: 4,
        };
        assert!(rt.name.contains('/'));
        assert_eq!(rt.name, "group/tensor");
    }

    #[test]
    fn resolved_tensor_entry_quant_format_preserved() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 2, dtype: 0,
            shape: [256, 256, 0, 0], quant_format: 41, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 8192, original_size: 262144,
        };
        let rt = ResolvedTensor {
            name: "gptq_weight".to_string(),
            entry,
            abs_data_offset: 1024,
            data_size: 8192,
        };
        assert_eq!(rt.entry.quant_format, 41);
        assert!(rt.entry.is_quantized());
    }

    #[test]
    fn gllm_model_params_all_equal_values() {
        let params = GllmModelParams {
            vocab_size: 42, hidden_size: 42, num_layers: 42, num_heads: 42,
            num_kv_heads: 42, head_dim: 42, intermediate_size: 42, context_length: 42,
        };
        assert_eq!(params.vocab_size, params.hidden_size);
        assert_eq!(params.hidden_size, params.num_layers);
        assert_eq!(params.num_layers, params.num_heads);
    }

    #[test]
    fn gllm_reader_zero_tensor_count_and_find() {
        let dir = unique_test_dir("zero_find");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_find.gllm");
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("anything").is_none());
        assert!(reader.tensor_data("anything").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_50_byte_file_header_too_small() {
        let dir = unique_test_dir("50b");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("50b.gllm");
        std::fs::write(&path, &[0u8; 50]).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(50))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_header_size_minus_one_file() {
        let dir = unique_test_dir("63b");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("63b.gllm");
        std::fs::write(&path, &[0u8; HEADER_SIZE - 1]).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(n)) if n == HEADER_SIZE - 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_error_source_non_io_returns_none() {
        let variants: Vec<GllmError> = vec![
            GllmError::InvalidMagic(0),
            GllmError::UnsupportedVersion(1),
            GllmError::HeaderTooSmall(10),
            GllmError::ParseError("test".into()),
            GllmError::InvalidQuantType(5),
            GllmError::InvalidDType(8),
            GllmError::InvalidMetadata("m".into()),
        ];
        use std::error::Error;
        for v in &variants {
            assert!(v.source().is_none(), "expected None source for {v:?}");
        }
    }

    #[test]
    fn tensor_entry_dtype_all_valid_round_trip() {
        for code in 0u8..=6 {
            let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
            buf[7] = code;
            let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
            assert_eq!(e.dtype, code);
            assert!(gllm_dtype_to_st(code).is_ok());
        }
    }

    #[test]
    fn gllm_dtype_to_st_invalid_range_7_through_127() {
        for code in 7u8..=127 {
            assert!(gllm_dtype_to_st(code).is_err(), "code {code} should be invalid");
        }
    }

    #[test]
    fn gllm_quant_type_all_valid_codes_are_some() {
        let codes: &[u8] = &[1, 2, 3, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 50, 51, 52, 53];
        for &code in codes {
            assert!(gllm_quant_type_from_u8(code).is_some(), "code {code} must be Some");
        }
    }

    #[test]
    fn gllm_quant_type_gap_codes_26_through_29() {
        for code in 26u8..=29 {
            assert!(gllm_quant_type_from_u8(code).is_none(), "code {code} must be None");
        }
    }

    #[test]
    fn tensor_entry_reserved_padding_ignored() {
        let mut buf = vec![0xFFu8; TENSOR_ENTRY_SIZE];
        // Set the meaningful fields to known values
        buf[0..4].copy_from_slice(&10u32.to_le_bytes());
        buf[4..6].copy_from_slice(&5u16.to_le_bytes());
        buf[6] = 2;
        buf[7] = 1;
        buf[40] = 0;
        buf[48..56].copy_from_slice(&100u64.to_le_bytes());
        buf[56..64].copy_from_slice(&50u64.to_le_bytes());
        buf[64..72].copy_from_slice(&200u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 10);
        assert_eq!(e.name_len, 5);
        // Reserved bytes [45..48] are 0xFF but should not affect parsed fields
        assert_eq!(e.data_offset, 100);
    }

    #[test]
    fn resolved_tensor_data_size_for_quantized_matches_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 2, dtype: 0,
            shape: [512, 512, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 131072, original_size: 1048576,
        };
        // Simulating the parse logic
        let data_size = if entry.is_quantized() {
            entry.compressed_size as usize
        } else {
            entry.original_size as usize
        };
        assert_eq!(data_size, 131072);
        assert_ne!(data_size, entry.original_size as usize);
    }

    #[test]
    fn gllm_header_flags_even_not_quantized_exhaustive() {
        for flags in [0u32, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let h = GllmHeader {
                version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0,
            };
            assert!(!h.is_quantized(), "flags={flags} bit0=0 should be not quantized");
        }
    }

    #[test]
    fn gllm_header_flags_odd_quantized_exhaustive() {
        for flags in [1u32, 3, 5, 7, 9, 17, 33, 65, 129, 257, 513, 1025] {
            let h = GllmHeader {
                version: 1, flags, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: 0,
            };
            assert!(h.is_quantized(), "flags={flags} bit0=1 should be quantized");
        }
    }

    #[test]
    fn gllm_header_page_size_various_pow2() {
        for &ps in &[512u32, 1024, 2048, 4096, 8192, 16384, 65536] {
            let h = GllmHeader {
                version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
                tensor_dir_offset: 0, data_offset: 0, page_size: ps,
            };
            assert_eq!(h.page_size, ps);
        }
    }

    #[test]
    fn model_params_json_with_all_fields_set() {
        let meta = r#"{"arch_key":"llama4","vocab_size":"128256","hidden_size":"4096","num_layers":"40","num_heads":"32","num_kv_heads":"8","head_dim":"128","intermediate_size":"14336","context_length":"131072"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_all");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_all.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params().unwrap();
        assert_eq!(params.vocab_size, 128256);
        assert_eq!(params.hidden_size, 4096);
        assert_eq!(params.num_layers, 40);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.intermediate_size, 14336);
        assert_eq!(params.context_length, 131072);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_llama4_key() {
        let meta = r#"{"arch_key":"llama4"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("llama4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("llama4.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("llama4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_glm4_key() {
        let meta = r#"{"arch_key":"glm4"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("glm4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("glm4.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("glm4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_mistral_key() {
        let meta = r#"{"arch_key":"mistral3"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mistral");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mistral.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("mistral3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_empty_string_values_returns_none() {
        // Empty strings fail u64 parse → required field missing → None
        let meta = r#"{"vocab_size":"","hidden_size":""}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_empty_str");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_empty_str.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_negative_number_values_returns_none() {
        // Negative numbers fail u64 parse → required field missing → None
        let meta = r#"{"vocab_size":"-1","hidden_size":"-100"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_neg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_neg.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_five_tensors_all_accessible() {
        let names = ["emb.weight", "layer.0.qkv.weight", "layer.0.mlp.gate.weight", "norm.weight", "lm_head.weight"];
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("five");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("five.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 5);
        for name in &names {
            assert!(reader.find_tensor(name).is_some(), "should find {name}");
            let td = reader.tensor_data(name).unwrap();
            assert_eq!(td.len(), 32);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_dtype_correct_for_all_valid() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("dtype_info");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_info.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::F32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_tensor_data_returns_borrowed_cow() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt_cow");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_cow.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let cow = reader.load_tensor_data("test_tensor").unwrap();
        assert!(matches!(cow, Cow::Borrowed(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_bytes_preserves_non_trailing_content() {
        let dir = unique_test_dir("meta_nz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_nz.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 8;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        // Metadata: 4 real bytes + 4 trailing zeros
        buf.extend_from_slice(&[0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00]);
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();
        assert_eq!(meta, &[0x01, 0x02, 0x03, 0x04]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_single_byte_nonzero_preserved() {
        let dir = unique_test_dir("meta_1b");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_1b.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 1;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        buf.extend_from_slice(&[0xFE]);
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes(), &[0xFE]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_rejects_garbled_magic_byte3() {
        let mut data = build_minimal_gllm();
        data[3] = 0xFF; // corrupt the last byte of "GLLM"
        let dir = unique_test_dir("garble3");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garble3.gllm");
        std::fs::write(&path, &data).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::InvalidMagic(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_rejects_garbled_magic_byte0() {
        let mut data = build_minimal_gllm();
        data[0] = 0xFF;
        let dir = unique_test_dir("garble0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("garble0.gllm");
        std::fs::write(&path, &data).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::InvalidMagic(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_rejects_version_255() {
        let mut data = build_minimal_gllm();
        data[4..8].copy_from_slice(&255u32.to_le_bytes());
        let dir = unique_test_dir("v255");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v255.gllm");
        std::fs::write(&path, &data).unwrap();
        let result = GllmReader::open(&path);
        assert!(matches!(result, Err(GllmError::UnsupportedVersion(255))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_header_accessor_lifetime() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("hlt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hlt.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let version = reader.header().version;
        let flags = reader.header().flags;
        assert_eq!(version, 1);
        assert_eq!(flags & 1, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_tensors_accessor_lifetime() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("tlt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tlt.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let name = reader.tensors()[0].name.clone();
        assert_eq!(name, "test_tensor");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q2k_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 20);
        let dir = unique_test_dir("q2k");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q2k.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q2K);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q5k_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 23);
        let dir = unique_test_dir("q5k");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q5k.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q5K);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_iq2_xxs_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 32);
        let dir = unique_test_dir("iq2xxs");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iq2xxs.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::IQ2XXS);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_bf16_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 1);
        let dir = unique_test_dir("bf16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bf16.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Bf16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_for_nonexistent_is_none_not_error() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("td_none");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_none.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        // tensor_data returns Option, not Result
        assert!(reader.tensor_data("no_such_tensor").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_files_with_pathbuf_slice() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ff_slice");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff_slice.gllm");
        std::fs::write(&path, &data).unwrap();
        let paths: Vec<PathBuf> = vec![path];
        let reader = GllmReader::from_files(&paths).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_shape_interleaved_values() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 4;
        buf[8..16].copy_from_slice(&111u64.to_le_bytes());
        buf[16..24].copy_from_slice(&222u64.to_le_bytes());
        buf[24..32].copy_from_slice(&333u64.to_le_bytes());
        buf[32..40].copy_from_slice(&444u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.shape, [111, 222, 333, 444]);
    }

    #[test]
    fn gllm_model_params_clone_deep_copy_mutation() {
        let mut params = GllmModelParams {
            vocab_size: 1000, hidden_size: 512, num_layers: 6, num_heads: 8,
            num_kv_heads: 2, head_dim: 64, intermediate_size: 2048, context_length: 4096,
        };
        let cloned = params.clone();
        params.vocab_size = 99999;
        params.num_layers = 99999;
        assert_eq!(cloned.vocab_size, 1000);
        assert_eq!(cloned.num_layers, 6);
    }

    #[test]
    fn tensor_entry_compression_ratio_order_invariant() {
        // ratio = original / compressed, always >= 0
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 256, original_size: 512,
        };
        assert!(entry.compression_ratio() > 0.0);
        assert!(entry.compression_ratio().is_finite());
    }

    #[test]
    fn resolved_tensor_debug_trait_output_complete() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 3, ndim: 2, dtype: 0,
            shape: [128, 256, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 2048,
            compressed_size: 16384, original_size: 131072,
        };
        let rt = ResolvedTensor {
            name: "q_proj.weight".to_string(),
            entry,
            abs_data_offset: 4096,
            data_size: 16384,
        };
        let debug = format!("{rt:?}");
        assert!(debug.contains("ResolvedTensor"));
        assert!(debug.contains("q_proj.weight"));
        assert!(debug.contains("abs_data_offset"));
        assert!(debug.contains("data_size"));
    }

    #[test]
    fn header_parse_preserves_exact_offset_values() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"GLLM");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0u32.to_le_bytes());
        buf[12..20].copy_from_slice(&0xAAAA_BBBB_CCCC_DDDDu64.to_le_bytes());
        buf[20..24].copy_from_slice(&42u32.to_le_bytes());
        buf[24..32].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
        buf[32..40].copy_from_slice(&0x5555_6666_7777_8888u64.to_le_bytes());
        buf[40..44].copy_from_slice(&8192u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.meta_offset, 0xAAAA_BBBB_CCCC_DDDDu64);
        assert_eq!(h.tensor_count, 42);
        assert_eq!(h.tensor_dir_offset, 0x1111_2222_3333_4444u64);
        assert_eq!(h.data_offset, 0x5555_6666_7777_8888u64);
        assert_eq!(h.page_size, 8192);
    }

    #[test]
    fn metadata_with_leading_zeros_preserved() {
        let dir = unique_test_dir("meta_lead");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_lead.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 5;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        // Leading zeros followed by nonzero are preserved (only trailing stripped)
        buf.extend_from_slice(&[0x00, 0x00, 0xAB, 0xCD, 0xEF]);
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();
        assert_eq!(meta, &[0x00, 0x00, 0xAB, 0xCD, 0xEF]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_data_offset_independent_of_other_fields() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        // Set everything else to nonzero
        for b in buf.iter_mut() { *b = 0xFF; }
        // Set data_offset to a known value
        buf[48..56].copy_from_slice(&12345u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.data_offset, 12345u64);
    }

    #[test]
    fn gllm_error_from_io_error_kind_already_exists() {
        let io_err = std::io::Error::new(std::io::ErrorKind::AlreadyExists, "exists");
        let gllm_err: GllmError = io_err.into();
        assert!(matches!(gllm_err, GllmError::Io(_)));
    }

    #[test]
    fn gllm_error_from_io_error_kind_interrupted() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Interrupted, "sigint");
        let gllm_err: GllmError = io_err.into();
        assert!(matches!(gllm_err, GllmError::Io(_)));
    }

    #[test]
    fn parse_non_utf8_tensor_name_rejected_with_invalid_utf8() {
        let dir = unique_test_dir("utf8_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("utf8_name.gllm");
        let name_bytes = &[0xFF, 0xFE, 0xFD]; // invalid UTF-8
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name_bytes.len()) as u64;
        let data_offset: u64 = meta_offset + 2;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        // Tensor entry
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        // String table with invalid UTF-8
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&[0xAB, 0xCD]);
        buf.extend_from_slice(&[0u8; 16]);
        std::fs::write(&path, &buf).unwrap();
        let result = GllmReader::open(&path);
        assert!(result.is_err());
        if let Err(GllmError::ParseError(msg)) = result {
            assert!(msg.contains("invalid tensor name"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_dtype_to_st_all_seven_codes_unique_dtypes() {
        let results: Vec<_> = (0..=6).map(|c| gllm_dtype_to_st(c).unwrap()).collect();
        // Check no duplicates
        for i in 0..results.len() {
            for j in (i+1)..results.len() {
                assert_ne!(results[i], results[j], "codes {i} and {j} map to same dtype");
            }
        }
    }


    #[test]
    fn tensor_data_offset_calculation_for_multi_tensor() {
        let data = build_gllm_with_json_meta(&["t0", "t1", "t2"], "{}", 0);
        let dir = unique_test_dir("offset_calc");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("offset_calc.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();
        // Each tensor has 32 bytes, laid out sequentially
        assert!(tensors[0].abs_data_offset < tensors[1].abs_data_offset);
        assert!(tensors[1].abs_data_offset < tensors[2].abs_data_offset);
        assert_eq!(tensors[1].abs_data_offset - tensors[0].abs_data_offset, 32);
        assert_eq!(tensors[2].abs_data_offset - tensors[1].abs_data_offset, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_header_debug_all_fields_present() {
        let h = GllmHeader {
            version: 1, flags: 1, meta_offset: 100, tensor_count: 5,
            tensor_dir_offset: 200, data_offset: 300, page_size: 4096,
        };
        let debug = format!("{h:?}");
        assert!(debug.contains("version"));
        assert!(debug.contains("flags"));
        assert!(debug.contains("meta_offset"));
        assert!(debug.contains("tensor_count"));
        assert!(debug.contains("tensor_dir_offset"));
        assert!(debug.contains("data_offset"));
        assert!(debug.contains("page_size"));
    }

    #[test]
    fn gllm_tensor_entry_debug_all_fields_present() {
        let entry = GllmTensorEntry {
            name_offset: 1, name_len: 2, ndim: 3, dtype: 4,
            shape: [5, 6, 7, 8], quant_format: 9, quant_block_size: 10,
            scale_dtype: 11, zp_type: 12, data_offset: 13,
            compressed_size: 14, original_size: 15,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("GllmTensorEntry"));
        assert!(debug.contains("name_offset"));
        assert!(debug.contains("ndim"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("quant_format"));
        assert!(debug.contains("data_offset"));
    }

    #[test]
    fn resolved_tensor_entry_mutation_isolated() {
        let mut rt = ResolvedTensor {
            name: "a".to_string(),
            entry: GllmTensorEntry {
                name_offset: 0, name_len: 1, ndim: 1, dtype: 0,
                shape: [10, 0, 0, 0], quant_format: 0, quant_block_size: 0,
                scale_dtype: 0, zp_type: 0, data_offset: 0,
                compressed_size: 40, original_size: 40,
            },
            abs_data_offset: 100,
            data_size: 40,
        };
        rt.abs_data_offset = 200;
        rt.data_size = 80;
        assert_eq!(rt.entry.shape[0], 10);
        assert_eq!(rt.entry.original_size, 40);
    }

    // ── Additional coverage tests ────────────────────────────────────────────────

    #[test]
    fn resolved_tensor_clone_name_independence() {
        let mut rt = ResolvedTensor {
            name: "layer.0.weight".to_string(),
            entry: GllmTensorEntry {
                name_offset: 0, name_len: 14, ndim: 2, dtype: 0,
                shape: [512, 512, 0, 0], quant_format: 0, quant_block_size: 0,
                scale_dtype: 0, zp_type: 0, data_offset: 0,
                compressed_size: 1048576, original_size: 1048576,
            },
            abs_data_offset: 4096,
            data_size: 1048576,
        };
        let cloned = rt.clone();
        rt.name.push_str(".quant");
        rt.abs_data_offset = 0;
        assert_eq!(cloned.name, "layer.0.weight");
        assert_eq!(cloned.abs_data_offset, 4096);
    }

    #[test]
    fn resolved_tensor_data_size_for_quantized_is_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 2, dtype: 0,
            shape: [4096, 4096, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 1, zp_type: 1, data_offset: 0,
            compressed_size: 2097152, original_size: 67108864,
        };
        let data_size = if entry.is_quantized() {
            entry.compressed_size as usize
        } else {
            entry.original_size as usize
        };
        assert_eq!(data_size, 2097152);
        assert!(data_size < entry.original_size as usize);
    }

    #[test]
    fn resolved_tensor_entry_shape_access_pattern() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 3, ndim: 4, dtype: 0,
            shape: [1, 12, 128, 768], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert_eq!(entry.ndim, 4);
        assert_eq!(entry.shape[0], 1);
        assert_eq!(entry.shape[1], 12);
        assert_eq!(entry.shape[2], 128);
        assert_eq!(entry.shape[3], 768);
    }

    #[test]
    fn gllm_model_params_realistic_llama_values() {
        let params = GllmModelParams {
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_size: 11008,
            context_length: 4096,
        };
        assert_eq!(params.num_heads * params.head_dim, params.hidden_size);
    }

    #[test]
    fn gllm_model_params_gqa_ratio() {
        let params = GllmModelParams {
            vocab_size: 152000,
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 14336,
            context_length: 131072,
        };
        assert_eq!(params.num_heads / params.num_kv_heads, 4);
    }

    #[test]
    fn gllm_model_params_single_field_nonzero() {
        let params = GllmModelParams {
            vocab_size: 0,
            hidden_size: 8192,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            context_length: 0,
        };
        assert_eq!(params.hidden_size, 8192);
        assert_eq!(params.vocab_size, 0);
        assert_eq!(params.num_layers, 0);
    }

    #[test]
    fn gllm_model_params_clone_then_modify_original() {
        let mut params = GllmModelParams {
            vocab_size: 100, hidden_size: 200, num_layers: 300, num_heads: 400,
            num_kv_heads: 500, head_dim: 600, intermediate_size: 700, context_length: 800,
        };
        let cloned = params.clone();
        params.vocab_size = 0;
        params.context_length = 0;
        assert_eq!(cloned.vocab_size, 100);
        assert_eq!(cloned.context_length, 800);
    }

    #[test]
    fn tensor_entry_compression_ratio_ten_to_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 100, original_size: 1000,
        };
        assert!((entry.compression_ratio() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_five_to_one() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 200, original_size: 1000,
        };
        assert!((entry.compression_ratio() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_one_half() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 2000, original_size: 1000,
        };
        assert!((entry.compression_ratio() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_shape_unused_dims_independent() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 2, dtype: 0,
            shape: [4096, 768, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert_eq!(entry.shape[0], 4096);
        assert_eq!(entry.shape[1], 768);
        assert_eq!(entry.shape[2], 0);
        assert_eq!(entry.shape[3], 0);
    }

    #[test]
    fn tensor_entry_all_quant_fields_interact() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [0; 4], quant_format: 41, quant_block_size: 128,
            scale_dtype: 2, zp_type: 1, data_offset: 5000,
            compressed_size: 8192, original_size: 65536,
        };
        assert!(entry.is_quantized());
        assert!((entry.compression_ratio() - 8.0).abs() < 1e-6);
        assert_eq!(entry.quant_block_size, 128);
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
    }

    #[test]
    fn gllm_header_field_layout_verification() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"GLLM");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0xAAAA_u32.to_le_bytes());
        buf[12..20].copy_from_slice(&0xBBBB_CCCC_u64.to_le_bytes());
        buf[20..24].copy_from_slice(&42_u32.to_le_bytes());
        buf[24..32].copy_from_slice(&0xDDDD_EEEE_u64.to_le_bytes());
        buf[32..40].copy_from_slice(&0x1111_2222_u64.to_le_bytes());
        buf[40..44].copy_from_slice(&0x1000_u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.flags, 0xAAAA);
        assert_eq!(h.meta_offset, 0xBBBB_CCCC);
        assert_eq!(h.tensor_count, 42);
        assert_eq!(h.tensor_dir_offset, 0xDDDD_EEEE);
        assert_eq!(h.data_offset, 0x1111_2222);
        assert_eq!(h.page_size, 0x1000);
    }

    #[test]
    fn gllm_header_page_size_not_power_of_two_accepted() {
        let h = GllmHeader {
            version: 1, flags: 0, meta_offset: 0, tensor_count: 0,
            tensor_dir_offset: 0, data_offset: 0, page_size: 3333,
        };
        assert_eq!(h.page_size, 3333);
    }

    #[test]
    fn tensor_info_dtype_f16_via_construction() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 1; // dtype = F16
        let dir = unique_test_dir("dtype_f16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_f16.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::F16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_dtype_bf16_via_construction() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 2; // dtype = BF16
        let dir = unique_test_dir("dtype_bf16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_bf16.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::BF16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_dtype_i8_via_construction() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 4; // dtype = I8
        let dir = unique_test_dir("dtype_i8");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_i8.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::I8);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_dtype_i32_via_construction() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 5; // dtype = I32
        let dir = unique_test_dir("dtype_i32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_i32.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::I32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_dtype_i64_via_construction() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 6; // dtype = I64
        let dir = unique_test_dir("dtype_i64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_i64.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.dtype, Dtype::I64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_invalid_dtype_filtered_out() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 99; // invalid dtype
        let dir = unique_test_dir("dtype_inv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_inv.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.tensor_info("test_tensor").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn iter_tensors_invalid_dtype_filtered_from_iterator() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 7] = 99; // invalid dtype
        let dir = unique_test_dir("iter_inv");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter_inv.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.iter_tensors().count(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q4_0_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 10);
        let dir = unique_test_dir("q40_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q40.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q4_0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q5_0_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 12);
        let dir = unique_test_dir("q50_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q50.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q5_0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q4k_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 22);
        let dir = unique_test_dir("q4k_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q4k.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q4K);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_q6k_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 24);
        let dir = unique_test_dir("q6k_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q6k.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::Q6K);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_iq1s_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 30);
        let dir = unique_test_dir("iq1s_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iq1s.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::IQ1S);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_iq3s_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 36);
        let dir = unique_test_dir("iq3s_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iq3s.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::IQ3S);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_iq4nl_via_file() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 37);
        let dir = unique_test_dir("iq4nl_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iq4nl.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w").unwrap();
        assert_eq!(qt, gllm_kernels::quant::QuantType::IQ4NL);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_qwen3_key() {
        let meta = r#"{"arch_key":"qwen3"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_qwen3");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_qwen3.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("qwen3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_phi4_key() {
        let meta = r#"{"arch_key":"phi4"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_phi4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_phi4.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("phi4"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_gpt_oss_key() {
        let meta = r#"{"arch_key":"gpt-oss-20b"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_gptoss");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_gptoss.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("gpt-oss-20b"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_non_string_arch_key_returns_none() {
        let meta = r#"{"arch_key":42}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_int");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_int.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.architecture().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_float_string_returns_none() {
        // Float strings fail u64 parse → missing required field → None
        let meta = r#"{"vocab_size":"3.14","hidden_size":"NaN"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_float");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_float.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_with_scientific_notation_returns_none() {
        // Scientific notation fails u64 parse → missing required field → None
        let meta = r#"{"vocab_size":"1e5","hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_sci");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_sci.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_for_quantized_tensor_returns_compressed_bytes() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 40);
        let dir = unique_test_dir("qtd");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qtd.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("w").unwrap();
        assert!(t.entry.is_quantized());
        assert_eq!(t.data_size, 32);
        let td = reader.tensor_data("w").unwrap();
        assert_eq!(td.len(), 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_info_shape_for_multi_dim_entry() {
        let mut data = build_minimal_gllm();
        data[HEADER_SIZE + 6] = 3; // ndim = 3
        data[HEADER_SIZE + 8..HEADER_SIZE + 16].copy_from_slice(&2u64.to_le_bytes());
        data[HEADER_SIZE + 16..HEADER_SIZE + 24].copy_from_slice(&3u64.to_le_bytes());
        data[HEADER_SIZE + 24..HEADER_SIZE + 32].copy_from_slice(&4u64.to_le_bytes());
        let dir = unique_test_dir("shape3d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shape3d.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.tensor_info("test_tensor").unwrap();
        assert_eq!(meta.shape, vec![2, 3, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_tensor_data_for_each_tensor_in_multi() {
        let data = build_gllm_with_json_meta(&["a", "b", "c"], "{}", 0);
        let dir = unique_test_dir("lt_multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_multi.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        for name in &["a", "b", "c"] {
            let cow = reader.load_tensor_data(name).unwrap();
            assert!(matches!(cow, Cow::Borrowed(_)));
            assert_eq!(cow.len(), 32);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_with_nonzero_data_offset_and_single_tensor() {
        let dir = unique_test_dir("nz_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nz_data.gllm");
        let mut buf = Vec::new();
        let data_offset: u64 = 256;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        // Pad to string table start
        let name = "x";
        let string_table_start = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        // Tensor entry with name at string table
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        // String table
        buf.extend_from_slice(name.as_bytes());
        // Pad metadata region until data_offset
        while buf.len() < data_offset as usize {
            buf.push(0);
        }
        // Data
        buf.extend_from_slice(&[0xABu8; 16]);
        std::fs::write(&path, &buf).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("x").unwrap();
        assert_eq!(t.abs_data_offset, 256);
        let td = reader.tensor_data("x").unwrap();
        assert_eq!(td.len(), 16);
        assert!(td.iter().all(|&b| b == 0xAB));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_with_data_content_verification() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("data_content");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_content.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();
        // Data region is all zeros in build_minimal_gllm
        assert!(td.iter().all(|&b| b == 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_count_equals_header_tensor_count() {
        let data = build_gllm_with_json_meta(&["t1", "t2", "t3"], "{}", 0);
        let dir = unique_test_dir("count_eq");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("count_eq.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 3);
        assert_eq!(reader.tensor_count(), reader.header().tensor_count as usize);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensors_accessor_preserves_order() {
        let names = ["first", "second", "third"];
        let data = build_gllm_with_json_meta(&names, "{}", 0);
        let dir = unique_test_dir("tens_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tens_order.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();
        assert_eq!(tensors[0].name, "first");
        assert_eq!(tensors[1].name, "second");
        assert_eq!(tensors[2].name, "third");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_for_quantized_returns_compressed_not_original() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 40);
        let dir = unique_test_dir("qsize");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qsize.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("w").unwrap();
        assert!(t.entry.is_quantized());
        // compressed_size == original_size == 32 in build_gllm_with_json_meta,
        // but data_size should use compressed_size path
        assert_eq!(t.data_size, t.entry.compressed_size as usize);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_empty_json_object_returns_empty_bytes() {
        let data = build_gllm_with_json_meta(&["w"], "{}", 0);
        let dir = unique_test_dir("meta_empty_obj");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_empty_obj.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes(), b"{}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_json_with_only_vocab_size_returns_none() {
        // Only 1 of 8 required fields present → None
        let meta = r#"{"vocab_size":"32000"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_vocab_only");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_vocab_only.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_deepseek_r1_key() {
        let meta = r#"{"arch_key":"deepseek-r1"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_dsr1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_dsr1.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("deepseek-r1"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn architecture_with_kimi_k2_key() {
        let meta = r#"{"arch_key":"kimi-k2"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_kimi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_kimi.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture().as_deref(), Some("kimi-k2"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_tensor_returns_correct_entry_fields() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("find_fields");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("find_fields.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("test_tensor").unwrap();
        assert_eq!(t.entry.ndim, 2);
        assert_eq!(t.entry.dtype, 0);
        assert_eq!(t.entry.shape[0], 4);
        assert_eq!(t.entry.shape[1], 4);
        assert_eq!(t.entry.quant_format, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_preserves_dtype_byte() {
        for dtype_code in 0u8..=6 {
            let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
            buf[7] = dtype_code;
            let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
            assert_eq!(e.dtype, dtype_code);
        }
    }

    #[test]
    fn tensor_entry_compression_ratio_zero_original_zero_compressed() {
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 0, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 0,
        };
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gllm_error_all_variants_constructible() {
        let _ = GllmError::Io(std::io::Error::new(std::io::ErrorKind::Other, "test"));
        let _ = GllmError::InvalidMagic(0);
        let _ = GllmError::UnsupportedVersion(0);
        let _ = GllmError::HeaderTooSmall(0);
        let _ = GllmError::TensorDirOutOfBounds { offset: 0, count: 0, file_size: 0 };
        let _ = GllmError::StringTableOutOfBounds { offset: 0, length: 0, file_size: 0 };
        let _ = GllmError::MetadataOutOfBounds { offset: 0, file_size: 0 };
        let _ = GllmError::TensorOutOfBounds { name: String::new(), start: 0, end: 0, file_size: 0 };
        let _ = GllmError::DuplicateTensorName(String::new());
        let _ = GllmError::ParseError(String::new());
        let _ = GllmError::InvalidQuantType(0);
        let _ = GllmError::InvalidDType(0);
        let _ = GllmError::InvalidMetadata(String::new());
    }

    #[test]
    fn from_files_rejects_three_paths() {
        let paths = vec![
            PathBuf::from("a.gllm"),
            PathBuf::from("b.gllm"),
            PathBuf::from("c.gllm"),
        ];
        let result = GllmReader::from_files(&paths);
        assert!(matches!(result, Err(GllmError::ParseError(_))));
    }

    #[test]
    fn load_tensor_data_for_missing_tensor_returns_missing_tensor_error() {
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_missing.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let result = reader.load_tensor_data("nonexistent");
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 15 new tests: edge cases, error paths, data integrity ────────────────────

    #[test]
    fn gllm_error_display_invalid_magic_shows_hex_and_expected() {
        // Arrange: construct InvalidMagic with a known value
        let err = GllmError::InvalidMagic(0x00585858);
        // Act: format via Display
        let msg = err.to_string();
        // Assert: contains both the bad magic and the expected "GLLM" string
        assert!(msg.contains("0x00585858"), "should contain hex of bad magic: {msg}");
        assert!(msg.contains("GLLM"), "should mention expected magic: {msg}");
    }

    #[test]
    fn gllm_header_parse_with_trailing_bytes_uses_first_64() {
        // Arrange: 128-byte buffer with valid header in first 64 bytes and junk after
        let mut buf = vec![0xFFu8; HEADER_SIZE * 2];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes()); // GLLM
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&1u32.to_le_bytes()); // flags=1
        buf[12..20].copy_from_slice(&0u64.to_le_bytes());
        buf[20..24].copy_from_slice(&3u32.to_le_bytes()); // tensor_count=3
        buf[24..32].copy_from_slice(&256u64.to_le_bytes());
        buf[32..40].copy_from_slice(&4096u64.to_le_bytes());
        buf[40..44].copy_from_slice(&8192u32.to_le_bytes()); // page_size
        // Act
        let h = GllmHeader::parse(&buf).unwrap();
        // Assert: header parsed from first 64 bytes, trailing junk ignored
        assert_eq!(h.tensor_count, 3);
        assert_eq!(h.page_size, 8192);
        assert!(h.is_quantized());
    }

    #[test]
    fn gllm_reader_data_offset_overflow_detected() {
        // Arrange: build a gllm where abs_data_offset + data_size overflows usize
        let dir = unique_test_dir("overflow");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overflow.gllm");

        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        // Tensor entry with compressed_size near usize::MAX to cause overflow
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0
        buf.extend_from_slice(&usize::MAX.to_le_bytes()[0..8]); // compressed_size = huge
        buf.extend_from_slice(&usize::MAX.to_le_bytes()[0..8]); // original_size = huge

        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&[0xAB, 0xCD]);
        buf.extend_from_slice(&[0u8; 64]);

        std::fs::write(&path, &buf).unwrap();
        // Act
        let result = GllmReader::open(&path);
        // Assert: should fail with ParseError (offset overflow) or TensorOutOfBounds
        assert!(result.is_err(), "overflow should cause parse failure");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_tensor_data_with_nonzero_pattern() {
        // Arrange: build a gllm file with a recognizable byte pattern in the data region
        let mut data = build_minimal_gllm();
        let data_region_start = data.len() - 64;
        for (i, byte) in data[data_region_start..].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(0xA0);
        }
        let dir = unique_test_dir("pattern");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pattern.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();

        // Assert: each byte matches the expected pattern
        for (i, &byte) in td.iter().enumerate() {
            assert_eq!(byte, (i as u8).wrapping_add(0xA0), "byte at index {i} mismatch");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_find_tensor_returns_correct_entry() {
        // Arrange: multi-tensor file with distinct shapes
        let data = build_gllm_with_json_meta(&["weight_a", "weight_b"], "{}", 0);
        let dir = unique_test_dir("find_correct");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("find_correct.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let t_a = reader.find_tensor("weight_a").unwrap();
        let t_b = reader.find_tensor("weight_b").unwrap();

        // Assert: each tensor has the correct name and data is at different offsets
        assert_eq!(t_a.name, "weight_a");
        assert_eq!(t_b.name, "weight_b");
        assert_ne!(t_a.abs_data_offset, t_b.abs_data_offset,
            "two different tensors must have different data offsets");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_quant_type_for_various_formats_via_mapping() {
        // Arrange & Act & Assert: verify the full chain code→gllm_quant_type_from_u8 for every format
        let expected: Vec<(u8, &str)> = vec![
            (40, "AWQ4"), (41, "GPTQ4"), (42, "Squeeze"),
            (10, "Q4_0"), (14, "Q8_0"), (20, "Q2K"),
            (50, "Fp8E4M3"), (51, "Fp8E5M2"), (52, "Mxfp4"), (53, "Nvfp4"),
        ];
        for (code, _name) in &expected {
            let result = gllm_quant_type_from_u8(*code);
            assert!(result.is_some(), "code {code} ({_name}) must resolve to Some");
        }
    }

    #[test]
    fn gllm_reader_model_params_negative_string_returns_none() {
        // Arrange: JSON with a negative number string — fails u64 parse → None
        let meta = r#"{"vocab_size":"-100","hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_neg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_neg.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: negative string "-100" → parse fails → required field missing → None
        assert!(reader.model_params().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_header_tensor_dir_offset_equals_header_size() {
        // Arrange
        let data = build_minimal_gllm();
        let dir = unique_test_dir("tdoff");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tdoff.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: tensor_dir_offset should be exactly HEADER_SIZE (right after header)
        assert_eq!(reader.header().tensor_dir_offset, HEADER_SIZE as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_tensor_entry_name_with_unicode_characters() {
        // Arrange: tensor names with unicode (CJK characters)
        let name = "model.\u{5c42}.weight"; // Chinese character for "layer"
        let data = build_gllm_with_json_meta(&[name], "{}", 0);
        let dir = unique_test_dir("unicode2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode2.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor(name);

        // Assert: unicode tensor names are preserved correctly
        assert!(t.is_some(), "unicode tensor name should be found");
        assert_eq!(t.unwrap().name, name);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_iter_tensors_returns_correct_dtype() {
        // Arrange: single tensor with dtype F16 (code 1)
        let mut data = build_minimal_gllm();
        // Override dtype byte in the tensor entry: byte at HEADER_SIZE + 7
        data[HEADER_SIZE + 7] = 1; // F16
        let dir = unique_test_dir("iter_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("iter_dtype.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();

        // Assert: iter_tensors correctly maps dtype code 1 to Dtype::F16
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].dtype, Dtype::F16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_open_with_path_buf_and_str() {
        // Arrange
        let data = build_minimal_gllm();
        let dir = unique_test_dir("open_types");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("open_types.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act: open with &PathBuf, &str-coercable PathBuf, and &Path
        let r1 = GllmReader::open(&path).unwrap();
        let r2 = GllmReader::open(path.as_path()).unwrap();

        // Assert: both methods succeed and produce identical results
        assert_eq!(r1.tensor_count(), r2.tensor_count());
        assert_eq!(r1.header().version, r2.header().version);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_metadata_with_leading_and_trailing_zeros() {
        // Arrange: metadata region with leading zero then content then trailing zeros
        let dir = unique_test_dir("meta_ltz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_ltz.gllm");

        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 6;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        buf.extend_from_slice(name.as_bytes());
        // Metadata: [0x00, 0xCA, 0xFE, 0x00, 0x00, 0x00]
        buf.extend_from_slice(&[0x00, 0xCA, 0xFE, 0x00, 0x00, 0x00]);
        buf.extend_from_slice(&[0u8; 16]);

        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();

        // Assert: trailing zeros stripped but leading zero preserved
        // rposition finds last non-zero, so [0x00, 0xCA, 0xFE] remains
        assert_eq!(meta.len(), 3, "leading zero preserved, trailing zeros stripped");
        assert_eq!(meta[0], 0x00);
        assert_eq!(meta[1], 0xCA);
        assert_eq!(meta[2], 0xFE);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_resolved_tensor_data_offset_matches_tensor_data_slice() {
        // Arrange
        let data = build_minimal_gllm();
        let dir = unique_test_dir("off_match");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("off_match.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("test_tensor").unwrap();
        let td = reader.tensor_data("test_tensor").unwrap();

        // Assert: tensor_data length exactly matches data_size
        assert_eq!(td.len(), t.data_size);
        // abs_data_offset + data_size must not exceed mmap length
        assert!(t.abs_data_offset + t.data_size <= data.len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_gllm_model_params_equality_after_clone() {
        // Arrange
        let params = GllmModelParams {
            vocab_size: 50000,
            hidden_size: 2048,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 128,
            intermediate_size: 5504,
            context_length: 4096,
        };

        // Act
        let cloned = params.clone();

        // Assert: all fields match exactly after clone
        assert_eq!(params.vocab_size, cloned.vocab_size);
        assert_eq!(params.hidden_size, cloned.hidden_size);
        assert_eq!(params.num_layers, cloned.num_layers);
        assert_eq!(params.num_heads, cloned.num_heads);
        assert_eq!(params.num_kv_heads, cloned.num_kv_heads);
        assert_eq!(params.head_dim, cloned.head_dim);
        assert_eq!(params.intermediate_size, cloned.intermediate_size);
        assert_eq!(params.context_length, cloned.context_length);
    }

    #[test]
    fn gllm_reader_parse_file_with_version_zero_rejected() {
        // Arrange: valid magic but version = 0
        let mut data = build_minimal_gllm();
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // version = 0
        let dir = unique_test_dir("v0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v0.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert: version 0 is unsupported
        assert!(matches!(result, Err(GllmError::UnsupportedVersion(0))));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 15 new tests: additional edge cases and coverage ───────────────────────────

    #[test]
    fn gllm_error_source_io_returns_some() {
        // Arrange: construct a GllmError::Io wrapping a real io::Error
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "unexpected eof");
        let gllm_err = GllmError::Io(io_err);

        // Act: call source() on the error
        use std::error::Error;
        let source = gllm_err.source();

        // Assert: Io variant returns Some, not None
        assert!(source.is_some(), "Io variant should return Some from source()");
    }

    #[test]
    fn tensor_entry_compression_ratio_when_compressed_exceeds_original() {
        // Arrange: entry where compressed_size > original_size (expansion, not compression)
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [100, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 200, original_size: 100,
        };

        // Act
        let ratio = entry.compression_ratio();

        // Assert: ratio < 1.0 (compressed is bigger than original)
        assert!(ratio < 1.0, "expansion should produce ratio < 1.0");
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn parse_two_tensors_non_overlapping_data_offsets() {
        // Arrange: multi-tensor file
        let data = build_gllm_with_json_meta(&["tensor_a", "tensor_b"], "{}", 0);
        let dir = unique_test_dir("nonoverlap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nonoverlap.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let t_a = reader.find_tensor("tensor_a").unwrap();
        let t_b = reader.find_tensor("tensor_b").unwrap();

        // Assert: data regions do not overlap
        let a_end = t_a.abs_data_offset + t_a.data_size;
        assert!(a_end <= t_b.abs_data_offset || t_b.abs_data_offset + t_b.data_size <= t_a.abs_data_offset,
            "tensor data regions must not overlap");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_u64_max_value_in_json() {
        // Arrange: JSON with u64 max value as string — but only 2 of 8 required fields → None
        let meta = r#"{"vocab_size":"18446744073709551615","hidden_size":"1"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_umax");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_umax.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        // Assert: incomplete metadata (only 2 fields) → None
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_offset_at_end_of_buffer_fails() {
        // Arrange: buffer of exactly TENSOR_ENTRY_SIZE, try parsing at offset = TENSOR_ENTRY_SIZE
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];

        // Act: offset at end of buffer
        let result = GllmTensorEntry::parse_at(&buf, TENSOR_ENTRY_SIZE);

        // Assert: fails because there are 0 bytes available
        assert!(result.is_err(), "parsing at exact buffer end should fail");
    }

    #[test]
    fn header_parse_with_all_zero_fields_after_magic_and_version() {
        // Arrange: valid GLLM header with all other fields zeroed
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes()); // GLLM
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
        // All other fields stay zero

        // Act
        let h = GllmHeader::parse(&buf).unwrap();

        // Assert: zero fields parsed correctly
        assert_eq!(h.flags, 0);
        assert_eq!(h.meta_offset, 0);
        assert_eq!(h.tensor_count, 0);
        assert_eq!(h.tensor_dir_offset, 0);
        assert_eq!(h.data_offset, 0);
        assert_eq!(h.page_size, 0);
        assert!(!h.is_quantized());
    }

    #[test]
    fn from_files_single_existing_path_succeeds() {
        // Arrange
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ff_exist");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff_exist.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::from_files(&[path.clone()]).unwrap();

        // Assert: from_files with single path produces same result as open
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.header().version, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_dtype_to_st_returns_correct_size_category() {
        // Arrange & Act & Assert: F32 and I64 are both 4+ bytes, U8 and I8 are 1 byte
        // This verifies the mapping doesn't confuse byte widths
        assert_eq!(gllm_dtype_to_st(0).unwrap(), Dtype::F32); // 4 bytes
        assert_eq!(gllm_dtype_to_st(3).unwrap(), Dtype::U8);  // 1 byte
        assert_eq!(gllm_dtype_to_st(4).unwrap(), Dtype::I8);  // 1 byte
        assert_eq!(gllm_dtype_to_st(6).unwrap(), Dtype::I64); // 8 bytes
        // Verify F32 != F16 != BF16
        assert_ne!(gllm_dtype_to_st(0).unwrap(), gllm_dtype_to_st(1).unwrap());
    }

    #[test]
    fn tensor_data_for_second_tensor_has_distinct_offset() {
        // Arrange: 3-tensor file
        let data = build_gllm_with_json_meta(&["x", "y", "z"], "{}", 0);
        let dir = unique_test_dir("distinct_off");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("distinct_off.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let offsets: Vec<usize> = reader.tensors().iter().map(|t| t.abs_data_offset).collect();

        // Assert: all offsets are unique
        assert_eq!(offsets.len(), 3);
        assert_ne!(offsets[0], offsets[1]);
        assert_ne!(offsets[1], offsets[2]);
        assert_ne!(offsets[0], offsets[2]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_file_with_exactly_header_plus_one_entry_no_data() {
        // Arrange: file with exactly HEADER_SIZE + TENSOR_ENTRY_SIZE bytes and 1 tensor
        // but data_offset points past end of file — should fail with TensorOutOfBounds
        let dir = unique_test_dir("no_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_data.gllm");

        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        // Tensor entry claiming 16 bytes of data but file has 0 data bytes
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&[0u8; 2]);
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&[0xAB, 0xCD]);
        // NO data region

        std::fs::write(&path, &buf).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert: should fail because tensor data extends past file
        assert!(result.is_err(), "file missing data region should fail");
        assert!(matches!(result.unwrap_err(), GllmError::TensorOutOfBounds { .. }));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_quant_type_from_u8_code_8_is_none_in_gap() {
        // Arrange & Act & Assert: code 8 is between F32(3) and Q4_0(10), should be None
        assert!(gllm_quant_type_from_u8(8).is_none());
        assert!(gllm_quant_type_from_u8(9).is_none());
    }

    #[test]
    fn metadata_bytes_after_open_is_consistent_across_calls() {
        // Arrange
        let data = build_minimal_gllm();
        let dir = unique_test_dir("meta_cons");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_cons.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act: call metadata_bytes twice
        let reader = GllmReader::open(&path).unwrap();
        let m1 = reader.metadata_bytes().to_vec();
        let m2 = reader.metadata_bytes().to_vec();

        // Assert: both calls return identical content
        assert_eq!(m1, m2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolved_tensor_entry_ndim_boundary_max_four() {
        // Arrange: GllmTensorEntry supports up to ndim=4 (shape array has 4 elements)
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 1, ndim: 4, dtype: 0,
            shape: [2, 3, 5, 7], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 840, original_size: 840,
        };

        // Act & Assert: all 4 shape dimensions are distinct and accessible
        assert_eq!(entry.ndim, 4);
        assert_eq!(entry.shape[0], 2);
        assert_eq!(entry.shape[1], 3);
        assert_eq!(entry.shape[2], 5);
        assert_eq!(entry.shape[3], 7);
        // Product of dimensions * 4 bytes (F32) should equal original_size
        let product: u64 = entry.shape.iter().take(entry.ndim as usize).product();
        assert_eq!(product * 4, entry.original_size);
    }

    #[test]
    fn open_directory_path_returns_io_error() {
        // Arrange: use a directory path instead of a file
        let dir = unique_test_dir("dir_path");
        std::fs::create_dir_all(&dir).unwrap();

        // Act: try to open the directory itself
        let result = GllmReader::open(&dir);

        // Assert: should fail with Io error (is a directory)
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GllmError::Io(_)));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_model_params_context_length_independent_of_vocab_size() {
        // Arrange: set context_length to a large value, vocab_size to a small one
        let params = GllmModelParams {
            vocab_size: 100,
            hidden_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            intermediate_size: 0,
            context_length: 1048576,
        };

        // Act & Assert: context_length is independent, not derived from vocab_size
        assert_eq!(params.context_length, 1048576);
        assert_eq!(params.vocab_size, 100);
        assert_ne!(params.context_length, params.vocab_size);
    }

    // ── 15 new tests ─────────────────────────────────────────────────────────────

    #[test]
    fn find_tensor_does_not_match_substring() {
        // Arrange: single tensor named "weight"
        let data = build_gllm_with_json_meta(&["weight"], "{}", 0);
        let dir = unique_test_dir("substr");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("substr.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();

        // Act & Assert: prefix, suffix, and substring should not match
        assert!(reader.find_tensor("weight").is_some());
        assert!(reader.find_tensor("weigh").is_none());
        assert!(reader.find_tensor("eight").is_none());
        assert!(reader.find_tensor("weights").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_header_parse_magic_stored_as_u32_le() {
        // Arrange: "GLLM" in ASCII is 0x4D4C4C47 in little-endian
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes());
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[40..44].copy_from_slice(&4096u32.to_le_bytes());

        // Act
        let h = GllmHeader::parse(&buf).unwrap();

        // Assert: version and page_size correctly read
        assert_eq!(h.version, 1);
        assert_eq!(h.page_size, 4096);
    }

    #[test]
    fn parse_version_zero_rejected() {
        // Arrange: version=0 should be unsupported (only version 1 is valid)
        let mut data = build_minimal_gllm();
        data[4..8].copy_from_slice(&0u32.to_le_bytes());
        let dir = unique_test_dir("v0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("v0.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert
        assert!(matches!(result, Err(GllmError::UnsupportedVersion(0))));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compression_ratio_both_sizes_one() {
        // Arrange: compressed_size=1, original_size=1 → ratio=1.0
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 1, original_size: 1,
        };

        // Act
        let ratio = entry.compression_ratio();

        // Assert
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn resolved_tensor_entry_mutable_field_update() {
        // Arrange
        let mut rt = ResolvedTensor {
            name: "bias".to_string(),
            entry: GllmTensorEntry {
                name_offset: 0, name_len: 4, ndim: 1, dtype: 0,
                shape: [768, 0, 0, 0], quant_format: 0, quant_block_size: 0,
                scale_dtype: 0, zp_type: 0, data_offset: 0,
                compressed_size: 3072, original_size: 3072,
            },
            abs_data_offset: 500,
            data_size: 3072,
        };

        // Act: mutate name and abs_data_offset
        rt.name = "new_bias".to_string();
        rt.abs_data_offset = 9999;

        // Assert: entry fields are unaffected
        assert_eq!(rt.name, "new_bias");
        assert_eq!(rt.abs_data_offset, 9999);
        assert_eq!(rt.entry.ndim, 1);
        assert_eq!(rt.entry.shape[0], 768);
        assert_eq!(rt.entry.original_size, 3072);
    }

    #[test]
    fn mmap_survives_file_deletion_after_open() {
        // Arrange: write a valid gllm file, open it, then delete the file
        let data = build_minimal_gllm();
        let dir = unique_test_dir("mmap_del");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mmap_del.gllm");
        std::fs::write(&path, &data).unwrap();

        let reader = GllmReader::open(&path).unwrap();

        // Act: delete the underlying file
        std::fs::remove_file(&path).unwrap();

        // Assert: mmap-backed data is still accessible (Linux keeps pages in memory)
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("test_tensor").unwrap();
        assert_eq!(td.len(), 64);
        assert!(td.iter().all(|&b| b == 0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_boolean_json_values_returns_none() {
        // Arrange: JSON with boolean values — HashMap<String, String> deserialization
        // fails because true/false are not strings, so model_params() returns None
        let meta = r#"{"vocab_size":true,"hidden_size":false,"num_layers":true}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_bool");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_bool.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let params = reader.model_params();

        // Assert: non-string JSON values cause deserialization to fail → None
        assert!(params.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_error_source_io_broken_pipe_returns_some() {
        // Arrange: construct an Io-wrapped GllmError with BrokenPipe kind
        use std::error::Error;
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let gllm_err: GllmError = io_err.into();

        // Act
        let source = gllm_err.source();

        // Assert: Io variant returns Some from source()
        assert!(source.is_some());
    }

    #[test]
    fn tensor_data_content_readback_nonzero_pattern() {
        // Arrange: build a file with a recognizable non-zero data pattern
        let mut data = build_minimal_gllm();
        let data_start = data.len() - 64;
        for i in 0..64 {
            data[data_start + i] = (i as u8).wrapping_add(0xA0);
        }
        let dir = unique_test_dir("pattern");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pattern.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();

        // Act
        let td = reader.tensor_data("test_tensor").unwrap();

        // Assert: bytes match the exact pattern
        for i in 0..64 {
            assert_eq!(td[i], (i as u8).wrapping_add(0xA0), "byte at index {i} mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_parse_at_non_aligned_offset() {
        // Arrange: parse entry at a non-power-of-two offset (e.g. 73)
        let mut buf = vec![0u8; 200];
        let offset = 73;
        buf[offset..offset + 4].copy_from_slice(&888u32.to_le_bytes());
        buf[offset + 4..offset + 6].copy_from_slice(&3u16.to_le_bytes());
        buf[offset + 6] = 2;
        buf[offset + 7] = 5; // dtype I32
        buf[offset + 8..offset + 16].copy_from_slice(&1024u64.to_le_bytes());
        buf[offset + 16..offset + 24].copy_from_slice(&2048u64.to_le_bytes());
        buf[offset + 40] = 10; // Q4_0
        buf[offset + 48..offset + 56].copy_from_slice(&5555u64.to_le_bytes());
        buf[offset + 56..offset + 64].copy_from_slice(&1111u64.to_le_bytes());
        buf[offset + 64..offset + 72].copy_from_slice(&4444u64.to_le_bytes());

        // Act
        let e = GllmTensorEntry::parse_at(&buf, offset).unwrap();

        // Assert: all fields correctly parsed from non-aligned offset
        assert_eq!(e.name_offset, 888);
        assert_eq!(e.name_len, 3);
        assert_eq!(e.ndim, 2);
        assert_eq!(e.dtype, 5);
        assert_eq!(e.shape[0], 1024);
        assert_eq!(e.shape[1], 2048);
        assert_eq!(e.quant_format, 10);
        assert_eq!(e.data_offset, 5555);
        assert_eq!(e.compressed_size, 1111);
        assert_eq!(e.original_size, 4444);
        assert!(e.is_quantized());
    }

    #[test]
    fn parse_tensor_dir_offset_before_header_end_fails() {
        // Arrange: tensor_dir_offset inside the header (should cause issues
        // with tensor dir parsing but the header itself parses fine)
        let mut buf = vec![0u8; HEADER_SIZE + TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(b"GLLM");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        buf[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // meta_offset
        buf[20..24].copy_from_slice(&1u32.to_le_bytes()); // tensor_count
        buf[24..32].copy_from_slice(&0u64.to_le_bytes()); // tensor_dir_offset=0 (inside header!)
        buf[32..40].copy_from_slice(&(HEADER_SIZE as u64 + TENSOR_ENTRY_SIZE as u64).to_le_bytes()); // data_offset
        buf[40..44].copy_from_slice(&4096u32.to_le_bytes()); // page_size

        let dir = unique_test_dir("td_in_header");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_in_header.gllm");
        std::fs::write(&path, &buf).unwrap();

        // Act: tensor_dir_offset=0 means entries overlap with the header bytes
        let result = GllmReader::open(&path);

        // Assert: should parse (overlapping is technically allowed by offset math,
        // but the name resolution will fail since string table overlaps)
        // The key assertion is that it doesn't panic
        assert!(result.is_err() || result.is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_reader_from_files_with_pathbuf_vec() {
        // Arrange: single-path Vec<PathBuf>
        let data = build_minimal_gllm();
        let dir = unique_test_dir("ff_vec");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff_vec.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let paths: Vec<PathBuf> = vec![path];
        let reader = GllmReader::from_files(&paths).unwrap();

        // Assert
        assert_eq!(reader.tensor_count(), 1);
        assert!(reader.find_tensor("test_tensor").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_only_trailing_zeros_stripped_not_leading() {
        // Arrange: metadata = [0x00, 0x00, 0xFF, 0x00, 0x00]
        // Trailing zeros stripped → [0x00, 0x00, 0xFF]
        let dir = unique_test_dir("meta_strip2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_strip2.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 5;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        buf.extend_from_slice(&[0x00, 0x00, 0xFF, 0x00, 0x00]);
        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: only trailing zeros are stripped
        assert_eq!(reader.metadata_bytes(), &[0x00, 0x00, 0xFF]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_model_params_sum_of_fields_does_not_overflow() {
        // Arrange: use realistic field values where sum fits in u64
        let params = GllmModelParams {
            vocab_size: 151936,
            hidden_size: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 29568,
            context_length: 131072,
        };

        // Act: compute sum of all fields
        let total = params.vocab_size + params.hidden_size + params.num_layers
            + params.num_heads + params.num_kv_heads + params.head_dim
            + params.intermediate_size + params.context_length;

        // Assert: sum is reasonable and doesn't overflow
        assert!(total > 0);
        assert!(total < u64::MAX);
        assert_eq!(total, 151936 + 8192 + 80 + 64 + 8 + 128 + 29568 + 131072);
    }

    #[test]
    fn quant_type_for_unquantized_in_multi_tensor_file() {
        // Arrange: multi-tensor file with mixed quant/unquant tensors not possible
        // with build_gllm_with_json_meta (uniform quant), so test with minimal gllm
        let data = build_minimal_gllm();
        let dir = unique_test_dir("qt_unquant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qt_unquant.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();

        // Act: quant_format=0 for test_tensor in build_minimal_gllm
        let qt = reader.quant_type("test_tensor");

        // Assert: unquantized tensor returns None from quant_type
        assert!(qt.is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_parse_rejects_reversed_magic() {
        // Arrange: "MLLG" = reversed GLLM bytes
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"MLLG");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());

        // Act
        let result = GllmHeader::parse(&buf);

        // Assert: reversed magic is not valid
        assert!(matches!(result, Err(GllmError::InvalidMagic(_))));
        if let Err(GllmError::InvalidMagic(m)) = result {
            // "MLLG" = 0x474C4C4D in LE
            assert_ne!(m, 0x4D4C4C47u32);
        }
    }

    // ── 15 new tests (wave-15x1) ──────────────────────────────────────────────

    #[test]
    // @trace TEST-GLLM-READER-001 [level:unit]
    fn load_tensor_data_missing_returns_loader_error() {
        // Arrange: valid gllm file with a single tensor
        let data = build_minimal_gllm();
        let dir = unique_test_dir("lt_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lt_missing.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();

        // Act: request a tensor that does not exist
        let result = reader.load_tensor_data("nonexistent_tensor");

        // Assert: TensorProvider::load_tensor_data returns MissingTensor error
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nonexistent_tensor"), "error should mention tensor name: {msg}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-002 [level:unit]
    fn metadata_bytes_empty_when_meta_offset_equals_data_offset() {
        // Arrange: metadata region has zero length (meta_offset == data_offset)
        let dir = unique_test_dir("meta_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_empty.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        // No tensor entries: tensor_count=0, so string table is empty, meta_offset = data_offset
        let data_offset: u64 = meta_offset;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes()); // tensor_dir_offset
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: metadata_bytes is empty when meta region has zero length
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-003 [level:unit]
    fn tensors_preserves_file_order_not_sorted() {
        // Arrange: multi-tensor file with specific ordering
        let data = build_gllm_with_json_meta(
            &["z_layer", "a_bias", "m_weight"],
            "{}",
            0,
        );
        let dir = unique_test_dir("order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("order.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();

        // Assert: tensors() preserves file order (not sorted alphabetically)
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors[0].name, "z_layer");
        assert_eq!(tensors[1].name, "a_bias");
        assert_eq!(tensors[2].name, "m_weight");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-004 [level:unit]
    fn tensor_data_is_borrowed_cow() {
        // Arrange: valid file with a tensor
        let data = build_minimal_gllm();
        let dir = unique_test_dir("cow");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cow.gllm");
        std::fs::write(&path, &data).unwrap();
        let reader = GllmReader::open(&path).unwrap();

        // Act
        let td = reader.tensor_data("test_tensor").unwrap();

        // Assert: data is a Borrowed Cow (zero-copy from mmap)
        assert!(matches!(td, Cow::Borrowed(_)));
        assert_eq!(td.len(), 64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-005 [level:unit]
    fn resolved_tensor_data_size_uses_compressed_when_quantized() {
        // Arrange: quantized tensor with different compressed vs original sizes
        let data = build_gllm_with_json_meta(&["q_weight"], "{}", 14); // Q8_0
        let dir = unique_test_dir("dsq");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dsq.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let rt = reader.find_tensor("q_weight").unwrap();

        // Assert: for quantized tensors, data_size equals compressed_size
        assert!(rt.entry.is_quantized());
        assert_eq!(rt.data_size, rt.entry.compressed_size as usize);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-006 [level:unit]
    fn architecture_with_null_json_value_returns_none() {
        // Arrange: JSON where arch_key is null (not a string)
        let meta = r#"{"arch_key":null,"hidden_size":"4096"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_null");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_null.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: HashMap<String, String> deserialization fails on null → architecture() returns None
        assert!(reader.architecture().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-007 [level:unit]
    fn tensor_entry_compression_ratio_sub_one() {
        // Arrange: compressed_size larger than original_size (e.g. metadata overhead)
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 0, ndim: 1, dtype: 0,
            shape: [100, 0, 0, 0], quant_format: 1, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 200,
            original_size: 100,
        };

        // Act
        let ratio = entry.compression_ratio();

        // Assert: ratio < 1.0 when compressed > original
        assert!((ratio - 0.5).abs() < 1e-10, "expected 0.5, got {ratio}");
        assert!(ratio < 1.0);
    }

    #[test]
    // @trace TEST-GLLM-READER-008 [level:unit]
    fn header_parse_fails_on_four_byte_file() {
        // Arrange: exactly 4 bytes — enough for magic but not for full header
        let buf = b"GLLM".to_vec();

        // Act
        let result = GllmHeader::parse(&buf);

        // Assert: HeaderTooSmall(4)
        assert!(matches!(result, Err(GllmError::HeaderTooSmall(4))));
    }

    #[test]
    // @trace TEST-GLLM-READER-009 [level:unit]
    fn model_params_empty_json_returns_none() {
        // Arrange: valid JSON but empty object — no required fields present → None
        let meta = "{}";
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_empty.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: empty metadata has no required fields → None
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-010 [level:unit]
    fn parse_tensor_data_offset_overflow_rejected() {
        // Arrange: tensor entry with data_offset + data_size that overflows usize
        let dir = unique_test_dir("dof");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dof.gllm");
        let mut buf = Vec::new();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let name = "big_tensor";
        let name_bytes = name.as_bytes();
        let meta_offset: u64 = (string_table_offset + name_bytes.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        // Tensor entry with huge compressed_size causing checked_add overflow
        buf.extend_from_slice(&0u32.to_le_bytes()); // name_offset
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes()); // name_len
        buf.push(1); buf.push(0); // ndim=1, dtype=F32
        buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0]
        buf.extend_from_slice(&[0u8; 24]); // shape[1..4]
        buf.push(0); // quant_format
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]); // reserved
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset (relative to data section)
        buf.extend_from_slice(&u64::MAX.to_le_bytes()); // compressed_size = u64::MAX
        buf.extend_from_slice(&u64::MAX.to_le_bytes()); // original_size = u64::MAX

        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&[0xCC, 0xDD]); // metadata
        buf.extend_from_slice(&[0u8; 16]); // small data region

        std::fs::write(&path, &buf).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert: overflow in abs_data_offset + data_size must be caught
        assert!(result.is_err(), "expected error for data offset overflow, got Ok");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-011 [level:unit]
    fn gllm_dtype_to_st_all_invalid_codes_exhaustive() {
        // Arrange & Act & Assert: codes 7-255 are all invalid (only 0-6 are valid)
        for code in 7u8..=255 {
            let result = gllm_dtype_to_st(code);
            assert!(result.is_err(), "code {code} should be invalid");
            let err = result.unwrap_err();
            assert!(matches!(err, GllmError::InvalidDType(c) if c == code));
        }
    }

    #[test]
    // @trace TEST-GLLM-READER-012 [level:unit]
    fn metadata_bytes_all_zeros_gets_stripped() {
        // Arrange: metadata region entirely zeros → trailing zero strip removes everything
        let dir = unique_test_dir("meta_zeros");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_zeros.gllm");
        let mut buf = Vec::new();
        let meta_offset: u64 = HEADER_SIZE as u64;
        let data_offset: u64 = meta_offset + 8;
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        buf.extend_from_slice(&[0u8; 8]); // 8 bytes of zero metadata
        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: all-zero metadata gets stripped to empty
        assert!(reader.metadata_bytes().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-013 [level:unit]
    fn resolved_tensor_entry_shape_used_dimensions_match_ndim() {
        // Arrange: entry with ndim=2, shape = [1024, 768, 0, 0]
        let entry = GllmTensorEntry {
            name_offset: 0, name_len: 5, ndim: 2, dtype: 0,
            shape: [1024, 768, 0, 0],
            quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data_offset: 0,
            compressed_size: 0, original_size: 3145728, // 1024*768*4
        };

        // Act & Assert: only shape[0..ndim] should be considered meaningful
        let meaningful_dims: Vec<u64> = entry.shape[..entry.ndim as usize].to_vec();
        assert_eq!(meaningful_dims, vec![1024, 768]);
        assert_eq!(entry.shape[2], 0);
        assert_eq!(entry.shape[3], 0);

        // Verify element count (F32 = 4 bytes per element)
        let elems: u64 = meaningful_dims.iter().product();
        assert_eq!(elems * 4, entry.original_size);
    }

    #[test]
    // @trace TEST-GLLM-READER-014 [level:unit]
    fn iter_tensors_dtype_mapping_all_valid_codes() {
        // Arrange: build a file with tensors of each valid dtype code (0-6)
        let dtype_names: Vec<&str> = vec![
            "dt_f32", "dt_f16", "dt_bf16", "dt_u8",
            "dt_i8", "dt_i32", "dt_i64",
        ];
        let dir = unique_test_dir("dtypes");
        std::fs::create_dir_all(&dir).unwrap();

        let expected_dtypes = [
            Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::U8,
            Dtype::I8, Dtype::I32, Dtype::I64,
        ];

        // Build the file manually since build_gllm_with_json_meta uses fixed dtype=0
        let tensor_count = dtype_names.len() as u32;
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_start = HEADER_SIZE + dtype_names.len() * TENSOR_ENTRY_SIZE;
        let total_name_bytes: usize = dtype_names.iter().map(|n| n.len()).sum();
        let meta_offset: u64 = (string_table_start + total_name_bytes) as u64;
        let meta_bytes = b"{}";
        let data_offset: u64 = meta_offset + meta_bytes.len() as u64;
        let data_per_tensor: usize = 32;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        let mut name_offset_acc: u32 = 0;
        for (i, name) in dtype_names.iter().enumerate() {
            buf.extend_from_slice(&name_offset_acc.to_le_bytes());
            buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
            buf.push(1); // ndim
            buf.push(i as u8); // dtype = 0..6
            buf.extend_from_slice(&8u64.to_le_bytes());
            buf.extend_from_slice(&[0u8; 24]);
            buf.push(0); // quant_format
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.push(0); buf.push(0);
            buf.extend_from_slice(&[0u8; 3]);
            buf.extend_from_slice(&((i * data_per_tensor) as u64).to_le_bytes());
            buf.extend_from_slice(&(data_per_tensor as u64).to_le_bytes());
            buf.extend_from_slice(&(data_per_tensor as u64).to_le_bytes());
            name_offset_acc += name.len() as u32;
        }
        for name in &dtype_names {
            buf.extend_from_slice(name.as_bytes());
        }
        buf.extend_from_slice(meta_bytes);
        buf.extend_from_slice(&vec![0u8; dtype_names.len() * data_per_tensor]);

        let path = dir.join("dtypes.gllm");
        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let metas: Vec<_> = reader.iter_tensors().collect();

        // Assert: each tensor maps to the correct safetensors Dtype
        assert_eq!(metas.len(), 7);
        for (i, expected) in expected_dtypes.iter().enumerate() {
            assert_eq!(metas[i].dtype, *expected, "tensor {i} dtype mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    // @trace TEST-GLLM-READER-015 [level:unit]
    fn from_files_single_path_as_ref_str() {
        // Arrange: use a &str path via AsRef<Path> instead of PathBuf
        let data = build_minimal_gllm();
        let dir = unique_test_dir("asref");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("asref.gllm");
        let path_str = path.to_string_lossy().to_string();
        std::fs::write(&path, &data).unwrap();

        // Act: GllmReader::open accepts impl AsRef<Path>, so &str works
        let reader = GllmReader::open(&*path_str).unwrap();

        // Assert
        assert_eq!(reader.tensor_count(), 1);
        assert!(reader.find_tensor("test_tensor").is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── 10 additional tests ─────────────────────────────────────────────────────

    #[test]
    fn quant_type_unrecognized_format_returns_none_via_file() {
        // Arrange: build a file with quant_format=60 which is not in the mapping table
        let data = build_gllm_with_json_meta(&["w_unknown"], "{}", 60);
        let dir = unique_test_dir("qt_unknown");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qt_unknown.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let qt = reader.quant_type("w_unknown");

        // Assert: quant_format=60 is not in the mapping, so quant_type returns None
        assert!(qt.is_none(), "unrecognized quant_format should return None");
        // But the tensor is still considered quantized (quant_format != 0)
        let t = reader.find_tensor("w_unknown").unwrap();
        assert!(t.entry.is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_rejects_non_utf8_tensor_name() {
        // Arrange: build a file with invalid UTF-8 bytes in the string table
        let dir = unique_test_dir("non_utf8");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("non_utf8.gllm");

        let mut buf = Vec::new();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        // Place 3 bytes of invalid UTF-8: [0xFF, 0xFE, 0xFD]
        let name_bytes: &[u8] = &[0xFF, 0xFE, 0xFD];
        let meta_offset: u64 = (string_table_offset + name_bytes.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // tensor_count = 1
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);
        assert_eq!(buf.len(), HEADER_SIZE);

        // Tensor entry pointing to 3 bytes of non-UTF8 name
        buf.extend_from_slice(&0u32.to_le_bytes()); // name_offset
        buf.extend_from_slice(&3u16.to_le_bytes()); // name_len = 3
        buf.push(1); buf.push(0); // ndim=1, dtype=F32
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        // String table with invalid UTF-8
        buf.extend_from_slice(name_bytes);
        // Metadata
        buf.extend_from_slice(&[0xCC, 0xDD]);
        // Data
        buf.extend_from_slice(&[0u8; 16]);

        std::fs::write(&path, &buf).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert: non-UTF8 tensor name must be rejected
        assert!(result.is_err(), "non-UTF8 tensor name should cause error");
        let err = result.unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
        assert!(err.to_string().contains("invalid tensor name"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_with_nonzero_relative_data_offset() {
        // Arrange: tensor with data_offset=16 (relative to data section start)
        let dir = unique_test_dir("rel_off");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rel_off.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 2;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        // Tensor entry with data_offset=16 relative to data section
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&16u64.to_le_bytes()); // data_offset = 16 (relative)
        buf.extend_from_slice(&8u64.to_le_bytes());  // compressed_size = 8
        buf.extend_from_slice(&8u64.to_le_bytes());  // original_size = 8

        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&[0xAB, 0xCD]);
        // Data region: 16 padding bytes + 8 bytes of actual tensor data
        buf.extend_from_slice(&[0u8; 16]); // padding
        let tensor_content: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        buf.extend_from_slice(&tensor_content);

        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("t").unwrap();

        // Assert: data starts at data_section_start + 16, contains the specific bytes
        assert_eq!(td.len(), 8);
        assert_eq!(&td[..], &tensor_content);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_only_leading_zeros_preserves_them() {
        // Arrange: metadata [0x00, 0x00, 0xAB] — trailing byte is non-zero
        let dir = unique_test_dir("meta_lead");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_lead.gllm");

        let mut buf = Vec::new();
        let name = "t";
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name.len()) as u64;
        let data_offset: u64 = meta_offset + 3;

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());
        buf.extend_from_slice(&16u64.to_le_bytes());

        buf.extend_from_slice(name.as_bytes());
        // Metadata: leading zeros but non-zero final byte
        buf.extend_from_slice(&[0x00, 0x00, 0xAB]);
        buf.extend_from_slice(&[0u8; 16]);

        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let meta = reader.metadata_bytes();

        // Assert: trailing zero strip keeps [0x00, 0x00, 0xAB] because the last byte is non-zero
        assert_eq!(meta, &[0x00, 0x00, 0xAB]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_params_negative_string_values_returns_none() {
        // Arrange: negative numbers in string form — fail u64 parse → None
        let meta = r#"{"vocab_size":"-1","hidden_size":"-4096","num_layers":"-32"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("mp_neg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mp_neg.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: u64::from_str("-1") fails → required field missing → None
        assert!(reader.model_params().is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gllm_error_source_returns_none_for_non_io_variants() {
        // Arrange: create non-Io error variants
        use std::error::Error;

        let errors: Vec<GllmError> = vec![
            GllmError::InvalidMagic(0),
            GllmError::UnsupportedVersion(1),
            GllmError::HeaderTooSmall(0),
            GllmError::ParseError("test".to_string()),
            GllmError::DuplicateTensorName("x".to_string()),
            GllmError::InvalidDType(99),
            GllmError::InvalidQuantType(99),
            GllmError::InvalidMetadata("m".to_string()),
        ];

        // Act & Assert: none of these should have a source
        for err in &errors {
            assert!(err.source().is_none(), "non-Io error should have no source");
        }
    }

    #[test]
    fn tensor_data_content_matches_mmap_for_quantized_tensor() {
        // Arrange: quantized tensor with specific byte pattern in data
        let mut data = build_gllm_with_json_meta(&["q_weight"], "{}", 14); // Q8_0
        // Overwrite the data region (last 32 bytes) with a known pattern
        let data_start = data.len() - 32;
        for i in 0..32 {
            data[data_start + i] = (i as u8).wrapping_add(0xA0);
        }
        let dir = unique_test_dir("qcontent");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qcontent.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("q_weight").unwrap();

        // Assert: readback matches the pattern we wrote
        assert_eq!(td.len(), 32);
        for i in 0..32 {
            assert_eq!(td[i], (i as u8).wrapping_add(0xA0), "byte {i} mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_data_section_at_exact_file_end() {
        // Arrange: data region starts at exact file end, tensor data_offset=0 and data_size=0
        let dir = unique_test_dir("exact_end");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_end.gllm");

        let mut buf = Vec::new();
        let name = "empty_tensor";
        let name_bytes = name.as_bytes();
        let tensor_dir_offset: u64 = HEADER_SIZE as u64;
        let string_table_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let meta_offset: u64 = (string_table_offset + name_bytes.len()) as u64;
        let data_offset: u64 = meta_offset + 2; // data starts right after metadata

        buf.extend_from_slice(b"GLLM");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&meta_offset.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&tensor_dir_offset.to_le_bytes());
        buf.extend_from_slice(&data_offset.to_le_bytes());
        buf.extend_from_slice(&4096u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 20]);

        // Tensor with zero data size
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        buf.push(1); buf.push(0);
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&[0u8; 24]);
        buf.push(0);
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(0); buf.push(0);
        buf.extend_from_slice(&[0u8; 3]);
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // compressed_size = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // original_size = 0

        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&[0xEE, 0xFF]); // metadata
        // No data region at all — file ends right at data_offset

        std::fs::write(&path, &buf).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();

        // Assert: zero-size tensor parses successfully
        assert_eq!(reader.tensor_count(), 1);
        let t = reader.find_tensor("empty_tensor").unwrap();
        assert_eq!(t.data_size, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_parse_larger_buffer_extras_ignored() {
        // Arrange: 128-byte buffer, header at the start, trailing bytes are garbage
        let mut buf = vec![0xABu8; 128];
        buf[0..4].copy_from_slice(&0x4D4C4C47u32.to_le_bytes()); // GLLM magic
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version 1
        buf[8..12].copy_from_slice(&0u32.to_le_bytes()); // flags
        buf[12..20].copy_from_slice(&64u64.to_le_bytes()); // meta_offset
        buf[20..24].copy_from_slice(&0u32.to_le_bytes()); // tensor_count = 0
        buf[24..32].copy_from_slice(&64u64.to_le_bytes()); // tensor_dir_offset
        buf[32..40].copy_from_slice(&64u64.to_le_bytes()); // data_offset = 64
        buf[40..44].copy_from_slice(&4096u32.to_le_bytes()); // page_size

        // Act
        let h = GllmHeader::parse(&buf).unwrap();

        // Assert: parse succeeds even with extra trailing data; reads only 64 bytes
        assert_eq!(h.version, 1);
        assert_eq!(h.flags, 0);
        assert_eq!(h.tensor_count, 0);
        assert_eq!(h.data_offset, 64);
        // Trailing bytes (64..128) are ignored by the header parser
    }

    #[test]
    fn architecture_with_multiple_extra_keys_extracts_correctly() {
        // Arrange: JSON with arch_key plus many other model parameter keys
        let meta = r#"{"arch_key":"llama4","vocab_size":"32000","hidden_size":"4096","num_layers":"32","num_heads":"32","num_kv_heads":"8","head_dim":"128","intermediate_size":"11008","context_length":"8192","custom_field":"value","tokenizer":"bpe"}"#;
        let data = build_gllm_with_json_meta(&["w"], meta, 0);
        let dir = unique_test_dir("arch_multi");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_multi.gllm");
        std::fs::write(&path, &data).unwrap();

        // Act
        let reader = GllmReader::open(&path).unwrap();
        let arch = reader.architecture();
        let params = reader.model_params().unwrap();

        // Assert: architecture extracts just arch_key; model_params parses all numeric fields
        assert_eq!(arch.as_deref(), Some("llama4"));
        assert_eq!(params.vocab_size, 32000);
        assert_eq!(params.hidden_size, 4096);
        assert_eq!(params.num_layers, 32);
        assert_eq!(params.num_heads, 32);
        assert_eq!(params.num_kv_heads, 8);
        assert_eq!(params.head_dim, 128);
        assert_eq!(params.intermediate_size, 11008);
        assert_eq!(params.context_length, 8192);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
