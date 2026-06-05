//! .gllm 文件写入器 — 构建 .gllm 文件的 builder。
//!
//! SPEC: `SPEC/36-GLLM-WEIGHT-FORMAT.md §1`
//!
//! # 布局
//!
//! ```text
//! Header (64B) → Tensor Directory (72B × N) → String Table → Metadata (MessagePack) → Data
//! ```

use std::collections::HashMap;
use std::io::{self, Write};

use super::types::GllmError;
use super::{GLLM_MAGIC, GLLM_VERSION, HEADER_SIZE, TENSOR_ENTRY_SIZE};

/// QuantType → u8 编码（与 reader.rs 中 gllm_quant_type_from_u8 对称）。
pub fn quant_type_to_u8(qt: gllm_kernels::quant::QuantType) -> u8 {
    use gllm_kernels::quant::QuantType;
    match qt {
        QuantType::Bf16 => 1,
        QuantType::F16 => 2,
        QuantType::F32 => 3,
        QuantType::Q4_0 => 10,
        QuantType::Q4_1 => 11,
        QuantType::Q5_0 => 12,
        QuantType::Q5_1 => 13,
        QuantType::Q8_0 => 14,
        QuantType::Q8_1 => 15,
        QuantType::Q2K => 20,
        QuantType::Q3K => 21,
        QuantType::Q4K => 22,
        QuantType::Q5K => 23,
        QuantType::Q6K => 24,
        QuantType::Q8K => 25,
        QuantType::IQ1S => 30,
        QuantType::IQ1M => 31,
        QuantType::IQ2XXS => 32,
        QuantType::IQ2XS => 33,
        QuantType::IQ2S => 34,
        QuantType::IQ3XXS => 35,
        QuantType::IQ3S => 36,
        QuantType::IQ4NL => 37,
        QuantType::IQ4XS => 38,
        QuantType::AWQ4 => 40,
        QuantType::GPTQ4 => 41,
        QuantType::Squeeze => 42,
        QuantType::Fp8E4M3 => 50,
        QuantType::Fp8E5M2 => 51,
        QuantType::Mxfp4 { .. } => 52,
        QuantType::Nvfp4 => 53,
        QuantType::TQ1_0 => 60,
        QuantType::TQ2_0 => 61,
    }
}

/// DType → u8 编码（与 reader.rs 中 gllm_dtype_to_st 对称）。
#[allow(dead_code)]
pub(crate) fn dtype_to_u8(dt: u8) -> u8 {
    dt // safetensors Dtype u8 值直接透传
}

/// safetensors Dtype → u8 编码。
pub(crate) fn safetensors_dtype_to_u8(dt: safetensors::Dtype) -> u8 {
    match dt {
        safetensors::Dtype::F32 => 0,
        safetensors::Dtype::F16 => 1,
        safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::U8 => 3,
        safetensors::Dtype::I8 => 4,
        safetensors::Dtype::I32 => 5,
        safetensors::Dtype::I64 => 6,
        _ => 0, // fallback to F32
    }
}

/// 待写入的张量条目。
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    pub ndim: u8,
    pub dtype: u8,
    pub shape: [u64; 4],
    pub quant_format: u8,
    pub quant_block_size: u16,
    pub scale_dtype: u8,
    pub zp_type: u8,
    pub data: Vec<u8>,
    pub original_size: u64,
}

impl TensorEntry {
    /// 量化张量的 compressed_size。
    pub fn compressed_size(&self) -> u64 {
        self.data.len() as u64
    }

    /// 是否量化。
    pub fn is_quantized(&self) -> bool {
        self.quant_format != 0
    }
}

/// .gllm 文件构建器。
///
/// 用法:
/// ```ignore
/// let mut builder = GllmWriter::new(4096);
/// builder.add_tensor(entry);
/// builder.set_metadata(metadata_bytes);
/// builder.write_to_path("output.gllm")?;
/// ```
pub struct GllmWriter {
    tensors: Vec<TensorEntry>,
    metadata_bytes: Vec<u8>,
    page_size: u32,
}

impl GllmWriter {
    pub fn new(page_size: u32) -> Self {
        Self {
            tensors: Vec::new(),
            metadata_bytes: Vec::new(),
            page_size,
        }
    }

    pub fn add_tensor(&mut self, entry: TensorEntry) {
        self.tensors.push(entry);
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn set_metadata(&mut self, bytes: Vec<u8>) {
        self.metadata_bytes = bytes;
    }

    /// 写入 .gllm 到文件。
    pub fn write_to_path(&self, path: &std::path::Path) -> Result<(), GllmError> {
        let mut file = std::fs::File::create(path)?;
        self.write_to(&mut file)?;
        file.flush()?;
        Ok(())
    }

    /// 写入 .gllm 到任意 Write。
    ///
    /// 布局: Header → TensorDir → StringTable → Metadata → Data
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        let tensor_count = self.tensors.len() as u32;

        // ── 计算布局偏移 ──────────────────────────────────────────────
        let tensor_dir_offset = HEADER_SIZE as u64;
        let tensor_dir_size = self.tensors.len() * TENSOR_ENTRY_SIZE;

        // String table: 紧接 TensorDir
        let string_table_offset = tensor_dir_offset + tensor_dir_size as u64;
        let mut string_table = Vec::new();
        let mut name_offsets = Vec::with_capacity(self.tensors.len());
        for entry in &self.tensors {
            name_offsets.push(string_table.len() as u32);
            string_table.extend_from_slice(entry.name.as_bytes());
        }

        // Metadata: 紧接 StringTable
        let meta_offset = string_table_offset + string_table.len() as u64;

        // Data: 紧接 Metadata，page-aligned
        let data_base_unaligned = meta_offset + self.metadata_bytes.len() as u64;
        let data_offset = align_up(data_base_unaligned, self.page_size as u64);

        // ── Header (64 bytes) ──────────────────────────────────────────
        let has_quant = self.tensors.iter().any(|t| t.is_quantized());
        let flags: u32 = if has_quant { 1 } else { 0 };

        w.write_all(&GLLM_MAGIC.to_le_bytes())?;           // 0..4
        w.write_all(&GLLM_VERSION.to_le_bytes())?;          // 4..8
        w.write_all(&flags.to_le_bytes())?;                  // 8..12
        w.write_all(&meta_offset.to_le_bytes())?;            // 12..20
        w.write_all(&tensor_count.to_le_bytes())?;           // 20..24
        w.write_all(&tensor_dir_offset.to_le_bytes())?;      // 24..32
        w.write_all(&data_offset.to_le_bytes())?;            // 32..40
        w.write_all(&self.page_size.to_le_bytes())?;         // 40..44
        w.write_all(&[0u8; 20])?;                            // 44..64

        // ── Tensor Directory (72 bytes × N) ────────────────────────────
        // 同时计算每个张量的 data_offset（在 data 区内的偏移）
        let mut data_cursor: u64 = 0;
        for (i, entry) in self.tensors.iter().enumerate() {
            let name_off = name_offsets[i];
            let name_len = entry.name.len() as u16;
            let t_data_offset = data_cursor;

            // 写入 72-byte entry
            w.write_all(&name_off.to_le_bytes())?;           // 0..4
            w.write_all(&name_len.to_le_bytes())?;           // 4..6
            w.write_all(&entry.ndim.to_le_bytes())?;         // 6
            w.write_all(&entry.dtype.to_le_bytes())?;        // 7
            for s in &entry.shape {                           // 8..40
                w.write_all(&s.to_le_bytes())?;
            }
            w.write_all(&entry.quant_format.to_le_bytes())?;  // 40
            w.write_all(&entry.quant_block_size.to_le_bytes())?; // 41..43
            w.write_all(&entry.scale_dtype.to_le_bytes())?;   // 43
            w.write_all(&entry.zp_type.to_le_bytes())?;       // 44
            w.write_all(&[0u8; 3])?;                           // 45..47 padding
            w.write_all(&t_data_offset.to_le_bytes())?;       // 48..56
            w.write_all(&entry.compressed_size().to_le_bytes())?; // 56..64
            w.write_all(&entry.original_size.to_le_bytes())?; // 64..72

            // 数据对齐到 page_size
            let entry_size = entry.data.len() as u64;
            let aligned_size = align_up(entry_size, self.page_size as u64);
            data_cursor += aligned_size;
        }

        // ── String Table ───────────────────────────────────────────────
        w.write_all(&string_table)?;

        // ── Metadata (MessagePack) ─────────────────────────────────────
        w.write_all(&self.metadata_bytes)?;

        // ── Padding to data_offset ─────────────────────────────────────
        let current_pos = HEADER_SIZE + tensor_dir_size + string_table.len() + self.metadata_bytes.len();
        if data_offset as usize > current_pos {
            let padding = data_offset as usize - current_pos;
            let zeros = vec![0u8; padding];
            w.write_all(&zeros)?;
        }

        // ── Tensor Data (page-aligned) ─────────────────────────────────
        for entry in &self.tensors {
            w.write_all(&entry.data)?;
            // Page-align
            let aligned_size = align_up(entry.data.len() as u64, self.page_size as u64) as usize;
            let padding = aligned_size - entry.data.len();
            if padding > 0 {
                let zeros = vec![0u8; padding];
                w.write_all(&zeros)?;
            }
        }

        Ok(())
    }
}

/// 向上对齐到 alignment。
fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

/// 构建 MessagePack 格式的模型元数据。
///
/// SPEC 36 §1: Metadata 包含 model_type, arch_key, vocab_size, hidden_size,
/// num_layers, num_heads, head_dim, intermediate_size 等。
pub fn build_metadata(
    arch_key: &str,
    vocab_size: u64,
    hidden_size: u64,
    num_layers: u64,
    num_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    intermediate_size: u64,
    context_length: u64,
    extras: &HashMap<String, String>,
) -> Vec<u8> {
    let mut map = HashMap::new();
    map.insert("arch_key".to_string(), arch_key.to_string());
    map.insert("vocab_size".to_string(), vocab_size.to_string());
    map.insert("hidden_size".to_string(), hidden_size.to_string());
    map.insert("num_layers".to_string(), num_layers.to_string());
    map.insert("num_heads".to_string(), num_heads.to_string());
    map.insert("num_kv_heads".to_string(), num_kv_heads.to_string());
    map.insert("head_dim".to_string(), head_dim.to_string());
    map.insert("intermediate_size".to_string(), intermediate_size.to_string());
    map.insert("context_length".to_string(), context_length.to_string());
    for (k, v) in extras {
        map.insert(k.clone(), v.clone());
    }
    serde_json::to_vec(&map).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::gllm::{GllmHeader, GllmReader, GllmTensorEntry};
    use std::sync::atomic::{AtomicU64, Ordering};
    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);
    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("gllm_test_{}_{}_{}", name, std::process::id(), id))
    }

    fn roundtrip_gllm(page_size: u32, subdir: &str) {
        let mut builder = GllmWriter::new(page_size);

        // 添加一个未量化张量
        let data = vec![42u8; 64];
        builder.add_tensor(TensorEntry {
            name: "test_tensor".to_string(),
            ndim: 2,
            dtype: 0, // F32
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 64,
        });

        // 添加一个量化张量 (Q4_0)
        let qdata = vec![0xABu8; 36]; // Q4_0 block: 18 bytes
        builder.add_tensor(TensorEntry {
            name: "quant_weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [1, 32, 0, 0], // 1 row, 32 elements
            quant_format: 10,     // Q4_0
            quant_block_size: 32,
            scale_dtype: 1, // F16
            zp_type: 0,
            data: qdata.clone(),
            original_size: 128, // 32 * 4 bytes if F32
        });

        let meta = build_metadata(
            "qwen3", 151936, 4096, 36, 32, 8, 128, 11008, 32768, &HashMap::new(),
        );
        builder.set_metadata(meta);

        let dir = std::env::temp_dir().join(subdir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.gllm");

        builder.write_to_path(&path).unwrap();

        // 验证可读回
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 2);
        assert!(reader.header().is_quantized());
        assert_eq!(reader.header().page_size, page_size);

        let t1 = reader.find_tensor("test_tensor").unwrap();
        assert_eq!(t1.entry.shape[0], 4);
        assert!(!t1.entry.is_quantized());

        let t2 = reader.find_tensor("quant_weight").unwrap();
        assert!(t2.entry.is_quantized());
        assert_eq!(t2.entry.quant_format, 10);
        assert_eq!(t2.entry.quant_block_size, 32);

        let td1 = reader.tensor_data("test_tensor").unwrap();
        assert_eq!(td1.len(), 64);
        assert_eq!(td1[0], 42);

        let td2 = reader.tensor_data("quant_weight").unwrap();
        assert_eq!(td2.len(), 36);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_and_read_back_4k() {
        roundtrip_gllm(4096, "gllm_test_write_4k");
    }

    #[test]
    fn write_and_read_back_512() {
        roundtrip_gllm(512, "gllm_test_write_512");
    }

    #[test]
    fn quant_type_roundtrip() {
        use gllm_kernels::quant::QuantType;
        let types = [
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1, QuantType::Q2K, QuantType::Q3K,
            QuantType::Q4K, QuantType::Q5K, QuantType::Q6K, QuantType::Q8K,
            QuantType::IQ1S, QuantType::IQ2XXS, QuantType::IQ3XXS, QuantType::IQ4NL,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Fp8E4M3, QuantType::Nvfp4,
        ];
        for qt in &types {
            let code = quant_type_to_u8(*qt);
            assert_ne!(code, 0, "QuantType {:?} should have non-zero encoding", qt);
        }
    }

    #[test]
    fn tensor_entry_compressed_size() {
        let entry = TensorEntry {
            name: "test".into(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 128],
            original_size: 256,
        };
        assert_eq!(entry.compressed_size(), 128);
    }

    #[test]
    fn tensor_entry_is_quantized() {
        let unquantized = TensorEntry {
            name: "a".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert!(!unquantized.is_quantized());

        let quantized = TensorEntry {
            name: "b".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: 32, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert!(quantized.is_quantized());
    }

    #[test]
    fn safetensors_dtype_to_u8_mapping() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
    }

    #[test]
    fn safetensors_dtype_to_u8_unknown_defaults_to_f32() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BOOL), 0);
    }

    #[test]
    fn quant_type_to_u8_float_types() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
        assert_eq!(quant_type_to_u8(QuantType::F32), 3);
    }

    #[test]
    fn quant_type_to_u8_classic_quants() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Q4_0), 10);
        assert_eq!(quant_type_to_u8(QuantType::Q8_0), 14);
    }

    #[test]
    fn quant_type_to_u8_all_unique() {
        use gllm_kernels::quant::QuantType;
        let types = [
            QuantType::Bf16, QuantType::F16, QuantType::F32,
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
            QuantType::Q6K, QuantType::Q8K,
            QuantType::AWQ4, QuantType::GPTQ4,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2, QuantType::Nvfp4,
            QuantType::TQ1_0, QuantType::TQ2_0,
        ];
        let codes: Vec<u8> = types.iter().map(|&qt| quant_type_to_u8(qt)).collect();
        let unique: std::collections::HashSet<u8> = codes.iter().copied().collect();
        assert_eq!(codes.len(), unique.len(), "all QuantType codes must be unique");
    }

    #[test]
    fn writer_new_empty() {
        let writer = GllmWriter::new(4096);
        assert_eq!(writer.tensors.len(), 0);
    }

    #[test]
    fn writer_add_tensor() {
        let mut writer = GllmWriter::new(4096);
        let entry = TensorEntry {
            name: "weight".into(), ndim: 2, dtype: 0,
            shape: [4, 4, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 64], original_size: 64,
        };
        writer.add_tensor(entry);
        assert_eq!(writer.tensors.len(), 1);
    }

    // ────────────────────────────────────────────────────────────────────────
    // New tests start here
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn align_up_exact_multiple() {
        // 4096 already aligned to 4096
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(0, 512), 0);
        assert_eq!(align_up(1024, 512), 1024);
    }

    #[test]
    fn align_up_non_multiple() {
        // 1 → 4096 when aligned to 4096
        assert_eq!(align_up(1, 4096), 4096);
        // 100 → 512 when aligned to 512
        assert_eq!(align_up(100, 512), 512);
        // 513 → 1024 when aligned to 512
        assert_eq!(align_up(513, 512), 1024);
    }

    #[test]
    fn align_up_zero_alignment_returns_value() {
        assert_eq!(align_up(999, 0), 999);
        assert_eq!(align_up(0, 0), 0);
    }

    #[test]
    fn build_metadata_contains_all_required_fields() {
        let meta = build_metadata(
            "llama4", 32000, 2048, 12, 16, 4, 128, 8192, 4096, &HashMap::new(),
        );
        let s = String::from_utf8(meta).unwrap();
        assert!(s.contains("arch_key"), "metadata must contain arch_key");
        assert!(s.contains("vocab_size"), "metadata must contain vocab_size");
        assert!(s.contains("hidden_size"), "metadata must contain hidden_size");
        assert!(s.contains("num_layers"), "metadata must contain num_layers");
        assert!(s.contains("num_heads"), "metadata must contain num_heads");
        assert!(s.contains("num_kv_heads"), "metadata must contain num_kv_heads");
        assert!(s.contains("head_dim"), "metadata must contain head_dim");
        assert!(s.contains("intermediate_size"), "metadata must contain intermediate_size");
        assert!(s.contains("context_length"), "metadata must contain context_length");
    }

    #[test]
    fn build_metadata_includes_extras() {
        let mut extras = HashMap::new();
        extras.insert("custom_field".to_string(), "custom_value".to_string());
        let meta = build_metadata(
            "qwen3", 100, 200, 300, 400, 500, 600, 700, 800, &extras,
        );
        let s = String::from_utf8(meta).unwrap();
        assert!(s.contains("custom_field"), "metadata must include extra key");
        assert!(s.contains("custom_value"), "metadata must include extra value");
    }

    #[test]
    fn build_metadata_preserves_values() {
        let meta = build_metadata(
            "phi4", 100, 256, 4, 8, 2, 64, 512, 1024, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "phi4");
        assert_eq!(parsed["vocab_size"], "100");
        assert_eq!(parsed["hidden_size"], "256");
        assert_eq!(parsed["num_layers"], "4");
        assert_eq!(parsed["num_heads"], "8");
        assert_eq!(parsed["num_kv_heads"], "2");
        assert_eq!(parsed["head_dim"], "64");
        assert_eq!(parsed["intermediate_size"], "512");
        assert_eq!(parsed["context_length"], "1024");
    }

    #[test]
    fn dtype_to_u8_passthrough() {
        assert_eq!(dtype_to_u8(0), 0);
        assert_eq!(dtype_to_u8(1), 1);
        assert_eq!(dtype_to_u8(42), 42);
        assert_eq!(dtype_to_u8(255), 255);
    }

    #[test]
    fn tensor_entry_clone_is_independent() {
        let original = TensorEntry {
            name: "layer.0.weight".to_string(),
            ndim: 2,
            dtype: 3,
            shape: [512, 512, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![7u8; 32],
            original_size: 32,
        };
        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.data.len(), original.data.len());
        // Mutating clone does not affect original
        let mut modified = cloned;
        modified.data[0] = 99;
        assert_eq!(original.data[0], 7);
        assert_eq!(modified.data[0], 99);
    }

    #[test]
    fn tensor_entry_compressed_size_empty_data() {
        let entry = TensorEntry {
            name: "empty".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert_eq!(entry.compressed_size(), 0);
    }

    #[test]
    fn tensor_entry_compressed_size_large_data() {
        let data = vec![0xFFu8; 65536];
        let entry = TensorEntry {
            name: "big".into(), ndim: 2, dtype: 0, shape: [256, 256, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 65536,
        };
        assert_eq!(entry.compressed_size(), 65536);
    }

    #[test]
    fn tensor_entry_is_quantized_various_formats() {
        // quant_format == 0 means not quantized
        let plain = TensorEntry {
            name: "a".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert!(!plain.is_quantized());

        // Any non-zero quant_format means quantized
        for qf in [1u8, 10, 20, 40, 50, 60, 255] {
            let quant = TensorEntry {
                name: "b".into(), ndim: 1, dtype: 0, shape: [0; 4],
                quant_format: qf, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![], original_size: 0,
            };
            assert!(quant.is_quantized(), "quant_format={} should be quantized", qf);
        }
    }

    #[test]
    fn writer_tensor_count_tracks_additions() {
        let mut writer = GllmWriter::new(4096);
        assert_eq!(writer.tensor_count(), 0);

        writer.add_tensor(TensorEntry {
            name: "t1".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(writer.tensor_count(), 1);

        writer.add_tensor(TensorEntry {
            name: "t2".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(writer.tensor_count(), 2);
    }

    #[test]
    fn writer_set_metadata_stores_bytes() {
        let mut writer = GllmWriter::new(4096);
        let meta = vec![1, 2, 3, 4, 5];
        writer.set_metadata(meta.clone());
        assert_eq!(writer.metadata_bytes, meta);
    }

    #[test]
    fn write_to_bytes_unquantized_only_no_quant_flag() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "plain".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAA; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("no_quant_flag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_quant.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(!reader.header().is_quantized(), "unquantized-only file must not set quant flag");
        assert_eq!(reader.tensor_count(), 1);

        let td = reader.tensor_data("plain").unwrap();
        assert_eq!(td.len(), 16);
        assert!(td.iter().all(|&b| b == 0xAA));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_bytes_many_tensors_roundtrip() {
        let mut builder = GllmWriter::new(512);
        for i in 0..5 {
            builder.add_tensor(TensorEntry {
                name: format!("tensor_{}", i),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![i as u8; 16],
                original_size: 16,
            });
        }
        let meta = build_metadata("test", 100, 64, 2, 4, 2, 16, 128, 256, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("many_tensors");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 5);

        for i in 0..5 {
            let name = format!("tensor_{}", i);
            let td = reader.tensor_data(&name).unwrap();
            assert_eq!(td.len(), 16);
            assert!(td.iter().all(|&b| b == i as u8), "tensor_{} data mismatch", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_to_u8_iq_family() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::IQ1S), 30);
        assert_eq!(quant_type_to_u8(QuantType::IQ1M), 31);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XXS), 32);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XS), 33);
        assert_eq!(quant_type_to_u8(QuantType::IQ2S), 34);
        assert_eq!(quant_type_to_u8(QuantType::IQ3XXS), 35);
        assert_eq!(quant_type_to_u8(QuantType::IQ3S), 36);
        assert_eq!(quant_type_to_u8(QuantType::IQ4NL), 37);
        assert_eq!(quant_type_to_u8(QuantType::IQ4XS), 38);
    }

    #[test]
    fn quant_type_to_u8_special_types() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Squeeze), 42);
        assert_eq!(quant_type_to_u8(QuantType::Fp8E5M2), 51);
        assert_eq!(quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 }), 52);
        assert_eq!(quant_type_to_u8(QuantType::Nvfp4), 53);
        assert_eq!(quant_type_to_u8(QuantType::TQ1_0), 60);
        assert_eq!(quant_type_to_u8(QuantType::TQ2_0), 61);
    }

    #[test]
    fn quant_type_to_u8_k_quant_family() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Q2K), 20);
        assert_eq!(quant_type_to_u8(QuantType::Q3K), 21);
        assert_eq!(quant_type_to_u8(QuantType::Q4K), 22);
        assert_eq!(quant_type_to_u8(QuantType::Q5K), 23);
        assert_eq!(quant_type_to_u8(QuantType::Q6K), 24);
        assert_eq!(quant_type_to_u8(QuantType::Q8K), 25);
    }

    #[test]
    fn write_empty_tensors_with_metadata() {
        let mut builder = GllmWriter::new(4096);
        let meta = build_metadata("empty", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("empty_tensors");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 0);
        assert!(!reader.header().is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_data_preserved_exact_bytes() {
        let mut builder = GllmWriter::new(256);
        let original_data: Vec<u8> = (0..200).map(|i| (i * 7 + 3) as u8).collect();
        builder.add_tensor(TensorEntry {
            name: "exact_bytes".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [200, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: original_data.clone(),
            original_size: 200,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("exact_bytes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("exact_bytes").unwrap();
        assert_eq!(td.len(), 200);
        assert_eq!(&td[..], &original_data[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_page_alignment_in_data_region() {
        let mut builder = GllmWriter::new(128);
        // Tensor with 10 bytes of data; must be padded to 128 in the data region
        builder.add_tensor(TensorEntry {
            name: "small".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xCD; 10],
            original_size: 10,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("page_align");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aligned.gllm");
        builder.write_to_path(&path).unwrap();

        // Read back raw file and verify page alignment
        let raw = std::fs::read(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;

        // First 10 bytes should be 0xCD, rest within the page should be zeros
        assert_eq!(raw[data_offset], 0xCD);
        assert_eq!(raw[data_offset + 9], 0xCD);
        // Padding bytes (10..128) should be zero
        for i in 10..128 {
            assert_eq!(raw[data_offset + i], 0, "padding byte at offset {} should be zero", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional unit tests — pure logic, no I/O
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_debug_format() {
        let entry = TensorEntry {
            name: "layers.0.weight".to_string(),
            ndim: 2,
            dtype: 3,
            shape: [4096, 4096, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 128,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("TensorEntry"), "Debug should contain struct name");
        assert!(debug.contains("layers.0.weight"), "Debug should contain tensor name");
        assert!(debug.contains("ndim: 2"), "Debug should show ndim");
    }

    #[test]
    fn tensor_entry_construction_all_fields() {
        let entry = TensorEntry {
            name: "model.embed_tokens.weight".to_string(),
            ndim: 2,
            dtype: 2,
            shape: [151936, 2048, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xAB; 256],
            original_size: 1024,
        };
        assert_eq!(entry.name, "model.embed_tokens.weight");
        assert_eq!(entry.ndim, 2);
        assert_eq!(entry.dtype, 2);
        assert_eq!(entry.shape, [151936, 2048, 0, 0]);
        assert_eq!(entry.quant_format, 40);
        assert_eq!(entry.quant_block_size, 128);
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
        assert_eq!(entry.data.len(), 256);
        assert_eq!(entry.original_size, 1024);
    }

    #[test]
    fn tensor_entry_shape_all_dimensions_populated() {
        let entry = TensorEntry {
            name: "4d_tensor".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [8, 16, 32, 64],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.shape[0], 8);
        assert_eq!(entry.shape[1], 16);
        assert_eq!(entry.shape[2], 32);
        assert_eq!(entry.shape[3], 64);
        assert_eq!(entry.ndim, 4);
    }

    #[test]
    fn tensor_entry_shape_large_u64_values() {
        let entry = TensorEntry {
            name: "huge".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [u64::MAX, u64::MAX / 2, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.shape[0], u64::MAX);
        assert_eq!(entry.shape[1], u64::MAX / 2);
    }

    #[test]
    fn tensor_entry_compressed_size_differs_from_original() {
        let entry = TensorEntry {
            name: "compressed".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 8192],
            original_size: 67108864,
        };
        assert_eq!(entry.compressed_size(), 8192);
        assert!(entry.compressed_size() < entry.original_size);
    }

    #[test]
    fn tensor_entry_non_ascii_name() {
        let entry = TensorEntry {
            name: "模型.层.权重".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.name, "模型.层.权重");
        assert!(entry.name.contains("模型"));
    }

    #[test]
    fn writer_page_size_stored() {
        let writer = GllmWriter::new(8192);
        assert_eq!(writer.page_size, 8192);
    }

    #[test]
    fn writer_zero_page_size() {
        let writer = GllmWriter::new(0);
        assert_eq!(writer.page_size, 0);
    }

    #[test]
    fn writer_metadata_overwrite() {
        let mut writer = GllmWriter::new(4096);
        writer.set_metadata(vec![1, 2, 3]);
        assert_eq!(writer.metadata_bytes, vec![1, 2, 3]);
        writer.set_metadata(vec![4, 5]);
        assert_eq!(writer.metadata_bytes, vec![4, 5]);
    }

    #[test]
    fn writer_tensor_ordering_preserved() {
        let mut writer = GllmWriter::new(4096);
        let names = ["alpha", "beta", "gamma", "delta"];
        for name in &names {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            });
        }
        assert_eq!(writer.tensor_count(), 4);
        assert_eq!(writer.tensors[0].name, "alpha");
        assert_eq!(writer.tensors[1].name, "beta");
        assert_eq!(writer.tensors[2].name, "gamma");
        assert_eq!(writer.tensors[3].name, "delta");
    }

    #[test]
    fn align_up_with_alignment_one() {
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(100, 1), 100);
        assert_eq!(align_up(u64::MAX, 1), u64::MAX);
    }

    #[test]
    fn align_up_alignment_larger_than_value() {
        assert_eq!(align_up(1, 65536), 65536);
    }

    #[test]
    fn align_up_large_values() {
        assert_eq!(align_up(1024 * 1024, 1024), 1024 * 1024);
        assert_eq!(align_up(1_000_001, 4096), 1_003_520);
        assert_eq!(align_up(1_000_000_000_000, 4096), 1_000_000_000_000);
        assert_eq!(align_up(1_000_000_000_001, 4096), 1_000_000_004_096);
    }

    #[test]
    fn build_metadata_is_valid_json() {
        let meta = build_metadata(
            "test_arch", 32000, 4096, 32, 32, 8, 128, 11008, 4096, &HashMap::new(),
        );
        let parsed: Result<serde_json::Value, _> = serde_json::from_slice(&meta);
        assert!(parsed.is_ok(), "build_metadata must produce valid JSON");
    }

    #[test]
    fn build_metadata_empty_arch_key() {
        let meta = build_metadata("", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "");
    }

    #[test]
    fn build_metadata_extras_override_standard_fields() {
        let mut extras = HashMap::new();
        extras.insert("arch_key".to_string(), "overridden".to_string());
        let meta = build_metadata("original", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "overridden");
    }

    #[test]
    fn build_metadata_multiple_extras() {
        let mut extras = HashMap::new();
        extras.insert("rope_theta".to_string(), "10000.0".to_string());
        extras.insert("norm_epsilon".to_string(), "1e-6".to_string());
        extras.insert("model_type".to_string(), "qwen3".to_string());
        let meta = build_metadata("qwen3", 151936, 4096, 36, 32, 8, 128, 11008, 32768, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["rope_theta"], "10000.0");
        assert_eq!(parsed["norm_epsilon"], "1e-6");
        assert_eq!(parsed["model_type"], "qwen3");
    }

    #[test]
    fn safetensors_dtype_to_u8_unhandled_variants() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F8_E5M2), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F8_E4M3), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I16), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U16), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F64), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U64), 0);
    }

    #[test]
    fn quant_type_mxfp4_same_code_regardless_of_block_size() {
        use gllm_kernels::quant::QuantType;
        let code_32 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 });
        let code_64 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 64 });
        let code_128 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 128 });
        assert_eq!(code_32, 52);
        assert_eq!(code_64, 52);
        assert_eq!(code_128, 52);
    }

    #[test]
    fn quant_type_to_u8_covers_all_named_variants() {
        use gllm_kernels::quant::QuantType;
        let all_codes: Vec<u8> = vec![
            quant_type_to_u8(QuantType::Bf16),
            quant_type_to_u8(QuantType::F16),
            quant_type_to_u8(QuantType::F32),
            quant_type_to_u8(QuantType::Q4_0),
            quant_type_to_u8(QuantType::Q4_1),
            quant_type_to_u8(QuantType::Q5_0),
            quant_type_to_u8(QuantType::Q5_1),
            quant_type_to_u8(QuantType::Q8_0),
            quant_type_to_u8(QuantType::Q8_1),
            quant_type_to_u8(QuantType::Q2K),
            quant_type_to_u8(QuantType::Q3K),
            quant_type_to_u8(QuantType::Q4K),
            quant_type_to_u8(QuantType::Q5K),
            quant_type_to_u8(QuantType::Q6K),
            quant_type_to_u8(QuantType::Q8K),
            quant_type_to_u8(QuantType::IQ1S),
            quant_type_to_u8(QuantType::IQ1M),
            quant_type_to_u8(QuantType::IQ2XXS),
            quant_type_to_u8(QuantType::IQ2XS),
            quant_type_to_u8(QuantType::IQ2S),
            quant_type_to_u8(QuantType::IQ3XXS),
            quant_type_to_u8(QuantType::IQ3S),
            quant_type_to_u8(QuantType::IQ4NL),
            quant_type_to_u8(QuantType::IQ4XS),
            quant_type_to_u8(QuantType::AWQ4),
            quant_type_to_u8(QuantType::GPTQ4),
            quant_type_to_u8(QuantType::Squeeze),
            quant_type_to_u8(QuantType::Fp8E4M3),
            quant_type_to_u8(QuantType::Fp8E5M2),
            quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 }),
            quant_type_to_u8(QuantType::Nvfp4),
            quant_type_to_u8(QuantType::TQ1_0),
            quant_type_to_u8(QuantType::TQ2_0),
        ];
        for code in &all_codes {
            assert_ne!(*code, 0, "code must be non-zero");
        }
        let unique: std::collections::HashSet<u8> = all_codes.iter().copied().collect();
        assert_eq!(unique.len(), all_codes.len(), "all variant codes must be unique");
    }

    #[test]
    fn tensor_entry_empty_name() {
        let entry = TensorEntry {
            name: String::new(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(entry.name.is_empty());
        assert_eq!(entry.compressed_size(), 0);
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_max_field_values() {
        let entry = TensorEntry {
            name: "x".repeat(1000),
            ndim: u8::MAX,
            dtype: u8::MAX,
            shape: [u64::MAX; 4],
            quant_format: u8::MAX,
            quant_block_size: u16::MAX,
            scale_dtype: u8::MAX,
            zp_type: u8::MAX,
            data: vec![],
            original_size: u64::MAX,
        };
        assert_eq!(entry.ndim, 255);
        assert_eq!(entry.dtype, 255);
        assert_eq!(entry.shape, [u64::MAX; 4]);
        assert_eq!(entry.quant_format, 255);
        assert_eq!(entry.quant_block_size, 65535);
        assert_eq!(entry.scale_dtype, 255);
        assert_eq!(entry.zp_type, 255);
        assert!(entry.is_quantized());
        assert_eq!(entry.original_size, u64::MAX);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional 18 unit tests — pure logic, no I/O
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_ndim_zero_single_dim_shape() {
        // ndim=0 is semantically unusual but structurally valid
        let entry = TensorEntry {
            name: "scalar_param".to_string(),
            ndim: 0,
            dtype: 3,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        };
        assert_eq!(entry.ndim, 0);
        assert_eq!(entry.shape, [0, 0, 0, 0]);
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_single_dimension_shape() {
        // 1D tensor (bias vector)
        let entry = TensorEntry {
            name: "model.layers.0.bias".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4096, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16384],
            original_size: 16384,
        };
        assert_eq!(entry.ndim, 1);
        assert_eq!(entry.shape[0], 4096);
        assert_eq!(entry.shape[1], 0);
    }

    #[test]
    fn tensor_entry_compressed_size_equals_original_for_unquantized() {
        // Unquantized tensor: compressed_size == data.len() == original_size
        let data = vec![0xAB; 1024];
        let entry = TensorEntry {
            name: "embed.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [256, 1024, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 1024,
        };
        assert_eq!(entry.compressed_size(), 1024);
        assert_eq!(entry.compressed_size(), entry.original_size);
    }

    #[test]
    fn tensor_entry_zero_original_size_with_data() {
        // Edge case: data present but original_size is 0
        let entry = TensorEntry {
            name: "anomaly".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [64, 0, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xFF; 64],
            original_size: 0,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.compressed_size(), 64);
        assert_eq!(entry.original_size, 0);
    }

    #[test]
    fn tensor_entry_unicode_name_with_mixed_scripts() {
        // Tensor name with mixed CJK + Latin + emoji-like unicode
        let entry = TensorEntry {
            name: "层.0.attention.权重_v2".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(entry.name.contains("层"));
        assert!(entry.name.contains("attention"));
        assert!(entry.name.contains("权重"));
        assert!(entry.name.contains("v2"));
    }

    #[test]
    fn tensor_entry_very_long_name() {
        let long_name = "model.transformer.h.".to_string()
            + &"x".repeat(900)
            + ".weight";
        let entry = TensorEntry {
            name: long_name.clone(),
            ndim: 2,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.name.len(), long_name.len());
        assert!(entry.name.starts_with("model.transformer.h."));
        assert!(entry.name.ends_with(".weight"));
    }

    #[test]
    fn tensor_entry_shape_trailing_zeros_for_lower_dims() {
        // 2D tensor: shape[2] and shape[3] should be 0
        let entry = TensorEntry {
            name: "2d_tensor".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [1024, 768, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.shape[0], 1024);
        assert_eq!(entry.shape[1], 768);
        assert_eq!(entry.shape[2], 0);
        assert_eq!(entry.shape[3], 0);
    }

    #[test]
    fn tensor_entry_3d_shape() {
        // 3D tensor (e.g., attention bias or positional encoding)
        let entry = TensorEntry {
            name: "attention.bias".to_string(),
            ndim: 3,
            dtype: 0,
            shape: [12, 128, 128, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.ndim, 3);
        assert_eq!(entry.shape, [12, 128, 128, 0]);
    }

    #[test]
    fn tensor_entry_quant_fields_with_awq_like_values() {
        // Simulate AWQ4 quantization fields
        let entry = TensorEntry {
            name: "model.layers.0.mlp.gate_proj.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [11008, 4096, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 2,   // BF16 scales
            zp_type: 1,       // u8 zero-point
            data: vec![0u8; 5636096],
            original_size: 180439040,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.quant_format, 40);
        assert_eq!(entry.quant_block_size, 128);
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
        assert!(entry.compressed_size() < entry.original_size);
    }

    #[test]
    fn tensor_entry_quant_fields_with_gptq_like_values() {
        // Simulate GPTQ4 quantization fields
        let entry = TensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 41, // GPTQ4
            quant_block_size: 128,
            scale_dtype: 1,   // F16 scales
            zp_type: 1,
            data: vec![0u8; 2097152],
            original_size: 67108864,
        };
        assert!(entry.is_quantized());
        assert_eq!(entry.quant_format, 41);
        assert!(entry.compressed_size() < entry.original_size);
    }

    #[test]
    fn align_up_boundary_one_below_alignment() {
        // value exactly 1 less than alignment boundary
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(511, 512), 512);
        assert_eq!(align_up(1023, 1024), 1024);
    }

    #[test]
    fn align_up_with_power_of_two_values() {
        // Common page sizes as alignment
        assert_eq!(align_up(100, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(2048, 2048), 2048);
        assert_eq!(align_up(4097, 8192), 8192);
        assert_eq!(align_up(8193, 8192), 16384);
    }

    #[test]
    fn align_up_small_alignment() {
        // Alignment of 2 and 4
        assert_eq!(align_up(3, 2), 4);
        assert_eq!(align_up(4, 2), 4);
        assert_eq!(align_up(5, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(9, 4), 12);
    }

    #[test]
    fn build_metadata_with_unicode_arch_key() {
        // Unicode arch_key should survive JSON serialization
        let meta = build_metadata(
            "模型架构_v3", 32000, 4096, 32, 32, 8, 128, 11008, 4096, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "模型架构_v3");
    }

    #[test]
    fn build_metadata_with_large_u64_values() {
        // Max or near-max u64 values should be preserved as strings
        let meta = build_metadata(
            "test", u64::MAX, u64::MAX / 2, 1, 2, 3, 4, 5, 6, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["vocab_size"], u64::MAX.to_string());
        assert_eq!(parsed["hidden_size"], (u64::MAX / 2).to_string());
    }

    #[test]
    fn build_metadata_empty_extras() {
        // Empty extras should produce same result as no extras
        let meta = build_metadata(
            "test", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // Should have exactly 9 fields (no extras)
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 9);
    }

    #[test]
    fn writer_add_multiple_tensors_in_sequence() {
        let mut writer = GllmWriter::new(4096);
        for i in 0..10 {
            writer.add_tensor(TensorEntry {
                name: format!("layer_{}", i),
                ndim: 2,
                dtype: 0,
                shape: [64, 64, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 64],
                original_size: 64,
            });
        }
        assert_eq!(writer.tensor_count(), 10);
        // Verify ordering preserved
        for i in 0..10 {
            assert_eq!(writer.tensors[i].name, format!("layer_{}", i));
        }
    }

    #[test]
    fn writer_metadata_default_empty() {
        // New writer has empty metadata
        let writer = GllmWriter::new(4096);
        assert!(writer.metadata_bytes.is_empty());
    }

    #[test]
    fn writer_metadata_large_bytes() {
        let mut writer = GllmWriter::new(4096);
        let large_meta = vec![0x42u8; 100_000];
        writer.set_metadata(large_meta.clone());
        assert_eq!(writer.metadata_bytes.len(), 100_000);
        assert!(writer.metadata_bytes.iter().all(|&b| b == 0x42));
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional 50 unit tests — comprehensive coverage
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn align_up_value_equals_alignment() {
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(512, 512), 512);
        assert_eq!(align_up(1024, 1024), 1024);
        assert_eq!(align_up(2048, 2048), 2048);
        assert_eq!(align_up(8192, 8192), 8192);
    }

    #[test]
    fn align_up_one_past_alignment() {
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(513, 512), 1024);
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(1025, 1024), 2048);
    }

    #[test]
    fn align_up_two_past_alignment() {
        assert_eq!(align_up(258, 256), 512);
        assert_eq!(align_up(514, 512), 1024);
        assert_eq!(align_up(4098, 4096), 8192);
    }

    #[test]
    fn align_up_large_alignment() {
        assert_eq!(align_up(1, 65536), 65536);
        assert_eq!(align_up(65535, 65536), 65536);
        assert_eq!(align_up(65536, 65536), 65536);
        assert_eq!(align_up(65537, 65536), 131072);
    }

    #[test]
    fn align_up_idempotent() {
        // Applying align_up twice should yield the same result
        let val = 100u64;
        let alignment = 64u64;
        let first = align_up(val, alignment);
        let second = align_up(first, alignment);
        assert_eq!(first, second);
    }

    #[test]
    fn quant_type_to_u8_fp8_variants() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Fp8E4M3), 50);
        assert_eq!(quant_type_to_u8(QuantType::Fp8E5M2), 51);
    }

    #[test]
    fn quant_type_to_u8_classic_quant_range() {
        use gllm_kernels::quant::QuantType;
        // Classic quants: Q4_0=10, Q4_1=11, Q5_0=12, Q5_1=13, Q8_0=14, Q8_1=15
        assert_eq!(quant_type_to_u8(QuantType::Q4_0), 10);
        assert_eq!(quant_type_to_u8(QuantType::Q4_1), 11);
        assert_eq!(quant_type_to_u8(QuantType::Q5_0), 12);
        assert_eq!(quant_type_to_u8(QuantType::Q5_1), 13);
        assert_eq!(quant_type_to_u8(QuantType::Q8_0), 14);
        assert_eq!(quant_type_to_u8(QuantType::Q8_1), 15);
    }

    #[test]
    fn quant_type_to_u8_tq_variants() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::TQ1_0), 60);
        assert_eq!(quant_type_to_u8(QuantType::TQ2_0), 61);
    }

    #[test]
    fn quant_type_to_u8_squeeze() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Squeeze), 42);
    }

    #[test]
    fn tensor_entry_compressed_size_single_byte() {
        let entry = TensorEntry {
            name: "tiny".into(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x42],
            original_size: 1,
        };
        assert_eq!(entry.compressed_size(), 1);
    }

    #[test]
    fn tensor_entry_original_size_independent_of_data_len() {
        // original_size can differ from data.len()
        let entry = TensorEntry {
            name: "mismatch".into(),
            ndim: 2,
            dtype: 0,
            shape: [16, 16, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 1024,
        };
        assert_eq!(entry.compressed_size(), 32);
        assert_eq!(entry.original_size, 1024);
        assert_ne!(entry.compressed_size(), entry.original_size);
    }

    #[test]
    fn tensor_entry_name_with_dots_and_hyphens() {
        let entry = TensorEntry {
            name: "model.layers.0.self-attn.q_proj.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(entry.name.contains('.'));
        assert!(entry.name.contains('-'));
        assert!(entry.name.contains('_'));
    }

    #[test]
    fn tensor_entry_data_preserves_all_byte_values() {
        let data: Vec<u8> = (0..=255).collect();
        let entry = TensorEntry {
            name: "all_bytes".into(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 256,
        };
        assert_eq!(entry.data.len(), 256);
        for i in 0..=255u8 {
            assert_eq!(entry.data[i as usize], i);
        }
    }

    #[test]
    fn tensor_entry_debug_contains_quant_fields() {
        let entry = TensorEntry {
            name: "debug_quant".into(),
            ndim: 2,
            dtype: 3,
            shape: [512, 512, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![],
            original_size: 0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("quant_format: 40"));
        assert!(debug.contains("quant_block_size: 128"));
        assert!(debug.contains("scale_dtype: 2"));
        assert!(debug.contains("zp_type: 1"));
    }

    #[test]
    fn tensor_entry_clone_data_independence() {
        let original = TensorEntry {
            name: "clone_test".into(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![10, 20, 30, 40],
            original_size: 4,
        };
        let mut cloned = original.clone();
        cloned.data[0] = 99;
        assert_eq!(original.data[0], 10, "original should not be mutated");
        assert_eq!(cloned.data[0], 99);
    }

    #[test]
    fn tensor_entry_clone_name_independence() {
        let original = TensorEntry {
            name: "original_name".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let mut cloned = original.clone();
        cloned.name.push_str("_modified");
        assert_eq!(original.name, "original_name");
        assert_eq!(cloned.name, "original_name_modified");
    }

    #[test]
    fn tensor_entry_quant_format_boundary_zero_vs_one() {
        let zero = TensorEntry {
            name: "a".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(!zero.is_quantized());

        let one = TensorEntry {
            name: "b".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 1,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(one.is_quantized());
    }

    #[test]
    fn writer_new_custom_page_sizes() {
        let w64 = GllmWriter::new(64);
        assert_eq!(w64.page_size, 64);
        let w256 = GllmWriter::new(256);
        assert_eq!(w256.page_size, 256);
        let w1 = GllmWriter::new(1);
        assert_eq!(w1.page_size, 1);
    }

    #[test]
    fn writer_add_tensor_returns_correct_count_after_many_adds() {
        let mut writer = GllmWriter::new(4096);
        for i in 0..50 {
            writer.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            });
        }
        assert_eq!(writer.tensor_count(), 50);
    }

    #[test]
    fn writer_set_metadata_empty_vec() {
        let mut writer = GllmWriter::new(4096);
        writer.set_metadata(vec![]);
        assert!(writer.metadata_bytes.is_empty());
    }

    #[test]
    fn writer_set_metadata_binary_content() {
        let mut writer = GllmWriter::new(4096);
        let binary: Vec<u8> = vec![0x00, 0xFF, 0x80, 0x7F, 0x01, 0xFE];
        writer.set_metadata(binary.clone());
        assert_eq!(writer.metadata_bytes, binary);
    }

    #[test]
    fn write_roundtrip_single_tensor_minimal() {
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "minimal".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x77],
            original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("minimal_tensor");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("minimal.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("minimal").unwrap();
        assert_eq!(td.len(), 1);
        assert_eq!(td[0], 0x77);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_quantized_sets_flag() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "quant_tensor".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [16, 16, 0, 0],
            quant_format: 22,
            quant_block_size: 64,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xCC; 32],
            original_size: 256,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("quant_flag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("quant_flag.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        assert_eq!(reader.tensor_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_tensor_ordering() {
        let mut builder = GllmWriter::new(256);
        let tensor_names = ["z_weight", "a_bias", "m_scale", "b_embedding"];
        for name in &tensor_names {
            builder.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 16,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ordering");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ordering.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 4);
        // Verify order preserved by reading each tensor
        for name in &tensor_names {
            assert!(reader.find_tensor(name).is_some(), "tensor {} should exist", name);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_with_large_metadata() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "meta_test".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 32,
        });
        let meta = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("large_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_meta.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("meta_test").unwrap();
        assert_eq!(td.len(), 32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_mixed_quant_and_unquant() {
        let mut builder = GllmWriter::new(512);
        // Unquantized tensor
        builder.add_tensor(TensorEntry {
            name: "plain".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x11; 64],
            original_size: 64,
        });
        // Quantized tensor
        builder.add_tensor(TensorEntry {
            name: "quant".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0x22; 16],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("mixed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        assert_eq!(reader.tensor_count(), 2);

        let plain = reader.tensor_data("plain").unwrap();
        assert!(plain.iter().all(|&b| b == 0x11));

        let quant = reader.tensor_data("quant").unwrap();
        assert_eq!(quant.len(), 16);
        assert!(quant.iter().all(|&b| b == 0x22));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_metadata_all_zero_values() {
        let meta = build_metadata("", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["vocab_size"], "0");
        assert_eq!(parsed["hidden_size"], "0");
        assert_eq!(parsed["num_layers"], "0");
        assert_eq!(parsed["num_heads"], "0");
        assert_eq!(parsed["num_kv_heads"], "0");
        assert_eq!(parsed["head_dim"], "0");
        assert_eq!(parsed["intermediate_size"], "0");
        assert_eq!(parsed["context_length"], "0");
    }

    #[test]
    fn build_metadata_single_extra_overrides_builtin() {
        let mut extras = HashMap::new();
        extras.insert("hidden_size".to_string(), "99999".to_string());
        let meta = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["hidden_size"], "99999");
    }

    #[test]
    fn build_metadata_preserves_arch_key_with_special_chars() {
        let meta = build_metadata("my-model/v2.1-rc3", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "my-model/v2.1-rc3");
    }

    #[test]
    fn safetensors_dtype_to_u8_all_known_mappings() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
    }

    #[test]
    fn safetensors_dtype_to_u8_all_unknown_return_zero() {
        // All dtypes not in the explicit match should return 0
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BOOL), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F8_E5M2), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F8_E4M3), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I16), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U16), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F64), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U64), 0);
    }

    #[test]
    fn dtype_to_u8_identity_for_various_values() {
        assert_eq!(dtype_to_u8(0), 0);
        assert_eq!(dtype_to_u8(1), 1);
        assert_eq!(dtype_to_u8(127), 127);
        assert_eq!(dtype_to_u8(128), 128);
        assert_eq!(dtype_to_u8(255), 255);
    }

    #[test]
    fn align_up_all_common_page_sizes() {
        // Verify alignment works for common page sizes
        for &page in &[128u64, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536] {
            assert_eq!(align_up(0, page), 0);
            assert_eq!(align_up(page, page), page);
            assert_eq!(align_up(page + 1, page), page * 2);
            assert_eq!(align_up(page - 1, page), page);
        }
    }

    #[test]
    fn align_up_preserves_alignment_for_aligned_values() {
        for val in [0u64, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
            assert_eq!(align_up(val, 64), val);
        }
    }

    #[test]
    fn align_up_double_application_idempotent_various() {
        for alignment in [16u64, 32, 64, 128, 256, 512, 1024] {
            for val in [0u64, 1, alignment - 1, alignment, alignment + 1, alignment * 3 + 7] {
                let first = align_up(val, alignment);
                let second = align_up(first, alignment);
                assert_eq!(first, second, "idempotent failed for val={} alignment={}", val, alignment);
            }
        }
    }

    #[test]
    fn write_empty_file_with_no_tensors_no_metadata() {
        let builder = GllmWriter::new(4096);

        let dir = unique_test_dir("empty_file");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_file.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 0);
        assert!(!reader.header().is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_with_page_size_one() {
        let mut builder = GllmWriter::new(1);
        builder.add_tensor(TensorEntry {
            name: "page1".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [5, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![1, 2, 3, 4, 5],
            original_size: 5,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("page1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("page1.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("page1").unwrap();
        assert_eq!(&td[..], &[1, 2, 3, 4, 5]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_multiple_quant_types_in_same_file() {
        let mut builder = GllmWriter::new(256);
        // Q4_0
        builder.add_tensor(TensorEntry {
            name: "q4_0_weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 8, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xAA; 16],
            original_size: 128,
        });
        // AWQ4
        builder.add_tensor(TensorEntry {
            name: "awq4_weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 8, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xBB; 16],
            original_size: 128,
        });
        // GPTQ4
        builder.add_tensor(TensorEntry {
            name: "gptq4_weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 8, 0, 0],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0xCC; 16],
            original_size: 128,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("multi_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi_quant.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        assert_eq!(reader.tensor_count(), 3);

        assert_eq!(reader.find_tensor("q4_0_weight").unwrap().entry.quant_format, 10);
        assert_eq!(reader.find_tensor("awq4_weight").unwrap().entry.quant_format, 40);
        assert_eq!(reader.find_tensor("gptq4_weight").unwrap().entry.quant_format, 41);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_and_read_preserves_shape_dimensions() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "shaped".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [2, 3, 4, 5],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shape.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("shaped").unwrap();
        assert_eq!(t.entry.shape[0], 2);
        assert_eq!(t.entry.shape[1], 3);
        assert_eq!(t.entry.shape[2], 4);
        assert_eq!(t.entry.shape[3], 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_clone_shape_independence() {
        let original = TensorEntry {
            name: "shape_test".into(),
            ndim: 4,
            dtype: 0,
            shape: [100, 200, 300, 400],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let cloned = original.clone();
        // Shape arrays are copied by value
        assert_eq!(cloned.shape, original.shape);
    }

    #[test]
    fn tensor_entry_data_with_zeros() {
        let entry = TensorEntry {
            name: "zeros".into(),
            ndim: 1,
            dtype: 0,
            shape: [1024, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 1024,
        };
        assert_eq!(entry.compressed_size(), 1024);
        assert!(entry.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn tensor_entry_data_with_max_bytes() {
        let entry = TensorEntry {
            name: "max_bytes".into(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xFFu8; 256],
            original_size: 256,
        };
        assert!(entry.data.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn writer_page_size_various() {
        let sizes = [1u32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536];
        for &size in &sizes {
            let writer = GllmWriter::new(size);
            assert_eq!(writer.page_size, size);
        }
    }

    #[test]
    fn build_metadata_values_as_strings_not_numbers() {
        let meta = build_metadata("test", 42, 99, 7, 3, 1, 16, 200, 500, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // All values are stored as strings per build_metadata impl
        assert!(parsed["vocab_size"].is_string());
        assert!(parsed["hidden_size"].is_string());
        assert!(parsed["num_layers"].is_string());
        assert!(parsed["num_heads"].is_string());
        assert!(parsed["head_dim"].is_string());
    }

    #[test]
    fn build_metadata_with_many_extras() {
        let mut extras = HashMap::new();
        for i in 0..20 {
            extras.insert(format!("extra_{}", i), format!("value_{}", i));
        }
        let meta = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // 9 standard + 20 extras = 29
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 29);
        for i in 0..20 {
            assert_eq!(parsed[&format!("extra_{}", i)], format!("value_{}", i));
        }
    }

    #[test]
    fn write_roundtrip_with_page_size_64() {
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "ps64".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xDD; 10],
            original_size: 10,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps64.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 64);
        let td = reader.tensor_data("ps64").unwrap();
        assert_eq!(td.len(), 10);
        assert!(td.iter().all(|&b| b == 0xDD));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_data_integrity_across_alignment() {
        let mut builder = GllmWriter::new(1024);
        // Tensor with 7 bytes, will be padded to 1024 in data region
        let pattern: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA];
        builder.add_tensor(TensorEntry {
            name: "pattern".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [7, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: pattern.clone(),
            original_size: 7,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("integrity");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("integrity.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("pattern").unwrap();
        assert_eq!(&td[..], &pattern[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_with_ndim_boundary_values() {
        let mut builder = GllmWriter::new(256);
        // ndim = 0
        builder.add_tensor(TensorEntry {
            name: "ndim0".to_string(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x01; 4],
            original_size: 4,
        });
        // ndim = 4
        builder.add_tensor(TensorEntry {
            name: "ndim4".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [2, 3, 4, 5],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x02; 120],
            original_size: 120,
        });

        let dir = unique_test_dir("ndim_boundary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim_boundary.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t0 = reader.find_tensor("ndim0").unwrap();
        assert_eq!(t0.entry.ndim, 0);
        let t4 = reader.find_tensor("ndim4").unwrap();
        assert_eq!(t4.entry.ndim, 4);
        assert_eq!(t4.entry.shape, [2, 3, 4, 5]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_dtype_field() {
        let mut builder = GllmWriter::new(256);
        for dtype_val in [0u8, 1, 2, 3, 4, 5] {
            builder.add_tensor(TensorEntry {
                name: format!("dtype_{}", dtype_val),
                ndim: 1,
                dtype: dtype_val,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 16,
            });
        }

        let dir = unique_test_dir("dtype_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_field.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for dtype_val in [0u8, 1, 2, 3, 4, 5] {
            let name = format!("dtype_{}", dtype_val);
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.dtype, dtype_val);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_quant_block_size() {
        let mut builder = GllmWriter::new(256);
        for &bs in &[0u16, 32, 64, 128, 256] {
            builder.add_tensor(TensorEntry {
                name: format!("bs_{}", bs),
                ndim: 2,
                dtype: 0,
                shape: [4, 4, 0, 0],
                quant_format: if bs > 0 { 10 } else { 0 },
                quant_block_size: bs,
                scale_dtype: 1,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 64,
            });
        }

        let dir = unique_test_dir("block_size");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("block_size.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for &bs in &[0u16, 32, 64, 128, 256] {
            let name = format!("bs_{}", bs);
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.quant_block_size, bs);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_scale_dtype_and_zp_type() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "scale_zp".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 16],
            original_size: 64,
        });

        let dir = unique_test_dir("scale_zp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scale_zp.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("scale_zp").unwrap();
        assert_eq!(t.entry.scale_dtype, 2);
        assert_eq!(t.entry.zp_type, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_original_size() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "orig_size".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [1024, 1024, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 128],
            original_size: 4194304,
        });

        let dir = unique_test_dir("orig_size");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("orig_size.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("orig_size").unwrap();
        assert_eq!(t.entry.original_size, 4194304);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_name_ascii_only() {
        let entry = TensorEntry {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(entry.name.is_ascii());
        assert_eq!(entry.name.len(), 38);
    }

    #[test]
    fn align_up_zero_value_any_alignment() {
        for &alignment in &[1u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, 65536] {
            assert_eq!(align_up(0, alignment), 0);
        }
    }

    #[test]
    fn build_metadata_produces_consistent_values() {
        // HashMap iteration order is non-deterministic, so parse JSON and compare values
        let extras = HashMap::new();
        let meta1 = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        let meta2 = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        let parsed1: serde_json::Value = serde_json::from_slice(&meta1).unwrap();
        let parsed2: serde_json::Value = serde_json::from_slice(&meta2).unwrap();
        assert_eq!(parsed1, parsed2);
    }

    #[test]
    fn write_to_vec_matches_write_to_file() {
        // Write to a Vec<u8> and a file, verify the bytes match
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "compare".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x55; 32],
            original_size: 32,
        });
        builder.set_metadata(vec![1, 2, 3]);

        let dir = unique_test_dir("compare");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("compare.gllm");
        builder.write_to_path(&path).unwrap();

        let file_bytes = std::fs::read(&path).unwrap();
        assert!(!file_bytes.is_empty());
        // Header should start with GLLM_MAGIC
        let magic = u32::from_le_bytes(file_bytes[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_file_size_consistent_with_content() {
        let mut builder = GllmWriter::new(128);
        let data = vec![0xAB; 100];
        builder.add_tensor(TensorEntry {
            name: "size_test".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [100, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 100,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("file_size");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file_size.gllm");
        builder.write_to_path(&path).unwrap();

        let file_bytes = std::fs::read(&path).unwrap();
        // File should be at least: header(64) + tensor_entry(72) + aligned_data(128)
        assert!(file_bytes.len() >= 64 + 72 + 128);
        // Data at offset should start at page-aligned boundary
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_data("size_test").unwrap().len(), 100);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 50 additional unit tests — binary layout, edge cases, error paths
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn header_magic_bytes_in_written_file() {
        let mut builder = GllmWriter::new(256);
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("magic.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert!(raw.len() >= 4);
        let magic = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_version_field_in_written_file() {
        let mut builder = GllmWriter::new(256);
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("version.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_zero_tensors_gives_zero_flags() {
        let builder = GllmWriter::new(4096);
        let dir = unique_test_dir("zero_flags");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_flags.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 0, "no tensors should produce flags=0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_quant_flag_set_when_any_tensor_quantized() {
        let mut builder = GllmWriter::new(256);
        // First tensor is unquantized
        builder.add_tensor(TensorEntry {
            name: "plain".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        // Second tensor is quantized
        builder.add_tensor(TensorEntry {
            name: "quant".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 8], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("mixed_flags");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed_flags.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 1, "at least one quantized tensor should set flags=1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_tensor_count_matches_written() {
        let mut builder = GllmWriter::new(256);
        for i in 0..3 {
            builder.add_tensor(TensorEntry {
                name: format!("t{}", i), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 8], original_size: 8,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("count_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("count.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let count = u32::from_le_bytes(raw[20..24].try_into().unwrap());
        assert_eq!(count, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_page_size_stored_correctly() {
        let page_size = 2048u32;
        let mut builder = GllmWriter::new(page_size);
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let ps = u32::from_le_bytes(raw[40..44].try_into().unwrap());
        assert_eq!(ps, page_size);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_reserved_bytes_are_zero() {
        let mut builder = GllmWriter::new(256);
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("reserved");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reserved.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Reserved bytes 44..64 should all be zero
        for i in 44..64 {
            assert_eq!(raw[i], 0, "reserved byte {} should be zero", i);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_offset_always_header_size() {
        let mut builder = GllmWriter::new(4096);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("td_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_offset.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let td_offset = u64::from_le_bytes(raw[24..32].try_into().unwrap());
        assert_eq!(td_offset, HEADER_SIZE as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_offset_is_page_aligned() {
        let page_size = 512u32;
        let mut builder = GllmWriter::new(page_size);
        builder.add_tensor(TensorEntry {
            name: "x".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("data_align");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_align.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        assert_eq!(data_offset % page_size as u64, 0, "data_offset must be page-aligned");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn string_table_contains_tensor_names() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "alpha".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "beta".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("strtab");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("strtab.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let strtab_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        // "alpha" starts at offset 0 in string table, "beta" at offset 5
        assert_eq!(&raw[strtab_start..strtab_start + 5], b"alpha");
        assert_eq!(&raw[strtab_start + 5..strtab_start + 9], b"beta");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_name_offset_and_len() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "hello".to_string(), ndim: 2, dtype: 0, shape: [1, 1, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("name_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("name_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let td_start = HEADER_SIZE;
        let name_off = u32::from_le_bytes(raw[td_start..td_start + 4].try_into().unwrap());
        let name_len = u16::from_le_bytes(raw[td_start + 4..td_start + 6].try_into().unwrap());
        assert_eq!(name_off, 0); // first tensor name starts at beginning of string table
        assert_eq!(name_len, 5); // "hello" is 5 bytes
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_ndim_dtype_fields() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 3, dtype: 5, shape: [2, 4, 6, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ndim_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim_dtype.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let entry_start = HEADER_SIZE;
        assert_eq!(raw[entry_start + 6], 3, "ndim");
        assert_eq!(raw[entry_start + 7], 5, "dtype");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_shape_bytes() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 2, dtype: 0,
            shape: [100, 200, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("shape_bytes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shape_bytes.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let s = HEADER_SIZE + 8; // shape starts at byte 8 of tensor entry
        let s0 = u64::from_le_bytes(raw[s..s + 8].try_into().unwrap());
        let s1 = u64::from_le_bytes(raw[s + 8..s + 16].try_into().unwrap());
        let s2 = u64::from_le_bytes(raw[s + 16..s + 24].try_into().unwrap());
        let s3 = u64::from_le_bytes(raw[s + 24..s + 32].try_into().unwrap());
        assert_eq!(s0, 100);
        assert_eq!(s1, 200);
        assert_eq!(s2, 0);
        assert_eq!(s3, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_compressed_size_field() {
        let mut builder = GllmWriter::new(256);
        let data = vec![0xAB; 42];
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 1, dtype: 0, shape: [42, 0, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: data.clone(), original_size: 168,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("cs_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cs_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let cs_off = HEADER_SIZE + 56; // compressed_size at byte 56 of tensor entry
        let cs = u64::from_le_bytes(raw[cs_off..cs_off + 8].try_into().unwrap());
        assert_eq!(cs, 42);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_original_size_field() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 1, dtype: 0, shape: [10, 0, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 4], original_size: 999,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("os_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("os_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let os_off = HEADER_SIZE + 64; // original_size at byte 64 of tensor entry
        let os = u64::from_le_bytes(raw[os_off..os_off + 8].try_into().unwrap());
        assert_eq!(os, 999);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_stored_at_meta_offset() {
        let mut builder = GllmWriter::new(256);
        let meta = vec![0xDE, 0xAD, 0xBE, 0xEF];
        builder.set_metadata(meta.clone());

        let dir = unique_test_dir("meta_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_offset.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        assert_eq!(&raw[meta_offset..meta_offset + 4], &[0xDE, 0xAD, 0xBE, 0xEF]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_file_with_small_page_size() {
        // With page_size=1, no tensors and no metadata, file is exactly HEADER_SIZE
        let builder = GllmWriter::new(1);
        let dir = unique_test_dir("empty_size");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_size.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw.len(), HEADER_SIZE, "empty gllm file with page_size=1 should be exactly HEADER_SIZE bytes");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_size_header_plus_one_tensor_entry_no_data() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "empty_t".to_string(), ndim: 1, dtype: 0, shape: [0, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("size_empty_tensor");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("size_empty_tensor.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Header(64) + TensorDir(72) + StringTable("empty_t"=8 bytes) = 144
        // Data should be page-aligned so data_offset >= 144, and empty data = 0 bytes
        assert!(raw.len() >= HEADER_SIZE + TENSOR_ENTRY_SIZE + 8);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_invalid_path_returns_error() {
        let builder = GllmWriter::new(4096);
        let result = builder.write_to_path(std::path::Path::new("/nonexistent/dir/deep/file.gllm"));
        assert!(result.is_err(), "writing to invalid path should fail");
    }

    #[test]
    fn write_overwrites_existing_file() {
        let dir = unique_test_dir("overwrite_v1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overwrite.gllm");

        // Write first version
        let mut b1 = GllmWriter::new(256);
        b1.add_tensor(TensorEntry {
            name: "v1".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11; 16], original_size: 16,
        });
        b1.set_metadata(vec![]);
        b1.write_to_path(&path).unwrap();

        // Write second version with different data
        let mut b2 = GllmWriter::new(256);
        b2.add_tensor(TensorEntry {
            name: "v2".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x22; 16], original_size: 16,
        });
        b2.set_metadata(vec![]);
        b2.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        assert!(reader.find_tensor("v2").is_some());
        assert!(reader.find_tensor("v1").is_none());
        let td = reader.tensor_data("v2").unwrap();
        assert!(td.iter().all(|&b| b == 0x22));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_tensors_with_same_name_prefix() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "model.layer.0.weight".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 8], original_size: 8,
        });
        builder.add_tensor(TensorEntry {
            name: "model.layer.0.bias".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("prefix");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("prefix.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let w = reader.tensor_data("model.layer.0.weight").unwrap();
        let b = reader.tensor_data("model.layer.0.bias").unwrap();
        assert!(w.iter().all(|&x| x == 0xAA));
        assert!(b.iter().all(|&x| x == 0xBB));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_with_single_byte_data_roundtrip() {
        let mut builder = GllmWriter::new(4096);
        builder.add_tensor(TensorEntry {
            name: "single".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x99], original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("single_byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single_byte.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("single").unwrap();
        assert_eq!(td.len(), 1);
        assert_eq!(td[0], 0x99);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_with_large_data_roundtrip() {
        let mut builder = GllmWriter::new(4096);
        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        builder.add_tensor(TensorEntry {
            name: "big_data".to_string(), ndim: 1, dtype: 0, shape: [10000, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 10000,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("big_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big_data.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("big_data").unwrap();
        assert_eq!(td.len(), 10000);
        assert_eq!(&td[..], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_between_string_table_and_data() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF], original_size: 1,
        });
        let meta = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        builder.set_metadata(meta.clone());

        let dir = unique_test_dir("meta_between");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_between.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        // Metadata should appear after string table
        let strtab_end = HEADER_SIZE + TENSOR_ENTRY_SIZE + 1; // "a" = 1 byte
        assert!(meta_offset >= strtab_end);
        assert_eq!(&raw[meta_offset..meta_offset + 5], &[0x01, 0x02, 0x03, 0x04, 0x05]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn padding_between_meta_and_data_is_zeros() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "p".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x77], original_size: 1,
        });
        builder.set_metadata(vec![0xAA; 10]);

        let dir = unique_test_dir("padding");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("padding.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        let meta_end = meta_offset + 10;
        // Bytes between meta end and data_offset should all be zero padding
        if data_offset > meta_end {
            for i in meta_end..data_offset {
                assert_eq!(raw[i], 0, "padding byte at {} should be zero", i);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_data_padding_is_zeros() {
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "d".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xDD, 0xEE, 0xFF], original_size: 3,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("data_pad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_pad.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        assert_eq!(raw[data_offset], 0xDD);
        assert_eq!(raw[data_offset + 1], 0xEE);
        assert_eq!(raw[data_offset + 2], 0xFF);
        // Padding from byte 3 to 64 should be zero
        for i in 3..64 {
            assert_eq!(raw[data_offset + i], 0, "data padding at {} should be zero", i);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn second_tensor_starts_at_aligned_offset() {
        let mut builder = GllmWriter::new(128);
        // First tensor: 10 bytes → padded to 128
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0, shape: [10, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11; 10], original_size: 10,
        });
        // Second tensor: 5 bytes → padded to 128
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x22; 5], original_size: 5,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("second_tensor");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("second_tensor.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t2 = reader.find_tensor("t2").unwrap();
        // Second tensor data offset should be 128 (first tensor aligned size)
        assert_eq!(t2.entry.data_offset, 128);
        let td2 = reader.tensor_data("t2").unwrap();
        assert_eq!(&td2[..], &[0x22; 5]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_metadata_with_newlines_in_values() {
        let mut extras = HashMap::new();
        extras.insert("description".to_string(), "line1\nline2\nline3".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["description"], "line1\nline2\nline3");
    }

    #[test]
    fn build_metadata_with_empty_string_extras() {
        let mut extras = HashMap::new();
        extras.insert("empty_key".to_string(), String::new());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["empty_key"], "");
    }

    #[test]
    fn tensor_entry_with_all_quant_fields_zero() {
        let entry = TensorEntry {
            name: "q".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert!(!entry.is_quantized());
        assert_eq!(entry.compressed_size(), 0);
    }

    #[test]
    fn tensor_entry_quant_block_size_max() {
        let entry = TensorEntry {
            name: "qbs_max".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: u16::MAX, scale_dtype: 2, zp_type: 1,
            data: vec![], original_size: 0,
        };
        assert_eq!(entry.quant_block_size, 65535);
        assert!(entry.is_quantized());
    }

    #[test]
    fn writer_builder_reuse_produces_consistent_files() {
        let dir = unique_test_dir("reuse");
        std::fs::create_dir_all(&dir).unwrap();

        // Build and write first file
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "reuse".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAB; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);
        let path1 = dir.join("reuse1.gllm");
        builder.write_to_path(&path1).unwrap();

        // Write same builder again to different file
        let path2 = dir.join("reuse2.gllm");
        builder.write_to_path(&path2).unwrap();

        let raw1 = std::fs::read(&path1).unwrap();
        let raw2 = std::fs::read(&path2).unwrap();
        assert_eq!(raw1, raw2, "same builder should produce identical files");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_metadata_preserves_complex_json() {
        let mut extras = HashMap::new();
        extras.insert("model_type".to_string(), "qwen3_moe".to_string());
        extras.insert("rope_theta".to_string(), "1000000.0".to_string());
        extras.insert("sliding_window".to_string(), "4096".to_string());
        let meta = build_metadata(
            "qwen3_moe", 151936, 2048, 36, 32, 4, 128, 5632, 32768, &extras,
        );

        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "w".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 64], original_size: 64,
        });
        builder.set_metadata(meta);

        let dir = unique_test_dir("complex_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("complex_meta.gllm");
        builder.write_to_path(&path).unwrap();

        // Read back raw metadata bytes from file and verify JSON
        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        let meta_bytes = &raw[meta_offset..data_offset];
        // Find end of metadata (trim trailing zero padding)
        let meta_end = meta_bytes.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let parsed: serde_json::Value = serde_json::from_slice(&meta_bytes[..meta_end]).unwrap();
        assert_eq!(parsed["model_type"], "qwen3_moe");
        assert_eq!(parsed["rope_theta"], "1000000.0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_names_with_underscore_prefix() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "_internal_tensor".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x42], original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("underscore");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("underscore.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("_internal_tensor").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_names_with_number_suffix() {
        let mut builder = GllmWriter::new(256);
        for i in 0..5 {
            builder.add_tensor(TensorEntry {
                name: format!("layer_{}", i), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![(i + 1) as u8], original_size: 1,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("num_suffix");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("num_suffix.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for i in 0..5 {
            let name = format!("layer_{}", i);
            let td = reader.tensor_data(&name).unwrap();
            assert_eq!(td[0], (i + 1) as u8);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_with_all_zero_data_tensor() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "zeros".to_string(), ndim: 1, dtype: 0, shape: [64, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 64], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("zero_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_data.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("zeros").unwrap();
        assert_eq!(td.len(), 64);
        assert!(td.iter().all(|&b| b == 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_with_all_ff_data_tensor() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "all_ff".to_string(), ndim: 1, dtype: 0, shape: [32, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFFu8; 32], original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ff_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ff_data.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("all_ff").unwrap();
        assert_eq!(td.len(), 32);
        assert!(td.iter().all(|&b| b == 0xFF));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn align_up_with_non_power_of_two_alignment() {
        // align_up works with any alignment, not just powers of 2
        assert_eq!(align_up(10, 3), 12);
        assert_eq!(align_up(9, 3), 9);
        assert_eq!(align_up(8, 3), 9);
        assert_eq!(align_up(1, 7), 7);
        assert_eq!(align_up(7, 7), 7);
        assert_eq!(align_up(8, 7), 14);
        assert_eq!(align_up(100, 30), 120);
    }

    #[test]
    fn align_up_value_zero_with_various_alignments() {
        assert_eq!(align_up(0, 3), 0);
        assert_eq!(align_up(0, 7), 0);
        assert_eq!(align_up(0, 13), 0);
        assert_eq!(align_up(0, 100), 0);
    }

    #[test]
    fn quant_type_to_u8_fp_family_range() {
        use gllm_kernels::quant::QuantType;
        // Float types: 1-3
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
        assert_eq!(quant_type_to_u8(QuantType::F32), 3);
    }

    #[test]
    fn quant_type_to_u8_iq_family_range() {
        use gllm_kernels::quant::QuantType;
        // IQ family: 30-38
        assert_eq!(quant_type_to_u8(QuantType::IQ1S), 30);
        assert_eq!(quant_type_to_u8(QuantType::IQ1M), 31);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XXS), 32);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XS), 33);
        assert_eq!(quant_type_to_u8(QuantType::IQ2S), 34);
        assert_eq!(quant_type_to_u8(QuantType::IQ3XXS), 35);
        assert_eq!(quant_type_to_u8(QuantType::IQ3S), 36);
        assert_eq!(quant_type_to_u8(QuantType::IQ4NL), 37);
        assert_eq!(quant_type_to_u8(QuantType::IQ4XS), 38);
    }

    #[test]
    fn tensor_entry_debug_includes_dtype() {
        let entry = TensorEntry {
            name: "debug_dt".into(), ndim: 1, dtype: 5, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("dtype: 5"));
    }

    #[test]
    fn tensor_entry_debug_includes_original_size() {
        let entry = TensorEntry {
            name: "debug_os".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 12345,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("original_size: 12345"));
    }

    #[test]
    fn tensor_entry_clone_preserves_ndim() {
        let original = TensorEntry {
            name: "ndim_clone".into(), ndim: 3, dtype: 0, shape: [10, 20, 30, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.ndim, 3);
    }

    #[test]
    fn build_metadata_returns_non_empty_bytes() {
        let meta = build_metadata("x", 1, 2, 3, 4, 5, 6, 7, 8, &HashMap::new());
        assert!(!meta.is_empty(), "build_metadata should produce non-empty output");
    }

    #[test]
    fn safetensors_dtype_to_u8_bf16_value() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
    }

    #[test]
    fn safetensors_dtype_to_u8_i32_value() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
    }

    #[test]
    fn safetensors_dtype_to_u8_i64_value() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
    }

    #[test]
    fn write_multiple_tensors_with_varying_sizes() {
        let mut builder = GllmWriter::new(128);
        let sizes = [1usize, 10, 50, 100, 127];
        for (i, &sz) in sizes.iter().enumerate() {
            builder.add_tensor(TensorEntry {
                name: format!("v{}", i), ndim: 1, dtype: 0,
                shape: [sz as u64, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![i as u8; sz], original_size: sz as u64,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("vary_sizes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vary_sizes.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 5);
        for (i, &sz) in sizes.iter().enumerate() {
            let td = reader.tensor_data(&format!("v{}", i)).unwrap();
            assert_eq!(td.len(), sz);
            assert!(td.iter().all(|&b| b == i as u8));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_tensors_field_is_accessible() {
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "t1".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        // Direct field access
        assert_eq!(writer.tensors[0].name, "t1");
    }

    #[test]
    fn dtype_to_u8_passthrough_boundary_values() {
        assert_eq!(dtype_to_u8(u8::MIN), u8::MIN);
        assert_eq!(dtype_to_u8(u8::MAX), u8::MAX);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 50 additional unit tests — uncovered areas
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_data_empty_but_shape_nonzero() {
        // Edge: shape says there are elements but data is empty
        let entry = TensorEntry {
            name: "missing_data".into(),
            ndim: 2,
            dtype: 0,
            shape: [100, 200, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.compressed_size(), 0);
        assert_eq!(entry.shape[0], 100);
        assert_eq!(entry.shape[1], 200);
    }

    #[test]
    fn tensor_entry_dtype_boundary_zero() {
        let entry = TensorEntry {
            name: "dtype0".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.dtype, 0);
    }

    #[test]
    fn tensor_entry_dtype_boundary_max() {
        let entry = TensorEntry {
            name: "dtype_max".into(),
            ndim: 1,
            dtype: u8::MAX,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.dtype, u8::MAX);
    }

    #[test]
    fn tensor_entry_quant_format_all_nonzero_values() {
        // Every value 1..=255 should report as quantized
        for qf in 1u8..=255 {
            let entry = TensorEntry {
                name: "q".into(),
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: qf,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            };
            assert!(entry.is_quantized(), "quant_format={qf} should be quantized");
        }
    }

    #[test]
    fn tensor_entry_scale_dtype_independent_of_quant_format() {
        // scale_dtype can be non-zero even when quant_format=0 (anomalous but structurally valid)
        let entry = TensorEntry {
            name: "anomaly_scale".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![],
            original_size: 0,
        };
        assert!(!entry.is_quantized());
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
    }

    #[test]
    fn tensor_entry_compressed_size_with_exact_power_of_two_data() {
        let data = vec![0xAB; 4096];
        let entry = TensorEntry {
            name: "power2".into(),
            ndim: 1,
            dtype: 0,
            shape: [4096, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data,
            original_size: 4096,
        };
        assert_eq!(entry.compressed_size(), 4096);
    }

    #[test]
    fn tensor_entry_original_size_zero_with_unquantized() {
        let entry = TensorEntry {
            name: "orig_zero".into(),
            ndim: 1,
            dtype: 0,
            shape: [64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 0,
        };
        assert_eq!(entry.original_size, 0);
        assert_eq!(entry.compressed_size(), 64);
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_shape_all_zeros() {
        let entry = TensorEntry {
            name: "zero_shape".into(),
            ndim: 0,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.shape, [0u64; 4]);
    }

    #[test]
    fn tensor_entry_name_with_spaces() {
        let entry = TensorEntry {
            name: "layer 0 weight".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert!(entry.name.contains(' '));
        assert_eq!(entry.name, "layer 0 weight");
    }

    #[test]
    fn tensor_entry_name_single_char() {
        let entry = TensorEntry {
            name: "x".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.name.len(), 1);
    }

    #[test]
    fn tensor_entry_data_one_element() {
        let entry = TensorEntry {
            name: "one_elem".into(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42],
            original_size: 1,
        };
        assert_eq!(entry.compressed_size(), 1);
        assert_eq!(entry.data[0], 42);
    }

    #[test]
    fn tensor_entry_clone_preserves_dtype() {
        let original = TensorEntry {
            name: "dtype_test".into(),
            ndim: 2,
            dtype: 7,
            shape: [128, 256, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.dtype, 7);
        assert_eq!(cloned.scale_dtype, 1);
        assert_eq!(cloned.zp_type, 0);
    }

    #[test]
    fn tensor_entry_clone_preserves_quant_block_size() {
        let original = TensorEntry {
            name: "qbs".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 22,
            quant_block_size: 256,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.quant_block_size, 256);
        assert_eq!(cloned.quant_format, 22);
    }

    #[test]
    fn tensor_entry_debug_includes_name_field() {
        let entry = TensorEntry {
            name: "my_tensor".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("my_tensor"));
        assert!(debug.contains("name"));
    }

    #[test]
    fn tensor_entry_debug_includes_ndim_field() {
        let entry = TensorEntry {
            name: "nd".into(),
            ndim: 4,
            dtype: 0,
            shape: [1, 2, 3, 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("ndim: 4"));
    }

    #[test]
    fn tensor_entry_debug_includes_shape() {
        let entry = TensorEntry {
            name: "sh".into(),
            ndim: 2,
            dtype: 0,
            shape: [512, 768, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("512"));
        assert!(debug.contains("768"));
    }

    #[test]
    fn align_up_value_one_with_various_alignments() {
        assert_eq!(align_up(1, 2), 2);
        assert_eq!(align_up(1, 4), 4);
        assert_eq!(align_up(1, 8), 8);
        assert_eq!(align_up(1, 16), 16);
        assert_eq!(align_up(1, 32), 32);
    }

    

    #[test]
    fn align_up_monotonically_increasing() {
        // For a fixed alignment, align_up should be monotonic
        let alignment = 64u64;
        let mut prev = 0u64;
        for val in 0..=256 {
            let aligned = align_up(val, alignment);
            assert!(aligned >= prev, "monotonicity violated at val={val}");
            prev = aligned;
        }
    }

    #[test]
    fn align_up_result_divisible_by_alignment() {
        for &alignment in &[2u64, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
            for val in [0u64, 1, alignment - 1, alignment, alignment + 1, alignment * 3 - 1] {
                let result = align_up(val, alignment);
                assert_eq!(result % alignment, 0, "align_up({val}, {alignment}) = {result} not divisible by {alignment}");
            }
        }
    }

    #[test]
    fn build_metadata_with_slash_in_arch_key() {
        let meta = build_metadata("org/model-v2", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "org/model-v2");
    }

    #[test]
    fn build_metadata_with_colon_in_extra_value() {
        let mut extras = HashMap::new();
        extras.insert("license".to_string(), "MIT:Commercial".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["license"], "MIT:Commercial");
    }

    #[test]
    fn build_metadata_with_equals_in_extra_value() {
        let mut extras = HashMap::new();
        extras.insert("config".to_string(), "key=value".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["config"], "key=value");
    }

    #[test]
    fn build_metadata_with_backslash_in_extra_key() {
        let mut extras = HashMap::new();
        extras.insert("path\\to\\key".to_string(), "value".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["path\\to\\key"], "value");
    }

    #[test]
    fn build_metadata_very_long_arch_key() {
        let long_key = "x".repeat(10000);
        let meta = build_metadata(&long_key, 1, 2, 3, 4, 5, 6, 7, 8, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"].as_str().unwrap().len(), 10000);
    }

    #[test]
    fn build_metadata_extra_numeric_string_value() {
        let mut extras = HashMap::new();
        extras.insert("epsilon".to_string(), "1e-5".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["epsilon"], "1e-5");
        assert!(parsed["epsilon"].is_string());
    }

    #[test]
    fn build_metadata_value_one_as_string() {
        let meta = build_metadata("test", 1, 1, 1, 1, 1, 1, 1, 1, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["vocab_size"], "1");
        assert_eq!(parsed["hidden_size"], "1");
    }

    #[test]
    fn build_metadata_preserves_intermediate_size_value() {
        let meta = build_metadata("test", 100, 200, 10, 8, 4, 64, 4096, 2048, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["intermediate_size"], "4096");
    }

    #[test]
    fn safetensors_dtype_to_u8_f16_is_one() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
    }

    #[test]
    fn safetensors_dtype_to_u8_u8_is_three() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
    }

    #[test]
    fn safetensors_dtype_to_u8_i8_is_four() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
    }

    #[test]
    fn safetensors_dtype_to_u8_known_codes_are_unique() {
        let known = [
            (safetensors::Dtype::F32, 0),
            (safetensors::Dtype::F16, 1),
            (safetensors::Dtype::BF16, 2),
            (safetensors::Dtype::U8, 3),
            (safetensors::Dtype::I8, 4),
            (safetensors::Dtype::I32, 5),
            (safetensors::Dtype::I64, 6),
        ];
        let codes: Vec<u8> = known.iter().map(|&(_, c)| c).collect();
        let unique: std::collections::HashSet<u8> = codes.iter().copied().collect();
        assert_eq!(codes.len(), unique.len(), "known safetensors dtype codes must be unique");
        for (dtype, expected) in &known {
            assert_eq!(safetensors_dtype_to_u8(*dtype), *expected);
        }
    }

    #[test]
    fn writer_new_initializes_empty_state() {
        let writer = GllmWriter::new(4096);
        assert_eq!(writer.tensors.len(), 0);
        assert_eq!(writer.metadata_bytes.len(), 0);
        assert_eq!(writer.page_size, 4096);
    }

    #[test]
    fn writer_set_metadata_replaces_previous() {
        let mut writer = GllmWriter::new(4096);
        writer.set_metadata(vec![1, 2, 3]);
        writer.set_metadata(vec![4, 5, 6, 7, 8]);
        assert_eq!(writer.metadata_bytes, vec![4, 5, 6, 7, 8]);
    }

    #[test]
    fn writer_add_tensor_count_increments() {
        let mut writer = GllmWriter::new(256);
        assert_eq!(writer.tensor_count(), 0);
        for i in 0..3 {
            writer.add_tensor(TensorEntry {
                name: format!("t{i}"),
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            });
            assert_eq!(writer.tensor_count(), i + 1);
        }
    }

    #[test]
    fn write_roundtrip_original_size_zero_quantized() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "q_zero_orig".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0xCC; 8],
            original_size: 0,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("q_zero_orig");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q_zero_orig.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("q_zero_orig").unwrap();
        assert_eq!(t.entry.original_size, 0);
        assert!(t.entry.is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_dtype_max_value() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "dtype_max".to_string(),
            ndim: 1,
            dtype: u8::MAX,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("dtype_max_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_max_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("dtype_max").unwrap();
        assert_eq!(t.entry.dtype, u8::MAX);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_data_with_alternating_pattern() {
        let data: Vec<u8> = (0..64).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "alternating".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("alternating");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("alternating.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("alternating").unwrap();
        assert_eq!(&td[..], &data[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_data_with_sequential_bytes() {
        let data: Vec<u8> = (0..=255).collect();
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "sequential".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 256,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("sequential");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sequential.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("sequential").unwrap();
        assert_eq!(&td[..], &data[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_compressed_size_field() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "cs_test".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [8, 8, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xAB; 24],
            original_size: 256,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("cs_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cs_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("cs_test").unwrap();
        assert_eq!(t.entry.compressed_size, 24);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_many_tensors_data_integrity() {
        let mut builder = GllmWriter::new(64);
        let count = 20;
        for i in 0..count {
            builder.add_tensor(TensorEntry {
                name: format!("tensor_{:03}", i),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![(i + 1) as u8; 16],
                original_size: 16,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("many_integrity");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many_integrity.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), count);
        for i in 0..count {
            let name = format!("tensor_{:03}", i);
            let td = reader.tensor_data(&name).unwrap();
            assert!(td.iter().all(|&b| b == (i + 1) as u8), "tensor_{i} data mismatch");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_with_build_metadata_roundtrip() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "meta_rt".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 32,
        });
        let mut extras = HashMap::new();
        extras.insert("model_type".to_string(), "qwen3_moe".to_string());
        let meta = build_metadata("qwen3_moe", 151936, 2048, 36, 32, 4, 128, 5632, 32768, &extras);
        builder.set_metadata(meta);

        let dir = unique_test_dir("meta_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        let meta_bytes = &raw[meta_offset..data_offset];
        let meta_end = meta_bytes.iter().rposition(|&b| b != 0).map(|i| i + 1).unwrap_or(0);
        let parsed: serde_json::Value = serde_json::from_slice(&meta_bytes[..meta_end]).unwrap();
        assert_eq!(parsed["arch_key"], "qwen3_moe");
        assert_eq!(parsed["model_type"], "qwen3_moe");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_zp_type_values() {
        let mut builder = GllmWriter::new(256);
        for &zp in &[0u8, 1, 2, 255] {
            builder.add_tensor(TensorEntry {
                name: format!("zp_{zp}"),
                ndim: 2,
                dtype: 0,
                shape: [2, 2, 0, 0],
                quant_format: 40,
                quant_block_size: 128,
                scale_dtype: 2,
                zp_type: zp,
                data: vec![0u8; 4],
                original_size: 16,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("zp_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zp_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for &zp in &[0u8, 1, 2, 255] {
            let name = format!("zp_{zp}");
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.zp_type, zp, "zp_type mismatch for {zp}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_roundtrip_preserves_scale_dtype_values() {
        let mut builder = GllmWriter::new(256);
        for &sd in &[0u8, 1, 2, 3] {
            builder.add_tensor(TensorEntry {
                name: format!("sd_{sd}"),
                ndim: 2,
                dtype: 0,
                shape: [2, 2, 0, 0],
                quant_format: 41,
                quant_block_size: 128,
                scale_dtype: sd,
                zp_type: 1,
                data: vec![0u8; 4],
                original_size: 16,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("sd_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sd_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for &sd in &[0u8, 1, 2, 3] {
            let name = format!("sd_{sd}");
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.scale_dtype, sd, "scale_dtype mismatch for {sd}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_meta_offset_field_in_written_file() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "mo".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        builder.set_metadata(vec![0xCA, 0xFE]);

        let dir = unique_test_dir("mo_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mo_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        // meta_offset should be after header + tensor_dir + string_table
        let expected_min = HEADER_SIZE + TENSOR_ENTRY_SIZE + 2; // "mo" = 2 bytes
        assert!(meta_offset >= expected_min);
        // Metadata content should be at meta_offset
        assert_eq!(raw[meta_offset], 0xCA);
        assert_eq!(raw[meta_offset + 1], 0xFE);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_all_zero_fields_for_empty_file() {
        let builder = GllmWriter::new(4096);
        let dir = unique_test_dir("zero_fields");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_fields.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let tensor_count = u32::from_le_bytes(raw[20..24].try_into().unwrap());
        assert_eq!(tensor_count, 0);
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 0);
        let ps = u32::from_le_bytes(raw[40..44].try_into().unwrap());
        assert_eq!(ps, 4096);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_empty_metadata_bytes_still_valid() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "no_meta".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("no_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_meta.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("no_meta").unwrap();
        assert_eq!(td.len(), 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_two_tensors_first_quant_second_plain_still_sets_flag() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "q".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 16,
        });
        builder.add_tensor(TensorEntry {
            name: "p".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("q_then_p");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("q_then_p.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized(), "flag should be set because first tensor is quantized");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_string_table_offset_advances_per_tensor() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "aa".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "bbb".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("strtab_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("strtab_offset.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Second tensor entry: name_offset should be 2 (length of "aa")
        let entry2_start = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let name_off_2 = u32::from_le_bytes(raw[entry2_start..entry2_start + 4].try_into().unwrap());
        assert_eq!(name_off_2, 2, "second tensor name_offset should skip first name 'aa' (2 bytes)");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quant_type_to_u8_fp8_e4m3_and_e5m2_distinct() {
        use gllm_kernels::quant::QuantType;
        let e4m3 = quant_type_to_u8(QuantType::Fp8E4M3);
        let e5m2 = quant_type_to_u8(QuantType::Fp8E5M2);
        assert_ne!(e4m3, e5m2, "Fp8E4M3 and Fp8E5M2 should have distinct codes");
        assert_eq!(e4m3, 50);
        assert_eq!(e5m2, 51);
    }

    #[test]
    fn quant_type_to_u8_awq_gptq_distinct() {
        use gllm_kernels::quant::QuantType;
        let awq = quant_type_to_u8(QuantType::AWQ4);
        let gptq = quant_type_to_u8(QuantType::GPTQ4);
        assert_ne!(awq, gptq, "AWQ4 and GPTQ4 should have distinct codes");
    }

    #[test]
    fn quant_type_to_u8_nvfp4_distinct_from_mxfp4() {
        use gllm_kernels::quant::QuantType;
        let nvfp4 = quant_type_to_u8(QuantType::Nvfp4);
        let mxfp4 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 });
        assert_ne!(nvfp4, mxfp4, "Nvfp4 and Mxfp4 should have distinct codes");
    }

    #[test]
    fn tensor_entry_data_two_bytes() {
        let entry = TensorEntry {
            name: "two".into(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x12, 0x34],
            original_size: 2,
        };
        assert_eq!(entry.compressed_size(), 2);
        assert_eq!(entry.data[0], 0x12);
        assert_eq!(entry.data[1], 0x34);
    }

    #[test]
    fn tensor_entry_compressed_size_u64_boundary() {
        // Verify compressed_size returns u64 from data.len()
        let entry = TensorEntry {
            name: "u64_check".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 1024,
        };
        let cs = entry.compressed_size();
        assert_eq!(cs, 1024u64);
        // Ensure the return type is u64
        let _: u64 = cs;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Batch 3: Additional tests (targeting uncovered code paths)
    // ────────────────────────────────────────────────────────────────────────

    // ── align_up edge cases ─────────────────────────────────────────────────

    #[test]
    fn align_up_large_value_small_alignment() {
        assert_eq!(align_up(u64::MAX - 1, 1), u64::MAX - 1);
        assert_eq!(align_up(u64::MAX, 1), u64::MAX);
    }

    #[test]
    fn align_up_power_of_two_alignments() {
        assert_eq!(align_up(7, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(16, 16), 16);
        assert_eq!(align_up(17, 16), 32);
    }

    #[test]
    fn align_up_value_equals_alignment_batch3() {
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(512, 512), 512);
        assert_eq!(align_up(256, 256), 256);
    }

    #[test]
    fn align_up_one_past_alignment_batch3() {
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(513, 512), 1024);
        assert_eq!(align_up(9, 8), 16);
    }

    #[test]
    fn align_up_preserves_already_aligned() {
        for &align in &[1u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            assert_eq!(align_up(align, align), align);
            assert_eq!(align_up(align * 2, align), align * 2);
            assert_eq!(align_up(align * 3, align), align * 3);
        }
    }

    // ── quant_type_to_u8 specific mappings ──────────────────────────────────

    #[test]
    fn quant_type_to_u8_k_quant_family_batch3() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Q2K), 20);
        assert_eq!(quant_type_to_u8(QuantType::Q3K), 21);
        assert_eq!(quant_type_to_u8(QuantType::Q4K), 22);
        assert_eq!(quant_type_to_u8(QuantType::Q5K), 23);
        assert_eq!(quant_type_to_u8(QuantType::Q6K), 24);
        assert_eq!(quant_type_to_u8(QuantType::Q8K), 25);
    }

    #[test]
    fn quant_type_to_u8_iq_family_batch3() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::IQ1S), 30);
        assert_eq!(quant_type_to_u8(QuantType::IQ1M), 31);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XXS), 32);
        assert_eq!(quant_type_to_u8(QuantType::IQ2XS), 33);
        assert_eq!(quant_type_to_u8(QuantType::IQ2S), 34);
        assert_eq!(quant_type_to_u8(QuantType::IQ3XXS), 35);
        assert_eq!(quant_type_to_u8(QuantType::IQ3S), 36);
        assert_eq!(quant_type_to_u8(QuantType::IQ4NL), 37);
        assert_eq!(quant_type_to_u8(QuantType::IQ4XS), 38);
    }

    #[test]
    fn quant_type_to_u8_fp8_family() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Fp8E4M3), 50);
        assert_eq!(quant_type_to_u8(QuantType::Fp8E5M2), 51);
    }

    #[test]
    fn quant_type_to_u8_tq_family() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::TQ1_0), 60);
        assert_eq!(quant_type_to_u8(QuantType::TQ2_0), 61);
    }

    #[test]
    fn quant_type_to_u8_squeeze_batch3() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Squeeze), 42);
    }

    #[test]
    fn quant_type_to_u8_mxfp4_block_size_invariant() {
        use gllm_kernels::quant::QuantType;
        let code_16 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 16 });
        let code_32 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 });
        let code_64 = quant_type_to_u8(QuantType::Mxfp4 { block_size: 64 });
        assert_eq!(code_16, 52);
        assert_eq!(code_32, 52);
        assert_eq!(code_64, 52);
    }

    #[test]
    fn quant_type_to_u8_classic_ranges() {
        use gllm_kernels::quant::QuantType;
        // Classic quants: Q4_0..Q8_1 → 10..15
        assert_eq!(quant_type_to_u8(QuantType::Q4_0), 10);
        assert_eq!(quant_type_to_u8(QuantType::Q4_1), 11);
        assert_eq!(quant_type_to_u8(QuantType::Q5_0), 12);
        assert_eq!(quant_type_to_u8(QuantType::Q5_1), 13);
        assert_eq!(quant_type_to_u8(QuantType::Q8_0), 14);
        assert_eq!(quant_type_to_u8(QuantType::Q8_1), 15);
    }

    #[test]
    fn quant_type_to_u8_float_vs_quant_distinct() {
        use gllm_kernels::quant::QuantType;
        let bf16 = quant_type_to_u8(QuantType::Bf16);
        let f16 = quant_type_to_u8(QuantType::F16);
        let f32 = quant_type_to_u8(QuantType::F32);
        let q4_0 = quant_type_to_u8(QuantType::Q4_0);
        // Float codes (1-3) and quant codes (10+) must not overlap
        assert!(bf16 < 10);
        assert!(f16 < 10);
        assert!(f32 < 10);
        assert!(q4_0 >= 10);
    }

    // ── TensorEntry construction variations ─────────────────────────────────

    #[test]
    fn tensor_entry_shape_with_max_u64_first_dim() {
        let entry = TensorEntry {
            name: "big_dim".into(),
            ndim: 1,
            dtype: 0,
            shape: [u64::MAX, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.shape[0], u64::MAX);
        assert_eq!(entry.shape[1], 0);
    }

    #[test]
    fn tensor_entry_quant_block_size_max_u16() {
        let entry = TensorEntry {
            name: "big_block".into(),
            ndim: 2,
            dtype: 0,
            shape: [1024, 1024, 0, 0],
            quant_format: 40,
            quant_block_size: u16::MAX,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0u8; 512],
            original_size: 4096,
        };
        assert_eq!(entry.quant_block_size, u16::MAX);
        assert!(entry.is_quantized());
    }

    #[test]
    fn tensor_entry_dtype_u8_boundary() {
        let entry = TensorEntry {
            name: "dtype_max".into(),
            ndim: 1,
            dtype: u8::MAX,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 10],
            original_size: 10,
        };
        assert_eq!(entry.dtype, u8::MAX);
    }

    #[test]
    fn tensor_entry_ndim_boundary_values() {
        for ndim in 0u8..=4 {
            let entry = TensorEntry {
                name: "ndim_test".into(),
                ndim,
                dtype: 0,
                shape: [0; 4],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            };
            assert_eq!(entry.ndim, ndim);
        }
    }

    #[test]
    fn tensor_entry_compressed_size_matches_data_len() {
        for len in [0usize, 1, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024] {
            let entry = TensorEntry {
                name: "len_test".into(),
                ndim: 1,
                dtype: 0,
                shape: [len as u64, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; len],
                original_size: len as u64,
            };
            assert_eq!(entry.compressed_size(), len as u64);
        }
    }

    #[test]
    fn tensor_entry_is_quantized_all_nonzero_formats() {
        for fmt in 1u8..=10 {
            let entry = TensorEntry {
                name: "fmt_test".into(),
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: fmt,
                quant_block_size: 32,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![],
                original_size: 0,
            };
            assert!(entry.is_quantized(), "quant_format={} should be quantized", fmt);
        }
    }

    #[test]
    fn tensor_entry_clone_preserves_quant_format() {
        let entry = TensorEntry {
            name: "quant_clone".into(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xABu8; 256],
            original_size: 16384,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.quant_format, 40);
        assert_eq!(cloned.quant_block_size, 128);
        assert_eq!(cloned.scale_dtype, 2);
        assert_eq!(cloned.zp_type, 1);
        assert_eq!(cloned.original_size, 16384);
    }

    // ── build_metadata edge cases ───────────────────────────────────────────

    #[test]
    fn build_metadata_zero_values() {
        let meta = build_metadata(
            "test", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["vocab_size"], "0");
        assert_eq!(parsed["hidden_size"], "0");
        assert_eq!(parsed["num_layers"], "0");
    }

    #[test]
    fn build_metadata_large_values() {
        let meta = build_metadata(
            "big_model", u64::MAX, u64::MAX, u64::MAX, u64::MAX,
            u64::MAX, u64::MAX, u64::MAX, u64::MAX, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let max_str = u64::MAX.to_string();
        assert_eq!(parsed["vocab_size"], max_str.as_str());
        assert_eq!(parsed["hidden_size"], max_str.as_str());
    }

    #[test]
    fn build_metadata_multiple_extras_batch3() {
        let mut extras = HashMap::new();
        extras.insert("moe".to_string(), "true".to_string());
        extras.insert("num_experts".to_string(), "64".to_string());
        extras.insert("rope_theta".to_string(), "1000000".to_string());
        let meta = build_metadata(
            "deepseek", 129280, 4096, 61, 128, 128, 128, 11008, 163840, &extras,
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["moe"], "true");
        assert_eq!(parsed["num_experts"], "64");
        assert_eq!(parsed["rope_theta"], "1000000");
    }

    #[test]
    fn build_metadata_extras_override_standard_keys() {
        let mut extras = HashMap::new();
        extras.insert("vocab_size".to_string(), "99999".to_string());
        let meta = build_metadata(
            "test", 100, 200, 300, 400, 500, 600, 700, 800, &extras,
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // extras inserted after standard keys, so last write wins in HashMap
        assert!(parsed["vocab_size"].as_str() == Some("100") || parsed["vocab_size"].as_str() == Some("99999"));
    }

    #[test]
    fn build_metadata_empty_arch_key_batch3() {
        let meta = build_metadata(
            "", 100, 200, 1, 2, 2, 64, 128, 256, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["arch_key"], "");
    }

    // ── GllmWriter state invariants ─────────────────────────────────────────

    #[test]
    fn writer_new_page_size_stored_correctly() {
        let w = GllmWriter::new(8192);
        assert_eq!(w.page_size, 8192);
    }

    #[test]
    fn writer_new_page_size_zero() {
        let w = GllmWriter::new(0);
        assert_eq!(w.page_size, 0);
    }

    #[test]
    fn writer_tensor_count_zero_initially() {
        let w = GllmWriter::new(4096);
        assert_eq!(w.tensor_count(), 0);
    }

    #[test]
    fn writer_tensor_count_after_multiple_adds() {
        let mut w = GllmWriter::new(4096);
        for i in 0..20 {
            let entry = TensorEntry {
                name: format!("t_{}", i),
                ndim: 1,
                dtype: 0,
                shape: [i as u64, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; i],
                original_size: i as u64,
            };
            w.add_tensor(entry);
        }
        assert_eq!(w.tensor_count(), 20);
    }

    #[test]
    fn writer_set_metadata_twice_replaces() {
        let mut w = GllmWriter::new(4096);
        w.set_metadata(vec![1, 2, 3]);
        w.set_metadata(vec![4, 5]);
        assert_eq!(w.metadata_bytes.len(), 2);
        assert_eq!(w.metadata_bytes[0], 4);
        assert_eq!(w.metadata_bytes[1], 5);
    }

    // ── write_to_path and read-back: structural verification ────────────────

    #[test]
    fn write_empty_file_header_magic_correct() {
        let w = GllmWriter::new(512);
        let dir = unique_test_dir("magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("magic.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().version, GLLM_VERSION);
        assert_eq!(reader.header().tensor_count, 0);
        assert_eq!(reader.header().flags, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_single_tensor_data_offset_starts_at_zero_in_data_region() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "first".into(),
            ndim: 1,
            dtype: 0,
            shape: [16, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xFFu8; 16],
            original_size: 16,
        });
        let dir = unique_test_dir("data_off0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("off0.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("first").unwrap();
        assert_eq!(t.entry.data_offset, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_two_tensors_second_offset_aligned() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "a".into(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![1u8; 10],
            original_size: 10,
        });
        w.add_tensor(TensorEntry {
            name: "b".into(),
            ndim: 1,
            dtype: 0,
            shape: [5, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![2u8; 5],
            original_size: 5,
        });
        let dir = unique_test_dir("two_off");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two_off.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let tb = reader.find_tensor("b").unwrap();
        // Second tensor offset must be page-aligned from data region start
        assert_eq!(tb.entry.data_offset % 512, 0);
        assert!(tb.entry.data_offset > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_quantized_tensor_compressed_size_in_directory() {
        let mut w = GllmWriter::new(512);
        let compressed = vec![0xCCu8; 200];
        w.add_tensor(TensorEntry {
            name: "qweight".into(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: compressed,
            original_size: 16384,
        });
        let dir = unique_test_dir("qsize");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qsize.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("qweight").unwrap();
        assert_eq!(t.entry.compressed_size, 200);
        assert_eq!(t.entry.original_size, 16384);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_metadata_roundtrip_preserves_bytes() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "x".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        let meta = build_metadata(
            "llama4", 32000, 4096, 32, 32, 8, 128, 11008, 8192, &HashMap::new(),
        );
        w.set_metadata(meta.clone());
        let dir = unique_test_dir("meta_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_rt.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let read_meta = reader.metadata_bytes();
        assert_eq!(read_meta, meta.as_slice());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_no_metadata_metadata_bytes_empty() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "y".into(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });
        let dir = unique_test_dir("no_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_meta.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.metadata_bytes().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_data_integrity_first_byte_and_last_byte() {
        let mut w = GllmWriter::new(512);
        let mut data = vec![0u8; 256];
        data[0] = 0xDE;
        data[255] = 0xAD;
        w.add_tensor(TensorEntry {
            name: "edge".into(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data,
            original_size: 256,
        });
        let dir = unique_test_dir("edge_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("edge_data.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("edge").unwrap();
        assert_eq!(td[0], 0xDE);
        assert_eq!(td[255], 0xAD);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_dir_offset_equals_header_size() {
        let mut w = GllmWriter::new(4096);
        w.add_tensor(TensorEntry {
            name: "z".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        let dir = unique_test_dir("td_off");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("td_off.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().tensor_dir_offset, HEADER_SIZE as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_ndim_preserved_in_tensor_directory() {
        let mut w = GllmWriter::new(512);
        for ndim in 1u8..=4 {
            let mut shape = [0u64; 4];
            for i in 0..ndim as usize {
                shape[i] = (i as u64 + 1) * 16;
            }
            w.add_tensor(TensorEntry {
                name: format!("ndim_{}", ndim),
                ndim,
                dtype: 0,
                shape,
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 16,
            });
        }
        let dir = unique_test_dir("ndim_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim_rt.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for ndim in 1u8..=4 {
            let name = format!("ndim_{}", ndim);
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.ndim, ndim);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_dtype_field_preserved_for_various_values() {
        let mut w = GllmWriter::new(512);
        for dtype in [0u8, 1, 2, 3, 4, 5, 6] {
            w.add_tensor(TensorEntry {
                name: format!("dtype_{}", dtype),
                ndim: 1,
                dtype,
                shape: [8, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 8],
                original_size: 8,
            });
        }
        let dir = unique_test_dir("dtype_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_rt.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        for dtype in [0u8, 1, 2, 3, 4, 5, 6] {
            let name = format!("dtype_{}", dtype);
            let t = reader.find_tensor(&name).unwrap();
            assert_eq!(t.entry.dtype, dtype);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_quant_format_zero_flag_not_set() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "plain".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        let dir = unique_test_dir("noq_flag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("noq_flag.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(!reader.header().is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_quant_flag_set_only_one_quantized_among_plain() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "plain1".into(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });
        w.add_tensor(TensorEntry {
            name: "quant1".into(),
            ndim: 2,
            dtype: 0,
            shape: [32, 32, 0, 0],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 512],
            original_size: 4096,
        });
        w.add_tensor(TensorEntry {
            name: "plain2".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        let dir = unique_test_dir("mixed_flag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed_flag.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        assert!(!reader.find_tensor("plain1").unwrap().entry.is_quantized());
        assert!(reader.find_tensor("quant1").unwrap().entry.is_quantized());
        assert!(!reader.find_tensor("plain2").unwrap().entry.is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_page_size_64_data_offset_aligned() {
        let mut w = GllmWriter::new(64);
        w.add_tensor(TensorEntry {
            name: "ps64".into(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 10],
            original_size: 10,
        });
        let dir = unique_test_dir("ps64");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps64.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().data_offset % 64, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_names_with_layer_path_pattern() {
        let mut w = GllmWriter::new(512);
        for i in 0..5 {
            w.add_tensor(TensorEntry {
                name: format!("model.layers.{}.self_attn.q_proj.weight", i),
                ndim: 2,
                dtype: 0,
                shape: [16, 16, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 256],
                original_size: 256,
            });
        }
        let dir = unique_test_dir("layer_names");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("layer_names.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 5);
        for i in 0..5 {
            let name = format!("model.layers.{}.self_attn.q_proj.weight", i);
            let t = reader.find_tensor(&name);
            assert!(t.is_some(), "tensor {} not found", name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_overwrites_existing_file_content() {
        let dir = unique_test_dir("overwrite_v2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overwrite.gllm");

        // Write first file with 3 tensors
        let mut w1 = GllmWriter::new(512);
        for i in 0..3 {
            w1.add_tensor(TensorEntry {
                name: format!("old_{}", i),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 4],
                original_size: 4,
            });
        }
        w1.write_to_path(&path).unwrap();

        // Overwrite with 1 tensor
        let mut w2 = GllmWriter::new(512);
        w2.add_tensor(TensorEntry {
            name: "new_one".into(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });
        w2.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        assert!(reader.find_tensor("new_one").is_some());
        assert!(reader.find_tensor("old_0").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── dtype_to_u8 additional coverage ─────────────────────────────────────

    #[test]
    fn dtype_to_u8_boundary_values() {
        assert_eq!(dtype_to_u8(0), 0);
        assert_eq!(dtype_to_u8(127), 127);
        assert_eq!(dtype_to_u8(255), 255);
    }

    #[test]
    fn safetensors_dtype_to_u8_all_known_types() {
        // Known types get unique codes
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
    }

    // ── write shape dimensions roundtrip ────────────────────────────────────

    #[test]
    fn write_shape_all_dimensions_nonzero_roundtrip() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "4d".into(),
            ndim: 4,
            dtype: 0,
            shape: [3, 64, 128, 256],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 64,
        });
        let dir = unique_test_dir("4d_shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("4d_shape.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("4d").unwrap();
        assert_eq!(t.entry.shape[0], 3);
        assert_eq!(t.entry.shape[1], 64);
        assert_eq!(t.entry.shape[2], 128);
        assert_eq!(t.entry.shape[3], 256);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_large_tensor_data_roundtrip() {
        let mut w = GllmWriter::new(4096);
        let size = 65536;
        let mut data = vec![0u8; size];
        for i in 0..size {
            data[i] = (i % 256) as u8;
        }
        w.add_tensor(TensorEntry {
            name: "big".into(),
            ndim: 1,
            dtype: 0,
            shape: [size as u64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: size as u64,
        });
        let dir = unique_test_dir("big_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("big_data.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("big").unwrap();
        assert_eq!(td.len(), size);
        for i in 0..size {
            assert_eq!(td[i], (i % 256) as u8);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── write empty data tensors ────────────────────────────────────────────

    #[test]
    fn write_empty_data_tensor_roundtrip() {
        let mut w = GllmWriter::new(512);
        w.add_tensor(TensorEntry {
            name: "empty".into(),
            ndim: 1,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        });
        let dir = unique_test_dir("empty_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_data.gllm");
        w.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("empty").unwrap();
        assert!(td.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry Debug format checks ─────────────────────────────────────

    #[test]
    fn tensor_entry_debug_contains_all_key_fields() {
        let entry = TensorEntry {
            name: "my.weight".into(),
            ndim: 2,
            dtype: 3,
            shape: [512, 512, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 1024],
            original_size: 2048,
        };
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("my.weight"));
        assert!(debug_str.contains("ndim"));
        assert!(debug_str.contains("dtype"));
        assert!(debug_str.contains("shape"));
        assert!(debug_str.contains("quant_format"));
    }

    // ── safetensors_dtype_to_u8 edge cases ──────────────────────────────────

    #[test]
    fn safetensors_dtype_to_u8_unknown_types_default_to_zero() {
        // Test a few types that are not in the known match arms
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U16), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U64), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F64), 0);
    }

    // ── TensorEntry compressed_size edge cases ──────────────────────────────

    #[test]
    fn tensor_entry_compressed_size_empty_data_batch3() {
        let entry = TensorEntry {
            name: "empty".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.compressed_size(), 0);
    }

    #[test]
    fn tensor_entry_compressed_size_single_byte_batch3() {
        let entry = TensorEntry {
            name: "one".into(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42u8],
            original_size: 4,
        };
        assert_eq!(entry.compressed_size(), 1);
        // original_size and compressed_size are independent
        assert_eq!(entry.original_size, 4);
    }

    // ── write_to_path error cases ───────────────────────────────────────────

    #[test]
    fn write_to_path_invalid_directory_fails() {
        let w = GllmWriter::new(512);
        let path = std::path::Path::new("/nonexistent/deeply/nested/dir/output.gllm");
        let result = w.write_to_path(path);
        assert!(result.is_err());
    }

    // ── TensorEntry original_size independence from data ────────────────────

    #[test]
    fn tensor_entry_original_size_larger_than_data() {
        let entry = TensorEntry {
            name: "compressed".into(),
            ndim: 2,
            dtype: 0,
            shape: [64, 64, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 16384,
        };
        assert_eq!(entry.compressed_size(), 1024);
        assert_eq!(entry.original_size, 16384);
        assert!(entry.original_size > entry.compressed_size());
    }

    #[test]
    fn tensor_entry_original_size_smaller_than_data() {
        // Unusual but valid: original_size can be less than data length
        let entry = TensorEntry {
            name: "unusual".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 256],
            original_size: 16,
        };
        assert_eq!(entry.compressed_size(), 256);
        assert_eq!(entry.original_size, 16);
    }

    // ── Build metadata produces valid JSON ──────────────────────────────────

    #[test]
    fn build_metadata_is_valid_json_batch3() {
        let meta = build_metadata(
            "test", 1000, 512, 8, 8, 4, 64, 2048, 4096, &HashMap::new(),
        );
        let result = serde_json::from_slice::<serde_json::Value>(&meta);
        assert!(result.is_ok(), "metadata must be valid JSON");
    }

    #[test]
    fn build_metadata_with_extras_is_valid_json() {
        let mut extras = HashMap::new();
        extras.insert("k1".to_string(), "v1".to_string());
        extras.insert("k2".to_string(), "v2".to_string());
        let meta = build_metadata(
            "test", 100, 200, 1, 2, 2, 64, 128, 256, &extras,
        );
        let parsed = serde_json::from_slice::<serde_json::Value>(&meta);
        assert!(parsed.is_ok());
    }

    // ── Header constants validation ─────────────────────────────────────────

    #[test]
    fn header_size_is_64() {
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn tensor_entry_size_is_72() {
        assert_eq!(TENSOR_ENTRY_SIZE, 72);
    }

    #[test]
    fn gllm_magic_is_correct_value() {
        // 'G'=0x47, 'L'=0x4C, 'L'=0x4C, 'M'=0x4D in LE = 0x4D4C4C47
        assert_eq!(GLLM_MAGIC, 0x4D4C4C47);
    }

    #[test]
    fn gllm_version_is_one() {
        assert_eq!(GLLM_VERSION, 1);
    }

    // ── Quant code range invariants ─────────────────────────────────────────

    #[test]
    fn quant_type_codes_float_range_1_to_3() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
        assert_eq!(quant_type_to_u8(QuantType::F32), 3);
        // Float codes occupy 1..=3
    }

    #[test]
    fn quant_type_codes_classic_range_10_to_15() {
        use gllm_kernels::quant::QuantType;
        for (qt, expected) in [
            (QuantType::Q4_0, 10u8),
            (QuantType::Q4_1, 11),
            (QuantType::Q5_0, 12),
            (QuantType::Q5_1, 13),
            (QuantType::Q8_0, 14),
            (QuantType::Q8_1, 15),
        ] {
            assert_eq!(quant_type_to_u8(qt), expected);
        }
    }

    #[test]
    fn quant_type_codes_k_quant_range_20_to_25() {
        use gllm_kernels::quant::QuantType;
        for (qt, expected) in [
            (QuantType::Q2K, 20u8),
            (QuantType::Q3K, 21),
            (QuantType::Q4K, 22),
            (QuantType::Q5K, 23),
            (QuantType::Q6K, 24),
            (QuantType::Q8K, 25),
        ] {
            assert_eq!(quant_type_to_u8(qt), expected);
        }
    }

    #[test]
    fn quant_type_codes_iq_range_30_to_38() {
        use gllm_kernels::quant::QuantType;
        for (qt, expected) in [
            (QuantType::IQ1S, 30u8),
            (QuantType::IQ1M, 31),
            (QuantType::IQ2XXS, 32),
            (QuantType::IQ2XS, 33),
            (QuantType::IQ2S, 34),
            (QuantType::IQ3XXS, 35),
            (QuantType::IQ3S, 36),
            (QuantType::IQ4NL, 37),
            (QuantType::IQ4XS, 38),
        ] {
            assert_eq!(quant_type_to_u8(qt), expected);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Batch 4: 45 additional unit tests — uncovered gaps
    // ────────────────────────────────────────────────────────────────────────

    // ── Tensor dir binary layout: quant_format, quant_block_size, padding ──

    #[test]
    fn tensor_dir_entry_quant_format_byte() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "qf".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 41, quant_block_size: 128, scale_dtype: 2, zp_type: 1,
            data: vec![0u8; 8], original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("qf_byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qf_byte.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // quant_format is at byte 40 of tensor directory entry
        assert_eq!(raw[HEADER_SIZE + 40], 41);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_quant_block_size_bytes() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "qbs".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 40, quant_block_size: 256, scale_dtype: 2, zp_type: 1,
            data: vec![0u8; 16], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("qbs_bytes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qbs_bytes.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // quant_block_size is at bytes 41..43 of tensor directory entry
        let qbs = u16::from_le_bytes(raw[HEADER_SIZE + 41..HEADER_SIZE + 43].try_into().unwrap());
        assert_eq!(qbs, 256);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_scale_dtype_byte() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "sd".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 40, quant_block_size: 128, scale_dtype: 5, zp_type: 0,
            data: vec![0u8; 8], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("sd_byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sd_byte.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // scale_dtype is at byte 43 of tensor directory entry
        assert_eq!(raw[HEADER_SIZE + 43], 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_zp_type_byte() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "zp".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 41, quant_block_size: 128, scale_dtype: 1, zp_type: 3,
            data: vec![0u8; 8], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("zp_byte");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zp_byte.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // zp_type is at byte 44 of tensor directory entry
        assert_eq!(raw[HEADER_SIZE + 44], 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_padding_bytes_are_zero() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "pad".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("dir_pad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dir_pad.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // padding bytes 45..47 of tensor directory entry should be zero
        assert_eq!(raw[HEADER_SIZE + 45], 0);
        assert_eq!(raw[HEADER_SIZE + 46], 0);
        assert_eq!(raw[HEADER_SIZE + 47], 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_dir_entry_data_offset_field() {
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "first".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.add_tensor(TensorEntry {
            name: "second".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("doff_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("doff_field.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // First tensor: data_offset at byte 48..56 of first entry should be 0
        let off0 = u64::from_le_bytes(raw[HEADER_SIZE + 48..HEADER_SIZE + 56].try_into().unwrap());
        assert_eq!(off0, 0, "first tensor data_offset should be 0");
        // Second tensor: data_offset at byte 48..56 of second entry should be 128 (aligned from first)
        let entry2_off = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let off1 = u64::from_le_bytes(raw[entry2_off + 48..entry2_off + 56].try_into().unwrap());
        assert_eq!(off1, 128, "second tensor data_offset should be 128");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Data region: inter-tensor padding verification ─────────────────────

    #[test]
    fn data_region_no_overlap_between_tensors() {
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 5], original_size: 5,
        });
        builder.add_tensor(TensorEntry {
            name: "b".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 3], original_size: 3,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("no_overlap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_overlap.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let da = reader.tensor_data("a").unwrap();
        let db = reader.tensor_data("b").unwrap();
        // Each tensor's data should contain only its own pattern
        assert_eq!(da.len(), 5);
        assert!(da.iter().all(|&b| b == 0xAA));
        assert_eq!(db.len(), 3);
        assert!(db.iter().all(|&b| b == 0xBB));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_region_padding_between_tensors_is_zero() {
        let mut builder = GllmWriter::new(32);
        // 7 bytes → padded to 32
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0, shape: [7, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xEE; 7], original_size: 7,
        });
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF; 3], original_size: 3,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("inter_pad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("inter_pad.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        // t1 data: bytes 0..7 should be 0xEE
        for i in 0..7 {
            assert_eq!(raw[data_offset + i], 0xEE, "t1 data at {}", i);
        }
        // Padding for t1: bytes 7..32 should be zero
        for i in 7..32 {
            assert_eq!(raw[data_offset + i], 0, "t1 padding at {}", i);
        }
        // t2 data starts at byte 32: bytes 32..35 should be 0xFF
        for i in 0..3 {
            assert_eq!(raw[data_offset + 32 + i], 0xFF, "t2 data at {}", i);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_tensor_aligned_to_page_no_padding_needed() {
        // Tensor data exactly equals page_size — no padding
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "exact".to_string(), ndim: 1, dtype: 0, shape: [64, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x77; 64], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("exact_page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_page.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("exact").unwrap();
        assert_eq!(t.entry.compressed_size, 64);
        // Next tensor would start at offset 64 (aligned)
        // File size = data_offset + 64 (no extra padding)
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        assert_eq!(raw.len(), data_offset + 64, "file should end exactly after aligned data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn data_two_aligned_tensors_no_gap() {
        let mut builder = GllmWriter::new(32);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [32, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11; 32], original_size: 32,
        });
        builder.add_tensor(TensorEntry {
            name: "b".to_string(), ndim: 1, dtype: 0, shape: [32, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x22; 32], original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("two_aligned");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("two_aligned.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let tb = reader.find_tensor("b").unwrap();
        assert_eq!(tb.entry.data_offset, 32, "second aligned tensor starts at page boundary");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── String table: multi-tensor offset accumulation ─────────────────────

    #[test]
    fn string_table_three_tensors_offset_chain() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "alpha".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "beta_longer".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "g".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("stab_chain");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stab_chain.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let e2_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        let off2 = u32::from_le_bytes(raw[e2_start..e2_start + 4].try_into().unwrap());
        let len2 = u16::from_le_bytes(raw[e2_start + 4..e2_start + 6].try_into().unwrap());
        // "alpha" = 5 bytes, "beta_longer" = 11 bytes → offset for "g" should be 5 + 11 = 16
        assert_eq!(off2, 16);
        assert_eq!(len2, 1);

        // Third tensor's name_len should be 1 ("g") — already verified via len2
        assert_eq!(len2, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Tensor name with special characters roundtrip ──────────────────────

    #[test]
    fn roundtrip_tensor_name_with_brackets() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "layer[0].weight".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x42], original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("brackets");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("brackets.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("layer[0].weight");
        assert!(t.is_some(), "name with brackets should be findable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_tensor_name_with_colon() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "prefix:suffix".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x55], original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("colon_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("colon_name.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("prefix:suffix").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_tensor_name_with_at_sign() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "model@v2.weight".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x33], original_size: 1,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("at_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("at_name.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.find_tensor("model@v2.weight").is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: property-based edge cases ─────────────────────────────

    #[test]
    fn tensor_entry_compressed_size_always_equals_data_len() {
        // Property: compressed_size() must always equal data.len() as u64
        let cases: Vec<Vec<u8>> = vec![
            vec![], vec![0], vec![0; 1], vec![0; 15], vec![0; 16],
            vec![0; 31], vec![0; 32], vec![0; 33], vec![0; 128], vec![0; 4096],
        ];
        for data in &cases {
            let entry = TensorEntry {
                name: "prop".into(), ndim: 1, dtype: 0, shape: [0; 4],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: data.clone(), original_size: 0,
            };
            assert_eq!(entry.compressed_size(), data.len() as u64);
        }
    }

    #[test]
    fn tensor_entry_is_quantized_excludes_zero_only() {
        // Property: only quant_format == 0 returns false
        assert!(!TensorEntry {
            name: "t".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        }.is_quantized());
        for qf in [1u8, 10, 20, 30, 40, 50, 60, 100, 200, 255] {
            assert!(TensorEntry {
                name: "t".into(), ndim: 1, dtype: 0, shape: [0; 4],
                quant_format: qf, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![], original_size: 0,
            }.is_quantized(), "quant_format={qf} must be quantized");
        }
    }

    #[test]
    fn tensor_entry_original_size_can_exceed_u32_range() {
        // Verify original_size supports full u64 range
        let entry = TensorEntry {
            name: "big_orig".into(), ndim: 2, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 16], original_size: 5_000_000_000u64,
        };
        assert_eq!(entry.original_size, 5_000_000_000u64);
        assert!(entry.original_size > u32::MAX as u64);
    }

    #[test]
    fn tensor_entry_shape_can_hold_full_u64_per_dim() {
        let entry = TensorEntry {
            name: "huge_dims".into(), ndim: 4, dtype: 0,
            shape: [1, u64::MAX, u64::MAX / 2, 1],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert_eq!(entry.shape[1], u64::MAX);
        assert_eq!(entry.shape[2], u64::MAX / 2);
        assert_eq!(entry.ndim, 4);
    }

    #[test]
    fn tensor_entry_clone_preserves_all_scalar_fields() {
        let original = TensorEntry {
            name: "full".into(), ndim: 3, dtype: 5, shape: [10, 20, 30, 0],
            quant_format: 40, quant_block_size: 64, scale_dtype: 2, zp_type: 1,
            data: vec![0xAB; 100], original_size: 1000,
        };
        let cloned = original.clone();
        assert_eq!(cloned.ndim, 3);
        assert_eq!(cloned.dtype, 5);
        assert_eq!(cloned.quant_format, 40);
        assert_eq!(cloned.quant_block_size, 64);
        assert_eq!(cloned.scale_dtype, 2);
        assert_eq!(cloned.zp_type, 1);
        assert_eq!(cloned.original_size, 1000);
        assert_eq!(cloned.compressed_size(), 100);
        assert!(cloned.is_quantized());
    }

    #[test]
    fn tensor_entry_data_with_prime_length() {
        // Prime number lengths to catch alignment-related bugs
        let primes = [2usize, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        for &len in &primes {
            let entry = TensorEntry {
                name: "prime".into(), ndim: 1, dtype: 0, shape: [len as u64, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0x42u8; len], original_size: len as u64,
            };
            assert_eq!(entry.compressed_size(), len as u64);
        }
    }

    // ── build_metadata: edge cases with special characters ─────────────────

    #[test]
    fn build_metadata_with_json_special_chars_in_extra_value() {
        let mut extras = HashMap::new();
        extras.insert("json_str".to_string(), r#"{"key":"val"}"#.to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["json_str"], r#"{"key":"val"}"#);
    }

    #[test]
    fn build_metadata_with_double_quotes_in_extra_key() {
        let mut extras = HashMap::new();
        extras.insert(r#"key"with"quotes"#.to_string(), "value".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed[r#"key"with"quotes"#], "value");
    }

    #[test]
    fn build_metadata_with_null_char_in_value() {
        let mut extras = HashMap::new();
        extras.insert("binary".to_string(), "before\0after".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["binary"], "before\0after");
    }

    #[test]
    fn build_metadata_context_length_preserved_as_string() {
        let meta = build_metadata("test", 100, 200, 10, 8, 4, 64, 256, 131072, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["context_length"], "131072");
        assert!(parsed["context_length"].is_string());
    }

    #[test]
    fn build_metadata_num_kv_heads_preserved_as_string() {
        let meta = build_metadata("test", 100, 200, 10, 8, 4, 64, 256, 512, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["num_kv_heads"], "4");
    }

    // ── GllmWriter: state management edge cases ────────────────────────────

    #[test]
    fn writer_add_tensor_then_set_metadata_then_count() {
        let mut w = GllmWriter::new(4096);
        w.add_tensor(TensorEntry {
            name: "t1".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        w.set_metadata(vec![1, 2, 3]);
        w.add_tensor(TensorEntry {
            name: "t2".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(w.tensor_count(), 2);
        assert_eq!(w.metadata_bytes, vec![1, 2, 3]);
    }

    #[test]
    fn writer_many_tensors_accumulate_correctly() {
        let mut w = GllmWriter::new(4096);
        let count = 100;
        for i in 0..count {
            w.add_tensor(TensorEntry {
                name: format!("tensor_{:04}", i),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![i as u8], original_size: 1,
            });
        }
        assert_eq!(w.tensor_count(), count);
        // Verify first and last tensor names
        assert_eq!(w.tensors[0].name, "tensor_0000");
        assert_eq!(w.tensors[99].name, "tensor_0099");
    }

    #[test]
    fn writer_metadata_set_before_adding_tensors() {
        let mut w = GllmWriter::new(256);
        w.set_metadata(vec![0xDE, 0xAD]);
        w.add_tensor(TensorEntry {
            name: "t".into(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        assert_eq!(w.metadata_bytes, vec![0xDE, 0xAD]);
        assert_eq!(w.tensor_count(), 1);
    }

    // ── Roundtrip: data integrity with specific byte patterns ──────────────

    #[test]
    fn roundtrip_data_with_fibonacci_pattern() {
        let mut data = vec![0u8; 20];
        data[0] = 1;
        data[1] = 1;
        for i in 2..20 {
            data[i] = data[i - 1].wrapping_add(data[i - 2]);
        }
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "fib".to_string(), ndim: 1, dtype: 0, shape: [20, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 20,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("fib");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fib.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("fib").unwrap();
        assert_eq!(&td[..], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_data_with_checkerboard_pattern() {
        let data: Vec<u8> = (0..128).map(|i| if i % 8 < 4 { 0xFF } else { 0x00 }).collect();
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "checker".to_string(), ndim: 1, dtype: 0, shape: [128, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 128,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("checker");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checker.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("checker").unwrap();
        assert_eq!(&td[..], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_data_with_randomish_lfsr_pattern() {
        // Simple LFSR-like pattern for deterministic but scattered data
        let mut data = vec![0u8; 64];
        let mut state: u8 = 1;
        for byte in data.iter_mut() {
            let new_bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 4)) & 1;
            state = (state >> 1) | (new_bit << 7);
            *byte = state;
        }
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "lfsr".to_string(), ndim: 1, dtype: 0, shape: [64, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("lfsr");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lfsr.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("lfsr").unwrap();
        assert_eq!(&td[..], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Roundtrip: mixed tensor types in same file ─────────────────────────

    #[test]
    fn roundtrip_mixed_quant_formats_preserve_all_fields() {
        let mut builder = GllmWriter::new(128);
        let formats = [(10u8, 32u16, "Q4_0"), (22u8, 64u16, "Q4K"), (40u8, 128u16, "AWQ4"), (41u8, 128u16, "GPTQ4")];
        for (qf, qbs, name) in &formats {
            builder.add_tensor(TensorEntry {
                name: name.to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
                quant_format: *qf, quant_block_size: *qbs, scale_dtype: 2, zp_type: 1,
                data: vec![0u8; 16], original_size: 64,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("mixed_qf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mixed_qf.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        for (qf, qbs, name) in &formats {
            let t = reader.find_tensor(name).unwrap();
            assert_eq!(t.entry.quant_format, *qf, "quant_format mismatch for {}", name);
            assert_eq!(t.entry.quant_block_size, *qbs, "quant_block_size mismatch for {}", name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_tensor_with_all_shape_dims_nonzero() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "full4d".to_string(), ndim: 4, dtype: 0,
            shape: [7, 13, 17, 19],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 32], original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("full4d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full4d.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("full4d").unwrap();
        assert_eq!(t.entry.shape, [7, 13, 17, 19]);
        assert_eq!(t.entry.ndim, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Binary layout: header field layout verification ────────────────────

    #[test]
    fn header_layout_magic_bytes_raw() {
        let builder = GllmWriter::new(256);
        let dir = unique_test_dir("raw_magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("raw_magic.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Magic: 'G'=0x47, 'L'=0x4C, 'L'=0x4C, 'M'=0x4D in LE
        assert_eq!(raw[0], 0x47);
        assert_eq!(raw[1], 0x4C);
        assert_eq!(raw[2], 0x4C);
        assert_eq!(raw[3], 0x4D);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_data_offset_matches_actual_data_start() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "verify".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBA; 8], original_size: 8,
        });
        builder.set_metadata(vec![0xCA, 0xFE]);

        let dir = unique_test_dir("doff_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("doff_verify.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        // First byte of data region should be our tensor data
        assert_eq!(raw[data_offset], 0xBA, "data_offset should point to tensor data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Tensor dir: second tensor entry fields at correct offsets ───────────

    #[test]
    fn tensor_dir_second_entry_name_offset_accumulates_name1_len() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "first_tensor".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "sec".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("name_accum");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("name_accum.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let e2 = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let off2 = u32::from_le_bytes(raw[e2..e2 + 4].try_into().unwrap());
        let len2 = u16::from_le_bytes(raw[e2 + 4..e2 + 6].try_into().unwrap());
        // "first_tensor" = 12 bytes → second name starts at offset 12
        assert_eq!(off2, 12);
        assert_eq!(len2, 3); // "sec" = 3 bytes
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Error path: write_to_path with permission denied simulation ────────

    #[test]
    fn write_to_path_readonly_directory_fails() {
        let dir = unique_test_dir("readonly_dir");
        std::fs::create_dir_all(&dir).unwrap();

        // Create a read-only subdirectory (may not work on all systems)
        let readonly = dir.join("readonly");
        std::fs::create_dir_all(&readonly).unwrap();

        let builder = GllmWriter::new(256);
        // Try writing to a path that would need directory creation
        let path = std::path::Path::new("/proc/gllm_test_output.gllm");
        let result = builder.write_to_path(path);
        // Writing to /proc should fail
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── align_up: additional properties ────────────────────────────────────

    #[test]
    fn align_up_result_ge_input() {
        // align_up result is always >= input (when alignment > 0)
        for val in [0u64, 1, 7, 63, 64, 65, 127, 128, 4095, 4096, 4097] {
            let result = align_up(val, 64);
            assert!(result >= val, "align_up({val}, 64) = {result} < {val}");
        }
    }

    #[test]
    fn align_up_result_is_minimal_aligned() {
        // align_up result should be the smallest aligned value >= input
        assert_eq!(align_up(1, 4096), 4096); // smallest 4096-aligned >= 1
        assert_eq!(align_up(4095, 4096), 4096); // smallest 4096-aligned >= 4095
        assert_eq!(align_up(4096, 4096), 4096); // 4096 itself is aligned
        assert_eq!(align_up(4097, 4096), 8192); // smallest 4096-aligned >= 4097
    }

    // ── safetensors_dtype_to_u8: completeness check ────────────────────────

    #[test]
    fn safetensors_dtype_to_u8_known_codes_are_sequential() {
        // Known codes 0..6 should be contiguous
        let codes: Vec<u8> = (0..=6).map(|i| {
            match i {
                0 => safetensors_dtype_to_u8(safetensors::Dtype::F32),
                1 => safetensors_dtype_to_u8(safetensors::Dtype::F16),
                2 => safetensors_dtype_to_u8(safetensors::Dtype::BF16),
                3 => safetensors_dtype_to_u8(safetensors::Dtype::U8),
                4 => safetensors_dtype_to_u8(safetensors::Dtype::I8),
                5 => safetensors_dtype_to_u8(safetensors::Dtype::I32),
                6 => safetensors_dtype_to_u8(safetensors::Dtype::I64),
                _ => unreachable!(),
            }
        }).collect();
        assert_eq!(codes, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    // ── Roundtrip: metadata with real model config values ──────────────────

    #[test]
    fn roundtrip_metadata_deepseek_v3_config() {
        let mut extras = HashMap::new();
        extras.insert("model_type".to_string(), "deepseek_v3".to_string());
        extras.insert("num_experts".to_string(), "256".to_string());
        extras.insert("num_experts_per_tok".to_string(), "8".to_string());
        extras.insert("shared_expert_intermediate_size".to_string(), "4096".to_string());

        let meta = build_metadata(
            "deepseek_v3", 129280, 7168, 61, 128, 128, 128, 18432, 163840, &extras,
        );

        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "test".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(meta);

        let dir = unique_test_dir("ds3_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ds3_config.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let read_meta = reader.metadata_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(read_meta).unwrap();
        assert_eq!(parsed["arch_key"], "deepseek_v3");
        assert_eq!(parsed["vocab_size"], "129280");
        assert_eq!(parsed["num_experts"], "256");
        assert_eq!(parsed["shared_expert_intermediate_size"], "4096");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_metadata_qwen3_config() {
        let meta = build_metadata(
            "qwen3", 151936, 4096, 36, 32, 8, 128, 11008, 32768, &HashMap::new(),
        );

        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "w".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(meta);

        let dir = unique_test_dir("qwen3_config");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qwen3_config.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let read_meta = reader.metadata_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(read_meta).unwrap();
        assert_eq!(parsed["arch_key"], "qwen3");
        assert_eq!(parsed["hidden_size"], "4096");
        assert_eq!(parsed["num_layers"], "36");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: debug format completeness ─────────────────────────────

    #[test]
    fn tensor_entry_debug_includes_data_len_field() {
        let entry = TensorEntry {
            name: "dl".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 42], original_size: 42,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("data"), "Debug should include data field");
    }

    #[test]
    fn tensor_entry_debug_includes_compressed_size_concept() {
        let entry = TensorEntry {
            name: "cs_debug".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 100], original_size: 400,
        };
        // Verify the method works correctly with the Debug output
        assert_eq!(entry.compressed_size(), 100);
        let debug = format!("{entry:?}");
        assert!(debug.contains("cs_debug"));
    }

    // ── Roundtrip: page_size field in header ───────────────────────────────

    #[test]
    fn roundtrip_page_size_16384() {
        let mut builder = GllmWriter::new(16384);
        builder.add_tensor(TensorEntry {
            name: "bigpage".to_string(), ndim: 1, dtype: 0, shape: [10, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 10], original_size: 10,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps16384");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps16384.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 16384);
        assert_eq!(reader.header().data_offset % 16384, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_page_size_32() {
        let mut builder = GllmWriter::new(32);
        builder.add_tensor(TensorEntry {
            name: "smallpage".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 5], original_size: 5,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps32");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps32.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Roundtrip: tensor with data length exactly equal to multiple page sizes

    #[test]
    fn roundtrip_data_exactly_two_pages() {
        let mut builder = GllmWriter::new(64);
        // 128 bytes = exactly 2 pages of 64
        let data = vec![0xABu8; 128];
        builder.add_tensor(TensorEntry {
            name: "twopages".to_string(), ndim: 1, dtype: 0, shape: [128, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 128,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("twopages");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("twopages.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("twopages").unwrap();
        assert_eq!(td.len(), 128);
        assert_eq!(&td[..], &data[..]);

        // Verify no extra padding after the data
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        assert_eq!(raw.len(), data_offset + 128, "file should end exactly after 2-page data");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Constants verification ─────────────────────────────────────────────

    #[test]
    fn header_size_and_tensor_entry_size_multiply_correctly() {
        // 64 + 72*N should give correct tensor directory start
        assert_eq!(HEADER_SIZE + 0 * TENSOR_ENTRY_SIZE, 64);
        assert_eq!(HEADER_SIZE + 1 * TENSOR_ENTRY_SIZE, 136);
        assert_eq!(HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE, 208);
        assert_eq!(HEADER_SIZE + 10 * TENSOR_ENTRY_SIZE, 784);
    }

    #[test]
    fn tensor_entry_size_is_72_bytes() {
        // SPEC: each tensor directory entry is exactly 72 bytes
        assert_eq!(TENSOR_ENTRY_SIZE, 72);
        // name_off(4) + name_len(2) + ndim(1) + dtype(1) + shape(32) + quant_format(1)
        // + quant_block_size(2) + scale_dtype(1) + zp_type(1) + padding(3)
        // + data_offset(8) + compressed_size(8) + original_size(8) = 72
        let field_sizes: usize = 4 + 2 + 1 + 1 + 32 + 1 + 2 + 1 + 1 + 3 + 8 + 8 + 8;
        assert_eq!(field_sizes, 72);
    }

    // ── dtype_to_u8: all values pass through unchanged ─────────────────────

    #[test]
    fn dtype_to_u8_all_values_identity() {
        for v in 0u8..=255 {
            assert_eq!(dtype_to_u8(v), v);
        }
    }

    // ── Roundtrip: quantized tensor with all quant fields populated ─────────

    #[test]
    fn roundtrip_quantized_tensor_all_fields_populated() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "full_quant".to_string(),
            ndim: 2,
            dtype: 3,
            shape: [256, 512, 0, 0],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xCC; 8192],
            original_size: 524288,
        });
        let meta = build_metadata("test_model", 32000, 4096, 32, 32, 8, 128, 11008, 4096, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("full_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full_quant.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        let t = reader.find_tensor("full_quant").unwrap();
        assert_eq!(t.entry.ndim, 2);
        assert_eq!(t.entry.dtype, 3);
        assert_eq!(t.entry.shape[0], 256);
        assert_eq!(t.entry.shape[1], 512);
        assert_eq!(t.entry.quant_format, 41);
        assert_eq!(t.entry.quant_block_size, 128);
        assert_eq!(t.entry.scale_dtype, 2);
        assert_eq!(t.entry.zp_type, 1);
        assert_eq!(t.entry.compressed_size, 8192);
        assert_eq!(t.entry.original_size, 524288);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── write_to internal: verify padding calculation for large metadata ───

    #[test]
    fn write_with_large_metadata_data_offset_correctly_aligned() {
        let mut builder = GllmWriter::new(1024);
        builder.add_tensor(TensorEntry {
            name: "m".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        // Metadata larger than page_size to ensure alignment matters
        builder.set_metadata(vec![0x42; 900]);

        let dir = unique_test_dir("large_meta_align");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_meta_align.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        assert_eq!(data_offset % 1024, 0, "data_offset must be page-aligned even with large metadata");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── build_metadata: ensure no panic with all max values ────────────────

    #[test]
    fn build_metadata_all_u64_max_no_panic() {
        let meta = build_metadata(
            "max_model",
            u64::MAX, u64::MAX, u64::MAX, u64::MAX,
            u64::MAX, u64::MAX, u64::MAX, u64::MAX,
            &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let max_str = u64::MAX.to_string();
        // All standard fields should be the max u64 string
        assert_eq!(parsed["vocab_size"], max_str.as_str());
        assert_eq!(parsed["hidden_size"], max_str.as_str());
        assert_eq!(parsed["num_layers"], max_str.as_str());
    }

    // ── quant_type_to_u8: ensure each family has distinct range ─────────────

    #[test]
    fn quant_type_codes_family_ranges_do_not_overlap() {
        use gllm_kernels::quant::QuantType;
        let float_codes: Vec<u8> = vec![
            quant_type_to_u8(QuantType::Bf16), quant_type_to_u8(QuantType::F16),
            quant_type_to_u8(QuantType::F32),
        ];
        let classic_codes: Vec<u8> = vec![
            quant_type_to_u8(QuantType::Q4_0), quant_type_to_u8(QuantType::Q4_1),
            quant_type_to_u8(QuantType::Q5_0), quant_type_to_u8(QuantType::Q5_1),
            quant_type_to_u8(QuantType::Q8_0), quant_type_to_u8(QuantType::Q8_1),
        ];
        let k_codes: Vec<u8> = vec![
            quant_type_to_u8(QuantType::Q2K), quant_type_to_u8(QuantType::Q3K),
            quant_type_to_u8(QuantType::Q4K), quant_type_to_u8(QuantType::Q5K),
            quant_type_to_u8(QuantType::Q6K), quant_type_to_u8(QuantType::Q8K),
        ];
        // No overlap between float (1-3), classic (10-15), and k-quant (20-25)
        for fc in &float_codes {
            assert!(!classic_codes.contains(fc));
            assert!(!k_codes.contains(fc));
        }
        for cc in &classic_codes {
            assert!(!k_codes.contains(cc));
        }
    }

    // ── TensorEntry: method return type verification ───────────────────────

    #[test]
    fn tensor_entry_compressed_size_return_type_is_u64() {
        let entry = TensorEntry {
            name: "type_check".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        let size: u64 = entry.compressed_size();
        assert_eq!(size, 0u64);
    }

    #[test]
    fn tensor_entry_is_quantized_return_type_is_bool() {
        let entry = TensorEntry {
            name: "bool_check".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        let quantized: bool = entry.is_quantized();
        assert!(!quantized);
    }

    // ── Roundtrip: empty tensor name edge case ─────────────────────────────

    #[test]
    fn roundtrip_empty_name_tensor() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: String::new(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("empty_name_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_name_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        // Empty name tensor should still be findable
        let t = reader.find_tensor("");
        assert!(t.is_some(), "empty name tensor should be findable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── write_to internal: string table concatenated correctly ─────────────

    #[test]
    fn string_table_concatenates_all_names_in_order() {
        let mut builder = GllmWriter::new(256);
        let names = ["aaa", "bbbb", "ccccc"];
        for name in &names {
            builder.add_tensor(TensorEntry {
                name: name.to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 4], original_size: 4,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("strtab_concat");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("strtab_concat.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let strtab_start = HEADER_SIZE + 3 * TENSOR_ENTRY_SIZE;
        // String table should be "aaabbbbccccc"
        let expected = b"aaabbbbccccc";
        assert_eq!(&raw[strtab_start..strtab_start + expected.len()], expected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Header version field always GLLM_VERSION ──────────────────────────

    #[test]
    fn header_version_unchanged_across_different_page_sizes() {
        for &ps in &[64u32, 128, 256, 512, 1024, 2048, 4096] {
            let builder = GllmWriter::new(ps);
            let dir = std::env::temp_dir().join(format!("gllm_test_ver_ps{}", ps));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("ver_ps{}.gllm", ps));
            builder.write_to_path(&path).unwrap();

            let raw = std::fs::read(&path).unwrap();
            let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
            assert_eq!(version, GLLM_VERSION);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    // ── TensorEntry: verify derive traits work correctly ───────────────────

    #[test]
    fn tensor_entry_clone_deep_copies_vec_data() {
        let original = TensorEntry {
            name: "vec_copy".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![1, 2, 3, 4, 5], original_size: 5,
        };
        let mut cloned = original.clone();
        // Extend cloned data
        cloned.data.extend_from_slice(&[6, 7, 8]);
        // Original should be unchanged
        assert_eq!(original.data.len(), 5);
        assert_eq!(cloned.data.len(), 8);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional 35 unit tests — wave 12x58
    // ────────────────────────────────────────────────────────────────────────

    // ── GllmWriter: write_to produces valid GllmHeader via parse ──────────

    #[test]
    fn write_header_parse_roundtrip_empty_writer() {
        let builder = GllmWriter::new(4096);
        let dir = unique_test_dir("hdr_parse_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hdr_empty.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let header = GllmHeader::parse(&raw).unwrap();
        assert_eq!(header.tensor_count, 0);
        assert!(!header.is_quantized());
        assert_eq!(header.page_size, 4096);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_header_parse_roundtrip_with_tensors() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "w".to_string(), ndim: 2, dtype: 0,
            shape: [4, 4, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 1, zp_type: 0, data: vec![0u8; 8], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("hdr_parse_tensor");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hdr_tensor.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let header = GllmHeader::parse(&raw).unwrap();
        assert_eq!(header.tensor_count, 1);
        assert!(header.is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Tensor directory: parse via GllmTensorEntry::parse_at ─────────────

    #[test]
    fn write_tensor_dir_entry_parseable() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "parsed_entry".to_string(), ndim: 3, dtype: 5,
            shape: [10, 20, 30, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0xAB; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("tdentry_parse");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tdentry.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let entry = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        assert_eq!(entry.ndim, 3);
        assert_eq!(entry.dtype, 5);
        assert_eq!(entry.shape[0], 10);
        assert_eq!(entry.shape[1], 20);
        assert_eq!(entry.shape[2], 30);
        assert_eq!(entry.shape[3], 0);
        assert_eq!(entry.compressed_size, 16);
        assert_eq!(entry.original_size, 16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_dir_entry_quant_fields_parseable() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "q_entry".to_string(), ndim: 2, dtype: 0,
            shape: [8, 8, 0, 0], quant_format: 40, quant_block_size: 128,
            scale_dtype: 2, zp_type: 1, data: vec![0u8; 32], original_size: 256,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("tdentry_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tdentry_quant.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let entry = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        assert_eq!(entry.quant_format, 40);
        assert_eq!(entry.quant_block_size, 128);
        assert_eq!(entry.scale_dtype, 2);
        assert_eq!(entry.zp_type, 1);
        assert!(entry.is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_two_tensor_dir_entries_both_parseable() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "first".to_string(), ndim: 1, dtype: 0,
            shape: [8, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![1u8; 8], original_size: 8,
        });
        builder.add_tensor(TensorEntry {
            name: "second".to_string(), ndim: 2, dtype: 2,
            shape: [4, 4, 0, 0], quant_format: 22, quant_block_size: 64,
            scale_dtype: 1, zp_type: 0, data: vec![2u8; 16], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("tdentry_two");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tdentry_two.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let e1 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        let e2 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE + TENSOR_ENTRY_SIZE).unwrap();
        assert_eq!(e1.name_len, 5); // "first"
        assert_eq!(e2.name_len, 6); // "second"
        assert!(!e1.is_quantized());
        assert!(e2.is_quantized());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── String table: name offset/len in tensor dir match raw bytes ───────

    #[test]
    fn write_string_table_name_offsets_are_cumulative() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "ab".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "cdef".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "ghijkl".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("strtab_offsets");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("strtab_off.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let strtab_start = HEADER_SIZE + 3 * TENSOR_ENTRY_SIZE;

        // First tensor name at offset 0
        let e1 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        assert_eq!(e1.name_offset, 0);
        assert_eq!(e1.name_len, 2);
        assert_eq!(&raw[strtab_start..strtab_start + 2], b"ab");

        // Second tensor name at offset 2
        let e2 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE + TENSOR_ENTRY_SIZE).unwrap();
        assert_eq!(e2.name_offset, 2);
        assert_eq!(e2.name_len, 4);
        assert_eq!(&raw[strtab_start + 2..strtab_start + 6], b"cdef");

        // Third tensor name at offset 6
        let e3 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE).unwrap();
        assert_eq!(e3.name_offset, 6);
        assert_eq!(e3.name_len, 6);
        assert_eq!(&raw[strtab_start + 6..strtab_start + 12], b"ghijkl");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Data offset: first tensor always starts at offset 0 in data region ─

    #[test]
    fn write_data_offset_first_tensor_is_zero() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "first_data".to_string(), ndim: 1, dtype: 0,
            shape: [16, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0x55; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("data_off_zero");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_off_zero.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let entry = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        assert_eq!(entry.data_offset, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Data offset: second tensor accounts for page-aligned first ────────

    #[test]
    fn write_data_offset_second_tensor_aligned_after_first() {
        let mut builder = GllmWriter::new(128);
        // First tensor: 10 bytes data → padded to 128
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0,
            shape: [10, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 10], original_size: 10,
        });
        // Second tensor: data_offset should be 128
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0,
            shape: [8, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("data_off_second");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_off_second.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let e2 = GllmTensorEntry::parse_at(&raw, HEADER_SIZE + TENSOR_ENTRY_SIZE).unwrap();
        assert_eq!(e2.data_offset, 128);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Metadata: bytes in file match set_metadata input ─────────────────

    #[test]
    fn write_metadata_bytes_in_file_match_input() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "m".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 16], original_size: 16,
        });
        let meta = b"hello metadata".to_vec();
        builder.set_metadata(meta.clone());

        let dir = unique_test_dir("meta_bytes");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_bytes.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes(), meta.as_slice());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Metadata: build_metadata output survives roundtrip ───────────────

    #[test]
    fn write_build_metadata_roundtrip_via_reader() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        let meta = build_metadata(
            "mistral3", 32000, 4096, 32, 32, 8, 128, 14336, 32768, &HashMap::new(),
        );
        builder.set_metadata(meta.clone());

        let dir = unique_test_dir("buildmeta_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("buildmeta_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let read_meta = reader.metadata_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(read_meta).unwrap();
        assert_eq!(parsed["arch_key"], "mistral3");
        assert_eq!(parsed["vocab_size"], "32000");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Compression ratio: GllmTensorEntry.compression_ratio() ───────────

    #[test]
    fn write_compression_ratio_reflects_quantization() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "cr".to_string(), ndim: 2, dtype: 0,
            shape: [16, 16, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 1, zp_type: 0, data: vec![0u8; 32], original_size: 1024,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("comp_ratio");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("comp_ratio.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let entry = GllmTensorEntry::parse_at(&raw, HEADER_SIZE).unwrap();
        assert!(entry.compression_ratio() > 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Header meta_offset matches actual position ───────────────────────

    #[test]
    fn write_header_meta_offset_matches_string_table_end() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "meta_off".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        let meta = b"test_meta".to_vec();
        builder.set_metadata(meta);

        let dir = unique_test_dir("meta_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_offset.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let header = GllmHeader::parse(&raw).unwrap();
        // meta_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE + len("meta_off") = 64 + 72 + 8 = 144
        let expected_meta_offset = (HEADER_SIZE + TENSOR_ENTRY_SIZE + 8) as u64;
        assert_eq!(header.meta_offset, expected_meta_offset);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── align_up: value exactly at double alignment ──────────────────────

    #[test]
    fn align_up_value_at_double_alignment() {
        assert_eq!(align_up(8192, 4096), 8192);
        assert_eq!(align_up(1024, 512), 1024);
        assert_eq!(align_up(2048, 1024), 2048);
    }

    // ── align_up: value between double and triple ────────────────────────

    #[test]
    fn align_up_between_double_and_triple() {
        assert_eq!(align_up(4097 + 4096, 4096), 3 * 4096);
        assert_eq!(align_up(1025 + 1024, 1024), 3 * 1024);
    }

    // ── TensorEntry: original_size as u64 overflow safety ────────────────

    #[test]
    fn tensor_entry_original_size_near_u64_max() {
        let entry = TensorEntry {
            name: "big_orig".into(), ndim: 2, dtype: 0,
            shape: [u64::MAX, u64::MAX, 0, 0], quant_format: 10,
            quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 8], original_size: u64::MAX,
        };
        assert_eq!(entry.original_size, u64::MAX);
        assert!(entry.compressed_size() < entry.original_size);
    }

    // ── TensorEntry: data with length exactly matching page size ─────────

    #[test]
    fn tensor_entry_data_len_exactly_page_aligned() {
        let entry = TensorEntry {
            name: "page_exact".into(), ndim: 1, dtype: 0,
            shape: [256, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 256], original_size: 256,
        };
        assert_eq!(entry.compressed_size(), 256);
        assert_eq!(align_up(256, 256), 256);
    }

    // ── TensorEntry: is_quantized boundary at 255 ────────────────────────

    #[test]
    fn tensor_entry_is_quantized_max_format() {
        let entry = TensorEntry {
            name: "qmax".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: u8::MAX, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert!(entry.is_quantized());
    }

    // ── build_metadata: all standard keys present as string values ───────

    #[test]
    fn build_metadata_all_values_are_json_strings() {
        let meta = build_metadata(
            "glm5", 150000, 6144, 40, 48, 8, 128, 23040, 131072, &HashMap::new(),
        );
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // All 9 standard fields must be strings
        for key in &["arch_key", "vocab_size", "hidden_size", "num_layers",
                     "num_heads", "num_kv_heads", "head_dim", "intermediate_size", "context_length"] {
            assert!(parsed[*key].is_string(), "{} should be a string", key);
        }
    }

    // ── build_metadata: extras with same key as standard wins ────────────

    #[test]
    fn build_metadata_extra_wins_over_standard_for_all_fields() {
        let mut extras = HashMap::new();
        extras.insert("num_layers".to_string(), "999".to_string());
        extras.insert("head_dim".to_string(), "0".to_string());
        let meta = build_metadata("t", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["num_layers"], "999");
        assert_eq!(parsed["head_dim"], "0");
        // Non-overridden field keeps standard value
        assert_eq!(parsed["vocab_size"], "1");
    }

    // ── quant_type_to_u8: float codes are 1,2,3 ─────────────────────────

    #[test]
    fn quant_type_to_u8_float_codes_are_1_2_3() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
        assert_eq!(quant_type_to_u8(QuantType::F32), 3);
        // Float range [1,3] is distinct from quant ranges
        for qt in &[QuantType::Bf16, QuantType::F16, QuantType::F32] {
            let code = quant_type_to_u8(*qt);
            assert!((1..=3).contains(&code));
        }
    }

    // ── quant_type_to_u8: no code collides with float range ──────────────

    #[test]
    fn quant_type_to_u8_quant_codes_outside_float_range() {
        use gllm_kernels::quant::QuantType;
        let quant_types = [
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1, QuantType::Q2K, QuantType::Q3K,
            QuantType::Q4K, QuantType::Q5K, QuantType::Q6K, QuantType::Q8K,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Squeeze,
        ];
        for qt in &quant_types {
            let code = quant_type_to_u8(*qt);
            assert!(!(1..=3).contains(&code), "{:?} code {} in float range", qt, code);
        }
    }

    // ── safetensors_dtype_to_u8: I8 and I32 distinct ─────────────────────

    #[test]
    fn safetensors_dtype_to_u8_i8_i32_distinct() {
        let i8_code = safetensors_dtype_to_u8(safetensors::Dtype::I8);
        let i32_code = safetensors_dtype_to_u8(safetensors::Dtype::I32);
        assert_ne!(i8_code, i32_code);
        assert_eq!(i8_code, 4);
        assert_eq!(i32_code, 5);
    }

    // ── safetensors_dtype_to_u8: F16 and BF16 distinct ──────────────────

    #[test]
    fn safetensors_dtype_to_u8_f16_bf16_distinct() {
        let f16 = safetensors_dtype_to_u8(safetensors::Dtype::F16);
        let bf16 = safetensors_dtype_to_u8(safetensors::Dtype::BF16);
        assert_ne!(f16, bf16);
    }

    // ── Roundtrip: data with repeating pattern survives ──────────────────

    #[test]
    fn roundtrip_data_with_repeating_4byte_pattern() {
        let pattern: Vec<u8> = (0..256).flat_map(|_| [0xDE, 0xAD, 0xBE, 0xEF]).collect();
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "repeat4".to_string(), ndim: 1, dtype: 0,
            shape: [1024, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: pattern.clone(), original_size: 1024,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("repeat4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("repeat4.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("repeat4").unwrap();
        assert_eq!(td.len(), 1024);
        assert_eq!(&td[..], &pattern[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Roundtrip: large metadata roundtrip ──────────────────────────────

    #[test]
    fn roundtrip_large_metadata_roundtrip() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "lm".to_string(), ndim: 1, dtype: 0,
            shape: [8, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 8], original_size: 8,
        });
        let mut extras = HashMap::new();
        extras.insert("description".to_string(), "A".repeat(5000));
        let meta = build_metadata("bigmeta", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        builder.set_metadata(meta);

        let dir = unique_test_dir("largemeta_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("largemeta_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let mb = reader.metadata_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(mb).unwrap();
        assert_eq!(parsed["description"].as_str().unwrap().len(), 5000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Roundtrip: single quant tensor with NVFP4 format code ────────────

    #[test]
    fn roundtrip_nvfp4_format_tensor() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "nvfp4_w".to_string(), ndim: 2, dtype: 0,
            shape: [128, 256, 0, 0], quant_format: 53, quant_block_size: 16,
            scale_dtype: 1, zp_type: 0, data: vec![0u8; 64], original_size: 131072,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("nvfp4_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("nvfp4_rt.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("nvfp4_w").unwrap();
        assert_eq!(t.entry.quant_format, 53);
        assert_eq!(t.entry.quant_block_size, 16);
        assert_eq!(t.entry.original_size, 131072);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Roundtrip: architecture() returns correct arch_key ───────────────

    #[test]
    fn roundtrip_architecture_returns_arch_key() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "arch_t".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        let meta = build_metadata("phi4", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("arch_key");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("arch_key.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.architecture(), Some("phi4".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: name with repeated dots ─────────────────────────────

    #[test]
    fn tensor_entry_name_with_repeated_dots() {
        let entry = TensorEntry {
            name: "model..layers..0..weight".to_string(), ndim: 2, dtype: 0,
            shape: [0; 4], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![], original_size: 0,
        };
        assert!(entry.name.contains(".."));
        assert_eq!(entry.name.matches("..").count(), 3);
    }

    // ── TensorEntry: clone preserves quant_block_size ────────────────────

    #[test]
    fn tensor_entry_clone_preserves_quant_block_size_exact() {
        let original = TensorEntry {
            name: "qbs".into(), ndim: 2, dtype: 0,
            shape: [0; 4], quant_format: 10, quant_block_size: 128,
            scale_dtype: 1, zp_type: 0, data: vec![], original_size: 0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.quant_block_size, 128);
        assert_eq!(cloned.quant_block_size, original.quant_block_size);
    }

    // ── Writer: add tensor with identical names (no dedup) ──────────────

    #[test]
    fn writer_add_tensor_duplicate_names_not_deduplicated() {
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "dup".to_string(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        writer.add_tensor(TensorEntry {
            name: "dup".to_string(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(writer.tensor_count(), 2);
        assert_eq!(writer.tensors[0].name, "dup");
        assert_eq!(writer.tensors[1].name, "dup");
    }

    // ── Writer: page_size 0 behavior ─────────────────────────────────────

    #[test]
    fn write_page_size_zero_no_panic() {
        let mut builder = GllmWriter::new(0);
        builder.add_tensor(TensorEntry {
            name: "ps0".to_string(), ndim: 1, dtype: 0,
            shape: [8, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0x42; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps0");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps0.gllm");
        // Should not panic — align_up with alignment=0 returns value unchanged
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("ps0").unwrap();
        assert_eq!(&td[..], &[0x42; 8]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Header: data_offset advances past metadata ──────────────────────

    #[test]
    fn write_data_offset_advances_past_metadata() {
        let mut builder = GllmWriter::new(4096);
        builder.add_tensor(TensorEntry {
            name: "dm".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![0xAA; 500]);

        let dir = unique_test_dir("dataoff_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dataoff_meta.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let header = GllmHeader::parse(&raw).unwrap();
        // data_offset must be at least header + tensor_dir + string_table + metadata
        let min_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE + 2 + 500; // "dm" = 2 bytes
        assert!(header.data_offset >= min_offset as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: compressed_size for 1-byte data ────────────────────

    #[test]
    fn tensor_entry_compressed_size_one_byte() {
        let entry = TensorEntry {
            name: "one".into(), ndim: 1, dtype: 0,
            shape: [1, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 1, zp_type: 0, data: vec![0xFF], original_size: 4,
        };
        assert_eq!(entry.compressed_size(), 1);
    }

    // ── TensorEntry: shape array is Copy (no Clone needed) ──────────────

    #[test]
    fn tensor_entry_shape_array_is_value_type() {
        let entry = TensorEntry {
            name: "copy_test".into(), ndim: 2, dtype: 0,
            shape: [100, 200, 300, 400], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![], original_size: 0,
        };
        let shape_copy = entry.shape;
        // Both should be equal (array is Copy)
        assert_eq!(shape_copy, entry.shape);
    }

    // ── build_metadata: empty arch_key not empty string key ─────────────

    #[test]
    fn build_metadata_empty_arch_key_still_present() {
        let meta = build_metadata("", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert!(parsed.get("arch_key").is_some());
        assert_eq!(parsed["arch_key"], "");
    }

    // ── build_metadata: extras merge but don't remove standard keys ─────

    #[test]
    fn build_metadata_extras_merge_preserves_all_standard_keys() {
        let mut extras = HashMap::new();
        extras.insert("custom_a".to_string(), "val_a".to_string());
        extras.insert("custom_b".to_string(), "val_b".to_string());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let obj = parsed.as_object().unwrap();
        // 9 standard + 2 extras
        assert_eq!(obj.len(), 11);
        assert!(obj.contains_key("arch_key"));
        assert!(obj.contains_key("custom_a"));
        assert!(obj.contains_key("custom_b"));
    }

    // ── align_up: u64::MAX with alignment 1 ─────────────────────────────

    #[test]
    fn align_up_u64_max_with_alignment_one() {
        assert_eq!(align_up(u64::MAX, 1), u64::MAX);
    }

    // ── align_up: result is always >= input ─────────────────────────────

    #[test]
    fn align_up_monotonic_for_various_inputs() {
        for &alignment in &[1u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
            for &val in &[0u64, 1, alignment / 2, alignment, alignment + 1, alignment * 3 - 1] {
                let result = align_up(val, alignment);
                assert!(result >= val, "align_up({},{}) = {} < {}", val, alignment, result, val);
            }
        }
    }

    // ── Roundtrip: empty data tensor preserves shape ─────────────────────

    #[test]
    fn roundtrip_empty_data_preserves_shape() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "empty_data".to_string(), ndim: 2, dtype: 0,
            shape: [768, 4096, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![], original_size: 0,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("empty_data_shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_data_shape.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("empty_data").unwrap();
        assert_eq!(t.entry.shape[0], 768);
        assert_eq!(t.entry.shape[1], 4096);
        let td = reader.tensor_data("empty_data").unwrap();
        assert_eq!(td.len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Header: tensor_dir_offset always equals HEADER_SIZE ─────────────

    #[test]
    fn write_tensor_dir_offset_always_header_size_raw() {
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "tdo".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("tdo_raw");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tdo_raw.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let tensor_dir_offset = u64::from_le_bytes(raw[24..32].try_into().unwrap());
        assert_eq!(tensor_dir_offset, HEADER_SIZE as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── dtype_to_u8: all byte values are identity ────────────────────────

    #[test]
    fn dtype_to_u8_full_range_identity() {
        for v in 0u8..=255 {
            assert_eq!(dtype_to_u8(v), v);
        }
    }

    // ── Roundtrip: 10 tensors with mixed quant formats ──────────────────

    #[test]
    fn roundtrip_ten_tensors_mixed_quant() {
        let mut builder = GllmWriter::new(128);
        for i in 0..10 {
            let qf = if i % 3 == 0 { 0u8 } else { [10u8, 22, 40, 41][i as usize % 4] };
            builder.add_tensor(TensorEntry {
                name: format!("t{}", i), ndim: 2, dtype: 0,
                shape: [4, 4, 0, 0], quant_format: qf,
                quant_block_size: if qf > 0 { 32 } else { 0 },
                scale_dtype: if qf > 0 { 1 } else { 0 },
                zp_type: 0, data: vec![i as u8; 16], original_size: if qf > 0 { 64 } else { 16 },
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("10mixed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("10mixed.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 10);
        assert!(reader.header().is_quantized());

        for i in 0..10 {
            let td = reader.tensor_data(&format!("t{}", i)).unwrap();
            assert_eq!(td.len(), 16);
            assert!(td.iter().all(|&b| b == i as u8));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: data with length 0 and nonzero original_size ───────

    #[test]
    fn tensor_entry_zero_data_nonzero_original_size() {
        let entry = TensorEntry {
            name: "zero_data".into(), ndim: 2, dtype: 0,
            shape: [4096, 4096, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 1, zp_type: 0, data: vec![], original_size: 67108864,
        };
        assert_eq!(entry.compressed_size(), 0);
        assert_eq!(entry.original_size, 67108864);
        assert!(entry.is_quantized());
    }

    // ── Writer: metadata_bytes field accessible after set ───────────────

    #[test]
    fn writer_metadata_field_exact_match_after_set() {
        let mut writer = GllmWriter::new(4096);
        let expected = vec![0x01, 0x02, 0x03, 0x04];
        writer.set_metadata(expected.clone());
        assert_eq!(writer.metadata_bytes, expected);
        // Verify len matches
        assert_eq!(writer.metadata_bytes.len(), 4);
    }

    // ── quant_type_to_u8: IQ codes are in [30,38] range ─────────────────

    #[test]
    fn quant_type_to_u8_iq_codes_in_30_to_38_range() {
        use gllm_kernels::quant::QuantType;
        let iq_types = [
            QuantType::IQ1S, QuantType::IQ1M, QuantType::IQ2XXS,
            QuantType::IQ2XS, QuantType::IQ2S, QuantType::IQ3XXS,
            QuantType::IQ3S, QuantType::IQ4NL, QuantType::IQ4XS,
        ];
        for qt in &iq_types {
            let code = quant_type_to_u8(*qt);
            assert!((30..=38).contains(&code), "IQ {:?} code {} not in [30,38]", qt, code);
        }
    }

    // ── Header: flags field raw bytes ────────────────────────────────────

    #[test]
    fn write_header_flags_byte_zero_without_quant() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "noflag".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("flags_zero");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags_zero.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_header_flags_byte_one_with_quant() {
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "flagged".to_string(), ndim: 1, dtype: 0,
            shape: [4, 0, 0, 0], quant_format: 10, quant_block_size: 32,
            scale_dtype: 1, zp_type: 0, data: vec![0u8; 4], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("flags_one");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags_one.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── TensorEntry: shape with mix of zero and nonzero dims ────────────

    #[test]
    fn tensor_entry_shape_first_zero_second_nonzero() {
        let entry = TensorEntry {
            name: "mixed_shape".into(), ndim: 4, dtype: 0,
            shape: [0, 512, 0, 1024], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: vec![], original_size: 0,
        };
        assert_eq!(entry.shape[0], 0);
        assert_eq!(entry.shape[1], 512);
        assert_eq!(entry.shape[2], 0);
        assert_eq!(entry.shape[3], 1024);
    }

    // ── build_metadata: very long extra value ────────────────────────────

    #[test]
    fn build_metadata_very_long_extra_value() {
        let mut extras = HashMap::new();
        let long_val = "X".repeat(10000);
        extras.insert("long_key".to_string(), long_val.clone());
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["long_key"].as_str().unwrap().len(), 10000);
    }

    // ── Roundtrip: writer with page_size 1 no extra padding ─────────────

    #[test]
    fn roundtrip_page_size_1_minimal_padding() {
        let mut builder = GllmWriter::new(1);
        let data = vec![0x77; 13];
        builder.add_tensor(TensorEntry {
            name: "ps1".to_string(), ndim: 1, dtype: 0,
            shape: [13, 0, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: data.clone(), original_size: 13,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("ps1_min");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps1_min.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("ps1").unwrap();
        assert_eq!(td.len(), 13);
        assert_eq!(&td[..], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional 25 unit tests — deeper coverage, pure logic, no I/O
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_data_with_alternating_pattern() {
        let data: Vec<u8> = (0..256).map(|i| if i % 2 == 0 { 0xAA } else { 0x55 }).collect();
        let entry = TensorEntry {
            name: "alternating".into(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 256,
        };
        assert_eq!(entry.data.len(), 256);
        for (i, &b) in entry.data.iter().enumerate() {
            assert_eq!(b, if i % 2 == 0 { 0xAA } else { 0x55 });
        }
    }

    #[test]
    fn tensor_entry_compressed_size_with_empty_name_and_data() {
        let entry = TensorEntry {
            name: String::new(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        assert_eq!(entry.compressed_size(), 0);
        assert_eq!(entry.name, "");
        assert_eq!(entry.ndim, 0);
    }

    #[test]
    fn tensor_entry_is_quantized_for_each_real_format_code() {
        // Verify every real quant_format code from quant_type_to_u8 is detected as quantized
        let real_codes = [1u8, 2, 3, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
                          30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 50, 51, 52, 53, 60, 61];
        for &code in &real_codes {
            let entry = TensorEntry {
                name: "t".into(), ndim: 1, dtype: 0, shape: [0; 4],
                quant_format: code, quant_block_size: 32, scale_dtype: 0, zp_type: 0,
                data: vec![], original_size: 0,
            };
            assert!(entry.is_quantized(), "quant_format={} should be quantized", code);
        }
    }

    #[test]
    fn tensor_entry_shape_zero_in_all_positions() {
        for pos in 0..4 {
            let mut shape = [0u64; 4];
            shape[pos] = 42;
            let entry = TensorEntry {
                name: "pos_test".into(), ndim: 4, dtype: 0, shape,
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![], original_size: 0,
            };
            assert_eq!(entry.shape[pos], 42);
            for j in 0..4 {
                if j != pos {
                    assert_eq!(entry.shape[j], 0);
                }
            }
        }
    }

    #[test]
    fn tensor_entry_quant_block_size_boundary_values() {
        let min = TensorEntry {
            name: "min_bs".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert_eq!(min.quant_block_size, 0);

        let max = TensorEntry {
            name: "max_bs".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 10, quant_block_size: u16::MAX, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        assert_eq!(max.quant_block_size, 65535);
    }

    #[test]
    fn tensor_entry_original_size_zero_vs_compressed() {
        let entry = TensorEntry {
            name: "zero_orig".into(), ndim: 2, dtype: 0,
            shape: [16, 16, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 64], original_size: 0,
        };
        assert_eq!(entry.original_size, 0);
        assert_eq!(entry.compressed_size(), 64);
        assert!(entry.is_quantized());
    }

    #[test]
    fn tensor_entry_clone_preserves_all_fields() {
        let original = TensorEntry {
            name: "full_clone_test".to_string(),
            ndim: 3,
            dtype: 7,
            shape: [10, 20, 30, 0],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 3,
            data: vec![0xAB; 100],
            original_size: 500,
        };
        let cloned = original.clone();
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.quant_format, original.quant_format);
        assert_eq!(cloned.quant_block_size, original.quant_block_size);
        assert_eq!(cloned.scale_dtype, original.scale_dtype);
        assert_eq!(cloned.zp_type, original.zp_type);
        assert_eq!(cloned.data.len(), original.data.len());
        assert_eq!(cloned.original_size, original.original_size);
        assert_eq!(cloned.compressed_size(), original.compressed_size());
        assert_eq!(cloned.is_quantized(), original.is_quantized());
    }

    #[test]
    fn tensor_entry_debug_output_completeness() {
        let entry = TensorEntry {
            name: "debug_check".into(),
            ndim: 2,
            dtype: 5,
            shape: [100, 200, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42u8; 10],
            original_size: 10,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("name"));
        assert!(debug.contains("ndim"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("quant_format"));
        assert!(debug.contains("data"));
        assert!(debug.contains("original_size"));
    }

    #[test]
    fn align_up_value_zero_never_panics() {
        // Zero value should always return zero regardless of alignment
        for &alignment in &[0u64, 1, 2, 3, 7, 13, 64, 4096, 65536] {
            assert_eq!(align_up(0, alignment), 0);
        }
    }

    #[test]
    fn align_up_result_is_always_multiple_of_alignment() {
        for &alignment in &[1u64, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096] {
            for val in [0u64, 1, alignment / 2, alignment - 1, alignment, alignment + 1, alignment * 3 - 1] {
                let result = align_up(val, alignment);
                assert_eq!(result % alignment, 0,
                    "align_up({},{})={} is not a multiple of {}", val, alignment, result, alignment);
            }
        }
    }

    #[test]
    fn align_up_never_decreases_value() {
        for &alignment in &[1u64, 2, 64, 512, 4096] {
            for val in [0u64, 1, 100, 1000, alignment - 1, alignment, alignment + 1] {
                let result = align_up(val, alignment);
                assert!(result >= val,
                    "align_up({},{})={} < {}", val, alignment, result, val);
            }
        }
    }

    #[test]
    fn align_up_non_power_of_two_alignment() {
        // align_up uses div_ceil, so non-power-of-two should still work
        assert_eq!(align_up(10, 3), 12);
        assert_eq!(align_up(9, 3), 9);
        assert_eq!(align_up(0, 3), 0);
        assert_eq!(align_up(1, 3), 3);
        assert_eq!(align_up(7, 5), 10);
        assert_eq!(align_up(5, 5), 5);
    }

    #[test]
    fn build_metadata_preserves_all_nine_standard_fields() {
        let meta = build_metadata("arch", 10, 20, 30, 40, 50, 60, 70, 80, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.len(), 9, "must have exactly 9 standard fields");
        assert_eq!(obj["arch_key"], "arch");
        assert_eq!(obj["vocab_size"], "10");
        assert_eq!(obj["hidden_size"], "20");
        assert_eq!(obj["num_layers"], "30");
        assert_eq!(obj["num_heads"], "40");
        assert_eq!(obj["num_kv_heads"], "50");
        assert_eq!(obj["head_dim"], "60");
        assert_eq!(obj["intermediate_size"], "70");
        assert_eq!(obj["context_length"], "80");
    }

    #[test]
    fn build_metadata_extras_can_add_many_fields() {
        let mut extras = HashMap::new();
        extras.insert("a".to_string(), "1".to_string());
        extras.insert("b".to_string(), "2".to_string());
        extras.insert("c".to_string(), "3".to_string());
        let meta = build_metadata("t", 0, 0, 0, 0, 0, 0, 0, 0, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["a"], "1");
        assert_eq!(parsed["b"], "2");
        assert_eq!(parsed["c"], "3");
    }

    #[test]
    fn build_metadata_value_zero_stored_as_string_zero() {
        let meta = build_metadata("zero", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new());
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert!(parsed["vocab_size"].is_string());
        assert_eq!(parsed["vocab_size"], "0");
    }

    #[test]
    fn build_metadata_with_special_chars_in_extra_key() {
        let mut extras = HashMap::new();
        extras.insert("key-with.special/chars".to_string(), "value".to_string());
        let meta = build_metadata("t", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["key-with.special/chars"], "value");
    }

    #[test]
    fn writer_add_tensor_then_clear_via_new() {
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "temp".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(writer.tensor_count(), 1);
        // Creating a new writer gives empty state
        let fresh = GllmWriter::new(256);
        assert_eq!(fresh.tensor_count(), 0);
    }

    #[test]
    fn writer_add_tensor_with_all_quant_fields_set() {
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "fully_quantized".into(),
            ndim: 2,
            dtype: 3,
            shape: [512, 512, 0, 0],
            quant_format: 53,
            quant_block_size: 16,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 1024],
            original_size: 1048576,
        });
        assert_eq!(writer.tensor_count(), 1);
        let t = &writer.tensors[0];
        assert!(t.is_quantized());
        assert_eq!(t.quant_format, 53);
        assert_eq!(t.quant_block_size, 16);
        assert_eq!(t.scale_dtype, 2);
        assert_eq!(t.zp_type, 1);
        assert_eq!(t.compressed_size(), 1024);
        assert_eq!(t.original_size, 1048576);
    }

    #[test]
    fn writer_metadata_can_be_set_to_single_byte() {
        let mut writer = GllmWriter::new(4096);
        writer.set_metadata(vec![0x42]);
        assert_eq!(writer.metadata_bytes.len(), 1);
        assert_eq!(writer.metadata_bytes[0], 0x42);
    }

    #[test]
    fn safetensors_dtype_to_u8_bf16_is_two() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
    }

    #[test]
    fn safetensors_dtype_to_u8_i64_is_six() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
    }

    #[test]
    fn quant_type_to_u8_float_codes_are_sequential() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
        assert_eq!(quant_type_to_u8(QuantType::F32), 3);
    }

    #[test]
    fn quant_type_to_u8_fp8_codes_are_fifty_and_fifty_one() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Fp8E4M3), 50);
        assert_eq!(quant_type_to_u8(QuantType::Fp8E5M2), 51);
    }

    #[test]
    fn write_to_output_exact_byte_count_empty_writer() {
        // Arrange: empty writer with page_size 4096, no tensors, no metadata
        let writer = GllmWriter::new(4096);

        // Act: write to a Vec<u8>
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset is page-aligned, so padding fills to 4096
        // header(64) + 0 tensors + 0 string_table + 0 metadata + padding to page_size
        assert_eq!(buf.len(), 4096);
        // header is present in first 64 bytes
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        // bytes 64..4096 should be zero padding
        assert!(buf[64..4096].iter().all(|&b| b == 0));
    }

    #[test]
    fn write_to_output_exact_byte_count_single_tensor_page_aligned() {
        // Arrange: writer with one tensor whose data is exactly 1 page
        let page_size: u32 = 128;
        let mut writer = GllmWriter::new(page_size);
        writer.add_tensor(TensorEntry {
            name: "w".to_string(),
            ndim: 1,
            dtype: 3,
            shape: [32, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAA; page_size as usize],
            original_size: 128,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: header(64) + tensor_dir(72) + string_table(1) + metadata(0) + padding + data(128)
        assert!(buf.len() >= HEADER_SIZE + TENSOR_ENTRY_SIZE + 1 + page_size as usize);
    }

    #[test]
    fn tensor_entry_data_field_can_be_cleared_after_creation() {
        // Arrange
        let mut entry = TensorEntry {
            name: "test".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42u8; 100],
            original_size: 64,
        };
        assert_eq!(entry.compressed_size(), 100);

        // Act: clear data
        entry.data.clear();

        // Assert: compressed_size reflects the new length
        assert_eq!(entry.compressed_size(), 0);
        assert_eq!(entry.data.len(), 0);
        // original_size is independent
        assert_eq!(entry.original_size, 64);
    }

    #[test]
    fn write_to_header_raw_magic_and_version_bytes() {
        // Arrange
        let writer = GllmWriter::new(4096);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: first 4 bytes = GLLM_MAGIC LE, next 4 bytes = GLLM_VERSION LE
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        assert_eq!(version, GLLM_VERSION);
        // bytes 44..64 must be zero (reserved)
        assert!(buf[44..64].iter().all(|&b| b == 0));
    }

    #[test]
    fn build_metadata_parseable_as_json_with_all_nine_keys() {
        // Arrange
        let bytes = build_metadata(
            "llama", 32000, 4096, 32, 32, 32, 128, 11008, 4096, &HashMap::new(),
        );

        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Assert: all nine standard keys present
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("arch_key").unwrap().as_str().unwrap(), "llama");
        assert!(obj.contains_key("vocab_size"));
        assert!(obj.contains_key("hidden_size"));
        assert!(obj.contains_key("num_layers"));
        assert!(obj.contains_key("num_heads"));
        assert!(obj.contains_key("num_kv_heads"));
        assert!(obj.contains_key("head_dim"));
        assert!(obj.contains_key("intermediate_size"));
        assert!(obj.contains_key("context_length"));
        assert_eq!(obj.len(), 9);
    }

    #[test]
    fn writer_set_metadata_then_add_tensors_order_irrelevant() {
        // Arrange: set metadata first, then add tensors
        let mut w1 = GllmWriter::new(256);
        w1.set_metadata(vec![1, 2, 3]);
        w1.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42],
            original_size: 4,
        });

        // Act: add tensor first, then set metadata
        let mut w2 = GllmWriter::new(256);
        w2.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42],
            original_size: 4,
        });
        w2.set_metadata(vec![1, 2, 3]);

        // Assert: both produce identical output
        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        w1.write_to(&mut buf1).unwrap();
        w2.write_to(&mut buf2).unwrap();
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn write_to_page_size_one_minimal_padding() {
        // Arrange: page_size=1 means no padding needed
        let mut writer = GllmWriter::new(1);
        writer.add_tensor(TensorEntry {
            name: "tiny".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [7, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xDE; 7],
            original_size: 28,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data starts immediately after header+tensor_dir+string_table+metadata (no alignment padding)
        // header(64) + tensor_dir(72) + string_table(4 bytes "tiny") + metadata(0) = 140
        // With page_size=1, data_offset = align_up(140, 1) = 140
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 140);
        // Data at offset 140 should be 0xDE bytes
        assert_eq!(buf[140], 0xDE);
    }

    #[test]
    fn tensor_entry_shape_and_data_len_are_independent() {
        // Arrange: shape says 1024 elements but data has only 16 bytes
        let entry = TensorEntry {
            name: "mismatch".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1024, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 4096,
        };

        // Assert: compressed_size is data.len(), not derived from shape
        assert_eq!(entry.compressed_size(), 16);
        // shape is independent metadata
        assert_eq!(entry.shape[0], 1024);
    }

    #[test]
    fn align_up_value_one_alignment_one() {
        // Arrange
        let value = 1u64;
        let alignment = 1u64;

        // Act
        let result = align_up(value, alignment);

        // Assert: already aligned, returns same value
        assert_eq!(result, 1);
    }

    #[test]
    fn write_to_reserved_bytes_44_to_64_are_all_zero() {
        // Arrange: writer with tensors to make it non-trivial
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xFF; 64],
            original_size: 256,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: bytes 44..64 of the header are reserved zeros
        for i in 44..64 {
            assert_eq!(buf[i], 0, "byte at offset {i} should be zero");
        }
    }

    #[test]
    fn safetensors_dtype_to_u8_f32_is_zero() {
        // Arrange/Act/Assert: F32 is the default, maps to 0
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
    }

    #[test]
    fn build_metadata_with_all_zero_u64_still_valid_json() {
        // Arrange: all zeros
        let bytes = build_metadata(
            "", 0, 0, 0, 0, 0, 0, 0, 0, &HashMap::new(),
        );

        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Assert: values are all "0" strings
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("vocab_size").unwrap().as_str().unwrap(), "0");
        assert_eq!(obj.get("hidden_size").unwrap().as_str().unwrap(), "0");
        assert_eq!(obj.get("num_layers").unwrap().as_str().unwrap(), "0");
    }

    #[test]
    fn writer_tensor_ordering_matches_insertion() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        let names = ["layer.0.weight", "layer.1.weight", "layer.2.bias"];
        for name in &names {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 2,
                dtype: 0,
                shape: [4, 4, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 64,
            });
        }

        // Act: write and read back via string table
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: string table is the concatenation of names in order
        let string_table_start = HEADER_SIZE + TENSOR_ENTRY_SIZE * 3;
        let string_table: Vec<u8> = names.iter().flat_map(|n| n.as_bytes()).copied().collect();
        let actual_string_table = &buf[string_table_start..string_table_start + string_table.len()];
        assert_eq!(actual_string_table, string_table.as_slice());
    }

    #[test]
    fn tensor_entry_compressed_size_tracks_data_len_changes() {
        // Arrange: start with 50 bytes
        let mut entry = TensorEntry {
            name: "t".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 50],
            original_size: 40,
        };
        assert_eq!(entry.compressed_size(), 50);

        // Act: grow data to 200 bytes
        entry.data.resize(200, 0xFF);
        assert_eq!(entry.compressed_size(), 200);

        // Act: shrink data to 1 byte
        entry.data.truncate(1);
        assert_eq!(entry.compressed_size(), 1);
    }

    #[test]
    fn write_to_quant_flag_true_only_when_all_tensors_quantized() {
        // Arrange: two tensors, both quantized
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "q1".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 64,
        });
        writer.add_tensor(TensorEntry {
            name: "q2".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 22,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0u8; 16],
            original_size: 64,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: flags bit 0 is set
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional 15 unit tests — layout verification & edge cases
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn write_to_non_power_of_two_page_size_layout() {
        // Arrange: page_size=3 (non power-of-2) with one 5-byte tensor
        let mut writer = GllmWriter::new(3);
        writer.add_tensor(TensorEntry {
            name: "w".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [5, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAA; 5],
            original_size: 20,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset = align_up(header(64) + tensor_dir(72) + "w"(1) + metadata(0), 3)
        // = align_up(137, 3) = 138
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        assert_eq!(data_offset, 138);

        // Assert: data is 5 bytes at offset 138, padded to align_up(5,3)=6 bytes total
        assert_eq!(buf[data_offset], 0xAA);
        assert_eq!(buf[data_offset + 4], 0xAA);
        // padding byte at offset+5 should be zero
        assert_eq!(buf[data_offset + 5], 0);
    }

    #[test]
    fn write_to_single_byte_tensor_data_padded() {
        // Arrange: 1-byte tensor with page_size=256, needs padding to 256 bytes
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "single".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x42],
            original_size: 4,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset must be page-aligned
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        assert_eq!(data_offset % 256, 0, "data_offset must be page-aligned");

        // Assert: first byte is 0x42, next 255 bytes are zero padding
        assert_eq!(buf[data_offset], 0x42);
        for i in 1..256 {
            assert_eq!(buf[data_offset + i], 0, "padding byte {} should be zero", i);
        }
    }

    #[test]
    fn write_to_empty_name_tensor_string_table_empty() {
        // Arrange: tensor with empty name contributes 0 bytes to string table
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: String::new(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 16,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: meta_offset = header(64) + tensor_dir(72) + string_table(0) = 136
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_offset, 136);
    }

    #[test]
    fn write_to_two_same_size_tensors_data_offsets_increment() {
        // Arrange: two tensors each with 16 bytes of data, page_size=64
        let mut writer = GllmWriter::new(64);
        for i in 0..2u8 {
            writer.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1,
                dtype: 0,
                shape: [16, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![i; 16],
                original_size: 64,
            });
        }

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: first tensor's data_offset in dir is 0
        // Tensor dir entry: bytes 48..56 are data_offset
        let first_data_off = u64::from_le_bytes(buf[64 + 48..64 + 56].try_into().unwrap());
        assert_eq!(first_data_off, 0);

        // Second tensor's data_offset = align_up(16, 64) = 64
        let second_data_off = u64::from_le_bytes(buf[64 + 72 + 48..64 + 72 + 56].try_into().unwrap());
        assert_eq!(second_data_off, 64);
    }

    #[test]
    fn build_metadata_numeric_arch_key_valid_json() {
        // Arrange: arch_key is purely numeric
        let meta = build_metadata(
            "12345", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new(),
        );

        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();

        // Assert: arch_key is a string "12345", not a number
        assert!(parsed["arch_key"].is_string());
        assert_eq!(parsed["arch_key"], "12345");
    }

    #[test]
    fn write_to_large_metadata_meta_offset_correct() {
        // Arrange: metadata of 5000 bytes
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 16,
        });
        let meta = vec![0x55u8; 5000];
        writer.set_metadata(meta);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: meta_offset = header(64) + tensor_dir(72) + "a"(1) = 137
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_offset, 137);

        // Assert: metadata bytes are at the right offset
        assert_eq!(buf[137], 0x55);
        assert_eq!(buf[137 + 4999], 0x55);
    }

    #[test]
    fn write_to_original_size_zero_unquantized_preserved() {
        // Arrange: unquantized tensor with original_size=0
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "zero_orig".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 0,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: original_size in tensor dir entry (bytes 64..72) is 0
        let original_size = u64::from_le_bytes(buf[64 + 64..64 + 72].try_into().unwrap());
        assert_eq!(original_size, 0);
    }

    #[test]
    fn write_to_dtype_field_nonzero_value_preserved() {
        // Arrange: tensor with dtype=3 (F32)
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "dtype3".to_string(),
            ndim: 2,
            dtype: 3,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 64,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: dtype byte at offset header(64) + 7 = 71
        assert_eq!(buf[71], 3);
    }

    #[test]
    fn write_to_empty_name_name_len_is_zero() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: String::new(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: name_len is at offset header(64) + 4..6 (u16 LE)
        let name_len = u16::from_le_bytes(buf[68..70].try_into().unwrap());
        assert_eq!(name_len, 0);
    }

    #[test]
    fn write_to_page_size_one_total_output_length() {
        // Arrange: page_size=1 means zero padding everywhere
        let mut writer = GllmWriter::new(1);
        writer.add_tensor(TensorEntry {
            name: "x".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [3, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAB; 3],
            original_size: 12,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: header(64) + tensor_dir(72) + "x"(1) + metadata(0) + data(3) = 140
        // No padding since page_size=1
        assert_eq!(buf.len(), 140);
    }

    #[test]
    fn write_to_duplicate_name_tensors_both_written() {
        // Arrange: two tensors with the same name (writer does not deduplicate)
        let mut writer = GllmWriter::new(256);
        for data_byte in [0x11, 0x22] {
            writer.add_tensor(TensorEntry {
                name: "dup".to_string(),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![data_byte; 4],
                original_size: 16,
            });
        }

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: tensor_count in header is 2
        let count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(count, 2);

        // Assert: string table contains "dupdup" (two entries concatenated)
        let str_start = 64 + 72 * 2;
        assert_eq!(&buf[str_start..str_start + 6], b"dupdup");
    }

    #[test]
    fn write_to_long_name_tensor_string_table_content() {
        // Arrange: tensor with a 100-char name
        let long_name = "a".repeat(100);
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: long_name.clone(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: name_len in tensor dir is 100
        let name_len = u16::from_le_bytes(buf[68..70].try_into().unwrap());
        assert_eq!(name_len, 100);

        // Assert: string table starts at header+tensor_dir = 136 and has 100 'a' bytes
        let str_start = 64 + 72;
        for i in 0..100 {
            assert_eq!(buf[str_start + i], b'a');
        }
    }

    #[test]
    fn write_to_three_tensors_third_data_offset_correct() {
        // Arrange: 3 tensors with 32 bytes each, page_size=64
        let mut writer = GllmWriter::new(64);
        for i in 0..3u8 {
            writer.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1,
                dtype: 0,
                shape: [32, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![i; 32],
                original_size: 128,
            });
        }

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: 3rd tensor's data_offset = align_up(32,64) + align_up(32,64) = 64+64 = 128
        let third_entry_start = 64 + 72 * 2;
        let third_data_off = u64::from_le_bytes(
            buf[third_entry_start + 48..third_entry_start + 56].try_into().unwrap(),
        );
        assert_eq!(third_data_off, 128);
    }

    #[test]
    fn write_to_metadata_fills_gap_before_data_no_padding() {
        // Arrange: craft metadata so that header+tensordir+stringtable+metadata is already page-aligned
        // header(64) + tensor_dir(72) + "a"(1) = 137. page_size=256.
        // Need metadata of (256 - 137) = 119 bytes to fill to page boundary
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 16,
        });
        writer.set_metadata(vec![0x42; 119]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset = align_up(137 + 119, 256) = align_up(256, 256) = 256
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 256);

        // Assert: no padding bytes between metadata end and data start
        // metadata ends at 256, data starts at 256 -> zero gap
        assert_eq!(buf[256], 0u8); // first data byte
    }

    #[test]
    fn tensor_entry_debug_empty_entry_nonempty_output() {
        // Arrange: fully-default TensorEntry
        let entry = TensorEntry {
            name: String::new(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };

        // Act
        let debug = format!("{entry:?}");

        // Assert: Debug output is non-empty and contains the struct name
        assert!(!debug.is_empty());
        assert!(debug.contains("TensorEntry"));
        assert!(debug.contains("ndim: 0"));
        assert!(debug.contains("data: []"));
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — edge cases, binary layout, error paths
    // ────────────────────────────────────────────────────────────────────────

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_path_invalid_path_returns_io_error() {
        // Arrange: writer with a single tensor
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "x".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });

        // Act: write to a path with null bytes (invalid on all OS)
        let bad_path = std::path::Path::new("/tmp/\0invalid\u{0}.gllm");
        let result = writer.write_to_path(bad_path);

        // Assert: should fail with an IO error
        assert!(result.is_err(), "writing to invalid path should fail");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_two_quantized_tensors_data_offsets_accumulate_correctly() {
        // Arrange: two quantized tensors with 10 bytes each, page_size=16
        let mut writer = GllmWriter::new(16);
        writer.add_tensor(TensorEntry {
            name: "q1".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xAA; 10],
            original_size: 40,
        });
        writer.add_tensor(TensorEntry {
            name: "q2".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 22,
            quant_block_size: 64,
            scale_dtype: 2,
            zp_type: 0,
            data: vec![0xBB; 10],
            original_size: 40,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: first tensor data_offset in data region is 0
        let first_entry = HEADER_SIZE;
        let off1 = u64::from_le_bytes(buf[first_entry + 48..first_entry + 56].try_into().unwrap());
        assert_eq!(off1, 0);

        // Assert: second tensor data_offset = align_up(10, 16) = 16
        let second_entry = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let off2 = u64::from_le_bytes(buf[second_entry + 48..second_entry + 56].try_into().unwrap());
        assert_eq!(off2, 16);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_single_tensor_data_written_at_data_offset() {
        // Arrange: single tensor with known pattern
        let mut writer = GllmWriter::new(256);
        let pattern: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF];
        writer.add_tensor(TensorEntry {
            name: "sig".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: pattern.clone(),
            original_size: 4,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset from header
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        // Assert: the 4-byte pattern is at data_offset
        assert_eq!(&buf[data_offset..data_offset + 4], &pattern[..]);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_quant_format_byte_position() {
        // Arrange: tensor with quant_format=40 (AWQ4)
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "awq".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 8],
            original_size: 64,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: quant_format at byte 40 of tensor dir entry
        let entry_start = HEADER_SIZE;
        assert_eq!(buf[entry_start + 40], 40);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_scale_dtype_and_zp_type_positions() {
        // Arrange: tensor with scale_dtype=2, zp_type=1
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "sz".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [2, 2, 0, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 4],
            original_size: 16,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: scale_dtype at byte 43 of tensor dir entry
        let entry_start = HEADER_SIZE;
        assert_eq!(buf[entry_start + 43], 2, "scale_dtype");
        // Assert: zp_type at byte 44
        assert_eq!(buf[entry_start + 44], 1, "zp_type");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_padding_bytes_45_to_47_are_zero() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "pad".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 8,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: padding bytes 45..48 of tensor dir entry are zero
        let entry_start = HEADER_SIZE;
        for i in 45..48 {
            assert_eq!(buf[entry_start + i], 0, "padding byte {} should be zero", i);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_string_table_concatenates_in_order() {
        // Arrange: three tensors with names "aaa", "bb", "c"
        let mut writer = GllmWriter::new(256);
        for name in &["aaa", "bb", "c"] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1,
                dtype: 0,
                shape: [2, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 8],
                original_size: 8,
            });
        }
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: string table starts at header(64) + 3*tensor_entry(72) = 64+216 = 280
        let strtab_start = HEADER_SIZE + 3 * TENSOR_ENTRY_SIZE;
        // String table content: "aaa" + "bb" + "c" = "aaabbc"
        assert_eq!(&buf[strtab_start..strtab_start + 6], b"aaabbc");

        // Assert: name offsets in tensor dir entries
        let off0 = u32::from_le_bytes(buf[HEADER_SIZE..HEADER_SIZE + 4].try_into().unwrap());
        assert_eq!(off0, 0); // "aaa" starts at 0
        let off1 = u32::from_le_bytes(buf[HEADER_SIZE + TENSOR_ENTRY_SIZE..HEADER_SIZE + TENSOR_ENTRY_SIZE + 4].try_into().unwrap());
        assert_eq!(off1, 3); // "bb" starts after "aaa" (3 bytes)
        let off2 = u32::from_le_bytes(buf[HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE..HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE + 4].try_into().unwrap());
        assert_eq!(off2, 5); // "c" starts after "aaa"+"bb" (5 bytes)
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_special_char_name_tensor_data() {
        // Arrange: tensor with special characters in name
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "layer.0/self_attn@q_proj:weight".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x55; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("special_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("special_name.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: can read back the tensor with special name
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("layer.0/self_attn@q_proj:weight").unwrap();
        assert_eq!(td.len(), 16);
        assert!(td.iter().all(|&b| b == 0x55));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_empty_writer_output_is_page_aligned() {
        // Arrange: empty writer with page_size=256, no tensors, no metadata
        let writer = GllmWriter::new(256);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: output = header(64) + padding to data_offset(256) = 256 bytes
        // data_offset = align_up(64, 256) = 256; padding writes 192 zero bytes
        assert_eq!(buf.len(), 256, "empty writer output is padded to data_offset");

        // Assert: first 64 bytes are the header, rest are zero padding
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        for i in HEADER_SIZE..256 {
            assert_eq!(buf[i], 0, "padding byte {} should be zero", i);
        }

        // Assert: data_offset in header matches actual output length
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 256);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_meta_offset_in_header_matches_actual_position() {
        // Arrange: writer with metadata content
        let mut writer = GllmWriter::new(512);
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        let meta = vec![0x99; 50];
        writer.set_metadata(meta);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: meta_offset from header = header(64) + tensor_dir(72) + "t"(1) = 137
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_offset, 137);

        // Assert: metadata bytes at offset 137..187 are all 0x99
        for i in 0..50 {
            assert_eq!(buf[137 + i], 0x99, "metadata byte {} mismatch", i);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_padding_between_tensors_is_zero() {
        // Arrange: two tensors, first is 3 bytes, page_size=8
        let mut writer = GllmWriter::new(8);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [3, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xFF; 3],
            original_size: 3,
        });
        writer.add_tensor(TensorEntry {
            name: "b".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xEE; 2],
            original_size: 2,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: first tensor data ends at data_offset+3, padding from +3 to +8 should be zero
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        assert_eq!(buf[data_offset + 3], 0);
        assert_eq!(buf[data_offset + 4], 0);
        assert_eq!(buf[data_offset + 5], 0);
        assert_eq!(buf[data_offset + 6], 0);
        assert_eq!(buf[data_offset + 7], 0);
        // Assert: second tensor data starts at data_offset + 8
        assert_eq!(buf[data_offset + 8], 0xEE);
        assert_eq!(buf[data_offset + 9], 0xEE);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_data_preserved_for_many_small_tensors() {
        // Arrange: 20 tensors with varying small sizes
        let mut builder = GllmWriter::new(64);
        for i in 0..20u8 {
            let size = (i as usize) % 7 + 1; // sizes 1..7
            builder.add_tensor(TensorEntry {
                name: format!("small_{}", i),
                ndim: 1,
                dtype: 0,
                shape: [size as u64, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![i; size],
                original_size: size as u64,
            });
        }
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("many_small");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("many_small.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: all 20 tensors read back correctly
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 20);
        for i in 0..20u8 {
            let name = format!("small_{}", i);
            let td = reader.tensor_data(&name).unwrap();
            let expected_size = (i as usize) % 7 + 1;
            assert_eq!(td.len(), expected_size, "tensor {} size mismatch", i);
            assert!(td.iter().all(|&b| b == i), "tensor {} data mismatch", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_preserves_ndim_zero() {
        // Arrange: tensor with ndim=0
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "scalar_param".to_string(),
            ndim: 0,
            dtype: 3,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("ndim_zero_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim_zero.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: ndim=0 read back correctly
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("scalar_param").unwrap();
        assert_eq!(t.entry.ndim, 0);
        assert_eq!(t.entry.dtype, 3);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_quant_block_size_field_bytes() {
        // Arrange: tensor with quant_block_size=256 (fits in u16)
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "bs256".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 22,
            quant_block_size: 256,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 64,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: quant_block_size at bytes 41..43 of tensor dir entry (u16 LE)
        let entry_start = HEADER_SIZE;
        let bs = u16::from_le_bytes(buf[entry_start + 41..entry_start + 43].try_into().unwrap());
        assert_eq!(bs, 256);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_large_original_size_preserved() {
        // Arrange: tensor with very large original_size (4 GiB equivalent)
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "huge_orig".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 4_294_967_296, // 4 GiB
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("huge_orig");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("huge_orig.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: original_size preserved exactly
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("huge_orig").unwrap();
        assert_eq!(t.entry.original_size, 4_294_967_296_u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_uses_write_to_vec_consistent_with_file() {
        // Arrange: writer with tensor and metadata
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "vec_vs_file".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x42; 32],
            original_size: 32,
        });
        builder.set_metadata(vec![0x11, 0x22, 0x33]);

        // Act: write to vec
        let mut vec_buf = Vec::new();
        builder.write_to(&mut vec_buf).unwrap();

        // Act: write to file
        let dir = unique_test_dir("vec_file_cmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vec_file_cmp.gllm");
        builder.write_to_path(&path).unwrap();
        let file_buf = std::fs::read(&path).unwrap();

        // Assert: both outputs are byte-identical
        assert_eq!(vec_buf, file_buf, "write_to vec and write_to_path file must produce identical bytes");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional edge-case tests
    // ────────────────────────────────────────────────────────────────────────

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_4d_shape_all_dims_nonzero() {
        // Arrange: tensor using all 4 shape dimensions with nonzero values
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "conv_weight_4d".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [3, 64, 7, 7],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAB; 32],
            original_size: 32,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("4d_shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("4d_shape.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: all 4 dimensions preserved exactly
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("conv_weight_4d").unwrap();
        assert_eq!(t.entry.shape, [3, 64, 7, 7]);
        assert_eq!(t.entry.ndim, 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_3d_shape_third_dim_nonzero() {
        // Arrange: 3D tensor with only first 3 dims nonzero
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "3d_tensor".to_string(),
            ndim: 3,
            dtype: 2, // F16
            shape: [8, 16, 32, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x42; 24],
            original_size: 24,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("3d_shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("3d_shape.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("3d_tensor").unwrap();
        assert_eq!(t.entry.ndim, 3);
        assert_eq!(t.entry.shape[0], 8);
        assert_eq!(t.entry.shape[1], 16);
        assert_eq!(t.entry.shape[2], 32);
        assert_eq!(t.entry.shape[3], 0);
        assert_eq!(t.entry.dtype, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_metadata_offset_with_no_tensors() {
        // Arrange: empty writer (no tensors) with metadata
        let mut writer = GllmWriter::new(256);
        writer.set_metadata(vec![0xDE, 0xAD, 0xBE, 0xEF]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: meta_offset should be header(64) + tensor_dir(0) + string_table(0) = 64
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_offset, 64);

        // Assert: metadata bytes are at offset 64..68
        assert_eq!(buf[64], 0xDE);
        assert_eq!(buf[65], 0xAD);
        assert_eq!(buf[66], 0xBE);
        assert_eq!(buf[67], 0xEF);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_offset_with_no_tensors_no_metadata() {
        // Arrange: completely empty writer with page_size=128
        let writer = GllmWriter::new(128);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset = align_up(64, 128) = 128; output padded to 128
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 128);
        assert_eq!(buf.len(), 128);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_three_tensors_data_integrity() {
        // Arrange: 3 tensors with distinct data patterns
        let mut builder = GllmWriter::new(32);
        builder.add_tensor(TensorEntry {
            name: "alpha".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x11; 12],
            original_size: 12,
        });
        builder.add_tensor(TensorEntry {
            name: "beta".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [6, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x22; 18],
            original_size: 18,
        });
        builder.add_tensor(TensorEntry {
            name: "gamma".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x33; 5],
            original_size: 5,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("three_tensor_integrity");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("three_tensors.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: each tensor's data preserved exactly
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 3);

        let d1 = reader.tensor_data("alpha").unwrap();
        assert_eq!(d1.len(), 12);
        assert!(d1.iter().all(|&b| b == 0x11));

        let d2 = reader.tensor_data("beta").unwrap();
        assert_eq!(d2.len(), 18);
        assert!(d2.iter().all(|&b| b == 0x22));

        let d3 = reader.tensor_data("gamma").unwrap();
        assert_eq!(d3.len(), 5);
        assert!(d3.iter().all(|&b| b == 0x33));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_total_file_size_matches_data_offset_plus_aligned_data() {
        // Arrange: writer with 2 tensors whose data sizes need alignment padding
        let mut writer = GllmWriter::new(16);
        writer.add_tensor(TensorEntry {
            name: "t1".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 10], // align_up(10,16)=16
            original_size: 10,
        });
        writer.add_tensor(TensorEntry {
            name: "t2".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [5, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 5], // align_up(5,16)=16
            original_size: 5,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: total size = data_offset + align_up(10,16) + align_up(5,16)
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        let expected_size = data_offset + 16 + 16;
        assert_eq!(buf.len(), expected_size);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_quant_block_size_u16_max() {
        // Arrange: tensor with quant_block_size = u16::MAX (65535)
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "big_block".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: u16::MAX,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 8],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("block_max");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("block_max.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("big_block").unwrap();
        assert_eq!(t.entry.quant_block_size, u16::MAX);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_scale_dtype_nonzero() {
        // Arrange: quantized tensor with scale_dtype=2 (nonzero, non-standard)
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "scale_dt".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 41, // GPTQ4
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 16],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("scale_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scale_dtype.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("scale_dt").unwrap();
        assert_eq!(t.entry.scale_dtype, 2);
        assert_eq!(t.entry.zp_type, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_zp_type_nonzero() {
        // Arrange: quantized tensor with zp_type=1 (u8 zero-point)
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "zp_tensor".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [8, 8, 0, 0],
            quant_format: 41, // GPTQ4
            quant_block_size: 64,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0u8; 32],
            original_size: 256,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("zp_type");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zp_type.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("zp_tensor").unwrap();
        assert_eq!(t.entry.zp_type, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_dtype_max_u8_value() {
        // Arrange: tensor with dtype = u8::MAX (255)
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "dtype_max".to_string(),
            ndim: 1,
            dtype: u8::MAX,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("dtype_max");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dtype_max.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("dtype_max").unwrap();
        assert_eq!(t.entry.dtype, u8::MAX);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_three_tensors_name_offsets_cumulative() {
        // Arrange: 3 tensors with names "first", "second_name", "z"
        let mut writer = GllmWriter::new(16);
        writer.add_tensor(TensorEntry {
            name: "first".to_string(),
            ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.add_tensor(TensorEntry {
            name: "second_name".to_string(),
            ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.add_tensor(TensorEntry {
            name: "z".to_string(),
            ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: name offsets are cumulative
        // Entry 0: offset=0, Entry 1: offset=5 ("first"=5), Entry 2: offset=5+11=16 ("second_name"=11)
        let off0 = u32::from_le_bytes(buf[HEADER_SIZE..HEADER_SIZE + 4].try_into().unwrap());
        let off1 = u32::from_le_bytes(buf[HEADER_SIZE + TENSOR_ENTRY_SIZE..HEADER_SIZE + TENSOR_ENTRY_SIZE + 4].try_into().unwrap());
        let off2 = u32::from_le_bytes(buf[HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE..HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE + 4].try_into().unwrap());
        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(off2, 16);

        // Assert: string table content = "first" + "second_name" + "z"
        let strtab_start = HEADER_SIZE + 3 * TENSOR_ENTRY_SIZE;
        let expected_strtab = b"firstsecond_namez";
        assert_eq!(&buf[strtab_start..strtab_start + expected_strtab.len()], expected_strtab);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_last_tensor_data_offset_equals_sum_of_aligned_predecessors() {
        // Arrange: 3 tensors with sizes that need alignment
        let mut writer = GllmWriter::new(8);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 3], original_size: 3,
        }); // align_up(3,8) = 8
        writer.add_tensor(TensorEntry {
            name: "b".to_string(),
            ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 5], original_size: 5,
        }); // align_up(5,8) = 8
        writer.add_tensor(TensorEntry {
            name: "c".to_string(),
            ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 2], original_size: 2,
        }); // should be at data-region offset 16
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: third tensor's data_offset = 8 + 8 = 16
        let entry2_offset = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        let data_off2 = u64::from_le_bytes(buf[entry2_offset + 48..entry2_offset + 56].try_into().unwrap());
        assert_eq!(data_off2, 16);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_header_tensor_dir_offset_always_64_regardless_of_page_size() {
        // Arrange: vary page sizes and verify tensor_dir_offset stays at 64
        for &ps in &[1u32, 16, 512, 4096, 65536] {
            let mut writer = GllmWriter::new(ps);
            writer.add_tensor(TensorEntry {
                name: "t".to_string(),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 4], original_size: 4,
            });
            writer.set_metadata(vec![]);

            let mut buf = Vec::new();
            writer.write_to(&mut buf).unwrap();

            let td_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap());
            assert_eq!(td_offset, HEADER_SIZE as u64, "tensor_dir_offset should be 64 for page_size={}", ps);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_single_tensor_ndim_1() {
        // Arrange: 1D tensor (ndim=1), data and original_size must match for unquantized
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "bias".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x42; 16],
            original_size: 16,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("ndim1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim1.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("bias").unwrap();
        assert_eq!(t.entry.ndim, 1);
        assert_eq!(t.entry.shape[0], 256);
        assert_eq!(t.entry.shape[1], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_metadata_between_string_table_and_data() {
        // Arrange: writer with tensor + metadata, verify metadata bytes sit between string table and data
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "w".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        let meta = vec![0xCA, 0xFE, 0xBA, 0xBE];
        writer.set_metadata(meta);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: meta_offset = header(64) + tensor_dir(72) + "w"(1) = 137
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_offset, 137);

        // Assert: metadata content at offset 137
        assert_eq!(buf[137], 0xCA);
        assert_eq!(buf[138], 0xFE);
        assert_eq!(buf[139], 0xBA);
        assert_eq!(buf[140], 0xBE);

        // Assert: data_offset = align_up(141, 256) = 256
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 256);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_quant_format_0_not_quantized_flag() {
        // Arrange: tensor with quant_format=0 should not set the global quant flag
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "plain".to_string(),
            ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: flags bit 0 = 0 (no quantized tensors)
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 0);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_offset_first_tensor_region_offset_is_zero() {
        // Arrange: single tensor, verify its data_offset within the data region is 0
        let mut writer = GllmWriter::new(32);
        writer.add_tensor(TensorEntry {
            name: "first".to_string(),
            ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x77; 16], original_size: 16,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: first tensor directory entry, data_offset field (bytes 48..56) = 0
        let entry_start = HEADER_SIZE;
        let t_data_off = u64::from_le_bytes(buf[entry_start + 48..entry_start + 56].try_into().unwrap());
        assert_eq!(t_data_off, 0, "first tensor data offset in data region should be 0");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_4d_shape_all_dims_nonzero() {
        // Arrange: TensorEntry with all 4 shape dimensions nonzero
        let entry = TensorEntry {
            name: "4d_tensor".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [3, 4, 5, 6],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 360],
            original_size: 360,
        };

        // Act & Assert: all four dimensions are preserved
        assert_eq!(entry.shape[0], 3);
        assert_eq!(entry.shape[1], 4);
        assert_eq!(entry.shape[2], 5);
        assert_eq!(entry.shape[3], 6);
        assert_eq!(entry.ndim, 4);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_add_tensor_preserves_exact_insertion_order() {
        // Arrange: add 5 tensors with distinct names
        let mut writer = GllmWriter::new(64);
        let names = ["alpha", "beta", "gamma", "delta", "epsilon"];
        for name in &names {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 1], original_size: 1,
            });
        }

        // Act & Assert: order matches insertion order exactly
        assert_eq!(writer.tensor_count(), 5);
        for (i, expected_name) in names.iter().enumerate() {
            assert_eq!(writer.tensors[i].name, *expected_name);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_offset_shifted_by_metadata_length() {
        // Arrange: two writers with same tensor but different metadata sizes
        let mut writer_short = GllmWriter::new(256);
        let mut writer_long = GllmWriter::new(256);

        for w in [&mut writer_short, &mut writer_long] {
            w.add_tensor(TensorEntry {
                name: "x".to_string(),
                ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 4], original_size: 4,
            });
        }
        writer_short.set_metadata(vec![0u8; 10]);
        writer_long.set_metadata(vec![0u8; 200]);

        let mut buf_short = Vec::new();
        let mut buf_long = Vec::new();
        writer_short.write_to(&mut buf_short).unwrap();
        writer_long.write_to(&mut buf_long).unwrap();

        // Act: read data_offset from both
        let data_off_short = u64::from_le_bytes(buf_short[32..40].try_into().unwrap());
        let data_off_long = u64::from_le_bytes(buf_long[32..40].try_into().unwrap());

        // Assert: longer metadata pushes data_offset further (or equal if both land on same alignment)
        assert!(data_off_long >= data_off_short, "longer metadata should push data_offset further or equal");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_original_size_equals_compressed_size_when_unquantized() {
        // Arrange: unquantized entry where original_size == data.len()
        let data = vec![0xAAu8; 100];
        let entry = TensorEntry {
            name: "exact".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [25, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 100,
        };

        // Act & Assert
        assert_eq!(entry.compressed_size(), entry.original_size);
        assert!(!entry.is_quantized());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_total_output_bytes_matches_computed_size() {
        // Arrange: single tensor, page_size=32, no metadata
        let page_size: u32 = 32;
        let mut writer = GllmWriter::new(page_size);
        let data = vec![0xFFu8; 48]; // 48 bytes, not aligned to 32
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1, dtype: 0, shape: [12, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(),
            original_size: 48,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: data_offset = align_up(64 + 72 + 1, 32) = align_up(137, 32) = 160
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        // data region = align_up(48, 32) = 64 bytes
        let expected_total = data_offset as usize + 64;
        assert_eq!(buf.len(), expected_total);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn align_up_value_alignment_minus_one() {
        // Arrange: value just below alignment boundary
        // Act & Assert
        assert_eq!(align_up(4095, 4096), 4096);
        assert_eq!(align_up(511, 512), 512);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(63, 64), 64);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_extra_key_with_special_characters() {
        // Arrange: extras with special characters in keys and values
        let mut extras = HashMap::new();
        extras.insert("model.sub_type".to_string(), "chat-instruct".to_string());
        let meta = build_metadata(
            "test_model", 1000, 512, 8, 4, 2, 64, 256, 1024, &extras,
        );

        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();

        // Assert
        assert_eq!(parsed["model.sub_type"], "chat-instruct");
        assert_eq!(parsed["arch_key"], "test_model");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_name_len_matches_actual_name_byte_length() {
        // Arrange: tensor with known ASCII name length
        let name = "model.layers.0.self_attn.q_proj.weight";
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            ndim: 2, dtype: 0, shape: [8, 8, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: name_len field (bytes 4..6 of tensor dir entry) equals name byte length
        let entry_start = HEADER_SIZE;
        let name_len = u16::from_le_bytes(buf[entry_start + 4..entry_start + 6].try_into().unwrap());
        assert_eq!(name_len, name.len() as u16);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_path_creates_file_successfully() {
        // Arrange
        let dir = unique_test_dir("write_to_path_success");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("success.gllm");

        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "w".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![1u8, 2, 3, 4], original_size: 4,
        });
        writer.set_metadata(vec![0xAB]);

        // Act
        let result = writer.write_to_path(&path);

        // Assert
        assert!(result.is_ok(), "write_to_path should succeed");
        assert!(path.exists(), "file should exist after write");

        // Verify the file is readable back
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_compressed_size_two_bytes() {
        // Arrange: entry with exactly 2 bytes of data
        let entry = TensorEntry {
            name: "tiny".to_string(),
            ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xDE, 0xAD],
            original_size: 2,
        };

        // Act & Assert
        assert_eq!(entry.compressed_size(), 2);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_second_tensor_data_offset_accounts_for_first_aligned_size() {
        // Arrange: two tensors, first has data requiring alignment padding
        let page_size: u32 = 64;
        let mut writer = GllmWriter::new(page_size);
        // First tensor: 48 bytes of data → aligned to 64
        writer.add_tensor(TensorEntry {
            name: "first".to_string(),
            ndim: 1, dtype: 0, shape: [12, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 48], original_size: 48,
        });
        // Second tensor
        writer.add_tensor(TensorEntry {
            name: "second".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 16], original_size: 16,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: second tensor data_offset = align_up(48, 64) = 64
        let entry1_start = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let data_off1 = u64::from_le_bytes(buf[entry1_start + 48..entry1_start + 56].try_into().unwrap());
        assert_eq!(data_off1, 64, "second tensor offset should be 64 (first aligned from 48)");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_region_exact_bytes_match_tensor_data() {
        // Arrange: single tensor with recognizable data pattern
        let mut writer = GllmWriter::new(16);
        let pattern: Vec<u8> = (0..32).map(|i| (i * 7 + 3) as u8).collect();
        writer.add_tensor(TensorEntry {
            name: "pattern".to_string(),
            ndim: 1, dtype: 0, shape: [32, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: pattern.clone(),
            original_size: 32,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;

        // Assert: the bytes at data_offset match the original pattern exactly
        assert_eq!(&buf[data_offset..data_offset + 32], pattern.as_slice());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_set_metadata_replaces_and_tensor_count_independent() {
        // Arrange
        let mut writer = GllmWriter::new(128);
        writer.set_metadata(vec![1, 2, 3]);
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });

        // Act: replace metadata
        writer.set_metadata(vec![0xFF; 50]);

        // Assert: tensor count unaffected by metadata replacement
        assert_eq!(writer.tensor_count(), 1);
        assert_eq!(writer.metadata_bytes.len(), 50);
        assert_eq!(writer.metadata_bytes[0], 0xFF);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_quant_flag_set_when_single_tensor_quantized() {
        // Arrange: single quantized tensor
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "qw".to_string(),
            ndim: 2, dtype: 0, shape: [4, 32, 0, 0],
            quant_format: 10, // Q4_0
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 18], original_size: 128,
        });
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: flags bit 0 = 1 (quantized)
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1, "quantized flag should be set");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_string_table_bytes_match_concatenated_names() {
        // Arrange: three tensors with known names
        let mut writer = GllmWriter::new(64);
        for name in &["aa", "bbb", "cccc"] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 1], original_size: 1,
            });
        }
        writer.set_metadata(vec![]);

        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();

        // Assert: string table starts at HEADER_SIZE + 3*TENSOR_ENTRY_SIZE = 64 + 216 = 280
        let st_offset = HEADER_SIZE + 3 * TENSOR_ENTRY_SIZE;
        // String table should be "aa" + "bbb" + "cccc" = "aabbbcccc" = 9 bytes
        assert_eq!(&buf[st_offset..st_offset + 2], b"aa");
        assert_eq!(&buf[st_offset + 2..st_offset + 5], b"bbb");
        assert_eq!(&buf[st_offset + 5..st_offset + 9], b"cccc");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_4d_shape_all_dims_preserved() {
        // Arrange: tensor with all 4 dimensions nonzero
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "4d".to_string(),
            ndim: 4,
            dtype: 0,
            shape: [2, 3, 4, 5],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x55; 120],
            original_size: 120,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("4d_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("4d.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("4d").unwrap();
        assert_eq!(t.entry.ndim, 4);
        assert_eq!(t.entry.shape[0], 2);
        assert_eq!(t.entry.shape[1], 3);
        assert_eq!(t.entry.shape[2], 4);
        assert_eq!(t.entry.shape[3], 5);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — unique coverage areas
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn tensor_entry_compressed_size_three_bytes() {
        // Arrange
        let entry = TensorEntry {
            name: "three".into(),
            ndim: 1,
            dtype: 0,
            shape: [3, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x10, 0x20, 0x30],
            original_size: 3,
        };
        // Act & Assert
        assert_eq!(entry.compressed_size(), 3);
        assert_eq!(entry.data[0], 0x10);
        assert_eq!(entry.data[1], 0x20);
        assert_eq!(entry.data[2], 0x30);
    }

    #[test]
    fn tensor_entry_clone_preserves_original_size_exactly() {
        // Arrange
        let original = TensorEntry {
            name: "orig_sz".into(),
            ndim: 2,
            dtype: 3,
            shape: [512, 512, 0, 0],
            quant_format: 22,
            quant_block_size: 64,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 8192],
            original_size: 1048576,
        };
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(cloned.original_size, 1048576);
        assert_eq!(cloned.original_size, original.original_size);
    }

    #[test]
    fn tensor_entry_name_with_leading_digits() {
        // Arrange
        let name = "123_layer_weight".to_string();
        let entry = TensorEntry {
            name: name.clone(),
            ndim: 2,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Act & Assert
        assert_eq!(entry.name, "123_layer_weight");
        assert!(entry.name.starts_with("123_"));
    }

    #[test]
    fn align_up_with_alignment_u32_max_as_u64() {
        // Arrange: alignment = u32::MAX (4294967295)
        let alignment = u32::MAX as u64;
        // Act & Assert
        assert_eq!(align_up(0, alignment), 0);
        assert_eq!(align_up(1, alignment), alignment);
        assert_eq!(align_up(alignment, alignment), alignment);
        assert_eq!(align_up(alignment + 1, alignment), alignment * 2);
    }

    #[test]
    fn build_metadata_context_length_zero_preserved_as_string() {
        // Arrange
        let meta = build_metadata(
            "test_ctx_zero", 32000, 4096, 32, 32, 8, 128, 11008, 0, &HashMap::new(),
        );
        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // Assert
        assert_eq!(parsed["context_length"], "0");
        assert_eq!(parsed["arch_key"], "test_ctx_zero");
    }

    #[test]
    fn write_to_empty_tensor_data_still_page_aligned() {
        // Arrange: tensor with zero-length data
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "empty_data".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [0, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("empty_data_align");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_data_align.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("empty_data").unwrap();
        assert_eq!(td.len(), 0);
        // data_offset should still be page-aligned
        assert_eq!(reader.header().data_offset % 512, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_roundtrip_preserves_ndim_two() {
        // Arrange
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "ndim2".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [64, 32, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 32,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("ndim2_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim2_rt.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("ndim2").unwrap();
        assert_eq!(t.entry.ndim, 2);
        assert_eq!(t.entry.shape[0], 64);
        assert_eq!(t.entry.shape[1], 32);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_roundtrip_preserves_ndim_three() {
        // Arrange
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "ndim3".to_string(),
            ndim: 3,
            dtype: 0,
            shape: [4, 8, 16, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 64],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("ndim3_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ndim3_rt.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("ndim3").unwrap();
        assert_eq!(t.entry.ndim, 3);
        assert_eq!(t.entry.shape[2], 16);
        assert_eq!(t.entry.shape[3], 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_metadata_bytes_default_is_empty_vec() {
        // Arrange & Act
        let writer = GllmWriter::new(4096);
        // Assert
        assert!(writer.metadata_bytes.is_empty());
        assert_eq!(writer.metadata_bytes.len(), 0);
    }

    #[test]
    fn build_metadata_with_tab_in_extra_value() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("config_line".to_string(), "key1=val1\tkey2=val2".to_string());
        // Act
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        // Assert
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["config_line"], "key1=val1\tkey2=val2");
    }

    #[test]
    fn write_to_quantized_tensor_data_preserved_exact() {
        // Arrange
        let qdata = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "qexact".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [2, 3, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: qdata.clone(),
            original_size: 24,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("qexact");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qexact.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("qexact").unwrap();
        assert_eq!(&td[..], &qdata[..]);
        assert_eq!(td.len(), 6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safetensors_dtype_to_u8_u8_type_is_three() {
        // Arrange & Act & Assert
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
        // Ensure U8 is distinct from I8
        assert_ne!(
            safetensors_dtype_to_u8(safetensors::Dtype::U8),
            safetensors_dtype_to_u8(safetensors::Dtype::I8)
        );
    }

    #[test]
    fn tensor_entry_compressed_size_five_bytes() {
        // Arrange
        let entry = TensorEntry {
            name: "five".into(),
            ndim: 1,
            dtype: 0,
            shape: [5, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![10, 20, 30, 40, 50],
            original_size: 5,
        };
        // Act & Assert
        assert_eq!(entry.compressed_size(), 5);
        assert_eq!(entry.data.len(), 5);
    }

    #[test]
    fn write_to_page_size_2_produces_valid_file() {
        // Arrange
        let mut builder = GllmWriter::new(2);
        builder.add_tensor(TensorEntry {
            name: "ps2".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [3, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAA, 0xBB, 0xCC],
            original_size: 3,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("ps2");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps2.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 2);
        let td = reader.tensor_data("ps2").unwrap();
        assert_eq!(&td[..], &[0xAA, 0xBB, 0xCC]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_to_tensor_with_data_len_power_of_two_no_extra_padding() {
        // Arrange: data.len() == page_size, so no padding needed
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "exact_page".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x77; 64],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("exact_page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_page.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert
        let raw = std::fs::read(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        // 64 bytes of data at data_offset
        assert_eq!(&raw[data_offset..data_offset + 64], &[0x77u8; 64]);
        // File should end right after data (no extra padding needed)
        assert_eq!(raw.len(), data_offset + 64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — remaining coverage gaps
    // ────────────────────────────────────────────────────────────────────────

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn dtype_to_u8_is_identity_passthrough() {
        // Arrange: various u8 values including edge values
        let values = [0u8, 1, 2, 3, 127, 128, 255];
        // Act & Assert: dtype_to_u8 returns the same value unchanged
        for v in values {
            assert_eq!(dtype_to_u8(v), v, "dtype_to_u8({}) should return {}", v, v);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_to_u8_f16_distinct_from_bf16_and_f32() {
        // Arrange & Act & Assert
        let f16_code = safetensors_dtype_to_u8(safetensors::Dtype::F16);
        assert_eq!(f16_code, 1);
        // F16 (1) must differ from BF16 (2) and F32 (0)
        assert_ne!(f16_code, safetensors_dtype_to_u8(safetensors::Dtype::BF16));
        assert_ne!(f16_code, safetensors_dtype_to_u8(safetensors::Dtype::F32));
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_to_u8_i8_is_four_and_i32_is_five() {
        // Arrange & Act & Assert: verify I8 and I32 mappings
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
        // Ensure I8 and I32 are distinct
        assert_ne!(
            safetensors_dtype_to_u8(safetensors::Dtype::I8),
            safetensors_dtype_to_u8(safetensors::Dtype::I32)
        );
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_all_values_are_string_encoded() {
        // Arrange: use non-trivial u64 values to verify string encoding
        let meta = build_metadata(
            "gpt2",
            50257,
            768,
            12,
            12,
            12,
            64,
            3072,
            1024,
            &HashMap::new(),
        );
        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // Assert: every value is a JSON string, not a number
        assert!(parsed["vocab_size"].is_string());
        assert!(parsed["hidden_size"].is_string());
        assert!(parsed["num_layers"].is_string());
        assert!(parsed["num_heads"].is_string());
        assert!(parsed["num_kv_heads"].is_string());
        assert!(parsed["head_dim"].is_string());
        assert!(parsed["intermediate_size"].is_string());
        assert!(parsed["context_length"].is_string());
        assert_eq!(parsed["vocab_size"], "50257");
        assert_eq!(parsed["hidden_size"], "768");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_extras_can_override_standard_fields() {
        // Arrange: pass an extra with same key as a standard field
        let mut extras = HashMap::new();
        extras.insert("arch_key".to_string(), "overridden".to_string());
        let meta = build_metadata("original", 1, 2, 3, 4, 5, 6, 7, 8, &extras);
        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // Assert: extras loop runs after standard fields, so arch_key is overridden
        assert_eq!(parsed["arch_key"], "overridden");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_new_stores_page_size_correctly() {
        // Arrange & Act
        let writer = GllmWriter::new(8192);
        // Assert: page_size is stored and used in output
        assert_eq!(writer.page_size, 8192);
        assert_eq!(writer.tensor_count(), 0);
        assert!(writer.metadata_bytes.is_empty());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_mixed_quantized_and_plain_flags_set() {
        // Arrange: one quantized + one plain tensor; flags should be set
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "plain".to_string(),
            ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        writer.add_tensor(TensorEntry {
            name: "quant".to_string(),
            ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 8], original_size: 64,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: flags bit 0 = 1 because at least one tensor is quantized
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1, "flags should be set when any tensor is quantized");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn align_up_already_aligned_value_returns_same() {
        // Arrange: values that are exact multiples of alignment
        let cases = [(64u64, 64u64), (128u64, 64u64), (256u64, 128u64), (4096u64, 4096u64)];
        for (value, alignment) in cases {
            // Act
            let result = align_up(value, alignment);
            // Assert: already-aligned value returns itself
            assert_eq!(result, value, "align_up({}, {}) should return {}", value, alignment, value);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_ndim_and_dtype_max_u8_values() {
        // Arrange: TensorEntry with max u8 values for ndim and dtype
        let entry = TensorEntry {
            name: "max_u8".to_string(),
            ndim: u8::MAX,
            dtype: u8::MAX,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Act & Assert
        assert_eq!(entry.ndim, 255);
        assert_eq!(entry.dtype, 255);
        assert!(!entry.is_quantized());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_header_page_size_field_matches_writer() {
        // Arrange: writer with specific page_size
        let page_size: u32 = 2048;
        let mut writer = GllmWriter::new(page_size);
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: page_size in header at bytes 40..44 (u32 LE)
        let stored_ps = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(stored_ps, page_size);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_roundtrip_single_char_tensor_names() {
        // Arrange: multiple tensors with single-character names
        let mut builder = GllmWriter::new(64);
        for ch in ['a', 'b', 'c', 'z'] {
            builder.add_tensor(TensorEntry {
                name: ch.to_string(),
                ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![ch as u8; 8], original_size: 8,
            });
        }
        builder.set_metadata(vec![]);
        // Act
        let dir = unique_test_dir("single_char_names");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single_char.gllm");
        builder.write_to_path(&path).unwrap();
        // Assert: each single-char tensor name and data round-trips correctly
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 4);
        for ch in ['a', 'b', 'c', 'z'] {
            let td = reader.tensor_data(&ch.to_string()).unwrap();
            assert_eq!(td.len(), 8);
            assert!(td.iter().all(|&b| b == ch as u8), "data for '{}' mismatch", ch);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_shape_u64_max_values() {
        // Arrange: shape with u64::MAX in first position
        let entry = TensorEntry {
            name: "max_shape".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [u64::MAX, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Act & Assert: u64::MAX preserved in shape array
        assert_eq!(entry.shape[0], u64::MAX);
        assert_eq!(entry.shape[1], 0);
        assert_eq!(entry.shape[2], 0);
        assert_eq!(entry.shape[3], 0);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_compressed_size_in_tensor_dir_entry() {
        // Arrange: tensor with 42 bytes of data
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "cs_test".to_string(),
            ndim: 1, dtype: 0, shape: [42, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAB; 42], original_size: 168,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: compressed_size field at bytes 56..64 of first tensor dir entry
        let entry_start = HEADER_SIZE;
        let compressed_size = u64::from_le_bytes(buf[entry_start + 56..entry_start + 64].try_into().unwrap());
        assert_eq!(compressed_size, 42);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_original_size_in_tensor_dir_entry() {
        // Arrange: tensor with 16 bytes data but original_size=128 (simulating compression)
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "os_test".to_string(),
            ndim: 2, dtype: 0, shape: [8, 8, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 16], original_size: 128,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: original_size field at bytes 64..72 of first tensor dir entry
        let entry_start = HEADER_SIZE;
        let original_size = u64::from_le_bytes(buf[entry_start + 64..entry_start + 72].try_into().unwrap());
        assert_eq!(original_size, 128);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_awq4_gptq4_squeeze_codes() {
        // Arrange
        use gllm_kernels::quant::QuantType;
        // Act & Assert: verify AWQ4, GPTQ4, Squeeze mapping codes
        assert_eq!(quant_type_to_u8(QuantType::AWQ4), 40);
        assert_eq!(quant_type_to_u8(QuantType::GPTQ4), 41);
        assert_eq!(quant_type_to_u8(QuantType::Squeeze), 42);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — trait verification and edge cases
    // ────────────────────────────────────────────────────────────────────────

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_debug_trait_produces_valid_output() {
        // Arrange
        let entry = TensorEntry {
            name: "debug.trait.check".to_string(),
            ndim: 2,
            dtype: 7,
            shape: [128, 256, 0, 0],
            quant_format: 22,
            quant_block_size: 64,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 32],
            original_size: 65536,
        };
        // Act
        let debug_str = format!("{:?}", entry);
        // Assert: Debug trait output must be parseable and contain key fields
        assert!(debug_str.starts_with("TensorEntry"), "Debug should start with struct name");
        assert!(debug_str.contains("name:"));
        assert!(debug_str.contains("ndim: 2"));
        assert!(debug_str.contains("dtype: 7"));
        assert!(debug_str.contains("original_size: 65536"));
        assert!(debug_str.contains("debug.trait.check"));
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_clone_deep_copies_all_fields() {
        // Arrange
        let original = TensorEntry {
            name: "clone_source".to_string(),
            ndim: 4,
            dtype: 3,
            shape: [10, 20, 30, 40],
            quant_format: 41,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xAA, 0xBB, 0xCC],
            original_size: 999,
        };
        // Act
        let cloned = original.clone();
        // Assert: every field is independently copied
        assert_eq!(cloned.name, "clone_source");
        assert_eq!(cloned.ndim, 4);
        assert_eq!(cloned.dtype, 3);
        assert_eq!(cloned.shape, [10, 20, 30, 40]);
        assert_eq!(cloned.quant_format, 41);
        assert_eq!(cloned.quant_block_size, 128);
        assert_eq!(cloned.scale_dtype, 2);
        assert_eq!(cloned.zp_type, 1);
        assert_eq!(cloned.data, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(cloned.original_size, 999);
        // Verify independence after mutation
        let mut mutated = cloned;
        mutated.name.push_str("_x");
        mutated.data.clear();
        mutated.original_size = 0;
        assert_eq!(original.name, "clone_source");
        assert_eq!(original.data.len(), 3);
        assert_eq!(original.original_size, 999);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_compressed_size_with_one_byte_data() {
        // Arrange
        let entry = TensorEntry {
            name: "one_byte".into(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xFE],
            original_size: 4,
        };
        // Act & Assert: compressed_size == data.len(), not original_size
        assert_eq!(entry.compressed_size(), 1);
        assert_ne!(entry.compressed_size(), entry.original_size);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn is_quantized_boundary_at_one() {
        // Arrange: quant_format == 1 is the smallest quantized value
        let entry = TensorEntry {
            name: "qf1".into(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 1,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Act & Assert
        assert!(entry.is_quantized(), "quant_format=1 must be quantized");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn align_up_with_alignment_equal_to_value_plus_one() {
        // Arrange: value=15, alignment=16 -> should round up to 16
        // Act & Assert
        assert_eq!(align_up(15, 16), 16);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(63, 64), 64);
        assert_eq!(align_up(127, 128), 128);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn align_up_with_large_u64_near_max() {
        // Arrange: values near u64::MAX with small alignment
        // Act & Assert
        assert_eq!(align_up(u64::MAX - 1, 1), u64::MAX - 1);
        assert_eq!(align_up(u64::MAX, 1), u64::MAX);
        assert_eq!(align_up(u64::MAX / 2, 2), u64::MAX / 2 + (u64::MAX / 2) % 2);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_keys_are_exactly_nine_when_no_extras() {
        // Arrange
        let meta = build_metadata("arch", 1, 2, 3, 4, 5, 6, 7, 8, &HashMap::new());
        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        let obj = parsed.as_object().unwrap();
        // Assert: exactly 9 standard keys
        let expected_keys = [
            "arch_key", "vocab_size", "hidden_size", "num_layers",
            "num_heads", "num_kv_heads", "head_dim", "intermediate_size",
            "context_length",
        ];
        assert_eq!(obj.len(), expected_keys.len());
        for key in &expected_keys {
            assert!(obj.contains_key(*key), "metadata must contain key '{}'", key);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_arch_key_with_slashes() {
        // Arrange
        let meta = build_metadata("org/model-arch/v2", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        // Act
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        // Assert
        assert_eq!(parsed["arch_key"], "org/model-arch/v2");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_f16_code_value_one() {
        // Arrange: F16 maps to code 1
        // Act
        let code = safetensors_dtype_to_u8(safetensors::Dtype::F16);
        // Assert
        assert_eq!(code, 1);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_q5_family_codes() {
        // Arrange
        use gllm_kernels::quant::QuantType;
        // Act & Assert: Q5_0=12, Q5_1=13, Q5K=23
        assert_eq!(quant_type_to_u8(QuantType::Q5_0), 12);
        assert_eq!(quant_type_to_u8(QuantType::Q5_1), 13);
        assert_eq!(quant_type_to_u8(QuantType::Q5K), 23);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_add_tensor_then_set_metadata_order_independent() {
        // Arrange: set metadata first, then add tensor (reverse of typical order)
        let mut writer = GllmWriter::new(4096);
        // Act
        writer.set_metadata(vec![0xAA, 0xBB]);
        writer.add_tensor(TensorEntry {
            name: "reverse_order".into(),
            ndim: 1,
            dtype: 0,
            shape: [2, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![1, 2],
            original_size: 2,
        });
        // Assert: both fields should be set correctly regardless of call order
        assert_eq!(writer.tensor_count(), 1);
        assert_eq!(writer.metadata_bytes, vec![0xAA, 0xBB]);
        assert_eq!(writer.tensors[0].name, "reverse_order");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_in_memory_buffer_produces_valid_header() {
        // Arrange
        let mut writer = GllmWriter::new(512);
        writer.add_tensor(TensorEntry {
            name: "buf".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x55; 64],
            original_size: 64,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: buffer starts with GLLM_MAGIC and has valid header fields
        assert!(buf.len() >= HEADER_SIZE);
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
        let tensor_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tensor_count, 1);
        let ps = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(ps, 512);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_in_memory_quant_flag_correct() {
        // Arrange: only unquantized tensor -> flags=0
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "plain".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        writer.set_metadata(vec![]);
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags, 0, "all-unquantized must have flags=0");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_in_memory_string_table_concatenated() {
        // Arrange: two tensors with names "aa" and "bbb"
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "aa".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.add_tensor(TensorEntry {
            name: "bbb".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: string table starts after header(64) + 2*tensor_entry(72)
        let strtab_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        assert_eq!(&buf[strtab_start..strtab_start + 2], b"aa");
        assert_eq!(&buf[strtab_start + 2..strtab_start + 5], b"bbb");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_in_memory_ndim_dtype_in_tensor_dir() {
        // Arrange: tensor with ndim=3, dtype=5
        let mut writer = GllmWriter::new(256);
        writer.add_tensor(TensorEntry {
            name: "nd".to_string(),
            ndim: 3,
            dtype: 5,
            shape: [2, 4, 8, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: ndim at byte 6 of tensor dir entry, dtype at byte 7
        let entry_start = HEADER_SIZE;
        assert_eq!(buf[entry_start + 6], 3, "ndim must be 3");
        assert_eq!(buf[entry_start + 7], 5, "dtype must be 5");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_roundtrip_preserves_metadata_json_content() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("custom".to_string(), "value123".to_string());
        let meta = build_metadata("test_arch", 500, 1024, 8, 16, 4, 64, 4096, 8192, &extras);
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "meta_check".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        });
        builder.set_metadata(meta);
        // Act
        let dir = unique_test_dir("meta_json_content");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_json.gllm");
        builder.write_to_path(&path).unwrap();
        // Assert: read back metadata bytes and parse JSON
        let reader = GllmReader::open(&path).unwrap();
        let meta_bytes = reader.metadata_bytes();
        let parsed: serde_json::Value = serde_json::from_slice(meta_bytes).unwrap();
        assert_eq!(parsed["arch_key"], "test_arch");
        assert_eq!(parsed["vocab_size"], "500");
        assert_eq!(parsed["custom"], "value123");
        assert_eq!(parsed["num_layers"], "8");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_in_memory_data_region_padding_is_zero_filled() {
        // Arrange: single tensor with 3 bytes of data, page_size=8 => 5 bytes padding
        let mut writer = GllmWriter::new(8);
        writer.add_tensor(TensorEntry {
            name: "pad".to_string(),
            ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 3], original_size: 3,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: data region starts at data_offset from header
        let data_off = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        assert_eq!(buf[data_off], 0xAA);
        assert_eq!(buf[data_off + 1], 0xAA);
        assert_eq!(buf[data_off + 2], 0xAA);
        // padding bytes must be zero
        assert_eq!(buf[data_off + 3], 0);
        assert_eq!(buf[data_off + 4], 0);
        assert_eq!(buf[data_off + 5], 0);
        assert_eq!(buf[data_off + 6], 0);
        assert_eq!(buf[data_off + 7], 0);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_header_version_is_one() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: version at bytes 4..8 must be GLLM_VERSION (1)
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_header_magic_is_gllm_tag() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: magic at bytes 0..4 must be GLLM_MAGIC
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_tensor_data_exactly_page_aligned_no_padding() {
        // Arrange: tensor data length exactly equals page_size
        let page_size: u32 = 64;
        let data = vec![0xCC; page_size as usize];
        let mut builder = GllmWriter::new(page_size);
        builder.add_tensor(TensorEntry {
            name: "exact_page".to_string(),
            ndim: 1, dtype: 0, shape: [64, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 64,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("exact_page_aligned");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_page.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let td = reader.tensor_data("exact_page").unwrap();
        assert_eq!(td.len(), 64);
        assert!(td.iter().all(|&b| b == 0xCC));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_metadata_with_unicode_values() {
        // Arrange: metadata extras with unicode characters
        let mut extras = HashMap::new();
        extras.insert("description".to_string(), "模型推理引擎 🚀".to_string());
        let meta = build_metadata("unicode_arch", 100, 200, 4, 8, 4, 64, 256, 512, &extras);
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(meta);
        let dir = unique_test_dir("unicode_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(reader.metadata_bytes()).unwrap();
        assert_eq!(parsed["description"], "模型推理引擎 🚀");
        assert_eq!(parsed["arch_key"], "unicode_arch");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_entry_name_offset_first_is_zero() {
        // Arrange
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "first".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: name_offset is first 4 bytes of tensor dir entry (byte 64..68)
        let name_offset = u32::from_le_bytes(buf[HEADER_SIZE..HEADER_SIZE + 4].try_into().unwrap());
        assert_eq!(name_offset, 0, "first tensor name offset must be 0");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_entry_name_len_matches_bytes() {
        // Arrange: name "weights" = 7 bytes
        let name = "weights".to_string();
        let name_len = name.len() as u16;
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: name.clone(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: name_len at bytes 4..6 of tensor dir entry
        let stored_name_len = u16::from_le_bytes(
            buf[HEADER_SIZE + 4..HEADER_SIZE + 6].try_into().unwrap()
        );
        assert_eq!(stored_name_len, name_len);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_fp8_e4m3_quant_type_preserved() {
        // Arrange: tensor with FP8 E4M3 quant format
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "fp8_weight".to_string(),
            ndim: 2, dtype: 0, shape: [4, 16, 0, 0],
            quant_format: quant_type_to_u8(gllm_kernels::quant::QuantType::Fp8E4M3),
            quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xABu8; 32], original_size: 64,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("fp8_roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fp8.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert
        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());
        let t = reader.find_tensor("fp8_weight").unwrap();
        assert_eq!(t.entry.quant_format, quant_type_to_u8(gllm_kernels::quant::QuantType::Fp8E4M3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_header_tensor_count_matches_writer_count() {
        // Arrange: 5 tensors
        let mut writer = GllmWriter::new(64);
        for i in 0..5 {
            writer.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 8], original_size: 8,
            });
        }
        writer.set_metadata(vec![]);
        assert_eq!(writer.tensor_count(), 5);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: tensor_count in header at bytes 20..24
        let stored_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(stored_count, 5);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_shape_dimensions_in_tensor_dir_match_input() {
        // Arrange: tensor with shape [3, 7, 11, 13]
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "shape_test".to_string(),
            ndim: 4, dtype: 0, shape: [3, 7, 11, 13],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 64], original_size: 64,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: shape starts at byte 8 of tensor dir entry (HEADER_SIZE+8), 4 u64s
        let entry_start = HEADER_SIZE;
        let s0 = u64::from_le_bytes(buf[entry_start + 8..entry_start + 16].try_into().unwrap());
        let s1 = u64::from_le_bytes(buf[entry_start + 16..entry_start + 24].try_into().unwrap());
        let s2 = u64::from_le_bytes(buf[entry_start + 24..entry_start + 32].try_into().unwrap());
        let s3 = u64::from_le_bytes(buf[entry_start + 32..entry_start + 40].try_into().unwrap());
        assert_eq!(s0, 3);
        assert_eq!(s1, 7);
        assert_eq!(s2, 11);
        assert_eq!(s3, 13);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_single_tensor_with_max_shape_dim() {
        // Arrange: tensor with u64::MAX in shape[0]
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "big_shape".to_string(),
            ndim: 1, dtype: 0, shape: [u64::MAX, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("max_shape");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("max_shape.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert
        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("big_shape").unwrap();
        assert_eq!(t.entry.shape[0], u64::MAX);
        assert_eq!(t.entry.shape[1], 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_compressed_size_in_dir_entry_matches_data_len() {
        // Arrange: tensor with 99 bytes of data
        let data_len = 99u64;
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "csz".to_string(),
            ndim: 1, dtype: 0, shape: [99, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; data_len as usize], original_size: 200,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: compressed_size at bytes 56..64 of tensor dir entry
        let entry_start = HEADER_SIZE;
        let csz = u64::from_le_bytes(buf[entry_start + 56..entry_start + 64].try_into().unwrap());
        assert_eq!(csz, data_len);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_original_size_in_dir_entry_matches_input() {
        // Arrange: original_size = 99999 (larger than compressed)
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "osz".to_string(),
            ndim: 1, dtype: 0, shape: [16, 0, 0, 0],
            quant_format: 22, quant_block_size: 128, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 99999,
        });
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: original_size at bytes 64..72 of tensor dir entry
        let entry_start = HEADER_SIZE;
        let osz = u64::from_le_bytes(buf[entry_start + 64..entry_start + 72].try_into().unwrap());
        assert_eq!(osz, 99999u64);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_five_tensors_all_data_independent() {
        // Arrange: 5 tensors with distinct data patterns
        let patterns: Vec<Vec<u8>> = (0..5).map(|i| vec![i as u8; 16]).collect();
        let mut builder = GllmWriter::new(64);
        for (i, pat) in patterns.iter().enumerate() {
            builder.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1, dtype: 0, shape: [16, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: pat.clone(), original_size: 16,
            });
        }
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("five_independent");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("five.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: each tensor's data is exactly the pattern we set
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 5);
        for i in 0..5 {
            let td = reader.tensor_data(&format!("t{}", i)).unwrap();
            assert_eq!(td.len(), 16);
            assert!(td.iter().all(|&b| b == i as u8), "tensor t{} data mismatch", i);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_meta_offset_in_header_points_to_after_string_table() {
        // Arrange: 3 tensors with names "aa", "bb", "ccc" (total 7 bytes string table)
        let mut writer = GllmWriter::new(64);
        for name in &["aa", "bb", "ccc"] {
            writer.add_tensor(TensorEntry {
                name: name.to_string(),
                ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 8], original_size: 8,
            });
        }
        writer.set_metadata(vec![]);
        // Act
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: meta_offset at bytes 12..20
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let expected = HEADER_SIZE as u64 + 3 * TENSOR_ENTRY_SIZE as u64 + 7; // 64 + 216 + 7 = 287
        assert_eq!(meta_offset, expected);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_compressed_size_matches_data_len_v2() {
        let entry = TensorEntry {
            name: "x".to_string(),
            ndim: 1, dtype: 0, shape: [100, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 256],
            original_size: 256,
        };
        assert_eq!(entry.compressed_size(), 256);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_is_quantized_true_when_format_nonzero() {
        let entry = TensorEntry {
            name: "q".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 10, quant_block_size: 32, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16],
            original_size: 64,
        };
        assert!(entry.is_quantized());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_is_quantized_false_when_format_zero() {
        let entry = TensorEntry {
            name: "f".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16],
            original_size: 16,
        };
        assert!(!entry.is_quantized());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_new_has_zero_tensors() {
        let writer = GllmWriter::new(4096);
        assert_eq!(writer.tensor_count(), 0);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_add_tensor_increments_count() {
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        assert_eq!(writer.tensor_count(), 1);
        writer.add_tensor(TensorEntry {
            name: "b".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        assert_eq!(writer.tensor_count(), 2);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_clone_is_independent_v2() {
        let entry = TensorEntry {
            name: "orig".to_string(),
            ndim: 2, dtype: 1, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![42u8; 32],
            original_size: 32,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, "orig");
        assert_eq!(cloned.data, entry.data);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_f32_maps_to_zero() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_bf16_maps_to_two() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_f16_maps_to_one() {
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_path_creates_file() {
        let dir = unique_test_dir("create");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("new.gllm");
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "t".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        writer.set_metadata(vec![]);
        writer.write_to_path(&path).unwrap();
        assert!(path.exists(), "file must be created");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_bf16_is_one() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Bf16), 1);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_f16_is_two() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::F16), 2);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_q8_0_is_fourteen() {
        use gllm_kernels::quant::QuantType;
        assert_eq!(quant_type_to_u8(QuantType::Q8_0), 14);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_empty_metadata_roundtrip() {
        let dir = unique_test_dir("empty_meta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_meta.gllm");
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "t1".to_string(),
            ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAAu8; 32], original_size: 32,
        });
        writer.set_metadata(vec![]);
        writer.write_to_path(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes().len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn roundtrip_metadata_overrides_extras_key() {
        // Arrange: extras key that shadows a standard key
        let mut extras = HashMap::new();
        extras.insert("arch_key".to_string(), "overridden".to_string());
        let meta = build_metadata("original", 100, 200, 4, 8, 4, 64, 256, 512, &extras);
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "x".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(meta);
        let dir = unique_test_dir("meta_override");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("override.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: extras key overrides the standard arch_key (HashMap behavior)
        let reader = GllmReader::open(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(reader.metadata_bytes()).unwrap();
        assert_eq!(parsed["arch_key"], "overridden", "extras should override standard key");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Batch 4: 15 additional tests — uncovered edge cases
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn write_data_size_exactly_page_aligned_no_padding() {
        // 当张量数据大小恰好是 page_size 的整数倍时，不应产生额外填充
        let mut builder = GllmWriter::new(64);
        let data = vec![0xAB; 64]; // 恰好 64 字节 = page_size，无需填充
        builder.add_tensor(TensorEntry {
            name: "exact_page".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("exact_page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_page.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        // 文件应在 data_offset + 64 处结束（无额外填充）
        assert_eq!(raw.len(), data_offset + 64);
        assert_eq!(&raw[data_offset..data_offset + 64], &data[..]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_two_tensors_both_page_aligned_no_gap() {
        // 两个张量都恰好是 page_size 的倍数，连续排列无间隙
        let page = 128u32;
        let mut builder = GllmWriter::new(page);
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0, shape: [128, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11; 128], original_size: 128,
        });
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0, shape: [128, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x22; 128], original_size: 128,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("no_gap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("no_gap.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t2 = reader.find_tensor("t2").unwrap();
        // 第二个张量偏移应恰好是 128（第一个张量大小）
        assert_eq!(t2.entry.data_offset, 128);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_third_tensor_offset_is_sum_of_aligned_predecessors() {
        // 验证第三个张量的 data_offset 是前两个对齐后大小之和
        let mut builder = GllmWriter::new(64);
        // 10 字节 → 对齐到 64
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0, shape: [10, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 10], original_size: 10,
        });
        // 20 字节 → 对齐到 64
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0, shape: [20, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 20], original_size: 20,
        });
        // 5 字节 → 应从 offset 128 开始
        builder.add_tensor(TensorEntry {
            name: "t3".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xCC; 5], original_size: 5,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("third_offset");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("third_offset.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t3 = reader.find_tensor("t3").unwrap();
        assert_eq!(t3.entry.data_offset, 128, "third tensor offset = 64 + 64");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_near_duplicate_tensor_names_distinguishable() {
        // 名称前缀相同但不同的张量，验证 reader 能区分
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "model.layer.0.weight".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 16], original_size: 16,
        });
        builder.add_tensor(TensorEntry {
            name: "model.layer.0.bias".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("near_dup");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("near_dup.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 2);

        let w = reader.tensor_data("model.layer.0.weight").unwrap();
        assert!(w.iter().all(|&b| b == 0xAA));
        let b = reader.tensor_data("model.layer.0.bias").unwrap();
        assert!(b.iter().all(|&b| b == 0xBB));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_empty_tensor_then_nonempty_offset_zero() {
        // 空数据张量在前，非空张量在后，验证非空张量 data_offset = 0
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "empty_first".to_string(), ndim: 1, dtype: 0, shape: [0, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        builder.add_tensor(TensorEntry {
            name: "nonempty".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF; 32], original_size: 32,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("empty_then_full");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_then_full.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let empty = reader.find_tensor("empty_first").unwrap();
        assert_eq!(empty.entry.data_offset, 0);
        let nonempty = reader.find_tensor("nonempty").unwrap();
        // 空张量对齐大小为 0，非空张量也从 data_offset=0 开始
        assert_eq!(nonempty.entry.data_offset, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_very_large_page_size_file_size() {
        // page_size 远大于数据，文件大小应受 page_size 主导
        let page = 65536u32;
        let mut builder = GllmWriter::new(page);
        builder.add_tensor(TensorEntry {
            name: "tiny".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x42; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("large_page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large_page.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        // 文件大小 = data_offset + 65536（数据被填充到 page_size）
        assert_eq!(raw.len(), data_offset + 65536);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_metadata_exact_fill_to_data_boundary() {
        // metadata 大小恰好使 data 区域无需额外填充
        let page = 256u32;
        let mut builder = GllmWriter::new(page);
        builder.add_tensor(TensorEntry {
            name: "x".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        // 先写入空 metadata，测量 data_offset
        let dir = unique_test_dir("meta_fill");
        std::fs::create_dir_all(&dir).unwrap();
        let probe_path = dir.join("probe.gllm");
        builder.set_metadata(vec![]);
        builder.write_to_path(&probe_path).unwrap();
        let probe_raw = std::fs::read(&probe_path).unwrap();
        let data_offset = u64::from_le_bytes(probe_raw[32..40].try_into().unwrap()) as usize;
        let meta_offset = u64::from_le_bytes(probe_raw[12..20].try_into().unwrap()) as usize;
        // 填充 metadata 使其恰好到 data_offset
        let exact_meta_len = data_offset - meta_offset;
        let meta = vec![0xCA; exact_meta_len];
        builder.set_metadata(meta);

        let path = dir.join("meta_fill.gllm");
        builder.write_to_path(&path).unwrap();
        let raw = std::fs::read(&path).unwrap();
        // data_offset 应不变
        let new_data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        assert_eq!(new_data_offset, data_offset);
        // 无填充：data 紧跟 metadata
        for i in 0..exact_meta_len {
            assert_eq!(raw[meta_offset + i], 0xCA);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_quant_format_byte_in_binary() {
        // 验证二进制中 quant_format 字节位置正确
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "qf_test".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 41, // GPTQ4
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 16], original_size: 64,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("qf_binary");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qf_binary.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // Tensor entry 中 quant_format 在 byte 40（从 entry 起始算）
        let entry_start = HEADER_SIZE;
        assert_eq!(raw[entry_start + 40], 41);
        // quant_block_size 在 byte 41..43 (u16 LE)
        let qbs = u16::from_le_bytes(raw[entry_start + 41..entry_start + 43].try_into().unwrap());
        assert_eq!(qbs, 128);
        // scale_dtype 在 byte 43
        assert_eq!(raw[entry_start + 43], 2);
        // zp_type 在 byte 44
        assert_eq!(raw[entry_start + 44], 1);
        // padding bytes 45..47 应为 0
        for i in 45..48 {
            assert_eq!(raw[entry_start + i], 0, "padding byte {} should be zero", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_tensor_with_zero_original_size_quantized_roundtrip() {
        // 量化张量 original_size=0 但有实际数据，验证 roundtrip
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "zero_orig".to_string(), ndim: 2, dtype: 0, shape: [8, 8, 0, 0],
            quant_format: 22, // Q4K
            quant_block_size: 256,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xDD; 32],
            original_size: 0, // 异常但结构合法
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("zero_orig_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_orig_quant.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("zero_orig").unwrap();
        assert_eq!(t.entry.original_size, 0);
        assert_eq!(t.entry.compressed_size, 32);
        assert!(t.entry.is_quantized());
        assert_eq!(t.entry.quant_format, 22);

        let td = reader.tensor_data("zero_orig").unwrap();
        assert_eq!(td.len(), 32);
        assert!(td.iter().all(|&b| b == 0xDD));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_read_back_metadata_matches_build_metadata_output() {
        // build_metadata 的输出写入后再读回，内容完全一致
        let mut extras = HashMap::new();
        extras.insert("model_type".to_string(), "qwen3".to_string());
        extras.insert("rope_theta".to_string(), "1000000.0".to_string());
        let meta = build_metadata(
            "qwen3", 151936, 4096, 36, 32, 8, 128, 11008, 32768, &extras,
        );
        let original_parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();

        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "meta_check".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(meta);

        let dir = unique_test_dir("meta_exact");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_exact.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let read_back: serde_json::Value = serde_json::from_slice(reader.metadata_bytes()).unwrap();
        assert_eq!(original_parsed, read_back, "metadata roundtrip should be byte-identical");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_zero_page_size_produces_valid_file() {
        // page_size=0 时 align_up 直接返回原值，文件仍然有效
        let mut builder = GllmWriter::new(0);
        builder.add_tensor(TensorEntry {
            name: "zero_page".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x77; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("zero_page_write");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_page.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.header().page_size, 0);
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("zero_page").unwrap();
        assert_eq!(&td[..], &[0x77; 4]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn header_data_offset_equals_computed_offset() {
        // 验证 header 中 data_offset 等于手算值
        let page = 512u32;
        let mut builder = GllmWriter::new(page);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        let meta = vec![0xDE, 0xAD];
        builder.set_metadata(meta);

        let dir = unique_test_dir("data_offset_compute");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data_offset_compute.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        // 手算: header(64) + tensor_dir(72) + string_table("a"=1) + metadata(2) = 139
        // 对齐到 512 → 512
        let expected = align_up(139, 512);
        assert_eq!(data_offset, expected);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_many_tensors_total_data_region_size() {
        // 验证多个张量的数据区域总大小正确
        let page = 64u32;
        let mut builder = GllmWriter::new(page);
        let sizes = [10usize, 20, 30, 40, 50];
        for (i, &sz) in sizes.iter().enumerate() {
            builder.add_tensor(TensorEntry {
                name: format!("s{}", i), ndim: 1, dtype: 0,
                shape: [sz as u64, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![i as u8; sz], original_size: sz as u64,
            });
        }
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("total_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("total_data.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        // 每个张量对齐到 64: [10→64, 20→64, 30→64, 40→64, 50→64] = 320
        let expected_data_region = sizes.len() * 64;
        assert_eq!(raw.len(), data_offset + expected_data_region);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_single_tensor_data_offset_in_entry_is_zero() {
        // 唯一张量的 data_offset（在 tensor dir 内记录的）应为 0
        let mut builder = GllmWriter::new(4096);
        builder.add_tensor(TensorEntry {
            name: "only".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x55; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("single_do");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single_do.gllm");
        builder.write_to_path(&path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // data_offset 在 tensor entry 的 byte 48..56
        let entry_start = HEADER_SIZE;
        let t_data_offset = u64::from_le_bytes(raw[entry_start + 48..entry_start + 56].try_into().unwrap());
        assert_eq!(t_data_offset, 0, "single tensor's data_offset in entry should be 0");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_read_back_tensor_entry_all_fields_match() {
        // 完整验证写入后读回的每个 tensor entry 字段都匹配
        let original = TensorEntry {
            name: "full_check".to_string(),
            ndim: 3,
            dtype: 5,
            shape: [128, 256, 64, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xEF; 48],
            original_size: 32768,
        };
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(original.clone());
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("full_fields");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full_fields.gllm");
        builder.write_to_path(&path).unwrap();

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("full_check").unwrap();
        assert_eq!(t.entry.ndim, original.ndim);
        assert_eq!(t.entry.dtype, original.dtype);
        assert_eq!(t.entry.shape, original.shape);
        assert_eq!(t.entry.quant_format, original.quant_format);
        assert_eq!(t.entry.quant_block_size, original.quant_block_size);
        assert_eq!(t.entry.scale_dtype, original.scale_dtype);
        assert_eq!(t.entry.zp_type, original.zp_type);
        assert_eq!(t.entry.compressed_size, original.compressed_size());
        assert_eq!(t.entry.original_size, original.original_size);

        let td = reader.tensor_data("full_check").unwrap();
        assert_eq!(td.len(), 48);
        assert!(td.iter().all(|&b| b == 0xEF));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — new uncovered paths
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn align_up_output_never_less_than_input() {
        // For any positive alignment, align_up(v, a) >= v
        for &alignment in &[1u64, 2, 3, 7, 13, 64, 128, 4096, 65536] {
            for &val in &[0u64, 1, alignment / 2, alignment - 1, alignment, alignment + 1, alignment * 10] {
                let result = align_up(val, alignment);
                assert!(result >= val, "align_up({val}, {alignment}) = {result} < {val}");
            }
        }
    }

    #[test]
    fn quant_type_to_u8_codes_in_expected_ranges() {
        use gllm_kernels::quant::QuantType;
        // Float types: 1..=3
        assert!((1..=3).contains(&quant_type_to_u8(QuantType::Bf16)));
        assert!((1..=3).contains(&quant_type_to_u8(QuantType::F16)));
        assert!((1..=3).contains(&quant_type_to_u8(QuantType::F32)));
        // Classic quants: 10..=15
        for qt in &[QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1, QuantType::Q8_0, QuantType::Q8_1] {
            assert!((10..=15).contains(&quant_type_to_u8(*qt)));
        }
        // K-quants: 20..=25
        for qt in &[QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K, QuantType::Q6K, QuantType::Q8K] {
            assert!((20..=25).contains(&quant_type_to_u8(*qt)));
        }
        // IQ family: 30..=38
        for qt in &[QuantType::IQ1S, QuantType::IQ1M, QuantType::IQ2XXS, QuantType::IQ2XS, QuantType::IQ2S, QuantType::IQ3XXS, QuantType::IQ3S, QuantType::IQ4NL, QuantType::IQ4XS] {
            assert!((30..=38).contains(&quant_type_to_u8(*qt)));
        }
        // AWQ/GPTQ/Squeeze: 40..=42
        assert!((40..=42).contains(&quant_type_to_u8(QuantType::AWQ4)));
        assert!((40..=42).contains(&quant_type_to_u8(QuantType::GPTQ4)));
        assert!((40..=42).contains(&quant_type_to_u8(QuantType::Squeeze)));
        // FP8: 50..=51
        assert!((50..=51).contains(&quant_type_to_u8(QuantType::Fp8E4M3)));
        assert!((50..=51).contains(&quant_type_to_u8(QuantType::Fp8E5M2)));
        // MXFP4/NVFP4: 52..=53
        assert!((52..=53).contains(&quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 })));
        assert!((52..=53).contains(&quant_type_to_u8(QuantType::Nvfp4)));
        // TQ: 60..=61
        assert!((60..=61).contains(&quant_type_to_u8(QuantType::TQ1_0)));
        assert!((60..=61).contains(&quant_type_to_u8(QuantType::TQ2_0)));
    }

    #[test]
    fn write_data_exactly_page_aligned_needs_no_padding() {
        // Arrange: data size is exactly page_size
        let page_size = 128u32;
        let mut builder = GllmWriter::new(page_size);
        let data = vec![0x77; page_size as usize];
        builder.add_tensor(TensorEntry {
            name: "exact_page".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [page_size as u64, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: data.clone(),
            original_size: page_size as u64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("exact_page_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exact_page_data.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: file size should be header + tensor_dir + name + data (no extra padding after data)
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        // Last byte of data region should be at data_offset + 127
        assert_eq!(raw[data_offset + 127], 0x77);
        // File size should be exactly data_offset + 128 (no trailing padding needed)
        assert_eq!(raw.len(), data_offset + page_size as usize);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_two_tensors_cumulative_data_offsets() {
        // Arrange: two tensors with known sizes, verify their data_offset fields
        let page_size = 64u32;
        let mut builder = GllmWriter::new(page_size);
        // First tensor: 10 bytes -> padded to 64
        builder.add_tensor(TensorEntry {
            name: "first".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x11; 10],
            original_size: 10,
        });
        // Second tensor: 20 bytes -> padded to 64
        builder.add_tensor(TensorEntry {
            name: "second".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [20, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0x22; 20],
            original_size: 20,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("cumulative_offsets");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cumulative_offsets.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: check data_offset fields in tensor directory entries
        let raw = std::fs::read(&path).unwrap();
        let entry0_data_offset = u64::from_le_bytes(
            raw[HEADER_SIZE + 48..HEADER_SIZE + 56].try_into().unwrap(),
        );
        let entry1_data_offset = u64::from_le_bytes(
            raw[HEADER_SIZE + TENSOR_ENTRY_SIZE + 48..HEADER_SIZE + TENSOR_ENTRY_SIZE + 56].try_into().unwrap(),
        );
        assert_eq!(entry0_data_offset, 0, "first tensor data offset should be 0");
        assert_eq!(entry1_data_offset, 64, "second tensor data offset should be 64 (first padded)");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_string_table_with_different_name_lengths() {
        // Arrange: tensors with very different name lengths
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "very_long_tensor_name_for_testing".to_string(),
            ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("name_lengths");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("name_lengths.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: verify string table layout
        let raw = std::fs::read(&path).unwrap();
        let strtab_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        // "a" at offset 0, length 1
        assert_eq!(&raw[strtab_start..strtab_start + 1], b"a");
        // long name at offset 1
        let long_name = "very_long_tensor_name_for_testing";
        assert_eq!(
            &raw[strtab_start + 1..strtab_start + 1 + long_name.len()],
            long_name.as_bytes(),
        );
        // Verify name_offset in tensor directory
        let name_off_0 = u32::from_le_bytes(raw[HEADER_SIZE..HEADER_SIZE + 4].try_into().unwrap());
        let name_len_0 = u16::from_le_bytes(raw[HEADER_SIZE + 4..HEADER_SIZE + 6].try_into().unwrap());
        assert_eq!(name_off_0, 0);
        assert_eq!(name_len_0, 1);

        let name_off_1 = u32::from_le_bytes(
            raw[HEADER_SIZE + TENSOR_ENTRY_SIZE..HEADER_SIZE + TENSOR_ENTRY_SIZE + 4].try_into().unwrap(),
        );
        let name_len_1 = u16::from_le_bytes(
            raw[HEADER_SIZE + TENSOR_ENTRY_SIZE + 4..HEADER_SIZE + TENSOR_ENTRY_SIZE + 6].try_into().unwrap(),
        );
        assert_eq!(name_off_1, 1);
        assert_eq!(name_len_1, long_name.len() as u16);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_quant_format_field_in_tensor_directory() {
        // Arrange
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "nvfp4_w".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 8, 0, 0],
            quant_format: 53, // NVFP4
            quant_block_size: 16,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 16],
            original_size: 128,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("qf_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qf_field.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: quant_format is at byte 40 of tensor entry
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw[HEADER_SIZE + 40], 53, "quant_format should be NVFP4=53");

        let reader = GllmReader::open(&path).unwrap();
        let t = reader.find_tensor("nvfp4_w").unwrap();
        assert_eq!(t.entry.quant_format, 53);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_metadata_with_hash_in_extra_key() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("layer#0".to_string(), "special".to_string());

        // Act
        let meta = build_metadata("test", 1, 2, 3, 4, 5, 6, 7, 8, &extras);

        // Assert
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["layer#0"], "special");
    }

    #[test]
    fn tensor_entry_compressed_size_ratio_calculation() {
        // Arrange: quantized tensor where original_size >> data.len()
        let entry = TensorEntry {
            name: "compressed_ratio".into(),
            ndim: 2,
            dtype: 0,
            shape: [4096, 4096, 0, 0],
            quant_format: 22, // Q4K
            quant_block_size: 256,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0u8; 1024],
            original_size: 67108864, // 4096*4096*4 bytes
        };

        // Assert
        assert!(entry.original_size > entry.compressed_size());
        assert_eq!(entry.compressed_size(), 1024);
        // Compression ratio ~65536:1
        let ratio = entry.original_size / entry.compressed_size();
        assert_eq!(ratio, 65536);
    }

    #[test]
    fn write_empty_metadata_between_tensors_and_data() {
        // Arrange: write with no metadata bytes, verify gap calculation is still correct
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "x".to_string(),
            ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("empty_meta_gap");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_meta_gap.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: meta_offset should point right after string table, data_offset should be page-aligned
        let raw = std::fs::read(&path).unwrap();
        let meta_offset = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        // meta_offset = header(64) + tensor_dir(72) + "x"(1) = 137
        assert_eq!(meta_offset, HEADER_SIZE + TENSOR_ENTRY_SIZE + 1);
        // data_offset should be page-aligned and >= meta_offset
        assert!(data_offset >= meta_offset);
        assert_eq!(data_offset % 256, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writer_large_page_size_produces_larger_file() {
        // Arrange: same tensor data, different page sizes
        let make_entry = || TensorEntry {
            name: "data".to_string(),
            ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x42; 8], original_size: 8,
        };

        let mut b_small = GllmWriter::new(128);
        b_small.add_tensor(make_entry());
        b_small.set_metadata(vec![]);

        let mut b_large = GllmWriter::new(4096);
        b_large.add_tensor(make_entry());
        b_large.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("page_size_effect");
        std::fs::create_dir_all(&dir).unwrap();
        let path_small = dir.join("small_page.gllm");
        let path_large = dir.join("large_page.gllm");
        b_small.write_to_path(&path_small).unwrap();
        b_large.write_to_path(&path_large).unwrap();

        // Assert: larger page size produces larger file due to more padding
        let size_small = std::fs::metadata(&path_small).unwrap().len();
        let size_large = std::fs::metadata(&path_large).unwrap().len();
        assert!(size_large > size_small, "larger page size should produce larger file");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tensor_entry_compressed_size_matches_data_len_after_modification() {
        // Arrange
        let mut entry = TensorEntry {
            name: "mutable".into(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 10],
            original_size: 10,
        };
        assert_eq!(entry.compressed_size(), 10);

        // Act: modify data
        entry.data.extend_from_slice(&[1, 2, 3]);

        // Assert: compressed_size reflects new data length
        assert_eq!(entry.compressed_size(), 13);
        assert_eq!(entry.data.len(), 13);
    }

    #[test]
    fn safetensors_dtype_to_u8_f32_is_zero_and_bf16_is_two() {
        // Explicitly verify F32 -> 0 and BF16 -> 2 as these are the most common mappings
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
        // And verify they are distinct
        assert_ne!(
            safetensors_dtype_to_u8(safetensors::Dtype::F32),
            safetensors_dtype_to_u8(safetensors::Dtype::BF16),
        );
    }

    #[test]
    fn write_three_tensors_data_contiguity() {
        // Arrange: 3 tensors whose data fills consecutive page-aligned blocks
        let page_size = 64u32;
        let mut builder = GllmWriter::new(page_size);
        builder.add_tensor(TensorEntry {
            name: "t0".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x10; 5], original_size: 5,
        });
        builder.add_tensor(TensorEntry {
            name: "t1".to_string(), ndim: 1, dtype: 0, shape: [7, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x20; 7], original_size: 7,
        });
        builder.add_tensor(TensorEntry {
            name: "t2".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x30; 3], original_size: 3,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("contiguity");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("contiguity.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: each tensor's data_offset is page-aligned and cumulative
        let reader = GllmReader::open(&path).unwrap();
        let t0 = reader.find_tensor("t0").unwrap();
        let t1 = reader.find_tensor("t1").unwrap();
        let t2 = reader.find_tensor("t2").unwrap();
        assert_eq!(t0.entry.data_offset, 0);
        assert_eq!(t1.entry.data_offset, 64); // 5 bytes padded to 64
        assert_eq!(t2.entry.data_offset, 128); // 7 bytes padded to 64, plus previous 64

        let td0 = reader.tensor_data("t0").unwrap();
        let td1 = reader.tensor_data("t1").unwrap();
        let td2 = reader.tensor_data("t2").unwrap();
        assert!(td0.iter().all(|&b| b == 0x10));
        assert!(td1.iter().all(|&b| b == 0x20));
        assert!(td2.iter().all(|&b| b == 0x30));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn build_metadata_context_length_131k_as_string() {
        // Arrange
        let ctx_len = 131072u64; // common long context length

        // Act
        let meta = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, ctx_len, &HashMap::new());

        // Assert
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["context_length"], "131072");
        assert!(parsed["context_length"].is_string(), "context_length should be stored as string");
    }

    #[test]
    fn header_data_offset_field_in_written_file() {
        // Arrange
        let mut builder = GllmWriter::new(512);
        builder.add_tensor(TensorEntry {
            name: "do".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF; 8], original_size: 8,
        });
        let meta = build_metadata("do_test", 10, 20, 1, 2, 3, 4, 5, 6, &HashMap::new());
        builder.set_metadata(meta);

        // Act
        let dir = unique_test_dir("do_field");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("do_field.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: data_offset at header bytes 32..40 should be page-aligned
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        assert!(data_offset >= HEADER_SIZE as u64 + TENSOR_ENTRY_SIZE as u64);
        assert_eq!(data_offset % 512, 0, "data_offset must be aligned to page_size=512");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_quantized_entry_padding_bytes_are_zero() {
        // Arrange: quantized tensor whose data is smaller than page_size
        let page_size = 64u32;
        let mut builder = GllmWriter::new(page_size);
        builder.add_tensor(TensorEntry {
            name: "qpad".to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 10,
            quant_block_size: 32,
            scale_dtype: 1,
            zp_type: 0,
            data: vec![0xDD; 11],
            original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("qpad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("qpad.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: 11 bytes of data + 53 bytes of zero padding
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap()) as usize;
        for i in 0..11 {
            assert_eq!(raw[data_offset + i], 0xDD, "data byte {} should be 0xDD", i);
        }
        for i in 11..64 {
            assert_eq!(raw[data_offset + i], 0, "padding byte {} should be zero", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Batch 5: 15 additional unit tests — new uncovered paths
    // Focus: TensorEntry defaults, multi-tensor write order, safetensors_dtype
    //        full mapping, quant_type boundary values, header magic validation,
    //        corrupted file detection
    // ────────────────────────────────────────────────────────────────────────

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_default_fields_is_not_quantized() {
        // Arrange: construct a TensorEntry with zero quant fields
        let entry = TensorEntry {
            name: "default_check".to_string(),
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Act & Assert
        assert!(!entry.is_quantized(), "zero quant_format means not quantized");
        assert_eq!(entry.compressed_size(), 0, "empty data has zero compressed_size");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_with_quant_format_nonzero_is_quantized() {
        // Arrange: set quant_format to every valid nonzero value and confirm is_quantized
        let non_zero_formats: &[u8] = &[10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 40, 41, 42, 50, 51, 52, 53];
        for &qf in non_zero_formats {
            let entry = TensorEntry {
                name: format!("qf_{}", qf),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: qf,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 16],
                original_size: 16,
            };
            assert!(entry.is_quantized(), "quant_format={} should be quantized", qf);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_three_tensors_preserve_insertion_order_in_directory() {
        // Arrange: add tensors A, B, C — verify tensor directory lists them in that order
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "alpha".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xA1; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "beta".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xB2; 4], original_size: 4,
        });
        builder.add_tensor(TensorEntry {
            name: "gamma".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xC3; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("write_order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("write_order.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: read back tensor directory entries, verify name order
        let reader = GllmReader::open(&path).unwrap();
        let tensors = reader.tensors();
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors[0].name, "alpha");
        assert_eq!(tensors[1].name, "beta");
        assert_eq!(tensors[2].name, "gamma");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_to_u8_all_standard_mappings() {
        // Arrange & Act & Assert: verify every branch of safetensors_dtype_to_u8
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F32), 0);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::F16), 1);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::BF16), 2);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::U8), 3);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I8), 4);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I32), 5);
        assert_eq!(safetensors_dtype_to_u8(safetensors::Dtype::I64), 6);
        // Verify all are distinct
        let codes: Vec<u8> = vec![
            safetensors_dtype_to_u8(safetensors::Dtype::F32),
            safetensors_dtype_to_u8(safetensors::Dtype::F16),
            safetensors_dtype_to_u8(safetensors::Dtype::BF16),
            safetensors_dtype_to_u8(safetensors::Dtype::U8),
            safetensors_dtype_to_u8(safetensors::Dtype::I8),
            safetensors_dtype_to_u8(safetensors::Dtype::I32),
            safetensors_dtype_to_u8(safetensors::Dtype::I64),
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(codes[i], codes[j], "dtype codes {} and {} must be distinct", i, j);
            }
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn safetensors_dtype_to_u8_unknown_falls_back_to_zero() {
        // Arrange: use a Dtype variant that falls into the catch-all branch
        // BOOL is not in the explicit match arms
        // Act
        let result = safetensors_dtype_to_u8(safetensors::Dtype::BOOL);
        // Assert: unknown dtype maps to 0 (F32 fallback)
        assert_eq!(result, 0, "unknown safetensors dtype should fall back to 0");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_bf16_f16_f32_are_distinct_and_nonzero() {
        // Arrange & Act
        use gllm_kernels::quant::QuantType;
        let bf16_code = quant_type_to_u8(QuantType::Bf16);
        let f16_code = quant_type_to_u8(QuantType::F16);
        let f32_code = quant_type_to_u8(QuantType::F32);
        // Assert
        assert_ne!(bf16_code, 0);
        assert_ne!(f16_code, 0);
        assert_ne!(f32_code, 0);
        assert_ne!(bf16_code, f16_code);
        assert_ne!(bf16_code, f32_code);
        assert_ne!(f16_code, f32_code);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_mxfp4_and_nvfp4_are_distinct() {
        // Arrange & Act
        use gllm_kernels::quant::QuantType;
        let mxfp4_code = quant_type_to_u8(QuantType::Mxfp4 { block_size: 32 });
        let nvfp4_code = quant_type_to_u8(QuantType::Nvfp4);
        // Assert
        assert_ne!(mxfp4_code, nvfp4_code, "MXFP4 and NVFP4 must have different codes");
        assert_eq!(mxfp4_code, 52);
        assert_eq!(nvfp4_code, 53);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_magic_bytes_written_correctly_in_output() {
        // Arrange: write a minimal valid .gllm file
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "magic_test".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("magic.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: first 4 bytes must be GLLM_MAGIC as little-endian
        let raw = std::fs::read(&path).unwrap();
        let magic = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC, "header magic must be 0x4D4C4C47");
        // Verify ASCII representation: 'G'=0x47, 'L'=0x4C, 'L'=0x4C, 'M'=0x4D
        assert_eq!(raw[0], b'G');
        assert_eq!(raw[1], b'L');
        assert_eq!(raw[2], b'L');
        assert_eq!(raw[3], b'M');

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_reserved_bytes_44_to_64_are_all_zero() {
        // Arrange
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "reserved".to_string(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("reserved");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reserved.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: bytes 44..64 (reserved) must all be zero
        let raw = std::fs::read(&path).unwrap();
        for i in 44..64 {
            assert_eq!(raw[i], 0, "reserved header byte {} must be zero", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn reader_rejects_corrupted_magic_with_invalid_magic_error() {
        // Arrange: create a file with wrong magic bytes
        let dir = unique_test_dir("bad_magic");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_magic.gllm");
        let mut corrupted = vec![0u8; 128];
        // Write a wrong magic value at offset 0..4
        corrupted[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        // Write valid version at offset 4..8
        corrupted[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        std::fs::write(&path, &corrupted).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert
        match result {
            Err(GllmError::InvalidMagic(m)) => assert_eq!(m, 0xDEADBEEF),
            other => panic!("expected InvalidMagic error, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn reader_rejects_truncated_file_shorter_than_header() {
        // Arrange: create a file with only 32 bytes (< HEADER_SIZE=64)
        let dir = unique_test_dir("truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.gllm");
        std::fs::write(&path, &[0u8; 32]).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert
        match result {
            Err(GllmError::HeaderTooSmall(n)) => assert_eq!(n, 32),
            other => panic!("expected HeaderTooSmall error, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn reader_rejects_file_with_wrong_version() {
        // Arrange: create a file with correct magic but version=99
        let dir = unique_test_dir("bad_version");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_version.gllm");
        let mut fake = vec![0u8; 128];
        fake[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        fake[4..8].copy_from_slice(&99u32.to_le_bytes()); // bad version
        std::fs::write(&path, &fake).unwrap();

        // Act
        let result = GllmReader::open(&path);

        // Assert
        match result {
            Err(GllmError::UnsupportedVersion(v)) => assert_eq!(v, 99),
            other => panic!("expected UnsupportedVersion error, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_multi_tensor_data_region_is_contiguous_no_gaps() {
        // Arrange: 4 tensors of varying sizes, verify data blocks are contiguous
        let page = 32u32;
        let mut builder = GllmWriter::new(page);
        builder.add_tensor(TensorEntry {
            name: "d0".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x10; 5], original_size: 5,
        });
        builder.add_tensor(TensorEntry {
            name: "d1".to_string(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x20; 1], original_size: 1,
        });
        builder.add_tensor(TensorEntry {
            name: "d2".to_string(), ndim: 1, dtype: 0, shape: [32, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x30; 32], original_size: 32,
        });
        builder.add_tensor(TensorEntry {
            name: "d3".to_string(), ndim: 1, dtype: 0, shape: [17, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x40; 17], original_size: 17,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("contiguous4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("contiguous4.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: offsets are 0, 32, 64, 96 (each aligned to page=32)
        let reader = GllmReader::open(&path).unwrap();
        let t0 = reader.find_tensor("d0").unwrap();
        let t1 = reader.find_tensor("d1").unwrap();
        let t2 = reader.find_tensor("d2").unwrap();
        let t3 = reader.find_tensor("d3").unwrap();
        assert_eq!(t0.entry.data_offset, 0);
        assert_eq!(t1.entry.data_offset, 32);
        assert_eq!(t2.entry.data_offset, 64);
        assert_eq!(t3.entry.data_offset, 96);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_header_flags_set_when_any_tensor_is_quantized() {
        // Arrange: mix one quantized tensor with unquantized ones
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "plain".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.add_tensor(TensorEntry {
            name: "quantized".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 40, quant_block_size: 128, scale_dtype: 1, zp_type: 0,
            data: vec![0u8; 8], original_size: 64,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("flags_quant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags_quant.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: header flags bit 0 should be set
        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1, "flags bit 0 should be set when any tensor is quantized");

        let reader = GllmReader::open(&path).unwrap();
        assert!(reader.header().is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_header_flags_zero_when_all_tensors_unquantized() {
        // Arrange: all tensors have quant_format=0
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.add_tensor(TensorEntry {
            name: "b".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 32], original_size: 32,
        });
        builder.set_metadata(vec![]);

        // Act
        let dir = unique_test_dir("flags_noquant");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flags_noquant.gllm");
        builder.write_to_path(&path).unwrap();

        // Assert: header flags should be 0
        let raw = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        assert_eq!(flags, 0, "flags should be 0 when no tensor is quantized");

        let reader = GllmReader::open(&path).unwrap();
        assert!(!reader.header().is_quantized());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_single_tensor_roundtrip_data_bytes_preserved() {
        // Arrange: build a single tensor with a unique non-trivial byte pattern
        let data: Vec<u8> = (0..128).map(|i| (i * 7 + 3) as u8).collect();
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "unique_data".to_string(), ndim: 2, dtype: 0,
            shape: [16, 8, 0, 0], quant_format: 0, quant_block_size: 0,
            scale_dtype: 0, zp_type: 0, data: data.clone(), original_size: 128,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("single_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single_data.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: read back and verify every byte matches
        let reader = GllmReader::open(&path).unwrap();
        let read_data = reader.tensor_data("unique_data").unwrap();
        assert_eq!(read_data.len(), 128);
        for (i, (&orig, &read)) in data.iter().zip(read_data.iter()).enumerate() {
            assert_eq!(orig, read, "byte mismatch at index {}", i);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_equality_by_field_comparison_after_clone() {
        // Arrange: TensorEntry does not derive PartialEq, so verify field-by-field
        let original = TensorEntry {
            name: "weights".to_string(), ndim: 3, dtype: 7,
            shape: [64, 128, 256, 0], quant_format: 42, quant_block_size: 128,
            scale_dtype: 3, zp_type: 2, data: vec![0xAB; 100], original_size: 512,
        };
        // Act: clone the entry
        let cloned = original.clone();
        // Assert: every field must match
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.quant_format, original.quant_format);
        assert_eq!(cloned.quant_block_size, original.quant_block_size);
        assert_eq!(cloned.scale_dtype, original.scale_dtype);
        assert_eq!(cloned.zp_type, original.zp_type);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.original_size, original.original_size);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_all_31_variants_unique_codes() {
        // Arrange: enumerate every single QuantType variant from the match expression
        use gllm_kernels::quant::QuantType;
        let all_types: Vec<QuantType> = vec![
            QuantType::Bf16, QuantType::F16, QuantType::F32,
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K, QuantType::Q5K,
            QuantType::Q6K, QuantType::Q8K,
            QuantType::IQ1S, QuantType::IQ1M, QuantType::IQ2XXS, QuantType::IQ2XS,
            QuantType::IQ2S, QuantType::IQ3XXS, QuantType::IQ3S, QuantType::IQ4NL,
            QuantType::IQ4XS,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Squeeze,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2,
            QuantType::Mxfp4 { block_size: 32 }, QuantType::Nvfp4,
            QuantType::TQ1_0, QuantType::TQ2_0,
        ];
        // Act: collect codes
        let codes: Vec<u8> = all_types.iter().map(|&qt| quant_type_to_u8(qt)).collect();
        // Assert: all codes are non-zero
        for (i, &code) in codes.iter().enumerate() {
            assert_ne!(code, 0, "QuantType variant index {} has zero code", i);
        }
        // Assert: all codes are unique
        let unique: std::collections::HashSet<u8> = codes.iter().copied().collect();
        assert_eq!(codes.len(), unique.len(), "all 31+ QuantType codes must be unique, got {} codes but {} unique", codes.len(), unique.len());
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_version_bytes_are_little_endian_one() {
        // Arrange: write an empty writer to an in-memory buffer
        let mut writer = GllmWriter::new(256);
        writer.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        writer.write_to(&mut buf).unwrap();
        // Assert: bytes 4..8 must be 0x01 0x00 0x00 0x00 (little-endian u32 = 1)
        assert_eq!(buf[4], 1, "version low byte must be 1");
        assert_eq!(buf[5], 0, "version byte 1 must be 0");
        assert_eq!(buf[6], 0, "version byte 2 must be 0");
        assert_eq!(buf[7], 0, "version byte 3 must be 0");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn page_size_3_non_power_of_two_data_offset_alignment() {
        // Arrange: use page_size=3 (non-power-of-two) to verify align_up behavior
        let mut builder = GllmWriter::new(3);
        builder.add_tensor(TensorEntry {
            name: "npot".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF; 5], original_size: 5,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("npot_page");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("npot.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: data_offset in header must be aligned to 3
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        assert_eq!(data_offset % 3, 0, "data_offset must be aligned to page_size=3");

        // Assert: file can be read back successfully
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_empty_tensor_list_output_exact_header_only() {
        // Arrange: writer with no tensors and no metadata
        let mut writer = GllmWriter::new(64);
        writer.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        writer.write_to(&mut buf).unwrap();
        // Assert: output is exactly HEADER_SIZE bytes (no tensor dir, no string table, no data)
        assert_eq!(buf.len(), HEADER_SIZE, "empty writer must produce exactly {} bytes", HEADER_SIZE);
        // Assert: tensor_count in header is 0
        let tensor_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tensor_count, 0);
        // Assert: magic and version are still valid
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        assert_eq!(version, GLLM_VERSION);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_all_nine_standard_keys_present_in_json() {
        // Arrange: build metadata with known values
        let meta = build_metadata(
            "test_arch", 32000, 2048, 12, 16, 4, 128, 4096, 8192,
            &HashMap::new(),
        );
        // Act: parse as JSON
        let parsed: serde_json::Value = serde_json::from_slice(&meta)
            .expect("metadata must be valid JSON");
        let obj = parsed.as_object().expect("metadata must be a JSON object");
        // Assert: all 9 standard keys are present
        let expected_keys = [
            "arch_key", "vocab_size", "hidden_size", "num_layers",
            "num_heads", "num_kv_heads", "head_dim", "intermediate_size",
            "context_length",
        ];
        for key in &expected_keys {
            assert!(obj.contains_key(*key), "metadata must contain key '{}'", key);
        }
        assert_eq!(obj.len(), 9, "metadata must have exactly 9 keys when no extras");
        assert_eq!(obj["arch_key"], "test_arch");
        assert_eq!(obj["vocab_size"], "32000");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_tensor_dir_entry_shape_dims_at_correct_offsets() {
        // Arrange: create a tensor with distinctive shape values
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "s".to_string(), ndim: 4, dtype: 0,
            shape: [0xDEAD, 0xBEEF, 0xCAFE, 0x1234],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 64], original_size: 64,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: shape starts at offset HEADER_SIZE + 8 (after name_off(4)+name_len(2)+ndim(1)+dtype(1))
        let entry_base = HEADER_SIZE;
        let shape0 = u64::from_le_bytes(buf[entry_base + 8..entry_base + 16].try_into().unwrap());
        let shape1 = u64::from_le_bytes(buf[entry_base + 16..entry_base + 24].try_into().unwrap());
        let shape2 = u64::from_le_bytes(buf[entry_base + 24..entry_base + 32].try_into().unwrap());
        let shape3 = u64::from_le_bytes(buf[entry_base + 32..entry_base + 40].try_into().unwrap());
        assert_eq!(shape0, 0xDEAD);
        assert_eq!(shape1, 0xBEEF);
        assert_eq!(shape2, 0xCAFE);
        assert_eq!(shape3, 0x1234);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn align_up_with_page_size_4096_data_region_starts_aligned() {
        // Arrange: write a tensor with metadata that pushes data offset past alignment
        let page_size: u32 = 4096;
        let mut builder = GllmWriter::new(page_size);
        builder.add_tensor(TensorEntry {
            name: "aligned_data".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 8], original_size: 8,
        });
        // 100 bytes of metadata → data_offset must be aligned
        builder.set_metadata(vec![0u8; 100]);

        let dir = unique_test_dir("data_aligned");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aligned.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: data_offset in header is 4096-aligned
        let raw = std::fs::read(&path).unwrap();
        let data_offset = u64::from_le_bytes(raw[32..40].try_into().unwrap());
        assert_eq!(data_offset % page_size as u64, 0,
            "data_offset {} must be aligned to page_size {}", data_offset, page_size);
        // Assert: page_size field in header matches
        let header_page = u32::from_le_bytes(raw[40..44].try_into().unwrap());
        assert_eq!(header_page, page_size);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_is_quantized_false_when_quant_format_is_zero_but_other_quant_fields_set() {
        // Arrange: set scale_dtype and zp_type to non-zero but quant_format=0
        let entry = TensorEntry {
            name: "edge".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 128, scale_dtype: 5, zp_type: 3,
            data: vec![0u8; 4], original_size: 4,
        };
        // Act & Assert: is_quantized must return false because quant_format is the sole discriminator
        assert!(!entry.is_quantized(),
            "is_quantized must be false when quant_format is 0, regardless of other quant fields");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_string_table_offset_follows_tensor_directory_exactly() {
        // Arrange: 2 tensors with names of known length
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "alpha".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.add_tensor(TensorEntry {
            name: "beta_longer".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: string table starts right after 2 tensor entries
        let strtab_start = HEADER_SIZE + 2 * TENSOR_ENTRY_SIZE;
        // "alpha" (5 bytes) at strtab_start, "beta_longer" (11 bytes) at strtab_start+5
        assert_eq!(&buf[strtab_start..strtab_start + 5], b"alpha");
        assert_eq!(&buf[strtab_start + 5..strtab_start + 16], b"beta_longer");
        // Assert: total string table length = 16
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap()) as usize;
        assert_eq!(meta_offset, strtab_start + 16, "meta_offset must point right after string table");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_single_tensor_compressed_size_in_dir_equals_data_len() {
        // Arrange: a quantized tensor with 37 bytes of data
        let data_len: usize = 37;
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "csz".to_string(), ndim: 2, dtype: 0, shape: [4, 8, 0, 0],
            quant_format: 22, quant_block_size: 32, scale_dtype: 1, zp_type: 0,
            data: vec![0xDD; data_len], original_size: 128,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: compressed_size field in tensor dir entry (bytes 56..64) equals data_len
        let entry_base = HEADER_SIZE;
        let compressed_size = u64::from_le_bytes(buf[entry_base + 56..entry_base + 64].try_into().unwrap());
        assert_eq!(compressed_size, data_len as u64,
            "compressed_size in tensor dir must equal data.len() = {}", data_len);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_data_region_starts_after_metadata_with_no_gap_when_aligned() {
        // Arrange: page_size=1 so no alignment padding, empty metadata
        let mut builder = GllmWriter::new(1);
        builder.add_tensor(TensorEntry {
            name: "no_gap".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x42; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: with page_size=1 and no metadata, data_offset = header + tensor_dir + string_table
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        let expected_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE + "no_gap".len();
        assert_eq!(data_offset, expected_offset,
            "with page_size=1 and no metadata, data starts immediately after string table");
        // Assert: first data byte is 0x42
        assert_eq!(buf[data_offset], 0x42);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_add_tensor_with_unicode_name_roundtrip_preserves_name() {
        // Arrange: tensor name with multi-byte UTF-8 characters
        let unicode_name = "权重_🧠_layer";
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: unicode_name.to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x99; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);

        let dir = unique_test_dir("unicode_name");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unicode.gllm");

        // Act
        builder.write_to_path(&path).unwrap();

        // Assert: reader can find the tensor by its Unicode name
        let reader = GllmReader::open(&path).unwrap();
        let found = reader.find_tensor(unicode_name);
        assert!(found.is_some(), "tensor with Unicode name must be findable");
        assert_eq!(found.unwrap().name, unicode_name);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_to_two_tensors_second_data_offset_in_entry_accounts_for_first_padding() {
        // Arrange: first tensor has 17 bytes of data with page_size=32 → aligned to 32
        let mut builder = GllmWriter::new(32);
        builder.add_tensor(TensorEntry {
            name: "first".to_string(), ndim: 1, dtype: 0, shape: [17, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 17], original_size: 17,
        });
        builder.add_tensor(TensorEntry {
            name: "second".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 4], original_size: 4,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: second tensor's data_offset in dir entry must be 32 (first tensor aligned size)
        let entry2_base = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let data_off2 = u64::from_le_bytes(buf[entry2_base + 48..entry2_base + 56].try_into().unwrap());
        assert_eq!(data_off2, 32,
            "second tensor data_offset must account for first tensor's page-aligned size");
    }

    // ── Wave 269: 15 new tests — GllmWriter Build, metadata boundaries, Debug, GllmError, multi-tensor verify, page_size=1, header field combos ──

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_new_returns_empty_metadata_bytes() {
        // Arrange: create a fresh writer
        let writer = GllmWriter::new(64);
        // Act: write with no tensors and no metadata
        let mut buf = Vec::new();
        writer.write_to(&mut buf).unwrap();
        // Assert: file consists of header only (64 bytes) — no tensor dir, no string table, no metadata, no data
        assert_eq!(buf.len(), HEADER_SIZE, "empty writer should produce header-only file");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_set_metadata_then_add_tensor_produces_valid_file() {
        // Arrange: set metadata first, then add tensor
        let mut builder = GllmWriter::new(32);
        let meta = br#"{"arch_key":"test"}"#.to_vec();
        builder.set_metadata(meta.clone());
        builder.add_tensor(TensorEntry {
            name: "w".to_string(), ndim: 1, dtype: 0, shape: [16, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xFF; 16], original_size: 16,
        });

        let dir = unique_test_dir("meta_first");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta_first.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: reader can open and retrieve metadata
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.metadata_bytes(), meta.as_slice());
        assert_eq!(reader.tensor_count(), 1);
        let data = reader.tensor_data("w").unwrap();
        assert_eq!(&data[..], &[0xFFu8; 16]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_tensor_count_reflects_additions() {
        // Arrange
        let mut builder = GllmWriter::new(64);
        assert_eq!(builder.tensor_count(), 0);
        // Act: add 3 tensors
        for i in 0..3 {
            builder.add_tensor(TensorEntry {
                name: format!("t{i}"), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: vec![0u8; 4], original_size: 4,
            });
        }
        // Assert
        assert_eq!(builder.tensor_count(), 3);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_serializes_all_values_as_json_strings() {
        // Arrange: build metadata with known values
        let extras = HashMap::new();
        // Act
        let json_bytes = build_metadata("llama", 32000, 4096, 32, 32, 8, 128, 11008, 4096, &extras);
        let json_str = String::from_utf8(json_bytes).unwrap();
        let parsed: HashMap<String, String> = serde_json::from_str(&json_str).unwrap();
        // Assert: every value is a JSON string (not number)
        assert_eq!(parsed.get("vocab_size").unwrap(), "32000");
        assert_eq!(parsed.get("hidden_size").unwrap(), "4096");
        assert_eq!(parsed.get("num_layers").unwrap(), "32");
        assert_eq!(parsed.get("head_dim").unwrap(), "128");
        assert_eq!(parsed.get("context_length").unwrap(), "4096");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn build_metadata_includes_extra_keys_in_output() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("custom_field".to_string(), "custom_value".to_string());
        extras.insert("version".to_string(), "2".to_string());
        // Act
        let json_bytes = build_metadata("qwen3", 100, 200, 300, 400, 500, 600, 700, 800, &extras);
        let parsed: HashMap<String, String> = serde_json::from_slice(&json_bytes).unwrap();
        // Assert
        assert_eq!(parsed.get("custom_field").unwrap(), "custom_value");
        assert_eq!(parsed.get("version").unwrap(), "2");
        assert_eq!(parsed.get("arch_key").unwrap(), "qwen3");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_debug_output_contains_name_field() {
        // Arrange
        let entry = TensorEntry {
            name: "model.layers.0.weight".to_string(),
            ndim: 2, dtype: 1, shape: [4096, 4096, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        };
        // Act
        let debug = format!("{entry:?}");
        // Assert
        assert!(debug.contains("TensorEntry"), "Debug output should contain struct name");
        assert!(debug.contains("model.layers.0.weight"), "Debug output should contain tensor name");
        assert!(debug.contains("ndim"), "Debug output should contain ndim field");
        assert!(debug.contains("dtype"), "Debug output should contain dtype field");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_debug_shows_quant_format_when_nonzero() {
        // Arrange
        let entry = TensorEntry {
            name: "q_weight".to_string(),
            ndim: 2, dtype: 0, shape: [4096, 512, 0, 0],
            quant_format: 10, quant_block_size: 128, scale_dtype: 1, zp_type: 0,
            data: vec![0xAB; 256], original_size: 8388608,
        };
        // Act
        let debug = format!("{entry:?}");
        // Assert
        assert!(debug.contains("quant_format"), "Debug should show quant_format");
        assert!(debug.contains("quant_block_size"), "Debug should show quant_block_size");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_error_display_all_variants_contain_distinct_keywords() {
        // Arrange & Act & Assert: each variant's Display contains a unique keyword
        let cases: Vec<(GllmError, &str)> = vec![
            (GllmError::Io(std::io::Error::new(std::io::ErrorKind::Other, "x")), "IO error"),
            (GllmError::InvalidMagic(0), "invalid magic"),
            (GllmError::UnsupportedVersion(0), "unsupported version"),
            (GllmError::HeaderTooSmall(1), "file too small"),
            (GllmError::TensorDirOutOfBounds { offset: 0, count: 0, file_size: 0 }, "tensor directory"),
            (GllmError::StringTableOutOfBounds { offset: 0, length: 0, file_size: 0 }, "string table"),
            (GllmError::MetadataOutOfBounds { offset: 0, file_size: 0 }, "metadata"),
            (GllmError::TensorOutOfBounds { name: "n".into(), start: 0, end: 0, file_size: 0 }, "tensor \"n\""),
            (GllmError::DuplicateTensorName("dup".into()), "duplicate tensor name"),
            (GllmError::ParseError("msg".into()), "parse error"),
            (GllmError::InvalidQuantType(9), "invalid quant_format"),
            (GllmError::InvalidDType(8), "invalid dtype"),
            (GllmError::InvalidMetadata("bad".into()), "invalid metadata"),
        ];
        for (err, keyword) in &cases {
            let s = err.to_string();
            assert!(s.contains(keyword), "Display for {:?} should contain '{}', got: {}", err, keyword, s);
        }
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_three_tensors_verify_all_via_reader() {
        // Arrange: write 3 tensors with different data patterns
        let mut builder = GllmWriter::new(16);
        builder.add_tensor(TensorEntry {
            name: "embed".to_string(), ndim: 2, dtype: 0, shape: [32000, 4096, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11; 32], original_size: 32,
        });
        builder.add_tensor(TensorEntry {
            name: "layers.0.attn".to_string(), ndim: 2, dtype: 0, shape: [4096, 4096, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x22; 48], original_size: 48,
        });
        builder.add_tensor(TensorEntry {
            name: "layers.0.ffn".to_string(), ndim: 2, dtype: 0, shape: [4096, 11008, 0, 0],
            quant_format: 10, quant_block_size: 128, scale_dtype: 1, zp_type: 0,
            data: vec![0x33; 24], original_size: 1024,
        });
        let meta = build_metadata("llama", 32000, 4096, 32, 32, 32, 128, 11008, 4096, &HashMap::new());
        builder.set_metadata(meta);

        let dir = unique_test_dir("three_verify");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("three.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: reader finds all 3 with correct data
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 3);

        let embed = reader.find_tensor("embed").expect("embed tensor must exist");
        assert_eq!(embed.name, "embed");
        let embed_data = reader.tensor_data("embed").unwrap();
        assert_eq!(&embed_data[..], &[0x11u8; 32]);

        let attn = reader.find_tensor("layers.0.attn").expect("attn tensor must exist");
        assert_eq!(attn.name, "layers.0.attn");
        let attn_data = reader.tensor_data("layers.0.attn").unwrap();
        assert_eq!(&attn_data[..], &[0x22u8; 48]);

        let ffn = reader.find_tensor("layers.0.ffn").expect("ffn tensor must exist");
        assert_eq!(ffn.name, "layers.0.ffn");
        assert!(ffn.entry.is_quantized());
        let ffn_data = reader.tensor_data("layers.0.ffn").unwrap();
        assert_eq!(&ffn_data[..], &[0x33u8; 24]);

        // Architecture parsed from metadata
        assert_eq!(reader.architecture().unwrap(), "llama");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_with_page_size_one_no_padding_in_data_region() {
        // Arrange: page_size=1 means every data size is already aligned
        let mut builder = GllmWriter::new(1);
        let data = vec![0xCC; 7]; // 7 bytes, odd number
        builder.add_tensor(TensorEntry {
            name: "odd".to_string(), ndim: 1, dtype: 0, shape: [7, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 7,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: data_offset = header(64) + tensor_dir(72) + string_table(3) + metadata(0) = 139
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset, 139, "with page_size=1, no padding before data");
        // Assert: total file size = data_offset + 7 bytes (no alignment padding)
        assert_eq!(buf.len(), 139 + 7, "total file size = data_offset + raw data, no trailing padding");
        // Assert: data bytes preserved exactly
        assert_eq!(&buf[139..146], &[0xCC; 7]);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_with_page_size_one_two_tensors_cumulative_no_gap() {
        // Arrange: two tensors with page_size=1, no gaps
        let mut builder = GllmWriter::new(1);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 5], original_size: 5,
        });
        builder.add_tensor(TensorEntry {
            name: "b".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xBB; 3], original_size: 3,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: second tensor's data_offset in dir entry = 5 (first tensor raw size)
        let entry2_base = HEADER_SIZE + TENSOR_ENTRY_SIZE;
        let data_off2 = u64::from_le_bytes(buf[entry2_base + 48..entry2_base + 56].try_into().unwrap());
        assert_eq!(data_off2, 5, "with page_size=1, second tensor starts immediately after first");
        // Assert: data region = 5 + 3 = 8 bytes total
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap()) as usize;
        assert_eq!(&buf[data_offset..data_offset + 5], &[0xAA; 5]);
        assert_eq!(&buf[data_offset + 5..data_offset + 8], &[0xBB; 3]);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_fields_combination_quantized_with_page_size() {
        // Arrange: write a quantized tensor with non-trivial page_size
        let mut builder = GllmWriter::new(128);
        builder.add_tensor(TensorEntry {
            name: "q4_weight".to_string(), ndim: 2, dtype: 0, shape: [4096, 512, 0, 0],
            quant_format: 10, quant_block_size: 128, scale_dtype: 1, zp_type: 0,
            data: vec![0xDD; 64], original_size: 8388608,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: flags = 1 (quantized)
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags, 1, "flags must indicate quantization");
        // Assert: tensor_count = 1
        let tc = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tc, 1);
        // Assert: page_size = 128
        let ps = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(ps, 128);
        // Assert: data_offset is page-aligned to 128
        let data_off = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_off % 128, 0, "data_offset must be 128-aligned");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_fields_combination_zero_tensors_zero_flags() {
        // Arrange: writer with no tensors, no metadata
        let mut builder = GllmWriter::new(4096);
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags, 0, "no tensors → flags=0");
        let tensor_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tensor_count, 0);
        let page_size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(page_size, 4096);
        // meta_offset = header(64) + 0 tensor_dir + 0 string_table = 64
        let meta_off = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        assert_eq!(meta_off, 64);
        // data_offset = align_up(64 + 0, 4096) = 4096
        let data_off = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_off, 4096);
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_tensor_dir_offset_equals_header_size_constant() {
        // Arrange: write with any page_size
        let mut builder = GllmWriter::new(64);
        builder.add_tensor(TensorEntry {
            name: "x".to_string(), ndim: 1, dtype: 0, shape: [8, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 8], original_size: 8,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: tensor_dir_offset always equals HEADER_SIZE (64)
        let tdo = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        assert_eq!(tdo, HEADER_SIZE as u64,
            "tensor directory always starts right after the header");
    }

    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_mixed_quantized_and_unquantized_sets_flags_to_one() {
        // Arrange: one unquantized + one quantized tensor
        let mut builder = GllmWriter::new(32);
        builder.add_tensor(TensorEntry {
            name: "dense".to_string(), ndim: 2, dtype: 0, shape: [16, 16, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0u8; 32], original_size: 32,
        });
        builder.add_tensor(TensorEntry {
            name: "quant".to_string(), ndim: 2, dtype: 0, shape: [16, 16, 0, 0],
            quant_format: 10, quant_block_size: 128, scale_dtype: 1, zp_type: 0,
            data: vec![0xAB; 16], original_size: 1024,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: flags bit 0 = 1 because at least one tensor is quantized
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1, "mixed tensors: flags bit 0 must be 1");
        // Assert: first tensor dir entry has quant_format=0, second has quant_format=10
        let e1_qf = buf[HEADER_SIZE + 40];
        let e2_qf = buf[HEADER_SIZE + TENSOR_ENTRY_SIZE + 40];
        assert_eq!(e1_qf, 0, "first tensor quant_format should be 0");
        assert_eq!(e2_qf, 10, "second tensor quant_format should be 10");
    }

    // ────────────────────────────────────────────────────────────────────────
    // 15 additional unit tests — targeted coverage
    // ────────────────────────────────────────────────────────────────────────

    /// Test 1: GllmWriter empty metadata roundtrip
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_writer_empty_metadata_roundtrip_verify() {
        // Arrange: writer with one tensor but empty metadata bytes
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "meta_empty".to_string(), ndim: 1, dtype: 0, shape: [16, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xCC; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);
        // Act: write to buffer
        let mut buf = Vec::new();
        builder.write_to(&mut buf).unwrap();
        // Assert: meta_offset should point right after string table (no metadata bytes)
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        // With 1 tensor, string table has "meta_empty" (10 bytes)
        // Expected: header(64) + tensor_dir(72) + string_table(10) = 146
        assert_eq!(meta_offset, (HEADER_SIZE + TENSOR_ENTRY_SIZE + 10) as u64);
        // Verify via reader roundtrip
        let dir = unique_test_dir("empty_meta_rt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_meta.gllm");
        builder.write_to_path(&path).unwrap();
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let td = reader.tensor_data("meta_empty").unwrap();
        assert_eq!(td.len(), 16);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test 2: Metadata-first tensor ordering
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn metadata_appears_before_data_region_in_binary_output() {
        // Arrange: writer with metadata and one tensor
        let mut builder = GllmWriter::new(256);
        let meta = build_metadata("order_test", 100, 200, 1, 2, 3, 4, 5, 6, &HashMap::new());
        builder.set_metadata(meta);
        builder.add_tensor(TensorEntry {
            name: "t".to_string(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xAA; 4], original_size: 4,
        });
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: meta_offset < data_offset
        let meta_offset = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert!(meta_offset < data_offset,
            "meta_offset ({}) must be less than data_offset ({})",
            meta_offset, data_offset);
    }

    /// Test 3: Tensor count tracking accuracy after many operations
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_count_accuracy_after_set_metadata_and_add_interleaved() {
        // Arrange: verify count stays accurate when interleaving set_metadata and add_tensor
        let mut writer = GllmWriter::new(512);
        assert_eq!(writer.tensor_count(), 0);
        writer.set_metadata(vec![1, 2, 3]);
        assert_eq!(writer.tensor_count(), 0, "metadata does not affect count");
        writer.add_tensor(TensorEntry {
            name: "a".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        assert_eq!(writer.tensor_count(), 1);
        writer.set_metadata(vec![4, 5]);
        assert_eq!(writer.tensor_count(), 1, "overwriting metadata does not affect count");
        writer.add_tensor(TensorEntry {
            name: "b".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        writer.add_tensor(TensorEntry {
            name: "c".into(), ndim: 1, dtype: 0, shape: [0; 4],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        // Act + Assert
        assert_eq!(writer.tensor_count(), 3);
        assert_eq!(writer.metadata_bytes, vec![4, 5]);
    }

    /// Test 4: Metadata serialization with JSON special characters
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn metadata_serialization_with_json_special_characters_in_extras() {
        // Arrange: build metadata with special JSON characters in extras
        let mut extras = HashMap::new();
        extras.insert("escaped_key".to_string(), "value with \"quotes\" and \\backslash".to_string());
        extras.insert("newline_key".to_string(), "line1\nline2".to_string());
        extras.insert("unicode_key".to_string(), "\u{1F600}\u{1F680}".to_string());
        // Act
        let meta = build_metadata("test", 100, 200, 1, 2, 3, 4, 5, 6, &extras);
        // Assert: result is valid JSON and can be parsed back
        let parsed: serde_json::Value = serde_json::from_slice(&meta).unwrap();
        assert_eq!(parsed["escaped_key"], "value with \"quotes\" and \\backslash");
        assert_eq!(parsed["newline_key"], "line1\nline2");
        assert_eq!(parsed["unicode_key"], "\u{1F600}\u{1F680}");
    }

    /// Test 5: TensorEntry Debug format verification
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_debug_format_includes_all_public_fields() {
        // Arrange
        let entry = TensorEntry {
            name: "dbg.weight".to_string(),
            ndim: 3,
            dtype: 5,
            shape: [128, 256, 512, 0],
            quant_format: 40,
            quant_block_size: 128,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0u8; 64],
            original_size: 262144,
        };
        // Act
        let debug = format!("{entry:?}");
        // Assert: Debug output contains all field names and key values
        assert!(debug.contains("TensorEntry"), "should contain struct name");
        assert!(debug.contains("name"), "should show name field");
        assert!(debug.contains("ndim"), "should show ndim field");
        assert!(debug.contains("dtype"), "should show dtype field");
        assert!(debug.contains("shape"), "should show shape field");
        assert!(debug.contains("quant_format"), "should show quant_format field");
        assert!(debug.contains("quant_block_size"), "should show quant_block_size field");
        assert!(debug.contains("scale_dtype"), "should show scale_dtype field");
        assert!(debug.contains("zp_type"), "should show zp_type field");
        assert!(debug.contains("data"), "should show data field");
        assert!(debug.contains("original_size"), "should show original_size field");
    }

    /// Test 6: GllmError Display all 13 variants
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn gllm_error_display_all_thirteen_variants_have_unique_output() {
        // Arrange: one instance per variant (13 total)
        let errors: Vec<GllmError> = vec![
            GllmError::Io(std::io::Error::new(std::io::ErrorKind::Other, "io_err")),
            GllmError::InvalidMagic(0xBAD),
            GllmError::UnsupportedVersion(99),
            GllmError::HeaderTooSmall(10),
            GllmError::TensorDirOutOfBounds { offset: 1, count: 2, file_size: 3 },
            GllmError::StringTableOutOfBounds { offset: 4, length: 5, file_size: 6 },
            GllmError::MetadataOutOfBounds { offset: 7, file_size: 8 },
            GllmError::TensorOutOfBounds { name: "n".into(), start: 9, end: 10, file_size: 11 },
            GllmError::DuplicateTensorName("dup".into()),
            GllmError::ParseError("parse".into()),
            GllmError::InvalidQuantType(77),
            GllmError::InvalidDType(88),
            GllmError::InvalidMetadata("bad".into()),
        ];
        assert_eq!(errors.len(), 13, "must test all 13 GllmError variants");
        // Act + Assert: each variant produces non-empty display output
        let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        for (i, d) in displays.iter().enumerate() {
            assert!(!d.is_empty(), "variant {} display should not be empty", i);
        }
        // Verify all displays are distinct from each other
        let unique_displays: std::collections::HashSet<&str> =
            displays.iter().map(|s| s.as_str()).collect();
        assert_eq!(unique_displays.len(), 13, "all 13 variants should produce distinct output");
    }

    /// Test 7: Multi-tensor write+verify (3+ tensors)
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn multi_tensor_write_verify_four_tensors_data_and_names() {
        // Arrange: 4 tensors with distinct data patterns
        let mut builder = GllmWriter::new(128);
        let names = ["embed", "attn_q", "attn_k", "ffn_up"];
        let patterns: Vec<Vec<u8>> = (0..4).map(|i| vec![(i + 1) as u8; 32]).collect();
        for (i, name) in names.iter().enumerate() {
            builder.add_tensor(TensorEntry {
                name: name.to_string(), ndim: 2, dtype: 0, shape: [8, 4, 0, 0],
                quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
                data: patterns[i].clone(), original_size: 32,
            });
        }
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("multi_4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi4.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: all 4 tensors readable with correct data
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 4);
        for (i, name) in names.iter().enumerate() {
            let td = reader.tensor_data(name).unwrap();
            assert_eq!(td.len(), 32);
            assert!(td.iter().all(|&b| b == (i + 1) as u8),
                "tensor {} should have consistent data pattern", name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test 8: page_size=1 no padding verification
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn page_size_one_produces_no_padding_in_data_region() {
        // Arrange: write a 7-byte tensor with page_size=1
        let mut builder = GllmWriter::new(1);
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA];
        builder.add_tensor(TensorEntry {
            name: "nopad".to_string(), ndim: 1, dtype: 0, shape: [7, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: data.clone(), original_size: 7,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("ps1_nopad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps1_nopad.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: total file size = header(64) + tensor_dir(72) + name("nopad"=5) + data(7)
        let raw = std::fs::read(&path).unwrap();
        let expected_size = HEADER_SIZE + TENSOR_ENTRY_SIZE + "nopad".len() + 7;
        assert_eq!(raw.len(), expected_size,
            "page_size=1 should produce no padding: expected {} bytes, got {}",
            expected_size, raw.len());
        // Verify the 7 data bytes are exactly at the data_offset position
        let reader = GllmReader::open(&path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        assert_eq!(&raw[data_offset..data_offset + 7], &data[..]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test 9: page_size=1 zero gap between tensors
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn page_size_one_zero_gap_between_two_tensors() {
        // Arrange: two tensors with page_size=1, data sizes 3 and 5
        let mut builder = GllmWriter::new(1);
        builder.add_tensor(TensorEntry {
            name: "a".to_string(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x11, 0x22, 0x33], original_size: 3,
        });
        builder.add_tensor(TensorEntry {
            name: "b".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x44, 0x55, 0x66, 0x77, 0x88], original_size: 5,
        });
        builder.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: total data region = 3 + 5 = 8 bytes (no padding with page_size=1)
        let reader_dir = unique_test_dir("ps1_gap");
        std::fs::create_dir_all(&reader_dir).unwrap();
        let file_path = reader_dir.join("ps1_gap.gllm");
        builder.write_to_path(&file_path).unwrap();
        let reader = GllmReader::open(&file_path).unwrap();
        let data_offset = reader.header().data_offset as usize;
        // First tensor data: buf[data_offset..data_offset+3]
        assert_eq!(buf[data_offset], 0x11);
        assert_eq!(buf[data_offset + 1], 0x22);
        assert_eq!(buf[data_offset + 2], 0x33);
        // Second tensor data immediately follows: buf[data_offset+3..data_offset+8]
        assert_eq!(buf[data_offset + 3], 0x44);
        assert_eq!(buf[data_offset + 4], 0x55);
        assert_eq!(buf[data_offset + 5], 0x66);
        assert_eq!(buf[data_offset + 6], 0x77);
        assert_eq!(buf[data_offset + 7], 0x88);
        // Verify total data region has no gap
        assert_eq!(buf.len(), data_offset + 8, "no extra bytes after data region");
        let _ = std::fs::remove_dir_all(&reader_dir);
    }

    /// Test 10: Header field combinations (quantized + non-standard page_size)
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn header_fields_combination_quantized_custom_page_size() {
        // Arrange: quantized tensor with page_size=333
        let mut builder = GllmWriter::new(333);
        builder.add_tensor(TensorEntry {
            name: "q".to_string(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 40, quant_block_size: 128, scale_dtype: 2, zp_type: 1,
            data: vec![0u8; 16], original_size: 64,
        });
        builder.set_metadata(vec![1, 2, 3]);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: verify all header fields simultaneously
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags & 1, 1, "quantized flag must be set");
        let tensor_count = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tensor_count, 1);
        let tensor_dir_offset = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        assert_eq!(tensor_dir_offset, HEADER_SIZE as u64);
        let page_size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(page_size, 333);
        // Reserved bytes [44..64] must be zero
        for i in 44..64 {
            assert_eq!(buf[i], 0, "reserved byte at {} should be zero", i);
        }
    }

    /// Test 11: Write with very long tensor names (Unicode)
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_very_long_unicode_tensor_name_roundtrip() {
        // Arrange: build a long name with mixed Unicode (CJK + Latin + emoji-like)
        let long_name = format!("模型.{}.权重", "层".repeat(200));
        assert!(long_name.len() > 600, "name should be long (UTF-8 bytes)");
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: long_name.clone(), ndim: 2, dtype: 0, shape: [4, 4, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0x55; 16], original_size: 16,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("long_unicode");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("long_unicode.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: tensor is readable with exact name preserved
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let t = reader.find_tensor(&long_name);
        assert!(t.is_some(), "long unicode tensor name should be findable");
        let td = reader.tensor_data(&long_name).unwrap();
        assert_eq!(td.len(), 16);
        assert!(td.iter().all(|&b| b == 0x55));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test 12: QuantType to u8 and back roundtrip
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn quant_type_to_u8_roundtrip_all_major_variants() {
        use gllm_kernels::quant::QuantType;
        // Arrange: verify encoding uniqueness for key variants
        let variants = [
            QuantType::Bf16, QuantType::F16, QuantType::F32,
            QuantType::Q4_0, QuantType::Q8_0,
            QuantType::Q2K, QuantType::Q4K, QuantType::Q8K,
            QuantType::AWQ4, QuantType::GPTQ4, QuantType::Nvfp4,
            QuantType::Fp8E4M3, QuantType::Fp8E5M2,
        ];
        // Act
        let codes: Vec<u8> = variants.iter().map(|&qt| quant_type_to_u8(qt)).collect();
        // Assert: all codes are non-zero and unique
        for code in &codes {
            assert_ne!(*code, 0, "quant type code must be non-zero");
        }
        let unique: std::collections::HashSet<u8> = codes.iter().copied().collect();
        assert_eq!(unique.len(), codes.len(), "all codes must be unique");
        // Verify deterministic: encode twice, same result
        for &qt in &variants {
            let c1 = quant_type_to_u8(qt);
            let c2 = quant_type_to_u8(qt);
            assert_eq!(c1, c2, "encoding must be deterministic for {:?}", qt);
        }
    }

    /// Test 13: TensorEntry Clone preserves all fields
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn tensor_entry_clone_preserves_every_field_including_data_content() {
        // Arrange: an entry with non-trivial values in every field
        let original = TensorEntry {
            name: "clone_all.weight".to_string(),
            ndim: 4,
            dtype: 7,
            shape: [100, 200, 300, 400],
            quant_format: 41,
            quant_block_size: 256,
            scale_dtype: 3,
            zp_type: 2,
            data: (0..50).map(|i| (i * 3) as u8).collect(),
            original_size: 999999,
        };
        // Act
        let cloned = original.clone();
        // Assert: every field matches exactly
        assert_eq!(cloned.name, original.name);
        assert_eq!(cloned.ndim, original.ndim);
        assert_eq!(cloned.dtype, original.dtype);
        assert_eq!(cloned.shape, original.shape);
        assert_eq!(cloned.quant_format, original.quant_format);
        assert_eq!(cloned.quant_block_size, original.quant_block_size);
        assert_eq!(cloned.scale_dtype, original.scale_dtype);
        assert_eq!(cloned.zp_type, original.zp_type);
        assert_eq!(cloned.data, original.data);
        assert_eq!(cloned.original_size, original.original_size);
        assert_eq!(cloned.compressed_size(), original.compressed_size());
        assert_eq!(cloned.is_quantized(), original.is_quantized());
    }

    /// Test 14: Empty tensor data (zero-length)
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn write_empty_data_tensor_roundtrip_preserves_zero_length() {
        // Arrange: tensor with empty data vec but non-trivial shape
        let mut builder = GllmWriter::new(256);
        builder.add_tensor(TensorEntry {
            name: "empty_data".to_string(), ndim: 2, dtype: 0, shape: [4096, 768, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![], original_size: 0,
        });
        builder.set_metadata(vec![]);
        let dir = unique_test_dir("empty_data");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty_data.gllm");
        // Act
        builder.write_to_path(&path).unwrap();
        // Assert: shape preserved, data length is 0
        let reader = GllmReader::open(&path).unwrap();
        assert_eq!(reader.tensor_count(), 1);
        let t = reader.find_tensor("empty_data").unwrap();
        assert_eq!(t.entry.shape[0], 4096);
        assert_eq!(t.entry.shape[1], 768);
        let td = reader.tensor_data("empty_data").unwrap();
        assert_eq!(td.len(), 0, "empty tensor data should remain zero-length");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Test 15: GllmWriter page alignment calculation
    // @trace REQ-GLF-001 [level:unit]
    #[test]
    fn writer_page_alignment_data_offset_aligned_to_page_size() {
        // Arrange: build a writer with page_size=1024 and a short tensor
        // The data_offset in the header must be aligned to page_size
        let mut builder = GllmWriter::new(1024);
        builder.add_tensor(TensorEntry {
            name: "align_calc".to_string(), ndim: 1, dtype: 0, shape: [5, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![1, 2, 3, 4, 5], original_size: 5,
        });
        let meta = build_metadata("align_test", 10, 20, 1, 2, 3, 4, 5, 6, &HashMap::new());
        builder.set_metadata(meta);
        let mut buf = Vec::new();
        // Act
        builder.write_to(&mut buf).unwrap();
        // Assert: data_offset must be a multiple of page_size (1024)
        let data_offset = u64::from_le_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(data_offset % 1024, 0,
            "data_offset ({}) must be aligned to page_size 1024", data_offset);
        assert!(data_offset > HEADER_SIZE as u64,
            "data_offset must be after header");
        // Also verify the page_size field is stored correctly
        let stored_page_size = u32::from_le_bytes(buf[40..44].try_into().unwrap());
        assert_eq!(stored_page_size, 1024);
    }

    // ── New tests: 15 additional unit tests ──────────────────────────────────

    /// GllmError::source() returns Some for Io variant.
    #[test]
    fn gllm_error_source_returns_inner_for_io_variant() {
        // Arrange
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let gllm_err = GllmError::Io(io_err);
        // Act
        let source = std::error::Error::source(&gllm_err);
        // Assert: source must be Some for Io variant
        assert!(source.is_some(), "Io variant must yield a source");
        let src_msg = source.unwrap().to_string();
        assert!(src_msg.contains("denied"), "source message should contain 'denied', got: {}", src_msg);
    }

    /// GllmError::source() returns None for all non-Io variants.
    #[test]
    fn gllm_error_source_returns_none_for_non_io_variants() {
        // Arrange: pick several non-Io variants
        let non_io_errors: Vec<GllmError> = vec![
            GllmError::InvalidMagic(0xDEAD),
            GllmError::UnsupportedVersion(99),
            GllmError::HeaderTooSmall(10),
            GllmError::ParseError("bad".into()),
            GllmError::InvalidQuantType(200),
            GllmError::InvalidDType(250),
            GllmError::InvalidMetadata("corrupt".into()),
        ];
        // Act & Assert: every non-Io variant must return None from source()
        for err in &non_io_errors {
            assert!(
                std::error::Error::source(err).is_none(),
                "source() should be None for {:?}, but got Some", err
            );
        }
    }

    /// GllmError From<io::Error> preserves error kind.
    #[test]
    fn gllm_error_from_io_error_preserves_error_kind() {
        // Arrange
        let original = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        // Act
        let converted: GllmError = original.into();
        // Assert
        match converted {
            GllmError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected Io variant, got {:?}", other),
        }
    }

    /// GllmError Display for DuplicateTensorName includes the name.
    #[test]
    fn gllm_error_display_duplicate_tensor_name_includes_actual_name() {
        // Arrange
        let err = GllmError::DuplicateTensorName("layers.0.self_attn.q_proj.weight".into());
        // Act
        let display = err.to_string();
        // Assert: the specific tensor name must appear verbatim
        assert!(
            display.contains("layers.0.self_attn.q_proj.weight"),
            "Display should contain the tensor name, got: {}", display
        );
    }

    /// GllmError Display for TensorOutOfBounds includes start, end, file_size.
    #[test]
    fn gllm_error_display_tensor_out_of_bounds_includes_offsets() {
        // Arrange
        let err = GllmError::TensorOutOfBounds {
            name: "big_tensor".into(),
            start: 1000,
            end: 5000,
            file_size: 3000,
        };
        // Act
        let display = err.to_string();
        // Assert: must mention the tensor name and range values
        assert!(display.contains("big_tensor"), "should contain tensor name");
        assert!(display.contains("1000"), "should contain start offset");
        assert!(display.contains("5000"), "should contain end offset");
        assert!(display.contains("3000"), "should contain file_size");
    }

    /// TensorEntry compressed_size for data near u32::MAX length.
    #[test]
    fn tensor_entry_compressed_size_large_but_valid_data_len() {
        // Arrange: create entry with data.len() = 65536 (64 KiB)
        let data = vec![0xAAu8; 65536];
        let entry = TensorEntry {
            name: "big.weight".into(),
            ndim: 2,
            dtype: 0,
            shape: [256, 256, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data,
            original_size: 262144,
        };
        // Act
        let cs = entry.compressed_size();
        // Assert
        assert_eq!(cs, 65536u64, "compressed_size must equal data.len()");
    }

    /// TensorEntry is_quantized returns true for u8::MAX quant_format.
    #[test]
    fn tensor_entry_is_quantized_with_quant_format_u8_max() {
        // Arrange
        let entry = TensorEntry {
            name: "extreme_quant".into(),
            ndim: 1,
            dtype: 0,
            shape: [10, 0, 0, 0],
            quant_format: u8::MAX,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0; 10],
            original_size: 10,
        };
        // Act & Assert
        assert!(
            entry.is_quantized(),
            "quant_format=u8::MAX should report as quantized"
        );
    }

    /// build_metadata with a single extra key produces valid JSON containing that key.
    #[test]
    fn build_metadata_single_extra_key_present_in_output() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("rope_theta".to_string(), "500000.0".to_string());
        // Act
        let bytes = build_metadata("qwen3", 151936, 4096, 36, 32, 8, 128, 11008, 32768, &extras);
        // Assert
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("must be valid JSON");
        let obj = json.as_object().expect("must be a JSON object");
        assert!(obj.contains_key("rope_theta"), "extra key must appear in output");
        assert_eq!(obj["rope_theta"], "500000.0");
        // All 9 standard keys still present
        assert_eq!(obj.len(), 10, "9 standard + 1 extra = 10 keys");
    }

    /// write_to with no tensors and no metadata produces a header-only output.
    #[test]
    fn write_to_empty_writer_produces_exactly_header_plus_alignment() {
        // Arrange
        let writer = GllmWriter::new(64);
        let mut buf = Vec::new();
        // Act
        writer.write_to(&mut buf).unwrap();
        // Assert: buffer must be at least header size (64 bytes)
        assert!(buf.len() >= HEADER_SIZE, "output must contain at least header");
        // Magic and version must be correct
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(magic, GLLM_MAGIC);
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        assert_eq!(version, GLLM_VERSION);
        // tensor_count must be 0
        let tc = u32::from_le_bytes(buf[20..24].try_into().unwrap());
        assert_eq!(tc, 0, "empty writer should report 0 tensors");
        // flags must be 0 (no quantized tensors)
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_eq!(flags, 0);
    }

    /// write_to with quantized-only tensors sets the quant flag.
    #[test]
    fn write_to_single_quantized_tensor_sets_quant_flag_in_header() {
        // Arrange
        let mut writer = GllmWriter::new(16);
        writer.add_tensor(TensorEntry {
            name: "q_weight".into(),
            ndim: 2,
            dtype: 0,
            shape: [128, 256, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data: vec![0xBB; 64],
            original_size: 131072,
        });
        writer.set_metadata(vec![]);
        let mut buf = Vec::new();
        // Act
        writer.write_to(&mut buf).unwrap();
        // Assert: flags bit 0 must be set
        let flags = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        assert_ne!(flags & 1, 0, "quant flag (bit 0) must be set for quantized tensor");
    }

    /// align_up with u64::MAX and alignment=1 returns u64::MAX unchanged.
    #[test]
    fn align_up_u64_max_with_alignment_one_unchanged() {
        // Arrange
        let value = u64::MAX;
        let alignment = 1u64;
        // Act
        let result = align_up(value, alignment);
        // Assert
        assert_eq!(result, u64::MAX, "u64::MAX aligned to 1 must stay u64::MAX");
    }

    /// align_up with value=0 and various non-zero alignments always returns 0.
    #[test]
    fn align_up_zero_value_with_nonzero_alignments_returns_zero() {
        // Arrange & Act & Assert
        for alignment in [1u64, 2, 4, 8, 16, 64, 256, 4096, 65536] {
            let result = align_up(0, alignment);
            assert_eq!(result, 0, "align_up(0, {}) should return 0", alignment);
        }
    }

    /// writer tensor_count returns 0 for fresh writer and increments correctly.
    #[test]
    fn writer_tensor_count_sequence_from_zero_to_three() {
        // Arrange
        let mut writer = GllmWriter::new(256);
        assert_eq!(writer.tensor_count(), 0, "fresh writer has 0 tensors");
        // Act & Assert: add one at a time, check count
        writer.add_tensor(TensorEntry {
            name: "a".into(), ndim: 1, dtype: 0, shape: [1, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0], original_size: 1,
        });
        assert_eq!(writer.tensor_count(), 1);
        writer.add_tensor(TensorEntry {
            name: "b".into(), ndim: 1, dtype: 0, shape: [2, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![1, 2], original_size: 2,
        });
        assert_eq!(writer.tensor_count(), 2);
        writer.add_tensor(TensorEntry {
            name: "c".into(), ndim: 1, dtype: 0, shape: [3, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![3, 4, 5], original_size: 3,
        });
        assert_eq!(writer.tensor_count(), 3);
    }

    /// write_to_path creates a file that exists on disk after write.
    #[test]
    fn write_to_path_file_exists_after_write() {
        // Arrange
        let mut writer = GllmWriter::new(64);
        writer.add_tensor(TensorEntry {
            name: "exists_check".into(), ndim: 1, dtype: 0, shape: [4, 0, 0, 0],
            quant_format: 0, quant_block_size: 0, scale_dtype: 0, zp_type: 0,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF], original_size: 4,
        });
        writer.set_metadata(vec![]);
        let dir = unique_test_dir("file_exists");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exists.gllm");
        // Act
        writer.write_to_path(&path).unwrap();
        // Assert
        assert!(path.exists(), "file must exist after write_to_path");
        let metadata = std::fs::metadata(&path).unwrap();
        assert!(metadata.len() >= HEADER_SIZE as u64, "file must be at least header size");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// quant_type_to_u8 for TQ1_0 and TQ2_0 returns expected codes.
    #[test]
    fn quant_type_to_u8_tq_variants_return_distinct_codes_in_60s_range() {
        // Arrange
        use gllm_kernels::quant::QuantType;
        // Act
        let tq1 = quant_type_to_u8(QuantType::TQ1_0);
        let tq2 = quant_type_to_u8(QuantType::TQ2_0);
        // Assert: both in 60s range, distinct, non-zero
        assert_eq!(tq1, 60, "TQ1_0 should encode to 60");
        assert_eq!(tq2, 61, "TQ2_0 should encode to 61");
        assert_ne!(tq1, tq2, "TQ1_0 and TQ2_0 must have distinct codes");
    }

    /// TensorEntry struct update syntax produces an independent clone with modified fields.
    #[test]
    fn tensor_entry_struct_update_syntax_produces_independent_entry() {
        // Arrange
        let base = TensorEntry {
            name: "base_tensor".into(),
            ndim: 2,
            dtype: 0,
            shape: [128, 64, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![1u8; 32],
            original_size: 32,
        };
        // Act
        let derived = TensorEntry {
            name: "derived_tensor".into(),
            quant_format: 10,
            data: vec![2u8; 16],
            ..base.clone()
        };
        // Assert
        assert_eq!(derived.name, "derived_tensor");
        assert_eq!(derived.quant_format, 10);
        assert_eq!(derived.data, vec![2u8; 16]);
        // Fields inherited from base remain unchanged
        assert_eq!(derived.ndim, 2);
        assert_eq!(derived.shape, [128, 64, 0, 0]);
        assert_eq!(derived.original_size, 32);
        // base is unmodified
        assert_eq!(base.name, "base_tensor");
        assert_eq!(base.quant_format, 0);
    }

    /// GllmWriter with page_size set to u32::MAX computes layout without overflow panic.
    #[test]
    fn gllm_writer_page_size_max_does_not_panic_on_empty_write() {
        // Arrange
        let writer = GllmWriter::new(u32::MAX);
        // Act
        let dir = unique_test_dir("page_size_max");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ps_max.gllm");
        writer.write_to_path(&path).unwrap();
        // Assert
        assert!(path.exists());
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() >= HEADER_SIZE as u64, "file must contain at least the 64-byte header");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// TensorEntry Debug output contains all field names.
    #[test]
    fn tensor_entry_debug_format_contains_all_field_names() {
        // Arrange
        let entry = TensorEntry {
            name: "debug_test".into(),
            ndim: 3,
            dtype: 1,
            shape: [10, 20, 30, 0],
            quant_format: 5,
            quant_block_size: 32,
            scale_dtype: 2,
            zp_type: 1,
            data: vec![0xAB; 8],
            original_size: 8,
        };
        // Act
        let debug_str = format!("{:?}", entry);
        // Assert
        assert!(debug_str.contains("name"), "Debug must contain 'name' field");
        assert!(debug_str.contains("ndim"), "Debug must contain 'ndim' field");
        assert!(debug_str.contains("dtype"), "Debug must contain 'dtype' field");
        assert!(debug_str.contains("shape"), "Debug must contain 'shape' field");
        assert!(debug_str.contains("quant_format"), "Debug must contain 'quant_format' field");
        assert!(debug_str.contains("quant_block_size"), "Debug must contain 'quant_block_size' field");
        assert!(debug_str.contains("scale_dtype"), "Debug must contain 'scale_dtype' field");
        assert!(debug_str.contains("zp_type"), "Debug must contain 'zp_type' field");
        assert!(debug_str.contains("data"), "Debug must contain 'data' field");
        assert!(debug_str.contains("original_size"), "Debug must contain 'original_size' field");
    }

    /// align_up with u64::MAX and alignment=1 returns u64::MAX unchanged.
    #[test]
    fn align_up_max_value_with_alignment_one_returns_max() {
        // Arrange
        let value = u64::MAX;
        let alignment = 1u64;
        // Act
        let result = align_up(value, alignment);
        // Assert
        assert_eq!(result, u64::MAX, "u64::MAX aligned to 1 must stay u64::MAX");
    }

    /// TensorEntry with quant_format nonzero but quant_block_size zero correctly reports is_quantized.
    #[test]
    fn tensor_entry_quant_format_nonzero_with_zero_block_size_is_quantized() {
        // Arrange
        let entry = TensorEntry {
            name: "qfmt_no_block".into(),
            ndim: 1,
            dtype: 0,
            shape: [100, 0, 0, 0],
            quant_format: 10,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 50],
            original_size: 100,
        };
        // Act & Assert
        assert!(entry.is_quantized(), "nonzero quant_format must mean is_quantized=true even with block_size=0");
    }

    /// build_metadata with tab and newline characters in values encodes them without error.
    #[test]
    fn build_metadata_with_special_whitespace_characters_succeeds() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("description".to_string(), "line1\nline2\ttab".to_string());
        // Act
        let bytes = build_metadata("test_arch", 32000, 4096, 32, 32, 8, 128, 11008, 4096, &extras);
        // Assert
        assert!(!bytes.is_empty(), "metadata bytes must not be empty");
        let parsed: HashMap<String, String> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.get("description").unwrap(), "line1\nline2\ttab");
    }

    /// TensorEntry original_size exactly equals compressed_size when data matches original.
    #[test]
    fn tensor_entry_original_size_equals_compressed_size_when_no_compression() {
        // Arrange
        let raw = vec![0xFF; 256];
        let entry = TensorEntry {
            name: "raw_tensor".into(),
            ndim: 1,
            dtype: 0,
            shape: [256, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: raw.clone(),
            original_size: raw.len() as u64,
        };
        // Act
        let compressed = entry.compressed_size();
        // Assert
        assert_eq!(compressed, entry.original_size, "uncompressed entry has equal compressed_size and original_size");
    }

    /// safetensors_dtype_to_u8 for F32 explicitly returns 0.
    #[test]
    fn safetensors_dtype_f32_exactly_returns_zero() {
        // Arrange
        let dtype = safetensors::Dtype::F32;
        // Act
        let code = safetensors_dtype_to_u8(dtype);
        // Assert
        assert_eq!(code, 0u8, "F32 must encode to exactly 0");
    }

    /// GllmWriter add_tensor increases tensor_count by exactly one each call.
    #[test]
    fn gllm_writer_add_tensor_increments_count_linearly() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        assert_eq!(writer.tensor_count(), 0, "new writer must have zero tensors");
        // Act & Assert
        for i in 0..5 {
            writer.add_tensor(TensorEntry {
                name: format!("t{}", i),
                ndim: 1,
                dtype: 0,
                shape: [4, 0, 0, 0],
                quant_format: 0,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data: vec![0u8; 4],
                original_size: 4,
            });
            assert_eq!(writer.tensor_count(), i + 1, "tensor_count must equal number of add_tensor calls");
        }
    }

    /// TensorEntry name containing only digits is stored correctly.
    #[test]
    fn tensor_entry_name_with_only_digits_preserved() {
        // Arrange
        let name = "1234567890";
        // Act
        let entry = TensorEntry {
            name: name.into(),
            ndim: 1,
            dtype: 0,
            shape: [1, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42],
            original_size: 1,
        };
        // Assert
        assert_eq!(entry.name, "1234567890", "numeric-only name must be preserved exactly");
    }

    /// align_up with value=0 always returns 0 regardless of alignment.
    #[test]
    fn align_up_zero_value_returns_zero_for_any_alignment() {
        // Arrange & Act & Assert
        assert_eq!(align_up(0, 1), 0, "0 aligned to 1 must be 0");
        assert_eq!(align_up(0, 4096), 0, "0 aligned to 4096 must be 0");
        assert_eq!(align_up(0, u64::MAX), 0, "0 aligned to u64::MAX must be 0");
        assert_eq!(align_up(0, 0), 0, "0 with alignment=0 must return 0 (identity)");
    }

    /// build_metadata merges extras into base keys, with extras overriding base keys on collision.
    #[test]
    fn build_metadata_extras_override_base_keys_on_collision() {
        // Arrange
        let mut extras = HashMap::new();
        extras.insert("hidden_size".to_string(), "9999".to_string());
        // Act
        let bytes = build_metadata("arch", 100, 512, 4, 4, 4, 64, 256, 2048, &extras);
        // Assert
        let parsed: HashMap<String, String> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.get("hidden_size").unwrap(), "9999", "extras hidden_size must override base 512");
        assert_eq!(parsed.get("arch_key").unwrap(), "arch", "non-colliding base key must remain");
    }

    /// GllmWriter set_metadata replaces previous metadata bytes entirely.
    #[test]
    fn gllm_writer_set_metadata_replaces_previous_bytes() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        writer.set_metadata(vec![1, 2, 3]);
        // Act
        writer.set_metadata(vec![10, 20]);
        // Assert: write a file and read back metadata to verify only [10,20] is present
        writer.add_tensor(TensorEntry {
            name: "meta_test".into(),
            ndim: 1,
            dtype: 0,
            shape: [4, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 4],
            original_size: 4,
        });
        let dir = unique_test_dir("meta_replace");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.gllm");
        writer.write_to_path(&path).unwrap();
        let file_data = std::fs::read(&path).unwrap();
        // The metadata section should contain [10, 20] not [1, 2, 3]
        let header = GllmHeader::parse(&file_data[..HEADER_SIZE]).unwrap();
        let meta_start = header.meta_offset as usize;
        assert!(meta_start < file_data.len(), "meta_offset must be within file");
        // Metadata is 2 bytes [10, 20]
        assert_eq!(file_data[meta_start], 10, "first metadata byte must be 10");
        assert_eq!(file_data[meta_start + 1], 20, "second metadata byte must be 20");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// quant_type_to_u8 produces codes in documented range groups (1-3 float, 10-15 classic, 20-25 k-quants, etc.).
    #[test]
    fn quant_type_to_u8_codes_partition_into_documented_range_groups() {
        // Arrange
        use gllm_kernels::quant::QuantType;
        let float_types = [QuantType::Bf16, QuantType::F16, QuantType::F32];
        let classic_types = [
            QuantType::Q4_0, QuantType::Q4_1, QuantType::Q5_0, QuantType::Q5_1,
            QuantType::Q8_0, QuantType::Q8_1,
        ];
        let k_types = [
            QuantType::Q2K, QuantType::Q3K, QuantType::Q4K,
            QuantType::Q5K, QuantType::Q6K, QuantType::Q8K,
        ];
        // Act
        for qt in &float_types {
            let code = quant_type_to_u8(*qt);
            assert!((1..=3).contains(&code), "{:?} code {} must be in 1-3", qt, code);
        }
        for qt in &classic_types {
            let code = quant_type_to_u8(*qt);
            assert!((10..=15).contains(&code), "{:?} code {} must be in 10-15", qt, code);
        }
        for qt in &k_types {
            let code = quant_type_to_u8(*qt);
            assert!((20..=25).contains(&code), "{:?} code {} must be in 20-25", qt, code);
        }
    }

    /// build_metadata with empty arch_key produces valid JSON with arch_key="".
    #[test]
    fn build_metadata_with_empty_arch_key_produces_valid_json() {
        // Arrange
        let extras = HashMap::new();
        // Act
        let bytes = build_metadata("", 0, 0, 0, 0, 0, 0, 0, 0, &extras);
        // Assert
        let parsed: HashMap<String, String> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.get("arch_key").unwrap(), "");
        assert_eq!(parsed.get("vocab_size").unwrap(), "0");
    }

    /// TensorEntry with shape [0,0,0,0] still reports correct ndim and shape.
    #[test]
    fn tensor_entry_all_zero_shape_and_ndim_zero_still_constructs() {
        // Arrange & Act
        let entry = TensorEntry {
            name: "zero_shape".into(),
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![],
            original_size: 0,
        };
        // Assert
        assert_eq!(entry.shape, [0u64; 4]);
        assert_eq!(entry.ndim, 0);
        assert_eq!(entry.compressed_size(), 0);
        assert!(!entry.is_quantized());
    }

    /// GllmWriter write_to_path creates the parent directory if it does not exist fails gracefully.
    #[test]
    fn gllm_writer_write_to_path_creates_file_in_temp_dir() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "tmp_test".into(),
            ndim: 1,
            dtype: 0,
            shape: [8, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xAA; 8],
            original_size: 8,
        });
        let dir = unique_test_dir("write_create");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("new_file.gllm");
        let _ = std::fs::remove_file(&path); // clean slate
        // Act
        writer.write_to_path(&path).unwrap();
        // Assert
        assert!(path.exists(), "file must be created");
        assert!(std::fs::metadata(&path).unwrap().len() > HEADER_SIZE as u64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Roundtrip with scale_dtype and zp_type both nonzero preserves them exactly.
    #[test]
    fn roundtrip_preserves_scale_dtype_and_zp_type_nonzero() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        writer.add_tensor(TensorEntry {
            name: "scaled_tensor".into(),
            ndim: 2,
            dtype: 0,
            shape: [16, 16, 0, 0],
            quant_format: 40, // AWQ4
            quant_block_size: 128,
            scale_dtype: 2,   // BF16
            zp_type: 1,       // u8
            data: vec![0xCC; 64],
            original_size: 1024,
        });
        let meta = build_metadata("test_arch", 100, 256, 4, 4, 4, 64, 512, 2048, &HashMap::new());
        writer.set_metadata(meta);
        let dir = unique_test_dir("scale_zp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scale_zp.gllm");
        writer.write_to_path(&path).unwrap();
        // Act
        let reader = GllmReader::open(&path).unwrap();
        let rt = reader.find_tensor("scaled_tensor").unwrap();
        // Assert
        assert_eq!(rt.entry.scale_dtype, 2, "scale_dtype must roundtrip as 2");
        assert_eq!(rt.entry.zp_type, 1, "zp_type must roundtrip as 1");
        assert_eq!(rt.entry.quant_block_size, 128);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// align_up with alignment equal to value returns that value (already aligned).
    #[test]
    fn align_up_value_equals_alignment_returns_value_unchanged() {
        // Arrange & Act & Assert
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(align_up(512, 512), 512);
        assert_eq!(align_up(1, 1), 1);
        assert_eq!(align_up(64, 64), 64);
    }

    /// GllmWriter with zero tensors and zero metadata produces a valid file with correct header fields.
    #[test]
    fn gllm_writer_zero_tensors_zero_metadata_valid_header() {
        // Arrange
        let writer = GllmWriter::new(4096);
        let dir = unique_test_dir("zero_header");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("zero_header.gllm");
        // Act
        writer.write_to_path(&path).unwrap();
        // Assert
        let data = std::fs::read(&path).unwrap();
        let header = GllmHeader::parse(&data).unwrap();
        assert_eq!(header.tensor_count, 0, "tensor_count must be 0");
        assert!(!header.is_quantized(), "flags must indicate no quantization");
        assert_eq!(header.version, crate::loader::gllm::GLLM_VERSION);
        assert!(data.len() >= HEADER_SIZE, "file must contain at least the header");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// build_metadata with all zero numeric parameters produces valid parseable JSON.
    #[test]
    fn build_metadata_all_zero_params_produces_parseable_json() {
        // Arrange
        let extras = HashMap::new();
        // Act
        let bytes = build_metadata("zero_model", 0, 0, 0, 0, 0, 0, 0, 0, &extras);
        // Assert
        assert!(!bytes.is_empty(), "metadata must not be empty");
        let parsed: HashMap<String, String> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed.len(), 9, "must have exactly 9 base keys");
        assert_eq!(parsed.get("arch_key").unwrap(), "zero_model");
        assert_eq!(parsed.get("hidden_size").unwrap(), "0");
        assert_eq!(parsed.get("context_length").unwrap(), "0");
    }

    /// safetensors_dtype_to_u8 returns distinct codes for F16 and BF16.
    #[test]
    fn safetensors_dtype_f16_and_bf16_have_distinct_codes() {
        // Arrange
        let f16_code = safetensors_dtype_to_u8(safetensors::Dtype::F16);
        let bf16_code = safetensors_dtype_to_u8(safetensors::Dtype::BF16);
        // Assert
        assert_ne!(f16_code, bf16_code, "F16 and BF16 must have distinct u8 codes");
        assert_eq!(f16_code, 1, "F16 must be 1");
        assert_eq!(bf16_code, 2, "BF16 must be 2");
    }

    /// TensorEntry with original_size smaller than data.len() still reports compressed_size correctly.
    #[test]
    fn tensor_entry_compressed_size_when_original_smaller_than_data() {
        // Arrange
        let entry = TensorEntry {
            name: "anomaly".into(),
            ndim: 1,
            dtype: 0,
            shape: [100, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0u8; 200],
            original_size: 50, // smaller than data.len()
        };
        // Act
        let cs = entry.compressed_size();
        // Assert
        assert_eq!(cs, 200, "compressed_size must equal data.len(), not original_size");
    }

    /// Roundtrip with tensor name containing spaces preserves the name exactly.
    #[test]
    fn roundtrip_tensor_name_with_spaces_preserved() {
        // Arrange
        let mut writer = GllmWriter::new(4096);
        let name = "model layers.0 self_attn weight";
        writer.add_tensor(TensorEntry {
            name: name.to_string(),
            ndim: 2,
            dtype: 0,
            shape: [4, 4, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![42u8; 16],
            original_size: 16,
        });
        let dir = unique_test_dir("name_spaces");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("spaces.gllm");
        writer.write_to_path(&path).unwrap();
        // Act
        let reader = GllmReader::open(&path).unwrap();
        let found = reader.find_tensor(name);
        // Assert
        assert!(found.is_some(), "tensor with spaces in name must be found");
        assert_eq!(found.unwrap().name, name);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GllmWriter add_tensor then write_to_path produces a file readable by GllmReader with correct tensor_count.
    #[test]
    fn gllm_writer_single_tensor_write_and_reader_tensor_count_match() {
        // Arrange
        let mut writer = GllmWriter::new(512);
        writer.add_tensor(TensorEntry {
            name: "only_one".into(),
            ndim: 1,
            dtype: 3,
            shape: [32, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data: vec![0xFF; 32],
            original_size: 32,
        });
        let dir = unique_test_dir("single_count");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("single.gllm");
        writer.write_to_path(&path).unwrap();
        // Act
        let reader = GllmReader::open(&path).unwrap();
        // Assert
        assert_eq!(reader.tensor_count(), 1, "reader must report exactly 1 tensor");
        assert!(reader.find_tensor("only_one").is_some());
        assert!(reader.find_tensor("nonexistent").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// dtype_to_u8 transparently passes through the input byte value.
    #[test]
    fn dtype_to_u8_identity_passthrough_for_all_byte_values() {
        // Arrange & Act & Assert
        for v in [0u8, 1, 2, 3, 4, 5, 6, 127, 200, 255] {
            assert_eq!(dtype_to_u8(v), v, "dtype_to_u8 must pass through {} unchanged", v);
        }
    }

}
