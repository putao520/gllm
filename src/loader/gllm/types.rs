//! .gllm 格式类型定义。
//!
//! SPEC: `SPEC/36-GLLM-WEIGHT-FORMAT.md §1`

use std::fmt;

use super::{GLLM_MAGIC, GLLM_VERSION, HEADER_SIZE, TENSOR_ENTRY_SIZE};

// ── Error ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GllmError {
    Io(std::io::Error),
    InvalidMagic(u32),
    UnsupportedVersion(u32),
    HeaderTooSmall(usize),
    TensorDirOutOfBounds { offset: usize, count: usize, file_size: usize },
    StringTableOutOfBounds { offset: usize, length: usize, file_size: usize },
    MetadataOutOfBounds { offset: usize, file_size: usize },
    TensorOutOfBounds { name: String, start: usize, end: usize, file_size: usize },
    DuplicateTensorName(String),
    ParseError(String),
    InvalidQuantType(u8),
    InvalidDType(u8),
    InvalidMetadata(String),
}

impl fmt::Display for GllmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidMagic(m) => write!(f, "invalid magic: 0x{m:08X} (expected 0x{GLLM_MAGIC:08X} \"GLLM\")"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version {v} (expected {GLLM_VERSION})"),
            Self::HeaderTooSmall(n) => write!(f, "file too small for header: {n} bytes (need {HEADER_SIZE})"),
            Self::TensorDirOutOfBounds { offset, count, file_size } => {
                let need = offset + count * TENSOR_ENTRY_SIZE;
                write!(f, "tensor directory [{offset}..{need}) exceeds file size {file_size}")
            }
            Self::StringTableOutOfBounds { offset, length, file_size } => {
                write!(f, "string table [{offset}..{}) exceeds file size {file_size}", offset + length)
            }
            Self::MetadataOutOfBounds { offset, file_size } => {
                write!(f, "metadata offset {offset} exceeds file size {file_size}")
            }
            Self::TensorOutOfBounds { name, start, end, file_size } => {
                write!(f, "tensor \"{name}\" [{start}..{end}) exceeds file size {file_size}")
            }
            Self::DuplicateTensorName(name) => write!(f, "duplicate tensor name: \"{name}\""),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::InvalidQuantType(v) => write!(f, "invalid quant_format: {v}"),
            Self::InvalidDType(v) => write!(f, "invalid dtype: {v}"),
            Self::InvalidMetadata(msg) => write!(f, "invalid metadata: {msg}"),
        }
    }
}

impl std::error::Error for GllmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GllmError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ── Header (64 bytes) ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GllmHeader {
    /// File format version.
    pub version: u32,
    /// Bit flags. Bit 0: global quantization flag.
    pub flags: u32,
    /// Offset to MessagePack metadata section.
    pub meta_offset: u64,
    /// Number of tensors in the Tensor Directory.
    pub tensor_count: u32,
    /// Offset to Tensor Directory (array of 64-byte entries).
    pub tensor_dir_offset: u64,
    /// Offset to tensor data region.
    pub data_offset: u64,
    /// Page alignment size (typically 4096).
    pub page_size: u32,
}

impl GllmHeader {
    /// Whether any tensor uses quantized format.
    pub fn is_quantized(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Parse header from raw bytes. Returns header and byte position after header.
    pub fn parse(bytes: &[u8]) -> Result<Self, GllmError> {
        if bytes.len() < HEADER_SIZE {
            return Err(GllmError::HeaderTooSmall(bytes.len()));
        }

        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != GLLM_MAGIC {
            return Err(GllmError::InvalidMagic(magic));
        }

        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != GLLM_VERSION {
            return Err(GllmError::UnsupportedVersion(version));
        }

        let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let meta_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let tensor_count = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let tensor_dir_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        let page_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());

        Ok(Self {
            version,
            flags,
            meta_offset,
            tensor_count,
            tensor_dir_offset,
            data_offset,
            page_size,
        })
    }
}

// ── Tensor Directory Entry (64 bytes) ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GllmTensorEntry {
    /// Offset into string table for tensor name.
    pub name_offset: u32,
    /// Length of tensor name.
    pub name_len: u16,
    /// Number of dimensions (1-4).
    pub ndim: u8,
    /// Element data type (DType enum value).
    pub dtype: u8,
    /// Shape per dimension (up to 4).
    pub shape: [u64; 4],
    /// Quantization format (QuantType enum value, 0 = none).
    pub quant_format: u8,
    /// Quantization block size (e.g. 128 for AWQ4).
    pub quant_block_size: u16,
    /// Scale factor data type.
    pub scale_dtype: u8,
    /// Zero-point type (0 = none, 1 = u8).
    pub zp_type: u8,
    /// Byte offset into data region.
    pub data_offset: u64,
    /// Compressed (quantized) size in bytes.
    pub compressed_size: u64,
    /// Original (FP32) size in bytes.
    pub original_size: u64,
}

impl GllmTensorEntry {
    /// Parse a single tensor directory entry at the given offset.
    pub fn parse_at(bytes: &[u8], offset: usize) -> Result<Self, GllmError> {
        let end = offset + TENSOR_ENTRY_SIZE;
        if end > bytes.len() {
            return Err(GllmError::ParseError(format!(
                "tensor entry at {offset} requires {TENSOR_ENTRY_SIZE} bytes, file has {}",
                bytes.len()
            )));
        }
        let b = &bytes[offset..end];

        let name_offset = u32::from_le_bytes(b[0..4].try_into().unwrap());
        let name_len = u16::from_le_bytes(b[4..6].try_into().unwrap());
        let ndim = b[6];
        let dtype = b[7];
        let shape = [
            u64::from_le_bytes(b[8..16].try_into().unwrap()),
            u64::from_le_bytes(b[16..24].try_into().unwrap()),
            u64::from_le_bytes(b[24..32].try_into().unwrap()),
            u64::from_le_bytes(b[32..40].try_into().unwrap()),
        ];
        let quant_format = b[40];
        let quant_block_size = u16::from_le_bytes(b[41..43].try_into().unwrap());
        let scale_dtype = b[43];
        let zp_type = b[44];
        // b[45..48] = reserved padding (3 bytes)
        let data_offset = u64::from_le_bytes(b[48..56].try_into().unwrap());
        let compressed_size = u64::from_le_bytes(b[56..64].try_into().unwrap());
        let original_size = u64::from_le_bytes(b[64..72].try_into().unwrap());

        Ok(Self {
            name_offset,
            name_len,
            ndim,
            dtype,
            shape,
            quant_format,
            quant_block_size,
            scale_dtype,
            zp_type,
            data_offset,
            compressed_size,
            original_size,
        })
    }

    /// Whether this tensor uses quantized storage.
    pub fn is_quantized(&self) -> bool {
        self.quant_format != 0
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size == 0 {
            return 1.0;
        }
        self.original_size as f64 / self.compressed_size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    // ── GllmError display ──

    #[test]
    fn error_display_variants() {
        let e = GllmError::InvalidMagic(0xDEAD);
        assert!(e.to_string().contains("0x0000DEAD"));
        let e = GllmError::UnsupportedVersion(99);
        assert!(e.to_string().contains("99"));
        let e = GllmError::HeaderTooSmall(10);
        assert!(e.to_string().contains("10"));
        let e = GllmError::DuplicateTensorName("foo".into());
        assert!(e.to_string().contains("foo"));
        let e = GllmError::InvalidQuantType(42);
        assert!(e.to_string().contains("42"));
        let e = GllmError::InvalidDType(7);
        assert!(e.to_string().contains("7"));
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof");
        let gllm_err: GllmError = io_err.into();
        assert!(matches!(gllm_err, GllmError::Io(_)));
    }

    #[test]
    fn error_source_io() {
        let e = GllmError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof"));
        assert!(e.source().is_some());
        let e = GllmError::InvalidMagic(0);
        assert!(e.source().is_none());
    }

    // ── GllmHeader ──

    #[test]
    fn header_parse_valid() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        buf[8..12].copy_from_slice(&1u32.to_le_bytes()); // flags (quantized)
        buf[12..20].copy_from_slice(&1024u64.to_le_bytes()); // meta_offset
        buf[20..24].copy_from_slice(&3u32.to_le_bytes()); // tensor_count
        buf[24..32].copy_from_slice(&256u64.to_le_bytes()); // tensor_dir_offset
        buf[32..40].copy_from_slice(&2048u64.to_le_bytes()); // data_offset
        buf[40..44].copy_from_slice(&4096u32.to_le_bytes()); // page_size

        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.version, 1);
        assert!(h.is_quantized());
        assert_eq!(h.tensor_count, 3);
        assert_eq!(h.page_size, 4096);
    }

    #[test]
    fn header_parse_too_small() {
        let buf = vec![0u8; 10];
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::HeaderTooSmall(10)));
    }

    #[test]
    fn header_parse_bad_magic() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&0xBADBADu32.to_le_bytes());
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::InvalidMagic(_)));
    }

    #[test]
    fn header_parse_bad_version() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&99u32.to_le_bytes());
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::UnsupportedVersion(99)));
    }

    #[test]
    fn header_is_quantized_flag() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        // flags = 0 → not quantized
        let h = GllmHeader::parse(&buf).unwrap();
        assert!(!h.is_quantized());

        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert!(h.is_quantized());
    }

    // ── GllmTensorEntry ──

    #[test]
    fn tensor_entry_parse() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&10u32.to_le_bytes()); // name_offset
        buf[4..6].copy_from_slice(&5u16.to_le_bytes()); // name_len
        buf[6] = 2; // ndim
        buf[7] = 1; // dtype
        buf[8..16].copy_from_slice(&4096u64.to_le_bytes()); // shape[0]
        buf[16..24].copy_from_slice(&768u64.to_le_bytes()); // shape[1]
        buf[40] = 3; // quant_format
        buf[41..43].copy_from_slice(&128u16.to_le_bytes()); // quant_block_size
        buf[43] = 1; // scale_dtype
        buf[44] = 1; // zp_type
        buf[48..56].copy_from_slice(&4096u64.to_le_bytes()); // data_offset
        buf[56..64].copy_from_slice(&1024u64.to_le_bytes()); // compressed_size
        buf[64..72].copy_from_slice(&4096u64.to_le_bytes()); // original_size

        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 10);
        assert_eq!(e.name_len, 5);
        assert_eq!(e.ndim, 2);
        assert_eq!(e.shape[0], 4096);
        assert_eq!(e.shape[1], 768);
        assert!(e.is_quantized());
        assert!((e.compression_ratio() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_entry_not_quantized() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE]; // quant_format = 0
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!(!e.is_quantized());
    }

    #[test]
    fn tensor_entry_compression_ratio_zero_size() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_parse_too_small() {
        let buf = vec![0u8; 10];
        let err = GllmTensorEntry::parse_at(&buf, 0).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    // ── GllmError Display: remaining variants ──

    #[test]
    fn error_display_io() {
        let e = GllmError::Io(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        let s = e.to_string();
        assert!(s.contains("IO error"));
        assert!(s.contains("short read"));
    }

    #[test]
    fn error_display_tensor_dir_out_of_bounds() {
        let e = GllmError::TensorDirOutOfBounds {
            offset: 100,
            count: 5,
            file_size: 200,
        };
        let s = e.to_string();
        assert!(s.contains("100"));
        assert!(s.contains("200"));
        assert!(s.contains("tensor directory"));
    }

    #[test]
    fn error_display_string_table_out_of_bounds() {
        let e = GllmError::StringTableOutOfBounds {
            offset: 300,
            length: 50,
            file_size: 200,
        };
        let s = e.to_string();
        assert!(s.contains("string table"));
        assert!(s.contains("300"));
        assert!(s.contains("200"));
    }

    #[test]
    fn error_display_metadata_out_of_bounds() {
        let e = GllmError::MetadataOutOfBounds {
            offset: 9999,
            file_size: 500,
        };
        let s = e.to_string();
        assert!(s.contains("9999"));
        assert!(s.contains("500"));
        assert!(s.contains("metadata"));
    }

    #[test]
    fn error_display_tensor_out_of_bounds() {
        let e = GllmError::TensorOutOfBounds {
            name: "layers.0.weight".into(),
            start: 100,
            end: 200,
            file_size: 150,
        };
        let s = e.to_string();
        assert!(s.contains("layers.0.weight"));
        assert!(s.contains("100"));
        assert!(s.contains("200"));
        assert!(s.contains("150"));
    }

    #[test]
    fn error_display_parse_error() {
        let e = GllmError::ParseError("bad data at offset 42".into());
        let s = e.to_string();
        assert!(s.contains("parse error"));
        assert!(s.contains("bad data at offset 42"));
    }

    #[test]
    fn error_display_invalid_metadata() {
        let e = GllmError::InvalidMetadata("missing required field".into());
        let s = e.to_string();
        assert!(s.contains("invalid metadata"));
        assert!(s.contains("missing required field"));
    }

    // ── GllmError Debug + Error trait ──

    #[test]
    fn error_debug_format() {
        let e = GllmError::InvalidMagic(0x1234);
        let debug_str = format!("{e:?}");
        assert!(debug_str.contains("InvalidMagic"));
    }

    #[test]
    fn error_source_non_io_is_none() {
        let variants: Vec<GllmError> = vec![
            GllmError::DuplicateTensorName("x".into()),
            GllmError::ParseError("msg".into()),
            GllmError::InvalidQuantType(1),
            GllmError::InvalidDType(2),
            GllmError::InvalidMetadata("m".into()),
            GllmError::HeaderTooSmall(5),
            GllmError::TensorDirOutOfBounds { offset: 0, count: 0, file_size: 0 },
            GllmError::StringTableOutOfBounds { offset: 0, length: 0, file_size: 0 },
            GllmError::MetadataOutOfBounds { offset: 0, file_size: 0 },
            GllmError::TensorOutOfBounds { name: "n".into(), start: 0, end: 0, file_size: 0 },
            GllmError::InvalidMagic(0),
            GllmError::UnsupportedVersion(0),
        ];
        for v in &variants {
            assert!(v.source().is_none(), "expected None source for {v:?}");
        }
    }

    // ── GllmHeader: field access and construction ──

    #[test]
    fn header_field_access() {
        let h = GllmHeader {
            version: 1,
            flags: 0,
            meta_offset: 4096,
            tensor_count: 7,
            tensor_dir_offset: 512,
            data_offset: 8192,
            page_size: 2048,
        };
        assert_eq!(h.version, 1);
        assert_eq!(h.flags, 0);
        assert_eq!(h.meta_offset, 4096);
        assert_eq!(h.tensor_count, 7);
        assert_eq!(h.tensor_dir_offset, 512);
        assert_eq!(h.data_offset, 8192);
        assert_eq!(h.page_size, 2048);
    }

    #[test]
    fn header_clone() {
        let h = GllmHeader {
            version: 1,
            flags: 3,
            meta_offset: 100,
            tensor_count: 2,
            tensor_dir_offset: 200,
            data_offset: 300,
            page_size: 4096,
        };
        let cloned = h.clone();
        assert_eq!(cloned.version, h.version);
        assert_eq!(cloned.flags, h.flags);
        assert_eq!(cloned.meta_offset, h.meta_offset);
        assert_eq!(cloned.tensor_count, h.tensor_count);
        assert_eq!(cloned.tensor_dir_offset, h.tensor_dir_offset);
        assert_eq!(cloned.data_offset, h.data_offset);
        assert_eq!(cloned.page_size, h.page_size);
    }

    #[test]
    fn header_debug_format() {
        let h = GllmHeader {
            version: 1,
            flags: 1,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: 0,
            page_size: 4096,
        };
        let debug = format!("{h:?}");
        assert!(debug.contains("GllmHeader"));
        assert!(debug.contains("version: 1"));
        assert!(debug.contains("flags: 1"));
        assert!(debug.contains("page_size: 4096"));
    }

    #[test]
    fn header_parse_exactly_header_size() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.version, GLLM_VERSION);
        assert_eq!(h.flags, 0);
        assert_eq!(h.meta_offset, 0);
        assert_eq!(h.tensor_count, 0);
        assert_eq!(h.tensor_dir_offset, 0);
        assert_eq!(h.data_offset, 0);
        assert_eq!(h.page_size, 0);
    }

    #[test]
    fn header_parse_larger_buffer_ignores_trailing() {
        let mut buf = vec![0u8; HEADER_SIZE + 100];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&2u32.to_le_bytes()); // flags
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.flags, 2);
        assert!(!h.is_quantized()); // bit 0 = 0
    }

    #[test]
    fn header_is_quantized_bit_flags() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        // flags = 2 → bit 0 is 0 → not quantized, but other bits set
        buf[8..12].copy_from_slice(&2u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert!(!h.is_quantized());
        // flags = 3 → bit 0 is 1 → quantized
        buf[8..12].copy_from_slice(&3u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert!(h.is_quantized());
    }

    #[test]
    fn header_parse_max_values() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        buf[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        buf[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.flags, u32::MAX);
        assert_eq!(h.meta_offset, u64::MAX);
        assert_eq!(h.tensor_count, u32::MAX);
        assert_eq!(h.tensor_dir_offset, u64::MAX);
        assert_eq!(h.data_offset, u64::MAX);
        assert_eq!(h.page_size, u32::MAX);
        assert!(h.is_quantized());
    }

    // ── GllmTensorEntry: additional coverage ──

    #[test]
    fn tensor_entry_parse_at_nonzero_offset() {
        let offset = 16;
        let mut buf = vec![0u8; offset + TENSOR_ENTRY_SIZE];
        let start = offset;
        buf[start..start + 4].copy_from_slice(&42u32.to_le_bytes()); // name_offset
        buf[start + 4..start + 6].copy_from_slice(&8u16.to_le_bytes()); // name_len
        buf[start + 6] = 3; // ndim
        buf[start + 7] = 2; // dtype
        buf[start + 40] = 5; // quant_format
        buf[start + 41..start + 43].copy_from_slice(&256u16.to_le_bytes()); // quant_block_size
        buf[start + 43] = 3; // scale_dtype
        buf[start + 44] = 2; // zp_type
        buf[start + 48..start + 56].copy_from_slice(&1234u64.to_le_bytes()); // data_offset
        buf[start + 56..start + 64].copy_from_slice(&500u64.to_le_bytes()); // compressed_size
        buf[start + 64..start + 72].copy_from_slice(&2000u64.to_le_bytes()); // original_size

        let e = GllmTensorEntry::parse_at(&buf, offset).unwrap();
        assert_eq!(e.name_offset, 42);
        assert_eq!(e.name_len, 8);
        assert_eq!(e.ndim, 3);
        assert_eq!(e.dtype, 2);
        assert_eq!(e.quant_format, 5);
        assert_eq!(e.quant_block_size, 256);
        assert_eq!(e.scale_dtype, 3);
        assert_eq!(e.zp_type, 2);
        assert_eq!(e.data_offset, 1234);
        assert_eq!(e.compressed_size, 500);
        assert_eq!(e.original_size, 2000);
        assert!(e.is_quantized());
    }

    #[test]
    fn tensor_entry_parse_at_boundary_exactly_fits() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&1u32.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_offset, 1);
    }

    #[test]
    fn tensor_entry_parse_at_just_past_boundary() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE + 1];
        // Parsing at offset 2 needs TENSOR_ENTRY_SIZE bytes from byte 2,
        // but buffer only has TENSOR_ENTRY_SIZE - 1 bytes after that.
        let err = GllmTensorEntry::parse_at(&buf, 2).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    #[test]
    fn tensor_entry_shape_all_dimensions() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 4; // ndim = 4
        buf[8..16].copy_from_slice(&10u64.to_le_bytes());    // shape[0]
        buf[16..24].copy_from_slice(&20u64.to_le_bytes());   // shape[1]
        buf[24..32].copy_from_slice(&30u64.to_le_bytes());   // shape[2]
        buf[32..40].copy_from_slice(&40u64.to_le_bytes());   // shape[3]
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, 4);
        assert_eq!(e.shape, [10, 20, 30, 40]);
    }

    #[test]
    fn tensor_entry_clone() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 2;
        buf[40] = 7;
        buf[8..16].copy_from_slice(&128u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        let cloned = e.clone();
        assert_eq!(cloned.name_offset, e.name_offset);
        assert_eq!(cloned.name_len, e.name_len);
        assert_eq!(cloned.ndim, e.ndim);
        assert_eq!(cloned.dtype, e.dtype);
        assert_eq!(cloned.shape, e.shape);
        assert_eq!(cloned.quant_format, e.quant_format);
        assert_eq!(cloned.quant_block_size, e.quant_block_size);
        assert_eq!(cloned.scale_dtype, e.scale_dtype);
        assert_eq!(cloned.zp_type, e.zp_type);
        assert_eq!(cloned.data_offset, e.data_offset);
        assert_eq!(cloned.compressed_size, e.compressed_size);
        assert_eq!(cloned.original_size, e.original_size);
    }

    #[test]
    fn tensor_entry_debug_format() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 1;
        buf[40] = 3;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        let debug = format!("{e:?}");
        assert!(debug.contains("GllmTensorEntry"));
        assert!(debug.contains("ndim: 1"));
        assert!(debug.contains("quant_format: 3"));
    }

    #[test]
    fn tensor_entry_compression_ratio_non_trivial() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[56..64].copy_from_slice(&256u64.to_le_bytes()); // compressed_size
        buf[64..72].copy_from_slice(&1024u64.to_le_bytes()); // original_size
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert!((e.compression_ratio() - 4.0).abs() < 1e-6);
    }

    // ── Constants ──

    #[test]
    fn constants_are_consistent() {
        assert_eq!(GLLM_MAGIC, 0x4D4C4C47);
        assert_eq!(GLLM_VERSION, 1);
        assert_eq!(HEADER_SIZE, 64);
        assert_eq!(TENSOR_ENTRY_SIZE, 72);
    }

    #[test]
    fn magic_bytes_are_gllm_ascii() {
        let bytes = GLLM_MAGIC.to_le_bytes();
        assert_eq!(bytes[0], b'G');
        assert_eq!(bytes[1], b'L');
        assert_eq!(bytes[2], b'L');
        assert_eq!(bytes[3], b'M');
    }

    // ── GllmError From<io::Error> round-trip ──

    #[test]
    fn error_from_io_preserves_kind() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let gllm_err: GllmError = io_err.into();
        if let GllmError::Io(ref inner) = gllm_err {
            assert_eq!(inner.kind(), std::io::ErrorKind::PermissionDenied);
        } else {
            panic!("expected Io variant");
        }
    }

    #[test]
    fn error_source_returns_inner_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke");
        let gllm_err = GllmError::Io(io_err);
        let source = gllm_err.source().expect("Io variant should have source");
        assert!(source.downcast_ref::<std::io::Error>().is_some());
    }

    // ── GllmHeader: additional edge cases ──

    #[test]
    fn header_parse_empty_buffer() {
        let buf: &[u8] = &[];
        let err = GllmHeader::parse(buf).unwrap_err();
        assert!(matches!(err, GllmError::HeaderTooSmall(0)));
    }

    #[test]
    fn header_parse_one_byte() {
        let buf = vec![0u8; 1];
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::HeaderTooSmall(1)));
    }

    #[test]
    fn header_parse_header_size_minus_one() {
        let buf = vec![0u8; HEADER_SIZE - 1];
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::HeaderTooSmall(n) if n == HEADER_SIZE - 1));
    }

    #[test]
    fn header_is_quantized_flags_max() {
        let h = GllmHeader {
            version: 1,
            flags: u32::MAX,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: 0,
            page_size: 0,
        };
        assert!(h.is_quantized());
    }

    #[test]
    fn header_is_quantized_flags_zero() {
        let h = GllmHeader {
            version: 1,
            flags: 0,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: 0,
            page_size: 0,
        };
        assert!(!h.is_quantized());
    }

    // ── GllmTensorEntry: direct construction and edge cases ──

    #[test]
    fn tensor_entry_direct_construction() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 0,
            dtype: 0,
            shape: [0u64; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 0,
            original_size: 0,
        };
        assert_eq!(entry.name_offset, 0);
        assert_eq!(entry.name_len, 0);
        assert_eq!(entry.ndim, 0);
        assert_eq!(entry.dtype, 0);
        assert_eq!(entry.shape, [0u64; 4]);
        assert!(!entry.is_quantized());
    }

    #[test]
    fn tensor_entry_direct_construction_max_values() {
        let entry = GllmTensorEntry {
            name_offset: u32::MAX,
            name_len: u16::MAX,
            ndim: u8::MAX,
            dtype: u8::MAX,
            shape: [u64::MAX; 4],
            quant_format: u8::MAX,
            quant_block_size: u16::MAX,
            scale_dtype: u8::MAX,
            zp_type: u8::MAX,
            data_offset: u64::MAX,
            compressed_size: u64::MAX,
            original_size: u64::MAX,
        };
        assert_eq!(entry.name_offset, u32::MAX);
        assert_eq!(entry.name_len, u16::MAX);
        assert_eq!(entry.ndim, u8::MAX);
        assert_eq!(entry.shape, [u64::MAX; 4]);
        assert!(entry.is_quantized());
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_equal_sizes() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [1024, 0, 0, 0],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 4096,
            original_size: 4096,
        };
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_compression_ratio_high() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [1024, 0, 0, 0],
            quant_format: 1,
            quant_block_size: 128,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 1,
            original_size: 1024,
        };
        assert!((entry.compression_ratio() - 1024.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_entry_parse_empty_buffer() {
        let buf: &[u8] = &[];
        let err = GllmTensorEntry::parse_at(buf, 0).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    #[test]
    fn tensor_entry_parse_at_offset_equals_length() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];
        let err = GllmTensorEntry::parse_at(&buf, TENSOR_ENTRY_SIZE).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    #[test]
    fn tensor_entry_parse_at_one_past_last_valid_offset() {
        // last valid offset = buf.len() - TENSOR_ENTRY_SIZE
        let buf = vec![0u8; TENSOR_ENTRY_SIZE * 2];
        let last_valid = buf.len() - TENSOR_ENTRY_SIZE;
        assert!(GllmTensorEntry::parse_at(&buf, last_valid).is_ok());
        let err = GllmTensorEntry::parse_at(&buf, last_valid + 1).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    #[test]
    fn tensor_entry_is_quantized_with_nonzero_format() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [100, 0, 0, 0],
            quant_format: 1,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 0,
            original_size: 0,
        };
        assert!(entry.is_quantized());
    }

    #[test]
    fn tensor_entry_clone_equality() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[0..4].copy_from_slice(&99u32.to_le_bytes());
        buf[6] = 2;
        buf[7] = 3;
        buf[40] = 4;
        buf[41..43].copy_from_slice(&64u16.to_le_bytes());
        buf[48..56].copy_from_slice(&500u64.to_le_bytes());
        buf[56..64].copy_from_slice(&100u64.to_le_bytes());
        buf[64..72].copy_from_slice(&400u64.to_le_bytes());

        let original = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        let cloned = original.clone();

        assert_eq!(original.name_offset, cloned.name_offset);
        assert_eq!(original.name_len, cloned.name_len);
        assert_eq!(original.ndim, cloned.ndim);
        assert_eq!(original.dtype, cloned.dtype);
        assert_eq!(original.shape, cloned.shape);
        assert_eq!(original.quant_format, cloned.quant_format);
        assert_eq!(original.quant_block_size, cloned.quant_block_size);
        assert_eq!(original.scale_dtype, cloned.scale_dtype);
        assert_eq!(original.zp_type, cloned.zp_type);
        assert_eq!(original.data_offset, cloned.data_offset);
        assert_eq!(original.compressed_size, cloned.compressed_size);
        assert_eq!(original.original_size, cloned.original_size);
    }

    #[test]
    fn header_clone_independence() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        buf[20..24].copy_from_slice(&5u32.to_le_bytes());

        let original = GllmHeader::parse(&buf).unwrap();
        let mut cloned = original.clone();

        // Modify clone, original should be unchanged
        cloned.tensor_count = 999;
        assert_eq!(original.tensor_count, 5);
        assert_eq!(cloned.tensor_count, 999);
    }

    // ── GllmHeader: byte-range verification ──

    #[test]
    fn header_parse_meta_offset_byte_range() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[12..20].copy_from_slice(&0x0100_0000_0000_0002u64.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.meta_offset, 0x0100_0000_0000_0002u64);
    }

    #[test]
    fn header_parse_tensor_dir_offset_byte_range() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[24..32].copy_from_slice(&0xABCD_EF01_2345_6789u64.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.tensor_dir_offset, 0xABCD_EF01_2345_6789u64);
    }

    #[test]
    fn header_parse_data_offset_byte_range() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[32..40].copy_from_slice(&0x1000u64.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.data_offset, 0x1000);
    }

    #[test]
    fn header_parse_page_size_byte_range() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[40..44].copy_from_slice(&8192u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.page_size, 8192);
    }

    #[test]
    fn header_parse_reserved_bytes_ignored() {
        let mut buf = vec![0xFFu8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&0u32.to_le_bytes());
        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.version, GLLM_VERSION);
        assert_eq!(h.flags, 0);
    }

    #[test]
    fn header_mutation_after_construction() {
        let mut h = GllmHeader {
            version: 1,
            flags: 0,
            meta_offset: 0,
            tensor_count: 0,
            tensor_dir_offset: 0,
            data_offset: 0,
            page_size: 0,
        };
        h.tensor_count = 42;
        h.data_offset = 9999;
        assert_eq!(h.tensor_count, 42);
        assert_eq!(h.data_offset, 9999);
    }

    #[test]
    fn header_is_quantized_odd_flags() {
        for flags in [1u32, 3, 5, 255, u32::MAX] {
            let h = GllmHeader {
                version: 1,
                flags,
                meta_offset: 0,
                tensor_count: 0,
                tensor_dir_offset: 0,
                data_offset: 0,
                page_size: 0,
            };
            assert!(h.is_quantized(), "flags={flags} should be quantized");
        }
    }

    #[test]
    fn header_is_quantized_even_flags() {
        for flags in [0u32, 2, 4, 6, 254, u32::MAX - 1] {
            let h = GllmHeader {
                version: 1,
                flags,
                meta_offset: 0,
                tensor_count: 0,
                tensor_dir_offset: 0,
                data_offset: 0,
                page_size: 0,
            };
            assert!(!h.is_quantized(), "flags={flags} should not be quantized");
        }
    }

    #[test]
    fn header_parse_all_fields_nontrivial() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&GLLM_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&1u32.to_le_bytes());
        buf[12..20].copy_from_slice(&100u64.to_le_bytes());
        buf[20..24].copy_from_slice(&10u32.to_le_bytes());
        buf[24..32].copy_from_slice(&200u64.to_le_bytes());
        buf[32..40].copy_from_slice(&300u64.to_le_bytes());
        buf[40..44].copy_from_slice(&2048u32.to_le_bytes());

        let h = GllmHeader::parse(&buf).unwrap();
        assert_eq!(h.flags, 1);
        assert_eq!(h.meta_offset, 100);
        assert_eq!(h.tensor_count, 10);
        assert_eq!(h.tensor_dir_offset, 200);
        assert_eq!(h.data_offset, 300);
        assert_eq!(h.page_size, 2048);
        assert!(h.is_quantized());
    }

    #[test]
    fn header_default_like_construction() {
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

    // ── GllmTensorEntry: additional field and boundary coverage ──

    #[test]
    fn tensor_entry_shape_dims_independent() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = 4;
        buf[8..16].copy_from_slice(&11u64.to_le_bytes());
        buf[16..24].copy_from_slice(&22u64.to_le_bytes());
        buf[24..32].copy_from_slice(&33u64.to_le_bytes());
        buf[32..40].copy_from_slice(&44u64.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.shape[0], 11);
        assert_eq!(e.shape[1], 22);
        assert_eq!(e.shape[2], 33);
        assert_eq!(e.shape[3], 44);
    }

    #[test]
    fn tensor_entry_ndim_zero() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, 0);
    }

    #[test]
    fn tensor_entry_ndim_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[6] = u8::MAX;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.ndim, u8::MAX);
    }

    #[test]
    fn tensor_entry_dtype_field_access() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[7] = 42;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.dtype, 42);
    }

    #[test]
    fn tensor_entry_quant_fields_all_set() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[40] = 7;
        buf[41..43].copy_from_slice(&512u16.to_le_bytes());
        buf[43] = 2;
        buf[44] = 1;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.quant_format, 7);
        assert_eq!(e.quant_block_size, 512);
        assert_eq!(e.scale_dtype, 2);
        assert_eq!(e.zp_type, 1);
        assert!(e.is_quantized());
    }

    #[test]
    fn tensor_entry_compression_ratio_original_zero_compressed_nonzero() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 100,
            original_size: 0,
        };
        assert_eq!(entry.compression_ratio(), 0.0);
    }

    #[test]
    fn tensor_entry_compression_ratio_both_nonzero_large() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 1,
            original_size: 1_000_000,
        };
        let ratio = entry.compression_ratio();
        assert!((ratio - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn tensor_entry_parse_at_large_offset() {
        let offset = 1024 * 1024;
        let mut buf = vec![0u8; offset + TENSOR_ENTRY_SIZE];
        buf[offset..offset + 4].copy_from_slice(&777u32.to_le_bytes());
        buf[offset + 6] = 2;
        buf[offset + 7] = 5;
        buf[offset + 40] = 3;
        buf[offset + 48..offset + 56].copy_from_slice(&9999u64.to_le_bytes());

        let e = GllmTensorEntry::parse_at(&buf, offset).unwrap();
        assert_eq!(e.name_offset, 777);
        assert_eq!(e.ndim, 2);
        assert_eq!(e.dtype, 5);
        assert_eq!(e.quant_format, 3);
        assert_eq!(e.data_offset, 9999);
    }

    #[test]
    fn tensor_entry_parse_multiple_entries_same_buffer() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE * 3];
        buf[0..4].copy_from_slice(&10u32.to_le_bytes());
        buf[6] = 1;
        let off1 = TENSOR_ENTRY_SIZE;
        buf[off1..off1 + 4].copy_from_slice(&20u32.to_le_bytes());
        buf[off1 + 6] = 2;
        let off2 = TENSOR_ENTRY_SIZE * 2;
        buf[off2..off2 + 4].copy_from_slice(&30u32.to_le_bytes());
        buf[off2 + 6] = 3;

        let e0 = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        let e1 = GllmTensorEntry::parse_at(&buf, off1).unwrap();
        let e2 = GllmTensorEntry::parse_at(&buf, off2).unwrap();
        assert_eq!(e0.name_offset, 10);
        assert_eq!(e0.ndim, 1);
        assert_eq!(e1.name_offset, 20);
        assert_eq!(e1.ndim, 2);
        assert_eq!(e2.name_offset, 30);
        assert_eq!(e2.ndim, 3);
    }

    #[test]
    fn tensor_entry_name_len_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[4..6].copy_from_slice(&u16::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.name_len, u16::MAX);
    }

    #[test]
    fn tensor_entry_quant_block_size_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[41..43].copy_from_slice(&u16::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.quant_block_size, u16::MAX);
    }

    #[test]
    fn tensor_entry_data_offset_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.data_offset, u64::MAX);
    }

    #[test]
    fn tensor_entry_compressed_size_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.compressed_size, u64::MAX);
    }

    #[test]
    fn tensor_entry_original_size_max() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[64..72].copy_from_slice(&u64::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.original_size, u64::MAX);
    }

    #[test]
    fn tensor_entry_shape_max_all_dims() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        buf[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.shape, [u64::MAX; 4]);
    }

    #[test]
    fn tensor_entry_mutation_after_construction() {
        let mut entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 0,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 0,
            original_size: 0,
        };
        entry.quant_format = 5;
        entry.data_offset = 12345;
        entry.shape[0] = 768;
        assert!(entry.is_quantized());
        assert_eq!(entry.data_offset, 12345);
        assert_eq!(entry.shape[0], 768);
    }

    #[test]
    fn tensor_entry_clone_independence() {
        let mut entry = GllmTensorEntry {
            name_offset: 10,
            name_len: 5,
            ndim: 2,
            dtype: 1,
            shape: [4096, 768, 0, 0],
            quant_format: 3,
            quant_block_size: 128,
            scale_dtype: 1,
            zp_type: 1,
            data_offset: 5000,
            compressed_size: 1024,
            original_size: 4096,
        };
        let cloned = entry.clone();
        entry.name_offset = 999;
        entry.shape[0] = 0;
        assert_eq!(cloned.name_offset, 10);
        assert_eq!(cloned.shape[0], 4096);
    }

    #[test]
    fn tensor_entry_is_quantized_various_formats() {
        for fmt in 1u8..=5 {
            let entry = GllmTensorEntry {
                name_offset: 0,
                name_len: 0,
                ndim: 1,
                dtype: 0,
                shape: [0; 4],
                quant_format: fmt,
                quant_block_size: 0,
                scale_dtype: 0,
                zp_type: 0,
                data_offset: 0,
                compressed_size: 0,
                original_size: 0,
            };
            assert!(entry.is_quantized(), "quant_format={fmt} should be quantized");
        }
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn tensor_entry_parse_at_usize_max_offset_panics() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE];
        // usize::MAX + TENSOR_ENTRY_SIZE overflows, causing a panic in the addition.
        let _ = GllmTensorEntry::parse_at(&buf, usize::MAX);
    }

    #[test]
    fn tensor_entry_parse_preserves_all_quant_fields() {
        let mut buf = vec![0u8; TENSOR_ENTRY_SIZE];
        buf[40] = 9;
        buf[41..43].copy_from_slice(&1024u16.to_le_bytes());
        buf[43] = 4;
        buf[44] = 2;
        let e = GllmTensorEntry::parse_at(&buf, 0).unwrap();
        assert_eq!(e.quant_format, 9);
        assert_eq!(e.quant_block_size, 1024);
        assert_eq!(e.scale_dtype, 4);
        assert_eq!(e.zp_type, 2);
    }

    #[test]
    fn tensor_entry_compression_ratio_range() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 500,
            original_size: 1000,
        };
        let ratio = entry.compression_ratio();
        assert!(ratio > 0.0);
        assert!(ratio.is_finite());
    }

    #[test]
    fn tensor_entry_compression_ratio_near_zero_original() {
        let entry = GllmTensorEntry {
            name_offset: 0,
            name_len: 0,
            ndim: 1,
            dtype: 0,
            shape: [0; 4],
            quant_format: 0,
            quant_block_size: 0,
            scale_dtype: 0,
            zp_type: 0,
            data_offset: 0,
            compressed_size: 1,
            original_size: 1,
        };
        assert!((entry.compression_ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_entry_parse_at_offset_one_before_valid() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE + 1];
        let result = GllmTensorEntry::parse_at(&buf, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn tensor_entry_parse_at_offset_two_before_valid_fails() {
        let buf = vec![0u8; TENSOR_ENTRY_SIZE + 1];
        let err = GllmTensorEntry::parse_at(&buf, 2).unwrap_err();
        assert!(matches!(err, GllmError::ParseError(_)));
    }

    // ── GllmError: additional display and computed range coverage ──

    #[test]
    fn error_display_tensor_dir_out_of_bounds_computed_end() {
        let e = GllmError::TensorDirOutOfBounds {
            offset: 100,
            count: 5,
            file_size: 200,
        };
        let s = e.to_string();
        assert!(s.contains("460"), "message should contain computed end 460: {s}");
    }

    #[test]
    fn error_display_string_table_out_of_bounds_computed_end() {
        let e = GllmError::StringTableOutOfBounds {
            offset: 50,
            length: 100,
            file_size: 120,
        };
        let s = e.to_string();
        assert!(s.contains("150"), "message should contain computed end 150: {s}");
    }

    #[test]
    fn error_display_duplicate_tensor_name() {
        let e = GllmError::DuplicateTensorName("model.embed_tokens.weight".into());
        let s = e.to_string();
        assert!(s.contains("duplicate tensor name"));
        assert!(s.contains("model.embed_tokens.weight"));
    }

    #[test]
    fn error_display_invalid_quant_type_message() {
        let e = GllmError::InvalidQuantType(99);
        let s = e.to_string();
        assert!(s.contains("99"));
        assert!(s.contains("quant_format"));
    }

    #[test]
    fn error_display_invalid_dtype_message() {
        let e = GllmError::InvalidDType(13);
        let s = e.to_string();
        assert!(s.contains("13"));
        assert!(s.contains("dtype"));
    }

    #[test]
    fn error_display_unsupported_version_message() {
        let e = GllmError::UnsupportedVersion(42);
        let s = e.to_string();
        assert!(s.contains("42"));
        assert!(s.contains("unsupported version"));
    }

    #[test]
    fn error_display_header_too_small_message() {
        let e = GllmError::HeaderTooSmall(7);
        let s = e.to_string();
        assert!(s.contains("7"));
        assert!(s.contains("header"));
    }

    #[test]
    fn error_display_invalid_magic_hex_format() {
        let e = GllmError::InvalidMagic(0xAB_CD);
        let s = e.to_string();
        assert!(s.contains("0x0000ABCD"));
        assert!(s.contains("expected"));
        assert!(s.contains("GLLM"));
    }

    #[test]
    fn error_from_io_multiple_kinds() {
        for kind in [
            std::io::ErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied,
            std::io::ErrorKind::AlreadyExists,
            std::io::ErrorKind::InvalidInput,
            std::io::ErrorKind::TimedOut,
        ] {
            let io_err = std::io::Error::new(kind, "test");
            let gllm_err: GllmError = io_err.into();
            assert!(
                matches!(gllm_err, GllmError::Io(ref e) if e.kind() == kind),
                "expected Io with kind {:?}",
                kind
            );
        }
    }

    #[test]
    fn error_source_only_io_has_source() {
        let io_variant = GllmError::Io(std::io::Error::new(std::io::ErrorKind::WriteZero, "w"));
        assert!(io_variant.source().is_some());

        let non_io = GllmError::InvalidMagic(0);
        assert!(non_io.source().is_none());
    }

    #[test]
    fn error_debug_all_variants_have_name() {
        let variants: Vec<GllmError> = vec![
            GllmError::Io(std::io::Error::new(std::io::ErrorKind::Other, "x")),
            GllmError::InvalidMagic(0),
            GllmError::UnsupportedVersion(0),
            GllmError::HeaderTooSmall(0),
            GllmError::TensorDirOutOfBounds {
                offset: 0,
                count: 0,
                file_size: 0,
            },
            GllmError::StringTableOutOfBounds {
                offset: 0,
                length: 0,
                file_size: 0,
            },
            GllmError::MetadataOutOfBounds {
                offset: 0,
                file_size: 0,
            },
            GllmError::TensorOutOfBounds {
                name: "n".into(),
                start: 0,
                end: 0,
                file_size: 0,
            },
            GllmError::DuplicateTensorName("x".into()),
            GllmError::ParseError("x".into()),
            GllmError::InvalidQuantType(0),
            GllmError::InvalidDType(0),
            GllmError::InvalidMetadata("x".into()),
        ];
        let variant_names = [
            "Io",
            "InvalidMagic",
            "UnsupportedVersion",
            "HeaderTooSmall",
            "TensorDirOutOfBounds",
            "StringTableOutOfBounds",
            "MetadataOutOfBounds",
            "TensorOutOfBounds",
            "DuplicateTensorName",
            "ParseError",
            "InvalidQuantType",
            "InvalidDType",
            "InvalidMetadata",
        ];
        for (v, name) in variants.iter().zip(variant_names.iter()) {
            let debug = format!("{v:?}");
            assert!(
                debug.contains(name),
                "debug for {v:?} should contain {name}"
            );
        }
    }

    // ── Constants: byte-level layout verification ──

    #[test]
    fn header_size_is_power_of_two() {
        assert!(
            HEADER_SIZE.is_power_of_two(),
            "HEADER_SIZE should be a power of 2"
        );
    }

    #[test]
    fn tensor_entry_size_is_aligned() {
        assert_eq!(
            TENSOR_ENTRY_SIZE % 8,
            0,
            "TENSOR_ENTRY_SIZE should be 8-byte aligned"
        );
    }

    #[test]
    fn tensor_entry_size_layout_breakdown() {
        let expected = 4 + 2 + 1 + 1 + 32 + 1 + 2 + 1 + 1 + 3 + 8 + 8 + 8;
        assert_eq!(TENSOR_ENTRY_SIZE, expected);
    }

    #[test]
    fn header_parse_rejects_magic_with_extra_byte() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        buf[3] = b'N';
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::InvalidMagic(_)));
    }

    #[test]
    fn header_parse_rejects_zero_magic() {
        let buf = vec![0u8; HEADER_SIZE];
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::InvalidMagic(0)));
    }

    #[test]
    fn header_parse_rejects_version_zero() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&GLLM_MAGIC.to_le_bytes());
        let err = GllmHeader::parse(&buf).unwrap_err();
        assert!(matches!(err, GllmError::UnsupportedVersion(0)));
    }
}
