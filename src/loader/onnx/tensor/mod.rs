use prost::bytes::Bytes;
use safetensors::Dtype;

use super::external::ExternalDataResolver;
use super::{pack, proto, LoaderError, Result, TensorSlice};

mod parse;

#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// True if this tensor contains STRING data (dtype will be U8 as placeholder)
    pub is_string: bool,
    data: Bytes,
}

#[derive(Debug, Clone)]
pub struct OnnxSparseTensor {
    pub values: OnnxTensor,
    pub indices: OnnxTensor,
    pub dims: Vec<usize>,
    pub format: OnnxSparseFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxSparseFormat {
    Coo,
    Csr,
    Csc,
}

impl OnnxTensor {
    pub(super) fn from_initializer(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        Self::from_proto(proto, resolver, NamePolicy::Require)
    }

    pub(super) fn from_attribute(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
        fallback: &str,
    ) -> Result<Self> {
        Self::from_proto(proto, resolver, NamePolicy::Fallback(fallback))
    }

    pub(super) fn slice(&self) -> TensorSlice<'_> {
        TensorSlice {
            dtype: self.dtype,
            shape: self.shape.clone(),
            data: self.data.as_ref(),
        }
    }

    pub(crate) fn scalar_f32(&self) -> Option<f32> {
        if self.element_count() != 1 {
            return None;
        }
        let bytes = self.data.as_ref();
        match self.dtype {
            Dtype::F32 => parse::slice_to_f32(bytes.get(0..4)?),
            Dtype::F16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::f16::from_bits(bits).to_f32()),
            Dtype::BF16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::bf16::from_bits(bits).to_f32()),
            Dtype::I32 => parse::slice_to_i32(bytes.get(0..4)?).map(|value| value as f32),
            Dtype::I64 => parse::slice_to_i64(bytes.get(0..8)?).map(|value| value as f32),
            Dtype::U8 => bytes.first().map(|b| *b as f32),
            Dtype::U16 => parse::slice_to_u16(bytes.get(0..2)?).map(|value| value as f32),
            Dtype::U32 => parse::slice_to_u32(bytes.get(0..4)?).map(|value| value as f32),
            Dtype::U64 => parse::slice_to_u64(bytes.get(0..8)?).map(|value| value as f32),
            _ => None,
        }
    }

    pub(crate) fn scalar_i64(&self) -> Option<i64> {
        if self.element_count() != 1 {
            return None;
        }
        let bytes = self.data.as_ref();
        match self.dtype {
            Dtype::I64 => parse::slice_to_i64(bytes.get(0..8)?),
            Dtype::I32 => parse::slice_to_i32(bytes.get(0..4)?).map(i64::from),
            Dtype::U8 => bytes.first().map(|v| i64::from(*v)),
            Dtype::U16 => parse::slice_to_u16(bytes.get(0..2)?).map(i64::from),
            Dtype::U32 => parse::slice_to_u32(bytes.get(0..4)?).map(i64::from),
            Dtype::U64 => parse::slice_to_u64(bytes.get(0..8)?).map(|v| v as i64),
            Dtype::F32 => parse::slice_to_f32(bytes.get(0..4)?).map(|v| v as i64),
            Dtype::F16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::f16::from_bits(bits).to_f32() as i64),
            Dtype::BF16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::bf16::from_bits(bits).to_f32() as i64),
            _ => None,
        }
    }

    pub fn raw_data(&self) -> &[u8] {
        self.data.as_ref()
    }

    fn from_proto(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
        name_policy: NamePolicy<'_>,
    ) -> Result<Self> {
        let proto::TensorProto {
            dims,
            data_type,
            segment,
            float_data,
            int32_data,
            string_data,
            int64_data,
            name,
            raw_data,
            double_data,
            uint64_data,
            data_location,
            external_data,
            ..
        } = proto;

        let name = resolve_name(name.unwrap_or_default(), name_policy)?; // LEGAL: protobuf 可选字段
        let data_type = parse::parse_data_type(
            data_type
                .ok_or_else(|| LoaderError::Onnx(format!("tensor {name} missing data_type")))?,
            &name,
        )?;
        let is_string = data_type == proto::tensor_proto::DataType::String;
        let dtype = parse::map_dtype(data_type, &name)?;
        let shape = parse::parse_dims(&dims, &name)?;
        let element_count = parse::element_count(&shape, &name)?;

        if segment.is_some() {
            return Err(LoaderError::Onnx(format!(
                "segmented tensor not supported: {name}"
            )));
        }

        let data_location = data_location.unwrap_or_default(); // LEGAL: protobuf 可选字段
        let data = if data_location == proto::tensor_proto::DataLocation::External as i32 {
            parse::load_external_data(resolver, &external_data, data_type, dtype, element_count, &name)?
        } else {
            pack::build_tensor_bytes(pack::TensorPackInput {
                data_type,
                dtype,
                element_count,
                raw_data: raw_data.unwrap_or_default(), // LEGAL: protobuf 可选字段
                float_data,
                int32_data,
                int64_data,
                double_data,
                uint64_data,
                string_data,
                name: &name,
            })?
        };

        Ok(Self {
            name,
            dtype,
            shape,
            is_string,
            data,
        })
    }

    fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            return 1;
        }
        self.shape.iter().product()
    }

    /// Create a new OnnxTensor with provided data
    pub fn new(name: String, dtype: Dtype, shape: Vec<usize>, data: Bytes) -> Self {
        Self {
            name,
            dtype,
            shape,
            is_string: false,
            data,
        }
    }

    /// Create a new STRING OnnxTensor
    pub fn new_string(name: String, shape: Vec<usize>, data: Bytes) -> Self {
        Self {
            name,
            dtype: Dtype::U8, // placeholder for string type
            shape,
            is_string: true,
            data,
        }
    }
}

impl OnnxSparseTensor {
    pub(super) fn from_proto(
        proto: proto::SparseTensorProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        let proto::SparseTensorProto {
            values,
            indices,
            dims,
        } = proto;
        let values = values
            .ok_or_else(|| LoaderError::Onnx("sparse tensor missing values tensor".to_string()))?;
        let indices = indices
            .ok_or_else(|| LoaderError::Onnx("sparse tensor missing indices tensor".to_string()))?;
        let values = OnnxTensor::from_initializer(values, resolver)?;
        let indices = OnnxTensor::from_attribute(indices, resolver, "sparse_indices")?;
        let dims = parse::parse_dims(&dims, "sparse_tensor")?;
        let format = infer_sparse_format(&values.name, &indices.name, &indices.shape, &dims);
        Ok(Self {
            values,
            indices,
            dims,
            format,
        })
    }
}

fn infer_sparse_format(
    values_name: &str,
    indices_name: &str,
    indices_shape: &[usize],
    dense_shape: &[usize],
) -> OnnxSparseFormat {
    let values_lower = values_name.to_ascii_lowercase();
    let indices_lower = indices_name.to_ascii_lowercase();
    if values_lower.contains("csr") || indices_lower.contains("csr") {
        return OnnxSparseFormat::Csr;
    }
    if values_lower.contains("csc") || indices_lower.contains("csc") {
        return OnnxSparseFormat::Csc;
    }

    if indices_shape.len() == 2
        && indices_shape[1] == 1
        && dense_shape.len() == 2
        && (values_lower.contains("row") || indices_lower.contains("row"))
    {
        return OnnxSparseFormat::Csr;
    }

    if indices_shape.len() == 2
        && indices_shape[1] == 1
        && dense_shape.len() == 2
        && (values_lower.contains("col") || indices_lower.contains("col"))
    {
        return OnnxSparseFormat::Csc;
    }

    OnnxSparseFormat::Coo
}

enum NamePolicy<'a> {
    Require,
    Fallback(&'a str),
}

fn resolve_name(name: String, policy: NamePolicy<'_>) -> Result<String> {
    if !name.is_empty() {
        return Ok(name);
    }
    match policy {
        NamePolicy::Require => Err(LoaderError::Onnx(
            "initializer tensor missing name".to_string(),
        )),
        NamePolicy::Fallback(fallback) => Ok(fallback.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OnnxTensor construction ───────────────────────────────────────

    #[test]
    fn onnx_tensor_new() {
        let t = OnnxTensor::new(
            "weight".to_string(),
            Dtype::F32,
            vec![2, 3],
            Bytes::from_static(&[0u8; 24]),
        );
        assert_eq!(t.name, "weight");
        assert_eq!(t.dtype, Dtype::F32);
        assert_eq!(t.shape, vec![2, 3]);
        assert!(!t.is_string);
        assert_eq!(t.raw_data().len(), 24);
    }

    #[test]
    fn onnx_tensor_new_string() {
        let t = OnnxTensor::new_string(
            "labels".to_string(),
            vec![3],
            Bytes::from_static(&[0u8; 8]),
        );
        assert!(t.is_string);
        assert_eq!(t.dtype, Dtype::U8);
    }

    #[test]
    fn onnx_tensor_scalar_f32() {
        let data = 3.14f32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - 3.14).abs() < 0.01);
    }

    #[test]
    fn onnx_tensor_scalar_f32_non_scalar_returns_none() {
        let t = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        assert!(t.scalar_f32().is_none());
    }

    #[test]
    fn onnx_tensor_scalar_i64() {
        let data = 42i64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(42));
    }

    #[test]
    fn onnx_tensor_scalar_u8() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![], Bytes::from(vec![7u8]));
        assert_eq!(t.scalar_f32(), Some(7.0));
    }

    #[test]
    fn onnx_tensor_clone() {
        let t = OnnxTensor::new("x".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let c = t.clone();
        assert_eq!(c.name, "x");
        assert_eq!(c.shape, vec![1]);
    }

    #[test]
    fn onnx_tensor_slice() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![2, 2], Bytes::from(vec![0u8; 16]));
        let sl = t.slice();
        assert_eq!(sl.shape, vec![2, 2]);
        assert_eq!(sl.data.len(), 16);
    }

    // ── OnnxSparseFormat ──────────────────────────────────────────────

    #[test]
    fn sparse_format_equality() {
        assert_eq!(OnnxSparseFormat::Coo, OnnxSparseFormat::Coo);
        assert_ne!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csc);
    }

    // ── infer_sparse_format ───────────────────────────────────────────

    #[test]
    fn infer_sparse_format_coo_default() {
        let fmt = infer_sparse_format("values", "indices", &[10, 2], &[3, 4]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    #[test]
    fn infer_sparse_format_csr_by_name() {
        let fmt = infer_sparse_format("csr_values", "indices", &[], &[3, 4]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    #[test]
    fn infer_sparse_format_csc_by_name() {
        let fmt = infer_sparse_format("values", "csc_indices", &[], &[3, 4]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    #[test]
    fn infer_sparse_format_csr_by_row_name() {
        let fmt = infer_sparse_format("row_values", "indices", &[5, 1], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    #[test]
    fn infer_sparse_format_csc_by_col_name() {
        let fmt = infer_sparse_format("values", "col_indices", &[5, 1], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    // ── resolve_name ──────────────────────────────────────────────────

    #[test]
    fn resolve_name_non_empty() {
        let result = resolve_name("my_tensor".to_string(), NamePolicy::Require).unwrap();
        assert_eq!(result, "my_tensor");
    }

    #[test]
    fn resolve_name_empty_require_fails() {
        let result = resolve_name(String::new(), NamePolicy::Require);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_name_empty_fallback() {
        let result = resolve_name(String::new(), NamePolicy::Fallback("default")).unwrap();
        assert_eq!(result, "default");
    }

    // ── element_count edge cases ──────────────────────────────────────

    #[test]
    fn element_count_empty_shape_is_one() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.element_count(), 1);
    }

    #[test]
    fn element_count_single_dim() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![7], Bytes::from(vec![0u8; 28]));
        assert_eq!(t.element_count(), 7);
    }

    #[test]
    fn element_count_multi_dim() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![2, 3, 4], Bytes::from(vec![0u8; 96]));
        assert_eq!(t.element_count(), 24);
    }

    #[test]
    fn element_count_zero_dim_produces_zero() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![0, 5], Bytes::new());
        assert_eq!(t.element_count(), 0);
    }

    // ── scalar_f32 dtype variants ─────────────────────────────────────

    #[test]
    fn scalar_f32_from_i32() {
        let data = (-3i32).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(-3.0));
    }

    #[test]
    fn scalar_f32_from_u16() {
        let data = 1000u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(1000.0));
    }

    #[test]
    fn scalar_f32_from_u32() {
        let data = 500000u32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(500000.0));
    }

    #[test]
    fn scalar_f32_from_u64() {
        let data = 12345u64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(12345.0));
    }

    #[test]
    fn scalar_f32_from_i64() {
        let data = (-99i64).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(-99.0));
    }

    #[test]
    fn scalar_f32_unsupported_dtype_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F64, vec![], Bytes::from(vec![0u8; 8]));
        assert!(t.scalar_f32().is_none());
    }

    // ── scalar_i64 dtype variants ─────────────────────────────────────

    #[test]
    fn scalar_i64_from_u8() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![], Bytes::from(vec![250u8]));
        assert_eq!(t.scalar_i64(), Some(250));
    }

    #[test]
    fn scalar_i64_from_i32_negative() {
        let data = (-7i32).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(-7));
    }

    #[test]
    fn scalar_i64_from_u64() {
        let data = 9876543210u64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(9876543210i64));
    }

    #[test]
    fn scalar_i64_from_f32() {
        let data = 42.0f32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(42));
    }

    #[test]
    fn scalar_i64_unsupported_dtype_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F64, vec![], Bytes::from(vec![0u8; 8]));
        assert!(t.scalar_i64().is_none());
    }

    // ── raw_data and slice consistency ─────────────────────────────────

    #[test]
    fn raw_data_matches_bytes_input() {
        let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let t = OnnxTensor::new("s".to_string(), Dtype::F64, vec![], Bytes::from(bytes.clone()));
        assert_eq!(t.raw_data(), bytes.as_slice());
    }

    #[test]
    fn slice_dtype_and_shape_match_tensor() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![2, 3], Bytes::from(vec![0u8; 24]));
        let sl = t.slice();
        assert_eq!(sl.dtype, t.dtype);
        assert_eq!(sl.shape, t.shape);
        assert_eq!(sl.data, t.raw_data());
    }

    // ── OnnxSparseFormat Copy trait ────────────────────────────────────

    #[test]
    fn sparse_format_copy_and_compare() {
        let a = OnnxSparseFormat::Csr;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn sparse_format_all_variants_distinct() {
        let variants = [OnnxSparseFormat::Coo, OnnxSparseFormat::Csr, OnnxSparseFormat::Csc];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ── infer_sparse_format edge cases ────────────────────────────────

    #[test]
    fn infer_sparse_format_csr_by_values_name_case_insensitive() {
        let fmt = infer_sparse_format("CSR_VALUES", "indices", &[], &[]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    #[test]
    fn infer_sparse_format_csc_by_indices_name_case_insensitive() {
        let fmt = infer_sparse_format("values", "CSC_IDX", &[], &[]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    // ── Debug trait ───────────────────────────────────────────────────

    #[test]
    fn onnx_tensor_debug_output_contains_name() {
        let t = OnnxTensor::new("my_tensor".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let debug = format!("{:?}", t);
        assert!(debug.contains("my_tensor"));
    }

    #[test]
    fn onnx_sparse_tensor_debug_output() {
        let values = OnnxTensor::new("vals".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let indices = OnnxTensor::new("idx".to_string(), Dtype::I64, vec![3], Bytes::from(vec![0u8; 24]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 4],
            format: OnnxSparseFormat::Coo,
        };
        let debug = format!("{:?}", sparse);
        assert!(debug.contains("Coo"));
        assert!(debug.contains("vals"));
    }

    // ── resolve_name: non-empty name ignores policy ────────────────────

    #[test]
    fn resolve_name_non_empty_ignores_fallback_policy() {
        let result = resolve_name("explicit".to_string(), NamePolicy::Fallback("ignored")).unwrap();
        assert_eq!(result, "explicit");
    }

    // ── new_string sets is_string and dtype ────────────────────────────

    #[test]
    fn new_string_preserves_shape_and_data() {
        let data = vec![72u8, 101, 108, 108, 111];
        let t = OnnxTensor::new_string("msg".to_string(), vec![5], Bytes::from(data.clone()));
        assert!(t.is_string);
        assert_eq!(t.dtype, Dtype::U8);
        assert_eq!(t.shape, vec![5]);
        assert_eq!(t.raw_data(), data.as_slice());
    }

    // ── scalar methods on multi-element tensor return none ─────────────

    #[test]
    fn scalar_i64_non_scalar_returns_none() {
        let t = OnnxTensor::new("v".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        assert!(t.scalar_i64().is_none());
    }

    // ── scalar on tensor with zero elements (zero-dim shape) ───────────

    #[test]
    fn scalar_f32_zero_dim_tensor_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![0], Bytes::new());
        assert!(t.scalar_f32().is_none());
    }

    // ── OnnxSparseFormat Hash consistency ──────────────────────────────

    #[test]
    fn sparse_format_hash_equal_for_equal_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |v: OnnxSparseFormat| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(OnnxSparseFormat::Coo), hash_of(OnnxSparseFormat::Coo));
        assert_eq!(hash_of(OnnxSparseFormat::Csr), hash_of(OnnxSparseFormat::Csr));
        assert_eq!(hash_of(OnnxSparseFormat::Csc), hash_of(OnnxSparseFormat::Csc));
    }

    #[test]
    fn sparse_format_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |v: OnnxSparseFormat| {
            let mut h = DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(OnnxSparseFormat::Coo), hash_of(OnnxSparseFormat::Csr));
        assert_ne!(hash_of(OnnxSparseFormat::Csr), hash_of(OnnxSparseFormat::Csc));
        assert_ne!(hash_of(OnnxSparseFormat::Csc), hash_of(OnnxSparseFormat::Coo));
    }

    // ── OnnxSparseFormat Debug output ─────────────────────────────────

    #[test]
    fn sparse_format_debug_output_coo() {
        let debug = format!("{:?}", OnnxSparseFormat::Coo);
        assert_eq!(debug, "Coo");
    }

    #[test]
    fn sparse_format_debug_output_csr() {
        let debug = format!("{:?}", OnnxSparseFormat::Csr);
        assert_eq!(debug, "Csr");
    }

    #[test]
    fn sparse_format_debug_output_csc() {
        let debug = format!("{:?}", OnnxSparseFormat::Csc);
        assert_eq!(debug, "Csc");
    }

    // ── OnnxSparseFormat Clone via Copy ───────────────────────────────

    #[test]
    fn sparse_format_clone_produces_equal() {
        let a = OnnxSparseFormat::Csc;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── OnnxTensor scalar_f32 with F16 ────────────────────────────────

    #[test]
    fn scalar_f32_from_f16_one() {
        // f16 1.0 = 0x3C00
        let data = 0x3C00u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - 1.0).abs() < 0.001);
    }

    #[test]
    fn scalar_f32_from_f16_zero() {
        let data = 0u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(0.0));
    }

    #[test]
    fn scalar_f32_from_bf16_one() {
        // bf16 1.0 = 0x3F80
        let data = 0x3F80u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - 1.0).abs() < 0.01);
    }

    // ── scalar_i64 with F16 and BF16 ──────────────────────────────────

    #[test]
    fn scalar_i64_from_f16() {
        // f16 42.0 = 0x5140
        let data = 0x5140u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(42));
    }

    #[test]
    fn scalar_i64_from_bf16() {
        // bf16 42.0 = 0x4228 (approximate)
        let data = 0x4228u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_i64().unwrap();
        // bf16 has limited precision; just verify it's close to 42
        assert!((val - 42).abs() <= 1);
    }

    #[test]
    fn scalar_i64_from_u16() {
        let data = 60000u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(60000));
    }

    #[test]
    fn scalar_i64_from_u32() {
        let data = 3_000_000_000u32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(3_000_000_000));
    }

    // ── scalar_f32 special float values ───────────────────────────────

    #[test]
    fn scalar_f32_negative_zero() {
        let data = (-0.0f32).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert_eq!(val, 0.0);
        assert!(val.is_sign_negative());
    }

    #[test]
    fn scalar_f32_infinity() {
        let data = f32::INFINITY.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn scalar_f32_neg_infinity() {
        let data = f32::NEG_INFINITY.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_infinite() && val.is_sign_negative());
    }

    #[test]
    fn scalar_f32_nan() {
        let data = f32::NAN.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_nan());
    }

    #[test]
    fn scalar_f32_max() {
        let data = f32::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(f32::MAX));
    }

    #[test]
    fn scalar_f32_min_positive() {
        let data = f32::MIN_POSITIVE.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(f32::MIN_POSITIVE));
    }

    // ── scalar_f32 with I32 edge values ───────────────────────────────

    #[test]
    fn scalar_f32_from_i32_zero() {
        let data = 0i32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(0.0));
    }

    #[test]
    fn scalar_f32_from_i32_max() {
        let data = i32::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - i32::MAX as f32).abs() < 1.0);
    }

    // ── scalar_i64 edge values ────────────────────────────────────────

    #[test]
    fn scalar_i64_from_i64_zero() {
        let data = 0i64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(0));
    }

    #[test]
    fn scalar_i64_from_i64_negative_large() {
        let data = (-999_999_999i64).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(-999_999_999));
    }

    #[test]
    fn scalar_i64_from_i64_max() {
        let data = i64::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(i64::MAX));
    }

    #[test]
    fn scalar_i64_from_i64_min() {
        let data = i64::MIN.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(i64::MIN));
    }

    // ── scalar_i64 with F32 negative ──────────────────────────────────

    #[test]
    fn scalar_i64_from_f32_negative() {
        let data = (-5.0f32).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(-5));
    }

    // ── OnnxTensor element_count boundary ─────────────────────────────

    #[test]
    fn element_count_single_element() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.element_count(), 1);
    }

    #[test]
    fn element_count_all_ones() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![1, 1, 1, 1], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.element_count(), 1);
    }

    #[test]
    fn element_count_leading_zero() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![0, 5, 3], Bytes::new());
        assert_eq!(t.element_count(), 0);
    }

    // ── OnnxTensor raw_data with empty bytes ──────────────────────────

    #[test]
    fn raw_data_empty_bytes() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![0], Bytes::new());
        assert!(t.raw_data().is_empty());
    }

    // ── OnnxTensor slice with empty shape ─────────────────────────────

    #[test]
    fn slice_scalar_tensor() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        let sl = t.slice();
        assert!(sl.shape.is_empty());
        assert_eq!(sl.data.len(), 4);
    }

    // ── OnnxTensor new with empty name ────────────────────────────────

    #[test]
    fn onnx_tensor_new_empty_name() {
        let t = OnnxTensor::new(String::new(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.name, "");
    }

    // ── OnnxTensor new_string with empty data ─────────────────────────

    #[test]
    fn new_string_empty_data() {
        let t = OnnxTensor::new_string("s".to_string(), vec![], Bytes::new());
        assert!(t.is_string);
        assert!(t.raw_data().is_empty());
    }

    // ── infer_sparse_format: CSR priority over shape heuristics ───────

    #[test]
    fn infer_sparse_format_csr_name_takes_priority_over_shape() {
        // Even though shape matches COO condition, name should win
        let fmt = infer_sparse_format("csr_vals", "indices", &[10, 2], &[3, 4]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    // ── infer_sparse_format: CSC priority over shape heuristics ───────

    #[test]
    fn infer_sparse_format_csc_name_takes_priority_over_shape() {
        let fmt = infer_sparse_format("values", "csc_idx", &[10, 2], &[3, 4]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    // ── infer_sparse_format: shape-based Csr requires row keyword ─────

    #[test]
    fn infer_sparse_format_shape_without_row_keyword_is_coo() {
        // shape [5, 1] with dense [3, 3], but no "row" or "col" keyword -> Coo
        let fmt = infer_sparse_format("values", "indices", &[5, 1], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    // ── infer_sparse_format: shape-based with 1D indices is Coo ───────

    #[test]
    fn infer_sparse_format_1d_indices_is_coo() {
        let fmt = infer_sparse_format("values", "indices", &[5], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    // ── infer_sparse_format: non-2d dense shape with row keyword ──────

    #[test]
    fn infer_sparse_format_row_keyword_non_2d_dense_is_coo() {
        // dense_shape is 1D, so shape heuristics don't apply
        let fmt = infer_sparse_format("row_values", "indices", &[5, 1], &[10]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    // ── resolve_name: error message content ───────────────────────────

    #[test]
    fn resolve_name_require_error_contains_text() {
        let err = resolve_name(String::new(), NamePolicy::Require).unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("missing name") || msg.to_lowercase().contains("name"));
    }

    // ── resolve_name: fallback with non-empty string ignores fallback ──

    #[test]
    fn resolve_name_fallback_with_whitespace_name_is_ok() {
        // A name with only whitespace is non-empty, so it passes
        let result = resolve_name(" ".to_string(), NamePolicy::Fallback("def")).unwrap();
        assert_eq!(result, " ");
    }

    // ── OnnxTensor clone independence ──────────────────────────────────

    #[test]
    fn onnx_tensor_clone_is_independent() {
        let t = OnnxTensor::new("x".to_string(), Dtype::I32, vec![2], Bytes::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
        let c = t.clone();
        // Both should have identical shape and data
        assert_eq!(t.shape, c.shape);
        assert_eq!(t.raw_data(), c.raw_data());
        assert_eq!(t.name, c.name);
        assert_eq!(t.dtype, c.dtype);
    }

    // ── OnnxSparseTensor construction ─────────────────────────────────

    #[test]
    fn sparse_tensor_construction_csr() {
        let values = OnnxTensor::new("vals".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let indices = OnnxTensor::new("idx".to_string(), Dtype::I64, vec![3], Bytes::from(vec![0u8; 24]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 4],
            format: OnnxSparseFormat::Csr,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csr);
        assert_eq!(sparse.dims, vec![4, 4]);
        assert_eq!(sparse.values.name, "vals");
        assert_eq!(sparse.indices.name, "idx");
    }

    #[test]
    fn sparse_tensor_construction_csc() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![3, 3],
            format: OnnxSparseFormat::Csc,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csc);
    }

    #[test]
    fn sparse_tensor_clone() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2, 2],
            format: OnnxSparseFormat::Coo,
        };
        let c = sparse.clone();
        assert_eq!(c.format, OnnxSparseFormat::Coo);
        assert_eq!(c.dims, sparse.dims);
    }

    // ── OnnxSparseTensor debug contains format ────────────────────────

    #[test]
    fn sparse_tensor_debug_csr_contains_format() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2, 2],
            format: OnnxSparseFormat::Csr,
        };
        let debug = format!("{:?}", sparse);
        assert!(debug.contains("Csr"));
    }

    // ── OnnxTensor with large shape ───────────────────────────────────

    #[test]
    fn onnx_tensor_large_shape_product() {
        let t = OnnxTensor::new("big".to_string(), Dtype::U8, vec![100, 100], Bytes::from(vec![0u8; 10000]));
        assert_eq!(t.element_count(), 10000);
    }

    // ── scalar_f32 from U8 max ────────────────────────────────────────

    #[test]
    fn scalar_f32_from_u8_max() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![], Bytes::from(vec![255u8]));
        assert_eq!(t.scalar_f32(), Some(255.0));
    }

    // ── scalar_f32 from U8 zero ───────────────────────────────────────

    #[test]
    fn scalar_f32_from_u8_zero() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![], Bytes::from(vec![0u8]));
        assert_eq!(t.scalar_f32(), Some(0.0));
    }

    // ── scalar_f32 from I64 negative ──────────────────────────────────

    #[test]
    fn scalar_f32_from_i64_negative() {
        let data = (-1i64).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(-1.0));
    }

    // ── scalar_i64 from F32 negative fractional ───────────────────────

    #[test]
    fn scalar_i64_from_f32_negative_truncates() {
        let data = (-3.7f32).to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_i64().unwrap();
        // f32 -> i64 truncation; -3.7 -> -3
        assert_eq!(val, -3);
    }

    // ── OnnxTensor with all supported dtypes via new() ────────────────

    #[test]
    fn onnx_tensor_with_dtype_f64() {
        let data = 1.5f64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F64, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.dtype, Dtype::F64);
        assert_eq!(t.raw_data().len(), 8);
    }

    #[test]
    fn onnx_tensor_with_dtype_i8() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I8, vec![], Bytes::from(vec![0u8]));
        assert_eq!(t.dtype, Dtype::I8);
    }

    #[test]
    fn onnx_tensor_with_dtype_i16() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I16, vec![], Bytes::from(vec![0u8; 2]));
        assert_eq!(t.dtype, Dtype::I16);
    }

    #[test]
    fn onnx_tensor_with_dtype_bf16() {
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(vec![0u8; 2]));
        assert_eq!(t.dtype, Dtype::BF16);
    }

    // ── OnnxSparseFormat can be used in HashSet ────────────────────────

    #[test]
    fn sparse_format_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<OnnxSparseFormat> = [
            OnnxSparseFormat::Coo,
            OnnxSparseFormat::Csr,
            OnnxSparseFormat::Csc,
            OnnxSparseFormat::Coo, // duplicate
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── NamePolicy is an enum with two variants (compile test) ────────

    #[test]
    fn name_policy_require_matches() {
        let result = resolve_name("tensor_a".to_string(), NamePolicy::Require);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "tensor_a");
    }

    // ── OnnxTensor scalar methods with data too short return None ─────

    #[test]
    fn scalar_f32_insufficient_data_returns_none() {
        // Shape is scalar (empty), dtype is F32, but data has only 2 bytes
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 2]));
        assert!(t.scalar_f32().is_none());
    }

    #[test]
    fn scalar_i64_insufficient_data_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(vec![0u8; 4]));
        assert!(t.scalar_i64().is_none());
    }

    // ── infer_sparse_format: CSR in both values and indices ───────────

    #[test]
    fn infer_sparse_format_csr_in_indices_name() {
        let fmt = infer_sparse_format("values", "my_csr_ind", &[], &[2, 2]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    #[test]
    fn infer_sparse_format_csc_in_values_name() {
        let fmt = infer_sparse_format("csc_data", "indices", &[], &[2, 2]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    // ── OnnxTensor new_string preserves data content ──────────────────

    #[test]
    fn new_string_with_utf8_data() {
        let data = "hello world".as_bytes().to_vec();
        let t = OnnxTensor::new_string("greeting".to_string(), vec![11], Bytes::from(data.clone()));
        assert!(t.is_string);
        assert_eq!(t.raw_data(), data.as_slice());
        assert_eq!(t.shape, vec![11]);
    }

    // ── element_count for high-dimensional tensor ─────────────────────

    #[test]
    fn element_count_5d_tensor() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![2, 3, 4, 5, 6], Bytes::from(vec![0u8; 720]));
        assert_eq!(t.element_count(), 720);
    }

    // ═══════════════════════════════════════════════════════════════════
    // NEW TESTS (~40) — covering untested boundaries and public API
    // ═══════════════════════════════════════════════════════════════════

    // ── scalar_f32: U16 max boundary ──────────────────────────────────

    #[test]
    fn scalar_f32_from_u16_max() {
        let data = u16::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(65535.0));
    }

    // ── scalar_f32: U32 max — property check (positive finite) ───────

    #[test]
    fn scalar_f32_from_u32_max_is_finite_positive() {
        let data = u32::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_finite());
        assert!(val > 0.0);
    }

    // ── scalar_f32: I32 min boundary ─────────────────────────────────

    #[test]
    fn scalar_f32_from_i32_min() {
        let data = i32::MIN.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val < 0.0);
        assert!((val - i32::MIN as f32).abs() < 1.0);
    }

    // ── scalar_f32: U64 large — property check ───────────────────────

    #[test]
    fn scalar_f32_from_u64_large_is_finite() {
        let data = 4_000_000_000u64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U64, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_finite());
        assert!(val > 3e9);
    }

    // ── scalar_f32: I64 max — property check ─────────────────────────

    #[test]
    fn scalar_f32_from_i64_max_is_finite() {
        let data = i64::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_finite());
        assert!(val > 0.0);
    }

    // ── scalar_f32: I64 min — property check ─────────────────────────

    #[test]
    fn scalar_f32_from_i64_min_is_finite_negative() {
        let data = i64::MIN.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_finite());
        assert!(val < 0.0);
    }

    // ── scalar_f32: BF16 zero ────────────────────────────────────────

    #[test]
    fn scalar_f32_from_bf16_zero() {
        let data = 0u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_f32(), Some(0.0));
    }

    // ── scalar_f32: BF16 negative value ──────────────────────────────

    #[test]
    fn scalar_f32_from_bf16_negative() {
        // bf16 -1.0 = 0xBF80
        let data = 0xBF80u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - (-1.0)).abs() < 0.01);
    }

    // ── scalar_f32: F16 negative value ───────────────────────────────

    #[test]
    fn scalar_f32_from_f16_negative() {
        // f16 -1.0 = 0xBC00
        let data = 0xBC00u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!((val - (-1.0)).abs() < 0.001);
    }

    // ── scalar_f32: F16 max — property: finite positive ──────────────

    #[test]
    fn scalar_f32_from_f16_max_is_finite_positive() {
        // f16 max = 0x7BFF
        let data = 0x7BFFu16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        let val = t.scalar_f32().unwrap();
        assert!(val.is_finite());
        assert!(val > 0.0);
    }

    // ── scalar_i64: I32 max boundary ─────────────────────────────────

    #[test]
    fn scalar_i64_from_i32_max() {
        let data = i32::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(i32::MAX as i64));
    }

    // ── scalar_i64: I32 min boundary ─────────────────────────────────

    #[test]
    fn scalar_i64_from_i32_min() {
        let data = i32::MIN.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(i32::MIN as i64));
    }

    // ── scalar_i64: U32 max boundary ─────────────────────────────────

    #[test]
    fn scalar_i64_from_u32_max() {
        let data = u32::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(u32::MAX as i64));
    }

    // ── scalar_i64: U16 max boundary ─────────────────────────────────

    #[test]
    fn scalar_i64_from_u16_max() {
        let data = u16::MAX.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::U16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(65535));
    }

    // ── scalar_i64: F32 positive fractional truncates ────────────────

    #[test]
    fn scalar_i64_from_f32_positive_truncates() {
        let data = 7.9f32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(7));
    }

    // ── scalar_i64: F32 zero ─────────────────────────────────────────

    #[test]
    fn scalar_i64_from_f32_zero() {
        let data = 0.0f32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(0));
    }

    // ── scalar_i64: F16 zero ─────────────────────────────────────────

    #[test]
    fn scalar_i64_from_f16_zero() {
        let data = 0u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(0));
    }

    // ── scalar_i64: BF16 zero ────────────────────────────────────────

    #[test]
    fn scalar_i64_from_bf16_zero() {
        let data = 0u16.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::BF16, vec![], Bytes::from(data.to_vec()));
        assert_eq!(t.scalar_i64(), Some(0));
    }

    // ── OnnxTensor: data integrity through new() ─────────────────────

    #[test]
    fn onnx_tensor_new_preserves_exact_bytes() {
        let original = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE];
        let t = OnnxTensor::new("magic".to_string(), Dtype::U8, vec![8], Bytes::from(original.clone()));
        assert_eq!(t.raw_data(), original.as_slice());
    }

    // ── OnnxTensor: dtype BOOL ───────────────────────────────────────

    #[test]
    fn onnx_tensor_with_dtype_bool() {
        let t = OnnxTensor::new("flag".to_string(), Dtype::BOOL, vec![1], Bytes::from(vec![1u8]));
        assert_eq!(t.dtype, Dtype::BOOL);
    }

    // ── OnnxTensor: name with unicode characters ─────────────────────

    #[test]
    fn onnx_tensor_name_with_unicode() {
        let t = OnnxTensor::new("权重_🧠".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.name, "权重_🧠");
    }

    // ── OnnxTensor: name preserves surrounding whitespace ────────────

    #[test]
    fn onnx_tensor_name_preserves_whitespace() {
        let t = OnnxTensor::new("  padded  ".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        assert_eq!(t.name, "  padded  ");
    }

    // ── OnnxTensor: new_string sets dtype to U8 ──────────────────────

    #[test]
    fn new_string_always_sets_dtype_u8() {
        let t = OnnxTensor::new_string("s".to_string(), vec![2], Bytes::from(vec![65u8, 66]));
        assert_eq!(t.dtype, Dtype::U8);
        assert!(t.is_string);
    }

    // ── OnnxTensor: new always sets is_string false ──────────────────

    #[test]
    fn new_always_sets_is_string_false() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![1], Bytes::from(vec![0u8]));
        assert!(!t.is_string);
    }

    // ── OnnxTensor: shape with zero in middle dimension ──────────────

    #[test]
    fn element_count_zero_middle_dim() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![3, 0, 5], Bytes::new());
        assert_eq!(t.element_count(), 0);
    }

    // ── OnnxTensor: element_count large product ──────────────────────

    #[test]
    fn element_count_moderate_4d() {
        let t = OnnxTensor::new("s".to_string(), Dtype::U8, vec![2, 3, 4, 5], Bytes::from(vec![0u8; 120]));
        assert_eq!(t.element_count(), 120);
    }

    // ── OnnxSparseTensor: Coo construction ───────────────────────────

    #[test]
    fn sparse_tensor_construction_coo() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![5, 5],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Coo);
        assert_eq!(sparse.dims, vec![5, 5]);
    }

    // ── OnnxSparseTensor: clone produces independent copy ────────────

    #[test]
    fn sparse_tensor_clone_independence() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![3],
            format: OnnxSparseFormat::Csr,
        };
        let c = sparse.clone();
        assert_eq!(c.format, sparse.format);
        assert_eq!(c.dims, sparse.dims);
        assert_eq!(c.values.name, sparse.values.name);
        assert_eq!(c.indices.name, sparse.indices.name);
    }

    // ── OnnxSparseTensor: Debug for Coo variant ──────────────────────

    #[test]
    fn sparse_tensor_debug_coo_contains_format() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2, 2],
            format: OnnxSparseFormat::Coo,
        };
        let debug = format!("{:?}", sparse);
        assert!(debug.contains("Coo"));
    }

    // ── OnnxSparseTensor: Debug for Csc variant ──────────────────────

    #[test]
    fn sparse_tensor_debug_csc_contains_format() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2, 2],
            format: OnnxSparseFormat::Csc,
        };
        let debug = format!("{:?}", sparse);
        assert!(debug.contains("Csc"));
    }

    // ── OnnxSparseTensor: empty dims ─────────────────────────────────

    #[test]
    fn sparse_tensor_dims_empty() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![],
            format: OnnxSparseFormat::Coo,
        };
        assert!(sparse.dims.is_empty());
    }

    // ── OnnxSparseTensor: 3D dims ────────────────────────────────────

    #[test]
    fn sparse_tensor_dims_3d() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 5, 6],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.dims, vec![4, 5, 6]);
    }

    // ── OnnxSparseTensor: values raw_data accessible ─────────────────

    #[test]
    fn sparse_tensor_values_raw_data() {
        let data = vec![1u8, 2, 3, 4];
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(data.clone()));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.values.raw_data(), data.as_slice());
    }

    // ── OnnxSparseTensor: indices raw_data accessible ────────────────

    #[test]
    fn sparse_tensor_indices_raw_data() {
        let data = 7i64.to_le_bytes();
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(data.to_vec()));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![2],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.indices.raw_data(), data.as_slice());
    }

    // ── infer_sparse_format: both names contain csr ──────────────────

    #[test]
    fn infer_sparse_format_both_names_csr() {
        let fmt = infer_sparse_format("csr_vals", "csr_idx", &[], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    // ── infer_sparse_format: values name csr wins over indices csc ───

    #[test]
    fn infer_sparse_format_csr_values_over_csc_indices() {
        // CSR check runs before CSC check in the function
        let fmt = infer_sparse_format("csr_data", "csc_idx", &[], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    // ── infer_sparse_format: both empty names defaults to Coo ────────

    #[test]
    fn infer_sparse_format_both_empty_names() {
        let fmt = infer_sparse_format("", "", &[], &[]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    // ── infer_sparse_format: row keyword in indices name with shape ──

    #[test]
    fn infer_sparse_format_row_in_indices_shape_heuristic() {
        let fmt = infer_sparse_format("data", "row_idx", &[5, 1], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    // ── infer_sparse_format: col keyword in values name with shape ───

    #[test]
    fn infer_sparse_format_col_in_values_shape_heuristic() {
        let fmt = infer_sparse_format("col_data", "idx", &[5, 1], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csc);
    }

    // ── infer_sparse_format: 2D indices but second dim != 1 is Coo ───

    #[test]
    fn infer_sparse_format_2d_indices_second_dim_not_1() {
        let fmt = infer_sparse_format("row_vals", "idx", &[5, 2], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Coo);
    }

    // ── resolve_name: non-empty name with Require is OK ──────────────

    #[test]
    fn resolve_name_valid_name_with_require() {
        let result = resolve_name("tensor_abc".to_string(), NamePolicy::Require).unwrap();
        assert_eq!(result, "tensor_abc");
    }

    // ── resolve_name: fallback with long fallback string ─────────────

    #[test]
    fn resolve_name_fallback_long_string() {
        let fallback = "a_very_long_default_tensor_name_for_testing";
        let result = resolve_name(String::new(), NamePolicy::Fallback(fallback)).unwrap();
        assert_eq!(result, fallback);
    }

    // ── OnnxTensor: multiple shapes with same dtype ──────────────────

    #[test]
    fn onnx_tensor_different_shapes_same_dtype() {
        let shapes = [vec![], vec![1], vec![1, 1], vec![2, 3]];
        for shape in &shapes {
            let t = OnnxTensor::new("s".to_string(), Dtype::F32, shape.clone(), Bytes::new());
            assert_eq!(t.shape, *shape);
            assert_eq!(t.dtype, Dtype::F32);
        }
    }

    // ── OnnxTensor: scalar methods return None for multi-dim shape ───

    #[test]
    fn scalar_f32_rank2_multi_elem_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![2, 2], Bytes::from(vec![0u8; 16]));
        assert!(t.scalar_f32().is_none());
    }

    // ── OnnxTensor: scalar_i64 returns None for rank2 multi-elem ─────

    #[test]
    fn scalar_i64_rank2_multi_elem_returns_none() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![2, 2], Bytes::from(vec![0u8; 32]));
        assert!(t.scalar_i64().is_none());
    }

    // ── OnnxTensor: slice data pointer matches raw_data ──────────────

    #[test]
    fn slice_data_is_same_as_raw_data() {
        let bytes = vec![10u8, 20, 30, 40];
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![1], Bytes::from(bytes.clone()));
        let sl = t.slice();
        assert_eq!(sl.data, t.raw_data());
    }

    // ── OnnxTensor: new with large data ──────────────────────────────

    #[test]
    fn onnx_tensor_new_large_data() {
        let data: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let t = OnnxTensor::new("big".to_string(), Dtype::U8, vec![4096], Bytes::from(data.clone()));
        assert_eq!(t.raw_data().len(), 4096);
        assert_eq!(t.raw_data()[0], 0);
        assert_eq!(t.raw_data()[255], 255);
    }

    // ═══════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS — covering remaining untested boundaries
    // ═══════════════════════════════════════════════════════════════════

    // ── scalar_f32: BOOL dtype returns none (not in match arms) ──────

    #[test]
    fn scalar_f32_bool_dtype_returns_none() {
        let t = OnnxTensor::new("flag".to_string(), Dtype::BOOL, vec![], Bytes::from(vec![1u8]));
        assert!(t.scalar_f32().is_none());
    }

    // ── scalar_i64: BOOL dtype returns none (not in match arms) ──────

    #[test]
    fn scalar_i64_bool_dtype_returns_none() {
        let t = OnnxTensor::new("flag".to_string(), Dtype::BOOL, vec![], Bytes::from(vec![1u8]));
        assert!(t.scalar_i64().is_none());
    }

    // ── OnnxTensor: raw_data from static bytes shares data ───────────

    #[test]
    fn raw_data_from_static_bytes_preserves_content() {
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![2], Bytes::from_static(&[0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89]));
        assert_eq!(t.raw_data(), &[0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89]);
    }

    // ── OnnxTensor: clone of string tensor preserves is_string ───────

    #[test]
    fn clone_of_string_tensor_preserves_is_string() {
        let data = vec![72u8, 105, 33];
        let t = OnnxTensor::new_string("greet".to_string(), vec![3], Bytes::from(data));
        let c = t.clone();
        assert!(c.is_string);
        assert_eq!(c.dtype, Dtype::U8);
        assert_eq!(c.name, "greet");
    }

    // ── OnnxSparseTensor: format field independent of actual dims ────

    #[test]
    fn sparse_tensor_format_independent_of_dims_rank() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        // 1D dims with Csr format — format is just a field, no validation
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![10],
            format: OnnxSparseFormat::Csr,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csr);
        assert_eq!(sparse.dims.len(), 1);
    }

    // ── resolve_name: Fallback with empty fallback string returns empty ─

    #[test]
    fn resolve_name_fallback_with_empty_string_returns_empty() {
        let result = resolve_name(String::new(), NamePolicy::Fallback("")).unwrap();
        assert_eq!(result, "");
    }

    // ── infer_sparse_format: CSC in values, CSR in indices — CSR wins ─

    #[test]
    fn infer_sparse_format_csr_indices_over_csc_values() {
        // CSR is checked first in the function body
        let fmt = infer_sparse_format("csc_data", "csr_idx", &[], &[3, 3]);
        assert_eq!(fmt, OnnxSparseFormat::Csr);
    }

    // ── OnnxSparseFormat: usable as HashMap key ───────────────────────

    #[test]
    fn sparse_format_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(OnnxSparseFormat::Coo, "coordinate");
        map.insert(OnnxSparseFormat::Csr, "compressed_row");
        map.insert(OnnxSparseFormat::Csc, "compressed_col");
        assert_eq!(map.get(&OnnxSparseFormat::Coo), Some(&"coordinate"));
        assert_eq!(map.get(&OnnxSparseFormat::Csr), Some(&"compressed_row"));
        assert_eq!(map.get(&OnnxSparseFormat::Csc), Some(&"compressed_col"));
        assert_eq!(map.len(), 3);
    }

    // ── OnnxTensor: slice borrows data, does not own ─────────────────

    #[test]
    fn slice_lifetime_does_not_consume_tensor() {
        let t = OnnxTensor::new("s".to_string(), Dtype::I32, vec![3], Bytes::from(vec![0u8; 12]));
        let sl = t.slice();
        assert_eq!(sl.data.len(), 12);
        // tensor is still accessible after slice
        assert_eq!(t.raw_data().len(), 12);
    }

    // ── OnnxTensor: scalar methods with shape [1] count as single elem ─

    #[test]
    fn scalar_f32_shape_one_is_scalar() {
        let data = 2.5f32.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::F32, vec![1], Bytes::from(data.to_vec()));
        assert_eq!(t.element_count(), 1);
        let val = t.scalar_f32().unwrap();
        assert!((val - 2.5).abs() < 0.01);
    }

    // ── OnnxTensor: scalar_i64 with shape [1] count as single elem ────

    #[test]
    fn scalar_i64_shape_one_is_scalar() {
        let data = 123i64.to_le_bytes();
        let t = OnnxTensor::new("s".to_string(), Dtype::I64, vec![1], Bytes::from(data.to_vec()));
        assert_eq!(t.element_count(), 1);
        assert_eq!(t.scalar_i64(), Some(123));
    }
}
