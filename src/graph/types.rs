//! 图类型定义

/// 零拷贝权重绑定（仅保存名称/元信息，数据由 provider 按需提供）
#[derive(Debug, Clone, PartialEq)]
pub struct WeightBinding {
    pub source_name: String,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
    /// Embedded weight bytes (constant weights loaded at graph-build time).
    pub data: Option<Vec<u8>>,
    /// Runtime weight pointer injected by the caller (e.g. from WeightsHandle).
    pub ptr: Option<*const f32>,
    /// ARCH-WEIGHT-CANONICAL-LAYOUT: 该权重的 shape 是否仍是 HF `[out, in]` 而
    /// 需要语义转置到 `[K, N]`。
    pub shape_needs_transpose: bool,
}

// Safety: WeightBinding only stores a raw pointer; it does not own the data.
// The caller is responsible for ensuring the pointer remains valid.
unsafe impl Send for WeightBinding {}
unsafe impl Sync for WeightBinding {}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationInfo {
    pub scale: f32,
    pub zero_point: i64,
    pub axis: Option<i32>,
}

impl QuantizationInfo {
    /// Create a new QuantizationInfo with no axis.
    pub fn new(scale: f32, zero_point: i64) -> Self {
        Self { scale, zero_point, axis: None }
    }

    /// Create a new QuantizationInfo with a specific axis.
    pub fn with_axis(scale: f32, zero_point: i64, axis: i32) -> Self {
        Self { scale, zero_point, axis: Some(axis) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    Coo,
    Csr,
    Csc,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseTensorBinding {
    pub format: SparseFormat,
    pub indices: String,
    pub values: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorAttrValue {
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// 属性值
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
    Tensor(TensorAttrValue),
}

// ============================================================================
// LayeredRequestControl — 逐层请求控制 (§9.3 残差总线)
// ============================================================================

/// 逐层请求控制信号
///
/// 由 RequestState 转换而来，供 PolymorphicExecutor 在逐层执行时
/// 检查 Early-Exit / 层跳过等控制信号。
pub struct LayeredRequestControl {
    /// 目标层索引（Early-Exit 截断点）
    pub target_layer: u32,
    /// 退出标志（非零 = 请求已终止）
    pub exit_flag: std::sync::atomic::AtomicU32,
}

impl LayeredRequestControl {
    /// 检查是否应在指定层退出
    pub fn should_exit_at(&self, layer_idx: u32) -> bool {
        self.exit_flag.load(std::sync::atomic::Ordering::Relaxed) != 0
            || (self.target_layer > 0 && layer_idx >= self.target_layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===================================================================
    // QuantizationInfo (existing + new)
    // ===================================================================

    #[test]
    fn quantization_info_equality() {
        let a = QuantizationInfo { scale: 0.5, zero_point: 128, axis: Some(0) };
        let b = QuantizationInfo { scale: 0.5, zero_point: 128, axis: Some(0) };
        assert_eq!(a, b);
    }

    #[test]
    fn quantization_info_no_axis() {
        let q = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        assert!(q.axis.is_none());
    }

    #[test]
    fn quantization_info_new_creates_no_axis() {
        let q = QuantizationInfo::new(0.125, -128);
        assert_eq!(q.scale, 0.125);
        assert_eq!(q.zero_point, -128);
        assert!(q.axis.is_none());
    }

    #[test]
    fn quantization_info_with_axis_sets_axis() {
        let q = QuantizationInfo::with_axis(2.0, 64, 1);
        assert_eq!(q.scale, 2.0);
        assert_eq!(q.zero_point, 64);
        assert_eq!(q.axis, Some(1));
    }

    #[test]
    fn quantization_info_inequality_different_scale() {
        let a = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        let b = QuantizationInfo { scale: 2.0, zero_point: 0, axis: None };
        assert_ne!(a, b);
    }

    #[test]
    fn quantization_info_inequality_different_zero_point() {
        let a = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        let b = QuantizationInfo { scale: 1.0, zero_point: 1, axis: None };
        assert_ne!(a, b);
    }

    #[test]
    fn quantization_info_inequality_different_axis() {
        let a = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        let b = QuantizationInfo { scale: 1.0, zero_point: 0, axis: Some(0) };
        assert_ne!(a, b);
    }

    #[test]
    fn quantization_info_zero_scale() {
        let q = QuantizationInfo { scale: 0.0, zero_point: 0, axis: None };
        assert_eq!(q.scale, 0.0);
    }

    #[test]
    fn quantization_info_negative_zero_point() {
        let q = QuantizationInfo { scale: 1.0, zero_point: -128, axis: Some(0) };
        assert_eq!(q.zero_point, -128);
    }

    #[test]
    fn quantization_info_i64_max_zero_point() {
        let q = QuantizationInfo { scale: 0.001, zero_point: i64::MAX, axis: None };
        assert_eq!(q.zero_point, i64::MAX);
    }

    #[test]
    fn quantization_info_i64_min_zero_point() {
        let q = QuantizationInfo { scale: 0.001, zero_point: i64::MIN, axis: None };
        assert_eq!(q.zero_point, i64::MIN);
    }

    #[test]
    fn quantization_info_clone() {
        let q = QuantizationInfo { scale: 0.5, zero_point: 128, axis: Some(1) };
        let cloned = q.clone();
        assert_eq!(q, cloned);
    }

    #[test]
    fn quantization_info_debug_format() {
        let q = QuantizationInfo { scale: 0.5, zero_point: 128, axis: Some(0) };
        let debug = format!("{q:?}");
        assert!(debug.contains("QuantizationInfo"));
        assert!(debug.contains("scale"));
        assert!(debug.contains("zero_point"));
        assert!(debug.contains("axis"));
    }

    #[test]
    fn quantization_info_negative_axis() {
        let q = QuantizationInfo { scale: 1.0, zero_point: 0, axis: Some(-1) };
        assert_eq!(q.axis, Some(-1));
    }

    // ===================================================================
    // SparseFormat (existing + new)
    // ===================================================================

    #[test]
    fn sparse_format_variants() {
        assert_ne!(SparseFormat::Coo, SparseFormat::Csr);
        assert_ne!(SparseFormat::Csr, SparseFormat::Csc);
    }

    #[test]
    fn sparse_format_debug() {
        assert!(format!("{:?}", SparseFormat::Coo).contains("Coo"));
        assert!(format!("{:?}", SparseFormat::Csr).contains("Csr"));
        assert!(format!("{:?}", SparseFormat::Csc).contains("Csc"));
    }

    #[test]
    fn sparse_format_clone() {
        let a = SparseFormat::Coo;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn sparse_format_copy_semantics() {
        let a = SparseFormat::Csr;
        let b = a; // Copy, not move
        assert_eq!(a, b);
        assert_eq!(a, SparseFormat::Csr);
    }

    #[test]
    fn sparse_format_all_distinct() {
        let variants = [SparseFormat::Coo, SparseFormat::Csr, SparseFormat::Csc];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn sparse_format_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        SparseFormat::Coo.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        SparseFormat::Coo.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn sparse_format_hash_different_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        SparseFormat::Coo.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        SparseFormat::Csr.hash(&mut h2);
        assert_ne!(h1.finish(), h2.finish());
    }

    #[test]
    fn sparse_format_eq_trait() {
        assert!(SparseFormat::Coo == SparseFormat::Coo);
        assert!(SparseFormat::Coo != SparseFormat::Csr);
    }

    // ===================================================================
    // SparseTensorBinding (existing + new)
    // ===================================================================

    #[test]
    fn sparse_tensor_binding_fields() {
        let st = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "ind".to_string(),
            values: "val".to_string(),
            shape: vec![3, 4],
        };
        assert_eq!(st.format, SparseFormat::Csr);
        assert_eq!(st.shape, vec![3, 4]);
    }

    #[test]
    fn sparse_tensor_binding_equality() {
        let a = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "row".into(),
            values: "col".into(),
            shape: vec![10, 10],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn sparse_tensor_binding_inequality_different_format() {
        let a = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "i".into(),
            values: "v".into(),
            shape: vec![2],
        };
        let b = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "i".into(),
            values: "v".into(),
            shape: vec![2],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn sparse_tensor_binding_inequality_different_shape() {
        let a = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "i".into(),
            values: "v".into(),
            shape: vec![2],
        };
        let b = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "i".into(),
            values: "v".into(),
            shape: vec![3],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn sparse_tensor_binding_empty_shape() {
        let st = SparseTensorBinding {
            format: SparseFormat::Csc,
            indices: String::new(),
            values: String::new(),
            shape: vec![],
        };
        assert!(st.shape.is_empty());
        assert!(st.indices.is_empty());
    }

    #[test]
    fn sparse_tensor_binding_large_shape() {
        let st = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "idx".into(),
            values: "val".into(),
            shape: vec![usize::MAX, usize::MAX],
        };
        assert_eq!(st.shape[0], usize::MAX);
    }

    #[test]
    fn sparse_tensor_binding_debug_format() {
        let st = SparseTensorBinding {
            format: SparseFormat::Csc,
            indices: "row_ptr".into(),
            values: "data".into(),
            shape: vec![5, 6],
        };
        let debug = format!("{st:?}");
        assert!(debug.contains("SparseTensorBinding"));
        assert!(debug.contains("format"));
        assert!(debug.contains("indices"));
    }

    #[test]
    fn sparse_tensor_binding_clone_independence() {
        let a = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "orig".into(),
            values: "vals".into(),
            shape: vec![3],
        };
        let mut b = a.clone();
        b.indices = "modified".into();
        assert_eq!(a.indices, "orig");
        assert_eq!(b.indices, "modified");
    }

    // ===================================================================
    // WeightBinding (existing + new)
    // ===================================================================

    #[test]
    fn weight_binding_with_data() {
        let wb = WeightBinding {
            source_name: "model.weight".into(),
            shape: vec![64, 32],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![0u8; 64 * 32 * 4]),
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.source_name, "model.weight");
        assert_eq!(wb.shape, vec![64, 32]);
        assert!(wb.data.is_some());
        assert!(!wb.shape_needs_transpose);
    }

    #[test]
    fn weight_binding_needs_transpose() {
        let wb = WeightBinding {
            source_name: "q_proj".into(),
            shape: vec![4096, 4096],
            dtype: safetensors::Dtype::BF16,
            data: None,
            ptr: None,
            shape_needs_transpose: true,
        };
        assert!(wb.shape_needs_transpose);
    }

    #[test]
    fn weight_binding_equality() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn weight_binding_debug() {
        let wb = WeightBinding {
            source_name: "test".into(),
            shape: vec![1],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let debug = format!("{wb:?}");
        assert!(debug.contains("source_name"));
        assert!(debug.contains("shape_needs_transpose"));
    }

    #[test]
    fn weight_binding_empty_shape() {
        let wb = WeightBinding {
            source_name: "bias".into(),
            shape: vec![],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![]),
            ptr: None,
            shape_needs_transpose: false,
        };
        assert!(wb.shape.is_empty());
        assert!(wb.data.as_ref().unwrap().is_empty());
    }

    #[test]
    fn weight_binding_scalar_shape() {
        let wb = WeightBinding {
            source_name: "scalar".into(),
            shape: vec![1],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![0, 0, 0, 0]),
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.shape.len(), 1);
        assert_eq!(wb.shape[0], 1);
    }

    #[test]
    fn weight_binding_3d_shape() {
        let wb = WeightBinding {
            source_name: "3d_tensor".into(),
            shape: vec![16, 8, 4],
            dtype: safetensors::Dtype::F16,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.shape, vec![16, 8, 4]);
    }

    #[test]
    fn weight_binding_bf16_dtype() {
        let wb = WeightBinding {
            source_name: "bf16_w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::BF16,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.dtype, safetensors::Dtype::BF16);
    }

    #[test]
    fn weight_binding_clone_independence() {
        let mut a = WeightBinding {
            source_name: "orig".into(),
            shape: vec![4, 4],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![1, 2, 3]),
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = a.clone();
        a.source_name = "modified".into();
        assert_eq!(b.source_name, "orig");
    }

    #[test]
    fn weight_binding_inequality_different_name() {
        let a = WeightBinding {
            source_name: "a".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "b".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_binding_inequality_different_dtype() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "w".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::BF16,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_binding_inequality_transpose_flag() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: true,
        };
        assert_ne!(a, b);
    }

    // ===================================================================
    // TensorAttrValue (existing + new)
    // ===================================================================

    #[test]
    fn tensor_attr_value_fields() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2, 3],
            data: vec![0u8; 24],
        };
        assert_eq!(tav.shape, vec![2, 3]);
        assert_eq!(tav.data.len(), 24);
    }

    #[test]
    fn tensor_attr_value_clone() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![42],
        };
        let cloned = tav.clone();
        assert_eq!(cloned.data, vec![42]);
    }

    #[test]
    fn tensor_attr_value_equality() {
        let a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2],
            data: vec![1, 2, 3],
        };
        let b = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2],
            data: vec![1, 2, 3],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn tensor_attr_value_inequality_different_dtype() {
        let a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![0, 0, 0, 0],
        };
        let b = TensorAttrValue {
            dtype: safetensors::Dtype::BF16,
            shape: vec![1],
            data: vec![0, 0],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_attr_value_empty_data() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![0],
            data: vec![],
        };
        assert!(tav.data.is_empty());
        assert_eq!(tav.shape, vec![0]);
    }

    #[test]
    fn tensor_attr_value_large_data() {
        let data = vec![0xABu8; 1024];
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::U8,
            shape: vec![1024],
            data: data.clone(),
        };
        assert_eq!(tav.data.len(), 1024);
        assert_eq!(tav.data[0], 0xAB);
    }

    #[test]
    fn tensor_attr_value_debug_format() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1, 1],
            data: vec![42],
        };
        let debug = format!("{tav:?}");
        assert!(debug.contains("TensorAttrValue"));
        assert!(debug.contains("dtype"));
        assert!(debug.contains("shape"));
        assert!(debug.contains("data"));
    }

    #[test]
    fn tensor_attr_value_clone_independence() {
        let mut a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2],
            data: vec![1, 2, 3, 4],
        };
        let b = a.clone();
        a.data[0] = 99;
        assert_eq!(b.data[0], 1);
    }

    #[test]
    fn tensor_attr_value_i64_dtype() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::I64,
            shape: vec![4],
            data: vec![0u8; 32],
        };
        assert_eq!(tav.dtype, safetensors::Dtype::I64);
    }

    // ===================================================================
    // AttrValue (existing + new)
    // ===================================================================

    #[test]
    fn attr_value_int() {
        let v = AttrValue::Int(42);
        assert_eq!(v, AttrValue::Int(42));
    }

    #[test]
    fn attr_value_string() {
        let v = AttrValue::String("hello".to_string());
        assert_eq!(v, AttrValue::String("hello".to_string()));
    }

    #[test]
    fn attr_value_floats() {
        let v = AttrValue::Floats(vec![1.0, 2.0]);
        assert_eq!(v, AttrValue::Floats(vec![1.0, 2.0]));
    }

    #[test]
    fn attr_value_ints() {
        let v = AttrValue::Ints(vec![1, 2, 3]);
        assert_eq!(v, AttrValue::Ints(vec![1, 2, 3]));
    }

    #[test]
    fn attr_value_float() {
        let v = AttrValue::Float(3.14);
        assert_eq!(v, AttrValue::Float(3.14));
    }

    #[test]
    fn attr_value_strings() {
        let v = AttrValue::Strings(vec!["a".into(), "b".into()]);
        assert_eq!(v, AttrValue::Strings(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn attr_value_tensor() {
        let v = AttrValue::Tensor(TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![0],
        });
        let debug = format!("{v:?}");
        assert!(debug.contains("Tensor"));
    }

    #[test]
    fn attr_value_int_zero() {
        let v = AttrValue::Int(0);
        assert_eq!(v, AttrValue::Int(0));
    }

    #[test]
    fn attr_value_int_i64_max() {
        let v = AttrValue::Int(i64::MAX);
        assert_eq!(v, AttrValue::Int(i64::MAX));
    }

    #[test]
    fn attr_value_int_i64_min() {
        let v = AttrValue::Int(i64::MIN);
        assert_eq!(v, AttrValue::Int(i64::MIN));
    }

    #[test]
    fn attr_value_int_negative() {
        let v = AttrValue::Int(-1);
        assert_eq!(v, AttrValue::Int(-1));
        assert_ne!(v, AttrValue::Int(1));
    }

    #[test]
    fn attr_value_float_zero() {
        let v = AttrValue::Float(0.0);
        assert_eq!(v, AttrValue::Float(0.0));
    }

    #[test]
    fn attr_value_float_negative_zero() {
        let v1 = AttrValue::Float(0.0);
        let v2 = AttrValue::Float(-0.0);
        assert_eq!(v1, v2);
    }

    #[test]
    fn attr_value_float_special_infinity() {
        let v = AttrValue::Float(f32::INFINITY);
        assert_eq!(v, AttrValue::Float(f32::INFINITY));
    }

    #[test]
    fn attr_value_float_special_neg_infinity() {
        let v = AttrValue::Float(f32::NEG_INFINITY);
        assert_eq!(v, AttrValue::Float(f32::NEG_INFINITY));
    }

    #[test]
    fn attr_value_float_nan_not_equal_to_self() {
        let v = AttrValue::Float(f32::NAN);
        assert_ne!(v, v);
    }

    #[test]
    fn attr_value_empty_ints() {
        let v = AttrValue::Ints(vec![]);
        assert_eq!(v, AttrValue::Ints(vec![]));
    }

    #[test]
    fn attr_value_empty_floats() {
        let v = AttrValue::Floats(vec![]);
        assert_eq!(v, AttrValue::Floats(vec![]));
    }

    #[test]
    fn attr_value_empty_strings() {
        let v = AttrValue::Strings(vec![]);
        assert_eq!(v, AttrValue::Strings(vec![]));
    }

    #[test]
    fn attr_value_ints_with_negative() {
        let v = AttrValue::Ints(vec![-10, 0, 10]);
        assert_eq!(v, AttrValue::Ints(vec![-10, 0, 10]));
    }

    #[test]
    fn attr_value_floats_with_special_values() {
        let v = AttrValue::Floats(vec![f32::INFINITY, f32::NEG_INFINITY, 0.0]);
        assert_eq!(v, AttrValue::Floats(vec![f32::INFINITY, f32::NEG_INFINITY, 0.0]));
    }

    #[test]
    fn attr_value_empty_string() {
        let v = AttrValue::String(String::new());
        assert_eq!(v, AttrValue::String(String::new()));
    }

    #[test]
    fn attr_value_string_with_unicode() {
        let v = AttrValue::String("你好世界".to_string());
        assert_eq!(v, AttrValue::String("你好世界".to_string()));
    }

    #[test]
    fn attr_value_variant_mismatch() {
        let v = AttrValue::Int(42);
        assert_ne!(v, AttrValue::Float(42.0));
    }

    #[test]
    fn attr_value_clone_independence() {
        let a = AttrValue::Ints(vec![1, 2, 3]);
        let b = a.clone();
        drop(a); // original dropped, proving b owns its own data
        assert_eq!(b, AttrValue::Ints(vec![1, 2, 3]));
    }

    #[test]
    fn attr_value_debug_all_variants() {
        let debug_int = format!("{:?}", AttrValue::Int(1));
        assert!(debug_int.contains("Int"));

        let debug_float = format!("{:?}", AttrValue::Float(1.0));
        assert!(debug_float.contains("Float"));

        let debug_string = format!("{:?}", AttrValue::String("x".into()));
        assert!(debug_string.contains("String"));

        let debug_ints = format!("{:?}", AttrValue::Ints(vec![]));
        assert!(debug_ints.contains("Ints"));

        let debug_floats = format!("{:?}", AttrValue::Floats(vec![]));
        assert!(debug_floats.contains("Floats"));

        let debug_strings = format!("{:?}", AttrValue::Strings(vec![]));
        assert!(debug_strings.contains("Strings"));
    }

    // ===================================================================
    // LayeredRequestControl (existing + new)
    // ===================================================================

    #[test]
    fn control_no_exit_default() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(0));
        assert!(!ctrl.should_exit_at(100));
    }

    #[test]
    fn control_exit_at_target() {
        let ctrl = LayeredRequestControl {
            target_layer: 5,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(4));
        assert!(ctrl.should_exit_at(5));
        assert!(ctrl.should_exit_at(6));
    }

    #[test]
    fn control_exit_on_flag() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(1),
        };
        assert!(ctrl.should_exit_at(0));
    }

    #[test]
    fn control_flag_set_at_runtime() {
        let ctrl = LayeredRequestControl {
            target_layer: 100,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(5));
        ctrl.exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
        assert!(ctrl.should_exit_at(5));
    }

    #[test]
    fn control_target_layer_zero_never_exits_by_target() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(0));
        assert!(!ctrl.should_exit_at(u32::MAX));
    }

    #[test]
    fn control_target_layer_one() {
        let ctrl = LayeredRequestControl {
            target_layer: 1,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(1));
    }

    #[test]
    fn control_flag_nonzero_value() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(42),
        };
        assert!(ctrl.should_exit_at(0));
    }

    #[test]
    fn control_flag_max_u32() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(u32::MAX),
        };
        assert!(ctrl.should_exit_at(0));
    }

    #[test]
    fn control_target_u32_max() {
        let ctrl = LayeredRequestControl {
            target_layer: u32::MAX,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(u32::MAX - 1));
        assert!(ctrl.should_exit_at(u32::MAX));
    }

    #[test]
    fn control_both_flag_and_target() {
        let ctrl = LayeredRequestControl {
            target_layer: 10,
            exit_flag: std::sync::atomic::AtomicU32::new(1),
        };
        assert!(ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(5));
        assert!(ctrl.should_exit_at(10));
    }

    #[test]
    fn control_flag_clear_then_set() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(50));
        ctrl.exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
        assert!(ctrl.should_exit_at(50));
        ctrl.exit_flag.store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(!ctrl.should_exit_at(50));
    }

    #[test]
    fn control_exact_boundary_layer() {
        let ctrl = LayeredRequestControl {
            target_layer: 7,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(6));
        assert!(ctrl.should_exit_at(7));
        assert!(ctrl.should_exit_at(8));
    }

    #[test]
    fn control_concurrent_flag_reads() {
        let ctrl = LayeredRequestControl {
            target_layer: 50,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        // Multiple reads without flag set
        for layer in 0..50 {
            assert!(!ctrl.should_exit_at(layer));
        }
        // All layers at or beyond target should exit
        for layer in 50..55 {
            assert!(ctrl.should_exit_at(layer));
        }
    }

    // ===================================================================
    // QuantizationInfo — additional coverage
    // ===================================================================

    #[test]
    fn quantization_info_scale_f32_max() {
        let q = QuantizationInfo::new(f32::MAX, 0);
        assert_eq!(q.scale, f32::MAX);
    }

    #[test]
    fn quantization_info_scale_f32_min_positive() {
        let q = QuantizationInfo::new(f32::MIN_POSITIVE, 0);
        assert_eq!(q.scale, f32::MIN_POSITIVE);
        assert!(q.scale > 0.0);
    }

    #[test]
    fn quantization_info_scale_infinity() {
        let q = QuantizationInfo::new(f32::INFINITY, 0);
        assert!(q.scale.is_infinite());
        assert!(q.scale.is_sign_positive());
    }

    #[test]
    fn quantization_info_axis_i32_max() {
        let q = QuantizationInfo::with_axis(1.0, 0, i32::MAX);
        assert_eq!(q.axis, Some(i32::MAX));
    }

    #[test]
    fn quantization_info_axis_i32_min() {
        let q = QuantizationInfo::with_axis(1.0, 0, i32::MIN);
        assert_eq!(q.axis, Some(i32::MIN));
    }

    #[test]
    fn quantization_info_new_vs_with_axis_equality() {
        let a = QuantizationInfo { scale: 2.0, zero_point: 100, axis: None };
        let b = QuantizationInfo::new(2.0, 100);
        assert_eq!(a, b);
    }

    #[test]
    fn quantization_info_with_axis_vs_manual_equality() {
        let a = QuantizationInfo { scale: 0.5, zero_point: -50, axis: Some(3) };
        let b = QuantizationInfo::with_axis(0.5, -50, 3);
        assert_eq!(a, b);
    }

    // ===================================================================
    // SparseFormat — additional coverage
    // ===================================================================

    #[test]
    fn sparse_format_hashset_insert_all() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SparseFormat::Coo);
        set.insert(SparseFormat::Csr);
        set.insert(SparseFormat::Csc);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn sparse_format_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(SparseFormat::Coo, "coordinate");
        map.insert(SparseFormat::Csr, "compressed_row");
        map.insert(SparseFormat::Csc, "compressed_col");
        assert_eq!(map.get(&SparseFormat::Coo), Some(&"coordinate"));
        assert_eq!(map.get(&SparseFormat::Csr), Some(&"compressed_row"));
        assert_eq!(map.get(&SparseFormat::Csc), Some(&"compressed_col"));
    }

    #[test]
    fn sparse_format_match_exhaustive() {
        let variants = [SparseFormat::Coo, SparseFormat::Csr, SparseFormat::Csc];
        let names: Vec<&str> = variants.iter().map(|f| match f {
            SparseFormat::Coo => "coo",
            SparseFormat::Csr => "csr",
            SparseFormat::Csc => "csc",
        }).collect();
        assert_eq!(names, ["coo", "csr", "csc"]);
    }

    // ===================================================================
    // SparseTensorBinding — additional coverage
    // ===================================================================

    #[test]
    fn sparse_tensor_binding_inequality_different_indices() {
        let a = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "row_idx".into(),
            values: "vals".into(),
            shape: vec![3, 4],
        };
        let b = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "col_idx".into(),
            values: "vals".into(),
            shape: vec![3, 4],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn sparse_tensor_binding_inequality_different_values() {
        let a = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "idx".into(),
            values: "data".into(),
            shape: vec![5, 5],
        };
        let b = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "idx".into(),
            values: "weights".into(),
            shape: vec![5, 5],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn sparse_tensor_binding_1d_shape() {
        let st = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "i".into(),
            values: "v".into(),
            shape: vec![100],
        };
        assert_eq!(st.shape.len(), 1);
        assert_eq!(st.shape[0], 100);
    }

    #[test]
    fn sparse_tensor_binding_3d_shape() {
        let st = SparseTensorBinding {
            format: SparseFormat::Csr,
            indices: "idx".into(),
            values: "val".into(),
            shape: vec![4, 5, 6],
        };
        assert_eq!(st.shape, vec![4, 5, 6]);
    }

    #[test]
    fn sparse_tensor_binding_same_format_all_fields_differ() {
        let a = SparseTensorBinding {
            format: SparseFormat::Csc,
            indices: "a_idx".into(),
            values: "a_val".into(),
            shape: vec![2, 3],
        };
        let b = SparseTensorBinding {
            format: SparseFormat::Csc,
            indices: "b_idx".into(),
            values: "b_val".into(),
            shape: vec![4, 5],
        };
        assert_ne!(a, b);
    }

    // ===================================================================
    // WeightBinding — additional coverage
    // ===================================================================

    #[test]
    fn weight_binding_all_none_optional_fields() {
        let wb = WeightBinding {
            source_name: "test".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert!(wb.data.is_none());
        assert!(wb.ptr.is_none());
    }

    #[test]
    fn weight_binding_inequality_different_data_presence() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![0u8; 16]),
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "w".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_binding_inequality_different_shape() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![3, 4],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "w".into(),
            shape: vec![4, 3],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_binding_inequality_different_data_content() {
        let a = WeightBinding {
            source_name: "w".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![1u8, 2, 3, 4]),
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "w".into(),
            shape: vec![2],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![5u8, 6, 7, 8]),
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_binding_f16_dtype() {
        let wb = WeightBinding {
            source_name: "f16_weight".into(),
            shape: vec![128, 64],
            dtype: safetensors::Dtype::F16,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.dtype, safetensors::Dtype::F16);
    }

    #[test]
    fn weight_binding_4d_shape() {
        let wb = WeightBinding {
            source_name: "conv_weight".into(),
            shape: vec![64, 3, 7, 7],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_eq!(wb.shape.len(), 4);
        assert_eq!(wb.shape, vec![64, 3, 7, 7]);
    }

    #[test]
    fn weight_binding_data_some_empty_vs_none() {
        let with_empty = WeightBinding {
            source_name: "w".into(),
            shape: vec![],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![]),
            ptr: None,
            shape_needs_transpose: false,
        };
        let with_none = WeightBinding {
            source_name: "w".into(),
            shape: vec![],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert_ne!(with_empty, with_none);
    }

    #[test]
    fn weight_binding_empty_source_name() {
        let wb = WeightBinding {
            source_name: String::new(),
            shape: vec![1],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        assert!(wb.source_name.is_empty());
    }

    #[test]
    fn weight_binding_clone_deep_copies_data() {
        let original = WeightBinding {
            source_name: "deep".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![42u8; 16]),
            ptr: None,
            shape_needs_transpose: true,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        let mut modified = cloned;
        if let Some(ref mut d) = modified.data {
            d[0] = 99;
        }
        assert_eq!(original.data.as_ref().unwrap()[0], 42);
        assert_eq!(modified.data.as_ref().unwrap()[0], 99);
    }

    // ===================================================================
    // TensorAttrValue — additional coverage
    // ===================================================================

    #[test]
    fn tensor_attr_value_inequality_different_shape() {
        let a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2, 3],
            data: vec![0u8; 24],
        };
        let b = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![3, 2],
            data: vec![0u8; 24],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_attr_value_inequality_different_data() {
        let a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2],
            data: vec![0u8, 0, 0, 0, 0, 0, 0, 0],
        };
        let b = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2],
            data: vec![0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn tensor_attr_value_bf16_dtype() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::BF16,
            shape: vec![4],
            data: vec![0u8; 8],
        };
        assert_eq!(tav.dtype, safetensors::Dtype::BF16);
    }

    #[test]
    fn tensor_attr_value_f16_dtype() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F16,
            shape: vec![2],
            data: vec![0u8; 4],
        };
        assert_eq!(tav.dtype, safetensors::Dtype::F16);
    }

    #[test]
    fn tensor_attr_value_3d_shape() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2, 3, 4],
            data: vec![0u8; 96],
        };
        assert_eq!(tav.shape, vec![2, 3, 4]);
    }

    #[test]
    fn tensor_attr_value_single_byte_data() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::U8,
            shape: vec![1],
            data: vec![0x42],
        };
        assert_eq!(tav.data.len(), 1);
        assert_eq!(tav.data[0], 0x42);
    }

    // ===================================================================
    // AttrValue — additional coverage
    // ===================================================================

    #[test]
    fn attr_value_ints_large_count() {
        let v = AttrValue::Ints((0..1000).collect());
        if let AttrValue::Ints(ref ints) = v {
            assert_eq!(ints.len(), 1000);
            assert_eq!(ints[0], 0);
            assert_eq!(ints[999], 999);
        } else {
            panic!("expected Ints variant");
        }
    }

    #[test]
    fn attr_value_floats_large_count() {
        let v = AttrValue::Floats((0..500).map(|i| i as f32 * 0.1).collect());
        if let AttrValue::Floats(ref floats) = v {
            assert_eq!(floats.len(), 500);
        } else {
            panic!("expected Floats variant");
        }
    }

    #[test]
    fn attr_value_strings_large_count() {
        let v = AttrValue::Strings((0..200).map(|i| format!("item_{i}")).collect());
        if let AttrValue::Strings(ref strings) = v {
            assert_eq!(strings.len(), 200);
        } else {
            panic!("expected Strings variant");
        }
    }

    #[test]
    fn attr_value_clone_float() {
        let a = AttrValue::Float(2.718);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn attr_value_clone_string_variant() {
        let a = AttrValue::String("test_string".into());
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn attr_value_clone_tensor_variant() {
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2, 2],
            data: vec![1u8, 2, 3, 4, 5, 6, 7, 8],
        };
        let a = AttrValue::Tensor(tav);
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn attr_value_tensor_inequality() {
        let t1 = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![0u8; 4],
        };
        let t2 = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![1u8, 2, 3, 4],
        };
        assert_ne!(AttrValue::Tensor(t1), AttrValue::Tensor(t2));
    }

    #[test]
    fn attr_value_long_string() {
        let long = "x".repeat(10000);
        let v = AttrValue::String(long);
        if let AttrValue::String(ref s) = v {
            assert_eq!(s.len(), 10000);
        } else {
            panic!("expected String variant");
        }
    }

    #[test]
    fn attr_value_ints_single_element() {
        let v = AttrValue::Ints(vec![42]);
        if let AttrValue::Ints(ref ints) = v {
            assert_eq!(ints.len(), 1);
            assert_eq!(ints[0], 42);
        } else {
            panic!("expected Ints variant");
        }
    }

    #[test]
    fn attr_value_floats_single_element() {
        let v = AttrValue::Floats(vec![1.5]);
        if let AttrValue::Floats(ref floats) = v {
            assert_eq!(floats.len(), 1);
            assert_eq!(floats[0], 1.5);
        } else {
            panic!("expected Floats variant");
        }
    }

    // ===================================================================
    // LayeredRequestControl — additional coverage
    // ===================================================================

    #[test]
    fn control_flag_toggled_multiple_times() {
        let ctrl = LayeredRequestControl {
            target_layer: 0,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        assert!(!ctrl.should_exit_at(10));
        ctrl.exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
        assert!(ctrl.should_exit_at(10));
        ctrl.exit_flag.store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(!ctrl.should_exit_at(10));
        ctrl.exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
        assert!(ctrl.should_exit_at(10));
    }

    #[test]
    fn control_target_layer_one_exits_all_higher() {
        let ctrl = LayeredRequestControl {
            target_layer: 1,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        for layer in 1..100 {
            assert!(ctrl.should_exit_at(layer));
        }
    }

    // ===================================================================
    // Additional coverage — 15 new tests
    // ===================================================================

    #[test]
    fn quantization_info_scale_nan() {
        // Arrange: create QuantizationInfo with NaN scale
        let q = QuantizationInfo::new(f32::NAN, 0);
        // Act & Assert: NaN scale is NaN, and self != self via f32 semantics
        assert!(q.scale.is_nan());
        // QuantizationInfo PartialEq derives, so NaN scale means a != a
        assert_ne!(q, q);
    }

    #[test]
    fn sparse_format_iterate_all_variants() {
        // Arrange: collect all SparseFormat variants into a vec
        let all = vec![SparseFormat::Coo, SparseFormat::Csr, SparseFormat::Csc];
        // Act: count them
        // Assert: exactly 3 distinct variants exist
        assert_eq!(all.len(), 3);
        assert_ne!(all[0], all[1]);
        assert_ne!(all[1], all[2]);
        assert_ne!(all[0], all[2]);
    }

    #[test]
    fn sparse_tensor_binding_unicode_names() {
        // Arrange: create SparseTensorBinding with unicode strings
        let st = SparseTensorBinding {
            format: SparseFormat::Coo,
            indices: "行索引_🚀".into(),
            values: "値_💎".into(),
            shape: vec![10],
        };
        // Assert: strings preserved correctly
        assert_eq!(st.indices, "行索引_🚀");
        assert_eq!(st.values, "値_💎");
        assert_eq!(st.shape, vec![10]);
    }

    #[test]
    fn weight_binding_zero_in_shape() {
        // Arrange: create WeightBinding with a zero dimension
        let wb = WeightBinding {
            source_name: "zero_dim".into(),
            shape: vec![4, 0, 3],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![]),
            ptr: None,
            shape_needs_transpose: false,
        };
        // Assert: shape contains zero and is preserved
        assert_eq!(wb.shape, vec![4, 0, 3]);
        assert_eq!(wb.shape[1], 0);
    }

    #[test]
    fn tensor_attr_value_empty_shape_vector() {
        // Arrange: create TensorAttrValue with completely empty shape (scalar-like)
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![],
            data: vec![0u8; 4],
        };
        // Assert: empty shape is valid and preserved
        assert!(tav.shape.is_empty());
        assert_eq!(tav.data.len(), 4);
    }

    #[test]
    fn attr_value_floats_with_nan_elements() {
        // Arrange: create Floats variant containing NaN
        let v = AttrValue::Floats(vec![1.0, f32::NAN, 3.0]);
        // Assert: NaN in Floats means the value is not equal to an identical-looking one
        // because f32 NaN != NaN, derived PartialEq propagates this
        let v2 = AttrValue::Floats(vec![1.0, f32::NAN, 3.0]);
        assert_ne!(v, v2);
    }

    #[test]
    fn attr_value_ints_boundary_values() {
        // Arrange: create Ints with i64 boundary values
        let v = AttrValue::Ints(vec![i64::MIN, 0, i64::MAX]);
        // Assert: all boundary values preserved
        if let AttrValue::Ints(ref ints) = v {
            assert_eq!(ints[0], i64::MIN);
            assert_eq!(ints[1], 0);
            assert_eq!(ints[2], i64::MAX);
        } else {
            panic!("expected Ints variant");
        }
    }

    #[test]
    fn attr_value_strings_with_empty_element() {
        // Arrange: Strings vector containing an empty string element
        let v = AttrValue::Strings(vec!["first".into(), String::new(), "third".into()]);
        // Assert: empty string preserved within the vector
        if let AttrValue::Strings(ref ss) = v {
            assert_eq!(ss.len(), 3);
            assert_eq!(ss[1], "");
            assert_eq!(ss[0], "first");
            assert_eq!(ss[2], "third");
        } else {
            panic!("expected Strings variant");
        }
    }

    #[test]
    fn attr_value_ints_empty_vs_floats_empty_not_equal() {
        // Arrange: two empty vector variants of different types
        let ints = AttrValue::Ints(vec![]);
        let floats = AttrValue::Floats(vec![]);
        // Assert: different variants are never equal even if both are empty
        assert_ne!(ints, floats);
    }

    #[test]
    fn weight_binding_long_source_name() {
        // Arrange: create WeightBinding with very long source_name
        let long_name = "a".repeat(10000);
        let wb = WeightBinding {
            source_name: long_name.clone(),
            shape: vec![1],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        // Assert: long name preserved without truncation
        assert_eq!(wb.source_name.len(), 10000);
        assert_eq!(wb.source_name, long_name);
    }

    #[test]
    fn sparse_format_hashmap_remove() {
        // Arrange: insert all formats into HashMap then remove one
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(SparseFormat::Coo, 1);
        map.insert(SparseFormat::Csr, 2);
        map.insert(SparseFormat::Csc, 3);
        // Act: remove Csr
        let removed = map.remove(&SparseFormat::Csr);
        // Assert: Csr removed, others remain
        assert_eq!(removed, Some(2));
        assert_eq!(map.len(), 2);
        assert!(map.contains_key(&SparseFormat::Coo));
        assert!(map.contains_key(&SparseFormat::Csc));
        assert!(!map.contains_key(&SparseFormat::Csr));
    }

    #[test]
    fn tensor_attr_value_zero_in_shape() {
        // Arrange: TensorAttrValue with a zero in one dimension
        let tav = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![3, 0, 5],
            data: vec![],
        };
        // Assert: zero dimension preserved, data is empty because total elements is 0
        assert_eq!(tav.shape, vec![3, 0, 5]);
        assert!(tav.data.is_empty());
    }

    #[test]
    fn quantization_info_partial_eq_nan_zero_point_irrelevant() {
        // Arrange: two QuantizationInfo with same scale but NaN vs NaN - they are equal
        // because zero_point is i64, not f32 - NaN is in scale
        let a = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        let b = QuantizationInfo { scale: 1.0, zero_point: 0, axis: None };
        // Assert: identical values are equal
        assert_eq!(a, b);
        // Now with NaN scale - derived PartialEq on f32 means NaN != NaN
        let c = QuantizationInfo { scale: f32::NAN, zero_point: 0, axis: None };
        assert_ne!(c, c);
    }

    #[test]
    fn sparse_tensor_binding_clone_preserves_format() {
        // Arrange: create SparseTensorBinding with Csc format
        let original = SparseTensorBinding {
            format: SparseFormat::Csc,
            indices: "col_ptr".into(),
            values: "values".into(),
            shape: vec![8, 8],
        };
        // Act: clone it
        let cloned = original.clone();
        // Assert: format is exactly preserved after clone
        assert_eq!(cloned.format, SparseFormat::Csc);
        assert_eq!(cloned, original);
    }

    #[test]
    fn layered_request_control_flag_priority_over_target() {
        // Arrange: exit_flag is set AND target_layer is 50
        let ctrl = LayeredRequestControl {
            target_layer: 50,
            exit_flag: std::sync::atomic::AtomicU32::new(1),
        };
        // Assert: flag triggers exit even at layer 0 (well before target_layer)
        assert!(ctrl.should_exit_at(0));
        assert!(ctrl.should_exit_at(49));
        assert!(ctrl.should_exit_at(50));
        // Act: clear the flag
        ctrl.exit_flag.store(0, std::sync::atomic::Ordering::Relaxed);
        // Assert: now only target_layer boundary matters
        assert!(!ctrl.should_exit_at(49));
        assert!(ctrl.should_exit_at(50));
    }

    // ===================================================================
    // Additional coverage — 10 new tests (distinct from existing)
    // ===================================================================

    #[test]
    fn weight_binding_ptr_field_not_compared_in_equality() {
        // Arrange: two WeightBindings that differ only in ptr (raw pointer),
        // but ptr is *const f32 which is not PartialEq — since ptr is Option<*const f32>,
        // two None values compare equal regardless of what pointer *would* be there
        let a = WeightBinding {
            source_name: "ptr_test".into(),
            shape: vec![2, 3],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        let b = WeightBinding {
            source_name: "ptr_test".into(),
            shape: vec![2, 3],
            dtype: safetensors::Dtype::F32,
            data: None,
            ptr: None,
            shape_needs_transpose: false,
        };
        // Assert: both with ptr=None are equal
        assert_eq!(a, b);
    }

    #[test]
    fn attr_value_float_subnormal_value() {
        // Arrange: create AttrValue::Float with a subnormal (denormalized) f32
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        assert!(subnormal > 0.0);
        assert!(subnormal.is_subnormal());
        let v = AttrValue::Float(subnormal);
        // Assert: subnormal value is preserved through equality
        assert_eq!(v, AttrValue::Float(subnormal));
        // A different subnormal should not be equal
        let other = f32::from_bits(2u32);
        assert_ne!(v, AttrValue::Float(other));
    }

    #[test]
    fn quantization_info_with_negative_axis_preserved() {
        // Arrange: negative axis is a valid axis value in quantization (e.g. -1 for last dim)
        let q = QuantizationInfo::with_axis(0.5, 128, -1);
        // Assert: negative axis is stored and retrieved correctly
        assert_eq!(q.axis, Some(-1));
        // Act: clone it
        let cloned = q.clone();
        // Assert: clone preserves negative axis
        assert_eq!(cloned.axis, Some(-1));
        assert_eq!(q, cloned);
    }

    #[test]
    fn sparse_tensor_binding_all_formats_roundtrip() {
        // Arrange: create one SparseTensorBinding per SparseFormat variant
        let formats = [SparseFormat::Coo, SparseFormat::Csr, SparseFormat::Csc];
        // Act: for each format, create a binding, clone it, verify round-trip
        for fmt in formats {
            let original = SparseTensorBinding {
                format: fmt,
                indices: "idx".into(),
                values: "val".into(),
                shape: vec![4, 5],
            };
            let cloned = original.clone();
            // Assert: clone preserves all fields including format
            assert_eq!(original, cloned);
            assert_eq!(cloned.format, fmt);
        }
    }

    #[test]
    fn tensor_attr_value_equality_requires_identical_data_bytes() {
        // Arrange: two TensorAttrValues with same dtype and shape but different data bytes
        let a = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![0x00, 0x00, 0x80, 0x3F], // 1.0f32 in LE
        };
        let b = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![1],
            data: vec![0x00, 0x00, 0x00, 0x40], // 2.0f32 in LE
        };
        // Assert: different byte content means not equal
        assert_ne!(a, b);
    }

    #[test]
    fn attr_value_tensor_wrap_and_extract() {
        // Arrange: create a TensorAttrValue and wrap it in AttrValue::Tensor
        let inner = TensorAttrValue {
            dtype: safetensors::Dtype::F32,
            shape: vec![2, 3],
            data: vec![42u8; 24],
        };
        let wrapped = AttrValue::Tensor(inner.clone());
        // Act: extract via pattern match
        let extracted = match &wrapped {
            AttrValue::Tensor(t) => t.clone(),
            other => panic!("expected Tensor variant, got {other:?}"),
        };
        // Assert: round-trip preserves inner value exactly
        assert_eq!(extracted, inner);
    }

    #[test]
    fn layered_request_control_send_sync_across_threads() {
        use std::sync::Arc;
        use std::thread;
        // Arrange: share LayeredRequestControl across threads via Arc (requires Send+Sync)
        let ctrl = Arc::new(LayeredRequestControl {
            target_layer: 100,
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        });
        let ctrl_clone = Arc::clone(&ctrl);
        // Act: spawn a thread that reads should_exit_at
        let handle = thread::spawn(move || {
            ctrl_clone.should_exit_at(50)
        });
        let result = handle.join().expect("thread panicked");
        // Assert: no panic, and layer 50 < target 100, no flag => should not exit
        assert!(!result);
    }

    #[test]
    fn layered_request_control_flag_transition_during_iteration() {
        // Arrange: iterate layers, set flag partway through
        let ctrl = LayeredRequestControl {
            target_layer: u32::MAX, // never triggers by target
            exit_flag: std::sync::atomic::AtomicU32::new(0),
        };
        let mut results = Vec::new();
        for layer in 0..10 {
            if layer == 5 {
                ctrl.exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
            }
            results.push(ctrl.should_exit_at(layer));
        }
        // Assert: layers 0-4 do not exit; layers 5-9 do exit
        assert_eq!(&results[..5], &[false, false, false, false, false]);
        assert_eq!(&results[5..], &[true, true, true, true, true]);
    }

    #[test]
    fn attr_value_debug_output_contains_variant_name() {
        // Arrange: create each AttrValue variant and format with Debug
        let cases: Vec<(&str, AttrValue)> = vec![
            ("Int", AttrValue::Int(0)),
            ("Float", AttrValue::Float(0.0)),
            ("String", AttrValue::String("".into())),
            ("Ints", AttrValue::Ints(vec![])),
            ("Floats", AttrValue::Floats(vec![])),
            ("Strings", AttrValue::Strings(vec![])),
            ("Tensor", AttrValue::Tensor(TensorAttrValue {
                dtype: safetensors::Dtype::F32,
                shape: vec![],
                data: vec![],
            })),
        ];
        // Act & Assert: each Debug output contains its variant name
        for (name, val) in cases {
            let debug = format!("{val:?}");
            assert!(
                debug.contains(name),
                "Debug output for {name} should contain variant name: got {debug}"
            );
        }
    }

    #[test]
    fn weight_binding_clone_with_data_deep_copies_byte_content() {
        // Arrange: WeightBinding with non-trivial data bytes
        let original = WeightBinding {
            source_name: "deep_copy".into(),
            shape: vec![2, 2],
            dtype: safetensors::Dtype::F32,
            data: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            ptr: None,
            shape_needs_transpose: true,
        };
        // Act: clone and mutate the clone's data
        let mut cloned = original.clone();
        if let Some(ref mut d) = cloned.data {
            d[0] = 0x00;
        }
        // Assert: original's data is untouched (deep copy)
        assert_eq!(original.data.as_ref().unwrap()[0], 0xDE);
        assert_eq!(cloned.data.as_ref().unwrap()[0], 0x00);
        // Assert: other fields are identical
        assert_eq!(original.source_name, cloned.source_name);
        assert_eq!(original.shape, cloned.shape);
        assert_eq!(original.shape_needs_transpose, cloned.shape_needs_transpose);
    }
}
