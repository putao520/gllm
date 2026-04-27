//! 图类型定义

use std::collections::HashMap;

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

#[derive(Debug, Clone, PartialEq, Eq)]
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
