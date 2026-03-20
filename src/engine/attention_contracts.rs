use crate::graph::types::RoPEConfig;
use crate::loader::adapter::DType;
use crate::loader::QuantizedTensor;
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightBacking {
    SafeTensorsMmap,
    OnnxInitializer,
    GgufBlockBytes,
    DevicePtr(u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackingDescriptor {
    pub block_size: Option<usize>,
    pub has_scales: bool,
    pub has_zeros: bool,
}

impl PackingDescriptor {
    pub const fn dense() -> Self {
        Self {
            block_size: None,
            has_scales: false,
            has_zeros: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantScheme {
    Gguf {
        quant_type: gllm_kernels::quant::QuantType,
        ggml_dtype_name: String,
        block_size: Option<usize>,
    },
    Custom {
        name: String,
        block_size: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightView {
    pub logical_shape: Vec<usize>,
    pub storage_dtype: DType,
    pub quant_scheme: Option<QuantScheme>,
    pub base_ptr: usize,
    pub byte_len: usize,
    pub stride: Vec<usize>,
    pub packing: PackingDescriptor,
    pub backing: WeightBacking,
}

impl WeightView {
    pub fn from_dense_slice<T>(
        data: &[T],
        logical_shape: Vec<usize>,
        storage_dtype: DType,
        backing: WeightBacking,
    ) -> Self {
        Self {
            logical_shape,
            storage_dtype,
            quant_scheme: None,
            base_ptr: data.as_ptr() as usize,
            byte_len: std::mem::size_of_val(data),
            stride: Vec::new(),
            packing: PackingDescriptor::dense(),
            backing,
        }
    }

    pub fn from_quantized_tensor(tensor: &QuantizedTensor) -> Self {
        Self {
            logical_shape: tensor.shape.clone(),
            storage_dtype: DType::U8,
            quant_scheme: Some(QuantScheme::Gguf {
                quant_type: tensor.quant_type,
                ggml_dtype_name: format!("{:?}", tensor.ggml_dtype),
                block_size: None,
            }),
            base_ptr: tensor.data.as_ptr() as usize,
            byte_len: tensor.data.len(),
            stride: Vec::new(),
            packing: PackingDescriptor {
                block_size: None,
                has_scales: true,
                has_zeros: true,
            },
            backing: WeightBacking::GgufBlockBytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionContract {
    ContiguousRange { start: u32, len: usize },
    ExplicitArray { ptr: usize, len: usize },
    OffsetStride { offset: u32, stride: u32, len: usize },
}

impl PositionContract {
    pub fn len(&self) -> usize {
        match self {
            Self::ContiguousRange { len, .. }
            | Self::ExplicitArray { len, .. }
            | Self::OffsetStride { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStorageKind {
    DenseSeqMajor,
    HeadMajorPersistent,
    PagedHeadMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSplitMode {
    SplitHalf,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAppendSemantics {
    AppendOnly,
    Overwrite,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayoutContract {
    pub storage_kind: KvStorageKind,
    pub kv_split: KvSplitMode,
    pub layer_stride: usize,
    pub head_stride: usize,
    pub token_stride: usize,
    pub dtype: DType,
    pub append_semantics: KvAppendSemantics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KvView {
    DenseLocal {
        k_ptr: usize,
        v_ptr: usize,
        seq_len: usize,
        kv_dim: usize,
        dtype: DType,
    },
    PersistentCache {
        base_ptr: u64,
        contract: KvLayoutContract,
        layer: usize,
        visible_range: Range<usize>,
    },
    PagedCache {
        page_table_ptr: usize,
        page_data_ptr: u64,
        contract: KvLayoutContract,
        layer: usize,
        visible_range: Range<usize>,
    },
    CompositeView {
        cached: Box<KvView>,
        append: Box<KvView>,
    },
}

impl KvView {
    pub fn visible_len(&self) -> usize {
        match self {
            Self::DenseLocal { seq_len, .. } => *seq_len,
            Self::PersistentCache { visible_range, .. }
            | Self::PagedCache { visible_range, .. } => visible_range.len(),
            Self::CompositeView { cached, append } => cached.visible_len() + append.visible_len(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    Causal,
    Bidirectional,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadMode {
    MHA,
    GQA { ratio: usize },
    MQA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingMode {
    InverseSqrtHeadDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityMode {
    Prefill { seq_len: usize },
    Decode { total_seq: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionSemantics {
    pub mask_mode: MaskMode,
    pub head_mode: HeadMode,
    pub scaling: ScalingMode,
    pub rope: Option<RoPEConfig>,
    pub visibility: VisibilityMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_contract_len_matches_variant() {
        assert_eq!(PositionContract::ContiguousRange { start: 10, len: 4 }.len(), 4);
        assert_eq!(PositionContract::ExplicitArray { ptr: 0x1234, len: 2 }.len(), 2);
        assert_eq!(PositionContract::OffsetStride { offset: 0, stride: 2, len: 8 }.len(), 8);
    }

    #[test]
    fn kv_view_visible_len_works() {
        let dense = KvView::DenseLocal {
            k_ptr: 1,
            v_ptr: 2,
            seq_len: 5,
            kv_dim: 192,
            dtype: DType::F32,
        };
        let persistent = KvView::PersistentCache {
            base_ptr: 0x1000,
            contract: KvLayoutContract {
                storage_kind: KvStorageKind::HeadMajorPersistent,
                kv_split: KvSplitMode::SplitHalf,
                layer_stride: 1024,
                head_stride: 256,
                token_stride: 64,
                dtype: DType::F32,
                append_semantics: KvAppendSemantics::AppendOnly,
            },
            layer: 0,
            visible_range: 0..7,
        };
        let composite = KvView::CompositeView {
            cached: Box::new(persistent),
            append: Box::new(dense),
        };
        assert_eq!(composite.visible_len(), 12);
    }

    #[test]
    fn weight_view_from_dense_slice_preserves_bytes() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let view = WeightView::from_dense_slice(&data, vec![2, 2], DType::F32, WeightBacking::SafeTensorsMmap);
        assert_eq!(view.byte_len, 16);
        assert_eq!(view.logical_shape, vec![2, 2]);
        assert!(view.quant_scheme.is_none());
    }

    #[test]
    fn attention_semantics_can_represent_prefill_and_decode() {
        let prefill = AttentionSemantics {
            mask_mode: MaskMode::Causal,
            head_mode: HeadMode::GQA { ratio: 3 },
            scaling: ScalingMode::InverseSqrtHeadDim,
            rope: Some(RoPEConfig {
                head_dim: 64,
                rope_theta: 100000.0,
                max_seq_len: 8192,
                interleaved: false,
            }),
            visibility: VisibilityMode::Prefill { seq_len: 5 },
        };
        let decode = AttentionSemantics {
            visibility: VisibilityMode::Decode { total_seq: 9 },
            ..prefill.clone()
        };
        assert!(matches!(prefill.visibility, VisibilityMode::Prefill { .. }));
        assert!(matches!(decode.visibility, VisibilityMode::Decode { .. }));
    }
}
