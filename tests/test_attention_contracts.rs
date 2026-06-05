use gllm::engine::{
    AttentionSemantics, HeadMode, KvAppendSemantics, KvLayoutContract, KvSplitMode, KvStorageKind,
    KvView, MaskMode, PackingDescriptor, PositionContract, ScalingMode, VisibilityMode,
    WeightBacking, WeightView,
};
use gllm::loader::adapter::DType;

#[test]
fn zero_overhead_contracts_do_not_copy_dense_payload() {
    let dense = vec![1.0f32, 2.0, 3.0, 4.0];
    let view = WeightView::from_dense_slice(&dense, vec![2, 2], DType::F32, WeightBacking::SafeTensorsMmap);

    assert_eq!(view.base_ptr, dense.as_ptr() as usize);
    assert_eq!(view.byte_len, dense.len() * std::mem::size_of::<f32>());
    assert_eq!(view.packing, PackingDescriptor::dense());
}

#[test]
fn kv_contracts_can_model_prefill_and_decode_without_reformat() {
    let contract = KvLayoutContract {
        storage_kind: KvStorageKind::HeadMajorPersistent,
        kv_split: KvSplitMode::SplitHalf,
        layer_stride: 30 * 3 * 8192 * 64 * 4,
        head_stride: 8192 * 64 * 4,
        token_stride: 64 * 4,
        dtype: DType::F32,
        append_semantics: KvAppendSemantics::AppendOnly,
    };

    let decode_view = KvView::PersistentCache {
        base_ptr: 0x1000,
        contract: contract.clone(),
        layer: 0,
        visible_range: 0..9,
    };

    let prefill_view = KvView::DenseLocal {
        k_ptr: 0x2000,
        v_ptr: 0x3000,
        seq_len: 5,
        kv_dim: 192,
        dtype: DType::F32,
    };

    let semantics = AttentionSemantics {
        mask_mode: MaskMode::Causal,
        head_mode: HeadMode::GQA { ratio: 3 },
        scaling: ScalingMode::InverseSqrtHeadDim,
        rope: None,
        visibility: VisibilityMode::Decode { total_seq: 9 },
    };

    assert_eq!(decode_view.visible_len(), 9);
    assert_eq!(prefill_view.visible_len(), 5);
    assert!(matches!(semantics.visibility, VisibilityMode::Decode { total_seq: 9 }));
    assert_eq!(contract.storage_kind, KvStorageKind::HeadMajorPersistent);
}

#[test]
fn position_contract_can_encode_contiguous_and_explicit_positions() {
    let contiguous = PositionContract::ContiguousRange { start: 4, len: 5 };
    let positions = vec![8u32, 9, 12];
    let explicit = PositionContract::ExplicitArray {
        ptr: positions.as_ptr() as usize,
        len: positions.len(),
    };

    assert_eq!(contiguous.len(), 5);
    assert_eq!(explicit.len(), 3);
    assert_eq!(positions.as_ptr() as usize, match explicit { PositionContract::ExplicitArray { ptr, .. } => ptr, _ => 0 });
}
