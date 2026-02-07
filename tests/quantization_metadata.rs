//! 量化元数据格式测试
//!
//! 演示 Ω1 元数据驱动的量化配置

use gllm::loader::QuantizationMetadata;

#[test]
fn test_parse_quantization_metadata() {
    let json = r#"
    {
        "qweight": {
            "bits": 4,
            "signed": false,
            "block_size": 128,
            "companions": {
                "scales": "scales",
                "zeros": "qzeros"
            }
        },
        "qweight_2": {
            "bits": 8,
            "signed": true,
            "block_size": 256,
            "companions": {
                "scales": "scales_2"
            }
        }
    }
    "#;

    let metadata: std::collections::HashMap<String, QuantizationMetadata> =
        serde_json::from_str(json).expect("解析成功");

    // 验证第一个量化组
    let qweight_meta = metadata.get("qweight").expect("存在 qweight 元数据");
    assert_eq!(qweight_meta.bits, 4);
    assert_eq!(qweight_meta.signed, false);
    assert_eq!(qweight_meta.block_size, 128);
    assert_eq!(qweight_meta.companions.scales.as_ref().unwrap(), "scales");
    assert_eq!(qweight_meta.companions.zeros.as_ref().unwrap(), "qzeros");

    // 验证第二个量化组
    let qweight_2_meta = metadata.get("qweight_2").expect("存在 qweight_2 元数据");
    assert_eq!(qweight_2_meta.bits, 8);
    assert_eq!(qweight_2_meta.signed, true);
    assert_eq!(qweight_2_meta.block_size, 256);
    assert_eq!(qweight_2_meta.companions.scales.as_ref().unwrap(), "scales_2");
    assert!(qweight_2_meta.companions.zeros.is_none()); // 无 zeros
}

#[test]
fn test_quantization_metadata_validation() {
    // 有效配置
    let valid = QuantizationMetadata {
        bits: 4,
        signed: false,
        block_size: 128,
        companions: Default::default(),
    };
    assert!(valid.validate().is_ok());

    // 无效：不支持的位宽
    let invalid_bits = QuantizationMetadata {
        bits: 7, // 不是 4 或 8
        signed: false,
        block_size: 128,
        companions: Default::default(),
    };
    assert!(invalid_bits.validate().is_err());

    // 无效：block_size 为 0
    let invalid_block = QuantizationMetadata {
        bits: 4,
        signed: false,
        block_size: 0,
        companions: Default::default(),
    };
    assert!(invalid_block.validate().is_err());
}

#[test]
fn test_serialization_roundtrip() {
    let original = r#"
    {
        "qweight": {
            "bits": 4,
            "signed": false,
            "block_size": 128,
            "companions": {
                "scales": "scales",
                "zeros": "qzeros"
            }
        }
    }
    "#;

    let metadata: std::collections::HashMap<String, QuantizationMetadata> =
        serde_json::from_str(original).expect("解析成功");

    let serialized = serde_json::to_string_pretty(&metadata).expect("序列化成功");
    let reparsed: std::collections::HashMap<String, QuantizationMetadata> =
        serde_json::from_str(&serialized).expect("重新解析成功");

    assert_eq!(metadata, reparsed);
}
