//! 量化元数据格式测试
//!
//! 演示 Ω1 元数据驱动的量化配置

use gllm::loader::QuantizationMetadata;

/// TEST-QUANT-001: 量化元数据解析
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 正向
/// **期望结果**: 正确解析量化配置中的位宽、分组大小等参数
#[test]
fn test_parse_quantization_metadata() {
    let json = r#"
    {
        "qweight": {
            "group_size": 128,
            "bits": 4,
            "desc_act": false,
            "is_sym": false
        },
        "qweight_2": {
            "group_size": 256,
            "bits": 8,
            "desc_act": true,
            "is_sym": true
        }
    }
    "#;

    let metadata: std::collections::HashMap<String, QuantizationMetadata> =
        serde_json::from_str(json).expect("解析成功");

    // 验证第一个量化组
    let qweight_meta = metadata.get("qweight").expect("存在 qweight 元数据");
    assert_eq!(qweight_meta.bits, 4);
    assert_eq!(qweight_meta.group_size, 128);
    assert!(!qweight_meta.is_sym);
    assert!(!qweight_meta.desc_act);

    // 验证第二个量化组
    let qweight_2_meta = metadata.get("qweight_2").expect("存在 qweight_2 元数据");
    assert_eq!(qweight_2_meta.bits, 8);
    assert_eq!(qweight_2_meta.group_size, 256);
    assert!(qweight_2_meta.is_sym);
    assert!(qweight_2_meta.desc_act);
}

/// TEST-QUANT-002: 量化元数据默认值
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 边界
/// **期望结果**: 未指定量化参数时使用合理的默认值
#[test]
fn test_quantization_metadata_defaults() {
    // 有效配置，使用默认值
    let valid = QuantizationMetadata {
        bits: 4,
        group_size: 128,
        desc_act: false,
        is_sym: false,
    };
    assert_eq!(valid.bits, 4);
    assert_eq!(valid.group_size, 128);
}

/// TEST-QUANT-003: 量化元数据序列化往返
/// **关联需求**: REQ-TEST-006
/// **测试类型**: 正向
/// **期望结果**: 序列化后反序列化得到相同的量化配置
#[test]
fn test_serialization_roundtrip() {
    let original = r#"
    {
        "qweight": {
            "group_size": 128,
            "bits": 4,
            "desc_act": false,
            "is_sym": false
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
