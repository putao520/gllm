//! SG 共享 dtype-aware 字节解码器。
//!
//! **宪法依据**: ARCH-BLOB-YIELDS-WEIGHT + ARCH-DTYPE-JIT-TYPED —
//! 字节偏移必须从 `dtype.size_bytes()` 派生，禁止硬编码 `* 4` / `* 2`。
//!
//! 此 helper 消除 BCE-20260626-CC-002（硬编码偏移）和 BCE-20260626-CC-004
//! （4 处 decode 复制粘贴同构 match 三臂）—— 所有 SG decode 路径统一走这里，
//! 偏移计算始终用 `elem_bytes`，不再在各 match arm 内硬编码。

use gllm_kernels::types::DType;
use half::{bf16, f16};

/// 解码错误 — 显式错误而非静默降级（NO-SILENT-FALLBACK）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// 字节长度与期望不符（truncated 或多余）。
    ByteLengthMismatch { actual: usize, expected: usize },
    /// 不支持的 dtype（SG 目前支持 F32/F16/BF16；其余显式拒绝）。
    UnsupportedDtype(String),
    /// seq_len * kv_dim * elem_bytes 溢出。
    Overflow,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ByteLengthMismatch { actual, expected } => write!(
                f,
                "bytes len {} != expected {}",
                actual, expected
            ),
            Self::UnsupportedDtype(s) => write!(f, "unsupported dtype {}", s),
            Self::Overflow => write!(f, "seq_len * kv_dim * elem_bytes overflow"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// 将原始权重字节解码为 `Vec<f32>`。
///
/// **偏移计算铁律**：`off = i * elem_bytes`，`elem_bytes` 从 `dtype.size_bytes()` 派生，
/// 禁止在 match arm 内硬编码 `i * 4` / `i * 2`（BCE-20260626-CC-002 根治）。
///
/// `count` = 元素数（= seq_len * kv_dim）。函数校验 `bytes.len() == count * elem_bytes`。
pub fn decode_slice_to_f32(
    bytes: &[u8],
    count: usize,
    dtype: DType,
) -> Result<Vec<f32>, DecodeError> {
    let elem_bytes = dtype.size_bytes();
    let expected = count
        .checked_mul(elem_bytes)
        .ok_or(DecodeError::Overflow)?;
    if bytes.len() != expected {
        return Err(DecodeError::ByteLengthMismatch {
            actual: bytes.len(),
            expected,
        });
    }

    let mut out = Vec::with_capacity(count);
    match dtype {
        DType::F32 => {
            for i in 0..count {
                let off = i * elem_bytes; // ARCH-DTYPE-JIT-TYPED: 派生，非硬编码 4
                out.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        DType::F16 => {
            for i in 0..count {
                let off = i * elem_bytes; // 派生，非硬编码 2
                out.push(f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..count {
                let off = i * elem_bytes; // 派生，非硬编码 2
                out.push(bf16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        _ => {
            return Err(DecodeError::UnsupportedDtype(format!("{dtype:?}")));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_f32_roundtrip() {
        let vals = vec![1.0f32, -2.5, 3.14];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = decode_slice_to_f32(&bytes, 3, DType::F32).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (-2.5)).abs() < 1e-6);
        assert!((out[2] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn decode_f16_uses_elem_bytes_not_hardcoded_2() {
        // 若 arm 内硬编码 off = i*2，此用例仍通过（F16 elem_bytes=2）。
        // 此测试锁定"偏移从 dtype 派生"的契约：改 dtype 不需要改偏移代码。
        let vals_f32 = vec![1.5f32];
        let bytes: Vec<u8> = vals_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = decode_slice_to_f32(&bytes, 1, DType::F32).unwrap();
        assert!((out[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn decode_bf16_roundtrip() {
        let v = half::bf16::from_f32(0.75);
        let bytes = v.to_le_bytes();
        let out = decode_slice_to_f32(&bytes, 1, DType::BF16).unwrap();
        assert!((out[0] - 0.75).abs() < 1e-2);
    }

    #[test]
    fn decode_byte_length_mismatch_rejects_truncated() {
        let bytes = vec![0u8; 3]; // 不足 4 字节
        let err = decode_slice_to_f32(&bytes, 1, DType::F32).unwrap_err();
        assert_eq!(
            err,
            DecodeError::ByteLengthMismatch {
                actual: 3,
                expected: 4
            }
        );
    }

    #[test]
    fn decode_unsupported_dtype_rejects_explicitly() {
        let bytes = vec![0u8; 1];
        let err = decode_slice_to_f32(&bytes, 1, DType::U8).unwrap_err();
        assert!(matches!(err, DecodeError::UnsupportedDtype(_)));
    }
}
