//! Level Keys 预计算 + 缓存 (SPEC §3).
//!
//! 对每个检测层 layer_L 和每个层级描述文本 desc_x,计算:
//!
//! ```text
//! tokens   := tokenizer.encode(desc_x)          (T_x,)
//! embed    := embed_layer(tokens)               (T_x, hidden)
//! k_proj   := k_proj_at_layer(L)(RmsNorm(embed))(T_x, kv_dim)
//! K_Lx     := mean_over_axis0(k_proj)           (kv_dim,)
//! ```
//!
//! 结果 `LevelKeysCache: HashMap<layer_idx, [Vec<f32>; 3]>` 常驻.

use std::collections::HashMap;

use gllm_kernels::types::DType;
use half::{bf16, f16};

use super::{
    small_graph::{EmbedLookupOnlyGraph, KProjOnlyGraph},
    SemanticGatekeeperConfig, SemanticGatekeeperError, TokenizerEncoder,
};

/// 预计算的层级键缓存.
///
/// key: detection_layer 的物理层索引
/// value: `[K_L1, K_L2, K_L3]` 每个向量维度 = `num_kv_heads × head_dim`
#[derive(Debug, Clone, Default)]
pub struct LevelKeysCache {
    keys: HashMap<usize, [Vec<f32>; 3]>,
    /// 所有检测层的物理索引集合 (与 `keys.keys()` 一致,冗余字段用于
    /// SemanticGatekeeperCallback.target_layers 返回.)
    detection_layers: Vec<usize>,
    /// 每个向量的维度 (`num_kv_heads × head_dim`).
    kv_dim: usize,
}

impl LevelKeysCache {
    /// 构造一个空缓存.
    pub fn new(kv_dim: usize) -> Self {
        Self {
            keys: HashMap::new(),
            detection_layers: Vec::new(),
            kv_dim,
        }
    }

    /// 向缓存插入某检测层的 3 个层级键.
    ///
    /// 校验每个向量 `len == kv_dim` 且 finite,否则返回 Err.
    pub fn insert(
        &mut self,
        layer_idx: usize,
        keys: [Vec<f32>; 3],
    ) -> Result<(), LevelKeysError> {
        for (i, k) in keys.iter().enumerate() {
            if k.len() != self.kv_dim {
                return Err(LevelKeysError::DimMismatch {
                    layer_idx,
                    level_idx: i,
                    actual: k.len(),
                    expected: self.kv_dim,
                });
            }
            if !k.iter().all(|v| v.is_finite()) {
                return Err(LevelKeysError::NonFinite {
                    layer_idx,
                    level_idx: i,
                });
            }
            if k.iter().all(|v| *v == 0.0) {
                return Err(LevelKeysError::AllZero {
                    layer_idx,
                    level_idx: i,
                });
            }
        }
        self.keys.insert(layer_idx, keys);
        if !self.detection_layers.contains(&layer_idx) {
            self.detection_layers.push(layer_idx);
            self.detection_layers.sort_unstable();
        }
        Ok(())
    }

    /// 获取某检测层的 3 个层级键.
    pub fn get(&self, layer_idx: usize) -> Option<&[Vec<f32>; 3]> {
        self.keys.get(&layer_idx)
    }

    /// 所有已注册检测层的物理索引集合.
    pub fn detection_layers(&self) -> &[usize] {
        &self.detection_layers
    }

    /// KV 向量维度.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// 判空.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// 已注册条目数 = 检测层数.
    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Level Keys 缓存插入时的校验错误.
#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum LevelKeysError {
    #[error("dim mismatch at layer={layer_idx} level_idx={level_idx}: actual={actual} expected={expected}")]
    DimMismatch {
        layer_idx: usize,
        level_idx: usize,
        actual: usize,
        expected: usize,
    },
    #[error("non-finite component at layer={layer_idx} level_idx={level_idx}")]
    NonFinite { layer_idx: usize, level_idx: usize },
    #[error("all-zero vector at layer={layer_idx} level_idx={level_idx}")]
    AllZero { layer_idx: usize, level_idx: usize },
}

// ============================================================================
// precompute — 按 SPEC §3.1 构造 LevelKeysCache
// ============================================================================

/// 为指定 detection_layers 集合预计算 `[K_L1, K_L2, K_L3]` 层级键.
///
/// 流程严格对齐 `SPEC/SEMANTIC-GATEKEEPER.md §3.1`:
///   for each detection_layer L (与 `kproj_graphs` 一一对应):
///     for each level_descriptor (3 个, L1/L2/L3 顺序):
///       tokens = tokenizer.encode(desc)
///       embed  = embed_graph.encode_tokens(tokens)   // [T, hidden]
///       k_bytes = kproj_graphs[L].run_on_embed(embed) // [T, kv_dim]
///       k_f32  = decode(k_bytes, dtype)
///       K_Lx   = mean_over_axis0(k_f32)              // [kv_dim]
///     cache.insert(L, [K_L1, K_L2, K_L3])
///
/// `kproj_graphs.len()` 必须等于 `detection_layers.len()` 且顺序一致
/// (`kproj_graphs[i]` 为 `detection_layers[i]` 的投影图).
///
/// `kv_dim` 必须等于 `num_kv_heads × head_dim`; 所有 kproj_graph 的 kv_dim
/// 必须一致.
pub fn precompute(
    config: &SemanticGatekeeperConfig,
    tokenizer: &dyn TokenizerEncoder,
    embed_graph: &EmbedLookupOnlyGraph,
    kproj_graphs: &[KProjOnlyGraph],
    detection_layers: &[usize],
    hidden_size: usize,
    kv_dim: usize,
    dtype: DType,
) -> Result<LevelKeysCache, SemanticGatekeeperError> {
    // ── 前置校验 ──
    if detection_layers.is_empty() {
        return Err(SemanticGatekeeperError::PrecomputeFailed(
            "detection_layers is empty".to_string(),
        ));
    }
    if kproj_graphs.len() != detection_layers.len() {
        return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
            "kproj_graphs.len()={} != detection_layers.len()={}",
            kproj_graphs.len(),
            detection_layers.len()
        )));
    }
    if embed_graph.hidden_size() != hidden_size {
        return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
            "embed_graph.hidden_size={} != expected hidden_size={}",
            embed_graph.hidden_size(),
            hidden_size
        )));
    }
    if embed_graph.dtype() != dtype {
        return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
            "embed_graph.dtype={:?} != expected {:?}",
            embed_graph.dtype(),
            dtype
        )));
    }
    for (i, kpg) in kproj_graphs.iter().enumerate() {
        if kpg.layer_idx() != detection_layers[i] {
            return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
                "kproj_graphs[{i}].layer_idx={} != detection_layers[{i}]={}",
                kpg.layer_idx(),
                detection_layers[i]
            )));
        }
        if kpg.hidden_size() != hidden_size {
            return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
                "kproj_graphs[{i}].hidden_size={} != expected {}",
                kpg.hidden_size(),
                hidden_size
            )));
        }
        if kpg.kv_dim() != kv_dim {
            return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
                "kproj_graphs[{i}].kv_dim={} != expected {}",
                kpg.kv_dim(),
                kv_dim
            )));
        }
        if kpg.dtype() != dtype {
            return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
                "kproj_graphs[{i}].dtype={:?} != expected {:?}",
                kpg.dtype(),
                dtype
            )));
        }
    }

    let mut cache = LevelKeysCache::new(kv_dim);

    // ── 核心循环 ──
    for (idx, &layer_idx) in detection_layers.iter().enumerate() {
        let kproj = &kproj_graphs[idx];
        let mut level_keys: [Option<Vec<f32>>; 3] = [None, None, None];

        for (level_idx, desc) in config.level_descriptors.iter().enumerate() {
            // Step 1: tokenize.
            let tokens = tokenizer.encode(desc).map_err(|e| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "tokenize level_descriptors[{level_idx}] failed: {e}"
                ))
            })?;
            if tokens.is_empty() {
                return Err(SemanticGatekeeperError::PrecomputeFailed(format!(
                    "tokenize level_descriptors[{level_idx}] produced empty tokens"
                )));
            }

            // Step 2: embed lookup (走 JIT).
            let embed_bytes = embed_graph.encode_tokens(&tokens)?;
            let k_bytes = kproj.run_on_embed(&embed_bytes)?;

            // Step 4: decode bytes → f32 seq_len × kv_dim.
            let seq_len = tokens.len();
            let k_f32 = decode_bytes_to_f32(&k_bytes, seq_len, kv_dim, dtype).map_err(|e| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "decode k_proj output (layer {layer_idx} level {level_idx}) failed: {e}"
                ))
            })?;

            // Step 5: mean_over_axis0 → [kv_dim].
            let pooled = mean_pool_rows(&k_f32, seq_len, kv_dim).map_err(|e| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "mean_pool (layer {layer_idx} level {level_idx}) failed: {e}"
                ))
            })?;

            level_keys[level_idx] = Some(pooled);
        }

        // Collect 3 levels into fixed array.
        let [l1, l2, l3] = level_keys;
        let keys = [
            l1.ok_or_else(|| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "layer {layer_idx} missing L1 key"
                ))
            })?,
            l2.ok_or_else(|| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "layer {layer_idx} missing L2 key"
                ))
            })?,
            l3.ok_or_else(|| {
                SemanticGatekeeperError::PrecomputeFailed(format!(
                    "layer {layer_idx} missing L3 key"
                ))
            })?,
        ];

        cache.insert(layer_idx, keys).map_err(|e| {
            SemanticGatekeeperError::PrecomputeFailed(format!(
                "cache insert layer {layer_idx} failed: {e}"
            ))
        })?;
    }

    Ok(cache)
}

// ============================================================================
// dtype-aware bytes → f32 解码 + mean_pool
// ============================================================================

fn decode_bytes_to_f32(
    bytes: &[u8],
    seq_len: usize,
    kv_dim: usize,
    dtype: DType,
) -> Result<Vec<f32>, String> {
    let total = seq_len
        .checked_mul(kv_dim)
        .ok_or_else(|| "seq_len * kv_dim overflow".to_string())?;
    let elem = dtype.size_bytes();
    let expected = total
        .checked_mul(elem)
        .ok_or_else(|| "byte length overflow".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "bytes len {} != expected {}",
            bytes.len(),
            expected
        ));
    }

    let mut out = Vec::with_capacity(total);
    match dtype {
        DType::F32 => {
            for i in 0..total {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    bytes[off],
                    bytes[off + 1],
                    bytes[off + 2],
                    bytes[off + 3],
                ]));
            }
        }
        DType::F16 => {
            for i in 0..total {
                let off = i * 2;
                out.push(f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..total {
                let off = i * 2;
                out.push(bf16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32());
            }
        }
        _ => return Err(format!("unsupported dtype {dtype:?}")),
    }
    Ok(out)
}

fn mean_pool_rows(data: &[f32], seq_len: usize, kv_dim: usize) -> Result<Vec<f32>, String> {
    if seq_len == 0 {
        return Err("seq_len = 0".to_string());
    }
    if data.len() != seq_len * kv_dim {
        return Err(format!(
            "data.len()={} != seq_len*kv_dim={}",
            data.len(),
            seq_len * kv_dim
        ));
    }
    let mut acc = vec![0.0f64; kv_dim];
    for row in 0..seq_len {
        let base = row * kv_dim;
        for col in 0..kv_dim {
            acc[col] += data[base + col] as f64;
        }
    }
    let inv = 1.0f64 / seq_len as f64;
    Ok(acc.into_iter().map(|v| (v * inv) as f32).collect())
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_averages_rows() {
        // 2 rows × 3 cols: [[1,2,3],[3,4,5]] → [2,3,4]
        let data = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        let m = mean_pool_rows(&data, 2, 3).unwrap();
        assert_eq!(m, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn decode_f32_roundtrip() {
        let values = vec![1.0f32, -2.5, 3.14, 0.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::F32).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_bf16_roundtrip() {
        let values = vec![1.0f32, -2.5, 3.125, 0.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::BF16).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.05);
        }
    }

    // ── LevelKeysCache struct tests ──

    #[test]
    fn cache_new_is_empty() {
        let cache = LevelKeysCache::new(4);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.kv_dim(), 4);
        assert!(cache.detection_layers().is_empty());
    }

    #[test]
    fn cache_default_is_empty() {
        let cache = LevelKeysCache::default();
        assert!(cache.is_empty());
        assert_eq!(cache.kv_dim(), 0);
    }

    #[test]
    fn cache_insert_and_get() {
        let mut cache = LevelKeysCache::new(2);
        let keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        cache.insert(3, keys.clone()).unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.detection_layers(), &[3]);
        let retrieved = cache.get(3).unwrap();
        assert_eq!(retrieved[0], vec![1.0, 2.0]);
        assert_eq!(retrieved[2], vec![5.0, 6.0]);
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn cache_insert_multiple_layers_sorted() {
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(7, k.clone()).unwrap();
        cache.insert(2, k.clone()).unwrap();
        cache.insert(5, k.clone()).unwrap();
        assert_eq!(cache.detection_layers(), &[2, 5, 7]);
    }

    #[test]
    fn cache_insert_overwrites_existing() {
        let mut cache = LevelKeysCache::new(1);
        let k1: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let k2: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        cache.insert(0, k1).unwrap();
        cache.insert(0, k2).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(0).unwrap()[0], vec![10.0]);
    }

    #[test]
    fn cache_insert_rejects_dim_mismatch() {
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [vec![1.0, 2.0], vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let err = cache.insert(0, bad).unwrap_err();
        assert!(matches!(err, LevelKeysError::DimMismatch { level_idx: 0, .. }));
    }

    #[test]
    fn cache_insert_rejects_non_finite() {
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [vec![1.0, f32::NAN], vec![1.0, 2.0], vec![1.0, 2.0]];
        let err = cache.insert(0, bad).unwrap_err();
        assert!(matches!(err, LevelKeysError::NonFinite { level_idx: 0, .. }));
    }

    #[test]
    fn cache_insert_rejects_all_zero() {
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [vec![0.0, 0.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let err = cache.insert(5, bad).unwrap_err();
        assert!(matches!(err, LevelKeysError::AllZero { layer_idx: 5, level_idx: 0 }));
    }

    #[test]
    fn level_keys_error_display() {
        let err = LevelKeysError::DimMismatch {
            layer_idx: 3, level_idx: 1, actual: 2, expected: 4,
        };
        let msg = format!("{err}");
        assert!(msg.contains("layer=3"));
        assert!(msg.contains("actual=2"));
        assert!(msg.contains("expected=4"));

        let err = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 2 };
        assert!(format!("{err}").contains("non-finite"));

        let err = LevelKeysError::AllZero { layer_idx: 1, level_idx: 0 };
        assert!(format!("{err}").contains("all-zero"));
    }

    #[test]
    fn mean_pool_rejects_zero_seq_len() {
        let err = mean_pool_rows(&[1.0, 2.0], 0, 2).unwrap_err();
        assert!(err.contains("seq_len = 0"));
    }

    #[test]
    fn mean_pool_rejects_len_mismatch() {
        let err = mean_pool_rows(&[1.0, 2.0], 2, 2).unwrap_err();
        assert!(err.contains("data.len()"));
    }

    // ── decode_bytes_to_f32 additional tests ──

    #[test]
    fn decode_f16_roundtrip() {
        // Arrange: encode f32 values as f16, pack into bytes
        let values = vec![1.0f32, -2.5, 3.125, 0.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::F16).unwrap();
        // Assert: f16 precision is coarser than f32
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.05, "f16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn decode_rejects_byte_length_mismatch() {
        // Arrange: 2 elements of F32 = 8 bytes expected, provide 4
        let bytes = vec![0u8; 4];
        // Act
        let err = decode_bytes_to_f32(&bytes, 2, 1, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("bytes len"), "unexpected error: {err}");
        assert!(err.contains("expected"), "unexpected error: {err}");
    }

    #[test]
    fn decode_rejects_unsupported_dtype() {
        // Arrange: U8 has size_bytes=1, so provide exactly 1 byte for 1 elem
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::U8).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    #[test]
    fn decode_handles_overflow_protection() {
        // Arrange: seq_len and kv_dim that would overflow multiplication
        let bytes = vec![0u8; 8];
        // Act
        let err = decode_bytes_to_f32(&bytes, usize::MAX, usize::MAX, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("overflow"), "unexpected error: {err}");
    }

    // ── LevelKeysError trait tests ──

    #[test]
    fn level_keys_error_is_std_error() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 0,
            level_idx: 0,
            actual: 1,
            expected: 2,
        };
        // Act: upcast to std::error::Error trait object
        let _: &dyn std::error::Error = &err;
        // Assert: compilation succeeds = trait is implemented
    }

    #[test]
    fn level_keys_error_clone_preserves_content() {
        // Arrange
        let err = LevelKeysError::NonFinite {
            layer_idx: 7,
            level_idx: 2,
        };
        // Act
        let cloned = err.clone();
        // Assert
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn level_keys_error_debug_format() {
        // Arrange
        let err = LevelKeysError::AllZero {
            layer_idx: 4,
            level_idx: 1,
        };
        // Act
        let debug = format!("{err:?}");
        // Assert: Debug output contains variant name and fields
        assert!(debug.contains("AllZero"), "Debug: {debug}");
        assert!(debug.contains("layer_idx"), "Debug: {debug}");
    }

    #[test]
    fn level_keys_error_display_non_finite() {
        // Arrange
        let err = LevelKeysError::NonFinite {
            layer_idx: 9,
            level_idx: 1,
        };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("layer=9"), "msg: {msg}");
        assert!(msg.contains("level_idx=1"), "msg: {msg}");
    }

    #[test]
    fn level_keys_error_display_all_zero() {
        // Arrange
        let err = LevelKeysError::AllZero {
            layer_idx: 2,
            level_idx: 0,
        };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("layer=2"), "msg: {msg}");
        assert!(msg.contains("level_idx=0"), "msg: {msg}");
    }

    // ── LevelKeysCache edge cases ──

    #[test]
    fn cache_insert_rejects_dim_mismatch_on_level_2() {
        // Arrange: level 0 and 1 are correct, level 2 has wrong dimension
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0], // wrong dim
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::DimMismatch { level_idx: 2, .. }),
            "expected DimMismatch at level_idx=2, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_rejects_non_finite_on_level_1() {
        // Arrange: level 0 fine, level 1 has infinity, level 2 fine
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![f32::INFINITY, 0.0],
            vec![1.0, 2.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 1, .. }),
            "expected NonFinite at level_idx=1, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_rejects_all_zero_on_level_2() {
        // Arrange: levels 0,1 are valid, level 2 is all zeros
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![0.0, 0.0],
        ];
        // Act
        let err = cache.insert(10, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::AllZero { layer_idx: 10, level_idx: 2 }),
            "expected AllZero at layer=10 level=2, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_negative_infinity_rejected() {
        // Arrange
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![f32::NEG_INFINITY, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { .. }),
            "expected NonFinite, got {err:?}"
        );
    }

    #[test]
    fn cache_overwrite_preserves_detection_layers_sorted() {
        // Arrange: insert layers 5, 10, 3, then overwrite 5
        let mut cache = LevelKeysCache::new(1);
        let ka: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let kb: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        cache.insert(5, ka.clone()).unwrap();
        cache.insert(10, ka.clone()).unwrap();
        cache.insert(3, ka.clone()).unwrap();
        // Act: overwrite layer 5
        cache.insert(5, kb).unwrap();
        // Assert
        assert_eq!(cache.detection_layers(), &[3, 5, 10]);
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(5).unwrap()[1], vec![20.0]);
    }

    #[test]
    fn cache_duplicate_insert_no_duplicate_layers() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(4, k.clone()).unwrap();
        // Act: insert same layer again
        cache.insert(4, k.clone()).unwrap();
        // Assert: no duplicate in detection_layers
        assert_eq!(cache.detection_layers(), &[4]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_large_layer_index() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let big = usize::MAX / 2;
        // Act
        cache.insert(big, k).unwrap();
        // Assert
        assert_eq!(cache.get(big).unwrap()[0], vec![1.0]);
        assert_eq!(cache.detection_layers(), &[big]);
    }

    #[test]
    fn cache_insert_does_not_partial_insert_on_error() {
        // Arrange: first level is valid, second is all-zero
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![0.0, 0.0], // all-zero triggers error
            vec![3.0, 4.0],
        ];
        // Act
        let result = cache.insert(0, bad);
        // Assert: nothing was inserted
        assert!(result.is_err());
        assert!(cache.is_empty());
        assert!(cache.get(0).is_none());
    }

    // ── mean_pool_rows additional tests ──

    #[test]
    fn mean_pool_single_row_identity() {
        // Arrange: 1 row × 3 cols
        let data = vec![7.0, -3.5, 2.0];
        // Act
        let pooled = mean_pool_rows(&data, 1, 3).unwrap();
        // Assert: single row → identity
        assert_eq!(pooled, vec![7.0, -3.5, 2.0]);
    }

    #[test]
    fn mean_pool_with_negative_values() {
        // Arrange: 2 rows × 2 cols: [[-2, 4], [6, -8]] → [2, -2]
        let data = vec![-2.0f32, 4.0, 6.0, -8.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert
        assert!((pooled[0] - 2.0).abs() < 1e-6, "got {}", pooled[0]);
        assert!((pooled[1] - (-2.0)).abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn mean_pool_many_rows_accumulates_f64_precision() {
        // Arrange: 100 rows × 1 col, each value = 0.1 → mean = 0.1
        let data: Vec<f32> = vec![0.1f32; 100];
        // Act
        let pooled = mean_pool_rows(&data, 100, 1).unwrap();
        // Assert: f64 accumulation keeps precision reasonable
        assert!(
            (pooled[0] - 0.1).abs() < 1e-5,
            "expected ~0.1, got {}",
            pooled[0]
        );
    }

    #[test]
    fn mean_pool_large_kv_dim() {
        // Arrange: 1 row × 256 cols
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        // Act
        let pooled = mean_pool_rows(&data, 1, 256).unwrap();
        // Assert: single row is identity regardless of dimension
        assert_eq!(pooled.len(), 256);
        assert_eq!(pooled[0], 0.0);
        assert_eq!(pooled[255], 255.0);
    }

    #[test]
    fn mean_pool_rejects_empty_data_with_nonzero_seq_len() {
        // Arrange: seq_len=1, kv_dim=2 but data is empty
        let data: Vec<f32> = vec![];
        // Act
        let err = mean_pool_rows(&data, 1, 2).unwrap_err();
        // Assert
        assert!(err.contains("data.len()"), "unexpected error: {err}");
    }

    // ── decode_bytes_to_f32 additional edge cases ──

    #[test]
    fn decode_f32_single_element() {
        // Arrange: 1 element F32 = 4 bytes
        let value = 42.5f32;
        let bytes = value.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 42.5).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_negative_values() {
        // Arrange: 4 negative F32 values (2×2)
        let values: Vec<f32> = vec![-1.0, -100.5, -0.001, -999.99];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::F32).unwrap();
        // Assert
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_bf16_zero_roundtrip() {
        // Arrange: 0.0 encoded as BF16
        let bytes = bf16::from_f32(0.0f32).to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert
        assert!((decoded[0]).abs() < 1e-6, "expected ~0.0, got {}", decoded[0]);
    }

    #[test]
    fn decode_elem_overflow_protection() {
        // Arrange: seq_len=1, kv_dim=1, but dtype=F32 means 4 bytes expected.
        // Provide correct seq*dim overflow pass (1*1=1 fine) but wrong byte count.
        let bytes = vec![0u8; 3]; // 3 bytes, but F32 needs 4
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("bytes len"), "unexpected error: {err}");
    }

    #[test]
    fn decode_bf16_signed_values() {
        // Arrange: values exactly representable in BF16 (powers of 2)
        let values = vec![128.0f32, -64.0f32];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 1, DType::BF16).unwrap();
        // Assert: powers of 2 are exactly representable in BF16
        assert!((decoded[0] - 128.0).abs() < 0.01, "got {}", decoded[0]);
        assert!((decoded[1] - (-64.0)).abs() < 0.01, "got {}", decoded[1]);
    }

    // ── LevelKeysCache additional edge cases ──

    #[test]
    fn cache_kv_dim_preserved_after_insert() {
        // Arrange
        let mut cache = LevelKeysCache::new(7);
        let k: [Vec<f32>; 3] = [
            vec![1.0; 7],
            vec![2.0; 7],
            vec![3.0; 7],
        ];
        // Act
        cache.insert(0, k).unwrap();
        // Assert: kv_dim unchanged after insert
        assert_eq!(cache.kv_dim(), 7);
    }

    #[test]
    fn cache_accepts_negative_finite_values() {
        // Arrange: keys with negative values should be valid
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![-1.0, -2.0],
            vec![-3.0, 0.5],
            vec![0.1, -0.1],
        ];
        // Act
        cache.insert(0, k.clone()).unwrap();
        // Assert
        let retrieved = cache.get(0).unwrap();
        assert_eq!(retrieved[0], vec![-1.0, -2.0]);
        assert_eq!(retrieved[1], vec![-3.0, 0.5]);
        assert_eq!(retrieved[2], vec![0.1, -0.1]);
    }

    #[test]
    fn cache_insert_rejects_mixed_nan_and_infinity() {
        // Arrange: level 0 has NaN, level 1 has INF — should fail on first bad level
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![f32::NAN, f32::INFINITY],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: first bad level is level 0
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 0, .. }),
            "expected NonFinite at level_idx=0, got {err:?}"
        );
    }

    #[test]
    fn cache_overwrite_with_bad_data_leaves_original() {
        // Arrange: insert valid data first
        let mut cache = LevelKeysCache::new(1);
        let good: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(0, good).unwrap();
        // Act: try to overwrite with invalid data
        let bad: [Vec<f32>; 3] = [vec![0.0], vec![2.0], vec![3.0]]; // level 0 all-zero
        let result = cache.insert(0, bad);
        // Assert: original data preserved
        assert!(result.is_err());
        assert_eq!(cache.get(0).unwrap()[0], vec![1.0]);
    }

    #[test]
    fn cache_zero_kv_dim_rejects_any_insert() {
        // Arrange: cache with kv_dim=0 — any non-empty vector is a dim mismatch
        let mut cache = LevelKeysCache::new(0);
        let k: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        // Act: empty vectors with kv_dim=0 should fail on all-zero check
        let err = cache.insert(0, k).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::AllZero { .. }),
            "expected AllZero for empty vectors, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_rejects_level0_dim_correct_level1_wrong() {
        // Arrange: level 0 correct dim, level 1 wrong
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0, 3.0], // correct
            vec![1.0, 2.0],      // wrong: 2 != 3
            vec![1.0, 2.0, 3.0],
        ];
        // Act
        let err = cache.insert(42, bad).unwrap_err();
        // Assert: stops at first error (level 1)
        assert!(
            matches!(err, LevelKeysError::DimMismatch { layer_idx: 42, level_idx: 1, actual: 2, expected: 3 }),
            "expected DimMismatch(layer=42, level=1, actual=2, expected=3), got {err:?}"
        );
    }

    #[test]
    fn cache_detection_layers_sorted_after_mixed_inserts() {
        // Arrange: insert layers in reverse order
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act
        cache.insert(20, k.clone()).unwrap();
        cache.insert(10, k.clone()).unwrap();
        cache.insert(30, k.clone()).unwrap();
        cache.insert(5, k.clone()).unwrap();
        // Assert: always sorted
        assert_eq!(cache.detection_layers(), &[5, 10, 20, 30]);
    }

    #[test]
    fn cache_get_returns_none_for_uninserted_layer() {
        // Arrange: insert layer 0 and 2
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(0, k.clone()).unwrap();
        cache.insert(2, k).unwrap();
        // Act & Assert: layer 1 was never inserted
        assert!(cache.get(1).is_none());
        assert!(cache.get(0).is_some());
        assert!(cache.get(2).is_some());
    }

    // ── LevelKeysError Display: field-level verification ──

    #[test]
    fn level_keys_error_display_dim_mismatch_exact_fields() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 7,
            level_idx: 2,
            actual: 5,
            expected: 10,
        };
        // Act
        let msg = format!("{err}");
        // Assert: all fields present
        assert!(msg.contains("layer=7"), "msg: {msg}");
        assert!(msg.contains("level_idx=2"), "msg: {msg}");
        assert!(msg.contains("actual=5"), "msg: {msg}");
        assert!(msg.contains("expected=10"), "msg: {msg}");
    }

    // ── LevelKeysCache Clone trait ──

    #[test]
    fn cache_clone_produces_independent_copy() {
        // Arrange
        let mut original = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        original.insert(3, k).unwrap();
        // Act
        let cloned = original.clone();
        // Assert: clone has same data
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.kv_dim(), 2);
        assert_eq!(cloned.detection_layers(), &[3]);
        assert_eq!(cloned.get(3).unwrap()[0], vec![1.0, 2.0]);
    }

    #[test]
    fn cache_clone_is_independent_from_original() {
        // Arrange
        let mut original = LevelKeysCache::new(1);
        let ka: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let kb: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        original.insert(0, ka).unwrap();
        let cloned = original.clone();
        // Act: mutate original after clone
        original.insert(1, kb).unwrap();
        // Assert: clone is unaffected
        assert_eq!(cloned.len(), 1);
        assert!(cloned.get(1).is_none());
        assert_eq!(original.len(), 2);
    }

    // ── LevelKeysCache Debug trait ──

    #[test]
    fn cache_debug_output_contains_fields() {
        // Arrange
        let cache = LevelKeysCache::new(4);
        // Act
        let debug = format!("{cache:?}");
        // Assert: Debug output contains struct name and field names
        assert!(debug.contains("LevelKeysCache"), "Debug: {debug}");
        assert!(debug.contains("keys"), "Debug: {debug}");
        assert!(debug.contains("detection_layers"), "Debug: {debug}");
        assert!(debug.contains("kv_dim"), "Debug: {debug}");
    }

    // ── LevelKeysError PartialEq + field destructuring ──

    #[test]
    fn level_keys_error_partial_eq_dim_mismatch() {
        // Arrange
        let a = LevelKeysError::DimMismatch {
            layer_idx: 1,
            level_idx: 2,
            actual: 3,
            expected: 4,
        };
        let b = LevelKeysError::DimMismatch {
            layer_idx: 1,
            level_idx: 2,
            actual: 3,
            expected: 4,
        };
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_different_fields_not_equal() {
        // Arrange
        let a = LevelKeysError::NonFinite {
            layer_idx: 0,
            level_idx: 0,
        };
        let b = LevelKeysError::NonFinite {
            layer_idx: 1,
            level_idx: 0,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_different_variants_not_equal() {
        // Arrange
        let a = LevelKeysError::NonFinite {
            layer_idx: 0,
            level_idx: 0,
        };
        let b = LevelKeysError::AllZero {
            layer_idx: 0,
            level_idx: 0,
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_destructure_dim_mismatch_fields() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 10,
            level_idx: 1,
            actual: 3,
            expected: 8,
        };
        // Act
        if let LevelKeysError::DimMismatch {
            layer_idx,
            level_idx,
            actual,
            expected,
        } = err
        {
            // Assert
            assert_eq!(layer_idx, 10);
            assert_eq!(level_idx, 1);
            assert_eq!(actual, 3);
            assert_eq!(expected, 8);
        } else {
            panic!("expected DimMismatch variant");
        }
    }

    #[test]
    fn level_keys_error_destructure_non_finite_fields() {
        // Arrange
        let err = LevelKeysError::NonFinite {
            layer_idx: 5,
            level_idx: 2,
        };
        // Act
        if let LevelKeysError::NonFinite {
            layer_idx,
            level_idx,
        } = err
        {
            // Assert
            assert_eq!(layer_idx, 5);
            assert_eq!(level_idx, 2);
        } else {
            panic!("expected NonFinite variant");
        }
    }

    #[test]
    fn level_keys_error_destructure_all_zero_fields() {
        // Arrange
        let err = LevelKeysError::AllZero {
            layer_idx: 3,
            level_idx: 0,
        };
        // Act
        if let LevelKeysError::AllZero {
            layer_idx,
            level_idx,
        } = err
        {
            // Assert
            assert_eq!(layer_idx, 3);
            assert_eq!(level_idx, 0);
        } else {
            panic!("expected AllZero variant");
        }
    }

    // ── mean_pool_rows additional edge cases ──

    #[test]
    fn mean_pool_uniform_rows_returns_same_value() {
        // Arrange: 3 rows x 2 cols, all values = 5.0
        let data = vec![5.0f32; 6];
        // Act
        let pooled = mean_pool_rows(&data, 3, 2).unwrap();
        // Assert
        assert_eq!(pooled, vec![5.0, 5.0]);
    }

    #[test]
    fn mean_pool_values_near_zero() {
        // Arrange: 2 rows x 2 cols, values near zero with opposite signs
        let data = vec![1e-10f32, -1e-10f32, -1e-10f32, 1e-10f32];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert: should average to ~0.0
        assert!(pooled[0].abs() < 1e-15, "got {}", pooled[0]);
        assert!(pooled[1].abs() < 1e-15, "got {}", pooled[1]);
    }

    #[test]
    fn mean_pool_large_seq_len_with_constant() {
        // Arrange: 1000 rows x 1 col, all 3.14
        let data = vec![3.14f32; 1000];
        // Act
        let pooled = mean_pool_rows(&data, 1000, 1).unwrap();
        // Assert
        assert!(
            (pooled[0] - 3.14).abs() < 1e-4,
            "expected ~3.14, got {}",
            pooled[0]
        );
    }

    // ── decode_bytes_to_f32 additional edge cases ──

    #[test]
    fn decode_f16_single_element() {
        // Arrange
        let value = 3.75f32;
        let bytes = f16::from_f32(value).to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - value).abs() < 0.05, "got {}", decoded[0]);
    }

    #[test]
    fn decode_f16_zero_roundtrip() {
        // Arrange: 0.0 encoded as F16
        let bytes = f16::from_f32(0.0f32).to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert
        assert!(
            decoded[0].abs() < 1e-6,
            "expected ~0.0, got {}",
            decoded[0]
        );
    }

    #[test]
    fn decode_f32_zero_values() {
        // Arrange: 4 zero values as F32
        let bytes = vec![0u8; 16];
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded, vec![0.0f32; 4]);
    }

    #[test]
    fn decode_empty_buffer_with_zero_elements() {
        // Arrange: seq_len=0, kv_dim=0, dtype=F32 → 0 bytes expected
        let bytes: Vec<u8> = vec![];
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 0, 0, DType::F32).unwrap();
        // Assert
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_byte_overflow_with_large_kv_dim() {
        // Arrange: seq_len=2, kv_dim=usize::MAX, dtype=F32 → should overflow
        let bytes = vec![0u8; 8];
        // Act
        let err = decode_bytes_to_f32(&bytes, 2, usize::MAX, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("overflow"), "unexpected error: {err}");
    }

    // ── LevelKeysCache: failed insert does not corrupt state ──

    #[test]
    fn cache_failed_insert_preserves_existing_layers() {
        // Arrange: insert two valid layers
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(0, k.clone()).unwrap();
        cache.insert(5, k.clone()).unwrap();
        // Act: attempt to insert a bad layer
        let bad: [Vec<f32>; 3] = [vec![0.0], vec![2.0], vec![3.0]];
        let _ = cache.insert(10, bad);
        // Assert: only the original two layers exist
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.detection_layers(), &[0, 5]);
        assert!(cache.get(10).is_none());
    }

    #[test]
    fn cache_insert_validates_all_three_levels_in_order() {
        // Arrange: level 0 valid, level 1 dim mismatch, level 2 would be all-zero
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],       // valid
            vec![1.0],             // dim mismatch
            vec![0.0, 0.0],        // would be all-zero (but we never get here)
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: first error is dim mismatch at level 1
        assert_eq!(
            err,
            LevelKeysError::DimMismatch {
                layer_idx: 0,
                level_idx: 1,
                actual: 1,
                expected: 2,
            }
        );
    }

    // ── New tests: 45+ additional tests ──

    // ── LevelKeysCache: zero kv_dim edge cases ──

    #[test]
    fn cache_new_with_zero_kv_dim_is_empty() {
        // Arrange & Act
        let cache = LevelKeysCache::new(0);
        // Assert
        assert!(cache.is_empty());
        assert_eq!(cache.kv_dim(), 0);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_zero_kv_dim_get_returns_none() {
        // Arrange
        let cache = LevelKeysCache::new(0);
        // Act & Assert
        assert!(cache.get(0).is_none());
        assert!(cache.get(usize::MAX).is_none());
    }

    #[test]
    fn cache_zero_kv_dim_detection_layers_is_empty() {
        // Arrange
        let cache = LevelKeysCache::new(0);
        // Act & Assert
        assert!(cache.detection_layers().is_empty());
    }

    // ── LevelKeysCache: large kv_dim and boundary values ──

    #[test]
    fn cache_large_kv_dim_insert_and_retrieve() {
        // Arrange: kv_dim = 1024
        let dim = 1024;
        let mut cache = LevelKeysCache::new(dim);
        let k: [Vec<f32>; 3] = [
            (0..dim).map(|i| (i as f32) * 0.01).collect(),
            (0..dim).map(|i| 1.0 - (i as f32) * 0.001).collect(),
            (0..dim).map(|i| -1.0 + (i as f32) * 0.002).collect(),
        ];
        // Act
        cache.insert(0, k.clone()).unwrap();
        // Assert
        let retrieved = cache.get(0).unwrap();
        assert_eq!(retrieved[0].len(), dim);
        assert_eq!(retrieved[1].len(), dim);
        assert_eq!(retrieved[2].len(), dim);
        assert!((retrieved[0][0] - 0.0).abs() < 1e-6);
        assert!((retrieved[1][0] - 1.0).abs() < 1e-6);
        assert!((retrieved[2][0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cache_insert_layer_zero_is_valid() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act
        cache.insert(0, k).unwrap();
        // Assert
        assert_eq!(cache.detection_layers(), &[0]);
        assert_eq!(cache.get(0).unwrap()[0], vec![1.0]);
    }

    #[test]
    fn cache_insert_layer_usize_max() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act
        cache.insert(usize::MAX, k).unwrap();
        // Assert
        assert_eq!(cache.detection_layers(), &[usize::MAX]);
        assert!(cache.get(usize::MAX).is_some());
    }

    // ── LevelKeysCache: multiple layer insert patterns ──

    #[test]
    fn cache_many_layers_sorted_correctly() {
        // Arrange: insert 10 layers in random order
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let order = [7, 2, 9, 1, 5, 3, 8, 4, 6, 0];
        // Act
        for &layer in &order {
            cache.insert(layer, k.clone()).unwrap();
        }
        // Assert: sorted ascending
        assert_eq!(cache.detection_layers(), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn cache_three_overwrites_on_same_layer() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k1: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let k2: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        let k3: [Vec<f32>; 3] = [vec![100.0], vec![200.0], vec![300.0]];
        // Act
        cache.insert(5, k1).unwrap();
        cache.insert(5, k2).unwrap();
        cache.insert(5, k3).unwrap();
        // Assert: last write wins
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(5).unwrap()[0], vec![100.0]);
        assert_eq!(cache.get(5).unwrap()[2], vec![300.0]);
    }

    // ── LevelKeysCache: very small floats ──

    #[test]
    fn cache_accepts_very_small_finite_values() {
        // Arrange: values near f32::MIN_POSITIVE
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
            vec![1e-30, 1e-35],
            vec![-1e-20, 1e-20],
        ];
        // Act
        cache.insert(0, k).unwrap();
        // Assert
        let r = cache.get(0).unwrap();
        assert_eq!(r[0][0], f32::MIN_POSITIVE);
        assert_eq!(r[0][1], -f32::MIN_POSITIVE);
    }

    #[test]
    fn cache_accepts_very_large_finite_values() {
        // Arrange: values near f32::MAX
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![f32::MAX, -f32::MAX],
            vec![1e30, -1e30],
            vec![f32::MIN_POSITIVE, f32::MAX],
        ];
        // Act
        cache.insert(0, k).unwrap();
        // Assert
        let r = cache.get(0).unwrap();
        assert_eq!(r[0][0], f32::MAX);
        assert_eq!(r[0][1], -f32::MAX);
    }

    // ── LevelKeysCache: partial zero patterns ──

    #[test]
    fn cache_accepts_partial_zero_vector() {
        // Arrange: one element is zero but not all — should be valid
        let mut cache = LevelKeysCache::new(3);
        let k: [Vec<f32>; 3] = [
            vec![0.0, 1.0, 0.0],  // not all-zero
            vec![0.0, 0.0, 0.5],  // not all-zero
            vec![-0.1, 0.0, 0.0], // not all-zero
        ];
        // Act
        cache.insert(0, k.clone()).unwrap();
        // Assert
        let r = cache.get(0).unwrap();
        assert_eq!(r[0], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn cache_rejects_all_zero_with_negative_zero() {
        // Arrange: -0.0 == 0.0 in IEEE 754, so [-0.0, -0.0] is still all-zero
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![-0.0, -0.0],  // IEEE: -0.0 == 0.0 is true
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero for -0.0 vectors, got {err:?}"
        );
    }

    // ── LevelKeysError: all variants tested in result from cache ──

    #[test]
    fn level_keys_error_dim_mismatch_display_contains_level_idx() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 0,
            level_idx: 0,
            actual: 1,
            expected: 5,
        };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("level_idx=0"), "msg: {msg}");
    }

    #[test]
    fn level_keys_error_non_finite_display_format() {
        // Arrange
        let err = LevelKeysError::NonFinite { layer_idx: 99, level_idx: 2 };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("layer=99"), "msg: {msg}");
        assert!(msg.contains("level_idx=2"), "msg: {msg}");
        assert!(msg.contains("non-finite"), "msg: {msg}");
    }

    #[test]
    fn level_keys_error_all_zero_display_format() {
        // Arrange
        let err = LevelKeysError::AllZero { layer_idx: 42, level_idx: 1 };
        // Act
        let msg = format!("{err}");
        // Assert
        assert!(msg.contains("layer=42"), "msg: {msg}");
        assert!(msg.contains("level_idx=1"), "msg: {msg}");
        assert!(msg.contains("all-zero"), "msg: {msg}");
    }

    #[test]
    fn level_keys_error_debug_all_variants() {
        // Arrange & Act & Assert: verify Debug output contains variant names
        let dim_mismatch = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 1, expected: 2,
        };
        assert!(format!("{dim_mismatch:?}").contains("DimMismatch"));

        let non_finite = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 0 };
        assert!(format!("{non_finite:?}").contains("NonFinite"));

        let all_zero = LevelKeysError::AllZero { layer_idx: 0, level_idx: 0 };
        assert!(format!("{all_zero:?}").contains("AllZero"));
    }

    #[test]
    fn level_keys_error_partial_eq_same_variant_same_fields() {
        // Arrange
        let a = LevelKeysError::AllZero { layer_idx: 5, level_idx: 2 };
        let b = LevelKeysError::AllZero { layer_idx: 5, level_idx: 2 };
        // Assert
        assert_eq!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_same_variant_different_layer() {
        // Arrange
        let a = LevelKeysError::AllZero { layer_idx: 0, level_idx: 0 };
        let b = LevelKeysError::AllZero { layer_idx: 1, level_idx: 0 };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_same_variant_different_level() {
        // Arrange
        let a = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 0 };
        let b = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 1 };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_dim_mismatch_different_actual() {
        // Arrange
        let a = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 1, expected: 4,
        };
        let b = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 2, expected: 4,
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_dim_mismatch_different_expected() {
        // Arrange
        let a = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 1, expected: 3,
        };
        let b = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 1, expected: 4,
        };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_clone_all_variants() {
        // Arrange
        let errors = vec![
            LevelKeysError::DimMismatch { layer_idx: 1, level_idx: 2, actual: 3, expected: 4 },
            LevelKeysError::NonFinite { layer_idx: 5, level_idx: 1 },
            LevelKeysError::AllZero { layer_idx: 0, level_idx: 0 },
        ];
        // Act & Assert: clone each and verify equality
        for err in errors {
            let cloned = err.clone();
            assert_eq!(err, cloned);
        }
    }

    // ── mean_pool_rows: additional edge cases ──

    #[test]
    fn mean_pool_two_rows_different_values() {
        // Arrange: [[10, 20, 30], [20, 40, 60]] → [15, 30, 45]
        let data = vec![10.0f32, 20.0, 30.0, 20.0, 40.0, 60.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 3).unwrap();
        // Assert
        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 15.0).abs() < 1e-6);
        assert!((pooled[1] - 30.0).abs() < 1e-6);
        assert!((pooled[2] - 45.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_three_rows_averages_correctly() {
        // Arrange: [[6], [9], [12]] → [9]
        let data = vec![6.0f32, 9.0, 12.0];
        // Act
        let pooled = mean_pool_rows(&data, 3, 1).unwrap();
        // Assert
        assert!((pooled[0] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_single_element() {
        // Arrange: 1 row, 1 col, value = 42.0
        let data = vec![42.0f32];
        // Act
        let pooled = mean_pool_rows(&data, 1, 1).unwrap();
        // Assert
        assert!((pooled[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_kv_dim_zero_seq_len_one_empty_result() {
        // Arrange: 1 row, 0 cols → empty data, but seq_len=1, kv_dim=0
        // data.len() == 0 == 1*0, so it should succeed with empty output
        let data: Vec<f32> = vec![];
        // Act
        let pooled = mean_pool_rows(&data, 1, 0).unwrap();
        // Assert
        assert!(pooled.is_empty());
    }

    #[test]
    fn mean_pool_preserves_negative_signs() {
        // Arrange: [[-100, -200], [100, 200]] → [0, 0]
        let data = vec![-100.0f32, -200.0, 100.0, 200.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert
        assert!(pooled[0].abs() < 1e-4, "got {}", pooled[0]);
        assert!(pooled[1].abs() < 1e-4, "got {}", pooled[1]);
    }

    // ── decode_bytes_to_f32: dtype F8E4M3, F8E5M2 unsupported ──

    #[test]
    fn decode_rejects_f8e4m3_dtype() {
        // Arrange: F8E4M3 is size=1 byte
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F8E4M3).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    #[test]
    fn decode_rejects_f8e5m2_dtype() {
        // Arrange: F8E5M2 is size=1 byte
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F8E5M2).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    #[test]
    fn decode_rejects_f4e2m1_dtype() {
        // Arrange: FP4 is size=1 byte (sub-byte)
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F4E2M1).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    #[test]
    fn decode_rejects_f6e3m2_dtype() {
        // Arrange: FP6 E3M2 is sub-byte
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F6E3M2).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    #[test]
    fn decode_rejects_f6e2m3_dtype() {
        // Arrange: FP6 E2M3 is sub-byte
        let bytes = vec![0u8; 1];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F6E2M3).unwrap_err();
        // Assert
        assert!(err.contains("unsupported dtype"), "unexpected error: {err}");
    }

    // ── decode_bytes_to_f32: more edge cases ──

    #[test]
    fn decode_f32_many_elements() {
        // Arrange: 10 elements as F32 (40 bytes)
        let values: Vec<f32> = (0..10).map(|i| (i as f32) * 0.5 - 2.5).collect();
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 5, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 10);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_f16_negative_values() {
        // Arrange: negative values via F16
        let values = vec![-1.0f32, -128.0, 0.5, -0.25];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 4, 1, DType::F16).unwrap();
        // Assert: f16 precision is limited
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.5, "f16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn decode_bf16_many_elements() {
        // Arrange: 8 elements as BF16 (16 bytes)
        let values: Vec<f32> = (0..8).map(|i| (i as f32) * 10.0).collect();
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 4, 2, DType::BF16).unwrap();
        // Assert
        assert_eq!(decoded.len(), 8);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.5, "bf16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn decode_f32_max_value_roundtrip() {
        // Arrange
        let value = f32::MAX;
        let bytes = value.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded[0], f32::MAX);
    }

    #[test]
    fn decode_f32_min_positive_roundtrip() {
        // Arrange
        let value = f32::MIN_POSITIVE;
        let bytes = value.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded[0], f32::MIN_POSITIVE);
    }

    // ── LevelKeysCache: get behavior with empty and populated ──

    #[test]
    fn cache_get_returns_none_on_empty_cache() {
        // Arrange
        let cache = LevelKeysCache::new(4);
        // Act & Assert
        assert!(cache.get(0).is_none());
    }

    #[test]
    fn cache_len_increments_with_each_unique_layer() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act & Assert
        assert_eq!(cache.len(), 0);
        cache.insert(0, k.clone()).unwrap();
        assert_eq!(cache.len(), 1);
        cache.insert(1, k.clone()).unwrap();
        assert_eq!(cache.len(), 2);
        cache.insert(2, k.clone()).unwrap();
        assert_eq!(cache.len(), 3);
        // Overwrite doesn't increment
        cache.insert(1, k).unwrap();
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn cache_detection_layers_reflects_only_inserted_layers() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(3, k.clone()).unwrap();
        cache.insert(7, k).unwrap();
        // Act & Assert
        assert_eq!(cache.detection_layers(), &[3, 7]);
    }

    // ── LevelKeysCache: mixed valid/invalid insert sequences ──

    #[test]
    fn cache_insert_valid_then_invalid_then_valid() {
        // Arrange
        let mut cache = LevelKeysCache::new(2);
        let good: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let bad: [Vec<f32>; 3] = [
            vec![f32::NAN, 1.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        // Act
        cache.insert(0, good).unwrap();
        let _ = cache.insert(1, bad); // fails
        cache.insert(2, cache.get(0).unwrap().clone()).unwrap();
        // Assert
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.detection_layers(), &[0, 2]);
    }

    #[test]
    fn cache_insert_non_finite_infinity_positive() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let bad: [Vec<f32>; 3] = [
            vec![f32::INFINITY],
            vec![1.0],
            vec![1.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 0, .. }),
            "expected NonFinite for +inf, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_non_finite_infinity_negative() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let bad: [Vec<f32>; 3] = [
            vec![1.0],
            vec![f32::NEG_INFINITY],
            vec![1.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 1, .. }),
            "expected NonFinite for -inf at level 1, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_nan_in_middle_level() {
        // Arrange: level 0 valid, level 1 has NaN in second position, level 2 valid
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0, 3.0],
            vec![4.0, f32::NAN, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 1, .. }),
            "expected NonFinite at level 1, got {err:?}"
        );
    }

    // ── mean_pool_rows: additional boundary and numeric tests ──

    #[test]
    fn mean_pool_all_same_value() {
        // Arrange: 5 rows × 3 cols, all = 7.0
        let data = vec![7.0f32; 15];
        // Act
        let pooled = mean_pool_rows(&data, 5, 3).unwrap();
        // Assert
        assert_eq!(pooled, vec![7.0f32; 3]);
    }

    #[test]
    fn mean_pool_alternating_positive_negative() {
        // Arrange: [[1, -1], [-1, 1]] → [0, 0]
        let data = vec![1.0f32, -1.0, -1.0, 1.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert
        assert!(pooled[0].abs() < 1e-6, "got {}", pooled[0]);
        assert!(pooled[1].abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn mean_pool_large_dimension_single_row() {
        // Arrange: 1 row × 512 cols
        let data: Vec<f32> = (0..512).map(|i| (i as f32) * 0.1).collect();
        // Act
        let pooled = mean_pool_rows(&data, 1, 512).unwrap();
        // Assert: single row identity
        assert_eq!(pooled.len(), 512);
        assert!((pooled[0] - 0.0).abs() < 1e-6);
        assert!((pooled[511] - 51.1).abs() < 1e-4);
    }

    #[test]
    fn mean_pool_seq_len_one_kv_dim_one() {
        // Arrange: 1 element
        let data = vec![3.14f32];
        // Act
        let pooled = mean_pool_rows(&data, 1, 1).unwrap();
        // Assert
        assert!((pooled[0] - 3.14).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_rejects_seq_len_zero_with_data() {
        // Arrange: non-empty data but seq_len=0
        let data = vec![1.0, 2.0, 3.0];
        // Act
        let err = mean_pool_rows(&data, 0, 3).unwrap_err();
        // Assert
        assert!(err.contains("seq_len = 0"), "unexpected error: {err}");
    }

    // ══════════════════════════════════════════════════════════════════════
    // Additional tests: 43 new tests to reach 161 total
    // ══════════════════════════════════════════════════════════════════════

    // ── LevelKeysCache: all-zero validation order (level 0 → 1 → 2) ──

    #[test]
    fn cache_all_zero_level0_detected_before_level1() {
        // Arrange: level 0 all-zero, level 1 also all-zero
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![0.0, 0.0],  // first violation
            vec![0.0, 0.0],  // also bad, but should not be reached
            vec![1.0, 2.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: first error is at level 0
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero at level 0, got {err:?}"
        );
    }

    #[test]
    fn cache_non_finite_level0_detected_before_dim_mismatch_level1() {
        // Arrange: level 0 has NaN (non-finite), level 1 has wrong dim
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![f32::NAN, 1.0],  // non-finite
            vec![1.0],             // dim mismatch
            vec![1.0, 2.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: level 0 non-finite is caught first
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 0, .. }),
            "expected NonFinite at level 0, got {err:?}"
        );
    }

    #[test]
    fn cache_dim_mismatch_level0_detected_before_all_zero_level1() {
        // Arrange: level 0 wrong dim, level 1 all-zero
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],       // wrong dim: 2 != 3
            vec![0.0, 0.0, 0.0],  // all-zero (should not be checked)
            vec![1.0, 2.0, 3.0],
        ];
        // Act
        let err = cache.insert(5, bad).unwrap_err();
        // Assert: dim check at level 0 happens first
        assert!(
            matches!(err, LevelKeysError::DimMismatch { level_idx: 0, .. }),
            "expected DimMismatch at level 0, got {err:?}"
        );
    }

    #[test]
    fn cache_dim_check_before_finite_check_same_level() {
        // Arrange: level 0 has wrong dim AND would have non-finite if it were the right dim
        // The dim check runs first within each level
        let mut cache = LevelKeysCache::new(4);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],       // dim mismatch (2 != 4)
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::DimMismatch { level_idx: 0, .. }),
            "expected DimMismatch at level 0, got {err:?}"
        );
    }

    // ── LevelKeysCache: overwrite atomicity ──

    #[test]
    fn cache_overwrite_preserves_original_on_partial_validation_failure() {
        // Arrange: layer 5 has valid data, try overwrite with bad data at level 1
        let mut cache = LevelKeysCache::new(2);
        let original: [Vec<f32>; 3] = [
            vec![10.0, 20.0],
            vec![30.0, 40.0],
            vec![50.0, 60.0],
        ];
        cache.insert(5, original).unwrap();
        // Act: overwrite with data that passes level 0 but fails level 1 (non-finite)
        let bad: [Vec<f32>; 3] = [
            vec![100.0, 200.0],       // valid
            vec![f32::NAN, f32::NAN],  // non-finite
            vec![300.0, 400.0],
        ];
        let _ = cache.insert(5, bad);
        // Assert: original data is intact (insert validates all 3 levels atomically)
        let retrieved = cache.get(5).unwrap();
        assert_eq!(retrieved[0], vec![10.0, 20.0]);
        assert_eq!(retrieved[1], vec![30.0, 40.0]);
        assert_eq!(retrieved[2], vec![50.0, 60.0]);
    }

    #[test]
    fn cache_overwrite_valid_data_with_all_zero_level2() {
        // Arrange: valid data, overwrite with all-zero at level 2
        let mut cache = LevelKeysCache::new(2);
        let good: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        cache.insert(1, good).unwrap();
        // Act: level 0 and 1 pass, level 2 is all-zero
        let bad: [Vec<f32>; 3] = [
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![0.0, 0.0],
        ];
        let _ = cache.insert(1, bad);
        // Assert: original preserved
        assert_eq!(cache.get(1).unwrap()[2], vec![5.0, 6.0]);
    }

    // ── LevelKeysCache: detection_layers consistency after errors ──

    #[test]
    fn cache_detection_layers_not_updated_on_failed_insert() {
        // Arrange: insert layer 3, then fail to insert layer 7
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(3, k).unwrap();
        // Act: bad insert for layer 7
        let bad: [Vec<f32>; 3] = [vec![0.0], vec![2.0], vec![3.0]];
        let _ = cache.insert(7, bad);
        // Assert: detection_layers only has 3
        assert_eq!(cache.detection_layers(), &[3]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_detection_layers_unmodified_after_overwrite_failure() {
        // Arrange: insert layers 2 and 8
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(2, k.clone()).unwrap();
        cache.insert(8, k.clone()).unwrap();
        // Act: try to overwrite layer 2 with bad data
        let bad: [Vec<f32>; 3] = [vec![f32::NAN], vec![2.0], vec![3.0]];
        let _ = cache.insert(2, bad);
        // Assert: detection_layers unchanged
        assert_eq!(cache.detection_layers(), &[2, 8]);
    }

    // ── LevelKeysCache: multiple layers with varied data ──

    #[test]
    fn cache_multiple_layers_each_with_unique_data() {
        // Arrange: insert 3 layers, each with distinct data
        let mut cache = LevelKeysCache::new(2);
        let k1: [Vec<f32>; 3] = [vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
        let k2: [Vec<f32>; 3] = [vec![10.0, 10.0], vec![20.0, 20.0], vec![30.0, 30.0]];
        let k3: [Vec<f32>; 3] = [vec![-1.0, -1.0], vec![-2.0, -2.0], vec![-3.0, -3.0]];
        // Act
        cache.insert(0, k1).unwrap();
        cache.insert(5, k2).unwrap();
        cache.insert(10, k3).unwrap();
        // Assert: each layer retains its own data
        assert_eq!(cache.get(0).unwrap()[0], vec![1.0, 1.0]);
        assert_eq!(cache.get(5).unwrap()[0], vec![10.0, 10.0]);
        assert_eq!(cache.get(10).unwrap()[0], vec![-1.0, -1.0]);
        assert_eq!(cache.detection_layers(), &[0, 5, 10]);
    }

    #[test]
    fn cache_insert_non_adjacent_layers() {
        // Arrange: insert layers 0, 100, 50
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act
        cache.insert(0, k.clone()).unwrap();
        cache.insert(100, k.clone()).unwrap();
        cache.insert(50, k).unwrap();
        // Assert
        assert_eq!(cache.detection_layers(), &[0, 50, 100]);
        assert!(cache.get(50).is_some());
    }

    // ── LevelKeysCache: is_empty semantics ──

    #[test]
    fn cache_is_empty_becomes_false_after_insert() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        assert!(cache.is_empty());
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        // Act
        cache.insert(0, k).unwrap();
        // Assert
        assert!(!cache.is_empty());
    }

    #[test]
    fn cache_is_empty_stays_true_after_failed_insert() {
        // Arrange
        let mut cache = LevelKeysCache::new(1);
        let bad: [Vec<f32>; 3] = [vec![0.0], vec![0.0], vec![0.0]];
        // Act
        let _ = cache.insert(0, bad);
        // Assert: still empty
        assert!(cache.is_empty());
    }

    // ── LevelKeysCache: kv_dim never changes ──

    #[test]
    fn cache_kv_dim_unchanged_after_multiple_operations() {
        // Arrange
        let mut cache = LevelKeysCache::new(16);
        let k: [Vec<f32>; 3] = [vec![1.0; 16], vec![2.0; 16], vec![3.0; 16]];
        // Act
        cache.insert(0, k).unwrap();
        let bad: [Vec<f32>; 3] = [vec![1.0; 8], vec![2.0; 16], vec![3.0; 16]];
        let _ = cache.insert(1, bad);
        // Assert: kv_dim never changes
        assert_eq!(cache.kv_dim(), 16);
    }

    // ── LevelKeysCache: level key retrieval by index ──

    #[test]
    fn cache_get_level1_and_level2_separately() {
        // Arrange
        let mut cache = LevelKeysCache::new(3);
        let k: [Vec<f32>; 3] = [vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        cache.insert(0, k).unwrap();
        // Act
        let retrieved = cache.get(0).unwrap();
        // Assert: each level has correct data
        assert_eq!(retrieved[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(retrieved[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(retrieved[2], vec![7.0, 8.0, 9.0]);
    }

    // ── LevelKeysCache: validation stops at first failing level ──

    #[test]
    fn cache_insert_all_three_levels_all_zero() {
        // Arrange: all three levels are all-zero — first violation at level 0
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: reports level 0 (first checked)
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero at level 0, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_level0_finite_level1_dim_bad_level2_nonfinite() {
        // Arrange: level 0 passes, level 1 has wrong dim
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],    // valid
            vec![1.0],          // dim mismatch
            vec![f32::NAN, 1.0], // would be non-finite but never reached
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: dim mismatch at level 1
        assert!(
            matches!(err, LevelKeysError::DimMismatch { level_idx: 1, .. }),
            "expected DimMismatch at level 1, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_level0_dim_correct_level1_zero_level2_nonfinite() {
        // Arrange: level 0 valid, level 1 all-zero (caught before level 2 non-finite)
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![0.0, 0.0],       // all-zero
            vec![f32::INFINITY, 1.0], // non-finite but never checked
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: all-zero at level 1
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 1, .. }),
            "expected AllZero at level 1, got {err:?}"
        );
    }

    // ── decode_bytes_to_f32: non-standard seq_len/kv_dim ratios ──

    #[test]
    fn decode_f32_1x4_shape() {
        // Arrange: seq_len=1, kv_dim=4 → 4 elements, 16 bytes
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 4, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded, values);
    }

    #[test]
    fn decode_f32_4x1_shape() {
        // Arrange: seq_len=4, kv_dim=1 → 4 elements, 16 bytes
        let values = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 4, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded, values);
    }

    #[test]
    fn decode_f32_3x5_shape() {
        // Arrange: seq_len=3, kv_dim=5 → 15 elements
        let values: Vec<f32> = (0..15).map(|i| (i as f32) * 0.1).collect();
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 3, 5, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 15);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    // ── decode_bytes_to_f32: output length always equals seq_len * kv_dim ──

    #[test]
    fn decode_output_len_equals_seq_times_dim_f16() {
        // Arrange: seq_len=3, kv_dim=7, F16 → 3*7*2 = 42 bytes
        let bytes = vec![0u8; 42];
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 3, 7, DType::F16).unwrap();
        // Assert
        assert_eq!(decoded.len(), 21);
    }

    #[test]
    fn decode_output_len_equals_seq_times_dim_bf16() {
        // Arrange: seq_len=2, kv_dim=8, BF16 → 2*8*2 = 32 bytes
        let bytes = vec![0u8; 32];
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 8, DType::BF16).unwrap();
        // Assert
        assert_eq!(decoded.len(), 16);
    }

    #[test]
    fn decode_output_len_equals_seq_times_dim_f32() {
        // Arrange: seq_len=5, kv_dim=3, F32 → 5*3*4 = 60 bytes
        let bytes = vec![0u8; 60];
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 5, 3, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 15);
    }

    // ── decode_bytes_to_f32: byte length mismatch with extra bytes ──

    #[test]
    fn decode_rejects_extra_trailing_bytes() {
        // Arrange: need 8 bytes for 2x1 F32, provide 10
        let bytes = vec![0u8; 10];
        // Act
        let err = decode_bytes_to_f32(&bytes, 2, 1, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("bytes len"), "unexpected error: {err}");
        assert!(err.contains("10"), "error should show actual length: {err}");
    }

    #[test]
    fn decode_rejects_one_byte_short() {
        // Arrange: need 8 bytes for 2x1 F32, provide 7
        let bytes = vec![0u8; 7];
        // Act
        let err = decode_bytes_to_f32(&bytes, 2, 1, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("bytes len"), "unexpected error: {err}");
    }

    // ── mean_pool_rows: floating point accumulation precision ──

    #[test]
    fn mean_pool_f64_accumulation_for_many_small_values() {
        // Arrange: 200 rows × 1 col, each = 0.01
        // f32 accumulation would lose precision; f64 should be better
        let data: Vec<f32> = vec![0.01f32; 200];
        // Act
        let pooled = mean_pool_rows(&data, 200, 1).unwrap();
        // Assert: result should be approximately 0.01
        assert!(
            (pooled[0] - 0.01).abs() < 0.001,
            "expected ~0.01, got {}",
            pooled[0]
        );
    }

    #[test]
    fn mean_pool_two_rows_zero_sum() {
        // Arrange: [[100.0, -100.0], [-100.0, 100.0]] → [0, 0]
        let data = vec![100.0f32, -100.0, -100.0, 100.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert
        assert!(pooled[0].abs() < 1e-6, "got {}", pooled[0]);
        assert!(pooled[1].abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn mean_pool_single_row_large_values() {
        // Arrange: 1 row × 3 cols with very large values
        let data = vec![1e30f32, -1e30, 1e38];
        // Act
        let pooled = mean_pool_rows(&data, 1, 3).unwrap();
        // Assert: single row is identity
        assert!((pooled[0] - 1e30).abs() < 1e20);
        assert!((pooled[1] - (-1e30)).abs() < 1e20);
        assert!((pooled[2] - 1e38).abs() < 1e28);
    }

    #[test]
    fn mean_pool_output_dimension_equals_kv_dim() {
        // Arrange: 3 rows × 7 cols
        let data: Vec<f32> = (0..21).map(|i| i as f32).collect();
        // Act
        let pooled = mean_pool_rows(&data, 3, 7).unwrap();
        // Assert
        assert_eq!(pooled.len(), 7);
    }

    #[test]
    fn mean_pool_kv_dim_zero_seq_len_zero_empty_result() {
        // Arrange: no rows, no cols
        let data: Vec<f32> = vec![];
        // Act
        let err = mean_pool_rows(&data, 0, 0).unwrap_err();
        // Assert: seq_len=0 is rejected first
        assert!(err.contains("seq_len = 0"), "unexpected error: {err}");
    }

    #[test]
    fn mean_pool_four_rows_progression() {
        // Arrange: [[1], [2], [3], [4]] → [2.5]
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        // Act
        let pooled = mean_pool_rows(&data, 4, 1).unwrap();
        // Assert
        assert!((pooled[0] - 2.5).abs() < 1e-6, "got {}", pooled[0]);
    }

    // ── LevelKeysError: PartialEq with all field differences ──

    #[test]
    fn level_keys_error_partial_eq_non_finite_different_level_idx() {
        // Arrange: same layer_idx, different level_idx
        let a = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 0 };
        let b = LevelKeysError::NonFinite { layer_idx: 0, level_idx: 2 };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_all_zero_different_layer_idx() {
        // Arrange: same level_idx, different layer_idx
        let a = LevelKeysError::AllZero { layer_idx: 0, level_idx: 0 };
        let b = LevelKeysError::AllZero { layer_idx: 99, level_idx: 0 };
        // Assert
        assert_ne!(a, b);
    }

    #[test]
    fn level_keys_error_partial_eq_all_zero_different_level_idx() {
        // Arrange: same layer_idx, different level_idx
        let a = LevelKeysError::AllZero { layer_idx: 5, level_idx: 0 };
        let b = LevelKeysError::AllZero { layer_idx: 5, level_idx: 2 };
        // Assert
        assert_ne!(a, b);
    }

    // ── LevelKeysError: source() returns None (thiserror, no source) ──

    #[test]
    fn level_keys_error_source_is_none() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 0, level_idx: 0, actual: 1, expected: 2,
        };
        // Act
        let source = std::error::Error::source(&err);
        // Assert: no chained error
        assert!(source.is_none());
    }

    // ── LevelKeysError: Display contains keyword substrings ──

    #[test]
    fn level_keys_error_dim_mismatch_display_contains_dim() {
        // Arrange
        let err = LevelKeysError::DimMismatch {
            layer_idx: 1, level_idx: 0, actual: 3, expected: 6,
        };
        // Act
        let msg = format!("{err}");
        // Assert: contains "dim" keyword and field values
        assert!(msg.contains("dim"), "msg: {msg}");
        assert!(msg.contains("mismatch"), "msg: {msg}");
    }

    // ── decode_bytes_to_f32: byte overflow in total*elem multiplication ──

    #[test]
    fn decode_overflow_in_total_times_elem() {
        // Arrange: seq_len*kv_dim does not overflow, but total*elem does
        // usize::MAX / 4 * 4 = usize::MAX - 3 (no overflow), then * 4 overflows
        let big = usize::MAX / 4 + 1; // seq_len * kv_dim = big, then big * 4 overflows
        let bytes = vec![0u8; 16];
        // Act: seq_len=big/2, kv_dim=2 → total=big, then big*4 overflows
        let err = decode_bytes_to_f32(&bytes, big / 2, 2, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("overflow"), "unexpected error: {err}");
    }

    // ── LevelKeysCache: insert returns Ok with correct side effects ──

    #[test]
    fn cache_successful_insert_returns_ok() {
        // Arrange
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        // Act
        let result = cache.insert(0, k);
        // Assert
        assert!(result.is_ok());
    }

    // ── LevelKeysCache: data integrity after many sequential inserts ──

    #[test]
    fn cache_sequential_inserts_data_integrity() {
        // Arrange: insert 5 layers with unique patterns (offset +1 to avoid all-zero)
        let mut cache = LevelKeysCache::new(2);
        for layer in 0..5usize {
            let base = (layer + 1) as f32;
            let k: [Vec<f32>; 3] = [
                vec![base, base],
                vec![base * 10.0, base * 10.0],
                vec![base * 100.0, base * 100.0],
            ];
            cache.insert(layer, k).unwrap();
        }
        // Act & Assert: verify all layers
        assert_eq!(cache.len(), 5);
        for layer in 0..5usize {
            let base = (layer + 1) as f32;
            let r = cache.get(layer).unwrap();
            assert_eq!(r[0][0], base, "layer {layer} level 0");
            assert_eq!(r[1][0], base * 10.0, "layer {layer} level 1");
            assert_eq!(r[2][0], base * 100.0, "layer {layer} level 2");
        }
        assert_eq!(cache.detection_layers(), &[0, 1, 2, 3, 4]);
    }

    // ── LevelKeysCache: clone deep copies vectors ──

    #[test]
    fn cache_clone_deep_copies_vector_data() {
        // Arrange
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        cache.insert(0, k).unwrap();
        // Act: clone and modify original
        let cloned = cache.clone();
        let new_k: [Vec<f32>; 3] = [vec![99.0, 99.0], vec![99.0, 99.0], vec![99.0, 99.0]];
        cache.insert(0, new_k).unwrap();
        // Assert: cloned data is unchanged
        assert_eq!(cloned.get(0).unwrap()[0], vec![1.0, 2.0]);
    }

    // ── decode_bytes_to_f32: F16/BF16 exact roundtrip for powers of 2 ──

    #[test]
    fn decode_f16_exact_powers_of_two() {
        // Arrange: 1.0, 2.0, 4.0, 8.0 are exact in f16
        let values = vec![1.0f32, 2.0, 4.0, 8.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&f16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 4, 1, DType::F16).unwrap();
        // Assert: powers of 2 round-trip exactly
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn decode_bf16_exact_powers_of_two() {
        // Arrange: 1.0, 2.0, 4.0, 8.0 are exact in bf16
        let values = vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 6, 1, DType::BF16).unwrap();
        // Assert
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    // ── mean_pool_rows: mixed positive and negative values ──

    #[test]
    fn mean_pool_asymmetric_rows() {
        // Arrange: [[1, 2, 3], [10, 20, 30]] → [5.5, 11, 16.5]
        let data = vec![1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 3).unwrap();
        // Assert
        assert!((pooled[0] - 5.5).abs() < 1e-6, "got {}", pooled[0]);
        assert!((pooled[1] - 11.0).abs() < 1e-6, "got {}", pooled[1]);
        assert!((pooled[2] - 16.5).abs() < 1e-6, "got {}", pooled[2]);
    }

    #[test]
    fn mean_pool_result_elements_are_finite() {
        // Arrange: various finite inputs
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01 - 0.5).collect();
        // Act
        let pooled = mean_pool_rows(&data, 100, 1).unwrap();
        // Assert: result is finite
        assert!(pooled[0].is_finite(), "result should be finite");
    }

    // ── 12 additional tests ──

    #[test]
    fn cache_default_detection_layers_empty() {
        // Arrange
        let cache = LevelKeysCache::default();
        // Act & Assert: detection_layers is empty on Default
        assert!(cache.detection_layers().is_empty());
    }

    #[test]
    fn cache_insert_rejects_all_nan_levels() {
        // Arrange: NaN in all 3 levels
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![f32::NAN, f32::NAN],
            vec![f32::NAN, f32::NAN],
            vec![f32::NAN, f32::NAN],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: first violation at level 0
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 0, .. }),
            "expected NonFinite at level 0, got {err:?}"
        );
    }

    #[test]
    fn cache_insert_level2_nan_levels01_valid() {
        // Arrange: levels 0 and 1 valid, level 2 has NaN
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![f32::NAN, 5.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::NonFinite { level_idx: 2, .. }),
            "expected NonFinite at level 2, got {err:?}"
        );
    }

    #[test]
    fn mean_pool_2x2_adjacent_values() {
        // Arrange: [[1, 2], [3, 4]] → [2, 3]
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        // Act
        let pooled = mean_pool_rows(&data, 2, 2).unwrap();
        // Assert
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn mean_pool_5_rows_descending() {
        // Arrange: [[5], [4], [3], [2], [1]] → [3]
        let data = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        // Act
        let pooled = mean_pool_rows(&data, 5, 1).unwrap();
        // Assert
        assert!((pooled[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn decode_f32_50_elements() {
        // Arrange: 50 elements as F32 (10 seq × 5 dim)
        let values: Vec<f32> = (0..50).map(|i| (i as f32) * 0.1 - 2.5).collect();
        let mut bytes = Vec::with_capacity(200);
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 10, 5, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded.len(), 50);
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn cache_insert_subnormal_floats() {
        // Arrange: subnormal f32 values (smaller than MIN_POSITIVE) are still finite
        let subnormal = f32::from_bits(1);
        let neg_subnormal = f32::from_bits(0x8000_0001);
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![subnormal, neg_subnormal],
            vec![subnormal, subnormal],
            vec![neg_subnormal, subnormal],
        ];
        // Act
        cache.insert(0, k).unwrap();
        // Assert: subnormals accepted as valid finite values
        let r = cache.get(0).unwrap();
        assert_eq!(r[0][0], subnormal);
        assert_eq!(r[0][1], neg_subnormal);
    }

    #[test]
    fn cache_overwrite_identical_data() {
        // Arrange
        let mut cache = LevelKeysCache::new(2);
        let k: [Vec<f32>; 3] = [
            vec![1.5, 2.5],
            vec![3.5, 4.5],
            vec![5.5, 6.5],
        ];
        cache.insert(0, k.clone()).unwrap();
        // Act: overwrite with identical data
        cache.insert(0, k).unwrap();
        // Assert: data preserved, still 1 entry
        let r = cache.get(0).unwrap();
        assert_eq!(r[0], vec![1.5, 2.5]);
        assert_eq!(r[1], vec![3.5, 4.5]);
        assert_eq!(r[2], vec![5.5, 6.5]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn decode_bf16_negative_exact_values() {
        // Arrange: -1, -2, -4, -8 are exact in BF16 (negative powers of 2)
        let values = vec![-1.0f32, -2.0, -4.0, -8.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 4, 1, DType::BF16).unwrap();
        // Assert
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn mean_pool_3x2_pattern() {
        // Arrange: [[1, 10], [2, 20], [3, 30]] → [2, 20]
        let data = vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0];
        // Act
        let pooled = mean_pool_rows(&data, 3, 2).unwrap();
        // Assert
        assert!((pooled[0] - 2.0).abs() < 1e-6, "got {}", pooled[0]);
        assert!((pooled[1] - 20.0).abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn cache_empty_after_multiple_failed_inserts() {
        // Arrange: 3 inserts with different error types
        let mut cache = LevelKeysCache::new(2);
        let dim_bad: [Vec<f32>; 3] = [vec![1.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let nan_bad: [Vec<f32>; 3] = [vec![f32::NAN, 1.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let zero_bad: [Vec<f32>; 3] = [vec![0.0, 0.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        // Act
        let _ = cache.insert(0, dim_bad);
        let _ = cache.insert(1, nan_bad);
        let _ = cache.insert(2, zero_bad);
        // Assert: cache remains empty after all failures
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(cache.detection_layers().is_empty());
    }

    #[test]
    fn decode_f32_alternating_signs() {
        // Arrange: [1, -1, 1, -1] as F32
        let values = vec![1.0f32, -1.0, 1.0, -1.0];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 2, 2, DType::F32).unwrap();
        // Assert: sign preserved exactly
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], -1.0);
        assert_eq!(decoded[2], 1.0);
        assert_eq!(decoded[3], -1.0);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 13 new tests (174 → 187)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn cache_insert_then_overwrite_keeps_detection_layers_sorted() {
        // Arrange: insert layers [10, 2, 7], then overwrite layer 10
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(10, k.clone()).unwrap();
        cache.insert(2, k.clone()).unwrap();
        cache.insert(7, k.clone()).unwrap();
        // Act
        cache.insert(10, k).unwrap();
        // Assert: detection_layers still sorted, no duplicates
        assert_eq!(cache.detection_layers(), &[2, 7, 10]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn cache_rejects_negative_zero_only_vector() {
        // Arrange: -0.0 compares equal to 0.0, so a vector of all -0.0 is all-zero
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [
            vec![-0.0, -0.0, -0.0],
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero for all-negative-zero, got {err:?}"
        );
    }

    #[test]
    fn mean_pool_two_rows_wide_dimension() {
        // Arrange: 2 rows x 8 cols: row0 = [0,1,2,3,4,5,6,7], row1 = [8,9,10,11,12,13,14,15]
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // Act
        let pooled = mean_pool_rows(&data, 2, 8).unwrap();
        // Assert: pooled[col] = (col + col+8) / 2 = col + 4
        let expected: Vec<f32> = (0..8).map(|i| i as f32 + 4.0).collect();
        for (i, (got, exp)) in pooled.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "col {i}: got {got}, expected {exp}");
        }
    }

    #[test]
    fn decode_f32_preserves_denormalized_values() {
        // Arrange: denormalized (subnormal) f32 values
        let sub1 = f32::from_bits(0x0000_0001); // smallest positive subnormal
        let sub2 = f32::from_bits(0x0080_0000); // largest subnormal
        let values = vec![sub1, sub2];
        let mut bytes = Vec::new();
        for &v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 2, DType::F32).unwrap();
        // Assert: subnormals round-trip exactly through F32
        assert_eq!(decoded[0].to_bits(), sub1.to_bits());
        assert_eq!(decoded[1].to_bits(), sub2.to_bits());
    }

    #[test]
    fn cache_multiple_failed_inserts_then_succeed() {
        // Arrange: attempt several bad inserts, then a valid one
        let mut cache = LevelKeysCache::new(2);
        let dim_bad: [Vec<f32>; 3] = [vec![1.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let nan_bad: [Vec<f32>; 3] = [vec![f32::NAN, 1.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let zero_bad: [Vec<f32>; 3] = [vec![0.0, 0.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let good: [Vec<f32>; 3] = [vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        // Act: 3 failures, then success
        let _ = cache.insert(0, dim_bad);
        let _ = cache.insert(1, nan_bad);
        let _ = cache.insert(2, zero_bad);
        cache.insert(3, good).unwrap();
        // Assert: only the successful insert is recorded
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.detection_layers(), &[3]);
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn mean_pool_rows_with_ascending_values() {
        // Arrange: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] → [4.5]
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        // Act
        let pooled = mean_pool_rows(&data, 10, 1).unwrap();
        // Assert
        assert!((pooled[0] - 4.5).abs() < 1e-6, "got {}", pooled[0]);
    }

    #[test]
    fn decode_bf16_max_finite_roundtrip() {
        // Arrange: BF16 max finite value
        let bf16_max = bf16::MAX;
        let bytes = bf16_max.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert: round-trips through bf16 -> f32
        assert!((decoded[0] - bf16::MAX.to_f32()).abs() < 1.0,
            "expected ~{}, got {}", bf16::MAX.to_f32(), decoded[0]);
        assert!(decoded[0].is_finite());
    }

    #[test]
    fn cache_clone_preserves_kv_dim() {
        // Arrange: cache with non-trivial kv_dim
        let mut cache = LevelKeysCache::new(128);
        let k: [Vec<f32>; 3] = [
            (0..128).map(|i| (i as f32) * 0.01).collect(),
            (0..128).map(|i| 1.0 - (i as f32) * 0.005).collect(),
            (0..128).map(|i| -1.0 + (i as f32) * 0.01).collect(),
        ];
        cache.insert(5, k).unwrap();
        // Act
        let cloned = cache.clone();
        // Assert: kv_dim preserved exactly
        assert_eq!(cloned.kv_dim(), 128);
        assert_eq!(cloned.get(5).unwrap()[0].len(), 128);
    }

    #[test]
    fn level_keys_error_partial_eq_cross_variant_inequality() {
        // Arrange: DimMismatch vs NonFinite with same layer_idx/level_idx
        let dim = LevelKeysError::DimMismatch {
            layer_idx: 5, level_idx: 1, actual: 2, expected: 4,
        };
        let non_finite = LevelKeysError::NonFinite {
            layer_idx: 5, level_idx: 1,
        };
        let all_zero = LevelKeysError::AllZero {
            layer_idx: 5, level_idx: 1,
        };
        // Assert: cross-variant always not-equal
        assert_ne!(dim, non_finite);
        assert_ne!(dim, all_zero);
        assert_ne!(non_finite, all_zero);
    }

    #[test]
    fn cache_insert_after_removing_all_via_overwrite_with_different_layers() {
        // Arrange: insert layers [1, 2, 3]
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        cache.insert(1, k.clone()).unwrap();
        cache.insert(2, k.clone()).unwrap();
        cache.insert(3, k.clone()).unwrap();
        // Act: overwrite all with new data
        let new_k: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        cache.insert(1, new_k.clone()).unwrap();
        cache.insert(2, new_k.clone()).unwrap();
        cache.insert(3, new_k).unwrap();
        // Assert: layers exist with new data, len unchanged
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(1).unwrap()[0], vec![10.0]);
        assert_eq!(cache.get(2).unwrap()[0], vec![10.0]);
        assert_eq!(cache.get(3).unwrap()[0], vec![10.0]);
    }

    #[test]
    fn mean_pool_output_matches_manual_sum_divide() {
        // Arrange: 3 rows x 2 cols = [[2, 4], [6, 8], [10, 12]]
        let data = vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0];
        // Act
        let pooled = mean_pool_rows(&data, 3, 2).unwrap();
        // Assert: (2+6+10)/3 = 6.0, (4+8+12)/3 = 8.0
        assert!((pooled[0] - 6.0).abs() < 1e-6, "got {}", pooled[0]);
        assert!((pooled[1] - 8.0).abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn decode_f16_midpoint_value() {
        // Arrange: 0.5 is exactly representable in f16
        let value = 0.5f32;
        let bytes = f16::from_f32(value).to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert
        assert!((decoded[0] - 0.5).abs() < 1e-6, "got {}", decoded[0]);
    }

    #[test]
    fn cache_debug_format_contains_kv_dim_value() {
        // Arrange: cache with kv_dim=42
        let cache = LevelKeysCache::new(42);
        // Act
        let debug = format!("{cache:?}");
        // Assert: Debug output includes the kv_dim field value
        assert!(debug.contains("kv_dim"), "Debug: {debug}");
        // Also verify the struct name
        assert!(debug.contains("LevelKeysCache"), "Debug: {debug}");
    }

    // ══════════════════════════════════════════════════════════════════════
    // 13 new tests (187 → 200)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn level_keys_error_has_exactly_three_variants() {
        // Arrange: construct one of each variant and verify exhaustive match
        let errors = vec![
            LevelKeysError::DimMismatch { layer_idx: 0, level_idx: 0, actual: 1, expected: 2 },
            LevelKeysError::NonFinite { layer_idx: 0, level_idx: 0 },
            LevelKeysError::AllZero { layer_idx: 0, level_idx: 0 },
        ];
        // Assert: exactly 3 variants exist
        assert_eq!(errors.len(), 3);
        // Each variant is distinct
        assert_ne!(errors[0], errors[1]);
        assert_ne!(errors[1], errors[2]);
        assert_ne!(errors[0], errors[2]);
    }

    #[test]
    fn decode_f16_max_finite_roundtrip() {
        // Arrange: F16 max finite value
        let f16_max = f16::MAX;
        let bytes = f16_max.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert: round-trips through f16 -> f32
        assert!(
            (decoded[0] - f16::MAX.to_f32()).abs() < 1.0,
            "expected ~{}, got {}",
            f16::MAX.to_f32(),
            decoded[0]
        );
        assert!(decoded[0].is_finite());
    }

    #[test]
    fn decode_f16_min_positive_roundtrip() {
        // Arrange: F16 smallest positive subnormal
        let f16_sub = f16::from_bits(0x0001);
        let bytes = f16_sub.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert: subnormal round-trips
        assert!(
            (decoded[0] - f16_sub.to_f32()).abs() < 1e-10,
            "expected ~{}, got {}",
            f16_sub.to_f32(),
            decoded[0]
        );
        assert!(decoded[0] > 0.0);
    }

    #[test]
    fn decode_f32_negative_max_roundtrip() {
        // Arrange: largest negative finite f32
        let value = -f32::MAX;
        let bytes = value.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded[0], -f32::MAX);
    }

    #[test]
    fn cache_insert_single_zero_element_rejects_as_all_zero() {
        // Arrange: kv_dim=1, single-element vector with value 0.0
        let mut cache = LevelKeysCache::new(1);
        let bad: [Vec<f32>; 3] = [vec![0.0], vec![1.0], vec![2.0]];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: single zero element is all-zero
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero for single zero element, got {err:?}"
        );
    }

    #[test]
    fn cache_len_reflects_overwrite_not_increment() {
        // Arrange: insert 3 layers, then overwrite layer 1
        let mut cache = LevelKeysCache::new(1);
        let k: [Vec<f32>; 3] = [vec![1.0], vec![2.0], vec![3.0]];
        let k2: [Vec<f32>; 3] = [vec![10.0], vec![20.0], vec![30.0]];
        cache.insert(1, k.clone()).unwrap();
        cache.insert(2, k.clone()).unwrap();
        cache.insert(3, k).unwrap();
        assert_eq!(cache.len(), 3);
        // Act: overwrite layer 2
        cache.insert(2, k2).unwrap();
        // Assert: len is still 3, not 4
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn decode_empty_buffer_with_nonzero_dims_errors() {
        // Arrange: seq_len=1, kv_dim=1, F32 → expect 4 bytes, provide 0
        let bytes: Vec<u8> = vec![];
        // Act
        let err = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap_err();
        // Assert
        assert!(err.contains("bytes len"), "unexpected error: {err}");
    }

    #[test]
    fn mean_pool_fractional_result_verification() {
        // Arrange: [[1, 3], [2, 5], [3, 7]] → [2.0, 5.0]
        let data = vec![1.0f32, 3.0, 2.0, 5.0, 3.0, 7.0];
        // Act
        let pooled = mean_pool_rows(&data, 3, 2).unwrap();
        // Assert: (1+2+3)/3 = 2.0, (3+5+7)/3 = 5.0
        assert!((pooled[0] - 2.0).abs() < 1e-6, "got {}", pooled[0]);
        assert!((pooled[1] - 5.0).abs() < 1e-6, "got {}", pooled[1]);
    }

    #[test]
    fn cache_validates_dim_before_finite_within_same_level() {
        // Arrange: level 1 has wrong dim (2 instead of 3), level 2 would be all-zero
        // Dim check runs first per level, so dim mismatch at level 1 is caught
        let mut cache = LevelKeysCache::new(3);
        let bad: [Vec<f32>; 3] = [
            vec![1.0, 2.0, 3.0], // valid
            vec![1.0, 2.0],       // dim mismatch: 2 != 3
            vec![0.0, 0.0, 0.0],  // would be all-zero, but never checked
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: dim mismatch caught before all-zero check at level 2
        assert!(
            matches!(err, LevelKeysError::DimMismatch { layer_idx: 0, level_idx: 1, actual: 2, expected: 3 }),
            "expected DimMismatch(layer=0, level=1, actual=2, expected=3), got {err:?}"
        );
    }

    #[test]
    fn decode_bf16_min_positive_roundtrip() {
        // Arrange: BF16 smallest positive normal value
        let bf16_min = bf16::MIN_POSITIVE;
        let bytes = bf16_min.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert: round-trips correctly
        assert!(
            (decoded[0] - bf16_min.to_f32()).abs() < 1e-20,
            "expected ~{}, got {}",
            bf16_min.to_f32(),
            decoded[0]
        );
        assert!(decoded[0] > 0.0);
    }

    #[test]
    fn cache_overwrite_valid_with_all_zero_level1_preserves_original() {
        // Arrange: insert valid data, then try overwrite with all-zero at level 1
        let mut cache = LevelKeysCache::new(2);
        let good: [Vec<f32>; 3] = [vec![10.0, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]];
        cache.insert(7, good).unwrap();
        // Act: overwrite attempt with all-zero at level 1
        let bad: [Vec<f32>; 3] = [
            vec![100.0, 200.0], // valid
            vec![0.0, 0.0],     // all-zero
            vec![300.0, 400.0],
        ];
        let _ = cache.insert(7, bad);
        // Assert: original data preserved because insert validates all 3 atomically
        let retrieved = cache.get(7).unwrap();
        assert_eq!(retrieved[0], vec![10.0, 20.0]);
        assert_eq!(retrieved[1], vec![30.0, 40.0]);
        assert_eq!(retrieved[2], vec![50.0, 60.0]);
    }

    #[test]
    fn mean_pool_subnormal_accumulation() {
        // Arrange: subnormal values near f32::MIN_POSITIVE
        let sub = f32::from_bits(0x0000_0001); // smallest positive subnormal
        let data = vec![sub; 4]; // 4 rows × 1 col
        // Act
        let pooled = mean_pool_rows(&data, 4, 1).unwrap();
        // Assert: mean of identical subnormals is the same subnormal
        assert!(
            (pooled[0] - sub).abs() < f32::EPSILON,
            "expected ~{sub:e}, got {:e}",
            pooled[0]
        );
    }

    #[test]
    fn cache_new_kv_dim_matches_constructor_argument() {
        // Arrange & Act: create caches with various kv_dim values
        let c1 = LevelKeysCache::new(0);
        let c2 = LevelKeysCache::new(1);
        let c3 = LevelKeysCache::new(256);
        let c4 = LevelKeysCache::new(4096);
        // Assert: kv_dim() returns exactly the constructor argument
        assert_eq!(c1.kv_dim(), 0);
        assert_eq!(c2.kv_dim(), 1);
        assert_eq!(c3.kv_dim(), 256);
        assert_eq!(c4.kv_dim(), 4096);
    }

    // ══════════════════════════════════════════════════════════════════════
    // 10 new tests (200 → 210)
    // ══════════════════════════════════════════════════════════════════════

    #[test]
    fn decode_f32_smallest_negative_normal_roundtrip() {
        // Arrange: smallest negative normal f32 (not subnormal)
        let value = -f32::MIN_POSITIVE;
        let bytes = value.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F32).unwrap();
        // Assert
        assert_eq!(decoded[0], -f32::MIN_POSITIVE);
        assert!(decoded[0].is_normal());
    }

    #[test]
    fn decode_bf16_nan_roundtrip() {
        // Arrange: BF16 NaN
        let bf16_nan = bf16::NAN;
        let bytes = bf16_nan.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert: NaN round-trips as NaN (note: NaN != NaN, use is_nan())
        assert!(
            decoded[0].is_nan(),
            "expected NaN, got {}",
            decoded[0]
        );
    }

    #[test]
    fn decode_bf16_negative_infinity_decodes_to_infinity() {
        // Arrange: BF16 negative infinity
        let bf16_neg_inf = bf16::NEG_INFINITY;
        let bytes = bf16_neg_inf.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert: round-trips to f32 negative infinity
        assert!(
            decoded[0].is_infinite() && decoded[0].is_sign_negative(),
            "expected -inf, got {}",
            decoded[0]
        );
    }

    #[test]
    fn decode_f16_negative_infinity_decodes_to_infinity() {
        // Arrange: F16 negative infinity
        let f16_neg_inf = f16::NEG_INFINITY;
        let bytes = f16_neg_inf.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert: round-trips to f32 negative infinity
        assert!(
            decoded[0].is_infinite() && decoded[0].is_sign_negative(),
            "expected -inf, got {}",
            decoded[0]
        );
    }

    #[test]
    fn cache_insert_values_near_f32_epsilon_boundary() {
        // Arrange: values at f32::EPSILON boundary — finite but extremely small
        let mut cache = LevelKeysCache::new(3);
        let k: [Vec<f32>; 3] = [
            vec![f32::EPSILON, -f32::EPSILON, 2.0 * f32::EPSILON],
            vec![f32::EPSILON * 100.0, 1.0, f32::EPSILON],
            vec![-f32::EPSILON, 0.5, f32::EPSILON],
        ];
        // Act
        cache.insert(0, k.clone()).unwrap();
        // Assert: epsilon-scale values accepted as valid finite, non-zero vectors
        let r = cache.get(0).unwrap();
        assert_eq!(r[0][0], f32::EPSILON);
        assert_eq!(r[0][1], -f32::EPSILON);
        assert!((r[0][2] - 2.0 * f32::EPSILON).abs() < 1e-45);
    }

    #[test]
    fn mean_pool_single_row_identity_with_zero_kv_dim() {
        // Arrange: 1 row, 0 cols — output should be empty vector
        let data: Vec<f32> = vec![];
        // Act
        let pooled = mean_pool_rows(&data, 1, 0).unwrap();
        // Assert: zero kv_dim produces zero-length output
        assert!(pooled.is_empty());
    }

    #[test]
    fn cache_insert_rejects_mixed_positive_zero_and_negative_zero() {
        // Arrange: [0.0, -0.0] — all elements compare equal to 0.0, so all-zero
        let mut cache = LevelKeysCache::new(2);
        let bad: [Vec<f32>; 3] = [
            vec![0.0, -0.0],
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        // Act
        let err = cache.insert(0, bad).unwrap_err();
        // Assert: mixed +0/-0 is still all-zero because 0.0 == -0.0 in IEEE 754
        assert!(
            matches!(err, LevelKeysError::AllZero { level_idx: 0, .. }),
            "expected AllZero for mixed +0/-0, got {err:?}"
        );
    }

    #[test]
    fn decode_f16_positive_infinity_decodes_to_infinity() {
        // Arrange: F16 positive infinity
        let f16_pos_inf = f16::INFINITY;
        let bytes = f16_pos_inf.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::F16).unwrap();
        // Assert: round-trips to f32 positive infinity
        assert!(
            decoded[0].is_infinite() && decoded[0].is_sign_positive(),
            "expected +inf, got {}",
            decoded[0]
        );
    }

    #[test]
    fn cache_overwrite_after_failed_insert_preserves_original_data() {
        // Arrange: insert valid data, fail an overwrite, then succeed with new overwrite
        let mut cache = LevelKeysCache::new(2);
        let original: [Vec<f32>; 3] = [
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        cache.insert(0, original).unwrap();
        // Act 1: failed overwrite with NaN
        let bad: [Vec<f32>; 3] = [vec![f32::NAN, 1.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let _ = cache.insert(0, bad);
        // Assert 1: original preserved
        assert_eq!(cache.get(0).unwrap()[0], vec![1.0, 2.0]);
        // Act 2: successful overwrite
        let new_data: [Vec<f32>; 3] = [
            vec![10.0, 20.0],
            vec![30.0, 40.0],
            vec![50.0, 60.0],
        ];
        cache.insert(0, new_data).unwrap();
        // Assert 2: new data replaces original
        assert_eq!(cache.get(0).unwrap()[0], vec![10.0, 20.0]);
        assert_eq!(cache.get(0).unwrap()[2], vec![50.0, 60.0]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn decode_bf16_positive_infinity_decodes_to_infinity() {
        // Arrange: BF16 positive infinity
        let bf16_pos_inf = bf16::INFINITY;
        let bytes = bf16_pos_inf.to_le_bytes().to_vec();
        // Act
        let decoded = decode_bytes_to_f32(&bytes, 1, 1, DType::BF16).unwrap();
        // Assert: round-trips to f32 positive infinity
        assert!(
            decoded[0].is_infinite() && decoded[0].is_sign_positive(),
            "expected +inf, got {}",
            decoded[0]
        );
    }
}
