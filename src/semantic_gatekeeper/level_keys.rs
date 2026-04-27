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
#[derive(Debug, thiserror::Error, Clone)]
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
}
