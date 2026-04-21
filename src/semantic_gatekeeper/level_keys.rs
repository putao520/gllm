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
