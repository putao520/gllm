//! TP Weight Sharding — Tensor Parallelism 权重分片 (REQ-DIST-004)
//!
//! 在权重加载时根据 `ParallelConfig.rank` 和 `ParallelConfig.tp_size` 执行
//! 张量并行分片。列切分 (Column Parallel) 和行切分 (Row Parallel) 两种策略。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::ParallelConfig;

// ── ShardStrategy (REQ-DIST-004) ──────────────────────────────────────────

/// 权重分片策略
// @trace REQ-DIST-004 [entity:ENT-DIST-TP-SHARD]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardStrategy {
    /// 列切分：按输出维度/列维度切分。
    /// QKVProj 在 head 维度切分，rank r 获取 heads[r*H/tp_size..(r+1)*H/tp_size]。
    /// lm_head 在输出维度切分。
    ColumnParallel,
    /// 行切分：按输入维度/行维度切分。
    /// OProj 在输入维度切分（对应列切分 QKV 的输出拼接）。
    /// DownProj 在输入维度切分（对应列切分 GateUp 的输出拼接）。
    RowParallel,
}

// ── shard_weight (REQ-DIST-004) ───────────────────────────────────────────

/// 对权重张量执行分片。
///
/// 权重布局约定：`[rows, cols]` 行主序存储（C-contiguous），
/// 即 `data[r * cols + c]` = 第 r 行第 c 列元素。
///
/// 列切分 (ColumnParallel): `shape [rows, cols] → [rows, cols/tp_size]`
///   每个 rank 取 cols/tp_size 列（连续列块），即 rank r 取
///   `data[r * cols + start_col .. r * cols + start_col + shard_cols]`。
///
/// 行切分 (RowParallel): `shape [rows, cols] → [rows/tp_size, cols]`
///   每个 rank 取 rows/tp_size 行（连续行块），即 rank r 取
///   `data[start_row * cols .. (start_row + shard_rows) * cols]`。
///
/// 当 `tp_size <= 1` 时立即返回 Ok(())（单机无需分片）。
/// 当维度不能被 tp_size 整除时返回 Err。
// @trace REQ-DIST-004 [entity:ENT-DIST-TP-SHARD] [dataflow:DF-DIST-002]
pub fn shard_weight(
    data: &mut Vec<f32>,
    rows: usize,
    cols: usize,
    config: &ParallelConfig,
    strategy: ShardStrategy,
) -> Result<(), String> {
    if config.tp_size <= 1 {
        return Ok(());
    }
    let rank = config.rank as usize;
    let tp_size = config.tp_size as usize;

    match strategy {
        ShardStrategy::ColumnParallel => {
            let shard_cols = cols / tp_size;
            if shard_cols * tp_size != cols {
                return Err(format!(
                    "ColumnParallel: cols={} not divisible by tp_size={}",
                    cols, tp_size
                ));
            }
            if data.len() != rows * cols {
                return Err(format!(
                    "ColumnParallel: data length={} doesn't match rows={} * cols={}",
                    data.len(),
                    rows,
                    cols
                ));
            }
            let start_col = rank * shard_cols;
            let mut sharded = Vec::with_capacity(rows * shard_cols);
            for r in 0..rows {
                let row_start = r * cols + start_col;
                sharded.extend_from_slice(&data[row_start..row_start + shard_cols]);
            }
            *data = sharded;
            Ok(())
        }
        ShardStrategy::RowParallel => {
            let shard_rows = rows / tp_size;
            if shard_rows * tp_size != rows {
                return Err(format!(
                    "RowParallel: rows={} not divisible by tp_size={}",
                    rows, tp_size
                ));
            }
            if data.len() != rows * cols {
                return Err(format!(
                    "RowParallel: data length={} doesn't match rows={} * cols={}",
                    data.len(),
                    rows,
                    cols
                ));
            }
            let start_row = rank * shard_rows;
            let offset = start_row * cols;
            let len = shard_rows * cols;
            *data = data[offset..offset + len].to_vec();
            Ok(())
        }
    }
}

// ── infer_shard_strategy (REQ-DIST-004) ───────────────────────────────────

/// 根据权重名称推断分片策略。
///
/// 使用 `TensorRole` 的 canonical 名称后缀匹配（与 `match_tensor_role`
/// 的 SUFFIX_PATTERNS 对齐），而非 `contains()` 启发式。
///
/// 返回 `None` 表示该权重不需要分片（如 embedding, layer_norm 等）。
///
/// 映射规则:
/// - QKVProj (q_proj/k_proj/v_proj) → ColumnParallel
/// - OProj (o_proj/out_proj) → RowParallel
/// - GateUpProj (gate_proj/up_proj) → ColumnParallel
/// - DownProj (down_proj) → RowParallel
/// - lm_head/output → ColumnParallel
/// - MLA q_a_proj/q_b_proj/kv_b_proj → ColumnParallel
/// - MLA k_pe_proj → None（位置编码不参与 TP 切分）
/// - MoE shared_expert gate/up → ColumnParallel
/// - MoE shared_expert down → RowParallel
/// - MoE gate/router → None（路由表不切分，每个 rank 持有完整路由权重）
// @trace REQ-DIST-004 [entity:ENT-DIST-TP-SHARD] [dataflow:DF-DIST-002]
pub fn infer_shard_strategy(weight_name: &str) -> Option<ShardStrategy> {
    // Strip any layer prefix like "L0.", "L12.", "mtp_proj.1."
    // to get the core projection suffix.
    let suffix = weight_name
        .rsplit('.')
        .next()
        .unwrap_or(weight_name);

    match suffix {
        // ── Attention projections ──
        "q_proj" | "k_proj" | "v_proj" | "qkv_proj" => Some(ShardStrategy::ColumnParallel),
        "o_proj" | "out_proj" => Some(ShardStrategy::RowParallel),

        // ── FFN projections ──
        "gate_proj" | "up_proj" => Some(ShardStrategy::ColumnParallel),
        "down_proj" => Some(ShardStrategy::RowParallel),

        // ── Output head ──
        "lm_head" | "output" | "score" | "classifier" => Some(ShardStrategy::ColumnParallel),

        // ── MLA projections ──
        "q_a_proj" | "q_b_proj" | "kv_b_proj" | "k_b_proj" | "v_b_proj" => {
            Some(ShardStrategy::ColumnParallel)
        }
        // k_pe_proj: position encoding, no TP sharding
        "k_pe_proj" => None,

        // ── MoE shared expert ──
        // These follow the same pattern as dense FFN projections
        // (the suffix after "shared_expert." is "gate_proj"/"up_proj"/"down_proj",
        //  which is already handled above)

        // ── Everything else: norms, embeddings, biases, MoE gate/router ──
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(tp_size: u32, rank: u32) -> ParallelConfig {
        ParallelConfig {
            tp_size,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank,
            world_size: tp_size,
            unique_id: String::new(),
        }
    }

    // ── shard_weight: column parallel ──

    #[test]
    fn column_parallel_tp2_even_split() {
        // [2, 4] matrix, tp_size=2
        // [[0,1,2,3],[4,5,6,7]]
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 0);

        shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        // rank 0: cols [0..2] → [[0,1],[4,5]]
        assert_eq!(data, vec![0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn column_parallel_tp2_rank1() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 1);

        shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        // rank 1: cols [2..4] → [[2,3],[6,7]]
        assert_eq!(data, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn column_parallel_tp4_rank2() {
        // [2, 4] matrix, tp_size=4, rank=2
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(4, 2);

        shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        // rank 2: cols [2..3] → [[2],[6]]
        assert_eq!(data, vec![2.0, 6.0]);
    }

    #[test]
    fn column_parallel_not_divisible() {
        let mut data = vec![0.0f32; 6];
        let config = make_config(4, 0);

        let result = shard_weight(&mut data, 2, 3, &config, ShardStrategy::ColumnParallel);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not divisible"));
    }

    // ── shard_weight: row parallel ──

    #[test]
    fn row_parallel_tp2_rank0() {
        // [4, 2] matrix, tp_size=2
        // [[0,1],[2,3],[4,5],[6,7]]
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 0);

        shard_weight(&mut data, 4, 2, &config, ShardStrategy::RowParallel).unwrap();
        // rank 0: rows [0..2] → [[0,1],[2,3]]
        assert_eq!(data, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn row_parallel_tp2_rank1() {
        let mut data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let config = make_config(2, 1);

        shard_weight(&mut data, 4, 2, &config, ShardStrategy::RowParallel).unwrap();
        // rank 1: rows [2..4] → [[4,5],[6,7]]
        assert_eq!(data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn row_parallel_not_divisible() {
        let mut data = vec![0.0f32; 6];
        let config = make_config(4, 0);

        let result = shard_weight(&mut data, 3, 2, &config, ShardStrategy::RowParallel);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not divisible"));
    }

    // ── shard_weight: single-node no-op ──

    #[test]
    fn shard_weight_tp1_is_noop() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        let config = make_config(1, 0);

        shard_weight(&mut data, 2, 2, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn shard_weight_tp0_is_noop() {
        // tp_size=0 should be treated as single-node (though invalid per validate())
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut data = original.clone();
        let config = ParallelConfig {
            tp_size: 0,
            pp_size: 1,
            ep_size: 1,
                cp_size: 1,
            rank: 0,
            world_size: 0,
            unique_id: String::new(),
        };

        shard_weight(&mut data, 2, 2, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, original);
    }

    // ── shard_weight: data length mismatch ──

    #[test]
    fn column_parallel_data_length_mismatch() {
        let mut data = vec![0.0f32; 5]; // should be 2*4=8
        let config = make_config(2, 0);

        let result = shard_weight(&mut data, 2, 4, &config, ShardStrategy::ColumnParallel);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("doesn't match"));
    }

    #[test]
    fn row_parallel_data_length_mismatch() {
        let mut data = vec![0.0f32; 7]; // should be 4*2=8
        let config = make_config(2, 0);

        let result = shard_weight(&mut data, 4, 2, &config, ShardStrategy::RowParallel);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("doesn't match"));
    }

    // ── infer_shard_strategy ──

    #[test]
    fn infer_attention_query() {
        assert_eq!(
            infer_shard_strategy("L0.q_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_attention_key() {
        assert_eq!(
            infer_shard_strategy("L3.k_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_attention_value() {
        assert_eq!(
            infer_shard_strategy("L5.v_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_attention_output() {
        assert_eq!(
            infer_shard_strategy("L0.o_proj"),
            Some(ShardStrategy::RowParallel)
        );
    }

    #[test]
    fn infer_gate_proj() {
        assert_eq!(
            infer_shard_strategy("L0.gate_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_up_proj() {
        assert_eq!(
            infer_shard_strategy("L2.up_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_down_proj() {
        assert_eq!(
            infer_shard_strategy("L0.down_proj"),
            Some(ShardStrategy::RowParallel)
        );
    }

    #[test]
    fn infer_lm_head() {
        assert_eq!(
            infer_shard_strategy("lm_head"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_embedding_no_shard() {
        assert_eq!(infer_shard_strategy("embed"), None);
    }

    #[test]
    fn infer_norm_no_shard() {
        assert_eq!(infer_shard_strategy("L0.input_layernorm"), None);
    }

    #[test]
    fn infer_final_norm_no_shard() {
        assert_eq!(infer_shard_strategy("final_norm"), None);
    }

    #[test]
    fn infer_moe_gate_no_shard() {
        assert_eq!(infer_shard_strategy("L0.moe_gate"), None);
    }

    #[test]
    fn infer_bias_no_shard() {
        assert_eq!(infer_shard_strategy("L0.q_proj.bias"), None);
        // "bias" suffix → not matched by any rule
    }

    #[test]
    fn infer_mla_projections() {
        assert_eq!(
            infer_shard_strategy("L0.q_a_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
        assert_eq!(
            infer_shard_strategy("L0.q_b_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
        assert_eq!(
            infer_shard_strategy("L0.kv_b_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
        assert_eq!(infer_shard_strategy("L0.k_pe_proj"), None);
    }

    #[test]
    fn infer_score_classifier() {
        assert_eq!(
            infer_shard_strategy("score"),
            Some(ShardStrategy::ColumnParallel)
        );
        assert_eq!(
            infer_shard_strategy("classifier"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_fused_qkv() {
        assert_eq!(
            infer_shard_strategy("L0.qkv_proj"),
            Some(ShardStrategy::ColumnParallel)
        );
    }

    #[test]
    fn infer_out_proj_row() {
        assert_eq!(
            infer_shard_strategy("L0.out_proj"),
            Some(ShardStrategy::RowParallel)
        );
    }

    // ── round-trip: column then row ──

    #[test]
    fn column_parallel_preserves_rows() {
        let mut data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let config = make_config(3, 1);

        shard_weight(&mut data, 3, 4, &config, ShardStrategy::ColumnParallel).unwrap();
        // 3 rows, 4/3 cols = 1 col per rank
        assert_eq!(data.len(), 3 * 1);
    }

    #[test]
    fn row_parallel_preserves_cols() {
        let mut data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let config = make_config(3, 1);

        shard_weight(&mut data, 3, 4, &config, ShardStrategy::RowParallel).unwrap();
        // 3/3 rows = 1 row per rank, 4 cols
        assert_eq!(data.len(), 1 * 4);
    }

    // ── Edge cases ──

    #[test]
    fn shard_weight_empty_data() {
        let mut data: Vec<f32> = vec![];
        let config = make_config(2, 0);

        let result = shard_weight(&mut data, 0, 0, &config, ShardStrategy::ColumnParallel);
        // 0 cols / 2 = 0, divisible; data.len()=0 == 0*0
        assert!(result.is_ok());
        assert!(data.is_empty());
    }

    #[test]
    fn shard_weight_single_element() {
        let mut data = vec![42.0f32];
        let config = make_config(1, 0);

        shard_weight(&mut data, 1, 1, &config, ShardStrategy::ColumnParallel).unwrap();
        assert_eq!(data, vec![42.0]);
    }

    #[test]
    fn shard_strategy_equality() {
        assert_eq!(ShardStrategy::ColumnParallel, ShardStrategy::ColumnParallel);
        assert_eq!(ShardStrategy::RowParallel, ShardStrategy::RowParallel);
        assert_ne!(ShardStrategy::ColumnParallel, ShardStrategy::RowParallel);
    }

    #[test]
    fn shard_strategy_copy() {
        let a = ShardStrategy::ColumnParallel;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn shard_strategy_debug_format() {
        let col = format!("{:?}", ShardStrategy::ColumnParallel);
        let row = format!("{:?}", ShardStrategy::RowParallel);
        assert!(col.contains("ColumnParallel"));
        assert!(row.contains("RowParallel"));
        assert_ne!(col, row);
    }

    #[test]
    fn shard_strategy_hash_consistency() {
        use std::collections::HashSet;
        let set: HashSet<ShardStrategy> = [
            ShardStrategy::ColumnParallel,
            ShardStrategy::RowParallel,
        ]
        .into_iter()
        .collect();
        assert_eq!(set.len(), 2);
    }

    // ── Larger matrix test ──

    #[test]
    fn column_parallel_4x8_tp2() {
        // [4, 8] matrix with sequential values
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let config = make_config(2, 1);

        shard_weight(&mut data, 4, 8, &config, ShardStrategy::ColumnParallel).unwrap();
        // rank 1: cols [4..8] of each row
        assert_eq!(data.len(), 4 * 4);
        // Row 0: [4,5,6,7]
        assert_eq!(data[0..4], [4.0, 5.0, 6.0, 7.0]);
        // Row 1: [12,13,14,15]
        assert_eq!(data[4..8], [12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn row_parallel_8x4_tp2() {
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let config = make_config(2, 1);

        shard_weight(&mut data, 8, 4, &config, ShardStrategy::RowParallel).unwrap();
        // rank 1: rows [4..8]
        assert_eq!(data.len(), 4 * 4);
        // Row 4: [16,17,18,19]
        assert_eq!(data[0..4], [16.0, 17.0, 18.0, 19.0]);
    }
}
