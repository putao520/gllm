//! TP Weight Sharding — Tensor Parallelism 权重分片 (REQ-DIST-004)
//! PP Weight Filtering — Pipeline Parallelism 权重分区过滤 (REQ-DIST-019)
//!
//! TP 分片：在权重加载时根据 `ParallelConfig.rank` 和 `ParallelConfig.tp_size` 执行
//! 张量并行分片。列切分 (Column Parallel) 和行切分 (Row Parallel) 两种策略。
//!
//! PP 分区：根据 `stage_id` 和 `pp_size` 过滤权重，仅加载本 stage 负责的层权重。
//! 共享层（Embedding/LayerNorm）被所有 stage 加载。
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

// ── PP Weight Filtering (REQ-DIST-019) ─────────────────────────────────────

/// 判断权重名称是否属于共享层（所有 PP stage 都需加载）(REQ-DIST-019)
///
/// 共享层包括：
/// - Embedding 层（`embed_tokens`/`wte`/`word_embeddings`）
/// - 最终 LayerNorm（`final_norm`/`norm`/`ln_f`）
/// - 输出头（`lm_head`/`output`/`score`/`classifier`）
///
/// 注意：`lm_head` 虽然会被 TP 切分，但在 PP 维度上所有 stage 都需要它
/// （stage 0 需要它做 token embedding，最后一个 stage 需要它做输出）。
/// 实际上 `lm_head` 通常只在最后一个 stage 使用，但 SPEC 明确要求共享层
/// 被所有 stage 加载，这里按最保守策略处理。
// @trace REQ-DIST-019 [entity:ENT-DIST-PP-SHARD]
pub fn is_shared_weight(weight_name: &str) -> bool {
    let suffix = weight_name
        .rsplit('.')
        .next()
        .unwrap_or(weight_name);

    match suffix {
        // ── Embedding layers (shared by all stages) ──
        "embed_tokens" | "wte" | "word_embeddings" | "embed" | "embedding" => true,

        // ── Final normalization (shared by all stages) ──
        "final_norm" | "norm" | "ln_f" | "layer_norm" => {
            // "norm" is ambiguous — could be a final norm or a per-layer norm.
            // Heuristic: if there's no "L" prefix, it's likely a final/shared norm.
            // Per-layer norms like "L0.input_layernorm" have a layer prefix,
            // which means the suffix would be "input_layernorm", not "norm".
            !weight_name.contains("L") || weight_name.starts_with("norm")
        }

        // ── Output head (shared — TP-sharded but PP-replicated) ──
        "lm_head" | "output" | "score" | "classifier" => true,

        // ── Position embedding (shared by all stages) ──
        "wpe" | "position_embedding" | "position_embeddings" => true,

        _ => false,
    }
}

/// 从权重名称中提取层索引 (REQ-DIST-019)
///
/// 权重命名约定：`"L{layer_idx}.{suffix}"`，如 `"L0.q_proj"`、`"L15.gate_proj"`。
/// 返回 `None` 表示非层相关权重（如 embedding、final_norm、lm_head）。
///
/// 识别模式：
/// - `L{N}.xxx` — 标准层权重
/// - `mtp_proj.{N}.xxx` — MTP 投影层权重
// @trace REQ-DIST-019 [entity:ENT-DIST-PP-SHARD]
pub fn extract_layer_index(weight_name: &str) -> Option<u32> {
    // Standard pattern: "L{N}.suffix"
    if weight_name.starts_with('L') {
        let after_l = &weight_name[1..];
        if let Some(dot_pos) = after_l.find('.') {
            if let Ok(idx) = after_l[..dot_pos].parse::<u32>() {
                return Some(idx);
            }
        }
    }

    // MTP pattern: "mtp_proj.{N}.suffix"
    if weight_name.starts_with("mtp_proj.") {
        let after_prefix = &weight_name["mtp_proj.".len()..];
        if let Some(dot_pos) = after_prefix.find('.') {
            if let Ok(idx) = after_prefix[..dot_pos].parse::<u32>() {
                return Some(idx);
            }
        }
    }

    None
}

/// 判断权重是否应被当前 PP stage 加载 (REQ-DIST-019)
///
/// 根据 `stage_id` 和 `pp_size` 计算层范围 `[stage_id * L/pp, (stage_id+1) * L/pp)`，
/// 仅加载该范围内层的权重。共享层（embedding/final_norm/lm_head）被所有 stage 加载。
///
/// 当 `pp_size <= 1` 时返回 `true`（单机模式，所有权重都加载）。
///
/// # Arguments
/// * `weight_name` - 权重名称，如 `"L0.q_proj"`、`"lm_head"`
/// * `stage_id` - 当前 stage ID，范围 [0, pp_size)
/// * `pp_size` - Pipeline Parallel 维度
/// * `num_layers` - 模型总层数
// @trace REQ-DIST-019 [entity:ENT-DIST-PP-SHARD] [dataflow:DF-DIST-004]
pub fn should_load_weight_for_stage(
    weight_name: &str,
    stage_id: u32,
    pp_size: u32,
    num_layers: u32,
) -> bool {
    // pp_size <= 1: single-device mode, load everything
    if pp_size <= 1 {
        return true;
    }

    // Shared layers are loaded by all stages (REQ-DIST-019 验收标准 4)
    if is_shared_weight(weight_name) {
        return true;
    }

    // Extract layer index; non-layer weights (no "L" prefix) are treated as shared
    let Some(layer_idx) = extract_layer_index(weight_name) else {
        // Weights without a layer prefix (e.g., global bias, learned scaling)
        // are shared by all stages
        return true;
    };

    // Compute this stage's layer range: [stage_id * L/pp, (stage_id+1) * L/pp)
    let layers_per_stage = (num_layers + pp_size - 1) / pp_size; // ceil division
    let start = stage_id * layers_per_stage;
    let end = ((stage_id + 1) * layers_per_stage).min(num_layers);

    layer_idx >= start && layer_idx < end
}

/// 过滤权重名称列表，仅保留当前 PP stage 应加载的权重 (REQ-DIST-019)
///
/// 便利函数，对一组权重名称调用 `should_load_weight_for_stage` 并返回
/// 过滤后的列表。未分配到本 stage 的层权重不占用内存（零加载）。
// @trace REQ-DIST-019 [entity:ENT-DIST-PP-SHARD] [dataflow:DF-DIST-004]
pub fn filter_weights_by_stage<'a>(
    weight_names: &'a [String],
    stage_id: u32,
    pp_size: u32,
    num_layers: u32,
) -> Vec<&'a String> {
    weight_names
        .iter()
        .filter(|name| should_load_weight_for_stage(name, stage_id, pp_size, num_layers))
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(tp_size: u32, rank: u32) -> ParallelConfig {
        let cp_size: u32 = 1;
        ParallelConfig {
            tp_size,
            pp_size: 1,
            ep_size: 1,
            cp_size,
            rank,
            world_size: tp_size,
            unique_id: String::new(),
            stage_id: rank / (tp_size * cp_size),
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
            stage_id: 0,
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

    // ═══════════════════════════════════════════════════════════════════════
    // PP Weight Filtering tests (REQ-DIST-019)
    // ═══════════════════════════════════════════════════════════════════════

    // ── is_shared_weight ──

    #[test]
    fn shared_weight_embed_tokens() {
        assert!(is_shared_weight("embed_tokens"));
    }

    #[test]
    fn shared_weight_wte() {
        assert!(is_shared_weight("wte"));
    }

    #[test]
    fn shared_weight_word_embeddings() {
        assert!(is_shared_weight("word_embeddings"));
    }

    #[test]
    fn shared_weight_final_norm() {
        assert!(is_shared_weight("final_norm"));
    }

    #[test]
    fn shared_weight_ln_f() {
        assert!(is_shared_weight("ln_f"));
    }

    #[test]
    fn shared_weight_lm_head() {
        assert!(is_shared_weight("lm_head"));
    }

    #[test]
    fn shared_weight_output() {
        assert!(is_shared_weight("output"));
    }

    #[test]
    fn shared_weight_wpe() {
        assert!(is_shared_weight("wpe"));
    }

    #[test]
    fn not_shared_layer_weight() {
        assert!(!is_shared_weight("L0.q_proj"));
        assert!(!is_shared_weight("L3.gate_proj"));
        assert!(!is_shared_weight("L15.down_proj"));
    }

    #[test]
    fn not_shared_moe_gate() {
        assert!(!is_shared_weight("L0.moe_gate"));
    }

    // ── extract_layer_index ──

    #[test]
    fn extract_layer_index_standard() {
        assert_eq!(extract_layer_index("L0.q_proj"), Some(0));
        assert_eq!(extract_layer_index("L3.gate_proj"), Some(3));
        assert_eq!(extract_layer_index("L15.down_proj"), Some(15));
        assert_eq!(extract_layer_index("L31.o_proj"), Some(31));
    }

    #[test]
    fn extract_layer_index_mtp() {
        assert_eq!(extract_layer_index("mtp_proj.0.q_proj"), Some(0));
        assert_eq!(extract_layer_index("mtp_proj.1.q_proj"), Some(1));
    }

    #[test]
    fn extract_layer_index_non_layer() {
        assert_eq!(extract_layer_index("embed_tokens"), None);
        assert_eq!(extract_layer_index("lm_head"), None);
        assert_eq!(extract_layer_index("final_norm"), None);
    }

    #[test]
    fn extract_layer_index_invalid_format() {
        assert_eq!(extract_layer_index("Lx.q_proj"), None);
        assert_eq!(extract_layer_index("L.q_proj"), None);
    }

    // ── should_load_weight_for_stage ──

    #[test]
    fn pp1_loads_everything() {
        // pp_size=1: single-device mode, all weights loaded
        assert!(should_load_weight_for_stage("L0.q_proj", 0, 1, 32));
        assert!(should_load_weight_for_stage("L15.gate_proj", 0, 1, 32));
        assert!(should_load_weight_for_stage("L31.o_proj", 0, 1, 32));
        assert!(should_load_weight_for_stage("lm_head", 0, 1, 32));
    }

    #[test]
    fn pp2_stage0_loads_first_half() {
        // pp=2, 32 layers: stage 0 = [0, 16), stage 1 = [16, 32)
        assert!(should_load_weight_for_stage("L0.q_proj", 0, 2, 32));
        assert!(should_load_weight_for_stage("L15.gate_proj", 0, 2, 32));
        assert!(!should_load_weight_for_stage("L16.q_proj", 0, 2, 32));
        assert!(!should_load_weight_for_stage("L31.o_proj", 0, 2, 32));
    }

    #[test]
    fn pp2_stage1_loads_second_half() {
        // pp=2, 32 layers: stage 1 = [16, 32)
        assert!(!should_load_weight_for_stage("L0.q_proj", 1, 2, 32));
        assert!(!should_load_weight_for_stage("L15.gate_proj", 1, 2, 32));
        assert!(should_load_weight_for_stage("L16.q_proj", 1, 2, 32));
        assert!(should_load_weight_for_stage("L31.o_proj", 1, 2, 32));
    }

    #[test]
    fn pp4_stage_distribution() {
        // pp=4, 32 layers: 8 layers per stage
        // stage 0 = [0,8), stage 1 = [8,16), stage 2 = [16,24), stage 3 = [24,32)
        for stage in 0..4u32 {
            for layer in 0..32u32 {
                let name = format!("L{}.q_proj", layer);
                let should_load = should_load_weight_for_stage(&name, stage, 4, 32);
                let expected = layer >= stage * 8 && layer < (stage + 1) * 8;
                assert_eq!(should_load, expected, "stage={}, layer={}", stage, layer);
            }
        }
    }

    #[test]
    fn shared_weights_loaded_by_all_stages() {
        // Shared weights are loaded regardless of stage_id
        for stage in 0..4u32 {
            assert!(should_load_weight_for_stage("embed_tokens", stage, 4, 32));
            assert!(should_load_weight_for_stage("final_norm", stage, 4, 32));
            assert!(should_load_weight_for_stage("lm_head", stage, 4, 32));
            assert!(should_load_weight_for_stage("wpe", stage, 4, 32));
        }
    }

    #[test]
    fn non_layer_weights_treated_as_shared() {
        // Weights without layer prefix are treated as shared
        assert!(should_load_weight_for_stage("some_global_weight", 0, 2, 32));
        assert!(should_load_weight_for_stage("some_global_weight", 1, 2, 32));
    }

    #[test]
    fn no_overlap_full_coverage_pp2() {
        // Verify: no layer weight loaded by both stages, all layer weights covered
        let num_layers = 32u32;
        let pp_size = 2u32;
        let mut covered = vec![false; num_layers as usize];

        for layer in 0..num_layers {
            let name = format!("L{}.q_proj", layer);
            let load_by_0 = should_load_weight_for_stage(&name, 0, pp_size, num_layers);
            let load_by_1 = should_load_weight_for_stage(&name, 1, pp_size, num_layers);

            // No overlap: a layer weight is loaded by at most one stage
            assert!(!(load_by_0 && load_by_1), "layer {} loaded by both stages", layer);

            // Full coverage: a layer weight is loaded by at least one stage
            assert!(load_by_0 || load_by_1, "layer {} not loaded by any stage", layer);

            covered[layer as usize] = true;
        }

        assert!(covered.iter().all(|&c| c));
    }

    #[test]
    fn no_overlap_full_coverage_pp4() {
        let num_layers = 32u32;
        let pp_size = 4u32;

        for layer in 0..num_layers {
            let name = format!("L{}.q_proj", layer);
            let mut loaded_by = vec![];
            for stage in 0..pp_size {
                if should_load_weight_for_stage(&name, stage, pp_size, num_layers) {
                    loaded_by.push(stage);
                }
            }
            // Exactly one stage loads this layer weight
            assert_eq!(loaded_by.len(), 1, "layer {} loaded by {:?} stages", layer, loaded_by);
        }
    }

    #[test]
    fn non_divisible_layers_pp4_33layers() {
        // 33 layers, pp=4: layers_per_stage = ceil(33/4) = 9
        // stage 0: [0,9), stage 1: [9,18), stage 2: [18,27), stage 3: [27,33)
        assert!(should_load_weight_for_stage("L8.q_proj", 0, 4, 33));
        assert!(!should_load_weight_for_stage("L9.q_proj", 0, 4, 33));
        assert!(should_load_weight_for_stage("L9.q_proj", 1, 4, 33));
        assert!(should_load_weight_for_stage("L26.q_proj", 2, 4, 33));
        assert!(!should_load_weight_for_stage("L27.q_proj", 2, 4, 33));
        assert!(should_load_weight_for_stage("L27.q_proj", 3, 4, 33));
        assert!(should_load_weight_for_stage("L32.q_proj", 3, 4, 33));
    }

    // ── filter_weights_by_stage ──

    #[test]
    fn filter_weights_pp2_stage0() {
        let names: Vec<String> = vec![
            "embed_tokens".into(),
            "L0.q_proj".into(),
            "L0.k_proj".into(),
            "L15.o_proj".into(),
            "L16.q_proj".into(),
            "L31.down_proj".into(),
            "final_norm".into(),
            "lm_head".into(),
        ];
        let filtered = filter_weights_by_stage(&names, 0, 2, 32);
        let filtered_names: Vec<&str> = filtered.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            filtered_names,
            vec!["embed_tokens", "L0.q_proj", "L0.k_proj", "L15.o_proj", "final_norm", "lm_head"]
        );
    }

    #[test]
    fn filter_weights_pp2_stage1() {
        let names: Vec<String> = vec![
            "embed_tokens".into(),
            "L0.q_proj".into(),
            "L0.k_proj".into(),
            "L15.o_proj".into(),
            "L16.q_proj".into(),
            "L31.down_proj".into(),
            "final_norm".into(),
            "lm_head".into(),
        ];
        let filtered = filter_weights_by_stage(&names, 1, 2, 32);
        let filtered_names: Vec<&str> = filtered.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            filtered_names,
            vec!["embed_tokens", "L16.q_proj", "L31.down_proj", "final_norm", "lm_head"]
        );
    }

    #[test]
    fn filter_weights_pp1_loads_all() {
        let names: Vec<String> = vec![
            "embed_tokens".into(),
            "L0.q_proj".into(),
            "L31.down_proj".into(),
            "lm_head".into(),
        ];
        let filtered = filter_weights_by_stage(&names, 0, 1, 32);
        assert_eq!(filtered.len(), 4);
    }
}
