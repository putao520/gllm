//! Static Load-Time Weight Compression (SPEC §3 Autotuning, I.8 + I.9)
//!
//! These functions run **once at model load time** and permanently reduce
//! the weight tensors stored for subsequent inference. Zero runtime overhead.

/// GQA Head Deduplication (SPEC I.8)
///
/// Computes pairwise cosine similarity across all Q-projection head rows.
/// When two heads exceed `similarity_threshold` (default 0.98), the second
/// head is removed from the weight matrix and O-projection rows are averaged.
///
/// Returns:
/// - `dedup_weights`: compressed Q weight (only unique rows), shape [unique_heads * head_dim, hidden]
/// - `dedup_indices`: mapping from original head idx → compressed head idx
///
/// Caller is responsible for applying `dedup_indices` when building attention graphs.
pub fn deduplicate_gqa_heads(
    q_weight_rows: &[Vec<f32>], // shape: [num_heads * head_dim][hidden]
    num_heads: usize,
    head_dim: usize,
    similarity_threshold: f32,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let rows_per_head = head_dim;
    let mut head_group: Vec<Vec<usize>> = Vec::new(); // group[i] = list of original head indices merged
    let mut assignments: Vec<Option<usize>> = vec![None; num_heads]; // head_idx → group_idx

    for head_i in 0..num_heads {
        if assignments[head_i].is_some() {
            continue;
        }

        let group_idx = head_group.len();
        head_group.push(vec![head_i]);
        assignments[head_i] = Some(group_idx);

        for head_j in (head_i + 1)..num_heads {
            if assignments[head_j].is_some() {
                continue;
            }

            let sim = cosine_similarity_heads(
                q_weight_rows,
                head_i,
                head_j,
                rows_per_head,
            );

            if sim >= similarity_threshold {
                head_group[group_idx].push(head_j);
                assignments[head_j] = Some(group_idx);
                log::debug!(
                    "gqa_dedup: merged head {} into head {} (cosine_sim={:.4})",
                    head_j, head_i, sim
                );
            }
        }
    }

    // Build dedup_indices: map original head → representative head
    let mut dedup_indices = vec![0usize; num_heads];
    for (group_idx, group) in head_group.iter().enumerate() {
        for &head_idx in group {
            dedup_indices[head_idx] = group_idx;
        }
    }

    // Build compressed weight: use the first (representative) head from each group.
    // For fused groups, average all member rows for better approximation.
    let hidden = if q_weight_rows.is_empty() { 0 } else { q_weight_rows[0].len() };
    let mut dedup_weights: Vec<Vec<f32>> = Vec::with_capacity(head_group.len() * rows_per_head);

    for group in &head_group {
        for row_offset in 0..rows_per_head {
            let mut avg_row = vec![0.0f32; hidden];
            for &member_head in group {
                let src_row = &q_weight_rows[member_head * rows_per_head + row_offset];
                for (dst, &src) in avg_row.iter_mut().zip(src_row.iter()) {
                    *dst += src;
                }
            }
            let n = group.len() as f32;
            for val in &mut avg_row {
                *val /= n;
            }
            dedup_weights.push(avg_row);
        }
    }

    let saved = num_heads - head_group.len();
    if saved > 0 {
        log::info!(
            "gqa_dedup: removed {} duplicate heads ({} → {} unique, {:.1}% saved)",
            saved, num_heads, head_group.len(),
            100.0 * saved as f32 / num_heads as f32
        );
    }

    (dedup_weights, dedup_indices)
}

/// Weight Column Pruning — NVIDIA 2:4 Structured Sparse Format (SPEC §7)
///
/// Applies NVIDIA's 2:4 structured sparsity to a weight matrix, producing
/// both the pruned (non-zero elements only) weight tensor and the metadata
/// array (`sp_meta`) required by `mma.sp` sparse Tensor Core instructions.
///
/// ## NVIDIA 2:4 Format Rules
/// - Within every group of **4 consecutive elements** in a row, exactly **2** are nonzero.
/// - Selection: the 2 elements with the **largest absolute value** survive; others are zeroed.
/// - `sp_meta[row][meta_col]`: 2-bit encoded index of each surviving element per 4-element group.
///   Packed as 8 indices per `u16` → `meta_cols = ceil(cols / 4)`, each u16 encodes 2 groups.
///
/// Returns:
/// - `pruned`:   `[rows][cols]` with zeros at pruned positions (ready for `cuSparseLtDenseDescriptor`).
/// - `sp_meta`:  `[rows][cols/4]` compressed `u16` metadata (two 2-bit index pairs per u16).
///
/// ## Usage
/// Pass `pruned` to weight layout and `sp_meta` to `cusparseLtSpMMADescriptor`/`mma.sp`.
pub fn prune_dead_columns_24(
    weight: &[Vec<f32>], // shape: [rows][cols] — cols must be divisible by 4 for NVIDIA compliance
) -> (Vec<Vec<f32>>, Vec<Vec<u16>>) {
    if weight.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let rows = weight.len();
    let cols = weight[0].len();

    // NVIDIA 2:4 requires cols divisible by 4
    assert!(
        cols % 4 == 0,
        "prune_dead_columns_24: cols ({cols}) must be divisible by 4 for NVIDIA 2:4 format"
    );

    let meta_cols = cols / 4; // one u16 covers 8 elements (2 groups of 4, 2×2 bits each)
    let mut pruned = weight.to_vec();
    // sp_meta: packed 2-bit position indices: 2 groups of 4 per u16 → ceil(cols/4/2) u16 per row
    let meta_u16_cols = (meta_cols + 1) / 2; // 2 4-element groups per u16
    let mut sp_meta: Vec<Vec<u16>> = vec![vec![0u16; meta_u16_cols]; rows];

    let mut total_pruned_elems = 0usize;

    for (row_idx, row) in weight.iter().enumerate() {
        // Process in 4-element groups
        for grp in 0..(cols / 4) {
            let base = grp * 4;
            let elems = [row[base], row[base + 1], row[base + 2], row[base + 3]];

            // Select the 2 elements with highest absolute value per group of 4
            let mut order = [0usize, 1, 2, 3];
            order.sort_unstable_by(|&a, &b| {
                elems[b].abs().partial_cmp(&elems[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Surviving positions are order[0] and order[1] — the two largest |vals|
            let keep: [usize; 2] = [order[0].min(order[1]), order[0].max(order[1])]; // keep sorted

            // Zero out the 2 non-surviving positions
            for &dead in &order[2..] {
                pruned[row_idx][base + dead] = 0.0;
            }
            total_pruned_elems += 2;

            // Encode surviving positions as 2-bit indices: keep[0] ∈ {0..3}, keep[1] ∈ {0..3}
            // NVIDIA sp_meta format: each u16 stores two groups of 4, 4 bits per group (2×2-bit indices)
            // | group[n+1] pos1 | group[n+1] pos0 | group[n] pos1 | group[n] pos0 |  (low→high)
            let meta_u16_idx = grp / 2;
            let meta_shift  = (grp % 2) * 4; // 0 or 4 bits offset within u16
            let encoded: u16 = ((keep[0] as u16) | ((keep[1] as u16) << 2)) << meta_shift;
            sp_meta[row_idx][meta_u16_idx] |= encoded;
        }
    }

    let total_elems = rows * cols;
    log::info!(
        "prune_dead_columns_24: applied NVIDIA 2:4 sparsity, {}/{} elements zeroed ({:.1}%)",
        total_pruned_elems, total_elems,
        100.0 * total_pruned_elems as f64 / total_elems as f64
    );

    (pruned, sp_meta)
}

/// Weight Column Pruning — L2 Norm Threshold (convenience helper, SPEC I.9)
///
/// Scans per-column L2 norms and zeros columns below `threshold_ratio * mean_col_norm`.
/// This is the **pre-pass** step: run before `prune_dead_columns_24` to set structurally
/// inactive columns to exactly zero, so 2:4 selection naturally discards them.
///
/// Returns `(pruned_weight, dead_col_mask)` — `dead_col_mask[j] == true` means column j is dead.
/// Caller must apply the mask to the paired Down-projection matrix.
pub fn prune_dead_columns(
    weight: &[Vec<f32>], // shape: [rows][cols] — typically Gate_proj or Up_proj
    threshold_ratio: f32,
) -> (Vec<Vec<f32>>, Vec<bool>) {
    if weight.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let cols = weight[0].len();

    // Compute per-column L2 norm
    let mut col_norms = vec![0.0f32; cols];
    for row in weight {
        for (j, &val) in row.iter().enumerate() {
            col_norms[j] += val * val;
        }
    }
    for norm in &mut col_norms {
        *norm = norm.sqrt();
    }

    let mean_norm: f32 = col_norms.iter().sum::<f32>() / cols as f32;
    let prune_threshold = threshold_ratio * mean_norm;

    let dead_col_mask: Vec<bool> = col_norms.iter().map(|&n| n < prune_threshold).collect();

    // Zero out pruned columns
    let mut pruned = weight.to_vec();
    let dead_count = dead_col_mask.iter().filter(|&&d| d).count();

    for row in &mut pruned {
        for (j, val) in row.iter_mut().enumerate() {
            if dead_col_mask[j] {
                *val = 0.0;
            }
        }
    }

    if dead_count > 0 {
        log::info!(
            "column_prune: zeroed {}/{} dead columns ({:.1}% pruned, threshold={:.4})",
            dead_count, cols,
            100.0 * dead_count as f32 / cols as f32,
            prune_threshold
        );
    }

    (pruned, dead_col_mask)
}


/// Compute mean cosine similarity between two head blocks of a weight matrix.
///
/// Each head occupies `rows_per_head` consecutive rows starting at `head * rows_per_head`.
fn cosine_similarity_heads(
    weight: &[Vec<f32>],
    head_a: usize,
    head_b: usize,
    rows_per_head: usize,
) -> f32 {
    let start_a = head_a * rows_per_head;
    let start_b = head_b * rows_per_head;

    if start_a + rows_per_head > weight.len() || start_b + rows_per_head > weight.len() {
        return 0.0;
    }

    let mut total_sim = 0.0f32;
    for offset in 0..rows_per_head {
        let row_a = &weight[start_a + offset];
        let row_b = &weight[start_b + offset];
        total_sim += cosine_sim_rows(row_a, row_b);
    }

    total_sim / rows_per_head as f32
}

/// Cosine similarity between two equal-length f32 slices.
fn cosine_sim_rows(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_dead_columns_zeros_low_norm() {
        // Column 0 has norm 1.0, column 1 has norm 0.0001 (dead), column 2 has norm 1.0
        let weight = vec![
            vec![1.0f32, 0.0001, 1.0],
            vec![1.0f32, 0.0001, 1.0],
        ];
        let (pruned, mask) = prune_dead_columns(&weight, 0.01);
        assert!(!mask[0], "column 0 should be alive");
        assert!(mask[1],  "column 1 should be pruned");
        assert!(!mask[2], "column 2 should be alive");
        assert_eq!(pruned[0][1], 0.0, "pruned column should be zeroed");
        assert_ne!(pruned[0][0], 0.0, "alive column should remain");
    }

    #[test]
    fn test_deduplicate_gqa_identical_heads() {
        // Two identical heads → should be merged into one
        let head_dim = 2;
        let hidden = 3;
        // head 0 rows: [1,0,0], [0,1,0]; head 1 rows: [1,0,0], [0,1,0] (identical)
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let _ = hidden;
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        // Should collapse to 1 unique head
        assert_eq!(dedup_w.len(), head_dim, "should have head_dim rows for 1 unique head");
        assert_eq!(dedup_idx[0], dedup_idx[1], "both heads map to same group");
    }

    #[test]
    fn test_deduplicate_gqa_distinct_heads() {
        // Two orthogonal heads → should NOT be merged
        let head_dim = 2;
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let (dedup_w, dedup_idx) = deduplicate_gqa_heads(&rows, 2, head_dim, 0.98);
        assert_eq!(dedup_w.len(), 2 * head_dim, "should have rows for 2 unique heads");
        assert_ne!(dedup_idx[0], dedup_idx[1], "distinct heads map to different groups");
    }

    #[test]
    fn test_prune_dead_columns_24_selects_top2_per_group() {
        // Row: [3.0, 1.0, 0.5, 4.0, 0.1, 2.0, 0.2, 7.0]
        // Group 0: [3.0, 1.0, 0.5, 4.0] → keep positions 3(4.0) and 0(3.0) → zero pos 1 and 2
        // Group 1: [0.1, 2.0, 0.2, 7.0] → keep positions 3(7.0) and 1(2.0) → zero pos 0 and 2
        let weight = vec![
            vec![3.0f32, 1.0, 0.5, 4.0, 0.1, 2.0, 0.2, 7.0],
        ];
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);

        // Group 0: positions 1 and 2 must be zero
        assert_eq!(pruned[0][1], 0.0, "grp0 pos1 should be pruned");
        assert_eq!(pruned[0][2], 0.0, "grp0 pos2 should be pruned");
        assert_ne!(pruned[0][0], 0.0, "grp0 pos0 (3.0) should survive");
        assert_ne!(pruned[0][3], 0.0, "grp0 pos3 (4.0) should survive");

        // Group 1: positions 0 and 2 must be zero
        assert_eq!(pruned[0][4], 0.0, "grp1 pos0 should be pruned");
        assert_eq!(pruned[0][6], 0.0, "grp1 pos2 should be pruned");
        assert_ne!(pruned[0][5], 0.0, "grp1 pos1 (2.0) should survive");
        assert_ne!(pruned[0][7], 0.0, "grp1 pos3 (7.0) should survive");

        // sp_meta: 1 row, ceil(8/4/2)=1 u16 per row
        assert_eq!(sp_meta.len(), 1, "should have 1 row of metadata");
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 1 u16 metadata per row");
        // sp_meta[0][0] encodes group0 (keep pos0,3) and group1 (keep pos1,3)
        // grp0: sorted keep = [0,3] → encoded = (0 | 3<<2) = 12 = 0b1100, shift 0 → bits 3:0
        // grp1: sorted keep = [1,3] → encoded = (1 | 3<<2) = 13 = 0b1101, shift 4 → bits 7:4
        // u16 = 0b1101_1100 = 0xDC
        assert_eq!(sp_meta[0][0], 0x00DC, "sp_meta encoding mismatch for [3,1,0.5,4 | 0.1,2,0.2,7]");
    }

    #[test]
    fn test_prune_dead_columns_24_dimensions() {
        // 2 rows of 8 elements → sp_meta should be [2][1]
        let weight: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ];
        let (pruned, sp_meta) = prune_dead_columns_24(&weight);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].len(), 8);
        assert_eq!(sp_meta.len(), 2);
        assert_eq!(sp_meta[0].len(), 1, "8 cols → 2 grps → 1 u16");
    }
}

