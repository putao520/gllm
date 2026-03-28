fn main() {
    let rows = 8;
    let cols = 16;
    let data = vec![1.0f32; 128];
    let rows_data: Vec<Vec<f32>> = data.chunks(cols).map(|c| c.to_vec()).collect();
    println!("rows_data len: {}", rows_data.len());
    
    // Simulate prune_dead_columns_24
    let mut sp_meta = vec![vec![0u16; 2]; 8];
    for r in 0..8 {
        for grp in 0..4 {
            let meta_u16_idx = grp / 2;
            let meta_shift = (grp % 2) * 4;
            // keep0=0, keep1=1 -> 0 | 1<<2 = 4
            sp_meta[r][meta_u16_idx] |= 4 << meta_shift;
        }
    }
    let pruned_rows = rows_data;
    
    let mut compressed_data = Vec::with_capacity(rows * (cols / 2));
    
    for (r_idx, row) in pruned_rows.iter().enumerate() {
        let meta_row = &sp_meta[r_idx];
        for grp in 0..(cols / 4) {
            let base = grp * 4;
            let meta_u16_idx = grp / 2;
            let meta_shift = (grp % 2) * 4;
            let encoded = (meta_row[meta_u16_idx] >> meta_shift) & 0x0F;
            
            let keep0 = (encoded & 0x03) as usize;
            let keep1 = ((encoded >> 2) & 0x03) as usize;
            
            compressed_data.push(row[base + keep0]);
            compressed_data.push(row[base + keep1]);
        }
    }
    
    println!("compressed len: {}", compressed_data.len());
}
