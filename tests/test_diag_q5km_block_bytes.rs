//! Q5_K_M 权重 blob 逐 block 字节验证 (用户第一性原理第17方向).
//! JIT 机器码已验证对称 (第16方向), bug 不在代码 → 在运行时数据.
//! 验证 Q5K 权重: GGUF 原始 block 字节 → scalar decode → 对比 JIT decode 同 block.
//! 重点: pack 进 blob 的 Q5K block 是否与 GGUF 原始一致 (sig 偏移对 ≠ 字节内容对).
#![cfg(test)]
use gllm::{Client, ModelKind};
use gllm::loader::gguf::GgufReader;
use half::f16;

/// Q5K scalar decode (ground truth, 与 q5k_decode_step_native 同算法)
/// lane_offset 是 block 内偏移 (0..256), 不跨 block. block_idx 仅用于诊断.
fn q5k_scalar_decode_block(block: &[u8], n_elems: usize) -> Vec<f32> {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];   // 12 bytes
    let qh = &block[16..48];      // 32 bytes
    let qs = &block[48..176];     // 128 bytes
    let mut out = Vec::with_capacity(n_elems);
    for i in 0..n_elems {
        // global_elem = lane_offset + i, block 内位置 (0..256)
        let mini = i / 32;
        let l = i % 32;
        let group = mini / 2;
        let half = mini % 2;
        let packed = qs[group * 32 + l];
        let lo4 = if half == 0 { packed & 0x0F } else { packed >> 4 };
        let hi1 = (qh[l] >> mini) & 1;
        let q5 = (lo4 as i32) | ((hi1 as i32) << 4);
        let (sc, m) = if mini < 4 {
            ((scales[mini] & 63) as f32, (scales[mini + 4] & 63) as f32)
        } else {
            let sc = ((scales[mini + 4] & 0x0F) | ((scales[mini - 4] >> 6) << 4)) as f32;
            let m = ((scales[mini + 4] >> 4) | ((scales[mini] >> 6) << 4)) as f32;
            (sc, m)
        };
        out.push(d * sc * (q5 as f32) - dmin * m);
    }
    out
}

#[test]
#[ignore]
fn diag_q5km_gguf_block_decode() {
    eprintln!("\n=== Q5_K_M GGUF block 0 scalar decode (ground truth) ===");
    let path = "/home/putao/.gllm/models/huggingface/models--bartowski--Qwen_Qwen3-0.6B-GGUF/snapshots/60b85c0e3d8fe0f6474f406922a26d12aca4550d/Qwen_Qwen3-0.6B-Q5_K_M.gguf";
    let r = GgufReader::open(path).expect("open gguf");
    let q_proj = r.tensors().iter().find(|t| t.name.as_ref().contains("blk.0.attn_q.weight"))
        .expect("find q_proj");
    eprintln!("[GGUF] q_proj: dtype={:?} offset={} size={}", q_proj.dtype, q_proj.offset, q_proj.size);

    // 读 q_proj 原始字节 (TensorInfo.offset 已是绝对偏移 = data_offset + rel_offset, 勿再加)
    let q_start = q_proj.offset;
    let file_bytes = std::fs::read(path).expect("read file");
    let q_bytes = &file_bytes[q_start..q_start + q_proj.size];
    eprintln!("[GGUF] q_proj raw bytes = {}", q_bytes.len());

    // Q5K block = 176B, block_size=256 elem
    let block_bytes = 176;
    let n_blocks = q_bytes.len() / block_bytes;
    eprintln!("[GGUF] q_proj blocks = {}", n_blocks);

    // decode block 0 前 8 elem (ground truth)
    let block0 = &q_bytes[0..176];
    let d = f16::from_le_bytes([block0[0], block0[1]]).to_f32();
    let dmin = f16::from_le_bytes([block0[2], block0[3]]).to_f32();
    eprintln!("[GGUF] block0 d={:.6} dmin={:.6} scales12={:02x?}", d, dmin, &block0[4..16]);

    let gt = q5k_scalar_decode_block(block0, 8);
    eprintln!("[GGUF] block0 scalar decode first 8 = {:?}", gt);

    // 对比 block 1 (看 block 间变化)
    if n_blocks > 1 {
        let block1 = &q_bytes[176..352];
        let d1 = f16::from_le_bytes([block1[0], block1[1]]).to_f32();
        let dmin1 = f16::from_le_bytes([block1[2], block1[3]]).to_f32();
        eprintln!("[GGUF] block1 d={:.6} dmin={:.6}", d1, dmin1);
        let gt1 = q5k_scalar_decode_block(block1, 8);
        eprintln!("[GGUF] block1 scalar decode first 8 = {:?}", gt1);
    }

    // 验证: GGUF q_proj 的 dtype 必须是 Q5_K
    eprintln!("\n[验证] q_proj dtype = {:?} (应为 Q5_K)", q_proj.dtype);
    if q_proj.dtype != gllm::loader::gguf::GgmlDType::Q5_K {
        eprintln!("[警告] q_proj 不是 Q5_K! pack 可能误判 dtype");
    } else {
        eprintln!("[OK] q_proj 是 Q5_K, dtype 正确");
    }

    // ★关键: 对比 blob 的 L0.q_proj block0 字节 vs GGUF block0 字节
    eprintln!("\n=== ★对比 blob L0.q_proj block0 vs GGUF block0 ===");
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m")
        .build().expect("Q5_K_M client");
    let _ = client.encode("test").expect("encode");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    let blob = client.diagnostic_weight_blob_bytes().expect("weight blob");
    eprintln!("[BLOB] weight_blob len = {} bytes", blob.len());

    let woffs = client.diagnostic_weight_offsets().expect("weight offsets");
    let l0_q_proj = woffs.iter().find(|(n, _, _)| n == "L0.q_proj")
        .expect("find L0.q_proj");
    eprintln!("[BLOB] L0.q_proj: offset={} dtype={:?}", l0_q_proj.1, l0_q_proj.2);

    let blob_q_proj = &blob[l0_q_proj.1..l0_q_proj.1 + 176]; // block0
    let gguf_block0 = &q_bytes[0..176];

    eprintln!("[BLOB] block0 head 16B = {:02x?}", &blob_q_proj[0..16]);
    eprintln!("[GGUF] block0 head 16B = {:02x?}", &gguf_block0[0..16]);

    // 逐字节对比前 176 bytes (整个 block0)
    let mut mismatches = 0;
    let mut first_mismatch = None;
    for i in 0..176 {
        if blob_q_proj[i] != gguf_block0[i] {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some(i);
            }
        }
    }
    if mismatches == 0 {
        eprintln!("[✓] blob L0.q_proj block0 == GGUF block0 (176/176 bytes 匹配) — pack 正确");
    } else {
        eprintln!("[✗] blob L0.q_proj block0 != GGUF block0: {} 处不匹配, 首个在 byte {}",
            mismatches, first_mismatch.unwrap_or(0));
        eprintln!("  blob[{}..{}+8] = {:02x?}", first_mismatch.unwrap_or(0), first_mismatch.unwrap_or(0),
            &blob_q_proj[first_mismatch.unwrap_or(0)..first_mismatch.unwrap_or(0)+8]);
        eprintln!("  gguf[{}..{}+8] = {:02x?}", first_mismatch.unwrap_or(0), first_mismatch.unwrap_or(0),
            &gguf_block0[first_mismatch.unwrap_or(0)..first_mismatch.unwrap_or(0)+8]);
    }

    // 对比 block1 (offset 176)
    if q_proj.size >= 352 {
        let blob_block1 = &blob[l0_q_proj.1 + 176..l0_q_proj.1 + 352];
        let gguf_block1 = &q_bytes[176..352];
        let m1 = blob_block1.iter().zip(gguf_block1.iter()).filter(|(a,b)| a != b).count();
        eprintln!("[block1] {} 处不匹配", m1);
    }

    eprintln!("\n=== 结论 ===");
    eprintln!("若 blob == GGUF (0 不匹配) → pack 正确, bug 在别处 (运行时地址/层循环)");
    eprintln!("若 blob != GGUF → pack 有 BUG (第17方向命中, 权重字节被错误转换)");
}
