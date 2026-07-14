//! ★第一性原理根因验证★ 字节级对比 layer0 vs layer3 v_proj 权重
//! 假设: layer3 v_proj 是 Q5K(720896B), 但被 raw copy 到 Q6K slot(860160B)
//! 期待: layer3 v_proj 前 720896B 是 Q5K 字节, 后 139264B 是未填充(0或旧数据)
#![cfg(test)]
use gllm::{Client, ModelKind};

#[test]
#[ignore]
fn verify_layer3_vproj_bytes() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "4");
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    let blob = c.diagnostic_weight_blob_bytes().expect("blob");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::File::create("/tmp/wblob_verify.txt").expect("f");
    let _ = writeln!(f, "blob len={} scratchpad len={}", blob.len(), sp.data.len());
    // layer_blob_base = 234593280, v_proj rel_off = 2166784, stride = 11379712
    let base = 234593280usize;
    let v_rel = 2166784usize;
    let stride = 11379712usize;
    let v_size_q6k = 860160usize;  // layer0 (Q6K)
    let v_size_q5k = 720896usize;  // layer3 (Q5K)
    // layer0 v_proj (Q6K, 860160B)
    let l0_off = base + v_rel;
    // layer3 v_proj (Q5K, 720896B, but packed to Q6K slot 860160B)
    let l3_off = base + v_rel + 3 * stride;
    let _ = writeln!(f, "layer0 v_proj off={} (Q6K, {}B)", l0_off, v_size_q6k);
    let _ = writeln!(f, "layer3 v_proj off={} (should be Q5K {}B, packed to Q6K slot {}B)", l3_off, v_size_q5k, v_size_q6k);
    // dump layer0 v_proj first 32 bytes (Q6K block header: d f16, dmin f16, ...)
    let _ = writeln!(f, "\nlayer0 v_proj first32: {:02x?}", &blob[l0_off..l0_off+32]);
    // dump layer3 v_proj first 32 bytes (Q5K block header: d f16, dmin f16, scales[12], qh[32], qs[128])
    let _ = writeln!(f, "layer3 v_proj first32: {:02x?}", &blob[l3_off..l3_off+32]);
    // 关键: layer3 v_proj 在 720896..860160 (offset l3_off+720896 .. l3_off+860160) 应该是未填充
    let gap_start = l3_off + v_size_q5k;
    let gap_end = l3_off + v_size_q6k;
    let _ = writeln!(f, "\nlayer3 gap [{}..{}): first32: {:02x?}", gap_start, gap_end, &blob[gap_start..gap_start+32]);
    // 检查 gap 是否全0 (未初始化) 或有旧数据
    let gap_nonzero = blob[gap_start..gap_end].iter().filter(|&&b| b != 0).count();
    let _ = writeln!(f, "layer3 gap nonzero bytes: {}/{}", gap_nonzero, gap_end-gap_start);
    // 对比: 如果 layer3 真是 Q5K, 它的 block 结构不同于 Q6K
    // Q6K block[0:2]=d, [2:4]=dmin, [4:132]=ql[128], [132:196]=qh[64], [196:210]=sc[16]... wait 210=2+2+128+64+? 
    // Q5K block[0:2]=d, [2:4]=dmin, [4:16]=scales[12], [16:48]=qh[32], [48:176]=qs[128]
    // layer0 (Q6K) byte[4] should be ql[0]. layer3 (Q5K) byte[4] should be scales[0].
    let _ = f.flush();
    eprintln!("done /tmp/wblob_verify.txt");
}
