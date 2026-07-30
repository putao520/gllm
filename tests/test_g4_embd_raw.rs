//! 读 Gemma4 token_embd block0 原始字节 + 标准 Q4_K dequant 对照
#![cfg(test)]
use gllm::loader::gguf::GgufReader;
#[test]
#[ignore]
fn g4_embd_raw_dequant() {
    let path = "/tmp/gemma4_e2b/gemma-4-E2B-it-Q8_0.gguf";
    let r = GgufReader::open(path).expect("open gguf");
    let t = r.tensors().iter().find(|t| t.name.as_ref() == "token_embd.weight")
        .expect("find token_embd.weight");
    eprintln!("token_embd: dtype={:?} shape={:?} offset={}", t.dtype, t.shape, t.offset);
    let f = std::fs::File::open(path).unwrap();
    let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
    let off = t.offset as usize;
    let block0 = &m[off..off+144];
    let d = half::f16::from_le_bytes([block0[0], block0[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block0[2], block0[3]]).to_f32();
    eprintln!("block0: d={:.6} dmin={:.6}", d, dmin);
    eprintln!("scales[0..12] = {:02x?}", &block0[4..16]);
    eprintln!("qs[0..16] = {:02x?}", &block0[16..32]);
    let scales = &block0[4..16];
    for j in 0..8usize {
        let (sc, mn) = if j < 4 {
            ((scales[j] as u32 & 0x3F) as f32, (scales[j+4] as u32 & 0x3F) as f32)
        } else {
            let s = ((scales[j] as u32 & 0xF) | ((scales[j-4] as u32 >> 6) << 4)) as f32;
            let mm = ((scales[j+4] as u32 >> 4) | ((scales[j] as u32 >> 6) << 4)) as f32;
            (s, mm)
        };
        let d_sub = d * sc;
        let m_sub = dmin * mn;
        let qptr = &block0[16 + j*16..16 + j*16 + 16];
        let v0 = (qptr[0] & 0x0F) as f32;
        let v1 = (qptr[0] >> 4) as f32;
        let val0 = d_sub * v0 + m_sub;
        let val1 = d_sub * v1 + m_sub;
        eprintln!("  sub[{}] d_sub={:.6} m_sub={:.6} v0={} v1={} → {:.4} {:.4}", j, d_sub, m_sub, v0, v1, val0, val1);
    }
    eprintln!("DONE");
}
