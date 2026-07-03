//! BCE-NaN-LOCATE: AVX-512 x86 codegen NaN 源定位 (CPU diagnostic, runs on 5070Ti AMD).
//!
//! 5070Ti 的 AMD 9950X3D 支持 AVX-512 → DeviceProfile 选 IsaLevel::Avx512 →
//! AVX-512 codegen 产生 NaN. Intel 10900KF (无 AVX-512) → Avx2 → 有限但错误 logits.
//! 所以 NaN 源是 AVX-512 x86 codegen 路径, 不是 GPU PTX (diagnostic_prefill_logits
//! 调用 CPU entry_fn, GPU PTX 编译但未 launch).
//!
//! 本测试扫描所有 intermediate tensor 的 scratchpad 区域, 找首个 NaN 产生的 op.

#![allow(dead_code)]
use gllm::{BackendType, Client, ModelKind};
use std::io::Write as _;

fn stats(v: &[f32]) -> (usize, usize, f32, f32) {
    let nan = v.iter().filter(|x| x.is_nan()).count();
    let inf = v.iter().filter(|x| x.is_infinite()).count();
    let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_v = v.iter().cloned().fold(f32::INFINITY, f32::min);
    (nan, inf, min_v, max_v)
}

#[test]
fn avx512_nan_locate() {
    eprintln!("[OOMPROBE-TEST] avx512_nan_locate entered");
    std::io::stderr().flush().ok();
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    const PROMPT: &str = "The meaning of life is";

    let client = Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cpu)
        .build()
        .expect("cpu client build");
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("[DIAG] tokens={:?}", tokens);

    // Run prefill, get full scratchpad + intermediate tensor (name,offset,dtype) map.
    let sp = client
        .diagnostic_prefill_scratchpad(&tokens)
        .expect("cpu prefill scratchpad");

    // Scan every intermediate tensor for NaN/inf, find FIRST one with NaN/inf.
    // Order by offset to follow forward execution order.
    let mut named: Vec<(&String, usize, gllm_kernels::types::DType)> = sp
        .named_offsets
        .iter()
        .map(|(n, o, d)| (n, *o, *d))
        .collect();
    named.sort_by_key(|(_, o, _)| *o);

    eprintln!("[DIAG] scanning {} intermediate tensors (sorted by offset)", named.len());
    let mut first_bad: Option<(&str, usize)> = None;
    for (name, off, dt) in &named {
        // Read a sample of elements at this offset. We don't know the exact elem count,
        // so read up to next tensor offset or a cap (e.g. 64 elems).
        let elem_bytes = dt.size_bytes().max(1);
        let cap_bytes = 64 * elem_bytes;
        let end = (*off + cap_bytes).min(sp.data.len());
        let avail = end.saturating_sub(*off);
        let n_elems = avail / elem_bytes;
        if n_elems == 0 {
            continue;
        }
        let slice = decode_sample(&sp.data, *off, n_elems, *dt);
        let (nan, inf, min_v, max_v) = stats(&slice);
        let flag = if nan > 0 { "NaN" } else if inf > 0 { "INF" } else { "ok" };
        eprintln!(
            "[DIAG-TENSOR] {name:<30} off={off:>8} dt={dt:?} n={n_elems:>4} nan={nan:>3} inf={inf:>3} min={min_v:.3} max={max_v:.3} [{flag}]"
        );
        if (nan > 0 || inf > 0) && first_bad.is_none() {
            first_bad = Some((name.as_str(), *off));
        }
    }
    std::io::stderr().flush().ok();

    // Also check embedding (offset 0) and logits.
    let emb = decode_sample(&sp.data, 0, 8.min(sp.data.len() / 4), gllm_kernels::types::DType::F32);
    eprintln!("[DIAG-EMBED] first8={:?}", &emb);
    let (nan, inf, _, _) = stats(&emb);
    eprintln!("[DIAG-EMBED] nan={nan} inf={inf}");

    match first_bad {
        Some((name, off)) => {
            eprintln!("[DIAG-FIRST-NAN] tensor='{name}' offset={off}");
            eprintln!("[DIAG-CONCLUSION] NaN 源 = AVX-512 codegen 在 op producing '{name}' 处产生 NaN");
        }
        None => {
            eprintln!("[DIAG-FIRST-NAN] none — all intermediates finite (unexpected for 5070Ti AVX-512)");
        }
    }
    std::io::stderr().flush().ok();
    // No hard assert (diagnostic probe).
    assert!(named.len() > 0, "no intermediate tensors recorded");
}

fn decode_sample(data: &[u8], off: usize, n: usize, dt: gllm_kernels::types::DType) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let eb = dt.size_bytes().max(1);
    for i in 0..n {
        let o = off + i * eb;
        if o + eb > data.len() {
            break;
        }
        let v = match dt {
            gllm_kernels::types::DType::F32 => {
                f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]])
            }
            gllm_kernels::types::DType::BF16 => {
                half::bf16::from_bits(u16::from_le_bytes([data[o], data[o + 1]])).to_f32()
            }
            gllm_kernels::types::DType::F16 => {
                half::f16::from_bits(u16::from_le_bytes([data[o], data[o + 1]])).to_f32()
            }
            _ => 0.0,
        };
        out.push(v);
    }
    out
}
