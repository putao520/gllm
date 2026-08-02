//! CPU NaN-layer probe (BCE-20260703-NaN-LOCATE).
//!
//! Validates `DiagnosticScratchpad::find_nan_tensors` on the CPU path: ensures
//! `named_offsets` is populated, the scan walks every intermediate tensor without
//! panicking, and reports the first NaN tensor (if any). On a healthy CPU path
//! there should be zero NaN tensors — this test mainly proves the probe machinery
//! works so the GPU probe (tests/test_e2e_gpu.rs::gpu_diag_smollm2_nan_layer)
//! can be trusted to localize the GPU NaN.

use gllm::{BackendType, Client, ModelKind};
use std::io::Write as _;

#[test]
fn cpu_smollm2_nan_layer_probe() {
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
    const PROMPT: &str = "The meaning of life is";

    let client = Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .backend(BackendType::Cpu)
        .build()
        .expect("cpu client build");
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("[DIAG-CPU-LAYER] tokens={:?}", tokens);

    let sp = client
        .diagnostic_prefill_scratchpad(&tokens)
        .expect("cpu prefill scratchpad");

    eprintln!(
        "[DIAG-CPU-LAYER] scratchpad bytes={}, named_offsets={}, logits_off={}, vocab={}, hidden={}, dtype={:?}",
        sp.data.len(), sp.named_offsets.len(), sp.logits_offset, sp.vocab_size, sp.hidden_size, sp.compute_dtype
    );
    assert!(sp.named_offsets.len() > 0, "named_offsets empty — compile did not surface intermediate tensors");

    let mut sorted: Vec<&(String, usize, gllm_kernels::types::DType)> = sp.named_offsets.iter().collect();
    sorted.sort_by_key(|e| e.1);
    eprintln!("[DIAG-CPU-LAYER] === named_offsets (first 30, offset order) ===");
    for (name, off, dt) in sorted.iter().take(30) {
        eprintln!("[DIAG-CPU-LAYER]   off={:>8} dt={:?} name={}", off, dt, name);
    }
    eprintln!("[DIAG-CPU-LAYER] ... (total {} named tensors)", sorted.len());

    let hits = sp.find_nan_tensors();
    eprintln!("[DIAG-CPU-LAYER] NaN tensor hits: {}", hits.len());
    for h in &hits {
        eprintln!(
            "[DIAG-CPU-LAYER]   NaN name={} off={} dt={:?} nan={}/{} unsupported={}",
            h.name, h.offset, h.dtype, h.nan_count, h.sample_count, h.unsupported_dtype
        );
    }

    let emb = sp.embedding();
    let emb_nan = emb.iter().filter(|x| x.is_nan()).count();
    for (i, v) in emb.iter().enumerate() {
        if v.is_nan() || v.is_infinite() {
            let token = i / sp.hidden_size;
            let dim = i % sp.hidden_size;
            eprintln!(
                "[DIAG-NAN-POS] emb idx={} token={} dim={} val={:?}",
                i, token, dim, v
            );
        }
    }
    eprintln!("[DIAG-NAN-POS] emb first8: {:?}", &emb[..8]);
    eprintln!(
        "[DIAG-NAN-POS] emb last8: {:?}",
        &emb[emb.len() - 8..]
    );
    let logits = sp.last_token_logits();
    let logits_nan = logits.iter().filter(|x| x.is_nan()).count();
    let logits_inf = logits.iter().filter(|x| x.is_infinite()).count();
    eprintln!(
        "[DIAG-CPU-LAYER] embedding len={} nan={}; logits len={} nan={} inf={}",
        emb.len(), emb_nan, logits.len(), logits_nan, logits_inf
    );

    // [DIAG-NAN-POS] final_norm: the REAL NaN source (computed RMSNorm output, 1 token).
    // final_norm is at named_offsets entry "final_norm" (BF16, 576 elems = 1 token hidden).
    if let Some((_, fn_off, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == "final_norm") {
        let fn_vals = sp.read_dtype_aware(*fn_off, sp.hidden_size);
        let fn_nan = fn_vals.iter().filter(|x| x.is_nan()).count();
        let fn_inf = fn_vals.iter().filter(|x| x.is_infinite()).count();
        eprintln!("[DIAG-NAN-POS] final_norm off={} len={} nan={} inf={}", fn_off, fn_vals.len(), fn_nan, fn_inf);
        for (i, v) in fn_vals.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                eprintln!("[DIAG-NAN-POS] final_norm idx={} dim={} val={:?}", i, i % sp.hidden_size, v);
            }
        }
        eprintln!("[DIAG-NAN-POS] final_norm first8: {:?}", &fn_vals[..8]);
    }
    // [DIAG-NAN-POS] final_normed (F32 post-norm, the input to lm_head)
    if let Some((_, fn_off, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == "final_normed") {
        let fn_vals = sp.read_dtype_aware(*fn_off, sp.hidden_size);
        let fn_nan = fn_vals.iter().filter(|x| x.is_nan()).count();
        let fn_inf = fn_vals.iter().filter(|x| x.is_infinite()).count();
        eprintln!("[DIAG-NAN-POS] final_normed off={} len={} nan={} inf={}", fn_off, fn_vals.len(), fn_nan, fn_inf);
        eprintln!("[DIAG-NAN-POS] final_normed first8: {:?}", &fn_vals[..8]);
    }
    // [DIAG-LAYER0] layer.normed (L0 RMSNorm output = first computed tensor after embedding gather)
    // Compare this F32 between local(AVX2) and 5070Ti(AVX-512) to isolate embedding-gather vs RMSNorm divergence.
    if let Some((_, ln_off, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == "layer.normed") {
        let n = sp.hidden_size;
        let vals: Vec<f32> = (0..n).map(|i| {
            let b = ln_off + i * 4;
            if b + 4 <= sp.data.len() {
                f32::from_le_bytes([sp.data[b], sp.data[b+1], sp.data[b+2], sp.data[b+3]])
            } else { 0.0 }
        }).collect();
        let nan = vals.iter().filter(|x| x.is_nan()).count();
        let maxv = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let minv = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[DIAG-LAYER0] layer.normed(F32) off={} nan={} min={:?} max={:?}", ln_off, nan, minv, maxv);
        eprintln!("[DIAG-LAYER0] layer.normed(F32) first16: {:?}", &vals[..16.min(vals.len())]);
        // also dump embedding (off=0) as F32 — first 16, this is post-gather pre-norm input
        let emb_f32: Vec<f32> = (0..16).map(|i| {
            let b = i * 4;
            if b + 4 <= sp.data.len() {
                f32::from_le_bytes([sp.data[b], sp.data[b+1], sp.data[b+2], sp.data[b+3]])
            } else { 0.0 }
        }).collect();
        eprintln!("[DIAG-LAYER0] embedding(F32) first16: {:?}", &emb_f32);
    }
    // [DIAG-F32-REAL] Force F32 read (bypass compute_dtype=BF16 metadata bug).
    // WidenCompute stores F32 in scratchpad; reading as BF16 gives F32-high16-as-BF16 garbage.
    // This block reads the raw F32 bytes to see the ACTUAL computed values.
    eprintln!("[DIAG-F32-REAL] === forced F32 reads (actual scratchpad storage) ===");
    {
        let off = 0usize;
        let n = sp.prompt_len * sp.hidden_size;
        let emb_f32: Vec<f32> = (0..n).map(|i| {
            let b = i * 4;
            if b + 4 <= sp.data.len() {
                f32::from_le_bytes([sp.data[off+b], sp.data[off+b+1], sp.data[off+b+2], sp.data[off+b+3]])
            } else { 0.0 }
        }).collect();
        let emb_nan = emb_f32.iter().filter(|x| x.is_nan()).count();
        let emb_inf = emb_f32.iter().filter(|x| x.is_infinite()).count();
        eprintln!("[DIAG-F32-REAL] embedding(F32) len={} nan={} inf={}", emb_f32.len(), emb_nan, emb_inf);
        eprintln!("[DIAG-F32-REAL] embedding(F32) first8: {:?}", &emb_f32[..8]);
    }
    {
        // final_normed as F32
        if let Some((_, fn_off, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == "final_normed") {
            let n = sp.hidden_size;
            let vals: Vec<f32> = (0..n).map(|i| {
                let b = fn_off + i * 4;
                if b + 4 <= sp.data.len() {
                    f32::from_le_bytes([sp.data[b], sp.data[b+1], sp.data[b+2], sp.data[b+3]])
                } else { 0.0 }
            }).collect();
            let nan = vals.iter().filter(|x| x.is_nan()).count();
            let inf = vals.iter().filter(|x| x.is_infinite()).count();
            let maxv = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let minv = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            eprintln!("[DIAG-F32-REAL] final_normed(F32) off={} len={} nan={} inf={} min={:?} max={:?}", fn_off, vals.len(), nan, inf, minv, maxv);
            eprintln!("[DIAG-F32-REAL] final_normed(F32) first8: {:?}", &vals[..8]);
        }
    }
    {
        // logits as F32
        let lo = sp.logits_offset;
        let n = sp.vocab_size;
        let vals: Vec<f32> = (0..n).map(|i| {
            let b = lo + i * 4;
            if b + 4 <= sp.data.len() {
                f32::from_le_bytes([sp.data[b], sp.data[b+1], sp.data[b+2], sp.data[b+3]])
            } else { 0.0 }
        }).collect();
        let nan = vals.iter().filter(|x| x.is_nan()).count();
        let inf = vals.iter().filter(|x| x.is_infinite()).count();
        let nonzero = vals.iter().filter(|x| **x != 0.0).count();
        let maxv = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let minv = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("[DIAG-F32-REAL] logits(F32) off={} len={} nan={} inf={} nonzero={} min={:?} max={:?}", lo, vals.len(), nan, inf, nonzero, minv, maxv);
        eprintln!("[DIAG-F32-REAL] logits(F32) first8: {:?}", &vals[..8]);
        // argmax
        if nonzero > 0 {
            let amax = vals.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0);
            eprintln!("[DIAG-F32-REAL] logits(F32) argmax={} val={:?}", amax, vals[amax]);
        }
    }
    // [DIAG-NAN-POS] logits Inf positions (cosine_sim=NaN suggests Inf/overflow)
    let log_inf_pos: Vec<usize> = logits.iter().enumerate().filter(|(_, v)| v.is_infinite() || v.is_nan()).map(|(i, _)| i).take(10).collect();
    eprintln!("[DIAG-NAN-POS] logits first8: {:?}", &logits[..8]);
    if !log_inf_pos.is_empty() {
        eprintln!("[DIAG-NAN-POS] logits nan/inf first10 indices: {:?}", log_inf_pos);
        eprintln!("[DIAG-NAN-POS] logits max={:?} min={:?}", logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max), logits.iter().cloned().fold(f32::INFINITY, f32::min));
    }

    std::io::stderr().flush().ok();
    // On a healthy CPU path, no NaN. If there is NaN, this probe localizes it.
    assert_eq!(emb_nan, 0, "embedding has NaN — upstream of all layers broken");
    assert_eq!(logits_nan, 0, "CPU logits NaN — CPU path also broken (not GPU-specific)");
}
