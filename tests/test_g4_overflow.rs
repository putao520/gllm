//! Gemma4 Q4_0 per-layer overflow diagnostic (BCE-20260730-Q4K-MIN-SIGN post-fix).
//! After the min-sign fix, embed should be finite but logits still =30 (overflow).
//! This test dumps per-layer hidden RMS to find the first overflow layer.
#![cfg(test)]

use gllm::Client;

fn rms(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 { return 0.0; }
    let sum: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    (sum / n as f64).sqrt() as f32
}

#[test]
#[ignore]
fn g4_overflow_diag() {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "2");

    let client = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q4_0.gguf")
        .expect("load Gemma4 Q4_0");
    let tokens = client.encode("The").expect("encode");
    eprintln!("tokens={tokens:?}");

    if let Some(sp) = client.diagnostic_prefill_scratchpad(&tokens) {
        eprintln!("SP len={}", sp.data.len());
        // Print named offsets
        for (n, o, d) in &sp.named_offsets {
            eprintln!("  {} off={} dtype={:?}", n, o, d);
        }
        // Read key tensors
        for name in ["embedding", "layer_sliding_small.normed", "layer_sliding_small.q",
                     "layer_sliding_small.attn", "layer_sliding_small.attn_resid",
                     "layer_sliding_small.post_normed",
                     "layer_sliding_small.ffn_act", "layer_sliding_small.post_ffn_sandwich",
                     "layer_full_small.normed", "layer_full_small.q"] {
            if let Some((_, o, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == name) {
                let n_elems = 256.min((sp.data.len().saturating_sub(*o)) / 4);
                if n_elems > 0 {
                    let x = sp.read_f32_at(*o, n_elems);
                    let nan = x.iter().filter(|v| v.is_nan()).count();
                    let inf = x.iter().filter(|v| v.is_infinite()).count();
                    let max_abs = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    eprintln!("VAL {} off={} n={} rms={} nan={} inf={} max_abs={} first={:?}",
                        name, o, x.len(), rms(&x), nan, inf, max_abs, &x[..x.len().min(8)]);
                }
            }
        }
        // Logits
        let logits_off = sp.logits_offset;
        if logits_off > 0 && logits_off + 1024 <= sp.data.len() {
            let lg = sp.read_f32_at(logits_off, 256);
            let nan = lg.iter().filter(|v| v.is_nan()).count();
            let max_abs = lg.iter().map(|v| v.abs()).fold(0f32, f32::max);
            eprintln!("LOGITS off={} nan={} max_abs={} first={:?}", logits_off, nan, max_abs, &lg[..8]);
        }
    } else {
        eprintln!("SCRATCHPAD ERR");
    }

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
