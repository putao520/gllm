//! Gemma4 Q4_0 per-layer overflow diagnostic (BCE-20260730-Q4K-MIN-SIGN post-fix).
//! After the min-sign fix, embed should be finite but logits still =30 (overflow).
//! This test dumps per-layer hidden RMS to find the first overflow layer.
//! NOTE: embed/logits share offset 0 (alias), so we read L0.* tensors instead.
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

        // Read key tensors by their real named_offsets names
        let names = [
            "L0.input_norm", "L0.q_proj", "L0.k_proj", "L0.v_proj",
            "L0.q_norm", "L0.k_norm", "L0.o_proj",
            "L0.post_attention_norm", "L0.post_attn_norm",
            "L0.gate_proj", "L0.up_proj", "L0.down_proj",
            "L0.post_ffw_norm",
            "L4.input_norm", "L4.q_proj", "L4.o_proj", "L4.post_ffw_norm",
            "L15.input_norm", "L15.q_proj", "L15.post_ffw_norm",
            "final_norm", "lm_head",
        ];
        for name in &names {
            if let Some((_, o, _)) = sp.named_offsets.iter().find(|(n, _, _)| n == *name) {
                let avail = sp.data.len().saturating_sub(*o);
                let n_elems = 256.min(avail / 4);
                if n_elems > 0 {
                    let x = sp.read_f32_at(*o, n_elems);
                    let nan = x.iter().filter(|v| v.is_nan()).count();
                    let inf = x.iter().filter(|v| v.is_infinite()).count();
                    let max_abs = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    eprintln!("VAL {} off={} n={} rms={:.6} nan={} inf={} max_abs={:.6} first={:?}",
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
            eprintln!("LOGITS off={} nan={} max_abs={:.6} first={:?}", logits_off, nan, max_abs, &lg[..8]);
        }
    } else {
        eprintln!("SCRATCHPAD ERR");
    }

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}
