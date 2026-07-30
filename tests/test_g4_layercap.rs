//! Gemma4 per-layer hidden state capture (diagnostic-layer-capture feature).
//! Reads each layer's hidden output RMS to find the first overflow layer.
//! BCE-20260730-Q4K-MIN-SIGN post-fix: residual overflow investigation.
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
fn g4_layer_capture_diag() {
    // Use full model (no truncation) to see all 35 layers
    let client = Client::new_chat("/tmp/gemma4_e2b/gemma-4-E2B-it-Q4_0.gguf")
        .expect("load Gemma4 Q4_0");
    let tokens = client.encode("The").expect("encode");
    eprintln!("tokens={tokens:?}");

    let stride = client.diagnostic_layer_capture_stride();
    eprintln!("capture_stride={}", stride);
    if stride == 0 {
        eprintln!("ERROR: layer capture not enabled (build without diagnostic-layer-capture feature)");
        return;
    }

    if let Some(sp) = client.diagnostic_prefill_scratchpad(&tokens) {
        eprintln!("SP len={}", sp.data.len());
        // Read per-layer captured hidden states
        let cap_off = sp.named_offsets.iter()
            .find(|(n, _, _)| n == "layer_capture")
            .map(|(_, o, _)| *o)
            .unwrap_or(0);
        eprintln!("layer_capture off={}", cap_off);
        if cap_off > 0 {
            for i in 0..35 {
                let off = cap_off + i * stride;
                if off + 1024 <= sp.data.len() {
                    let x = sp.read_f32_at(off, 256);
                    let nan = x.iter().filter(|v| v.is_nan()).count();
                    let inf = x.iter().filter(|v| v.is_infinite()).count();
                    let max_abs = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    eprintln!("LAYER {:2} rms={:.6} nan={} inf={} max_abs={:.6} first={:?}",
                        i, rms(&x), nan, inf, max_abs, &x[..x.len().min(8)]);
                }
            }
        }
    } else {
        eprintln!("SCRATCHPAD ERR");
    }
}
