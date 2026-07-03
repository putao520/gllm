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
    let logits = sp.last_token_logits();
    let logits_nan = logits.iter().filter(|x| x.is_nan()).count();
    eprintln!(
        "[DIAG-CPU-LAYER] embedding len={} nan={}; logits len={} nan={}",
        emb.len(), emb_nan, logits.len(), logits_nan
    );

    std::io::stderr().flush().ok();
    // On a healthy CPU path, no NaN. If there is NaN, this probe localizes it.
    assert_eq!(emb_nan, 0, "embedding has NaN — upstream of all layers broken");
    assert_eq!(logits_nan, 0, "CPU logits NaN — CPU path also broken (not GPU-specific)");
}
