//! BCE-20260716-BUG-A: ActivationSwap runtime ptr trace.
//! Dumps per-swap ping/pong physical pointer values (from telemetry swap-log)
//! for Q5_K_M mixed-quant N=2 vs Q6_K standard N=2, to locate which swap
//! produces a wrong pointer (non 0 / non 167772160) → pong zeroing.
//!
//! JIT instrumentation (gllm-kernels commit 9cc62a4d): when GLLM_TRACE_SWAP=1 at
//! compile time, each ActivationSwap writes 32B to telemetry_ptr+1024+idx*32:
//!   [0..8]   ping-before (reg_a pre-xchg)
//!   [8..16]  pong-before (reg_b pre-xchg)
//!   [16..24] ping-after  (reg_a post-xchg)
//!   [24..32] pong-after  (reg_b post-xchg)
//! idx is a compile-time counter (one per compiled ActivationSwap instruction);
//! at runtime the layer loop overwrites the same idx slot → last-iteration value.
#![cfg(test)]
use gllm::{Client, ModelKind};

const PING_ADDR: u64 = 0;          // ping buffer offset
const PONG_ADDR: u64 = 167772160;  // pong buffer offset (max_seq*hidden*4)
const SWAP_LOG_BASE: usize = 1024;
const SWAP_RECORD_BYTES: usize = 32;

fn read_u64(buf: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    u64::from_le_bytes(b)
}

fn dump_swaps(model_filter: &str, n: usize, label: &str, f: &mut std::fs::File) {
    use std::io::Write;
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(model_filter).build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    let _ = writeln!(f, "=== {} N={} (telemetry len={}) ===", label, n, sp.telemetry.len());
    // Walk swap-log records until we hit an all-zero record (no more swaps written).
    let mut idx = 0;
    while SWAP_LOG_BASE + (idx + 1) * SWAP_RECORD_BYTES <= sp.telemetry.len() {
        let base = SWAP_LOG_BASE + idx * SWAP_RECORD_BYTES;
        let ping_b = read_u64(&sp.telemetry, base);
        let pong_b = read_u64(&sp.telemetry, base + 8);
        let ping_a = read_u64(&sp.telemetry, base + 16);
        let pong_a = read_u64(&sp.telemetry, base + 24);
        // Stop at first completely-empty record (sentinel 0, no swap wrote here).
        if ping_b == 0 && pong_b == 0 && ping_a == 0 && pong_a == 0 {
            break;
        }
        let flag = |v: u64| -> &'static str {
            match v {
                PING_ADDR => "PING(0)",
                PONG_ADDR => "PONG(167M)",
                _ => "!!!WRONG",
            }
        };
        let _ = writeln!(f, "  swap[{}]: before ping={:#x}{} pong={:#x}{} | after ping={:#x}{} pong={:#x}{}",
            idx, ping_b, flag(ping_b), pong_b, flag(pong_b),
            ping_a, flag(ping_a), pong_a, flag(pong_a));
        idx += 1;
        if idx > 64 { break; } // safety cap
    }
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_swap_trace() {
    let mut f = std::fs::File::create("/tmp/swap_trace.txt").expect("f");
    // Q5_K_M mixed-quant N=2 (BUG-A: NaN)
    dump_swaps("q5_k_m", 2, "Q5_K_M", &mut f);
    // Q6_K standard N=2 (works)
    dump_swaps("q6_k", 2, "Q6_K", &mut f);
}
