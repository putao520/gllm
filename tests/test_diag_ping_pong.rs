//! 精准验证: N=1 vs N=2 的 ping(offset 0) / pong(offset 167772160) 内容.
//! 理论: N=1 后 pong(167772160)=layer0_out(非零). N=2 后 pong(167772160)=layer0_out(仍非零, layer1 写 ping).
//! 若 N=2 pong 全零 → layer0_out 被清零 (layer1 bug).
#![cfg(test)]
use gllm::{Client, ModelKind};

const PONG_OFF: usize = 167772160;

fn dump_act(label: &str, filter: &str, n: usize) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter(filter).build().expect("client");
    let t = c.encode("The capital of France is").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    // 读 ping (offset 0) 和 pong (offset 167772160) 前 8 个 f32
    let ping = sp.read_dtype_aware(0, 8);
    let pong = sp.read_dtype_aware(PONG_OFF, 8);
    let ping_max = ping.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    let pong_max = pong.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    let pong_nz = ping.iter().filter(|v| v.abs() > 1e-10).count();
    use std::fs::OpenOptions;
    use std::io::Write;
    let mut f = OpenOptions::new().create(true).append(true).open("/tmp/q5km_pingpong.txt").expect("f");
    let _ = writeln!(f, "[{} N={}] ping|max|={:.4} {:?} | pong|max|={:.4} {:?}",
        label, n, ping_max, &ping[..4.min(ping.len())], pong_max, &pong[..4.min(pong.len())]);
    let _ = f.flush();
    eprintln!("[{} N={}] pong|max|={:.4}", label, n, pong_max);
}

#[test]
#[ignore]
fn diag_ping_pong_n1_n2() {
    let _ = std::fs::write("/tmp/q5km_pingpong.txt", "");  // clear
    eprintln!("\n=== ping/pong 内容 N=1 vs N=2 (Q5_K_M vs Q6_K) ===");
    dump_act("Q5_K_M", "q5_k_m", 1);
    dump_act("Q5_K_M", "q5_k_m", 2);
    dump_act("Q6_K", "q6_k", 1);
    dump_act("Q6_K", "q6_k", 2);
    eprintln!("\n理论 N=1: pong=layer0_out(非零)");
    eprintln!("理论 N=2: pong=layer0_out(仍非零, layer1写ping非pong)");
    eprintln!("若 Q5_K_M N=2 pong=零 → layer0_out 被清零 (bug)");
}
