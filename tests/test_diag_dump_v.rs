//! 第一性原理: dump N=4 layer.v (Q6K v_proj 输出) 完整向量, 对比 N=3
//! v_proj N=4=-4694979.5(异常) N=3=0.0792(正常). 看 v 全向量是否全异常.
#![cfg(test)]
use gllm::{Client, ModelKind};

fn dump_v(n: usize, outfile: &str) {
    std::env::set_var("GLLM_TRUNCATE_LAYERS", n.to_string());
    let c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat)
        .gguf_file_filter("q5_k_m").build().expect("client");
    let t = c.encode(" ").expect("encode");
    let sp = c.diagnostic_prefill_scratchpad(&t).expect("sp");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(outfile).expect("f");
    // layer.v offset=1174405120
    let v_off = sp.named_offsets.iter().find(|(n,_,_)| n == "layer.v").map(|(_,o,_)| *o).unwrap_or(1174405120);
    let _ = writeln!(f, "[N={}] layer.v off={}", n, v_off);
    // 读前 32 个 f32
    let vals = sp.read_dtype_aware(v_off, 32);
    let _ = writeln!(f, "  first32: {:?}", &vals);
    let max = vals.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    let nan_count = vals.iter().filter(|v| v.is_nan()).count();
    let _ = writeln!(f, "  |max|={:.4} nan_count={}/32", max, nan_count);
    let _ = f.flush();
}

#[test]
#[ignore]
fn diag_dump_v_n3_n4() {
    let _ = std::fs::write("/tmp/v_dump.txt", "");
    dump_v(3, "/tmp/v_dump.txt");
    dump_v(4, "/tmp/v_dump.txt");
    dump_v(5, "/tmp/v_dump.txt");
}
