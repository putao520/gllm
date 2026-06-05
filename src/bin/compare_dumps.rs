//! 对比 PyTorch 参考与 gllm JIT 的逐层 hidden state dump。
//!
//! 格式: 4B u32 seq_len + 4B u32 feature_dim + seq*feat*4B f32 raw
//!
//! 用法:
//!   cargo run --bin compare_dumps -- \
//!     --ref /tmp/pytorch_ref --jit /tmp/gllm_dump
//!
//! 输出每层 max_abs_diff, mean_abs_diff, cosine similarity。对齐靠 name 映射。

use std::collections::HashMap;
use std::fs;
use std::io::Read;

fn read_dump(path: &str) -> std::io::Result<(u32, u32, Vec<f32>)> {
    let mut f = fs::File::open(path)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let seq_len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    let feat = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
    let n = (seq_len as usize) * (feat as usize);
    let mut data = vec![0f32; n];
    let bytes = n * 4;
    let mut buf = vec![0u8; bytes];
    f.read_exact(&mut buf)?;
    for i in 0..n {
        data[i] = f32::from_le_bytes([buf[4*i], buf[4*i+1], buf[4*i+2], buf[4*i+3]]);
    }
    Ok((seq_len, feat, data))
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    // (max_abs_diff, mean_abs_diff, cosine)
    let n = a.len().min(b.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..n {
        let d = (a[i] - b[i]).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        dot += (a[i] as f64) * (b[i] as f64);
        norm_a += (a[i] as f64) * (a[i] as f64);
        norm_b += (b[i] as f64) * (b[i] as f64);
    }
    let cos = dot / (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    (max_abs, (sum_abs / n as f64) as f32, cos as f32)
}

fn scan_dir(dir: &str) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let fname = e.file_name().to_string_lossy().to_string();
            if fname.ends_with(".bin") {
                names.push(fname);
            }
        }
    }
    names.sort();
    names
}

/// 把 JIT 的 node name (带 idx 前缀) 映射到参考 dump 的逻辑 name
fn jit_to_ref_name(jit_fname: &str) -> Option<String> {
    let without_idx = jit_fname.splitn(2, '_').nth(1)?;
    let stem = without_idx.strip_suffix(".bin")?;

    // layer_{i}_XXX.bin → layer_{i:02d}_XXX.bin (匹配 PyTorch dump 格式)
    if stem.starts_with("layer_") {
        let rest = stem.strip_prefix("layer_")?;
        let mut parts = rest.splitn(2, '_');
        let layer_num: usize = parts.next()?.parse().ok()?;
        let tail = parts.next()?;
        // 映射 tail 到 PyTorch dump 的名称
        let mapped_tail = match tail {
            "q_proj" => "q",
            "k_proj" => "k",
            "v_proj" => "v",
            "q_biased" => "q",  // PyTorch 里 q_proj.bias 已被 dense 层应用了, 我们的 q 是 MatMul 后, q_biased 是 +bias 后; PyTorch dense 输出已含 bias
            "k_biased" => "k",
            "v_biased" => "v",
            "attn" => "attn_out",
            "o_proj" => "attn_proj",  // attention.output.dense output
            "attn_biased" => "attn_proj",  // 已含 bias
            "attn_residual" => continue_skip(),
            "attn_normed" => "attn_normed",
            "intermediate" => "inter_biased",  // PyTorch intermediate.dense 已含 bias
            "inter_biased" => "inter_biased",
            "gelu" => "act",
            "act" => "act",
            "output" => "out_biased",  // output.dense 含 bias
            "out_biased" => "out_biased",
            "ffn_residual" => continue_skip(),
            "ffn_residual_out" => continue_skip(),
            "output_norm" => "output_norm",
            _ => return None,
        };
        if mapped_tail.is_empty() { return None; }
        return Some(format!("layer_{:02}_{}.bin", layer_num, mapped_tail));
    }

    if stem == "embed_norm" { return Some("hidden_0_init.bin".to_string()); }
    if stem == "classifier_dense_matmul" { return Some("classifier_dense_biased.bin".to_string()); }
    if stem == "classifier_dense_bias" { return Some("classifier_dense_biased.bin".to_string()); }
    if stem == "classifier_tanh" { return Some("classifier_tanh_out.bin".to_string()); }
    if stem == "classifier_out_proj_matmul" { return Some("rerank_logit.bin".to_string()); }
    if stem == "classifier_out_bias" { return Some("rerank_logit.bin".to_string()); }
    None
}

fn continue_skip() -> &'static str { "" }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut ref_dir = String::new();
    let mut jit_dir = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ref" => { ref_dir = args[i+1].clone(); i += 2; }
            "--jit" => { jit_dir = args[i+1].clone(); i += 2; }
            _ => i += 1,
        }
    }
    if ref_dir.is_empty() || jit_dir.is_empty() {
        eprintln!("用法: compare_dumps --ref <dir> --jit <dir>");
        std::process::exit(1);
    }

    // 读参考 dump: 按 logical name → (seq, feat, data)
    let mut ref_map: HashMap<String, (u32, u32, Vec<f32>)> = HashMap::new();
    for name in scan_dir(&ref_dir) {
        let path = format!("{}/{}", ref_dir, name);
        match read_dump(&path) {
            Ok(x) => { ref_map.insert(name.clone(), x); }
            Err(e) => eprintln!("read_dump ref {path}: {e}"),
        }
    }

    println!("# node_idx\tjit_name\tref_name\tseq\tfeat\tmax_abs\tmean_abs\tcosine");
    // 扫描 jit dir
    for jit_name in scan_dir(&jit_dir) {
        let path = format!("{}/{}", jit_dir, jit_name);
        let (js, jf, jd) = match read_dump(&path) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let ref_name = match jit_to_ref_name(&jit_name) {
            Some(n) => n,
            None => continue,
        };
        let Some((rs, rf, rd)) = ref_map.get(&ref_name) else {
            continue;
        };
        // 只比较 live 部分 (min seq)。PyTorch 和 gllm 的 seq 应该一致; feat 应该一致。
        let seq = (js as usize).min(*rs as usize);
        let feat = (jf as usize).min(*rf as usize);
        let n = seq * feat;
        if n == 0 { continue; }
        let (max_abs, mean_abs, cos) = compare(&jd[..n], &rd[..n]);
        // JIT idx 前缀用于排序
        let idx: usize = jit_name.splitn(2, '_').next().and_then(|s| s.parse().ok()).unwrap_or(999);
        println!("{idx}\t{jit_name}\t{ref_name}\t{seq}\t{feat}\t{:.3e}\t{:.3e}\t{:.6}",
            max_abs, mean_abs, cos);
    }
}
