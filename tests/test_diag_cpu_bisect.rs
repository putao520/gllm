//! CPU BUG 逐层 bisection 诊断 (ARCH-UNIFIED-EXEC 块1 后续).
//!
//! 块1 实测 (commit 2364ff48): CPU argmax=967 (golden=253), cosine=-0.465.
//! 本测试按架构师 sessionId 5d98f4f4 诊断路径定位根因:
//!   Step 0: 行选排除 (5 行 logits 逐行 cosine)
//!   Step 1: 嵌入对比 (gllm encode_to_layer layer0 vs golden hidden_layer_0)
//!   Step 2: 逐层 bisection (encode_to_layer layer N vs golden hidden_layer_N)
//!
//! 不带 cuda feature 跑: cargo +nightly test --test test_diag_cpu_bisect -- --nocapture

#![cfg(target_os = "linux")]

use gllm::{Client, ModelKind};
use gllm::head_routing::{LayerAnchor, PoolMode};
use std::io::Write as _;

// ─── Golden helpers (mirror test_e2e_cpu.rs) ─────────────────────────

fn load_golden_f32_tensor(path: &std::path::Path, name: &str) -> Vec<f32> {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("read golden {e}"));
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap_or_else(|e| panic!("parse {e}"));
    let view = tensors.tensor(name).unwrap_or_else(|_| panic!("missing tensor {name}"));
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_golden_next_token_id(path: &std::path::Path) -> u32 {
    let v = load_golden_f32_tensor(path, "next_token_id");
    v[0] as u32
}

/// golden logits [seq_len, vocab_size] → 返回完整 (不切片), 供逐行 cosine.
fn load_golden_all_logits(path: &std::path::Path) -> Vec<f32> {
    load_golden_f32_tensor(path, "logits")
}

/// golden hidden_layer_N [seq_len=5, hidden=576].
fn load_golden_hidden_layer(path: &std::path::Path, layer_idx: usize) -> Vec<f32> {
    load_golden_f32_tensor(path, &format!("hidden_layer_{layer_idx}"))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)) as f32 }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0usize, |mi, (i, &x)| if x > v[mi] { i } else { mi })
}

const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";
const PROMPT: &str = "The meaning of life is";
const SEQ_LEN: usize = 5;
const VOCAB_SIZE: usize = 49152;
const HIDDEN_SIZE: usize = 576;
const NUM_LAYERS: usize = 30;

fn golden_path() -> std::path::PathBuf {
    std::path::PathBuf::from("tests/e2e_alignment/data/golden_smollm2_135m.safetensors")
}

fn build_cpu_client() -> Client {
    Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .build()
        .unwrap_or_else(|e| panic!("build CPU client: {e}"))
}

// ─── Step 0: 行选排除 ────────────────────────────────────────────────
//
// diagnostic_prefill_logits 读 row 0 (ARCH-DECODE-LOGITS-ROW0). golden logits
// shape (5, 49152) 是 5 行. 抽 CPU 每一行 vs golden 每一行 cosine, 看是否有 >0.99.

#[test]
fn diag_step0_row_selection() {
    eprintln!("=== Step 0: 行选排除 ===");
    std::io::stderr().flush().ok();

    let path = golden_path();
    assert!(path.exists(), "golden missing");
    let golden_all = load_golden_all_logits(&path);
    let golden_next = load_golden_next_token_id(&path);
    eprintln!("golden next_token_id = {golden_next}");

    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("tokens = {:?}", tokens);

    // diagnostic_prefill_logits 返回 (seq_len, vocab) 完整? 还是只 row 0?
    // 块1 验证 len=49152 (=1*vocab), 只 row 0. 需 scratchpad 抽多行.
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");
    eprintln!("scratchpad: data len={}, logits_offset={}, vocab={}, prompt_len={}, dtype={:?}, elem_bytes={}",
        sp.data.len(), sp.logits_offset, sp.vocab_size, sp.prompt_len, sp.compute_dtype, sp.elem_bytes());

    // logits region: [seq_len, vocab] in scratchpad starting at logits_offset
    let elem = sp.elem_bytes();
    let vocab = sp.vocab_size;
    let row_bytes = vocab * elem;
    eprintln!("logits region: row_bytes={}, seq_len={}", row_bytes, sp.prompt_len);

    // CPU logits: 5 rows (seq_len=5), each vocab=49152
    let cpu_rows: Vec<Vec<f32>> = (0..sp.prompt_len).map(|r| {
        let off = sp.logits_offset + r * row_bytes;
        // dtype-aware: 用 DiagnosticScratchpad 的 read 方法或直接按 compute_dtype 解
        // 简化: 假设 F32 (elem=4). 若非 F32, read_dtype_aware 更准.
        let mut row = Vec::with_capacity(vocab);
        for i in 0..vocab {
            let b = &sp.data[off + i * elem..off + (i + 1) * elem];
            let v = match elem {
                4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                2 => {
                    // BF16: 高16位, 拼成 F32
                    let bits = (b[1] as u32) << 8 | b[0] as u32;
                    let f32_bits = bits << 16;
                    f32::from_bits(f32_bits)
                }
                _ => 0.0,
            };
            row.push(v);
        }
        row
    }).collect();

    eprintln!("CPU logits rows extracted: {}", cpu_rows.len());

    // golden 也是 5 行 [seq_len, vocab]
    let golden_rows: Vec<Vec<f32>> = (0..SEQ_LEN).map(|r| {
        let off = r * VOCAB_SIZE;
        golden_all[off..off + VOCAB_SIZE].to_vec()
    }).collect();

    // 逐行 cosine 矩阵
    eprintln!("\n--- cosine(cpu_row, golden_row) 矩阵 ---");
    for cr in 0..cpu_rows.len().min(SEQ_LEN) {
        for gr in 0..SEQ_LEN {
            let cos = cosine(&cpu_rows[cr], &golden_rows[gr]);
            eprintln!("  cpu[{cr}] vs golden[{gr}] = {cos:.4}");
        }
    }

    // 诊断: 是否有任一 >0.99
    let mut max_cos = -1.0f32;
    let mut max_pair = (0usize, 0usize);
    for cr in 0..cpu_rows.len().min(SEQ_LEN) {
        for gr in 0..SEQ_LEN {
            let c = cosine(&cpu_rows[cr], &golden_rows[gr]);
            if c > max_cos { max_cos = c; max_pair = (cr, gr); }
        }
    }
    eprintln!("\n>>> max cosine = {max_cos:.4} at cpu[{0}] vs golden[{1}]", max_pair.0, max_pair.1);
    if max_cos > 0.99 {
        eprintln!(">>> 行选可能 OK 或行错位: cpu row {} ≈ golden row {}", max_pair.0, max_pair.1);
    } else {
        eprintln!(">>> 行选无匹配 (>0.99), 排除纯行选 bug — 问题在计算路径");
    }
}

// ─── Step 1 + 2: 逐层 bisection (encode_to_layer) ────────────────────
//
// encode_to_layer 返回 pooled hidden. 用 ClsToken pool (取 row 0) 后 cosine vs
// golden hidden_layer_N row 0. 简化: LastToken pool vs golden row (seq_len-1).
// 注意: encode_to_layer 内部 pool, 我们对比 pooled 后的值.
// 更准: 用 MeanPool vs golden mean. 但 first diverge layer 仍能定位.

#[test]
fn diag_step1_embedding_compare() {
    eprintln!("\n=== Step 1: 嵌入对比 (layer 0 hidden) ===");
    std::io::stderr().flush().ok();

    let path = golden_path();
    let golden_h0 = load_golden_hidden_layer(&path, 0); // (5, 576)
    eprintln!("golden hidden_layer_0: len={}, expected {}", golden_h0.len(), SEQ_LEN * HIDDEN_SIZE);

    let client = build_cpu_client();
    let _tokens = client.encode(PROMPT).expect("encode");

    // encode_to_layer layer 0, MeanPool → (576,)
    let anchor = LayerAnchor::Absolute(0);
    let gllm_h0_pooled = client
        .encode_to_layer(PROMPT, anchor, PoolMode::MeanPool)
        .expect("encode_to_layer layer0");
    eprintln!("gllm layer0 pooled: len={}", gllm_h0_pooled.len());

    // golden h0 mean over seq → (576,)
    let golden_h0_mean: Vec<f32> = (0..HIDDEN_SIZE).map(|h| {
        let mut s = 0.0f32;
        for r in 0..SEQ_LEN { s += golden_h0[r * HIDDEN_SIZE + h]; }
        s / SEQ_LEN as f32
    }).collect();
    eprintln!("golden layer0 mean: len={}", golden_h0_mean.len());

    let cos = cosine(&gllm_h0_pooled, &golden_h0_mean);
    eprintln!("cosine(gllm layer0 MeanPool, golden h0 mean) = {cos:.4}");
    if cos > 0.99 {
        eprintln!(">>> 嵌入对齐 (layer0 OK) — BF16 解码 + gather + 权重名 OK, 问题在层内 (norm/GEMM)");
    } else {
        eprintln!(">>> 嵌入不对齐 — BF16 解码/权重名/gather 可能有 bug");
    }
}

#[test]
fn diag_step2_layer_bisect() {
    eprintln!("\n=== Step 2: 逐层 bisection ===");
    std::io::stderr().flush().ok();

    let path = golden_path();
    let client = build_cpu_client();
    let _tokens = client.encode(PROMPT).expect("encode");

    let mut first_diverge: Option<usize> = None;
    for n in 0..=NUM_LAYERS {
        let golden_hn = load_golden_hidden_layer(&path, n); // (5,576)
        let golden_mean: Vec<f32> = (0..HIDDEN_SIZE).map(|h| {
            let mut s = 0.0f32;
            for r in 0..SEQ_LEN { s += golden_hn[r * HIDDEN_SIZE + h]; }
            s / SEQ_LEN as f32
        }).collect();

        let gllm_hn = match client.encode_to_layer(PROMPT, LayerAnchor::Absolute(n), PoolMode::MeanPool) {
            Ok(v) => v,
            Err(e) => { eprintln!("layer {n}: encode_to_layer err {e}"); continue; }
        };

        let next = n + 1;
        let cos = cosine(&gllm_hn, &golden_mean);
        eprintln!("layer {n:2}: cosine = {cos:.4}");
        if cos < 0.99 && first_diverge.is_none() {
            first_diverge = Some(n);
        }
    }

    eprintln!("\n>>> 首个发散层 (cosine<0.99) = {:?}", first_diverge);
    match first_diverge {
        Some(0) => eprintln!(">>> 发散在 layer 0: embedding/norm/QKV proj 错 (查 golden_layer0_ops)"),
        Some(n) => eprintln!(">>> 发散在 layer {n}: 该层 GEMM/norm 符号错 (BF16 fallback GEMM?)"),
        None => eprintln!(">>> 所有层都 OK 但 final logits 错: lm_head/final norm 问题"),
    }
}

// ─── 架构师 sessionId 5d98f4f4 第6轮纠错: diag_step1/2 比对错位一层 ───
// encode_to_layer(N) 返回 layer N **输出**, 应比 golden hidden_layer_{N+1}
// (不是 hidden_layer_N). 之前全错位, cosine≈0 可能是正常错位噪声.

#[test]
fn diag_step3_buffer_layout_dump() {
    eprintln!("\n=== Step 3: buffer_layout 段基址 (确认 offset 37748736 落哪个段) ===");
    std::io::stderr().flush().ok();
    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");

    eprintln!("scratchpad data len = {} bytes ({:.1} MB)", sp.data.len(), sp.data.len() as f64 / 1_048_576.0);
    eprintln!("logits_offset = {} ({:.1} MB)", sp.logits_offset, sp.logits_offset as f64 / 1_048_576.0);
    eprintln!("vocab={}, prompt_len={}, hidden={}, dtype={:?}, elem_bytes={}",
        sp.vocab_size, sp.prompt_len, sp.hidden_size, sp.compute_dtype, sp.elem_bytes());
    eprintln!("\nnamed_offsets ({} 个):", sp.named_offsets.len());
    for (name, off, dt) in &sp.named_offsets {
        eprintln!("  {name:30} off={off:>12} ({:.1} MB) dtype={:?}", off / 1_048_576, dt);
    }

    // 查 embedding offset
    if let Some(off) = client.diagnostic_tensor_offset("embedding") {
        eprintln!("\n>>> embedding offset (diagnostic_tensor_offset) = {} ({:.1} MB)", off, off as f64 / 1_048_576.0);
    }
    if let Some(off) = client.diagnostic_tensor_offset("embed") {
        eprintln!(">>> embed (weight) offset = {} ({:.1} MB)", off, off as f64 / 1_048_576.0);
    }
}

#[test]
fn diag_step4_embedding_direct_read() {
    eprintln!("\n=== Step 4: 直读 embedding (绕开 layer 0) vs golden hidden_layer_0 ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let golden_h0 = load_golden_hidden_layer(&path, 0); // (5, 576) = embedding 输出
    eprintln!("golden hidden_layer_0 (embedding): len={}, expected {}", golden_h0.len(), SEQ_LEN * HIDDEN_SIZE);

    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");

    // 用 diagnostic_tensor_offset 查 embedding (动态 offset, 不硬编码)
    let emb_off = client.diagnostic_tensor_offset("embedding")
        .expect("embedding tensor offset not found in named_offsets");
    let emb_dtype = sp.compute_dtype; // 全图统一 compute dtype (BF16 权重 + F32 激活, embedding 输出 = 激活 dtype)
    eprintln!("embedding offset = {} dtype = {:?}", emb_off, emb_dtype);
    eprintln!("scratchpad data len = {}, emb_off + seq*hidden*elem_bytes = {}",
        sp.data.len(), emb_off + SEQ_LEN * HIDDEN_SIZE * emb_dtype.size_bytes());

    // 从 scratchpad 的 emb_off 读 [5, 576] embedding (按 dtype)
    let elem = emb_dtype.size_bytes();
    let mut gllm_embed = vec![0.0f32; SEQ_LEN * HIDDEN_SIZE];
    for i in 0..SEQ_LEN * HIDDEN_SIZE {
        let b = &sp.data[emb_off + i * elem..emb_off + (i + 1) * elem];
        gllm_embed[i] = match elem {
            4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            2 => {
                // BF16: 高16位拼 F32
                let bits = (b[1] as u32) << 8 | b[0] as u32;
                f32::from_bits(bits << 16)
            }
            _ => 0.0,
        };
    }
    eprintln!("gllm embedding read: len={}, first 5 = {:?}", gllm_embed.len(), &gllm_embed[0..5]);
    eprintln!("golden h0 first 5 = {:?}", &golden_h0[0..5]);

    // row 0 cosine
    let cos_row0 = cosine(&gllm_embed[0..HIDDEN_SIZE], &golden_h0[0..HIDDEN_SIZE]);
    // mean over seq cosine
    let gllm_mean: Vec<f32> = (0..HIDDEN_SIZE).map(|h| {
        let mut s = 0.0f32;
        for r in 0..SEQ_LEN { s += gllm_embed[r * HIDDEN_SIZE + h]; }
        s / SEQ_LEN as f32
    }).collect();
    let golden_mean: Vec<f32> = (0..HIDDEN_SIZE).map(|h| {
        let mut s = 0.0f32;
        for r in 0..SEQ_LEN { s += golden_h0[r * HIDDEN_SIZE + h]; }
        s / SEQ_LEN as f32
    }).collect();
    let cos_mean = cosine(&gllm_mean, &golden_mean);

    eprintln!("cosine(gllm embedding row0, golden h0 row0) = {cos_row0:.4}");
    eprintln!("cosine(gllm embedding mean, golden h0 mean) = {cos_mean:.4}");
    if cos_mean > 0.99 {
        eprintln!(">>> embedding 对齐! 之前 embedding 正交 = diag_step1/2 错位比对伪信号. 根因在 layer 之后");
    } else {
        eprintln!(">>> embedding 真的不对齐 (语义已确认). 根因在 embedding/gather 数据路径");
    }
}

#[test]
fn diag_step5_layer_bisect_fixed() {
    eprintln!("\n=== Step 5: 逐层 bisection (修正错位: encode_to_layer(N) vs golden hidden_layer_(N+1)) ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let client = build_cpu_client();
    let _tokens = client.encode(PROMPT).expect("encode");

    // SmolLM2 30 层, golden hidden_layer_0..30 (31 个). encode_to_layer(N) 返回 layer N 输出
    // = golden hidden_layer_{N+1}. N=0..29 对应 golden hidden_layer_1..30.
    let mut first_diverge: Option<usize> = None;
    for n in 0..NUM_LAYERS {
        let golden_hn_out = load_golden_hidden_layer(&path, n + 1); // layer N 输出 = golden hidden_layer_{N+1}
        let golden_mean: Vec<f32> = (0..HIDDEN_SIZE).map(|h| {
            let mut s = 0.0f32;
            for r in 0..SEQ_LEN { s += golden_hn_out[r * HIDDEN_SIZE + h]; }
            s / SEQ_LEN as f32
        }).collect();

        let gllm_hn = match client.encode_to_layer(PROMPT, LayerAnchor::Absolute(n), PoolMode::MeanPool) {
            Ok(v) => v,
            Err(e) => { eprintln!("layer {n}: encode_to_layer err {e}"); continue; }
        };
        let next = n + 1;
        let cos = cosine(&gllm_hn, &golden_mean);
        eprintln!("layer {n:2} output: cosine(vs golden hidden_layer_{}) = {cos:.4}", n + 1);
        if cos < 0.99 && first_diverge.is_none() {
            first_diverge = Some(n);
        }
    }
    eprintln!("\n>>> 首个发散层 (修正错位后) = {:?}", first_diverge);
    match first_diverge {
        Some(n) => eprintln!(">>> 发散在 layer {n} 输出: 该层处理错 (attention/FFN/norm)"),
        None => eprintln!(">>> 所有层都对, 根因在 final norm / lm_head"),
    }
}

// ─── 架构师第8轮: per-row cosine + 交叉比对 (token 序列已确认一致) ───
// gllm tokens [504,2455,282,1029,314] == golden input_ids (逐位相同)
// 排除 token 序列错位. 进 per-row 分析 embedding 部分对齐根因.

#[test]
fn diag_step6_per_row_and_cross() {
    eprintln!("\n=== Step 6: per-row cosine + 交叉比对 ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let golden_h0 = load_golden_hidden_layer(&path, 0); // (5, 576)
    eprintln!("确认: gllm tokens = golden input_ids = [504, 2455, 282, 1029, 314] (逐位相同)");

    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("gllm tokens = {:?}", tokens);
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");
    let emb_off = client.diagnostic_tensor_offset("embedding").expect("embedding offset");
    let elem = sp.elem_bytes();

    // 读 gllm embedding [5, 576]
    let mut gllm_rows: Vec<Vec<f32>> = Vec::with_capacity(SEQ_LEN);
    for r in 0..SEQ_LEN {
        let row_off = emb_off + r * HIDDEN_SIZE * elem;
        let row: Vec<f32> = (0..HIDDEN_SIZE).map(|i| {
            let b = &sp.data[row_off + i * elem..row_off + (i + 1) * elem];
            match elem {
                4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                2 => { let bits = (b[1] as u32) << 8 | b[0] as u32; f32::from_bits(bits << 16) }
                _ => 0.0,
            }
        }).collect();
        gllm_rows.push(row);
    }
    let golden_rows: Vec<Vec<f32>> = (0..SEQ_LEN).map(|r| {
        golden_h0[r * HIDDEN_SIZE..(r + 1) * HIDDEN_SIZE].to_vec()
    }).collect();

    eprintln!("\n--- per-row cosine (gllm[i] vs golden[i]) ---");
    for i in 0..SEQ_LEN {
        let c = cosine(&gllm_rows[i], &golden_rows[i]);
        eprintln!("  row {i}: cosine = {c:.4}");
    }

    eprintln!("\n--- 交叉比对 (gllm[i] vs golden[i+1]) — 测 token 错位+1 ---");
    for i in 0..SEQ_LEN.saturating_sub(1) {
        let c = cosine(&gllm_rows[i], &golden_rows[i + 1]);
        eprintln!("  gllm[{i}] vs golden[{}] = {c:.4}", i + 1);
    }

    eprintln!("\n--- 交叉比对 (gllm[i+1] vs golden[i]) — 测 token 错位-1 ---");
    for i in 0..SEQ_LEN.saturating_sub(1) {
        let c = cosine(&gllm_rows[i + 1], &golden_rows[i]);
        eprintln!("  gllm[{}] vs golden[{i}] = {c:.4}", i + 1);
    }

    // 模式判定
    let diag: Vec<f32> = (0..SEQ_LEN).map(|i| cosine(&gllm_rows[i], &golden_rows[i])).collect();
    let cross_plus: Vec<f32> = (0..SEQ_LEN-1).map(|i| cosine(&gllm_rows[i], &golden_rows[i+1])).collect();
    let cross_minus: Vec<f32> = (0..SEQ_LEN-1).map(|i| cosine(&gllm_rows[i+1], &golden_rows[i])).collect();
    let diag_max = diag.iter().cloned().fold(-1.0f32, f32::max);
    let cross_plus_max = cross_plus.iter().cloned().fold(-1.0f32, f32::max);
    let cross_minus_max = cross_minus.iter().cloned().fold(-1.0f32, f32::max);
    eprintln!("\n>>> max: diag={diag_max:.4} cross+1={cross_plus_max:.4} cross-1={cross_minus_max:.4}");
    if cross_plus_max > 0.9 {
        eprintln!(">>> gllm token 序列比 golden 错位 +1 (gllm[i]≈golden[i+1])");
    } else if cross_minus_max > 0.9 {
        eprintln!(">>> gllm token 序列比 golden 错位 -1 (gllm[i+1]≈golden[i])");
    } else if diag_max > 0.9 {
        eprintln!(">>> per-row 对齐 (某行高), 非错位");
    } else {
        eprintln!(">>> 所有模式都低, 非简单错位 — gather 每行读取有系统偏移 (代码 bug)");
    }
}

// ─── 架构师第9轮 + topology 资料库: 单 token prefill 重建 embedding (方法A) ───
// GenerateLoop M=1 覆盖 row0, 单次调用只 row0 有数据.
// 方法A: 每位置 i 用前缀 tokens[0..i+1] 调 scratchpad, 读 row0 = token i embedding,
// 拼接 [seq, hidden] vs golden hidden_layer_0. layer0 ops 不依赖 KV cache可重建.

#[test]
fn diag_step7_single_token_prefill_rebuild() {
    eprintln!("\n=== Step 7: 单 token prefill 重建 embedding (方法A, topology 资料库) ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let golden_h0 = load_golden_hidden_layer(&path, 0); // (5, 576) golden embedding
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");
    eprintln!("all_tokens = {:?}", all_tokens);

    let emb_off_resolver = client.diagnostic_tensor_offset("embedding")
        .expect("embedding offset");

    // 对每个位置 i, 用前缀 tokens[0..i+1] prefill, 读 row0 = token i embedding
    let mut rebuilt: Vec<f32> = Vec::with_capacity(SEQ_LEN * HIDDEN_SIZE);
    let mut per_row_cos = Vec::with_capacity(SEQ_LEN);
    for i in 0..SEQ_LEN {
        let prefix: Vec<u32> = all_tokens[0..=i].to_vec();
        let sp = client.diagnostic_prefill_scratchpad(&prefix).expect("scratchpad");
        let elem = sp.elem_bytes();
        // row0 = 最后 token (token i) 的 embedding, 在 emb_off 处 [576] (M=1 只 row0)
        let mut row = vec![0.0f32; HIDDEN_SIZE];
        for h in 0..HIDDEN_SIZE {
            let b = &sp.data[emb_off_resolver + h * elem..emb_off_resolver + (h + 1) * elem];
            row[h] = match elem {
                4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                2 => { let bits = (b[1] as u32) << 8 | b[0] as u32; f32::from_bits(bits << 16) }
                _ => 0.0,
            };
        }
        let golden_row = &golden_h0[i * HIDDEN_SIZE..(i + 1) * HIDDEN_SIZE];
        let cos = cosine(&row, golden_row);
        eprintln!("token {i} (id={}): row0 cosine vs golden h0 row{i} = {cos:.4}", all_tokens[i]);
        per_row_cos.push(cos);
        rebuilt.extend_from_slice(&row);
    }

    // 整体 cosine (拼接 [5, 576])
    let cos_full = cosine(&rebuilt, &golden_h0);
    eprintln!("\n>>> 重建 [seq, hidden] cosine vs golden h0 = {cos_full:.4}");
    eprintln!(">>> per-row min/max = {:.4} / {:.4}", per_row_cos.iter().cloned().fold(1.0f32, f32::min), per_row_cos.iter().cloned().fold(-1.0f32, f32::max));
    if cos_full > 0.99 {
        eprintln!(">>> embedding 完全正确 (单 token 重建证明)! 之前 'embedding bug' 是诊断 harness 读多行错位");
    } else {
        eprintln!(">>> embedding 真错 (语义对齐后仍不对), 根因在 embedding/gather 数据路径");
    }
}

#[test]
fn diag_step8b_single_token_capture() {
    eprintln!("\n=== Step 8b: 单 token prefill capture (隔离 GenerateLoop 覆盖) ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");
    // 单 token: 只用 token 0
    let single = vec![all_tokens[0]];
    let cap_off = client.diagnostic_tensor_offset("layer_capture").expect("cap off");
    let cap_stride = client.diagnostic_layer_capture_stride();
    assert!(cap_stride > 0);
    let sp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    let elem = 4usize;
    let mut l0 = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[cap_off + h*elem..cap_off + (h+1)*elem];
        l0[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    eprintln!("single-token capture layer0 first 5 = {:?}", &l0[0..5]);
    // 单 token prefill: golden hidden_layer_1 row0 (token 0 的 layer 0 输出)
    let golden_h1 = load_golden_hidden_layer(&path, 1);
    let golden_row0 = &golden_h1[0..HIDDEN_SIZE];
    eprintln!("golden h1 row0 first 5 = {:?}", &golden_row0[0..5]);
    eprintln!("cosine(capture layer0, golden h1 row0) = {:.4}", cosine(&l0, golden_row0));
    // 也试 golden h0 row0 (embedding token 0)
    let golden_h0 = load_golden_hidden_layer(&path, 0);
    let golden_emb_row0 = &golden_h0[0..HIDDEN_SIZE];
    eprintln!("cosine(capture layer0, golden h0 row0) = {:.4}", cosine(&l0, golden_emb_row0));

    // 关键: 单 token prefill 的 embedding 是否对? (验证输入正确性)
    let emb_off = client.diagnostic_tensor_offset("embedding").expect("emb off");
    let mut emb = vec![0.0f32; HIDDEN_SIZE];
    let esp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    let eelem = esp.elem_bytes();
    for h in 0..HIDDEN_SIZE {
        let b = &esp.data[emb_off + h*eelem..emb_off + (h+1)*eelem];
        emb[h] = match eelem {
            4 => f32::from_le_bytes([b[0],b[1],b[2],b[3]]),
            2 => { let bits = (b[1] as u32)<<8 | b[0] as u32; f32::from_bits(bits<<16) }
            _ => 0.0,
        };
    }
    eprintln!("single-token embedding row0 vs golden h0 row0 = {:.4}", cosine(&emb, golden_emb_row0));

    // 逐层 capture (单 token) vs golden hidden_layer_{N+1} row0
    eprintln!("\n=== 单 token 逐层 capture vs golden ===");
    for n in 0..NUM_LAYERS.min(5) {
        let mut ln = vec![0.0f32; HIDDEN_SIZE];
        for h in 0..HIDDEN_SIZE {
            let o = cap_off + n * cap_stride + h*elem;
            if o + elem <= sp.data.len() {
                let b = &sp.data[o..o+elem];
                ln[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
            }
        }
        let golden_h = load_golden_hidden_layer(&path, n + 1);
        let golden_row = &golden_h[0..HIDDEN_SIZE]; // row0 = token 0
        let cos = cosine(&ln, golden_row);
        let norm: f64 = ln.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        eprintln!("  layer{n} vs golden h{} row0: cos={:.4} norm={:.3}", n+1, cos, norm);
    }
}

#[test]
fn diag_step8c_capture_inter_layer() {
    eprintln!("\n=== Step 8c: capture 层间相似度 (检测残流/恒等问题) ===");
    std::io::stderr().flush().ok();
    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    let cap_off = client.diagnostic_tensor_offset("layer_capture").expect("cap off");
    let cap_stride = client.diagnostic_layer_capture_stride();
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("sp");
    let elem = 4usize;
    let read_layer = |n: usize| -> Vec<f32> {
        let o = cap_off + n * cap_stride;
        (0..HIDDEN_SIZE).map(|h| {
            let b = &sp.data[o + h*elem..o + (h+1)*elem];
            f32::from_le_bytes([b[0],b[1],b[2],b[3]])
        }).collect()
    };
    let l0 = read_layer(0);
    // layer0 vs embedding (golden h0 row4)
    let golden_h0 = load_golden_hidden_layer(&golden_path(), 0);
    let emb = &golden_h0[(SEQ_LEN-1)*HIDDEN_SIZE..SEQ_LEN*HIDDEN_SIZE];
    eprintln!("capture layer0 vs golden embedding row4 = {:.4}", cosine(&l0, emb));
    for n in 1..5 {
        let ln = read_layer(n);
        let lprev = read_layer(n-1);
        eprintln!("capture layer{n} vs layer{} = {:.4}", n-1, cosine(&ln, &lprev));
    }
    // 也看 layer0 的 norm (是否 RMSNorm 后的值)
    eprintln!("capture layer0 norm = {:.4}", l0.iter().map(|x| x*x).sum::<f32>().sqrt());
    eprintln!("golden embedding row4 norm = {:.4}", emb.iter().map(|x| x*x).sum::<f32>().sqrt());
    // dump gllm capture layer0 (5-token, last token=row4 semantics) to file for Python comparison
    {
        let mut buf = Vec::with_capacity(HIDDEN_SIZE * 4);
        for v in &l0 { buf.extend_from_slice(&v.to_le_bytes()); }
        let _ = std::fs::write("/tmp/gllm_capture_layer0_5token.bin", &buf);
        eprintln!("[DUMP] gllm capture layer0 → /tmp/gllm_capture_layer0_5token.bin");
    }
}

//
// 前置: gllm-kernels diagnostic-layer-capture feature 启用.
//   cargo test --features gllm-kernels/diagnostic-layer-capture --test test_diag_cpu_bisect diag_step8 -- --nocapture
//
// capture 区在 scratchpad, offset = diagnostic_tensor_offset("layer_capture"),
// per-layer stride = diagnostic_layer_capture_stride(). 第 N 层输出 (M=1 row0)
// 在 offset + N * stride, hidden=576 F32.
//
// 注意 GenerateLoop M=1: 单次 prefill 只处理最后 token, capture 每层是最后 token
// 的 hidden (seq_len=1 row0). golden hidden_layer_N shape (5, 576) — 比最后行 (row4).

#[test]
fn diag_step8_layer_capture_bisect() {
    eprintln!("\n=== Step 8: Ring-Buffer 逐层捕获 — 30 层 bisection ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let client = build_cpu_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("tokens = {:?}", tokens);

    let cap_off = client.diagnostic_tensor_offset("layer_capture")
        .expect("layer_capture offset not found (feature not enabled?)");
    let cap_stride = client.diagnostic_layer_capture_stride();
    assert!(cap_stride > 0, "layer_capture_stride=0 (feature not enabled?)");
    eprintln!("cap_off={} stride={}", cap_off, cap_stride);

    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");
    eprintln!("scratchpad len={} bytes", sp.data.len());
    let elem = 4usize; // capture 区 F32 (side_channel_copy dtype=F32)

    // 每层 N: 读 hidden_layer_N 最后行 (row seq_len-1) vs golden hidden_layer_{N+1} row4
    // 注意: golden hidden_layer_0 = embedding 输出; hidden_layer_N (N>=1) = layer N-1 输出.
    // gllm capture 第 N 层 (counter N) = layer N 的输出 = golden hidden_layer_{N+1}.
    eprintln!("\nlayer | cosine vs golden hidden_layer_(N+1) row4 | nonzero");
    let mut first_bad: Option<usize> = None;
    for n in 0..NUM_LAYERS {
        let layer_off = cap_off + n * cap_stride;
        if layer_off + HIDDEN_SIZE * elem > sp.data.len() {
            eprintln!("layer {n}: OUT OF BOUNDS (off={}, len={})", layer_off, sp.data.len());
            break;
        }
        let mut row = vec![0.0f32; HIDDEN_SIZE];
        for h in 0..HIDDEN_SIZE {
            let b = &sp.data[layer_off + h * elem..layer_off + (h + 1) * elem];
            row[h] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        let nonzero = row.iter().filter(|x| x.abs() > 1e-12).count();
        let golden_idx = n + 1;
        let golden_h = load_golden_hidden_layer(&path, golden_idx);
        let golden_last = &golden_h[(SEQ_LEN - 1) * HIDDEN_SIZE..SEQ_LEN * HIDDEN_SIZE];
        let cos = cosine(&row, golden_last);
        eprintln!("  {n:2}  | cos={cos:.4}  | nonzero={nonzero}/{}", HIDDEN_SIZE);
        if first_bad.is_none() && cos < 0.99 {
            first_bad = Some(n);
        }
    }
    match first_bad {
        Some(n) => eprintln!("\n>>> 首个发散层: layer {} (capture layer {} vs golden hidden_layer_{})", n, n, n + 1),
        None => eprintln!("\n>>> 全 30 层 cosine > 0.99, 层路径无发散 — 根因在层后 (final_norm/lm_head)"),
    }
}

#[test]
fn diag_step9_encode_at_layer_row0() {
    eprintln!("\n=== Step 9: encode_at_layer(0) row0 vs capture layer0 vs golden h1 ===");
    std::io::stderr().flush().ok();
    let path = golden_path();
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");

    let single = vec![all_tokens[0]];
    let out = client.encode_to_layer(PROMPT, LayerAnchor::Absolute(0), PoolMode::LastToken)
        .expect("encode_to_layer");
    eprintln!("encode_to_layer(Layer0, LastToken) output len={} (expected hidden={})", out.len(), HIDDEN_SIZE);
    let row0 = &out[0..HIDDEN_SIZE];
    eprintln!("encode_to_layer row0 first 5 = {:?}", &row0[0..5]);

    let cap_off = client.diagnostic_tensor_offset("layer_capture").expect("cap off");
    let cap_stride = client.diagnostic_layer_capture_stride();
    let sp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    let elem = 4usize;
    let mut cap_l0 = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[cap_off + h*elem..cap_off + (h+1)*elem];
        cap_l0[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    eprintln!("capture layer0 first 5 = {:?}", &cap_l0[0..5]);
    eprintln!("cosine(encode_at_layer(0) row0, capture layer0) = {:.4}", cosine(row0, &cap_l0));

    let golden_h1 = load_golden_hidden_layer(&path, 1);
    let golden_row0 = &golden_h1[0..HIDDEN_SIZE];
    eprintln!("cosine(encode_at_layer(0) row0, golden h1 row0) = {:.4}", cosine(row0, golden_row0));
    eprintln!("cosine(capture layer0, golden h1 row0) = {:.4}", cosine(&cap_l0, golden_row0));
}

#[test]
fn diag_step10_weight_byte_verify() {
    eprintln!("\n=== Step 10: 权重字节验证 (路C) — layer0 input_norm weight ===");
    std::io::stderr().flush().ok();
    let client = build_cpu_client();
    let blob = client.diagnostic_weight_blob_bytes().expect("weight blob");
    eprintln!("gllm weight_blob len={} bytes ({:.1} MB)", blob.len(), blob.len() as f64 / 1_048_576.0);

    let golden_path = std::path::PathBuf::from(
        "/home/putao/.gllm/models/huggingface/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/model.safetensors"
    );
    let data = std::fs::read(&golden_path).expect("read golden model");
    use std::convert::TryInto;
    let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&data[8..8+n]).expect("header");
    let info = header["model.layers.0.input_layernorm.weight"].as_object().expect("tensor");
    let off = info["data_offsets"][0].as_u64().unwrap() as usize + 8 + n;
    let golden_bf16: Vec<u8> = data[off..off + 576 * 2].to_vec();
    let golden_f32: Vec<f32> = (0..576).map(|i| {
        let b = &golden_bf16[i*2..i*2+2];
        let bits = (b[1] as u32) << 8 | b[0] as u32;
        f32::from_bits(bits << 16)
    }).collect();
    eprintln!("golden input_norm first 5 = {:?}", &golden_f32[0..5]);

    let pattern = &golden_bf16;
    let mut found_offsets = Vec::new();
    for i in 0..blob.len().saturating_sub(pattern.len()) {
        if &blob[i..i+pattern.len()] == pattern {
            found_offsets.push(i);
        }
    }
    eprintln!("golden input_norm pattern 在 gllm blob 找到 {} 处: {:?}", found_offsets.len(), found_offsets);
    if found_offsets.is_empty() {
        eprintln!(">>> 未找到! 权重字节不一致 (loader 转换/转置/偏移错)");
    } else {
        eprintln!(">>> 找到! 权重字节一致 (input_norm weight 正确)");
    }
    // dump full weight blob for Python analysis
    let _ = std::fs::write("/tmp/gllm_weight_blob.bin", &blob);
    eprintln!("[DUMP] weight_blob → /tmp/gllm_weight_blob.bin ({} bytes)", blob.len());
}

#[test]
fn diag_step12_single_layer_intermediates() {
    eprintln!("\n=== Step 12: GLLM_SINGLE_LAYER=1 读 layer0 所有中间张量 ===");
    std::io::stderr().flush().ok();
    std::env::set_var("GLLM_SINGLE_LAYER", "1");
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");
    let single = vec![all_tokens[0]];
    let sp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    std::env::remove_var("GLLM_SINGLE_LAYER");
    let elem = 4usize;
    let read_tensor = |name: &str| -> Vec<f32> {
        let off = client.diagnostic_tensor_offset(name).unwrap_or(usize::MAX);
        if off == usize::MAX { return Vec::new(); }
        (0..HIDDEN_SIZE).map(|h| {
            let b = &sp.data[off + h*elem..off + (h+1)*elem];
            f32::from_le_bytes([b[0],b[1],b[2],b[3]])
        }).collect()
    };
    let tensors = ["embedding", "layer.normed", "layer.q", "layer.k", "layer.v",
                   "layer.q_rope", "layer.k_rope", "layer.attn", "layer.o",
                   "layer.attn_resid", "layer.post_normed", "layer.gate", "layer.up",
                   "layer.ffn_act", "layer.down", "layer.ffn_resid"];
    for name in &tensors {
        let data = read_tensor(name);
        if data.is_empty() { eprintln!("{:20} NOT FOUND", name); continue; }
        let norm: f64 = data.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let safe_name = name.replace('.', "_");
        let path = format!("/tmp/gllm_{}.bin", safe_name);
        let mut buf = Vec::with_capacity(HIDDEN_SIZE * 4);
        for v in &data { buf.extend_from_slice(&v.to_le_bytes()); }
        let _ = std::fs::write(&path, &buf);
        eprintln!("{:20} norm={:8.3} first5={:?} → {}", name, norm, &data[0..5], path);
    }
}

#[test]
fn diag_step13_two_layer_intermediates() {
    eprintln!("\n=== Step 13: GLLM_DEBUG_LAYERS=2 读 layer1 中间张量 ===");
    std::io::stderr().flush().ok();
    std::env::set_var("GLLM_DEBUG_LAYERS", "2");
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");
    let single = vec![all_tokens[0]];
    let sp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    std::env::remove_var("GLLM_DEBUG_LAYERS");
    let elem = 4usize;
    let v_off = client.diagnostic_tensor_offset("layer.v").unwrap();
    let mut v = vec![0.0f32; 192];
    for h in 0..192 {
        let b = &sp.data[v_off + h*elem..v_off + (h+1)*elem];
        v[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    let v_norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    eprintln!("layer.v (last layer=layer1) norm={:.3} first5={:?}", v_norm, &v[0..5]);
    let mut buf = Vec::new();
    for x in &v { buf.extend_from_slice(&x.to_le_bytes()); }
    let _ = std::fs::write("/tmp/gllm_l1_v.bin", &buf);
    let ffn_off = client.diagnostic_tensor_offset("layer.ffn_resid").unwrap();
    let mut ffn = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[ffn_off + h*elem..ffn_off + (h+1)*elem];
        ffn[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    let ffn_norm: f64 = ffn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    eprintln!("layer.ffn_resid (layer1 out) norm={:.3} first5={:?}", ffn_norm, &ffn[0..5]);
}

#[test]
fn diag_step14_ping_pong_buffers() {
    eprintln!("\n=== Step 14: dump ping/pong buffer 内容验证 ActivationSwap ===");
    std::io::stderr().flush().ok();
    std::env::set_var("GLLM_DEBUG_LAYERS", "2");
    let client = build_cpu_client();
    let all_tokens = client.encode(PROMPT).expect("encode");
    let single = vec![all_tokens[0]];
    let sp = client.diagnostic_prefill_scratchpad(&single).expect("sp");
    std::env::remove_var("GLLM_DEBUG_LAYERS");
    let elem = 4usize;
    // 读 named_offsets 找 layer.v (唯一 offset, layer1 最后写入)
    // ping/pong sentinel 不在 named_offsets, 需从 scratch_base 推断
    // 但 layer.normed / layer.q / layer.o 等共享 slot — 这些是 activation buffer
    // 读 layer.v (layer1 v_proj 输出, 唯一 offset 100663296)
    let v_off = client.diagnostic_tensor_offset("layer.v").unwrap();
    // dump layer1 attention 输入 (layer.normed slot = layer1 rmsnorm1 输出, 但被覆盖)
    // 改读 layer.attn (81788928, layer1 attn out, 唯一)
    let attn_off = client.diagnostic_tensor_offset("layer.attn").unwrap();
    let mut attn = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[attn_off + h*elem..attn_off + (h+1)*elem];
        attn[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    eprintln!("layer.attn (layer1) first5={:?} norm={:.3}", &attn[0..5],
        attn.iter().map(|x| x*x).sum::<f32>().sqrt());
    // layer1 v_proj 输入 = layer1 rmsnorm1 输出. 如果 = rmsnorm(embedding), layer1 读 embedding
    // 读 layer.ffn_resid (9437184, layer1 最终输出)
    let ffn_off = client.diagnostic_tensor_offset("layer.ffn_resid").unwrap();
    let mut ffn = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[ffn_off + h*elem..ffn_off + (h+1)*elem];
        ffn[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    eprintln!("layer.ffn_resid (layer1 out) first5={:?} norm={:.3}", &ffn[0..5],
        ffn.iter().map(|x| x*x).sum::<f32>().sqrt());
    // 关键: 读 layer0 输出. layer0 输出在 pong (ActivationSwap 前) 或 ping (后)
    // layer.ffn_resid 在单层时是 layer0 输出, 2层时是 layer1 输出(覆盖)
    // 用 capture 读 layer0 (cap_off + 0)
    let cap_off = client.diagnostic_tensor_offset("layer_capture").unwrap();
    let cap_stride = client.diagnostic_layer_capture_stride();
    let mut l0_cap = vec![0.0f32; HIDDEN_SIZE];
    for h in 0..HIDDEN_SIZE {
        let b = &sp.data[cap_off + h*elem..cap_off + (h+1)*elem];
        l0_cap[h] = f32::from_le_bytes([b[0],b[1],b[2],b[3]]);
    }
    eprintln!("capture layer0 (单token) first5={:?} norm={:.3}", &l0_cap[0..5],
        l0_cap.iter().map(|x| x*x).sum::<f32>().sqrt());
}
