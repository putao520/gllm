//! P1: real-loop 早退逐层重建可信断点 (capture 不可信后换路径).
//!
//! 背景 (architect 五轮 consult 裁决, docs/domain-knowledge/q4_0-crosslayer-diagnostic-ladder.md):
//! diagnostic-layer-capture 工具 ≥4 处缺陷 (derive_capture_hidden fallback / per_layer_stride
//! 用 max_seq_len 上界 / emit stride 2× alloc / layer0 读位=base+0×stride=base 与 stride
//! 无关却 cos=0.0293). → 所有逐层历史数据 (layer27 cos=0.005, layer0-in-loop cos=1.0) 全
//! 不可信. 除 GLLM_SINGLE_LAYER=1 单层隔离 layer0 cos=1.0 外.
//!
//! 换路径: P1 用 real-loop 早退 (encode_to_layer 在 layer N 早退, MidLayerEncode callback
//! + ExitEarly) + 正常 decode 路径读 [seq,hidden]. 不碰 ring-buffer, 用真实循环 (真 stride
//! 步进 + 真 swap). 这是唯一已验证可信的路径 (单层隔离验证过同机制).
//!
//! 机制 (src/engine/callbacks/mid_layer_encode.rs):
//!   - MidLayerEncodeCallback::new(target_layer) 在 target layer 的 post_node 缓存最新
//!     [seq,hidden] 节点输出 (layer anchor 最终输出 = output_add 写回 hidden_0).
//!   - 切换出 target layer 时 pre_node emit ExitEarly { logits: <hidden as f32> }.
//!   - encode_at_layer_for_prompt reshape [seq,hidden] + pool.apply.
//!   - target_layers()=None (runtime callback, 不触发 per-layer JIT 重编译).
//!
//! Golden 对齐语义 (transformers output_hidden_states):
//!   - hs_0 = embedding 输出 (5, 1024)
//!   - hs_N (N>=1) = layer (N-1) 输出
//!   - encode_to_layer(N, LastToken) = layer N 输出最后 token = golden hs_{N+1} row 4
//!
//! ★第一验证点: layer0 真实循环早退 (N=0) 是否 cos≈1.0:
//!   - cos≈1.0 → 断点在 layer0→1 之后, 回原假设, 找首个发散层进 P2
//!   - cos<0.99 → "layer0 在循环里对" 一直是 capture 假象, 断点比想象早, 故事重写
//!
//! 运行 (gllm 目录, 不需 diagnostic-layer-capture feature):
//!   cargo test --test test_diag_qwen3_p1_real_loop -- --nocapture --test-threads=1
//! 若全 28 层太慢 (>10min), 设 P1_KEY_LAYERS_ONLY=1 只跑关键层定断点:
//!   P1_KEY_LAYERS_ONLY=1 cargo test --test test_diag_qwen3_p1_real_loop -- --nocapture --test-threads=1

#![cfg(target_os = "linux")]

use gllm::{Client, ModelKind};
use gllm::head_routing::{LayerAnchor, PoolMode};
use std::io::Write as _;

// ─── 常量 (Qwen3-0.6B) ────────────────────────────────────────────────
const MODEL: &str = "bartowski/Qwen_Qwen3-0.6B-GGUF";
const PROMPT: &str = "The capital of France is";
const SEQ_LEN: usize = 5;
const HIDDEN_SIZE: usize = 1024;
const NUM_LAYERS: usize = 28; // Qwen3-0.6B 28 层

/// Golden bin: 29 层 (hs_0=embed .. hs_28=layer27 out), 每个 (5, 1024) f32 = 593920 bytes.
const GOLDEN_BIN: &str = "tests/e2e_alignment/data/golden_qwen3_layers.bin";

fn golden_bin_path() -> std::path::PathBuf {
    std::path::PathBuf::from(GOLDEN_BIN)
}

/// 读 golden hs_{layer_idx} 最后 token (row 4) → (1024,) f32.
/// 偏移 = layer_idx * (5*1024*4) + 4 * (1024*4), 读 1024 个 f32.
fn load_golden_hs_last_row(layer_idx: usize) -> Vec<f32> {
    assert!(layer_idx <= 28, "layer_idx {layer_idx} out of range 0..=28");
    let data = std::fs::read(golden_bin_path())
        .unwrap_or_else(|e| panic!("read golden bin {GOLDEN_BIN}: {e}"));
    let per_layer = SEQ_LEN * HIDDEN_SIZE * 4;
    assert_eq!(data.len(), 29 * per_layer);
    let row_bytes = HIDDEN_SIZE * 4;
    let off = layer_idx * per_layer + (SEQ_LEN - 1) * row_bytes;
    data[off..off + row_bytes]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)) as f32 }
}

fn build_qwen3_client() -> Client {
    Client::builder()
        .model(MODEL)
        .kind(ModelKind::Chat)
        .build()
        .unwrap_or_else(|e| panic!("build Qwen3-0.6B Q4_0 client: {e}"))
}

/// 形状指纹: first 5 values + L2 norm (诊断对齐/发散的特征).
fn shape_fingerprint(v: &[f32]) -> String {
    let first5: Vec<String> = v.iter().take(5).map(|x| format!("{:.5}", x)).collect();
    let norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    format!("[{}] norm={:.4}", first5.join(", "), norm)
}

/// 决定跑哪些层: 全 28 层 (默认) 或关键层 (P1_KEY_LAYERS_ONLY=1 时).
fn layers_to_run() -> Vec<usize> {
    if std::env::var("P1_KEY_LAYERS_ONLY").is_ok() {
        eprintln!("[P1] P1_KEY_LAYERS_ONLY=1 → 只跑关键层 0/1/2/3/14/27 定断点");
        vec![0, 1, 2, 3, 14, 27]
    } else {
        (0..NUM_LAYERS).collect()
    }
}

// ─── P1 主诊断: real-loop 早退逐层 bisection ──────────────────────────

#[test]
fn p1_real_loop_layer_bisect() {
    eprintln!("\n=== P1: real-loop 早退逐层 bisection (encode_to_layer vs golden hs_{{N+1}} last row) ===");
    std::io::stderr().flush().ok();

    let path = golden_bin_path();
    assert!(path.exists(), "golden bin missing: {GOLDEN_BIN} (需先运行 python 预转)");

    let client = build_qwen3_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("tokens = {:?} (len={})", tokens, tokens.len());
    assert_eq!(tokens.len(), SEQ_LEN, "Qwen3 prompt 应 tokenize 成 {SEQ_LEN} tokens");
    eprintln!("P1 路径: encode_to_layer(N, Absolute, LastToken) → MidLayerEncode callback ExitEarly");
    eprintln!("  (real-loop 真循环, 不碰 ring-buffer capture, target_layers()=None 无 per-layer 重编译)");

    let layers = layers_to_run();
    eprintln!("\n layer | cosine(gllm_last, golden_hs_{{N+1}}_last_row) | verdict | shape(gllm vs golden)");
    eprintln!("-------|--------------------------------------------|---------|----------------------");

    let mut first_diverge: Option<(usize, f32)> = None;
    let mut layer0_cos: Option<f32> = None;
    let mut results: Vec<(usize, f32)> = Vec::with_capacity(layers.len());

    for &n in &layers {
        let gllm_last = match client.encode_to_layer(
            PROMPT,
            LayerAnchor::Absolute(n),
            PoolMode::LastToken,
        ) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  {n:2}   | ENCODE ERROR: {e}");
                if n == 0 {
                    eprintln!("\n*** layer0 encode_to_layer 失败 — MidLayerEncode callback 未 wire 或 layer 越界 ***");
                }
                continue;
            }
        };
        // encode_to_layer LastToken 返回 [hidden_size]
        assert_eq!(
            gllm_last.len(), HIDDEN_SIZE,
            "layer {n}: encode_to_layer LastToken 返回 len={} 期望 {}", gllm_last.len(), HIDDEN_SIZE
        );

        let golden_last = load_golden_hs_last_row(n + 1); // layer N 输出 = golden hs_{N+1}
        let cos = cosine(&gllm_last, &golden_last);
        let verdict = if cos > 0.99 { "PASS" } else { "*** DIVERGE ***" };
        let shape_gllm = shape_fingerprint(&gllm_last);
        let shape_gold = shape_fingerprint(&golden_last);
        eprintln!("  {n:2}   | cos={cos:.4}                              | {verdict} |");
        eprintln!("       |   gllm  {shape_gllm}");
        eprintln!("       |   golden{shape_gold}");

        results.push((n, cos));
        if n == 0 {
            layer0_cos = Some(cos);
        }
        if first_diverge.is_none() && cos < 0.99 {
            first_diverge = Some((n, cos));
        }
        std::io::stderr().flush().ok();
    }

    eprintln!();
    eprintln!("=== ★第一验证点: layer0 真实循环早退 cos ===");
    match layer0_cos {
        Some(c) => {
            eprintln!("layer0 (N=0) real-loop 早退 cos = {:.4}", c);
            if c > 0.99 {
                eprintln!(">>> layer0 cos≈1.0 → 断点在 layer0→1 之后, 回原假设");
                eprintln!(">>> 'layer0 在循环里对' 在 real-loop 路径下确认 (非 capture 假象)");
            } else {
                eprintln!(">>> layer0 就发散 (cos<0.99) → 'layer0 在循环里对' 一直是 capture 假象");
                eprintln!(">>> 断点比想象早, 问题在 layer0 循环上下文 (首次 swap / 输入注入), 故事重写");
            }
        }
        None => {
            eprintln!(">>> layer0 未跑 (layers_to_run 不含 0) 或 encode 失败 — 无法判定第一验证点");
        }
    }

    eprintln!("\n=== 逐层结果汇总 ===");
    for (n, c) in &results {
        let mark = if c > &0.99 { " " } else { "*" };
        eprintln!("  layer {n:2}: cos={c:.4} {mark}");
    }

    eprintln!();
    match first_diverge {
        Some((n, cos)) => {
            eprintln!("=== ★首个发散层: layer {n} (cosine={cos:.4}) ===");
            // 跨层断点证据 (若 n>0 且前一层在结果里)
            if n > 0 {
                if let Some((_, prev_c)) = results.iter().find(|(ln, _)| *ln == n - 1) {
                    eprintln!("=== 跨层断点证据: layer {} cos={:.4} (对) → layer {n} cos={cos:.4} (错) ===",
                        n - 1, prev_c);
                } else {
                    eprintln!("(前一层 {} 未在本次运行中, 跳过跨层断点证据)", n - 1);
                }
            }
            if n == 0 {
                eprintln!(">>> layer 0 就发散: embedding/norm/QKV proj 在 real-loop 路径错 (查 golden 对齐)");
                eprintln!(">>> 注意: GLLM_SINGLE_LAYER=1 单层隔离 layer0 cos=1.0 已验证 — 若 P1 real-loop");
                eprintln!(">>>    layer0 也发散, 则 bug 在多迭代循环上下文 (首次 swap / KV 注入), 非 layer0 算子本身");
            } else {
                eprintln!(">>> layer {n} 发散 (前层都对): 跨层传递 bug 在 layer {}→{n} 之间", n - 1);
                eprintln!(">>> 三嫌疑: activation_alias (ping-pong swap) / weight_stride / KV cache 跨层");
                eprintln!(">>> 下一步 P2: 单层隔离 index={n} 注入 golden hs_{n} + layer{n} 权重, 无 swap:");
                eprintln!(">>>   输出==golden hs_{} → layer{n} 权重+模板对, bug 在多迭代 plumbing → A 坐实", n + 1);
                eprintln!(">>>   输出!=golden hs_{} → layer{n} 读错权重/decode, 与 A 无关", n + 1);
            }
        }
        None => {
            eprintln!("=== 全 {} 层 (本次运行) cosine > 0.99: 层路径无发散 ===", results.len());
            eprintln!(">>> 根因在层后 (final_norm / lm_head) — 单独诊断");
        }
    }

    // 不做硬 assert (诊断测试, 发散即信息). 但 layer0 cos 是关键信号, 显式打印供 V 验收.
    eprintln!("\n[P1 DONE] layer0_cos={:?} first_diverge={:?}", layer0_cos, first_diverge);
}
