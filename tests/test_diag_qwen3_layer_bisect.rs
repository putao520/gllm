//! Qwen3-0.6B Q4_0 逐层 bisection 诊断 (跨层传递 bug 定位).
//!
//! 背景: Q4_0 bug 已数值定位到"层循环跨层传递" (非单层算子):
//!   - layer0 单独跑 (GLLM_SINGLE_LAYER=1) + 全循环都对 (cosine=1.0 vs golden hs_1)
//!   - 28 层循环跑完 layer27 全错 (cosine=0.005 vs golden hs_28)
//!   - 三嫌疑: activation_alias (ping-pong 错位) / weight_stride (权重跨层偏移错)
//!     / KV cache 跨层
//!
//! 本测试用 gllm-kernels 的 diagnostic-layer-capture ring-buffer 机制:
//!   - capture region 在 scratchpad, offset = diagnostic_tensor_offset("layer_capture")
//!   - 每层 stride = diagnostic_layer_capture_stride() = max_seq_len × hidden × 4
//!   - 第 N 层 (counter=N) 输出在 capture_base + N × stride, 完整 (seq, hidden) F32
//!   - capture 在 ActivationSwap 之前从 pong buffer copy (捕获当前层输出)
//!
//! Golden 对齐语义 (transformers output_hidden_states):
//!   - hs_0 = embedding 输出 (5, 1024)
//!   - hs_N (N>=1) = layer (N-1) 输出
//!   - 所以 capture 第 N 层 = golden hs_{N+1}
//!
//! 预转 golden: tests/e2e_alignment/data/golden_qwen3_layers.bin
//!   (29 个 (5,1024) f32 连续 = hs_0..hs_28, 共 593920 bytes)
//!
//! 运行:
//!   cargo test --features diagnostic-layer-capture \
//!     --test test_diag_qwen3_layer_bisect -- --nocapture --test-threads=1

#![cfg(target_os = "linux")]

use gllm::{Client, ModelKind};
use std::io::Write as _;

// ─── 常量 (Qwen3-0.6B) ────────────────────────────────────────────────
const MODEL: &str = "bartowski/Qwen_Qwen3-0.6B-GGUF";
const PROMPT: &str = "The capital of France is";
const SEQ_LEN: usize = 5;
const HIDDEN_SIZE: usize = 1024;
const NUM_LAYERS: usize = 28; // Qwen3-0.6B 28 层

/// Golden bin: 29 层 (hs_0=embed .. hs_28=layer27 out), 每个 (5, 1024) f32.
const GOLDEN_BIN: &str = "tests/e2e_alignment/data/golden_qwen3_layers.bin";

fn golden_bin_path() -> std::path::PathBuf {
    std::path::PathBuf::from(GOLDEN_BIN)
}

/// 读 golden hs_{layer_idx} (layer_idx 0..=28) → 展平 (5*1024,) f32.
fn load_golden_hs(layer_idx: usize) -> Vec<f32> {
    assert!(layer_idx <= 28, "layer_idx {layer_idx} out of range 0..=28");
    let data = std::fs::read(golden_bin_path())
        .unwrap_or_else(|e| panic!("read golden bin {GOLDEN_BIN}: {e}"));
    let per_layer = SEQ_LEN * HIDDEN_SIZE * 4; // f32 = 4 bytes
    assert_eq!(
        data.len(),
        29 * per_layer,
        "golden bin size {} != expected {}",
        data.len(),
        29 * per_layer
    );
    let off = layer_idx * per_layer;
    data[off..off + per_layer]
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

/// 从 capture region 读第 n 层输出 (seq_len, hidden_size) 展平 f32.
/// capture 区是 F32 side-channel copy (emit_side_channel_copy dtype=F32).
fn read_capture_layer(sp: &gllm::engine::mega_kernel::DiagnosticScratchpad, cap_off: usize, stride: usize, n: usize) -> Vec<f32> {
    let elem = 4usize; // capture = F32 side-channel
    let layer_off = cap_off + n * stride;
    let need = SEQ_LEN * HIDDEN_SIZE * elem;
    assert!(
        layer_off + need <= sp.data.len(),
        "capture layer {n} OOB: off={} need={} len={}", layer_off, need, sp.data.len()
    );
    (0..SEQ_LEN * HIDDEN_SIZE)
        .map(|i| {
            let b = &sp.data[layer_off + i * elem..layer_off + (i + 1) * elem];
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
        .collect()
}

// ─── 主诊断: 逐层 bisection ───────────────────────────────────────────

#[test]
fn diag_qwen3_layer_bisect() {
    eprintln!("\n=== Qwen3-0.6B Q4_0 逐层 bisection (capture vs golden hs_{{N+1}}) ===");
    std::io::stderr().flush().ok();

    let path = golden_bin_path();
    assert!(path.exists(), "golden bin missing: {GOLDEN_BIN} (需先运行 python 预转)");

    let client = build_qwen3_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("tokens = {:?} (len={})", tokens, tokens.len());
    assert_eq!(tokens.len(), SEQ_LEN, "Qwen3 prompt 应 tokenize 成 {SEQ_LEN} tokens");

    let cap_off = client
        .diagnostic_tensor_offset("layer_capture")
        .expect("layer_capture offset not found (diagnostic-layer-capture feature 未生效?)");
    let cap_stride = client.diagnostic_layer_capture_stride();
    assert!(cap_stride > 0, "layer_capture_stride=0 (feature 未生效或图无层循环)");
    eprintln!("cap_off={} cap_stride={} (= {} × {} × 4 = {})",
        cap_off, cap_stride, SEQ_LEN, HIDDEN_SIZE, SEQ_LEN * HIDDEN_SIZE * 4);

    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");
    eprintln!("scratchpad: data len={} bytes ({:.1} MB), dtype={:?}, elem_bytes={}",
        sp.data.len(), sp.data.len() as f64 / 1_048_576.0, sp.compute_dtype, sp.elem_bytes());

    // 逐层对比: capture layer N (0-based) = golden hs_{N+1}
    eprintln!("\n layer | cosine(capture_N, golden_hs_{{N+1}}) | verdict");
    eprintln!("-------|-----------------------------------|---------");
    let mut first_bad: Option<(usize, f32)> = None;
    for n in 0..NUM_LAYERS {
        let layer_out = read_capture_layer(&sp, cap_off, cap_stride, n);
        let golden_hs = load_golden_hs(n + 1); // capture layer N → golden hs_{N+1}
        let cos = cosine(&layer_out, &golden_hs);
        let verdict = if cos > 0.99 { "PASS" } else { "*** DIVERGE ***" };
        eprintln!("  {n:2}   | cos={cos:.4}                          | {verdict}");
        if first_bad.is_none() && cos < 0.99 {
            first_bad = Some((n, cos));
        }
    }

    eprintln!();
    match first_bad {
        Some((n, cos)) => {
            eprintln!("=== 首个发散层: layer {n} (cosine={cos:.4}) ===");
            // 证明跨层断点: N-1 对 (如果 N>0), N 错
            if n > 0 {
                // 重新读 layer (n-1) 的 cosine 证明跨层断点 (N-1 对 → N 错)
                let prev_layer_out = read_capture_layer(&sp, cap_off, cap_stride, n - 1);
                let prev_golden = load_golden_hs(n);
                let prev_c = cosine(&prev_layer_out, &prev_golden);
                eprintln!("=== 跨层断点证据: layer {} cos={:.4} (对) → layer {n} cos={cos:.4} (错) ===",
                    n - 1, prev_c);
            }
            // 诊断方向
            if n == 0 {
                eprintln!(">>> layer 0 就发散: capture 机制理解错 / golden 对齐错 / embedding 数据路径错");
                eprintln!(">>> 检查: raw first5 vs golden first5 (见 diag_qwen3_raw_check)");
            } else {
                eprintln!(">>> layer {n} 发散 (前 {} 层都对): 跨层传递 bug 在 layer {}→{n} 之间",
                    n, n - 1);
                eprintln!(">>> 三嫌疑: activation_alias (ping-pong 错位) / weight_stride (权重跨层偏移) / KV cache 跨层");
            }
        }
        None => {
            eprintln!("=== 全 {NUM_LAYERS} 层 cosine > 0.99: 层路径无发散 ===");
            eprintln!(">>> 根因在层后 (final_norm / lm_head) — 单独诊断");
        }
    }
}

// ─── raw 对齐检查 (layer 0 就发散时用) ────────────────────────────────

#[test]
fn diag_qwen3_raw_check() {
    eprintln!("\n=== Qwen3 raw capture layer0 vs golden hs_1 (对齐检查) ===");
    std::io::stderr().flush().ok();
    let client = build_qwen3_client();
    let tokens = client.encode(PROMPT).expect("encode");
    eprintln!("tokens = {:?}", tokens);

    let cap_off = client
        .diagnostic_tensor_offset("layer_capture")
        .expect("layer_capture offset not found");
    let cap_stride = client.diagnostic_layer_capture_stride();
    assert!(cap_stride > 0, "stride=0: feature 未生效");
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");

    // capture layer 0
    let l0 = read_capture_layer(&sp, cap_off, cap_stride, 0);
    let golden_h1 = load_golden_hs(1); // layer 0 out = golden hs_1
    eprintln!("capture layer0 first 5 = {:?}", &l0[0..5]);
    eprintln!("golden hs_1   first 5 = {:?}", &golden_h1[0..5]);
    let cos = cosine(&l0, &golden_h1);
    eprintln!("cosine(capture layer0, golden hs_1) = {cos:.4}");
    if cos > 0.99 {
        eprintln!(">>> layer 0 对齐! bug 在 layer 1+ 的跨层传递");
    } else {
        eprintln!(">>> layer 0 不对齐 — 需查 embedding / capture 语义");
    }

    // embedding 对齐 (capture layer 0 输入 = embedding = golden hs_0)
    let emb_off = client
        .diagnostic_tensor_offset("embedding")
        .expect("embedding offset not found");
    let elem = sp.elem_bytes();
    let mut emb = vec![0.0f32; SEQ_LEN * HIDDEN_SIZE];
    for i in 0..SEQ_LEN * HIDDEN_SIZE {
        let b = &sp.data[emb_off + i * elem..emb_off + (i + 1) * elem];
        emb[i] = match elem {
            4 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            2 => {
                // BF16: 高 16 位拼 F32
                let bits = (b[1] as u32) << 8 | b[0] as u32;
                f32::from_bits(bits << 16)
            }
            _ => 0.0,
        };
    }
    let golden_h0 = load_golden_hs(0); // embedding = golden hs_0
    eprintln!("\nembedding first 5 = {:?}", &emb[0..5]);
    eprintln!("golden hs_0 first 5 = {:?}", &golden_h0[0..5]);
    let cos_emb = cosine(&emb, &golden_h0);
    eprintln!("cosine(embedding, golden hs_0) = {cos_emb:.4}");
    if cos_emb > 0.99 {
        eprintln!(">>> embedding 对齐 (输入正确), bug 在 layer 0 计算 或 跨层");
    } else {
        eprintln!(">>> embedding 不对齐 — 根因在 embedding/gather/权重数据路径");
    }
}

// ─── 每 token 行交叉对齐 (检测行错位) ─────────────────────────────────

#[test]
fn diag_qwen3_per_row_alignment() {
    eprintln!("\n=== Qwen3 逐行 cosine (检测 token 行错位) ===");
    std::io::stderr().flush().ok();
    let client = build_qwen3_client();
    let tokens = client.encode(PROMPT).expect("encode");

    let cap_off = client.diagnostic_tensor_offset("layer_capture").expect("cap");
    let cap_stride = client.diagnostic_layer_capture_stride();
    assert!(cap_stride > 0);
    let sp = client.diagnostic_prefill_scratchpad(&tokens).expect("scratchpad");

    // 抽 layer 0 和 layer 27 (最后层) 做逐行对比
    for layer_n in [0usize, NUM_LAYERS - 1] {
        let layer_out = read_capture_layer(&sp, cap_off, cap_stride, layer_n);
        let golden = load_golden_hs(layer_n + 1);
        eprintln!("\n--- layer {layer_n} 逐行 cosine (capture vs golden hs_{}) ---", layer_n + 1);
        for r in 0..SEQ_LEN {
            let cap_row = &layer_out[r * HIDDEN_SIZE..(r + 1) * HIDDEN_SIZE];
            let gold_row = &golden[r * HIDDEN_SIZE..(r + 1) * HIDDEN_SIZE];
            let c = cosine(cap_row, gold_row);
            eprintln!("  row {r}: cosine = {c:.4}");
        }
        // 整体
        let full = cosine(&layer_out, &golden);
        eprintln!("  full (5×1024): cosine = {full:.4}");
    }
}

// ─── 步骤 1: compute_dtype 双 run 确认 (最高杠杆) ─────────────────────
//
// 诊断阶梯步骤 1: 读 SmolLM2(BF16 pass) 与 Qwen3(Q4_0 fail) 两次 run 的
// ModelGeometry.compute_dtype. 两者都 F32 → swap 作用于相同 F32 buffer,
// BF16 全过 → swap 没坏 → A(activation_alias) 排除 → 嫌疑=Q4_0 特有 decode/kernel.
// BF16 run 是 native BF16 → A 存活 (BF16 2B buffer vs Q4_0 F32 4B).
//
// compute_dtype 派生: model_config_fragments/types.inc.rs:167
//   compute_dtype = config.compute_dtype.unwrap_or(config.dtype)
// 用户未 override 时 = 权重 native dtype. Q4_0 是 storage dtype, 非计算 dtype,
// loader 对 Q4_0 的 config.dtype 解析决定 compute_dtype.
//
// 本测试读 DiagnosticScratchpad.compute_dtype (从 JIT 编译产物贯穿).

#[test]
fn diag_qwen3_step1_compute_dtype() {
    eprintln!("\n=== 步骤 1: compute_dtype 双 run 确认 ===");
    std::io::stderr().flush().ok();

    // ── Run A: Qwen3-0.6B Q4_0 (fail run) ──
    let qwen3_client = build_qwen3_client();
    let qwen3_tokens = qwen3_client.encode(PROMPT).expect("encode");
    let qwen3_sp = qwen3_client
        .diagnostic_prefill_scratchpad(&qwen3_tokens)
        .expect("qwen3 scratchpad");
    eprintln!("[Run A: Qwen3-0.6B Q4_0]");
    eprintln!("  tokens = {:?}", qwen3_tokens);
    eprintln!("  scratchpad.compute_dtype = {:?}", qwen3_sp.compute_dtype);
    eprintln!("  scratchpad.elem_bytes    = {}", qwen3_sp.elem_bytes());
    eprintln!("  scratchpad.data.len      = {} bytes ({:.1} MB)",
        qwen3_sp.data.len(), qwen3_sp.data.len() as f64 / 1_048_576.0);

    // ── Run B: SmolLM2-135M BF16 (pass run) ──
    // SmolLM2 权重是 BF16 safetensors. 用户未 override compute_dtype 时
    // compute_dtype = config.dtype = BF16. 若 gllm 默认对 BF16 权重 widen 到 F32,
    // 则 compute_dtype = F32.
    let smollm2_client = Client::builder()
        .model("HuggingFaceTB/SmolLM2-135M-Instruct")
        .kind(ModelKind::Chat)
        .build()
        .unwrap_or_else(|e| panic!("build SmolLM2 client: {e}"));
    let smollm2_tokens = smollm2_client.encode("The meaning of life is").expect("encode");
    let smollm2_sp = smollm2_client
        .diagnostic_prefill_scratchpad(&smollm2_tokens)
        .expect("smollm2 scratchpad");
    eprintln!("[Run B: SmolLM2-135M BF16]");
    eprintln!("  tokens = {:?}", smollm2_tokens);
    eprintln!("  scratchpad.compute_dtype = {:?}", smollm2_sp.compute_dtype);
    eprintln!("  scratchpad.elem_bytes    = {}", smollm2_sp.elem_bytes());
    eprintln!("  scratchpad.data.len      = {} bytes ({:.1} MB)",
        smollm2_sp.data.len(), smollm2_sp.data.len() as f64 / 1_048_576.0);

    // ── 判定 ──
    let qwen3_dt = qwen3_sp.compute_dtype;
    let smollm2_dt = smollm2_sp.compute_dtype;
    eprintln!("\n=== 步骤 1 判定 ===");
    eprintln!("  Qwen3(Q4_0 fail) compute_dtype = {:?}", qwen3_dt);
    eprintln!("  SmolLM2(BF16 pass) compute_dtype = {:?}", smollm2_dt);
    if qwen3_dt == smollm2_dt && qwen3_dt == gllm_kernels::types::DType::F32 {
        eprintln!(">>> 两 run 都 F32 → swap 逻辑作用于相同 F32 buffer, BF16 全过 → swap 没坏");
        eprintln!(">>> A(activation_alias) 排除 → 嫌疑 = Q4_0 特有 decode/kernel");
    } else if qwen3_dt != smollm2_dt {
        eprintln!(">>> dtype 不同 (Qwen3={:?} vs SmolLM2={:?}) → A 存活", qwen3_dt, smollm2_dt);
        eprintln!(">>> BF16 过不能排除 Q4_0 的 A (不同尺寸 buffer)");
    } else {
        eprintln!(">>> 两 run 同 dtype={:?} (非 F32) → 需结合尺寸判断", qwen3_dt);
    }
}

// ─── 步骤 3: T2 量 δ (运行时验证静态 weight_stride 结论) ──────────────
//
// 诊断阶梯步骤 3: dump weight_blob, 量 layer1 q_proj 实际起始偏移 vs
// layer_blob_base_offset + 1*weight_stride + rel_off(q_proj) 的 δ.
// 静态推 Qwen3 下 δ 应=0 (全 32 倍数无 padding); T2 验证运行时真=0.
//
// named_offsets 存参考层 canonical 名 (L0.q_proj 等), 绝对 blob offset.
// 因 named_offsets 只存 L0 参考, 无法直接读 L1.q_proj offset → 用字节签名搜索:
// 取 L0.q_proj 首 block 18 字节 (Q4_0: 2B fp16 scale + 16B packed),
// 在 blob 中搜该签名出现位置, 位置之差 = weight_stride (若 L1 q_proj 与 L0 相同 scale).
// 但 Q4_0 不同层 scale 不同 → 改用: 读 GGUF 文件 layer1 q_proj 首 block scale,
// 在 blob 中搜 layer1 scale 签名, 对比期望偏移.
//
// 更稳健: 直接断言 blob 边界 + rel_off 落在单层内 (静态结论的运行时不变量).

#[test]
fn diag_qwen3_step3_weight_blob_delta() {
    eprintln!("\n=== 步骤 3: T2 量 δ (weight_blob 偏移运行时验证) ===");
    std::io::stderr().flush().ok();

    let client = build_qwen3_client();
    let blob = client
        .diagnostic_weight_blob_bytes()
        .expect("weight blob");
    eprintln!("weight_blob len = {} bytes ({:.1} MB)", blob.len(), blob.len() as f64 / 1_048_576.0);

    // 读 named weight offsets (含 L0.* 参考层 + global)
    let weight_offs = client
        .diagnostic_weight_offsets()
        .expect("weight offsets");
    eprintln!("named weight offsets: {} entries", weight_offs.len());

    // 找 L0.q_proj + 所有 L0.* 层权重 + global (embed/final_norm/lm_head)
    let mut l0_weights: Vec<(String, usize, gllm_kernels::types::DType)> = Vec::new();
    let mut global_weights: Vec<(String, usize, gllm_kernels::types::DType)> = Vec::new();
    let mut l0_q_proj: Option<(usize, gllm_kernels::types::DType)> = None;
    for (name, off, dt) in &weight_offs {
        if name.starts_with("L0.") {
            l0_weights.push((name.clone(), *off, *dt));
            if name == "L0.q_proj" {
                l0_q_proj = Some((*off, *dt));
            }
        } else {
            global_weights.push((name.clone(), *off, *dt));
        }
    }
    eprintln!("\nL0.* per-layer weights ({} entries):", l0_weights.len());
    for (n, o, d) in &l0_weights {
        eprintln!("  {n:40} off={o:>12} dtype={:?}", d);
    }
    eprintln!("\nglobal weights ({} entries):", global_weights.len());
    for (n, o, d) in &global_weights {
        eprintln!("  {n:40} off={o:>12} dtype={:?}", d);
    }

    // layer_blob_base_offset = global weights 之后的层区起点.
    // 典型: embed 在前, 层区紧跟. layer_blob_base_offset = max(global weight end)
    // 但精确值在 LayerLoopConfig (gllm-kernels, 无 runtime accessor).
    // 经验推算: layer_blob_base_offset = 第一个 L0.* 权重的最小 offset.
    let layer_blob_base_offset = l0_weights.iter().map(|(_, o, _)| *o).min().unwrap_or(0);
    eprintln!("\nlayer_blob_base_offset (empirical, min L0.* offset) = {}", layer_blob_base_offset);

    // 单层权重字节总和 = max(L0.* end) - min(L0.* start)
    let l0_max_end = l0_weights.iter().map(|(_, o, _)| *o).max().unwrap_or(0);
    // 注: 这是 offset 不是 end; 单层 stride 无法从 offset 精确算 (需 tensor sizes).
    // 但可断言: global weights 全在 layer_blob_base_offset 之前 (offset < base)
    let global_max = global_weights.iter().map(|(_, o, _)| *o).max().unwrap_or(0);
    eprintln!("global weights max offset = {} (应 < layer_blob_base_offset={})",
        global_max, layer_blob_base_offset);

    // ── 断言 1: L0.q_proj rel_off 落在单层内 ──
    // rel_off = L0.q_proj.offset - layer_blob_base_offset ∈ [0, weight_stride)
    // 无 weight_stride runtime accessor → 用单层总字节估算 weight_stride.
    // Qwen3-0.6B 单层 Q4_0 字节 (粗算):
    //   q_proj: 1024×2048 Q4_0 = (1024*2048/32)*18 = 1179648
    //   k/v_proj: 1024×1024 Q4_0 = (1024*1024/32)*18 = 589824 each
    //   o_proj: 2048×1024 Q4_0 = 1179648
    //   gate/up: 1024×3072 Q4_0 = (1024*3072/32)*18 = 1769472 each
    //   down: 3072×1024 Q4_0 = 1769472
    //   norms (input/post/attn): BF16 or F32, ~1024*2 or *4
    //   ≈ 8.8 MB/layer
    if let Some((q_off, q_dt)) = l0_q_proj {
        let rel_off = q_off.saturating_sub(layer_blob_base_offset);
        eprintln!("\nL0.q_proj: offset={} dtype={:?} rel_off={}", q_off, q_dt, rel_off);
        eprintln!("  (rel_off 应 ∈ [0, weight_stride); weight_stride ≈ 8-9 MB for Qwen3 Q4_0)");

        // ── 断言 2: layer_blob_base_offset + 28*weight_stride ≤ blob.len() ──
        // 反推 weight_stride 下界: (blob.len() - layer_blob_base_offset) / 28
        let remaining = blob.len().saturating_sub(layer_blob_base_offset);
        let implied_stride = remaining / NUM_LAYERS;
        eprintln!("\n断言 2: blob.len() - layer_blob_base_offset = {}", remaining);
        eprintln!("  implied weight_stride (remaining/{}) = {} bytes ({:.1} MB)",
            NUM_LAYERS, implied_stride, implied_stride as f64 / 1_048_576.0);
        eprintln!("  尾部残留 (final_norm/lm_head) = {} bytes ({:.1} MB)",
            remaining - implied_stride * NUM_LAYERS,
            (remaining - implied_stride * NUM_LAYERS) as f64 / 1_048_576.0);
        assert!(
            layer_blob_base_offset + NUM_LAYERS * implied_stride <= blob.len(),
            "blob 边界溢出: base + {}*{} > blob.len()",
            NUM_LAYERS, implied_stride
        );
        assert!(
            rel_off < implied_stride,
            "L0.q_proj rel_off {} >= implied_stride {} (rel_off 不在单层内 → weight_stride 定义错)",
            rel_off, implied_stride
        );
        eprintln!(">>> 断言通过: blob 边界 OK + rel_off ∈ [0, stride)");

        // ── T2: 读 layer1 q_proj 期望偏移的首 block scale ──
        // 期望偏移 = layer_blob_base_offset + 1*implied_stride + rel_off
        let layer1_q_proj_expected_off = layer_blob_base_offset + implied_stride + rel_off;
        eprintln!("\nT2: layer1 q_proj 期望偏移 = {} + {} + {} = {}",
            layer_blob_base_offset, implied_stride, rel_off, layer1_q_proj_expected_off);
        if layer1_q_proj_expected_off + 18 <= blob.len() {
            let scale_bytes = &blob[layer1_q_proj_expected_off..layer1_q_proj_expected_off + 2];
            let scale_bits = u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]);
            let scale_f16 = half::f16::from_bits(scale_bits);
            let scale_f32 = scale_f16.to_f32();
            eprintln!("  layer1 q_proj 首 block: fp16 scale bits=0x{:04x} → f16={:.6} → f32={:.6}",
                scale_bits, scale_f16, scale_f32);
            eprintln!("  (δ 验证: 对比 GGUF 文件 layer1 q_proj 首 block scale — 需 python 脚本)");
        }
    } else {
        eprintln!(">>> L0.q_proj 未在 named_offsets 中找到 — 跳过 T2 δ 量");
        eprintln!("    (named_offsets 可能用不同 canonical 名, 见 weight_offsets 列表上方)");
    }

    // dump blob 末尾 + 开头供 python 对比
    let _ = std::fs::write("/tmp/gllm_qwen3_weight_blob.bin", &blob);
    eprintln!("\n[DUMP] weight_blob → /tmp/gllm_qwen3_weight_blob.bin ({} bytes)", blob.len());
}

// ─── 步骤 4: 条件分支分析 (据 1-3 结果) + 未决问题确认 ────────────────
//
// 汇总步骤 1-3 结果, 输出条件分支结论 + 三个未决问题答案.
// 本测试只做分析输出 (不跑推理), 依赖步骤 1-3 已运行的数据 (重新读 compute_dtype).

#[test]
fn diag_qwen3_step4_conditional_analysis() {
    eprintln!("\n=== 步骤 4: 条件分支分析 + 未决问题确认 ===");
    std::io::stderr().flush().ok();

    // 重读 compute_dtype (步骤 1 结论)
    let qwen3_client = build_qwen3_client();
    let qwen3_tokens = qwen3_client.encode(PROMPT).expect("encode");
    let qwen3_sp = qwen3_client
        .diagnostic_prefill_scratchpad(&qwen3_tokens)
        .expect("sp");
    let qwen3_dt = qwen3_sp.compute_dtype;

    let smollm2_client = Client::builder()
        .model("HuggingFaceTB/SmolLM2-135M-Instruct")
        .kind(ModelKind::Chat)
        .build()
        .unwrap_or_else(|e| panic!("build SmolLM2: {e}"));
    let smollm2_tokens = smollm2_client.encode("The meaning of life is").expect("encode");
    let smollm2_sp = smollm2_client
        .diagnostic_prefill_scratchpad(&smollm2_tokens)
        .expect("sp");
    let smollm2_dt = smollm2_sp.compute_dtype;

    eprintln!("--- 步骤 1 结论 ---");
    eprintln!("  Qwen3(Q4_0) compute_dtype = {:?}", qwen3_dt);
    eprintln!("  SmolLM2(BF16) compute_dtype = {:?}", smollm2_dt);

    let a_excluded = qwen3_dt == smollm2_dt && qwen3_dt == gllm_kernels::types::DType::F32;
    eprintln!("\n--- 步骤 4 条件分支 ---");
    if a_excluded {
        eprintln!(">>> A(activation_alias) 排除 (两 run 都 F32)");
        eprintln!(">>> 进 Q4_0 GEMM/GEMV 路径查跨迭代复用未重置 buffer/累加器 (decode scratch 污染)");
        eprintln!(">>> 嫌疑落点: quant_gemm.inc.rs / lower_op.inc.rs Q4_0 Assisted GEMV 路径");
        eprintln!(">>> 若也无跨层复用 → 问题在更隐蔽处 (Q4_0 权重指针 per-layer 常量未随 li 更新)");
    } else {
        eprintln!(">>> A 存活 (dtype 不同 或 非 F32) → 需单层隔离 index=1 测试");
        eprintln!(">>> 注入 golden hs_1 输入 + layer1 权重, 无 swap: 输出==golden hs_2 → A 坐实");
    }

    eprintln!("\n--- 三个未决问题 (executor 确认) ---");
    eprintln!("Q1: 单层隔离 index=1 能否注入 golden hs_1 + layer1 权重?");
    eprintln!("  当前: GLLM_SINGLE_LAYER=1 只跑 layer0 (num_layers=1, 权重偏移 base+0).");
    eprintln!("  无 knob 跑任意层 index + 注入输入. 需加 knob: GLLM_SINGLE_LAYER_INDEX=N");
    eprintln!("  + 注入输入 hook. harness 不支持 → 步骤4 A存活分支需先加 knob.");
    eprintln!("Q2: Q4_0 GEMV 路径有无跨 layer 迭代复用未重置 buffer/累加器?");
    eprintln!("  需 grep quant_gemm.inc.rs / lower_op.inc.rs Assisted GEMV 路径 (本测试外静态确认).");
    eprintln!("Q3: ActivationSwap 是否 dtype 无关单一路径?");
    eprintln!("  BF16/Q4_0 compute_dtype 都 F32 (步骤1) → swap 作用于相同 F32 buffer →");
    eprintln!("  共用同 swap 代码路径 (dtype 无关). BF16 过 → swap 没坏 → A 排除 (若步骤1 两F32).");
}

// ─── architect round 26: 0层截断 embed dump (Q4_0 vs BF16) ───────────
//
// 0层截断乱码 + 9 oracle 过 → bug 在 embed(QuantGather) 或 lm_head(QuantGemm) 真实组合.
// 0层无 ActivationSwap → embed 输出不被 ping-pong 覆盖 → dump 可信 (round-16 工件不适用).
// 对比 Q4_0 embed vs BF16 embed (同 prompt token 0):
//   - 差异大 → Q4_0 embed (QuantGather) 错
//   - 相似 → embed 对, bug 在 final_norm/lm_head

#[test]
fn diag_qwen3_0layer_embed_dump() {
    eprintln!("\n=== architect round 26: 1层截断 logits dump (Q4_0 vs BF16) ===");
    std::io::stderr().flush().ok();

    let prompt = "The capital of France is";

    // 1层截断 (0层 SIGSEGV, num_layers=0 不合法). 1层 = embed→1层→final_norm→lm_head→argmax
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");

    // ── Run A: Q4_0 1层 ──
    let q4_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("q4_0")
        .build()
        .expect("Q4_0 client");
    let q4_tokens = q4_client.encode(prompt).expect("encode");
    let q4_sp = q4_client.diagnostic_prefill_scratchpad(&q4_tokens).expect("q4 scratchpad");
    // logits 在最后一行 (seq_len-1) 的 vocab 区
    let q4_vocab = q4_sp.vocab_size;
    let q4_logits = q4_sp.read_dtype_aware(
        q4_sp.logits_offset, q4_vocab
    );
    let q4_argmax = q4_logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i,_)| i).unwrap_or(0);
    let q4_max = q4_logits.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[Q4_0 1层] argmax={} (BF16 应 Paris=12095), logits|max|={:.4}, vocab={}", q4_argmax, q4_max, q4_vocab);
    drop(q4_client);

    // ── Run B: BF16 1层 ──
    let bf_client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(ModelKind::Chat)
        .gguf_file_filter("bf16")
        .build()
        .expect("BF16 client");
    let bf_tokens = bf_client.encode(prompt).expect("encode");
    let bf_sp = bf_client.diagnostic_prefill_scratchpad(&bf_tokens).expect("bf16 scratchpad");
    let bf_vocab = bf_sp.vocab_size;
    let bf_logits = bf_sp.read_dtype_aware(
        bf_sp.logits_offset, bf_vocab
    );
    let bf_argmax = bf_logits.iter().enumerate()
        .max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i,_)| i).unwrap_or(0);
    let bf_max = bf_logits.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[BF16 1层] argmax={} (Paris=12095), logits|max|={:.4}, vocab={}", bf_argmax, bf_max, bf_vocab);

    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    eprintln!("[对比] Q4_0 argmax={} vs BF16 argmax={}", q4_argmax, bf_argmax);
    eprintln!("  Q4_0 logits|max|={:.4} vs BF16 |max|={:.4} (量级差异大→Q4_0 dequant 错)", q4_max, bf_max);

    // ── dump 中间张量 |max| 序列, 定位量级放大点 ──
    eprintln!("\n=== 中间张量 |max| 序列 (定位量级放大点) ===");
    let dump_intermediates = |sp: &gllm::engine::mega_kernel::DiagnosticScratchpad, label: &str| {
        eprintln!("[{}] named_offsets count={}", label, sp.named_offsets.len());
        let mut sorted: Vec<&(String, usize, gllm_kernels::types::DType)> = sp.named_offsets.iter().collect();
        sorted.sort_by_key(|e| e.1);
        for (name, off, dt) in sorted.iter().take(40) {
            let elem = dt.size_bytes();
            if elem == 0 { continue; }
            // 读前 1024 elem 算 |max|
            let count = 1024.min((sp.data.len() - off) / elem);
            if count == 0 { continue; }
            let vals = sp.read_dtype_aware(*off, count);
            let mx = vals.iter().fold(0.0f32, |m, &v| if v.is_finite() { m.max(v.abs()) } else { m });
            eprintln!("  [{}] {:30} off={:8} dt={:?} |max|={:.6}", label, name, off, dt, mx);
        }
    };
    // 重新跑 Q4_0 + BF16 dump 中间张量 (前面 client 已 drop, 重跑)
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let q4_client2 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q4_0").build().expect("q4");
    let q4_tok2 = q4_client2.encode(prompt).expect("encode");
    let q4_sp2 = q4_client2.diagnostic_prefill_scratchpad(&q4_tok2).expect("sp");
    dump_intermediates(&q4_sp2, "Q4_0");
    drop(q4_client2);
    let bf_client2 = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("bf16").build().expect("bf");
    let bf_tok2 = bf_client2.encode(prompt).expect("encode");
    let bf_sp2 = bf_client2.diagnostic_prefill_scratchpad(&bf_tok2).expect("sp");
    dump_intermediates(&bf_sp2, "BF16");
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");

    // ── architect round 27: lm_head 权重 blob vs embed 字节级对照 ──
    // lm_head 是 embed 的 tied 副本. 若副本字节一致 → QuantGemm 读法错;
    // 若副本字节错位 → loader copy 错. 读真实 weight_blob (非 scratchpad).
    eprintln!("\n=== round 27: lm_head blob vs embed blob 字节级对照 (Q4_0) ===");
    std::env::set_var("GLLM_TRUNCATE_LAYERS", "1");
    let q4c = Client::builder().model("bartowski/Qwen_Qwen3-0.6B-GGUF").kind(ModelKind::Chat).gguf_file_filter("q4_0").build().expect("q4");
    let q4t = q4c.encode(prompt).expect("encode");
    let q4s = q4c.diagnostic_prefill_scratchpad(&q4t).expect("sp");
    let wblob = q4c.diagnostic_weight_blob_bytes().expect("weight_blob");
    let woffs = q4c.diagnostic_weight_offsets().expect("weight_offsets");
    let find_woff = |name: &str| -> Option<(usize, gllm_kernels::types::DType)> {
        woffs.iter().find(|(n,_,_)| n == name).map(|(_,o,d)| (*o, *d))
    };
    let (embed_off, _) = find_woff("embed").expect("embed woff");
    let (lm_off, _) = find_woff("lm_head").expect("lm_head woff");
    eprintln!("[round27] weight_blob len={} embed off={} lm_head off={}", wblob.len(), embed_off, lm_off);
    let row_bytes = 576usize;
    let emb_row0 = &wblob[embed_off..embed_off+row_bytes];
    let lm_row0  = &wblob[lm_off..lm_off+row_bytes];
    let byte_match = emb_row0.iter().zip(lm_row0.iter()).filter(|(a,b)| a==b).count();
    eprintln!("[round27] embed row0 vs lm_head row0 字节一致: {}/{}", byte_match, row_bytes);
    let decode_q4_0_block = |bytes: &[u8]| -> Vec<f32> {
        assert!(bytes.len() >= 18);
        let scale = half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32();
        let mut out = Vec::with_capacity(32);
        for j in 0..16 {
            let lo = (bytes[2 + j] & 0x0F) as f32;
            let hi = ((bytes[2 + j] >> 4) & 0x0F) as f32;
            out.push((lo - 8.0) * scale);
            out.push((hi - 8.0) * scale);
        }
        out
    };
    let emb_blk = decode_q4_0_block(&wblob[embed_off..embed_off+18]);
    let lm_blk  = decode_q4_0_block(&wblob[lm_off..lm_off+18]);
    eprintln!("[round27] embed block0 首8值: {:?}", &emb_blk[..8.min(emb_blk.len())]);
    eprintln!("[round27] lm_head block0 首8值: {:?}", &lm_blk[..8.min(lm_blk.len())]);
    let blk_match = emb_blk.iter().zip(lm_blk.iter()).filter(|(a,b)| ((*a)-(*b)).abs() < 1e-6).count();
    eprintln!("[round27] embed block0 vs lm_head block0 值一致: {}/32", blk_match);
    let lm_row1 = &wblob[lm_off+row_bytes..(lm_off+2*row_bytes).min(wblob.len())];
    let lm_row1_blk0 = decode_q4_0_block(&lm_row1[..18]);
    eprintln!("[round27] lm_head row1 block0 首4值: {:?}", &lm_row1_blk0[..4.min(lm_row1_blk0.len())]);
    // 逐块 scale 对照: lm_head row0 的 32 个块 scale (f16)
    let mut r0_scales = Vec::new();
    for b in 0..32 {
        let bo = lm_off + b*18;
        if bo+2 <= wblob.len() {
            r0_scales.push(half::f16::from_le_bytes([wblob[bo], wblob[bo+1]]).to_f32());
        }
    }
    eprintln!("[round27] lm_head row0 32块 scale (首8): {:?}", &r0_scales[..8.min(r0_scales.len())]);
    // embed row0 scales 对照
    let mut e0_scales = Vec::new();
    for b in 0..32 {
        let bo = embed_off + b*18;
        if bo+2 <= wblob.len() {
            e0_scales.push(half::f16::from_le_bytes([wblob[bo], wblob[bo+1]]).to_f32());
        }
    }
    eprintln!("[round27] embed   row0 32块 scale (首8): {:?}", &e0_scales[..8.min(e0_scales.len())]);
    // 跨多行统计 lm_head blob 是否非零
    let lm_region = &wblob[lm_off..(lm_off + 28*row_bytes).min(wblob.len())];
    let nonzero = lm_region.iter().filter(|b| **b != 0).count();
    eprintln!("[round27] lm_head 首28行非零字节: {}/{}", nonzero, lm_region.len());
    let emb_region = &wblob[embed_off..(embed_off + 28*row_bytes).min(wblob.len())];
    let emb_nonzero = emb_region.iter().filter(|b| **b != 0).count();
    eprintln!("[round27] embed  首28行非零字节: {}/{}", emb_nonzero, emb_region.len());
    let _ = q4s;
    // ── architect round 28: output.weight 是 Q6_K (非 Q4_0!). 标量参考解码 vs 我们路径 ──
    eprintln!("\n=== round 28: output.weight Q6_K 解码对照 (标量参考 vs 我们路径) ===");
    // Q6_K block: qs[128] + qh[64] + scales[16 i8] + d(f16) = 210B, block_size=256
    // 标量参考 (k_quant.rs:442): value = d * sc[idx] * (6bit_val - 32)
    //   6bit_val = (ql[l]&0xF) | ((qh[l]>>(2*q))&3)<<4)  q=0..3 分 4 quarter
    // 我们路径: NibbleWithHighBits, qh<<6 & 0x30 (疑似丢高 2 bit)
    let decode_q6k_row_scalar = |blob: &[u8], row_off: usize, n_elem: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; n_elem];
        let blocks = n_elem / 256;
        for blk in 0..blocks {
            let bo = row_off + blk * 210;
            let d = half::f16::from_le_bytes([blob[bo+208], blob[bo+209]]).to_f32();
            let qs = &blob[bo..bo+128];
            let qh = &blob[bo+128..bo+192];
            let sc = &blob[bo+192..bo+208];
            for n_group in 0..2usize {
                let ql_off = n_group*64; let qh_off = n_group*32; let sc_off = n_group*8; let out_off = n_group*128;
                for l in 0..32usize {
                    let is = l/16;
                    let q1 = (((qs[ql_off+l] & 0xF) | (((qh[qh_off+l] >> 0) & 3) << 4)) as i32) - 32;
                    let q2 = (((qs[ql_off+l+32] & 0xF) | (((qh[qh_off+l] >> 2) & 3) << 4)) as i32) - 32;
                    let q3 = (((qs[ql_off+l] >> 4) | (((qh[qh_off+l] >> 4) & 3) << 4)) as i32) - 32;
                    let q4 = (((qs[ql_off+l+32] >> 4) | (((qh[qh_off+l] >> 6) & 3) << 4)) as i32) - 32;
                    out[blk*256 + out_off + l]     = d * (sc[sc_off+is] as i8 as f32) * (q1 as f32);
                    out[blk*256 + out_off + l+32]   = d * (sc[sc_off+is+2] as i8 as f32) * (q2 as f32);
                    out[blk*256 + out_off + l+64]   = d * (sc[sc_off+is+4] as i8 as f32) * (q3 as f32);
                    out[blk*256 + out_off + l+96]   = d * (sc[sc_off+is+6] as i8 as f32) * (q4 as f32);
                }
            }
        }
        out
    };
    // lm_head row0 = first 1024 elements (hidden of vocab 0). Q6_K row stride = (1024/256)*210 = 840 bytes
    let q6k_row_bytes = (1024/256)*210; // 840
    let lm_row0_scalar = decode_q6k_row_scalar(&wblob, lm_off, 1024);
    let lm_row0_smax = lm_row0_scalar.iter().fold(0.0f32, |m,&v| if v.is_finite(){m.max(v.abs())}else{m});
    eprintln!("[round28] lm_head row0 Q6_K 标量参考解码: |max|={:.4} 首8值: {:?}", lm_row0_smax, &lm_row0_scalar[..8]);
    // 对照: 我们路径若丢高 2 bit, value=(lo4)-32, 范围[-32,-17]
    let decode_q6k_row_4bit = |blob: &[u8], row_off: usize, n_elem: usize| -> Vec<f32> {
        // 模拟我们的 buggy 路径: 仅取 lo4, 高 2 bit 丢失
        let mut out = vec![0.0f32; n_elem];
        let blocks = n_elem / 256;
        for blk in 0..blocks {
            let bo = row_off + blk*210;
            let d = half::f16::from_le_bytes([blob[bo+208], blob[bo+209]]).to_f32();
            let qs = &blob[bo..bo+128];
            let sc = &blob[bo+192..bo+208];
            for n_group in 0..2usize {
                for l in 0..32usize {
                    let is = l/16;
                    let q1 = (qs[n_group*64+l] & 0xF) as i32 - 32;
                    let q2 = (qs[n_group*64+l+32] & 0xF) as i32 - 32;
                    let q3 = (qs[n_group*64+l] >> 4) as i32 - 32;
                    let q4 = (qs[n_group*64+l+32] >> 4) as i32 - 32;
                    out[blk*256+n_group*128+l]    = d*(sc[n_group*8+is] as i8 as f32)*(q1 as f32);
                    out[blk*256+n_group*128+l+32]  = d*(sc[n_group*8+is+2] as i8 as f32)*(q2 as f32);
                    out[blk*256+n_group*128+l+64]  = d*(sc[n_group*8+is+4] as i8 as f32)*(q3 as f32);
                    out[blk*256+n_group*128+l+96]  = d*(sc[n_group*8+is+6] as i8 as f32)*(q4 as f32);
                }
            }
        }
        out
    };
    let lm_row0_4bit = decode_q6k_row_4bit(&wblob, lm_off, 1024);
    let lm_4bit_max = lm_row0_4bit.iter().fold(0.0f32, |m,&v| if v.is_finite(){m.max(v.abs())}else{m});
    eprintln!("[round28] lm_head row0 我们路径(4bit丢高2bit): |max|={:.4} 首8值: {:?}", lm_4bit_max, &lm_row0_4bit[..8]);
    // BF16 对照 (off from bf wblob, not available here; use prior knowledge ~0.3)
    eprintln!("[round28] 判定: 标量参考|max|≈0.3(BF16级)→我们路径若~0.3则Q6K解码对,bug在别处; 若>>0.3则Q6K解码错");

    // ── round 28b: 真实 dot product 对照. act=final_normed, weight=lm_head row j ──
    // 手算 dot(act, lm_head_row_j) Q6_K 标量参考, 对比 JIT logits[j]
    eprintln!("\n=== round 28b: dot(act, lm_head_row_j) 标量参考 vs JIT logits ===");
    // act = final_normed (sp.data @ final_normed off, 1024 f32 for last token)
    let fnorm_off = q4s.named_offsets.iter().find(|(n,_,_)| n=="final_normed").map(|(_,o,_)|*o).expect("final_normed off");
    // final_normed 是 [seq, hidden]; last token (pos seq_len-1) = offset + (seq_len-1)*hidden*4
    let ntok = q4s.prompt_len;
    // 先读 final_normed 第 0 个 token (前 1024 elem, 之前 dump 显示 |max|=82.77)
    let act0_off = fnorm_off;
    let act0: Vec<f32> = (0..1024).map(|i| {
        let b = act0_off + i*4;
        if b+4 <= q4s.data.len() { f32::from_le_bytes([q4s.data[b],q4s.data[b+1],q4s.data[b+2],q4s.data[b+3]]) } else { 0.0 }
    }).collect();
    let act0_max = act0.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[round28b] final_normed token0 (off={}) |max|={:.4}, prompt_len={}", act0_off, act0_max, ntok);
    let act_off = fnorm_off + (ntok.saturating_sub(1)) * 1024 * 4;
    let act: Vec<f32> = (0..1024).map(|i| {
        let b = act_off + i*4;
        if b+4 <= q4s.data.len() { f32::from_le_bytes([q4s.data[b],q4s.data[b+1],q4s.data[b+2],q4s.data[b+3]]) } else { 0.0 }
    }).collect();
    let act_max = act.iter().fold(0.0f32, |m,&v| m.max(v.abs()));
    eprintln!("[round28b] act(final_normed last token off={}) 1024 elem |max|={:.4}", act_off, act_max);
    // JIT logits
    let jit_logits: Vec<f32> = (0..q4s.vocab_size).map(|i| {
        let b = q4s.logits_offset + i*4;
        f32::from_le_bytes([q4s.data[b],q4s.data[b+1],q4s.data[b+2],q4s.data[b+3]])
    }).collect();
    let jit_argmax = jit_logits.iter().enumerate().max_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap()).map(|(i,_)| i).unwrap_or(0);
    eprintln!("[round28b] JIT logits argmax={} val={:.4}, |max|={:.4}", jit_argmax, jit_logits[jit_argmax], jit_logits.iter().fold(0.0f32,|m,&v|m.max(v.abs())));
    // 手算 dot for token 0 和 argmax token, 对比
    for &tok in &[0usize, jit_argmax, 12095usize] {
        let row_off = lm_off + tok * q6k_row_bytes;
        let row = decode_q6k_row_scalar(&wblob, row_off, 1024);
        let dot: f64 = act0.iter().zip(row.iter()).map(|(a,w)| (*a as f64)*(*w as f64)).sum();
        let jit_val = jit_logits.get(tok).copied().unwrap_or(f32::NAN);
        eprintln!("[round28b] token {}: 手算dot(act0,Q6K_row)={:.4} vs JIT logit={:.4} (差 {:.4})", tok, dot, jit_val, (dot as f32 - jit_val).abs());
    }
    std::env::remove_var("GLLM_TRUNCATE_LAYERS");
}

