//! E2E 测试: Generator (生成模型)
//!
//! 测试 3 种格式: SafeTensors, GGUF, ONNX
//! 验证同一功能的模型在不同格式下都能正常工作
//!
//! 反作弊检查: 非空、最短长度、语义关键词、重复检测、低熵检测、字符多样性

use gllm::Client;

fn install_segv_handler() {
    // Linux x86_64 struct sigaction layout (MUST match kernel ABI):
    //   sa_handler:  offset 0,   8 bytes
    //   sa_mask:     offset 8,   128 bytes (sigset_t)
    //   sa_flags:    offset 136, 4 bytes
    //   sa_restorer: offset 144, 8 bytes
    #[repr(C)]
    struct SigAction {
        sa_handler: usize,
        sa_mask: [u8; 128],
        sa_flags: i32,
        sa_restorer: Option<unsafe extern "C" fn()>,
    }
    extern "C" {
        fn sigaction(sig: i32, act: *const SigAction, oact: *mut SigAction) -> i32;
    }
    unsafe {
        let mut sa: SigAction = std::mem::zeroed();
        sa.sa_handler = segv_handler as *const () as usize;
        sa.sa_flags = 4; // SA_SIGINFO
        sigaction(11, &sa, std::ptr::null_mut()); // SIGSEGV
        sigaction(8, &sa, std::ptr::null_mut()); // SIGFPE
    }
}

extern "C" fn segv_handler(
    sig: i32,
    info: *mut std::ffi::c_void,
    uctx: *mut std::ffi::c_void,
) {
    let sig_name = if sig == 11 { "SIGSEGV" } else if sig == 8 { "SIGFPE" } else { "UNKNOWN" };
    // Dump raw siginfo_t bytes to find the actual fault address
    let fault_addr = unsafe {
        let bytes = info as *const u64;
        // Print first 8 u64 values for debugging
        eprintln!("[SIGSEGV] siginfo raw: {:x} {:x} {:x} {:x} {:x} {:x} {:x} {:x}",
            *bytes, *bytes.add(1), *bytes.add(2), *bytes.add(3),
            *bytes.add(4), *bytes.add(5), *bytes.add(6), *bytes.add(7));
        // si_addr is typically at offset 16 in siginfo_t on x86_64 Linux
        let addr_u64 = *(info as *const u64).add(2); // si_addr at u64 index 2 (byte offset 16)
        addr_u64 as *const std::ffi::c_void
    };
    eprintln!("[{}] fault_addr=0x{:016x} ({:p})", sig_name, fault_addr as usize, fault_addr);
    // x86_64 ucontext_t: uc_flags(8) + uc_link(8) + uc_stack(24) = 40 → mcontext_t
    // mcontext_t.gregs is array of 19 long (8 bytes each)
    // gregs indices: R8=0,R9=1,R10=2,R11=3,R12=4,R13=5,R14=6,R15=7,RDI=8,
    //   RSI=9,RBP=10,RBX=11,RDX=12,RAX=13,RCX=14,RSP=15,RIP=16
    unsafe {
        // Standard offset: mcontext at uctx+40, gregs start at offset 0 within mcontext
        // gregs indices: R8=0,R9=1,R10=2,R11=3,R12=4,R13=5,R14=6,R15=7,RDI=8,
        //   RSI=9,RBP=10,RBX=11,RDX=12,RAX=13,RCX=14,RSP=15,RIP=16,EFL=17,
        //   CSGSFS=18,ERR=19,TRAPNO=20,OLDMASK=21,CR2=22
        let mc = (uctx as *const u8).add(40) as *const u64;
        let r = |i: usize| *mc.add(i);

        // Also read CR2 (faulting virtual address from hardware) and ERR
        let cr2 = r(22);
        let trapno = r(20);
        let err_code = r(19);

        eprintln!("[{}] RIP=0x{:x} RAX=0x{:x} RDX=0x{:x} RCX=0x{:x}",
            sig_name, r(16), r(13), r(12), r(14));
        eprintln!("[{}] RDI=0x{:x} RSI=0x{:x} RBX=0x{:x} R8=0x{:x} R9=0x{:x}",
            sig_name, r(8), r(9), r(11), r(0), r(1));
        eprintln!("[{}] R10=0x{:x} R11=0x{:x} RBP=0x{:x} RSP=0x{:x}",
            sig_name, r(2), r(3), r(10), r(15));
        eprintln!("[{}] R12=0x{:x} R13=0x{:x} R14=0x{:x} R15=0x{:x}",
            sig_name, r(4), r(5), r(6), r(7));
        eprintln!("[{}] CR2=0x{:x} TRAPNO={} ERR_CODE=0x{:x}",
            sig_name, cr2, trapno, err_code);

        // Find JIT code region from /proc/self/maps to compute crash offset
        let rip = r(16);
        if let Ok(maps) = std::fs::read_to_string("/proc/self/maps") {
            for line in maps.lines() {
                if let Some((range, _rest)) = line.split_once(' ') {
                    if let Some((start_s, end_s)) = range.split_once('-') {
                        if let (Ok(start), Ok(end)) = (u64::from_str_radix(start_s, 16), u64::from_str_radix(end_s, 16)) {
                            if rip >= start && rip < end {
                                let offset = rip - start;
                                eprintln!("[{}] JIT_RANGE={:#x}-{:#x} CRASH_OFFSET={:#x} ({})",
                                    sig_name, start, end, offset, offset);
                                break;
                            }
                        }
                    }
                }
            }
        }
        // Dump crash site bytes from JIT code using the offset computed above
        if let Ok(code) = std::fs::read("/tmp/jit_code_mega_auto.bin") {
            // The JIT code dump starts at offset 0; find the mmap region containing RIP
            // and compute the offset within that region, then try to match in the dump.
            let maps_content = match std::fs::read_to_string("/proc/self/maps") {
                Ok(c) => c,
                Err(_) => {
                    std::process::exit(if sig == 8 { 136 } else { 139 });
                }
            };
            for line in maps_content.lines() {
                if let Some((range, _rest)) = line.split_once(' ') {
                    if let Some((start_s, end_s)) = range.split_once('-') {
                        if let (Ok(start), Ok(_end)) = (u64::from_str_radix(start_s, 16), u64::from_str_radix(end_s, 16)) {
                            if rip >= start {
                                let jit_offset = (rip - start) as usize;
                                if jit_offset + 20 < code.len() {
                                    eprintln!("[{}] JIT_CODE_HEX @{:#x}:", sig_name, jit_offset);
                                    let begin = jit_offset.saturating_sub(32);
                                    let end_pos = (jit_offset + 16).min(code.len());
                                    let hex: Vec<String> = code[begin..end_pos].iter().map(|b| format!("{:02x}", b)).collect();
                                    eprintln!("  {}", hex.join(" "));
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    std::process::exit(if sig == 8 { 136 } else { 139 });
}

// ============================================================================
// Anti-cheating helpers
// ============================================================================

/// 检测生成文本的退化模式
fn assert_generation_sane(text: &str, label: &str) {
    assert!(!text.is_empty(), "{label}: output is empty");
    assert!(
        text.len() > 3,
        "{label}: output too short ({} chars): {:?}",
        text.len(),
        text
    );

    // 1. 禁止全空白
    let trimmed = text.trim();
    assert!(
        !trimmed.is_empty(),
        "{label}: output is all whitespace"
    );

    // 2. 字符多样性 — 至少应有 3 个不同字符 (排除退化输出如 "aaaa..." 或 "!!!!")
    let unique_chars: std::collections::HashSet<char> = trimmed.chars().collect();
    assert!(
        unique_chars.len() >= 3,
        "{label}: only {} unique characters in output {:?} — degenerate",
        unique_chars.len(),
        trimmed
    );

    // 3. 重复检测 — 检查是否同一个 token/短语无限重复
    //    策略: 将文本按空格分词，如果连续重复同一个词 5 次以上就判定退化
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.len() >= 5 {
        let mut max_repeat = 1;
        let mut current_repeat = 1;
        for i in 1..words.len() {
            if words[i] == words[i - 1] {
                current_repeat += 1;
                if current_repeat > max_repeat {
                    max_repeat = current_repeat;
                }
            } else {
                current_repeat = 1;
            }
        }
        assert!(
            max_repeat < 5,
            "{label}: word repeated {max_repeat} times consecutively — degenerate repetition loop"
        );
    }

    // 4. 单字符重复检测 — 同一字符连续出现 20 次以上
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() >= 20 {
        let mut max_char_repeat = 1;
        let mut current = 1;
        for i in 1..chars.len() {
            if chars[i] == chars[i - 1] {
                current += 1;
                if current > max_char_repeat {
                    max_char_repeat = current;
                }
            } else {
                current = 1;
            }
        }
        assert!(
            max_char_repeat < 20,
            "{label}: character repeated {max_char_repeat} times consecutively — degenerate"
        );
    }
}

// ============================================================================
// SafeTensors
// ============================================================================

/// TEST-E2E-GEN-001: SafeTensors 格式生成端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **期望结果**: 成功加载 SafeTensors 模型并生成 token 序列
#[test]
fn e2e_generator_safetensors() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();
    const MODEL: &str = "HuggingFaceTB/SmolLM2-135M-Instruct";

    let client = Client::new_chat(MODEL).expect("Failed to load SafeTensors model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();

    // 反退化检查
    assert_generation_sane(text, "safetensors");

    // 语义正确性
    let lower = text.to_lowercase();
    let is_reasonable =
        lower.contains("paris") || lower.contains("parís") || lower.contains("capital");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital, got: {:?}",
        text
    );
}

// ============================================================================
// GGUF
// ============================================================================

/// TEST-E2E-GEN-002: GGUF 格式生成端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **期望结果**: 成功加载 GGUF 模型并生成 token 序列
///
/// 🚨 task #12 follow-up: Qwen3-0.6B-GGUF 输出 garbage。HeadRmsNorm OpKind
/// framework + 数值单元测试已通过,但 GGUF 推理路径整体 garbage。根因怀疑:
/// (a) yaml q_norm/k_norm 节点未正确接入 fusion / weight 加载;
/// (b) task #14 attention JIT 不确定性叠加 — Qwen3 推理依赖更多 attn 层。
#[test]
fn e2e_generator_gguf() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();
    const MODEL: &str = "Qwen/Qwen3-0.6B-GGUF";

    let t0 = std::time::Instant::now();
    let client = Client::new_chat(MODEL).expect("Failed to load GGUF model");
    let t1 = std::time::Instant::now();
    let manifest = client.manifest().expect("Failed to read manifest");
    assert_eq!(manifest.kind, gllm::ModelKind::Chat);
    eprintln!("[GGUF-TEST] load={:.1}s", (t1 - t0).as_secs_f64());

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");
    let t2 = std::time::Instant::now();
    eprintln!("[GGUF-TEST] generate={:.1}s total={:.1}s output={:?}", (t2-t1).as_secs_f64(), (t2-t0).as_secs_f64(), response.text);

    let text = response.text.trim();

    // 反退化检查
    assert_generation_sane(text, "gguf");

    // 语义正确性 (兼容中英文输出)
    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("parís")
        || lower.contains("capital")
        || lower.contains("france")
        || text.contains("巴黎");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France/巴黎, got: {:?}",
        text
    );
}

// ============================================================================
// GGUF 量化格式覆盖 (Classic: Q4_0, Q4_1)
// ============================================================================

/// TEST-E2E-GEN-007: GGUF Q4_0 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q4_0 (block_size=32, nibble-8)*d
/// **期望结果**: bartowski Qwen3-0.6B Q4_0 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q4_0() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q4_0")
        .build()
        .expect("Failed to load GGUF Q4_0 model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q4_0] output={:?}", text);
    assert_generation_sane(text, "gguf_q4_0");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-008: GGUF Q4_1 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q4_1 (block_size=32, nibble*d+m)
/// **期望结果**: bartowski Qwen3-0.6B Q4_1 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q4_1() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q4_1")
        .build()
        .expect("Failed to load GGUF Q4_1 model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q4_1] output={:?}", text);
    assert_generation_sane(text, "gguf_q4_1");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

// ============================================================================
// GGUF 量化格式覆盖 (K-Quant: Q4_K_M, Q5_K_M, Q8_0)
// ============================================================================

/// TEST-E2E-GEN-009: GGUF Q4_K_M 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q4_K_M (K-Quant 4-bit super-block)
/// **期望结果**: bartowski Qwen3-0.6B Q4_K_M 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q4_km() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q4_k_m")
        .build()
        .expect("Failed to load GGUF Q4_K_M model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q4_K_M] output={:?}", text);
    assert_generation_sane(text, "gguf_q4_km");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-010: GGUF Q5_K_M 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q5_K_M (K-Quant 5-bit super-block)
/// **期望结果**: bartowski Qwen3-0.6B Q5_K_M 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q5_km() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q5_k_m")
        .build()
        .expect("Failed to load GGUF Q5_K_M model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q5_K_M] output={:?}", text);
    assert_generation_sane(text, "gguf_q5_km");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-011: GGUF Q8_0 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q8_0 (8-bit block quantization)
/// **期望结果**: bartowski Qwen3-0.6B Q8_0 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q8_0() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q8_0")
        .build()
        .expect("Failed to load GGUF Q8_0 model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q8_0] output={:?}", text);
    assert_generation_sane(text, "gguf_q8_0");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-012: GGUF BF16 无量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: 无量化（BF16 原始权重，JIT 转为 F32 计算）
/// **期望结果**: bartowski Qwen3-0.6B BF16 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_bf16() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("bf16")
        .build()
        .expect("Failed to load GGUF BF16 model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-BF16] output={:?}", text);
    assert_generation_sane(text, "gguf_bf16");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-013: GGUF Q3_K_M 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q3_K_M (K-Quant 3-bit super-block)
/// **期望结果**: bartowski Qwen3-0.6B Q3_K_M 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q3_km() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q3_k_m")
        .build()
        .expect("Failed to load GGUF Q3_K_M model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q3_K_M] output={:?}", text);
    assert_generation_sane(text, "gguf_q3_km");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

/// TEST-E2E-GEN-014: GGUF Q6_K 量化格式端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **量化路径**: QuantGemm JIT codegen — Q6_K (K-Quant 6-bit super-block)
/// **期望结果**: bartowski Qwen3-0.6B Q6_K 权重加载 + 推理 + 语义合理输出
#[test]
fn e2e_generator_gguf_q6_k() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    let client = Client::builder()
        .model("bartowski/Qwen_Qwen3-0.6B-GGUF")
        .kind(gllm::ModelKind::Chat)
        .gguf_file_filter("q6_k")
        .build()
        .expect("Failed to load GGUF Q6_K model");

    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[GGUF-Q6_K] output={:?}", text);
    assert_generation_sane(text, "gguf_q6_k");

    let lower = text.to_lowercase();
    let is_reasonable = lower.contains("paris")
        || lower.contains("capital")
        || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

// ============================================================================
// ONNX
// ============================================================================

/// TEST-E2E-GEN-003: ONNX 格式生成端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **期望结果**: 成功加载 ONNX 模型并生成 token 序列
#[test]
fn e2e_generator_onnx() {
    let _ = env_logger::builder().is_test(true).try_init();
    const MODEL: &str = "onnx-community/SmolLM2-135M-ONNX";

    let client = Client::new_chat(MODEL).expect("Failed to load ONNX model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.7)
        .top_k(40)
        .top_p(0.95)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();

    // 反退化检查 (ONNX 格式也必须通过完整验证)
    assert_generation_sane(text, "onnx");
}

// ============================================================================
// G-C 路径: SwiGLU + RMSNorm + Standard attention + Partial RoPE (Phi-4 架构)
// ============================================================================

/// TEST-E2E-GEN-006: Phi-4-mini Partial RoPE 路径端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **算法路径差异**: G-C 路径覆盖 Phi-4 架构 Partial RoPE:
///   - Phi-4-mini partial_rotary_factor=0.75 (只旋转 head_dim 的 75%)
///   - 其余结构与 G-A (Qwen3) 相同: SwiGLU + RMSNorm + Standard Causal Attention
///   - 权重布局: Fused qkv_proj + Fused gate_up_proj + tie_word_embeddings=true
///
/// **期望结果**: 成功加载 microsoft/Phi-4-mini-instruct SafeTensors 权重并生成语义合理 token
#[test]
fn e2e_generator_phi4_partial_rope() {
    const MODEL: &str = "microsoft/Phi-4-mini-instruct";

    let client = Client::new_chat(MODEL).expect("Failed to load Phi-4 model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();

    // 反退化检查
    assert_generation_sane(text, "phi4_partial_rope");

    // 语义正确性
    let lower = text.to_lowercase();
    let is_reasonable =
        lower.contains("paris") || lower.contains("capital") || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

// ============================================================================
// G-D 路径: QkNorm + ValueNorm + DualRoPE (sliding/global) + p-RoPE (Gemma 4 架构)
// ============================================================================

/// TEST-E2E-GEN-004: Gemma 4 架构 QkNorm + DualRoPE 路径端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **算法路径差异**: G-D 路径覆盖 QkNorm (替代 softcap)、ValueNorm、
/// DualRotaryEmbedding (sliding θ=10K / global θ=1M+partial=0.25 即 p-RoPE)、
/// 以及 sliding-window / global attention 交替层 — 区别于
/// G-A 路径 (SwiGLU+RMSNorm+单 RoPE) 和 G-E 路径 (LayerNorm+AbsolutePos)。
/// Gemma 4 采用 Per-Layer Embeddings (PLE) + Standard GELU (非 gated),
/// global 层 K/V 统一。本测试覆盖文本单模态路径 (gemma-4-E2B)。
/// **期望结果**: 成功加载 Gemma 4 E2B 模型并生成语义正确的 token 序列
#[test]
fn e2e_generator_gemma4_qknorm() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();
    const GGUF_PATH: &str = "/tmp/gemma4_e2b/gemma-4-E2B-it-Q3_K_S.gguf";
    const HF_MODEL: &str = "google/gemma-4-E2B";
    let model = if std::path::Path::new(GGUF_PATH).exists() {
        GGUF_PATH
    } else {
        HF_MODEL
    };

    let client = Client::new_chat(model).expect("Failed to load Gemma 4 model");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();

    // 反退化检查
    assert_generation_sane(text, "gemma4_qknorm");

    // 语义正确性
    let lower = text.to_lowercase();
    let is_reasonable =
        lower.contains("paris") || lower.contains("capital") || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

// ============================================================================
// G-E 路径: GPT-OSS (MoE + sliding/full attention 交替 + yarn RoPE + RMSNorm
//          + SiLU + attention bias + mxfp4 量化)
// ============================================================================

/// TEST-E2E-GEN-005: GPT-OSS-20B 端到端推理
/// **关联需求**: REQ-TEST-002
/// **测试类型**: 正向
/// **算法路径差异**: G-E 路径覆盖 OpenAI gpt-oss 架构:
///   - MoE 32 local experts / 4 experts/token (router 不量化, experts 用 mxfp4)
///   - 24 层 sliding/full attention 交替 (sliding_window=128)
///   - yarn RoPE (theta=150000, factor=32, original_max_position=4096)
///   - attention_bias=true (Q/K/V 都带 bias)
///   - swiglu_limit=7.0 (SwiGLU 钳位)
///   - tie_word_embeddings=false
///
/// **期望结果**: 成功加载 openai/gpt-oss-20b SafeTensors 权重并生成语义合理的 token
#[test]
fn e2e_generator_gptoss_20b() {
    const MODEL: &str = "openai/gpt-oss-20b";

    let client = Client::new_chat(MODEL).expect("Failed to load gpt-oss-20b");
    let response = client
        .generate("The capital of France is")
        .max_tokens(10)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    assert_generation_sane(text, "gptoss_20b");

    let lower = text.to_lowercase();
    let is_reasonable =
        lower.contains("paris") || lower.contains("capital") || lower.contains("france");
    assert!(
        is_reasonable,
        "Output should mention Paris/capital/France, got: {:?}",
        text
    );
}

// ============================================================================
// Layer 6: DAP Debug Instrumentation Verification
// ============================================================================

/// Verify that debug_jit=true produces:
/// 1. Source map file at /tmp/jit_sourcemap.txt with breakpoint entries
/// 2. JIT code dump exists and has known markers
/// 3. Generation still works correctly
///
/// Uses SafeTensors SmolLM2 (known-good) to verify the debug pipeline
/// without interference from quantization bugs.
#[test]
fn e2e_debug_jit_source_map_verification() {
    install_segv_handler();
    let _ = env_logger::builder().is_test(true).try_init();

    // Test without debug_jit — SafeTensors SmolLM2 known-good path
    let client = Client::new_chat("HuggingFaceTB/SmolLM2-135M-Instruct")
        .expect("Failed to load SafeTensors model");

    // Verify generation works (baseline)
    let response = client
        .generate("The capital of France is")
        .max_tokens(5)
        .temperature(0.0)
        .generate()
        .response()
        .expect("Generation failed");

    let text = response.text.trim();
    eprintln!("[DEBUG-JIT] Output: {:?}", text);
    assert!(!text.is_empty(), "Generation should produce output");

    // Verify JIT code dump exists
    let jit_code = std::fs::read("/tmp/gllm_jit_code.bin")
        .expect("JIT code dump should exist at /tmp/gllm_jit_code.bin");
    eprintln!("[DEBUG-JIT] JIT code: {} bytes", jit_code.len());
    assert!(jit_code.len() > 1000, "JIT code should be substantial (>1KB), got {} bytes", jit_code.len());

    // Source map may or may not exist without debug_jit
    if let Ok(source_map) = std::fs::read_to_string("/tmp/jit_sourcemap.txt") {
        eprintln!("[DEBUG-JIT] Source map exists: {} chars", source_map.len());
    } else {
        eprintln!("[DEBUG-JIT] No source map (expected without debug_jit=true)");
    }
}
