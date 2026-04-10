//! JIT Codegen 诊断分析脚本
//!
//! 模拟执行 + 寄存器对齐 + 符号执行融合分析
//!
//! 诊断流程:
//!   1. 加载模型 → 构建 FusedGraph → 模拟编译
//!   2. 逐节点符号执行: 追踪 shape 传播, 验证 buffer size
//!   3. 权重绑定审计: 检查 shape 缺失、dtype 不匹配、对齐违规
//!   4. 输出诊断报告: 精确定位 size mismatch / alignment fault / shape error
//!
//! 使用:
//!   cargo run --example diagnose_jit -- [model_id]
//!   默认: HuggingFaceTB/SmolLM2-135M-Instruct

use std::collections::{HashMap, HashSet};

// ============================================================================
// §1 — 诊断结果类型
// ============================================================================

#[derive(Debug)]
enum Severity {
    Fatal,
    Error,
    Warn,
    Info,
}

#[derive(Debug)]
struct Finding {
    severity: Severity,
    node_idx: Option<usize>,
    node_name: String,
    category: &'static str,
    message: String,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let sev = match self.severity {
            Severity::Fatal => "FATAL",
            Severity::Error => "ERROR",
            Severity::Warn => " WARN",
            Severity::Info => " INFO",
        };
        let idx = self.node_idx.map(|i| format!("[{i:>3}]")).unwrap_or_default();
        write!(
            f,
            "[{sev}]{idx} {:<24} {:<30} {}",
            self.category, self.node_name, self.message
        )
    }
}

// ============================================================================
// §2 — 符号 Shape 追踪器
// ============================================================================

struct SymbolicShapeTracker {
    /// 每个张量名 → 已知 shape
    shapes: HashMap<String, Vec<usize>>,
    /// 每个张量名 → 字节大小
    byte_sizes: HashMap<String, usize>,
}

impl SymbolicShapeTracker {
    fn new() -> Self {
        Self {
            shapes: HashMap::new(),
            byte_sizes: HashMap::new(),
        }
    }

    fn set(&mut self, name: &str, shape: Vec<usize>, dtype_bytes: usize) {
        let numel: usize = shape.iter().product();
        self.byte_sizes.insert(name.to_string(), numel * dtype_bytes);
        self.shapes.insert(name.to_string(), shape);
    }

    fn get_shape(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(|v| v.as_slice())
    }

    fn get_bytes(&self, name: &str) -> Option<usize> {
        self.byte_sizes.get(name).copied()
    }
}

// ============================================================================
// §3 — 对齐检查器
// ============================================================================

fn check_alignment(ptr: usize, required: usize, label: &str) -> Option<Finding> {
    if ptr % required != 0 {
        Some(Finding {
            severity: Severity::Error,
            node_idx: None,
            node_name: label.to_string(),
            category: "ALIGNMENT",
            message: format!(
                "ptr 0x{ptr:x} misaligned: need {required}B alignment, offset={}",
                ptr % required
            ),
        })
    } else {
        None
    }
}

// ============================================================================
// §4 — 主诊断引擎
// ============================================================================

fn run_diagnosis(model_id: &str) -> Vec<Finding> {
    let mut findings = Vec::new();

    // ── §4.1 加载模型 ──
    println!("=== JIT Codegen 诊断分析 ===");
    println!("模型: {model_id}");
    println!();

    use gllm::manifest::ModelKind;
    let kind = if model_id.to_lowercase().contains("embed") {
        ModelKind::Embedding
    } else if model_id.to_lowercase().contains("rerank") {
        ModelKind::Reranker
    } else {
        ModelKind::Chat
    };

    // 使用底层 API 构建 executor 的各组件, 在编译前拦截
    let config = gllm::loader::LoaderConfig::from_env();
    let mut loader = match gllm::loader::Loader::from_source_with_config(model_id.to_string(), config) {
        Ok(l) => l,
        Err(e) => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: "loader".into(),
                category: "INIT",
                message: format!("Loader 初始化失败: {e}"),
            });
            return findings;
        }
    };

    loader = match loader.load() {
        Ok(l) => l,
        Err(e) => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: "loader".into(),
                category: "INIT",
                message: format!("模型加载失败: {e}"),
            });
            return findings;
        }
    };

    let arch = loader.detect_architecture();
    println!("检测到架构: {arch}");

    let manifest = gllm::manifest::ModelManifest {
        model_id: std::borrow::Cow::Owned(model_id.to_string()),
        arch: arch.clone(),
        kind,
        ..Default::default()
    };

    let model_config = match gllm::model_config::ModelConfig::from_loader(&manifest, &mut loader) {
        Ok(c) => c,
        Err(e) => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: "config".into(),
                category: "INIT",
                message: format!("ModelConfig 解析失败: {e}"),
            });
            return findings;
        }
    };

    let geometry = gllm::model_config::ModelGeometry::from_config(
        &model_config,
        manifest.moe_config,
    );

    println!("模型参数:");
    println!("  hidden_size      = {}", geometry.hidden_size);
    println!("  num_heads        = {}", geometry.num_heads);
    println!("  num_kv_heads     = {}", geometry.num_kv_heads);
    println!("  head_dim         = {}", geometry.head_dim);
    println!("  num_layers       = {}", geometry.num_layers);
    println!("  vocab_size       = {}", geometry.vocab_size);
    println!("  intermediate     = {}", geometry.intermediate_size);
    println!("  dtype            = {:?}", geometry.dtype);
    println!();

    // ── §4.2 构建 FusedGraph (模板展开 + 图优化) ──
    gllm::arch::register_builtin_templates();

    let resolved = gllm::arch::ResolvedConfig::from_geometry(
        &std::sync::Arc::new(geometry.clone()),
        HashMap::new(),
    );

    let template = match gllm::arch::get_template(&arch) {
        Some(t) => t,
        None => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: arch.clone(),
                category: "TEMPLATE",
                message: "架构模板未注册".into(),
            });
            return findings;
        }
    };

    let onnx_graph = match template.to_onnx_graph(&resolved) {
        Ok(g) => g,
        Err(e) => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: arch.clone(),
                category: "TEMPLATE",
                message: format!("模板展开失败: {e}"),
            });
            return findings;
        }
    };

    println!("OnnxGraph: {} 节点, {} 输入, {} 输出",
        onnx_graph.nodes.len(), onnx_graph.inputs.len(), onnx_graph.outputs.len());

    let arch_family = manifest.family();
    let ctx = gllm::graph::optimizer::OptimizationContext {
        geometry: std::sync::Arc::new(geometry.clone()),
        arch_family,
        ..Default::default()
    };
    let optimizer = gllm::graph::optimizer::GraphOptimizer::new(ctx);

    let fused = match optimizer.optimize(&onnx_graph) {
        Ok(f) => f,
        Err(e) => {
            findings.push(Finding {
                severity: Severity::Fatal,
                node_idx: None,
                node_name: "optimizer".into(),
                category: "GRAPH_OPT",
                message: format!("图优化失败: {e}"),
            });
            return findings;
        }
    };

    println!("FusedGraph: {} 节点 (融合: {}, 原子: {})",
        fused.node_count(), fused.fused_op_count(),
        fused.node_count() - fused.fused_op_count());
    println!("  inputs:  {:?}", fused.inputs);
    println!("  outputs: {:?}", fused.outputs);
    println!("  weight_bindings: {} 条", fused.weight_bindings.len());
    println!();

    // ── §4.3 权重绑定审计 ──
    println!("=== §4.3 权重绑定审计 ===");
    let mut all_weight_names: HashSet<String> = HashSet::new();
    let graph_inputs_set: HashSet<&str> = fused.inputs.iter().map(String::as_str).collect();
    let mut produced: HashSet<String> = HashSet::new();
    for node in &fused.nodes {
        for o in &node.outputs {
            produced.insert(o.clone());
        }
    }

    for node in &fused.nodes {
        for inp in &node.inputs {
            if !produced.contains(inp) && !graph_inputs_set.contains(inp.as_str()) && inp.contains('.') {
                all_weight_names.insert(inp.clone());
            }
        }
    }

    let mut weight_shape_empty = 0usize;
    for name in &all_weight_names {
        if let Some(wb) = fused.weight_bindings.get(name) {
            if wb.shape.is_empty() {
                weight_shape_empty += 1;
                findings.push(Finding {
                    severity: Severity::Error,
                    node_idx: None,
                    node_name: name.clone(),
                    category: "WEIGHT_SHAPE",
                    message: "weight_binding.shape 为空 — build_node_graph 将使用 [seq_len, hidden] 默认值, MatMul shape 推导错误".into(),
                });
            }
        } else {
            findings.push(Finding {
                severity: Severity::Warn,
                node_idx: None,
                node_name: name.clone(),
                category: "WEIGHT_MISSING",
                message: "权重未在 weight_bindings 中注册 (模板图无 TensorProvider)".into(),
            });
        }
    }

    println!("  引用权重: {} 个", all_weight_names.len());
    println!("  已绑定:   {} 个", fused.weight_bindings.len());
    println!("  shape空:  {} 个", weight_shape_empty);
    println!("  未注册:   {} 个", all_weight_names.len() - fused.weight_bindings.len());
    println!();

    // ── §4.4 逐节点符号执行 ──
    println!("=== §4.4 符号执行 — 逐节点 shape 传播 ===");
    let seq_len = 1usize;
    let hidden = geometry.hidden_size;
    let head_dim = geometry.head_dim;
    let num_heads = geometry.num_heads;
    let num_kv_heads = geometry.num_kv_heads;
    let intermediate = geometry.intermediate_size;
    let vocab_size = geometry.vocab_size;

    let mut tracker = SymbolicShapeTracker::new();
    tracker.set("input_ids", vec![1, seq_len], 4);
    tracker.set("hidden_0", vec![seq_len, hidden], 4);

    for (idx, node) in fused.nodes.iter().enumerate() {
        let op_name = node.op.name();
        let is_atomic = matches!(&node.op, gllm::graph::types::FusedOp::Atomic(_));

        // 推导输出 shape
        let (expected_output_shape, expected_numel) = match &node.op {
            gllm::graph::types::FusedOp::FlashAttention(c) => {
                let s = vec![seq_len, c.num_heads * c.head_dim];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::SwiGLU(c) => {
                let s = vec![seq_len, c.intermediate_size];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::FusedQkvRope(c) => {
                let q = seq_len * c.num_heads * c.head_dim;
                let kv = seq_len * c.num_kv_heads * c.head_dim;
                (vec![q + kv + kv], q + kv + kv)
            }
            gllm::graph::types::FusedOp::FusedRMSLinear(c) => {
                let s = vec![seq_len, c.hidden_size];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::GQA(c) => {
                let s = vec![seq_len, c.num_heads * c.head_dim];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::RoPE(_) => {
                let s = vec![seq_len, hidden];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::MoERouting(c) => {
                let s = vec![seq_len, c.num_experts];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::PerLayerEmbed(_) => {
                let s = vec![seq_len, hidden];
                let n = s.iter().product();
                (s, n)
            }
            gllm::graph::types::FusedOp::Atomic(a) => {
                // 关键: 模拟 build_node_graph 的 Atomic 分支
                if a.op_type == "Gather" || a.op_type == "Slice" || a.op_type == "Shape" {
                    (vec![0], 0)
                } else {
                    // 重现 input_shapes 推导逻辑
                    let input_shapes: Vec<Vec<usize>> = node.inputs.iter().map(|name| {
                        // 1. 从 weight_bindings 查
                        if let Some(wb) = fused.weight_bindings.get(name) {
                            if !wb.shape.is_empty() {
                                return wb.shape.clone();
                            }
                        }
                        // 2. 从 tracker 查
                        if let Some(s) = tracker.get_shape(name) {
                            return s.to_vec();
                        }
                        // 3. 默认 (与 build_node_graph 一致)
                        vec![seq_len, hidden]
                    }).collect();

                    // 重现 infer_output_shape
                    let output_shape = match a.op_type.as_str() {
                        "MatMul" | "Gemm" => {
                            if input_shapes.len() >= 2
                                && input_shapes[0].len() >= 2
                                && input_shapes[1].len() >= 2
                            {
                                let a_shape = &input_shapes[0];
                                let b_shape = &input_shapes[1];
                                vec![a_shape[a_shape.len() - 2], b_shape[b_shape.len() - 1]]
                            } else {
                                input_shapes.first().cloned().unwrap_or(vec![1])
                            }
                        }
                        _ => input_shapes.first().cloned().unwrap_or(vec![1]),
                    };
                    let numel: usize = output_shape.iter().product();

                    // 检查 MatMul 权重 shape 是否合理
                    if a.op_type == "MatMul" && node.inputs.len() >= 2 {
                        let w_name = &node.inputs[1];
                        let w_shape = &input_shapes[1];
                        if w_shape == &[seq_len, hidden] && !tracker.shapes.contains_key(w_name) {
                            findings.push(Finding {
                                severity: Severity::Fatal,
                                node_idx: Some(idx),
                                node_name: node.name.clone(),
                                category: "SHAPE_DEFAULT",
                                message: format!(
                                    "MatMul 权重 '{w_name}' shape 使用默认值 [{seq_len},{hidden}] — \
                                     weight_bindings 无此条目或 shape 为空, \
                                     导致 output_numel={numel} 可能错误 (实际权重 shape 未知)"
                                ),
                            });
                        }
                        // 检查 MatMul 维度一致性
                        let a_k = input_shapes[0].last().copied().unwrap_or(0);
                        let b_k = if input_shapes[1].len() >= 2 {
                            input_shapes[1][input_shapes[1].len() - 2]
                        } else {
                            0
                        };
                        if a_k != b_k && b_k != 0 {
                            findings.push(Finding {
                                severity: Severity::Error,
                                node_idx: Some(idx),
                                node_name: node.name.clone(),
                                category: "SHAPE_MISMATCH",
                                message: format!(
                                    "MatMul 维度不一致: A[..., {a_k}] × B[{b_k}, ...] — K 维度不匹配"
                                ),
                            });
                        }
                    }

                    (output_shape, numel)
                }
            }
        };

        // 更新 tracker
        if !node.outputs.is_empty() && expected_numel > 0 {
            tracker.set(&node.outputs[0], expected_output_shape.clone(), 4);
        }

        // 缓冲区大小检查
        let output_bytes = expected_numel * 4; // f32
        let scratchpad_min = expected_numel * 4; // 至少与输出一样大

        if is_atomic && expected_numel > 0 {
            println!(
                "  [{idx:>3}] {:<40} {:<20} inputs={} outputs={} numel={} bytes={}",
                node.name,
                op_name,
                node.inputs.len(),
                node.outputs.len(),
                expected_numel,
                output_bytes,
            );
        } else if expected_numel > 0 {
            println!(
                "  [{idx:>3}] {:<40} {:<20} numel={} bytes={}",
                node.name, op_name, expected_numel, output_bytes,
            );
        }

        // SIMD 对齐检查 (AVX2 = 32B, AVX-512 = 64B)
        if output_bytes > 0 && output_bytes % 32 != 0 {
            findings.push(Finding {
                severity: Severity::Warn,
                node_idx: Some(idx),
                node_name: node.name.clone(),
                category: "ALIGNMENT",
                message: format!(
                    "输出缓冲区 {output_bytes}B 不是 32B 对齐 (AVX2), 余数={}",
                    output_bytes % 32
                ),
            });
        }
    }

    println!();

    // ── §4.5 Weight Layout 交叉验证 ──
    println!("=== §4.5 Weight Layout 交叉验证 ===");

    // 用真实权重 shape 与模板默认 shape 对比
    // (SafeTensors config check omitted — using direct header parsing instead)

    // 从 loader 获取真实张量元数据
    let mut real_weight_shapes: HashMap<String, (Vec<usize>, String)> = HashMap::new();
    // 直接读取 safetensors 文件获取张量元数据
    for path in loader.weight_paths() {
        if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            if let Ok(data) = std::fs::read(path) {
                // safetensors 文件格式: 8字节 header_size (LE), 然后 JSON header
                if data.len() > 8 {
                    let header_size = u64::from_le_bytes(data[..8].try_into().unwrap_or([0; 8])) as usize;
                    if header_size > 0 && 8 + header_size <= data.len() {
                        if let Ok(header_str) = std::str::from_utf8(&data[8..8 + header_size]) {
                            if let Ok(header) = serde_json::from_str::<serde_json::Value>(header_str) {
                                if let Some(obj) = header.as_object() {
                                    for (name, info) in obj {
                                        if name == "__metadata__" { continue; }
                                        let shape = info.get("shape")
                                            .and_then(|s| s.as_array())
                                            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect::<Vec<_>>())
                                            .unwrap_or_default();
                                        let dtype = info.get("dtype")
                                            .and_then(|d| d.as_str())
                                            .unwrap_or("?")
                                            .to_string();
                                        real_weight_shapes.insert(name.clone(), (shape, dtype));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("  真实权重张量: {} 个", real_weight_shapes.len());

    // 对每个模板引用的权重名, 查找真实 shape 并对比
    let mut mismatch_count = 0;
    for weight_name in &all_weight_names {
        // 尝试直接查找或去前缀查找
        let real = real_weight_shapes.get(weight_name)
            .or_else(|| {
                for prefix in &["model.", "roberta.", "bert.", "encoder."] {
                    if let Some(stripped) = weight_name.strip_prefix(prefix) {
                        if let Some(r) = real_weight_shapes.get(stripped) {
                            return Some(r);
                        }
                    }
                }
                None
            });

        if let Some((real_shape, dtype)) = real {
            // 检查默认 [seq_len, hidden] 与真实 shape 是否一致
            if *real_shape != vec![seq_len, hidden] && real_shape.len() >= 2 {
                let real_numel: usize = real_shape.iter().product();
                let default_numel = seq_len * hidden;
                if real_numel != default_numel {
                    mismatch_count += 1;
                    // 只报告 MatMul 权重 (通常是 .weight 后缀)
                    if weight_name.ends_with(".weight") {
                        let severity = if real_shape.len() == 2 && real_shape[0] != hidden {
                            Severity::Fatal
                        } else {
                            Severity::Error
                        };
                        findings.push(Finding {
                            severity,
                            node_idx: None,
                            node_name: weight_name.clone(),
                            category: "WEIGHT_SHAPE_REAL",
                            message: format!(
                                "真实 shape={real_shape:?} ({dtype}), 默认=[{seq_len},{hidden}], \
                                 numel 差异: {} vs {default_numel}",
                                real_numel
                            ),
                        });
                    }
                }
            }
        } else {
            findings.push(Finding {
                severity: Severity::Warn,
                node_idx: None,
                node_name: weight_name.clone(),
                category: "WEIGHT_NOT_FOUND",
                message: "权重在模型文件中未找到 (可能前缀不匹配)".into(),
            });
        }
    }

    println!("  shape 不匹配: {mismatch_count} 个");
    println!();

    // ── §4.6 GEMM 维度不匹配分析 ──
    println!("=== §4.6 GEMM 维度不匹配分析 (SIGSEGV 精确定位) ===");

    // safetensors 权重存储为 [out_features, in_features]
    // MatMul: activation[M, K] × weight^T[K, N] → output[M, N]
    // 其中 weight^T 的 K = real_shape[1], N = real_shape[0]
    for (idx, node) in fused.nodes.iter().enumerate() {
        if let gllm::graph::types::FusedOp::Atomic(a) = &node.op {
            if a.op_type != "MatMul" || node.inputs.len() < 2 {
                continue;
            }
            let act_name = &node.inputs[0];
            let w_name = &node.inputs[1];
            let real = real_weight_shapes.get(w_name)
                .or_else(|| {
                    for prefix in &["model.", "roberta.", "bert.", "encoder."] {
                        if let Some(stripped) = w_name.strip_prefix(prefix) {
                            if let Some(r) = real_weight_shapes.get(stripped) {
                                return Some(r);
                            }
                        }
                    }
                    None
                });

            if let Some((real_shape, _)) = real {
                if real_shape.len() >= 2 {
                    // 真实 GEMM 维度 (safetensors: [out, in] → MatMul 需要 [in, out])
                    let real_k = real_shape[1]; // in_features
                    let real_n = real_shape[0]; // out_features

                    // 编译期默认维度 (build_node_graph 使用 [seq_len, hidden])
                    let compiled_k = hidden;
                    let compiled_n = hidden;

                    // 活跃张量的真实维度
                    let act_real_dim = tracker.get_shape(act_name)
                        .and_then(|s| s.last().copied())
                        .unwrap_or(hidden);

                    if real_k != compiled_k {
                        let compiled_weight_bytes = compiled_k * compiled_n * 4;
                        let real_weight_bytes = real_k * real_n * 4;
                        let compiled_scratchpad = compiled_k * 4; // BLIS pack 至少需要 K 个元素
                        let real_scratchpad = real_k * 4;

                        findings.push(Finding {
                            severity: Severity::Fatal,
                            node_idx: Some(idx),
                            node_name: node.name.clone(),
                            category: "GEMM_K_MISMATCH",
                            message: format!(
                                "GEMM K 维度不匹配: 编译期 K={compiled_k} 实际 K={real_k} (N: 编译={compiled_n} 实际={real_n})\n\
                                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\
                                 权重 blob: 编译 {compiled_weight_bytes}B vs 实际 {real_weight_bytes}B\n\
                                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\
                                 活跃张量 '{act_name}' 维度={act_real_dim} (应为 {real_k})\n\
                                 \x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\
                                 → BLIS pack 读取 K={real_k} 但缓冲区按 K={compiled_k} 分配 → 堆越界"
                            ),
                        });
                    } else if real_n != compiled_n {
                        findings.push(Finding {
                            severity: Severity::Error,
                            node_idx: Some(idx),
                            node_name: node.name.clone(),
                            category: "GEMM_N_MISMATCH",
                            message: format!(
                                "GEMM N 维度不匹配: 编译期 N={compiled_n} 实际 N={real_n} → 输出缓冲区大小错误"
                            ),
                        });
                    }
                }
            }
        }
    }

    println!();

    // ── §4.7 寄存器冲突分析 ──
    println!("=== §4.7 寄存器冲突分析 ===");
    // 检查多输出节点 (FusedQkvRope) 的 per_output_numel 是否与 r12/r13/r14 分配冲突
    for (idx, node) in fused.nodes.iter().enumerate() {
        if let gllm::graph::types::FusedOp::FusedQkvRope(c) = &node.op {
            let q_dim = c.num_heads * c.head_dim;
            let kv_dim = c.num_kv_heads * c.head_dim;
            let total = seq_len * (q_dim + 2 * kv_dim);
            println!(
                "  [{idx:>3}] FusedQkvRope: q={q_dim} kv={kv_dim} total_numel={total} \
                 outputs={} (r12/r13/r14 分配 {} 个指针)",
                node.outputs.len(),
                node.outputs.len().min(3),
            );
            if node.outputs.len() > 3 {
                findings.push(Finding {
                    severity: Severity::Error,
                    node_idx: Some(idx),
                    node_name: node.name.clone(),
                    category: "REG_CONFLICT",
                    message: format!(
                        "多输出节点有 {} 个输出, 但 x86_64 ABI 只分配 r12/r13/r14 (3 个指针)",
                        node.outputs.len(),
                    ),
                });
            }
        }
    }
    println!();

    findings
}

// ============================================================================
// §5 — 入口
// ============================================================================

fn main() {
    let model_id = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "HuggingFaceTB/SmolLM2-135M-Instruct".to_string());

    let findings = run_diagnosis(&model_id);

    // 汇总报告
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    诊断报告汇总                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    let fatal_count = findings.iter().filter(|f| matches!(f.severity, Severity::Fatal)).count();
    let error_count = findings.iter().filter(|f| matches!(f.severity, Severity::Error)).count();
    let warn_count = findings.iter().filter(|f| matches!(f.severity, Severity::Warn)).count();
    let info_count = findings.iter().filter(|f| matches!(f.severity, Severity::Info)).count();

    println!();
    println!("  FATAL: {fatal_count}  ERROR: {error_count}  WARN: {warn_count}  INFO: {info_count}");
    println!();

    // 按严重程度分组输出
    for f in &findings {
        if matches!(f.severity, Severity::Fatal) {
            println!("  {f}");
        }
    }
    if fatal_count > 0 { println!(); }
    for f in &findings {
        if matches!(f.severity, Severity::Error) {
            println!("  {f}");
        }
    }
    if error_count > 0 { println!(); }
    for f in &findings {
        if matches!(f.severity, Severity::Warn) {
            println!("  {f}");
        }
    }

    println!();
    if fatal_count > 0 {
        println!("结论: 发现 {fatal_count} 个致命问题 — JIT 执行必然 SIGSEGV");
        println!();
        println!("根因: YAML 模板构建的 FusedGraph 没有 TensorProvider,");
        println!("      weight_bindings 为空 → Atomic MatMul 的权重 shape 默认为 [seq_len, hidden]");
        println!("      → infer_output_shape 计算错误 → output_numel 错误");
        println!("      → JIT 内核写入超出分配的输出缓冲区 → 堆损坏/SIGSEGV");
        println!();
        println!("修复方案:");
        println!("  1. compile_with_cache 前, 用 WeightsHandle 填充 FusedGraph.weight_bindings 的 shape");
        println!("  2. 或在 build_node_graph 的 Atomic 分支中, 从 WeightsHandle 获取真实权重 shape");
    } else {
        println!("结论: 未发现致命问题");
    }
}
