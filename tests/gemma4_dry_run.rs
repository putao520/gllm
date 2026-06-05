//! T40: Gemma 4 E2B 全管线 dry-run 诊断
//!
//! 目的: 不依赖实际模型下载, 用模拟 config + 模拟 tensor shape 跑通
//!   阶段 A: template.to_onnx_graph (YAML 展开)
//!   阶段 B: GraphOptimizer.optimize (pattern fusion / constant folding / DCE)
//!   阶段 C: FusedGraphExecutor.from_graph (JIT compile)
//!
//! 每阶段 try-catch, 分别收集错误, 写最终 Markdown 报告到
//! `/tmp/gemma4_dry_run_report.md`, 测试自身不 panic — 目的是**列清单**,
//! 不是**挡合并**。
//!
//! 运行: cargo test --test gemma4_dry_run -- --nocapture

use gllm::arch::{ArchTemplate, ResolvedConfig};
use gllm::graph::optimizer::{GraphOptimizer, OptimizationContext};
use gllm::loader::{TensorMeta, TensorProvider};
use gllm::manifest::ArchFamily;
use safetensors::Dtype;
use std::borrow::Cow;
use std::collections::HashMap;

/// Mock provider that advertises every weight/tensor Gemma 4 YAML will
/// reference, with correct shapes. `load_tensor_data` is unused because
/// dry-run never reads bytes (only `bind_weight_shapes_fuzzy` is invoked).
#[derive(Debug)]
struct MockGemma4Provider {
    tensors: HashMap<String, TensorMeta>,
}

impl MockGemma4Provider {
    fn new(config: &ResolvedConfig) -> Self {
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let inter = config.intermediate_size.unwrap();
        let dim_per_layer = config.hidden_size_per_layer_input;
        let num_layers = config.num_hidden_layers;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut tensors = HashMap::new();
        let mut insert = |name: String, shape: Vec<usize>| {
            tensors.insert(
                name.clone(),
                TensorMeta {
                    name,
                    shape,
                    dtype: Dtype::F32,
                },
            );
        };

        // Global weights.
        insert("model.embed_tokens.weight".into(), vec![vocab, hidden]);
        insert("model.norm.weight".into(), vec![hidden]);

        // PLE globals.
        insert(
            "model.per_layer_embedding.embed_tokens.weight".into(),
            vec![vocab, num_layers * dim_per_layer],
        );
        insert(
            "model.per_layer_embedding.per_layer_projection.weight".into(),
            vec![hidden, dim_per_layer],
        );

        // Per-layer weights.
        for i in 0..num_layers {
            let prefix = format!("model.layers.{i}");
            insert(format!("{prefix}.self_attn.q_proj.weight"), vec![q_dim, hidden]);
            insert(format!("{prefix}.self_attn.k_proj.weight"), vec![kv_dim, hidden]);
            insert(format!("{prefix}.self_attn.v_proj.weight"), vec![kv_dim, hidden]);
            insert(format!("{prefix}.self_attn.o_proj.weight"), vec![hidden, q_dim]);
            insert(format!("{prefix}.mlp.gate_proj.weight"), vec![inter, hidden]);
            insert(format!("{prefix}.mlp.up_proj.weight"), vec![inter, hidden]);
            insert(format!("{prefix}.mlp.down_proj.weight"), vec![hidden, inter]);
            insert(format!("{prefix}.input_layernorm.weight"), vec![hidden]);
            insert(format!("{prefix}.post_attention_layernorm.weight"), vec![hidden]);
            insert(
                format!("{prefix}.post_mlp_projection.weight"),
                vec![hidden, dim_per_layer],
            );
        }

        Self { tensors }
    }
}

impl TensorProvider for MockGemma4Provider {
    fn tensor_info(&self, name: &str) -> Option<TensorMeta> {
        self.tensors.get(name).cloned()
    }

    fn iter_tensors(&self) -> impl Iterator<Item = TensorMeta> {
        self.tensors.values().cloned().collect::<Vec<_>>().into_iter()
    }

    fn load_tensor_data(&self, name: &str) -> gllm::loader::Result<Cow<'_, [u8]>> {
        // Not used by dry-run (we never actually execute), but keep it safe.
        if let Some(meta) = self.tensors.get(name) {
            let numel: usize = meta.shape.iter().product();
            Ok(Cow::Owned(vec![0u8; numel * 4]))
        } else {
            Ok(Cow::Borrowed(&[]))
        }
    }
}

/// Build the canonical Gemma 4 E2B `ResolvedConfig` used throughout T33–T39.
fn make_gemma4_e2b_config() -> ResolvedConfig {
    let mut config = ResolvedConfig::default();
    config.num_hidden_layers = 26;
    config.hidden_size = 2048;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    config.intermediate_size = Some(16384);
    config.vocab_size = 262208;
    config.rope_theta = 10_000.0;
    config.global_rope_theta = 1_000_000.0;
    config.sliding_window = 512;
    config.hidden_size_per_layer_input = 128;
    config.num_kv_shared_layers = 20;
    // Every 6th layer is global (indices 5, 11, 17, 23).
    config.attention_pattern = (0..26)
        .map(|i| if (i + 1) % 6 == 0 { 1u8 } else { 0u8 })
        .collect();
    config.has_per_layer_embedding = true;
    config.dtype = "f32".to_string();
    config
}

#[derive(Debug)]
struct Blocker {
    stage: &'static str, // A / B / C
    op_type: String,
    name: String,
    message: String,
    root_cause: &'static str,
    suggested_fix: &'static str,
    priority: &'static str, // High / Medium / Low
}

const UNKNOWN: &str = "(to be analysed — inspect message)";

/// Attempt each pipeline stage, collecting blockers (not panic-ing).
fn run_dry_run() -> Vec<Blocker> {
    let mut blockers: Vec<Blocker> = Vec::new();

    // ── Stage A: YAML → OnnxGraph ─────────────────────────────────────────
    let config = make_gemma4_e2b_config();
    let template_src = include_str!("../src/arch/templates/gemma4.yaml");
    let template = match ArchTemplate::from_yaml(template_src) {
        Ok(t) => t,
        Err(e) => {
            blockers.push(Blocker {
                stage: "A",
                op_type: "(yaml-parse)".into(),
                name: "gemma4.yaml".into(),
                message: format!("ArchTemplate::from_yaml: {e}"),
                root_cause: "gemma4.yaml 语法或字段错误",
                suggested_fix: "修正 YAML 语法,根据 serde_yaml 报错定位到具体字段。",
                priority: "High",
            });
            return blockers;
        }
    };

    let graph = match template.to_onnx_graph(&config) {
        Ok(g) => g,
        Err(e) => {
            blockers.push(Blocker {
                stage: "A",
                op_type: "(template-expand)".into(),
                name: "to_onnx_graph".into(),
                message: format!("{e}"),
                root_cause: "template.to_onnx_graph 展开失败(占位符缺失 / repeat 计数错 / expand_* 契约错)",
                suggested_fix: "定位到 template.rs 中的展开函数或 resolve.rs::substitute_placeholders,\
                                补齐占位符/字段映射。",
                priority: "High",
            });
            return blockers;
        }
    };

    println!(
        "[T40 dry-run] Stage A ok — {} nodes, {} inputs, {} outputs",
        graph.nodes.len(),
        graph.inputs.len(),
        graph.outputs.len()
    );

    // ── Stage B: GraphOptimizer.optimize ─────────────────────────────────
    let geometry = std::sync::Arc::new(gllm::model_config::ModelGeometry {
        hidden_size: config.hidden_size,
        num_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        num_layers: config.num_hidden_layers,
        vocab_size: config.vocab_size,
        intermediate_size: config.intermediate_size.unwrap(),
        max_seq_len: 4096,
        rope_theta: config.rope_theta,
        rope_scale: 1.0,
        rope_interleaved: false,
        dtype: gllm_kernels::types::DType::F32,
        norm_eps: 1e-6,
        num_experts: 0,
        moe_top_k: 0,
        expert_intermediate_size: 0,
        global_rope_theta: config.global_rope_theta,
        rope_partial_ratio: config.rope_partial_ratio,
        attention_pattern: config.attention_pattern.clone(),
        sliding_window: config.sliding_window,
        num_kv_shared_layers: config.num_kv_shared_layers,
        global_head_dim: config.global_head_dim,
        hidden_size_per_layer_input: config.hidden_size_per_layer_input,
        position_offset: None,
        rope_scaling: None,
        final_logit_softcapping: None,
        hidden_act: None,
        compute_dtype: gllm_kernels::types::DType::F32,
        mla_d_c: 0,
        mla_d_rope: 0,
        mla_unabsorbed_threshold: 0,
    });
    let ctx = OptimizationContext {
        geometry,
        arch_family: ArchFamily::Decoder,
        ..Default::default()
    };
    let optimizer = GraphOptimizer::new(ctx.clone());
    let fused = match optimizer.optimize(&graph) {
        Ok(f) => f,
        Err(e) => {
            blockers.push(Blocker {
                stage: "B",
                op_type: "(optimizer)".into(),
                name: "GraphOptimizer::optimize".into(),
                message: format!("{e}"),
                root_cause: "pattern_fusion / hardware_fusion / constant_folding / DCE 其中一个 Pass 报错",
                suggested_fix: "读错误消息确定是哪个 Pass,查对应模块的 run() 实现。",
                priority: "High",
            });
            return blockers;
        }
    };
    println!(
        "[T40 dry-run] Stage B ok — fused {} nodes ({} fused ops)",
        fused.nodes.len(),
        fused.fused_op_count()
    );

    // Tally fused op variants so we can tell which pattern passes actually fired.
    let mut op_tally: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for node in &fused.nodes {
        let key = match &node.op {
            gllm::graph::types::FusedOp::Atomic(a) => format!("Atomic({})", a.op_type),
            gllm::graph::types::FusedOp::FusedQkvNormRope(_) => "FusedQkvNormRope".into(),
            gllm::graph::types::FusedOp::FusedQkvRope(_) => "FusedQkvRope".into(),
            gllm::graph::types::FusedOp::SwiGLU(_) => "SwiGLU".into(),
            gllm::graph::types::FusedOp::FlashAttention(_) => "FlashAttention".into(),
            gllm::graph::types::FusedOp::GQA(_) => "GQA".into(),
            gllm::graph::types::FusedOp::FusedRMSLinear(_) => "FusedRMSLinear".into(),
            gllm::graph::types::FusedOp::MoERouting(_) => "MoERouting".into(),
            other => format!("{:?}", other).chars().take(30).collect::<String>(),
        };
        *op_tally.entry(key).or_default() += 1;
    }
    println!("[T40 dry-run] Op tally after Stage B:");
    for (k, v) in &op_tally {
        println!("    {k}: {v}");
    }

    // FusedQkvNormRope is the SPEC-intended fusion for Gemma 4 (has QkNorm + ValueNorm).
    // Expected count: num_hidden_layers (1 per layer).
    // Getting 0 means the pattern window didn't match — most likely because
    // template.rs::expand_qk_norm pre-splits QkNorm into 2 independent nodes
    // but FusedQkvNormRopeFusionPass expects a single 2-in/2-out QkNorm node.
    let qkv_norm_rope_count = *op_tally.get("FusedQkvNormRope").unwrap_or(&0);
    if qkv_norm_rope_count < config.num_hidden_layers {
        blockers.push(Blocker {
            stage: "B",
            op_type: "FusedQkvNormRope".into(),
            name: "pattern-fusion contract mismatch".into(),
            message: format!(
                "FusedQkvNormRope fired {} times, expected {} (1 per layer). \
                 Instead the graph retains {} atomic QkNorm + {} atomic ValueNorm + \
                 {} atomic RotaryEmbedding nodes (each layer should collapse them into 1 fused op).",
                qkv_norm_rope_count,
                config.num_hidden_layers,
                op_tally.get("Atomic(QkNorm)").unwrap_or(&0),
                op_tally.get("Atomic(ValueNorm)").unwrap_or(&0),
                op_tally.get("Atomic(RotaryEmbedding)").unwrap_or(&0),
            ),
            root_cause:
                "Contract 错配: template.rs::expand_qk_norm 把 YAML 的 QkNorm (2-in / 2-out) \
                 预先拆成 2 个独立 1-in / 1-out 节点,但 FusedQkvNormRopeFusionPass (pattern_fusion.rs:195) \
                 的滑动窗口期望 1 个 QkNorm (2-in / 2-out) + 1 个 ValueNorm + 2 个 RoPE 共 7 个连续节点。\
                 由于节点结构不匹配, 该 Pass 对 Gemma 4 任意层都不触发。",
            suggested_fix:
                "二选一:\n  \
                 (方案 A — 改 pattern_fusion): 把 FusedQkvNormRopeFusionPass 的窗口从 7 扩成 8 节点, \
                 接受 Q-QkNorm 和 K-QkNorm 两个独立节点, 保留 template 拆分语义。\n  \
                 (方案 B — 改 template): 不拆 QkNorm, 直接在 YAML 保留单个 2-in / 2-out 节点, \
                 把 head_dim 属性从 expand_qk_norm 移到 node_def_to_onnx 的公共注入点; \
                 要求 atomic_op_to_kind::QkNorm 支持多路输入。\n  \
                 方案 A 风险小(只改 optimizer 窗口大小), 推荐。",
            priority: "High",
        });
    }

    // ── Stage B.5: bind mock weight shapes into fused graph ──────────────
    // The executor's build_tensor_shape_map relies on weight_bindings for
    // MatMul/Gather shape inference. For a dry-run we just plug in shapes
    // via TensorProvider — no data allocation.
    let provider = MockGemma4Provider::new(&config);

    // Re-run optimize but feed shapes via graph.inputs propagation is not
    // supported for weight tensors, so we drop into the executor path and
    // populate weight_bindings manually on the FusedGraph the executor owns.
    let mut fused_with_weights = fused;
    // bind_weight_shapes_fuzzy covers both model.layers.{N}.X variants and
    // PLE aliases — the same path production loaders use.
    let bound = fused_with_weights.bind_weight_shapes_fuzzy(&provider, /*format_needs_transpose=*/ false);
    println!("[T40 dry-run] Stage B.5: bound {bound} weight shapes");

    // Report any unbound weight-like inputs (i.e. names used as node inputs
    // but not in weight_bindings and not produced by any node / not graph input).
    let produced: std::collections::HashSet<&str> = fused_with_weights
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(String::as_str))
        .collect();
    let graph_inputs: std::collections::HashSet<&str> = fused_with_weights
        .inputs
        .iter()
        .map(String::as_str)
        .collect();
    let mut unbound: Vec<String> = Vec::new();
    for node in &fused_with_weights.nodes {
        for input in &node.inputs {
            if input.is_empty()
                || produced.contains(input.as_str())
                || graph_inputs.contains(input.as_str())
                || fused_with_weights.weight_bindings.contains_key(input)
            {
                continue;
            }
            unbound.push(format!("{} (node={})", input, node.name));
        }
    }
    if !unbound.is_empty() {
        // Dedup.
        unbound.sort();
        unbound.dedup();

        // Classify: "weight-like" names end with ".weight"; anything else is
        // a missing **activation** (i.e. a node that consumed an output which
        // was never produced by any node). Missing activations are a
        // different class of bug (dangling edges from pattern fusion).
        let (missing_weights, missing_activations): (Vec<_>, Vec<_>) =
            unbound.iter().partition(|s| s.split(" (node=").next().unwrap_or("").ends_with(".weight"));

        if !missing_weights.is_empty() {
            let sample = missing_weights.iter().take(5).map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
            blockers.push(Blocker {
                stage: "B",
                op_type: "(weight-binding)".into(),
                name: "bind_weight_shapes_fuzzy".into(),
                message: format!(
                    "{} unbound `.weight` inputs after fuzzy bind (sample: {})",
                    missing_weights.len(),
                    sample
                ),
                root_cause: "Mock provider 缺少某些 `.weight` 张量,或 weight_names::all_*_aliases \
                             对 Gemma 4 命名规则覆盖不全。",
                suggested_fix: "对每个 sample 在 weight_names.rs 中检查/补齐 aliases;如果 mock provider \
                                缺就在 MockGemma4Provider::new 里补上对应 shape。",
                priority: "High",
            });
        }
        if !missing_activations.is_empty() {
            let sample = missing_activations.iter().take(5).map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
            blockers.push(Blocker {
                stage: "B",
                op_type: "(dangling-activation)".into(),
                name: "FusedRMSLinearFusionPass / DCE".into(),
                message: format!(
                    "{} node inputs are neither graph-input, weight, nor produced by any node. \
                     Sample: {}",
                    missing_activations.len(),
                    sample
                ),
                root_cause:
                    "FusedRMSLinearFusionPass (pattern_fusion.rs:381) 用 2-slot 滑动窗口 \
                     [RMSNorm → Linear] 无条件融合, 不检查 RMSNorm 输出是否有其它消费者。\n\
                     Gemma 4 中 input_norm 的输出 layer_i_normed 同时被 q_proj/k_proj/v_proj 三个节点消费, \
                     融合 [input_norm + q_proj] 后产生器消失, k_proj/v_proj 的输入 \
                     layer_i_normed 成了悬空引用。DCE 不处理这种情况(k_proj/v_proj 仍然是 live ops)。",
                suggested_fix:
                    "在 FusedRMSLinearFusionPass::run 的 fuse_window 谓词里增加 fanout 检查: \
                     仅当 rms_norm.outputs[0] 在全图中只有 1 个消费者时才执行融合。\n\
                     实现参考: 先一次扫描全图建 Map<output_name, consumer_count>, \
                     fuse_window 谓词内查表 == 1 才返回 Some(fused)。\n\
                     否则上游 fusion (FusedQkvNormRope 修复后会先消费三路 MatMul) 会规避此冲突。",
                priority: "High",
            });
        }
    }

    // ── Stage C: JIT compile ──────────────────────────────────────────────
    // Build an executor directly (skip the fuzzy-bind inside from_graph by
    // using FusedGraphExecutor::new on our already-optimized+bound graph, then
    // calling compile() with representative seq_len/hidden).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    {
        let mut executor = gllm::graph::executor::FusedGraphExecutor::new(fused_with_weights);
        let seq_len = 4usize;
        let hidden = config.hidden_size;
        match executor.compile(seq_len, hidden, gllm_kernels::types::DType::F32) {
            Ok(_) => {
                println!("[T40 dry-run] Stage C ok — JIT compile succeeded");
            }
            Err(e) => {
                // Try to attribute the failure to a specific op/node by parsing
                // the ExecutionError message (CPU backend includes
                // "node '<name>' (op: <op_type>)" in the formatted string).
                let msg = format!("{e}");
                let (op_type, name) = parse_compile_err(&msg);
                blockers.push(Blocker {
                    stage: "C",
                    op_type,
                    name,
                    message: msg,
                    root_cause: UNKNOWN,
                    suggested_fix: "定位到该节点的 atomic_op_to_kind/CompilerGraph 构建或 \
                                    InferenceCompiler::compile_graph 的失败点。",
                    priority: "High",
                });
            }
        }
    }

    blockers
}

fn parse_compile_err(msg: &str) -> (String, String) {
    // Known format: "JIT compilation failed for node 'X' (op: Y): ..."
    let op_type = msg
        .split("(op:")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "(unknown)".to_string());
    let name = msg
        .split("node '")
        .nth(1)
        .and_then(|s| s.split('\'').next())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "(unknown)".to_string());
    (op_type, name)
}

fn write_report(blockers: &[Blocker]) -> std::io::Result<()> {
    use std::io::Write;
    let path = "/tmp/gemma4_dry_run_report.md";
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# Gemma 4 E2B Dry-Run Report")?;
    writeln!(f, "")?;
    if blockers.is_empty() {
        writeln!(f, "✅ **Pipeline clean** — stages A (YAML→OnnxGraph), B (optimize + weight bind), C (JIT compile) all passed.")?;
        writeln!(f, "")?;
        return Ok(());
    }

    writeln!(f, "❌ **{} blocker(s) detected.**", blockers.len())?;
    writeln!(f, "")?;

    // Group by stage.
    for stage in ["A", "B", "C"] {
        let items: Vec<&Blocker> = blockers.iter().filter(|b| b.stage == stage).collect();
        if items.is_empty() {
            continue;
        }
        let stage_name = match stage {
            "A" => "Stage A — YAML → OnnxGraph",
            "B" => "Stage B — GraphOptimizer.optimize / weight binding",
            "C" => "Stage C — FusedGraphExecutor.compile (JIT)",
            _ => "Unknown",
        };
        writeln!(f, "## {stage_name}")?;
        writeln!(f, "")?;
        for (i, b) in items.iter().enumerate() {
            writeln!(f, "### Blocker {stage}#{}: [{}] {}", i + 1, b.op_type, b.name)?;
            writeln!(f, "")?;
            writeln!(f, "- **Priority**: {}", b.priority)?;
            writeln!(f, "")?;
            writeln!(f, "**Message**:")?;
            writeln!(f, "")?;
            writeln!(f, "```")?;
            writeln!(f, "{}", b.message)?;
            writeln!(f, "```")?;
            writeln!(f, "")?;
            writeln!(f, "**Root cause**:")?;
            writeln!(f, "")?;
            writeln!(f, "{}", b.root_cause)?;
            writeln!(f, "")?;
            writeln!(f, "**Suggested fix**:")?;
            writeln!(f, "")?;
            writeln!(f, "{}", b.suggested_fix)?;
            writeln!(f, "")?;
        }
    }
    Ok(())
}

#[test]
fn t40_gemma4_e2b_full_pipeline_dry_run() {
    let blockers = run_dry_run();

    write_report(&blockers).expect("write /tmp/gemma4_dry_run_report.md");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("T40 Gemma 4 E2B Full-Pipeline Dry-Run Summary");
    println!("═══════════════════════════════════════════════════════════════");
    if blockers.is_empty() {
        println!("✅ All stages passed (A: YAML expand, B: optimize+bind, C: JIT compile).");
    } else {
        println!("❌ {} blocker(s):", blockers.len());
        for (i, b) in blockers.iter().enumerate() {
            println!(
                "  [{}] Stage {} {}/{}: {}",
                i + 1,
                b.stage,
                b.op_type,
                b.name,
                b.message.lines().next().unwrap_or("")
            );
        }
    }
    println!("\nFull report: /tmp/gemma4_dry_run_report.md");

    // Test is **informational** — do not fail CI on blockers. Purpose is to
    // produce the report for T40 blocker triage.
}
