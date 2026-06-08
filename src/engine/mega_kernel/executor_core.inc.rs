/// Mega-Kernel 执行器 (§9.1)
///
/// 唯一推理路径: 编译 → 单次 CALL。
/// 编译在模型加载时完成，推理时零 Rust 开销。
#[allow(dead_code)]
pub struct MegaKernelExecutor {
    /// Mega-kernel 编译产物 (唯一路径 — SPEC/39 统一架构)
    mega_compiled: Option<MegaKernelCompiled>,
    /// 模型配置
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    dtype: DType,
    /// EOS token ID — 从 ModelConfig 读取，传给 JIT 停止条件
    eos_token_id: u32,
    /// §19 KV-OPT-009: Mega-Kernel Variant 矩阵 (按 PrecisionTier 编译独立 Variant)
    variant_registry: crate::jit::variant_registry::VariantRegistry,
    /// Weight page JIT injection config (REQ-WP-008).
    pub weight_page_inject: WeightPageJitConfig,
    /// KV page decompress injection config (REQ-COMP11).
    pub decompress_inject: KvPageDecompressConfig,
    /// Weight page table for fault recovery (REQ-WP-009).
    weight_page_table: std::sync::Mutex<Option<crate::scheduler::fault_recovery::WeightPageTable>>,
    /// Fault recovery handler (REQ-WP-009).
    fault_handler: std::sync::Mutex<Option<crate::scheduler::fault_recovery::FaultRecoveryHandler>>,
}

// SAFETY: MegaKernelExecutor contains JIT-compiled function pointers and weight blobs
// that are thread-safe — the compiled code is immutable after construction and all
// mutable state is synchronized through interior mutability (Mutex/Atomic).
unsafe impl Send for MegaKernelExecutor {}
unsafe impl Sync for MegaKernelExecutor {}

impl MegaKernelExecutor {
    /// 从 auto-derived CompilerGraph 编译 mega-kernel（含 Argmax 或无 Argmax 的图均适用）。
    ///
    /// Graph 由 `auto_graph::build_compiler_graph()` 从 tensor names + shapes 生成。
    /// 所有模型 geometry 从 graph 自动派生，外部只需传不可派生的字段。
    pub fn compile_from_auto_graph(
        graph: gllm_kernels::compiler::graph::CompilerGraph,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        raw_floats: &std::collections::HashMap<String, crate::loader::RawFloatTensor>,
        name_map: &crate::loader::name_map::TensorNameMap,
        max_seq_len: usize,
        eos_token_id: u32,
        business_config: gllm_kernels::compiler::BusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
        gpu_sm_version: Option<u32>,
    ) -> Result<Self, MegaKernelError> {
        // Derive all geometry from graph — CompilerGraph is the SSOT.
        let geometry =
            gllm_kernels::compiler::graph_geometry::GraphDerivedGeometry::from_graph(&graph)
                .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        // Build slim config with only non-derivable fields.
        // SPEC/39: mtp_config 已迁移到图拓扑（OpKind::MtpDraft），不再从 BusinessConfig 读取。
        let mtp_depth = graph.ops.iter().find_map(|op| match &op.kind {
            gllm_kernels::compiler::graph::OpKind::MtpDraft { depth, .. } => Some(*depth),
            _ => None,
        }).unwrap_or(0);
        let config = gllm_kernels::compiler::mega_kernel_abi::CompileConfig {
            max_seq_len,
            business_config,
            hetero: hetero_config.clone(),
        };

        let hetero_layout = hetero_config.as_ref().map(|hc| {
            gllm_kernels::compiler::mega_kernel_abi::HeteroKernelWeightLayout::from_geometry_and_config(
                &geometry, hc,
            )
        });

        // Pre-resolve weight layout before moving graph into compiler.
        // Clone graph for GPU compilation (CPU compiler takes ownership).
        // Also save layer_loop_config for weight packing (needed even without GPU).
        let layer_loop_cfg = graph.layer_loop_config.clone();
        let hetero_loop_cfg = graph.hetero_layer_loop_config.clone();
        let graph_for_gpu = if gpu_sm_version.is_some() {
            Some(graph.clone())
        } else {
            None
        };
        let weight_layout = graph.weight_layout();
        let named_offsets: Vec<(String, usize)> = weight_layout
            .offsets
            .iter()
            .filter_map(|&(tid, offset)| {
                graph
                    .tensors
                    .get(tid.0 as usize)
                    .map(|t| (t.name.clone(), offset))
            })
            .collect();
        // Expand total_weight_bytes for layer loop: the layout only contains 1 copy of
        // per-layer weights (L0.*), but packing replicates to num_layers copies.
        let total_weight_bytes = if let Some(ref hcfg) = hetero_loop_cfg {
            // Hetero mode: total = base + sum of all segment strides + post-layer globals
            let templates_blob = hcfg.sliding_small_stride + hcfg.full_small_stride
                + hcfg.sliding_large_stride + hcfg.full_large_stride;
            let small_segs = hcfg.large_ffn_start_segment;
            let large_segs = hcfg.num_segments - small_segs;
            let total_layers_blob = small_segs * hcfg.small_segment_stride
                + large_segs * hcfg.large_segment_stride;
            let graph_globals_start = hcfg.layer_blob_base_offset + templates_blob;
            let globals_size = weight_layout.total_bytes.saturating_sub(graph_globals_start);
            let total = hcfg.layer_blob_base_offset + total_layers_blob + globals_size;
            eprintln!("[BLOB-LAYOUT] HETERO: num_segments={} small_segs={} large_segs={} total_layers_blob={}(0x{:x}) total={}(0x{:x})",
                hcfg.num_segments, small_segs, large_segs, total_layers_blob, total_layers_blob, total, total);
            total
        } else if let Some(ref llcfg) = layer_loop_cfg {
            let num_layers = llcfg.num_layers;
            let stride = llcfg.weight_stride;
            let base = llcfg.layer_blob_base_offset;
            // Globals after layer area: final_norm, logits-producer, etc.
            let globals_start = base + stride;
            let globals_size = weight_layout.total_bytes.saturating_sub(globals_start);
            eprintln!("[BLOB-LAYOUT] num_layers={} stride={}(0x{:x}) base_offset={}(0x{:x}) globals_start={}(0x{:x}) wl_total={}(0x{:x}) globals_size={}(0x{:x}) total_blob={}(0x{:x})",
                num_layers, stride, stride, base, base, globals_start, globals_start,
                weight_layout.total_bytes, weight_layout.total_bytes,
                globals_size, globals_size,
                base + num_layers * stride + globals_size, base + num_layers * stride + globals_size);
            for (i, (tid, off)) in weight_layout.offsets.iter().enumerate() {
                let is_global = *off >= globals_start;
                let region = if *off < base { "PRE-LAYER" } else if is_global { "POST-LAYER" } else { "LAYER" };
                eprintln!("[BLOB-LAYOUT]   wl[{}]: tid={} offset={}(0x{:x}) region={}", i, tid.0, off, off, region);
            }
            base + num_layers * stride + globals_size
        } else {
            weight_layout.total_bytes
        };
        // Default graph compiles with kv_load_mode=None (Direct).
        // KIVI4 variant compiles with kv_load_mode=Kivi4 for compressed KV load.
        let graph_kivi4 = {
            let mut g = graph.clone();
            g.kv_load_mode = Some(gllm_kernels::compiler::codegen::vm::instr::KvLoadMode::Kivi4);
            g
        };

        let hetero_layout_for_kivi4 = hetero_layout.clone();
        // SPEC/39: 从图拓扑推导 norm residual 约定，替代硬编码。
        // Gemma 1/2/3: has_embedding_scale=true, has_qk_norm=false → (1+weight) residual
        // Gemma 4+: has_qk_norm=true → standard RMSNorm, no residual
        // All other models: no embedding_scale → no residual
        let has_gemma_norm_residual = graph.embedding_scale.is_some()
            && !graph.ops.iter().any(|op| matches!(op.kind, gllm_kernels::compiler::graph::OpKind::QkNorm { .. }));
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler
            .compile_mega_kernel_from_graph(graph, &config, hetero_layout)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let exec_code = output.layer_code;
        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };
        {
            let dump_path = "/tmp/jit_code_mega_auto.bin";
            let _ = std::fs::write(dump_path, exec_code.code_bytes());
            eprintln!("[JIT] Dumped {} bytes to {} entry_fn={:p}", exec_code.code_bytes().len(), dump_path, entry_fn);
        }

        // §19 KV-OPT-009: Compile KIVI4 variant for compressed KV attention.
        // TEMP: disable KIVI4 compilation for GGUF models (no raw_floats) —
        // RegAllocator on 28-layer N-layer takes 15min per compilation.
        // Will re-enable after RegAllocator optimization.
        let kivi4_exec = if !raw_floats.is_empty() {
            compiler
                .compile_mega_kernel_from_graph(graph_kivi4, &config, hetero_layout_for_kivi4)
                .ok()
        } else {
            None
        };

        // Pack weights using pre-resolved named offsets — unified with 无 Argmax 的图路径。
        let weight_blob = pack_weights_from_graph(
            &named_offsets,
            total_weight_bytes,
            weight_ptrs,
            weight_sizes,
            raw_floats,
            name_map,
            layer_loop_cfg.as_ref(),
            hetero_loop_cfg.as_ref(),
            geometry.num_layers,
            has_gemma_norm_residual,
        );

        // Verify Q4_0 embed data in packed blob at offset 0
        {
            // Read first block of embed table (should be Q4_0: 2B f16 scale + 16B packed nibbles)
            if weight_blob.len() >= 18 {
                let scale_bits = u16::from_le_bytes([weight_blob[0], weight_blob[1]]);
                let scale_f32 = half::f16::from_bits(scale_bits).to_f32();
                let first_nibbles: Vec<String> = (0..8).map(|i| {
                    let byte = weight_blob[2 + i];
                    let lo = byte & 0x0F;
                    let hi = (byte >> 4) & 0x0F;
                    format!("{}({})", lo as i32 - 8, hi as i32 - 8)
                }).collect();
                eprintln!("[Q4-VERIFY] embed@0: scale_f16={:.6} (bits=0x{:04x}) nibbles=[{}]",
                    scale_f32, scale_bits, first_nibbles.join(", "));
                // Also check token 785's row
                let tid = 785usize;
                let row_stride = 576usize; // 32 blocks * 18 bytes
                let row_off = tid * row_stride;
                if row_off + 18 <= weight_blob.len() {
                    let t_scale_bits = u16::from_le_bytes([weight_blob[row_off], weight_blob[row_off + 1]]);
                    let t_scale = half::f16::from_bits(t_scale_bits).to_f32();
                    let t_nibbles: Vec<String> = (0..4).map(|i| {
                        let byte = weight_blob[row_off + 2 + i];
                        format!("lo={}(f={:.4}) hi={}(f={:.4})",
                            byte as i32 & 0xF, ((byte & 0xF) as f32 - 8.0) * t_scale,
                            (byte >> 4) as i32 & 0xF, (((byte >> 4) & 0xF) as f32 - 8.0) * t_scale)
                    }).collect();
                    eprintln!("[Q4-VERIFY] token_{}@{}: scale={:.6} nibbles=[{}]",
                        tid, row_off, t_scale, t_nibbles.join(", "));
                }
            }
        }

        // Verify embed data in packed blob (first hidden F32 values = embed row 0)
        {
            // Tied weight verify: find the global (non-layer) output projection weight.
            // SPEC/39: no string-based "lm_head" search — identify by being the only
            // non-"L0." weight that is large enough to be an output projection.
            let lm_goff = named_offsets.iter()
                .find(|(n, off)| !n.starts_with("L0.") && !n.contains("embed") && !n.contains("final_norm") && *off > 0)
                .map(|(_, off)| *off);
            if let Some(lm_goff) = lm_goff {
                // logits-producer is a global weight packed at its graph offset — no layer
                // loop stride adjustment needed (that applies to per-layer weights only).
                let lm_blob_off = lm_goff;
                if lm_blob_off + 32 <= weight_blob.len() {
                    let embed_f32 = unsafe {
                        std::slice::from_raw_parts(weight_blob.as_ptr() as *const f32, 8)
                    };
                    let lm_f32 = unsafe {
                        std::slice::from_raw_parts(weight_blob[lm_blob_off..].as_ptr() as *const f32, 8)
                    };
                    let embed_vals: Vec<String> = embed_f32.iter().take(4).map(|v| format!("{:.4}", v)).collect();
                    let lm_vals: Vec<String> = lm_f32.iter().take(4).map(|v| format!("{:.4}", v)).collect();
                    let match_count = embed_f32.iter().zip(lm_f32.iter()).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
                    eprintln!("[TIED-WEIGHT] embed@0=[{}] lm_head@{}(graph={},adj={} layer_loop)=[{}] match={}/8",
                        embed_vals.join(","), lm_blob_off, lm_goff,
                        if layer_loop_cfg.is_some() { "with" } else { "no" },
                        lm_vals.join(","), match_count);
                }
            }
            let embed_f32 = unsafe {
                std::slice::from_raw_parts(
                    weight_blob.as_ptr() as *const f32,
                    8.min(weight_blob.len() / 4),
                )
            };
            let first_8: Vec<String> = embed_f32.iter().take(8).map(|v| format!("{:.4}", v)).collect();
            let any_nan = embed_f32.iter().any(|&v| v.is_nan());
            eprintln!("[PACK-VERIFY] blob[0..8] as f32: [{}] any_nan={}", first_8.join(","), any_nan);
        }
        // Verify packed weight data at correct blob offsets
        if let Some(ref llcfg) = layer_loop_cfg {
            // L0.input_norm: graph offset = layer_blob_base_offset, blob offset = 0 * stride + graph_offset
            let norm_graph_off = named_offsets.iter()
                .find(|(n, _)| n == "L0.input_norm")
                .map(|(_, off)| *off)
                .unwrap_or(llcfg.layer_blob_base_offset);
            if norm_graph_off + 32 <= weight_blob.len() {
                let norm_f32 = unsafe {
                    std::slice::from_raw_parts(
                        weight_blob[norm_graph_off..].as_ptr() as *const f32,
                        8,
                    )
                };
                let first_8: Vec<String> = norm_f32.iter().take(8).map(|v| format!("{:.4}", v)).collect();
                let any_nan = norm_f32.iter().any(|&v| v.is_nan());
                eprintln!("[PACK-VERIFY] L0.input_norm at offset {}: [{}] any_nan={}", norm_graph_off, first_8.join(","), any_nan);
            }
            // L0.q_proj: graph offset from named_offsets
            let q_proj_graph_off = named_offsets.iter()
                .find(|(n, _)| n == "L0.q_proj")
                .map(|(_, off)| *off)
                .unwrap_or(0);
            if q_proj_graph_off + 36 <= weight_blob.len() {
                let q_f32 = unsafe {
                    std::slice::from_raw_parts(
                        weight_blob[q_proj_graph_off..].as_ptr() as *const f32,
                        8.min((weight_blob.len() - q_proj_graph_off) / 4),
                    )
                };
                let first_8: Vec<String> = q_f32.iter().take(8).map(|v| format!("{:.4}", v)).collect();
                let any_nan = q_f32.iter().any(|&v| v.is_nan());
                eprintln!("[PACK-VERIFY] L0.q_proj at offset {}: first_8_f32=[{}] any_nan={}",
                    q_proj_graph_off, first_8.join(","), any_nan);
                let raw = &weight_blob[q_proj_graph_off..q_proj_graph_off + 36];
                let scale_bits = u16::from_le_bytes([raw[0], raw[1]]);
                let scale_f16 = half::f16::from_bits(scale_bits);
                let scale_f32 = scale_f16.to_f32();
                let first_8_int8: Vec<i8> = raw[2..10].iter().map(|&b| b as i8).collect();
                eprintln!("[PACK-VERIFY] L0.q_proj Q8_0 header: scale_f16={:.6} first_8_int8={:?}",
                    scale_f32, first_8_int8);
            }
            // L1.input_norm: blob offset = 1 * stride + graph_offset
            let l1_norm_off = llcfg.weight_stride + norm_graph_off;
            if l1_norm_off + 32 <= weight_blob.len() {
                let l1_f32 = unsafe {
                    std::slice::from_raw_parts(weight_blob[l1_norm_off..].as_ptr() as *const f32, 8)
                };
                let first_8: Vec<String> = l1_f32.iter().take(8).map(|v| format!("{:.4}", v)).collect();
                eprintln!("[PACK-VERIFY] L1.input_norm at offset {}: [{}]", l1_norm_off, first_8.join(","));
            }
            let l1_q_off = llcfg.weight_stride + q_proj_graph_off;
            if l1_q_off + 32 <= weight_blob.len() {
                let l1_q = unsafe {
                    std::slice::from_raw_parts(weight_blob[l1_q_off..].as_ptr() as *const f32, 8)
                };
                let first_8: Vec<String> = l1_q.iter().take(8).map(|v| format!("{:.4}", v)).collect();
                eprintln!("[PACK-VERIFY] L1.q_proj at offset {}: [{}]", l1_q_off, first_8.join(","));
            }
        }
        // Verify norm weight values at actual hetero-computed offsets
        {
            if let Some(hcfg) = hetero_loop_cfg.as_ref() {
                eprintln!("[NORM-WEIGHT-DIAG] blob_size={} layer_blob_base_offset={}", weight_blob.len(), hcfg.layer_blob_base_offset);
            }
            for &norm_name in &["L0.input_norm", "L0.q_norm", "L0.k_norm", "L0.post_attn_norm"] {
                let norm_off = named_offsets.iter()
                    .find(|(n, _)| n == &norm_name)
                    .map(|(_, off)| *off);
                if let Some(off) = norm_off {
                    // Read from raw named_offsets offset first (template region)
                    if off + 32 <= weight_blob.len() {
                        let f32_vals = unsafe {
                            std::slice::from_raw_parts(weight_blob[off..].as_ptr() as *const f32, 8)
                        };
                        let first_8: Vec<String> = f32_vals.iter().take(8).map(|v| format!("{:.6}", v)).collect();
                        let mean: f32 = f32_vals.iter().sum::<f32>() / f32_vals.len() as f32;
                        eprintln!("[NORM-WEIGHT-TEMPLATE] {} at template_off={}: [{}] mean={:.4}", norm_name, off, first_8.join(","), mean);
                    }
                    // Also try reading at hetero layer 0 absolute offset
                    if let Some(hcfg) = hetero_loop_cfg.as_ref() {
                        let type_base = hcfg.layer_blob_base_offset;
                        let abs_off = type_base + off;
                        if abs_off + 32 <= weight_blob.len() {
                            let f32_vals = unsafe {
                                std::slice::from_raw_parts(weight_blob[abs_off..].as_ptr() as *const f32, 8)
                            };
                            let first_8: Vec<String> = f32_vals.iter().take(8).map(|v| format!("{:.6}", v)).collect();
                            let mean: f32 = f32_vals.iter().sum::<f32>() / f32_vals.len() as f32;
                            eprintln!("[NORM-WEIGHT-HETERO] {} at base+off={}: [{}] mean={:.4}", norm_name, abs_off, first_8.join(","), mean);
                        }
                    }
                }
            }
        }

        // Compile GPU PTX/HIP code if sm_version provided and GPU JIT backend available.
        #[cfg(any(feature = "cuda", feature = "hip"))]
        let gpu_code = match (graph_for_gpu, gpu_sm_version) {
            (Some(g), Some(sm)) => match compiler.compile_mega_kernel_to_gpu(g, &config, sm) {
                Ok(gpu_output) => {
                    log::info!(
                        "[mega] GPU PTX compiled: {} bytes, {} layers",
                        gpu_output.gpu_code.len(),
                        gpu_output.num_layers,
                    );
                    Some(gpu_output.gpu_code)
                }
                Err(e) => {
                    log::warn!("[mega] GPU compilation failed (GPU path unavailable): {e}");
                    None
                }
            },
            _ => None,
        };
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        let gpu_code = {
            let _ = (graph_for_gpu, gpu_sm_version);
            None
        };

        let mtp_depth_extracted = mtp_depth;
        let mega_compiled = MegaKernelCompiled {
            weight_layout,
            named_offsets,
            buffer_layout: output.buffer_layout,
            logits_scratch_offset: output.logits_scratch_offset,
            weight_blob,
            exec_code,
            entry_fn,
            rope_cache: output.rope_cache,
            scratchpad_base_bytes: output.logits_scratch_offset,
            vocab_size: output.vocab_size,
            hidden: output.hidden,
            elem_bytes: { let eb = geometry.compute_dtype.size_bytes();
                eprintln!("[DIAG-EXEC-DTYPE] compute_dtype={:?} elem_bytes={} num_kv_heads={} head_dim={} kv_row_stride={}",
                    geometry.compute_dtype, eb, geometry.num_kv_heads, geometry.head_dim,
                    geometry.num_kv_heads * geometry.head_dim * eb);
                eb
            },

            gpu_code,
            source_map: output.source_map,
            num_kv_heads: geometry.num_kv_heads,
            head_dim: geometry.head_dim,
            max_seq_len,
            mtp_depth: mtp_depth_extracted,
        };

        Ok(Self {
            mega_compiled: Some(mega_compiled),
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden,
            vocab_size: geometry.vocab_size,
            dtype: geometry.compute_dtype,
            eos_token_id,
            variant_registry: {
                let mut registry = crate::jit::variant_registry::VariantRegistry::new();
                // Register default Direct variant (the compiled mega-kernel itself)
                let default_key = crate::jit::variant_registry::VariantRegistry::derive_key(
                    "default", false, false, None, false, 64, None, None, None,
                );
                let default_variant = crate::jit::variant_registry::CompiledVariant {
                    code: Vec::new(),                  // The actual code lives in mega_compiled
                    instruction_footprint_bytes: 8192, // estimated
                    mechanisms: vec![crate::jit::variant_registry::MechanismId::Dense],
                    section: crate::jit::variant_registry::CodeSection::Hot,
                    key: default_key.clone(),
                };
                let _ = registry.register(default_variant);

                // §19 KV-OPT-009: Register KIVI4 variant (compiled with Kivi4 kv_load_mode).
                if let Some(ref kivi4) = kivi4_exec {
                    let kivi4_key = crate::jit::variant_registry::VariantRegistry::derive_key(
                        "default", false, false, None, false, 64, None, Some("KIVI4".to_string()), None,
                    );
                    let kivi4_variant = crate::jit::variant_registry::CompiledVariant {
                        code: kivi4.layer_code.code_bytes().to_vec(),
                        instruction_footprint_bytes: kivi4.layer_code.code_bytes().len(),
                        mechanisms: vec![crate::jit::variant_registry::MechanismId::KiviQuant],
                        section: crate::jit::variant_registry::CodeSection::Hot,
                        key: kivi4_key,
                    };
                    let _ = registry.register(kivi4_variant);
                    log::info!("[mega] KV-OPT-009: KIVI4 variant compiled ({} bytes)",
                        kivi4.layer_code.code_bytes().len());
                }

                registry
            },
            weight_page_inject: WeightPageJitConfig::default(),
            decompress_inject: KvPageDecompressConfig::default(),
            weight_page_table: std::sync::Mutex::new(None),
            fault_handler: std::sync::Mutex::new(None),
        })
    }

    /// Returns total scratchpad bytes needed for execution.
    pub fn total_scratchpad_bytes(&self) -> usize {
        self.mega_compiled
            .as_ref()
            .map(|m| m.scratchpad_base_bytes)
            .unwrap_or(0)
    }

    /// Set weight page table and fault handler for explicit fault recovery (REQ-WP-009).
    pub fn set_weight_page_table(
        &self,
        table: crate::scheduler::fault_recovery::WeightPageTable,
        handler: crate::scheduler::fault_recovery::FaultRecoveryHandler,
    ) {
        *self.weight_page_table.lock().unwrap() = Some(table);
        *self.fault_handler.lock().unwrap() = Some(handler);
    }

    /// REQ-WP-009: Ensure all weight pages for the current step are in GpuHbm (Tier::L1).
    ///
    /// Fast path: no weight page table registered → zero overhead (just a Mutex lock + None check).
    /// Slow path: iterates layers 0..num_layers, checks each page's tier via `page_tier()`,
    /// and calls `recover_fault()` for any page not in L1.
    fn ensure_weight_pages_resident(&self) {
        use crate::scheduler::fault_recovery::PageFault;
        use crate::scheduler::memory_manager::{GlobalMemoryManager, Tier};

        // Phase 1: Collect faults (immutable borrow of table)
        let faults: Vec<PageFault> = {
            let table_lock = self.weight_page_table.lock().unwrap();
            let Some(table) = table_lock.as_ref() else { return };
            let mut faults = Vec::new();
            for layer_idx in 0..self.num_layers {
                let Some(pages) = table.get_layer_pages(layer_idx) else { continue };
                for &pid in pages {
                    let tier = table.page_tier(pid).unwrap_or(Tier::L3);
                    if tier != Tier::L1 {
                        faults.push(PageFault {
                            page_id: pid,
                            current_tier: tier,
                            target_tier: Tier::L1,
                            fault_time: std::time::Instant::now(),
                            expert_key: None,
                            dense_layer_idx: Some(layer_idx),
                        });
                    }
                }
            }
            faults
        };

        // Fast path: all pages already in L1
        if faults.is_empty() {
            return;
        }

        // Phase 2: Recover faults (mutable borrow of table + handler)
        let mut table_lock = self.weight_page_table.lock().unwrap();
        let table = table_lock.as_mut().unwrap();
        let mut handler_lock = self.fault_handler.lock().unwrap();
        let handler = handler_lock.as_mut().unwrap();

        for fault in faults {
            let mut gmm = GlobalMemoryManager::new_with_capacities(0, 0, 0);
            if let Err(e) = handler.recover_fault(&fault, &mut gmm, table) {
                log::warn!("[WP-009] fault recovery failed for page {:?}: {}", fault.page_id, e);
            }
        }
    }

    /// 单序列 mega-kernel 生成。
    ///
    /// ARCH-RUST-IS-CODEGEN: 一次 CALL 完成。
    /// JIT mega-kernel 内部执行完整的 generate loop:
    ///   LoopBegin → embed → N 层 → logits-producer → Argmax → StoreToken → CheckStopCondition → LoopEnd
    /// Rust 只做：(1) 准备输入 (2) 预填 RoPE 表 (3) 一次 CALL (4) 读 output_tokens
    pub fn generate_single_sequence(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
    ) -> Result<Vec<u32>, MegaKernelError> {
        // REQ-KV-OPT-010: no position-agnostic range when called without it.
        self.generate_single_sequence_inner(
            prompt_tokens,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            hook_ctx_ptr,
            callback_table_ptr,
            page_table,
            pool_base,
            session_position,
            fused_hidden,
            num_mm_tokens,
            None,
        )
    }

    /// Inner implementation with optional position-agnostic range.
    ///
    /// When `position_agnostic_range` is `Some((start, end))`, RoPE is set to
    /// identity (cos=1, sin=0) for positions `[start, end)`. This implements
    /// REQ-KV-OPT-010 CacheSlide: system prompt pages skip position encoding
    /// to enable cross-request KV reuse.
    fn generate_single_sequence_inner(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
        position_agnostic_range: Option<(usize, usize)>,
    ) -> Result<Vec<u32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let prompt_len = prompt_tokens.len();
        let max_total = prompt_len + max_new_tokens;
        let mtp_depth = mega.mtp_depth;

        let mut input_ids = vec![0u32; max_total];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..max_total as u32).collect();
        // Output buffer: [0..max_new_tokens) = main tokens,
        // [max_new_tokens..max_new_tokens + max_new_tokens * mtp_depth) = MTP candidates.
        let output_size = max_new_tokens * (1 + mtp_depth);
        let mut output_tokens = vec![0u32; output_size];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(max_total)];
        eprintln!("[SCRATCHPAD-DBG] max_total={} scratchpad_bytes={} buffer_layout.total={}",
            max_total, scratchpad.len(), mega.buffer_layout.total_scratchpad_bytes);
        eprintln!("[SCRATCHPAD-DBG] logits_off={} vocab={} hidden={} max_seq_len={}",
            mega.logits_scratch_offset, mega.vocab_size, mega.hidden, mega.max_seq_len);
        eprintln!("[SCRATCHPAD-DBG] activation_a_off={} activation_b_off={} activation_bytes={}",
            mega.buffer_layout.activation_a_offset, mega.buffer_layout.activation_b_offset,
            mega.buffer_layout.activation_bytes);
        eprintln!("[SCRATCHPAD-DBG] scratchpad_ptr={:p} scratchpad_end={:p}",
            scratchpad.as_ptr(), unsafe { scratchpad.as_ptr().add(scratchpad.len()) });

        // Pre-fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
            // Primary cache
            let rope_elems = max_total * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions[..max_total],
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling,
                );
            }
            // Secondary cache (for heterogeneous models with 2 head_dim values)
            if let Some(ref sec) = rc.secondary_cache {
                let sec_elems = max_total * sec.head_dim;
                let sec_bytes = sec_elems * 4;
                if sec.cache_offset + sec_bytes <= scratchpad.len() {
                    let sec_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                            sec_elems,
                        )
                    };
                    gllm_kernels::compiler::fill_cos_sin_table_partial(
                        sec_slice,
                        &positions[..max_total],
                        sec.head_dim,
                        sec.theta,
                        sec.partial,
                        sec.rope_scaling,
                    );
                }
            }
        }

        // REQ-KV-OPT-010: Position-agnostic RoPE (CacheSlide).
        // For system prompt pages marked as position-agnostic, zero the rotation
        // by setting cos=1 and sin=0 for those positions in the RoPE cache.
        // This allows cross-request KV reuse without position encoding mismatch.
        if let Some((start, end)) = position_agnostic_range {
            let end = end.min(max_total);
            if start < end {
                if let Some(ref rc) = mega.rope_cache {
                    let half = rc.head_dim / 2;
                    // Primary cache
                    let rope_elems = max_total * rc.head_dim;
                    if rc.cache_offset + rope_elems * 4 <= scratchpad.len() {
                        let rope_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                                rope_elems,
                            )
                        };
                        for pos in start..end {
                            let row_start = pos * rc.head_dim;
                            // cos half: set to 1.0 (identity rotation)
                            for i in 0..half {
                                rope_slice[row_start + i] = 1.0;
                            }
                            // sin half: set to 0.0 (no rotation)
                            for i in half..rc.head_dim {
                                rope_slice[row_start + i] = 0.0;
                            }
                        }
                    }
                    // Secondary cache (heterogeneous models)
                    if let Some(ref sec) = rc.secondary_cache {
                        let sec_half = sec.head_dim / 2;
                        let sec_elems = max_total * sec.head_dim;
                        if sec.cache_offset + sec_elems * 4 <= scratchpad.len() {
                            let sec_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                                    sec_elems,
                                )
                            };
                            for pos in start..end {
                                let row_start = pos * sec.head_dim;
                                for i in 0..sec_half {
                                    sec_slice[row_start + i] = 1.0;
                                }
                                for i in sec_half..sec.head_dim {
                                    sec_slice[row_start + i] = 0.0;
                                }
                            }
                        }
                    }
                }
                log::debug!(
                    "[mega] REQ-KV-OPT-010: position-agnostic RoPE for positions [{}, {})",
                    start,
                    end,
                );
            }
        }

        let page_table_ptr = page_table.map_or(std::ptr::null(), |pt| pt.as_ptr());
        // Allocate KV cache buffer for contiguous KV attention.
        // The mega-kernel writes K/V data here after each layer's GEMM and reads
        // from it during attention, enabling the model to attend to all previous tokens.
        let kv_cache_bytes = mega.kv_cache_bytes(self.num_layers);
        let mut kv_cache = if kv_cache_bytes > 0 {
            eprintln!("[KV-CACHE] allocating {} bytes ({} layers, stride {} per layer)",
                kv_cache_bytes, self.num_layers, mega.kv_layer_stride());
            vec![0u8; kv_cache_bytes]
        } else {
            Vec::new()
        };
        let effective_pool_base = if !kv_cache.is_empty() {
            kv_cache.as_mut_ptr() as *const u8
        } else {
            pool_base
        };

        // REQ-WP-009: Verify weight pages are in GpuHbm (L1) before CALL.
        // Fast path: no weight page table registered → zero overhead.
        self.ensure_weight_pages_resident();

        let generated_count = unsafe {
            // R1: Build KernelContext for single-pointer ABI transition.
            // Parameters are organized into the flat struct; legacy ABI args
            // are extracted from it at the call site.
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.kv_cache_ptr = effective_pool_base as *mut u8;
            ctx.hook_ctx_ptr = hook_ctx_ptr as *mut u8;
            ctx.callback_table_ptr = callback_table_ptr as *const u64;
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();
            ctx.batch_ctx_ptr = std::ptr::null();
            ctx.telemetry_ptr = std::ptr::null_mut();

            // REQ-COMP11: Wire KV page decompress injection.
            // When decompress_inject is enabled, the JIT reads KvPageHeader.codec
            // via kv_page_header_ptr before each KV page access. If codec != None,
            // the JIT invokes the corresponding decompress callback (Lz4/BitPackRle/Nvcomp)
            // registered in the callback table before reading the page data.
            if self.decompress_inject.enabled {
                ctx.decompress_inject_flags = 1; // bit 0 = enabled
            }

            // Save MXCSR before JIT call — JIT may modify FP exception masks
            let mut mxcsr_saved: u32 = 0;
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr_saved, options(nostack));
            let result = (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                ctx.kv_cache_ptr,
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                temperature.to_bits() as usize,
                top_k,
                top_p.to_bits() as usize,
                max_new_tokens,
                self.eos_token_id as usize,
                0,            // output_mode_selector: Generate
                ctx.hook_ctx_ptr as *const u8,
                ctx.telemetry_ptr,
                session_position, // session_position (0=new, >0=resume)
                fused_hidden.map_or(std::ptr::null(), |fh| fh.as_ptr() as *const u8), // fused_hidden_ptr
                num_mm_tokens,      // num_mm_tokens
                ctx.callback_table_ptr as *const u8,
                page_table_ptr,     // page_table_ptr: NULL = contiguous KV, u32[] = paged KV
                ctx.batch_ctx_ptr,
            );
            // Restore MXCSR: reset all FP exception flags and set default masks
            // Default MXCSR = 0x1F80 (all exceptions masked, round-to-nearest, no flush-to-zero)
            std::arch::asm!("ldmxcsr [{}]", in(reg) &0x1F80u32, options(nostack));
            let _ = mxcsr_saved; // used for save/restore pattern, suppress unused warning
            result
        };

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={} output_first={}",
            prompt_len,
            max_new_tokens,
            generated_count,
            self.eos_token_id,
            output_tokens.first().copied().unwrap_or(0),
        );
        // DIAG: dump output tokens, logits at pos 0, activation buffers
        {
            let n_dump = 20.min(output_tokens.len());
            eprintln!("[DIAG-OUTPUT] output_tokens[0..{}]: {:?}", n_dump, &output_tokens[..n_dump]);
            eprintln!("[DIAG-INPUT] prompt_len={} input_ids={:?}", prompt_len, &input_ids[..prompt_len.min(20)]);
            // Logits at pos 0 (after decode, this is the last decode step's logits)
            let logits_off = mega.logits_scratch_offset;
            let vocab = mega.vocab_size;
            for &check_pos in &[0usize, prompt_len.saturating_sub(1)] {
                let byte_off = logits_off + check_pos * vocab * 4;
                if byte_off + vocab * 4 <= scratchpad.len() {
                    let full = unsafe { std::slice::from_raw_parts(scratchpad[byte_off..].as_ptr() as *const f32, vocab) };
                    let max_val = full.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min_val = full.iter().cloned().fold(f32::INFINITY, f32::min);
                    let nan_n = full.iter().filter(|x| x.is_nan()).count();
                    eprintln!("[DIAG-LOGITS] pos={} nan={} min={:.4} max={:.4} first8={:?}",
                        check_pos, nan_n, min_val, max_val, &full[..8.min(vocab)]);
                    if max_val > f32::NEG_INFINITY {
                        let mut idx: Vec<(usize, f32)> = full.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                        idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        eprintln!("[DIAG-TOP5] pos={} {:?}", check_pos, &idx[..5]);
                    }
                }
            }
            // Activation buffers at pos 0
            let hidden = mega.hidden;
            let row_bytes = hidden * 4;
            for (label, off) in [("act_a", mega.buffer_layout.activation_a_offset), ("act_b", mega.buffer_layout.activation_b_offset)] {
                if off + row_bytes <= scratchpad.len() {
                    let v = unsafe { std::slice::from_raw_parts(scratchpad[off..].as_ptr() as *const f32, 8.min(hidden)) };
                    eprintln!("[DIAG-ACT-POS0] {} {:?}", label, v);
                }
            }
            // KV cache dump for layer 0 (K and V at positions 0 and prompt_len-1)
            if !kv_cache.is_empty() {
                let kv_ptr = kv_cache.as_ptr();
                let kv_len = kv_cache.len();
                let max_seq = mega.max_seq_len;
                let kv_row_stride = mega.num_kv_heads * mega.head_dim * 4;
                let kv_layer_stride = 2 * max_seq * kv_row_stride;
                let v_offset_in_layer = max_seq * kv_row_stride;
                for &pos in &[0usize, prompt_len.saturating_sub(1)] {
                    let k_off = pos * kv_row_stride;
                    let v_off = v_offset_in_layer + pos * kv_row_stride;
                    let n = 8.min(kv_row_stride / 4);
                    if k_off + n * 4 <= kv_len {
                        let k_vals = unsafe { std::slice::from_raw_parts(kv_ptr.add(k_off) as *const f32, n) };
                        eprintln!("[DIAG-KV] layer=0 pos={} k={:?} all_zero={}", pos, &k_vals[..n], k_vals.iter().all(|&x| x == 0.0));
                    }
                    if v_off + n * 4 <= kv_len {
                        let v_vals = unsafe { std::slice::from_raw_parts(kv_ptr.add(v_off) as *const f32, n) };
                        eprintln!("[DIAG-KV] layer=0 pos={} v={:?} all_zero={}", pos, &v_vals[..n], v_vals.iter().all(|&x| x == 0.0));
                    }
                }
                for &layer_idx in &[1usize, 4] {
                    let n = 4.min(kv_row_stride / 4);
                    let k_off_0 = layer_idx * kv_layer_stride;
                    let k_off_last = layer_idx * kv_layer_stride + (prompt_len - 1) * kv_row_stride;
                    if k_off_0 + n * 4 <= kv_len {
                        let k_vals = unsafe { std::slice::from_raw_parts(kv_ptr.add(k_off_0) as *const f32, n) };
                        eprintln!("[DIAG-KV] layer={} pos=0 k={:?}", layer_idx, k_vals);
                    }
                    if k_off_last + n * 4 <= kv_len {
                        let k_vals = unsafe { std::slice::from_raw_parts(kv_ptr.add(k_off_last) as *const f32, n) };
                        eprintln!("[DIAG-KV] layer={} pos={} k={:?}", layer_idx, prompt_len - 1, k_vals);
                    }
                }
            }
        }
        let actual_count = if generated_count == 0 && max_new_tokens > 0 && output_tokens[0] != 0 {
            1
        } else {
            generated_count
        };
        // Build output: main tokens followed by MTP candidate tokens (if enabled).
        // MTP candidates layout: output_tokens[max_new_tokens + step * mtp_depth + k]
        let mut result = Vec::with_capacity(actual_count * (1 + mtp_depth));
        result.extend_from_slice(&output_tokens[..actual_count]);
        if mtp_depth > 0 && actual_count > 0 {
            for step in 0..actual_count {
                let mtp_base = max_new_tokens + step * mtp_depth;
                for k in 0..mtp_depth {
                    if mtp_base + k < output_tokens.len() {
                        result.push(output_tokens[mtp_base + k]);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Like `generate_single_sequence`, but with position-agnostic range support.
    ///
    /// REQ-KV-OPT-010: When system prompt pages are marked position-agnostic,
    /// the RoPE cache is set to identity (cos=1, sin=0) for positions `[agnostic_start, agnostic_end)`.
    /// This enables CacheSlide cross-request KV reuse.
    pub fn generate_single_sequence_with_position_agnostic(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
        agnostic_start: usize,
        agnostic_end: usize,
    ) -> Result<Vec<u32>, MegaKernelError> {
        self.generate_single_sequence_inner(
            prompt_tokens,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            hook_ctx_ptr,
            callback_table_ptr,
            page_table,
            pool_base,
            session_position,
            fused_hidden,
            num_mm_tokens,
            Some((agnostic_start, agnostic_end)),
        )
    }

    /// SPEC/20 REQ-BCI-003/008: Batch inference via single mega-kernel CALL.
    ///
    /// One CALL: prefill (M=total_prefill_tokens) → per-seq argmax →
    /// decode step loop (M=num_active per step) → all sequences complete.
    /// Sampling params read from batch_ctx.sampling_params_ptr per-seq.
    ///
    /// Returns total decode steps completed across all sequences.
    pub fn generate_batch(
        &self,
        batch_ctx: &super::batch_context::BatchContext,
        input_ids_flat: &[u32],
        positions_flat: &[u32],
        total_prefill_tokens: usize,
        max_decode_steps: usize,
        pool_base: *const u8,
    ) -> Result<usize, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        // Scratchpad must hold both prefill + decode (max_decode_steps × num_seqs tokens)
        let num_seqs = batch_ctx.num_seqs;
        let max_decode_tokens = max_decode_steps.max(1) * num_seqs;
        let max_total = total_prefill_tokens + max_decode_tokens;
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(max_total)];

        // Fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = max_total * rc.head_dim;
            if rc.cache_offset + rope_elems * 4 <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                let positions: Vec<u32> = (0..max_total as u32).collect();
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling,
                );
            }
            if let Some(ref sec) = rc.secondary_cache {
                let sec_elems = max_total * sec.head_dim;
                if sec.cache_offset + sec_elems * 4 <= scratchpad.len() {
                    let sec_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                            sec_elems,
                        )
                    };
                    let positions: Vec<u32> = (0..max_total as u32).collect();
                    gllm_kernels::compiler::fill_cos_sin_table_partial(
                        sec_slice,
                        &positions,
                        sec.head_dim,
                        sec.theta,
                        sec.partial,
                        sec.rope_scaling,
                    );
                }
            }
        }

        // Output tokens buffer sized for all sequences (prompt + decode)
        let mut output_tokens = vec![0u32; max_total];

        // Allocate KV cache buffer
        let kv_cache_bytes = mega.kv_cache_bytes(self.num_layers);
        let mut kv_cache = if kv_cache_bytes > 0 {
            vec![0u8; kv_cache_bytes]
        } else {
            Vec::new()
        };
        let effective_pool_base = if !kv_cache.is_empty() {
            kv_cache.as_mut_ptr() as *const u8
        } else {
            pool_base
        };

        let generated_count = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.kv_cache_ptr = effective_pool_base as *mut u8;
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();
            ctx.batch_ctx_ptr = batch_ctx.as_ptr();

            (mega.entry_fn)(
                input_ids_flat.as_ptr(),
                ctx.weight_blob_ptr,
                ctx.kv_cache_ptr,
                positions_flat.as_ptr(),
                std::ptr::null(),               // aux
                1,                              // batch_size (forward pass dimension)
                total_prefill_tokens,           // prompt_len (forward dimension)
                ctx.scratch_buffer_ptr,         // scratchpad
                output_tokens.as_mut_ptr(),     // output
                0,                              // temperature (batch mode: read from sampling_params_ptr per-seq)
                0,                              // top_k (batch mode: read from sampling_params_ptr per-seq)
                0,                              // top_p (batch mode: read from sampling_params_ptr per-seq)
                max_decode_steps,               // max_new_tokens — non-zero triggers decode step loop
                0,                              // eos_token_id (batch mode: read from sampling_params_ptr per-seq)
                0,                              // output_mode
                std::ptr::null(),               // hook_ctx (from batch_ctx)
                std::ptr::null_mut(),           // telemetry
                0,                              // session_position (from batch_ctx)
                std::ptr::null(),               // fused_hidden (from batch_ctx)
                0,                              // num_mm_tokens
                std::ptr::null(),               // callback_table (from batch_ctx)
                std::ptr::null(),               // page_table (from batch_ctx)
                ctx.batch_ctx_ptr,              // batch_ctx_ptr — triggers JIT batch path
            )
        };

        log::debug!(
            "[mega] batch: total_prefill_tokens={} max_decode_steps={} generated_count={}",
            total_prefill_tokens,
            max_decode_steps,
            generated_count,
        );

        Ok(generated_count)
    }
}
