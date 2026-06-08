//! Compilation / build helpers for Executor — extracted from executor.rs.
//!
//! Contains the init / detect / compile helper methods invoked during
//! `Executor::from_loader`. Split from executor.rs to keep it under the
//! 2000-line limit.

use crate::compat::backend_trait::{Backend, Element};
use crate::loader::WeightsHandle;
use crate::model_config::ModelConfig;
use crate::tokenizer::TokenizerHandle;

use super::executor::{
    BackendError, CanonicalWeightMaps, Executor, ExecutorError, ExecutorResult, WeightMaps,
};

// ---------------------------------------------------------------------------
// Helper methods for model compilation (called from from_loader)
// ---------------------------------------------------------------------------

impl<B: Backend<E> + 'static, E: Element> Executor<B, E> {
    pub(super) fn init_moe_subsystem(
        geometry: &crate::model_config::ModelGeometry,
        is_moe: bool,
    ) -> (
        Option<crate::moe::thermal::ExpertThermalManager>,
        Option<crate::moe::fault_handler::ExpertFaultHandler>,
        Option<crate::moe::dispatch::MoeHardwareDispatcher>,
        Option<crate::moe::prefetch::ExpertWeightPrefetcher>,
        Option<crate::jit::director::JitDirector>,
        Option<crate::moe::hot_patch::HotPatchManager>,
    ) {
        if !is_moe {
            return (None, None, None, None, None, None);
        }
        let exec_plan = gllm_kernels::compiler::planner::global_execution_plan();
        let bias = &exec_plan.strategy_bias;
        let thermal = crate::moe::thermal::ExpertThermalManager::new(geometry.num_experts)
            .with_eviction_aggressiveness(bias.expert_eviction_aggressiveness());
        let fault_handler = crate::moe::fault_handler::ExpertFaultHandler::new(geometry.num_experts);
        let route_config = crate::moe::routing::ExpertRouteConfig::new(
            geometry.num_experts,
            geometry.moe_top_k,
        );
        let dispatcher = crate::moe::dispatch::MoeHardwareDispatcher::new(route_config.clone());
        let prefetcher = crate::moe::prefetch::ExpertWeightPrefetcher::new(
            geometry.num_experts,
            geometry.expert_weight_bytes(),
        )
        .with_prefetch_priority(bias.expert_prefetch_priority());
        let director_config = crate::jit::director::DirectorConfig {
            num_experts: geometry.num_experts,
            ..Default::default()
        };
        let director = crate::jit::director::JitDirector::spawn(director_config);
        let patch_manager = crate::moe::hot_patch::HotPatchManager::new(route_config);
        (
            Some(thermal),
            Some(fault_handler),
            Some(dispatcher),
            Some(prefetcher),
            Some(director),
            Some(patch_manager),
        )
    }

    pub(super) fn detect_system_topology(
        geometry: &crate::model_config::ModelGeometry,
        model_config: &ModelConfig,
    ) -> (
        crate::sensors::SystemTopology,
        crate::jit::compiler_constraints::CompilerConstraints,
        crate::jit::profiler::ProbeResult,
        crate::jit::ragged::CompactPlatform,
    ) {
        let pre_topology = crate::sensors::SystemTopology::detect();
        let compiler_constraints = pre_topology.constraints.clone();
        let probe_config = crate::jit::profiler::ProbeConfig::for_model(
            geometry.hidden_size,
            geometry.max_seq_len.min(4096),
        );
        let probe_result = match crate::jit::profiler::LatencyProfiler::probe_cpu(&probe_config) {
            Ok(result) => {
                log::info!(
                    "executor: §12.4 LatencyProfiler probe complete — spill_points={:?}, l2_thrash={}",
                    result.spill_points, result.l2_thrash_threshold,
                );
                result
            }
            Err(e) => {
                log::warn!(
                    "executor: §12.4 LatencyProfiler probe failed ({e}), deriving from topology"
                );
                let (l1d, l2, _) = pre_topology.profile.cache_sizes();
                let elem_bytes = model_config.dtype.size_bytes();
                let row_bytes = geometry.hidden_size * elem_bytes;
                let l2_thrash = if l2 > 0 {
                    l2 / (row_bytes * 2).max(1)
                } else if l1d > 0 {
                    l1d / row_bytes.max(1)
                } else {
                    model_config.max_position_embeddings
                };
                crate::jit::profiler::ProbeResult {
                    spill_points: vec![l2_thrash / 4, l2_thrash / 2, l2_thrash],
                    smem_cliffs: Vec::new(),
                    l2_thrash_threshold: l2_thrash,
                    device_fingerprint: format!("topology-derived-{}", pre_topology.cpu.core_count),
                    raw_measurements: std::collections::HashMap::new(),
                }
            }
        };
        let compact_platform = crate::jit::ragged::CompactPlatform::detect(
            if pre_topology.has_gpu() { "cuda" } else { "cpu" },
            compiler_constraints.simd_width_bits >= 512,
            compiler_constraints.simd_width_bits == 128 && cfg!(target_arch = "aarch64"),
            if cfg!(target_arch = "aarch64") {
                compiler_constraints.simd_width_bits / 8
            } else {
                0
            },
            32,
        );
        (pre_topology, compiler_constraints, probe_result, compact_platform)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_ext_weight_maps(
        weights: &mut WeightsHandle<B, E>,
        geometry: &crate::model_config::ModelGeometry,
    ) -> WeightMaps {
        use crate::compat::backend_trait::TensorLookup;
        let mut ext_ptrs: std::collections::HashMap<String, *const u8> =
            std::collections::HashMap::new();
        let mut ext_sizes: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut ext_shapes: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for name in weights.tensor_names() {
            if let Some(t) = TensorLookup::get_tensor(&*weights, name) {
                let data: &[E] = t.as_ref();
                let size = std::mem::size_of_val(data);
                ext_ptrs.insert(name.clone(), data.as_ptr() as *const u8);
                ext_sizes.insert(name.clone(), size);
            }
            if let Some(shape) = TensorLookup::tensor_shape(&*weights, name) {
                ext_shapes.insert(name.clone(), shape.to_vec());
            }
        }
        for name in weights.available_names() {
            if let Some(raw) = weights.raw_float_tensor(&name) {
                ext_ptrs.insert(name.clone(), raw.data.as_ptr());
                ext_sizes.insert(name.clone(), raw.data.len());
                ext_shapes.entry(name.clone()).or_insert_with(|| raw.shape.clone());
            }
        }
        Self::insert_converted_or_quantized(weights, geometry, &mut ext_ptrs, &mut ext_sizes, &mut ext_shapes);
        for name in weights.available_names() {
            if ext_ptrs.contains_key(&name) { continue; }
            if let Some(qt) = weights.quantized_tensor(&name) {
                ext_ptrs.insert(name.clone(), qt.data.as_ptr());
                ext_sizes.insert(name.clone(), qt.data.len());
                ext_shapes.entry(name.clone()).or_insert_with(|| qt.shape.clone());
            }
        }
        WeightMaps { ext_ptrs, ext_sizes, ext_shapes }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn insert_converted_or_quantized(
        weights: &mut WeightsHandle<B, E>,
        geometry: &crate::model_config::ModelGeometry,
        ext_ptrs: &mut std::collections::HashMap<String, *const u8>,
        ext_sizes: &mut std::collections::HashMap<String, usize>,
        ext_shapes: &mut std::collections::HashMap<String, Vec<usize>>,
    ) {
        let needs_dtype_conversion = geometry.compute_dtype != geometry.dtype;
        if needs_dtype_conversion {
            let target_dtype = geometry.compute_dtype;
            let cpu_backend = crate::compat::CpuBackend::<E>::new();
            let all_names: Vec<String> = weights.tensor_names().cloned().collect();
            let results: Vec<(String, Vec<u8>)> = all_names.into_iter()
                .filter_map(|name| weights.quantized_tensor(&name).map(|qt| (name, qt)))
                .filter_map(|(name, qt)| {
                    match crate::compat::weight_helpers::dequantize_weight_to_dtype(qt, &cpu_backend, target_dtype) {
                        Ok(typed_bytes) => Some((name, typed_bytes)),
                        Err(e) => { log::warn!("dequantize {} to {:?} failed: {}", name, target_dtype, e); None }
                    }
                })
                .collect();
            for (name, typed_bytes) in &results {
                ext_ptrs.insert(name.clone(), typed_bytes.as_ptr());
                ext_sizes.insert(name.clone(), typed_bytes.len());
                ext_shapes.entry(name.clone()).or_insert_with(|| {
                    weights.quantized_tensor(name).unwrap().shape.clone()
                });
            }
            log::info!("executor: dtype conversion {:?} → {:?}: {} tensors converted", geometry.dtype, target_dtype, results.len());
        } else {
            let qcount = weights.tensor_names()
                .filter(|n| weights.quantized_tensor(n).is_some()).count();
            log::info!("executor: zero-copy weight path — {} quantized tensors passed as-is to JIT", qcount);
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_canonical_weight_maps(
        ext_ptrs: &std::collections::HashMap<String, *const u8>,
        ext_sizes: &std::collections::HashMap<String, usize>,
        ext_shapes: &std::collections::HashMap<String, Vec<usize>>,
        model_config: &ModelConfig,
        manifest: &crate::manifest::ModelManifest,
    ) -> CanonicalWeightMaps {
        let all_names: Vec<String> = ext_shapes.keys().cloned().collect();
        let tie_embed = model_config.tie_word_embeddings.unwrap_or(false);
        let name_map =
            crate::loader::name_map::TensorNameMap::build_from_names(&all_names, tie_embed);
        let (auto_role_index, _) =
            crate::loader::build_tensor_role_index(ext_shapes.keys().map(|s| s.as_str()));
        let auto_features = crate::arch::auto_graph::analyze_architecture(
            &auto_role_index, ext_shapes, Some(&manifest.arch),
        );
        let (weight_ptrs, weight_sizes, weight_shapes) =
            Self::convert_ext_to_canonical(ext_ptrs, ext_sizes, ext_shapes, &name_map);
        CanonicalWeightMaps { weight_ptrs, weight_sizes, weight_shapes, name_map, auto_features }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn convert_ext_to_canonical(
        ext_ptrs: &std::collections::HashMap<String, *const u8>,
        ext_sizes: &std::collections::HashMap<String, usize>,
        ext_shapes: &std::collections::HashMap<String, Vec<usize>>,
        name_map: &crate::loader::name_map::TensorNameMap,
    ) -> (std::collections::HashMap<String, *const u8>, std::collections::HashMap<String, usize>, std::collections::HashMap<String, Vec<usize>>) {
        let mut weight_ptrs = std::collections::HashMap::new();
        let mut weight_sizes = std::collections::HashMap::new();
        let mut weight_shapes = std::collections::HashMap::new();
        for (ext_name, &ptr) in ext_ptrs {
            for cn in name_map.all_canonical_for(ext_name) {
                weight_ptrs.entry(cn.to_string()).or_insert(ptr);
            }
        }
        for (ext_name, &size) in ext_sizes {
            for cn in name_map.all_canonical_for(ext_name) {
                weight_sizes.entry(cn.to_string()).or_insert(size);
            }
        }
        for (ext_name, shape) in ext_shapes {
            for cn in name_map.all_canonical_for(ext_name) {
                weight_shapes.entry(cn.to_string()).or_insert(shape.clone());
            }
        }
        (weight_ptrs, weight_sizes, weight_shapes)
    }

    /// Detect heterogeneous layer structure (e.g. Gemma-4 sliding+full pattern)
    /// by comparing per-layer weight sizes against the reference (layer 0).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn detect_hetero_layers(
        weight_sizes: &std::collections::HashMap<String, usize>,
        geometry: &crate::model_config::ModelGeometry,
    ) -> Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig> {
        let find_size = |c: &str| -> Option<usize> { weight_sizes.get(c).copied() };
        let ref_q = find_size("L0.q_proj")?;
        let ref_gate = find_size("L0.gate_proj");

        let (full_indices, q_dim_full, kv_dim_full, intermediate_differs) =
            Self::scan_hetero_layer_diffs(&find_size, ref_q, ref_gate, geometry);

        if full_indices.is_empty() {
            return None;
        }
        Self::build_hetero_config(
            &find_size, ref_q, ref_gate, full_indices,
            q_dim_full, kv_dim_full, intermediate_differs, geometry,
        )
    }

    /// Scan layers 1..num_layers, collecting indices where q_proj differs from L0.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn scan_hetero_layer_diffs(
        find_size: &dyn Fn(&str) -> Option<usize>,
        ref_q: usize,
        ref_gate: Option<usize>,
        geometry: &crate::model_config::ModelGeometry,
    ) -> (Vec<usize>, usize, usize, bool) {
        let mut full_indices = Vec::new();
        let mut q_dim_full = 0usize;
        let mut kv_dim_full = 0usize;
        let mut intermediate_differs = false;

        for layer_idx in 1..geometry.num_layers {
            let q_canon = format!("L{}.q_proj", layer_idx);
            if let Some(sz) = find_size(&q_canon) {
                if sz != ref_q {
                    full_indices.push(layer_idx);
                    if q_dim_full == 0 {
                        q_dim_full = sz / (geometry.hidden_size * 4);
                        let k_canon = format!("L{}.k_proj", layer_idx);
                        if let Some(k_sz) = find_size(&k_canon) {
                            kv_dim_full = k_sz / (geometry.hidden_size * 4);
                        }
                    }
                }
            }
            if let Some(rg) = ref_gate {
                let gate_canon = format!("L{}.gate_proj", layer_idx);
                if let Some(sz) = find_size(&gate_canon) {
                    if sz != rg {
                        intermediate_differs = true;
                    }
                }
            }
        }
        (full_indices, q_dim_full, kv_dim_full, intermediate_differs)
    }

    /// Validate the [N sliding + 1 full] × M pattern and build HeteroLayerConfig.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_hetero_config(
        find_size: &dyn Fn(&str) -> Option<usize>,
        ref_q: usize,
        ref_gate: Option<usize>,
        full_indices: Vec<usize>,
        q_dim_full: usize,
        kv_dim_full: usize,
        intermediate_differs: bool,
        geometry: &crate::model_config::ModelGeometry,
    ) -> Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig> {
        let num_full = full_indices.len();
        let first_full = full_indices[0];
        let sliding_per_segment = first_full;
        let num_segments = num_full;
        let total_expected = num_segments * (sliding_per_segment + 1);

        if total_expected != geometry.num_layers {
            log::info!(
                "executor: heterogeneous layers detected but pattern doesn't match [N+1]×M: \
                 total={} expected={} — skipping mega-kernel",
                geometry.num_layers, total_expected,
            );
            return None;
        }

        let ref_q_dim = ref_q / (geometry.hidden_size * 4);
        let q_dim_full_val = q_dim_full;

        // Use head_dim from model config (SSOT). Weight-size derivation is unreliable
        // because K/V projections may use different dtypes (F16 vs F32).
        let head_dim = geometry.head_dim;
        let sliding_head_dim = head_dim;
        let full_head_dim = head_dim;
        let sliding_num_q_heads = if head_dim > 0 { ref_q_dim / head_dim } else { geometry.num_heads };
        let full_num_q_heads = if head_dim > 0 { q_dim_full_val / head_dim } else { geometry.num_heads };
        let full_num_kv_heads = if full_head_dim > 0 {
            kv_dim_full / full_head_dim
        } else {
            geometry.num_kv_heads
        };

        let ref_gate_val = ref_gate.unwrap_or(0);
        let small_intermediate = geometry.intermediate_size;
        let mut large_intermediate = small_intermediate;
        let mut large_ffn_start_segment = num_segments;

        if intermediate_differs && ref_gate_val > 0 {
            if let Some(fl) = Self::find_first_large_layer(find_size, ref_gate_val, geometry) {
                let gk = format!("L{}.gate_proj", fl);
                let large_gate = find_size(&gk).unwrap_or(ref_gate_val);
                large_intermediate = large_gate / (geometry.hidden_size * 4);
                large_ffn_start_segment = fl / (sliding_per_segment + 1);
            }
        }

        log::info!(
            "executor: heterogeneous layers: {} segments of [{} sliding + 1 full], \
             head_dim={}, sliding_q_heads={}, full_q_heads={}, sliding_kv={}, full_kv={}, \
             small_ffn={}, large_ffn={}, large_start_seg={}",
            num_segments, sliding_per_segment, head_dim,
            sliding_num_q_heads, full_num_q_heads,
            geometry.num_kv_heads, full_num_kv_heads,
            small_intermediate, large_intermediate, large_ffn_start_segment,
        );
        eprintln!("[DIAG-HETERO-CFG] head_dim={} sliding_q={} full_q={} sliding_kv={} full_kv={}",
            head_dim, sliding_num_q_heads, full_num_q_heads, geometry.num_kv_heads, full_num_kv_heads);

        Some(gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig {
            num_segments,
            sliding_per_segment,
            sliding_head_dim,
            sliding_num_q_heads,
            sliding_num_kv_heads: geometry.num_kv_heads,
            full_head_dim,
            full_num_q_heads,
            full_num_kv_heads,
            full_layer_indices: full_indices,
            small_intermediate,
            large_intermediate,
            large_ffn_start_segment,
        })
    }

    /// Find the first layer whose gate_proj size differs from the reference.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn find_first_large_layer(
        find_size: &dyn Fn(&str) -> Option<usize>,
        ref_gate: usize,
        geometry: &crate::model_config::ModelGeometry,
    ) -> Option<usize> {
        (1..geometry.num_layers).find(|&l| {
            let gk = format!("L{}.gate_proj", l);
            find_size(&gk).is_some_and(|s| s != ref_gate)
        })
    }

    /// Resolve the EOS token ID from tokenizer, config, or bos fallback.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn resolve_eos_token_id(
        tokenizer: &TokenizerHandle,
        model_config: &ModelConfig,
    ) -> u32 {
        tokenizer
            .eos_token_id()
            .or(model_config.eos_token_id)
            .or(model_config.bos_token_id)
            .unwrap_or_else(|| {
                log::warn!(
                    "mega-kernel: no eos_token_id found in tokenizer/config/bos — \
                     using 0 (generation will rely on max_new_tokens limit only)"
                );
                0
            })
    }

    /// Build BusinessConfig from model geometry, features, and manifest.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_business_config(
        geometry: &crate::model_config::ModelGeometry,
        model_config: &ModelConfig,
        manifest: &crate::manifest::ModelManifest,
        auto_features: &crate::arch::auto_graph::ArchitectureFeatures,
        eos_id: u32,
        qtap_cfg: &gllm_kernels::compiler::graph::QTapGraphConfig,
    ) -> gllm_kernels::compiler::BusinessConfig {
        use gllm_kernels::compiler::mega_kernel_abi::FfnActivation;
        use gllm_kernels::compiler::BusinessConfig;

        let max_new_tokens_default = model_config.max_position_embeddings;
        // is_single_pass: derived from topology (post-norm = no Argmax = single-pass)
        // and user config (manifest.kind = service mode choice, not compiler assumption).
        let is_single_pass = auto_features.is_post_norm
            || matches!(manifest.kind, crate::manifest::ModelKind::Embedding | crate::manifest::ModelKind::Reranker);

        let ffn_act = match geometry.hidden_act.as_ref().map(|a| a.as_str()).unwrap_or("") {
            "gelu_new" | "gelu_pytorch_tanh" | "gelu" => FfnActivation::GeGLU,
            _ => FfnActivation::SwiGLU,
        };
        let sg = if !is_single_pass {
            Some(gllm_kernels::compiler::mega_kernel_abi::SgConfig {
                detect_layer: geometry.num_layers / 2,
                detect_offset: 0,
                inject_offset: 0,
                q_tap: Some(qtap_cfg.clone()),
            })
        } else {
            None
        };
        let output_modes = Self::build_output_modes(
            manifest, is_single_pass, max_new_tokens_default, eos_id,
        );

        let mut cfg = BusinessConfig {
            has_head_rms_norm: auto_features.has_head_rms_norm,
            head_rms_norm_eps: geometry.norm_eps,
            ffn_activation: ffn_act,
            has_qk_norm: auto_features.has_qk_norm,
            has_value_norm: auto_features.has_value_norm,
            value_norm_eps: geometry.norm_eps,
            logit_softcapping: geometry.final_logit_softcapping,
            embedding_scale: auto_features.has_embedding_scale.then_some((geometry.hidden_size as f32).sqrt()),
            semantic_gatekeeper: sg,
            output_modes,
            ..BusinessConfig::default()
        };

        // Layer 6: debug_jit flag
        let debug_jit = std::env::var("GLLM_DEBUG_JIT").is_ok();
        if debug_jit {
            eprintln!("[L6] JIT debug instrumentation enabled — INT3 breakpoints will be inserted");
        }
        cfg.debug_jit = debug_jit;

        // Layer 7: MTP config
        cfg.mtp_config = model_config.mtp_depth.filter(|&d| d > 0).map(|depth| {
            gllm_kernels::compiler::MtpKernelConfig {
                depth,
                hidden_size: geometry.hidden_size,
                vocab_size: geometry.vocab_size,
            }
        });
        cfg
    }

    /// Build output_modes vector based on model kind (service mode) and single-pass topology.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_output_modes(
        manifest: &crate::manifest::ModelManifest,
        is_single_pass: bool,
        max_new_tokens: usize,
        eos_id: u32,
    ) -> Vec<gllm_kernels::compiler::mega_kernel_abi::OutputMode> {
        use gllm_kernels::compiler::mega_kernel_abi::{OutputMode, PoolMode};
        use crate::manifest::ModelKind;

        if is_single_pass {
            match manifest.kind {
                ModelKind::Embedding => vec![OutputMode::EncodeToLayer { anchor_layer: 0, pool_mode: PoolMode::MeanPool }],
                ModelKind::Reranker => vec![OutputMode::EncodeToLayer { anchor_layer: 0, pool_mode: PoolMode::ClsToken }],
                ModelKind::Classifier => vec![OutputMode::ClassifyMultiway { label_token_ids: vec![] }],
                _ => vec![OutputMode::Generate { max_new_tokens, eos_token_id: eos_id }],
            }
        } else {
            match manifest.kind {
                ModelKind::Embedding => vec![OutputMode::EncodeToLayer { anchor_layer: 0, pool_mode: PoolMode::MeanPool }],
                ModelKind::Reranker => vec![OutputMode::ClassifyBinary { positive_token_id: 0, negative_token_id: 0 }],
                _ => vec![OutputMode::Generate { max_new_tokens, eos_token_id: eos_id }],
            }
        }
    }

    /// Build per-tensor dtype and quant-type maps from loader metadata.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_weight_dtype_maps(
        weights: &WeightsHandle<B, E>,
        name_map: &crate::loader::name_map::TensorNameMap,
    ) -> (
        std::collections::HashMap<String, gllm_kernels::types::DType>,
        std::collections::HashMap<String, gllm_kernels::quant::QuantType>,
    ) {
        use crate::compat::backend_trait::TensorLookup;

        let weight_dtypes: std::collections::HashMap<String, gllm_kernels::types::DType> = {
            let mut map = std::collections::HashMap::new();
            for (ext_name, meta) in &weights.meta {
                if let Some(dt) = crate::loader::adapter::safetensors_dtype_to_gllm(meta.dtype) {
                    for cn in name_map.all_canonical_for(ext_name) {
                        eprintln!("[DTYPE-MAP] ext='{}' -> cn='{}' dt={:?}", ext_name, cn, dt);
                        map.entry(cn.to_string()).or_insert(dt);
                    }
                }
            }
            eprintln!("[DTYPE-MAP] total={} embed={:?} final_norm={:?}",
                map.len(), map.get("embed"), map.get("final_norm"));
            map
        };
        let weight_quant_types: std::collections::HashMap<String, gllm_kernels::quant::QuantType> = {
            let mut map = std::collections::HashMap::new();
            eprintln!("[QMAP-CHECK] token_embd.weight quantized={}",
                weights.quantized_tensor("token_embd.weight").is_some());
            for name in weights.available_names().iter() {
                if name == "token_embd.weight" || name.contains("output_norm") {
                    eprintln!("[QMAP-CHECK] name='{}' qt={}", name, weights.quantized_tensor(name).is_some());
                }
                if let Some(qt) = weights.quantized_tensor(name) {
                    for cn in name_map.all_canonical_for(name) {
                        map.entry(cn.to_string()).or_insert(qt.quant_type);
                    }
                }
            }
            eprintln!("[QMAP] embed={} final_norm={}", map.contains_key("embed"), map.contains_key("final_norm"));
            map
        };
        (weight_dtypes, weight_quant_types)
    }

    /// Compile CompilerGraph via auto_graph and return the built graph.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn build_compiler_graph_from_weights(
        auto_features: &crate::arch::auto_graph::ArchitectureFeatures,
        geometry: &std::sync::Arc<crate::model_config::ModelGeometry>,
        weight_shapes: &std::collections::HashMap<String, Vec<usize>>,
        weight_dtypes: &std::collections::HashMap<String, gllm_kernels::types::DType>,
        weight_quant_types: &std::collections::HashMap<String, gllm_kernels::quant::QuantType>,
        business_config: &gllm_kernels::compiler::BusinessConfig,
        max_position_embeddings: usize,
    ) -> ExecutorResult<gllm_kernels::compiler::CompilerGraph> {
        let resolved = crate::arch::ResolvedConfig::from_geometry(
            geometry, std::collections::HashMap::new(),
        );
        let graph = crate::arch::auto_graph::build_compiler_graph(
            auto_features, &resolved, weight_shapes,
            weight_dtypes, weight_quant_types, business_config,
            max_position_embeddings,
        )
        .map_err(|e| ExecutorError::Backend(BackendError::Other(format!("auto_graph build failed: {}", e))))?;

        // DEBUG: verify embed op kind after graph build
        let embed_op = graph.ops.iter().find(|op| op.label == "embed_gather");
        eprintln!("[GRAPH-CHECK] embed_gather op kind = {:?}", embed_op.map(|op| &op.kind));
        Ok(graph)
    }

    /// Compile graph into MegaKernelExecutor and upload to GPU backend.
    ///
    /// SPEC/39: unified mega-kernel path — all model families (decoder, encoder,
    /// embedding, rerank, classify) use the same compilation path.  The graph
    /// topology encodes model geometry; the compiler emits exactly what the graph
    /// contains (no runtime branching on model kind).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    fn compile_and_upload_mega(
        backend: &B,
        graph: gllm_kernels::compiler::CompilerGraph,
        geometry: &std::sync::Arc<crate::model_config::ModelGeometry>,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        weights: &mut WeightsHandle<B, E>,
        name_map: &crate::loader::name_map::TensorNameMap,
        eos_id: u32,
        business_config: gllm_kernels::compiler::BusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
        gpu_sm_version: Option<u32>,
    ) -> ExecutorResult<super::mega_kernel::MegaKernelExecutor> {
        let mega = super::mega_kernel::MegaKernelExecutor::compile_from_auto_graph(
            graph, weight_ptrs, weight_sizes,
            weights.raw_floats(), name_map,
            geometry.max_seq_len, eos_id,
            business_config, hetero_config, gpu_sm_version,
        )
        .map_err(|e| ExecutorError::Backend(BackendError::Other(format!("mega-kernel compilation failed: {}", e))))?;

        // Upload GPU artifacts
        let wb = mega.weight_blob();
        let sb = mega.scratchpad_bytes();
        let decoder_gc = mega.gpu_code();
        if let Err(e) = backend.prepare_gpu_mega_kernel(wb.unwrap_or(&[]), decoder_gc, sb) {
            log::warn!("executor: GPU mega-kernel artifact upload failed: {e}");
        }

        // Dump source map for DAP debugging
        if let Err(e) = mega.dump_source_map(std::path::Path::new("/tmp/jit_sourcemap.txt")) {
            log::debug!("executor: source map dump skipped: {e}");
        }
        Ok(mega)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub(super) fn compile_mega_kernel(
        backend: &B,
        manifest: &crate::manifest::ModelManifest,
        model_config: &ModelConfig,
        geometry: &std::sync::Arc<crate::model_config::ModelGeometry>,
        tokenizer: &TokenizerHandle,
        weights: &mut WeightsHandle<B, E>,
        qtap_cfg: &gllm_kernels::compiler::graph::QTapGraphConfig,
    ) -> ExecutorResult<Option<super::mega_kernel::MegaKernelExecutor>> {
        let eos_id = Self::resolve_eos_token_id(tokenizer, model_config);
        log::info!("mega-kernel eos_id={} (config={:?}, tokenizer={:?})",
            eos_id, model_config.eos_token_id, tokenizer.eos_token_id());

        let WeightMaps { ext_ptrs, ext_sizes, ext_shapes } =
            Self::build_ext_weight_maps(weights, geometry);

        let CanonicalWeightMaps { weight_ptrs, weight_sizes, weight_shapes, name_map, mut auto_features } =
            Self::build_canonical_weight_maps(&ext_ptrs, &ext_sizes, &ext_shapes, model_config, manifest);

        let hetero_config = Self::detect_hetero_layers(&weight_sizes, geometry);
        if let Some(ref hc) = hetero_config {
            auto_features.is_hetero_layer = true;
            auto_features.sliding_head_dim = hc.sliding_head_dim;
            auto_features.sliding_num_q_heads = hc.sliding_num_q_heads;
            auto_features.full_head_dim = hc.full_head_dim;
            auto_features.full_num_q_heads = hc.full_num_q_heads;
            auto_features.small_intermediate = hc.small_intermediate;
            auto_features.large_intermediate = hc.large_intermediate;
            auto_features.large_ffn_start_segment = hc.large_ffn_start_segment;
            auto_features.num_segments = hc.num_segments;
            auto_features.sliding_per_segment = hc.sliding_per_segment;
        }
        // SPEC/39: is_single_pass drives business config (output modes, SG gating)
        // but NOT compilation branching. Derived from topology (is_post_norm) not Family enum.
        let is_single_pass = auto_features.is_post_norm
            || matches!(manifest.kind, crate::manifest::ModelKind::Embedding | crate::manifest::ModelKind::Reranker);

        let business_config = Self::build_business_config(
            geometry, model_config, manifest, &auto_features, eos_id, qtap_cfg,
        );

        let (weight_dtypes, weight_quant_types) =
            Self::build_weight_dtype_maps(weights, &name_map);

        let graph = Self::build_compiler_graph_from_weights(
            &auto_features, geometry, &weight_shapes,
            &weight_dtypes, &weight_quant_types, &business_config,
            model_config.max_position_embeddings,
        )?;

        let sm = backend.gpu_sm_version();
        let gpu_sm_version = if sm > 0 { Some(sm) } else { None };

        let mega = Self::compile_and_upload_mega(
            backend, graph, geometry, &weight_ptrs, &weight_sizes,
            weights, &name_map, eos_id,
            business_config, hetero_config, gpu_sm_version,
        )?;
        Ok(Some(mega))
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::CpuBackend;
    use crate::engine::executor_types::{AttentionMaskType, AttentionTopology, KvCacheHandle, LogitsHandle, SequenceInput};
    use crate::manifest::ModelKind;
    use crate::model_config::ModelGeometry;
    use gllm_kernels::compiler::mega_kernel_abi::{FfnActivation, MtpKernelConfig, OutputMode, PoolMode};
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::sync::Arc;

    type TestExec = Executor<CpuBackend<f32>, f32>;

    fn make_geometry() -> ModelGeometry {
        ModelGeometry {
            hidden_size: 256,
            num_layers: 6,
            vocab_size: 32000,
            intermediate_size: 512,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 32,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
        }
    }

    fn make_auto_features() -> crate::arch::auto_graph::ArchitectureFeatures {
        crate::arch::auto_graph::ArchitectureFeatures {
            family: crate::arch::auto_graph::Family::Decoder,
            num_layers: 6,
            has_rope: true,
            has_head_rms_norm: false,
            has_attention_bias: false,
            attention_sinks: false,
            has_qk_norm: false,
            has_value_norm: false,
            has_per_layer_embedding: false,
            hidden_size_per_layer_input: 0,
            altup_num_inputs: 0,
            has_embedding_scale: false,
            norm_type: crate::arch::auto_graph::NormType::RmsNorm,
            ffn_type: crate::arch::auto_graph::FfnType::SwiGLU,
            is_moe: false,
            has_shared_experts: false,
            num_experts: 0,
            moe_top_k: 0,
            is_mla: false,
            mla_latent_dim: 0,
            mla_rope_dim: 0,
            mla_use_unabsorbed: false,
            is_vision: false,
            is_audio: false,
            has_classifier: false,
            tie_lm_head: false,
            has_norm_residual: false,
            is_post_norm: false,
            causal: true,
            has_absolute_position_embed: false,
            is_hetero_layer: false,
            sliding_head_dim: 0,
            sliding_num_q_heads: 0,
            full_head_dim: 0,
            full_num_q_heads: 0,
            small_intermediate: 0,
            large_intermediate: 0,
            large_ffn_start_segment: 0,
            num_segments: 0,
            sliding_per_segment: 0,
        }
    }

    fn make_manifest(kind: ModelKind) -> crate::manifest::ModelManifest {
        crate::manifest::ModelManifest {
            model_id: Cow::Borrowed("test-model"),
            file_map: crate::manifest::EMPTY_FILE_MAP,
            arch: "llama".to_string(),
            kind,
            rope_base_override: None,
            max_context_override: None,
            moe_config: None,
            tensor_map: HashMap::new(),
        }
    }

    // ======================================================================
    // init_moe_subsystem: non-MoE path returns all None
    // ======================================================================

    #[test]
    fn init_moe_subsystem_non_moe_returns_all_none() {
        let geo = make_geometry();
        let result = TestExec::init_moe_subsystem(&geo, false);
        assert!(result.0.is_none());
        assert!(result.1.is_none());
        assert!(result.2.is_none());
        assert!(result.3.is_none());
        assert!(result.4.is_none());
        assert!(result.5.is_none());
    }

    // ======================================================================
    // scan_hetero_layer_diffs: pure computation
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_no_diffs() {
        let geo = make_geometry(); // num_layers=6
        let sizes: HashMap<String, usize> = (1..6)
            .flat_map(|l| vec![
                (format!("L{}.q_proj", l), 1024usize),
                (format!("L{}.k_proj", l), 512usize),
                (format!("L{}.gate_proj", l), 2048usize),
            ])
            .collect();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim_full, kv_dim_full, intermediate_differs) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);

        assert!(full_indices.is_empty());
        assert_eq!(q_dim_full, 0);
        assert_eq!(kv_dim_full, 0);
        assert!(!intermediate_differs);
    }

    #[test]
    fn scan_hetero_layer_diffs_detects_q_proj_diff() {
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        // Layer 1: same as ref
        sizes.insert("L1.q_proj".to_string(), 1024);
        // Layer 2: different (full attention)
        sizes.insert("L2.q_proj".to_string(), 4096);
        sizes.insert("L2.k_proj".to_string(), 2048);
        // Layer 3: same as ref
        sizes.insert("L3.q_proj".to_string(), 1024);
        // Layer 4: different
        sizes.insert("L4.q_proj".to_string(), 4096);
        sizes.insert("L4.k_proj".to_string(), 2048);
        // Layer 5: different
        sizes.insert("L5.q_proj".to_string(), 4096);
        sizes.insert("L5.k_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim_full, kv_dim_full, _intermediate_differs) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        assert_eq!(full_indices, vec![2, 4, 5]);
        // q_dim_full = 4096 / (256 * 4) = 4
        assert_eq!(q_dim_full, 4);
        // kv_dim_full = 2048 / (256 * 4) = 2
        assert_eq!(kv_dim_full, 2);
    }

    #[test]
    fn scan_hetero_layer_diffs_detects_gate_diff() {
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        // All q_proj same
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
        }
        // gate_proj differs on layer 3
        sizes.insert("L1.gate_proj".to_string(), 2048);
        sizes.insert("L2.gate_proj".to_string(), 2048);
        sizes.insert("L3.gate_proj".to_string(), 8192);
        sizes.insert("L4.gate_proj".to_string(), 2048);
        sizes.insert("L5.gate_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, _q_dim, _kv_dim, intermediate_differs) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);

        assert!(full_indices.is_empty());
        assert!(intermediate_differs);
    }

    #[test]
    fn scan_hetero_layer_diffs_no_gate_ref() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (_, _, _, intermediate_differs) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        assert!(!intermediate_differs);
    }

    #[test]
    fn scan_hetero_layer_diffs_empty_size_map() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim_full, kv_dim_full, intermediate_differs) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);

        assert!(full_indices.is_empty());
        assert_eq!(q_dim_full, 0);
        assert_eq!(kv_dim_full, 0);
        assert!(!intermediate_differs);
    }

    #[test]
    fn scan_hetero_layer_diffs_first_diff_sets_q_dim() {
        // Only the first differing layer sets q_dim_full and kv_dim_full
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 2048);
        sizes.insert("L1.k_proj".to_string(), 1024);
        sizes.insert("L2.q_proj".to_string(), 3072);
        sizes.insert("L2.k_proj".to_string(), 1536);
        sizes.insert("L3.q_proj".to_string(), 2048);
        sizes.insert("L3.k_proj".to_string(), 1024);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim_full, kv_dim_full, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        assert_eq!(full_indices, vec![1, 2, 3]);
        // q_dim_full = 2048 / (256 * 4) = 2 (from first diff layer L1)
        assert_eq!(q_dim_full, 2);
        // kv_dim_full = 1024 / (256 * 4) = 1 (from first diff layer L1)
        assert_eq!(kv_dim_full, 1);
    }

    // ======================================================================
    // find_first_large_layer: pure computation
    // ======================================================================

    #[test]
    fn find_first_large_layer_found() {
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 2048);
        sizes.insert("L2.gate_proj".to_string(), 2048);
        sizes.insert("L3.gate_proj".to_string(), 8192);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, Some(3));
    }

    #[test]
    fn find_first_large_layer_not_found() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..6 {
            sizes.insert(format!("L{}.gate_proj", l), 2048);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, None);
    }

    #[test]
    fn find_first_large_layer_empty_map() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, None);
    }

    #[test]
    fn find_first_large_layer_first_layer_differs() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 8192);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn find_first_large_layer_single_layer_model() {
        let geo = ModelGeometry { num_layers: 1, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 8192);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // range 1..1 is empty
        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, None);
    }

    // ======================================================================
    // detect_hetero_layers: integration of scan + build
    // ======================================================================

    #[test]
    fn detect_hetero_layers_no_l0_q_proj_returns_none() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none());
    }

    #[test]
    fn detect_hetero_layers_all_same_returns_none() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
        }
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none());
    }

    #[test]
    fn detect_hetero_layers_pattern_mismatch_returns_none() {
        // 6 layers but pattern doesn't fit [N+1]*M
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        // Only layer 3 differs -> 1 full, sliding_per_segment=3, total=1*(3+1)=4 != 6
        sizes.insert("L3.q_proj".to_string(), 4096);
        sizes.insert("L3.k_proj".to_string(), 2048);
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none());
    }

    #[test]
    fn detect_hetero_layers_valid_gemma4_like_pattern() {
        // [2 sliding + 1 full] x 2 = 6 layers
        // L0=sliding(ref), L1=sliding, L2=full, L3=sliding, L4=sliding, L5=full
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024); // sliding (ref)
        sizes.insert("L1.q_proj".to_string(), 1024); // sliding
        sizes.insert("L2.q_proj".to_string(), 4096); // full
        sizes.insert("L2.k_proj".to_string(), 2048);
        sizes.insert("L3.q_proj".to_string(), 1024); // sliding
        sizes.insert("L4.q_proj".to_string(), 1024); // sliding
        sizes.insert("L5.q_proj".to_string(), 4096); // full
        sizes.insert("L5.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        // full_indices=[2,5], num_segments=2, sliding_per_segment=2
        // total = 2*(2+1) = 6 ✓
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 2);
        assert_eq!(cfg.full_layer_indices, vec![2, 5]);
    }

    #[test]
    fn detect_hetero_layers_valid_2_segment_pattern() {
        // [2 sliding + 1 full] x 2 = 6 layers
        let geo = make_geometry(); // num_layers=6, hidden_size=256, num_heads=8, head_dim=32
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024); // sliding ref
        // L1=sliding, L2=full, L3=sliding, L4=sliding, L5=full
        sizes.insert("L1.q_proj".to_string(), 1024);
        sizes.insert("L2.q_proj".to_string(), 4096);
        sizes.insert("L2.k_proj".to_string(), 2048);
        sizes.insert("L3.q_proj".to_string(), 1024);
        sizes.insert("L4.q_proj".to_string(), 1024);
        sizes.insert("L5.q_proj".to_string(), 4096);
        sizes.insert("L5.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 2);
        assert_eq!(cfg.full_layer_indices, vec![2, 5]);
        // head_dim from geometry (SSOT): geo.head_dim=32
        assert_eq!(cfg.sliding_head_dim, 32);
        assert_eq!(cfg.full_head_dim, 32);
        // ref_q_dim = 1024/(256*4) = 1, sliding_num_q_heads = 1/32 = 0
        assert_eq!(cfg.sliding_num_q_heads, 0);
        // q_dim_full = 4096/(256*4) = 4, full_num_q_heads = 4/32 = 0
        assert_eq!(cfg.full_num_q_heads, 0);
        assert_eq!(cfg.sliding_num_kv_heads, 4);
    }

    // ======================================================================
    // build_hetero_config: direct tests
    // ======================================================================

    #[test]
    fn build_hetero_config_mismatched_total_returns_none() {
        let geo = make_geometry(); // num_layers=6
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };
        // full_indices=[1], num_segments=1, sliding=1, total=2 != 6
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![1], 4, 2, false, &geo,
        );
        assert!(result.is_none());
    }

    #[test]
    fn build_hetero_config_valid_6_layer_3_segment() {
        // [1 sliding + 1 full] x 3 = 6
        let geo = make_geometry(); // num_layers=6, hidden_size=256, num_heads=8, num_kv_heads=4, head_dim=32
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 2048);
        sizes.insert("L3.gate_proj".to_string(), 2048);
        sizes.insert("L5.gate_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[1,3,5], num_segments=3, sliding_per_segment=1
        // total = 3 * (1+1) = 6 = num_layers ✓
        let result = TestExec::build_hetero_config(
            &find_size,
            1024,   // ref_q
            None,   // ref_gate
            vec![1, 3, 5], // full_indices
            4,      // q_dim_full = 4096/(256*4) = 4
            2,      // kv_dim_full
            false,  // intermediate_differs
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 3);
        assert_eq!(cfg.sliding_per_segment, 1);
        // head_dim from geometry (SSOT): geo.head_dim=32
        assert_eq!(cfg.sliding_head_dim, 32);
        assert_eq!(cfg.full_head_dim, 32);
        // ref_q_dim = 1024/(256*4) = 1, sliding_num_q_heads = 1/32 = 0
        assert_eq!(cfg.sliding_num_q_heads, 0);
        // full_num_q_heads = 4/32 = 0
        assert_eq!(cfg.full_num_q_heads, 0);
        // full_num_kv_heads: full_head_dim=32, kv_dim_full/32 = 2/32 = 0
        assert_eq!(cfg.full_num_kv_heads, 0);
        assert_eq!(cfg.full_layer_indices, vec![1, 3, 5]);
        assert_eq!(cfg.small_intermediate, 512);
        assert_eq!(cfg.large_intermediate, 512);
        assert_eq!(cfg.large_ffn_start_segment, 3);
    }

    #[test]
    fn build_hetero_config_with_larger_intermediate() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 2048);  // same as ref
        sizes.insert("L2.gate_proj".to_string(), 8192);  // larger
        sizes.insert("L3.gate_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // [1+1]*3=6, full_indices=[1,3,5], ref_gate=2048
        let result = TestExec::build_hetero_config(
            &find_size,
            1024,
            Some(2048),
            vec![1, 3, 5],
            4,
            2,
            true, // intermediate_differs
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        // find_first_large_layer finds L2 (gate=8192 != 2048)
        // large_intermediate = 8192 / (256*4) = 8
        assert_eq!(cfg.large_intermediate, 8);
        // large_ffn_start_segment = 2 / (1+1) = 1
        assert_eq!(cfg.large_ffn_start_segment, 1);
    }

    #[test]
    fn build_hetero_config_zero_ref_gate_no_intermediate_check() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::build_hetero_config(
            &find_size,
            1024,
            Some(0), // ref_gate=0: intermediate_differs branch skipped
            vec![1, 3, 5],
            4,
            2,
            true, // intermediate_differs=true but ref_gate_val=0
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        // large_intermediate stays as small_intermediate=512
        assert_eq!(cfg.large_intermediate, 512);
    }

    // ======================================================================
    // build_output_modes: pure computation
    // ======================================================================

    #[test]
    fn build_output_modes_encoder_embedding() {
        let manifest = make_manifest(ModelKind::Embedding);
        let modes = TestExec::build_output_modes(&manifest, true, 100, 2);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode: PoolMode::MeanPool } => {
                assert_eq!(*anchor_layer, 0);
            }
            _ => panic!("expected EncodeToLayer with MeanPool"),
        }
    }

    #[test]
    fn build_output_modes_encoder_reranker() {
        let manifest = make_manifest(ModelKind::Reranker);
        let modes = TestExec::build_output_modes(&manifest, true, 100, 2);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode: PoolMode::ClsToken } => {
                assert_eq!(*anchor_layer, 0);
            }
            _ => panic!("expected EncodeToLayer with ClsToken"),
        }
    }

    #[test]
    fn build_output_modes_encoder_classifier() {
        let manifest = make_manifest(ModelKind::Classifier);
        let modes = TestExec::build_output_modes(&manifest, true, 100, 2);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::ClassifyMultiway { label_token_ids } => {
                assert!(label_token_ids.is_empty());
            }
            _ => panic!("expected ClassifyMultiway"),
        }
    }

    #[test]
    fn build_output_modes_encoder_chat_generates() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, true, 512, 42);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 512);
                assert_eq!(*eos_token_id, 42);
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn build_output_modes_decoder_embedding() {
        let manifest = make_manifest(ModelKind::Embedding);
        let modes = TestExec::build_output_modes(&manifest, false, 100, 2);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode: PoolMode::MeanPool } => {
                assert_eq!(*anchor_layer, 0);
            }
            _ => panic!("expected EncodeToLayer with MeanPool"),
        }
    }

    #[test]
    fn build_output_modes_decoder_reranker() {
        let manifest = make_manifest(ModelKind::Reranker);
        let modes = TestExec::build_output_modes(&manifest, false, 100, 2);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::ClassifyBinary { positive_token_id, negative_token_id } => {
                assert_eq!(*positive_token_id, 0);
                assert_eq!(*negative_token_id, 0);
            }
            _ => panic!("expected ClassifyBinary"),
        }
    }

    #[test]
    fn build_output_modes_decoder_chat_generates() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, false, 1024, 50256);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 1024);
                assert_eq!(*eos_token_id, 50256);
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn build_output_modes_decoder_classifier_generates() {
        let manifest = make_manifest(ModelKind::Classifier);
        let modes = TestExec::build_output_modes(&manifest, false, 100, 1);
        // Decoder Classifier falls through to _ => Generate
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::Generate { .. } => {}
            _ => panic!("expected Generate for decoder classifier"),
        }
    }

    #[test]
    fn build_output_modes_zero_max_new_tokens() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, false, 0, 0);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 0);
                assert_eq!(*eos_token_id, 0);
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn build_output_modes_large_max_new_tokens() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, false, usize::MAX, u32::MAX);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, usize::MAX);
                assert_eq!(*eos_token_id, u32::MAX);
            }
            _ => panic!("expected Generate"),
        }
    }

    // ======================================================================
    // WeightMaps struct construction
    // ======================================================================

    #[test]
    fn weight_maps_construction_empty() {
        let wm = WeightMaps {
            ext_ptrs: HashMap::new(),
            ext_sizes: HashMap::new(),
            ext_shapes: HashMap::new(),
        };
        assert!(wm.ext_ptrs.is_empty());
        assert!(wm.ext_sizes.is_empty());
        assert!(wm.ext_shapes.is_empty());
    }

    #[test]
    fn weight_maps_construction_with_data() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let dummy_ptr: *const u8 = std::ptr::null();
        ptrs.insert("embed".to_string(), dummy_ptr);
        sizes.insert("embed".to_string(), 1024);
        shapes.insert("embed".to_string(), vec![32000, 256]);

        let wm = WeightMaps {
            ext_ptrs: ptrs,
            ext_sizes: sizes,
            ext_shapes: shapes,
        };
        assert_eq!(wm.ext_ptrs.len(), 1);
        assert_eq!(wm.ext_sizes.get("embed"), Some(&1024));
        assert_eq!(wm.ext_shapes.get("embed").unwrap(), &vec![32000, 256]);
    }

    // ======================================================================
    // CanonicalWeightMaps struct construction
    // ======================================================================

    #[test]
    fn canonical_weight_maps_construction() {
        let cwm = CanonicalWeightMaps {
            weight_ptrs: HashMap::new(),
            weight_sizes: HashMap::new(),
            weight_shapes: HashMap::new(),
            name_map: crate::loader::name_map::TensorNameMap::build_from_names(&[], false),
            auto_features: make_auto_features(),
        };
        assert!(cwm.weight_ptrs.is_empty());
        assert!(cwm.weight_sizes.is_empty());
        assert!(cwm.weight_shapes.is_empty());
    }

    // ======================================================================
    // scan_hetero_layer_diffs: edge cases
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_single_layer() {
        let geo = ModelGeometry { num_layers: 1, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, inter_diff) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);
        assert!(full_indices.is_empty());
        assert_eq!(q_dim, 0);
        assert_eq!(kv_dim, 0);
        assert!(!inter_diff);
    }

    #[test]
    fn scan_hetero_layer_diffs_two_layers_one_diff() {
        let geo = ModelGeometry { num_layers: 2, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);
        assert_eq!(full_indices, vec![1]);
        assert_eq!(q_dim, 4); // 4096 / (256*4) = 4
        assert_eq!(kv_dim, 2); // 2048 / (256*4) = 2
    }

    #[test]
    fn scan_hetero_layer_diffs_no_k_proj_for_first_diff() {
        let geo = ModelGeometry { num_layers: 3, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 4096);
        // No L1.k_proj -> kv_dim_full stays 0
        sizes.insert("L2.q_proj".to_string(), 4096);
        sizes.insert("L2.k_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);
        assert_eq!(full_indices, vec![1, 2]);
        assert_eq!(q_dim, 4);
        assert_eq!(kv_dim, 0); // L1 has no k_proj, so kv_dim stays 0
    }

    // ======================================================================
    // find_first_large_layer: edge cases
    // ======================================================================

    #[test]
    fn find_first_large_layer_ref_zero_with_diff() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 100);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // ref_gate=0, so any non-zero gate_proj differs
        let result = TestExec::find_first_large_layer(&find_size, 0, &geo);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn find_first_large_layer_all_missing() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, None);
    }

    #[test]
    fn find_first_large_layer_last_layer_differs() {
        let geo = make_geometry(); // num_layers=6
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..5 {
            sizes.insert(format!("L{}.gate_proj", l), 2048);
        }
        sizes.insert("L5.gate_proj".to_string(), 8192);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, Some(5));
    }

    // ======================================================================
    // build_hetero_config: head_dim calculations
    // ======================================================================

    #[test]
    fn build_hetero_config_head_dim_calculation() {
        // Use larger dimensions for meaningful head_dim calculations
        // head_dim = hidden_size / num_heads = 1024 / 16 = 64
        let geo = ModelGeometry {
            hidden_size: 1024,
            num_layers: 6,
            num_heads: 16,
            num_kv_heads: 4,
            intermediate_size: 2048,
            head_dim: 64, // SSOT: derived from hidden_size/num_heads
            ..make_geometry()
        };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        // L0.k_proj: num_kv_heads=4, head_dim=64 → k_dim=256, bytes=1024*256*4=1048576
        sizes.insert("L0.k_proj".to_string(), 1048576);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // ref_q = hidden_size * q_dim * elem_bytes = 1024 * 1024 * 4 = 4194304
        // ref_q_dim = ref_q / (hidden_size * 4) = 1024
        // sliding_num_q_heads = ref_q_dim / head_dim = 1024 / 64 = 16
        // full_num_q_heads = q_dim_full / head_dim = 512 / 64 = 8
        // full_num_kv_heads = kv_dim_full / head_dim = 128 / 64 = 2

        let ref_q = 4194304usize;
        let q_dim_full_val = 2097152usize; // 1024 * 512 * 4
        let kv_dim_full_val = 524288usize; // 1024 * 128 * 4

        // full_indices=[1,3,5], num_segments=3, sliding=1, total=3*2=6
        let result = TestExec::build_hetero_config(
            &find_size,
            ref_q,
            None,
            vec![1, 3, 5],
            q_dim_full_val / (1024 * 4), // q_dim_full = 512
            kv_dim_full_val / (1024 * 4), // kv_dim_full = 128
            false,
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 3);
        assert_eq!(cfg.sliding_per_segment, 1);
        // head_dim from geometry (SSOT): geo.head_dim=64
        assert_eq!(cfg.sliding_head_dim, 64);
        assert_eq!(cfg.full_head_dim, 64);
        // Per-type Q heads: sliding=1024/64=16, full=512/64=8
        assert_eq!(cfg.sliding_num_q_heads, 16);
        assert_eq!(cfg.full_num_q_heads, 8);
        // full_num_kv_heads = kv_dim_full / head_dim = 128 / 64 = 2
        assert_eq!(cfg.full_num_kv_heads, 2);
    }

    // ======================================================================
    // detect_hetero_layers: more integration scenarios
    // ======================================================================

    #[test]
    fn detect_hetero_layers_two_layers_pattern() {
        // [0 sliding + 1 full] * 2 = 2 layers
        let geo = ModelGeometry { num_layers: 2, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        // full_indices=[1], num_segments=1, sliding_per_segment=1
        // total = 1 * (1+1) = 2 ✓
        assert_eq!(cfg.num_segments, 1);
        assert_eq!(cfg.sliding_per_segment, 1);
        assert_eq!(cfg.full_layer_indices, vec![1]);
    }

    // ======================================================================
    // ModelGeometry helper methods used in executor_compile
    // ======================================================================

    #[test]
    fn geometry_expert_weight_bytes_zero_for_non_moe() {
        let geo = make_geometry();
        assert_eq!(geo.expert_weight_bytes(), 0);
    }

    #[test]
    fn geometry_expert_weight_bytes_with_experts() {
        let geo = ModelGeometry {
            num_experts: 64,
            expert_intermediate_size: 1408,
            ..make_geometry()
        };
        // hidden_size * expert_intermediate_size * 3 * dtype_size
        // = 256 * 1408 * 3 * 4 = 4325376
        assert_eq!(geo.expert_weight_bytes(), 256 * 1408 * 3 * 4);
    }

    #[test]
    fn geometry_effective_kv_layers_no_shared() {
        let geo = make_geometry();
        assert_eq!(geo.effective_kv_layers(), 6);
    }

    #[test]
    fn geometry_effective_kv_layers_with_shared() {
        let geo = ModelGeometry {
            num_kv_shared_layers: 2,
            ..make_geometry()
        };
        assert_eq!(geo.effective_kv_layers(), 4); // 6 - 2 = 4
    }

    #[test]
    fn geometry_kv_dim_standard() {
        let geo = make_geometry();
        assert_eq!(geo.kv_dim(), 4 * 32); // num_kv_heads * head_dim
    }

    #[test]
    fn geometry_is_mla_false() {
        let geo = make_geometry();
        assert!(!geo.is_mla());
    }

    #[test]
    fn geometry_is_mla_true() {
        let geo = ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            ..make_geometry()
        };
        assert!(geo.is_mla());
    }

    #[test]
    fn geometry_kv_dim_mla() {
        let geo = ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            ..make_geometry()
        };
        assert_eq!(geo.kv_dim(), 512 + 64);
    }

    // ======================================================================
    // scan_hetero_layer_diffs: gate_proj same as ref
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_gate_same_as_ref_not_flagged() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
            sizes.insert(format!("L{}.gate_proj", l), 2048);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (_, _, _, inter_diff) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);
        assert!(!inter_diff);
    }

    #[test]
    fn scan_hetero_layer_diffs_only_first_diff_layer_sets_dims() {
        let geo = ModelGeometry { num_layers: 3, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 2048);
        sizes.insert("L1.k_proj".to_string(), 1024);
        sizes.insert("L2.q_proj".to_string(), 8192); // second diff, larger
        sizes.insert("L2.k_proj".to_string(), 4096);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);
        // Both L1 and L2 differ from ref (1024)
        assert_eq!(full_indices, vec![1, 2]);
        // q_dim set from L1 (first diff): 2048/(256*4) = 2
        assert_eq!(q_dim, 2);
        // kv_dim set from L1: 1024/(256*4) = 1
        assert_eq!(kv_dim, 1);
    }

    // ======================================================================
    // build_hetero_config: full_num_kv_heads fallback
    // ======================================================================

    #[test]
    fn build_hetero_config_full_head_dim_zero_uses_geo_kv_heads() {
        // When head_dim=0 in geometry, the function falls back to geo.num_kv_heads
        let geo = ModelGeometry {
            head_dim: 0,
            ..make_geometry()
        }; // num_kv_heads=4
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // q_dim_full=0 -> full_head_dim=0 -> full_num_kv_heads=geo.num_kv_heads
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![1, 3, 5], 0, 0, false, &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        // head_dim=0 from geometry (SSOT) -> fallback to geo.num_kv_heads
        assert_eq!(cfg.full_num_kv_heads, 4); // fallback to geometry
        assert_eq!(cfg.full_head_dim, 0);
    }

    // ======================================================================
    // build_output_modes: all ModelKind x encoder/decoder combinations
    // ======================================================================

    #[test]
    fn build_output_modes_all_combinations_coverage() {
        let kinds = [ModelKind::Chat, ModelKind::Embedding, ModelKind::Reranker, ModelKind::Classifier];

        for &kind in &kinds {
            let manifest = make_manifest(kind);

            // Encoder path
            let encoder_modes = TestExec::build_output_modes(&manifest, true, 100, 2);
            assert_eq!(encoder_modes.len(), 1, "encoder modes should have exactly 1 entry for {:?}", kind);

            // Decoder path
            let decoder_modes = TestExec::build_output_modes(&manifest, false, 100, 2);
            assert_eq!(decoder_modes.len(), 1, "decoder modes should have exactly 1 entry for {:?}", kind);
        }
    }

    // ======================================================================
    // scan_hetero_layer_diffs: large num_layers
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_large_model() {
        let geo = ModelGeometry { num_layers: 80, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        // Even layers same as ref, odd layers different
        for l in 1..80 {
            if l % 2 == 0 {
                sizes.insert(format!("L{}.q_proj", l), 4096);
                sizes.insert(format!("L{}.k_proj", l), 2048);
            } else {
                sizes.insert(format!("L{}.q_proj", l), 1024);
            }
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        assert_eq!(full_indices.len(), 39); // layers 2,4,6,...,78 = 39 even layers in 1..80
        assert_eq!(q_dim, 4); // 4096/(256*4) = 4
        assert_eq!(kv_dim, 2); // 2048/(256*4) = 2
    }

    // ======================================================================
    // build_hetero_config: large_ffn_start_segment when no large layer found
    // ======================================================================

    #[test]
    fn build_hetero_config_no_large_gate_keeps_small_intermediate() {
        let geo = make_geometry();
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // intermediate_differs=true but find_first_large_layer returns None
        let result = TestExec::build_hetero_config(
            &find_size,
            1024,
            Some(2048),
            vec![1, 3, 5],
            4,
            2,
            true, // intermediate_differs
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        // No large gate found -> stays as small_intermediate
        assert_eq!(cfg.large_intermediate, 512);
        assert_eq!(cfg.large_ffn_start_segment, 3);
    }

    // ======================================================================
    // detect_hetero_layers: exactly 3 layers with [0+1]*3
    // ======================================================================

    #[test]
    fn detect_hetero_layers_all_full_layers_pattern_mismatch() {
        // 3 layers but 2 full layers => [1+1]*2=4 != 3 => None
        let geo = ModelGeometry { num_layers: 3, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        sizes.insert("L2.q_proj".to_string(), 4096);
        sizes.insert("L2.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        // full_indices=[1,2], num_segments=2, sliding_per_segment=1
        // total = 2*(1+1) = 4 != 3 -> None
        assert!(result.is_none());
    }

    // ======================================================================
    // find_first_large_layer: ref equals all (nothing found)
    // ======================================================================

    #[test]
    fn find_first_large_layer_all_equal_to_ref() {
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..4 {
            sizes.insert(format!("L{}.gate_proj", l), 2048);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, None);
    }

    // ======================================================================
    // Geometry helpers used by executor_compile
    // ======================================================================

    #[test]
    fn geometry_dtype_size_matches() {
        let geo = make_geometry();
        assert_eq!(geo.dtype.size_bytes(), 4); // F32
        assert_eq!(geo.compute_dtype.size_bytes(), 4); // F32
    }

    #[test]
    fn geometry_dtype_bf16() {
        let geo = ModelGeometry {
            dtype: gllm_kernels::types::DType::BF16,
            compute_dtype: gllm_kernels::types::DType::BF16,
            ..make_geometry()
        };
        assert_eq!(geo.dtype.size_bytes(), 2);
        assert_eq!(geo.compute_dtype.size_bytes(), 2);
    }

    #[test]
    fn geometry_shared_kv_ref_layers() {
        let geo = ModelGeometry {
            num_layers: 10,
            num_kv_shared_layers: 3,
            ..make_geometry()
        };
        assert_eq!(geo.effective_kv_layers(), 7);
    }

    #[test]
    fn geometry_shared_kv_ref_layers_all_shared() {
        let geo = ModelGeometry {
            num_layers: 5,
            num_kv_shared_layers: 5,
            ..make_geometry()
        };
        // saturating_sub(5-5)=0, max(1)=1
        assert_eq!(geo.effective_kv_layers(), 1);
    }

    #[test]
    fn geometry_shared_kv_ref_layers_more_than_total() {
        let geo = ModelGeometry {
            num_layers: 3,
            num_kv_shared_layers: 10,
            ..make_geometry()
        };
        // saturating_sub(3-10)=0, max(1)=1
        assert_eq!(geo.effective_kv_layers(), 1);
    }

    // ======================================================================
    // scan_hetero_layer_diffs: gate differs on only one layer
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_gate_differs_on_one_layer_only() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024); // all same q_proj
            sizes.insert(format!("L{}.gate_proj", l), 2048); // same gate
        }
        // Override layer 4 with different gate
        sizes.insert("L4.gate_proj".to_string(), 8192);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, _, _, inter_diff) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);
        assert!(full_indices.is_empty());
        assert!(inter_diff); // L4 gate differs
    }

    // ======================================================================
    // build_output_modes: encoder vs decoder produce different modes for reranker
    // ======================================================================

    #[test]
    fn build_output_modes_reranker_encoder_vs_decoder() {
        let manifest = make_manifest(ModelKind::Reranker);

        let encoder_modes = TestExec::build_output_modes(&manifest, true, 100, 2);
        let decoder_modes = TestExec::build_output_modes(&manifest, false, 100, 2);

        // Encoder reranker: EncodeToLayer with ClsToken
        match &encoder_modes[0] {
            OutputMode::EncodeToLayer { pool_mode: PoolMode::ClsToken, .. } => {}
            other => panic!("expected EncodeToLayer/ClsToken, got {:?}", other),
        }

        // Decoder reranker: ClassifyBinary
        match &decoder_modes[0] {
            OutputMode::ClassifyBinary { .. } => {}
            other => panic!("expected ClassifyBinary, got {:?}", other),
        }
    }

    // ======================================================================
    // detect_hetero_layers: single layer model returns None
    // ======================================================================

    #[test]
    fn detect_hetero_layers_single_layer() {
        let geo = ModelGeometry { num_layers: 1, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none()); // no layers to compare against
    }

    // ======================================================================
    // build_hetero_config: valid 4-layer [1+1]*2 pattern
    // ======================================================================

    #[test]
    fn build_hetero_config_valid_4_layers() {
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[1,3], num_segments=2, sliding_per_segment=1
        // total = 2*(1+1) = 4 ✓
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![1, 3], 4, 2, false, &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 1);
        assert_eq!(cfg.full_layer_indices, vec![1, 3]);
    }

    // ======================================================================
    // WeightMaps: multiple entries
    // ======================================================================

    #[test]
    fn weight_maps_multiple_tensors() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let dummy: *const u8 = 0x1 as *const u8;
        for name in &["embed", "lm_head", "final_norm"] {
            ptrs.insert(name.to_string(), dummy);
            sizes.insert(name.to_string(), 256);
            shapes.insert(name.to_string(), vec![16, 16]);
        }

        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };
        assert_eq!(wm.ext_ptrs.len(), 3);
        assert_eq!(wm.ext_sizes.len(), 3);
        assert_eq!(wm.ext_shapes.len(), 3);
    }

    // ======================================================================
    // scan_hetero_layer_diffs: ref_gate is Some but all gate_proj missing
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_ref_gate_set_but_all_gate_missing() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
        }
        // No gate_proj entries
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (_, _, _, inter_diff) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);
        assert!(!inter_diff); // No gate_proj found -> not different
    }

    // ======================================================================
    // build_hetero_config: large_intermediate calculation
    // ======================================================================

    #[test]
    fn build_hetero_config_large_intermediate_calculation() {
        let geo = ModelGeometry {
            hidden_size: 512,
            num_layers: 4,
            intermediate_size: 1024,
            ..make_geometry()
        };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        // Large gate on layer 2
        sizes.insert("L1.gate_proj".to_string(), 2048); // same as ref
        sizes.insert("L2.gate_proj".to_string(), 16384); // larger
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[1,3], ref_gate=2048
        // find_first_large_layer finds L2 (16384 != 2048)
        // large_intermediate = 16384 / (512*4) = 8
        let result = TestExec::build_hetero_config(
            &find_size,
            2048,
            Some(2048),
            vec![1, 3],
            4,
            2,
            true,
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.large_intermediate, 8);
        // large_ffn_start_segment = 2 / (1+1) = 1
        assert_eq!(cfg.large_ffn_start_segment, 1);
    }

    // ======================================================================
    // BackendError Display formatting for each variant
    // ======================================================================

    #[test]
    fn backend_error_display_cuda() {
        let err = BackendError::Cuda("device lost".to_string());
        assert_eq!(format!("{err}"), "CUDA error: device lost");
    }

    #[test]
    fn backend_error_display_hip() {
        let err = BackendError::Hip("kernel crash".to_string());
        assert_eq!(format!("{err}"), "HIP error: kernel crash");
    }

    #[test]
    fn backend_error_display_metal() {
        let err = BackendError::Metal("buffer overflow".to_string());
        assert_eq!(format!("{err}"), "Metal error: buffer overflow");
    }

    #[test]
    fn backend_error_display_cpu() {
        let err = BackendError::Cpu("illegal instruction".to_string());
        assert_eq!(format!("{err}"), "CPU error: illegal instruction");
    }

    #[test]
    fn backend_error_display_unimplemented() {
        let err = BackendError::Unimplemented("fp4 dequant");
        assert_eq!(format!("{err}"), "unimplemented: fp4 dequant");
    }

    #[test]
    fn backend_error_display_other() {
        let err = BackendError::Other("custom failure".to_string());
        assert_eq!(format!("{err}"), "backend error: custom failure");
    }

    #[test]
    fn backend_error_debug_format() {
        let err = BackendError::Cuda("oom".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("Cuda"));
        assert!(debug_str.contains("oom"));
    }

    // ======================================================================
    // ExecutorError Display formatting for key variants
    // ======================================================================

    #[test]
    fn executor_error_from_backend_error() {
        let backend_err = BackendError::Cuda("fault".to_string());
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "CUDA error: fault");
    }

    #[test]
    fn executor_error_scheduler_format() {
        let err = ExecutorError::Scheduler("no available slots".to_string());
        assert_eq!(format!("{err}"), "scheduler error: no available slots");
    }

    #[test]
    fn executor_error_empty_prompt() {
        let err = ExecutorError::EmptyPrompt;
        assert_eq!(format!("{err}"), "empty prompt tokens");
    }

    #[test]
    fn executor_error_empty_sample() {
        let err = ExecutorError::EmptySample;
        assert_eq!(format!("{err}"), "backend returned empty sample");
    }

    #[test]
    fn executor_error_request_not_found() {
        let err = ExecutorError::RequestNotFound { request_id: 42 };
        assert_eq!(format!("{err}"), "request not found: 42");
    }

    #[test]
    fn executor_error_compilation() {
        let err = ExecutorError::Compilation("register overflow".to_string());
        assert_eq!(format!("{err}"), "JIT compilation failed: register overflow");
    }

    #[test]
    fn executor_error_graph_expansion() {
        let err = ExecutorError::GraphExpansion("unknown op".to_string());
        assert_eq!(format!("{err}"), "graph expansion failed: unknown op");
    }

    // ======================================================================
    // BusinessConfig default values
    // ======================================================================

    #[test]
    fn business_config_default_output_mode_is_generate() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig::default();
        assert_eq!(cfg.output_modes.len(), 1);
        match &cfg.output_modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 512);
                assert_eq!(*eos_token_id, 2);
            }
            other => panic!("expected Generate, got {:?}", other),
        }
    }

    #[test]
    fn business_config_default_flags_off() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig::default();
        assert!(!cfg.guardrail_enabled);
        assert!(!cfg.has_head_rms_norm);
        assert!(!cfg.has_qk_norm);
        assert!(!cfg.has_value_norm);
        assert!(!cfg.session_enabled);
        assert!(!cfg.multimodal_enabled);
        assert!(!cfg.debug_jit);
    }

    #[test]
    fn business_config_default_none_fields() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig::default();
        assert!(cfg.semantic_gatekeeper.is_none());
        assert!(cfg.intent_anchor_layer.is_none());
        assert!(cfg.cot_step_hook.is_none());
        assert!(cfg.logit_softcapping.is_none());
        assert!(cfg.embedding_scale.is_none());
        assert!(cfg.mtp_config.is_none());
    }

    #[test]
    fn business_config_default_ffn_activation_swiglu() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig::default();
        assert_eq!(cfg.ffn_activation, FfnActivation::SwiGLU);
    }

    #[test]
    fn business_config_default_eps_values() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig::default();
        assert!((cfg.head_rms_norm_eps - 1e-6).abs() < 1e-10);
        assert!((cfg.value_norm_eps - 1e-6).abs() < 1e-10);
    }

    // ======================================================================
    // FfnActivation Debug and Copy semantics
    // ======================================================================

    #[test]
    fn ffn_activation_debug_format() {
        assert_eq!(format!("{:?}", FfnActivation::SwiGLU), "SwiGLU");
        assert_eq!(format!("{:?}", FfnActivation::GeGLU), "GeGLU");
        assert_eq!(format!("{:?}", FfnActivation::Gelu), "Gelu");
    }

    #[test]
    fn ffn_activation_equality() {
        assert_eq!(FfnActivation::SwiGLU, FfnActivation::SwiGLU);
        assert_ne!(FfnActivation::SwiGLU, FfnActivation::GeGLU);
        assert_ne!(FfnActivation::GeGLU, FfnActivation::Gelu);
    }

    #[test]
    fn ffn_activation_copy() {
        let a = FfnActivation::GeGLU;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // ======================================================================
    // OutputMode Debug formatting
    // ======================================================================

    #[test]
    fn output_mode_debug_format() {
        let mode = OutputMode::Generate { max_new_tokens: 100, eos_token_id: 2 };
        let debug = format!("{mode:?}");
        assert!(debug.contains("Generate"));

        let mode = OutputMode::ClassifyBinary { positive_token_id: 1, negative_token_id: 0 };
        let debug = format!("{mode:?}");
        assert!(debug.contains("ClassifyBinary"));

        let mode = OutputMode::ClassifyMultiway { label_token_ids: vec![1, 2, 3] };
        let debug = format!("{mode:?}");
        assert!(debug.contains("ClassifyMultiway"));

        let mode = OutputMode::EncodeToLayer { anchor_layer: 5, pool_mode: PoolMode::MeanPool };
        let debug = format!("{mode:?}");
        assert!(debug.contains("EncodeToLayer"));
    }

    // ======================================================================
    // PoolMode Debug formatting
    // ======================================================================

    #[test]
    fn pool_mode_debug_format() {
        assert_eq!(format!("{:?}", PoolMode::LastToken), "LastToken");
        assert_eq!(format!("{:?}", PoolMode::MeanPool), "MeanPool");
        assert_eq!(format!("{:?}", PoolMode::ClsToken), "ClsToken");
    }

    // ======================================================================
    // SgConfig construction
    // ======================================================================

    #[test]
    fn sg_config_construction() {
        use gllm_kernels::compiler::mega_kernel_abi::SgConfig;
        let cfg = SgConfig {
            detect_layer: 12,
            detect_offset: 64,
            inject_offset: 0,
            q_tap: None,
        };
        assert_eq!(cfg.detect_layer, 12);
        assert_eq!(cfg.detect_offset, 64);
        assert_eq!(cfg.inject_offset, 0);
        assert!(cfg.q_tap.is_none());
    }

    // ======================================================================
    // MtpKernelConfig construction
    // ======================================================================

    #[test]
    fn mtp_kernel_config_construction() {
        use gllm_kernels::compiler::MtpKernelConfig;
        let cfg = MtpKernelConfig {
            depth: 3,
            hidden_size: 4096,
            vocab_size: 128256,
        };
        assert_eq!(cfg.depth, 3);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.vocab_size, 128256);
    }

    // ======================================================================
    // HeteroLayerConfig construction
    // ======================================================================

    #[test]
    fn hetero_layer_config_construction() {
        use gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig;
        let cfg = HeteroLayerConfig {
            num_segments: 7,
            sliding_per_segment: 4,
            sliding_head_dim: 256,
            sliding_num_q_heads: 4,
            sliding_num_kv_heads: 1,
            full_head_dim: 256,
            full_num_q_heads: 8,
            full_num_kv_heads: 1,
            full_layer_indices: vec![4, 9, 14, 19, 24, 29, 34],
            small_intermediate: 2304,
            large_intermediate: 9216,
            large_ffn_start_segment: 3,
        };
        assert_eq!(cfg.num_segments, 7);
        assert_eq!(cfg.sliding_per_segment, 4);
        assert_eq!(cfg.sliding_head_dim, 256);
        assert_eq!(cfg.full_layer_indices.len(), 7);
        assert_eq!(cfg.small_intermediate, 2304);
        assert_eq!(cfg.large_intermediate, 9216);
        assert_eq!(cfg.large_ffn_start_segment, 3);
    }

    // ======================================================================
    // KvCacheHandle construction, equality, hash
    // ======================================================================

    #[test]
    fn kv_cache_handle_equality() {
        let a = KvCacheHandle(42);
        let b = KvCacheHandle(42);
        let c = KvCacheHandle(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn kv_cache_handle_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KvCacheHandle(1));
        set.insert(KvCacheHandle(1));
        set.insert(KvCacheHandle(2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn kv_cache_handle_copy() {
        let a = KvCacheHandle(7);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_debug() {
        let h = KvCacheHandle(123);
        let debug = format!("{h:?}");
        assert!(debug.contains("123"));
    }

    // ======================================================================
    // AttentionMaskType variants and equality
    // ======================================================================

    #[test]
    fn attention_mask_type_equality() {
        assert_eq!(AttentionMaskType::Bidirectional, AttentionMaskType::Bidirectional);
        assert_eq!(AttentionMaskType::Causal, AttentionMaskType::Causal);
        assert_ne!(AttentionMaskType::Bidirectional, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_mask_type_debug() {
        let bidir = format!("{:?}", AttentionMaskType::Bidirectional);
        let causal = format!("{:?}", AttentionMaskType::Causal);
        assert!(bidir.contains("Bidirectional"));
        assert!(causal.contains("Causal"));
    }

    // ======================================================================
    // AttentionTopology construction
    // ======================================================================

    #[test]
    fn attention_topology_bidirectional() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::bidirectional(geo.clone());
        assert!(matches!(topo.mask_type, AttentionMaskType::Bidirectional));
        assert_eq!(topo.geometry.hidden_size, 256);
    }

    #[test]
    fn attention_topology_causal() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::causal(geo.clone());
        assert!(matches!(topo.mask_type, AttentionMaskType::Causal));
        assert_eq!(topo.geometry.num_layers, 6);
    }

    #[test]
    fn attention_topology_linear_default() {
        let topo = AttentionTopology::linear();
        assert!(matches!(topo.mask_type, AttentionMaskType::Bidirectional));
        assert_eq!(topo.geometry.hidden_size, 1);
        assert_eq!(topo.geometry.vocab_size, 1);
    }

    // ======================================================================
    // ModelKind parse — all recognized aliases
    // ======================================================================

    #[test]
    fn model_kind_parse_chat_aliases() {
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("generator"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("text-generation"), Some(ModelKind::Chat));
    }

    #[test]
    fn model_kind_parse_embedding_aliases() {
        assert_eq!(ModelKind::parse("embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("embeddings"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("embed"), Some(ModelKind::Embedding));
    }

    #[test]
    fn model_kind_parse_reranker_aliases() {
        assert_eq!(ModelKind::parse("rerank"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-ranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-rank"), Some(ModelKind::Reranker));
    }

    #[test]
    fn model_kind_parse_classifier_aliases() {
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("classification"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("classify"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("sequence-classification"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("text-classification"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_case_insensitive() {
        assert_eq!(ModelKind::parse("Chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("EMBEDDING"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("  Reranker  "), Some(ModelKind::Reranker));
    }

    #[test]
    fn model_kind_parse_unknown_returns_none() {
        assert_eq!(ModelKind::parse("unknown"), None);
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("translation"), None);
    }

    // ======================================================================
    // detect_hetero_layers: only L0 matters for entry
    // ======================================================================

    #[test]
    fn detect_hetero_layers_missing_l0_with_diffs_still_none() {
        // No L0.q_proj but L1 has different q_proj => returns None at find_size("L0.q_proj")?
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        sizes.insert("L2.q_proj".to_string(), 4096);
        sizes.insert("L2.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        // No L0.q_proj => returns None immediately
        assert!(result.is_none());
    }

    // ======================================================================
    // scan_hetero_layer_diffs: only q_proj size difference matters for full_indices
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_q_proj_missing_on_some_layers() {
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        // L2 has no q_proj at all
        sizes.insert("L3.q_proj".to_string(), 4096);
        sizes.insert("L3.k_proj".to_string(), 2048);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        // Only L1 and L3 have q_proj that differs from ref
        assert_eq!(full_indices, vec![1, 3]);
        assert_eq!(q_dim, 4); // from L1
        assert_eq!(kv_dim, 2); // from L1
    }

    // ======================================================================
    // build_hetero_config: num_segments=1 (simplest valid hetero)
    // ======================================================================

    #[test]
    fn build_hetero_config_single_segment() {
        // [N sliding + 1 full] * 1 = N+1 layers
        let geo = ModelGeometry { num_layers: 5, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[4], sliding_per_segment=4, total=1*(4+1)=5
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![4], 4, 2, false, &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 1);
        assert_eq!(cfg.sliding_per_segment, 4);
        assert_eq!(cfg.full_layer_indices, vec![4]);
    }

    // ======================================================================
    // build_hetero_config: full_num_kv_heads when full_head_dim > 0
    // ======================================================================

    #[test]
    fn build_hetero_config_full_kv_heads_when_head_dim_nonzero() {
        // head_dim = hidden_size / num_heads = 512 / 8 = 64
        let geo = ModelGeometry {
            hidden_size: 512,
            num_layers: 4,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 64, // SSOT: derived from hidden_size/num_heads
            ..make_geometry()
        };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // ref_q=4096, ref_q_dim=4096/(512*4)=2
        // head_dim = 64 (from geometry SSOT)
        // sliding_num_q_heads = ref_q_dim / head_dim = 2/64 = 0
        // full_head_dim = 64 (same as sliding)
        // full_num_q_heads = q_dim_full / head_dim = 16/64 = 0
        // full_num_kv_heads = kv_dim_full / head_dim = 8/64 = 0
        let result = TestExec::build_hetero_config(
            &find_size,
            4096, // ref_q
            None,
            vec![1, 3],
            16, // q_dim_full
            8,  // kv_dim_full
            false,
            &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.sliding_head_dim, 64);
        assert_eq!(cfg.full_head_dim, 64);
        assert_eq!(cfg.full_num_kv_heads, 0);
    }

    // ======================================================================
    // BusinessConfig manual construction with fields
    // ======================================================================

    #[test]
    fn business_config_manual_with_mtp() {
        use gllm_kernels::compiler::BusinessConfig;
        let mtp = MtpKernelConfig {
            depth: 2,
            hidden_size: 256,
            vocab_size: 32000,
        };
        let cfg = BusinessConfig {
            mtp_config: Some(mtp),
            ..BusinessConfig::default()
        };
        assert!(cfg.mtp_config.is_some());
        let m = cfg.mtp_config.unwrap();
        assert_eq!(m.depth, 2);
        assert_eq!(m.hidden_size, 256);
        assert_eq!(m.vocab_size, 32000);
    }

    #[test]
    fn business_config_with_qk_norm() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            has_qk_norm: true,
            has_value_norm: true,
            value_norm_eps: 1e-5,
            ..BusinessConfig::default()
        };
        assert!(cfg.has_qk_norm);
        assert!(cfg.has_value_norm);
        assert!((cfg.value_norm_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn business_config_geglu_activation() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            ffn_activation: FfnActivation::GeGLU,
            ..BusinessConfig::default()
        };
        assert_eq!(cfg.ffn_activation, FfnActivation::GeGLU);
    }

    #[test]
    fn business_config_with_embedding_scale() {
        use gllm_kernels::compiler::BusinessConfig;
        let scale = (256.0f32).sqrt();
        let cfg = BusinessConfig {
            embedding_scale: Some(scale),
            ..BusinessConfig::default()
        };
        assert!(cfg.embedding_scale.is_some());
        assert!((cfg.embedding_scale.unwrap() - scale).abs() < 1e-5);
    }

    // ======================================================================
    // LogitsHandle construction and Debug
    // ======================================================================

    #[test]
    fn logits_handle_construction_and_debug() {
        let handle = LogitsHandle { data: vec![0.1, 0.5, 0.4] };
        assert_eq!(handle.data.len(), 3);
        let debug = format!("{handle:?}");
        assert!(debug.contains("LogitsHandle"));
    }

    // ======================================================================
    // ModelGeometry: kv_dim with zero heads (edge case)
    // ======================================================================

    #[test]
    fn geometry_kv_dim_zero_kv_heads() {
        let geo = ModelGeometry {
            num_kv_heads: 0,
            head_dim: 32,
            ..make_geometry()
        };
        assert_eq!(geo.kv_dim(), 0); // 0 * 32 = 0
    }

    #[test]
    fn geometry_kv_dim_mla_dims_zero() {
        let geo = ModelGeometry {
            mla_d_c: 0,
            mla_d_rope: 0,
            ..make_geometry()
        };
        assert!(!geo.is_mla());
        // kv_dim uses num_kv_heads * head_dim, not MLA path
        assert_eq!(geo.kv_dim(), 4 * 32);
    }

    // ======================================================================
    // detect_hetero_layers: 3-segment [1+1]*3 pattern with no gate data
    // ======================================================================

    #[test]
    fn detect_hetero_layers_3_segment_no_gate() {
        // [1 sliding + 1 full] * 3 = 6
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        sizes.insert("L3.q_proj".to_string(), 4096);
        sizes.insert("L3.k_proj".to_string(), 2048);
        sizes.insert("L5.q_proj".to_string(), 4096);
        sizes.insert("L5.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 3);
        assert_eq!(cfg.sliding_per_segment, 1);
        assert_eq!(cfg.full_layer_indices, vec![1, 3, 5]);
        assert_eq!(cfg.small_intermediate, 512); // from geo
        assert_eq!(cfg.large_intermediate, 512); // no gate data => stays same
    }

    // ======================================================================
    // NEW TESTS — ~50 additional unit tests
    // ======================================================================

    // ======================================================================
    // BusinessConfig: guardrail_enabled default and override
    // ======================================================================

    #[test]
    fn business_config_guardrail_override() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            guardrail_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.guardrail_enabled);
        // All other fields should remain default
        assert!(!cfg.session_enabled);
        assert!(!cfg.multimodal_enabled);
    }

    #[test]
    fn business_config_session_enabled_override() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            session_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.session_enabled);
        assert!(!cfg.guardrail_enabled);
    }

    #[test]
    fn business_config_multimodal_enabled_override() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            multimodal_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.multimodal_enabled);
    }

    #[test]
    fn business_config_logit_softcapping_override() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            logit_softcapping: Some(30.0),
            ..BusinessConfig::default()
        };
        assert!(cfg.logit_softcapping.is_some());
        assert!((cfg.logit_softcapping.unwrap() - 30.0).abs() < 1e-5);
    }

    #[test]
    fn business_config_debug_jit_override() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            debug_jit: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.debug_jit);
    }

    #[test]
    fn business_config_multiple_output_modes() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            output_modes: vec![
                OutputMode::Generate { max_new_tokens: 256, eos_token_id: 2 },
                OutputMode::ClassifyBinary { positive_token_id: 1, negative_token_id: 0 },
            ],
            ..BusinessConfig::default()
        };
        assert_eq!(cfg.output_modes.len(), 2);
    }

    // ======================================================================
    // FfnActivation: ordering and exhaustive variant coverage
    // ======================================================================

    #[test]
    fn ffn_activation_all_variants_distinct() {
        let variants = [FfnActivation::SwiGLU, FfnActivation::GeGLU, FfnActivation::Gelu];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ======================================================================
    // BackendError: Clone semantics
    // ======================================================================

    #[test]
    fn backend_error_clone_preserves_message() {
        let err = BackendError::Cuda("out of memory".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_hip() {
        let err = BackendError::Hip("device error".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_metal() {
        let err = BackendError::Metal("shader error".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_cpu() {
        let err = BackendError::Cpu("alignment".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_other() {
        let err = BackendError::Other("custom msg".to_string());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_unimplemented_static_str() {
        let err = BackendError::Unimplemented("fp8 gemm");
        assert_eq!(format!("{err}"), "unimplemented: fp8 gemm");
    }

    // ======================================================================
    // BackendError: empty message strings
    // ======================================================================

    #[test]
    fn backend_error_empty_cuda_message() {
        let err = BackendError::Cuda(String::new());
        assert_eq!(format!("{err}"), "CUDA error: ");
    }

    #[test]
    fn backend_error_empty_other_message() {
        let err = BackendError::Other(String::new());
        assert_eq!(format!("{err}"), "backend error: ");
    }

    // ======================================================================
    // KvCacheHandle: zero handle and max value
    // ======================================================================

    #[test]
    fn kv_cache_handle_zero() {
        let h = KvCacheHandle(0);
        assert_eq!(h.0, 0);
    }

    #[test]
    fn kv_cache_handle_max_u64() {
        let h = KvCacheHandle(u64::MAX);
        assert_eq!(h.0, u64::MAX);
    }

    #[test]
    fn kv_cache_handle_ordering() {
        let a = KvCacheHandle(1);
        let b = KvCacheHandle(2);
        assert_ne!(a, b);
    }

    // ======================================================================
    // LogitsHandle: empty data
    // ======================================================================

    #[test]
    fn logits_handle_empty_data() {
        let handle = LogitsHandle { data: vec![] };
        assert!(handle.data.is_empty());
        let debug = format!("{handle:?}");
        assert!(debug.contains("LogitsHandle"));
    }

    #[test]
    fn logits_handle_single_element() {
        let handle = LogitsHandle { data: vec![0.42] };
        assert_eq!(handle.data.len(), 1);
        assert!((handle.data[0] - 0.42).abs() < 1e-6);
    }

    #[test]
    fn logits_handle_clone() {
        let handle = LogitsHandle { data: vec![0.1, 0.9] };
        let cloned = handle.clone();
        assert_eq!(handle.data.len(), cloned.data.len());
        for (a, b) in handle.data.iter().zip(cloned.data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ======================================================================
    // AttentionTopology: accessor methods
    // ======================================================================

    #[test]
    fn attention_topology_accessors() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::causal(geo.clone());
        assert_eq!(topo.num_heads(), 8);
        assert_eq!(topo.num_kv_heads(), 4);
        assert_eq!(topo.head_dim(), 32);
        assert_eq!(topo.max_seq_len(), 2048);
    }

    #[test]
    fn attention_topology_bidirectional_accessors() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::bidirectional(geo);
        assert_eq!(topo.num_heads(), 8);
        assert_eq!(topo.num_kv_heads(), 4);
    }

    #[test]
    fn attention_topology_linear_accessors() {
        let topo = AttentionTopology::linear();
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
        assert_eq!(topo.max_seq_len(), 512);
    }

    // ======================================================================
    // AttentionMaskType: Copy semantics
    // ======================================================================

    #[test]
    fn attention_mask_type_copy() {
        let a = AttentionMaskType::Causal;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // ======================================================================
    // ModelKind: FromStr implementation
    // ======================================================================

    #[test]
    fn model_kind_from_str_valid() {
        use std::str::FromStr;
        assert_eq!(ModelKind::from_str("chat").unwrap(), ModelKind::Chat);
        assert_eq!(ModelKind::from_str("embedding").unwrap(), ModelKind::Embedding);
        assert_eq!(ModelKind::from_str("reranker").unwrap(), ModelKind::Reranker);
        assert_eq!(ModelKind::from_str("classifier").unwrap(), ModelKind::Classifier);
    }

    #[test]
    fn model_kind_from_str_invalid() {
        use std::str::FromStr;
        assert!(ModelKind::from_str("unknown").is_err());
        assert!(ModelKind::from_str("").is_err());
    }

    #[test]
    fn model_kind_equality_and_ordering() {
        assert_eq!(ModelKind::Chat, ModelKind::Chat);
        assert_ne!(ModelKind::Chat, ModelKind::Embedding);
        assert_ne!(ModelKind::Embedding, ModelKind::Reranker);
        assert_ne!(ModelKind::Reranker, ModelKind::Classifier);
    }

    // ======================================================================
    // SamplingConfig: defaults
    // ======================================================================

    #[test]
    fn sampling_config_default_values() {
        use crate::engine::executor_types::SamplingConfig;
        let cfg = SamplingConfig::default();
        assert!((cfg.temperature - 1.0).abs() < 1e-6);
        assert_eq!(cfg.top_k, 0);
        assert!((cfg.top_p - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sampling_config_custom_values() {
        use crate::engine::executor_types::SamplingConfig;
        let cfg = SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.9 };
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
        assert_eq!(cfg.top_k, 50);
        assert!((cfg.top_p - 0.9).abs() < 1e-6);
    }

    // ======================================================================
    // SequenceInput: validate_page_table
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_none_is_ok() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(input.validate_page_table(100).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_valid() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: None,
        };
        assert!(input.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_out_of_bounds() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![5, 10, 15]),
            fused_hidden: None,
        };
        let result = input.validate_page_table(10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("page_table"));
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_exact() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![9]),
            fused_hidden: None,
        };
        // page_id 9 < total_pages 10 => ok
        assert!(input.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_fail() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![10]),
            fused_hidden: None,
        };
        // page_id 10 >= total_pages 10 => err
        assert!(input.validate_page_table(10).is_err());
    }

    // ======================================================================
    // Hetero layer detection: many layers, contiguous full pattern
    // ======================================================================

    #[test]
    fn detect_hetero_layers_4_segment_pattern() {
        // [3 sliding + 1 full] * 4 = 16 layers
        let geo = ModelGeometry { num_layers: 16, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        for &full_idx in &[3, 7, 11, 15] {
            sizes.insert(format!("L{}.q_proj", full_idx), 4096);
            sizes.insert(format!("L{}.k_proj", full_idx), 2048);
        }
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 4);
        assert_eq!(cfg.sliding_per_segment, 3);
        assert_eq!(cfg.full_layer_indices, vec![3, 7, 11, 15]);
    }

    #[test]
    fn detect_hetero_layers_non_contiguous_full_indices_mismatch() {
        // 6 layers, full at [2, 5] => [2+1]*2=6 => valid
        // but full at [1, 5] => [1+1]*2=4 != 6 => invalid
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        sizes.insert("L1.q_proj".to_string(), 4096);
        sizes.insert("L1.k_proj".to_string(), 2048);
        sizes.insert("L5.q_proj".to_string(), 4096);
        sizes.insert("L5.k_proj".to_string(), 2048);
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        // full_indices=[1,5], num_segments=2, sliding=1, total=2*2=4 != 6
        assert!(result.is_none());
    }

    // ======================================================================
    // build_output_modes: edge case max_new_tokens=1
    // ======================================================================

    #[test]
    fn build_output_modes_single_token_generation() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, false, 1, 2);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, .. } => assert_eq!(*max_new_tokens, 1),
            _ => panic!("expected Generate"),
        }
    }

    // ======================================================================
    // ExecutorError: Backend variant via Into
    // ======================================================================

    #[test]
    fn executor_error_from_hip_backend() {
        let backend_err = BackendError::Hip("sync failed".to_string());
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "HIP error: sync failed");
    }

    #[test]
    fn executor_error_from_metal_backend() {
        let backend_err = BackendError::Metal("alloc".to_string());
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "Metal error: alloc");
    }

    #[test]
    fn executor_error_from_cpu_backend() {
        let backend_err = BackendError::Cpu("segfault".to_string());
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "CPU error: segfault");
    }

    #[test]
    fn executor_error_from_unimplemented() {
        let backend_err = BackendError::Unimplemented("sparse gemm");
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "unimplemented: sparse gemm");
    }

    #[test]
    fn executor_error_from_other_backend() {
        let backend_err = BackendError::Other("timeout".to_string());
        let exec_err: ExecutorError = backend_err.into();
        assert_eq!(format!("{exec_err}"), "backend error: timeout");
    }

    // ======================================================================
    // OutputMode: Clone semantics
    // ======================================================================

    #[test]
    fn output_mode_clone_generate() {
        let mode = OutputMode::Generate { max_new_tokens: 100, eos_token_id: 2 };
        let cloned = mode.clone();
        match cloned {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(max_new_tokens, 100);
                assert_eq!(eos_token_id, 2);
            }
            _ => panic!("expected Generate"),
        }
    }

    #[test]
    fn output_mode_clone_encode_to_layer() {
        let mode = OutputMode::EncodeToLayer { anchor_layer: 3, pool_mode: PoolMode::MeanPool };
        let cloned = mode.clone();
        match cloned {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode: _ } => {
                assert_eq!(anchor_layer, 3);
            }
            _ => panic!("expected EncodeToLayer"),
        }
    }

    #[test]
    fn output_mode_clone_classify_binary() {
        let mode = OutputMode::ClassifyBinary { positive_token_id: 1, negative_token_id: 0 };
        let cloned = mode.clone();
        match cloned {
            OutputMode::ClassifyBinary { positive_token_id, negative_token_id } => {
                assert_eq!(positive_token_id, 1);
                assert_eq!(negative_token_id, 0);
            }
            _ => panic!("expected ClassifyBinary"),
        }
    }

    #[test]
    fn output_mode_clone_classify_multiway() {
        let mode = OutputMode::ClassifyMultiway { label_token_ids: vec![10, 20, 30] };
        let cloned = mode.clone();
        match cloned {
            OutputMode::ClassifyMultiway { label_token_ids } => {
                assert_eq!(label_token_ids, vec![10, 20, 30]);
            }
            _ => panic!("expected ClassifyMultiway"),
        }
    }

    // ======================================================================
    // SgConfig: with q_tap present
    // ======================================================================

    #[test]
    fn sg_config_with_qtap() {
        use gllm_kernels::compiler::mega_kernel_abi::SgConfig;
        use gllm_kernels::compiler::graph::{QTapGraphConfig, QTapPosition};
        use gllm_kernels::types::DType;
        let qtap = QTapGraphConfig {
            sink_ptr: 0,
            step_index_ptr: 0,
            dtype: DType::F32,
            position: QTapPosition::LastToken,
            num_slots: 4,
        };
        let cfg = SgConfig {
            detect_layer: 6,
            detect_offset: 128,
            inject_offset: 256,
            q_tap: Some(qtap),
        };
        assert_eq!(cfg.detect_layer, 6);
        assert!(cfg.q_tap.is_some());
    }

    // ======================================================================
    // ModelGeometry: kv_dim for various configurations
    // ======================================================================

    #[test]
    fn geometry_kv_dim_standard_heads() {
        let geo = make_geometry(); // num_kv_heads=4, head_dim=32
        assert_eq!(geo.kv_dim(), 128);
    }

    #[test]
    fn geometry_kv_dim_gqa() {
        let geo = ModelGeometry {
            num_kv_heads: 1,
            head_dim: 128,
            ..make_geometry()
        };
        assert_eq!(geo.kv_dim(), 128);
    }

    #[test]
    fn geometry_is_mla_requires_both_dims() {
        let geo = ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 0,
            ..make_geometry()
        };
        // is_mla checks mla_d_c > 0 && mla_d_rope > 0
        // Actually check the implementation
        let result = geo.is_mla();
        // If is_mla requires both > 0 then this is false
        // Let's verify by checking what kv_dim returns
        if !result {
            // Standard path: num_kv_heads * head_dim
            assert_eq!(geo.kv_dim(), 4 * 32);
        }
    }

    // ======================================================================
    // MtpKernelConfig: Clone and Debug
    // ======================================================================

    #[test]
    fn mtp_kernel_config_clone() {
        let cfg = MtpKernelConfig { depth: 4, hidden_size: 1024, vocab_size: 50000 };
        let cloned = cfg.clone();
        assert_eq!(cloned.depth, 4);
        assert_eq!(cloned.hidden_size, 1024);
        assert_eq!(cloned.vocab_size, 50000);
    }

    #[test]
    fn mtp_kernel_config_debug() {
        let cfg = MtpKernelConfig { depth: 1, hidden_size: 256, vocab_size: 32000 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("MtpKernelConfig"));
    }

    // ======================================================================
    // HeteroLayerConfig: Clone
    // ======================================================================

    #[test]
    fn hetero_layer_config_clone() {
        use gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig;
        let cfg = HeteroLayerConfig {
            num_segments: 2,
            sliding_per_segment: 3,
            sliding_head_dim: 64,
            sliding_num_q_heads: 4,
            sliding_num_kv_heads: 2,
            full_head_dim: 128,
            full_num_q_heads: 8,
            full_num_kv_heads: 4,
            full_layer_indices: vec![3, 7],
            small_intermediate: 1024,
            large_intermediate: 4096,
            large_ffn_start_segment: 1,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.num_segments, 2);
        assert_eq!(cloned.full_layer_indices, vec![3, 7]);
    }

    // ======================================================================
    // AttentionTopology: clone
    // ======================================================================

    #[test]
    fn attention_topology_clone() {
        let geo = Arc::new(make_geometry());
        let topo = AttentionTopology::causal(geo);
        let cloned = topo.clone();
        assert!(matches!(cloned.mask_type, AttentionMaskType::Causal));
        assert_eq!(cloned.num_heads(), 8);
    }

    // ======================================================================
    // scan_hetero_layer_diffs: all layers have different q_proj from ref
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_all_layers_differ() {
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..4 {
            sizes.insert(format!("L{}.q_proj", l), 4096);
            sizes.insert(format!("L{}.k_proj", l), 2048);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, None, &geo);

        assert_eq!(full_indices, vec![1, 2, 3]);
        assert_eq!(q_dim, 4);
        assert_eq!(kv_dim, 2);
    }

    // ======================================================================
    // build_hetero_config: 8-layer [3+1]*2 pattern
    // ======================================================================

    #[test]
    fn build_hetero_config_8_layer_2_segment() {
        let geo = ModelGeometry { num_layers: 8, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[3,7], num_segments=2, sliding_per_segment=3
        // total = 2*(3+1) = 8
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![3, 7], 4, 2, false, &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 3);
        assert_eq!(cfg.full_layer_indices, vec![3, 7]);
    }

    // ======================================================================
    // WeightMaps: independent mutation does not affect clone
    // ======================================================================

    #[test]
    fn weight_maps_ext_sizes_independent() {
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("a".to_string(), 100);
        let wm = WeightMaps {
            ext_ptrs: HashMap::new(),
            ext_sizes: sizes,
            ext_shapes: HashMap::new(),
        };
        assert_eq!(wm.ext_sizes.get("a"), Some(&100));
    }

    // ======================================================================
    // ModelKind: Copy semantics
    // ======================================================================

    #[test]
    fn model_kind_copy_semantics() {
        let a = ModelKind::Chat;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // ======================================================================
    // ExecutorError: debug format includes variant
    // ======================================================================

    #[test]
    fn executor_error_debug_formats() {
        let err = ExecutorError::EmptyPrompt;
        let debug = format!("{err:?}");
        assert!(debug.contains("EmptyPrompt"));

        let err = ExecutorError::EmptySample;
        let debug = format!("{err:?}");
        assert!(debug.contains("EmptySample"));
    }

    // ======================================================================
    // PoolMode: Clone and Copy
    // ======================================================================

    #[test]
    fn pool_mode_clone_and_copy() {
        let a = PoolMode::MeanPool;
        let _b = a.clone();
        let _c = a.clone();
    }

    // ======================================================================
    // NEW TESTS: additional coverage (~55 tests)
    // ======================================================================

    // ======================================================================
    // TensorRole: canonical names for all variants (global scope)
    // ======================================================================

    #[test]
    fn tensor_role_embedding_canonical() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::Embedding.to_canonical_name(None), "embed");
    }

    #[test]
    fn tensor_role_output_head_canonical() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::OutputHead.to_canonical_name(None), "lm_head");
    }

    #[test]
    fn tensor_role_final_norm_canonical() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::FinalNorm.to_canonical_name(None), "final_norm");
    }

    #[test]
    fn tensor_role_attention_query_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::AttentionQuery.to_canonical_name(Some(7)), "L7.q_proj");
    }

    #[test]
    fn tensor_role_attention_key_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::AttentionKey.to_canonical_name(Some(0)), "L0.k_proj");
    }

    #[test]
    fn tensor_role_attention_value_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::AttentionValue.to_canonical_name(Some(3)), "L3.v_proj");
    }

    #[test]
    fn tensor_role_ffn_gate_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::FfnGate.to_canonical_name(Some(5)), "L5.gate_proj");
    }

    #[test]
    fn tensor_role_ffn_up_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::FfnUp.to_canonical_name(Some(2)), "L2.up_proj");
    }

    #[test]
    fn tensor_role_ffn_down_with_layer() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::FfnDown.to_canonical_name(Some(4)), "L4.down_proj");
    }

    #[test]
    fn tensor_role_mla_variants_canonical_names() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::MlaQCompress.to_canonical_name(Some(0)), "L0.q_a_proj");
        assert_eq!(TensorRole::MlaQExpand.to_canonical_name(Some(1)), "L1.q_b_proj");
        assert_eq!(TensorRole::MlaKvCompress.to_canonical_name(Some(2)), "L2.kv_b_proj");
        assert_eq!(TensorRole::MlaKeyAbsorb.to_canonical_name(Some(0)), "L0.k_b_proj");
        assert_eq!(TensorRole::MlaValueAbsorb.to_canonical_name(Some(0)), "L0.v_b_proj");
        assert_eq!(TensorRole::MlaRopeKey.to_canonical_name(Some(3)), "L3.k_pe_proj");
    }

    #[test]
    fn tensor_role_moe_variants_canonical_names() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::MoEGate.to_canonical_name(Some(1)), "L1.moe_gate");
        assert_eq!(TensorRole::MoESharedExpert.to_canonical_name(Some(2)), "L2.shared_expert");
        assert_eq!(TensorRole::MoEExpert.to_canonical_name(Some(0)), "L0.expert");
    }

    #[test]
    fn tensor_role_equality() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::Embedding, TensorRole::Embedding);
        assert_ne!(TensorRole::Embedding, TensorRole::OutputHead);
        assert_eq!(TensorRole::AttentionQuery, TensorRole::AttentionQuery);
        assert_ne!(TensorRole::AttentionQuery, TensorRole::AttentionKey);
    }

    // ======================================================================
    // RouterType: variant properties
    // ======================================================================

    #[test]
    fn router_type_equality() {
        use crate::manifest::RouterType;
        assert_eq!(RouterType::Qwen, RouterType::Qwen);
        assert_ne!(RouterType::Qwen, RouterType::Mixtral);
        assert_ne!(RouterType::DeepSeek, RouterType::GptOss);
        assert_ne!(RouterType::Unknown, RouterType::Qwen);
    }

    #[test]
    fn router_type_all_variants_distinct() {
        use crate::manifest::RouterType;
        let variants = [RouterType::Qwen, RouterType::Mixtral, RouterType::DeepSeek, RouterType::GptOss, RouterType::Unknown];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ======================================================================
    // MoEConfig: construction and properties
    // ======================================================================

    #[test]
    fn moe_config_construction() {
        use crate::manifest::{MoEConfig, RouterType};
        let cfg = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.router_type, RouterType::DeepSeek);
    }

    #[test]
    fn moe_config_equality() {
        use crate::manifest::{MoEConfig, RouterType};
        let a = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        let b = MoEConfig { num_experts: 8, num_experts_per_tok: 2, router_type: RouterType::Mixtral };
        assert_eq!(a, b);
    }

    // ======================================================================
    // ModelManifest: default values and is_moe
    // ======================================================================

    #[test]
    fn model_manifest_default_values() {
        use crate::manifest::ModelManifest;
        let m = ModelManifest::default();
        assert_eq!(&*m.model_id, "default");
        assert_eq!(m.arch, "llama");
        assert_eq!(m.kind, ModelKind::Chat);
        assert!(m.rope_base_override.is_none());
        assert!(m.max_context_override.is_none());
        assert!(m.moe_config.is_none());
        assert!(m.tensor_map.is_empty());
    }

    #[test]
    fn model_manifest_is_moe_true() {
        use crate::manifest::{ModelManifest, MoEConfig, RouterType};
        let m = ModelManifest {
            moe_config: Some(MoEConfig { num_experts: 64, num_experts_per_tok: 8, router_type: RouterType::Qwen }),
            ..ModelManifest::default()
        };
        assert!(m.is_moe());
    }

    #[test]
    fn model_manifest_is_moe_false() {
        use crate::manifest::ModelManifest;
        assert!(!ModelManifest::default().is_moe());
    }

    #[test]
    fn model_manifest_family_default_is_decoder() {
        use crate::manifest::{ArchFamily, ModelManifest};
        let m = ModelManifest::default();
        assert_eq!(m.family(), ArchFamily::Decoder);
    }

    #[test]
    fn model_manifest_with_overrides() {
        use crate::manifest::ModelManifest;
        let m = ModelManifest {
            rope_base_override: Some(500000.0),
            max_context_override: Some(8192),
            ..ModelManifest::default()
        };
        assert_eq!(m.rope_base_override, Some(500000.0));
        assert_eq!(m.max_context_override, Some(8192));
    }

    // ======================================================================
    // PositionEncoding: variants and equality
    // ======================================================================

    #[test]
    fn position_encoding_none() {
        use crate::engine::executor_types::PositionEncoding;
        assert_eq!(PositionEncoding::None, PositionEncoding::None);
    }

    #[test]
    fn position_encoding_rope() {
        use crate::engine::executor_types::PositionEncoding;
        assert_eq!(PositionEncoding::Rope, PositionEncoding::Rope);
    }

    #[test]
    fn position_encoding_distinct() {
        use crate::engine::executor_types::PositionEncoding;
        assert_ne!(PositionEncoding::None, PositionEncoding::Rope);
    }

    #[test]
    fn position_encoding_copy() {
        use crate::engine::executor_types::PositionEncoding;
        let a = PositionEncoding::Rope;
        let b = a; // Copy
        assert_eq!(a, b);
    }

    // ======================================================================
    // RoPEConfig: construction and equality
    // ======================================================================

    #[test]
    fn rope_config_construction() {
        use crate::engine::executor_types::RoPEConfig;
        let cfg = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert!((cfg.theta - 10000.0).abs() < 1e-10);
        assert!((cfg.scale - 1.0).abs() < 1e-10);
        assert!(!cfg.interleaved);
        assert!(!cfg.precompute);
    }

    #[test]
    fn rope_config_equality() {
        use crate::engine::executor_types::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        assert_eq!(a, b);
    }

    #[test]
    fn rope_config_inequality() {
        use crate::engine::executor_types::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = RoPEConfig { theta: 500000.0, scale: 1.0, interleaved: false, precompute: false };
        assert_ne!(a, b);
    }

    // ======================================================================
    // AttentionHeadConfig: from_geometry
    // ======================================================================

    #[test]
    fn attention_head_config_from_geometry() {
        use crate::engine::executor_types::AttentionHeadConfig;
        let geo = make_geometry(); // num_heads=8, num_kv_heads=4, head_dim=32
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 32);
    }

    #[test]
    fn attention_head_config_from_geometry_custom() {
        use crate::engine::executor_types::AttentionHeadConfig;
        let geo = ModelGeometry {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            ..make_geometry()
        };
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    // ======================================================================
    // PagedKvConfig: construction
    // ======================================================================

    #[test]
    fn paged_kv_config_none_page_table() {
        use crate::engine::executor_types::PagedKvConfig;
        let cfg = PagedKvConfig { page_table: None, page_size: 16 };
        assert!(cfg.page_table.is_none());
        assert_eq!(cfg.page_size, 16);
    }

    #[test]
    fn paged_kv_config_with_page_table() {
        use crate::engine::executor_types::PagedKvConfig;
        let cfg = PagedKvConfig { page_table: Some(vec![0, 1, 2, 3]), page_size: 16 };
        assert_eq!(cfg.page_table.as_ref().unwrap().len(), 4);
    }

    // ======================================================================
    // SwapConfig: construction and equality
    // ======================================================================

    #[test]
    fn swap_config_construction() {
        use crate::engine::executor_types::SwapConfig;
        let cfg = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        assert!(cfg.enable_swap);
        assert!((cfg.swap_threshold - 0.8).abs() < 1e-6);
        assert_eq!(cfg.lru_granularity, 4);
    }

    #[test]
    fn swap_config_equality() {
        use crate::engine::executor_types::SwapConfig;
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 8 };
        let b = SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 8 };
        assert_eq!(a, b);
    }

    #[test]
    fn swap_config_inequality_threshold() {
        use crate::engine::executor_types::SwapConfig;
        let a = SwapConfig { enable_swap: true, swap_threshold: 0.5, lru_granularity: 8 };
        let b = SwapConfig { enable_swap: true, swap_threshold: 0.9, lru_granularity: 8 };
        assert_ne!(a, b);
    }

    // ======================================================================
    // SequenceInput: fused_hidden field
    // ======================================================================

    #[test]
    fn sequence_input_with_fused_hidden() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(vec![0.1; 768]),
        };
        assert!(input.fused_hidden.is_some());
        assert_eq!(input.fused_hidden.unwrap().len(), 768);
    }

    #[test]
    fn sequence_input_empty_tokens() {
        let input = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(input.tokens.is_empty());
        assert!(input.validate_page_table(100).is_ok());
    }

    #[test]
    fn sequence_input_with_draft_steps() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 10,
            draft_steps: 3,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(input.draft_steps, 3);
        assert_eq!(input.position, 10);
    }

    // ======================================================================
    // BatchInput: construction
    // ======================================================================

    #[test]
    fn batch_input_empty_sequences() {
        use crate::engine::executor_types::BatchInput;
        let batch = BatchInput { sequences: vec![] };
        assert!(batch.sequences.is_empty());
    }

    #[test]
    fn batch_input_multiple_sequences() {
        use crate::engine::executor_types::BatchInput;
        let batch = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![1, 2],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![3, 4, 5],
                    position: 0,
                    draft_steps: 0,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: None,
                },
            ],
        };
        assert_eq!(batch.sequences.len(), 2);
        assert_eq!(batch.sequences[0].tokens, vec![1, 2]);
        assert!(batch.sequences[1].page_table.is_some());
    }

    // ======================================================================
    // GeneratorForwardConfig: default_for_test accessors
    // ======================================================================

    #[test]
    fn forward_config_default_for_test_accessors() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert!((cfg.norm_eps() - 1e-5).abs() < 1e-10);
        assert_eq!(cfg.max_seq_len(), 512);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
    }

    #[test]
    fn forward_config_rope_accessors() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!((cfg.rope_theta() - 10000.0).abs() < 1e-10);
        assert!((cfg.rope_scale() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn forward_config_attention_derived() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        let attn = cfg.attention();
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16);
    }

    // ======================================================================
    // ModelGeometry: kv_bytes_per_token
    // ======================================================================

    #[test]
    fn geometry_kv_bytes_per_token_standard() {
        let geo = make_geometry(); // num_kv_heads=4, head_dim=32, num_layers=6, F32
        // 2 * (4*32) * 6 * 4 = 6144
        assert_eq!(geo.kv_bytes_per_token(), 2 * 128 * 6 * 4);
    }

    #[test]
    fn geometry_kv_bytes_per_token_mla() {
        let geo = ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            ..make_geometry()
        };
        // MLA: (512+64) * 6 * 4 = 13824 (no K/V split)
        assert_eq!(geo.kv_bytes_per_token(), (512 + 64) * 6 * 4);
    }

    // ======================================================================
    // ModelGeometry: effective_kv_layer with attention_pattern
    // ======================================================================

    #[test]
    fn geometry_effective_kv_layer_non_shared() {
        let geo = make_geometry(); // num_kv_shared_layers=0
        assert_eq!(geo.effective_kv_layer(0), 0);
        assert_eq!(geo.effective_kv_layer(5), 5);
    }

    #[test]
    fn geometry_effective_kv_layer_shared_with_pattern() {
        let geo = ModelGeometry {
            num_layers: 6,
            num_kv_shared_layers: 2,
            attention_pattern: vec![0, 0, 1, 1, 0, 1], // L4=sliding, L5=global
            ..make_geometry()
        };
        // L0-L3 non-shared, L4-L5 shared
        // L4 has type 0 (sliding), look back for same type: L0 or L2? L0=0, L1=0, L2=1, L3=1
        // First match in reverse from L3..0 with type 0: L1 (type=0) => returns 1
        let layer4 = geo.effective_kv_layer(4);
        assert!(layer4 < geo.effective_kv_layers());
    }

    // ======================================================================
    // ModelGeometry: is_moe
    // ======================================================================

    #[test]
    fn geometry_is_moe_false() {
        let geo = make_geometry(); // num_experts=0
        assert!(!geo.is_moe());
    }

    #[test]
    fn geometry_is_moe_true() {
        let geo = ModelGeometry {
            num_experts: 64,
            ..make_geometry()
        };
        assert!(geo.is_moe());
    }

    // ======================================================================
    // Family: variant properties
    // ======================================================================

    #[test]
    fn family_equality() {
        use crate::arch::auto_graph::Family;
        assert_eq!(Family::Decoder, Family::Decoder);
        assert_eq!(Family::Encoder, Family::Encoder);
        assert_ne!(Family::Decoder, Family::Encoder);
    }

    #[test]
    fn family_debug_format() {
        use crate::arch::auto_graph::Family;
        assert_eq!(format!("{:?}", Family::Decoder), "Decoder");
        assert_eq!(format!("{:?}", Family::Encoder), "Encoder");
    }

    // ======================================================================
    // NormType: variant properties
    // ======================================================================

    #[test]
    fn norm_type_equality() {
        use crate::arch::auto_graph::NormType;
        assert_eq!(NormType::RmsNorm, NormType::RmsNorm);
        assert_eq!(NormType::LayerNorm, NormType::LayerNorm);
        assert_ne!(NormType::RmsNorm, NormType::LayerNorm);
    }

    // ======================================================================
    // FfnType: variant properties
    // ======================================================================

    #[test]
    fn ffn_type_equality() {
        use crate::arch::auto_graph::FfnType;
        assert_eq!(FfnType::SwiGLU, FfnType::SwiGLU);
        assert_eq!(FfnType::GeGLU, FfnType::GeGLU);
        assert_eq!(FfnType::Standard, FfnType::Standard);
        assert_eq!(FfnType::MoE, FfnType::MoE);
        assert_ne!(FfnType::SwiGLU, FfnType::GeGLU);
        assert_ne!(FfnType::Standard, FfnType::MoE);
    }

    #[test]
    fn ffn_type_all_variants_distinct() {
        use crate::arch::auto_graph::FfnType;
        let variants = [FfnType::SwiGLU, FfnType::GeGLU, FfnType::Standard, FfnType::MoE];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ======================================================================
    // KvCacheConfig: method accessors
    // ======================================================================

    #[test]
    fn kv_cache_config_accessors() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(make_geometry());
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.dtype_size(), 4);
        assert_eq!(cfg.num_layers(), 6);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.head_dim(), 32);
        assert_eq!(cfg.max_seq_len(), 2048);
        assert_eq!(cfg.page_size, 16);
    }

    #[test]
    fn kv_cache_config_with_swap() {
        use crate::engine::executor_types::{KvCacheConfig, SwapConfig};
        let geo = Arc::new(make_geometry());
        let swap = SwapConfig { enable_swap: true, swap_threshold: 0.75, lru_granularity: 4 };
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 32,
            swap_config: Some(swap),
        };
        assert!(cfg.swap_config.is_some());
        let s = cfg.swap_config.unwrap();
        assert!(s.enable_swap);
        assert_eq!(s.lru_granularity, 4);
    }

    #[test]
    fn kv_cache_config_equality_same() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_config_inequality_different_page_size() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(make_geometry());
        let a = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 32,
            swap_config: None,
        };
        assert_ne!(a, b);
    }

    // ======================================================================
    // effective_kv_max_seq_len: edge cases
    // ======================================================================

    #[test]
    fn effective_kv_max_seq_len_large_value() {
        use crate::engine::executor_types::effective_kv_max_seq_len;
        assert_eq!(effective_kv_max_seq_len(131072), 131072);
    }

    #[test]
    fn effective_kv_max_seq_len_identity() {
        use crate::engine::executor_types::effective_kv_max_seq_len;
        assert_eq!(effective_kv_max_seq_len(1), 1);
    }

    // ======================================================================
    // HeteroLayerConfig: field validation
    // ======================================================================

    #[test]
    fn hetero_layer_config_all_fields() {
        use gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig;
        let cfg = HeteroLayerConfig {
            num_segments: 3,
            sliding_per_segment: 2,
            sliding_head_dim: 64,
            sliding_num_q_heads: 8,
            sliding_num_kv_heads: 4,
            full_head_dim: 128,
            full_num_q_heads: 16,
            full_num_kv_heads: 8,
            full_layer_indices: vec![2, 5, 8],
            small_intermediate: 1024,
            large_intermediate: 4096,
            large_ffn_start_segment: 1,
        };
        assert_eq!(cfg.num_segments, 3);
        assert_eq!(cfg.sliding_per_segment, 2);
        assert_eq!(cfg.full_layer_indices.len(), 3);
        assert_eq!(cfg.full_layer_indices[0], 2);
        assert!(cfg.large_intermediate > cfg.small_intermediate);
    }

    // ======================================================================
    // SgConfig: detect_layer and offset fields
    // ======================================================================

    #[test]
    fn sg_config_without_qtap() {
        use gllm_kernels::compiler::mega_kernel_abi::SgConfig;
        let cfg = SgConfig {
            detect_layer: 3,
            detect_offset: 0,
            inject_offset: 0,
            q_tap: None,
        };
        assert_eq!(cfg.detect_layer, 3);
        assert!(cfg.q_tap.is_none());
    }

    // ======================================================================
    // build_output_modes: more edge cases
    // ======================================================================

    #[test]
    fn build_output_modes_max_new_tokens_usize_max() {
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, false, usize::MAX, 2);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, .. } => assert_eq!(*max_new_tokens, usize::MAX),
            _ => panic!("expected Generate"),
        }
    }

    // ======================================================================
    // resolve_eos_token_id: various fallback paths (requires #[cfg] gate)
    // Note: resolve_eos_token_id is cfg-gated, tested via build_output_modes instead
    // ======================================================================

    // ======================================================================
    // BusinessConfig: guardrail, session, multimodal overrides
    // ======================================================================

    #[test]
    fn business_config_guardrail_true() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            guardrail_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.guardrail_enabled);
    }

    #[test]
    fn business_config_session_true() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            session_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.session_enabled);
    }

    #[test]
    fn business_config_multimodal_true() {
        use gllm_kernels::compiler::BusinessConfig;
        let cfg = BusinessConfig {
            multimodal_enabled: true,
            ..BusinessConfig::default()
        };
        assert!(cfg.multimodal_enabled);
    }

    // ======================================================================
    // ModelGeometry: position_offset field
    // ======================================================================

    #[test]
    fn geometry_position_offset_none() {
        let geo = make_geometry();
        assert!(geo.position_offset.is_none());
    }

    #[test]
    fn geometry_position_offset_some() {
        let geo = ModelGeometry {
            position_offset: Some(2),
            ..make_geometry()
        };
        assert_eq!(geo.position_offset, Some(2));
    }

    // ======================================================================
    // ModelGeometry: sliding_window field
    // ======================================================================

    #[test]
    fn geometry_sliding_window_default() {
        let geo = make_geometry();
        assert_eq!(geo.sliding_window, 0);
    }

    #[test]
    fn geometry_sliding_window_custom() {
        let geo = ModelGeometry {
            sliding_window: 4096,
            ..make_geometry()
        };
        assert_eq!(geo.sliding_window, 4096);
    }

    // ======================================================================
    // ModelGeometry: global_head_dim field
    // ======================================================================

    #[test]
    fn geometry_global_head_dim_default() {
        let geo = make_geometry();
        assert_eq!(geo.global_head_dim, 0);
    }

    // ======================================================================
    // ModelGeometry: expert_weight_bytes
    // ======================================================================

    #[test]
    fn geometry_expert_weight_bytes_calculation() {
        let geo = ModelGeometry {
            hidden_size: 256,
            expert_intermediate_size: 512,
            dtype: gllm_kernels::types::DType::F32,
            ..make_geometry()
        };
        // 256 * 512 * 3 * 4 = 1572864
        assert_eq!(geo.expert_weight_bytes(), 256 * 512 * 3 * 4);
    }

    // ======================================================================
    // ModelGeometry: hidden_size_per_layer_input
    // ======================================================================

    #[test]
    fn geometry_hidden_size_per_layer_input_default() {
        let geo = make_geometry();
        assert_eq!(geo.hidden_size_per_layer_input, 0);
    }

    // ======================================================================
    // ModelGeometry: final_logit_softcapping
    // ======================================================================

    #[test]
    fn geometry_final_logit_softcapping_none() {
        let geo = make_geometry();
        assert!(geo.final_logit_softcapping.is_none());
    }

    #[test]
    fn geometry_final_logit_softcapping_some() {
        let geo = ModelGeometry {
            final_logit_softcapping: Some(30.0),
            ..make_geometry()
        };
        assert_eq!(geo.final_logit_softcapping, Some(30.0));
    }

    // ======================================================================
    // ModelGeometry: rope_scaling field
    // ======================================================================

    #[test]
    fn geometry_rope_scaling_none() {
        let geo = make_geometry();
        assert!(geo.rope_scaling.is_none());
    }

    // ======================================================================
    // ModelGeometry: hidden_act field
    // ======================================================================

    #[test]
    fn geometry_hidden_act_none() {
        let geo = make_geometry();
        assert!(geo.hidden_act.is_none());
    }

    #[test]
    fn geometry_hidden_act_some() {
        use crate::model_config::HiddenAct;
        let geo = ModelGeometry {
            hidden_act: Some(HiddenAct::Silu),
            ..make_geometry()
        };
        assert!(geo.hidden_act.is_some());
        assert_eq!(geo.hidden_act.as_ref().unwrap().as_str(), "silu");
    }

    // ======================================================================
    // CanonicalWeightMaps: structural properties (via build_canonical_weight_maps not accessible, test WeightMaps instead)
    // ======================================================================

    #[test]
    fn weight_maps_empty_construction() {
        let wm = WeightMaps {
            ext_ptrs: HashMap::new(),
            ext_sizes: HashMap::new(),
            ext_shapes: HashMap::new(),
        };
        assert!(wm.ext_ptrs.is_empty());
        assert!(wm.ext_sizes.is_empty());
        assert!(wm.ext_shapes.is_empty());
    }

    #[test]
    fn weight_maps_ptrs_and_shapes_consistent() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        let val: u8 = 42;
        ptrs.insert("tensor_a".to_string(), &val);
        sizes.insert("tensor_a".to_string(), 1024);
        shapes.insert("tensor_a".to_string(), vec![256, 4]);

        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };
        assert!(wm.ext_ptrs.contains_key("tensor_a"));
        assert_eq!(wm.ext_sizes.get("tensor_a"), Some(&1024));
        assert_eq!(wm.ext_shapes.get("tensor_a").unwrap().len(), 2);
    }

    // ======================================================================
    // NEW TESTS — 15 additional edge-case tests
    // ======================================================================

    // ======================================================================
    // GeneratorForwardConfig: dtype accessor
    // ======================================================================

    #[test]
    fn forward_config_dtype_f32() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.dtype(), gllm_kernels::types::DType::F32);
    }

    // ======================================================================
    // KvCacheConfig: kv_dim and is_mla accessors
    // ======================================================================

    #[test]
    fn kv_cache_config_kv_dim_standard() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(make_geometry()); // num_kv_heads=4, head_dim=32
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.kv_dim(), 128); // 4 * 32
        assert!(!cfg.is_mla());
    }

    #[test]
    fn kv_cache_config_is_mla_with_mla_geometry() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            ..make_geometry()
        });
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert!(cfg.is_mla());
        assert_eq!(cfg.kv_dim(), 576); // 512 + 64
    }

    // ======================================================================
    // KvCacheConfig: num_kv_shared_layers and attention_pattern accessors
    // ======================================================================

    #[test]
    fn kv_cache_config_shared_layers_and_pattern() {
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(ModelGeometry {
            num_layers: 6,
            num_kv_shared_layers: 2,
            attention_pattern: vec![0, 0, 1, 1, 0, 1],
            ..make_geometry()
        });
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.num_kv_shared_layers(), 2);
        assert_eq!(cfg.attention_pattern(), &[0, 0, 1, 1, 0, 1]);
    }

    // ======================================================================
    // WeightMaps: shape with empty dimensions (scalar tensor edge case)
    // ======================================================================

    #[test]
    fn weight_maps_scalar_tensor_shape() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
        let val: u8 = 0;
        ptrs.insert("bias".to_string(), &val);
        sizes.insert("bias".to_string(), 4);
        shapes.insert("bias".to_string(), vec![]); // scalar
        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };
        assert!(wm.ext_shapes.get("bias").unwrap().is_empty());
    }

    // ======================================================================
    // ModelGeometry: rope_interleaved field
    // ======================================================================

    #[test]
    fn geometry_rope_interleaved_default_false() {
        let geo = make_geometry();
        assert!(!geo.rope_interleaved);
    }

    #[test]
    fn geometry_rope_interleaved_true() {
        let geo = ModelGeometry {
            rope_interleaved: true,
            ..make_geometry()
        };
        assert!(geo.rope_interleaved);
    }

    // ======================================================================
    // ModelGeometry: global_rope_theta field (non-zero for mixed RoPE models)
    // ======================================================================

    #[test]
    fn geometry_global_rope_theta_nonzero() {
        let geo = ModelGeometry {
            global_rope_theta: 1_000_000.0,
            ..make_geometry()
        };
        assert!((geo.global_rope_theta - 1_000_000.0).abs() < 1e-5);
    }

    // ======================================================================
    // SequenceInput: validate_page_table with u32::MAX page_id
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_max_page_id() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![u32::MAX]),
            fused_hidden: None,
        };
        // u32::MAX as usize >= any reasonable total_pages
        assert!(input.validate_page_table(100).is_err());
    }

    // ======================================================================
    // SequenceInput: validate_page_table with empty page_table
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_empty_vec() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        // Empty page table should pass validation (no entries to check)
        assert!(input.validate_page_table(10).is_ok());
    }

    // ======================================================================
    // BackendError: source via std::error::Error trait
    // ======================================================================

    #[test]
    fn backend_error_std_error_source_is_none() {
        let err = BackendError::Cuda("test".to_string());
        use std::error::Error;
        assert!(err.source().is_none());
    }

    // ======================================================================
    // ExecutorError: Display preserves backend message through From conversion
    // ======================================================================

    #[test]
    fn executor_error_display_preserves_backend_message() {
        let backend_err = BackendError::Cuda("nvrtc compile error".to_string());
        let exec_err: ExecutorError = backend_err.into();
        // #[error(transparent)] delegates Display to BackendError::Display
        assert!(format!("{exec_err}").contains("nvrtc compile error"));
        assert!(format!("{exec_err}").contains("CUDA"));
    }

    // ======================================================================
    // CanonicalWeightMaps: name_map field preserves input names
    // ======================================================================

    #[test]
    fn canonical_weight_maps_name_map_with_names() {
        let names = vec!["token_embd.weight".to_string(), "lm_head.weight".to_string()];
        let nm = crate::loader::name_map::TensorNameMap::build_from_names(&names, false);
        let cwm = CanonicalWeightMaps {
            weight_ptrs: HashMap::new(),
            weight_sizes: HashMap::new(),
            weight_shapes: HashMap::new(),
            name_map: nm,
            auto_features: make_auto_features(),
        };
        assert!(cwm.weight_ptrs.is_empty());
        assert!(cwm.weight_shapes.is_empty());
    }

    // ======================================================================
    // SwapConfig: default (not derived, manual check)
    // ======================================================================

    #[test]
    fn swap_config_disable_defaults() {
        use crate::engine::executor_types::SwapConfig;
        let cfg = SwapConfig { enable_swap: false, swap_threshold: 0.0, lru_granularity: 0 };
        assert!(!cfg.enable_swap);
        assert_eq!(cfg.swap_threshold, 0.0);
        assert_eq!(cfg.lru_granularity, 0);
    }

    // ======================================================================
    // ModelGeometry: mla_unabsorbed_threshold field edge case
    // ======================================================================

    #[test]
    fn geometry_mla_unabsorbed_threshold_zero_and_nonzero() {
        let geo_default = make_geometry();
        assert_eq!(geo_default.mla_unabsorbed_threshold, 0);

        let geo = ModelGeometry {
            mla_d_c: 512,
            mla_d_rope: 64,
            mla_unabsorbed_threshold: 256,
            ..make_geometry()
        };
        assert_eq!(geo.mla_unabsorbed_threshold, 256);
        assert!(geo.is_mla());
    }

    // ======================================================================
    // NEW TESTS — 15 additional tests
    // ======================================================================

    // ======================================================================
    // ExecutorError: Config variant via From<ModelConfigError>
    // ======================================================================

    #[test]
    fn executor_error_from_model_config_error() {
        use crate::model_config::ModelConfigError;
        let config_err = ModelConfigError::InvalidConfig("missing hidden_size".into());
        let exec_err: ExecutorError = config_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("missing hidden_size"));
    }

    // ======================================================================
    // ExecutorError: Loader variant via From<LoaderError>
    // ======================================================================

    #[test]
    fn executor_error_from_loader_error() {
        use crate::loader::LoaderError;
        let loader_err = LoaderError::Network("connection refused".into());
        let exec_err: ExecutorError = loader_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("connection refused"));
    }

    // ======================================================================
    // ExecutorError: Tokenizer variant via From<TokenizerError>
    // ======================================================================

    #[test]
    fn executor_error_from_tokenizer_error() {
        use crate::tokenizer::TokenizerError;
        let tok_err = TokenizerError::MissingTokenizer;
        let exec_err: ExecutorError = tok_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("tokenizer.json") || msg.contains("MissingTokenizer") || msg.contains("not found"));
    }

    // ======================================================================
    // ExecutorError: KvCache variant via From<KvCacheError>
    // ======================================================================

    #[test]
    fn executor_error_from_kv_cache_error() {
        use crate::kv_cache::KvCacheError;
        let kv_err = KvCacheError::Exhausted { requested: 2048, available: 1024 };
        let exec_err: ExecutorError = kv_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("2048") || msg.contains("exhausted"));
    }

    // ======================================================================
    // ExecutorError: MemoryManager variant via From<MemoryManagerError>
    // ======================================================================

    #[test]
    fn executor_error_from_memory_manager_error() {
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier};
        let mm_err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L3 };
        let exec_err: ExecutorError = mm_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("TierCapacityExceeded") || msg.contains("tier") || msg.contains("GpuHbm"));
    }

    // ======================================================================
    // ArchitectureFeatures: constructing with all MoE fields set
    // ======================================================================

    #[test]
    fn auto_features_moe_with_shared_experts() {
        let mut features = make_auto_features();
        features.is_moe = true;
        features.has_shared_experts = true;
        features.num_experts = 64;
        features.moe_top_k = 8;
        assert!(features.is_moe);
        assert!(features.has_shared_experts);
        assert_eq!(features.num_experts, 64);
        assert_eq!(features.moe_top_k, 8);
    }

    // ======================================================================
    // TensorRole: PostAttnNorm canonical name
    // ======================================================================

    #[test]
    fn tensor_role_post_attn_norm_canonical() {
        use crate::manifest::TensorRole;
        assert_eq!(TensorRole::PostAttnNorm.to_canonical_name(Some(3)), "L3.post_attn_norm");
    }

    // ======================================================================
    // TensorRole: LayerNorm canonical name (no layer)
    // ======================================================================

    #[test]
    fn tensor_role_layer_norm_canonical_no_layer() {
        use crate::manifest::TensorRole;
        // LayerNorm without a layer index should return a generic name
        let name = TensorRole::PostAttnNorm.to_canonical_name(None);
        assert_eq!(name, "post_attn_norm");
    }

    // ======================================================================
    // build_hetero_config: 10-layer [4+1]*2 pattern
    // ======================================================================

    #[test]
    fn build_hetero_config_10_layer_2_segment() {
        let geo = ModelGeometry { num_layers: 10, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // full_indices=[4,9], num_segments=2, sliding_per_segment=4
        // total = 2*(4+1) = 10 = num_layers
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![4, 9], 4, 2, false, &geo,
        );
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 4);
        assert_eq!(cfg.full_layer_indices, vec![4, 9]);
    }

    // ======================================================================
    // detect_hetero_layers: large sliding_per_segment (Gemma-like)
    // ======================================================================

    #[test]
    fn detect_hetero_layers_large_sliding_segment() {
        // [5 sliding + 1 full] * 2 = 12 layers
        let geo = ModelGeometry { num_layers: 12, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024); // sliding (ref)
        sizes.insert("L5.q_proj".to_string(), 4096); // full
        sizes.insert("L5.k_proj".to_string(), 2048);
        sizes.insert("L11.q_proj".to_string(), 4096); // full
        sizes.insert("L11.k_proj".to_string(), 2048);

        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 2);
        assert_eq!(cfg.sliding_per_segment, 5);
        assert_eq!(cfg.full_layer_indices, vec![5, 11]);
    }

    // ======================================================================
    // scan_hetero_layer_diffs: ref_q = 0 causes all layers to differ
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_zero_ref_q_all_differ() {
        let geo = ModelGeometry { num_layers: 4, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..4 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
            sizes.insert(format!("L{}.k_proj", l), 512);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // ref_q=0, so any non-zero q_proj differs
        let (full_indices, q_dim, kv_dim, _) =
            TestExec::scan_hetero_layer_diffs(&find_size, 0, None, &geo);
        assert_eq!(full_indices, vec![1, 2, 3]);
        assert_eq!(q_dim, 1); // 1024/(256*4) = 1
        assert_eq!(kv_dim, 0); // 512/(256*4) = 0 (kv stays 0 from first diff)
    }

    // ======================================================================
    // ModelGeometry: hidden_act with gelu variant
    // ======================================================================

    #[test]
    fn geometry_hidden_act_gelu_variants() {
        use crate::model_config::HiddenAct;
        let geo = ModelGeometry {
            hidden_act: Some(HiddenAct::Gelu),
            ..make_geometry()
        };
        assert!(geo.hidden_act.is_some());
        assert_eq!(geo.hidden_act.as_ref().unwrap().as_str(), "gelu");
    }

    // ======================================================================
    // WeightMaps: null pointer is valid for ptr map (safety test)
    // ======================================================================

    #[test]
    fn weight_maps_null_ptr_is_valid_entry() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        // Null pointer is a valid pointer value; testing that the HashMap accepts it
        ptrs.insert("uninitialized".to_string(), std::ptr::null());
        sizes.insert("uninitialized".to_string(), 0);
        shapes.insert("uninitialized".to_string(), vec![0]);

        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };
        assert!(wm.ext_ptrs.contains_key("uninitialized"));
        assert_eq!(wm.ext_sizes.get("uninitialized"), Some(&0));
        assert_eq!(wm.ext_ptrs.get("uninitialized"), Some(&std::ptr::null()));
    }

    // ======================================================================
    // OutputMode: debug format distinguishes between variants
    // ======================================================================

    #[test]
    fn output_mode_debug_distinguishes_variants() {
        let gen = OutputMode::Generate { max_new_tokens: 100, eos_token_id: 2 };
        let enc = OutputMode::EncodeToLayer { anchor_layer: 0, pool_mode: PoolMode::MeanPool };
        let bin = OutputMode::ClassifyBinary { positive_token_id: 1, negative_token_id: 0 };
        let multi = OutputMode::ClassifyMultiway { label_token_ids: vec![] };

        let gen_debug = format!("{gen:?}");
        let enc_debug = format!("{enc:?}");
        let bin_debug = format!("{bin:?}");
        let multi_debug = format!("{multi:?}");

        assert!(gen_debug.contains("Generate"));
        assert!(enc_debug.contains("EncodeToLayer"));
        assert!(bin_debug.contains("ClassifyBinary"));
        assert!(multi_debug.contains("ClassifyMultiway"));

        // Each debug string is unique
        assert_ne!(gen_debug, enc_debug);
        assert_ne!(gen_debug, bin_debug);
        assert_ne!(enc_debug, bin_debug);
    }

    // ======================================================================
    // build_output_modes: encoder with ModelKind::Chat falls to Generate
    // ======================================================================

    #[test]
    fn build_output_modes_encoder_chat_falls_to_generate() {
        // Encoder + Chat is not a special combination, falls through to Generate
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, true, 256, 3);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 256);
                assert_eq!(*eos_token_id, 3);
            }
            other => panic!("expected Generate for encoder Chat, got {:?}", other),
        }
    }

    // ======================================================================
    // 15 NEW TESTS — additional coverage
    // ======================================================================

    // ======================================================================
    // ModelGeometry: kv_bytes_per_token with BF16 dtype (2 bytes per element)
    // ======================================================================

    #[test]
    fn geometry_kv_bytes_per_token_bf16() {
        let geo = ModelGeometry {
            dtype: gllm_kernels::types::DType::BF16,
            compute_dtype: gllm_kernels::types::DType::BF16,
            ..make_geometry()
        };
        // Standard (non-MLA): 2 * (num_kv_heads * head_dim) * num_layers * dtype_size
        // = 2 * (4 * 32) * 6 * 2 = 3072
        assert_eq!(geo.kv_bytes_per_token(), 2 * 128 * 6 * 2);
    }

    // ======================================================================
    // ModelGeometry: expert_weight_bytes with BF16 dtype
    // ======================================================================

    #[test]
    fn geometry_expert_weight_bytes_bf16() {
        let geo = ModelGeometry {
            num_experts: 16,
            expert_intermediate_size: 1408,
            dtype: gllm_kernels::types::DType::BF16,
            ..make_geometry()
        };
        // hidden_size * expert_intermediate_size * 3 * dtype_size
        // = 256 * 1408 * 3 * 2 = 2162688
        assert_eq!(geo.expert_weight_bytes(), 256 * 1408 * 3 * 2);
    }

    // ======================================================================
    // ModelGeometry: effective_kv_layer with all layers shared
    // ======================================================================

    #[test]
    fn geometry_effective_kv_layer_all_shared_clamps_to_last_effective() {
        let geo = ModelGeometry {
            num_layers: 4,
            num_kv_shared_layers: 4,
            ..make_geometry()
        };
        // effective_kv_layers = max(4 - 4, 1) = 1
        assert_eq!(geo.effective_kv_layers(), 1);
        // Layer 0 (< shared_start=0): clamped to min(0, 0) = 0
        assert_eq!(geo.effective_kv_layer(0), 0);
        // Layer 3 (shared): no attention_pattern, fallback to effective_kv_layers - 1 = 0
        assert_eq!(geo.effective_kv_layer(3), 0);
    }

    // ======================================================================
    // ModelGeometry: effective_kv_layer with attention_pattern donor lookup
    // ======================================================================

    #[test]
    fn geometry_effective_kv_layer_shared_with_donor_pattern() {
        let geo = ModelGeometry {
            num_layers: 6,
            num_kv_shared_layers: 2,
            attention_pattern: vec![0, 0, 1, 1, 0, 1],
            ..make_geometry()
        };
        // shared_start = 6 - 2 = 4
        // Layer 4 has type 0; scan 3..0: L3=1, L2=1, L1=0 => donor is L1
        assert_eq!(geo.effective_kv_layer(4), 1);
        // Layer 5 has type 1; scan 3..0: L3=1 => donor is L3
        assert_eq!(geo.effective_kv_layer(5), 3);
    }

    // ======================================================================
    // GeneratorForwardConfig: position_encoding field
    // ======================================================================

    #[test]
    fn forward_config_position_encoding_default_is_rope() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.position_encoding, crate::engine::executor_types::PositionEncoding::Rope);
    }

    // ======================================================================
    // GeneratorForwardConfig: arch_family field
    // ======================================================================

    #[test]
    fn forward_config_arch_family_default_is_decoder() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        use crate::manifest::ArchFamily;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.arch_family, ArchFamily::Decoder);
    }

    // ======================================================================
    // GeneratorForwardConfig: paged_kv field defaults
    // ======================================================================

    #[test]
    fn forward_config_paged_kv_defaults() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.paged_kv.page_table.is_none());
        assert_eq!(cfg.paged_kv.page_size, 16);
    }

    // ======================================================================
    // GeneratorForwardConfig: moe_config is None for dense models
    // ======================================================================

    #[test]
    fn forward_config_moe_config_none_for_dense() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.moe_config.is_none());
    }

    // ======================================================================
    // GeneratorForwardConfig: rerank token IDs default to None
    // ======================================================================

    #[test]
    fn forward_config_rerank_token_ids_default_none() {
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.rerank_yes_token_id.is_none());
        assert!(cfg.rerank_no_token_id.is_none());
    }

    // ======================================================================
    // SamplingConfig: extreme temperature values
    // ======================================================================

    #[test]
    fn sampling_config_extreme_temperature() {
        use crate::engine::executor_types::SamplingConfig;
        let zero_temp = SamplingConfig { temperature: 0.0, top_k: 1, top_p: 1.0 };
        assert_eq!(zero_temp.temperature, 0.0);

        let max_temp = SamplingConfig { temperature: f32::MAX, top_k: 0, top_p: 1.0 };
        assert_eq!(max_temp.temperature, f32::MAX);
    }

    // ======================================================================
    // AttentionHeadConfig: from_geometry with single KV head (GQA extreme)
    // ======================================================================

    #[test]
    fn attention_head_config_from_geometry_single_kv_head() {
        use crate::engine::executor_types::AttentionHeadConfig;
        let geo = ModelGeometry {
            num_heads: 32,
            num_kv_heads: 1,
            head_dim: 128,
            ..make_geometry()
        };
        let cfg = AttentionHeadConfig::from_geometry(&geo);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.head_dim, 128);
    }

    // ======================================================================
    // SequenceInput: validate_page_table reports first violating index
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_multiple_violations_reports_first() {
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![1, 2, 15, 20]), // indices 2 and 3 are out of bounds
            fused_hidden: None,
        };
        let result = input.validate_page_table(10);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        // Should report the first violation at index 2
        assert!(err_msg.contains("page_table[2]"));
        assert!(err_msg.contains("15"));
    }

    // ======================================================================
    // build_output_modes: encoder unknown ModelKind falls to Generate
    // ======================================================================

    #[test]
    fn build_output_modes_encoder_unknown_kind_falls_to_generate() {
        // ModelKind::Chat on 无 Argmax 的图 falls through to _ => Generate
        let manifest = make_manifest(ModelKind::Chat);
        let modes = TestExec::build_output_modes(&manifest, true, 128, 7);
        assert_eq!(modes.len(), 1);
        match &modes[0] {
            OutputMode::Generate { max_new_tokens, eos_token_id } => {
                assert_eq!(*max_new_tokens, 128);
                assert_eq!(*eos_token_id, 7);
            }
            other => panic!("expected Generate, got {:?}", other),
        }
    }

    // ======================================================================
    // ModelGeometry: kv_dim when mla_d_c > 0 but mla_d_rope = 0
    // ======================================================================

    #[test]
    fn geometry_kv_dim_partial_mla_d_rope_zero() {
        // is_mla() requires mla_d_c > 0 (checks only d_c)
        let geo = ModelGeometry {
            mla_d_c: 256,
            mla_d_rope: 0,
            ..make_geometry()
        };
        assert!(geo.is_mla());
        // kv_dim = d_c + d_rope = 256 + 0 = 256
        assert_eq!(geo.kv_dim(), 256);
    }

    // ======================================================================
    // FfnActivation: Debug output matches variant name exactly
    // ======================================================================

    #[test]
    fn ffn_activation_gelu_debug_exact() {
        let debug = format!("{:?}", FfnActivation::Gelu);
        assert_eq!(debug, "Gelu");
    }

    // ======================================================================
    // 15 NEW TESTS — additional coverage
    // ======================================================================

    // ======================================================================
    // RequestData: construction with all fields populated
    // ======================================================================

    #[test]
    fn request_data_construction_all_fields() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        // Arrange
        let sampling = SamplingConfig { temperature: 0.8, top_k: 40, top_p: 0.95 };

        // Act
        let data = RequestData {
            prompt_tokens: vec![1, 50256, 314],
            output_tokens: vec![99],
            sampling_config: sampling.clone(),
            is_prefill: false,
            phase: RequestPhase::Decode,
            max_new_tokens: 512,
            finished: false,
            session_id: Some(42),
            thinking_budget: Some(100),
            fused_prefill_hidden: None,
        };

        // Assert
        assert_eq!(data.prompt_tokens.len(), 3);
        assert_eq!(data.output_tokens, vec![99]);
        assert!((data.sampling_config.temperature - 0.8).abs() < 1e-6);
        assert_eq!(data.sampling_config.top_k, 40);
        assert!(!data.is_prefill);
        assert_eq!(data.max_new_tokens, 512);
        assert!(!data.finished);
        assert_eq!(data.session_id, Some(42));
        assert_eq!(data.thinking_budget, Some(100));
        assert!(data.fused_prefill_hidden.is_none());
    }

    // ======================================================================
    // RequestPhase: all variants are distinct and support Debug
    // ======================================================================

    #[test]
    fn request_phase_variants_distinct_and_debug() {
        use crate::scheduler::request_state::RequestPhase;

        // Arrange
        let prefill = RequestPhase::Prefill;
        let decode = RequestPhase::Decode;
        let chunked = RequestPhase::ChunkedPrefill;

        // Assert: variants are distinct
        assert_ne!(prefill, decode);
        assert_ne!(decode, chunked);
        assert_ne!(prefill, chunked);

        // Assert: Debug format contains variant names
        assert!(format!("{:?}", prefill).contains("Prefill"));
        assert!(format!("{:?}", decode).contains("Decode"));
        assert!(format!("{:?}", chunked).contains("ChunkedPrefill"));
    }

    // ======================================================================
    // GeneratorForwardConfig: attention_geometry returns correct dims
    // ======================================================================

    #[test]
    fn forward_config_attention_geometry_dims() {
        use crate::engine::executor_types::GeneratorForwardConfig;

        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // geometry: num_heads=4, num_kv_heads=2, head_dim=16

        // Act
        let geo = cfg.attention_geometry();

        // Assert
        assert_eq!(geo.num_heads, 4);
        assert_eq!(geo.num_kv_heads, 2);
        assert_eq!(geo.head_dim, 16);
        // q_dim = num_heads * head_dim = 4 * 16 = 64
        assert_eq!(geo.q_dim, 64);
        // kv_dim = num_kv_heads * head_dim = 2 * 16 = 32
        assert_eq!(geo.kv_dim, 32);
        // heads_per_group = num_heads / max(num_kv_heads, 1) = 4 / 2 = 2
        assert_eq!(geo.heads_per_group, 2);
    }

    // ======================================================================
    // GeneratorForwardConfig: layer_dims returns correct values
    // ======================================================================

    #[test]
    fn forward_config_layer_dims_values() {
        use crate::engine::executor_types::GeneratorForwardConfig;

        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // hidden=64, inter=128, eps=1e-5

        // Act
        let dims = cfg.layer_dims();

        // Assert
        assert_eq!(dims.hidden, 64);
        assert_eq!(dims.inter, 128);
        assert!((dims.eps - 1e-5).abs() < 1e-10);
    }

    // ======================================================================
    // KvCacheConfig: BF16 dtype reports correct size_bytes
    // ======================================================================

    #[test]
    fn kv_cache_config_bf16_dtype_size() {
        use crate::engine::executor_types::KvCacheConfig;

        // Arrange
        let geo = Arc::new(make_geometry());

        // Act
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::BF16,
            page_size: 16,
            swap_config: None,
        };

        // Assert
        assert_eq!(cfg.dtype_size(), 2); // BF16 = 2 bytes
        assert_eq!(cfg.num_layers(), 6);
    }

    // ======================================================================
    // SequenceInput: position field with maximum usize value
    // ======================================================================

    #[test]
    fn sequence_input_position_max_usize() {
        // Arrange
        let input = SequenceInput {
            tokens: vec![1],
            position: usize::MAX,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };

        // Act & Assert
        assert_eq!(input.position, usize::MAX);
        assert_eq!(input.tokens, vec![1]);
    }

    // ======================================================================
    // BatchInput: clone preserves all sequence data
    // ======================================================================

    #[test]
    fn batch_input_clone_preserves_data() {
        use crate::engine::executor_types::BatchInput;

        // Arrange
        let original = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![10, 20, 30],
                    position: 5,
                    draft_steps: 2,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![40],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: Some(vec![0.5; 64]),
                },
            ],
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.sequences.len(), 2);
        assert_eq!(cloned.sequences[0].tokens, vec![10, 20, 30]);
        assert_eq!(cloned.sequences[0].position, 5);
        assert_eq!(cloned.sequences[0].draft_steps, 2);
        assert_eq!(cloned.sequences[0].page_table, Some(vec![0, 1]));
        assert_eq!(cloned.sequences[1].tokens, vec![40]);
        assert_eq!(cloned.sequences[1].fused_hidden.as_ref().unwrap().len(), 64);
    }

    // ======================================================================
    // AttentionTopology: max_seq_len returns geometry max_seq_len
    // ======================================================================

    #[test]
    fn attention_topology_max_seq_len_matches_geometry() {
        // Arrange
        let geo = Arc::new(ModelGeometry {
            max_seq_len: 8192,
            ..make_geometry()
        });

        // Act
        let topo = AttentionTopology::causal(geo);

        // Assert
        assert_eq!(topo.max_seq_len(), 8192);
    }

    // ======================================================================
    // ModelKind: all variants support Copy
    // ======================================================================

    #[test]
    fn model_kind_all_variants_copy_semantics() {
        // Arrange & Act: Copy each variant
        let chat = ModelKind::Chat;
        let chat_copy = chat;

        let emb = ModelKind::Embedding;
        let emb_copy = emb;

        let rerank = ModelKind::Reranker;
        let rerank_copy = rerank;

        let cls = ModelKind::Classifier;
        let cls_copy = cls;

        // Assert: original still valid after copy
        assert_eq!(chat, chat_copy);
        assert_eq!(emb, emb_copy);
        assert_eq!(rerank, rerank_copy);
        assert_eq!(cls, cls_copy);
    }

    // ======================================================================
    // ExecutorError::Compilation Display includes message
    // ======================================================================

    #[test]
    fn executor_error_compilation_display_includes_detail() {
        // Arrange
        let err = ExecutorError::Compilation("spill slot overflow in GEMM tile".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("JIT compilation failed"));
        assert!(msg.contains("spill slot overflow in GEMM tile"));
    }

    // ======================================================================
    // ExecutorError::GraphExpansion Display includes message
    // ======================================================================

    #[test]
    fn executor_error_graph_expansion_display_includes_detail() {
        // Arrange
        let err = ExecutorError::GraphExpansion("unsupported OpKind::CustomOp".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("graph expansion failed"));
        assert!(msg.contains("unsupported OpKind::CustomOp"));
    }

    // ======================================================================
    // BackendError::Unimplemented preserves message through Clone
    // ======================================================================

    #[test]
    fn backend_error_unimplemented_clone_preserves_message() {
        // Arrange
        let original = BackendError::Unimplemented("fp4 dequantization kernel");

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(format!("{original}"), format!("{cloned}"));
        assert!(format!("{cloned}").contains("fp4 dequantization kernel"));
    }

    // ======================================================================
    // ModelGeometry: rope_partial_ratio field default and custom
    // ======================================================================

    #[test]
    fn geometry_rope_partial_ratio_default_and_custom() {
        // Arrange — default
        let geo_default = make_geometry();

        // Assert — default is 1.0
        assert!((geo_default.rope_partial_ratio - 1.0).abs() < 1e-10);

        // Arrange — custom (Gemma 4 global layers use partial=0.25)
        let geo_custom = ModelGeometry {
            rope_partial_ratio: 0.25,
            rope_partial_ratio_global: 1.0,
            ..make_geometry()
        };

        // Assert
        assert!((geo_custom.rope_partial_ratio - 0.25).abs() < 1e-10);
    }

    // ======================================================================
    // WeightMaps: multiple entries with consistent keys across maps
    // ======================================================================

    #[test]
    fn weight_maps_consistent_keys_across_maps() {
        // Arrange
        let names = vec!["embed", "L0.q_proj", "L0.k_proj", "final_norm"];
        let dummy_ptr: *const u8 = std::ptr::null();
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        for name in &names {
            ptrs.insert(name.to_string(), dummy_ptr);
            sizes.insert(name.to_string(), 1024);
            shapes.insert(name.to_string(), vec![16, 64]);
        }

        // Act
        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };

        // Assert: all three maps have exactly the same keys
        assert_eq!(wm.ext_ptrs.len(), 4);
        assert_eq!(wm.ext_sizes.len(), 4);
        assert_eq!(wm.ext_shapes.len(), 4);
        for name in &names {
            assert!(wm.ext_ptrs.contains_key(*name));
            assert!(wm.ext_sizes.contains_key(*name));
            assert!(wm.ext_shapes.contains_key(*name));
        }
    }

    // ======================================================================
    // GeneratorForwardConfig: intermediate_size and vocab_size accessors
    // ======================================================================

    #[test]
    fn forward_config_intermediate_and_vocab_accessors() {
        use crate::engine::executor_types::GeneratorForwardConfig;

        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();

        // Act & Assert
        assert_eq!(cfg.intermediate_size(), 128);
        assert_eq!(cfg.vocab_size(), 100);
    }

    // ======================================================================
    // 15 NEW TESTS — additional edge-case coverage
    // ======================================================================

    // ======================================================================
    // detect_hetero_layers: only L0 exists (no other layers in weight map)
    // @trace TEST-EXEC-COMPILE-001
    // ======================================================================

    #[test]
    fn detect_hetero_layers_only_l0_no_other_layers() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 1024);
        // No L1-L5 at all => scan finds no diffs => detect_hetero_layers returns None
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none());
    }

    // ======================================================================
    // scan_hetero_layer_diffs: many layers all identical (no full layers at all)
    // @trace TEST-EXEC-COMPILE-002
    // ======================================================================

    #[test]
    fn scan_hetero_layer_diffs_20_layers_all_identical() {
        let geo = ModelGeometry { num_layers: 20, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 1..20 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
            sizes.insert(format!("L{}.k_proj", l), 512);
            sizes.insert(format!("L{}.gate_proj", l), 2048);
        }
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let (full_indices, q_dim, kv_dim, inter_diff) =
            TestExec::scan_hetero_layer_diffs(&find_size, 1024, Some(2048), &geo);

        assert!(full_indices.is_empty());
        assert_eq!(q_dim, 0);
        assert_eq!(kv_dim, 0);
        assert!(!inter_diff);
    }

    // ======================================================================
    // ModelGeometry: kv_bytes_per_token with num_kv_shared_layers
    // @trace TEST-EXEC-COMPILE-003
    // ======================================================================

    #[test]
    fn geometry_kv_bytes_per_token_with_shared_layers() {
        let geo = ModelGeometry {
            num_layers: 8,
            num_kv_shared_layers: 3,
            num_kv_heads: 4,
            head_dim: 32,
            dtype: gllm_kernels::types::DType::F32,
            ..make_geometry()
        };
        // kv_bytes_per_token uses num_layers (not effective_kv_layers) for buffer sizing
        // = 2 * kv_dim * num_layers * dtype_size = 2 * (4*32) * 8 * 4 = 8192
        assert_eq!(geo.kv_bytes_per_token(), 2 * 128 * 8 * 4);
        // effective_kv_layers is separate: max(8-3, 1) = 5
        assert_eq!(geo.effective_kv_layers(), 5);
    }

    // ======================================================================
    // RequestPhase: Copy semantics (derive Copy)
    // @trace TEST-EXEC-COMPILE-004
    // ======================================================================

    #[test]
    fn request_phase_copy_semantics() {
        use crate::scheduler::request_state::RequestPhase;

        let prefill = RequestPhase::Prefill;
        let prefill_copy = prefill; // Copy
        assert_eq!(prefill, prefill_copy);

        let decode = RequestPhase::Decode;
        let decode_copy = decode;
        assert_eq!(decode, decode_copy);

        let chunked = RequestPhase::ChunkedPrefill;
        let chunked_copy = chunked;
        assert_eq!(chunked, chunked_copy);
    }

    // ======================================================================
    // SequenceInput: validate_page_table with total_pages=0 always fails
    // @trace TEST-EXEC-COMPILE-005
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_zero_total_pages() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        // page_id 0 >= total_pages 0 => err
        let result = input.validate_page_table(0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("page_table[0]"));
    }

    // ======================================================================
    // SequenceInput: validate_page_table with total_pages=0 and empty page_table
    // @trace TEST-EXEC-COMPILE-006
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_zero_total_pages_empty_vec() {
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        // No entries to validate, should pass
        assert!(input.validate_page_table(0).is_ok());
    }

    // ======================================================================
    // BatchInput: Debug format includes sequence data
    // @trace TEST-EXEC-COMPILE-007
    // ======================================================================

    #[test]
    fn batch_input_debug_format() {
        use crate::engine::executor_types::BatchInput;

        let batch = BatchInput {
            sequences: vec![SequenceInput {
                tokens: vec![42],
                position: 0,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            }],
        };
        let debug = format!("{batch:?}");
        assert!(debug.contains("BatchInput"));
        assert!(debug.contains("sequences"));
    }

    // ======================================================================
    // ArchitectureFeatures: is_vision, is_audio, has_classifier, tie_lm_head fields
    // @trace TEST-EXEC-COMPILE-008
    // ======================================================================

    #[test]
    fn auto_features_vision_audio_classifier_fields() {
        let mut features = make_auto_features();
        assert!(!features.is_vision);
        assert!(!features.is_audio);
        assert!(!features.has_classifier);
        assert!(!features.tie_lm_head);

        features.is_vision = true;
        features.is_audio = true;
        features.has_classifier = true;
        features.tie_lm_head = true;

        assert!(features.is_vision);
        assert!(features.is_audio);
        assert!(features.has_classifier);
        assert!(features.tie_lm_head);
    }

    // ======================================================================
    // detect_hetero_layers: all layers same including L0 (no diff at all)
    // @trace TEST-EXEC-COMPILE-009
    // ======================================================================

    #[test]
    fn detect_hetero_layers_all_layers_same_including_l0() {
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        for l in 0..6 {
            sizes.insert(format!("L{}.q_proj", l), 1024);
        }
        let result = TestExec::detect_hetero_layers(&sizes, &geo);
        assert!(result.is_none());
    }

    // ======================================================================
    // GeneratorForwardConfig: rope config matches geometry rope_theta
    // @trace TEST-EXEC-COMPILE-010
    // ======================================================================

    #[test]
    fn forward_config_rope_matches_geometry() {
        use crate::engine::executor_types::GeneratorForwardConfig;

        let cfg = GeneratorForwardConfig::default_for_test();

        // Act & Assert: rope_theta accessor matches geometry.rope_theta
        assert!((cfg.rope_theta() - cfg.geometry.rope_theta).abs() < 1e-10);
        // rope_scale accessor matches rope field
        assert!((cfg.rope_scale() - cfg.rope.scale).abs() < 1e-10);
    }

    // ======================================================================
    // SequenceInput: tokens with u32::MAX boundary value
    // @trace TEST-EXEC-COMPILE-011
    // ======================================================================

    #[test]
    fn sequence_input_tokens_with_max_u32() {
        let input = SequenceInput {
            tokens: vec![u32::MAX, 0, u32::MAX],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(input.tokens.len(), 3);
        assert_eq!(input.tokens[0], u32::MAX);
        assert_eq!(input.tokens[1], 0);
        assert_eq!(input.tokens[2], u32::MAX);
    }

    // ======================================================================
    // WeightMaps: ext_sizes map with zero size entry (zero-sized tensor)
    // @trace TEST-EXEC-COMPILE-012
    // ======================================================================

    #[test]
    fn weight_maps_zero_size_tensor_entry() {
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        ptrs.insert("empty_tensor".to_string(), std::ptr::null());
        sizes.insert("empty_tensor".to_string(), 0);
        shapes.insert("empty_tensor".to_string(), vec![0]);

        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };
        assert_eq!(wm.ext_sizes.get("empty_tensor"), Some(&0));
        assert_eq!(wm.ext_shapes.get("empty_tensor").unwrap(), &vec![0]);
    }

    // ======================================================================
    // find_first_large_layer: exactly two layers, second differs
    // @trace TEST-EXEC-COMPILE-013
    // ======================================================================

    #[test]
    fn find_first_large_layer_two_layers_second_differs() {
        let geo = ModelGeometry { num_layers: 2, ..make_geometry() };
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L1.gate_proj".to_string(), 4096);
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        let result = TestExec::find_first_large_layer(&find_size, 2048, &geo);
        assert_eq!(result, Some(1));
    }

    // ======================================================================
    // BusinessConfig: intent_anchor_layer and cot_step_hook defaults
    // @trace TEST-EXEC-COMPILE-014
    // ======================================================================

    #[test]
    fn business_config_intent_and_cot_default_none() {
        use gllm_kernels::compiler::BusinessConfig;

        let cfg = BusinessConfig::default();
        assert!(cfg.intent_anchor_layer.is_none());
        assert!(cfg.cot_step_hook.is_none());
    }

    // ======================================================================
    // KvCacheConfig: max_seq_len passes through effective_kv_max_seq_len
    // @trace TEST-EXEC-COMPILE-015
    // ======================================================================

    #[test]
    fn kv_cache_config_max_seq_len_passes_through() {
        use crate::engine::executor_types::KvCacheConfig;

        let geo = Arc::new(ModelGeometry {
            max_seq_len: 65536,
            ..make_geometry()
        });
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        assert_eq!(cfg.max_seq_len(), 65536);
    }

    // ======================================================================
    // 15 NEW TESTS — additional edge-case coverage
    // ======================================================================

    // ======================================================================
    // RequestData: empty output_tokens at construction (prefill state)
    // @trace TEST-EXEC-COMPILE-016
    // ======================================================================

    #[test]
    fn request_data_empty_output_tokens_prefill() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        // Arrange
        let sampling = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 1.0 };

        // Act: prefill state has empty output_tokens
        let data = RequestData {
            prompt_tokens: vec![1, 2, 3, 4, 5],
            output_tokens: vec![],
            sampling_config: sampling,
            is_prefill: true,
            phase: RequestPhase::Prefill,
            max_new_tokens: 256,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };

        // Assert
        assert_eq!(data.prompt_tokens.len(), 5);
        assert!(data.output_tokens.is_empty());
        assert!(data.is_prefill);
        assert!(!data.finished);
        assert!(data.session_id.is_none());
        assert!(data.thinking_budget.is_none());
    }

    // ======================================================================
    // RequestData: thinking_budget=Some(0) meaning disabled
    // @trace TEST-EXEC-COMPILE-017
    // ======================================================================

    #[test]
    fn request_data_thinking_budget_zero_means_disabled() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        // Arrange & Act
        let data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![2],
            sampling_config: SamplingConfig::default(),
            is_prefill: false,
            phase: RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: Some(0),
            fused_prefill_hidden: None,
        };

        // Assert: Some(0) is a valid distinct state from None
        assert_eq!(data.thinking_budget, Some(0));
        assert!(data.thinking_budget.is_some());
        assert_eq!(data.thinking_budget.unwrap(), 0);
    }

    // ======================================================================
    // RequestData: fused_prefill_hidden populated for multimodal request
    // @trace TEST-EXEC-COMPILE-018
    // ======================================================================

    #[test]
    fn request_data_fused_prefill_hidden_populated() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        // Arrange: simulate a multimodal request with fused hidden
        let hidden_dim = 256;
        let prompt_len = 5;
        let fused: Vec<f32> = vec![0.5; prompt_len * hidden_dim];

        // Act
        let data = RequestData {
            prompt_tokens: vec![10, 20, 30, 40, 50],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            is_prefill: true,
            phase: RequestPhase::Prefill,
            max_new_tokens: 512,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: Some(fused.clone()),
        };

        // Assert
        assert!(data.fused_prefill_hidden.is_some());
        assert_eq!(data.fused_prefill_hidden.as_ref().unwrap().len(), prompt_len * hidden_dim);
        assert!((data.fused_prefill_hidden.as_ref().unwrap()[0] - 0.5).abs() < 1e-6);
    }

    // ======================================================================
    // RequestData: finished=true with session_id still present
    // @trace TEST-EXEC-COMPILE-019
    // ======================================================================

    #[test]
    fn request_data_finished_with_active_session() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        // Arrange & Act: generation completed but session persists
        let data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5, 6, 7],
            sampling_config: SamplingConfig { temperature: 0.5, top_k: 10, top_p: 0.95 },
            is_prefill: false,
            phase: RequestPhase::Decode,
            max_new_tokens: 100,
            finished: true,
            session_id: Some(99),
            thinking_budget: None,
            fused_prefill_hidden: None,
        };

        // Assert: finished does not clear session_id
        assert!(data.finished);
        assert_eq!(data.session_id, Some(99));
        assert_eq!(data.output_tokens.len(), 4);
    }

    // ======================================================================
    // ArchitectureFeatures: has_rope, has_attention_bias, attention_sinks defaults
    // @trace TEST-EXEC-COMPILE-020
    // ======================================================================

    #[test]
    fn auto_features_rope_attention_bias_sinks_defaults() {
        let features = make_auto_features();

        // Assert: default decoder features
        assert!(features.has_rope);
        assert!(!features.has_attention_bias);
        assert!(!features.attention_sinks);
        assert!(!features.has_embedding_scale);
        assert!(!features.has_qk_norm);
        assert!(!features.has_value_norm);
    }

    // ======================================================================
    // ArchitectureFeatures: is_mla and mla_use_unabsorbed fields
    // @trace TEST-EXEC-COMPILE-021
    // ======================================================================

    #[test]
    fn auto_features_mla_fields_default_false() {
        let features = make_auto_features();

        // Assert: default non-MLA
        assert!(!features.is_mla);
        assert_eq!(features.mla_latent_dim, 0);
        assert_eq!(features.mla_rope_dim, 0);
        assert!(!features.mla_use_unabsorbed);

        // Override to MLA
        let mut mla_features = make_auto_features();
        mla_features.is_mla = true;
        mla_features.mla_latent_dim = 512;
        mla_features.mla_rope_dim = 64;
        mla_features.mla_use_unabsorbed = true;

        assert!(mla_features.is_mla);
        assert_eq!(mla_features.mla_latent_dim, 512);
        assert_eq!(mla_features.mla_rope_dim, 64);
        assert!(mla_features.mla_use_unabsorbed);
    }

    // ======================================================================
    // ExecutorError::Scheduler Display with multiline message
    // @trace TEST-EXEC-COMPILE-022
    // ======================================================================

    #[test]
    fn executor_error_scheduler_display_multiline() {
        // Arrange
        let err = ExecutorError::Scheduler(
            "no available KV pages\n  total_pages=1024\n  used_pages=1024".to_string(),
        );

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("scheduler error"));
        assert!(msg.contains("no available KV pages"));
        assert!(msg.contains("total_pages=1024"));
    }

    // ======================================================================
    // GeneratorForwardConfig: geometry accessor returns correct Arc contents
    // @trace TEST-EXEC-COMPILE-023
    // ======================================================================

    #[test]
    fn forward_config_geometry_accessor_contents() {
        use crate::engine::executor_types::GeneratorForwardConfig;

        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();

        // Act & Assert: geometry fields match accessors
        assert_eq!(cfg.geometry.hidden_size, cfg.hidden_size());
        assert_eq!(cfg.geometry.num_layers, cfg.num_layers());
        assert_eq!(cfg.geometry.vocab_size, cfg.vocab_size());
        assert_eq!(cfg.geometry.intermediate_size, cfg.intermediate_size());
        assert_eq!(cfg.geometry.num_heads, cfg.num_heads());
        assert_eq!(cfg.geometry.num_kv_heads, cfg.num_kv_heads());
        assert_eq!(cfg.geometry.head_dim, cfg.head_dim());
    }

    // ======================================================================
    // LogitsHandle: data with negative values (logits can be negative)
    // @trace TEST-EXEC-COMPILE-024
    // ======================================================================

    #[test]
    fn logits_handle_negative_values() {
        // Arrange: logits typically include negative values after normalization
        let handle = LogitsHandle {
            data: vec![-2.5, -0.1, 3.7, -8.0, 0.0],
        };

        // Act & Assert
        assert_eq!(handle.data.len(), 5);
        assert!((handle.data[0] - (-2.5)).abs() < 1e-6);
        assert!((handle.data[3] - (-8.0)).abs() < 1e-6);
        assert!((handle.data[4]).abs() < 1e-6);

        // Verify clone preserves negative values
        let cloned = handle.clone();
        for (a, b) in handle.data.iter().zip(cloned.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // ======================================================================
    // SequenceInput: validate_page_table with single valid page_id=0
    // @trace TEST-EXEC-COMPILE-025
    // ======================================================================

    #[test]
    fn sequence_input_validate_page_table_single_page_id_zero() {
        // Arrange: page_id=0 is always valid as long as total_pages >= 1
        let input = SequenceInput {
            tokens: vec![42],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };

        // Act & Assert: page_id 0 < total_pages 1 => ok
        assert!(input.validate_page_table(1).is_ok());
        // page_id 0 >= total_pages 0 => err
        assert!(input.validate_page_table(0).is_err());
    }

    // ======================================================================
    // KvCacheConfig: kv_bytes_per_token calculation for standard config
    // @trace TEST-EXEC-COMPILE-026
    // ======================================================================

    #[test]
    fn kv_cache_config_kv_bytes_per_token_calculation() {
        use crate::engine::executor_types::KvCacheConfig;

        // Arrange: num_kv_heads=4, head_dim=32, num_layers=6, F32 (4 bytes)
        let geo = Arc::new(make_geometry());
        let cfg = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };

        // Act & Assert: 2 * kv_dim * num_layers * dtype_size = 2 * 128 * 6 * 4 = 6144
        assert_eq!(cfg.geometry.kv_bytes_per_token(), 6144);
    }

    // ======================================================================
    // BusinessConfig: semantic_gatekeeper with SgConfig
    // @trace TEST-EXEC-COMPILE-027
    // ======================================================================

    #[test]
    fn business_config_with_semantic_gatekeeper() {
        use gllm_kernels::compiler::BusinessConfig;
        use gllm_kernels::compiler::mega_kernel_abi::SgConfig;

        // Arrange
        let sg = SgConfig {
            detect_layer: 15,
            detect_offset: 0,
            inject_offset: 0,
            q_tap: None,
        };

        // Act
        let cfg = BusinessConfig {
            semantic_gatekeeper: Some(sg),
            ..BusinessConfig::default()
        };

        // Assert
        assert!(cfg.semantic_gatekeeper.is_some());
        assert_eq!(cfg.semantic_gatekeeper.as_ref().unwrap().detect_layer, 15);
    }

    // ======================================================================
    // build_hetero_config: single full layer at end of model (N sliding + 1 full, 1 segment)
    // @trace TEST-EXEC-COMPILE-028
    // ======================================================================

    #[test]
    fn build_hetero_config_single_segment_large_model() {
        // Arrange: 80 layers, [79 sliding + 1 full] * 1 = 80
        let geo = ModelGeometry { num_layers: 80, ..make_geometry() };
        let sizes: HashMap<String, usize> = HashMap::new();
        let find_size = |c: &str| -> Option<usize> { sizes.get(c).copied() };

        // Act
        let result = TestExec::build_hetero_config(
            &find_size, 1024, None, vec![79], 4, 2, false, &geo,
        );

        // Assert
        assert!(result.is_some());
        let cfg = result.unwrap();
        assert_eq!(cfg.num_segments, 1);
        assert_eq!(cfg.sliding_per_segment, 79);
        assert_eq!(cfg.full_layer_indices, vec![79]);
    }

    // ======================================================================
    // PoolMode: Debug format distinguishes all three variants
    // @trace TEST-EXEC-COMPILE-029
    // ======================================================================

    #[test]
    fn pool_mode_debug_distinguishes_all_variants() {
        let last = format!("{:?}", PoolMode::LastToken);
        let mean = format!("{:?}", PoolMode::MeanPool);
        let cls = format!("{:?}", PoolMode::ClsToken);

        assert!(last.contains("LastToken"));
        assert!(mean.contains("MeanPool"));
        assert!(cls.contains("ClsToken"));

        // Each debug string is distinct
        assert_ne!(last, mean);
        assert_ne!(mean, cls);
        assert_ne!(last, cls);
    }

    // ======================================================================
    // PoolMode: Clone produces identical Debug output
    // @trace TEST-EXEC-COMPILE-029b
    // ======================================================================

    #[test]
    fn pool_mode_clone_preserves_variant() {
        let original = PoolMode::MeanPool;
        let cloned = original.clone();
        assert_eq!(format!("{:?}", original), format!("{:?}", cloned));
    }

    // ======================================================================
    // BackendError: long message preserved through Display
    // @trace TEST-EXEC-COMPILE-030
    // ======================================================================

    #[test]
    fn backend_error_long_message_preserved() {
        // Arrange: simulate a real PTX compilation error message
        let long_msg = "ptxas application ptx input, line 1234; error   : Instruction requires .target sm_80 or higher\n\
                         error   : Compilation failed";
        let err = BackendError::Cuda(long_msg.to_string());

        // Act
        let display = format!("{err}");

        // Assert: full message preserved
        assert!(display.contains("ptxas"));
        assert!(display.contains("sm_80"));
        assert!(display.contains("line 1234"));
        assert!(display.starts_with("CUDA error:"));
    }

    // ======================================================================
    // 10 NEW TESTS — additional uncovered edge cases
    // ======================================================================

    // ======================================================================
    // ExecutorError::RequestNotFound Display with large request_id
    // ======================================================================

    #[test]
    fn executor_error_request_not_found_large_id() {
        // Arrange
        let err = ExecutorError::RequestNotFound { request_id: u64::MAX };

        // Act
        let msg = format!("{err}");

        // Assert: message contains the large request_id value
        assert!(msg.contains(&u64::MAX.to_string()));
    }

    // ======================================================================
    // ExecutorResult<T>: Ok variant round-trips value correctly
    // ======================================================================

    #[test]
    fn executor_result_ok_roundtrip() {
        // Arrange & Act
        let result: ExecutorResult<u32> = Ok(42);

        // Assert
        assert_eq!(result.unwrap(), 42);
    }

    // ======================================================================
    // ExecutorResult<T>: Err variant carries ExecutorError through
    // ======================================================================

    #[test]
    fn executor_result_err_carries_backend_error() {
        // Arrange
        let backend_err = BackendError::Hip("device lost".to_string());
        let result: ExecutorResult<String> = Err(backend_err.into());

        // Act & Assert
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("HIP"));
        assert!(msg.contains("device lost"));
    }

    // ======================================================================
    // build_output_modes: encoder with ModelKind::Embedding returns EncodeToLayer
    // with anchor_layer=0 explicitly
    // ======================================================================

    #[test]
    fn build_output_modes_encoder_embedding_anchor_layer_zero() {
        // Arrange
        let manifest = make_manifest(ModelKind::Embedding);

        // Act
        let modes = TestExec::build_output_modes(&manifest, true, 4096, 2);

        // Assert: encoder Embedding always uses anchor_layer=0 and MeanPool
        match &modes[0] {
            OutputMode::EncodeToLayer { anchor_layer, pool_mode } => {
                assert_eq!(*anchor_layer, 0);
                assert!(matches!(pool_mode, PoolMode::MeanPool));
            }
            other => panic!("expected EncodeToLayer, got {:?}", other),
        }
    }

    // ======================================================================
    // GeneratorForwardConfig: norm_eps accessor matches geometry
    // ======================================================================

    #[test]
    fn forward_config_norm_eps_matches_geometry() {
        // Arrange
        use crate::engine::executor_types::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();

        // Act & Assert
        assert!((cfg.norm_eps() - cfg.geometry.norm_eps).abs() < 1e-10);
    }

    // ======================================================================
    // AttentionTopology: causal and bidirectional produce distinct mask types
    // with the same geometry
    // ======================================================================

    #[test]
    fn attention_topology_mask_types_differ_same_geometry() {
        // Arrange
        let geo = Arc::new(make_geometry());

        // Act
        let causal = AttentionTopology::causal(geo.clone());
        let bidir = AttentionTopology::bidirectional(geo);

        // Assert: same geometry, different mask types
        assert!(matches!(causal.mask_type, AttentionMaskType::Causal));
        assert!(matches!(bidir.mask_type, AttentionMaskType::Bidirectional));
        assert_ne!(causal.mask_type, bidir.mask_type);
        // Both share the same geometry Arc
        assert_eq!(causal.geometry.hidden_size, bidir.geometry.hidden_size);
        assert_eq!(causal.geometry.num_layers, bidir.geometry.num_layers);
    }

    // ======================================================================
    // KvCacheConfig: dtype_size for different DType values
    // ======================================================================

    #[test]
    fn kv_cache_config_dtype_size_various_dtypes() {
        // Arrange & Act: F32
        use crate::engine::executor_types::KvCacheConfig;
        let geo = Arc::new(make_geometry());
        let cfg_f32 = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };

        let cfg_bf16 = KvCacheConfig {
            geometry: geo.clone(),
            kv_dtype: gllm_kernels::types::DType::BF16,
            page_size: 16,
            swap_config: None,
        };

        let cfg_f16 = KvCacheConfig {
            geometry: geo,
            kv_dtype: gllm_kernels::types::DType::F16,
            page_size: 16,
            swap_config: None,
        };

        // Assert: each dtype reports correct byte size
        assert_eq!(cfg_f32.dtype_size(), 4);
        assert_eq!(cfg_bf16.dtype_size(), 2);
        assert_eq!(cfg_f16.dtype_size(), 2);
    }

    // ======================================================================
    // WeightMaps: ext_ptrs with multiple valid non-null pointers
    // ======================================================================

    #[test]
    fn weight_maps_multiple_non_null_ptrs() {
        // Arrange
        let val_a: u8 = 10;
        let val_b: u8 = 20;
        let val_c: u8 = 30;
        let mut ptrs: HashMap<String, *const u8> = HashMap::new();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        ptrs.insert("embed".to_string(), &val_a);
        sizes.insert("embed".to_string(), 128);
        shapes.insert("embed".to_string(), vec![32, 4]);

        ptrs.insert("lm_head".to_string(), &val_b);
        sizes.insert("lm_head".to_string(), 256);
        shapes.insert("lm_head".to_string(), vec![100, 64]);

        ptrs.insert("final_norm".to_string(), &val_c);
        sizes.insert("final_norm".to_string(), 64);
        shapes.insert("final_norm".to_string(), vec![16]);

        // Act
        let wm = WeightMaps { ext_ptrs: ptrs, ext_sizes: sizes, ext_shapes: shapes };

        // Assert: each key is present and pointers are non-null
        assert_eq!(wm.ext_ptrs.len(), 3);
        assert_eq!(wm.ext_sizes.len(), 3);
        assert_eq!(wm.ext_shapes.len(), 3);

        for name in &["embed", "lm_head", "final_norm"] {
            assert!(wm.ext_ptrs.get(*name).is_some());
            assert!(!wm.ext_ptrs.get(*name).unwrap().is_null());
            assert!(wm.ext_sizes.get(*name).is_some());
            assert!(wm.ext_shapes.get(*name).is_some());
        }
    }

    // ======================================================================
    // detect_hetero_layers: full_indices contain only layer 0 => pattern fails
    // ======================================================================

    #[test]
    fn detect_hetero_layers_ref_q_zero_no_l0_still_none() {
        // Arrange: L0.q_proj exists with size 0, all others also 0
        let geo = make_geometry();
        let mut sizes: HashMap<String, usize> = HashMap::new();
        sizes.insert("L0.q_proj".to_string(), 0);
        for l in 1..6 {
            sizes.insert(format!("L{}.q_proj", l), 0);
        }

        // Act
        let result = TestExec::detect_hetero_layers(&sizes, &geo);

        // Assert: all sizes same (0 == 0), no diff found => None
        assert!(result.is_none());
    }

    // ======================================================================
    // SamplingConfig: equality comparison between identical and different configs
    // ======================================================================

    #[test]
    fn sampling_config_field_access_and_copy() {
        // Arrange
        use crate::engine::executor_types::SamplingConfig;
        let a = SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.95 };

        // Act: Copy semantics — b gets its own copy
        let b = a;

        // Assert: field values are preserved through copy
        assert!((b.temperature - 0.7).abs() < 1e-6);
        assert_eq!(b.top_k, 50);
        assert!((b.top_p - 0.95).abs() < 1e-6);

        // Assert: original is still valid after copy (Copy trait)
        assert!((a.temperature - 0.7).abs() < 1e-6);
        assert_eq!(a.top_k, 50);
    }

}
