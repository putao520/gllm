// ============================================================================
// Weight Blob Packing (graph.weight_layout() driven)
// ============================================================================

/// 通用权重打包：按预解析的 (name, offset) 对将权重排列到连续 blob。
///
/// 完全由图结构驱动，所有名字都是 canonical name。
/// weight_ptrs/weight_sizes 以 canonical name 为 key。
/// raw_floats 以外部名为 key，通过 name_map 做反向查找。
fn pack_weights_from_graph(
    named_offsets: &[(String, usize)],
    total_bytes: usize,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
    raw_floats: &std::collections::HashMap<String, crate::loader::RawFloatTensor>,
    name_map: &crate::loader::name_map::TensorNameMap,
    layer_config: Option<&gllm_kernels::compiler::graph::LayerLoopConfig>,
    hetero_config: Option<&gllm_kernels::compiler::graph::HeteroLayerLoopConfig>,
    num_layers: usize,
    gemma_norm_residual: bool,
) -> Vec<u8> {
    let mut blob = vec![0u8; total_bytes];
    let mut packed_count = 0usize;
    let mut missing_count = 0usize;

    // ── Heterogeneous layer offset computation ──
    // For Gemma 4 E2B: 4 layer types with different strides.
    // Each layer's absolute offset depends on its segment and position.
    // named_offsets uses reference layer canonical names (L0., L4., L15., L19.)
    // because weight tensors are registered with cn_layer(ref_layer, suffix).
    let _ = hetero_config; // used in packing loop below
    // SPEC/39: 从 HeteroLayerConfig 推导参考层索引，替代硬编码 [0,4,15,19]。
    // 4 种异构层类型的参考层（每种取第一个出现的层索引）：
    //   ss_ref = 0 (第一个 sliding_small)
    //   fs_ref = sliding_per_segment (第一个 full_small)
    //   sl_ref = large_ffn_start_segment * layers_per_seg (第一个 sliding_large)
    //   fl_ref = sl_ref + sliding_per_segment (第一个 full_large)
    let (ss_ref, fs_ref, sl_ref, fl_ref) = if let Some(ref hcfg) = hetero_config {
        let lps = hcfg.sliding_per_segment + 1; // layers per segment
        let ss = 0;
        let fs = hcfg.sliding_per_segment;
        let sl = hcfg.large_ffn_start_segment * lps;
        let fl = sl + hcfg.sliding_per_segment;
        (ss, fs, sl, fl)
    } else {
        (0, 4, 15, 19) // fallback (shouldn't be reached in non-hetero mode)
    };
    let hetero_ref_layers: [usize; 4] = [ss_ref, fs_ref, sl_ref, fl_ref];
    let hetero_ref_prefixes: [String; 4] = [
        format!("L{}.", hetero_ref_layers[0]),
        format!("L{}.", hetero_ref_layers[1]),
        format!("L{}.", hetero_ref_layers[2]),
        format!("L{}.", hetero_ref_layers[3]),
    ];

    /// Compute absolute byte offset for a layer in hetero mode.
    /// Returns (abs_offset, stride_for_this_type).
    fn hetero_layer_offset(
        layer_idx: usize, hcfg: &gllm_kernels::compiler::graph::HeteroLayerLoopConfig,
    ) -> usize {
        let layers_per_seg = hcfg.sliding_per_segment + 1;
        let seg = layer_idx / layers_per_seg;
        let pos = layer_idx % layers_per_seg;
        let is_sliding = pos < hcfg.sliding_per_segment;
        let is_small = seg < hcfg.large_ffn_start_segment;

        let seg_base = if is_small {
            seg * hcfg.small_segment_stride
        } else {
            let small_segs = hcfg.large_ffn_start_segment;
            small_segs * hcfg.small_segment_stride
                + (seg - small_segs) * hcfg.large_segment_stride
        };

        let pos_off = if is_sliding {
            let s = if is_small { hcfg.sliding_small_stride } else { hcfg.sliding_large_stride };
            pos * s
        } else {
            let s = if is_small { hcfg.sliding_small_stride } else { hcfg.sliding_large_stride };
            hcfg.sliding_per_segment * s
            // plus the full-type stride is added below
        };

        // For full layers, add the full-type offset after all sliding layers
        let full_off = if !is_sliding {
            let _f = if is_small { hcfg.full_small_stride } else { hcfg.full_large_stride };
            // The position within the segment for a full layer is always 0
            // (there's exactly one full layer per segment, after all sliding layers)
            0 // already accounted for in seg_base + pos_off
        } else {
            0
        };

        hcfg.layer_blob_base_offset + seg_base + pos_off + full_off
    }

    /// Determine which hetero type a layer belongs to.
    /// Returns index: 0=sliding_small, 1=full_small, 2=sliding_large, 3=full_large.
    fn hetero_type_index(
        layer_idx: usize, hcfg: &gllm_kernels::compiler::graph::HeteroLayerLoopConfig,
    ) -> usize {
        let layers_per_seg = hcfg.sliding_per_segment + 1;
        let seg = layer_idx / layers_per_seg;
        let pos = layer_idx % layers_per_seg;
        let is_sliding = pos < hcfg.sliding_per_segment;
        let is_small = seg < hcfg.large_ffn_start_segment;
        match (is_sliding, is_small) {
            (true, true) => 0,   // sliding_small
            (false, true) => 1,  // full_small
            (true, false) => 2,  // sliding_large
            (false, false) => 3, // full_large
        }
    }
    if false && !raw_floats.is_empty() {
        for (cn, offset) in named_offsets {
            if cn.contains("qkv_proj") || cn.contains("gate_proj") {
                let ext = name_map.resolve_external_to_string(cn);
                if let Some(raw) = raw_floats.get(&ext) {
                    let esz = match raw.dtype {
                        ::safetensors::Dtype::BF16 | ::safetensors::Dtype::F16 => 2,
                        _ => 4,
                    };
                    let numel = raw.data.len() / esz;
                    eprintln!(
                        "[pack] {} -> {} dtype={:?} numel={} f32_bytes={} offset={}",
                        cn,
                        ext,
                        raw.dtype,
                        numel,
                        numel * 4,
                        offset
                    );
                }
            }
        }
    }

    for (canonical_name, offset) in named_offsets {
        let ext_name = name_map.resolve_external_to_string(canonical_name);
        // Debug: trace norm weight packing path
        if canonical_name.contains("q_norm") || canonical_name.contains("k_norm") || canonical_name.contains("input_norm") {
            let in_raw = raw_floats.get(&ext_name).is_some();
            let in_ptr = weight_ptrs.get(canonical_name).is_some();
            eprintln!("[PACK-TRACE] {} ext={} raw_floats={} weight_ptrs={}", canonical_name, ext_name, in_raw, in_ptr);
            if in_raw {
                if let Some(raw) = raw_floats.get(&ext_name) {
                    match raw.dtype {
                        ::safetensors::Dtype::BF16 => {
                            let src = unsafe {
                                std::slice::from_raw_parts(raw.data.as_ptr() as *const half::bf16, raw.data.len() / 2)
                            };
                            let first_4: Vec<f32> = src.iter().take(4).map(|v| v.to_f32()).collect();
                            eprintln!("[PACK-RAW] {} dtype=BF16 first_4=[{:.6},{:.6},{:.6},{:.6}]", canonical_name, first_4[0], first_4[1], first_4[2], first_4[3]);
                        }
                        other => { eprintln!("[PACK-RAW] {} dtype={:?}", canonical_name, other); }
                    }
                }
            }
        }
        if let Some(raw) = raw_floats.get(&ext_name) {
            let elem_size = match raw.dtype {
                ::safetensors::Dtype::BF16 | ::safetensors::Dtype::F16 => 2,
                ::safetensors::Dtype::F32 => 4,
                _ => 4,
            };
            let numel = raw.data.len() / elem_size;
            let f32_bytes = numel * 4;

            // Per-layer weights: compute absolute offsets and replicate for all layers.
            // Hetero mode: detect by template prefixes (graph tensor names); Homogeneous: detect by "L0." prefix.
            let is_hetero_layer_rf = hetero_config.is_some()
                && hetero_ref_prefixes.iter().any(|p| canonical_name.starts_with(p));
            let is_per_layer_rf = (layer_config.is_some()
                && canonical_name.starts_with("L0.")
                && !canonical_name.contains("embed")
                // SPEC/39: global weights (final_norm, output head) don't start with "L0."
                // so the starts_with check already excludes them; "embed" guard is for safety
                // in case future PLE variants use "L0.embed" naming.
                && !canonical_name.contains("final_norm"))
                || is_hetero_layer_rf;

            if is_per_layer_rf {
                // Determine offset computation mode
                let (abs_off_fn, ref_layer_ext, hetero_suffix): (Box<dyn Fn(usize) -> usize>, String, Option<String>) =
                    if let Some(hcfg) = hetero_config {
                        // Match by template prefix (graph tensor name), then map to ref layer for weight lookup
                        let (type_idx, suffix) = hetero_ref_prefixes.iter().enumerate()
                            .filter_map(|(i, p)| {
                                if canonical_name.starts_with(p) {
                                    Some((i, &canonical_name[p.len()..]))
                                } else {
                                    None
                                }
                            })
                            .next()
                            .unwrap();
                        let ref_layer = hetero_ref_layers[type_idx];
                        let ref_cn = format!("L{}.{}", ref_layer, suffix);
                        let ref_ext = name_map.resolve_external_to_string(&ref_cn);
                        // Compute template-relative offset: offset in named_offsets is the absolute
                        // offset within the weight layout. For hetero, we need the offset relative
                        // to this template type's base (type_base_offsets[type_idx]).
                        let type_base_offsets: [usize; 4] = [
                            hcfg.layer_blob_base_offset,
                            hcfg.layer_blob_base_offset + hcfg.sliding_small_stride,
                            hcfg.layer_blob_base_offset + hcfg.sliding_small_stride + hcfg.full_small_stride,
                            hcfg.layer_blob_base_offset + hcfg.sliding_small_stride + hcfg.full_small_stride + hcfg.sliding_large_stride,
                        ];
                        let rel_off = (*offset).saturating_sub(type_base_offsets[type_idx]);
                        (Box::new(move |li| hetero_layer_offset(li, hcfg) + rel_off), ref_ext, Some(suffix.to_string()))
                    } else {
                        let cfg = layer_config.unwrap();
                        let stride = cfg.weight_stride;
                        let rel_off = *offset;
                        let ref_ext = name_map.resolve_external_to_string(canonical_name);
                        (Box::new(move |li| li * stride + rel_off), ref_ext, None)
                    };

                // Use the reference layer's external name for raw_floats lookup
                let raw_ref = raw_floats.get(&ref_layer_ext)
                    .or_else(|| raw_floats.get(&ext_name));
                if let Some(layer_raw) = raw_ref {
                    let lelem_size = match layer_raw.dtype {
                        ::safetensors::Dtype::BF16 | ::safetensors::Dtype::F16 => 2,
                        _ => 4,
                    };
                    let lnumel = layer_raw.data.len() / lelem_size;
                    let lf32_bytes = lnumel * 4;

                    for layer_idx in 0..num_layers {
                        // In hetero mode, skip layers of different type
                        if let Some(hcfg) = hetero_config {
                            let type_idx = hetero_ref_prefixes.iter().enumerate()
                                .filter_map(|(i, p)| {
                                    if canonical_name.starts_with(p) { Some(i) } else { None }
                                })
                                .next().unwrap();
                            let layer_type = hetero_type_index(layer_idx, hcfg);
                            if layer_type != type_idx { continue; }
                        }

                        let abs_off = abs_off_fn(layer_idx);
                        let copy_size = lf32_bytes.min(blob.len().saturating_sub(abs_off));
                        if copy_size == 0 || abs_off >= blob.len() { continue; }

                        // For non-reference layers, try per-layer lookup first
                        let layer_ext = if let Some(ref suffix) = hetero_suffix {
                            let layer_cn = format!("L{}.{}", layer_idx, suffix);
                            name_map.resolve_external_to_string(&layer_cn)
                        } else if layer_idx == 0 {
                            ref_layer_ext.clone()
                        } else {
                            let lcn = canonical_name.replacen("L0.", &format!("L{}.", layer_idx), 1);
                            name_map.resolve_external_to_string(&lcn)
                        };

                        // Gemma RMSNorm stores weights as residuals: gamma = 1.0 + weight.
                        // Pre-shift during packing so the JIT RMSNorm code (which does x * weight)
                        // applies the correct gamma.
                        // Gemma RMSNorm stores q_norm/k_norm weights as residuals: gamma = 1.0 + weight.
                        // Only q_norm and k_norm use residual convention.
                        // input_norm/post_attn_norm use standard gamma (already the actual value).
                        let is_gemma_norm = gemma_norm_residual
                            && (canonical_name.ends_with(".q_norm")
                                || canonical_name.ends_with(".k_norm"));

                        if let Some(lr) = raw_floats.get(&layer_ext) {
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    blob[abs_off..].as_mut_ptr() as *mut f32,
                                    copy_size / 4,
                                )
                            };
                            match lr.dtype {
                                ::safetensors::Dtype::BF16 => {
                                    let src = unsafe {
                                        std::slice::from_raw_parts(lr.data.as_ptr() as *const half::bf16, lr.data.len() / 2)
                                    };
                                    for (i, &v) in src.iter().enumerate() {
                                        if i >= dst.len() { break; }
                                        let mut f = v.to_f32();
                                        if is_gemma_norm { f += 1.0; }
                                        dst[i] = f;
                                    }
                                    if is_gemma_norm && layer_idx == 0 && dst.len() >= 4 {
                                        eprintln!("[PACK-SHIFT] {} layer={} gemma_norm={} first_4=[{:.4},{:.4},{:.4},{:.4}]",
                                            canonical_name, layer_idx, is_gemma_norm,
                                            dst[0], dst[1], dst[2], dst[3]);
                                    }
                                }
                                ::safetensors::Dtype::F16 => {
                                    let src = unsafe {
                                        std::slice::from_raw_parts(lr.data.as_ptr() as *const half::f16, lr.data.len() / 2)
                                    };
                                    for (i, &v) in src.iter().enumerate() {
                                        if i >= dst.len() { break; }
                                        let mut f = v.to_f32();
                                        if is_gemma_norm { f += 1.0; }
                                        dst[i] = f;
                                    }
                                }
                                _ => {
                                    let cs = lr.data.len().min(blob.len().saturating_sub(abs_off));
                                    blob[abs_off..abs_off + cs].copy_from_slice(&lr.data[..cs]);
                                    // Pre-shift F32 norm weights in-place
                                    if is_gemma_norm && cs >= 4 {
                                        let f32_dst = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                blob[abs_off..].as_mut_ptr() as *mut f32,
                                                cs / 4,
                                            )
                                        };
                                        for v in f32_dst.iter_mut() {
                                            *v += 1.0;
                                        }
                                    }
                                }
                            }
                        } else {
                            // Fallback: reuse reference layer data
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    blob[abs_off..].as_mut_ptr() as *mut f32,
                                    copy_size / 4,
                                )
                            };
                            match layer_raw.dtype {
                                ::safetensors::Dtype::BF16 => {
                                    let src = unsafe {
                                        std::slice::from_raw_parts(layer_raw.data.as_ptr() as *const half::bf16, lnumel)
                                    };
                                    for (i, &v) in src.iter().enumerate() {
                                        if i >= dst.len() { break; }
                                        let mut f = v.to_f32();
                                        if is_gemma_norm { f += 1.0; }
                                        dst[i] = f;
                                    }
                                }
                                ::safetensors::Dtype::F16 => {
                                    let src = unsafe {
                                        std::slice::from_raw_parts(layer_raw.data.as_ptr() as *const half::f16, lnumel)
                                    };
                                    for (i, &v) in src.iter().enumerate() {
                                        if i >= dst.len() { break; }
                                        let mut f = v.to_f32();
                                        if is_gemma_norm { f += 1.0; }
                                        dst[i] = f;
                                    }
                                }
                                _ => {
                                    let cs = layer_raw.data.len().min(blob.len().saturating_sub(abs_off));
                                    blob[abs_off..abs_off + cs].copy_from_slice(&layer_raw.data[..cs]);
                                    if is_gemma_norm && cs >= 4 {
                                        let f32_dst = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                blob[abs_off..].as_mut_ptr() as *mut f32,
                                                cs / 4,
                                            )
                                        };
                                        for v in f32_dst.iter_mut() {
                                            *v += 1.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Global weight: post-layer globals need offset adjustment for layer replication.
                // Pre-layer globals (embed, offset < base) stay at their graph offsets.
                let blob_off = if let Some(hcfg) = hetero_config {
                    let templates_blob = hcfg.sliding_small_stride + hcfg.full_small_stride
                        + hcfg.sliding_large_stride + hcfg.full_large_stride;
                    let small_segs = hcfg.large_ffn_start_segment;
                    let large_segs = hcfg.num_segments - small_segs;
                    let total_layers_blob = small_segs * hcfg.small_segment_stride
                        + large_segs * hcfg.large_segment_stride;
                    let graph_globals_start = hcfg.layer_blob_base_offset + templates_blob;
                    if *offset >= graph_globals_start {
                        *offset + total_layers_blob.saturating_sub(templates_blob)
                    } else {
                        *offset
                    }
                } else if let Some(cfg) = layer_config {
                    let globals_start = cfg.layer_blob_base_offset + cfg.weight_stride;
                    if *offset >= globals_start {
                        (num_layers - 1) * cfg.weight_stride + *offset
                    } else {
                        *offset
                    }
                } else {
                    *offset
                };
                let copy_size = f32_bytes.min(blob.len().saturating_sub(blob_off));
                if copy_size == 0 || blob_off >= blob.len() {
                    continue;
                }
                let dst_f32s = unsafe {
                    std::slice::from_raw_parts_mut(
                        blob[blob_off..].as_mut_ptr() as *mut f32,
                        copy_size / 4,
                    )
                };
                match raw.dtype {
                    ::safetensors::Dtype::BF16 => {
                        let src = unsafe {
                            std::slice::from_raw_parts(raw.data.as_ptr() as *const half::bf16, numel)
                        };
                        for (i, &v) in src.iter().enumerate() {
                            if i >= dst_f32s.len() {
                                break;
                            }
                            let mut f = v.to_f32();
                            if gemma_norm_residual && (canonical_name.ends_with(".q_norm") || canonical_name.ends_with(".k_norm")) { f += 1.0; }
                            dst_f32s[i] = f;
                        }
                    }
                    ::safetensors::Dtype::F16 => {
                        let src = unsafe {
                            std::slice::from_raw_parts(raw.data.as_ptr() as *const half::f16, numel)
                        };
                        for (i, &v) in src.iter().enumerate() {
                            if i >= dst_f32s.len() {
                                break;
                            }
                            let mut f = v.to_f32();
                            if gemma_norm_residual && (canonical_name.ends_with(".q_norm") || canonical_name.ends_with(".k_norm")) { f += 1.0; }
                            dst_f32s[i] = f;
                        }
                    }
                    _ => {
                        let copy_size = raw.data.len().min(blob.len().saturating_sub(blob_off));
                        blob[blob_off..blob_off + copy_size].copy_from_slice(&raw.data[..copy_size]);
                        if gemma_norm_residual && (canonical_name.ends_with(".q_norm") || canonical_name.ends_with(".k_norm")) && copy_size >= 4 {
                            let f32_dst = unsafe {
                                std::slice::from_raw_parts_mut(
                                    blob[blob_off..].as_mut_ptr() as *mut f32,
                                    copy_size / 4,
                                )
                            };
                            for v in f32_dst.iter_mut() {
                                *v += 1.0;
                            }
                        }
                    }
                }
            }
            packed_count += 1;
            continue;
        }

        // Hetero template weights: named_offsets uses template prefixes (layer_sliding_small.)
        // but weight_ptrs uses reference layer canonical names (L0.).
        // Match by template prefix, then map to ref layer for weight lookup.
        if let Some(hcfg) = hetero_config {
            if let Some((type_idx, suffix)) = hetero_ref_prefixes.iter().enumerate()
                .filter_map(|(i, p)| {
                    if canonical_name.starts_with(p) { Some((i, &canonical_name[p.len()..])) } else { None }
                })
                .next()
            {
                let ref_layer = hetero_ref_layers[type_idx];
                let ref_cn = format!("L{}.{}", ref_layer, suffix);
                let ref_ptr = match weight_ptrs.get(&ref_cn) {
                    Some(&p) if !p.is_null() => p,
                    _ => { missing_count += 1; continue; }
                };
                let ref_size = *weight_sizes.get(&ref_cn).unwrap_or(&0);
                if ref_size == 0 { continue; }

                let type_base_offsets: [usize; 4] = [
                    hcfg.layer_blob_base_offset,
                    hcfg.layer_blob_base_offset + hcfg.sliding_small_stride,
                    hcfg.layer_blob_base_offset + hcfg.sliding_small_stride + hcfg.full_small_stride,
                    hcfg.layer_blob_base_offset + hcfg.sliding_small_stride + hcfg.full_small_stride + hcfg.sliding_large_stride,
                ];
                let rel_off = offset.saturating_sub(type_base_offsets[type_idx]);

                // Gemma RMSNorm residual pre-shift detection for hetero path
                let is_gemma_norm_hetero = gemma_norm_residual
                    && (suffix.ends_with(".q_norm")
                        || suffix.ends_with(".k_norm"));
                if canonical_name.contains("q_norm") {
                    eprintln!("[HETERO-NORM] {} suffix={} is_gemma={} gemma_norm_residual={}", canonical_name, suffix, is_gemma_norm_hetero, gemma_norm_residual);
                }

                for layer_idx in 0..num_layers {
                    if hetero_type_index(layer_idx, hcfg) != type_idx { continue; }
                    let abs_off = hetero_layer_offset(layer_idx, hcfg) + rel_off;
                    let copy_size = ref_size.min(blob.len().saturating_sub(abs_off));
                    if copy_size == 0 || abs_off >= blob.len() { continue; }

                    let layer_ptr = if layer_idx == ref_layer {
                        ref_ptr
                    } else {
                        let layer_cn = format!("L{}.{}", layer_idx, suffix);
                        weight_ptrs.get(&layer_cn)
                            .map(|&p| if !p.is_null() { p } else { ref_ptr })
                            .unwrap_or(ref_ptr)
                    };
                    let src = unsafe { std::slice::from_raw_parts(layer_ptr, copy_size) };
                    blob[abs_off..abs_off + copy_size].copy_from_slice(src);

                    // Post-copy: pre-shift norm weights (+1.0 for Gemma residual convention)
                    if is_gemma_norm_hetero && copy_size >= 4 {
                        let f32_dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                blob[abs_off..].as_mut_ptr() as *mut f32,
                                copy_size / 4,
                            )
                        };
                        for v in f32_dst.iter_mut() {
                            *v += 1.0;
                        }
                    }
                }
                packed_count += 1;
                continue;
            }
        }

        // Standard weight: direct lookup by canonical name.
        let ptr = match weight_ptrs.get(canonical_name) {
            Some(&p) if !p.is_null() => p,
            _ => {
                missing_count += 1;
                continue;
            }
        };
        let size = *weight_sizes.get(canonical_name).unwrap_or(&0);
        if size == 0 {
            continue;
        }

        // For per-layer weights with layer_loop_config, replicate across all layers.
        let is_per_layer = layer_config.is_some()
            && canonical_name.starts_with("L0.")
            && !canonical_name.contains("embed")
            // SPEC/39: global weights (final_norm, output head) don't start with "L0."
            // so the starts_with check already excludes them; "embed" guard is for safety.
            && !canonical_name.contains("final_norm");

        if is_per_layer {
            let cfg = layer_config.unwrap();
            let stride = cfg.weight_stride;
            let rel_off = *offset;

            for layer_idx in 0..num_layers {
                let abs_off = layer_idx * stride + rel_off;
                let copy_size = size.min(blob.len().saturating_sub(abs_off));
                if copy_size == 0 || abs_off >= blob.len() {
                    continue;
                }

                if layer_idx == 0 {
                    let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
                    blob[abs_off..abs_off + copy_size].copy_from_slice(src);
                } else {
                    let layer_cn = canonical_name.replacen("L0.", &format!("L{}.", layer_idx), 1);
                    if let Some(&layer_ptr) = weight_ptrs.get(&layer_cn) {
                        if !layer_ptr.is_null() {
                            let layer_size = *weight_sizes.get(&layer_cn).unwrap_or(&size);
                            let layer_copy = layer_size.min(copy_size);
                            let src = unsafe { std::slice::from_raw_parts(layer_ptr, layer_copy) };
                            blob[abs_off..abs_off + layer_copy].copy_from_slice(src);
                        }
                    } else {
                        let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
                        blob[abs_off..abs_off + copy_size].copy_from_slice(src);
                    }
                }
            }
            packed_count += 1;
        } else {
            // Global weight: post-layer globals need offset adjustment for layer replication.
            let blob_off = if let Some(hcfg) = hetero_config {
                let templates_blob = hcfg.sliding_small_stride + hcfg.full_small_stride
                    + hcfg.sliding_large_stride + hcfg.full_large_stride;
                let small_segs = hcfg.large_ffn_start_segment;
                let large_segs = hcfg.num_segments - small_segs;
                let total_layers_blob = small_segs * hcfg.small_segment_stride
                    + large_segs * hcfg.large_segment_stride;
                let graph_globals_start = hcfg.layer_blob_base_offset + templates_blob;
                if *offset >= graph_globals_start {
                    *offset + total_layers_blob.saturating_sub(templates_blob)
                } else {
                    *offset
                }
            } else if let Some(cfg) = layer_config {
                let globals_start = cfg.layer_blob_base_offset + cfg.weight_stride;
                if *offset >= globals_start {
                    (num_layers - 1) * cfg.weight_stride + *offset
                } else {
                    *offset
                }
            } else {
                *offset
            };
            let copy_size = size.min(blob.len().saturating_sub(blob_off));
            if copy_size == 0 || blob_off >= blob.len() {
                continue;
            }
            let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
            blob[blob_off..blob_off + copy_size].copy_from_slice(src);
            packed_count += 1;
        }
    }
    log::debug!(
        "pack_weights_from_graph: packed={packed_count}, missing={missing_count}, total_bytes={total_bytes}"
    );
    blob
}

// ============================================================================

/// Structured observation extracted from Mega-Kernel epilogue telemetry buffer.
#[derive(Debug, Clone, Copy)]
pub struct MegaKernelObservation {
    pub layer_idx: usize,
    pub entropy: f32,
    pub residual_delta: f32,
    pub cosine_similarity: f32,
    pub dead_neuron_count: u32,
    pub is_attention_sink: bool,
    pub per_channel_scale: f32,
    pub row_l1_norm: f32,
    pub row_max: f32,
}

impl MegaKernelObservation {
    pub fn from_buffer(layer_idx: usize, buffer: &[u8]) -> Self {
        use gllm_kernels::compiler::graph::telemetry_offsets;

        let read_f32 = |offset: usize| -> f32 {
            if offset + 4 <= buffer.len() {
                f32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0.0
            }
        };
        let read_u32 = |offset: usize| -> u32 {
            if offset + 4 <= buffer.len() {
                u32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0
            }
        };

        Self {
            layer_idx,
            entropy: read_f32(telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET),
            residual_delta: read_f32(telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            cosine_similarity: read_f32(telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            dead_neuron_count: read_u32(telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET),
            is_attention_sink: read_u32(telemetry_offsets::IS_ATTENTION_SINK_OFFSET) != 0,
            per_channel_scale: read_f32(telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET),
            row_l1_norm: read_f32(telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET),
            row_max: read_f32(telemetry_offsets::GEMM_ROW_MAX_OFFSET),
        }
    }

    pub fn dead_neuron_ratio(&self, hidden_size: usize) -> f32 {
        if hidden_size == 0 {
            return 0.0;
        }
        self.dead_neuron_count as f32 / hidden_size as f32
    }

    pub fn is_bypass_candidate(&self, delta_threshold: f32, cosine_threshold: f32) -> bool {
        self.residual_delta < delta_threshold && self.cosine_similarity > cosine_threshold
    }
}

// ============================================================================
// Diagnostic types for intermediate activation inspection
// ============================================================================

/// Diagnostic scratchpad data returned by `diagnostic_prefill_scratchpad`.
#[derive(Debug)]
pub struct DiagnosticScratchpad {
    pub data: Vec<u8>,
    pub logits_offset: usize,
    pub vocab_size: usize,
    pub prompt_len: usize,
    pub hidden_size: usize,
}

impl DiagnosticScratchpad {
    /// Read f32 values from scratchpad at given byte offset and count.
    pub fn read_f32_at(&self, byte_offset: usize, count: usize) -> Vec<f32> {
        let end = byte_offset + count * 4;
        if end > self.data.len() {
            return vec![];
        }
        let mut out = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data[byte_offset..].as_ptr() as *const f32,
                out.as_mut_ptr(),
                count,
            );
        }
        out
    }

    /// Read embedding output from scratchpad (at offset 0).
    pub fn embedding(&self) -> Vec<f32> {
        let count = self.prompt_len * self.hidden_size;
        self.read_f32_at(0, count)
    }

    /// Read logits for the last prompt token.
    pub fn last_token_logits(&self) -> Vec<f32> {
        let row_bytes = self.vocab_size * 4;
        let off = self.logits_offset + (self.prompt_len - 1) * row_bytes;
        self.read_f32_at(off, self.vocab_size)
    }
}
