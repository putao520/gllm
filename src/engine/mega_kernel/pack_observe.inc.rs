// ============================================================================
// Weight Blob Packing (graph.weight_layout() driven)
// ============================================================================

/// Apply +1.0 in-place to each element of raw bytes, preserving original dtype.
/// Used for Gemma q_norm/k_norm residual convention: gamma = 1.0 + weight.
/// @trace REQ-DTYPE-CHAIN-001 REQ-DTYPE-CHAIN-003
fn add_one_inplace(dst: &mut [u8], dtype: ::safetensors::Dtype) {
    match dtype {
        ::safetensors::Dtype::BF16 => {
            for chunk in dst.chunks_exact_mut(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = half::bf16::from_bits(bits).to_f32() + 1.0;
                chunk.copy_from_slice(&half::bf16::from_f32(f).to_bits().to_le_bytes());
            }
        }
        ::safetensors::Dtype::F16 => {
            for chunk in dst.chunks_exact_mut(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let f = half::f16::from_bits(bits).to_f32() + 1.0;
                chunk.copy_from_slice(&half::f16::from_f32(f).to_bits().to_le_bytes());
            }
        }
        ::safetensors::Dtype::F32 => {
            for chunk in dst.chunks_exact_mut(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                chunk.copy_from_slice(&(v + 1.0f32).to_le_bytes());
            }
        }
        _ => {
            log::warn!(
                "add_one_inplace: dtype {:?} not supported for +1.0 residual — gemma_norm_residual weights will be incorrect",
                dtype
            );
        }
    }
}

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
    for (canonical_name, offset) in named_offsets {
        let ext_name = name_map.resolve_external_to_string(canonical_name);
        if let Some(raw) = raw_floats.get(&ext_name) {
            // @trace REQ-DTYPE-CHAIN-001 REQ-DTYPE-CHAIN-003

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
                    // @trace REQ-DTYPE-CHAIN-001

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
                        // @trace REQ-DTYPE-CHAIN-001 REQ-DTYPE-CHAIN-003
                        // Only q_norm and k_norm use Gemma residual convention (gamma = 1.0 + weight).
                        let is_gemma_norm = gemma_norm_residual
                            && (canonical_name.ends_with(".q_norm")
                                || canonical_name.ends_with(".k_norm"));

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

                        let (src_data, src_dtype) = if let Some(lr) = raw_floats.get(&layer_ext) {
                            (lr.data.as_slice(), lr.dtype)
                        } else {
                            (layer_raw.data.as_slice(), layer_raw.dtype)
                        };
                        let copy_size = src_data.len().min(blob.len().saturating_sub(abs_off));
                        if copy_size == 0 || abs_off >= blob.len() { continue; }
                        blob[abs_off..abs_off + copy_size].copy_from_slice(&src_data[..copy_size]);
                        if is_gemma_norm {
                            add_one_inplace(&mut blob[abs_off..abs_off + copy_size], src_dtype);
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
                // @trace REQ-DTYPE-CHAIN-001 REQ-DTYPE-CHAIN-003
                let copy_size = raw.data.len().min(blob.len().saturating_sub(blob_off));
                if copy_size == 0 || blob_off >= blob.len() {
                    continue;
                }
                blob[blob_off..blob_off + copy_size].copy_from_slice(&raw.data[..copy_size]);
                if gemma_norm_residual && (canonical_name.ends_with(".q_norm") || canonical_name.ends_with(".k_norm")) {
                    add_one_inplace(&mut blob[blob_off..blob_off + copy_size], raw.dtype);
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
                            if layer_size != copy_size && layer_size > 0 && copy_size > 0 {
                                // Mixed quantization: layer uses a different format than the template.
                                // Detect Q4_0 → Q4_1 conversion: template is 20 bytes/block, layer is 18 bytes/block.
                                let template_bb = copy_size / (layer_size / 18);
                                if template_bb == 20 && layer_size % 18 == 0 && copy_size % 20 == 0 {
                                    // Re-quantize Q4_0 → Q4_1:
                                    // Q4_0 block: [scale_f16(2B)][nibbles(16B)] = 18B, value = (nibble-8)*scale
                                    // Q4_1 block: [d_f16(2B)][m_f16(2B)][nibbles(16B)] = 20B, value = nibble*d + m
                                    // Conversion: d = scale, m = -8 * scale (f16), nibbles = same
                                    let num_blocks = layer_size / 18;
                                    let src = unsafe { std::slice::from_raw_parts(layer_ptr, layer_size) };
                                    let dst = &mut blob[abs_off..abs_off + num_blocks * 20];
                                    for bi in 0..num_blocks {
                                        let s_off = bi * 18;
                                        let d_off = bi * 20;
                                        let scale_f16_bits = u16::from_le_bytes([src[s_off], src[s_off + 1]]);
                                        let scale_f32 = half::f16::from_bits(scale_f16_bits).to_f32();
                                        // d = scale
                                        dst[d_off] = src[s_off];
                                        dst[d_off + 1] = src[s_off + 1];
                                        // m = -8 * scale (as f16)
                                        let m_f16 = half::f16::from_f32(-8.0 * scale_f32);
                                        let m_bits = m_f16.to_bits();
                                        dst[d_off + 2] = m_bits as u8;
                                        dst[d_off + 3] = (m_bits >> 8) as u8;
                                        // nibbles = same
                                        dst[d_off + 4..d_off + 20].copy_from_slice(&src[s_off + 2..s_off + 18]);
                                    }
                                    continue; // skip the raw copy below
                                }
                                log::warn!("{} L{} size mismatch: layer_size={} copy_size={} — no requant handler",
                                    canonical_name, layer_idx, layer_size, copy_size);
                            }
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

/// Attention sink status for a position in the sequence.
///
/// ARCH-JIT-DATA-YIELDS: replaces `is_attention_sink: bool` with a semantic enum.
/// Sink tokens receive special treatment in KV cache eviction and quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionSinkStatus {
    /// Normal token — eligible for standard eviction/quantization.
    Normal,
    /// Sink token — protected from eviction, preserved in quantization.
    SinkToken,
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
    pub sink_status: AttentionSinkStatus,
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
            sink_status: if read_u32(telemetry_offsets::IS_ATTENTION_SINK_OFFSET) != 0 { AttentionSinkStatus::SinkToken } else { AttentionSinkStatus::Normal },
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
    /// Compute dtype — needed to distinguish BF16 vs F16 decoding (exponent bias differs).
    /// ARCH-JIT-DATA-YIELDS: dtype stored as DType, not just elem_bytes, so that
    /// BF16 (8-bit exponent) and F16 (5-bit exponent) bit patterns decode correctly.
    pub compute_dtype: gllm_kernels::types::DType,
}

impl DiagnosticScratchpad {
    /// Bytes per element for the compute dtype.
    #[inline]
    pub fn elem_bytes(&self) -> usize {
        self.compute_dtype.size_bytes()
    }

    /// Read `count` elements from scratchpad at `byte_offset`, converting from
    /// the scratchpad's native dtype to f32. ARCH-JIT-DATA-YIELDS: dtype-aware read.
    /// Panics on unsupported DType (NO-SILENT-FALLBACK).
    pub fn read_dtype_aware(&self, byte_offset: usize, count: usize) -> Vec<f32> {
        let elem_bytes = self.elem_bytes();
        let byte_end = byte_offset + count * elem_bytes;
        if byte_end > self.data.len() {
            panic!(
                "read_dtype_aware: byte_end {} > data len {}, offset={}, count={}, dtype={:?}",
                byte_end, self.data.len(), byte_offset, count, self.compute_dtype
            );
        }
        match self.compute_dtype {
            gllm_kernels::types::DType::F32 => {
                // F32: direct read
                self.read_f32_at(byte_offset, count)
            }
            gllm_kernels::types::DType::BF16 => {
                // BF16: read raw bytes and convert (8-bit exponent, bias=127)
                let src = &self.data[byte_offset..byte_end];
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                    result.push(half::bf16::from_bits(bits).to_f32());
                }
                result
            }
            gllm_kernels::types::DType::F16 => {
                // F16: read raw bytes and convert (5-bit exponent, bias=15)
                let src = &self.data[byte_offset..byte_end];
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([src[off], src[off + 1]]);
                    result.push(half::f16::from_bits(bits).to_f32());
                }
                result
            }
            _ => panic!(
                "DiagnosticScratchpad::read_dtype_aware: unsupported compute_dtype={:?}, only F32/BF16/F16 supported",
                self.compute_dtype
            ),
        }
    }

    /// Read a single f32 element from scratchpad at `byte_offset`.
    /// Panics on unsupported DType (NO-SILENT-FALLBACK).
    pub fn read_single_element(&self, byte_offset: usize) -> f32 {
        match self.compute_dtype {
            gllm_kernels::types::DType::F32 => {
                if byte_offset + 4 <= self.data.len() {
                    unsafe {
                        let ptr = self.data.as_ptr().add(byte_offset) as *const f32;
                        *ptr
                    }
                } else {
                    panic!(
                        "read_single_element: byte_offset {} out of bounds (len {}), dtype={:?}",
                        byte_offset, self.data.len(), self.compute_dtype
                    );
                }
            }
            gllm_kernels::types::DType::BF16 => {
                if byte_offset + 2 <= self.data.len() {
                    let b0 = self.data[byte_offset];
                    let b1 = self.data[byte_offset + 1];
                    let bits = u16::from_le_bytes([b0, b1]);
                    half::bf16::from_bits(bits).to_f32()
                } else {
                    panic!(
                        "read_single_element: byte_offset {} out of bounds (len {}), dtype={:?}",
                        byte_offset, self.data.len(), self.compute_dtype
                    );
                }
            }
            gllm_kernels::types::DType::F16 => {
                if byte_offset + 2 <= self.data.len() {
                    let b0 = self.data[byte_offset];
                    let b1 = self.data[byte_offset + 1];
                    let bits = u16::from_le_bytes([b0, b1]);
                    half::f16::from_bits(bits).to_f32()
                } else {
                    panic!(
                        "read_single_element: byte_offset {} out of bounds (len {}), dtype={:?}",
                        byte_offset, self.data.len(), self.compute_dtype
                    );
                }
            }
            _ => panic!(
                "DiagnosticScratchpad::read_single_element: unsupported compute_dtype={:?}",
                self.compute_dtype
            ),
        }
    }

    /// Read f32 values from scratchpad at given byte offset and count.
    pub fn read_f32_at(&self, byte_offset: usize, count: usize) -> Vec<f32> {
        let end = byte_offset + count * 4;
        if end > self.data.len() {
            panic!(
                "read_f32_at: byte_offset {} + count*4 = {} out of bounds (data len {})",
                byte_offset, end, self.data.len()
            );
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
    ///
    /// ARCH-JIT-DATA-YIELDS: Embedding data is stored in compute dtype.
    /// This method returns f32 values regardless of the underlying dtype.
    pub fn embedding(&self) -> Vec<f32> {
        let count = self.prompt_len * self.hidden_size;
        self.read_dtype_aware(0, count)
    }

    /// Read logits for the last prompt token.
    ///
    /// ARCH-JIT-DATA-YIELDS: Logits are stored in compute dtype.
    /// This method returns f32 values regardless of the underlying dtype.
    pub fn last_token_logits(&self) -> Vec<f32> {
        if self.prompt_len == 0 {
            log::warn!("last_token_logits: prompt_len=0 — no tokens processed, returning empty logits");
            return Vec::new();
        }
        let row_bytes = self.vocab_size * self.elem_bytes();
        let off = self.logits_offset + (self.prompt_len - 1) * row_bytes;
        self.read_dtype_aware(off, self.vocab_size)
    }
}
