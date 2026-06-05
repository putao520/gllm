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
    num_layers: usize,
) -> Vec<u8> {
    let mut blob = vec![0u8; total_bytes];
    let mut packed_count = 0usize;
    let mut missing_count = 0usize;
    if !raw_floats.is_empty() {
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
                } else {
                    eprintln!("[pack] {} -> {} NOT IN raw_floats", cn, ext);
                }
            }
        }
    }

    for (canonical_name, offset) in named_offsets {
        eprintln!("[PACK-DEBUG] cn='{}' offset={} is_layer={}", canonical_name, offset,
            canonical_name.starts_with("L0.") && layer_config.is_some());
        // Resolve canonical → external name for raw_floats lookup.
        let ext_name = name_map.resolve_external_to_string(canonical_name);
        {
            let rf = raw_floats.get(&ext_name);
            let wp = weight_ptrs.get(canonical_name);
            let ws = weight_sizes.get(canonical_name);
            eprintln!("[PACK-ALL] cn='{}' ext='{}' rf={} wp={} ws={:?}",
                canonical_name, ext_name, rf.is_some(), wp.is_some(), ws);
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
            let is_per_layer_rf = layer_config.is_some()
                && canonical_name.starts_with("L0.")
                && !canonical_name.contains("embed")
                && !canonical_name.contains("lm_head")
                && !canonical_name.contains("final_norm");

            if is_per_layer_rf {
                let cfg = layer_config.unwrap();
                // Count how many layers we successfully find in raw_floats
                let mut layer_hits = 0usize;
                for layer_idx in 0..num_layers {
                    if layer_idx == 0 {
                        layer_hits += 1;
                    } else {
                        let lcn = canonical_name.replacen("L0.", &format!("L{}.", layer_idx), 1);
                        let lext = name_map.resolve_external_to_string(&lcn);
                        if raw_floats.get(&lext).is_some() { layer_hits += 1; }
                    }
                }
                if canonical_name.contains("q_proj") {
                    eprintln!("[PACK-LAYER-RF] cn='{}' found {}/{} layers in raw_floats stride={}",
                        canonical_name, layer_hits, num_layers, cfg.weight_stride);
                }
                let stride = cfg.weight_stride;
                let rel_off = *offset;

                for layer_idx in 0..num_layers {
                    let abs_off = layer_idx * stride + rel_off;
                    let copy_size = f32_bytes.min(blob.len().saturating_sub(abs_off));
                    if copy_size == 0 || abs_off >= blob.len() {
                        continue;
                    }
                    // For layers > 0, look up per-layer raw_floats if available
                    if layer_idx == 0 {
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(
                                blob[abs_off..].as_mut_ptr() as *mut f32,
                                copy_size / 4,
                            )
                        };
                        match raw.dtype {
                            ::safetensors::Dtype::BF16 => {
                                let src = unsafe {
                                    std::slice::from_raw_parts(raw.data.as_ptr() as *const half::bf16, numel)
                                };
                                for (i, &v) in src.iter().enumerate() {
                                    if i >= dst.len() { break; }
                                    dst[i] = v.to_f32();
                                }
                            }
                            ::safetensors::Dtype::F16 => {
                                let src = unsafe {
                                    std::slice::from_raw_parts(raw.data.as_ptr() as *const half::f16, numel)
                                };
                                for (i, &v) in src.iter().enumerate() {
                                    if i >= dst.len() { break; }
                                    dst[i] = v.to_f32();
                                }
                            }
                            _ => {
                                let cs = raw.data.len().min(blob.len().saturating_sub(abs_off));
                                blob[abs_off..abs_off + cs].copy_from_slice(&raw.data[..cs]);
                            }
                        }
                    } else {
                        let layer_cn = canonical_name.replacen("L0.", &format!("L{}.", layer_idx), 1);
                        let layer_ext = name_map.resolve_external_to_string(&layer_cn);
                        if let Some(layer_raw) = raw_floats.get(&layer_ext) {
                            let ln = layer_raw.data.len() / elem_size;
                            let lb = ln * 4;
                            let lcs = lb.min(blob.len().saturating_sub(abs_off));
                            if lcs > 0 {
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        blob[abs_off..].as_mut_ptr() as *mut f32,
                                        lcs / 4,
                                    )
                                };
                                match layer_raw.dtype {
                                    ::safetensors::Dtype::BF16 => {
                                        let src = unsafe {
                                            std::slice::from_raw_parts(layer_raw.data.as_ptr() as *const half::bf16, ln)
                                        };
                                        for (i, &v) in src.iter().enumerate() {
                                            if i >= dst.len() { break; }
                                            dst[i] = v.to_f32();
                                        }
                                    }
                                    ::safetensors::Dtype::F16 => {
                                        let src = unsafe {
                                            std::slice::from_raw_parts(layer_raw.data.as_ptr() as *const half::f16, ln)
                                        };
                                        for (i, &v) in src.iter().enumerate() {
                                            if i >= dst.len() { break; }
                                            dst[i] = v.to_f32();
                                        }
                                    }
                                    _ => {
                                        let cs2 = layer_raw.data.len().min(blob.len().saturating_sub(abs_off));
                                        blob[abs_off..abs_off + cs2].copy_from_slice(&layer_raw.data[..cs2]);
                                    }
                                }
                            }
                        } else {
                            // Reuse layer 0 data from correct position (rel_off = graph offset)
                            if layer_idx < 2 && canonical_name.contains("q_proj") {
                                eprintln!("[PACK-FALLBACK] L{}.{} layer_ext='{}' not in raw_floats, reusing L0 data from blob[{}]",
                                    layer_idx, canonical_name, layer_ext, rel_off);
                            }
                            let src_off = rel_off;
                            let src_f32 = unsafe {
                                std::slice::from_raw_parts(blob[src_off..].as_ptr() as *const f32, f32_bytes.min(blob.len().saturating_sub(src_off)) / 4)
                            };
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(blob[abs_off..].as_mut_ptr() as *mut f32, f32_bytes.min(blob.len().saturating_sub(abs_off)) / 4)
                            };
                            let copy_len = src_f32.len().min(dst.len());
                            dst[..copy_len].copy_from_slice(&src_f32[..copy_len]);
                        }
                    }
                }
            } else {
                // Global weight: post-layer globals need offset adjustment for layer replication.
                // Pre-layer globals (embed, offset < base) stay at their graph offsets.
                let blob_off = if let Some(cfg) = layer_config {
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
                            dst_f32s[i] = v.to_f32();
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
                            dst_f32s[i] = v.to_f32();
                        }
                    }
                    _ => {
                        let copy_size = raw.data.len().min(blob.len().saturating_sub(blob_off));
                        blob[blob_off..blob_off + copy_size].copy_from_slice(&raw.data[..copy_size]);
                    }
                }
            }
            packed_count += 1;
            continue;
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
        // The offset in named_offsets is RELATIVE to layer start. We compute absolute
        // offset as: layer_blob_base_offset + layer_idx * weight_stride + relative_offset.
        let is_per_layer = layer_config.is_some()
            && canonical_name.starts_with("L0.")
            && !canonical_name.contains("embed")
            && !canonical_name.contains("lm_head")
            && !canonical_name.contains("final_norm");

        if is_per_layer && packed_count < 3 {
            eprintln!("[PACK-LAYER] cn='{}' rel_offset={} -> will replicate to {} layers, stride={}",
                canonical_name, offset, num_layers, layer_config.unwrap().weight_stride);
        }

        if is_per_layer {
            let cfg = layer_config.unwrap();
            let stride = cfg.weight_stride;
            let rel_off = *offset; // graph offset (includes base)

            // Pack layer 0 data (L0.xxx) for all N layers
            for layer_idx in 0..num_layers {
                let abs_off = layer_idx * stride + rel_off;
                let copy_size = size.min(blob.len().saturating_sub(abs_off));
                if copy_size == 0 || abs_off >= blob.len() {
                    continue;
                }

                // For layers > 0, look up their specific weight data (L1.xxx, L2.xxx, etc.)
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
                        // Layer weight not found: reuse layer 0 data
                        let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
                        blob[abs_off..abs_off + copy_size].copy_from_slice(src);
                    }
                }
            }
            packed_count += 1;
        } else {
            // Global weight: post-layer globals need offset adjustment for layer replication.
            // Pre-layer globals (embed, offset < base) stay at their graph offsets.
            let blob_off = if let Some(cfg) = layer_config {
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
    eprintln!("[PACK] blob size={} packed={}/{} named_offsets={}", blob.len(), packed_count, packed_count + missing_count, named_offsets.len());
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
