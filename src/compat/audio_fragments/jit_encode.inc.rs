// JIT Conformer Block — 单块 CompilerGraph builder
// ============================================================================

/// 构造单 Conformer block 的 `CompilerGraph`。
///
/// 输入 (inputs 顺序与 weights pack 对齐):
/// 0. input        [seq, hidden]  — 前一层 hidden state
/// 1. ff1_norm_w   [hidden]       — FF1 pre-norm LayerNorm 权重
/// 2. ff1_norm_b   [hidden]       — FF1 pre-norm LayerNorm bias
/// 3. ff1_in_w     [hidden, inter]  — FF1 expand GEMM
/// 4. ff1_out_w    [inter, hidden]  — FF1 project GEMM
/// 5. attn_norm_w  [hidden]
/// 6. attn_norm_b  [hidden]
/// 7. w_q          [hidden, hidden]
/// 8. w_k          [hidden, hidden]
/// 9. w_v          [hidden, hidden]
/// 10. w_o         [hidden, hidden]
/// 11. conv_norm_w [hidden]
/// 12. conv_norm_b [hidden]
/// 13. conv_pw1_w  [hidden, hidden]   — pointwise conv1 (1x1 线性)
/// 14. dw_w        [hidden, kernel]   — DepthwiseConv1D 权重
/// 15. conv_bn_w   [hidden]
/// 16. conv_bn_b   [hidden]
/// 17. conv_pw2_w  [hidden, hidden]
/// 18. ff2_norm_w  [hidden]
/// 19. ff2_norm_b  [hidden]
/// 20. ff2_in_w    [hidden, inter]
/// 21. ff2_out_w   [inter, hidden]
/// 22. final_norm_w[hidden]
/// 23. final_norm_b[hidden]
///
/// 输出: [seq, hidden]
fn build_conformer_block_graph(
    seq_len: usize,
    config: &AudioConfig,
) -> CompilerGraph {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim();
    let eps = config.layer_norm_eps;
    let kernel = config.conv_kernel_size;
    // 所有算子统一 f32 (CPU JIT 路径默认 f32)。
    let dt = DType::F32;
    let ft = DType::F32;
    let s_dim = SymDim::Concrete(seq_len);

    let mut g = CompilerGraph::new();

    // ── Tensors ──
    let input = g.add_tensor_concrete("input", &[seq_len, hidden], ft);
    let ff1_norm_w = g.add_tensor_concrete("ff1_norm_w", &[hidden], ft);
    let ff1_norm_b = g.add_tensor_concrete("ff1_norm_b", &[hidden], ft);
    let ff1_in_w = g.add_tensor_concrete("ff1_in_w", &[hidden, inter], dt);
    let ff1_out_w = g.add_tensor_concrete("ff1_out_w", &[inter, hidden], dt);
    let attn_norm_w = g.add_tensor_concrete("attn_norm_w", &[hidden], ft);
    let attn_norm_b = g.add_tensor_concrete("attn_norm_b", &[hidden], ft);
    let w_q = g.add_tensor_concrete("w_q", &[hidden, hidden], dt);
    let w_k = g.add_tensor_concrete("w_k", &[hidden, hidden], dt);
    let w_v = g.add_tensor_concrete("w_v", &[hidden, hidden], dt);
    let w_o = g.add_tensor_concrete("w_o", &[hidden, hidden], dt);
    let conv_norm_w = g.add_tensor_concrete("conv_norm_w", &[hidden], ft);
    let conv_norm_b = g.add_tensor_concrete("conv_norm_b", &[hidden], ft);
    let conv_pw1_w = g.add_tensor_concrete("conv_pw1_w", &[hidden, hidden], dt);
    let dw_w = g.add_tensor_concrete("dw_w", &[hidden, kernel], dt);
    let conv_bn_w = g.add_tensor_concrete("conv_bn_w", &[hidden], ft);
    let conv_bn_b = g.add_tensor_concrete("conv_bn_b", &[hidden], ft);
    let conv_pw2_w = g.add_tensor_concrete("conv_pw2_w", &[hidden, hidden], dt);
    let ff2_norm_w = g.add_tensor_concrete("ff2_norm_w", &[hidden], ft);
    let ff2_norm_b = g.add_tensor_concrete("ff2_norm_b", &[hidden], ft);
    let ff2_in_w = g.add_tensor_concrete("ff2_in_w", &[hidden, inter], dt);
    let ff2_out_w = g.add_tensor_concrete("ff2_out_w", &[inter, hidden], dt);
    let final_norm_w = g.add_tensor_concrete("final_norm_w", &[hidden], ft);
    let final_norm_b = g.add_tensor_concrete("final_norm_b", &[hidden], ft);

    g.inputs = vec![
        input,
        ff1_norm_w, ff1_norm_b, ff1_in_w, ff1_out_w,
        attn_norm_w, attn_norm_b, w_q, w_k, w_v, w_o,
        conv_norm_w, conv_norm_b, conv_pw1_w, dw_w, conv_bn_w, conv_bn_b, conv_pw2_w,
        ff2_norm_w, ff2_norm_b, ff2_in_w, ff2_out_w,
        final_norm_w, final_norm_b,
    ];

    // ── FF1 half-step ──
    // LayerNorm → Linear → SiLU → Linear → Residual
    // 注: gllm_kernels 的 LayerNorm OpKind 只带 eps 字段, bias 仍由
    // graph 层显式提供。当前 lower 把 [norm_w, norm_b] 合并进 norm pattern,
    // 已对齐 BERT encoder 用法(见 compiler/codegen/vm/lower.rs lower_layernorm)。
    let ff1_normed = g.add_tensor_concrete("ff1_normed", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![input, ff1_norm_w, ff1_norm_b],
        vec![ff1_normed],
        "ff1_layernorm",
    );
    let ff1_inter = g.add_tensor_concrete("ff1_inter", &[seq_len, inter], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: inter, k: hidden, dtype: dt, trans_b: false, },
        vec![ff1_normed, ff1_in_w],
        vec![ff1_inter],
        "ff1_gemm_in",
    );
    let ff1_act = g.add_tensor_concrete("ff1_act", &[seq_len, inter], ft);
    g.add_op(OpKind::Silu, vec![ff1_inter], vec![ff1_act], "ff1_silu");
    let ff1_proj = g.add_tensor_concrete("ff1_proj", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: inter, dtype: dt, trans_b: false, },
        vec![ff1_act, ff1_out_w],
        vec![ff1_proj],
        "ff1_gemm_out",
    );
    let after_ff1 = g.add_tensor_concrete("after_ff1", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Add,
        vec![input, ff1_proj],
        vec![after_ff1],
        "ff1_residual",
    );

    // ── Self-Attention (full, non-causal) ──
    let attn_normed = g.add_tensor_concrete("attn_normed", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![after_ff1, attn_norm_w, attn_norm_b],
        vec![attn_normed],
        "attn_layernorm",
    );
    let q = g.add_tensor_concrete("q", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![attn_normed, w_q],
        vec![q],
        "attn_q",
    );
    let k = g.add_tensor_concrete("k", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![attn_normed, w_k],
        vec![k],
        "attn_k",
    );
    let v = g.add_tensor_concrete("v", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![attn_normed, w_v],
        vec![v],
        "attn_v",
    );
    let attn_out = g.add_tensor_concrete("attn_out", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: s_dim.clone(),
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            causal: false,
            attention_sinks: false,
        },
        vec![q, k, v],
        vec![attn_out],
        "attn_mha",
    );
    let attn_proj = g.add_tensor_concrete("attn_proj", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![attn_out, w_o],
        vec![attn_proj],
        "attn_o",
    );
    let after_attn = g.add_tensor_concrete("after_attn", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Add,
        vec![after_ff1, attn_proj],
        vec![after_attn],
        "attn_residual",
    );

    // ── Convolution module ──
    // LayerNorm → PointwiseConv1 → SiLU (GLU proxy) → DepthwiseConv1D →
    // LayerNorm → SiLU → PointwiseConv2 → Residual
    let conv_normed = g.add_tensor_concrete("conv_normed", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![after_attn, conv_norm_w, conv_norm_b],
        vec![conv_normed],
        "conv_layernorm",
    );
    let conv_pw1_out = g.add_tensor_concrete("conv_pw1", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![conv_normed, conv_pw1_w],
        vec![conv_pw1_out],
        "conv_pw1_gemm",
    );
    let conv_glu = g.add_tensor_concrete("conv_glu", &[seq_len, hidden], ft);
    g.add_op(OpKind::Silu, vec![conv_pw1_out], vec![conv_glu], "conv_silu_gate");
    let conv_dw = g.add_tensor_concrete("conv_dw", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::DepthwiseConv1D {
            channels: hidden,
            kernel_size: kernel,
            causal: false,
        },
        vec![conv_glu, dw_w],
        vec![conv_dw],
        "conv_depthwise",
    );
    let conv_bn_out = g.add_tensor_concrete("conv_bn", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![conv_dw, conv_bn_w, conv_bn_b],
        vec![conv_bn_out],
        "conv_bn_layernorm",
    );
    let conv_act = g.add_tensor_concrete("conv_act", &[seq_len, hidden], ft);
    g.add_op(OpKind::Silu, vec![conv_bn_out], vec![conv_act], "conv_silu_post");
    let conv_pw2_out = g.add_tensor_concrete("conv_pw2", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: hidden, dtype: dt, trans_b: false, },
        vec![conv_act, conv_pw2_w],
        vec![conv_pw2_out],
        "conv_pw2_gemm",
    );
    let after_conv = g.add_tensor_concrete("after_conv", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Add,
        vec![after_attn, conv_pw2_out],
        vec![after_conv],
        "conv_residual",
    );

    // ── FF2 half-step ──
    let ff2_normed = g.add_tensor_concrete("ff2_normed", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![after_conv, ff2_norm_w, ff2_norm_b],
        vec![ff2_normed],
        "ff2_layernorm",
    );
    let ff2_inter = g.add_tensor_concrete("ff2_inter", &[seq_len, inter], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: inter, k: hidden, dtype: dt, trans_b: false, },
        vec![ff2_normed, ff2_in_w],
        vec![ff2_inter],
        "ff2_gemm_in",
    );
    let ff2_act = g.add_tensor_concrete("ff2_act", &[seq_len, inter], ft);
    g.add_op(OpKind::Silu, vec![ff2_inter], vec![ff2_act], "ff2_silu");
    let ff2_proj = g.add_tensor_concrete("ff2_proj", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Gemm{ m: s_dim.clone(), n: hidden, k: inter, dtype: dt, trans_b: false, },
        vec![ff2_act, ff2_out_w],
        vec![ff2_proj],
        "ff2_gemm_out",
    );
    let after_ff2 = g.add_tensor_concrete("after_ff2", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::Add,
        vec![after_conv, ff2_proj],
        vec![after_ff2],
        "ff2_residual",
    );

    // ── Block-end LayerNorm ──
    let out = g.add_tensor_concrete("output", &[seq_len, hidden], ft);
    g.add_op(
        OpKind::LayerNorm { eps },
        vec![after_ff2, final_norm_w, final_norm_b],
        vec![out],
        "block_final_layernorm",
    );
    g.outputs = vec![out];
    g
}

// ============================================================================
// 权重打包 — 按 g.inputs 顺序生成 contiguous f32 blob
// ============================================================================

/// 把 `AudioTensorLookup` 中某 layer 的权重按 `build_conformer_block_graph`
/// 的 g.inputs 顺序收集起来,返回 f32 扁平 blob。
///
/// Gemma 4 audio tower 命名约定:
/// - `audio_tower.encoder.layers.{i}.norm_ff1.{weight,bias}`
/// - `audio_tower.encoder.layers.{i}.ff1_module.linear_in.weight`
/// - `audio_tower.encoder.layers.{i}.ff1_module.linear_out.weight`
/// - `audio_tower.encoder.layers.{i}.norm_self_attn.{weight,bias}`
/// - `audio_tower.encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `audio_tower.encoder.layers.{i}.conv_module.norm.{weight,bias}`
/// - `audio_tower.encoder.layers.{i}.conv_module.pointwise_conv1.weight`
/// - `audio_tower.encoder.layers.{i}.conv_module.depthwise_conv.weight`
/// - `audio_tower.encoder.layers.{i}.conv_module.bn.{weight,bias}`
/// - `audio_tower.encoder.layers.{i}.conv_module.pointwise_conv2.weight`
/// - `audio_tower.encoder.layers.{i}.norm_ff2.{weight,bias}`
/// - `audio_tower.encoder.layers.{i}.ff2_module.linear_in.weight`
/// - `audio_tower.encoder.layers.{i}.ff2_module.linear_out.weight`
/// - `audio_tower.encoder.layers.{i}.norm_final.{weight,bias}`
fn pack_layer_weights(
    layer_idx: usize,
    weights: &dyn AudioTensorLookup,
) -> Result<Vec<f32>, BackendError> {
    let base = format!("audio_tower.encoder.layers.{layer_idx}");
    let names: [String; 23] = [
        format!("{base}.norm_ff1.weight"),
        format!("{base}.norm_ff1.bias"),
        format!("{base}.ff1_module.linear_in.weight"),
        format!("{base}.ff1_module.linear_out.weight"),
        format!("{base}.norm_self_attn.weight"),
        format!("{base}.norm_self_attn.bias"),
        format!("{base}.self_attn.q_proj.weight"),
        format!("{base}.self_attn.k_proj.weight"),
        format!("{base}.self_attn.v_proj.weight"),
        format!("{base}.self_attn.o_proj.weight"),
        format!("{base}.conv_module.norm.weight"),
        format!("{base}.conv_module.norm.bias"),
        format!("{base}.conv_module.pointwise_conv1.weight"),
        format!("{base}.conv_module.depthwise_conv.weight"),
        format!("{base}.conv_module.bn.weight"),
        format!("{base}.conv_module.bn.bias"),
        format!("{base}.conv_module.pointwise_conv2.weight"),
        format!("{base}.norm_ff2.weight"),
        format!("{base}.norm_ff2.bias"),
        format!("{base}.ff2_module.linear_in.weight"),
        format!("{base}.ff2_module.linear_out.weight"),
        format!("{base}.norm_final.weight"),
        format!("{base}.norm_final.bias"),
    ];
    let mut packed: Vec<f32> = Vec::new();
    for name in &names {
        let slice = weights.get_audio_tensor(name).ok_or_else(|| {
            BackendError::Other(format!(
                "AudioTensorLookup: 缺失权重 '{name}'; caller 必须为该 layer 注入所有 Conformer block 权重"
            ))
        })?;
        packed.extend_from_slice(slice);
    }
    Ok(packed)
}

// ============================================================================
// Mel → Conformer 输入投影 (JIT GEMM)
// ============================================================================

/// 构建 mel spectrogram → hidden_size 的单层 GEMM 投影图。
/// 输入 0: mel_frames [num_frames, num_mel_bins]
/// 输入 1: proj_w     [num_mel_bins, hidden_size]
fn build_mel_projection_graph(num_frames: usize, config: &AudioConfig) -> CompilerGraph {
    let dt = DType::F32;
    let mut g = CompilerGraph::new();
    let s_dim = SymDim::Concrete(num_frames);
    let mel = g.add_tensor_concrete("mel", &[num_frames, config.num_mel_bins], dt);
    let proj_w = g.add_tensor_concrete(
        "audio_tower.feature_projection.weight",
        &[config.num_mel_bins, config.hidden_size],
        dt,
    );
    g.inputs = vec![mel, proj_w];
    let out = g.add_tensor_concrete("hidden_0", &[num_frames, config.hidden_size], dt);
    g.add_op(
        OpKind::Gemm{
            m: s_dim,
            n: config.hidden_size,
            k: config.num_mel_bins,
            dtype: dt,
                trans_b: false,
            },
        vec![mel, proj_w],
        vec![out],
        "mel_projection",
    );
    g.outputs = vec![out];
    g
}

/// 构建最终 encoder LayerNorm 图: [num_frames, hidden] → [num_frames, hidden]。
fn build_final_norm_graph(num_frames: usize, config: &AudioConfig) -> CompilerGraph {
    let dt = DType::F32;
    let mut g = CompilerGraph::new();
    let input = g.add_tensor_concrete("input", &[num_frames, config.hidden_size], dt);
    let w = g.add_tensor_concrete(
        "audio_tower.encoder.final_norm.weight",
        &[config.hidden_size],
        dt,
    );
    let b = g.add_tensor_concrete(
        "audio_tower.encoder.final_norm.bias",
        &[config.hidden_size],
        dt,
    );
    g.inputs = vec![input, w, b];
    let out = g.add_tensor_concrete("output", &[num_frames, config.hidden_size], dt);
    g.add_op(
        OpKind::LayerNorm {
            eps: config.layer_norm_eps,
        },
        vec![input, w, b],
        vec![out],
        "encoder_final_layernorm",
    );
    g.outputs = vec![out];
    g
}

// ============================================================================
// audio_encode — 对外主入口 (JIT 全管线)
// ============================================================================

#[inline]
#[allow(dead_code)]
fn f32_as_u8(slice: &[f32]) -> &[u8] {
    // SAFETY: f32 和 u8 具有相同的内存对齐约束 (4:1 字节比例),
    // 读取为 &[u8] 是合法的 transmute。
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            std::mem::size_of_val(slice),
        )
    }
}

/// 把音频 PCM 编码为 Conformer hidden-state tokens。
///
/// Pipeline:
/// 1. Mel spectrogram 提取 (纯 Rust FFT + mel filterbank)
/// 2. Mel → hidden_size 投影 (JIT GEMM)
/// 3. Conformer blocks × num_layers (JIT 每层一次)
/// 4. Encoder final LayerNorm (JIT)
///
/// 返回: row-major `[num_frames, hidden_size]` f32。
pub fn audio_encode(
    raw_audio: &[f32],
    config: &AudioConfig,
    weights: &dyn AudioTensorLookup,
) -> Result<Vec<f32>, BackendError> {
    config.validate()?;

    // ── 1. Mel spectrogram ──
    let (mel_flat, num_frames) = mel_spectrogram(raw_audio, config)?;
    if num_frames == 0 {
        return Err(BackendError::Other(
            "audio_encode: mel_spectrogram 产生 0 帧".into(),
        ));
    }
    // stride 下采样 (直接跨帧,学习型 subsample conv 留给后续 task)
    let (mel_flat, num_frames) = if config.stride > 1 {
        downsample_mel(&mel_flat, num_frames, config.num_mel_bins, config.stride)
    } else {
        (mel_flat, num_frames)
    };

    let hidden = config.hidden_size;

    let mut compiler = InferenceCompiler::new();

    // ── 2. Mel projection GEMM ──
    let proj_graph = build_mel_projection_graph(num_frames, config);
    let proj_config = CompileConfig {
        max_seq_len: num_frames,
        debug_jit: false,
        hetero: None,
    };
    let proj_compiled = compiler
        .compile_mega_kernel_from_graph(proj_graph, &proj_config, None)
        .map_err(|e| BackendError::Other(format!("audio_encode: mel projection compile: {e}")))?
        .layer_code;

    let proj_w = weights
        .get_audio_tensor("audio_tower.feature_projection.weight")
        .ok_or_else(|| {
            BackendError::Other(
                "AudioTensorLookup: 缺失 'audio_tower.feature_projection.weight'".into(),
            )
        })?;
    let expected_proj = config.num_mel_bins * hidden;
    if proj_w.len() != expected_proj {
        return Err(BackendError::Other(format!(
            "audio_encode: 'audio_tower.feature_projection.weight' 长度 {} != {} (num_mel_bins × hidden)",
            proj_w.len(),
            expected_proj,
        )));
    }

    let mut hidden_buf = vec![0.0f32; num_frames * hidden];
    let mut scratch = vec![0u8; proj_compiled.scratchpad_bytes.max(65536)];
    unsafe {
        proj_compiled.execute_as_mega_kernel(
            mel_flat.as_ptr() as *const u8,
            proj_w.as_ptr() as *const u8,
            1,
            num_frames,
            hidden_buf.as_mut_ptr() as *mut u8,
            scratch.as_mut_ptr(),
        );
    }
    if let Some((i, &v)) = hidden_buf.iter().enumerate().find(|(_, &v)| !v.is_finite()) {
        return Err(BackendError::Other(format!(
            "audio_encode: NaN after mel projection at [{i}]: {v}"
        )));
    }

    // ── 3. Conformer blocks ──
    // 同一 seq_len 下所有 block 的 graph 结构等价,编译一次复用 num_layers 次

    let block_graph = build_conformer_block_graph(num_frames, config);
    let block_config = CompileConfig {
        max_seq_len: num_frames,
        debug_jit: false,
        hetero: None,
    };
    let block_compiled = compiler
        .compile_mega_kernel_from_graph(block_graph, &block_config, None)
        .map_err(|e| BackendError::Other(format!("audio_encode: conformer block compile: {e}")))?
        .layer_code;

    let mut scratch_block = vec![0u8; block_compiled.scratchpad_bytes.max(65536)];
    let mut out_buf = vec![0.0f32; num_frames * hidden];

    for layer_idx in 0..config.num_layers {
        let weights_packed = pack_layer_weights(layer_idx, weights)?;

        unsafe {
            block_compiled.execute_as_mega_kernel(
                hidden_buf.as_ptr() as *const u8,
                weights_packed.as_ptr() as *const u8,
                1,
                num_frames,
                out_buf.as_mut_ptr() as *mut u8,
                scratch_block.as_mut_ptr(),
            );
        }

        if let Some((i, &v)) = out_buf.iter().enumerate().find(|(_, &v)| !v.is_finite()) {
            return Err(BackendError::Other(format!(
                "audio_encode: NaN after conformer block layer {layer_idx} at [{i}]: {v}"
            )));
        }
        std::mem::swap(&mut hidden_buf, &mut out_buf);
    }

    // ── 4. Encoder final LayerNorm ──
    let final_graph = build_final_norm_graph(num_frames, config);
    let final_config = CompileConfig {
        max_seq_len: num_frames,
        debug_jit: false,
        hetero: None,
    };
    let final_compiled = compiler
        .compile_mega_kernel_from_graph(final_graph, &final_config, None)
        .map_err(|e| BackendError::Other(format!("audio_encode: final norm compile: {e}")))?
        .layer_code;

    let final_w = weights
        .get_audio_tensor("audio_tower.encoder.final_norm.weight")
        .ok_or_else(|| {
            BackendError::Other(
                "AudioTensorLookup: 缺失 'audio_tower.encoder.final_norm.weight'".into(),
            )
        })?;
    let final_b = weights
        .get_audio_tensor("audio_tower.encoder.final_norm.bias")
        .ok_or_else(|| {
            BackendError::Other(
                "AudioTensorLookup: 缺失 'audio_tower.encoder.final_norm.bias'".into(),
            )
        })?;
    if final_w.len() != hidden || final_b.len() != hidden {
        return Err(BackendError::Other(format!(
            "audio_encode: encoder final norm weight/bias 长度不一致 (got {}/{}, expected {})",
            final_w.len(),
            final_b.len(),
            hidden,
        )));
    }
    let mut final_weights = Vec::with_capacity(2 * hidden);
    final_weights.extend_from_slice(final_w);
    final_weights.extend_from_slice(final_b);

    let mut scratch_final = vec![0u8; final_compiled.scratchpad_bytes.max(65536)];
    unsafe {
        final_compiled.execute_as_mega_kernel(
            hidden_buf.as_ptr() as *const u8,
            final_weights.as_ptr() as *const u8,
            1,
            num_frames,
            out_buf.as_mut_ptr() as *mut u8,
            scratch_final.as_mut_ptr(),
        );
    }
    Ok(out_buf)
}

/// 跨帧 stride 下采样: 每 `stride` 帧取 1 帧。
fn downsample_mel(
    mel: &[f32],
    num_frames: usize,
    num_mels: usize,
    stride: usize,
) -> (Vec<f32>, usize) {
    if stride <= 1 || num_frames == 0 {
        return (mel.to_vec(), num_frames);
    }
    let new_frames = num_frames.div_ceil(stride);
    let mut out = Vec::with_capacity(new_frames * num_mels);
    for i in 0..new_frames {
        let src = i * stride;
        let start = src * num_mels;
        let end = start + num_mels;
        out.extend_from_slice(&mel[start..end]);
    }
    (out, new_frames)
}

// ============================================================================
