impl ModelConfig {
    pub fn from_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let mut gguf_metadata_error: Option<ModelConfigError> = None;
        let mut safetensors_metadata_error: Option<ModelConfigError> = None;

        let mut config = None;

        // GGUF 格式：优先从张量+元数据推导配置
        if loader.weight_format() == WeightFormat::Gguf {
            match Self::from_gguf_loader(manifest, loader) {
                Ok(c) => config = Some(c),
                Err(err) => gguf_metadata_error = Some(err),
            }
        }

        // SafeTensors / ONNX 格式：统一张量驱动推导 (REQ-LOADER-022, REQ-LOADER-023)
        if config.is_none() && matches!(loader.weight_format(), WeightFormat::SafeTensors | WeightFormat::Onnx) {
            match Self::from_tensor_driven(manifest, loader) {
                Ok(c) => config = Some(c),
                Err(e) => {
                    safetensors_metadata_error = Some(e);
                }
            }
        }

        // Fallback: load from config.json file (HuggingFace standard format)
        if config.is_none() {
            if let Some(config_path) = loader.config_path() {
                if config_path.exists() {
                    let content = std::fs::read_to_string(config_path)?;
                    let value: Value = serde_json::from_str(&content)?;
                    let weight_dtype = loader
                        .detect_weight_dtype()
                        .ok()
                        .flatten();
                    config = Some(Self::from_value(manifest, &value, weight_dtype)?);
                }
            }
        }

        if let Some(mut cfg) = config {
            // Inject user-specified compute dtype from Loader
            cfg.compute_dtype = loader.compute_dtype();
            return Ok(cfg);
        }

        if let Some(err) = gguf_metadata_error {
            return Err(ModelConfigError::MissingConfigAndMetadata(err.to_string()));
        }
        if let Some(err) = safetensors_metadata_error {
            return Err(ModelConfigError::MissingConfigAndMetadata(err.to_string()));
        }

        Err(ModelConfigError::MissingConfig)
    }

    /// REQ-LOADER-022 / REQ-LOADER-023: 统一张量驱动配置推导 (SafeTensors + ONNX)
    ///
    /// 遵循 ARCH-TENSOR-DRIVEN 原则：
    /// - 优先级：张量形状 > 元数据
    /// - 从 embedding 张量推导 vocab_size（取较大维度）
    /// - 从 Q 投影张量推导 head_dim（q_out / num_heads）
    /// - 从权重张量 dtype 推导 dtype_size
    fn from_tensor_driven(
        manifest: &ModelManifest,
        loader: &mut Loader,
    ) -> ModelConfigResult<Self> {
        // 1. Extract hints (SafeTensors: from config.json; ONNX: default)
        let hints = match loader.weight_format() {
            WeightFormat::SafeTensors => Self::extract_head_dim_hint_from_config(loader),
            _ => TensorDeriveHints::default(),
        };

        // 2. Derive config from tensors via the active TensorProvider
        let derived = match loader.weight_format() {
            WeightFormat::SafeTensors => {
                let st = loader.safetensors_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access SafeTensors loader: {err}"
                    ))
                })?;
                derive_config_from_tensors_with_hints(st, hints)?
            }
            WeightFormat::Onnx => {
                let onnx = loader.onnx_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access ONNX loader: {err}"
                    ))
                })?;
                derive_config_from_tensors_with_hints(onnx, hints)?
            }
            _ => unreachable!("from_tensor_driven only called for SafeTensors/ONNX"),
        };

        // 3. Extract metadata value (SafeTensors: gllm.config; ONNX: onnx metadata)
        let base_value = match loader.weight_format() {
            WeightFormat::SafeTensors => loader
                .safetensors_gllm_config()
                .map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to load safetensors gllm.config metadata: {err}"
                    ))
                })?
                .cloned()
                .ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "tensor-driven derivation requires gllm.config metadata for non-tensor fields"
                            .to_string(),
                    )
                })?,
            WeightFormat::Onnx => {
                let onnx = loader.onnx_loader().map_err(|err| {
                    ModelConfigError::InvalidConfig(format!(
                        "failed to access ONNX loader: {err}"
                    ))
                })?;
                onnx_config_from_metadata(onnx)?.ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "tensor-driven ONNX derivation requires metadata attributes for non-tensor fields"
                            .to_string(),
                    )
                })?
            }
            _ => unreachable!(),
        };

        // 4. Build config
        let base = Self::from_value(manifest, &base_value, Some(derived.dtype))?;
        apply_tensor_derived(base, derived)
    }

    /// Extract head_dim hint from config.json's num_attention_heads field.
    /// Returns default hints if config.json is unavailable or missing the field.
    fn extract_head_dim_hint_from_config(loader: &Loader) -> TensorDeriveHints {
        let config_path = match loader.config_path() {
            Some(p) => p,
            None => return TensorDeriveHints::default(),
        };
        let data = match std::fs::read_to_string(config_path) {
            Ok(d) => d,
            Err(e) => {
                log::debug!("config.json read failed, using default tensor hints: {}", e);
                return TensorDeriveHints::default();
            }
        };
        let json: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                log::debug!("config.json parse failed, using default tensor hints: {}", e);
                return TensorDeriveHints::default();
            }
        };
        let num_heads = json
            .get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0);
        let hidden_size = json
            .get("hidden_size")
            .and_then(|v| v.as_u64())
            .filter(|&n| n > 0);
        match (num_heads, hidden_size) {
            (Some(nh), Some(hs)) if (hs as usize).is_multiple_of(nh as usize) => TensorDeriveHints {
                head_dim: Some(hs as usize / nh as usize),
            },
            _ => TensorDeriveHints::default(),
        }
    }

    fn from_gguf_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let (
            derived,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern_metadata,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
            final_logit_softcapping,
            feed_forward_lengths,
            mla_config,
            mtp_depth,
        ) = {
            let reader = loader.gguf_reader().map_err(|err| {
                ModelConfigError::InvalidConfig(format!("failed to load GGUF metadata: {err}"))
            })?;
            let arch = reader.architecture().map_err(|err| {
                ModelConfigError::InvalidConfig(format!(
                    "missing GGUF architecture metadata: {err}"
                ))
            })?;
            let metadata_hidden_size =
                optional_gguf_usize(reader.embedding_length(), "embedding_length")?;
            let metadata_num_attention_heads =
                optional_gguf_usize(reader.head_count(), "attention.head_count")?;
            let metadata_num_kv_heads =
                optional_gguf_usize(reader.head_count_kv(), "attention.head_count_kv")?;
            let metadata_head_dim = gguf_arch_usize(reader, arch, "attention.head_dim")
                .or_else(|| gguf_arch_usize(reader, arch, "attention.key_length_swa"))
                .or_else(|| {
                    reader
                        .rope_dimension_count()
                        .and_then(|value| usize::try_from(value).ok())
                })
                .or_else(|| {
                    let hidden = metadata_hidden_size?;
                    let heads = metadata_num_attention_heads?;
                    (heads > 0 && hidden % heads == 0).then_some(hidden / heads)
                });
            let mut derived = derive_config_from_tensors_with_hints(
                reader,
                TensorDeriveHints {
                    head_dim: metadata_head_dim,
                },
            )?;

            // Ω1: Tensors are truth. Capture the physical Q-proj output size derived from tensors.
            // If metadata overrides num_heads, we MUST adjust head_dim to maintain this physical invariant.
            // q_out = n_head * head_dim
            let tensor_q_out = derived.num_attention_heads * derived.head_dim;

            if let Some(value) = metadata_num_attention_heads {
                derived.num_attention_heads = value;
                if derived.num_attention_heads > 0 {
                    derived.head_dim = tensor_q_out / derived.num_attention_heads;
                }
            }
            if let Some(value) = metadata_num_kv_heads {
                derived.num_key_value_heads = value;
            }
            if let Some(value) = metadata_head_dim {
                // If metadata provides head_dim, check if it conflicts with tensor physics
                if value != derived.head_dim {
                    // We trust tensor physics (derived.head_dim calculated from q_out) over metadata here
                    // because CpuBackend validation relies on tensor shapes.
                    // However, if q_out was derived using this hint, they should match unless
                    // num_heads was also changed.
                }
            }

            if derived.num_attention_heads == 0
                || derived.num_key_value_heads == 0
                || derived.head_dim == 0
                || derived.num_attention_heads % derived.num_key_value_heads != 0
            {
                return Err(ModelConfigError::InvalidConfig(
                    "invalid GGUF attention topology after tensor derivation".to_string(),
                ));
            }

            let intermediate_size =
                optional_gguf_usize(reader.feed_forward_length(), "feed_forward_length")?;
            // Per-layer FFN intermediate sizes (Gemma 4 E2B: [6144×15, 12288×20]).
            // When present, indicates heterogeneous layer structure.
            let feed_forward_lengths = gguf_arch_array_usize(reader, arch, "feed_forward_length")
                .filter(|arr| arr.len() > 1 && arr.iter().any(|&v| v != arr[0]));
            let num_experts = optional_gguf_usize(reader.num_experts(), "num_experts")?;
            if matches!(num_experts, Some(0)) {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: num_experts".to_string(),
                ));
            }
            let num_experts_per_tok = optional_gguf_usize(reader.num_experts_per_tok(), "expert_used_count")?;
            let expert_intermediate_size = optional_gguf_usize(
                reader.expert_intermediate_size(),
                "expert_intermediate_size",
            )?;
            if matches!(expert_intermediate_size, Some(0)) {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: expert_intermediate_size".to_string(),
                ));
            }
            let max_position_embeddings =
                require_gguf_usize(reader.context_length(), "context_length")?;
            let rope_scaling = rope_scaling_from_gguf(reader, arch)?;
            let rope_theta = rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| require_gguf_f32(reader.rope_freq_base(), "rope.freq_base").ok())
                .ok_or_else(|| {
                    ModelConfigError::InvalidConfig(
                        "GGUF metadata missing: rope.freq_base (rope_theta)".to_string(),
                    )
                })?;

            // Ω1: GGUF 模型必须有 attention（已在 line 353 验证 num_attention_heads > 0）
            // 因此 rope_theta 必须 > 0
            if rope_theta == 0.0 {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata invalid: rope.freq_base (rope_theta) must be > 0".to_string(),
                ));
            }

            let rope_scale = gguf_arch_f32(reader, arch, "rope.scale")
                .or_else(|| {
                    rope_scaling
                        .as_ref()
                        .and_then(RopeScalingConfig::runtime_factor)
                })
                // Ω1: rope_scale = 1.0 is the industry standard (no scaling)
                .unwrap_or(1.0); // LEGAL: GGUF 元数据可选字段，缺失时使用行业标准默认值
            if !rope_scale.is_finite() || rope_scale <= 0.0 {
                return Err(ModelConfigError::InvalidConfig(
                    "GGUF metadata field invalid: rope.scale".to_string(),
                ));
            }

            let rope_interleaved =
                gguf_arch_bool(reader, arch, "rope.interleaved").unwrap_or(false); // LEGAL: GGUF 元数据可选字段，缺失时使用行业标准默认值
            let attention_dropout =
                gguf_arch_f32(reader, arch, "attention.dropout").filter(|v| v.is_finite());
            let hidden_act = gguf_arch_str(reader, arch, "feed_forward.activation")
                .or_else(|| gguf_arch_str(reader, arch, "hidden_act"))
                .map(HiddenAct::parse);
            let layer_norm_epsilon = gguf_arch_f32(reader, arch, "layer_norm_epsilon")
                .or_else(|| gguf_arch_f32(reader, arch, "layer_norm_rms_epsilon"))
                .or_else(|| gguf_arch_f32(reader, arch, "attention.layer_norm_rms_epsilon"))
                .filter(|v| v.is_finite() && *v > 0.0);

            // ── Gemma 4 / dual-attention family metadata ───────────────────
            // Ω1: load from GGUF when present. Rope/partial/global-head/PLE keys
            // are sensitive (affect numerics) so 0/absent stays as `None` at
            // this layer — `ModelGeometry::from_config` keeps the "0 means not
            // enabled" contract explicit.
            //
            // Community key conventions (as of 2026 Q1, llama.cpp/unsloth):
            //   {arch}.attention.sliding_window
            //   {arch}.attention.num_kv_shared_layers
            //   {arch}.attention.global_head_dim
            //   {arch}.attention.pattern                 (ARRAY[U8])
            //   {arch}.rope.global.freq_base             (Gemma 4 dual-RoPE global θ)
            //   {arch}.rope.partial_ratio                (p-RoPE fraction, 0..=1)
            //   {arch}.embedding.per_layer_input         (PLE injection width)
            let global_rope_theta = gguf_arch_f32(reader, arch, "rope.global.freq_base")
                .or_else(|| gguf_arch_f32(reader, arch, "global_rope_theta"))
                .filter(|v| v.is_finite() && *v > 0.0);
            let rope_partial_ratio = gguf_arch_f32(reader, arch, "rope.partial_ratio")
                .or_else(|| gguf_arch_f32(reader, arch, "rope.global.partial_ratio"))
                .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0);
            let attention_pattern_metadata = gguf_arch_array_u8(reader, arch, "attention.pattern")
                .or_else(|| gguf_arch_array_u8(reader, arch, "attention_pattern"))
                .or_else(|| {
                    // Gemma 4 GGUF (unsloth): bool array → convert to u8
                    gguf_arch_array_bool(reader, arch, "attention.sliding_window_pattern")
                        .map(|bools| bools.into_iter().map(|b| if b { 0 } else { 1 }).collect())
                });
            let sliding_window = gguf_arch_usize(reader, arch, "attention.sliding_window")
                .or_else(|| gguf_arch_usize(reader, arch, "sliding_window"));
            let num_kv_shared_layers = {
                // TEMP: disable shared_kv for now — JIT SharedKvRef lowering
                // needs the graph-layer GprCondAction to match the reduced KV
                // cache allocation.  Re-enable after DualRoPE is validated.
                let _ = gguf_arch_usize(reader, arch, "attention.shared_kv_layers");
                None
            };
            let global_head_dim = gguf_arch_usize(reader, arch, "attention.global_head_dim")
                .or_else(|| gguf_arch_usize(reader, arch, "global_head_dim"))
                .or_else(|| gguf_arch_usize(reader, arch, "attention.key_length"))
                .filter(|&v| v > 0);
            let hidden_size_per_layer_input =
                gguf_arch_usize(reader, arch, "embedding.per_layer_input")
                    .or_else(|| gguf_arch_usize(reader, arch, "hidden_size_per_layer_input"))
                    .or_else(|| gguf_arch_usize(reader, arch, "embedding_length_per_layer_input"));
            let final_logit_softcapping = gguf_arch_f32(reader, arch, "final_logit_softcapping")
                .filter(|v| v.is_finite() && *v > 0.0);

            (
                derived,
                intermediate_size,
                num_experts,
                num_experts_per_tok,
                expert_intermediate_size,
                max_position_embeddings,
                rope_theta,
                rope_scale,
                rope_interleaved,
                rope_scaling,
                attention_dropout,
                hidden_act,
                layer_norm_epsilon,
                reader.bos_token_id(),
                reader.eos_token_id(),
                reader
                    .get_metadata_u64("tokenizer.ggml.padding_token_id")
                    .and_then(|v| u32::try_from(v).ok()),
                global_rope_theta,
                rope_partial_ratio,
                attention_pattern_metadata,
                sliding_window,
                num_kv_shared_layers,
                global_head_dim,
                hidden_size_per_layer_input,
                final_logit_softcapping,
                feed_forward_lengths,
                reader.kv_lora_rank().map(|d_c| {
                    let d_rope = reader.qk_rope_head_dim().unwrap_or(64) as usize;
                    MlaConfig { d_c: d_c as usize, d_rope, unabsorbed_threshold: 4096 }
                }),
                optional_gguf_usize(reader.mtp_depth(), "mtp_depth")?,
            )
        };

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings); // LEGAL: manifest 可选字段，缺失时使用推导值
        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }
        let rope_theta = manifest.rope_base_override.unwrap_or(rope_theta); // LEGAL: manifest 可选字段，缺失时使用推导值

        // ── Gemma 4 dual-RoPE correction ──────────────────────────────────
        // Unsloth/llama.cpp GGUF only stores rope_freq_base=1000000 (the *global*
        // theta). For Gemma 4 the *sliding* theta is 10000. When the GGUF lacks
        // explicit `rope.global.freq_base` but we detect the model as dual-RoPE
        // (attention_pattern exists + rope_freq_base >= 100000), reinterpret:
        //   rope_theta ← 10000  (sliding, used by 29/35 layers)
        //   global_rope_theta ← original rope_freq_base (1M)
        //   rope_partial_ratio ← 0.25 (p-RoPE for global layers)
        let (rope_theta, global_rope_theta, rope_partial_ratio): (f32, Option<f32>, Option<f32>) = {
            // Detect dual-RoPE models (Gemma 4): GGUF stores only the global theta
            // (1M) as rope_freq_base without per-layer metadata. We infer dual-RoPE
            // when: (a) no explicit global_rope_theta, (b) freq_base >= 100K,
            // (c) Gemma 4 family signals present (sliding_window or PLE).
            let has_gemma4_signal = hidden_size_per_layer_input
                .map(|v| v > 0)
                .unwrap_or(false)
                || sliding_window.map(|v| v > 0).unwrap_or(false);
            let is_dual_rope_candidate = global_rope_theta.is_none()
                && rope_theta >= 100_000.0
                && has_gemma4_signal;
            if is_dual_rope_candidate {
                // rope_theta becomes sliding (10K), global_rope_theta gets original (1M).
                // rope_partial_ratio stays unchanged (sliding partial = 1.0 default).
                (
                    10_000.0_f32,                                   // sliding theta
                    Some(rope_theta),                               // global theta = original freq_base
                    rope_partial_ratio,                             // sliding partial (None→1.0 default)
                )
            } else {
                (rope_theta, global_rope_theta, rope_partial_ratio)
            }
        };

        let base_derived = derived.clone();

        // Gemma 4 attention_pattern fallback: GGUF 未提供时，按 SPEC 默认模式
        // (每 6 层第 6 层 global) 派生。仅当 PLE/global-rope 等信号表明确为
        // Gemma-4 家族时启用；否则保持 None 以免污染其他 arch。
        let enable_gemma4_pattern_default = hidden_size_per_layer_input
            .map(|v| v > 0)
            .unwrap_or(false)
            || global_rope_theta.is_some()
            || sliding_window.map(|v| v > 0).unwrap_or(false);
        let attention_pattern = attention_pattern_metadata.or_else(|| {
            if enable_gemma4_pattern_default && base_derived.num_hidden_layers > 0 {
                Some(derive_default_attention_pattern(
                    base_derived.num_hidden_layers,
                ))
            } else {
                None
            }
        });

        let base = Self {
            hidden_size: base_derived.hidden_size,
            num_attention_heads: base_derived.num_attention_heads,
            num_key_value_heads: base_derived.num_key_value_heads,
            num_hidden_layers: base_derived.num_hidden_layers,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            vocab_size: base_derived.vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            kv_cache_block_size: base_derived.head_dim.max(base_derived.num_key_value_heads),
            head_dim: base_derived.head_dim,
            dtype: base_derived.dtype,
            compute_dtype: None,
            use_cache: None,
            tie_word_embeddings: None,
            attention_dropout,
            hidden_act,
            layer_norm_epsilon,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            tensor_map: HashMap::new(),
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
            mtp_depth,
            mla_config,
            vision_config: None,
            audio_config: None,
            multimodal_token_ids: None,
            final_logit_softcapping,
            feed_forward_lengths,
            use_double_wide_mlp: None,
            add_special_tokens: None, // GGUF has no add_special_tokens concept, default true downstream
        };
        apply_tensor_derived(base, derived)
    }

    pub fn from_value(
        manifest: &ModelManifest,
        value: &Value,
        weight_dtype: Option<DType>,
    ) -> ModelConfigResult<Self> {
        // ARCH-MULTIMODAL-NESTED-CONFIG: Multi-modal 模型(Gemma 4 / SigLIP+LLaMA 等)
        // 把文本骨干字段嵌套在 `text_config` 子对象内,顶层只有
        // `architectures/audio_config/text_config/vision_config`。所有 require/find
        // 都先查顶层,再查 text_config 子对象。
        let hidden_size = require_usize(value, &[
            "hidden_size", "n_embd", "d_model",
            "text_config.hidden_size", "text_config.n_embd", "text_config.d_model",
        ])?;
        let num_attention_heads = require_usize(value, &[
            "num_attention_heads", "n_head", "num_heads",
            "text_config.num_attention_heads", "text_config.n_head", "text_config.num_heads",
        ])?;
        let num_key_value_heads = find_usize(
            value,
            &[
                "num_key_value_heads",
                "num_kv_heads",
                "n_kv_head",
                "attention.num_key_value_heads",
                "text_config.num_key_value_heads",
                "text_config.num_kv_heads",
            ],
        )
        .unwrap_or_else(|| {
            log::debug!("num_key_value_heads not found: defaulting to num_attention_heads = {}", num_attention_heads);
            num_attention_heads // LEGAL: num_key_value_heads 默认等于 num_attention_heads（非 GQA 模型）
        });
        let num_hidden_layers = require_usize(value, &[
            "num_hidden_layers", "n_layer", "num_layers",
            "text_config.num_hidden_layers", "text_config.n_layer", "text_config.num_layers",
        ])?;
        let intermediate_size = find_usize(
            value,
            &[
                "intermediate_size", "n_inner", "ffn_inter_dim", "d_ff",
                "text_config.intermediate_size", "text_config.n_inner",
            ],
        );
        let num_experts = find_usize(value, &[
            "num_experts", "moe.num_experts", "num_local_experts", "n_routed_experts",
            "text_config.num_experts", "text_config.num_local_experts",
        ]);
        let num_experts_per_tok = find_usize(value, &[
            "num_experts_per_tok", "num_selected_experts", "num_experts_per_token",
            "moe.num_experts_per_tok",
            "text_config.num_experts_per_tok",
        ]);
        let expert_intermediate_size = find_usize(
            value,
            &[
                "expert_intermediate_size",
                "moe.expert_intermediate_size",
                "moe_intermediate_size",
                "text_config.expert_intermediate_size",
                "text_config.moe_intermediate_size",
            ],
        );
        if matches!(num_experts, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "num_experts must be > 0 when provided".to_string(),
            ));
        }
        if matches!(expert_intermediate_size, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "expert_intermediate_size must be > 0 when provided".to_string(),
            ));
        }
        let vocab_size = require_usize(value, &["vocab_size", "text_config.vocab_size"])?;
        let rope_scaling = rope_scaling_from_metadata_json(value)?;

        let max_position_embeddings = find_usize(
            value,
            &[
                "max_position_embeddings",
                "max_seq_len",
                "max_sequence_length",
                "seq_length",
                "n_positions",
                "rope_scaling.original_max_position_embeddings",
                "text_config.max_position_embeddings",
                "text_config.rope_scaling.original_max_position_embeddings",
            ]
        )
        .unwrap_or_else(|| {
            log::debug!("max_position_embeddings not found: defaulting to 0 prior to manifest override");
            0 // LEGAL: 默认 0，后续由 manifest.override 或错误检查处理
        });

        let max_position_embeddings = manifest
            .max_context_override
            .unwrap_or(max_position_embeddings); // LEGAL: manifest 可选字段，缺失时使用推导值

        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }

        // Ω1: rope_theta 从模型配置或 manifest 中读取。
        // Encoder 模型（BERT/XLM-R）使用绝对位置编码，不含 rope_theta → 默认 0.0（表示无 RoPE）。
        // 下游 executor 通过 PositionEncoding::None 正确处理 rope_theta == 0.0 的 encoder 模型。
        let rope_theta = if let Some(override_value) = manifest.rope_base_override {
            override_value
        } else {
            rope_scaling
                .as_ref()
                .and_then(|cfg| cfg.base)
                .or_else(|| find_f32(value, &["rope_theta", "rope_base", "rope_base_value", "text_config.rope_theta"]))
                // Gemma 4 nested: text_config.rope_parameters.sliding_attention.rope_theta
                .or_else(|| find_f32(value, &["text_config.rope_parameters.sliding_attention.rope_theta"]))
                .or_else(|| find_f32(value, &["rope_parameters.sliding_attention.rope_theta"]))
                .unwrap_or_else(|| {
                    log::debug!("rope_theta not found: defaulting to 0.0 (model uses absolute position embeddings)");
                    0.0 // LEGAL: encoder 模型（BERT/XLM-R）无 RoPE，0.0 表示不使用旋转位置编码
                })
        };

        // RoPE 缩放系数优先读取完整 rope_scaling 对象；缺失时保持无缩放 (1.0)。
        // Ω1: rope_scale = 1.0 is the industry standard (no rotation scaling)
        let rope_scale = find_f32(value, &["rope_scale", "rope_factor", "rope.scaling.factor"])
            .or_else(|| {
                rope_scaling
                    .as_ref()
                    .and_then(RopeScalingConfig::runtime_factor)
            })
            .unwrap_or_else(|| {
                log::debug!("rope_scale not found: defaulting to 1.0 (no scaling)");
                1.0 // LEGAL: rope_scale=1.0 是 RoPE 的行业标准默认值（无缩放）
            });
        if !rope_scale.is_finite() || rope_scale <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scale must be positive".to_string(),
            ));
        }

        let rope_interleaved = find_bool(
            value,
            &[
                "rope_interleaved",
                "rotary_interleaved",
                "interleaved_rotary",
                "rope_scaling.interleaved",
            ]
        )
        .unwrap_or_else(|| {
            log::debug!("rope_interleaved not found: defaulting to false");
            false // LEGAL: rope_interleaved=false 是 RoPE 的行业标准默认值（非交错）
        });

        // Ω1: head_dim 从元数据读取，或使用标准公式计算
        // 注意：Embedding 模型可能没有 num_attention_heads，此时 head_dim = 0
        let head_dim = if num_attention_heads > 0 {
            find_usize(value, &["attention.head_dim", "head_dim", "kv_channels", "text_config.head_dim", "text_config.attention.head_dim"])
                .unwrap_or_else(|| {
                    let derived = hidden_size / num_attention_heads;
                    log::debug!("head_dim not found: deriving from hidden_size/num_heads => {}", derived);
                    derived // LEGAL: head_dim 可由 hidden_size/num_heads 推导（标准公式）
                })
        } else {
            // Embedding 模型没有 attention heads
            0
        };

        if num_attention_heads > 0 && head_dim == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid head_dim for non-embedding model".to_string(),
            ));
        }

        // Ω1: rope_theta == 0.0 合法 — encoder 模型（BERT/XLM-R）有 attention 但不使用 RoPE。
        // 下游 executor 会根据 (ModelKind, rope_theta) 选择 PositionEncoding::None 或 Rope。

        let kv_cache_block_size = find_usize(
            value,
            &["kv_cache_block_size", "kv_block_size", "page_size"],
        )
        .unwrap_or_else(|| {
            let derived = head_dim.max(num_key_value_heads);
            log::debug!("kv_cache_block_size not found: deriving from head_dim.max(num_key_value_heads) => {}", derived);
            derived // LEGAL: kv_cache_block_size 可由 head_dim.max(num_kv_heads) 推导（标准公式）
        });
        if kv_cache_block_size == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid kv_cache_block_size".to_string(),
            ));
        }

        // Ω1: dtype 大小必须从实际权重中读取，不从外部配置推断。
        let weight_dtype = weight_dtype.ok_or_else(|| {
            ModelConfigError::InvalidConfig(
                "无法确定模型 dtype：权重中缺少可识别浮点 dtype".to_string(),
            )
        })?;

        // Derive dtype: prefer torch_dtype from config.json, fall back to detected weight dtype.
        let dtype = find_string(value, &["torch_dtype", "dtype"])
            .and_then(|s| match s.as_str() {
                "bfloat16" => Some(DType::BF16),
                "float16" => Some(DType::F16),
                "float32" => Some(DType::F32),
                "float64" => Some(DType::F32), // f64 降级到 f32
                other => {
                    log::warn!("Unknown torch_dtype: {}, using detected weight dtype", other);
                    None
                }
            })
            .unwrap_or(weight_dtype); // LEGAL: 配置缺失时使用从权重检测的 dtype


        let attention_dropout = find_f32(value, &["attention_dropout", "attention.dropout"])
            .filter(|v| v.is_finite() && *v >= 0.0);
        let layer_norm_epsilon = find_f32(
            value,
            &[
                "layer_norm_epsilon",
                "layer_norm_eps",
                "rms_norm_eps",
                "norm_epsilon",
                "text_config.rms_norm_eps",
                "text_config.layer_norm_epsilon",
            ],
        )
        .filter(|v| v.is_finite() && *v > 0.0);

        // ── Gemma 4 specific fields ──
        let global_rope_theta = find_f32(value, &["global_rope_theta"])
            .or_else(|| find_f32(value, &["text_config.rope_parameters.full_attention.rope_theta"]));
        let rope_partial_ratio = find_f32(value, &[
            "rope_partial_ratio", "global_rope_partial", "partial_rotary_factor",
            "text_config.rope_parameters.full_attention.partial_rotary_factor",
        ]);
        // Gemma 4: layer_types=["sliding_attention","full_attention",...]
        //           → attention_pattern=[0,0,0,0,1,...] (0=sliding,1=full)
        let attention_pattern = value.get("attention_pattern")
            .or_else(|| value.get("text_config").and_then(|tc| tc.get("attention_pattern")))
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u8)).collect())
            .or_else(|| {
                // Fallback: parse layer_types string array
                let lt = value.get("text_config").and_then(|tc| tc.get("layer_types"))
                    .and_then(|v| v.as_array())?;
                Some(lt.iter().filter_map(|v| {
                    let s = v.as_str()?;
                    Some(match s {
                        "full_attention" => 1u8,
                        _ => 0u8,
                    })
                }).collect())
            });
        let sliding_window = find_usize(value, &["sliding_window", "sliding_window_size",
            "text_config.sliding_window", "text_config.sliding_window_size"]);
        let num_kv_shared_layers = find_usize(value, &["num_kv_shared_layers",
            "text_config.num_kv_shared_layers"]);
        let global_head_dim = find_usize(value, &["global_head_dim",
            "text_config.global_head_dim"]);
        let hidden_size_per_layer_input = find_usize(value, &["hidden_size_per_layer_input",
            "text_config.hidden_size_per_layer_input"]);

        // ── MTP (Multi-Token Prediction) ──
        let mtp_depth = find_usize(value, &["num_nextn_predict_layers", "mtp_depth"]);

        // ── MLA (Multi-head Latent Attention) ──
        // DeepSeek V3/R1: kv_lora_rank = d_c, q_lora_rank exists
        // Kimi-K2: same DeepSeek MLA architecture
        let mla_config = find_usize(value, &["kv_lora_rank", "text_config.kv_lora_rank"])
            .map(|d_c| {
                let d_rope = find_usize(value, &[
                    "rope_dimension_count", "qk_rope_head_dim",
                    "text_config.rope_dimension_count", "text_config.qk_rope_head_dim",
                ]).unwrap_or(64);
                MlaConfig {
                    d_c,
                    d_rope,
                    unabsorbed_threshold: 4096,
                }
            });

        // ── Multimodal: Vision Encoder (SigLIP) ──
        let vision_config = value.get("vision_config").and_then(|vc| {
            let image_size = find_usize(vc, &["image_size"])?;
            let patch_size = find_usize(vc, &["patch_size"])?;
            let vis_hidden = find_usize(vc, &["hidden_size"])?;
            let num_layers = find_usize(vc, &["num_hidden_layers", "num_layers"])?;
            let num_heads = find_usize(vc, &["num_attention_heads", "num_heads"])?;
            let vis_intermediate = find_usize(vc, &["intermediate_size"])?;
            Some(crate::compat::vision_forward::VisionConfig {
                image_size,
                patch_size,
                hidden_size: vis_hidden,
                num_layers,
                num_heads,
                intermediate_size: vis_intermediate,
            })
        });

        // ── Multimodal: Audio Encoder (USM Conformer) ──
        // 仅当 `audio_config` 存在且至少声明了核心几何维度时构造 AudioConfig;
        // 其余字段 (conv_kernel_size / fft_size / hop_length / ...) 默认值对齐
        // Gemma 4 USM-v2 官方参数。
        let audio_config = value.get("audio_config").and_then(|ac| {
            let default = crate::compat::audio_forward::AudioConfig::default();
            let hidden_size = find_usize(ac, &["hidden_size"]).unwrap_or(default.hidden_size);
            let num_layers = find_usize(ac, &["num_hidden_layers", "num_layers"]).unwrap_or(default.num_layers);
            let num_heads = find_usize(ac, &["num_attention_heads", "num_heads"]).unwrap_or(default.num_heads);
            let intermediate_size = find_usize(ac, &["intermediate_size"]).unwrap_or(default.intermediate_size);
            let conv_kernel_size = find_usize(ac, &["conv_kernel_size", "depthwise_kernel_size"])
                .unwrap_or(default.conv_kernel_size);
            let sample_rate = find_usize(ac, &["sample_rate", "sampling_rate"]).unwrap_or(default.sample_rate);
            let num_mel_bins = find_usize(ac, &["num_mel_bins", "n_mels"]).unwrap_or(default.num_mel_bins);
            let fft_size = find_usize(ac, &["fft_size", "n_fft"]).unwrap_or(default.fft_size);
            let hop_length = find_usize(ac, &["hop_length"]).unwrap_or(default.hop_length);
            let win_length = find_usize(ac, &["win_length"]).unwrap_or(default.win_length);
            let cfg = crate::compat::audio_forward::AudioConfig {
                sample_rate,
                hidden_size,
                num_layers,
                num_heads,
                conv_kernel_size,
                intermediate_size,
                num_mel_bins,
                fft_size,
                hop_length,
                win_length,
                layer_norm_eps: default.layer_norm_eps,
                stride: default.stride,
            };
            cfg.validate().ok().map(|_| cfg)
        });

        // ── Multimodal special token IDs (T58) ──
        // 优先从 config 顶层字段读取（兼容 Gemma 4 `boi_token_id` /
        // `image_token_id` 两种命名），否则从 tokenizer special_tokens_map
        // 读取（由 loader 侧合并到 value 时生效）。T58 铁律：禁止在 Rust
        // 源码里硬编码 ID。仅当 `vision_config` 声明存在但顶层 token 字段
        // 缺失时，才回退到 Gemma-4 默认值作为兜底，避免模型声明了多模态
        // 能力却无法路由。
        let image_tok = find_u32(value, &["image_token_id", "boi_token_id"]);
        let audio_tok = find_u32(value, &["audio_token_id", "boa_token_id"]);
        let eoi_tok = find_u32(value, &["eoi_token_id", "image_end_token_id"]);
        let eoa_tok = find_u32(value, &["eoa_token_id", "audio_end_token_id"]);
        let multimodal_token_ids = match (image_tok, audio_tok) {
            (Some(img), Some(aud)) => Some(crate::compat::multimodal::MultimodalTokenIds {
                image_token_id: img,
                audio_token_id: aud,
                eoi_token_id: eoi_tok.unwrap_or(img + 2),
                eoa_token_id: eoa_tok.unwrap_or(aud + 2),
            }),
            _ => {
                if vision_config.is_some() || audio_config.is_some() {
                    Some(crate::compat::multimodal::MultimodalTokenIds::gemma4_defaults())
                } else {
                    None
                }
            }
        };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta,
            rope_scale,
            rope_interleaved,
            rope_scaling,
            kv_cache_block_size,
            head_dim,
            dtype,
            compute_dtype: None, // Default: use model native dtype; set by user at load time
            use_cache: find_bool(value, &["use_cache"]),
            tie_word_embeddings: find_bool(value, &["tie_word_embeddings"]),
            attention_dropout,
            hidden_act: find_string(value, &["hidden_act", "hidden_activation"]).map(|s| HiddenAct::parse(&s)),
            layer_norm_epsilon,
            bos_token_id: find_u32(value, &["bos_token_id"]),
            eos_token_id: find_u32(value, &["eos_token_id"]),
            pad_token_id: find_u32(value, &["pad_token_id"]),
            tensor_map: manifest.tensor_map.clone(),
            global_rope_theta,
            rope_partial_ratio,
            attention_pattern,
            sliding_window,
            num_kv_shared_layers,
            global_head_dim,
            hidden_size_per_layer_input,
            mtp_depth,
            mla_config,
            vision_config,
            audio_config,
            multimodal_token_ids,
            final_logit_softcapping: find_f32(value, &["final_logit_softcapping"]),
            feed_forward_lengths: None, // JSON config path doesn't expose per-layer FFN sizes
            use_double_wide_mlp: find_bool(value, &["use_double_wide_mlp"]),
            add_special_tokens: find_bool(value, &["add_bos_token", "add_special_tokens"]),
        })
    }

    /// Build MoEConfig from extracted metadata, if this is a MoE model.
    pub fn build_moe_config(&self, arch: &str) -> Option<crate::manifest::MoEConfig> {
        let num_experts = self.num_experts?;
        if num_experts <= 1 {
            return None;
        }
        let num_experts_per_tok = self.num_experts_per_tok.unwrap_or(2); // LEGAL: num_experts_per_tok=2 是 MoE 的行业标准默认值
        let router_type = crate::arch::resolve_moe_router(arch)
            .unwrap_or(crate::manifest::RouterType::Mixtral);
        Some(crate::manifest::MoEConfig {
            num_experts,
            num_experts_per_tok,
            router_type,
        })
    }
}
