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

        // Fallback: load from config.json file (HuggingFace standard format) — #14: uses cached config
        if config.is_none() {
            if let Some(value) = loader.config_json() {
                let weight_dtype = loader
                    .detect_weight_dtype()
                    .ok()
                    .flatten();
                config = Some(Self::from_value(manifest, value, weight_dtype)?);
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

    /// GGUF path: tensor-driven derivation + FieldDef registry assembly.
    ///
    /// BCE-040 Task 23: Replaced ~376 lines of hand-rolled GGUF field parsing with
    /// registry-driven resolution. The flow now converges with the JSON path on a
    /// shared `CanonicalConfig` → `build_model_config()` assembly.
    ///
    /// ARCH-TENSOR-DRIVEN: tensor shapes take priority over metadata. A tensor-derived
    /// pass runs first to capture physical invariants (q_out = n_head * head_dim), and
    /// its values override the registry-resolved metadata afterwards.
    fn from_gguf_loader(manifest: &ModelManifest, loader: &mut Loader) -> ModelConfigResult<Self> {
        let reader = loader.gguf_reader().map_err(|err| {
            ModelConfigError::InvalidConfig(format!("failed to load GGUF metadata: {err}"))
        })?;
        let arch = reader.architecture().map_err(|err| {
            ModelConfigError::InvalidConfig(format!(
                "missing GGUF architecture metadata: {err}"
            ))
        })?;

        // ── Step 1: tensor-derived (ARCH-TENSOR-DRIVEN, must preserve) ──
        // Build head_dim hint from metadata, then let tensors determine the physical
        // attention topology. Tensors are truth; metadata only refines num_heads/kv_heads.
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
        // If metadata overrides num_heads, we MUST adjust head_dim to maintain this physical
        // invariant. q_out = n_head * head_dim
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
        if derived.num_attention_heads == 0
            || derived.num_key_value_heads == 0
            || derived.head_dim == 0
            || derived.num_attention_heads % derived.num_key_value_heads != 0
        {
            return Err(ModelConfigError::InvalidConfig(
                "invalid GGUF attention topology after tensor derivation".to_string(),
            ));
        }

        // ── Step 2: GGUF registry — resolve all metadata fields to CanonicalConfig ──
        let mut canonical = apply_gguf_field_registry(reader, arch, FIELD_DEFS)?;

        // ── Step 3: inject tensor-derived values (override metadata; tensor is truth) ──
        canonical.hidden_size = Some(derived.hidden_size);
        canonical.num_attention_heads = Some(derived.num_attention_heads);
        canonical.num_key_value_heads = Some(derived.num_key_value_heads);
        canonical.num_hidden_layers = Some(derived.num_hidden_layers);
        canonical.vocab_size = Some(derived.vocab_size);
        canonical.head_dim = Some(derived.head_dim);
        canonical.intermediate_size = canonical.intermediate_size.or(derived.intermediate_size);

        // ── Step 4: GGUF-specific post-processing ──
        // 4a: Gemma 4 dual-RoPE correction (reinterprets freq_base=1M as global theta)
        apply_gguf_dual_rope_correction(&mut canonical);
        // 4b: attention_pattern default derivation (Gemma 4 "every 6th layer global")
        apply_gguf_attention_pattern_default(&mut canonical);

        // ── Step 5: validate GGUF-specific invariants (NO-SILENT-FALLBACK) ──
        validate_gguf_canonical(&canonical)?;

        // ── Step 6: post-process defaults + manifest overrides ──
        apply_post_process(&mut canonical)?;
        if let Some(override_value) = manifest.max_context_override {
            canonical.max_position_embeddings = Some(override_value);
        }
        if let Some(override_value) = manifest.rope_base_override {
            canonical.rope_theta = Some(override_value);
        }

        // ── Step 7: build ModelConfig (reuse JSON path's assembler) ──
        let max_position_embeddings = canonical.max_position_embeddings.unwrap_or(0);
        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }
        let rope_scale = canonical.rope_scale.unwrap_or(1.0);
        let cfg = Self::build_model_config(
            canonical,
            max_position_embeddings,
            rope_scale,
            Some(derived.dtype),
            manifest,
        )?;
        apply_tensor_derived(cfg, derived)
    }

    /// Parse ModelConfig from a JSON config value using the FieldDef registry.
    ///
    /// BCE-040: Replaced ~480 lines of per-field find_*/require_* calls with
    /// registry-driven parsing: normalize → apply_field_registry → post_process → validate → build.
    pub fn from_value(
        manifest: &ModelManifest,
        value: &Value,
        weight_dtype: Option<DType>,
    ) -> ModelConfigResult<Self> {
        // Step 1: Normalize — flatten text_config.* to root level
        let normalized = normalize_text_config(value);

        // Step 2: Apply registry — extract all Alias/Derived fields
        let mut canonical = apply_field_registry(&normalized, FIELD_DEFS)?;

        // Step 2b: Validate rope_scaling errors (registry's Derived parser can't propagate errors)
        // rope_scaling_from_metadata_json can fail on invalid configs (negative factor, etc.)
        // Call it again for validation; if registry already parsed it, this is idempotent.
        rope_scaling_from_metadata_json(&normalized)?;

        // Step 3: Post-process — compute derived fields
        apply_post_process(&mut canonical)?;

        // Step 4: Apply manifest overrides
        if let Some(override_value) = manifest.max_context_override {
            canonical.max_position_embeddings = Some(override_value);
        }
        if let Some(override_value) = manifest.rope_base_override {
            canonical.rope_theta = Some(override_value);
        }

        // Step 5: Validate required fields
        let max_position_embeddings = canonical.max_position_embeddings.unwrap_or(0);
        if max_position_embeddings == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "missing max_position_embeddings".to_string(),
            ));
        }
        if matches!(canonical.num_experts, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "num_experts must be > 0 when provided".to_string(),
            ));
        }
        if matches!(canonical.expert_intermediate_size, Some(0)) {
            return Err(ModelConfigError::InvalidConfig(
                "expert_intermediate_size must be > 0 when provided".to_string(),
            ));
        }
        let rope_scale = canonical.rope_scale.unwrap_or(1.0);
        if !rope_scale.is_finite() || rope_scale <= 0.0 {
            return Err(ModelConfigError::InvalidConfig(
                "rope_scale must be positive".to_string(),
            ));
        }

        // Step 6: Build ModelConfig from CanonicalConfig
        Self::build_model_config(canonical, max_position_embeddings, rope_scale, weight_dtype, manifest)
    }

    /// Assemble ModelConfig from a fully-resolved CanonicalConfig.
    ///
    /// Extracted from from_value() — this function only does type conversion
    /// and field assembly, no parsing or lookup.
    fn build_model_config(
        c: CanonicalConfig,
        max_position_embeddings: usize,
        rope_scale: f32,
        weight_dtype: Option<DType>,
        manifest: &ModelManifest,
    ) -> ModelConfigResult<ModelConfig> {
        let hidden_size = c.hidden_size.ok_or_else(|| {
            ModelConfigError::InvalidConfig("missing hidden_size".to_string())
        })?;
        let num_attention_heads = c.num_attention_heads.ok_or_else(|| {
            ModelConfigError::InvalidConfig("missing num_attention_heads".to_string())
        })?;
        let num_hidden_layers = c.num_hidden_layers.ok_or_else(|| {
            ModelConfigError::InvalidConfig("missing num_hidden_layers".to_string())
        })?;
        let vocab_size = c.vocab_size.ok_or_else(|| {
            ModelConfigError::InvalidConfig("missing vocab_size".to_string())
        })?;

        let num_key_value_heads = c.num_key_value_heads.unwrap_or_else(|| {
            log::debug!("num_key_value_heads defaulting to num_attention_heads = {}", num_attention_heads);
            num_attention_heads
        });

        let head_dim = c.head_dim.unwrap_or_else(|| {
            let derived = hidden_size / num_attention_heads;
            log::debug!("head_dim deriving from hidden_size/num_heads => {}", derived);
            derived
        });

        if num_attention_heads > 0 && head_dim == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid head_dim for non-embedding model".to_string(),
            ));
        }

        let kv_cache_block_size = c.kv_cache_block_size.unwrap_or_else(|| {
            let derived = head_dim.max(num_key_value_heads);
            log::debug!("kv_cache_block_size deriving from head_dim.max(num_kv_heads) => {}", derived);
            derived
        });
        if kv_cache_block_size == 0 {
            return Err(ModelConfigError::InvalidConfig(
                "invalid kv_cache_block_size".to_string(),
            ));
        }

        // dtype: prefer torch_dtype from config, fall back to detected weight dtype
        let weight_dtype = weight_dtype.ok_or_else(|| {
            ModelConfigError::InvalidConfig(
                "无法确定模型 dtype：权重中缺少可识别浮点 dtype".to_string(),
            )
        })?;
        let dtype = c.torch_dtype.as_deref().and_then(|s| match s {
            "bfloat16" => Some(DType::BF16),
            "float16" => Some(DType::F16),
            "float32" => Some(DType::F32),
            "float64" => Some(DType::F32), // f64 → f32
            other => {
                log::warn!("Unknown torch_dtype: {}, using detected weight dtype", other);
                None
            }
        }).unwrap_or(weight_dtype);

        // num_kv_shared_layers: TEMP disabled (JIT MHA lowering needs donor-layer KV remap)
        let _ = c.num_kv_shared_layers;
        let num_kv_shared_layers = None;

        Ok(ModelConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_hidden_layers,
            intermediate_size: c.intermediate_size,
            num_experts: c.num_experts,
            num_experts_per_tok: c.num_experts_per_tok,
            expert_intermediate_size: c.expert_intermediate_size,
            vocab_size,
            max_position_embeddings,
            rope_theta: c.rope_theta.unwrap_or(0.0),
            rope_scale,
            rope_interleaved: c.rope_interleaved.unwrap_or(false),
            rope_scaling: c.rope_scaling,
            kv_cache_block_size,
            head_dim,
            dtype,
            compute_dtype: None,
            use_cache: c.use_cache,
            tie_word_embeddings: c.tie_word_embeddings,
            attention_dropout: c.attention_dropout.filter(|v| v.is_finite() && *v >= 0.0),
            hidden_act: c.hidden_act.map(|s| HiddenAct::parse(&s)),
            layer_norm_epsilon: c.layer_norm_epsilon.filter(|v| v.is_finite() && *v > 0.0),
            bos_token_id: c.bos_token_id,
            eos_token_id: c.eos_token_id,
            pad_token_id: c.pad_token_id,
            tensor_map: manifest.tensor_map.clone(),
            global_rope_theta: c.global_rope_theta,
            rope_partial_ratio: c.rope_partial_ratio,
            attention_pattern: c.attention_pattern,
            sliding_window: c.sliding_window,
            num_kv_shared_layers,
            global_head_dim: c.global_head_dim,
            hidden_size_per_layer_input: c.hidden_size_per_layer_input,
            mtp_depth: c.mtp_depth,
            mla_config: c.mla_config,
            vision_config: c.vision_config,
            audio_config: c.audio_config,
            multimodal_token_ids: c.multimodal_token_ids,
            final_logit_softcapping: c.final_logit_softcapping,
            feed_forward_lengths: c.feed_forward_lengths,
            use_double_wide_mlp: c.use_double_wide_mlp,
            add_special_tokens: c.add_special_tokens,
            qk_norm: c.qk_norm,
            value_norm: c.value_norm,
            embedding_scale_factor: c.embedding_scale_factor,
            rope_partial_ratio_global: c.rope_partial_ratio_global,
            mla_use_unabsorbed: c.mla_use_unabsorbed,
        })
    }

    /// Build MoEConfig from extracted metadata, if this is a MoE model.
    ///
    /// Returns `Ok(None)` for non-MoE models (num_experts absent or <= 1).
    /// Returns `Err` if this is a MoE model but required fields are missing
    /// (NO-SILENT-FALLBACK: top-k experts must be explicit, never inferred).
    pub fn build_moe_config(
        &self,
        arch: &str,
    ) -> ModelConfigResult<Option<crate::manifest::MoEConfig>> {
        let num_experts = self.num_experts;
        if num_experts.is_none() || num_experts == Some(0) || num_experts == Some(1) {
            return Ok(None);
        }
        let num_experts = num_experts.expect("checked above");
        let num_experts_per_tok = self.num_experts_per_tok.ok_or_else(|| {
            ModelConfigError::InvalidConfig(format!(
                "MoE model metadata missing: num_experts_per_tok (top-k experts) \
                 is required when num_experts ({num_experts}) > 1; \
                 cannot infer a default — NO-SILENT-FALLBACK"
            ))
        })?;
        let router_type = crate::arch::resolve_moe_router(arch)
            .unwrap_or(crate::manifest::RouterType::Mixtral);
        Ok(Some(crate::manifest::MoEConfig {
            num_experts,
            num_experts_per_tok,
            router_type,
        }))
    }
}
